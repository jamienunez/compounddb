import mysql.connector
import pandas as pd
import numpy as np


class DatabaseManager():

    def __init__(self):
        '''Connect to compound DB and store handles'''

        # Connect to DB
        self.conn = mysql.connector.connect(
            host='localhost',
            user='root',
            passwd='password',
            database='compound_db'
        )
        self.cursor = self.conn.cursor(buffered=True)

        # Get permitted property categories
        self.cursor.execute('SELECT name FROM category')
        self.categories = [x[0] for x in self.cursor.fetchall()]

        return

    def get_property_categories(self):
        return self.categories[:]

    def get_cursor(self):
        return self.cursor

    def get_connector(self):
        return self.conn

    def run_query(self, query):
        # TODO: check query
        return pd.read_sql(query, con=self.conn)

    def create_cpd_ids(self, df, inplace=False, start_id=None):
        '''Add cpd_id column to df to reference during insertion'''

        # Throw error if there is no inchi_key col
        if 'inchi_key' not in df.columns:
            raise ValueError('inchi_key column required but missing.')

        # Throw error if any inchikeys are nan
        if df['inchi_key'].isnull().sum() > 0:
            m = 'Nan InChIKeys exist, remove before running function.'
            raise ValueError(m.format(i))

        # Work from copy if original shouldn't be edited
        if not inplace:
            df = df.copy()

        # Get ID to start from
        if start_id is None:
            next_id = self.get_max_id('compound') + 1

        # Add cpd_id column
        df['cpd_id'] = -1

        # Dictionary to track cpd_ids within this lib
        # Since these cpd_ids haven't been commited, they aren't
        # trackable through the DB yet.
        d = {}

        # Iterate through df, updating with appropriate ID
        for i, row in df.iterrows():

            val = row['inchi_key']

            # If this cpd is already in the DB, use that cpd_id
            if val in d:
                df.at[i, 'cpd_id'] = d[val]
            elif self.val_exists('compound', 'inchi_key', val):
                cpd_id = self.get_cpd_id(val)
                df.at[i, 'cpd_id'] = cpd_id
                d[val] = cpd_id
                print('Hit: {}'.format(cpd_id))
            else:  # Otherwise, use the next available id
                df.at[i, 'cpd_id'] = next_id
                d[val] = next_id
                next_id += 1

        return df

    def prep_input(self, filename, rename_dict=None, start_id=None, header=0,
                   sheet_name=0):
        '''Prepare df for insertion into DB'''

        # Load file
        df = pd.read_excel(filename, header=header, sheet_name=sheet_name)

        # Rename columns as needed to match those expected in DB
        if rename_dict is not None:
            df.rename(columns=rename_dict, inplace=True)

        # Remove entries with nan InChIKey. Avoids errors with next call
        df.dropna(subset=['inchi_key'], inplace=True)
        df.reset_index(inplace=True)

        # Create cpd_id col
        self.create_cpd_ids(df, inplace=True, start_id=start_id)

        return df

    def prep_string(self, s):
        '''Get string ready for DB query by adding quotes if needed'''
        if type(s) == str and (s[0] != '\'' or s[-1] != '\''):
            return '\'' + s + '\''
        return s

    def val_exists(self, table, col, val, e=None):
        '''Check if the given value exists in the given table+col'''

        # NaN values are not checked.
        if val is None:
            return False

        # Switch for string value
        if type(val) == str:
            return self.val_exists_string(table, col, val)

        # Assume float or integer otherwise
        return self.val_exists_within_error(table, col, val, e)

    def val_exists_string(self, table, col, val):
        '''Check if the given value exists in the given table+col
        Note: this value is treated exactly, with no error considered.
        This function is best for strings and integers.'''
        val = self.prep_string(val)
        query = 'SELECT * FROM %s WHERE %s = %s;' % (table, col, val)
        return bool(len(pd.read_sql(query, con=self.conn)))

    def val_exists_within_error(self, table, col, val, e):
        '''Check if the given value exists in the given table+col. The error
        provided should already be divded by desired level of accuracy
        (e.g. 5 / 100 for 5% error). Use e=0 for integers.'''
        if e is None or e == 0:
            query = 'SELECT * FROM %s WHERE %s = %s;' % (table, col, val)
        else:
            min_val = val * (1 - e)
            max_val = val * (1 + e)
            query = 'SELECT * FROM {} WHERE {} BETWEEN {} and {};'
            query = query.format(table, col, min_val, max_val)
        return bool(len(pd.read_sql(query, con=self.conn)))

    def get_cpd_id(self, inchi_key):
        '''Get cpd_id of given InChIKey'''
        inchi_key = self.prep_string(inchi_key)
        query = 'SELECT cpd_id FROM compound WHERE inchi_key = %s;' % inchi_key
        return pd.read_sql(query, con=self.conn).values[0][0]

    def get_num_entries(self, table):
        '''Get height of table'''
        self.cursor.execute('SELECT COUNT(*) FROM %s' % table)
        return self.cursor.fetchone()[0]

    def get_max_id(self, table):
        '''Get max unique identifier for this table.'''

        # Assign col to check
        col = 'cpd_id'
        if 'property' in table:
            col = 'property_id'
        elif 'spectra' in table:
            col = 'spectra_id'
        elif table in ['category, libraryentry']:
            return None

        # Query
        self.cursor.execute('SELECT MAX(%s) FROM %s' % (col, table))
        max_val = self.cursor.fetchone()[0]

        # Return result
        if max_val is None:
            return -1
        return max_val

    def drop_ineligible_properties(self, df, exceptions=[], inplace=False):
        '''
        Remove columns that can not be loaded into the database. To see
        properties available for insertion into DB, use
        get_property_categories()
        '''

        # Get property categories available
        categories = self.get_property_categories()

        # Add in exceptions (cols not incategories but should remain in df)
        categories.extend(exceptions)

        # Get columns to drop
        drop_cols = list(set(df.columns).difference(set(categories)))

        # Report which columns did not pass
        m = 'The following columns could not be added to properties table: %s'
        print(m % (', '.join(drop_cols)))

        # Handle inplace
        if inplace:
            df.drop(columns=drop_cols, inplace=inplace)
            return df

        # Return df with dropped columns
        return df.drop(columns=drop_cols)

    def _populate_db(self, df, table, commit=True):
        '''Drops any entries with nan values then loads full table into DB.
        Also tracks the number actually inserted into DB.
        Returns cursor, the number of entries added, and the number of entries
        ignored (not including those with NA values)'''

        # Ignore any entries with nan results
        df = df.dropna()

        # Get number of entries in DB table before adding anything
        max1 = self.get_num_entries(table)

        # Design query
        query = 'insert ignore into {0}({1}) values ({2})'
        s = ['\'%s\'' for x in range(len(df.columns))]
        query = query.format(table, ','.join(df.columns), ','.join(s))

        # Make calls to db, add each row
        for row in df.itertuples(index=False):
            try:
                self.cursor.execute(query % tuple(row))
            except mysql.connector.ProgrammingError:
                print('Failed query: {}'.format(query % tuple(row)))

        # Commit changes
        if commit:
            self.conn.commit()

        # Assess entries added vs skipped
        max2 = self.get_num_entries(table)
        entries_added = max2 - max1
        entries_skipped = len(df) - entries_added

        return entries_added, entries_skipped

    def print_added_entries(self, table, num1, num2):
        m = '{}: {} entries added successfully. {} not added.\n'
        print(m.format(table, num1, num2))
        return

    def populate_compounds(self, df):

        # Form table and populate
        num1, num2 = self._populate_db(df[['cpd_id', 'inchi_key']], 'compound')

        # Report
        self.print_added_entries('Compound', num1, num2)
        return

    def populate_libraryentry(self, df, lib_name):

        # Form table
        df = df[['cpd_id', 'lib_id']]
        df['lib_name'] = lib_name

        # Populate
        num1, num2 = self._populate_db(df, 'libraryentry')

        # Report
        self.print_added_entries('LibraryEntry', num1, num2)
        return

    def populate_mass(self, df):

        # Form table
        df = df[['cpd_id', 'Mass']]
        df.rename(columns={'Mass': 'value'}, inplace=True)

        # Populate
        num1, num2 = self._populate_db(df, 'mass')

        # Report
        self.print_added_entries('Mass', num1, num2)
        return

    def populate_ms2_spectra(self, df, frag_split=';', info_split=','):
        '''MS2 cols must all be labeled MSMS {Mode} {Voltage}
        (e.g. 'MSMS Positive 10'). Spectra muss be in the following
        format: {mass}{info_split}{intensity}{frag_slit}{mass}{info_split}...'''

        # Get MS2 cols
        cols = [x for x in df.columns if 'MSMS' in x]

        # Track total number of entries added
        num1 = 0  # main table (ms2_spectra)
        num2 = 0  # main table (ms2_spectra)
        num1d = 0  # detail table (fragment)
        num2d = 0  # detail table (fragment)

        for col in cols:

            # Add spectra to head table
            info = col.split(' ')
            spectra = df[['cpd_id', col]]

            # Drop any nans
            spectra = spectra.dropna()

            # Assign voltage (boolean)
            if 'pos' in info[1].lower():
                spectra['mode'] = 1
            else:
                spectra['mode'] = 0

            # Add voltage
            spectra['voltage'] = int(info[2])

            # Add spectra id
            start_id = self.get_max_id('ms2_spectra') + 1
            spectra['spectra_id'] = range(start_id, start_id + len(spectra))

            # Populate master table
            num1t, num2t = self._populate_db(spectra.drop(columns=[col]), 'ms2_spectra')
            num1 += num1t
            num2 += num2t

            # Add each spectra to detail table (int and mass info)
            for i, row in spectra.iterrows():

                # Text to matrix (col 0: mass, col 1: intensity)
                fragments = np.array([y.split(info_split)
                                      for y in row[col].split(frag_split)],
                                     np.float64)

                # Make all intensities 0-1
                fragments[:, 1] /= np.sum(fragments[:, 1])

                # Matrix to dataframe
                fragments = pd.DataFrame(fragments,
                                         columns=['mass', 'relative_intensity'],
                                         dtype=np.float64)
                fragments['spectra_id'] = row['spectra_id']

                # Round numbers to decrease storage size
                fragments = np.around(fragments, 4)

                # Populate fragment table
                num1t, num2t = self._populate_db(fragments, 'fragment')
                num1d += num1t
                num2d += num2t

        # Report spectra additions
        self.print_added_entries('MS2_Spectra', num1, num2)

        # Report fragment additions
        self.print_added_entries('Fragment', num1d, num2d)

        return num1, num2, num1d, num2d


    def populate_property(self, df, exceptions=['cpd_id']):

        # Remove columns with unknown properties
        df = df.copy()
        self.drop_ineligible_properties(df, exceptions=exceptions, inplace=True)

        # Form table
        value_vars = list(df.columns)
        value_vars.remove('cpd_id')
        df = pd.melt(df, id_vars=['cpd_id'], value_vars=value_vars)
        df.rename(columns={'variable': 'category'}, inplace=True)

        # Drop empty properties
        df.dropna(axis='rows', inplace=True)

        # Populate
        num1, num2 = self._populate_db(df, 'property')

        # Report
        self.print_added_entries('Property', num1, num2)
        return

    def populate_db(self, df, lib_name):
        '''
        Generic function to load full library into the database. Assumes
        provided df has the following columns: 'cpd_id', 'inchi_key', 'lib_id'
        '''

        # Populate compounds
        self.populate_compounds(df)

        # Populate library entries
        self.populate_libraryentry(df, lib_name)

        # Populate masses
        self.populate_mass(df)

        # Populate MS2s
        self.populate_ms2_spectra(df)

        # Populate all other properties
        self.populate_property(df)

        return

    def close_connection(self):
        '''Close DB connection'''
        self.conn.close()
        self.conn = None
        self.cursor = None
        self.categories = None
