import mysql.connector
import pandas as pd
import numpy as np
import os

# Check secure_file_priv for available folders to save to
TMP_SAVE_PATH = 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/'


class DatabaseManager():

    def __init__(self, host='localhost', user='root', passwd='password',
                 database='compound_db', stay_open=False):

        # Initialize attributes
        self.host = host
        self.user = user
        self.passwd = passwd
        self.database = database
        self.stay_open = stay_open
        self.conn = None
        self.cursor = None
        self.categories = None
        self.adduct_tables = {'[m+h]': 'mph',
                              '[m+na]': 'mpna',
                              '[m-h]': 'mmh'}

        # If this manager should keep the connection open, start it now
        if stay_open:
            self.open_db()

        return

    def open_db(self, override=False):
        '''Connect to compound DB and store handles'''
        if self.conn is None or self.cursor is None or override:
            self.conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                passwd=self.passwd,
                database=self.database
            )
            self.cursor = self.conn.cursor(buffered=True)
        return

    def close_db(self, override=False):
        '''Close DB connection'''
        if not self.stay_open or override:
            self.conn.close()
            self.conn = None
            self.cursor = None
        return

    # ----------------------------------------------------------------------
    # Getters

    def get_property_categories(self):
        # Get permitted property categories
        if self.categories is None:
            query = 'SELECT name FROM category'
            res = self.run_query(query)
            self.categories = res['name'].values.tolist()
        return self.categories[:]

    def get_cursor(self):
        return self.cursor

    def get_connector(self):
        return self.conn

    def get_adducts(self):
        return self.adduct_tables.keys()

    # ----------------------------------------------------------------------
    # Setters

    def set_stay_open(self, stay_open):
        if type(stay_open) != bool:
            raise ValueError('Invalid entry for stay_open. Use boolean')
        self.stay_open = stay_open

    # ----------------------------------------------------------------------
    # Query database

    def run_query(self, query, single_val=False, no_return=False):
        # TODO: check query
        self.open_db()
        if single_val or no_return:
            self.cursor.execute(query)
            if no_return:
                return
            res = self.cursor.fetchone()[0]
        else:
            res = pd.read_sql(query, con=self.conn)
        self.close_db()
        return res

    def prep_string(self, s):
        '''Get string ready for DB query by adding quotes if needed'''
        if type(s) == str and (s[0] != '\'' or s[-1] != '\''):
            return '\'' + s + '\''
        return s

    def match_exists(self, table, col, val, rel_error=0, abs_error=0):
        '''Check if the given value exists in the given table+col'''
        return bool(len(self.fetch_matches(table, col, val, rel_error=rel_error,
                                           abs_error=abs_error)))

    def fetch_matches(self, table, col, val, rel_error=0, abs_error=0,
                      append_str='', num_dec=None):
        '''Return matches in the given table+col'''

        # NaNs raise errors
        if None in [table, col, val]:
            m = 'NaN passed in match query: {}'
            raise ValueError(m.format([table, col, val]))

        # Switch for string value
        if type(val) == str:
            return self.fetch_matches_string(table, col, val,
                                             append_str=append_str)

        # Assume float or integer otherwise
        return self.fetch_matches_within_error(table, col, val,
                                               rel_error=rel_error,
                                               abs_error=abs_error,
                                               append_str=append_str,
                                               num_dec=num_dec)

    def fetch_matches_string(self, table, col, val, append_str=''):
        '''Return matches for a string value in the given table+col'''
        val = self.prep_string(val)
        query = 'SELECT * FROM {} WHERE {} = {}{};'
        query = query.format(table, col, val, append_str)
        return self.run_query(query)

    # Rel and abs error, change to return actual val
    def fetch_matches_within_error(self, table, col, val, rel_error=0,
                                   abs_error=0, append_str='', num_dec=None):
        '''Return matches for a numeric value in the given table+col. The error
        provided should already be divded by desired level of accuracy
        (e.g. 5 / 100 for 5% error).'''
        if rel_error == 0 and abs_error == 0:
            query = 'SELECT * FROM {} WHERE {} = {}{};'
            query = query.format(table, col, val, append_str)
        else:
            min_val, max_val = self.get_range(val, rel_error=rel_error,
                                              abs_error=abs_error)
            if num_dec is not None:
                min_val = round(min_val, num_dec)
                max_val = round(max_val, num_dec)
            query = 'SELECT * FROM {} WHERE {} BETWEEN {} and {}{};'
            query = query.format(table, col, min_val, max_val, append_str)
        return self.run_query(query)

    def get_range(self, val, abs_error=0, rel_error=0):
        '''Calculate range of values covered by provided value and
        associated error. If both abs_error and rel_error are passed,
        returns results based on the larger error range. Assumes val is
        positive.'''

        # Check input
        if abs_error < 0:
            m = 'Negative error passed for absolute error: {}'
            raise ValueError(m.format(abs_error))
        if rel_error < 0:
            m = 'Negative error passed for relative error: {}'
            raise ValueError(m.format(rel_error))

        # Return range based on the larger error window
        if abs_error > val * rel_error:
            return val - abs_error, val + abs_error
        return val * (1 - rel_error), val * (1 + rel_error)

    def fetch_mass_ccs_matches(self, mass, ccs, adduct, mass_rel_error=0,
                               mass_abs_error=0, ccs_rel_error=0,
                               ccs_abs_error=0):

        adduct = self._translate_adduct(adduct, throw_error=True)
        db_table = 'ccs_{}'.format(adduct)

        # Calculate mass and CCSs ranges to search for
        min_mass, max_mass = self.get_range(mass, rel_error=mass_rel_error,
                                            abs_error=mass_abs_error)
        min_ccs, max_ccs = self.get_range(ccs, rel_error=ccs_rel_error,
                                          abs_error=ccs_abs_error)

        # Design query
        query = '''
                SELECT mass.cpd_id, mass.value mass, c.value ccs
                FROM mass
                LEFT JOIN {} c
                ON mass.cpd_id=c.cpd_id
                WHERE mass.value BETWEEN {:.4f} and {:.4f}
                AND c.value BETWEEN {:.2f} and {:.2f}
                '''
        query = query.format(db_table, min_mass, max_mass, min_ccs, max_ccs)

        # Query and return resulting table
        return self.run_query(query)

    def fetch_property_match(self, category, val, rel_error=0, abs_error=0,
                             num_dec=None):
        if category not in self.get_property_categories():
            m = 'Invalid category for property table: {}. Use get_property_categories() for available categories.'
            raise ValueError(m.format(category))
        append_str = ' AND category = {}'.format(self.prep_string(category))
        return self.fetch_matches('property', 'value', val,
                                  rel_error=rel_error, abs_error=abs_error,
                                  append_str=append_str, num_dec=num_dec)

    def get_cpd_id(self, inchi_key):
        '''Get cpd_id of given InChIKey'''

        inchi_key = self.prep_string(inchi_key)
        query = 'SELECT cpd_id FROM compound WHERE inchi_key = %s;' % inchi_key
        res = self.run_query(query).values[0][0]
        return res

    def get_num_entries(self, table):
        '''Get height of table'''
        query = 'SELECT COUNT(*) FROM %s' % table
        return self.run_query(query, single_val=True)

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
        query = 'SELECT MAX(%s) FROM %s' % (col, table)
        max_val = self.run_query(query, single_val=True)

        # Return result
        if max_val is None:
            return -1
        return max_val

    # ----------------------------------------------------------------------
    # Prep data for input into database

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
            elif self.match_exists('compound', 'inchi_key', val):
                cpd_id = self.get_cpd_id(val)
                df.at[i, 'cpd_id'] = cpd_id
                d[val] = cpd_id
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
        self.set_stay_open(True)
        self.create_cpd_ids(df, inplace=True, start_id=start_id)
        self.set_stay_open(False)

        return df

    # ----------------------------------------------------------------------
    # Populate database

    def _populate_db(self, df, table, commit=True, sep='\t',
                     line_terminator='\n',
                     filename=os.path.join(TMP_SAVE_PATH, 'tmp.txt')):
        '''Drops any entries with nan values then loads full table into DB.
        Also tracks the number actually inserted into DB.
        Returns the number of entries added, and the number of entries
        ignored (not including those with NA values)'''

        # Ignore any entries with nan results
        df = df.dropna()

        # Get number of entries in DB table before adding anything
        max1 = self.get_num_entries(table)

        # Save df as .txt to allow bulk upload into database
        df.to_csv(filename, sep=sep, line_terminator=line_terminator,
                  index=False, header=False)

        # Design query for bulk upload to database
        query = '''
                LOAD DATA INFILE %s
                   IGNORE
                   INTO TABLE %s
                   FIELDS TERMINATED BY %s
                   LINES TERMINATED BY %s;
                '''
        query = query % (self.prep_string(filename),
                         table,
                         self.prep_string(sep),
                         self.prep_string(line_terminator))

        # Upload
        self.open_db()
        self.cursor.execute(query)
        if commit:
            self.conn.commit()
        self.close_db()

        # Assess entries added vs skipped
        max2 = self.get_num_entries(table)
        entries_added = max2 - max1
        entries_skipped = len(df) - entries_added

        return entries_added, entries_skipped

    def print_added_entries(self, table, num1, num2):
        m = '{}: {} entries added successfully. {} not added.\n'
        print(m.format(table, num1, num2))
        return

    def _translate_adduct(self, adduct, throw_error=False):

        # Translate adduct name into table identifier
        adduct = adduct.lower()
        for k, v in self.adduct_tables.items():
            if k.lower() in adduct or v.lower() == adduct:
                return v

        # Throw error if requested
        if throw_error:
            m = 'No adduct name found: {}. Use get_adducts() for available options.'
            raise ValueError(m.format(adduct))

        # Otherwise, just return None
        return None

    def populate_library(self, lib_info):

        # Form table and populate
        df = pd.DataFrame(lib_info, index=[0])
        num1, num2 = self._populate_db(df, 'library')

        # Report
        self.print_added_entries('Library', num1, num2)
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

        # Reorder to fit table definition
        df = df[['lib_id', 'lib_name', 'cpd_id']]

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
    def _populate_ms2_spectra(self, spectra, fragments, adduct):
        num1s, num2s = self._populate_db(spectra, 'ms2_spectra')
        num1f, num2f = self._populate_db(fragments, 'fragment_{}'.format(adduct))
        return num1s, num2s, num1f, num2f

    def _process_spectra(self, spectra, info, start_id):

        # Drop any nans
        spectra = spectra.dropna()

        # Add voltage
        spectra['voltage'] = int(info[2])

        # Add spectra id
        n = len(spectra)
        spectra['spectra_id'] = list(range(start_id, start_id + n))
        start_id += n

        return spectra, start_id

    def _process_fragments(self, s, id, frag_split=';', info_split=','):

        # Text to matrix (col 0: mass, col 1: intensity)
        fragments = np.array([y.split(info_split)
                              for y in s.split(frag_split)],
                             np.float64)

        # Make all intensities 0-1
        fragments[:, 1] /= np.sum(fragments[:, 1])

        # Matrix to dataframe
        fragments = pd.DataFrame(fragments,
                                 columns=['mass', 'relative_intensity'],
                                 dtype=np.float64)
        fragments['spectra_id'] = id
        fragments = fragments[['spectra_id', 'mass',
                               'relative_intensity']]

        # Round numbers to decrease storage size
        fragments = fragments.round({'mass': 4,
                                     'relative_intensity': 4})

        return fragments

    def populate_ms2_spectra(self, df, frag_split=';', info_split=',',
                             max_len=100000):
        '''MS2 cols must all be labeled MSMS {Mode} {Voltage}
        (e.g. 'MSMS Positive 10'). Spectra must be in the following
        format: {mass}{info_split}{intensity}{frag_slit}{mass}{info_split}...'''

        # Get MS2 cols
        cols = [x for x in df.columns if 'MSMS' in x]

        # Init tables that will be added to the database
        spectra_all = pd.DataFrame(columns=['cpd_id', 'adduct', 'voltage', 'spectra_id'])
        fragments_all = pd.DataFrame(columns=['spectra_id', 'mass', 'relative_intensity'])

        # Get spectra_id to start with
        start_id = self.get_max_id('ms2_spectra') + 1

        # Initialize upload counts
        nums = [0,  # Number of entries added to spectra table
                0,  # Number of entries prepared but not added to spectra table
                0,  # Number of entries added to fragment table
                0]  # Number of entries prepared but not added to fragment table

        # Cycle through all columns with MS2 data
        for col in cols:

            # Assign adduct
            adduct = self._translate_adduct(col)
            if adduct is None:
                m = 'Could not add potential MSMS column: {}. Check adduct name.'
                print(m.format(col))

            if adduct is not None:
                spectra['adduct'] = adduct

                # Get head spectra info for this column
                spectra, start_id = self._process_spectra(df[['cpd_id', col]],
                                                          col.split(' '), start_id)

                # Add to master df
                spectra_all = pd.concat([spectra_all, spectra.drop(columns=[col])],
                                        ignore_index=True)

                # Add each spectra to detail table (int and mass info)
                for i, row in spectra.iterrows():

                    # Get fragments in this cell
                    fragments = self._process_fragments(row[col], row['spectra_id'],
                                                        frag_split=frag_split,
                                                        info_split=info_split)

                    # Add to master df
                    fragments_all = pd.concat([fragments_all, fragments],
                                              ignore_index=True)

                    # If over max_len, upload now, reset, then continue
                    # Avoids too large of a df hogging memory
                    if len(fragments_all) >= max_len:
                        numst = self._populate_ms2_spectra(spectra_all, fragments_all, adduct)
                        nums = [nums[i] + numst[i] for i in range(len(nums))]

                        # Reset
                        spectra_all, fragments_all = spectra_all[0: 0], fragments_all[0: 0]
                        print('Here')

                # Populate tables
                numst = self._populate_ms2_spectra(spectra_all, fragments_all, adduct)
                nums = [nums[i] + numst[i] for i in range(len(nums))]

                # Report spectra additions
                self.print_added_entries('MS2_Spectra', nums[0], nums[1])
                self.print_added_entries('Fragment_{}'.format(adduct), nums[2], nums[3])

        return

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

        # Reorder to fit table definition
        df = df[['cpd_id', 'category', 'value']]

        # Populate
        num1, num2 = self._populate_db(df, 'property')

        # Report
        self.print_added_entries('Property', num1, num2)
        return

    def _prepare_lib_info(self, lib_info):

        # If only a string is passed, assume this is the library name and
        # convert to dictionary
        if type(lib_info) == str:
            lib_info = {'name': lib_info}

        # Non-dict is not accepted after this stage
        if type(lib_info) != dict:
            m = 'Provided library information must be a string or dictionary, not {}'
            raise TypeError(m.format(type(lib_info)))

        # Ensure keys in dictionary are in lowercase to avoid downstream errors
        lib_info = dict((k.lower(), v) for k, v in lib_info.items())

        # Throw error if a name is not provided (the only NOT NULL col)
        if 'name' not in lib_info.keys():
            raise ValueError('No name provided in library info')

        # Only keep entries that can be inserted into database
        allowable_keys = ['name', 'doi', 'date']
        lib_info = {k: lib_info[k] for k in allowable_keys if k in lib_info}

        return lib_info

    def populate_db(self, df, lib_info, max_len=100000):
        '''
        Generic function to load full library into the database. Assumes
        provided df has the following columns: 'cpd_id', 'inchi_key', 'lib_id'
        '''

        # Prepare library information for downstream use
        lib_info = self._prepare_lib_info(lib_info)

        # Populate library
        self.populate_library(lib_info)

        # Populate compounds
        self.populate_compounds(df)

        # Populate library entries
        self.populate_libraryentry(df, lib_info['name'])

        # Populate masses
        self.populate_mass(df)

        # Populate MS2s
        self.populate_ms2_spectra(df, max_len=max_len)

        # Populate all other properties
        self.populate_property(df)

        return
