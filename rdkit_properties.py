from rdkit.Chem import MolFromSmiles, MolFromInchi, Descriptors, rdMolDescriptors, MolToSmiles, MolToInchiKey, MolToInchi
import pandas as pd


def calculate_rdkit_properties(structure, properties=['Formula', 'Mass', 'SMILES', 'inchi_key', 'InChI']):
    '''Calculates RDKit properties for a given compound. Properties to choose
    from: 'Formula', 'Mass', 'SMILES', 'inchi_key', 'InChI'.'''

    # Convert to mol
    if 'InChI' in structure:
        mol = MolFromInchi(structure)
    else:  # Assume SMILES
        mol = MolFromSmiles(structure)

    # If mol failed, return an empty row
    if mol is None:
        print('Structure failed: {}'.format(structure))
        return pd.Series()

    # Mol created. Convert to desired properties
    d = {}
    if 'Formula' in properties:
        d['Formula'] = [rdMolDescriptors.CalcMolFormula(mol)]
    if 'Mass' in properties:
        d['Mass'] = [Descriptors.ExactMolWt(mol)]
    if 'SMILES' in properties:
        d['SMILES'] = [MolToSmiles(mol)]
    if 'inchi_key' in properties:
        d['inchi_key'] = [MolToInchiKey(mol)]
    if 'InChI' in properties:
        d['InChI'] = [MolToInchi(mol)]

    return pd.Series(d)


def add_rdkit_properties(df, structure_col='SMILES',
                         properties=['Formula', 'Mass', 'SMILES', 'inchi_key', 'InChI']):
    '''
    Adds RDKit properties for all compounds in given df. Properties to choose
    from: 'Formula', 'Mass', 'SMILES', 'inchi_key', 'InChI'. Takes approx
    3.87ms/cpd (judging from HMDB)
    '''

    # Initialize
    props = pd.DataFrame(columns=properties)

    # Iterate through each compound
    for i, row in df.iterrows():
        new_row = calculate_rdkit_properties(row[structure_col], properties=properties)
        new_row.name = i
        props = props.append(new_row)

    return pd.concat((df, props), axis=1)
