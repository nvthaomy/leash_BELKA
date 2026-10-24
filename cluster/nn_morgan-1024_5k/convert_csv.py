import pandas as pd

protein_names = ['BRD4','HSA','sEH']

for protein in protein_names:
    file = 'test_{}_1024bits.pkl'.format(protein)
    print('\nReading file:',file)
    df = pd.read_pickle(file)
    # only keep columns id, molecule_smiles, and binds
    try:
        df = df[['id','molecule_smiles','binds']]
    except:
        df = df[['id','molecule_smiles']]
    # modify the molecule_smiles column
    df['molecule_smiles_Me'] = df['molecule_smiles'].str.replace('[Dy]','C')
    df['molecule_smiles_Bu'] = df['molecule_smiles'].str.replace('[Dy]','CCCC')

    # check
    print('\nChecking the first entry in the dataframe')
    print(df['molecule_smiles'].iloc[2])
    print(df['molecule_smiles_Me'].iloc[2])
    print(df['molecule_smiles_Bu'].iloc[2])

    # save as csv
    file_out = file.replace('.pkl','.csv')
    df.to_csv(file_out, index=False)
