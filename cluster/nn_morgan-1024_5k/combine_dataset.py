import pandas as pd
import pyarrow.parquet as pq
import numpy as np

df1 = pq.read_table('train_BRD4_5000_posrate-0.05_1024bits.parquet').to_pandas() 
df2 = pd.read_csv('train_BRD4_5k_posrate_0_05_smiles_Bu_docked.csv')
fileout = 'train_BRD4_5000_posrate-0.05_1024bits_Bu_docked.parquet'

print('Original dataset shape:', df1.shape)

# remove extra columns
df2 = df2.drop(columns=['binds'])
df2 = df2.drop(columns=['molecule_smiles'])

df2_columns = df2.columns

# sort df2 and df1 by 'id'
df1 = df1.sort_values(by='id')
df2 = df2.sort_values(by='id')

# copy unique columns of df2 to df1
for column in df2_columns:
    if not column in df1.columns:
        df1[column] = np.array(df2[column])

print('Combined dataset shape:', df1.shape)
print('Columns:', df1.columns)
print(df1.head())
# save the combined dataset to parquet
df1.to_parquet(fileout)
df1.to_csv(fileout.replace('parquet', 'csv'), index=False)