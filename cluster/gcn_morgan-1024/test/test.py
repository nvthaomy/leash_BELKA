import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.utils import shuffle

#train_data = pq.read_table('train_BRD4_posrate-0.050_size-5.0M.parquet').to_pandas().sort_values(by='id')
train_data_short = pq.read_table('train_short_BRD4_posrate-0.050_size-5.0M.parquet').to_pandas().sort_values(by='id')
train_data_short_shuffle = pq.read_table('train_short_shuffle_BRD4_posrate-0.050_size-5.0M.parquet').to_pandas().sort_values(by='id')

# print('train_data')
# print(train_data[['id','binds','BRD4_posrate-0.050_size-5.0M']].head(5))
# print(train_data.columns)

print('\ntrain_data_short')
print(train_data_short[['id','binds','BRD4_posrate-0.050_size-5.0M']].head(5))
print(train_data_short.columns)

print('\ntrain_data_short_shuffle')
print(train_data_short_shuffle[['id','binds','BRD4_posrate-0.050_size-5.0M']].head(5))
print(train_data_short_shuffle.columns)

col = 'molecule_smiles'
# val_list_1 = train_data[col].tolist()
val_list_2 = train_data_short[col].tolist()
val_list_3 = train_data_short_shuffle[col].tolist()

# compare if two lists are the same
# print('val_list_1 == val_list_2:', val_list_1 == val_list_2)
print('val_list_2 == val_list_3:', val_list_2 == val_list_3)