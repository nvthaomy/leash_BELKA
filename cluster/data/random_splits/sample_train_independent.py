import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import gc, os
from time import time

"""sample from each protein independently then combine them """

n_samples_tar = int(10e6) # number of samples to draw for each protein
posrate_tar = 0.05 # positive rate for each protein
output = './train_folds.parquet'

t0 = time()
print('\nLoading data...')
#train_data = pd.read_pickle('../train_short_9000.pkl')
train_data = pq.read_table('./train_folds.parquet').to_pandas()
#train_data = pq.read_table('./train_folds.parquet').to_pandas()
print('Data loaded in {:.2f} seconds'.format(time()-t0))

print(train_data.head())
print(train_data.columns)
 
print('Train data shape:', train_data.shape)
for protein in ['BRD4', 'HSA', 'sEH']:
    print('\nProtein: {}'.format(protein))
    print('Positive rate: {:.3f}'.format(train_data[train_data['protein_name'] == protein]['binds'].mean()))
    print('Positive count: {:.3f}'.format(train_data[train_data['protein_name'] == protein]['binds'].sum()))

# find number of positive samples for each protein:
n_pos_tar = {'BRD4': min(int(n_samples_tar * posrate_tar), 456964),
         'HSA': min(int(n_samples_tar * posrate_tar), 408410),
         'sEH': min(int(n_samples_tar * posrate_tar), 724532)}
posrate_tar = {key: n_pos_tar[key]/n_samples_tar for key in n_pos_tar.keys()}
n_pos_tar_min = min(n_pos_tar.values())
print('\nTarget numbers of positive samples:\n{}'.format(n_pos_tar))

# prepare new split column
split_dict = {}
for protein in ['BRD4', 'HSA', 'sEH']:
    split_name = '{}_posrate-{:.3f}_size-{}M'.format(protein, posrate_tar[protein], n_samples_tar/1e6)
    train_data[split_name] = False
    split_dict[protein] = split_name

for protein in ['BRD4', 'HSA', 'sEH']:
    split_name = split_dict[protein]

    # get positive samples
    selected_rows = train_data[train_data['protein_name'] == protein][train_data['binds']==1].sample(n=n_pos_tar[protein])
    train_data.loc[selected_rows.index, split_name] = True

    # get negative samples
    selected_rows = train_data[train_data['protein_name'] == protein][train_data['binds']==0].sample(n=n_samples_tar-n_pos_tar[protein])
    train_data.loc[selected_rows.index, split_name] = True

    # print number of samples in new split
    print('\nStatistics for split {}:'.format(split_name))
    print('\nPositive rates:\n{}'.format(train_data[train_data[split_name] == True]['binds'].mean()))
    print('\nPositive counts:\n{}'.format(train_data[train_data[split_name] == True]['binds'].sum()))

split_names = list(split_dict.values())
n_share = train_data[(train_data[split_names[0]] == True) & (train_data[split_names[1]] == True) & (train_data[split_names[2]] == True)].shape[0]
print('\nNumber of samples shared between 3 splits: {}'.format(n_share))

train_data.to_parquet(output)

# write columns to file
with open('train_folds_columns.txt', 'w') as f:
    for item in train_data.columns:
        f.write("%s\n" % item)

print('\nDone!')
