import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import gc, os


"""sample from each protein independently then combine them """

n_samples_tar = int(10e6) # number of samples to draw for each protein
posrate_tar = 0.05 # positive rate for each protein
combine = False # combine all proteins into one dataset or not

print('\nLoading data...')
train_data = pq.read_table('train_folds.parquet').to_pandas()

print(train_data.head())
print(train_data.columns)
 
print('Train data shape:', train_data.shape)
print('\nPositive rates:\n{}'.format(train_data[['binds_BRD4','binds_HSA', 'binds_sEH']].mean()))
print('\nPositive counts:\n{}'.format(train_data[['binds_BRD4','binds_HSA', 'binds_sEH']].sum()))

# find number of positive samples for each protein:
n_pos_tar = {'BRD4': min(int(n_samples_tar * posrate_tar), 456964),
         'HSA': min(int(n_samples_tar * posrate_tar), 408410),
         'sEH': min(int(n_samples_tar * posrate_tar), 724532)}
posrate_tar = {key: n_pos_tar[key]/n_samples_tar for key in n_pos_tar.keys()}
n_pos_tar_min = min(n_pos_tar.values())
print('\nTarget numbers of positive samples:\n{}'.format(n_pos_tar))

# prepare new split column
if combine:
    split_name = 'new_split'
    train_data[split_name] = False
else:
    split_dict = {}
    for protein in ['BRD4', 'HSA', 'sEH']:
        split_name = '{}_posrate-{:.3f}_size-{}M'.format(protein, posrate_tar[protein], n_samples_tar/1e6)
        train_data[split_name] = False
        split_dict[protein] = split_name

for protein in ['BRD4', 'HSA', 'sEH']:
    if combine: 
        split_name = 'new_split'
    else:
        split_name = split_dict[protein]

    # get positive samples
    selected_rows = train_data[train_data['binds_{}'.format(protein)] == True].sample(n=n_pos_tar[protein])
    train_data.loc[selected_rows.index, split_name] = True

    # get negative samples
    selected_rows = train_data[train_data['binds_{}'.format(protein)] == False].sample(n=n_samples_tar-n_pos_tar[protein])
    train_data.loc[selected_rows.index, split_name] = True

    # print number of samples in new split
    if not combine:
        print('\nStatistics for split {}:'.format(split_name))
        print('\nPositive rates:\n{}'.format(train_data[train_data[split_name] == True][['binds_BRD4','binds_HSA', 'binds_sEH']].mean()))
        print('\nPositive counts:\n{}'.format(train_data[train_data[split_name] == True][['binds_BRD4','binds_HSA', 'binds_sEH']].sum()))

if combine:
    n_samples_total = train_data[split_name].sum()
    # rename column
    split_name = 'indp_posrate-{:.3f}_size-each-{}M_size-total-{:.3f}M'.format(min(posrate_tar.values()), n_samples_tar/1e6, n_samples_total/1e6)
    train_data.rename(columns={'new_split': split_name}, inplace=True)
    print('\nNumber of samples in new split: {}'.format(n_samples_total))
    print('\nPositive rates:\n{}'.format(train_data[train_data[split_name] == True][['binds_BRD4','binds_HSA', 'binds_sEH']].mean()))
    print('\nPositive counts:\n{}'.format(train_data[train_data[split_name] == True][['binds_BRD4','binds_HSA', 'binds_sEH']].sum()))
else:
    split_names = list(split_dict.values())
    n_share = train_data[(train_data[split_names[0]] == True) & (train_data[split_names[1]] == True) & (train_data[split_names[2]] == True)].shape[0]
    print('\nNumber of samples shared between 3 splits: {}'.format(n_share))

train_data.to_parquet('train_folds.parquet')

# write columns to file
with open('train_folds_columns.txt', 'w') as f:
    for item in train_data.columns:
        f.write("%s\n" % item)

# n_share = train_data[(train_data[split_name] == True) & (train_data['full_test'] == True)].shape[0]
# print('\nNumber of samples in new split that are in the full_test set: {}'.format(n_share))

print('\nDone!')
