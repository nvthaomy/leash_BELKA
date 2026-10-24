import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import gc, os
from tools import *
"""sample from from 7 scenerio of encounter positive samples in 3 proteins together """

n_samples_tar = int(5e6)
posrate_tar = 0.01

print('\nLoading data...')
train_data = pq.read_table('train_folds.parquet').to_pandas() 
print(train_data.head())
print(train_data.columns)

# train_data = pq.read_table('../data/split1/train_folds_small-10000.parquet').to_pandas() 
print('Train data shape:', train_data.shape)
print('\nPositive rates:\n{}'.format(train_data[['binds_BRD4','binds_HSA', 'binds_sEH']].mean()))
print('\nPositive counts:\n{}'.format(train_data[['binds_BRD4','binds_HSA', 'binds_sEH']].sum()))

n_pos_max = {(0,1,2): train_data[(train_data['binds_BRD4'] == True) & (train_data['binds_HSA'] == True) & (train_data['binds_sEH'] == True)].shape[0], 
               (0,1): train_data[(train_data['binds_BRD4'] == True) & (train_data['binds_HSA'] == True) & (train_data['binds_sEH'] == False)].shape[0],
               (0,2): train_data[(train_data['binds_BRD4'] == True) & (train_data['binds_HSA'] == False) & (train_data['binds_sEH'] == True)].shape[0],
               (1,2): train_data[(train_data['binds_BRD4'] == False) & (train_data['binds_HSA'] == True) & (train_data['binds_sEH'] == True)].shape[0],
               0: train_data[(train_data['binds_BRD4'] == True) & (train_data['binds_HSA'] == False) & (train_data['binds_sEH'] == False)].shape[0],
               1: train_data[(train_data['binds_BRD4'] == False) & (train_data['binds_HSA'] == True) & (train_data['binds_sEH'] == False)].shape[0],
               2: train_data[(train_data['binds_BRD4'] == False) & (train_data['binds_HSA'] == False) & (train_data['binds_sEH'] == True)].shape[0]}

print('\nNumber of positive samples in original data set based on different scenarios:\n{}'.format(n_pos_max))

# find number of positive samples for each protein:
n_pos_tar = {'BRD4': min(int(n_samples_tar * posrate_tar), 456964),
         'HSA': min(int(n_samples_tar * posrate_tar), 408410),
         'sEH': min(int(n_samples_tar * posrate_tar), 724532)}
posrate_tar = min(n_pos_tar.values()) / n_samples_tar
n_pos_tar_min = min(n_pos_tar.values())
print('\nTarget numbers of positive samples:\n{}'.format(n_pos_tar))

# prepare new split column
split_name = 'posrate-{:.3f}_size-{}M'.format(posrate_tar, n_samples_tar/1e6)
train_data[split_name] = False

train_data_sample = train_data.copy()
n_cycles = 20
n_pos_curr = {'BRD4': 0, 'HSA': 0, 'sEH': 0}

# composition of postive samples based on n_pos_tar_min, indices indicate which proteins are in the positive sample
# >~ 5M samples, 0.05 posrate
composition = {(0,1,2): 0.000152, 
               (0,1): 0.00058,
               (0,2): 0.001462,
               (1,2): 0.19385,
               0: 0.26799,
               1: 0.26799,
               2: 0.26799}
# 5M samples, 0.01 posrate
composition = {(0,1,2): 0.00100, 
               (0,1): 0.00500,
               (0,2): 0.00500,
               (1,2): 0.01,
               0: 0.32633,
               1: 0.32633,
               2: 0.32633}

n_pos = {key: int(n_pos_tar_min * value) for key, value in composition.items()}
print('\nInitial number of positive samples to draw for each case\n{}'.format(n_pos))

# adjusting n_pos so that the total number of positive samples matches n_pos_tar
for i, protein in enumerate(['BRD4', 'HSA', 'sEH']):
    n_pos_protein = 0
    for indices, n_samples in n_pos.items():
        # check if indices is tuple or int
        if isinstance(indices, int):
            if i == indices:
                n_pos_protein += n_samples
        else:
            if i in list(indices):
                n_pos_protein += n_samples
    diff = n_pos_tar[protein] - n_pos_protein
    n_pos[(i)] += diff

print('\nNumber of positive samples to draw for each case after matching the target positive samples\n{}'.format(n_pos))

# adjusting n_pos so that the number for each case does not exceed n_pos_max
for key, count in n_pos.items():
    if count > n_pos_max[key]:
        n_pos[key] = n_pos_max[key]
print('\nNumber of positive samples to draw for each case after adjusted to the maxium positive samples in original data set\n{}'.format(n_pos))

for indices, n_samples in n_pos.items():    

    if indices == (0,1,2):
        selected_rows = train_data[(train_data['binds_BRD4'] == True) & (train_data['binds_HSA'] == True) & (train_data['binds_sEH'] == True)].sample(n=n_samples)
    elif indices == (0,1):
        selected_rows = train_data[(train_data['binds_BRD4'] == True) & (train_data['binds_HSA'] == True) & (train_data['binds_sEH'] == False)].sample(n=n_samples)
    elif indices == (0,2):
        selected_rows = train_data[(train_data['binds_BRD4'] == True) & (train_data['binds_HSA'] == False) & (train_data['binds_sEH'] == True)].sample(n=n_samples)
    elif indices == (1,2):
        selected_rows = train_data[(train_data['binds_BRD4'] == False) & (train_data['binds_HSA'] == True) & (train_data['binds_sEH'] == True)].sample(n=n_samples)
    elif indices == 0:
        selected_rows = train_data[(train_data['binds_BRD4'] == True) & (train_data['binds_HSA'] == False) & (train_data['binds_sEH'] == False)].sample(n=n_samples)
    elif indices == 1:
        selected_rows = train_data[(train_data['binds_BRD4'] == False) & (train_data['binds_HSA'] == True) & (train_data['binds_sEH'] == False)].sample(n=n_samples)
    elif indices == 2:
        selected_rows = train_data[(train_data['binds_BRD4'] == False) & (train_data['binds_HSA'] == False) & (train_data['binds_sEH'] == True)].sample(n=n_samples)
    
    train_data.loc[selected_rows.index, split_name] = True

# get negative samples
n_neg_samples = n_samples_tar - train_data[split_name].sum()
selected_rows = train_data[(train_data['binds_BRD4'] == False) & (train_data['binds_HSA'] == False) & (train_data['binds_sEH'] == False)].sample(n=n_neg_samples)
train_data.loc[selected_rows.index, split_name] = True  

print('\nNumber of samples in new split: {}'.format(train_data[split_name].sum()))
print('\nPositive rates:\n{}'.format(train_data[train_data[split_name] == True][['binds_BRD4','binds_HSA', 'binds_sEH']].mean()))
print('\nPositive counts:\n{}'.format(train_data[train_data[split_name] == True][['binds_BRD4','binds_HSA', 'binds_sEH']].sum()))
train_data.to_parquet('train_folds.parquet')

n_share = train_data[(train_data[split_name] == True) & (train_data['full_test'] == True)].shape[0]

print('\nNumber of samples in new split that are in the full_test set: {}'.format(n_share))

# write columns to file
with open('train_folds_columns.txt', 'w') as f:
    for item in train_data.columns:
        f.write("%s\n" % item)

print('\nDone!')
# while train_data[train_data[split_name] == True].shape[0] < n_samples_tar:
#     for cycle in range(n_cycles):
#         for protein in ['BRD4', 'HSA', 'sEH']:

#             n_pos_this = int((n_pos_tar[protein]-n_pos_curr[protein]) / ((n_cycles-cycle) * 3)) 
#             n_neg_this = int(n_samples_tar / (n_cycles * 3) - n_pos_this)
#             print('\nDraw {} positive samples for protein {}'.format(n_pos_this, protein))
        
#             df_temp_pos = train_data_sample[train_data_sample['binds_{}'.format(protein)] == True].sample(n=n_pos_this)
#             df_temp_neg = train_data_sample[train_data_sample['binds_{}'.format(protein)] == False].sample(n=n_neg_this)
            
#             # update split
#             train_data.loc[df_temp_pos.index, split_name] = True
#             # remove samples from train_data_sample
#             train_data_sample = train_data_sample.drop(df_temp_pos.index)
#             train_data_sample = train_data_sample.drop(df_temp_neg.index)

#             posrate_curr = train_data[train_data[split_name] == True][['binds_BRD4','binds_HSA', 'binds_sEH']].mean()
#             n_pos_curr = {'BRD4': train_data[train_data[split_name] == True]['binds_BRD4'].sum(),
#                             'HSA': train_data[train_data[split_name] == True]['binds_HSA'].sum(),
#                             'sEH': train_data[train_data[split_name] == True]['binds_sEH'].sum()}

#         print('\nCycle: {}, Current positive rates:\n{}'.format(cycle+1, posrate_curr))

#         if cycle % 10 == 0:
#             train_data.to_parquet('train_folds.parquet')
#             print('Number of samples in new split:', train_data[split_name].sum())

# posrates = {'BRD4': 0, 'HSA': -1, 'sEH': 0}
# posrate_curr = 0

# train_data_sample = train_data.copy()
# filters = ['binds_BRD4', 'binds_HSA', 'binds_sEH']
# cycle = 1
# n_warnings_max = 10
# n_warnings = 0
# check_memory(pid=os.getpid())

# n_pos_samples = int(n_samples_tar / 100 / 3) # positive samples to draw per cycle
# n_neg_samples = int((n_pos_samples - posrate_tar * n_pos_samples) / posrate_tar )# negative samples to draw per cycle
# print('Drawing {} samples per cycle'.format(n_pos_samples))

# while posrate_curr < posrate_tar:
#     # find key in posrates with lowest value
#     filter = filters[np.argmin([posrates['BRD4'], posrates['HSA'], posrates['sEH']])]
    
#     if train_data_sample[train_data_sample[filter] == True].shape[0] < n_pos_samples:
#         n_warnings += 1
#         if n_warnings > n_warnings_max: # terminate if see many consecutive warnings
#             print('No more samples to draw from, terminate')
#             break
#         else:
#             n_warnings = 0 # reset counter
#             continue

#     df_temp_pos = train_data_sample[train_data_sample[filter] == True].sample(n=n_pos_samples)
#     df_temp_neg = train_data_sample[train_data_sample[filter] == False].sample(n=n_neg_samples)

#     # remove sampled from train_data_sample
#     train_data_sample = train_data_sample.drop(df_temp_pos.index)
#     train_data_sample = train_data_sample.drop(df_temp_neg.index)
    
#     # update split
#     train_data.loc[df_temp_pos.index, split_name] = True
#     train_data.loc[df_temp_neg.index, split_name] = True
#     posrate_curr = train_data[train_data[split_name] == True][['binds_BRD4','binds_HSA', 'binds_sEH']].mean()

#     # save to dict
#     posrates = {'BRD4': train_data[train_data[split_name] == True]['binds_BRD4'].mean(),
#                 'HSA': train_data[train_data[split_name] == True]['binds_HSA'].mean(),
#                 'sEH': train_data[train_data[split_name] == True]['binds_sEH'].mean()}
                
#     check_memory(pid=os.getpid())
#     print('\nCycle: {}, Current positive rates:\n{}'.format(cycle, posrate_curr))
#     cycle += 1
#     posrate_curr = min(posrate_curr)

#     if cycle % 10 == 0:
#         train_data.to_parquet('train_folds.parquet')
#         print('Number of samples in new split:', train_data[split_name].sum())

    
# # count number of samples in split
# print('\nNumber of samples in new split:', train_data[split_name].sum())
# # save train_data
# train_data.to_parquet('train_folds.parquet')