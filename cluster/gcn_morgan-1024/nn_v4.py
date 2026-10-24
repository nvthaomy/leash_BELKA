import pandas as pd
import numpy as np
from time import time, sleep
from tools import *
import gc
import psutil
import pickle

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import average_precision_score

import torch
from torch import nn, optim
import torch.utils.data as data
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.loader import DataLoader as DataLoaderGCN
from imblearn.over_sampling import SMOTE
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt
import os, sys
from sklearn.utils import shuffle

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

import pyarrow.parquet as pq
import argparse as ap
from glob import glob
""" Train data summary:
Overall positive example rate in train:  0.005385006165857902

Protein: BRD4
Train binds:
 0    97958646
1      456964
Positive example rate in train:  0.004643206499456743

Protein: HSA
Train binds:
 0    98007200
1      408410
Name: binds, dtype: int64
Positive example rate in train:  0.004149849805330679

Protein: sEH
Train binds:
 0    97691078
1      724532
Name: binds, dtype: int64
Positive example rate in train:  0.007361962192786287
"""
parser = ap.ArgumentParser()
parser.add_argument('-model_name', type=str, default='NN4', help='Model name: NN/NN4/NN5')
parser.add_argument('-checkpoint', type=str, help='load model from checkpoint', default=None)
parser.add_argument('-load_model', type=str, help='load model from file', default=None)
parser.add_argument('-model_index', type=str, help='model index', default='00')
parser.add_argument('-fp_size', type=int, help='fingerprint size', default=1024)
parser.add_argument('-fp_invar_size', type=int, help='feature-based invariant fingerprint size', default=0)
parser.add_argument('-debug', action='store_true', help='load small sample of data for testing')
parser.add_argument('-gpu', action='store_true', help='use cuda if available')
parser.add_argument('-split_name', type=str, help='split name', default=None)
parser.add_argument('-fold', type=int, help='number of k-folds', default=5)
parser.add_argument('-process_train', action='store_true', help = 'process train data')
parser.add_argument('-process_test', action='store_true', help = 'process test data')
parser.add_argument('-protein', type=str, help='protein name, if provided only do prediction on 1 protein', default=None)
parser.add_argument('-skip_train', action='store_true', help='skip training model')
parser.add_argument('-train_path', type=str, default='../data/split2/train_folds.parquet')

# input to create split
parser.add_argument('-get_split', action='store_true', help='generate new split')
parser.add_argument('-n_samples', type=int, help='number of train data points', default=5e6)
parser.add_argument('-posrate', type=int, help='positive example rate, n_pos/n_samples', default=0.05)
args = parser.parse_args()

print('Debug mode: ', args.debug)

if args.debug:
    # sample small set for testing
    train_data_path = '../data/split1/train_folds_small-10000.parquet'
else:
    train_data_path = args.train_path
    
if not args.split_name: 
    if args.get_split:
        split_name = 'new_split'
        generate_split = True
    elif not args.get_split:
        split_name = 'all'
        generate_split = False
else:
    split_name = args.split_name
    generate_split = False

print('Split name: ', split_name)

load_processed_train_data = not(args.process_train) # if available, load processed data from pickle or parquet file with fingerprints for faster processing
train_processed = sorted(glob('processed_data/train_{}_graph_{}bits_invar{}bits_chunk*.pkl'.format(split_name,args.fp_size, args.fp_invar_size)))
                          
load_processed_test_data = not(args.process_test)
test_processed = sorted(glob('processed_data/test_graph_{}bits_invar{}bits_chunk*.pkl'.format(args.fp_size, args.fp_invar_size)))

# model parameters
model_name = args.model_name
model_index = args.model_index
checkpoint_interval = 5  # Save a checkpoint every 5 epochs
hyperparameters = { 
    'n_epochs': 500,
    'resampling': False,
    'reweighting': False, # add weight based on class imbalance
    'dropout_rate': 0.3,
    'learning_rate': 0.0001,
    'weight_decay': 0.001,
    'patience': 10  # Number of epochs to wait for improvement before stopping, +5 for first fold to explore more
}


# Load train data, get splits/generate splits, then calculate features
if not load_processed_train_data or generate_split:
    time0 = time()
    print('\nLoading train data from {}'.format(train_data_path))
    if train_data_path.endswith('.parquet'):
        train_data = pq.read_table(train_data_path).to_pandas()
    elif train_data_path.endswith('.pkl'):
        train_data = pd.read_pickle(train_data_path)
    elif train_data_path.endswith('.csv'):
        train_data = pd.read_csv(train_data_path)
    print('Time elapsed (min): ', round((time()-time0)/60, 2))

    print('Columns: ', train_data.columns)

    if not generate_split:
        print('\nSplit name: ', split_name)
        if not split_name == 'all':
            train_data = train_data[train_data[split_name] == True]
            print('\nStatistics in train_data[{} == True]:'.format(split_name))
        else:
            split_name = os.path.basename(train_data_path).split('.')[:-1]
            split_name = '.'.join(split_name)
            split_name = split_name.split('train_')[1]
            print('New split name inferred from train_data_path: ', split_name)
            print('\nStatistics in train_data:'.format(split_name))

        train_data = shuffle(train_data)
        
        print('Total count: {}'.format(train_data.shape[0]))
        try:
            print('Positive rates: {}'.format(train_data['binds'].mean()))
        except:
            print('Positive rates:\n{}'.format(train_data[['binds_BRD4','binds_HSA','binds_sEH']].mean()))
    
    else:
        if args.protein == 'BRD4':
            n_pos = min(int(args.n_samples * args.posrate), 456964)
        elif args.protein == 'HSA':
            n_pos = min(int(args.n_samples * args.posrate), 408410)
        elif args.protein == 'sEH':
            n_pos = min(int(args.n_samples * args.posrate), 724532)
        posrate = n_pos / args.n_samples
        
        split_name = '{}_posrate-{:.3f}_size-{}M'.format(args.protein,posrate,args.n_samples/1e6)
        print('\nGenerating split for protein {}, positive rate = {}, total number of train data = {}'.format(args.protein, args.posrate, args.n_samples))
        
        # create new split column in original data set
        train_data[split_name] = False
        train_data_short = pd.DataFrame()
        
        try:
            pos_data = train_data[train_data['protein_name']==args.protein][train_data['binds']==1].sample(n=n_pos)
        except:
            pos_data = train_data[train_data['binds_{}'.format(args.protein)]==1].sample(n=n_pos)
        train_data.loc[pos_data.index, split_name] = True
        
        try:
            neg_data = train_data[train_data['protein_name']==args.protein][train_data['binds']==0].sample(n=int(args.n_samples - n_pos))
        except:
            neg_data = train_data[train_data['binds_{}'.format(args.protein)]==0].sample(n=int(args.n_samples - n_pos))
        train_data.loc[neg_data.index, split_name] = True

        train_data_short = pd.concat([train_data_short, pos_data])
        train_data_short = pd.concat([train_data_short, neg_data])

        # check positive example rate in train_data[split_name] and train_data_short
        print('\nStatistics in train_data[{} == True]:'.format(split_name))
        print('Total count: {}'.format(train_data[train_data[split_name] == True].shape[0]))
        try:
            print('Positive rates: {}'.format(train_data[train_data[split_name] == True]['binds'].mean()))
        except:
            print('Positive rates:\n{}'.format(train_data[train_data[split_name] == True][['binds_BRD4','binds_HSA','binds_sEH']].mean()))

        print('\nStatistics in train_data_short (copy of train_data[{} == True]):'.format(split_name))
        print('Total count: {}'.format(train_data_short.shape[0]))
        try:
            print('Positive rates: {}'.format(train_data_short['binds'].mean()))
        except:
            print('Positive rates:\n{}'.format(train_data_short[['binds_BRD4','binds_HSA','binds_sEH']].mean()))
        
        # save
        train_data.to_parquet('train_folds.parquet')
        with open('train_folds_columns.txt', 'w') as f:
            for item in train_data.columns:
                f.write("%s\n" % item)

        # check
        train_data = train_data[train_data[split_name] == True].sort_values(by='id')
        train_data_short = train_data_short.sort_values(by='id')

        train_data = shuffle(train_data_short)
        train_data.to_parquet('train_{}.parquet'.format(split_name))

        del train_data_short
        gc.collect()

    # calculate finger prints for molecules
    chunk_size = int(1e6)
    time0 = time()

    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')

    print('\nCalculating fingerprints for train data')
    chunk_id = 1
    train_processed = []
    for chunk in range(0, len(train_data), chunk_size):
        print('Processing chunk: {}/{}'.format(chunk_id, len(range(0, len(train_data), chunk_size))))
        train_data_chunk = train_data.iloc[chunk:chunk+chunk_size]
        
        batch_size = 1000 # number of molecules to process at a time
        train_fingerprints = parallel_process(train_data_chunk['molecule_smiles'], partial(batch_smiles_to_fp, fp_size=args.fp_size, fp_invariant_size=args.fp_invar_size), batch_size, n_jobs=12)
        train_fingerprints = [item for sublist in train_fingerprints for item in sublist]  # Flatten the list of lists
        train_data_chunk['molecule_fp'] = train_fingerprints

        batch_size = 1000
        train_graphs = parallel_process(train_data_chunk['molecule_smiles'], batch_smiles_to_graph, batch_size, n_jobs=12)
        train_graphs = [pickle.loads(item) for sublist in train_graphs for item in sublist]
        train_data_chunk['molecule_graph'] = train_graphs
        
        try:
            train_data_chunk = train_data_chunk[['id','molecule_fp','molecule_graph','protein_name','binds']]
        except:
            train_data_chunk = train_data_chunk[['molecule_fp','molecule_graph','binds_BRD4','binds_HSA','binds_sEH']]
            
        # save to file
        file = 'processed_data/train_{}_graph_{}bits_invar{}bits_chunk{}.pkl'.format(split_name,args.fp_size,args.fp_invar_size,chunk_id)
        
        #train_data_chunk.to_parquet(file)
        train_data_chunk.to_pickle(file)
        train_processed.append(file)
        
        del train_fingerprints, train_data_chunk
        gc.collect()
        
        chunk_id += 1
    
    del train_data
    gc.collect()
    print('Time elapsed (min): ', round((time()-time0)/60, 2))

check_memory(pid=os.getpid())

if not load_processed_test_data:
    test_processed = []
    print('\nLoading test data')
    test_data = pq.read_table('../data/test.parquet').to_pandas()

    print('Protein counts: ')
    print(test_data['protein_name'].value_counts())

    # add columns in test data
    test_data['is_BRD4'] = test_data['protein_name'] == 'BRD4'
    test_data['is_HSA'] = test_data['protein_name'] == 'HSA'
    test_data['is_sEH'] = test_data['protein_name'] == 'sEH'

    time0 = time()

    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')

    print('\nCalculating fingerprints for test data')
    chunk_size = int(5e5)

    chunk_id = 1
    test_processed = []
    for chunk in range(0, len(test_data), chunk_size):
        print('Processing chunk: {}/{}'.format(chunk_id, len(range(0, len(test_data), chunk_size))))
        test_data_chunk = test_data.iloc[chunk:chunk+chunk_size]
        fingerprints = []
        
        batch_size = 1000 # number of molecules to process at a time
        test_fingerprints = parallel_process(test_data_chunk['molecule_smiles'], partial(batch_smiles_to_fp, fp_size=args.fp_size, fp_invariant_size=args.fp_invar_size), batch_size, n_jobs=12)
        test_fingerprints = [item for sublist in test_fingerprints for item in sublist]  # Flatten the list of lists
        test_data_chunk['molecule_fp'] = test_fingerprints

        batch_size = 1000
        test_graphs = parallel_process(test_data_chunk['molecule_smiles'], batch_smiles_to_graph, batch_size, n_jobs=12)
        test_graphs = [pickle.loads(item) for sublist in test_graphs for item in sublist]
        test_data_chunk['molecule_graph'] = test_graphs

        test_data_chunk = test_data_chunk[['id','molecule_fp','molecule_graph','is_BRD4','is_HSA','is_sEH']]
        
        # save to file
        file = 'processed_data/test_graph_{}bits_invar{}bits_chunk{}.pkl'.format(args.fp_size,args.fp_invar_size,chunk_id)
        
        test_data_chunk.to_pickle(file)
        test_processed.append(file)

        del test_fingerprints, test_data_chunk
        gc.collect()
        
        chunk_id += 1

    del test_data
    gc.collect()
    print('Time elapsed (min): ', round((time()-time0)/60, 2))

if args.skip_train:
    print('Skip training model. Done!')
    sys.exit()

# Load processed data with fingerprints with only necessary columns
# print('\nLoading processed train data from pickle/parquet files')
# train_data = []

# if args.debug:
#     train_processed = ['train_small-1000_1024bits.parquet']

# for i,file in enumerate(train_processed):
#     print(file)
#     if file.endswith('.pkl'):
#         df = pd.read_pickle(file)

#     elif file.endswith('.parquet'):
#         try:
#             df = pq.read_table(file, 
#                         columns=['molecule_fp','binds_BRD4','binds_HSA','binds_sEH']).to_pandas()
#         except:
#             df = pq.read_table(file, columns=['molecule_fp', 'protein_name', 'binds']).to_pandas()
    
#     # add columns if missing
#     if 'binds_BRD4' not in df.columns:
#         df['binds_BRD4'] = (df['protein_name'] == 'BRD4') & df['binds']
#         df['binds_HSA'] = (df['protein_name'] == 'HSA') & df['binds']
#         df['binds_sEH'] = (df['protein_name'] == 'sEH') & df['binds']
            
#     if i == 0:
#         if args.model_name in ['GCN']:
#             X_train = list(df['molecule_graph'])
#         elif args.model_name in ['FPGCN']:
#             X_train_fp, X_train_graph = np.array(list(df['molecule_fp'])), list(df['molecule_graph'])
#         else:
#             X_train = np.array(list(df['molecule_fp']))
#         y_train = df[['binds_BRD4', 'binds_HSA', 'binds_sEH']].values
#     else:
#         if args.model_name in ['GCN']:
#             X_train.extend(list(df['molecule_graph']))
#         elif args.model_name in ['FPGCN']:
#             X_train_fp = np.append(X_train_fp, np.array(list(df['molecule_fp'])), axis=0)
#             X_train_graph.extend(list(df['molecule_graph']))
#         else:
#             X_train = np.append(X_train, np.array(list(df['molecule_fp'])), axis=0)
#         y_train = np.append(y_train, df[['binds_BRD4', 'binds_HSA', 'binds_sEH']].values, axis=0)

#     # free memory
#     del df
#     gc.collect()
    
#     sleep(5)
#     check_memory(pid=os.getpid())

# if args.protein:
#     y_train = y_train[:,['BRD4','HSA','sEH'].index(args.protein)]
#     y_train = y_train.reshape(-1,1)

# # shuffle samples, important!
# idx = np.random.permutation(y_train.shape[0])
# y_train = y_train[idx]

# if args.model_name in ['GCN']:
#     X_train = [X_train[i] for i in idx]
#     print('\nX_train length: ', len(X_train))
# elif args.model_name in ['FPGCN']:
#     X_train_fp = X_train_fp[idx]
#     X_train_graph = [X_train_graph[i] for i in idx]
#     X_train = [X_train_fp, X_train_graph]
#     print('\nX_train_fp shape: ', X_train_fp.shape)
#     print('X_train_graph length: ', len(X_train_graph))
# else:
#     X_train = X_train[idx]
#     print('\nX_train shape: ', X_train.shape)

# print('y_train shape: ', y_train.shape)

# print('\nPositive example rate in train:')
# if args.protein:
#     print('  {}: '.format(args.protein), np.mean(y_train))
# else:
#     print('  BRD4: ', np.mean(y_train[:,0]))
#     print('  HSA: ', np.mean(y_train[:,1]))
#     print('  sEH: ', np.mean(y_train[:,2]))

check_memory(pid=os.getpid())

# prepare for training
# generate hyperparameter from the dictionary with variable name = key and value = value
for key, value in hyperparameters.items():
    exec(key + '= value')

if args.debug:
    n_epochs = 1
    checkpoint_interval = 1

# check CUDA
pid = psutil.Process()
if args.gpu:
    print('Is CUDA available: ',torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
    
print('Device: ', device)

# set up directory to save checkpoints
if args.protein:
    prefix = '{}_{}_{}_{}bits_invar{}bits_resampling-{}_reweighting-{}'.format(split_name, model_name, model_index, args.fp_size, args.fp_invar_size, resampling, reweighting)
    n_classes = 1
else:
    prefix = '{}_{}_{}_{}bits_invar{}bits_resampling-{}_reweighting-{}'.format(split_name, model_name, model_index, args.fp_size, args.fp_invar_size, resampling, reweighting)
    n_classes = 3

checkpoint_dir = 'checkpoints_' + prefix  # Directory to save checkpoints
print('\nCheckpoint directory: ', checkpoint_dir)

# Ensure the checkpoint directory exists
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize the model, loss function, and optimizer
model_dict = {'NN': Net(n_features=args.fp_size+args.fp_invar_size, dropout_rate=dropout_rate, n_classes=n_classes),
              'NN4': Net4(n_features=args.fp_size+args.fp_invar_size, dropout_rate=dropout_rate, n_classes=n_classes),
              'NN4b': Net4b(n_features=args.fp_size+args.fp_invar_size, dropout_rate=dropout_rate, n_classes=n_classes),
              'NN5': Net5(n_features=args.fp_size+args.fp_invar_size, dropout_rate=dropout_rate, n_classes=n_classes),
              'NN5b': Net5b(n_features=args.fp_size+args.fp_invar_size, dropout_rate=dropout_rate, n_classes=n_classes),
              'CNN1d': CNN1d(dropout_rate=dropout_rate, n_classes=n_classes),
              'CNN1d_b': CNN1d_b(dropout_rate=dropout_rate, n_classes=n_classes),
              'GCN': GCN(dropout_rate=dropout_rate, n_classes=n_classes),
              'FPGCN': FPGCN(dropout_rate=dropout_rate, n_classes=n_classes, 
                            fc_layer_nodes=[args.fp_size+args.fp_invar_size, 512], gnn_layer_nodes=[20,64,64], 
                            gnn_n_out=128, fc_combined_layer_nodes=[1024,512],
                            device=device),}
# save hyperparameters to file
with open(checkpoint_dir+'/hyperparameters.txt', 'w') as file:
    file.write(str(hyperparameters))
    file.write('\n')
    file.write(str(model_dict[model_name]))

if args.load_model:
    model = torch.load(args.load_model, map_location=device)
    print('Loaded model: ', args.load_model)
else:
    model = model_dict[model_name].to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.BCELoss()

if args.checkpoint:
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print('Model loaded from checkpoint: ', args.checkpoint)
    print('Resuming training from epoch: ', epoch+1)

# prepare dataset
load_all_data = True
print('\nPrepare dataset')
if args.model_name in ['GCN']:
    custom_dataset = CustomDataset(train_processed, use_fp=False, use_graph=True, load_all_data=load_all_data)
    collate_fn = collate_fn_graph
elif args.model_name in ['FPGCN']:
    custom_dataset = CustomDataset(train_processed, use_fp=True, use_graph=True, load_all_data=load_all_data)
    collate_fn = collate_fn_fpgraph
else:
    custom_dataset = CustomDataset(train_processed, use_fp=True, use_graph=False, load_all_data=load_all_data)
    collate_fn = collate_fn_fp
check_memory(pid=os.getpid())

# Loop through CV folds and train the model
kf = KFold(n_splits=args.fold, shuffle=True)
fold_index = 0
best_loss = float('inf') 
train_scores = []
val_scores = []
train_losses = []
val_losses = []

splits = kf.split(range(len(custom_dataset)))
# append another fold with all data for training and zero for validation into splits
cv_splits = list(splits)
cv_splits.append((range(len(custom_dataset)), []))

for fold_index, (train_index, val_index) in enumerate(cv_splits):
    print('\n=== Fold {}/{} ==='.format(fold_index+1, args.fold))
    print('train length: ', len(train_index))
    print('validation length: ', len(val_index))

    # Create a DataLoader for the training data
    if args.debug:
        batch_size = 32
    else:
        batch_size = 256

    # Create samplers
    train_sampler = SubsetRandomSampler(train_index)
    train_loader = DataLoader(custom_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn)
    if len(val_index):
        val_sampler = SubsetRandomSampler(val_index)
        val_loader = DataLoader(custom_dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=collate_fn)
    else:
        val_sampler = None
        val_loader = None

    if fold_index > 0:
        print('Load model from previous fold')
        model = torch.load(prev_model_path, map_location=device)
    
    if fold_index == 0:
        patience_fold = patience + 10
    else:
        patience_fold = patience

    model, train_loss, train_score, val_loss, val_score = train_model(model, train_loader, criterion, optimizer, device, n_epochs=n_epochs, 
                checkpoint_dir=checkpoint_dir, checkpoint_interval=checkpoint_interval, patience=patience_fold, 
                val_loader=val_loader, fold_index=fold_index)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_scores.append(train_score)
    val_scores.append(val_score)

    if len(val_index):
        if val_loss < best_loss:
            #prev_model = model
            prev_model_path = checkpoint_dir + '/model_fold-{}.pth'.format(fold_index)
            
    # free memory
    del train_loader, val_loader, train_sampler, val_sampler
    torch.cuda.empty_cache()
    sleep(20)

    # evaluate on test data using the last model (trained on all data)
    print('\nMaking predictions on test data...')
    print('Loading processed test data from pickle/parquet files')
    test_data = []
    y_test_pred = []
    model.eval()

    for file in test_processed:
        print(file)
        if file.endswith('.pkl'):
            df = pd.read_pickle(file)
        
        elif file.endswith('.parquet'):
            df = pq.read_table(file).to_pandas()
        
        # add columns if missing
        if 'is_BRD4' not in df.columns:
            df['is_BRD4'] = df['protein_name'] == 'BRD4'
            df['is_HSA'] = df['protein_name'] == 'HSA'
            df['is_sEH'] = df['protein_name'] == 'sEH'    

        test_data.append(df[['id','is_BRD4','is_HSA','is_sEH']])

        batch_size = 4096
        if args.model_name in ['GCN']:
            X_test_t = list(df['molecule_graph'])
            y_dummy = np.zeros((len(X_test_t)))
            test_data_ = [(graph, label) for graph, label in zip(X_test_t, y_dummy)]
            test_loader = DataLoaderGCN(GraphDataset(test_data_), batch_size=batch_size, shuffle=False, collate_fn=collate_fn_graph)
            del X_test_t, y_dummy
        elif args.model_name in ['FPGCN']:
            X_test_fp, X_test_graph = np.array(list(df['molecule_fp'])), list(df['molecule_graph'])
            y_dummy = np.zeros((len(X_test_fp)))
            test_data_ = [(fp, graph, label) for fp, graph, label in zip(X_test_fp, X_test_graph, y_dummy)]
            test_loader = DataLoader(MixedDataset(test_data_), batch_size=batch_size, shuffle=False, collate_fn=collate_fn_fpgraph)
            del X_test_fp, X_test_graph, y_dummy
        else:
            X_test_t = torch.tensor(np.array(list(df['molecule_fp'])), dtype=torch.float32)
            y_dummy = torch.tensor(np.zeros((X_test_t.shape[0])), dtype=torch.float32)
            test_loader = DataLoader(TensorDataset(X_test_t,y_dummy), batch_size=batch_size, shuffle=False)
            del X_test_t, y_dummy
        
        del df
        gc.collect()

        # evaluate on test data
        for inputs,_ in test_loader:
            #print('Batch: {}/{}'.format(i_batch+1, len(test_loader)))
            with torch.no_grad():
                if isinstance(inputs, list) and len(inputs) == 2:
                    inputs = [i.to(device) for i in inputs]
                else:
                    inputs = inputs.to(device)
                tmp = model(inputs)
                y_test_pred.append(tmp.cpu().numpy()) 
            del tmp
            gc.collect()

        del test_loader, test_data_
        gc.collect()
        check_memory(pid=os.getpid())

    y_test_pred = np.concatenate(y_test_pred, axis=0)
    print('y_test_pred shape:', y_test_pred.shape)
    test_data = pd.concat(test_data)
    print(test_data.head())

    if fold_index == 0:
        # only keep columns id, binds_BRD4, binds_HSA, binds_sEH in test_data
        test_data_ids = test_data[['id','is_BRD4','is_HSA','is_sEH']]
        test_data_ids.to_csv('test_ids.csv', index=False)
        print(test_data[['is_BRD4','is_HSA','is_sEH']].sum())
        del test_data_ids

    # free memory
    del test_data
    gc.collect()
    check_memory(pid=os.getpid())

    submission = pd.read_csv('test_ids.csv')
    
    if not args.protein:
        # Use columns is_BRD4, is_HSA, is_sEH as mask to the correct index in y_test_pred
        mask = submission[['is_BRD4', 'is_HSA', 'is_sEH']].values.astype(bool)
        y_test_pred_masked = y_test_pred[mask]
        submission['binds'] = y_test_pred_masked
    else:
        submission['binds'] = y_test_pred
        submission = submission[submission['is_{}'.format(args.protein)] == True]

    submission = submission[['id', 'binds']]

    # make save predictions to file
    submission.to_csv(checkpoint_dir + '/test_pred_fold-{}.csv'.format(fold_index), index=False)

    # histogram of the predictions
    plt.figure(figsize=(4,4))
    plt.hist(submission['binds'], bins=10)
    plt.xlabel('Prediction')
    plt.ylabel('Frequency')
    # make y log scale
    plt.yscale('log')
    plt.title('Histogram of predictions')
    plt.savefig('{}/histogram_test-prediction_fold-{}.png'.format(checkpoint_dir,fold_index), dpi=500, bbox_inches='tight')

    # free memory
    del submission, y_test_pred
    gc.collect()
    sleep(5)
    
# remove None values from val_scores and val_losses
val_scores = [x for x in val_scores if x is not None]
val_losses = [x for x in val_losses if x is not None]

# plot loss
plt.figure(figsize=(4, 4))
plt.plot(train_losses, 'o-', label='Train')
plt.plot(val_losses, 'o-', label='Validation')
plt.xlabel('Fold')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend(fontsize=6)
plt.savefig('{}/loss.png'.format(checkpoint_dir), dpi=500, bbox_inches='tight')

# plot score
plt.figure(figsize=(4, 4))
plt.plot(train_scores, 'o-', label='Train')
plt.plot(val_scores, 'o-', label='Validation')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.title('Score')
plt.legend(fontsize=6)
plt.savefig('{}/score.png'.format(checkpoint_dir), dpi=500, bbox_inches='tight')

print('\nFinished')
