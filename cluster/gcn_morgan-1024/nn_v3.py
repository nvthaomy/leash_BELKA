import pandas as pd
import numpy as np
from time import time, sleep
from tools import *
import gc
import psutil

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import average_precision_score

import torch
from torch import nn, optim
import torch.utils.data as data
from torch.utils.data import TensorDataset, DataLoader
from imblearn.over_sampling import SMOTE

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
parser.add_argument('-model_name', type=str, default='NN', help='Model name: NN/NN4/NN5')
parser.add_argument('-checkpoint', type=str, help='load model from checkpoint', default=None)
parser.add_argument('-load_model', type=str, help='load model from file', default=None)
parser.add_argument('-model_index', type=str, help='model index', default='10')
parser.add_argument('-fp_size', type=int, help='fingerprint size', default=1024)
parser.add_argument('-fp_invar_size', type=int, help='feature-based invariant fingerprint size', default=0)
parser.add_argument('-debug', action='store_true', help='load small sample of data for testing')
parser.add_argument('-gpu', action='store_true', help='use cuda if available')
parser.add_argument('-split_name', type=str, help='split name', default='posrate-0.050_size-0.005M')
parser.add_argument('-fold', type=int, help='number of k-folds', default=5)
parser.add_argument('-train_split', type=float, help='train split', default=0.8)
parser.add_argument('-process_train', action='store_true', help = 'process train data')
parser.add_argument('-process_test', action='store_true', help = 'process test data')
parser.add_argument('-protein', type=str, help='protein name, if provided only do prediction on 1 protein', default=None)
args = parser.parse_args()

print('Debug mode: ', args.debug)
split_name = args.split_name

print('Split name: ', split_name)

load_processed_train_data = not(args.process_train) # if available, load processed data from pickle or parquet file with fingerprints for faster processing
# train_processed = sorted(glob('../nn_morgan-1024_5M/train_{}_1024bits.parquet'.format(split_name)))
train_processed = sorted(glob('train_{}_{}bits_invar{}bits_chunk*.parquet'.format(split_name,args.fp_size, args.fp_invar_size)))

load_processed_test_data = not(args.process_test)
# test_processed = ['../split1_separate-train_nn_morgan-1024/test_1024bits_chunk1.parquet',
#                  '../split1_separate-train_nn_morgan-1024/test_1024bits_chunk2.parquet',
#                  '../split1_separate-train_nn_morgan-1024/test_1024bits_chunk3.parquet']

# test_processed = sorted(glob('test_{}_{}bits_invar{}bits_chunk*.parquet'.format(args.protein, args.fp_size, args.fp_invar_size)))
test_processed = sorted(glob('../split1_separate-train_nn_morgan-1024/test_1024bits_chunk*.parquet'))

# model parameters
model_name = args.model_name
model_index = args.model_index
checkpoint_interval = 5  # Save a checkpoint every 5 epochs
hyperparameters = { 
    'n_epochs': 1000,
    'resampling': False,
    'reweighting': False, # add weight based on class imbalance
    'dropout_rate': 0.3,
    'learning_rate': 0.0001,
    'weight_decay': 0.001,
    'patience': 20  # Number of epochs to wait for improvement before stopping
}

# load train and test data
if not load_processed_train_data:
    # load DataFrame from a parquet file
    print('\nLoading train data')
    if args.debug:
        # sample small set for testing
        train_data = pq.read_table('../data/split1/train_folds_small-10000.parquet').to_pandas()
    else:
        train_data = pq.read_table('../data/random_splits/train_folds.parquet').to_pandas() 
    
    print('Columns: ', train_data.columns)
    print('Length of train data in split:', len(train_data[train_data[split_name] == True]) )

    train_data = train_data[train_data[split_name] == True]
    
    # calculate finger prints for molecules
    chunk_size = int(1e6)
    batch_size = 1000 # number of molecules to process at a time
    time0 = time()
    print('\nCalculating fingerprints for train data')

    chunk_id = 1
    train_processed = []
    for chunk in range(0, len(train_data), chunk_size):
        print('Processing chunk: {}/{}'.format(chunk_id, len(range(0, len(train_data), chunk_size))))
        train_data_chunk = train_data.iloc[chunk:chunk+chunk_size]
        
        train_fingerprints = parallel_process(train_data_chunk['molecule_smiles'], partial(batch_smiles_to_fp, fp_size=args.fp_size, fp_invariant_size=args.fp_invar_size), batch_size, n_jobs=12)
        train_fingerprints = [item for sublist in train_fingerprints for item in sublist]  # Flatten the list of lists
        train_data_chunk['molecule_fp'] = train_fingerprints
        try:
            train_data_chunk = train_data_chunk[['molecule_fp','protein_name','binds']]
        except:
            train_data_chunk = train_data_chunk[['molecule_fp','binds_BRD4','binds_HSA','binds_sEH']]
            
        # save to file
        file = 'train_{}_{}bits_invar{}bits_chunk{}.parquet'.format(split_name,args.fp_size,args.fp_invar_size,chunk_id)
        train_data_chunk.to_parquet(file)
        train_processed.append(file)

        del train_fingerprints, train_data_chunk
        gc.collect()
        
        chunk_id += 1
    
    del train_data
    gc.collect()
    print('Time elapsed (min): ', round((time()-time0)/60, 2))

check_memory(pid=os.getpid())

# Load processed data with fingerprints with only necessary columns
print('\nLoading processed train data from pickle/parquet files')
train_data = []

if args.debug:
    train_processed = ['train_small-1000_1024bits.parquet']

for i,file in enumerate(train_processed):
    print(file)
    if file.endswith('.pkl'):
        df = pd.read_pickle(file)
        if not os.path.exists(file.replace('.pkl','.parquet')):
            df.to_parquet(file.replace('.pkl','.parquet'))
    
    elif file.endswith('.parquet'):
        try:
            df = pq.read_table(file, 
                        columns=['molecule_fp','binds_BRD4','binds_HSA','binds_sEH']).to_pandas()
        except:
            df = pq.read_table(file, columns=['molecule_fp', 'protein_name', 'binds']).to_pandas()
    
    # add columns if missing
    if 'binds_BRD4' not in df.columns:
        df['binds_BRD4'] = (df['protein_name'] == 'BRD4') & df['binds']
        df['binds_HSA'] = (df['protein_name'] == 'HSA') & df['binds']
        df['binds_sEH'] = (df['protein_name'] == 'sEH') & df['binds']
            
    if i == 0:
        X_train = np.array(list(df['molecule_fp']))
        y_train = df[['binds_BRD4', 'binds_HSA', 'binds_sEH']].values
    else:
        X_train = np.append(X_train, np.array(list(df['molecule_fp'])), axis=0)
        y_train = np.append(y_train, df[['binds_BRD4', 'binds_HSA', 'binds_sEH']].values, axis=0)

    # free memory
    del df
    gc.collect()
    
    sleep(5)
    check_memory(pid=os.getpid())

if args.protein:
    y_train = y_train[:,['BRD4','HSA','sEH'].index(args.protein)]
    y_train = y_train.reshape(-1,1)

print('\nTrain data length:', len(y_train))
print('Positive example rate in train:')
if args.protein:
    print('  {}: '.format(args.protein), np.mean(y_train))
else:
    print('  BRD4: ', np.mean(y_train[:,0]))
    print('  HSA: ', np.mean(y_train[:,1]))
    print('  sEH: ', np.mean(y_train[:,2]))

check_memory(pid=os.getpid())

print('\nX_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)

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
    print('\nCalculating fingerprints for test data')
    chunk_size = int(5e5)
    batch_size = 1000

    chunk_id = 1
    test_processed = []
    for chunk in range(0, len(test_data), chunk_size):
        print('Processing chunk: {}/{}'.format(chunk_id, len(range(0, len(test_data), chunk_size))))
        test_data_chunk = test_data.iloc[chunk:chunk+chunk_size]
        fingerprints = []
        test_fingerprints = parallel_process(test_data_chunk['molecule_smiles'], partial(batch_smiles_to_fp, fp_size=args.fp_size, fp_invariant_size=args.fp_invar_size), batch_size, n_jobs=12)
        test_fingerprints = [item for sublist in test_fingerprints for item in sublist]  # Flatten the list of lists
        test_data_chunk['molecule_fp'] = test_fingerprints
        test_data_chunk = test_data_chunk[['id','molecule_fp','is_BRD4','is_HSA','is_sEH']]
        
        # save to file
        file = 'test_{}bits_invar{}bits_chunk{}.parquet'.format(args.fp_size,args.fp_invar_size,chunk_id)
        
        test_data_chunk.to_parquet(file)
        test_processed.append(file)

        del test_fingerprints, test_data_chunk
        gc.collect()
        
        chunk_id += 1

    del test_data
    gc.collect()
    print('Time elapsed (min): ', round((time()-time0)/60, 2))

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
# save hyperparameters to file
with open(checkpoint_dir+'/hyperparameters.txt', 'w') as file:
    file.write(str(hyperparameters))

# Initialize the model, loss function, and optimizer
model_dict = {'NN': Net(n_features=args.fp_size+args.fp_invar_size, dropout_rate=dropout_rate, n_classes=n_classes),
              'NN4': Net4(n_features=args.fp_size+args.fp_invar_size, dropout_rate=dropout_rate, n_classes=n_classes),
              'NN4b': Net4b(n_features=args.fp_size+args.fp_invar_size, dropout_rate=dropout_rate, n_classes=n_classes),
              'NN5': Net5(n_features=args.fp_size+args.fp_invar_size, dropout_rate=dropout_rate, n_classes=n_classes),
              'NN5b': Net5b(n_features=args.fp_size+args.fp_invar_size, dropout_rate=dropout_rate, n_classes=n_classes),
              'CNN1d': CNN1d(dropout_rate=dropout_rate, n_classes=n_classes),
              'CNN1d_b': CNN1d_b(dropout_rate=dropout_rate, n_classes=n_classes)
              }

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

# Loop through CV folds and train the model
kf = KFold(n_splits=args.fold, shuffle=True)
fold_index = 0
best_loss = float('inf') 
train_scores = []
val_scores = []
train_losses = []
val_losses = []

splits = kf.split(X_train)
# append another fold with all data for training and zero for validation into splits
cv_splits = list(splits)
cv_splits.append((np.arange(len(X_train)), []))

for train_index, val_index in cv_splits:
    print('\n=== Fold {}/{} ==='.format(fold_index+1, args.fold))
    # check if val_index is empty
    if not len(val_index) == 0:
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    else:
        X_val_fold, y_val_fold = [], []
        X_train_fold = X_train
        y_train_fold = y_train

    # Prepare train and val data
    if resampling:
        target_class_index = 1 # 0: binds_BRD4, 1: binds_HSA, 2: binds_sEH
        target_positive_rate = 0.05
        n_majority = np.sum(y_train[:,target_class_index] == 0)
        sampling_strategy = target_positive_rate * y_train.shape[1] / n_majority
        # Apply SMOTE for oversampling the minority class
        smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
        X_train_fold_resample, y_train_fold_resample = smote.fit_resample(X_train_fold, y_train_fold)
        del X_train_fold, y_train_fold
        gc.collect()

    # Assuming X_train, y_train, X_test, y_test are numpy arrays
    # Convert them to PyTorch tensors
    if resampling:
        X_train_t = torch.tensor(X_train_fold_resample, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train_fold_resample, dtype=torch.float32).to(device)
        del X_train_fold_resample, y_train_fold_resample
        gc.collect()
    else:
        X_train_t = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train_fold, dtype=torch.float32).to(device)
        del X_train_fold, y_train_fold
        gc.collect()
    
    if len(X_val_fold) > 0:
        X_val_t = torch.tensor(X_val_fold, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val_fold, dtype=torch.float32).to(device)
        del X_val_fold, y_val_fold
        gc.collect()
    else: 
        X_val_t, y_val_t = None, None

    # Create a DataLoader for the training data
    train_data = TensorDataset(X_train_t, y_train_t)
    if args.debug:
        train_loader = DataLoader(train_data, batch_size=32, shuffle=False)
    else:
        train_loader = DataLoader(train_data, batch_size=256, shuffle=False)

    if fold_index > 0:
        print('Load model from previous fold')
        model = torch.load(prev_model_path, map_location=device)
        # model = prev_model
        # del prev_model
        # gc.collect()

    if fold_index == 0:
        print('X_train_t shape: ', X_train_t.shape)
        print('y_train_t shape: ', y_train_t.shape)
        if X_val_t is not None:
            print('X_val_t shape: ', X_val_t.shape)
            print('y_val_t shape: ', y_val_t.shape)

    model, train_loss, train_score, val_loss, val_score = train_model(model, train_loader, criterion, optimizer, device, n_epochs=n_epochs, 
                checkpoint_dir=checkpoint_dir, checkpoint_interval=checkpoint_interval, patience=patience, 
                y_train_t=y_train_t, y_val_t=y_val_t, X_val_t=X_val_t, fold_index=fold_index)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_scores.append(train_score)
    val_scores.append(val_score)

    if X_val_t is not None:
        if val_loss < best_loss:
                #prev_model = model
                prev_model_path = checkpoint_dir + '/model_fold-{}.pth'.format(fold_index)
            
    # free memory
    del X_train_t, y_train_t, X_val_t, y_val_t, train_loader
    gc.collect()
    torch.cuda.empty_cache()
    sleep(5)

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
            if not os.path.exists(file.replace('.pkl','.parquet')):
                df.to_parquet(file.replace('.pkl','.parquet'))
        
        elif file.endswith('.parquet'):
            df = pq.read_table(file).to_pandas()
        
        # add columns if missing
        if 'is_BRD4' not in df.columns:
            df['is_BRD4'] = df['protein_name'] == 'BRD4'
            df['is_HSA'] = df['protein_name'] == 'HSA'
            df['is_sEH'] = df['protein_name'] == 'sEH'    

        test_data.append(df[['id','is_BRD4','is_HSA','is_sEH']])
        X_test = np.array(list(df['molecule_fp']))
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    
        # free memory
        del df, X_test
        gc.collect()

        # evaluate on test data
        batch_size = 256
        for i in range(0, len(X_test_t), batch_size):
            with torch.no_grad():
                tmp = model(X_test_t[i:i+batch_size])
                y_test_pred.append(tmp.cpu().numpy()) 
            del tmp
            gc.collect()

        del X_test_t
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

    fold_index += 1
    
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
