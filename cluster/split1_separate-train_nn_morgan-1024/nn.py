import pandas as pd
import numpy as np
from time import time
import sys
#sys.path.append('../scripts/')
from tools import *
import gc
import psutil

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
parser.add_argument('-protein_name', type=str, default='BRD4', help='Protein name: BRD4/HSA/sEH')
parser.add_argument('-checkpoint', type=str, help='load model from checkpoint', default=None)
parser.add_argument('-load_model', type=str, help='load model from file', default=None)
parser.add_argument('-model_index', type=str, help='model index', default='10')
parser.add_argument('-fp_size', type=int, help='fingerprint size', default=1024)
parser.add_argument('-debug', action='store_true', help='load small sample of data for testing')
parser.add_argument('-gpu', action='store_true', help='use cuda if available')
parser.add_argument('-resampling', type=float, help='positive rate for resampling data', default=None)
args = parser.parse_args()

print('Debug mode: ', args.debug)
split_name = 'full_test'

load_processed_train_data = True # if available, load processed data from pickle or parquet file with fingerprints for faster processing
train_processed = ['../split1_nn_morgan-1024/train_full_test_1024bits_chunk1.parquet',
                   '../split1_nn_morgan-1024/train_full_test_1024bits_chunk2.parquet',
                   '../split1_nn_morgan-1024/train_full_test_1024bits_chunk3.parquet',
                   '../split1_nn_morgan-1024/train_full_test_1024bits_chunk4.parquet',
                   '../split1_nn_morgan-1024/train_full_test_1024bits_chunk5.parquet'
                   ]
load_processed_test_data = True
test_processed = ['test_1024bits_chunk1.parquet','test_1024bits_chunk2.parquet','test_1024bits_chunk3.parquet']

# model parameters
model_name = args.model_name
model_index = args.model_index
checkpoint_interval = 5  # Save a checkpoint every 5 epochs
if not args.resampling == None:
    resampling = True
else:
    resampling = False
hyperparameters = { 
    'n_epochs': 1000,
    'resampling': resampling, # resample data to balance the classes
    'reweighting': False, # add weight based on class imbalance
    'dropout_rate': 0.3,
    'learning_rate': 0.0001,
    'weight_decay': 0.001,
    'patience': 10  # Number of epochs to wait for improvement before stopping
}

# load train and test data
if not load_processed_train_data:
    # load DataFrame from a parquet file
    print('\nLoading train data')
    if args.debug:
        # sample small set for testing
        train_data = pq.read_table('../data/split1/train_folds_small-10000.parquet').to_pandas()
    else:
        train_data = pq.read_table('../data/split1/train_folds.parquet').to_pandas() 

    train_data = train_data[train_data[split_name] == True]

    # calculate finger prints for molecules
    chunk_size = int(3e6)
    batch_size = 1000 # number of molecules to process at a time
    time0 = time()
    print('\nCalculating fingerprints for train data')

    chunk_id = 1
    train_processed = []
    for chunk in range(0, len(train_data), chunk_size):
        print('Processing chunk: {}/{}'.format(chunk_id, len(range(0, len(train_data), chunk_size))))
        train_data_chunk = train_data.iloc[chunk:chunk+chunk_size]
        fingerprints = []
        train_fingerprints = parallel_process(train_data_chunk['molecule_smiles'], partial(batch_smiles_to_fp, fp_size=args.fp_size), batch_size, n_jobs=12)
        train_fingerprints = [item for sublist in train_fingerprints for item in sublist]  # Flatten the list of lists
        train_data_chunk['molecule_fp'] = train_fingerprints
        train_data_chunk = train_data_chunk[['molecule_fp','binds_BRD4','binds_HSA','binds_sEH']]
        
        # save to file
        file = 'train_{}_{}bits_chunk{}.parquet'.format(split_name,args.fp_size,chunk_id)
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
X_train = []
y_train = []

if args.debug:
    train_processed = ['../split1_nn_morgan-1024/train_small-1000_1024bits.parquet']
for file in train_processed:
    print(file)
    if file.endswith('.pkl'):
        df = pd.read_pickle(file)
        if not os.path.exists(file.replace('.pkl','.parquet')):
            df.to_parquet(file.replace('.pkl','.parquet'))
    
    elif file.endswith('.parquet'):
        df = pq.read_table(file, 
                        columns=['molecule_fp','binds_{}'.format(args.protein_name)]).to_pandas()
    
    X_train.append(np.array(list(df['molecule_fp'])))
    y_train.append(df[['binds_{}'.format(args.protein_name)]].values)

    # free memory
    del df
    gc.collect()

    check_memory(pid=os.getpid())

# remove flatten the first dimension of X_train and y_train
X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)

print('\nTrain data length:', len(y_train))
print('Positive example rate in train: ', np.mean(y_train))

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
    print('\nCalculating fingerprints for test data')
    chunk_size = int(560000)
    batch_size = 1000

    chunk_id = 1
    test_processed = []
    for chunk in range(0, len(test_data), chunk_size):
        print('Processing chunk: {}/{}'.format(chunk_id, len(range(0, len(test_data), chunk_size))))
        test_data_chunk = test_data.iloc[chunk:chunk+chunk_size]
        fingerprints = []
        test_fingerprints = parallel_process(test_data_chunk['molecule_smiles'], partial(batch_smiles_to_fp, fp_size=args.fp_size), batch_size, n_jobs=12)
        test_fingerprints = [item for sublist in test_fingerprints for item in sublist]  # Flatten the list of lists
        test_data_chunk['molecule_fp'] = test_fingerprints
        test_data_chunk = test_data_chunk[['id','molecule_fp','is_BRD4','is_HSA','is_sEH']]
        
        # save to file
        file = 'test_{}bits_chunk{}.parquet'.format(args.fp_size,chunk_id)
        test_data_chunk.to_parquet(file)
        test_processed.append(file)

        del test_fingerprints, test_data_chunk
        gc.collect()
        
        chunk_id += 1

    del test_data
    gc.collect()
    print('Time elapsed (min): ', round((time()-time0)/60, 2))

print('\nLoading processed test data from pickle/parquet files')
X_test = []
test_data = []
if args.debug:
    test_processed = test_processed[:1]
for file in test_processed:
    if file.endswith('.pkl'):
        df = pd.read_pickle(file)
        if not os.path.exists(file.replace('.pkl','.parquet')):
            df.to_parquet(file.replace('.pkl','.parquet'))
    
    elif file.endswith('.parquet'):
        try:
            df = pq.read_table(file, filters=[('protein_name', '==', args.protein_name)]).to_pandas()
        except:
            df = pq.read_table(file, filters=[('is_{}'.format(args.protein_name), '==', True)]).to_pandas()

    # only keep rows with protein_name = args.protein_name
    try:
        df = df[df['protein_name'] == args.protein_name]
    except:
        df = df[df['is_{}'.format(args.protein_name)] == True]   

    test_data.append(df[['id']])
    X_test.append(np.array(list(df['molecule_fp'])))
  
    # free memory
    del df
    gc.collect()

    check_memory(pid=os.getpid())

X_test = np.concatenate(X_test)
test_data = pd.concat(test_data)
print(test_data.head())

# only keep columns id, binds_BRD4, binds_HSA, binds_sEH in test_data
test_data_ids = test_data[['id']]
test_data_ids.to_csv('test_ids_{}.csv'.format(args.protein_name), index=False)

# free memory
del test_data, test_data_ids
gc.collect()

check_memory(pid=os.getpid())

print('\nX_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_test shape: ', X_test.shape)

# prepare for training
# generate hyperparameter from the dictionary with variable name = key and value = value
for key, value in hyperparameters.items():
    exec(key + '= value')

if args.debug:
    n_epochs = 1000
    checkpoint_interval = 5

# check CUDA
pid = psutil.Process()
if args.gpu:
    print('Is CUDA available: ',torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
    
print('Device: ', device)

# set up directory to save checkpoints
if resampling:
    prefix = '{}_{}_{}_{}bits_resampling-{}_reweighting-{}'.format(model_name, args.protein_name, model_index, args.fp_size, args.resampling, reweighting)
else:
    prefix = '{}_{}_{}_{}bits_resampling-{}_reweighting-{}'.format(model_name, args.protein_name, model_index, args.fp_size, resampling, reweighting)
checkpoint_dir = 'checkpoints_' + prefix  # Directory to save checkpoints

# Ensure the checkpoint directory exists
os.makedirs(checkpoint_dir, exist_ok=True)
print('Checkpoint directory: ', checkpoint_dir)
# save hyperparameters to file
with open(checkpoint_dir+'/hyperparameters.txt', 'w') as file:
    file.write(str(hyperparameters))

if resampling:
    target_positive_rate = args.resampling
    print('\nResampling data to balance the classes with positive rate: ', target_positive_rate)
    # Calculate the number of majority class samples
    n_majority = y_train.shape[0] - y_train.sum()
    sampling_strategy = target_positive_rate * y_train.shape[0] / n_majority
    print('Sampling strategy = n samples minority after resampling / n samples majority = {} '.format(sampling_strategy))
    print('Increase in minority class samples by: ', int(target_positive_rate * y_train.shape[0]) - y_train.sum())
    # Apply SMOTE for oversampling the minority class
    smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
    X_train_resample, y_train_resample = smote.fit_resample(X_train, y_train)
    del X_train, y_train
    gc.collect()

# Assuming X_train, y_train, X_test, y_test are numpy arrays
# Convert them to PyTorch tensors
if resampling:
    X_train_t = torch.tensor(X_train_resample, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_resample, dtype=torch.float32)
    del X_train_resample, y_train_resample
    gc.collect()
else:
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    del X_train, y_train
    gc.collect()

X_test_t = torch.tensor(X_test, dtype=torch.float32)
del X_test
gc.collect()

# send data to device
X_train_t, y_train_t = X_train_t.to(device), y_train_t.to(device)
X_test_t = X_test_t.to(device)

# Create a DataLoader for the training data
train_data = TensorDataset(X_train_t, y_train_t)
if args.debug:
    train_loader = DataLoader(train_data, batch_size=32, shuffle=False)
else:
    train_loader = DataLoader(train_data, batch_size=256, shuffle=False)

# Calculate class weights
if reweighting:
    # n_neg = y_train.shape[0] - y_train.sum()
    # n_pos = y_train.sum()
    pos_weight = torch.tensor([(y_train.shape[0] - y_train.sum()) / y_train.sum()], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_type = 'BCEWithLogitsLoss'
else:
    criterion = nn.BCELoss()
    loss_type = 'BCELoss'

# Initialize the model, loss function, and optimizer
model_dict = {'NN': Net(n_features=args.fp_size, dropout_rate=dropout_rate),
              'NN4': Net4(n_features=args.fp_size, dropout_rate=dropout_rate),
              'NN4b': Net4b(n_features=args.fp_size, dropout_rate=dropout_rate, loss_type=loss_type),
              'NN5': Net5(n_features=args.fp_size, dropout_rate=dropout_rate),
              'NN5b': Net5b(n_features=args.fp_size, dropout_rate=dropout_rate),}
if args.load_model:
    model = torch.load(args.load_model)
    print('\nLoaded model: ', args.load_model)
else:
    model = model_dict[model_name].to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Early stopping parameters
best_loss = float('inf')  # Best loss value so far, initialized to infinity
epochs_no_improve = 0  # Number of epochs with no improvement, initialized to 0

train_losses = []
train_scores = []

check_memory(pid=os.getpid())

# Train the model
print('\nTraining the model...')
if args.checkpoint:
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print('Model loaded from checkpoint: ', args.checkpoint)
    print('Resuming training from epoch: ', epoch+1)

time0 = time()
for epoch in range(n_epochs):  # 100 epochs
    y_train_pred = []
    for inputs, targets in train_loader:
        # Move data to the GPU if available
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        if loss_type == 'BCEWithLogitsLoss': # calculate loss on logits
            loss = criterion(outputs, torch.log(targets / (1 - targets)))
            y_train_pred.append(torch.sigmoid(outputs).detach().cpu().numpy())
        elif loss_type == 'BCELoss': # calculate loss on probabilities
            loss = criterion(outputs, targets)
            y_train_pred.append(outputs.detach().cpu().numpy())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_losses.append(loss.item())

    # Evaluate on train set
    y_train_pred = np.concatenate(y_train_pred)
    # print('loss\n', loss.item())
    # print('outputs\n', outputs)
    # print('outputs_sigmoid\n', torch.sigmoid(outputs).detach().cpu().numpy())
    # print('y_train_pred\n', y_train_pred)
    train_score = average_precision_score(y_train_t.cpu().flatten(), y_train_pred.flatten())
    train_scores.append(train_score)

    if (epoch+1) % 2 == 0:
        print(f'\nEpoch {epoch+1}/{n_epochs}, Loss: {loss.item()}, Train Score: {train_score}')
        check_memory(pid=os.getpid())
        print('Time elapsed (min): ', round((time()-time0)/60, 2))

    # Save a checkpoint
    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, checkpoint_path)
        #print(f'Checkpoint saved at {checkpoint_path}')

    # Check for improvement
    if train_losses[-1] < best_loss:
        best_loss = train_losses[-1]
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    # Early stopping
    if epochs_no_improve == patience:
        print('Early stopping!')
        break

    model.train()

print('Time elapsed (min): ', round((time()-time0)/60, 2))
s='Train Loss: {}\n'.format(loss.item())
s+='Train Score: {}\n'.format(train_scores[-1])
print(s)

with open(checkpoint_dir+'/results.txt', 'w') as file:
    file.write(s)

# save model to file
torch.save(model, '{}/model.pth'.format(checkpoint_dir))

# evaluate on test data
print('\nMaking predictions on test data...')
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_t)
    if loss_type == 'BCEWithLogitsLoss':
        y_test_pred = torch.sigmoid(y_test_pred).cpu().numpy()
    elif loss_type == 'BCELoss':
        y_test_pred = y_test_pred.cpu().numpy() # this is a (n_samples, 3) array

submission = pd.read_csv('test_ids_{}.csv'.format(args.protein_name))
submission['binds'] = y_test_pred
submission = submission[['id', 'binds']]

# make save predictions to file
submission.to_csv(checkpoint_dir + '/test_pred_{}.csv'.format(args.protein_name), index=False)

# histogram of the predictions
plt.figure(figsize=(4,4))
plt.hist(submission['binds'], bins=10)
plt.xlabel('Prediction')
plt.ylabel('Frequency')
# make y log scale
plt.yscale('log')
plt.title('Histogram of predictions')
plt.savefig('{}/histogram_test-prediction.png'.format(checkpoint_dir), dpi=500, bbox_inches='tight')

# plot loss
plt.figure(figsize=(4, 4))
plt.plot(train_losses, label='Train, score={:.2e}, loss={:.2e}'.format(train_score,loss.item()))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend(fontsize=6)
plt.savefig('{}/loss.png'.format(checkpoint_dir), dpi=500, bbox_inches='tight')

# plot score
plt.figure(figsize=(4, 4))
plt.plot(train_scores, label='Train, score={:.2e}, loss={:.2e}'.format(train_score,loss.item()))
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Training Score')
plt.legend(fontsize=6)
plt.savefig('{}/score.png'.format(checkpoint_dir), dpi=500, bbox_inches='tight')

print('\nFinished')
