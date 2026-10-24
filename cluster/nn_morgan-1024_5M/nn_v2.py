import pandas as pd
import numpy as np
from time import time
from tools import *
import gc
import psutil
from functools import partial

import torch
from torch import nn, optim
import torch.utils.data as data
from torch.utils.data import TensorDataset, DataLoader
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score

import matplotlib.pyplot as plt
import os, sys
from sklearn.utils import shuffle

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
parser.add_argument('protein_name', type=str, help='Protein name to train: BRD4, HSA, sEH')
parser.add_argument('-model_name', type=str, default='NN', help='Model name: NN/NN4/NN5')
parser.add_argument('-checkpoint', type=str, help='load model from checkpoint', default=None)
parser.add_argument('-model_index', type=int, help='model index', default=10)
parser.add_argument('-fp_size', type=int, help='fingerprint size', default=1024)
args = parser.parse_args()

check_memory(pid=os.getpid())

# protein name to train
protein_name = args.protein_name

# input data
generate_short_data = False # generate shorter data for testing
n_samples = int(10e6)
resample_pos_rate = 0.05 # positive rate in the short data
data_ext = '10M_posrate-0.005' # extension for the data file

load_processed_data = True # if available, load processed data from pickle or parquet file with fingerprints for faster processing
train_processed = 'train_{}_5000000_posrate-0.05_{}bits.parquet'.format(protein_name, args.fp_size)
test_processed = 'test_{}_{}bits.pkl'.format(protein_name, args.fp_size)

# model parameters
model_name = args.model_name
model_index = args.model_index

hyperparameters = { 
    'n_epochs': 1000,
    'resampling': False,
    'reweighting': False, # add weight based on class imbalance
    'dropout_rate': 0.3,
    'learning_rate': 0.0002,
    'weight_decay': 0.001,
    'patience': 10  # Number of epochs to wait for improvement before stopping
}
# generate hyperparameter from the dictionary with variable name = key and value = value
for key, value in hyperparameters.items():
    exec(key + '= value')
checkpoint_interval = 5  # Save a checkpoint every 5 epochs

# check CUDA
pid = psutil.Process()
print('Is CUDA available: ',torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load train and test data
if not load_processed_data:
    print('Loading train and test data')
    test_data = pd.read_csv('../data/test.csv')
    # load DataFrame from a pickle file
    train_data_original = pd.read_pickle('../data/train.pkl')
    # train_data_original =  pd.read_pickle('../data/train_short_9000.pkl')

    # filter data by protein name
    train_data_temp = train_data_original[train_data_original['protein_name'] == protein_name]
    print('Original train data shape for protein {}: '.format(protein_name), train_data_temp.shape)
    
    # free memory
    del train_data_original
    gc.collect()

    # create shorter train data of this protein for testing
    if generate_short_data:
        if protein_name == 'BRD4':
            n_pos = min(int(n_samples * resample_pos_rate), 456964)
        elif protein_name == 'HSA':
            n_pos = min(int(n_samples * resample_pos_rate), 408410)
        elif protein_name == 'sEH':
            n_pos = min(int(n_samples * resample_pos_rate), 724532)
        resample_pos_rate = n_pos / n_samples
        print('\nGenerating shorter train data for protein {}, positive rate = {}, total number of train data = {}'.format(protein_name, resample_pos_rate, n_samples))

        train_data_short = pd.DataFrame()
        
        pos_data = train_data_temp[train_data_temp['protein_name']==protein_name][train_data_temp['binds']==1].sample(n=n_pos)
        neg_data = train_data_temp[train_data_temp['protein_name']==protein_name][train_data_temp['binds']==0].sample(n=n_samples - n_pos)
        train_data_short = pd.concat([train_data_short, pos_data])
        train_data_short = pd.concat([train_data_short, neg_data])
        train_data_short = shuffle(train_data_short)
        train_data_temp = train_data_short
        train_data_temp.to_pickle('train_{}_{}.pkl'.format(protein_name,data_ext))
        print('Short train data shape for protein {}: '.format(protein_name), train_data_temp.shape)

        del train_data_short
        gc.collect()

    test_data = test_data[test_data['protein_name'] == protein_name]
    test_ids = test_data['id']
    # save test_ids to csv file
    test_ids.to_csv('test_ids_' + protein_name + '.csv', index=False)
    
    # calculate finger prints for molecules
    total_pos_rate = len(train_data_temp[train_data_temp['binds'] == 1]) / len(train_data_temp)
    batch_size = 1000 # number of molecules to process at a time
    time0 = time()
    print('\nCalculating fingerprints for train data')
    train_fingerprints = parallel_process(train_data_temp['molecule_smiles'], partial(batch_smiles_to_fp, fp_size=args.fp_size), batch_size, n_jobs=12)
    train_fingerprints = [item for sublist in train_fingerprints for item in sublist]  # Flatten the list of lists
    train_data_temp['molecule_fp'] = train_fingerprints
    del train_fingerprints
    gc.collect()
    # save to pickle file
    train_data_temp.to_pickle('train_{}_{}_posrate-{}_{}bits.pkl'.format(protein_name,len(train_data_temp),total_pos_rate,args.fp_size))
    train_data_temp.to_parquet('train_{}_{}_posrate-{}_{}bits.parquet'.format(protein_name,len(train_data_temp),total_pos_rate,args.fp_size))
    print('Time elapsed (min): ', round((time()-time0)/60, 2))

    time0 = time()
    print('\nCalculating fingerprints for test data')
    fingerprints = []
    test_fingerprints = parallel_process(test_data['molecule_smiles'], partial(batch_smiles_to_fp, fp_size=args.fp_size), batch_size, n_jobs=12)
    test_fingerprints = [item for sublist in test_fingerprints for item in sublist]  # Flatten the list of lists
    test_data['molecule_fp'] = test_fingerprints
    del test_fingerprints
    gc.collect()
    # save to pickle file
    test_data.to_pickle('test_' + protein_name + '_{}bits.pkl'.format(args.fp_size))
    test_data.to_parquet('test_' + protein_name + '_{}bits.parquet'.format(args.fp_size))
    print('Time elapsed (min): ', round((time()-time0)/60, 2))

else: 
    print('Loading processed data from pickle files')
    # load the data from pickle files and save to parquet files for faster processing
    # check extension of the file
    if train_processed.endswith('.pkl'):
        train_data_temp = pd.read_pickle(train_processed)
        train_parquet_morgan = train_processed.replace('.pkl','.parquet')
        if not os.path.exists(train_parquet_morgan):
            train_data_temp.to_parquet(train_parquet_morgan)
    elif train_processed.endswith('.parquet'):
        train_data_temp = pq.read_table(train_processed, columns=['id','binds']).to_pandas()
        train_parquet_morgan = train_processed

check_memory(pid=os.getpid())

# get id and binds column of train data
train_data_trunc = train_data_temp[['id','binds']]

 # free memory
del train_data_temp
gc.collect()

# split train data into train and validation
total_pos_rate = len(train_data_trunc[train_data_trunc['binds'] == 1]) / len(train_data_trunc)
val_pos_rate = 0
while val_pos_rate < 1e-5: # to ensure that validation data has positive examples
    train_data_trunc = shuffle(train_data_trunc)
    train_data_trunc = train_data_trunc.reset_index(drop=True)

    # split train data into train and validation
    train_data = train_data_trunc[:int(0.9*len(train_data_trunc))]
    validation_data = train_data_trunc[int(0.9*len(train_data_trunc)):]

    # print count of unique values in binds column
    val_pos_rate = len(validation_data[validation_data['binds'] == 1]) / len(validation_data)

# free memory
del train_data_trunc
gc.collect()

print('\nTrain size: ', len(train_data))
print('Validation size: ', len(validation_data))
print('\nTrain positive example rate:\n', len(train_data[train_data['binds'] == 1]) / len(train_data))
print('Validation positive example rate:\n', val_pos_rate)
print('Overall positive example rate in train + validation: ', total_pos_rate)

# set up directory to save checkpoints
prefix = '{}_{}_{}_trainsize-{}_posrate-{}_resampling-{}_reweighting-{}'.format(model_name, protein_name, model_index, len(train_data) + len(validation_data), round(total_pos_rate,4), resampling, reweighting)
checkpoint_dir = 'checkpoints_' + prefix  # Directory to save checkpoints

# Ensure the checkpoint directory exists
os.makedirs(checkpoint_dir, exist_ok=True)
# save hyperparameters to file
with open(checkpoint_dir+'/hyperparameters.txt', 'w') as file:
    file.write(str(hyperparameters))

# First load id as "feature" then get fingerprints on the fly during training to save memory
X_train = np.array(train_data['id'])
y_train =np.array(train_data['binds'])
# free memory
del train_data
gc.collect()

df_val = pq.read_table(train_parquet_morgan, filters=[('id', 'in', np.array(validation_data['id']))], columns=['molecule_fp','binds'])
df_val = df_val.to_pandas()
X_validation = list(df_val['molecule_fp'])
X_validation = np.array([x.tolist() for x in X_validation])
y_validation = np.array(df_val['binds'])

# free memory
del validation_data
del df_val
gc.collect()

if load_processed_data:
    test_data = pd.read_pickle(test_processed)

X_test = np.array(list(test_data['molecule_fp']))

# free memory
del test_data
gc.collect()

check_memory(pid=os.getpid())

# print('\nafter removing None values')
print('\nX_train shape (id column only, fingerprints are retrieved on the fly): ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_validation shape: ', X_validation.shape)
print('y_validation shape: ', y_validation.shape)
print('X_test shape: ', X_test.shape)

if resampling:
    # Apply SMOTE for oversampling the minority class
    smote = SMOTE(random_state=42)
    X_train_resample, y_train_resample = smote.fit_resample(X_train, y_train)

# Assuming X_train, y_train, X_test, y_test are numpy arrays
# Convert them to PyTorch tensors
if resampling:
    X_train_t = torch.tensor(X_train_resample, dtype=torch.int32)
    y_train_t = torch.tensor(y_train_resample, dtype=torch.float32)
else:
    X_train_t = torch.tensor(X_train, dtype=torch.int32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_validation_t = torch.tensor(X_validation, dtype=torch.float32)
y_validation_t = torch.tensor(y_validation, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

# send data to device
X_validation_t, y_validation_t = X_validation_t.to(device), y_validation_t.to(device)
X_test_t = X_test_t.to(device)

# Create a DataLoader for the training data
train_data = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)

# Initialize the model, loss function, and optimizer
model_dict = {'NN': Net(n_features=X_test.shape[1], dropout_rate=dropout_rate),
              'NN4': Net4(n_features=X_test.shape[1], dropout_rate=dropout_rate),
              'NN5': Net5(n_features=X_test.shape[1], dropout_rate=dropout_rate)}
model = model_dict[model_name].to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Calculate class weights
class_counts = np.bincount(y_train_t.numpy().astype(int))
class_weights = 1. / class_counts
print('\nclass weights: ', class_weights)
if reweighting:
    pos_weight = torch.tensor([class_weights[1]], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
else:
    criterion = nn.BCELoss()

# Early stopping parameters
best_loss = float('inf')  # Best loss value so far, initialized to infinity
epochs_no_improve = 0  # Number of epochs with no improvement, initialized to 0

train_losses = []
validation_losses = []
train_scores = []
validation_scores = []

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
    train_scores_epoch = []
    for train_ids, targets in train_loader:
        # get fingerprints on the fly from 'id' column
        train_ids = train_ids.numpy()
        df = pq.read_table(train_parquet_morgan, filters=[('id', 'in', train_ids.tolist())], columns=['binds','molecule_fp'])
        df = df.to_pandas()
        inputs = list(df['molecule_fp']) 
        inputs = np.array([x.tolist() for x in inputs])
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        
        targets = np.array(df['binds']) 
        targets = torch.tensor(targets, dtype=torch.float32).to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets.view(-1, 1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate on train set
        train_score = average_precision_score(targets.view(-1, 1).cpu(), outputs.view(-1, 1).detach().cpu())
        train_scores_epoch.append(train_score)

        # free memory
        del inputs
        del df
        gc.collect()

    current_train_score = np.mean(train_scores_epoch)
    train_scores.append(current_train_score)
    train_losses.append(loss.item())

    if (epoch+1) % 1 == 0:
        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}')
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

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        y_validation_pred = model(X_validation_t)
        validation_loss = criterion(y_validation_pred, y_validation_t.view(-1, 1))
        validation_losses.append(validation_loss.item())
        validation_score = average_precision_score(y_validation, y_validation_pred.cpu())
        validation_scores.append(validation_score)

    # Check for improvement
    if validation_loss < best_loss:
        best_loss = validation_loss
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

# Evaluate the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():    
    y_validation_pred = model(X_validation_t)
    validation_loss = criterion(y_validation_pred, y_validation_t.view(-1, 1))
    validation_score = average_precision_score(y_validation, y_validation_pred.cpu())
s+='Validation Loss: {}\n\n'.format(validation_loss.item())
s+='Train Score: {}\n'.format(current_train_score)
s+='Validation Score: {}\n'.format(validation_score)
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
    y_test_pred = y_test_pred.cpu().numpy().flatten()
test_ids = pd.read_csv('test_ids_' + protein_name + '.csv')
test_ids = test_ids.to_numpy().flatten()

# make a dataframe with id and binds columns
test_results = pd.DataFrame({'id': test_ids, 'binds': y_test_pred})
test_results.to_csv(checkpoint_dir + '/test_pred_{}.csv'.format(protein_name), index=False)

# plot loss
plt.figure(figsize=(4, 4))
plt.plot(train_losses, label='Train, score={:.2e}, loss={:.2e}'.format(train_score,loss.item()))
plt.plot(validation_losses, label='Validation, score={:.2e}, loss={:.2e}'.format(validation_score,validation_loss.item()))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend(fontsize=6)
plt.savefig('{}/loss.png'.format(checkpoint_dir), dpi=500, bbox_inches='tight')

# plot score
plt.figure(figsize=(4, 4))
plt.plot(train_scores, label='Train, score={:.2e}, loss={:.2e}'.format(train_score,loss.item()))
plt.plot(validation_scores, label='Validation, score={:.2e}, loss={:.2e}'.format(validation_score,validation_loss.item()))
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Training Score')
plt.legend(fontsize=6)
plt.savefig('{}/score.png'.format(checkpoint_dir), dpi=500, bbox_inches='tight')
