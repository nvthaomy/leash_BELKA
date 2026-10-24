import pandas as pd
import numpy as np
from time import time
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

# protein name to train
protein_name = str(sys.argv[1]) # sEH HSA BRD4

# input data
load_processed_data = False # if available, load processed data from pickle file with fingerprints for faster processing
train_processed = '../test/train_2400_' + protein_name + '_256bits.pkl'
validation_processed = '../test/validation_600_' + protein_name + '_256bits.pkl'
test_processed = '../test/test_' + protein_name + '.pkl'

# model parameters
model_name = 'NN' 
n_epochs = 1000
resampling = True
reweighting = False # add weight based on class imbalance
dropout_rate = 0.3
learning_rate = 0.0001
weight_decay=0.001
checkpoint_interval = 10  # Save a checkpoint every 5 epochs
prefix = model_name + '_' + protein_name + '_resampling-{}_reweighting-{}'.format(resampling,reweighting) 
checkpoint_dir = 'checkpoints_' + prefix  # Directory to save checkpoints
# Ensure the checkpoint directory exists
os.makedirs(checkpoint_dir, exist_ok=True)

print('=== {} ==='.format(prefix))

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

    test_data = test_data[test_data['protein_name'] == protein_name]
    test_ids = test_data['id']
    # save test_ids to csv file
    test_ids.to_csv('test_ids_' + protein_name + '.csv', index=False)

    val_pos_rate = 0
    while val_pos_rate < 1e-5: # to ensure that validation data has positive examples
        train_data_temp = shuffle(train_data_temp)
        train_data_temp = train_data_temp.reset_index(drop=True)

        # split train data into train and validation
        train_data = train_data_temp[:int(0.9*len(train_data_temp))]
        validation_data = train_data_temp[int(0.9*len(train_data_temp)):]

        # print count of unique values in binds column
        val_pos_rate = validation_data[validation_data['binds'] == 1].value_counts().sum() / validation_data['binds'].value_counts().sum()
    
    print('\nTrain size: ', len(train_data))
    print('Validation size: ', len(validation_data))
    print('\nTrain positive example rate:\n', train_data[train_data['binds'] == 1].value_counts().sum() / train_data['binds'].value_counts().sum())
    print('Validation positive example rate:\n', val_pos_rate)

    # free memory
    del train_data_temp
    gc.collect()

    batch_size = 1000 # number of molecules to process at a time
    
    # calculate finger prints for molecules
    time0 = time()
    print('\nCalculating fingerprints for train data')
    train_fingerprints = parallel_process(train_data['molecule_smiles'], batch_smiles_to_fp, batch_size)
    train_fingerprints = [item for sublist in train_fingerprints for item in sublist]  # Flatten the list of lists
    train_data['molecule_fp'] = train_fingerprints
    del train_fingerprints
    gc.collect()
    # save to pickle file
    train_data.to_pickle('train_' + protein_name + '_256bits.pkl')
    print('Time elapsed (min): ', round((time()-time0)/60, 2))

    time0 = time()
    print('\nCalculating fingerprints for validation data')
    validation_fingerprints = parallel_process(validation_data['molecule_smiles'], batch_smiles_to_fp, batch_size)
    validation_fingerprints = [item for sublist in validation_fingerprints for item in sublist]  # Flatten the list of lists
    validation_data['molecule_fp'] = validation_fingerprints
    del validation_fingerprints
    gc.collect()
    # save to pickle file
    validation_data.to_pickle('validation_' + protein_name + '_256bits.pkl')
    print('Time elapsed (min): ', round((time()-time0)/60, 2))

    time0 = time()
    print('\nCalculating fingerprints for test data')
    fingerprints = []
    test_fingerprints = parallel_process(test_data['molecule_smiles'], batch_smiles_to_fp, batch_size)
    test_fingerprints = [item for sublist in test_fingerprints for item in sublist]  # Flatten the list of lists
    test_data['molecule_fp'] = test_fingerprints
    del test_fingerprints
    gc.collect()
    # save to pickle file
    test_data.to_pickle('test_' + protein_name + '_256bits.pkl')
    print('Time elapsed (min): ', round((time()-time0)/60, 2))

else: 
    print('Loading processed data from pickle files')
    # load the data from pickle files
    train_data = pd.read_pickle(train_processed)
    validation_data = pd.read_pickle(validation_processed)
    test_data = pd.read_pickle(test_processed)

# build a simple a model using the Morgan fingerprints as features and binds as target
# use average_precision_score as the evaluation metric

X_train = np.array(list(train_data['molecule_fp']))
y_train =np.array(train_data['binds'])

X_validation = np.array(list(validation_data['molecule_fp']))
y_validation = np.array(validation_data['binds'])

X_test = np.array(list(test_data['molecule_fp']))

# clean data
print('\nbefore removing None values')
print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_validation shape: ', X_validation.shape)
print('y_validation shape: ', y_validation.shape)
print('X_test shape: ', X_test.shape)

# remove None values
X_train = X_train[~np.any(np.isnan(X_train), axis=1)]
y_train = y_train[~np.any(np.isnan(X_train), axis=1)]
X_validation = X_validation[~np.any(np.isnan(X_validation), axis=1)]
y_validation = y_validation[~np.any(np.isnan(X_validation), axis=1)]
X_test = X_test[~np.any(np.isnan(X_test), axis=1)]

X_train = X_train[~np.isnan(y_train)]
y_train = y_train[~np.isnan(y_train)]
X_validation = X_validation[~np.isnan(y_validation)]
y_validation = y_validation[~np.isnan(y_validation)]


X_train = np.array(list(X_train))
X_validation = np.array(list(X_validation))
X_test = np.array(list(X_test))

print('\nafter removing None values')
print('X_train shape: ', X_train.shape)
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
    X_train_t = torch.tensor(X_train_resample, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_resample, dtype=torch.float32)
else:
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_validation_t = torch.tensor(X_validation, dtype=torch.float32)
y_validation_t = torch.tensor(y_validation, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

# Create a DataLoader for the training data
train_data = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Initialize the model, loss function, and optimizer
model = Net(n_features=X_train.shape[1], dropout_rate=dropout_rate).to(device)
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
patience = 20  # Number of epochs to wait for improvement before stopping
best_loss = float('inf')  # Best loss value so far, initialized to infinity
epochs_no_improve = 0  # Number of epochs with no improvement, initialized to 0

train_losses = []
validation_losses = []
train_scores = []
validation_scores = []

# Train the model
print('\nTraining the model...')
time0 = time()
for epoch in range(n_epochs):  # 100 epochs
    for inputs, targets in train_loader:
        # Move data to the GPU if available
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets.view(-1, 1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_losses.append(loss.item())

    if (epoch+1) % 50 == 0:
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
        validation_score = average_precision_score(y_validation, y_validation_pred)
        validation_scores.append(validation_score)

        y_train_pred = model(X_train_t)
        train_score = average_precision_score(y_train_t, y_train_pred)
        train_scores.append(train_score)

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
print(f'\nTrain Loss: {loss.item()}')

# Evaluate the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    y_train_pred = model(X_train_t)
    train_score = average_precision_score(y_train_t, y_train_pred)
    
    y_validation_pred = model(X_validation_t)
    validation_loss = criterion(y_validation_pred, y_validation_t.view(-1, 1))
    validation_score = average_precision_score(y_validation, y_validation_pred)
print(f'Validation Loss: {validation_loss.item()}')
print(f'\nTrain Score: {train_score}')
print(f'Validation Score: {validation_score}')

# save model to file
torch.save(model, 'model_{}.pth'.format(prefix))

# evaluate on test data
print('\nMaking predictions on test data...')
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_t)
    y_test_pred = y_test_pred.numpy().flatten()
test_ids = pd.read_csv('test_ids_' + protein_name + '.csv')
test_ids = test_ids.to_numpy().flatten()

# make a dataframe with id and binds columns
test_results = pd.DataFrame({'id': test_ids, 'binds': y_test_pred})
test_results.to_csv('test_pred_' + prefix + '.csv', index=False)


# plot
plot_scatter(y_train_t, y_train_pred, y_validation, y_validation_pred, prefix=prefix, fold=0)

# plot loss
plt.figure(figsize=(4, 4))
plt.plot(train_losses, label='Train, score={:.2e}, loss={:.2e}'.format(train_score,loss.item()))
plt.plot(validation_losses, label='Validation, score={:.2e}, loss={:.2e}'.format(validation_score,validation_loss.item()))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend(fontsize=6)
plt.savefig('{}_loss.png'.format(prefix), dpi=500, bbox_inches='tight')

# plot score
plt.figure(figsize=(4, 4))
plt.plot(train_scores, label='Train, score={:.2e}, loss={:.2e}'.format(train_score,loss.item()))
plt.plot(validation_scores, label='Validation, score={:.2e}, loss={:.2e}'.format(validation_score,validation_loss.item()))
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Training Score')
plt.legend(fontsize=6)
plt.savefig('{}_score.png'.format(prefix), dpi=500, bbox_inches='tight')
