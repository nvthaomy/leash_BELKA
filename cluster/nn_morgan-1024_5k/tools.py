import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle

import torch
from torch import nn

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import DataStructs

from tqdm import tqdm
import multiprocessing
from functools import partial
import psutil

print('Number of available GPUs: ', torch.cuda.device_count())
print('Number of available CPUs: ', multiprocessing.cpu_count())

# Get the process object using PID
def check_memory(pid=os.getpid()):
    p = psutil.Process(pid)
    # Get the information of memory usage
    mem_info = p.memory_info()
    # Print memory usage in GB
    print('Memory used (in GB):', mem_info.rss / 1024 / 1024 / 1024)

def process_batch(batch, func):
    return func(batch)

def parallel_process(data, func, batch_size, n_jobs=multiprocessing.cpu_count()):
    with multiprocessing.Pool(processes=min(n_jobs, multiprocessing.cpu_count())) as pool:
        result = pool.map(partial(process_batch, func=func), [data[i:i+batch_size] for i in range(0, len(data), batch_size)])
    return result

# Define a function to generate Morgan fingerprints for a batch of SMILES strings
def batch_smiles_to_fp(smiles_batch, fp_size=1024):
    fps = []
    for smiles in smiles_batch:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            morgan_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
            fp = morgan_fp_gen.GetFingerprint(mol)
            fps.append(np.array(fp))
        else:
            fps.append(None)
    return fps

# Define a function to process the dataset in batches
def process_in_batches(data, batch_size):
    num_batches = (len(data) + batch_size - 1) // batch_size  # Calculate the number of batches
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        batch = data.iloc[start_idx:end_idx]
        yield batch

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    morgan_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    return morgan_fp_gen.GetFingerprint(mol)

def plot_scores(train_scores, validation_scores, train_rmse, validation_rmse, prefix='model'):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_scores, 'o', linestyle='None', ms=5, label='Train')
    plt.plot(validation_scores, 'o', linestyle='None', ms=5, label='Validation')
    plt.xlabel('Fold')
    plt.ylabel('Average Precision Score')
    plt.legend()
    plt.title('Average Precision Score - {}'.format(prefix))
    
    plt.subplot(1, 2, 2)
    plt.plot(train_rmse, 'o', linestyle='None', ms=5, label='Train')
    plt.plot(validation_rmse, 'o', linestyle='None', ms=5, label='Validation')
    plt.xlabel('Fold')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('RMSE - {}'.format(prefix))
    plt.savefig(prefix + '_scores.png', dpi=500, bbox_inches='tight')


def plot_scatter(y_train, y_train_pred, y_validation, y_validation_pred, prefix='model', fold=0):
    plt.figure(figsize=(4, 4))
    plt.scatter(y_train, y_train_pred, marker='o', alpha=0.5, label='Train')
    plt.scatter(y_validation, y_validation_pred, marker='s', alpha=0.5, label='Validation')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title('True vs Predicted - {} - Fold {}'.format(prefix, fold))
    # use same scale for both axes
    plt.xlim(min(min(y_train), min(y_validation)) * 0.7, max(max(y_train), max(y_validation)) * 1.1)
    plt.ylim(min(min(y_train), min(y_validation)) * 0.7, max(max(y_train), max(y_validation)) * 1.1)
    # add a diagonal line
    plt.plot([min(min(y_train), min(y_validation)), max(max(y_train), max(y_validation))], [min(min(y_train), min(y_validation)), max(max(y_train), max(y_validation))], 'k--')
    plt.legend(fontsize=6)
    plt.savefig(prefix + '_scatter.png', dpi=500, bbox_inches='tight')


class Net(nn.Module):
    def __init__(self, n_features=1024, dropout_rate=0.5):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate) 
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc4(x))
        return x

class Net4(nn.Module):
    def __init__(self, n_features=1024, dropout_rate=0.5):
        super(Net4, self).__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate) 
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.sigmoid(self.fc4(x))
        return x

class Net4b(nn.Module):
    def __init__(self, n_features=1024, dropout_rate=0.5):
        super(Net4b, self).__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate) 
        self.bn1 = nn.BatchNorm1d(512)
        #self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        # x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.sigmoid(self.fc4(x))
        return x
    
class Net5(nn.Module):
    def __init__(self, n_features=1024, dropout_rate=0.5):
        super(Net5, self).__init__()
        self.fc1 = nn.Linear(n_features, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate) 
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.relu(self.bn5(self.fc5(x)))
        x = self.dropout(x)
        x = self.sigmoid(self.fc6(x))
        return x

class Net5b(nn.Module):
    def __init__(self, n_features=1024, dropout_rate=0.5):
        super(Net5b, self).__init__()
        self.fc1 = nn.Linear(n_features, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate) 
        self.bn1 = nn.BatchNorm1d(2048)
        #self.bn2 = nn.BatchNorm1d(1024)
        #self.bn3 = nn.BatchNorm1d(512)
        #self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.bn5(self.fc5(x)))
        x = self.dropout(x)
        x = self.sigmoid(self.fc6(x))
        return x
