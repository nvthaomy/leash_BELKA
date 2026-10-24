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

print('Number of available GPUs: ', torch.cuda.device_count())
print('Number of available CPUs: ', multiprocessing.cpu_count())

def process_batch(batch, func):
    return func(batch)

def parallel_process(data, func, batch_size):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        result = pool.map(partial(process_batch, func=func), [data[i:i+batch_size] for i in range(0, len(data), batch_size)])
    return result

def calculate_fingerprints(batch):
    fingerprints = []
    print('batch: ', batch)
    for smiles in batch:
        fp = batch_smiles_to_fp(smiles)
        fp = [np.array(f) for f in fp]
        fingerprints.extend(fp)
    return fingerprints

# Define a function to generate Morgan fingerprints for a batch of SMILES strings
def batch_smiles_to_fp(smiles_batch):
    fps = []
    for smiles in smiles_batch:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            morgan_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)
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
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate) 
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x