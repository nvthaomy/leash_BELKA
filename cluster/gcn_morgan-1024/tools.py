import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle
from sklearn.metrics import average_precision_score
import pickle
import pandas as pd

import gc
import torch
from torch import nn
from torch.utils.data import Dataset

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Batch

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import DataStructs

from tqdm import tqdm
import multiprocessing
#import torch.multiprocessing as multiprocessing
from functools import partial
import psutil

from time import time

print('Number of available GPUs: ', torch.cuda.device_count())
print('Number of available CPUs: ', multiprocessing.cpu_count())

# custom dataset class

class CustomDataset(Dataset):
    def __init__(self, file_list, use_fp=True, use_graph=False, protein=None, load_all_data=True, transform=None):
        """
        Args:
            file_list (list): List of file paths to .pkl files.
            use_fp (bool): Whether to use the 'molecule_fp' feature.
            use_graph (bool): Whether to use the 'molecule_graph' feature.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_list = file_list
        self.use_fp = use_fp
        self.use_graph = use_graph
        self.protein = protein
        if load_all_data:
            self.data = self._load_data()
            self.indices = None
        else:
            self.indices = self._get_indices()
            self.data = None

    def _load_data(self):
        """Load all dataframes and concatenate them into a single dataframe."""
        print('Loading data...')
        dfs = []
        for file_path in self.file_list:
            with open(file_path, 'rb') as f:
                print(f'File: {file_path}')
                df = pickle.load(f)
                dfs.append(df)
                del df
                gc.collect()
            check_memory(pid=os.getpid())
        return pd.concat(dfs, ignore_index=True)
    
    def _get_indices(self):
        """Create a list of cumulative indices for efficient look-up."""
        indices = []
        cum_sum = 0
        for file_path in self.file_list:
            with open(file_path, 'rb') as f:
                df = pickle.load(f)
                num_examples = len(df)
                indices.append((cum_sum, cum_sum + num_examples, file_path))
                cum_sum += num_examples
                del df
                gc.collect()
        return indices

    def _find_file(self, idx):
        """Find the file and local index corresponding to the global index."""
        for start, end, file_path in self.indices:
            if start <= idx < end:
                return file_path, idx - start
        raise IndexError("Index out of range")

    def __len__(self):
        if self.data is not None:
            return len(self.data)
        else:
            return self.indices[-1][1]

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to be fetched.
        
        Returns:
            sample: A dictionary containing features and labels.
        """

        if self.data is not None:
            row = self.data.iloc[idx]
        else:
            file_path, local_idx = self._find_file(idx)
            
            with open(file_path, 'rb') as f:
                df = pickle.load(f)
            
            # Extract the row corresponding to the local index
            row = df.iloc[local_idx]
        
        # Retrieve labels
        label = np.array(row[['binds_BRD4', 'binds_HSA', 'binds_sEH']].values, dtype=np.float32)
        #print('label: ',label)
        if self.protein:
            label = label[['BRD4','HSA','sEH'].index(self.protein)]
        label_tensor = torch.tensor(label, dtype=torch.float32)

        # Retrieve features
        if self.use_fp:
            fingerprint = np.array(row['molecule_fp'])
            #print('fingerprint: ',fingerprint)
            fingerprint = torch.tensor(np.array(row['molecule_fp']), dtype=torch.float32)
        if self.use_graph:
            #print('graph: ',row['molecule_graph'])
            graph = row['molecule_graph']
        
        # Return the sample
        if self.use_fp and self.use_graph:
            sample = (fingerprint, graph), label_tensor
        elif self.use_fp:
            sample = fingerprint, label_tensor
        elif self.use_graph:
            sample = graph, label_tensor
        return sample

class TensorCustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature, label = self.data[idx]
        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return feature_tensor, label_tensor
    
class GraphDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        graph, label = self.data[idx]
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return graph, label_tensor

class MixedDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        fp, graph, label = self.data[idx]
        fingerprint = torch.tensor(fp, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return (fingerprint, graph), label_tensor
    
def collate_fn_graph(batch):
    # used when features are graphs
    graphs = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    return Batch.from_data_list(graphs), labels
    
def collate_fn_fpgraph(batch):
    # used when features are fingerprints and graphs
    fingerprints = [item[0][0] for item in batch]
    graphs = [item[0][1] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    fingerprints = torch.stack(fingerprints)
    return [fingerprints, Batch.from_data_list(graphs)], labels

def collate_fn_fp(batch):
    # used when features are fingerprints and graphs
    fingerprints = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    fingerprints = torch.stack(fingerprints)
    return fingerprints, labels

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
        #result = pool.map(partial(process_batch, func=func), data)
    return result

def serial_process(data, func, batch_size):
    result = []
    for i in range(0, len(data), batch_size):
        result.append(func(data[i:i+batch_size]))
    return result

# Define a function to generate Morgan fingerprints for a batch of SMILES strings
# https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
def batch_smiles_to_fp(smiles_batch, fp_size=1024, fp_invariant_size=0):
    fps = [] # regular Morgan fingerprints
    fps_invariants = [] # Morgan fingerprints with invariants

    for smiles in smiles_batch:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            morgan_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
            fp = morgan_fp_gen.GetFingerprint(mol)
            fps.append(np.array(fp))

            if fp_invariant_size > 0:
                invgen = AllChem.GetMorganFeatureAtomInvGen()
                morgan_fp_gen_invariants = AllChem.GetMorganGenerator(radius=2, fpSize=fp_invariant_size, atomInvariantsGenerator=invgen)
                fp = morgan_fp_gen_invariants.GetFingerprint(mol)
                fps_invariants.append(np.array(fp))
        else:
            fps.append(None)
            raise ValueError("Invalid SMILES string")
    
    # concat both types of fingerprints row by row
    if fp_invariant_size > 0:
        return np.concatenate([np.array(fps), np.array(fps_invariants)], axis=1)
    else:
        return np.array(fps)    

def batch_smiles_to_graph(smiles_batch):

    non_metal_atoms = ['H', 'He', 'C', 'N', 'O', 'F', 'Ne', 'P', 'S', 
                       'Cl', 'Ar', 'Se', 'Br', 'Kr', 'I', 'Xe', 'At', 'Rn']
    halogen_atoms = ['F', 'Cl', 'Br', 'I', 'At', 'Ts']
    
    # one-hot encoding of atom type
    atom_types = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'P', 'I']
    hybrids = ['SP2','SP3']

    data_list = []

    for smiles in smiles_batch:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        # Extract node features (atom types)
        node_features = []
        for atom in mol.GetAtoms():
            atom_type = atom.GetSymbol()
            hybrid = str(atom.GetHybridization()).upper()

            # one-hot encoding of atom type and hybridization
            atom_type = [float(atom_type == x) for x in atom_types]
            hybrid = [float(hybrid == x) for x in hybrids]

            feature = [atom.GetMass(),
                    atom.GetDegree(),
                    atom.GetTotalNumHs(),
                    float(atom.IsInRing()),
                    float(atom.GetIsAromatic()),
                    float(not atom.GetSymbol() in non_metal_atoms),
                    float(atom.GetSymbol() in halogen_atoms),
                    float(atom.GetFormalCharge() >=0),
                    float(atom.GetFormalCharge() < 0)
                    ]
            feature.extend(atom_type)
            feature.extend(hybrid)
            node_features.append(feature)  

            # print(f'\n==={atom.GetSymbol()}===')
            # print(feature)
            # print(len(feature))
        
        node_features = torch.tensor(node_features, dtype=torch.float32)

        # Extract edge indices
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i])
        edge_indices = torch.tensor(edge_indices, dtype=torch.int32).t().contiguous()

        data_list.append(Data(x=node_features, edge_index=edge_indices))
    
    # Serialize the Data objects to bytes
    serialized_data_list = [pickle.dumps(data) for data in data_list]
    
    return serialized_data_list

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

def train_model(model, train_loader, criterion, optimizer, device, n_epochs=100, 
                checkpoint_dir='checkpoints', checkpoint_interval=10, patience=10, 
                val_loader=None, fold_index=0):

    # Early stopping parameters
    best_loss = float('inf')  # Best loss value so far, initialized to infinity
    best_score = -100
    epochs_no_improve = 0  # Number of epochs with no improvement, initialized to 0

    train_losses = []
    train_scores = []
    val_losses = []
    val_scores = []

    check_memory(pid=os.getpid())

    # Train the model
    print('\nTraining the model...')
    time0 = time()
    for epoch in range(n_epochs):  # 100 epochs
        y_train_pred = []
        y_train_tar = []
        for inputs, targets in train_loader:
            # Move data to the GPU if available
            # print('targets.shape: ',targets.shape)
            if isinstance(inputs, list) and len(inputs) == 2:
                #print('inputs[0].shape: ',inputs[0].shape)
                inputs = [i.to(device) for i in inputs]
            else:
                #print('inputs.shape: ',inputs.shape)
                inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_train_pred.append(outputs.detach().cpu().numpy())
            y_train_tar.append(targets.cpu().numpy())

        train_losses.append(loss.item())

        # Evaluate on train set
        y_train_pred = np.concatenate(y_train_pred)
        y_train_tar = np.concatenate(y_train_tar)
        train_score = average_precision_score(y_train_tar.flatten(), y_train_pred.flatten())
        train_scores.append(train_score)

        # Evaluate on validation set
        if not val_loader is None:
            model.eval()
            y_val_pred = []
            y_val_tar = []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    if isinstance(inputs, list) and len(inputs) == 2:
                        inputs = [i.to(device) for i in inputs]
                    else:
                        inputs = inputs.to(device)
                    outputs = model(inputs)
                    y_val_pred.append(outputs.cpu().numpy())
                    y_val_tar.append(targets.cpu().numpy())
                
            y_val_pred = np.concatenate(y_val_pred)
            y_val_tar = np.concatenate(y_val_tar)
            y_val_pred = torch.tensor(y_val_pred, dtype=torch.float32).to(device)
            y_val_tar = torch.tensor(y_val_tar, dtype=torch.float32).to(device)

            val_loss = criterion(y_val_pred, y_val_tar)
            val_losses.append(val_loss.item())

            val_score = average_precision_score(y_val_tar.cpu().numpy().flatten(), y_val_pred.cpu().numpy().flatten())
            val_scores.append(val_score)

        else:
            val_losses.append(None)
            val_scores.append(None)

        if (epoch+1) % 5 == 0:
            print(f'\nEpoch {epoch+1}/{n_epochs}')
            print(f'Train Loss: {loss.item()}, Train Score: {train_score}')
            print(f'Validation Loss: {val_losses[-1]}, Validation Score: {val_scores[-1]}')
            check_memory(pid=os.getpid())
            print('Time elapsed (min): ', round((time()-time0)/60, 2))

        # Save a checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'model_checkpoint_fold-{fold_index}.pth')
            torch.save(model, checkpoint_path)

        # Check for improvement based on loss
        if not val_loader is None:
            if val_losses[-1] < best_loss:
                best_loss = val_losses[-1]
                epochs_no_improve = 0
                best_epoch = epoch
                # save model to file
                torch.save(model, '{}/model_best_fold-{}.pth'.format(checkpoint_dir,fold_index))
            else:
                epochs_no_improve += 1
        else:
            if train_losses[-1] < best_loss:
                best_loss = train_losses[-1]
                epochs_no_improve = 0
                best_epoch = epoch
                # save model to file
                torch.save(model, '{}/model_best_fold-{}.pth'.format(checkpoint_dir,fold_index))
            else:
                epochs_no_improve += 1

        model.train()

        # Early stopping
        if epochs_no_improve == patience:
            print('Early stopping!')

            # load the best model
            model = torch.load('{}/model_best_fold-{}.pth'.format(checkpoint_dir,fold_index))
            break

    print('Time elapsed (min): ', round((time()-time0)/60, 2))
    s='Train Loss: {}\n'.format(train_losses[best_epoch])
    s+='Train Score: {}\n'.format(train_scores[best_epoch])
    s+='\nValidation Loss: {}\n'.format(val_losses[best_epoch])
    s+='Validation Score: {}\n'.format(val_scores[best_epoch])
    print('\n' + s)

    with open(checkpoint_dir+'/results_fold-{}.txt'.format(fold_index), 'w') as file:
        file.write(s)

    # save model to file
    model_path = '{}/model_fold-{}.pth'.format(checkpoint_dir,fold_index)
    torch.save(model, model_path)

    # plot loss
    plt.figure(figsize=(4, 4))
    plt.plot(train_losses, label='Train, score={:.2e}, loss={:.2e}'.format(train_score,loss.item()))
    if not val_loader is None:
        plt.plot(val_losses, label='Validation, score={:.2e}, loss={:.2e}'.format(val_scores[-1],val_losses[-1]))
        plt.vlines(best_epoch, min(min(train_losses),min(val_losses)), max(max(train_losses),max(val_losses)), linestyles='dashed', colors='gray', label='Best epoch')
    else:
        plt.vlines(best_epoch, min(train_losses), max(train_losses), linestyles='dashed', colors='gray', label='Best epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend(fontsize=6)
    plt.savefig('{}/loss_fold-{}.png'.format(checkpoint_dir,fold_index), dpi=500, bbox_inches='tight')

    # plot score
    plt.figure(figsize=(4, 4))
    plt.plot(train_scores, label='Train, score={:.2e}, loss={:.2e}'.format(train_score,loss.item()))
    if not val_loader is None:
        plt.plot(val_scores, label='Validation, score={:.2e}, loss={:.2e}'.format(val_scores[-1],val_losses[-1]))
        plt.vlines(best_epoch, min(min(train_scores),min(val_scores)), max(max(train_scores),max(val_scores)), linestyles='dashed', colors='gray', label='Best epoch')
    else:
        plt.vlines(best_epoch, min(train_scores), max(train_scores), linestyles='dashed', colors='gray', label='Best epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Score')
    plt.legend(fontsize=6)
    plt.savefig('{}/score_fold-{}.png'.format(checkpoint_dir,fold_index), dpi=500, bbox_inches='tight')

    plt.close('all')

    del y_train_tar, y_train_pred, train_loader
    if not val_loader is None:
        del val_loader, y_val_tar, y_val_pred
    gc.collect()

    return model, train_losses[best_epoch], train_scores[best_epoch], val_losses[best_epoch], val_scores[best_epoch]

class Net(nn.Module):
    def __init__(self, n_features=1024, dropout_rate=0.5, n_classes=3):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_classes)
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
    def __init__(self, n_features=1024, dropout_rate=0.5, n_classes=3):
        super(Net4, self).__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_classes)
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
    def __init__(self, n_features=1024, dropout_rate=0.5, n_classes=3):
        super(Net4b, self).__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_classes)
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
    def __init__(self, n_features=1024, dropout_rate=0.5, n_classes=3):
        super(Net5, self).__init__()
        self.fc1 = nn.Linear(n_features, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, n_classes)
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
    def __init__(self, n_features=1024, dropout_rate=0.5, n_classes=3):
        super(Net5b, self).__init__()
        self.fc1 = nn.Linear(n_features, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, n_classes)
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
    
class CNN1d(nn.Module):
    def __init__(self, input_channels=1, n_classes=3, dropout_rate=0.5):
        super(CNN1d, self).__init__()
        # dimension of each chanel after each conv layer
        # n_out = (n_in + 2*padding - kernel_size) / stride + 1
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1) # n_in = n_out = 1024
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1) # n_in = n_out = 512
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0) # n_out = n_in / 2

        # Calculate the size of the input to the fully connected layer after convolutions
        self.fc_input_size = 64 * 256

        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool(self.relu(self.bn1(self.conv1(x)))) # n_in = 1024, n_out = 512
        x = self.pool(self.relu(self.bn2(self.conv2(x)))) # n_in = 512, n_out = 256

        # Flatten before passing to dense layers
        x = x.view(-1, self.fc_input_size)

        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))

        return x

class CNN1d_b(nn.Module):
    def __init__(self, input_channels=1, n_classes=3, dropout_rate=0.5):
        super(CNN1d_b, self).__init__()
        # dimension of each chanel after each conv layer
        # n_out = (n_in + 2*padding - kernel_size) / stride + 1
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, stride=1, padding=1) # n_in = n_out = 1024
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1) # n_in = n_out = 512
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1) # n_in = n_out = 256
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0) # n_out = n_in / 2

        # Calculate the size of the input to the fully connected layer after convolutions
        self.fc_input_size = 128 * 64

        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool(self.relu(self.bn1(self.conv1(x)))) # n_in = 1024, n_out = 512
        x = self.pool(self.relu(self.bn2(self.conv2(x)))) # n_in = 512, n_out = 256
        x = self.pool(self.relu(self.bn3(self.conv3(x)))) # n_in = 256, n_out = 128

        # Flatten before passing to dense layers
        x = x.view(-1, self.fc_input_size)

        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))

        return x

class GCN(torch.nn.Module):
    def __init__(self, graph_layer_nodes=[20,64,64], fc_layer_nodes=[1024,512], dropout_rate=0.5, n_classes=3):
        super(GCN, self).__init__()
        
        conv_list = []
        for i in range(len(graph_layer_nodes)-1):
            conv_list.append(GCNConv(graph_layer_nodes[i], graph_layer_nodes[i+1]))
        self.conv_list = nn.ModuleList(conv_list)
        
        n_fc_in = graph_layer_nodes[-1] * 3
        fc_layer_nodes = [n_fc_in] + fc_layer_nodes
        fc_list = []
        for i in range(len(fc_layer_nodes)-1):
            fc_list.append(torch.nn.Linear(fc_layer_nodes[i], fc_layer_nodes[i+1]))
            fc_list.append(torch.nn.BatchNorm1d(fc_layer_nodes[i+1]))
            fc_list.append(torch.nn.ReLU())
            fc_list.append(torch.nn.Dropout(dropout_rate))
        fc_list.append(torch.nn.Linear(fc_layer_nodes[-1], n_classes))
        fc_list.append(torch.nn.Sigmoid())
        self.fc_layers = nn.Sequential(*fc_list)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.conv_list:
            x = self.relu(conv(x, edge_index))
            x = self.dropout(x)
        
        # Graph-level pooling (mean of node embeddings per graph)
        x_mean = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x_add = global_add_pool(x, batch)  # [batch_size, hidden_channels]
        x_max = global_max_pool(x, batch)  # [batch_size, hidden_channels]
        x = torch.cat([x_mean, x_add, x_max], dim=1)

        x = self.fc_layers(x)
        return x    

class FCLayers(nn.Module):
    def __init__(self, layer_nodes=[1024, 512, 256], dropout_rate=0.5, device='cpu'):
        super(FCLayers, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        layers = []
        for i in range(len(layer_nodes)-1):
            layers.append(nn.Linear(layer_nodes[i], layer_nodes[i+1]))
            if i < 1:
                layers.append(nn.BatchNorm1d(layer_nodes[i+1]))
            layers.append(self.relu)
            layers.append(self.dropout)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
class GNNLayers(torch.nn.Module):
    def __init__(self, layer_nodes=[20,64,64], n_out=256, dropout_rate=0.5, device='cpu'):
        super(GNNLayers, self).__init__()
        
        conv_list = []
        for i in range(len(layer_nodes)-1):
            conv_list.append(GCNConv(layer_nodes[i], layer_nodes[i+1]).to(device))
        self.conv_list = nn.ModuleList(conv_list)

        n_fc_in = layer_nodes[-1] * 3
        self.fc = torch.nn.Linear(n_fc_in, n_out)  # For regression
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, data):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.conv_list:
            x = self.relu(conv(x, edge_index))
            x = self.dropout(x)

        # Graph-level pooling (mean/add/max of node embeddings per graph)
        x_mean = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x_add = global_add_pool(x, batch)  # [batch_size, hidden_channels]
        x_max = global_max_pool(x, batch)  # [batch_size, hidden_channels]
        x = torch.cat([x_mean, x_add, x_max], dim=1)
        
        # FC layers
        x = self.dropout(x)
        x = self.relu(self.fc(x))
    
        return x   
    
class FPGCN(torch.nn.Module):
    def __init__(self, fc_layer_nodes=[1024, 512], gnn_layer_nodes=[20,64,64], gnn_n_out=128, 
                 fc_combined_layer_nodes=[1024,512], n_classes=3, dropout_rate=0.5, device='cpu'):
        
        super(FPGCN, self).__init__()
        
        # fc_layer_nodes: first element is the number of fingerprint features
        # gnn_layer_nodes: first element is the number of node features
        # fc_combined_layer_nodes: exclude the number of nodes in input and final layer
        
        self.fp_net = FCLayers(layer_nodes=fc_layer_nodes, dropout_rate=dropout_rate, device=device).to(device)
        self.gcn_net = GNNLayers(layer_nodes=gnn_layer_nodes, n_out=gnn_n_out, dropout_rate=dropout_rate, device=device).to(device)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)

        # Combine outputs of FP and GCN models
        n_combined_in = fc_layer_nodes[-1] + gnn_n_out
        self.fc_combined_list = []
        self.bn_combined_list = []
        fc_combined_layer_nodes = [n_combined_in] + fc_combined_layer_nodes
        
        layers = []
        for i in range(len(fc_combined_layer_nodes)-1):
            layers.append(nn.Linear(fc_combined_layer_nodes[i], fc_combined_layer_nodes[i+1]))
            if i <= 1:
                layers.append(nn.BatchNorm1d(fc_combined_layer_nodes[i+1]))
            layers.append(self.relu)
            layers.append(self.dropout)
        layers.append(nn.Linear(fc_combined_layer_nodes[-1], n_classes))
        layers.append(self.sigmoid)
        self.fc_layers = nn.Sequential(*layers)
        
        print("FPGCN device:", next(self.parameters()).device)

    def forward(self, data):
        fp, graph = data

        x_fp = self.fp_net(fp)
        x_gcn = self.gcn_net(graph)
        
        x = torch.cat([x_fp, x_gcn], dim=1)
        x = self.fc_layers(x)
        return x    