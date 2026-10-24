import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle
from sklearn.metrics import average_precision_score

import gc
import torch
from torch import nn
from torch.utils.data import Dataset

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import DataStructs

from tqdm import tqdm
import multiprocessing
from functools import partial
import psutil

from time import time

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
    
    return data_list

class LargeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature, label = self.data[idx]
        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return feature_tensor, label_tensor
    
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
            inputs, targets = inputs.to(device), targets.to(device)

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
                    inputs, targets = inputs.to(device), targets.to(device)
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
    def __init__(self, n_features=2048, dropout_rate=0.5, n_classes=3):
        super(Net5, self).__init__()
        self.fc1 = nn.Linear(n_features, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, n_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate) 
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.sigmoid(self.fc5(x))
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
    def __init__(self, n_features=20, hidden_channels=64):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(n_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, 1)  # For regression

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.mean(x, dim=0)  # Graph-level pooling (mean of node embeddings), computes the mean of the node embeddings along the node dimension => output dimensions = [hidden_channels]
        x = self.fc(x)
        return x    
