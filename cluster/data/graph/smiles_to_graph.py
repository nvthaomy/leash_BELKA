import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from rdkit import Chem
from rdkit.Chem import rdmolops
import numpy as np

def smiles_to_graph(smiles):

    non_metal_atoms = ['H', 'He', 'C', 'N', 'O', 'F', 'Ne', 'P', 'S', 
                       'Cl', 'Ar', 'Se', 'Br', 'Kr', 'I', 'Xe', 'At', 'Rn']
    halogen_atoms = ['F', 'Cl', 'Br', 'I', 'At', 'Ts']
    
    # one-hot encoding of atom type
    atom_types = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'P', 'I']
    hybrids = ['SP2','SP3']

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
    edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    return Data(x=node_features, edge_index=edge_indices)
