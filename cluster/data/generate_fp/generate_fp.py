from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import DataStructs

import numpy as np
import os

from tqdm import tqdm
import multiprocessing
from functools import partial

import pyarrow.parquet as pq
import pandas as pd

def process_batch(batch, func):
    return func(batch)

def parallel_process(data, func, batch_size, n_jobs=multiprocessing.cpu_count()):
    with multiprocessing.Pool(processes=n_jobs) as pool:
        result = pool.map(partial(process_batch, func=func), [data[i:i+batch_size] for i in range(0, len(data), batch_size)])
    return result

# Define a function to generate Morgan fingerprints for a batch of SMILES strings
def batch_smiles_to_fp(smiles_batch):
    fps = []
    for smiles in smiles_batch:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            morgan_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
            fp = morgan_fp_gen.GetFingerprint(mol)
            fps.append(np.array(fp))
        else:
            fps.append(None)
    return fps

data_file = '../train.parquet'
chunk_index = 0

# Load id columns
table = pq.read_table(data_file, columns=['id'])
df = table.to_pandas()
ids = df.values.flatten()

# split ids into chunks
chunk_size = int(10e6)
chunks = [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]
