import pandas as pd
import numpy as np
import torch
from tools import * 
import gc
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import os 
from torch_geometric.loader import DataLoader as DataLoaderGCN

protein_name = None
ext='_fold-0'

load_model = True
checkpoint_dir = 'checkpoints_posrate-0.082_size-5.0M_GCN_00_1024bits_invar0bits_resampling-False_reweighting-False'
checkpoint_file = ''
model = None 

model_file = 'model_fold-0.pth'
test_processed = ['processed_data/test_graph_1024bits_invar0bits_chunk1.pkl',
                    'processed_data/test_graph_1024bits_invar0bits_chunk2.pkl',
                    'processed_data/test_graph_1024bits_invar0bits_chunk3.pkl',
                    'processed_data/test_graph_1024bits_invar0bits_chunk4.pkl',]
            
submission_temp_file = './test_ids.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if load_model:
    model = torch.load(checkpoint_dir + '/' + model_file, map_location=torch.device(device))
    model.eval()
else:
    checkpoint = torch.load(checkpoint_dir + '/' + checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

# Load test features
print('\nLoading processed test data from pickle/parquet files')
X_test = []
y_test_pred = []
for file in test_processed:
    print(file)
    if file.endswith('.pkl'):
        df = pd.read_pickle(file)
    
    elif file.endswith('.parquet'):
        df = pq.read_table(file).to_pandas()
    
    # add columns if missing
    if 'is_BRD4' not in df.columns:
        df['is_BRD4'] = df['protein_name'] == 'BRD4'
        df['is_HSA'] = df['protein_name'] == 'HSA'
        df['is_sEH'] = df['protein_name'] == 'sEH'    

    batch_size = 4096
    X_test_t = list(df['molecule_graph'])
    y_dummy = torch.tensor(np.zeros((len(X_test_t))), dtype=torch.float32)
    test_data_ = [(graph, label) for graph, label in zip(X_test_t, y_dummy)]
    test_loader = DataLoaderGCN(GraphDataset(test_data_), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    # free memory
    del df, X_test_t
    gc.collect()

   # evaluate on test data
    for inputs,_ in test_loader:
        #print('Batch: {}/{}'.format(i_batch+1, len(test_loader)))
        with torch.no_grad():
            tmp = model(inputs.to(device))
            y_test_pred.append(tmp.cpu().numpy()) 
        del tmp
        gc.collect()

    del test_loader, y_dummy, test_data_
    gc.collect()
    check_memory(pid=os.getpid())

y_test_pred = np.concatenate(y_test_pred, axis=0)
print('y_test_pred shape:', y_test_pred.shape)

submission = pd.read_csv(submission_temp_file)

# Use columns is_BRD4, is_HSA, is_sEH as mask to the correct index in y_test_pred
if not protein_name:
    mask = submission[['is_BRD4', 'is_HSA', 'is_sEH']].values.astype(bool)
    y_test_pred_masked = y_test_pred[mask]
    submission['binds'] = y_test_pred_masked
else:
    submission['binds'] = y_test_pred
    submission = submission[submission['is_{}'.format(protein_name)] == True]

submission = submission[['id', 'binds']]

# make save predictions to file
submission.to_csv(checkpoint_dir + '/test_pred{}.csv'.format(ext), index=False)

# histogram of the predictions
plt.figure(figsize=(4,4))
plt.hist(submission['binds'], bins=10)
plt.xlabel('Prediction')
plt.ylabel('Frequency')
# make y log scale
plt.yscale('log')
plt.title('Histogram of predictions')
plt.savefig('{}/histogram_test-prediction{}.png'.format(checkpoint_dir,ext), dpi=500, bbox_inches='tight')
