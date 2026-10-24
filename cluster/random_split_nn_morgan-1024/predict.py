import pandas as pd
import numpy as np
import torch
from tools import * 
import gc
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import os 

protein_name = 'BRD4'
ext='_fold-0'

load_model = True
checkpoint_dir = 'checkpoints_BRD4_posrate-0.046_size-10.0M_NN4_00_1024bits_invar0bits_resampling-False_reweighting-False'
checkpoint_file = ''
model = None 

model_file = 'model_fold-0.pth'
test_processed = ['../split1_separate-train_nn_morgan-1024/test_1024bits_chunk1.parquet',
                  '../split1_separate-train_nn_morgan-1024/test_1024bits_chunk2.parquet',
                  '../split1_separate-train_nn_morgan-1024/test_1024bits_chunk3.parquet']

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
        if not os.path.exists(file.replace('.pkl','.parquet')):
            df.to_parquet(file.replace('.pkl','.parquet'))
    
    elif file.endswith('.parquet'):
        df = pq.read_table(file).to_pandas()
    
    # add columns if missing
    if 'is_BRD4' not in df.columns:
        df['is_BRD4'] = df['protein_name'] == 'BRD4'
        df['is_HSA'] = df['protein_name'] == 'HSA'
        df['is_sEH'] = df['protein_name'] == 'sEH'    

    X_test = np.array(list(df['molecule_fp']))
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    # free memory
    del df, X_test
    gc.collect()
    check_memory(pid=os.getpid())

   # evaluate on test data
    batch_size = 256
    for i in range(0, len(X_test_t), batch_size):
        with torch.no_grad():
            tmp = model(X_test_t[i:i+batch_size])
            y_test_pred.append(tmp.cpu().numpy()) 
        del tmp
        gc.collect()

    del X_test_t
    gc.collect()

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
