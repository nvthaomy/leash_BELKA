import pandas as pd
import numpy as np
import torch
from tools import * 
import gc
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

model = Net4b(n_features=1024)
checkpoint_dir = 'checkpoints_NN4b_0_1024bits_resampling-False_reweighting-False'
chekcpoint_file = 'checkpoint_epoch_16.pth'
test_processed = ['../split1_separate-train_nn_morgan-1024/test_1024bits_chunk1.parquet',
                  '../split1_separate-train_nn_morgan-1024/test_1024bits_chunk2.parquet',
                  '../split1_separate-train_nn_morgan-1024/test_1024bits_chunk3.parquet']

submission_temp_file = './test_ids.csv'

# Load test features
print('\nLoading processed test data from pickle/parquet files')
X_test = []

for file in test_processed:
    print(file)
    if file.endswith('.pkl'):
        df = pd.read_pickle(file)
        if not os.path.exists(file.replace('.pkl','.parquet')):
            df.to_parquet(file.replace('.pkl','.parquet'))
    
    elif file.endswith('.parquet'):
        df = pq.read_table(file).to_pandas()
    
    X_test.append(np.array(list(df['molecule_fp'])))
  
    # free memory
    del df
    gc.collect()

    check_memory(pid=os.getpid())

gc.collect()

X_test = np.concatenate(X_test)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
print('X_test shape:', X_test.shape)

del X_test
gc.collect()

# Load model
checkpoint = torch.load(checkpoint_dir + '/' + chekcpoint_file)
model.load_state_dict(checkpoint['model_state_dict'])
loss = checkpoint['loss']
print(f'Predict using model with loss: {loss}')

model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_t)
    y_test_pred = y_test_pred.cpu().numpy() # this is a (n_samples, 3) array

submission = pd.read_csv(submission_temp_file)
# Use columns is_BRD4, is_HSA, is_sEH as mask to the correct index in y_test_pred
mask = submission[['is_BRD4', 'is_HSA', 'is_sEH']].values.astype(bool)
y_test_pred_masked = y_test_pred[mask]
submission['binds'] = y_test_pred_masked
submission = submission[['id', 'binds']]

# make save predictions to file
submission.to_csv(checkpoint_dir + '/test_pred.csv', index=False)

# histogram of the predictions
plt.figure(figsize=(4,4))
plt.hist(submission['binds'], bins=10)
plt.xlabel('Prediction')
plt.ylabel('Frequency')
# make y log scale
plt.yscale('log')
plt.title('Histogram of predictions')
plt.savefig('{}/histogram_test-prediction.png'.format(checkpoint_dir), dpi=500, bbox_inches='tight')
