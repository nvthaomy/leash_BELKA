import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

checkpoint_dir = 'checkpoints_posrate-0.027_size-15.0M_NN4_00_1024bits_invar0bits_resampling-False_reweighting-False'
csv_files = glob('{}/test_pred_fold-*.csv'.format(checkpoint_dir))

submission = pd.DataFrame()
for i,file in enumerate(csv_files):
    df = pd.read_csv(file).sort_values('id')
    if i == 0:
        submission = df[['id','binds']]
    else:
        submission['binds'] += df['binds']

submission['binds'] /= len(csv_files)

submission.to_csv('{}/test_pred_avg.csv'.format(checkpoint_dir,), index=False)

print('number of predictions: {}'.format(submission.shape[0]))
print('range of prediction: [{}, {}]'.format(submission['binds'].min(), submission['binds'].max()))

# histogram of the predictions
plt.figure(figsize=(4,4))
plt.hist(submission['binds'], bins=20)
plt.xlabel('Prediction')
plt.ylabel('Frequency')
# make y log scale
plt.yscale('log')
plt.title('Histogram of predictions')
plt.savefig('{}/histogram_test-prediction_avg.png'.format(checkpoint_dir), dpi=500, bbox_inches='tight')
