import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ext = '10_trainsize-1000000_posrate-0.05_resampling-False_reweighting-False'
threshold = 0.7

brd4 = pd.read_csv('checkpoints_NN_BRD4_{}/test_pred_BRD4.csv'.format(ext))
hsa = pd.read_csv('checkpoints_NN_HSA_{}/test_pred_HSA.csv'.format(ext))
seh = pd.read_csv('checkpoints_NN_sEH_{}/test_pred_sEH.csv'.format(ext))


# combine the predictions
submission = pd.concat([brd4, hsa, seh], axis=0)
# sort by column id
submission = submission.sort_values('id')
submission.to_csv('submission_{}.csv'.format(ext), index=False)

# copy submission to submission_threshold
submission_threshold = submission.copy()
# apply threshold, if binds > threshold, binds = 1, else binds = 0
submission_threshold['binds'] = np.where(submission_threshold['binds'] > threshold, 1, 0)
submission_threshold.to_csv('submission_{}-threshold_{}.csv'.format(threshold, ext), index=False)

print('range of prediction: [{}, {}]'.format(submission['binds'].min(), submission['binds'].max()))

# histogram of the predictions
plt.figure(figsize=(4,4))
plt.hist(submission['binds'], bins=10)
plt.xlabel('Prediction')
plt.ylabel('Frequency')
# make y log scale
plt.yscale('log')
plt.title('Histogram of predictions')
plt.savefig('histogram_{}.png'.format(ext), dpi=500, bbox_inches='tight')

# histogram of the predictions with threshold
plt.figure(figsize=(4,4))
plt.hist(submission_threshold['binds'], bins=10)
plt.xlabel('Prediction')
plt.ylabel('Frequency')
# make y log scale
plt.yscale('log')
plt.title('Histogram of predictions with threshold')
plt.savefig('histogram_{}-threshold_{}.png'.format(threshold, ext), dpi=500, bbox_inches='tight')