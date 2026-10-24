import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ext = '10-v1_trainsize-5000000_posrate-0.05_resampling-False_reweighting-False'
model = 'NN4'
brd4 = pd.read_csv('checkpoints_{}_BRD4_{}/test_pred_BRD4.csv'.format(model,ext))
hsa = pd.read_csv('checkpoints_{}_HSA_{}/test_pred_HSA.csv'.format(model,ext))
seh = pd.read_csv('checkpoints_{}_sEH_{}/test_pred_sEH.csv'.format(model,ext))
# brd4 = pd.read_csv('checkpoints_NN_BRD4_10-cpu_trainsize-5000000_posrate-0.05_resampling-False_reweighting-False/test_pred_BRD4.csv')
# hsa = pd.read_csv('checkpoints_NN_HSA_10_trainsize-5000000_posrate-0.05_resampling-False_reweighting-False/test_pred_HSA.csv')
# seh = pd.read_csv('../nn_morgan-1024_1M/checkpoints_NN_sEH_10_trainsize-1000000_posrate-0.05_resampling-False_reweighting-False/test_pred_sEH.csv')

# combine the predictions
submission = pd.concat([brd4, hsa, seh], axis=0)
# sort by column id
submission = submission.sort_values('id')
submission.to_csv('submission_{}_{}.csv'.format(model,ext), index=False)

print('range of prediction: [{}, {}]'.format(submission['binds'].min(), submission['binds'].max()))
# histogram of the predictions
plt.figure(figsize=(4,4))
plt.hist(submission['binds'], bins=10)
plt.xlabel('Prediction')
plt.ylabel('Frequency')
# make y log scale
plt.yscale('log')
plt.title('Histogram of predictions')
plt.savefig('histogram_{}_{}.png'.format(model,ext), dpi=500, bbox_inches='tight')
