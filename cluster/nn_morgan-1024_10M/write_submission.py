import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ext = '10_trainsize-10000000_posrate-0.05_resampling-False_reweighting-False'
model = 'NN5'
brd4 = pd.read_csv('checkpoints_{}_BRD4_10_trainsize-10000000_posrate-0.0457_resampling-False_reweighting-False/test_pred_BRD4.csv'.format(model))
hsa = pd.read_csv('checkpoints_{}_HSA_10_trainsize-10000000_posrate-0.0408_resampling-False_reweighting-False//test_pred_HSA.csv'.format(model))
seh = pd.read_csv('checkpoints_{}_sEH_10_trainsize-10000000_posrate-0.05_resampling-False_reweighting-False/test_pred_sEH.csv'.format(model))

# combine the predictions
submission = pd.concat([brd4, hsa, seh], axis=0)
# sort by column id
submission = submission.sort_values('id')
submission.to_csv('submission_{}_{}.csv'.format(model,ext), index=False)
print('number of predictions: {}'.format(submission.shape[0]))
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
