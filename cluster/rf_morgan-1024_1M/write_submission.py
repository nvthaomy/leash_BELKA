import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ext = '0_trainsize-5000_posrate-0.1_resampling-True_reweighting-False'

brd4 = pd.read_csv('checkpoints_NN_BRD4_{}/test_pred_BRD4.csv'.format(ext))
hsa = pd.read_csv('checkpoints_NN_HSA_{}/test_pred_HSA.csv'.format(ext))
seh = pd.read_csv('checkpoints_NN_sEH_{}/test_pred_sEH.csv'.format(ext))


# combine the predictions
submission = pd.concat([brd4, hsa, seh], axis=0)
# sort by column id
submission = submission.sort_values('id')
submission.to_csv('submission_{}.csv'.format(ext), index=False)

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