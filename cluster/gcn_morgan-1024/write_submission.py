import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

model = 'NN4'
ext = 'posrate-0.050_size-5.0M_{}_00_1024bits_invar0bits_resampling-False_reweighting-False'.format(model)

brd4 = pd.read_csv('checkpoints_BRD4_{}/test_pred_fold-0.csv'.format(ext))
hsa = pd.read_csv('checkpoints_HSA_{}/test_pred_fold-0.csv'.format(ext))
seh = pd.read_csv('checkpoints_sEH_{}/test_pred_fold-0.csv'.format(ext))

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
plt.savefig('submission_histogram_{}_{}.png'.format(model,ext), dpi=500, bbox_inches='tight')
