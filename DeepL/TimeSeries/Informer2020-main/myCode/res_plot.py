import numpy as np
import matplotlib.pyplot as plt
# setting = 'informer_ETTh1_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0'
setting = 'informer_linsData_ftMS_sl192_ll96_pl100_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0'
pred = np.load('../results/'+setting+'/pred.npy')
true = np.load('../results/'+setting+'/true.npy')

metrics = np.load('../results/'+setting+'/metrics.npy')
print(pred.shape)
print(true.shape)

print(metrics.shape)
print(metrics)

# rPred = np.load('../results/'+setting+'/real_prediction.npy')
# print(rPred.shape)
# plt.plot(rPred[0,:,-1], label='Prediction')
plt.figure()
plt.plot(true[0,:,-1], label='GroundTruth')
plt.plot(pred[0,:,-1], label='Prediction')

plt.legend()
plt.show()
