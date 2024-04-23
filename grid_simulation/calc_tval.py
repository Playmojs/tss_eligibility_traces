import numpy as np
import scipy.stats as stats

gscores = np.load("grid_simulation/Results/analysis/simspam/gscores4.npz")["gscores"]

reg_vals = np.ndarray.flatten(gscores[2,:,-1,:])
bn_vals = np.ndarray.flatten(gscores[0,:,-1,:])
wn_vals = np.ndarray.flatten(gscores[1,:,-1,:])

reg_means = np.mean(gscores[2, :, -1], axis = -1)
bn_means = np.mean(gscores[0, :, -1], axis = -1)
wn_means = np.mean(gscores[1, :, -1], axis = -1)

reg_bn_tval = stats.ttest_ind(reg_vals, bn_vals)
reg_wn_tval = stats.ttest_ind(reg_vals, wn_vals)
bn_wn_tval = stats.ttest_ind(bn_vals, wn_vals)

reg_bn_mean_tval = stats.ttest_ind(reg_means, bn_means)
reg_wn_mean_tval = stats.ttest_ind(reg_means, wn_means)
bn_wn_mean_tval = stats.ttest_ind(bn_means, wn_means)

x = 7