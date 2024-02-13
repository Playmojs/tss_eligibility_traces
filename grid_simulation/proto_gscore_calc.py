import os
import matplotlib.pyplot as plt
import utils
import sys
import numpy as np
from brian2 import second
from scipy.ndimage import gaussian_filter

pxs = 48
times = ['10', '20', '30', '40', '50', '60', '70', '80', '90']
appendix = 'min_Spikes.npz'
base_path = 'grid_simulation/Results/data/GJ_tests'
sub_dirs = utils.getSortedEntries(base_path, 'directory', True)

n_groups = 7
n_simuls = 1 #len(sub_dirs) // n_groups
n_times = len(times)
ng = 13
skip_calc = False

legends = np.empty(n_groups, dtype = object)
multi_hists = np.empty((n_groups, n_simuls, n_times, ng, pxs, pxs))

for j, sub_dir in enumerate(sub_dirs):
    j1 = j // n_simuls
    j2 = j % n_simuls
    sys.stdout.write(f"\rProgress: {j + 1} / {n_groups * n_simuls}")
    sys.stdout.flush()
    legends[j1] = sub_dir
    if skip_calc:
        continue
    for i, time in enumerate(times):
        with np.load(sub_dir + '/' + time + appendix, allow_pickle = True) as data:
            spike_trains = data['spike_train'].item()
            X = data['positions']
        for z in range(13):
            x = 5
            spike_times = spike_trains[z]/second
            spike_indices = np.floor(spike_times*int(1000/100))
            spike_positions = X[np.ndarray.astype(spike_indices, int)]
            spike_hist, _, __ = np.histogram2d(spike_positions[:,1], spike_positions[:,0], pxs, [[0,1],[0,1]])
            gauss_spike_hist = gaussian_filter(spike_hist, 2)
            multi_hists[j1, j2, i, z] = gauss_spike_hist
if not skip_calc:
    print("Initiate auto-correlation")
    corr_gauss = utils.autoCorr(multi_hists)
    print("Initiate gridness scores")
    gauss_gscores, _ = utils.gridnessScore(corr_gauss, pxs, 0.12)
    np.savez("grid_simulation/Results/GJ_gscores", gscores = gauss_gscores)

gscores = np.load('grid_simulation/Results/GJ_gscores.npz')['gscores']
mean_gscores = np.nanmean(gscores, axis = (1,3))
var_gscores = np.nanvar(gscores, axis = (1,3)) / (n_simuls * ng)
#plot_gscore = np.reshape(gscores, (n_groups*n_simuls, n_times, ng))

plt.plot(times, mean_gscores.T)
#plt.fill_between(times, mean_gscores.T + np.sqrt(var_gscores.T), mean_gscores.T - np.sqrt(var_gscores.T), alpha = 0.3)
plt.legend(np.arange(5))
plt.show()
