import os
import matplotlib.pyplot as plt
import utils
import sys
import numpy as np
from brian2 import second
from scipy.ndimage import gaussian_filter

pxs = 48
times = ['0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90', '95']
# times = ['10', '20', '30', '40', '50', '60', '70', '80', '90']
appendix = 'min_Spikes.npz'
base_path = 'grid_simulation/Results/data/multi-grid'
sub_dirs = utils.getSortedEntries(base_path, 'directory', True)

n_simuls = np.array([30, 22]) #len(sub_dirs) // n_groups
n_groups = len(n_simuls)
n_times = len(times)
ngs = np.array([37, 23])

skip_calc = True

legends = np.empty(n_groups, dtype = object)
multi_hists = np.empty((n_groups, np.max(n_simuls), n_times, np.max(ngs), pxs, pxs))

for j, sub_dir in enumerate(sub_dirs):
    # Find which group and simul number within group this j is:
    j1 = 0
    j2 = j
    while j2 - n_simuls[j1] >= 0:
        j2 -= n_simuls[j1]
        j1 += 1

    sys.stdout.write(f"\rProgress: {j + 1} / {np.sum(n_simuls)}")
    sys.stdout.flush()
    legends[j1] = sub_dir
    if skip_calc:
        continue
    for i, time in enumerate(times):
        with np.load(sub_dir + '/' + time + appendix, allow_pickle = True) as data:
            spike_trains = data['spike_train'].item()
            X = data['positions']
        for z in range(ngs[j1]):
            spike_times = spike_trains[z]/second
            spike_indices = np.floor(spike_times*int(1000/100))
            spike_positions = X[np.ndarray.astype(spike_indices, int)]
            spike_hist, _, __ = np.histogram2d(spike_positions[:,1], spike_positions[:,0], pxs, [[0,1],[0,1]])
            gauss_spike_hist = gaussian_filter(spike_hist, 2)
            multi_hists[j1, j2, i, z] = gauss_spike_hist
print("\n")
if not skip_calc:
    print("Initiate auto-correlation")
    corr_gauss = utils.autoCorr(multi_hists)
    print("Initiate gridness scores")
    gauss_gscores, _ = utils.gridnessScore(corr_gauss, pxs, 0.12)
    np.savez("grid_simulation/Results/multi-grid", gscores = gauss_gscores)

gscores = np.load('grid_simulation/Results/multi-grid.npz')['gscores']
mean_gscores = np.nanmean(gscores, axis = (1,3))
var_gscores = np.nanvar(gscores, axis = (1,3)) / (n_simuls * ngs)[:,np.newaxis]
leg = ["37", "23"]
for line_mean, line_var, legend in zip(mean_gscores, var_gscores, leg):
    plt.plot(times, line_mean)
    plt.legend(legend)
    plt.fill_between(times, line_mean + np.sqrt(line_var), line_mean - np.sqrt(line_var), alpha=0.3, label = '_nolegend_')
# plt.plot(np.mean(gscores[0, ...], axis = 2).T)
# plt.legend(np.arange(25))

plt.show()

fig, ax = plt.subplots(1, len(gscores))
for i in range(len(gscores)):
    hist, edge = np.histogram(gscores[i,:,-1,:], range = [-1,1.5])
    ax[i].plot(edge[1:] - (edge[1:] - edge[:-1]) / 2, hist)

plt.show()