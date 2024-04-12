import os
import matplotlib.pyplot as plt
import utils
import sys
import numpy as np
from brian2 import second
from scipy.ndimage import gaussian_filter

pxs = 48
# times = ['0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90', '95']
# times = ['10', '20', '30', '40', '50', '60', '70', '80', '90']
times = np.linspace(0, 100, 20, False)
appendix = 'min_Spikes.npz'
base_path = 'grid_simulation/Results/'
simulation = 'noise_sims2'
sub_dirs = utils.getSortedEntries(base_path +'data/' + simulation, 'directory', True)

n_groups = 2
n_simuls = 30 #len(sub_dirs) // n_groups
n_times = len(times)
ngs = np.array([13, 13])
sigma = 0.12
sigmas = np.linspace(0.08, 0.13, 20)
skip_calc = False

multi_hists = np.empty((n_groups, n_simuls, np.max(ngs), pxs, pxs))

for j, sub_dir in enumerate(sub_dirs):
    if skip_calc:
        continue
    j1 = j // n_simuls
    j2 = j % n_simuls
    sys.stdout.write(f"\rProgress: {j + 1} / {n_groups * n_simuls}")
    sys.stdout.flush()
    
    with np.load(sub_dir + '/' + str(times[-1]) + appendix, allow_pickle = True) as data:
        spike_trains = data['spike_train'].item()
        X = data['positions']
    for z in range(ngs[j1]):
        spike_times = spike_trains[z]/second
        spike_indices = np.floor(spike_times*int(1000/100))
        spike_positions = X[np.ndarray.astype(spike_indices, int)]
        spike_hist, _, __ = np.histogram2d(spike_positions[:,1], spike_positions[:,0], pxs, [[0,1],[0,1]])
        gauss_spike_hist = gaussian_filter(spike_hist, 2)
        multi_hists[j1, j2, z] = gauss_spike_hist
print("\n")
if not skip_calc:
    sigma_gscores = np.empty((n_groups, n_simuls, len(sigmas), np.max(ngs)))
    print("Compute auto-correlation")
    corr_gauss = utils.autoCorr(multi_hists)
    print("Compute gridness scores")
    for i, sig in enumerate(sigmas):
        sigma_gscores[:,:, i], _ = utils.gridnessScore(corr_gauss, pxs, sig)
    np.savez_compressed(base_path + 'analysis/' + simulation + "/sigmas.npz", sigma_gscores = sigma_gscores, sigmas = sigmas)


with np.load(base_path + 'analysis/' + simulation + "/sigmas.npz", allow_pickle= True) as data:
    sigma_gscores = data['sigma_gscores']
    sigmas = data['sigmas']

mean_gscores = np.nanmean(sigma_gscores, axis = (1,3))
var_gscores = np.nanvar(sigma_gscores, axis = (1,3)) / (n_simuls * ngs)[:,np.newaxis]
legends = np.arange(2)

# max_gscore_ind = np.nanargmax(sigma_gscores, axis = 2).reshape(n_groups, -1)
# sigma_max = sigmas[max_gscore_ind]
# plt.hist(sigma_max[2], 10)

for line_mean, line_var, legend in zip(mean_gscores, var_gscores, legends):
    plt.plot(sigmas, line_mean)
    plt.legend(str(legend))
    plt.fill_between(sigmas, line_mean + np.sqrt(line_var), line_mean - np.sqrt(line_var), alpha=0.3, label = '_nolegend_')


plt.show()
