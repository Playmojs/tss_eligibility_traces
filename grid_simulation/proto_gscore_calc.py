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
base_path = 'grid_simulation/Results/data/noisy_white_params'
str_sep = 'noisy_white'

sub_dirs = os.walk(base_path)

n_groups = 5
n_simuls = len(next(sub_dirs)[1]) // n_groups
n_times = len(times)
ng = 13

gscores = np.zeros((n_groups, n_simuls, n_times, ng))
legends = np.empty(n_groups, dtype = object)
for j, sub_dir in enumerate(sub_dirs):
    str_app = sub_dir[0].split('\\')[-1]
    iterator_number = int(str_app.split(str_sep)[-1])
    j1 = iterator_number // n_groups
    j2 = iterator_number % n_groups
    sys.stdout.write(f"\rProgress: {j + 1} / {n_groups * n_simuls}")
    sys.stdout.flush()
    legends[j1] = str_app
    for i, time in enumerate(times):
        with np.load(sub_dir[0] + '/' + time + appendix, allow_pickle = True) as data:
            spike_trains = data['spike_train'].item()
            X = data['positions']
        for z in range(13):
            x = 5
            spike_times = spike_trains[z]/second
            spike_indices = np.floor(spike_times*int(1000/100))
            spike_positions = X[np.ndarray.astype(spike_indices, int)]
            spike_hist, _, __ = np.histogram2d(spike_positions[:,1], spike_positions[:,0], pxs, [[0,1],[0,1]])
            gauss_spike_hist = gaussian_filter(spike_hist, 1)
            corr_gauss = utils.normcorr2d(gauss_spike_hist)
            gauss_gscore, _ = utils.gridness_score(corr_gauss, pxs, 0.1)
            gscores[j1, j2, i, z] = gauss_gscore
np.savez("grid_simulation/Results/noisy_white_gscores", gscores = gscores)

gscores = np.load('grid_simulation/Results/noisy_white_gscores.npz')['gscores']
mean_gscores = np.nanmean(gscores, axis = (1,3))
var_gscores = np.nanvar(gscores, axis = (1,3)) / (n_simuls * ng)

plt.plot(times, mean_gscores.T)
#plt.fill_between(times, mean_gscores.T + np.sqrt(var_gscores.T), mean_gscores.T - np.sqrt(var_gscores.T))
plt.legend(legends)
plt.show()

# pxs = 48
# times = ['10', '20', '30', '40', '50', '60', '70', '80', '90']
# appendix = 'min_Spikes.npz'
# base_path = 'grid_simulation/Results/data/80vs100bl/noisy_white'
# apps = [0,1,2,3,4,5,6,7,8,9]
# # #fig = plt.figure()
# n_simuls = len(apps)
# gscores = np.zeros((3, n_simuls, len(times), 13))
# legends = np.empty(n_simuls, dtype = object)
# for j, simul in enumerate(apps):
#     sub_dir = base_path + str(simul)
#     legends[j] = sub_dir
#     sys.stdout.write(f"\rProgress: {j + 1} / {n_simuls}")
#     sys.stdout.flush()
#     for i, time in enumerate(times):
#         z = sub_dir + '/' + time + appendix
#         with np.load(sub_dir + '/' + time + appendix, allow_pickle = True) as data:
#             spike_trains = data['spike_train'].item()
#             X = data['positions']
#         for z in range(13):
#             spike_times = spike_trains[z]/second
#             spike_indices = np.floor(spike_times*int(1000/100))
#             spike_positions = X[np.ndarray.astype(spike_indices, int)]
#             spike_hist, _, __ = np.histogram2d(spike_positions[:,1], spike_positions[:,0], pxs, [[0,1],[0,1]])
#             gauss_spike_hist = gaussian_filter(spike_hist, 1)
#             corr_gauss = utils.normcorr2d(gauss_spike_hist)
#             gauss_gscore, _ = utils.gridness_score(corr_gauss, pxs, 0.1)
#             gscores[j, i, z] = gauss_gscore
# np.savez("grid_simulation/Results/80vs100_gscores_white_noise", gscores = gscores)
# prefix = 'grid_simulation/Results/80vs100_'
# files = ['gscores', 'gscores_blue_noise', 'gscores_white_noise']
# for i, file in enumerate(files):
#     gscores[i] = np.load(prefix + f'{file}' + '.npz')['gscores']

# mean_gscores = np.mean(gscores, axis = 3)
# means1 = np.mean(mean_gscores[:, 0:5, :], axis = 1)
# means2 = np.mean(mean_gscores[:, 5:10, :], axis = 1)

# use_simul_var = False
# if use_simul_var:
#     var1 = np.var(mean_gscores[:, 0:5, ...], axis = 1) / 5
#     var2 = np.var(mean_gscores[:, 5:10, ...], axis = 1) / 5
# else:
#     var1 = np.var(gscores[:, 0:5, ...], axis = (1,3)) / (5*13)
#     var2 = np.var(gscores[:, 5:10, ...], axis = (1,3)) / (5*13)

# for i in range(3):
#     plt.plot(times, means1[i])
#     plt.plot(times, means2[i])
#     plt.fill_between(times, means1[i] - np.sqrt(var1[i]), means1[i] + np.sqrt(var1[i]), alpha = 0.1, label = '_nolegend_')
#     plt.fill_between(times, means2[i] - np.sqrt(var2[i]), means2[i] + np.sqrt(var2[i]), alpha = 0.1, label = '_nolegend_')

# plt.legend(["80 bl regular", "120 bl regular", "80 bl blue noise", "120 bl blue noise", "80 bl white noise", "120 bl white noise",])
# plt.show()