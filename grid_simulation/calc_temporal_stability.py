import matplotlib.pyplot as plt
import utils
import sys
import numpy as np

pxs = 48
times = ['0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90', '95']
# times = ['10', '20', '30', '40', '50', '60', '70', '80', '90']
# times = np.linspace(0, 100, 20, False)
appendix = 'min_Spikes.npz'
base_path = 'grid_simulation/Results/data/'
simulation = 'GJ_model'
sub_dirs = utils.getSortedEntries(base_path + simulation, 'directory', True)

n_simuls = np.array([30]) #len(sub_dirs) // n_groups
n_groups = len(n_simuls)
n_times = len(times)
Ndendrites = 24
ngs = np.array([13])

skip_calc = False

legends = np.empty(n_groups, dtype = object)
multi_hists = np.empty((n_groups, np.max(n_simuls), n_times, np.max(ngs), pxs, pxs))
shuffled_hists = np.empty((n_groups, np.max(n_simuls), n_times, np.max(ngs), pxs, pxs))
pairwise_var = np.empty((n_groups, np.max(n_simuls), n_times - 1, np.max(ngs)))
pairwise_shuffled_var = np.empty((n_groups, np.max(n_simuls), n_times - 1, np.max(ngs)))

indices_y = np.arange(48)
indices_x = np.arange(48)

for j, sub_dir in enumerate(sub_dirs):
    # Find which group and simul number within group this j is:
    j1 = 0
    j2 = j
    while j2 - n_simuls[j1] >= 0:
        j2 -= n_simuls[j1]
        j1 += 1

    sys.stdout.write(f"\rLoading data: {j + 1} / {np.sum(n_simuls)}")
    sys.stdout.flush()
    legends[j1] = sub_dir
    if skip_calc:
        continue
    for i, time in enumerate(times):
        multi_hists[j1, j2, i, :ngs[j1]] = utils.getPopulationSpikePlot(sub_dir + '/' + str(time) + appendix, ngs[j1], pxs, True)
        for z in range(ngs[j1]):
            np.random.shuffle(indices_y)
            np.random.shuffle(indices_x)
            shuff_arr = multi_hists[j1, j2, i, z, indices_y, :]
            shuff_arr = shuff_arr[:, indices_x]
            shuffled_hists[j1, j2, i, z] = shuff_arr

if not skip_calc:
    for i in range(n_times - 1):
        pairwise_var[:, :, i] = np.nanmean(np.nanvar(multi_hists[:, :, i:i+2], axis = 2), axis = (-2, -1))
        pairwise_shuffled_var[:, :, i] = np.nanmean(np.nanvar(shuffled_hists[:, :, i:i+2], axis = 2), axis = (-2, -1))

    temp_var = np.nanvar(multi_hists[:, :, 10:], axis = -4)
    shuff_var = np.nanvar(shuffled_hists[:, :, 10:], axis = -4)
    
    temp_stability = np.nanmean(temp_var, axis = (-2, -1)) * 4
    shuff_stability = np.nanmean(shuff_var, axis = (-2, -1)) * 4
    np.savez(f"grid_simulation/Results/analysis/{simulation}/temporal_stability", \
            temporal_stability = temp_stability, \
            shuffled_stability = shuff_stability, \
            pairwise_var = pairwise_var, \
            pairwise_shuffled_var = pairwise_shuffled_var)
    
with np.load(f"grid_simulation/Results/analysis/{simulation}/temporal_stability.npz") as data:
    temporal_stability = data["temporal_stability"]
    shuffled_stability = data["shuffled_stability"]
    pairwise_var = data["pairwise_var"]
    pairwise_shuffled_var = data["pairwise_shuffled_var"]

pairwise_mean = np.nanmean(pairwise_var, axis = (1, 3))
pairwise_shuffled_mean = np.nanmean(pairwise_shuffled_var, axis = (1, 3))

# x_vals = np.arange(n_groups)
# x_temp = x_vals - 0.3
# x_shuff = x_vals + 0.3
# plt.bar(x_temp, np.mean(temporal_stability, axis = (-2, -1)))
# plt.bar(x_shuff, np.mean(shuffled_stability, axis = (-2, -1)))

plt.plot(pairwise_mean[0, 1:])
plt.plot(pairwise_shuffled_mean[0, 1:])
plt.ylim([0, None])

plt.show()