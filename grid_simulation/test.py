import utils
import numpy as np
import time

with np.load(f"grid_simulation/Results/data/ThetaMSimuls/regular9.npz", allow_pickle=True) as data:
    spike_trains = data['spike_times'].item()
    Ndendrites = data['Ndendrites']
    Ng = data['Ng']
    sigma = data['sigma']
    weights = data['weights']
    save_tick = data['save_tick']
    input_positions = data['input_pos']
    baseline = data['baseline_effect']
    Apost = data['apost']
    wmax = data['wmax']

weights_t = weights[540]
grid_weights = np.moveaxis(np.reshape(weights, (len(weights), Ndendrites, Ndendrites, Ng)), 3, 1)

start_time = time.time()
auto_corr1 = utils.autoCorr(grid_weights)
grid_scores1, _ = utils.gridnessScore(auto_corr1, Ndendrites, sigma)
time_1 = time.time() - start_time
print(time_1)
print(grid_scores1.shape)


# auto_corr2 = np.zeros((Ng, 2*Ndendrites -1, 2*Ndendrites -1))
# grid_scores2 = np.zeros(Ng)
# for z in range(Ng):
#     auto_corr_z = utils.normcorr2d(grid_weights[z])
#     grid_scores2[z], _ = utils.gridness_score(auto_corr_z, Ndendrites, sigma)
# time_2 = time.time() - start_time + time_1

# print (time_2, time_1)
# print (grid_scores1, grid_scores2)