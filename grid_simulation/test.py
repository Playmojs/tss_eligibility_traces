import utils
import numpy as np
import matplotlib.pyplot as plt
import LIF_theta_GJ_model2_NL as GJ_NL
import time
import plotter

# with np.load(f"grid_simulation/Results/data/ThetaMSimuls/regular9.npz", allow_pickle=True) as data:
#     spike_trains = data['spike_times'].item()
#     Ndendrites = data['Ndendrites']
#     Ng = data['Ng']
#     sigma = data['sigma']
#     weights = data['weights']
#     save_tick = data['save_tick']
#     input_positions = data['input_pos']
#     baseline = data['baseline_effect']
#     Apost = data['apost']
#     wmax = data['wmax']

# weights_t = weights[540]
# grid_weights = np.moveaxis(np.reshape(weights, (len(weights), Ndendrites, Ndendrites, Ng)), 3, 1)

# start_time = time.time()
# auto_corr1 = utils.autoCorr(grid_weights)
# grid_scores1, _ = utils.gridnessScore(auto_corr1, Ndendrites, sigma)
# time_1 = time.time() - start_time
# print(time_1)
# print(grid_scores1.shape)

with np.load(f"grid_simulation/Results/test.npz", allow_pickle=True) as data:
    spike_trains = data['spike_times'].item()
    Ndendrites = data['Ndendrites']
    Ng = data['Ng']
    sigma = data['sigma']
    c = data['weights']
    save_tick = data['save_tick']
    input_positions = data['input_pos']
    baseline = data['baseline_effect']
    Apost = data['apost']
    wmax = data['wmax']

Z = GJ_NL.gridSimulation(Ndendrites, Ng, sigma, 10, 1, input_positions, c[3], True, 'test2')
with np.load('grid_simulation/Results/test2.npz', allow_pickle=True) as data:
    spikes = data['spike_train'].item()
    pos = data['positions']
plotter.gridPlotFromSpikeData(spikes, pos, 10, 10, sigma, Ndendrites, 10, 100, 'Test', True, np.moveaxis(np.reshape(c[3], (Ng, Ndendrites **2)), 0, 1))
plt.show()