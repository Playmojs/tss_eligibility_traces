import numpy as np
import sys
import matplotlib.pyplot as plt
import LIF_theta_model3_NL
import utils
import plotter

# score = np.load("grid_simulation\Results\data\mf3100_1_opt_g_score.npy")
# with np.load('grid_simulation/Results/data/m3f100_1.npz') as data:
#         old_scores = data['scores']
# plt.plot(np.mean(score, axis = 1))
# plt.plot(np.mean(old_scores, axis=1))
# plt.show()

#plotter.LinePlot(["data/ThetaMSimuls/regular0.npz"], "", False)

with np.load("grid_simulation/Results/data/ThetaMSimuls/regular0.npz", allow_pickle=True) as data:
    spike_trains = data['spike_times'].item()
    Ndendrites = data['Ndendrites']
    Ng = data['Ng']
    sigma = data['sigma']
    weights = data['weights']
    save_tick = data['save_tick']
    input_positions = data['input_pos']

time = 70 #time in minutes for simulation
weights_index = int(time * 60000 / save_tick)
reps = 5

# spike_trains = LIF_theta_model3_NL.gridSimulation(Ndendrites, Ng, sigma, 48, reps, input_positions, weights[weights_index, ...], True, "data/ThetaMSimuls/regular0_70minSpikes")

with np.load("grid_simulation/Results/data/ThetaMSimuls/regular0_70minSpikes.npz", allow_pickle=True) as data:
    spike_trains2 = data['spike_train'].item()
    pos = data['positions']
time2 = int(reps*48*48/10)

plotter.gridPlotFromSpikeData(spike_trains2, pos, time2, time2, sigma, Ndendrites, 48, 100, plot_weights = True, weights = np.reshape(weights[weights_index], (Ndendrites **2, Ng)))