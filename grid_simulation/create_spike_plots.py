import numpy as np
import sys
import matplotlib.pyplot as plt
import LIF_theta_model3_NL
import utils
import plotter

file_base = 'regular0'
BVC = False
root_dir = 'grid_simulation/Results/data/simspam'
with np.load(f"{root_dir}/{file_base}.npz", allow_pickle=True) as data:
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

b = baseline*Ng*Ndendrites
print(b)
print(Ng)
print(wmax)
print(Apost)

time = 600 #time in minutes for simulation
weights_index = int(time * 100 / save_tick)
reps = 5

plt.scatter(input_positions[:, 0], input_positions[:, 1])
plt.axis('square')

# spike_trains = LIF_theta_model3_NL.gridSimulation(Ndendrites, Ng, sigma, 48, reps, input_positions, weights[weights_index, ...], True, "data/ThetaMSimuls/regular0_70minSpikes")

for i in range (0, 100, 20):
    with np.load(f"{root_dir}/{file_base}/{i}min_Spikes.npz", allow_pickle=True) as data:
        spike_trains2 = data['spike_train'].item()
        pos = data['positions']
    time2 = int(reps*48*48/10)
    time = i * 600
    # if (BVC):
    #     pos += 0.5
    plotter.gridPlotFromSpikeData(spike_trains2, pos, time2, time2, 0.108, Ndendrites, 48, 100, title = f"{b}_baseline: {time // 60} minutes {time % 60} seconds", plot_weights = False, weights = np.moveaxis(np.reshape(weights[weights_index], (Ng, Ndendrites **2)), 0, 1))

plt.show()