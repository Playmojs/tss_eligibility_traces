import utils
import numpy as np
import matplotlib.pyplot as plt
import LIF_theta_GJ_model2_NL as GJ_NL
import LIF_theta_BVC_model3_NL as BVC_NL
import plotter
import sys
from brian2 import ms

pxs = 48
times = ['0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90', '95']
appendix = 'min_Spikes.npz'
base_path = 'grid_simulation/Results/data/simspam'
# sub_dirs = utils.getSortedEntries(base_path, 'directory', True)
sub_dirs = [base_path + "/regular10"]

n_groups = 1
n_simuls = 1 #len(sub_dirs) // n_groups
n_times = len(times)
ng = 13
skip_calc = False

sim_data = {}

for j, sub_dir in enumerate(sub_dirs):
    sys.stdout.write(f"\rProgress: {j + 1} / {n_groups * n_simuls}")
    sys.stdout.flush()
    if skip_calc:
        continue
    for i, time in enumerate(times):
        time_data = {}
        with np.load(sub_dir + '/' + time + appendix, allow_pickle = True) as data:
            spike_trains = data['spike_train'].item()
            X = data['positions']
        for z in range(13):
            x = 5
            spike_times = spike_trains[z]/ms
            time_data[z] = spike_times
        sim_data[time + '_minutes'] = time_data
    print('\n')

np.savez_compressed("grid_simulation/Results/spike_times", spike_data = sim_data)
with np.load('grid_simulation/Results/spike_times.npz', allow_pickle=True) as data:
    sim_data = data['spike_data'].item()
    print(sim_data.keys())