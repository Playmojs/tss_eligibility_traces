import utils
import numpy as np
import matplotlib.pyplot as plt
import LIF_theta_GJ_model2_NL as GJ_NL
import LIF_theta_BVC_model3_NL as BVC_NL
import plotter
import sys
from brian2 import ms

path = "grid_simulation/Results/data/multi-grid/Regular0"
appendix = "min_Spikes.npz"

times = ["0","5","10", "15","20","25","30", "35","40","45","50", "55","60","65","70", "75","80","85","90", "95"]
n_times = len(times)

ng = 37

all_spikes = {}

for i, time in enumerate(times):
    spike_trains = np.load(path + "/" + time + appendix, allow_pickle= True)["spike_train"].item()
    spike_data = {}
    for z, spike_train in enumerate(spike_trains):
        spike_data[z] = spike_train / ms
    all_spikes[time + "_minutes"] = spike_data
np.savez_compressed("grid_simulation/Results/37grid_spike_times", spike_data = all_spikes)
