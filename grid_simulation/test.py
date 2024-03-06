import utils
import numpy as np
import matplotlib.pyplot as plt
import LIF_theta_GJ_model2_NL as GJ_NL
import LIF_theta_BVC_model3_NL as BVC_NL
import plotter
import sys
from brian2 import ms

with np.load("grid_simulation/Results/test2.npz", allow_pickle= True) as data:
    print(data["spike_train"].item())
    print(data["positions"])