from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
import utils
import time
import copy
from scipy.ndimage import gaussian_filter
import h5py
import sys

def gridSimulation(Ndendrites, Ng, sigma, pxs, repeats, input_positions, weights, save_spike_train, file_name):
    # Total number of dendrites
    Ndendrites2 = Ndendrites**2

    # Set up input locations
    spatialns = utils.CoordinateSamplers(Ndendrites, sigma, distrib = 'premade', Xs = input_positions)

    # Set the rate of information from sensory to grid cells
    theta_rate = 1/10 # denominator is theta frequency used


    # Prepare plot 
    n_rows = 2
    fig = plt.figure()
    ax = []
    for y in range (n_rows):
        for z in range(Ng+1):
            ax.append(plt.subplot2grid((n_rows,Ng+1),(y,z)))

    xs = np.linspace(0, 1, pxs)
    ys = np.linspace(0, 1, pxs)
    X = np.reshape(np.meshgrid(xs, ys, indexing='xy'), (2, -1)).T
    X = np.repeat(X, repeats, axis = 0)
    
    # Input layer setup
    filter = 18

    current_time = time.time()

    # Precalculate the entire firing of the spike generator group (to avoid having to restart runs when positions update):
    print("Precalculate spatial input:")

    activity = np.round(spatialns.dist(X)/sigma*10 + np.max(2*np.random.rand(pxs**2, Ndendrites2)-1, 0), 1)
    act_indices = np.where(activity < filter)
    activation_times = activity[act_indices] + 100 * act_indices[0]
    neuron_indices = act_indices[1]

    input_layer = SpikeGeneratorGroup(Ndendrites2, neuron_indices, activation_times*ms)
    print(f"Time: {time.time() - current_time}")

    #Grid layer and inhibitory layer:

    tau_g = 10*ms
    grid_eq = '''dv/dt = - v / tau_g : 1 (unless refractory)'''

    grid_layer = NeuronGroup(Ng, grid_eq, threshold = 'v > 1.0', reset = 'v = -0.1', refractory= 30*ms, method = 'exact')

    # Set up synapses from input to grid layer with STDP learning rule and randomized start weights

    input_weights = Synapses(input_layer, grid_layer, '''
                w : 1
                ''',
                on_pre='''
                v_post += w
                ''',
                delay = 3*ms)
    input_weights.connect()
    input_weights.w = weights

    # # Set up inhibitory layer:
    inhibit_layer = NeuronGroup(Ng, grid_eq, threshold='v > 0.5', refractory = 0*ms, reset = 'v = 0', method = 'exact')

    grid_to_inhibit = Synapses(grid_layer, inhibit_layer, 'w : 1', on_pre = 'v_post += w', delay = 0.6*ms)
    grid_to_inhibit.connect(condition = 'i==j')
    grid_to_inhibit.w = 0.7

    inhibit_to_grid = Synapses(inhibit_layer, grid_layer, 'w : 1', on_pre = 'v_post = -7')
    inhibit_to_grid.connect(condition = 'i!=j')
    inhibit_to_grid.w = 2


    @network_operation(dt = repeats*pxs*100*ms)
    def printProgress(t):
        progress = t / (repeats*pxs*100*ms) 
        sys.stdout.write(f"\rProgress: {progress} / {pxs}")
        sys.stdout.flush()

    print("Initialize done")

    G = SpikeMonitor(grid_layer)
    run(len(X)*100*ms)

    if save_spike_train:
        np.savez_compressed(f"grid_simulation/Results/{file_name}",  \
            spike_train = G.spike_trains(), \
            positions = X)
    print()
    return G