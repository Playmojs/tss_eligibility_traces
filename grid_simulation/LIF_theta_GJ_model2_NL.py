from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import tanh
import utils
from scipy.ndimage import gaussian_filter
import h5py
import sys


def gridSimulation(Ndendrites, Ng, sigma, pxs, repeats, input_positions, conductivities, save_spike_train, file_name):
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
    

    # Precalculate the entire firing of the spike generator group (to avoid having to restart runs when positions update):
    print("Precalculate spatial input:")

    # Input layer setup
    filter = 18

    activity = np.round(spatialns.dist(X)/sigma*10 + (np.random.normal(0, 2,(pxs**2, Ndendrites2))), 1)
    act_indices = np.where(activity < filter)
    activation_times = activity[act_indices] + 100 * act_indices[0]
    neuron_indices = act_indices[1]

    input_layer = SpikeGeneratorGroup(Ndendrites2, neuron_indices, activation_times*ms)

    tau_d = 10*ms
    dendrite_eq = '''dv/dt = -v/tau_d : 1
                    c : 1
                    Ve = tanh(v)*c : 1'''
    dendrite_layer = NeuronGroup(Ndendrites2 * Ng, dendrite_eq, method = 'exact')
    dendrite_layer.c = conductivities 

    input_to_dendrites = Synapses(input_layer, dendrite_layer, '''w : 1''', 
                                on_pre = '''
                                v_post +=w''')
    input_to_dendrites.connect(condition = 'i == j % Ndendrites2')
    input_to_dendrites.w = 10

    #Grid layer and inhibitory layer:

    tau_g = 10*ms
    tau_y = 20*ms
    grid_eq = '''v = (Igap - y)*int(not_refractory) : 1
                dy/dt = -y/tau_y : 1
                Igap : 1'''

    grid_layer = NeuronGroup(Ng, grid_eq, threshold = 'v > 0.5', refractory= 30*ms, method = 'exact')



    # Set up synapses from input to grid layer with STDP learning rule and randomized start weights

    dendrite_to_grid = Synapses(dendrite_layer, grid_layer,
                '''Igap_post = Ve_pre : 1 (summed)''')
    dendrite_to_grid.connect(condition = 'j == i // Ndendrites2')



    # # Set up inhibitory layer:
    tau_i = 10*ms
    inhibit_eq = '''dv/dt = -v/tau_i : 1 (unless refractory)'''
    inhibit_layer = NeuronGroup(1, inhibit_eq, threshold='v > 0.5', reset = 'v = 0', refractory = 30*ms, method = 'exact')

    grid_to_inhibit = Synapses(grid_layer, inhibit_layer, 'w : 1', on_pre = 'v_post += w', delay = 0.6*ms)
    grid_to_inhibit.connect()
    grid_to_inhibit.w = 1

    inhibit_to_grid = Synapses(inhibit_layer, grid_layer, 'w : 1', on_pre = 'y_post += Ndendrites2/200')
    inhibit_to_grid.connect()

    @network_operation(dt = repeats*pxs*100*ms)
    def printProgress(t):
        progress = t / (repeats*pxs*100*ms) 
        sys.stdout.write(f"\rProgress: {progress} / {pxs}")
        sys.stdout.flush()

    G = SpikeMonitor(grid_layer)
    print("Initialize done")

    run(len(X)*100*ms)

    if save_spike_train:
        np.savez_compressed(f"grid_simulation/Results/{file_name}",  \
            spike_train = G.spike_trains(), \
            positions = X)
    print()
    return G

if __name__ == '__main__':
    Ndendrites = 24
    Ng = 13
    sigma = 0.12
    stationary = False
    plot_spike_hist = True
    plot_weights = True
    if stationary:
        duration = 3000
        visualize_tick = 200
    else:
        duration = 2 * 10**6
        visualize_tick = 10000
    spike_plot = not (plot_spike_hist or plot_weights)
    save_data = False
    save_tick = 1000
    distrib = 'regular'
    output_filename = 'test.npz'
    baseline_effect = 1.6 / (Ndendrites*Ng)
    gridSimulation(Ndendrites, Ng, sigma, baseline_effect, duration, stationary, distrib, visualize_tick, plot_spike_hist, plot_weights, spike_plot, save_data, save_tick, output_filename)
