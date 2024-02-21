from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import tanh
from numpy import exp
import utils
from scipy.ndimage import gaussian_filter
import h5py
import sys


def gridSimulation(Ndendrites, Ng, sigma, Nthetas, Ndists, distribution, pxs, repeats, BVC_connections, conductivities, save_spike_train, file_name):    
    # Total number of dendrites
    Ndendrites2 = Ndendrites**2

    # Set up boundary cell values
    max_dist = 0.5
    Nbvcs = Nthetas*Ndists
    boundary_cells = np.meshgrid(np.arange(0,180, int(180/Nthetas), dtype =int), np.linspace(0, max_dist, Ndists))
    boundary_cells = np.reshape(boundary_cells, (2,-1))

    # Set up input locations
    spatialns = utils.CoordinateSamplers(Ndendrites, sigma)

    # Set the rate of information from sensory to grid cells
    theta_rate = 1/10 # denominator is theta frequency used

    # Read file to get trajectory and speed and boundaries. The positions are assumed to be sampled at the theta-frequency.

    print("Generating trajectory")
    import trajectory_gen
    boundary_shape = 'square'
    xs = np.linspace(0, 1, pxs)
    ys = np.linspace(0, 1, pxs)
    X = np.reshape(np.meshgrid(xs, ys, indexing='xy'), (2, -1)).T
    X = np.repeat(X, repeats, axis = 0)
    
    boundary_vectors = trajectory_gen.generateTrajectory(0.1, np.size(X, 0)/10, "", boundary_shape = boundary_shape, pos = X, save_to_f= False)


    print("Precalculate spatial inputs")
    delays = utils.BVC_act(boundary_cells,boundary_vectors, Nbvcs, sigma, noise_level= 0.005, alg = 'simple')

    # Filter based on delay
    indices = np.where(delays < 20) 
    spike_times = delays[indices] + 100*indices[0]
    neuron_indices = indices[1]

    BVC_layer = SpikeGeneratorGroup(Nbvcs, neuron_indices, spike_times*ms)

    tau_d = 10*ms 
    dendrite_eq = '''dv/dt = -v/tau_d : 1
                    c : 1
                    Ve = int(v > 1.1) : 1'''
    dendrite_layer = NeuronGroup(Ndendrites2 * Ng, dendrite_eq, method = 'exact')
    dendrite_layer.c = conductivities 

    BVC_synapses = Synapses(BVC_layer, dendrite_layer, 'w : 1', on_pre='''v_post +=w''')
    BVC_synapses.connect(i = BVC_connections[0], j = BVC_connections[1])
    weights = 1
    BVC_synapses.w = weights

    #Grid layer and inhibitory layer:

    tau_g = 10*ms
    tau_y = 20*ms
    grid_eq = '''v = (Igap - y)*int(not_refractory) : 1
                dy/dt = -y/tau_y : 1
                Igap : 1'''

    grid_layer = NeuronGroup(Ng, grid_eq, threshold = 'v > 0.5', refractory= 30*ms, method = 'exact')
    G = SpikeMonitor(grid_layer)

    # Set up synapses from input to grid layer with STDP learning rule and randomized start weights

    dendrite_to_grid = Synapses(dendrite_layer, grid_layer, 
                '''Igap_post = Ve_pre*c_pre : 1 (summed)''')
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

    print("Initialize done")

    run(len(X)*100*ms)

    if save_spike_train:
        np.savez_compressed(f"grid_simulation/Results/{file_name}",  \
            spike_train = G.spike_trains(), \
            positions = X)
    print()
    return G
