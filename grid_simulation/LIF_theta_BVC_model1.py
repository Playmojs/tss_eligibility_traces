from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
import utils
from scipy.ndimage import gaussian_filter
import h5py
import sys

# Total number of dendrites
Ndendrites = 20
Ndendrites2 = Ndendrites**2

# Total number of grid cells to simulate
Ng = 7

# Set up boundary cell values
Nthetas = 12
Ndists = 11
max_dist = 1
Nbvcs = Nthetas*Ndists
boundary_cells = np.meshgrid(np.arange(0,180, int(180/Nthetas), dtype =int), np.linspace(0, max_dist, Ndists))
boundary_cells = np.reshape(boundary_cells, (2,-1))

# Dendritic tree overlap
sigma = 0.1

# Set up input locations
spatialns = utils.CoordinateSamplers(Ndendrites, sigma)

# Set the rate of information from sensory to grid cells
theta_rate = 1/10 # denominator is theta frequency used

# Simulation variables
duration = 900000 # Duration of simulation in ms
stationary = False
spike_plot = False
visualize = True
dendrite_plot = False
n_den_plots = 10
visualize_tick = 10000
input_file = "Square/900s.npz"

assert(not (visualize and dendrite_plot))

save_data = False
save_tick = 10000
fname = "m3f0_1"
if save_data:
    weight_tracker = np.zeros((duration*1000 // save_tick + 1, Ng*Ndendrites2))
    score_tracker = np.zeros((duration*1000 // save_tick + 1, Ng))
save_id = [0]

# Prepare plot

def setup_plot_grid(n_rows, n_cols):
    import matplotlib
    import matplotlib.pyplot as plt

    # suppress deprecation warning from matplotlib.
    import warnings
    warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
    fig = plt.figure()
    ax = []
    for y in range (n_rows):
        for z in range(n_cols):
            ax.append(plt.subplot2grid((n_rows, n_cols),(y,z)))
    return fig, ax

if visualize:
    fig, ax = setup_plot_grid(3, Ng+1)
if dendrite_plot:
    fig, ax = setup_plot_grid(2, n_den_plots)

# Read file to get trajectory and speed and boundaries. The positions are assumed to be sampled at the theta-frequency.
X, speed, boundaries, boundary_vectors = utils.getTrajValues(f"grid_simulation/Trajectories/{input_file}")
mean_speed = np.mean(speed)
tMax = len(X)

# Input layer setup
inputs = np.empty((2,0))
filter = 2*sigma
neuron_indices = np.arange(Nbvcs, dtype = int16)

# Precalculate the entire firing of the spike generator group (to avoid having to restart runs when positions update):
print("Precalculating spatial input")
for i in np.arange(0, duration//100):
    time_ms = i*100
    sys.stdout.write("\rStatus: %3.4f" % ((i+1)*100/duration))
    sys.stdout.flush()

    iBoundaries = boundary_vectors[i]
    activity = np.abs(boundary_cells[1] - iBoundaries[np.ndarray.astype(boundary_cells[0], int)] + (np.random.rand(len(boundary_cells[1])) -0.5)*filter/10)
    
    filtered_neuron_indices = neuron_indices[activity<filter]
    filtered_spike_times = activity[activity<filter]/filter*20 # These values scale the activity so the slowest neurons fire 20 ms after theta, and the fastest up to theta
    temp_input = np.array((filtered_neuron_indices, filtered_spike_times + time_ms))
    temp_input = temp_input[:, np.argsort(temp_input[1,:])]
    inputs = np.hstack((inputs, temp_input))
print("\n")
del(i)

BVC_layer = SpikeGeneratorGroup(Nbvcs, inputs[0], inputs[1]*ms, sorted = True)

tau_d = 10*ms
taupost = 80*ms
base_conductivity = 10 / Ndendrites2 
Apost = -base_conductivity / 4 
Apre = base_conductivity / 2 
dendrite_eq = '''dv/dt = -v / tau_d : 1
                dapost/dt = -apost/taupost : 1
                c : 1'''
dendrite_layer = NeuronGroup(Ng * Ndendrites2, dendrite_eq, method = "exact")
dendrite_layer.c = base_conductivity #conductivity from dendrite to grid cell
c_max = 1.5*base_conductivity
if dendrite_plot:
    D = StateMonitor(dendrite_layer, 'v', np.arange(0,n_den_plots))

BVC_synapses = Synapses(BVC_layer, dendrite_layer, 'w : 1', on_pre='''v_post +=w
                        c = clip(c + apost*int(v_post>0.5), 0, c_max)''')
BVC_connections = utils.getBVCtoDendriteConnectivity(Nbvcs, Ndendrites2 * Ng, distribution = 'orthogonal', rate = 0.1, bvc_params = [12,11])
BVC_synapses.connect(i = BVC_connections[1], j = BVC_connections[0])
weights = 0.4
BVC_synapses.w = weights

#Grid layer and inhibitory layer:

tau_g = 20*ms
grid_eq = '''dv/dt = (-v + Igap) / tau_g : 1 (unless refractory)
            Igap : 1'''

grid_layer = NeuronGroup(Ng, grid_eq, threshold = 'v > 0.07', reset = 'v = 0', refractory= 30*ms, method = 'exact')
G = SpikeMonitor(grid_layer)



dendrite_to_grid = Synapses(dendrite_layer, grid_layer, '''Igap_post = c_pre*(v_pre-v_post)*int(v_pre > weights) : 1 (summed)''',
                            on_post = '''c_pre = clip(c_pre + (v_pre*Apre*int(v_pre>weights) + 0/(Ng*Ndendrites2)*(c_max-c_pre)), 0, c_max)
                            apost_pre += Apost*int(v_pre > 0.1)'''
                            )
dendrite_to_grid.connect(condition = 'i // Ndendrites2 == j')


# Set up inhibitory layer:
inhibit_layer = NeuronGroup(Ng, grid_eq, threshold='v > 0.5', reset = 'v = 0', method = 'exact')

grid_to_inhibit = Synapses(grid_layer, inhibit_layer, 'w : 1', on_pre = 'v_post += w', delay = 1*ms)
grid_to_inhibit.connect(condition = 'i==j')
grid_to_inhibit.w = 0.7

inhibit_to_grid = Synapses(inhibit_layer, grid_layer, 'w : 1', on_pre = 'v_post = -10')
inhibit_to_grid.connect(condition = 'i!=j')
inhibit_to_grid.w = 2

# # @network_operation(dt = theta_rate*ms)
# # def update_learning_rate(t):
# #     if stationary:
# #         learning_speed = 1
# #     else:
# #         current_speed = speed[int(t/(delta_t*ms))]
# #         learning_speed = np.exp(-(mean_speed-current_speed)**2/mean_speed)
# #     input_weights.l_speed = learning_speed

def plot_weights(t):
    time_ms = t/ms
    time_id = int(time_ms/100)
    x = X[time_id, :] if not stationary else X[0,:] +  np.array([sigma/2, sigma/2])* (np.floor(time_ms/1000))
    x += 0.5
    i68, i95, i99 = spatialns.get68_95(x)
    pxs = 25
    ax[0].cla()
    ax[0].imshow(spatialns.act(x) * i68, interpolation='none', origin='lower')


    position_hist, _, __ = np.histogram2d(X[max(0, time_id - 3000):time_id, 1], X[max(0,time_id-3000):time_id, 0], pxs, [[-0.5,0.5],[-0.5,0.5]])
    

    spike_times = G.t/ms
    spike_times = spike_times[spike_times > (time_ms - 300000)]
    spike_indices = np.floor(spike_times/100)

    spike_positions = X[np.ndarray.astype(spike_indices, int)]
    if len(spike_positions) == 0:
        spike_positions = np.vstack((spike_positions, [0,0]))
    spike_hist, _, __ = np.histogram2d(spike_positions[:,1], spike_positions[:,0], pxs, [[-0.5,0.5],[-0.5,0.5]])

    ax[Ng+1].cla()
    ax[Ng+1].imshow(spike_hist, interpolation = 'none', origin = 'lower')
    ax[Ng+1].set_title("mean")

    ax[2*Ng+2].cla()
    ax[2*Ng+2].imshow(position_hist, interpolation = 'none', origin = 'lower')
    ax[2*Ng+2].set_title("trajectory")

    mean_score = 0
    spike_trains = G.spike_trains()

    for z in range(Ng):
        spike_times = spike_trains[z]/ms
        spike_times = spike_times[spike_times > (time_ms - 300000)]
        spike_indices = np.floor(spike_times/100)

        spike_positions = X[np.ndarray.astype(spike_indices, int)]
        if len(spike_positions) == 0:
            spike_positions = np.vstack((spike_positions, [0,0]))
        spike_hist, _, __ = np.histogram2d(spike_positions[:,1], spike_positions[:,0], pxs, [[-0.5,0.5],[-0.5,0.5]])
        # compute gridness score
        gauss_spike_hist = gaussian_filter(spike_hist, 1)
        corr_w = utils.normcorr2d(gauss_spike_hist)
        gscore, _ = utils.gridness_score(corr_w, pxs, sigma)
        cntr_xy = corr_w.shape[0]//2

        mean_score += gscore/Ng
        # only consider cells with a score > 0 (as is common in
        # literature)
        if gscore > 0:
            orientation, closest_r, _ = utils.grid_orientation(corr_w, Ndendrites, sigma)
        else:
            orientation = -1
            closest_r = np.array([0, 0])

        # show weights
        ax[z+1].imshow(spike_hist, interpolation='none', origin='lower')
        ax[z+1].set_title("%3.4f" % (gscore))

        # show gaussian filtered weights
        ax[2+Ng+z].imshow(gauss_spike_hist, interpolation='none', origin = 'lower')

        # show auto-correlation and nearest blod tracker
        ax[3+2*Ng+z].cla()
        ax[3+2*Ng+z].imshow(corr_w, interpolation='none', origin='lower')
        ax[3+2*Ng+z].autoscale(False)
        ax[3+2*Ng+z].plot([cntr_xy, cntr_xy + closest_r[0]], [cntr_xy, cntr_xy + closest_r[1]], linewidth=2.0, color='black')
    ax[2+Ng].set_title("%3.2f" % (np.mean(dendrite_layer.c)/c_max))
    ax[3+Ng].set_title("%3.2f" % (np.sqrt(np.var(dendrite_layer.c))/c_max))
    ax[4+Ng].set_title("%3.2f" % (np.max(dendrite_layer.c)/c_max))
    fig.suptitle(f"{t/second // 60} mins {(t/second) % 60} seconds")
    plt.pause(10)

def plot_dendrite_activity(t, dendrite_state_monitor):
    if t == 0*second:
        return
    pxs = 25 # discretization of positions
    
    # get a list of times ten ms after each theta cycle:
    time_ids = np.arange(0, t, 0.1)
    time_ids += 0.01
    time_ids = np.ndarray.astype(time_ids*10000, int)

    # get xy - indices for each theta cycle based on the discretization
    place_id = int(t/ms/100)
    places = X[0:place_id] # Get places at each recorded time until now
    places = (places - np.min(boundaries, axis=0)) / (np.max(boundaries, axis = 0) - np.min(boundaries, axis = 0)) # Standardise places to 0-1
    places = np.ndarray.astype(np.floor(places*pxs), int) # Discretize places to grid

    for z in range(n_den_plots):
        plot = np.zeros((pxs,pxs))
        plot[places[:,0], places[:,1]] = dendrite_state_monitor.v[z, time_ids]
        ax[z].imshow(plot, vmax = 0.8, interpolation='none', origin = 'lower')
        ax[z + n_den_plots].imshow(gaussian_filter(plot, 0.5), interpolation = 'none', origin= 'lower')
    fig.suptitle(f"{t/second // 60} mins {(t/second) % 60} seconds")
    plt.pause(3)


@network_operation(dt = visualize_tick*ms)
def update_plot(t):
    if visualize:
       plot_weights(t)
    elif dendrite_plot:
        plot_dendrite_activity(t, D)

# @network_operation(dt = save_tick*ms)
# def save_weights(t):
#     if not save_data:
#         return
#     sys.stdout.write("\rProgress: %3.4f" % ((t/second)/duration))
#     sys.stdout.flush()
#     weight_tracker[save_id[0], ...] = input_weights.w

#     grid_weights = np.reshape(input_weights.w, (Ndendrites2, Ng))
#     for z in range(Ng):    
#         weight2d = np.reshape(grid_weights[:, z], (Ndendrites, Ndendrites))
#         corr_w = utils.normcorr2d(weight2d)
#         gscore, _ = utils.gridness_score(corr_w, Ndendrites, sigma)
#         score_tracker[save_id[0],z] = gscore
#     save_id[0]+=1

if spike_plot:
    M = SpikeMonitor(BVC_layer)
    D = StateMonitor(dendrite_layer, ['v','c'], [0,1,2])
    Gs = StateMonitor(grid_layer, ['v', 'Igap'], 0)

print("Initialize done")

run(duration*ms)

# if save_data:
#     np.savez_compressed(f'grid_simulation/Results/{fname}', \
#         Ndendrites = Ndendrites, \
#         sigma = sigma, \
#         Ng = Ng, \
#         save_tick = save_tick, \
#         duration = duration, \
#         weights = weight_tracker, \
#         scores = score_tracker)

if visualize or dendrite_plot:
    plt.show()

if spike_plot:
    # print(G.t/ms)
    # print(G.i)
    # print(R.t/ms)
    plt.figure(figsize = (12,12))
    plt.subplot(221)
    plt.plot(Gs.t/ms, Gs.v[0], 'C0')
    plt.plot(Gs.t/ms, Gs.Igap[0], 'Magenta')
    plt.subplot(222)
    plt.plot(D.t/ms, D.v[0], 'C0')
    plt.plot(D.t/ms, D.c[0]/c_max, 'Green')
    plt.vlines(G.t/ms, 0, 1, color = "red")
    plt.subplot(223)
    plt.plot(D.t/ms, D.v[1], 'C0')
    plt.plot(D.t/ms, D.c[1]/c_max, 'Green')
    plt.vlines(G.t/ms, 0, 1, color = "red")
    plt.subplot(224)
    plt.plot(D.t/ms, D.v[2], 'C0')
    plt.plot(D.t/ms, D.c[2]/c_max, 'Green')
    plt.vlines(G.t/ms, 0, 1, color = "red")
    plt.show()


