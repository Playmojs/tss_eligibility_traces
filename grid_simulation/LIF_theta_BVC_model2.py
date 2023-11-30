from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
import utils
from scipy.ndimage import gaussian_filter
import h5py
import sys

# Total number of dendrites
Ndendrites = 48
Ndendrites2 = Ndendrites**2

# Total number of grid cells to simulate
Ng = 13

# Dendritic tree overlap
sigma = 0.1

# Set up input locations
spatialns = utils.CoordinateSamplers(Ndendrites, sigma)

# Set the rate of information from sensory to grid cells
theta_rate = 1/10 # denominator is theta frequency used

# Set up boundary cell values
Nthetas = 12
Ndists = 11
max_dist = 1
Nbvcs = Nthetas*Ndists
boundary_cells = np.meshgrid(np.arange(0,180, int(180/Nthetas), dtype =int), np.linspace(0, max_dist, Ndists))
boundary_cells = np.reshape(boundary_cells, (2,-1))

# Simulation variables
duration = 9 * 10**5
stationary = False
visualize = True
spike_plot = False
visualize_tick = 10000
input_file = "Square/900s.npz"

save_data = False
save_tick = 10000
fname = "m4f100_1"
if save_data:
    weight_tracker = np.zeros((duration*1000 // save_tick + 1, Ng*Ndendrites2))
    score_tracker = np.zeros((duration*1000 // save_tick + 1, Ng))
save_id = [0]

# Prepare plot

if visualize:
    import matplotlib
    import matplotlib.pyplot as plt

    # suppress deprecation warning from matplotlib.
    import warnings
    warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
    n_rows = 3
    fig = plt.figure()
    ax = []
    for y in range (n_rows):
        for z in range(Ng+1):
            ax.append(plt.subplot2grid((n_rows,Ng+1),(y,z)))



# Read file to get trajectory and speed
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
for i in np.arange(0, duration//100, dtype = int):
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

#Grid layer and inhibitory layer:

tau_g = 10*ms
grid_eq = '''dv/dt = - v / tau_g : 1 (unless refractory)'''

grid_layer = NeuronGroup(Ng, grid_eq, threshold = 'v > 0.5', reset = 'v = -0.1', refractory= 30*ms, method = 'exact')
if visualize or spike_plot:
    G = SpikeMonitor(grid_layer)


# Set up synapses from input to grid layer with STDP learning rule and randomized start weights

taupre = 8*ms
taupost = 80*ms
Apre = 0.01
Apost = -0.005
input_weights = Synapses(BVC_layer, grid_layer, '''
            w : 1
            l_speed : 1
            dapre/dt = -apre/taupre : 1 (event-driven)
            dapost/dt = -apost/taupost : 1 (event-driven)
            ''',
            on_pre='''
            v_post += w
            apre += Apre
            w = clip(w+(apost+5/(Ng*Nbvcs)*(wmax_i-w))*l_speed, 0, wmax_i)
            ''',
            on_post='''
            apost += Apost
            w = clip(w+apre*l_speed, 0, wmax_i)
            ''', delay = 3*ms)
input_weights.connect(p = 1)
n_weights = len(input_weights.w)
wmax_i = 12 * Ng / n_weights

weights = np.random.rand(n_weights)*0.6*wmax_i
# weights[weights<0.95] = 0
# weights[weights>0] = 0.1
input_weights.w = weights

# # Set up inhibitory layer:
inhibit_layer = NeuronGroup(Ng, grid_eq, threshold='v > 0.5', reset = 'v = 0', method = 'exact')

grid_to_inhibit = Synapses(grid_layer, inhibit_layer, 'w : 1', on_pre = 'v_post += w', delay = 0.6*ms)
grid_to_inhibit.connect(condition = 'i==j')
grid_to_inhibit.w = 0.7

inhibit_to_grid = Synapses(inhibit_layer, grid_layer, 'w : 1', on_pre = 'v_post = -10')
inhibit_to_grid.connect(condition = 'i!=j')
inhibit_to_grid.w = 2

@network_operation(dt = theta_rate*second)
def update_learning_rate(t):
    if stationary:
        learning_speed = 1
    else:
        current_speed = speed[int(t/(second * theta_rate))]
        learning_speed = 1.5*np.exp(-(mean_speed-current_speed)**2/mean_speed)
    input_weights.l_speed = learning_speed

@network_operation(dt = visualize_tick*ms)
def update_plot(t):
    plot_weights(t)

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
    n_spikes = 0

    for z in range(Ng):
        spike_times = spike_trains[z]/ms
        spike_times = spike_times[spike_times > (time_ms - 300000)]
        n_spikes += len(spike_times)
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
    
    ax[Ng+2].set_title(f"#Spikes: {n_spikes}")

    fig.suptitle(f"{t/second // 60} mins {(t/second) % 60} seconds")
    plt.pause(10)

@network_operation(dt = save_tick*ms)
def save_weights(t):
    if not save_data:
        return
    sys.stdout.write("\rProgress: %3.4f" % ((t/second)/duration))
    sys.stdout.flush()
    weight_tracker[save_id[0], ...] = input_weights.w

    grid_weights = np.reshape(input_weights.w, (Ndendrites2, Ng))
    for z in range(Ng):    
        weight2d = np.reshape(grid_weights[:, z], (Ndendrites, Ndendrites))
        corr_w = utils.normcorr2d(weight2d)
        gscore, _ = utils.gridness_score(corr_w, Ndendrites, sigma)
        score_tracker[save_id[0],z] = gscore
    save_id[0]+=1

if spike_plot:
    M = SpikeMonitor(BVC_layer)
    S = StateMonitor(grid_layer, True, [0,1,2] if Ng >= 3 else 0)
    I = SpikeMonitor(inhibit_layer)

print("Initialize done")

run(duration*ms)

if save_data:
    np.savez_compressed(f'grid_simulation/Results/{fname}', \
        Ndendrites = Ndendrites, \
        sigma = sigma, \
        Ng = Ng, \
        save_tick = save_tick, \
        duration = duration, \
        weights = weight_tracker, \
        scores = score_tracker)

if visualize:
    plot_weights(duration*ms)
    plt.show()

if spike_plot:
    # print(G.t/ms)
    # print(G.i)
    # print(R.t/ms)
    plt.figure(figsize = (12,12))
    plt.subplot(221)
    plt.plot(M.t/ms, M.i, '.k')
    plt.vlines(G.t/ms, 0, Nbvcs)
    plt.vlines(I.t/ms, 0, Nbvcs, colors = 'r')
    plt.subplot(222)
    plt.plot(S.t/ms, S.v[0], 'C0')
    plt.subplot(223)
    plt.plot(S.t/ms, S.v[1], 'C0')
    plt.subplot(224)
    plt.plot(S.t/ms, S.v[2], 'C0')
    plt.show()


