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

# Simulation variables
duration = 6000 # NB: in this model, duration is in seconds, for convenience of implementation
stationary = False
visualize = False
spike_plot = False
visualize_tick = 20000

save_data = True
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
X, speed = utils.getCoords(h5py.File("grid_simulation/trajectory_square_2d_0.01dt_long.hdf5", "r"))
delta_t = 10 # Sampling frequency in the trajectory file
mean_speed = np.mean(speed)
tMax = len(X)


# Input layer setup
inputs = np.empty((2,0))
filter = 18

# Precalculate the entire firing of the spike generator group (to avoid having to restart runs when positions update):
print("Precalculate spatial input:")
for i in np.arange(0, duration, theta_rate):
    time_ms = i*1000
    sys.stdout.write("\rStatus: %3.4f" % ((i+theta_rate)/duration))
    sys.stdout.flush()

    x = X[int(time_ms/delta_t), :] if not stationary else X[0,:] + np.array([sigma/2, sigma/2])* (np.floor(time_ms/1000))
    
    activity = np.round(np.ndarray.flatten(spatialns.dist(x))/sigma * 10 + np.max(2*np.random.rand(Ndendrites2)-1, 0), 1)
    neuron_indices = np.arange(Ndendrites2, dtype = int16)
    filtered_neuron_indices = neuron_indices[activity<filter]
    filtered_activity = activity[activity<filter]
    temp_input = np.array((filtered_neuron_indices, filtered_activity + time_ms))
    temp_input = temp_input[:, np.argsort(temp_input[1,:])]
    inputs = np.hstack((inputs, temp_input))

print("\nDone")

input_layer = SpikeGeneratorGroup(Ndendrites2, inputs[0], inputs[1]*ms, sorted = True)

#Grid layer and inhibitory layer:

tau_g = 10*ms
grid_eq = '''dv/dt = - v / tau_g : 1 (unless refractory)'''

grid_layer = NeuronGroup(Ng, grid_eq, threshold = 'v > 1.0', reset = 'v = -0.1', refractory= 30*ms, method = 'exact')


# Set up synapses from input to grid layer with STDP learning rule and randomized start weights

taupre = 8*ms
taupost = 80*ms
wmax_i = 0.1
Apre = 0.01
Apost = -0.005
input_weights = Synapses(input_layer, grid_layer, '''
            w : 1
            l_speed : 1
            dapre/dt = -apre/taupre : 1 (event-driven)
            dapost/dt = -apost/taupost : 1 (event-driven)
            ''',
            on_pre='''
            v_post += w
            apre += Apre
            w = clip(w+(apost+100/(Ng*Ndendrites2)*(wmax_i-w))*l_speed, 0, wmax_i)
            ''',
            on_post='''
            apost += Apost
            w = clip(w+apre*l_speed, 0, wmax_i)
            ''', delay = 3*ms)
input_weights.connect()

weights = np.random.rand(Ndendrites2 * Ng)*0.06
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

@network_operation(dt = theta_rate*ms)
def update_learning_rate(t):
    if stationary:
        learning_speed = 1
    else:
        current_speed = speed[int(t/(delta_t*ms))]
        learning_speed = np.exp(-(mean_speed-current_speed)**2/mean_speed)*np.exp(-t/(3000*second))
    input_weights.l_speed = learning_speed

@network_operation(dt = visualize_tick*ms)
def update_plot(t):
    # Visualize grid weights if wanted
    if not visualize:
        return
    time_ms = t/ms
    x = X[int(time_ms/delta_t), :] if not stationary else X[0,:] +  np.array([sigma/2, sigma/2])* (np.floor(time_ms/1000))
    i68, i95, i99 = spatialns.get68_95(x)
    ax[0].cla()
    ax[0].imshow(spatialns.act(x) * i68, interpolation='none', origin='lower')
    ax[0].set_title("%d" % time_ms)


    grid_weights = np.reshape(input_weights.w, (Ndendrites2, Ng))
    
    mean_weights = np.reshape(np.mean(grid_weights, axis = 1 ), (Ndendrites, Ndendrites))
    ax[Ng+1].cla()
    ax[Ng+1].imshow(mean_weights, interpolation = 'none', origin = 'lower')
    ax[Ng+1].set_title("mean")

    position_hist, _, __ = (np.histogram2d(X[0:int(time_ms/delta_t), 1], X[0:int(time_ms/delta_t), 0], 20, [[0,1],[0,1]]))

    ax[2*Ng+2].cla()
    ax[2*Ng+2].imshow(position_hist, interpolation = 'none', origin = 'lower')
    ax[2*Ng+2].set_title("trajectory")

    mean_score = 0

    for z in range(Ng):
        
        weight2d = np.reshape(grid_weights[:, z], (Ndendrites, Ndendrites))
 
        # compute gridness score
        corr_w = utils.normcorr2d(weight2d)
        gscore, _ = utils.gridness_score(corr_w, Ndendrites, sigma)
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
        ax[z+1].imshow(weight2d, interpolation='none', origin='lower')
        ax[z+1].set_title("%3.4f" % (gscore))

        # show gaussian filtered weights
        ax[2+Ng+z].imshow(gaussian_filter(weight2d, 1.5), interpolation='none', origin = 'lower')

        # show auto-correlation and nearest blod tracker
        ax[3+2*Ng+z].cla()
        ax[3+2*Ng+z].imshow(corr_w, interpolation='none', origin='lower')
        ax[3+2*Ng+z].autoscale(False)
        ax[3+2*Ng+z].plot([cntr_xy, cntr_xy + closest_r[0]], [cntr_xy, cntr_xy + closest_r[1]], linewidth=2.0, color='black')
    ax[2+Ng].set_title("%3.4f" % (mean_score))
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
    M = SpikeMonitor(input_layer)
    G = SpikeMonitor(grid_layer)

print("Initialize done")

run(duration*second)

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
    plt.show()

if spike_plot:
    # print(G.t/ms)
    # print(G.i)
    # print(R.t/ms)
    plt.figure(figsize = (12,12))
    # plt.subplot(221)
    plt.plot(M.t/ms, M.i, '.k')
    plt.vlines(G.t/ms, 0, Ndendrites2)
    # plt.vlines(R.t/ms, 0, Ndendrites2, colors = 'r')
    # plt.subplot(222)
    # plt.plot(S.t/ms, S.v[0], 'C0')
    # plt.subplot(223)
    # plt.plot(S.t/ms, S.v[1], 'C0')
    # plt.subplot(224)
    # plt.plot(S.t/ms, S.v[2], 'C0')
    plt.show()


