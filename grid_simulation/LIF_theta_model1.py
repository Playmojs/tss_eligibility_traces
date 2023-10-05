from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
import utils
from scipy.ndimage import gaussian_filter
import h5py

# Total number of dendrites
Ndendrites = 48
Ndendrites2 = Ndendrites**2

# Total number of grid cells to simulate
Ng = 3

# Dendritic tree overlap
sigma = 0.08

# Set up input locations
spatialns = utils.CoordinateSamplers(Ndendrites, sigma)

# Set the rate of information from sensory to grid cells
theta_rate = 1/10 # Denominator is theta frequency used

# Simulation variables
duration = 200 # NB: in this model, duration is in seconds, for convenience of implementation
stationary = False
visualize = True
spike_plot = not visualize
visualize_tick = 10000

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
    ax.append(plt.subplot2grid((n_rows,Ng+1), (0,0)))
    for y in range (n_rows):
        for z in range(1,Ng+1):
            ax.append(plt.subplot2grid((n_rows,Ng+1),(y,z)))



# Read file to get trajectory and speed
X, speed = utils.getCoords(h5py.File("grid_simulation/trajectory_square_2d_0.01dt_long.hdf5", "r"))
delta_t = 10 # Sampling frequency in the trajectory file
mean_speed = np.mean(speed)
tMax = len(X)

# Layer setup

tau_g = 10*ms
grid_eq = '''dv/dt = - v / tau_g : 1 (unless refractory)'''

input_layer = SpikeGeneratorGroup(Ndendrites2, np.array([]),np.array([])*ms)
grid_layer = NeuronGroup(Ng, grid_eq, threshold = 'v > 1.0', reset = 'v = -0.1', refractory= 30*ms, method = 'exact')
inhibit_layer = NeuronGroup(Ng, grid_eq, threshold='v > 0.5', reset = 'v = 0', method = 'exact')


# Set up synapses from input to grid layer with STDP learning rule and randomized start weights

taupre = 8*ms
taupost = 80*ms
wmax_i = 0.07
Apre = 0.008
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
            w = clip(w+apost*l_speed, 0, wmax_i)
            ''',
            on_post='''
            apost += Apost
            w = clip(w+apre*l_speed, 0, wmax_i)
            ''', delay = 4*ms)
input_weights.connect()

weights = np.random.rand(Ndendrites2 * Ng)*0.03
# weights[weights<0.85] = 0
# weights[weights>0] = 0.1
input_weights.w = weights

# # Set up inhibitory layer:
grid_to_inhibit = Synapses(grid_layer, inhibit_layer, 'w : 1', on_pre = 'v_post += w')
grid_to_inhibit.connect(condition = 'i==j')
grid_to_inhibit.w = 0.7

inhibit_to_grid = Synapses(inhibit_layer, grid_layer, 'w : 1', on_pre = 'v_post = -3')
inhibit_to_grid.connect(condition = 'i!=j')
inhibit_to_grid.w = 2

@network_operation(dt = visualize_tick*ms)
def update_plot(t):
    # Visualize grid weights if wanted
    if not visualize:
        return
    time = t/second * 1000

    i68, i95, i99 = spatialns.get68_95(x)
    ax[0].cla()
    ax[0].imshow(spatialns.act(X[int(time/delta_t), :]) * i68, interpolation='none', origin='lower')
    ax[0].set_title("%d" % time)

    grid_weights = np.reshape(input_weights.w, (Ndendrites2, Ng))

    for z in range(Ng):
        
        weight2d = np.reshape(grid_weights[:, z], (Ndendrites, Ndendrites))
 
        # compute gridness score
        corr_w = utils.normcorr2d(weight2d)
        gscore, _ = utils.gridness_score(corr_w, Ndendrites, sigma)
        cntr_xy = corr_w.shape[0]//2

        # only consider cells with a score > 0 (as is common in
        # literature)
        if gscore > 0:
            orientation, closest_r, _ = utils.grid_orientation(corr_w, Ndendrites, sigma)
        else:
            orientation = -1
            closest_r = np.array([0, 0])

        # show weights
        ax[z+1].imshow(weight2d, interpolation='none', origin='lower')
        ax[z+1].set_title("%3.4f, %4.2f" % (gscore, orientation))

        # show gaussian filtered weights
        ax[1+Ng+z].imshow(gaussian_filter(weight2d, 1.5), interpolation='none', origin = 'lower')

        # show auto-correlation and nearest blod tracker
        ax[1+2*Ng+z].cla()
        ax[1+2*Ng+z].imshow(corr_w, interpolation='none', origin='lower')
        ax[1+2*Ng+z].autoscale(False)
        ax[1+2*Ng+z].plot([cntr_xy, cntr_xy + closest_r[0]], [cntr_xy, cntr_xy + closest_r[1]], linewidth=2.0, color='black')

    plt.pause(3)

if spike_plot:
    M = SpikeMonitor(input_layer)
    G = SpikeMonitor(grid_layer)
    #R = SpikeMonitor(inhibit_layer)
    #S = StateMonitor(grid_layer, 'v', record = [0,1,2])

for i in np.arange(0, duration, theta_rate):
    time_ms = i*1000
    print(time_ms)

    if stationary:
        x = X[0,:] if time_ms < 1000 else X[0, :] + [sigma/2, sigma/2]
        learning_speed = 1
    else:
        x = X[int(time_ms/delta_t), :]
        current_speed = speed[int(time_ms/delta_t)]
        learning_speed = np.exp(-current_speed**2/mean_speed)
    
    activity = np.ndarray.flatten(spatialns.dist(x))/sigma * 10 + 2*np.random.rand(Ndendrites2)-1
    activity[activity>20] = 10000

    input_weights.l_speed = learning_speed
    input_layer.set_spikes(np.arange(Ndendrites2), (time_ms+activity)*ms)
    run(theta_rate*second)

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
