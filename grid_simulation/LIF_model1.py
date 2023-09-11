from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
import utils
import h5py

# Total number of dendrites
Ndendrites = 48
Ndendrites2 = Ndendrites**2

# Total number of grid cells to simulate
Ng = 1

# Dendritic tree overlap
sigma = 0.08

# Set up input locations
spatialns = utils.CoordinateSamplers(Ndendrites, sigma)

# Prepare plot
spike_plot = False
visualize = True
spike_plot = not visualize
visualize_tick = 10000

if visualize:
    import matplotlib
    import matplotlib.pyplot as plt

    # suppress deprecation warning from matplotlib.
    import warnings
    warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

    fig = plt.figure()
    ax = []
    ax.append(plt.subplot2grid((2,Ng+1), (0,0)))
    for z in range(1,Ng+1):
        ax.append(plt.subplot2grid((2,Ng+1),(0,z)))
    for z in range(1,Ng+1):
        ax.append(plt.subplot2grid((2,Ng+1),(1,z)))


# Read file to get positions and velocities
X, speed = utils.getCoords(h5py.File("grid_simulation/trajectory_square_2d_0.01dt_long.hdf5", "r"))
delta_t = 10
mean_speed = np.mean(speed)
tMax = len(X)

# Layer setup
tau = 10*ms 
sig = 0.01

input_eq = '''
dv/dt = (v0-v)/tau  + sig*xi*tau**-0.5: 1 (unless refractory)
v0 : 1
'''

grid_eq = '''dv/dt = - v / tau : 1 (unless refractory)'''

input_layer = NeuronGroup(Ndendrites2, input_eq, threshold = 'v > 0.165', reset = 'v = -0.1', refractory = 20*ms, method = 'euler')
grid_layer = NeuronGroup(1, grid_eq, threshold = 'v > 1.6', reset = 'v = -0.1', refractory= 20*ms, method = 'exact')

# Set up synapses from input to grid layer with STDP learning rule and randomized start weights

taupre = 35*ms
taupost = 80*ms
wmax_i = 0.2
Apre = 0.0115
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
            w = clip(w+(apost*l_speed), 0, wmax_i)
            ''',
            on_post='''
            apost += Apost
            w = clip(w+(apre*l_speed), 0, wmax_i)
            ''', delay = 3*ms)
input_weights.connect()

weights = np.random.rand(Ndendrites2)*0.1
input_weights.w = weights

# Set up recurrent synapses for center-surround dynamics
taupre_r = 40 *ms
taupost_r = 10 * ms
wmax_r = 0.1
Apre_r = 0.01
Apost_r = -0.005
recurrent_weights = Synapses(grid_layer, input_layer, '''
            w : 1
            l_speed : 1
            dapre/dt = -apre/taupre : 1 (event-driven)
            dapost/dt = -apost/taupost : 1 (event-driven)
            ''',
            on_pre='''
            v_post += w
            apre += Apre_r
            w = clip(w+apost*l_speed, 0, wmax_r)
            ''',
            on_post='''
            apost += Apost_r
            w = clip(w+apre*l_speed, 0, wmax_r)
            ''', 
            delay = 5*ms)
recurrent_weights.connect()

weights = np.random.rand(Ndendrites2)*0.1
recurrent_weights.w = weights


# Update rule for changing position
@network_operation(dt=delta_t*ms)
def change(t):
    # Update position
    time = t/second * 1000
    
    # x = X[0,:]  if time < 5000 else X[0, :] + [sigma, sigma]
    # learning_speed = 1

    x = X[int(time/delta_t), :]
    current_speed = speed[int(time/delta_t)]
    learning_speed = np.exp(-1/mean_speed*current_speed**2)
    
    activity = spatialns.act(x)
    input_layer.v0 = np.ndarray.flatten(activity)/2.5
    
    input_weights.l_speed = learning_speed
    recurrent_weights.l_speed = learning_speed

    # Bump weights following baseline rule
    # input_weights.w += (2/Ndendrites2 * (1-input_weights.w) *sigma/10)
    # input_weights.w[input_weights.w > wmax_i] = wmax_i

    weight2d = np.reshape(input_weights.w, [Ndendrites, Ndendrites])
    # Visualize grid weights if wanted

    if not visualize or time % visualize_tick != 0:
        return
    
    i68, i95, i99 = spatialns.get68_95(x)
    ax[0].cla()
    ax[0].imshow(activity * i68, interpolation='none', origin='lower')
    ax[0].set_title("%d" % time)

    for z in range(Ng):
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

        # show auto-correlation and nearest blod tracker
        ax[1+Ng+z].cla()
        ax[1+Ng+z].imshow(corr_w, interpolation='none', origin='lower')
        ax[1+Ng+z].autoscale(False)
        ax[1+Ng+z].plot([cntr_xy, cntr_xy + closest_r[0]], [cntr_xy, cntr_xy + closest_r[1]], linewidth=2.0, color='black')
    plt.pause(1)

if spike_plot:
    M = SpikeMonitor(input_layer)
    G = SpikeMonitor(grid_layer)
    S = StateMonitor(grid_layer, 'v', record = 0)

run(6000000*ms)

if visualize:
    plt.show()

if spike_plot:
    plt.figure(figsize = (12,4))
    plt.subplot(121)
    plt.plot(M.t/ms, M.i, '.k')
    plt.vlines(G.t/ms, 0, Ndendrites2)
    plt.subplot(122)
    plt.plot(S.t/ms, S.v[0], 'C0')
    plt.show()
