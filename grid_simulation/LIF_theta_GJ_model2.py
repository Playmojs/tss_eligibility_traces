from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import tanh
import utils
from scipy.ndimage import gaussian_filter
import h5py
import sys


def gridSimulation(Ndendrites: int, Ng: int, sigma, duration, stationary, visualize, visualize_tick, spike_plot, save_data, save_tick: int, file_name):
    # Total number of dendrites
    Ndendrites2 = Ndendrites**2

    # Set up input locations
    spatialns = utils.CoordinateSamplers(Ndendrites, sigma)

    # Set the rate of information from sensory to grid cells
    theta_rate = 1/10 # denominator is theta frequency used

    # Simulation variables
    if save_data:
        weight_tracker = np.zeros((int(duration // save_tick + 1), Ng*Ndendrites2))
        score_tracker = np.zeros((int(duration // save_tick + 1), Ng))
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
        for y_ in range (n_rows):
            for z in range(Ng+1):
                ax.append(plt.subplot2grid((n_rows,Ng+1),(y_,z)))



    # Read file to get trajectory and speed
    if stationary:
        rand_theta = np.random.uniform(0, 2*np.pi)
        pos = np.outer(np.linspace(0, 3*sigma, 3, True), np.array([np.cos(rand_theta), np.sin(rand_theta)]))
        X = np.concatenate((np.zeros((9, 2)), pos, np.ones((int(duration // 100 - (np.size(pos, 0) + 9)), 2)) * pos[-1]), axis = 0) + 0.5
        delta_t = 100
    else:
        X, speed = utils.getCoords(h5py.File("grid_simulation/trajectory_square_2d_0.01dt_long.hdf5", "r"))
        delta_t = 10 # Sampling rate (ms) in the trajectory file
        mean_speed = np.mean(speed)


    # Input layer setup
    filter = 18

    # Precalculate the entire firing of the spike generator group (to avoid having to restart runs when positions update):
    print("Precalculate spatial input")

    end_ix = int(duration / delta_t * 10 * theta_rate)
    step = int(1000*theta_rate/delta_t)
    x = X[0:end_ix:step, :]
    activity = np.round(spatialns.dist(x)/sigma*10 + np.max(2*np.random.rand(int(end_ix/step), Ndendrites2)-1, 0), 1)
    act_indices = np.where(activity < filter)
    activation_times = activity[act_indices] + 100 * act_indices[0]
    neuron_indices = act_indices[1]

    input_layer = SpikeGeneratorGroup(Ndendrites2, neuron_indices, activation_times*ms)

    tau_d = 10*ms
    taupre = 8*ms
    taupost = 80*ms
    Apre = 0.01
    Apost = -0.007  

    dendrite_eq = '''dv/dt = -v/tau_d : 1
                    dapost/dt = -apost/taupost : 1
                    dapre/dt = -apre/taupre : 1
                    c : 1
                    Ve = tanh(v)*c : 1
                    l_speed : 1'''
    base_conductivity = 25 / Ndendrites2
    c_max = 1.5*base_conductivity
    dendrite_layer = NeuronGroup(Ndendrites2 * Ng, dendrite_eq, method = 'exact')
    conductivities = np.random.rand(Ndendrites2 * Ng)*base_conductivity
    dendrite_layer.c = conductivities 

    baseline_effect = 10 # Factor that determines how much conductivity strengthens upon input - I think 20 is considered a strong baseline here.
    input_to_dendrites = Synapses(input_layer, dendrite_layer, '''w : 1''', 
                                on_pre = '''
                                v_post +=w
                                c = clip(c + l_speed_post*(apost + baseline_effect/(Ng*Ndendrites2)*(c_max-c)), 0, c_max)
                                apre_post += Apre''')
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
                '''Igap_post = Ve_pre : 1 (summed)''', on_post = '''
                c_pre = clip(c_pre + l_speed_pre*apre_pre, 0, c_max)
                apost_pre += Apost''')
    dendrite_to_grid.connect(condition = 'j == i // Ndendrites2')



    # # Set up inhibitory layer:
    tau_i = 10*ms
    inhibit_eq = '''dv/dt = -v/tau_i : 1 (unless refractory)'''
    inhibit_layer = NeuronGroup(1, inhibit_eq, threshold='v > 0.5', reset = 'v = 0', refractory = 30*ms, method = 'exact')

    grid_to_inhibit = Synapses(grid_layer, inhibit_layer, 'w : 1', on_pre = 'v_post += w', delay = 1*ms)
    grid_to_inhibit.connect()
    grid_to_inhibit.w = 1

    inhibit_to_grid = Synapses(inhibit_layer, grid_layer, 'w : 1', on_pre = 'y_post += Ndendrites2/200')
    inhibit_to_grid.connect()

    @network_operation(dt = theta_rate*ms)
    def update_learning_rate(t):
        if stationary:
            learning_speed = 1
        else:
            current_speed = speed[int(t/(delta_t*ms))]
            learning_speed = 0.8*np.exp(-(mean_speed-current_speed)**2/mean_speed)
        dendrite_layer.l_speed = learning_speed

    if visualize:
        @network_operation(dt = visualize_tick*ms)
        def update_plot(t):
            # Visualize grid weights if wanted
            time_ms = t/ms
            x = X[int(time_ms/delta_t), :]
            pos_plot = np.zeros((Ndendrites, Ndendrites))
            x_ind = np.ndarray.astype(Ndendrites * x, int)
            pos_plot[tuple(x_ind)] = 1
            ax[0].cla()
            ax[0].imshow(pos_plot, interpolation='none', origin='lower')


            grid_weights = np.reshape(dendrite_layer.c, (Ng, Ndendrites2))
            
            mean_weights = np.reshape(np.mean(grid_weights, axis = 0 ), (Ndendrites, Ndendrites))
            ax[Ng+1].cla()
            ax[Ng+1].imshow(mean_weights, interpolation = 'none', origin = 'lower')
            ax[Ng+1].set_title("mean")

            position_hist, _, __ = (np.histogram2d(X[0:int(time_ms/delta_t), 1], X[0:int(time_ms/delta_t), 0], 20, [[0,1],[0,1]]))

            ax[2*Ng+2].cla()
            ax[2*Ng+2].imshow(position_hist, interpolation = 'none', origin = 'lower')
            ax[2*Ng+2].set_title("trajectory")

            mean_score = 0

            for z in range(Ng):
                
                weight2d = np.reshape(grid_weights[z, :], (Ndendrites, Ndendrites))
        
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
                ax[z+1].cla()
                ax[z+1].imshow(weight2d, interpolation='none', origin='lower')
                ax[z+1].set_title("%3.4f" % (gscore))

                # show gaussian filtered weights
                ax[2+Ng+z].cla()
                ax[2+Ng+z].imshow(gaussian_filter(weight2d, 1.5), interpolation='none', origin = 'lower')

                # show auto-correlation and nearest blod tracker
                ax[3+2*Ng+z].cla()
                ax[3+2*Ng+z].imshow(corr_w, interpolation='none', origin='lower')
                ax[3+2*Ng+z].autoscale(False)
                ax[3+2*Ng+z].plot([cntr_xy, cntr_xy + closest_r[0]], [cntr_xy, cntr_xy + closest_r[1]], linewidth=2.0, color='black')
            ax[2+Ng].set_title("%3.4f" % (mean_score))
            fig.suptitle(f"{t/second // 60} mins {(t/second) % 60} seconds")
            plt.pause(3)

    if save_data:
        @network_operation(dt = save_tick*ms)
        def save_weights(t):
            sys.stdout.write("\rProgress: %3.4f" % ((t/ms)/duration))
            sys.stdout.flush()
            weight_tracker[save_id[0], ...] = dendrite_layer.c

            grid_weights = np.reshape(dendrite_layer.c, (Ng, Ndendrites2))
            for z in range(Ng):    
                weight2d = np.reshape(grid_weights[z, :], (Ndendrites, Ndendrites))
                corr_w = utils.normcorr2d(weight2d)
                gscore, _ = utils.gridness_score(corr_w, Ndendrites, sigma)
                score_tracker[save_id[0],z] = gscore
            save_id[0]+=1

    if spike_plot:
        M = SpikeMonitor(input_layer)
        G = SpikeMonitor(grid_layer)
        S = StateMonitor(grid_layer, True, [0,1,2] if Ng >= 3 else 0)
        I = SpikeMonitor(inhibit_layer)

    print("Initialize done")

    run(duration*ms)

    if save_data:
        np.savez_compressed(f'grid_simulation/Results/{file_name}', \
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
        plt.subplot(221)
        plt.plot(M.t/ms, M.i, '.k')
        plt.vlines(G.t/ms, 0, Ndendrites2)
        plt.vlines(I.t/ms, 0, Ndendrites2, colors = 'r')
        plt.subplot(222)
        plt.plot(S.t/ms, S.v[0], 'C0')
        plt.plot(S.t/ms, S.Igap[0], 'green')
        plt.plot(S.t/ms, S.y[0], 'red')
        if Ng >= 3:
            plt.subplot(223)
            plt.plot(S.t/ms, S.v[1], 'C0')
            plt.plot(S.t/ms, S.Igap[1], 'green')
            plt.plot(S.t/ms, S.y[1], 'red')
            plt.subplot(224)
            plt.plot(S.t/ms, S.v[2], 'C0')
            plt.plot(S.t/ms, S.Igap[2], 'green')
            plt.plot(S.t/ms, S.y[2], 'red')
        plt.show()

if __name__ == '__main__':
    Ndendrites = 24
    Ng = 13
    sigma = 0.1
    stationary = False
    visualize = True
    if stationary:
        duration = 3000
        visualize_tick = 200
    else:
        duration = 0.01 * 10**6
        visualize_tick = 1000
    spike_plot = not visualize
    save_data = True
    save_tick = 1000
    output_filename = 'test.npz'
    gridSimulation(Ndendrites, Ng, sigma, duration, stationary, visualize, visualize_tick, spike_plot, save_data, save_tick, output_filename)
