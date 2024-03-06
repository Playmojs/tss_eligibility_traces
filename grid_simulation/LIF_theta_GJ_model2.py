from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import tanh
import utils
from scipy.ndimage import gaussian_filter
import h5py
import sys


def gridSimulation(Ndendrites, Ng, sigma, baseline_effect, duration, stationary, input_distribution, visualize_tick, plot_spike_hist, plot_weights, spike_plot, save_data, save_tick, file_name):
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
    visualize = plot_spike_hist or plot_weights
    if visualize:

        # suppress deprecation warning from matplotlib.
        import warnings
        warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
        pxs = 48
        n_rows = 2 * plot_spike_hist + 3 * plot_weights
        fig = plt.figure()
        ax = []
        for y in range (n_rows):
            for z in range(Ng+1):
                ax.append(plt.subplot2grid((n_rows,Ng+1),(y,z)))



    if stationary:
        rand_theta = np.random.uniform(0, 2*np.pi)
        pos = np.outer(np.linspace(0, 3*sigma, 3, True), np.array([np.cos(rand_theta), np.sin(rand_theta)]))
        X = np.concatenate((np.zeros((9, 2)), pos, np.ones((int(duration // 100 - (np.size(pos, 0) + 9)), 2)) * pos[-1]), axis = 0) + 0.5
        delta_t = 100
    else:
        X, speed = utils.getCoords(h5py.File("grid_simulation/Trajectories/trajectory_square_2d_0.01dt_long.hdf5", "r"))
        delta_t = 10 # Sampling rate (ms) in the trajectory file
        mean_speed = np.mean(speed)


    # Input layer setup
    filter = 18

    # Precalculate the entire firing of the spike generator group (to avoid having to restart runs when positions update):
    print("Precalculate spatial input")

    end_ix = int(duration / delta_t * 10 * theta_rate)
    step = int(1000*theta_rate/delta_t)
    x = X[0:end_ix:step, :]
    activity = np.round(spatialns.dist(x)/sigma*10 + (np.random.normal(0, 2,(end_ix//step, Ndendrites2))), 1)
    act_indices = np.where(activity < filter)
    activation_times = activity[act_indices] + 100 * act_indices[0]
    neuron_indices = act_indices[1]

    input_layer = SpikeGeneratorGroup(Ndendrites2, neuron_indices, activation_times*ms)

    tau_d = 10*ms
    taupre = 8*ms
    taupost = 80*ms
    Apre = 0.01
    Apost = -0.007  
    c_max = 30 / Ndendrites2

    dendrite_eq = '''dv/dt = -v/tau_d : 1
                    dapost/dt = -apost/taupost : 1
                    dapre/dt = -apre/taupre : 1
                    c : 1
                    Ve = tanh(v)*c : 1
                    l_speed : 1'''
    base_conductivity = 0.75*c_max
    dendrite_layer = NeuronGroup(Ndendrites2 * Ng, dendrite_eq, method = 'exact')
    conductivities = np.random.rand(Ndendrites2 * Ng)*base_conductivity
    dendrite_layer.c = conductivities 

    baseline_effect = baseline_effect # Factor that determines how much conductivity strengthens upon input - I think 20 is considered a strong baseline here.
    input_to_dendrites = Synapses(input_layer, dendrite_layer, '''w : 1''', 
                                on_pre = '''
                                v_post +=w
                                c = clip(c + l_speed_post*(apost + baseline_effect*(c_max-c)), 0, c_max)
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

    grid_to_inhibit = Synapses(grid_layer, inhibit_layer, 'w : 1', on_pre = 'v_post += w', delay = 0.6*ms)
    grid_to_inhibit.connect()
    grid_to_inhibit.w = 1

    inhibit_to_grid = Synapses(inhibit_layer, grid_layer, 'w : 1', on_pre = 'y_post += Ndendrites2/200')
    inhibit_to_grid.connect()

    nu = 0.5
    @network_operation(dt = theta_rate*ms)
    def update_learning_rate(t):
        if stationary:
            learning_speed = 1
        else:
            current_speed = speed[int(t/(delta_t*ms))]
            learning_speed = nu*np.exp(-(mean_speed-current_speed)**2/mean_speed)
        dendrite_layer.l_speed = learning_speed

    @network_operation(dt = visualize_tick*ms)
    def update_plot(t):
        # Visualize grid weights if wanted
        if not visualize:
            return
        
        time_ms = t/ms
        # x = X[int(time_ms/delta_t), :] if not stationary else X[0,:] +  np.array([sigma/2, sigma/2])* (np.floor(time_ms/1000))
        # i68, i95, i99 = spatialns.get68_95(np.array([x]))
        # ax[0].cla()
        # ax[0].imshow(np.reshape(spatialns.act(np.array([x])) * i68, (Ndendrites, Ndendrites)), interpolation='none', origin='lower')
        
        position_hist, _, __ = (np.histogram2d(X[0:int(time_ms/delta_t), 1], X[0:int(time_ms/delta_t), 0], pxs, [[0,1],[0,1]]))

        ax[0].cla()
        ax[0].imshow(position_hist, interpolation = 'none', origin = 'lower')

        grid_weights = np.reshape(dendrite_layer.c, (Ng, Ndendrites2))
        
        if n_rows > 1:
            mean_weights = np.reshape(np.mean(grid_weights, axis = 0 ), (Ndendrites, Ndendrites))
            ax[Ng + 1].cla()
            ax[Ng + 1].imshow(mean_weights, interpolation = 'none', origin = 'lower')

        mean_score = 0

        if plot_spike_hist:
            spike_trains = G.spike_trains()
            time_filter = time_ms - 300000 if time_ms < 2000000 else 1500000
            visited_pxs = position_hist>0

        for i in range(n_rows):
            ax[i*(Ng +1)].axis('off')

        for z in range(Ng):
            
            i = 0
            if plot_spike_hist:

                # Spike histogram
                
                ax[i * (Ng + 1) + z + 1].cla()
                ax[(i + 1) * (Ng + 1) + z + 1].cla()

                spike_times = spike_trains[z]/ms
                spike_times = spike_times[spike_times > time_filter]
                spike_indices = np.floor(spike_times/10)
                spike_positions = X[np.ndarray.astype(spike_indices, int)]
                spike_hist, _, __ = np.histogram2d(spike_positions[:,1], spike_positions[:,0], pxs, [[0,1],[0,1]])
                gauss_spike_hist = gaussian_filter(spike_hist, 1)
                if len(spike_positions) == 0:
                    spike_positions = np.vstack((spike_positions, [-10,-10]))
                else: #lazily avoid divide by zero warnings by only doing gscore if the cells have spiked within the time window
                    # Get grid score:

                    corr_gauss = utils.normcorr2d(gauss_spike_hist)
                    gauss_gscore, _ = utils.gridness_score(corr_gauss, pxs, sigma)
                    ax[(i + 1) * (Ng + 1) + z + 1].set_title("%3.4f" % (gauss_gscore))
                    mean_score += gauss_gscore/Ng

                spike_hist[visited_pxs] = spike_hist[visited_pxs] / position_hist[visited_pxs]
                spike_hist_max = 1 if stationary else np.max(spike_hist)
                ax[i * (Ng + 1) + z + 1].imshow(spike_hist, vmax = spike_hist_max, interpolation='none', origin='lower')
                ax[i * (Ng + 1) + z + 1].axis('off')
                i += 1

                ax[i * (Ng + 1) + z + 1].imshow(gauss_spike_hist + 1, norm = 'log', interpolation='none', origin = 'lower')
                ax[i * (Ng + 1) + z + 1].axis('off')
                i += 1

            if not plot_weights:
                continue
            weight2d = np.reshape(grid_weights[z], (Ndendrites, Ndendrites))
    
            # compute gridness score
            corr_w = utils.normcorr2d(weight2d)
            gscore, _ = utils.gridness_score(corr_w, Ndendrites, sigma)
            cntr_xy = corr_w.shape[0]//2
            if not plot_spike_hist:
                mean_score += gscore/Ng
            # only consider cells with a score > 0 (as is common in
            # literature)
            if gscore > 0:
                orientation, closest_r, _ = utils.grid_orientation(corr_w, Ndendrites, sigma)
            else:
                orientation = -1
                closest_r = np.array([0, 0])

            # show weights
            ax[i * (Ng + 1) + z + 1].cla()
            ax[i * (Ng + 1) + z + 1].imshow(weight2d, interpolation='none', origin='lower')
            ax[i * (Ng + 1) + z + 1].set_title("%3.4f" % (gscore))
            ax[i * (Ng + 1) + z + 1].axis('off')
            i += 1
            # show gaussian filtered weights
            ax[i * (Ng + 1) + z + 1].cla()
            ax[i * (Ng + 1) + z + 1].imshow(gaussian_filter(weight2d, 1.5), interpolation='none', origin = 'lower')
            ax[i * (Ng + 1) + z + 1].axis('off')
            i += 1


            # show auto-correlation and nearest blod tracker
            ax[i * (Ng + 1) + z + 1].cla()
            ax[i * (Ng + 1) + z + 1].imshow(corr_w, interpolation='none', origin='lower')
            ax[i * (Ng + 1) + z + 1].autoscale(False)
            ax[i * (Ng + 1) + z + 1].plot([cntr_xy, cntr_xy + closest_r[0]], [cntr_xy, cntr_xy + closest_r[1]], linewidth=2.0, color='black')
            ax[i * (Ng + 1) + z + 1].axis('off')
            i += 1
        ax[0].set_title("Mean:\n %3.4f" % (mean_score))
        fig.suptitle(f"{t/second // 60} mins {(t/second) % 60} seconds")
        plt.pause(10)

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

    G = SpikeMonitor(grid_layer)

    if spike_plot:
        M = SpikeMonitor(input_layer)
        S = StateMonitor(grid_layer, True, [0,1,2] if Ng >= 3 else 0)
        I = SpikeMonitor(inhibit_layer)

    print("Initialize done")

    run(duration*ms)

    if save_data:
        print()
        spike_data = G.spike_trains()
        np.savez_compressed(f'grid_simulation/Results/{file_name}', \
            Ndendrites = Ndendrites, \
            sigma = sigma, \
            Ng = Ng, \
            save_tick = save_tick, \
            duration = duration, \
            weights = weight_tracker, \
            scores = score_tracker, \
            spike_times = spike_data, \
            input_pos = spatialns.Xs, \
            baseline_effect = baseline_effect, \
            apre = Apre, \
            apost = Apost, \
            nu = nu, \
            wmax = c_max)


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
    sigma = 0.12
    stationary = False
    plot_spike_hist = True
    plot_weights = True
    if stationary:
        duration = 3000
        visualize_tick = 200
    else:
        duration = 0.1 * 10**6
        visualize_tick = 1000
    spike_plot = not (plot_spike_hist or plot_weights)
    save_data = False
    save_tick = 1000
    distrib = 'regular'
    output_filename = 'test.npz'
    baseline_effect = 1.6 / (Ndendrites*Ng)
    gridSimulation(Ndendrites, Ng, sigma, baseline_effect, duration, stationary, distrib, visualize_tick, plot_spike_hist, plot_weights, spike_plot, save_data, save_tick, output_filename)
