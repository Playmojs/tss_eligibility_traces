from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
import utils
import time
import copy
from scipy.ndimage import gaussian_filter
import h5py
import sys

def gridSimulation(Ndendrites, Ng, sigma, baseline_effect, duration, stationary, input_distribution, visualize_tick, plot_spike_hist, plot_weights, spike_plot, save_data, save_tick, file_name):
    # Total number of dendrites
    Ndendrites2 = Ndendrites**2

    # Set up input locations
    spatialns = utils.CoordinateSamplers(Ndendrites, sigma, distrib = input_distribution)

    # Set the rate of information from sensory to grid cells
    theta_rate = 1/10 # denominator is theta frequency used

    # Simulation variables
    
    if save_data:
        weight_tracker = np.zeros((duration*1000 // save_tick + 1, Ng*Ndendrites2))
        score_tracker = np.zeros((duration*1000 // save_tick + 1, Ng))
    save_id = [0]

    visualize = plot_spike_hist or plot_weights

    # Prepare plot 
    if visualize:
        import matplotlib
        import matplotlib.pyplot as plt

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


    # Read file to get trajectory and speed
    rb = False
    if not rb:
        X, speed = utils.getCoords(h5py.File("grid_simulation/Trajectories/trajectory_square_2d_0.01dt_ultra.hdf5", "r"))
        delta_t = 10 # Sampling frequency in the trajectory file, in ms
    else:
        X, speed, _, __ = utils.getTrajValues(f"grid_simulation/Trajectories/Square/7200s.npz")
        X += 0.5
        delta_t = 100
    mean_speed = np.mean(speed)
    tMax = len(X)


    # Input layer setup
    inputs = np.empty((2,0))
    filter = 18

    current_time = time.time()

    # Precalculate the entire firing of the spike generator group (to avoid having to restart runs when positions update):
    print("Precalculate spatial input:")

    end_ix = int(duration*100/(delta_t*theta_rate))
    step = int(1000 * theta_rate/delta_t)
    x = X[0:end_ix:step, :] #TODO: add stationary here
    activity = np.round(spatialns.dist(x)/sigma*10 + (np.random.normal(0, 2,(end_ix//step, Ndendrites2))), 1)
    act_indices = np.where(activity < filter)
    activation_times = activity[act_indices] + 100 * act_indices[0]
    activation_times[activation_times<0] = 0
    neuron_indices = act_indices[1]

    input_layer = SpikeGeneratorGroup(Ndendrites2, neuron_indices, activation_times*ms)
    print(f"Time: {time.time() - current_time}")

    #Grid layer and inhibitory layer:

    tau_g = 10*ms
    grid_eq = '''dv/dt = - v / tau_g : 1 (unless refractory)'''

    grid_layer = NeuronGroup(Ng, grid_eq, threshold = 'v > 1.0', reset = 'v = -0.1', refractory= 30*ms, method = 'exact')

    # Set up synapses from input to grid layer with STDP learning rule and randomized start weights

    taupre = 8*ms
    taupost = 80*ms
    wmax_i = 3.5 / Ndendrites
    Apre = 0.01
    Apost = -0.007
    baseline_effect = baseline_effect
    input_weights = Synapses(input_layer, grid_layer, '''
                w : 1
                l_speed : 1
                dapre/dt = -apre/taupre : 1 (event-driven)
                dapost/dt = -apost/taupost : 1 (event-driven)
                ''',
                on_pre='''
                v_post += w
                apre += Apre
                w = clip(w+(apost+baseline_effect*(wmax_i-w))*l_speed, 0, wmax_i)
                ''',
                on_post='''
                apost += Apost
                w = clip(w+apre*l_speed, 0, wmax_i)
                ''', delay = 3 * ms)
    input_weights.connect()

    weights = np.random.rand(Ndendrites2 * Ng) * 0.85 * wmax_i
    input_weights.w = weights

    # # Set up inhibitory layer:
    inhibit_layer = NeuronGroup(Ng, grid_eq, threshold='v > 0.5', refractory = 0*ms, reset = 'v = 0', method = 'exact')

    grid_to_inhibit = Synapses(grid_layer, inhibit_layer, 'w : 1', on_pre = 'v_post += w', delay = 0.6*ms)
    grid_to_inhibit.connect(condition = 'i==j')
    grid_to_inhibit.w = 0.7

    inhibit_to_grid = Synapses(inhibit_layer, grid_layer, 'w : 1', on_pre = 'v_post = -7')
    inhibit_to_grid.connect(condition = 'i!=j')
    inhibit_to_grid.w = 2

    nu = 1
    @network_operation(dt = theta_rate*ms)
    def update_learning_rate(t):
        if stationary:
            learning_speed = 1
        else:
            current_speed = speed[int(t/(delta_t*ms))]
            
            learning_speed = nu * np.exp(-(mean_speed-current_speed)**2/mean_speed)
        input_weights.l_speed = learning_speed

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

        grid_weights = np.reshape(input_weights.w, (Ndendrites2, Ng))
        
        if n_rows > 1:
            mean_weights = np.reshape(np.mean(grid_weights, axis = 1 ), (Ndendrites, Ndendrites))
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
                spike_indices = np.floor(spike_times/delta_t)
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
            weight2d = np.reshape(grid_weights[:, z], (Ndendrites, Ndendrites))
    
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

    @network_operation(dt = save_tick*ms)
    def save_weights(t):
        if not save_data:
            return
        sys.stdout.write("\rProgress: %3.4f" % ((t/second + save_tick/1000)/duration))
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
        I = SpikeMonitor(inhibit_layer)
    if spike_plot or plot_spike_hist or save_data:
        G = SpikeMonitor(grid_layer)
    print("Initialize done")

    run(duration*second)

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
            wmax = wmax_i)

    if visualize:
        plt.show()

    if spike_plot:
        import matplotlib.pyplot as plt
        # print(G.t/ms)
        # print(G.i)
        # print(R.t/ms)
        plt.figure(figsize = (12,12))
        
        plt.plot(M.t/ms, M.i, '.k')
        plt.vlines(G.t/ms, 0, Ndendrites2)
        plt.vlines(I.t/ms, 0, Ndendrites2, colors = 'r')
        # plt.subplot(222)
        # plt.plot(S.t/ms, S.v[0], 'C0')
        # plt.subplot(223)
        # plt.plot(S.t/ms, S.v[1], 'C0')
        # plt.subplot(224)
        # plt.plot(S.t/ms, S.v[2], 'C0')
        np.savez_compressed("grid_simulation/Results/input_spikes", \
                            input_spikes = M.t/ms, \
                            input_id = M.i, \
                            grid_spikes = G.t/ms, \
                            grid_id = G.i, \
                            inhibit_spikes = I.t/ms, \
                            inhibit_id = I.i)
        plt.show()


if __name__ == '__main__':
    distrib = 'regular'
    print(f"Input distribution: {distrib}")
    Ndendrites = 24
    Ng = 13
    gridSimulation(Ndendrites, Ng, 0.12, 1.6 / (Ndendrites*Ng), 500, False, distrib, 10000, True, True, False, False, 10000, f'data/{distrib}_3000s.npz')