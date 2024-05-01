from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import tanh
from numpy import exp
import utils
import time
from scipy.ndimage import gaussian_filter
import h5py
import sys


def gridSimulation(Ndendrites, Ng, sigma, Nthetas, Ndists, baseline_effect, distribution, duration, input_file, dt, stationary, visualize_tick, plot_spike_hist, plot_weights, spike_plot, save_data, save_tick, file_name):
    visualize = plot_spike_hist or plot_weights
    dendrite_plot = False
    
    # Total number of dendrites
    Ndendrites2 = Ndendrites**2

    # Set up boundary cell values
    max_dist = 0.5
    Nbvcs = Nthetas*Ndists
    boundary_cells = np.meshgrid(np.arange(0,180, int(180/Nthetas), dtype =int), np.linspace(0, max_dist, Ndists, False))
    boundary_cells = np.reshape(boundary_cells, (2,-1))

    # Set up input locations
    spatialns = utils.CoordinateSamplers(Ndendrites, sigma)

    # Set the rate of information from sensory to grid cells
    theta_rate = 1/10 # denominator is theta frequency used

    if save_data:
        weight_tracker = np.zeros((int(duration // save_tick) + 1, Ng*Ndendrites2))
        score_tracker = np.zeros((int(duration // save_tick) + 1, Ng))
    save_id = [0]

    # Prepare plot
    def setup_plot_grid(n_rows, n_cols):
        fig = plt.figure()
        ax = []
        for y in range (n_rows):
            for z in range(n_cols):
                ax.append(plt.subplot2grid((n_rows, n_cols),(y,z)))
        return fig, ax

    if visualize:
        pxs = 48
        n_rows = 2 * plot_spike_hist + 3 * plot_weights
        fig, ax = setup_plot_grid(n_rows, Ng+1)

    # Read file to get trajectory and speed and boundaries. The positions are assumed to be sampled at the theta-frequency.
    if stationary:
        print("Generating trajectory")
        import trajectory_gen
        # Pick some direction to move in:
        rand_theta = np.random.uniform(0,2*np.pi)

        boundary_shape = 'square'
        # Create a list of positions on a linear track towards some endpoint, 2*sigma
        pos = np.outer(np.linspace(0, 3*sigma, 3, True), np.array([np.cos(rand_theta), np.sin(rand_theta)]))
        X = np.concatenate((np.zeros((9,2)), pos, np.ones((int(duration // 100 - (np.size(pos, 0) + 9)), 2)) * pos[-1]), axis = 0)
        boundary_vectors = trajectory_gen.generateTrajectory(0.1, np.size(X, 0)/10, "", boundary_shape = boundary_shape, pos = X, save_to_f= False)
        dt = 100

    else:
        # Read file to get trajectory and speed and boundaries. The positions are assumed to be sampled at the theta-frequency.
        X, speed, boundaries, boundary_vectors = utils.getTrajValues(f"grid_simulation/Trajectories/{input_file}")
        mean_speed = np.mean(speed)
        dt = dt

    # Precalculate the entire firing of the spike generator group (to avoid having to restart runs when positions update):
    print("Precalculating spatial input")
    delays = utils.BVC_act(boundary_cells,boundary_vectors[:int(duration//dt):int(100/dt)], Nbvcs, sigma, noise_level= 0.002, alg = 'simple')

    # Filter based on delay
    indices = np.where(delays < 22) 
    spike_times = delays[indices] + 100*indices[0]
    neuron_indices = indices[1]

    BVC_layer = SpikeGeneratorGroup(Nbvcs, neuron_indices, spike_times*ms)

   
    tau_d = 15*ms
    taupre = 8*ms
    taupost = 30*ms
    Apre = 0.01
    Apost = -0.006
    c_max = 27 / Ndendrites2

    dendrite_eq = '''dv/dt = -v/tau_d : 1
                    dapost/dt = -apost/taupost : 1
                    dapre/dt = -apre/taupre : 1
                    c : 1
                    Ve = int(v > 1.1) : 1
                    l_speed : 1'''
    base_conductivity = 0.75*c_max
    dendrite_layer = NeuronGroup(Ndendrites2 * Ng, dendrite_eq, method = 'exact')
    conductivities = np.random.rand(Ndendrites2 * Ng)*base_conductivity
    dendrite_layer.c = conductivities 

    baseline_effect = baseline_effect

    BVC_synapses = Synapses(BVC_layer, dendrite_layer, 'w : 1', on_pre='''v_post +=w
                            V = int(v_post > 1.1)
                            c = clip(c + V*l_speed_post*(apost + baseline_effect*(c_max-c)), 0, c_max)
                            apre += Apre*V''')
    if distribution == "uniform":
        BVC_synapses.connect(p = 0.03)
    else:
        BVC_connections = utils.getBVCtoDendriteConnectivity(Nbvcs, Ndendrites2 * Ng, bvc_params = [Nthetas,Ndists], distribution = distribution, rate = 0.1)
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
                '''Igap_post = Ve_pre*c_pre : 1 (summed)''', on_post = '''
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

    inhibit_to_grid = Synapses(inhibit_layer, grid_layer, 'w : 1', on_pre = 'y_post += Ndendrites2/150')
    inhibit_to_grid.connect()

    nu = 0.3
    @network_operation(dt = theta_rate*ms)
    def update_learning_rate(t):
        if stationary:
            learning_speed = 1
        else:
            current_speed = speed[int(t/(dt*ms))]
            learning_speed = nu*np.exp(-(mean_speed-current_speed)**2/mean_speed)
        dendrite_layer.l_speed = learning_speed

    class Timer():
        current_time = 0
        last_time = 0
        start_time = 0
    timer = Timer()

    def plot_g(t):
        # Visualize grid weights if wanted
        if not visualize:
            return
        if t/ms == 0:
            timer.start_time = time.time()
            timer.last_time = timer.start_time
        else:
            timer.last_time = timer.current_time
        timer.current_time = time.time()
        total_time = timer.current_time - timer.start_time
        middle_time = timer.current_time - timer.last_time
        sys.stdout.write(f"\rRuntime: {total_time//60:.0f} minutes {total_time%60:.0f} seconds. Time since last visualization: {middle_time // 60:.0f} minutes {middle_time % 60:.0f} seconds.")
        time_ms = t/ms
        # x = X[int(time_ms/delta_t), :] if not stationary else X[0,:] +  np.array([sigma/2, sigma/2])* (np.floor(time_ms/1000))
        # i68, i95, i99 = spatialns.get68_95(np.array([x]))
        # ax[0].cla()
        # ax[0].imshow(np.reshape(spatialns.act(np.array([x])) * i68, (Ndendrites, Ndendrites)), interpolation='none', origin='lower')
        
        position_hist, _, __ = (np.histogram2d(X[0:int(time_ms/dt), 1], X[0:int(time_ms/dt), 0], pxs, [[-0.5,0.5],[-0.5,0.5]]))

        ax[0].cla()
        ax[0].imshow(position_hist, interpolation = 'none', origin = 'lower')


        spike_times = G.t/ms
        spike_indices = np.floor(spike_times/dt)
        spike_positions = X[np.ndarray.astype(spike_indices, int)]
        
        if len(spike_positions) == 0:
            spike_positions = np.vstack((spike_positions, [-10,-10]))
        spike_hist, _, __ = np.histogram2d(spike_positions[:,1], spike_positions[:,0], pxs, [[-0.5,0.5],[-0.5,0.5]])

        ax[Ng+1].cla()
        ax[Ng+1].imshow(spike_hist, interpolation = 'none', origin = 'lower')
        ax[Ng+1].set_title("mean")
        
        grid_weights = np.reshape(dendrite_layer.c, (Ng, Ndendrites2))
        
        if n_rows > 2:
            mean_weights = np.reshape(np.mean(grid_weights, axis = 0 ), (Ndendrites, Ndendrites))
            ax[2*(Ng + 1)].cla()
            ax[2*(Ng + 1)].imshow(mean_weights, interpolation = 'none', origin = 'lower')

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
                spike_indices = np.floor(spike_times/dt)
                spike_positions = X[np.ndarray.astype(spike_indices, int)]
                spike_hist, _, __ = np.histogram2d(spike_positions[:,1], spike_positions[:,0], pxs, [[-0.5,0.5],[-0.5,0.5]])
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

    # def plot_dendrite_activity(t, dendrite_state_monitor):
    #     if t == 0*second:
    #         return
    #     pxs = 25 # discretization of positions
        
    #     # get a list of times ten ms after each theta cycle:
    #     time_ids = np.arange(0, t, 0.1)
    #     time_ids += 0.01
    #     time_ids = np.ndarray.astype(time_ids*10000, int)

    #     # get xy - indices for each theta cycle based on the discretization
    #     place_id = int(t/ms/100)
    #     places = X[0:place_id] # Get places at each recorded time until now
    #     places = (places - np.min(boundaries, axis=0)) / (np.max(boundaries, axis = 0) - np.min(boundaries, axis = 0)) # Standardise places to 0-1
    #     places = np.ndarray.astype(np.floor(places*pxs), int) # Discretize places to grid

    #     for z in range(n_den_plots):
    #         plot = np.zeros((pxs,pxs))
    #         plot[places[:,0], places[:,1]] = dendrite_state_monitor.v[z, time_ids]
    #         ax[z].imshow(plot, vmax = 0.8, interpolation='none', origin = 'lower')
    #         ax[z + n_den_plots].imshow(gaussian_filter(plot, 0.5), interpolation = 'none', origin= 'lower')
    #     fig.suptitle(f"{t/second // 60} mins {(t/second) % 60} seconds")
    #     plt.pause(3)


    @network_operation(dt = visualize_tick*ms)
    def update_plot(t):
        if visualize:
            plot_g(t)
        # elif dendrite_plot:
        #     plot_dendrite_activity(t, D)

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
        M = SpikeMonitor(BVC_layer)
        D = StateMonitor(dendrite_layer, ['v','c', 'Ve'], [0,1,2])
        Gs = StateMonitor(grid_layer, ['v', 'Igap'], 0)

    print("Initialize done")
    run(duration*ms)

    if save_data:
        print()
        weight_tracker[-1] = dendrite_layer.c
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
            wmax = c_max, \
            input_file = input_file, \
            distribution = distribution, \
            BVCs = boundary_cells,\
            BVC_connections = BVC_connections, \
            Nthetas = Nthetas, \
            Ndists = Ndists)


    if visualize or dendrite_plot:
        plt.show()

    if spike_plot:
        # print(G.t/ms)
        # print(G.i)
        # print(R.t/ms)
        plt.figure(figsize = (12,12))
        plt.subplot(221)
        # plt.plot(Gs.t/ms, Gs.v[0], 'C0')
        # plt.plot(Gs.t/ms, Gs.Igap[0], 'Magenta')
        plt.plot(M.t/ms, M.i, '.k')
        plt.vlines(G.t/ms, 0, Nbvcs)
        plt.subplot(222)
        plt.plot(D.t/ms, D.v[0], 'C0')
        plt.plot(D.t/ms, D.Ve[0], 'Black')
        plt.plot(D.t/ms, D.c[0]/c_max, 'Green')
        plt.vlines(G.t/ms, 0, 1, color = "red")
        plt.subplot(223)
        plt.plot(D.t/ms, D.v[1], 'C0')
        plt.plot(D.t/ms, D.Ve[1], 'Black')
        plt.plot(D.t/ms, D.c[1]/c_max, 'Green')
        plt.vlines(G.t/ms, 0, 1, color = "red")
        plt.subplot(224)
        plt.plot(D.t/ms, D.v[2], 'C0')
        plt.plot(D.t/ms, D.Ve[2], 'Black')
        plt.plot(D.t/ms, D.c[2]/c_max, 'Green')
        plt.vlines(G.t/ms, 0, 1, color = "red")
        plt.show()
    
    
if __name__ == '__main__':
    Ndendrites = 24
    Ng = 13
    sigma = 0.1
    stationary = False
    plot_spike_hist =  True
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
    output_filename = 'test.npz'
    baseline_effect = 2 / (Ndendrites*Ng)
    input = "Square/7200s_10ms.npz"
    dt = 10
    distribution = 'orthoregular'
    Nthetas = 4
    Ndists = 12
    gridSimulation(Ndendrites, Ng, sigma, Nthetas, Ndists, baseline_effect, distribution, duration, input, dt, stationary, visualize_tick, plot_spike_hist, plot_weights, spike_plot, save_data, save_tick, output_filename)
