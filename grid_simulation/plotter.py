import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import utils
import sys
import h5py
import os
from scipy.ndimage import gaussian_filter
from brian2 import second

def LinePlot(input_files, legends, add_error_bars = False, groupings = None):
    # Preliminary file-reading for plot params:
    n_files = len(input_files)
    save_ticks = np.zeros(n_files)
    Ngs = np.zeros(n_files, dtype = int)
    durations = np.zeros(n_files)
    for i, file in enumerate(input_files):
        f = np.load(str(file))
        save_ticks[i] = f['save_tick'] / 1000
        Ngs[i] = f['Ng']
        durations[i] = f['duration']
    min_tick = np.min(save_ticks)
    max_duration = np.max(durations)
    max_Ng = np.max(Ngs)
    if type(groupings) == None:
        groupings = np.arange(len(input_files))
    uniq, counts = np.unique(groupings, return_counts = True)
    n_groups = len(uniq)
    max_count = np.max(counts)
    
    scores = np.empty((n_groups, max_count, max_Ng, int(max_duration//min_tick) + 1))
    x_vals = np.arange(max_duration//min_tick + 1)*min_tick
    group_iter = np.zeros(n_groups, dtype = int)

    for i, file in enumerate(input_files):
        f = np.load(file, allow_pickle=True)
        group_i = groupings[i]
        max_ind = int(durations[i]//min_tick)
        step = int(save_ticks[i]//min_tick)
        scores[group_i, group_iter[group_i], :Ngs[i], 0:(max_ind+step):step] = f['scores'].T
        group_iter[group_i] += 1
    mean_scores = np.mean(scores, axis = (1,2))
    std_scores = np.std(scores, axis = (1,2))

    fig = plt.figure()

    plt.plot(x_vals / 60, mean_scores.T)
    if add_error_bars:
        pass
        #plt.fill_between(x_vals // 60, y_mean - deviation[0:-1], y_mean + deviation[0:-1], alpha = 0.3, label = '_nolegend_')

    plt.hlines(0, -20, max_duration + 5, linestyles = 'dashed', colors = 'blue')
    plt.xlabel("Time(mins)")
    plt.ylabel("Mean gridscore across 13 grid cells")
    plt.xlim(-5, max_duration/60 + 5)
    #plt.rcParams.update({'font.size': 100})
    plt.legend(legends)
    plt.show()

def WeightsGif(input_file, output_file):
    f = np.load('grid_simulation/Results/' + input_file)
    Ndendrites = f['Ndendrites']
    Ndendrites2 = Ndendrites**2
    sigma = f['sigma']
    Ng = int(f['Ng'])
    save_tick = int(f['save_tick'])
    duration = int(f['duration'])
    weights = f['weights']
    Ng = 13
    n_rows = 3
    fig = plt.figure(figsize=(15,10))
    ax = []
    for y in range (n_rows):
        for z in range(Ng):
            ax.append(plt.subplot2grid((n_rows,Ng),(y,z)))
    
    def init():
        ax[0].scatter([],[])
    
    def update(frame):

    
        grid_weights = np.reshape(weights[frame], (Ndendrites2, Ng))
        print(save_tick/1000*frame/duration)

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
            ax[z].imshow(weight2d, interpolation='none', origin='lower')
            ax[z].set_title("%3.2f" % (gscore))

            # show gaussian filtered weights
            ax[Ng+z].imshow(gaussian_filter(weight2d, 1.5), interpolation='none', origin = 'lower')

            # show auto-correlation and nearest blod tracker
            ax[2*Ng+z].cla()
            ax[2*Ng+z].imshow(corr_w, interpolation='none', origin='lower')
            ax[2*Ng+z].autoscale(False)
            ax[2*Ng+z].plot([cntr_xy, cntr_xy + closest_r[0]], [cntr_xy, cntr_xy + closest_r[1]], linewidth=2.0, color='black')
        fig.suptitle(f"{save_tick*frame // 60000} mins {(save_tick*frame/1000) % 60} seconds")
    ani = animation.FuncAnimation(fig, update, duration*1000//save_tick, init, interval = 2000)
    ani.save(f"grid_simulation/Results/{output_file}.gif")

def estimateOptimalSigma(filename, use_gaussian_filter = False):
    with np.load('grid_simulation/Results/'+filename) as data:
        scores = data['scores']
        weights = data['weights']
        sigma = data['sigma']
        Ndendrites = int(data['Ndendrites'])
        Ng = int(data['Ng'])
    Ndendrites2 = Ndendrites ** 2
    n_samples = 5
    n_data_points = len(scores)
    inds = np.random.choice(n_data_points//2, n_samples) + n_data_points // 2
    sigmas = np.linspace(sigma*0.6, sigma*1.0, 20)
    grid_scores = np.zeros((n_samples, len(sigmas)))
    for i, ind in enumerate(inds):
        grid_weights = np.reshape(weights[ind], (Ndendrites2, Ng))
        for z in range(Ng):
            grid2d = np.reshape(grid_weights[:,z], (Ndendrites, Ndendrites))
            if use_gaussian_filter:
                grid2d = gaussian_filter(grid2d, 1)
            corr_w = utils.normcorr2d(grid2d)
            for j, sig in enumerate(sigmas):
                gscore, _ = utils.gridness_score(corr_w, Ndendrites, sig)
                grid_scores[i, j] += gscore / Ng
    mean_scores = np.mean(grid_scores, axis = 0)
    plt.plot(sigmas, mean_scores)
    plt.show()
    return(grid_scores)

def calculateGridScores(filename, output_filename, sigma, use_gaussian_filter = False, format = 'theta'):
    with np.load('grid_simulation/Results/'+filename) as data:
        scores = data['scores']
        weights = data['weights']
        Ndendrites = int(data['Ndendrites'])
        Ng = int(data['Ng'])
    Ndendrites2 = Ndendrites ** 2
    n_points = len(scores)
    g_scores = np.zeros((n_points, Ng))
    match format:
        case 'theta':
            shape = tuple((Ndendrites, Ndendrites, Ng))
            iterate_axis = 2
        case 'GJ':
            shape = tuple((Ng, Ndendrites, Ndendrites))
            iterate_axis = 0
        case _:
            raise ValueError("Unsupported shape")

    for i in range(n_points):
        sys.stdout.write("\rProgress: %3.4f" % ((i+1) /n_points))
        sys.stdout.flush()
        grid_weights = np.moveaxis(np.reshape(weights[i], shape), iterate_axis, 0)
        for z in range(Ng):
            grid2d = grid_weights[z]
            if use_gaussian_filter:
                grid2d = gaussian_filter(grid2d, 1)
            corr_w = utils.normcorr2d(grid2d)
            gscore, _ = utils.gridness_score(corr_w, Ndendrites, sigma)
            g_scores[i, z] = gscore
    print(np.max(np.mean(g_scores, axis = 1)))
    np.save(f"grid_simulation/Results/{output_filename}", g_scores)

def gridPlotFromSpikeData(spike_trains, X, time_s, duration_s, sigma, Ndendrites, pxs = 48, dt = 10, title = "", plot_weights = False, weights = [[0,0]]):
    fig, axs = plt.subplots(nrows = 3 + plot_weights, ncols = len(spike_trains)+1)
    time_filter = np.clip(time_s - duration_s, 0, time_s)
    relevant_positions = X[time_filter*100:time_s*100: int(100 / dt)]
    position_hist, _, __ = np.histogram2d(relevant_positions[:,1], relevant_positions[:,0], pxs, [[0,1], [0,1]])
    
    axs[0,0].imshow(position_hist, interpolation = 'none', origin = 'lower')
    axs[0,0].axis('off')
    axs[1,0].axis('off')
    axs[2,0].axis('off')
    if plot_weights:
        axs[3,0].axis('off')
    mean_score = 0
    for z in spike_trains:
        spike_times = spike_trains[z]/second
        spike_times = spike_times[np.logical_and(spike_times > time_filter, spike_times < time_s)]
        spike_indices = np.floor(spike_times*int(1000/dt))
        spike_positions = X[np.ndarray.astype(spike_indices, int)]
        spike_hist, _, __ = np.histogram2d(spike_positions[:,1], spike_positions[:,0], pxs, [[0,1],[0,1]])
        #spike_hist = spike_hist**2
        gauss_spike_hist = gaussian_filter(spike_hist, 2)
        if len(spike_positions) == 0:
            spike_positions = np.vstack((spike_positions, [-10,-10]))
            gauss_gscore = 0
            corr_gauss = [[0,0]]
            cntr_xy = 0
        else: #lazily avoid divide by zero warnings by only doing gscore if the cells have spiked within the time window
            # Get grid score:
            corr_gauss = utils.normcorr2d(gauss_spike_hist)
            gauss_gscore, _ = utils.gridness_score(corr_gauss, pxs, sigma)
            axs[0, z+1].set_title("%3.4f" % (gauss_gscore))
            mean_score += gauss_gscore/len(spike_trains)
            cntr_xy = corr_gauss.shape[0]//2

        axs[0, z+1].imshow(spike_hist, interpolation='none', origin='lower')
        axs[0, z+1].axis('off')

        axs[1, z+1].imshow(gauss_spike_hist,  interpolation='none', origin = 'lower')
        axs[1, z+1].axis('off')

        if gauss_gscore > 0:
                _, closest_r, _ = utils.grid_orientation(corr_gauss, Ndendrites, sigma)
        else:
            closest_r = np.array([0, 0])

        axs[2, z+1].cla()
        axs[2, z+1].imshow(corr_gauss, interpolation='none', origin='lower')
        axs[2, z+1].autoscale(False)
        axs[2, z+1].plot([cntr_xy, cntr_xy + closest_r[0]], [cntr_xy, cntr_xy + closest_r[1]], linewidth=2.0, color='black')
        axs[2, z+1].axis('off')

        if plot_weights:
            weight2d = np.reshape(weights[:, z], (Ndendrites, Ndendrites))
            weight_corr = utils.normcorr2d(weight2d)
            gscore, _ = utils.gridness_score(weight_corr, Ndendrites, sigma)
            axs[3, z + 1].imshow(weight2d, interpolation='none', origin='lower')
            axs[3, z + 1].set_title(("%3.4f" % (gscore)))
            axs[3, z + 1].axis('off')

    axs[0,0].set_title("Mean:\n %3.4f" % (mean_score))
    fig.suptitle(title)

    


if __name__ == '__main__':
    basepath = 'grid_simulation/Results/data/24dendrites/'
    files = utils.getSortedEntries(basepath, 'npz')
    # for entry in os.listdir(basepath):
    #     tot_entry = os.path.join(basepath, entry)
    #     if os.path.isfile(tot_entry):
    #         input_files.append(tot_entry)
    groupings = np.repeat(np.arange(6), 5)

    legends = ['Low base', 'Normal', 'Lowest Wmax', 'Old', 'bla', 'blabla']
    LinePlot(files, legends, False, groupings = groupings)
    #estimateOptimalSigma("data/m3f100_1.npz", True)
    # calculateGridScores("data/model2_6000s_0.npz", "data/model2_6000s_0_opt_g_score", 0.115, False, 'GJ')
    # calculateGridScores("data/m3f100_1.npz", "data/mf3100_1_opt_g_score", 0.086, False, 'theta')
