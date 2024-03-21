import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib.patches import ConnectionPatch
import utils

def getFilteredInputSpikes(time_window):
    spike_data = [None]*3
    spike_ids = [None]*3
    filters = [None]*3
    filtered_data = [None]*3
    filtered_ids = [None]*3
    with np.load("grid_simulation/Results/input_spikes.npz") as data:
        spike_data[0] = data["input_spikes"]
        spike_ids[0] = data["input_id"]
        spike_data[1] = data["grid_spikes"]
        spike_ids[1] = data["grid_id"]
        spike_data[2] = data["inhibit_spikes"]
        spike_ids[2] = data["inhibit_id"]
    for i, (data, id) in enumerate(zip(spike_data, spike_ids)):
        filters[i] = np.where(np.logical_and(data > time_window[0], data < time_window[1]))
        filtered_data[i] = data[filters[i]]
        filtered_ids[i] = id[filters[i]]
    return filtered_data, filtered_ids

plots = ['orientation_plot']
save = False

if("distribution_plot" in plots):
    inputs = ["grid_simulation/Results/data/24dend2/regular0.npz", "grid_simulation/Results/data/24dend2/noisy_blue0.npz", "grid_simulation/Results/data/24dend2/noisy_white0.npz"]
    ax_title = ["Regular", "Blue noise", "White noise"]
    fig, axs = plt.subplots(1, 3)
    fig.set_facecolor("#212121")
    for input, ax, title in zip(inputs, axs, ax_title):
        input_positions = np.load(input, allow_pickle=True)['input_pos']
        ax.scatter(input_positions[:, 0], input_positions[:, 1], s = 3, color ='steelblue')
        ax.axis('square')
        ax.set_xlim([-0.025,1.025])
        ax.set_ylim([-0.025,1.025])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(title, size = 10, color = "#adadad")
        [x.set_linewidth(2) for x in ax.spines.values()]
    if save:
        fig.savefig("grid_simulation/Results/distribution_plot", dpi = 500, bbox_inches = 'tight')


if("input_plot" in plots):
    time_window = np.array([70, 1000])
    filtered_data, filtered_ids = getFilteredInputSpikes(time_window)
    
    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    ax.scatter(filtered_data[0], filtered_ids[0], s = 7, color = 'black')
    ax.set_xlim(time_window)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_xlabel('Time')
    ax.set_ylabel('Input Cell')
    [x.set_linewidth(2) for x in ax.spines.values()]
    if save:
        fig.savefig("grid_simulation/Documents/Figures/input_plot", dpi = 500, bbox_inches = 'tight')

if("input_STDP_plot" in plots):
    time_window = np.array([1500, 1520])
    filtered_data, filtered_ids = getFilteredInputSpikes(time_window)
    
    color = np.full(len(filtered_data[0]), 'brown')
    color[filtered_data[0] >= filtered_data[1][0]] = 'navy'

    fig, ax = plt.subplots()
    ax.scatter(filtered_data[0], filtered_ids[0], s = 20, c = color)
    ax.vlines(filtered_data[1][0], 100, 450, linewidth = 2, color = 'forestgreen')
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_ylabel('Input Cell', fontsize = 20)
    [x.set_linewidth(2) for x in ax.spines.values()]
    if save:
        fig.savefig("grid_simulation/Documents/Figures/input_STDP_plot", dpi = 500, bbox_inches = 'tight')

if("input_inhibit_plot" in plots):
    time_window = np.array([1500, 1520])
    filtered_data, filtered_ids = getFilteredInputSpikes(time_window)
    
    fig, ax = plt.subplots()
    ax.scatter(filtered_data[0], filtered_ids[0], s = 20, color = 'black')
    ax.vlines(filtered_data[1], 100, 450, linewidth = 2, color = 'forestgreen')
    ax.vlines(filtered_data[2][0], 100, 450, linewidth = 2, color = 'darkmagenta')
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_xlabel('Time', fontsize = 20)
    ax.set_ylabel('Input Cell', fontsize = 20)
    [x.set_linewidth(2) for x in ax.spines.values()]
    if save:
        fig.savefig("grid_simulation/Documents/Figures/input_inhibit_plot", dpi = 500, bbox_inches = 'tight')


if("mouse_plot" in plots):
    input_positions = np.load("grid_simulation/Results/data/24dend2/noisy_blue0.npz", allow_pickle=True)['input_pos']
    diffs = input_positions - np.array([0.5,0.5])
    norm = np.linalg.norm(diffs,axis = 1)

    inn = 0.12
    out = 0.20
    filter_inner = np.where(norm<inn)
    filter_outer = np.where(norm<out)
    colors = np.full(len(input_positions), 'darkslategray')
    alphas = np.full(len(input_positions), 0.2)
    colors[filter_outer] = 'navy'
    colors[filter_inner] = 'brown'
    alphas[filter_outer] = 0.7
    
    fig, ax = plt.subplots()
    im = plt.imread('grid_simulation/Results/mouse.png')
    ax.imshow(im, zorder = 5)

    im_size = 800
    ax.scatter(input_positions[:, 0]*im_size, input_positions[:, 1]*im_size, s = 4, color = colors, alpha = alphas, zorder = 2)

    inner_circle = plt.Circle((im_size/2,im_size/2), im_size*inn, fill = False, zorder = 0, linewidth = 1.5)
    outer_circle = plt.Circle((im_size/2,im_size/2), im_size*out, fill = False, zorder = 0, linewidth = 1.5)
    ax.add_patch(inner_circle)
    ax.add_patch(outer_circle)

    ax.axis('square')
    lim_lo = -0.0025 * 800
    lim_hi = 1.025 * 800
    ax.set_xlim([lim_lo,lim_hi])
    ax.set_ylim([lim_lo,lim_hi])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    [x.set_linewidth(1) for x in ax.spines.values()]
    if save:
        fig.savefig("grid_simulation/Documents/Figures/mouse_plot", dpi = 500, bbox_inches = 'tight')        

if('spike_temp_plot' in plots):
    base_path = "grid_simulation/Results/data/simspam/regular5"
    ng = 13
    pxs = 48
    nrows = 5
    rel_times = np.array([0,5,10,40,95])
    all_times = np.arange(20)
    rel_n = len(rel_times)
    all_n = len(all_times)
    # picks = np.random.choice(ng, nrows, False)
    picks = np.array([0,1,3,4,5])
    hist = np.empty((all_n, ng, pxs, pxs))

    for i in all_times:
        hist[i] = utils.getPopulationSpikePlot(f"{base_path}/{i*5}min_Spikes.npz", 13, 48)
    fig, ax = plt.subplots(nrows, rel_n + 1)
    for r in range (nrows):
        ind = picks[r]
        for z, rel_time in enumerate(rel_times):
            ax[r,z].imshow(hist[rel_time//5, ind],  interpolation='none', origin = 'lower')
            ax[r,z].axis('off')
            ax[0,z].set_title(f"{rel_time} minutes", fontsize = 10)
        ax[r, rel_n].imshow(np.sum(hist[:, ind], 0), interpolation='none', origin = 'lower')
        ax[r,rel_n].axis('off')
    ax[0, rel_n].set_title("Sum", fontsize = 10)
    

    if save:
        fig.savefig("grid_simulation/Documents/Figures/spike_temp_plot", dpi = 500, bbox_inches = 'tight')      

if('gscore_line_plot' in plots):
    gscores = np.load('grid_simulation/Results/analysis/simspam/gscores4.npz')['gscores']
    shape = gscores.shape
    mean_gscores = np.nanmean(gscores, axis = (1,3))
    var_gscores = np.nanvar(gscores, axis = (1,3)) / (shape[1] * shape[3])
    times = np.linspace(0, 100, shape[2], endpoint=False)
    fig, ax = plt.subplots()
    legends = ["Blue Noise", "White Noise", "Regular"]
    for line_mean, line_var in zip(mean_gscores, var_gscores):
        ax.plot(times, line_mean)
        ax.fill_between(times, line_mean + np.sqrt(line_var), line_mean - np.sqrt(line_var), alpha=0.4, interpolate=True, label = '_nolegend_')
    ax.set_ylim([0, 0.65])
    ax.set_xlim([0,95])
    ax.legend(legends)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Gridness Score")
    ax.set_xticks(times[::2])
    if save:
        fig.savefig("grid_simulation/Documents/Figures/gscore_line_plot", dpi = 500, bbox_inches = 'tight')  

if('gscore_boxplot' in plots):
    gscores = np.load('grid_simulation/Results/analysis/simspam/gscores4.npz')['gscores']
    shape = gscores.shape
    end_gscores = np.reshape(gscores[:,:,-1,:],(shape[0], shape[1]*shape[3]))
    fig, ax = plt.subplots()
    ax.boxplot(end_gscores.T)

if('gscore_hist' in plots):
    gscores = np.load('grid_simulation/Results/analysis/simspam/gscores4.npz')['gscores']
    fig, ax = plt.subplots(1, len(gscores))
    time = -1
    for i in range(len(gscores)):
        ax[i].hist(np.ndarray.flatten(gscores[i,:,time, :]), 7, histtype = 'step', color = 'darkslateblue', linewidth = 5,range= (-1.2, 1.6))
        ax[i].set_ylim([0, None])
        ax[i].set_xlim([-1.2, 1.8])
        ax[i].set_xticks([-1, 0, 1])
        ax[i].plot(np.nanmean(gscores[i,:,time,:]), -2.5, "^", markersize = 10, clip_on = False, color='goldenrod')
        if i != 0:
            ax[i].yaxis.set_ticks([])

if('mean_gscore_hist' in plots):
    gscores = np.load('grid_simulation/Results/analysis/simspam/gscores4.npz')['gscores']
    fig, ax = plt.subplots(1, len(gscores))
    time = -1
    mean_gscores = np.nanmean(gscores[:,:,time,:], axis = (2))
    for i in range(len(gscores)):
        ax[i].hist(np.ndarray.flatten(mean_gscores[i]), 3, histtype = 'step', color = 'darkslateblue', linewidth = 5)
        ax[i].set_ylim([0, None])
        ax[i].set_xlim([-1.2, 1.8])
        ax[i].set_xticks([-1, 0, 1])
        ax[i].plot(np.nanmean(gscores[i,:,time,:]), -2.5, "^", markersize = 10, clip_on = False, color='goldenrod')
        if i != 0:
            ax[i].yaxis.set_ticks([])

if ('orientation_plot' in plots):
    orientations = np.load("grid_simulation/Results/analysis/simspam/orientations3.npz")["orientations"]
    thetas = np.arange(60)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    delta_t = 30
    thetas = np.linspace(-np.pi/6, np.pi/6, delta_t)
    histcount, _ = np.histogram((np.ndarray.flatten(orientations) + 30)%60, delta_t)
    X_Y_Spline = make_interp_spline(thetas, histcount)
    x_vals = np.linspace(-np.pi/6, np.pi/6, 6*delta_t)
    y_vals = X_Y_Spline(x_vals)
    ax.plot(x_vals, y_vals, linewidth = 4, color = 'firebrick')
    ax.set_xlim([-np.pi/6, np.pi/6])
    ax.set_xticks(np.linspace(-np.pi/6, np.pi/6, 7))
    ax.tick_params(axis='both', which='major', labelsize=12, labelcolor = "#adadad")
    ax.set_yticks([])
    ax.grid(False)
    for i, spine in enumerate(ax.spines.values()):
        spine.set_linewidth(0)
    if save:
        fig.savefig("grid_simulation/Results/orientation_plot", dpi = 500, bbox_inches = 'tight', transparent = True)      

if ('yearbook_plot' in plots):
    base_path = "grid_simulation/Results/data/simspam/regular5"
    gscores = gscores = np.load('grid_simulation/Results/simspam3.npz')['gscores'][2]
    ng = 13
    pxs = 48
    rel_times = np.array([0,5,15,40,95])
    all_times = np.arange(20)
    rel_n = len(rel_times)
    all_n = len(all_times)
    # picks = np.random.choice(ng, nrows, False)
    ind = 6
    hist = np.empty((all_n, ng, pxs, pxs))

    for i in all_times:
        hist[i] = utils.getPopulationSpikePlot(f"{base_path}/{i*5}min_Spikes.npz", 13, 48)
    fig = plt.figure()
    plt.subplot(2,1,1)
    X_Y_Spline = make_interp_spline(all_times * 5, np.mean(gscores, (0,2)))
    x_vals = np.linspace(0,95, 40)
    y_vals = X_Y_Spline(x_vals)
    plt.plot(x_vals, y_vals, linewidth = 4, color = 'darkslateblue')
    plt.ylabel("Gridness", size = 12)
    plt.xlabel("Time (minutes)             ", size = 12, loc = 'right')
    y_min = np.min(y_vals)
    plt.ylim([y_min, None])
    plt.xlim([None, 95])
    ax1 = plt.gca()

    for z, rel_time in enumerate(rel_times):
        plt.subplot(2, rel_n, z + rel_n + 1)
        plt.imshow(hist[rel_time//5, ind],  interpolation='none', origin = 'lower')
        plt.axis('off')
        ax2 = plt.gca()
        con = ConnectionPatch(xyA = (rel_time, y_min), xyB = (pxs/2, pxs-1), coordsA="data", coordsB="data", axesA=ax1, axesB=ax2, color="black", arrowstyle = '->', shrinkA = 2, shrinkB = 10)
        ax2.add_artist(con)
        # ax[z].set_title(f"{rel_time} minutes", fontsize = 10)
    
    if save:
        fig.savefig("grid_simulation/Results/yearbook_plot", dpi = 500, bbox_inches = 'tight')      

plt.show()
