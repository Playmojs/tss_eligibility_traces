import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib.patches import ConnectionPatch
import scipy.ndimage as ndimage
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

plots = ['phase_dist']
save = False

if("distribution_plot" in plots):
    inputs = ["grid_simulation/Results/data/24dend2/regular0.npz", "grid_simulation/Results/data/24dend2/noisy_blue0.npz", "grid_simulation/Results/data/24dend2/noisy_white0.npz"]
    ax_title = ["Regular", "Blue noise", "White noise"]
    fig, axs = plt.subplots(1, 3)
    cmap = plt.get_cmap('nipy_spectral')
    col_vals = [cmap(0.9), cmap(0.2), cmap(0.47)]

    for input, ax, title, color in zip(inputs, axs, ax_title, col_vals):
        input_positions = np.load(input, allow_pickle=True)['input_pos']
        ax.scatter(input_positions[:, 0], input_positions[:, 1], s = 3, color =color)
        ax.axis('square')
        ax.set_xlim([-0.025,1.025])
        ax.set_ylim([-0.025,1.025])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(title, size = 10)
        [x.set_linewidth(2) for x in ax.spines.values()]
    plt.gcf().set_size_inches(8, 2.67)
    if save:
        fig.savefig("grid_simulation/Documents/Figures/distribution_plot", dpi = 500, bbox_inches = 'tight')


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
    fig, ax = plt.subplots(nrows + 1, rel_n + 1)
    for r in range (nrows):
        ind = picks[r]
        for z, rel_time in enumerate(rel_times):
            ax[r,z].imshow(hist[rel_time//5, ind],  interpolation='none', origin = 'lower')
            ax[r,z].axis('off')
            ax[0,z].set_title(f"{rel_time} minutes", fontsize = 10)
            ax[nrows, z].axis('off')
        ax[r, rel_n].imshow(np.sum(hist[:, ind], 0), interpolation='none', origin = 'lower')
        ax[r,rel_n].axis('off')
    ax[0, rel_n].set_title("Sum", fontsize = 10)
    ax[nrows, rel_n].imshow(np.sum(hist, (0,1)), interpolation = 'none', origin = 'lower')
    ax[nrows, rel_n].axis('off')
    

    if save:
        fig.savefig("grid_simulation/Documents/Figures/spike_temp_plot", dpi = 500, bbox_inches = 'tight')      

if('gscore_line_plot' in plots):
    gscores = np.load('grid_simulation/Results/analysis/simspam/gscores4.npz')['gscores']
    gscores = np.array([gscores[2], gscores[0], gscores[1]])
    shape = gscores.shape
    mean_gscores = np.nanmean(gscores, axis = (1,3))
    var_gscores = np.nanvar(gscores, axis = (1,3)) / (shape[1] * shape[3])
    times = np.linspace(0, 100, shape[2], endpoint=False)
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('nipy_spectral')
    col_vals = [cmap(0.9), cmap(0.2), cmap(0.47)]

    legends = ["Regular", "Blue Noise", "White Noise"]
    for color, line_mean, line_var in zip(col_vals, mean_gscores, var_gscores):
        ax.plot(times, line_mean, linewidth = 2, c = color, zorder = 2)
        ax.fill_between(times, line_mean + 2*np.sqrt(line_var), line_mean - 2*np.sqrt(line_var), alpha=0.4, interpolate=True, label = '_nolegend_', color = color, zorder = 1)
    ax.set_ylim([0, 0.65])
    ax.set_xlim([0,95])
    ax.legend(legends, fontsize = 10)
    ax.set_xlabel("Time (minutes)", size = 10)
    ax.set_ylabel("Gridness Score", size = 10)
    ax.set_xticks(times[::2])
    [x.set_linewidth(2) for x in ax.spines.values()]
    plt.gcf().set_size_inches(8, 4)
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

def make_orientation_plot(file: str, output: str, save: bool = False) -> None:
    orientations = np.load(file)["orientations"]
    thetas = np.arange(60)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    delta_t = 30
    thetas = np.linspace(-np.pi/6, np.pi/6, delta_t)
    histcount, _ = np.histogram((np.ndarray.flatten(orientations) + 30)%60, delta_t)
    X_Y_Spline = make_interp_spline(thetas, histcount)
    x_vals = np.linspace(-np.pi/6, np.pi/6, 6*delta_t)
    y_vals = X_Y_Spline(x_vals)
    ax.plot(x_vals, y_vals, linewidth = 4, color = 'black')
    ax.set_xlim([-np.pi/6, np.pi/6])
    ax.set_xticks(np.linspace(-np.pi/6, np.pi/6, 7))
    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.set_yticks([])
    ax.grid(False)
    for i, spine in enumerate(ax.spines.values()):
        spine.set_linewidth(0)
    if save:
        fig.savefig(output, dpi = 500, bbox_inches = 'tight', transparent = True)      
 

if ('model_comparison_orientation' in plots):
    models = ['simspam', 'multi-grid', 'no_delay', 'noise_sims2', 'noise_sims', 'noise_sims3', 'GJ_model']
    ngroups = [3, 2, 1, 1, 1, 1, 1]
    gscore_file_app = ["_reg", "_bn", "_wn", "_37", "_23", "1", "3", "2", "", ""]
    output_app = ["_regular", "_blue_noise", "_white_noise", "_37grid", "_23grid", "_no_delay", "_1ms_noise", "_2ms_noise", "_4ms_noise", "_GJ"]
    iter = 0
    for i, model in enumerate(models):
        for j in range(ngroups[i]):
            make_orientation_plot(f'grid_simulation/Results/analysis/{model}/orientations{gscore_file_app[iter]}.npz', f'grid_simulation/Documents/Figures/model_comparison/model_comparison_orientation{output_app[iter]}', save)
            iter += 1

def make_phase_plot(file: str, output: str, save: bool = False):
    with np.load(file) as data:
        phases = data["phases"]
        sigma = data["sigma"]
    
    pxs = 48
    base_length = 3*sigma*pxs
    x0, x1 = 0, base_length
    y0, y1 = 0, base_length*(np.sqrt(3)/2)

    fig, ax = plt.subplots()

    ax.scatter(phases[0], phases[1])
    
    ax.plot([x0, x1, x1 + 0.5 * y1], [y0, y0, y1], c = "black", linewidth = 3)
    ax.plot([x0, 0.5 * y1, x1 + 0.5 * y1], [y0, y1, y1], c = "black", linewidth = 3)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.set_xlim([0, 4.35*sigma*pxs])
    ax.set_ylim([-0.2, 5*sigma*pxs])
    [x.set_linewidth(0) for x in ax.spines.values()]

    if save:
        fig.savefig(output, dpi = 500, bbox_inches = 'tight', transparent = True)      


if('model_comparison_phase' in plots):
    models = ['simspam', 'multi-grid', 'no_delay', 'noise_sims2', 'noise_sims', 'noise_sims3', 'GJ_model']
    ngroups = [3, 2, 1, 1, 1, 1, 1]
    gscore_file_app = ["_reg", "_bn", "_wn", "_37", "_23", "", "", "", "", ""]
    output_app = ["_regular", "_blue_noise", "_white_noise", "_37grid", "_23grid", "_no_delay", "_1ms_noise", "_2ms_noise", "_4ms_noise", "_GJ"]
    iter = 0
    for i, model in enumerate(models):
        for j in range(ngroups[i]):
            make_phase_plot(f'grid_simulation/Results/analysis/{model}/phase{gscore_file_app[iter]}.npz', f'grid_simulation/Documents/Figures/model_comparison/model_comparison_phase{output_app[iter]}', save)
            iter += 1

if ('model_comparison_gscore' in plots):
    models = ['simspam', 'multi-grid', 'no_delay', 'noise_sims', 'noise_sims2', 'noise_sims3', 'GJ_model']
    order = np.array([1, 2, 0, 3, 4, 5, 7, 6, 8, 9], dtype = int)
    ngroups = [3, 2, 1, 1, 1, 1, 1]
    n_col = sum(ngroups)
    concatenate = [False, False, False, True, True, False, False]
    gscore_file_app = ["4", "", '1', "3", "3", "", ""]
    gscores_container = np.empty(0)
    ind_container = np.empty(0)
    means = np.empty(n_col)
    iter = 0

    for model, ngroup, concat, app in zip(models, ngroups, concatenate, gscore_file_app):
        gscores =  np.load('grid_simulation/Results/analysis/' + model + '/gscores' + app + '.npz')['gscores']
        if concat:
            s0, s1, s2, s3 = gscores.shape
            gscores = np.reshape(gscores, (1, s0 * s1, s2, s3))
        for group in gscores:
            data = np.nanmean(group[:, -1, :], 1)
            data = data[~np.isnan(data)]
            gscores_container = np.append(gscores_container, data)
            means[order[iter]] = np.mean(data)
            ind_container = np.append(ind_container, np.full(data.size, order[iter]))
            iter += 1
        
    scatter_ind = ind_container + (np.random.random(len(ind_container)) - 0.5) * 0.2
    fig, ax = plt.subplots()

    cmap = plt.get_cmap('tab10')
    darkened_colors = [cmap(i) for i in np.linspace(0, 1, n_col)]
    darkened_colors = [(max(0, c[0] - 0.35), max(0, c[1] - 0.35), max(0, c[2] - 0.35)) for c in darkened_colors]

    ax.scatter(scatter_ind, gscores_container, c = ind_container, cmap = 'tab10', s = 7)
    ax.hlines(means, xmin = np.arange(n_col) - 0.3, xmax = np.arange(n_col) + 0.3, linewidth = 3, colors = darkened_colors)
    ax.hlines(0, xmin = -1, xmax = 11, colors = 'black')
    ax.set_xlim([-0.7, 9.7])
    ax.set_ylabel("Mean gridness score", size = 10)
    ax.set_xticks([])

    plt.gcf().set_size_inches(8, 2.67)
    if save:
        fig.savefig("grid_simulation/Documents/Figures/model_comparison/model_comparison_gscores", dpi = 500, bbox_inches = 'tight', transparent = True)      

if ('model_comparison_sigma' in plots):
    models = ['simspam', 'multi-grid', 'no_delay', 'noise_sims', 'noise_sims2', 'noise_sims3', 'GJ_model']
    order = np.array([1, 2, 0, 3, 4, 5, 7, 6, 8, 9], dtype = int)
    ngroups = [3, 2, 1, 1, 1, 1, 1]
    n_cols = sum(ngroups)
    concatenate = [False, False, False, True, True, False, False]
    gscore_file_app = ["4", "", '1', "3", "3", "", ""]
    sigma_container = np.empty(0)
    ind_container = np.empty(0)
    means = np.empty(n_cols)
    iter = 0
    
    for model, ngroup, concat in zip(models, ngroups, concatenate):
        gscores =  np.load('grid_simulation/Results/analysis/' + model + '/sigmas.npz')['sigma_gscores']
        sigmas =  np.load('grid_simulation/Results/analysis/' + model + '/sigmas.npz')['sigmas']
        if concat:
            s0, s1, s2, s3 = gscores.shape
            gscores = np.reshape(gscores, (1, s0 * s1, s2, s3))
        for group in gscores:
            best_sigma = sigmas[np.argmax(group, axis = 1)]
            best_sigma = best_sigma[~np.isnan(best_sigma)]
            sigma_container = np.append(sigma_container, best_sigma)
            means[order[iter]] = np.mean(best_sigma)
            ind_container = np.append(ind_container, np.full(len(best_sigma), order[iter]))
            iter += 1

    scatter_ind = ind_container + (np.random.random(len(ind_container)) - 0.5) * 0.2
    scatter_sig = sigma_container + (np.random.random(len(ind_container)) - 0.5) * 0.01
    
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('tab10')
    darkened_colors = [cmap(i) for i in np.linspace(0, 1, n_cols)]
    darkened_colors = [(max(0, c[0] - 0.35), max(0, c[1] - 0.35), max(0, c[2] - 0.35)) for c in darkened_colors]

    ax.set_ylim([7, 20])
    ax.set_xlim([-0.7, 9.7])
    ax.scatter(scatter_ind, scatter_sig * 100, c = ind_container, s = 5, linewidth = 0.1, cmap = 'tab10')
    ax.hlines(means * 100, xmin = np.arange(n_cols) - 0.3, xmax = np.arange(n_cols) + 0.3, linewidth = 3, colors = darkened_colors)
    ax.set_ylabel("Grid Spacing (cm)", size = 10)
    ax.set_xticks([])

    plt.gcf().set_size_inches(8, 2.67)
    if save:
        fig.savefig("grid_simulation/Documents/Figures/model_comparison/model_comparison_sigmas", dpi = 500, bbox_inches = 'tight', transparent = True)      
    

if ('model_comparison_temporal_stability' in plots):
    models = ['simspam', 'multi-grid', 'no_delay', 'noise_sims', 'noise_sims2', 'noise_sims3', 'GJ_model']
    order = np.array([1, 2, 0, 3, 4, 5, 7, 6, 8, 9], dtype = int)
    ngroups = [3, 2, 1, 1, 1, 1, 1]
    n_cols = sum(ngroups)
    concatenate = [False, False, False, True, True, False, False]
    gscore_file_app = ["", "", '', "", "", "", ""]
    values_container = np.empty(n_cols)
    shuffle_container = np.empty(n_cols)
    x_vals = np.arange(n_cols)
    x_data = x_vals - 0.3
    x_shuffle = x_vals + 0.3
    
    iter = 0
    
    for model, ngroup, concat in zip(models, ngroups, concatenate):
        with np.load('grid_simulation/Results/analysis/' + model + '/temporal_stability.npz') as data:
            temporal_stability =  data['temporal_stability']
            shuffled_stability =  data['shuffled_stability']
        if concat:
            s0, s1, s2 = temporal_stability.shape
            temporal_stability = np.reshape(temporal_stability, (1, s0 * s1, s2))
            shuffled_stability = np.reshape(shuffled_stability, (1, s0 * s1, s2))
        for temp, shuffle in zip(temporal_stability, shuffled_stability):
            values_container[iter] = np.mean(temp, axis = (-2, -1))
            shuffle_container[iter] = np.mean(shuffle, axis = (-2, -1))
            iter += 1
    
    fig, ax = plt.subplots()
    # cmap = plt.get_cmap('tab10')
    # darkened_colors = [cmap(i) for i in np.linspace(0, 1, n_col)]
    # darkened_colors = [(max(0, c[0] - 0.35), max(0, c[1] - 0.35), max(0, c[2] - 0.35)) for c in darkened_colors]

    ax.bar(x_data, values_container)
    ax.bar(x_shuffle, shuffle_container)

    if save:
        fig.savefig("grid_simulation/Documents/Figures/model_comparison/model_comparison_temporal_stability", dpi = 500, bbox_inches = 'tight', transparent = True)


if('low-high_gscores' in plots):
    base_path = "grid_simulation/Results/data/simspam/regular8"
    ng = 13
    pxs = 48
    rel_times = np.array([0, 5, 10, 40, 95])
    rel_n = len(rel_times)
    picks = np.array([10, 8, 11, 2, 0, 1])
    n_cols = len(picks)
    hist = np.empty((rel_n, 13, pxs, pxs))

    for i, time in enumerate(rel_times):
        hist[i] = utils.getPopulationSpikePlot(f"{base_path}/{time}min_Spikes.npz", 13, 48)
    fig, ax = plt.subplots(rel_n, n_cols)
    for r in range(n_cols):
        ind = picks[r]
        for z in range(rel_n):
            ax[z,r].imshow(hist[z, ind],  interpolation='none', origin = 'lower')
            ax[z,r].axis('off')

    if save:
        fig.savefig("grid_simulation/Documents/Figures/low-high_gscores", dpi = 500, bbox_inches = 'tight') 

if('phase_dist' in plots):
    models = ['simspam', 'multi-grid']
    model_ind = [0, 1, 1]
    file_app = ['_reg', '_37', '_23']
    cell_counts = np.zeros((3, 30))
    phases = [None, None, None]
    fig, ax = plt.subplots()

    sigma = np.load(f"grid_simulation/Results/analysis/simspam/phase_reg.npz")["sigma"]
    pxs = 48
    base_length = 3*sigma*pxs
    x0, x1 = 0, base_length
    y0, y1 = 0, base_length*(np.sqrt(3)/2)

    for i in range(3):
        phases = np.load(f"grid_simulation/Results/analysis/{models[model_ind[i]]}/phase{file_app[i]}.npz")["phases"]

        noise = np.random.random(phases.shape) * 2 - 1
        noised_vals = phases + noise
        x_shear_val = noised_vals[0] - noised_vals[1]/2
        x_condition = np.logical_and(x_shear_val > 0, x_shear_val < x1)

        y_condition = np.logical_and(noised_vals[1] > 0, noised_vals[1] < y1)
        phases[0, x_condition] = noised_vals[0, x_condition]
        phases[1, y_condition] = noised_vals[1, y_condition]
        
        ax.scatter(phases[0], phases[1])
    
    ax.plot([x0, x1, x1 + 0.5 * y1], [y0, y0, y1], c = 'black')
    ax.plot([x0, 0.5 * y1, x1 + 0.5 * y1], [y0, y1, y1], c = 'black')
    ax.set_aspect('equal')

    ax.set_xlim([-0.2, 4.35*sigma*pxs])
    ax.set_ylim([-0.5, 5*sigma*pxs])
    ax.set_xticks([])
    ax.set_yticks([])
    [x.set_linewidth(0) for x in ax.spines.values()]
    if save:
        fig.savefig("grid_simulation/Documents/Figures/phase_dist", dpi = 500, bbox_inches = 'tight') 



if ('yearbook_plot' in plots):
    base_path = "grid_simulation/Results/data/simspam/regular5"
    gscores = np.load('grid_simulation/Results/simspam3.npz')['gscores'][2]
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
