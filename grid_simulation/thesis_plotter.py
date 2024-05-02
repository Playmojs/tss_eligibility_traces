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

plots = ['distribution_plot']
save = True
models = np.array(['simspam', 'multi-grid', 'noise_sims2', 'noise_sims', 'noise_sims3', 'no_delay', 'GJ_model'], dtype= object)
concatenate = [False, False, True, True, False, False, False]
order = np.array([1, 2, 0, 4, 3, 5, 6, 7, 8, 9], dtype = int)
simuls_per_model=[3, 2, 1, 1, 1, 1, 1]
simul_names = np.array(['Standard', 'Blue noise inputs', 'White noise inputs', '23 cells', '37 cells', '1 ms noise', '2 ms noise', '4 ms noise', 'No delay', 'MC model'], dtype = object)
gscore_file_app = ["4", "", "3", "3", "","1", ""]

cmap = plt.get_cmap('nipy_spectral')
col_vals = np.array([cmap(0.9), cmap(0.2), cmap(0.32), cmap(0.45), cmap(0.55),  cmap(0.825), cmap(0.79), cmap(0.75), cmap(0.13), cmap(0.05)], dtype = object)
darkened_colors = np.array([(max(0, c[0] - 0.35), max(0, c[1] - 0.35), max(0, c[2] - 0.35)) for c in col_vals], dtype = object)

if("distribution_plot" in plots):
    inputs = ["grid_simulation/Results/data/24dend2/regular0.npz", "grid_simulation/Results/data/24dend2/noisy_blue0.npz", "grid_simulation/Results/data/24dend2/noisy_white0.npz"]
    ax_title = ["Regular", "Blue noise", "White noise"]
    fig, axs = plt.subplots(1, 3)

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
    points_x = np.array([1.0, 2.0, 2.5, 2.7, 3.4, 4.5, 5.2, 5.8, 5.9, 6.7, 7.0, 8.1, 8.3, 8.7, 9.3, 9.4, 9.9])
    points_y = np.array([5.1, 4.3, 5.6, 4.4, 6.1, 4.5, 6.0, 5.2, 4.1, 7.0, 3.5, 2.1, 8.0, 5.3, 8.9, 1.6, 4.7])

    color = np.full(len(points_x), 'navy', dtype = object)
    color[points_x < 5] = 'brown'

    fig, ax = plt.subplots()
    ax.scatter(points_x, points_y, s = 100, c = color)
    ax.vlines(5, 0, 10, linewidth = 4, color = 'forestgreen')
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_ylabel('Input Cell', fontsize = 20)
    ax.set_xlabel('Time', fontsize = 20)
    [x.set_linewidth(2) for x in ax.spines.values()]
    if save:
        fig.savefig("grid_simulation/Documents/Figures/input_STDP_plot", dpi = 500, bbox_inches = 'tight')

if("input_inhibit_plot" in plots):
    points_x = np.array([1.0, 2.0, 2.5, 2.7, 3.4, 4.5, 5.2, 5.8, 5.9, 6.7, 7.0, 8.1, 8.3, 8.7, 9.3, 9.4, 9.9])
    points_y = np.array([5.1, 4.3, 5.6, 4.4, 6.1, 4.5, 6.0, 5.2, 4.1, 7.0, 3.5, 2.1, 8.0, 5.3, 8.9, 1.6, 4.7])
  
    fig, ax = plt.subplots()
    ax.scatter(points_x, points_y, s = 100, color = 'black')
    ax.vlines(np.array([5, 6]), 0, 10, linewidth = 4, color = 'forestgreen')
    ax.vlines(6.5, 0, 10, linewidth = 4, color = 'darkmagenta')
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

def make_orientation_plot(file: str):
    orientations = np.load(file)["orientations"]
    thetas = np.arange(60)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    delta_t = 30
    thetas = np.linspace(-np.pi/6, np.pi/6, delta_t)
    histcount, _ = np.histogram((np.ndarray.flatten(orientations) + 30)%60, delta_t)
    X_Y_Spline = make_interp_spline(thetas, histcount)
    x_vals = np.linspace(-np.pi/6, np.pi/6, 6*delta_t)
    y_vals = X_Y_Spline(x_vals)
    ax.plot(x_vals, y_vals, linewidth = 2, color = 'black')
    ax.set_xlim([-np.pi/6, np.pi/6])
    ax.set_xticks(np.linspace(-np.pi/6, np.pi/6, 7))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_yticks([])
    ax.grid(False)
    for i, spine in enumerate(ax.spines.values()):
        spine.set_linewidth(0)
    return fig, ax
        
if ('model_comparison_orientation' in plots):
    gscore_file_app = ["_reg", "_bn", "_wn", "_23", "_37", "1", "3", "2", "", ""]
    output_app = ["_regular", "_blue_noise", "_white_noise","_23grid", "_37grid", "_no_delay", "_1ms_noise", "_2ms_noise", "_4ms_noise", "_GJ"]
    iter = 0
    for i, model in enumerate(models):
        for j in range(simuls_per_model[i]):
            output =  f'grid_simulation/Documents/Figures/model_comparison/model_comparison_orientation{output_app[iter]}'
            fig, ax = make_orientation_plot(f'grid_simulation/Results/analysis/{model}/orientations{gscore_file_app[iter]}.npz')
            ax.set_title(simul_names[iter], size = 25, pad = 50)
            if iter == 0:
                ax.set_ylabel("Orientation", size =30)

            plt.gcf().set_size_inches(4, 4)
            if save:
                fig.savefig(output, dpi = 500, bbox_inches = 'tight', transparent = True)  
            iter += 1

def make_phase_plot(file: str, color):
    with np.load(file) as data:
        phases = data["phases"]
        sigma = data["sigma"]
    
    pxs = 48
    base_length = 3*sigma*pxs
    x0, x1 = 0, base_length
    y0, y1 = 0, base_length*(np.sqrt(3)/2)

    fig, ax = plt.subplots()

    ax.scatter(phases[0], phases[1], color = color)
    
    ax.plot([x0, x1, x1 + 0.5 * y1], [y0, y0, y1], c = "black", linewidth = 3)
    ax.plot([x0, 0.5 * y1, x1 + 0.5 * y1], [y0, y1, y1], c = "black", linewidth = 3)
    ax.set_aspect('equal')
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis='both', which='major', labelsize=0)

    ax.set_xlim([0, 4.35*sigma*pxs])
    ax.set_ylim([-0.2, 3*sigma*pxs])
    [x.set_linewidth(0) for x in ax.spines.values()]

    return fig, ax
    

if('model_comparison_phase' in plots):
    phase_file_app = ["_reg", "_bn", "_wn", "_23", "_37", "", "", "", "", ""]
    output_app = ["_regular", "_blue_noise", "_white_noise", "_23grid", "_37grid", "_no_delay", "_1ms_noise", "_2ms_noise", "_4ms_noise", "_GJ"]
    iter = 0
    for i, model in enumerate(models):
        for j in range(simuls_per_model[i]):
            output = f'grid_simulation/Documents/Figures/model_comparison/model_comparison_phase{output_app[iter]}'
            fig, ax = make_phase_plot(f'grid_simulation/Results/analysis/{model}/phase{phase_file_app[iter]}.npz', color = col_vals[iter])
            if iter == 0:
                ax.set_ylabel("Phase", size = 30)

            plt.gcf().set_size_inches(4, 4)

            if save:
                fig.savefig(output, dpi = 500, bbox_inches = 'tight', transparent = True)      
            iter += 1

def make_distribution_plot(gscores, color_idx):
    fig, ax = plt.subplots()
    ax.hist(np.ndarray.flatten(gscores[:,-1, :]), 8, histtype = 'stepfilled', color = col_vals[color_idx], edgecolor = darkened_colors[color_idx], linewidth = 5, range= (-1.2, 1.6))
    ax.tick_params(axis = 'both', which = 'major', labelsize = 12)

    return fig, ax

if('model_comparison_distribution' in plots):
    output_app = ["regular", "_blue_noise", "_white_noise", "_23grid", "_37grid", "_1ms_noise", "_2ms_noise", "_4ms_noise","_no_delay", "_GJ"]
    gscores_container = np.empty(0)
    iter = 0

    for model, nsimuls, concat, app in zip(models, simuls_per_model, concatenate, gscore_file_app):
        gscores =  np.load('grid_simulation/Results/analysis/' + model + '/gscores' + app + '.npz')['gscores']
        if concat:
            s0, s1, s2, s3 = gscores.shape
            gscores = np.reshape(gscores, (1, s0 * s1, s2, s3))
        for group in gscores:
            fig, ax = make_distribution_plot(group, order[iter])
            ax.set_xticks([-1, -0.5, 0, 0.5, 1.0, 1.5])
            ax.set_ylabel(simul_names[order[iter]], size = 15)
            plt.gcf().set_size_inches(4, 2)
            if save:               
               fig.savefig(f'grid_simulation/Documents/Figures/model_comparison/distribution{output_app[order[iter]]}', dpi = 500, bbox_inches = 'tight', transparent = True)      
            iter += 1

            
if ('model_comparison_gscore' in plots):
    n_col = sum(simuls_per_model)
    gscores_container = np.empty(0)
    ind_container = np.empty(0, dtype = int)
    means = np.empty(n_col)
    iter = 0

    for model, nsimuls, concat, app in zip(models, simuls_per_model, concatenate, gscore_file_app):
        gscores =  np.load('grid_simulation/Results/analysis/' + model + '/gscores' + app + '.npz')['gscores']
        if concat:
            s0, s1, s2, s3 = gscores.shape
            gscores = np.reshape(gscores, (1, s0 * s1, s2, s3))
        for group in gscores:
            data = np.nanmean(group[:, -1, :], 1)
            data = data[~np.isnan(data)]
            gscores_container = np.append(gscores_container, data)
            means[order[iter]] = np.mean(data)
            ind_container = np.append(ind_container, np.full(data.size, int(order[iter])))
            iter += 1
        
    scatter_ind = ind_container + (np.random.random(len(ind_container)) - 0.5) * 0.2
    fig, ax = plt.subplots()

    ax.scatter(scatter_ind, gscores_container, c = col_vals[ind_container], s = 7)
    ax.hlines(means, xmin = np.arange(n_col) - 0.3, xmax = np.arange(n_col) + 0.3, linewidth = 3, colors = darkened_colors)
    ax.hlines(0, xmin = -1, xmax = 11, colors = 'black')
    ax.set_xlim([-0.7, 9.7])
    ax.set_ylabel("Mean gridness score", size = 15)
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(simul_names, rotation = 30, size = 12, ha = 'right')

    plt.gcf().set_size_inches(12, 4)
    if save:
        fig.savefig("grid_simulation/Documents/Figures/model_comparison/model_comparison_gscores", dpi = 500, bbox_inches = 'tight', transparent = True)      

if ('model_comparison_sigma' in plots):
    n_cols = sum(simuls_per_model)
    sigma_container = np.empty(0)
    ind_container = np.empty(0, dtype = int)
    means = np.empty(n_cols)
    iter = 0
    
    for model, ngroup, concat in zip(models, simuls_per_model, concatenate):
        gscores =  np.load('grid_simulation/Results/analysis/' + model + '/sigmas.npz')['sigma_gscores']
        sigmas =  np.load('grid_simulation/Results/analysis/' + model + '/sigmas.npz')['sigmas']
        if concat:
            s0, s1, s2, s3 = gscores.shape
            gscores = np.reshape(gscores, (1, s0 * s1, s2, s3))
        for group in gscores:
            pos_gscores = np.nanmax(group, axis = 1) > 0
            filter = np.argmax(group, axis = 1)[pos_gscores]
            best_sigma = sigmas[filter]
            best_sigma = best_sigma[~np.isnan(best_sigma)]
            sigma_container = np.append(sigma_container, best_sigma)
            means[order[iter]] = np.median(best_sigma)
            ind_container = np.append(ind_container, np.full(len(best_sigma), order[iter]))
            iter += 1

    scatter_ind = ind_container + (np.random.random(len(ind_container)) - 0.5) * 0.2
    scatter_sig = sigma_container + (np.random.random(len(ind_container)) - 0.5) * 0.01
    
    fig, ax = plt.subplots()

    ax.set_ylim([21, 70])
    ax.set_xlim([-0.7, 9.7])
    ax.scatter(scatter_ind, scatter_sig * 300, c = col_vals[ind_container], s = 5, linewidth = 0.1)
    ax.hlines(means * 300, xmin = np.arange(n_cols) - 0.3, xmax = np.arange(n_cols) + 0.3, linewidth = 3, colors = darkened_colors)
    ax.set_ylabel("Grid Spacing (cm)", size = 15)
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(simul_names, rotation = 30, size = 12, ha = 'right')

    plt.gcf().set_size_inches(12, 4)
    if save:
        fig.savefig("grid_simulation/Documents/Figures/model_comparison/model_comparison_sigmas", dpi = 500, bbox_inches = 'tight', transparent = True)      
    

if ('model_comparison_temporal_stability' in plots):
    n_cols = sum(simuls_per_model)
    values_container = np.empty((n_cols, 19))
    shuffle_container = np.empty((n_cols, 19))
    x_vals = np.arange(n_cols)
    
    iter = 0
    
    for model, ngroup, concat in zip(models, simuls_per_model, concatenate):
        with np.load('grid_simulation/Results/analysis/' + model + '/temporal_stability.npz') as data:
            pairwise_var =  data['pairwise_var']
            pairwise_shuffled_var = data["pairwise_shuffled_var"]
        if concat:
            s0, s1, s2, s3 = pairwise_var.shape
            pairwise_var = np.reshape(pairwise_var, (1, s0 * s1, s2, s3))
            pairwise_shuffled_var = np.reshape(pairwise_shuffled_var, (1, s0 * s1, s2, s3))
        for temp, shuffle in zip(pairwise_var, pairwise_shuffled_var):
            values_container[order[iter]] = np.mean(temp, axis = (0, 2))
            shuffle_container[order[iter]] = np.mean(shuffle, axis = (0, 2))
            iter += 1
    
    fig, ax = plt.subplots()
    bar_values = np.mean(values_container[:, 9:] / shuffle_container[:, 9:], axis = 1)

    ax.bar(x_vals, bar_values, width = 0.4, color = col_vals, edgecolor = darkened_colors, linewidth = 2)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(simul_names, rotation = 30, size = 12, ha = 'right')
    ax.set_ylim([0, 1.2])
    ax.set_xlim([-0.5, 9.5])
    ax.set_ylabel("Temporal variance", size = 15)
    ax.hlines(1, -1, 10, color = 'black', linestyle = 'dashed')
    ax.text(1, 1.03, 'Shuffled variance', size = 12)
    plt.gcf().set_size_inches(12, 4)

    if save:
        fig.savefig("grid_simulation/Documents/Figures/model_comparison/model_comparison_temporal_stability", dpi = 500, bbox_inches = 'tight', transparent = True)

def gscore_line_plot(scores, vars, times, col_inds, output_file: str, x_label = True, y_label = True, y_lim = [None, None]):
    fig, ax = plt.subplots()
    for col_ind, score, var in zip(col_inds, scores, vars):
        ax.plot(times, score, linewidth = 2, c = col_vals[col_ind], zorder = 2)
        ax.fill_between(times, score + 2*np.sqrt(var), score - 2*np.sqrt(var), alpha=0.4, interpolate=True, label = '_nolegend_', color = darkened_colors[col_ind], zorder = 1)
    ax.hlines(0, 0, 95, colors = 'Black', linestyles = 'dashed')

    ax.set_ylim(y_lim)
    if x_label:
        ax.set_xlabel("Time (minutes)", size = 15)
    else:
        ax.set_xlabel(" ", size = 15)  
    if y_label:
        ax.set_ylabel("Gridness Score", size = 18)
    else:
        ax.set_ylabel(" ", size = 15)
    ax.set_xlim([0, 95])
        
    ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax.legend(simul_names[col_inds], loc = 'upper left', fontsize = 12)

    plt.gcf().set_size_inches(6, 3)
    if save:
        fig.savefig(output_file, dpi = 500, bbox_inches = 'tight', transparent = True)

if('model_comparison_line_plots' in plots):
    model_groups = [[0, 9], [1, 2], [4, 3, 8], [5, 6, 7]]
    output_base = "grid_simulation/Documents/Figures/model_comparison/line_plot"
    times = np.linspace(0, 100, 20, endpoint = False)
    ngs = [np.array([13, 13, 13]), np.array([37, 23]), np.array([13]), np.array([13]), np.array([13]), np.array([13]), np.array([13])]
    gscores_means_container = []
    gscores_vars_container = []
    for i, model in enumerate(models):
        gscore = np.load(f"grid_simulation/Results/analysis/{model}/gscores{gscore_file_app[i]}.npz")["gscores"]
        gscores_means_container.append(np.nanmean(gscore, axis = (1,3)))
        gscores_vars_container.append(np.nanvar(gscore, axis = (1, 3)) / (ngs[i]*30)[:,np.newaxis])

    gscore_line_plot(np.array([gscores_means_container[0][2], gscores_means_container[-1][0]]), np.array([gscores_vars_container[0][2], gscores_vars_container[-1][0]]), times, model_groups[0], f'{output_base}{0}', True, True, [-0.2, 0.8])
    gscore_line_plot(gscores_means_container[0][0:2], gscores_vars_container[0][0:2], times, model_groups[1], f'{output_base}{1}', False, False, [-0.2, 0.8])
    gscore_line_plot(np.array([gscores_means_container[1][0], gscores_means_container[1][1], gscores_means_container[5][0]]), np.array([gscores_vars_container[1][0], gscores_vars_container[1][1], gscores_vars_container[2][0]]), times, model_groups[2], f'{output_base}{2}', False, False, [-0.2, 0.8])
    gscore_line_plot(np.array([gscores_means_container[2][0], gscores_means_container[3][0], gscores_means_container[4][0]]), np.array([gscores_vars_container[3][0], gscores_vars_container[4][0], gscores_vars_container[5][0]]), times, model_groups[3], f'{output_base}{3}', False , False, [-0.2, 0.8])

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



if('reg_distribution' in plots):
    path = "grid_simulation/Results/analysis/"
    gscore_ind = [2, 0, 0, 0]
    col_inds = np.array([0, 6, 7, 8])

    gscores = np.load(f"{path}simspam/gscores4.npz")["gscores"][2]

    simulation = "grid_simulation/Results/data/simspam/regular8"
    ng = 13
    pxs = 48
    rel_times = np.array([95])
    rel_n = len(rel_times)
    picks = np.array([8, 10, 11, 2, 1, 0])
    n_cols = len(picks)
    hist = np.empty((rel_n, 13, pxs, pxs))

    for i, time in enumerate(rel_times[::-1]):
        hist[i] = utils.getPopulationSpikePlot(f"{simulation}/{time}min_Spikes.npz", 13, 48)
    


    hist_y_size = 2
    fig_dist, ax_dist = plt.subplots(rel_n + hist_y_size, n_cols)
    
    # A bunch of trickery is necessary to make this simple plot...
    
    fig_dist.subplots_adjust(hspace = 0.7)
    gs = ax_dist[0,0].get_gridspec()
    for axs in ax_dist[0:hist_y_size]:
        for ax in axs:
            ax.remove()

    ax_hist = fig_dist.add_subplot(gs[0:hist_y_size, :])
    ax_hist.hist(np.ndarray.flatten(gscores[:,-1, :]), 8, histtype = 'stepfilled', color = col_vals[0], edgecolor = darkened_colors[0], linewidth = 5, range= (-1.2, 1.6))
    ax_hist.set_xlabel("Gridness Score", loc = "left", size = 12)
    ax_hist.set_ylabel("Cell Count", size = 15)
    ax_hist.tick_params(axis = 'both', which = 'major', labelsize = 12)

    for r in range(n_cols):
        ind = picks[r]
        for z in range(rel_n):
            ax_dist[z + hist_y_size,r].imshow(hist[z, ind],  interpolation='none', origin = 'lower')
            ax_dist[z + hist_y_size,r].axis('off')
    

    sim_gscores = gscores[8, -1, picks]
    shrinks = [15, 7, 7, 4.5, 6, 6]
    for col, score in enumerate(sim_gscores):
        shrink = shrinks[col]
        con = ConnectionPatch(xyA = (score, 0), xyB = (48/2, 48-1), coordsA="data", coordsB="data", axesA=ax_hist, axesB=ax_dist[hist_y_size, col], color="black", arrowstyle = '->', shrinkA = shrink, shrinkB = shrink, linewidth = 2)
        ax_dist[hist_y_size, col].add_artist(con)

    plt.gcf().set_size_inches(12, 6)

    if save:
        fig_dist.savefig("grid_simulation/Documents/Figures/reg/distribution", dpi = 500, bbox_inches = 'tight') 

if('reg_spikeplot' in plots):
    simulation = "grid_simulation/Results/data/simspam/regular6"
    ng = 13
    pxs = 48
    rel_times = np.array([0, 5, 10, 20, 40, 70, 95])
    rel_n = len(rel_times)
    picks = np.array([8, 10, 11, 2, 0, 7, 6])
    n_cols = len(picks)
    hist = np.empty((rel_n, 13, pxs, pxs))
    
    fig, ax = plt.subplots(rel_n, n_cols)
    for i, time in enumerate(rel_times):
        hist[i] = utils.getPopulationSpikePlot(f"{simulation}/{time}min_Spikes.npz", 13, 48)
        ax[i, 0].set_ylabel(f'{time} minutes', size = 12)

    for r in range(n_cols):
        ind = picks[r]
        for z in range(rel_n):
            ax[z, r].imshow(hist[z, ind],  interpolation='none', origin = 'lower')
            ax[z, r].set_xticks([])
            ax[z, r].set_yticks([])

    plt.gcf().set_size_inches(12, 12)

    if save:
        fig.savefig("grid_simulation/Documents/Figures/reg/spike", dpi = 500, bbox_inches = 'tight') 

if('spacing_spikeplots' in plots):
    model_inds = np.array([0, 2, 3, 4])
    simul_number = np.array([3, 8, 8, 8])
    cell_inds = [np.array([0, 4]), np.array([4, 5, 8, 10]), np.array([2, 11, 0, 4]), np.array([3, 4, 6, 12])]
    titles = ["0 ms noise", "1 ms noise", "2 ms noise", "4 ms noise"]
    for model, simul, cells, title in zip(models[model_inds], simul_number, cell_inds, titles):
        time_app = "95min_spikes" if model != "noise_sims2" else "95.0min_spikes"
        hist = utils.getPopulationSpikePlot(f"grid_simulation/Results/data/{model}/regular{simul}/{time_app}.npz", 13, 48, True)
        ncols = len(cells)//2
        fig, ax = plt.subplots(nrows = 2, ncols = ncols, squeeze = False)
        for i, cell in enumerate(cells):
            i1 = i % 2
            i2 = i // 2
            ax[i1, i2].imshow(hist[cell], interpolation = 'none', origin = 'lower')
            ax[i1, i2].axis('off')
        fig.suptitle(title, size = 15)
        fig_width = ncols * 1.5 + 1
        plt.gcf().set_size_inches(fig_width, 4)
        if save:
            fig.savefig(f"grid_simulation/Documents/Figures/model_comparison/spacing_spikeplot_{model}", dpi = 500, bbox_inches = 'tight')

if('sum_spikeplots' in plots):
    path = "grid_simulation/Results/analysis/"
    pxs = 48 

    true_inds = [4, 9, 8]
    model_inds = [1, 6, 5]
    simuls = [3, 15, 23]
    ngs = [37, 13, 13]
    all_cells = [np.array([0, 3]), np.array([12, 3]), np.array([1, 6])]

    for ind, model, simul, ng, cells in zip(true_inds, models[model_inds], simuls, ngs, all_cells):
        hist = np.zeros((20, ng, pxs, pxs))
        for i in range(20):
            hist[i]= utils.getPopulationSpikePlot(f"grid_simulation/Results/data/{model}/regular{simul}/{i*5}min_spikes.npz", ng, pxs, True)
        summed_hist = np.sum(hist, axis = 0)
        
        ncols = 2
        fig, ax = plt.subplots(nrows = 2, ncols = 2)
        for i, cell in enumerate(cells):
            ax[0, i].imshow(hist[-1, cell], interpolation = 'none', origin = 'lower')
            ax[1, i].imshow(summed_hist[cell], interpolation = 'none', origin = 'lower')
            ax[0, i].set_xticks([])
            ax[0, i].set_yticks([])
            ax[1, i].set_xticks([])
            ax[1, i].set_yticks([])

        if ind == true_inds[0]:
            ax[0, 0].set_ylabel("Activity at \n 95 minutes", size = 15, labelpad = 10)
            ax[1, 0].set_ylabel("Activity \n summed", size = 15, labelpad = 10)
        fig.suptitle(simul_names[ind], size = 15)
        plt.gcf().set_size_inches(4, 4)
        if save:
            fig.savefig(f"grid_simulation/Documents/Figures/model_comparison/sum_spikeplot_{model}", dpi = 500, bbox_inches = 'tight')


if('BVC_spikeplot' in plots):
    simul = 'orthoregular18'
    simulation = f"grid_simulation/Results/data/BVC_tests/{simul}"
    ng = 13
    pxs = 48
    rel_times = np.array([0, 5, 20, 95])
    rel_n = len(rel_times)
    picks = np.array([8, 10, 11, 2, 3, 7, 6])
    n_cols = len(picks)
    hist = np.empty((rel_n, 13, pxs, pxs))

    fig, ax = plt.subplots(rel_n, n_cols)
    for i, time in enumerate(rel_times):
        hist[i] = utils.getPopulationSpikePlot(f"{simulation}/{time}min_Spikes.npz", 13, 48, True, True)
        ax[i, 0].set_ylabel(f'{time} minutes', size = 12)

    autocorr = utils.autoCorr(hist)
    gscores, _ = utils.gridnessScore(autocorr, 48, 0.127) # Shape(n_groups, n_simuls, n_times, ng)

    for r in range(n_cols):
        ind = picks[r]
        ax[-1, r].set_xlabel(f'{gscores[-1, ind]:.2f}', size = 12)
   
        for z in range(rel_n):
            ax[z, r].imshow(hist[z, ind],  interpolation='none', origin = 'lower')
            ax[z, r].set_xticks([])
            ax[z, r].set_yticks([])

    plt.gcf().set_size_inches(12, 7)

    if save:
        fig.savefig("grid_simulation/Documents/Figures/BVC_spikeplot", dpi = 500, bbox_inches = 'tight') 
    

    conductances = np.load(f'grid_simulation/Results/data/BVC_tests/{simul}.npz')['weights'][6*95]
    conductances = np.reshape(conductances, (13, 24, 24))[picks]

    fig, ax = plt.subplots(1, n_cols, squeeze = False)

    for r in range(n_cols):
        ax[0, r].imshow(conductances[r], interpolation = 'none', origin = 'lower', cmap = 'plasma')
        ax[0, r].set_yticks([])
        ax[0, r].set_xticks([])
    if save:
        fig.savefig("grid_simulation/Documents/Figures/BVC_spikeplot_weights", dpi = 500, bbox_inches = 'tight') 

if ('BVC_aux' in plots):
    with np.load('grid_simulation/Results/misc/BVC_test.npz') as data:
        positions = data['positions']
        input_indices = data['bvc_inds']
        input_spikes = data['bvc_times']
        dendrite_times = data['dendrite_times']
        dendrite_acts = data['dendrite_acts']

    BVC_picks = np.array([4, 43, 17, 34], dtype = int)
    dendrite_picks = np.array([6, 1, 3, 4], dtype = int)

    bvc_fig, bvc_ax = plt.subplots(2, 2)
    den_fig1, den_ax1 = plt.subplots(2, 2)
    den_fig2, den_ax2 = plt.subplots(2, 2)

    pxs = 48
    for i, (BVC_ind, dendrite_ind) in enumerate(zip(BVC_picks, dendrite_picks)):
        i1 = i // 2
        i2 = i % 2

        spike_times = input_spikes[input_indices == BVC_ind]
        spike_indices = np.floor(spike_times/100)
        spike_positions = positions[np.ndarray.astype(spike_indices, int)]
        BVC_hist, _, __ = np.histogram2d(spike_positions[:,1], spike_positions[:,0], pxs, [[-0.5,0.5],[-0.5,0.5]])
        BVC_hist = ndimage.gaussian_filter(BVC_hist, 2)
        bvc_ax[i1, i2].imshow(BVC_hist, interpolation = 'none', origin = 'lower')
        bvc_ax[i1, i2].set_xticks([])
        bvc_ax[i1, i2].set_yticks([])
        bvc_fig.suptitle('BVC activity')

        dendrite_act = dendrite_acts[dendrite_ind]

        dendrite_act1 = np.sum(np.reshape(dendrite_act, (len(dendrite_act)// 1000, 1000))[:, 0:100], axis = 1)
        spike_positions = positions[dendrite_act1 > 5]
        dendrite_hist, _, __ = np.histogram2d(spike_positions[:,1], spike_positions[:,0], pxs, [[-0.5,0.5],[-0.5,0.5]])
        dendrite_hist = ndimage.gaussian_filter(dendrite_hist, 2)
        den_ax1[i1, i2].imshow(dendrite_hist, interpolation = 'none', origin = 'lower')
        den_ax1[i1, i2].set_xticks([])
        den_ax1[i1, i2].set_yticks([])
        den_fig1.suptitle('Dendrite activity \n 10 ms')

        dendrite_act2 = np.sum(np.reshape(dendrite_act, (len(dendrite_act)// 1000, 1000))[:, 0:220], axis = 1)
        spike_positions = positions[dendrite_act2 > 40]
        dendrite_hist, _, __ = np.histogram2d(spike_positions[:,1], spike_positions[:,0], pxs, [[-0.5,0.5],[-0.5,0.5]])
        dendrite_hist = ndimage.gaussian_filter(dendrite_hist, 2)
        den_ax2[i1, i2].imshow(dendrite_hist, interpolation = 'none', origin = 'lower')
        den_ax2[i1, i2].set_xticks([])
        den_ax2[i1, i2].set_yticks([])
        den_fig2.suptitle('Dendrite activity \n 20 ms')

    if save:
        for fig, title in zip([bvc_fig, den_fig1, den_fig2], ['bvc_act', 'den_act1', 'den_act2']):
            fig.set_size_inches(4, 4)
            fig.savefig(f"grid_simulation/Documents/Figures/{title}", dpi = 500, bbox_inches = 'tight') 




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
