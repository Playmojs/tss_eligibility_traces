import numpy as np
import matplotlib.pyplot as plt

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

plots = ["input_plot"]
save = False

if("distribution_plot" in plots):
    inputs = ["grid_simulation/Results/data/24dend2/regular0.npz", "grid_simulation/Results/data/24dend2/noisy_blue0.npz", "grid_simulation/Results/data/24dend2/noisy_white0.npz"]
    ax_title = ["Regular", "Blue noise", "White noise"]
    fig, axs = plt.subplots(1, 3)
    for input, ax, title in zip(inputs, axs, ax_title):
        input_positions = np.load(input, allow_pickle=True)['input_pos']
        ax.scatter(input_positions[:, 0], input_positions[:, 1], s = 3, color ='brown')
        ax.axis('square')
        ax.set_xlim([-0.025,1.025])
        ax.set_ylim([-0.025,1.025])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(title, size = 10)
        [x.set_linewidth(2) for x in ax.spines.values()]
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
    ax.set_xlabel('Time')
    ax.set_ylabel('Input Cell')
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
    ax.set_xlabel('Time')
    ax.set_ylabel('Input Cell')
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
    [x.set_linewidth(2) for x in ax.spines.values()]
    if save:
        fig.savefig("grid_simulation/Documents/Figures/mouse_plot", dpi = 500, bbox_inches = 'tight')        

plt.show()
