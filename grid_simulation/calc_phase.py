import numpy as np
import utils
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform  


def unsheared_modulus(points, base):
    # Define the shear transformation matrix to align the rhomboids with x and y axes
    shear_matrix = np.array([[1, 0.5], [0, 1]])
    inverse_shear = np.linalg.inv(shear_matrix)

    # Apply shear transformation to firing rates
    unsheared_points = inverse_shear @ points
    unsheared_size = np.dot(inverse_shear.T, np.array([base,base]))
    unsheared_size = np.array([base, base * np.sqrt(3) / 2])


    mod_unsheared = np.mod(unsheared_points, unsheared_size[:, np.newaxis])

    resheared_points = shear_matrix @ mod_unsheared
    return resheared_points

pxs = 48
appendix = 'min_Spikes.npz'
base_path = 'grid_simulation/Results/data/'
simulation = 'simspam'
sub_dirs = utils.getSortedEntries(base_path + simulation, 'directory', True)

n_simuls = 30
n_groups = 3
Ndendrites = 24
ng = np.array([13, 13, 13])
orientation_app = "_reg"
sigma = 0.108

hists = np.empty((n_groups, n_simuls, np.max(ng), pxs, pxs))

for j, sub_dir in enumerate(sub_dirs):
    j1 = j // n_simuls
    j2 = j % n_simuls
    hists[j1, j2, :ng[j1]] = utils.getPopulationSpikePlot(sub_dir + '/95' + appendix, ng[j1], pxs, True)

ind = 0
hists = hists[ind][np.newaxis]
gscore_mask = np.load(f"grid_simulation/Results/analysis/{simulation}/orientations{orientation_app}.npz")['mask']
orientations = np.load(f"grid_simulation/Results/analysis/{simulation}/orientations{orientation_app}.npz")['orientations']
norm_orientation = orientations%60
orientation_mask = np.logical_or(norm_orientation > 55, norm_orientation < 5).nonzero()

masked_hists = hists[gscore_mask[0],gscore_mask[1],gscore_mask[2]]
orig_sim = gscore_mask[1] + gscore_mask[0]*n_simuls

masked_hists = masked_hists[orientation_mask]
orig_sim = orig_sim[orientation_mask]

data = masked_hists[0]

data_max = ndimage.maximum_filter(masked_hists, size = 2*pxs*0.108, axes=(-2,-1))
maxima = np.nonzero(masked_hists==data_max)
min_sum_maxima = np.full((len(masked_hists), 2), np.inf)

for m, x, y in zip(maxima[0], maxima[1], maxima[2]):
    if x + y < np.sum(min_sum_maxima[m]):
        min_sum_maxima[m] = np.array([x, y])

grouped_hists = {}
grouped_min_maxima = {}
for i, hist in enumerate(masked_hists):
    sim = orig_sim[i]

    if sim not in grouped_hists:
        grouped_hists[sim] = np.empty((0, pxs, pxs))
        grouped_min_maxima[sim] = np.empty((0, 2), dtype = int)
    grouped_hists[sim] = np.append(grouped_hists[sim], hist[np.newaxis, ...], axis = 0)
    grouped_min_maxima[sim] = np.append(grouped_min_maxima[sim], np.ndarray.astype(min_sum_maxima[np.newaxis, i], int), axis = 0)

# corr = utils.normcorr2d(grouped_hists[0][1], grouped_hists[0][2])
# plt.imshow(corr, origin = 'lower')
# plt.show()

phase_array = np.empty((0, 2))
cell_count = np.zeros(n_simuls * 1, dtype = int)

for sim, cells in grouped_hists.items():
    cell_count[sim] = int(len(cells))
    if len(cells) == 1:
        continue
    min_maxima_ind = np.argmin(np.sum(grouped_min_maxima[sim], axis = 1))

    for i, cell_hist in enumerate(cells):
        if i == min_maxima_ind:
            continue
        corr_slice = utils.normcorr2d(cells[min_maxima_ind], cell_hist)[pxs:pxs+16, pxs:pxs-16:-1]
        phase_diff = np.unravel_index(np.argmax(corr_slice), np.shape(corr_slice))
        phase_array = np.append(phase_array, np.array([phase_diff]), axis=0)

base_length = 3*sigma*pxs
modulated_array = unsheared_modulus(phase_array.T, base_length)
x0, x1 = 0, base_length
y0, y1 = 0, base_length*(np.sqrt(3)/2)

np.savez("grid_simulation/Results/analysis/" + simulation + "/phase_reg",
        phases = modulated_array, \
        sigma = sigma, \
        orientation_ind = orientation_app, \
        cell_count = cell_count, \
        sim_ind = orig_sim)

fig, ax = plt.subplots(ncols= 2)
ax[0].scatter(modulated_array[0], modulated_array[1])

ax[0].plot([x0, x1, x1 + 0.5 * y1], [y0, y0, y1])
ax[0].plot([x0, 0.5 * y1, x1 + 0.5 * y1], [y0, y1, y1])
ax[0].set_aspect('equal')

ax[0].set_xlim([0, 5*sigma*pxs])
ax[0].set_ylim([0, 5*sigma*pxs])

ax[1].hist(cell_count, np.max(cell_count) + 1, histtype = 'step')

plt.show()


    # fig, ax = plt.subplots(ncols = len(cells) + 1)
    # ax[-1].imshow(np.sum(cells, axis = 0), origin = 'lower')
    # fig.suptitle(sim)
    # plt.show()