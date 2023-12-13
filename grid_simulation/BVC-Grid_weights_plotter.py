import numpy as np
import matplotlib.pyplot as plt
import utils
import trajectory_gen

pxs = 25
sigma = 0.3
wall_angle_offset = 7.5
boundary_shape = 'square'

Nthetas = 20
Ndists = 21
max_dist = 1
Nbvcs = Nthetas*Ndists
boundary_cells = np.meshgrid(np.arange(0,180, int(180/Nthetas), dtype =int), np.linspace(0, max_dist, Ndists))
boundary_cells = np.reshape(boundary_cells, (2,-1))

BVC_weights = np.zeros((Ndists, Nthetas))

firing_field = utils.createHexField(pxs, sigma, wall_angle_offset, boundary_shape)
firing_field = firing_field - np.mean(firing_field)
max_f = np.max(firing_field)
BVC_activity, plot_dims = trajectory_gen.global_bvc_act(boundary_cells, Nbvcs, boundary_shape, pxs)

temp_weights = firing_field[..., np.newaxis] * np.reshape(BVC_activity, (plot_dims[0], plot_dims[1], Nbvcs))
temp_weights = np.reshape(temp_weights, (plot_dims[0], plot_dims[1], Ndists, Nthetas))

#it = np.nditer(firing_field, flags = ['multi_index'])

fig, ax = plt.subplot_mosaic([['firing field', 'weights']])
for i in range(pxs):
    temp_firing_field = firing_field.copy()
    temp_firing_field[i] = max_f
    ax['firing field'].cla()
    ax['firing field'].imshow(temp_firing_field, origin = 'lower')

    BVC_weights += np.sum(temp_weights[i], axis =0)
    ax['weights'].cla()
    ax['weights'].imshow(BVC_weights, origin='lower')
    plt.pause(5)

plt.show()


