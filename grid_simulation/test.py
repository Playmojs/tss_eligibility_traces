import numpy as np
import matplotlib.pyplot as plt
import utils

# grid= utils.createHexField(25, 0.3, np.radians(7.5), 'square')
# plt.imshow(grid, origin = 'lower')
# plt.show()

spatialns = utils.CoordinateSamplers(48, 0.1, )
positions = np.random.rand(50,2)
dists = spatialns.dist(positions)
print(np.shape(dists))

# Nthetas = 20
# Ndists = 21
# max_dist = np.sqrt(2)
# Nbvcs = Nthetas*Ndists
# boundary_cells = np.meshgrid(np.arange(0,180, int(180/Nthetas), dtype =int), np.linspace(0, max_dist, Ndists))
# boundary_cells = np.reshape(boundary_cells, (2,-1))

# z = np.reshape(boundary_cells[1,:], (Ndists,Nthetas)).T
# print(z)

# duration = 1 * 10**5

# X, speed, boundaries, boundary_vectors = utils.getTrajValues(f"grid_simulation/Trajectories/square/900s.npz")


# acts = utils.BVC_act(boundary_cells, boundary_vectors[:1000], Nbvcs)
# delays = (1/acts - 1)
# delays = delays / np.max(delays) * 80

# indices = np.where(delays < 20)
# spike_times = delays[indices] + 100*indices[0]
# neuron_indices = indices[1]

# sorted_indices = np.argsort(spike_times)