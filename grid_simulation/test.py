import numpy as np

Nthetas = 12
Ndists = 11
max_dist = 1
Nbvcs = Nthetas*Ndists
boundary_cells = np.meshgrid(np.arange(0,180, int(180/Nthetas), dtype =int), np.linspace(0, max_dist, Ndists))
boundary_cells = np.reshape(boundary_cells, (2,-1))
print(boundary_cells)
