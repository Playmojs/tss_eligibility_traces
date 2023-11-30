import numpy as np

nsymbols = 10
xrange = [0,10]
yrange = [0,10]
base_dist = [0, 2, 4, 6]

coords = np.hstack((np.random.uniform(xrange[0], xrange[1], (nsymbols,1)), np.random.uniform(yrange[0], yrange[1], (nsymbols,1))))

pairwise_distances = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=2)

ranges_broadcasted = np.broadcast_to(np.array(base_dist)[:, np.newaxis, np.newaxis], (len(base_dist), nsymbols, nsymbols))
print(ranges_broadcasted)

distances_mask = np.logical_and(pairwise_distances > ranges_broadcasted[:-1], pairwise_distances <= ranges_broadcasted[1:])

print (distances_mask)

#print(transition_indices)
#print(np.shape(transition_indices))
