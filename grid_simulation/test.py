import numpy as np
import matplotlib.pyplot as plt
import utils

Nthetas = 12
Ndists = 11
max_dist = np.sqrt(2)
Nbvcs = Nthetas*Ndists
boundary_cells = np.meshgrid(np.arange(0,180, int(180/Nthetas), dtype =int), np.linspace(0, max_dist, Ndists))
boundary_cells = np.reshape(boundary_cells, (2,-1))

duration = 1 * 10**5

X, speed, boundaries, boundary_vectors = utils.getTrajValues(f"grid_simulation/Trajectories/square/900s.npz")


acts = utils.BVC_act(boundary_cells, boundary_vectors[:1000], Nbvcs)
delays = (1/acts - 1)
delays = delays / np.max(delays) * 80

indices = np.where(delays < 20)
spike_times = delays[indices] + 100*indices[0]
neuron_indices = indices[1]

sorted_indices = np.argsort(spike_times)



# filter = 0.37
# neuron_indices = np.arange(Nbvcs, dtype = int) # shape (Nbvcs, )

# theta_times = np.arange(0, duration, 100) # shape (Ndurations, )
# activity = utils.BVC_act(boundary_cells, boundary_vectors, Nbvcs) # shape (Ndurations, Nbvcs)
# BVC_delay = 1/activity - 1
# filtered_neuron_indices = neuron_indices[BVC_delay<filter] 
# filtered_spike_times = BVC_delay[BVC_delay<filter]*20/filter 

# for i in np.arange(0, duration//100, dtype = int):
#     time_ms = i*100

#     # Getting BVC rates, as described in literature:
    
#     iBoundaries = boundary_vectors[i]
#     activity = utils.BVC_act(boundary_cells, iBoundaries, Nbvcs)

#     # Translating rates to activation delays, filtering and sorting:
#     BVC_delay = (1/(activity) - 1)
#     filtered_neuron_indices = neuron_indices[BVC_delay<filter]
#     filtered_spike_times = BVC_delay[BVC_delay<filter]*20/filter # The factor scales the activity so the slowest neurons fire 20 ms after theta, and the fastest up to theta
#     temp_input = np.array((filtered_neuron_indices, filtered_spike_times + time_ms))
#     temp_input = temp_input[:, np.argsort(temp_input[1,:])]
#     inputs = np.hstack((inputs, temp_input))
# print("\n")
# del(i)