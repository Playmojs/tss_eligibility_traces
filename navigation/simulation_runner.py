#%%
import nav_eligibility4 as nav_el
import module_utils as mod_util
import matplotlib.pyplot as plt
import numpy as np

#%%
# Setup data structure:

variances = [1.8,2.2]
distances = [0, 2, 4, 6, 8, 10]
transition_dist = 1
delay = 5
neuron_number = 400
reps = 100

data = np.empty((len(variances), len(distances), reps), dtype=object)

#%%
for i, variance in enumerate(variances):
    variance_complete = False
    sims_with_valid_distance = np.ones(len(distances), dtype = int) * reps
    while not variance_complete:
        print(np.sum(sims_with_valid_distance))
        symbols = mod_util.generateRandomSymbols(neuron_number, delay, [0,10], [0,10], variance)
        [start, goal] = np.random.choice(len(symbols), 2, False)
        dist = np.linalg.norm(symbols[start].coord - symbols[goal].coord)
        dist_diffs = dist - distances
        idx = np.where(dist_diffs > 0, dist_diffs, np.inf).argmin()
        if sims_with_valid_distance[idx] == 0:
            continue
        sims_with_valid_distance[idx] -= 1
        data[i, idx, sims_with_valid_distance[idx]] = nav_el.eligibilityNavigation(symbols, start, goal, transition_dist, 1, 2, delay, variance, False, False, save_data = True)
        if np.sum(sims_with_valid_distance) == 0:
            variance_complete = True


# %%
for i in range(len(variances)):
    for j in range(len(distances)):
        np.savez(file=f"result_data/m4_data/m4_{variances[i]}ms_{distances[j]}dist", data = data[i,j,:])
# %%
nav_successes = np.empty((3,6,100), dtype = bool)
tag_successes = np.empty((3,6,100), dtype = bool)

#%%
it = np.nditer(data, flags = ['multi_index', 'refs_ok'])

#%%
for rec in it:
    nav_successes[it.multi_index] = rec.item().success
    tag_successes[it.multi_index] = rec.item().correct_tag
#%%
x_vals = ["1-2", "2-4", "4-6", "6-8", "8-10", "10-14"]
plt.plot(x_vals, np.transpose(results['success_rates']))
plt.xlabel("Number of transitions")
plt.ylabel("p(success)")
plt.legend(variances)
# %%
