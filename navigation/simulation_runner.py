#%%
import nav_eligibility4 as nav_el
import module_utils as mod_util
import matplotlib.pyplot as plt
import numpy as np

#%%
# Setup data structure:

variances = [0,1,2,3,4]
distances = [0, 2, 4, 6, 8, 10]
transition_dist = 1
delay = 5
neuron_number = 400
reps = 100

success_rates = np.zeros((len(variances), len(distances)))

#%%
for i, variance in enumerate(variances):
    variance_complete = False
    sims_with_valid_distance = np.ones(len(distances)) * reps
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
        success_rates[i, idx] += nav_el.eligibilityNavigation(symbols, start, goal, transition_dist, 1, 2, delay, variance, False, False) / 100
        if np.sum(sims_with_valid_distance) == 0:
            variance_complete = True
    print(success_rates)


# %%
np.savez(file="variance_distance_numbers", success_rates = success_rates)
# %%
results = np.load("variance_distance_numbers.npz")
#%%
x_vals = ["1-2", "2-4", "4-6", "6-8", "8-10", "10-14"]
plt.plot(x_vals, np.transpose(results['success_rates']))
plt.xlabel("Number of transitions")
plt.ylabel("p(success)")
plt.legend(variances)
# %%
