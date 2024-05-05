#%%
import numpy as np
import matplotlib.pyplot as plt
import module_utils
import sys

# %%
def getSymbolLayer(symbol_data, index):
    keys = list(symbol_data)
    return symbol_data[keys[index]]

def extractTagged(symbols):
    ids = np.empty(0)
    for i, symbol in enumerate(symbols):
        if symbol.tag:
            ids = np.append(ids, i)
    return ids

def evaluateTag(symbols, tag_ids, goal_id):
    transitions = module_utils.findAllTransitions(symbols, 1, 1, True)
    if len(tag_ids) == 0:
        return True
    correctly_tagged = np.array([goal_id], dtype = int)
    i = 0
    while len(correctly_tagged) > i:
        transition_ids = transitions[correctly_tagged[i]].transition_ids
        for id in transition_ids:
            if symbols[id].tag and id not in correctly_tagged:
                correctly_tagged = np.append(correctly_tagged, id)
        i += 1
    #print(np.sort(correctly_tagged))
    #print(tag_ids)
    if len(correctly_tagged) < len(tag_ids):
        return False
    elif len(correctly_tagged) == len(tag_ids):
        return True
    else:
        raise Exception("Whoops, algorithm fault. Couldn't determine the available tags")
    
def determineSimulationTagFault(symbol_data, goal_id):
    n_iterations = len(symbol_data)
    for i in range(n_iterations):
        symbols = getSymbolLayer(symbol_data, i)
        tag_ids = extractTagged(symbols)
        if not evaluateTag(symbols, tag_ids, goal_id):
            #print(f"Failed on layer {i+1} out of {n_iterations}")
            return False
        #print(f"Success for layer {i+1} out of {n_iterations}")
    return True

    
#%%
variances = [0, 0.5, 1, 1.5, 1.8, 2, 2.2]
distances = [0, 2, 4, 6, 8, 10]
l_var = len(variances)
l_dist = len(distances)
#%%
nav_successes = np.empty((l_var,l_dist,100), dtype = bool)
tag_successes = np.empty((l_var,l_dist,100), dtype = bool)
#%%
for i in range(l_var):
    print(f"Progress: {i/l_var}")
    for j in range(l_dist):
        print(f"Subprogress: {j/l_dist}")
        with np.load(f"result_data/m4_data/m4_{variances[i]}ms_{distances[j]}dist.npz", allow_pickle=True) as d:
            data = d['data']
        for k in range(len(data)):
            data_rec = data[k]
            nav_successes[i,j,k] = data_rec.success
            tag_successes[i,j,k] = data_rec.correct_tag

#%%
np.savez('analysis/rates_comparison', \
          nav_successes = nav_successes, \
          tag_successes = tag_successes, \
          noise_levels = variances, \
          distances = distances)

#%%
with np.load('analysis/rates_comparison.npz') as data:
    nav_successes = data['nav_successes']
    tag_successes = data['tag_successes']
# %%
mean_success = np.mean(nav_successes, axis= 2)
print(mean_success)
# %%
mean_tag_success = np.mean(tag_successes, axis = 2)
print(mean_tag_success)
# %%
p_success_cond_tagS = np.empty((l_var,l_dist))
n_tag_success = np.empty((l_var,l_dist))
p_success_cond_no_tagS = np.empty((l_var,l_dist))
for i in range(l_var):
    for j in range(l_dist):
        temp_cond_success = nav_successes[i,j, tag_successes[i,j]]
        temp_cond_fail = nav_successes[i,j, np.invert(tag_successes[i,j])]
        p_success_cond_tagS[i,j] = np.mean(temp_cond_success)
        p_success_cond_no_tagS[i,j] = np.mean(temp_cond_fail)
        n_tag_success[i,j] = len(tag_successes[i,j,tag_successes[i,j]])

print(p_success_cond_tagS)
print(p_success_cond_no_tagS)
print(n_tag_success)


# %%
fig, axs = plt.subplots(2,2)
fig.set_figwidth(12)
fig.set_figheight(8)
# fig.set_facecolor("#212121")
lines = [None]*4
labels = ["0 %", "6.25 %", "12.5 %", "18.75 %", "22.5 %", "25 %", "27.5 %"]
variances = [0,1,2]
distances = [0,2,4,6,8,10]
ylabels = ["P(success)", "P(correct tag)", "P(success | correct tag)", "P(success | incorrect tag)"]
plot_data = [mean_success.T, mean_tag_success.T, p_success_cond_tagS.T, p_success_cond_no_tagS.T]
axs[0, 0].set_xlabel("Distance from start to goal (m)", size = 12)
for i, ax in enumerate(axs.flatten()):
    lines[i] = ax.plot(distances,plot_data[i], '-o')
    ax.set_ylabel(ylabels[i], size = 15)
    ax.tick_params(axis ='both', labelsize = 12)
axs[0, 0].legend(labels)

# %%
