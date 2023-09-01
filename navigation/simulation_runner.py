#%%
import navigation.nav_eligibility1 as nav_el
import module_utils as mod_util
import numpy as np

#%%

symbols = mod_util.generateRandomSymbols(100, [0,10], [0,10],1)

#%%
[start, goal] = np.random.choice(len(symbols), 2, False)
print(symbols[start].coord, symbols[goal].coord)
# %%

f_values = [0, 2, 4, 5, 6]
titles = ["0", "2", "4", "5", "6"]

#%%

for i in range(len(f_values)):
    nav_el.eligibilityNavigation(symbols, start, goal, 2, 1, f_values[i], titles[i])
# %%
