import numpy as np
import sys
import matplotlib.pyplot as plt
import utils

# score = np.load("grid_simulation\Results\data\mf3100_1_opt_g_score.npy")
# with np.load('grid_simulation/Results/data/m3f100_1.npz') as data:
#         old_scores = data['scores']
# plt.plot(np.mean(score, axis = 1))
# plt.plot(np.mean(old_scores, axis=1))
# plt.show()

with np.load("grid_simulation/Results/data/test_regular0.npz", allow_pickle=True) as data:
    print(data.files)
    print(data['input_pos'])