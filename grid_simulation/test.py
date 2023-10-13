import numpy as np
import matplotlib.pyplot as plt

files = np.load('grid_simulation/Results/test.npz')
scores = files['scores']
plt.plot(scores[0:-1])
plt.show()