import numpy as np

x = np.array(([ 0,  -0.2],
 [ 0,   0.2],
 [ 1.5,  0.5],
 [ 1.5, -0.5]))

print(x)
print(np.min(x, axis = 0))