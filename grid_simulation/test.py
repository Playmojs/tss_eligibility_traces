import numpy as np

ran = np.arange(0,10)

rand = np.random.exponential(ran, len(ran))
print(ran, rand)