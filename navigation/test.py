import numpy as np

nscales = 3
inhibit_window = np.ones((nscales, 2)) * np.inf
trace = np.zeros(nscales)
trace = np.vstack((trace, np.full(nscales, nscales))).T
print(trace)

for i in trace:
    print(i)