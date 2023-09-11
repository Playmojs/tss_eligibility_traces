from brian2 import *
import matplotlib.pyplot as plt
import utils
import numpy as np

x = utils.CoordinateSamplers(48, 0.1, [0,1], [0,1])
y = x.act([0,0])

tau = 10*ms

eqs = '''
dv/dt = (v0-v)/tau : 1
v0 : 1
'''

G = NeuronGroup(48**2, eqs, threshold = 'v> 0.2', reset = 'v = 0', method = 'exact')
M = StateMonitor(G, 'v', record = 95)

G.v0 = np.ndarray.flatten(y)

run(50*ms)
plt.plot(M.t/ms, M.v[0])
plt.xlabel('Time (ms)')
plt.ylabel('v')
plt.show()
