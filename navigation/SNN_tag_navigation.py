from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import module_utils
import simulation_utils as sim_utils
import recorder as rec

# Set up constants
n_symbols = 100
dist = 2
frames_per_ms = 1
min_delay = 6
base_delay = 8
recorder = rec.Recorder(frames_per_ms)

# Generate symbols and transition values
symbols = module_utils.generateRandomSymbols(n_symbols, base_delay, [0,10], [0,10], 1)
transition_ids = module_utils.findAllTransitions(symbols, dist)
[start, goal] = np.random.choice(len(symbols), 2, False)
start_delays = [symbol.spike_delay_ms for symbol in symbols]*ms

# Create neuron layer of symbols
symbol_eq = '''dv/dt = -v/(10*ms) : 1 (unless refractory)
            depsilon/dt = - epsilon/(15*ms) : 1
            dinhibitE/dt = - inhibitE/(1*ms) : 1
            delays : second
            inhibitPower : 1
            halt : boolean'''

symbol_layer = NeuronGroup(n_symbols, symbol_eq, threshold = 'v > 1 or (t/second==0 and i == start)', reset='v = 0', refractory = 20*ms, method = 'exact')
symbol_layer.delays = start_delays
symbol_layer.halt = np.zeros(n_symbols)

# Set up synaptic values between symbols compatible with brian2 syntax
pre_post_ids = np.empty((2,0), dtype = int)
for i, transitions in enumerate(transition_ids):
    for transition_id in transitions.transition_ids:
        pre_post_ids = np.hstack((pre_post_ids, np.reshape([i, transition_id],(2,1))))
 

# Set up synapses within symbol layer
transition_synapses = Synapses(symbol_layer, symbol_layer, model = 'neuron_delay = delays_pre : second', on_pre = '''v += 2
                               epsilon_pre = 2
                               delays_post = (min_delay + (delays_post/ms-min_delay) * (1 - epsilon_post * inhibitE_post))*ms''')
transition_synapses.connect(i = pre_post_ids[0,:], j = pre_post_ids[1,:])

# Set up inhibitory layer and synapses
inhibitory_layer = NeuronGroup(n_symbols, '''dv/dt = (y-v)/ms : 1
                               dy/dt = -y/(tau*ms) : 1
                               tau : 1''', method = 'exact', threshold = 'v > 1', reset = 'v = 0')
inhibitory_layer.tau = min_delay
inhibitory_layer[goal].tau += 4

inhibitory_synapses = Synapses(symbol_layer, inhibitory_layer, on_pre = 'y_post = inhibitPower_pre')
inhibitory_synapses.connect(i = 'j')

inhibit_synapses = Synapses(inhibitory_layer, symbol_layer, on_pre = '''v_post = -5
                            inhibitE_post = 1.5''')
inhibit_synapses.connect()

# Set up unique goal-start connection to restart the wave
restart_synapse = Synapses(symbol_layer, symbol_layer, on_pre = '''v = 1.2
                           delays_pre = 7*ms
                           halt_pre = True''', delay = 30*ms)
restart_synapse.connect(i = goal, j = start)

# Set up spike monitor and run update functions to generate figures
place_monitor = SpikeMonitor(symbol_layer)
S = StateMonitor(symbol_layer, ['v', 'epsilon', 'inhibitE'], record = [start, 7])
I = StateMonitor(inhibitory_layer, ['v', 'y'], record = goal)

@network_operation(dt = frames_per_ms*ms)
def catch_frame(t):
    plot_data = np.empty((2,0))
    alphas = np.empty(0)
    colors = np.empty(0)
    time = t/ms
 
    for i, activation_time in zip(place_monitor.i, place_monitor.t):
        passed_time_ms = time - activation_time/ms
        if passed_time_ms > 20:
            continue
        plot_data = np.hstack((plot_data, np.reshape(symbols[i].coord,(2,1))))
        norm_delt = passed_time_ms / 20
        alphas = np.append(alphas, (norm_delt**3-2*norm_delt**2+norm_delt)/0.15)
        colors = np.append(colors, 'y' if i == start else 'g' if i == goal else 'b')
    recorder.plots.append(plot_data)
    recorder.alphas.append(alphas)
    recorder.color_codes.append(colors)

@network_operation(dt=1*ms)
def check_stop():
    if np.any(symbol_layer.halt):
        symbol_layer.halt = np.zeros(n_symbols)
        stop()

for i in range(4):
    
    # Set synapse delays according to the central neuron delays (brian2 doesn't currently support dynamic delays otherwise)
    transition_synapses.delay = transition_synapses.neuron_delay

    # Update other synaptic variables that depend on the delay
    symbol_layer.inhibitPower = 4*(1-1/(1+np.exp(-50*(symbol_layer.delays/ms-base_delay+0.1))))
    symbol_layer.inhibitPower[goal] = 5

    # Reinitiate run
    run(200*ms)

print(symbol_layer.delays)
print(symbol_layer.inhibitPower)

recorder.createAnimation()
plt.close()

plt.subplot(221)
plt.plot(S.t/ms, S.v[0], 'C0')
plt.plot(S.t/ms, S.epsilon[0], 'b')
plt.plot(S.t/ms, S.inhibitE[0], 'r')
plt.subplot(222)
plt.plot(I.t/ms, I.v[0], 'C0')
plt.subplot(223)
plt.plot(I.t/ms, I.y[0], 'C0')
plt.subplot(224)
plt.plot(S.t/ms, S.v[1], 'C0')
plt.show()