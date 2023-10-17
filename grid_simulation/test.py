import numpy as np
import matplotlib.pyplot as plt

file_names = ['m3f0_0.npz', 'm3f50_0.npz', 'm3f100_0.npz', 'm3f100_1.npz', 'm3f150_0.npz']

scores = [None]*len(file_names)
times = [None]*len(file_names)
deviations = [None]*len(file_names)
durs = [None]*len(file_names)
print(deviations)

for i, file in enumerate(file_names):
    f = np.load('grid_simulation/Results/'+file)
    scores[i] = f['scores']
    times[i] = f['save_tick']
    z = scores[i]
    deviations[i] = np.std(z, 1) / np.sqrt(13)
    durs[i] = 50 if 'duration' not in f else f['duration']/60

fig = plt.figure()
legends = ["No baseline", "Low Baseline", "Moderate baseline", "Moderate baseline", "High baseline"]

for time, score, deviation, duration in zip(times, scores, deviations, durs):
    x_vals = np.arange(0, duration, time/60000)
    y_mean = np.mean(score[0:-1], 1)
    plt.plot(x_vals, y_mean)
    #plt.fill_between(x_vals, y_mean - deviation[0:-1], y_mean + deviation[0:-1], alpha = 0.3, label = '_nolegend_')

plt.hlines(0, -20, duration + 5, linestyles = 'dashed', colors = 'blue')
plt.xlabel("Time(mins)")
plt.ylabel("Mean gridscore across 13 grid cells")
plt.xlim(-5, np.max(durs) + 5)
plt.legend(legends)
plt.show()