import numpy as np
import matplotlib.pyplot as plt

file_names = ['m3f0_0.npz', 'm3f100_0.npz', 'm3f150_0.npz']

scores = [0,0,0]
times = [0,0,0]
deviations = [0,0,0]
for i, file in enumerate(file_names):
    f = np.load('grid_simulation/Results/'+file)
    scores[i] = f['scores']
    times[i] = f['save_tick']
    deviations[i] = np.std(scores[i], 1) / np.sqrt(13)

duration = 3000 / 60

fig = plt.figure()
legends = ["No baseline", "Moderate baseline", "High baseline"]

for time, score, deviation in zip(times, scores, deviations):
    x_vals = np.arange(0, duration, time/60000)
    y_mean = np.mean(score[0:-1], 1)
    plt.plot(x_vals, y_mean)
    plt.fill_between(x_vals, y_mean - deviation[0:-1], y_mean + deviation[0:-1], alpha = 0.3, label = '_nolegend_')

plt.hlines(0, -20, duration + 5, linestyles = 'dashed', colors = 'blue')
plt.xlabel("Time(mins)")
plt.ylabel("Mean gridscore across 13 grid cells")
plt.xlim(-5, duration + 5)
plt.legend(legends)
plt.show()