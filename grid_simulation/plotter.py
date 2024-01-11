import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import utils
from scipy.ndimage import gaussian_filter

def LinePlot(input_files, legends, add_error_bars = False):
    scores = [None]*len(input_files)
    times = [None]*len(input_files)
    deviations = [None]*len(input_files)
    durs = [None]*len(input_files)
    print(deviations)

    for i, file in enumerate(input_files):
        f = np.load('grid_simulation/Results/'+file)
        scores[i] = f['scores']
        times[i] = f['save_tick']
        z = scores[i]
        deviations[i] = np.std(z, 1) / np.sqrt(13)
        durs[i] = 50 if 'duration' not in f else f['duration']//60

    fig = plt.figure()

    for time, score, deviation, duration in zip(times, scores, deviations, durs):
        x_vals = np.arange(0, duration, time//60)
        y_mean = np.mean(score[0:-1], 1)
        plt.plot(x_vals[0:-1], y_mean)
        if add_error_bars:
            plt.fill_between(x_vals[0:-1], y_mean - deviation[0:-1], y_mean + deviation[0:-1], alpha = 0.3, label = '_nolegend_')

    plt.hlines(0, -20, duration + 5, linestyles = 'dashed', colors = 'blue')
    plt.xlabel("Time(mins)")
    plt.ylabel("Mean gridscore across 13 grid cells")
    plt.xlim(-5, np.max(durs) + 5)
    plt.rcParams.update({'font.size': 100})
    plt.legend(legends)
    plt.show()

def WeightsGif(input_file, output_file):
    f = np.load('grid_simulation/Results/' + input_file)
    Ndendrites = f['Ndendrites']
    Ndendrites2 = Ndendrites**2
    sigma = f['sigma']
    Ng = int(f['Ng'])
    save_tick = int(f['save_tick'])
    duration = int(f['duration'])
    weights = f['weights']
    Ng = 13
    n_rows = 3
    fig = plt.figure(figsize=(15,10))
    ax = []
    for y in range (n_rows):
        for z in range(Ng):
            ax.append(plt.subplot2grid((n_rows,Ng),(y,z)))
    
    def init():
        ax[0].scatter([],[])
    
    def update(frame):

    
        grid_weights = np.reshape(weights[frame], (Ndendrites2, Ng))
        print(save_tick/1000*frame/duration)

        for z in range(Ng):
            
            weight2d = np.reshape(grid_weights[:, z], (Ndendrites, Ndendrites))
    
            # compute gridness score
            corr_w = utils.normcorr2d(weight2d)
            gscore, _ = utils.gridness_score(corr_w, Ndendrites, sigma)
            cntr_xy = corr_w.shape[0]//2

            # only consider cells with a score > 0 (as is common in
            # literature)
            if gscore > 0:
                orientation, closest_r, _ = utils.grid_orientation(corr_w, Ndendrites, sigma)
            else:
                orientation = -1
                closest_r = np.array([0, 0])

            # show weights
            ax[z].imshow(weight2d, interpolation='none', origin='lower')
            ax[z].set_title("%3.2f" % (gscore))

            # show gaussian filtered weights
            ax[Ng+z].imshow(gaussian_filter(weight2d, 1.5), interpolation='none', origin = 'lower')

            # show auto-correlation and nearest blod tracker
            ax[2*Ng+z].cla()
            ax[2*Ng+z].imshow(corr_w, interpolation='none', origin='lower')
            ax[2*Ng+z].autoscale(False)
            ax[2*Ng+z].plot([cntr_xy, cntr_xy + closest_r[0]], [cntr_xy, cntr_xy + closest_r[1]], linewidth=2.0, color='black')
        fig.suptitle(f"{save_tick*frame // 60000} mins {(save_tick*frame/1000) % 60} seconds")
    ani = animation.FuncAnimation(fig, update, duration*1000//save_tick, init, interval = 2000)
    ani.save(f"grid_simulation/Results/{output_file}.gif")

if __name__ == '__main__':
    #WeightsGif('m3f100_1.npz', 'test_gif2')
    # input_files = ['m3f0_0.npz', 'm3f0_1.npz', 'm3f50_0.npz', 'm3f100_0.npz', 'm3f100_1.npz', 'm3f150_0.npz']
    # legends = ["No baseline", 'No baseline', "Low Baseline", "Moderate baseline", "Moderate baseline", "High baseline"]
    # long_files = ["m3f0_1.npz", "m3f100_1.npz", "m4f100_1.npz"]
    # long_legends = ["No baseline or time dependence", "Moderate baseline", "Moderate baseline and time dependence"]
    LinePlot(["test.npz"], ["_nolegend_"], True)
