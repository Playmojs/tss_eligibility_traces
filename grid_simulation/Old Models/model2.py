import os
import sys
import h5py
import argparse
import numpy as np
import utils
import matplotlib.pyplot as plt
from timeit import default_timer as timer


# Total number of dendrites
Ndendrites = 48

# Total number of grid cells to simulate
Ng = 3

# Dendritic tree overlap
sigma = 0.05

# Exploration length
long_exploration = True

# Save frequency
save_tick = 1000

# Visualize
visualize = True
visualize_tick = 5000

# Fix further constants
Ndendrites2 = Ndendrites **2

# Set up grid cells and input cells
grid_cells = [utils.GridCell(i, Ndendrites, sigma) for i in range(Ng)]
spatialns = utils.CoordinateSamplers(Ndendrites, sigma)

# Read file to get positions and velocities
X, speed = utils.getCoords(h5py.File("grid_simulation/trajectory_square_2d_0.01dt.hdf5", "r"))
mean_speed = np.mean(speed)
tMax = len(X)

# Set up data storing variables
weight_tracker = np.zeros((tMax // save_tick + 1, Ng, Ndendrites, Ndendrites))
corr_tracker = np.zeros((tMax // save_tick + 1, Ng, Ndendrites * 2 - 1, Ndendrites * 2 - 1))
score_tracker = np.zeros((tMax // save_tick + 1, Ng, 1))
orientation_tracker = np.zeros((tMax // save_tick + 1, Ng, 1))
closest_tracker = np.zeros((tMax // save_tick + 1, Ng, 2))
save_id = 0


# Setup plot if wanted

if visualize:
    import matplotlib
    import matplotlib.pyplot as plt

    # suppress deprecation warning from matplotlib.
    import warnings
    warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

    fig = plt.figure()
    ax = []
    ax.append(plt.subplot2grid((2,Ng+1), (0,0)))
    for z in range(1,Ng+1):
        ax.append(plt.subplot2grid((2,Ng+1),(0,z)))
    for z in range(1,Ng+1):
        ax.append(plt.subplot2grid((2,Ng+1),(1,z)))

# Simulate
wins = []
print ("Commence")
for t in range (tMax):
    # Current coordinate
    x = X[t, :]

    # Get input activation: 
    # A is all input, 
    # B is only input from nearby place cells, 
    # C is input from intermediary place cells, 
    # D is input from both B and C

    A = spatialns.act(x)
    i68, i95, i99 = spatialns.get68_95(x)
    B = A.copy()
    C = A.copy()
    B[~i68] = 0
    C[~i95] = 0
    D = A.copy()
    D[~np.logical_or(i68, i95)] = 0

    # The activation of each grid cell, using only place cells in B:
    gact = np.array([grid_cells[i].activity(B) / np.sum(B).sum() for i in range(Ng)])
    
    #print(gact)
    
    #TODO: remove the instant win/lose - dynamics, introducing delay
    win_id = np.argmax(gact)
    wins.append(win_id)

    #Learning speed, based on the animal's speed of movement:
    eta = 1.0 * np.exp(-1/mean_speed * speed[t]**2)
    
    mu0 = (Ng-1)/Ng * np.sum(B>0) #Sum of activation of all active input?
    mu1 = (Ng-1)/Ng * np.sum(C>0) #Sum of activation of all mid-active input?

    # Update weights:
    for grid_cell in range(Ng):
        # Probably smart to treat the actual weights with care, and mess around with the copies instead:
        w = grid_cells[grid_cell].w.copy()
        
        # I have yet to figure out why this line is here, but it seems to punish high weights more than low weights
        baseline = (1/Ng) * (-2 / Ndendrites2 * (1-w)) * (D>0)

        # Punish cells that were highly active in this location:
        coact = 0
        for j in range(Ng):
            if not j == grid_cell:
                coact += gact[j] * gact[grid_cell]
        coact /= (Ng-1)

        # However, if this cell is the winning cell, strengthen its weights in this location. 
        # Also, weaken weighs in the surround area
        # This creates an ON-center OFF-surround mechanic

        on  = mu0 * (B > 0) * (-4/Ndendrites2 * w) * np.exp((gact[win_id]-gact[grid_cell])*10/gact[win_id]) # correlation
        off = mu1 * (C > 0) * (+4/Ndendrites2 * w) * np.exp((gact[win_id]-gact[grid_cell])*10/gact[win_id]) # decorrelation

        # Combine all the weight changes:
        w = w - eta * (baseline + coact + (on + off)) / 3

        # Non-linear weight modification and clamping
        w = w[D > 0]
        w = np.tanh(w)
        w[w < 0] = 0

        # Update weights stored along grid cell objects
        grid_cells[grid_cell].w[D > 0] = w.copy()

    #Track data
    if (t % save_tick == 0) or (t == tMax):
        sys.stdout.write('.')
        sys.stdout.flush()
        for z in range(Ng):
            # compute gridness score
            corr_w = utils.normcorr2d(grid_cells[z].w)
            gscore, _ = utils.gridness_score(corr_w, Ndendrites, sigma)

            # center location of the correlation
            cntr_xy = corr_w.shape[0]//2

            # only consider cells with a score > 0 (as is common in
            # literature)
            if gscore > 0:
                orientation, closest_r, _ = utils.grid_orientation(corr_w, Ndendrites, sigma)
            else:
                orientation = -1
                closest_r = np.array([0, 0])

            # save data to trackers

            weight_tracker[save_id, z, ...] = grid_cells[z].w
            corr_tracker[save_id, z, ...] = corr_w
            score_tracker[save_id, z] = gscore
            orientation_tracker[save_id, z] = orientation
            closest_tracker[save_id, z, ...] = closest_r
        save_id += 1
    if visualize and (t % visualize_tick == 0 or t+1 == tMax):
        # show current location activity
        ax[0].cla()
        ax[0].imshow(A * i68, interpolation='none', origin='lower')
        ax[0].set_title("%d" % t)

        for z in range(Ng):
            # compute gridness score
            corr_w = utils.normcorr2d(grid_cells[z].w)
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
            ax[z+1].imshow(grid_cells[z].w, interpolation='none', origin='lower')
            ax[z+1].set_title("%3.4f, %4.2f" % (gscore, orientation))

            # show auto-correlation and nearest blod tracker
            ax[1+Ng+z].cla()
            ax[1+Ng+z].imshow(corr_w, interpolation='none', origin='lower')
            ax[1+Ng+z].autoscale(False)
            ax[1+Ng+z].plot([cntr_xy, cntr_xy + closest_r[0]], [cntr_xy, cntr_xy + closest_r[1]], linewidth=2.0, color='black')

        # ax[4].set_title(grid_cells[0].w.min())
        # ax[5].set_title(grid_cells[0].w.mean())
        # ax[6].set_title(grid_cells[0].w.max())

        plt.pause(1)
if visualize:
    plt.show()

#print(wins)
