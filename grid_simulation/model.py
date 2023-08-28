
import os
import sys
import h5py
import argparse
import numpy as np
import scipy.signal as sig

import matplotlib.pyplot as plt
from timeit import default_timer as timer

import scipy.ndimage as ndimage


# Total number of dendrites

Ndendrites = 48

# Total number of grid cells to simulate

Ng = 3

# Dendritic tree overlap
sigma = 0.10

# Exploration length
long_exploration = True

# Save frequency:
save_tick = 1000

# Visualize
visualize = True
visualize_tick = 5000


def getCoords(f5):
    # Position is returned as a numpy array of 2d-arrays
    positions= f5['agent/position']
    positions = np.asarray(positions)
    positions += 0.5
    # Speed is returned as a numpy array of positive scalars
    velocity = f5['agent/velocity']
    speed = np.linalg.norm(velocity, axis = 1)
    speed = np.append(speed, 0)
    print(len(speed))
    return positions, speed

def gauss(D, sig):
    return 1 / np.sqrt(2 * np.pi * sig) * np.exp(- (D)**2 / (2 * sig**2))


def gaussNonnorm(D, sig):
    return np.exp(- (D)**2 / (2 * sig**2))


# The three functions below are simply copied
def normcorr2d(A, B=None):
    """normalized correlation of a 2D input array A with mask B. If B is
    not specified, the function computes the normalized utocorrelation
    of A."""

    if B == None:
        B = A

    N = A.shape[0] * A.shape[1]
    R = A - np.mean(A)
    S = B - np.mean(B)
    T = sig.correlate2d(R, S, mode='full') / (N * np.std(A) * np.std(B))
    return T

def gridness_score(R, s0, s1):
    """Compute the Gridness Score of a autocorrelated rate map R"""

    dim0 = R.shape[0]
    cntr = dim0 // 2

    # create a ring filter to search for the six closest fields
    in_ra = int(2 * sigma * Ndendrites)
    out_ra = int(4 * sigma * Ndendrites)
    RingFilt = np.zeros((dim0, dim0))
    for i in range(dim0):
        for j in range(dim0):
                cntr_i = (cntr - i)**2
                cntr_j = (cntr - j)**2
                dist = cntr_i + cntr_j
                if in_ra**2 <= dist and dist <= out_ra**2:
                    RingFilt[i, j] = 1

    Tmp = np.multiply(R, RingFilt)
    Tmp = Tmp[cntr - out_ra - 1:cntr + out_ra + 1, cntr - out_ra - 1 : cntr + out_ra + 1]
    nx, ny = Tmp.shape[0], Tmp.shape[1]

    corrot = np.zeros(180)
    for idrot in range(180):
        rot = ndimage.rotate(Tmp, idrot, reshape=False)
        A = Tmp
        B = rot

        dotAA = np.sum(A*A, axis=0)
        dotA0 = np.sum(A*np.ones((nx, ny)), axis=0)
        dotBB = np.sum(B*B, axis=0)
        dotB0 = np.sum(B*np.ones((nx, ny)), axis=0)
        dotAB = np.sum(A*B, axis=0)

        corrot[idrot] = (nx * ny * np.sum(dotAB) - np.sum(dotA0) * np.sum(dotB0)) / \
                 (np.sqrt(nx * ny * np.sum(dotAA) - np.sum(dotA0)**2)*np.sqrt(nx*ny*np.sum(dotBB)) - np.sum(dotB0)**2)

    gridscore = min(corrot[59],corrot[119]) - max(corrot[29], corrot[89], corrot[149])
    return gridscore, Tmp

def grid_orientation(Tmp, Ndendrites, sigma):
    """Compute the orientation of the grid using the cut-out of the
    auto-correlation map which was used forgridness score computation""" 

    # center of the correlogram
    cntr_xy = Tmp.shape[0]//2

    # find all maxima in the figure
    nbr = 2 * sigma * Ndendrites
    data_min = 0
    data_max = ndimage.maximum_filter(Tmp, nbr)
    maxima = (Tmp == data_max)

    # retrieve all coordinates that are above the 'zero line'
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    xs, ys = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        y_center = (dy.start + dy.stop - 1)/2

        # compute distance
        d = np.sqrt((x_center - cntr_xy)**2 + (y_center - cntr_xy)**2)

        # restrict to above line, not center, and within the 2-sigma ring
        if (d <= 2 * nbr) and (y_center >= cntr_xy) and not (x_center == cntr_xy and y_center == cntr_xy):
            xs.append(x_center)
            ys.append(y_center)

    # compute locations relative to autocorrelogram center
    if len(xs) == 0 or len(ys) == 0:
        return -1, np.array([0, 0]), 0
    xo = np.asarray([x - cntr_xy for x in xs])
    yo = np.asarray([y - cntr_xy for y in ys])
    dmin = xo.argmax()

    # compute orientation and closest coordinate
    orientation = np.arctan2(yo[dmin],  xo[dmin]) * 180 / np.pi
    closest_relative = np.array([xo[dmin], yo[dmin]])
    closest_absolute = np.array([xs[dmin], ys[dmin]])

    return orientation, closest_relative, closest_absolute


class GridCell():
    def __init__(self, id, Ndendrites, sigma):
        self.id = id
        self.w = np.random.rand(Ndendrites,Ndendrites)
        self.w[self.w < 0.9] = 0
        self.w[self.w > 0] = 1.0
        self.w = np.tanh(self.w)
        self.sigma = sigma
    def activity(self, inp):
        return np.sum(np.sum(self.w * inp, axis = 1))

#TODO: Replace coordinate samplers with border cells
class CoordinateSamplers():
    def __init__(self, N, tuning_width, xlim=np.array([0,1]), ylim=np.array([0,1]), distrib = 'regular'):
        self.N = N
        self.N2 = N**2
        self.tuning_width = tuning_width
        self.xlim = xlim
        self.ylim = ylim
        self.samplers = np.zeros((N,N))

        if distrib == 'regular':
            # generate meshgrid of locations on the coordinate space
            x = np.linspace(xlim[0], xlim[1], N)
            y = np.linspace(ylim[0], ylim[1], N)
            xv, yv = np.meshgrid(x, y, indexing='xy')
            self.Xs = np.array((xv, yv))
            self.Xs = np.rollaxis(self.Xs, 0, 3)

    def dist(self, X):
        return np.sqrt(np.sum((self.Xs - X)**2, axis=2))

    def act(self, X):
        D = self.dist(X)
        A = gaussNonnorm(D, self.tuning_width)
        return A
    
    def get68_95(self, X):
        """Compute all neurons which are within one sigma or two sigma of the
        epi-center located at X. The computation follows according to the
        68-95-99.7 rule. The returned sets contain flattened indexes."""

        D = self.dist(X)
        i68 = D <= self.tuning_width
        i95 = (D > self.tuning_width) & (D < 2.0*self.tuning_width)
        i99 = (D > 2.0*self.tuning_width) & (D < 3.0*self.tuning_width)
        # return self.mask_to_flattened_index(i68), self.mask_to_flattened_index(i95)
        return i68, i95, i99

# Fix further constants:
Ndendrites2 = Ndendrites **2

# Set up grid cells and input cells:
grid_cells = [GridCell(i, Ndendrites, sigma) for i in range(Ng)]
spatialns = CoordinateSamplers(Ndendrites, sigma)

# Read file to get positions and velocities:
X, speed = getCoords(h5py.File("grid_simulation/trajectory_square_2d_0.01dt.hdf5", "r"))
mean_speed = np.mean(speed)
tMax = len(X)


weight_tracker = np.zeros((tMax // save_tick + 1, Ng, Ndendrites, Ndendrites))
corr_tracker = np.zeros((tMax // save_tick + 1, Ng, Ndendrites * 2 - 1, Ndendrites * 2 - 1))
score_tracker = np.zeros((tMax // save_tick + 1, Ng, 1))
orientation_tracker = np.zeros((tMax // save_tick + 1, Ng, 1))
closest_tracker = np.zeros((tMax // save_tick + 1, Ng, 2))
save_id = 0

plt.plot(X[:,0], X[:,1])



#Simulate

if visualize:
    import matplotlib
    import matplotlib.pyplot as plt

    # suppress deprecation warning from matplotlib. I know it's soooooo bad...
    import warnings
    warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

    fig = plt.figure()
    ax = []
    ax.append(plt.subplot2grid((2,Ng+1), (0,0)))
    for z in range(1,Ng+1):
        ax.append(plt.subplot2grid((2,Ng+1),(0,z)))
    for z in range(1,Ng+1):
        ax.append(plt.subplot2grid((2,Ng+1),(1,z)))

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
    
    #TODO: remove the instant win/lose - dynamics, introducing delay
    win = np.argmax(gact)
    wins.append(win)
    #Learning speed, based on the animal's speed of movement:
    eta = 1.0 * np.exp(-1/mean_speed * speed[t]**2)

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
        on = 0
        off = 0
        if grid_cell == win:
            mu0 = (Ng-1)/Ng * np.sum(B>0) #Number of center cells
            mu1 = (Ng-1)/Ng * np.sum(C>0) #Number of surround cells
            on  = mu0 * (B > 0) * (-4/Ndendrites2 * w) # correlation
            off = mu1 * (C > 0) * (+4/Ndendrites2 * w) # decorrelation

        # Combine all the weight changes:
        w = w - eta * (baseline + coact + on + off) / 3

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
            corr_w = normcorr2d(grid_cells[z].w)
            gscore, _ = gridness_score(corr_w, Ndendrites, sigma)

            # center location of the correlation
            cntr_xy = corr_w.shape[0]//2

            # only consider cells with a score > 0 (as is common in
            # literature)
            if gscore > 0:
                orientation, closest_r, _ = grid_orientation(corr_w, Ndendrites, sigma)
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
    if visualize and (t % visualize_tick == 0):
        # show current location activity
        ax[0].cla()
        ax[0].imshow(A * i68, interpolation='none', origin='lower')
        ax[0].set_title("%d" % t)

        for z in range(Ng):
            # compute gridness score
            corr_w = normcorr2d(grid_cells[z].w)
            gscore, _ = gridness_score(corr_w, Ndendrites, sigma)
            cntr_xy = corr_w.shape[0]//2

            # only consider cells with a score > 0 (as is common in
            # literature)
            if gscore > 0:
                orientation, closest_r, _ = grid_orientation(corr_w, Ndendrites, sigma)
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

        ax[4].set_title(grid_cells[0].w.min())
        ax[5].set_title(grid_cells[0].w.mean())
        ax[6].set_title(grid_cells[0].w.max())

        plt.pause(0.1)
if visualize:
    plt.show()

#print(wins)
