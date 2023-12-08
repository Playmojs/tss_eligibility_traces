import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as sig

def getCoords(f5):
    # Position is returned as a numpy array of 2d-arrays
    positions= f5['agent/position']
    positions = np.asarray(positions)
    positions += 0.5
    # Speed is returned as a numpy array of positive scalars
    velocity = f5['agent/velocity']
    speed = np.linalg.norm(velocity, axis = 1)
    speed = np.append(speed, 0)
    return positions, speed

def getTrajValues(file):
    f = np.load(file)
    positions = f['positions']
    velocities = f['vels']
    speeds = np.linalg.norm(velocities, axis = 1)
    speeds = np.append(speeds, 0)
    boundaries = f['boundaries']
    boundary_vectors = f['boundary_distances']
    return positions, speeds, boundaries, boundary_vectors

def BVC_act(BVCs, boundary_vectors, Nbvcs, noise_level = 0.01):
    dists = BVCs[1, :, np.newaxis] - boundary_vectors[:, np.newaxis, :]
    thetas = np.expand_dims((BVCs[0, :, np.newaxis] - np.arange(180))/180, 0)
    activity = np.sum(np.exp(dists**2/1+thetas**2/1), axis = 2)
    activity = np.clip(activity/np.expand_dims(np.max(activity, axis = 1), 1) + (np.random.rand(len(boundary_vectors), Nbvcs) - 0.5)*noise_level, 0, 1)
    return activity

def gauss(D, sig):
    return 1 / np.sqrt(2 * np.pi * sig) * np.exp(- (D)**2 / (2 * sig**2))


def gaussNonnorm(D, sig):
    return np.exp(- (D)**2 / (2 * sig**2))

def generateSpikeTrainFromGaussian(A):
    S = np.empty(0)
    n_spikes = np.random.poisson(A*10)
    times = np.random.rand(n_spikes)*10
    S = np.append(S, times)
    return S


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

def gridness_score(R,Ndendrites, sigma):
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
    
    def spike_act(self, X):
        A = self.act(X)
        S = generateSpikeTrainFromGaussian(A)

def getBVCtoDendriteConnectivity(n_bvcs, n_dendrites2, distribution = 'uniform', rate = 0.1, bvc_params = [12, 11], verbose = False):
    # Return two lists, one of dendrite number, the other of bvc number.
    bvc_range = np.arange(n_bvcs)
    bvc_range = np.reshape(np.tile(bvc_range, n_dendrites2), (n_dendrites2, n_bvcs))
    if distribution == 'uniform': #each dendrite gets input from a random number of BVCs indicated by 'rate'
        temp_connections = np.random.rand(n_dendrites2, n_bvcs)
    if distribution=='orthogonal': #each dendrite gets input from two BVCs, which align with the x-y-axis
        temp_connections = np.random.randint(0, bvc_params[1], (n_dendrites2 * 2))*bvc_params[0]
        temp_connections += np.tile(np.arange(2), n_dendrites2)*(bvc_params[0]//4) + np.random.randint(0,2, n_dendrites2*2)*bvc_params[0]//2
        indices = np.repeat(np.arange(n_dendrites2), 2)
        connections = np.array([indices, temp_connections])
    if verbose:
        print(connections)
    return connections


#getBVCtoDendriteConnectivity(132, 15, distribution = 'orthogonal', verbose = True)