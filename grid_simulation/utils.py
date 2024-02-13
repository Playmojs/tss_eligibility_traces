import numpy as np
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
import matplotlib.path as mpath
import scipy.signal as sig
import sys
import os
import re

def getSortedEntries(directory, file_type=None, full_path = True):
    entries = os.listdir(directory)

    #GPT based, sorry :P
    # To get both file types, let file_type == None
    if file_type == 'npz':
        entries = [entry for entry in entries if entry.endswith('.npz')]
    elif file_type == 'directory':
        entries = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
    
    # Define a regular expression pattern to extract the prefix and numeric part
    pattern = re.compile(r'([a-zA-Z]+)(\d+)')
    
    # Sort the entries based on the prefix and numeric part
    entries.sort(key=lambda x: (pattern.match(x).group(1), int(pattern.match(x).group(2))) if pattern.match(x) else (x,))

    if full_path:
        entries = [os.path.join(directory, entry) for entry in entries]

    return entries

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

def BVC_act(BVCs, boundary_vectors, Nbvcs, sigma, noise_level = 0.01, alg = 'full'):
    if alg == 'simple':
        activity = np.abs(BVCs[1] - boundary_vectors[:,np.ndarray.astype(BVCs[0], int)] + (np.random.rand(len(boundary_vectors), Nbvcs) -0.5)*noise_level)
        delays = activity / (2*sigma) * 20
    
    elif alg == 'full':
        dists = BVCs[1, :, np.newaxis] - boundary_vectors[:, np.newaxis, :]
        temp_thetas = (BVCs[0, :, np.newaxis] - np.arange(180)) % 180
        temp_thetas[temp_thetas > 90] -= 180
        thetas = np.expand_dims(temp_thetas/90, 0)
        activity = np.sum(np.exp(-(dists**2/1+thetas**2/1)), axis = 2)
        activity = np.clip(activity/np.expand_dims(np.max(activity, axis = 1), 1) + (np.random.rand(len(boundary_vectors), Nbvcs) - 0.5)*noise_level, 0, 1)
        delays = (1/activity - 1) # Shape (Nsamples, Nbvcs)
        delays = delays / (np.max(delays, axis = 1, keepdims= True) * 2 * sigma) * 20
    return delays

def getBoundaryVectorsFromShape(shape: str):
    match shape:
        case 'square':
            boundary_vecs = [[-0.5,-0.5], [0.5,-0.5], [0.5,0.5], [-0.5, 0.5]]
            x_min, x_max, y_min, y_max = -0.5, 0.5, -0.5, 0.5
        case 'circular':
            boundary_vecs = [[0.5*np.cos(t),0.5*np.sin(t)] for t in np.linspace(0,2*np.pi,100)]
            x_min, x_max, y_min, y_max = -0.5, 0.5, -0.5, 0.5
        case 'trapezoid':
            boundary_vecs = [[0, -0.2], [0, 0.2], [1.5, 0.5], [1.5, -0.5]]
            x_min, x_max, y_min, y_max = 0, 1.5, -0.5, 0.5
        case _:
            raise Exception("Invalid boundary shape")
    return boundary_vecs

def maskArea(pxs, boundary_vecs, x_range, y_range):
    path = mpath.Path(boundary_vecs)
    x_min, x_max = x_range[0], x_range[1]
    y_min, y_max = y_range[0], y_range[1]
    points = np.reshape(np.meshgrid(np.linspace(x_min, x_max, int(pxs*(x_max-x_min))), np.linspace(y_min, y_max, int(pxs*(y_max-y_min)))), (2,-1)).T
    mask = mpath.Path.contains_points(path, points, radius = 1/pxs)
    mask = mask[:, np.newaxis]
    return mask

def createHexField(pxs, sigma, wall_angle_offset, shape = 'square'):
    x_vals, y_vals = np.meshgrid(
        np.arange(-1000, 2000, sigma *1000 * np.sqrt(3) / 2),
        np.arange(-1000, 2000, sigma*1000)
    )
    
    x_vals[1::2] += sigma * 500 * np.sqrt(3) / 2  # Shift every other row
    
    # Flatten the grid and apply rotation
    x_vals_flat = x_vals.flatten() + np.random.uniform(0, sigma * 500)
    y_vals_flat = y_vals.flatten() + np.random.uniform(0, sigma * 500)
    rotated_x = (x_vals_flat * np.cos(wall_angle_offset) - y_vals_flat * np.sin(wall_angle_offset))
    rotated_y = (x_vals_flat * np.sin(wall_angle_offset) + y_vals_flat * np.cos(wall_angle_offset))
    indices = np.logical_and(np.logical_and(rotated_x > 0, rotated_x <1000), np.logical_and(rotated_y>0, rotated_y < 1000))
    rotated_x = np.ndarray.astype(rotated_x[indices], int)
    rotated_y = np.ndarray.astype(rotated_y[indices], int)

    grid_mesh = np.zeros((1000,1000))
    grid_mesh[rotated_x, rotated_y] = 1
    grid_mesh = ndimage.gaussian_filter(grid_mesh, sigma*100)

    # Define the interpolation function
    interp_function = interpolate.interp2d(np.arange(1000), np.arange(1000), grid_mesh, kind='linear')

    # Define the new grid for the nxn array
    new_x = np.linspace(0, 999, pxs)
    new_y = np.linspace(0, 999, pxs)

    # Perform the interpolation
    reduced_grid_mesh = interp_function(new_x, new_y)

    area_mask = maskArea(pxs, getBoundaryVectorsFromShape(shape), [-0.5,0.5], [-0.5,0.5])

    reduced_grid_mesh = reduced_grid_mesh *np.reshape(area_mask, (pxs, pxs))

    return reduced_grid_mesh



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
    T = sig.correlate2d(R, S, mode='full') 
    norm_factor = (N * np.std(A) * np.std(B))
    return T / norm_factor

def autoCorr(A):
    """Normalized autocorrelation of arrayvector A, with shape (..., X, X)"""
    X = A.shape[-1]
    mean_A = np.mean(A, axis=(-2, -1), keepdims=True)
    std_A = np.std(A, axis=(-2, -1), keepdims=True)

    R = A - mean_A
    autocorr = sig.fftconvolve(R, R[..., ::-1, ::-1], axes = (-2, -1))
    norm_factor = X * X * std_A ** 2

    return autocorr/norm_factor

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

        corrot[idrot] = (nx * ny * np.sum(dotAB) - np.sum(dotA0) * np.sum(dotB0)) / (np.sqrt(nx * ny * np.sum(dotAA) - np.sum(dotA0)**2)*np.sqrt(nx*ny*np.sum(dotBB)) - np.sum(dotB0)**2)

    gridscore = min(corrot[59],corrot[119]) - max(corrot[29], corrot[89], corrot[149])
    return gridscore, Tmp

def gridnessScore(R, Ndendrites, sigma):
    """
    Compute the Gridness Score of autocorrelated rate maps R.
    
    Args:
    - R: Input array of autocorrelation matrices of shape (some_shape, X, X).
    - Ndendrites: Scalar number representing dendrites.
    - sigma: Scalar number representing sigma.

    Returns:
    - gridscore: Array of grid scores with shape (some_shape).
    - Tmp: Array of corresponding temporary results.
    """
    # Compute the dimensions
    dim0 = R.shape[-2]
    cntr = dim0 // 2

    # Create the ring filter
    in_ra = int(2 * sigma * Ndendrites)
    out_ra = int(4 * sigma * Ndendrites)
    RingFilt = np.zeros((dim0, dim0))
    for i in range(dim0):
        for j in range(dim0):
            cntr_i = (cntr - i) ** 2
            cntr_j = (cntr - j) ** 2
            dist = cntr_i + cntr_j
            if in_ra ** 2 <= dist <= out_ra ** 2:
                RingFilt[i, j] = 1

    # Apply the filter
    Tmp = R * RingFilt[np.newaxis, ...]
    Tmp = Tmp[..., cntr - out_ra - 1:cntr + out_ra + 1, cntr - out_ra - 1: cntr + out_ra + 1]

    # Initialize arrays for storing grid scores
    gridscore = np.zeros(R.shape[:-2])

    angles = [29, 59, 89, 119, 149]
    corrot = np.tile(np.zeros(R.shape[:-2])[..., np.newaxis], len(angles))
    nx, ny = Tmp.shape[-2], Tmp.shape[-1]
    for i, idrot in enumerate(angles):
        rot = ndimage.rotate(Tmp, idrot, axes=(-2, -1), reshape=False)

        dotAA = np.sum(Tmp * Tmp, axis=(-2, -1))
        dotA0 = np.sum(Tmp * np.ones_like(Tmp), axis=(-2, -1))
        dotBB = np.sum(rot * rot, axis=(-2, -1))
        dotB0 = np.sum(rot * np.ones_like(rot), axis=(-2, -1))
        dotAB = np.sum(Tmp * rot, axis=(-2, -1))

        temp = (nx * ny * dotAB - dotA0 * dotB0) / (
                    np.sqrt(nx * ny * dotAA - dotA0 ** 2) * np.sqrt(
                nx * ny * dotBB) - dotB0 ** 2)
        corrot[..., i] = temp
    p = np.array([1, 3], dtype = int)
    d = np.array([0, 2, 4], dtype = int)

    gridscore = np.min(corrot[..., p], axis = -1) - np.max(corrot[..., d], axis = -1)

    return gridscore, Tmp


def genWhiteNoiseCoords(N, xlim, ylim):
    return np.array([np.random.uniform(xlim[0], xlim[1], N), np.random.uniform(ylim[0], ylim[1], N)]).T

def genBlueNoiseCoords(N, xlim = [0,1], ylim = [0,1]):
    coords = np.zeros((N,2))
    coords[0:6] = genWhiteNoiseCoords(1, xlim, ylim)
    m = 1
    for i in range(1, N):
        print(f"\rBlue noise coords: {i}/{N}", end = "")
        coord_opts = genWhiteNoiseCoords(i*m, xlim, ylim)
        thetas = np.linalg.norm(coords[:i, np.newaxis, :] - coord_opts, axis = 2)
        ind = np.argmax(np.min(thetas, 0))
        coords[i] = coord_opts[ind]
    sys.stdout.write("\n")
    return(coords)

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

class CoordinateSamplers():
    def __init__(self, N, tuning_width, xlim=np.array([0,1]), ylim=np.array([0,1]), distrib = 'regular', noise_level = 0, Xs = [0,0]):
        self.N = N
        self.N2 = N**2
        self.tuning_width = tuning_width
        self.xlim = xlim
        self.ylim = ylim
        self.samplers = np.zeros((N,N))

        match distrib:
            case 'regular':
                x = np.linspace(xlim[0], xlim[1], N)
                y = np.linspace(ylim[0], ylim[1], N)
                Xs = np.meshgrid(x, y, indexing='xy')
                self.Xs = np.reshape(Xs, (2,-1)).T
        
            case 'noisy_regular':
                x = np.linspace(xlim[0], xlim[1], N)
                y = np.linspace(ylim[0], ylim[1], N)
                Xs = np.reshape(np.meshgrid(x, y, indexing='xy'), (2,-1)).T
                Xs += np.random.normal(0, 1/(10*N**2), Xs.shape)
                Xs[:,0] = np.clip(Xs[:,0], 0, 1)
                Xs[:,1] = np.clip(Xs[:,1], 0, 1)
                self.Xs = Xs
            case 'noisy_blue':
                Xs = genBlueNoiseCoords(N**2, xlim, ylim)
                self.Xs = Xs
            case 'noisy_white':
                Xs = genWhiteNoiseCoords(N**2, xlim, ylim)
                ind = np.lexsort((Xs[:,0], Xs[:,1]))
                self.Xs = Xs[ind]
            case 'premade':
                assert (len(Xs) == self.N2)
                self.Xs = Xs
            case _:
                assert(False)
            

    def dist(self, X):
        return np.sqrt(np.sum((self.Xs[np.newaxis, ...] - X[:, np.newaxis, :])**2, axis=2))

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

def getBVCtoDendriteConnectivity(n_bvcs, n_dendrites2, bvc_params = [12, 11], distribution = 'uniform', rate = 0.1, verbose = False):
    # Return two lists, one of dendrite number, the other of bvc number.
    bvc_range = np.arange(n_bvcs)
    bvc_range = np.reshape(np.tile(bvc_range, n_dendrites2), (n_dendrites2, n_bvcs))
    if distribution == 'uniform': # each dendrite gets input from a random number of BVCs indicated by 'rate'
        temp_connections = np.random.rand(n_dendrites2, n_bvcs)
    if distribution=='orthogonal': # each dendrite gets input from two BVCs, which align with the x-y-axis
        temp_connections = np.random.randint(0, bvc_params[1], (n_dendrites2 * 2))*bvc_params[0]
        temp_connections += np.tile(np.arange(2), n_dendrites2)*(bvc_params[0]//4) + np.random.randint(0,2, n_dendrites2*2)*bvc_params[0]//2
        indices = np.repeat(np.arange(n_dendrites2), 2)
        connections = np.array([temp_connections, indices])
    if distribution == 'orthoregular': # same as orthogonal, but carefully paired so that each dendrite will respond best to some x-y-position in a square environment
        temp_connections = np.empty(2*n_dendrites2)
        thetas = bvc_params[0]
        dist = bvc_params[1]
        ng = n_dendrites2 // (2*dist)**2
        base = np.arange(0, n_bvcs, 4)
        # horiz = np.concatenate((base+thetas, 4*thetas - 1 - base))
        # vert = np.concatenate((base, 3*thetas - 1 - base))
        horiz = np.concatenate((base + 2, n_bvcs - 4 - base))
        vert = np.concatenate((base + 3, n_bvcs - 3 - base ))

        temp_connections[::2] = np.tile(np.repeat(vert, 2*dist), ng)
        temp_connections[1::2] = np.tile(horiz, ng * 2* dist)
        
        indices = np.repeat(np.arange(n_dendrites2), 2)
        connections = np.array([temp_connections, indices], dtype = int)
    if verbose:
        print(connections)
    return connections


#getBVCtoDendriteConnectivity(132, 15, distribution = 'orthogonal', verbose = True)