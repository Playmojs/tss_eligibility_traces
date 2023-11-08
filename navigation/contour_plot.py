import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import nav_dijkstra as nav

def make_contour(coords, activation_times, ax, vmin, vmax, nlevels):
    x = coords[:,0]
    y = coords[:,1]
    triang = tri.Triangulation(x, y)
    refiner = tri.UniformTriRefiner(triang)
    tri_refi, z_test_refi = refiner.refine_field(activation_times, subdiv=5)

    ax.set_xlim((0,10))
    ax.set_ylim((0,10))
    ax.set_aspect('equal')
    ax.tricontourf(tri_refi, z_test_refi, nlevels, vmin = vmin, vmax = vmax, cmap = 'GnBu')
    ax.tricontour(tri_refi, z_test_refi, nlevels, vmin = vmin, vmax = vmax,
              colors='0.25',
              linewidths=1)


with np.load("navigation/result_data/400_1ms_2.npz", allow_pickle=True) as data:
    inhibit_range= data['inhibit_ranges']
    symbols = data['symbols']
    start = data['start']
    goal = data['goal']
    distance = data['dist']

symbols = symbols.item()

keys = list(symbols)
n_keys = len(keys)
fig, ax = plt.subplots(3, 5)
colors = ['black', 'red', 'orange']

for plot in range(n_keys-1):
    coords = np.empty((len(symbols[0]), 2))
    act_times = np.empty(len(symbols[0]))
    color_indices = np.empty(len(symbols[0]), dtype = int)
    alphas = np.empty(len(symbols[0]))
    for i, symbol in enumerate(symbols[keys[plot+1]]):
        coords[i] = symbol.coord
        act_times[i] = symbol.activated_at
        color_indices[i] = 0 if i in [start, goal] else 1 if symbol.tag else 2
        alphas[i] = 1 if i in[start, goal] else 0.7 if symbol.tag else 0.3
    all_coords = coords.copy()
    dist, prev = nav.dijkstra(symbols[keys[plot]], start, goal, distance, complete = True, time_cost= True)
    if plot == 0:
        vmax = max(dist)
        print(vmax)
    ax_x = ax[plot//5, plot%5]

    n_levels = int((n_keys)//(vmax/max(dist)))
    make_contour(all_coords, dist, ax_x, 0, vmax, n_levels)

    mask = np.invert(np.bitwise_or(np.isnan(act_times), act_times < keys[plot]))
    coords = coords[mask]
    act_times = act_times[mask]
    color_indices = color_indices[mask]
    print(max(act_times))
    alphas = alphas[mask]
    ax_x.scatter(coords[:,0], coords[:,1], alpha = alphas, color = np.array(colors)[color_indices])
    ax_x.set_title("%3.0f ms" % (max(act_times)-min(act_times)))

plt.show()