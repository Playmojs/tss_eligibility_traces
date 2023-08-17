#%%
import numpy as np
import matplotlib.pyplot as plt
import utils
            
# %%

def dijkstra(symbols, start, goal, max_dist):
    n_vertices = len(symbols)

    dist = np.inf * np.ones(n_vertices)
    prev = [np.nan] * n_vertices
    Q = np.arange(n_vertices)
    dist[start] = 0

    while len(Q) != 0:
        u = Q[dist[Q].argmin()]
        if np.isinf(dist[u]):
            raise Exception("No complete path found with maximum distance at ", max_dist)
        if u == goal:
            print("Done")
            return dist, prev

        Q = np.delete(Q, np.searchsorted(Q , u))

        for vertice in Q:
            distance = np.linalg.norm(symbols[u].coord - symbols[vertice].coord)
            if distance > max_dist:
                continue
            alt = distance + dist[u]
            if alt < dist[vertice]:
                dist[vertice] = alt
                prev[vertice] = u

    return False


def back_track_dijkstra(prev, goal):
    sequence = []
    u = goal
    if prev[u] != np.nan or u == goal:
        while not np.isnan(u):
            sequence.append(u)
            u = prev[u]
    return sequence

#%%
#Generate symbols

symbols = utils.generateRandomSymbols(100, [0, 10], [0,10], 1)

plt.plot([symbol.coord[0] for symbol in symbols], [symbol.coord[1] for symbol in symbols], 'o')

# %%
#Pick random start and goal from symbols:

[start, goal] = np.random.choice(len(symbols), 2, False)
print("From:", start, "to", goal)
print ("at ", symbols[start].coord, "and", symbols[goal].coord)
#%%
#Find path
dist, prev = dijkstra(symbols, start, goal, 1.435)
nodes = back_track_dijkstra(prev, goal)
print("These nodes were visited:", nodes)


# %%
#Plot path
plt.plot([symbol.coord[0] for symbol in symbols], [symbol.coord[1] for symbol in symbols], 'o')
plt.plot([symbols[i].coord[0] for i in nodes], [symbols[i].coord[1] for i in nodes], 'ro')
plt.show()

# %%
