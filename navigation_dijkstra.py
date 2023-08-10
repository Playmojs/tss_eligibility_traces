#%%

import numpy as np
import matplotlib.pyplot as plt



#%%
#Create random nodes:

class Symbol():
    def __init__(self, coord, nscales=1):
        self.coord = coord

def spawnSymbol(coord, nscales):
    return Symbol(coord, nscales)

def generateRandomSymbols(n_symbols, x_range, y_range, nscales):
    symbols = []
    
    for i in range(0, n_symbols):
        coord = np.array([np.random.uniform(x_range[0], x_range[1],1), np.random.uniform(y_range[0], y_range[1],1)])
        symbols.append(spawnSymbol(coord, nscales))
    return symbols

#%%
#Generate symbols

symbols = generateRandomSymbols(100, [0, 10], [0,10], 1)

plt.plot([symbol.coord[0] for symbol in symbols], [symbol.coord[1] for symbol in symbols], 'o')


# %%
#Select start and target symbols:

[start, goal] = np.random.choice(len(symbols), 2, False)
print(start, goal)
print (symbols[start].coord, symbols[goal].coord)


# %%

def dijkstra(symbols, start, goal, max_dist):
    n_vertices = len(symbols)
    
    dist = np.inf * np.ones(n_vertices)
    prev = [np.nan] * n_vertices
    Q = np.arange(n_vertices)
    dist[start] = 0

    while len(Q) != 0:
        print(dist)
        u = Q[dist[Q].argmin()]
        print(u)
        if u == goal:
            print("Done")
            return dist, prev
        
        Q = np.delete(Q, np.searchsorted(Q , u))
        print(Q)

        for vertice in Q:
            distance = np.linalg.norm(symbols[u].coord - symbols[vertice].coord)
            if distance > max_dist:
                continue
            alt = distance + dist[u]
            print(distance)
            if alt < dist[vertice]:
                dist[vertice] = alt
                prev[vertice] = (u)

    return False


def back_track_dijkstra(prev, goal):
    sequence = []
    u = goal
    if prev[u] != np.nan or u == goal:
        while not np.isnan(u):
            print(u)
            sequence.append(u)
            u = prev[u]
    return sequence 

# %%

dist, prev = dijkstra(symbols, start, goal, 1)

# %%

path = back_track_dijkstra(prev, goal)

print(path)
# %%
