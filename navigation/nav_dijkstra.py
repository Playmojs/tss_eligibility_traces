import numpy as np
import matplotlib.pyplot as plt
import module_utils
            
def dijkstra(symbols, start, goal, max_dist, complete = False, time_cost = False):
    n_vertices = len(symbols)

    dist = np.inf * np.ones(n_vertices)
    prev = [np.nan] * n_vertices
    Q = np.arange(n_vertices)
    dist[start] = 0

    while len(Q) != 0:
        u = Q[dist[Q].argmin()]
        if np.isinf(dist[u]):
            raise Exception("No complete path found with maximum distance at ", max_dist)
        if u == goal and not complete:
            print("Done")
            return dist, prev

        Q = np.delete(Q, np.searchsorted(Q , u))

        for vertice in Q:
            distance = np.linalg.norm(symbols[u].coord - symbols[vertice].coord)
            if distance > max_dist:
                continue
            if time_cost:
                alt = symbols[vertice].spike_delay_ms + dist[u]
            else:
                alt = distance + dist[u]
            if alt < dist[vertice]:
                dist[vertice] = alt
                prev[vertice] = u

    return dist, prev

def back_track_dijkstra(prev, goal):
    sequence = []
    u = goal
    if prev[u] != np.nan or u == goal:
        while not np.isnan(u):
            sequence.append(u)
            u = prev[u]
    return sequence

