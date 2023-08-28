import numpy as np
from matplotlib import pyplot as plt
import module_utils


symbols = module_utils.generateRandomSymbols(100, [0,10], [0,10], 1)
plt.plot([symbol.coord[0] for symbol in symbols], [symbol.coord[1] for symbol in symbols], 'o')
transitions = module_utils.findAllTransitions(symbols, 2)


start_neuron = 4
wanted_transitions = transitions[start_neuron].transitions.astype(int)
print(wanted_transitions)
plt.plot([symbol.coord[0] for symbol in symbols], [symbol.coord[1] for symbol in symbols], 'o')
plt.plot([symbols[i].coord[0] for i in wanted_transitions], [symbols[i].coord[1] for i in wanted_transitions], 'ro')
plt.plot(symbols[start_neuron].coord[0], symbols[start_neuron].coord[1], 'go')


def dijkstra_with_transition(symbols, start_id, goal_id, range):
    n_symbols = len(symbols)
    valid_vertices = module_utils.findAllTransitions(symbols, range)

    available_symbols = np.ones(n_symbols, dtype = bool)
    dist = np.inf * np.ones(n_symbols)
    prev = [np.nan] * n_symbols
    dist[start_id] = 0

    while np.any(available_symbols):
        min_dist = dist[available_symbols].min()
        u = np.where(dist == min_dist)[0][0]
        if np.isinf(dist[u]):
            raise Exception("No complete path found with maximum distance at ", range)
        if u == goal_id:
            print("Done")
            return dist, prev
        
        available_symbols[u] = 0

        for neighbor in valid_vertices[u].transitions:
            neighbor = int(neighbor)
            distance = np.linalg.norm(symbols[u].coord - symbols[neighbor].coord)
            alt = distance + dist[u]
            if alt < dist[neighbor]:
                dist[neighbor] = alt
                prev[neighbor] = u

    return False


def back_track_dijkstra(prev, goal):
    sequence = []
    u = goal
    if prev[u] != np.nan or u == goal:
        while not np.isnan(u):
            sequence.append(u)
            u = prev[u]
    return sequence


#Pick random start and goal from symbols:

[start, goal] = np.random.choice(len(symbols), 2, False)
print("From:", start, "to", goal)
print ("at ", symbols[start].coord, "and", symbols[goal].coord)

#Find path
dist, prev = dijkstra_with_transition(symbols, start, goal, 1.7)
nodes = back_track_dijkstra(prev, goal)
print("These nodes were visited:", nodes)


#Plot path
plt.plot([symbol.coord[0] for symbol in symbols], [symbol.coord[1] for symbol in symbols], 'o')
plt.plot([symbols[i].coord[0] for i in nodes], [symbols[i].coord[1] for i in nodes], 'ro')
plt.show()

