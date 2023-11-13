import numpy as np
import modules

def findAllTransitions(symbols, base_dist, nscales: int = 1, compressed = True):
    assert(nscales > 0)
    transitions = np.empty((len(symbols),nscales), dtype = object)
    ranges = np.logspace(0, nscales - 1, nscales, base = np.sqrt(2)) * base_dist
    ranges = np.insert(ranges, 0,0)
    for id in range(len(symbols)):
        for i in range(nscales):
            transitions[id, i] = (modules.Transition(symbols, id, [ranges[i], ranges[i+1]]))
    if compressed:
        transitions = np.ndarray.flatten(transitions)
    return(transitions)

def spawnSymbol(coord, base_delay, variance, nscales, compressed):
    return modules.Symbol(coord, base_delay, variance, nscales, compressed)

def generateRandomSymbols(n_symbols, base_delay, x_range, y_range, variance, nscales = 1, compressed = False):
    symbols = []
    for i in range(0, n_symbols):
        coord = np.array([np.random.uniform(x_range[0], x_range[1],1)[0], np.random.uniform(y_range[0], y_range[1],1)[0]])
        symbols.append(spawnSymbol(coord, base_delay, variance, nscales, compressed))
    return symbols