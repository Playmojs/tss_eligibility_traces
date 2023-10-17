import numpy as np
import modules

def findAllTransitions(symbols, dist):
    transitions = []
    for id in range(len(symbols)):
        transitions.append(modules.Transition(symbols, id, dist))
    return(transitions)

def spawnSymbol(coord, base_delay, variance):
    return modules.Symbol(coord, base_delay, variance)

def generateRandomSymbols(n_symbols, base_delay, x_range, y_range, variance):
    symbols = []
    for i in range(0, n_symbols):
        coord = np.array([np.random.uniform(x_range[0], x_range[1],1)[0], np.random.uniform(y_range[0], y_range[1],1)[0]])
        symbols.append(spawnSymbol(coord, base_delay, variance))
    return symbols