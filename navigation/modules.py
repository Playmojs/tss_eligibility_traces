import numpy as np

class Symbol():
    def __init__(self, coord, nscales=1):
        self.coord = coord
        self.activated = False
        self.activated_at = None
        self.spike_delay_ms = 8 + np.random.rand()*4
        self.tag = False
        self.layer = np.inf
    def reset(self):
        self.tag = False
        self.spike_delay_ms = 8 + np.random.rand()*4

        

def getAvailableTransitions(symbols, self_id, range):
        transitions = np.empty(0,dtype = int)
        self_coord = symbols[self_id].coord
        for id, symbol in enumerate(symbols):
            if  id == self_id:
                continue
            if np.linalg.norm(symbol.coord-self_coord) < range:
                transitions = np.append(transitions, int(id))
        return transitions

class Transition():
    def __init__(self, symbols, id, range):
        self.transition_ids = getAvailableTransitions(symbols, id, range)
    
class TimedEvent():
     def __init__(self):
          self.try_activate = np.empty(0, dtype = int)
          self.global_inhibition = False
          self.catch_frame = False

class EventSeries():
     def __init_(self):
          self.scheduled_events = {}