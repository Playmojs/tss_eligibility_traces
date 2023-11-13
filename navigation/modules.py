import numpy as np

class Symbol():
    def __init__(self, coord, base_delay, variance, nscales = 1, compressed = True):
        self.coord = coord
        self.nscales = nscales
        self.base_delay = base_delay
        self.activated = False
        self.variance = variance
        self.tagable = False
        self.has_sped_up = False
        self.reset(0, compressed)
    def reset(self, base_delay, compressed = True):
        self.tag = np.zeros(self.nscales, dtype = bool)
        self.spike_delay_ms = self.base_delay + np.random.rand()*self.variance
        self.original_spike_delay_ms = self.spike_delay_ms
        self.activated_at = np.full(self.nscales, np.nan)
        self.inhibit_window = np.ones((self.nscales, 2)) * np.inf
        self.inhibit_trace = np.zeros(self.nscales)
        self.feedback_window = np.ones((self.nscales, 2)) * np.inf
        if compressed:
             self.inhibit_window = np.ndarray.flatten(self.inhibit_window)
             self.feedback_window = np.ndarray.flatten(self.feedback_window)
             self.activated_at = None
             self.inhibit_trace = 0
             self.tag = False

def getAvailableTransitions(symbols, self_id, range):
        transitions = np.empty(0,dtype = int)
        self_coord = symbols[self_id].coord
        for id, symbol in enumerate(symbols):
            if  id == self_id:
                continue
            if range[0] < np.linalg.norm(symbol.coord-self_coord) < range[1]:
                transitions = np.append(transitions, int(id))
        return transitions

class Transition():
    def __init__(self, symbols, id, range):
        self.transition_ids = getAvailableTransitions(symbols, id, range)
    
class TimedEvent():
     def __init__(self):
          self.try_activate = np.empty(0, dtype = int)
          self.spike_ids = np.empty(0, dtype = int)
          self.receive_input_ids = np.empty(0, dtype = int)
          self.spike_events = np.empty((0,2), dtype = int)
          self.synapse_events = np.empty((0,2), dtype = int)
          self.global_inhibition = False
          self.catch_frame = False


class EventSeries():
     def __init_(self):
          self.scheduled_events = {}