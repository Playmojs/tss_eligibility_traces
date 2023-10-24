import numpy as np

class Symbol():
    def __init__(self, coord, base_delay, variance):
        self.coord = coord
        self.activated = False
        self.activated_at = None
        self.inhibit_window = [np.inf, np.inf]
        self.inhibit_trace = False
        self.feedback_window = [np.inf, np.inf]
        self.variance = variance
        self.spike_delay_ms = base_delay + np.random.rand()*self.variance
        self.original_spike_delay_ms = self.spike_delay_ms
        self.tag = False
        self.tagable = False
        self.has_sped_up = False
        self.layer = np.inf
    def reset(self, base_delay):
        self.tag = False
        self.spike_delay_ms = base_delay + np.random.rand()*self.variance
        self.original_spike_delay_ms = self.spike_delay_ms
        self.activated_at = None

        

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
          self.spike_ids = np.empty(0, dtype=int)
          self.receive_input_ids = np.empty(0, dtype = int)
          self.global_inhibition = False
          self.catch_frame = False


class EventSeries():
     def __init_(self):
          self.scheduled_events = {}