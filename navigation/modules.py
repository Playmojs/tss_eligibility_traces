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
        self.no_tag = False
        self.spike_delay_ms = self.base_delay + np.random.rand()*self.variance
        self.original_spike_delay_ms = self.spike_delay_ms
        self.activated_at = np.full(self.nscales, np.nan)
        self.inhibit_window = np.ones((self.nscales, 2)) * np.inf
        self.inhibit_trace = np.zeros(self.nscales)
        self.feedback_window = np.ones((self.nscales, 2)) * np.inf
        if compressed:
             self.inhibit_window = np.ndarray.flatten(self.inhibit_window)
             self.feedback_window = np.ndarray.flatten(self.feedback_window)
             self.activated_at = np.nan
             self.inhibit_trace = 0
             self.tag = False

class Symbols():
    def __init__(self, nsymbols, xrange, yrange, base_delay, variance, nscales = 1):
          self.coords = np.hstack((np.random.uniform(xrange[0], xrange[1], (nsymbols,1)), np.random.uniform(yrange[0], yrange[1], (nsymbols,1))))
          self.start = np.zeros(nsymbols, dtype = bool)
          self.goal = np.zeros(nsymbols, dtype = bool)
          self.base_delay = base_delay
          self.variance = variance
          self.nscales = nscales
          self.nsymbols = nsymbols
          self.ids = np.arange(nsymbols, dtype = int)
          self.reset()
    def reset(self):
         nsymbols = self.nsymbols
         nscales = self.nscales
         self.tag = np.zeros((nsymbols, nscales), dtype = bool)
         self.no_tag = np.zeros(nsymbols, dtype = bool)
         self.spike_delay_ms = np.random.rand(nsymbols) * self.variance + self.base_delay
         self.original_spike_delay_ms = self.spike_delay_ms
         self.activated_at = np.full((nsymbols, nscales), np.nan)
         self.inhibit_window = np.full((nsymbols, nscales, 2), np.inf)
         self.inhibit_trace = np.zeros((nsymbols, nscales))
         self.feedback_window = np.full((nsymbols, nscales, 2), np.inf)


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

class Transitions():
    '''The transitions object holds a boolean mask of shape (nscales, nsymbols, nsymbols), 
    and is True for index [i, j, k] if symbol j can transition to symbol k at scale i, and False otherwise'''
    
    
    def __init__(self, symbols, base_range, nscales = 1):
        ranges = np.logspace(0, nscales - 1, nscales, base = np.sqrt(2)) * base_range
        ranges = np.insert(ranges, 0,0)
        self.ranges = ranges
        self.get_transitions(symbols)
    def get_transitions(self, symbols):
        points = symbols.coords
        nsymbols = symbols.nsymbols
        nscales = symbols.nscales
        pairwise_distances = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
        ranges_broadcasted = np.broadcast_to(np.array(self.ranges)[:, np.newaxis, np.newaxis], (nscales + 1, nsymbols, nsymbols))
        transition_mask = np.logical_and(pairwise_distances > ranges_broadcasted[:-1], pairwise_distances <= ranges_broadcasted[1:])
        self.transition_mask = transition_mask

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