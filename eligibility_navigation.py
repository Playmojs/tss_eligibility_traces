import numpy as np
from matplotlib import pyplot as plt
import utils
import modules

#%%
def control_schedule_time(event_series, time):
     if time not in event_series:
        event_series[time] = modules.TimedEvent()

def schedule_activate(event_series, time, transition_ids):
    control_schedule_time(event_series, time)
    event_series[time].try_activate = np.append(event_series[time].try_activate, transition_ids)

#%%
def eligibility_navigation(symbols, start, goal, distance):
    valid_transitions = utils.findAllTransitions(symbols,distance)
    current_time = 0
    goal_found = False
    event_series = {}
    
    symbols[start].activated = True
    schedule_activate(event_series, symbols[start].spike_delay_ms, valid_transitions[start].transition_ids)
    
    while not goal_found:
        current_time = min(event_series)
        print(current_time)
        print(event_series[current_time].try_activate)
        for activate_id in event_series[current_time].try_activate:
            if ~symbols[activate_id].activated:
                if activate_id == goal:
                    goal_found = True
                symbols[activate_id].activated = True
                schedule_activate(event_series, 
                                current_time + symbols[activate_id].spike_delay_ms, 
                                valid_transitions[activate_id].transition_ids)
        if current_time > 100:
            raise Exception("Too much time passed, out of bounds")
        del event_series[current_time]
    print("The goal was found after", current_time, "ms!")
    return

#%%
symbols = utils.generateRandomSymbols(100, [0,10], [0,10], 1)
plt.plot([symbol.coord[0] for symbol in symbols], [symbol.coord[1] for symbol in symbols], 'o')
transitions = utils.findAllTransitions(symbols, 2)


#%%
#Pick random start and goal from symbols:

[start, goal] = np.random.choice(len(symbols), 2, False)
print("From:", start, "to", goal)
print ("at ", symbols[start].coord, "and", symbols[goal].coord)

# %%

eligibility_navigation(symbols, start, goal, 2)
# %%
