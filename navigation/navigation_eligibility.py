#%%
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import module_utils
import simulation_utils as sim_utils
import recorder as rec

def eligibilityNavigation(symbols, start, goal, distance, ms_per_frame):
    valid_transitions = module_utils.findAllTransitions(symbols,distance) #Get a list containing, for each symbol, all their valid transitions
    current_time = 0
    goal_found = False
    event_series = {}
    recorder = rec.Recorder()

    #reset symbols
    for symbol in symbols:
        symbol.tag = False
        symbol.spike_delay_ms = 10
    
    symbols[goal].tag = True
    
    for i in range(10):
        for frame in range(0,100, ms_per_frame):
            sim_utils.scheduleFrame(event_series, current_time + frame)

        for symbol in symbols:
            symbol.activated = False
            symbol.activated_at = None

        symbols[start].activated = True
        symbols[start].activated_at = current_time
        goal_found = False
        final_time = np.inf

        sim_utils.scheduleActivate(event_series, current_time + symbols[start].spike_delay_ms, valid_transitions[start].transition_ids)

        while final_time + 15 > current_time: #keeps running until 15 ms after goal is found, or exception
            current_time = min(event_series)
            print(current_time,"ms")
            for activate_id in event_series[current_time].try_activate:
                if symbols[activate_id].activated or goal_found:
                    continue
                if activate_id == goal:
                    goal_found = True
                    final_time = current_time
                if symbols[activate_id].tag:
                    print("Add tag")
                    sim_utils.addTrace(symbols, valid_transitions[activate_id].transition_ids)
                symbols[activate_id].activated = True
                symbols[activate_id].activated_at = current_time
                sim_utils.scheduleActivate(event_series, 
                                current_time + symbols[activate_id].spike_delay_ms, 
                                valid_transitions[activate_id].transition_ids)

            if event_series[current_time].catch_frame:
                sim_utils.catchFrame(symbols, current_time, recorder)

            if current_time > 1000:
                raise Exception("Too much time passed, out of bounds")
            del event_series[current_time]
        print("The goal was found after", final_time, "ms!")
  
        event_series.clear()

    print(len(recorder.plots))
    recorder.createAnimation()
    return

#%%
symbols = module_utils.generateRandomSymbols(100, [0,10], [0,10], 1)
plt.plot([symbol.coord[0] for symbol in symbols], [symbol.coord[1] for symbol in symbols], 'o')
transitions = module_utils.findAllTransitions(symbols, 2)


#%%
#Pick random start and goal from symbols:

[start, goal] = np.random.choice(len(symbols), 2, False)
print("From:", start, "to", goal)
print ("at ", symbols[start].coord, "and", symbols[goal].coord)

# %%

eligibilityNavigation(symbols, start, goal, 2, 1)

# %%
