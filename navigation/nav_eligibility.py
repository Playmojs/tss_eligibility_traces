import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import module_utils
import simulation_utils as sim_utils
import recorder as rec

def eligibilityNavigation(symbols, start, goal, distance, ms_per_frame):
    valid_transitions = module_utils.findAllTransitions(symbols,distance) # Get a list containing, for each symbol, all their valid transitions
    current_time = 0
    goal_found = False
    event_series = {}
    recorder = rec.Recorder()

    # Reset symbols
    for symbol in symbols:
        symbol.reset()
    
    symbols[goal].tag = True
    
    for i in range(10):

        for frame in range(0,100, ms_per_frame):
            sim_utils.scheduleFrame(event_series, current_time + frame)

        # Reset symbol to restart wave
        # TODO: Make the symbol deactivation plausible with a timing system. This should also work for restartin the wave.
        for symbol in symbols:
            symbol.activated = False
            symbol.activated_at = None

        goal_found = False
        final_time = np.inf

        # Activate start to initiate wave propagation:
        sim_utils.activate(symbols, start, valid_transitions[start].transition_ids, event_series, current_time)

        while final_time + 15 > current_time: # Keeps running until at least 15 ms after goal is found, or exception
            
            if(len(event_series) == 0 and not goal_found):
                recorder.createAnimation()
                raise Exception("No further events. Check animation to see what happened")
            
            previous_time = current_time
            current_time = min(event_series)
            print(current_time,"ms")

            for activate_id in event_series[current_time].try_activate:

                if symbols[activate_id].activated or goal_found:
                    continue

                if activate_id == goal:
                    goal_found = True
                    final_time = current_time
                    #symbols[goal].spike_delay_ms = 3 + (symbols[goal].spike_delay_ms-3)*0.7
                    print("Goal found at", current_time)

                if symbols[activate_id].tag:
                    sim_utils.addTrace(symbols, valid_transitions[activate_id].transition_ids)

                # Activates symbol, and schedule activate for all valid transitions:
                sim_utils.activate(symbols, activate_id,
                                valid_transitions[activate_id].transition_ids, event_series, current_time)

            if event_series[current_time].catch_frame:
                sim_utils.catchFrame(symbols, current_time, start, goal, recorder)

            if current_time > 1000:
                raise Exception("Too much time passed, out of bounds")
            del event_series[current_time]
        print("The goal was found after", final_time, "ms!")

        event_series.clear()
    print("From:", start, "to", goal)
    print ("at ", symbols[start].coord, "and", symbols[goal].coord)
    print(len(recorder.plots))
    recorder.createAnimation()
    return


symbols = module_utils.generateRandomSymbols(100, [0,10], [0,10], 1)
plt.plot([symbol.coord[0] for symbol in symbols], [symbol.coord[1] for symbol in symbols], 'o')
transitions = module_utils.findAllTransitions(symbols, 2)


#Pick random start and goal from symbols:

[start, goal] = np.random.choice(len(symbols), 2, False)

eligibilityNavigation(symbols, start, goal, 2, 1)