import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import module_utils
import simulation_utils as sim_utils
import recorder as rec

def eligibilityNavigation(symbols, start, goal, distance, ms_per_frame, f = 3, gif_name = "test"):
    valid_transitions = module_utils.findAllTransitions(symbols,distance) # Get a list containing, for each symbol, all their valid transitions
    current_time = 0
    goal_found = False
    event_series = {}
    recorder = rec.Recorder()
    inhibit_until = 0
    will_inhibit = False
    trigger_delay = 3

    # Reset symbols
    for symbol in symbols:
        symbol.reset()
    
    symbols[goal].tag = True
    
    for i in range(10):
        print("Start from start")
        for frame in range(0,500, ms_per_frame):
            sim_utils.scheduleFrame(event_series, current_time + frame)

        # Reset symbol to restart wave
        # TODO: Make the symbol deactivation plausible with a timing system. This should also work for restarting the wave.
        for symbol in symbols:
            symbol.activated = False
            symbol.activated_at = None
            symbol.tagable = False

        goal_found = False
        final_time = np.inf

        # Activate start to initiate wave propagation:
        sim_utils.scheduleSpikeEvent(event_series, current_time + symbols[start].spike_delay_ms, start)
        symbols[start].activated = True

        while final_time + 15 > current_time: # Keeps running until at least 15 ms after goal is found, or exception
            
            if(len(event_series) == 0):
                if not goal_found:
                    recorder.createAnimation()
                    raise Exception("No further events. Check animation to see what happened")
                else:
                    for frame in range(0,16, ms_per_frame):
                        sim_utils.scheduleFrame(event_series, current_time + frame)
            else:
                current_time = min(event_series)
                #print(current_time,"ms")

            # Neurons spike at this time if conditions are met, and receive speed-up:
            for spike_id in event_series[current_time].receive_input_ids:
                if inhibit_until > current_time or symbols[spike_id].activated or goal_found:
                    continue
                if symbols[spike_id].tag: 
                    symbols[spike_id].spike_delay_ms = f + (symbols[spike_id].spike_delay_ms-f)*0.5 # Speed up
                symbols[spike_id].activated = True
                sim_utils.scheduleSpikeEvent(event_series, current_time + trigger_delay, spike_id)

            # Output after spike:
            for output_id in event_series[current_time].spike_ids:
                if inhibit_until > current_time or goal_found:
                    continue
                for transition_id in valid_transitions[output_id].transition_ids:
                    sim_utils.scheduleSynapseEvent(event_series, current_time + symbols[transition_id].spike_delay_ms, transition_id)
                if output_id == goal:
                    goal_found = True
                    final_time = current_time
                    print("Goal found at", final_time)
                symbols[output_id].tagable = True
                symbols[output_id].activated_at = current_time
                if symbols[output_id].tag: # Add tags and inhibit
                    sim_utils.addTrace(symbols, valid_transitions[output_id].transition_ids)
                    will_inhibit = True

            # Set inhibition duriation
            if will_inhibit:
                inhibit_until = current_time + f
                will_inhibit = False          

            # Record the current state if desired
            if event_series[current_time].catch_frame:
                sim_utils.catchFrame(symbols, current_time, start, goal, recorder)

            # Delete the entries at current time, to let the flow of time be in a strictly forward direction
            del event_series[current_time]
        
        event_series.clear()
    print("From:", start, "to", goal)
    print ("at ", symbols[start].coord, "and", symbols[goal].coord)
    recorder.createAnimation(gif_name)
    return

#For random symbol, start and goal simulation:

symbols = module_utils.generateRandomSymbols(100, [0,10], [0,10], 1)
[start, goal] = np.random.choice(len(symbols), 2, False)
eligibilityNavigation(symbols, start, goal, 2, 1, 3)