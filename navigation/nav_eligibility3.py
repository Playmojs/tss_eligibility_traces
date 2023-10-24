import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import module_utils
import simulation_utils as sim_utils
import recorder as rec

def eligibilityNavigation(symbols, start, goal, distance, ms_per_frame, f = 3, gif_name = "test"):
    valid_transitions = module_utils.findAllTransitions(symbols,distance) # Get a list containing, for each symbol, all their valid transitions
    event_series = {}
    recorder = rec.Recorder()
    
    inhibit_until = 0
    will_inhibit = False
    trigger_delay = 3
    path_found = False
    refraction_time = 12

    current_time = 0
    first_run_time = None
    start_time = 0

    # Reset symbols
    for symbol in symbols:
        symbol.reset(5)
    
    symbols[goal].tag = True
    
    
    print("Start from start")
    for frame in range(0,1000, ms_per_frame):
        sim_utils.scheduleFrame(event_series, current_time + frame)

    # Reset symbol to restart wave
    for symbol in symbols:
        symbol.activated_at = None

    # Activate start to initiate wave propagation:
    sim_utils.scheduleSpikeEvent(event_series, symbols[start].spike_delay_ms, start)

    while not path_found: # Keeps running until the speed up has been sufficient
        
        # Verify that the schedule should still run, and update time accordingly
        if(len(event_series) == 0 or current_time > 1000):
            recorder.createAnimation()
            raise Exception("No further events or ran too long. Check animation to see what happened")
        else:
            current_time = min(event_series)
            #print(current_time,"ms")

        global_inhibit = inhibit_until > current_time        
        # Neurons spike at this time if conditions are met, and receive speed-up:
        for spike_id in event_series[current_time].receive_input_ids:
            symbol = symbols[spike_id]
            if global_inhibit or (symbol.activated_at is not None and symbol.activated_at > current_time - (refraction_time+6)):
                continue

            if symbol.tag:
                if symbol.activated_at is None:
                    recovery = 0
                else:
                    recovery = (symbol.original_spike_delay_ms - symbol.spike_delay_ms)*(1-np.exp(-(current_time - symbol.activated_at)/500))
                symbol.spike_delay_ms = symbol.spike_delay_ms + recovery
                if not symbol.activated:
                    symbol.spike_delay_ms = f + (symbol.spike_delay_ms-f)*(0.5+(np.random.rand()-0.5)*0.2) # Speed up
                    symbol.activated = True
                    #print("Recovery:", recovery)
                    print("Delay:", spike_id, symbol.spike_delay_ms)
            sim_utils.scheduleSpikeEvent(event_series, current_time + trigger_delay, spike_id)

        # Output after spike:
        for output_id in event_series[current_time].spike_ids:
            symbol = symbols[output_id]
            if global_inhibit or (symbol.activated_at is not None and symbol.activated_at > current_time - (refraction_time + 6)):
                continue
            
            symbol.activated = False
            # Activate next layer:
            symbol.activated_at = current_time
            for transition_id in valid_transitions[output_id].transition_ids:
                sim_utils.scheduleSynapseEvent(event_series, current_time + symbols[transition_id].spike_delay_ms, transition_id)

            # Add tag:
            if symbol.tag: # Add tags and inhibit
                sim_utils.addTrace(symbols, current_time, valid_transitions[output_id].transition_ids)
                will_inhibit = True

            # Check for goal funkiness:
            if output_id == goal:
                # If this is the first iteration, store the time it took:
                if first_run_time is None:
                    first_run_time = current_time

                # Find the time it took and check if the speed increase is sufficient:
                final_time = current_time - start_time
                print("Goal found in", final_time)
                
                if final_time < first_run_time * 0.8:
                    i += 1
                    print("Possible path found")
                else:
                    i = 0
                
                if i >= 3:
                    path_found = True
                    print("Success, path successfully found!")
                    print("Time reduced by factor", final_time / first_run_time)

                # Initiate next iteration after a long inhibition to reset symbols:
                start_time = current_time + refraction_time
                inhibit_until = start_time
                sim_utils.scheduleSynapseEvent(event_series, start_time, start)


        # Set inhibition duriation
        if will_inhibit:
            inhibit_until = max(current_time + f + 0.13, inhibit_until)
            will_inhibit = False          

        # Record the current state if desired
        if event_series[current_time].catch_frame:
            sim_utils.catchFrame(symbols, current_time, start, goal, recorder, global_inhibit)

        # Delete the entries at current time, to let the flow of time be in a strictly forward direction
        del event_series[current_time]


    event_series.clear()
    for _ in range(40):
        current_time += 1
        sim_utils.catchFrame(symbols, current_time, start, goal, recorder)

    print ("From ", symbols[start].coord, "to ", symbols[goal].coord)
    print("Total time: ", current_time)
    recorder.createAnimation(gif_name)
    return

#For random symbol, start and goal simulation:

symbols = module_utils.generateRandomSymbols(100, 5, [0,10], [0,10], 1)
[start, goal] = np.random.choice(len(symbols), 2, False)
eligibilityNavigation(symbols, start, goal, 2, 1, 3)