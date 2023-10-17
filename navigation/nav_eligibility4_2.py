import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import module_utils
import simulation_utils as sim_utils
import recorder as rec

def eligibilityNavigation(symbols, start, goal, distance, ms_per_frame, f = 3, min_base_delay = 8, variance=2,  gif_name = "test"):
    valid_transitions = module_utils.findAllTransitions(symbols,distance) # Get a list containing, for each symbol, all their valid transitions
    event_series = {}
    recorder = rec.Recorder()
    inhibit_until = 0
    will_inhibit = False

    path_found = False
    refraction_time = 20

    start_time, current_time = 0, 0
    first_run_time = None

    # Reset symbols
    for symbol in symbols:
        symbol.reset(min_base_delay)
    
    symbols[goal].tag = True
    symbols[goal].spike_delay_ms = f
    
    for frame in range(0,1000, ms_per_frame):
        sim_utils.scheduleFrame(event_series, current_time + frame)

    # Activate start to initiate wave propagation:
    sim_utils.scheduleOutput(event_series, current_time, start)

    while not path_found: # Keeps running until the speed up has been sufficient
        
        # Verify that the schedule should still run, and update time accordingly
        if(len(event_series) == 0 or current_time > 1000):
            recorder.createAnimation()
            raise Exception("No further events or ran too long. Check gif to see what happened")
        else:
            current_time = min(event_series)
            #print(current_time,"ms")

        global_inhibit = inhibit_until > current_time

        # Activity
        for output_id in event_series[current_time].output_ids:
            symbol = symbols[output_id]

            # Only proceed if the symbol isn't inhibited or in refractory
            if global_inhibit or (symbol.activated_at is not None and symbol.activated_at > current_time - (refraction_time + 8)):
                continue
            
            # Set trace values
            symbol.activated_at = current_time
            symbol.inhibit_window = [current_time + 3 + f, current_time + 3 + min_base_delay]
            symbol.feedback_window = [current_time + 6 + f, current_time + 6 + min_base_delay]
            symbol.inhibit_trace = False

            # Activate next layer:
            sim_utils.scheduleSpike(event_series, current_time + 3, valid_transitions[output_id].transition_ids)
            
            # Add tag and inhibit: (This should happen after activation, so speed up only happens next circuit)
            if symbol.tag:
                will_inhibit = True
            
            # Check for goal funkiness:
            if output_id == goal:

                # If this is the first iteration, store the time it took:
                if first_run_time is None:
                    first_run_time = current_time

                # Find the time it took and check if the speed increase is sufficient:
                final_time = current_time - start_time
                print("Goal found in", final_time)
                print(final_time / first_run_time)
                
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
                start_time = current_time + refraction_time + 8
                inhibit_until = start_time
                sim_utils.scheduleOutput(event_series, start_time, start)

        # Receive input
        for spike_id in event_series[current_time].try_spike_ids:
            
            spike_symbol = symbols[spike_id]
            
            # Check if symbol should receive tag:
            if spike_symbol.inhibit_trace and spike_symbol.feedback_window[0] < current_time < spike_symbol.feedback_window[1]:
                spike_symbol.tag = True
                spike_symbol.spike_delay_ms = f #+ (spike_symbol.spike_delay_ms-f)*(0.5+(np.random.rand()-0.5)*0.2) # Speed up

            if spike_symbol.tag: # Recover from past speed up
                recovery = 0 #if spike_symbol.activated_at is None else (spike_symbol.original_spike_delay_ms - spike_symbol.spike_delay_ms)*(1-np.exp(-(current_time - spike_symbol.activated_at)/200))
                spike_symbol.spike_delay_ms = spike_symbol.spike_delay_ms + recovery
            sim_utils.scheduleOutput(event_series, current_time + spike_symbol.spike_delay_ms, spike_id)

        # Set inhibition duriation
        if will_inhibit:
            inhibit_until = max(current_time + f + 0.13, inhibit_until)
            for symbol in symbols:
                if symbol.inhibit_window[0] < current_time < symbol.inhibit_window[1]:
                    symbol.inhibit_trace = True
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
variance = 0
min_base_delay = 8
symbols = module_utils.generateRandomSymbols(100, min_base_delay, [0,10], [0,10], variance)
[start, goal] = np.random.choice(len(symbols), 2, False)
eligibilityNavigation(symbols, start, goal, 2, 1, 5, min_base_delay, variance)