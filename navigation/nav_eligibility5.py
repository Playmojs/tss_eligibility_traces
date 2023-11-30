import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import module_utils
import simulation_utils as sim_utils
import recorder as rec
import copy

def eligibilityNavigation(symbols, start, goal, distance, nscales = 1, 
                          ms_per_frame = 1, f = 3, min_base_delay = 8, variance=2, 
                          record = True, verbose = True, save_data = False, color_rule = 'advanced_tag', gif_name = "test", 
                          output_filepath = "output"):


    valid_transitions = module_utils.findAllTransitions(symbols,distance, nscales, compressed = False) # Get a list containing, for each symbol, all their valid transitions
    event_series = {}
    recorder = rec.Recorder()
    if save_data:
        symbol_data = dict()
        symbol_data[0] = copy.deepcopy(symbols)
        inhibit_data = np.empty((0,2))

    inhibit_until, will_inhibit = np.zeros(nscales), np.zeros(nscales)

    scale_time_penalty = np.linspace(0,1, nscales)
    path_found = False
    refraction_time = 20

    start_time,  current_time = 0, 0
    first_run_time = None

    # Reset symbols
    for symbol in symbols:
        symbol.reset(min_base_delay, compressed = False)
    
    symbols[goal].tag[:] = True
    symbols[goal].spike_delay_ms = f
    
    for frame in range(0,2000, ms_per_frame):
        sim_utils.scheduleFrame(event_series, current_time + frame)

    # Activate start to initiate wave propagation:
    for i in range(nscales-1, -1, -1):
        sim_utils.scheduleSynapseEvent(event_series, current_time, np.array([start]), scale = i)

    while not path_found: # Keeps running until time reduction is sufficiently big
        
        # Verify that the schedule should still run, and update time accordingly
        if(len(event_series) == 0 or current_time > 1000):
            if record:
                recorder.createAnimation()
            #raise Exception("No further events or ran too long. Check gif to see what happened")
            return False
        else:
            current_time = min(event_series)
            print(current_time)

        global_inhibit = inhibit_until > current_time

        # Activity
        for spike_info in event_series[current_time].spike_events:
            symbol = symbols[spike_info[0]]
            scale = spike_info[1]

            # Only proceed if the symbol isn't inhibited or in refractory
            if global_inhibit[scale] or (symbol.activated_at[scale] is not None and symbol.activated_at[scale] > current_time - (refraction_time + 8)):
                continue
            
            # Set trace values
            symbol.activated_at[scale:nscales] = current_time
            symbol.inhibit_window[scale] = [current_time + 3 + f, current_time + 3 + min_base_delay] + scale_time_penalty[scale]
            symbol.feedback_window[scale] = [current_time + 6 + f, current_time + 6 + min_base_delay] + 2*scale_time_penalty[scale]
            symbol.inhibit_trace[scale] = False

            # Activate next layer:
            sim_utils.scheduleSynapseEvent(event_series, current_time + 3 + scale_time_penalty[scale], valid_transitions[spike_info[0], scale].transition_ids, scale)
            
            # Add tag and inhibit: (This should happen after activation, so speed up only happens next circuit)
            if symbol.tag[scale]:
                will_inhibit[scale:nscales] = True
            
            # Check for goal funkiness:
            if spike_info[0] == goal:
                
                inhibit_until[scale] = current_time + refraction_time + 8

                if scale != 0:
                    continue
                # If this is the first iteration, store the time it took:
                if first_run_time is None:
                    first_run_time = current_time

                # Find the time it took and check if the speed increase is sufficient:
                final_time = current_time - start_time
                if verbose: 
                    print("Goal found in", final_time)
                    print(final_time / first_run_time)
                
                if save_data:
                    symbol_data[current_time] = copy.deepcopy(symbols)

                if final_time < first_run_time * 0.8:
                    i += 1
                    if verbose:
                        print("Possible path found")
                else:
                    i = 0
                
                if i >= 3:
                    path_found = True
                    if verbose:
                        print("Success, path successfully found!")
                        print("Time reduced by factor", final_time / first_run_time)

                # Initiate next iteration after a long inhibition to reset symbols:
                start_time = current_time + refraction_time + 8
                for iscale in range(nscales-1, -1, -1):
                    sim_utils.scheduleSynapseEvent(event_series, start_time, np.array([start]), iscale)

        # Receive input
        for input_info in event_series[current_time].synapse_events: 
            
            input_symbol = symbols[input_info[0]]
            scale = input_info[1]

            if not input_symbol.tag[scale] and np.random.rand() > 2**(scale/2):
                continue
            
            # Check if symbol should receive tag:
            if input_symbol.inhibit_trace[scale] and input_symbol.feedback_window[scale, 0] <= current_time <= input_symbol.feedback_window[scale, 1]:
                input_symbol.tag[scale] = True
                input_symbol.spike_delay_ms = f + (input_symbol.spike_delay_ms-f)*(1-(0.5 + (np.random.rand()-0.5)*0.2))#/(np.sqrt(scale+1))) # Speed up

            if input_symbol.tag[scale]: # Recover from past speed up
                recovery = 0 #if input_symbol.activated_at is None else (input_symbol.original_spike_delay_ms - input_symbol.spike_delay_ms)*(1-np.exp(-(current_time - input_symbol.activated_at)/200))
                input_symbol.spike_delay_ms = input_symbol.spike_delay_ms + recovery
            sim_utils.scheduleSpikeEvent(event_series, current_time + input_symbol.spike_delay_ms, input_info[0], scale)

        # Set inhibition duriation
        for iscale, inhib in enumerate(will_inhibit):    
            if inhib:
                inhibit_until[iscale] = max(current_time + 3 + f, inhibit_until[iscale])
                if save_data:
                    inhibit_data = np.vstack((inhibit_data, [current_time, inhibit_until]))
                for symbol in symbols:
                    if symbol.inhibit_window[iscale][0] <= current_time <= symbol.inhibit_window[iscale][1]:
                        symbol.inhibit_trace[iscale] = True
                will_inhibit[iscale] = False        

        # Record the current state if desired
        if event_series[current_time].catch_frame:
            sim_utils.catchFrame(symbols, current_time, start, goal, recorder, global_inhibit, color_rule)

        # Delete the entries at current time, to let the flow of time be in a strictly forward direction
        del event_series[current_time]

    event_series.clear()
    for _ in range(40):
        current_time += 1
        sim_utils.catchFrame(symbols, current_time, start, goal, recorder, color_rule)

    if verbose:
        print ("From ", symbols[start].coord, "to ", symbols[goal].coord)
        print("Total time: ", current_time)
    if record:
        recorder.createAnimation(gif_name)
    if save_data:
        np.savez_compressed(f"navigation/result_data/{output_filepath}", \
                            symbols = symbol_data, \
                            start = start, \
                            goal = goal, \
                            dist = distance, \
                            inhibit_ranges = inhibit_data
                            )
    return True

#For random symbol, start and goal simulation:
variance = 1
min_base_delay = 5
valid_goal = False
min_dist = 10
n_scales = 7
model_num = 11
symbols = module_utils.generateRandomSymbols(400, min_base_delay, [0,10], [0,10], variance, n_scales, False)
while not valid_goal:
    [start, goal] = np.random.choice(len(symbols), 2, False)
    valid_goal = np.linalg.norm(symbols[start].coord - symbols[goal].coord) > min_dist
eligibilityNavigation(symbols, start, goal, 1, n_scales, 1, 2, min_base_delay, variance, color_rule= 'simple_scale', save_data= False, output_filepath="400_1ms_2")