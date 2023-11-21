import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import module_utils
import simulation_utils as sim_utils
import recorder as rec
import copy

def eligibilityNavigation(symbols, start, goal, distance, ms_per_frame, f = 3, min_base_delay = 8, variance=2, record = True, verbose = True, save_data = False, gif_name = "test", output_filepath = "output"):
    valid_transitions = module_utils.findAllTransitions(symbols,distance) # Get a list containing, for each symbol, all their valid transitions
    event_series = {}
    recorder = rec.Recorder()
    if save_data:
        symbol_data = dict()
        symbol_data[0] = copy.deepcopy(symbols)
        inhibit_data = np.empty((0,2))

    inhibit_until = 0
    will_inhibit = False
    correct_tag = True

    path_found = False
    refraction_time = 20

    start_time, current_time = 0, 0
    first_run_time = None

    # Reset symbols
    for symbol in symbols:
        symbol.reset(min_base_delay)
    
    symbols[goal].tag = True
    symbols[goal].spike_delay_ms = f
    
    for frame in range(0,2000, ms_per_frame):
        sim_utils.scheduleFrame(event_series, current_time + frame)

    # Activate start to initiate wave propagation:
    sim_utils.scheduleSynapseEvent(event_series, current_time, start)

    while not path_found: # Keeps running until the speed up has been sufficient
        
        # Verify that the schedule should still run, and update time accordingly
        if(len(event_series) == 0 or current_time > 2000):
            if record:
                recorder.createAnimation()
            if save_data:
                return rec.DataRec(symbol_data, start, goal, inhibit_data, correct_tag, False, distance, variance)
            #raise Exception("No further events or ran too long. Check gif to see what happened")
            return False
        else:
            current_time = min(event_series)
            #print(current_time,"ms")

        global_inhibit = inhibit_until > current_time

        # Activity
        for spike_id in event_series[current_time].spike_ids:
            symbol = symbols[spike_id]

            # Only proceed if the symbol isn't inhibited or in refractory
            if global_inhibit or (symbol.activated_at is not None and symbol.activated_at > current_time - (refraction_time + 8)):
                continue
            
            # Set trace values
            symbol.activated_at = current_time
            symbol.inhibit_window = [current_time + 2 + f, current_time + 3 + min_base_delay]
            symbol.feedback_window = [current_time + 5 + f, current_time + 6 + min_base_delay]
            symbol.inhibit_trace = False

            # Activate next layer:
            sim_utils.scheduleSynapseEvent(event_series, current_time + 3, valid_transitions[spike_id].transition_ids)
            
            # Add tag and inhibit: (This should happen after activation, so speed up only happens next circuit)
            if symbol.tag:
                will_inhibit = True
            
            # Check for goal funkiness:
            if spike_id == goal:

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
                    if correct_tag:
                        tag_ids = sim_utils.extractTagged(symbols)
                        correct_tag = sim_utils.evaluateTag(symbols, tag_ids, valid_transitions, goal)

                if final_time < first_run_time * 0.8:
                    i += 1
                    if verbose:
                        print("Possible path found")
                else:
                    i = 0
                
                if i >= 2:
                    path_found = True
                    if verbose:
                        print("Success, path successfully found!")
                        print("Time reduced by factor", final_time / first_run_time)

                # Initiate next iteration after a long inhibition to reset symbols:
                start_time = current_time + refraction_time + 8
                inhibit_until = start_time
                sim_utils.scheduleSynapseEvent(event_series, start_time, start)

        # Receive input
        for input_id in event_series[current_time].receive_input_ids: 
            
            input_symbol = symbols[input_id]
            
            # Check if symbol should receive tag:
            if input_symbol.inhibit_trace and input_symbol.feedback_window[0] < current_time < input_symbol.feedback_window[1]:
                input_symbol.tag = True
                input_symbol.spike_delay_ms = f + (input_symbol.spike_delay_ms-f)*(0.5+(np.random.rand()-0.5)*0.2) # Speed up

            if input_symbol.tag: # Recover from past speed up
                recovery = 0 #if input_symbol.activated_at is None else (input_symbol.original_spike_delay_ms - input_symbol.spike_delay_ms)*(1-np.exp(-(current_time - input_symbol.activated_at)/200))
                input_symbol.spike_delay_ms = input_symbol.spike_delay_ms + recovery
            sim_utils.scheduleSpikeEvent(event_series, current_time + input_symbol.spike_delay_ms, input_id)

        # Set inhibition duriation
        if will_inhibit:
            inhibit_until = max(current_time + 3 + f, inhibit_until)
            if save_data:
                inhibit_data = np.vstack((inhibit_data, [current_time, inhibit_until]))
            for symbol in symbols:
                if symbol.inhibit_window[0] < current_time < symbol.inhibit_window[1]:
                    symbol.inhibit_trace = True
            will_inhibit = False          

        # Record the current state if desired
        if event_series[current_time].catch_frame and record:
            sim_utils.catchFrame(symbols, current_time, start, goal, recorder, False, 'tag')

        # Delete the entries at current time, to let the flow of time be in a strictly forward direction
        del event_series[current_time]

    event_series.clear()
    for _ in range(40):
        current_time += 1
        sim_utils.catchFrame(symbols, current_time, start, goal, recorder, False, 'tag')

    if verbose:
        print ("From ", symbols[start].coord, "to ", symbols[goal].coord)
        print("Total time: ", current_time)
    if record:
        recorder.createAnimation(gif_name)
    if save_data:
        return(rec.DataRec(symbol_data, start, goal, inhibit_data, correct_tag, True, distance, variance))
    return True

#For random symbol, start and goal simulation:
variance = 1
min_base_delay = 5
valid_goal = False
min_dist = 10
model_num = 10
symbols = module_utils.generateRandomSymbols(200, min_base_delay, [0,10], [0,10], variance)
while not valid_goal:
    [start, goal] = np.random.choice(len(symbols), 2, False)
    valid_goal = np.linalg.norm(symbols[start].coord - symbols[goal].coord) > min_dist
eligibilityNavigation(symbols, start, goal, 2, 1, 2, min_base_delay, variance, record = True, save_data= True, output_filepath="navigation/ppt_viz")