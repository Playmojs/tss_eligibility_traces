import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import modules
import simulation_utils as sim_utils
import recorder as rec
import copy

def eligibilityNavigation(symbols, start, goal, distance, nscales = 1, 
                          ms_per_frame = 1, f = 3, min_base_delay = 8, variance=2, 
                          record = True, verbose = True, save_data = False, color_rule = 'no_tag_scale', gif_name = "test", 
                          output_filepath = "output"):


    transitions = modules.Transitions(symbols,distance, nscales)
    event_series = {}
    recorder = rec.Recorder()
    if save_data:
        symbol_data = dict()
        symbol_data[0] = copy.deepcopy(symbols)
        inhibit_data = np.empty((0,2))

    inhibit_until, will_inhibit = np.zeros(nscales), np.zeros(nscales)
    
    spike_is_scheduled = np.zeros((nscales, symbols.nsymbols), dtype = bool)

    scale_time_penalty = np.linspace(0,1, nscales)
    path_found = False
    refraction_time = 20

    start_time,  current_time = 0, 0
    first_run_time = None

    # Reset symbols
    symbols.reset()
    
    symbols.tag[goal, :] = True
    symbols.spike_delay_ms[goal] = f
    
    time_limit = 500 #ms

    for frame in range(0,time_limit, ms_per_frame):
        sim_utils.scheduleFrame(event_series, current_time + frame)

    # Activate start to initiate wave propagation:
    for i in range(nscales-1, -1, -1):
        sim_utils.scheduleSynapseEvent(event_series, current_time, np.array([start]), scale = i)
    
    n_loops= 0
    n_conts = 0

    while not path_found: # Keeps running until time reduction is sufficiently big
        
        # Verify that the schedule should still run, and update time accordingly
        if(len(event_series) == 0 or current_time > time_limit):
            if record:
                recorder.createAnimation()
            #raise Exception("No further events or ran too long. Check gif to see what happened")
            return False
        else:
            current_time = min(event_series)
            n_loops+= 1
            #print(n_loops)

        global_inhibit = inhibit_until > current_time

        # Spike Activity:
        for spike_info in event_series[current_time].spike_events:
            id = spike_info[0]
            scale = spike_info[1]

            spike_is_scheduled[scale, id] = False
            # Only proceed if the symbol isn't inhibited or in refractory
            if global_inhibit[scale] or (symbols.activated_at[id, scale] > current_time - (refraction_time + 8)):
                n_conts += 1
                print(n_conts)
                continue
            
            # Set trace values
            symbols.activated_at[id, scale:nscales] = current_time
            symbols.inhibit_window[id, scale] = [current_time + 3 + f, current_time + 3 + min_base_delay] + scale_time_penalty[scale]
            symbols.feedback_window[id,scale] = [current_time + 6 + f, current_time + 6 + min_base_delay] + 2*scale_time_penalty[scale]
            symbols.inhibit_trace[id, scale] = False

            # Activate next layer:
            transition_targets = symbols.ids[transitions.transition_mask[scale, id, :]]
            sim_utils.scheduleSynapseEvent(event_series, current_time + 3 + scale_time_penalty[scale], transition_targets, scale)
            
            # Add tag and inhibit: (This should happen after activation, so speed up only happens next circuit)
            if symbols.tag[id, 0]:
                will_inhibit[scale:nscales] = True
            
            # Check for goal funkiness:
            if id == goal:
                
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

                if final_time < first_run_time * 0.7:
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
                symbols.no_tag = np.zeros(symbols.nsymbols, dtype = bool)
                for iscale in range(nscales-1, -1, -1):
                    sim_utils.scheduleSynapseEvent(event_series, start_time, np.array([start]), iscale)

        # Receive input
        for input_info in event_series[current_time].synapse_events: 
            
            id = input_info[0]
            scale = input_info[1]

            # Check if symbol should receive tag:
            if symbols.inhibit_trace[id, scale] and symbols.feedback_window[id, scale, 0] <= current_time <= symbols.feedback_window[id, scale, 1] and not symbols.no_tag[id]:
                if scale == 0:
                    symbols.tag[id, scale] = True
                    symbols.spike_delay_ms[id] = f + (symbols.spike_delay_ms[id]-f)*(1-(0.5 + (np.random.rand()-0.5)*0.2))#/(np.sqrt(scale+1))) # Speed up
                else:
                    symbols.no_tag[id] = True

            if scale == 0 and symbols.tag[id, scale]: # Recover from past speed up
                recovery = 0 #if input_symbol.activated_at is None else (input_symbol.original_spike_delay_ms - input_symbol.spike_delay_ms)*(1-np.exp(-(current_time - input_symbol.activated_at)/200))
                symbols.spike_delay_ms[id] = symbols.spike_delay_ms[id] + recovery
            spike_event_time = current_time + symbols.spike_delay_ms[id]
            inhibited_or_refractive = inhibit_until[scale] > spike_event_time or symbols.activated_at[id, scale] > spike_event_time - (refraction_time + 8)
            if inhibit_until[scale] > spike_event_time or symbols.activated_at[id, scale] > spike_event_time - (refraction_time + 8) or spike_is_scheduled[scale, id]:
                continue
            spike_is_scheduled[scale, id] = True
            sim_utils.scheduleSpikeEvent(event_series, spike_event_time, id, scale)

        # Set inhibition duriation
        for iscale, inhib in enumerate(will_inhibit):    
            if inhib:
                inhibit_until[iscale] = max(current_time + 3 + f + (3+refraction_time)*int(iscale>0), inhibit_until[iscale])
                if save_data:
                    inhibit_data = np.vstack((inhibit_data, [current_time, inhibit_until]))
                symbols.inhibit_trace[:,iscale] = np.logical_and(symbols.inhibit_window[:,iscale, 0] <= current_time, current_time <= symbols.inhibit_window[:,iscale,1])
                will_inhibit[iscale] = False        

        # Record the current state if desired
        if event_series[current_time].catch_frame:
            sim_utils.catchF(symbols, current_time, start, goal, recorder, global_inhibit, color_rule)

        # Delete the entries at current time, to let the flow of time be in a strictly forward direction
        del event_series[current_time]

    event_series.clear()
    for _ in range(40):
        current_time += 1
        sim_utils.catchF(symbols, current_time, start, goal, recorder, color_rule)

    if verbose:
        print ("From ", symbols.coords[start], "to ", symbols.coords[goal])
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
nsymbols = 400
variance = 1
base_delay = 5
valid_goal = False
min_dist = 10
n_scales = 7
model_num = 11
symbols = modules.Symbols(nsymbols, [0,10], [0,10], base_delay, variance, n_scales)
while not valid_goal:
    [start, goal] = np.random.choice(nsymbols, 2, False)
    valid_goal = np.linalg.norm(symbols.coords[start] - symbols.coords[goal]) > min_dist
symbols.start[start] = True
symbols.goal[goal] = True
eligibilityNavigation(symbols, start, goal, 1, n_scales, 1, 2, base_delay, variance, save_data= False)