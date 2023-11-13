import numpy as np
import modules

def controlScheduleTime(event_series, time):
     if time not in event_series:
        event_series[time] = modules.TimedEvent()

def scheduleActivate(event_series, time, transition_ids):
    controlScheduleTime(event_series, time)
    event_series[time].try_activate = np.append(event_series[time].try_activate, transition_ids)

def activate(symbols, activate_id, transition_ids, event_series, current_time, goal_found, f = 3):
    symbols[activate_id].activated = True
    symbols[activate_id].activated_at = current_time
    for transition_id in transition_ids:
        if symbols[transition_id].tag and not symbols[transition_id].has_sped_up and not goal_found:
            print(transition_id)
            print("Prior delay:", symbols[transition_id].spike_delay_ms)
            symbols[transition_id].spike_delay_ms = f + (symbols[transition_id].spike_delay_ms-f)*0.5
            print("Latter delay:", symbols[transition_id].spike_delay_ms)
            symbols[transition_id].has_sped_up = True
        scheduleActivate(event_series, current_time + symbols[transition_id].spike_delay_ms, transition_id)
       
def scheduleFrame(event_series, time):
    controlScheduleTime(event_series, time)
    event_series[time].catch_frame = True

def scheduleSpikeEvent(event_series, time, id, scale = -1):
    controlScheduleTime(event_series, time)
    if scale == -1:
        event_series[time].spike_ids = np.append(event_series[time].spike_ids, id)
    else:
        event_series[time].spike_events = np.vstack((event_series[time].spike_events, [id, scale]))

def scheduleSynapseEvent(event_series, time, ids, scale = -1):
    controlScheduleTime(event_series, time)
    if scale == -1:
        event_series[time].receive_input_ids = np.append(event_series[time].receive_input_ids, ids)
    else:
        event_info = np.vstack((ids, np.full(ids.size, scale))).T
        event_series[time].synapse_events = np.vstack((event_series[time].synapse_events, event_info))

def addTrace(symbols, current_time, ids):
    for id in ids:
        if symbols[id].activated_at is None or symbols[id].activated_at < current_time - 18:
            continue
        #if not symbols[id].tag:
            #print("New symbol tagged:", id)
        symbols[id].tag = True

def catchFrame(symbols, time,  start, goal, recorder, is_inhibit = False, color_rule = 'advanced_tag'):
    plot_data = np.empty((2,0))
    alphas = np.empty(0)
    colors = np.empty(0)
    background_color = 'white' if not np.any(is_inhibit) else 'lavender' if is_inhibit[0] else 'blanchedalmond' if is_inhibit[1] else 'khaki' if is_inhibit[2] else 'white'
    for id, symbol in enumerate(symbols):
        if np.isnan(symbol.activated_at).all():
            continue 
        delta_t = time - np.nanmax(symbol.activated_at)
        if delta_t > 20:
            continue
        plot_data = np.hstack((plot_data, np.reshape(symbol.coord,(2,1))))
        norm_delt = delta_t / 20
        alphas = np.append(alphas, (norm_delt**3-2*norm_delt**2+norm_delt)/0.15)
        if color_rule == "simple":
            color = 'blue'
        elif color_rule == 'tag':
            color = 'black' if id in [start,goal] else 'darkred' if np.any(symbol.tag) else 'blue'
        elif color_rule == 'advanced_tag':
            color = 'black' if id in [start, goal] else 'darkred' if np.any(symbol.tag) else 'darksalmon' if np.nanmax(symbol.feedback_window[:,0]) < time < np.nanmax(symbol.feedback_window[:,1]) and np.any(symbol.inhibit_trace) else 'darkorchid' if np.any(symbol.inhibit_trace) and time < np.nanmax(symbol.feedback_window[:,0]) else 'cyan' if np.nanmax(symbol.inhibit_window[:,0]) < time < np.nanmax(symbol.inhibit_window[:,1]) else 'blue'
        elif color_rule == 'simple_scale':
            palette = ['deepskyblue', 'indianred', 'goldenrod']
            tag_palette = np.array(['blue', 'darkred', 'darkgoldenrod'])
            color = 'black' if id in [start, goal] else tag_palette[symbol.tag][0] if np.any(symbol.tag) else palette[min(np.nanargmax(symbol.activated_at),2)]
        colors = np.append(colors, color)
    recorder.backgrounds.append(background_color)
    recorder.plots.append(plot_data)
    recorder.alphas.append(alphas)
    recorder.color_codes.append(colors)

def saveData(symbols, time, recorder):
    recorder.symbol_data.append(symbols)
    recorder.time.append(time)