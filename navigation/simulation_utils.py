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

def scheduleOutput(event_series, time, id):
    controlScheduleTime(event_series, time)
    event_series[time].output_ids = np.append(event_series[time].output_ids, id)

def scheduleSpike(event_series, time, ids):
    controlScheduleTime(event_series, time)
    event_series[time].try_spike_ids = np.append(event_series[time].try_spike_ids, ids)


def addTrace(symbols, current_time, ids):
    for id in ids:
        if symbols[id].activated_at is None or symbols[id].activated_at < current_time - 18:
            continue
        #if not symbols[id].tag:
            #print("New symbol tagged:", id)
        symbols[id].tag = True

def catchFrame(symbols, time,  start, goal, recorder, is_inhibit = False):
    plot_data = np.empty((2,0))
    alphas = np.empty(0)
    colors = np.empty(0)
    background_color = 'white' if not is_inhibit else 'lavender'
    for id, symbol in enumerate(symbols):
        if symbol.activated_at is None:
            continue 
        delta_t = time - symbol.activated_at
        if delta_t > 20:
            continue
        plot_data = np.hstack((plot_data, np.reshape(symbol.coord,(2,1))))
        norm_delt = delta_t / 20
        alphas = np.append(alphas, (norm_delt**3-2*norm_delt**2+norm_delt)/0.15)
        color = 'black' if id in [start, goal] else 'darkred' if symbol.tag else 'darksalmon' if symbol.feedback_window[0] < time < symbol.feedback_window[1] and symbol.inhibit_trace else 'lightskyblue' if symbol.inhibit_trace else 'cyan' if symbol.inhibit_window[0] < time < symbol.inhibit_window[1] else 'blue'
        colors = np.append(colors, color)
    recorder.backgrounds.append(background_color)
    recorder.plots.append(plot_data)
    recorder.alphas.append(alphas)
    recorder.color_codes.append(colors)