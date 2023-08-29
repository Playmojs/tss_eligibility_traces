import numpy as np
import modules

def controlScheduleTime(event_series, time):
     if time not in event_series:
        event_series[time] = modules.TimedEvent()

def scheduleActivate(event_series, time, transition_ids):
    controlScheduleTime(event_series, time)
    event_series[time].try_activate = np.append(event_series[time].try_activate, transition_ids)

def activate(symbols, activate_id, transition_ids, event_series, current_time):
    symbols[activate_id].activated = True
    symbols[activate_id].activated_at = current_time
    speed_up = True if symbols[activate_id].tag else False

    for transition_id in transition_ids:
        scheduleActivate(event_series, current_time + symbols[transition_id].spike_delay_ms, transition_id)
        if symbols[transition_id].tag and not symbols[transition_id].activated:
            speed_up = True
    
    if speed_up:
        symbols[activate_id].spike_delay_ms = 3 + (symbols[activate_id].spike_delay_ms-3)*0.5

def scheduleFrame(event_series, time):
    controlScheduleTime(event_series, time)
    event_series[time].catch_frame = True

def addTrace(symbols, ids):
    for id in ids:
        if not symbols[id].activated:
            continue
        if not symbols[id].tag:
            print("New symbol tagged")
        symbols[id].tag = True


def catchFrame(symbols, time,  start, goal, recorder):
    plot_data = np.empty((2,0))
    alphas = np.empty(0)
    colors = np.empty(0)
    for id, symbol in enumerate(symbols):
        if symbol.activated_at is None:
            continue 
        delta_t = time - symbol.activated_at
        if delta_t > 20:
            continue
        plot_data = np.hstack((plot_data, np.reshape(symbol.coord,(2,1))))
        norm_delt = delta_t / 20
        alphas = np.append(alphas, (norm_delt**3-2*norm_delt**2+norm_delt)/0.15)
        colors = np.append(colors, 'y' if id == start else 'g' if id == goal else 'r' if symbol.tag else 'b')

    recorder.plots.append(plot_data)
    recorder.alphas.append(alphas)
    recorder.color_codes.append(colors)