import numpy as np
import modules

def controlScheduleTime(event_series, time):
     if time not in event_series:
        event_series[time] = modules.TimedEvent()

def scheduleActivate(event_series, time, transition_ids, layer):
    controlScheduleTime(event_series, time)
    event_series[time].try_activate = np.append(event_series[time].try_activate, transition_ids)

def scheduleFrame(event_series, time):
    controlScheduleTime(event_series, time)
    event_series[time].catch_frame = True

def addTrace(symbols, ids):
    for id in ids:
        if symbols[id].activated:
            if not symbols[id].tag:
                print("New symbol tagged")
            symbols[id].tag = True
            symbols[id].spike_delay_ms *= 0.9

def catchFrame(symbols, time, recorder):
    plot_data = np.empty((2,0))
    alphas = np.empty(0)
    for symbol in symbols:
        if symbol.activated_at is None:
            continue 
        delta_t = time - symbol.activated_at
        if delta_t < 20:
            plot_data = np.hstack((plot_data, np.reshape(symbol.coord,(2,1))))
            norm_delt = delta_t / 20
            alphas = np.append(alphas, (norm_delt**3-2*norm_delt**2+norm_delt)/0.15)
    # plt.scatter(plot_data[:][0], plot_data[:][1])
    # plt.xlim(0,10)
    # plt.ylim(0,10)
    # plt.show()
    recorder.plots.append(plot_data)
    recorder.alphas.append(alphas)