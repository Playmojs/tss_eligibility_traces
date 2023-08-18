#%%
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import utils
import modules
import recorder as rec

#%%
def controlScheduleTime(event_series, time):
     if time not in event_series:
        event_series[time] = modules.TimedEvent()

def scheduleActivate(event_series, time, transition_ids):
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
    for symbol in symbols:
        if symbol.activated_at is None:
            continue 
        if time-symbol.activated_at < 10:
            plot_data = np.hstack((plot_data, np.reshape(symbol.coord,(2,1))))
    # plt.scatter(plot_data[:][0], plot_data[:][1])
    # plt.xlim(0,10)
    # plt.ylim(0,10)
    # plt.show()
    recorder.plots.append(plot_data)


def eligibilityNavigation(symbols, start, goal, distance, ms_per_frame):
    valid_transitions = utils.findAllTransitions(symbols,distance) #Get a list containing, for each symbol, all their valid transitions
    current_time = 0
    goal_found = False
    event_series = {}
    recorder = rec.Recorder()

    #reset
    for symbol in symbols:
        symbol.tag = False
        symbol.spike_delay_ms = 10
    
    symbols[goal].tag = True
    
    for i in range(10):
        for frame in range(0,100, ms_per_frame):
            scheduleFrame(event_series, current_time + frame)

        for symbol in symbols:
            symbol.activated = False
            symbol.activated_at = None

        symbols[start].activated = True
        symbols[start].activated_at = current_time
        goal_found = False

        scheduleActivate(event_series, current_time + symbols[start].spike_delay_ms, valid_transitions[start].transition_ids)

        while not goal_found:
            current_time = min(event_series)
            print(current_time,"ms")
            for activate_id in event_series[current_time].try_activate:
                if symbols[activate_id].activated:
                    continue
                if activate_id == goal:
                    goal_found = True
                if symbols[activate_id].tag:
                    print("Add tag")
                    addTrace(symbols, valid_transitions[activate_id].transition_ids)
                symbols[activate_id].activated = True
                symbols[activate_id].activated_at = current_time
                scheduleActivate(event_series, 
                                current_time + symbols[activate_id].spike_delay_ms, 
                                valid_transitions[activate_id].transition_ids)

            if event_series[current_time].catch_frame:
                catchFrame(symbols, current_time, recorder)



            if current_time > 1000:
                raise Exception("Too much time passed, out of bounds")
            del event_series[current_time]
        final_time = current_time
        print("The goal was found after", final_time, "ms!")
        while final_time + 15 > current_time:
            current_time = min(event_series)
            if event_series[current_time].catch_frame:
                catchFrame(symbols, current_time, recorder)
            del event_series[current_time]

        event_series.clear()

    print(len(recorder.plots))
    recorder.createAnimation()
    return

#%%
symbols = utils.generateRandomSymbols(100, [0,10], [0,10], 1)
plt.plot([symbol.coord[0] for symbol in symbols], [symbol.coord[1] for symbol in symbols], 'o')
transitions = utils.findAllTransitions(symbols, 2)


#%%
#Pick random start and goal from symbols:

[start, goal] = np.random.choice(len(symbols), 2, False)
print("From:", start, "to", goal)
print ("at ", symbols[start].coord, "and", symbols[goal].coord)

# %%

eligibilityNavigation(symbols, start, goal, 2, 1)

# %%
