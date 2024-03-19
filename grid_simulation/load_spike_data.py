import numpy as np

f_name = 'grid_simulation/Results/37grid_spike_times.npz'
with np.load(f_name, allow_pickle=True) as data:
    spike_data = data['spike_data'].item()
    print(spike_data.keys()) # Dict med data fra fryste tidspunkter tatt hvert 5. minutt gjennom hele simuleringen 
    print(spike_data['0_minutes'].keys()) # Dict med data fra hver gridcelle (13 stykk her). Spike data fra hver celle er oppgitt i ms.