#include<stdlib.h>
#include "brianlib/cuda_utils.h"
#include "objects.h"
#include<ctime>

#include "code_objects/neurongroup_1_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_1_stateupdater_codeobject.h"
#include "code_objects/neurongroup_2_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_2_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_2_stateupdater_codeobject.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include "code_objects/spikegeneratorgroup_codeobject.h"
#include "code_objects/synapses_1_post_codeobject.h"
#include "code_objects/synapses_1_post_push_spikes.h"
#include "code_objects/synapses_1_summed_variable_Igap_post_codeobject.h"
#include "code_objects/synapses_2_pre_codeobject.h"
#include "code_objects/synapses_2_pre_push_spikes.h"
#include "code_objects/synapses_2_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_3_pre_codeobject.h"
#include "code_objects/synapses_3_pre_push_spikes.h"
#include "code_objects/synapses_3_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_pre_codeobject.h"
#include "code_objects/synapses_pre_push_spikes.h"


void brian_start()
{
    _init_arrays();
    _load_arrays();
    srand(time(NULL));

    // Initialize clocks (link timestep and dt to the respective arrays)
    brian::defaultclock.timestep = brian::_array_defaultclock_timestep;
    brian::defaultclock.dt = brian::_array_defaultclock_dt;
    brian::defaultclock.t = brian::_array_defaultclock_t;
    brian::networkoperation_1_clock.timestep = brian::_array_networkoperation_1_clock_timestep;
    brian::networkoperation_1_clock.dt = brian::_array_networkoperation_1_clock_dt;
    brian::networkoperation_1_clock.t = brian::_array_networkoperation_1_clock_t;
    brian::networkoperation_2_clock.timestep = brian::_array_networkoperation_2_clock_timestep;
    brian::networkoperation_2_clock.dt = brian::_array_networkoperation_2_clock_dt;
    brian::networkoperation_2_clock.t = brian::_array_networkoperation_2_clock_t;
    brian::networkoperation_clock.timestep = brian::_array_networkoperation_clock_timestep;
    brian::networkoperation_clock.dt = brian::_array_networkoperation_clock_dt;
    brian::networkoperation_clock.t = brian::_array_networkoperation_clock_t;
}

void brian_end()
{
    _write_arrays();
    _dealloc_arrays();
}


