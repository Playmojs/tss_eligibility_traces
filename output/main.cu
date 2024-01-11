#include <stdlib.h>
#include "objects.h"
#include <ctime>
#include <time.h>
#include "run.h"
#include "brianlib/common_math.h"
#include "brianlib/cuda_utils.h"
#include "rand.h"

#include "code_objects/synapses_2_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_3_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_1_summed_variable_Igap_post_codeobject.h"
#include "code_objects/neurongroup_1_stateupdater_codeobject.h"
#include "code_objects/neurongroup_2_stateupdater_codeobject.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include "code_objects/neurongroup_1_spike_thresholder_codeobject.h"
#include "code_objects/after_run_neurongroup_1_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_2_spike_thresholder_codeobject.h"
#include "code_objects/after_run_neurongroup_2_spike_thresholder_codeobject.h"
#include "code_objects/spikegeneratorgroup_codeobject.h"
#include "code_objects/synapses_2_pre_push_spikes.h"
#include "code_objects/before_run_synapses_2_pre_push_spikes.h"
#include "code_objects/synapses_2_pre_codeobject.h"
#include "code_objects/synapses_3_pre_push_spikes.h"
#include "code_objects/before_run_synapses_3_pre_push_spikes.h"
#include "code_objects/synapses_3_pre_codeobject.h"
#include "code_objects/synapses_pre_push_spikes.h"
#include "code_objects/before_run_synapses_pre_push_spikes.h"
#include "code_objects/synapses_pre_codeobject.h"
#include "code_objects/synapses_1_post_push_spikes.h"
#include "code_objects/before_run_synapses_1_post_push_spikes.h"
#include "code_objects/synapses_1_post_codeobject.h"
#include "code_objects/neurongroup_2_spike_resetter_codeobject.h"


#include <iostream>
#include <fstream>
#include <string>
#include "cuda_profiler_api.h"




int main(int argc, char **argv)
{
        


    // seed variable set in Python through brian2.seed() calls can use this
    // variable (see device.py CUDAStandaloneDevice.generate_main_source())
    unsigned long long seed;

    //const std::clock_t _start_time = std::clock();

    CUDA_SAFE_CALL(
            cudaSetDevice(0)
            );

    cudaDeviceProp props;
    CUDA_SAFE_CALL(
            cudaGetDeviceProperties(&props, 0)
            );
    size_t limit = 128 * 1024 * 1024;
    CUDA_SAFE_CALL(
            cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit)
            );
    CUDA_SAFE_CALL(
            cudaDeviceSynchronize()
            );

    //const double _run_time2 = (double)(std::clock() -_start_time)/CLOCKS_PER_SEC;
    //printf("INFO: setting cudaDevice stuff took %f seconds\n", _run_time2);

    brian_start();

        


    //const std::clock_t _start_time3 = std::clock();
    {
        using namespace brian;

                
        for(int i=0; i<_num__array_spikegeneratorgroup__spikespace; i++)
        {
            _array_spikegeneratorgroup__spikespace[i] = - 1;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_spikegeneratorgroup__spikespace[current_idx_array_spikegeneratorgroup__spikespace],
                &_array_spikegeneratorgroup__spikespace[0],
                sizeof(_array_spikegeneratorgroup__spikespace[0])*_num__array_spikegeneratorgroup__spikespace,
                cudaMemcpyHostToDevice
            )
        );
        _array_spikegeneratorgroup__spikespace[_num__array_spikegeneratorgroup__spikespace - 1] = 0;
                            CUDA_SAFE_CALL(
                                cudaMemcpy(
                                    dev_array_spikegeneratorgroup__spikespace[current_idx_array_spikegeneratorgroup__spikespace] + _num__array_spikegeneratorgroup__spikespace - 1,
                                    &_array_spikegeneratorgroup__spikespace[_num__array_spikegeneratorgroup__spikespace - 1],
                                    sizeof(_array_spikegeneratorgroup__spikespace[_num__array_spikegeneratorgroup__spikespace - 1]),
                                    cudaMemcpyHostToDevice
                                )
                            );
        for(int i=0; i<_num__array_neurongroup_2__spikespace; i++)
        {
            _array_neurongroup_2__spikespace[i] = - 1;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_2__spikespace[current_idx_array_neurongroup_2__spikespace],
                &_array_neurongroup_2__spikespace[0],
                sizeof(_array_neurongroup_2__spikespace[0])*_num__array_neurongroup_2__spikespace,
                cudaMemcpyHostToDevice
            )
        );
        _array_neurongroup_2__spikespace[_num__array_neurongroup_2__spikespace - 1] = 0;
                            CUDA_SAFE_CALL(
                                cudaMemcpy(
                                    dev_array_neurongroup_2__spikespace[current_idx_array_neurongroup_2__spikespace] + _num__array_neurongroup_2__spikespace - 1,
                                    &_array_neurongroup_2__spikespace[_num__array_neurongroup_2__spikespace - 1],
                                    sizeof(_array_neurongroup_2__spikespace[_num__array_neurongroup_2__spikespace - 1]),
                                    cudaMemcpyHostToDevice
                                )
                            );
        for(int i=0; i<_num__array_neurongroup_1__spikespace; i++)
        {
            _array_neurongroup_1__spikespace[i] = - 1;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_1__spikespace[current_idx_array_neurongroup_1__spikespace],
                &_array_neurongroup_1__spikespace[0],
                sizeof(_array_neurongroup_1__spikespace[0])*_num__array_neurongroup_1__spikespace,
                cudaMemcpyHostToDevice
            )
        );
        _array_neurongroup_1__spikespace[_num__array_neurongroup_1__spikespace - 1] = 0;
                            CUDA_SAFE_CALL(
                                cudaMemcpy(
                                    dev_array_neurongroup_1__spikespace[current_idx_array_neurongroup_1__spikespace] + _num__array_neurongroup_1__spikespace - 1,
                                    &_array_neurongroup_1__spikespace[_num__array_neurongroup_1__spikespace - 1],
                                    sizeof(_array_neurongroup_1__spikespace[_num__array_neurongroup_1__spikespace - 1]),
                                    cudaMemcpyHostToDevice
                                )
                            );
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _dynamic_array_spikegeneratorgroup_spike_number.resize(213084);
        THRUST_CHECK_ERROR(dev_dynamic_array_spikegeneratorgroup_spike_number.resize(213084));
        for(int i=0; i<_num__static_array__dynamic_array_spikegeneratorgroup_spike_number; i++)
        {
            _dynamic_array_spikegeneratorgroup_spike_number[i] = _static_array__dynamic_array_spikegeneratorgroup_spike_number[i];
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&dev_dynamic_array_spikegeneratorgroup_spike_number[0]),
                &_dynamic_array_spikegeneratorgroup_spike_number[0],
                sizeof(_dynamic_array_spikegeneratorgroup_spike_number[0])*_dynamic_array_spikegeneratorgroup_spike_number.size(),
                cudaMemcpyHostToDevice
            )
        );
        _dynamic_array_spikegeneratorgroup_neuron_index.resize(213084);
        THRUST_CHECK_ERROR(dev_dynamic_array_spikegeneratorgroup_neuron_index.resize(213084));
        for(int i=0; i<_num__static_array__dynamic_array_spikegeneratorgroup_neuron_index; i++)
        {
            _dynamic_array_spikegeneratorgroup_neuron_index[i] = _static_array__dynamic_array_spikegeneratorgroup_neuron_index[i];
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&dev_dynamic_array_spikegeneratorgroup_neuron_index[0]),
                &_dynamic_array_spikegeneratorgroup_neuron_index[0],
                sizeof(_dynamic_array_spikegeneratorgroup_neuron_index[0])*_dynamic_array_spikegeneratorgroup_neuron_index.size(),
                cudaMemcpyHostToDevice
            )
        );
        _dynamic_array_spikegeneratorgroup_spike_time.resize(213084);
        THRUST_CHECK_ERROR(dev_dynamic_array_spikegeneratorgroup_spike_time.resize(213084));
        for(int i=0; i<_num__static_array__dynamic_array_spikegeneratorgroup_spike_time; i++)
        {
            _dynamic_array_spikegeneratorgroup_spike_time[i] = _static_array__dynamic_array_spikegeneratorgroup_spike_time[i];
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&dev_dynamic_array_spikegeneratorgroup_spike_time[0]),
                &_dynamic_array_spikegeneratorgroup_spike_time[0],
                sizeof(_dynamic_array_spikegeneratorgroup_spike_time[0])*_dynamic_array_spikegeneratorgroup_spike_time.size(),
                cudaMemcpyHostToDevice
            )
        );
        _dynamic_array_spikegeneratorgroup__timebins.resize(213084);
        THRUST_CHECK_ERROR(dev_dynamic_array_spikegeneratorgroup__timebins.resize(213084));
        _array_spikegeneratorgroup__lastindex[0] = 0;
                            CUDA_SAFE_CALL(
                                cudaMemcpy(
                                    dev_array_spikegeneratorgroup__lastindex + 0,
                                    &_array_spikegeneratorgroup__lastindex[0],
                                    sizeof(_array_spikegeneratorgroup__lastindex[0]),
                                    cudaMemcpyHostToDevice
                                )
                            );
        _array_spikegeneratorgroup_period[0] = 0.0;
                            CUDA_SAFE_CALL(
                                cudaMemcpy(
                                    dev_array_spikegeneratorgroup_period + 0,
                                    &_array_spikegeneratorgroup_period[0],
                                    sizeof(_array_spikegeneratorgroup_period[0]),
                                    cudaMemcpyHostToDevice
                                )
                            );
        for(int i=0; i<_num__static_array__array_neurongroup_c; i++)
        {
            _array_neurongroup_c[i] = _static_array__array_neurongroup_c[i];
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_c,
                &_array_neurongroup_c[0],
                sizeof(_array_neurongroup_c[0])*_num__array_neurongroup_c,
                cudaMemcpyHostToDevice
            )
        );
        for(int i=0; i<_num__array_neurongroup_1_lastspike; i++)
        {
            _array_neurongroup_1_lastspike[i] = - 10000.0;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_1_lastspike,
                &_array_neurongroup_1_lastspike[0],
                sizeof(_array_neurongroup_1_lastspike[0])*_num__array_neurongroup_1_lastspike,
                cudaMemcpyHostToDevice
            )
        );
        for(int i=0; i<_num__array_neurongroup_1_not_refractory; i++)
        {
            _array_neurongroup_1_not_refractory[i] = true;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_1_not_refractory,
                &_array_neurongroup_1_not_refractory[0],
                sizeof(_array_neurongroup_1_not_refractory[0])*_num__array_neurongroup_1_not_refractory,
                cudaMemcpyHostToDevice
            )
        );
        _dynamic_array_synapses_2_delay.resize(1);
        THRUST_CHECK_ERROR(dev_dynamic_array_synapses_2_delay.resize(1));
        _dynamic_array_synapses_2_delay.resize(1);
        THRUST_CHECK_ERROR(dev_dynamic_array_synapses_2_delay.resize(1));
        _dynamic_array_synapses_2_delay[0] = 0.001;
        _run_synapses_2_synapses_create_generator_codeobject();
        for(int i=0; i<_dynamic_array_synapses_2_w.size(); i++)
        {
            _dynamic_array_synapses_2_w[i] = 0.7;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&dev_dynamic_array_synapses_2_w[0]),
                &_dynamic_array_synapses_2_w[0],
                sizeof(_dynamic_array_synapses_2_w[0])*_dynamic_array_synapses_2_w.size(),
                cudaMemcpyHostToDevice
            )
        );
        _run_synapses_3_synapses_create_generator_codeobject();
        for(int i=0; i<_dynamic_array_synapses_3_w.size(); i++)
        {
            _dynamic_array_synapses_3_w[i] = 2;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&dev_dynamic_array_synapses_3_w[0]),
                &_dynamic_array_synapses_3_w[0],
                sizeof(_dynamic_array_synapses_3_w[0])*_dynamic_array_synapses_3_w.size(),
                cudaMemcpyHostToDevice
            )
        );
        _array_networkoperation_clock_dt[0] = 0.0001;
        _array_networkoperation_clock_dt[0] = 0.0001;
        _array_networkoperation_1_clock_dt[0] = 10.0;
        _array_networkoperation_1_clock_dt[0] = 10.0;
        _array_networkoperation_2_clock_dt[0] = 10.0;
        _array_networkoperation_2_clock_dt[0] = 10.0;
        _array_networkoperation_1_clock_timestep[0] = 0;
        _array_networkoperation_1_clock_t[0] = 0.0;
        _array_networkoperation_2_clock_timestep[0] = 0;
        _array_networkoperation_2_clock_t[0] = 0.0;
        _array_defaultclock_timestep[0] = 0;
        _array_defaultclock_t[0] = 0.0;
        _array_networkoperation_clock_timestep[0] = 0;
        _array_networkoperation_clock_t[0] = 0.0;
        _array_spikegeneratorgroup__lastindex[0] = 0;
                            CUDA_SAFE_CALL(
                                cudaMemcpy(
                                    dev_array_spikegeneratorgroup__lastindex + 0,
                                    &_array_spikegeneratorgroup__lastindex[0],
                                    sizeof(_array_spikegeneratorgroup__lastindex[0]),
                                    cudaMemcpyHostToDevice
                                )
                            );
        for(int i=0; i<_num__static_array__dynamic_array_spikegeneratorgroup__timebins; i++)
        {
            _dynamic_array_spikegeneratorgroup__timebins[i] = _static_array__dynamic_array_spikegeneratorgroup__timebins[i];
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&dev_dynamic_array_spikegeneratorgroup__timebins[0]),
                &_dynamic_array_spikegeneratorgroup__timebins[0],
                sizeof(_dynamic_array_spikegeneratorgroup__timebins[0])*_dynamic_array_spikegeneratorgroup__timebins.size(),
                cudaMemcpyHostToDevice
            )
        );
        _array_spikegeneratorgroup__period_bins[0] = 0.0;
                            CUDA_SAFE_CALL(
                                cudaMemcpy(
                                    dev_array_spikegeneratorgroup__period_bins + 0,
                                    &_array_spikegeneratorgroup__period_bins[0],
                                    sizeof(_array_spikegeneratorgroup__period_bins[0]),
                                    cudaMemcpyHostToDevice
                                )
                            );
        _before_run_synapses_2_pre_push_spikes();
        _before_run_synapses_3_pre_push_spikes();
        _before_run_synapses_pre_push_spikes();
        _before_run_synapses_1_post_push_spikes();
        dev_dynamic_array_synapses_2__synaptic_pre.clear();
        dev_dynamic_array_synapses_2__synaptic_pre.shrink_to_fit();
        dev_dynamic_array_synapses_3__synaptic_pre.clear();
        dev_dynamic_array_synapses_3__synaptic_pre.shrink_to_fit();
        dev_dynamic_array_synapses__synaptic_pre.clear();
        dev_dynamic_array_synapses__synaptic_pre.shrink_to_fit();
        dev_dynamic_array_synapses_delay.clear();
        dev_dynamic_array_synapses_delay.shrink_to_fit();
        dev_dynamic_array_synapses_1_delay.clear();
        dev_dynamic_array_synapses_1_delay.shrink_to_fit();
        dev_dynamic_array_synapses_2_delay.clear();
        dev_dynamic_array_synapses_2_delay.shrink_to_fit();
        dev_dynamic_array_synapses_3_delay.clear();
        dev_dynamic_array_synapses_3_delay.shrink_to_fit();
        magicnetwork.clear();
        magicnetwork.add(&networkoperation_1_clock, _run_random_number_buffer);
        magicnetwork.add(&networkoperation_2_clock, _run_random_number_buffer);
        magicnetwork.add(&defaultclock, _run_random_number_buffer);
        magicnetwork.add(&networkoperation_clock, _run_random_number_buffer);
        magicnetwork.add(&defaultclock, _run_synapses_1_summed_variable_Igap_post_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_1_stateupdater_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_2_stateupdater_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_stateupdater_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_1_spike_thresholder_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_2_spike_thresholder_codeobject);
        magicnetwork.add(&defaultclock, _run_spikegeneratorgroup_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_2_pre_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_2_pre_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_3_pre_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_3_pre_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_pre_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_pre_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_1_post_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_1_post_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_2_spike_resetter_codeobject);
        magicnetwork.add(&networkoperation_1_clock, NULL);
        magicnetwork.add(&networkoperation_2_clock, NULL);
        magicnetwork.add(&networkoperation_clock, NULL);
        magicnetwork.run(500.0, NULL, 10.0);
        random_number_buffer.run_finished();
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        CUDA_SAFE_CALL(cudaProfilerStop());
        _after_run_neurongroup_1_spike_thresholder_codeobject();
        _after_run_neurongroup_2_spike_thresholder_codeobject();
        _debugmsg_synapses_2_pre_codeobject();
        
        _debugmsg_synapses_3_pre_codeobject();
        
        _debugmsg_synapses_pre_codeobject();
        
        _debugmsg_synapses_1_post_codeobject();

    }

    //const double _run_time3 = (double)(std::clock() -_start_time3)/CLOCKS_PER_SEC;
    //printf("INFO: main_lines took %f seconds\n", _run_time3);

        

    brian_end();
        


    // Profiling
    //const double _run_time = (double)(std::clock() -_start_time)/CLOCKS_PER_SEC;
    //printf("INFO: main function took %f seconds\n", _run_time);

    return 0;
}