
#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/cuda_utils.h"
#include "network.h"
#include "rand.h"
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <utility>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand.h>
#include <curand_kernel.h>

size_t brian::used_device_memory = 0;

//////////////// clocks ///////////////////
Clock brian::defaultclock;
Clock brian::networkoperation_1_clock;
Clock brian::networkoperation_2_clock;
Clock brian::networkoperation_clock;

//////////////// networks /////////////////
Network brian::magicnetwork;

//////////////// arrays ///////////////////
double * brian::_array_defaultclock_dt;
double * brian::dev_array_defaultclock_dt;
__device__ double * brian::d_array_defaultclock_dt;
const int brian::_num__array_defaultclock_dt = 1;

double * brian::_array_defaultclock_t;
double * brian::dev_array_defaultclock_t;
__device__ double * brian::d_array_defaultclock_t;
const int brian::_num__array_defaultclock_t = 1;

int64_t * brian::_array_defaultclock_timestep;
int64_t * brian::dev_array_defaultclock_timestep;
__device__ int64_t * brian::d_array_defaultclock_timestep;
const int brian::_num__array_defaultclock_timestep = 1;

double * brian::_array_networkoperation_1_clock_dt;
double * brian::dev_array_networkoperation_1_clock_dt;
__device__ double * brian::d_array_networkoperation_1_clock_dt;
const int brian::_num__array_networkoperation_1_clock_dt = 1;

double * brian::_array_networkoperation_1_clock_t;
double * brian::dev_array_networkoperation_1_clock_t;
__device__ double * brian::d_array_networkoperation_1_clock_t;
const int brian::_num__array_networkoperation_1_clock_t = 1;

int64_t * brian::_array_networkoperation_1_clock_timestep;
int64_t * brian::dev_array_networkoperation_1_clock_timestep;
__device__ int64_t * brian::d_array_networkoperation_1_clock_timestep;
const int brian::_num__array_networkoperation_1_clock_timestep = 1;

double * brian::_array_networkoperation_2_clock_dt;
double * brian::dev_array_networkoperation_2_clock_dt;
__device__ double * brian::d_array_networkoperation_2_clock_dt;
const int brian::_num__array_networkoperation_2_clock_dt = 1;

double * brian::_array_networkoperation_2_clock_t;
double * brian::dev_array_networkoperation_2_clock_t;
__device__ double * brian::d_array_networkoperation_2_clock_t;
const int brian::_num__array_networkoperation_2_clock_t = 1;

int64_t * brian::_array_networkoperation_2_clock_timestep;
int64_t * brian::dev_array_networkoperation_2_clock_timestep;
__device__ int64_t * brian::d_array_networkoperation_2_clock_timestep;
const int brian::_num__array_networkoperation_2_clock_timestep = 1;

double * brian::_array_networkoperation_clock_dt;
double * brian::dev_array_networkoperation_clock_dt;
__device__ double * brian::d_array_networkoperation_clock_dt;
const int brian::_num__array_networkoperation_clock_dt = 1;

double * brian::_array_networkoperation_clock_t;
double * brian::dev_array_networkoperation_clock_t;
__device__ double * brian::d_array_networkoperation_clock_t;
const int brian::_num__array_networkoperation_clock_t = 1;

int64_t * brian::_array_networkoperation_clock_timestep;
int64_t * brian::dev_array_networkoperation_clock_timestep;
__device__ int64_t * brian::d_array_networkoperation_clock_timestep;
const int brian::_num__array_networkoperation_clock_timestep = 1;

int32_t * brian::_array_neurongroup_1_i;
int32_t * brian::dev_array_neurongroup_1_i;
__device__ int32_t * brian::d_array_neurongroup_1_i;
const int brian::_num__array_neurongroup_1_i = 13;

double * brian::_array_neurongroup_1_Igap;
double * brian::dev_array_neurongroup_1_Igap;
__device__ double * brian::d_array_neurongroup_1_Igap;
const int brian::_num__array_neurongroup_1_Igap = 13;

double * brian::_array_neurongroup_1_lastspike;
double * brian::dev_array_neurongroup_1_lastspike;
__device__ double * brian::d_array_neurongroup_1_lastspike;
const int brian::_num__array_neurongroup_1_lastspike = 13;

char * brian::_array_neurongroup_1_not_refractory;
char * brian::dev_array_neurongroup_1_not_refractory;
__device__ char * brian::d_array_neurongroup_1_not_refractory;
const int brian::_num__array_neurongroup_1_not_refractory = 13;

double * brian::_array_neurongroup_1_y;
double * brian::dev_array_neurongroup_1_y;
__device__ double * brian::d_array_neurongroup_1_y;
const int brian::_num__array_neurongroup_1_y = 13;

int32_t * brian::_array_neurongroup_2_i;
int32_t * brian::dev_array_neurongroup_2_i;
__device__ int32_t * brian::d_array_neurongroup_2_i;
const int brian::_num__array_neurongroup_2_i = 13;

double * brian::_array_neurongroup_2_v;
double * brian::dev_array_neurongroup_2_v;
__device__ double * brian::d_array_neurongroup_2_v;
const int brian::_num__array_neurongroup_2_v = 13;

double * brian::_array_neurongroup_apost;
double * brian::dev_array_neurongroup_apost;
__device__ double * brian::d_array_neurongroup_apost;
const int brian::_num__array_neurongroup_apost = 7488;

double * brian::_array_neurongroup_apre;
double * brian::dev_array_neurongroup_apre;
__device__ double * brian::d_array_neurongroup_apre;
const int brian::_num__array_neurongroup_apre = 7488;

double * brian::_array_neurongroup_c;
double * brian::dev_array_neurongroup_c;
__device__ double * brian::d_array_neurongroup_c;
const int brian::_num__array_neurongroup_c = 7488;

int32_t * brian::_array_neurongroup_i;
int32_t * brian::dev_array_neurongroup_i;
__device__ int32_t * brian::d_array_neurongroup_i;
const int brian::_num__array_neurongroup_i = 7488;

double * brian::_array_neurongroup_l_speed;
double * brian::dev_array_neurongroup_l_speed;
__device__ double * brian::d_array_neurongroup_l_speed;
const int brian::_num__array_neurongroup_l_speed = 7488;

double * brian::_array_neurongroup_v;
double * brian::dev_array_neurongroup_v;
__device__ double * brian::d_array_neurongroup_v;
const int brian::_num__array_neurongroup_v = 7488;

int32_t * brian::_array_spikegeneratorgroup__lastindex;
int32_t * brian::dev_array_spikegeneratorgroup__lastindex;
__device__ int32_t * brian::d_array_spikegeneratorgroup__lastindex;
const int brian::_num__array_spikegeneratorgroup__lastindex = 1;

int32_t * brian::_array_spikegeneratorgroup__period_bins;
int32_t * brian::dev_array_spikegeneratorgroup__period_bins;
__device__ int32_t * brian::d_array_spikegeneratorgroup__period_bins;
const int brian::_num__array_spikegeneratorgroup__period_bins = 1;

int32_t * brian::_array_spikegeneratorgroup_i;
int32_t * brian::dev_array_spikegeneratorgroup_i;
__device__ int32_t * brian::d_array_spikegeneratorgroup_i;
const int brian::_num__array_spikegeneratorgroup_i = 576;

double * brian::_array_spikegeneratorgroup_period;
double * brian::dev_array_spikegeneratorgroup_period;
__device__ double * brian::d_array_spikegeneratorgroup_period;
const int brian::_num__array_spikegeneratorgroup_period = 1;

int32_t * brian::_array_synapses_1_N;
int32_t * brian::dev_array_synapses_1_N;
__device__ int32_t * brian::d_array_synapses_1_N;
const int brian::_num__array_synapses_1_N = 1;

int32_t * brian::_array_synapses_2_N;
int32_t * brian::dev_array_synapses_2_N;
__device__ int32_t * brian::d_array_synapses_2_N;
const int brian::_num__array_synapses_2_N = 1;

int32_t * brian::_array_synapses_3_N;
int32_t * brian::dev_array_synapses_3_N;
__device__ int32_t * brian::d_array_synapses_3_N;
const int brian::_num__array_synapses_3_N = 1;

int32_t * brian::_array_synapses_N;
int32_t * brian::dev_array_synapses_N;
__device__ int32_t * brian::d_array_synapses_N;
const int brian::_num__array_synapses_N = 1;


//////////////// eventspaces ///////////////
// we dynamically create multiple eventspaces in no_or_const_delay_mode
// for initiating the first spikespace, we need a host pointer
// for choosing the right spikespace, we need a global index variable
int32_t * brian::_array_neurongroup_1__spikespace;
const int brian::_num__array_neurongroup_1__spikespace = 14;
thrust::host_vector<int32_t*> brian::dev_array_neurongroup_1__spikespace(1);
int brian::current_idx_array_neurongroup_1__spikespace = 0;
int32_t * brian::_array_neurongroup_2__spikespace;
const int brian::_num__array_neurongroup_2__spikespace = 14;
thrust::host_vector<int32_t*> brian::dev_array_neurongroup_2__spikespace(1);
int brian::current_idx_array_neurongroup_2__spikespace = 0;
int32_t * brian::_array_spikegeneratorgroup__spikespace;
const int brian::_num__array_spikegeneratorgroup__spikespace = 577;
thrust::host_vector<int32_t*> brian::dev_array_spikegeneratorgroup__spikespace(1);
int brian::current_idx_array_spikegeneratorgroup__spikespace = 0;
int brian::previous_idx_array_spikegeneratorgroup__spikespace;

//////////////// dynamic arrays 1d /////////
thrust::host_vector<int32_t> brian::_dynamic_array_spikegeneratorgroup__timebins;
thrust::device_vector<int32_t> brian::dev_dynamic_array_spikegeneratorgroup__timebins;
thrust::host_vector<int32_t> brian::_dynamic_array_spikegeneratorgroup_neuron_index;
thrust::device_vector<int32_t> brian::dev_dynamic_array_spikegeneratorgroup_neuron_index;
thrust::host_vector<int32_t> brian::_dynamic_array_spikegeneratorgroup_spike_number;
thrust::device_vector<int32_t> brian::dev_dynamic_array_spikegeneratorgroup_spike_number;
thrust::host_vector<double> brian::_dynamic_array_spikegeneratorgroup_spike_time;
thrust::device_vector<double> brian::dev_dynamic_array_spikegeneratorgroup_spike_time;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_1__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_1__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_1__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_1__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_1_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_1_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_1_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_1_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_1_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_1_N_outgoing;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_2__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_2__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_2__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_2__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_2_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_2_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_2_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_2_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_2_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_2_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_2_w;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_2_w;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_3__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_3__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_3__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_3__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_3_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_3_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_3_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_3_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_3_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_3_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_3_w;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_3_w;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_w;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_w;

//////////////// dynamic arrays 2d /////////

/////////////// static arrays /////////////
double * brian::_static_array__array_neurongroup_c;
double * brian::dev_static_array__array_neurongroup_c;
__device__ double * brian::d_static_array__array_neurongroup_c;
const int brian::_num__static_array__array_neurongroup_c = 7488;
int32_t * brian::_static_array__dynamic_array_spikegeneratorgroup__timebins;
int32_t * brian::dev_static_array__dynamic_array_spikegeneratorgroup__timebins;
__device__ int32_t * brian::d_static_array__dynamic_array_spikegeneratorgroup__timebins;
const int brian::_num__static_array__dynamic_array_spikegeneratorgroup__timebins = 213084;
int64_t * brian::_static_array__dynamic_array_spikegeneratorgroup_neuron_index;
int64_t * brian::dev_static_array__dynamic_array_spikegeneratorgroup_neuron_index;
__device__ int64_t * brian::d_static_array__dynamic_array_spikegeneratorgroup_neuron_index;
const int brian::_num__static_array__dynamic_array_spikegeneratorgroup_neuron_index = 213084;
int32_t * brian::_static_array__dynamic_array_spikegeneratorgroup_spike_number;
int32_t * brian::dev_static_array__dynamic_array_spikegeneratorgroup_spike_number;
__device__ int32_t * brian::d_static_array__dynamic_array_spikegeneratorgroup_spike_number;
const int brian::_num__static_array__dynamic_array_spikegeneratorgroup_spike_number = 213084;
double * brian::_static_array__dynamic_array_spikegeneratorgroup_spike_time;
double * brian::dev_static_array__dynamic_array_spikegeneratorgroup_spike_time;
__device__ double * brian::d_static_array__dynamic_array_spikegeneratorgroup_spike_time;
const int brian::_num__static_array__dynamic_array_spikegeneratorgroup_spike_time = 213084;

//////////////// synapses /////////////////
// synapses
int32_t synapses_source_start_index;
int32_t synapses_source_stop_index;
bool brian::synapses_multiple_pre_post = false;
// synapses_pre
__device__ int* brian::synapses_pre_num_synapses_by_pre;
__device__ int* brian::synapses_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_pre_unique_delays;
__device__ int* brian::synapses_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_pre_global_bundle_id_start_by_pre;
int brian::synapses_pre_bundle_size_max = 0;
int brian::synapses_pre_bundle_size_min = 0;
double brian::synapses_pre_bundle_size_mean = 0;
double brian::synapses_pre_bundle_size_std = 0;
int brian::synapses_pre_max_size = 0;
__device__ int* brian::synapses_pre_num_unique_delays_by_pre;
int brian::synapses_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_pre_synapse_ids;
__device__ int* brian::synapses_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_pre;
int brian::synapses_pre_eventspace_idx = 0;
int brian::synapses_pre_delay;
bool brian::synapses_pre_scalar_delay;
// synapses_1
int32_t synapses_1_source_start_index;
int32_t synapses_1_source_stop_index;
bool brian::synapses_1_multiple_pre_post = false;
// synapses_1_post
__device__ int* brian::synapses_1_post_num_synapses_by_pre;
__device__ int* brian::synapses_1_post_num_synapses_by_bundle;
__device__ int* brian::synapses_1_post_unique_delays;
__device__ int* brian::synapses_1_post_synapses_offset_by_bundle;
__device__ int* brian::synapses_1_post_global_bundle_id_start_by_pre;
int brian::synapses_1_post_bundle_size_max = 0;
int brian::synapses_1_post_bundle_size_min = 0;
double brian::synapses_1_post_bundle_size_mean = 0;
double brian::synapses_1_post_bundle_size_std = 0;
int brian::synapses_1_post_max_size = 0;
__device__ int* brian::synapses_1_post_num_unique_delays_by_pre;
int brian::synapses_1_post_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_1_post_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_1_post_synapse_ids;
__device__ int* brian::synapses_1_post_unique_delay_start_idcs;
__device__ int* brian::synapses_1_post_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_1_post;
int brian::synapses_1_post_eventspace_idx = 0;
int brian::synapses_1_post_delay;
bool brian::synapses_1_post_scalar_delay;
// synapses_2
int32_t synapses_2_source_start_index;
int32_t synapses_2_source_stop_index;
bool brian::synapses_2_multiple_pre_post = false;
// synapses_2_pre
__device__ int* brian::synapses_2_pre_num_synapses_by_pre;
__device__ int* brian::synapses_2_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_2_pre_unique_delays;
__device__ int* brian::synapses_2_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_2_pre_global_bundle_id_start_by_pre;
int brian::synapses_2_pre_bundle_size_max = 0;
int brian::synapses_2_pre_bundle_size_min = 0;
double brian::synapses_2_pre_bundle_size_mean = 0;
double brian::synapses_2_pre_bundle_size_std = 0;
int brian::synapses_2_pre_max_size = 0;
__device__ int* brian::synapses_2_pre_num_unique_delays_by_pre;
int brian::synapses_2_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_2_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_2_pre_synapse_ids;
__device__ int* brian::synapses_2_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_2_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_2_pre;
int brian::synapses_2_pre_eventspace_idx = 0;
int brian::synapses_2_pre_delay;
bool brian::synapses_2_pre_scalar_delay;
// synapses_3
int32_t synapses_3_source_start_index;
int32_t synapses_3_source_stop_index;
bool brian::synapses_3_multiple_pre_post = false;
// synapses_3_pre
__device__ int* brian::synapses_3_pre_num_synapses_by_pre;
__device__ int* brian::synapses_3_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_3_pre_unique_delays;
__device__ int* brian::synapses_3_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_3_pre_global_bundle_id_start_by_pre;
int brian::synapses_3_pre_bundle_size_max = 0;
int brian::synapses_3_pre_bundle_size_min = 0;
double brian::synapses_3_pre_bundle_size_mean = 0;
double brian::synapses_3_pre_bundle_size_std = 0;
int brian::synapses_3_pre_max_size = 0;
__device__ int* brian::synapses_3_pre_num_unique_delays_by_pre;
int brian::synapses_3_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_3_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_3_pre_synapse_ids;
__device__ int* brian::synapses_3_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_3_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_3_pre;
int brian::synapses_3_pre_eventspace_idx = 0;
int brian::synapses_3_pre_delay;
bool brian::synapses_3_pre_scalar_delay;

int brian::num_parallel_blocks;
int brian::max_threads_per_block;
int brian::max_threads_per_sm;
int brian::max_shared_mem_size;
int brian::num_threads_per_warp;

__global__ void synapses_pre_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_pre.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}
__global__ void synapses_1_post_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_1_post.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}
__global__ void synapses_2_pre_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_2_pre.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}
__global__ void synapses_3_pre_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_3_pre.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}

// Profiling information for each code object

//////////////random numbers//////////////////
curandGenerator_t brian::curand_generator;
__device__ unsigned long long* brian::d_curand_seed;
unsigned long long* brian::dev_curand_seed;
// dev_{co.name}_{rng_type}_allocator
//      pointer to start of generated random numbers array
//      at each generation cycle this array is refilled
// dev_{co.name}_{rng_type}
//      pointer moving through generated random number array
//      until it is regenerated at the next generation cycle
curandState* brian::dev_curand_states;
__device__ curandState* brian::d_curand_states;
RandomNumberBuffer brian::random_number_buffer;

void _init_arrays()
{
    using namespace brian;

    std::clock_t start_timer = std::clock();

    CUDA_CHECK_MEMORY();
    size_t used_device_memory_start = used_device_memory;

    cudaDeviceProp props;
    CUDA_SAFE_CALL(
            cudaGetDeviceProperties(&props, 0)
            );

    num_parallel_blocks = 1;
    max_threads_per_block = props.maxThreadsPerBlock;
    max_threads_per_sm = props.maxThreadsPerMultiProcessor;
    max_shared_mem_size = props.sharedMemPerBlock;
    num_threads_per_warp = props.warpSize;

    // Random seeds might be overwritten in main.cu
    unsigned long long seed = time(0);

    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_curand_seed,
                sizeof(unsigned long long))
            );

    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_curand_seed, &dev_curand_seed,
                sizeof(unsigned long long*))
            );

    CUDA_SAFE_CALL(
            curandCreateGenerator(&curand_generator, CURAND_RNG_PSEUDO_DEFAULT)
            );


    // this sets seed for host and device api RNG
    random_number_buffer.set_seed(seed);

    synapses_pre_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            576
            );
    CUDA_CHECK_ERROR("synapses_pre_init");
    synapses_1_post_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1__synaptic_post[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1__synaptic_pre[0]),
            0,  //was dt, maybe irrelevant?
            0,
            13
            );
    CUDA_CHECK_ERROR("synapses_1_post_init");
    synapses_2_pre_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_2__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_2__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            13
            );
    CUDA_CHECK_ERROR("synapses_2_pre_init");
    synapses_3_pre_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_3__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_3__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            13
            );
    CUDA_CHECK_ERROR("synapses_3_pre_init");

    // Arrays initialized to 0
            _array_defaultclock_dt = new double[1];
            for(int i=0; i<1; i++) _array_defaultclock_dt[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_defaultclock_dt, sizeof(double)*_num__array_defaultclock_dt)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_defaultclock_dt, _array_defaultclock_dt, sizeof(double)*_num__array_defaultclock_dt, cudaMemcpyHostToDevice)
                    );
            _array_defaultclock_t = new double[1];
            for(int i=0; i<1; i++) _array_defaultclock_t[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_defaultclock_t, sizeof(double)*_num__array_defaultclock_t)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_defaultclock_t, _array_defaultclock_t, sizeof(double)*_num__array_defaultclock_t, cudaMemcpyHostToDevice)
                    );
            _array_defaultclock_timestep = new int64_t[1];
            for(int i=0; i<1; i++) _array_defaultclock_timestep[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_defaultclock_timestep, sizeof(int64_t)*_num__array_defaultclock_timestep)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_defaultclock_timestep, _array_defaultclock_timestep, sizeof(int64_t)*_num__array_defaultclock_timestep, cudaMemcpyHostToDevice)
                    );
            _array_networkoperation_1_clock_dt = new double[1];
            for(int i=0; i<1; i++) _array_networkoperation_1_clock_dt[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_networkoperation_1_clock_dt, sizeof(double)*_num__array_networkoperation_1_clock_dt)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_networkoperation_1_clock_dt, _array_networkoperation_1_clock_dt, sizeof(double)*_num__array_networkoperation_1_clock_dt, cudaMemcpyHostToDevice)
                    );
            _array_networkoperation_1_clock_t = new double[1];
            for(int i=0; i<1; i++) _array_networkoperation_1_clock_t[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_networkoperation_1_clock_t, sizeof(double)*_num__array_networkoperation_1_clock_t)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_networkoperation_1_clock_t, _array_networkoperation_1_clock_t, sizeof(double)*_num__array_networkoperation_1_clock_t, cudaMemcpyHostToDevice)
                    );
            _array_networkoperation_1_clock_timestep = new int64_t[1];
            for(int i=0; i<1; i++) _array_networkoperation_1_clock_timestep[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_networkoperation_1_clock_timestep, sizeof(int64_t)*_num__array_networkoperation_1_clock_timestep)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_networkoperation_1_clock_timestep, _array_networkoperation_1_clock_timestep, sizeof(int64_t)*_num__array_networkoperation_1_clock_timestep, cudaMemcpyHostToDevice)
                    );
            _array_networkoperation_2_clock_dt = new double[1];
            for(int i=0; i<1; i++) _array_networkoperation_2_clock_dt[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_networkoperation_2_clock_dt, sizeof(double)*_num__array_networkoperation_2_clock_dt)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_networkoperation_2_clock_dt, _array_networkoperation_2_clock_dt, sizeof(double)*_num__array_networkoperation_2_clock_dt, cudaMemcpyHostToDevice)
                    );
            _array_networkoperation_2_clock_t = new double[1];
            for(int i=0; i<1; i++) _array_networkoperation_2_clock_t[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_networkoperation_2_clock_t, sizeof(double)*_num__array_networkoperation_2_clock_t)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_networkoperation_2_clock_t, _array_networkoperation_2_clock_t, sizeof(double)*_num__array_networkoperation_2_clock_t, cudaMemcpyHostToDevice)
                    );
            _array_networkoperation_2_clock_timestep = new int64_t[1];
            for(int i=0; i<1; i++) _array_networkoperation_2_clock_timestep[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_networkoperation_2_clock_timestep, sizeof(int64_t)*_num__array_networkoperation_2_clock_timestep)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_networkoperation_2_clock_timestep, _array_networkoperation_2_clock_timestep, sizeof(int64_t)*_num__array_networkoperation_2_clock_timestep, cudaMemcpyHostToDevice)
                    );
            _array_networkoperation_clock_dt = new double[1];
            for(int i=0; i<1; i++) _array_networkoperation_clock_dt[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_networkoperation_clock_dt, sizeof(double)*_num__array_networkoperation_clock_dt)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_networkoperation_clock_dt, _array_networkoperation_clock_dt, sizeof(double)*_num__array_networkoperation_clock_dt, cudaMemcpyHostToDevice)
                    );
            _array_networkoperation_clock_t = new double[1];
            for(int i=0; i<1; i++) _array_networkoperation_clock_t[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_networkoperation_clock_t, sizeof(double)*_num__array_networkoperation_clock_t)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_networkoperation_clock_t, _array_networkoperation_clock_t, sizeof(double)*_num__array_networkoperation_clock_t, cudaMemcpyHostToDevice)
                    );
            _array_networkoperation_clock_timestep = new int64_t[1];
            for(int i=0; i<1; i++) _array_networkoperation_clock_timestep[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_networkoperation_clock_timestep, sizeof(int64_t)*_num__array_networkoperation_clock_timestep)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_networkoperation_clock_timestep, _array_networkoperation_clock_timestep, sizeof(int64_t)*_num__array_networkoperation_clock_timestep, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_i = new int32_t[13];
            for(int i=0; i<13; i++) _array_neurongroup_1_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_i, sizeof(int32_t)*_num__array_neurongroup_1_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_i, _array_neurongroup_1_i, sizeof(int32_t)*_num__array_neurongroup_1_i, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_Igap = new double[13];
            for(int i=0; i<13; i++) _array_neurongroup_1_Igap[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_Igap, sizeof(double)*_num__array_neurongroup_1_Igap)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_Igap, _array_neurongroup_1_Igap, sizeof(double)*_num__array_neurongroup_1_Igap, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_lastspike = new double[13];
            for(int i=0; i<13; i++) _array_neurongroup_1_lastspike[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_lastspike, sizeof(double)*_num__array_neurongroup_1_lastspike)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_lastspike, _array_neurongroup_1_lastspike, sizeof(double)*_num__array_neurongroup_1_lastspike, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_not_refractory = new char[13];
            for(int i=0; i<13; i++) _array_neurongroup_1_not_refractory[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_not_refractory, sizeof(char)*_num__array_neurongroup_1_not_refractory)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_not_refractory, _array_neurongroup_1_not_refractory, sizeof(char)*_num__array_neurongroup_1_not_refractory, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_y = new double[13];
            for(int i=0; i<13; i++) _array_neurongroup_1_y[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_y, sizeof(double)*_num__array_neurongroup_1_y)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_y, _array_neurongroup_1_y, sizeof(double)*_num__array_neurongroup_1_y, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_2_i = new int32_t[13];
            for(int i=0; i<13; i++) _array_neurongroup_2_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_2_i, sizeof(int32_t)*_num__array_neurongroup_2_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_2_i, _array_neurongroup_2_i, sizeof(int32_t)*_num__array_neurongroup_2_i, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_2_v = new double[13];
            for(int i=0; i<13; i++) _array_neurongroup_2_v[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_2_v, sizeof(double)*_num__array_neurongroup_2_v)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_2_v, _array_neurongroup_2_v, sizeof(double)*_num__array_neurongroup_2_v, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_apost = new double[7488];
            for(int i=0; i<7488; i++) _array_neurongroup_apost[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_apost, sizeof(double)*_num__array_neurongroup_apost)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_apost, _array_neurongroup_apost, sizeof(double)*_num__array_neurongroup_apost, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_apre = new double[7488];
            for(int i=0; i<7488; i++) _array_neurongroup_apre[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_apre, sizeof(double)*_num__array_neurongroup_apre)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_apre, _array_neurongroup_apre, sizeof(double)*_num__array_neurongroup_apre, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_c = new double[7488];
            for(int i=0; i<7488; i++) _array_neurongroup_c[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_c, sizeof(double)*_num__array_neurongroup_c)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_c, _array_neurongroup_c, sizeof(double)*_num__array_neurongroup_c, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_i = new int32_t[7488];
            for(int i=0; i<7488; i++) _array_neurongroup_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_i, _array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_l_speed = new double[7488];
            for(int i=0; i<7488; i++) _array_neurongroup_l_speed[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_l_speed, sizeof(double)*_num__array_neurongroup_l_speed)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_l_speed, _array_neurongroup_l_speed, sizeof(double)*_num__array_neurongroup_l_speed, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_v = new double[7488];
            for(int i=0; i<7488; i++) _array_neurongroup_v[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_v, sizeof(double)*_num__array_neurongroup_v)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_v, _array_neurongroup_v, sizeof(double)*_num__array_neurongroup_v, cudaMemcpyHostToDevice)
                    );
            _array_spikegeneratorgroup__lastindex = new int32_t[1];
            for(int i=0; i<1; i++) _array_spikegeneratorgroup__lastindex[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikegeneratorgroup__lastindex, sizeof(int32_t)*_num__array_spikegeneratorgroup__lastindex)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikegeneratorgroup__lastindex, _array_spikegeneratorgroup__lastindex, sizeof(int32_t)*_num__array_spikegeneratorgroup__lastindex, cudaMemcpyHostToDevice)
                    );
            _array_spikegeneratorgroup__period_bins = new int32_t[1];
            for(int i=0; i<1; i++) _array_spikegeneratorgroup__period_bins[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikegeneratorgroup__period_bins, sizeof(int32_t)*_num__array_spikegeneratorgroup__period_bins)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikegeneratorgroup__period_bins, _array_spikegeneratorgroup__period_bins, sizeof(int32_t)*_num__array_spikegeneratorgroup__period_bins, cudaMemcpyHostToDevice)
                    );
            _array_spikegeneratorgroup_i = new int32_t[576];
            for(int i=0; i<576; i++) _array_spikegeneratorgroup_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikegeneratorgroup_i, sizeof(int32_t)*_num__array_spikegeneratorgroup_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikegeneratorgroup_i, _array_spikegeneratorgroup_i, sizeof(int32_t)*_num__array_spikegeneratorgroup_i, cudaMemcpyHostToDevice)
                    );
            _array_spikegeneratorgroup_period = new double[1];
            for(int i=0; i<1; i++) _array_spikegeneratorgroup_period[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikegeneratorgroup_period, sizeof(double)*_num__array_spikegeneratorgroup_period)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikegeneratorgroup_period, _array_spikegeneratorgroup_period, sizeof(double)*_num__array_spikegeneratorgroup_period, cudaMemcpyHostToDevice)
                    );
            _array_synapses_1_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_1_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_1_N, sizeof(int32_t)*_num__array_synapses_1_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_1_N, _array_synapses_1_N, sizeof(int32_t)*_num__array_synapses_1_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_2_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_2_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_2_N, sizeof(int32_t)*_num__array_synapses_2_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_2_N, _array_synapses_2_N, sizeof(int32_t)*_num__array_synapses_2_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_3_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_3_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_3_N, sizeof(int32_t)*_num__array_synapses_3_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_3_N, _array_synapses_3_N, sizeof(int32_t)*_num__array_synapses_3_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_N, sizeof(int32_t)*_num__array_synapses_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_N, _array_synapses_N, sizeof(int32_t)*_num__array_synapses_N, cudaMemcpyHostToDevice)
                    );
            _dynamic_array_spikegeneratorgroup__timebins.resize(213084);
            THRUST_CHECK_ERROR(dev_dynamic_array_spikegeneratorgroup__timebins.resize(213084));
            for(int i=0; i<213084; i++)
            {
                _dynamic_array_spikegeneratorgroup__timebins[i] = 0;
                dev_dynamic_array_spikegeneratorgroup__timebins[i] = 0;
            }
            _dynamic_array_synapses_2_delay.resize(1);
            THRUST_CHECK_ERROR(dev_dynamic_array_synapses_2_delay.resize(1));
            for(int i=0; i<1; i++)
            {
                _dynamic_array_synapses_2_delay[i] = 0;
                dev_dynamic_array_synapses_2_delay[i] = 0;
            }

    // Arrays initialized to an "arange"
    _array_neurongroup_1_i = new int32_t[13];
    for(int i=0; i<13; i++) _array_neurongroup_1_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_1_i, sizeof(int32_t)*_num__array_neurongroup_1_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_neurongroup_1_i, _array_neurongroup_1_i, sizeof(int32_t)*_num__array_neurongroup_1_i, cudaMemcpyHostToDevice)
            );
    _array_neurongroup_2_i = new int32_t[13];
    for(int i=0; i<13; i++) _array_neurongroup_2_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_2_i, sizeof(int32_t)*_num__array_neurongroup_2_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_neurongroup_2_i, _array_neurongroup_2_i, sizeof(int32_t)*_num__array_neurongroup_2_i, cudaMemcpyHostToDevice)
            );
    _array_neurongroup_i = new int32_t[7488];
    for(int i=0; i<7488; i++) _array_neurongroup_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_neurongroup_i, _array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyHostToDevice)
            );
    _array_spikegeneratorgroup_i = new int32_t[576];
    for(int i=0; i<576; i++) _array_spikegeneratorgroup_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_spikegeneratorgroup_i, sizeof(int32_t)*_num__array_spikegeneratorgroup_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_spikegeneratorgroup_i, _array_spikegeneratorgroup_i, sizeof(int32_t)*_num__array_spikegeneratorgroup_i, cudaMemcpyHostToDevice)
            );

    // static arrays
    _static_array__array_neurongroup_c = new double[7488];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_neurongroup_c, sizeof(double)*7488)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_neurongroup_c, &dev_static_array__array_neurongroup_c, sizeof(double*))
            );
    _static_array__dynamic_array_spikegeneratorgroup__timebins = new int32_t[213084];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__dynamic_array_spikegeneratorgroup__timebins, sizeof(int32_t)*213084)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__dynamic_array_spikegeneratorgroup__timebins, &dev_static_array__dynamic_array_spikegeneratorgroup__timebins, sizeof(int32_t*))
            );
    _static_array__dynamic_array_spikegeneratorgroup_neuron_index = new int64_t[213084];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__dynamic_array_spikegeneratorgroup_neuron_index, sizeof(int64_t)*213084)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__dynamic_array_spikegeneratorgroup_neuron_index, &dev_static_array__dynamic_array_spikegeneratorgroup_neuron_index, sizeof(int64_t*))
            );
    _static_array__dynamic_array_spikegeneratorgroup_spike_number = new int32_t[213084];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__dynamic_array_spikegeneratorgroup_spike_number, sizeof(int32_t)*213084)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__dynamic_array_spikegeneratorgroup_spike_number, &dev_static_array__dynamic_array_spikegeneratorgroup_spike_number, sizeof(int32_t*))
            );
    _static_array__dynamic_array_spikegeneratorgroup_spike_time = new double[213084];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__dynamic_array_spikegeneratorgroup_spike_time, sizeof(double)*213084)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__dynamic_array_spikegeneratorgroup_spike_time, &dev_static_array__dynamic_array_spikegeneratorgroup_spike_time, sizeof(double*))
            );


    // eventspace_arrays
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_1__spikespace[0], sizeof(int32_t)*_num__array_neurongroup_1__spikespace)
            );
    // initialize eventspace with -1
    _array_neurongroup_1__spikespace = new int32_t[14];
    for (int i=0; i<14-1; i++)
    {
        _array_neurongroup_1__spikespace[i] = -1;
    }
    // initialize eventspace counter with 0
    _array_neurongroup_1__spikespace[14 - 1] = 0;
    CUDA_SAFE_CALL(
        cudaMemcpy(
            dev_array_neurongroup_1__spikespace[0],
            _array_neurongroup_1__spikespace,
            sizeof(int32_t) * _num__array_neurongroup_1__spikespace,
            cudaMemcpyHostToDevice
        )
    );
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_2__spikespace[0], sizeof(int32_t)*_num__array_neurongroup_2__spikespace)
            );
    // initialize eventspace with -1
    _array_neurongroup_2__spikespace = new int32_t[14];
    for (int i=0; i<14-1; i++)
    {
        _array_neurongroup_2__spikespace[i] = -1;
    }
    // initialize eventspace counter with 0
    _array_neurongroup_2__spikespace[14 - 1] = 0;
    CUDA_SAFE_CALL(
        cudaMemcpy(
            dev_array_neurongroup_2__spikespace[0],
            _array_neurongroup_2__spikespace,
            sizeof(int32_t) * _num__array_neurongroup_2__spikespace,
            cudaMemcpyHostToDevice
        )
    );
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_spikegeneratorgroup__spikespace[0], sizeof(int32_t)*_num__array_spikegeneratorgroup__spikespace)
            );
    // initialize eventspace with -1
    _array_spikegeneratorgroup__spikespace = new int32_t[577];
    for (int i=0; i<577-1; i++)
    {
        _array_spikegeneratorgroup__spikespace[i] = -1;
    }
    // initialize eventspace counter with 0
    _array_spikegeneratorgroup__spikespace[577 - 1] = 0;
    CUDA_SAFE_CALL(
        cudaMemcpy(
            dev_array_spikegeneratorgroup__spikespace[0],
            _array_spikegeneratorgroup__spikespace,
            sizeof(int32_t) * _num__array_spikegeneratorgroup__spikespace,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK_MEMORY();
    const double to_MB = 1.0 / (1024.0 * 1024.0);
    double tot_memory_MB = (used_device_memory - used_device_memory_start) * to_MB;
    double time_passed = (double)(std::clock() - start_timer) / CLOCKS_PER_SEC;
    std::cout << "INFO: _init_arrays() took " <<  time_passed << "s";
    if (tot_memory_MB > 0)
        std::cout << " and used " << tot_memory_MB << "MB of device memory.";
    std::cout << std::endl;
}

void _load_arrays()
{
    using namespace brian;

    ifstream f_static_array__array_neurongroup_c;
    f_static_array__array_neurongroup_c.open("static_arrays/_static_array__array_neurongroup_c", ios::in | ios::binary);
    if(f_static_array__array_neurongroup_c.is_open())
    {
        f_static_array__array_neurongroup_c.read(reinterpret_cast<char*>(_static_array__array_neurongroup_c), 7488*sizeof(double));
    } else
    {
        std::cout << "Error opening static array _static_array__array_neurongroup_c." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_neurongroup_c, _static_array__array_neurongroup_c, sizeof(double)*7488, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__dynamic_array_spikegeneratorgroup__timebins;
    f_static_array__dynamic_array_spikegeneratorgroup__timebins.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup__timebins", ios::in | ios::binary);
    if(f_static_array__dynamic_array_spikegeneratorgroup__timebins.is_open())
    {
        f_static_array__dynamic_array_spikegeneratorgroup__timebins.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup__timebins), 213084*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup__timebins." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__dynamic_array_spikegeneratorgroup__timebins, _static_array__dynamic_array_spikegeneratorgroup__timebins, sizeof(int32_t)*213084, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__dynamic_array_spikegeneratorgroup_neuron_index;
    f_static_array__dynamic_array_spikegeneratorgroup_neuron_index.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup_neuron_index", ios::in | ios::binary);
    if(f_static_array__dynamic_array_spikegeneratorgroup_neuron_index.is_open())
    {
        f_static_array__dynamic_array_spikegeneratorgroup_neuron_index.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup_neuron_index), 213084*sizeof(int64_t));
    } else
    {
        std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup_neuron_index." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__dynamic_array_spikegeneratorgroup_neuron_index, _static_array__dynamic_array_spikegeneratorgroup_neuron_index, sizeof(int64_t)*213084, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__dynamic_array_spikegeneratorgroup_spike_number;
    f_static_array__dynamic_array_spikegeneratorgroup_spike_number.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup_spike_number", ios::in | ios::binary);
    if(f_static_array__dynamic_array_spikegeneratorgroup_spike_number.is_open())
    {
        f_static_array__dynamic_array_spikegeneratorgroup_spike_number.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup_spike_number), 213084*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup_spike_number." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__dynamic_array_spikegeneratorgroup_spike_number, _static_array__dynamic_array_spikegeneratorgroup_spike_number, sizeof(int32_t)*213084, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__dynamic_array_spikegeneratorgroup_spike_time;
    f_static_array__dynamic_array_spikegeneratorgroup_spike_time.open("static_arrays/_static_array__dynamic_array_spikegeneratorgroup_spike_time", ios::in | ios::binary);
    if(f_static_array__dynamic_array_spikegeneratorgroup_spike_time.is_open())
    {
        f_static_array__dynamic_array_spikegeneratorgroup_spike_time.read(reinterpret_cast<char*>(_static_array__dynamic_array_spikegeneratorgroup_spike_time), 213084*sizeof(double));
    } else
    {
        std::cout << "Error opening static array _static_array__dynamic_array_spikegeneratorgroup_spike_time." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__dynamic_array_spikegeneratorgroup_spike_time, _static_array__dynamic_array_spikegeneratorgroup_spike_time, sizeof(double)*213084, cudaMemcpyHostToDevice)
            );
}

void _write_arrays()
{
    using namespace brian;

    ofstream outfile__array_defaultclock_dt;
    outfile__array_defaultclock_dt.open("results\\_array_defaultclock_dt_1978099143", ios::binary | ios::out);
    if(outfile__array_defaultclock_dt.is_open())
    {
        outfile__array_defaultclock_dt.write(reinterpret_cast<char*>(_array_defaultclock_dt), 1*sizeof(double));
        outfile__array_defaultclock_dt.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_dt." << endl;
    }
    ofstream outfile__array_defaultclock_t;
    outfile__array_defaultclock_t.open("results\\_array_defaultclock_t_2669362164", ios::binary | ios::out);
    if(outfile__array_defaultclock_t.is_open())
    {
        outfile__array_defaultclock_t.write(reinterpret_cast<char*>(_array_defaultclock_t), 1*sizeof(double));
        outfile__array_defaultclock_t.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_t." << endl;
    }
    ofstream outfile__array_defaultclock_timestep;
    outfile__array_defaultclock_timestep.open("results\\_array_defaultclock_timestep_144223508", ios::binary | ios::out);
    if(outfile__array_defaultclock_timestep.is_open())
    {
        outfile__array_defaultclock_timestep.write(reinterpret_cast<char*>(_array_defaultclock_timestep), 1*sizeof(int64_t));
        outfile__array_defaultclock_timestep.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_timestep." << endl;
    }
    ofstream outfile__array_networkoperation_1_clock_dt;
    outfile__array_networkoperation_1_clock_dt.open("results\\_array_networkoperation_1_clock_dt_2362319888", ios::binary | ios::out);
    if(outfile__array_networkoperation_1_clock_dt.is_open())
    {
        outfile__array_networkoperation_1_clock_dt.write(reinterpret_cast<char*>(_array_networkoperation_1_clock_dt), 1*sizeof(double));
        outfile__array_networkoperation_1_clock_dt.close();
    } else
    {
        std::cout << "Error writing output file for _array_networkoperation_1_clock_dt." << endl;
    }
    ofstream outfile__array_networkoperation_1_clock_t;
    outfile__array_networkoperation_1_clock_t.open("results\\_array_networkoperation_1_clock_t_260408168", ios::binary | ios::out);
    if(outfile__array_networkoperation_1_clock_t.is_open())
    {
        outfile__array_networkoperation_1_clock_t.write(reinterpret_cast<char*>(_array_networkoperation_1_clock_t), 1*sizeof(double));
        outfile__array_networkoperation_1_clock_t.close();
    } else
    {
        std::cout << "Error writing output file for _array_networkoperation_1_clock_t." << endl;
    }
    ofstream outfile__array_networkoperation_1_clock_timestep;
    outfile__array_networkoperation_1_clock_timestep.open("results\\_array_networkoperation_1_clock_timestep_85940369", ios::binary | ios::out);
    if(outfile__array_networkoperation_1_clock_timestep.is_open())
    {
        outfile__array_networkoperation_1_clock_timestep.write(reinterpret_cast<char*>(_array_networkoperation_1_clock_timestep), 1*sizeof(int64_t));
        outfile__array_networkoperation_1_clock_timestep.close();
    } else
    {
        std::cout << "Error writing output file for _array_networkoperation_1_clock_timestep." << endl;
    }
    ofstream outfile__array_networkoperation_2_clock_dt;
    outfile__array_networkoperation_2_clock_dt.open("results\\_array_networkoperation_2_clock_dt_1744412435", ios::binary | ios::out);
    if(outfile__array_networkoperation_2_clock_dt.is_open())
    {
        outfile__array_networkoperation_2_clock_dt.write(reinterpret_cast<char*>(_array_networkoperation_2_clock_dt), 1*sizeof(double));
        outfile__array_networkoperation_2_clock_dt.close();
    } else
    {
        std::cout << "Error writing output file for _array_networkoperation_2_clock_dt." << endl;
    }
    ofstream outfile__array_networkoperation_2_clock_t;
    outfile__array_networkoperation_2_clock_t.open("results\\_array_networkoperation_2_clock_t_906543021", ios::binary | ios::out);
    if(outfile__array_networkoperation_2_clock_t.is_open())
    {
        outfile__array_networkoperation_2_clock_t.write(reinterpret_cast<char*>(_array_networkoperation_2_clock_t), 1*sizeof(double));
        outfile__array_networkoperation_2_clock_t.close();
    } else
    {
        std::cout << "Error writing output file for _array_networkoperation_2_clock_t." << endl;
    }
    ofstream outfile__array_networkoperation_2_clock_timestep;
    outfile__array_networkoperation_2_clock_timestep.open("results\\_array_networkoperation_2_clock_timestep_752348259", ios::binary | ios::out);
    if(outfile__array_networkoperation_2_clock_timestep.is_open())
    {
        outfile__array_networkoperation_2_clock_timestep.write(reinterpret_cast<char*>(_array_networkoperation_2_clock_timestep), 1*sizeof(int64_t));
        outfile__array_networkoperation_2_clock_timestep.close();
    } else
    {
        std::cout << "Error writing output file for _array_networkoperation_2_clock_timestep." << endl;
    }
    ofstream outfile__array_networkoperation_clock_dt;
    outfile__array_networkoperation_clock_dt.open("results\\_array_networkoperation_clock_dt_2017287913", ios::binary | ios::out);
    if(outfile__array_networkoperation_clock_dt.is_open())
    {
        outfile__array_networkoperation_clock_dt.write(reinterpret_cast<char*>(_array_networkoperation_clock_dt), 1*sizeof(double));
        outfile__array_networkoperation_clock_dt.close();
    } else
    {
        std::cout << "Error writing output file for _array_networkoperation_clock_dt." << endl;
    }
    ofstream outfile__array_networkoperation_clock_t;
    outfile__array_networkoperation_clock_t.open("results\\_array_networkoperation_clock_t_675949438", ios::binary | ios::out);
    if(outfile__array_networkoperation_clock_t.is_open())
    {
        outfile__array_networkoperation_clock_t.write(reinterpret_cast<char*>(_array_networkoperation_clock_t), 1*sizeof(double));
        outfile__array_networkoperation_clock_t.close();
    } else
    {
        std::cout << "Error writing output file for _array_networkoperation_clock_t." << endl;
    }
    ofstream outfile__array_networkoperation_clock_timestep;
    outfile__array_networkoperation_clock_timestep.open("results\\_array_networkoperation_clock_timestep_2734590571", ios::binary | ios::out);
    if(outfile__array_networkoperation_clock_timestep.is_open())
    {
        outfile__array_networkoperation_clock_timestep.write(reinterpret_cast<char*>(_array_networkoperation_clock_timestep), 1*sizeof(int64_t));
        outfile__array_networkoperation_clock_timestep.close();
    } else
    {
        std::cout << "Error writing output file for _array_networkoperation_clock_timestep." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_i, dev_array_neurongroup_1_i, sizeof(int32_t)*_num__array_neurongroup_1_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_i;
    outfile__array_neurongroup_1_i.open("results\\_array_neurongroup_1_i_3674354357", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_i.is_open())
    {
        outfile__array_neurongroup_1_i.write(reinterpret_cast<char*>(_array_neurongroup_1_i), 13*sizeof(int32_t));
        outfile__array_neurongroup_1_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_Igap, dev_array_neurongroup_1_Igap, sizeof(double)*_num__array_neurongroup_1_Igap, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_Igap;
    outfile__array_neurongroup_1_Igap.open("results\\_array_neurongroup_1_Igap_4276324654", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_Igap.is_open())
    {
        outfile__array_neurongroup_1_Igap.write(reinterpret_cast<char*>(_array_neurongroup_1_Igap), 13*sizeof(double));
        outfile__array_neurongroup_1_Igap.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_Igap." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_lastspike, dev_array_neurongroup_1_lastspike, sizeof(double)*_num__array_neurongroup_1_lastspike, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_lastspike;
    outfile__array_neurongroup_1_lastspike.open("results\\_array_neurongroup_1_lastspike_1163579662", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_lastspike.is_open())
    {
        outfile__array_neurongroup_1_lastspike.write(reinterpret_cast<char*>(_array_neurongroup_1_lastspike), 13*sizeof(double));
        outfile__array_neurongroup_1_lastspike.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_lastspike." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_not_refractory, dev_array_neurongroup_1_not_refractory, sizeof(char)*_num__array_neurongroup_1_not_refractory, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_not_refractory;
    outfile__array_neurongroup_1_not_refractory.open("results\\_array_neurongroup_1_not_refractory_897855399", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_not_refractory.is_open())
    {
        outfile__array_neurongroup_1_not_refractory.write(reinterpret_cast<char*>(_array_neurongroup_1_not_refractory), 13*sizeof(char));
        outfile__array_neurongroup_1_not_refractory.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_not_refractory." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_y, dev_array_neurongroup_1_y, sizeof(double)*_num__array_neurongroup_1_y, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_y;
    outfile__array_neurongroup_1_y.open("results\\_array_neurongroup_1_y_3333759697", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_y.is_open())
    {
        outfile__array_neurongroup_1_y.write(reinterpret_cast<char*>(_array_neurongroup_1_y), 13*sizeof(double));
        outfile__array_neurongroup_1_y.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_y." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_2_i, dev_array_neurongroup_2_i, sizeof(int32_t)*_num__array_neurongroup_2_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_2_i;
    outfile__array_neurongroup_2_i.open("results\\_array_neurongroup_2_i_3645148396", ios::binary | ios::out);
    if(outfile__array_neurongroup_2_i.is_open())
    {
        outfile__array_neurongroup_2_i.write(reinterpret_cast<char*>(_array_neurongroup_2_i), 13*sizeof(int32_t));
        outfile__array_neurongroup_2_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_2_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_2_v, dev_array_neurongroup_2_v, sizeof(double)*_num__array_neurongroup_2_v, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_2_v;
    outfile__array_neurongroup_2_v.open("results\\_array_neurongroup_2_v_1414299929", ios::binary | ios::out);
    if(outfile__array_neurongroup_2_v.is_open())
    {
        outfile__array_neurongroup_2_v.write(reinterpret_cast<char*>(_array_neurongroup_2_v), 13*sizeof(double));
        outfile__array_neurongroup_2_v.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_2_v." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_apost, dev_array_neurongroup_apost, sizeof(double)*_num__array_neurongroup_apost, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_apost;
    outfile__array_neurongroup_apost.open("results\\_array_neurongroup_apost_2265534849", ios::binary | ios::out);
    if(outfile__array_neurongroup_apost.is_open())
    {
        outfile__array_neurongroup_apost.write(reinterpret_cast<char*>(_array_neurongroup_apost), 7488*sizeof(double));
        outfile__array_neurongroup_apost.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_apost." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_apre, dev_array_neurongroup_apre, sizeof(double)*_num__array_neurongroup_apre, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_apre;
    outfile__array_neurongroup_apre.open("results\\_array_neurongroup_apre_86047374", ios::binary | ios::out);
    if(outfile__array_neurongroup_apre.is_open())
    {
        outfile__array_neurongroup_apre.write(reinterpret_cast<char*>(_array_neurongroup_apre), 7488*sizeof(double));
        outfile__array_neurongroup_apre.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_apre." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_c, dev_array_neurongroup_c, sizeof(double)*_num__array_neurongroup_c, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_c;
    outfile__array_neurongroup_c.open("results\\_array_neurongroup_c_2100369566", ios::binary | ios::out);
    if(outfile__array_neurongroup_c.is_open())
    {
        outfile__array_neurongroup_c.write(reinterpret_cast<char*>(_array_neurongroup_c), 7488*sizeof(double));
        outfile__array_neurongroup_c.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_c." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_i, dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_i;
    outfile__array_neurongroup_i.open("results\\_array_neurongroup_i_2649026944", ios::binary | ios::out);
    if(outfile__array_neurongroup_i.is_open())
    {
        outfile__array_neurongroup_i.write(reinterpret_cast<char*>(_array_neurongroup_i), 7488*sizeof(int32_t));
        outfile__array_neurongroup_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_l_speed, dev_array_neurongroup_l_speed, sizeof(double)*_num__array_neurongroup_l_speed, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_l_speed;
    outfile__array_neurongroup_l_speed.open("results\\_array_neurongroup_l_speed_269790516", ios::binary | ios::out);
    if(outfile__array_neurongroup_l_speed.is_open())
    {
        outfile__array_neurongroup_l_speed.write(reinterpret_cast<char*>(_array_neurongroup_l_speed), 7488*sizeof(double));
        outfile__array_neurongroup_l_speed.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_l_speed." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_v, dev_array_neurongroup_v, sizeof(double)*_num__array_neurongroup_v, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_v;
    outfile__array_neurongroup_v.open("results\\_array_neurongroup_v_283966581", ios::binary | ios::out);
    if(outfile__array_neurongroup_v.is_open())
    {
        outfile__array_neurongroup_v.write(reinterpret_cast<char*>(_array_neurongroup_v), 7488*sizeof(double));
        outfile__array_neurongroup_v.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_v." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikegeneratorgroup__lastindex, dev_array_spikegeneratorgroup__lastindex, sizeof(int32_t)*_num__array_spikegeneratorgroup__lastindex, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikegeneratorgroup__lastindex;
    outfile__array_spikegeneratorgroup__lastindex.open("results\\_array_spikegeneratorgroup__lastindex_987837788", ios::binary | ios::out);
    if(outfile__array_spikegeneratorgroup__lastindex.is_open())
    {
        outfile__array_spikegeneratorgroup__lastindex.write(reinterpret_cast<char*>(_array_spikegeneratorgroup__lastindex), 1*sizeof(int32_t));
        outfile__array_spikegeneratorgroup__lastindex.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikegeneratorgroup__lastindex." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikegeneratorgroup__period_bins, dev_array_spikegeneratorgroup__period_bins, sizeof(int32_t)*_num__array_spikegeneratorgroup__period_bins, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikegeneratorgroup__period_bins;
    outfile__array_spikegeneratorgroup__period_bins.open("results\\_array_spikegeneratorgroup__period_bins_4200411184", ios::binary | ios::out);
    if(outfile__array_spikegeneratorgroup__period_bins.is_open())
    {
        outfile__array_spikegeneratorgroup__period_bins.write(reinterpret_cast<char*>(_array_spikegeneratorgroup__period_bins), 1*sizeof(int32_t));
        outfile__array_spikegeneratorgroup__period_bins.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikegeneratorgroup__period_bins." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikegeneratorgroup_i, dev_array_spikegeneratorgroup_i, sizeof(int32_t)*_num__array_spikegeneratorgroup_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikegeneratorgroup_i;
    outfile__array_spikegeneratorgroup_i.open("results\\_array_spikegeneratorgroup_i_1329498599", ios::binary | ios::out);
    if(outfile__array_spikegeneratorgroup_i.is_open())
    {
        outfile__array_spikegeneratorgroup_i.write(reinterpret_cast<char*>(_array_spikegeneratorgroup_i), 576*sizeof(int32_t));
        outfile__array_spikegeneratorgroup_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikegeneratorgroup_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikegeneratorgroup_period, dev_array_spikegeneratorgroup_period, sizeof(double)*_num__array_spikegeneratorgroup_period, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikegeneratorgroup_period;
    outfile__array_spikegeneratorgroup_period.open("results\\_array_spikegeneratorgroup_period_3457314764", ios::binary | ios::out);
    if(outfile__array_spikegeneratorgroup_period.is_open())
    {
        outfile__array_spikegeneratorgroup_period.write(reinterpret_cast<char*>(_array_spikegeneratorgroup_period), 1*sizeof(double));
        outfile__array_spikegeneratorgroup_period.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikegeneratorgroup_period." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_1_N, dev_array_synapses_1_N, sizeof(int32_t)*_num__array_synapses_1_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_1_N;
    outfile__array_synapses_1_N.open("results\\_array_synapses_1_N_1771729519", ios::binary | ios::out);
    if(outfile__array_synapses_1_N.is_open())
    {
        outfile__array_synapses_1_N.write(reinterpret_cast<char*>(_array_synapses_1_N), 1*sizeof(int32_t));
        outfile__array_synapses_1_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_1_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_2_N, dev_array_synapses_2_N, sizeof(int32_t)*_num__array_synapses_2_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_2_N;
    outfile__array_synapses_2_N.open("results\\_array_synapses_2_N_1809632310", ios::binary | ios::out);
    if(outfile__array_synapses_2_N.is_open())
    {
        outfile__array_synapses_2_N.write(reinterpret_cast<char*>(_array_synapses_2_N), 1*sizeof(int32_t));
        outfile__array_synapses_2_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_2_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_3_N, dev_array_synapses_3_N, sizeof(int32_t)*_num__array_synapses_3_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_3_N;
    outfile__array_synapses_3_N.open("results\\_array_synapses_3_N_1780393473", ios::binary | ios::out);
    if(outfile__array_synapses_3_N.is_open())
    {
        outfile__array_synapses_3_N.write(reinterpret_cast<char*>(_array_synapses_3_N), 1*sizeof(int32_t));
        outfile__array_synapses_3_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_3_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_N, dev_array_synapses_N, sizeof(int32_t)*_num__array_synapses_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_N;
    outfile__array_synapses_N.open("results\\_array_synapses_N_483293785", ios::binary | ios::out);
    if(outfile__array_synapses_N.is_open())
    {
        outfile__array_synapses_N.write(reinterpret_cast<char*>(_array_synapses_N), 1*sizeof(int32_t));
        outfile__array_synapses_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_N." << endl;
    }

    _dynamic_array_spikegeneratorgroup__timebins = dev_dynamic_array_spikegeneratorgroup__timebins;
    ofstream outfile__dynamic_array_spikegeneratorgroup__timebins;
    outfile__dynamic_array_spikegeneratorgroup__timebins.open("results\\_dynamic_array_spikegeneratorgroup__timebins_4247931087", ios::binary | ios::out);
    if(outfile__dynamic_array_spikegeneratorgroup__timebins.is_open())
    {
        outfile__dynamic_array_spikegeneratorgroup__timebins.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikegeneratorgroup__timebins[0])), _dynamic_array_spikegeneratorgroup__timebins.size()*sizeof(int32_t));
        outfile__dynamic_array_spikegeneratorgroup__timebins.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup__timebins." << endl;
    }
    _dynamic_array_spikegeneratorgroup_neuron_index = dev_dynamic_array_spikegeneratorgroup_neuron_index;
    ofstream outfile__dynamic_array_spikegeneratorgroup_neuron_index;
    outfile__dynamic_array_spikegeneratorgroup_neuron_index.open("results\\_dynamic_array_spikegeneratorgroup_neuron_index_2789266935", ios::binary | ios::out);
    if(outfile__dynamic_array_spikegeneratorgroup_neuron_index.is_open())
    {
        outfile__dynamic_array_spikegeneratorgroup_neuron_index.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikegeneratorgroup_neuron_index[0])), _dynamic_array_spikegeneratorgroup_neuron_index.size()*sizeof(int32_t));
        outfile__dynamic_array_spikegeneratorgroup_neuron_index.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup_neuron_index." << endl;
    }
    _dynamic_array_spikegeneratorgroup_spike_number = dev_dynamic_array_spikegeneratorgroup_spike_number;
    ofstream outfile__dynamic_array_spikegeneratorgroup_spike_number;
    outfile__dynamic_array_spikegeneratorgroup_spike_number.open("results\\_dynamic_array_spikegeneratorgroup_spike_number_3584615111", ios::binary | ios::out);
    if(outfile__dynamic_array_spikegeneratorgroup_spike_number.is_open())
    {
        outfile__dynamic_array_spikegeneratorgroup_spike_number.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikegeneratorgroup_spike_number[0])), _dynamic_array_spikegeneratorgroup_spike_number.size()*sizeof(int32_t));
        outfile__dynamic_array_spikegeneratorgroup_spike_number.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup_spike_number." << endl;
    }
    _dynamic_array_spikegeneratorgroup_spike_time = dev_dynamic_array_spikegeneratorgroup_spike_time;
    ofstream outfile__dynamic_array_spikegeneratorgroup_spike_time;
    outfile__dynamic_array_spikegeneratorgroup_spike_time.open("results\\_dynamic_array_spikegeneratorgroup_spike_time_2775435892", ios::binary | ios::out);
    if(outfile__dynamic_array_spikegeneratorgroup_spike_time.is_open())
    {
        outfile__dynamic_array_spikegeneratorgroup_spike_time.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikegeneratorgroup_spike_time[0])), _dynamic_array_spikegeneratorgroup_spike_time.size()*sizeof(double));
        outfile__dynamic_array_spikegeneratorgroup_spike_time.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikegeneratorgroup_spike_time." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1__synaptic_post;
    outfile__dynamic_array_synapses_1__synaptic_post.open("results\\_dynamic_array_synapses_1__synaptic_post_1999337987", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_1__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1__synaptic_post[0])), _dynamic_array_synapses_1__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_1__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1__synaptic_pre;
    outfile__dynamic_array_synapses_1__synaptic_pre.open("results\\_dynamic_array_synapses_1__synaptic_pre_681065502", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_1__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1__synaptic_pre[0])), _dynamic_array_synapses_1__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_1__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1_delay;
    outfile__dynamic_array_synapses_1_delay.open("results\\_dynamic_array_synapses_1_delay_2373823482", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_delay.is_open())
    {
        outfile__dynamic_array_synapses_1_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_delay[0])), _dynamic_array_synapses_1_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_1_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_delay." << endl;
    }
    _dynamic_array_synapses_1_N_incoming = dev_dynamic_array_synapses_1_N_incoming;
    ofstream outfile__dynamic_array_synapses_1_N_incoming;
    outfile__dynamic_array_synapses_1_N_incoming.open("results\\_dynamic_array_synapses_1_N_incoming_3469555706", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_1_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_N_incoming[0])), _dynamic_array_synapses_1_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_1_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_N_incoming." << endl;
    }
    _dynamic_array_synapses_1_N_outgoing = dev_dynamic_array_synapses_1_N_outgoing;
    ofstream outfile__dynamic_array_synapses_1_N_outgoing;
    outfile__dynamic_array_synapses_1_N_outgoing.open("results\\_dynamic_array_synapses_1_N_outgoing_3922806560", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_1_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_N_outgoing[0])), _dynamic_array_synapses_1_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_1_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_N_outgoing." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2__synaptic_post;
    outfile__dynamic_array_synapses_2__synaptic_post.open("results\\_dynamic_array_synapses_2__synaptic_post_1591987953", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_2__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2__synaptic_post[0])), _dynamic_array_synapses_2__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_2__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2__synaptic_pre;
    outfile__dynamic_array_synapses_2__synaptic_pre.open("results\\_dynamic_array_synapses_2__synaptic_pre_971331175", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_2__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2__synaptic_pre[0])), _dynamic_array_synapses_2__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_2__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2_delay;
    outfile__dynamic_array_synapses_2_delay.open("results\\_dynamic_array_synapses_2_delay_3163926887", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_delay.is_open())
    {
        outfile__dynamic_array_synapses_2_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2_delay[0])), _dynamic_array_synapses_2_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_2_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_delay." << endl;
    }
    _dynamic_array_synapses_2_N_incoming = dev_dynamic_array_synapses_2_N_incoming;
    ofstream outfile__dynamic_array_synapses_2_N_incoming;
    outfile__dynamic_array_synapses_2_N_incoming.open("results\\_dynamic_array_synapses_2_N_incoming_3109283082", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_2_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2_N_incoming[0])), _dynamic_array_synapses_2_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_2_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_N_incoming." << endl;
    }
    _dynamic_array_synapses_2_N_outgoing = dev_dynamic_array_synapses_2_N_outgoing;
    ofstream outfile__dynamic_array_synapses_2_N_outgoing;
    outfile__dynamic_array_synapses_2_N_outgoing.open("results\\_dynamic_array_synapses_2_N_outgoing_2656015824", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_2_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2_N_outgoing[0])), _dynamic_array_synapses_2_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_2_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_N_outgoing." << endl;
    }
    _dynamic_array_synapses_2_w = dev_dynamic_array_synapses_2_w;
    ofstream outfile__dynamic_array_synapses_2_w;
    outfile__dynamic_array_synapses_2_w.open("results\\_dynamic_array_synapses_2_w_1828017567", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_w.is_open())
    {
        outfile__dynamic_array_synapses_2_w.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2_w[0])), _dynamic_array_synapses_2_w.size()*sizeof(double));
        outfile__dynamic_array_synapses_2_w.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_w." << endl;
    }
    ofstream outfile__dynamic_array_synapses_3__synaptic_post;
    outfile__dynamic_array_synapses_3__synaptic_post.open("results\\_dynamic_array_synapses_3__synaptic_post_4035665760", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_3__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_3__synaptic_post[0])), _dynamic_array_synapses_3__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_3__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_3__synaptic_pre;
    outfile__dynamic_array_synapses_3__synaptic_pre.open("results\\_dynamic_array_synapses_3__synaptic_pre_2149485967", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_3__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_3__synaptic_pre[0])), _dynamic_array_synapses_3__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_3__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_3_delay;
    outfile__dynamic_array_synapses_3_delay.open("results\\_dynamic_array_synapses_3_delay_451066579", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3_delay.is_open())
    {
        outfile__dynamic_array_synapses_3_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_3_delay[0])), _dynamic_array_synapses_3_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_3_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3_delay." << endl;
    }
    _dynamic_array_synapses_3_N_incoming = dev_dynamic_array_synapses_3_N_incoming;
    ofstream outfile__dynamic_array_synapses_3_N_incoming;
    outfile__dynamic_array_synapses_3_N_incoming.open("results\\_dynamic_array_synapses_3_N_incoming_586590565", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_3_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_3_N_incoming[0])), _dynamic_array_synapses_3_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_3_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3_N_incoming." << endl;
    }
    _dynamic_array_synapses_3_N_outgoing = dev_dynamic_array_synapses_3_N_outgoing;
    ofstream outfile__dynamic_array_synapses_3_N_outgoing;
    outfile__dynamic_array_synapses_3_N_outgoing.open("results\\_dynamic_array_synapses_3_N_outgoing_99277247", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_3_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_3_N_outgoing[0])), _dynamic_array_synapses_3_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_3_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3_N_outgoing." << endl;
    }
    _dynamic_array_synapses_3_w = dev_dynamic_array_synapses_3_w;
    ofstream outfile__dynamic_array_synapses_3_w;
    outfile__dynamic_array_synapses_3_w.open("results\\_dynamic_array_synapses_3_w_1832337320", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3_w.is_open())
    {
        outfile__dynamic_array_synapses_3_w.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_3_w[0])), _dynamic_array_synapses_3_w.size()*sizeof(double));
        outfile__dynamic_array_synapses_3_w.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3_w." << endl;
    }
    ofstream outfile__dynamic_array_synapses__synaptic_post;
    outfile__dynamic_array_synapses__synaptic_post.open("results\\_dynamic_array_synapses__synaptic_post_1801389495", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_post[0])), _dynamic_array_synapses__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses__synaptic_pre;
    outfile__dynamic_array_synapses__synaptic_pre.open("results\\_dynamic_array_synapses__synaptic_pre_814148175", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_pre[0])), _dynamic_array_synapses__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_delay;
    outfile__dynamic_array_synapses_delay.open("results\\_dynamic_array_synapses_delay_3246960869", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_delay.is_open())
    {
        outfile__dynamic_array_synapses_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_delay[0])), _dynamic_array_synapses_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_delay." << endl;
    }
    _dynamic_array_synapses_N_incoming = dev_dynamic_array_synapses_N_incoming;
    ofstream outfile__dynamic_array_synapses_N_incoming;
    outfile__dynamic_array_synapses_N_incoming.open("results\\_dynamic_array_synapses_N_incoming_1151751685", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_N_incoming[0])), _dynamic_array_synapses_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_N_incoming." << endl;
    }
    _dynamic_array_synapses_N_outgoing = dev_dynamic_array_synapses_N_outgoing;
    ofstream outfile__dynamic_array_synapses_N_outgoing;
    outfile__dynamic_array_synapses_N_outgoing.open("results\\_dynamic_array_synapses_N_outgoing_1673144031", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_N_outgoing[0])), _dynamic_array_synapses_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_N_outgoing." << endl;
    }
    _dynamic_array_synapses_w = dev_dynamic_array_synapses_w;
    ofstream outfile__dynamic_array_synapses_w;
    outfile__dynamic_array_synapses_w.open("results\\_dynamic_array_synapses_w_441891901", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_w.is_open())
    {
        outfile__dynamic_array_synapses_w.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_w[0])), _dynamic_array_synapses_w.size()*sizeof(double));
        outfile__dynamic_array_synapses_w.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_w." << endl;
    }


    // Write last run info to disk
    ofstream outfile_last_run_info;
    outfile_last_run_info.open("results/last_run_info.txt", ios::out);
    if(outfile_last_run_info.is_open())
    {
        outfile_last_run_info << (Network::_last_run_time) << " " << (Network::_last_run_completed_fraction) << std::endl;
        outfile_last_run_info.close();
    } else
    {
        std::cout << "Error writing last run info to file." << std::endl;
    }
}

__global__ void synapses_pre_destroy()
{
    using namespace brian;

    synapses_pre.destroy();
}
__global__ void synapses_1_post_destroy()
{
    using namespace brian;

    synapses_1_post.destroy();
}
__global__ void synapses_2_pre_destroy()
{
    using namespace brian;

    synapses_2_pre.destroy();
}
__global__ void synapses_3_pre_destroy()
{
    using namespace brian;

    synapses_3_pre.destroy();
}

void _dealloc_arrays()
{
    using namespace brian;


    CUDA_SAFE_CALL(
            curandDestroyGenerator(curand_generator)
            );

    synapses_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_pre_destroy");
    synapses_1_post_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_1_post_destroy");
    synapses_2_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_2_pre_destroy");
    synapses_3_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_3_pre_destroy");

    dev_dynamic_array_spikegeneratorgroup__timebins.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_spikegeneratorgroup__timebins);
    _dynamic_array_spikegeneratorgroup__timebins.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_spikegeneratorgroup__timebins);
    dev_dynamic_array_spikegeneratorgroup_neuron_index.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_spikegeneratorgroup_neuron_index);
    _dynamic_array_spikegeneratorgroup_neuron_index.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_spikegeneratorgroup_neuron_index);
    dev_dynamic_array_spikegeneratorgroup_spike_number.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_spikegeneratorgroup_spike_number);
    _dynamic_array_spikegeneratorgroup_spike_number.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_spikegeneratorgroup_spike_number);
    dev_dynamic_array_spikegeneratorgroup_spike_time.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_spikegeneratorgroup_spike_time);
    _dynamic_array_spikegeneratorgroup_spike_time.clear();
    thrust::host_vector<double>().swap(_dynamic_array_spikegeneratorgroup_spike_time);
    dev_dynamic_array_synapses_1__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_1__synaptic_post);
    _dynamic_array_synapses_1__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_1__synaptic_post);
    dev_dynamic_array_synapses_1__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_1__synaptic_pre);
    _dynamic_array_synapses_1__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_1__synaptic_pre);
    dev_dynamic_array_synapses_1_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_1_delay);
    _dynamic_array_synapses_1_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_1_delay);
    dev_dynamic_array_synapses_1_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_1_N_incoming);
    _dynamic_array_synapses_1_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_1_N_incoming);
    dev_dynamic_array_synapses_1_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_1_N_outgoing);
    _dynamic_array_synapses_1_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_1_N_outgoing);
    dev_dynamic_array_synapses_2__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_2__synaptic_post);
    _dynamic_array_synapses_2__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_2__synaptic_post);
    dev_dynamic_array_synapses_2__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_2__synaptic_pre);
    _dynamic_array_synapses_2__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_2__synaptic_pre);
    dev_dynamic_array_synapses_2_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_2_delay);
    _dynamic_array_synapses_2_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_2_delay);
    dev_dynamic_array_synapses_2_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_2_N_incoming);
    _dynamic_array_synapses_2_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_2_N_incoming);
    dev_dynamic_array_synapses_2_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_2_N_outgoing);
    _dynamic_array_synapses_2_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_2_N_outgoing);
    dev_dynamic_array_synapses_2_w.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_2_w);
    _dynamic_array_synapses_2_w.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_2_w);
    dev_dynamic_array_synapses_3__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_3__synaptic_post);
    _dynamic_array_synapses_3__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_3__synaptic_post);
    dev_dynamic_array_synapses_3__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_3__synaptic_pre);
    _dynamic_array_synapses_3__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_3__synaptic_pre);
    dev_dynamic_array_synapses_3_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_3_delay);
    _dynamic_array_synapses_3_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_3_delay);
    dev_dynamic_array_synapses_3_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_3_N_incoming);
    _dynamic_array_synapses_3_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_3_N_incoming);
    dev_dynamic_array_synapses_3_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_3_N_outgoing);
    _dynamic_array_synapses_3_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_3_N_outgoing);
    dev_dynamic_array_synapses_3_w.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_3_w);
    _dynamic_array_synapses_3_w.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_3_w);
    dev_dynamic_array_synapses__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses__synaptic_post);
    _dynamic_array_synapses__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses__synaptic_post);
    dev_dynamic_array_synapses__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses__synaptic_pre);
    _dynamic_array_synapses__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses__synaptic_pre);
    dev_dynamic_array_synapses_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_delay);
    _dynamic_array_synapses_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_delay);
    dev_dynamic_array_synapses_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_N_incoming);
    _dynamic_array_synapses_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_N_incoming);
    dev_dynamic_array_synapses_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_N_outgoing);
    _dynamic_array_synapses_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_N_outgoing);
    dev_dynamic_array_synapses_w.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_w);
    _dynamic_array_synapses_w.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_w);

    if(_array_defaultclock_dt!=0)
    {
        delete [] _array_defaultclock_dt;
        _array_defaultclock_dt = 0;
    }
    if(dev_array_defaultclock_dt!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_defaultclock_dt)
                );
        dev_array_defaultclock_dt = 0;
    }
    if(_array_defaultclock_t!=0)
    {
        delete [] _array_defaultclock_t;
        _array_defaultclock_t = 0;
    }
    if(dev_array_defaultclock_t!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_defaultclock_t)
                );
        dev_array_defaultclock_t = 0;
    }
    if(_array_defaultclock_timestep!=0)
    {
        delete [] _array_defaultclock_timestep;
        _array_defaultclock_timestep = 0;
    }
    if(dev_array_defaultclock_timestep!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_defaultclock_timestep)
                );
        dev_array_defaultclock_timestep = 0;
    }
    if(_array_networkoperation_1_clock_dt!=0)
    {
        delete [] _array_networkoperation_1_clock_dt;
        _array_networkoperation_1_clock_dt = 0;
    }
    if(dev_array_networkoperation_1_clock_dt!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_networkoperation_1_clock_dt)
                );
        dev_array_networkoperation_1_clock_dt = 0;
    }
    if(_array_networkoperation_1_clock_t!=0)
    {
        delete [] _array_networkoperation_1_clock_t;
        _array_networkoperation_1_clock_t = 0;
    }
    if(dev_array_networkoperation_1_clock_t!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_networkoperation_1_clock_t)
                );
        dev_array_networkoperation_1_clock_t = 0;
    }
    if(_array_networkoperation_1_clock_timestep!=0)
    {
        delete [] _array_networkoperation_1_clock_timestep;
        _array_networkoperation_1_clock_timestep = 0;
    }
    if(dev_array_networkoperation_1_clock_timestep!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_networkoperation_1_clock_timestep)
                );
        dev_array_networkoperation_1_clock_timestep = 0;
    }
    if(_array_networkoperation_2_clock_dt!=0)
    {
        delete [] _array_networkoperation_2_clock_dt;
        _array_networkoperation_2_clock_dt = 0;
    }
    if(dev_array_networkoperation_2_clock_dt!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_networkoperation_2_clock_dt)
                );
        dev_array_networkoperation_2_clock_dt = 0;
    }
    if(_array_networkoperation_2_clock_t!=0)
    {
        delete [] _array_networkoperation_2_clock_t;
        _array_networkoperation_2_clock_t = 0;
    }
    if(dev_array_networkoperation_2_clock_t!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_networkoperation_2_clock_t)
                );
        dev_array_networkoperation_2_clock_t = 0;
    }
    if(_array_networkoperation_2_clock_timestep!=0)
    {
        delete [] _array_networkoperation_2_clock_timestep;
        _array_networkoperation_2_clock_timestep = 0;
    }
    if(dev_array_networkoperation_2_clock_timestep!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_networkoperation_2_clock_timestep)
                );
        dev_array_networkoperation_2_clock_timestep = 0;
    }
    if(_array_networkoperation_clock_dt!=0)
    {
        delete [] _array_networkoperation_clock_dt;
        _array_networkoperation_clock_dt = 0;
    }
    if(dev_array_networkoperation_clock_dt!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_networkoperation_clock_dt)
                );
        dev_array_networkoperation_clock_dt = 0;
    }
    if(_array_networkoperation_clock_t!=0)
    {
        delete [] _array_networkoperation_clock_t;
        _array_networkoperation_clock_t = 0;
    }
    if(dev_array_networkoperation_clock_t!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_networkoperation_clock_t)
                );
        dev_array_networkoperation_clock_t = 0;
    }
    if(_array_networkoperation_clock_timestep!=0)
    {
        delete [] _array_networkoperation_clock_timestep;
        _array_networkoperation_clock_timestep = 0;
    }
    if(dev_array_networkoperation_clock_timestep!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_networkoperation_clock_timestep)
                );
        dev_array_networkoperation_clock_timestep = 0;
    }
    if(_array_neurongroup_1_i!=0)
    {
        delete [] _array_neurongroup_1_i;
        _array_neurongroup_1_i = 0;
    }
    if(dev_array_neurongroup_1_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_i)
                );
        dev_array_neurongroup_1_i = 0;
    }
    if(_array_neurongroup_1_Igap!=0)
    {
        delete [] _array_neurongroup_1_Igap;
        _array_neurongroup_1_Igap = 0;
    }
    if(dev_array_neurongroup_1_Igap!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_Igap)
                );
        dev_array_neurongroup_1_Igap = 0;
    }
    if(_array_neurongroup_1_lastspike!=0)
    {
        delete [] _array_neurongroup_1_lastspike;
        _array_neurongroup_1_lastspike = 0;
    }
    if(dev_array_neurongroup_1_lastspike!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_lastspike)
                );
        dev_array_neurongroup_1_lastspike = 0;
    }
    if(_array_neurongroup_1_not_refractory!=0)
    {
        delete [] _array_neurongroup_1_not_refractory;
        _array_neurongroup_1_not_refractory = 0;
    }
    if(dev_array_neurongroup_1_not_refractory!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_not_refractory)
                );
        dev_array_neurongroup_1_not_refractory = 0;
    }
    if(_array_neurongroup_1_y!=0)
    {
        delete [] _array_neurongroup_1_y;
        _array_neurongroup_1_y = 0;
    }
    if(dev_array_neurongroup_1_y!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_y)
                );
        dev_array_neurongroup_1_y = 0;
    }
    if(_array_neurongroup_2_i!=0)
    {
        delete [] _array_neurongroup_2_i;
        _array_neurongroup_2_i = 0;
    }
    if(dev_array_neurongroup_2_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_2_i)
                );
        dev_array_neurongroup_2_i = 0;
    }
    if(_array_neurongroup_2_v!=0)
    {
        delete [] _array_neurongroup_2_v;
        _array_neurongroup_2_v = 0;
    }
    if(dev_array_neurongroup_2_v!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_2_v)
                );
        dev_array_neurongroup_2_v = 0;
    }
    if(_array_neurongroup_apost!=0)
    {
        delete [] _array_neurongroup_apost;
        _array_neurongroup_apost = 0;
    }
    if(dev_array_neurongroup_apost!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_apost)
                );
        dev_array_neurongroup_apost = 0;
    }
    if(_array_neurongroup_apre!=0)
    {
        delete [] _array_neurongroup_apre;
        _array_neurongroup_apre = 0;
    }
    if(dev_array_neurongroup_apre!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_apre)
                );
        dev_array_neurongroup_apre = 0;
    }
    if(_array_neurongroup_c!=0)
    {
        delete [] _array_neurongroup_c;
        _array_neurongroup_c = 0;
    }
    if(dev_array_neurongroup_c!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_c)
                );
        dev_array_neurongroup_c = 0;
    }
    if(_array_neurongroup_i!=0)
    {
        delete [] _array_neurongroup_i;
        _array_neurongroup_i = 0;
    }
    if(dev_array_neurongroup_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_i)
                );
        dev_array_neurongroup_i = 0;
    }
    if(_array_neurongroup_l_speed!=0)
    {
        delete [] _array_neurongroup_l_speed;
        _array_neurongroup_l_speed = 0;
    }
    if(dev_array_neurongroup_l_speed!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_l_speed)
                );
        dev_array_neurongroup_l_speed = 0;
    }
    if(_array_neurongroup_v!=0)
    {
        delete [] _array_neurongroup_v;
        _array_neurongroup_v = 0;
    }
    if(dev_array_neurongroup_v!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_v)
                );
        dev_array_neurongroup_v = 0;
    }
    if(_array_spikegeneratorgroup__lastindex!=0)
    {
        delete [] _array_spikegeneratorgroup__lastindex;
        _array_spikegeneratorgroup__lastindex = 0;
    }
    if(dev_array_spikegeneratorgroup__lastindex!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikegeneratorgroup__lastindex)
                );
        dev_array_spikegeneratorgroup__lastindex = 0;
    }
    if(_array_spikegeneratorgroup__period_bins!=0)
    {
        delete [] _array_spikegeneratorgroup__period_bins;
        _array_spikegeneratorgroup__period_bins = 0;
    }
    if(dev_array_spikegeneratorgroup__period_bins!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikegeneratorgroup__period_bins)
                );
        dev_array_spikegeneratorgroup__period_bins = 0;
    }
    if(_array_spikegeneratorgroup_i!=0)
    {
        delete [] _array_spikegeneratorgroup_i;
        _array_spikegeneratorgroup_i = 0;
    }
    if(dev_array_spikegeneratorgroup_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikegeneratorgroup_i)
                );
        dev_array_spikegeneratorgroup_i = 0;
    }
    if(_array_spikegeneratorgroup_period!=0)
    {
        delete [] _array_spikegeneratorgroup_period;
        _array_spikegeneratorgroup_period = 0;
    }
    if(dev_array_spikegeneratorgroup_period!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikegeneratorgroup_period)
                );
        dev_array_spikegeneratorgroup_period = 0;
    }
    if(_array_synapses_1_N!=0)
    {
        delete [] _array_synapses_1_N;
        _array_synapses_1_N = 0;
    }
    if(dev_array_synapses_1_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_1_N)
                );
        dev_array_synapses_1_N = 0;
    }
    if(_array_synapses_2_N!=0)
    {
        delete [] _array_synapses_2_N;
        _array_synapses_2_N = 0;
    }
    if(dev_array_synapses_2_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_2_N)
                );
        dev_array_synapses_2_N = 0;
    }
    if(_array_synapses_3_N!=0)
    {
        delete [] _array_synapses_3_N;
        _array_synapses_3_N = 0;
    }
    if(dev_array_synapses_3_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_3_N)
                );
        dev_array_synapses_3_N = 0;
    }
    if(_array_synapses_N!=0)
    {
        delete [] _array_synapses_N;
        _array_synapses_N = 0;
    }
    if(dev_array_synapses_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_N)
                );
        dev_array_synapses_N = 0;
    }


    // static arrays
    if(_static_array__array_neurongroup_c!=0)
    {
        delete [] _static_array__array_neurongroup_c;
        _static_array__array_neurongroup_c = 0;
    }
    if(_static_array__dynamic_array_spikegeneratorgroup__timebins!=0)
    {
        delete [] _static_array__dynamic_array_spikegeneratorgroup__timebins;
        _static_array__dynamic_array_spikegeneratorgroup__timebins = 0;
    }
    if(_static_array__dynamic_array_spikegeneratorgroup_neuron_index!=0)
    {
        delete [] _static_array__dynamic_array_spikegeneratorgroup_neuron_index;
        _static_array__dynamic_array_spikegeneratorgroup_neuron_index = 0;
    }
    if(_static_array__dynamic_array_spikegeneratorgroup_spike_number!=0)
    {
        delete [] _static_array__dynamic_array_spikegeneratorgroup_spike_number;
        _static_array__dynamic_array_spikegeneratorgroup_spike_number = 0;
    }
    if(_static_array__dynamic_array_spikegeneratorgroup_spike_time!=0)
    {
        delete [] _static_array__dynamic_array_spikegeneratorgroup_spike_time;
        _static_array__dynamic_array_spikegeneratorgroup_spike_time = 0;
    }


}

