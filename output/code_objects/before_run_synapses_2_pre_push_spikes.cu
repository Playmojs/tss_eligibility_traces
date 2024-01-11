#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>
#include <iostream>
#include <ctime>
#include <limits.h>
#include <tuple>
#include <string>
#include <iomanip>
#include <vector>
#include "objects.h"
#include "code_objects/synapses_2_pre_push_spikes.h"
#include "brianlib/cuda_utils.h"

// Makro for file and line information in _cudaSafeCall
#define COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(a, b, c, d, e) \
    _copyHostArrayToDeviceSymbol(a, b, c, d, e, __FILE__, __LINE__)

namespace {
    // vector_t<T> is an alias for thrust:host_vector<T>
    template <typename T> using vector_t = thrust::host_vector<T>;
    // tuple type typedef
    typedef std::tuple<std::string, size_t, int> tuple_t;

    std::vector<tuple_t> memory_recorder;

    // Functions for online update of mean and std
    // for a new value newValue, compute the new count, new mean, the new M2.
    // mean accumulates the mean of the entire dataset
    // M2 aggregates the squared distance from the mean
    // count aggregates the number of samples seen so far
    inline void updateMeanStd(int &count, double &mean, double& M2, double newValue){
        count += 1;
        double delta = newValue - mean;
        mean += delta / count;
        double delta2 = newValue - mean;
        M2 += delta * delta2;
    }

    // get std from aggregated M2 value
    double getStd(int count, double M2){
        if (count < 2){
            return NAN;
        }
        double variance = M2 / (count - 1);
        double stdValue = sqrt(variance);
        return stdValue;
    }

    // Copy the data from a host array to global device memory and copy the
    // symbol to a global device variable.
    // device_array: device pointer to allocate data for and which to copy to device symbol
    // host_array: host array with data to copy
    // device_symbol: global __device__ variable of same type as `host_array`
    // num_elements: number of elements in host_array to copy
    // NOTE: T can be a pointer variable itself (when copying 2D arrays)
    template <typename T>
    inline void _copyHostArrayToDeviceSymbol(T *device_array, const T *host_array,
            T *&device_symbol, int num_elements, const char* name, const char* file,
            const int line){
        size_t bytes = sizeof(T) * num_elements;
        // allocate device memory
        _cudaSafeCall(
                cudaMalloc((void**)&device_array, bytes),
                file, line, "cudaMalloc");
        // copy data from host array to device
        _cudaSafeCall(
                cudaMemcpy(device_array, host_array, bytes, cudaMemcpyHostToDevice),
                file, line, "cudaMemcpy");
        // copy the device data pointer to the global device symbol
        _cudaSafeCall(
                cudaMemcpyToSymbol(device_symbol, &device_array, sizeof(T*)),
                file, line, "cudaMemcpyToSymbol");
        memory_recorder.push_back(std::make_tuple(name, bytes, num_elements));
    }
}


__global__ void _before_run_kernel_synapses_2_pre_push_spikes(
    int _source_N,
    int _num_blocks,
    int _num_threads,
    double _dt,
    int _syn_N,
    int num_delays,
    bool scalar_delay)
{
    using namespace brian;

    int tid = threadIdx.x;

    if (scalar_delay)
    {
        if (tid == 0)
        {
            synapses_2_pre.queue->num_blocks = _num_blocks;
            synapses_2_pre.queue->num_delays = num_delays;
        }
    }
    else
    {
        synapses_2_pre.queue->prepare(
            tid,
            _num_threads,
            _num_blocks,
            0,
            _source_N,
            _syn_N,
            num_delays,
            synapses_2_pre_num_synapses_by_pre,
            synapses_2_pre_num_synapses_by_bundle,
            synapses_2_pre_num_unique_delays_by_pre,
            synapses_2_pre_unique_delays,
            synapses_2_pre_global_bundle_id_start_by_pre,
            synapses_2_pre_synapses_offset_by_bundle,
            synapses_2_pre_synapse_ids,
            synapses_2_pre_synapse_ids_by_pre,
            synapses_2_pre_unique_delays_offset_by_pre,
            synapses_2_pre_unique_delay_start_idcs
        );
    }
    synapses_2_pre.no_or_const_delay_mode = scalar_delay;
}

///// Support code /////

template < typename T1, typename T2 > struct _higher_type;
template < > struct _higher_type<int,int> { typedef int type; };
template < > struct _higher_type<int,long> { typedef long type; };
template < > struct _higher_type<int,long long> { typedef long long type; };
template < > struct _higher_type<int,float> { typedef float type; };
template < > struct _higher_type<int,double> { typedef double type; };
template < > struct _higher_type<long,int> { typedef long type; };
template < > struct _higher_type<long,long> { typedef long type; };
template < > struct _higher_type<long,long long> { typedef long long type; };
template < > struct _higher_type<long,float> { typedef float type; };
template < > struct _higher_type<long,double> { typedef double type; };
template < > struct _higher_type<long long,int> { typedef long long type; };
template < > struct _higher_type<long long,long> { typedef long long type; };
template < > struct _higher_type<long long,long long> { typedef long long type; };
template < > struct _higher_type<long long,float> { typedef float type; };
template < > struct _higher_type<long long,double> { typedef double type; };
template < > struct _higher_type<float,int> { typedef float type; };
template < > struct _higher_type<float,long> { typedef float type; };
template < > struct _higher_type<float,long long> { typedef float type; };
template < > struct _higher_type<float,float> { typedef float type; };
template < > struct _higher_type<float,double> { typedef double type; };
template < > struct _higher_type<double,int> { typedef double type; };
template < > struct _higher_type<double,long> { typedef double type; };
template < > struct _higher_type<double,long long> { typedef double type; };
template < > struct _higher_type<double,float> { typedef double type; };
template < > struct _higher_type<double,double> { typedef double type; };
template < typename T1, typename T2 >
__host__ __device__ static inline typename _higher_type<T1,T2>::type
_brian_mod(T1 x, T2 y)
{{
    return x-y*floor(1.0*x/y);
}}
template < typename T1, typename T2 >
__host__ __device__ static inline typename _higher_type<T1,T2>::type
_brian_floordiv(T1 x, T2 y)
{{
    return floor(1.0*x/y);
}}
#ifdef _MSC_VER
#define _brian_pow(x, y) (pow((double)(x), (y)))
#else
#define _brian_pow(x, y) (pow((x), (y)))
#endif
                inline __device__ int _brian_atomicAdd(int* address, int val)
                {
        // hardware implementation
        return atomicAdd(address, val);
                }
                inline __device__ float _brian_atomicAdd(float* address, float val)
                {
        // hardware implementation
        return atomicAdd(address, val);
                }
                inline __device__ double _brian_atomicAdd(double* address, double val)
                {
                        #if (__CUDA_ARCH__ >= 600)
        // hardware implementation
        return atomicAdd(address, val);
                        #else
        // software implementation
        unsigned long long int* address_as_int = (unsigned long long int*)address;
        unsigned long long int old = *address_as_int, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_int, assumed,
                            __double_as_longlong(val +
                                   __longlong_as_double(assumed)));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);
        return __longlong_as_double(old);
                        #endif
                }
                inline __device__ int _brian_atomicMul(int* address, int val)
                {
                    // software implementation
                    int old = *address, assumed;
                    do {
                        assumed = old;
                        old = atomicCAS(address, assumed, val * assumed);
                    } while (assumed != old);
                    return old;
                }
                inline __device__ float _brian_atomicMul(float* address, float val)
                {
        // software implementation
        int* address_as_int = (int*)address;
        int old = *address_as_int, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_int, assumed,
                            __float_as_int(val *
                                   __int_as_float(assumed)));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);
        return __int_as_float(old);
                }
                inline __device__ double _brian_atomicMul(double* address, double val)
                {
        // software implementation
        unsigned long long int* address_as_int = (unsigned long long int*)address;
        unsigned long long int old = *address_as_int, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_int, assumed,
                            __double_as_longlong(val *
                                   __longlong_as_double(assumed)));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);
        return __longlong_as_double(old);
                }
                inline __device__ int _brian_atomicDiv(int* address, int val)
                {
                    // software implementation
                    int old = *address, assumed;
                    do {
                        assumed = old;
                        old = atomicCAS(address, assumed, val / assumed);
                    } while (assumed != old);
                    return old;
                }
                inline __device__ float _brian_atomicDiv(float* address, float val)
                {
        // software implementation
        int* address_as_int = (int*)address;
        int old = *address_as_int, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_int, assumed,
                            __float_as_int(val /
                                   __int_as_float(assumed)));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);
        return __int_as_float(old);
                }
                inline __device__ double _brian_atomicDiv(double* address, double val)
                {
        // software implementation
        unsigned long long int* address_as_int = (unsigned long long int*)address;
        unsigned long long int old = *address_as_int, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_int, assumed,
                            __double_as_longlong(val /
                                   __longlong_as_double(assumed)));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);
        return __longlong_as_double(old);
                }


void _before_run_synapses_2_pre_push_spikes()
{
    using namespace brian;

    std::clock_t start_timer = std::clock();
    const double to_MB = 1.0 / (1024.0 * 1024.0);
    static bool first_run = true;

    CUDA_CHECK_MEMORY();
    size_t used_device_memory_start = used_device_memory;

    ///// HOST_CONSTANTS ///////////
    const int _numN = 1;
		const int32_t _n_sources = 13;
		const int32_t _n_targets = 13;
		const int _num_spikespace = 14;
		double* const _array_synapses_2_delay = thrust::raw_pointer_cast(&_dynamic_array_synapses_2_delay[0]);
		const int _numdelay = _dynamic_array_synapses_2_delay.size();
		double* const dev_array_synapses_2_delay = thrust::raw_pointer_cast(&dev_dynamic_array_synapses_2_delay[0]);

    ///// pointers_lines /////
        
    int32_t*   _ptr_array_synapses_2_N = _array_synapses_2_N;
    double*   _ptr_array_defaultclock_dt = _array_defaultclock_dt;
    int32_t* __restrict  _ptr_array_neurongroup_1__spikespace = _array_neurongroup_1__spikespace;
    double*   _ptr_array_synapses_2_delay = _array_synapses_2_delay;


    int64_t syn_N_check = _ptr_array_synapses_2_N[0];

    if (syn_N_check == 0){
        return;
    }
    else if (syn_N_check > INT_MAX){
        printf("ERROR: There are more Synapses (%lu) than an int can "
               "hold on this system (%u).\n", syn_N_check, INT_MAX);
    }
    // total number of synapses
    int syn_N = (int)syn_N_check;

    // simulation time step
    double dt = _ptr_array_defaultclock_dt[0];
    // number of neurons in source group
    int source_N = _n_sources;
    // number of neurons in target group
    int target_N = _n_targets;

    //////////////////////
    // Scalar variables //
    //////////////////////

    // total number of (preID, postBlock) pairs
    int num_pre_post_blocks = num_parallel_blocks * source_N;
    // size of the connectivity matrix (equal number of synapses)
    int size_connectivity_matrix = 0;

    // statistics of number of synapses per (preID, postBlock) pair
    int sum_num_elements = 0;
    int count_num_elements = 0;
    double mean_num_elements = 0;
    double M2_num_elements = 0;


    ////////////////////////////////////////////////////////
    // Create array and vector variables (in host memory) //
    ////////////////////////////////////////////////////////

    /* VARIABLE NAMING:
     * Not scalar variables are named after TYPE_NAME_STRUCTURE, with:
     * STRUCTURE: the first array dimensions structure (`by_pre`, `by_bundle` or none)
     *   `by_pre`: Array (host pointer type) of size `num_pre_post_blocks`,
     *             which is the number of (preID, postBlock) pairs.
     *   `by_bundle`: thrust::host_vector, size of total number of bundles,
     *                which is one for each delay in each (preID, postBlock) pair.
     *                Different (preID, postBlock) pairs can have different sets
     *                of delay values -> each bundle gets a global bundleID
     *   none: If no STRUCTURE given, it's a one dim array storing everything
     * TYPE: data type in STRUCTURE (`h`, `h_vec`, `h_ptr`, `d_ptr`), with
     *       `h`: host value, `h_vec`: host vector, `h_ptr`: host pointer,
     *       `d_ptr`: device pointer (pointing to device, stored in host memory)
     * NAME: the variable name
     *
     * EXAMPLES:
     * `h_vec_delays_by_pre` - an array [size = num_pre_post_blocks] of host
     *                         vectors, each storing delay values of a
     *                         (preID, postBlock) pair
     * `h_num_synapses_by_bundle` - a host vector of integers specifying the
     *                              number of synapses in a bundle
     * `d_ptr_synapse_ids` - a device pointer to synapse IDs (all of them)
     */

    // synapse IDs for each (preID, postBlock) pair
    vector_t<int32_t>* h_vec_synapse_ids_by_pre = new vector_t<int32_t>[num_pre_post_blocks];
    // array of synapse IDs in device memory for each (preID, postBlock) pair
    int32_t** h_ptr_d_ptr_synapse_ids_by_pre;
    static int32_t **d_ptr_d_ptr_synapse_ids_by_pre;
    // number of synapses for each (preID, postBlock) pair
    int* h_num_synapses_by_pre;
    static int* d_ptr_num_synapses_by_pre;



    // we need to allocate device memory for synapse IDs independent of delay mode
    static int32_t* d_ptr_synapse_ids;
    size_t memory_synapse_ids = sizeof(int32_t) * syn_N;
    if (!first_run) {
        CUDA_SAFE_CALL(cudaFree(d_ptr_synapse_ids));
    }
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&d_ptr_synapse_ids, memory_synapse_ids)
            );
    memory_recorder.push_back(std::make_tuple("synapse IDs", memory_synapse_ids, syn_N));


    //fill vectors of connectivity matrix with synapse IDs and delays (in units of simulation time step)
    int max_delay = (int)(_dynamic_array_synapses_2_delay[0] / dt + 0.5);
    for(int syn_id = 0; syn_id < syn_N; syn_id++)  // loop through all synapses
    {


        // Code generation checks
        assert(0 == 0);

        assert(0 == 0);

        // pre/post_neuron_id are integers from 0 to Nsource/Ntarget (from corresponding
        // SynapticPathway) this is relevant only when using Subgroups where they might
        // be NOT equal to the idx in their NeuronGroup
        int32_t pre_neuron_id = _dynamic_array_synapses_2__synaptic_pre[syn_id] - 0;
        int32_t post_neuron_id = _dynamic_array_synapses_2__synaptic_post[syn_id] - 0;


        // each parallel executed cuda block gets an equal part of post neuron IDs
        int post_block_id = (post_neuron_id * num_parallel_blocks) / target_N;
        // we store synapses for each pre neuron and post block
        int pre_post_block_id = pre_neuron_id * num_parallel_blocks + post_block_id;

        h_vec_synapse_ids_by_pre[pre_post_block_id].push_back(syn_id);
    }
    int num_delays = max_delay;
    int num_queues = max_delay + 1;  // we also need a current step

    synapses_2_pre_delay = max_delay;
    CUDA_CHECK_MEMORY();
    size_t used_device_memory_after_dealloc = used_device_memory;

    ///////////////////////////////////////////////////////
    // Memory allocations which depend on the delay mode //
    ///////////////////////////////////////////////////////

    {
        h_num_synapses_by_pre = new int[num_pre_post_blocks];
        h_ptr_d_ptr_synapse_ids_by_pre = new int32_t*[num_pre_post_blocks];
    }


    // loop through connectivity matrix [(preID, postBlock) pairs]
    for(int i = 0; i < num_pre_post_blocks; i++)
    {
        // i is pre_post_block_id

        int num_elements = h_vec_synapse_ids_by_pre[i].size();
        size_connectivity_matrix += num_elements;
        if (num_elements > synapses_2_pre_max_size)
            synapses_2_pre_max_size = num_elements;

        {
            // copy the synapse IDs and the number of synapses for this
            // (preID, postBlock) to device and store the device pointer

            h_num_synapses_by_pre[i] = num_elements;

            h_ptr_d_ptr_synapse_ids_by_pre[i] = d_ptr_synapse_ids + sum_num_elements;
            CUDA_SAFE_CALL(
                    cudaMemcpy(h_ptr_d_ptr_synapse_ids_by_pre[i],
                        thrust::raw_pointer_cast(&(h_vec_synapse_ids_by_pre[i][0])),
                        sizeof(int32_t) * num_elements,
                        cudaMemcpyHostToDevice)
                    );
        }

        sum_num_elements += num_elements;
        updateMeanStd(count_num_elements, mean_num_elements, M2_num_elements, num_elements);
    }  // end for loop through connectivity matrix
    printf("INFO connectivity matrix has size %i, number of (pre neuron ID, post neuron block) pairs is %u\n",
            size_connectivity_matrix, num_pre_post_blocks);

    {
        if (!first_run) {
            CUDA_SAFE_CALL(cudaFree(d_ptr_num_synapses_by_pre));
            CUDA_SAFE_CALL(cudaFree(d_ptr_d_ptr_synapse_ids_by_pre));
        }
        COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(d_ptr_num_synapses_by_pre,
                h_num_synapses_by_pre, synapses_2_pre_num_synapses_by_pre,
                num_pre_post_blocks, "number of synapses per pre/post block");
        // synapses id
        COPY_HOST_ARRAY_TO_DEVICE_SYMBOL(d_ptr_d_ptr_synapse_ids_by_pre,
                h_ptr_d_ptr_synapse_ids_by_pre, synapses_2_pre_synapse_ids_by_pre,
                num_pre_post_blocks,
                "pointers to synapse IDs");
    }


    ////////////////////////////////////////////////////
    //// PRINT INFORMATION ON MEMORY USAGE AND TIME ////
    ////////////////////////////////////////////////////

    // TODO print statistics!

    // sum all allocated memory
    size_t total_memory = 0;
    int max_string_length = 0;
    for(auto const& tuple: memory_recorder){
        total_memory += std::get<1>(tuple);
        int str_len = std::get<0>(tuple).length();
        if (str_len > max_string_length)
            max_string_length = str_len;
    }
    double total_memory_MB = total_memory * to_MB;
    max_string_length += 5;

    // sort tuples by used memory
    std::sort(begin(memory_recorder), end(memory_recorder),
            [](tuple_t const &t1, tuple_t const &t2) {
            return std::get<1>(t1) > std::get<1>(t2); // or use a custom compare function
            }
            );

    double std_num_elements = getStd(count_num_elements, M2_num_elements);

    // print memory information
    std::cout.precision(1);
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout << "INFO: synapse statistics and memory usage for synapses_2_pre:\n"
        << "\tnumber of synapses: " << syn_N << "\n"
        << "\tnumber of pre/post blocks: " << num_pre_post_blocks << "\n"
        << "\tnumber of synapses over all pre/post blocks:\n"
        << "\t\tmean: " << mean_num_elements << "\tstd: "
            << std_num_elements << "\n"
    << "\n\tmemory usage: TOTAL: " << total_memory_MB << " MB (~"
        << total_memory_MB / syn_N * 1024.0 * 1024.0  << " byte per synapse)"
        << std::endl;

    for(auto const& tuple: memory_recorder){
        std::string name;
        size_t bytes;
        int num_elements;
        std::tie(name, bytes, num_elements) = tuple;
        double memory = bytes * to_MB;
        double fraction = memory / total_memory_MB * 100;
        std::cout << "\t\t" << std::setprecision(1) << std::fixed << fraction
            << "%\t" << std::setprecision(3) << std::fixed << memory << " MB\t"
            << name << " [" << num_elements << "]" << std::endl;
    }


    // Create circular eventspaces in no_or_const_delay_mode
    {
        int num_eventspaces = dev_array_neurongroup_1__spikespace.size();
        bool require_new_eventspaces = (num_queues > num_eventspaces);

        if (require_new_eventspaces)
        {
            // rotate circular eventspace such that the current idx is at the start
            // (logic copied from CSpikeQueue.expand() in Brian's cspikequeue.cpp)
            std::rotate(
                dev_array_neurongroup_1__spikespace.begin(),
                dev_array_neurongroup_1__spikespace.begin() + current_idx_array_neurongroup_1__spikespace,
                dev_array_neurongroup_1__spikespace.end()
            );
            current_idx_array_neurongroup_1__spikespace = 0;
            // add new eventspaces
            for (int i = num_eventspaces; i < num_queues; i++)
            {
                int32_t* new_eventspace;
                CUDA_SAFE_CALL(
                    cudaMalloc(
                        (void**)&new_eventspace,
                        sizeof(int32_t) * _num__array_neurongroup_1__spikespace
                    )
                );
                // initialize device eventspace with -1 and counter with 0
                CUDA_SAFE_CALL(
                    cudaMemcpy(
                        new_eventspace,
                        _array_neurongroup_1__spikespace,  // defined in objects.cu
                        sizeof(int32_t) * _num__array_neurongroup_1__spikespace,
                        cudaMemcpyHostToDevice
                    )
                );
                dev_array_neurongroup_1__spikespace.push_back(new_eventspace);
            }
        }
    }

    int num_threads = num_queues;
    if(num_threads >= max_threads_per_block)
    {
        num_threads = max_threads_per_block;
    }
    int num_blocks = 1;

    // check if we have enough ressources to call kernel with given number
    // of blocks and threads
    struct cudaFuncAttributes funcAttrib;
    cudaFuncGetAttributes(&funcAttrib, _before_run_kernel_synapses_2_pre_push_spikes);
    if (num_threads > funcAttrib.maxThreadsPerBlock)
    {
        // use the max num_threads before launch failure
        num_threads = funcAttrib.maxThreadsPerBlock;
        printf("WARNING Not enough ressources available to call "
               "_before_run_kernel_synapses_2_pre_push_spikes"
               "with maximum possible threads per block (%u). "
               "Reducing num_threads to %u. (Kernel needs %i "
               "registers per block, %i bytes of "
               "statically-allocated shared memory per block, %i "
               "bytes of local memory per thread and a total of %i "
               "bytes of user-allocated constant memory)\n",
               max_threads_per_block, num_threads, funcAttrib.numRegs,
               funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
               funcAttrib.constSizeBytes);
    }
    else
    {
        printf("INFO _before_run_kernel_synapses_2_pre_push_spikes\n"
               "\t%u blocks\n"
               "\t%u threads\n"
               "\t%i registers per thread\n"
               "\t%i bytes statically-allocated shared memory per block\n"
               "\t%i bytes local memory per thread\n"
               "\t%i bytes user-allocated constant memory\n"
               "",
               num_blocks, num_threads, funcAttrib.numRegs,
               funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
               funcAttrib.constSizeBytes);
    }

    _before_run_kernel_synapses_2_pre_push_spikes<<<num_blocks, num_threads>>>(
        source_N,
        num_parallel_blocks,
        num_threads,
        dt,
        syn_N,
        num_delays,
        true
    );

    {
        delete [] h_num_synapses_by_pre;
        delete [] h_ptr_d_ptr_synapse_ids_by_pre;
    }

    //delete temp arrays
    delete [] h_vec_synapse_ids_by_pre;

    synapses_2_pre_scalar_delay = true;

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        printf("ERROR initialising synapses_2_pre in %s:%d %s\n",
                __FILE__, __LINE__, cudaGetErrorString(status));
        _dealloc_arrays();
        exit(status);
    }

    CUDA_CHECK_MEMORY();
    double time_passed = (double)(std::clock() - start_timer) / CLOCKS_PER_SEC;
    std::cout << "INFO: synapses_2_pre initialisation took " <<  time_passed << "s";
    if (used_device_memory_after_dealloc < used_device_memory_start){
        size_t freed_bytes = used_device_memory_start - used_device_memory_after_dealloc;
        std::cout << ", freed " << freed_bytes * to_MB << "MB";
    }
    if (used_device_memory > used_device_memory_start){
        size_t used_bytes = used_device_memory - used_device_memory_start;
        std::cout << " and used " << used_bytes * to_MB << "MB of device memory.";
    }
    std::cout << std::endl;

    first_run = false;
}
