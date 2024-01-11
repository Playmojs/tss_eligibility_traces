#include "code_objects/neurongroup_2_spike_thresholder_codeobject.h"
#include "objects.h"
#include "brianlib/common_math.h"
#include "brianlib/cuda_utils.h"
#include "brianlib/stdint_compat.h"
#include <cmath>
#include <stdint.h>
#include <ctime>
#include <stdio.h>


void _after_run_neurongroup_2_spike_thresholder_codeobject()
{
    using namespace brian;

///// HOST_CONSTANTS ///////////
const int32_t N = 13;
		const int _num_spikespace = 14;
		const int _numv = 13;

const int _N = N;
int num_threads, num_blocks;
int min_num_threads; // The minimum grid size needed to achieve the maximum occupancy for a full device launch

CUDA_SAFE_CALL(
        cudaOccupancyMaxPotentialBlockSize(&min_num_threads, &num_threads,
            _reset_neurongroup_2_spike_thresholder_codeobject, 0, 0)  // last args: dynamicSMemSize, blockSizeLimit
        );

// Round up according to array size
num_blocks = (_N + num_threads - 1) / num_threads;

_reset_neurongroup_2_spike_thresholder_codeobject<<<num_blocks, num_threads>>>(
        dev_array_neurongroup_2__spikespace[current_idx_array_neurongroup_2__spikespace]
    );

CUDA_CHECK_ERROR("_reset_neurongroup_2_spike_thresholder_codeobject");
}
