#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "struct.cuh"

namespace edgedetection
{
	__device__ bool calculateEdge(
		const unsigned char* buffer,
		unsigned char* mask_buffer,
		const S_IMG_PROC_KERNEL_PARAMS& params,
		int startIdx, int startIdy, int startId);
}