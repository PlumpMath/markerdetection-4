#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "struct.cuh"

namespace clusterdetection
{
	__device__ void calculateCluster(
		unsigned char* mask_buffer,
		const unsigned char* pattern_buffer,
		const unsigned char* prop_buffer,
		const S_IMG_PROC_KERNEL_PARAMS& params,
		int startIdx, int startIdy, int startId,
		int realImageThreadx, int realImageThready);
}