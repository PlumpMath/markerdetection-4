#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "struct.cuh"

namespace matchpattern
{
	__device__ void readBinMask(unsigned char* binmask,  const S_IMG_PROC_KERNEL_PARAMS& params);
	__device__ bool calculatePatternScore(
		const unsigned char* img_buffer,
		unsigned char* mask_buffer,
		const unsigned char* pattern_buffer,
		const S_IMG_PROC_KERNEL_PARAMS& params,
		int startIdx, int startIdy, int startId,
		int imageX, int imageY);
}
