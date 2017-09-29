#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "struct.cuh"

namespace demosaicing
{
	__device__ void loadAndMinMax(unsigned char* minG, unsigned char* maxG,
		unsigned char* minR, unsigned char* maxR,
		unsigned char* minB, unsigned char* maxB,
		const S_IMG_PROC_KERNEL_PARAMS& params);
}