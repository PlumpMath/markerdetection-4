#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "struct.cuh"

namespace markercorners
{
	__device__ void calcCorners(
		float* cx,
		float* cy,
		int a,
		int b,
		int c,
		int nCentroids,
		unsigned int* p_marker,
		const S_IMG_PROC_KERNEL_PARAMS& params);
}