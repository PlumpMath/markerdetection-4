#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "struct.cuh"

namespace radiussearch
{

	__device__ void radiussearch(int* centerX, int* centerY, unsigned char* prop, unsigned int nCenter, float* centerMergedx, float* centerMergedy, unsigned int *nMerged, const S_IMG_PROC_KERNEL_PARAMS& params);
	int calculateSharedMemorySize();

}