#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "struct.cuh"

namespace loadandstore2bufer
{

	__device__  void store2buffer(unsigned char* const buffer,
		unsigned char* const mask_buffer,
		S_IMG_PROC_KERNEL_PARAMS& params);


	__device__ void readFrombuffer(const unsigned char* const buffer,
							 int bufferId,
							S_IMG_PROC_KERNEL_PARAMS& params);

	__device__ void readFrombuffermask(const unsigned char* const buffer,
		int bufferId,
		S_IMG_PROC_KERNEL_PARAMS& params);

	__device__ void initializebuffer(unsigned char* image_buffer, unsigned char* mask_buffer, unsigned char* prop_buffer, S_IMG_PROC_KERNEL_PARAMS& params);
}