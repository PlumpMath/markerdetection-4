#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "struct.cuh"

namespace markerrecognition
{

	__device__  void recognizeMarker(
		unsigned int* const intBuffer,
		unsigned char* const charBuffer,
		int* const maskIdBuffer,
		const unsigned char* const buffer,
		int* const markerId,
		int* const offsetA,
		int* const offsetB,
		const S_IMG_PROC_KERNEL_PARAMS& params);



}