#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "struct.cuh"

namespace rotateAndShift
{

	__device__ void binarizeMarker(
		float* const origin,
		float* const nx,
		float* const ny,
		float* const nxp,
		float* const nyp,
		float &dist_px_x,
		float &dist_px_y,
		unsigned char* const img_buffer,
		float* const edges,
		float* const corners,
		float* const a,
		float* const b,
		float* const c,
		const S_IMG_PROC_KERNEL_PARAMS& params);


}