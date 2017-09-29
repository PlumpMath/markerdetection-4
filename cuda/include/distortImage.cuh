#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "struct.cuh"
#include <vector>

namespace distortimage
{


	void initMasksAndMap(const double* Cin, const double* Hin, const double* kkin,
		int resx, int resy,
		std::vector<std::vector<double>> holeContour,
		DeviceBuffer<int>* index_x, DeviceBuffer<int>* index_y,
		DeviceBuffer<unsigned char>* mask);

	__device__ void undistort(const unsigned char* mask, const int* mapx, const int* mapy,
		const unsigned char* img, unsigned char* res, S_IMG_PROC_KERNEL_PARAMS& params);

}