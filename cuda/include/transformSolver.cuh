#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "struct.cuh"

namespace transformsolver
{
	//__device__ void rotateAndShift(
	//	float* cx,
	//	float* cy,
	//	int a,
	//	int b,
	//	int c,
	//	const S_IMG_PROC_KERNEL_PARAMS& params);

	__device__ void initSolver(
		float* const xm, float* const ym,
		float* const xi, float* const yi,
		float* const offsetTheta,
		float* const offsetX,
		float* const offsetY,		
		unsigned int* nvalidPoints,
		const S_IMG_PROC_KERNEL_PARAMS& params);

	__device__ void calcCartesianDist(double distx, double disty,
		const S_IMG_PROC_KERNEL_PARAMS& params,
		double& xc, double& yc);
	__device__ void rotateBackOrigin(double xin, double yin,
		double &xout, double &yout,
		const S_IMG_PROC_KERNEL_PARAMS& params);

	__device__ int getGlobalId();

	__device__ void setCostPoints(double* xc, double* yc, double* tc,
		double* currentIntervall,
		double* currentMinimum);

	__device__ void calculateCostsImage(double* costs, int* costIndex, int n,
		float* xmodel, float* ymodel,
		float* xmeas, float* ymeas,
		double* xc, double* yc, double* tc,
		const S_IMG_PROC_KERNEL_PARAMS& params);

	__device__ void findMinimumCosts(double* costs, int* costIndex,
		double* xc, double* yc, double* tc,
		double* currentMinimum, double* minimumCosts);


}