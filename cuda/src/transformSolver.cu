
#include "struct.cuh"
#include "transformSolver.cuh"

namespace transformsolver
{


	__device__ void getLocalId(int *x, int* y, int* z,
		int threadId)
	{
		int idz, idy, idx;

		idz = threadId / (blockDim.y * blockDim.x);
		idy = (threadId - idz*blockDim.y * blockDim.x) / blockDim.x;
		idx = threadId - idz*blockDim.y * blockDim.x - idy*blockDim.x;

		*x = idx;
		*y = idy;
		*z = idz;
	}

	__device__ int getGlobalId()
	{
		return threadIdx.z * blockDim.y * blockDim.x
			+ threadIdx.y * blockDim.x 
			+ threadIdx.x;
	}

	__device__ void findMinimumCosts(double* costs, int* costIndex,
									 double* xc, double* yc, double* tc,
									 double* currentMinimum, double* minimumCosts)
	{
		unsigned int tid = getGlobalId();
		unsigned int dim = N_SOLVER_DIMENSION_SIZE;
		unsigned int start = N_SOLVER_DIMENSION_SIZE / 2;

		int miniumIndex = tid;
		double minimumCost = DBL_MAX;
		int localX, localY, localZ;
		//if (tid == 0)
		//{

		//}


		////for (int i = start; i > 0;  i >>= 1)
		////{
		////	__syncthreads();
		////	if (tid < i && tid+i<N_SOLVER_DIMENSION_SIZE)
		////	{
		////		if (costs[tid] > costs[tid + i])
		////		{
		////			costs[tid] = costs[tid + i];
		////			costIndex[tid] = costIndex[tid + i];
		////		}

		////	}
		////}
		//__syncthreads();

	
		if (tid == 0)
		{
			miniumIndex = 0;
			for (int i = 0; i < N_SOLVER_DIMENSION_SIZE; i++)
			{
				if (costs[i] < minimumCost)
				{
					miniumIndex = i;
					minimumCost = costs[i];
				}

			}

			getLocalId(&localX, &localY, &localZ, miniumIndex);
			currentMinimum[0] = xc[localX];
			currentMinimum[1] = yc[localY];
			currentMinimum[2] = tc[localZ];
			minimumCosts[0] = costs[miniumIndex];

			//getLocalId(&localX, &localY, &localZ, costIndex[tid]);
			//currentMinimum[0] = xc[localX];
			//currentMinimum[1] = yc[localY];
			//currentMinimum[2] = tc[localZ];
			//minimumCosts[0] = costs[0];
		}
		__syncthreads();
	}

	__device__ void projectIntoCartesian(double xi, double yi, const S_IMG_PROC_KERNEL_PARAMS& params, double& xc, double& yc)
	{		
		xc = (params.cameraParameter.dist * (xi - params.cameraParameter.C[2])) / params.cameraParameter.C[0];
		yc = (params.cameraParameter.dist * (yi - params.cameraParameter.C[5])) / params.cameraParameter.C[4];
	}

	__device__ void rotateBack(double xin, double yin, double &xout, double &yout, const S_IMG_PROC_KERNEL_PARAMS& params)
	{
		double xdif = xin - params.cameraParameter.t[0];
		double ydif = yin - params.cameraParameter.t[1];		
		xout = params.cameraParameter.Ri[0] * xdif + params.cameraParameter.Ri[1] * ydif ;
		yout = params.cameraParameter.Ri[3] * xdif + params.cameraParameter.Ri[4] * ydif ;
	}

	__device__ void rotateBackOrigin(double xin, double yin, double &xout, double &yout, const S_IMG_PROC_KERNEL_PARAMS& params)
	{
		xout = params.cameraParameter.Ri[0] * xin + params.cameraParameter.Ri[1] * yin;
		yout = params.cameraParameter.Ri[3] * xin + params.cameraParameter.Ri[4] * yin;
	}

	__device__ void projectMapPointIntoImage(double xmap, double ymap, double xr, double yr, double tr, const S_IMG_PROC_KERNEL_PARAMS& params, double &xi, double& yi)
	{
		//double rot[4];

		double ct = cos(tr);
		double st = sin(tr);

		//rot[0] = ct;
		//rot[1] = -st;
		//rot[2] = st;
		//rot[3] = ct;


		double xcar = xmap  * ct - ymap  *st + xr;
		double ycar = xmap  * st + ymap  * ct + yr;
		double zcar = 0;



		double xc = params.cameraParameter.H[3] + params.cameraParameter.H[0] *xcar + params.cameraParameter.H[1] *ycar + params.cameraParameter.H[2] *zcar;
		double yc = params.cameraParameter.H[7] + params.cameraParameter.H[4] *xcar + params.cameraParameter.H[5] *ycar + params.cameraParameter.H[6] *zcar;
		double zc = params.cameraParameter.H[11] + params.cameraParameter.H[8] *xcar + params.cameraParameter.H[9] *ycar + params.cameraParameter.H[10] *zcar;


		double x = params.cameraParameter.C[0] * xc + params.cameraParameter.C[2] * zc;
		double y = params.cameraParameter.C[4] *  yc + params.cameraParameter.C[5] * zc;


		double imx = x / zc;
		double imy = y / zc;

		xi = imx;
		yi = imy;

		//double st = sin(tr);
		//double ct = cos(tr);
		//double z = params.cameraParameter.H[11] +
		//	params.cameraParameter.H[8] * (xr + ct*xmap - st*ymap) +
		//	params.cameraParameter.H[9] * (yr + ct*ymap + st*xmap);
		//
		//xi =( params.cameraParameter.C[0] * (params.cameraParameter.H[3] +
		//	params.cameraParameter.H[0] * (xr + ct*xmap - st*ymap) +
		//	params.cameraParameter.H[1] * (yr + ct*ymap + st*xmap)) +
		//	params.cameraParameter.C[1] * (params.cameraParameter.H[11] +
		//	params.cameraParameter.H[8] * (xr + ct*xmap - st*ymap) + 
		//	params.cameraParameter.H[9] * (yr + ct*ymap + st*xmap)))/z;

		//yi = (params.cameraParameter.C[0]*(params.cameraParameter.H[3] + 
		//	params.cameraParameter.H[0]*(xr + ct*xmap - st*ymap) +
		//	params.cameraParameter.H[1]*(yr + ct*ymap + st*xmap)) +
		//	params.cameraParameter.C[2]*(params.cameraParameter.H[11] +
		//	params.cameraParameter.H[8]*(xr + ct*xmap - st*ymap) + 
		//	params.cameraParameter.H[9]*(yr + ct*ymap + st*xmap)))/z;


	}



	__device__ void calcCartesianDist(double distx, double disty,  const S_IMG_PROC_KERNEL_PARAMS& params, double& xc, double& yc)
	{		
		double xc0, yc0;		
		projectIntoCartesian(distx, disty, params, xc0, yc0);
		rotateBack(xc0, yc0, xc, yc, params);
	}

	__device__ void calculateCostsImage(double* costs, int* costIndex, int n,
		float* xmodel, float* ymodel,
		float* xmeas, float* ymeas,
		double* xc, double* yc, double* tc,
		const S_IMG_PROC_KERNEL_PARAMS& params)
	{
		int threadId = getGlobalId();
		double c = 0;
		double dx, dy;
		double xi, yi;
		double xin, yin, tin;

		double xinput = xc[threadIdx.x];
		double yinput = yc[threadIdx.y];
		double tinput = tc[threadIdx.z];

		//printf(": %d\t%lf\t%lf\t%lf\n", threadId, xinput, yinput, tinput);
		for (int i = 0; i < n; i++)
		{
			

			projectMapPointIntoImage(xmodel[i], ymodel[i], xinput, yinput, tinput, params, xi, yi);

			dx = xmeas[i] - xi;
			dy = ymeas[i] - yi;

			c += sqrt(dx*dx + dy*dy);
		}


		
		costs[threadId]		= c;
		costIndex[threadId] = threadId;
	

		__syncthreads();
	}


	__device__ void calculateCosts(	double* costs, int* costIndex, int n,
									float* xmodel, float* ymodel,
									float* xmeas, float* ymeas,
									double* xc, double* yc, double* tc)
	{
		float c = 0;
		float dx, dy;
		float x, y, ct, st;
		float rotMatrix[4];
		float offset[2];

		ct = cos(tc[threadIdx.z]);
		st = sin(tc[threadIdx.z]);
		offset[0] = xc[threadIdx.x];
		offset[1] = yc[threadIdx.y];

		rotMatrix[0] = ct;
		rotMatrix[1] = -st;
		rotMatrix[2] = st;
		rotMatrix[3] = ct;

		
		for (int i = 0; i < n; i++)
		{
			x = xmeas[i] *rotMatrix[0] + ymeas[i] *rotMatrix[1] + offset[0];
			y = xmeas[i] *rotMatrix[2] + ymeas[i] *rotMatrix[3] + offset[1];

			dx = xmodel[i] - x;
			dy = ymodel[i] - y;

			c += dx*dx + dy*dy;
		}


		int threadId = getGlobalId();
		costs[threadId] = c;
		costIndex[threadId] = threadId;

		__syncthreads();
	}
	
	__device__ void setCostPoints(double* xc, double* yc, double* tc,
		double* currentIntervall,
		double* currentMinimum)
	{
		double dx, dy, dt;
		double val;
		int tid = getGlobalId();
		double dtid = ((double)tid);
		if (tid < N_SOLVER_DIMENSION)
		{		
			

			dx = currentIntervall[0] /((double) (N_SOLVER_DIMENSION - 1));
			val = dx*dtid - (currentIntervall[0] / 2.0) + currentMinimum[0];
			xc[tid] = val;
			//printf("%lf\n", val);

			dy = currentIntervall[1] / ((double)(N_SOLVER_DIMENSION - 1));
			val = dy*dtid - (currentIntervall[1] / 2.0) + currentMinimum[1];
			yc[tid] = val;
			//printf("%lf\n", val);

			dt = currentIntervall[2] / ((double)(N_SOLVER_DIMENSION - 1));
			val = dt*dtid - (currentIntervall[2] / 2.0) + currentMinimum[2];
			tc[tid] = val;
			//printf("%lf\n", val);

			//printf(": %d\t%lf\t%lf\t%lf\n", threadId, xinput, yinput, tinput);

		}
		__syncthreads();
		
	}

	__device__ void projectIntoImagePlane(double xr,  const S_IMG_PROC_KERNEL_PARAMS& params)
	{
		
		//c0*(h3 + h0*(xr + ct*xmap - st*ymap) + h1*(yr + ct*ymap + st*xmap)) + c1*(h11 + h8*(xr + ct*xmap - st*ymap) + h9*(yr + ct*ymap + st*xmap))
	}

	

	__device__ void initSolver(
		float* const xm, float* const ym,
		float* const xi, float* const yi,
		float* const offsetTheta, 
		float* const offsetX,
		float* const offsetY,
		unsigned int* nvalidPoints,
		const S_IMG_PROC_KERNEL_PARAMS& params)
	{
		int threadId = getGlobalId();


		int markerId, offsetA, offsetB;
		unsigned int index;

		
		float rotMatrix[4];		
		float xbc_A, ybc_A;
		float xbc_C, ybc_C;
		float xr, yr;
		float ct, st;
		int xindex;
		int yindex;
		float diffx, diffy, tmp;

		bool isInRange = false;
		bool isLoaded = false;

		if (threadId < *params.centroidParameter.nMarker)
		{
			isInRange = true;
			//first check if marker has been detected and decoded
			markerId = params.markerParameter.markerID[threadId];
			offsetA = params.markerParameter.markerOffsetA[threadId];
			offsetB = params.markerParameter.markerOffsetB[threadId];


			if (markerId != INT_NAN &&
				offsetA != INT_NAN &&
				offsetB != INT_NAN)
			{
				isLoaded = true;

				xindex = (offsetA + QR_MARKER_INDEX_OFFSET) * 3;
				yindex = (offsetB + QR_MARKER_INDEX_OFFSET) * 3;

				// store, diffx, diffy, difftheta for averaging
				index = atomicAdd(nvalidPoints, 3);

				xbc_A = params.mapParameter.ix[yindex*QR_MARKER_DIM + xindex + 0];
				ybc_A = params.mapParameter.iy[yindex*QR_MARKER_DIM + xindex + 0];

				xbc_C = params.mapParameter.ix[yindex*QR_MARKER_DIM + xindex + 2];
				ybc_C = params.mapParameter.iy[yindex*QR_MARKER_DIM + xindex + 2];


				xi[index + 0] = params.centroidParameter.centerAx[threadId];
				yi[index + 0] = params.centroidParameter.centerAy[threadId];

				xi[index + 1] = params.centroidParameter.centerBx[threadId];
				yi[index + 1] = params.centroidParameter.centerBy[threadId];

				xi[index + 2] = params.centroidParameter.centerCx[threadId];
				yi[index + 2] = params.centroidParameter.centerCy[threadId];


				xm[index + 0] = params.mapParameter.cx[yindex*QR_MARKER_DIM + xindex + 0];
				ym[index + 0] = params.mapParameter.cy[yindex*QR_MARKER_DIM + xindex + 0];

				xm[index + 1] = params.mapParameter.cx[yindex*QR_MARKER_DIM + xindex + 1];
				ym[index + 1] = params.mapParameter.cy[yindex*QR_MARKER_DIM + xindex + 1];

				xm[index + 2] = params.mapParameter.cx[yindex*QR_MARKER_DIM + xindex + 2];
				ym[index + 2] = params.mapParameter.cy[yindex*QR_MARKER_DIM + xindex + 2];
		
				float theta_image	= atan2f(yi[index + 2] - yi[index + 0], xi[index + 2] - xi[index + 0]);
				float theta_cart	= atan2f(ybc_C - ybc_A, xbc_C - xbc_A);
				float theta_diff = theta_image - theta_cart;

				//rotate cartesian point into image system
				ct = cos(theta_diff);
				st = sin(theta_diff);

				rotMatrix[0] = ct;
				rotMatrix[1] = -st;
				rotMatrix[2] = st;
				rotMatrix[3] = ct;

				xr = xbc_A * rotMatrix[0] + ybc_A * rotMatrix[1];
				yr = xbc_A * rotMatrix[2] + ybc_A * rotMatrix[3];

				offsetTheta[index / 3] = theta_diff;
				offsetX[index / 3] = xi[index + 0] - xr;
				offsetY[index / 3] = yi[index + 0] - yr;
				
			}
		}

		__syncthreads();

		float sumX = 0.f, sumY = 0.f, sumTheta = 0.f;
		if (threadId == 0)
		{
			int maxIndex = (*nvalidPoints) / 3;
			for (int i = 0; i < maxIndex; i++)
			{
				sumX += offsetX[i];
				sumY += offsetY[i];
				sumTheta += offsetTheta[i];
			}
			offsetX[0]		= sumX / ((float)maxIndex);
			offsetY[0]		= sumY / ((float)maxIndex);
			offsetTheta[0]	= sumTheta / ((float)maxIndex);
		}

		__syncthreads();

		//if (isInRange && !isLoaded)
		//{

		//	index = atomicAdd(nvalidPoints, 3);

		//	xi[index + 0] = params.centroidParameter.centerAx[threadId];
		//	yi[index + 0] = params.centroidParameter.centerAy[threadId];

		//	xi[index + 1] = params.centroidParameter.centerBx[threadId];
		//	yi[index + 1] = params.centroidParameter.centerBy[threadId];

		//	xi[index + 2] = params.centroidParameter.centerCx[threadId];
		//	yi[index + 2] = params.centroidParameter.centerCy[threadId];


		//	int bestOffsetA, bestOffsetB;
		//	float dist = FLT_MAX;
		//	for (int offsetA_tmp = -QR_MARKER_INDEX_OFFSET; offsetA_tmp <= QR_MARKER_INDEX_OFFSET; offsetA_tmp++)
		//	{
		//		for (int offsetB_tmp = -QR_MARKER_INDEX_OFFSET; offsetB_tmp <= QR_MARKER_INDEX_OFFSET; offsetB_tmp++)
		//		{
		//			if (offsetA_tmp == 0 || offsetA_tmp == 0)
		//				continue;

		//			xindex = (offsetA_tmp + QR_MARKER_INDEX_OFFSET) * 3;
		//			yindex = (offsetB_tmp + QR_MARKER_INDEX_OFFSET) * 3;

		//			xbc_A = params.mapParameter.ix[yindex*QR_MARKER_DIM + xindex + 0];
		//			ybc_A = params.mapParameter.iy[yindex*QR_MARKER_DIM + xindex + 0];


		//			//rotate cartesian point into image system
		//			ct = cos(offsetTheta[0]);
		//			st = sin(offsetTheta[0]);

		//			rotMatrix[0] = ct;
		//			rotMatrix[1] = -st;
		//			rotMatrix[2] = st;
		//			rotMatrix[3] = ct;

		//			xr = xbc_A * rotMatrix[0] + ybc_A * rotMatrix[1] + offsetX[0];
		//			yr = xbc_A * rotMatrix[2] + ybc_A * rotMatrix[3] + offsetY[0];
		//			
		//			diffx = xi[index + 0] - xr;
		//			diffy = yi[index + 0] - yr;

		//			tmp = sqrtf(diffx*diffx + diffy*diffy);
		//			if (tmp < dist)
		//			{
		//				dist = tmp;
		//				bestOffsetA = offsetA_tmp;
		//				bestOffsetB = offsetB_tmp;
		//			}


		//		}
		//	}

		//	xindex = (bestOffsetA + QR_MARKER_INDEX_OFFSET) * 3;
		//	yindex = (bestOffsetB + QR_MARKER_INDEX_OFFSET) * 3;


		//	xm[index + 0] = params.mapParameter.cx[yindex*QR_MARKER_DIM + xindex + 0];
		//	ym[index + 0] = params.mapParameter.cy[yindex*QR_MARKER_DIM + xindex + 0];

		//	xm[index + 1] = params.mapParameter.cx[yindex*QR_MARKER_DIM + xindex + 1];
		//	ym[index + 1] = params.mapParameter.cy[yindex*QR_MARKER_DIM + xindex + 1];

		//	xm[index + 2] = params.mapParameter.cx[yindex*QR_MARKER_DIM + xindex + 2];
		//	ym[index + 2] = params.mapParameter.cy[yindex*QR_MARKER_DIM + xindex + 2];




		//}
		__syncthreads();


	}


	__global__  void testSolver(S_IMG_PROC_KERNEL_PARAMS params)
	{

		__shared__ float xm[N_MAX_SOLVER_POINTS];
		__shared__ float ym[N_MAX_SOLVER_POINTS];

		__shared__ float xmeas[N_MAX_SOLVER_POINTS];
		__shared__ float ymeas[N_MAX_SOLVER_POINTS];

		__shared__ double xc[N_SOLVER_DIMENSION];
		__shared__ double yc[N_SOLVER_DIMENSION];
		__shared__ double tc[N_SOLVER_DIMENSION];

		__shared__ double costs[N_SOLVER_DIMENSION_SIZE];
		__shared__ int costs_index[N_SOLVER_DIMENSION_SIZE];


		__shared__ float rotMatrix[4];
		__shared__ float offset[2];

		__shared__ double currentIntervall[3];
		__shared__ double currentMinimum[3];
		__shared__ double currentMinimumCost;
		__shared__ int    currentIteration;
		


		float ct, st;


		int nPoints = *params.solverParameter.nPoints;
		int threadId = getGlobalId();

		if (threadId == 0)
		{
			ct = cos(params.solverParameter.tinit);
			st = sin(params.solverParameter.tinit);
			offset[0] = params.solverParameter.xinit;
			offset[1] = params.solverParameter.yinit;

			rotMatrix[0] = ct;
			rotMatrix[1] = -st;
			rotMatrix[2] = st;
			rotMatrix[3] = ct;

			currentIntervall[0] = params.solverParameter.xinit_intervall;
			currentIntervall[1] = params.solverParameter.yinit_intervall;
			currentIntervall[2] = params.solverParameter.tinit_intervall;

			currentMinimum[0] = 0.f;
			currentMinimum[1] = 0.f;
			currentMinimum[2] = 0.f;

			currentIteration = 0;

			currentMinimumCost = FLT_MAX;

		}


		__syncthreads();
		float x, y;
		if (threadId < nPoints)
		{
			xm[threadId] = params.solverParameter.xm[threadId];
			ym[threadId] = params.solverParameter.ym[threadId];

			x = params.solverParameter.xmeas[threadId];
			y = params.solverParameter.ymeas[threadId];

			xmeas[threadId] = x*rotMatrix[0] + y*rotMatrix[1] + offset[0];
			ymeas[threadId] = x*rotMatrix[2] + y*rotMatrix[3] + offset[1];
		}
		
		__syncthreads();




		//setting running optimisation
		while (currentIteration < params.solverParameter.maxIter &&
			currentMinimumCost > params.solverParameter.eps)
		{
			setCostPoints(xc, yc, tc, currentIntervall, currentMinimum);
			calculateCosts(costs,costs_index, nPoints, xm, ym, xmeas, ymeas,xc, yc, tc);
			findMinimumCosts(costs,costs_index, xc, yc, tc, currentMinimum, &currentMinimumCost);

			if (threadId == 0)
			{
				currentIntervall[0] /= 0.7072f*(float)(N_SOLVER_DIMENSION-1);
				currentIntervall[1] /= 0.7072f*(float)(N_SOLVER_DIMENSION-1);
				currentIntervall[2] /= 0.7072f*(float)(N_SOLVER_DIMENSION-1);
				currentIteration++;
			}
			__syncthreads();
		}
		
		if (threadId == 0)
		{
			//forward transformation of results
			x = currentMinimum[0];
			y = currentMinimum[1];

			params.solverParameter.optimalTranformation[0] = x*rotMatrix[0] + y*rotMatrix[1] + offset[0];
			params.solverParameter.optimalTranformation[1] = x*rotMatrix[2] + y*rotMatrix[3] + offset[1];
			params.solverParameter.optimalTranformation[2] = params.solverParameter.tinit +currentMinimum[2];


			params.solverParameter.optimalTranformation[0] = currentMinimum[0];
			params.solverParameter.optimalTranformation[1] = currentMinimum[1];
			params.solverParameter.optimalTranformation[2] = currentMinimum[2];
			params.solverParameter.minimumCosts[0] = currentMinimumCost;
		}
						
	}


	struct TransformSolverTest : testing::Test
	{




		S_IMG_PROC_KERNEL_PARAMS		params;
		S_LAUNCH_CONFIG					launchConfig;
		
		
		Data2D*		model;
		Data2D*		measurement;

		DeviceBuffer<double> *optimalTransform;
		DeviceBuffer<double> *optimalCosts;
		DeviceBuffer<int> *nPoints;


		TransformSolverTest()
		{

			model = new Data2D(48);
			measurement = new Data2D(48);

			optimalTransform = new DeviceBuffer<double>(3);
			optimalCosts = new DeviceBuffer<double>(1);
			nPoints = new DeviceBuffer<int>(1);

			int offset = 45;
			int dist = 25;

			int cx, cy;
			int i = 0;
			int d = 3;

			for (int x = 0; x < 4; x++)
			{
				for (int y = 0; y < 4; y++)
				{
					cx = x * 25 - 45;
					cy = y * 25 - 45;

					model->x->getHostData()[i] = cx - d;
					model->y->getHostData()[i] = cy - d;
					i++;

					model->x->getHostData()[i] = cx + d;
					model->y->getHostData()[i] = cy - d;
					i++;

					model->x->getHostData()[i] = cx - d;
					model->y->getHostData()[i] = cy + d;
					i++;
				}
			}

			nPoints->getHostData()[0] = 48;
			nPoints->set();

			params.solverParameter.nPoints = nPoints->getDeviceData();

			params.solverParameter.xm = model->x->getDeviceData();
			params.solverParameter.ym = model->y->getDeviceData();

			params.solverParameter.xmeas = measurement->x->getDeviceData();
			params.solverParameter.ymeas = measurement->y->getDeviceData();

			params.solverParameter.xinit = 0.0f;
			params.solverParameter.yinit = 0.0f;
			params.solverParameter.tinit = 0.0f;

			params.solverParameter.xinit_intervall = 2.0;
			params.solverParameter.yinit_intervall = 2.0;
			params.solverParameter.tinit_intervall = 0.4;

			params.solverParameter.eps = 1e-5;
			params.solverParameter.maxIter = 1000;

			params.solverParameter.optimalTranformation = optimalTransform->getDeviceData();
			params.solverParameter.minimumCosts = optimalCosts->getDeviceData();


			launchConfig.grid = dim3(1);
			launchConfig.block = dim3(N_SOLVER_DIMENSION, N_SOLVER_DIMENSION, N_SOLVER_DIMENSION);

			cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

		}

		void runTest()
		{

		}

		~TransformSolverTest()
		{
			delete model;
			delete measurement;
			delete optimalTransform;
			delete optimalCosts;
			delete nPoints;
		}
	};

	TEST_F(TransformSolverTest, transformSolver)
	{

		//bufferMarker->set(lTestMarker[i].marker);

		float xoffset = 0.5;
		float yoffset = 0.5;
		float t = 0.1;

		float ct = cos(t);
		float st = sin(t);

		float rotMatrix[4];
		rotMatrix[0] = ct;
		rotMatrix[1] = -st;
		rotMatrix[2] = st;
		rotMatrix[3] = ct;

		float rotMatrixInverse[4];
		rotMatrixInverse[0] = ct;
		rotMatrixInverse[1] = st;
		rotMatrixInverse[2] = -st;
		rotMatrixInverse[3] = ct;

		float x, y, xr, yr;




		for (int i = 0; i < nPoints->getHostData()[0]; i++)
		{
			x = model->x->getHostData()[i]- xoffset;
			y = model->y->getHostData()[i]- yoffset;

			xr = x*rotMatrixInverse[0] + y*rotMatrixInverse[1];
			yr = x*rotMatrixInverse[2] + y*rotMatrixInverse[3];

			measurement->x->getHostData()[i] = xr;
			measurement->y->getHostData()[i] = yr;
		}

		model->set();
		measurement->set();


		//run cuda
		testSolver << <launchConfig.grid, launchConfig.block >> > (params);
		cudaCheckError();

		cudaDeviceSynchronize();
		cudaCheckError();

		optimalTransform->get();


		xoffset = optimalTransform->getHostData()[0];
		yoffset = optimalTransform->getHostData()[1];
		ct = cos(optimalTransform->getHostData()[2]);
		st = sin(optimalTransform->getHostData()[2]);
		rotMatrix[0] = ct;
		rotMatrix[1] = -st;
		rotMatrix[2] = st;
		rotMatrix[3] = ct;

		float cost = 0;
		float diffx, diffy;
		for (int i = 0; i < model->size->getHostData()[0]; i++)
		{
			x = measurement->x->getHostData()[i];
			y = measurement->y->getHostData()[i];

			xr = x*rotMatrix[0] + y*rotMatrix[1] + xoffset;
			yr = x*rotMatrix[2] + y*rotMatrix[3] + yoffset;

			diffx = model->x->getHostData()[i] - xr;
			diffy = model->y->getHostData()[i] - yr;
			cost += diffx*diffx + diffy*diffy;
		}
		EXPECT_LE(cost, params.solverParameter.eps);
		
		//cudaDeviceSynchronize();
		//cudaCheckError();

		//EXPECT_EQ(marker->id->getHostData()[0], lTestMarker[i].id);
		//EXPECT_EQ(marker->a->getHostData()[0], lTestMarker[i].a);
		//EXPECT_EQ(marker->b->getHostData()[0], lTestMarker[i].b);

	};

}