#include "MD.cuh"
#include <istream>
#include <fstream>
#include <ctime>
#include <ratio>
#include <chrono>

#include "loadAndStore2Buffer.cuh"
#include "distortImage.cuh"
#include "EdgeDetectionKernel.cuh"
#include "matchpattern.cuh"
#include "clusterDetection.cuh"
#include "radiusSearch.cuh"
#include "clusterMarkerCorners.cuh"
#include "imageRotateAndShift.cuh"
#include "markerRecognition.cuh"
#include "transformSolver.cuh"
#include "demosaicing.cuh"


namespace markerdetection
{

	__constant__ unsigned char cudaConstParams[16384];

	__global__ void global_step0()
	{		
		S_IMG_PROC_KERNEL_PARAMS *p_params = (S_IMG_PROC_KERNEL_PARAMS *)cudaConstParams;
		S_IMG_PROC_KERNEL_PARAMS& params = *p_params;

		__shared__ unsigned char minG[SHARED_BUFFER_SIZE];
		__shared__ unsigned char maxG[SHARED_BUFFER_SIZE];
		__shared__ unsigned char minR[SHARED_BUFFER_SIZE];
		__shared__ unsigned char maxR[SHARED_BUFFER_SIZE];
		__shared__ unsigned char minB[SHARED_BUFFER_SIZE];
		__shared__ unsigned char maxB[SHARED_BUFFER_SIZE];

		demosaicing::loadAndMinMax(minG, maxG,
			minR, maxR,
			minB, maxB,
			params);


	}


	
	__global__ void global_step1()
	{
		extern __shared__ unsigned char buffer_s1[];

		S_IMG_PROC_KERNEL_PARAMS *p_params = (S_IMG_PROC_KERNEL_PARAMS *)cudaConstParams;
		S_IMG_PROC_KERNEL_PARAMS& params = *p_params;

		
		unsigned char* p_imagebuffer	 =   &buffer_s1[0 * params.loadAndStoreParams.bufferHeight * params.loadAndStoreParams.bufferWidth];
		unsigned char* p_maskbuffer		 =	 &buffer_s1[1 * params.loadAndStoreParams.bufferHeight * params.loadAndStoreParams.bufferWidth];
		unsigned char* p_prop			 =	 &buffer_s1[2 * params.loadAndStoreParams.bufferHeight * params.loadAndStoreParams.bufferWidth];
		unsigned char* p_pattern		 =	 &buffer_s1[3 * params.loadAndStoreParams.bufferHeight * params.loadAndStoreParams.bufferWidth];

		int startIdx = threadIdx.x - params.loadAndStoreParams.maskOffset;
		int startIdy = threadIdx.y - params.loadAndStoreParams.maskOffset;
		int startId = startIdy*params.loadAndStoreParams.bufferWidth + startIdx;

		int realImageThreadx = threadIdx.x + blockDim.x*blockIdx.x;
		int realImageThready = threadIdx.y + blockDim.y*blockIdx.y;

		//if (realImageThreadx < 0 || realImageThreadx >= params.imageParameter.imageWidth ||
		//	realImageThready < 0 || realImageThready >= params.imageParameter.imageHeight)
		//{
		//	return;
		//}

		matchpattern::readBinMask(p_pattern, params);		
		loadandstore2bufer::initializebuffer(p_imagebuffer, p_maskbuffer, p_prop, params);
		loadandstore2bufer::store2buffer(p_imagebuffer, p_maskbuffer, params);

		__syncthreads();

		
		

		
		
		__syncthreads();
		bool ret;
		if (p_maskbuffer[startId])
		{
			ret = edgedetection::calculateEdge(p_imagebuffer, p_maskbuffer, params, startIdx, startIdy, startId);
			__syncthreads();
			if (ret)
			{
				matchpattern::calculatePatternScore(p_imagebuffer, p_maskbuffer, p_pattern, params, startIdx, startIdy, startId, realImageThreadx, realImageThready);
			}

		}
	


		__syncthreads();
		loadandstore2bufer::readFrombuffer(p_imagebuffer, startId,params);		
		loadandstore2bufer::readFrombuffermask(p_maskbuffer, startId, params);
		__syncthreads();

	}

	

	__global__ void global_step2()
	{
		extern __shared__ unsigned char buffer_s2[];

		S_IMG_PROC_KERNEL_PARAMS *p_params = (S_IMG_PROC_KERNEL_PARAMS *)cudaConstParams;
		S_IMG_PROC_KERNEL_PARAMS& params = *p_params;

		int indexA;
		int indexB;
		int indexC;
		int nMarkerCornerThreadSizeX;
		int nMarkerCornerThreadSizeY;
		int nMarkerCornerThreadSizeZ;
		int nCenterMerged;

		unsigned int nCentroids = (*params.centroidParameter.nCentroids);
		unsigned char* p = buffer_s2;

		float*		   p_xr = (float*)p;
		p += N_MAX_CENTROIDS_MERGED * sizeof(float);

		float*		   p_yr = (float*)(p);
		p += N_MAX_CENTROIDS_MERGED * sizeof(float);

		int*		   p_x = (int*)(p);
		p += N_MAX_CENTROIDS * sizeof(int);

		int*		   p_y = (int*)(p);
		p += N_MAX_CENTROIDS * sizeof(int);

		unsigned int*  p_n = (unsigned int*)(p);
		p += sizeof(unsigned int);

		unsigned int*  p_marker = (unsigned int*)(p);
		p += sizeof(unsigned int);

		unsigned char* p_prop = p;


		if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
		{
			*p_n = 0;
			*p_marker = 0;



		}



		__syncthreads();

		radiussearch::radiussearch(p_x, p_y, p_prop, nCentroids, p_xr, p_yr, p_n, params);

		__syncthreads();
		nCenterMerged = *p_n;

		//if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
		//{
		//	
		//	//for (int i = 0; i < nCenterMerged; i++)
		//	//{
		//	//	params.centroidParameter.centerAx[i] = p_xr[i];
		//	//	params.centroidParameter.centerAy[i] = p_yr[i];
		//	//}

		//	//*params.centroidParameter.nMarker = *p_n;

		//}
		//__syncthreads();

		

			


		nMarkerCornerThreadSizeX = divceil(nCenterMerged, blockDim.x);
		nMarkerCornerThreadSizeY = divceil(nCenterMerged, blockDim.y);
		nMarkerCornerThreadSizeZ = divceil(nCenterMerged, blockDim.z);
			
		for (int x = 0; x < nMarkerCornerThreadSizeX; x++)
		{
			for (int y = 0; y < nMarkerCornerThreadSizeY; y++)
			{
				for (int z = 0; z < nMarkerCornerThreadSizeZ; z++)
				{
					indexA = threadIdx.x*nMarkerCornerThreadSizeX + x;
					indexB = threadIdx.y*nMarkerCornerThreadSizeY + y;
					indexC = threadIdx.z*nMarkerCornerThreadSizeZ + z;

					markercorners::calcCorners(
						p_xr,
						p_yr,
						indexA,
						indexB,
						indexC,
						nCenterMerged,
						p_marker, 
						params);
						
					}

				}
			}

			__syncthreads();
			if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
			{
				*params.centroidParameter.nMarker = *p_marker;
			}

			

	}

	
	__global__ void global_step3()
	{
		S_IMG_PROC_KERNEL_PARAMS *p_params = (S_IMG_PROC_KERNEL_PARAMS *)cudaConstParams;
		S_IMG_PROC_KERNEL_PARAMS& params = *p_params;

		__shared__ float origin[2];
		__shared__ float nx[2];
		__shared__ float ny[2];
		__shared__ float nxp[2];
		__shared__ float nyp[2];
		__shared__ float dist_px_x;
		__shared__ float dist_px_y;
		__shared__ unsigned char img_buffer[N_PIXEL_MARKER_WDTH*N_PIXEL_MARKER_WDTH];
		__shared__ float edges[N_EDGES_PER_POINT*N_EDGES*N_DIM*N_POINTS];
		__shared__ float corners[N_DIM*N_POINTS];

		__shared__ float a[2];
		__shared__ float b[2];
		__shared__ float c[2];

		__shared__ unsigned int intBuffer[3];
		__shared__ unsigned char charBuffer[4];
		__shared__ int maskIdBuffer[1];

		__shared__ int markerId;
		__shared__ int offsetA;
		__shared__ int offsetB;

		
		rotateAndShift::binarizeMarker(origin, nx, ny, nxp, nyp, dist_px_x, dist_px_y,
			img_buffer, edges, corners, a, b, c, params);


		__syncthreads();


		markerrecognition::recognizeMarker(intBuffer, charBuffer, maskIdBuffer, img_buffer,
			&markerId, &offsetA, &offsetB, params);

		__syncthreads();

		if (params.debug && threadIdx.x == 0 && threadIdx.y == 0)
		{
			params.markerParameter.markerID[blockIdx.x] = markerId;
			params.markerParameter.markerOffsetA[blockIdx.x] = offsetA;
			params.markerParameter.markerOffsetB[blockIdx.x] = offsetB;
		}

	}

	__device__ int getGlobalId()
	{
		return threadIdx.z * blockDim.y * blockDim.x
			+ threadIdx.y * blockDim.x
			+ threadIdx.x;
	}

	__global__ void global_step4()
	{
		S_IMG_PROC_KERNEL_PARAMS *p_params = (S_IMG_PROC_KERNEL_PARAMS *)cudaConstParams;
		S_IMG_PROC_KERNEL_PARAMS& params = *p_params;

		__shared__ float xi[N_MAX_CENTROIDS_MERGED];
		__shared__ float yi[N_MAX_CENTROIDS_MERGED];
		
		__shared__ float xm[N_MAX_CENTROIDS_MERGED];
		__shared__ float ym[N_MAX_CENTROIDS_MERGED];
		
		__shared__ float offsetX[N_MAX_MARKERS];
		__shared__ float offsetY[N_MAX_MARKERS];
		__shared__ float offsetTheta[N_MAX_MARKERS];

		__shared__ double initialGuess[3];
		__shared__ unsigned int nvalidPoints;

		__shared__ double currentIntervall[3];
		__shared__ double currentMinimum[3];
		__shared__ double currentMinimumCost[1];
		__shared__ int   currentIteration[1];

		__shared__ double xc[N_SOLVER_DIMENSION];
		__shared__ double yc[N_SOLVER_DIMENSION];
		__shared__ double tc[N_SOLVER_DIMENSION];

		__shared__ double costs[N_SOLVER_DIMENSION_SIZE];
		__shared__ int costs_index[N_SOLVER_DIMENSION_SIZE];



		int threadId =  getGlobalId();
		
		if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
		{
			nvalidPoints = 0;
			currentIteration[0] = 0;
			currentMinimumCost[0] = DBL_MAX;

			currentIntervall[0] = params.solverParameter.xinit_intervall;
			currentIntervall[1] = params.solverParameter.yinit_intervall;
			currentIntervall[2] = params.solverParameter.tinit_intervall;
		}
		costs[threadId] = DBL_MAX;
		__syncthreads();

		transformsolver::initSolver(xm, ym, xi, yi, offsetTheta, offsetX, offsetY, &nvalidPoints, params);

		__syncthreads();
		double xtmp, ytmp, xout, yout;
		if (threadId == 0 )
		{








			xtmp = offsetX[0];
			ytmp = offsetY[0];

			xout = params.cameraParameter.Ri[0] * xtmp + params.cameraParameter.Ri[1] * ytmp + params.cameraParameter.Ri[2] * params.cameraParameter.dist;
			yout = params.cameraParameter.Ri[3] * xtmp + params.cameraParameter.Ri[4] * ytmp + params.cameraParameter.Ri[5] * params.cameraParameter.dist;

			currentMinimum[0] = params.cameraParameter.converionPx2Mm *xout;
			currentMinimum[1] = params.cameraParameter.converionPx2Mm *yout;


			xtmp = cos(offsetTheta[0]);
			ytmp = sin(offsetTheta[0]);

			xout = params.cameraParameter.Ri[0] * xtmp + params.cameraParameter.Ri[1] * ytmp + params.cameraParameter.Ri[2] * params.cameraParameter.dist;
			yout = params.cameraParameter.Ri[3] * xtmp + params.cameraParameter.Ri[4] * ytmp + params.cameraParameter.Ri[5] * params.cameraParameter.dist;

			currentMinimum[2] = atan2(yout , xout);

			//currentMinimum[0] = 0;
			//currentMinimum[1] = 0;
			//currentMinimum[2] = 0;

			if (params.debug)
			{
				params.mapParameter.initalGuessImage[0] = offsetX[0];
				params.mapParameter.initalGuessImage[1] = offsetY[0];
				params.mapParameter.initalGuessImage[2] = offsetTheta[0];

				params.mapParameter.initalGuess[0] = currentMinimum[0];
				params.mapParameter.initalGuess[1] = currentMinimum[1];
				params.mapParameter.initalGuess[2] = currentMinimum[2];				
			}

			//nvalidPoints = 3;

		}

		//__syncthreads();
		//
		//
		////setting running optimisation
		//while (currentIteration[0] < params.solverParameter.maxIter &&
		//	currentMinimumCost[0] > params.solverParameter.eps && 
		//	currentIntervall[0] > 1e-3 && 
		//	currentIntervall[1] > 1e-3 &&
		//	currentIntervall[2] > 1e-8)
		//{
		//	transformsolver::setCostPoints(xc, yc, tc, currentIntervall, currentMinimum);	
		//	__syncthreads();


		//	

		//	transformsolver::calculateCostsImage(costs, costs_index, nvalidPoints,  xm, xm, xi, yi, xc, yc, tc, params);

		//	__syncthreads();
		//	transformsolver::findMinimumCosts(costs, costs_index, xc, yc, tc, currentMinimum, currentMinimumCost);
		//	__syncthreads();
		//	if (threadId == 0)
		//	{
		//		currentIntervall[0] /= (0.2*(double)(N_SOLVER_DIMENSION - 1));
		//		currentIntervall[1] /= (0.2*(double)(N_SOLVER_DIMENSION - 1));
		//		currentIntervall[2] /= (0.2*(double)(N_SOLVER_DIMENSION - 1));
		//		currentIteration[0]++;
		//	}
		//	__syncthreads();
		//}

		//if (threadId == 0)
		//{

		//	params.solverParameter.optimalTranformation[0] = currentMinimum[0];
		//	params.solverParameter.optimalTranformation[1] = currentMinimum[1];
		//	params.solverParameter.optimalTranformation[2] = currentMinimum[2];
		//	params.solverParameter.minimumCosts[0] = currentMinimumCost[0];
		//}



	}






	using namespace cv;
	MD::MD(E_RESOLUTION res, E_CAMERA_TYPE type, const double const *C, const double const *H, const double const *kk,
		map<int, cv::Vec<float, 3>> m, 
		bool debug):m_debug(debug)
	{
		_map = m;
		
		memcpy(params.cameraParameter.C, C, N_ELMENTS_C_ROW*N_ELMENTS_C_COL*sizeof(double));




		Mat camera2origin(N_ELMENTS_H_ROW, N_ELMENTS_H_COL, CV_64FC1);
		memcpy(camera2origin.data, H, N_ELMENTS_H_COL* N_ELMENTS_H_ROW * sizeof(double));
		Mat origin2camera = camera2origin.inv();
		double height = origin2camera.at<double>(2, 3);
		height += robotParams.heightOverGround_mm;
		origin2camera.at<double>(2, 3) = height;

		camera2Robot = origin2camera;
		camera2origin = origin2camera.inv();
		camera2Robot = camera2origin;

		memcpy(params.cameraParameter.H, camera2origin.data, N_ELMENTS_H_ROW*N_ELMENTS_H_COL * sizeof(double));
		params.cameraParameter.dist = std::abs(H[11]);
		

		memcpy(params.cameraParameter.kk, kk, N_ELMENTS_kk_ROW*N_ELMENTS_kk_COL * sizeof(double));
		params.debug = false;
		if (type == E_CAMERA_TYPE::RAYTRIX)
		{
			params.maskParameter.debayer = true;
		}

		Mat rot(Size(3, 3), CV_64FC1);		
		for (int r = 0; r < rot.rows; r++)
		{
			for (int c = 0; c < rot.cols; c++)
			{
				rot.at<double>(r, c) = camera2origin.at<double>(r, c);
			}
		}
		Mat roti = rot.inv();
		memcpy(params.cameraParameter.Ri, roti.data, N_ELMENTS_R_ROW*N_ELMENTS_R_COL * sizeof(double));
		params.cameraParameter.t[0] = H[3];
		params.cameraParameter.t[1] = H[7];
		


		cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

		switch (res)
		{
		case E_RESOLUTION::E_1280X720:
			params.imageParameter.imageWidth = 1280;
			params.imageParameter.imageHeight = 720;			

			launchConfig_step1.block = dim3(32, 16);
			launchConfig_step1.grid = dim3(40, 34);

			launchConfig_step2.block = dim3(8, 8, 16);
			launchConfig_step2.grid = dim3(1);

		case E_RESOLUTION::E_960X540:
			params.imageParameter.imageWidth = 960;
			params.imageParameter.imageHeight = 540;

			launchConfig_step1.block = dim3(32, 32);
			launchConfig_step1.grid = dim3(30, 17);

			launchConfig_step2.block = dim3(8, 8, 16);
			launchConfig_step2.grid = dim3(1);
			break;
		};
		if (debug)
		{
			namedWindow("win1", WINDOW_AUTOSIZE);
			img = Mat(params.imageParameter.imageHeight, params.imageParameter.imageWidth, CV_8UC1);
			namedWindow("win2", WINDOW_AUTOSIZE);
			marker_img = Mat(N_PIXEL_MARKER_WDTH,N_MAX_MARKERS*N_PIXEL_MARKER_WDTH, CV_8UC1);

			params.debug = true;
		}


		initParameters();



		cudaMemcpyToSymbol(cudaConstParams, &params, sizeof(S_IMG_PROC_KERNEL_PARAMS));
		cudaCheckError();

	};

	void MD::initParameters()
	{

		//first init masks
		double u1, v1, u2, v2, x = 0, y = 0;
		projectPointsFromGround2ImagePlane(x, y, u1, v1);

		x = markerDim.indexAx_na* markerDim.edgeLength_mm / (double)markerDim.edgeLength_px;
		projectPointsFromGround2ImagePlane(x, y, u2, v2);

		double edge2Center = abs(u2 - u1);
		x = markerDim.widthBlackMarker* markerDim.edgeLength_mm / (double)markerDim.edgeLength_px;
		projectPointsFromGround2ImagePlane(x, y, u2, v2);
		double widthBlack = abs(u2 - u1);
		double radBlack = widthBlack / 2;

		x = markerDim.widthWhiteMarker* markerDim.edgeLength_mm / (double)markerDim.edgeLength_px;
		projectPointsFromGround2ImagePlane(x, y, u2, v2);
		double widthWhite = abs(u2 - u1);
		double radWhite = widthWhite / 2;

		int offset = 3;
		params.maskParameter.offsetEdge2Center = std::round(edge2Center)+1;
		params.maskParameter.dimEdge = 3;// std::round(radBlack);

		params.maskParameter.offsetEdge2CenterCorner = std::round(edge2Center / std::sqrt(2))+1;
		params.maskParameter.dimEdgeCorner = 3;// std::round(radBlack / std::sqrt(2));

		int w = std::ceil(widthWhite);
		w = w % 2 == 1 ? w : w + 1;

		params.maskParameter.maskWidth = w;
		params.maskParameter.maskOffset = -((w - 1) >> 1);

		mainPattern = new DeviceBuffer<unsigned char>(w*w);

		double radius;
		double diffx;
		double diffy;
		int nW = 0;		
		for (int y = 0; y < w; y++)
		{
			for (int x = 0; x < w; x++)
			{
				diffx = (((double)x) - w / 2.0);
				diffy = (((double)y) - w / 2.0);
				radius = sqrt(pow(diffx, 2.0) + pow(diffy, 2.0));
				if (radius <= radBlack)
				{
					mainPattern->getHostData()[y*w + x] = 0;
					//cout << 0;
				}
				else if (radius <= radWhite)
				{
					mainPattern->getHostData()[y*w + x] = 1;
					nW++;	
					//cout << 1;
				}
				else
				{
					mainPattern->getHostData()[y*w + x] = 2;				
					//cout << 2;
				}				
			}			
			//cout << endl;
		}

	
		mainPattern->set();
		params.maskParameter.nMaskWhite = nW;
		params.maskParameter.pattern = mainPattern->getDeviceData();

		params.maskParameter.minAbsoluteDiffEdge = 0;
		params.maskParameter.minRelativeDiffEdge = 0.0f * 255;

		params.maskParameter.threadFaktorX = std::ceil((double)w / (double)launchConfig_step1.block.x);
		params.maskParameter.threadFaktorY = std::ceil((double)w / (double)launchConfig_step1.block.y);

		params.maskParameter.threshPattern = std::round(0.5 * 255);

		// init load and store buffer
		int maxWidth = std::max(2*(params.maskParameter.offsetEdge2Center + params.maskParameter.dimEdge), -params.maskParameter.maskOffset);
		params.loadAndStoreParams.init(maxWidth, params.imageParameter.imageWidth, params.imageParameter.imageHeight, launchConfig_step1.block.x, launchConfig_step1.block.y);

		// calculating shared memory buffer size
		int size = params.loadAndStoreParams.bufferWidth *params.loadAndStoreParams.bufferHeight;
		int sizeMarker = mainPattern->getSize();
		launchConfig_step1.sharedMemorySize = 3 * size + sizeMarker;

		img_raw = new DeviceBuffer<unsigned char>(params.imageParameter.imageWidth*params.imageParameter.imageHeight);
		img_dev = new DeviceBuffer<unsigned char>(params.imageParameter.imageWidth*params.imageParameter.imageHeight);
		imgres_dev = new DeviceBuffer<unsigned char>(params.imageParameter.imageWidth*params.imageParameter.imageHeight);
		maskres_dev  = new DeviceBuffer<unsigned char>(params.imageParameter.imageWidth*params.imageParameter.imageHeight);

		params.imageParameter.img_raw = img_raw->getDeviceData();
		params.imageParameter.img = img_dev->getDeviceData();
		params.imageParameter.prop = imgres_dev->getDeviceData();
		params.imageParameter.mask = maskres_dev->getDeviceData();

		launchConfig_step0.block = dim3(32, 32);
		launchConfig_step0.grid = dim3(1);



		std::vector<std::vector<double>> holeContour;

		double dimx = robotParams.lengthHole_mm;
		double dimy = robotParams.widthHole_mm;

		std::vector<double> p;
		p.push_back(-dimx / 2);
		p.push_back(-dimy / 2);
		holeContour.push_back(p);


		p.clear();
		p.push_back(dimx / 2);
		p.push_back(-dimy / 2);
		holeContour.push_back(p);

		p.clear();
		p.push_back(dimx / 2);
		p.push_back(dimy / 2);
		holeContour.push_back(p);

		p.clear();
		p.push_back(-dimx / 2);
		p.push_back(dimy / 2);
		holeContour.push_back(p);

		p.clear();
		p.push_back(-dimx / 2);
		p.push_back(-dimy / 2);
		holeContour.push_back(p);

		imageMask = new DeviceBuffer<unsigned char>(params.imageParameter.imageWidth * params.imageParameter.imageHeight);
		index_x = new DeviceBuffer<int>(params.imageParameter.imageWidth * params.imageParameter.imageHeight);
		index_y = new DeviceBuffer<int>(params.imageParameter.imageWidth * params.imageParameter.imageHeight);

		/**********************************************************************************/
		distortimage::initMasksAndMap(params.cameraParameter.C, params.cameraParameter.H, params.cameraParameter.kk,
			params.imageParameter.imageWidth,
			params.imageParameter.imageHeight,
			holeContour,
			index_x, 
			index_y,
			imageMask);

		params.undistortParamer.mask = imageMask->getDeviceData();
		params.undistortParamer.mapx = index_x->getDeviceData();
		params.undistortParamer.mapy = index_y->getDeviceData();
		params.undistortParamer.img = img_dev->getDeviceData();
		params.undistortParamer.resimg = imgres_dev->getDeviceData();

		/***********************************************************************************/
		nCentroids = new DeviceBuffer<unsigned int>(1);
		centroidX = new DeviceBuffer<int>(N_MAX_CENTROIDS);
		centroidY = new DeviceBuffer<int>(N_MAX_CENTROIDS);
		centroidP = new DeviceBuffer<unsigned char>(N_MAX_CENTROIDS);

		params.centroidParameter.centerX = centroidX->getDeviceData();
		params.centroidParameter.centerY = centroidY->getDeviceData();
		params.centroidParameter.centerProp = centroidP->getDeviceData();
		params.centroidParameter.nCentroids = nCentroids->getDeviceData();
		params.centroidParameter.radiusThreshold = params.maskParameter.maskWidth*sqrt(2.0);



		Ax = new DeviceBuffer<float>(N_MAX_MARKERS);
		Ay = new DeviceBuffer<float>(N_MAX_MARKERS);

		Bx = new DeviceBuffer<float>(N_MAX_MARKERS);
		By = new DeviceBuffer<float>(N_MAX_MARKERS);

		Cx = new DeviceBuffer<float>(N_MAX_MARKERS);
		Cy = new DeviceBuffer<float>(N_MAX_MARKERS);

		Cn = new DeviceBuffer<unsigned int>(1);



		params.centroidParameter.centerAx = Ax->getDeviceData();
		params.centroidParameter.centerAy = Ay->getDeviceData();

		params.centroidParameter.centerBx = Bx->getDeviceData();
		params.centroidParameter.centerBy = By->getDeviceData();

		params.centroidParameter.centerCx = Cx->getDeviceData();
		params.centroidParameter.centerCy = Cy->getDeviceData();

		params.centroidParameter.nMarker = Cn->getDeviceData();



		
		x = 0;
		y = 0;
		projectPointsFromGround2ImagePlane(x, y, u1, v1);

		double deltaX = (double)markerDim.indexCx_na - (double)markerDim.indexAx_na;
		x = deltaX* markerDim.edgeLength_mm / (double)markerDim.edgeLength_px;
		projectPointsFromGround2ImagePlane(x, y, u2, v2);

		double distab = abs(u2 - u1);

		

		params.qrCodeParameter.squareDistanceABMin = pow(markerDim.scaleMinEdge * distab, 2);
		params.qrCodeParameter.squareDistanceABMax = pow(markerDim.scaleMaxEdge * distab, 2);

		params.qrCodeParameter.squareDistanceBCMin = pow(markerDim.scaleMinEdge * distab, 2) + pow(markerDim.scaleMinEdge * distab, 2);
		params.qrCodeParameter.squareDistanceBCMax = pow(markerDim.scaleMaxEdge * distab, 2) + pow(markerDim.scaleMaxEdge * distab, 2);

		launchConfig_step2.sharedMemorySize = radiussearch::calculateSharedMemorySize();

		//params.markerParameter.markerHeight = N_PIXEL_MARKER_WDTH;
		//params.markerParameter.markerWidth = N_PIXEL_MARKER_WDTH;

		launchConfig_step3.block = dim3(N_PIXEL_MARKER_WDTH, N_PIXEL_MARKER_WDTH);
		launchConfig_step3.sharedMemorySize = 0;

		x = 0;
		y = 0;
		projectPointsFromGround2ImagePlane(x, y, u1, v1);
		deltaX = (double)markerDim.indexAx_na;
		x = deltaX* markerDim.edgeLength_mm / (double)markerDim.edgeLength_px;
		projectPointsFromGround2ImagePlane(x, y, u2, v2);
		distab = abs(u2 - u1);

		params.markerParameter.offsetFeatureA2Origin_x = -distab;
		params.markerParameter.offsetFeatureA2Origin_y = -distab;


		x = 0;
		y = 0;
		projectPointsFromGround2ImagePlane(x, y, u1, v1);
		deltaX = 1;
		x = deltaX* markerDim.edgeLength_mm / (double)markerDim.edgeLength_px;
		projectPointsFromGround2ImagePlane(x, y, u2, v2);
		distab = abs(u2 - u1);

		params.markerParameter.makerIndex2PixelIndex = distab;
		params.markerParameter.binarisationThreshold = 127;

		binaryMarker = new DeviceBuffer<unsigned char>(N_MAX_MARKERS*N_PIXEL_MARKER_WDTH*N_PIXEL_MARKER_WDTH);
		params.qrCodeParameter.markerKorrigiert = binaryMarker->getDeviceData();
		params.qrCodeParameter.nBlocksBufferWidth = N_MAX_MARKERS;


		marker = new MARKER{ N_MAX_MARKERS };

		//setting parameters
		for (int i = 0; i < N_DEOCDER_MASKS; i++)
		{
			params.markerParameter.decodeMasks[i] =
				marker->decoderMasks->getDeviceData() + i * N_PIXEL_MARKER_WDTH*N_PIXEL_MARKER_WDTH;
		}



		params.markerParameter.indexNumeric0 = marker->indexNumeric0->getDeviceData();
		params.markerParameter.indexNumeric1 = marker->indexNumeric1->getDeviceData();
		params.markerParameter.indexNumeric2 = marker->indexNumeric2->getDeviceData();

		params.markerParameter.indexByte0 = marker->indexByte0->getDeviceData();
		params.markerParameter.indexByte1 = marker->indexByte1->getDeviceData();
		params.markerParameter.indexByte2 = marker->indexByte2->getDeviceData();
		params.markerParameter.indexByte3 = marker->indexByte3->getDeviceData();
		

		params.markerParameter.markerID = marker->id->getDeviceData();
		params.markerParameter.markerOffsetA = marker->a->getDeviceData();
		params.markerParameter.markerOffsetB = marker->b->getDeviceData();


		double u, v;
		for (int i = 0; i < MAX_SIZE_QR_MARKER; i++)
		{
			x = qrMarker.x->getHostData()[i];
			y = qrMarker.y->getHostData()[i];
			if (isnan(x) || isnan(y))
			{
				qrMarker.ix->getHostData()[i] = nan("");
				qrMarker.iy->getHostData()[i] = nan("");
			}
			else
			{
				projectPointsFromGround2ImagePlane(qrMarker.x->getHostData()[i], qrMarker.y->getHostData()[i], u, v);
				qrMarker.ix->getHostData()[i] = (float)u;
				qrMarker.iy->getHostData()[i] = (float)v;
			}

		}
		qrMarker.set();

		
		launchConfig_step4.grid = dim3(1);
		launchConfig_step4.block = dim3(N_SOLVER_DIMENSION, N_SOLVER_DIMENSION, N_SOLVER_DIMENSION);
		

		params.mapParameter.ix = qrMarker.ix->getDeviceData();
		params.mapParameter.iy = qrMarker.iy->getDeviceData();

		params.mapParameter.cx = qrMarker.x->getDeviceData();
		params.mapParameter.cy = qrMarker.y->getDeviceData();

		initalGuess = new DeviceBuffer<double>(3);
		initalGuessImage = new DeviceBuffer<double>(3);

		params.mapParameter.initalGuess = initalGuess->getDeviceData();
		params.mapParameter.initalGuessImage = initalGuessImage->getDeviceData();


		x = 0; y = 0;
		double u0, v0;
		projectPointsFromGround2ImagePlane(x, y, u0, v0);
		x = 1;
		projectPointsFromGround2ImagePlane(x, y, u1, v1);
		double diffu = u1 - u0;
		double diffv = v1 - v0;
		double diffImage = sqrt(diffu*diffu + diffv*diffv);
		params.cameraParameter.converionPx2Mm		= 1 / diffImage;
		params.cameraParameter.rotatationCart2Image = atan2(diffv, diffu);

		double ct = cos(params.cameraParameter.rotatationCart2Image);
		double st = sin(params.cameraParameter.rotatationCart2Image);

		params.cameraParameter.rotatationCart2ImageMatrix[0] = ct;
		params.cameraParameter.rotatationCart2ImageMatrix[1] = -st;
		params.cameraParameter.rotatationCart2ImageMatrix[2] = st;
		params.cameraParameter.rotatationCart2ImageMatrix[3] = ct;

		double xtmp = 1;
		double ytmp = 0;

		double xout = params.cameraParameter.Ri[0] * xtmp + params.cameraParameter.Ri[1] * ytmp + params.cameraParameter.Ri[2] * params.cameraParameter.dist;
		double yout = params.cameraParameter.Ri[3] * xtmp + params.cameraParameter.Ri[4] * ytmp + params.cameraParameter.Ri[5] * params.cameraParameter.dist;

		params.cameraParameter.rotatationCart2Image = atan2(yout, xout);


		optimalTransform = new DeviceBuffer<double>(3);
		optimalCosts = new  DeviceBuffer<double >(1);

		params.solverParameter.xinit = 0.0f;
		params.solverParameter.yinit = 0.0f;
		params.solverParameter.tinit = 0.0f;

		params.solverParameter.xinit_intervall = 180;
		params.solverParameter.yinit_intervall = 180;
		params.solverParameter.tinit_intervall = 3;

		params.solverParameter.eps = 1e-5;
		params.solverParameter.maxIter = 100000;

		params.solverParameter.optimalTranformation = optimalTransform->getDeviceData();
		params.solverParameter.minimumCosts = optimalCosts->getDeviceData();

		


	}

	void MD::step4_host()
	{
		Cn->get();
		marker->get();

		Ax->get(Cn->getHostData()[0]);
		Ay->get(Cn->getHostData()[0]);
		Bx->get(Cn->getHostData()[0]);
		By->get(Cn->getHostData()[0]);
		Cx->get(Cn->getHostData()[0]);
		Cy->get(Cn->getHostData()[0]);


		vector<Vec<double, 2>> pointsImage;
		vector<Vec<double, 3>> pointsCartesian;

		float a[2], b[2], c[2];
		int markerId, offsetA, offsetB;
		for (int i = 0; i < Cn->getHostData()[0]; i++)
		{			
			markerId = marker->id->getHostData()[i];
			offsetA = marker->a->getHostData()[i];
			offsetB = marker->b->getHostData()[i];

			if (markerId != INT_NAN &&
				offsetA != INT_NAN &&
				offsetB != INT_NAN)
			{

				qrMarker.getMarkerFromAB(offsetA, offsetB, a, b, c);

				
				pointsCartesian.push_back(Vec<double, 3>(a[0], a[1], 0));
				pointsCartesian.push_back(Vec<double, 3>(b[0], b[1], 0));
				pointsCartesian.push_back(Vec<double, 3>(c[0], c[1], 0));

				qrMarker.getMarkerFromABImage(offsetA, offsetB, a, b, c);
				pointsImage.push_back(Vec<double, 2>(Ax->getHostData()[i], Ay->getHostData()[i]));
				pointsImage.push_back(Vec<double, 2>(Bx->getHostData()[i], By->getHostData()[i]));
				pointsImage.push_back(Vec<double, 2>(Cx->getHostData()[i], Cy->getHostData()[i]));
			}

		}


		Mat camera2origin(N_ELMENTS_H_ROW, N_ELMENTS_H_COL, CV_64FC1);
		memcpy(camera2origin.data, params.cameraParameter.H, N_ELMENTS_H_ROW*N_ELMENTS_H_COL * sizeof(double));

		Mat rot(Size(3, 3), CV_64FC1);
		Mat trans(Size(1, 3), CV_64FC1);
		for (int r = 0; r < rot.rows; r++)
		{
			for (int c = 0; c < rot.cols; c++)
			{
				rot.at<double>(r, c) = camera2origin.at<double>(r, c);
			}
		}
		trans.at<double>(0) = camera2origin.at<double>(0, 3);
		trans.at<double>(1) = camera2origin.at<double>(1, 3);
		trans.at<double>(2) = camera2origin.at<double>(2, 3);
		Mat rot_vec;
		Rodrigues(rot, rot_vec);



		Mat Cmat(N_ELMENTS_C_COL, N_ELMENTS_C_ROW, CV_64FC1);
		memcpy(Cmat.data, params.cameraParameter.C, N_ELMENTS_C_COL* N_ELMENTS_C_ROW * sizeof(double));

		bool hasSolved = solvePnP(pointsCartesian, pointsImage, Cmat, Mat(), rot_vec, trans, false);

		if (hasSolved)
		{
			rotResult = rot_vec.clone();
			transResult = trans.clone();



			Rodrigues(rot_vec, rot);
			Mat H = Mat::eye(Size(4, 4), CV_64FC1);
			
			for (int r = 0; r < rot.rows; r++)
			{
				for (int c = 0; c < rot.cols; c++)
				{
					H.at<double>(r, c) = rot.at<double>(r, c);
				}
			}
			H.at<double>(0, 3) = trans.at<double>(0);
			H.at<double>(1, 3) = trans.at<double>(1);
			H.at<double>(2, 3) = trans.at<double>(2);



			Mat marker2Camera = H;
			Mat marker2Robot = marker2Camera*camera2Robot;

			for (int r = 0; r < rot.rows; r++)
			{
				for (int c = 0; c < rot.cols; c++)
				{
					rot.at<double>(r, c) = marker2Robot.at<double>(r, c);
				}
			}
			trans.at<double>(0) = marker2Robot.at<double>(0, 3);
			trans.at<double>(1) = marker2Robot.at<double>(1, 3);
			trans.at<double>(2) = marker2Robot.at<double>(2, 3);
			Mat rot_vec;
			Rodrigues(rot, rot_vec);
			

			double mx = _map[markerId][0];
			double my = _map[markerId][1];
			double mt = _map[markerId][2];

			double rotMatrixCartesian[4];
			double ct = cos(mt);
			double st = sin(mt);

			rotMatrixCartesian[0] = ct;
			rotMatrixCartesian[1] = -st;
			rotMatrixCartesian[2] = st;
			rotMatrixCartesian[3] = ct;

			double x = trans.at<double>(0);
			double y = trans.at<double>(1);

			double xr = x  * rotMatrixCartesian[0] + y  * rotMatrixCartesian[1] + mx;
			double yr = x  * rotMatrixCartesian[2] + y  * rotMatrixCartesian[3] + my;

			robotPose[0] = xr / 1000;
			robotPose[1] = yr / 1000;
			robotPose[2] = mt + rot_vec.at<double>(2);


		}

	}

	void MD::projectPointsFromGround2ImagePlane(double x, double y, double& u, double& v)
	{


		//__device__ void projectMapPointIntoImage(double xmap, double ymap, double xr, double yr, double tr, const S_IMG_PROC_KERNEL_PARAMS& params, double &xi, double& yi)
		
			//double rot[4];

			//double ct = cos(tr);
			//double st = sin(tr);

			//rot[0] = ct;
			//rot[1] = -st;
			//rot[2] = st;
			//rot[3] = ct;


			//double xcar = x;
			//double ycar = y;
			//double zcar = 0;



			//double xc = params.cameraParameter.H[3] + params.cameraParameter.H[0] * xcar + params.cameraParameter.H[1] * ycar + params.cameraParameter.H[2] * zcar;
			//double yc = params.cameraParameter.H[7] + params.cameraParameter.H[4] * xcar + params.cameraParameter.H[5] * ycar + params.cameraParameter.H[6] * zcar;
			//double zc = params.cameraParameter.H[11] + params.cameraParameter.H[8] * xcar + params.cameraParameter.H[9] * ycar + params.cameraParameter.H[10] * zcar;


			//double xi = params.cameraParameter.C[0] * xc + params.cameraParameter.C[2] * zc;
			//double yi = params.cameraParameter.C[4] * yc + params.cameraParameter.C[5] * zc;


			//double imx = xi / zc;
			//double imy = yi / zc;

			//u = imx;
			//v = imy;

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


		


		Mat camera2origin(N_ELMENTS_H_ROW, N_ELMENTS_H_COL, CV_64FC1);
		memcpy(camera2origin.data, params.cameraParameter.H, N_ELMENTS_H_ROW*N_ELMENTS_H_COL * sizeof(double));

		Mat rot(Size(3, 3), CV_64FC1);
		Mat trans(Size(1, 3), CV_64FC1);
		for (int r = 0; r < rot.rows; r++)
		{
			for (int c = 0; c < rot.cols; c++)
			{
				rot.at<double>(r, c) = camera2origin.at<double>(r, c);
			}
		}
		trans.at<double>(0) = camera2origin.at<double>(0, 3);
		trans.at<double>(1) = camera2origin.at<double>(1, 3);
		trans.at<double>(2) = camera2origin.at<double>(2, 3);
		Mat rot_vec;
		Rodrigues(rot, rot_vec);

		vector<Vec<double, 3>> points;
		Vec<double, 3> vec;
		vec.val[0] = x;
		vec.val[1] = y;
		vec.val[2] = 0;
		points.push_back(vec);

		Mat Cmat(N_ELMENTS_C_COL, N_ELMENTS_C_ROW, CV_64FC1);
		memcpy(Cmat.data, params.cameraParameter.C, N_ELMENTS_C_COL* N_ELMENTS_C_ROW * sizeof(double));

		Mat res;
		projectPoints(points, rot_vec, trans, Cmat, Mat(), res);

		Vec<double, 2> t1 = res.at<Vec<double, 2>>(0);
		u = t1.val[0];
		v = t1.val[1];
	}

	void MD::step0()
	{
		global_step0 << <launchConfig_step0.grid, launchConfig_step0.block >> > ();
		cudaCheckError();

		cudaDeviceSynchronize();
		cudaCheckError();
	}

	void MD::step1()
	{
		nCentroids->reset();

		global_step1 << <launchConfig_step1.grid, launchConfig_step1.block, launchConfig_step1.sharedMemorySize >> > ();
		cudaCheckError();

		cudaDeviceSynchronize();
		cudaCheckError();
	}

	void MD::step2()
	{
		nCentroids->get();
		
		Cn->reset();

		
		global_step2 << <launchConfig_step2.grid, launchConfig_step2.block, launchConfig_step2.sharedMemorySize >> > ();
		cudaCheckError();

		cudaDeviceSynchronize();
		cudaCheckError();

		

	}

	void MD::step3()
	{
		Cn->get();
		
		if (Cn->getHostData()[0] > 0)
		{
			launchConfig_step3.grid = dim3(Cn->getHostData()[0]);
			global_step3 << <launchConfig_step3.grid, launchConfig_step3.block >> > ();
			cudaCheckError();

			cudaDeviceSynchronize();
			cudaCheckError();
		}

	}

	void MD::step4()
	{
		if (Cn->getHostData()[0] > 0)
		{
			global_step4 << <launchConfig_step4.grid, launchConfig_step4.block >> > ();
			cudaCheckError();

			cudaDeviceSynchronize();
			cudaCheckError();
		}


		//initalGuess->get();



	}

	

	MD::~MD()
	{
		delete mainPattern;
		delete img_raw;
		delete img_dev;
		delete imgres_dev;
		delete imageMask;
		delete index_x;
		delete index_y;
		delete nCentroids;
		delete centroidX;
		delete centroidY;
		delete centroidP;
		delete Ax;
		delete Ay;
		delete Bx;
		delete By;
		delete Cx;
		delete Cy;
		delete Cn;
		delete binaryMarker;
		delete marker;
		delete initalGuess;
		delete initalGuessImage;
		delete optimalTransform;
		delete optimalCosts;
	}

	

	void MD::processData(const unsigned char* const p_data)
	{

		img_raw->set((unsigned char*)p_data);
		using namespace std::chrono;

		
		//high_resolution_clock::time_point t1 = high_resolution_clock::now();


		//for (int i = 0; i < 100; ++i)
		//{
			step0();
			step1();
			step2();
			step3();
			//step4();
			//step4_host();
			
		//}


		//high_resolution_clock::time_point t2 = high_resolution_clock::now();

		//duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

		//std::cout << "It took me " << 100 / time_span.count() << " hz.";
		//std::cout << std::endl;



		if (m_debug)
		{
			imgres_dev->get();
			maskres_dev->get();

			nCentroids->get();
			centroidX->get(nCentroids->getHostData()[0]);
			centroidY->get(nCentroids->getHostData()[0]);

			Cn->get();
			Ax->get(Cn->getHostData()[0]);
			Ay->get(Cn->getHostData()[0]);
			Bx->get(Cn->getHostData()[0]);
			By->get(Cn->getHostData()[0]);
			Cx->get(Cn->getHostData()[0]);
			Cy->get(Cn->getHostData()[0]);

			initalGuess->get();
			initalGuessImage->get();
			optimalTransform->get();

			binaryMarker->get();
			marker->get();


			memcpy(img.data, imgres_dev->getHostData(), params.imageParameter.imageHeight*params.imageParameter.imageWidth);
			memcpy(marker_img.data, binaryMarker->getHostData(), binaryMarker->getSize());
			resize(marker_img, markerresize, Size(), 10, 10);
			
			
			cvtColor(img, img_colored, cv::COLOR_GRAY2BGR);

			overlay(maskres_dev->getHostData(), 0, 255, 0, 0.9, 2);	
			//drawCenter(centroidX->getHostData(), centroidY->getHostData(), nCentroids->getHostData()[0]);
			drawCenter(Ax->getHostData(), Ay->getHostData(), Cn->getHostData()[0]);


			//drawCoordinateCross(Ax->getHostData(), Ay->getHostData(),
			//					Bx->getHostData(), By->getHostData(), 
			//					Cx->getHostData(), Cy->getHostData(), Cn->getHostData()[0]);
			//
			//drawQRText();
			//reprojectMarker();

			imwrite("img_colored.png", img_colored);
			imwrite("marker.png", marker_img);
			imshow("win1", img_colored);
			imshow("win2", markerresize);
		}

	}

	void MD::overlay(const unsigned char* mask, int r, int g, int b, float alphain, unsigned char maskValue)
	{
		Vec3b red(r,g,b);		
		float alpha = alphain*255;
		for (int y = 0; y<img_colored.rows; y++)
			for (int x = 0; x<img.cols; x++)
			{
				if (mask[y*img_colored.cols + x] == maskValue)
				{
					img_colored.at<Vec3b>(y, x)[0] = (int)std::min((1 - alpha / 255.0) * img_colored.at<Vec3b>(y, x)[0] + (alpha * red[0] / 255), 255.0);
					img_colored.at<Vec3b>(y, x)[1] = (int)std::min((1 - alpha / 255.0) * img_colored.at<Vec3b>(y, x)[1] + (alpha * red[1] / 255), 255.0);
					img_colored.at<Vec3b>(y, x)[2] = (int)std::min((1 - alpha / 255.0) * img_colored.at<Vec3b>(y, x)[2] + (alpha * red[2] / 255), 255.0);
				}
			}		
	}

	void MD::drawQRText()
	{
		// A-------C
		// | _   /
		// | _  /
		// | _ /
		// B  /

		
		for (int i = 0; i < Cn->getHostData()[0]; i++)
		{

			Vec2f a(Ax->getHostData()[i], Ay->getHostData()[i]);
			Vec2f b(Bx->getHostData()[i], By->getHostData()[i]);
			Vec2f c(Cx->getHostData()[i], Cy->getHostData()[i]);

			Vec2f d =a - c;
			double dist = norm(d);
			Vec2f r = (d/dist);
			Vec2f v = (-45*r) + a;
			
			int ac = marker->id->getHostData()[i];
			int bc = marker->a->getHostData()[i];
			int cc = marker->b->getHostData()[i];

			if (ac == INT_NAN ||
				bc == INT_NAN ||
				cc == INT_NAN)
			{
				continue;
			}
			stringstream ss;
			ss << marker->id->getHostData()[i];
			ss << marker->a->getHostData()[i];
			ss << marker->b->getHostData()[i];
			string str = ss.str();
			putText(img_colored, str, Point(v), 0, 0.5, Vec3b(0, 255, 0));

		}
	}

	void MD::reprojectMarker()
	{

		

		double rotMatrixImage[4];
		double rotMatrixCartesian[4];
		double rotMatrixCartesianSolution[4];

		double ct = cos(initalGuessImage->getHostData()[2]);
		double st = sin(initalGuessImage->getHostData()[2]);

		rotMatrixImage[0] = ct;
		rotMatrixImage[1] = -st;
		rotMatrixImage[2] = st;
		rotMatrixImage[3] = ct;

		ct = cos(initalGuess->getHostData()[2]);
		st = sin(initalGuess->getHostData()[2]);

		rotMatrixCartesian[0] = ct;
		rotMatrixCartesian[1] = -st;
		rotMatrixCartesian[2] = st;
		rotMatrixCartesian[3] = ct;


		
		
		ct = cos(optimalTransform->getHostData()[2]);
		st = sin(optimalTransform->getHostData()[2]);

		rotMatrixCartesianSolution[0] = ct;
		rotMatrixCartesianSolution[1] = -st;
		rotMatrixCartesianSolution[2] = st;
		rotMatrixCartesianSolution[3] = ct;

		double x, y, xr, yr, resx, resy, resx0, resy0;
		double u0, v0, u1, v1;
		Vec3b red(0, 0, 255);
		double t_dbg[] = { -10, +20 };


		Mat Cmat(N_ELMENTS_C_COL, N_ELMENTS_C_ROW, CV_64FC1);
		memcpy(Cmat.data, params.cameraParameter.C, N_ELMENTS_C_COL* N_ELMENTS_C_ROW * sizeof(double));

		Mat res;
		

		for (int i = 0; i < qrMarker.size; i++)
		{
			if (!isnan(qrMarker.ix->getHostData()[i]) && !isnan(qrMarker.iy->getHostData()[i]))
			{
				x = qrMarker.x->getHostData()[i];
				y = qrMarker.y->getHostData()[i];

				//xr = x  * rotMatrixCartesian[0] + y  * rotMatrixCartesian[1] + initalGuess->getHostData()[0];
				//yr = x  * rotMatrixCartesian[2] + y  * rotMatrixCartesian[3] + initalGuess->getHostData()[1];

				//projectPointsFromGround2ImagePlane(xr, yr, u1, v1);
				//drawMarker(img_colored, Point(u1, v1), Vec3b(0,255, 255), 0, 40);

				//x = qrMarker.ix->getHostData()[i];
				//y = qrMarker.iy->getHostData()[i];

				//xr = x  * rotMatrixImage[0] + y  * rotMatrixImage[1] + initalGuessImage->getHostData()[0];
				//yr = x  * rotMatrixImage[2] + y  * rotMatrixImage[3] + initalGuessImage->getHostData()[1];
				//drawMarker(img_colored, Point(xr, yr), red, 0, 40);


				//x = qrMarker.x->getHostData()[i];
				//y = qrMarker.y->getHostData()[i];

				//xr = x  * rotMatrixCartesianSolution[0] + y  * rotMatrixCartesianSolution[1] + optimalTransform->getHostData()[0];
				//yr = x  * rotMatrixCartesianSolution[2] + y  * rotMatrixCartesianSolution[3] + optimalTransform->getHostData()[1];

				//projectPointsFromGround2ImagePlane(xr, yr, u1, v1);
				//drawMarker(img_colored, Point(u1, v1), Vec3b( 255, 0,255), 0, 40);


				x = qrMarker.x->getHostData()[i];
				y = qrMarker.y->getHostData()[i];

				//xr = x  * rotMatrixCartesianSolution[0] + y  * rotMatrixCartesianSolution[1] + optimalTransform->getHostData()[0];
				//yr = x  * rotMatrixCartesianSolution[2] + y  * rotMatrixCartesianSolution[3] + optimalTransform->getHostData()[1];

				//projectPointsFromGround2ImagePlane(xr, yr, u1, v1);
				

				vector<Vec<double, 3>> points;
				Vec<double, 3> vec;
				vec.val[0] = x;
				vec.val[1] = y;
				vec.val[2] = 0;
				points.push_back(vec);


				projectPoints(points, rotResult, transResult, Cmat, Mat(), res);
				

				Vec<double, 2> t1 = res.at<Vec<double, 2>>(0);
				u1 = t1.val[0];
				v1 = t1.val[1]; 

				drawMarker(img_colored, Point(u1, v1), Vec3b(255, 0, 255), 0, 40);

			}

			vector<Vec<double, 3>> points;
			Vec<double, 3> vec;
			vec.val[0] = 0;
			vec.val[1] = 0;
			vec.val[2] = 0;
			points.push_back(vec);


			projectPoints(points, rotResult, transResult, Cmat, Mat(), res);


			Vec<double, 2> t1 = res.at<Vec<double, 2>>(0);
			u1 = t1.val[0];
			v1 = t1.val[1];

			drawMarker(img_colored, Point(u1, v1), Vec3b(255, 0, 255), 0, 40);
		}





	}

	void MD::drawCenter(int* x, int* y, int n)
	{
		Vec3b red(0, 0, 255);

		for (int i = 0; i < n; i++)
		{			
			drawMarker(img_colored, Point(x[i], y[i]), red);
		}
	}

	void MD::drawCenter(float* x, float* y, int n)
	{
		Vec3b red(0, 0, 255);

		for (int i = 0; i < n; i++)
		{
			drawMarker(img_colored, Point(x[i], y[i]), red);
		}
	}

	

	void MD::drawCoordinateCross(float* Ax, float* Ay, float* Bx, float* By, float* Cx, float* Cy, int n)
	{

		// A-------C
		// | _   /
		// | _  /
		// | _ /
		// B  /
		for (int i = 0; i < n; i++)
		{
			line(img_colored, Point(Ax[i], Ay[i]), Point(Bx[i], By[i]), Vec3b(0, 0, 255));
			line(img_colored, Point(Ax[i], Ay[i]), Point(Cx[i], Cy[i]), Vec3b(0, 255, 0));			
		}
	}











	struct MarkerDetectionTest : testing::Test
	{

		void runTest()
		{

		}


	};

	TEST_F(MarkerDetectionTest, transformSolver)
	{
		using namespace cv;
		Mat img;		
		img = imread("distorted.png", CV_LOAD_IMAGE_GRAYSCALE);

		if (!img.data)
		{
			throw exception("Could not open or find the image");
		}

		int n = 960 * 540;
		unsigned char* buffer = new unsigned char[n];
		
		ifstream ifs("test.raw", ios::in | ios::binary);
		ifs.read((char*)buffer, n);

		if (ifs)
			std::cout << "all characters read successfully.";
		else
			std::cout << "error: only " << ifs.gcount() << " could be read";		
		ifs.close();

		//Mat image(Size(960, 540), CV_8U);
		//memcpy(image.data, buffer, n);
		//namedWindow("win1", WINDOW_AUTOSIZE);
		//imshow("win1", image);
		//waitKey(0);


		double C[N_ELMENTS_C_ROW*N_ELMENTS_C_COL];
		double H[N_ELMENTS_H_ROW*N_ELMENTS_H_COL];
		double kk[N_ELMENTS_kk_ROW*N_ELMENTS_kk_COL];

		Mat Cmat, kkmat, Hmat;
		try {
			FileStorage fs("raytrix.yml", FileStorage::READ);
			fs["camera_matrix"] >> Cmat;
			fs["distortion_coefficients"] >> kkmat;
			fs["camera_extrinsics"] >> Hmat;
			fs.release();
		}
		catch (...) {
			throw runtime_error("Error reading Camera matrix");
		}

		map<int, cv::Vec<float, 3>> labels;
		// Struktur: {id: [x, y, t], id: [x, y, t]}
		try {
			FileStorage fs("bodenmarker.yml", FileStorage::READ);
			FileNode root = fs["labels"];
			//cout << "Labels size: " << root.size() << root.type() << endl;
			for (FileNodeIterator fit = root.begin(); fit != root.end(); ++fit) {
				FileNode item = *fit;
				Vec3f data(item[1][0], item[1][1], item[1][2]);
				int id = item[0];
				labels[id] = data;
			}
		}
		catch (...) {
			throw runtime_error("Error reading labels");
		}




		memcpy(C, Cmat.data, N_ELMENTS_C_ROW*N_ELMENTS_C_COL * sizeof(double));
		memcpy(H, Hmat.data, N_ELMENTS_H_ROW*N_ELMENTS_H_COL * sizeof(double));
		memcpy(kk, kkmat.data, N_ELMENTS_kk_ROW*N_ELMENTS_kk_COL * sizeof(double));

		MD md(E_RESOLUTION::E_1280X720, E_CAMERA_TYPE::RAYTRIX, C, H, kk, labels, true);
		md.processData(buffer);

		delete buffer;
		
		waitKey(0);

	}

	


}