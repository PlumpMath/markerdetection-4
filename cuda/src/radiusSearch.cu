#include "radiusSearch.cuh"

namespace radiussearch
{

	__device__ int roundUpIntegerDivision(int x, int y)
	{
		return (x % y) ? x / y + 1 : x / y;
	}

	__device__ int roundClosesIntegerDivision(int x, int y)
	{
		return (x + (y / 2)) / y;
	}

	__device__ void radiussearch(int* centerX, int* centerY, unsigned char* prop, unsigned int nCenter, float* centerMergedx, float* centerMergedy, unsigned int *nMerged,  const S_IMG_PROC_KERNEL_PARAMS& params)
	{
		int blocksize = blockDim.x*blockDim.y*blockDim.x;
		int threadId = blockIdx.x * blockDim.x * blockDim.y * blockDim.z
			+ threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

		int id;		
		int cyclesPerThread = divceil(nCenter, blocksize);

		//copy into local memory
		for (int i = 0; i < cyclesPerThread; i++)
		{
			id = cyclesPerThread*threadId + i;
			if (id < nCenter)
			{
				centerX[id] = params.centroidParameter.centerX[id];
				centerY[id] = params.centroidParameter.centerY[id];
				prop[id] = params.centroidParameter.centerProp[id];
			}
		}
		__syncthreads();


		 int sumWeightedCorrdX = 0;
		 int sumWeightedCorrdY = 0;
		 int sumWeighted = 0;

		 int centerx;
		 int centery;

		int diffx;
		int diffy;
		float dist;
		int maxProp = 0;
		int res_id = 0;
		for (int i = 0; i < cyclesPerThread; i++)
		{
			id = cyclesPerThread*threadId + i;
			if (id < nCenter)
			{
				centerx = centerX[id];
				centery = centerY[id];
				maxProp = 0;
				for (int j = 0; j < nCenter; j++)
				{

					diffx = centerX[j] - centerx;
					diffy = centerY[j] - centery;
					dist = diffx *diffx + diffy * diffy;

					if (dist < params.centroidParameter.radiusThreshold)
					{
						if (prop[j] > maxProp)
						{
							maxProp = prop[j];
							res_id = j;
						}

						//sumWeightedCorrdX += prop[j] * centerX[j];
						//sumWeightedCorrdY += prop[j] * centerY[j];
						//sumWeighted += prop[j];
					}

				}

				//int indexX = roundClosesIntegerDivision(sumWeightedCorrdX, sumWeighted);
				//int indexY = roundClosesIntegerDivision(sumWeightedCorrdY, sumWeighted);
				//if (indexX == centerx && indexY == centery)

				if (id== res_id)
				{
					unsigned int index = atomicInc(nMerged, N_MAX_CENTROIDS);
					centerMergedx[index] = centerX[res_id];
					centerMergedy[index] = centerY[res_id];
					//centerMergedx[index] = ((float)sumWeightedCorrdX) / ((float)sumWeighted);
					//centerMergedy[index] = ((float)sumWeightedCorrdY) / ((float)sumWeighted);

				}
			}
		}



	}

	int calculateSharedMemorySize()
	{

		
		int size = 	2 * N_MAX_CENTROIDS_MERGED * sizeof(float) +
					2 * N_MAX_CENTROIDS * sizeof(int) +
					1 * N_MAX_CENTROIDS * sizeof(unsigned char) +
					2 * sizeof(unsigned int);
		return size;
	}

	__global__ void testRadiusSearch(S_IMG_PROC_KERNEL_PARAMS		params)
	{
		extern __shared__ unsigned char buffer[];

		unsigned int nCentroids = (*params.centroidParameter.nCentroids);
		unsigned char* p = buffer;

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
		
		
		
		radiussearch(p_x, p_y,p_prop, nCentroids,
			params.centroidParameter.centerAx, params.centroidParameter.centerAy, 
			params.centroidParameter.nMarker, params);
	}


	struct RadiusSearchTest : testing::Test
	{

		DeviceBuffer<int> *X;
		DeviceBuffer<int> *Y;
		DeviceBuffer<unsigned char> *P;
		DeviceBuffer<unsigned int> *n;

		DeviceBuffer<float> *rX;
		DeviceBuffer<float> *rY;
		DeviceBuffer<unsigned int> *rn;


		S_IMG_PROC_KERNEL_PARAMS		params;
		S_LAUNCH_CONFIG					launchConfig;

		

		RadiusSearchTest()
		{
			X = new DeviceBuffer<int>(10);
			Y = new DeviceBuffer<int>(10);
			P = new DeviceBuffer<unsigned char>(10);
			n = new DeviceBuffer<unsigned int>(1);


			rX = new DeviceBuffer<float>(10);
			rY = new DeviceBuffer<float>(10);
			rn = new DeviceBuffer<unsigned int>(1);

			params.centroidParameter.radiusThreshold = 2;
			params.centroidParameter.centerX = X->getDeviceData();
			params.centroidParameter.centerY = Y->getDeviceData();
			params.centroidParameter.centerProp = P->getDeviceData();

			params.centroidParameter.nCentroids = n->getDeviceData();

			params.centroidParameter.centerAx = rX->getDeviceData();
			params.centroidParameter.centerAy = rY->getDeviceData();
			params.centroidParameter.nMarker = rn->getDeviceData();

			launchConfig.grid = dim3(1);
			launchConfig.block = dim3(10, 10, 10);
			launchConfig.sharedMemorySize = 2 * 10 * sizeof(float) + 2 * 10 * sizeof(int) +  1*10*sizeof(unsigned char) + 1 * sizeof(unsigned int);



		}

		void runTest()
		{
			X->set();
			Y->set();
			P->set();
			n->set();
		
			testRadiusSearch << <launchConfig.grid, launchConfig.block, launchConfig.sharedMemorySize >> > (params);
			cudaCheckError();

			cudaDeviceSynchronize();
			cudaCheckError();

			rX->get();
			rY->get();
			rn->get();

		}

		~RadiusSearchTest()
		{
			delete X;
			delete Y;
			delete P;
			delete n;
			delete rX;
			delete rY;
			delete rn;
		}

	};

	//TEST_F(RadiusSearchTest, radiusSearchTest1)
	//{
	//	n->getHostData()[0] = 2;		
	//	P->getHostData()[0] = 255;	
	//	P->getHostData()[1] = 255;
	//	X->getHostData()[1] = 1;		
	//	Y->getHostData()[1] = 1;		
	//	params.centroidParameter.radiusThreshold = 4;

	//	runTest();

	//	EXPECT_NEAR(rX->getHostData()[0], 0.5, 1e-3);
	//	EXPECT_NEAR(rY->getHostData()[0], 0.5, 1e-3);
	//	EXPECT_EQ(rn->getHostData()[0], 1);

	//};


	//TEST_F(RadiusSearchTest, radiusSearchTest2)
	//{
	//	n->getHostData()[0] = 2;
	//	P->getHostData()[0] = 255;
	//	P->getHostData()[1] = 255;
	//	X->getHostData()[1] = 10;
	//	Y->getHostData()[1] = 10;
	//	params.centroidParameter.radiusThreshold = 4;

	//	runTest();
	//	EXPECT_EQ(rn->getHostData()[0], 2);
	//	int n = rn->getHostData()[0];
	//	bool doesCenterExist = false;
	//	float cx = 10;
	//	float cy = 10;
	//	for (int i = 0; i < n; i++)
	//	{
	//		if (abs(rX->getHostData()[i] - cx) < 1e-3 && abs(rY->getHostData()[i] - cy) < 1e-3)
	//		{
	//			doesCenterExist = true;
	//			break;
	//		}
	//	}
	//	EXPECT_TRUE(doesCenterExist);

	//};



}