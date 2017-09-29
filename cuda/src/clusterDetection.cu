
#include "clusterDetection.cuh"

namespace clusterdetection
{



	__device__ void calculateCluster(		
		unsigned char* mask_buffer,
		const unsigned char* pattern_buffer,
		const unsigned char* prop_buffer,
		const S_IMG_PROC_KERNEL_PARAMS& params,
		int startIdx, int startIdy, int startId,
		int realImageThreadx,int realImageThready)
	{

		if (mask_buffer[startId] < 3)
		{			
			return;
		}


		int bufferId;
		int patternId;
		int imageid;
		int realBufferThreadx;
		int realBufferThready;

		float realX;
		float realY;



		int realImageThreadxLocal;
		int realImageThreadyLocal;
		int realImageThreadIdLocal;

		double sumWeightedCorrdX = 0;
		double sumWeightedCorrdY = 0;
		double sumWeighted = 0;
		


		bool pointsOutSide = false;



		for (int r = params.maskParameter.maskOffset; r <= -params.maskParameter.maskOffset; r++)
		{
			realBufferThready = startIdy + r;
			realImageThreadyLocal = threadIdx.y + r + blockDim.y*blockIdx.y;

			for (int c = params.maskParameter.maskOffset; c <= -params.maskParameter.maskOffset; c++)
			{
				realBufferThreadx = startIdx + c;
				realImageThreadxLocal = threadIdx.x + c + blockDim.x*blockIdx.x;
				
				
				bufferId = realBufferThready*params.loadAndStoreParams.bufferWidth + realBufferThreadx;

				patternId = (r - params.maskParameter.maskOffset) * params.maskParameter.maskWidth + (c - params.maskParameter.maskOffset);

				imageid = realImageThreadyLocal * params.imageParameter.imageWidth + realImageThreadxLocal;
				
				if (prop_buffer[bufferId] > 0  && pattern_buffer[patternId] == 0)
				{
					//weighted points inside
					sumWeightedCorrdX += (double)prop_buffer[bufferId] * realImageThreadxLocal;
					sumWeightedCorrdY += (double)prop_buffer[bufferId] * realImageThreadyLocal;
					sumWeighted += (double)prop_buffer[bufferId];

					sumWeightedCorrdX +=  realImageThreadxLocal;
					sumWeightedCorrdY +=  realImageThreadyLocal;
					sumWeighted += 1;
				}
				if (prop_buffer[bufferId] > 0 && pattern_buffer[patternId] == 1)
				{
					pointsOutSide = true;
				}
			}
		}

		if (!pointsOutSide)
		{
			double px = ((double)sumWeightedCorrdX) / ((double)sumWeighted);
			double py = ((double)sumWeightedCorrdY) / ((double)sumWeighted);
			
			int indexX = round(px);
			int indexY = round(py);



			if (indexX == realImageThreadx && indexY == realImageThready)
			{
				unsigned int index = atomicInc(params.centroidParameter.nCentroids, N_MAX_CENTROIDS);
				if (index < N_MAX_CENTROIDS)
				{
					params.centroidParameter.centerX[index] = (float)px;
					params.centroidParameter.centerY[index] = (float)py;
					mask_buffer[bufferId]++;
				//	printf("%d\t%lf\t%lf\t%d\t%d\t%d\t%d\n", index, params.centroidParameter.centerX[index], params.centroidParameter.centerY[index], blockIdx.x, blockIdx.y, realImageThreadx, realImageThready);
				}
			}
		}
	}




	__global__ void testClusterAdapter(S_IMG_PROC_KERNEL_PARAMS params)
	{

		int startIdx = threadIdx.x - params.maskParameter.maskOffset;
		int startIdy = threadIdx.y - params.maskParameter.maskOffset;
		int startId = startIdy*params.loadAndStoreParams.bufferWidth + startIdx;

		int realImageThreadx = threadIdx.x + blockDim.x*blockIdx.x;
		int realImageThready = threadIdx.y + blockDim.y*blockIdx.y;



			calculateCluster(
				params.imageParameter.mask,
				params.maskParameter.pattern,
				params.imageParameter.prop,
				params,
				startIdx, startIdy, startId,
				realImageThreadx, realImageThready);
		
	
	}


	struct cluster_state :test_state
	{


		int nPoints;
		float centerX[N_TEST_IMAGE_SIZE];
		float centerY[N_TEST_IMAGE_SIZE];

		cluster_state(unsigned char testImage[], unsigned char   testMask[],
			unsigned char   testPattern[], unsigned char testProb[], 
			int nPoints,
			float* centerX,
			float* centerY)
			:test_state(testImage, testMask, testPattern)
		{

			memcpy(prop, testProb, N_TEST_IMAGE_SIZE);


			this->nPoints = nPoints;
			memcpy(this->centerX, centerX, N_TEST_IMAGE_SIZE*sizeof(float));
			memcpy(this->centerY, centerY, N_TEST_IMAGE_SIZE * sizeof(float));
			

		}

		~cluster_state()
		{
;
		}




	};



	struct ClusterParameterTest : Kernel_Test, testing::WithParamInterface<cluster_state>
	{
		ClusterParameterTest()
		{

			data->img->set((unsigned char*)GetParam().img);
			data->mask->set((unsigned char*)GetParam().mask);
			data->pattern->set((unsigned char*)GetParam().pattern);
			data->prop->set((unsigned char*)GetParam().prop);
			data->centroids.reset();
			//data->centroids.set((float*)GetParam().centerX, (float*)GetParam().centerY,(int*) &GetParam().nPoints);
			
			launchConfig.grid = dim3(1, 1);
			launchConfig.block = dim3(3, 3);

		}

		void runTest()
		{
			data->centroids.reset();
			prepareTest();
			

			testClusterAdapter << <launchConfig.grid, launchConfig.block >> > (params);

			cleanUpTest();

		}

	};

	TEST_P(ClusterParameterTest, clusterDetection)
	{
		runTest();

		int n = *data->centroids.nCentroidsHost;
		EXPECT_EQ(n, GetParam().nPoints);
		for (int i = 0; i < n; i++)
		{

			EXPECT_FLOAT_EQ(data->centroids.centroidXHost[i], GetParam().centerX[i]);
			EXPECT_FLOAT_EQ(data->centroids.centroidYHost[i], GetParam().centerY[i]);
		}
	

		
	}





	unsigned char img1[] = {
		255,255,255,255,255,
		255,255,255,255,255,
		255,255,255,255,255,
		255,255,255,255,255,
		255,255,255,255,255
	};
	unsigned char mask1[] =
	{
		2,2,2,2,2,
		2,2,2,2,2,
		2,2,2,2,2,
		2,2,2,2,2,
		2,2,2,2,2
	};

	unsigned char prob1[] =
	{
		0,0,0,0,0,
		0,1,1,1,0,
		0,1,1,1,0,
		0,1,1,1,0,
		0,0,0,0,0
	};



	unsigned char pattern1[] = {
		1,1,1,
		1,1,1,
		1,1,1,
	};

	float centerX1[] = {
		1.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f
	};

	float centerY1[] = {
		1.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f
	};


	unsigned char prob2[] =
	{
		0,0,0,0,0,
		0,0,1,1,1,
		0,0,1,1,1,
		0,0,1,1,1,
		0,0,0,0,0
	};

	float centerX2[] = {
		2.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f
	};

	float centerY2[] = {
		1.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f
	};

	unsigned char prob3[] =
	{
		0,0,0,0,0,
		0,0,1,0,0,
		0,1,1,1,0,
		0,0,1,0,0,
		0,0,0,0,0
	};

	unsigned char pattern3[] = {
		0,1,0,
		1,1,1,
		0,1,0,
	};

	float centerX3[] = {
		1.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f
	};

	float centerY3[] = {
		1.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f
	};




	unsigned char prob4[] =
	{
		0,0,0,0,0,
		0,0,0,0,0,
		0,0,1,0,0,
		0,1,1,1,0,
		0,0,1,0,0
	};

	unsigned char pattern4[] = {
		0,1,0,
		1,1,1,
		0,1,0,
	};

	float centerX4[] = {
		1.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f
	};

	float centerY4[] = {
		2.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f
	};


	unsigned char prob5[] =
	{
		0,0,0,0,0,
		0,0,0,0,0,
		0,0,1,0,0,
		0,1,1,1,0,
		0,0,2,0,0
	};

	unsigned char pattern5[] = {
		0,1,0,
		1,1,1,
		0,1,0,
	};

	float centerX5[] = {
		1.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f
	};

	float centerY5[] = {
		2.16666666f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f,
		0.0f,0.0f,0.0f,0.0f,0.0f
	};

	//INSTANTIATE_TEST_CASE_P(Cluster, ClusterParameterTest, testing::Values(
	//	//cluster_state{img1, mask1, pattern1, prob1, 1, centerX1, centerY1 },
	//	//cluster_state{ img1, mask1, pattern1, prob2, 1, centerX2, centerY2 },
	//	//cluster_state{ img1, mask1, pattern3, prob3, 1, centerX3, centerY3 },
	//	//cluster_state{ img1, mask1, pattern4, prob4, 1, centerX4, centerY4 },
	//	cluster_state{ img1, mask1, pattern5, prob5, 1, centerX5, centerY5 }
	//));


}