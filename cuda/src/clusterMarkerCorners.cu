#include "clusterMarkerCorners.cuh"


namespace markercorners
{
	__device__ void calcCorners(
		float* cx,
		float* cy,		
		int a,
		int b,
		int c,
		int nCentroids,
		unsigned int* p_marker, 
		const S_IMG_PROC_KERNEL_PARAMS& params)
	{
		// A-------C
		// | _   /
		// | _  /
		// | _ /
		// B  /


		if (a >= nCentroids || b >= nCentroids || c >= nCentroids ||
			a == b || a == c || b == c)
		{
			return;
		}

		//check first distance between bc
		float dist_x = (cx[b] - cx[c]);
		float dist_y = (cy[b] - cy[c]);
		float dist = dist_x*dist_x + dist_y*dist_y;
		if (dist < params.qrCodeParameter.squareDistanceBCMin || dist > params.qrCodeParameter.squareDistanceBCMax)
		{
			return;
		}

		// checkistance ab
		float dist_ab_x = (cx[b] - cx[a]);
		float dist_ab_y = (cy[b] - cy[a]);
		dist = dist_ab_x*dist_ab_x + dist_ab_y*dist_ab_y;
		if (dist < params.qrCodeParameter.squareDistanceABMin || dist > params.qrCodeParameter.squareDistanceABMax)
		{
			return;
		}

		// checkistance ac
		float dist_ac_x = (cx[c] - cx[a]);
		float dist_ac_y = (cy[c] - cy[a]);
		dist = dist_ac_x*dist_ac_x + dist_ac_y*dist_ac_y;
		if (dist < params.qrCodeParameter.squareDistanceABMin || dist > params.qrCodeParameter.squareDistanceABMax)
		{
			return;
		}
		
		// check ac stands orthogonal to ab
		float norm_ab = sqrtf(dist_ab_x*dist_ab_x + dist_ab_y*dist_ab_y);
		float norm_ac = sqrtf(dist_ac_x*dist_ac_x + dist_ac_y*dist_ac_y);

		dist_ab_x /= norm_ab;
		dist_ab_y /= norm_ab;

		dist_ac_x /= norm_ac;
		dist_ac_y /= norm_ac;

		float k = dist_ab_x*dist_ac_y - dist_ab_y*dist_ac_x;				
		if (k > -0.95 || k < -1.05)
		{
			return;
		}
		// all criterias fullfilled
		unsigned int markerIndex = atomicInc(p_marker, N_MAX_CENTROIDS_MERGED);
		params.centroidParameter.centerAx[markerIndex] = cx[a];
		params.centroidParameter.centerAy[markerIndex] = cy[a];
		params.centroidParameter.centerBx[markerIndex] = cx[b];
		params.centroidParameter.centerBy[markerIndex] = cy[b];
		params.centroidParameter.centerCx[markerIndex] = cx[c];
		params.centroidParameter.centerCy[markerIndex] = cy[c];


	}

	//__global__ void testMarkerCornerAdapter(S_IMG_PROC_KERNEL_PARAMS params)
	//{

	//	int indexA;
	//	int indexB;
	//	int indexC;
	//	
	//	for (int x = 0; x < params.centroidParameter.nMarkerCornerThreadSize; x++)
	//	{
	//		for (int y = 0; y < params.centroidParameter.nMarkerCornerThreadSize; y++)
	//		{
	//			for (int z = 0; z < params.centroidParameter.nMarkerCornerThreadSize; z++)
	//			{
	//				indexA = threadIdx.x*params.centroidParameter.nMarkerCornerThreadSize + x;
	//				indexB = threadIdx.y*params.centroidParameter.nMarkerCornerThreadSize + y;
	//				indexC = threadIdx.z*params.centroidParameter.nMarkerCornerThreadSize + z;

	//				calcCorners(
	//					params.centroidParameter.centerX,
	//					params.centroidParameter.centerY,
	//					indexA,
	//					indexB,
	//					indexC,
	//					params);

	//			}

	//		}
	//	}
	//}





	struct centroid_test_state
	{

		float centroidX[20];
		float centroidY[20];
		unsigned int nCenters;


		float centroidAx[20];
		float centroidAy[20];

		float centroidBx[20];
		float centroidBy[20];

		float centroidCx[20];
		float centroidCy[20];


		unsigned int nMarkers;

		centroid_test_state(float ccx[], float ccy[],unsigned int n, 
							float ax[], float ay[],
							float bx[], float by[],
							float cx[], float cy[],
							int nMakers):nMarkers(0)
		{
			nCenters = n;
			memcpy(centroidX, ccx, n * sizeof(float));
			memcpy(centroidY, ccy, n * sizeof(float));

			this->nMarkers = nMakers;
			memcpy(centroidAx, ax, nMakers * sizeof(float));
			memcpy(centroidAy, ay, nMakers * sizeof(float));
			memcpy(centroidBx, bx, nMakers * sizeof(float));
			memcpy(centroidBy, by, nMakers * sizeof(float));
			memcpy(centroidCx, cx, nMakers * sizeof(float));
			memcpy(centroidCy, cy, nMakers * sizeof(float));
		}

		virtual ~centroid_test_state()
		{


		}

		friend std::ostream& operator<<(std::ostream& os, const centroid_test_state& obj)
		{
			
			for (int i = 0; i < obj.nCenters; i++)
			{
				os << obj.centroidX[i] << "\t" << obj.centroidY[i] << endl;
			}
			return os;
		}
	};


/*
	struct Centroid_Test : testing::Test
	{


		S_LAUNCH_CONFIG				launchConfig;
		S_IMG_PROC_KERNEL_PARAMS	params;
		

		CENTROIDS *centroids;
		CENTROIDS *a;
		CENTROIDS *b;
		CENTROIDS *c;



		Centroid_Test()
		{

			a = new CENTROIDS{ 20 };
			b = new CENTROIDS{ 20 };
			c = new CENTROIDS{ 20 };
			centroids = new CENTROIDS{ 20 };
			

			params.maskParameter.maskWidth = 3;
			params.maskParameter.maskOffset = -1;
			params.loadAndStoreParams.bufferWidth = 5;
			params.loadAndStoreParams.bufferHeight = 5;
			params.imageParameter.imageWidth = 5;
			params.imageParameter.imageHeight = 5;

			params.centroidParameter.centerX = centroids->centroidX;
			params.centroidParameter.centerY = centroids->centroidY;
			params.centroidParameter.nCentroids = centroids->nCentroids;

			params.centroidParameter.centerAx = a->centroidX;
			params.centroidParameter.centerAy = a->centroidY;

			params.centroidParameter.centerBx = b->centroidX;
			params.centroidParameter.centerBy = b->centroidY;

			params.centroidParameter.centerCx = c->centroidX;
			params.centroidParameter.centerCy = c->centroidY;

			params.centroidParameter.nMarker = a->nCentroids;



		}

		virtual ~Centroid_Test()
		{
			delete a;
			delete b;
			delete c;
		}

		void prepareTest()
		{
			a->reset();
			b->reset();
			c->reset();

			cudaDeviceSynchronize();
			cudaCheckError();
		}

		void cleanUpTest()
		{

			cudaDeviceSynchronize();
			cudaCheckError();

			a->get();
			b->get();
			c->get();
			centroids->get();

			cudaDeviceSynchronize();
			cudaCheckError();
		}


	};

	struct CentroidParameterTest : Centroid_Test, testing::WithParamInterface<centroid_test_state>
	{
		CentroidParameterTest()
		{


			centroids->reset();
			int nCenter = GetParam().nCenters;
			centroids->set((float*)GetParam().centroidX, (float*)GetParam().centroidY, (int*) &GetParam().nCenters);

			params.centroidParameter.nMarkerCornerThreadSize = 1;
			params.qrCodeParameter.squareDistanceBCMin = 1.8f;
			params.qrCodeParameter.squareDistanceBCMax = 2.2f;
			params.qrCodeParameter.squareDistanceABMin = 0.9f;
			params.qrCodeParameter.squareDistanceABMax = 1.1f;
			

			launchConfig.grid = dim3(1);
			launchConfig.block = dim3(nCenter, nCenter, nCenter);

		}

		void runTest()
		{
		
			prepareTest();


			testMarkerCornerAdapter << <launchConfig.grid, launchConfig.block >> > (params);

			cleanUpTest();

		}

	};

	TEST_P(CentroidParameterTest, markerDetection)
	{
		runTest();

		int n = *a->nCentroidsHost;
		EXPECT_EQ(n, GetParam().nMarkers);
		for (int i = 0; i < n; i++)
		{

			EXPECT_FLOAT_EQ(a->centroidXHost[i], GetParam().centroidAx[i]);
			EXPECT_FLOAT_EQ(a->centroidYHost[i], GetParam().centroidAy[i]);
			EXPECT_FLOAT_EQ(b->centroidXHost[i], GetParam().centroidBx[i]);
			EXPECT_FLOAT_EQ(b->centroidYHost[i], GetParam().centroidBy[i]);
			EXPECT_FLOAT_EQ(c->centroidXHost[i], GetParam().centroidCx[i]);
			EXPECT_FLOAT_EQ(c->centroidYHost[i], GetParam().centroidCy[i]);
		}



	}


	float ccx1[] = { 0 , 0, 1};
	float ccy1[] = { 0 , 1, 0};
	float ax1[] = { 0 };
	float ay1[] = { 0 };
	float bx1[] = { 0 };
	float by1[] = { 1.0f };
	float cx1[] = { 1.0f };
	float cy1[] = { 0 };

	float ccx2[] = { 0 , 0, 1 };
	float ccy2[] = { 1 , 0, 0 };
	float ax2[] = { 0 };
	float ay2[] = { 0 };
	float bx2[] = { 0 };
	float by2[] = { 1.0f };
	float cx2[] = { 1.0f };
	float cy2[] = { 0 };

	float ccx3[] = { 0 , 0, 1 , 2, 2, 3};
	float ccy3[] = { 1 , 0, 0 , 3, 2, 2};
	float ax3[] = { 0 , 2.0f };
	float ay3[] = { 0 , 2.0f };
	float bx3[] = { 0 , 2.0f };
	float by3[] = { 1.0f,3.0f };
	float cx3[] = { 1.0f, 3.0f };
	float cy3[] = { 0 , 2.0f };


	INSTANTIATE_TEST_CASE_P(Marker, CentroidParameterTest, testing::Values(
		centroid_test_state{ ccx1, ccy1, 3,
							 ax1, ay1,
							 bx1, by1,
							 cx1, cy1,
							1},
		centroid_test_state{ ccx2, ccy2, 3,
							ax2, ay2,
							bx2, by2,
							cx2, cy2,
							1 },
		centroid_test_state{ ccx3, ccy3, 6,
		ax3, ay3,
		bx3, by3,
		cx3, cy3,
		2 }
	));*/

}