#include "matchpattern.cuh"

namespace matchpattern
{

	__device__ void readBinMask(unsigned char* binmask, const S_IMG_PROC_KERNEL_PARAMS& params)
	{
		
		int realBufferThreadx;
		int realBufferThready;
		for (int c = 0; c < params.maskParameter.threadFaktorX; c++)
		{
			realBufferThreadx = params.maskParameter.threadFaktorX *threadIdx.x + c;
			for (int r = 0; r < params.maskParameter.threadFaktorY; r++)
			{
				realBufferThready = params.maskParameter.threadFaktorY*threadIdx.y + r;
				if (realBufferThreadx< params.maskParameter.maskWidth && realBufferThready < params.maskParameter.maskWidth)
				{
					binmask[realBufferThready*params.maskParameter.maskWidth + realBufferThreadx] = params.maskParameter.pattern[realBufferThready*params.maskParameter.maskWidth + realBufferThreadx];
				}

			}
		}

	}




	__device__ bool calculatePatternScore(
		const unsigned char* img_buffer,
		unsigned char* mask_buffer,
		const unsigned char* pattern_buffer,		
		const S_IMG_PROC_KERNEL_PARAMS& params,
		int startIdx, int startIdy, int startId,
		int imageX, int imageY)
	{

		if (mask_buffer[startId] < 2)
		{			
			return false;
		}


		int bufferId;
		int patternId;
		int realBufferThreadx;
		int realBufferThready;

		float diffGrad;

		int sumWhite = 0;
		int sumBlack = 0;
		int maxWhite = 0;
		unsigned char p;





		for (int c = params.maskParameter.maskOffset; c <= -params.maskParameter.maskOffset; c++)
		{
			realBufferThreadx = startIdx + c;

			for (int r = params.maskParameter.maskOffset; r <= -params.maskParameter.maskOffset; r++)
			{
				realBufferThready = startIdy + r;
				bufferId = realBufferThready*params.loadAndStoreParams.bufferWidth + realBufferThreadx;

				patternId = (r - params.maskParameter.maskOffset) * params.maskParameter.maskWidth + (c - params.maskParameter.maskOffset);

				if (pattern_buffer[patternId] == 1)
				{
					sumWhite += img_buffer[bufferId];
					if (img_buffer[bufferId] > maxWhite)
					{
						maxWhite = img_buffer[bufferId];
					}
				}
				if (pattern_buffer[patternId] == 0)
				{
					sumBlack += img_buffer[bufferId];
				}
			}
		}

		if (maxWhite > 0)
		{

			diffGrad = (float((sumWhite - sumBlack) * 255)) / (float(maxWhite*params.maskParameter.nMaskWhite));
			if (roundf(diffGrad) > 255)
			{
				p = 255;
			}
			else
			{
				p = (unsigned char)roundf(diffGrad);
			}
			
			if (p >= params.maskParameter.threshPattern)
			{
			
				int index = atomicInc(params.centroidParameter.nCentroids, N_MAX_CENTROIDS);
				params.centroidParameter.centerX[index] = imageX;
				params.centroidParameter.centerY[index] = imageY;
				params.centroidParameter.centerProp[index] = p;
				mask_buffer[startId]++;				
				return true;
			}
			else
			{				
				return false;
			}
		}
		else
		{			
			return false;
		}
	


	}














	//__global__ void testBinMaskAdapter(S_IMG_PROC_KERNEL_PARAMS params)
	//{

	//	int startIdx = threadIdx.x - params.maskParameter.maskOffset;
	//	int startIdy = threadIdx.y - params.maskParameter.maskOffset;
	//	int startId = startIdy*params.loadAndStoreParams.bufferWidth + startIdx;

	//	printf("%d\t%d", startIdx, startIdy);

	//	calculatePatternScore(
	//		params.imageParameter.img,
	//		params.imageParameter.mask,
	//		params.maskParameter.pattern,
	//		params.imageParameter.prop,
	//		params,
	//		startIdx, startIdy, startId);

	//}




/*
	struct mask_state :test_state
	{

		int indexProb;
		int valueProb;

		mask_state(unsigned char testImage[], unsigned char   testMask[], unsigned char   testPattern[], int indexProb, unsigned char  valueProb)
			:test_state(testImage, testMask, testPattern)
		{

			this->indexProb = indexProb;
			this->valueProb = valueProb;
			this->prop[indexProb] = valueProb;

		}

	};


	struct edge_state :test_state
	{

		int offsetEdge2Center;
		int offsetEdge2CenterCorner;
		int dimEdge;
		int minAbsoluteDiffEdge;
		int minRelativeDiffEdge;
		int indexProb;
		int valueProb;

		edge_state(unsigned char testImage[], unsigned char   testMask[], unsigned char   testPattern[],
			int offsetEdge2Center,
			int offsetEdge2CenterCorner,
			int dimEdge,
			int minAbsoluteDiffEdge,
			int minRelativeDiffEdge,
			int indexProb, unsigned char  valueProb)
			:test_state(testImage, testMask, testPattern)
		{

			this->indexProb = indexProb;
			this->valueProb = valueProb;
			this->prop[indexProb] = valueProb;

			this->offsetEdge2Center = offsetEdge2Center;
			this->offsetEdge2CenterCorner = offsetEdge2CenterCorner;
			this->dimEdge = dimEdge;
			this->minAbsoluteDiffEdge = minAbsoluteDiffEdge;
			this->minRelativeDiffEdge = minRelativeDiffEdge;


		}

	};










	struct BinMaskParameterTest : Kernel_Test, testing::WithParamInterface<mask_state>
	{
		BinMaskParameterTest()
		{
			data->img->set((unsigned char*)GetParam().img);
			data->mask->set((unsigned char*)GetParam().mask);
			data->pattern->set((unsigned char*)GetParam().pattern);
			data->prop->reset();

			int n = 0;
			for (int i = 0; i < 9; i++)
			{
				if (data->pattern->getHostData()[i] == 1)
				{
					n++;
				}
			}
			params.maskParameter.nMaskWhite = n;
			params.maskParameter.threshPattern = 0;
		}

		void runTest()
		{
			prepareTest();

			testBinMaskAdapter << <launchConfig.grid, launchConfig.block >> > (params);

			cleanUpTest();
		}
	};







	TEST_P(BinMaskParameterTest, completeMask3)
	{
		runTest();
		EXPECT_EQ(data->prop->getHostData()[GetParam().indexProb], GetParam().valueProb);
	}



	unsigned char img1[] = { 255,255,255,255,255,
	255,255,255,255,255,
	255,255,255,255,255,
	255,255,255,255,255,
	255,255,255,255,255
	};
	unsigned char mask1[] = {
		1,1,1,1,1,
		1,1,1,1,1,
		1,1,1,1,1,
		1,1,1,1,1,
		1,1,1,1,1
	};
	unsigned char pattern1[] = { 1,1,1,
	1,1,1,
	1,1,1,
	};


	unsigned char img2[] = {
		0,0,0,255,255,255,
		255,255,255,255,255,255,
		255,255,255,255,255,255,
		255,255,255,255,255,255,
		255,255,255,255,255,255
	};

	unsigned char pattern2[] = {
		1,1,1,
		1,1,1,
		1,1,1,
	};

	unsigned char img3[] = {
		0,0,0,255,255,
		255,255,255,255,255,
		255,255,255,255,255,
		255,255,255,255,255,
		255,255,255,255,255
	};

	unsigned char pattern3[] = { 0,0,0,
	1,1,1,
	1,1,1,
	};

	unsigned char img4[] = {
		0,0,0,128,128,
		128,128,128,128,128,
		128,128,128,128,128,
		128,128,128,128,128,
		128,128,128,128,128
	};

	unsigned char pattern4[] = { 0,0,0,
	1,1,1,
	1,1,1,
	};


	unsigned char img5[] =
	{
		0,0,0,128,128,
		0,0,0,128,128,
		0,0,0,128,128,
		128,128,128,128,128,
		128,128,128,128,128,
	};

	unsigned char pattern5[] =
	{
		0,0,0,
		0,0,0,
		0,0,0,
	};

	INSTANTIATE_TEST_CASE_P(Default, BinMaskParameterTest,
		testing::Values(
			mask_state{ img1, mask1, pattern1, 6, 255 },
			mask_state{ img2, mask1, pattern2, 6, 170 },
			mask_state{ img3, mask1, pattern3, 6, 255 },
			mask_state{ img4, mask1, pattern4, 6, 255 },
			mask_state{ img5, mask1, pattern5, 6, 0 }
		)
	);*/
}

