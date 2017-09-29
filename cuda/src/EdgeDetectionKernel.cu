#include "EdgeDetectionKernel.cuh"


namespace edgedetection
{

	__device__ bool isEdge(const S_IMG_PROC_KERNEL_PARAMS& params, const int& sumWhite, const int& sumBlack)
	{
		int diff;
		int diffGrad;
		if (sumWhite > 0)
		{
			diff = sumWhite - sumBlack;
			diffGrad = (diff * 255) / (sumWhite);
		}
		else
		{
			diffGrad = 0;
		}


		if (diff > params.maskParameter.minAbsoluteDiffEdge && diffGrad > params.maskParameter.minRelativeDiffEdge)
		{
			return true;
		}
		return false;
	}

__device__ bool calculateEdge(
	const unsigned char* buffer,
	unsigned char* mask_buffer,
	const S_IMG_PROC_KERNEL_PARAMS& params,
	int startIdx, int startIdy, int startId)
{
	int bufferId;
	int realBufferThreadx;
	int realBufferThready;



	int sumWhite = 0;
	int sumBlack = 0;


	bool hastNorth = false, hasWest = false, hasNorthWest = false, hastSouthWest = false;

	////pattern west
	realBufferThready = startIdy;

	realBufferThreadx = startIdx - params.maskParameter.offsetEdge2Center - params.maskParameter.dimEdge;
	bufferId = realBufferThready*params.loadAndStoreParams.bufferWidth + realBufferThreadx;
	sumWhite = buffer[bufferId];

	realBufferThreadx = startIdx - params.maskParameter.offsetEdge2Center + params.maskParameter.dimEdge;
	bufferId = realBufferThready*params.loadAndStoreParams.bufferWidth + realBufferThreadx;
	sumBlack = buffer[bufferId];

	hasWest = isEdge(params, sumWhite, sumBlack);

	//pattern north	
	realBufferThreadx = startIdx;
	realBufferThready = startIdy - params.maskParameter.offsetEdge2Center - params.maskParameter.dimEdge;
	bufferId = realBufferThready*params.loadAndStoreParams.bufferWidth + realBufferThreadx;
	sumWhite = buffer[bufferId];

	realBufferThready = startIdy - params.maskParameter.offsetEdge2Center + params.maskParameter.dimEdge;
	bufferId = realBufferThready*params.loadAndStoreParams.bufferWidth + realBufferThreadx;
	sumBlack = buffer[bufferId];

	hastNorth = isEdge(params, sumWhite, sumBlack);
	
	//if (hasWest && hastNorth)
	if (hastNorth)
	{
		mask_buffer[startId]++;
		return true;
	}



	////pattern south west	
	//realBufferThreadx = startIdx - params.maskParameter.offsetEdge2CenterCorner - params.maskParameter.dimEdgeCorner;
	//realBufferThready = startIdy + params.maskParameter.offsetEdge2CenterCorner + params.maskParameter.dimEdgeCorner;
	//bufferId = realBufferThready*params.loadAndStoreParams.bufferWidth + realBufferThreadx;
	//sumWhite = buffer[bufferId];

	//realBufferThreadx = startIdx - params.maskParameter.offsetEdge2CenterCorner + params.maskParameter.dimEdgeCorner;
	//realBufferThready = startIdy + params.maskParameter.offsetEdge2CenterCorner - params.maskParameter.dimEdgeCorner;
	//bufferId = realBufferThready*params.loadAndStoreParams.bufferWidth + realBufferThreadx;
	//sumBlack = buffer[bufferId];

	//hastSouthWest = isEdge(params, sumWhite, sumBlack);
	//


	//////pattern north west	
	//realBufferThreadx = startIdx - params.maskParameter.offsetEdge2CenterCorner - params.maskParameter.dimEdgeCorner;
	//realBufferThready = startIdy - params.maskParameter.offsetEdge2CenterCorner - params.maskParameter.dimEdgeCorner;
	//bufferId = realBufferThready*params.loadAndStoreParams.bufferWidth + realBufferThreadx;
	//sumWhite = buffer[bufferId];

	//realBufferThreadx = startIdx - params.maskParameter.offsetEdge2CenterCorner + params.maskParameter.dimEdgeCorner;
	//realBufferThready = startIdy - params.maskParameter.offsetEdge2CenterCorner + params.maskParameter.dimEdgeCorner;
	//bufferId = realBufferThready*params.loadAndStoreParams.bufferWidth + realBufferThreadx;
	//sumBlack = buffer[bufferId];

	//hasNorthWest = isEdge(params, sumWhite, sumBlack);;
	//


	//if (hastSouthWest && hasNorthWest)
	//{
	//	mask_buffer[startId]++;
	//	return true;
	//}

	return false;

}

__global__ void testEdgeAdapter(S_IMG_PROC_KERNEL_PARAMS params)
{

	int startIdx = threadIdx.x+4 ;
	int startIdy = threadIdx.y+2 ;
	int startId = startIdy*params.loadAndStoreParams.bufferWidth + startIdx;

	calculateEdge(
		params.imageParameter.img,
		params.imageParameter.mask,
		params,
		startIdx, startIdy, startId);

}


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




struct EdgeParameterTest : Kernel_Test, testing::WithParamInterface<edge_state>
{
	EdgeParameterTest()
	{
		data->img->set((unsigned char*)GetParam().img);
		data->mask->set((unsigned char*)GetParam().mask);
		data->pattern->set((unsigned char*)GetParam().pattern);
		data->prop->reset();

		params.maskParameter.offsetEdge2Center = GetParam().offsetEdge2Center;
		params.maskParameter.offsetEdge2CenterCorner = GetParam().offsetEdge2CenterCorner;
		params.maskParameter.dimEdge = GetParam().dimEdge;
		params.maskParameter.dimEdgeCorner = GetParam().dimEdge;
		params.maskParameter.minAbsoluteDiffEdge = GetParam().minAbsoluteDiffEdge;
		params.maskParameter.minRelativeDiffEdge = GetParam().minRelativeDiffEdge;
	}


	void runTest()
	{
		data->mask->reset();

		prepareTest();

		testEdgeAdapter << <launchConfig.grid, launchConfig.block >> > (params);


		cleanUpTest();

	}
};

TEST_P(EdgeParameterTest, edgeDetector)
{
	runTest();	
	EXPECT_EQ(data->mask->getHostData()[GetParam().indexProb], GetParam().valueProb);
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
1,1,1,1,1,
1,1,1,1,1,
1,1,1,1,1,
1,1,1,1,1
};
unsigned char pattern1[] = {
1,1,1,
1,1,1,
1,1,1,
};


unsigned char img2[] = {
	0,0,0,0,  0,
	0,0,0,0,  255,
	0,0,255,0,0,
	0,0,0,0,0,0,
	0,0,0,0,0,0
};

unsigned char img3[] = {
	0,0,0,0,  255,
	0,0,0,0,  0,
	0,0,255,0,0,
	0,0,0,0,0,0,
	0,0,0,0,0,0
};



unsigned char img4[] = {
	0,0,0,0,  2,
	0,0,0,0,  0,
	0,0,2,0,0,
	0,0,0,0,0,0,
	0,0,0,0,0,0
};



unsigned char img5[] =
{
	0,0,0,0,  2,
	0,0,0,0,  0,
	0,0,1,0,0,
	0,0,0,0,0,0,
	0,0,0,0,0,0
};

unsigned char img6[] =
{
	0,0,0,0,  1,
	0,0,0,0,  0,
	0,0,2,0,0,
	0,0,0,0,0,0,
	0,0,0,0,0,0
};
unsigned char img7[] =
{
	0,0,0,0,  1,
	0,0,0,0,  0,
	0,0,2,0,0,
	0,0,0,0,0,0,
	0,0,0,0,0,0
};

unsigned char img8[] =
{
	0,0,0,0,  2,
	0,0,0,0,  0,
	0,0,1,0,1,
	0,0,0,0,0,0,
	0,0,0,0,0,0
};

unsigned char img9[] =
{
	0,0,0,0,  2,
	0,0,0,0,  0,
	0,0,2,0,0,
	0,0,0,0,0,0,
	0,0,0,0,0,0
};

unsigned char img10[] =
{
	0,0,2,0,0,
	0,0,0,0,0,
	0,0,0,0,0,
	0,0,0,0,0,
	0,0,2,0,0,
};

unsigned char img11[] =
{
	0,0,1,0,  0,
	0,0,0,0,  0,
	0,0,0,0,0,
	0,0,0,0,0,0,
	0,0,2,0,0,0
};








INSTANTIATE_TEST_CASE_P(Edge, EdgeParameterTest,
	testing::Values(
		edge_state{ img1, mask1, pattern1,1,0,1,1,1, 14, 0 },
		edge_state{ img2, mask1, pattern1,1,0,1,1,1, 14, 0 },
		edge_state{ img3, mask1, pattern1,1,0,1,1,1, 14, 1 },
		edge_state{ img4, mask1, pattern1,1,0,1,1,1, 14, 1 },
		edge_state{ img5, mask1, pattern1,1,0,1,1,1, 14, 0 },
		edge_state{ img6, mask1, pattern1,1,0,1,1,1, 14, 0 },
		edge_state{ img7, mask1, pattern1,1,0,1,1,2, 14, 0 },
		edge_state{ img8, mask1, pattern1,1,0,1,1,2, 14, 0 },
		edge_state{ img9, mask1, pattern1,1,0,1,1,2, 14, 1 },
		edge_state{ img10, mask1, pattern1,0,1,1,1,2, 14, 1 },
		edge_state{ img11, mask1, pattern1,0,1,1,1,2, 14, 0 }
	)
);
 }