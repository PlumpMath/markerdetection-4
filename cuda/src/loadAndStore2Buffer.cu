#include "loadAndStore2Buffer.cuh"


#include <iostream>
#include <sstream>
#include <exception>

namespace loadandstore2bufer
{
	
	using namespace std;

	__device__ void initializebuffer(unsigned char* image_buffer, unsigned char* mask_buffer, unsigned char* prop_buffer, S_IMG_PROC_KERNEL_PARAMS& params)
	{

		int bufferId;
		int realBufferThreadx;
		int realBufferThready;

		for (int r = 0; r < params.loadAndStoreParams.threadFaktorY; r++)
		{
			realBufferThready = params.loadAndStoreParams.threadFaktorY*threadIdx.y + r;			
			for (int c = 0; c < params.loadAndStoreParams.threadFaktorX; c++)
			{
				realBufferThreadx = params.loadAndStoreParams.threadFaktorX*threadIdx.x + c;				
				if (realBufferThreadx >= 0 && realBufferThreadx < params.loadAndStoreParams.bufferWidth &&
					realBufferThready >= 0 && realBufferThready < params.loadAndStoreParams.bufferHeight )
				{
					bufferId = realBufferThready*params.loadAndStoreParams.bufferWidth + realBufferThreadx;
					image_buffer[bufferId] = 0;
					mask_buffer[bufferId] = 0;
					prop_buffer[bufferId] = 0;
				}
			}
		}
	}


	__device__ void readFrombuffer(const unsigned char* const buffer, int bufferId, S_IMG_PROC_KERNEL_PARAMS& params)
	{
		int imageid;
		int realImageThreadx;
		int realImageThready;

		realImageThreadx = threadIdx.x + blockDim.x*blockIdx.x;
		realImageThready = threadIdx.y + blockDim.y*blockIdx.y;
		imageid = realImageThready *gridDim.x*blockDim.x + realImageThreadx;

		if (realImageThreadx >= 0 && realImageThreadx < params.imageParameter.imageWidth &&
			realImageThready >= 0 && realImageThready < params.imageParameter.imageHeight)
		{
			params.imageParameter.prop[imageid] = buffer[bufferId];
		}

		
	}

	__device__ void readFrombuffermask(const unsigned char* const buffer, int bufferId, S_IMG_PROC_KERNEL_PARAMS& params)
	{
		int imageid;
		int realImageThreadx;
		int realImageThready;

		realImageThreadx = threadIdx.x + blockDim.x*blockIdx.x;
		realImageThready = threadIdx.y + blockDim.y*blockIdx.y;
		imageid = realImageThready *gridDim.x*blockDim.x + realImageThreadx;

		if (realImageThreadx >= 0 && realImageThreadx < params.imageParameter.imageWidth &&
			realImageThready >= 0 && realImageThready < params.imageParameter.imageHeight)
		{
			params.imageParameter.mask[imageid] = buffer[bufferId];
		}


	}


	__device__ void store2buffer(unsigned char* const buffer,
								 unsigned char* const mask_buffer,
								 S_IMG_PROC_KERNEL_PARAMS& params)
	{
		int imageid;
		int bufferId;

		int realImageThreadx;
		int realImageThready;

		int realBufferThreadx;
		int realBufferThready;

		for (int r = 0; r < params.loadAndStoreParams.threadFaktorY; r++)
		{
			realBufferThready = params.loadAndStoreParams.threadFaktorY*threadIdx.y + r;
			realImageThready = params.loadAndStoreParams.threadFaktorY*threadIdx.y + params.loadAndStoreParams.maskOffset + r + blockDim.y*blockIdx.y;
			for (int c = 0; c < params.loadAndStoreParams.threadFaktorX; c++)
			{
				realBufferThreadx = params.loadAndStoreParams.threadFaktorX*threadIdx.x + c;
				realImageThreadx = params.loadAndStoreParams.threadFaktorX*threadIdx.x + params.loadAndStoreParams.maskOffset + c + blockDim.x*blockIdx.x;

				if (realBufferThreadx >= 0 && realBufferThreadx < params.loadAndStoreParams.bufferWidth &&
					realBufferThready >= 0 && realBufferThready < params.loadAndStoreParams.bufferHeight &&
					realImageThreadx >= 0 && realImageThreadx < params.imageParameter.imageWidth &&
					realImageThready >= 0 && realImageThready < params.imageParameter.imageHeight)
				{



					bufferId = realBufferThready*params.loadAndStoreParams.bufferWidth + realBufferThreadx;
					imageid = realImageThready * params.imageParameter.imageWidth + realImageThreadx;

					if (params.undistortParamer.mask[imageid])
					{

						buffer[bufferId] = params.imageParameter.img[params.undistortParamer.mapy[imageid] * params.imageParameter.imageWidth
							+ params.undistortParamer.mapx[imageid]];
						mask_buffer[bufferId] = 1;
					}


					
					
				}

			}
		}

	}

	__global__ void testLoadAndStorKernel(S_IMG_PROC_KERNEL_PARAMS params, int blockx, int blocky)
	{
		if (blockIdx.x == blockx-1 && blockIdx.y == blocky-1)
		{
			store2buffer(params.imageParameter.prop, params.imageParameter.mask, params);
		}
	}

	


	struct LoadAndStoreTest : testing::Test
	{

		S_IMG_PROC_KERNEL_PARAMS		params;
		S_LAUNCH_CONFIG					launchConfig;

		DeviceBuffer<unsigned char> *image;
		DeviceBuffer<unsigned char> *buffer;
		DeviceBuffer<unsigned char> *maskbuffer;

		int resx;
		int resy;
		int threadIdx;
		int threadIdy;
		
		



		void init(int resx, int resy, int blocksizex, int blocksizey, int maskSize, int threadIdx, int threadIdy)
		{
			this->resx = resx;
			this->resy = resy;
			this->threadIdx = threadIdx;
			this->threadIdy = threadIdy;
			

			image = new DeviceBuffer<unsigned char>(resx * resy);
			buffer = new DeviceBuffer<unsigned char>(resx * resy);
			maskbuffer = new DeviceBuffer<unsigned char>(resx * resy);



			params.imageParameter.imageHeight = resy;
			params.imageParameter.imageWidth = resx;

			params.imageParameter.img = image->getDeviceData();
			params.imageParameter.mask = maskbuffer->getDeviceData();
			params.imageParameter.prop = buffer->getDeviceData();

			params.loadAndStoreParams.init(maskSize, resx, resy, blocksizex, blocksizey);

			for (int x = 0; x < resx; x++)
			{
				for (int y = 0; y < resy; y++)
				{
					image->getHostData()[y*resx + x] = y*resx + x;
				}
			}
			image->set();
			launchConfig.grid = dim3(threadIdx, threadIdy);
			launchConfig.block = dim3(blocksizex, blocksizey);
		}

		void runTest()
		{



			testLoadAndStorKernel << <launchConfig.grid, launchConfig.block >> > (params, threadIdx, threadIdy);
			cudaCheckError();

			cudaDeviceSynchronize();
			cudaCheckError();

			buffer->get();
			maskbuffer->get();
			



		}

		~LoadAndStoreTest()
		{
			delete image;
			delete buffer;
			delete maskbuffer;
		}
	};

	/*TEST_F(LoadAndStoreTest, loadAndStoreTest1)
	{

		init(10, 10, 1, 1, 3, 1, 1);
		runTest();

		EXPECT_EQ(buffer->getHostData()[0], 0);
		EXPECT_EQ(buffer->getHostData()[1], 0);
		EXPECT_EQ(buffer->getHostData()[2], 0);

		EXPECT_EQ(buffer->getHostData()[3], 0);
		EXPECT_EQ(buffer->getHostData()[4], 0);
		EXPECT_EQ(buffer->getHostData()[5], 1);

		EXPECT_EQ(buffer->getHostData()[6], 0);
		EXPECT_EQ(buffer->getHostData()[7], 10);
		EXPECT_EQ(buffer->getHostData()[8], 11);

		EXPECT_EQ(maskbuffer->getHostData()[0], 0);
		EXPECT_EQ(maskbuffer->getHostData()[1], 0);
		EXPECT_EQ(maskbuffer->getHostData()[2], 0);

		EXPECT_EQ(maskbuffer->getHostData()[3], 0);
		EXPECT_EQ(maskbuffer->getHostData()[4], 1);
		EXPECT_EQ(maskbuffer->getHostData()[5], 1);

		EXPECT_EQ(maskbuffer->getHostData()[6], 0);
		EXPECT_EQ(maskbuffer->getHostData()[7], 1);
		EXPECT_EQ(maskbuffer->getHostData()[8], 1);

	}

	TEST_F(LoadAndStoreTest, loadAndStoreTest2)
	{

		init(10, 10, 1, 1, 3, 2, 2);
		runTest();

		EXPECT_EQ(buffer->getHostData()[0], 0);
		EXPECT_EQ(buffer->getHostData()[1], 1);
		EXPECT_EQ(buffer->getHostData()[2], 2);

		EXPECT_EQ(buffer->getHostData()[3], 10);
		EXPECT_EQ(buffer->getHostData()[4], 11);
		EXPECT_EQ(buffer->getHostData()[5], 12);

		EXPECT_EQ(buffer->getHostData()[6], 20);
		EXPECT_EQ(buffer->getHostData()[7], 21);
		EXPECT_EQ(buffer->getHostData()[8], 22);

		EXPECT_EQ(maskbuffer->getHostData()[0], 1);
		EXPECT_EQ(maskbuffer->getHostData()[1], 1);
		EXPECT_EQ(maskbuffer->getHostData()[2], 1);

		EXPECT_EQ(maskbuffer->getHostData()[3], 1);
		EXPECT_EQ(maskbuffer->getHostData()[4], 1);
		EXPECT_EQ(maskbuffer->getHostData()[5], 1);

		EXPECT_EQ(maskbuffer->getHostData()[6], 1);
		EXPECT_EQ(maskbuffer->getHostData()[7], 1);
		EXPECT_EQ(maskbuffer->getHostData()[8], 1);

	}


	TEST_F(LoadAndStoreTest, loadAndStoreTest3)
	{

		init(10, 10, 1, 1, 3, 10, 10);
		runTest();

		EXPECT_EQ(buffer->getHostData()[0], 88);
		EXPECT_EQ(buffer->getHostData()[1], 89);
		EXPECT_EQ(buffer->getHostData()[2], 0);

		EXPECT_EQ(buffer->getHostData()[3], 98);
		EXPECT_EQ(buffer->getHostData()[4], 99);
		EXPECT_EQ(buffer->getHostData()[5], 0);

		EXPECT_EQ(buffer->getHostData()[6], 0);
		EXPECT_EQ(buffer->getHostData()[7], 0);
		EXPECT_EQ(buffer->getHostData()[8], 0);

		EXPECT_EQ(maskbuffer->getHostData()[0], 1);
		EXPECT_EQ(maskbuffer->getHostData()[1], 1);
		EXPECT_EQ(maskbuffer->getHostData()[2], 0);

		EXPECT_EQ(maskbuffer->getHostData()[3], 1);
		EXPECT_EQ(maskbuffer->getHostData()[4], 1);
		EXPECT_EQ(maskbuffer->getHostData()[5], 0);

		EXPECT_EQ(maskbuffer->getHostData()[6], 0);
		EXPECT_EQ(maskbuffer->getHostData()[7], 0);
		EXPECT_EQ(maskbuffer->getHostData()[8], 0);

	}*/

}