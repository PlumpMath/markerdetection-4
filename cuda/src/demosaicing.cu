
#include "demosaicing.cuh"

#include <istream>
#include <fstream>





namespace demosaicing
{
	
	#define min_img(v, minv) ((v < minv) ? v : minv)
	#define max_img(v, maxv) ((v > maxv) ? v : maxv)


	__device__ void loadAndMinMax(	unsigned char* minG, unsigned char* maxG,
									unsigned char* minR, unsigned char* maxR,
									unsigned char* minB, unsigned char* maxB,
									const S_IMG_PROC_KERNEL_PARAMS& params)
	{
		int threadFaktorX = divceil(params.imageParameter.imageWidth, blockDim.x);
		int threadFaktorY = divceil(params.imageParameter.imageHeight, blockDim.y);
		int dest_id = threadIdx.y*blockDim.x + threadIdx.x;
		int x, y;

		unsigned char maxRs = 0, maxGs = 0, maxBs = 0;
		unsigned char minRs = 255, minGs = 255, minBs = 255;
		int v;

		for (int yo = 0; yo < threadFaktorY; yo++)
		{
			y = threadFaktorY*threadIdx.y + yo;
			for (int xo = 0; xo < threadFaktorX; xo++)
			{
				x = threadFaktorX*threadIdx.x + xo;
				if (x < params.imageParameter.imageWidth &&
					y < params.imageParameter.imageHeight)
				{

					v = params.imageParameter.img_raw[y*params.imageParameter.imageWidth + x];
					if (x % 2 == 0 && y % 2 == 0 || (x + 1) % 2 == 0 && (y + 1) % 2 == 0)
					{
						//green
						minGs = min_img(v, minGs);
						maxGs = max_img(v, maxGs);
					}else if((x + 1) % 2 == 0 && y % 2 == 0)
					{
						//blue
						minRs = min_img(v, minRs);
						maxRs = max_img(v, maxRs);
					}
					else if (x % 2 == 0 && (y + 1) % 2 == 0)
					{
						//red
						minBs = min_img(v, minBs);
						maxBs = max_img(v, maxBs);
					}

				}

			}
		}
		minG[dest_id] = minGs;
		maxG[dest_id] = maxGs;
		minB[dest_id] = minBs;
		maxB[dest_id] = maxBs;
		minR[dest_id] = minRs;
		maxR[dest_id] = maxRs;
		__syncthreads();






		for (int i = SHARED_BUFFER_SIZE / 2; i > 0;  i >>= 1)
		{
			__syncthreads();
			if (dest_id < i)
			{
				minG[dest_id] = min_img(minG[dest_id], minG[dest_id + i]);
				maxG[dest_id] = max_img(maxG[dest_id], maxG[dest_id + i]);

				minB[dest_id] = min_img(minB[dest_id], minB[dest_id + i]);
				maxB[dest_id] = max_img(maxB[dest_id], maxB[dest_id + i]);

				minR[dest_id] = min_img(minR[dest_id], minR[dest_id + i]);
				maxR[dest_id] = max_img(maxR[dest_id], maxR[dest_id + i]);

			}
		}
		__syncthreads();
		if (dest_id == 0)
		{
			//use m for max and o for min first calc
			maxG[dest_id] = maxG[dest_id] - minG[dest_id];
			maxR[dest_id] = maxR[dest_id] - minR[dest_id];
			maxB[dest_id] = maxB[dest_id] - minB[dest_id];
			
		}

		__syncthreads();

		//normalize Image
		
		
		for (int yo = 0; yo < threadFaktorY; yo++)
		{
			y = threadFaktorY*threadIdx.y + yo;
			for (int xo = 0; xo < threadFaktorX; xo++)
			{
				x = threadFaktorX*threadIdx.x + xo;
				if (x < params.imageParameter.imageWidth &&
					y < params.imageParameter.imageHeight)
				{

					v = params.imageParameter.img_raw[y*params.imageParameter.imageWidth + x];
					if (x % 2 == 0 && y % 2 == 0 || (x + 1) % 2 == 0 && (y + 1) % 2 == 0)
					{
						//green						
						v = (v - ((int)minG[0])) * 255;
						v /= maxG[0];
						params.imageParameter.img[y * params.imageParameter.imageWidth + x] = (unsigned char)v;
					}
					else if ((x + 1) % 2 == 0 && y % 2 == 0)
					{
						//red						
						v = (v - ((int)minR[0])) * 255;
						v /= maxR[0];
						params.imageParameter.img[y * params.imageParameter.imageWidth + x] = (unsigned char)v;
					}
					else if (x % 2 == 0 && (y + 1) % 2 == 0)
					{
						//blue						
						v = (v - ((int)minB[0])) * 255;
						v /= maxB[0];
						params.imageParameter.img[y * params.imageParameter.imageWidth + x] = (unsigned char)v;
					}
					 

				}

			}
		}
		__syncthreads();

	}


	


	__global__ void testDebayer(S_IMG_PROC_KERNEL_PARAMS params)
	{

		__shared__ unsigned char minG[SHARED_BUFFER_SIZE];			
		__shared__ unsigned char maxG[SHARED_BUFFER_SIZE];
		__shared__ unsigned char minR[SHARED_BUFFER_SIZE];
		__shared__ unsigned char maxR[SHARED_BUFFER_SIZE];
		__shared__ unsigned char minB[SHARED_BUFFER_SIZE];
		__shared__ unsigned char maxB[SHARED_BUFFER_SIZE];



		loadAndMinMax(	minG, maxG,
						minR, maxR,
						minB, maxB,
						params);

	}



	struct DemosaicingTest : testing::Test
	{


		DeviceBuffer<unsigned char> *img;
		DeviceBuffer<unsigned char> *img_res;
		unsigned char				*expected;
		int width = 960;
		int height = 540;

		S_IMG_PROC_KERNEL_PARAMS		params;
		S_LAUNCH_CONFIG					launchConfig;


		DemosaicingTest():width(960), height(540)
		{

			
			int n = width * height;
			

			img = new DeviceBuffer<unsigned char>(n);
			img_res = new DeviceBuffer<unsigned char>(n);
			expected = new unsigned char[n];
			unsigned char* buffer = new unsigned char[n];
			try {

				

				ifstream ifs("bayering_input.bin", ios::in | ios::binary);
				ifs.read((char*)buffer, n);
				ifs.close();
				memcpy(img->getHostData(), buffer, n);

				ifs = ifstream("bayering.bin", ios::in | ios::binary);
				ifs.read((char*)buffer, n);
				ifs.close();
				memcpy(expected, buffer, n);
			}	
			catch (const ifstream::failure& e) {
				cout << "Exception opening/reading file";
				return;
			}


			delete buffer;			
			img->set();

			launchConfig.block = dim3(32, 32);
			launchConfig.grid = dim3(1, 1);

			params.imageParameter.imageHeight = 540;
			params.imageParameter.imageWidth = 960;
			params.imageParameter.img_raw = img->getDeviceData();
			params.imageParameter.img = img_res->getDeviceData();
		}

		void runTest()
		{

			testDebayer << <launchConfig.grid, launchConfig.block >> > (params);
			cudaCheckError();

			cudaDeviceSynchronize();
			cudaCheckError();

			img_res->get();
			for (int i = 0; i < width*height; i++)
			{
				EXPECT_EQ(expected[i], img_res->getHostData()[i]);
			}
		}


		~DemosaicingTest()
		{
			delete img;
			delete img_res;
			delete expected;

		}

	};





	TEST_F(DemosaicingTest, test1)
	{
		runTest();
	};

}