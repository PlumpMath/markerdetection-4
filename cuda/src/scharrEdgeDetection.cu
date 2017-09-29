#include "scharrEdgeDetection.cuh"


#include <istream>
#include <fstream>


namespace scharredgedetection
{
	using namespace std;
	struct ScharredgedetectionTest : testing::Test
	{


		DeviceBuffer<unsigned char> *img;
		DeviceBuffer<unsigned char> *img_res;
		unsigned char				*expected;
		int width = 960;
		int height = 540;

		S_IMG_PROC_KERNEL_PARAMS		params;
		S_LAUNCH_CONFIG					launchConfig;


		ScharredgedetectionTest() :width(960), height(540)
		{


			int n = width * height;


			img = new DeviceBuffer<unsigned char>(n);
			img_res = new DeviceBuffer<unsigned char>(n);
			expected = new unsigned char[n];
			unsigned char* buffer = new unsigned char[n];
			try {



				ifstream ifs("scharr_input.bin", ios::in | ios::binary);
				ifs.read((char*)buffer, n);
				ifs.close();
				memcpy(img->getHostData(), buffer, n);

				ifs = ifstream("scharr.bin", ios::in | ios::binary);
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

			//testDebayer << <launchConfig.grid, launchConfig.block >> > (params);
			//cudaCheckError();

			cudaDeviceSynchronize();
			cudaCheckError();

			img_res->get();
			for (int i = 0; i < width*height; i++)
			{
				EXPECT_EQ(expected[i], img_res->getHostData()[i]);
			}
		}


		~ScharredgedetectionTest()
		{
			delete img;
			delete img_res;
			delete expected;

		}

	};

}