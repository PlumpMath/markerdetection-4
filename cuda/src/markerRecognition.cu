#include "markerRecognition.cuh"
#include <istream>
#include <fstream>

namespace markerrecognition
{
	__device__  void encodeNumeric(unsigned int* p, int* indices, unsigned char* enc, const unsigned char* const marker, const S_IMG_PROC_KERNEL_PARAMS& params)
	{
		int in;
		int number = 0;
		unsigned char v = 0;
		for (int i = 0; i < 10; i++)
		{
			in = indices[i];
			if (enc[in] == 1)
			{
				v = ~(0x01 & marker[in]);
			}
			else
			{
				v = (0x01 & marker[in]);
			}
			if (0x01 & v)
			{				
				number |= 1 << (9-i);				
			}

		}
		*p = number;
	}

	__device__  void encodeByte(unsigned char* p, int* indices, unsigned char* enc, const unsigned char* const marker,const S_IMG_PROC_KERNEL_PARAMS& params)
	{
		int in;
		unsigned char number = 0;
		unsigned char v = 0;
		for (int i = 0; i < 8; i++)
		{
			in = indices[i];
			if (enc[in] == 1)
			{
				v = ~(0x01 & marker[in]);
			}
			else
			{
				v = (0x01 & marker[in]);
			}
			if (0x01 & v)
			{
				number |= 1 << (7 - i);
			}
		}
		*p = number;

	}

	__device__  void recognizeMarker(
		unsigned int* const intBuffer,
		unsigned char* const charBuffer,
		int* const maskIdBuffer,
		const unsigned char* const buffer,
		int* const markerId,
		int* const offsetA, 
		int* const offsetB,
		const S_IMG_PROC_KERNEL_PARAMS& params)
	{
		int* p_IntIndices;
		int* p_CharIndices;
		int maskId = 0;
		int threadId = threadIdx.y*blockDim.x + threadIdx.x;

		if (threadId == 0)
		{
			*markerId = INT_NAN;
			*offsetA = INT_NAN;
			*offsetB = INT_NAN;
		}

		if (threadId < 3)
		{
			intBuffer[threadId] = 0;
		}

		if (threadId < 4)
		{
			charBuffer[threadId] = 0;
		}



		if (threadId < 1)
		{
			if (buffer[172])
			{
				maskId |= 1 << 0;
			}
			if (buffer[171])
			{
				maskId |= 1 << 1;
			}
			if (buffer[170])
			{
				maskId |= 1 << 2;
			}
			maskId ^= 5;
			maskIdBuffer[0] = maskId;
		}

		__syncthreads();
		maskId = maskIdBuffer[0];
		if (maskId < 0 || maskId >= N_DEOCDER_MASKS)
		{
			return;
		}

		///
		if (threadId < 3)
		{

			switch (threadId)
			{
			case 0:
				p_IntIndices = params.markerParameter.indexNumeric0;
				break;
			case 1:
				p_IntIndices = params.markerParameter.indexNumeric1;
				break;
			case 2:
				p_IntIndices = params.markerParameter.indexNumeric2;
				break;
			default:
				p_IntIndices = NULL;
			};

			encodeNumeric(intBuffer + threadId,
				p_IntIndices,
				params.markerParameter.decodeMasks[maskId],
				buffer, params);
		}
		else if (threadId >= 3 && threadId < 7)
		{
			switch (threadId)
			{
			case 3:
				p_CharIndices = params.markerParameter.indexByte0;
				break;
			case 4:
				p_CharIndices = params.markerParameter.indexByte1;
				break;
			case 5:
				p_CharIndices = params.markerParameter.indexByte2;
				break;
			case 6:
				p_CharIndices = params.markerParameter.indexByte3;
				break;
			default:
				p_CharIndices = NULL;
			};

			if (p_CharIndices != NULL)
			{
				encodeByte(charBuffer + threadId - 3,
					p_CharIndices,
					params.markerParameter.decodeMasks[maskId],
					buffer, params);
			}

		}
		__syncthreads();
		if (threadId < 1)
		{

			//params.markerParameter.markerID[blockIdx.x]
			*markerId = 1000000 * intBuffer[0] +
				1000 * intBuffer[1] +
				intBuffer[2];
			if (*markerId < 0 || *markerId > 1000)
			{
				*markerId = INT_NAN;
			}

			//-48 ascii hack , just works for 9 offsets
			if (charBuffer[0] == '-')
			{
				//params.markerParameter.markerOffsetA[blockIdx.x]
				
				*offsetA = -(charBuffer[1] - 48);
			}
			else if (charBuffer[0] == '+')
			{
				*offsetA = charBuffer[1] - 48;
			}
			else
			{
				*offsetA = INT_NAN;
			}

			if (charBuffer[2] == '-')
			{
				*offsetB = -(charBuffer[3] - 48);
			}
			else if(charBuffer[2] == '+')
			{
				*offsetB = charBuffer[3] - 48;
			}
			else
			{
				*offsetB = INT_NAN;
			}



		}


	}


	__global__ void testMarkerRecognition(S_IMG_PROC_KERNEL_PARAMS params)
	{
		__shared__ unsigned int intBuffer[3];
		__shared__ unsigned char charBuffer[4];
		__shared__ int maskIdBuffer[1];


		int markerId = 0;
		int offsetA = 0;
		int offsetB =0;


		recognizeMarker(
			intBuffer,
			charBuffer,
			maskIdBuffer,
			params.qrCodeParameter.markerKorrigiert,
			&markerId,
			&offsetA,
			&offsetB,
			params);


		if (params.debug && threadIdx.x == 0 && threadIdx.y == 0)
		{
			params.markerParameter.markerID[blockIdx.x] = markerId;
			params.markerParameter.markerOffsetA[blockIdx.x] = offsetA;
			params.markerParameter.markerOffsetB[blockIdx.x] = offsetB;
		}
		

	}








	struct test_marker {
		unsigned char marker[21 * 21];
		int id;
		int a;
		int b;		
	};
	

	struct MarkerRecognitionTest : testing::Test
	{


		std::string fN = "markerTestData.bin";
		test_marker*					lTestMarker;
		int								nMarker;

		S_IMG_PROC_KERNEL_PARAMS		params;
		S_LAUNCH_CONFIG					launchConfig;
		DeviceBuffer<unsigned char>		*bufferMarker;		
		MARKER							*marker;
		


		MarkerRecognitionTest():lTestMarker(NULL)
		{
		
			try {

				std::ifstream is("markerTestData.bin", std::fstream::binary);
				
				
				// get length of file:				
				is.seekg(0, ios::beg);
				
				is.read((char*)&nMarker, 4);
				lTestMarker = new test_marker[nMarker];
				for (int i = 0; i < nMarker; i++)
				{
					is.read((char*)&lTestMarker[i].id, 4);					
					is.read((char*)&lTestMarker[i].a, 4);					
					is.read((char*)&lTestMarker[i].b, 4);
					for (int j = 0; j < 21 * 21; j++)
					{
						is.read((char*)&lTestMarker[i].marker[j], 1);
					}
				}
				is.get();
				if (!is.eof())
				{
					std::cout << "Exception opening/reading file";
				}
				is.close();
			}
			catch (const ifstream::failure& e) {
				cout << "Exception opening/reading file";
				return;
			}

			bufferMarker = new DeviceBuffer<unsigned char>(21*21);
			marker = new MARKER{ 1 };


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

			params.qrCodeParameter.markerKorrigiert = bufferMarker->getDeviceData();

			params.markerParameter.markerID = marker->id->getDeviceData();
			params.markerParameter.markerOffsetA = marker->a->getDeviceData();
			params.markerParameter.markerOffsetB = marker->b->getDeviceData();

			launchConfig.grid = dim3(1);
			launchConfig.block = dim3(7);
		
		}

		void runTest()
		{

		}

		~MarkerRecognitionTest()
		{
			delete bufferMarker;
			if (lTestMarker != NULL)			
				delete[] lTestMarker;
		}
	};

	
	TEST(Marker, Test1)
	{
		S_QR_MARKER test;

		float a[2];
		float b[2];
		float c[2];
		int ai = 1;
		int bi = 1;

		test.getMarkerFromAB(ai, bi, a, b, c);

		EXPECT_FLOAT_EQ(a[0], 20.0);
		EXPECT_FLOAT_EQ(a[1], 20.0);

		EXPECT_FLOAT_EQ(b[0], 5.0);
		EXPECT_FLOAT_EQ(b[1], 20.0);

		EXPECT_FLOAT_EQ(c[0], 20.0);
		EXPECT_FLOAT_EQ(c[1], 5.0);

	}


	TEST(Marker, Test2)
	{
		S_QR_MARKER test;

		float a[2];
		float b[2];
		float c[2];
		int ai = 0;
		int bi = 0;

		test.getMarkerFromAB(ai, bi, a, b, c);

		EXPECT_TRUE(isnan(a[0]));
		EXPECT_TRUE(isnan(a[1]));

		EXPECT_TRUE(isnan(b[0]));
		EXPECT_TRUE(isnan(b[1]));

		EXPECT_TRUE(isnan(c[0]));
		EXPECT_TRUE(isnan(c[1]));

	}

	TEST(Marker, Test3)
	{
		S_QR_MARKER test;

		float a[2];
		float b[2];
		float c[2];
		int ai = -3;
		int bi = 1;

		test.getMarkerFromAB(ai, bi, a, b, c);

		EXPECT_TRUE(isnan(a[0]));
		EXPECT_TRUE(isnan(a[1]));

		EXPECT_TRUE(isnan(b[0]));
		EXPECT_TRUE(isnan(b[1]));

		EXPECT_TRUE(isnan(c[0]));
		EXPECT_TRUE(isnan(c[1]));

	}

	TEST(Marker, Test4)
	{
		S_QR_MARKER test;


		float a[2];
		float b[2];
		float c[2];
		int ai = -2;
		int bi = -2;

		test.getMarkerFromAB(ai, bi, a, b, c);

		EXPECT_FLOAT_EQ(a[0], -30.0);
		EXPECT_FLOAT_EQ(a[1], -30.0);

		EXPECT_FLOAT_EQ(b[0], -45.0);
		EXPECT_FLOAT_EQ(b[1], -30.0);

		EXPECT_FLOAT_EQ(c[0], -30.0);
		EXPECT_FLOAT_EQ(c[1], -45.0);

	}

	TEST(Marker, Test5)
	{
		S_QR_MARKER test;

		float a[2];
		float b[2];
		float c[2];
		int ai = 1;
		int bi = -3;

		test.getMarkerFromAB(ai, bi, a, b, c);

		EXPECT_TRUE(isnan(a[0]));
		EXPECT_TRUE(isnan(a[1]));

		EXPECT_TRUE(isnan(b[0]));
		EXPECT_TRUE(isnan(b[1]));

		EXPECT_TRUE(isnan(c[0]));
		EXPECT_TRUE(isnan(c[1]));

	}


	TEST_F(MarkerRecognitionTest, markerRecognition)
	{
		for (int i = 7; i < nMarker; i++)
		{
			bufferMarker->set(lTestMarker[i].marker);

			//cudaDeviceSynchronize();
			//cudaCheckError();

			//run cuda
			testMarkerRecognition << <launchConfig.grid, launchConfig.block >> > (params);
			cudaCheckError();

			cudaDeviceSynchronize();
			cudaCheckError();

			marker->get();

			//cudaDeviceSynchronize();
			//cudaCheckError();
			
			if (marker->id->getHostData()[0] != lTestMarker[i].id ||
				marker->a->getHostData()[0] != lTestMarker[i].a ||
				marker->b->getHostData()[0] != lTestMarker[i].b)
			{
				std::cout << "run marker test nr: " << i << endl;
			}
			EXPECT_EQ(marker->id->getHostData()[0], lTestMarker[i].id);
			EXPECT_EQ(marker->a->getHostData()[0], lTestMarker[i].a);
			EXPECT_EQ(marker->b->getHostData()[0], lTestMarker[i].b);

		}
	}




}
