#include "distortImage.cuh"


#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <sstream>
#include <exception>

namespace distortimage
{
	using namespace cv;
	using namespace std;


		//1 2 1
		//2 4 2
		//1 2 1



	__device__ void debayer(const unsigned char const* image_buffer_raw, const unsigned char const* mask_buffer, unsigned char* image_buffer, S_IMG_PROC_KERNEL_PARAMS& params)
	{

		int bufferId;
		int bufferIdInner;

		
		int maskId;
		int realBufferThreadx;
		int realBufferThready;

		int realBufferThreadxInner;
		int realBufferThreadyInner;

		int sumMask = 0;
		int sumValue = 0;
		int tmp;
		int realImageThreadx;
		int realImageThready;
		int ci;
		int ri;
		unsigned char v;
		for (int r = 0; r < params.loadAndStoreParams.threadFaktorY; r++)
		{
			realBufferThready = params.loadAndStoreParams.threadFaktorY*threadIdx.y + r;
			
			for (int c = 0; c < params.loadAndStoreParams.threadFaktorX; c++)
			{
				realBufferThreadx = params.loadAndStoreParams.threadFaktorX*threadIdx.x + c;
				bufferId = realBufferThready*params.loadAndStoreParams.bufferWidth + realBufferThreadx;

				//if (mask_buffer[bufferId] == 0)
				//{
				//	image_buffer[bufferId] = 0;
				//	continue;
				//}

				maskId = 0;
				sumMask = 0;
				sumValue = 0;
				realImageThreadx = params.loadAndStoreParams.threadFaktorX*threadIdx.x + params.loadAndStoreParams.maskOffset + c + blockDim.x*blockIdx.x;
				realImageThready = params.loadAndStoreParams.threadFaktorY*threadIdx.y + params.loadAndStoreParams.maskOffset + r + blockDim.y*blockIdx.y;

				if ((realImageThreadx+1 )% 2 == 0 && (realImageThready+1) % 2 == 0)
				{
					//upper right corner
					ci = 0;
					ri = 0;

					realBufferThreadxInner = realBufferThreadx + ci;
					realBufferThreadyInner = realBufferThready + ri;
					if (realBufferThreadxInner >= 0 && realBufferThreadxInner < params.loadAndStoreParams.bufferWidth &&
						realBufferThreadyInner >= 0 && realBufferThreadyInner < params.loadAndStoreParams.bufferHeight)
					{

						bufferIdInner = realBufferThreadyInner*params.loadAndStoreParams.bufferWidth + realBufferThreadxInner;
						sumValue += image_buffer_raw[bufferIdInner];
						sumMask++;
					}

					ci = -1;
					ri = -1;

					realBufferThreadxInner = realBufferThreadx + ci;
					realBufferThreadyInner = realBufferThready + ri;
					if (realBufferThreadxInner >= 0 && realBufferThreadxInner < params.loadAndStoreParams.bufferWidth &&
						realBufferThreadyInner >= 0 && realBufferThreadyInner < params.loadAndStoreParams.bufferHeight)
					{

						bufferIdInner = realBufferThreadyInner*params.loadAndStoreParams.bufferWidth + realBufferThreadxInner;
						sumValue += image_buffer_raw[bufferIdInner];
						sumMask++;
					}

					if (sumMask > 0)
					{
						v = sumValue / sumMask;
					}
					else
					{
						v = 0;
					}

					ci = 0;
					ri = -1;

					realBufferThreadxInner = realBufferThreadx + ci;
					realBufferThreadyInner = realBufferThready + ri;
					if (realBufferThreadxInner >= 0 && realBufferThreadxInner < params.loadAndStoreParams.bufferWidth &&
						realBufferThreadyInner >= 0 && realBufferThreadyInner < params.loadAndStoreParams.bufferHeight)
					{

						bufferIdInner = realBufferThreadyInner*params.loadAndStoreParams.bufferWidth + realBufferThreadxInner;
						image_buffer[bufferIdInner] = v;
						
					}

					ci = -1;
					ri = 0;

					realBufferThreadxInner = realBufferThreadx + ci;
					realBufferThreadyInner = realBufferThready + ri;
					if (realBufferThreadxInner >= 0 && realBufferThreadxInner < params.loadAndStoreParams.bufferWidth &&
						realBufferThreadyInner >= 0 && realBufferThreadyInner < params.loadAndStoreParams.bufferHeight)
					{

						bufferIdInner = realBufferThreadyInner*params.loadAndStoreParams.bufferWidth + realBufferThreadxInner;
						image_buffer[bufferIdInner] = v;

					}

					ci = 0;
					ri = 0;

					realBufferThreadxInner = realBufferThreadx + ci;
					realBufferThreadyInner = realBufferThready + ri;
					if (realBufferThreadxInner >= 0 && realBufferThreadxInner < params.loadAndStoreParams.bufferWidth &&
						realBufferThreadyInner >= 0 && realBufferThreadyInner < params.loadAndStoreParams.bufferHeight)
					{

						bufferIdInner = realBufferThreadyInner*params.loadAndStoreParams.bufferWidth + realBufferThreadxInner;
						image_buffer[bufferIdInner] = v;

					}

					ci = -1;
					ri = -1;

					realBufferThreadxInner = realBufferThreadx + ci;
					realBufferThreadyInner = realBufferThready + ri;
					if (realBufferThreadxInner >= 0 && realBufferThreadxInner < params.loadAndStoreParams.bufferWidth &&
						realBufferThreadyInner >= 0 && realBufferThreadyInner < params.loadAndStoreParams.bufferHeight)
					{

						bufferIdInner = realBufferThreadyInner*params.loadAndStoreParams.bufferWidth + realBufferThreadxInner;
						image_buffer[bufferIdInner] = v;

					}
					//image_buffer[bufferId] = 0;
								

				}			
				else
				{
					//image_buffer[bufferId] = image_buffer_raw[bufferId];
				}


				//for (int ri = -1; ri <= 1; ri++)
				//{					
				//	for (int ci = -1; ci <= 1; ci++)
				//	{

				//		realBufferThreadxInner = realBufferThreadx + ci;
				//		realBufferThreadyInner = realBufferThready + ri;
				//		

				//		if (realBufferThreadxInner >= 0 && realBufferThreadxInner < params.loadAndStoreParams.bufferWidth &&
				//			realBufferThreadyInner >= 0 && realBufferThreadyInner < params.loadAndStoreParams.bufferHeight)
				//		{

				//			bufferIdInner = realBufferThreadyInner*params.loadAndStoreParams.bufferWidth + realBufferThreadxInner;
				//			
				//			
				//			sumValue += (((int)image_buffer_raw[bufferIdInner]) * ((int)params.maskParameter.debayerBatter[maskId]));
				//			sumMask += params.maskParameter.debayerBatter[maskId];

				//			//if (mask_buffer[bufferId])
				//			//{
				//			//	sumValue += image_buffer[bufferId] * params.maskParameter.debayerBatter[maskId];
				//			//	sumMask += params.maskParameter.debayerBatter[maskId];
				//			//}
				//		}

				//		maskId++;
				//	}
				//}

				//bufferId = realBufferThready*params.loadAndStoreParams.bufferWidth + realBufferThreadx;				
				//image_buffer[bufferId] = image_buffer_raw[bufferId];
				//image_buffer[bufferId] = sumValue / sumMask;
				//if (params.maskParameter.debayer && sumMask > 0)
				//{
				//	image_buffer[bufferId] = sumValue / sumMask;
				//}
				//else
				//{
				//	image_buffer[bufferId] = image_buffer_raw[bufferId];
				//}
			}
				
			
		}



	}

	__device__ void undistort(const unsigned char* mask, const int* mapx, const int* mapy, 
		const unsigned char* img, unsigned char* res, S_IMG_PROC_KERNEL_PARAMS& params)
	{
		int realImageThreadx = threadIdx.x + blockDim.x*blockIdx.x;
		int realImageThready = threadIdx.y + blockDim.y*blockIdx.y;
		int imageId = params.imageParameter.imageWidth*realImageThready + realImageThreadx;
		int x, y;

		if (mask[imageId])
		{
			x = mapx[imageId];
			y = mapy[imageId];
			res[imageId] = img[y*params.imageParameter.imageWidth + x];
		}

	}

	__global__ void testundistortKernel(S_IMG_PROC_KERNEL_PARAMS params)
	{
		undistort(params.undistortParamer.mask,
			params.undistortParamer.mapx,
			params.undistortParamer.mapy,
			params.undistortParamer.img,
			params.undistortParamer.resimg, params);
	}

	void initMasksAndMap(const double* Cin, const double* Hin, const double* kkin, 
		int resx, int resy, 
		std::vector<std::vector<double>> holeContour,
		DeviceBuffer<int>* index_x, DeviceBuffer<int>* index_y, 
		DeviceBuffer<unsigned char>* mask)
	{

		Mat C(N_ELMENTS_C_COL, N_ELMENTS_C_ROW, CV_64FC1);
		memcpy(C.data, Cin, N_ELMENTS_C_COL* N_ELMENTS_C_ROW * sizeof(double));

		Mat H(N_ELMENTS_H_COL, N_ELMENTS_H_ROW, CV_64FC1);
		memcpy(H.data, Hin, N_ELMENTS_H_COL* N_ELMENTS_H_ROW * sizeof(double));

		Mat kk(N_ELMENTS_kk_COL, N_ELMENTS_kk_ROW, CV_64FC1);
		memcpy(kk.data, kkin, N_ELMENTS_kk_COL* N_ELMENTS_kk_ROW * sizeof(double));
				
		Mat mapx, mapy;
		initUndistortRectifyMap(C, kk, Mat(), C, Size(resx, resy), CV_32FC1, mapx, mapy);

		float index_r, float index_c;
		int round_r, round_c;
		for (int r = 0; r < resy; r++)
		{
			for (int c = 0; c < resx; c++)
			{
				index_r = mapy.at<float>(r, c);
				index_c = mapx.at<float>(r, c);
				round_r = std::floor(index_r);
				round_c = std::floor(index_c);

				if (round_r >= 0 && round_c >= 0 &&
					round_r < resy && round_c < resx)
				{
					index_x->getHostData()[r * resx + c] = round_c;
					index_y->getHostData()[r * resx + c] = round_r;
				}
			}
		}

		Mat rot(Size(3, 3), CV_64FC1);
		Mat trans(Size(1, 3), CV_64FC1);
		for (int r = 0; r < rot.rows; r++)
		{
			for (int c = 0; c < rot.cols; c++)
			{
				rot.at<double>(r, c) = H.at<double>(r, c);
			}
		}
		trans.at<double>(0) = H.at<double>(0, 3);
		trans.at<double>(1) = H.at<double>(1, 3);
		trans.at<double>(2) = H.at<double>(2, 3);
		Mat rot_vec;
		Rodrigues(rot, rot_vec);

		
		vector<Vec<double, 3>> points;
		for (int i = 0; i < holeContour.size(); i++)
		{
			Vec<double, 3> vec;
			vec.val[0] = holeContour[i][0];
			vec.val[1] = holeContour[i][1];
			vec.val[2] = 0;
			points.push_back(vec);
		}
		Mat holeContour2d;
		projectPoints(points, rot_vec, trans, C, Mat(), holeContour2d);
		
		vector<Point2f> vert;
		for (int r = 0; r < holeContour2d.rows; r++)
		{
			Vec<double, 2> t = holeContour2d.at<Vec<double, 2>>(r);
			Point2f p(t.val[0], t.val[1]);
			vert.push_back(p);
		}
		


		for (int r = 0; r < resy; r++)
		{
			for (int c = 0; c < resx; c++)
			{
				if (pointPolygonTest(vert, Point2f(c, r), false) > 0)
				{
					mask->getHostData()[r*resx + c] = 1;
				}				
			}
		}

		index_x->set();
		index_y->set();
		mask->set();
		

	}


	struct UndistortImageTest : testing::Test
	{




		S_IMG_PROC_KERNEL_PARAMS		params;
		S_LAUNCH_CONFIG					launchConfig;


		DeviceBuffer<unsigned char> *image;
		DeviceBuffer<unsigned char> *imageundist;
		DeviceBuffer<unsigned char> *imageundistres;
		DeviceBuffer<unsigned char> *mask;


		DeviceBuffer<int> *index_x;
		DeviceBuffer<int> *index_y;

		double C[N_ELMENTS_C_ROW*N_ELMENTS_C_COL];
		double H[N_ELMENTS_H_ROW*N_ELMENTS_H_COL];
		double kk[N_ELMENTS_kk_ROW*N_ELMENTS_kk_COL];

		int resx;
		int resy;

		std::vector<std::vector<double>> holeContour;

		UndistortImageTest()
		{





			Mat img, imgun;
			img = imread("distorted.png", CV_LOAD_IMAGE_GRAYSCALE);   
			imgun = imread("distorted.png", CV_LOAD_IMAGE_GRAYSCALE);

			if (!img.data || !imgun.data)
			{				
				throw exception("Could not open or find the image");
			}
			resx = img.cols;
			resy = img.rows;

			image = new DeviceBuffer<unsigned char>(resx * resy);
			imageundist = new DeviceBuffer<unsigned char>(resx * resy);
			imageundistres = new DeviceBuffer<unsigned char>(resx * resy);

			mask = new DeviceBuffer<unsigned char>(resx * resy);

			index_x = new DeviceBuffer<int>(resx * resy);
			index_y = new DeviceBuffer<int>(resx * resy);

			memcpy(image->getHostData(), img.data, resx * resy);
			memcpy(imageundistres->getHostData(), imgun.data, resx * resy);
			image->set();


			Mat Cmat, kkmat, Hmat;
			try {
				FileStorage fs("test.yml", FileStorage::READ);
				fs["camera_matrix"] >> Cmat;
				fs["distortion_coefficients"] >> kkmat;
				fs["camera_extrinsics"] >> Hmat;				
				fs.release();
			}
			catch (...) {
				throw runtime_error("Error reading Camera matrix");
			}
			
			memcpy(C, Cmat.data,  N_ELMENTS_C_ROW*N_ELMENTS_C_COL * sizeof(double));
			memcpy(H, Hmat.data, N_ELMENTS_H_ROW*N_ELMENTS_H_COL * sizeof(double));
			memcpy(kk, kkmat.data, N_ELMENTS_kk_ROW*N_ELMENTS_kk_COL * sizeof(double));

			double dimx = 125;
			double dimy = 90;
			
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


			
			params.imageParameter.imageHeight = resy;
			params.imageParameter.imageWidth = resx;

			params.undistortParamer.img = image->getDeviceData();
			params.undistortParamer.resimg = imageundist->getDeviceData();
			params.undistortParamer.mapx = index_x->getDeviceData();
			params.undistortParamer.mapy = index_y->getDeviceData();
			params.undistortParamer.mask = mask->getDeviceData();



			launchConfig.grid = dim3(40, 45);
			launchConfig.block = dim3(32, 16);
			
		}

		void runTest()
		{

			initMasksAndMap(C, H, kk,
				resx, resy,
				holeContour,
				index_x, index_y,
				mask);

			
			testundistortKernel << <launchConfig.grid, launchConfig.block >> > (params);
			cudaCheckError();

			cudaDeviceSynchronize();
			cudaCheckError();

			imageundist->get();

			//Mat img(resy, resx, CV_8UC1);
			//memcpy(img.data, imageundist->getHostData(), resx*resy);
			//namedWindow("win1", WINDOW_AUTOSIZE);
			//imshow("win1", img);
			//waitKey(0);



		}

		~UndistortImageTest()
		{
			delete image;
			delete imageundist;
			delete imageundistres;
			delete mask;
			delete index_x;
			delete index_y;
		}
	};

	//TEST_F(UndistortImageTest, transformSolver)
	//{
	//	runTest();
	//}
}