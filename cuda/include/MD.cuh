#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "struct.cuh"
#include <opencv2\opencv.hpp>
using namespace cv;


namespace markerdetection
{
	enum E_CAMERA_TYPE
	{
		E_CON,
		RAYTRIX
	};

	enum E_RESOLUTION
	{		
		E_1280X720,		
		E_960X540,
	};


	class MD
	{

	public:

		MD(E_RESOLUTION res, E_CAMERA_TYPE type, 
			const double const *C, const double const *H, const double const *kk,
			map<int, cv::Vec<float, 3>> map, 
			bool debug);

		void processData(const unsigned char* const p_data);

		virtual ~MD();

	private:
		double robotPose[3];

		void step0();
		void step1();
		void step2();
		void step3();
		void step4();

		void step4_host();

		void initParameters();
		void overlay(const unsigned char* mask, int r, int g, int b, float alpha, unsigned char maskValue);
		void projectPointsFromGround2ImagePlane(double x, double y, double& u, double& v);
		void drawCenter(int* x, int* y, int n);
		void drawCenter(float* x, float* y, int n);
		void drawQRText();
		void drawCoordinateCross(float* Ax, float* Ay, float* Bx, float* By, float* Cx, float* Cy, int n);
		void reprojectMarker();

		bool m_debug;

		Mat img;
		Mat img_colored;
		Mat marker_img;
		Mat markerresize;
		
		S_IMG_PROC_KERNEL_PARAMS	params;
		S_LAUNCH_CONFIG				launchConfig_step0;
		S_LAUNCH_CONFIG				launchConfig_step1;
		S_LAUNCH_CONFIG				launchConfig_step2;
		S_LAUNCH_CONFIG				launchConfig_step3;
		S_LAUNCH_CONFIG				launchConfig_step4;

		S_MARKER_DIM				markerDim{ 15.0,21.0 };
		S_ROBOT_PARAMER				robotParams{ 125, 90, 10 };

		


		DeviceBuffer<unsigned char> *mainPattern;

		DeviceBuffer<unsigned char> *img_raw;
		DeviceBuffer<unsigned char> *img_dev;
		DeviceBuffer<unsigned char> *imgres_dev;
		DeviceBuffer<unsigned char> *maskres_dev;

		DeviceBuffer<int>* index_x;
		DeviceBuffer<int>* index_y;
		DeviceBuffer<unsigned char>* imageMask;


		DeviceBuffer<unsigned int> *nCentroids;
		DeviceBuffer<int>*			centroidY;
		DeviceBuffer<int>*			centroidX;
		DeviceBuffer<unsigned char>* centroidP;

		
		DeviceBuffer<float> *				Ax;
		DeviceBuffer<float> *				Ay;

		DeviceBuffer<float> *				Bx;
		DeviceBuffer<float> *				By;

		DeviceBuffer<float> *				Cx;
		DeviceBuffer<float> *				Cy;

		DeviceBuffer<unsigned int> *		Cn;

		DeviceBuffer<unsigned char>*		binaryMarker;
		MARKER*								marker;
		S_QR_MARKER							qrMarker;

		DeviceBuffer<double>*				initalGuess;
		DeviceBuffer<double>*				initalGuessImage;

		DeviceBuffer<double>*				optimalTransform;
		DeviceBuffer<double >*				optimalCosts;

		Mat rotResult;
		Mat transResult;
		Mat camera2Robot;

		map<int, cv::Vec<float, 3>>			_map;





	};




}