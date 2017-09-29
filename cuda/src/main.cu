#include <gtest/gtest.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "struct.cuh"
#include <iostream>

#include "EdgeDetectionKernel.cuh"

using namespace std;




#include <opencv2/core/core.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;

void projectIntoImagePlane(const double* C, const double* HCar2Image, const double* car, double * u, double* v)
{




	double xcar = car[0];
	double ycar = car[1];
	double zcar = car[2];




	double c1 = C[0];
	double c2 = C[2];
	double c3 = C[4];
	double c4 = C[5];

	double h0 = HCar2Image[0];
	double h1 = HCar2Image[1];
	double h2 = HCar2Image[2];
	double h3 = HCar2Image[3];

	double h4 = HCar2Image[4];
	double h5 = HCar2Image[5];
	double h6 = HCar2Image[6];
	double h7 = HCar2Image[7];

	double h8 = HCar2Image[8];
	double h9 = HCar2Image[9];
	double h10 = HCar2Image[10];
	double h11 = HCar2Image[11];



	double xc = h3 + h0*xcar + h1*ycar + h2*zcar;
	double yc = h7 + h4*xcar + h5*ycar + h6*zcar;
	double zc = h11 + h8*xcar + h9*ycar + h10*zcar;


	double x = c1 * xc + c2 * zc;
	double y = c3*  yc + c4 * zc;


	double imx = x / zc;
	double imy = y / zc;

	*u = imx;
	*v = imy;

}

 void invert4_device(const double const *mat, double *dst)
{
	double tmp[12]; /* temp array for pairs */
	double src[16]; /* array of transpose source matrix */
	double det; /* determinant */
			   /* transpose matrix */
	for (int i = 0; i < 4; i++) {
		src[i] = mat[i * 4];
		src[i + 4] = mat[i * 4 + 1];
		src[i + 8] = mat[i * 4 + 2];
		src[i + 12] = mat[i * 4 + 3];
	}
	/* calculate pairs for first 8 elements (cofactors) */
	tmp[0] = src[10] * src[15];
	tmp[1] = src[11] * src[14];
	tmp[2] = src[9] * src[15];
	tmp[3] = src[11] * src[13];
	tmp[4] = src[9] * src[14];
	tmp[5] = src[10] * src[13];
	tmp[6] = src[8] * src[15];
	tmp[7] = src[11] * src[12];
	tmp[8] = src[8] * src[14];
	tmp[9] = src[10] * src[12];
	tmp[10] = src[8] * src[13];
	tmp[11] = src[9] * src[12];
	/* calculate first 8 elements (cofactors) */
	dst[0] = tmp[0] * src[5] + tmp[3] * src[6] + tmp[4] * src[7];
	dst[0] -= tmp[1] * src[5] + tmp[2] * src[6] + tmp[5] * src[7];
	dst[1] = tmp[1] * src[4] + tmp[6] * src[6] + tmp[9] * src[7];
	dst[1] -= tmp[0] * src[4] + tmp[7] * src[6] + tmp[8] * src[7];
	dst[2] = tmp[2] * src[4] + tmp[7] * src[5] + tmp[10] * src[7];
	dst[2] -= tmp[3] * src[4] + tmp[6] * src[5] + tmp[11] * src[7];
	dst[3] = tmp[5] * src[4] + tmp[8] * src[5] + tmp[11] * src[6];
	dst[3] -= tmp[4] * src[4] + tmp[9] * src[5] + tmp[10] * src[6];
	dst[4] = tmp[1] * src[1] + tmp[2] * src[2] + tmp[5] * src[3];
	dst[4] -= tmp[0] * src[1] + tmp[3] * src[2] + tmp[4] * src[3];
	dst[5] = tmp[0] * src[0] + tmp[7] * src[2] + tmp[8] * src[3];
	dst[5] -= tmp[1] * src[0] + tmp[6] * src[2] + tmp[9] * src[3];
	dst[6] = tmp[3] * src[0] + tmp[6] * src[1] + tmp[11] * src[3];
	dst[6] -= tmp[2] * src[0] + tmp[7] * src[1] + tmp[10] * src[3];
	dst[7] = tmp[4] * src[0] + tmp[9] * src[1] + tmp[10] * src[2];
	dst[7] -= tmp[5] * src[0] + tmp[8] * src[1] + tmp[11] * src[2];
	/* calculate pairs for second 8 elements (cofactors) */
	tmp[0] = src[2] * src[7];
	tmp[1] = src[3] * src[6];
	tmp[2] = src[1] * src[7];
	tmp[3] = src[3] * src[5];
	tmp[4] = src[1] * src[6];
	tmp[5] = src[2] * src[5];

	tmp[6] = src[0] * src[7];
	tmp[7] = src[3] * src[4];
	tmp[8] = src[0] * src[6];
	tmp[9] = src[2] * src[4];
	tmp[10] = src[0] * src[5];
	tmp[11] = src[1] * src[4];
	/* calculate second 8 elements (cofactors) */
	dst[8] = tmp[0] * src[13] + tmp[3] * src[14] + tmp[4] * src[15];
	dst[8] -= tmp[1] * src[13] + tmp[2] * src[14] + tmp[5] * src[15];
	dst[9] = tmp[1] * src[12] + tmp[6] * src[14] + tmp[9] * src[15];
	dst[9] -= tmp[0] * src[12] + tmp[7] * src[14] + tmp[8] * src[15];
	dst[10] = tmp[2] * src[12] + tmp[7] * src[13] + tmp[10] * src[15];
	dst[10] -= tmp[3] * src[12] + tmp[6] * src[13] + tmp[11] * src[15];
	dst[11] = tmp[5] * src[12] + tmp[8] * src[13] + tmp[11] * src[14];
	dst[11] -= tmp[4] * src[12] + tmp[9] * src[13] + tmp[10] * src[14];
	dst[12] = tmp[2] * src[10] + tmp[5] * src[11] + tmp[1] * src[9];
	dst[12] -= tmp[4] * src[11] + tmp[0] * src[9] + tmp[3] * src[10];
	dst[13] = tmp[8] * src[11] + tmp[0] * src[8] + tmp[7] * src[10];
	dst[13] -= tmp[6] * src[10] + tmp[9] * src[11] + tmp[1] * src[8];
	dst[14] = tmp[6] * src[9] + tmp[11] * src[11] + tmp[3] * src[8];
	dst[14] -= tmp[10] * src[11] + tmp[2] * src[8] + tmp[7] * src[9];
	dst[15] = tmp[10] * src[10] + tmp[4] * src[8] + tmp[9] * src[9];
	dst[15] -= tmp[8] * src[9] + tmp[11] * src[10] + tmp[5] * src[8];
	/* calculate determinant */
	det = src[0] * dst[0] + src[1] * dst[1] + src[2] * dst[2] + src[3] * dst[3];
	/* calculate matrix inverse */
	det = 1 / det;
	for (int j = 0; j < 16; j++)
		dst[j] *= det;
}


int main2(int argc, char** argv)
{



	Mat image;
	image = imread("r1.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file

	if (!image.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	cout << image.cols << "\t" << image.rows << endl;

	namedWindow("win1", WINDOW_AUTOSIZE);// Create a window for display.
	namedWindow("win2", WINDOW_AUTOSIZE);// Create a window for display.
	//imshow("Display window", image);                   // Show our image inside it.


	Mat m_C, m_distortion_coefficients, m_C_extrinics, m_C_hom;
	try {
		FileStorage fs("econ.yml", FileStorage::READ);
		fs["camera_matrix"] >> m_C;
		fs["distortion_coefficients"] >> m_distortion_coefficients;
		fs["camera_extrinsics"] >> m_C_extrinics;
		fs["camera_matrix_hom"] >> m_C_hom;
		fs.release();
		cout << "Camera_Matrix" << m_C << endl;
		cout << "Distortion_Matrix" << m_distortion_coefficients << endl; 
		cout << "Hom" << m_C_hom << endl;
	}
	catch (...) {
		throw runtime_error("Error reading Camera matrix");
	}

	cout << m_C << endl;
	Mat imageUndistorted, Cundistorted,mapx, mapy;
	undistort(image, imageUndistorted, m_C, m_distortion_coefficients);
	initUndistortRectifyMap(m_C, m_distortion_coefficients,Mat(), m_C, Size(1280, 720),  CV_32FC1, mapx, mapy);
	
	cout << imageUndistorted.cols << "\t" << imageUndistorted.rows << endl;
	cout << m_C << endl;
	//imshow("win1", image);
	//imshow("win2", imageUndistorted);
	//waitKey(1);
	
	Mat correctedImage = Mat::zeros(Size(image.cols, image.rows), CV_8UC1);
	float index_r, index_c;
	int round_r, round_c;
	Mat imagegray;
	unsigned char val;
	cvtColor(image, imagegray, cv::COLOR_RGB2GRAY);
	for (int r = 0; r < correctedImage.rows; r++)
	{
		for (int c = 0; c < correctedImage.cols; c++)
		{
			index_r = mapy.at<float>(r, c);
			index_c = mapx.at<float>(r, c);
			round_r = std::floor(index_r);
			round_c = std::floor(index_c);
			if (round_r >= 0 && round_c >= 0 &&
				round_r < 720 && round_c < 1280)
			{
				val = imagegray.at < unsigned char>(round_r, round_c);
				correctedImage.at<unsigned char>(r, c) = val;
			}
		}
	}

	//imshow("win2", imagegray);
	//imshow("win2", correctedImage);
	////imwrite("distorted.jpg", imageUndistorted);
	////imwrite("undistorted.jpg", imageUndistorted);
	//waitKey(0);






	//Mat testMat1(4,4, CV_32FC1);
	//testMat1.at<float>(0, 0) = 1;
	//testMat1.at<float>(0, 0) = 0;
	//testMat1.at<float>(0, 0) = 0;
	//testMat1.at<float>(0, 0) = 0;



	Mat testMat2(4, 1, CV_32FC1);
	//testMat2.at<float>(0, 0) = 1;
	//testMat2.at<float>(1, 0) = 2;
	//testMat2.at<float>(2, 0) = 3;
	//cout << "v" << m_C_extrinics << endl;

	cout << m_C_extrinics.inv() << endl;

	//m_C_extrinics = Mat::eye(4, 4, CV_64F);
	//m_C_extrinics.at<double>(0, 3) = 0.04;

	Mat cvInvert(m_C_extrinics);
	//cvInvert = cvInvert.inv();

	

	double matDataInvert[16];	
	double matData[16];
	for (int i = 0; i < cvInvert.rows; ++i)
		for (int j = 0; j < cvInvert.cols; ++j)
		{
			matDataInvert[i * 4 + j] = cvInvert.at<double>(i, j);
			matData[i * 4 + j] = m_C_extrinics.at<double>(i, j);
		}
	double test[16];
	invert4_device(matData, test);

	cout << std::fixed << std::setw(11) << std::setprecision(6);
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			cout << test[i * 4 + j] << "\t";
		}
		cout << endl;

	}

	double C[9];
	for (int i = 0; i < m_C.rows; ++i)
		for (int j = 0; j < m_C.cols; ++j)
		{
			C[i * 3 + j] = m_C.at<double>(i, j);
		}


	//matDataInvert[3] /= 1000;
	//matDataInvert[7] /= 1000;
	//matDataInvert[11] /= 1000;

	//matData[3] /= 1000;
	//matData[7] /= 1000;
	//matData[11] /= 1000;


	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			cout << C[i*3+j] << "\t";
		}				
		cout << endl;
		
	}

	double dimx = 125;
	double dimy = 90;

	double raster = 1e-1;
	
	double car[3];
	car[2] = 0;

	unsigned char* roi = new unsigned char[1280 * 720];
	memset(roi, 0, 1280 * 720);
	


	//projectIntoImagePlane(C, matData, car, &u, &v);


	Mat rot(Size(3, 3), CV_64FC1);
	Mat trans(Size(1,3), CV_64FC1);
	for (int r = 0; r < rot.rows; r++)
	{
		for (int c = 0; c < rot.cols; c++)
		{
			rot.at<double>(r, c) = cvInvert.at<double>(r, c);
		}
	}
	trans.at<double>(0) = cvInvert.at<double>(0, 3);
	trans.at<double>(1) = cvInvert.at<double>(1, 3);
	trans.at<double>(2) = cvInvert.at<double>(2, 3);

	cout << rot << endl;
	cout << trans << endl;

	Mat rot_vec;
	Rodrigues(rot, rot_vec);

	cout << rot_vec << endl;

	Mat zero_coeff;
	zero_coeff.zeros(Size(1, 5), CV_64FC1);
	

	double u, v;
	for (double x =-50; x <= 50; x += 10)
	{		
		for (double y = -30; y <= 30; y += raster)
		{
			car[0] = x;
			car[1] = y;
			car[2] = 0;

			Vec<double, 3> vec;
			vec.val[0] = car[0];
			vec.val[1] = car[1];
			vec.val[2] = car[2];
			
			Mat points;
			points.push_back(vec);

			Mat output;
			//m_distortion_coefficients
			projectPoints(points, rot_vec, trans, m_C, Mat(), output);
			Vec<double, 2> p= output.at<Vec<double, 2>>(0);
			u = p.val[0];
			v = p.val[1];
			projectIntoImagePlane(C, matData, car, &u, &v);

			if(u >= 0 && v >= 0 && u<1280 && v < 720)
			{

				int uint = std::floor(u);
				int vint = std::floor(v);
				roi[vint*1280 + uint] = 1;

				Vec3b &source = imageUndistorted.at<Vec3b>(v, u);
				source.val[0] = 255;
				source.val[1] = 0;
				source.val[2] = 0;
				
			}
		}
	}


	for (double x = -50; x <= 50; x += raster)
	{
		for (double y = -30; y <= 30; y += 10)
		{
			car[0] = x;
			car[1] = y;
			car[2] = 0;

			Vec<double, 3> vec;
			vec.val[0] = car[0];
			vec.val[1] = car[1];
			vec.val[2] = car[2];

			Mat points;
			points.push_back(vec);

			Mat output;
			//m_distortion_coefficients
			projectPoints(points, rot_vec, trans, m_C, Mat(), output);
			Vec<double, 2> p = output.at<Vec<double, 2>>(0);
			u = p.val[0];
			v = p.val[1];
			projectIntoImagePlane(C, matData, car, &u, &v);

			if (u >= 0 && v >= 0 && u<1280 && v < 720)
			{

				int uint = std::floor(u);
				int vint = std::floor(v);
				roi[vint * 1280 + uint] = 1;

				Vec3b &source = imageUndistorted.at<Vec3b>(v, u);
				source.val[0] = 0;
				source.val[1] = 255;
				source.val[2] = 0;

			}
		}
	}

	for (double x = -62.5; x <= 62.5; x += 125)
	{
		for (double y = -20; y <= 20; y += 40)
		{
			car[0] = x;
			car[1] = y;
			projectIntoImagePlane(C, matData, car, &u, &v);
			circle(imageUndistorted, Point(u, v), 5, Scalar(110, 220, 0));
		}
	}

	vector<Point2f> vert(5);
	car[0] = -dimx / 2;
	car[1] = -dimy / 2;
	projectIntoImagePlane(C, matData, car, &u, &v);
	vert[0] = Point(u, v);

	car[0] = dimx / 2;
	car[1] = -dimy / 2;
	projectIntoImagePlane(C, matData, car, &u, &v);
	vert[1] = Point(u, v);

	car[0] = dimx / 2;
	car[1] = dimy / 2;
	projectIntoImagePlane(C, matData, car, &u, &v);
	vert[2] = Point(u, v);

	car[0] = -dimx / 2;
	car[1] = dimy / 2;
	projectIntoImagePlane(C, matData, car, &u, &v);
	vert[3] = Point(u, v);

	car[0] = -dimx / 2;
	car[1] = -dimy / 2;
	projectIntoImagePlane(C, matData, car, &u, &v);
	vert[4] = Point(u, v);
	

	//vert[0] = Point(0, 0);
	//vert[1] = Point(1, 0);
	//vert[2] = Point(1, 1);
	//vert[3] = Point(0, 1);
	//vert[4] = Point(0, 0);
	double ret = pointPolygonTest(vert, Point2f(0, 320), false);



	 
	for (int r = 0; r < imageUndistorted.rows; r++)
	{
		for (int c = 0; c < imageUndistorted.cols; c++)
		{
			if (pointPolygonTest(vert, Point2f(c, r), false) < 0)
			{
				Vec3b &source = imageUndistorted.at<Vec3b>(r, c);
				source.val[0] = 0;
				source.val[1] = 100;
				source.val[2] = 0;
				correctedImage.at<unsigned char>(r, c) = 0;
			}
		}
	}
	imshow("win1", imagegray);
	imshow("win2", correctedImage);
	imwrite("distorted.png", imagegray);
	imwrite("undistorted.png", correctedImage);
	waitKey(0);



	imshow("win2", imageUndistorted);
	imwrite("respic.jpg", imageUndistorted);
 	waitKey(0);


		
		
		
		
		
		
		
	imshow("win1", imageUndistorted);
	for (int r = 0; r < imageUndistorted.rows; r++)
	{
		for (int c = 0; c < imageUndistorted.cols; c++)
		{
			//if (r < 100 && c < 100)
			if (!(roi[r * 1280 + c] == 1))
			{
				Vec3b &source = imageUndistorted.at<Vec3b>(r, c);
				source.val[0] = 0;
				source.val[1] = 0;
				source.val[2] = 0;				
				cout << c << "\t" << r << endl;
				//image.at<Vec3b>(r, c) = source;
			}
		}
	}
	imshow("win2", imageUndistorted);
	
	//for (int r = 0; r < 500; r++)
	//{
	//	for (int c = 0; c < 500; c++)
	//	{
	//		Vec3b& source = imageUndistorted.at<Vec3b>(r, c);
	//		//source.val[0] = 0;
	//		//source.val[1] = 0;
	//		source.val[2] = 255;
	//	}
	//}


	
	waitKey(0);                                          // Wait for a keystroke in the window
	return 0;
}


int main(int argc, char* argv[])
{
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}