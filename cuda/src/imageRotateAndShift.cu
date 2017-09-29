#include "imageRotateAndShift.cuh"

#define N_DIST (1.0f)
#define N_CORNER_DIST_PX		(3.0f)
#define N_CORNER_DIST_PX_MINOR  (0.0f)
#define CORNER_2_CENTER_DIST_PX (3.5f)
#define N_MARKER_BUFFER_SIZE (444)
#define N_DYNAMIC_RANGE (4)
//#define N_MASK_HEIGHT (7)
//#define EDGE_THRESHOLD (64)
#define N_MASK_HEIGHT (11)
#define EDGE_THRESHOLD (64)

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <sstream>
#include <exception>


namespace rotateAndShift
{



	__device__  float norm2d(const float* const v)
	{
		return sqrtf(v[0] * v[0] + v[1] * v[1]);
	}

	__device__  void vec_sub(const float* const a, const float* const b, float* c)
	{
		c[0] = a[0] - b[0];
		c[1] = a[1] - b[1];
	}

	__device__  void normVecs(
		int	px,
		const float* const	a,
		const float* const  b,
		const float* const	c,
		float*  xn,
		float*  yn,
		float*  xpn,
		float*  ypn,
		float*  dist_px_x,
		float*  dist_px_y)
	{

		float dx;
		float dy;
		float x[2];
		float y[2];

		vec_sub(c, a, x);
		vec_sub(b, a, y);

		xn[0] = x[0] / (float)px;
		xn[1] = x[1] / (float)px;

		yn[0] = y[0] / (float)px;
		yn[1] = y[1] / (float)px;

		dx = norm2d(x);
		dy = norm2d(y);

		*dist_px_x = dx / (float)px;
		*dist_px_y = dy / (float)px;

		xpn[0] = x[0] / dx;
		xpn[1] = x[1] / dx;

		ypn[0] = y[0] / dy;
		ypn[1] = y[1] / dy;


	}


	__device__  bool isBlack2(unsigned char i_c, unsigned char i_m, int pix, int piy, const S_IMG_PROC_KERNEL_PARAMS& params)
	{
		unsigned val = params.imageParameter.prop[piy*params.imageParameter.imageWidth + pix];
		unsigned norm_val = ((val - i_c) * 255) / i_m;
		if (norm_val < params.markerParameter.binarisationThreshold)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	__device__ bool isInBounce(int pix, int piy, const S_IMG_PROC_KERNEL_PARAMS& params)
	{
		if (pix < 0 || pix >= params.imageParameter.imageWidth ||
			piy < 0 || piy >= params.imageParameter.imageHeight)
		{
			return false;
		}
		return true;
	}

	__device__ unsigned char getImageValue(int pix, int piy, const S_IMG_PROC_KERNEL_PARAMS& params)
	{
		return params.imageParameter.prop[piy*params.imageParameter.imageWidth + pix];
	}

	__device__ unsigned char normalizeValue(unsigned char i_c, unsigned char i_m, unsigned char v)
	{
		int norm_val = ((v - i_c) * 255) / i_m;
		if (norm_val > 255)
		{
			norm_val = 255;
		}
		if (norm_val < 0)
		{
			norm_val = 0;
		}
		return (unsigned char)norm_val;
	}

	__device__  bool isBlack(unsigned char i_c,
		unsigned char i_m,
		int pix, int piy, 
		const S_IMG_PROC_KERNEL_PARAMS& params)
	{
		if (pix < 0 || pix >= params.imageParameter.imageWidth ||
			piy < 0 || piy >= params.imageParameter.imageHeight)
		{
			return true;
		}
		
		unsigned val = params.imageParameter.prop[piy*params.imageParameter.imageWidth + pix];
		unsigned norm_val = ((val - i_c) * 255) /i_m;
		if (norm_val < params.markerParameter.binarisationThreshold)
		{
			return true;
		}
		else
		{
			return false;
		}

	}

	__device__  void findSingleEdgeFromBlack2WhiteArray(
		const float* const vn,
		const float* const start,
		const float* const vc,
		unsigned char i_c,
		unsigned char i_m,
		float* pi,
		const S_IMG_PROC_KERNEL_PARAMS& params)
	{
		float d  = 0.0f;
		float db = -1.0f;
		float px, py;
		int	pix, piy;
			
		int sumWhite = 0, sumBlack = 0;

	
		unsigned char val;
		for (int i = 0; i<params.markerParameter.maxEnd; i++)
		{
			
			for (int j = -(N_MASK_HEIGHT - 1) / 2; j < (N_MASK_HEIGHT - 1) / 2; j++)
			{
				//first black
				px = start[0] + db*vn[0] + (float)j*vc[0];
				py = start[1] + db*vn[1] + (float)j*vc[1];
				pix = floor(px);
				piy = floor(py);
				if (isInBounce(pix, piy, params))
				{
					val = getImageValue(pix, piy, params);
					val = normalizeValue(i_c, i_m, val);
					sumBlack += val;
					
				}

				//second white
				px = start[0] + d*vn[0] + (float)j*vc[0];
				py = start[1] + d*vn[1] + (float)j*vc[1];
				pix = floor(px);
				piy = floor(py);
				if (isInBounce(pix, piy, params))
				{
					val = getImageValue(pix, piy, params);
					val = normalizeValue(i_c, i_m, val);
					sumWhite += val;
					
				}
			}

			if ((sumWhite - sumBlack) > EDGE_THRESHOLD*N_MASK_HEIGHT)
			{
				break;
			}
			d += N_DIST;
			db+= N_DIST;
		}
		pi[0] = pix - 0.5f*vn[0];
		pi[1] = piy - 0.5f*vn[1];
	}


	__device__  void findSingleEdgeFromBlack2White(
		const float* const vn,
		const float* const start,
		unsigned char i_c,
		unsigned char i_m,
		float* pi,
		const S_IMG_PROC_KERNEL_PARAMS& params)
	{
		float d = 0.0f;		
		float px, py;
		int	pix, piy;
		for(int i=0; i<params.markerParameter.maxEnd; i++)
		{			
			px = start[0] + d*vn[0];
			py = start[1] + d*vn[1];
			pix = floor(px);
			piy = floor(py);

			
			if (!isBlack(i_c, i_m, pix, piy, params))
			{
				break;
			}
			d += N_DIST;
		}
		pi[0] = pix - 0.5f*vn[0];
		pi[1] = piy - 0.5f*vn[1];
	}


	__device__  void fastLineIntersectionVector(
		const float* const C,
		float*  I)
	{
		float x_1 = C[0];
		float y_1 = C[1];
		float x_2 = C[2];
		float y_2 = C[3];
		float x_3 = C[4];
		float y_3 = C[5];
		float x_4 = C[6];
		float y_4 = C[7];

		I[0] = ((x_4 - x_3)*(x_2*y_1 - x_1*y_2) - (x_2 - x_1)*(x_4*y_3 - x_3*y_4)) / ((y_4 - y_3)*(x_2 - x_1) - (y_2 - y_1)*(x_4 - x_3));
		I[1] = ((y_1 - y_2)*(x_4*y_3 - x_3*y_4) - (y_3 - y_4)*(x_2*y_1 - x_1*y_2)) / ((y_4 - y_3)*(x_2 - x_1) - (y_2 - y_1)*(x_4 - x_3));

	}





	

	__device__  void findEdgeX(
		const float* const aa,
		const float* const x,
		const float* const y,
		const float* const xp,	
		const float* const yp,
		unsigned char i_c,
		unsigned char i_m,
		const S_IMG_PROC_KERNEL_PARAMS& params,
		float* const c)
	{

		

		float a[2];

		float test[2];

		a[0] = x[0] * N_CORNER_DIST_PX + y[0] * N_CORNER_DIST_PX_MINOR + aa[0];
		a[1] = x[1] * N_CORNER_DIST_PX + y[1] * N_CORNER_DIST_PX_MINOR + aa[1];

	//	findSingleEdgeFromBlack2White(xp, a, i_c, i_m, c, params);
		findSingleEdgeFromBlack2WhiteArray(xp, a, yp, i_c, i_m, c, params);

		c[2] = c[0] + N_CORNER_DIST_PX * y[0];
		c[3] = c[1] + N_CORNER_DIST_PX * y[1];


/*		float a[2];

		a[0] = x[0] * N_CORNER_DIST_PX + y[0] * N_CORNER_DIST_PX + aa[0];
		a[1] = x[1] * N_CORNER_DIST_PX + y[1] * N_CORNER_DIST_PX + aa[1];

		findSingleEdgeFromBlack2White(xp, a, i_c, i_m, c, params);

		c[2] = c[0] + N_CORNER_DIST_PX * y[0];
		c[3] = c[1] + N_CORNER_DIST_PX * y[1];	*/	
	}


	__device__  void findEdgeY(
		const float* const aa,
		const float* const x,
		const float* const y,
		const float* const xp,
		const float* const yp,
		unsigned char i_c,
		unsigned char i_m,
		const S_IMG_PROC_KERNEL_PARAMS& params,
		float* const c)
	{

		float a[2];

		a[0] = x[0] * N_CORNER_DIST_PX_MINOR + y[0] * N_CORNER_DIST_PX + aa[0];
		a[1] = x[1] * N_CORNER_DIST_PX_MINOR + y[1] * N_CORNER_DIST_PX + aa[1];

		//findSingleEdgeFromBlack2White(yp, a, i_c, i_m, c, params);
		findSingleEdgeFromBlack2WhiteArray(yp, a,xp, i_c, i_m, c, params);

		c[2] = c[0] + N_CORNER_DIST_PX * x[0];
		c[3] = c[1] + N_CORNER_DIST_PX * x[1];

		//float a[2];

		//a[0] = x[0] * N_CORNER_DIST_PX + y[0] * N_CORNER_DIST_PX + aa[0];
		//a[1] = x[1] * N_CORNER_DIST_PX + y[1] * N_CORNER_DIST_PX + aa[1];

		//findSingleEdgeFromBlack2White(yp, a, i_c, i_m, c, params);

		//c[2] = c[0] + N_CORNER_DIST_PX * x[0];
		//c[3] = c[1] + N_CORNER_DIST_PX * x[1];
	}






	// A-------C
	// | _   /
	// | _  /
	// | _ /
	// B  /



	__device__  int normRange(int range)
	{
		if (range % 2 == 0)
		{
			return max(range++, 3);
		}
		else
		{
			return max(range, 3);
		}
	}


	__device__  void calculate_dynamic_range(
		const float* const nx,
		const float* const ny,
		const float* const nxp,
		const float* const nyp,
		const float* const origin,
		float dist_x,
		float dist_y,
		int*  i_c,
		int*  i_m,
		const S_IMG_PROC_KERNEL_PARAMS& params)
	{
		float p[2];
		p[0] = threadIdx.x*nx[0] + threadIdx.y*ny[0] + origin[0];
		p[1] = threadIdx.x*nx[1] + threadIdx.y*ny[1] + origin[1];

		

		int range_x = ceilf((float)N_DYNAMIC_RANGE*dist_x);
		int range_y = ceilf((float)N_DYNAMIC_RANGE*dist_y);
	

		range_x = normRange(range_x);
		range_y = normRange(range_y);

		
	

		float pr[2];
		int ix, iy, index;
		int value, min_value = 255, max_value = 0;
		
		for (int x = -(range_x - 1) / 2; x <= (range_x - 1) / 2; x++)
		{
			for (int y = -(range_y - 1) / 2; y <= (range_y - 1) / 2; y++)
			{
				pr[0] = x*nxp[0] + y*nyp[0] + p[0];
				pr[1] = x*nxp[1] + y*nyp[1] + p[1];
				ix = floor(pr[0]);
				iy = floor(pr[1]);
				
				
				if (ix >= 0 && ix < params.imageParameter.imageWidth &&
					iy >= 0 && iy < params.imageParameter.imageHeight)
				{
					value = params.imageParameter.prop[iy*params.imageParameter.imageWidth + ix];
					index = threadIdx.y * N_PIXEL_MARKER_WDTH + threadIdx.x;
				
					min_value = min((int)min_value, (int)value);
					max_value = max((int)max_value, (int)value);
				}

			}
		}
		*i_c = min_value;
		*i_m = max_value - min_value;
		

	}



	__device__   void negateVector(const float* const in, float* const ret)
	{
		ret[0] = -in[0];
		ret[1] = -in[1];
	}

	__device__  void NotNegateVector(const float* const in, float* const ret)
	{
		ret[0] = in[0];
		ret[1] = in[1];
	}

	__device__  void negateDirection(const float* const in1, const float* const in2,
		float* const ret1, float* const ret2)
	{
		negateVector(in1, ret1);
		negateVector(in2, ret2);
	}

	__device__  void notNegateDirection(const float* const in1, const float* const in2,
		float* const ret1, float* const ret2)
	{
		NotNegateVector(in1, ret1);
		NotNegateVector(in2, ret2);
	}

	__device__  void fillQRCodeBuffer(float dist_px_x, float dist_px_y,
		const float* const nx,
		const float* const ny,
		const float* const nxp,
		const float* const nyp,
		const float* const origin,
		int i_c,
		int i_m,
		int blockIndex,
		unsigned char*const qr_buffer,
		const S_IMG_PROC_KERNEL_PARAMS& params)
	{


		int ncountx = floor(dist_px_x);
		if (ncountx % 2 == 0)
		{
			ncountx = max((int)1, (int)(ncountx - 1));
		}
		else
		{
			ncountx = max(3, (int)ncountx);
		}



		int ncounty = floor(dist_px_y);
		if (ncounty % 2 == 0)
		{
			ncounty = max((int)1, (int)(ncounty - 1));
		}
		else
		{
			ncounty = max(3, (int)ncounty);
		}

		int xim;
		int yim;

		float xc, yc;

		int isNotBlackN = 0, isBlackN = 0;
		for (int index_x = -((ncountx - 1) / 2); index_x <= ((ncountx - 1) / 2); index_x++)
		{
			for (int index_y = -((ncounty - 1) / 2); index_y <= ((ncounty - 1) / 2); index_y++)
			{
				xc = ((float)threadIdx.x + 0.5f)*nx[0] + ((float)threadIdx.y + 0.5f)*ny[0] + origin[0];
				yc = ((float)threadIdx.x + 0.5f)*nx[1] + ((float)threadIdx.y + 0.5f)*ny[1] + origin[1];

				xim = floor(((float)index_x)*nxp[0] + ((float)index_y)*nyp[0] + xc);
				yim = floor(((float)index_x)*nxp[1] + ((float)index_y)*nyp[1] + yc);

				if (xim > 0 && xim < params.imageParameter.imageWidth &&
					yim > 0 && yim < params.imageParameter.imageHeight)
				{

					if (isBlack2(i_c, i_m, xim, yim, params))
					{
						isBlackN++;
					}
					else
					{
						isNotBlackN++;
					}
				}
			}
		}
		unsigned char ret;
		unsigned char retDbg;
		if (isBlackN > isNotBlackN)
		{
			qr_buffer[threadIdx.y*N_PIXEL_MARKER_WDTH + threadIdx.x] = 1;
			retDbg = 0;			
		}
		else
		{
			qr_buffer[threadIdx.y*N_PIXEL_MARKER_WDTH + threadIdx.x] = 0;
			retDbg = 255;
		}

		if (params.debug)
		{			
			params.qrCodeParameter.markerKorrigiert[threadIdx.y * params.qrCodeParameter.nBlocksBufferWidth*N_PIXEL_MARKER_WDTH +
				blockIndex*N_PIXEL_MARKER_WDTH + threadIdx.x] = retDbg;
			
		}


	}

	__device__  void binarizeMarker(
		float* const origin,
		float* const nx,
		float* const ny,
		float* const nxp,
		float* const nyp,
		float &dist_px_x,
		float &dist_px_y,
		unsigned char* const img_buffer,
		float* const edges,
		float* const corners,
		float* const a,
		float* const b,
		float* const c,
		const S_IMG_PROC_KERNEL_PARAMS& params)
	{



		float x[2];
		float y[2];
		float xp[2];
		float yp[2];
		float center[2];

		int i_c, i_m;
		int blockId = gridDim.y*blockIdx.y + blockIdx.x;
		//calculating intial value
		if (threadIdx.x == 0 && threadIdx.y == 0)
		{
			a[0] = params.centroidParameter.centerAx[blockId];
			a[1] = params.centroidParameter.centerAy[blockId];

			b[0] = params.centroidParameter.centerBx[blockId];
			b[1] = params.centroidParameter.centerBy[blockId];

			c[0] = params.centroidParameter.centerCx[blockId];
			c[1] = params.centroidParameter.centerCy[blockId];

			normVecs(DIST_AB, a, b, c, nx, ny, nxp, nyp, &dist_px_x, &dist_px_y);

			origin[0] = CORNER_2_CENTER_DIST_PX * (-nx[0] - ny[0]) + a[0];
			origin[1] = CORNER_2_CENTER_DIST_PX * (-nx[1] - ny[1]) + a[1];
		}
		__syncthreads();


		calculate_dynamic_range(nx, ny, nxp, nyp, origin, dist_px_x, dist_px_y, &i_c, &i_m, params);
		__syncthreads();


		if (threadIdx.x < N_DIM*N_EDGES*N_POINTS && threadIdx.y == 0)
		{
			switch (threadIdx.x)
			{
			case 0:
				negateDirection(nxp, nx, xp, x);
				negateDirection(nyp, ny, yp, y);
				findEdgeX(a, x, y, xp,yp, i_c, i_m, params, &edges[threadIdx.x*N_EDGES*N_DIM]);
				break;
			case 1:
				negateDirection(nxp, nx, xp, x);
				negateDirection(nyp, ny, yp, y);
				findEdgeY(a, x, y, xp, yp, i_c, i_m, params, &edges[threadIdx.x*N_EDGES*N_DIM]);
				break;

			case 2:
				negateDirection(nxp, nx, xp, x);
				notNegateDirection(nyp, ny, yp, y);
				findEdgeX(b, x, y, xp,yp, i_c, i_m, params, &edges[threadIdx.x*N_EDGES*N_DIM]);
				break;
			case 3:
				negateDirection(nxp, nx, xp, x);
				notNegateDirection(nyp, ny, yp, y);
				findEdgeY(b, x, y, xp, yp, i_c, i_m, params, &edges[threadIdx.x*N_EDGES*N_DIM]);
				break;

			case 4:
				notNegateDirection(nxp, nx, xp, x);
				negateDirection(nyp, ny, yp, y);
				findEdgeX(c, x, y, xp,yp, i_c, i_m, params, &edges[threadIdx.x*N_EDGES*N_DIM]);
				break;
			case 5:
				notNegateDirection(nxp, nx, xp, x);
				negateDirection(nyp, ny, yp, y);
				findEdgeY(c, x, y,xp, yp, i_c, i_m, params, &edges[threadIdx.x*N_EDGES*N_DIM]);
				break;

			default:
				break;
			}

		}

		__syncthreads();
		if (threadIdx.x < N_POINTS && threadIdx.y == 0)
		{
			fastLineIntersectionVector(&edges[threadIdx.x*N_EDGES_PER_POINT*N_EDGES*N_DIM], &corners[threadIdx.x*N_DIM]);
		}
		__syncthreads();


		if (threadIdx.x == 0 && threadIdx.y == 0)
		{

			params.centroidParameter.centerAx[blockId] = corners[0 * N_DIM + 0];
			params.centroidParameter.centerAy[blockId] = corners[0 * N_DIM + 1];

			params.centroidParameter.centerBx[blockId] = corners[1 * N_DIM + 0];
			params.centroidParameter.centerBy[blockId] = corners[1 * N_DIM + 1];

			params.centroidParameter.centerCx[blockId] = corners[2 * N_DIM + 0];
			params.centroidParameter.centerCy[blockId] = corners[2 * N_DIM + 1];

			normVecs(N_PIXEL_MARKER_WDTH, &corners[0 * N_DIM], &corners[1 * N_DIM], &corners[2 * N_DIM], nx, ny, nxp, nyp, &dist_px_x, &dist_px_y);

			origin[0] = corners[0];
			origin[1] = corners[1];
		}
		__syncthreads();


		fillQRCodeBuffer(dist_px_x, dist_px_y, nx, ny, nxp, nyp, origin, i_c, i_m, blockId, img_buffer, params);

		__syncthreads();
	}


	__global__ void testRotateAdapter(S_IMG_PROC_KERNEL_PARAMS params)	
	{
		//S_IMG_PROC_KERNEL_PARAMS *p_params = (S_IMG_PROC_KERNEL_PARAMS *)cudaConstParams;
		//S_IMG_PROC_KERNEL_PARAMS& params = *p_params;

		
		__shared__ float origin[2];
		__shared__ float nx[2];
		__shared__ float ny[2];
		__shared__ float nxp[2];
		__shared__ float nyp[2];
		__shared__ float dist_px_x;
		__shared__ float dist_px_y;
		__shared__ unsigned char img_buffer[N_PIXEL_MARKER_WDTH*N_PIXEL_MARKER_WDTH]; 
		__shared__ float edges[N_EDGES_PER_POINT*N_EDGES*N_DIM*N_POINTS];
		__shared__ float corners[N_DIM*N_POINTS];

		__shared__ float a[2];
		__shared__ float b[2];
		__shared__ float c[2];


		binarizeMarker(	origin, nx, ny, nxp, nyp, dist_px_x, dist_px_y,
			img_buffer,
			edges,
			corners,
			a,
			b,
			c,
			params);



	}
	using namespace cv;
	using namespace std;




	struct ReadQrMarkerTest : testing::Test
	{




		S_IMG_PROC_KERNEL_PARAMS		params;
		S_LAUNCH_CONFIG					launchConfig;


		DeviceBuffer<unsigned char> *image_input;
		DeviceBuffer<unsigned char> *image_res1;

		DeviceBuffer<float> *				Ax;
		DeviceBuffer<float> *				Ay;

		DeviceBuffer<float> *				Bx;
		DeviceBuffer<float> *				By;

		DeviceBuffer<float> *				Cx;
		DeviceBuffer<float> *				Cy;

		Mat img, img_res;

		ReadQrMarkerTest()
		{





			
			img = imread("qrtest_input.png", CV_LOAD_IMAGE_GRAYSCALE);
			img_res = imread("qrtest_res.png", CV_LOAD_IMAGE_GRAYSCALE);

			if (!img.data || !img_res.data)
			{
				throw exception("Could not open or find the image");
			}
			

			image_input = new DeviceBuffer<unsigned char>(img.cols*img.rows);
			image_res1 = new DeviceBuffer<unsigned char>(N_MARKER_BUFFER_SIZE);


			

			Ax = new DeviceBuffer<float>(N_MAX_MARKERS);
			Ay = new DeviceBuffer<float>(N_MAX_MARKERS);
			

			Bx = new DeviceBuffer<float>(N_MAX_MARKERS);
			By = new DeviceBuffer<float>(N_MAX_MARKERS);

			Cx = new DeviceBuffer<float>(N_MAX_MARKERS);
			Cy = new DeviceBuffer<float>(N_MAX_MARKERS);

			



			params.centroidParameter.centerAx = Ax->getDeviceData();
			params.centroidParameter.centerAy = Ay->getDeviceData();

			params.centroidParameter.centerBx = Bx->getDeviceData();
			params.centroidParameter.centerBy = By->getDeviceData();

			params.centroidParameter.centerCx = Cx->getDeviceData();
			params.centroidParameter.centerCy = Cy->getDeviceData();
			params.debug = true;

			//b = [124 124]';
			//	c = [41 37]';
			//	a = [40 121]';
			
			Bx->getHostData()[0] = 124;
			By->getHostData()[0] = 124;

			Cx->getHostData()[0] = 41;
			Cy->getHostData()[0] = 37;

			Ax->getHostData()[0] = 40;
			Ay->getHostData()[0] = 121;

			Ax->set();
			Ay->set();

			Bx->set();
			By->set();

			Cx->set();
			Cy->set();
			
			memcpy(image_input->getHostData(), img.data, img.cols * img.rows);
			image_input->set();

			params.imageParameter.imageWidth = img.cols;
			params.imageParameter.imageHeight = img.rows;
			params.markerParameter.maxEnd = 100;
			params.markerParameter.binarisationThreshold = 127;


			params.imageParameter.prop				= image_input->getDeviceData();
			params.qrCodeParameter.markerKorrigiert = image_res1->getDeviceData();
			params.qrCodeParameter.nBlocksBufferWidth = 1;

		
			launchConfig.grid = dim3(1);
			launchConfig.block = dim3(N_PIXEL_MARKER_WDTH, N_PIXEL_MARKER_WDTH);

		}

		void runTest()
		{
			using namespace cv;
			using namespace std;

			

			testRotateAdapter << <launchConfig.grid, launchConfig.block >> > (params);
			cudaCheckError();

			cudaDeviceSynchronize();
			cudaCheckError();

			image_res1->get();
			
			
			for (int i = 0; i < N_PIXEL_MARKER_WDTH*N_PIXEL_MARKER_WDTH; i++)
			{
				EXPECT_EQ(img_res.data[i], image_res1->getHostData()[i]);
			}
			


			//Mat img(N_PIXEL_MARKER_WDTH, N_PIXEL_MARKER_WDTH, CV_8UC1);
			//memcpy(img.data, image_res1->getHostData(), N_PIXEL_MARKER_WDTH*N_PIXEL_MARKER_WDTH);
			//namedWindow("win1", WINDOW_AUTOSIZE);
			//imshow("win1", img);
			//imwrite("marker_res1.png", img);
			//imwrite("marker_res2.png", img_res);
			//waitKey(0);



		}

		~ReadQrMarkerTest()
		{
			delete image_input;
			delete image_res1;

			delete Ax;
			delete Ay;
			delete Bx;
			delete By;
			delete Cx;
			delete Cy;

		}
	};

	TEST_F(ReadQrMarkerTest, qrreadTest)
	{
		runTest();
	}


}