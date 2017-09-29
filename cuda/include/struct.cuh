#pragma once
#include <stdio.h>
#include <cstring>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <gtest/gtest.h>

#define N_TEST_IMAGE_SIZE (25)
#define N_MAX_CENTROIDS	  (1000)
#define N_MAX_MARKERS	  (16)
#define N_MAX_CENTROIDS_MERGED (3*N_MAX_MARKERS)
#define N_PIXEL_MARKER_WDTH (21)
#define DIST_AB (14)
#define DIST_A0 (3.5f)
#define N_DEOCDER_MASKS (8)

#define N_ELMENTS_C_ROW (3)
#define N_ELMENTS_C_COL (3)

#define N_ELMENTS_R_ROW (3)
#define N_ELMENTS_R_COL (3)


#define N_ELMENTS_H_ROW (4)
#define N_ELMENTS_H_COL (4)

#define N_ELMENTS_kk_ROW (1)
#define N_ELMENTS_kk_COL (5)

#define N_EDGES_PER_POINT (2)
#define N_EDGES (2)
#define N_DIM (2)
#define N_POINTS (3)
#define INT_NAN (-1000)
#define SHARED_BUFFER_SIZE (1024)

#define N_MAX_SOLVER_POINTS (N_MAX_CENTROIDS_MERGED)
#define N_SOLVER_DIMENSION (8)
#define N_SOLVER_DIMENSION_SIZE (N_SOLVER_DIMENSION*N_SOLVER_DIMENSION*N_SOLVER_DIMENSION)

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
 }                                                                 \
}

#define divceil(x, y) ((x % y) ? x / y + 1 : x / y)

struct S_ROBOT_PARAMER
{
	double lengthHole_mm;
	double widthHole_mm;
	double heightOverGround_mm;
	S_ROBOT_PARAMER(double v1, double v2, double v3)
	{
		lengthHole_mm = v1;
		widthHole_mm = v2;
		heightOverGround_mm = v3;
	}
};


template<class T>class DeviceBuffer
{

private:

	T *data;
	T *data_host;
	int size;

public:
	DeviceBuffer(int size) :size(0), data(NULL), data_host(NULL)
	{
		this->size = size;
		cudaMalloc(&data, size * sizeof(T));
		cudaCheckError();
		cudaMemset(data, 0, size * sizeof(T));
		cudaDeviceSynchronize();
		cudaCheckError();

		data_host = new T[size];
		memset(data_host, 0, size * sizeof(T));

	}

	int getSize() { return size; }

	void set(T * d)
	{
		memcpy(data_host, d, size);
		cudaMemcpy(data, d, size * sizeof(T), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		cudaCheckError();
	}

	void set()
	{
		cudaMemcpy(data, data_host, size * sizeof(T), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		cudaCheckError();
	}

	void get(T* d)
	{
		get();
		memccpy(d, data_host, size * sizeof(T));
	}

	void get()
	{
		cudaMemcpy(data_host, data, size * sizeof(T), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		cudaCheckError();
	}

	void get(int s)
	{
		cudaMemcpy(data_host, data, s * sizeof(T), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		cudaCheckError();
	}

void reset()
{
	cudaMemset(data, 0, size * sizeof(T));
	cudaDeviceSynchronize();
	cudaCheckError();
	memset(data_host, 0, size * sizeof(T));
}


T* getHostData(void)
{
	return data_host;
}

T* getDeviceData(void)
{
	return data;
}


virtual ~DeviceBuffer()
{
	if (data != NULL)
	{
		cudaFree(data);
		data = NULL;
	}
	if (data_host != NULL)
	{
		delete[] data_host;
		data_host = NULL;
	}

}



};


struct S_MARKER_DIM
{
	double edgeLength_mm;
	int edgeLength_px;


	S_MARKER_DIM(double v1, int v2)
	{
		edgeLength_mm = v1;
		edgeLength_px = v2;
	}

	// A-------C
	// | _   /
	// | _  /
	// | _ /
	// B  /

	double indexAx_na = 3.5;
	double indexAy_na = 3.5;

	double indexBx_na = 3.5;
	double indexBy_na = 17.5;

	double indexCx_na = 17.5;
	double indexCy_na = 3.5;

	double widthBlackMarker = 3;
	double widthWhiteMarker = 5;

	double dimEdge = 1;

	double scaleMinEdge = 0.7;
	double scaleMaxEdge = 1.3;


};

#define MAX_SIZE_QR_MARKER (75)
#define QR_MARKER_INDEX_OFFSET (2)
#define QR_MARKER_DIM (5)
struct S_QR_MARKER
{
	DeviceBuffer<float> *x;
	DeviceBuffer<float> *y;

	DeviceBuffer<float> *ix;
	DeviceBuffer<float> *iy;

	int size = MAX_SIZE_QR_MARKER;





	void getMarkerFromAB(int ai, int bi,
		float* const a,
		float* const b,
		float* const c)
	{
		if (ai < -(QR_MARKER_DIM - 1) / 2 || ai >(QR_MARKER_DIM - 1) / 2  ||
			bi < -(QR_MARKER_DIM - 1) / 2 || bi >(QR_MARKER_DIM - 1) / 2 )
		{
			a[0] = nan("");
			a[1] = nan("");

			b[0] = nan("");
			b[1] = nan("");

			c[0] = nan("");
			c[1] = nan("");
			return;
		}



		int xi = (ai + QR_MARKER_INDEX_OFFSET)*3;
		int yi = (bi + QR_MARKER_INDEX_OFFSET)*3;

		a[0] = x->getHostData()[yi*QR_MARKER_DIM + xi + 0];
		a[1] = y->getHostData()[yi*QR_MARKER_DIM + xi + 0];

		b[0] = x->getHostData()[yi*QR_MARKER_DIM + xi + 1];
		b[1] = y->getHostData()[yi*QR_MARKER_DIM + xi + 1];

		c[0] = x->getHostData()[yi*QR_MARKER_DIM + xi + 2];
		c[1] = y->getHostData()[yi*QR_MARKER_DIM + xi + 2];

	}

	void getMarkerFromABImage(int ai, int bi,
		float* const a,
		float* const b,
		float* const c)
	{
		if (ai < -(QR_MARKER_DIM - 1) / 2 || ai >(QR_MARKER_DIM - 1) / 2 ||
			bi < -(QR_MARKER_DIM - 1) / 2 || bi >(QR_MARKER_DIM - 1) / 2)
		{
			a[0] = nan("");
			a[1] = nan("");

			b[0] = nan("");
			b[1] = nan("");

			c[0] = nan("");
			c[1] = nan("");
			return;
		}



		int xi = (ai + QR_MARKER_INDEX_OFFSET) * 3;
		int yi = (bi + QR_MARKER_INDEX_OFFSET) * 3;

		a[0] = ix->getHostData()[yi*QR_MARKER_DIM + xi + 0];
		a[1] = iy->getHostData()[yi*QR_MARKER_DIM + xi + 0];

		b[0] = ix->getHostData()[yi*QR_MARKER_DIM + xi + 1];
		b[1] = iy->getHostData()[yi*QR_MARKER_DIM + xi + 1];

		c[0] = ix->getHostData()[yi*QR_MARKER_DIM + xi + 2];
		c[1] = iy->getHostData()[yi*QR_MARKER_DIM + xi + 2];

	}
	
	void getNearestMarkerFromA(const float* const A,
		float* const a,
		float* const b,
		float* const c)
	{
		//find shortest
		float dist = FLT_MAX;
		int index = -1;
		float diffx, diffy;
		float tmp;
		for (int i = 0; i < QR_MARKER_DIM*QR_MARKER_DIM; i++)
		{
			diffx = x->getHostData()[(3 * i) + 0] - A[0];
			diffy = y->getHostData()[(3 * i) + 0] - A[1];
			tmp = sqrtf(diffx*diffx + diffy*diffy);
			if (tmp < dist)
			{
				dist = tmp;
				index = i;
			}
		}
		

		a[0] = x->getHostData()[(3 * index) + 0];
		a[1] = y->getHostData()[(3 * index) + 0];

		b[0] = x->getHostData()[(3 * index) + 1];
		b[1] = y->getHostData()[(3 * index) + 1];

		c[0] = x->getHostData()[(3 * index) + 2];
		c[1] = y->getHostData()[(3 * index) + 2];

	}

	S_QR_MARKER()
	{
		x = new DeviceBuffer<float>(MAX_SIZE_QR_MARKER);
		y = new DeviceBuffer<float>(MAX_SIZE_QR_MARKER);		

		ix = new DeviceBuffer<float>(MAX_SIZE_QR_MARKER);
		iy = new DeviceBuffer<float>(MAX_SIZE_QR_MARKER);

		int start;
		float index_offset_x[] = { -30, -5 , nan(""), 20 , 45 };
		float index_offset_y[] = { -30, -5 , nan(""), 20 , 45 };
		

		for (int xi = 0; xi < 5; xi++)
		{
			for (int yi = 0; yi < 5; yi++)
			{
				start = (yi * 5 + xi) * 3;
				if (xi == 2 || yi == 2)
				{
					for (int i = start; i < start + 3; i++)
					{
						x->getHostData()[i] = nan("");
						y->getHostData()[i] = nan("");
					}
				}
				else
				{
					// setting corner A
					x->getHostData()[start + 0] = index_offset_x[xi];
					y->getHostData()[start + 0] = index_offset_y[yi];

					// setting corner B
					x->getHostData()[start + 1] = index_offset_x[xi] - 15;
					y->getHostData()[start + 1] = index_offset_y[yi]; 

					// setting corner C
					x->getHostData()[start + 2] = index_offset_x[xi];
					y->getHostData()[start + 2] = index_offset_y[yi] - 15;

				}

			}
		}

		

	}

	void set()
	{
		x->set();
		y->set();
		ix->set();
		iy->set();
	}

	~S_QR_MARKER()
	{
		delete x;
		delete y;

		delete ix;
		delete iy;
	}

};


struct S_LAUNCH_CONFIG
{
	dim3 grid;
	dim3 block;	
	int  sharedMemorySize;
};

struct S_UNDISTORT_IMAGE
{
	unsigned char* img;
	unsigned char* resimg;
	unsigned char* mask;
	int*		   mapx;
	int*		   mapy;
};

struct CENTROIDS
{
	
	float* centroidX;
	float* centroidY;
	unsigned int*   nCentroids;
	
	float* centroidXHost;
	float* centroidYHost;
	unsigned int*	nCentroidsHost;
	
	int size;
	


	CENTROIDS(int size) :nCentroids(NULL), centroidX(NULL), centroidY(NULL), nCentroidsHost(NULL), centroidXHost(NULL), centroidYHost(NULL), size(0)
	{
		this->size = size;
		
		cudaMalloc(&nCentroids, sizeof(int));
		cudaCheckError();
		cudaMalloc(&centroidX, size * sizeof(float));
		cudaCheckError();
		cudaMalloc(&centroidY, size * sizeof(float));
		cudaCheckError();

		nCentroidsHost = new unsigned int[1];
		centroidXHost = new float[size];
		centroidYHost = new float[size];

		reset();
	}

	

	void set(float* cx, float* cy, int* cn)
	{
		cudaMemcpy(nCentroids, cn, sizeof(int),			cudaMemcpyHostToDevice);
		cudaMemcpy(centroidX, cx, size * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(centroidY, cy, size * sizeof(float), cudaMemcpyHostToDevice);
		cudaCheckError();
		memcpy(nCentroidsHost, cn,sizeof(int));
		memcpy(centroidXHost, cx, size * sizeof(float));
		memcpy(centroidYHost, cy, size * sizeof(float));
	}





	~CENTROIDS()
	{
		if (size> 0)
		{
			cudaFree(nCentroids);
			nCentroids = NULL;
			cudaFree(centroidX);
			centroidX = NULL;
			cudaFree(centroidY);
			centroidY = NULL;
			delete nCentroidsHost;
			delete centroidXHost;
			delete centroidYHost;
			size = 0;
		}
	}

	void get()
	{
		cudaMemcpy(nCentroidsHost, nCentroids, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(centroidXHost, centroidX, size * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(centroidYHost, centroidY, size * sizeof(float), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		cudaCheckError();
	}

	void reset()
	{
		cudaMemset(nCentroids, 0, sizeof(int));
		cudaDeviceSynchronize();
		cudaCheckError();
		memset(nCentroidsHost, 0, sizeof(unsigned int));
	}

};




struct MARKER
{
	DeviceBuffer<int> *id;
	DeviceBuffer<int> *a;
	DeviceBuffer<int> *b;
	DeviceBuffer<int> *nMarker;
	int maxNMarker;

	DeviceBuffer<unsigned char> *decoderMasks;
	DeviceBuffer<int> *indexNumeric0;
	DeviceBuffer<int> *indexNumeric1;
	DeviceBuffer<int> *indexNumeric2;

	DeviceBuffer<int> *indexByte0;
	DeviceBuffer<int> *indexByte1;
	DeviceBuffer<int> *indexByte2;
	DeviceBuffer<int> *indexByte3;


	int numeric0[10] = { 293,
		292,
		272,
		271,
		251,
		250,
		230,
		229,
		209,
		208};

	int numeric1[20] = { 207,
		206,
		228,
		227,
		249,
		248,
		270,
		269,
		291,
		290};

	int numeric2[20] = { 312,
		311,
		333,
		332,
		354,
		353,
		375,
		374,
		396,
		395};

	int byte0[8] = { 352,
		351,
		331,
		330,
		310,
		309,
		289,
		288};

	int byte1[8] = { 268,
		267,
		247,
		246,
		226,
		225,
		205,
		204};

	int byte2[8] = { 203,
		202,
		224,
		223,
		245,
		244,
		266,
		265};

	int byte3[8] = { 287,
		286,
		308,
		307,
		329,
		328,
		350,
		349};

	MARKER(int maxNMarker)
	{
		this->maxNMarker = maxNMarker;
		this->id = new DeviceBuffer<int>(maxNMarker);
		this->a = new DeviceBuffer<int>(maxNMarker);
		this->b = new DeviceBuffer<int>(maxNMarker);
		this->nMarker = new DeviceBuffer<int>(1);
		this->decoderMasks = new DeviceBuffer<unsigned char>(N_PIXEL_MARKER_WDTH * N_PIXEL_MARKER_WDTH * N_DEOCDER_MASKS);

		indexNumeric0 = new DeviceBuffer<int>(10);
		indexNumeric1 = new DeviceBuffer<int>(10);
		indexNumeric2 = new DeviceBuffer<int>(10);

		indexByte0 = new DeviceBuffer<int>(8);
		indexByte1 = new DeviceBuffer<int>(8);
		indexByte2 = new DeviceBuffer<int>(8);
		indexByte3 = new DeviceBuffer<int>(8);

		indexNumeric0->set(this->numeric0);
		indexNumeric1->set(this->numeric1);
		indexNumeric2->set(this->numeric2);

		indexByte0->set(this->byte0);
		indexByte1->set(this->byte1);
		indexByte2->set(this->byte2);
		indexByte3->set(this->byte3);

		//init deocer masks
		unsigned char* p_EncodeMasks;

		p_EncodeMasks = decoderMasks->getHostData() + 0 * N_PIXEL_MARKER_WDTH*N_PIXEL_MARKER_WDTH;
		for (int i = 0; i < N_PIXEL_MARKER_WDTH; i++)
		{
			for (int j = 0; j < N_PIXEL_MARKER_WDTH; j++)
			{
				if ((i + j) % 2 == 0)
				{
					p_EncodeMasks[i*N_PIXEL_MARKER_WDTH + j] = 1;
				}
				else
				{
					p_EncodeMasks[i*N_PIXEL_MARKER_WDTH + j] = 0;
				}
			}
		}

		p_EncodeMasks = decoderMasks->getHostData()+ 1* N_PIXEL_MARKER_WDTH*N_PIXEL_MARKER_WDTH;
		for (int i = 0; i < N_PIXEL_MARKER_WDTH; i++)
		{
			for (int j = 0; j < N_PIXEL_MARKER_WDTH; j++)
			{
				if (i  % 2 == 0)
				{
					p_EncodeMasks[i*N_PIXEL_MARKER_WDTH + j] = 1;
				}
				else
				{
					p_EncodeMasks[i*N_PIXEL_MARKER_WDTH + j] = 0;
				}
			}
		}

		p_EncodeMasks = decoderMasks->getHostData() + 2 * N_PIXEL_MARKER_WDTH*N_PIXEL_MARKER_WDTH;
		for (int i = 0; i < N_PIXEL_MARKER_WDTH; i++)
		{
			for (int j = 0; j < N_PIXEL_MARKER_WDTH; j++)
			{
				if (j % 3 == 0)
				{
					p_EncodeMasks[i*N_PIXEL_MARKER_WDTH + j] = 1;
				}
				else
				{
					p_EncodeMasks[i*N_PIXEL_MARKER_WDTH + j] = 0;
				}
			}
		}

		p_EncodeMasks = decoderMasks->getHostData() + 3 * N_PIXEL_MARKER_WDTH*N_PIXEL_MARKER_WDTH;
		for (int i = 0; i < N_PIXEL_MARKER_WDTH; i++)
		{
			for (int j = 0; j < N_PIXEL_MARKER_WDTH; j++)
			{
				if ((i+j) % 3 == 0)
				{
					p_EncodeMasks[i*N_PIXEL_MARKER_WDTH + j] = 1;
				}
				else
				{
					p_EncodeMasks[i*N_PIXEL_MARKER_WDTH + j] = 0;
				}
			}
		}

		p_EncodeMasks = decoderMasks->getHostData() + 4 * N_PIXEL_MARKER_WDTH*N_PIXEL_MARKER_WDTH;
		for (int i = 0; i < N_PIXEL_MARKER_WDTH; i++)
		{
			for (int j = 0; j < N_PIXEL_MARKER_WDTH; j++)
			{
				if ((i/2 + j/3) % 2 == 0)
				{
					p_EncodeMasks[i*N_PIXEL_MARKER_WDTH + j] = 1;
				}
				else
				{
					p_EncodeMasks[i*N_PIXEL_MARKER_WDTH + j] = 0;
				}
			}
		}

		p_EncodeMasks = decoderMasks->getHostData() + 5 * N_PIXEL_MARKER_WDTH*N_PIXEL_MARKER_WDTH;
		for (int i = 0; i < N_PIXEL_MARKER_WDTH; i++)
		{
			for (int j = 0; j < N_PIXEL_MARKER_WDTH; j++)
			{
				if ((i*j)% 2 + (i*j) % 3 == 0)
				{
					p_EncodeMasks[i*N_PIXEL_MARKER_WDTH + j] = 1;
				}
				else
				{
					p_EncodeMasks[i*N_PIXEL_MARKER_WDTH + j] = 0;
				}
			}
		}

		p_EncodeMasks = decoderMasks->getHostData() + 6 * N_PIXEL_MARKER_WDTH*N_PIXEL_MARKER_WDTH;
		for (int i = 0; i < N_PIXEL_MARKER_WDTH; i++)
		{
			for (int j = 0; j < N_PIXEL_MARKER_WDTH; j++)
			{
				if (((i*j) % 3 +i*j) % 2 == 0)
				{
					p_EncodeMasks[i*N_PIXEL_MARKER_WDTH + j] = 1;
				}
				else
				{
					p_EncodeMasks[i*N_PIXEL_MARKER_WDTH + j] = 0;
				}
			}
		}

		p_EncodeMasks = decoderMasks->getHostData() + 7 * N_PIXEL_MARKER_WDTH*N_PIXEL_MARKER_WDTH;
		for (int i = 0; i < N_PIXEL_MARKER_WDTH; i++)
		{
			for (int j = 0; j < N_PIXEL_MARKER_WDTH; j++)
			{
				if (((i*j) % 3 + i+j) % 2 == 0)
				{
					p_EncodeMasks[i*N_PIXEL_MARKER_WDTH + j] = 1;
				}
				else
				{
					p_EncodeMasks[i*N_PIXEL_MARKER_WDTH + j] = 0;
				}
			}
		}

		decoderMasks->set();
	}

	~MARKER()
	{
		delete this->id;
		delete this->a;
		delete this->b;
		delete this->nMarker;
		delete this->decoderMasks;

		delete indexNumeric0;
		delete indexNumeric1;
		delete indexNumeric2;

		delete indexByte0;
		delete indexByte1;
		delete indexByte2;
		delete indexByte3;


	}

	void get()
	{
		this->id->get();
		this->a->get();
		this->b->get();
		this->nMarker->get();
	}

};

struct Data2D
{
	DeviceBuffer<float> *x;
	DeviceBuffer<float> *y;
	DeviceBuffer<int>	*size;

	Data2D(int size)
	{
		this->x = new DeviceBuffer<float>(size);
		this->y = new DeviceBuffer<float>(size);
		this->size = new DeviceBuffer<int>(1);
		this->size->set(&size);
	}

	~Data2D()
	{
		delete this->x;
		delete this->y;
		delete this->size;
	}

	void get()
	{
		this->x->get();
		this->y->get();		
	}

	void set()
	{
		this->x->set();
		this->y->set();
	}

};

struct S_SOLVER_PARAMETER
{
	float* xm;
	float* ym;

	float* xmeas;
	float* ymeas;

	int*	nPoints;

	int maxIter;
	double eps;

	float xinit;
	float yinit;
	float tinit;

	double xinit_intervall;
	double yinit_intervall;
	double tinit_intervall;

	double* optimalTranformation;
	double* minimumCosts;

};

struct S_MARKER_PARAMETER
{
	float offsetFeatureA2Origin_x;
	float offsetFeatureA2Origin_y;
	float makerIndex2PixelIndex;
	unsigned char binarisationThreshold;

	//int	markerHeight;
	//int	markerWidth;
	//int	markerActionPerThread;

	unsigned char* decodeMasks[N_DEOCDER_MASKS];

	int* indexNumeric0;
	int* indexNumeric1;
	int* indexNumeric2;

	int* indexByte0;
	int* indexByte1;
	int* indexByte2;
	int* indexByte3;

	int* markerID;
	int* markerOffsetA;
	int* markerOffsetB;

	int maxEnd = 25;
	


};

struct S_LOAD_AND_STORE_PARAMS
{
	int maskOffset;
	int threadFaktorX;
	int threadFaktorY;
	int bufferWidth;
	int bufferHeight;

	void init(int maxMaximumMaskSize, int imageWidth, int imageHeight, int blockSizeX, int blockSizeY)
	{
		int size = maxMaximumMaskSize > 2 ? maxMaximumMaskSize : 3;

		if ((size + 1) % 2 == 1)
		{
			size++;
		}
		maskOffset = -((size - 1) / 2);
		bufferWidth = blockSizeX + (-2 * maskOffset);
		bufferHeight = blockSizeY + (-2 * maskOffset);

		threadFaktorX = (int)ceil((bufferWidth / blockSizeX));
		threadFaktorY = (int)ceil((bufferHeight / blockSizeY));

	}
};

struct S_MASK_PARAMETER
{
	int maskWidth;
	int maskOffset;
	int nMaskWhite;
	int offsetEdge2Center;
	int offsetEdge2CenterCorner;
	int dimEdge;
	int dimEdgeCorner;
	int minAbsoluteDiffEdge;
	int minRelativeDiffEdge;
	unsigned char*  pattern;

	int threadFaktorX;
	int threadFaktorY;

	unsigned char threshPattern;

	unsigned char debayerBatter[9] = { 1, 2, 1,2, 4, 2,1, 2, 1 };
	bool debayer = false;
};

struct S_IMAGE_PARAMETER
{
	int imageWidth;
	int imageHeight;
	unsigned char*  img_raw;
	unsigned char*  img;
	unsigned char*  prop;
	unsigned char*  mask;
};

struct S_CENTROID_PARAMETER
{


	int				nMarkerCornerThreadSize;
	int*			centerX;
	int*			centerY;
	unsigned char*	centerProp;

	unsigned int*	nCentroids;
	float			radiusThreshold;

	float*			centerAx;
	float*			centerAy;

	float*			centerBx;
	float*			centerBy;

	float*			centerCx;
	float*			centerCy;

	unsigned int*	nMarker;
};

struct S_QR_CODE_PARAMTER
{
	float			squareDistanceBCMin;
	float			squareDistanceBCMax;

	float			squareDistanceABMin;
	float			squareDistanceABMax;

	unsigned char*	markerKorrigiert;
	int				nBlocksBufferWidth;

	float*			centerAxInMarkerCoords;
	float*			centerAyInMarkerCoords;

	float*			centerBxInMarkerCoords;
	float*			centerByInMarkerCoords;

	float*			centerCxInMarkerCoords;
	float*			centerCyInMarkerCoords;
};

struct S_MAP_PARAMETER
{
	float*			cx;
	float*			cy;
	float*			ix;
	float*			iy;
	double*			initalGuess;
	double*			initalGuessImage;
	
};




struct S_CAMERA_PARAMETER
{
	double C[N_ELMENTS_C_ROW*N_ELMENTS_C_COL];
	double H[N_ELMENTS_H_COL* N_ELMENTS_H_ROW];
	double kk[N_ELMENTS_kk_ROW*N_ELMENTS_kk_COL];
	double Ri[N_ELMENTS_R_ROW*N_ELMENTS_R_COL];
	double t[2];
	double dist;

	double rotatationCart2Image;
	double rotatationCart2ImageMatrix[4];
	double converionPx2Mm;
};

struct S_IMG_PROC_KERNEL_PARAMS
{
	S_MASK_PARAMETER			maskParameter;
	S_MARKER_PARAMETER			markerParameter;
	S_QR_CODE_PARAMTER			qrCodeParameter;
	S_IMAGE_PARAMETER			imageParameter;
	
	S_CENTROID_PARAMETER		centroidParameter;
	
	S_SOLVER_PARAMETER			solverParameter;
	S_UNDISTORT_IMAGE			undistortParamer;
	S_LOAD_AND_STORE_PARAMS		loadAndStoreParams;
	S_MAP_PARAMETER				mapParameter;
	S_CAMERA_PARAMETER			cameraParameter;

	bool debug;
	
};




//struct S_IMG_PROC_KERNEL_DATA
//{
//	DeviceBuffer<unsigned char> *img;
//	DeviceBuffer<unsigned char> *mask;
//	DeviceBuffer<unsigned char> *prop;
//	DeviceBuffer<unsigned char> *pattern;
//
//
//	CENTROIDS centroids{ 25 };
//
//	S_IMG_PROC_KERNEL_DATA()
//	{
//		img = new DeviceBuffer<unsigned char>(25);
//		mask = new DeviceBuffer<unsigned char>(25);
//		prop = new DeviceBuffer<unsigned char>(25);
//		pattern = new DeviceBuffer<unsigned char>(9);
//
//	}
//
//	~S_IMG_PROC_KERNEL_DATA()
//	{
//		delete img;
//		delete mask;
//		delete prop;
//		delete pattern;
//	}
//
//
//	void set()
//	{
//		img->set();
//		mask->set();
//		prop->set();
//		pattern->set();
//		centroids.reset();
//	}
//
//	void get()
//	{
//		img->get();
//		mask->get();
//		prop->get();
//		pattern->get();
//		centroids.get();
//	}
//};

using namespace std;

struct test_state
{

	unsigned char img[N_TEST_IMAGE_SIZE];
	unsigned char mask[N_TEST_IMAGE_SIZE];
	unsigned char pattern[9];
	unsigned char prop[N_TEST_IMAGE_SIZE];


	int indexProb;
	int valueProb;

	test_state(unsigned char testImage[], unsigned char   testMask[], unsigned char   testPattern[])
	{
		memcpy(img, testImage, N_TEST_IMAGE_SIZE);
		memcpy(mask, testMask, N_TEST_IMAGE_SIZE);
		memcpy(pattern, testPattern, 9);
		memset(prop, 0, N_TEST_IMAGE_SIZE);
	}

	virtual ~test_state()
	{
		

	}

	friend std::ostream& operator<<(std::ostream& os, const test_state& obj)
	{
		os << endl << endl << "IMG|MASK|PATTERN|PROP" << endl;
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 5; j++)
			{

				os << static_cast<unsigned int>(obj.img[i * 5 + j]) << "|" <<
					static_cast<unsigned int>(obj.mask[i * 5 + j]) << "|" <<
					static_cast<unsigned int>(obj.pattern[i * 5 + j]) << "|" <<
					static_cast<unsigned int>(obj.prop[i * 5 + j]) << "\t";
			}
			os << endl;
		}
		return os;
	}
};

//
//struct Kernel_Test : testing::Test
//{
//
//
//	S_LAUNCH_CONFIG				launchConfig;
//	S_IMG_PROC_KERNEL_PARAMS	params;
//	S_IMG_PROC_KERNEL_DATA		*data;
//
//
//
//
//	Kernel_Test()
//	{
//		data = new S_IMG_PROC_KERNEL_DATA{};
//		
//		params.maskParameter.maskWidth = 3;
//		params.maskParameter.maskOffset = -1;
//		params.loadAndStoreParams.bufferWidth = 5;
//		params.loadAndStoreParams.bufferHeight = 5;
//		params.imageParameter.imageWidth = 5;
//		params.imageParameter.imageHeight = 5;
//
//		params.imageParameter.img = data->img->getDeviceData();
//		params.imageParameter.mask = data->mask->getDeviceData();
//		params.maskParameter.pattern = data->pattern->getDeviceData();
//		params.imageParameter.prop = data->prop->getDeviceData();
//
//		//params.centroidParameter.centerX = data->centroids.centroidX;
//		//params.centroidParameter.centerY = data->centroids.centroidY;
//		//params.centroidParameter.nCentroids = data->centroids.nCentroids;
//		
//
//		launchConfig.grid = dim3(1, 1);
//		launchConfig.block = dim3(1, 1);
//	}
//
//	virtual ~Kernel_Test()
//	{
//		delete data;
//	}
//
//	void prepareTest()
//	{
//
//
//		data->set();
//
//		cudaDeviceSynchronize();
//		cudaCheckError();
//	}
//
//	void cleanUpTest()
//	{
//
//		cudaDeviceSynchronize();
//		cudaCheckError();
//
//		data->get();
//
//		cudaDeviceSynchronize();
//		cudaCheckError();
//	}
//
//
//};


