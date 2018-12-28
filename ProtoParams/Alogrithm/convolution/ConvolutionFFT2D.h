#pragma once
#include "convolutionFFT2D_common.h"
#include "cufft.h"
class CConvolutionFFT2D
{
public:
	CConvolutionFFT2D();
	~CConvolutionFFT2D();

	bool Initia(int iImageHeight, int iImageWidth, int iImageStep, int iKernelHeight, int iKernelWidth, int iCenterY=-1,int iCenterX = -1, cudaStream_t* devStream = NULL);
	bool SetKernel(float* dev_pKerenl, int iKernelStep);
	bool Execute(float* dev_srcImg, float* dev_dstImg, double* dTime);
	void GetResult(float* dev_dstImg);
private:
	int m_iImageWidth, m_iImageHeight, m_iImageStep;
	int m_iKernelWidth, m_iKernelHeight, m_iCy, m_iCx;
	int m_iFFTWidth, m_iFFTHeight;
	fComplex *d_DataSpectrum, *d_KernelSpectrum;

	cufftHandle m_fftPlanFwd, m_fftPlanInv;

	float *d_PaddedKernel, *d_PaddedData;

	cudaStream_t* dev_Stream;

	int snapTransformSize(int dataSize);
};

