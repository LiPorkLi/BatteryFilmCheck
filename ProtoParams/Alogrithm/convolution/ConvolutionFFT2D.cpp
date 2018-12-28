#include "stdafx.h"
#include "ConvolutionFFT2D.h"
#include "helper_cuda.h"
#include "cuda_runtime_api.h"
#include <algorithm>
#include "helper_timer.h"


CConvolutionFFT2D::CConvolutionFFT2D()
{
	d_DataSpectrum = d_KernelSpectrum = NULL; 
	d_PaddedKernel = d_PaddedData = NULL;
}


CConvolutionFFT2D::~CConvolutionFFT2D()
{
	if (d_DataSpectrum)
	{
		checkCudaErrors(cudaFree(d_DataSpectrum));
		checkCudaErrors(cudaFree(d_KernelSpectrum));
		checkCudaErrors(cudaFree(d_PaddedData));
		checkCudaErrors(cudaFree(d_PaddedKernel));
		checkCudaErrors(cufftDestroy(m_fftPlanInv));
		checkCudaErrors(cufftDestroy(m_fftPlanFwd));
	}
}

bool CConvolutionFFT2D::Initia(int iImageHeight, int iImageWidth, int iImageStep, int iKernelHeight, int iKernelWidth, int iCenterY/* = -1*/, int iCenterX/* = -1*/, cudaStream_t* devStream/* = NULL*/)
{
	if (iImageStep!=sizeof(float)*iImageWidth)
	{
		return false;
	}
	m_iImageWidth = iImageWidth;
	m_iImageHeight = iImageHeight;
	m_iImageStep = sizeof(float)* m_iImageWidth;
	
	m_iKernelHeight = iKernelHeight;
	m_iKernelWidth = iKernelWidth;

	m_iFFTHeight = snapTransformSize(m_iImageHeight + m_iKernelHeight - 1);
	m_iFFTWidth = snapTransformSize(m_iImageWidth + m_iKernelWidth - 1);

	if (d_DataSpectrum)
	{
		checkCudaErrors(cudaFree(d_DataSpectrum));
		checkCudaErrors(cudaFree(d_KernelSpectrum));
		checkCudaErrors(cudaFree(d_PaddedData));
		checkCudaErrors(cudaFree(d_PaddedKernel));
		checkCudaErrors(cufftDestroy(m_fftPlanInv));
		checkCudaErrors(cufftDestroy(m_fftPlanFwd));
	}

	checkCudaErrors(cudaMalloc((void **)&d_PaddedData, m_iFFTHeight * m_iFFTWidth * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_PaddedKernel, m_iFFTHeight * m_iFFTWidth * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_DataSpectrum, m_iFFTHeight * (m_iFFTWidth / 2 + 1) * sizeof(fComplex)));
	checkCudaErrors(cudaMalloc((void **)&d_KernelSpectrum, m_iFFTHeight * (m_iFFTWidth / 2 + 1) * sizeof(fComplex)));

	dev_Stream = devStream;

	checkCudaErrors(cufftPlan2d(&m_fftPlanFwd, m_iFFTHeight, m_iFFTWidth, CUFFT_R2C));
	checkCudaErrors(cufftPlan2d(&m_fftPlanInv, m_iFFTHeight, m_iFFTWidth, CUFFT_C2R));

	if (dev_Stream!=NULL)
	{
		cufftSetStream(m_fftPlanFwd, *dev_Stream);
		cufftSetStream(m_fftPlanInv, *dev_Stream);
	}


	if (iCenterX==-1)
	{
		m_iCx = (m_iKernelWidth >> 1);
	}
	else
	{
		m_iCx = std::min(iCenterX, m_iKernelWidth - 1);
	}

	if (iCenterY == -1)
	{
		m_iCy = (m_iKernelHeight >> 1);
	}
	else
	{
		m_iCy = std::min(iCenterY, m_iKernelHeight - 1);
	}

	return true;
}

bool CConvolutionFFT2D::SetKernel(float* dev_pKerenl, int iKernelStep)
{
	if (iKernelStep!=sizeof(float)*m_iKernelWidth)
	{
		return false;
	}
	padKernel(
		d_PaddedKernel,
		dev_pKerenl,
		m_iFFTHeight,
		m_iFFTWidth,
		m_iKernelHeight,
		m_iKernelWidth,
		m_iCy,
		m_iCx,
		dev_Stream
		);

	printf("...transforming convolution kernel\n");
	checkCudaErrors(cufftExecR2C(m_fftPlanFwd, (cufftReal *)d_PaddedKernel, (cufftComplex *)d_KernelSpectrum));

	return true;
}
void CConvolutionFFT2D::GetResult(float* dev_dstImg)
{
	//cudaMemcpyAsync();
	if (dev_Stream == NULL)
	{
		checkCudaErrors(cudaMemcpy2D((unsigned char*)dev_dstImg, m_iImageStep/*m_iFFTWidth * sizeof(float)*/, (unsigned char*)d_PaddedData, m_iFFTWidth * sizeof(float), m_iImageWidth*sizeof(float), m_iImageHeight, cudaMemcpyDeviceToDevice));
	}
	else
	{
		checkCudaErrors(cudaMemcpy2DAsync((unsigned char*)dev_dstImg, m_iImageStep/*m_iFFTWidth * sizeof(float)*/, (unsigned char*)d_PaddedData, m_iFFTWidth * sizeof(float), m_iImageWidth*sizeof(float), m_iImageHeight, cudaMemcpyDeviceToDevice, *dev_Stream));
	}
}
bool CConvolutionFFT2D::Execute(float* dev_srcImg, float* dev_dstImg, double* dTime)
{
// 	StopWatchInterface *hTimer = NULL;
// 	sdkCreateTimer(&hTimer);
// 	sdkStartTimer(&hTimer);

	padDataClampToBorder(
		d_PaddedData,
		dev_srcImg,
		m_iFFTHeight,
		m_iFFTWidth,
		m_iImageHeight,
		m_iImageWidth,
		m_iKernelHeight,
		m_iKernelWidth,
		m_iCy,
		m_iCx,
		dev_Stream
		);


	//printf("...running GPU FFT convolution: ");
	checkCudaErrors(cufftExecR2C(m_fftPlanFwd, (cufftReal *)d_PaddedData, (cufftComplex *)d_DataSpectrum));
	modulateAndNormalize(d_DataSpectrum, d_KernelSpectrum, m_iFFTHeight, m_iFFTWidth, 1, dev_Stream);
	checkCudaErrors(cufftExecC2R(m_fftPlanInv, (cufftComplex *)d_DataSpectrum, (cufftReal *)d_PaddedData));

	//cudaMemcpyAsync();
// 	if (dev_Stream==NULL)
// 	{
// 		checkCudaErrors(cudaMemcpy2D((unsigned char*)dev_dstImg, m_iImageStep/*m_iFFTWidth * sizeof(float)*/, (unsigned char*)d_PaddedData, m_iFFTWidth * sizeof(float), m_iImageWidth*sizeof(float), m_iImageHeight, cudaMemcpyDeviceToDevice));
// 	}
// 	else
// 	{
// 		checkCudaErrors(cudaMemcpy2DAsync((unsigned char*)dev_dstImg, m_iImageStep/*m_iFFTWidth * sizeof(float)*/, (unsigned char*)d_PaddedData, m_iFFTWidth * sizeof(float), m_iImageWidth*sizeof(float), m_iImageHeight, cudaMemcpyDeviceToDevice, *dev_Stream));
// 	}

// 	checkCudaErrors(cudaDeviceSynchronize());
// 	sdkStopTimer(&hTimer);
// 	*dTime = sdkGetTimerValue(&hTimer);

	return true;
}


int CConvolutionFFT2D::snapTransformSize(int dataSize)
{
	int hiBit;
	unsigned int lowPOT, hiPOT;

	dataSize = iAlignUp(dataSize, 16);

	for (hiBit = 31; hiBit >= 0; hiBit--)
		if (dataSize & (1U << hiBit))
		{
			break;
		}

	lowPOT = 1U << hiBit;

	if (lowPOT == (unsigned int)dataSize)
	{
		return dataSize;
	}

	hiPOT = 1U << (hiBit + 1);

	if (hiPOT <= 1024)
	{
		return hiPOT;
	}
	else
	{
		return iAlignUp(dataSize, 512);
	}
}