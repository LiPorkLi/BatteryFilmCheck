#pragma once
#include "CheckMethod.h"
#include "AlogrithmBase.h"
#include "fftw3.h"
//#include "cufftw.h"
#include "convolution/ConvolutionFFT2D.h"

class CDOGCheck : public CCheckMethod, public CAlogrithmBase
{
public:
	CDOGCheck(MainParam::param* p,  std::shared_ptr<CRunTimeHandle> pHandle/* = NULL*/);
	~CDOGCheck();

	bool check(cv::Mat& img, cv::Mat& diffImg, double* dTime);
	bool check(cv::cuda::GpuMat& img, cv::cuda::GpuMat& diffImg, double* dTime);
	void SetParam(void* param);
private:
	std::vector<std::future<bool>> m_vecReturn;
	std::vector<cv::Rect> m_vecTruth, m_vecSplit;

	cv::Mat m_RefKernel, m_FundKernel, m_DiffKernel;
	float m_fRefSigma, m_fFundSigma;

	int m_iMarkDOG;
	std::vector<cv::Mat> m_vecDOGKernelFFT;
	std::vector<cv::Size> m_vecDiffKernelSize;
	cv::Size m_fftNormSize;
	std::vector<cv::Mat> m_vecBuffPatch, m_vecComplexImg, m_vecComplexMul;
	std::vector<fftwf_plan> m_vecFFTWForward, m_vecFFTWBackward;
	
	//Gpu fft
	cv::cuda::Stream cvStream0, cvStream1;
	cudaStream_t dev_Stream0, dev_Stream1;
	CConvolutionFFT2D* m_pConv;
	float* m_pSrc, *m_pKernel, *m_pDst;
	cv::cuda::GpuMat m_BuffPatch_gpu, m_Kernel_gpu, m_DstImg;

	CConvolutionFFT2D* m_pConv1;
	float* m_pSrc1,  *m_pDst1;
	cv::cuda::GpuMat m_BuffPatch_gpu1, m_DstImg1;
	//cpu fft
	std::map<std::thread::id, int> m_mapThreadIndex;
	cv::Mat m_DogKernelFFtw;
	int m_iThresholdDark, m_iThresholdLight;


	cv::Mat GetGaussKernel(cv::Size& kernelSize, float fSigmaW /*= 0.0f*/, float fSigmaH /*= 0.0f*/);
	bool GetFFTwParam3(cv::Size& imgSize, cv::Mat& diffKernel, cv::Mat& DOGKernelFFT);
	bool DOGCheck4GaussianThread(cv::Mat* img, cv::Rect& rtSplit, cv::Rect& rtTruth, cv::Size& refSize, float fRefSigma, cv::Size& fundSize, float fFundSigma, cv::Mat* diffImg, int iDarkThr, int iLightThr);
	bool DOGCheck4FFTwThread3(cv::Mat* img, cv::Rect& rtSplit, cv::Rect& rtTruth, cv::Size& normSize, cv::Mat* KernelFFT, cv::Size& DiffKernelSize,
		std::vector<fftwf_plan>* vecForwardImg, std::vector<fftwf_plan>* vecBackward, std::vector<cv::Mat>* vecBuffPatch, std::vector<cv::Mat>* vecBuffComplexImg,
		std::vector<cv::Mat>* vecBuffComplexBackward, cv::Mat* diffImg, std::map<std::thread::id, int>* threadIndex, int iDarkThr, int iLightThr);
	cv::Mat GetDiffKernel3(cv::Mat& ref, cv::Mat& fund);
	void KernelCopyToBuff(cv::Mat& Kernel, cv::Mat& buff);

	//bool GetFFTwParam3_gpu( std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma, cv::Size& diffKernelSize);
	bool InitiaConvolution();
	void KernelCopyToBuff_gpu(cv::cuda::GpuMat& Kernel, cv::cuda::GpuMat& buff);
	void GpuMatFFt(cv::cuda::GpuMat& src, cv::cuda::GpuMat& fftdst, fftwf_plan& plan, float* pSrc, float* pDst);
};

