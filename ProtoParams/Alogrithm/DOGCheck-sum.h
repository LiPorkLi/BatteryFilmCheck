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

	std::vector<cv::Size> m_vecKernelSize;
	std::vector<float> m_vecSigma;
	int m_iMarkDOG;
	std::vector<cv::Mat> m_vecDOGKernelFFT;
	std::vector<cv::Size> m_vecDiffKernelSize;
	cv::Size m_fftNormSize;
	std::vector<cv::Mat> m_vecBuffPatch, m_vecComplexImg, m_vecComplexMul;
	std::vector<fftwf_plan> m_vecFFTWForward, m_vecFFTWBackward;
	
	//Gpu
	//fftwf_plan m_FFTWForward_gpu, m_FFTWBackward_gpu;
	CConvolutionFFT2D* m_pConv;
	//float* m_pSrc, *m_pComplex, *m_pDOGKerelComplex;
	//cv::cuda::GpuMat m_BuffPatch_gpu, m_ComplexImg_gpu, m_DogKernelFFtw_gpu;
	float* m_pSrc, *m_pKernel, *m_pDst;
	cv::cuda::GpuMat m_BuffPatch_gpu, m_Kernel_gpu, m_DstImg;

	std::map<std::thread::id, int> m_mapThreadIndex;
	cv::Mat m_DogKernelFFtw;
	cv::Size m_sumKernelSize;
	int m_iDiffKernelNum;
	int m_iThresholdDark, m_iThresholdLight;


	cv::Mat GetDiffKernel1(cv::Size& s1, cv::Size& s2, float fSigma1, float fSigma2);//kw = kw1+kw2-1 kh=kh1+kh2-1
	cv::Mat GetDiffKernel2(cv::Size& s1, cv::Size& s2, cv::Size& maxSize, float fSigma1, float fSigma2);//kw = std::max(kw1,kw2) kh=std::max(kh1,kh2)
	cv::Mat GetGaussKernel(cv::Size& kernelSize, float fSigmaW /*= 0.0f*/, float fSigmaH /*= 0.0f*/);
	bool GetFFTParam(cv::Size& imgSize, std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma, std::vector<cv::Mat>& vecDOGKernelFFT, std::vector<cv::Size>& vecDiffKernelSize);
	cv::Size GetMaxSize(std::vector<cv::Size>& vecKernelSize);
	bool GetFFTwParam(cv::Size& imgSize, std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma, std::vector<cv::Mat>& vecDOGKernelFFT, std::vector<cv::Size>& vecDiffKernelSize);
	bool GetFFTwParam3(cv::Size& imgSize, std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma, cv::Mat& DOGKernelFFT, cv::Size& diffKernelSize);
	bool GetFFTwParam4(cv::Size& imgSize, std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma, float* _DOGKernelFFT);
	bool DOGCheck4GaussianThread(cv::Mat* img, cv::Rect& rtSplit, cv::Rect& rtTruth, std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma, cv::Mat* diffImg, int iDarkThr, int iLightThr);
	bool DOGCheck4FFTwThread3(cv::Mat* img, cv::Rect& rtSplit, cv::Rect& rtTruth, cv::Size& normSize, int iDiffKernelNum, cv::Mat* KernelFFT, cv::Size& DiffKernelSize,
		std::vector<fftwf_plan>* vecForwardImg, std::vector<fftwf_plan>* vecBackward, std::vector<cv::Mat>* vecBuffPatch, std::vector<cv::Mat>* vecBuffComplexImg,
		std::vector<cv::Mat>* vecBuffComplexBackward, cv::Mat* diffImg, std::map<std::thread::id, int>* threadIndex, int iDarkThr, int iLightThr);
	cv::Mat GetDiffKernel3(std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma, cv::Size& refSize, float fRefSigma);
	void KernelCopyToBuff(cv::Mat& Kernel, cv::Mat& buff);

	//bool GetFFTwParam3_gpu( std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma, cv::Size& diffKernelSize);
	bool InitiaConvolution(std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma);
	void KernelCopyToBuff_gpu(cv::cuda::GpuMat& Kernel, cv::cuda::GpuMat& buff);
	void GpuMatFFt(cv::cuda::GpuMat& src, cv::cuda::GpuMat& fftdst, fftwf_plan& plan, float* pSrc, float* pDst);
};

