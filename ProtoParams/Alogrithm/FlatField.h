#pragma once
#include "AlogrithmBase.h"
#include "cudaFunction.h"
class CFlatField: public CAlogrithmBase
{
public:
	CFlatField(MainParam::param* p,  std::shared_ptr<CRunTimeHandle> pHandle = NULL);
	~CFlatField();

	bool GetParam(cv::Mat& imgFream, cv::Mat& frdMask, double* dTime, bool bIncrementTrain = true);
	bool GetParam(cv::cuda::GpuMat& imgFream, cv::cuda::GpuMat& frdMask, double* dTime, bool bIncrementTrain = true);
	bool TuneImg(cv::Mat& imgFream, cv::Mat& datImg, cv::Mat& frdMask, double* dTime);
	bool TuneImg(cv::cuda::GpuMat& imgFream, cv::cuda::GpuMat& datImg, cv::cuda::GpuMat& frdMask, double* dTime);
	void SetParam(void* param);

	void TuneImgSelf(cv::Mat& srcdst, cv::Mat& frdMask);
	bool TuneImgSelf(cv::cuda::GpuMat& srcdst, cv::cuda::GpuMat& frdMask, double* dTime);
private:
	cv::Mat m_target;
	cv::cuda::GpuMat m_target_gpu;
	cv::Mat m_bgdSum_Num, m_frdSum_Num, m_frdParam, m_bgdParam;
	cv::cuda::GpuMat m_bgdSum_Num_gpu, m_frdSum_Num_gpu, m_frdParam_gpu, m_bgdParam_gpu;
	std::vector<std::future<bool>> m_vecReturn;
	int m_iDstFrd, m_iDstBgd, m_iFreamIndex;

	int m_iOffsetHeightIndex;

	cv::Ptr<cv::cuda::Filter> m_blur;

	bool ImageAdd(cv::Mat& imgFream, cv::Mat& avgBgd, cv::Mat& avgFrd, cv::Mat& frdMask, int iFreadIdx = 0);//软件白平衡
	bool GetFlatFieldParam(cv::Mat& avgBgd, cv::Mat& avgFrd, cv::Mat& bgdParam, cv::Mat& frdParam, int iDstBgd = 240, int iDstFrd = 128);
	bool FlatField(cv::Mat* SrcImg, cv::Mat* DstImg, cv::Mat* bgdParam, cv::Mat* frdParam, cv::Mat* frdMask, cv::Rect roi);
	bool GetAvgMask(cv::Mat* src, cv::Mat* mask, cv::Mat* lineSrc, cv::Mat* lineMask);
	bool ReduceAdd(cv::Mat* src, cv::Mat* mask, cv::Mat* lineSrc, cv::Mat* lineMask, cv::Rect roi);

	bool GetAvgMask_gpu2(cv::cuda::GpuMat* src, cv::cuda::GpuMat* mask, cv::cuda::GpuMat* lineSrcFrd, cv::cuda::GpuMat* lineMaskFrd, cv::cuda::GpuMat* lineSrcBgd, cv::cuda::GpuMat* lineMaskBgd);
	bool ImageAdd_gpu(cv::cuda::GpuMat& imgFream, cv::cuda::GpuMat& avgBgd, cv::cuda::GpuMat& avgFrd, cv::cuda::GpuMat& frdMask, int iFreadIdx = 0);//软件白平衡
	bool GetFlatFieldParam_gpu(cv::cuda::GpuMat& avgBgd, cv::cuda::GpuMat& avgFrd, cv::cuda::GpuMat& bgdParam, cv::cuda::GpuMat& frdParam, int iDstBgd = 240, int iDstFrd = 128);
};

