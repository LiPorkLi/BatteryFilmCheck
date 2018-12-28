#pragma once
#include "AlogrithmBase.h"
#include <core.hpp>
#include "cudaFunction.h"
/*#include <core.hpp>*/
class CPreprocess : public CAlogrithmBase
{
public:
	CPreprocess(MainParam::param* p,  std::shared_ptr<CRunTimeHandle> pHandle);
	~CPreprocess();

	//bool Preprocess(cv::cuda::GpuMat& SrcDstImg, double* dTime);
	bool Preprocess(cv::Mat& SrcDstImg, double* dTime);
	bool Preprocess(cv::cuda::GpuMat& SrcDstImg, bool hr,  double* dTime);
	void SetParam(void* param);

	bool UnPadding(cv::Mat& SrcDstImg, double* dTime);
	bool UnPadding(cv::cuda::GpuMat& SrcDstImg, double* dTime);
private:
	cv::Size m_imgSize;
	std::vector<std::future<bool>> m_vecReturn;
	cv::Ptr<cv::cuda::Filter> m_blur;
	cv::Rect m_RectOffset;

	cv::cuda::GpuMat m_gpu_mask, m_gpu_mask4;

	bool MeanFilter(cv::Mat* SrcDstImg, cv::Rect& rtSplit, cv::Rect& rtTruth);
	bool Padding(cv::Mat* SrcDstImg, int iOffset, int iY0, int iY1, int iX, int iRoiWidth);
	bool Padding_Cpu(cv::Mat& SrcDstImg, cv::Rect& rectOffset);
	bool Padding_Gpu(cv::cuda::GpuMat* SrcDstImg, cv::Rect& rectOffset);
};

