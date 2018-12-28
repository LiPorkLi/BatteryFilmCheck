#pragma once
#include "CheckMethod.h"
#include "AlogrithmBase.h"
class CMaxMinValueCheck : public CCheckMethod, public CAlogrithmBase
{
public:
	CMaxMinValueCheck(MainParam::param* p,  std::shared_ptr<CRunTimeHandle> pHandle);
	~CMaxMinValueCheck();

	bool TrainTemplate(cv::Mat& img, bool bIncrementTrain = true);
	bool check(cv::Mat& img, cv::Mat& diffImg, double* dTime);
	bool check(cv::cuda::GpuMat& img, cv::cuda::GpuMat& diffImg, double* dTime){ return true; };
	void SetParam(void* param);
private:
	int m_iTrainCount;
	cv::Mat m_rowMin, m_rowMax;
	cv::Mat m_freamMin, m_freamMax;
	int m_iThreshold_low, m_iThreshold_high;
	std::vector<std::future<bool>> m_vecReturn;

	void GetMaxMinModel(cv::Mat& imgFream, cv::Mat& rowMin, cv::Mat& rowMax, int iFreadIdx/* = 0*/);
	void ExtendLineToFream(cv::Size& imgSize, cv::Mat& rowImg, cv::Mat& FreamImg);
	bool MaxMinCheckThread(cv::Mat* img, cv::Rect& rtTruth, cv::Mat* imgMin, cv::Mat* imgMax, cv::Mat* DiffImg);
};

