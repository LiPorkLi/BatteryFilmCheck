#pragma once
#include "CheckMethod.h"
#include "AlogrithmBase.h"
class CMeanStandardCheck : public CCheckMethod, public CAlogrithmBase
{
public:
	CMeanStandardCheck(MainParam::param* p,  std::shared_ptr<CRunTimeHandle> pHandle);
	~CMeanStandardCheck();

	bool TrainTemplate(cv::Mat& img, bool bIncrementTrain = true);
	bool check(cv::Mat& img, cv::Mat& diffImg, double* dTime);
	bool check(cv::cuda::GpuMat& img, cv::cuda::GpuMat& diffImg, double* dTime){ return true; };
	void SetParam(void* param);
private:
	int m_iTrainCount;
	float m_fSigmaTime;
	cv::Mat m_rowExp, m_rowStd, m_rowMin, m_rowMax;
	cv::Mat m_freamMin, m_freamMax;
	std::vector<std::future<bool>> m_vecReturn;
	void GetExpVarModel(cv::Mat& imgFream, cv::Mat& rowExp, cv::Mat& rowSigma, int iFreadIdx/* = 0*/);
	void ExtendLineToFream(cv::Size& imgSize, cv::Mat& rowImg, cv::Mat& FreamImg);
	bool MaxMinCheckThread(cv::Mat* img, cv::Rect& rtTruth, cv::Mat* imgMin, cv::Mat* imgMax, cv::Mat* DiffImg);
};

