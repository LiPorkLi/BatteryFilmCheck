#pragma once
//#include "RunTimeHandle.h"
#include "DOGCheck.h"
#include "MaxMinValueCheck.h"
#include "MeanStandardCheck.h"
class CCheckProcess
{
public:
	CCheckProcess(MainParam::param* p, std::shared_ptr<CRunTimeHandle> pHandle);
	~CCheckProcess();

	bool Execute(cv::Mat& img, cv::Mat& diffImg, double* dTime);
	bool Execute(cv::cuda::GpuMat& img, cv::cuda::GpuMat& diffImg, double* dTime);
	bool IsDOG();
private:
	//std::shared_ptr<CRunTimeHandle> m_pHandle;
	std::shared_ptr<CCheckMethod> m_method;
	int m_iCheckType;
	int m_iRefKernelWidth;
};

