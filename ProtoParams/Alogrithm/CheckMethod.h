#pragma once
#include "RunTimeHandle.h"
class CCheckMethod
{
public:
	CCheckMethod();
	~CCheckMethod();
	
	virtual bool check(cv::Mat& img, cv::Mat& diffImg, double* dTime) = 0;
	virtual bool check(cv::cuda::GpuMat& img, cv::cuda::GpuMat& diffImg, double* dTime) = 0;
};

