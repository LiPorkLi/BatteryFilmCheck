#pragma once

#include "threadpool.h"
#include "InspectionAlogrithm.h"
#include "GlobalData.h"
class CCheckProc
{
public:
	CCheckProc();
	~CCheckProc();
	void SetParam();
	void StartInspectThread();
	void StopInspectThread();
	void Pipline();
	void InspectImage(cv::Mat& fream, std::vector<cv::Mat>& tempFilterImg, cv::Mat& filterImg);
	void ReTrain();
private:
	std::threadpool executor;
	Parameters::InspectParam m_inspectParam;
	Parameters::RunTimeParam m_runtimeParam;
	Parameters::SystemRectifyParam m_systemRectiyParm;
	//system rectify param
	float m_fPhysicResolution;
	//处理流程
	int m_iDowmSampleParam;
	bool m_isInspect;
	//数据流
	cv::Size m_imgSize;
	std::vector<cv::Rect> m_vecSplit, m_vecTruthRect;
	//DOG
	std::vector<cv::Size> m_vecKernelSize;
	std::vector<float> m_vecSigma;
	int m_iMarkDOG;
	std::vector<cv::Mat> m_vecDOGKernelFFT;
	std::vector<cv::Size> m_vecDiffKernelSize;
	cv::Size m_fftNormSize;
	std::vector<cv::Mat> m_vecBuffPatch, m_vecComplexImg, m_vecComplexMul;
	std::vector<fftwf_plan> m_vecFFTWForward, m_vecFFTWBackward;
	std::map<std::thread::id, int> m_mapThreadIndex;
	cv::Mat m_DogKernelFFtw;
	cv::Size m_sumKernelSize;
	int m_iDiffKernelNum;
	int m_iThresholdDark, m_iThresholdLight;
	//MaxMIn ExpStd
	int m_iTrainCount;
	float m_fSigmaTime;
	cv::Mat m_rowMin, m_rowMax, m_rowExp, m_rowStd;
	cv::Mat m_freamMin, m_freamMax, m_freamExp, m_freamStd;
	int m_iThreshold_low, m_iThreshold_high;
	//Blob Filter param
	int m_iBlobThrsh;
	int m_iBoundaryOffset;
	//Blob
	int m_iMarkClassification;
	std::vector<GeoClassifyModel> m_vecGeoClassifiyParam;
	//flat field
	cv::Mat m_bgdFream, m_frdFream, m_bgdParam, m_frdParam;
	int m_iDstFrd, m_iDstBgd;
	
	//Boundary Search
	std::vector<std::vector<std::pair<int, int>>> m_vecvecBoundary;
	std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> m_vecSearchBoundary;

	void InspectImage(std::shared_ptr<GrabImgInfo> img);
};

