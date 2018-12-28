#pragma once
#include "cv.h"
#include "highgui.h"
#include "fftw3.h"
#include <thread>
#include "..\GlobalData.h"

class CInspectionAlogrithm
{
public:
	CInspectionAlogrithm();
	~CInspectionAlogrithm();

public:
	/*=======Test======*/
	static void GaussBlur_fft(cv::Mat& SrcImg, cv::Mat& DstImg, cv::Size& kernelSize, float fSigmaW = 0.0f, float fSigmaH = 0.0f);//边界像素差值方式与GaussianBlur差值方式不同，所以边界像素相差较大
	static void Conv_FFT32f(cv::Mat& SrcImg, cv::Mat& DstImg, cv::Mat& kernel);
	static void Conv_FFT32f_fftw(cv::Mat& SrcImg, cv::Mat& DstImg, cv::Mat& kernel);
	/*=======Test======*/

	/*=======Public======*/
	static cv::Mat GetGaussKernel(cv::Size& kernelSize, float fSigmaW = 0.0f, float fSigmaH = 0.0f);
	static cv::Mat GetHist(cv::Mat& img);
	static bool GetFrdMask(cv::Size& imgSize, std::vector<std::vector<std::pair<int, int>>>& vecvecBoundary, cv::Mat& frd);
	/*=======Public======*/

	/*=======Image Base Process======*/
	static void DeNoise(cv::Mat& SrcImg, cv::Mat& DstImg);
	/*=======Image Base Process======*/

	/*=======Camera Process======*/
	static bool ImageAdd(cv::Mat& imgFream, cv::Mat& bgdSum_Num, cv::Mat& frdSum_Num, cv::Mat& frdMask, int iFreadIdx = 0);//软件白平衡
	static void GetFlatFieldParam(cv::Mat& bgdSum_Num, cv::Mat& frdSum_Num, cv::Mat& bgdParam, cv::Mat& frdParam, int iDstFrd = 128, int iDstBgd = 240);
	static bool FlatField(cv::Mat& SrcDstImg, cv::Mat& bgdParam, cv::Mat& frdParam, cv::Mat& frdMask);
	//static void SystemRectify(cv::Mat& SrcImg, cv::Mat& );
	/*=======Image Base Process======*/

	/*====DATA Split===*/
	static void SplitImg(cv::Size& imageSize, cv::Size& cellSize, cv::Size& padding, std::vector<cv::Rect>& vecSplitRect, std::vector<cv::Rect>& vecTruthRect);
	/*====DATA Split===*/

	/*=======DOG=======*/
	static bool DOGCheck4GaussianThread(cv::Mat& img, cv::Rect& rtSplit, cv::Rect& rtTruth, std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma, cv::Mat& vecRlt, int id);
	static bool DOGCheck4FFTThread(cv::Mat& img, cv::Rect& rtSplit, cv::Rect& rtTruth, cv::Size& normSize, std::vector<cv::Mat>& vecKernelFFT, std::vector<cv::Size>& vecDiffKernelSize,
		std::vector<cv::Mat>& vecRlt, int id);
	static bool DOGCheck4FFTwThread(cv::Mat& img, cv::Rect& rtSplit, cv::Rect& rtTruth, cv::Size& normSize, std::vector<cv::Mat>& vecKernelFFT, std::vector<cv::Size>& vecDiffKernelSize,
		std::vector<fftwf_plan>& vecForwardImg, std::vector<fftwf_plan>& vecBackward, std::vector<cv::Mat>& vecBuffPatch, std::vector<cv::Mat>& vecBuffComplexImg, std::vector<cv::Mat>& vecBuffComplexBackward,
		std::vector<cv::Mat>& vecRlt, std::map<std::thread::id, int>& threadIndex, int id);
	static void InitiaFFtw(fftwf_plan* forwardImg, fftwf_plan* backward, cv::Mat* buffPatch, cv::Mat* buffComplexImg, cv::Mat* buffComplexMul);
	static cv::Size GetMaxSize(std::vector<cv::Size>& vecKernelSize);
	static bool GetFFTParam(cv::Size& imgSize, std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma, std::vector<cv::Mat>& vecDOGKernelFFT, std::vector<cv::Size>& vecDiffKernelSize);
	static bool GetFFTwParam(cv::Size& imgSize, std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma, std::vector<cv::Mat>& vecDOGKernelFFT, std::vector<cv::Size>& vecDiffKernelSize);
	static cv::Mat GetDiffKernel3(std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma, cv::Size& refSize, float fRefSigma);
	static bool GetFFTwParam3(cv::Size& imgSize, std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma, cv::Mat& DOGKernelFFT, cv::Size& diffKernelSize);
	static bool DOGCheck4FFTwThread3(cv::Mat& img, cv::Rect& rtSplit, cv::Rect& rtTruth, cv::Size& normSize, int iDiffKernelNum, cv::Mat& KernelFFT, cv::Size& DiffKernelSize, 
		std::vector<fftwf_plan>& vecForwardImg, std::vector<fftwf_plan>& vecBackward, std::vector<cv::Mat>& vecBuffPatch, std::vector<cv::Mat>& vecBuffComplexImg, std::vector<cv::Mat>& vecBuffComplexBackward, cv::Mat& vecRlt, std::map<std::thread::id, int>& threadIndex, int id);
	/*=======DOG=======*/
	
	/*=======Train Model Inspection=======*/
	static void GetMaxMinModel(cv::Mat& imgFream, cv::Mat& rowMin, cv::Mat& rowMax, int iFreadIdx = 0);
	static bool MaxMinCheckThread(cv::Mat& img, cv::Rect& rtSplit, cv::Rect& rtTruth, cv::Mat& imgMin, cv::Mat& imgMax, cv::Mat& DiffImg, int id);
	static void ExtendLineToFream(cv::Size& imgSize, cv::Mat& rowImg, cv::Mat& FreamImg);
	static void GetExpVarModel(cv::Mat& imgFream, cv::Mat& rowExp, cv::Mat& rowSigma, int iFreadIdx = 0);
	/*=======Train Model Inspection=======*/
	
	/*====Boundary Search===*/
	static bool BoundarySearch(cv::Mat& SrcImg, std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>>& vecBoudaryParam, std::vector<std::vector<std::pair<int, int>>>& vecvecBoundary);
	/*====Boundary Search===*/

	/*====Blob===*/
	static bool GetBlobThread(cv::Mat& binaryImg, cv::Rect& RoiRt, std::vector<std::vector<cv::Point>>* vecvecContour1, std::vector<std::vector<cv::Point>>* vecvecContour2, std::mutex* mtx1, std::mutex* mtx2);
	static void MergeContour(std::vector<std::vector<cv::Point>>& vecTempContour, std::vector<std::vector<cv::Point>>& vecvecContour);
	static float GetContourDist(std::vector<cv::Point>& c1, std::vector<cv::Point>& c2);
	static bool Geo_BlobToDefectThread(cv::Mat& diffImg, std::vector<cv::Point>& contour, int iBlobThreshold, std::vector<GeoClassifyModel>& vecGeoClassifyParam,
		std::vector<std::pair<int, DefectData>>* vecDefectRect, std::mutex* mtx, int& contourIdx);
	/*====Blob===*/
private:
	/*=======DOG=======*/
	static cv::Mat GetDiffKernel1(cv::Size& s1, cv::Size& s2, float fSigma1, float fSigma2);//kw = kw1+kw2-1 kh=kh1+kh2-1
	static cv::Mat GetDiffKernel2(cv::Size& s1, cv::Size& s2, cv::Size& maxSize, float fSigma1, float fSigma2);//kw = std::max(kw1,kw2) kh=std::max(kh1,kh2)
	/*=======DOG=======*/


	/*====Boundary Search===*/
	static void SearchLine(cv::Mat& img, int& iS, int& iE, std::vector<int>& vecLine);
	static void GetPatchImgBoundary(cv::Mat& PatchImg, int iThr1, int iThr2, int& iLeft, int& iRight);
	/*====Boundary Search===*/

	/*=======Public======*/
	
	/*=======Public======*/
};

