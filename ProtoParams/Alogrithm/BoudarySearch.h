#pragma once
#include "AlogrithmBase.h"
class CBoudarySearch : public CAlogrithmBase
{
public:
	CBoudarySearch(MainParam::param* p,  std::shared_ptr<CRunTimeHandle> pHandle);
	~CBoudarySearch();

	bool BoundarySearch(cv::Mat& SrcImg, cv::cuda::GpuMat& frdMask, double* dTime);
	bool BoundarySearch(cv::cuda::GpuMat& SrcImg, cv::cuda::GpuMat& frdMask_gpu, cv::Mat& frdMask_cpu, double* dTime);

	bool ExpandBoundary(cv::Mat& SrcDstImg, int iSizeX, double* dTime);
	bool ExpandBoundary(cv::cuda::GpuMat& SrcDstImg, int iSizeX, double* dTime);

	bool ErodeDiffImgBoundary(cv::cuda::GpuMat& DiffImg, cv::cuda::GpuMat& DiffMask, int iOffsetLeft, int iOffsetRight, double* dTime);
	bool ErodeDiffImgBoundary(cv::Mat& DiffImg, cv::Mat& DiffMask, int iOffsetLeft, int iOffsetRight, double* dTime);

	bool GetBoundaryPix(int* iLeft, int* iRight);

	void SetParam(void* param);
private:
	std::vector<std::vector<std::pair<int, int>>> m_vecvecBoundary;
	std::vector<std::future<void>> m_vecReturn;
	std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> m_vecBoudaryParam;

	void SearchLine2(cv::Mat* img, int& iS, int& iE, std::vector<int>* vecLine);
	void SearchLine(cv::Mat* img, int& iS, int& iE, std::vector<int>* vecLine);
	void GetPatchImgBoundary(cv::Mat& PatchImg, int iThr1, int iThr2, int* iLeft, int* iRight);
	bool GetFrdMask(cv::Size& imgSize, std::vector<std::vector<std::pair<int, int>>>& vecvecBoundary, cv::Mat& frd);
	bool GetFrdMask(cv::Size& imgSize, std::vector<std::vector<std::pair<int, int>>>& vecvecBoundary, cv::cuda::GpuMat& frd);
	bool copyBoundary(cv::Mat* SrcDstImg, std::vector<std::vector<std::pair<int, int>>>* vecvecBoundary, int iSizeX, int iRow0, int iRow1);
	bool copyBoundary_gpu(cv::cuda::GpuMat* SrcDstImg, std::vector<std::vector<std::pair<int, int>>>* vecvecBoundary, int iSizeX, int iRow0, int iRow1);

	void SearchLine_gpu(cv::cuda::GpuMat& img, int& iS, int& iE, std::vector<int>* vecLine);
	void GetPatchImgBoundary_gpu(cv::cuda::GpuMat& PatchImg, int iThr1, int iThr2, int* iLeft, int* iRight);

	void SetMask_gpu(cv::cuda::GpuMat* srcdst, std::vector<std::pair<int, int>>* vecBoundary, int iStart, int iEnd);
	void SetMask(cv::Mat* srcdst, std::vector<std::pair<int, int>>* vecBoundary, int iStart, int iEnd);
private:
	cv::Ptr<cv::cuda::Filter> m_f_erode, m_f_dilate, m_f_sobel;
	int* dev_pLeft, *dev_pRight;
	int *m_pLeft, *m_pRight;
};

