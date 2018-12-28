#pragma once
#include "AlogrithmBase.h"
class CBlobAnalysis : public CAlogrithmBase
{
public:
	CBlobAnalysis(MainParam::param* p, std::shared_ptr<CRunTimeHandle> pHandle);
	~CBlobAnalysis();

	//bool BlobAnalysis(cv::Mat& maskImg, std::vector<DefectData>& vecDefect);
	//bool BlobAnalysis(cv::Mat& frdMask, std::vector<std::vector<cv::Point>>& vecvecBlob, double* dTime);
	bool BlobAnalysis(cv::Mat& frdMask, cv::Mat& diffMat, std::vector<DefectData>& vecDefectInfo, double* dTime);
	bool BlobAnalysis(cv::Mat& frdMask, cv::cuda::GpuMat& diffMat, std::vector<DefectData>& vecDefectInfo, double* dTime);

	bool SetNoCheckArea(cv::cuda::GpuMat& DiffImg, cv::cuda::GpuMat& DiffMask, double* dTime);
	//bool BlobAnalysis(cv::cuda::GpuMat& diffImg_gpu, cv::cuda::GpuMat& frdMask_gpu, cv::Mat& frdMask_cpu, std::vector<std::vector<cv::Point>>& vecvecBlob, double* dTime);
	void SetParam(void* param);

	void GetBlobBoundaryOffset(int& iLeft, int& iRight);

	void GetStripBoundary(int& iLeft, int& iRight);
private:
	// std::shared_ptr<CRunTimeHandle> m_pHandle;
	int m_iBlobThrsh, m_iBoundaryOffsetLeft, m_iBoundaryOffsetRight;
	cv::Rect m_rtLeftSetNoCheck, m_rtRightSetNoCheck;
	std::vector<std::future<bool>> m_vecReturn;
	//cv::Ptr<cv::cuda::Filter> m_f_erode;
	cv::Mat m_tempMark;
	std::vector<cv::Point> m_vecPt1, m_vecPt2;
	std::vector<cv::Rect> m_vecSplit, m_vecTruth;
	std::vector<std::vector<cv::Point>> m_vecvecBlob;

	bool GetBlobThread1(cv::Mat* binaryImg, cv::Rect& RoiRt, std::vector<std::vector<cv::Point>>* vecvecContour1, std::vector<std::vector<cv::Point>>* vecvecContour2, int iBlobThresh, std::mutex* mtx1, std::mutex* mtx2);
	bool GetBlobThread(cv::Mat* binaryImg, cv::Rect& RoiRt, std::vector<std::vector<cv::Point>>* vecvecContour1, std::vector<std::vector<cv::Point>>* vecvecContour2, int iBlobThresh, std::mutex* mtx1, std::mutex* mtx2);
	void MergeContour(cv::Mat& frd, std::vector<std::vector<cv::Point>>& vecTempContour, std::vector<std::vector<cv::Point>>& vecvecContour, int iBlobThresh);
	float GetContourDist(std::vector<cv::Point>& c1, std::vector<cv::Point>& c2);
	float GetContourDist(cv::Rect& rt1, cv::Rect& rt2);

	bool BlobAnalysis(cv::Mat& frdMask, std::vector<std::vector<cv::Point>>& vecvecBlob);
	bool Geo_BlobToDefectThread(cv::Mat* diffImg, std::vector<cv::Point>* contour, std::vector<DefectData>* vecDefectRect, std::mutex* mtx);
	bool Geo_BlobToDefectThread_gpu(cv::cuda::GpuMat& diffImg, std::vector<cv::Point>& contour, std::vector<DefectData>& vecDefectRect);
	void BlobCluster(std::vector<DefectData>& vecDefectInfo, float fDistThr);
	bool IsRange(cv::Point& pt_s, cv::Point& pt_e, cv::Point& pt);
	float PointToLineSegment(cv::Point& pt_s, cv::Point& pt_e, cv::Point& pt);
	float GetDistRect(cv::Rect& rt1, cv::Rect& rt2);
	void MergeDefectData(DefectData& SrcDst, DefectData& info);
	//模板函数
	template<class T1, class T2>
	int Partition_DEC(T1* R, int i, int j, T2* index)
	{
		T1 pivot = R[i];
		T2 intd = index[i];
		while (i < j)
		{
			while (i < j&&R[j] <= pivot)
				j--;
			if (i < j)
			{
				R[i] = R[j];
				index[i++] = index[j];
			}
			while (i < j&&R[i] >= pivot)
				i++;
			if (i < j)
			{
				R[j] = R[i];
				index[j--] = index[i];
			}
		}
		R[i] = pivot;
		index[i] = intd;
		return i;
	}
	template<class T1, class T2>
	void QuickSort_DEC(T1* R, int low, int high, T2* index)//由大到小
	{
		int pivotpos = 0;
		if (low < high)
		{
			pivotpos = Partition_DEC(R, low, high, index);
			QuickSort_DEC(R, low, pivotpos - 1, index);
			QuickSort_DEC(R, pivotpos + 1, high, index);
		}
	}
};

