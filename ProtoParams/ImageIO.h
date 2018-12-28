#pragma once
#include "cv.hpp"
#include "highgui.hpp"
#include "GlobalData.h"
class CImageIO
{
public:
	CImageIO();
	~CImageIO();

public:
	struct ImageHead
	{
		int iImgIdx, iDefectIdx;
		int iSrcImgType, iDefectImgType;
		int iRows, iCols;
		unsigned int uiSrcLength;
		unsigned int uiDefectLength;
		char defectName[32];
		float fPy_width, fPy_height, fPy_area;
		int iMeanDiff;
		float fPy_imgWidth, fPy_imgHeight;
	};

	void SetSavePath(char* pSavePath);
	void Reset();

	void SetDataToFile(ImageInspectResult& rlt);
	bool GetDataFromBuff(std::pair<int, int>& ResultIndex, std::pair<cv::Mat, cv::Mat>& ResultImg);
	bool GetDataFromBuff(std::pair<int, int>& ResultIndex, std::pair<cv::Mat, cv::Mat>& ResultImg, std::string& ResultDefectName);
	bool GetDataFromBuff(std::pair<int, int>& ResultIndex, std::pair<cv::Mat, cv::Mat>& ResultImg, ImageHead& ResultHead);
	bool GetDataFromBuff(std::vector<std::pair<int, int>>& vecResultIndex, std::vector<std::pair<cv::Mat, cv::Mat>>& vecResultImg);
	bool GetDataFromBuff(std::vector<std::pair<int, int>>& vecResultIndex, std::vector<std::pair<cv::Mat, cv::Mat>>& vecResultImg, std::vector<std::string>& vecResultDefectName);
	bool GetDataFromBuff(std::vector<std::pair<int, int>>& vecResultIndex, std::vector<std::pair<cv::Mat, cv::Mat>>& vecResultImg, std::vector<ImageHead>& vecResultHead);
	bool ReadData(char* filepath);

	//void FromImageToBin(char* pImagePath, char* pBinPatch);
private:
	//std::vector<uchar> m_vecImgList;
	int m_iDefectNum;
	

	std::vector<std::pair<int, int>> m_vecResultIndex;
	std::vector<std::pair<cv::Mat, cv::Mat>> m_vecResultImg;
	std::vector<ImageHead> m_vecResultHead;

	char m_SavePath[256];

	bool SaveBuff(std::vector<std::pair<int, int>>& vecIndex, std::vector<std::pair<cv::Mat, cv::Mat>>& vecImg, std::vector<ImageHead>& vecHead, std::vector<int>& vecIndex2);
};

