#include "stdafx.h"
#include "ImageIO.h"
#include <fstream>

CImageIO::CImageIO()
{
	m_vecResultImg.clear();
	m_vecResultIndex.clear();
	m_vecResultHead.clear();
}

CImageIO::~CImageIO()
{
	m_vecResultImg.clear();
	m_vecResultIndex.clear();
	m_vecResultHead.clear();
}

void CImageIO::Reset()
{
	m_vecResultImg.clear();
	m_vecResultIndex.clear();
	m_vecResultHead.clear();
}

void CImageIO::SetDataToFile(ImageInspectResult& rlt)
{
	std::pair<int, int> rltIdx;
	rltIdx.first = rlt.idx;
	std::pair<cv::Mat, cv::Mat> rltImg;

	std::vector<int> vecIndex2;
	vecIndex2.clear();
	int iTemp = (int)m_vecResultIndex.size();
	ImageHead h;
	for (int i = 0;i<rlt.m_vecDefectList.size(); i++)
	{
		rltIdx.second = i;
		rltImg.first = rlt.srcImage(rlt.m_vecDefectList[i].imgRect).clone();
		//rltImg.second = rlt.diffImage(rlt.m_vecDefectList[i].imgRect).clone();
		rltImg.second = rlt.GetMask(i);

		if (rltImg.first.cols>100 || rltImg.first.rows>100)
		{
			cv::resize(rltImg.first, rltImg.first, cv::Size(100, 100), 0, 0, cv::INTER_NEAREST);
			cv::resize(rltImg.second, rltImg.second, cv::Size(100, 100), 0, 0, cv::INTER_NEAREST);
		}

		m_vecResultIndex.push_back(rltIdx);
		m_vecResultImg.push_back(rltImg);

		h.iImgIdx = rltIdx.first;
		h.iDefectIdx = rltIdx.second;
		h.iSrcImgType = rltImg.first.type();
		h.iDefectImgType = rltImg.first.type();
		h.iRows = rltImg.first.rows;
		h.iCols = rltImg.first.cols;
		h.uiSrcLength = rltImg.first.rows * rltImg.first.step[0];
		h.uiDefectLength = rltImg.second.rows * rltImg.second.step[0];
		h.fPy_area = rlt.m_vecDefectList[i].fPyArea;
		h.fPy_height = rlt.m_vecDefectList[i].fPy_height;
		h.fPy_width = rlt.m_vecDefectList[i].fPy_width;
		h.iMeanDiff = rlt.m_vecDefectList[i].iMeanDiff;
		h.fPy_imgWidth = rlt.m_vecDefectList[i].fPy_imgWidth;
		h.fPy_imgHeight = rlt.m_vecDefectList[i].fPy_imgHeight;
		strcpy(h.defectName, rlt.m_vecDefectList[i].defectName.c_str());

		m_vecResultHead.push_back(h);

		vecIndex2.push_back(iTemp);
		iTemp++;
	}

	SaveBuff(m_vecResultIndex, m_vecResultImg, m_vecResultHead, vecIndex2);
	vecIndex2.clear();
}

bool CImageIO::GetDataFromBuff(std::pair<int, int>& ResultIndex, std::pair<cv::Mat, cv::Mat>& ResultImg)
{
	if (m_vecResultImg.size() == 0 || m_vecResultIndex.size() == 0 || m_vecResultImg.size() != m_vecResultIndex.size())
	{
		return false;
	}

	for (int i = 0; i < m_vecResultIndex.size(); i++)
	{
		if (ResultIndex==m_vecResultIndex[i])
		{
			ResultImg = m_vecResultImg[i];
			return true;
		}
	}
	return false;
}

bool CImageIO::GetDataFromBuff(std::vector<std::pair<int, int>>& vecResultIndex, std::vector<std::pair<cv::Mat, cv::Mat>>& vecResultImg)
{
	vecResultIndex.clear();
	vecResultImg.clear();
	if (m_vecResultImg.size() == 0 || m_vecResultIndex.size() == 0 || m_vecResultImg.size() != m_vecResultIndex.size())
	{
		return false;
	}
	vecResultIndex = m_vecResultIndex;
	vecResultImg = m_vecResultImg;
	return true;
}

bool CImageIO::GetDataFromBuff(std::pair<int, int>& ResultIndex, std::pair<cv::Mat, cv::Mat>& ResultImg, std::string& ResultDefectName)
{
	if (m_vecResultImg.size() == 0 || m_vecResultIndex.size() == 0 || m_vecResultImg.size() != m_vecResultIndex.size())
	{
		return false;
	}

	for (int i = 0; i < m_vecResultIndex.size(); i++)
	{
		if (ResultIndex == m_vecResultIndex[i])
		{
			ResultImg = m_vecResultImg[i];
			ResultDefectName = std::string(m_vecResultHead[i].defectName);
			return true;
		}
	}
	return false;
}

bool CImageIO::GetDataFromBuff(std::vector<std::pair<int, int>>& vecResultIndex, std::vector<std::pair<cv::Mat, cv::Mat>>& vecResultImg, std::vector<std::string>& vecResultDefectName)
{
	vecResultIndex.clear();
	vecResultImg.clear();
	if (m_vecResultImg.size() == 0 || m_vecResultIndex.size() == 0 || m_vecResultImg.size() != m_vecResultIndex.size())
	{
		return false;
	}
	vecResultIndex = m_vecResultIndex;
	vecResultImg = m_vecResultImg;

	vecResultDefectName.clear();
	for (int i = 0; i < m_vecResultHead.size(); i++)
	{
		vecResultDefectName.push_back(std::string(m_vecResultHead[i].defectName));
	}

	return true;
}

bool CImageIO::GetDataFromBuff(std::pair<int, int>& ResultIndex, std::pair<cv::Mat, cv::Mat>& ResultImg, ImageHead& ResultHead)
{
	if (m_vecResultImg.size() == 0 || m_vecResultIndex.size() == 0 || m_vecResultImg.size() != m_vecResultIndex.size())
	{
		return false;
	}

	for (int i = 0; i < m_vecResultIndex.size(); i++)
	{
		if (ResultIndex == m_vecResultIndex[i])
		{
			ResultImg = m_vecResultImg[i];
			ResultHead = m_vecResultHead[i];
			return true;
		}
	}
	return false;
}

bool CImageIO::GetDataFromBuff(std::vector<std::pair<int, int>>& vecResultIndex, std::vector<std::pair<cv::Mat, cv::Mat>>& vecResultImg, std::vector<ImageHead>& vecResultHead)
{
	vecResultIndex.clear();
	vecResultImg.clear();
	vecResultHead.clear();
	if (m_vecResultImg.size() == 0 || m_vecResultIndex.size() == 0 || m_vecResultImg.size() != m_vecResultIndex.size())
	{
		return false;
	}
	vecResultIndex = m_vecResultIndex;
	vecResultImg = m_vecResultImg;
	vecResultHead = m_vecResultHead;


	return true;
}

bool CImageIO::ReadData(char* filepath)
{
	FILE* fr = fopen(filepath, "rb");
	if (fr == NULL)
	{
		return false;
	}

	int iNum = 0;
	fread(&iNum, sizeof(int), 1, fr);
	m_vecResultImg.resize(iNum);
	m_vecResultIndex.resize(iNum);
	m_vecResultHead.resize(iNum);
	for (int i = 0; i < iNum; i++)
	{
		ImageHead h;
		fread(&h, sizeof(ImageHead), 1, fr);
		m_vecResultIndex[i].first = h.iImgIdx;
		m_vecResultIndex[i].second = h.iDefectIdx;
		m_vecResultImg[i].first = cv::Mat(h.iRows, h.iCols, h.iSrcImgType);
		m_vecResultImg[i].second = cv::Mat(h.iRows, h.iCols, h.iDefectImgType);
		m_vecResultHead[i] = h;
		fread(m_vecResultImg[i].first.data, sizeof(uchar), m_vecResultImg[i].first.rows*m_vecResultImg[i].first.step[0], fr);
		fread(m_vecResultImg[i].second.data, sizeof(uchar), m_vecResultImg[i].second.rows*m_vecResultImg[i].second.step[0], fr);
	}

	fclose(fr);
	return true;
}

void CImageIO::SetSavePath(char* pSavePath)
{
	sprintf_s(m_SavePath, "%s", pSavePath);
}

bool CImageIO::SaveBuff(std::vector<std::pair<int, int>>& vecIndex, std::vector<std::pair<cv::Mat, cv::Mat>>& vecImg, std::vector<ImageHead>& vecHead, std::vector<int>& vecIndex2)
{
	//FILE* fr = fopen(m_SavePath, "a+");
	//fseek(fr, 0, SEEK_END);
	//int size = ftell(fr);

	int iNum = vecIndex2.size();
	std::fstream fr(m_SavePath, std::ios::in | std::ios::out | std::ios::binary);
	fr.seekg(0, fr.end);
	int size = fr.tellg();
	if (size == -1)
	{
		fr.open(m_SavePath, std::ios::out | std::ios::binary);
		fr.close();
		fr.open(m_SavePath, std::ios::in | std::ios::out | std::ios::binary);
		fr.seekp(0, fr.beg);
		fr.write((char*)(&iNum), sizeof(int));
	}
	else
	{
		fr.seekg(0, fr.beg);
		int iTemp = 0;
		fr.read((char*)(&iTemp), sizeof(int));
		iNum += iTemp;
		fr.seekp(0, fr.beg);
		fr.write((char*)(&iNum), sizeof(int));
	}

	fr.seekp(0, fr.end);
	for (int k = 0; k < vecIndex2.size(); k++)
	{
		int i = vecIndex2[k];
		
		fr.write((char*)(&vecHead[i]), sizeof(ImageHead));
		fr.write((char*)(vecImg[i].first.data), sizeof(uchar)*vecImg[i].first.rows*vecImg[i].first.step[0]);
		fr.write((char*)(vecImg[i].second.data), sizeof(uchar)*vecImg[i].second.rows*vecImg[i].second.step[0]);
	}

	fr.close();

	return true;
}
// 
// void CImageIO::FromImageToBin(char* pImagePath, char* pBinPatch)
// {
// 
// }
