#include "stdafx.h"
#include "MaxMinValueCheck.h"

CMaxMinValueCheck::CMaxMinValueCheck(MainParam::param* p, std::shared_ptr<CRunTimeHandle> pHandle) : CAlogrithmBase(p, pHandle)
{

	ParamHelper<Parameters::InspectParam> helper(getParam());
	Parameters::InspectParam inspectionParam = helper.getRef();
	//ParamHelper<Parameters::SystemRectifyParam> sysHelper(p);
	//Parameters::SystemRectifyParam sysParam = sysHelper.getRef();
	//float fPhysicResolution = sysParam.physicresolution();
	//int iDownSampleParam = m_pHandle->DownSampleFator();

	m_iTrainCount = 0;
	m_iThreshold_low = std::max(inspectionParam.maxminparam().threshlow(), 10);
	m_iThreshold_high = std::max(inspectionParam.maxminparam().threshigh(), 10);

	m_iThreshold_low = std::min(m_iThreshold_low, 110);
	m_iThreshold_high = std::min(m_iThreshold_high, 110);
}


CMaxMinValueCheck::~CMaxMinValueCheck()
{
	m_vecReturn.clear();
}

void CMaxMinValueCheck::GetMaxMinModel(cv::Mat& imgFream, cv::Mat& rowMin, cv::Mat& rowMax, int iFreadIdx/* = 0*/)
{
	uchar* pBuff = imgFream.data;
	if (iFreadIdx == 0)
	{
		rowMax = cv::Mat::zeros(1, imgFream.cols, CV_8U);
		rowMin = rowMax.clone();
		rowMin.setTo(0xff);
	}
	uchar* pMin = rowMin.data;
	uchar* pMax = rowMax.data;
	int iStep = 0;
	for (int i = 0; i < imgFream.rows; i++)
	{
		for (int j = 0; j < imgFream.cols; j++)
		{
			pMin[j] = std::min(pMin[j], pBuff[iStep + j]);
			pMax[j] = std::max(pMax[j], pBuff[iStep + j]);
		}
		iStep += imgFream.cols;
	}
}

void CMaxMinValueCheck::ExtendLineToFream(cv::Size& imgSize, cv::Mat& rowImg, cv::Mat& FreamImg)
{
	FreamImg = cv::Mat();
	if (imgSize.width != rowImg.cols || rowImg.rows != 1)
	{
		return;
	}
	FreamImg = cv::Mat(imgSize, rowImg.type());
	for (int i = 0; i < imgSize.height; i++)
	{
		rowImg.copyTo(FreamImg.rowRange(i, i + 1));
	}
}


bool CMaxMinValueCheck::TrainTemplate(cv::Mat& img, bool bIncrementTrain /*= true*/)
{
	if (bIncrementTrain == false)
	{
		m_iTrainCount = 0;
	}

	GetMaxMinModel(img, m_rowMin, m_rowMax, m_iTrainCount++);
	ExtendLineToFream(getHandle()->ImageSizePre(), m_rowMin, m_freamMin);
	ExtendLineToFream(getHandle()->ImageSizePre(), m_rowMax, m_freamMax);
	m_freamMin.convertTo(m_freamMin, CV_8U, 1, -m_iThreshold_low);
	m_freamMax.convertTo(m_freamMax, CV_8U, 1, m_iThreshold_high);

	return true;
}

bool CMaxMinValueCheck::MaxMinCheckThread(cv::Mat* img, cv::Rect& rtTruth, cv::Mat* imgMin, cv::Mat* imgMax, cv::Mat* DiffImg)
{
	if (DiffImg->cols != img->cols || DiffImg->type() != CV_16S || DiffImg->rows != img->rows || img->size != imgMin->size || img->size != imgMax->size)
	{
		return false;
	}
	cv::Mat PatchImg, tempMin, tempMax;
	(*img)(rtTruth).convertTo(PatchImg, CV_16S);
	(*imgMin)(rtTruth).convertTo(tempMin, CV_16S);
	(*imgMax)(rtTruth).convertTo(tempMax, CV_16S);
	cv::Mat diffMat_min = PatchImg - tempMin;
	cv::Mat diffMat_max = PatchImg - tempMax;
	cv::Mat mask_min = (diffMat_min < 0);
	cv::Mat mask_max = (diffMat_max > 0);
	PatchImg.setTo(0x00);
	diffMat_max.copyTo(PatchImg, mask_max);
	diffMat_min.copyTo(PatchImg, mask_min);
	PatchImg.copyTo((*DiffImg)(rtTruth));
	return true;
}


bool CMaxMinValueCheck::check(cv::Mat& img, cv::Mat& diffImg, double* dTime)
{
	*dTime = cvGetTickCount();
	//diffImg = cv::Mat::zeros(getHandle()->ImageSizePre(), CV_16S);
	for (int i = 0; i < getHandle()->DataPatchNum(); i++)
	{
		m_vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CMaxMinValueCheck::MaxMinCheckThread, this, &img, getHandle()->TruthRect(i), &m_freamMin, &m_freamMax, &diffImg)));
	}
	for (int i = 0; i < m_vecReturn.size(); i++)
	{
		bool hr = m_vecReturn[i].get();
		if (hr == false)
		{
			//Å×³öÒì³£....
			printf("MaxMin check occurred some unhappy! Info: Rect: x = %d, y= %d, width = %d, height=%d\n", getHandle()->TruthRect(i).x, getHandle()->TruthRect(i).y, getHandle()->TruthRect(i).width, getHandle()->TruthRect(i).height);
			return false;
		}
	}
	m_vecReturn.clear();
	*dTime = (cv::getTickCount() - (*dTime)) / (1000 * cvGetTickFrequency());
	return true;
}

void CMaxMinValueCheck::SetParam(void* param)
{

}
