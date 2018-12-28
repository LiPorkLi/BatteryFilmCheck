#include "stdafx.h"
#include "MeanStandardCheck.h"

CMeanStandardCheck::CMeanStandardCheck(MainParam::param* p, std::shared_ptr<CRunTimeHandle> pHandle) : CAlogrithmBase(p, pHandle)
{
	ParamHelper<Parameters::InspectParam> helper(getParam());
	Parameters::InspectParam inspectionParam = helper.getRef();
	m_iTrainCount = 0;
	m_fSigmaTime = inspectionParam.expvarparam().sigmatime();
	m_fSigmaTime = std::min(m_fSigmaTime, 3.0f);
	m_fSigmaTime = std::max(m_fSigmaTime, 0.1f);
}


CMeanStandardCheck::~CMeanStandardCheck()
{
	m_vecReturn.clear();
}

void CMeanStandardCheck::GetExpVarModel(cv::Mat& imgFream, cv::Mat& rowExp, cv::Mat& rowSigma, int iFreadIdx/* = 0*/)
{
	cv::Mat img32f;
	imgFream.convertTo(img32f, CV_32F);
	if (iFreadIdx == 0)
	{
		rowExp = cv::Mat::zeros(1, imgFream.cols, CV_32F);
		cv::reduce(img32f, rowExp, 0, CV_REDUCE_AVG);
		//imgExp.convertTo(imgExp, CV_32F, 1.0 / (double)imgFream.rows);

		cv::Mat imgsquare = img32f.mul(img32f);
		rowSigma = cv::Mat::zeros(1, imgFream.cols, CV_32F);
		cv::reduce(imgsquare, rowSigma, 0, CV_REDUCE_AVG);
		//imgSigma.convertTo(imgSigma, CV_32F, 1.0 / (double)imgFream.rows);

		rowSigma = rowSigma - rowExp.mul(rowExp);
		cv::sqrt(rowSigma, rowSigma);

		return;
	}
	cv::Mat img32fRow, tempMat1, tempMat2, tempMat3;
	int iLineCount = iFreadIdx * imgFream.rows;
	for (int i = 0; i < imgFream.rows; i++)
	{
		tempMat1 = img32f.rowRange(i, i + 1) - rowExp;
		tempMat2 = tempMat1.mul(tempMat1);
		tempMat2.convertTo(tempMat2, CV_32F, 1.0 / (double)(iLineCount));
		rowSigma.convertTo(tempMat3, CV_32F, double(iLineCount - 2) / double(iLineCount - 1));
		rowSigma = tempMat3 + tempMat2;
		tempMat1.convertTo(tempMat1, CV_32F, 1.0 / (double)(iLineCount));
		rowExp = rowExp + tempMat1;
		iLineCount++;
		cv::sqrt(rowSigma, rowSigma);
	}
}

void CMeanStandardCheck::ExtendLineToFream(cv::Size& imgSize, cv::Mat& rowImg, cv::Mat& FreamImg)
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


bool CMeanStandardCheck::TrainTemplate(cv::Mat& img, bool bIncrementTrain /*= true*/)
{
	if (bIncrementTrain == false)
	{
		m_iTrainCount = 0;
	}
	GetExpVarModel(img, m_rowExp, m_rowStd, m_iTrainCount++);
	cv::Mat tempMat;
	m_rowStd.convertTo(tempMat, CV_32F, m_fSigmaTime);
	m_rowMin = m_rowExp - tempMat;
	m_rowMax = m_rowExp + tempMat;
	ExtendLineToFream(getHandle()->ImageSizePre(), m_rowMin, m_freamMin);
	ExtendLineToFream(getHandle()->ImageSizePre(), m_rowMax, m_freamMax);

	return true;
}

bool CMeanStandardCheck::MaxMinCheckThread(cv::Mat* img, cv::Rect& rtTruth, cv::Mat* imgMin, cv::Mat* imgMax, cv::Mat* DiffImg)
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


bool CMeanStandardCheck::check(cv::Mat& img, cv::Mat& diffImg, double* dTime)
{
	*dTime = cvGetTickCount();
	//diffImg = cv::Mat::zeros(getHandle()->ImageSizePre(), CV_16S);
	for (int i = 0; i < getHandle()->DataPatchNum(); i++)
	{
		m_vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CMeanStandardCheck::MaxMinCheckThread, this, &img, getHandle()->TruthRect(i), &m_freamMin, &m_freamMax, &diffImg)));
	}
	for (int i = 0; i < m_vecReturn.size(); i++)
	{
		bool hr = m_vecReturn[i].get();
		if (hr == false)
		{
			//Å×³öÒì³£....
			printf("MeanStandard check occurred some unhappy! Info: Rect: x = %d, y= %d, width = %d, height=%d\n", getHandle()->TruthRect(i).x, getHandle()->TruthRect(i).y, getHandle()->TruthRect(i).width, getHandle()->TruthRect(i).height);
			return false;
		}
	}
	m_vecReturn.clear();
	*dTime = (cv::getTickCount() - (*dTime)) / (1000 * cvGetTickFrequency());
	return true;
}

void CMeanStandardCheck::SetParam(void* param)
{

}
