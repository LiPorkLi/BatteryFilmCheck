#include "stdafx.h"
#include "Classify.h"


CClassify::CClassify(MainParam::param* p, std::shared_ptr<CRunTimeHandle> pHandle) : CAlogrithmBase(p, pHandle)
{
	ParamHelper<Parameters::InspectParam> helper(getParam());
	Parameters::InspectParam inspectionParam = helper.getRef();
	//m_modelPath = inspectionParam.modelpath();
	m_modelPath = inspectionParam.caffeparam().modelpath();
	m_iProdctType = inspectionParam.layertype();

	m_pCls = CreateXCClsInstance(m_modelPath.c_str(), true);

	m_iCountImg = 0;
	m_iCountDefect = 0;
	m_iLastIndex = -1;
}


CClassify::~CClassify()
{
	ReleaseXCClsInstance(&m_pCls);
}

void CClassify::GetStatus(cv::Mat& src0, cv::Mat& diff0, DefectData& defect, Extr_Info& p)
{
	cv::Mat binBackground, bin2, bin1;
	cv::threshold(src0, binBackground, 200, 0xff, cv::THRESH_BINARY_INV);
	cv::threshold(src0, bin1, 10, 0xff, cv::THRESH_BINARY);
	{
		cv::Mat tempMat = ~binBackground;
		tempMat = tempMat | (~bin1);
		cv::Scalar avg = cv::sum(tempMat);
		avg.val[0] = avg.val[0] / 255.0;
		p.fNoValidPer = avg.val[0] / double(src0.cols*src0.rows);
		if (p.fNoValidPer > 0.7f)
		{
			//defect.defectName = std::string("无产品");
			defect.defectName = std::string("误报");
			return;
		}
	}

	binBackground = binBackground & bin1;
	bin2 = ~diff0;
	binBackground = binBackground & bin2;
	cv::Scalar avg_back = cv::mean(src0, binBackground);
	cv::Scalar avg_defect = cv::mean(src0, diff0);

	if (defect.iMeanDiff < 0)
	{
		double d = 0;
		cv::Mat tempMat(src0.rows, src0.cols, CV_8U);
		tempMat.setTo(0xff);
		src0.copyTo(tempMat, diff0);
		cv::minMaxLoc(tempMat, &d);
		p.fMaxMinDiff = d - avg_back.val[0];
		p.fAvgDiff = avg_defect.val[0] - avg_back.val[0];
	}
	else
	{
		double d = 0;
		cv::Mat tempMat(src0.rows, src0.cols, CV_8U);
		tempMat.setTo(0x00);
		src0.copyTo(tempMat, diff0);
		cv::minMaxLoc(tempMat, NULL, &d);
		p.fMaxMinDiff = d - avg_back.val[0];
		p.fAvgDiff = avg_defect.val[0] - avg_back.val[0];
	}

	cv::Mat grad_x, grad_y;
	cv::Mat abs_grad_x, abs_grad_y;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	/*std::vector<std::vector<cv::Point>> vecvecContour;
	{
		cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
		cv::dilate(diff0, bin1, k);
		bin2 = bin1.clone();
		cv::findContours(bin2, vecvecContour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		if (vecvecContour.size()==0)
		{
			p.fAvgMangnitude = 0;
			return;
		}
		int iMaxIndex = 0;
		double dMaxArea = fabs(cv::contourArea(vecvecContour[0]));
		for (int i = 1; i < vecvecContour.size(); i++)
		{
			double d = fabs(cv::contourArea(vecvecContour[i]));
			if (dMaxArea < d)
			{
				dMaxArea = d;
				iMaxIndex = i;
			}
		}

		cv::Rect rt = cv::boundingRect(vecvecContour[iMaxIndex]);
		cv::Mat src1 = src0(rt).clone();
		cv::Mat diff1 = bin1(rt).clone();
		Sobel(src1, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);

		/// 求Y方向梯度
		//Scharr(src0, grad_y, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT);
		Sobel(src1, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);
		convertScaleAbs(grad_y, abs_grad_y);

		/// 合并梯度(近似)
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad_x);
		cv::threshold(grad_x, bin2, 10, 0xff, cv::THRESH_TOZERO);
		bin2 = bin2 & diff1;
		cv::Scalar dsum = cv::mean(grad_x, bin2);
		p.fAvgMangnitude = dsum.val[0];
	}
	for (int i = 0; i < vecvecContour.size(); i++)
	{
		vecvecContour[i].clear();
	}
	vecvecContour.clear();*/

	/// 求 X方向梯度
	//Scharr( src0, grad_x, ddepth, 1, 0, scale, delta, cv::BORDER_DEFAULT );
	Sobel(src0, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	/// 求Y方向梯度
	//Scharr(src0, grad_y, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT);
	Sobel(src0, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	/// 合并梯度(近似)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad_x);
	//cv::threshold(grad_x, bin2, 10, 0xff, cv::THRESH_TOZERO);
	binBackground = binBackground | diff0;
	cv::threshold(grad_x, bin2, 10, 0xff, cv::THRESH_TOZERO);
	binBackground = binBackground & bin2;
	cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	cv::dilate(diff0, bin1, k);
	binBackground = binBackground & bin1;

	cv::Scalar dsum = cv::mean(grad_x, binBackground);

	p.fAvgMangnitude = dsum.val[0];
}


int CClassify::Clssify(std::shared_ptr<ImageInspectResult> buff, double* dTime)
{
	*dTime = cvGetTickCount();
	int iNum = buff->m_vecDefectList.size();
	if (iNum==0)
	{
		*dTime = (cvGetTickCount() - *dTime) / (1000 * cvGetTickFrequency());
		return 0;
	}

	std::vector<DefectData> vecTempDefect, vecNoProdctDefect;
	vecTempDefect.clear();
	vecNoProdctDefect.clear();

	std::vector<cv::Mat> vecSrc;
	vecSrc.clear();

	for (int i = 0; i < iNum; i++)
	{
		cv::Mat src = buff->srcImage(buff->m_vecDefectList[i].imgRect);
		/*cv::Mat diff = buff->GetMask(i);
		Extr_Info p;
		GetStatus(src, diff, buff->m_vecDefectList[i], p);
		if (buff->m_vecDefectList[i].defectName == std::string("无产品"))
		{
			vecNoProdctDefect.push_back(buff->m_vecDefectList[i]);
			continue;
		}
		if (buff->m_vecDefectList[i].iMeanDiff >= -2 && buff->m_vecDefectList[i].iMeanDiff<3 && 
			buff->m_vecDefectList[i].fPyArea < 5 && p.fAvgMangnitude < 25)
		{
			continue;
		}*/
		vecTempDefect.push_back(buff->m_vecDefectList[i]); 
		vecSrc.push_back(src.clone());
	}
	if (vecTempDefect.size()!=0)
	{
		std::vector<std::string> vecRlt;
		m_pCls->GetCls(vecSrc, vecRlt);
		for (int i = 0; i < vecTempDefect.size(); i++)
		{
			if (m_iProdctType == 1 && vecRlt[i] == std::string("亮斑杂质"))
			{
				vecRlt[i] = std::string("气泡");
			}
			vecTempDefect[i].defectName = vecRlt[i];
		}
		vecRlt.clear();
	}
	buff->m_vecDefectList.swap(vecTempDefect);
	buff->m_vecDefectList.insert(buff->m_vecDefectList.end(), vecNoProdctDefect.begin(), vecNoProdctDefect.end());

	vecSrc.clear();
	vecTempDefect.clear();
	vecNoProdctDefect.clear();

	int hr = 0;
	float fSumArea = 0.0;
	for (int i = 0; i < buff->m_vecDefectList.size(); i++)
	{
		fSumArea += buff->m_vecDefectList[i].fPyArea;
	}
	if (buff->m_vecDefectList.size()>50 || fSumArea > 1000)
	{
		hr = 1;
	}
	else if ((buff->idx - m_iLastIndex == 1) && (buff->m_vecDefectList.size() != 0))
	{
		m_iCountImg++;
		m_iCountDefect += buff->m_vecDefectList.size();

		if (m_iCountDefect > 200 && m_iCountImg > 10)//连续10张有缺陷或者总缺陷大于200,应报警
		{
			hr = 2;

			std::stringstream ss;
			ss.clear();
			ss.str("");
			ss << "........................Conunt defect:" << m_iCountImg << " ,defectNum:" << m_iCountDefect << std::endl;
			getHandle()->PushLog(ss.str());
		}
	}
	else
	{
		m_iCountDefect = 0;
		m_iCountImg = 0;
	}

	m_iLastIndex = buff->idx;
	*dTime = (cvGetTickCount() - *dTime) / (1000 * cvGetTickFrequency());
	return hr;
}

bool CClassify::Clssify(ImageInspectResult* buff, double* dTime)
{
	*dTime = cvGetTickCount();
	int iNum = buff->m_vecDefectList.size();
	if (iNum == 0)
	{
		*dTime = (cvGetTickCount() - *dTime) / (1000 * cvGetTickFrequency());
		return true;
	}
	std::vector<cv::Mat> vecSrc;
	vecSrc.resize(iNum);
	for (int i = 0; i < iNum; i++)
	{
		vecSrc[i] = buff->srcImage(buff->m_vecDefectList[i].imgRect).clone();

	}
	std::vector<std::string> vecRlt;
	m_pCls->GetCls(vecSrc, vecRlt);
	if (vecRlt.size() != iNum)
	{
		return false;
	}
	for (int i = 0; i < iNum; i++)
	{
		if (m_iProdctType == 1 && vecRlt[i]==std::string("亮斑杂质"))
		{
			vecRlt[i] = std::string("气泡");
		}
		buff->m_vecDefectList[i].defectName = vecRlt[i];
	}
	vecSrc.clear();
	vecRlt.clear();
	*dTime = (cvGetTickCount() - *dTime) / (1000 * cvGetTickFrequency());
	return true;
}
