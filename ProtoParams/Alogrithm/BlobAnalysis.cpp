#include "stdafx.h"
#include "BlobAnalysis.h"
#include <imgproc.hpp>
//#include "core\base.hpp"
#define _MIN_SHOW_LEN_ 100
#define _DEFECT_BODER_SIZE_ 20

CBlobAnalysis::CBlobAnalysis(MainParam::param* p, std::shared_ptr<CRunTimeHandle> pHandle) : CAlogrithmBase(p, pHandle)
{
	//CAlogrithm::CAlogrithm(p, pHandle);
//	m_pHandle = pHandle;
	ParamHelper<Parameters::InspectParam> helper(getParam());
	Parameters::InspectParam inspectionParam = helper.getRef();

	int iDownSampleParam = getHandle()->DownSampleFator();

	//Blob Filter Param
	/*m_iBlobThrsh = int(inspectionParam.defectfilter().blobthr() * fPhysicResolution * fPhysicResolution + 0.5f);
	m_iBoundaryOffset = int(inspectionParam.defectfilter().boundoffset() * fPhysicResolution + 0.5f);
	if (iDownSampleParam != 0)
	{
		m_iBlobThrsh = m_iBlobThrsh >> (iDownSampleParam + 1);
		m_iBoundaryOffset = m_iBoundaryOffset >> iDownSampleParam;
	}*/
	m_iBlobThrsh = int(PyhsicToPixel_2D(inspectionParam.blobthr(), getHandle()->PhysicResolution_x(),getHandle()->PhysicResolution_y(), getHandle()->DownSampleFator()) + 0.5f);
	
	if (getHandle()->IsFromLeftToRight()==true)
	{
		m_rtLeftSetNoCheck.x = 0;
		m_rtLeftSetNoCheck.y = 0;
		m_rtLeftSetNoCheck.width = std::max(0, int(PyhsicToPixel_1D(getHandle()->GetStripOffset(), getHandle()->PhysicResolution_x(), getHandle()->DownSampleFator()) + 0.5f));
		m_rtLeftSetNoCheck.width = std::min(m_rtLeftSetNoCheck.width, getHandle()->ImageSizePre().width);
		m_rtLeftSetNoCheck.height = getHandle()->ImageSizePre().height;

		//float fAllLenth = getHandle()->GetStripOffset() + getHandle()->GetStripAllLength();
		//int iX = int(PyhsicToPixel_1D((getHandle()->GetStripOffset() + getHandle()->GetStripAllLength()), getHandle()->PhysicResolution_x(), getHandle()->DownSampleFator()) + 0.5f);
		m_rtRightSetNoCheck.x = std::min(int(PyhsicToPixel_1D((getHandle()->GetStripOffset() + getHandle()->GetStripAllLength()), getHandle()->PhysicResolution_x(), getHandle()->DownSampleFator()) + 0.5f), getHandle()->ImageSizePre().width-1);
		m_rtRightSetNoCheck.x = std::max(0, m_rtRightSetNoCheck.x);
		m_rtRightSetNoCheck.y = 0;
		m_rtRightSetNoCheck.width = getHandle()->ImageSizePre().width - m_rtRightSetNoCheck.x;
		m_rtRightSetNoCheck.height = getHandle()->ImageSizePre().height;
	}
	else
	{
		m_rtLeftSetNoCheck.x = 0;
		m_rtLeftSetNoCheck.y = 0;
		m_rtLeftSetNoCheck.width = std::max(0, getHandle()->ImageSizePre().width - int(PyhsicToPixel_1D((getHandle()->GetStripOffset() + getHandle()->GetStripAllLength()), getHandle()->PhysicResolution_x(), getHandle()->DownSampleFator()) + 0.5f));
		m_rtLeftSetNoCheck.width = std::min(m_rtLeftSetNoCheck.width, getHandle()->ImageSizePre().width);
		m_rtLeftSetNoCheck.height = getHandle()->ImageSizePre().height;

		m_rtRightSetNoCheck.x = std::min(getHandle()->ImageSizePre().width - int(PyhsicToPixel_1D(getHandle()->GetStripOffset(), getHandle()->PhysicResolution_x(), getHandle()->DownSampleFator()) + 0.5f), getHandle()->ImageSizePre().width - 1);
		m_rtRightSetNoCheck.x = std::max(0, m_rtRightSetNoCheck.x);
		m_rtRightSetNoCheck.y = 0;
		m_rtRightSetNoCheck.width = getHandle()->ImageSizePre().width - m_rtRightSetNoCheck.x;
		m_rtRightSetNoCheck.height = getHandle()->ImageSizePre().height;
	}

	ParamHelper<Parameters::SheetInfo> shelper(getParam());
	Parameters::SheetInfo sheetParam = shelper.getRef();
	m_iBoundaryOffsetLeft = int(PyhsicToPixel_1D(sheetParam.boundoffsetleft(), getHandle()->PhysicResolution_x(), getHandle()->DownSampleFator()) + 0.5f);
	m_iBoundaryOffsetRight = int(PyhsicToPixel_1D(sheetParam.boundoffsetright(), getHandle()->PhysicResolution_x(), getHandle()->DownSampleFator()) + 0.5f);
	
	std::stringstream ss;
	ss.clear(); ss.str("");
	ss << "Boundary Offset: left: " << m_iBoundaryOffsetLeft << " right: " << m_iBoundaryOffsetRight <<"\n";
	getHandle()->PushLog(ss.str());
	
	if (getHandle()->IsGpu()==true)
	{
// 		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size((m_iBoundaryOffset << 1) + 1, 1));
// 		m_f_erode = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, kernel);
	}


	m_vecPt1.resize(4);
	m_vecPt2.resize(4);


	cv::Size SplitSize, PaddingSize, TempSize;
	TempSize.width = getHandle()->ImageSizePre().width/* >> 1*/;
	TempSize.height = getHandle()->ImageSizePre().height/* >> 1*/;
	//m_tempMark = cv::Mat(TempSize, CV_8U);

	SplitSize.width = (TempSize.width >> 1);
	SplitSize.height = (TempSize.height >> 1);

	PaddingSize.width = std::min(5, SplitSize.width >> 1);
	PaddingSize.height = std::min(5, SplitSize.height >> 1);
	getHandle()->SplitImg(TempSize, SplitSize, PaddingSize, m_vecSplit, m_vecTruth);

	m_tempMark = cv::Mat(TempSize, CV_32S);
}



CBlobAnalysis::~CBlobAnalysis()
{
	m_vecReturn.clear();
	m_vecSplit.clear();
	m_vecTruth.clear();
	m_vecPt2.clear();
	m_vecPt1.clear();

	for (auto&& pt : m_vecvecBlob)
	{
		pt.clear();
	}
	m_vecvecBlob.clear();
}

/*bool CBlobAnalysis::BlobAnalysis(cv::Mat& frdMask, std::vector<std::vector<cv::Point>>& vecvecBlob, double* dTime)
{ 
	*dTime = cvGetTickCount();
	if (getHandle()->ImageSizePre() != frdMask.size())
	{
		return false;
	}
	//cv::resize(frdMask, m_tempMark, cv::Size(frdMask.cols >> 1, frdMask.rows >> 1), 0, 0, cv::INTER_NEAREST);
	for (auto&& pt : vecvecBlob)
	{
		pt.clear();
	}
	vecvecBlob.clear();


	std::vector<std::vector<cv::Point>> tempContours;
	tempContours.clear();
	m_vecReturn.clear();
	std::mutex mtx1, mtx2;
	for (int i = 0; i < m_vecSplit.size(); i++)
	{
		m_vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CBlobAnalysis::GetBlobThread, this, &frdMask, m_vecSplit[i], &vecvecBlob, &tempContours, m_iBlobThrsh, &mtx1, &mtx2)));
	}

	for (int i = 0; i < m_vecReturn.size(); i++)
	{
		bool hr = m_vecReturn[i].get();
		if (hr == false)
		{
			//抛出异常....
			printf("Blob analysis occurred some unhappy! Info: Rect: x = %d, y= %d, width = %d, height=%d\n", m_vecSplit[i].x, m_vecSplit[i].y, m_vecSplit[i].width, m_vecSplit[i].height);
			return false;
		}
	}
	m_vecReturn.clear();
	MergeContour(frdMask, tempContours, vecvecBlob, m_iBlobThrsh);
	for (int i = 0; i < tempContours.size(); i++)
	{
		tempContours[i].clear();
	}
	tempContours.clear();


	*dTime = (cv::getTickCount() - (*dTime)) / (1000 * cvGetTickFrequency());
	return true;

}*/

bool CBlobAnalysis::BlobAnalysis(cv::Mat& frdMask, std::vector<std::vector<cv::Point>>& vecvecBlob)
{
	if (getHandle()->ImageSizePre() != frdMask.size())
	{
		return false;
	}

	for (auto&& pt : vecvecBlob)
	{
		pt.clear();
	}
	vecvecBlob.clear();

	std::vector<std::vector<cv::Point>> tempContours;
	tempContours.clear();
	m_vecReturn.clear();
	std::mutex mtx1, mtx2;
	for (int i = 0; i < m_vecSplit.size(); i++)
	{
		m_vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CBlobAnalysis::GetBlobThread, this, &frdMask, m_vecSplit[i], &vecvecBlob, &tempContours, m_iBlobThrsh, &mtx1, &mtx2)));
	}

	for (int i = 0; i < m_vecReturn.size(); i++)
	{
		bool hr = m_vecReturn[i].get();
		if (hr == false)
		{
			//抛出异常....
			printf("Blob analysis occurred some unhappy! Info: Rect: x = %d, y= %d, width = %d, height=%d\n", m_vecSplit[i].x, m_vecSplit[i].y, m_vecSplit[i].width, m_vecSplit[i].height);
			return false;
		}
	}
	m_vecReturn.clear();
	MergeContour(frdMask, tempContours, vecvecBlob, m_iBlobThrsh);
	for (int i = 0; i < tempContours.size(); i++)
	{
		tempContours[i].clear();
	}
	tempContours.clear();

	return true;
}

bool CBlobAnalysis::Geo_BlobToDefectThread_gpu(cv::cuda::GpuMat& diffImg, std::vector<cv::Point>& contour, std::vector<DefectData>& vecDefectRect)
{
	if (contour.size() == 0 || diffImg.size() != getHandle()->ImageSizePre())
	{
		return false;
	}

	cv::Rect rt = cv::boundingRect(contour);
	double dArea = fabs(cv::contourArea(contour));
	cv::Mat mask = cv::Mat::zeros(rt.height, rt.width, CV_8U);
	std::vector<std::vector<cv::Point>> tempContour(1);
	tempContour[0].resize(contour.size());
	for (int i = 0; i < contour.size(); i++)
	{
		tempContour[0][i] = contour[i] - cv::Point(rt.x, rt.y);
	}

	cv::fillPoly(mask, tempContour, cv::Scalar::all(0xff));
	tempContour[0].clear();
	tempContour.clear();

	cv::Mat tempDiffImg(rt.height, rt.width, CV_16S);
	diffImg(rt).download(tempDiffImg);
	cv::Scalar avg, standar;
	cv::meanStdDev(tempDiffImg, avg, standar, mask);

// 	if (avg.val[0]<0)
// 	{
// 		avg.val
// 	}

	DefectData defect;
	
	if (getHandle()->IsFromLeftToRight()==true)
	{
		defect.fPy_x = PixelToPyhsic_1D(rt.x, getHandle()->PhysicResolution_x(), getHandle()->DownSampleFator());//起始位置
	}
	else
	{
		defect.fPy_x = PixelToPyhsic_1D(getHandle()->ImageSizePre().width-rt.x, getHandle()->PhysicResolution_x(), getHandle()->DownSampleFator());//起始位置
	}
	defect.fPy_y = PixelToPyhsic_1D(rt.y, getHandle()->PhysicResolution_y(), getHandle()->DownSampleFator()) + getHandle()->GetPhysicLength();
	defect.fPy_width = PixelToPyhsic_1D(rt.width, getHandle()->PhysicResolution_x(), getHandle()->DownSampleFator());//尺寸信息
	defect.fPy_height = PixelToPyhsic_1D(rt.height, getHandle()->PhysicResolution_y(), getHandle()->DownSampleFator());
	defect.fPyArea = PixelToPyhsic_2D(dArea, getHandle()->PhysicResolution_x(), getHandle()->PhysicResolution_y(), getHandle()->DownSampleFator()); //物理面积
	defect.iBlobSize = PixToPix_PreToSrc_2D(rt.width*rt.height, getHandle()->DownSampleFator());//像素面积
// 	defect.imgRect.x = PixToPix_PreToSrc_1D(rt.x, getHandle()->DownSampleFator());
// 	defect.imgRect.y = PixToPix_PreToSrc_1D(rt.y, getHandle()->DownSampleFator());
// 	defect.imgRect.width = PixToPix_PreToSrc_1D(rt.width, getHandle()->DownSampleFator());
// 	defect.imgRect.height = PixToPix_PreToSrc_1D(rt.height, getHandle()->DownSampleFator());
	defect.defectRect = rt;
	//
	/*if (fabs(avg.val[0]) < 3 && (defect.fPy_width < 0.2 || defect.fPy_height < 0.2))
	{
		return true;
	}*/
	//
	defect.defectName = std::string("未分类");//缺陷类别
	defect.iMeanDiff = (int)(avg.val[0]);//缺陷残差信息


	if (getHandle()->GetStripNum() != 0)
	{
		if (defect.fPy_x < getHandle()->GetStripOffset())
		{
			return true;
		}
		int k = 0;
		float fOffset = getHandle()->GetStripOffset();
		for (; k < getHandle()->GetStripNum(); k++)
		{
			if (defect.fPy_x < fOffset + getHandle()->GetStripLength(k))
			{
				defect.iStripIndex = k + 1;
				defect.fStripPyOffset_x = defect.fPy_x - fOffset;
				break;
			}
			fOffset += getHandle()->GetStripLength(k);
		}
		if (k == getHandle()->GetStripNum())
		{
			return true;
		}
	}
	else
	{
		defect.iStripIndex = 1;
		defect.fStripPyOffset_x = 0;
	}

	int iMaxLength = std::max(defect.defectRect.width, defect.defectRect.height);
	int iLen = 0;
	if (_MIN_SHOW_LEN_ - iMaxLength >= _DEFECT_BODER_SIZE_)
	{
		iLen = _MIN_SHOW_LEN_;
	}
	else
	{
		iLen = iMaxLength + _DEFECT_BODER_SIZE_;
	}
	int iCenterX = defect.defectRect.x + (defect.defectRect.width >> 1);
	int iCenterY = defect.defectRect.y + (defect.defectRect.height >> 1);
	int iImgRow = diffImg.rows;
	int iImgCol = diffImg.cols;
	defect.imgRect.x = std::max(iCenterX - (iLen >> 1), 0);
	defect.imgRect.y = std::max(iCenterY - (iLen >> 1), 0);
	defect.imgRect.width = std::min(defect.imgRect.x + iLen - 1, iImgCol - 1);
	defect.imgRect.height = std::min(defect.imgRect.y + iLen - 1, iImgRow - 1);
	defect.imgRect.width = defect.imgRect.width - defect.imgRect.x + 1;
	defect.imgRect.height = defect.imgRect.height - defect.imgRect.y + 1;

	defect.fPy_imgWidth = float(defect.imgRect.width) / getHandle()->PhysicResolution_x();
	defect.fPy_imgHeight = float(defect.imgRect.height) / getHandle()->PhysicResolution_y();

	vecDefectRect.push_back(defect);
	return true;
}

bool CBlobAnalysis::Geo_BlobToDefectThread(cv::Mat* diffImg, std::vector<cv::Point>* contour, std::vector<DefectData>* vecDefectRect, std::mutex* mtx)
{
	if (contour->size() == 0 || diffImg->size()!=getHandle()->ImageSizePre())
	{
		return false;
	}

	cv::Rect rt = cv::boundingRect(*contour);
	double dArea = fabs(cv::contourArea(*contour));

	cv::Mat mask = cv::Mat::zeros(rt.height, rt.width, CV_8U);
	std::vector<std::vector<cv::Point>> tempContour(1);
	tempContour[0].resize(contour->size());
	for (int i = 0; i < contour->size(); i++)
	{
		tempContour[0][i] = (*contour)[i] - cv::Point(rt.x, rt.y);
	}

	cv::fillPoly(mask, tempContour, cv::Scalar::all(0xff));
	tempContour[0].clear();
	tempContour.clear();

	cv::Scalar avg, standar;
	cv::meanStdDev((*diffImg)(rt), avg, standar, mask);

	DefectData defect;
	defect.fPy_x = PixelToPyhsic_1D(rt.x, getHandle()->PhysicResolution_x(), getHandle()->DownSampleFator());//起始位置
	defect.fPy_y = PixelToPyhsic_1D(rt.y, getHandle()->PhysicResolution_y(), getHandle()->DownSampleFator()) + getHandle()->GetPhysicLength();
	defect.fPy_width = PixelToPyhsic_1D(rt.width, getHandle()->PhysicResolution_x(), getHandle()->DownSampleFator());//尺寸信息
	defect.fPy_height = PixelToPyhsic_1D(rt.height, getHandle()->PhysicResolution_y(), getHandle()->DownSampleFator());
	defect.fPyArea = PixelToPyhsic_2D(dArea, getHandle()->PhysicResolution_x(), getHandle()->PhysicResolution_y(), getHandle()->DownSampleFator()); //物理面积
	defect.iBlobSize = PixToPix_PreToSrc_2D(rt.width*rt.height, getHandle()->DownSampleFator());//像素面积
	defect.defectRect.x = PixToPix_PreToSrc_1D(rt.x, getHandle()->DownSampleFator());
	defect.defectRect.y = PixToPix_PreToSrc_1D(rt.y, getHandle()->DownSampleFator());
	defect.defectRect.width = PixToPix_PreToSrc_1D(rt.width, getHandle()->DownSampleFator());
	defect.defectRect.height = PixToPix_PreToSrc_1D(rt.height, getHandle()->DownSampleFator());
	defect.defectName = std::string("未分类");//缺陷类别
	defect.iMeanDiff = (int)(avg.val[0]);//缺陷残差信息

	if (getHandle()->GetStripNum() != 0)
	{
		if (defect.fPy_x < getHandle()->GetStripOffset())
		{
			return true;
		}
		int k = 0;
		float fOffset = getHandle()->GetStripOffset();
		for (; k < getHandle()->GetStripNum(); k++)
		{
			if (defect.fPy_x < fOffset + getHandle()->GetStripLength(k))
			{
				defect.iStripIndex = k + 1;
				defect.fStripPyOffset_x = defect.fPy_x - fOffset;
				break;
			}
			fOffset += getHandle()->GetStripLength(k);
		}
		if (k == getHandle()->GetStripNum())
		{
			return true;
		}
	}
	else
	{
		defect.iStripIndex = 1;
		defect.fStripPyOffset_x = 0;
	}

	{
		std::unique_lock<std::mutex> lck(*mtx);
		vecDefectRect->push_back(defect);
	}

	return true;
}


bool CBlobAnalysis::BlobAnalysis(cv::Mat& frdMask, cv::Mat& diffMat, std::vector<DefectData>& vecDefectInfo, double* dTime)
{
	*dTime = cvGetTickCount();
	bool hr = BlobAnalysis(frdMask, m_vecvecBlob);
	if (hr == false)
	{
		return false;
	}
	vecDefectInfo.clear();

	m_vecReturn.clear();
	std::mutex mtx;
	for (int i = 0; i < m_vecvecBlob.size(); i++)
	{
		m_vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CBlobAnalysis::Geo_BlobToDefectThread, this, &diffMat, &m_vecvecBlob[i], &vecDefectInfo, &mtx)));
	}
	for (int i = 0; i < m_vecReturn.size(); i++)
	{
		bool hr = m_vecReturn[i].get();
		if (hr == false)
		{
			//抛出异常....
			printf("Get defect infor occurred some unhappy! Info: blob = %d\n", i);
			return false;
		}
	}
	m_vecReturn.clear();


	*dTime = (cv::getTickCount() - (*dTime)) / (1000 * cvGetTickFrequency());
	return true;
}

bool CBlobAnalysis::IsRange(cv::Point& pt_s, cv::Point& pt_e, cv::Point& pt)
{
	cv::Point diff_s = pt - pt_s;
	cv::Point diff_e = pt - pt_e;
	cv::Point diff_se = pt_s - pt_e;
	float dist_s = sqrt(float(diff_s.x*diff_s.x + diff_s.y*diff_s.y));
	float dist_e = sqrt(float(diff_e.x*diff_e.x + diff_e.y*diff_e.y));
	float dist = sqrt(float(diff_se.x*diff_se.x + diff_se.y*diff_se.y));
	if (fabs(dist_e + dist_s - dist) < 3.0f)
	{
		return true;
	}
	return false;
}

float CBlobAnalysis::PointToLineSegment(cv::Point& pt_s, cv::Point& pt_e, cv::Point& pt)
{
	cv::Point minPt;
	if (pt_s.x == pt_e.x)//竖直线x=c
	{
		minPt.x = pt_s.x;
		minPt.y = pt.y;
	}
	if (pt_s.y == pt_e.y)//水平线y=c
	{
		minPt.x = pt.x;
		minPt.y = pt_s.y;
	}
	if (IsRange(pt_s, pt_e, minPt))
	{
		cv::Point diff = minPt - pt;
		float fDist = sqrt(float(diff.x*diff.x + diff.y*diff.y));
		return fDist;
	}
	else
	{
		cv::Point diff_s = pt - pt_s;
		cv::Point diff_e = pt - pt_e;
		float dist_s = sqrt(float(diff_s.x*diff_s.x + diff_s.y*diff_s.y));
		float dist_e = sqrt(float(diff_e.x*diff_e.x + diff_e.y*diff_e.y));
		return std::min(dist_s, dist_e);
	}
	return FLT_MAX;
}
float CBlobAnalysis::GetDistRect(cv::Rect& rt1, cv::Rect& rt2)
{
	float fMinDist = FLT_MAX;
	cv::Point pt1[4];
	cv::Point pt2[4];
	pt1[0] = rt1.tl();
	pt1[2] = rt1.br();
	pt1[1].x = pt1[2].x;
	pt1[1].y = pt1[0].y;
	pt1[3].x = pt1[0].x;
	pt1[3].y = pt1[2].y;

	pt2[0] = rt2.tl();
	pt2[2] = rt2.br();
	pt2[1].x = pt2[2].x;
	pt2[1].y = pt2[0].y;
	pt2[3].x = pt2[0].x;
	pt2[3].y = pt2[2].y;

	for (int k = 0; k < 4; k++)
	{
		for (int i = 0; i < 4; i++)
		{
			fMinDist = std::min(fMinDist, PointToLineSegment(pt1[i], pt1[(i + 1)==4 ? 0 : (i + 1)], pt2[k]));
		}
		for (int i = 0; i < 4; i++)
		{
			fMinDist = std::min(fMinDist, PointToLineSegment(pt2[i], pt2[(i + 1) == 4 ? 0 : (i + 1)], pt1[k]));
		}
	}
	
	return fMinDist;
}
void CBlobAnalysis::MergeDefectData(DefectData& SrcDst, DefectData& info)
{
	int iMinX = std::min(SrcDst.defectRect.x, info.defectRect.x);
	int iMinY = std::min(SrcDst.defectRect.y, info.defectRect.y);
	int iMaxX = std::max(SrcDst.defectRect.x + SrcDst.defectRect.width - 1, info.defectRect.x + info.defectRect.width - 1);
	int iMaxY = std::max(SrcDst.defectRect.y + SrcDst.defectRect.height - 1, info.defectRect.y + info.defectRect.height - 1);


	DefectData defect;
	defect.defectRect.x = iMinX;
	defect.defectRect.y = iMinY;
	defect.defectRect.width = iMaxX - iMinX + 1;
	defect.defectRect.height = iMaxY - iMinY + 1;

	if (getHandle()->IsFromLeftToRight() == true)
	{
		defect.fPy_x = PixelToPyhsic_1D(defect.defectRect.x, getHandle()->PhysicResolution_x(), getHandle()->DownSampleFator());//起始位置
	}
	else
	{
		defect.fPy_x = PixelToPyhsic_1D(getHandle()->ImageSizePre().width - defect.defectRect.x, getHandle()->PhysicResolution_x(), getHandle()->DownSampleFator());//起始位置
	}
	defect.fPy_y = PixelToPyhsic_1D(defect.defectRect.y, getHandle()->PhysicResolution_y(), getHandle()->DownSampleFator()) + getHandle()->GetPhysicLength();
	defect.fPy_width = PixelToPyhsic_1D(defect.defectRect.width, getHandle()->PhysicResolution_x(), getHandle()->DownSampleFator());//尺寸信息
	defect.fPy_height = PixelToPyhsic_1D(defect.defectRect.height, getHandle()->PhysicResolution_y(), getHandle()->DownSampleFator());
	defect.fPyArea = SrcDst.fPyArea + info.fPyArea;
	defect.iBlobSize = PixToPix_PreToSrc_2D(defect.defectRect.width*defect.defectRect.height, getHandle()->DownSampleFator());//像素面积

	defect.defectName = std::string("未分类");//缺陷类别
	//defect.iMeanDiff = (int)(avg.val[0]);//缺陷残差信息
	if (SrcDst.fPyArea < info.fPyArea)
	{
		defect.iMeanDiff = info.iMeanDiff;
	}
	else
	{
		defect.iMeanDiff = SrcDst.iMeanDiff;
	}


	if (getHandle()->GetStripNum() != 0)
	{
		int k = 0;
		float fOffset = getHandle()->GetStripOffset();
		for (; k < getHandle()->GetStripNum(); k++)
		{
			if (defect.fPy_x < fOffset + getHandle()->GetStripLength(k))
			{
				defect.iStripIndex = k + 1;
				defect.fStripPyOffset_x = defect.fPy_x - fOffset;
				break;
			}
			fOffset += getHandle()->GetStripLength(k);
		}
	}
	else
	{
		defect.iStripIndex = 1;
		defect.fStripPyOffset_x = 0;
	}

	int iMaxLength = std::max(defect.defectRect.width, defect.defectRect.height);
	int iLen = 0;
	if (_MIN_SHOW_LEN_ - iMaxLength >= _DEFECT_BODER_SIZE_)
	{
		iLen = _MIN_SHOW_LEN_;
	}
	else
	{
		iLen = iMaxLength + _DEFECT_BODER_SIZE_;
	}
	int iCenterX = defect.defectRect.x + (defect.defectRect.width >> 1);
	int iCenterY = defect.defectRect.y + (defect.defectRect.height >> 1);
	int iImgRow = getHandle()->ImageSizePre().height;
	int iImgCol = getHandle()->ImageSizePre().width;
	defect.imgRect.x = std::max(iCenterX - (iLen >> 1), 0);
	defect.imgRect.y = std::max(iCenterY - (iLen >> 1), 0);
	defect.imgRect.width = std::min(defect.imgRect.x + iLen - 1, iImgCol - 1);
	defect.imgRect.height = std::min(defect.imgRect.y + iLen - 1, iImgRow - 1);
	defect.imgRect.width = defect.imgRect.width - defect.imgRect.x + 1;
	defect.imgRect.height = defect.imgRect.height - defect.imgRect.y + 1;

	defect.fPy_imgWidth = float(defect.imgRect.width) / getHandle()->PhysicResolution_x();
	defect.fPy_imgHeight = float(defect.imgRect.height) / getHandle()->PhysicResolution_y();

	SrcDst = defect;
}
void CBlobAnalysis::BlobCluster(std::vector<DefectData>& vecDefectInfo, float fDistThr)
{
	int iNum = (int)vecDefectInfo.size();
	if (iNum==0)
	{
		return;
	}
	std::vector<DefectData> vecTempInfo;
	std::vector<bool> vecMark(iNum, false);
	for (int i = 0; i < iNum; i++)
	{
		if (vecMark[i])
		{
			continue;
		}
		vecMark[i] = true;
		cv::Rect rt1 = vecDefectInfo[i].defectRect;
		for (int j = i+1; j < iNum; j++)
		{
			if (vecMark[j])
			{
				continue;
			}
			cv::Rect rt2 = vecDefectInfo[j].defectRect;
			float fDist = GetDistRect(rt1, rt2);
			if (fDist < fDistThr)
			{
				MergeDefectData(vecDefectInfo[i], vecDefectInfo[j]);
				rt2 = vecDefectInfo[i].defectRect;
				vecMark[j] = true;
			}
		}
		vecTempInfo.push_back(vecDefectInfo[i]);
	}

	vecTempInfo.swap(vecDefectInfo);
	vecTempInfo.clear();
}
bool CBlobAnalysis::BlobAnalysis(cv::Mat& frdMask, cv::cuda::GpuMat& diffMat, std::vector<DefectData>& vecDefectInfo, double* dTime)
{
	*dTime = cvGetTickCount();
	bool hr = BlobAnalysis(frdMask, m_vecvecBlob);
	if (hr == false)
	{
		return false;
	}
	//printf("blob num is: %d\n", m_vecvecBlob.size());
	vecDefectInfo.clear();

	for (int i = 0; i < m_vecvecBlob.size(); i++)
	{
		hr = Geo_BlobToDefectThread_gpu(diffMat, m_vecvecBlob[i], vecDefectInfo);
		if (hr == false)
		{
			//抛出异常....
			printf("Get defect infor occurred some unhappy! Info: blob = %d\n", i);
			return false;
		}
	}

	BlobCluster(vecDefectInfo, 200);

	*dTime = (cv::getTickCount() - (*dTime)) / (1000 * cvGetTickFrequency());
	return true;
}

// bool CBlobAnalysis::BlobAnalysis(cv::cuda::GpuMat& diffImg_gpu, cv::cuda::GpuMat& frdMask_gpu, cv::Mat& frdMask_cpu, std::vector<std::vector<cv::Point>>& vecvecBlob, double* dTime)
// {
// 	*dTime = cvGetTickCount();
// 	if (diffImg_gpu.size() != frdMask_gpu.size() || diffImg_gpu.size() != frdMask_cpu.size())
// 	{
// 		return false;
// 	}
// 	//Blob.....
// 	/*cv::Mat K = cv::getStructuringElement(cv::MORPH_RECT, cv::Size((m_iBoundaryOffset << 1) + 1, 1));
// 	cv::Mat mask;
// 	cv::erode(frdMask, mask, K);
// 	diffImg.setTo(0x00, ~mask);
// 	mask = (diffImg != 0);*/
// 	m_f_erode->apply(frdMask_gpu, frdMask_gpu);
// 	cv::cuda::bitwise_not(frdMask_gpu, frdMask_gpu);
// 	diffImg_gpu.setTo(0x00, frdMask_gpu);
// 	/*cv::cuda::compare(diffImg_gpu, 0, frdMask_gpu, cv::CMP_NE);
// 	frdMask_gpu.download(frdMask_cpu);
// 
// 	for (auto&& pt : vecvecBlob)
// 	{
// 		pt.clear();
// 	}
// 	vecvecBlob.clear();
// 	std::vector<std::vector<cv::Point>> tempContours;
// 	tempContours.clear();
// 	m_vecReturn.clear();
// 	std::mutex mtx1, mtx2;
// 	for (int i = 0; i < getHandle()->DataPatchNum(); i++)
// 	{
// 		m_vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CBlobAnalysis::GetBlobThread, this, &frdMask_cpu, getHandle()->TruthRect(i), &vecvecBlob, &tempContours, m_iBlobThrsh, &mtx1, &mtx2)));
// 	}
// 	for (int i = 0; i < m_vecReturn.size(); i++)
// 	{
// 		bool hr = m_vecReturn[i].get();
// 		if (hr == false)
// 		{
// 			//抛出异常....
// 			printf("Blob analysis occurred some unhappy! Info: Rect: x = %d, y= %d, width = %d, height=%d\n", getHandle()->TruthRect(i).x, getHandle()->TruthRect(i).y, getHandle()->TruthRect(i).width, getHandle()->TruthRect(i).height);
// 			return false;
// 		}
// 	}
// 	m_vecReturn.clear();
// 	MergeContour(tempContours, vecvecBlob, m_iBlobThrsh);
// 	for (int i = 0; i < tempContours.size(); i++)
// 	{
// 		tempContours[i].clear();
// 	}
// 	tempContours.clear();*/
// 	*dTime = (cv::getTickCount() - (*dTime)) / (1000 * cvGetTickFrequency());
// 	return true;
// }

/*bool CBlobAnalysis::BlobAnalysis(cv::Mat& maskImg, std::vector<DefectData>& vecDefect)
{
	double t1 = cvGetTickCount();
	if (maskImg.size()!=getHandle()->ImageSizePre() || maskImg.type()!=CV_8U || maskImg.size()!=m_tempMark.size())
	{
		return false;
	}
	vecDefect.clear();

	m_tempMark.setTo(0x00);
	int* pMark = (int*)m_tempMark.data; 
	int iStep = maskImg.cols;

	int iMark = 1;
	int iMarkTop, iMarkLeft;
	std::vector<int> vecMark;
	vecMark.push_back(0);

	int iTempNum = 0;
	bool bTop, bLeft;
	//for (int i = 1; i < maskImg.rows-1; i++)
	for (int i = 0, idx=0; i < maskImg.rows ; i++)
	{
		//for (int j = 1; j < maskImg.cols-1; j++)
		for (int j = 0; j < maskImg.cols ; j++, idx++)
		{
			if (maskImg.data[idx] == 0x00)
			{
				continue;
			}
			bTop = bLeft = false;
			//iMarkTop = iMarkLeft = iMark;
			if (i!=0)
			{
				if (maskImg.data[idx - maskImg.cols] == 0xff)
				{
					bTop = true;
					iMarkTop = pMark[idx - maskImg.cols];
				}
			}
			
			if (j != 0)
			{
				if (maskImg.data[idx - 1] == 0xff)
				{
					bLeft = true;
					iMarkLeft = pMark[idx - 1];
				}
			}
			
			if ((!bTop) & (!bLeft))
			{
				vecMark.push_back(iMark);
				pMark[idx] = iMark++;
				continue;
			}

			if (bTop & bLeft)
			{
				if (iMarkTop == iMarkLeft)
				{
					pMark[idx] = iMarkTop;
					continue;
				}
				if (iMarkTop < iMarkLeft)
				{
					pMark[idx] = iMarkTop;
					vecMark[iMarkLeft] = iMarkTop;
					continue;
				}
				else
				{
					pMark[idx] = iMarkLeft;
					vecMark[iMarkTop] = iMarkLeft;
					continue;
				}
			}
			else if (bLeft)
			{
				pMark[idx] = iMarkLeft;
				continue;
			}
			else
			{
				pMark[idx] = iMarkTop;
				continue;
			}
		}
		//iStep += maskImg.cols;
	}
	if (vecMark.size()==1)
	{
		return true;
	}
	
	for (int i = vecMark.size() - 1; i > 1; i--)
	{
		int iTempMark = vecMark[i];
		while (iTempMark!=vecMark[iTempMark])
		{
			iTempMark = vecMark[iTempMark];
		}
		vecMark[i] = iTempMark;
	}
	std::vector<int> vecTempIndex,vecTempMark;
	for (int i = 0; i < vecMark.size(); i++)
	{
		vecTempIndex.push_back(i);
		vecTempMark.push_back(vecMark[i]);
	}
	QuickSort_DEC(&vecTempMark[0], 0, vecTempMark.size() - 1, &vecTempIndex[0]);

	int iBlobNum = 0;
	iMark = 0;
	for (int i = 0; i < vecTempMark.size()-1; i++)
	{
		if (vecTempMark[i] != iMark)
		{
			iBlobNum++;
			iMark = vecTempMark[i];
		}
		vecMark[vecTempIndex[i]] = iBlobNum;
	}
	vecTempMark.clear();
	vecTempIndex.clear();
	std::vector<std::vector<cv::Point>> vecvecBlob;
	vecvecBlob.resize(iBlobNum);
	
	iTempNum = 0;
	for (int i = 0, idx=0; i < maskImg.rows; i++)
	{
		for (int j = 0; j < maskImg.cols; j++,idx++)
		{
			if (pMark[idx] == 0)
			{
				continue;
			}
			iTempNum++;
			iMark = pMark[idx];
			iMark = vecMark[pMark[idx]]; 
			//iMark = vecMark[iMark];
			vecvecBlob[iMark - 1].push_back(cv::Point(j, i));
		}
	}

	printf("blob output: %f\n",(cvGetTickCount() - t1)/(1000*cvGetTickFrequency()));

	vecMark.clear();

	return true;
}*/

bool CBlobAnalysis::GetBlobThread1(cv::Mat* binaryImg, cv::Rect& RoiRt, std::vector<std::vector<cv::Point>>* vecvecContour1, std::vector<std::vector<cv::Point>>* vecvecContour2, int iBlobThresh, std::mutex* mtx1, std::mutex* mtx2)
{
	if (RoiRt.x<0 || RoiRt.x + RoiRt.width>binaryImg->cols || RoiRt.y<0 || RoiRt.y + RoiRt.height>binaryImg->rows)
	{
		return false;
	}
	cv::Mat maskImg = (*binaryImg)(RoiRt).clone();

	std::vector<std::vector<cv::Point>> vecvecContours;
	cv::findContours(maskImg, vecvecContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	if (vecvecContours.size() == 0)
	{
		return true;
	}

	for (auto&& c : vecvecContours)
	{
		cv::Rect boundRect = cv::boundingRect(c);
		for (int k = 0; k < c.size(); k++)
		{
			c[k].x += RoiRt.x;
			c[k].y += RoiRt.y;
		}
		if (boundRect.x > 0 && boundRect.y>0 && boundRect.x + boundRect.width < maskImg.cols && boundRect.y + boundRect.height < maskImg.rows)
		{
			double dArea = fabs(cv::contourArea(c));
			if (dArea < iBlobThresh)
			{
				continue;
			}
			std::unique_lock<std::mutex> lck(*mtx1);
			vecvecContour1->push_back(c);
		}
		else
		{
			std::unique_lock<std::mutex> lck(*mtx2);
			vecvecContour2->push_back(c);
		}
	}

	for (int i = 0; i < vecvecContours.size(); i++)
	{
		vecvecContours[i].clear();
	}
	vecvecContours.clear();

	return true;
}


bool CBlobAnalysis::GetBlobThread(cv::Mat* binaryImg, cv::Rect& RoiRt, std::vector<std::vector<cv::Point>>* vecvecContour1, std::vector<std::vector<cv::Point>>* vecvecContour2, int iBlobThresh, std::mutex* mtx1, std::mutex* mtx2)
{
	if (RoiRt.x<0 || RoiRt.x + RoiRt.width>binaryImg->cols || RoiRt.y<0 || RoiRt.y + RoiRt.height>binaryImg->rows)
	{
		return false;
	}
	cv::Mat maskImg = (*binaryImg)(RoiRt)/*.clone()*/;

	/*for (int i = 0; i < vecvecContour1.size(); i++)
	{
	vecvecContour1[i].clear();
	}
	vecvecContour1.clear();

	for (int i = 0; i < vecvecContour2.size(); i++)
	{
	vecvecContour2[i].clear();
	}
	vecvecContour2.clear();*/
	std::vector<std::vector<cv::Point>> vecvecContours;
	cv::findContours(maskImg, vecvecContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	if (vecvecContours.size() == 0)
	{
		return true;
	}
	std::vector<std::vector<cv::Point>> vecTempContour1, vecTempContour2;
	vecTempContour1.clear();
	vecTempContour2.clear();
	for (auto&& c : vecvecContours)
	{
		cv::Rect boundRect = cv::boundingRect(c);
		for (int k = 0; k < c.size(); k++)
		{
			c[k].x += RoiRt.x;
			c[k].y += RoiRt.y;
		}
		if (boundRect.x > 0 && boundRect.y>0 && boundRect.x + boundRect.width < maskImg.cols && boundRect.y + boundRect.height < maskImg.rows)
		{
			/*double dArea = fabs(cv::contourArea(c));
			if (dArea < iBlobThresh)
			{
				continue;
			}*/
			if (c.size() < iBlobThresh)
			{
				continue;
			}
			vecTempContour1.push_back(c);
// 			std::unique_lock<std::mutex> lck(*mtx1);
// 			vecvecContour1->push_back(c);
		}
		else
		{
			vecTempContour2.push_back(c);
// 			std::unique_lock<std::mutex> lck(*mtx2);
// 			vecvecContour2->push_back(c);
		}
	}

	for (int i = 0; i < vecvecContours.size(); i++)
	{
		vecvecContours[i].clear();
	}
	vecvecContours.clear();

	{
		std::unique_lock<std::mutex> lck(*mtx1);
		vecvecContour1->insert(vecvecContour1->end(), vecTempContour1.begin(), vecTempContour1.end());
		vecvecContour2->insert(vecvecContour2->end(), vecTempContour2.begin(), vecTempContour2.end());
	}
	for (int i = 0; i < vecTempContour1.size(); i++)
	{
		vecTempContour1[i].clear();
	}
	vecTempContour1.clear();
	for (int i = 0; i < vecTempContour2.size(); i++)
	{
		vecTempContour2[i].clear();
	}
	vecTempContour2.clear();

	return true;
}

float CBlobAnalysis::GetContourDist(std::vector<cv::Point>& c1, std::vector<cv::Point>& c2)
{
	float fMin = FLT_MAX;
	for (auto&& p1 : c1)
	{
		for (auto&& p2 : c2)
		{
			cv::Point p = p1 - p2;
			float fV = p.x*p.x + p.y*p.y;
			fMin = std::min(fMin, fV);
			if (fMin < 9.0f)
			{
				return sqrt(fMin);
			}
		}
	}
	return sqrt(fMin);
}

float CBlobAnalysis::GetContourDist(cv::Rect& rt1, cv::Rect& rt2)
{
	/*int x1_left = rt1.x;
	int x1_right = rt1.x + rt1.width - 1;
	int y1_top = rt1.y;
	int y1_bottom = rt1.y + rt1.height - 1;

	int x2_left = rt2.x;
	int x2_right = rt2.x + rt2.width - 1;
	int y2_top = rt2.y;
	int y2_bottom = rt2.y + rt2.height - 1;

	if (abs(x1_left - x2_right) < 2)
	{

	}*/

	m_vecPt1[0] = cv::Point(rt1.x, rt1.y);
	m_vecPt1[1] = cv::Point(rt1.x, rt1.y + rt1.height - 1);
	m_vecPt1[2] = cv::Point(rt1.x + rt1.width - 1, rt1.y + rt1.height - 1);
	m_vecPt1[3] = cv::Point(rt1.x + rt1.width - 1, rt1.y);

	m_vecPt2[0] = cv::Point(rt2.x, rt2.y);
	m_vecPt2[1] = cv::Point(rt2.x, rt2.y + rt2.height - 1);
	m_vecPt2[2] = cv::Point(rt2.x + rt2.width - 1, rt2.y + rt2.height - 1);
	m_vecPt2[3] = cv::Point(rt2.x + rt2.width - 1, rt2.y);

	return GetContourDist(m_vecPt2, m_vecPt1);
}

void CBlobAnalysis::MergeContour(cv::Mat& frd, std::vector<std::vector<cv::Point>>& vecTempContour, std::vector<std::vector<cv::Point>>& vecvecContour, int iBlobThresh)
{
	int iNum = int(vecTempContour.size());
	if (iNum == 0)
	{
		return;
	}

	std::vector<uchar> vecMark(iNum, 0);
	std::vector<cv::Rect> vecRect(iNum);
	for (int i = 0; i < iNum; i++)
	{
		vecRect[i] = cv::boundingRect(vecTempContour[i]);
	}
	for (int i = 0; i < iNum; i++)
	{
		if (vecMark[i] == 1)
		{
			continue;;
		}
		std::vector<cv::Point> vecTemp = vecTempContour[i];
		cv::Rect rtTemp = cv::boundingRect(vecTemp);
		vecMark[i] = 1;
		for (int j = i + 1; j < iNum; j++)
		{
			float fDist = GetContourDist(rtTemp, vecRect[j]);
			if (fDist < 3.0f)
			{
				vecTemp.insert(vecTemp.end(), vecTempContour[j].begin(), vecTempContour[j].end());
				vecMark[j] = 1;
				rtTemp = cv::boundingRect(vecTemp);
			}
		}
		double dArea = fabs(cv::contourArea(vecTemp));
		if (dArea < iBlobThresh)
		{
			continue;
		}
		vecvecContour.push_back(vecTemp);
		vecTemp.clear();
	}
	vecMark.clear();
	vecRect.clear();
}

void CBlobAnalysis::SetParam(void* param)
{

}

bool CBlobAnalysis::SetNoCheckArea(cv::cuda::GpuMat& DiffImg, cv::cuda::GpuMat& DiffMask, double* dTime)
{
	*dTime = cvGetTickCount();
	DiffImg(m_rtLeftSetNoCheck).setTo(0x00);
	DiffImg(m_rtRightSetNoCheck).setTo(0x00);
	//DiffImg.convertTo(DiffMask, CV_8U);
	//DiffMask(m_rtLeftSetNoCheck).setTo(0x00);
	//DiffMask(m_rtRightSetNoCheck).setTo(0x00);
	*dTime = (cvGetTickCount() - (*dTime)) / (1000 * cvGetTickFrequency());
	return true;
}

void CBlobAnalysis::GetBlobBoundaryOffset(int& iLeft, int& iRight)
{
	iLeft = m_iBoundaryOffsetLeft;
	iRight = m_iBoundaryOffsetRight;
}

void CBlobAnalysis::GetStripBoundary(int& iLeft, int& iRight)
{
	iLeft = std::max(0, m_rtLeftSetNoCheck.x + m_rtLeftSetNoCheck.width - 1);
	iRight = m_rtRightSetNoCheck.x;
}