#pragma once

#include "cv.h"
#include "highgui.h"
#include <vector>
#include "proto/param.pb.h"
#include "proto/checkparam.pb.h"
#include "safe_queue.h"
#include "ParamHelper.h"

//#include "Alogrithm/InspectProcedure.h"

struct DefectData
{
	cv::Rect defectRect, imgRect;
	float fPy_x, fPy_y;
	float fPy_width, fPy_height;
	int iStripIndex;
	float fStripPyOffset_x;
	int iBlobSize;
	float fPyArea;
	int iMeanDiff;
	//int iDefectType;
	float fPy_imgWidth, fPy_imgHeight;
	std::string defectName;
};

struct GeoClassifyModel
{
	float fMinWidth, fMaxWidth;
	float fMinHeight, fMaxHeight;
	int iMinDiff, iMaxDiff;
	float fMinArea, fMaxArea;
	int iDefectType;
};

struct ImageInspectResult
{
	cv::Mat srcImage, diffImage;//srcImage is CV_8U, diffImage is CV_8U
	std::vector<DefectData> m_vecDefectList;
	int idx;
	uchar isAlert;

	cv::Mat GetMask(int idx)
	{
		if (idx < 0 || idx > m_vecDefectList.size()-1)
		{
			return cv::Mat();
		}
		cv::Mat tempMat = cv::Mat::zeros(m_vecDefectList[idx].imgRect.height, m_vecDefectList[idx].imgRect.width, CV_8U);
		cv::Mat tempMat2 = diffImage(m_vecDefectList[idx].imgRect).clone();
		cv::Rect rt(m_vecDefectList[idx].defectRect.x - m_vecDefectList[idx].imgRect.x, m_vecDefectList[idx].defectRect.y - m_vecDefectList[idx].imgRect.y, 
			m_vecDefectList[idx].defectRect.width, m_vecDefectList[idx].defectRect.height);
		tempMat2(rt).copyTo(tempMat(rt));
		return tempMat;
	};
};

struct GrabImgInfo
{
	cv::Mat srcimg;//????
	int idx;

	enum ProcessType
	{
		_normal_,
		_train_,
		_flatfield_,
		_flatfield2_,
		_rectify_,
		_ignore_,
		_mark_zero_
	};
	ProcessType iMark;//train 1, inspect 0, flatfield 2, rectify 3, ignore 4
};
class Global
{
public: 
	Global();
	virtual ~Global();

public:
	static safe_queue<ImageInspectResult> g_InspectQueue;
	static MainParam::param g_param;
	static safe_queue<GrabImgInfo> g_ProcessQueue;
	static safe_queue<std::string> g_AlogrithmLog;
	static bool g_isStop;
//	static std::shared_ptr<CInspectProcedure> g_pInspect;
// 	static std::shared_ptr<safe_queue<ImageInspectResult>> g_InspectQueue;
// 	static std::shared_ptr<safe_queue<GrabImgInfo>> g_grabQueue;
// 	static std::shared_ptr< MainParam::param> g_param;
};
