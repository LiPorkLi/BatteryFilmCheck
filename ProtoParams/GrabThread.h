#pragma once
#include "SapClassBasic.h"
#include "Grab.h"
#include "cv.hpp"
#include "highgui.hpp"
#include "GlobalData.h"
#include <thread>

#define _GRAB_EXTR_PATH_  ("D:\\PVInspectionProject\\ccf\\cam-extr.ccf")
#define _GRAB_INNER_PATH_ ("D:\\PVInspectionProject\\ccf\\cam-inner.ccf")
/*struct GrabImgInfo
{
	cv::Mat srcimg;//????
	int idx;

	enum ProcessType
	{
		_normal_,
		_train_,
		_flatfield_,
		_rectify_,
		_ignore_,
		_mark_zero_
	};
	ProcessType iMark;//train 1, inspect 0, flatfield 2, rectify 3, ignore 4
};*/
class CGrabThread: public CCardGrabHook
{
public:
	CGrabThread(); 
	~CGrabThread();

	virtual int _stdcall OnFrameGrabbed(int frameIndex, BYTE ** imageDataTab, float time) ;
	virtual void _stdcall OnGrabStop() ;
	virtual void _stdcall OnEndOfNLine() ;

	bool InitiaGrab(safe_queue<GrabImgInfo>* grabQueue, safe_queue<std::string>* logQueue, bool isExtr = true);
	void StartGrabThread(bool isStartAll = false);
	void StopGrabThread();
	void FreeGrab();
	void SetSavePath(bool isSave, char* savePath);
	int GetGrabCount();
	float GetFrameFreq();

	void SetGain(double fGain);
	double GetGain();

	void SetExposureTime(double fTime);
	double GetExposureTime();
	void SetGainByProdct(float fThickness);
private:
	void StartSaveThread();
	void StopSaveThread();
	void save();
	bool *m_isStop, m_isStartSave;
	std::thread m_saveThread;
	int m_iSavedNum;

	std::string m_strLog;
	std::stringstream m_ss;
	CCardGrab* m_grab;
	GRAB_FORMAT m_GrabFormat;
	GrabImgInfo m_imgInfo, m_imgSave;
	safe_queue<GrabImgInfo>* m_GrabQueue;
	safe_queue<std::string>* m_LogQueue;
	//cv::Mat m_img;

	std::shared_ptr<safe_queue<GrabImgInfo>> m_saveQueue;

	bool m_isSave;
	char m_strSavePath[256], m_strSaveName[256];

	long m_nFrameCount;
	float m_fLastFrameTime;
	float m_fFramFreq;

};

