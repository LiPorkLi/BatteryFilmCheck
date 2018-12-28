#pragma once

#include "../safe_queue.h"
#include "../GlobalData.h"
#include <thread>

class CGetGrayCurver
{
public:
	CGetGrayCurver();
	~CGetGrayCurver();

	void SetParam(safe_queue<GrabImgInfo>* queue_grab, safe_queue<cv::Mat>* queue_curver, safe_queue<std::string>* queue_log);
	void StartInspectThread();
	void StopInspectThread();
private:
	std::string m_strLog;
	std::stringstream m_ss;

	bool m_isProcess, *m_isStop;
	std::thread m_processThread;
	safe_queue<GrabImgInfo>* m_queue_grab;
	safe_queue<cv::Mat>* m_queue_curver;
	safe_queue<std::string>* m_queue_log;

	void Pipline();
	bool Process(cv::Mat& img, cv::Mat& curver);
};

