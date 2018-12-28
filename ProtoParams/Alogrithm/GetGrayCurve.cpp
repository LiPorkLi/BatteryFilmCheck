#include "stdafx.h"
#include "GetGrayCurve.h"
//#include <synchapi.h>
//#define _PRINTF_
CGetGrayCurver::CGetGrayCurver()
{
	m_isProcess = false;
	m_isStop = new bool;
	*m_isStop = true;
	m_queue_grab = NULL;
	m_queue_curver = NULL;
	m_queue_log = NULL;
}


CGetGrayCurver::~CGetGrayCurver()
{
	StopInspectThread();
	delete m_isStop;
}

void CGetGrayCurver::StopInspectThread()
{
	m_isProcess = false;
	while (*m_isStop == false)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(10)); //–›√ﬂ10∫¡√Î  
	}
}


void CGetGrayCurver::StartInspectThread()
{
	StopInspectThread();
	m_isProcess = true;
	m_processThread = std::thread(std::bind(&CGetGrayCurver::Pipline, this));
	if (m_processThread.joinable())
	{
		*m_isStop = false;
		m_processThread.detach();
	}
}

void CGetGrayCurver::Pipline()
{
	while (1)
	{
		if (m_isProcess==false)
		{
			break;
		}
		std::shared_ptr<GrabImgInfo> g = m_queue_grab->wait_and_pop();
		if (g!=nullptr)
		{
			//process
			if (g->srcimg.size().width == 0 || g->srcimg.size().height==0)
			{
				//“Ï≥£
				m_ss.clear(); m_ss.str("");
				m_ss << "Image " << g->idx << " is NULL!\n";
				m_strLog = m_ss.str();
				m_queue_log->push(std::move(m_strLog));
				//*m_isStop = true;
				continue;
			}

			cv::Mat curver;
			bool hr = Process(g->srcimg, curver);
			//
			if (hr == false)
			{
				//“Ï≥£
				m_ss.clear(); m_ss.str("");
				m_ss << "Get curver image " << g->idx << " is FILED, and thread is stoped!\n";
				m_strLog = m_ss.str();
				m_queue_log->push(std::move(m_strLog));
				break;
			}
			else
			{
				m_queue_curver->push(std::move(curver));
			}
		}
	}
	*m_isStop = true;
}


void CGetGrayCurver::SetParam(safe_queue<GrabImgInfo>* queue_grab, safe_queue<cv::Mat>* queue_curver, safe_queue<std::string>* queue_log)
{
	StopInspectThread();

	m_queue_grab = queue_grab;
	m_queue_log = queue_log;
	m_queue_curver = queue_curver;
}

bool CGetGrayCurver::Process(cv::Mat& img, cv::Mat& curver)
{
	img.convertTo(img, CV_32F);
	curver = cv::Mat::zeros(1, img.cols, CV_32F);
	cv::reduce(img, curver, 0, CV_REDUCE_AVG);
	return true;
}
