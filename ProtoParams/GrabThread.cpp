#include "stdafx.h"
#include "GrabThread.h"


CGrabThread::CGrabThread()
{
	m_grab = NULL;
	m_isSave = false;
	m_isStop = new bool;
	*m_isStop = true;
	m_nFrameCount = 0;
}


CGrabThread::~CGrabThread()
{
	StopGrabThread();
	FreeGrab();
	delete m_isStop;
}

void CGrabThread::FreeGrab()
{
	if (m_grab)
	{
		m_grab->Release();
		m_grab = NULL;

		m_ss.clear();m_ss.str("");
		m_ss << "Grab is free!\n";
		m_strLog = m_ss.str();
		m_LogQueue->push(std::move(m_strLog));
	}
}

int _stdcall CGrabThread::OnFrameGrabbed(int frameIndex, BYTE ** imageDataTab, float time)
{
	float lTime = cvGetTickCount();
	float tstep = (lTime - m_fLastFrameTime) / (1000 * cvGetTickFrequency());
	m_fFramFreq = 1000.0 / tstep;
	m_fLastFrameTime = lTime;

	m_imgInfo.srcimg = cv::Mat(m_GrabFormat.height, m_GrabFormat.width, CV_8U);
 	memcpy(m_imgInfo.srcimg.data, imageDataTab[0], sizeof(uchar)*m_GrabFormat.height*m_GrabFormat.width);
	
	m_nFrameCount++;
	m_imgInfo.idx = m_nFrameCount;
	

	if (m_isSave && m_iSavedNum < 1000000)
	{
		m_imgSave.srcimg = cv::Mat(m_GrabFormat.height, m_GrabFormat.width, CV_8U);
		memcpy(m_imgSave.srcimg.data, imageDataTab[0], sizeof(uchar)*m_GrabFormat.height*m_GrabFormat.width);
		m_imgSave.idx = m_imgInfo.idx;
		m_saveQueue->push(std::move(m_imgSave));
		m_iSavedNum++;

		if (m_iSavedNum == 10000000)//保存的速度赶不上采集的速度，所以设置最多保存20张，不然主机会爆炸
		{
			*m_isStop = true;
		}
	}
	//printf("%d img Grab time: %f, img h:	%d, w:	%d\n",m_imgInfo.idx, tstep,  m_imgInfo.srcimg.rows, m_imgInfo.srcimg.cols);
	m_ss.clear();m_ss.str("");
	m_ss << m_imgInfo.idx << " img Grab time: " << tstep << std::endl;
	m_strLog = m_ss.str();
	m_LogQueue->push(std::move(m_strLog));
	
	m_GrabQueue->push(std::move(m_imgInfo));

	return 0;
}

void _stdcall CGrabThread::OnGrabStop()
{

}

void _stdcall CGrabThread::OnEndOfNLine()
{

}

void CGrabThread::StartGrabThread(bool isStartAll/* = false*/)
{
	//StopGrabThread();
	if (m_grab)
	{
		if (isStartAll==true)
		{
			m_imgInfo.idx = 0;
			m_nFrameCount = 0;
		}
	
		m_fFramFreq = 0;
		m_fLastFrameTime = 0;
		m_grab->Start(-1);

		m_ss.clear(); m_ss.str("");
		if (m_isSave==true)
		{
			//m_nFrameCount = 0;
			m_iSavedNum = 0;
			StartSaveThread();
			m_ss << "Image Save thread is start!	";
		}
		
		m_ss << "Grab is start!\n";
		m_strLog = m_ss.str();
		m_LogQueue->push(std::move(m_strLog));
	}
}

void CGrabThread::StopGrabThread()
{
	if (m_grab)
	{
		//m_nFrameCount = 0;
		m_fLastFrameTime = 0;
		m_grab->Stop();
		m_ss.clear();m_ss.str("");
		m_ss << "Grab is stopped!\n";
		m_strLog = m_ss.str();
		m_LogQueue->push(std::move(m_strLog));
		if (m_isSave==true)
		{
			StopSaveThread();
		}
	}
}

bool CGrabThread::InitiaGrab(safe_queue<GrabImgInfo>* grabQueue, safe_queue<std::string>* logQueue, bool isExtr/* = true*/)
{
	m_GrabQueue = grabQueue;
	m_LogQueue = logQueue;
	if (m_grab)
	{
		m_grab->Release();
		m_grab = NULL;
	}
	int iCardNum = SapManager::GetServerCount();

	m_ss.clear(); m_ss.str("");
	m_ss << m_imgInfo.idx << " Server Count: " << iCardNum << std::endl;
	m_strLog = m_ss.str();
	m_LogQueue->push(std::move(m_strLog));

	if (1)
	{
		char serverName[CORSERVER_MAX_STRLEN];
		int serverIndex = 0;
		for (; serverIndex < iCardNum; serverIndex++)
		{
			SapManager::GetServerName(serverIndex, serverName, sizeof(serverName));

			m_ss.clear(); m_ss.str("");
			m_ss << m_imgInfo.idx << " serverName: " << serverName << std::endl;
			m_strLog = m_ss.str();
			m_LogQueue->push(std::move(m_strLog));

			if (strstr(serverName, "MX") != NULL)
			{
				break;
			}
		}
		if (serverIndex == iCardNum)
		{
			m_ss.clear();m_ss.str("");
			m_ss << "card initialize failed\n";
			m_strLog = m_ss.str();
			m_LogQueue->push(std::move(m_strLog));

			return false;
		}

// 		FILE* fr = fopen("d:\\log-grab.txt", "a+");
// 		fprintf(fr, "Start!\n");
// 		fclose(fr);

		m_grab = CreateCCardGrabInstance();

// 		fr = fopen("d:\\log-grab.txt", "a+");
// 		fprintf(fr, "Start end! %x\n", m_grab);
// 		fclose(fr);

		//printf("card name: %s\n", serverName);
		m_ss.clear();m_ss.str("");
		m_ss << "card name: " << "Xtium-CL_MX4_1" << std::endl;
		m_strLog = m_ss.str();
		m_LogQueue->push(std::move(m_strLog));

		int iReturn = -1;
		if (isExtr==true)
		{
			iReturn = m_grab->Init(&m_GrabFormat, serverName, 0, _GRAB_EXTR_PATH_, this);
		}
		else
		{
// 			fr = fopen("d:\\log-grab.txt", "a+");
// 			fprintf(fr, "Initia  inner!\n");
// 			fclose(fr);

			iReturn = m_grab->Init(&m_GrabFormat, "Xtium-CL_MX4_1", 0, _GRAB_INNER_PATH_, this);
		}

// 		fr = fopen("d:\\log-grab.txt", "a+");
// 		fprintf(fr, "Initia  is OK!\n");
// 		fclose(fr);

		if (iReturn >= 0)
		{
			m_grab->ConnectToDeviceByAcq("Xtium-CL_MX4_1_Serial_0");
			if (m_GrabFormat.bit != 1)
			{
				m_grab->Release();
				m_grab = NULL;

				m_ss.clear();m_ss.str("");
				m_ss << "camera image is not Gray!\n";
				m_strLog = m_ss.str();
				m_LogQueue->push(std::move(m_strLog));

				return false;
			}
			//m_imgInfo.srcimg = cv::Mat(m_GrabFormat.height, m_GrabFormat.width, CV_8U);
			//m_img = m_imgInfo.srcimg.clone();

			//printf("camera initialize succeed: img w*h:	%d * %d, channel=%d\n", m_GrabFormat.width, m_GrabFormat.height, m_GrabFormat.bit);
			m_ss.clear();m_ss.str("");
			m_ss << "camera initialize succeed: img w*h:	" << m_GrabFormat.width << "*" << m_GrabFormat.height << ", channel=" << m_GrabFormat.bit<<std::endl;
			m_strLog = m_ss.str();
			m_LogQueue->push(std::move(m_strLog));
			return true;
		}
		else
		{
			m_ss.clear();m_ss.str("");
			m_ss << "camera initialize failed\n";
			m_strLog = m_ss.str();
			m_LogQueue->push(std::move(m_strLog));
		}
	}
	else
	{
		m_grab = NULL;

		m_ss.clear();m_ss.str("");
		m_ss << "Don't find card!\n";
		m_strLog = m_ss.str();
		m_LogQueue->push(std::move(m_strLog));

		return false;
	}
	return false;
}

void CGrabThread::SetSavePath(bool isSave, char* savePath)
{
	m_isSave = isSave;
	if (m_isSave && m_grab)
	{
		sprintf_s(m_strSavePath, "%s", savePath);
		m_saveQueue = std::make_shared<safe_queue<GrabImgInfo>>();

		//m_imgSave.srcimg = cv::Mat(m_GrabFormat.height, m_GrabFormat.width, CV_8U);
	}
}

void CGrabThread::StartSaveThread()
{
	StopSaveThread();
	m_isStartSave = true;
	m_saveThread = std::thread(std::bind(&CGrabThread::save, this));
	if (m_saveThread.joinable())
	{
		m_saveThread.detach();
		*m_isStop = false;
	}
}

void CGrabThread::save()
{
	while (1)
	{
		std::shared_ptr<GrabImgInfo> g = m_saveQueue->wait_and_pop();
		if (g != nullptr)
		{
			sprintf_s(m_strSaveName, "%s\\%d.png", m_strSavePath, g->idx);
			cv::imwrite(m_strSaveName, g->srcimg);

			m_ss.clear(); m_ss.str("");
			m_ss << g->idx<<" image is saved!\n";
			m_strLog = m_ss.str();
			m_LogQueue->push(std::move(m_strLog));
		}
		else
		{
			std::this_thread::sleep_for(std::chrono::microseconds(100));
		}

		if (m_isStartSave==false && (m_saveQueue->getPushCount()==m_saveQueue->getPopCount()))
		{
			break;
		}
	}
	*m_isStop = true;
}

void CGrabThread::StopSaveThread()
{
	m_isStartSave = false;
	//int i = 0;
	while (*m_isStop == false)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(10)); //休眠10毫秒  
	}
	m_ss.clear(); m_ss.str("");
	m_ss << "Save thread is stopped!\n";
	m_strLog = m_ss.str();
	m_LogQueue->push(std::move(m_strLog));
}

int CGrabThread::GetGrabCount()
{
	return m_nFrameCount;
}

float CGrabThread::GetFrameFreq()
{
	return m_fFramFreq;
}

void CGrabThread::SetGain(double fGain)
{
	if (m_grab == NULL)
	{
		return;
	}
	const char name[] = "Gain";
	//double fTemp = fGain;
	double fTemp = 2.2;
	m_grab->SetFeatureValue(name, (void*)&fTemp);
}

double CGrabThread::GetGain()
{
	if (m_grab == NULL)
	{
		return -1;
	}
	const char name[] = "Gain";
	double fTemp = 0;
	m_grab->GetFeatureValue(name, (void*)&fTemp);
	return fTemp;
}

void CGrabThread::SetExposureTime(double fTime)
{
	if (m_grab == NULL)
	{
		return;
	}
	const char name[] = "ExposureTime";
	double fTemp = fTime;
	m_grab->SetFeatureValue(name, (void*)&fTemp);
}

double CGrabThread::GetExposureTime()
{
	if (m_grab == NULL)
	{
		return -1;
	}
	const char name[] = "ExposureTime";
	double fTemp = 0;
	m_grab->GetFeatureValue(name, (void*)&fTemp);
	return fTemp;
}

void CGrabThread::SetGainByProdct(float fThickness)
{
	double fThick1 = 16.0;
	double fThick2 = 32.0;

	double fGain1 = 2.0;
	double fGain2 = 2.5;

	double slope = (fGain2 - fGain1) / (fThick2 - fThick1);
	double offset = fGain2 - slope * fThick2;

	double dGain = slope * fThickness + offset;

	SetGain(dGain);
}
