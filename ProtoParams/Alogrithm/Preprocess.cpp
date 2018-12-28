#include "stdafx.h"
#include "Preprocess.h"

CPreprocess::CPreprocess(MainParam::param* p, std::shared_ptr<CRunTimeHandle> pHandle) : CAlogrithmBase(p, pHandle)
{
	m_imgSize = getHandle()->ImageSizePre();
	if (getHandle()->IsGpu()==true)
	{
		m_blur = cv::cuda::createBoxFilter(CV_8U, CV_8U, cv::Size(3, 3));
	}
	m_RectOffset.x = 0;
	m_RectOffset.width = getHandle()->ImageSizeSrc().width;
	m_RectOffset.y = std::max(0, getHandle()->OffsetHeightIndex() - 2);//允许最多掉两根线
	m_RectOffset.height = getHandle()->ImageSizeSrc().height - m_RectOffset.y;

	m_gpu_mask = cv::cuda::GpuMat(getHandle()->ImageSizePre().height, getHandle()->ImageSizePre().width, CV_8U);
	m_gpu_mask4 = cv::cuda::GpuMat(getHandle()->ImageSizePre().height>>2, getHandle()->ImageSizePre().width>>2, CV_8U);
}


CPreprocess::~CPreprocess()
{
	m_vecReturn.clear();
}

bool CPreprocess::Preprocess(cv::Mat& SrcDstImg, double* dTime)
{
	*dTime = cvGetTickCount();
	Padding_Cpu(SrcDstImg, m_RectOffset);
	if (m_imgSize.width!=SrcDstImg.cols || m_imgSize.height!=SrcDstImg.rows)
	{
		cv::resize(SrcDstImg, SrcDstImg, m_imgSize, 0, 0, cv::INTER_LINEAR);
	}
	else
	{
		//根据线程图像情况开启
		/*m_vecReturn.clear();
		for (int i = 0; i < getHandle()->DataPatchNum(); i++)
		{
			m_vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CPreprocess::MeanFilter, this, &SrcDstImg, getHandle()->SplitRect(i), getHandle()->TruthRect(i))));
		}
		for (int i = 0; i < m_vecReturn.size(); i++)
		{
			bool hr = m_vecReturn[i].get();
			if (hr == false)
			{
				//抛出异常....
				printf("PreProcess occurred some unhappy! Info: Rect: x = %d, y= %d, width = %d, height=%d\n", getHandle()->TruthRect(i).x, getHandle()->TruthRect(i).y, getHandle()->TruthRect(i).width, getHandle()->TruthRect(i).height);
				return false;
			}
		}
		m_vecReturn.clear();*/
	}
	*dTime = (cv::getTickCount() - (*dTime)) / (1000 * cvGetTickFrequency());
	return true;
}

bool CPreprocess::Preprocess(cv::cuda::GpuMat& SrcDstImg, bool hr, double* dTime)
{
	*dTime = cvGetTickCount();
	if (hr==true)
	{
		Padding_Gpu(&SrcDstImg, m_RectOffset);
	}

	if (m_imgSize.width != SrcDstImg.cols || m_imgSize.height != SrcDstImg.rows)
	{
		//cv::cuda::GpuMat tempMat(m_imgSize.height, m_imgSize.width, SrcDstImg.type());
		cv::cuda::resize(SrcDstImg, SrcDstImg, m_imgSize, 0, 0, cv::INTER_LINEAR);
		//SrcDstImg = tempMat.clone();
	}
	else
	{
		//m_blur->apply(SrcDstImg, SrcDstImg);//根据现场情况开启
	}
	bool isProdct = true;

	/*cv::cuda::resize(SrcDstImg, m_gpu_mask, cv::Size(m_gpu_mask.cols >> 2, m_gpu_mask.rows >> 2), 0, 0, cv::INTER_NEAREST);
	cv::cuda::threshold(SrcDstImg, m_gpu_mask, 210, 0xff, cv::THRESH_BINARY);
	cv::cuda::resize(m_gpu_mask, m_gpu_mask4, cv::Size(m_gpu_mask.cols >> 2, m_gpu_mask.rows >> 2), 0, 0, cv::INTER_NEAREST);
	cv::Scalar s = cv::cuda::sum(m_gpu_mask4);
	s.val[0] = s.val[0] * 0.003922f;
	s.val[0] = s.val[0] / float(m_gpu_mask4.rows*m_gpu_mask4.cols);
	if (s.val[0] > 0.7f)
	{
		isProdct = false;
	}*/

	*dTime = (cv::getTickCount() - (*dTime)) / (1000 * cvGetTickFrequency());
	return isProdct;
}


bool CPreprocess::MeanFilter(cv::Mat* SrcDstImg, cv::Rect& rtSplit, cv::Rect& rtTruth)
{
	if (rtSplit.x<0 || rtSplit.y<0 || rtTruth.x<0 || rtTruth.y<0 || rtSplit.x + rtSplit.width>SrcDstImg->cols || rtSplit.y + rtSplit.height>SrcDstImg->rows ||
		rtTruth.x + rtTruth.width>SrcDstImg->cols  || rtTruth.y + rtTruth.height>SrcDstImg->rows )
	{
		return false;
	}
	cv::Mat PatchImg = (*SrcDstImg)(rtSplit);
	cv::Mat tempImg;
	cv::blur(PatchImg, tempImg, cv::Size(3, 3));
	cv::Rect Roi(abs(rtSplit.x - rtTruth.x), abs(rtSplit.y - rtTruth.y), rtTruth.width, rtTruth.height);
	tempImg(Roi).copyTo((*SrcDstImg)(rtTruth));

	return true;
}

void CPreprocess::SetParam(void* param)
{

}


bool CPreprocess::Padding_Gpu(cv::cuda::GpuMat* SrcDstImg, cv::Rect& rectOffset)
{
	if (rectOffset.height > SrcDstImg->rows>>1 || rectOffset.width > SrcDstImg->cols || rectOffset.y < SrcDstImg->rows>>1)
	{
		return false;
	}
	PaddingOffset(*SrcDstImg, rectOffset);
	return true;
}

bool CPreprocess::Padding(cv::Mat* SrcDstImg, int iOffset, int iY0, int iY1, int iX, int iRoiWidth)
{
	for (int i = iY0; i < iY1; i++)
	{
		memcpy(SrcDstImg->data + (iOffset + i)*sizeof(uchar)*SrcDstImg->cols + iX,
			SrcDstImg->data + (iOffset - 1 - i)*sizeof(uchar)*SrcDstImg->cols + iX,
			sizeof(uchar)*iRoiWidth);
	}
	return true;
}

bool CPreprocess::Padding_Cpu(cv::Mat& SrcDstImg, cv::Rect& rectOffset)
{
	if (rectOffset.height > SrcDstImg.rows >> 1 || rectOffset.width > SrcDstImg.cols || rectOffset.y < SrcDstImg.rows >> 1)
	{
		return false;
	}
	int iThreads = getHandle()->ThreadNum();
	int iStep = (rectOffset.height + iThreads - 1) / iThreads;
	m_vecReturn.clear();
	int iTemp = 0;
	for (int i = 0; i < iThreads; i++)
	{
		m_vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CPreprocess::Padding, this, &SrcDstImg, rectOffset.y, iTemp, std::min(iTemp + iStep, SrcDstImg.rows), rectOffset.x, rectOffset.width)));
		iTemp += iStep;
	}
	for (int i = 0; i < m_vecReturn.size(); i++)
	{
		bool hr = m_vecReturn[i].get();
		if (hr == false)
		{
			//抛出异常....
			printf("PreProcess occurred some unhappy! Info: Rect: x = %d, y= %d, width = %d, height=%d\n", getHandle()->TruthRect(i).x, getHandle()->TruthRect(i).y, getHandle()->TruthRect(i).width, getHandle()->TruthRect(i).height);
			return false;
		}
	}
	m_vecReturn.clear();

	return true;
}

bool CPreprocess::UnPadding(cv::Mat& SrcDstImg, double* dTime)
{
	*dTime = cvGetTickCount();
	if (SrcDstImg.size()==getHandle()->ImageSizeSrc())
	{
		SrcDstImg(m_RectOffset).setTo(0x00);
	}
	else if (SrcDstImg.size() == getHandle()->ImageSizePre())
	{
		cv::Rect rt;
		rt.x = m_RectOffset.x >> getHandle()->DownSampleFator();
		rt.y = m_RectOffset.y >> getHandle()->DownSampleFator();
		rt.width = SrcDstImg.cols;
		rt.height = SrcDstImg.rows - rt.y;
		SrcDstImg(rt).setTo(0x00);
	}
	else
		return false;

	*dTime = (cv::getTickCount() - (*dTime)) / (1000 * cvGetTickFrequency());
	return true;
}

bool CPreprocess::UnPadding(cv::cuda::GpuMat& SrcDstImg, double* dTime)
{
	*dTime = cvGetTickCount();
	if (SrcDstImg.size() == getHandle()->ImageSizeSrc())
	{
		SrcDstImg(m_RectOffset).setTo(0x00);
	}
	else if (SrcDstImg.size() == getHandle()->ImageSizePre())
	{
		cv::Rect rt;
		rt.x = m_RectOffset.x >> getHandle()->DownSampleFator();
		rt.y = m_RectOffset.y >> getHandle()->DownSampleFator();
		rt.width = SrcDstImg.cols;
		rt.height = SrcDstImg.rows - rt.y;
		SrcDstImg(rt).setTo(0x00);
	}
	else
		return false;

	*dTime = (cv::getTickCount() - (*dTime)) / (1000 * cvGetTickFrequency());
	return true;
}

