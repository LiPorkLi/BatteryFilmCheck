#include "stdafx.h"
#include "RunTimeHandle.h"



CRunTimeHandle::CRunTimeHandle(MainParam::param* p, safe_queue<std::string>* log_queue)
{
	m_logQueue = log_queue;
	//解析运行参数
	ParamHelper<Parameters::RunTimeParam> help_RunTimeParam(p);
	m_runtimeParam = help_RunTimeParam.getRef();
	//原始图像大小
	m_imgSizeSrc.width = m_runtimeParam.freamwidth();
	m_imgSizeSrc.height = m_runtimeParam.freamheight();

	m_ss.clear();m_ss.str("");
	m_ss << "Image src size: "<<m_imgSizeSrc.width<<" * "<<m_imgSizeSrc.height<<std::endl;
	m_strLog = m_ss.str();
	m_logQueue->push(std::move(m_strLog));

	//是否下采样
	m_iDowmSampleParam = m_runtimeParam.dowmsample();
	if (m_iDowmSampleParam == 0 || m_iDowmSampleParam > 5)
	{
		m_iDowmSampleParam = 0;
		m_imgSizePre = m_imgSizeSrc;
	}
	else
	{
		m_imgSizePre.width = m_imgSizeSrc.width >> m_iDowmSampleParam;
		m_imgSizePre.height = m_imgSizeSrc.height >> m_iDowmSampleParam;
	}

	m_ss.clear();m_ss.str("");
	m_ss << "Down Sample factor is : " << m_iDowmSampleParam << " ,and image pre size is: " << m_imgSizePre.width << " * " << m_imgSizePre.height << std::endl;
	m_strLog = m_ss.str();
	m_logQueue->push(std::move(m_strLog));

	//设置线程数并创建线程
	m_isGpu = m_runtimeParam.isgpu();

	cv::Size PaddingSize;

	m_SplitSize.width = m_runtimeParam.splitsizex();
	m_SplitSize.height = m_runtimeParam.splitsizey();

	//切分数据
	m_SplitSize.width = m_runtimeParam.splitsizex();
	m_SplitSize.height = m_runtimeParam.splitsizey();

	//建立多线程
	//m_executor.SetSize(unsigned short(m_runtimeParam.threadnum()));
	if (m_isGpu)
	{
		//切分数据:gpu模式下直接将数据切分为四份
		m_SplitSize.width = (m_imgSizeSrc.width >> 1);
		m_SplitSize.height = (m_imgSizeSrc.height >> 1);
		//建立多线程
		m_iThreadNum = 4;

		m_ss.clear();m_ss.str("");
		m_ss << "GPU mode is ON\n";
		m_strLog = m_ss.str();
		m_logQueue->push(std::move(m_strLog));
	}
	else
	{
		//切分数据
		m_SplitSize.width = m_runtimeParam.splitsizex();
		m_SplitSize.height = m_runtimeParam.splitsizey();
	
		//建立多线程
		m_iThreadNum = m_runtimeParam.threadnum();

		m_ss.clear();m_ss.str("");
		m_ss << "GPU mode is OFF\n";
		m_strLog = m_ss.str();
		m_logQueue->push(std::move(m_strLog));
	}
	m_executor.SetSize(unsigned short(m_iThreadNum));
	if (m_iDowmSampleParam != 0)
	{
		m_SplitSize.width >>= m_iDowmSampleParam;
		m_SplitSize.height >>= m_iDowmSampleParam;
	}
	PaddingSize.width = std::min(5, m_SplitSize.width >> 1);
	PaddingSize.height = std::min(5, m_SplitSize.height >> 1);
	
	m_vecSplit.clear();
	m_vecTruth.clear();
	SplitImg(m_imgSizePre, m_SplitSize, PaddingSize, m_vecSplit, m_vecTruth);


	ParamHelper<Parameters::SystemRectifyParam> sysHelper(p);
	Parameters::SystemRectifyParam sysParam = sysHelper.getRef();
	m_fPhysicResolution_x = 1.0 / sysParam.xphysicresolution();
	m_fPhysicResolution_y = 1.0 / sysParam.yphysicresolution();
	m_fPhysicHeightPerFream = sysParam.lengthperfream();//每帧图像的产品运动距离，Y方向，单位：mm

	m_iOffsetY = sysParam.offsety();//每帧图像下面有一部分无图像，该参数表示无图部分的固定起始位置

	m_ss.clear();m_ss.str("");
	m_ss << "Pyshic resolution is, x:" << m_fPhysicResolution_x<<" pix/mm, y: " << m_fPhysicResolution_y <<	" pix/mm, Length per fream is:	"<<m_fPhysicHeightPerFream<<"mm\n";
	m_strLog = m_ss.str();
	m_logQueue->push(std::move(m_strLog));

	ParamHelper<Parameters::SheetInfo> sheetHelper(p);
	Parameters::SheetInfo sheetInfo = sheetHelper.getRef();
	m_fStripOffset = sheetInfo.stripparam().leftoffset();//第一刀到零点的尺寸
	//计算零点像素坐标： 输入产品边缘到零点的物理尺寸，然后通过边界反算零点在图像上的像素位置：临时
// 	m_fPhysicBoundaryToZero = sheetInfo.offsetboundarytozero();
// 	if (m_fPhysicBoundaryToZero > m_fStripOffset)
// 	{
// 		m_ss.clear(); m_ss.str("");
// 		m_ss << "Warning: the size which zeros to boundary is :	" << m_fPhysicBoundaryToZero << " mm, and the distance which zero to first strip is: " << m_fStripOffset << " mm, the latter smaller than former\n";
// 		m_strLog = m_ss.str();
// 		m_logQueue->push(std::move(m_strLog));
// 	}
// 	m_iPixBoundaryToZero = PyhsicToPixel_1D(m_fPhysicBoundaryToZero, m_fPhysicResolution_x, m_iDowmSampleParam);

	m_fAllStripiLength = 0.0f;
	m_vecStrip.clear();
	for (int i = 0; i < sheetInfo.stripparam().stripwidth_size(); i++)
	{
		m_vecStrip.push_back(sheetInfo.stripparam().stripwidth(i));
		m_fAllStripiLength += m_vecStrip[i];
	}

	m_isLeftToRight = sheetInfo.islefttoright();
	m_fPhysicStartLength = sheetInfo.startlength();//Y方向起始长度
	m_fPhysicAllLength = m_fPhysicStartLength;//初始化总长度为起始长度


	ParamHelper<Parameters::InspectParam> helper(p);
	Parameters::InspectParam inspectionParam = helper.getRef();
	m_iThresholdDark = inspectionParam.dogparam().thresholddark();
	m_iThresholdLight = inspectionParam.dogparam().thresholdlight();
	m_iThresholdDark = std::max(-30, m_iThresholdDark);
	m_iThresholdDark = std::min(m_iThresholdDark, -5);
	m_iThresholdLight = std::max(5, m_iThresholdLight);
	m_iThresholdLight = std::min(m_iThresholdLight, 30);
}


CRunTimeHandle::~CRunTimeHandle()
{
	m_vecSplit.clear();
	m_vecTruth.clear();
	m_vecStrip.clear();
}


void CRunTimeHandle::SplitImg(cv::Size& imageSize, cv::Size& cellSize, cv::Size& padding, std::vector<cv::Rect>& vecSplitRect, std::vector<cv::Rect>& vecTruthRect)
{
	vecSplitRect.clear();
	vecTruthRect.clear();
	if (cellSize.width > imageSize.width || cellSize.height > imageSize.height)
	{
		vecSplitRect.push_back(cv::Rect(0, 0, imageSize.width, imageSize.height));
		vecTruthRect.push_back(cv::Rect(0, 0, imageSize.width, imageSize.height));
		return;
	}
	int iSplitNum_x = (imageSize.width + cellSize.width - 1) / cellSize.width;
	int iSplitNum_y = (imageSize.height + cellSize.height - 1) / cellSize.height;
	vecSplitRect.resize(iSplitNum_y*iSplitNum_x);
	vecTruthRect.resize(iSplitNum_x*iSplitNum_y);

	int iCount = 0;
	cv::Rect rtPadding, rtTruth;
	for (int i = 0; i < imageSize.height; i += cellSize.height)
	{
		rtPadding.y = i - padding.height + 1;
		if (rtPadding.y < 0)
		{
			rtPadding.y = 0;
		}
		rtPadding.height = i + cellSize.height + padding.height - 1;
		if (rtPadding.height > imageSize.height - 1)
		{
			rtPadding.height = imageSize.height - 1;
		}
		rtPadding.height = rtPadding.height - rtPadding.y + 1;

		rtTruth.y = i;
		rtTruth.height = i + cellSize.height - 1;
		if (rtTruth.height > imageSize.height - 1)
		{
			rtTruth.height = imageSize.height - 1;
		}
		rtTruth.height = rtTruth.height - rtTruth.y + 1;

		for (int j = 0; j < imageSize.width; j += cellSize.width)
		{
			rtPadding.x = j - padding.width + 1;
			if (rtPadding.x < 0)
			{
				rtPadding.x = 0;
			}
			rtPadding.width = j + cellSize.width + padding.width - 1;
			if (rtPadding.width > imageSize.width - 1)
			{
				rtPadding.width = imageSize.width - 1;
			}
			rtPadding.width = rtPadding.width - rtPadding.x + 1;

			rtTruth.x = j;
			rtTruth.width = j + cellSize.width - 1;
			if (rtTruth.width > imageSize.width - 1)
			{
				rtTruth.width = imageSize.width - 1;
			}
			rtTruth.width = rtTruth.width - rtTruth.x + 1;

			vecTruthRect[iCount] = rtTruth;
			vecSplitRect[iCount++] = rtPadding;
		}
	}
}


cv::Size CRunTimeHandle::ImageSizePre()
{
	return m_imgSizePre;
}

int CRunTimeHandle::DownSampleFator()
{
	return m_iDowmSampleParam;
}

int CRunTimeHandle::ThreadNum()
{
// 	if (m_isGpu)
// 	{
// 		return 0;
// 	}
	return m_iThreadNum;
}

cv::Rect CRunTimeHandle::TruthRect(int idx)
{
	return m_vecTruth[idx];
}

cv::Rect CRunTimeHandle::SplitRect(int idx)
{
	return m_vecSplit[idx];
}

int CRunTimeHandle::DataPatchNum()
{
	return int(m_vecSplit.size());
}

std::threadpool* CRunTimeHandle::Executor()
{
// 	if (m_isGpu)
// 	{
// 		return NULL;
// 	}
	return &m_executor;
}

cv::Size CRunTimeHandle::SplitCellSize()
{
	return m_SplitSize;
}

bool CRunTimeHandle::IsGpu()
{
	return m_isGpu;
}

cv::Size CRunTimeHandle::ImageSizeSrc()
{
	return m_imgSizeSrc;
}


float CRunTimeHandle::PhysicResolution_x()
{
	return m_fPhysicResolution_x;
}

float CRunTimeHandle::PhysicResolution_y()
{
	return m_fPhysicResolution_y;
}

void CRunTimeHandle::PushLog(std::string& log)
{
	m_logQueue->push(std::move(log));
}

int CRunTimeHandle::OffsetHeightIndex()//无图部分起始位置
{
	return m_iOffsetY;
}

void CRunTimeHandle::SetPhysicLength(int iImageCount)
{
	m_fPhysicAllLength = m_fPhysicStartLength + iImageCount * m_fPhysicHeightPerFream;
}
float CRunTimeHandle::GetPhysicLength()
{
	return m_fPhysicAllLength;
}
void CRunTimeHandle::SetStartLength(float fStartLength/* = 0*/)
{
	m_fPhysicStartLength = fStartLength;
	m_fPhysicAllLength = fStartLength;
}

float CRunTimeHandle::GetStripOffset()
{
	return m_fStripOffset;
}

int CRunTimeHandle::GetStripNum()
{
	return m_vecStrip.size();
}

float CRunTimeHandle::GetStripLength(int idx)
{
	return m_vecStrip[idx];
}

/*bool CRunTimeHandle::SetBoundaryPix(int iBoundaryLeft, int iBoundaryRight)//暂时只适用零点在拍摄幅面以内
{
	if (m_isLeftToRight==true)
	{
		m_iOffsetX = iBoundaryLeft - m_iPixBoundaryToZero;
		if (m_iOffsetX < 0)
		{
			return false;
		}
	}
	else
	{
		m_iOffsetX = m_imgSizePre.width - iBoundaryRight - m_iPixBoundaryToZero;
		if (m_iOffsetX < 0)
		{
			return false;
		}
	}
	
	return true;
}*/

/*int CRunTimeHandle::GetLocRelativeZero(int x)
{
	if (m_isLeftToRight==true)
	{
		return x - m_iOffsetX;
	}
	else
	{
		return m_imgSizePre.width - x - m_iOffsetX;
	}
}*/

bool CRunTimeHandle::IsFromLeftToRight()
{
	return m_isLeftToRight;
}

float CRunTimeHandle::GetStartLength()
{
	return m_fPhysicStartLength;
}

float CRunTimeHandle::GetStripAllLength()
{
	return m_fAllStripiLength;
}

void CRunTimeHandle::GetBlobThr(int& iDark, int& iLight)
{
	iDark = m_iThresholdDark;
	iLight = m_iThresholdLight;
}

// bool CRunTimeHandle::SetSrcImg(cv::Mat& img)
// {
// 	if (img.channels()!=1 || img.type()!=CV_8U)
// 	{
// 		return false;
// 	}
// 	dev_imgSize.width = img.cols;
// 	dev_imgSize.height = img.rows;
// 
// 	if (dev_SrcImg)
// 	{
// 		cudaFree(&dev_SrcImg);
// 	}
// 
// 	checkCudaErrors(cudaMalloc((void **)&dev_SrcImg, dev_imgSize.height* dev_imgSize.width   * sizeof(unsigned char)));
// 
// 	return true;
// }
// 
