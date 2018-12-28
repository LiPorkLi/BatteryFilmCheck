#pragma once
#include "threadpool.h"
//#include "cv.h"
//#include "highgui.h"
#include "cv.hpp"
#include "highgui.hpp"
#include <cudafilters.hpp>
#include <cudaarithm.hpp>
#include <cudacodec.hpp>
#include <cudaimgproc.hpp>
#include <cudafeatures2d.hpp>
#include <cudawarping.hpp>
#include <vector>
#include "../GlobalData.h"
#include <cuda_runtime_api.h>
#include <helper_cuda.h>

#define PyhsicToPixel_1D(xy, PhysicResolution_xy, ResampleFactor) (xy * PhysicResolution_xy / (1 << ResampleFactor))
#define PyhsicToPixel_2D(area, PhysicResolution_x, PhysicResolution_y, ResampleFactor) (area * PhysicResolution_x * PhysicResolution_y / (1 << (ResampleFactor<<1)))
#define PixelToPyhsic_1D(xy, PhysicResolution_xy, ResampleFactor) (xy * (1 << ResampleFactor) / PhysicResolution_xy)
#define PixelToPyhsic_2D(area, PhysicResolution_x, PhysicResoulution_y, ResampleFactor) (area * (1 << (ResampleFactor<<1)) / (PhysicResolution_x * PhysicResoulution_y))
#define PixToPix_PreToSrc_1D(x, ResampleFactor) (x<<ResampleFactor)
#define PixToPix_SrcToPre_1D(x, ResampleFactor) (x>>ResampleFactor)
#define PixToPix_PreToSrc_2D(x, ResampleFactor) (x<<(1<<(ResampleFactor<<1)))
#define PixToPix_SrcToPre_2D(x, ResampleFactor) (x>>(1<<(ResampleFactor<<1)))

//#define _DD_
class CRunTimeHandle
{
public:
	CRunTimeHandle(MainParam::param* p, safe_queue<std::string>* log_queue);
	~CRunTimeHandle();

	//
	void PushLog(std::string& log);
	//bool SetSrcImg(cv::Mat& img);
	cv::Size SplitCellSize();
	cv::Size ImageSizePre();
	cv::Size ImageSizeSrc();
	int DownSampleFator();
	int ThreadNum();
	cv::Rect TruthRect(int idx);
	cv::Rect SplitRect(int idx);
	int DataPatchNum();
	std::threadpool* Executor();
	bool IsGpu();
	//cv::cuda::GpuMat* DeviceImage();
	float PhysicResolution_x();
	float PhysicResolution_y();
	int OffsetHeightIndex(); //单帧图像采集后下部分无图像部分的起始索引

	//零点校正
	//int GetLocRelativeZero(int x);
	//bool SetBoundaryPix(int iBoundaryLeft, int iBoundaryRight);
	
	//分条信息
	float GetStripOffset();
	int GetStripNum();
	float GetStripLength(int idx);
	float GetStripAllLength();
	//是否镜像
	bool IsFromLeftToRight();
	//开始采集的位置
	float GetStartLength();
	void SetPhysicLength(int iImageCount);//根据grab数量更新总长度
	float GetPhysicLength();//获取总长度
	void SetStartLength(float fStartLength = 0);//设置起始长度

	void SplitImg(cv::Size& imageSize, cv::Size& cellSize, cv::Size& padding, std::vector<cv::Rect>& vecSplitRect, std::vector<cv::Rect>& vecTruthRect);

	void GetBlobThr(int& iDark, int& iLight);
private:
	cv::Size m_imgSizeSrc, m_imgSizePre, m_SplitSize;
	int m_iDowmSampleParam, m_iThreadNum;
	Parameters::RunTimeParam m_runtimeParam;
	std::vector<cv::Rect> m_vecTruth, m_vecSplit;
	std::threadpool m_executor;
	bool m_isGpu, m_isLeftToRight;
	int m_iOffsetY/*, m_iOffsetX*/;
	//device data
	//cv::cuda::GpuMat dev_image;//预处理后的全局图像在GPU上的备份 
	float m_fPhysicResolution_x, m_fPhysicResolution_y;
	safe_queue<std::string>* m_logQueue;

	//float m_fPhysicBoundaryToZero;
	//int m_iPixBoundaryToZero;

	float m_fPhysicStartLength;//开始采集的位置

	float m_fPhysicHeightPerFream, m_fPhysicAllLength;//每一帧图像物理长度，拍摄总长度
	float m_fStripOffset, m_fAllStripiLength;
	std::vector<float> m_vecStrip;

	std::string m_strLog;
	std::stringstream m_ss;

	int m_iThresholdDark, m_iThresholdLight;
};

