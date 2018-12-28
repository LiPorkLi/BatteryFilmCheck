#include "stdafx.h"
#include "Rectify.h"

CRectify::CRectify(MainParam::param* p, std::shared_ptr<CRunTimeHandle> pHandle) : CAlogrithmBase(p, pHandle)
{

}


CRectify::~CRectify()
{
}

bool CRectify::GetRectifyModel(cv::Mat& SrcImg, std::vector<float>& vecRectifyParam)
{
	cv::Size dstImgSize = getHandle()->ImageSizePre();

	return true;
}

bool CRectify::Rectify(cv::Mat& SrcDstImg)
{

	return true;
}

void CRectify::SetParam(void* param)
{

}
