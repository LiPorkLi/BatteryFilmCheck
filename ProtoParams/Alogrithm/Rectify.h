#pragma once
#include "AlogrithmBase.h"
class CRectify : public CAlogrithmBase
{
public:
	CRectify(MainParam::param* p, std::shared_ptr<CRunTimeHandle> pHandle);
	~CRectify();

	bool GetRectifyModel(cv::Mat& SrcImg, std::vector<float>& vecRectifyParam);
	bool Rectify(cv::Mat& SrcDstImg);
	void SetParam(void* param);
private:
};

