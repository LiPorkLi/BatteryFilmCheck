#pragma once
#include "AlogrithmBase.h"
#include "XCMobileCls.h"
class CClassify : public CAlogrithmBase
{
public:
	CClassify(MainParam::param* p, std::shared_ptr<CRunTimeHandle> pHandle);
	~CClassify();

	int Clssify(std::shared_ptr<ImageInspectResult> buff, double* dTime);
	bool Clssify(ImageInspectResult* buff, double* dTime);
	void SetParam(void* param) {};
private:
	int m_iProdctType;
	std::string m_modelPath;
	CXCCls* m_pCls;
	struct Extr_Info
	{
		float fAvgDiff, fMaxMinDiff, fAvgMangnitude, fNoValidPer;
	};
	//ÅÐ¶ÏÁ¬Ðø·Ï
	int m_iCountImg, m_iCountDefect, m_iLastIndex;

	void GetStatus(cv::Mat& src0, cv::Mat& diff0, DefectData& defect, Extr_Info& p);
};

