#pragma once
#include "ClassifyMethod.h"
class CSvmClassification : public CClassifyMethod
{
public:
	CSvmClassification(MainParam::param* p, std::shared_ptr<CRunTimeHandle> pHandle);
	~CSvmClassification();

	bool classify(std::vector<DefectData>& vecDefectList, double* dTime){ return true; };
};

