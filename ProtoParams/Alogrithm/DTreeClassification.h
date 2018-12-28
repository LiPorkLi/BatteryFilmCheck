#pragma once
#include "ClassifyMethod.h"
#include "AlogrithmBase.h"
class CDTreeClassification : public CClassifyMethod, public CAlogrithmBase
{
public:
	CDTreeClassification(MainParam::param* p, std::shared_ptr<CRunTimeHandle> pHandle);
	~CDTreeClassification();

	bool Train(std::vector<DefectData>& vecDefectList, double* dTime);

	bool classify(std::vector<DefectData>& vecDefectList, double* dTime);
	void SetParam(void* param);
};

