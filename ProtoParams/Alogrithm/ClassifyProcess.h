#pragma once
//#include "RunTimeHandle.h"
#include "GeoClassification.h"
#include "SvmClassification.h"
#include "DTreeClassification.h"
class CClassifyProcess
{
public:
	CClassifyProcess(MainParam::param* p, std::shared_ptr<CRunTimeHandle> pHandle);
	~CClassifyProcess();
	bool Execute(std::vector<DefectData>& vecDefectList, double* dTime);
private:
	std::shared_ptr<CClassifyMethod> m_method;
};

