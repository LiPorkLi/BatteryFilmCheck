#pragma once
#include "RunTimeHandle.h"
class CClassifyMethod
{
public:
	CClassifyMethod();
	~CClassifyMethod();

	virtual bool classify(std::vector<DefectData>& vecDefectList, double* dTime) = 0;
};

