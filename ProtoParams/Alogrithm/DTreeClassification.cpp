#include "stdafx.h"
#include "DTreeClassification.h"

CDTreeClassification::CDTreeClassification(MainParam::param* p, std::shared_ptr<CRunTimeHandle> pHandle) : CAlogrithmBase(p, pHandle)
{

}


CDTreeClassification::~CDTreeClassification()
{

}

bool CDTreeClassification::classify(std::vector<DefectData>& vecDefectList, double* dTime)
{

	return true;
}

void CDTreeClassification::SetParam(void* param)
{

}

bool CDTreeClassification::Train(std::vector<DefectData>& vecDefectList, double* dTime)
{
	return true;
}
