#include "stdafx.h"
#include "ClassifyProcess.h"


CClassifyProcess::CClassifyProcess(MainParam::param* p, std::shared_ptr<CRunTimeHandle> pHandle)
{
	ParamHelper<Parameters::InspectParam> helper(p);
	Parameters::InspectParam inspectParam = helper.getRef();

	if (inspectParam.classificcationtype()==Parameters::InspectParam::SVM)
	{
		m_method = std::make_shared<CSvmClassification>(p, pHandle);
	}
	else if (inspectParam.classificcationtype() == Parameters::InspectParam::DTREE)
	{
		m_method = std::make_shared<CDTreeClassification>(p, pHandle);
	}
	else
	{
		m_method = std::make_shared<CGeoClassification>(p, pHandle);
	}
}


CClassifyProcess::~CClassifyProcess()
{
}

bool CClassifyProcess::Execute(std::vector<DefectData>& vecDefectList, double* dTime)
{
	return m_method->classify(vecDefectList, dTime);
}

