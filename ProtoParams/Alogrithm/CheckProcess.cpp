#include "stdafx.h"
#include "CheckProcess.h"


CCheckProcess::CCheckProcess(MainParam::param* p, std::shared_ptr<CRunTimeHandle> pHandle)
{
	ParamHelper<Parameters::InspectParam> helper(p);
	Parameters::InspectParam inspectParam = helper.getRef();
	if (inspectParam.checktype() == Parameters::InspectParam::DOG)
	{
		//m_DOG = std::make_shared<CDOGCheck>(p, pHandle);
		m_method = std::make_shared<CDOGCheck>(p, pHandle);
		m_iRefKernelWidth = inspectParam.dogparam().refkernel().sizex();
		m_iCheckType = 0;
	}
	else if (inspectParam.checktype() == Parameters::InspectParam::EXPVAR)
	{
		m_method = std::make_shared<CMeanStandardCheck>(p, pHandle);
		m_iCheckType = 1;
	}
	else
	{
		m_method = std::make_shared<CMaxMinValueCheck>(p, pHandle);
		m_iCheckType = 2;
	}
}


CCheckProcess::~CCheckProcess()
{

}

bool CCheckProcess::Execute(cv::Mat& img, cv::Mat& diffImg, double* dTime)
{
	return m_method->check(img, diffImg, dTime);
}

bool CCheckProcess::Execute(cv::cuda::GpuMat& img, cv::cuda::GpuMat& diffImg, double* dTime)
{
	return m_method->check(img, diffImg, dTime);
}

bool CCheckProcess::IsDOG()
{
	if (m_iCheckType==0)
	{
		return true;
	}
	return false;
}
