#include "stdafx.h"
#include "AlogrithmBase.h"

CAlogrithmBase::CAlogrithmBase(MainParam::param* p, std::shared_ptr<CRunTimeHandle> pHandle)
{
	m_pHandle = pHandle;
	m_p = p;
}

CAlogrithmBase::CAlogrithmBase()
{
	int a = 0;
}


CAlogrithmBase::~CAlogrithmBase()
{

}

MainParam::param* CAlogrithmBase::getParam()
{
	return m_p;
}

std::shared_ptr<CRunTimeHandle> CAlogrithmBase::getHandle()
{
	return m_pHandle;
}

void CAlogrithmBase::SetParam(MainParam::param* p)
{
	m_p = p;
}

void CAlogrithmBase::SetRunTimeHandle(std::shared_ptr<CRunTimeHandle> pHandle)
{
	m_pHandle = pHandle;
}
