#pragma once
#include "RunTimeHandle.h"

class CAlogrithmBase
{
public:
	CAlogrithmBase();
	CAlogrithmBase(MainParam::param* p, std::shared_ptr<CRunTimeHandle> pHandle);
	virtual ~CAlogrithmBase();

	virtual void SetParam(void* param) = 0;
	virtual void SetParam(MainParam::param* p);

	virtual void SetRunTimeHandle(std::shared_ptr<CRunTimeHandle> pHandle);
protected:
	MainParam::param* getParam();
	std::shared_ptr<CRunTimeHandle> getHandle();
private:
	std::shared_ptr<CRunTimeHandle> m_pHandle;
	MainParam::param* m_p;
};

