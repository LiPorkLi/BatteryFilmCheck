#pragma once
#include "ClassifyMethod.h"
#include "AlogrithmBase.h"
class CGeoClassification : public CClassifyMethod, public CAlogrithmBase
{
public:
	CGeoClassification(MainParam::param* p, std::shared_ptr<CRunTimeHandle> pHandle);
	~CGeoClassification();

	bool classify(std::vector<DefectData>& vecDefectList, double* dTime);
	void SetParam(void* param);
private:
	
	std::vector<GeoClassifyModel> m_vecGeoClassifiyParam;
	std::vector<std::future<bool>> m_vecReturn;
	std::vector<float> m_vecStrip;
	float m_fStripOffset;

	bool Geo_BlobToDefectThread(DefectData* defect, std::vector<GeoClassifyModel>* vecClassifyParam);
};

