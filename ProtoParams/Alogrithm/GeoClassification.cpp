#include "stdafx.h"
#include "GeoClassification.h"
#include <random>

CGeoClassification::CGeoClassification(MainParam::param* p, std::shared_ptr<CRunTimeHandle> pHandle) : CAlogrithmBase(p, pHandle)
{
	ParamHelper<Parameters::InspectParam> helper(getParam());
	Parameters::InspectParam inspectionParam = helper.getRef();

	int iClassNum = inspectionParam.geoparamlist_size();
	m_vecGeoClassifiyParam.resize(iClassNum);

	for (int i = 0; i < iClassNum; i++)
	{
		m_vecGeoClassifiyParam[i].fMinWidth = inspectionParam.geoparamlist(i).minwidth();
		m_vecGeoClassifiyParam[i].fMaxWidth = inspectionParam.geoparamlist(i).maxwidth();

		m_vecGeoClassifiyParam[i].fMinHeight = inspectionParam.geoparamlist(i).minheight();
		m_vecGeoClassifiyParam[i].fMaxHeight = inspectionParam.geoparamlist(i).maxheight();

		m_vecGeoClassifiyParam[i].fMinArea = inspectionParam.geoparamlist(i).minarea();
		m_vecGeoClassifiyParam[i].fMaxArea = inspectionParam.geoparamlist(i).maxarea();

		m_vecGeoClassifiyParam[i].iMinDiff = inspectionParam.geoparamlist(i).mindiff();
		m_vecGeoClassifiyParam[i].iMaxDiff = inspectionParam.geoparamlist(i).maxdiff();

		m_vecGeoClassifiyParam[i].iDefectType = inspectionParam.geoparamlist(i).defectid();
	}
}


CGeoClassification::~CGeoClassification()
{
	m_vecGeoClassifiyParam.clear();
	m_vecReturn.clear();
	m_vecStrip.clear();
}

bool CGeoClassification::Geo_BlobToDefectThread(DefectData* defect, std::vector<GeoClassifyModel>* vecClassifyParam)
{
	defect->defectName = std::string("未分类");
	char strLabel[8];
	for (int i = 0; i < vecClassifyParam->size(); i++)
	{
		if (defect->fPy_height < (*vecClassifyParam)[i].fMinHeight || defect->fPy_height >(*vecClassifyParam)[i].fMaxHeight)
		{
			continue;
		}

		if (defect->fPy_width <  (*vecClassifyParam)[i].fMinWidth || defect->fPy_width >(*vecClassifyParam)[i].fMaxWidth)
		{
			continue;
		}

		if (defect->iMeanDiff < (*vecClassifyParam)[i].iMinDiff || defect->iMeanDiff  >(*vecClassifyParam)[i].iMaxDiff)
		{
			continue;
		}

		if (defect->fPyArea <  (*vecClassifyParam)[i].fMinArea || defect->fPyArea >(*vecClassifyParam)[i].fMaxArea)
		{
			continue;
		}
		//defect->iDefectType = (*vecClassifyParam)[i].iDefectType;
		sprintf_s(strLabel, "%d", (*vecClassifyParam)[i].iDefectType);
		defect->defectName = std::string(strLabel);
		break;
	}
	return true;
}


bool CGeoClassification::classify(std::vector<DefectData>& vecDefectList, double* dTime)
{
	*dTime = cvGetTickCount();
	std::vector<std::pair<int, DefectData>> vecDefectIdx;
	//vecDefectList.clear();

	m_vecReturn.clear();
	std::mutex mtx;
	for (int i = 0; i < vecDefectList.size(); i++)
	{
		m_vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CGeoClassification::Geo_BlobToDefectThread, this, &vecDefectList[i], &m_vecGeoClassifiyParam)));
	}
	for (int i = 0; i < m_vecReturn.size(); i++)
	{
		bool hr = m_vecReturn[i].get();
		if (hr == false)
		{
			//抛出异常....
			printf("Geo Defect classifiy occurred some unhappy! Info: blob = %d\n", i);
			vecDefectIdx.clear();
			return false;
		}
	}
	m_vecReturn.clear();

#ifdef _DD_
	std::default_random_engine e;
	std::uniform_int_distribution<> u(0, m_vecGeoClassifiyParam.size() + 1);
	for (auto&& d : vecDefectList)
	{
		d.iDefectType = u(e);
	}
#endif
	

	*dTime = (cv::getTickCount() - (*dTime)) / (1000 * cvGetTickFrequency());
	return true;
}


void CGeoClassification::SetParam(void* param)
{

}
