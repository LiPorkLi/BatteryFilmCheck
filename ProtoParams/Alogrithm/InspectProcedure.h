#pragma once

// #include "threadpool.h"
// #include "InspectionAlogrithm.h"
// #include "GlobalData.h"
#include "RunTimeHandle.h"
#include "PreProcess.h"
#include "BoudarySearch.h"
#include "CheckProcess.h"
#include "BlobAnalysis.h"
//#include "ClassifyProcess.h"
#include "FlatField.h"
#include "Classify.h"

class CInspectProcedure
{
public:
	CInspectProcedure();
	~CInspectProcedure();

	void SetParam(MainParam::param* p, safe_queue<GrabImgInfo>* queue_grab, safe_queue<ImageInspectResult>* queue_result, safe_queue<std::string>* queue_log);
	void StartInspectThread(bool isUpdate, float fStartLength = -1);
	void StopInspectThread();
	float GetLength();
private:
	safe_queue<GrabImgInfo>* m_queue_grab;
	safe_queue<ImageInspectResult>* m_queue_result;
	bool m_isInspect, *m_isStop;
	std::shared_ptr<CRunTimeHandle> m_run;
	std::shared_ptr<CPreprocess> m_preprocess;
	std::shared_ptr<CBoudarySearch> m_boudarysearch;
	std::shared_ptr<CCheckProcess> m_check;
	std::shared_ptr<CBlobAnalysis> m_blobanalysis;
	//std::shared_ptr<CClassifyProcess> m_defectclassification;
	std::shared_ptr<CClassify> m_cls;
	std::shared_ptr<CFlatField> m_flat;
	std::vector<double> m_vecTime;

	std::thread m_inspectThread;

	std::thread m_applyMemoryThread;
	//ImageInspectResult m_get;
	safe_queue<ImageInspectResult> m_queue_mem;
	bool *m_isPush;

	cv::Mat m_SrcImg, m_FrdMask, m_DiffImg16S, m_TempImg/*, m_DiffImg8U*/;
	cv::cuda::GpuMat m_SrcImg_gpu, m_FrdMask_gpu, m_DiffImg_gpu16S, m_DiffImg_gpu8U;
	ImageInspectResult m_TempResullt;

	//uchar* m_pHost8u;

	int m_iImageCount;
	//bool m_bStatusBoundarySearch;
	//GrabImgInfo::ProcessType m_bType;

	std::string m_strLog;
	std::stringstream m_ss;

	void ApplyMemory();
	void Pipline();

	uchar InspectImage(std::shared_ptr<GrabImgInfo> grabImg, std::shared_ptr<ImageInspectResult> buff);
	void CopyMat(cv::Mat* src, cv::Mat* dst, cv::Rect rt);
};

