#include "stdafx.h"
#include "CheckProc.h"

CCheckProc::CCheckProc()
{
	SetParam();
}


CCheckProc::~CCheckProc()
{
	m_vecKernelSize.clear();
	m_vecSigma.clear();
	m_vecSplit.clear();
	m_vecTruthRect.clear();

	m_vecDOGKernelFFT.clear();;
	m_vecDiffKernelSize.clear();
	if (m_iMarkDOG==2)
	{
		for (int i = 0; i < m_vecFFTWBackward.size(); i++)
		{
			fftwf_destroy_plan(m_vecFFTWBackward[i]);
		}
		for (int i = 0; i < m_vecFFTWForward.size(); i++)
		{
			fftwf_destroy_plan(m_vecFFTWForward[i]);
		}
	}

	m_vecBuffPatch.clear();
	m_vecComplexImg.clear(); 
	m_vecComplexMul.clear();
	m_vecFFTWForward.clear(); 
	m_vecFFTWBackward.clear();
	m_mapThreadIndex.clear();
	m_vecSearchBoundary.clear();
	m_vecGeoClassifiyParam.clear();
	for (int i = 0; i < m_vecvecBoundary.size(); i++)
	{
		m_vecvecBoundary[i].clear();
	}
	m_vecvecBoundary.clear();
}

void CCheckProc::SetParam()
{
	ParamHelper<Parameters::InspectParam> help_InspectParam(&Global::g_param);
	m_inspectParam = help_InspectParam.getRef();
	ParamHelper<Parameters::RunTimeParam> help_RunTimeParam(&Global::g_param);
	m_runtimeParam = help_RunTimeParam.getRef();
	ParamHelper<Parameters::SystemRectifyParam> help_SystemRectifyParam(&Global::g_param);
	m_systemRectiyParm = help_SystemRectifyParam.getRef();
	//系统校正参数
	m_fPhysicResolution = m_systemRectiyParm.physicresolution(); //单位： pix/mm
	//设置线程数并创建线程
	executor.SetSize(unsigned short(m_runtimeParam.threadnum()));
	//切分数据流
	m_iDowmSampleParam = m_runtimeParam.dowmsample();
	float fScale = 1.0;
	cv::Size SplitSize, PaddingSize;
	if (m_iDowmSampleParam == 0 || m_iDowmSampleParam > 5)
	{
		m_iDowmSampleParam = 0;
		m_imgSize.width = m_runtimeParam.freamwidth();
		m_imgSize.height = m_runtimeParam.freamheight();
		SplitSize.width = m_runtimeParam.splitsizex();
		SplitSize.height = m_runtimeParam.splitsizey();
		PaddingSize.width = std::min(5, SplitSize.width>>1);
		PaddingSize.height = std::min(5, SplitSize.height >> 1);
	}
	else
	{
		m_imgSize.width = m_runtimeParam.freamwidth() >> m_iDowmSampleParam;
		m_imgSize.height = m_runtimeParam.freamheight() >> m_iDowmSampleParam;
		SplitSize.width = m_runtimeParam.splitsizex() >> m_iDowmSampleParam;
		SplitSize.height = m_runtimeParam.splitsizey() >> m_iDowmSampleParam;
		PaddingSize.width = std::min(5, SplitSize.width >> 1);
		PaddingSize.height = std::min(5, SplitSize.height >> 1);

		fScale = 1.0 / float(1 << m_iDowmSampleParam);
	}
	if (m_inspectParam.checktype() == Parameters::InspectParam::MAXMIN)
	{
		m_iTrainCount = 0;
		m_iThreshold_low = std::max(m_inspectParam.maxminparam().threshlow(), 10);
		m_iThreshold_high = std::max(m_inspectParam.maxminparam().threshigh(), 10);
		
		m_iThreshold_low = std::min(m_iThreshold_low, 110);
		m_iThreshold_high = std::min(m_iThreshold_high, 110);
	}
	if (m_inspectParam.checktype() == Parameters::InspectParam::EXPVAR)
	{
		m_iTrainCount = 0;
		m_fSigmaTime = m_inspectParam.expvarparam().sigmatime();
		m_fSigmaTime = std::min(m_fSigmaTime, 3.0f);
		m_fSigmaTime = std::max(m_fSigmaTime, 0.1f);
	}
	if (m_inspectParam.checktype() == Parameters::InspectParam::DOG)
	{
		m_iMarkDOG = m_runtimeParam.markdog();
		int iKernelNum = m_inspectParam.dogparam().defectkernellist_size();
		m_vecKernelSize.resize(iKernelNum + 1);
		m_vecSigma.resize(iKernelNum + 1);

		m_vecKernelSize[0].width = m_inspectParam.dogparam().refkernel().sizex();
		m_vecKernelSize[0].height = m_inspectParam.dogparam().refkernel().sizey();
		m_vecSigma[0] = m_inspectParam.dogparam().refkernel().sigma();


		for (int i = 0; i < iKernelNum; i++)
		{
			m_vecKernelSize[i+1].width = m_inspectParam.dogparam().defectkernellist(i).sizex();
			m_vecKernelSize[i+1].height = m_inspectParam.dogparam().defectkernellist(i).sizey();
			m_vecSigma[i+1] = m_inspectParam.dogparam().defectkernellist(i).sigma();
		}


		if (m_iDowmSampleParam != 0)
		{
			for (int i = 0; i < iKernelNum+1; i++)
			{
				m_vecKernelSize[i].width >>= m_iDowmSampleParam;
				m_vecKernelSize[i].height >>= m_iDowmSampleParam;
			}
		}

		int padW = 0;
		int padH = 0;
		for (int i = 0; i < iKernelNum+1; i++)
		{
			if (m_vecKernelSize[i].width % 2 == 0)
			{
				m_vecKernelSize[i].width++;
			}
			if (m_vecKernelSize[i].height % 2 == 0)
			{
				m_vecKernelSize[i].height++;
			}
			if (i!=0)
			{
				padH = std::max(padH, m_vecKernelSize[i].height >> 1);
				padW = std::max(padW, m_vecKernelSize[i].width >> 1);
			}
		}
		if (m_iMarkDOG>0)// if the reference filter size smaller than other, the fft result will be wrong
		{
			PaddingSize.width = std::min((m_vecKernelSize[0].width >> 1) /*+ padW*/, SplitSize.width >> 1);
			PaddingSize.height = std::min((m_vecKernelSize[0].height >> 1)/* + padH*/, SplitSize.height >> 1);
		}

		m_iThresholdDark = m_inspectParam.dogparam().thresholddark();
		m_iThresholdLight = m_inspectParam.dogparam().thresholdlight();
		m_iThresholdDark = std::min(-30, m_iThresholdDark);
		m_iThresholdDark = std::max(m_iThresholdDark, -5);
		m_iThresholdLight = std::min(5, m_iThresholdLight);
		m_iThresholdLight = std::max(m_iThresholdLight, 30);
	}
	
	
	CInspectionAlogrithm::SplitImg(m_imgSize, SplitSize, PaddingSize, m_vecSplit, m_vecTruthRect);

	if (m_inspectParam.checktype() == Parameters::InspectParam::DOG && m_iMarkDOG==1)
	{
		m_fftNormSize = SplitSize + PaddingSize + PaddingSize - cv::Size(1,1);
		bool hr = CInspectionAlogrithm::GetFFTParam(m_fftNormSize, m_vecKernelSize, m_vecSigma, m_vecDOGKernelFFT, m_vecDiffKernelSize);
		if (hr == false)
		{
			//抛出异常
		}
	}

	if (m_inspectParam.checktype() == Parameters::InspectParam::DOG && m_iMarkDOG == 2)
	{
		//
		m_fftNormSize = SplitSize + PaddingSize + PaddingSize - cv::Size(1, 1);
		cv::Size MaxSize = CInspectionAlogrithm::GetMaxSize(m_vecKernelSize);
		cv::Size FFTWsize = MaxSize + m_fftNormSize - cv::Size(1, 1);
		cv::Size diffSize = FFTWsize - SplitSize;
		//
		
		bool hr = CInspectionAlogrithm::GetFFTwParam(m_fftNormSize, m_vecKernelSize, m_vecSigma, m_vecDOGKernelFFT, m_vecDiffKernelSize);
		if (hr == false)
		{
			//抛出异常
		}
		m_vecBuffPatch.resize(m_runtimeParam.threadnum()); 
		m_vecComplexImg.resize(m_runtimeParam.threadnum());
		m_vecComplexMul.resize(m_runtimeParam.threadnum());
		m_vecFFTWForward.resize(m_runtimeParam.threadnum());
		m_vecFFTWBackward.resize(m_runtimeParam.threadnum());
		std::vector<std::future<void>> vecReturn;
		double t1 = cvGetTickCount();
		for (int i = 0; i < m_runtimeParam.threadnum(); i++)
		{
			m_vecBuffPatch[i] = cv::Mat::zeros(FFTWsize.height, FFTWsize.width, CV_32F);
			m_vecComplexImg[i] = cv::Mat(FFTWsize.height, (FFTWsize.width >> 1) + 1, CV_32FC2);
			m_vecComplexMul[i] = m_vecComplexImg[i].clone();

			//vecReturn.push_back(executor.commit(CInspectionAlogrithm::InitiaFFtw, &m_vecFFTWForward[i], &m_vecFFTWBackward[i], &m_vecBuffPatch[i], &m_vecComplexImg[i], &m_vecComplexMul[i]));
			m_mapThreadIndex[executor.GetThreadId(i)] = i;

			m_vecFFTWForward[i] = fftwf_plan_dft_r2c_2d(m_vecBuffPatch[i].rows, m_vecBuffPatch[i].cols, (float*)m_vecBuffPatch[i].data, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>((float*)m_vecComplexImg[i].data)), FFTW_MEASURE);
			m_vecFFTWBackward[i] = fftwf_plan_dft_c2r_2d(m_vecBuffPatch[i].rows, m_vecBuffPatch[i].cols, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>((float*)m_vecComplexMul[i].data)), (float*)m_vecBuffPatch[i].data, FFTW_MEASURE);
		}
		double t2 = cvGetTickCount();
		
		printf("fftw handle Initia is OK, time=%f\n", (t2 - t1) / (1000 * cvGetTickFrequency()));
	}

	if (m_inspectParam.checktype() == Parameters::InspectParam::DOG && m_iMarkDOG == 3)
	{
		//
		m_fftNormSize = SplitSize + PaddingSize + PaddingSize - cv::Size(1, 1);
		cv::Size MaxSize = CInspectionAlogrithm::GetMaxSize(m_vecKernelSize);
		cv::Size FFTWsize = MaxSize + m_fftNormSize - cv::Size(1, 1);
		cv::Size diffSize = FFTWsize - SplitSize;
		m_iDiffKernelNum = int(m_vecKernelSize.size() - 1) + 1;//m_vecKernelSize第一个为背景滤波，再加上一个原图DOG
		//
		bool hr = CInspectionAlogrithm::GetFFTwParam3(m_fftNormSize, m_vecKernelSize, m_vecSigma, m_DogKernelFFtw, m_sumKernelSize);
		if (hr == false)
		{
			//抛出异常
		}
		m_vecBuffPatch.resize(m_runtimeParam.threadnum());
		m_vecComplexImg.resize(m_runtimeParam.threadnum());
		m_vecComplexMul.resize(m_runtimeParam.threadnum());
		m_vecFFTWForward.resize(m_runtimeParam.threadnum());
		m_vecFFTWBackward.resize(m_runtimeParam.threadnum());
		std::vector<std::future<void>> vecReturn;
		double t1 = cvGetTickCount();
		for (int i = 0; i < m_runtimeParam.threadnum(); i++)
		{
			m_vecBuffPatch[i] = cv::Mat::zeros(FFTWsize.height, FFTWsize.width, CV_32F);
			m_vecComplexImg[i] = cv::Mat(FFTWsize.height, (FFTWsize.width >> 1) + 1, CV_32FC2);
			m_vecComplexMul[i] = m_vecComplexImg[i].clone();
			//vecReturn.push_back(executor.commit(CInspectionAlogrithm::InitiaFFtw, &m_vecFFTWForward[i], &m_vecFFTWBackward[i], &m_vecBuffPatch[i], &m_vecComplexImg[i], &m_vecComplexMul[i]));
			m_mapThreadIndex[executor.GetThreadId(i)] = i;

			m_vecFFTWForward[i] = fftwf_plan_dft_r2c_2d(m_vecBuffPatch[i].rows, m_vecBuffPatch[i].cols, (float*)m_vecBuffPatch[i].data, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>((float*)m_vecComplexImg[i].data)), FFTW_MEASURE);
			//forwardKernel = fftwf_plan_dft_r2c_2d(buffPatch.rows, buffPatch.cols, (float*)buffPatch.data, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>((float*)out.data)), FFTW_MEASURE);
			m_vecFFTWBackward[i] = fftwf_plan_dft_c2r_2d(m_vecBuffPatch[i].rows, m_vecBuffPatch[i].cols, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>((float*)m_vecComplexMul[i].data)), (float*)m_vecBuffPatch[i].data, FFTW_MEASURE);
		}
		double t2 = cvGetTickCount();
		printf("fftw handle Initia is OK, time=%f\n", (t2 - t1) / (1000 * cvGetTickFrequency()));
	}

	//Search Boundary Param
	m_vecSearchBoundary.resize(m_inspectParam.boundsearchlist_size());//if boundary param list size is 0, then, set one product default, and left boundary search param is from 0 to half cols. 
	for (int i = 0; i < m_inspectParam.boundsearchlist_size(); i++)
	{
		int iLeft1 = int(m_inspectParam.boundsearchlist(i).leftrange1() * m_fPhysicResolution + 0.5f) >> m_iDowmSampleParam;
		int iLeft2 = int(m_inspectParam.boundsearchlist(i).leftrange2() * m_fPhysicResolution + 0.5f) >> m_iDowmSampleParam;
		int iRight1 = int(m_inspectParam.boundsearchlist(i).rightrange1() * m_fPhysicResolution + 0.5f) >> m_iDowmSampleParam;
		int iRight2 = int(m_inspectParam.boundsearchlist(i).rightrange2() * m_fPhysicResolution + 0.5f) >> m_iDowmSampleParam;
		if (iLeft2<iLeft1 || iRight2<iRight1 || iLeft1<0 || iRight1<0 || iLeft2>m_imgSize.width-1 || iRight2 > m_imgSize.width-1)
		{
			//抛出异常
		}

		m_vecSearchBoundary[i] = std::make_pair(std::make_pair(iLeft1, iLeft2), std::make_pair(iRight1, iRight2));
	}
	m_vecvecBoundary.clear();
	//Blob Filter Param
	m_iBlobThrsh = int(m_inspectParam.defectfilter().blobthr() * m_fPhysicResolution * m_fPhysicResolution + 0.5f);
	m_iBoundaryOffset = int(m_inspectParam.defectfilter().boundoffset() * m_fPhysicResolution + 0.5f);
	if (m_iDowmSampleParam != 0)
	{
		m_iBlobThrsh = m_iBlobThrsh >> (m_iDowmSampleParam + 1);
		m_iBoundaryOffset = m_iBoundaryOffset >> m_iDowmSampleParam;
	}
	//defect classifiy Param
	if (m_inspectParam.classificcationtype() == Parameters::InspectParam::Geometry)
	{
		m_iMarkClassification = 0;
		int iClassNum = m_inspectParam.geoparamlist_size();
		m_vecGeoClassifiyParam.resize(iClassNum);

		for (int i = 0; i < iClassNum; i++)
		{
			m_vecGeoClassifiyParam[i].fMinWidth = m_inspectParam.geoparamlist(i).minwidth() * m_fPhysicResolution * fScale;
			m_vecGeoClassifiyParam[i].fMaxWidth = m_inspectParam.geoparamlist(i).maxwidth() * m_fPhysicResolution * fScale;

			m_vecGeoClassifiyParam[i].fMinHeight = m_inspectParam.geoparamlist(i).minheight() * m_fPhysicResolution * fScale;
			m_vecGeoClassifiyParam[i].fMaxHeight = m_inspectParam.geoparamlist(i).maxheight() * m_fPhysicResolution * fScale;

			m_vecGeoClassifiyParam[i].iMinDiff = m_inspectParam.geoparamlist(i).mindiff();
			m_vecGeoClassifiyParam[i].iMaxDiff = m_inspectParam.geoparamlist(i).maxdiff();

			m_vecGeoClassifiyParam[i].fMinArea = m_inspectParam.geoparamlist(i).minarea() * m_fPhysicResolution * m_fPhysicResolution * fScale * fScale;
			m_vecGeoClassifiyParam[i].fMaxArea = m_inspectParam.geoparamlist(i).maxarea() * m_fPhysicResolution * m_fPhysicResolution * fScale * fScale;

			m_vecGeoClassifiyParam[i].iDefectType = m_inspectParam.geoparamlist(i).defectid();
		}
	}
	else
	{
		m_iMarkClassification = 1;
	}
	//flat field
	m_iDstFrd = m_inspectParam.dstvaluefrd();
	m_iDstBgd = m_inspectParam.dstvaluebgd();
}

void CCheckProc::InspectImage(cv::Mat& fream, std::vector<cv::Mat>& tempFilterImg, cv::Mat& filterImg)
{
	tempFilterImg.clear();
	cv::Mat SrcImg;
	if (fream.channels()!=1)
	{
		if (fream.channels()==3)
		{
			cv::cvtColor(fream, SrcImg, cv::COLOR_RGB2GRAY);
		}
		if (fream.channels()==4)
		{
			cv::cvtColor(fream, SrcImg, cv::COLOR_RGBA2GRAY);
		}
	}
	else
	{
		SrcImg = fream;
	}
	
	//DeNosie...
	if (m_iDowmSampleParam!=0)
	{
		cv::resize(SrcImg, SrcImg, m_imgSize, 0, 0, cv::INTER_CUBIC);
	}
	else
	{
		//CPreProcess::DeNoise(SrcImg, SrcImg);
	}

	if (m_inspectParam.checktype()==Parameters::InspectParam::DOG)
	{
		int iGussianKernelSize = m_vecKernelSize.size();
		tempFilterImg.resize(iGussianKernelSize);
		for (int i = 0; i < iGussianKernelSize; i++)
		{
			tempFilterImg[i] = cv::Mat::zeros(m_imgSize, CV_16S);
		}

		std::vector<std::future<bool>> vecReturn;
		if (m_iMarkDOG==0)
		{
			filterImg = cv::Mat::zeros(m_imgSize, CV_16S);
			for (int i = 0; i < m_vecSplit.size(); i++)
			{
				/*vecReturn.push_back(executor.commit([&]()->bool{
					return CInspectionAlogrithm::DOGCheck4GaussianThread(SrcImg, m_vecSplit[i], m_vecTruthRect[i], m_vecKernelSize, m_vecSigma, filterImg, i);
				}));*/
				vecReturn.push_back(executor.commit(CInspectionAlogrithm::DOGCheck4GaussianThread, SrcImg, m_vecSplit[i], m_vecTruthRect[i], m_vecKernelSize, m_vecSigma, filterImg, i));
			}
			for (int i = 0; i < m_vecSplit.size(); i++)
			{
				printf("gaussian: %d %d\n", i, int(vecReturn[i].get()));
			}
		}
		else if (m_iMarkDOG==1)
		{
			for (int i = 0; i < m_vecSplit.size(); i++)
			{
				vecReturn.push_back(executor.commit(CInspectionAlogrithm::DOGCheck4FFTThread, SrcImg, m_vecSplit[i], m_vecTruthRect[i], m_fftNormSize, m_vecDOGKernelFFT, m_vecDiffKernelSize,
					tempFilterImg, i));
			}
			for (int i = 0; i < m_vecSplit.size(); i++)
			{
				printf("dft: %d %d\n", i, int(vecReturn[i].get()));
			}
		}
		else if (m_iMarkDOG == 2)
		{
			for (int i = 0; i < m_vecSplit.size(); i++)
			{
				//std::thread::id thr_id = std::this_thread::get_id();
				vecReturn.push_back(executor.commit(CInspectionAlogrithm::DOGCheck4FFTwThread, SrcImg, m_vecSplit[i], m_vecTruthRect[i], m_fftNormSize, m_vecDOGKernelFFT, m_vecDiffKernelSize,
					m_vecFFTWForward, m_vecFFTWBackward, m_vecBuffPatch, m_vecComplexImg, m_vecComplexMul, tempFilterImg, m_mapThreadIndex, i));
			}
			for (int i = 0; i < m_vecSplit.size(); i++)
			{
				printf("fftw: %d %d\n", i, int(vecReturn[i].get()));
			}
		}
		else
		{
			filterImg = cv::Mat::zeros(m_imgSize, CV_16S);
			for (int i = 0; i < m_vecSplit.size(); i++)
			{
				//std::thread::id thr_id = std::this_thread::get_id();
				vecReturn.push_back(executor.commit(CInspectionAlogrithm::DOGCheck4FFTwThread3, SrcImg, m_vecSplit[i], m_vecTruthRect[i], m_fftNormSize, m_iDiffKernelNum, m_DogKernelFFtw, m_sumKernelSize,
					m_vecFFTWForward, m_vecFFTWBackward, m_vecBuffPatch, m_vecComplexImg, m_vecComplexMul, filterImg, m_mapThreadIndex, i));
			}
			for (int i = 0; i < m_vecSplit.size(); i++)
			{
				printf("sum fftw: %d %d\n", i, int(vecReturn[i].get()));
			}
		}
		
	}
}

void CCheckProc::InspectImage(std::shared_ptr<GrabImgInfo> img)
{
	if (img->iMark==GrabImgInfo::_ignore_)
	{
		return;
	}
	cv::Mat SrcImg;
	if (img->srcimg.channels() != 1)
	{
		if (img->srcimg.channels() == 3)
		{
			cv::cvtColor(img->srcimg, SrcImg, cv::COLOR_RGB2GRAY);
		}
		if (img->srcimg.channels() == 4)
		{
			cv::cvtColor(img->srcimg, SrcImg, cv::COLOR_RGBA2GRAY);
		}
	}
	else
	{
		SrcImg = img->srcimg.clone();
	}

	//------------------------------public:
	//DeNosie.....
	if (m_iDowmSampleParam != 0)
	{
		cv::resize(SrcImg, SrcImg, m_imgSize, 0, 0, cv::INTER_CUBIC);
	}
	else
	{
		CInspectionAlogrithm::DeNoise(SrcImg, SrcImg);
	}

	//Search Boundary.....
	if (CInspectionAlogrithm::BoundarySearch(SrcImg, m_vecSearchBoundary, m_vecvecBoundary) == false)
	{
		//抛出异常....
	}

	cv::Mat b_frd;
	ImageInspectResult defectInfo;
	defectInfo.srcImage = SrcImg.clone();
	defectInfo.m_vecDefectList.clear();

	if (CInspectionAlogrithm::GetFrdMask(m_imgSize, m_vecvecBoundary, b_frd) == false)
	{
		//抛出异常....
	}
	//------------------------------public:
	if (img->iMark == GrabImgInfo::ProcessType::_flatfield_)
	{
		bool hr = CInspectionAlogrithm::ImageAdd(SrcImg, m_bgdFream, m_frdFream, b_frd, m_iTrainCount++);
		if (hr == false)
		{
			//抛出异常
		}
		CInspectionAlogrithm::GetFlatFieldParam(m_bgdFream, m_frdFream, m_bgdParam, m_frdParam, m_iDstFrd, m_iDstBgd);
		CInspectionAlogrithm::FlatField(SrcImg, m_bgdParam, m_frdParam, b_frd);
		return;
	}

	if (img->iMark == GrabImgInfo::ProcessType::_rectify_)
	{

		return;
	}

	if (img->iMark == GrabImgInfo::ProcessType::_train_)
	{
		if (m_inspectParam.checktype() == Parameters::InspectParam::MAXMIN)
		{
			CInspectionAlogrithm::GetMaxMinModel(SrcImg, m_rowMin, m_rowMax, m_iTrainCount++);
			CInspectionAlogrithm::ExtendLineToFream(m_imgSize, m_rowMin, m_freamMin);
			CInspectionAlogrithm::ExtendLineToFream(m_imgSize, m_rowMax, m_freamMax);
			m_freamMin.convertTo(m_freamMin, m_freamMin.type(), 1, -m_iThreshold_low);
			m_freamMax.convertTo(m_freamMax, m_freamMax.type(), 1, m_iThreshold_high);
		}
		else if (m_inspectParam.checktype() == Parameters::InspectParam::EXPVAR)
		{
			CInspectionAlogrithm::GetExpVarModel(SrcImg, m_rowExp, m_rowStd, m_iTrainCount++);
			cv::Mat tempMat;
			m_rowStd.convertTo(tempMat, m_rowStd.type(), m_fSigmaTime);
			m_rowMin = m_rowExp - tempMat;
			m_rowMax = m_rowExp + tempMat;
			CInspectionAlogrithm::ExtendLineToFream(m_imgSize, m_rowMin, m_freamMin);
			CInspectionAlogrithm::ExtendLineToFream(m_imgSize, m_rowMax, m_freamMax);
		}
		else
		{
			//抛出异常....
		}

		return;
	}

	
	cv::Mat DiffImg = cv::Mat::zeros(m_imgSize, CV_16S);
	if (img->iMark == GrabImgInfo::ProcessType::_normal_)
	{
		std::vector<std::future<bool>> vecReturn;
		vecReturn.clear();

		if (m_inspectParam.checktype() == Parameters::InspectParam::DOG)
		{
			//std::vector<std::future<bool>> vecReturn;
			if (m_iMarkDOG == 0)
			{
				for (int i = 0; i < m_vecSplit.size(); i++)
				{
					vecReturn.push_back(executor.commit(CInspectionAlogrithm::DOGCheck4GaussianThread, SrcImg, m_vecSplit[i], m_vecTruthRect[i], m_vecKernelSize, m_vecSigma, DiffImg, i));
				}
				for (int i = 0; i < m_vecSplit.size(); i++)
				{
					bool hr = vecReturn[i].get();
					if (hr == false)
					{
						//抛出异常....
					}
					printf("gaussian: %d %d\n", i, int(hr));
				}
				//vecReturn.clear();
			}
			else
			{
				for (int i = 0; i < m_vecSplit.size(); i++)
				{
					//std::thread::id thr_id = std::this_thread::get_id();
					vecReturn.push_back(executor.commit(CInspectionAlogrithm::DOGCheck4FFTwThread3, SrcImg, m_vecSplit[i], m_vecTruthRect[i], m_fftNormSize, m_iDiffKernelNum, m_DogKernelFFtw, m_sumKernelSize,
						m_vecFFTWForward, m_vecFFTWBackward, m_vecBuffPatch, m_vecComplexImg, m_vecComplexMul, DiffImg, m_mapThreadIndex, i));
				}
				for (int i = 0; i < m_vecSplit.size(); i++)
				{
					bool hr = vecReturn[i].get();
					if (hr == false)
					{
						//抛出异常....
					}
					printf("fftw: %d %d\n", i, int(hr));
				}
			}
			//This code will run in multi-threads

			cv::Mat tempMat = (DiffImg > m_iThresholdDark & DiffImg < m_iThresholdLight);
			DiffImg.setTo(0x00, tempMat);
			cv::Mat maskDark = (DiffImg < 0);
			cv::Mat maskLight = (DiffImg > 0);
			cv::Mat darkMat, lightMat;
			DiffImg.convertTo(darkMat, CV_16S, 1, -m_iThresholdDark);
			DiffImg.convertTo(lightMat, CV_16S, 1, -m_iThresholdLight);
			darkMat.setTo(DiffImg, maskDark);
			lightMat.setTo(DiffImg, maskLight);
			//
		}
		else if (m_inspectParam.checktype() == Parameters::InspectParam::MAXMIN)
		{
			for (int i = 0; i < m_vecSplit.size(); i++)
			{
				vecReturn.push_back(executor.commit(CInspectionAlogrithm::MaxMinCheckThread, SrcImg, m_vecSplit[i], m_vecTruthRect[i], m_freamMin, m_freamMax, DiffImg, i));
			}
			for (int i = 0; i < m_vecSplit.size(); i++)
			{
				bool hr = vecReturn[i].get();
				if (hr == false)
				{
					//抛出异常....
				}
				printf("min max check: %d %d\n", i, int(hr));
			}
			vecReturn.clear();
		}
		else if (m_inspectParam.checktype() == Parameters::InspectParam::EXPVAR)
		{
			for (int i = 0; i < m_vecSplit.size(); i++)
			{
				vecReturn.push_back(executor.commit(CInspectionAlogrithm::MaxMinCheckThread, SrcImg, m_vecSplit[i], m_vecTruthRect[i], m_freamMin, m_freamMax, DiffImg, i));
			}
			for (int i = 0; i < m_vecSplit.size(); i++)
			{
				bool hr = vecReturn[i].get();
				if (hr == false)
				{
					//抛出异常....
				}
				printf("mean std check: %d %d\n", i, int(hr));
			}
			vecReturn.clear();
		}
		vecReturn.clear();

		//Blob.....
		cv::Mat mask = (DiffImg != 0);
		mask.setTo(0x00, ~b_frd);

		std::vector<std::vector<cv::Point>> vecvecContours, tempContours;
		vecvecContours.clear();
		tempContours.clear();
		vecReturn.clear();
		std::mutex mtx1, mtx2;
		for (int i = 0; i < m_vecTruthRect.size(); i++)
		{
			vecReturn.push_back(executor.commit(CInspectionAlogrithm::GetBlobThread, mask, m_vecTruthRect[i], &vecvecContours, &tempContours, &mtx1, &mtx2));
		}
		for (int i = 0; i < m_vecSplit.size(); i++)
		{
			bool hr = vecReturn[i].get();
			if (hr == false)
			{
				//抛出异常....
			}
			printf("blob: %d %d\n", i, int(hr));
		}
		vecReturn.clear();
		CInspectionAlogrithm::MergeContour(tempContours, vecvecContours);
		for (int i = 0; i < tempContours.size(); i++)
		{
			tempContours[i].clear();
		}
		tempContours.clear();

		std::vector<std::pair<int, DefectData>> vecDefectIdx;

		if (m_iMarkClassification == 0)
		{
			vecReturn.clear();
			std::mutex mtx;
			for (int i = 0; i < vecvecContours.size(); i++)
			{
				vecReturn.push_back(executor.commit(CInspectionAlogrithm::Geo_BlobToDefectThread, DiffImg, vecvecContours[i], m_iBlobThrsh, m_vecGeoClassifiyParam, &vecDefectIdx, &mtx, i));
			}
			for (int i = 0; i < vecvecContours.size(); i++)
			{
				bool hr = vecReturn[i].get();
				if (hr == false)
				{
					//抛出异常....
				}
				printf("blob geo classifiy: %d %d\n", i, int(hr));
			}
			vecReturn.clear();
		}
		else
		{

		}

		if (vecDefectIdx.size() != 0)
		{
			int iScale = (1 << m_iDowmSampleParam);
			std::vector<DefectData> vecDefect;
			for (int i = 0; i < vecDefectIdx.size(); i++)//还差分条信息
			{
				vecDefectIdx[i].second.fPy_x = vecDefectIdx[i].second.fPy_x * iScale / m_fPhysicResolution;
				vecDefectIdx[i].second.fPy_y = vecDefectIdx[i].second.fPy_y * iScale / m_fPhysicResolution;
				vecDefectIdx[i].second.fPyArea = vecDefectIdx[i].second.fPyArea * iScale * iScale / (m_fPhysicResolution * m_fPhysicResolution);
				vecDefect.push_back(vecDefectIdx[i].second);
			}

		
			defectInfo.m_vecDefectList.swap(vecDefect);
			vecDefect.clear();
		}
		for (int i = 0; i < vecvecContours.size(); i++)
		{
			vecvecContours[i].clear();
		}
		vecvecContours.clear();

		vecDefectIdx.clear();
	}

	//Get Diff image.....
	//std::vector<cv::Mat> vecDiffImg;

	Global::g_InspectQueue.push(std::move(defectInfo));
}


void CCheckProc::StartInspectThread()
{
	m_isInspect = true;
	std::thread t(std::bind(&CCheckProc::Pipline, this));
	t.detach();
}

void CCheckProc::Pipline()
{
	while (1)
	{
		std::vector<cv::Mat> filterImg;
		std::shared_ptr<GrabImgInfo> g = Global::g_ProcessQueue.wait_and_pop();
		InspectImage(g);

		if (m_isInspect==false && (Global::g_ProcessQueue.getPushCount()==Global::g_ProcessQueue.getPopCount()))
		{
			break;
		}
	}

}

void CCheckProc::StopInspectThread()
{
	m_isInspect = false;
}

void CCheckProc::ReTrain()
{
	m_iTrainCount = 0;
	for (int i = 0; i < m_vecvecBoundary.size(); i++)
	{
		m_vecvecBoundary[i].clear();
	}
	m_vecvecBoundary.clear();
}

