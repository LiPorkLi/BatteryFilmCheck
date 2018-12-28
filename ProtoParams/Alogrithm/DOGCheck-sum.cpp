#include "stdafx.h"
#include "DOGCheck.h"
#include "cudaFunction.h"

CDOGCheck::CDOGCheck(MainParam::param* p, std::shared_ptr<CRunTimeHandle> pHandle) : CAlogrithmBase(p, pHandle)
{
	ParamHelper<Parameters::InspectParam> helper(getParam());
	Parameters::InspectParam inspectionParam = helper.getRef();
	ParamHelper<Parameters::SystemRectifyParam> sysHelper(getParam());
	Parameters::SystemRectifyParam sysParam = sysHelper.getRef();
	//float fPhysicResolution = sysParam.physicresolution();
	int iDownSampleParam = getHandle()->DownSampleFator();
	
	cv::Size PaddingSize;
	cv::Size SplitSize = getHandle()->SplitCellSize();
	cv::Size ImgSize = getHandle()->ImageSizePre();
	
	int iKernelNum = inspectionParam.dogparam().defectkernellist_size();
	m_vecKernelSize.resize(iKernelNum + 1);
	m_vecSigma.resize(iKernelNum + 1);

	m_vecKernelSize[0].width = inspectionParam.dogparam().refkernel().sizex();
	m_vecKernelSize[0].height = inspectionParam.dogparam().refkernel().sizey();
	m_vecSigma[0] = inspectionParam.dogparam().refkernel().sigma();


	for (int i = 0; i < iKernelNum; i++)
	{
		m_vecKernelSize[i + 1].width = inspectionParam.dogparam().defectkernellist(i).sizex();
		m_vecKernelSize[i + 1].height = inspectionParam.dogparam().defectkernellist(i).sizey();
		m_vecSigma[i + 1] = inspectionParam.dogparam().defectkernellist(i).sigma();
	}


	for (int i = 0; i < iKernelNum + 1; i++)
	{
		m_vecKernelSize[i].width >>= iDownSampleParam;
		m_vecKernelSize[i].height >>= iDownSampleParam;
	}

	for (int i = 0; i < iKernelNum + 1; i++)
	{
		if (m_vecKernelSize[i].width % 2 == 0)
		{
			m_vecKernelSize[i].width++;
		}
		if (m_vecKernelSize[i].height % 2 == 0)
		{
			m_vecKernelSize[i].height++;
		}
	}

	m_iThresholdDark = inspectionParam.dogparam().thresholddark();
	m_iThresholdLight = inspectionParam.dogparam().thresholdlight();
	m_iThresholdDark = std::max(-30, m_iThresholdDark);
	m_iThresholdDark = std::min(m_iThresholdDark, -5);
	m_iThresholdLight = std::max(5, m_iThresholdLight);
	m_iThresholdLight = std::min(m_iThresholdLight, 30);

	//m_pSrc = m_pComplex = m_pDOGKerelComplex = NULL;
	m_pSrc = m_pKernel = NULL;
	m_pConv = NULL;
	if (getHandle()->IsGpu()==false)
	{
		//测试用
		ParamHelper<Parameters::RunTimeParam> runHelper(getParam());
		Parameters::RunTimeParam runParam = runHelper.getRef();
		m_iMarkDOG = runParam.markdog();
		if (m_iMarkDOG > 0)// if the reference filter size smaller than other, the fft result will be wrong
		{
			PaddingSize.width = std::min((m_vecKernelSize[0].width >> 1), SplitSize.width >> 1);
			PaddingSize.height = std::min((m_vecKernelSize[0].height >> 1), SplitSize.height >> 1);
			getHandle()->SplitImg(ImgSize, SplitSize, PaddingSize, m_vecSplit, m_vecTruth);
		}
		else
		{
			for (int i = 0; i < getHandle()->DataPatchNum(); i++)
			{
				m_vecSplit.push_back(getHandle()->SplitRect(i));
				m_vecTruth.push_back(getHandle()->TruthRect(i));
			}
		}
		if (m_iMarkDOG == 1)
		{
			m_fftNormSize = SplitSize + PaddingSize + PaddingSize - cv::Size(1, 1);
			bool hr = GetFFTParam(m_fftNormSize, m_vecKernelSize, m_vecSigma, m_vecDOGKernelFFT, m_vecDiffKernelSize);
			if (hr == false)
			{
				//抛出异常
			}
		}

		if (m_iMarkDOG == 2)
		{
			//
			int iThreadNum = getHandle()->ThreadNum();
			m_fftNormSize = SplitSize + PaddingSize + PaddingSize - cv::Size(1, 1);
			cv::Size MaxSize = GetMaxSize(m_vecKernelSize);
			cv::Size FFTWsize = MaxSize + m_fftNormSize - cv::Size(1, 1);
			cv::Size diffSize = FFTWsize - SplitSize;
			//

			bool hr = GetFFTwParam(m_fftNormSize, m_vecKernelSize, m_vecSigma, m_vecDOGKernelFFT, m_vecDiffKernelSize);
			if (hr == false)
			{
				//抛出异常
			}
			m_vecBuffPatch.resize(iThreadNum);
			m_vecComplexImg.resize(iThreadNum);
			m_vecComplexMul.resize(iThreadNum);
			m_vecFFTWForward.resize(iThreadNum);
			m_vecFFTWBackward.resize(iThreadNum);

			double t1 = cvGetTickCount();
			for (int i = 0; i < iThreadNum; i++)
			{
				m_vecBuffPatch[i] = cv::Mat::zeros(FFTWsize.height, FFTWsize.width, CV_32F);
				m_vecComplexImg[i] = cv::Mat(FFTWsize.height, (FFTWsize.width >> 1) + 1, CV_32FC2);
				m_vecComplexMul[i] = m_vecComplexImg[i].clone();

				//vecReturn.push_back(executor.commit(CInspectionAlogrithm::InitiaFFtw, &m_vecFFTWForward[i], &m_vecFFTWBackward[i], &m_vecBuffPatch[i], &m_vecComplexImg[i], &m_vecComplexMul[i]));
				m_mapThreadIndex[getHandle()->Executor()->GetThreadId(i)] = i;

				m_vecFFTWForward[i] = fftwf_plan_dft_r2c_2d(m_vecBuffPatch[i].rows, m_vecBuffPatch[i].cols, (float*)m_vecBuffPatch[i].data, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>((float*)m_vecComplexImg[i].data)), FFTW_MEASURE);
				m_vecFFTWBackward[i] = fftwf_plan_dft_c2r_2d(m_vecBuffPatch[i].rows, m_vecBuffPatch[i].cols, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>((float*)m_vecComplexMul[i].data)), (float*)m_vecBuffPatch[i].data, FFTW_MEASURE);
			}
			double t2 = cvGetTickCount();

			printf("fftw handle Initia is OK, time=%f\n", (t2 - t1) / (1000 * cvGetTickFrequency()));
		}

		if (m_iMarkDOG == 3)
		{
			//
			int iThreadNum = getHandle()->ThreadNum();
			m_fftNormSize = SplitSize + PaddingSize + PaddingSize - cv::Size(1, 1);
			//		cv::Size MaxSize = GetMaxSize(m_vecKernelSize);
			// 		cv::Size FFTWsize = MaxSize + m_fftNormSize - cv::Size(1, 1);
			// 		cv::Size diffSize = FFTWsize - SplitSize;
			m_iDiffKernelNum = int(m_vecKernelSize.size() - 1) + 1;//m_vecKernelSize第一个为背景滤波，再加上一个原图DOG
			//
			bool hr = GetFFTwParam3(m_fftNormSize, m_vecKernelSize, m_vecSigma, m_DogKernelFFtw, m_sumKernelSize);
			if (hr == false)
			{
				//抛出异常
			}
			m_vecBuffPatch.resize(iThreadNum);
			m_vecComplexImg.resize(iThreadNum);
			m_vecComplexMul.resize(iThreadNum);
			m_vecFFTWForward.resize(iThreadNum);
			m_vecFFTWBackward.resize(iThreadNum);
			double t1 = cvGetTickCount();
			for (int i = 0; i < iThreadNum; i++)
			{
				m_vecBuffPatch[i] = cv::Mat::zeros(m_fftNormSize.height, m_fftNormSize.width, CV_32F);
				m_vecComplexImg[i] = cv::Mat(m_fftNormSize.height, (m_fftNormSize.width >> 1) + 1, CV_32FC2);
				m_vecComplexMul[i] = m_vecComplexImg[i].clone();
				//vecReturn.push_back(executor.commit(CInspectionAlogrithm::InitiaFFtw, &m_vecFFTWForward[i], &m_vecFFTWBackward[i], &m_vecBuffPatch[i], &m_vecComplexImg[i], &m_vecComplexMul[i]));
				m_mapThreadIndex[getHandle()->Executor()->GetThreadId(i)] = i;

				m_vecFFTWForward[i] = fftwf_plan_dft_r2c_2d(m_vecBuffPatch[i].rows, m_vecBuffPatch[i].cols, (float*)m_vecBuffPatch[i].data, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>((float*)m_vecComplexImg[i].data)), FFTW_MEASURE);
				//forwardKernel = fftwf_plan_dft_r2c_2d(buffPatch.rows, buffPatch.cols, (float*)buffPatch.data, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>((float*)out.data)), FFTW_MEASURE);
				m_vecFFTWBackward[i] = fftwf_plan_dft_c2r_2d(m_vecBuffPatch[i].rows, m_vecBuffPatch[i].cols, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>((float*)m_vecComplexMul[i].data)), (float*)m_vecBuffPatch[i].data, FFTW_MEASURE);
			}
			double t2 = cvGetTickCount();
			printf("fftw handle Initia is OK, time=%f\n", (t2 - t1) / (1000 * cvGetTickFrequency()));
		}
		if (m_iMarkDOG == 4)
		{
			//GPU : split data to 4 patchs
			m_fftNormSize = SplitSize + PaddingSize + PaddingSize - cv::Size(1, 1);
		}
	}
	else
	{
		double t1 = cvGetTickCount();
		PaddingSize.width = std::min((m_vecKernelSize[0].width >> 1), SplitSize.width >> 1);
		PaddingSize.height = std::min((m_vecKernelSize[0].height >> 1), SplitSize.height >> 1);
		getHandle()->SplitImg(ImgSize, SplitSize, PaddingSize, m_vecSplit, m_vecTruth);

		m_fftNormSize = SplitSize + PaddingSize + PaddingSize - cv::Size(1, 1);
		m_iDiffKernelNum = int(m_vecKernelSize.size() - 1) + 1;//m_vecKernelSize第一个为背景滤波，再加上一个原图DOG
		//
		/*int iComplexCol = (m_fftNormSize.width >> 1) + 1;
		int iComplexChannel = 2;
		cudaMalloc((void**)(&m_pComplex), sizeof(float)*m_fftNormSize.height*iComplexCol* iComplexChannel);
		cudaMalloc((void**)(&m_pDOGKerelComplex), sizeof(float)*m_fftNormSize.height*iComplexCol* iComplexChannel);
		cudaMalloc((void**)(&m_pSrc), sizeof(float)*m_fftNormSize.width*m_fftNormSize.height);
		m_BuffPatch_gpu = cv::cuda::GpuMat(m_fftNormSize.height, m_fftNormSize.width, CV_32F, m_pSrc, sizeof(float)*m_fftNormSize.width);
		m_ComplexImg_gpu = cv::cuda::GpuMat(m_fftNormSize.height, iComplexCol, CV_32FC(iComplexChannel), m_pComplex, sizeof(float)*iComplexCol*iComplexChannel);
		m_DogKernelFFtw_gpu = cv::cuda::GpuMat(m_fftNormSize.height, iComplexCol, CV_32FC(iComplexChannel), m_pDOGKerelComplex, sizeof(float)*iComplexCol*iComplexChannel);
		m_FFTWForward_gpu = fftwf_plan_dft_r2c_2d(m_fftNormSize.height, m_fftNormSize.width, m_pSrc, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>(m_pComplex)), FFTW_MEASURE);
		m_FFTWBackward_gpu = fftwf_plan_dft_c2r_2d(m_fftNormSize.height, m_fftNormSize.width, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>(m_pComplex)), m_pSrc, FFTW_MEASURE);*/

		
		m_pConv = new CConvolutionFFT2D;
		InitiaConvolution(m_vecKernelSize, m_vecSigma);
		bool hr = m_pConv->Initia(m_BuffPatch_gpu.rows, m_BuffPatch_gpu.cols, m_BuffPatch_gpu.step, m_Kernel_gpu.rows, m_Kernel_gpu.cols);
		if (hr == false)
		{
			return;
		}
		hr = m_pConv->SetKernel((float*)m_Kernel_gpu.data, m_Kernel_gpu.step);
		if (hr == false)
		{
			return;
		}

// 		bool hr = GetFFTwParam3_gpu(m_vecKernelSize, m_vecSigma, m_sumKernelSize);
// 		if (hr == false)
// 		{
// 			//抛出异常
// 		}
		double t2 = cvGetTickCount();
		printf("fftw handle Initia is OK, time=%f\n", (t2 - t1) / (1000 * cvGetTickFrequency()));
	}
}


CDOGCheck::~CDOGCheck()
{
	for (int i = 0; i < m_vecReturn.size(); i++)
	{
		m_vecReturn[i].get();
	}
	m_vecReturn.clear();

	m_vecKernelSize.clear();
	m_vecSigma.clear();
	m_vecSplit.clear();
	m_vecTruth.clear();

	m_vecDOGKernelFFT.clear();;
	m_vecDiffKernelSize.clear();
	if (m_iMarkDOG>=2)
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
	if (m_pConv)
	{
		delete m_pConv;
	}
	m_pConv = NULL;

	m_vecBuffPatch.clear();
	m_vecComplexImg.clear();
	m_vecComplexMul.clear();
	m_vecFFTWForward.clear();
	m_vecFFTWBackward.clear();
	m_mapThreadIndex.clear();

	if (m_pSrc)
	{
		cudaFree(&m_pSrc);
	}
	if (m_pKernel)
	{
		cudaFree(&m_pKernel);
	}
	if (m_pDst)
	{
		cudaFree(&m_pDst);
	}
// 	if (m_pDOGKerelComplex)
// 	{
// 		cudaFree(&m_pDOGKerelComplex);
// 	}
// 	if (m_pComplex)
// 	{
// 		cudaFree(&m_pComplex);
// 	}

// 	m_vecBuffPatch_gpu.clear(); 
// 	m_vecComplexImg_gpu.clear();
// 	m_vecComplexMul_gpu.clear();
}

cv::Mat CDOGCheck::GetDiffKernel3(std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma, cv::Size& refSize, float fRefSigma)
{
	if (vecKernelSize.size() == 0)
	{
		return cv::Mat();
	}

	int iNum = int(vecKernelSize.size());
	cv::Size maxSize = refSize;
	for (int i = 0; i < vecKernelSize.size(); i++)
	{
		maxSize.width = std::max(maxSize.width, vecKernelSize[i].width);
		maxSize.height = std::max(maxSize.height, vecKernelSize[i].height);
	}

	cv::Mat RefKernel = GetGaussKernel(refSize, fRefSigma, fRefSigma);
	cv::Mat tempAdd = cv::Mat::zeros(maxSize, CV_32F);
	cv::Mat tempRef = tempAdd.clone();
	cv::Mat tempMat = tempAdd.clone();
	RefKernel.convertTo(tempRef(cv::Rect((maxSize.width >> 1) - (RefKernel.cols >> 1), (maxSize.height >> 1) - (RefKernel.rows >> 1), RefKernel.cols, RefKernel.rows)), CV_32F, double(vecKernelSize.size()));
	for (int i = 0; i < vecKernelSize.size(); i++)
	{
		cv::Mat kernel = GetGaussKernel(vecKernelSize[i], vecSigma[i], vecSigma[i]);
		tempMat.setTo(0x00);
		kernel.copyTo(tempMat(cv::Rect((maxSize.width >> 1) - (kernel.cols >> 1), (maxSize.height >> 1) - (kernel.rows >> 1), kernel.cols, kernel.rows)));
		tempAdd = tempAdd + tempMat;
	}
	tempMat = tempAdd - tempRef;
	return tempMat;
}

bool CDOGCheck::GetFFTwParam3(cv::Size& imgSize, std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma, cv::Mat& DOGKernelFFT, cv::Size& diffKernelSize)
{
	if (vecKernelSize.size() != vecSigma.size() || vecKernelSize.size() == 0)
	{
		return false;
	}
	int iKernelNum = int(vecKernelSize.size() - 1);
	cv::Size RefKernelSize = vecKernelSize[0];
	float fRefSigma = vecSigma[0];

	std::vector<cv::Size> tempKernel;
	tempKernel.push_back(cv::Size(1, 1));
	std::vector<float> tempSigma;
	tempSigma.push_back(0);
	for (int i = 0; i < iKernelNum; i++)
	{
		tempKernel.push_back(vecKernelSize[i + 1]);
		tempSigma.push_back(vecSigma[i + 1]);
	}
	cv::Mat diffKernel = GetDiffKernel3(tempKernel, tempSigma, RefKernelSize, fRefSigma);
	cv::Mat tempMat;
	cv::flip(diffKernel, tempMat, -5);
	diffKernel = tempMat.clone();

	tempKernel.clear();
	tempSigma.clear();

	diffKernelSize.width = diffKernel.cols;
	diffKernelSize.height = diffKernel.rows;
	int iFFTWRow = imgSize.height/* + diffKernelSize.height - 1*/;
	int iFFTWCol = imgSize.width/* + diffKernelSize.width - 1*/;
	cv::Mat buffPatch = cv::Mat::zeros(iFFTWRow, iFFTWCol, CV_32F);
	DOGKernelFFT = cv::Mat(iFFTWRow, (iFFTWCol >> 1) + 1, CV_32FC2);
	fftwf_plan forwardKernel;

	forwardKernel = fftwf_plan_dft_r2c_2d(buffPatch.rows, buffPatch.cols, (float*)buffPatch.data, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>((float*)DOGKernelFFT.data)), FFTW_MEASURE);
	buffPatch.setTo(0x00);
	//diffKernel.copyTo(buffPatch(cv::Rect(0, 0, diffKernel.cols, diffKernel.rows)));

	KernelCopyToBuff(diffKernel, buffPatch);

	fftwf_execute(forwardKernel);//buffPatch->out
	fftwf_destroy_plan(forwardKernel);

	return true;
}


bool CDOGCheck::GetFFTwParam(cv::Size& imgSize, std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma, std::vector<cv::Mat>& vecDOGKernelFFT, std::vector<cv::Size>& vecDiffKernelSize)
{
	vecDOGKernelFFT.clear();
	if (vecKernelSize.size() != vecSigma.size() || vecKernelSize.size() == 0)
	{
		return false;
	}
	int iKernelNum = int(vecKernelSize.size() - 1);
	cv::Size RefKernelSize = vecKernelSize[0];
	float fRefSigma = vecSigma[0];

	cv::Size MaxKernelSize = GetMaxSize(vecKernelSize);
	int iFFTWRow = imgSize.height + MaxKernelSize.height - 1;
	int iFFTWCol = imgSize.width + MaxKernelSize.width - 1;
	cv::Mat buffPatch = cv::Mat::zeros(iFFTWRow, iFFTWCol, CV_32F);
	cv::Mat out = cv::Mat(iFFTWRow, (iFFTWCol >> 1) + 1, CV_32FC2);
	fftwf_plan forwardKernel;

	//forwardImg = fftwf_plan_dft_r2c_2d(buffPatch.rows, buffPatch.cols, (float*)buffPatch.data, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>((float*)buffComplexImg.data)), FFTW_MEASURE);
	forwardKernel = fftwf_plan_dft_r2c_2d(buffPatch.rows, buffPatch.cols, (float*)buffPatch.data, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>((float*)out.data)), FFTW_MEASURE);
	//backward = fftwf_plan_dft_c2r_2d(buffPatch.rows, buffPatch.cols, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>((float*)buffComplexBackward.data)), (float*)buffPatch.data, FFTW_MEASURE);

	std::vector<cv::Mat> vecKernel;
	vecKernel.push_back(GetDiffKernel2(cv::Size(1, 1), RefKernelSize, MaxKernelSize, 0, fRefSigma).clone());
	if (iKernelNum != 0)
	{
		for (int i = 1; i < vecKernelSize.size(); i++)
		{
			vecKernel.push_back(GetDiffKernel2(vecKernelSize[i], RefKernelSize, MaxKernelSize, vecSigma[i], fRefSigma));
		}
	}
	for (int i = 0; i < vecKernel.size(); i++)
	{
		buffPatch.setTo(0x00);
		vecKernel[i].copyTo(buffPatch(cv::Rect(0, 0, vecKernel[i].cols, vecKernel[i].rows)));
		fftwf_execute(forwardKernel);//buffPatch->out
		vecDOGKernelFFT.push_back(out.clone());
		vecDiffKernelSize.push_back(cv::Size(vecKernel[i].cols, vecKernel[i].rows));
	}

	//fftwf_destroy_plan(forwardImg);
	fftwf_destroy_plan(forwardKernel);
	//fftwf_destroy_plan(backward);

	return true;
}


cv::Size CDOGCheck::GetMaxSize(std::vector<cv::Size>& vecKernelSize)
{
	cv::Size refSize(0, 0);
	for (int i = 0; i < vecKernelSize.size(); i++)
	{
		refSize.width = std::max(refSize.width, vecKernelSize[i].width);
		refSize.height = std::max(refSize.height, vecKernelSize[i].height);
	}
	return refSize;
}


cv::Mat CDOGCheck::GetDiffKernel1(cv::Size& s1, cv::Size& s2, float fSigma1, float fSigma2)
{
	cv::Mat g1, g2, diff_g, diff_f;
	cv::Mat kernel1 = GetGaussKernel(s1, fSigma1, fSigma1);
	cv::Mat kernel2 = GetGaussKernel(s2, fSigma2, fSigma2);
	int kw = s1.width + s2.width - 1;
	int kh = s1.height + s2.height - 1;
	cv::Mat temp_k1 = cv::Mat::zeros(kh, kw, CV_32F);//temp_k3_1 = zeros(kh3, kw3);
	cv::Mat temp_k2 = temp_k1.clone();// = zeros(kh3, kw3);
	kernel1.copyTo(temp_k1(cv::Rect(s2.width >> 1, s2.height >> 1, kernel1.cols, kernel1.rows)));
	kernel2.copyTo(temp_k2(cv::Rect(s1.width >> 1, s1.height >> 1, kernel2.cols, kernel2.rows)));
	cv::Mat Kernel3 = temp_k1 - temp_k2;
	return Kernel3;
}

cv::Mat CDOGCheck::GetDiffKernel2(cv::Size& s1, cv::Size& s2, cv::Size& maxSize, float fSigma1, float fSigma2)
{
	if (s1.width > maxSize.width || s1.height > maxSize.height || s2.width > maxSize.width || s2.height > maxSize.height)
	{
		return cv::Mat();
	}
	cv::Mat temp_k1 = cv::Mat::zeros(maxSize.height, maxSize.width, CV_32F);//temp_k3_1 = zeros(kh3, kw3);
	cv::Mat temp_k2 = temp_k1.clone();// = zeros(kh3, kw3);
	cv::Mat kernel1 = GetGaussKernel(s1, fSigma1, fSigma1);
	cv::Mat kernel2 = GetGaussKernel(s2, fSigma2, fSigma2);
	kernel1.copyTo(temp_k1(cv::Rect((maxSize.width >> 1) - (kernel1.cols >> 1), (maxSize.height >> 1) - (kernel1.rows >> 1), kernel1.cols, kernel1.rows)));
	kernel2.copyTo(temp_k2(cv::Rect((maxSize.width >> 1) - (kernel2.cols >> 1), (maxSize.height >> 1) - (kernel2.rows >> 1), kernel2.cols, kernel2.rows)));
	// 	kernel1.copyTo(temp_k1(cv::Rect(s2.width >> 1, s2.height >> 1, kernel1.cols, kernel1.rows)));
	// 	kernel2.copyTo(temp_k2(cv::Rect(s1.width >> 1, s1.height >> 1, kernel2.cols, kernel2.rows)));
	cv::Mat Kernel3 = temp_k1 - temp_k2;
	return Kernel3;
}
cv::Mat CDOGCheck::GetGaussKernel(cv::Size& kernelSize, float fSigmaW /*= 0.0f*/, float fSigmaH /*= 0.0f*/)
{
	cv::Mat kernel_w = cv::getGaussianKernel(kernelSize.width, fSigmaW, CV_32F);
	cv::transpose(kernel_w, kernel_w);
	cv::Mat kernel_h = cv::getGaussianKernel(kernelSize.height, fSigmaH, CV_32F);
	cv::Mat Kernel = kernel_h * kernel_w;
	return Kernel.clone();
}

bool CDOGCheck::GetFFTParam(cv::Size& imgSize, std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma, std::vector<cv::Mat>& vecDOGKernelFFT, std::vector<cv::Size>& vecDiffKernelSize)
{
	vecDOGKernelFFT.clear();
	if (vecKernelSize.size() != vecSigma.size() || vecKernelSize.size() == 0)
	{
		return false;
	}
	int iKernelNum = int(vecKernelSize.size() - 1);
	cv::Size RefKernelSize = vecKernelSize[0];
	float fRefSigma = vecSigma[0];
	cv::Mat padded, dft_img, dft_kernel, c_m;
	int m = cv::getOptimalDFTSize(imgSize.height);
	int n = cv::getOptimalDFTSize(imgSize.width);
	padded = cv::Mat::zeros(m, n, CV_32F);

	std::vector<cv::Mat> vecKernel;
	vecKernel.push_back(GetDiffKernel1(cv::Size(1, 1), RefKernelSize, 0, fRefSigma).clone());
	if (iKernelNum != 0)
	{
		for (int i = 1; i < vecKernelSize.size(); i++)
		{
			vecKernel.push_back(GetDiffKernel1(vecKernelSize[i], RefKernelSize, vecSigma[i], fRefSigma));
		}
	}
	for (int i = 0; i < vecKernel.size(); i++)
	{
		padded.setTo(0x00);
		vecKernel[i].copyTo(padded(cv::Rect(0, 0, vecKernel[i].cols, vecKernel[i].rows)));
		cv::dft(padded, dft_kernel, 0, padded.rows);
		vecDOGKernelFFT.push_back(std::move(dft_kernel));
		vecDiffKernelSize.push_back(cv::Size(vecKernel[i].cols, vecKernel[i].rows));
	}

	return true;
}


bool CDOGCheck::DOGCheck4GaussianThread(cv::Mat* img, cv::Rect& rtSplit, cv::Rect& rtTruth, std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma, cv::Mat* diffImg, int iDarkThr, int iLightThr)
{
	if (vecKernelSize.size() != vecSigma.size() || vecKernelSize.size() == 0 || diffImg->size != img->size)
	{
		return false;
	}
	// 	if (vecRlt.size!=img.size)
	// 	{
	// 		vecRlt = cv::Mat::zeros(img.rows, img.cols, CV_16S);
	// 	}
	cv::Mat PatchImg = (*img)(rtSplit);

	//std::vector<cv::Mat> vecR;
	//vecR.clear();
	cv::Mat diffRlt = cv::Mat::zeros(PatchImg.rows, PatchImg.cols, CV_32S);

	int iKernelNum = int(vecKernelSize.size() - 1);
	cv::Size RefKernelSize = vecKernelSize[0];
	float fRefSigma = vecSigma[0];
	cv::Mat RefImg, SrcImg32f, rltImg;
	cv::GaussianBlur(PatchImg, RefImg, RefKernelSize, fRefSigma, fRefSigma);
	//cv::GaussianBlur(PatchImg, RefImg, cv::Size(1, RefKernelSize.height), 0, fRefSigma);
	//cv::GaussianBlur(RefImg, RefImg, cv::Size(RefKernelSize.width, 1), fRefSigma, 0);
	RefImg.convertTo(RefImg, CV_32S);
	PatchImg.convertTo(SrcImg32f, CV_32S);
	rltImg = SrcImg32f - RefImg;
	//vecR.push_back(std::move(rltImg));
	diffRlt = diffRlt + rltImg;

	cv::Mat tempMat;
	if (iKernelNum != 0)
	{
		for (int i = 1; i < vecKernelSize.size(); i++)
		{
			cv::GaussianBlur(PatchImg, tempMat, vecKernelSize[i], vecSigma[i], vecSigma[i]);
			tempMat.convertTo(tempMat, CV_32S);
			rltImg = tempMat - RefImg;
			//vecR.push_back(std::move(rltImg));
			diffRlt = diffRlt + rltImg;
		}
	}
	diffRlt.convertTo(diffRlt, CV_16S, 1.0 / double(vecKernelSize.size()));

	tempMat = (diffRlt > iDarkThr & diffRlt < iLightThr);
	diffRlt.setTo(0x00, tempMat);
	cv::Mat maskDark = (diffRlt < 0);
	cv::Mat maskLight = (diffRlt > 0);
	cv::Mat darkMat, lightMat;
	diffRlt.convertTo(darkMat, CV_16S, 1, -iDarkThr);
	diffRlt.convertTo(lightMat, CV_16S, 1, -iLightThr);
	darkMat.copyTo(diffRlt, maskDark);
	lightMat.copyTo(diffRlt, maskLight);

	cv::Rect Roi(abs(rtSplit.x - rtTruth.x), abs(rtSplit.y - rtTruth.y), rtTruth.width, rtTruth.height);
	diffRlt(Roi).copyTo((*diffImg)(rtTruth));
	/*for (int i = 0; i < vecR.size(); i++)
	{
	vecR[i](Roi).copyTo(vecRlt[i](rtTruth));
	}
	vecR.clear();*/
	//printf("%d  split\n", id);
	return true;
}


bool CDOGCheck::DOGCheck4FFTwThread3(cv::Mat* img, cv::Rect& rtSplit, cv::Rect& rtTruth, cv::Size& normSize, int iDiffKernelNum, cv::Mat* KernelFFT, cv::Size& DiffKernelSize, 
	std::vector<fftwf_plan>* vecForwardImg, std::vector<fftwf_plan>* vecBackward, std::vector<cv::Mat>* vecBuffPatch, std::vector<cv::Mat>* vecBuffComplexImg, 
	std::vector<cv::Mat>* vecBuffComplexBackward, cv::Mat* diffImg, std::map<std::thread::id, int>* threadIndex, int iDarkThr, int iLightThr)
{
	//std::cout << " thread_id is " << std::this_thread::get_id() << std::endl;
	if (rtSplit.width > normSize.width || rtSplit.height > normSize.height ||
		vecForwardImg->size() == 0 || vecBackward->size() == 0 || vecBuffPatch->size() == 0 || vecBuffComplexImg->size() == 0 || vecBuffComplexBackward->size() == 0 || threadIndex->size() == 0)
	{
		return false;
	}
	//
	std::thread::id thr_id = std::this_thread::get_id();
	if (threadIndex->count(thr_id) == 0)
	{
		return false;
	}
	int idx = (*threadIndex)[thr_id];

	cv::Mat PatchImg = (*img)(rtSplit);

	(*vecBuffPatch)[idx].setTo(0x00);
	PatchImg.convertTo((*vecBuffPatch)[idx](cv::Rect(0, 0, PatchImg.cols, PatchImg.rows)), CV_32F);
	fftwf_execute((*vecForwardImg)[idx]);//buffPatch->buffComplexImg

	cv::Mat DstImg = cv::Mat::zeros(PatchImg.rows, PatchImg.cols, CV_32F);
	cv::mulSpectrums((*KernelFFT), (*vecBuffComplexImg)[idx], (*vecBuffComplexBackward)[idx], cv::DFT_COMPLEX_OUTPUT);
	fftwf_execute((*vecBackward)[idx]);//buffComplexBackward->buffPatch
	(*vecBuffPatch)[idx].convertTo((*vecBuffPatch)[idx], CV_32F, 1.0 / double((*vecBuffPatch)[idx].cols*(*vecBuffPatch)[idx].rows));
// 	int iHalfX = (DiffKernelSize.width >> 1);
// 	int iHalfY = (DiffKernelSize.height >> 1);
// 	cv::Rect SrcRoi1(iHalfX, iHalfY, PatchImg.cols/* - iHalfX*/, PatchImg.rows/* - iHalfY*/);
// 	cv::Rect DstRoi1(0, 0, PatchImg.cols/* - iHalfX*/, PatchImg.rows/* - iHalfY*/);
// 	(*vecBuffPatch)[idx](SrcRoi1).copyTo(DstImg(DstRoi1));

	(*vecBuffPatch)[idx].copyTo(DstImg);

	DstImg.convertTo(DstImg, CV_16S, 1.0 / (double)iDiffKernelNum);

	cv::Mat tempMat = (DstImg > iDarkThr & DstImg < iLightThr);
	DstImg.setTo(0x00, tempMat);
	cv::Mat maskDark = (DstImg < 0);
	cv::Mat maskLight = (DstImg > 0);
	cv::Mat darkMat, lightMat;
	DstImg.convertTo(darkMat, CV_16S, 1, -iDarkThr);
	DstImg.convertTo(lightMat, CV_16S, 1, -iLightThr);
	darkMat.copyTo(DstImg, maskDark);
	lightMat.copyTo(DstImg, maskLight);

	cv::Rect Roi(abs(rtSplit.x - rtTruth.x), abs(rtSplit.y - rtTruth.y), rtTruth.width, rtTruth.height);
	DstImg(Roi).copyTo((*diffImg)(rtTruth));

	

	return true;
}

bool CDOGCheck::check(cv::Mat& img, cv::Mat& diffImg, double* dTime)
{
	*dTime = cvGetTickCount();

	//diffImg = cv::Mat::zeros(getHandle()->ImageSizePre(), CV_16S);
	m_vecReturn.clear();
	if (m_iMarkDOG == 0)
	{
		for (int i = 0; i < m_vecSplit.size(); i++)
		{
			m_vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CDOGCheck::DOGCheck4GaussianThread, this, &img, m_vecSplit[i], m_vecTruth[i], m_vecKernelSize, m_vecSigma, &diffImg, m_iThresholdDark, m_iThresholdLight)));
			//m_vecReturn.push_back(executor.commit(CInspectionAlogrithm::DOGCheck4GaussianThread, SrcImg, m_vecSplit[i], m_vecTruthRect[i], m_vecKernelSize, m_vecSigma, DiffImg, i));
		}
		for (int i = 0; i < m_vecSplit.size(); i++)
		{
			bool hr = m_vecReturn[i].get();
			if (hr == false)
			{
				//抛出异常....
				printf("Gaussian DOG occurred some unhappy! Info: Rect: x = %d, y= %d, width = %d, height=%d\n", m_vecTruth[i].x, m_vecTruth[i].y, m_vecTruth[i].width, m_vecTruth[i].height);
				return false;
			}
		}
	}
	else
	{
		for (int i = 0; i < m_vecSplit.size(); i++)
		{
			//std::thread::id thr_id = std::this_thread::get_id();
			m_vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CDOGCheck::DOGCheck4FFTwThread3, this, &img, m_vecSplit[i], m_vecTruth[i], m_fftNormSize, m_iDiffKernelNum, &m_DogKernelFFtw, m_sumKernelSize,
				&m_vecFFTWForward, &m_vecFFTWBackward, &m_vecBuffPatch, &m_vecComplexImg, &m_vecComplexMul, &diffImg, &m_mapThreadIndex, m_iThresholdDark, m_iThresholdLight)));
		}
		for (int i = 0; i < m_vecSplit.size(); i++)
		{
			bool hr = m_vecReturn[i].get();
			if (hr == false)
			{
				//抛出异常....
				printf("FFTw DOG occurred some unhappy! Info: Rect: x = %d, y= %d, width = %d, height=%d\n", m_vecTruth[i].x, m_vecTruth[i].y, m_vecTruth[i].width, m_vecTruth[i].height);
				return false;
			}
		}
	}
	m_vecReturn.clear();
	
	//This code will run in multi-threads

	
	/*cv::Mat tempMat = (diffImg > m_iThresholdDark & diffImg < m_iThresholdLight);
	diffImg.setTo(0x00, tempMat);
	cv::Mat maskDark = (diffImg < 0);
	cv::Mat maskLight = (diffImg > 0);
	cv::Mat darkMat, lightMat;
	diffImg.convertTo(darkMat, CV_16S, 1, -m_iThresholdDark);
	diffImg.convertTo(lightMat, CV_16S, 1, -m_iThresholdLight);
	darkMat.copyTo(diffImg, maskDark);
	lightMat.copyTo(diffImg, maskLight);*/

	*dTime = (cv::getTickCount() - (*dTime)) / (1000 * cvGetTickFrequency());
	return true;
}

bool CDOGCheck::check(cv::cuda::GpuMat& img, cv::cuda::GpuMat& diffImg, double* dTime)
{
	*dTime = cvGetTickCount();
	//diffImg = cv::cuda::GpuMat(getHandle()->ImageSizePre(), CV_16S);
	diffImg.setTo(0x00);

	double dt = 0, d1 = 0;
	for (int i = 0; i < m_vecSplit.size(); i++)
	{
		if (m_vecSplit[i].width > m_fftNormSize.width || m_vecSplit[i].height > m_fftNormSize.height)
		{
			return false;
		}
		//cv::Mat showMat;
		//
		cv::cuda::GpuMat PatchImg = (img)(m_vecSplit[i]);
		//PatchImg.download(showMat);
		m_BuffPatch_gpu.setTo(0x00);
		PatchImg.convertTo(m_BuffPatch_gpu(cv::Rect(0, 0, PatchImg.cols, PatchImg.rows)), CV_32F);
		//m_BuffPatch_gpu.download(showMat);
		bool hr = m_pConv->Execute((float*)m_BuffPatch_gpu.data, (float*)m_BuffPatch_gpu.data, &dt);
		if (hr==false)
		{
			return false;
		}
		//d1 += dt;

		DiffFilter(&m_BuffPatch_gpu, &m_DstImg,m_iThresholdDark, m_iThresholdLight);
		//m_BuffPatch_gpu.convertTo(m_DstImg, CV_16S);
		//m_DstImg.download(showMat);

		/*fftwf_execute(m_FFTWForward_gpu);//buffPatch->buffComplexImg
		m_ComplexImg_gpu.download(showMat);
		cv::cuda::GpuMat DstImg(PatchImg.rows, PatchImg.cols, CV_32F);
		DstImg.setTo(0x00);
		m_DogKernelFFtw_gpu.download(showMat);
		cv::cuda::mulSpectrums(m_DogKernelFFtw_gpu, m_ComplexImg_gpu, m_ComplexImg_gpu, cv::DFT_COMPLEX_OUTPUT);
		m_ComplexImg_gpu.download(showMat);
		fftwf_execute(m_FFTWBackward_gpu);//buffComplexBackward->buffPatch
		m_BuffPatch_gpu.download(showMat);
		m_BuffPatch_gpu.convertTo(m_BuffPatch_gpu, CV_32F, 1.0 / double(m_BuffPatch_gpu.cols*m_BuffPatch_gpu.rows));
		m_BuffPatch_gpu.download(showMat);
		m_BuffPatch_gpu.copyTo(DstImg);
		cv::cuda::GpuMat DstImg1;
		DstImg.convertTo(DstImg1, CV_16S, 1.0 / (double)m_iDiffKernelNum);
		DstImg1.download(showMat);*/

		/*cv::Mat tempMat = (DstImg > iDarkThr & DstImg < iLightThr);
		DstImg.setTo(0x00, tempMat);
		cv::Mat maskDark = (DstImg < 0);
		cv::Mat maskLight = (DstImg > 0);
		cv::Mat darkMat, lightMat;
		DstImg.convertTo(darkMat, CV_16S, 1, -iDarkThr);
		DstImg.convertTo(lightMat, CV_16S, 1, -iLightThr);
		darkMat.copyTo(DstImg, maskDark);
		lightMat.copyTo(DstImg, maskLight);*/

		/*cv::cuda::GpuMat maskDark, maskLight, maskTemp;
		cv::cuda::compare(m_DstImg, m_iThresholdDark, maskDark, cv::CMP_GT);
		cv::cuda::compare(m_DstImg, m_iThresholdLight, maskLight, cv::CMP_LT);
		cv::cuda::bitwise_and(maskDark, maskLight, maskTemp);
		m_DstImg.setTo(0x00, maskTemp);
		cv::cuda::compare(m_DstImg, 0, maskLight, cv::CMP_GT);
		cv::cuda::compare(m_DstImg, 0, maskDark, cv::CMP_LT);
		cv::cuda::GpuMat darkMat, lightMat;
		m_DstImg.convertTo(darkMat, CV_16S, 1, -m_iThresholdDark);
		m_DstImg.convertTo(lightMat, CV_16S, 1, -m_iThresholdLight);
		darkMat.copyTo(m_DstImg, maskDark);
		lightMat.copyTo(m_DstImg, maskLight);*/

		cv::Rect Roi(abs(m_vecSplit[i].x - m_vecTruth[i].x), abs(m_vecSplit[i].y - m_vecTruth[i].y), m_vecTruth[i].width, m_vecTruth[i].height);
		m_DstImg(Roi).copyTo(diffImg(m_vecTruth[i]));
	}
	//printf("Dog Time: %f\n", d1);
	*dTime = (cvGetTickCount() - (*dTime)) / (1000 * cvGetTickFrequency());
	return true;
}

void CDOGCheck::SetParam(void* param)
{

}

void CDOGCheck::KernelCopyToBuff(cv::Mat& Kernel, cv::Mat& buff)
{
	cv::Rect rtSrc, rtDst;
	rtSrc.x = Kernel.cols >> 1;
	rtSrc.y = Kernel.rows >> 1;
	rtSrc.width = Kernel.cols - rtSrc.x;
	rtSrc.height = Kernel.rows - rtSrc.y;
	rtDst.x = rtDst.y = 0;
	rtDst.width = rtSrc.width;
	rtDst.height = rtSrc.height;
	Kernel(rtSrc).convertTo(buff(rtDst), buff.type());

	rtSrc.x = 0;
	rtSrc.y = 0;
	rtSrc.width = Kernel.cols >> 1;
	rtSrc.height = Kernel.rows >> 1;
	rtDst.x = buff.cols - rtSrc.width;
	rtDst.y = buff.rows - rtSrc.height;
	rtDst.width = rtSrc.width;
	rtDst.height = rtSrc.height;
	Kernel(rtSrc).convertTo(buff(rtDst), buff.type());

	rtSrc.x = 0;
	rtSrc.y = Kernel.rows >> 1;
	rtSrc.width = Kernel.cols >> 1;
	rtSrc.height = Kernel.rows >> 1;
	rtDst.x = buff.cols - rtSrc.width;
	rtDst.y = 0;
	rtDst.width = rtSrc.width;
	rtDst.height = rtSrc.height;
	Kernel(rtSrc).convertTo(buff(rtDst), buff.type());

	rtSrc.x = Kernel.cols >> 1;
	rtSrc.y = 0;
	rtSrc.width = Kernel.cols >> 1;
	rtSrc.height = Kernel.rows >> 1;
	rtDst.x = 0;
	rtDst.y = buff.rows - rtSrc.height;
	rtDst.width = rtSrc.width;
	rtDst.height = rtSrc.height;
	Kernel(rtSrc).convertTo(buff(rtDst), buff.type());
}

bool CDOGCheck::GetFFTwParam4(cv::Size& imgSize, std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma, float* _DOGKernelFFT)
{
	/*if (vecKernelSize.size() != vecSigma.size() || vecKernelSize.size() == 0)
	{
		return false;
	}
	int iKernelNum = int(vecKernelSize.size() - 1);
	cv::Size RefKernelSize = vecKernelSize[0];
	float fRefSigma = vecSigma[0];

	std::vector<cv::Size> tempKernel;
	tempKernel.push_back(cv::Size(1, 1));
	std::vector<float> tempSigma;
	tempSigma.push_back(0);
	for (int i = 0; i < iKernelNum; i++)
	{
		tempKernel.push_back(vecKernelSize[i + 1]);
		tempSigma.push_back(vecSigma[i + 1]);
	}
	cv::Mat diffKernel = GetDiffKernel3(tempKernel, tempSigma, RefKernelSize, fRefSigma);
	tempKernel.clear();
	tempSigma.clear();

	diffKernelSize.width = diffKernel.cols;
	diffKernelSize.height = diffKernel.rows;
	int iFFTWRow = imgSize.height;
	int iFFTWCol = imgSize.width;
	cv::Mat buffPatch = cv::Mat::zeros(iFFTWRow, iFFTWCol, CV_32F);

	DOGKernelFFT = cv::Mat(iFFTWRow, (iFFTWCol >> 1) + 1, CV_32FC2);
	fftwf_plan forwardKernel;

	forwardKernel = fftwf_plan_dft_r2c_2d(buffPatch.rows, buffPatch.cols, (float*)buffPatch.data, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>((float*)DOGKernelFFT.data)), FFTW_MEASURE);
	buffPatch.setTo(0x00);
	//diffKernel.copyTo(buffPatch(cv::Rect(0, 0, diffKernel.cols, diffKernel.rows)));

	KernelCopyToBuff(diffKernel, buffPatch);

	fftwf_execute(forwardKernel);//buffPatch->out
	fftwf_destroy_plan(forwardKernel);*/

	return true;
}
/*
bool CDOGCheck::GetFFTwParam3_gpu(std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma, cv::Size& diffKernelSize)
{
	if (vecKernelSize.size() != vecSigma.size() || vecKernelSize.size() == 0)
	{
		return false;
	}
	int iKernelNum = int(vecKernelSize.size() - 1);
	cv::Size RefKernelSize = vecKernelSize[0];
	float fRefSigma = vecSigma[0];

	std::vector<cv::Size> tempKernel;
	tempKernel.push_back(cv::Size(1, 1));
	std::vector<float> tempSigma;
	tempSigma.push_back(0);
	for (int i = 0; i < iKernelNum; i++)
	{
		tempKernel.push_back(vecKernelSize[i + 1]);
		tempSigma.push_back(vecSigma[i + 1]);
	}
	cv::Mat diffKernel = GetDiffKernel3(tempKernel, tempSigma, RefKernelSize, fRefSigma);
	tempKernel.clear();
	tempSigma.clear();

	diffKernelSize.width = diffKernel.cols;
	diffKernelSize.height = diffKernel.rows;

	m_BuffPatch_gpu.setTo(0x00);
	cv::cuda::GpuMat dev_kernel;
	dev_kernel.upload(diffKernel);
	KernelCopyToBuff_gpu(dev_kernel, m_BuffPatch_gpu);
	fftwf_execute(m_FFTWForward_gpu);
	m_DogKernelFFtw_gpu = m_ComplexImg_gpu.clone();
 	cv::Mat showMat;
	m_DogKernelFFtw_gpu.download(showMat);
	//fftwf_destroy_plan(forwardKernel);

	return true;
}*/

void CDOGCheck::KernelCopyToBuff_gpu(cv::cuda::GpuMat& Kernel, cv::cuda::GpuMat& buff)
{
	cv::Rect rtSrc, rtDst;
	rtSrc.x = Kernel.cols >> 1;
	rtSrc.y = Kernel.rows >> 1;
	rtSrc.width = Kernel.cols - rtSrc.x;
	rtSrc.height = Kernel.rows - rtSrc.y;
	rtDst.x = rtDst.y = 0;
	rtDst.width = rtSrc.width;
	rtDst.height = rtSrc.height;
	Kernel(rtSrc).convertTo(buff(rtDst), buff.type());

	rtSrc.x = 0;
	rtSrc.y = 0;
	rtSrc.width = Kernel.cols >> 1;
	rtSrc.height = Kernel.rows >> 1;
	rtDst.x = buff.cols - rtSrc.width;
	rtDst.y = buff.rows - rtSrc.height;
	rtDst.width = rtSrc.width;
	rtDst.height = rtSrc.height;
	Kernel(rtSrc).convertTo(buff(rtDst), buff.type());

	rtSrc.x = 0;
	rtSrc.y = Kernel.rows >> 1;
	rtSrc.width = Kernel.cols >> 1;
	rtSrc.height = Kernel.rows >> 1;
	rtDst.x = buff.cols - rtSrc.width;
	rtDst.y = 0;
	rtDst.width = rtSrc.width;
	rtDst.height = rtSrc.height;
	Kernel(rtSrc).convertTo(buff(rtDst), buff.type());

	rtSrc.x = Kernel.cols >> 1;
	rtSrc.y = 0;
	rtSrc.width = Kernel.cols >> 1;
	rtSrc.height = Kernel.rows >> 1;
	rtDst.x = 0;
	rtDst.y = buff.rows - rtSrc.height;
	rtDst.width = rtSrc.width;
	rtDst.height = rtSrc.height;
	Kernel(rtSrc).convertTo(buff(rtDst), buff.type());
}

void CDOGCheck::GpuMatFFt(cv::cuda::GpuMat& src, cv::cuda::GpuMat& fftdst, fftwf_plan& plan, float* pSrc, float* pDst)
{
	cudaMemcpy2D((uchar*)pSrc, src.cols*sizeof(float)*src.channels(), src.data, src.step, src.cols*sizeof(float)*src.channels(), src.rows, cudaMemcpyDeviceToDevice);

	fftwf_execute(plan);

	cudaMemcpy2D(fftdst.data, fftdst.step, (uchar*)pDst, sizeof(float)*fftdst.cols * fftdst.channels(), sizeof(float)*fftdst.cols * fftdst.channels(), fftdst.rows, cudaMemcpyDeviceToDevice);
}

bool CDOGCheck::InitiaConvolution(std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma)
{
	if (vecKernelSize.size() != vecSigma.size() || vecKernelSize.size() == 0)
	{
		return false;
	}
	int iKernelNum = int(vecKernelSize.size() - 1);
	cv::Size RefKernelSize = vecKernelSize[0];
	float fRefSigma = vecSigma[0];

	std::vector<cv::Size> tempKernel;
	tempKernel.push_back(cv::Size(1, 1));
	std::vector<float> tempSigma;
	tempSigma.push_back(0);
	for (int i = 0; i < iKernelNum; i++)
	{
		tempKernel.push_back(vecKernelSize[i + 1]);
		tempSigma.push_back(vecSigma[i + 1]);
	}
	cv::Mat diffKernel = GetDiffKernel3(tempKernel, tempSigma, RefKernelSize, fRefSigma);
	tempKernel.clear();
	tempSigma.clear();

	cv::Mat tempMat;
	cv::flip(diffKernel, tempMat, -5);
	diffKernel = tempMat.clone();

	cudaMalloc((void**)(&m_pSrc), sizeof(float)*m_fftNormSize.width*m_fftNormSize.height);
	m_BuffPatch_gpu = cv::cuda::GpuMat(m_fftNormSize.height, m_fftNormSize.width, CV_32F, m_pSrc, sizeof(float)*m_fftNormSize.width);
	cudaMalloc((void**)(&m_pDst), sizeof(short)*m_fftNormSize.width*m_fftNormSize.height);
	m_DstImg = cv::cuda::GpuMat(m_fftNormSize.height, m_fftNormSize.width, CV_16S, m_pDst, sizeof(short)*m_fftNormSize.width);
	cudaMalloc((void**)(&m_pKernel), sizeof(float)*diffKernel.cols*diffKernel.rows);
	m_Kernel_gpu = cv::cuda::GpuMat(diffKernel.rows, diffKernel.cols, CV_32F, m_pKernel, sizeof(float)*diffKernel.cols);
	m_Kernel_gpu.upload(diffKernel);
	return true;
}
