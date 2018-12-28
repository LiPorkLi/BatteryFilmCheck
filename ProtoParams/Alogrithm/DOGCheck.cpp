#include "stdafx.h"
#include "DOGCheck.h"
#include "cudaFunction.h"
#include "core\cuda_stream_accessor.hpp"

CDOGCheck::CDOGCheck(MainParam::param* p, std::shared_ptr<CRunTimeHandle> pHandle) : CAlogrithmBase(p, pHandle)
{
	ParamHelper<Parameters::InspectParam> helper(getParam());
	Parameters::InspectParam inspectionParam = helper.getRef();
	
	cv::Size PaddingSize;
	cv::Size SplitSize = getHandle()->SplitCellSize();
	cv::Size ImgSize = getHandle()->ImageSizePre();
	
	//int iKernelNum = inspectionParam.dogparam().defectkernellist_size();

	int iRefH = std::max(25, inspectionParam.dogparam().refkernel().sizey() >> getHandle()->DownSampleFator());
	int iRefW = std::max(25, inspectionParam.dogparam().refkernel().sizex() >> getHandle()->DownSampleFator());
	m_fRefSigma = inspectionParam.dogparam().refkernel().sigma();
	m_RefKernel = GetGaussKernel(cv::Size(iRefW, iRefH), m_fRefSigma, m_fRefSigma);
	m_fFundSigma = 0;
	m_FundKernel = GetGaussKernel(cv::Size(3, 3), m_fFundSigma, m_fFundSigma);
	m_DiffKernel = GetDiffKernel3(m_RefKernel, m_FundKernel);

	

	m_iThresholdDark = inspectionParam.dogparam().thresholddark();
	m_iThresholdLight = inspectionParam.dogparam().thresholdlight();
	m_iThresholdDark = std::max(-30, m_iThresholdDark);
	m_iThresholdDark = std::min(m_iThresholdDark, -5);
	m_iThresholdLight = std::max(5, m_iThresholdLight);
	m_iThresholdLight = std::min(m_iThresholdLight, 30);

	//m_pSrc = m_pComplex = m_pDOGKerelComplex = NULL;
	m_pDst = m_pDst1 = m_pSrc = m_pSrc1 = m_pKernel = NULL;
	m_pConv = m_pConv1 = NULL;
	if (getHandle()->IsGpu()==false)
	{
		//测试用
		ParamHelper<Parameters::RunTimeParam> runHelper(getParam());
		Parameters::RunTimeParam runParam = runHelper.getRef();
		//m_iMarkDOG = runParam.markdog();
		if (1)// fft cpu
		{
			PaddingSize.width = std::min((m_RefKernel.cols >> 1), SplitSize.width >> 1);
			PaddingSize.height = std::min((m_RefKernel.rows >> 1), SplitSize.height >> 1);
			getHandle()->SplitImg(ImgSize, SplitSize, PaddingSize, m_vecSplit, m_vecTruth);

			//
			int iThreadNum = getHandle()->ThreadNum();
			m_fftNormSize = SplitSize + PaddingSize + PaddingSize - cv::Size(1, 1);
			//
			bool hr = GetFFTwParam3(m_fftNormSize, m_DiffKernel, m_DogKernelFFtw);
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
				m_mapThreadIndex[getHandle()->Executor()->GetThreadId(i)] = i;

				m_vecFFTWForward[i] = fftwf_plan_dft_r2c_2d(m_vecBuffPatch[i].rows, m_vecBuffPatch[i].cols, (float*)m_vecBuffPatch[i].data, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>((float*)m_vecComplexImg[i].data)), FFTW_MEASURE);
				m_vecFFTWBackward[i] = fftwf_plan_dft_c2r_2d(m_vecBuffPatch[i].rows, m_vecBuffPatch[i].cols, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>((float*)m_vecComplexMul[i].data)), (float*)m_vecBuffPatch[i].data, FFTW_MEASURE);
			}
			double t2 = cvGetTickCount();
			printf("fftw handle Initia is OK, time=%f\n", (t2 - t1) / (1000 * cvGetTickFrequency()));
		}
		else//gaussian
		{
			for (int i = 0; i < getHandle()->DataPatchNum(); i++)
			{
				m_vecSplit.push_back(getHandle()->SplitRect(i));
				m_vecTruth.push_back(getHandle()->TruthRect(i));
			}
		}
	}
	else
	{
		SplitSize.width = (getHandle()->ImageSizePre().width >> 2);
		SplitSize.height = (getHandle()->ImageSizePre().height >> 2);

		double t1 = cvGetTickCount();
		PaddingSize.width = std::min((m_RefKernel.cols >> 1), SplitSize.width >> 1);
		PaddingSize.height = std::min((m_RefKernel.rows >> 1), SplitSize.height >> 1);
		getHandle()->SplitImg(ImgSize, SplitSize, PaddingSize, m_vecSplit, m_vecTruth);

		m_fftNormSize = SplitSize + PaddingSize + PaddingSize - cv::Size(1, 1);
		//
	
// 		cudaStreamCreate(&dev_Stream0);
// 		cudaStreamCreate(&dev_Stream1);
		dev_Stream0 = cv::cuda::StreamAccessor::getStream(cvStream0); 
		dev_Stream1 = cv::cuda::StreamAccessor::getStream(cvStream1);

		m_pConv = new CConvolutionFFT2D;
		m_pConv1 = new CConvolutionFFT2D;
		InitiaConvolution();
		bool hr = m_pConv->Initia(m_BuffPatch_gpu.rows, m_BuffPatch_gpu.cols, m_BuffPatch_gpu.step, m_Kernel_gpu.rows, m_Kernel_gpu.cols, -1,-1,&dev_Stream0);
		//cufftSetStream
		if (hr == false)
		{
			return;
		}
		hr = m_pConv1->Initia(m_BuffPatch_gpu1.rows, m_BuffPatch_gpu1.cols, m_BuffPatch_gpu1.step, m_Kernel_gpu.rows, m_Kernel_gpu.cols,-1,-1,&dev_Stream1);
		//cufftSetStream
		if (hr == false)
		{
			return;
		}
		hr = m_pConv->SetKernel((float*)m_Kernel_gpu.data, m_Kernel_gpu.step);
		if (hr == false)
		{
			return;
		}
		hr = m_pConv1->SetKernel((float*)m_Kernel_gpu.data, m_Kernel_gpu.step);
		if (hr == false)
		{
			return;
		}
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
	if (m_pConv1)
	{
		delete m_pConv1;
	}
	m_pConv1 = NULL;

	m_vecBuffPatch.clear();
	m_vecComplexImg.clear();
	m_vecComplexMul.clear();
	m_vecFFTWForward.clear();
	m_vecFFTWBackward.clear();
	m_mapThreadIndex.clear();

	if (m_pSrc)
	{
		cudaFree(m_pSrc);
	}
	if (m_pKernel)
	{
		cudaFree(m_pKernel);
	}
	if (m_pDst)
	{
		cudaFree(m_pDst);
	}
	if (m_pSrc1)
	{
		cudaFree(m_pSrc1);
	}

	if (m_pDst1)
	{
		cudaFree(m_pDst1);
	}
// 	cudaStreamDestroy(dev_Stream0);
// 	cudaStreamDestroy(dev_Stream1);
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

cv::Mat CDOGCheck::GetDiffKernel3(cv::Mat& ref, cv::Mat& fund)
{
	cv::Mat tempMat;
	cv::Mat tempFund = ref.clone();
	tempFund.setTo(0x00);
	fund.copyTo(tempFund(cv::Rect((ref.cols >> 1) - (fund.cols >> 1), (ref.rows >> 1) - (fund.rows >> 1), fund.cols, fund.rows)));
	tempMat = tempFund - ref;
	return tempMat;
}

bool CDOGCheck::GetFFTwParam3(cv::Size& imgSize, cv::Mat& diffKernel, cv::Mat& DOGKernelFFT)
{
	if (imgSize.width < diffKernel.cols || imgSize.height<diffKernel.rows)
	{
		return false;
	}
	
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
	fftwf_destroy_plan(forwardKernel);

	return true;
}


cv::Mat CDOGCheck::GetGaussKernel(cv::Size& kernelSize, float fSigmaW /*= 0.0f*/, float fSigmaH /*= 0.0f*/)
{
	cv::Mat kernel_w = cv::getGaussianKernel(kernelSize.width, fSigmaW, CV_32F);
	cv::transpose(kernel_w, kernel_w);
	cv::Mat kernel_h = cv::getGaussianKernel(kernelSize.height, fSigmaH, CV_32F);
	cv::Mat Kernel = kernel_h * kernel_w;
	return Kernel.clone();
}


bool CDOGCheck::DOGCheck4GaussianThread(cv::Mat* img, cv::Rect& rtSplit, cv::Rect& rtTruth, cv::Size& refSize, float fRefSigma, cv::Size& fundSize, float fFundSigma, cv::Mat* diffImg, int iDarkThr, int iLightThr)
{
	if (refSize.width <= fundSize.width || refSize.height <= fundSize.height)
	{
		return false;
	}
	cv::Mat PatchImg = (*img)(rtSplit);
	cv::Mat diffRlt = cv::Mat::zeros(PatchImg.rows, PatchImg.cols, CV_16S);
	cv::Mat RefImg, SrcImg, rltImg;
	cv::GaussianBlur(PatchImg, RefImg, refSize, fRefSigma, fRefSigma);
	cv::GaussianBlur(PatchImg, SrcImg, fundSize, fFundSigma, fFundSigma);
	RefImg.convertTo(RefImg, CV_16S);
	SrcImg.convertTo(SrcImg, CV_16S);
	diffRlt = SrcImg - RefImg;

	cv::Mat tempMat = (diffRlt > iDarkThr & diffRlt < iLightThr);
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
	return true;
}


bool CDOGCheck::DOGCheck4FFTwThread3(cv::Mat* img, cv::Rect& rtSplit, cv::Rect& rtTruth, cv::Size& normSize,  cv::Mat* KernelFFT, cv::Size& DiffKernelSize, 
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

	DstImg.convertTo(DstImg, CV_16S);

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
			m_vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CDOGCheck::DOGCheck4GaussianThread, this, 
				&img, m_vecSplit[i], m_vecTruth[i], cv::Size(m_RefKernel.cols,m_RefKernel.rows), m_fRefSigma,cv::Size(m_FundKernel.cols,m_FundKernel.rows),m_fFundSigma, &diffImg, m_iThresholdDark, m_iThresholdLight)));
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
			m_vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CDOGCheck::DOGCheck4FFTwThread3, this, &img, m_vecSplit[i], m_vecTruth[i], m_fftNormSize, &m_DogKernelFFtw, cv::Size(m_DiffKernel.cols,m_DiffKernel.rows),
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

	int iStep = (m_vecSplit.size() + 1) / 2;
	for (int i = 0; i<iStep; i++)
	{
		int idx = i * 2;
		if (m_vecSplit[idx].width > m_fftNormSize.width || m_vecSplit[idx].height > m_fftNormSize.height)
		{
			return false;
		}

		m_BuffPatch_gpu.setTo(0x00);
		m_BuffPatch_gpu1.setTo(0x00);

		cv::cuda::GpuMat PatchImg0 = (img)(m_vecSplit[idx]);
		PatchImg0.convertTo(m_BuffPatch_gpu(cv::Rect(0, 0, PatchImg0.cols, PatchImg0.rows)), CV_32F, 1.0, cvStream0);

		cv::cuda::GpuMat PatchImg1;
		if (idx+1<m_vecSplit.size())
		{
			PatchImg1 = (img)(m_vecSplit[idx + 1]);
			PatchImg1.convertTo(m_BuffPatch_gpu1(cv::Rect(0, 0, PatchImg1.cols, PatchImg1.rows)), CV_32F, 1.0, cvStream1);
		}

		
		//m_BuffPatch_gpu.download(showMat);
		bool hr = m_pConv->Execute((float*)m_BuffPatch_gpu.data, (float*)m_BuffPatch_gpu.data, &dt);
		if (hr == false)
		{
			return false;
		}
		m_pConv->GetResult((float*)m_BuffPatch_gpu.data);
		if (idx + 1 < m_vecSplit.size())
		{
			bool hr = m_pConv1->Execute((float*)m_BuffPatch_gpu1.data, (float*)m_BuffPatch_gpu1.data, &dt);
			if (hr == false)
			{
				return false;
			}
			m_pConv1->GetResult((float*)m_BuffPatch_gpu1.data);
		}
		//d1 += dt;

		DiffFilter(&m_BuffPatch_gpu, &m_DstImg, m_iThresholdDark, m_iThresholdLight, &dev_Stream0);

		if (idx + 1 < m_vecSplit.size())
		{
			DiffFilter(&m_BuffPatch_gpu1, &m_DstImg1, m_iThresholdDark, m_iThresholdLight, &dev_Stream1);
		}

		cv::Rect Roi(abs(m_vecSplit[idx].x - m_vecTruth[idx].x), abs(m_vecSplit[idx].y - m_vecTruth[idx].y), m_vecTruth[idx].width, m_vecTruth[idx].height);
		m_DstImg(Roi).copyTo(diffImg(m_vecTruth[idx]), cvStream0);

		if (idx + 1 < m_vecSplit.size())
		{
			cv::Rect Roi1(abs(m_vecSplit[idx + 1].x - m_vecTruth[idx + 1].x), abs(m_vecSplit[idx + 1].y - m_vecTruth[idx + 1].y), m_vecTruth[idx + 1].width, m_vecTruth[idx + 1].height);
			m_DstImg1(Roi1).copyTo(diffImg(m_vecTruth[idx + 1]), cvStream1);
		}
	}
	cudaStreamSynchronize(dev_Stream0);
	cudaStreamSynchronize(dev_Stream1);

// 	for (int i = 0; i < m_vecSplit.size(); i++)
// 	{
// 		if (m_vecSplit[i].width > m_fftNormSize.width || m_vecSplit[i].height > m_fftNormSize.height)
// 		{
// 			return false;
// 		}
// 		//cv::Mat showMat;
// 		//
// 		cv::cuda::GpuMat PatchImg = (img)(m_vecSplit[i]);
// 		//PatchImg.download(showMat);
// 		m_BuffPatch_gpu.setTo(0x00);
// 		PatchImg.convertTo(m_BuffPatch_gpu(cv::Rect(0, 0, PatchImg.cols, PatchImg.rows)), CV_32F);
// 		//m_BuffPatch_gpu.download(showMat);
// 		bool hr = m_pConv->Execute((float*)m_BuffPatch_gpu.data, (float*)m_BuffPatch_gpu.data, &dt);
// 		if (hr==false)
// 		{
// 			return false;
// 		}
// 		//d1 += dt;
// 
// 		DiffFilter(&m_BuffPatch_gpu, &m_DstImg,m_iThresholdDark, m_iThresholdLight);
// 		//m_BuffPatch_gpu.convertTo(m_DstImg, CV_16S);
// 		//m_DstImg.download(showMat);
// 
// 		/*fftwf_execute(m_FFTWForward_gpu);//buffPatch->buffComplexImg
// 		m_ComplexImg_gpu.download(showMat);
// 		cv::cuda::GpuMat DstImg(PatchImg.rows, PatchImg.cols, CV_32F);
// 		DstImg.setTo(0x00);
// 		m_DogKernelFFtw_gpu.download(showMat);
// 		cv::cuda::mulSpectrums(m_DogKernelFFtw_gpu, m_ComplexImg_gpu, m_ComplexImg_gpu, cv::DFT_COMPLEX_OUTPUT);
// 		m_ComplexImg_gpu.download(showMat);
// 		fftwf_execute(m_FFTWBackward_gpu);//buffComplexBackward->buffPatch
// 		m_BuffPatch_gpu.download(showMat);
// 		m_BuffPatch_gpu.convertTo(m_BuffPatch_gpu, CV_32F, 1.0 / double(m_BuffPatch_gpu.cols*m_BuffPatch_gpu.rows));
// 		m_BuffPatch_gpu.download(showMat);
// 		m_BuffPatch_gpu.copyTo(DstImg);
// 		cv::cuda::GpuMat DstImg1;
// 		DstImg.convertTo(DstImg1, CV_16S, 1.0 / (double)m_iDiffKernelNum);
// 		DstImg1.download(showMat);*/
// 
// 		/*cv::Mat tempMat = (DstImg > iDarkThr & DstImg < iLightThr);
// 		DstImg.setTo(0x00, tempMat);
// 		cv::Mat maskDark = (DstImg < 0);
// 		cv::Mat maskLight = (DstImg > 0);
// 		cv::Mat darkMat, lightMat;
// 		DstImg.convertTo(darkMat, CV_16S, 1, -iDarkThr);
// 		DstImg.convertTo(lightMat, CV_16S, 1, -iLightThr);
// 		darkMat.copyTo(DstImg, maskDark);
// 		lightMat.copyTo(DstImg, maskLight);*/
// 
// 		/*cv::cuda::GpuMat maskDark, maskLight, maskTemp;
// 		cv::cuda::compare(m_DstImg, m_iThresholdDark, maskDark, cv::CMP_GT);
// 		cv::cuda::compare(m_DstImg, m_iThresholdLight, maskLight, cv::CMP_LT);
// 		cv::cuda::bitwise_and(maskDark, maskLight, maskTemp);
// 		m_DstImg.setTo(0x00, maskTemp);
// 		cv::cuda::compare(m_DstImg, 0, maskLight, cv::CMP_GT);
// 		cv::cuda::compare(m_DstImg, 0, maskDark, cv::CMP_LT);
// 		cv::cuda::GpuMat darkMat, lightMat;
// 		m_DstImg.convertTo(darkMat, CV_16S, 1, -m_iThresholdDark);
// 		m_DstImg.convertTo(lightMat, CV_16S, 1, -m_iThresholdLight);
// 		darkMat.copyTo(m_DstImg, maskDark);
// 		lightMat.copyTo(m_DstImg, maskLight);*/
// 
// 		cv::Rect Roi(abs(m_vecSplit[i].x - m_vecTruth[i].x), abs(m_vecSplit[i].y - m_vecTruth[i].y), m_vecTruth[i].width, m_vecTruth[i].height);
// 		m_DstImg(Roi).copyTo(diffImg(m_vecTruth[i]));
// 	}
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

bool CDOGCheck::InitiaConvolution()
{
	cv::Mat tempMat;
	cv::flip(m_DiffKernel, tempMat, -5);
	m_DiffKernel = tempMat.clone();

	cudaMalloc((void**)(&m_pSrc), sizeof(float)*m_fftNormSize.width*m_fftNormSize.height);
	m_BuffPatch_gpu = cv::cuda::GpuMat(m_fftNormSize.height, m_fftNormSize.width, CV_32F, m_pSrc, sizeof(float)*m_fftNormSize.width);
	cudaMalloc((void**)(&m_pDst), sizeof(short)*m_fftNormSize.width*m_fftNormSize.height);
	m_DstImg = cv::cuda::GpuMat(m_fftNormSize.height, m_fftNormSize.width, CV_16S, m_pDst, sizeof(short)*m_fftNormSize.width);
	//
	cudaMalloc((void**)(&m_pSrc1), sizeof(float)*m_fftNormSize.width*m_fftNormSize.height);
	m_BuffPatch_gpu1 = cv::cuda::GpuMat(m_fftNormSize.height, m_fftNormSize.width, CV_32F, m_pSrc1, sizeof(float)*m_fftNormSize.width);
	cudaMalloc((void**)(&m_pDst1), sizeof(short)*m_fftNormSize.width*m_fftNormSize.height);
	m_DstImg1 = cv::cuda::GpuMat(m_fftNormSize.height, m_fftNormSize.width, CV_16S, m_pDst1, sizeof(short)*m_fftNormSize.width);
	//
	cudaMalloc((void**)(&m_pKernel), sizeof(float)*m_DiffKernel.cols*m_DiffKernel.rows);
	m_Kernel_gpu = cv::cuda::GpuMat(m_DiffKernel.rows, m_DiffKernel.cols, CV_32F, m_pKernel, sizeof(float)*m_DiffKernel.cols);
	m_Kernel_gpu.upload(m_DiffKernel);
	return true;
}
