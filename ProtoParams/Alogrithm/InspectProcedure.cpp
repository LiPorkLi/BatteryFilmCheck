#include "stdafx.h"
#include "InspectProcedure.h"
//#include <synchapi.h>
//#define _PRINTF_
//#define _TIME_
#define _TWO_FLAT_FIELD_
CInspectProcedure::CInspectProcedure()
{
	m_isInspect = false;
	m_isStop = new bool;
	m_isPush = new bool;
	*m_isStop = true;
	m_queue_grab = NULL;
	m_queue_result = NULL;
	m_vecTime.clear();
	//std::shared_ptr<GrabImgInfo> p(new GrabImgInfo);
	//std::shared_ptr<GrabImgInfo> p = std::make_shared<GrabImgInfo>();
	m_iImageCount = 0;
}


CInspectProcedure::~CInspectProcedure()
{
	StopInspectThread();

// 	m_ss.clear();m_ss.str("");
// 	m_ss << "Inspect thread is Released!\n";
// 	m_strLog = m_ss.str();
// 	m_run->PushLog(m_strLog);

//	cudaFreeHost(m_pHost8u);

	delete m_isStop;
	m_vecTime.clear();

	delete m_isPush;

	while (m_queue_mem.try_pop() != nullptr)
	{
#ifdef _TIME_
		m_ss.clear(); m_ss.str("");
		m_ss << "Memory is gone!\n";
		m_strLog = m_ss.str();
		m_run->PushLog(m_strLog);
#endif
	}
}
/***********************************************************************
step0: 将图像转换为单通道图像
step1：原图像从host端到device端
step2：1.当图像标志为_ignore_ || _rectify_时，对图像进行预处理，且不用对图像下部分为空的区域进行填充；2.正常情况下，对图像进行预处理，并填充下部分空区域
step3：预处理图像从device端到host端
step4：在图像中搜索产品边界
step4.1：零点校准
step5：产品边界mask图像从device端到host端
step6.0: 提取平场矫正参数
step6.1：测试平场矫正参数
step6.2: 平常矫正后的图像从device端到host端
step6：对图像进行平常矫正
step7：对图像产品边界向外扩展，主要是在做DOG时候会有边界效应，从而引起误报，所以先将边界往外扩展，然后做DOG，在DIFF上直接把扩展部分截断
step8：做DOG
step9：在diff上把扩展部分截断
step10：在diff上将step2中的对应的填充部分置零
step11：diff从device端到host端
step12：blob分析
step13：输出的预处理后图像，将step2中对应的填充部分置零
step14：缺陷分类
************************************************************************/
uchar CInspectProcedure::InspectImage(std::shared_ptr<GrabImgInfo> grabImg, std::shared_ptr<ImageInspectResult> buff)
{
	double t1 = cvGetTickCount();
	buff->idx = grabImg->idx;
	buff->isAlert = false;
	//inspectResult->m_vecDefectList.clear();
	/*buff->m_vecDefectList.clear();  
	buff->idx = grabImg->idx;
	buff->diffImage = cv::Mat::zeros(m_run->ImageSizePre(), CV_8U);
	buff->srcImage = cv::Mat::zeros(m_run->ImageSizePre(), CV_8U); */

	//printf(".......time: %f\n", (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency()));

	double time_pre, time_boudary, time_dog, time_blob, time_classifiy, time_flat, time_expandBoundary, time_all,t2,t3;
	bool hr = false;
	if (grabImg->srcimg.channels() != 1)
	{
		t2 = cvGetTickCount();
		if (grabImg->srcimg.channels() == 3)
		{
			cv::cvtColor(grabImg->srcimg, grabImg->srcimg, cv::COLOR_RGB2GRAY);
		}
		if (grabImg->srcimg.channels() == 4)
		{
			cv::cvtColor(grabImg->srcimg, grabImg->srcimg, cv::COLOR_RGBA2GRAY);
		}
		t3 = cvGetTickCount();
		m_ss.clear(); m_ss.str("");
		time_all = (t3 - t1) / (1000 * cvGetTickFrequency());
		t2 = (t3 - t2) / (1000 * cvGetTickFrequency());
		m_ss << "Step0: Img ID:" << grabImg->idx << " Convert channel OK! Time: " << t2 << " All Time: " << time_all << std::endl;
		m_strLog.clear();
		m_strLog = m_ss.str();
		//m_run->PushLog(m_strLog);
	}

	m_SrcImg = grabImg->srcimg;
	
	if (m_run->IsGpu())
	{
		t2 = cvGetTickCount();
		m_SrcImg_gpu.upload(m_SrcImg);//host->gpu
		t3 = cvGetTickCount();

		m_ss.clear(); m_ss.str("");
		time_all = (t3 - t1) / (1000 * cvGetTickFrequency());
		t2 = (t3 - t2) / (1000 * cvGetTickFrequency());
		m_ss << "Step1: Img ID:" << grabImg->idx << " src host->device. Time: " << t2 << " All Time: " << time_all << std::endl;
		m_strLog = m_ss.str();
		
#ifdef _TIME_
		m_run->PushLog(m_strLog);
#endif
	}
	if (grabImg->iMark == GrabImgInfo::_ignore_)
	{
		//preprocess
		if (m_run->IsGpu())
		{
			hr = m_preprocess->Preprocess(m_SrcImg_gpu, false, &time_pre);
		}
		
		if (hr == false)
		{
			m_ss.clear(); m_ss.str("");
			m_ss << "Step2: Img ID:" << grabImg->idx << " is not prodct!\n";
			m_strLog = m_ss.str();
			m_run->PushLog(m_strLog);
			//return 0;
		}

		m_ss.clear(); m_ss.str("");
		time_all = (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency());
		m_ss << "Step2: Img ID:" << grabImg->idx << " Preprocess OK! Time: " << time_pre << " All Time: " << time_all << std::endl;
		m_strLog.clear();
		m_strLog = m_ss.str();
		//m_run->PushLog(m_strLog);

		t2 = cvGetTickCount();
		m_SrcImg_gpu.download(buff->srcImage);
		t3 = cvGetTickCount();
		time_all = (t3- t1) / (1000 * cvGetTickFrequency());
		t2 = (t3 - t2) / (1000 * cvGetTickFrequency());
		m_ss.clear(); m_ss.str("");
		m_ss << "Ignored: Step3: Img ID:" << grabImg->idx << " pre device->host. Time: " << t2 << " All Time: " << time_all << std::endl;
		m_strLog = m_ss.str();
		//m_run->PushLog(m_strLog);

		return 0; 
	}


	//Rectify
	if (grabImg->iMark==GrabImgInfo::_rectify_) 
	{
		//preprocess
		if (m_run->IsGpu())
		{
			hr = m_preprocess->Preprocess(m_SrcImg_gpu, false, &time_pre);
		}

		if (hr == false)
		{
			m_ss.clear(); m_ss.str("");
			m_ss << "Step2: Img ID:" << grabImg->idx << " is not prodct!\n";
			m_strLog = m_ss.str();
			m_run->PushLog(m_strLog);
			//return 2;
		}

		m_ss.clear(); m_ss.str("");
		time_all = (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency());
		m_ss << "Step2: Img ID:" << grabImg->idx << " Preprocess OK! Time: " << time_pre << " All Time: " << time_all << std::endl;
		m_strLog.clear();
		m_strLog = m_ss.str();
		//m_run->PushLog(m_strLog);

		t2 = cvGetTickCount();
		m_SrcImg_gpu.download(buff->srcImage);
		t3 = cvGetTickCount();
		time_all = (t3 - t1) / (1000 * cvGetTickFrequency());
		t2 = (t3 - t2) / (1000 * cvGetTickFrequency());
		m_ss.clear(); m_ss.str("");
		m_ss << "Rectify: Step3: Img ID:" << grabImg->idx << " pre device->host. Time: " << t2 << " All Time: " << time_all << std::endl;
		m_strLog = m_ss.str();
		m_run->PushLog(m_strLog);
		return 0;
	}

	//std::vector<std::vector<cv::Point>> vecvecBlob;
	if (m_run->IsGpu())
	{
// 		m_SrcImg_gpu.upload(m_SrcImg);//host->gpu
// 
// 		m_ss.clear();m_ss.str("");
// 		m_ss << grabImg->idx << " image is from host to device, and time is " << (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency())<<std::endl;
// 		m_strLog = m_ss.str();
// 		m_run->PushLog(m_strLog);

		//preprocess
		hr = m_preprocess->Preprocess(m_SrcImg_gpu, true, &time_pre);
		if (hr == false)
		{
			m_ss.clear(); m_ss.str("");
			m_ss << "Step2: Img ID:" << grabImg->idx << " is not prodct!\n";
			m_strLog = m_ss.str();
			m_run->PushLog(m_strLog);
			return 0;
		}

		m_ss.clear(); m_ss.str("");
		time_all = (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency());
		m_ss << "Step2: Img ID:" << grabImg->idx << " Preprocess OK! Time: " << time_pre << " All Time: " << time_all << std::endl;
		m_strLog.clear();
		m_strLog = m_ss.str();
#ifdef _TIME_
		m_run->PushLog(m_strLog);
#endif

		t2 = cvGetTickCount();
		m_SrcImg_gpu.download(m_TempResullt.srcImage);
		std::future<void> copy1 = m_run->Executor()->commit(std::bind(&CInspectProcedure::CopyMat, this, &m_TempResullt.srcImage, &buff->srcImage, cv::Rect(0,0,m_TempResullt.srcImage.cols,m_TempResullt.srcImage.rows)));
		t3 = cvGetTickCount();

		time_all = (t3 - t1) / (1000 * cvGetTickFrequency());
		t2 = (t3 - t2) / (1000 * cvGetTickFrequency());
		m_ss.clear(); m_ss.str("");
		m_ss << "Step3: Img ID:" << grabImg->idx << " pre deviece->host + host->host. Time: " << t2 << " All Time: " << time_all << std::endl;
		m_strLog = m_ss.str();
#ifdef _TIME_
		m_run->PushLog(m_strLog);
#endif
		

#ifdef _PRINTF_
			char savepath[256];
			sprintf_s(savepath, "..//rlt//%d_mask.png", grabImg->idx);
			cv::imwrite(savepath, inspectResult->srcImage);
#endif // _PR

	
		//boudary search
		hr = m_boudarysearch->BoundarySearch(m_TempResullt.srcImage, m_FrdMask_gpu, &time_boudary);
		if (hr == false)
		{
			m_ss.clear(); m_ss.str("");
			m_ss << "Step4: Img ID:" << grabImg->idx << " Boundary Search error!\n";
			m_strLog = m_ss.str();
			m_run->PushLog(m_strLog);
			copy1.get();

			return 4;
		}
		
		m_ss.clear(); m_ss.str("");
		time_all = (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency());
		m_ss << "Step4: Img ID:" << grabImg->idx << " Boundary Search OK! Time: " << time_boudary << " All Time: " << time_all << std::endl;
		m_strLog.clear();
		m_strLog = m_ss.str();
#ifdef _TIME_
		m_run->PushLog(m_strLog);
#endif

		/*if (grabImg->iMark == GrabImgInfo::_mark_zero_)
		{
			int iLeft, iRight;
			hr = m_boudarysearch->GetBoundaryPix(&iLeft, &iRight);
			if (hr == false)
			{
				m_ss.clear(); m_ss.str("");
				m_ss << "Mark Zero: Step4.1: Img ID:" << grabImg->idx << " Mark Zero error!\n";
				m_strLog = m_ss.str();
				//m_run->PushLog(m_strLog);
				copy1.get();
				return 41;
			}
			m_run->SetBoundaryPix(iLeft, iRight);

			m_ss.clear(); m_ss.str("");
			m_ss << "Mark Zero: Step4.1: Img ID:" << grabImg->idx << " Mark Zero OK!\n";
			m_strLog = m_ss.str();
			m_run->PushLog(m_strLog);

			copy1.get();
			return 0;
		}*/

		t2 = cvGetTickCount();
		//m_FrdMask_gpu.upload(m_FrdMask);
		//cv::imwrite("D:\\temp.png", m_FrdMask);
		t3 = cvGetTickCount();
		
		time_all = (t3 - t1) / (1000 * cvGetTickFrequency());
		t2 = (t3 - t2) / (1000 * cvGetTickFrequency());
		m_ss.clear(); m_ss.str("");
		m_ss << "Step5: Img ID:" << grabImg->idx << " Frd mask device->host. Time: " << t2 << " All Time: " << time_all << std::endl;
		m_strLog = m_ss.str();
#ifdef _TIME_
		m_run->PushLog(m_strLog);
#endif

		//flat field
		if (1)// 双层
		{
			if (grabImg->iMark == GrabImgInfo::_flatfield_ || grabImg->iMark == GrabImgInfo::_flatfield2_)
			{
				copy1.get();
				return 0;
			}
			//cv::Mat tempMask;
			//m_FrdMask_gpu.download(tempMask);

			m_strLog.clear();
			m_strLog = std::string("Flat2....");
			m_run->PushLog(m_strLog);

			hr = m_flat->TuneImgSelf(m_SrcImg_gpu, m_FrdMask_gpu, &time_flat);
// 			cv::Mat tempMat;
// 			m_SrcImg_gpu.download(tempMat);
			//printf("++++++++++++%f\n", time_flat);
			if (hr == false)
			{
				//异常
				m_ss.clear(); m_ss.str("");
				m_ss << "Step6: Img ID:" << grabImg->idx << " Flatfield error!\n";
				m_strLog = m_ss.str();
				m_run->PushLog(m_strLog);
				copy1.get();
				return 6;
			}

			m_ss.clear(); m_ss.str("");
			time_all = (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency());
			m_ss << "Step6: Img ID:" << grabImg->idx << " Flatfield OK! Time: " << time_flat << " All Time: " << time_all << std::endl;
			m_strLog.clear();
			m_strLog = m_ss.str();
#ifdef _TIME_
			m_run->PushLog(m_strLog);
#endif

			/*int iLeft, iRight, iDark, iLight;
			m_blobanalysis->GetStripBoundary(iLeft, iRight);
			m_run->GetBlobThr(iDark, iLight);
			float fTime = CheckMeanFilter(m_SrcImg_gpu, m_FrdMask_gpu, m_DiffImg_gpu16S, m_run->OffsetHeightIndex(), iLeft, iRight, 257, iDark, iLight);

			cv::Mat tempMask;
			m_FrdMask_gpu.download(tempMask);

			cv::Mat tempMat;
			m_DiffImg_gpu16S.download(tempMat);
			cv::convertScaleAbs(tempMat, tempMat);*/
		}
		else
		{
			if (grabImg->iMark == GrabImgInfo::_flatfield_ || grabImg->iMark == GrabImgInfo::_flatfield2_)
			{
				bool hr1 = m_boudarysearch->ExpandBoundary(m_FrdMask_gpu, 255, &time_expandBoundary);
				hr = m_boudarysearch->ExpandBoundary(m_SrcImg_gpu, 255, &time_expandBoundary);

				// 			cv::Mat tempMat;
				// 			m_FrdMask_gpu.download(tempMat);
				// 			cv::imwrite("D:\\temp-m.png", tempMat);
				// 			m_SrcImg_gpu.download(tempMat);
				// 			cv::imwrite("D:\\temp-s.png", tempMat);

				if (hr & hr1 == false)
				{
					//异常
					m_ss.clear(); m_ss.str("");
					m_ss << "Step7: Img ID:" << grabImg->idx << " Expand Boundary error!\n";
					m_strLog = m_ss.str();
					m_run->PushLog(m_strLog);
					copy1.get();
					return 7;
				}
				m_ss.clear(); m_ss.str("");
				time_all = (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency());
				m_ss << "Step7: Img ID:" << grabImg->idx << " Expand Boundary OK! Time: " << time_expandBoundary << " All Time: " << time_all << std::endl;
				m_strLog.clear();
				m_strLog = m_ss.str();
				//m_run->PushLog(m_strLog);

				if (grabImg->iMark == GrabImgInfo::_flatfield_)
				{
					hr = m_flat->GetParam(m_SrcImg_gpu, m_FrdMask_gpu, &time_flat, false);
				}
				else
				{
					hr = m_flat->GetParam(m_SrcImg_gpu, m_FrdMask_gpu, &time_flat, true);
				}
				if (hr == false)
				{
					m_ss.clear(); m_ss.str("");
					m_ss << "Flatfield: Step6.0: Img ID:" << grabImg->idx << " Get Flat param error!\n";
					m_strLog = m_ss.str();
					m_run->PushLog(m_strLog);
					copy1.get();
					return 60;
				}
				m_ss.clear(); m_ss.str("");
				time_all = (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency());
				m_ss << "Flatfield: Step6.0: Img ID:" << grabImg->idx << " Get Flat param OK! Time: " << time_flat << " All Time: " << time_all << std::endl;
				m_strLog.clear();
				m_strLog = m_ss.str();
				//m_run->PushLog(m_strLog);

				hr = m_flat->TuneImg(m_SrcImg_gpu, m_SrcImg_gpu, m_FrdMask_gpu, &time_flat);
				if (hr == false)
				{
					m_ss.clear(); m_ss.str("");
					m_ss << "Flatfield: Step6.1: Img ID:" << grabImg->idx << " Test flatfield error!\n";
					m_strLog = m_ss.str();
					m_run->PushLog(m_strLog);
					copy1.get();
					return 61;
				}
				m_ss.clear(); m_ss.str("");
				time_all = (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency());
				m_ss << "Flatfield: Step6.1: Img ID:" << grabImg->idx << " Test flatfield OK! Time: " << time_flat << " All Time: " << time_all << std::endl;
				m_strLog.clear();
				m_strLog = m_ss.str();
				//m_run->PushLog(m_strLog);

				t2 = cvGetTickCount();
				m_SrcImg_gpu.download(buff->srcImage);
				//cv::imwrite("D:\\temp-f.png", buff->srcImage);
				t3 = cvGetTickCount();
				time_all = (t3 - t1) / (1000 * cvGetTickFrequency());
				t2 = (t3 - t2) / (1000 * cvGetTickFrequency());
				m_ss.clear(); m_ss.str("");
				m_ss << "Step6.2: Img ID:" << grabImg->idx << " flat img device->host. Time: " << t2 << " All Time: " << time_all << std::endl;
				m_strLog = m_ss.str();
				m_run->PushLog(m_strLog);

#ifdef _PRINTF_
				m_SrcImg_gpu.download(m_SrcImg);
				//char savepath[256];
				sprintf_s(savepath, "..//rlt//%d_preprocess.png", grabImg->idx);
				cv::imwrite(savepath, m_SrcImg);
#endif // _PR
				copy1.get();
				return 0;
			}
			else
			{
				m_strLog.clear();
				m_strLog = std::string("Flat1....");
				m_run->PushLog(m_strLog);
				//cv::Mat tempMat;
				//m_SrcImg_gpu.download(tempMat);
				hr = m_flat->TuneImg(m_SrcImg_gpu, m_SrcImg_gpu, m_FrdMask_gpu, &time_flat);			
				//m_SrcImg_gpu.download(tempMat);
				if (hr == false)
				{
					//异常
					m_ss.clear(); m_ss.str("");
					m_ss << "Step6: Img ID:" << grabImg->idx << " Flatfield error!\n";
					m_strLog = m_ss.str();
					m_run->PushLog(m_strLog);
					copy1.get();
					return 6;
				}

				m_ss.clear(); m_ss.str("");
				time_all = (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency());
				m_ss << "Step6: Img ID:" << grabImg->idx << " Flatfield OK! Time: " << time_flat << " All Time: " << time_all << std::endl;
				m_strLog.clear();
				m_strLog = m_ss.str();
#ifdef _TIME_
				m_run->PushLog(m_strLog);
#endif
			}
		}

																																	
		//DOG check
		if (m_check->IsDOG())
		{
			hr = m_boudarysearch->ExpandBoundary(m_SrcImg_gpu, 127, &time_expandBoundary);
			if (hr == false)
			{
				//异常
				m_ss.clear(); m_ss.str("");
				m_ss << "Step7: Img ID:" << grabImg->idx << " Expand Boundary error!\n";
				m_strLog = m_ss.str();
				m_run->PushLog(m_strLog);
				copy1.get();
				return 7;
			}
			m_ss.clear(); m_ss.str("");
			time_all = (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency());
			m_ss << "Step7: Img ID:" << grabImg->idx << " Expand Boundary OK! Time: " << time_expandBoundary << " All Time: " << time_all << std::endl;
			m_strLog.clear();
			m_strLog = m_ss.str();
#ifdef _TIME_
			m_run->PushLog(m_strLog);
#endif
		}

		//m_SrcImg_gpu.download(buff->srcImage);

		// 		cv::Mat tempMatd;
		// 		m_SrcImg_gpu.download(tempMatd);
		// 		cv::imwrite("D:\\temp-s.png", tempMatd);

		hr = m_check->Execute(m_SrcImg_gpu, m_DiffImg_gpu16S, &time_dog);

		if (hr == false)
		{
			m_ss.clear(); m_ss.str("");
			m_ss << "Step8: Img ID:" << grabImg->idx << " FFT error!\n";
			m_strLog = m_ss.str();
			m_run->PushLog(m_strLog);
			copy1.get();
			return 8;
		}
		//m_DiffImg_gpu16S.download(m_DiffImg16S);
		m_ss.clear(); m_ss.str("");
		time_all = (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency());
		m_ss << "Step8: Img ID:" << grabImg->idx << " FFT OK! Time: " << time_dog << " All Time: " << time_all << std::endl;
		m_strLog.clear();
		m_strLog = m_ss.str();
#ifdef _TIME_
		m_run->PushLog(m_strLog);
#endif


#ifdef _PRINTF_
		cv::cuda::abs(m_DiffImg_gpu16S, m_DiffImg_gpu16S);
		m_DiffImg_gpu16S.convertTo(m_DiffImg_gpu8U, CV_8U);
		m_DiffImg_gpu8U.download(m_DiffImg8U);
		//char savepath[256];
		sprintf_s(savepath, "..//rlt//%d_dog.png", grabImg->idx);
		cv::imwrite(savepath, m_DiffImg8U);
#endif // _PR

		hr = m_blobanalysis->SetNoCheckArea(m_DiffImg_gpu16S, m_FrdMask_gpu, &time_blob);

// 		cv::Mat tempMat8;
// 		m_FrdMask_gpu.download(tempMat8);

		int iOffsetLeft = 0, iOffsetRight = 0;
		m_blobanalysis->GetBlobBoundaryOffset(iOffsetLeft, iOffsetRight);

		// 		cv::Mat tempMat;
		// 		m_DiffImg_gpu16S.download(tempMat);
		// 		cv::convertScaleAbs(tempMat, tempMat);
		// 		cv::threshold(tempMat, tempMat, 1, 0xff, cv::THRESH_BINARY);
		// 		cv::imwrite("d:\\diff.png",tempMat);

		hr = m_boudarysearch->ErodeDiffImgBoundary(m_DiffImg_gpu16S, m_FrdMask_gpu, iOffsetLeft, iOffsetRight, &time_blob);

// 		cv::Mat tempMat;
// 		m_DiffImg_gpu16S.download(tempMat);
// 		cv::convertScaleAbs(tempMat, tempMat);
// 
// 		m_FrdMask_gpu.download(tempMat8);

		// 		m_DiffImg_gpu16S.download(tempMat);
		// 		cv::convertScaleAbs(tempMat, tempMat);
		// 		cv::threshold(tempMat, tempMat, 1, 0xff, cv::THRESH_BINARY);
		// 		cv::imwrite("d:\\diff-erod.png", tempMat);

		// 		cv::Mat tempMat;
		// 		m_FrdMask_gpu.download(tempMat);
		// 		cv::imwrite("D:\\temp-m.png", tempMat);

		if (hr == false)
		{
			m_ss.clear(); m_ss.str("");
			m_ss << "Step9: Img ID:" << grabImg->idx << " Erode Diff Boundary error!\n";
			m_strLog = m_ss.str();
			m_run->PushLog(m_strLog);
			copy1.get();
			return 9;
		}
		m_ss.clear(); m_ss.str("");
		time_all = (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency());
		m_ss << "Step9: Img ID:" << grabImg->idx << " Erode Diff Boundary OK! Time: " << time_blob << " All Time: " << time_all << std::endl;
		m_strLog.clear();
		m_strLog = m_ss.str();
#ifdef _TIME_
		m_run->PushLog(m_strLog);
#endif


		//t2 = cv::getTickCount();
		hr = m_preprocess->UnPadding(m_FrdMask_gpu, &time_pre);
		//m_FrdMask_gpu.download(tempMat8);

		if (hr == false)
		{
			m_ss.clear(); m_ss.str("");
			m_ss << "Step10: Img ID:" << grabImg->idx << " UpPadding error!\n";
			m_strLog = m_ss.str();
			m_run->PushLog(m_strLog);
			copy1.get();
			return 10;
		}
		m_ss.clear(); m_ss.str("");
		time_all = (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency());
		m_ss << "Step10: Img ID:" << grabImg->idx << " UpPadding OK! Time: " << time_pre << " All Time: " << time_all << std::endl;
		m_strLog.clear();
		m_strLog = m_ss.str();
#ifdef _TIME_
		m_run->PushLog(m_strLog);
#endif

		t2 = cv::getTickCount();
		//cudaMemcpy2D(m_pHost8u, m_run->ImageSizePre().width, m_FrdMask_gpu.data, m_FrdMask_gpu.step, m_FrdMask_gpu.cols, m_FrdMask_gpu.rows, cudaMemcpyDeviceToHost);
		m_FrdMask_gpu.download(m_TempResullt.diffImage);
		//m_TempResullt.diffImage.copyTo(buff->diffImage);
	

		std::vector<std::future<void>> vecReturn;
		for (int i = 0; i < m_run->DataPatchNum(); i++)
		{
			vecReturn.push_back(m_run->Executor()->commit(std::bind(&CInspectProcedure::CopyMat, this, &m_TempResullt.diffImage, &buff->diffImage, m_run->TruthRect(i))));
		}
		for (int i = 0; i < vecReturn.size(); i++)
		{
			vecReturn[i].get();
		}
		t3 = cvGetTickCount();
		//std::future<void> copy2 = m_run->Executor()->commit(std::bind(&CInspectProcedure::CopyMat, this, &m_TempResullt.diffImage, &buff->diffImage));
		//memcpy(buff->diffImage.data, m_pHost8u, sizeof(uchar)*buff->diffImage.cols*buff->diffImage.rows);
		time_all = (t3 - t1) / (1000 * cvGetTickFrequency());
		t2 = (t3 - t2) / (1000 * cvGetTickFrequency());
		m_ss.clear(); m_ss.str("");
		m_ss << "Step11: Img ID:" << grabImg->idx << " diff img deviece->host + host->host. Time: " << t2 << " All Time: " << time_all << std::endl;
		m_strLog = m_ss.str();
#ifdef _TIME_
		m_run->PushLog(m_strLog);
#endif

#ifdef _PRINTF_
		sprintf_s(savepath, "..//rlt//%d_diff.png", grabImg->idx);
		cv::imwrite(savepath, inspectResult->diffImage);
#endif // _PR
	
		//inspectResult->diffImage.copyTo(m_DiffImg8U);
		//hr = m_blobanalysis->BlobAnalysis(m_DiffImg8U, vecvecBlob, &time_blob);
		hr = m_blobanalysis->BlobAnalysis(m_TempResullt.diffImage, m_DiffImg_gpu16S, buff->m_vecDefectList, &time_blob);
		if (hr == false)
		{
			m_ss.clear(); m_ss.str("");
			m_ss << "Step12: Img ID:" << grabImg->idx << " Blob analysis error!\n";
			m_strLog = m_ss.str();
			m_run->PushLog(m_strLog);
			copy1.get();
			return 12;
		}
		m_ss.clear(); m_ss.str("");
		time_all = (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency());
		m_ss << "Step12: Img ID:" << grabImg->idx << " Blob analysis OK! Time: " << time_blob << " All Time: " << time_all << std::endl;
		m_strLog.clear();
		m_strLog = m_ss.str();
#ifdef _TIME_
		m_run->PushLog(m_strLog);
#endif

		copy1.get();
		hr = m_preprocess->UnPadding(buff->srcImage, &time_pre);
		if (hr == false)
		{
			m_ss.clear(); m_ss.str("");
			m_ss << "Step13: Img ID:" << grabImg->idx << " Diff UnPadding error!\n";
			m_strLog = m_ss.str();
			m_run->PushLog(m_strLog);
			return 13;
		}
		m_ss.clear(); m_ss.str("");
		time_all = (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency());
		m_ss << "Step13: Img ID:" << grabImg->idx << " Diff UnPadding OK! Time: " << time_pre << " All Time: " << time_all << std::endl;
		m_strLog.clear();
		m_strLog = m_ss.str();
#ifdef _TIME_
		m_run->PushLog(m_strLog);
#endif

		//hr = m_defectclassification->Execute(buff->m_vecDefectList, &time_classifiy);
		hr = m_cls->Clssify(buff, &time_classifiy);
		buff->isAlert = hr;

		/*if (hr == false)
		{
			m_ss.clear(); m_ss.str("");
			m_ss << "Step14: Img ID:" << grabImg->idx << " Defect classify error!\n";
			m_strLog = m_ss.str();
			m_run->PushLog(m_strLog);
			return 14;
		}*/
		m_ss.clear(); m_ss.str("");
		time_all = (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency());
		float dLen = m_run->GetPhysicLength();
		dLen = dLen / 1000.0;
		m_ss << /*"Step14:*/ "Img ID:" << grabImg->idx << " Defect classify  OK! Time: " << time_classifiy << " All Time: " << time_all << " and defect num is: " << buff->m_vecDefectList.size() << " ,Length is " << dLen << " m" <<std::endl;
		m_strLog.clear();
		m_strLog = m_ss.str();
		m_run->PushLog(m_strLog);
	}
	else
	{
		//preprocess
	/*	hr = m_preprocess->Preprocess(m_SrcImg, &time_pre);
		if (hr == false)
		{
			printf("%d img, preprocess is failed!\n", grabImg->idx);
			return false;
		}
		printf("%d img Pre Process time = %f	%f\n", grabImg->idx, (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency()), time_pre);
		m_SrcImg.copyTo(inspectResult->srcImage);
		printf("%d src img copy time: %f\n", grabImg->idx, (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency()));
		//boudary search
		hr = m_boudarysearch->BoundarySearch(m_SrcImg, m_FrdMask, &time_boudary);

#ifdef _PRINTF_
		char savepath[256];
		sprintf_s(savepath, "..//rlt//%d_mask_cpu.png", grabImg->idx);
		cv::imwrite(savepath, m_FrdMask);
#endif // _PR
		if (hr == false)
		{
			printf("%d img, boundary search is failed!\n", grabImg->idx);
			return false;
		}
		printf("%d img Boundary Search time = %f	%f\n", grabImg->idx, (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency()), time_boudary);
		//flat field
		if (grabImg->iMark == GrabImgInfo::_flatfield_)
		{
			hr = m_flat->GetParam(m_SrcImg, m_FrdMask, &time_flat, false);
			if (hr == false)
			{
				printf("%d img, get flatfield param is failed!\n", grabImg->idx);
				return false;
			}
			hr = m_flat->TuneImg(m_SrcImg, m_SrcImg, m_FrdMask, &time_flat);
			if (hr == false)
			{
				printf("%d img, tune image is failed!\n", grabImg->idx);
				return false;
			}

			printf("%d img flat param and tune img time: %f	%f\n", grabImg->idx, (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency()), time_flat);

			printf("%d gpu->host time ans all time = %f\n", grabImg->idx, (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency()));

#ifdef _PRINTF_
			sprintf_s(savepath, "..//rlt//%d_preprocess_cpu.png", grabImg->idx);
			cv::imwrite(savepath, m_SrcImg);
#endif // _PR

			return true;
		}
		else
		{

			hr = m_flat->TuneImg(m_SrcImg, m_SrcImg, m_FrdMask, &time_flat);
			if (hr == false)
			{
				//异常
				printf("%d img, tune image is failed!\n", grabImg->idx);
				return false;
			}
		}
		printf("%d img Flat field time = %f	%f\n", grabImg->idx, (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency()), time_flat);

#ifdef _PRINTF_
		sprintf_s(savepath, "..//rlt//%d_preprocess_cpu.png", grabImg->idx);
		cv::imwrite(savepath, m_SrcImg);
#endif // _PR

		//DOG check
		if (m_check->IsDOG())
		{
			hr = m_boudarysearch->ExpandBoundary(m_SrcImg, 127, &time_expandBoundary);
			if (hr == false)
			{
				//异常
				printf("%d img, expand boundary is failed!\n", grabImg->idx);
				return false;
			}
			printf("%d img Expand boundary time = %f	%f\n", grabImg->idx, (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency()), time_expandBoundary);
		}
		hr = m_check->Execute(m_SrcImg, m_DiffImg16S, &time_dog);
		if (hr == false)
		{
			printf("%d img, DOG check is error!\n", grabImg->idx);
			return false;
		}

		printf("%d img check time = %f	%f\n", grabImg->idx, (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency()), time_dog);

#ifdef _PRINTF_
		m_DiffImg16S = cv::abs(m_DiffImg16S);
		m_DiffImg16S.convertTo(m_DiffImg8U, CV_8U);
		//char savepath[256];
		sprintf_s(savepath, "..//rlt//%d_dog_cpu.png", grabImg->idx);
		cv::imwrite(savepath, m_DiffImg8U);
#endif // _PR


		hr = m_boudarysearch->ErodeDiffImgBoundary(m_DiffImg16S, m_FrdMask, m_blobanalysis->GetBlobBoundaryOffset(), &time_blob);
		if (hr == false)
		{
			printf("%d img, Boundary erode is error!\n", grabImg->idx);
			return false;
		}

		printf("%d diff img gpu->host time = %f	%f\n", grabImg->idx, (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency()));

		double t2 = cv::getTickCount();
		m_FrdMask.copyTo(inspectResult->diffImage);
		printf("%d mask img copy time = %f	%f\n", grabImg->idx, (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency()), (cvGetTickCount() - t2) / (1000 * cvGetTickFrequency()));

#ifdef _PRINTF_
		sprintf_s(savepath, "..//rlt//%d_diff_cpu.png", grabImg->idx);
		cv::imwrite(savepath, inspectResult->diffImage);
#endif // _PR
		
		hr = m_blobanalysis->BlobAnalysis(inspectResult->diffImage, m_DiffImg16S, inspectResult->m_vecDefectList, &time_blob);
		if (hr == false)
		{
			printf("%d img, blob analysis is error!\n", grabImg->idx);
			return false;
		}
		printf("%d img Blob analysis time = %f  %f	blob size= %d\n", grabImg->idx, (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency()), time_blob, inspectResult->m_vecDefectList.size());

		hr = m_defectclassification->Execute(inspectResult->m_vecDefectList, &time_classifiy);
		if (hr == false)
		{
			printf("%d img, defect classifiy is error!\n", grabImg->idx);
			return false;
		}
		printf("%d img classifiy time = %f %d\n", grabImg->idx, time_classifiy, inspectResult->m_vecDefectList.size());

		printf("%d gpu->all time = %f\n", grabImg->idx, (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency()));
*/
	}
	
	//defect classifiy
	/*hr = m_defectclassification->Execute(inspectResult->srcImage, inspectResult->diffImage, vecvecBlob, inspectResult->m_vecDefectList, &time_classifiy);
	if (hr == false)
	{
		//异常
	}
	printf("%d img classifiy time = %f %d\n", grabImg->idx, time_classifiy, inspectResult->m_vecDefectList.size());*/
// 	for (auto&& pt : vecvecBlob)
// 	{
// 		pt.clear();
// 	}
// 	vecvecBlob.clear();


	return 0;
}



void CInspectProcedure::StartInspectThread(bool isUpdate, float fStartLength /*= -1*/)
{
	StopInspectThread();
	m_isInspect = true;
	if (isUpdate)
	{
		if (fStartLength!=-1)//若为-1，采用工单设置的起始长度
		{
			m_run->SetStartLength(fStartLength);
		}
		m_iImageCount = 0;
	}
	m_inspectThread = std::thread(std::bind(&CInspectProcedure::Pipline, this));
	if (m_inspectThread.joinable())
	{
		*m_isStop = false;
		m_inspectThread.detach();
	}

	*m_isPush = false;
	m_applyMemoryThread = std::thread(std::bind(&CInspectProcedure::ApplyMemory, this));
	if (m_applyMemoryThread.joinable())
	{
		m_applyMemoryThread.detach();
	}
}

void CInspectProcedure::Pipline()
{
	while (1)
	{
		std::shared_ptr<GrabImgInfo> g = m_queue_grab->wait_and_pop();
		int iIgnoreIndex = 1;
		if (g!=nullptr)
		{
			if (g->idx < iIgnoreIndex+1)
			{
				g->iMark = GrabImgInfo::_ignore_;
			}
			else if (g->idx == iIgnoreIndex+1)
			{
				g->iMark = GrabImgInfo::_flatfield_;
			}
#ifdef _TWO_FLAT_FIELD_
			else if (g->idx == iIgnoreIndex + 2)
			{
				g->iMark = GrabImgInfo::_flatfield2_;
			}
#endif
			else
			{
				g->iMark = GrabImgInfo::_normal_;
			}


			if (g->srcimg.size()!=m_run->ImageSizeSrc())
			{
				//异常
				m_ss.clear(); m_ss.str("");
				m_ss << "Image " << g->idx << " is NULL!\n";
				m_strLog = m_ss.str();
				m_run->PushLog(m_strLog);

				*m_isStop = true;
				break;
			}
			
			m_vecTime.clear();

			//ImageInspectResult rltTemp;
			std::shared_ptr<ImageInspectResult> rltTemp = m_queue_mem.wait_and_pop();
			*m_isPush = true;
			uchar uStatus = InspectImage(g, rltTemp);
			if (uStatus == 0)
			{
				//m_queue_result->push(std::move(rltTemp));
				m_queue_result->push(std::move(*rltTemp));
				m_iImageCount++;
				m_run->SetPhysicLength(m_iImageCount);
			}
			else if (uStatus == 4)
			{
				m_queue_result->push(std::move(*rltTemp));
				m_iImageCount++;
				m_run->SetPhysicLength(m_iImageCount);

				if (g->iMark == GrabImgInfo::_ignore_)
				{
					iIgnoreIndex = g->idx + 1;
				}
				else if (g->iMark == GrabImgInfo::_flatfield_)
				{
					iIgnoreIndex = g->idx;
				}
#ifdef _TWO_FLAT_FIELD_
				else if (g->iMark == GrabImgInfo::_flatfield2_)
				{
					iIgnoreIndex = g->idx - 1;
				}
#endif
				else
				{

				}

			}
			else
			{ 
				//异常
				m_ss.clear();m_ss.str("");
				m_ss << "Process image " << g->idx<<" is FILED, and thread is stoped!\n";
				m_strLog = m_ss.str();
				m_run->PushLog(m_strLog);

				*m_isStop = true;
				break;
			}
		}

		if (m_isInspect == false && (m_queue_grab->getPushCount() == m_queue_grab->getPopCount()))
		{
			m_ss.clear();m_ss.str("");
			m_ss << "Inspect therad is Stoped!\n";
			m_strLog = m_ss.str();
			m_run->PushLog(m_strLog);

			*m_isStop = true;
			break;
		}
	}
	
}

void CInspectProcedure::StopInspectThread()
{
	m_isInspect = false;
	//int i = 0;
	while (*m_isStop == false)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(10)); //休眠10毫秒  
	}
}

void CInspectProcedure::SetParam(MainParam::param* p, safe_queue<GrabImgInfo>* queue_grab, safe_queue<ImageInspectResult>* queue_result, safe_queue<std::string>* queue_log)
{
	StopInspectThread();

	m_queue_grab = queue_grab;
	m_queue_result = queue_result;
	
	m_run = std::make_shared<CRunTimeHandle>(p, queue_log);
	m_preprocess = std::make_shared<CPreprocess>(p, m_run);
	m_boudarysearch = std::make_shared<CBoudarySearch>(p, m_run);
	m_check = std::make_shared<CCheckProcess>(p, m_run);
	m_blobanalysis = std::make_shared<CBlobAnalysis>(p, m_run);
	//m_defectclassification = std::make_shared<CClassifyProcess>(p, m_run);
	m_cls = std::make_shared<CClassify>(p, m_run);
	m_flat = std::make_shared<CFlatField>(p, m_run);

	if (m_run->IsGpu()==true)
	{
		m_DiffImg_gpu16S = cv::cuda::GpuMat(m_run->ImageSizePre(),CV_16S);
		m_DiffImg_gpu8U = cv::cuda::GpuMat(m_run->ImageSizePre(), CV_8U);
		m_FrdMask_gpu = cv::cuda::GpuMat(m_run->ImageSizePre(), CV_8U);
		m_TempImg = cv::Mat(m_run->ImageSizePre(), CV_8U);
	}

	m_DiffImg16S = cv::Mat(m_run->ImageSizePre(), CV_16S);
	//m_DiffImg8U = cv::Mat(m_run->ImageSizePre(), CV_8U);
	m_FrdMask = cv::Mat(m_run->ImageSizePre(), CV_8U);

	m_TempResullt.diffImage = cv::Mat(m_run->ImageSizePre(), CV_8U);
	m_TempResullt.srcImage = cv::Mat(m_run->ImageSizePre(), CV_8U);

	//cudaMallocHost((void**)&m_pHost8u, sizeof(uchar)*m_run->ImageSizePre().width*m_run->ImageSizePre().height);
}

float CInspectProcedure::GetLength()
{
	return m_run->GetPhysicLength();
}

void CInspectProcedure::CopyMat(cv::Mat* src, cv::Mat* dst, cv::Rect rt)
{
	(*src)(rt).copyTo((*dst)(rt));

	/*double time;
	bool hr = m_preprocess->UnPadding(*dst, &time);
	if (hr == false)
	{
		m_ss.clear(); m_ss.str("");
		m_ss << "Step13:" << " Src UnPadding error!\n";
		m_strLog = m_ss.str();
		m_run->PushLog(m_strLog);
		return;
	}*/
}

void CInspectProcedure::ApplyMemory()
{
	
	while (m_queue_mem.try_pop()!=nullptr)
	{

	}

	for (int i = 0; i < 5; i++)
	{
		ImageInspectResult buff;
		buff.m_vecDefectList.clear();
		buff.idx = 0;
		buff.diffImage = cv::Mat::zeros(m_run->ImageSizePre(), CV_8U);
		buff.srcImage = cv::Mat::zeros(m_run->ImageSizePre(), CV_8U);
		m_queue_mem.push(std::move(buff));
	}

	while ((*m_isStop) != true)
	{
		if ((*m_isPush)==true)
		{
			ImageInspectResult buff;
			buff.m_vecDefectList.clear();
			buff.idx = 0;
			buff.diffImage = cv::Mat::zeros(m_run->ImageSizePre(), CV_8U);
			buff.srcImage = cv::Mat::zeros(m_run->ImageSizePre(), CV_8U);
			m_queue_mem.push(std::move(buff));
			(*m_isPush) = false;
		
#ifdef _TIME_
			m_ss.clear(); m_ss.str("");
			m_ss << ".....................................................Set a buff " << m_queue_mem.getPushCount() - m_queue_mem.getPopCount() << "\n";
			m_strLog = m_ss.str();
			m_run->PushLog(m_strLog);
#endif
		}
		std::this_thread::sleep_for(std::chrono::microseconds(500));
	}
#ifdef _TIME_
	m_ss.clear(); m_ss.str("");
	m_ss << "Apply memory is end!\n";
	m_strLog = m_ss.str();
	m_run->PushLog(m_strLog);
#endif
}


