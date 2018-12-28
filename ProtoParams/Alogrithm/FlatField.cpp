#include "stdafx.h"
#include "FlatField.h"


CFlatField::CFlatField(MainParam::param* p, std::shared_ptr<CRunTimeHandle> pHandle /*= NULL*/) : CAlogrithmBase(p, pHandle)
{
	ParamHelper<Parameters::InspectParam> help_InspectParam(getParam());
	Parameters::InspectParam inspectParam = help_InspectParam.getRef();
	m_iDstFrd = inspectParam.dstvaluefrd();
	m_iDstBgd = inspectParam.dstvaluebgd();
	m_iDstFrd = std::max(90, m_iDstFrd);
	m_iDstFrd = std::min(180, m_iDstFrd);
	m_iDstBgd = std::max(180, m_iDstBgd);
	m_iDstBgd = std::min(240, m_iDstBgd);

	if (getHandle()->IsGpu() == true)
	{
		m_blur = cv::cuda::createBoxFilter(CV_8U, CV_8U, cv::Size(5, 1));
	}

	m_iOffsetHeightIndex = pHandle->OffsetHeightIndex();
}


CFlatField::~CFlatField()
{
	for (int i = 0; i < m_vecReturn.size(); i++)
	{
		m_vecReturn[i].get();
	}
	m_vecReturn.clear();
}

bool CFlatField::GetParam(cv::Mat& imgFream, cv::Mat& frdMask, double* dTime, bool bIncrementTrain /*= true*/)
{
	*dTime = cvGetTickCount();
	if (bIncrementTrain==false)
	{
		m_iFreamIndex = 1;
	}
	bool hr = ImageAdd(imgFream, m_bgdSum_Num, m_frdSum_Num, frdMask, m_iFreamIndex++);
	if (hr == false)
	{
		return false;
	}
	GetFlatFieldParam(m_bgdSum_Num, m_frdSum_Num, m_bgdParam, m_frdParam, m_iDstBgd, m_iDstFrd);
	*dTime = (cvGetTickCount() - *dTime) / (1000 * cvGetTickFrequency());
	return true;
}

bool CFlatField::GetParam(cv::cuda::GpuMat& imgFream, cv::cuda::GpuMat& frdMask, double* dTime, bool bIncrementTrain /*= true*/)
{
	*dTime = cvGetTickCount();
	if (bIncrementTrain == false)
	{
		m_iFreamIndex = 1;
	}
	//bool hr = ImageAdd(imgFream, m_bgdSum_Num, m_frdSum_Num, frdMask, m_iFreamIndex++);
	bool hr = ImageAdd_gpu(imgFream, m_bgdSum_Num_gpu, m_frdSum_Num_gpu, frdMask, m_iFreamIndex++);
	if (hr == false)
	{
		return false;
	}
	GetFlatFieldParam_gpu(m_bgdSum_Num_gpu, m_frdSum_Num_gpu, m_bgdParam_gpu, m_frdParam_gpu, m_iDstBgd, m_iDstFrd);
	*dTime = (cvGetTickCount() - *dTime) / (1000 * cvGetTickFrequency());
	return true;
}

bool CFlatField::TuneImg(cv::Mat& imgFream, cv::Mat& datImg, cv::Mat& frdMask, double* dTime)
{
	*dTime = cvGetTickCount();
	if (getHandle()==NULL)
	{
		bool hr = FlatField(&imgFream, &datImg, &m_bgdParam, &m_frdParam, &frdMask, cv::Rect(0,0,imgFream.cols,imgFream.rows));
		return hr;
	}
	else
	{
		m_vecReturn.clear();
		for (int i = 0; i < getHandle()->DataPatchNum(); i++)
		{
			//vecReturn.push_back(m_pHandle->m_executor.commit(this->FlatField, &imgFream, &datImg, &m_bgdParam, &m_frdParam, &frdMask, &m_pHandle->m_vecTruth[i]));
			//vecReturn.push_back();
		
			m_vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CFlatField::FlatField, this, &imgFream, &datImg, &m_bgdParam, &m_frdParam, &frdMask, getHandle()->TruthRect(i))));
		}
		for (int i = 0; i < m_vecReturn.size(); i++)
		{
			bool hr = m_vecReturn[i].get();
			if (hr==false)
			{
				printf("Flat field occurred some unhappy! Info: Rect: x = %d, y= %d, width = %d, height=%d\n", getHandle()->TruthRect(i).x, getHandle()->TruthRect(i).y, getHandle()->TruthRect(i).width, getHandle()->TruthRect(i).height);
				return false;
			}
		}
		m_vecReturn.clear();
		//datImg.setTo(0xff, ~frdMask);//open the line will increase 100 ms time;
	}
	*dTime = (cv::getTickCount() - (*dTime)) / (1000 * cvGetTickFrequency());
	return true;
}

bool CFlatField::TuneImg(cv::cuda::GpuMat& imgFream, cv::cuda::GpuMat& datImg, cv::cuda::GpuMat& frdMask, double* dTime)
{

	FlatField_gpu(&imgFream, &datImg, &m_frdParam_gpu, &frdMask, cv::Rect(0, 0, imgFream.cols, imgFream.rows), dTime, m_iDstBgd);
	//*dTime = (cv::getTickCount() - (*dTime)) / (1000 * cvGetTickFrequency());
	return true;
}


bool CFlatField::ImageAdd(cv::Mat& imgFream, cv::Mat& avgBgd, cv::Mat& avgFrd, cv::Mat& frdMask, int iFreadIdx/* = 0*/)
{
	if (frdMask.size != imgFream.size)
	{
		return false;
	}

	cv::Mat* imgFrd = new cv::Mat(imgFream.rows, imgFream.cols, CV_8U);
	imgFrd->setTo(0x00);
	cv::Mat* imgBgd = new cv::Mat(imgFream.rows, imgFream.cols, CV_8U);
	imgBgd->setTo(0x00);
	cv::Mat bgdMask = ~frdMask;
	//cv::Mat frdMask1 = frdMask.clone();
	imgFream.copyTo(*imgFrd, frdMask);
	imgFream.copyTo(*imgBgd, bgdMask);
	cv::Mat tempAvgFrd, tempAvgBgd, lineFrdMask, lineBgdMask;

	//GetAvgMask(imgFrd, &frdMask1, &tempAvgFrd, &lineFrdMask);
	//GetAvgMask(imgBgd, &bgdMask, &tempAvgBgd, &lineBgdMask);

	m_vecReturn.clear();
	m_vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CFlatField::GetAvgMask, this, imgFrd, &frdMask, &tempAvgFrd, &lineFrdMask)));
	m_vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CFlatField::GetAvgMask, this, imgBgd, &bgdMask, &tempAvgBgd, &lineBgdMask)));
	
	for (int i = 0; i < m_vecReturn.size(); i++)
	{
		bool hr = m_vecReturn[i].get();
		if (hr == false)
		{
			printf("Flat field Add occurred some unhappy! Info: %d\n",i);
			return false;
		}
	}
	delete imgFrd;
	delete imgBgd;
	m_vecReturn.clear();

	cv::Mat frdNum, bgdNum, frdSum, bgdSum, bgd[2], frd[2];
	if (iFreadIdx == 1)
	{
		avgBgd = tempAvgBgd;
		avgFrd = tempAvgFrd;
	}
	else
	{
		cv::Mat mask = (lineFrdMask != 0.0f);
		cv::Mat tempAvg;
		cv::add(avgFrd, tempAvgFrd, tempAvg, mask);
		tempAvg.convertTo(tempAvg, CV_32F, 1.0 / double(iFreadIdx));
		tempAvg.copyTo(avgFrd, mask);

		mask = (lineBgdMask != 0.0f);
		cv::add(avgBgd, tempAvgBgd, tempAvg, mask);
		tempAvg.convertTo(tempAvg, CV_32F, 1.0 / double(iFreadIdx));
		tempAvg.copyTo(avgBgd, mask);

		
		//bgdSum_Num.convertTo(bgdSum_Num, CV_32F, 1.0 / (double)iFreadIdx);
		//frdSum_Num.convertTo(frdSum_Num, CV_32F, 1.0 / (double)iFreadIdx);
		/*cv::split(bgdSum_Num, bgd);
		cv::split(frdSum_Num, frd);
		frdSum = frd[0];
		bgdSum = bgd[0];
		frdNum = frd[1];
		bgdNum = bgd[1];*/
	}

/*	cv::Mat frd_mask, bgd_mask;
	frd_mask = frdMask;
	bgd_mask = ~frd_mask;

	cv::Mat img32S;
	imgFream.convertTo(img32S, CV_32F);
	cv::add(frdSum, img32S, frdSum, frd_mask);
	cv::add(bgdSum, img32S, bgdSum, bgd_mask);

	frd_mask.convertTo(frd_mask, CV_32F, 1.0 / 255.0);
	bgd_mask.convertTo(bgd_mask, CV_32F, 1.0 / 255.0);
	frdNum = frdNum + frd_mask;
	bgdNum = bgdNum + bgd_mask;

	if (iFreadIdx == 0)
	{
		frd[0] = frdSum;
		bgd[0] = bgdSum;
		frd[1] = frdNum;
		bgd[1] = bgdNum;
	}
	cv::merge(frd, 2, frdSum_Num);
	cv::merge(bgd, 2, bgdSum_Num);*/

	return true;
}

bool CFlatField::GetFlatFieldParam(cv::Mat& avgBgd, cv::Mat& avgFrd, cv::Mat& bgdParam, cv::Mat& frdParam, int iDstBgd/* = 240*/, int iDstFrd/* = 128*/)
{
	/*cv::Mat bgd[2], frd[2];
	cv::split(bgdSum_Num, bgd);
	cv::split(frdSum_Num, frd);

	cv::Mat Sum, Num, Dst;
	Sum = cv::Mat::zeros(1, bgd[0].cols, CV_32F);
	Num = Sum.clone();
	Dst = Sum.clone();

	cv::reduce(bgd[0], Sum, 0, CV_REDUCE_SUM);
	cv::reduce(bgd[1], Num, 0, CV_REDUCE_SUM);
	//Sum.convertTo(Sum, CV_32F, 255.0);
	Sum = Sum / Num;
	Dst.setTo(float(iDstBgd));
	cv::Mat bgdLineParam = Dst / Sum;

	Sum.setTo(0x00);
	Num.setTo(0x00);
	cv::reduce(frd[0], Sum, 0, CV_REDUCE_SUM);
	cv::reduce(frd[1], Num, 0, CV_REDUCE_SUM);
	//Sum.convertTo(Sum, CV_32F, 255.0);
	Sum = Sum / Num;
	Dst.setTo(float(iDstFrd));
	cv::Mat frdLineParam = Dst / Sum;*/

	cv::Size imgSize = getHandle()->ImageSizePre();
	if (imgSize.width != avgBgd.cols || avgBgd.cols != avgFrd.cols)
	{
		return false;
	}

	cv::Mat Dst = cv::Mat::zeros(avgBgd.rows, avgBgd.cols, CV_32F);
	Dst.setTo(float(iDstBgd));
	cv::Mat bgdLineParam = Dst / avgBgd;

	Dst.setTo(float(iDstFrd));
	cv::Mat frdLineParam = Dst / avgFrd;

	bgdParam = cv::Mat(imgSize, CV_32F);
	frdParam = cv::Mat(imgSize, CV_32F);

	for (int i = 0; i < imgSize.height; i++)
	{
		memcpy(bgdParam.data + i*bgdParam.cols*sizeof(float), bgdLineParam.data, sizeof(float)*bgdParam.cols);
		memcpy(frdParam.data + i*frdParam.cols*sizeof(float), frdLineParam.data, sizeof(float)*frdParam.cols);
	}

	return true;
}
bool CFlatField::FlatField(cv::Mat* SrcImg, cv::Mat* DstImg, cv::Mat* bgdParam, cv::Mat* frdParam, cv::Mat* frdMask, cv::Rect roi/* = NULL*/)
{
	if (frdMask->size != SrcImg->size || SrcImg->size != bgdParam->size || SrcImg->size != frdParam->size)
	{
		return false;
	}

	//cv::Mat temp32F, dst32f;
	if (roi.x < 0 || roi.y<0 || roi.x + roi.width > SrcImg->cols || roi.y + roi.height > SrcImg->rows)
	{
		return false;
	}
	int iStep = roi.y*SrcImg->cols;
	float* pFrd = (float*)frdParam->data;
	for (int i = roi.y; i < roi.y+roi.height; i++)
	{
		for (int j = roi.x; j < roi.x + roi.width; j++)
		{
			if (frdMask->data[iStep+j]==0)
			{
				DstImg->data[iStep + j] = m_iDstBgd;
				continue;
			}
			DstImg->data[iStep + j] = std::min(255, int(SrcImg->data[iStep + j] * pFrd[iStep + j] + 0.5f));
		}
		iStep += SrcImg->cols;
	}
	//(*DstImg)(roi).setTo(0xff, ~((*frdMask)(roi)));
	/*cv::Mat PatchImg = (*SrcImg)(roi);
	cv::Mat frdp = (*frdParam)(roi);
	cv::Mat bgdp = (*bgdParam)(roi);
	cv::Mat mask = (*frdMask)(roi);
	cv::Mat dst = (*DstImg)(roi);

	PatchImg.convertTo(temp32F, CV_32F);
	dst32f = temp32F.mul(frdp);
	dst32f.setTo(0xff, ~mask);
	//temp32F = temp32F.mul(bgdp);
	//temp32F.copyTo(dst32f, ~mask);
	dst32f.convertTo(dst, CV_8U);*/

	return true;
}

void CFlatField::SetParam(void* param)
{

}

bool CFlatField::GetAvgMask(cv::Mat* src, cv::Mat* mask, cv::Mat* lineSrc, cv::Mat* lineMask)
{
	if (src->size!=mask->size)
	{
		return false;
	}

	(*lineSrc) = cv::Mat::zeros(1, src->cols, CV_32F);
	(*lineMask) = lineSrc->clone();

	float* pSrc = (float*)lineSrc->data;
	float* pMask = (float*)lineMask->data;
	int iStep = 0;
	for (int i = 0; i < src->rows; i++)
	{
		for (int j = 0; j < src->cols; j++)
		{
			pSrc[j] += src->data[iStep + j];
			pMask[j] += mask->data[iStep + j];
		}
		iStep += src->cols;
	}
	for (int j = 0; j < src->cols; j++)
	{
		pMask[j] *= 0.00392;
		if (pMask[j]==0)
		{
			pSrc[j] = 0;
			continue;
		}
		pSrc[j] /= pMask[j];
	}
	/*src->convertTo(*src, CV_32F);
	mask->convertTo(*mask, CV_32F);
	cv::reduce(*src, *lineSrc, 0, CV_REDUCE_SUM);
	cv::reduce(*mask, *lineMask, 0, CV_REDUCE_SUM);
	//lineSrc->convertTo(*lineSrc, CV_32F);
	lineMask->convertTo(*lineMask, CV_32F, 1.0 / 255.0);
	(*lineSrc) = (*lineSrc) / (*lineMask);*/
	return true;
}

bool CFlatField::ReduceAdd(cv::Mat* src, cv::Mat* mask, cv::Mat* lineSrc, cv::Mat* lineMask, cv::Rect roi)
{
	if (src->size!=mask->size || roi.x<0 || roi.y<0 || roi.x+roi.width>src->cols || roi.y+roi.height>src->rows)
	{
		return false;
	}

	(*lineSrc) = cv::Mat::zeros(1, src->cols, CV_32F);
	(*lineMask) = lineSrc->clone();
	cv::Mat tempSrc = (*src)(roi);
	cv::Mat tempMask = (*mask)(roi);

	cv::reduce(tempSrc, *lineSrc, 0, CV_REDUCE_SUM);
	cv::reduce(tempMask, *lineMask, 0, CV_REDUCE_SUM);

	lineMask->convertTo(*lineMask, CV_32F, 1.0 / 255.0);

	return true;
}

bool CFlatField::ImageAdd_gpu(cv::cuda::GpuMat& imgFream, cv::cuda::GpuMat& avgBgd, cv::cuda::GpuMat& avgFrd, cv::cuda::GpuMat& frdMask, int iFreadIdx /*= 0*/)
{
	if (frdMask.size() != imgFream.size())
	{
		return false;
	}

	cv::cuda::GpuMat tempAvgFrd, tempAvgBgd, lineFrdMask, lineBgdMask;

	GetAvgMask_gpu2(&imgFream, &frdMask, &tempAvgFrd, &lineFrdMask, &tempAvgBgd, &lineBgdMask);

	//printf("... is begin\n");
	/*cv::cuda::GpuMat* imgFrd = new cv::cuda::GpuMat(imgFream.rows, imgFream.cols, CV_8U);
	imgFrd->setTo(0x00);
	imgFream.copyTo(*imgFrd, frdMask);
	GetAvgMask_gpu2(imgFrd, &frdMask, &tempAvgFrd, &lineFrdMask);
	delete imgFrd;

	//printf("... is middle\n");
	cv::cuda::GpuMat* imgBgd = new cv::cuda::GpuMat(imgFream.rows, imgFream.cols, CV_8U);
	imgBgd->setTo(0x00);
	cv::cuda::GpuMat bgdMask;
	cv::cuda::bitwise_not(frdMask, bgdMask);
	imgFream.copyTo(*imgBgd, bgdMask);
	GetAvgMask_gpu2(imgBgd, &bgdMask, &tempAvgBgd, &lineBgdMask);
	delete imgBgd;*/
	
	//printf("... is ok\n");

	cv::Mat frdNum, bgdNum, frdSum, bgdSum, bgd[2], frd[2];
	if (iFreadIdx == 1)
	{
		avgBgd = tempAvgBgd;
		avgFrd = tempAvgFrd;
	}
	else
	{
		//逻辑运算
		/*cv::Mat tempLine;
		lineFrdMask.download(tempLine);
		cv::Mat tempMask = (tempLine != 0.0f);
		cv::cuda::GpuMat mask;
		mask.upload(tempMask);*/
// 		cv::Mat tempMat;
// 		avgFrd.download(tempMat);

		cv::cuda::GpuMat mask;
		cv::cuda::compare(lineFrdMask, 0, mask, cv::CMP_NE);
		//
		cv::cuda::GpuMat tempAvg;
		cv::cuda::add(avgFrd, tempAvgFrd, tempAvg, mask);
		tempAvg.convertTo(tempAvg, CV_32F, 1.0 / double(iFreadIdx));
		tempAvg.copyTo(avgFrd, mask);

		//avgFrd.download(tempMat);

		//逻辑运算
		/*lineBgdMask.download(tempLine);
		tempMask = (tempLine != 0.0f);
		mask.upload(tempMask);*/
		cv::cuda::compare(lineBgdMask, 0, mask, cv::CMP_NE);
		//
		cv::cuda::add(avgBgd, tempAvgBgd, tempAvg, mask);
		tempAvg.convertTo(tempAvg, CV_32F, 1.0 / double(iFreadIdx));
		tempAvg.copyTo(avgBgd, mask);
	}

	//m_blur->apply(avgFrd, avgFrd);
	//m_blur->apply(avgBgd, avgBgd);

	return true;
}

bool CFlatField::GetFlatFieldParam_gpu(cv::cuda::GpuMat& avgBgd, cv::cuda::GpuMat& avgFrd, cv::cuda::GpuMat& bgdParam, cv::cuda::GpuMat& frdParam, int iDstBgd /*= 240*/, int iDstFrd /*= 128*/)
{
	cv::Size imgSize = getHandle()->ImageSizePre();
	if (imgSize.width != avgBgd.cols || avgBgd.cols != avgFrd.cols)
	{
		return false;
	}

	cv::cuda::GpuMat Dst = cv::cuda::GpuMat(avgBgd.rows, avgBgd.cols, CV_32F);
	//Dst.setTo(float(iDstBgd));

	cv::cuda::GpuMat bgdLineParam, frdLineParam;
	//cv::cuda::divide(Dst, avgBgd, bgdLineParam);

	Dst.setTo(float(iDstFrd));
	cv::cuda::divide(Dst, avgFrd, frdParam);

	//bgdParam = cv::cuda::GpuMat(imgSize, CV_32F);
	/*frdParam = cv::cuda::GpuMat(1, imgSize.width, CV_32F);

	for (int i = 0; i < imgSize.height; i++)
	{
		//cudaMemcpy(bgdParam.data + i*bgdParam.cols*sizeof(float), bgdLineParam.data, sizeof(float)*bgdParam.cols, cudaMemcpyDeviceToDevice);
		cudaMemcpy(frdParam.data + i*frdParam.cols*sizeof(float), frdLineParam.data, sizeof(float)*frdParam.cols, cudaMemcpyDeviceToDevice);
	}*/

	return true;
}

bool CFlatField::GetAvgMask_gpu2(cv::cuda::GpuMat* src, cv::cuda::GpuMat* mask, cv::cuda::GpuMat* lineSrcFrd, cv::cuda::GpuMat* lineMaskFrd, cv::cuda::GpuMat* lineSrcBgd, cv::cuda::GpuMat* lineMaskBgd)
{
	/*(*lineSrc) = cv::cuda::GpuMat(1, src->cols, CV_32F);
	lineSrc->setTo(0x00);
	(*lineMask) = lineSrc->clone();
	cv::cuda::GpuMat tempSrc, tempMask;
	src->convertTo(tempSrc, CV_32F);
	mask->convertTo(tempMask, CV_32F);
	cv::cuda::reduce(tempSrc, *lineSrc, 0, CV_REDUCE_SUM);
	cv::cuda::reduce(tempMask, *lineMask, 0, CV_REDUCE_SUM);

	//lineSrc->convertTo(*lineSrc, CV_32F);
	lineMask->convertTo(*lineMask, CV_32F, 1.0 / 255.0);
	cv::cuda::divide(*lineSrc, *lineMask, *lineSrc);
	//(*lineSrc) = (*lineSrc) / (*lineMask);*/

	GetAvgMask_gpu(src, mask, lineSrcFrd, lineMaskFrd, lineSrcBgd, lineMaskBgd);
	//cv::Mat showMat;
	//lineSrc->download(showMat);
	//lineMask->download(showMat);
	//lineSrc->convertTo(*lineSrc, CV_32F);
	//lineSrc->download(showMat);
	//lineMask->convertTo(*lineMask, CV_32F/*, 1.0 / 255.0*/);
	//lineMask->download(showMat);
	cv::cuda::divide(*lineSrcFrd, *lineMaskFrd, *lineSrcFrd);
	cv::cuda::divide(*lineSrcBgd, *lineMaskBgd, *lineSrcBgd);
	//lineSrc->download(showMat);
	return true;
}

void CFlatField::TuneImgSelf(cv::Mat& srcdst, cv::Mat& frdMask)
{
	//抽取固定行数
	//cv::Mat tempMat, curver;
	srcdst.convertTo(m_frdSum_Num, CV_32F);
	cv::reduce(m_frdSum_Num, m_frdParam, 0, CV_REDUCE_AVG);
	if (m_target.size()!=m_frdParam.size())
	{
		m_target = cv::Mat::zeros(m_frdParam.size(), CV_32F);
		m_target.setTo(m_iDstFrd);
	}
	cv::divide(m_target, m_frdParam, m_frdParam);
	for (int i = 0; i < srcdst.rows; i++)
	{
		cv::Mat tempLine = m_frdSum_Num.rowRange(i, i + 1);
		tempLine = tempLine.mul(m_frdParam);
	}
	m_frdSum_Num.convertTo(srcdst, CV_8UC1);
	srcdst.setTo(m_iDstBgd, ~frdMask);
}

bool CFlatField::TuneImgSelf(cv::cuda::GpuMat& srcdst, cv::cuda::GpuMat& frdMask, double* dTime)
{
	*dTime = cvGetTickCount();
	if (m_target_gpu.rows != 1 || m_target_gpu.cols != srcdst.cols)
	{
		/*m_target_gpu = cv::cuda::GpuMat(1, srcdst.cols, CV_32F);
		m_target_gpu.setTo(m_iDstFrd);
		m_frdParam_gpu = cv::cuda::GpuMat(1, srcdst.cols, CV_32F);
		m_frdParam_gpu.setTo(0);*/

		m_target_gpu = cv::cuda::GpuMat(srcdst.size(), srcdst.type());
	}

 	//cv::Mat src1, mask1;
 	//srcdst.download(src1);
 	//frdMask.download(mask1);
	//int iTempI = m_iOffsetHeightIndex;
	//int iTempD = m_iDstBgd;
	//float fTime = TuneImgSelf_gpu(srcdst, frdMask, m_frdParam_gpu, m_target_gpu, m_iOffsetHeightIndex, m_iDstBgd);
	float fTime = TuneImgSelf_gpu(srcdst, m_target_gpu, frdMask,m_iOffsetHeightIndex, m_iDstFrd, m_iDstBgd);

 	//srcdst.download(src1);
 	//frdMask.download(mask1);
	*dTime = (cv::getTickCount() - (*dTime)) / (1000 * cvGetTickFrequency()) + fTime;
	return true;
}

/*
bool CFlatField::GetAvgMask_gpu(cv::cuda::GpuMat* src, cv::cuda::GpuMat* mask, cv::cuda::GpuMat* lineSrc, cv::cuda::GpuMat* lineMask)
{
	if (src->size != mask->size)
	{
		return false;
	}

	(*lineSrc) = cv::Mat::zeros(1, src->cols, CV_32F);
	(*lineMask) = lineSrc->clone();

	float* pSrc = (float*)lineSrc->data;
	float* pMask = (float*)lineMask->data;
	int iStep = 0;
	for (int i = 0; i < src->rows; i++)
	{
		for (int j = 0; j < src->cols; j++)
		{
			pSrc[j] += src->data[iStep + j];
			pMask[j] += mask->data[iStep + j];
		}
		iStep += src->cols;
	}
	for (int j = 0; j < src->cols; j++)
	{
		pMask[j] *= 0.00392;
		if (pMask[j] == 0)
		{
			pSrc[j] = 0;
			continue;
		}
		pSrc[j] /= pMask[j];
	}

	return true;
}*/

