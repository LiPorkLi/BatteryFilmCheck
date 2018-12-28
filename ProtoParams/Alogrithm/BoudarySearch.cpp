#include "stdafx.h"
#include "BoudarySearch.h"
#include "cudaFunction.h"

CBoudarySearch::CBoudarySearch(MainParam::param* p, std::shared_ptr<CRunTimeHandle> pHandle /*= NULL*/) : CAlogrithmBase(p, pHandle)
{
	ParamHelper<Parameters::InspectParam> helper(getParam());
	Parameters::InspectParam inspectionParam = helper.getRef();
	
	m_vecBoudaryParam.resize(inspectionParam.boundsearchlist_size());//if boundary param list size is 0, then, set one product default, and left boundary search param is from 0 to half cols. 
	for (int i = 0; i < inspectionParam.boundsearchlist_size(); i++)
	{
		int iLeft1 = int(PyhsicToPixel_1D(inspectionParam.boundsearchlist(i).leftrange1(), getHandle()->PhysicResolution_x(), getHandle()->DownSampleFator()) + 0.5f);
		int iLeft2 = int(PyhsicToPixel_1D(inspectionParam.boundsearchlist(i).leftrange1(), getHandle()->PhysicResolution_x(), getHandle()->DownSampleFator()) + 0.5f);
		int iRight1 = int(PyhsicToPixel_1D(inspectionParam.boundsearchlist(i).leftrange1(), getHandle()->PhysicResolution_x(), getHandle()->DownSampleFator()) + 0.5f);
		int iRight2 = int(PyhsicToPixel_1D(inspectionParam.boundsearchlist(i).leftrange1(), getHandle()->PhysicResolution_x(), getHandle()->DownSampleFator()) + 0.5f);

		/*int iLeft1 = int(inspectionParam.boundsearchlist(i).leftrange1() * fPhysicResolution + 0.5f) >> iDownSampleParam;
		int iLeft2 = int(inspectionParam.boundsearchlist(i).leftrange2() * fPhysicResolution + 0.5f) >> iDownSampleParam;
		int iRight1 = int(inspectionParam.boundsearchlist(i).rightrange1() * fPhysicResolution + 0.5f) >> iDownSampleParam;
		int iRight2 = int(inspectionParam.boundsearchlist(i).rightrange2() * fPhysicResolution + 0.5f) >> iDownSampleParam;*/
		if (iLeft2 < iLeft1 || iRight2 < iRight1 || iLeft1<0 || iRight1<0/* || iLeft2>m_imgSize.width - 1 || iRight2 > m_imgSize.width - 1*/)
		{
			//抛出异常
		}

		m_vecBoudaryParam[i] = std::make_pair(std::make_pair(iLeft1, iLeft2), std::make_pair(iRight1, iRight2));
	}
	m_vecvecBoundary.clear();

	dev_pLeft = dev_pRight = m_pLeft = m_pRight = NULL;
	if (getHandle()->IsGpu()==true)
	{
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(100, 1));
		m_f_erode = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, kernel);
		m_f_dilate = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, kernel);

		//cv::Sobel(PatchImg, dev_grad_x, CV_16S, 1, 0, 3);
		m_f_sobel = cv::cuda::createSobelFilter(CV_8UC1, CV_32S, 1, 0, 3);

		cudaMalloc((void**)&dev_pRight, sizeof(int)*getHandle()->ImageSizePre().height);
		cudaMalloc((void**)&dev_pLeft, sizeof(int)*getHandle()->ImageSizePre().height);
		m_pLeft = new int[getHandle()->ImageSizePre().height];
		m_pRight = new int[getHandle()->ImageSizePre().height];
	}
}


CBoudarySearch::~CBoudarySearch()
{ 
	for (int i = 0; i < m_vecvecBoundary.size(); i++)
	{
		m_vecvecBoundary[i].clear();
	}
	m_vecvecBoundary.clear();

	for (int i = 0; i < m_vecReturn.size(); i++)
	{
		m_vecReturn[i].get();
	}
	m_vecReturn.clear();

	m_vecBoudaryParam.clear();
	if (dev_pLeft)
	{
		cudaFree(dev_pLeft);
	}
	if (dev_pRight)
	{
		cudaFree(dev_pRight);
	}
	if (m_pLeft)
	{
		delete m_pLeft;
	}
	if (m_pRight)
	{
		delete m_pRight;
	}
}

bool CBoudarySearch::BoundarySearch(cv::Mat& SrcImg, cv::cuda::GpuMat& frdMask, double* dTime)
{
	*dTime = cvGetTickCount();
	if (m_vecBoudaryParam.size() == 0)
	{
		int iLeft1, iLeft2, iRight1, iRight2;
		if (m_vecvecBoundary.size() == 0)
		{
			//from middle to outside, and is one left boundary, one right boundary by default
			cv::Rect topRect, bottomRect;
			topRect.x = 0;
			topRect.width = SrcImg.cols;
			topRect.y = int(SrcImg.rows*0.01);
			topRect.height = std::min(topRect.y + 100, SrcImg.rows - 1);
			topRect.height = topRect.height - topRect.y + 1;

			bottomRect.x = 0;
			bottomRect.width = SrcImg.cols;
			bottomRect.y = int(SrcImg.rows*0.9);
			bottomRect.height = std::min(bottomRect.y + 100, SrcImg.rows - 1);
			bottomRect.height = bottomRect.height - bottomRect.y + 1;

			int left[2], right[2];
			cv::Mat PatchImgTop = SrcImg(topRect).clone();
			cv::Mat PatchImgBottom = SrcImg(bottomRect).clone();
			if (getHandle() == NULL)
			{
				GetPatchImgBoundary(PatchImgTop, 25, 230, &left[0], &right[0]);
				GetPatchImgBoundary(PatchImgBottom, 25, 230, &left[1], &right[1]);
			}
			else
			{
				m_vecReturn.clear();
				m_vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CBoudarySearch::GetPatchImgBoundary, this, PatchImgTop, 25, 230, &left[0], &right[0])));
				m_vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CBoudarySearch::GetPatchImgBoundary, this, PatchImgBottom, 25, 230, &left[1], &right[1])));
				for (int i = 0; i < m_vecReturn.size(); i++)
				{
					m_vecReturn[i].get();
				}
				m_vecReturn.clear();
			}

			if (right[0] == 0 && right[1] == 0)
			{
				printf("1.......................%d %d\n", right[0], right[1]);
				return false;
			}

			iLeft1 = std::max(0, std::min(left[0], left[1]) - 100);
			iLeft2 = std::min(SrcImg.cols - 1, std::max(left[0], left[1]) + 100);
			iRight1 = std::max(0, std::min(right[0], right[1]) - 100);
			iRight2 = std::min(SrcImg.cols - 1, std::max(right[0], right[1]) + 100);

			m_vecvecBoundary.resize(1);
			m_vecvecBoundary[0].resize(SrcImg.rows);
		}
		else
		{
			if (m_vecvecBoundary[0].size() != SrcImg.rows)
			{
				printf("2.......................\n");
				return false;
			}
			iLeft1 = iLeft2 = m_vecvecBoundary[0][0].first;
			iRight1 = iRight2 = m_vecvecBoundary[0][0].second;
			for (int i = 1; i < m_vecvecBoundary[0].size(); i++)
			{
				iLeft1 = std::min(iLeft1, m_vecvecBoundary[0][i].first);
				iLeft2 = std::max(iLeft2, m_vecvecBoundary[0][i].first);

				iRight1 = std::min(iRight1, m_vecvecBoundary[0][i].second);
				iRight2 = std::max(iRight2, m_vecvecBoundary[0][i].second);
			}

			iLeft1 = std::max(0, iLeft1 - 100);
			iLeft2 = std::min(SrcImg.cols - 1, iLeft2 + 100);

			iRight1 = std::max(0, iRight1 - 100);
			iRight2 = std::min(SrcImg.cols - 1, iRight2 + 100);
		}
		if (iLeft2 > iRight1)
		{
			printf("3.......................%d %d\n", iLeft2, iRight1);
			return false;
		}

		std::vector<int> vecLineLeft, vecLineRight;

		if (getHandle() == NULL)
		{
			//SearchLine(SrcImg, iLeft1, iLeft2, &vecLineLeft);
			//SearchLine(SrcImg, iRight1, iRight2, &vecLineRight);
		}
		else
		{
			m_vecReturn.clear();
			m_vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CBoudarySearch::SearchLine2, this, &SrcImg, iLeft1, iLeft2, &vecLineLeft)));
			m_vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CBoudarySearch::SearchLine2, this, &SrcImg, iRight1, iRight2, &vecLineRight)));
			for (int i = 0; i < m_vecReturn.size(); i++)
			{
				m_vecReturn[i].get();
			}
			m_vecReturn.clear();
		}
		
		if (vecLineLeft.size() != vecLineRight.size() || vecLineLeft.size() != SrcImg.rows)
		{
			vecLineLeft.clear();
			vecLineRight.clear();

			printf("4.......................\n");
			return false;
		}
		for (int j = 0; j < vecLineLeft.size(); j++)
		{
			if (vecLineLeft[j] > vecLineRight[j])
			{
				vecLineLeft.clear();
				vecLineRight.clear();
				printf("5......................\n");
				return false;
			}
			m_vecvecBoundary[0][j] = std::make_pair(vecLineLeft[j], vecLineRight[j]);
		}
		vecLineLeft.clear();
		vecLineRight.clear();
	}
	else
	{
		for (int i = 0; i < m_vecBoudaryParam.size(); i++)
		{
			if (m_vecBoudaryParam[i].first.first < 0 || m_vecBoudaryParam[i].first.second>SrcImg.cols - 1 || m_vecBoudaryParam[i].second.first < 0 || m_vecBoudaryParam[i].second.second>SrcImg.cols - 1)
			{
				printf("6.......................\n");
				return false;
			}
		}

		for (int i = 0; i < m_vecvecBoundary.size(); i++)
		{
			m_vecvecBoundary[i].clear();
		}
		m_vecvecBoundary.clear();

		int iMatchBoundaryNum = int(m_vecBoudaryParam.size());
		m_vecvecBoundary.resize(iMatchBoundaryNum);
		for (int i = 0; i < iMatchBoundaryNum; i++)
		{
			int iLeft1 = m_vecBoudaryParam[i].first.first;
			int iLeft2 = m_vecBoudaryParam[i].first.second;
			int iRight1 = m_vecBoudaryParam[i].second.first;
			int iRight2 = m_vecBoudaryParam[i].second.second;
			std::vector<int> vecLineLeft, vecLineRight;
			//SearchLine(SrcImg, iLeft1, iLeft2, vecLineLeft);
			//SearchLine(SrcImg, iRight1, iRight2, vecLineRight);
			if (getHandle() == NULL)
			{
				//SearchLine(SrcImg, iLeft1, iLeft2, &vecLineLeft);
				//SearchLine(SrcImg, iRight1, iRight2, &vecLineRight);
			}
			else
			{
				m_vecReturn.clear();
				m_vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CBoudarySearch::SearchLine2, this, &SrcImg, iLeft1, iLeft2, &vecLineLeft)));
				m_vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CBoudarySearch::SearchLine2, this, &SrcImg, iRight1, iRight2, &vecLineRight)));
				for (int k = 0; k < m_vecReturn.size(); k++)
				{
					m_vecReturn[k].get();
				}
				m_vecReturn.clear();
			}

			if (vecLineLeft.size() != vecLineRight.size() || vecLineLeft.size() != SrcImg.rows)
			{
				vecLineLeft.clear();
				vecLineRight.clear();
				printf("7.......................\n");
				return false;
			}
			for (int j = 0; j < vecLineLeft.size(); j++)
			{
				if (vecLineLeft[j] > vecLineRight[j])
				{
					vecLineLeft.clear();
					vecLineRight.clear();
					printf("8.......................%d %d\n", vecLineLeft[j], vecLineRight[j]);
					return false;
				}
				m_vecvecBoundary[i].push_back(std::make_pair(vecLineLeft[j], vecLineRight[j]));
			}
			vecLineLeft.clear();
			vecLineRight.clear();
		}
	}

	bool hr = GetFrdMask(cv::Size(SrcImg.cols,SrcImg.rows), m_vecvecBoundary, frdMask);

	*dTime = (cv::getTickCount() - (*dTime)) / (1000 * cvGetTickFrequency());
	return hr;
}

bool CBoudarySearch::BoundarySearch(cv::cuda::GpuMat& SrcImg, cv::cuda::GpuMat& frdMask_gpu, cv::Mat& frdMask_cpu, double* dTime)
{
	*dTime = cvGetTickCount();
	if (m_vecBoudaryParam.size() == 0)
	{
		//printf("Run This batch....if\n");
		int iLeft1, iLeft2, iRight1, iRight2;
		if (m_vecvecBoundary.size() == 0)
		{
			//from middle to outside, and is one left boundary, one right boundary by default
			cv::Rect topRect, bottomRect;
			topRect.x = 0;
			topRect.width = SrcImg.cols;
			topRect.y = int(SrcImg.rows*0.01);
			topRect.height = std::min(topRect.y + 100, SrcImg.rows - 1);
			topRect.height = topRect.height - topRect.y + 1;

			bottomRect.x = 0;
			bottomRect.width = SrcImg.cols;
			bottomRect.y = int(SrcImg.rows*0.9);
			bottomRect.height = std::min(bottomRect.y + 100, SrcImg.rows - 1);
			bottomRect.height = bottomRect.height - bottomRect.y + 1;

			int left[2], right[2];
			cv::cuda::GpuMat PatchImgTop = SrcImg(topRect).clone();
			cv::cuda::GpuMat PatchImgBottom = SrcImg(bottomRect).clone();
			GetPatchImgBoundary_gpu(PatchImgTop, 60, 190, &left[0], &right[0]);
			GetPatchImgBoundary_gpu(PatchImgBottom, 60, 190, &left[1], &right[1]);

			iLeft1 = std::max(0, std::min(left[0], left[1]) - 100);
			iLeft2 = std::min(SrcImg.cols - 1, std::max(left[0], left[1]) + 100);
			iRight1 = std::max(0, std::min(right[0], right[1]) - 100);
			iRight2 = std::min(SrcImg.cols - 1, std::max(right[0], right[1]) + 100);

			m_vecvecBoundary.resize(1);
			m_vecvecBoundary[0].resize(SrcImg.rows);
		}
		else
		{
			//printf("Run This batch....else\n");
			if (m_vecvecBoundary[0].size() != SrcImg.rows)
			{
				return false;
			}
			iLeft1 = iLeft2 = m_vecvecBoundary[0][0].first;
			iRight1 = iRight2 = m_vecvecBoundary[0][0].second;
			for (int i = 1; i < m_vecvecBoundary[0].size(); i++)
			{
				iLeft1 = std::min(iLeft1, m_vecvecBoundary[0][i].first);
				iLeft2 = std::max(iLeft2, m_vecvecBoundary[0][i].first);

				iRight1 = std::min(iRight1, m_vecvecBoundary[0][i].second);
				iRight2 = std::max(iRight2, m_vecvecBoundary[0][i].second);
			}

			iLeft1 = std::max(0, iLeft1 - 100);
			iLeft2 = std::min(SrcImg.cols - 1, iLeft2 + 100);

			iRight1 = std::max(0, iRight1 - 100);
			iRight2 = std::min(SrcImg.cols - 1, iRight2 + 100);
		}
		if (iLeft2 > iRight1)
		{
			return false;
		}

		std::vector<int> vecLineLeft, vecLineRight;

		SearchLine_gpu(SrcImg, iLeft1, iLeft2, &vecLineLeft);
		SearchLine_gpu(SrcImg, iRight1, iRight2, &vecLineRight);

		if (vecLineLeft.size() != vecLineRight.size() || vecLineLeft.size() != SrcImg.rows)
		{
			vecLineLeft.clear();
			vecLineRight.clear();
			return false;
		}
		for (int j = 0; j < vecLineLeft.size(); j++)
		{
			if (vecLineLeft[j] > vecLineRight[j])
			{
				vecLineLeft.clear();
				vecLineRight.clear();
				return false;
			}
			m_vecvecBoundary[0][j] = std::make_pair(vecLineLeft[j], vecLineRight[j]);
		}
		vecLineLeft.clear();
		vecLineRight.clear();
	}
	else
	{
	
		for (int i = 0; i < m_vecBoudaryParam.size(); i++)
		{
			if (m_vecBoudaryParam[i].first.first < 0 || m_vecBoudaryParam[i].first.second>SrcImg.cols - 1 || m_vecBoudaryParam[i].second.first < 0 || m_vecBoudaryParam[i].second.second>SrcImg.cols - 1)
			{
				return false;
			}
		}

		for (int i = 0; i < m_vecvecBoundary.size(); i++)
		{
			m_vecvecBoundary[i].clear();
		}
		m_vecvecBoundary.clear();

		int iMatchBoundaryNum = int(m_vecBoudaryParam.size());
		m_vecvecBoundary.resize(iMatchBoundaryNum);
		for (int i = 0; i < iMatchBoundaryNum; i++)
		{
			int iLeft1 = m_vecBoudaryParam[i].first.first;
			int iLeft2 = m_vecBoudaryParam[i].first.second;
			int iRight1 = m_vecBoudaryParam[i].second.first;
			int iRight2 = m_vecBoudaryParam[i].second.second;
			std::vector<int> vecLineLeft, vecLineRight;
			//SearchLine(SrcImg, iLeft1, iLeft2, vecLineLeft);
			//SearchLine(SrcImg, iRight1, iRight2, vecLineRight);
			SearchLine_gpu(SrcImg, iLeft1, iLeft2, &vecLineLeft);
			SearchLine_gpu(SrcImg, iRight1, iRight2, &vecLineRight);

			if (vecLineLeft.size() != vecLineRight.size() || vecLineLeft.size() != SrcImg.rows)
			{
				vecLineLeft.clear();
				vecLineRight.clear();
				return false;
			}
			for (int j = 0; j < vecLineLeft.size(); j++)
			{
				if (vecLineLeft[j] > vecLineRight[j])
				{
					vecLineLeft.clear();
					vecLineRight.clear();
					return false;
				}
				m_vecvecBoundary[i].push_back(std::make_pair(vecLineLeft[j], vecLineRight[j]));
			}
			vecLineLeft.clear();
			vecLineRight.clear();
		}
	}

	//cv::Mat tempFrdMask;
	bool hr = GetFrdMask(cv::Size(SrcImg.cols, SrcImg.rows), m_vecvecBoundary, frdMask_cpu);
	frdMask_gpu.upload(frdMask_cpu);

	*dTime = (cv::getTickCount() - (*dTime)) / (1000 * cvGetTickFrequency());
	return hr;
}

void CBoudarySearch::SearchLine2(cv::Mat* img, int& iS, int& iE, std::vector<int>* vecLine)
{
	cv::Mat tempImg = (*img)(cv::Rect(iS, 0, iE - iS + 1, img->rows));
	cv::Mat PatchImg;
	cv::resize(tempImg, PatchImg, cv::Size(tempImg.cols/*>>1*/, tempImg.rows/* >> 1*/));
	vecLine->resize(PatchImg.rows);
	//梯度
	cv::Mat grad_x;
	cv::Sobel(PatchImg, grad_x, CV_16S, 1, 0, 3);
	convertScaleAbs(grad_x, grad_x);
	grad_x.convertTo(grad_x, CV_32S);

	cv::Mat n = cv::Mat::zeros(1, PatchImg.cols, CV_32S);
	cv::Mat c = n.clone();
	cv::Mat record = cv::Mat::zeros(PatchImg.rows, PatchImg.cols, CV_32S);

	int* pc = (int*)c.data;
	int* pn = (int*)n.data;
	int* pr = (int*)record.data;
	int* pg = (int*)grad_x.data;
	memcpy(pc, pg, sizeof(int)*grad_x.cols);

	int iRadius = 2;
	for (int i = 1; i < PatchImg.rows; i++)
	{
		n.setTo(0x00);
		for (int j = iRadius; j < PatchImg.cols - iRadius; j++)
		{
			for (int k = -iRadius; k <= iRadius; k++)
			{
				if (pc[j + k] > pn[j])
				{
					pn[j] = pc[j + k];
					pr[j] = j + k;
				}
			}
			pn[j] += pg[i * grad_x.cols + j];
		}
		memcpy(pc, pn, sizeof(int)*grad_x.cols);
		pr += grad_x.cols;
	}

	int iMax = 0;
	for (int x = 0; x < grad_x.cols; x++)
	{
		if (pc[x] > iMax)
		{
			(*vecLine)[grad_x.rows - 1] = x;
			iMax = pc[x];
		}
	}

	pr = (int*)record.data;
	for (int h = grad_x.rows - 1; --h >= 0;)
	{
		(*vecLine)[h] = pr[h * grad_x.cols + (*vecLine)[h + 1]];
	}
	for (int i = 0; i < vecLine->size(); i++)
	{
		(*vecLine)[i] = iS + ((*vecLine)[i]/*<<1*/);
	}

	/*std::vector<int> vecTemp;
	vecTemp.resize(img->rows);
	int k = 0;
	for (; k < vecLine->size()-1; k++)
	{
		int idx = (k << 1);
		vecTemp[idx] = (*vecLine)[k];
		vecTemp[idx + 1] = int(((*vecLine)[k] + (*vecLine)[k + 1]) * 0.5f);
	}
	for (int i = (k << 1); i < img->rows; i++)
	{
		vecTemp[i] = (*vecLine)[k];
	}
	vecLine->swap(vecTemp);
	vecTemp.clear();*/
}
void CBoudarySearch::SearchLine(cv::Mat* img, int& iS, int& iE, std::vector<int>* vecLine)
{
	cv::Mat PatchImg = (*img)(cv::Rect(iS, 0, iE - iS + 1, img->rows));
	vecLine->resize(PatchImg.rows);
	//梯度
	cv::Mat grad_x;
	cv::Sobel(PatchImg, grad_x, CV_16S, 1, 0, 3);
	convertScaleAbs(grad_x, grad_x);
	grad_x.convertTo(grad_x, CV_32S);

	cv::Mat n(1, PatchImg.cols, CV_32S);
	cv::Mat c = n.clone();
	cv::Mat record(PatchImg.rows, PatchImg.cols, CV_32S);

	int* pc = (int*)c.data;
	int* pn = (int*)n.data;
	int* pr = (int*)record.data;
	int* pg = (int*)grad_x.data;
	memcpy(pc, pg, sizeof(int)*grad_x.cols);

	int iRadius = 2;
	for (int i = 1; i < PatchImg.rows; i++)
	{
		n.setTo(0x00);
		for (int j = iRadius; j < PatchImg.cols - iRadius; j++)
		{
			for (int k = -iRadius; k <= iRadius; k++)
			{
				if (pc[j + k] > pn[j])
				{
					pn[j] = pc[j + k];
					pr[j] = j + k;
				}
			}
			pn[j] += pg[i * grad_x.cols + j];
		}
		memcpy(pc, pn, sizeof(int)*grad_x.cols);
		pr += grad_x.cols;
	}

	int iMax = 0;
	for (int x = 0; x < grad_x.cols; x++)
	{
		if (pc[x] > iMax)
		{
			(*vecLine)[grad_x.rows - 1] = x;
			iMax = pc[x];
		}
	}

	pr = (int*)record.data;
	for (int h = grad_x.rows - 1; --h >= 0;)
	{
		(*vecLine)[h] = pr[h * grad_x.cols + (*vecLine)[h + 1]];
	}
	for (int i = 0; i < vecLine->size(); i++)
	{
		(*vecLine)[i] += iS;
	}
}

void CBoudarySearch::SearchLine_gpu(cv::cuda::GpuMat& img, int& iS, int& iE, std::vector<int>* vecLine)
{
	//cv::Mat PatchImg = img(cv::Rect(iS, 0, iE - iS + 1, img.rows)).clone();
	cv::cuda::GpuMat PatchImg = img(cv::Rect(iS, 0, iE - iS + 1, img.rows))/*.clone()*/;
	//cv::Mat showMat;
	//PatchImg.download(showMat);
	vecLine->resize(PatchImg.rows);
	//梯度
// 	cv::Mat grad_x;
//	cv::Sobel(PatchImg, grad_x, CV_16S, 1, 0, 3);
// 	convertScaleAbs(grad_x, grad_x);
// 	grad_x.convertTo(grad_x, CV_32S);

	cv::cuda::GpuMat dev_grad_x;
	m_f_sobel->apply(PatchImg, dev_grad_x);
	//dev_grad_x.download(showMat);
	//cv::Sobel(PatchImg, dev_grad_x, CV_16S, 1, 0, 3);
	cv::cuda::abs(dev_grad_x, dev_grad_x);
	//dev_grad_x.download(showMat);
	//dev_grad_x.convertTo(dev_grad_x, CV_32S);
	//dev_grad_x.download(showMat);
	cv::Mat grad_x;
	dev_grad_x.download(grad_x);

	cv::Mat n(1, PatchImg.cols, CV_32S);
	cv::Mat c = n.clone();
	cv::Mat record(PatchImg.rows, PatchImg.cols, CV_32S);

	int* pc = (int*)c.data;
	int* pn = (int*)n.data;
	int* pr = (int*)record.data;
	int* pg = (int*)grad_x.data;
	memcpy(pc, pg, sizeof(int)*grad_x.cols);

	int iRadius = 2;
	for (int i = 1; i < PatchImg.rows; i++)
	{
		n.setTo(0x00);
		for (int j = iRadius; j < PatchImg.cols - iRadius; j++)
		{
			for (int k = -iRadius; k <= iRadius; k++)
			{
				if (pc[j + k] > pn[j])
				{
					pn[j] = pc[j + k];
					pr[j] = j + k;
				}
			}
			pn[j] += pg[i * grad_x.cols + j];
		}
		memcpy(pc, pn, sizeof(int)*grad_x.cols);
		pr += grad_x.cols;
	}

	int iMax = 0;
	for (int x = 0; x < grad_x.cols; x++)
	{
		if (pc[x] > iMax)
		{
			(*vecLine)[grad_x.rows - 1] = x;
			iMax = pc[x];
		}
	}

	pr = (int*)record.data;
	for (int h = grad_x.rows - 1; --h >= 0;)
	{
		(*vecLine)[h] = pr[h * grad_x.cols + (*vecLine)[h + 1]];
	}
	for (int i = 0; i < vecLine->size(); i++)
	{
		(*vecLine)[i] += iS;
	}
}

void CBoudarySearch::GetPatchImgBoundary(cv::Mat& PatchImg, int iThr1, int iThr2, int* iLeft, int* iRight)
{
	cv::Mat b1, b2;
	cv::threshold(PatchImg, b1, iThr1, 0xff, cv::THRESH_BINARY);
	cv::threshold(PatchImg, b2, iThr2, 0xff, cv::THRESH_BINARY_INV);
	b1 = b1 & b2;
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(100, 1));
	cv::erode(b1, b1, kernel);
	cv::dilate(b1, b1, kernel);

	int iCount;
	*iLeft = *iRight = iCount = 0;
	int iLastLeft, iRightLeft;
	for (int i = 0; i < b1.rows; i++)
	{
		int x1, x2;
		for (int j = 0; j < b1.cols; j++)
		{
			if (b1.data[i*b1.cols + j] == 0xff)
			{
				x1 = j;
				break;
			}
		}
		for (int j = b1.cols; --j >= 0;)
		{
			if (b1.data[i*b1.cols + j] == 0xff)
			{
				x2 = j;
				break;
			}
		}
		if (x2 <= x1)
		{
			continue;
		}
		*iLeft += x1;
		*iRight += x2;
		iCount++;
	}
	if (iCount!=0)
	{
		*iLeft /= iCount;
		*iRight /= iCount;
	}
	else
	{
		*iLeft = getHandle()->ImageSizeSrc().width;
		*iRight = 0;
	}
}

void CBoudarySearch::GetPatchImgBoundary_gpu(cv::cuda::GpuMat& PatchImg, int iThr1, int iThr2, int* iLeft, int* iRight)
{
	//cv::Mat showMat;
	//PatchImg.download(showMat);

	cv::Mat b1, b2;
	cv::cuda::GpuMat dev_b1, dev_b2;
	cv::cuda::threshold(PatchImg, dev_b1, iThr1, 0xff, cv::THRESH_BINARY);
	//dev_b1.download(showMat);
	cv::cuda::threshold(PatchImg, dev_b2, iThr2, 0xff, cv::THRESH_BINARY_INV);
	//dev_b2.download(showMat);
	//逻辑运算
	/*dev_b1.download(b1);
	dev_b2.download(b2);
	b1 = b1 & b2;
	dev_b1.upload(b1);*/
	cv::cuda::bitwise_and(dev_b1, dev_b2, dev_b1);
	//dev_b1.download(showMat);
	//
	//dev_b1 = dev_b1 & dev_b2;
// 	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(100, 1));
// 	cv::Ptr<cv::cuda::Filter> f_e = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, kernel);
// 	cv::Ptr<cv::cuda::Filter> f_d = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, kernel);
// 	f->

// 	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(100, 1));
// 	cv::cuda::GpuMat k;
// 	k.upload(kernel);
// 	cv::erode(dev_b1, dev_b1, k);
// 	cv::dilate(dev_b1, dev_b1, k);

	m_f_erode->apply(dev_b1, dev_b1);
	//dev_b1.download(showMat);
	m_f_dilate->apply(dev_b1, dev_b1);
	dev_b1.download(b1);

	int iCount;
	*iLeft = *iRight = iCount = 0;
	for (int i = 0; i < b1.rows; i++)
	{
		int x1, x2;
		for (int j = 0; j < b1.cols; j++)
		{
			if (b1.data[i*b1.cols + j] == 0xff)
			{
				x1 = j;
				break;
			}
		}
		for (int j = b1.cols; --j >= 0;)
		{
			if (b1.data[i*b1.cols + j] == 0xff)
			{
				x2 = j;
				break;
			}
		}
		if (x2 <= x1)
		{
			continue;
		}
		*iLeft += x1;
		*iRight += x2;
		iCount++;
	}
	*iLeft /= iCount;
	*iRight /= iCount;
}
void CBoudarySearch::SetMask_gpu(cv::cuda::GpuMat* srcdst, std::vector<std::pair<int, int>>* vecBoundary, int iStart, int iEnd)
{
	for (int i = iStart; i < iEnd; i++)
	{
		cudaMemset(srcdst->data + i*srcdst->step + (*vecBoundary)[i].first, 0xff, sizeof(uchar)*((*vecBoundary)[i].second - (*vecBoundary)[i].first + 1));
	}
}

bool CBoudarySearch::GetFrdMask(cv::Size& imgSize, std::vector<std::vector<std::pair<int, int>>>& vecvecBoundary, cv::cuda::GpuMat& frd)
{
	if (vecvecBoundary.size() == 0)
	{
		return false;
	}

	for (int i = 0; i < vecvecBoundary.size(); i++)
	{
		if (vecvecBoundary[i].size() != imgSize.height)
		{
			return false;
		}
	}

	//frd = cv::Mat::zeros(imgSize.height, imgSize.width, CV_8U);
	frd.setTo(0x00);


	m_vecReturn.clear();
	int iThread = getHandle()->ThreadNum();
	int iStep = (imgSize.height + iThread - 1) / iThread;
	for (int i = 0; i < vecvecBoundary.size(); i++)
	{
		int iStart = 0;
		for (int j = 0; j < iThread/*vecvecBoundary[i].size()*/; j++)
		{
			//memset(frd.data + j*frd.cols + vecvecBoundary[i][j].first, 0xff, sizeof(uchar)*(vecvecBoundary[i][j].second - vecvecBoundary[i][j].first + 1));
			m_vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CBoudarySearch::SetMask_gpu, this, &frd, &vecvecBoundary[i], iStart, std::min(iStart + iStep, imgSize.height))));
			iStart += iStep;
			//m_vecReturn.push_back
		}
	}

	for (int i = 0; i < m_vecReturn.size(); i++)
	{
		m_vecReturn[i].get();
	}
	m_vecReturn.clear();

	return true;
}

void CBoudarySearch::SetMask(cv::Mat* srcdst, std::vector<std::pair<int, int>>* vecBoundary, int iStart, int iEnd)
{
	for (int i = iStart; i < iEnd; i++)
	{
		memset(srcdst->data + i*srcdst->cols + (*vecBoundary)[i].first, 0xff, sizeof(uchar)*((*vecBoundary)[i].second - (*vecBoundary)[i].first + 1));
	}
}

bool CBoudarySearch::GetFrdMask(cv::Size& imgSize, std::vector<std::vector<std::pair<int, int>>>& vecvecBoundary, cv::Mat& frd)
{
	if (vecvecBoundary.size() == 0)
	{
		return false;
	}

	for (int i = 0; i < vecvecBoundary.size(); i++)
	{
		if (vecvecBoundary[i].size() != imgSize.height)
		{
			return false;
		}
	}

	//frd = cv::Mat::zeros(imgSize.height, imgSize.width, CV_8U);
	frd.setTo(0x00);


	m_vecReturn.clear();
	int iThread = getHandle()->ThreadNum();
	int iStep = (imgSize.height + iThread - 1) / iThread;
	for (int i = 0; i < vecvecBoundary.size(); i++)
	{
		int iStart = 0;
		for (int j = 0; j < iThread/*vecvecBoundary[i].size()*/; j++)
		{
			//memset(frd.data + j*frd.cols + vecvecBoundary[i][j].first, 0xff, sizeof(uchar)*(vecvecBoundary[i][j].second - vecvecBoundary[i][j].first + 1));
			m_vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CBoudarySearch::SetMask, this, &frd, &vecvecBoundary[i], iStart, std::min(iStart+iStep,imgSize.height))));
			iStart += iStep;
			//m_vecReturn.push_back
		}
	}

	for (int i = 0; i < m_vecReturn.size(); i++)
	{
		m_vecReturn[i].get();
	}
	m_vecReturn.clear();

	return true;
}

void CBoudarySearch::SetParam(void* param)
{

}

bool CBoudarySearch::ExpandBoundary(cv::cuda::GpuMat& SrcDstImg, int iSizeX, double* dTime)
{
	*dTime = cvGetTickCount();

	if (m_vecvecBoundary.size() == 0 || SrcDstImg.rows != m_vecvecBoundary[0].size())
	{
		return false;
	}
	for (int i = 0; i < m_vecvecBoundary.size(); i++)
	{
		for (int j = 0; j < m_vecvecBoundary[i].size(); j++)
		{
// 			m_pLeft[j] = m_vecvecBoundary[i][j].first;
// 			m_pRight[j] = m_vecvecBoundary[i][j].second;

			m_pLeft[j] = m_vecvecBoundary[i][j].first + 10;
			m_pRight[j] = m_vecvecBoundary[i][j].second - 10;
		}
		cudaMemcpy(dev_pLeft, m_pLeft, sizeof(int)*SrcDstImg.rows, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_pRight, m_pRight, sizeof(int)*SrcDstImg.rows, cudaMemcpyHostToDevice);
		CopyBoundary(SrcDstImg, iSizeX, dev_pLeft, dev_pRight);
	}
	/*std::vector<std::future<bool>> vecReturn;
	int iThreads = getHandle()->ThreadNum();
	int iStep = (SrcDstImg.rows + iThreads - 1) / iThreads;
	int iRow0, iRow1;
	iRow0 = 0;
	for (int i = 0; i < iThreads; i++)
	{
		iRow1 = std::min(SrcDstImg.rows, iRow0 + iStep);
		vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CBoudarySearch::copyBoundary_gpu, this, &SrcDstImg, &m_vecvecBoundary, iSizeX, iRow0, iRow1)));
		iRow0 += iStep;
	}
	for (int i = 0; i < vecReturn.size(); i++)
	{
		vecReturn[i].get();
	}
	vecReturn.clear();*/

	*dTime = (cvGetTickCount() - *dTime) / (1000 * cvGetTickFrequency());

	return true;
}

bool CBoudarySearch::ExpandBoundary(cv::Mat& SrcDstImg, int iSizeX, double* dTime)
{
	*dTime = cvGetTickCount();
	if (m_vecvecBoundary.size() == 0 || SrcDstImg.rows != m_vecvecBoundary[0].size())
	{
		return false;
	}
	std::vector<std::future<bool>> vecReturn;
	int iThreads = getHandle()->ThreadNum();
	int iStep = (SrcDstImg.rows + iThreads - 1) / iThreads;
	int iRow0, iRow1;
	iRow0 = 0;
	for (int i = 0; i < iThreads; i++)
	{
		iRow1 = std::min(SrcDstImg.rows, iRow0 + iStep);
		vecReturn.push_back(getHandle()->Executor()->commit(std::bind(&CBoudarySearch::copyBoundary, this, &SrcDstImg, &m_vecvecBoundary, iSizeX, iRow0, iRow1)));
		iRow0 += iStep;
	}
	for (int i = 0; i < vecReturn.size(); i++)
	{
		vecReturn[i].get();
	}
	vecReturn.clear();

	*dTime = (cvGetTickCount() - *dTime) / (1000 * cvGetTickFrequency());

	return true;
}

bool CBoudarySearch::copyBoundary(cv::Mat* SrcDstImg, std::vector<std::vector<std::pair<int, int>>>* vecvecBoundary, int iSizeX, int iRow0, int iRow1)
{
	if (vecvecBoundary->size()==0 || (*vecvecBoundary)[0].size()!=SrcDstImg->rows)
	{
		return false;
	}

	int iLength, iDist, iS, iE, iStep = 0;
	for (int i = 0; i < vecvecBoundary->size(); i++)
	{
		iStep = iRow0 * SrcDstImg->step;
		for (int j = iRow0; j < iRow1; j++)
		{
			iDist = m_vecvecBoundary[i][j].second - m_vecvecBoundary[i][j].first + 1;
			iLength = std::min(iDist, iSizeX);
			iS = m_vecvecBoundary[i][j].first - iLength;
			if (iS < 0)
			{
				iS = 0;
				iLength = m_vecvecBoundary[i][j].first;
			}
			memcpy(SrcDstImg->data + iStep + iS, SrcDstImg->data + iStep + m_vecvecBoundary[i][j].first, sizeof(uchar)*iLength);

			iLength = std::min(iDist, iSizeX);
			iE = m_vecvecBoundary[i][j].second + iLength;
			if (iE > SrcDstImg->cols)
			{
				iE = SrcDstImg->cols - 1;
				iLength = iE - m_vecvecBoundary[i][j].second + 1;
			}
			memcpy(SrcDstImg->data + iStep + m_vecvecBoundary[i][j].second, SrcDstImg->data + iStep + (m_vecvecBoundary[i][j].second - iLength), sizeof(uchar)*iLength);

			iStep += SrcDstImg->step;
		}
	}

	return true;
}

bool CBoudarySearch::copyBoundary_gpu(cv::cuda::GpuMat* SrcDstImg, std::vector<std::vector<std::pair<int, int>>>* vecvecBoundary, int iSizeX, int iRow0, int iRow1)
{
	if (vecvecBoundary->size() == 0 || (*vecvecBoundary)[0].size() != SrcDstImg->rows)
	{
		return false;
	}

	int iLength, iDist, iS, iE, iStep = 0;
	for (int i = 0; i < vecvecBoundary->size(); i++)
	{
		iStep = iRow0 * SrcDstImg->step;
		for (int j = iRow0; j < iRow1; j++)
		{
			iDist = m_vecvecBoundary[i][j].second - m_vecvecBoundary[i][j].first + 1;
			iLength = std::min(iDist, iSizeX);
			iS = m_vecvecBoundary[i][j].first - iLength;
			if (iS < 0)
			{
				iS = 0;
				iLength = m_vecvecBoundary[i][j].first;
			}
			cudaMemcpy(SrcDstImg->data + iStep + iS, SrcDstImg->data + iStep + m_vecvecBoundary[i][j].first, sizeof(uchar)*iLength, cudaMemcpyDeviceToDevice);
			//memcpy(SrcDstImg->data + iStep + iS, SrcDstImg->data + iStep + m_vecvecBoundary[i][j].first, sizeof(uchar)*iLength);

			iE = m_vecvecBoundary[i][j].second + iLength;
			if (iE > SrcDstImg->cols)
			{
				iE = SrcDstImg->cols - 1;
				iLength = iE - m_vecvecBoundary[i][j].second + 1;
			}
			cudaMemcpy(SrcDstImg->data + iStep + m_vecvecBoundary[i][j].second, SrcDstImg->data + iStep + (m_vecvecBoundary[i][j].second - iLength), sizeof(uchar)*iLength, cudaMemcpyDeviceToDevice);

			iStep += SrcDstImg->step;
		}
	}

	return true;
}

bool CBoudarySearch::ErodeDiffImgBoundary(cv::cuda::GpuMat& DiffImg, cv::cuda::GpuMat& DiffMask, int iOffsetLeft, int iOffsetRight, double* dTime)
{
	*dTime = cvGetTickCount();
	if (DiffImg.size()!=getHandle()->ImageSizePre() ||  DiffImg.size()!=DiffImg.size() || DiffImg.type()!=CV_16S || DiffMask.type()!=CV_8U)
	{
		return false;
	}
	DiffMask.setTo(0x00);
	for (int i = 0; i < m_vecvecBoundary.size(); i++)
	{
		for (int j = 0; j < m_vecvecBoundary[i].size(); j++)
		{
			m_pLeft[j] = m_vecvecBoundary[i][j].first;
			m_pRight[j] = m_vecvecBoundary[i][j].second;

			//printf("left: %d, right: %d\n", m_pLeft[j], m_pRight[j]);
		}
		cudaMemcpy(dev_pLeft, m_pLeft, sizeof(int)*DiffImg.rows, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_pRight, m_pRight, sizeof(int)*DiffImg.rows, cudaMemcpyHostToDevice);
		ErodeDiffImg(DiffImg, DiffMask, iOffsetLeft, iOffsetRight, dev_pLeft, dev_pRight);
	}
	*dTime = (cvGetTickCount() - *dTime) / (1000 * cvGetTickFrequency());

	return true;
}

bool CBoudarySearch::ErodeDiffImgBoundary(cv::Mat& DiffImg, cv::Mat& DiffMask, int iOffsetLeft, int iOffsetRight, double* dTime)
{
	*dTime = cvGetTickCount();
	if (DiffImg.size() != getHandle()->ImageSizePre() || DiffImg.size() != DiffImg.size() || DiffImg.type() != CV_16S || DiffMask.type() != CV_8U)
	{
		return false;
	}
	DiffMask.setTo(0x00);
	int iLeft, iRight, iDist, iS, iE;
	int iThr = iOffsetLeft + iOffsetRight;
	for (int i = 0; i < m_vecvecBoundary.size(); i++)
	{
		for (int j = 0; j < m_vecvecBoundary[i].size(); j++)
		{
			iLeft = m_vecvecBoundary[i][j].first;
			iRight = m_vecvecBoundary[i][j].second;
			iDist = iRight - iLeft + 1;
			if (iThr > iDist)
				continue;

			iE = iLeft + iOffsetLeft;
			iS = iRight - iOffsetRight;
			memset(DiffImg.data+j*DiffImg.step, 0x00, sizeof(short)*iE);
			memset(DiffImg.data + j*DiffImg.step + iS*sizeof(short), 0x00, sizeof(short)*(DiffImg.cols - iS));
		}
	}
	DiffMask = (DiffImg != 0);
	*dTime = (cvGetTickCount() - *dTime) / (1000 * cvGetTickFrequency());

	return true;
}

bool CBoudarySearch::GetBoundaryPix(int* iLeft, int* iRight)
{
	if (m_vecvecBoundary.size()==0)
	{
		return false;
	}
	for (int i = 0; i < m_vecvecBoundary.size(); i++)
	{
		if (m_vecvecBoundary[i].size()!=getHandle()->ImageSizePre().height)
		{
			return false;
		}
	}
	int iLeft1, iLeft2, iRight1, iRight2;
	iLeft1 = iLeft2 = m_vecvecBoundary[0][0].first;
	iRight1 = iRight2 = m_vecvecBoundary[0][0].second;
	for (int i = 1; i < m_vecvecBoundary[0].size(); i++)
	{
		iLeft1 = std::min(iLeft1, m_vecvecBoundary[0][i].first);
		iLeft2 = std::max(iLeft2, m_vecvecBoundary[0][i].first);

		iRight1 = std::min(iRight1, m_vecvecBoundary[0][i].second);
		iRight2 = std::max(iRight2, m_vecvecBoundary[0][i].second);
	}

	*iLeft = int((iLeft2 + iLeft1)*0.5f + 0.5f);
	*iRight = int((iRight2 + iRight1)*0.5f + 0.5f);
	return true;
}

