#include "stdafx.h"
#include "InspectionAlogrithm.h"
#include <mutex>


CInspectionAlogrithm::CInspectionAlogrithm()
{
}


CInspectionAlogrithm::~CInspectionAlogrithm()
{
}


void CInspectionAlogrithm::DeNoise(cv::Mat& SrcImg, cv::Mat& DstImg)
{
	//cv::GaussianBlur(SrcImg, DstImg, cv::Size(3, 3), 0.6, 0.6);
	cv::blur(SrcImg, DstImg, cv::Size(3, 3));
}

bool CInspectionAlogrithm::ImageAdd(cv::Mat& imgFream, cv::Mat& bgdSum_Num, cv::Mat& frdSum_Num, cv::Mat& frdMask, int iFreadIdx/* = 0*/)
{
	if (frdMask.size != imgFream.size)
	{
		return false;
	}
	
	cv::Mat frdNum, bgdNum, frdSum, bgdSum, bgd[2], frd[2];
	if (iFreadIdx==0)
	{
		frdSum = cv::Mat::zeros(imgFream.rows, imgFream.cols, CV_32F);
		bgdSum = cv::Mat::zeros(imgFream.rows, imgFream.cols, CV_32F);
		frdNum = cv::Mat::zeros(imgFream.rows, imgFream.cols, CV_32F);
		bgdNum = cv::Mat::zeros(imgFream.rows, imgFream.cols, CV_32F);
	}
	else
	{
		cv::split(bgdSum_Num, bgd);
		cv::split(frdSum_Num, frd);
		frdSum = frd[0];
		bgdSum = bgd[0];
		frdNum = frd[1];
		bgdNum = bgd[1];
	}

	cv::Mat frd_mask, bgd_mask;
	frd_mask = frdMask;
	bgd_mask = ~frd_mask;

	cv::Mat img32S;
	imgFream.convertTo(img32S, CV_32F);
	cv::add(frdSum, img32S, frdSum, frd_mask);
	cv::add(bgdSum, img32S, bgdSum, bgd_mask);

	frd_mask.convertTo(frd_mask, CV_32F, 1.0/255.0);
	bgd_mask.convertTo(bgd_mask, CV_32F, 1.0/255.0);
	frdNum = frdNum + frd_mask;
	bgdNum = bgdNum + bgd_mask;

	if (iFreadIdx==0)
	{
		frd[0] = frdSum;
		bgd[0] = bgdSum;
		frd[1] = frdNum;
		bgd[1] = bgdNum;
	}
	cv::merge(frd, 2, frdSum_Num);
	cv::merge(bgd, 2, bgdSum_Num);

	return true;
}

void CInspectionAlogrithm::GetFlatFieldParam(cv::Mat& bgdSum_Num, cv::Mat& frdSum_Num, cv::Mat& bgdParam, cv::Mat& frdParam, int iDstFrd/* = 128*/, int iDstBgd/* = 240*/)
{
	cv::Mat bgd[2], frd[2];
	cv::split(bgdSum_Num, bgd);
	cv::split(frdSum_Num, frd);

	/*cv::Mat avgFrd = frd[0] / frd[1];
	cv::Mat avgBgd = bgd[0] / bgd[1];
	cv::Mat dst(bgd[0].rows, bgd[0].cols, CV_32F);
	dst.setTo(float(iDstFrd));
	frdParam = dst / avgFrd;
	dst.setTo(float(iDstBgd));
	bgdParam = dst / avgBgd;*/

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
	cv::Mat frdLineParam = Dst / Sum;

	bgdParam = cv::Mat(bgd[0].rows, bgd[0].cols, CV_32F);
	frdParam = cv::Mat(bgd[0].rows, bgd[0].cols, CV_32F);

	for (int i = 0; i < bgd[0].rows; i++)
	{
		memcpy(bgdParam.data + i*bgdParam.cols*sizeof(float), bgdLineParam.data, sizeof(float)*bgdParam.cols);
		memcpy(frdParam.data + i*frdParam.cols*sizeof(float), frdLineParam.data, sizeof(float)*frdParam.cols);
	}
}

void CInspectionAlogrithm::GetMaxMinModel(cv::Mat& imgFream, cv::Mat& rowMin, cv::Mat& rowMax, int iFreadIdx/* = 0*/)
{
	uchar* pBuff = imgFream.data;
	if (iFreadIdx==0)
	{
		rowMax = cv::Mat::zeros(1, imgFream.cols, CV_8U);
		rowMin = rowMax.clone();
		rowMin.setTo(0xff);
	}
	uchar* pMin = rowMin.data;
	uchar* pMax = rowMax.data;
	int iStep = 0;
	for (int i = 0; i < imgFream.rows; i++)
	{
		for (int j = 0; j < imgFream.cols; j++)
		{
			pMin[j] = std::min(pMin[j], pBuff[iStep + j]);
			pMax[j] = std::max(pMax[j], pBuff[iStep + j]);
		}
		iStep += imgFream.cols;
	}
}

void CInspectionAlogrithm::GetExpVarModel(cv::Mat& imgFream, cv::Mat& rowExp, cv::Mat& rowSigma, int iFreadIdx/* = 0*/)
{
	cv::Mat img32f;
	imgFream.convertTo(img32f, CV_32F);
	if (iFreadIdx == 0)
	{
		rowExp = cv::Mat::zeros(1, imgFream.cols, CV_32F);
		cv::reduce(img32f, rowExp, 0, CV_REDUCE_AVG);
		//imgExp.convertTo(imgExp, CV_32F, 1.0 / (double)imgFream.rows);

		cv::Mat imgsquare = img32f.mul(img32f);
		rowSigma = cv::Mat::zeros(1, imgFream.cols, CV_32F);
		cv::reduce(imgsquare, rowSigma, 0, CV_REDUCE_AVG);
		//imgSigma.convertTo(imgSigma, CV_32F, 1.0 / (double)imgFream.rows);

		rowSigma = rowSigma - rowExp.mul(rowExp);
		cv::sqrt(rowSigma, rowSigma);

		return;
	}
	cv::Mat img32fRow, tempMat1, tempMat2, tempMat3;
	int iLineCount = iFreadIdx * imgFream.rows;
	for (int i = 0; i < imgFream.rows; i++)
	{
		tempMat1 = img32f.rowRange(i, i + 1) - rowExp;
		tempMat2 = tempMat1.mul(tempMat1);
		tempMat2.convertTo(tempMat2, CV_32F, 1.0 / (double)(iLineCount));
		rowSigma.convertTo(tempMat3, CV_32F, double(iLineCount - 2) / double(iLineCount - 1));
		rowSigma = tempMat3 + tempMat2;
		tempMat1.convertTo(tempMat1, CV_32F, 1.0 / (double)(iLineCount));
		rowExp = rowExp + tempMat1;
		iLineCount++;
		cv::sqrt(rowSigma, rowSigma);
	}
}


void CInspectionAlogrithm::SplitImg(cv::Size& imageSize, cv::Size& cellSize, cv::Size& padding, std::vector<cv::Rect>& vecSplitRect, std::vector<cv::Rect>& vecTruthRect)
{
	vecSplitRect.clear();
	vecTruthRect.clear();
	if (cellSize.width > imageSize.width || cellSize.height > imageSize.height)
	{
		vecSplitRect.push_back(cv::Rect(0, 0, imageSize.width, imageSize.height));
		vecTruthRect.push_back(cv::Rect(0, 0, imageSize.width, imageSize.height));
		return;
	}
	int iSplitNum_x = (imageSize.width + cellSize.width - 1) / cellSize.width;
	int iSplitNum_y = (imageSize.height + cellSize.height - 1) / cellSize.height;
	vecSplitRect.resize(iSplitNum_y*iSplitNum_x);
	vecTruthRect.resize(iSplitNum_x*iSplitNum_y);

	int iCount = 0;
	cv::Rect rtPadding, rtTruth;
	for (int i = 0; i < imageSize.height; i += cellSize.height)
	{
		rtPadding.y = i - padding.height + 1;
		if (rtPadding.y < 0)
		{
			rtPadding.y = 0;
		}
		rtPadding.height = i + cellSize.height + padding.height - 1;
		if (rtPadding.height > imageSize.height - 1)
		{
			rtPadding.height = imageSize.height - 1;
		}
		rtPadding.height = rtPadding.height - rtPadding.y + 1;

		rtTruth.y = i;
		rtTruth.height = i + cellSize.height - 1;
		if (rtTruth.height > imageSize.height - 1)
		{
			rtTruth.height = imageSize.height - 1;
		}
		rtTruth.height = rtTruth.height - rtTruth.y + 1;

		for (int j = 0; j < imageSize.width; j += cellSize.width)
		{
			rtPadding.x = j - padding.width + 1;
			if (rtPadding.x < 0)
			{
				rtPadding.x = 0;
			}
			rtPadding.width = j + cellSize.width + padding.width - 1;
			if (rtPadding.width > imageSize.width - 1)
			{
				rtPadding.width = imageSize.width - 1;
			}
			rtPadding.width = rtPadding.width - rtPadding.x + 1;

			rtTruth.x = j;
			rtTruth.width = j + cellSize.width - 1;
			if (rtTruth.width > imageSize.width - 1)
			{
				rtTruth.width = imageSize.width - 1;
			}
			rtTruth.width = rtTruth.width - rtTruth.x + 1;

			vecTruthRect[iCount] = rtTruth;
			vecSplitRect[iCount++] = rtPadding;
		}
	}
}


void CInspectionAlogrithm::GaussBlur_fft(cv::Mat& SrcImg, cv::Mat& DstImg, cv::Size& kernelSize, float fSigmaW /*= 0.0f*/, float fSigmaH /*= 0.0f*/)
{
	//fft
	cv::Mat Kernel = GetGaussKernel(kernelSize, fSigmaW, fSigmaH);

	Conv_FFT32f(SrcImg, DstImg, Kernel);
	DstImg.convertTo(DstImg, CV_8U);
}

void CInspectionAlogrithm::Conv_FFT32f(cv::Mat& SrcImg, cv::Mat& DstImg, cv::Mat& Kernel)
{
	//fft
// 	cv::Mat padded, complexI, dft_img, dft_kernel, c_m, imgC[2];
// 	cv::Mat tempImg = cv::Mat::zeros(SrcImg.rows + Kernel.rows - 1, SrcImg.cols + Kernel.cols - 1, CV_8U);
// 	SrcImg.copyTo(tempImg(cv::Rect(0, 0, SrcImg.cols, SrcImg.rows)));
// 	int m = cv::getOptimalDFTSize(tempImg.rows);
// 	int n = cv::getOptimalDFTSize(tempImg.cols); // on the border add zero values
// 	copyMakeBorder(tempImg, padded, 0, m - tempImg.rows, 0, n - tempImg.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
// 	cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
// 	cv::merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
// 	dft_img = cv::Mat(padded.rows, padded.cols, CV_32FC2);
// 	cv::dft(complexI, dft_img, 0, complexI.rows);
// 
// 	tempImg = cv::Mat::zeros(SrcImg.rows + Kernel.rows - 1, SrcImg.cols + Kernel.cols - 1, CV_32F);
// 	Kernel.copyTo(tempImg(cv::Rect(0, 0, Kernel.cols, Kernel.rows)));
// 	copyMakeBorder(tempImg, padded, 0, m - tempImg.rows, 0, n - tempImg.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
// 	cv::Mat planes_k[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
// 	cv::merge(planes_k, 2, complexI);
// 	dft_kernel = cv::Mat(padded.rows, padded.cols, CV_32FC2);
// 	cv::dft(complexI, dft_kernel, 0, complexI.rows);
// 	cv::mulSpectrums(dft_img, dft_kernel, c_m, cv::DFT_COMPLEX_OUTPUT);
// 	cv::idft(c_m, complexI, cv::DFT_INVERSE + cv::DFT_SCALE, c_m.rows);
// 	cv::split(complexI, imgC);
// 	//imgC[0].convertTo(imgC[0], CV_8U);
// 	DstImg = imgC[0](cv::Rect(Kernel.cols >> 1, Kernel.rows >> 1, SrcImg.cols, SrcImg.rows));

	/*cv::Mat padded, complexI, dft_img, dft_kernel, c_m, imgC[2];//加入了虚部
	int m = cv::getOptimalDFTSize(SrcImg.rows);
	int n = cv::getOptimalDFTSize(SrcImg.cols); // on the border add zero values
	copyMakeBorder(SrcImg, padded, 0, m - SrcImg.rows, 0, n - SrcImg.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
	cv::merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
	dft_img = cv::Mat(padded.rows, padded.cols, CV_32FC2);
	cv::dft(complexI, dft_img, 0, complexI.rows);
	padded.convertTo(padded, CV_32F);
	padded.setTo(0x00);
	Kernel.copyTo(padded(cv::Rect(0, 0, Kernel.cols, Kernel.rows)));
	cv::Mat planes_k[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
	cv::merge(planes_k, 2, complexI);
	dft_kernel = cv::Mat(padded.rows, padded.cols, CV_32FC2);
	cv::dft(complexI, dft_kernel, 0, complexI.rows);
	cv::mulSpectrums(dft_img, dft_kernel, c_m, cv::DFT_COMPLEX_OUTPUT);
	cv::idft(c_m, complexI, cv::DFT_INVERSE + cv::DFT_SCALE, c_m.rows);
	cv::split(complexI, imgC);
	DstImg = cv::Mat::zeros(SrcImg.rows,SrcImg.cols,CV_32F);
	cv::Rect dstRoi((Kernel.cols >> 1), (Kernel.rows >> 1), SrcImg.cols - Kernel.cols, SrcImg.rows - Kernel.rows);
	cv::Rect srcRoi((Kernel.cols - 1), (Kernel.rows - 1), SrcImg.cols - Kernel.cols, SrcImg.rows - Kernel.rows);
	//DstImg = imgC[0](cv::Rect((padded.cols - SrcImg.cols) + (Kernel.cols>>1), (padded.rows - SrcImg.rows)+(Kernel.rows>>1) , SrcImg.cols-Kernel.cols, SrcImg.rows-Kernel.rows));
	imgC[0](srcRoi).copyTo(DstImg(dstRoi));*/

	cv::Mat padded, dft_img, dft_kernel, c_m;//仅使用实部
	int m = cv::getOptimalDFTSize(SrcImg.rows);
	int n = cv::getOptimalDFTSize(SrcImg.cols); // on the border add zero values
	copyMakeBorder(SrcImg, padded, 0, m - SrcImg.rows, 0, n - SrcImg.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	padded.convertTo(padded, CV_32F);
	dft_img = cv::Mat::zeros(padded.rows, padded.cols, CV_32F);
	cv::dft(padded, dft_img, 0, padded.rows);
	padded.setTo(0x00);
	Kernel.copyTo(padded(cv::Rect(0, 0, Kernel.cols, Kernel.rows)));
	dft_kernel = cv::Mat::zeros(padded.rows, padded.cols, CV_32F);
	cv::dft(padded, dft_kernel, 0, padded.rows);
	cv::mulSpectrums(dft_img, dft_kernel, c_m, cv::DFT_REAL_OUTPUT);
	cv::idft(c_m, padded, cv::DFT_INVERSE + cv::DFT_SCALE, c_m.rows);
	DstImg = cv::Mat::zeros(SrcImg.rows, SrcImg.cols, CV_32F);
	cv::Rect dstRoi((Kernel.cols >> 1), (Kernel.rows >> 1), SrcImg.cols - Kernel.cols, SrcImg.rows - Kernel.rows);
	cv::Rect srcRoi((Kernel.cols - 1), (Kernel.rows - 1), SrcImg.cols - Kernel.cols, SrcImg.rows - Kernel.rows);
	//DstImg = imgC[0](cv::Rect((padded.cols - SrcImg.cols) + (Kernel.cols>>1), (padded.rows - SrcImg.rows)+(Kernel.rows>>1) , SrcImg.cols-Kernel.cols, SrcImg.rows-Kernel.rows));
	padded(srcRoi).copyTo(DstImg(dstRoi));
}

cv::Mat CInspectionAlogrithm::GetGaussKernel(cv::Size& kernelSize, float fSigmaW /*= 0.0f*/, float fSigmaH /*= 0.0f*/)
{
	cv::Mat kernel_w = cv::getGaussianKernel(kernelSize.width, fSigmaW, CV_32F);
	cv::transpose(kernel_w, kernel_w);
	cv::Mat kernel_h = cv::getGaussianKernel(kernelSize.height, fSigmaH, CV_32F);
	cv::Mat Kernel = kernel_h * kernel_w;
	return Kernel.clone();
}

cv::Mat CInspectionAlogrithm::GetDiffKernel1(cv::Size& s1, cv::Size& s2, float fSigma1, float fSigma2)
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

cv::Mat CInspectionAlogrithm::GetDiffKernel3(std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma, cv::Size& refSize, float fRefSigma)
{
	if (vecKernelSize.size()==0)
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

	cv::Mat RefKernel = CInspectionAlogrithm::GetGaussKernel(refSize, fRefSigma, fRefSigma);
	cv::Mat tempAdd = cv::Mat::zeros(maxSize, CV_32F);
	cv::Mat tempRef = tempAdd.clone();
	cv::Mat tempMat = tempAdd.clone();
	RefKernel.convertTo(tempRef(cv::Rect((maxSize.width >> 1) - (RefKernel.cols >> 1), (maxSize.height >> 1) - (RefKernel.rows >> 1), RefKernel.cols, RefKernel.rows)), CV_32F, double(vecKernelSize.size()));
	for (int i = 0; i < vecKernelSize.size(); i++)
	{
		cv::Mat kernel = CInspectionAlogrithm::GetGaussKernel(vecKernelSize[i], vecSigma[i], vecSigma[i]);
		tempMat.setTo(0x00);
		kernel.copyTo(tempMat(cv::Rect((maxSize.width >> 1) - (kernel.cols >> 1), (maxSize.height >> 1) - (kernel.rows >> 1), kernel.cols, kernel.rows)));
		tempAdd = tempAdd + tempMat;
	}
	tempMat = tempAdd - tempRef;
	return tempMat;
}

bool CInspectionAlogrithm::GetFFTParam(cv::Size& imgSize, std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma, std::vector<cv::Mat>& vecDOGKernelFFT, std::vector<cv::Size>& vecDiffKernelSize)
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
		vecDiffKernelSize.push_back(cv::Size(vecKernel[i].cols,vecKernel[i].rows));
	}

	return true;
}

bool CInspectionAlogrithm::DOGCheck4GaussianThread(cv::Mat& img, cv::Rect& rtSplit, cv::Rect& rtTruth, std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma, cv::Mat& vecRlt, int id)
{
	if (vecKernelSize.size() != vecSigma.size() || vecKernelSize.size() == 0 || vecRlt.size!=img.size)
	{
		return false;
	}
// 	if (vecRlt.size!=img.size)
// 	{
// 		vecRlt = cv::Mat::zeros(img.rows, img.cols, CV_16S);
// 	}
	cv::Mat PatchImg = img(rtSplit).clone();

	//std::vector<cv::Mat> vecR;
	//vecR.clear();
	cv::Mat diffRlt = cv::Mat::zeros(PatchImg.rows, PatchImg.cols, CV_32S);

	int iKernelNum = int(vecKernelSize.size() - 1);
	cv::Size RefKernelSize = vecKernelSize[0];
	float fRefSigma = vecSigma[0];
	cv::Mat RefImg, SrcImg32f, rltImg;
	//cv::GaussianBlur(PatchImg, RefImg, RefKernelSize, fRefSigma, fRefSigma);
	cv::GaussianBlur(PatchImg, RefImg, cv::Size(1, RefKernelSize.height), 0, fRefSigma);
	cv::GaussianBlur(RefImg, RefImg, cv::Size(RefKernelSize.width, 1), fRefSigma, 0);
	RefImg.convertTo(RefImg, CV_32S);
	PatchImg.convertTo(SrcImg32f, CV_32S);
	rltImg = SrcImg32f - RefImg;
	//vecR.push_back(std::move(rltImg));
	diffRlt = diffRlt + rltImg;
	if (iKernelNum != 0)
	{
		for (int i = 1; i < vecKernelSize.size(); i++)
		{
			cv::Mat tempMat;
			cv::GaussianBlur(PatchImg, tempMat, vecKernelSize[i], vecSigma[i], vecSigma[i]);
			tempMat.convertTo(tempMat, CV_32S);
			rltImg = tempMat - RefImg;
			//vecR.push_back(std::move(rltImg));
			diffRlt = diffRlt + rltImg;
		}
	}
	diffRlt.convertTo(diffRlt, CV_16S, 1.0 / double(vecKernelSize.size()));
	
	cv::Rect Roi(abs(rtSplit.x - rtTruth.x), abs(rtSplit.y - rtTruth.y), rtTruth.width, rtTruth.height);
	diffRlt(Roi).copyTo(vecRlt(rtTruth));
	/*for (int i = 0; i < vecR.size(); i++)
	{
		vecR[i](Roi).copyTo(vecRlt[i](rtTruth));
	}
	vecR.clear();*/
	//printf("%d  split\n", id);
	return true;
}

bool CInspectionAlogrithm::DOGCheck4FFTThread(cv::Mat& img, cv::Rect& rtSplit, cv::Rect& rtTruth, cv::Size& normSize, std::vector<cv::Mat>& vecKernelFFT, std::vector<cv::Size>& vecDiffKernelSize, 
	std::vector<cv::Mat>& vecRlt, int id)
{
	if (vecKernelFFT.size() != vecDiffKernelSize.size() || vecKernelFFT.size() == 0 || rtSplit.width > normSize.width || rtSplit.height > normSize.height || vecRlt.size() == 0)
	{
		return false;
	}

	cv::Mat PatchImg = img(rtSplit).clone();

	PatchImg.convertTo(PatchImg, CV_32F);
	std::vector<cv::Mat> vecR;
	vecR.clear();
	
	cv::Mat padded = cv::Mat(vecKernelFFT[0].rows,vecKernelFFT[0].cols,CV_32F);
	PatchImg.copyTo(padded(cv::Rect(0,0,PatchImg.cols,PatchImg.rows)));
	cv::Mat dft_img = cv::Mat::zeros(padded.rows, padded.cols, CV_32F);
	cv::dft(padded, dft_img, 0, padded.rows);

	cv::Mat c_m;
	cv::Mat rltImg = cv::Mat::zeros(PatchImg.rows, PatchImg.cols, CV_32F);
	for (int i = 0; i < vecKernelFFT.size(); i++)
	{
		cv::mulSpectrums(dft_img, vecKernelFFT[i], c_m, cv::DFT_REAL_OUTPUT);
		cv::idft(c_m, padded, cv::DFT_INVERSE + cv::DFT_SCALE, c_m.rows);
		cv::Rect dstRoi((vecDiffKernelSize[i].width >> 1), (vecDiffKernelSize[i].height >> 1), PatchImg.cols - vecDiffKernelSize[i].width, PatchImg.rows - vecDiffKernelSize[i].height);
		cv::Rect srcRoi((vecDiffKernelSize[i].width - 1), (vecDiffKernelSize[i].height - 1), PatchImg.cols - vecDiffKernelSize[i].width, PatchImg.rows - vecDiffKernelSize[i].height);
		rltImg.setTo(0x00);
		padded(srcRoi).copyTo(rltImg(dstRoi));
		vecR.push_back(std::move(rltImg));
	}

	cv::Rect Roi(abs(rtSplit.x - rtTruth.x), abs(rtSplit.y - rtTruth.y), rtTruth.width, rtTruth.height);
	for (int i = 0; i < vecR.size(); i++)
	{
		vecR[i](Roi).convertTo(vecRlt[i](rtTruth), CV_16S);
	}
	vecR.clear();
	return true;
}

void CInspectionAlogrithm::Conv_FFT32f_fftw(cv::Mat& SrcImg, cv::Mat& DstImg, cv::Mat& kernel)
{
	double t1 = cvGetTickCount();
	int iRow = SrcImg.rows + kernel.rows - 1;
	int iCol = SrcImg.cols + kernel.cols - 1;

	cv::Mat buffPatch = cv::Mat::zeros(iRow, iCol, CV_64F);
	cv::Mat complex_out_img(iRow, (iCol >> 1) + 1, CV_64FC2);
	cv::Mat complex_out_kernel = complex_out_img.clone();
	cv::Mat out = complex_out_img.clone();
	fftw_plan forwardImg, forwardKernel, backward;


	forwardImg = fftw_plan_dft_r2c_2d(buffPatch.rows, buffPatch.cols, (double*)buffPatch.data, const_cast<fftw_complex*>(reinterpret_cast<fftw_complex*>((double*)complex_out_img.data)), FFTW_MEASURE);
	forwardKernel = fftw_plan_dft_r2c_2d(buffPatch.rows, buffPatch.cols, (double*)buffPatch.data, const_cast<fftw_complex*>(reinterpret_cast<fftw_complex*>((double*)complex_out_kernel.data)), FFTW_MEASURE);
	backward = fftw_plan_dft_c2r_2d(buffPatch.rows, buffPatch.cols, const_cast<fftw_complex*>(reinterpret_cast<fftw_complex*>((double*)out.data)), (double*)buffPatch.data, FFTW_MEASURE);
	

	double t2 = cvGetTickCount();
	SrcImg.convertTo(buffPatch(cv::Rect(0,0,SrcImg.cols,SrcImg.rows)), CV_64F);
	fftw_execute(forwardImg);
	buffPatch.setTo(0x00);
	kernel.convertTo(buffPatch(cv::Rect(0,0 , kernel.cols, kernel.rows)), CV_64F);
	fftw_execute(forwardKernel);

	cv::mulSpectrums(complex_out_img, complex_out_kernel, out, cv::DFT_COMPLEX_OUTPUT);

	fftw_execute(backward);

	buffPatch.convertTo(buffPatch, CV_64F, 1.0 / double(buffPatch.cols*buffPatch.rows));

	int iHalfX = (kernel.cols >> 1);
	int iHalfY = (kernel.rows >> 1);
	DstImg = cv::Mat::zeros(SrcImg.rows, SrcImg.cols, CV_32F);
	cv::Rect SrcRoi1(iHalfX, iHalfY, SrcImg.cols/* - iHalfX*/, SrcImg.rows/* - iHalfY*/);
	cv::Rect DstRoi1(0, 0, SrcImg.cols/* - iHalfX*/, SrcImg.rows/* - iHalfY*/);
	buffPatch(SrcRoi1).convertTo(DstImg(DstRoi1),CV_32F);
	
	fftw_destroy_plan(forwardImg);
	fftw_destroy_plan(forwardKernel);
	fftw_destroy_plan(backward);

	double t3 = cvGetTickCount();
	printf("%f %f %f\n\n", (t2 - t1) / (1000 * cvGetTickFrequency()), (t3 - t2) / (1000 * cvGetTickFrequency()));
}


bool CInspectionAlogrithm::DOGCheck4FFTwThread(cv::Mat& img, cv::Rect& rtSplit, cv::Rect& rtTruth, cv::Size& normSize, std::vector<cv::Mat>& vecKernelFFT, std::vector<cv::Size>& vecDiffKernelSize,
	std::vector<fftwf_plan>& vecForwardImg, std::vector<fftwf_plan>& vecBackward, std::vector<cv::Mat>& vecBuffPatch, std::vector<cv::Mat>& vecBuffComplexImg, std::vector<cv::Mat>& vecBuffComplexBackward,
	std::vector<cv::Mat>& vecRlt, std::map<std::thread::id, int>& threadIndex, int id)
{
	//std::cout << " thread_id is " << std::this_thread::get_id() << std::endl;
	if (vecKernelFFT.size() != vecDiffKernelSize.size() || vecKernelFFT.size() == 0 || rtSplit.width > normSize.width || rtSplit.height > normSize.height || vecRlt.size() == 0 ||
		vecForwardImg.size()==0 || vecBackward.size()==0 || vecBuffPatch.size()==0 || vecBuffComplexImg.size()==0 || vecBuffComplexBackward.size()==0 || threadIndex.size()==0)
	{
		return false;
	}
	//
	std::thread::id thr_id = std::this_thread::get_id();
	if (threadIndex.count(thr_id) == 0)
	{
		return false;
	}
	int idx = threadIndex[thr_id];

	cv::Mat PatchImg = img(rtSplit).clone();

	PatchImg.convertTo(vecBuffPatch[idx](cv::Rect(0,0,PatchImg.cols,PatchImg.rows)), CV_32F);
	fftwf_execute(vecForwardImg[idx]);//buffPatch->buffComplexImg

	std::vector<cv::Mat> vecR;
	vecR.clear();
	cv::Mat DstImg = cv::Mat::zeros(PatchImg.rows, PatchImg.cols, CV_32F);
	//tempMat = cv::Mat::zeros(srcSize.height, srcSize.width, CV_32F);
	for (int i = 0; i < vecKernelFFT.size(); i++)
	{
		cv::mulSpectrums(vecKernelFFT[i], vecBuffComplexImg[idx], vecBuffComplexBackward[idx], cv::DFT_COMPLEX_OUTPUT);
		fftwf_execute(vecBackward[idx]);//buffComplexBackward->buffPatch
		vecBuffPatch[idx].convertTo(vecBuffPatch[idx], CV_32F, 1.0 / double(vecBuffPatch[idx].cols*vecBuffPatch[idx].rows));

		int iHalfX = (vecDiffKernelSize[i].width >> 1);
		int iHalfY = (vecDiffKernelSize[i].height >> 1);
		DstImg.setTo(0x00);
		cv::Rect SrcRoi1(iHalfX, iHalfY, PatchImg.cols/* - iHalfX*/, PatchImg.rows/* - iHalfY*/);
		cv::Rect DstRoi1(0, 0, PatchImg.cols/* - iHalfX*/, PatchImg.rows/* - iHalfY*/);
		vecBuffPatch[idx](SrcRoi1).copyTo(DstImg(DstRoi1));
		vecR.push_back(std::move(DstImg));
	}

	cv::Rect Roi(abs(rtSplit.x - rtTruth.x), abs(rtSplit.y - rtTruth.y), rtTruth.width, rtTruth.height);
	for (int i = 0; i < vecR.size(); i++)
	{
		vecR[i](Roi).convertTo(vecRlt[i](rtTruth), CV_16S);
	}
	vecR.clear();
	return true;
}

bool CInspectionAlogrithm::GetFFTwParam(cv::Size& imgSize, std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma, std::vector<cv::Mat>& vecDOGKernelFFT, std::vector<cv::Size>& vecDiffKernelSize)
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

void CInspectionAlogrithm::InitiaFFtw(/*cv::Size& imgSize,*/ fftwf_plan* forwardImg, fftwf_plan* backward, cv::Mat* buffPatch, cv::Mat* buffComplexImg, cv::Mat* buffComplexMul)
{
	//buffPatch = cv::Mat::zeros(imgSize.height, imgSize.width, CV_32F);
	//buffComplexImg = cv::Mat(imgSize.height, (imgSize.width >> 1) + 1, CV_32FC2);
	//buffComplexMul = buffComplexImg.clone();
	//fftwf_plan forwardKernel;

	*forwardImg = fftwf_plan_dft_r2c_2d(buffPatch->rows, buffPatch->cols, (float*)buffPatch->data, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>((float*)buffComplexImg->data)), FFTW_MEASURE);
	//forwardKernel = fftwf_plan_dft_r2c_2d(buffPatch.rows, buffPatch.cols, (float*)buffPatch.data, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>((float*)out.data)), FFTW_MEASURE);
	*backward = fftwf_plan_dft_c2r_2d(buffPatch->rows, buffPatch->cols, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>((float*)buffComplexMul->data)), (float*)buffPatch->data, FFTW_MEASURE);

}

cv::Mat CInspectionAlogrithm::GetDiffKernel2(cv::Size& s1, cv::Size& s2, cv::Size& maxSize, float fSigma1, float fSigma2)
{
	if (s1.width > maxSize.width || s1.height > maxSize.height || s2.width > maxSize.width || s2.height > maxSize.height)
	{
		return cv::Mat();
	}
	cv::Mat temp_k1 = cv::Mat::zeros(maxSize.height, maxSize.width, CV_32F);//temp_k3_1 = zeros(kh3, kw3);
	cv::Mat temp_k2 = temp_k1.clone();// = zeros(kh3, kw3);
	cv::Mat kernel1 = CInspectionAlogrithm::GetGaussKernel(s1, fSigma1, fSigma1);
	cv::Mat kernel2 = CInspectionAlogrithm::GetGaussKernel(s2, fSigma2, fSigma2);
	kernel1.copyTo(temp_k1(cv::Rect((maxSize.width >> 1) - (kernel1.cols >> 1), (maxSize.height >> 1) - (kernel1.rows >> 1), kernel1.cols, kernel1.rows)));
	kernel2.copyTo(temp_k2(cv::Rect((maxSize.width >> 1) - (kernel2.cols >> 1), (maxSize.height >> 1) - (kernel2.rows >> 1), kernel2.cols, kernel2.rows)));
// 	kernel1.copyTo(temp_k1(cv::Rect(s2.width >> 1, s2.height >> 1, kernel1.cols, kernel1.rows)));
// 	kernel2.copyTo(temp_k2(cv::Rect(s1.width >> 1, s1.height >> 1, kernel2.cols, kernel2.rows)));
	cv::Mat Kernel3 = temp_k1 - temp_k2;
	return Kernel3;
}

cv::Size CInspectionAlogrithm::GetMaxSize(std::vector<cv::Size>& vecKernelSize)
{
	cv::Size refSize(0, 0);
	for (int i = 0; i < vecKernelSize.size(); i++)
	{
		refSize.width = std::max(refSize.width, vecKernelSize[i].width);
		refSize.height = std::max(refSize.height, vecKernelSize[i].height);
	}
	return refSize;
}

bool CInspectionAlogrithm::MaxMinCheckThread(cv::Mat& img, cv::Rect& rtSplit, cv::Rect& rtTruth, cv::Mat& imgMin, cv::Mat& imgMax, cv::Mat& DiffImg, int id)
{
	if (DiffImg.cols != img.cols || DiffImg.type() != CV_16S || DiffImg.rows != img.rows || img.size != imgMin.size || img.size != imgMax.size)
	{
		return false;
	}
	cv::Mat PatchImg, tempMin, tempMax;
	img(rtSplit).convertTo(PatchImg, CV_16S);
	imgMin(rtSplit).convertTo(tempMin, CV_16S);
	imgMax(rtSplit).convertTo(tempMax, CV_16S);
	cv::Mat diffMat_min = PatchImg - tempMin;
	cv::Mat diffMat_max = PatchImg - tempMax;
	cv::Mat mask_min = (diffMat_min < 0);
	cv::Mat mask_max = (diffMat_max > 0);
	PatchImg.setTo(0x00);
	diffMat_max.copyTo(PatchImg, mask_max);
	diffMat_min.copyTo(PatchImg, mask_min);
	cv::Rect Roi(abs(rtSplit.x - rtTruth.x), abs(rtSplit.y - rtTruth.y), rtTruth.width, rtTruth.height);
	PatchImg.copyTo(DiffImg(rtTruth));
	return true;
}

void CInspectionAlogrithm::ExtendLineToFream(cv::Size& imgSize, cv::Mat& rowImg, cv::Mat& FreamImg)
{
	FreamImg = cv::Mat();
	if (imgSize.width!=rowImg.cols || rowImg.rows!=1)
	{
		return;
	}
	FreamImg = cv::Mat(imgSize, rowImg.type());
	for (int i = 0; i < imgSize.height; i++)
	{
		rowImg.copyTo(FreamImg.rowRange(i,i+1));
	}
}

/*bool CInspectionAlogrithm::ExpStdCheckThread(cv::Mat& img, cv::Rect& rtSplit, cv::Rect& rtTruth, cv::Mat& imgExp, cv::Mat& imgStd, float fSigmaTime, cv::Mat& DiffImg, int id)
{
	if (DiffImg.cols != img.cols || DiffImg.type() != CV_16S || DiffImg.rows != img.rows || img.size != imgExp.size || img.size != imgStd.size || imgExp.type() != CV_32F ||
		imgStd.type()!=CV_32F)
	{
		return false;
	}
	cv::Mat PatchImg, tempMin, tempMax, tempMat;
	img(rtSplit).convertTo(PatchImg, CV_16S);
	imgStd(rtSplit).convertTo(tempMat, CV_32F, fSigmaTime);
	tempMin = imgExp(rtSplit) - tempMat;
	tempMax = imgExp(rtSplit) + tempMax;
	tempMin.convertTo(tempMin, CV_16S);
	tempMax.convertTo(tempMax, CV_16S);

	cv::Mat diffMat_min = PatchImg - tempMin;
	cv::Mat diffMat_max = PatchImg - tempMax;
	cv::Mat mask_min = (diffMat_min < 0);
	cv::Mat mask_max = (diffMat_max > 0);
	PatchImg.setTo(0x00);
	diffMat_max.copyTo(PatchImg, mask_max);
	diffMat_min.copyTo(PatchImg, mask_min);
	cv::Rect Roi(abs(rtSplit.x - rtTruth.x), abs(rtSplit.y - rtTruth.y), rtTruth.width, rtTruth.height);
	PatchImg.copyTo(DiffImg(rtTruth));

	return true;
}*/

bool CInspectionAlogrithm::BoundarySearch(cv::Mat& SrcImg, std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>>& vecBoudaryParam, std::vector<std::vector<std::pair<int, int>>>& vecvecBoundary)
{
	if (vecBoudaryParam.size() == 0)
	{
		int iLeft1, iLeft2, iRight1, iRight2;
		if (vecvecBoundary.size()==0)
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
			cv::Mat PatchImg = SrcImg(topRect).clone();
			CInspectionAlogrithm::GetPatchImgBoundary(PatchImg, 60, 190, left[0], right[0]);
			PatchImg = SrcImg(bottomRect).clone();
			CInspectionAlogrithm::GetPatchImgBoundary(PatchImg, 60, 190, left[1], right[1]);


			iLeft1 = std::max(0, std::min(left[0], left[1]) - 100);
			iLeft2 = std::min(SrcImg.cols - 1, std::max(left[0], left[1]) + 100);
			iRight1 = std::max(0, std::min(right[0], right[1]) - 100);
			iRight2 = std::min(SrcImg.cols - 1, std::max(right[0], right[1]) + 100);

			vecvecBoundary.resize(1);
			vecvecBoundary[0].resize(SrcImg.rows);
		}
		else
		{
			if (vecvecBoundary[0].size()!=SrcImg.rows)
			{
				return false;
			}
			iLeft1 = iLeft2 = vecvecBoundary[0][0].first;
			iRight1 = iRight2 = vecvecBoundary[0][0].second;
			for (int i = 1; i < vecvecBoundary[0].size(); i++)
			{
				iLeft1 = std::min(iLeft1, vecvecBoundary[0][0].first);
				iLeft2 = std::max(iLeft2, vecvecBoundary[0][0].first);

				iRight1 = std::min(iRight1, vecvecBoundary[0][0].second);
				iRight2 = std::max(iRight2, vecvecBoundary[0][0].second);
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
		SearchLine(SrcImg, iLeft1, iLeft2, vecLineLeft);
		SearchLine(SrcImg, iRight1, iRight2, vecLineRight);
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
			vecvecBoundary[0][j] = std::make_pair(vecLineLeft[j], vecLineRight[j]);
		}
		vecLineLeft.clear();
		vecLineRight.clear();
	}
	else
	{
		for (int i = 0; i < vecBoudaryParam.size(); i++)
		{
			if (vecBoudaryParam[i].first.first < 0 || vecBoudaryParam[i].first.second>SrcImg.cols - 1 || vecBoudaryParam[i].second.first < 0 || vecBoudaryParam[i].second.second>SrcImg.cols - 1)
			{
				return false;
			}
		}

		for (int i = 0; i < vecvecBoundary.size(); i++)
		{
			vecvecBoundary[i].clear();
		}
		vecvecBoundary.clear();

		int iMatchBoundaryNum = int(vecBoudaryParam.size());
		vecvecBoundary.resize(iMatchBoundaryNum);
		for (int i = 0; i < iMatchBoundaryNum; i++)
		{
			int iLeft1 = vecBoudaryParam[i].first.first;
			int iLeft2 = vecBoudaryParam[i].first.second;
			int iRight1 = vecBoudaryParam[i].second.first;
			int iRight2 = vecBoudaryParam[i].second.second;
			std::vector<int> vecLineLeft, vecLineRight;
			SearchLine(SrcImg, iLeft1, iLeft2, vecLineLeft);
			SearchLine(SrcImg, iRight1, iRight2, vecLineRight);
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
				vecvecBoundary[i].push_back(std::make_pair(vecLineLeft[j], vecLineRight[j]));
			}
			vecLineLeft.clear();
			vecLineRight.clear();
		}
	}
	
	
	return true;
}

void CInspectionAlogrithm::SearchLine(cv::Mat& img, int& iS, int& iE, std::vector<int>& vecLine)
{
	cv::Mat PatchImg = img(cv::Rect(iS, 0, iE - iS + 1, img.rows)).clone();
	vecLine.resize(PatchImg.rows);
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
				if (pc[j+k]>pn[j])
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
			vecLine[grad_x.rows - 1] = x;
			iMax = pc[x];
		}
	}

	pr = (int*)record.data;
	for (int h = grad_x.rows - 1; --h >= 0;)
	{
		vecLine[h] = pr[h * grad_x.cols + vecLine[h + 1]];
	}
	for (int i = 0; i < vecLine.size(); i++)
	{
		vecLine[i] += iS;
	}
}

bool CInspectionAlogrithm::GetFFTwParam3(cv::Size& imgSize, std::vector<cv::Size>& vecKernelSize, std::vector<float>& vecSigma, cv::Mat& DOGKernelFFT, cv::Size& diffKernelSize)
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
	cv::Mat diffKernel = CInspectionAlogrithm::GetDiffKernel3(tempKernel, tempSigma, RefKernelSize, fRefSigma);
	tempKernel.clear();
	tempSigma.clear();

	diffKernelSize.width = diffKernel.cols;
	diffKernelSize.height = diffKernel.rows;
	int iFFTWRow = imgSize.height + diffKernelSize.height - 1;
	int iFFTWCol = imgSize.width + diffKernelSize.width - 1;
	cv::Mat buffPatch = cv::Mat::zeros(iFFTWRow, iFFTWCol, CV_32F);
	DOGKernelFFT = cv::Mat(iFFTWRow, (iFFTWCol >> 1) + 1, CV_32FC2);
	fftwf_plan forwardKernel;

	forwardKernel = fftwf_plan_dft_r2c_2d(buffPatch.rows, buffPatch.cols, (float*)buffPatch.data, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>((float*)DOGKernelFFT.data)), FFTW_MEASURE);
	buffPatch.setTo(0x00);
	diffKernel.copyTo(buffPatch(cv::Rect(0, 0, diffKernel.cols, diffKernel.rows)));
	fftwf_execute(forwardKernel);//buffPatch->out
	fftwf_destroy_plan(forwardKernel);

	return true;
}

bool CInspectionAlogrithm::DOGCheck4FFTwThread3(cv::Mat& img, cv::Rect& rtSplit, cv::Rect& rtTruth, cv::Size& normSize, int iDiffKernelNum, cv::Mat& KernelFFT, cv::Size& DiffKernelSize, std::vector<fftwf_plan>& vecForwardImg, std::vector<fftwf_plan>& vecBackward, std::vector<cv::Mat>& vecBuffPatch, std::vector<cv::Mat>& vecBuffComplexImg, std::vector<cv::Mat>& vecBuffComplexBackward, cv::Mat& vecRlt, std::map<std::thread::id, int>& threadIndex, int id)
{
	//std::cout << " thread_id is " << std::this_thread::get_id() << std::endl;
	if (rtSplit.width > normSize.width || rtSplit.height > normSize.height ||
		vecForwardImg.size() == 0 || vecBackward.size() == 0 || vecBuffPatch.size() == 0 || vecBuffComplexImg.size() == 0 || vecBuffComplexBackward.size() == 0 || threadIndex.size() == 0)
	{
		return false;
	}
	//
	std::thread::id thr_id = std::this_thread::get_id();
	if (threadIndex.count(thr_id) == 0)
	{
		return false;
	}
	int idx = threadIndex[thr_id];

	cv::Mat PatchImg = img(rtSplit).clone();

	PatchImg.convertTo(vecBuffPatch[idx](cv::Rect(0, 0, PatchImg.cols, PatchImg.rows)), CV_32F);
	fftwf_execute(vecForwardImg[idx]);//buffPatch->buffComplexImg

	cv::Mat DstImg = cv::Mat::zeros(PatchImg.rows, PatchImg.cols, CV_32F);
	cv::mulSpectrums(KernelFFT, vecBuffComplexImg[idx], vecBuffComplexBackward[idx], cv::DFT_COMPLEX_OUTPUT);
	fftwf_execute(vecBackward[idx]);//buffComplexBackward->buffPatch
	vecBuffPatch[idx].convertTo(vecBuffPatch[idx], CV_32F, 1.0 / double(vecBuffPatch[idx].cols*vecBuffPatch[idx].rows));
	int iHalfX = (DiffKernelSize.width >> 1);
	int iHalfY = (DiffKernelSize.height >> 1);
	cv::Rect SrcRoi1(iHalfX, iHalfY, PatchImg.cols/* - iHalfX*/, PatchImg.rows/* - iHalfY*/);
	cv::Rect DstRoi1(0, 0, PatchImg.cols/* - iHalfX*/, PatchImg.rows/* - iHalfY*/);
	vecBuffPatch[idx](SrcRoi1).copyTo(DstImg(DstRoi1));
	cv::Rect Roi(abs(rtSplit.x - rtTruth.x), abs(rtSplit.y - rtTruth.y), rtTruth.width, rtTruth.height);
	DstImg(Roi).convertTo(vecRlt(rtTruth), CV_16S, 1.0 / (double)iDiffKernelNum);
	return true;
}

bool CInspectionAlogrithm::GetBlobThread(cv::Mat& binaryImg, cv::Rect& RoiRt, std::vector<std::vector<cv::Point>>* vecvecContour1, std::vector<std::vector<cv::Point>>* vecvecContour2, std::mutex* mtx1, std::mutex* mtx2)
{
	if (RoiRt.x<0 || RoiRt.x + RoiRt.width>binaryImg.cols || RoiRt.y<0 || RoiRt.y+RoiRt.height>binaryImg.rows)
	{
		return false;
	}
	cv::Mat maskImg = binaryImg(RoiRt).clone();

	/*for (int i = 0; i < vecvecContour1.size(); i++)
	{
		vecvecContour1[i].clear();
	}
	vecvecContour1.clear();

	for (int i = 0; i < vecvecContour2.size(); i++)
	{
		vecvecContour2[i].clear();
	}
	vecvecContour2.clear();*/

	std::vector<std::vector<cv::Point>> vecvecContours;
	cv::findContours(maskImg, vecvecContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	if (vecvecContours.size()==0)
	{
		return true;
	}

	for (auto&& c : vecvecContours)
	{
		cv::Rect boundRect = cv::boundingRect(c);
		for (int k = 0; k < c.size(); k++)
		{
			c[k].x += RoiRt.x;
			c[k].y += RoiRt.y;
		}
		if (boundRect.x > 0 && boundRect.y>0 && boundRect.x + boundRect.width < maskImg.cols && boundRect.y + boundRect.height < maskImg.rows)
		{
			std::unique_lock<std::mutex> lck(*mtx1);
			vecvecContour1->push_back(c);
		}
		else
		{
			std::unique_lock<std::mutex> lck(*mtx2);
			vecvecContour2->push_back(c);
		}
	}

	for (int i = 0; i < vecvecContours.size(); i++)
	{
		vecvecContours[i].clear();
	}
	vecvecContours.clear();

	return true;
}

void CInspectionAlogrithm::MergeContour(std::vector<std::vector<cv::Point>>& vecTempContour, std::vector<std::vector<cv::Point>>& vecvecContour)
{
	int iNum = int(vecTempContour.size());
	if (iNum==0)
	{
		return;
	}

	
	std::vector<uchar> vecMark(iNum, 0);
	for (int i = 0; i < iNum; i++)
	{
		if (vecMark[i]==1)
		{
			continue;;
		}
		std::vector<cv::Point> vecTemp = vecTempContour[i];
		vecMark[i] = 1;
		for (int j = i + 1; j < iNum; j++)
		{
			float fDist = GetContourDist(vecTemp, vecTempContour[j]);
			if (fDist < 3.0f)
			{
				vecTemp.insert(vecTemp.end(), vecTempContour[j].begin(), vecTempContour[j].end());
				vecMark[j] = 1;
			}
		}
		vecvecContour.push_back(vecTemp);
		vecTemp.clear();
	}
	vecMark.clear();
}

float CInspectionAlogrithm::GetContourDist(std::vector<cv::Point>& c1, std::vector<cv::Point>& c2)
{
	float fMin = FLT_MAX;
	for (auto&& p1 : c1)
	{
		for (auto&& p2 : c2)
		{
			cv::Point p = p1 - p2;
			float fV = p.x*p.x + p.y*p.y;
			fMin = std::min(fMin, fV);
			if (fMin<9.0f)
			{
				return sqrt(fMin);
			}
		}
	}
	return sqrt(fMin);
}

bool CInspectionAlogrithm::Geo_BlobToDefectThread(cv::Mat& diffImg, std::vector<cv::Point>& contour, int iBlobThreshold, std::vector<GeoClassifyModel>& vecGeoClassifyParam,
	std::vector<std::pair<int, DefectData>>* vecDefectRect, std::mutex* mtx, int& contourIdx)
{
	if (vecGeoClassifyParam.size()==0)
	{
		return false;
	}
	if (contour.size() < iBlobThreshold)
	{
		return true;
	}

	cv::Rect rt = cv::boundingRect(contour);
	double dArea = fabs(cv::contourArea(contour));
	cv::Mat mask = cv::Mat::zeros(rt.height, rt.width, CV_8U);
	std::vector<std::vector<cv::Point>> tempContour(1);
	tempContour[0].resize(contour.size());
	for (int i = 0; i < contour.size(); i++)
	{
		tempContour[0][i] = contour[i] - cv::Point(rt.x, rt.y);
	}

	cv::fillPoly(mask, tempContour, cv::Scalar::all(0xff));
	tempContour[0].clear();
	tempContour.clear();

	cv::Scalar avg, standar;
	cv::meanStdDev(diffImg(rt), avg, standar, mask);
	
	std::pair<int, DefectData> defectInfo;
	defectInfo.first = contourIdx;
	defectInfo.second.fPy_x = rt.x;
	defectInfo.second.fPy_y = rt.y;
	defectInfo.second.iBlobSize = int(dArea);
	defectInfo.second.fPyArea = float(dArea);
	defectInfo.second.imgRect = rt;
	defectInfo.second.iDefectType = -1;
	defectInfo.second.iMeanDiff = (int)(avg.val[0]);
	for (int i = 0; i < vecGeoClassifyParam.size(); i++)
	{
		if (rt.height < vecGeoClassifyParam[i].fMinHeight || rt.height >  vecGeoClassifyParam[i].fMaxHeight)
		{
			continue;
		}

		if (rt.width <  vecGeoClassifyParam[i].fMinWidth || rt.width >  vecGeoClassifyParam[i].fMaxWidth)
		{
			continue;
		}
	
		if (avg.val[0] < vecGeoClassifyParam[i].iMinDiff || avg.val[0] >  vecGeoClassifyParam[i].iMaxDiff)
		{
			continue;
		}
	
		if (rt.width*rt.height <  vecGeoClassifyParam[i].fMinArea || rt.width*rt.height >  vecGeoClassifyParam[i].fMaxArea)
		{
			continue;
		}
		defectInfo.second.iDefectType = vecGeoClassifyParam[i].iDefectType;
		break;
	}

	{
		std::unique_lock<std::mutex> lck(*mtx);
		vecDefectRect->push_back(defectInfo);
	}


	return true;
}

cv::Mat CInspectionAlogrithm::GetHist(cv::Mat& img)
{
	uchar* pData = img.data;
	cv::Mat hist = cv::Mat::zeros(1, 256, CV_32S);
	int* pHist = (int*)hist.data;
	for (int i = 0; i < img.cols*img.rows; i++)
	{
		pHist[pData[i]]++;
	}
	return hist;
}

void CInspectionAlogrithm::GetPatchImgBoundary(cv::Mat& PatchImg, int iThr1, int iThr2, int& iLeft, int& iRight)
{
	cv::Mat b1, b2;
	cv::threshold(PatchImg, b1, iThr1, 0xff, cv::THRESH_BINARY);
	cv::threshold(PatchImg, b2, iThr2, 0xff, cv::THRESH_BINARY_INV);
	b1 = b1 & b2;
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(50, 1));
	cv::erode(b1, b1, kernel);
	cv::dilate(b1, b1, kernel);

	int iCount;
	iLeft = iRight = iCount = 0;
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
		iLeft += x1;
		iRight += x2;
		iCount++;
	}
	iLeft /= iCount;
	iRight /= iCount;
}

bool CInspectionAlogrithm::GetFrdMask(cv::Size& imgSize, std::vector<std::vector<std::pair<int, int>>>& vecvecBoundary, cv::Mat& frd)
{
	if (vecvecBoundary.size()==0)
	{
		return false;
	}

	for (int i = 0; i < vecvecBoundary.size(); i++)
	{
		if (vecvecBoundary[i].size()!=imgSize.height)
		{
			return false;
		}
	}

	frd = cv::Mat::zeros(imgSize.height, imgSize.width, CV_8U);

	for (int i = 0; i < vecvecBoundary.size(); i++)
	{
		for (int j = 0; j < vecvecBoundary[i].size(); j++)
		{
			memset(frd.data + j*frd.cols + vecvecBoundary[i][j].first, 0xff, sizeof(uchar)*(vecvecBoundary[i][j].second - vecvecBoundary[i][j].first + 1));
		}
	}

	return true;
}


bool CInspectionAlogrithm::FlatField(cv::Mat& SrcDstImg, cv::Mat& bgdParam, cv::Mat& frdParam, cv::Mat& frdMask)
{
	if (frdMask.size!=SrcDstImg.size)
	{
		return false;
	}


	cv::Mat temp32F, dst32f;
	SrcDstImg.convertTo(temp32F, CV_32F);
	dst32f = temp32F.mul(frdParam);
	dst32f.setTo(0x00, ~frdMask);
	temp32F = temp32F.mul(bgdParam);
	temp32F.copyTo(dst32f, ~frdMask);

	dst32f.convertTo(SrcDstImg, CV_8U);

	return true;
}

