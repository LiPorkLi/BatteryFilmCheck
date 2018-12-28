#include <assert.h>

#include "cudaFunction.h"
#include <device_atomic_functions.h>
#include "helper_timer.h"
#include "helper_cuda.h"
#include "core\cuda_types.hpp"

#define  BLOCK_X_SIZE  (32) //block x size
#define  BLOCK_Y_SIZE  (8)   //block y size

__global__ void GPU_FlatField(unsigned char* pSrc, unsigned char* pDst, unsigned char* pMask, float* pParam, int iRoiWidth, int iRoiHeight, int iImgCols, int iDstBgd)
{
	unsigned int uiBaseX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int uiBaseY = blockIdx.y * blockDim.y + threadIdx.y;

	//unsigned int uiSrcId = uiBaseY * iRoiWidth + uiBaseX;
	unsigned int imgId = uiBaseY * iImgCols + uiBaseX;

	if (uiBaseX < iRoiWidth && uiBaseY < iRoiHeight)
	{
		if (pMask[imgId] == 0)
		{
			pDst[imgId] = iDstBgd > -1 ? iDstBgd : pSrc[imgId];
			return;
		}
		pDst[imgId] = min(255, int(pSrc[imgId] * pParam[uiBaseX] + 0.5f));
	}
}

bool FlatField_gpu(cv::cuda::GpuMat* SrcImg, cv::cuda::GpuMat* DstImg, cv::cuda::GpuMat* frdParam, cv::cuda::GpuMat* frdMask, cv::Rect roi, double* dTime, int iDstBgd/*=-1*/)
{
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	sdkStartTimer(&hTimer);

	dim3 threads, grid;
	if (frdMask->size() != SrcImg->size() || SrcImg->cols != frdParam->cols)
	{
		return false;
	}

	if (roi.x < 0 || roi.y<0 || roi.x + roi.width > SrcImg->cols || roi.y + roi.height > SrcImg->rows)
	{
		return false;
	}

	threads.x = BLOCK_X_SIZE;
	threads.y = BLOCK_Y_SIZE;
	grid.x = (roi.width - 1) / (threads.x) + 1;
	grid.y = (roi.height - 1) / (threads.y) + 1;

	int iOffset = roi.y * SrcImg->cols + roi.x;
	float* pParam = (float*)frdParam->data;
	GPU_FlatField << <grid, threads >> >(SrcImg->data + iOffset, DstImg->data + iOffset, frdMask->data + iOffset, pParam + roi.x, roi.width, roi.height, SrcImg->cols, iDstBgd);

	//cudaDeviceSynchronize();

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);
	*dTime = sdkGetTimerValue(&hTimer);

	return true;
}

__global__ void GPU_GetAvgMask(uchar* pSrc, uchar* pMask, float* pLineSrcFrd, float* pLineMaskFrd, float* pLineSrcBgd, float* pLineMaskBgd, int iSrcWidth, int iSrcHeight)
{
	unsigned int uiBaseX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int uiBaseY = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int uiSrcId = uiBaseY * iSrcWidth + uiBaseX;

	if (uiBaseX < iSrcWidth && uiBaseY < iSrcHeight)
	{
		if (pMask[uiSrcId] == 0xff)//frd
		{
			atomicAdd(&pLineSrcFrd[uiBaseX], float(pSrc[uiSrcId]));
			atomicAdd(&pLineMaskFrd[uiBaseX], 1);
		}
		else//Bgd
		{
			atomicAdd(&pLineSrcBgd[uiBaseX], float(pSrc[uiSrcId]));
			atomicAdd(&pLineMaskBgd[uiBaseX], 1);
		}
	}
	/*__syncthreads();
	if (uiBaseX < iSrcWidth && uiBaseY == 0)
	{
		if (pLineMask[uiBaseX] == 0)
			pLineSrc[uiBaseX] = 0;
		else
			pLineSrc[uiBaseX] /= pLineMask[uiBaseX];
	}*/
}

bool GetAvgMask_gpu(cv::cuda::GpuMat* src, cv::cuda::GpuMat* mask, cv::cuda::GpuMat* lineSrcFrd, cv::cuda::GpuMat* lineMaskFrd, cv::cuda::GpuMat* lineSrcBgd, cv::cuda::GpuMat* lineMaskBgd)
{
	if (src->size() != mask->size())
	{
		return false;
	}

	(*lineSrcFrd) = cv::cuda::GpuMat(1, src->cols, CV_32F);
	lineSrcFrd->setTo(0x00);
	(*lineMaskFrd) = lineSrcFrd->clone();
	(*lineSrcBgd) = lineSrcFrd->clone();
	(*lineMaskBgd) = lineSrcFrd->clone();

	dim3 threads, grid;
	threads.x = BLOCK_X_SIZE;
	threads.y = BLOCK_Y_SIZE;
	grid.x = (src->cols -1) / (threads.x) + 1;
	grid.y = (src->rows - 1) / (threads.y) + 1;

	GPU_GetAvgMask << <grid, threads >> >(src->data, mask->data, (float*)lineSrcFrd->data, (float*)lineMaskFrd->data, (float*)lineSrcBgd->data, (float*)lineMaskBgd->data, src->cols, src->rows);

	cudaDeviceSynchronize();

	return true; 
}

__global__ void GPU_DiffFilter(float* pDst32f, short* pDst16s, int iDstWidth, int iDstHeight, int iDarkThr, int iLightThr)
{
	unsigned int uiBaseX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int uiBaseY = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int uiSrcId = uiBaseY * iDstWidth + uiBaseX;

	if (uiBaseX < iDstWidth && uiBaseY < iDstHeight)
	{
		//pDst32f[uiSrcId] = pDst32f[uiSrcId] / float(iKernelNum);
		if (pDst32f[uiSrcId] > iDarkThr && pDst32f[uiSrcId] < iLightThr)
		{
			pDst16s[uiSrcId] = 0;
			return;
		}
		if (pDst32f[uiSrcId] <= iDarkThr)
		{
			pDst16s[uiSrcId] = short(pDst32f[uiSrcId] - float(iDarkThr));
			return;
		}
		if (pDst32f[uiSrcId] >= iLightThr)
		{
			pDst16s[uiSrcId] = short(pDst32f[uiSrcId] - float(iLightThr));
		}
	}
}


void DiffFilter(cv::cuda::GpuMat* DstImg32F, cv::cuda::GpuMat* DstImg16S, int iDarkThr, int iLightThr, cudaStream_t* devStream)
{
	dim3 threads, grid;
	threads.x = BLOCK_X_SIZE;
	threads.y = BLOCK_Y_SIZE;
	grid.x = (DstImg32F->cols - 1) / (threads.x) + 1;
	grid.y = (DstImg32F->rows - 1) / (threads.y) + 1;

	if (devStream==NULL)
	{
		GPU_DiffFilter << <grid, threads >> >((float*)DstImg32F->data, (short*)DstImg16S->data, DstImg32F->cols, DstImg32F->rows, iDarkThr, iLightThr);
	}
	else
	{
		GPU_DiffFilter << <grid, threads, 0, *devStream >> >((float*)DstImg32F->data, (short*)DstImg16S->data, DstImg32F->cols, DstImg32F->rows, iDarkThr, iLightThr);
	}
	

	cudaDeviceSynchronize();
}

__global__ void GPU_CopyBoundary(cv::cuda::PtrStep<uchar> srcdst, int iWidth,int iHeight, int iSizeX, int* pLeft,int* pRight)
{
	unsigned int uiBaseX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int uiBaseY = blockIdx.y * blockDim.y + threadIdx.y;


	//unsigned int uiSrcId = uiBaseY * iWidth + uiBaseX;

	int iLength, iDist, iLenLeft, iLenRight, iS, iE;

	if (uiBaseY < iHeight)
	{
		iDist = pRight[uiBaseY] - pLeft[uiBaseY] + 1;
		iLength = min(iDist, iSizeX);

		iS = pLeft[uiBaseY] - iLength;
		iLenLeft = iLength;
		if (iS < 0)
		{
			iS = 0;
			iLenLeft = pLeft[uiBaseY];
		}
		iE = pRight[uiBaseY] + iLength;
		iLenRight = iLength;
		if (iE > iWidth)
		{
			iE = iWidth - 1;
			iLenRight = iE - pRight[uiBaseY] + 1;
		}
	}
	if (uiBaseX < iLenLeft && uiBaseY < iHeight)
	{
		srcdst(uiBaseY, iS + uiBaseX) = srcdst(uiBaseY, pLeft[uiBaseY] + iLenLeft - uiBaseX);
	}
	if (uiBaseX < iLenRight && uiBaseY < iHeight)
	{
		srcdst(uiBaseY, iE - uiBaseX) = srcdst(uiBaseY, pRight[uiBaseY] - iLenRight + uiBaseX);
	}
}

void CopyBoundary(cv::cuda::GpuMat& SrcDstImg, int iSizeX, int* pLeft, int* pRight)
{
	dim3 threads, grid;
	threads.x = BLOCK_X_SIZE;
	threads.y = BLOCK_Y_SIZE;
	grid.x = (SrcDstImg.cols - 1) / (threads.x) + 1;
	grid.y = (SrcDstImg.rows - 1) / (threads.y) + 1;

	GPU_CopyBoundary << <grid, threads >> >(SrcDstImg, SrcDstImg.cols, SrcDstImg.rows, iSizeX, pLeft, pRight);

	cudaDeviceSynchronize();
}

__global__ void GPU_ErodeDiffImg(cv::cuda::PtrStep<short> diffImg, cv::cuda::PtrStep<uchar> diffMask, int iWidth, int iHeight, int iOffsetLeft, int iOffsetRight, int* pLeft, int* pRight)
{
	unsigned int uiBaseX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int uiBaseY = blockIdx.y * blockDim.y + threadIdx.y;


	//unsigned int uiSrcId = uiBaseY * iWidth + uiBaseX;

	int iDist = 0, iE, iS;
	int iThr = iOffsetLeft + iOffsetRight;
	if (uiBaseY < iHeight)
	{
		iDist = pRight[uiBaseY] - pLeft[uiBaseY] + 1;
		if (iThr > iDist)
			return;

		iE = pLeft[uiBaseY] + iOffsetLeft;
		iS = pRight[uiBaseY] - iOffsetRight;
	}
	if (uiBaseX < iE && uiBaseY < iHeight)
	{
		diffImg(uiBaseY, uiBaseX) = short(0);
	}
	if (uiBaseX < (iWidth - iS) && uiBaseY < iHeight)
	{
		diffImg(uiBaseY, iS + uiBaseX) = short(0);
	}
	if (uiBaseX < iWidth && uiBaseY < iHeight)
	{
		if (diffImg(uiBaseY, uiBaseX) != 0)
			diffMask(uiBaseY, uiBaseX) = 0xff;
		else
			diffMask(uiBaseY, uiBaseX) = 0;
	}
}

void ErodeDiffImg(cv::cuda::GpuMat& DiffImg, cv::cuda::GpuMat& DiffMask, int iOffsetLeft, int iOffsetRight, int* pLeft, int* pRight)
{
	dim3 threads, grid;
	threads.x = BLOCK_X_SIZE;
	threads.y = BLOCK_Y_SIZE;
	grid.x = (DiffImg.cols - 1) / (threads.x) + 1;
	grid.y = (DiffImg.rows - 1) / (threads.y) + 1;

	GPU_ErodeDiffImg << <grid, threads >> >(DiffImg, DiffMask, DiffImg.cols, DiffImg.rows, iOffsetLeft, iOffsetRight, pLeft, pRight);

	cudaDeviceSynchronize();
}

__global__ void GPU_PaddingOffset(cv::cuda::PtrStep<uchar> SrcDstImg, int iLeftX, int iTopY, int iRoiWidth, int iRoiHeight)
{
	unsigned int uiBaseX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int uiBaseY = blockIdx.y * blockDim.y + threadIdx.y;


	if (uiBaseY < iRoiHeight && uiBaseX < iRoiWidth)
	{
		SrcDstImg(iTopY + uiBaseY, uiBaseX + iLeftX) = SrcDstImg(iTopY - 1 - uiBaseY, uiBaseX + iLeftX);
	}

}

void PaddingOffset(cv::cuda::GpuMat& SrcDstImg, cv::Rect& rtOffset)
{
	dim3 threads, grid;
	threads.x = BLOCK_X_SIZE;
	threads.y = BLOCK_Y_SIZE;
	grid.x = (rtOffset.width - 1) / (threads.x) + 1;
	grid.y = (rtOffset.height - 1) / (threads.y) + 1;

	GPU_PaddingOffset << <grid, threads >> >(SrcDstImg, rtOffset.x, rtOffset.y, rtOffset.width, rtOffset.height);

	cudaDeviceSynchronize();
}

__global__ void GPU_GetTuneParam(cv::cuda::PtrStep<uchar> SrcDstImg, cv::cuda::PtrStep<uchar> frdMask, cv::cuda::PtrStep<float> frdParam, int iRangeHeight, int iHeight, int iWidth)
{
	unsigned int uiBaseX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int uiBaseY = blockIdx.y * blockDim.y + threadIdx.y;

	int iTempY = uiBaseY % 16;
	if (iTempY==0 && uiBaseY < iRangeHeight && uiBaseX < iWidth && frdMask(uiBaseY, uiBaseX) == 0xff)
	{
		int iN = iRangeHeight / 16 + 1;
		float fP = 1.0f / (float)iN;
		//SrcDstImg(iTopY + uiBaseY, uiBaseX + iLeftX) = SrcDstImg(iTopY - 1 - uiBaseY, uiBaseX + iLeftX);
		atomicAdd(&frdParam(0, uiBaseX), float(fP * SrcDstImg(uiBaseY, uiBaseX)));
	}
}

__global__ void GPU_TuneImgSelf(cv::cuda::PtrStep<uchar> SrcDstImg, cv::cuda::PtrStep<uchar> frdMask, cv::cuda::PtrStep<float> frdParam, int iHeight, int iWidth, int iBgdGray)
{
	unsigned int uiBaseX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int uiBaseY = blockIdx.y * blockDim.y + threadIdx.y;


	if (uiBaseY < iHeight && uiBaseX < iWidth)
	{
		if (frdMask(uiBaseY, uiBaseX) == 0x00)
		{
			SrcDstImg(uiBaseY, uiBaseX) = iBgdGray;
			return;
		}
		int iTemp = int(SrcDstImg(uiBaseY, uiBaseX) * frdParam(0, uiBaseX) + 0.5f);
		iTemp = min(255, iTemp);
		SrcDstImg(uiBaseY, uiBaseX) = iTemp;
	}
}

__global__ void GPU_TuneImgSelf_Test(cv::cuda::PtrStep<uchar> srcImg, cv::cuda::PtrStep<uchar> dstImg, cv::cuda::PtrStep<uchar> frdMask, int iHeight, int iWidth, int iBgdGray, int iFrdGray, int iOffsetHeight)
{
	unsigned int uiBaseX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int uiBaseY = blockIdx.y * blockDim.y + threadIdx.y;


	if (uiBaseY < iOffsetHeight && uiBaseX < iWidth)
	{
		if (frdMask(uiBaseY, uiBaseX) == 0x00)
		{
			dstImg(uiBaseY, uiBaseX) = iBgdGray;
			return;
		}
		int iRange = 1025;
		int iRadius = iRange >> 1;
		int iSumGray = 0;
		int iStart = uiBaseY - iRadius;
		int iEnd = iStart + iRange;
		if (iStart < 0)
		{
			iStart = 0;
			iEnd = iStart + iRange;
		}
		if (iEnd > iHeight)
		{
			iEnd = iHeight;
			iStart = iEnd - iRange;
		}
		for (int i = iStart; i < iEnd; i+=32)
			iSumGray += srcImg(i, uiBaseX);
	
		float fAvgGray = (iSumGray) / 33.0;
		float fParam = (float)iFrdGray / fAvgGray;
		int iTemp = int(srcImg(uiBaseY, uiBaseX) * fParam + 0.5f);
		iTemp = min(255, iTemp);
		dstImg(uiBaseY, uiBaseX) = iTemp;
	}
}

__global__ void GPU_TuneImgSelf_Cpoy(cv::cuda::PtrStep<uchar> srcImg, cv::cuda::PtrStep<uchar> dstImg, cv::cuda::PtrStep<uchar> frdMask, int iHeight, int iWidth, int iBgdGray, int iFrdGray, int iOffsetHeight)
{
	unsigned int uiBaseX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int uiBaseY = blockIdx.y * blockDim.y + threadIdx.y;

	if (uiBaseY < iOffsetHeight && uiBaseX < iWidth)
	{
		dstImg(uiBaseY, uiBaseX) = srcImg(uiBaseY, uiBaseX);
	}
	if (uiBaseY >= iOffsetHeight && uiBaseY < iHeight && uiBaseX < iWidth)
	{
		if (frdMask(uiBaseY, uiBaseX) == 0x00)
			dstImg(uiBaseY, uiBaseX) = iBgdGray;
		else
			dstImg(uiBaseY, uiBaseX) = srcImg(iOffsetHeight - (uiBaseY - iOffsetHeight) - 1, uiBaseX);	
	}
}

float TuneImgSelf_gpu(cv::cuda::GpuMat& SrcDstImg, cv::cuda::GpuMat& frdMask, cv::cuda::GpuMat& frdParam, cv::cuda::GpuMat& targetFrd, int iRangeHeight, int iBgdGray)
{
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	sdkStartTimer(&hTimer);

	frdParam.setTo(0x00);

	dim3 threads, grid;
	threads.x = BLOCK_X_SIZE;
	threads.y = BLOCK_Y_SIZE;
	grid.x = (SrcDstImg.cols - 1) / (threads.x) + 1;
	grid.y = (iRangeHeight - 1) / (threads.y) + 1;

	GPU_GetTuneParam << <grid, threads >> >(SrcDstImg, frdMask, frdParam, iRangeHeight, SrcDstImg.rows, SrcDstImg.cols);

	//cv::Mat tempParam,targetTemp;
	//frdParam.download(tempParam);
	//targetFrd.download(targetTemp);
	cv::cuda::divide(targetFrd, frdParam, frdParam);

	grid.y = (SrcDstImg.rows - 1) / (threads.y) + 1;
	GPU_TuneImgSelf << <grid, threads >> >(SrcDstImg, frdMask, frdParam, SrcDstImg.rows, SrcDstImg.cols, iBgdGray);


	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);
	return sdkGetTimerValue(&hTimer);
}


float TuneImgSelf_gpu(cv::cuda::GpuMat& SrcDstImg, cv::cuda::GpuMat& TempImg, cv::cuda::GpuMat& frdMask, int iOffsetHeight, int iFrdGray, int iBgdGray)
{
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	sdkStartTimer(&hTimer);


	dim3 threads, grid;
	threads.x = BLOCK_X_SIZE;
	threads.y = BLOCK_Y_SIZE;
	grid.x = (SrcDstImg.cols - 1) / (threads.x) + 1;
	grid.y = (iOffsetHeight - 1) / (threads.y) + 1;

	GPU_TuneImgSelf_Test << <grid, threads >> >(SrcDstImg, TempImg, frdMask, SrcDstImg.rows, SrcDstImg.cols, iBgdGray, iFrdGray, iOffsetHeight);
	//TempImg.copyTo(SrcDstImg);

	grid.y = (SrcDstImg.rows - 1) / (threads.y) + 1;
	GPU_TuneImgSelf_Cpoy << <grid, threads >> >(TempImg, SrcDstImg, frdMask, SrcDstImg.rows, SrcDstImg.cols, iBgdGray, iFrdGray, iOffsetHeight);

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);
	return sdkGetTimerValue(&hTimer);
}


__global__ void GPU_Check_MeanFilter_(cv::cuda::PtrStep<uchar> srcImg, 
									 cv::cuda::PtrStep<uchar> frdMask, 
									 cv::cuda::PtrStep<short> diff16, 
									 int iHeight, int iWidth, 
									 int iLeft, int iRight, int iOffsetHeight,
									 int iKernelWidth,
									 int iDarkThr, int iLightThr)
{
	unsigned int uiBaseX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int uiBaseY = blockIdx.y * blockDim.y + threadIdx.y;

	if (uiBaseY < iOffsetHeight && uiBaseX < iRight && uiBaseX > iLeft && frdMask(uiBaseY, uiBaseX)!=0x00)
	{
		int iSum = 0;
		int iRadius = iKernelWidth >> 1;

		int iStart_x = uiBaseX - iRadius;
		int iEnd_x = iStart_x + iKernelWidth;
		if (iStart_x < iLeft)
		{
			iStart_x = iLeft;
			iEnd_x = iStart_x + iKernelWidth;
		}
		if (iEnd_x > iRight)
		{
			iEnd_x = iRight;
			iStart_x = iEnd_x - iKernelWidth;
		}

		int iStart_y = uiBaseY - iRadius;
		int iEnd_y = iStart_y + iKernelWidth;
		if (iStart_y < 0)
		{
			iStart_y = 0;
			iEnd_y = iStart_y + iKernelWidth;
		}
		if (iEnd_y > iOffsetHeight)
		{
			iEnd_y = iOffsetHeight;
			iStart_y = iEnd_y - iKernelWidth;
		}

		int iCount = 0;
		for (int i = iStart_y; i < iEnd_y; i += 16)
		{
			for (int j = iStart_x; j < iEnd_x; j += 16)
			{
				iSum += srcImg(i, j);
				iCount++;
			}
		}

		int iAvg = iSum / iCount;
		diff16(uiBaseY, uiBaseX) = srcImg(uiBaseY, uiBaseX) - iAvg;
		
		if (diff16(uiBaseY, uiBaseX) > iDarkThr && diff16(uiBaseY, uiBaseX) < iLightThr)
		{
			frdMask(uiBaseY, uiBaseX) = 0;
			return;
		}
		if (diff16(uiBaseY, uiBaseX) <= iDarkThr)
		{
			diff16(uiBaseY, uiBaseX) = diff16(uiBaseY, uiBaseX) - short(iDarkThr);
			frdMask(uiBaseY, uiBaseX) = 0xff;
			return;
		}
		if (diff16(uiBaseY, uiBaseX) >= iLightThr)
		{
			diff16(uiBaseY, uiBaseX) = diff16(uiBaseY, uiBaseX) - short(iLightThr);
			frdMask(uiBaseY, uiBaseX) = 0xff;
			return;
		}
	}
}


float CheckMeanFilter(cv::cuda::GpuMat& SrcImg, cv::cuda::GpuMat& frdMask, cv::cuda::GpuMat& diff16, int iOffsetHeight, int iLeft, int iRight, int iKernelWidth, int iDarkThr, int iLightThr)
{
	diff16.setTo(0x00);
	//diffMask.setTo(0x00);

	dim3 threads, grid;
	threads.x = BLOCK_X_SIZE;
	threads.y = BLOCK_Y_SIZE;
	grid.x = (SrcImg.cols - 1) / (threads.x) + 1;
	grid.y = (iOffsetHeight - 1) / (threads.y) + 1;

	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	sdkStartTimer(&hTimer);

	GPU_Check_MeanFilter_ << <grid, threads >> >(SrcImg,
		frdMask,
		diff16,
		SrcImg.rows, SrcImg.cols,
		iLeft, iRight, iOffsetHeight,
		iKernelWidth,
		iDarkThr, iLightThr);

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);
	return sdkGetTimerValue(&hTimer);
}