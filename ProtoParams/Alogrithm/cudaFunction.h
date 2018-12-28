#pragma once
#include <cuda_runtime_api.h>
#include "cv.hpp"
#include "highgui.hpp"
#include <cudafilters.hpp>
#include <cudaarithm.hpp>


//flatfield
bool GetAvgMask_gpu(cv::cuda::GpuMat* src, cv::cuda::GpuMat* mask, cv::cuda::GpuMat* lineSrcFrd, cv::cuda::GpuMat* lineMaskFrd, cv::cuda::GpuMat* lineSrcBgd, cv::cuda::GpuMat* lineMaskBgd);
bool FlatField_gpu(cv::cuda::GpuMat* SrcImg, cv::cuda::GpuMat* DstImg, cv::cuda::GpuMat* frdParam, cv::cuda::GpuMat* frdMask, cv::Rect roi, double* dTime, int iDstBgd = -1);
//
void DiffFilter(cv::cuda::GpuMat* DstImg32F, cv::cuda::GpuMat* DstImg16S, int iDarkThr, int iLightThr, cudaStream_t* devStream);
//
void CopyBoundary(cv::cuda::GpuMat& SrcDstImg, int iSizeX, int* pLeft, int* pRight);

void ErodeDiffImg(cv::cuda::GpuMat& DiffImg, cv::cuda::GpuMat& DiffMask, int iOffsetLeft, int iOffsetRight, int* pLeft, int* pRight);

void PaddingOffset(cv::cuda::GpuMat& SrcDstImg, cv::Rect& rtOffset);

float TuneImgSelf_gpu(cv::cuda::GpuMat& SrcDstImg, cv::cuda::GpuMat& frdMask, cv::cuda::GpuMat& frdParam, cv::cuda::GpuMat& targetFrd, int iRangeHeight, int iBgdGray);

float TuneImgSelf_gpu(cv::cuda::GpuMat& SrcDstImg, cv::cuda::GpuMat& TempImg, cv::cuda::GpuMat& frdMask, int iOffsetHeight, int iFrdGray, int iBgdGray);

float CheckMeanFilter(cv::cuda::GpuMat& SrcImg, cv::cuda::GpuMat& frdMask, cv::cuda::GpuMat& diff16, int iOffsetHeight, int iLeft, int iRight, int iKernelWidth, int iDarkThr, int iLightThr);


