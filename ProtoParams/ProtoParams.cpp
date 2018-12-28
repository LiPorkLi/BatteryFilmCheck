// ProtoParams.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "ParamHelper.h"
#include "proto\param.pb.h"
//#include "safe_queue.h"
#include "GlobalData.h"
#include "proto\checkparam.pb.h"
#include "Alogrithm/InspectProcedure.h"
#include "ImageIO.h"

#ifndef DEBUG
#include "GrabThread.h"
#endif

safe_queue<ImageInspectResult> Global::g_InspectQueue;
MainParam::param Global::g_param;
safe_queue<GrabImgInfo> Global::g_ProcessQueue;
safe_queue<std::string> Global::g_AlogrithmLog;
#ifndef DEBUG
std::shared_ptr<CGrabThread> g_grab = std::make_shared<CGrabThread>();
#endif
std::shared_ptr<CImageIO> g_IO = std::make_shared<CImageIO>();

bool Global::g_isStop;
//std::shared_ptr<CInspectProcedure> Global::g_pInspect = std::make_shared<CInspectProcedure>();

// std::shared_ptr<safe_queue<ImageInspectResult>> Global::g_InspectQueue = std::make_shared<safe_queue<ImageInspectResult>>();
// std::shared_ptr<MainParam::param> Global::g_param = std::make_shared<MainParam::param>();
// std::shared_ptr<safe_queue<GrabImgInfo>> Global::g_grabQueue = std::make_shared<safe_queue<GrabImgInfo>>();
int _tmain1(int argc, _TCHAR* argv[])
{
	//设置检测参数
/*	if (1)
	{
		ParamHelper<Parameters::InspectParam> help_InspectParam(&Global::g_param);
		Parameters::InspectParam* pInspectParam = help_InspectParam.getPtr();
		pInspectParam->set_dstvaluebgd(50);//设置平场校正参数
		pInspectParam->set_dstvaluefrd(128);

		Parameters::BoundSearchParam* bA = pInspectParam->add_boundsearchlist();//设置两组边界搜索参数
		bA->set_leftrange1(10);
		bA->set_leftrange2(20);
		bA->set_rightrange1(1000);
		bA->set_rightrange2(1020);
		Parameters::BoundSearchParam* bB = pInspectParam->add_boundsearchlist();
		bB->set_leftrange1(100);
		bB->set_leftrange2(110);
		bB->set_rightrange1(500);
		bB->set_rightrange2(520);

		pInspectParam->set_checktype(Parameters::InspectParam::DOG);//设置检测算法
		Parameters::DefectFilter *d = pInspectParam->mutable_defectfilter();//设置缺陷过滤参数
		d->set_blobthr(50);
		d->set_boundoffsetleft(20);
		d->set_boundoffsetright(10);
		pInspectParam->set_classificcationtype(Parameters::InspectParam::Geometry);//设置缺陷分类算法为几何分类法
		Parameters::GeoClassifiyParam* g0 = pInspectParam->add_geoparamlist();//添加三组缺陷类别
		g0->set_defectname("dark dot");
		g0->set_mindiff(-255);
		g0->set_maxdiff(-180);
		g0->set_minarea(6.5);
		g0->set_maxarea(12.3);
		g0->set_minheight(5);
		g0->set_maxheight(10);
		g0->set_minwidth(3);
		g0->set_maxwidth(6);

		g0 = pInspectParam->add_geoparamlist();
		g0->set_defectname("light dot");
		g0->set_mindiff(180);
		g0->set_maxdiff(255);
		g0->set_minarea(6.5);
		g0->set_maxarea(12.3);
		g0->set_minheight(5);
		g0->set_maxheight(10);
		g0->set_minwidth(3);
		g0->set_maxwidth(6);

		g0 = pInspectParam->add_geoparamlist();
		g0->set_defectname("gap");
		g0->set_mindiff(180);
		g0->set_maxdiff(255);
		g0->set_minarea(22);
		g0->set_maxarea(99999);
		g0->set_minheight(5);
		g0->set_maxheight(10);
		g0->set_minwidth(60);
		g0->set_maxwidth(16384);

		//pInspectParam->set_classificcationtype(Parameters::InspectParam::SVM);//设置缺陷分类算法为SVM
		Parameters::SVMClassifyParam* s = pInspectParam->mutable_svmparam();//设置SVM算法参数
		s->set_modelpath("d:\\svm.xml");//设置SVM模型文件路径
		if (s->mutable_defectlabelname()->count(0) == 0)//插入三组分类类别
		{
			(*s->mutable_defectlabelname())[0] = "dark dot";
		}
		if (s->mutable_defectlabelname()->count(1) == 0)
		{
			(*s->mutable_defectlabelname())[1] = "light dot";
		}
		if (s->mutable_defectlabelname()->count(2) == 0)
		{
			(*s->mutable_defectlabelname())[2] = "gap";
		}

		pInspectParam->set_savepath("D:\\InspectParam.bin");//设置参数保存路径
		//设置检测参数: 完毕

		//设置工单信息
		ParamHelper<Parameters::SheetInfo> help_SheetInfo(&Global::g_param);
		Parameters::SheetInfo* pInfo = help_SheetInfo.getPtr();

		pInfo->set_username("super user"); // 用户名
		pInfo->set_machinenumber("M-0121"); //机台号
		pInfo->set_imagemodel("back illuminate"); // 成像方式
		pInfo->set_prodctname("tubu-033"); // 产品名
		pInfo->set_ordernumber("20180126-5"); // 订单号
		pInfo->set_batchnumber("AB953468-8-9"); // 批号
		pInfo->set_inspectparamname("d:\\InspectParam.bin"); // 检测参数档
		pInfo->set_remark("This is a demo!"); // 备注
		Parameters::StripParam* pStripInfo = pInfo->mutable_stripparam();
		pStripInfo->set_leftoffset(60.3);//设置分条起始位置
		//设置10条，每条60mm
		for (int i = 0; i < 10; i++)
		{
			pStripInfo->add_stripwidth(60.0);
		}
		float allwidth = 0.0f;
		for (int i = 0; i < pStripInfo->stripwidth_size(); i++)
		{
			allwidth += pStripInfo->stripwidth(i);
		}
		pStripInfo->set_allwidth(allwidth);//总检测宽度
		pStripInfo->set_stripcount(pStripInfo->stripwidth_size());//总条数
		//设置工单信息:完毕
		//help_InspectParam与help_SheetInfo生命周期结束时，自动调用ParamHelper的析构函数，将所有参数写入Global::g_param中
	}

	//读取刚才设置的信息
	if (1)
	{
		printf("This is Inspect param:\n");
		ParamHelper<Parameters::InspectParam> help_InspectParam(&Global::g_param);
		const Parameters::InspectParam pInspectParam = help_InspectParam.getRef();
		printf("	flat field param: %d %d\n", pInspectParam.dstvaluefrd(), pInspectParam.dstvaluebgd());
		printf("	This is boundary search param, num is %d:\n", pInspectParam.boundsearchlist_size());
		for (int i = 0; i < pInspectParam.boundsearchlist_size(); i++)
		{
			printf("		%d param\n", i);
			printf("			left Range 1: %d\n", pInspectParam.boundsearchlist(i).leftrange1());
			printf("			left Range 2: %d\n", pInspectParam.boundsearchlist(i).leftrange2());
			printf("			right Range 1: %d\n", pInspectParam.boundsearchlist(i).rightrange1());
			printf("			right Range 2: %d\n", pInspectParam.boundsearchlist(i).rightrange2());
		}
		printf("	The Inspect Alogrithm is %d\n", pInspectParam.checktype());
//		printf("	The Defect Filter param is: blob=%d, boundary offset=%d\n", pInspectParam.defectfilter().blobthr(), pInspectParam.defectfilter().boundoffset());
		int iClassifiy = pInspectParam.classificcationtype();
		if (iClassifiy==0)
		{
			printf("	The classifiy type is Geometry\n");
			printf("		The classes num is %d \n", pInspectParam.geoparamlist_size());
			for (int i = 0; i < pInspectParam.geoparamlist_size(); i++)
			{
				printf("		%s:\n", pInspectParam.geoparamlist(i).defectname().c_str());
				printf("			min diff: %d\n", pInspectParam.geoparamlist(i).mindiff());
				printf("			max diff: %d\n", pInspectParam.geoparamlist(i).maxdiff());
				printf("			min area: %f\n", pInspectParam.geoparamlist(i).minarea());
				printf("			max area: %f\n", pInspectParam.geoparamlist(i).maxarea());
				printf("			min height: %d\n", pInspectParam.geoparamlist(i).minheight());
				printf("			max height: %d\n", pInspectParam.geoparamlist(i).maxheight());
				printf("			min width: %d\n", pInspectParam.geoparamlist(i).minwidth());
				printf("			max width: %d\n", pInspectParam.geoparamlist(i).maxwidth());
			}
		}
		else
		{
			printf("	The classifiy type is SVM\n");
			printf("		The svm model path is %s\n", pInspectParam.svmparam().modelpath().c_str());
			google::protobuf::Map<google::protobuf::int32, google::protobuf::string>::const_iterator b = pInspectParam.svmparam().defectlabelname().begin();
			for (; b != pInspectParam.svmparam().defectlabelname().end(); ++b)
			{
				printf("			label: %d, name: %s\n", b->first, b->second.c_str());
			}
		}
		printf("\n\nThis is Sheet Info:\n");

		ParamHelper<Parameters::SheetInfo> help_SheetInfo(&Global::g_param);
		const Parameters::SheetInfo pInfo = help_SheetInfo.getRef();
		printf("	The user name is %s\n", pInfo.username().c_str());
		printf("	The machine number is %s\n", pInfo.machinenumber().c_str());
		printf("	The imagimg model is %s\n", pInfo.imagemodel().c_str());
		printf("	The product name is %s\n", pInfo.prodctname().c_str());
		printf("	The order number is %s\n", pInfo.ordernumber().c_str());
		printf("	The batch number is %s\n", pInfo.batchnumber().c_str());
		printf("	The inspect param file is %s\n", pInfo.inspectparamname().c_str());
		printf("	The remark is %s\n", pInfo.remark().c_str());
		printf("	The strip info are:\n");
		printf("		The strips num is: %d\n", pInfo.stripparam().stripcount());
		printf("		The check all width is: %d mm\n", pInfo.stripparam().allwidth());
		printf("		The strip left offset is: %d mm\n", pInfo.stripparam().leftoffset());
		for (int i = 0; i < pInfo.stripparam().stripwidth_size(); i++)
		{
			printf("			The %d strip width is: %f\n", i, pInfo.stripparam().stripwidth()[i]);
		}
	}


	if (0)
	{
		
	}
	system("pause");*/
	return 0;
} 
/*
void fun1(int slp)
{
	printf("  hello, fun1 !  %d\n", std::this_thread::get_id());
	if (slp > 0) {
		printf(" ======= fun1 sleep %d  =========  %d\n", slp, std::this_thread::get_id());
		std::this_thread::sleep_for(std::chrono::milliseconds(slp));
		//Sleep(slp );
	}
}

struct gfun {
	int operator()(int n) {
		printf("%d  hello, gfun !  %d\n", n, std::this_thread::get_id());
		return 42;
	}
};

class A {    //函数必须是 static 的才能使用线程池
public:
	static int Afun(int n = 0) {
		std::cout << n << "  hello, Afun !  " << std::this_thread::get_id() << std::endl;
		return n;
	}

	static std::string Bfun(int n, std::string str, char c) {
		std::cout << n << "  hello, Bfun !  " << str.c_str() << "  " << (int)c << "  " << std::this_thread::get_id() << std::endl;
		return str;
	}
};*/

//cv::Mutex lc;

#include "Alogrithm/BoudarySearch.h"
int mainoo(int argc, _TCHAR* argv[])
{
	cv::Mat src = cv::imread("D:\\福州透明膜\\6.bmp", cv::IMREAD_GRAYSCALE);
	/* std::shared_ptr<CRunTimeHandle> pTestRun = new CRunTimeHandle;
	pTestRun->SetParam(40, cv::Size(src.cols, src.rows), cv::Size(2048, 2048), cv::Size(5, 5));
	CBoudarySearch* pB = new CBoudarySearch(pTestRun);
	std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> vecBoudaryParam;
	cv::Mat frdMask;
	double dTime;
	pB->BoundarySearch(src,vecBoudaryParam,frdMask, &dTime);

	printf("Boundary search time is: %f\n", dTime);*/

	system("pause");
	return 1;
}

//线程函数
void popFuc(safe_queue<ImageInspectResult>* pQueue, bool* isStop)
{
	char saveFile[256];
	std::shared_ptr<ImageInspectResult> p;
	//cv::Mat src, diff;
	int iCount = 0;
	char savePath[256] = "F:\\data\\XC\\data\\rlt-c7180105GX0501-3";
	char fileName[256];
	while (*isStop == false)
	{
		p = pQueue->wait_and_pop();
		if (p == nullptr)
		{
			continue; 
		}
		iCount++;
// 		if (iCount>10)
// 		{
// 			continue;
// 		}
// 		if (p->m_vecDefectList.size()==0)
// 		{
// 			continue;
// 		}
		//cv::resize(p->srcImage, p->srcImage, cv::Size(p->srcImage.cols >> 1, p->srcImage.rows >> 1));
		//cv::resize(p->diffImage, p->diffImage, cv::Size(p->diffImage.cols >> 1, p->diffImage.rows >> 1));
		//sprintf_s(saveFile, "..\\rlt\\src_%d.bmp", p->idx);
		//cv::imwrite(saveFile, p->srcImage);
		if (p->idx>3)
		{
			sprintf_s(fileName, "%s\\%d-diff.png", savePath, p->idx);
			//cv::imwrite(fileName, p->diffImage);
		}
		if (p->m_vecDefectList.size()!=0)
		{
			g_IO->SetDataToFile(*p);
			for (int i = 0; i < p->m_vecDefectList.size(); i++)
			{
				sprintf_s(fileName, "%s\\%d-%d-diff-%d-x-%f-y-%f-a-%f.png", savePath, p->idx, i, p->m_vecDefectList[i].iMeanDiff, p->m_vecDefectList[i].fPy_width, p->m_vecDefectList[i].fPy_height, p->m_vecDefectList[i].fPyArea);
				cv::imwrite(fileName, p->GetMask(i));

				sprintf_s(fileName, "%s\\%d-%d-src-%s.png", savePath, p->idx, i, p->m_vecDefectList[i].defectName.c_str());
				cv::imwrite(fileName, p->srcImage(p->m_vecDefectList[i].imgRect));
			}
		}
	}
	printf("pop thread is cancel........................!\n");
}

//线程函数
void logFun(safe_queue<std::string>* pQueue, bool* isStop)
{
	FILE* fr = fopen("F:\\data\\XC\\data\\rlt-c7180105GX0501-3\\time.txt","w");
	std::shared_ptr<std::string> p;
	while (*isStop == false)
	{
		p = pQueue->wait_and_pop();
		if (p == nullptr)
		{
			continue;
		}
		printf(((*p).c_str()));
		fprintf(fr, "%s", ((*p).c_str()));
#ifndef DEBUG
// 		if (g_grab->GetGrabCount() == 20)
// 		{
// 			g_grab->StopGrabThread();
// 			break;
// 		}
#endif
	}
	fclose(fr);
	printf("log thread is cancel........................!\n");
}

int main(int argc, _TCHAR* argv[])
{
	{
		cv::Point2f ptSrc[4];
		cv::Point2f ptDst[4];
		ptSrc[0] = cv::Point2f(10.0f, 10.0f);
		ptSrc[1] = cv::Point2f(100.0f, 10.0f);
		ptSrc[2] = cv::Point2f(100.0f, 10000.0f);
		ptSrc[3] = cv::Point2f(10.0f, 10000.0f);

		ptDst[0] = cv::Point2f(105.0f, 15.0f);
		ptDst[1] = cv::Point2f(110.0f, 15.0f);
		ptDst[2] = cv::Point2f(110.0f, 20.0f);
		ptDst[3] = cv::Point2f(105.0f, 20.0f);

		cv::Mat h = cv::getPerspectiveTransform(ptSrc, ptDst);
		cv::Mat a = cv::getAffineTransform(ptSrc, ptDst);
	}
	if (0)
	{
		/*try {
			std::threadpool executor;
			executor.SetSize(8);
			A a;
			std::future<void> ff = executor.commit(fun1, 0);
			std::future<int> fg = executor.commit(gfun{}, 0);
			std::future<int> gg = executor.commit(a.Afun, 9999); //IDE提示错误,但可以编译运行
			std::future<std::string> gh = executor.commit(A::Bfun, 9998, "mult args", 123);
			std::future<std::string> fh = executor.commit([]()->std::string { std::cout << "hello, fh !  " << std::this_thread::get_id() << std::endl; return "hello,fh ret !"; });

			std::cout << " =======  sleep ========= " << std::this_thread::get_id() << std::endl;
			std::this_thread::sleep_for(std::chrono::microseconds(900));

			for (int i = 0; i < 50; i++) {
				executor.commit(fun1, i * 100);
			}
			std::cout << " =======  commit all ========= " << std::this_thread::get_id() << " idlsize=" << executor.idlCount() << std::endl;

			std::cout << " =======  sleep ========= " << std::this_thread::get_id() << std::endl;
			std::this_thread::sleep_for(std::chrono::seconds(3));

			ff.get(); //调用.get()获取返回值会等待线程执行完,获取返回值
			std::cout << fg.get() << "  " << fh.get().c_str() << "  " << std::this_thread::get_id() << std::endl;

			std::cout << " =======  sleep ========= " << std::this_thread::get_id() << std::endl;
			std::this_thread::sleep_for(std::chrono::seconds(3));

			std::cout << " =======  fun1,55 ========= " << std::this_thread::get_id() << std::endl;
			executor.commit(fun1, 55).get();    //调用.get()获取返回值会等待线程执行完

			std::cout << "end... " << std::this_thread::get_id() << std::endl;


			std::threadpool pool;
			pool.SetSize(4);
			std::vector< std::future<int> > results;

			for (int i = 0; i < 8; ++i) {
				results.emplace_back(
					pool.commit([i] {
					std::cout << "hello " << i << std::endl;
					std::this_thread::sleep_for(std::chrono::seconds(1));
					std::cout << "world " << i << std::endl;
					return i*i;
				})
					);
			}
			std::cout << " =======  commit all2 ========= " << std::this_thread::get_id() << std::endl;

			for (auto && result : results)
				std::cout << "result is" << result.get() << ' ' << std::endl;
			std::cout << std::endl;

			system("pause");
			return 0;
		}
		catch (std::exception& e) {
			std::cout << "some unhappy happened...  " << std::this_thread::get_id() << e.what() << std::endl;
		}*/
	}
	
	
	if (1)
	{	
		//cv::imwrite("D:\\src-img\\8.bmp", src);
		{
			ParamHelper<Parameters::InspectParam> help_InspectParam(&Global::g_param);
			Parameters::InspectParam* pInspectParam = help_InspectParam.getPtr();
			pInspectParam->set_checktype(Parameters::InspectParam::DOG);//设置检测算法
			Parameters::GaussianParam* pp = pInspectParam->mutable_dogparam()->mutable_refkernel();
			pp->set_sizex(255);
			pp->set_sizey(255);
			pInspectParam->mutable_dogparam()->set_thresholddark(-10);
			pInspectParam->mutable_dogparam()->set_thresholdlight(10);
			pInspectParam->set_blobthr(0.04);

			pInspectParam->mutable_caffeparam()->set_modelpath("D:\\PVInspectionProject\\model");


			pInspectParam->set_dstvaluebgd(240);
			pInspectParam->set_dstvaluefrd(128);

			ParamHelper<Parameters::RunTimeParam> help_RunTimeParam(&Global::g_param);
			Parameters::RunTimeParam* pRunTimeParam = help_RunTimeParam.getPtr(); 
			pRunTimeParam->set_dowmsample(0);
			pRunTimeParam->set_freamheight(5000);
			pRunTimeParam->set_freamwidth(16384);
			pRunTimeParam->set_splitsizex(2048);
			pRunTimeParam->set_splitsizey(4000);
			pRunTimeParam->set_threadnum(4);
			pRunTimeParam->set_isgpu(true);

			ParamHelper<Parameters::SheetInfo> sheetHelper(&Global::g_param);
			Parameters::SheetInfo* pSheetInfo = sheetHelper.getPtr();
			Parameters::StripParam* s = pSheetInfo->mutable_stripparam();
			s->set_leftoffset(0);
			s->add_stripwidth(300);
			s->add_stripwidth(300);
			s->add_stripwidth(300);
			s->add_stripwidth(300);
			s->add_stripwidth(300);

			pSheetInfo->set_startlength(0.0);
			pSheetInfo->set_islefttoright(true);
			pSheetInfo->set_boundoffsetleft(5);
			pSheetInfo->set_boundoffsetright(5);

			ParamHelper<Parameters::SystemRectifyParam> help_sys(&Global::g_param);
			Parameters::SystemRectifyParam* pSystem = help_sys.getPtr();
			pSystem->set_xphysicresolution(0.0937);
			pSystem->set_yphysicresolution(0.1227);
			pSystem->set_lengthperfream(502.4);
			pSystem->set_offsety(4095);
		}
		{
			//GrabImgInfo temp0;
			CInspectProcedure* pInspect = new CInspectProcedure();
			pInspect->SetParam(&Global::g_param, &Global::g_ProcessQueue, &Global::g_InspectQueue, &Global::g_AlogrithmLog);
			pInspect->StartInspectThread(true);
			g_IO->SetSavePath("F:\\data\\XC\\data\\rlt-c7180105GX0501-3\\img.img");
#ifndef DEBUG
			//CGrabThread* pGrab = new CGrabThread;
			/*g_grab->StopGrabThread();
			g_grab->InitiaGrab(&Global::g_ProcessQueue, &Global::g_AlogrithmLog, true);
			g_grab->SetSavePath(true, "E:\\Grab\\2018\\06\\22\\c-06-22");
			double fGain = g_grab->GetGain();
			printf("Gain is %f\n", fGain);
			g_grab->SetGain(2.2);
			fGain = g_grab->GetGain();
			printf("Gain is %f\n", fGain);
			g_grab->StartGrabThread(true);*/
#endif
			
			
 			Global::g_isStop = false;
			std::thread popResult(popFuc, &Global::g_InspectQueue, &Global::g_isStop);
			popResult.detach();

			std::thread logResult(logFun, &Global::g_AlogrithmLog, &Global::g_isStop);
			logResult.detach();

			char filePath[256];
			GrabImgInfo temp0;
			for (int i = 1; i < 5; i++)
			{
				sprintf_s(filePath, "F:\\data\\XC\\data\\c7180105GX0501\\%d.png", i);
				temp0.idx = i;
				//temp0.iMark = GrabImgInfo::_normal_;
				double t1 = cvGetTickCount();
				temp0.srcimg = cv::imread(filePath, cv::IMREAD_GRAYSCALE);
				Global::g_ProcessQueue.push(std::move(temp0));
				t1 = (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency());
				//printf("read img %d is %f\n", i, t1);
			}
			for (int i = 5; i < 1000; i++)
			{
				sprintf_s(filePath, "F:\\data\\XC\\data\\c7180105GX0501\\%d.png", i);
				temp0.idx = i;
				//temp0.iMark = GrabImgInfo::_normal_;
				double t1 = cvGetTickCount();
				temp0.srcimg = cv::imread(filePath, cv::IMREAD_GRAYSCALE);
				Global::g_ProcessQueue.push(std::move(temp0));
				t1 = (cvGetTickCount() - t1) / (1000 * cvGetTickFrequency());
				//printf("read img %d is %f\n", i, t1);
			}

			system("pause");
			delete pInspect;
#ifndef DEBUG
			printf("Stop......\n");
			g_grab->StopGrabThread();
			g_grab->FreeGrab();

#endif
			Global::g_isStop = true;
			std::this_thread::sleep_for(std::chrono::microseconds(1000));
		}
		printf("CInspectProcedure is deleted!\n");
	}
	else
	{
		cv::Mat img = imread("d:\\7.bmp", cv::IMREAD_GRAYSCALE);
		cv::blur(img, img, cv::Size(3, 3));
		cv::Mat ref, g1, g2, g3, g0;
		cv::GaussianBlur(img, ref, cv::Size(255, 255),0,0);
		cv::GaussianBlur(img, g0, cv::Size(3, 3), 0, 0);
		cv::GaussianBlur(img, g1, cv::Size(13, 13), 0, 0);
		cv::GaussianBlur(img, g2, cv::Size(65, 5), 0, 0);
		cv::GaussianBlur(img, g3, cv::Size(5, 65), 0, 0);

		g0.convertTo(g0, CV_16S);
		g1.convertTo(g1, CV_16S);
		g2.convertTo(g2, CV_16S);
		g3.convertTo(g3, CV_16S);
		ref.convertTo(ref, CV_16S);
		cv::Mat diff0, diff1, diff2, diff3;
		diff0 = g0 - ref;
		diff1 = g1 - ref;
		diff2 = g2 - ref;
		diff3 = g3 - ref;
	}
	
	system("pause");

	//delete pGrab;
	return 0;
}


cv::Mat GetGaussKernel(cv::Size& kernelSize, float fSigmaW /*= 0.0f*/, float fSigmaH /*= 0.0f*/)
{
	cv::Mat kernel_w = cv::getGaussianKernel(kernelSize.width, fSigmaW, CV_32F);
	cv::transpose(kernel_w, kernel_w);
	cv::Mat kernel_h = cv::getGaussianKernel(kernelSize.height, fSigmaH, CV_32F);
	cv::Mat Kernel = kernel_h * kernel_w;
	return Kernel.clone();
}/*
void Conv_FFT32f_fftw(cv::Mat& SrcImg, cv::Mat& DstImg, cv::Mat& kernel)
{
	double t1 = cvGetTickCount();
	//int iRow = SrcImg.rows + kernel.rows - 1;
	//int iCol = SrcImg.cols + kernel.cols - 1;
	int iRow = SrcImg.rows;
	int iCol = SrcImg.cols;

	cv::Mat buffPatch = cv::Mat::zeros(iRow, iCol, CV_32F);
	cv::Mat complex_out_img(iRow, (iCol >> 1) + 1, CV_32FC2);
	cv::Mat complex_out_kernel = complex_out_img.clone();
	cv::Mat out = complex_out_img.clone();
	fftwf_plan forwardImg, forwardKernel, backward;


	forwardImg = fftwf_plan_dft_r2c_2d(buffPatch.rows, buffPatch.cols, (float*)buffPatch.data, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>((float*)complex_out_img.data)), FFTW_MEASURE);
	forwardKernel = fftwf_plan_dft_r2c_2d(buffPatch.rows, buffPatch.cols, (float*)buffPatch.data, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>((float*)complex_out_kernel.data)), FFTW_MEASURE);
	backward = fftwf_plan_dft_c2r_2d(buffPatch.rows, buffPatch.cols, const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex*>((float*)out.data)), (float*)buffPatch.data, FFTW_MEASURE);


	double t2 = cvGetTickCount();
	SrcImg.convertTo(buffPatch(cv::Rect(0, 0, SrcImg.cols, SrcImg.rows)), CV_32F);
	fftwf_execute(forwardImg);
	buffPatch.setTo(0x00);

	cv::Rect rtSrc, rtDst;
	rtSrc.x = kernel.cols >> 1;
	rtSrc.y = kernel.rows >> 1;
	rtSrc.width = kernel.cols - rtSrc.x;
	rtSrc.height = kernel.rows - rtSrc.y;
	rtDst.x = rtDst.y = 0;
	rtDst.width = rtSrc.width;
	rtDst.height = rtSrc.height;
	kernel(rtSrc).convertTo(buffPatch(rtDst), CV_32F);

	rtSrc.x = 0;
	rtSrc.y = 0;
	rtSrc.width = kernel.cols>>1;
	rtSrc.height = kernel.rows>>1;
	rtDst.x = buffPatch.cols - rtSrc.width;
	rtDst.y = buffPatch.rows - rtSrc.height;
	rtDst.width = rtSrc.width;
	rtDst.height = rtSrc.height;
	kernel(rtSrc).convertTo(buffPatch(rtDst), CV_32F);

	rtSrc.x = 0;
	rtSrc.y = kernel.rows >> 1;
	rtSrc.width = kernel.cols >> 1;
	rtSrc.height = kernel.rows >> 1;
	rtDst.x = buffPatch.cols - rtSrc.width;
	rtDst.y = 0;
	rtDst.width = rtSrc.width;
	rtDst.height = rtSrc.height;
	kernel(rtSrc).convertTo(buffPatch(rtDst), CV_32F);

	rtSrc.x = kernel.cols >> 1;
	rtSrc.y = 0;
	rtSrc.width = kernel.cols >> 1;
	rtSrc.height = kernel.rows >> 1;
	rtDst.x = 0;
	rtDst.y = buffPatch.rows - rtSrc.height;
	rtDst.width = rtSrc.width;
	rtDst.height = rtSrc.height;
	kernel(rtSrc).convertTo(buffPatch(rtDst), CV_32F);

	//kernel.convertTo(buffPatch(cv::Rect(0, 0, kernel.cols, kernel.rows)), CV_64F);
	fftwf_execute(forwardKernel);

	cv::mulSpectrums(complex_out_img, complex_out_kernel, out, cv::DFT_COMPLEX_OUTPUT);

	fftwf_execute(backward);

	buffPatch.convertTo(buffPatch, CV_32F, 1.0 / double(buffPatch.cols*buffPatch.rows));

	int iHalfX = (kernel.cols >> 1);
	int iHalfY = (kernel.rows >> 1);
	DstImg = cv::Mat::zeros(SrcImg.rows, SrcImg.cols, CV_32F);

	buffPatch.convertTo(DstImg, CV_32F);

	fftwf_destroy_plan(forwardImg);
	fftwf_destroy_plan(forwardKernel);
	fftwf_destroy_plan(backward);

	double t3 = cvGetTickCount();
	printf("%f %f %f\n\n", (t2 - t1) / (1000 * cvGetTickFrequency()), (t3 - t2) / (1000 * cvGetTickFrequency()));
}*/

int main_e(int argc, _TCHAR* argv[])
{
	cv::Mat img = cv::imread("D:\\test.bmp", cv::IMREAD_GRAYSCALE);
	cv::Mat k = GetGaussKernel(cv::Size(255, 255), 0, 0);
	cv::Mat dstGauss, dstFFtw;
	cv::GaussianBlur(img, dstGauss, cv::Size(255, 255),0,0);

	//Conv_FFT32f_fftw(img, dstFFtw, k);

	dstGauss.convertTo(dstGauss, CV_32F);
	cv::Mat diff;
	cv::absdiff(dstGauss, dstFFtw, diff);
	double dmax;
	cv::minMaxLoc(diff, NULL, &dmax);
	cv::Scalar m, s;
	cv::meanStdDev(diff, m, s);
	printf("mean=%f, std=%f, max=%f\n", m.val[0],s.val[0],dmax);

	system("pause");
	return 0;
}