#include <stdio.h>
#include <tchar.h>
#include "cv.h"
#include "highgui.h"
#include "math.h"
#include <iostream>
#include <fstream> 
#include <iomanip>
#include <windows.h>
#include "io.h"
#include<time.h>
using namespace std;
using namespace cv;
struct CpC//统计每类像素的个数,及坐标
{
	vector<CvPoint>points[10];
	double IC = 0;//迭代系数控制停止
};

struct Histogram
{
	int bin = 16;//直方图的bin数
	vector<CvPoint>BinArray[17];//每类按直方图bin取样本
	vector<CvPoint>RatioSampling[17];


};
struct STSelect
{
	vector<CvPoint>winGetpoint;
	vector<int>Countlabel;//每类按直方图bin取样本
	vector<CvPoint>CandidateSample[17];
	vector<CvPoint>WaitingQuartilePoint;

};
struct WinPointVar
{
	vector<CvPoint>win_points;
	vector<double>WWVar;
	vector<double>TempWWVar;
	vector<CvPoint>SamplePoint;
};
struct VarFeature
{
	vector<int >mean_RGB;//所有四分位点取均值
	vector<double >var_RGB;//所有四分位点取方差
	vector<double>Rvar;
};
CpC Count_PerClass(IplImage* ClassMap1, IplImage* ClassMap2, IplImage* ClassMap3, int ClassNum);
Histogram Stratifiedsampling(IplImage* prime_img, IplImage* Graeyimg, IplImage* ClassMap1, CpC cpc, int ClassNum, char path[256]);
int Get_HisBin(int pixel, int bin);
STSelect SampleToSelected(Histogram his, IplImage* prime_img, IplImage* ClassMap1, int ClassNum);
vector<CvPoint> points(IplImage* img, CvPoint center, int winsize);
WinPointVar Get_RVarPoint(IplImage* prime_img, IplImage* ClassMap1, STSelect sts, Histogram his);
VarFeature VarF(IplImage* prime_img, vector<CvPoint>Wpoints, int WinSize);
int ReturnToPunctuation(vector<double>Primendata, double Sortdata);
void SamplePoint_TXT(vector<CvPoint>FianlSample, IplImage*primeimg, IplImage *ClassMap, char path[256]);
vector<double>Get_Mean(IplImage* img, vector<CvPoint>WinRegion);
vector<double> Get_Var(IplImage* img, vector<CvPoint>WinRegion, vector<int>Means);
void ICTxt(char path[256], double IC);
