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
struct CpC//ͳ��ÿ�����صĸ���,������
{
	vector<CvPoint>points[10];
	double IC = 0;//����ϵ������ֹͣ
};

struct Histogram
{
	int bin = 16;//ֱ��ͼ��bin��
	vector<CvPoint>BinArray[17];//ÿ�ఴֱ��ͼbinȡ����
	vector<CvPoint>RatioSampling[17];


};
struct STSelect
{
	vector<CvPoint>winGetpoint;
	vector<int>Countlabel;//ÿ�ఴֱ��ͼbinȡ����
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
	vector<int >mean_RGB;//�����ķ�λ��ȡ��ֵ
	vector<double >var_RGB;//�����ķ�λ��ȡ����
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
