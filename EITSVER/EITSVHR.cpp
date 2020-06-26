#include"head.h"
int main() {
	////////////////////////////////////////////////参数设置///////////////////////////////////////////////////////////////////////////
	IplImage* prime_img = cvLoadImage("加载原始影像路径");
	IplImage* Graeyimg = cvLoadImage("加载原始图像对应的灰度影像路径");
	IplImage* ClassMap1 = cvLoadImage("加载KNN分类器分类后影像路径");
	IplImage* ClassMap2 = cvLoadImage("加载MLC分类器分类后影像路径");
	IplImage* ClassMap3 = cvLoadImage("加载RT分类器分类后影像路径");
	string path = "产生训练样本路径";//产生每次训练样本路径
	int ClassNum = 7;//类别个数根据数据影像设置
	//如需更改直方图的bin数需要在heda.h文件中设置
	////////////////////////////////////////////////参数设置///////////////////////////////////////////////////////////////////////////
	
	char path1[256] = { 0 };
	CpC Percount;//统计每类的坐标
	Histogram histogram;//创建直方图对象
	Percount = Count_PerClass(ClassMap1, ClassMap2, ClassMap3, ClassNum);
	sprintf(path1, "%s", path.data());
	ICTxt(path1, Percount.IC);
	histogram = Stratifiedsampling(prime_img, Graeyimg, ClassMap1, Percount, ClassNum, path1);

	return 0;
}
//统计每类像素的坐标以及个数
CpC Count_PerClass(IplImage* ClassMap1, IplImage* ClassMap2, IplImage* ClassMap3, int ClassNum)
{
	CpC cpc;
	int Map1label = 0;
	int Map2label = 0;
	int Map3label = 0;
	CvPoint point;
	double PSCount = 0;
	for (int k = 1; k <= ClassNum; k++)
	{
		for (int i = 0; i < ClassMap1->height; i++)
		{
			for (int j = 0; j < ClassMap1->width; j++)
			{
				Map1label = *cvPtr2D(ClassMap1, i, j, NULL);
				Map2label = *cvPtr2D(ClassMap2, i, j, NULL);
				Map3label = *cvPtr2D(ClassMap3, i, j, NULL);
				if ((Map1label == Map2label) && (Map2label == Map3label) && (Map3label == k))
				{
					point.x = i;
					point.y = j;
					(cpc.points[k]).push_back(point);
					PSCount++;
				}
			}
		}
	}
	cpc.IC = PSCount / ((ClassMap1->height)*(ClassMap1->width));
	return cpc;
}
//统计直方图下的bin数及像素坐标
Histogram Stratifiedsampling(IplImage* prime_img, IplImage* Graeyimg, IplImage* ClassMap, CpC cpc, int ClassNum, char path[256])
{
	Histogram hist;//直方图对象
	STSelect stse;//选择样本对象
	WinPointVar SampVar;//箱线区取样本及计算方差
	int Pixel= 0;//取像素值
	int HistBin = 0;//像素值的bin数
	for (int i = 1; i <= ClassNum; i++)
	{
		for (int j = 0; j < (cpc.points[i]).size(); j++)
		{
			Pixel = *cvPtr2D(Graeyimg, (cpc.points[i][j]).x, (cpc.points[i][j]).y,NULL);
			HistBin = Get_HisBin(Pixel, hist.bin);
			for (int p = 1; p <= hist.bin; p++)
			{
				if (HistBin == p)
				{
					(hist.BinArray[p]).push_back(cpc.points[i][j]);
					break;
				}
			}
		}
		for (int m = 1; m <= hist.bin; m++)
		{
			if ((hist.BinArray[m].size())*0.5 > 100)
			{
				for (int k = 0; k < (hist.BinArray[m].size())*0.5; k++)
				{
					int random = rand() % (hist.BinArray[m].size());
					(hist.RatioSampling[m]).push_back(hist.BinArray[m][random]);
				}
			}
			else if ((hist.BinArray[m].size())*0.5 < 100 && (hist.BinArray[m].size())>100)
			{
				for (int w = 0; w < 100; w++)
				{
					int random = rand() % (hist.BinArray[m].size());
					(hist.RatioSampling[m]).push_back(hist.BinArray[m][random]);
				}
			}
			else if ((hist.BinArray[m].size())<100)
			{
				for (int r = 0; r < (hist.BinArray[m].size()); r++)
				{
					(hist.RatioSampling[m]).push_back(hist.BinArray[m][r]);
				}
			}
		}
		stse = SampleToSelected(hist, prime_img, ClassMap, ClassNum);
		SampVar = Get_RVarPoint(prime_img, ClassMap, stse, hist);
		SamplePoint_TXT(SampVar.SamplePoint,prime_img, ClassMap, path);
		//对数组进行清空
		for (int k = 1; k <= hist.bin; k++)
		{
			hist.BinArray[k].clear(); hist.BinArray[k].shrink_to_fit();
			hist.RatioSampling[k].clear(); hist.RatioSampling[k].shrink_to_fit();
		}
	}
	return hist;
}
//计算像素灰度值的直方图bin数
int Get_HisBin(int pixel, int bin)
{
	int bin_Num = 0;//直方图区间值
	for (int i = 1; i <= bin; i++)
	{
		bin_Num = i * (256 / bin);//取两个数中间的值
		if (bin_Num>pixel)
		{
			if (((i - 1)*(256 / bin)) <= pixel)
			{
				return i;
			}
		}
	}
}
//将取到的每类像素点以窗口的增长方式取同一类别的像素中心点找到同质性高的区域
STSelect SampleToSelected(Histogram his, IplImage* prime_img, IplImage* ClassMap, int ClassNum)
{
	STSelect stselect;
	CvPoint center;
	int label = 0;
	for (int a = 1; a <= his.bin; a++)
	{
		for (int i = 0; i < his.RatioSampling[a].size(); i++)
		{

			center.x = his.RatioSampling[a][i].x;
			center.y = his.RatioSampling[a][i].y;
			stselect.winGetpoint = points(prime_img, center, 3);
			stselect.Countlabel.clear(); stselect.Countlabel.shrink_to_fit();
			for (int j = 0; j < (stselect.winGetpoint).size(); j++)
			{
				label = *cvPtr2D(ClassMap, stselect.winGetpoint[j].x, stselect.winGetpoint[j].y, NULL);
				stselect.Countlabel.push_back(label);
			}
			int num = 0;
			for (int b = 0; b < stselect.Countlabel.size(); b++)
			{
				if (stselect.Countlabel[0] != stselect.Countlabel[b])
				{
					break;
				}
				else
				{
					num++;
				}
				if (num == 9)
				{
					stselect.CandidateSample[a].push_back(center);
				}
			}
		}
	}
	for (int c = 1; c <= his.bin; c++)
	{
		if ((stselect.CandidateSample[c].size())*0.02>100)
		{
			for (int d = 0; d < (stselect.CandidateSample[c].size())*0.02; d++)
			{
				int random = rand() % (stselect.CandidateSample[c].size());
				(stselect.WaitingQuartilePoint).push_back(stselect.CandidateSample[c][random]);
			}
		}
		else if ((stselect.CandidateSample[c].size())*0.02 < 100 && (stselect.CandidateSample[c].size())>100)
		{
			for (int q = 0; q < 100; q++)
			{
				int random = rand() % (stselect.CandidateSample[c].size());
				(stselect.WaitingQuartilePoint).push_back(stselect.CandidateSample[c][random]);
			}
		}
		else if ((stselect.CandidateSample[c].size()) < 100)
		{
			for (int e = 0; e < (stselect.CandidateSample[c].size()); e++)
			{
				(stselect.WaitingQuartilePoint).push_back(stselect.CandidateSample[c][e]);
			}
		}


	}
	return stselect;
}
//以窗口的方式扫描像素
vector<CvPoint> points(IplImage* img, CvPoint center, int winsize)
{
	vector<CvPoint> points;
	CvPoint point;
	int minX = center.x - (int)floor(0.5 * winsize);
	int maxX = center.x + (int)floor(0.5 * winsize);
	int minY = center.y - (int)floor(0.5 * winsize);
	int maxY = center.y + (int)floor(0.5 * winsize);
	for (int i = minX; i <= maxX; i++)
	{
		for (int j = minY; j <= maxY; j++)
		{
			if (i >= 0 && i < img->height - 1 && j >= 0 && j < img->width)
			{
				point.x = i; point.y = j;
				points.push_back(point);
			}
		}
	}
	return points;
}
//运用双窗口以及箱线原理排除异常样本后选择精炼训练样本
WinPointVar Get_RVarPoint(IplImage* prime_img, IplImage* ClassMap, STSelect sts, Histogram his)
{
	WinPointVar wpv;
	CvPoint center;
	VarFeature varfeature;
	VarFeature Tempvarfeature;
	vector<double>Rovarfeature;
	vector<double>RoTempvarfeature;
	vector<CvPoint>Goodpoint;//没有异常点
	vector<int>Coutlabel;
	int label = 0;
	double IQR = 0;//四分位距
	double ON_Outlier = 0;//上异常值
	double UP_Outlier = 0;//下异常值
	for (int i = 0; i < sts.WaitingQuartilePoint.size(); i++)
	{
		int PointNum = -1;
		center.x = sts.WaitingQuartilePoint[i].x;
		center.y = sts.WaitingQuartilePoint[i].y;
		for (int WinNum = 9; WinNum>3; WinNum=WinNum-2)
		{
			wpv.win_points.clear(); wpv.win_points.shrink_to_fit();
			Coutlabel.clear(); Coutlabel.shrink_to_fit();
			wpv.win_points = points(prime_img, center, WinNum);
			for (int g = 0; g < wpv.win_points.size(); g++)
			{
				label = *cvPtr2D(ClassMap, wpv.win_points[g].x, wpv.win_points[g].y, NULL);
				Coutlabel.push_back(label);
			}
			int num = 0;
			for (int h = 0; h < Coutlabel.size(); h++)
			{
				if (Coutlabel[0] != Coutlabel[h])
				{
					break;
				}
				else
				{
					num++;
				}
			}
			if (num == WinNum* WinNum)
			{
				wpv.win_points = points(prime_img, center, WinNum - 2);
				int SelecPoint[5] = { 0,(wpv.win_points.size()+1)*0.25,(wpv.win_points.size() + 1)*0.5,(wpv.win_points.size() + 1)*0.75,wpv.win_points.size()- 1};
				if (wpv.win_points.size() == (WinNum-2)*(WinNum-2))
				{
					Tempvarfeature = VarF(prime_img, wpv.win_points, WinNum - 2);
					varfeature = VarF(prime_img, wpv.win_points, WinNum - 2);
					sort(varfeature.Rvar.begin(), varfeature.Rvar.end());
					IQR = varfeature.Rvar[SelecPoint[3]] - varfeature.Rvar[SelecPoint[1]];
					ON_Outlier = varfeature.Rvar[SelecPoint[1]] - 3*IQR;//极度异常值
					UP_Outlier = varfeature.Rvar[SelecPoint[3]] + 3*IQR;//极度异常值
					RoTempvarfeature.clear(); RoTempvarfeature.shrink_to_fit();
					Rovarfeature.clear(); Rovarfeature.shrink_to_fit();
					Goodpoint.clear(); Goodpoint.shrink_to_fit();
					for (int  P = 0; P < Tempvarfeature.Rvar.size(); P++)
					{
						if (Tempvarfeature.Rvar[P]<ON_Outlier|| Tempvarfeature.Rvar[P]>UP_Outlier)
						{
							continue;
						}
						else
						{
							RoTempvarfeature.push_back(Tempvarfeature.Rvar[P]);
							Rovarfeature.push_back(Tempvarfeature.Rvar[P]);
							Goodpoint.push_back(wpv.win_points[P]);
						}
					}
					sort(Rovarfeature.begin(), Rovarfeature.end());
					int QuartilePoint[5] = { 0,(Rovarfeature.size()+1)*0.25,(Rovarfeature.size() + 1)*0.5,(Rovarfeature.size() + 1)*0.75, Rovarfeature.size()-1};
					for (int j = 0; j <5; j++)
					{
						PointNum = ReturnToPunctuation(RoTempvarfeature, Rovarfeature[QuartilePoint[j]]);
						wpv.SamplePoint.push_back(Goodpoint[PointNum]);
					}
					break;
				}
			}
		}
	}
	return wpv;
}
/*计算像素的均值及方差*/
VarFeature VarF(IplImage* prime_img, vector<CvPoint>Wpoints, int WinSize)
{
	VarFeature varf;
	vector<CvPoint>WWpoint;
	for (int i = 0; i < Wpoints.size(); i++)
	{
		WWpoint.clear(); WWpoint.shrink_to_fit();
		WWpoint = points(prime_img, Wpoints[i], WinSize);
		varf.mean_RGB = Get_Mean(prime_img, WWpoint);
		varf.var_RGB = Get_Var(prime_img, WWpoint, varf.mean_RGB);
		varf.Varvalue.push_back(varf.var_RGB);
		varf.Rvar.push_back(varf.Varvalue[i][2]);
	}

	return varf;
}
/*计算均值*/
vector<double>Get_Mean(IplImage* img, vector<CvPoint>WinRegion)
{
	CvScalar c;
	double Sum_R = 0;
	double Sum_G = 0;
	double Sum_B = 0;
	double mean_R = 0;
	double mena_G = 0;
	double mean_B = 0;
	vector<double>RGB_means;
	for (int i = 0; i < WinRegion.size(); i++)
	{
		c = cvGet2D(img, WinRegion[i].x, WinRegion[i].y);
		Sum_B += c.val[0];
		Sum_G += c.val[1];
		Sum_R += c.val[2];
	}
	mean_B = Sum_B / WinRegion.size();
	mena_G = Sum_G / WinRegion.size();
	mean_R = Sum_R / WinRegion.size();
	RGB_means.push_back(mean_B);
	RGB_means.push_back(mena_G);
	RGB_means.push_back(mean_R);
	return RGB_means;
}
/*计算方差*/
vector<double> Get_Var(IplImage* img, vector<CvPoint>WinRegion, vector<int>Means)
{
	double Var = 0;
	CvScalar c;
	double Var_R = 0;
	double Var_G = 0;
	double Var_B = 0;
	double Sum_Var_R = 0;
	double Sum_Var_G = 0;
	double Sum_Var_B = 0;
	vector<double> RGB_Var;
	for (int i = 0; i <WinRegion.size(); i++)
	{
		c = cvGet2D(img, WinRegion[i].x, WinRegion[i].y);
		Var_B = pow(c.val[0] - Means[0], 2) / WinRegion.size();
		Sum_Var_B += Var_B;
		Var_G = pow(c.val[1] - Means[1], 2) / WinRegion.size();
		Sum_Var_G += Var_G;
		Var_R = pow(c.val[2] - Means[2], 2) / WinRegion.size();
		Sum_Var_R += Var_R;
	}
	RGB_Var.push_back(Sum_Var_B);
	RGB_Var.push_back(Sum_Var_G);
	RGB_Var.push_back(Sum_Var_R);
	return RGB_Var;
}
int ReturnToPunctuation(vector<double>Primendata, double Sortdata)
{
	for (int i = 0; i < Primendata.size(); i++)
	{
		if (Primendata[i] == Sortdata)
		{
			return i;
		}
	}
}
/*写入精炼训练样本*/
void SamplePoint_TXT(vector<CvPoint>FianlSample, IplImage*primeimg, IplImage *ClassMap, char path[256])
{
	IplImage* ImageSample = cvCreateImage(cvGetSize(primeimg), primeimg->depth, 1);//避免重采样影像
	fstream dataFile;//读取文件流
	char adr[256] = { 0 };
	sprintf(adr, "%s\\Trainning.txt", path);
	dataFile.open(adr, ios::app);
	CvScalar c;//获取原始像素RGB值
	int label = 0;
	if (!dataFile)
	{
		cerr << "打开文件失败" << endl;
		exit(0);
	}
	//cout << "打开文件成功";
	//dataFile << "description" << "," << "label" << "," << "i" << "," << "j"<<","<< "B" << "," << "G" << "," << "R" << endl;
	for (int i = 0; i < FianlSample.size(); i++)
	{
		if (*(cvPtr2D(ImageSample, FianlSample[i].x, FianlSample[i].y, NULL)) != 255)
		{
			c = cvGet2D(primeimg, FianlSample[i].x, FianlSample[i].y);
			label = *cvPtr2D(ClassMap, FianlSample[i].x, FianlSample[i].y, NULL);
			dataFile << label << "," << FianlSample[i].x << "," << FianlSample[i].y << "," << c.val[0] << "," << c.val[1] << "," << c.val[2] << endl;
			*(cvPtr2D(ImageSample, FianlSample[i].x, FianlSample[i].y, NULL)) = 255;
		}
	}
}
//输出每次迭代系数
void ICTxt(char path[256],double IC)
{
	fstream dataFile;//读取文件流
	char adr[256] = { 0 };
	sprintf(adr, "%s\\IterationCoefficient.txt", path);
	dataFile.open(adr, ios::app);
	CvScalar c;//获取原始像素RGB值
	int label = 0;
	if (!dataFile)
	{
		cerr << "打开文件失败" << endl;
		exit(0);
	}
	//cout << "打开文件成功";
	dataFile << "迭代系数:" << IC << endl;
}
