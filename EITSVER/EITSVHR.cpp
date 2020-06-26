#include"head.h"
int main() {
	////////////////////////////////////////////////��������///////////////////////////////////////////////////////////////////////////
	IplImage* prime_img = cvLoadImage("����ԭʼӰ��·��");
	IplImage* Graeyimg = cvLoadImage("����ԭʼͼ���Ӧ�ĻҶ�Ӱ��·��");
	IplImage* ClassMap1 = cvLoadImage("����KNN�����������Ӱ��·��");
	IplImage* ClassMap2 = cvLoadImage("����MLC�����������Ӱ��·��");
	IplImage* ClassMap3 = cvLoadImage("����RT�����������Ӱ��·��");
	string path = "����ѵ������·��";//����ÿ��ѵ������·��
	int ClassNum = 7;//��������������Ӱ������
	//�������ֱ��ͼ��bin����Ҫ��heda.h�ļ�������
	////////////////////////////////////////////////��������///////////////////////////////////////////////////////////////////////////
	
	char path1[256] = { 0 };
	CpC Percount;//ͳ��ÿ�������
	Histogram histogram;//����ֱ��ͼ����
	Percount = Count_PerClass(ClassMap1, ClassMap2, ClassMap3, ClassNum);
	sprintf(path1, "%s", path.data());
	ICTxt(path1, Percount.IC);
	histogram = Stratifiedsampling(prime_img, Graeyimg, ClassMap1, Percount, ClassNum, path1);

	return 0;
}
//ͳ��ÿ�����ص������Լ�����
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
//ͳ��ֱ��ͼ�µ�bin������������
Histogram Stratifiedsampling(IplImage* prime_img, IplImage* Graeyimg, IplImage* ClassMap, CpC cpc, int ClassNum, char path[256])
{
	Histogram hist;//ֱ��ͼ����
	STSelect stse;//ѡ����������
	WinPointVar SampVar;//������ȡ���������㷽��
	int Pixel= 0;//ȡ����ֵ
	int HistBin = 0;//����ֵ��bin��
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
		//������������
		for (int k = 1; k <= hist.bin; k++)
		{
			hist.BinArray[k].clear(); hist.BinArray[k].shrink_to_fit();
			hist.RatioSampling[k].clear(); hist.RatioSampling[k].shrink_to_fit();
		}
	}
	return hist;
}
//�������ػҶ�ֵ��ֱ��ͼbin��
int Get_HisBin(int pixel, int bin)
{
	int bin_Num = 0;//ֱ��ͼ����ֵ
	for (int i = 1; i <= bin; i++)
	{
		bin_Num = i * (256 / bin);//ȡ�������м��ֵ
		if (bin_Num>pixel)
		{
			if (((i - 1)*(256 / bin)) <= pixel)
			{
				return i;
			}
		}
	}
}
//��ȡ����ÿ�����ص��Դ��ڵ�������ʽȡͬһ�����������ĵ��ҵ�ͬ���Ըߵ�����
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
//�Դ��ڵķ�ʽɨ������
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
//����˫�����Լ�����ԭ���ų��쳣������ѡ����ѵ������
WinPointVar Get_RVarPoint(IplImage* prime_img, IplImage* ClassMap, STSelect sts, Histogram his)
{
	WinPointVar wpv;
	CvPoint center;
	VarFeature varfeature;
	VarFeature Tempvarfeature;
	vector<double>Rovarfeature;
	vector<double>RoTempvarfeature;
	vector<CvPoint>Goodpoint;//û���쳣��
	vector<int>Coutlabel;
	int label = 0;
	double IQR = 0;//�ķ�λ��
	double ON_Outlier = 0;//���쳣ֵ
	double UP_Outlier = 0;//���쳣ֵ
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
					ON_Outlier = varfeature.Rvar[SelecPoint[1]] - 3*IQR;//�����쳣ֵ
					UP_Outlier = varfeature.Rvar[SelecPoint[3]] + 3*IQR;//�����쳣ֵ
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
/*�������صľ�ֵ������*/
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
/*�����ֵ*/
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
/*���㷽��*/
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
/*д�뾫��ѵ������*/
void SamplePoint_TXT(vector<CvPoint>FianlSample, IplImage*primeimg, IplImage *ClassMap, char path[256])
{
	IplImage* ImageSample = cvCreateImage(cvGetSize(primeimg), primeimg->depth, 1);//�����ز���Ӱ��
	fstream dataFile;//��ȡ�ļ���
	char adr[256] = { 0 };
	sprintf(adr, "%s\\Trainning.txt", path);
	dataFile.open(adr, ios::app);
	CvScalar c;//��ȡԭʼ����RGBֵ
	int label = 0;
	if (!dataFile)
	{
		cerr << "���ļ�ʧ��" << endl;
		exit(0);
	}
	//cout << "���ļ��ɹ�";
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
//���ÿ�ε���ϵ��
void ICTxt(char path[256],double IC)
{
	fstream dataFile;//��ȡ�ļ���
	char adr[256] = { 0 };
	sprintf(adr, "%s\\IterationCoefficient.txt", path);
	dataFile.open(adr, ios::app);
	CvScalar c;//��ȡԭʼ����RGBֵ
	int label = 0;
	if (!dataFile)
	{
		cerr << "���ļ�ʧ��" << endl;
		exit(0);
	}
	//cout << "���ļ��ɹ�";
	dataFile << "����ϵ��:" << IC << endl;
}
