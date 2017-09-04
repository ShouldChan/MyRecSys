//6����ȡÿ���û��ĺ�30%�ļ�¼ 
#include<iostream>
#include<algorithm>
#include<fstream>
#include<sstream>
#include<string.h>
#include<stdio.h>
#include<vector>
#include<map>
#include<cstdlib>
#include<time.h>
#include <functional> 
#include<stdlib.h>
#include <malloc.h>
using namespace std;

#define ifilename "./data_v1.txt"
#define test "./data_test.txt"
#define train "./data_train.txt"

//�Զ���ṹ�壬���ڴ洢ÿ������
struct Data{
	int userid; int poiid; string lat; string lon; string date; string time; 
	int npoiid; string nlat; string nlon; string ndate;string ntime;
	bool is_test;
};

//�Զ���ṹ�����ڼ�¼ÿ���û���checkin�������Ա�������������ĵ���
struct Usercount{
	int uid;
	int ucount;
};

//�Զ����vector��pushback����
void myPushback(vector<Data> & vecData, const int &uid, const int &pid, const string &la,
	const string &lo, const string &da, const string &ti, 
	const int &npid, const string &nla, const string &nlo, const string &nda, const string &nti,
	 const bool &istest){
	Data data;
	data.userid = uid; data.poiid = pid; data.lat = la; data.lon = lo; data.date = da; data.time = ti;//6
	data.npoiid = npid; data.nlat = nla; data.nlon = nlo; data.ndate = nda; data.ntime = nti; //5
	data.is_test = istest;
	vecData.push_back(data);
}
void countPushback(vector<Usercount>& vecUc, const int& uid, const int& ucount){
	Usercount userc;
	userc.uid = uid;		userc.ucount = ucount;
	vecUc.push_back(userc);
}

bool sortBy(const int &s1, const int &s2){
	return s1 < s2;
}

void getTest(){
	ifstream fin(ifilename);
	ofstream testout(test);
	ofstream trainout(train);
	if (!fin){
		cout << "error!" << endl;
		exit(1);
	}
	//һ·���м�¼����vector
	vector<Data> vecStr;
	string s;
	while (!fin.eof()){
		int uid, pid, npid;
		string str[8];
		fin >> uid >> pid >> str[0] >> str[1] >> str[2] >> str[3] >> npid >> str[4] >> str[5] >> str[6] >> str[7] ;
		getline(fin, s);
		myPushback(vecStr, uid, pid, str[0], str[1], str[2], str[3],npid, str[4], str[5], str[6], str[7], 0);
	}
	cout << "pushback ok..." << endl;

	//ͳ�Ʊ���vecStr����ÿ�û���checkin����
	int count = 1, j = 0;
	vector<Usercount> vecucount;
	for (int i = 0; i < vecStr.size() - 1; ++i){
		if (vecStr[i].userid == vecStr[i + 1].userid){
			//cout << "* ";
			++count;
		}
		else{
			countPushback(vecucount, vecStr[i].userid, count);
			++j;
			count = 0;
		}
	}
	countPushback(vecucount, 606, 46);//�� �Ժ����Ż���2293 110
	cout << "each user checkin nums save ok...." << endl;
	//��ȡtest��
	vector<Data> vecTest;
	int N = 0.7*vecucount[0].ucount;
	int M = vecucount[0].ucount;
	int jj = 0;
	for (int i = 0; i < vecStr.size() - 1; ++i){
		if (i >= N&&i <= M){
			vecStr[i].is_test = 1;
		}
		if (vecStr[i].userid != vecStr[i + 1].userid){
			N = i + 1 + 0.7*vecucount[++jj].ucount;
			M = i + 1 + vecucount[jj].ucount;
		}
	}
	cout << "set test ok..." << endl;
	//�����ֱ����Ϊ��test�ļ�¼�����vecTest
	for (int i = 0; i != vecStr.size(); i++){
		if (vecStr[i].is_test == 1){
			vecTest.push_back(vecStr[i]);
		}
	}
	for (vector<Data>::iterator iter = vecStr.begin(); iter != vecStr.end();){
		if (1 == (*iter).is_test)
			iter = vecStr.erase(iter);
		else
			++iter;
	}
	//д��
	for (vector<Data>::iterator it = vecStr.begin(); it != vecStr.end(); it++){
		trainout << it->userid << "\t" << it->poiid << "\t" << it->lat << "\t" << it->lon << "\t" << it->date << " " << it->time << "\t"
			 << it->npoiid << "\t" << it->nlat << "\t" << it->nlon << "\t"
			<< it->ndate << " " << it->ntime << endl;
	}
	for (vector<Data>::iterator it = vecTest.begin(); it != vecTest.end(); it++){
		testout << it->userid << "\t" << it->poiid << "\t" << it->lat << "\t" << it->lon << "\t" << it->date << " " << it->time << "\t"
			<< it->npoiid << "\t" << it->nlat << "\t" << it->nlon << "\t"
			<< it->ndate << " " << it->ntime << endl;
	}

	fin.close();
	trainout.close();
	testout.close();
}

int main(){
	getTest();
	return 0;
}