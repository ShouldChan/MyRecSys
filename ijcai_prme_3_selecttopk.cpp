#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
using namespace std;

#define list "./result/dist_list_prme.txt"
#define result "./result/top1_prme.txt"
#define K 1
struct Data{
	int user;	int poi;		double value;
};
vector<Data> vecTop;
void mypushback(vector<Data> &data, const int& u, const int& p, const double& v){
	Data d;
	d.user = u; d.poi = p; d.value = v;
	data.push_back(d);
}
void select(){
	ifstream fin(list);
	ofstream fout(result);
	string s;
	while (!fin.eof()){
		int u, p; double v;
		fin >> u >> p >> v;
		getline(fin, s);
		mypushback(vecTop, u, p, v);
	}
	cout << "push_back ok..." << endl;

	int count = 0;
	for (int i = 0; i != vecTop.size() - 1; i++){
		while (count < K){
			if (count == K-1)
				fout << vecTop[i].poi << endl;
			else
				fout << vecTop[i].poi << "\t";
			i++;
			count++;
		}
		if (vecTop[i].user != vecTop[i + 1].user)
			count = 0;
	}
	fin.close();
	fout.close();
}
int main(){
	select();
	return 0;
}