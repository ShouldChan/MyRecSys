#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
using namespace std;

#define test "./data/foursquare_test.txt"
#define result "./result/test_list_prme.txt"

struct Data{
	int user; int npoi;
};
void mypushback(vector<Data> &vecnpoi, const int& u, const int&np){
	Data da;
	da.user = u;	da.npoi = np;
	vecnpoi.push_back(da);
}
vector<Data> vecNpoi;
void selectTest(){
	ifstream fin(test);
	ofstream fout(result);
	string s;
	while (!fin.eof()){
		string str[9];
		int user, npoi;
		fin >> user >> str[0] >> str[1] >> str[2] >> str[3] >> str[4] >> npoi >> str[5] >> str[6] >> str[7] >> str[8];
		getline(fin, s);
		mypushback(vecNpoi, user, npoi);
	}
	cout << "vecNpoi pushback ok..." << endl;

	for (int i = 0; i != vecNpoi.size() - 1; i++){
		fout << vecNpoi[i].npoi << "\t";
		if (vecNpoi[i].user != vecNpoi[i + 1].user){
			fout << vecNpoi[i].npoi << endl;
		}
	}
	cout << "make test_list ok..." << endl;
	fin.close();
	fout.close();
}
int main(){
	selectTest();
	return 0;
}
