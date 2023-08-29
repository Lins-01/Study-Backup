#include<iostream>
#include<string>
#include<sstream>//使用string流时的头文件，即（istringstream,stringstream,ostringstream） 
#define maxsize 10001
using namespace std;


void strToint(string &str,int &num)
{
	stringstream ss;
	ss<<str;
	ss>>num;
}
int main()
{
	int line;
	int data[maxsize];
	int index=0;//记录date的下标，最后是总输入元素个数 
	cin>>line;
	cin.get();//在输入一个数后，要紧接着输入时，一定要用getchar（）读出缓冲区的回车 
	for(int i=0;i<line;i++)
	{
		string s;
		getline(cin,s);
		istringstream iss(s);
		string tmp;
		while(getline(iss,tmp,' '))
		{
			strToint(tmp,data[index++]);
		}
	}
	for(int j=0;j<index;j++)
	{
		cout<<data[j]<<" ";
	}
	 
	
	return 0;
}
