#include<iostream>
#include<string>
#include<sstream>//ʹ��string��ʱ��ͷ�ļ�������istringstream,stringstream,ostringstream�� 
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
	int index=0;//��¼date���±꣬�����������Ԫ�ظ��� 
	cin>>line;
	cin.get();//������һ������Ҫ����������ʱ��һ��Ҫ��getchar���������������Ļس� 
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
