#include<iostream>
#include<sstream>
#include<string>
using namespace std;

int main()
{
	string line;
	int sum=0,tmp;//tmp��Ϊ��ʱ�����洢lineת������ÿ���� 
	while(getline(cin,line))//�Ȱ�ÿ����Ϊ�ַ��������ַ���line�У���Ϊ�ǰ����ո��һ������ݣ�����ֱ����sum���㡣 
	{
		
		stringstream ss(line);//��string lineת��Ϊ�� 
		while(ss>>tmp)sum+=tmp;//��lineת������ss���뵽tmp ����Ϊ�����൱��cin>>tmp�����Զ�ʶ���������ͣ����Կո񲻻����뵽tmp 
		cout<<sum<<endl;
	}
	cout<<sum<<endl;  //15 8 E:\DevC++\C++����һ�д��ո���ַ���\δ����2.cpp [Error] 'sum' was not declared in this scope ???
	return 0;
}
