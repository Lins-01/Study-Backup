#include<iostream>
using namespace std;
int main()
{
	int a,b;
	int max,min;
	int res=1;
	cin>>a>>b;
	max=a;
	min=b;
	if(a<b)
	{
		max=b;
		min=a;
	} 
	for(int i=max/2;i>=1;i--)//��Լ������������һ��� 
	{
		if(max%i==0&&min%i==0)//˵��i��a��Լ�� //����Ƿ���b��Լ�� 
		{
				res=i;//��ǰ���Լ�� 
				break; //�Ӵ�С���ң����ٲ�ѯ���� 
		}
	}
	cout<<"���Լ��Ϊ��"<<res<<endl; 
	return 0;
} 
