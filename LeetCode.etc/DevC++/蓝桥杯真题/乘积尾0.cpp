#include<iostream>
using namespace std;

//�ж�β�����ٸ�0  β��Ϊ0����2��5����  ͳ�ƺ�������2��5�Ĵ���  ȡ��С�� ��Ϊβ��0����  
int main()
{
	int maxn=100;
	int a=0,b=0;//a��¼2���� b��¼5���� 
	int n;
	for(int i=0;i<10;i++)
	{
		for(int j=0;j<10;j++)
		{
			cin>>n;
			
			while(n%2==0)
			{
				a++;
				n/=2;
			}
			int tmp=n;
			while(tmp%5==0)
			{
				b++;
				tmp/=5;
			}
			//cout<<a<<' '<<b<<endl; 
		}
		cout<<a<<' '<<b<<endl; 
	}
	
	cout<<a<<' '<<b<<endl; //������������� ����  ���Բſ��� 
	
	return 0;
}
