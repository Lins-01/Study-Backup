#include<iostream>
#include<stdlib.h>
#include<math.h>
#define N 1200000
int vis[N]={0};//��¼�Ƿ���� 
using namespace std;
int main()
{
	
	long n;
	int z;
	char str[4];
	cin>>str;
	//��������ת������ 
	z=str[3]-'0';
	n=((str[0]-'0')*10+(str[1]-'0'))*pow(10,z);
	/*cout<<n<<endl;*/
	/*long dead=0;
	long cnt=0;
	long i=0;
	//û����ʱһֱѭ�� 
	while(dead<=n)
	{
		
		i++;
		if(i>n)i=1;//һ�ֺ��ͷ��ʼ 
		if(!vis[i])cnt++;//ģ�ⱨ����δ������
		if(cnt==2)
		{
			cnt=0;//��һ�ִ�0��ʼ�ظ����� 
			vis[i]=1;//���� 
			dead++;
			if(dead==n)
			{
				cout<<i<<endl; 
			}
		}
		
	}*/
	//��ͨ�ĳ�ʱ��ֱ���ϵݹ鹫ʽ 
	long f=0;
	for(int i=2;i<=n;i++)
	{
		f=(f+2)%i;
	}
	cout<<f+1<<endl;
	
	return 0;
} 
