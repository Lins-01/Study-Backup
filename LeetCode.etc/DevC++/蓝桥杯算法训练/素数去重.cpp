#include<iostream>
//memset ֻ�ܽ������ʼ��Ϊ�ַ��� ������������ֻ�ܳ�ʼ��Ϊ-1��0 
using namespace std;

int issushu(long long n)
{
	long long i;
	for(i=2;i*i<=n;i++)
	{
		if(n%i==0)return 0;
	}
	return 1;
}
int main()
{
	long long N,i,mulp=1;
	//long long num[maxn];
	//memset(num,1,sizeof(num));  ������������Ҫ��ʼ����memsetֻ�ܲ�����-1��0  ������ֻ��ѭ����ֵ
	/*for(int j=0;j<=maxn;j++) 
	{
		num[j]=1;
	}*///����Ҳ����һ������Ŵ������������ӣ�ֱ��������� 
	cin>>N;
	/*if(issushu(N))num[0]=N;//�������� ���ÿ��Ǳ��� 
	else{
	
		for(i=2;i<=N/2;i++)
		{
			if(N%i==0)
			{
				if(issushu(i))
				{
					num[k]=i;
					k++;
				}
			}
		}
	}*/
	for(i=2;i*i<=N;i++)
		{
			if(N%i==0)
			{
				if(issushu(i))
				{
					mulp*=i;
					while(N%i==0)N/=i;//������������ ȥ������i��N�����ӣ�Ҳ��ɸѡ�������� 
				}
			}
		}
		mulp*=N;//N����ֵҲ�ǳ�ʼN���������� 
		
/*	mulp=1;//����Ҳ����һ������Ŵ������������ӣ�ֱ��������� 
	for(i=0;i<=k;i++)
	{
		mulp=mulp*num[i];
	}*/
	cout<<mulp<<endl;
	return 0;
}
