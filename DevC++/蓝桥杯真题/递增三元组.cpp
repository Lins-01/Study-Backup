#include<iostream>
#define maxn 1000000
using namespace std;

//�����ȫ������ ������maxn�����޷�����  �һ�Ĭ�ϳ�ʼ��Ϊ0
//������main����������ջ�������ռ�ɶ�� 
	int A[maxn]; 
	int B[maxn]; 
	int C[maxn];
int main()
{
	long long N;
	long long count=0;
 
	//�������� 
	cin>>N;
	for(long long i=0;i<3;i++)
	{
		for(long long j=1;j<=N;j++)
		{
			
			if(i==0)cin>>A[j];
			if(i==1)cin>>B[j];
			if(i==2)cin>>C[j];
		}
	}
	
	for(long long i=1;i<=N;i++)
	{
		for(long long j=1;j<=N;j++)
		{
			for(long long k=1;k<=N;k++)
			{
				if(A[j]<B[j]&&B[j]<C[j])count++;
			}
		}
	} 
	 cout<<count<<endl;
	return 0;
}
