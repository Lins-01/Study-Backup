#include<iostream>
#define maxn 1000000
using namespace std;

//定义成全部变量 不会受maxn限制无法运行  且会默认初始化为0
//定义在main函数里会调用栈？超过空间啥的 
	int A[maxn]; 
	int B[maxn]; 
	int C[maxn];
int main()
{
	long long N;
	long long count=0;
 
	//数据输入 
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
