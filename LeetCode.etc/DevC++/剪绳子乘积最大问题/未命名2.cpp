#include<iostream>
#define maxsize 10000
using namespace std;
int maxcut(int n)
{
	int value[maxsize];
	//用数组存储绳子乘积最大值,为长度大于3的绳子服务 
	value[0]=0;
	value[1]=1;
	value[2]=2;
	value[3]=3; 
	if(n<=1)return 0;
	if(n==2)return 1;
	if(n==3)return 2;
	int max=0,val=0;
	for(int i=4;i<=n;i++)
	{
		max=0;
		for(int j=1;j<i/2;j++)
		{
			val=value[j]*value[i-j];
			max=val>max?val:max;
		}
		value[i]=max;
	}
	cout<<value[n]<<endl;
} 
int main()
{
	int n=10;
	maxcut(n);
	return 0;
}
