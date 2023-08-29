#include<iostream>
#define maxn 500
#define mod 1000000007
using namespace std;

int main()
{
	int f[maxn][maxn];
	int k,l;
	long long sum=0; 
	cin>>k>>l;
	for(int j=0;j<k;j++)//第一位可以填的数。因为是第一位所以都可以，记为一种
	{
		f[1][j]=1;
	}
	for(int i=2;i<=l;i++)//i表示位数 从第二位开始
		for(int j=0;j<k;j++)//j表示数字j放在首位
			for(int x=0;x<k;x++)
			{
				if(x!=j-1&&x!=j+1)//判断不为数字j的相邻数时加个数 
				{
					f[i][j]+=f[i-1][x];
					f[i][j]%=mod;//代表第i位放数字j的方法数 
				}
			} 
			
	for(int j=1;j<k;j++)//将最后一行累加，因为0不能作为数的开头，所以从j=1开始加
	{
		sum+=f[l][j];
		sum%=mod;
	} 
	cout<<sum<<endl;
	return 0;
}
