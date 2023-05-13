#include<iostream>
//memset 只能将数组初始化为字符， 对于整型数组只能初始化为-1或0 
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
	//memset(num,1,sizeof(num));  对整型数组想要初始化用memset只能操作到-1或0  其它数只能循环赋值
	/*for(int j=0;j<=maxn;j++) 
	{
		num[j]=1;
	}*///这里也不用一个数组才存数所有素因子，直接求积即可 
	cin>>N;
	/*if(issushu(N))num[0]=N;//素数因子 不用考虑本身 
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
					while(N%i==0)N/=i;//减少运算量， 去掉包含i的N的因子，也就筛选了素数！ 
				}
			}
		}
		mulp*=N;//N最后的值也是初始N的素数因子 
		
/*	mulp=1;//这里也不用一个数组才存数所有素因子，直接求积即可 
	for(i=0;i<=k;i++)
	{
		mulp=mulp*num[i];
	}*/
	cout<<mulp<<endl;
	return 0;
}
