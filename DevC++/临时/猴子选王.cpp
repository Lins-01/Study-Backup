#include<iostream>
#include<stdlib.h>
#include<math.h>
#define N 1200000
int vis[N]={0};//记录是否出列 
using namespace std;
int main()
{
	
	long n;
	int z;
	char str[4];
	cin>>str;
	//输入数据转换处理 
	z=str[3]-'0';
	n=((str[0]-'0')*10+(str[1]-'0'))*pow(10,z);
	/*cout<<n<<endl;*/
	/*long dead=0;
	long cnt=0;
	long i=0;
	//没死完时一直循环 
	while(dead<=n)
	{
		
		i++;
		if(i>n)i=1;//一轮后从头开始 
		if(!vis[i])cnt++;//模拟报数（未出列中
		if(cnt==2)
		{
			cnt=0;//下一轮从0开始重复报数 
			vis[i]=1;//出列 
			dead++;
			if(dead==n)
			{
				cout<<i<<endl; 
			}
		}
		
	}*/
	//普通的超时，直接上递归公式 
	long f=0;
	for(int i=2;i<=n;i++)
	{
		f=(f+2)%i;
	}
	cout<<f+1<<endl;
	
	return 0;
} 
