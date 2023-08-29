#include<iostream>
#include<string>
#include<cmath>
#define MaxSize 10000
using namespace std;
//16转8 先转10再转8 比先转2方便。 转10都是按位乘。16进制数96转10即9X16^1+6X16^0
//10转8/16 与转二进制一样。对8/16取余即可 
int main()
{
	int n; 
	long long len;
	string num16;//cin不能读写字符数组 
	long long tmp10=0; //累加一定要有初试值 
	long long  tmp16;
	long long tmp8;
	//数据输入
	cin>>n; 
	for(int j=0;j<n;j++)
	{
		
	
		cin>>num16; 
		//数据处理
		//16转10
		len=num16.length();
		for(long i=len-1;i>=0;i--)
		{
			//对字母处理
			if(num16[i]>='A'&&num16[i]<='Z')
			{
				tmp16=num16[i]-55;//字母转换为对应十进制 
				tmp10=tmp10+(tmp16*pow(16,(len-1)-i));//累加一定要有初试值 
			} else{
				tmp16=num16[i]-48;
				tmp10=tmp10+(tmp16*pow(16,(len-1)-i));//累加一定要有初试值 
			}
			
		} 
	//	cout<<tmp10<<endl; //测试16转10 pass
	 
		//tmp10即为对应10进制数
		//10转8   除8取余
		long long ten=1;
		long long  num8=0;
		while(tmp10!=0)
		{
			num8=num8+(tmp10%8)*ten;
			ten=ten*10;
			tmp10=tmp10/8;
		} 
		cout<<num8;
		cout<<endl;
	}
	 
	return 0;
}
