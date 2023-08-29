#include<iostream>
using namespace std;
int main()
{
	int a,b;
	int max,min;
	int res=1;
	cin>>a>>b;
	max=a;
	min=b;
	if(a<b)
	{
		max=b;
		min=a;
	} 
	for(int i=max/2;i>=1;i--)//公约数不会比自身的一半大 
	{
		if(max%i==0&&min%i==0)//说明i是a的约数 //检测是否是b的约数 
		{
				res=i;//当前最大公约数 
				break; //从大到小查找，减少查询次数 
		}
	}
	cout<<"最大公约数为："<<res<<endl; 
	return 0;
} 
