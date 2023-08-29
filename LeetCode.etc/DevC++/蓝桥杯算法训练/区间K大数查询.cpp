#include<iostream>
#include<algorithm>
#define MaxSize 10000
using namespace std;

//erro stray 241 in program  一般都是有中文符号导致的  找到 或者删了重新输入 
bool cmp(long a,long b)
{
	return a>b;//降序 
}
//sort只能对数组对全部元素进行排序,如果要指定长度，用另一个数组来存储 
int main()
{
	int n,m,l,r,k;
	long num[MaxSize];//序列元素从1开始标记，对应数组元素下标+1 
	//数据输入 
	cin>>n;
	for(int i=0;i<n;i++)
	{
		cin>>num[i];
	}
	cin>>m;
	for(int i=0;i<m;i++)//执行m次查找 
	{
		long arrTmp[MaxSize]; //sort只能对数组对全部元素进行排序,如果要指定长度，用另一个数组来存储 
		cin>>l>>r>>k;
		//中间数组赋值 
		int len=r-l+1;
		for(int j=0;j<len;j++)
		{
			arrTmp[j]=num[j+l-1];//题中给定的大小关系应该会保证不溢出 
		}
		//中间数组降序排序
		sort(arrTmp,arrTmp+len,cmp); 
		
		/*for(int q=0;q<len;q++)//查错时用来输出arrTmp 
		{
			cout<<arrTmp[q]<<' ';
		} 
		cout<<endl;*/
		
		//输出第K大的数
		cout<<arrTmp[k-1]<<endl;
	}
	return 0;
}
