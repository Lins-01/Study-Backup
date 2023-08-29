#include<iostream>
#include<string>
#include<algorithm>
using namespace std;

const int maxcol=60;
const int maxn=100+5;
string filenames[maxn];


//输出函数，长度不足时输出字符extra
void Print(const string &s,int len,char extra)
{
	cout<<s;
	for(int i=0;i<len-s.length();i++)
	{
		cout<<extra;
	}
} 

int main()
{
	int n;
	//输入并存储n个文件名 
	while(cin>>n)
	{//当成功输入n后执行下面语句。用while更易理解？ 
		int M=0;//记录最长文件名字符数，一定要初始化为0 
		for(int i=0;i<n;i++)
		{
			cin>>filenames[i];
			M=max(M,(int)filenames[i].length());//直接用STL中的max函数，保证M为最长的字符数 
		} 
			//计算列数cols和行数rows
			int cols=(maxcol-M)/(M+2)+1,row=(n-1)/cols+1;//?????
			Print("",60,'-');//按格式输出第一行--------- 
			cout<<"\n";
			sort(filenames,filenames+n);//排序 
			for(int r=0;r<row;r++)
			{
				for(int c=0;c<cols;c++)
				{
					int idx=c*row+r;//计算输出的是第几个元素对应的下标 
					if(idx<n)
						Print(filenames[idx],c==cols-1?M:M+2,' '); 
				}
			cout<<"\n";
		}
	}
	return 0;
} 















