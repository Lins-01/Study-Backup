#include<iostream>
using namespace std;
//汉字16*16表示 所以可以用32个字节表示 一个字节8位 32*8=16*16 
// 负数求其对应的二进制数  先直接对负数进行按位取反得到反码再加1即可 
	int num[8];//存储二进制数  全局 
void transf(int n)
{
	for(int i=7;i>=0;i--)
	{
		num[i]=n%2;
		n/=2;
	}
	return;
} 
void Print()
{
	for(int i=0;i<=7;i++)//打印的时候顺序输出数组即可 
	{
		if(num[i]==0)
		cout<<' ';
		else{
			cout<<'*';
		}
	}
}
int main()
{
	int col=4;//4行二进制数
	int n;//每个数

	for(int i=0;i<col;i++)
	{
		for(int j=0;j<8;j++)//每行读8个数
		{
			cin>>n;
			if(n<0)
			{
				/*n=-1*n;//转换为相反数 求其原码（二进制数）的 补码 （补码即反码末尾加1）  要把1变成0直接逻辑非即可 ！1=0  ~是按位操作 
				//先求原码
				transf(n);
				for(int i=0;i<7;i++)
				{
					num[i]=!num[i]; 
				} 
				num[7]*/
				//n=(-n^255)+1;//求负数二进制数的转换过程  转换后直接求对应原码就是负数的原码了
				//进阶的写法
				n=256+n; 
				transf(n); 
				Print();
			}
			else{
				transf(n);
				Print();
			}
			//控制输出格式 每俩次循环为一行
			if(j%2!=0)//奇数i换行
			{
				cout<<endl;
			} 
			 
		}
		 
	}
	return 0;
}
