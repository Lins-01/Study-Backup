#include<stdio.h>
#define MaxSize 1000 //定义一行最多的数 
int main()
{
	int n;//n行
	scanf("%d",&n);//数据输入 
	long long C[100 ][MaxSize]={0};//错误 这里不知道把多少行数据初始化为0  所以要先给定一个行数 也不可以太大 
	//memset(C,0,sizeof(C));
	//对每行进行操作 
	for(int i=0;i<n;i++)//共n行 
	{
		C[i][0]=1;//先给定因为使用数组存数，第一位就是0，如果0-1就越界了  正好每行第一个元素为1  
		for(int j=1;j<=i;j++)//从每行的第二个元素开始求  该行为第i+1行，元素个数为i+1个  但第一个元素已经给定过为1 所以j<=i+1-1 
		{
			C[i][j]=C[i-1][j-1]+C[i-1][j];
		} 
	} 
	for(int i=0;i<n;i++)
	{
		int j=0;
		while(C[i][j]!=0)
		{
			if(C[i][j+1]==0){
				printf("%d\n",C[i][j]);
				break;
			}
			printf("%d ",C[i][j]);
			j++; 
		}
	}
	return 0;
}
