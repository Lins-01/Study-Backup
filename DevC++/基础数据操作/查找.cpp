#include<stdio.h>
#define MaxSize 1001
int main()
{
	int n;
	int num[MaxSize];
	scanf("%d",&n);
	num[0]=-1;//从1开始计算 
	for(int i=1;i<=n;i++)
	{
		scanf("%d",&num[i]);
	} 
	int search;
	scanf("%d",&search);
	int i;
	for(i=1;i<=n;i++)
	{
		if(search==num[i])
		{
			printf("%d",i);
			break;
		}
	} 
	if(i==n+1)//未找到
	{
		printf("-1");
	} 
	return 0;
}
