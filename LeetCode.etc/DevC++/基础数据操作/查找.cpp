#include<stdio.h>
#define MaxSize 1001
int main()
{
	int n;
	int num[MaxSize];
	scanf("%d",&n);
	num[0]=-1;//��1��ʼ���� 
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
	if(i==n+1)//δ�ҵ�
	{
		printf("-1");
	} 
	return 0;
}
