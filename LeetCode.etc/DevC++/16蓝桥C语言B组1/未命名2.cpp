#include<stdio.h>
int main()
{
	int n,sum;
	sum=0;
	for(n=1;n<=100;n++)
	{
		sum=sum+n;
	}
	printf("%d",sum);
	getchar();
	return 0;
}
