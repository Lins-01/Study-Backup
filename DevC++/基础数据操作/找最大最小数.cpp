#include<stdio.h>
int main()
{
	int n;
	int min=10000,max=0,sum=0;
	int tmp;
	scanf("%d",&n);
	for(int i=0;i<n;i++)
	{
		scanf("%d",&tmp);
		if(min>tmp)min=tmp;
		if(max<tmp)max=tmp;
		sum+=tmp;
	}
	printf("%d\n",max);
	printf("%d\n",min);
	printf("%d\n",sum);
	return 0;
}
