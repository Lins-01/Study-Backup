#include<stdio.h>	

int main()
{
	int a,b,c;
	int n;
	scanf("%d",&n); 
	for(a=1;a<=9;a++)
		for(b=0;b<=9;b++)
			for(c=0;c<=9;c++)
			{
				if(a+b+c+b+a==n)
					printf("%d%d%d%d%d\n",a,b,c,b,a);
			}
	for(a=1;a<=9;a++)
		for(b=0;b<=9;b++)
			for(c=0;c<=9;c++)
			{
				if(a+b+c+c+b+a==n)
					printf("%d%d%d%d%d%d\n",a,b,c,c,b,a);
			} 
	return 0;
}
