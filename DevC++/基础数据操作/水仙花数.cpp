#include<stdio.h>
int main()
{
	int n=100;
	int tmp;
	int a,b,c;
	tmp=n;
	while(n<=999)
	{
		tmp=n;
		c=tmp%10;
		tmp=tmp/10;
		b=tmp%10;
		tmp=tmp/10;
		a=tmp%10;
		if(n==a*a*a+b*b*b+c*c*c)printf("%d\n",n); 
		n++;
	}
	return 0;
}
