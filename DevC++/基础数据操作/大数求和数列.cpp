#include<stdio.h>
int main()
{
	long long n;
	long long Smul;
	scanf("%ld",&n);
	Smul=((1+n)*n)/2;
	printf("%I64d\n",Smul);//���longlong���͵��� 
	return 0;
}
