#include<stdio.h>
#define flag 10000
#define N 100000000
int F[N];
int main()
{
	int n;
	scanf("%d",&n);
	
	F[1]=1;
	F[2]=1;
	F[0]=0; 
	if(n>2){
	for(int i=3;i<=n;i++){
		F[i]=(F[i-1]+F[i-2])%flag;
		}
	}
	printf("%d",F[n]);
	return 0; 
} 

