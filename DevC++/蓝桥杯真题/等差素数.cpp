#include<iostream>
using namespace std;

//look for all prime <n
#define n 100000
	 //store
long prime[n];
bool isprime(long N)
{
	
	for(long i=2;i*i<=N;i++)
	{
		if(N%i==0)return false;
	}
	if(N==2)return false;
	return true;
}
int main()
{
	
	 long k=0;
	 for(long i=1;i<=n;i++)
	 {
	 	if(isprime(i)){
	 		prime[k]=i;
	 		cout<<prime[k]<<endl;
	 		k++;
	 	}
	 }
	 for(long i=1;i<300;i++)//枚举公差 
	 {
	 	long j;
	 	for(j=0;j<k;j++)//遍历素数 是否从j项按公差i呈10次等差 
	 	{
	 		long x; 
	 		//判断是否满足往后10项等差
			 for( x=j+1;x<=j+9;x++)
			 {
			 	if((prime[x]-prime[j])!=(x-j)*i)
			 	{
			 		break;
			 	}
				
			 }
			 if(x==(j+10))
				 {
				 	cout<<i<<endl;
				 	return 1;
				 }
				 /*for(x=1;x<=10;x++)
				 {
				 	if(!isprime(prime[j]+i*x))break;
				 }
				 if(x==11)
				 {
				 	cout<<i<<endl;
				 	return 1;
				 }*/
			 
	 	}
	 }
	return 0;
}
