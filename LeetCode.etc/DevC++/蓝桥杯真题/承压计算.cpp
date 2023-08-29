#include<iostream>
#include<cstdio>
#include<algorithm>
#define maxn 435   //435¸ö 
int num[33];
int value[maxn+30+1];

using namespace std;
int main()
{
	for(int i=1;i<maxn+1;i++)
	{
		cin>>value[i];
	}
	
	int k=1,m=2;
	for(int i=1;i<maxn+1;i++)
	{
		value[i+k]=value[i]/2;
		value[i+k+1]=value[i]/2;
		if(i==i+m)
		{
			m++;
			k++;
		}
	}
	int f=0;
	for(int i=435;i<466;i++)
	{
		num[f]=value[i];
		f++;
	}
	sort(num,num+33);
	printf("%d",num[32]);
	return 0;
} 
