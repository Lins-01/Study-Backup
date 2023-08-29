#include<iostream>
#include<algorithm>
#define maxn 10000
using namespace std;
int num[maxn];
int main()
{
	int i=0,n;
	int len;
	int count=0;
	int f;
	long long result;
	cin>>n;
	for(i=0;i<n;i++)
	{
		cin>>num[i];
		if(num[i]<0)
		{
			count++;
			if(count==1)f=i;
			if(count>=2)
			{
				if(num[f]<0)num[f]=-num[f];
				
				num[i]=-num[i];
			}
		}
	}
	len=i;
	sort(num,num+len);
	result=num[i-3]*num[i-2]*num[i-1];
	cout<<result;
	return 0;
} 
