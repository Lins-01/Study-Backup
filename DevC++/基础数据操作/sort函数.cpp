#include<iostream>
#include<algorithm>
#define MaxSize 201
using namespace std;

int main()
{
	int n;
	int num[MaxSize];
	cin>>n;
	for(int i=0;i<n;i++)
	{
		cin>>num[i];
	}
	sort(num,num+n);
	for(int i=0;i<n;i++)
	cout<<num[i]<<' ';
	cout<<endl;
	return 0;
} 

