#include<iostream>
#define n 6
using namespace std;

int main()
{
	int a[n]={0,6,1,2,3,5};
	int mid=n/2;
	int max1=0,max2=0;
	for(int i =0;i<=n/2;i++)//�ҳ�ǰ�벿��max 
	{
		if(a[i]>max1)
			max1=a[i];
	}
	for(int j=n/2+1;j<n;j++)//��벿��max 
	{
		if(a[j]>max2)
			max2=a[j];
	}
	//�Ƚ�max1 max2
	if(max1>=max2)
	{
			cout<<"���Ԫ��"<<max1<<endl; 
	}
	else
	{
			cout<<"���Ԫ��"<<max2<<endl;
	}
	 
	return 0;
}
