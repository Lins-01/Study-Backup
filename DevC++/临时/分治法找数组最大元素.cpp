#include<iostream>
#define n 6
using namespace std;

int main()
{
	int a[n]={0,6,1,2,3,5};
	int mid=n/2;
	int max1=0,max2=0;
	for(int i =0;i<=n/2;i++)//找出前半部分max 
	{
		if(a[i]>max1)
			max1=a[i];
	}
	for(int j=n/2+1;j<n;j++)//后半部分max 
	{
		if(a[j]>max2)
			max2=a[j];
	}
	//比较max1 max2
	if(max1>=max2)
	{
			cout<<"最大元素"<<max1<<endl; 
	}
	else
	{
			cout<<"最大元素"<<max2<<endl;
	}
	 
	return 0;
}
