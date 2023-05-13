#include<iostream>
#define maxn 6
using namespace std;
int main()
{
	int arr[maxn]={9,8,7,6,5,4};
	
	int search=7;
	for(int i=0;i<6;i++)
	{
		if(arr[i]==search)
		{
			cout<<"查询结果为:"<<i<<endl;
		}
	}
	
	return 0;
}
