/*
	快速排序，一组数，先选任意一个数，将小于 它的数放倒他左边，大于它的放到他右边 
*/
#include<iostream>
using namespace std;
void swap(int &i,int &j)//c++的引用
{
	int tmp=i;
	i=j;
	j=tmp;
	return;
} 
void qsort(int arr[],int left,int right)//函数都要写return 
{
	if(left>=right)return;
	int i,j,temp;
	i=left;
	j=right;
	temp=arr[left];
	while(i!=j)
	{
		while(arr[i]<temp&&i<j)//查找，直到左边出现大于 temp的值
		i++;
		while(arr[j]>temp&&i<j)//查找，直到右边出现小于temp   中间也会出现与基准项相同的项，会被不断移动到中间 
		j--; 
		if(i<j)swap(arr[i],arr[j]);
	}
	//把基准项移动到中间
	 if(i<j)
	 {
	 	arr[left]=arr[i];
	 	arr[i]=temp;
	 }
	 qsort(arr,left,i-1);
	 qsort(arr,i+1,right);
	return;
} 

int main()
{
	int arr[]={-1,2-4,0,4,-3,-7};
	qsort(arr,0,7);
	for(int i=0;i<7;i++)
	{
		cout<<arr[i]<<" ";
	}
	return 0;
}
