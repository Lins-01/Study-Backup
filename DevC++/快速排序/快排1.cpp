/*
	��������һ��������ѡ����һ��������С�� �������ŵ�����ߣ��������ķŵ����ұ� 
*/
#include<iostream>
using namespace std;
void swap(int &i,int &j)//c++������
{
	int tmp=i;
	i=j;
	j=tmp;
	return;
} 
void qsort(int arr[],int left,int right)//������Ҫдreturn 
{
	if(left>=right)return;
	int i,j,temp;
	i=left;
	j=right;
	temp=arr[left];
	while(i!=j)
	{
		while(arr[i]<temp&&i<j)//���ң�ֱ����߳��ִ��� temp��ֵ
		i++;
		while(arr[j]>temp&&i<j)//���ң�ֱ���ұ߳���С��temp   �м�Ҳ��������׼����ͬ����ᱻ�����ƶ����м� 
		j--; 
		if(i<j)swap(arr[i],arr[j]);
	}
	//�ѻ�׼���ƶ����м�
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
