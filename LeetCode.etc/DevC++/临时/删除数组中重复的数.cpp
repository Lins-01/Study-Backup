#include<iostream>
using namespace std;

void deletere(int a[],int N)
{
	int b[100]={0};
	int i,k,j;
	for(i=0;i<N;i++)
		b[a[i]]++;
	j=0;
	k=0;
	for(i=0;i<100;i++)
	{
		if(b[i]!=0)
		{
			if(b[i]==2)
			{
				//记录k k表示有几个重复元素
				k++; 
			}
			a[j]=i;//每次当b元素不等于0时，说明改元素是a的元素，若等于2则表示重复 
			//a中重复元素并未“真正”删除 只是放在了数组最后
			  
			j++;
		}
	}
	//这里输出还可以做到让数组重新排序，只是有点不太理解的题目的相对次序是什么意思 
	for(i=0;i<N-k;i++)
		cout<<a[i]<<endl;
}

int main()
{
	int r[7]={2,3,5,3,1,1,7}; 
	deletere(r,7);
	return 0;
}
