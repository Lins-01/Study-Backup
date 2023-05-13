#include<cstring>
#include<iostream>
using namespace std;
//二叉树中，对于结点K，其左子树为2k，右子树为2k+1 
const int max_deepth=20;
int s[1<<max_deepth];//<<也是左移操作，当左边是1时，表示2^n

int main()
{
	int D,I;
	
	int k;
	while(cin>>D>>I){
		int n=(1<<D)-1;//n是最大结点个数 
			memset(s,0,sizeof(s));//memset()函数用来初始化，清零****头文件为<cstring> 
		for(int i=0;i<I;i++)
		{
			k=1;
			for(;;)
			{
				s[k]=!s[k];//当小球走过时，结点状态改变
				k=s[k]?(2*k):(2*k+1); 
				if(k>n)break;//当每个小球从最底层经过后k都会大于n 
			}
		}
		cout<<k/2<<endl;//输出出界前的叶子编号 
	}
	
	return 0;
} 
