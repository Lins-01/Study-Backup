#include<iostream>
using namespace std;

int f(int m,int n)
{
	//当n>m时，即 盘子多于苹果 次数和f(m,m)相同 
	if(n>m)return f(m,m);
	if(m==0)return 1;//没有苹果 那只有00000一种放法 
	if(n==0)return 0;//没有盘子，没有放法 所以返回0 
	return f(m,n-1)+f(m-n,n);//有空盘放法数+无空盘数 
}



int main()
{
	int t,m,n;//m苹果个数 n为盘子数 
	cin>>t;
	while(t--)//要执行t次可以这样用while写  方便一些 
	{
		cin>>m>>n;
		cout<<f(m,n)<<endl;
	}
	return 0;
} 
