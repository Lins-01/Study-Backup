#include<iostream>
#include<math.h>
using namespace std;

void Verify(int n);
int main()
{
	int M;
	int n;
	cin>>M;
	while(M>0)
	{
		M--;
		cin>>n;
		Verify(n);
	}
	return 0;
}

void Verify(int n)
{
	//先不考虑<=0 
	int N=1;
	while(N<10)
	{
		int res=N*n*n;//不用pow函数，pow在转为int时会出现少1的情况 
		/*cout<<"res="<<res<<endl;*/
		int temp=res;
		int n1=n;
		while(n>0)
		{
			if(n1%10!=temp%10) break;
			n1=n1/10;
			temp=temp/10;
			if(n1==0)
			{
				cout<<N<<" "<<res<<endl;
				return;
			}
		}
		N++; 
		
	}
	cout<<"No"<<endl;
}
