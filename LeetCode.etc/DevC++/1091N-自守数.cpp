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
	//�Ȳ�����<=0 
	int N=1;
	while(N<10)
	{
		int res=N*n*n;//����pow������pow��תΪintʱ�������1����� 
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
