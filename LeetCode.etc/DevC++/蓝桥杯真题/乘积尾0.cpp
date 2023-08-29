#include<iostream>
using namespace std;

//判断尾数多少个0  尾数为0是由2或5得来  统计含有因子2或5的次数  取较小的 即为尾数0个数  
int main()
{
	int maxn=100;
	int a=0,b=0;//a记录2个数 b记录5个数 
	int n;
	for(int i=0;i<10;i++)
	{
		for(int j=0;j<10;j++)
		{
			cin>>n;
			
			while(n%2==0)
			{
				a++;
				n/=2;
			}
			int tmp=n;
			while(tmp%5==0)
			{
				b++;
				tmp/=5;
			}
			//cout<<a<<' '<<b<<endl; 
		}
		cout<<a<<' '<<b<<endl; 
	}
	
	cout<<a<<' '<<b<<endl; //最后结果看不到， 闪退  调试才看见 
	
	return 0;
}
