#include<iostream>
#define maxn 500
#define mod 1000000007
using namespace std;

int main()
{
	int f[maxn][maxn];
	int k,l;
	long long sum=0; 
	cin>>k>>l;
	for(int j=0;j<k;j++)//��һλ�������������Ϊ�ǵ�һλ���Զ����ԣ���Ϊһ��
	{
		f[1][j]=1;
	}
	for(int i=2;i<=l;i++)//i��ʾλ�� �ӵڶ�λ��ʼ
		for(int j=0;j<k;j++)//j��ʾ����j������λ
			for(int x=0;x<k;x++)
			{
				if(x!=j-1&&x!=j+1)//�жϲ�Ϊ����j��������ʱ�Ӹ��� 
				{
					f[i][j]+=f[i-1][x];
					f[i][j]%=mod;//�����iλ������j�ķ����� 
				}
			} 
			
	for(int j=1;j<k;j++)//�����һ���ۼӣ���Ϊ0������Ϊ���Ŀ�ͷ�����Դ�j=1��ʼ��
	{
		sum+=f[l][j];
		sum%=mod;
	} 
	cout<<sum<<endl;
	return 0;
}
