#include<iostream>
using namespace std;

int f(int m,int n)
{
	//��n>mʱ���� ���Ӷ���ƻ�� ������f(m,m)��ͬ 
	if(n>m)return f(m,m);
	if(m==0)return 1;//û��ƻ�� ��ֻ��00000һ�ַŷ� 
	if(n==0)return 0;//û�����ӣ�û�зŷ� ���Է���0 
	return f(m,n-1)+f(m-n,n);//�п��̷ŷ���+�޿����� 
}



int main()
{
	int t,m,n;//mƻ������ nΪ������ 
	cin>>t;
	while(t--)//Ҫִ��t�ο���������whileд  ����һЩ 
	{
		cin>>m>>n;
		cout<<f(m,n)<<endl;
	}
	return 0;
} 
