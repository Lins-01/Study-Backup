#include<iostream>
#include<algorithm>
#define MaxSize 10000
using namespace std;

//erro stray 241 in program  һ�㶼�������ķ��ŵ��µ�  �ҵ� ����ɾ���������� 
bool cmp(long a,long b)
{
	return a>b;//���� 
}
//sortֻ�ܶ������ȫ��Ԫ�ؽ�������,���Ҫָ�����ȣ�����һ���������洢 
int main()
{
	int n,m,l,r,k;
	long num[MaxSize];//����Ԫ�ش�1��ʼ��ǣ���Ӧ����Ԫ���±�+1 
	//�������� 
	cin>>n;
	for(int i=0;i<n;i++)
	{
		cin>>num[i];
	}
	cin>>m;
	for(int i=0;i<m;i++)//ִ��m�β��� 
	{
		long arrTmp[MaxSize]; //sortֻ�ܶ������ȫ��Ԫ�ؽ�������,���Ҫָ�����ȣ�����һ���������洢 
		cin>>l>>r>>k;
		//�м����鸳ֵ 
		int len=r-l+1;
		for(int j=0;j<len;j++)
		{
			arrTmp[j]=num[j+l-1];//���и����Ĵ�С��ϵӦ�ûᱣ֤����� 
		}
		//�м����齵������
		sort(arrTmp,arrTmp+len,cmp); 
		
		/*for(int q=0;q<len;q++)//���ʱ�������arrTmp 
		{
			cout<<arrTmp[q]<<' ';
		} 
		cout<<endl;*/
		
		//�����K�����
		cout<<arrTmp[k-1]<<endl;
	}
	return 0;
}
