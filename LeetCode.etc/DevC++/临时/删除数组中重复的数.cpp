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
				//��¼k k��ʾ�м����ظ�Ԫ��
				k++; 
			}
			a[j]=i;//ÿ�ε�bԪ�ز�����0ʱ��˵����Ԫ����a��Ԫ�أ�������2���ʾ�ظ� 
			//a���ظ�Ԫ�ز�δ��������ɾ�� ֻ�Ƿ������������
			  
			j++;
		}
	}
	//�������������������������������ֻ���е㲻̫������Ŀ����Դ�����ʲô��˼ 
	for(i=0;i<N-k;i++)
		cout<<a[i]<<endl;
}

int main()
{
	int r[7]={2,3,5,3,1,1,7}; 
	deletere(r,7);
	return 0;
}
