#include<cstring>
#include<iostream>
using namespace std;
//�������У����ڽ��K����������Ϊ2k��������Ϊ2k+1 
const int max_deepth=20;
int s[1<<max_deepth];//<<Ҳ�����Ʋ������������1ʱ����ʾ2^n

int main()
{
	int D,I;
	
	int k;
	while(cin>>D>>I){
		int n=(1<<D)-1;//n���������� 
			memset(s,0,sizeof(s));//memset()����������ʼ��������****ͷ�ļ�Ϊ<cstring> 
		for(int i=0;i<I;i++)
		{
			k=1;
			for(;;)
			{
				s[k]=!s[k];//��С���߹�ʱ�����״̬�ı�
				k=s[k]?(2*k):(2*k+1); 
				if(k>n)break;//��ÿ��С�����ײ㾭����k�������n 
			}
		}
		cout<<k/2<<endl;//�������ǰ��Ҷ�ӱ�� 
	}
	
	return 0;
} 
