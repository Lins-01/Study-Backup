#include<iostream>
using namespace std;

int main()
{
	long long n,N,max,tmp;
	
	cin>>N;//��������
	//��̰�ĵ�˼�룬����������������ÿ�ζ������Ž�
	for(n=N;n>1;n--)
	{
		//����������n����ʾ
		 //nΪ����ʱ�������С������Ϊn(n-1)(n-1)    ���ڵ�������Ȼ��һ��Ϊ������ ������������һ��Ϊ������ 
		 if(n%2!=0)
		 {
		 	tmp=n*(n-1)*(n-2);
		 	if(max<tmp)max=tmp;
		 }else{//nΪż��ʱ��  ����λ����һ�𲻶����� ���Զ�����һλ(n-1)(n-2)(n-3)��n(n-1)(n-3)��������һ�㣬ֻҪ��֤n��n-3�����ʾͺá�����������ֻ��Ϊ3 
		 	if(n%3==0)
		 	{
		 		tmp=(n-1)*(n-2)*(n-3);
		 		if(max<tmp)max=tmp;
		 	}else{
		 		tmp=n*(n-1)*(n-3);
		 		if(max<tmp)max=tmp;
		 	}
		 }
	} 
	cout<<max<<endl;
	return 0;
} 
