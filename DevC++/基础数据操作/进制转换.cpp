#include<iostream>
#include<string>
#include<cmath>
#define MaxSize 10000
using namespace std;
//16ת8 ��ת10��ת8 ����ת2���㡣 ת10���ǰ�λ�ˡ�16������96ת10��9X16^1+6X16^0
//10ת8/16 ��ת������һ������8/16ȡ�༴�� 
int main()
{
	int n; 
	long long len;
	string num16;//cin���ܶ�д�ַ����� 
	long long tmp10=0; //�ۼ�һ��Ҫ�г���ֵ 
	long long  tmp16;
	long long tmp8;
	//��������
	cin>>n; 
	for(int j=0;j<n;j++)
	{
		
	
		cin>>num16; 
		//���ݴ���
		//16ת10
		len=num16.length();
		for(long i=len-1;i>=0;i--)
		{
			//����ĸ����
			if(num16[i]>='A'&&num16[i]<='Z')
			{
				tmp16=num16[i]-55;//��ĸת��Ϊ��Ӧʮ���� 
				tmp10=tmp10+(tmp16*pow(16,(len-1)-i));//�ۼ�һ��Ҫ�г���ֵ 
			} else{
				tmp16=num16[i]-48;
				tmp10=tmp10+(tmp16*pow(16,(len-1)-i));//�ۼ�һ��Ҫ�г���ֵ 
			}
			
		} 
	//	cout<<tmp10<<endl; //����16ת10 pass
	 
		//tmp10��Ϊ��Ӧ10������
		//10ת8   ��8ȡ��
		long long ten=1;
		long long  num8=0;
		while(tmp10!=0)
		{
			num8=num8+(tmp10%8)*ten;
			ten=ten*10;
			tmp10=tmp10/8;
		} 
		cout<<num8;
		cout<<endl;
	}
	 
	return 0;
}
