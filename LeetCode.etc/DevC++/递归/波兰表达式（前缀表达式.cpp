#include<iostream>
#include<cstdlib>
#include<cstdio>
using namespace std;

double exp()
{
	//���벨�����ʽ����ǰ׺���ʽ��
	char s[2];
	cin>>s;
	switch(s[0])
	{
		case '+':return exp()+exp();
		case '-':return exp()-exp();
		case '*':return exp()*exp();
		case '/':return exp()/exp();
		default :return atof(s);//atof(str) ���ַ���ת���ɸ�������
		break;
	} 
}

int main()
{
	printf("%lf",exp());
}
