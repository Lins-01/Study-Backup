#include<iostream>
#include<cstdlib>
#include<cstdio>
using namespace std;

double exp()
{
	//读入波兰表达式（即前缀表达式）
	char s[2];
	cin>>s;
	switch(s[0])
	{
		case '+':return exp()+exp();
		case '-':return exp()-exp();
		case '*':return exp()*exp();
		case '/':return exp()/exp();
		default :return atof(s);//atof(str) 把字符串转换成浮点型数
		break;
	} 
}

int main()
{
	printf("%lf",exp());
}
