#include<iostream>
using namespace std;
int main()
{
	int a = 10, b = 5;
	cout << "����ǰ��" << endl;
	cout << "a=" << a << "b=" << b << endl;
	a = a + b;//a��Ϊ֮��
	b = a - b;//b��Ϊ��a
	a = a - b;//֮�ͼ�ȥa�õ�b����ֵ��
	cout << "������" << endl;
	cout << "a=" << a << "b=" << b << endl;
	return 0;
}