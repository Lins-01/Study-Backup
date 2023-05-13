#include<iostream>
using namespace std;
int main()
{
	int a = 10, b = 5;
	cout << "交换前：" << endl;
	cout << "a=" << a << "b=" << b << endl;
	a = a + b;//a变为之和
	b = a - b;//b变为了a
	a = a - b;//之和减去a得到b，赋值给
	cout << "交换后：" << endl;
	cout << "a=" << a << "b=" << b << endl;
	return 0;
}