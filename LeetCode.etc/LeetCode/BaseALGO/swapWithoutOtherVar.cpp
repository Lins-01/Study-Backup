#include<iostream>
using namespace std;
int main()
{
	int a = 10, b = 5;
	cout << "Please 输入" << endl;
	cout << "a=" << a << "b=" << b << endl;
	a = a + b;
	b = a - b;
	a = a - b;
	cout << "2" << endl;
	cout << "a=" << a << "b=" << b << endl;
	return 0;
}