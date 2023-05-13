// 取近似值.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
using namespace std;
int main()
{
	float a;
	cin >> a;
	//way1:
	//int b = a;//浮点数转为整数，自动向下取整
	//if (a - b >= 0.5) {
	//	cout << b + 1;
	//}
	//else {
	//	cout << b;
	//}

	//way2:=0.5实现四舍五入
	int b = a + 0.5;
	cout << b;
	return 0;
}
