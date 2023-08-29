// 字符串反转.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
#include<bits/stdc++.h>
using namespace std;
int main() {
	//way1:reverse 函数
	/*string str;
	cin >> str;
	reverse(str.begin(), str.end());
	cout << str;
	return 0;*/

	char c[1000];
	int lens;
	//gets(c);   牛客网这里不知道怎么了，gets报错。无法识别，别人好像也是。
	scanf("%s", c);
	lens = strlen(c);//计算字符串长度，找到'\0'为止（不包括0
	//way2:直接逆向输出
	/*for (int i = lens-1; i > -1; i--)
	{
		cout << c[i];
	}*/
	int i, j;
	int temp;
	//way3:从两边向中间两两交换进行翻转。
	for (i = 0, j = lens - 1; i <= lens / 2; i++, j--) {
		temp = c[i];
		c[i] = c[j];
		c[j] = temp;
	}
	puts(c);
	return 0;
}