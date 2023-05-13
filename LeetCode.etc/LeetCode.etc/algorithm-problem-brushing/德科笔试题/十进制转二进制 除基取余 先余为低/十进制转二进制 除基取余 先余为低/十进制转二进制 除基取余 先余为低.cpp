// 十进制转二进制 除基取余 先余为低.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
using namespace std;
int main()
{
    int n;
    int count = 0;
    cin >> n;
    while (n != 0) {//商为0则为结束
        if (n % 2 == 1) {
            count++;
        }
        n = n / 2;//比基数小的值，除以基数，余它自己，然后商为0；商为0则为结束
    }
    cout << count;
}
