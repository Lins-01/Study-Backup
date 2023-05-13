// 完全数计算.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
using namespace std;
int main()
{
    int n;
    int count = 0;
    cin >> n;
    while (n > 1)//再根据找到的最小完全数改下
    {
        int sum = 1;
        for (int i = 2; i <= n / 2; i++) {//求因数，时，小于等于一半
            if (n % i == 0) {
                sum = sum + i;
               // cout << "11111asdad" << endl;
            }
        }
        if (sum == n) {
            count++;
        }
        n--;
    }
    cout << count;
    return 0;
}

