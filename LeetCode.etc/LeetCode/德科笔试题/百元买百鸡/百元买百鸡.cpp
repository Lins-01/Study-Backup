// 百元买百鸡.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
using namespace std;

int main()
{
    //暴力枚举
    //三层循环，遍历xyz每种排列组合
    //判断满足两个方程式即可
    int i;
    cin >> i;
    for (int x = 0; x <= 100; x++) {
        for (int y = 0; y <= 100; y++) {
            for (int z = 0; z <= 100; z++) {
                //等式里有分数，两边同乘分母，去分数就是
                if ((15 * x + 9 * y + z) == 300 && (x + y + z) == 100) {
                    cout << x << " " << y << " " << z << endl;
                }
            }
        }
    }
    return 0;
}


