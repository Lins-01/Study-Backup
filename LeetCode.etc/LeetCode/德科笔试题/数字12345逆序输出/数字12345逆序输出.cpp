#include<iostream>
using namespace std;
int main()
{
    int n;
    int c[1000];
    int i = 0;
    int temp;
    cin >> n;
    //way1：取余存储每次的最低位。最后遍历再输出。麻烦->way2
    /*while (n > 10) {

        c[i] = n % 10;
        n = n / 10;
        i++;

    }
    c[i] = n;
    for (int j = 0; j <= i; j++) {
        cout << c[j];
    }*/
    //way2:同样取余，但其实不用保存，再存储的。直接输出每次的最后一位即可
    while (n > 10) {
        cout << n % 10;
        n = n / 10;
    }
    cout << n;

    return 0;
}