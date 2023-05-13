// 德科1面-赛龙-erase练习.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <bits/stdc++.h>
using namespace std;

int main()
{
    //按每个单词长度排序，用erase()。
    //练习下erase，
    //找到一个最短的输出，若是第一个则首字母变大写。
    //输出后，erase这个，直到vector.empty()返回true
    vector<string> str;
    string s;
    string temp = "";
    getline(cin, s);
    int len = s.length();
    for (int i = 0; i < s.length(); i++) {
        //把首字母变为小写 用tolower()
        if (i == 0) {
            s[0] = tolower(s[0]);
        }
        if (s[i] == ' ') {
            str.push_back(temp);
            temp = "";
            continue;
        }
        if (i == len - 1) {
            temp += s[i];
            str.push_back(temp);
            break;
        }
        temp += s[i];
    }
    for (int i = 0; i < str.size(); i++) {
        cout << str[i];
    }
    cout << endl;
    
    
    //找到size个当前最短的 并直接输出
    int size = str.size();
    for (int j = 0; j < size; j++) {
        //找到第j次最短的
        int max_i = 0;
        int l_est = 99999;
        for (int i = 0; i < str.size(); i++) {
            if (l_est > str[i].length()) {
                l_est = str[i].length();
                max_i = i;
            }
        }
        string temp = str[max_i];
        //第一个输出的为句首，首字母变大写
        if(j==0){
            temp[0] = toupper(temp[0]);  
        }
        //最后一个时，不用加空格
        if (j == size - 1) {
            cout << temp;
            break;
        }
        cout << temp << " ";
        //删除已输出的
        str.erase(str.begin() + max_i);
    }
    
    return 0;
}


