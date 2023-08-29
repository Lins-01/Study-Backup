// 1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。

//示例 1:
//输入: s = "abcabcbb"
//输出 : 3
//解释 : 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
//
//示例 2 :
//    输入 : s = "bbbbb"
//    输出 : 1
//    解释 : 因为无重复字符的最长子串是 "b"，所以其长度为 1。
//
//    示例 3 :
//    输入 : s = "pwwkew"
//    输出 : 3
//    解释 : 因为无重复字符的最长子串是 "wke"，所以其长度为 3。


#include <bits/stdc++.h>
using namespace std;
int main()
{
    vector<string> v;
    string s;
    
    cin >> s;
    for (int i = 0; i < s.size(); i++) {
        string str = "";
        str += s[i];
        for (int j = i + 1; j < s.size(); j++) {
            char c = s[j];
            //cout << "find: " << str.find(c)<<endl;
            if (str.find(c)!=-1) {
                
                v.push_back(str);
                break;
            }
            str =str+ s[j];
        }
    }
    int len = 0;
    for (int i = 0; i < v.size(); i++) {
        if (len < v[i].length()) {
            len = v[i].length();
        }
    }
    cout << len;
    return 0;
}


