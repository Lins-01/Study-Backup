#include <bits/stdc++.h>
#include <vector>
using namespace std;
//回忆c++

int main()
{
    string s1;
    cin>>s1;
    s1.push_back('c');
    s1=s1+'c';
    for(int i=0;i<s1.size();i++)
    {
        if(s1[i]=='c')
            cout<<"ccc"<<endl;
    }
    cout<<s1.find("c",1)<<endl;

    vector<int> v1;
    vector<string> s2;
    for(int i=0;i<10;i++)
    {
        v1.push_back(i*2);
        
        s2.push_back("c");
    }
    for(int i=0;i<10;i++){
        cout<<v1[i]<<endl;
        cout<<s2[i]<<endl;
    }
    v1.pop_back();
    //v1.end()指向最后一个元素的下一个。v1.end()-1才是最后一个元素
    //v1.front()和.back()表示首尾元素的引用
    cout<<v1.back()<<endl;
    return 0;
}