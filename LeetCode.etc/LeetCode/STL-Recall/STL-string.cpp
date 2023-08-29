#include <bits/stdc++.h>
//不用写 string头文件。但不是高亮不好看，自己就写下吧
//#include <string> 写了也不高亮
using namespace std;
int main()
{
    //#######################string的定义和初始化
    string s1;//初始化字符串，空字符串
    string s2=s1;//不可以这么写！，要复制s2=s1输入后的值，应写在cin>>s1之后
    string s3="I am JingleQ";
    string s4(10,'a');//连续10个设置为a
    string s5(s4);
    string s6("I am Lins");
    string s7=string (6,'c');


    //常用操作
    string s8= s3+s6;//将两个字符串合并成一个
    s3=s6;//用一个字符串来代替另一个字符串的对应元素

    cin >>s1;
    //s2=s1;
    cout<<"s1："<<s1<<endl;
    cout<<"s2: "<<s2<<endl;
    cout<<s3<<endl;
    cout<<s4<<endl;
    cout<<s5<<endl;
    cout<<s6<<endl;
    cout<<s7<<endl;
    cout<<s8<<endl;
    cout<<"s7 size = "<<s7.size()<<endl;//字符串长度，！不包括结束符
    cout<<(s2.empty()?"This string is empty":"This string is not empty")<<endl;

    //################################string IO操作
    //cin读入字符时，遇到空白就停止读取，"Hello World"可以用s1存hello s2存world  s3=s1+s2;来实现
    //这样麻烦，可以用getline来获取一整行内容
    string str;
    getline(cin,str);
    cout<<"读取一整行："<<str<<endl;


    //当把string对象和字符面值及字符串面值混在一条语句中使用时，必须确保+两侧的运算对象至少有一个是string
    string s11 = s5+ ",";//正确
    //string s13= "s" + ",";//错误
    //string s14="hello"+","+s11;//错误！！！！对比下面
    string s15= s11 + "hello" + ",";//正确！因为注意=号右边的运算顺序，从左到右！


    //###############################访问字符串的每个字符
    //传统for i 遍历
    for(int i=0;i<s3.size();i++)
    {
        cout<<s3[i]<<endl;
        s3[i]='s';//每一位！只能是一个字符！
    }


    //迭代器，虽然有点难拼写，但后面vector遍历都是要用的。打印vector之类都要用到，仔细看下也不难拼写
    string str111("hi baby jin");
    for(string::iterator it = str111.begin();it!= str111.end();it++)//string::ite ra tor
    {
        cout<<*it<<endl;
    }

    //#####################string的好用函数，比如找子串
    string ssss("hellll sssaaa ss");//aaa找aa的话，返回第一个a的下标 10
    cout<<ssss.find("aa",0)<<endl;//返回的是子串的位置，第二额参数是查找的起始位置，如果找不到就返回string：：npos
    if(ssss.find("aa1",0)== string::npos);
    {
        cout<<"找不到子串！"<<endl;
    }

    ssss.push_back('c');//向字符串末尾添加c ,要用单引号
    //######字符串的操作！！！要用函数
    // 这是里面对象类型只能是char
    //c++这里你定义的字符串用的是string对象。类型不同，会报错
    // strcpy(s1,s2);
    // strcmp(s1,s2);

    // 返回0表示想等，大于小于表示其在ascii上差多少
    int compareResult = s1.compare(s2); // 使用 compare 函数来比较字符串

    cout << "Comparison result: " << compareResult << endl;
    
    return 0;
}