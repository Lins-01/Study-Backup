#include <bits/stdc++.h>
using namespace std;

template <typename T>
void showVector(vector<T> v)
{
    for (typename vector<T>::iterator it = v.begin(); it != v.end(); it++)
    {
        cout << *it;
    }
    cout << endl;
}
//sort函数三个参数，第三个为比较方法，可以不写，不写默认从小到大排序
//sort(start,end,comp)  end一般为数组名+数组长度
//compare函数
//bool comp1(vector<T> v)  comp中两个参数只要是数即可,bool类型
//即使是vector也只用int double float之类
bool comp1(int a, int b)
{
    return a < b;//a<b返回 true  实现从小到大排序  git_test1
    //git_test1 read11
}
bool comp2(int a, int b)
{
    return a > b;//a>b返回true  实现从大到小排序
}



//结构体&&按多个条件进行排序
//eg:先按成绩排序，成绩相同按学号排序
struct node
{
    int score;
    int num;
    int c;

};
bool comp(node x, node y)
{
    if (x.score != y.score)return x.score > y.score;//当xy成绩不等时，先按成绩降序排列,return 执行后不执行下面代码
    if (x.num != y.num)return x.num < y.num;//成绩相等，按学号
    return x.c < y.c;//都相等，按c排序
}

int main()
{
    vector<int> v = { 2,1,8,5 };
    cout << "v排序前："; showVector(v);
    sort(v.begin(), v.end());//默认从小到大
    cout << "v默认排序后："; showVector(v);
    sort(v.begin(), v.end(), comp1);
    cout << "vcomp1排序后："; showVector(v);
    sort(v.begin(), v.end(), comp2);
    cout << "vcomp2排序后："; showVector(v);


    //定义结构体数组
    //struct node people[3];
    //初始化1
    struct node people[3] = {
        {61,05,20},
        {61,02,30},
        {99,02,3}
    };
    // for(int i=0;i<3;i++)
    // {
    //    cin>>people[i].score>>people[i].num>>people[i].c;
    // }
    sort(people, people + 3, comp);
    cout << "多条件排序：";
    for (int i = 0; i < 3; i++)
    {
        cout << i << "的分数：" << people[i].score << "  学号：" << people[i].num << "  c: " << people[i].c << endl;
    }

    return 0;
}

#include <bits/stdc++.h>
//不用写 string头文件。但不是高亮不好看，自己就写下吧
//#include <string> 写了也不高亮
using namespace std;
int main()
{
    //#######################string的定义和初始化
    string s1;//初始化字符串，空字符串
    string s2 = s1;//不可以这么写！，要复制s2=s1输入后的值，应写在cin>>s1之后
    string s3 = "I am JingleQ";
    string s4(10, 'a');//连续10个设置为a
    string s5(s4);
    string s6("I am Lins");
    string s7 = string(6, 'c');


    //常用操作
    string s8 = s3 + s6;//将两个字符串合并成一个
    s3 = s6;//用一个字符串来代替另一个字符串的对应元素

    cin >> s1;
    //s2=s1;
    cout << "s1：" << s1 << endl;
    cout << "s2: " << s2 << endl;
    cout << s3 << endl;
    cout << s4 << endl;
    cout << s5 << endl;
    cout << s6 << endl;
    cout << s7 << endl;
    cout << s8 << endl;
    cout << "s7 size = " << s7.size() << endl;//字符串长度，！不包括结束符
    cout << (s2.empty() ? "This string is empty" : "This string is not empty") << endl;

    //################################string IO操作
    //cin读入字符时，遇到空白就停止读取，"Hello World"可以用s1存hello s2存world  s3=s1+s2;来实现
    //这样麻烦，可以用getline来获取一整行内容
    string str;
    getline(cin, str);//要有cin
    cout << "读取一整行：" << str << endl;


    //当把string对象和字符面值及字符串面值混在一条语句中使用时，必须确保+两侧的运算对象至少有一个是string
    string s11 = s5 + ",";//正确
    //string s13= "s" + ",";//错误
    //string s14="hello"+","+s11;//错误！！！！对比下面
    string s15 = s11 + "hello" + ",";//正确！因为注意=号右边的运算顺序，从左到右！


    //###############################访问字符串的每个字符
    //传统for i 遍历
    for (int i = 0; i < s3.size(); i++)
    {
        cout << s3[i] << endl;
        s3[i] = 's';//每一位！只能是一个字符！
    }


    //迭代器，虽然有点难拼写，但后面vector遍历都是要用的。打印vector之类都要用到，仔细看下也不难拼写
    string str111("hi baby jin");
    for (string::iterator it = str111.begin(); it != str111.end(); it++)//string::ite ra tor
    {
        cout << *it << endl;
    }

    //#####################string的好用函数，比如找子串
    string ssss("hellll sssaaa ss");//aaa找aa的话，返回第一个a的下标 10
    cout << ssss.find("aa", 0) << endl;//返回的是子串的位置，第二额参数是查找的起始位置，如果找不到就返回string：：npos
    if (ssss.find("aa1", 0) == string::npos);
    {
        cout << "找不到子串！" << endl;
    }

    ssss.push_back("c");//向字符串末尾添加c
    //######字符串的操作！！！要用函数
    strcpy(s1, s2);
    strcmp(s1, s2);

    return 0;
}