#include <bits/stdc++.h>
#include <vector>
using namespace std;

//封装一个打印函数
//用模板template来解决，用于实现以下场景：如过要实现相同功能，但仅数据类型不同，是不是要再写一个相应类型的函数来执行呢？
//用模板可以只写一个，使用不同数据类型，类似函数的重载，只不过模板更严格

template <typename T>//##########没有;号，且与下面函数之间不可以有任何其他代码！！！！！！，typename换成class一样的
void showvector(vector<T> v)
{
    for(typename vector<T>::iterator it=v.begin();it!=v.end();it++)
    {
        cout<<*it;
    }
    cout<<endl;
}
//编译错误need 'typename' before *** because *** is a dependent scope  
//在vector<T>::iterator前面加上typename显式地说明它是一个类型而, 非成员变量.
//https://blog.csdn.net/pb1995/article/details/49532285/




int main()
{
    //#################################vector就相当于c中的数组，只不过是变长的数组
    //定义时不用声明长度，还有多的一些相当于oop的方法
    

    //###################################定义和初始化

    //####vector的元素类型是int，默认初始化为0；为string，默认为空字符串
    vector<int> v1;//####可以这么声明，但添加元素时，必须要用push_back()，对已存在的元素进行操作时可以直接v[i]=....
    vector<string> v2;
    vector<vector<int> >vv;//########！！！！注意空格！！！！！！！！，这里相当于二维数组int a[n][n];
    vector<int> v5={1,2,3,4,5};//列表初始化，花括号
    vector<string> v6={"hi!","my","name","is","Lins"};
    vector<int> v7(5,-1);//初始化为5个-1
    vector<string> v8(3,"hi!");
    vector<int> v9(10);//默认初始化为0
    vector<string> v10(3);//默认初始化空字符串

    //#######如何向vector添加元素
    //###用push_back,加入数组尾部
    int temp;//用temp先cin输入，再push_back进去
    for(int i=0;i<20;i++)
    {
        cin>>temp;
        v2.push_back(temp);
        v1.push_back(i);//运行后，v1为0 1 2 3.....
    }


      //不能这么初始化vector，如下直接等于都可以。
       vector<int> v2;
       /* for (int i = 0; i < N; i++) {
            v2[i] = v[i];
        }*/
        //0x793AFC66 (ucrtbased.dll) (德科面2hjc - 吃香蕉.exe 中)处有未经处理的异常: 将一个无效参      数传递给了将无效参数视为严重错误的函数。
        v2 = v;//v已经用temp 。push_back初始化


    //v1==v2当且仅当他们的元素数量相同，且元素值都相同
    //v1!=v2
    //< <= > >= 按字典顺序进行比较
    //############访问和操作vector 的每个元素
    for(int i=0;i<v1.size();i++)
    {
        cout<<v1[i]<<endl;
        v1[i]=100;
        cout<<v1[i]<<endl;
    }
    
    //迭代器访问
    vector<string> vv6={"hi!","iterator","test!","I","like","it!"};
    for(vector<string>::iterator itt=vv6.begin();itt!=vv6.end();itt++)
    {   
        if(itt==vv6.end())
        {
            cout<<*itt;
            break;
        }
        cout<<*itt<<" ";

    }
    cout<<endl;

    //使用模板封装的打印函数
    vector<string> v11={"template!","let's"," go!"};
    showvector(v11);


    return 0;
}