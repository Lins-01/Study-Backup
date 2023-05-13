#include <bits/stdc++.h>
using namespace std;

template <typename T>
void showVector(vector<T> v)
{
    for(typename vector<T>::iterator it=v.begin();it!=v.end();it++)
    {
        cout<<*it;
    }
    cout<<endl;
}
//sort函数三个参数，第三个为比较方法，可以不写，不写默认从小到大排序
//sort(start,end,comp)  end一般为数组名+数组长度
//compare函数
//bool comp1(vector<T> v)  comp中两个参数只要是数即可,bool类型
//即使是vector也只用int double float之类
bool comp1(int a,int b)
{ 
    return a<b;//a<b返回 true  实现从小到大排序  git_test1
    //git_test1 read11
}
bool comp2(int a,int b)
{
    return a>b;//a>b返回true  实现从大到小排序
}



//结构体&&按多个条件进行排序
//eg:先按成绩排序，成绩相同按学号排序
struct node
{
    int score;
    int num;
    int c;

};
bool comp(node x,node y )
{
    if(x.score!=y.score)return x.score>y.score;//当xy成绩不等时，先按成绩降序排列,return 执行后不执行下面代码
    if(x.num!=y.num)return x.num<y.num;//成绩相等，按学号
    return x.c<y.c;//都相等，按c排序
}

int main()
{
    vector<int> v={2,1,8,5};
    cout<<"v排序前：";showVector(v);
    sort(v.begin(),v.end());//默认从小到大
    cout<<"v默认排序后：";showVector(v);
    sort(v.begin(),v.end(),comp1);
    cout<<"vcomp1排序后：";showVector(v);
    sort(v.begin(),v.end(),comp2);
    cout<<"vcomp2排序后：";showVector(v);


    //定义结构体数组
    //struct node people[3];
    //初始化1
    struct node people[3]={
        {61,05,20},
        {61,02,30},
        {99,02,3}
    };
    // for(int i=0;i<3;i++)
    // {
    //    cin>>people[i].score>>people[i].num>>people[i].c;
    // }
    sort(people,people+3,comp);
    cout<<"多条件排序：";
    for(int i=0;i<3;i++)
    {
        cout<<i<<"的分数："<<people[i].score<<"  学号："<<people[i].num<<"  c: "<<people[i].c<<endl;
    }

    return 0;
}