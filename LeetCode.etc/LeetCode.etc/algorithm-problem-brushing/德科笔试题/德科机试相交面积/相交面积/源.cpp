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
//sort��������������������Ϊ�ȽϷ��������Բ�д����дĬ�ϴ�С��������
//sort(start,end,comp)  endһ��Ϊ������+���鳤��
//compare����
//bool comp1(vector<T> v)  comp����������ֻҪ��������,bool����
//��ʹ��vectorҲֻ��int double float֮��
bool comp1(int a, int b)
{
    return a < b;//a<b���� true  ʵ�ִ�С��������  git_test1
    //git_test1 read11
}
bool comp2(int a, int b)
{
    return a > b;//a>b����true  ʵ�ִӴ�С����
}



//�ṹ��&&�����������������
//eg:�Ȱ��ɼ����򣬳ɼ���ͬ��ѧ������
struct node
{
    int score;
    int num;
    int c;

};
bool comp(node x, node y)
{
    if (x.score != y.score)return x.score > y.score;//��xy�ɼ�����ʱ���Ȱ��ɼ���������,return ִ�к�ִ���������
    if (x.num != y.num)return x.num < y.num;//�ɼ���ȣ���ѧ��
    return x.c < y.c;//����ȣ���c����
}

int main()
{
    vector<int> v = { 2,1,8,5 };
    cout << "v����ǰ��"; showVector(v);
    sort(v.begin(), v.end());//Ĭ�ϴ�С����
    cout << "vĬ�������"; showVector(v);
    sort(v.begin(), v.end(), comp1);
    cout << "vcomp1�����"; showVector(v);
    sort(v.begin(), v.end(), comp2);
    cout << "vcomp2�����"; showVector(v);


    //����ṹ������
    //struct node people[3];
    //��ʼ��1
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
    cout << "����������";
    for (int i = 0; i < 3; i++)
    {
        cout << i << "�ķ�����" << people[i].score << "  ѧ�ţ�" << people[i].num << "  c: " << people[i].c << endl;
    }

    return 0;
}

#include <bits/stdc++.h>
//����д stringͷ�ļ��������Ǹ������ÿ����Լ���д�°�
//#include <string> д��Ҳ������
using namespace std;
int main()
{
    //#######################string�Ķ���ͳ�ʼ��
    string s1;//��ʼ���ַ��������ַ���
    string s2 = s1;//��������ôд����Ҫ����s2=s1������ֵ��Ӧд��cin>>s1֮��
    string s3 = "I am JingleQ";
    string s4(10, 'a');//����10������Ϊa
    string s5(s4);
    string s6("I am Lins");
    string s7 = string(6, 'c');


    //���ò���
    string s8 = s3 + s6;//�������ַ����ϲ���һ��
    s3 = s6;//��һ���ַ�����������һ���ַ����Ķ�ӦԪ��

    cin >> s1;
    //s2=s1;
    cout << "s1��" << s1 << endl;
    cout << "s2: " << s2 << endl;
    cout << s3 << endl;
    cout << s4 << endl;
    cout << s5 << endl;
    cout << s6 << endl;
    cout << s7 << endl;
    cout << s8 << endl;
    cout << "s7 size = " << s7.size() << endl;//�ַ������ȣ���������������
    cout << (s2.empty() ? "This string is empty" : "This string is not empty") << endl;

    //################################string IO����
    //cin�����ַ�ʱ�������հ׾�ֹͣ��ȡ��"Hello World"������s1��hello s2��world  s3=s1+s2;��ʵ��
    //�����鷳��������getline����ȡһ��������
    string str;
    getline(cin, str);//Ҫ��cin
    cout << "��ȡһ���У�" << str << endl;


    //����string������ַ���ֵ���ַ�����ֵ����һ�������ʹ��ʱ������ȷ��+������������������һ����string
    string s11 = s5 + ",";//��ȷ
    //string s13= "s" + ",";//����
    //string s14="hello"+","+s11;//���󣡣������Ա�����
    string s15 = s11 + "hello" + ",";//��ȷ����Ϊע��=���ұߵ�����˳�򣬴����ң�


    //###############################�����ַ�����ÿ���ַ�
    //��ͳfor i ����
    for (int i = 0; i < s3.size(); i++)
    {
        cout << s3[i] << endl;
        s3[i] = 's';//ÿһλ��ֻ����һ���ַ���
    }


    //����������Ȼ�е���ƴд��������vector��������Ҫ�õġ���ӡvector֮�඼Ҫ�õ�����ϸ����Ҳ����ƴд
    string str111("hi baby jin");
    for (string::iterator it = str111.begin(); it != str111.end(); it++)//string::ite ra tor
    {
        cout << *it << endl;
    }

    //#####################string�ĺ��ú������������Ӵ�
    string ssss("hellll sssaaa ss");//aaa��aa�Ļ������ص�һ��a���±� 10
    cout << ssss.find("aa", 0) << endl;//���ص����Ӵ���λ�ã��ڶ�������ǲ��ҵ���ʼλ�ã�����Ҳ����ͷ���string����npos
    if (ssss.find("aa1", 0) == string::npos);
    {
        cout << "�Ҳ����Ӵ���" << endl;
    }

    ssss.push_back("c");//���ַ���ĩβ���c
    //######�ַ����Ĳ���������Ҫ�ú���
    strcpy(s1, s2);
    strcmp(s1, s2);

    return 0;
}