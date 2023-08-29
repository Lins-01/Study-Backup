#include<iostream>
#include<sstream>
#include<string>
using namespace std;

int main()
{
	string line;
	int sum=0,tmp;//tmp作为临时变量存储line转换来的每个数 
	while(getline(cin,line))//先把每行作为字符串存在字符串line中，因为是包含空格的一组的数据，不能直接用sum运算。 
	{
		
		stringstream ss(line);//将string line转换为流 
		while(ss>>tmp)sum+=tmp;//由line转换的流ss输入到tmp ，成为流后，相当于cin>>tmp，会自动识别输入类型，所以空格不会输入到tmp 
		cout<<sum<<endl;
	}
	cout<<sum<<endl;  //15 8 E:\DevC++\C++输入一行带空格的字符串\未命名2.cpp [Error] 'sum' was not declared in this scope ???
	return 0;
}
