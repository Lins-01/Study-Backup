// 单词接龙.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include<bits/stdc++.h>
using namespace std;
int main() {
	int k, N;
	string temp;
	cin >> k >> N;//能连续两个输入吗？test 
	vector<string> list;
	for (int i = 0; i < N; i++) {
		cin >> temp;
		list.push_back(temp);
	}
	
	string head = list[k];
	string res;
	res += head;
	//选过的从list去掉
	//v.begin()得到数组的头指针（地址），取值的话加个*
	//v.erase()删除指针指向的数据项。（传入地址）
	//取头元素的话，可以用下标就是0，为取首元素
	//取尾元素，用.size()-1，做下标
	list.erase(list.begin() + k);
	char tail = head[head.size() - 1];
	while (true) {
		set<string> mset;//集合，没有重复元素，且按字典排序
		//不能用下标找，遍历用迭代器，或者地址。再对地址加*可取值

		for (int i = 0; i < list.size(); i++) {
			string word = list[i];
			if (word.find(tail) == 0) {//是尾巴跟头一样，就加入集合
				mset.insert(word);
			}
		}
		if (mset.size() == 0)break;//循环终止条件，mset无重复集合内无元素了，即没有与上一个尾元素与一样的
		string first = *(mset.begin());
		int len = first.size();
		string aim = "";
		for (string s : mset) {//找最长的那个
			if (s.length() > len) {
				len = s.length();
				aim = s;
			}
		}
		//找到符合条件的加入res字符串
		string into;
		into = (len!=first.length()?aim:first);//判断是否第一个就是最长的 
		res += into;
		//修改下一轮的尾字母
		tail = into[into.size() - 1];
		//把list中的此次添加的删除。否则下次mset中可能还会出现
		//因为erase是用指针删的，所以遍历到into这里，用begin加上下标i。转为地址再删。
		for (int i = 0; i < list.size(); i++) {
			if (list[i] == into) {
				list.erase(list.begin() + i);
				i--;//重点，因为erase函数，会把后面元素前移，之前的后一个元素移动到了i的位置
				//因为for循环后面有个i++，所以防止后面一个元素被跳过，要i--
			}
		}
		
	}
	cout << res;
	return 0;
}


