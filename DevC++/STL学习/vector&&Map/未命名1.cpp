#include<iostream>
#include<string>
#include<cctype>
#include<vector>
#include<map>
#include<algorithm>
using namespace std;


//map：从键（key)到值(value)到映射！！ 


map<string,int> cnt;//用来图存储该单词是否符合条件，一一对应，所以方便输出 
vector<string>words;

//将单词s进行“标准化”
//即用tolower将单词全部变成小写，然后对单词对字母按字典排序，方便检测是否可以通过重组字母组成不同单词 
string repr(const string &s){
	string ans=s;
	for(int i=0;i<ans.length();i++)
		ans[i]=tolower(ans[i]);
	sort(ans.begin(),ans.end());
	return ans; 
} 

int main()
{
	int n=0;
	string s;
	while(cin>>s)
	{
		if(s[0]=='#')break;
		words.push_back(s);
		string r=repr(s);
		//count检测map中是否存在某个key 
		if(!cnt.count(r))cnt[r]=0;//cnt[r]==0表示该单词重复 
		cnt[r]++;
	}
	vector<string> ans;
	for(int i=0;i<words.size();i++)
		if(cnt[repr(words[i])]==1)ans.push_back(words[i]);
	sort(ans.begin(),ans.end());
	for(int i=0;i<ans.size();i++)
	{
		cout<<ans[i]<<endl;
	} 
	return 0;
}
