#include<iostream>
#include<algorithm>
#include<vector>
#include<set>
using namespace std;
//慢--->只是得到的答案自己无法接受 
//是-->刷题才是王道 每种题型都刷  刷的过程中就会学到对应算法 
#define maxn 100005
vector<int> t[maxn];//类似动态数组 不同指定大小 t[id][ts] 可以让一个id存储多个ts 不用判断 
set<int> s;//set中元素自动排序且去重   
long N,D,k;

//判断是否是热帖 
bool judge(long id)
{
	sort(t[id].begin(),t[id].end());//取尺法 先排序
	long l=0,r=0,sum=0;//l,r游标，左右端点  令r与l之间差k（即尺长度）  即满足尺长 
	while(l<=r&&r<t[id].size()) 
	{
		sum++;
		if(sum>=k)
		{
			if(t[id][r]-t[id][l]<D)
			{
				return true;
			}else{
				l++;
				sum--;//控制差为 k
			}
		}
		r++;//每次右移 
	}
	return false;
}

int main()
{
	long ts,id;
	//数据输入 
	cin>>N>>D>>k; 
	for(long i=0;i<N;i++)//存N行 
	{
	
		cin>>ts>>id;
		t[id].push_back(ts);//一定用t.push_back()赋值
		s.insert(id);//用来存储id且去掉重复id  用于顺序遍历每个id   
	}
	//数据输出 
	for(set<int>::iterator it=s.begin();it!=s.end();it++)
	{
		id=*it;
		if(judge(id))
		{
			cout<<id<<endl;
		}
	} 
	
	
	return 0;
}






















