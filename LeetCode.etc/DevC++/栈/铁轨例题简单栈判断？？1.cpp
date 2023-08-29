#include<iostream>
#include<cstdio>
#include<stack>
using namespace std;

const int MAXN=1000+10;

int n,target[MAXN];

int main()
{
	while(cin>>n){
		stack<int> s;
		int A=1,B=1;
		for(int i=1;i<=n;i++)
		{
			cin>>target[i];
		}
		int ok=1;//判断的标志位
		while(B<=n)
		{
			if(A==target[B]){A++;B++;}
			//empty() 为空时返回1，非空时返回0 
			else if(!s.empty()&&s.top()==target[B]){s.pop();B++;}//栈不为空且相同 出栈 ，出栈用s.pop() 
			else if(A<=0){s.push(A);}//入栈用s.push() 
			else{ok=0;break;}
		} 
		printf("%s",ok?"Yes":"No");
	}
	return 0;
} 
