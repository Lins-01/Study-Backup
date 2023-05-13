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
		int ok=1;//�жϵı�־λ
		while(B<=n)
		{
			if(A==target[B]){A++;B++;}
			//empty() Ϊ��ʱ����1���ǿ�ʱ����0 
			else if(!s.empty()&&s.top()==target[B]){s.pop();B++;}//ջ��Ϊ������ͬ ��ջ ����ջ��s.pop() 
			else if(A<=0){s.push(A);}//��ջ��s.push() 
			else{ok=0;break;}
		} 
		printf("%s",ok?"Yes":"No");
	}
	return 0;
} 
