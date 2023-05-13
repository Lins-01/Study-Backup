#include<iostream>
#include<queue>
#include<vector>
#include<set>
using namespace std;

typedef long long LL;
const int coeff[3]={2,3,5};

int main()
{
	priority_queue<LL,vector<LL>,greater<LL> >pq;
	set<LL> s;//set s 用来判断 用x，2x,3x,5x，这种方法生成的丑数是否已经存在过了 
	pq.push(1);
	s.insert(1);
	for(int i=1; ;i++)
	{
		LL x=pq.top();pq.pop();
		if(i==1500){
			cout<<"The 1500's ugly number is "<<x<<endl;
			break;
		}
		for(int j=0;j<3;j++)
		{
			LL x2=x*coeff[j];
			if(!s.count(x2)){//判断该丑数是否已经存在 
				s.insert(x2);
				pq.push(x2);
			}
		}
	}
	return 0;
} 
