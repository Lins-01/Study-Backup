#include<iostream>
#include<algorithm>
#include<vector>
#include<set>
using namespace std;
//��--->ֻ�ǵõ��Ĵ��Լ��޷����� 
//��-->ˢ��������� ÿ�����Ͷ�ˢ  ˢ�Ĺ����оͻ�ѧ����Ӧ�㷨 
#define maxn 100005
vector<int> t[maxn];//���ƶ�̬���� ��ָͬ����С t[id][ts] ������һ��id�洢���ts �����ж� 
set<int> s;//set��Ԫ���Զ�������ȥ��   
long N,D,k;

//�ж��Ƿ������� 
bool judge(long id)
{
	sort(t[id].begin(),t[id].end());//ȡ�߷� ������
	long l=0,r=0,sum=0;//l,r�α꣬���Ҷ˵�  ��r��l֮���k�����߳��ȣ�  ������߳� 
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
				sum--;//���Ʋ�Ϊ k
			}
		}
		r++;//ÿ������ 
	}
	return false;
}

int main()
{
	long ts,id;
	//�������� 
	cin>>N>>D>>k; 
	for(long i=0;i<N;i++)//��N�� 
	{
	
		cin>>ts>>id;
		t[id].push_back(ts);//һ����t.push_back()��ֵ
		s.insert(id);//�����洢id��ȥ���ظ�id  ����˳�����ÿ��id   
	}
	//������� 
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






















