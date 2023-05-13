#include<iostream>
#include<cstdlib>
#define MaxSize 1000
#define ElemType int
using namespace std;

//定义节点 双向链表
typedef struct LNode
{
	ElemType data;
	struct LNode *next;
	struct LNode *prior;
}LinkList;

//尾插法创建循环双链表 使元素顺序与数组一致
LinkList * CreateListF(LinkList *L,ElemType a[],int n)
{
	LinkList *s,*r;
	int i;
	L=(LinkList *)malloc(sizeof(LinkList));
	r=L;
	for(i=0;i<n;i++)
	{
		s=(LinkList *)malloc(sizeof(LinkList));
		s->data=a[i];
		r->next=s;
		s->prior=r;
		r=s;
	}
	r->next=L->next;
	L->next->prior=r;
	
	return L->next;
}

ElemType a[MaxSize];

void Inia(ElemType a[])
{
	for(int x=0;x<MaxSize;x++)
	{
		a[x]=x+1;
	}
}

int main()
{
	int n,m,k,x;
	cin>>n>>m;
	cout<<"输入K:";
	cin>>k;
	cout<<"从第几个人开始数:";
	 cin>>x;
	Inia(a);
	LinkList *L=CreateListF(L,a,n);;
	//CreateListF(L,a,n);

	//每次都循环m次
	LinkList *ptemp;
	ptemp=(LinkList *)malloc(sizeof(LinkList));
	//ptemp第一次为尾节点这样就可以使每次循环m-1次条件设置一样  因为假设第一次 开始点为 尾结点的下一个 
	ptemp=L;
	int headflag=1;
	int countk=0;
	int num; 
	while(ptemp->next!=ptemp->prior)
	{
		
		countk++;
		if(headflag==1){
				for(int j=1;j<x;j++)
			{
				ptemp=ptemp->next;
			} 
			headflag=0;
		}
		else{
			ptemp=ptemp->next;
		}
		for(int i=1;i<m;i++)
		{
			
			//找到第m个
			ptemp=ptemp->next;
		}
		// 出列 删除
		
		ptemp->prior->next=ptemp->next;
		ptemp->next->prior=ptemp->prior;
		cout<<ptemp->data<<endl;
		if(countk==k)
		{
			num=ptemp->data;
		}
	}
	ptemp=ptemp->next;
	cout<<ptemp->data<<endl;
	cout<<"第"<<k<<"个出列的序号为"<<num<<endl;
	
	return 0;
}

