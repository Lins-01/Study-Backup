#include<iostream>
#include<cstdlib>
#define MaxSize 1000
#define ElemType int
using namespace std;

//����ڵ� ˫������
typedef struct LNode
{
	ElemType data;
	struct LNode *next;
	struct LNode *prior;
}LinkList;

//β�巨����ѭ��˫���� ʹԪ��˳��������һ��
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
	cout<<"����K:";
	cin>>k;
	cout<<"�ӵڼ����˿�ʼ��:";
	 cin>>x;
	Inia(a);
	LinkList *L=CreateListF(L,a,n);;
	//CreateListF(L,a,n);

	//ÿ�ζ�ѭ��m��
	LinkList *ptemp;
	ptemp=(LinkList *)malloc(sizeof(LinkList));
	//ptemp��һ��Ϊβ�ڵ������Ϳ���ʹÿ��ѭ��m-1����������һ��  ��Ϊ�����һ�� ��ʼ��Ϊ β������һ�� 
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
			
			//�ҵ���m��
			ptemp=ptemp->next;
		}
		// ���� ɾ��
		
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
	cout<<"��"<<k<<"�����е����Ϊ"<<num<<endl;
	
	return 0;
}

