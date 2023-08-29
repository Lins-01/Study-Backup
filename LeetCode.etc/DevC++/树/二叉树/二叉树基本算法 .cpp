#include<iostream>
#include<cstdio>
#include<cstdlib>
#define ElemType char
#define MaxSize 1000 
using namespace std;
//�������洢�ṹ--�洢�������ڵ������Ԫ�� 
typedef struct node
{
	ElemType data; 
	struct node *lchild;
	struct node *rchild;	
	
}BTNode;

void CreateBTNode(BTNode *&b,char * str)
{
	//ʹ��һ��ջSt����˫�׽ڵ㣬topΪջ��ָ�룬kָ�������Ľڵ���˫�׽ڵ㣨ջ���ڵ㣩�����ӽڵ㣨k=1�������Һ��ӽڵ㣨k=2�� 
	BTNode *St[MaxSize],*p;
	int top=-1,k,j=0;
	char ch;
	b=NULL;// �����Ķ�������ʼʱΪ��
	ch=str[j];
	while(ch!='\0')//ѭ��ɨ��str��ÿ���ַ� 
	{
		switch(ch)
		{
			case '(':top++;St[top]=p;k=1;break;// ( ջ��Ԫ�ؽ�ջ k=1��һ������������Ϊ����
			case ')':top--;break;// �� ��ջһ�� ��ջ��ָ������ƶ�һ��
			case ',':k=2;break;// ����ʾk=2 ����������һ�����ݴ洢Ϊջ��Ԫ�ص��Һ���
			default:{
			//��������Ԫ�� �洢Ϊջ���ڵ�����ӻ����Һ��� 
				//��ʼ���ڵ㣨����ռ䣬��ֵ 
				p=(BTNode*)malloc(sizeof(BTNode));
				p->data=ch;p->lchild=p->rchild=NULL;
				//��δ�������ڵ�  (�����������ͷ 
				if(b==NULL){
					b=p;
				}
				else{//�ѽ������ڵ� 
					switch(k)
					{
						case 1:St[top]->lchild=p;break;//��ջ��Ԫ�ص�����Ϊp ��p��û�н�ջ  �ͱ�ʾ��������Ҷ�ӽڵ㶼������ջ�� ������ͨ�����һ��ջ��˫�׽ڵ����
						case 2:St[top]->rchild=p;break;//�Һ���Ϊp 
					}
				}	
			}
			
		}
		//����һ������д���� ��֮ǰд��switch������� 
		j++;//����ѭ�� 
		ch=str[j];//���϶����ַ�����Ԫ�� 
	} 
	
}

//�ڶ��������в���ֵΪx�Ľڵ�  �ݹ� 
BTNode *FindNode(BTNode * b, ElemType x)
{
	BTNode *p;
	if(b==NULL)return NULL;//����
	else if(b->data==x)return b;//�ݹ��ֹ���� 
	else{
		p=FindNode(b->lchild,x);
		if(p!=NULL)return p;//��ڵ����ҵ�  ���� 
		else {
			return FindNode(b->rchild,x); //�ҽڵ����ҵ� ���� 
		}
	} 
}

//���Һ��ӽڵ� ���� 
BTNode *LchildNode(BTNode *p)
{
	return p->lchild;
}
//���Һ��ӽڵ� ���� 
BTNode *RchildNode(BTNode *p)
{
	return p->rchild;
}


//�����ĸ߶� (�ݹ�
/*
	�ݹ鹫ʽ
	f(b)=0    ��b==NULL   ����ֹ����  
	f(b)=MAX{f(b->lchild),f(b->rchild)} +1  �������   �ݹ鹫ʽ 
*/ 
int BTNodeHeight(BTNode *b)
{
	int lchildh,rchildh;
	if(b==NULL)return (0); //return 0 �� return ��0��û������
	else{
		lchildh=BTNodeHeight(b->lchild);
		rchildh=BTNodeHeight(b->rchild);
		return (lchildh>rchildh)?(lchildh+1):(rchildh+1);
	} 
	
} 

//���������  ���ݹ� 
void DispBTNode(BTNode *b)
{
	if(b!=NULL)
	{
		printf("%c",b->data);
		if(b->lchild!=NULL||b->rchild!=NULL)
		{
			printf("(");//�к��ӽڵ�ʱ�����������
			DispBTNode(b->lchild);//�ݹ鴦��������
			if(b->rchild!=NULL)printf(",");//���Һ��ӽڵ�ʱ����������� 
			 DispBTNode(b->rchild);//�ݹ鴦�������� 
			 printf(")"); //�к��ӽڵ�ʱ�����������
		}
	}
}

//��������нڵ�ֵΪx�Ľڵ�Ĳ���(���������� ���ݹ� 

//��bΪ���� ����0
//�����ǰ���ڵ�Ľڵ�ֵΪx���򷵻�h���������������в��ң�������������δ�ҵ��������������в��� 
int Level(BTNode *b,ElemType x,int h)
{
	int l;//��h����Ϊ1  �Ӹ��ڵ㿪ʼ���ң����ڵ���Ϊ1 
	if(b==NULL)return 0;
	else if(b->data==x)return h;
	else{
		l=Level(b->lchild,x,h+1);//���������в���
		if(l!=0)return l;
		else{//����������δ�ҵ��������������в��� 
			return (Level(b->rchild,x,h+1));
		} 
	}
	
}

 
int main()
{
	//���������� 
	BTNode *b;
	int h;
	ElemType x;
	char str[]="A(B(D(,G)),C(E,F))";
	CreateBTNode(b,str);
	printf("%s\n",str);
	DispBTNode(b);
	cout<<endl;
	cout<<"b:";DispBTNode(b);cout<<endl;
	cout<<"�ڵ�ֵ��";
	scanf("%c",&x);
	h=Level(b,x,1);
	if(h==0)cout<<"b�в�����"<<x<<"�ڵ�"<<endl;
	else
	{
		cout<<"��b��"<<x<<"�ڵ�Ĳ��Ϊ"<<h<<endl; 
	} 
	 
	
	return 0;
}























