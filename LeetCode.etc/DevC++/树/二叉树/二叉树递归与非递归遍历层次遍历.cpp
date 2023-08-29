#include<iostream>
#include<cstdio>
#include<cstdlib>
#define ElemType char
#define MaxSize 1000 
using namespace std;
//二叉链存储结构--存储二叉树节点的数据元素 
typedef struct node
{
	ElemType data; 
	struct node *lchild;
	struct node *rchild;	
	
}BTNode;

void CreateBTNode(BTNode *&b,char * str)
{
	//使用一个栈St保存双亲节点，top为栈顶指针，k指定其后处理的节点是双亲节点（栈顶节点）的左孩子节点（k=1）还是右孩子节点（k=2） 
	BTNode *St[MaxSize],*p;
	int top=-1,k,j=0;
	char ch;
	b=NULL;// 建立的二叉树初始时为空
	ch=str[j];
	while(ch!='\0')//循环扫描str中每个字符 
	{
		switch(ch)
		{
			case '(':top++;St[top]=p;k=1;break;// ( 栈顶元素进栈 k=1下一个遇到的数据为左孩子
			case ')':top--;break;// ） 退栈一次 即栈顶指针向后移动一次
			case ',':k=2;break;// ，表示k=2 让遇到的下一个数据存储为栈顶元素的右孩子
			default:{
			//处理数据元素 存储为栈顶节点的左孩子或者右孩子 
				//初始化节点（申请空间，赋值 
				p=(BTNode*)malloc(sizeof(BTNode));
				p->data=ch;p->lchild=p->rchild=NULL;
				//若未建立根节点  (类似于链表表头 
				if(b==NULL){
					b=p;
				}
				else{//已建立根节点 
					switch(k)
					{
						case 1:St[top]->lchild=p;break;//让栈顶元素的左孩子为p 但p还没有进栈  就表示最后如果是叶子节点都不会在栈中 但可以通过最后一层栈中双亲节点输出
						case 2:St[top]->rchild=p;break;//右孩子为p 
					}
				}	
			}
			
		}
		//这里一定不能写错了 ，之前写在switch里面错了 
		j++;//控制循环 
		ch=str[j];//不断读入字符串中元素 
	} 
	
}

//在二叉树汇中查找值为x的节点  递归 
BTNode *FindNode(BTNode * b, ElemType x)
{
	BTNode *p;
	if(b==NULL)return NULL;//空树
	else if(b->data==x)return b;//递归截止条件 
	else{
		p=FindNode(b->lchild,x);
		if(p!=NULL)return p;//左节点中找到  返回 
		else {
			return FindNode(b->rchild,x); //右节点中找到 返回 
		}
	} 
}

//查找孩子节点 （左 
BTNode *LchildNode(BTNode *p)
{
	return p->lchild;
}
//查找孩子节点 （右 
BTNode *RchildNode(BTNode *p)
{
	return p->rchild;
}


//求树的高度 (递归
/*
	递归公式
	f(b)=0    若b==NULL   （截止条件  
	f(b)=MAX{f(b->lchild),f(b->rchild)} +1  其他情况   递归公式 
*/ 
int BTNodeHeight(BTNode *b)
{
	int lchildh,rchildh;
	if(b==NULL)return (0); //return 0 和 return （0）没有区别
	else{
		lchildh=BTNodeHeight(b->lchild);
		rchildh=BTNodeHeight(b->rchild);
		return (lchildh>rchildh)?(lchildh+1):(rchildh+1);
	} 
	
} 

//输出二叉树  （递归 
void DispBTNode(BTNode *b)
{
	if(b!=NULL)
	{
		printf("%c",b->data);
		if(b->lchild!=NULL||b->rchild!=NULL)
		{
			printf("(");//有孩子节点时才输出“（”
			DispBTNode(b->lchild);//递归处理左子树
			if(b->rchild!=NULL)printf(",");//有右孩子节点时才输出“，” 
			 DispBTNode(b->rchild);//递归处理右子树 
			 printf(")"); //有孩子节点时才输出“）”
		}
	}
}

//求二叉树中节点值为x的节点的层数(二叉树遍历 （递归 

//若b为空树 返回0
//如果当前根节点的节点值为x，则返回h，否则在左子树中查找，若在左子树中未找到，再在右子树中查找 
int Level(BTNode *b,ElemType x,int h)
{
	int l;//置h处置为1  从根节点开始查找，根节点层次为1 
	if(b==NULL)return 0;
	else if(b->data==x)return h;
	else{
		l=Level(b->lchild,x,h+1);//在左子树中查找
		if(l!=0)return l;
		else{//在左子树中未找到，再在右子树中查找 
			return (Level(b->rchild,x,h+1));
		} 
	}
	
}

//中序递归遍历二叉树 
void InOrderRe(BTNode *b)
{
	if(b!=NULL)
	{
		InOrderRe(b->lchild);
		printf("%c",b->data);
		InOrderRe(b->rchild); 
	}
} 
 
 
 
//中序非递归遍历二叉树  
void InOrder(BTNode *b)
{
	BTNode *St[MaxSize],*p;
	int top=-1;
	if(b!=NULL)
	{
		p=b;
		while(top>-1||p!=NULL)//处理*b节点的左子树
		{
			while(p!=NULL)
			{
				top++;
				St[top]=p;
				p=p->lchild;
			}
			//执行到此处时，栈顶元素没有左孩子或左子树均已访问过
			if(top>-1)
			{
				p=St[top];//出栈*p节点 
				top--;//为方便下一次执行 
				printf("%c",p->data);//访问该节点
				p=p->rchild;//转向处理*p的右孩子节点 
			} 
		} 
		printf("\n");
	}
}
 
//环形队列实现层次遍历
void LevelOrder(BTNode *b)
{
	BTNode *p;
	BTNode *qu[MaxSize];//定义环形队列，存放节点指针
	int front,rear;//定义对头和队尾指针
	front=rear=-1;//置队列为空队列
	rear++;
	qu[rear]=b;
	while(front!=rear)
	{
		front=(front+1)%MaxSize;
		p=qu[front];//对头出队列
		printf("%c",p->data);//访问节点
		if(p->lchild!=NULL)//有左孩子时将其进队 
		{
			rear=(rear+1)%MaxSize;
			qu[rear]=p->lchild;
		} 
		if(p->rchild!=NULL)//有右孩子时将其进队 
		{
			rear=(rear+1)%MaxSize;
			qu[rear]=p->rchild; 
		}
	} 
} 

 
int main()
{
	//创建二叉树 
	BTNode *b;
	int h;
	ElemType x;
	cout<<"请输入二叉树括号表示法字符串："; 
	//char str[]="A(B(D(,G)),C(E,F))";
	char str[MaxSize];
	gets(str);
	CreateBTNode(b,str);
	//printf("%s\n",str);
	//DispBTNode(b);
	cout<<"递归中序遍历二叉树："; 
	InOrderRe(b);
	cout<<endl;
	
	cout<<"非递归中序遍历二叉树："; 
	InOrder(b);
	cout<<endl;
	
	cout<<"环形队列实现层次遍历二叉树："; 
	LevelOrder(b);
	cout<<endl;
	//cout<<"b:";
	//DispBTNode(b);cout<<endl;
	//cout<<"请输入要查询的节点值：";
	scanf("%c",&x);
	h=Level(b,x,1);
	if(h==0)cout<<"b中不存在"<<x<<"节点"<<endl;
	else
	{
		cout<<"在b中"<<x<<"节点的层次为"<<h<<endl; 
	} 
	 
	
	return 0;
}























