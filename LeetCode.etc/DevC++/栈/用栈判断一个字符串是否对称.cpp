#include<iostream>
#include<cstdlib>
#define MaxSize 10000
#define ElemType char 
using namespace std;

typedef struct stack{
	ElemType data[MaxSize]; 
	int top;//栈顶指针，原来只是一个用来记录的int型整数 
}SqStack;

void InitStack(SqStack *&s)//初始化栈 
{
	s=(SqStack *)malloc(sizeof(SqStack));
	s->top=-1;
}

void DestroyStack(SqStack *&s)//销毁栈 ,跟链表不同，不用遍历，栈相当于链表的一个结点，数据存在数组中 
{
	free(s);
}

bool StackEmpty(SqStack *s)//判断栈是否为空
{
	return(s->top==-1);//因为初始化栈的时候将栈顶指针初试化为-1 
} 

bool Push(SqStack *&s,ElemType e)
{
	if(s->top==MaxSize-1)return false;//数组最后一位元素为MaxSize-1
	s->top++;//栈指针加1 
	s->data[s->top]=e;
	return true; 
} 

//出栈，把栈顶的值传给第二个参数e 
bool Pop(SqStack *&s,ElemType &e)//因为会改变e的值，所以用引用 
{
	if(s->top==-1)//栈为空
	return false;
	e=s->data[s->top];//取栈顶元素
	s->top--;//栈指针加1 
	return true; 
}

bool GetTop(SqStack*s,ElemType &e)
{
	if(s->top==-1)return false;//栈为空的情况
	e=s->data[s->top];
	return true; 
}

//判断一个字符串是否是对称串 
bool sysmetry(ElemType str[])
{
		
	ElemType tmp;
	SqStack *s;
	//s=(SqStack *)malloc(sizeof(SqStack));   //交给下面的初始化栈函数来做，你只需要定义一个栈 
	//初始化栈 
	InitStack(s);
	for(int i=0;str[i]!='\0';i++)
	{
		Push(s,str[i]);
	}
	//出栈与数组顺序遍历对比 
	for(int i=0;str[i]!='\0';i++)
	{
		Pop(s,tmp);
		if(tmp!=str[i]){
			DestroyStack(s);
		   	return false;
		}
	}
	DestroyStack(s);
	return true;
}

int main()
{
	ElemType str[MaxSize];
	int n=8;
	for(int i=0;i<n;i++)
	{
		cin>>str[i];
	} 
	if(sysmetry(str)){
		cout<<"66666"<<endl; 
	}
	else{
		cout<<"No"<<endl;
	}
	return 0;
}

































