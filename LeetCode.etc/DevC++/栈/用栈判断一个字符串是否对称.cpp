#include<iostream>
#include<cstdlib>
#define MaxSize 10000
#define ElemType char 
using namespace std;

typedef struct stack{
	ElemType data[MaxSize]; 
	int top;//ջ��ָ�룬ԭ��ֻ��һ��������¼��int������ 
}SqStack;

void InitStack(SqStack *&s)//��ʼ��ջ 
{
	s=(SqStack *)malloc(sizeof(SqStack));
	s->top=-1;
}

void DestroyStack(SqStack *&s)//����ջ ,������ͬ�����ñ�����ջ�൱�������һ����㣬���ݴ��������� 
{
	free(s);
}

bool StackEmpty(SqStack *s)//�ж�ջ�Ƿ�Ϊ��
{
	return(s->top==-1);//��Ϊ��ʼ��ջ��ʱ��ջ��ָ����Ի�Ϊ-1 
} 

bool Push(SqStack *&s,ElemType e)
{
	if(s->top==MaxSize-1)return false;//�������һλԪ��ΪMaxSize-1
	s->top++;//ջָ���1 
	s->data[s->top]=e;
	return true; 
} 

//��ջ����ջ����ֵ�����ڶ�������e 
bool Pop(SqStack *&s,ElemType &e)//��Ϊ��ı�e��ֵ������������ 
{
	if(s->top==-1)//ջΪ��
	return false;
	e=s->data[s->top];//ȡջ��Ԫ��
	s->top--;//ջָ���1 
	return true; 
}

bool GetTop(SqStack*s,ElemType &e)
{
	if(s->top==-1)return false;//ջΪ�յ����
	e=s->data[s->top];
	return true; 
}

//�ж�һ���ַ����Ƿ��ǶԳƴ� 
bool sysmetry(ElemType str[])
{
		
	ElemType tmp;
	SqStack *s;
	//s=(SqStack *)malloc(sizeof(SqStack));   //��������ĳ�ʼ��ջ������������ֻ��Ҫ����һ��ջ 
	//��ʼ��ջ 
	InitStack(s);
	for(int i=0;str[i]!='\0';i++)
	{
		Push(s,str[i]);
	}
	//��ջ������˳������Ա� 
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

































