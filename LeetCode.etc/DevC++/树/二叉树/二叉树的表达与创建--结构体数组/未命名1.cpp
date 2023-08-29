//二叉树的表达与创建，可以用链表或数组，大多用链表，这里用的数组，但也是结构体数组，又可成为静态链表 
#include<stdio.h>
#define ElementType char//记录该结点名称
#define Tree int //记录左右儿子
#define MaxTree 10 
#define Null -1
//第一步建立二叉树的结点Node
struct TreeNode{
	ElementType Element;//记录元素 
	Tree Left;
	Tree Right; 
}T1[MaxTree],T2[MaxTree];//定义结点并创建俩个二叉树（结构体数组） 

Tree BuildTree(struct TreeNode T[]); 
int main()
{
	Tree R1,R2;//这里不就是int型吗？用int型直接记录二叉树？虽然调用的时候是T1[R1]
	R1=BuildTree(T1);
	R2=BuildTree(T2); 
	return 0;
} 

Tree BuildTree(struct TreeNode T[])
{
	int N;
	int check[MaxTree*3];
	char cl,cr;
	int Root;
	if(0!=scanf("%d",&N))
	{
		for(int i=0;i<N;i++)check[i]=0;//定义一个数组存储哪些结点当过弟弟，便于找到root
		for(int i=0;i<N;i++)
		{
			scanf("%c %c %c",&T[i].Element,&cl,&cr);
			//对左边的弟弟一顿操作 
			if(cl!='-')
			{
				T[i].Left=cl-'0';//左弟弟存入当前结点
				check[T[i].Left]=1; 
			} else{
				T[i].Left=Null;
			}
			//右弟弟
			if(cr!='-')
			{
				T[i].Right=cr-'0';
				check[T[i].Right]=1;
			} else{
				T[i].Right=Null; 
			}
		} 
		//又是一顿遍历 找到Root
		int j;//j定义在for循环里会提示59行的错误，可能只能在for循环里调用吧 
		for(j=0;j<N;j++)
		{
				if(check[j]==0)break;
		}
		
		 Root =j;// [Error] name lookup of 'j' changed for ISO 'for' sco  
	}
	return Root; 
}
