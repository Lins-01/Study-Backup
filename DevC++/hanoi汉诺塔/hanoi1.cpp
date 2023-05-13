#include<stdio.h>
int i=1; //记录步数
void move(int n,char from,char to);

void hanoi(int n,char from,char depend_on,char to);

int main()
{
	int n;
	char x='A',y='B',z='C'; 
	printf("请输入盘子个数：");
 	scanf("%d",&n);
 	hanoi(n,x,y,z);
	return 0;
 } 
 
 void move(int n,char from,char to)
 {
 	printf("第%d步：将%d盘子%c--->%c\n",i++,n,from,to);
 }
 
 
 //hanoi递归函数
 //找到递归结束的条件，每次递归的任务 
 void hanoi(int n,char from,char depend_on,char to)
 {
 	if(n==1)
 	{
 		move(n,from,to);
	 }
	 else
	 {
	 	hanoi(n-1,from,to,depend_on);//将n-1个盘子先移动到中转柱上
		 move(n,from,to);//将身下的第n个盘子移动到目标柱上 
		 hanoi(n-1,depend_on,from,to); //将中转柱上的n-1个盘子移动到目标柱 
	 }
 }
