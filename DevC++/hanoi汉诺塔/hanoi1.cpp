#include<stdio.h>
int i=1; //��¼����
void move(int n,char from,char to);

void hanoi(int n,char from,char depend_on,char to);

int main()
{
	int n;
	char x='A',y='B',z='C'; 
	printf("���������Ӹ�����");
 	scanf("%d",&n);
 	hanoi(n,x,y,z);
	return 0;
 } 
 
 void move(int n,char from,char to)
 {
 	printf("��%d������%d����%c--->%c\n",i++,n,from,to);
 }
 
 
 //hanoi�ݹ麯��
 //�ҵ��ݹ������������ÿ�εݹ������ 
 void hanoi(int n,char from,char depend_on,char to)
 {
 	if(n==1)
 	{
 		move(n,from,to);
	 }
	 else
	 {
	 	hanoi(n-1,from,to,depend_on);//��n-1���������ƶ�����ת����
		 move(n,from,to);//�����µĵ�n�������ƶ���Ŀ������ 
		 hanoi(n-1,depend_on,from,to); //����ת���ϵ�n-1�������ƶ���Ŀ���� 
	 }
 }
