//�������ı���봴������������������飬��������������õ����飬��Ҳ�ǽṹ�����飬�ֿɳ�Ϊ��̬���� 
#include<stdio.h>
#define ElementType char//��¼�ý������
#define Tree int //��¼���Ҷ���
#define MaxTree 10 
#define Null -1
//��һ�������������Ľ��Node
struct TreeNode{
	ElementType Element;//��¼Ԫ�� 
	Tree Left;
	Tree Right; 
}T1[MaxTree],T2[MaxTree];//�����㲢�����������������ṹ�����飩 

Tree BuildTree(struct TreeNode T[]); 
int main()
{
	Tree R1,R2;//���ﲻ����int������int��ֱ�Ӽ�¼����������Ȼ���õ�ʱ����T1[R1]
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
		for(int i=0;i<N;i++)check[i]=0;//����һ������洢��Щ��㵱���ܵܣ������ҵ�root
		for(int i=0;i<N;i++)
		{
			scanf("%c %c %c",&T[i].Element,&cl,&cr);
			//����ߵĵܵ�һ�ٲ��� 
			if(cl!='-')
			{
				T[i].Left=cl-'0';//��ܴܵ��뵱ǰ���
				check[T[i].Left]=1; 
			} else{
				T[i].Left=Null;
			}
			//�ҵܵ�
			if(cr!='-')
			{
				T[i].Right=cr-'0';
				check[T[i].Right]=1;
			} else{
				T[i].Right=Null; 
			}
		} 
		//����һ�ٱ��� �ҵ�Root
		int j;//j������forѭ�������ʾ59�еĴ��󣬿���ֻ����forѭ������ð� 
		for(j=0;j<N;j++)
		{
				if(check[j]==0)break;
		}
		
		 Root =j;// [Error] name lookup of 'j' changed for ISO 'for' sco  
	}
	return Root; 
}
