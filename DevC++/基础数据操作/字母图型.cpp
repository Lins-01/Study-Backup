#include<stdio.h>
int main()
{
	char str[26];
	//��26����ĸ����������
	for(int i=0;i<26;i++)
		str[i]='A'+i;
	//��������
	int n,m,i,j;
	scanf("%d %d",&n,&m);
	int tmp; 
	//�ٿ���n>m��� ��while���Ҳ���A 
	for(i=0;i<n;i++)
	{
		tmp=i;//��i��ʾ�ӵڼ���Ԫ�ؿ�ʼ��� 
		while(1)//�����������A 
		{
			if(str[tmp]=='A')
			{
				printf("%c",str[tmp]);
				break;
			}
			if(i<m)
			{
				printf("%c",str[tmp]);
				tmp--;
			}else{//n>mʱֱ������ͺ���  i>=m
				for(int k=i;k>=(i-m+1);k--)
					printf("%c",str[k]);
				break;
			}
			
		} 
		//n>mʱ����ִ�� ���ﲻ�õ��� 
		if(i!=m)
		{
			for(int j=1;j<=(m-(i+1));j++)
			{
				printf("%c",str[j]);
			}
		} 
		printf("\n");
	}
	 
}
