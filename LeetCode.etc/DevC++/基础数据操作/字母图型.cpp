#include<stdio.h>
int main()
{
	char str[26];
	//把26个字母存入数组中
	for(int i=0;i<26;i++)
		str[i]='A'+i;
	//数据输入
	int n,m,i,j;
	scanf("%d %d",&n,&m);
	int tmp; 
	//再考虑n>m情况 即while中找不到A 
	for(i=0;i<n;i++)
	{
		tmp=i;//用i表示从第几个元素开始输出 
		while(1)//先逆序输出到A 
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
			}else{//n>m时直接输出就好了  i>=m
				for(int k=i;k>=(i-m+1);k--)
					printf("%c",str[k]);
				break;
			}
			
		} 
		//n>m时不会执行 这里不用担心 
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
