#include<iostream>
#include<string>
#include<algorithm>
using namespace std;

const int maxcol=60;
const int maxn=100+5;
string filenames[maxn];


//������������Ȳ���ʱ����ַ�extra
void Print(const string &s,int len,char extra)
{
	cout<<s;
	for(int i=0;i<len-s.length();i++)
	{
		cout<<extra;
	}
} 

int main()
{
	int n;
	//���벢�洢n���ļ��� 
	while(cin>>n)
	{//���ɹ�����n��ִ��������䡣��while������⣿ 
		int M=0;//��¼��ļ����ַ�����һ��Ҫ��ʼ��Ϊ0 
		for(int i=0;i<n;i++)
		{
			cin>>filenames[i];
			M=max(M,(int)filenames[i].length());//ֱ����STL�е�max��������֤MΪ����ַ��� 
		} 
			//��������cols������rows
			int cols=(maxcol-M)/(M+2)+1,row=(n-1)/cols+1;//?????
			Print("",60,'-');//����ʽ�����һ��--------- 
			cout<<"\n";
			sort(filenames,filenames+n);//���� 
			for(int r=0;r<row;r++)
			{
				for(int c=0;c<cols;c++)
				{
					int idx=c*row+r;//����������ǵڼ���Ԫ�ض�Ӧ���±� 
					if(idx<n)
						Print(filenames[idx],c==cols-1?M:M+2,' '); 
				}
			cout<<"\n";
		}
	}
	return 0;
} 















