#include<iostream>
using namespace std;
int main()
{
	int a,b,c,d,e;
	int count=0;
	for(a=1;a<=9;a++)
	{
		for(b=1;b<=9;b++)
		{
			if(a==b)continue;
			else{
					for(c=1;c<=9;c++)
				{
					if(c==b||c==a)continue;
					else{
						for(d=1;d<=9;d++)
						{
							if(d==c||d==a||d==b)continue;
							for(e=1;e<=9;e++)
							{
								if(e==a||e==b||e==c||e==d)continue;
								else{
									if((a*10+b)*(c*100+d*10+e)==(a*100+d*10+b)*(c*10+e))
									{
										count++;
									}
								}
							}
						}
					}
				}
			}
		}
	}
	cout<<count<<endl;
	return 0;
}
