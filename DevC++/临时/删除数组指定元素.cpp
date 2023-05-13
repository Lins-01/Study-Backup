#include<iostream>
#define N 4
using namespace std;
int main()
{
    int r[N]={7,7,6,7};
    int newR[N]; 
	int x=7; 
	int index=0;
	for(int i=0;i<4;i++)
    {
		if(r[i]!=x)
		{
			newR[index]=r[i];
			cout<<newR[index];
			index++;
		}
    }
    
    return 0;
}
