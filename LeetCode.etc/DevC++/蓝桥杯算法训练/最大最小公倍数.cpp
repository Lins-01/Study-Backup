#include<iostream>
using namespace std;

int main()
{
	long long n,N,max,tmp;
	
	cin>>N;//数据输入
	//用贪心的思想，找三个最大的数（即每次都是最优解
	for(n=N;n>1;n--)
	{
		//三个数都用n来表示
		 //n为奇数时，最大最小公倍数为n(n-1)(n-1)    相邻的俩个自然数一定为互质数 相邻俩个奇数一定为互质数 
		 if(n%2!=0)
		 {
		 	tmp=n*(n-1)*(n-2);
		 	if(max<tmp)max=tmp;
		 }else{//n为偶数时，  后俩位数在一起不定互质 所以都后移一位(n-1)(n-2)(n-3)但n(n-1)(n-3)看起来大一点，只要保证n与n-3不互质就好。俩个数质数只会为3 
		 	if(n%3==0)
		 	{
		 		tmp=(n-1)*(n-2)*(n-3);
		 		if(max<tmp)max=tmp;
		 	}else{
		 		tmp=n*(n-1)*(n-3);
		 		if(max<tmp)max=tmp;
		 	}
		 }
	} 
	cout<<max<<endl;
	return 0;
} 
