#include<stdio.h>
#define PI 3.14159265358979323

int main()
{
	int r; 
	double s;
	scanf("%d",&r);
	s=PI*r*r; 
	printf("%.7f",s); //pintf的格式化输出，可以.7直接实现输出7位小数，并四舍五入 
	return 0;
}
