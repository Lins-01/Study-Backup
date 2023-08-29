#include <stdio.h>
#include<stdlib.h>
int quick_select(int a[], int l, int r, int k) {
    int p = rand() % (r - l + 1) + l;//应该是取数组中任意一个元素 
    int x = a[p];
    {int t = a[p]; a[p] = a[r]; a[r] = t;}//数组中p位置元素和最后一个元素互换位置 
    int i = l, j = r;
    while(i < j) {
        while(i < j && a[i] < x) i++;//找到大于等于尾部元素的下标 
        if(i < j) {
            a[j] = a[i];
            j--;
        }
        while(i < j && a[j] > x) j--;//找到小于等于的 
        if(i < j) {
            a[i] = a[j];
            i++;
        }
    }
    a[i] = x;
    p = i;
    if(i - l + 1 == k) return a[i];
    if(i - l + 1 < k) return quick_select(a,i+1,r,(k-(i-l+1))); //填空
    else return quick_select(a, l, i - 1, k);
}
    
int main()
{
    int a[] = {1, 4, 2, 8, 5, 7, 23, 58, 16, 27, 55, 13, 26, 24, 12};
    printf("%d\n", quick_select(a, 0, 14, 5));
    return 0;
}
