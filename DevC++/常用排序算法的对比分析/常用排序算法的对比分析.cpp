#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 直接插入算法
//从小到大排列 
void InsertionSort(int *arr, int size)    
{    
    int i, j, tmp;    
    for (i = 1; i < size; i++) {  
	//如果i位比前面的位要小 说明i位要插入前面的序列    
        if (arr[i] < arr[i-1]) {    
            tmp = arr[i];    
            for (j = i - 1; j >= 0 && arr[j] > tmp; j--) {  
            //将i位即将要放入的位置后面的元素全部后移 
                arr[j+1] = arr[j];   
            }  
            //将i位放在合适的位置 
            arr[j+1] = tmp;    
        }          
    }    
}    

// 快速排序

void swap(int *a, int *b)    
{  
    int temp;  
    temp = *a;  
    *a = *b;  
    *b = temp;  
}  

void QuickSort(int *arr, int maxlen, int begin, int end)  
{  
    int i, j;  
    if (begin < end) {
        i = begin + 1;  
        j = end;        
        //保证i和j从俩边向中间遍历数组  直到i==j为止
		//取数组第一个元素为基准值 
        while (i < j) {
        	//找到比基准大的i 和基准交换 
            if(arr[i] > arr[begin]) {  
            
                swap(&arr[i], &arr[j]); 
                j--;
            } else {//找到比基准小的j 和基准交换 
                i++; 
            }  
        }  
        //把基准项移动到中间
        if (arr[i] >= arr[begin]) {  
            i--;  
        }  
        swap(&arr[begin], &arr[i]);      
        QuickSort(arr, maxlen, begin, i);  
        QuickSort(arr, maxlen, j, end);  
    }  
}  


   
int arr[200000];
int arr2[200000];

int main(int argc, char const *argv[])
{
    int i;

    srand((unsigned)time(NULL));
    for ( i = 0; i < 200000; i++)
    {
        arr2[i] = (rand()%500000);
    }

    clock_t start2,finish2;
    double totaltime2;
    start2=clock();
    QuickSort(arr2,200000,0,19999);
    finish2=clock();
    totaltime2=(double)(finish2-start2)/CLOCKS_PER_SEC;
    printf("快速排序算法运行的时间为%f",totaltime2);

    printf("\n");

    srand((unsigned)time(NULL));
    for ( i = 0; i < 200000; i++)
    {
        arr[i] = (rand()%500000);
    }
    clock_t start,finish;
    double totaltime;
    start=clock();
    InsertionSort(arr,200000);
    finish=clock();
    totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
    printf("直接插入算法运行的时间为%f",totaltime);

    return 0;
}

