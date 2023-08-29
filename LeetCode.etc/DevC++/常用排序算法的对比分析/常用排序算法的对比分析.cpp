#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// ֱ�Ӳ����㷨
//��С�������� 
void InsertionSort(int *arr, int size)    
{    
    int i, j, tmp;    
    for (i = 1; i < size; i++) {  
	//���iλ��ǰ���λҪС ˵��iλҪ����ǰ�������    
        if (arr[i] < arr[i-1]) {    
            tmp = arr[i];    
            for (j = i - 1; j >= 0 && arr[j] > tmp; j--) {  
            //��iλ����Ҫ�����λ�ú����Ԫ��ȫ������ 
                arr[j+1] = arr[j];   
            }  
            //��iλ���ں��ʵ�λ�� 
            arr[j+1] = tmp;    
        }          
    }    
}    

// ��������

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
        //��֤i��j���������м��������  ֱ��i==jΪֹ
		//ȡ�����һ��Ԫ��Ϊ��׼ֵ 
        while (i < j) {
        	//�ҵ��Ȼ�׼���i �ͻ�׼���� 
            if(arr[i] > arr[begin]) {  
            
                swap(&arr[i], &arr[j]); 
                j--;
            } else {//�ҵ��Ȼ�׼С��j �ͻ�׼���� 
                i++; 
            }  
        }  
        //�ѻ�׼���ƶ����м�
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
    printf("���������㷨���е�ʱ��Ϊ%f",totaltime2);

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
    printf("ֱ�Ӳ����㷨���е�ʱ��Ϊ%f",totaltime);

    return 0;
}

