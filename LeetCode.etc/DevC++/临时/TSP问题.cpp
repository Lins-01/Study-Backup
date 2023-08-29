#include <iostream>
#include<stdio.h>
#include<iomanip>
using namespace std;

int main()
{
    int n,temp,minDis;

        cin>>n;
        int dis[n][n];     //dis为距离矩阵

        for(int i=0;i<n;i++){     //以下为距离矩阵的初始化
            for(int j=0;j<n;j++){
                if(j!=i){
                    cin>>temp;
                    dis[i][j]=temp;
                }
                else{
                    dis[i][j]=1000;
                }
            }
        }


        int d[n][1<<(n-1)]; //1<<(n-1)]表示2的n-1次方，d[]为动态规划存储的城市经过矩阵
        for(int i=1;i<n;i++){     //将所有城市到第0个城市的路径初始化为两市间的距离
            d[i][0]=dis[i][0];
        }

        for(int j=1;j<1<<(n-1);j++){
            for(int i=1;i<n;i++){    //j用二进制表示的城市集合
                    if(((1<<(i-1))&j)==0){         //i不在j表示的城市集合中

                        minDis=60000;
                        for(int k=1;k<n;k++){
                        if(((1<<(k-1))&j)!=0)  {//k表示的城市在j表示的城市集合中

                        temp=dis[i][k]+d[k][j-(1<<(k-1))];
                        if(temp<minDis){
                            minDis=temp;   //所有k中最小的距离
                        }
                        }
                        }
                    }
                    d[i][j]=minDis;
            }
        }
        minDis=60000;
        for(int k=1;k<n;k++){
            temp=dis[0][k]+d[k][((1<<(n-1))-1)-(1<<(k-1))];
            if(minDis>temp){
                minDis=temp;
            }
        }
      /*  for(int i=0;i<n;i++){    //此部分可以输出看看形成的d[][]矩阵，便于理解执行过程
            for(int j=0;j<1<<(n-1);j++){
                cout<<d[i][j]<<"  ";
            }
            cout<<endl;
        } */
        d[0][(1<<(n-1))-1]=minDis;
        cout<<"输出结果："; 
        cout<<d[0][(1<<(n-1))-1];


    return 0;
}


