#include <iostream>
#include<stdio.h>
#include<iomanip>
using namespace std;

int main()
{
    int n,temp,minDis;

        cin>>n;
        int dis[n][n];     //disΪ�������

        for(int i=0;i<n;i++){     //����Ϊ�������ĳ�ʼ��
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


        int d[n][1<<(n-1)]; //1<<(n-1)]��ʾ2��n-1�η���d[]Ϊ��̬�滮�洢�ĳ��о�������
        for(int i=1;i<n;i++){     //�����г��е���0�����е�·����ʼ��Ϊ���м�ľ���
            d[i][0]=dis[i][0];
        }

        for(int j=1;j<1<<(n-1);j++){
            for(int i=1;i<n;i++){    //j�ö����Ʊ�ʾ�ĳ��м���
                    if(((1<<(i-1))&j)==0){         //i����j��ʾ�ĳ��м�����

                        minDis=60000;
                        for(int k=1;k<n;k++){
                        if(((1<<(k-1))&j)!=0)  {//k��ʾ�ĳ�����j��ʾ�ĳ��м�����

                        temp=dis[i][k]+d[k][j-(1<<(k-1))];
                        if(temp<minDis){
                            minDis=temp;   //����k����С�ľ���
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
      /*  for(int i=0;i<n;i++){    //�˲��ֿ�����������γɵ�d[][]���󣬱������ִ�й���
            for(int j=0;j<1<<(n-1);j++){
                cout<<d[i][j]<<"  ";
            }
            cout<<endl;
        } */
        d[0][(1<<(n-1))-1]=minDis;
        cout<<"��������"; 
        cout<<d[0][(1<<(n-1))-1];


    return 0;
}


