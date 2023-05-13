#include<stdio.h>

/*字符串长度*/
int StringLength(char *L)
{
    int i = 0;                //记录位置
    int count = 0;            //计数器，记录长度
    while(L[i])                //判断当前位置是否为空
    {
        count++;            //长度加1
        i++;                //计数器加1
    }
    return count;            //返回长度
}

/*朴素匹配算法*/
int Index(char *S, char *T)                    //S为主串，T为子串
{
    int Slength = StringLength(S);            //获得主串S的长度
    int Tlength = StringLength(T);            //获得子串T的长度
    int i = 0;                                //记录主串S当前位置
    int j = 0;                                //记录子串T当前位置
//    int count = 0;
    printf("朴素匹配算法中j回溯的值为：");
    while(i < Slength && j < Tlength)        //确保两个字符串的当前位置均小于其长度
    {
//        printf("%d ", j);
//        count++;
        if(S[i] == T[j])                    //判断主串S当前位置与子串T当前位置的字符是否相等
        {
            i++;                            //主串S的当前位置加1（后移）
            j++;                            //子串T的当前位置加1（后移）
        }
        else                                //如果两字符串的当前位置字符不等
        {
            i = i - j + 1;                    //主串S的当前位置i回溯到j==0时i位置的下一位置
            j = 0;                            //子串T的当前位置j归0
        }
    }    
//    printf("\nj共变化了%d次\n", count);
                                            //循环比较完毕
    if(j == Tlength)                        //判断位置j的数值是否与子串T的长度相等
        return i - Tlength;                    //若是，说明搜索成功，返回T在S中出现的首位置
    else
        return -1;                            //若不是，说明T不存在与S中，返回-1
}

/*KMP算法*/
void Get_Next(char *T, int next[])
{
    int Tlength = StringLength(T);            //获得字符串T的长度
    int i = 0;                                //T的后缀位置
    int j = -1;                                //T的前缀位置
    next[0] = -1;                            //next数组的首位赋值为-1
    while(i < Tlength)                        //确保后缀位置小于串长
    {
        if(j == -1 || T[i] == T[j])            //如果j==-1，说明前缀已经回退到最前方
        {                                    //如果T[i] == T[j]，说明当前前缀与后缀相等
            i++;                            //则后缀位置后移一位
            j++;                            //前缀位置后移一位
            next[i] = j;                    //当前后缀位置的next值为j
        }
        else
            j = next[j];                    //否则，j回退（还没完全搞懂回退到哪）
    }
}

int Index_KMP(char *S, char *T)
{
    int Slength = StringLength(S);            //获得主串S的长度
    int Tlength = StringLength(T);            //获得子串T的长度
    int i = 0;                                //记录S的当前位置
    int j = 0;                                //记录T的当前位置
    int next[255];                                //next数组
    Get_Next(T, next);                        //调用Get_Next函数，为next赋值
    int count = 0;
//    printf("KMP算法中j回溯的值为：");
    while(i < Slength && j < Tlength)
    {
//        printf("%d ", j);
//        count++;
        if(j == -1 || S[i] == T[j])            //如果j==-1，说明前缀已经回退到最前方
        {                                    //如果S[i] == T[j]，说明主串与子串当前位置字符相等
            i++;                            //S的当前位置后移一位
            j++;                            //T的当前位置后移一位
        }
        else
        {
            j = next[j];                    //否则，j回退（未弄懂回退到哪）
        }
    }
//    printf("\nj共变化了%d次\n", count);
    if(j == Tlength)                        //比较结束，判断j的值是否与T的长度相等
        return i - Tlength;                    //若是，返回T在S中出现的开始位置
    else
        return -1;                            //若不是，返回-1
}

/*KMP改进版算法*/
void Get_Next_Val(char *T, int nextVal[])
{
    int Tlength = StringLength(T);            //获得子串T的长度
    int i = 0;                                //记录后缀位置
    int j = -1;                                //记录前缀位置
    nextVal[0] = -1;                        //next数组第一位置赋值为-1
    while(i < Tlength)
    {
        if(j == -1 || T[i] == T[j])            //同上
        {
            i++;                            //同上
            j++;
            if(T[i] != T[j])                //如果位置后移一位后的值不相等
                nextVal[i] = j;                //nextVal等于j
            else                            //如果相等
                nextVal[i] = nextVal[j];    //当前后缀位置的nextVal值等于j位置的nextVal的值
        }
        else
            j = nextVal[j];                    //同上
    }
}

int Index_KMP_Val(char *S, char *T)
{
    int Slength = StringLength(S);            //获得主串S的长度
    int Tlength = StringLength(T);            //获得子串T的长度
    int i = 0;                                //记录S的当前位置
    int j = 0;                                //记录T的当前位置
    int next[255];                                //next数组
    Get_Next_Val(T, next);                        //调用Get_Next函数，为next赋值
    int count = 0;
    printf("KMP_Val算法中j回溯的值为：");
    while(i < Slength && j < Tlength)
    {
        printf("%d ", j);
        count++;
        if(j == -1 || S[i] == T[j])            //如果j==-1，说明前缀已经回退到最前方
        {                                    //如果S[i] == T[j]，说明主串与子串当前位置字符相等
            i++;                            //S的当前位置后移一位
            j++;                            //T的当前位置后移一位
        }
        else
        {
            j = next[j];                    //否则，j回退（未弄懂回退到哪）
        }
    }
    printf("\nj共变化了%d次\n", count);
    if(j == Tlength)                        //比较结束，判断j的值是否与T的长度相等
        return i - Tlength;                    //若是，返回T在S中出现的开始位置
    else
        return -1;                            //若不是，返回-1
}


    
void main()
{
    char *S = "aaaaaaaaaaaaaaaaaaaaabcde";
    char *T = "aaaaaaaaaaaaaaaaaaaaaaaax";
    int pos;
    pos = Index(S, T);
    if(pos != -1)
        printf("朴素匹配算法：子串T在主串S的下标为%d的位置上开始出现\n", pos);
    else
        printf("朴素匹配算法：子串T不存在与主串S中\n");
    printf("---------------------------------------------------------------------\n");
    int pos_KMP;
    pos_KMP = Index_KMP(S, T);
    if(pos_KMP != -1)
        printf("KMP匹配算法：子串T在主串S的下标为%d的位置上开始出现\n", pos_KMP);
    else
        printf("KMP匹配算法：子串T不存在与主串S中\n");
    printf("---------------------------------------------------------------------\n");
    int pos_KMP_val;
    pos_KMP_val = Index_KMP_Val(S, T);
    if(pos_KMP_val != -1)
        printf("KMP_Val匹配算法：子串T在主串S的下标为%d的位置上开始出现\n", pos_KMP_val);
    else
        printf("KMP_Val匹配算法：子串T不存在与主串S中\n");
}
