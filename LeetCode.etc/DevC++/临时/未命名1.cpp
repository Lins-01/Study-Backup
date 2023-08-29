#include<stdio.h>

/*�ַ�������*/
int StringLength(char *L)
{
    int i = 0;                //��¼λ��
    int count = 0;            //����������¼����
    while(L[i])                //�жϵ�ǰλ���Ƿ�Ϊ��
    {
        count++;            //���ȼ�1
        i++;                //��������1
    }
    return count;            //���س���
}

/*����ƥ���㷨*/
int Index(char *S, char *T)                    //SΪ������TΪ�Ӵ�
{
    int Slength = StringLength(S);            //�������S�ĳ���
    int Tlength = StringLength(T);            //����Ӵ�T�ĳ���
    int i = 0;                                //��¼����S��ǰλ��
    int j = 0;                                //��¼�Ӵ�T��ǰλ��
//    int count = 0;
    printf("����ƥ���㷨��j���ݵ�ֵΪ��");
    while(i < Slength && j < Tlength)        //ȷ�������ַ����ĵ�ǰλ�þ�С���䳤��
    {
//        printf("%d ", j);
//        count++;
        if(S[i] == T[j])                    //�ж�����S��ǰλ�����Ӵ�T��ǰλ�õ��ַ��Ƿ����
        {
            i++;                            //����S�ĵ�ǰλ�ü�1�����ƣ�
            j++;                            //�Ӵ�T�ĵ�ǰλ�ü�1�����ƣ�
        }
        else                                //������ַ����ĵ�ǰλ���ַ�����
        {
            i = i - j + 1;                    //����S�ĵ�ǰλ��i���ݵ�j==0ʱiλ�õ���һλ��
            j = 0;                            //�Ӵ�T�ĵ�ǰλ��j��0
        }
    }    
//    printf("\nj���仯��%d��\n", count);
                                            //ѭ���Ƚ����
    if(j == Tlength)                        //�ж�λ��j����ֵ�Ƿ����Ӵ�T�ĳ������
        return i - Tlength;                    //���ǣ�˵�������ɹ�������T��S�г��ֵ���λ��
    else
        return -1;                            //�����ǣ�˵��T��������S�У�����-1
}

/*KMP�㷨*/
void Get_Next(char *T, int next[])
{
    int Tlength = StringLength(T);            //����ַ���T�ĳ���
    int i = 0;                                //T�ĺ�׺λ��
    int j = -1;                                //T��ǰ׺λ��
    next[0] = -1;                            //next�������λ��ֵΪ-1
    while(i < Tlength)                        //ȷ����׺λ��С�ڴ���
    {
        if(j == -1 || T[i] == T[j])            //���j==-1��˵��ǰ׺�Ѿ����˵���ǰ��
        {                                    //���T[i] == T[j]��˵����ǰǰ׺���׺���
            i++;                            //���׺λ�ú���һλ
            j++;                            //ǰ׺λ�ú���һλ
            next[i] = j;                    //��ǰ��׺λ�õ�nextֵΪj
        }
        else
            j = next[j];                    //����j���ˣ���û��ȫ�㶮���˵��ģ�
    }
}

int Index_KMP(char *S, char *T)
{
    int Slength = StringLength(S);            //�������S�ĳ���
    int Tlength = StringLength(T);            //����Ӵ�T�ĳ���
    int i = 0;                                //��¼S�ĵ�ǰλ��
    int j = 0;                                //��¼T�ĵ�ǰλ��
    int next[255];                                //next����
    Get_Next(T, next);                        //����Get_Next������Ϊnext��ֵ
    int count = 0;
//    printf("KMP�㷨��j���ݵ�ֵΪ��");
    while(i < Slength && j < Tlength)
    {
//        printf("%d ", j);
//        count++;
        if(j == -1 || S[i] == T[j])            //���j==-1��˵��ǰ׺�Ѿ����˵���ǰ��
        {                                    //���S[i] == T[j]��˵���������Ӵ���ǰλ���ַ����
            i++;                            //S�ĵ�ǰλ�ú���һλ
            j++;                            //T�ĵ�ǰλ�ú���һλ
        }
        else
        {
            j = next[j];                    //����j���ˣ�δŪ�����˵��ģ�
        }
    }
//    printf("\nj���仯��%d��\n", count);
    if(j == Tlength)                        //�ȽϽ������ж�j��ֵ�Ƿ���T�ĳ������
        return i - Tlength;                    //���ǣ�����T��S�г��ֵĿ�ʼλ��
    else
        return -1;                            //�����ǣ�����-1
}

/*KMP�Ľ����㷨*/
void Get_Next_Val(char *T, int nextVal[])
{
    int Tlength = StringLength(T);            //����Ӵ�T�ĳ���
    int i = 0;                                //��¼��׺λ��
    int j = -1;                                //��¼ǰ׺λ��
    nextVal[0] = -1;                        //next�����һλ�ø�ֵΪ-1
    while(i < Tlength)
    {
        if(j == -1 || T[i] == T[j])            //ͬ��
        {
            i++;                            //ͬ��
            j++;
            if(T[i] != T[j])                //���λ�ú���һλ���ֵ�����
                nextVal[i] = j;                //nextVal����j
            else                            //������
                nextVal[i] = nextVal[j];    //��ǰ��׺λ�õ�nextValֵ����jλ�õ�nextVal��ֵ
        }
        else
            j = nextVal[j];                    //ͬ��
    }
}

int Index_KMP_Val(char *S, char *T)
{
    int Slength = StringLength(S);            //�������S�ĳ���
    int Tlength = StringLength(T);            //����Ӵ�T�ĳ���
    int i = 0;                                //��¼S�ĵ�ǰλ��
    int j = 0;                                //��¼T�ĵ�ǰλ��
    int next[255];                                //next����
    Get_Next_Val(T, next);                        //����Get_Next������Ϊnext��ֵ
    int count = 0;
    printf("KMP_Val�㷨��j���ݵ�ֵΪ��");
    while(i < Slength && j < Tlength)
    {
        printf("%d ", j);
        count++;
        if(j == -1 || S[i] == T[j])            //���j==-1��˵��ǰ׺�Ѿ����˵���ǰ��
        {                                    //���S[i] == T[j]��˵���������Ӵ���ǰλ���ַ����
            i++;                            //S�ĵ�ǰλ�ú���һλ
            j++;                            //T�ĵ�ǰλ�ú���һλ
        }
        else
        {
            j = next[j];                    //����j���ˣ�δŪ�����˵��ģ�
        }
    }
    printf("\nj���仯��%d��\n", count);
    if(j == Tlength)                        //�ȽϽ������ж�j��ֵ�Ƿ���T�ĳ������
        return i - Tlength;                    //���ǣ�����T��S�г��ֵĿ�ʼλ��
    else
        return -1;                            //�����ǣ�����-1
}


    
void main()
{
    char *S = "aaaaaaaaaaaaaaaaaaaaabcde";
    char *T = "aaaaaaaaaaaaaaaaaaaaaaaax";
    int pos;
    pos = Index(S, T);
    if(pos != -1)
        printf("����ƥ���㷨���Ӵ�T������S���±�Ϊ%d��λ���Ͽ�ʼ����\n", pos);
    else
        printf("����ƥ���㷨���Ӵ�T������������S��\n");
    printf("---------------------------------------------------------------------\n");
    int pos_KMP;
    pos_KMP = Index_KMP(S, T);
    if(pos_KMP != -1)
        printf("KMPƥ���㷨���Ӵ�T������S���±�Ϊ%d��λ���Ͽ�ʼ����\n", pos_KMP);
    else
        printf("KMPƥ���㷨���Ӵ�T������������S��\n");
    printf("---------------------------------------------------------------------\n");
    int pos_KMP_val;
    pos_KMP_val = Index_KMP_Val(S, T);
    if(pos_KMP_val != -1)
        printf("KMP_Valƥ���㷨���Ӵ�T������S���±�Ϊ%d��λ���Ͽ�ʼ����\n", pos_KMP_val);
    else
        printf("KMP_Valƥ���㷨���Ӵ�T������������S��\n");
}
