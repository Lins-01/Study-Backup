/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */

class Solution{
public:
    vector<int> reversePrint(ListNode* head)
    {
        //####即用vector模拟数据结构----栈

        //遍历单链表 存入vector数组中
        ListNode *temp;
        temp=head;
        
        vector<int> arry;
        while(temp!=NULL)//是temp!=NULL，不是temp->next,
        {
            arry.push_back(temp->val);//c++中不要用temp.next与对象.方法函数的用法相同，冲突
            temp=temp->next;
        }

        //翻转数组
        int tt;
        int len=arry.size();
        /**
         * 要加上判断:因为当数组为空时，len-1为-1 for循环出错
         * Line 1033: Char 9: runtime error: reference binding to null pointer of type 'int' (stl_vector.h)
        */
        //if(len==0)return arry;//为空时直接返回空数组即可
        /* for(int i=0,j=(len-1);i!=j;i++,j--)
        {
            tt=arry[i];
            arry[i]=arry[j];
            arry[j]=tt;
        } */
        /**
         * 翻转数组，不可以用i,j两个变量，分别从头或尾来换，因为当个数为偶数时，报错
         */
        for(int i=0;i<len/2;i++)
        {
            tt=arry[i];
            arry[i]=arry[len-i-1];
            arry[len-i-1]=tt;
        }


        return arry;//
        //return {arry.rbegin(),arry.rend()};
        /**
         * 使用rbegin()这种，整体时间空间消耗差不多 都是8.0+
         */
    }

};