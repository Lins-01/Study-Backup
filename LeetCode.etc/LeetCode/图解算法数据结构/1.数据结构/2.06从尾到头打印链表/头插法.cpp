/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> reversePrint(ListNode* head) {
       
        //###############区别 头结点与头指针  ##############
        //头结点有数据域与指针域，头指针仅指针
        //ListNode *t;定义一个头指针，用来遍历链表
        ListNode *t;
       //ListNode *n= new ListNode(); 定义一个头结点，也包含申请头结点地址
        //new.....相当于malloc申请地址，
        ListNode *n= new ListNode();//定义一个头结点
        //也可以说定义了一个链表。只要指针n或者其他指针，指向链表头地址，即可叫这个链表
        


        //这题默认 head是头节点，但是也是第一个元素
        //有头结点方便，自己加一个就是
     
        
        while(head!=NULL)
        {
            //###头插法 以实现倒序


            //引入新变量 保存开始的head->next;(之后语句会覆盖掉)
            t=head->next;
            //head为新加入的结点
            head->next=n->next;//插入结点，先接后断,
            n->next=head;//将新结点head插入新链表中，
            //head=head->next;//这里head->next值已经被上面语句覆盖，后面想直接用head=head->next往下走，不可行
            head=t;//执行完操作，往下走一个
        }

        //链表中已翻转，题目要求用数组输出。
        //存入数组中1
        vector<int> arry;
        ListNode *temp=n->next;//n在头插法跑完之后，成为了一个指向第一个元素结点的指针
        while(temp!=NULL)
        {
            arry.push_back(temp->val);
            temp=temp->next;
        }
        return arry;
    }
};