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
        //####递归 
        //递归遍历链表到表尾 再退出时输出即实现倒序、

        recur(head);
        return res;
    }
/* 上面是leetcode规定的函数，返回值只能是vector<int> 
*所以直接在函数内写递归，在执行if return;时返回的不是vector<int>型，返回类型不匹配
 */
private:
    vector<int> res;
    void recur(ListNode* head){
        if(head==NULL) return ;
        recur(head->next);
        res.push_back(head->val);
    }
};