class Solution{
public:
    string replaceSpace(string s){
        //上个解法时空复杂都为N,
        //利用c++字符串单个字符为可更改的性质，降低空间复杂度
        //不需像java和python一般只能用另一个数组保存修改后的字符串。
        

        //倒序遍历，根据空格个数修改过长度的字符串s
        //i,j分别指向旧新字符串的尾部
        //s[i]=空格时，就该j-2,j为%20
        //s[i]!=空格，s[j]=s[i]

        //修改s长度，需得知多少个空格
        int count_space=0;
        int len = s.size();//记录原本长度，因为之后倒序遍历修改长度后的s

        //遍历得知空格数
        for (auto c:s)
        {
            //if(c==" ")count_space++; 报错Char 17: warning: result of comparison against a string literal is unspecified
            //一个空格，一个字符一个字符的比较，用单引号！！！
            if(c==' ')count_space++;
        }

        //修改s长度
        s.resize(len+count_space*2);

        //倒序遍历
        for(int i=len,j=s.size();i!=j;i--)
        {
            if(s[i]==' '){
                s[j-2]='%';
                s[j-1]='2';
                s[j]='0';
                j-=3;//此时j要移动三次
            }else{
                s[j]=s[i];
                j--;
            }
        }

        return s;
    }



};