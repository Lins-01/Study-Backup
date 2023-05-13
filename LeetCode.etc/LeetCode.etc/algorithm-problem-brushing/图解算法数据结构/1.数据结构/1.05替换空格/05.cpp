class Solution {
public:
    string replaceSpace(string s) {
        string res;

        //####新的for循环实现方式
        //auto可以根据赋值自动推导变量类型
       
        
        //测试后，&引用加不加都正确


        for(auto &c:s) 
            if(c==' '){
                res.push_back('%');
                res.push_back('2');
                res.push_back('0');
            }else{
                res.push_back(c);
            }
        }
        return res;
    }
};
