print('hello world!')
#input()输入的返回值是字符串，要用数字的话，需要强转

# #懒得每次输入，先注释了
x1=input("请输入第一个数：")
# x2=input("请输入第二个数：")
# res=int(x1)*int(x2)
# print(x1,'x',x2,"=",res)
# #Python没有;结尾
# #冒号:结尾时，缩进的语句视为代码块。
# #不加冒号，报错
# if(res>10):
#     print('结果大于10')
# #Python中else if=elif
# elif(res<10):
#     print("结果小于10")
# else:
#     print('结果等于10')


#Python允许在数字中间以_分隔,方便数很大时看清楚。
#写成10_000_000_000和10000000000是完全一样的。
#十六进制数也可以写成0xa1b2_c3d4。

#字符串用""" ''都可以
## 但当字符串自身有'时，用""，比如"I'm fine.
## 如果字符串内部，包含'和",则可以用转义字符\来表示
## r'',表示''内部的字符串不进行转义

# python用'''xxx'''表示换行 等价于\n，因为觉得比\n更方便阅读
print('''隐恶扬善
执其两端
意思是说：多说别人的好处，少说别人的坏处，要记住这两点''')

# //地板除，整数相除，取整,就相当于c语言里的10/3，整除。
# python里的10/3得到的仍包含小数。
print('10/3=',10/3)
print('10/3取整=',10//3)

# 编码发展问题ASCII->Unicode->UTF-8
# https://www.liaoxuefeng.com/wiki/1016959663602400/1017075323632896

#占位符 %d %s %f %x 整数 字符串 浮点数 十六进制整数
#有几个%?占位符，后面就跟几个变量或者值，顺序要对应好。
#如果只有一个%?，括号可以省略。
# %s会把任何类型转换为字符串，不知道用什么可以用%s
#符串里面的%是一个普通字符怎么办？这个时候就需要转义，用%%来表示一个%
# 也可以用f-string来控制
#{xxx}里面xxx对应的变量，:.2f之类可以用来控制输出
r=2.5
s=3.14*r**2
print(f'The area of a circle with radius {r} is {s:.2f}')

#Python的list就是其他语言的数组，但数据类型更广

# len()获得字符串长度，和list中个数
print('字符串ABC长度为：',len('ABC'))
classmates=['tim','jack','niko']
print(classmates)
print('classmates的长度为：',len(classmates))
#索引访问list
print('索引访问list',classmates[0])

#Python中-1表示可以表示最后一个元素
print("python中输出列表最后一个元素",classmates[-1])
#获取list倒数第x个元素
print('获取list倒数第3个元素:',classmates[-3])

#插入 删除元素
#末尾添加
classmates.append('s1mple')
print(classmates)
#指定索引位置添加
classmates.insert(1,'zywOo')
print(classmates)
#末尾删除
classmates.pop()
print(classmates)
#删除指定位置元素  .pop(i)
classmates.pop(0)
print(classmates)
#替换元素
classmates[-2]='s1mple'
print(classmates)


#list里面的元素类型可以不一样
L=[123,'abc',True]
print(L)

#list里的元素也可以是另一个list
# 以此可以构成多维数组
s=['cc',['aa','bb'],'dd']
#访问aa
print("way1: 访问多维list:",s[1][0])
#或者如下定义也一样
p=['aa','bb']
s=['cc',p,'dd']
print("way2: 访问多维list:",s[1][0])
#L为空 长度为0
L=[]
print(len(L))

#list用方括号[]，tuple用的小括号()

#tuple与list基本一样，不过里面的数据定义后就不能动。
#不可变的tuple有什么意义？
# 因为tuple不可变，所以代码更安全。如果可能，能用tuple代替list就尽量用tuple。
#tuple的陷阱
#要定义一个只有1个元素的tuple，如果你这么定义
t=(1)
print('tuple的陷阱',t)
#定义的不是tuple，是1这个数！
#这是因为括号()既可以表示tuple，又可以表示数学公式中的小括号，这就产生了歧义
#因此，Python规定，这种情况下，按小括号进行计算，计算结果自然是1。
#所以，只有1个元素的tuple定义时必须加一个逗号,，来消除歧义
t=(1,)
print('tuple的陷阱的解决',t)  

#若tuple里面包含list，包含的list内容可以变
#所以tuple不变，指的大概是tuple指向的地址不变。


# range()，生成一个递增整数序列，从0开始
# 生成一个递增序列的list
l=list(range(11))
print(l)
#循环求和
sum=0
for x in l:
    sum=x+sum
print("计算1~10累加：",sum)
sum2=0
for x in range(6):
    sum2=x+sum2
print("计算1~5累加：",sum2)

# break 与continue与c中一样

# 键值对 字典 dict 其他语言叫map
d={'zywOo':100,'niko':99,'s1mple':98}
print('字典键值对查找zywOo：',d['zywOo'])
# dict中添加元素，用key
d['monesy']=90
print(d)
# 覆盖已有key的value
d['s1mple']=0
print(d)
#判断key是否存在
print('cc' in d)
print('niko' in d)
#dict的get方法
# 默认返回value,如果没有返回none，或者可以自己指定
print(d.get('niko'))
print(d.get('cc',-1))
#删除元素。pop
d.pop('monesy')
print(d)

#dict与list对比
# 查找和插入的速度极快，不会随着key的增加而变慢；
# 空间换时间，需要占用大量的内存，内存浪费多。

#set可以看成数学意义上的无序和无重复元素的集合
# 因此，两个set可以做数学意义上的交集、并集等操作
s = set([1, 1, 2, 2, 3, 3])
#打印出来只会有1，2，3
print(s)

s1=set([1,2,3])
s2=set([2,3,4])
#交集运算
print(s1&s2)
#并集运算
print(s1|s2)

# 字符串replace函数
a = 'abc'
b=a.replace('a','A')
#a没变，另复制个字符串到b
print('a=',a,'  b=',b)


print('*'*20)