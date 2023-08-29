import random

# random.来调用

# range，enumerate，max，min这些是python自带的函数
for i in range(3):
    # random.random()，默认返回[0,1)之间的随机数
    print(random.random())


members = ['John','niko','mosh']
print(random.choices(members,k=2))

# 一个掷色子的类
class Dice:
    # 每次调用得到两个筛子点数，元组返回
    # 类里面定义函数，第一步必须传入一个参数self
    def roll(self):
        return random.randint(1,6),random.randint(1,6)


niko = Dice()
print(niko.roll())