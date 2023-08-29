# exception 例外
# 就是程序有异常值时，执行expection下面的内容
# 异常也有很多类型，值异常valueerror，除0异常ZeroDivisionError


# 如果不用expection处理也可以，就是程序不正常退出，报错
# 你就是要对报错的每种自己写一种希望的解决方案
# 用expection处理程序执行完expection后正常退出

try: 
    age = int(input("Age:"))
    income = 2000
    risk = income / age
    print(age)
except ZeroDivisionError:
    print("Age cannot be 0.")
except ValueError:
    print('Invalid value')