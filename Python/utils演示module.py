# 在module里面定义一些类或者函数
# import这个module的文件，就可以用他们了

# 这样可以避免代码都写在一个文件里，导致又臭又长

def find_max(numbers):
    max = numbers[0]
    for index in numbers:
        if index>max:
            max=index
    return max


def find_min(numbers):
    min = numbers[0]
    for index in numbers:
        if index<min:
            min=index
    return min