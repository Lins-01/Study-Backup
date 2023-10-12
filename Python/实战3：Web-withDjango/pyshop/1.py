#!/usr/bin/python
from functools import reduce

def add(x, y):
    return x + y
sum1 = reduce(add, [1,2,3,4,5])
sum2 = reduce(lambda x, y: x+y, [1,2,3,4,5])
print(sum1)
print(sum2)