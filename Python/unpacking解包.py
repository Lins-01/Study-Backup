
coordinates = (1, 2, 3)
# 不光元组，列表啥的容器也可以
coordinates2 = [4, 5, 6]

x = coordinates[0]
y = coordinates[1]
z = coordinates[2]

# unpacking 和上面等价
a, b, c = coordinates
e, f, g = coordinates2

print(x, y, z)
print(a, b, c)
print(e,f,g)