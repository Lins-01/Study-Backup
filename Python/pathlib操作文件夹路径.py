# 导入Path类，大写是个类
from pathlib import Path

# 不传入参数，默认表示当前目录的路径。
path = Path()
# path.glob()按输入的相关模式，找所有文件 '*.*'，输出所有
for file in path.glob('*.py'):
    print(file)