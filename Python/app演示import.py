# 导入整个module
# import utils演示module

# 导入module中的某个函数或者类
from utils演示module import find_min

# 导入package中的module
import ecommerce.shipping
from ecommerce import shipping
# 导入module中的函数,一次多个
from ecommerce.shipping import calc_shipping,calc_tax

numbers =[8,10,3,1]

# max = utils演示module.find_max(numbers)
# print(max)

# max和min函数在python中本来就内置了
# 这里在pycharm中需要改名
min = find_min(numbers)
print(min)
calc_shipping()
calc_tax()