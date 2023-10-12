# path函数可以映射url 到 view function
from django.urls import path
# 导入views，要用里面的函数
# .表示从当前目录 / views是个通用名字，所以不能直接import
from . import views

#

# 一定要有个这样取名的变量
urlpatterns = [
    # ''表示根目录，也就是/product
    # 引用views.index , 注意不是不调用，需要的时候才会调用
    # 这个函数由Django托管
    path('',views.index),
    path('new',views.new)
]