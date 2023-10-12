# 通过HttpResonpse这个类，我们可以创建一个http响应，返回到客户端
from django.http import HttpResponse

from django.shortcuts import render
# 导入要展示的类
from .models import Product

# Create your views here.

# 要有request参数，服务器把用户的https请求传入这里
def index(request):
    # 拿到要展示的数据
    # 除了all还有很多方法
    # filter按条件，get展示单个
    # save方法用来 插入新产品/更新现有产品
    products = Product.objects.all()

    # render()用来渲染页面
    return render(request,'index.html',
                  {'products':products})

# 在子模块中新建下级页面就在子模块中写个view function+注册url就好了
def new(request):
    return HttpResponse('New Product')
