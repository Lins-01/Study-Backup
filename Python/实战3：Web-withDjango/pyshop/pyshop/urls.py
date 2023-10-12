"""
URL configuration for pyshop project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
# 导入include因为products要用
from django.urls import path,include

# 整个项目的总url管理

urlpatterns = [
    # 初始化项目就会有的
    # 表示pyshop内部的每个子模块都是由admin托管
    path("admin/", admin.site.urls),

    # 也在这里注册其它子模块的url 
    # 告诉任何url从/products开始的都发给products.urls
    path('products/',include('products.urls'))
]
