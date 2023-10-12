from django.contrib import admin

# .models是本路径下的models
# 再导入offer
from .models import Product, Offer


# Register your models here.

# 类名字约定就用  要展示的类名+Admin
class ProductAdmin(admin.ModelAdmin):
    # 继承后重写list_display
    # 设置为需要展示的属性
    list_display = ['name','price','stock']

# 注册的是Product
# 再加一个参数显示类的具体属性
admin.site.register(Product,ProductAdmin)

class OfferAdmin(admin.ModelAdmin):
    list_display = ['code','discount']

admin.site.register(Offer,OfferAdmin)