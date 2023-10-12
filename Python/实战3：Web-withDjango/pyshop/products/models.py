from django.db import models

# 定义项目中需要的各种类的地方

# Create your models here.

class Product(models.Model):
    # 字符类型的数据 ，也是用models下面的类来定义
    # 设置最大长度，防止恶意输入巨大长度
    name = models.CharField(max_length=255)
    # 浮点数定义价格
    price = models.FloatField()
    # 整数定义sotck（这里应该库存意思
    stock = models.IntegerField()
    # 定义图像，也用char
    # url最长是2083
    image_url = models.CharField(max_length=2083)


class Offer(models.Model):
    code = models.CharField(max_length=10)
    description = models.CharField(max_length=255)
    discount = models.FloatField()