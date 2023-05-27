# 示例代码
from wordcloud import WordCloud
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import jieba

# 打开文本
with open("C:/Users/lins/Desktop/business.txt",encoding="utf-8") as f:
    s = f.read()

# 中文分词
text = ' '.join(jieba.cut(s))

# 生成对象
img = Image.open("1.jpg") # 打开遮罩图片
mask = np.array(img) #将图片转换为数组

stopwords = ['the', 'a','are' ,'was','has','an','if','their', 'of', 'to', 'and', 'or', 'but', 'is', 'it','its','them' ,'they','there','that', 'this', 'these', 'those', 'in', 'on', 'at', 'for', 'with', 'from', 'by', 'as', 'about', 'have', 'be', 'use', 'do', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'all', 'any', 'none', 'only', 'some', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'] # 去掉不需要显示的词

wc = WordCloud(font_path="C:\Windows\Fonts\calibrili.ttf",
               mask=mask,
               width = 1000,
               height = 700,
               background_color='white',
               max_words=200,
               stopwords=stopwords).generate(text)

# 显示词云
plt.imshow(wc, interpolation='bilinear')# 用plt显示图片
plt.axis("off")  # 不显示坐标轴
plt.show() # 显示图片

# 保存到文件
wc.to_file("bussiness_mask.png")