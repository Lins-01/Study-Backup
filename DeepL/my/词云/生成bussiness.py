


import jieba
import wordcloud
# 读取文本
with open("C:/Users/lins/Desktop/business.txt",encoding="utf-8") as f:
    s = f.read()
print(s)
ls = jieba.lcut(s) # 生成分词列表
text = ' '.join(ls) # 连接成字符串


# 不想出现的词按同样格式添加进去就好： 英文的逗号和引号
stopwords = ['the', 'a','are' ,'was','has','an','if','their', 'of', 'to', 'and', 'or', 'but', 'is', 'it','its','them' ,'they','there','that', 'this', 'these', 'those', 'in', 'on', 'at', 'for', 'with', 'from', 'by', 'as', 'about', 'have', 'be', 'use', 'do', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'all', 'any', 'none', 'only', 'some', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'] # 去掉不需要显示的词

wc = wordcloud.WordCloud(font_path="C:\Windows\Fonts\calibrili.ttf",
                         width = 1000,
                         height = 700,
                         background_color='white',
                         max_words=100,stopwords=stopwords)

wc.generate(text) # 加载词云文本
wc.to_file("./bussiness.png") # 保存词云文件




