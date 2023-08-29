def emoji_converter(message):
    words = message.split(' ')
    outputs = ''
    # windows 按 win键+.  可以输入emoji
    emojis ={
        ':)':'😁',
        ':(':'😫'
    }

    for word in words:
        # 按键获取字典内的值，没有就可以输出默认
        # 默认输出可以自己用第二个参数指定
        outputs += emojis.get(word,word) + " "

    return outputs
message = input(">")
outputs = emoji_converter(message)
print(outputs)