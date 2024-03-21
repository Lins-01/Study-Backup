# 导入gradio 一个快速部署上线你的模型demo的一个python库，调用库即可，只需要设置传入参数  fn=你的模型， input=图片 output=输出即可。
# 也可以直接将launch参数改为true获得24小时免费上线，过时后可在gradio官网买
import gradio as gr
# 导入transformers相关包
from transformers import pipeline
# 通过Interface加载pipeline并启动阅读理解服务
# 如果无法通过这种方式加载，可以采用离线加载的方式
gr.Interface.from_pipeline(pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa")).launch()
