import time
import pyautogui
import pygetwindow as gw
import pandas as pd
from tee import StdoutTee

# 定义Excel文件路径
excel_file_path = './value.csv'
# 读取Excel数据
data = pd.read_csv(excel_file_path)

def focus_cmd_window(window_title):
    # 获取CMD窗口的焦点
    window = gw.getWindowsWithTitle(window_title)
    if window:
        window[0].activate()
    else:
        print(f"No window found with title: {window_title}")

def pause_simulation():
    # 按下F键暂停仿真程序
    pyautogui.press('f')
    pyautogui.press('enter')
    time.sleep(1) # 确保弹出

def input_power_value(value):
    # 输入数值
    pyautogui.typewrite(value + '\n', interval=0.1)  # 输入值并回车确认
    

def restart_simulation():
    # 按下R键恢复仿真程序
    time.sleep(0.5) # 确保输入完成
    pyautogui.press('r')

def main():
    window_title = 'Windows PowerShell'  # 替换为实际CMD窗口的标题
    for index, row in data.iterrows():
        input_value = f"1={row['value']}"  # 读取csv文件中value列
        
        focus_cmd_window(window_title)  # 获取CMD窗口的焦点
        time.sleep(1)  # 稍等一秒钟，确保焦点切换完成
        pause_simulation()  # 暂停仿真程序
        input_power_value(input_value)  # 输入数值
        restart_simulation()  # 重新启动仿真程序
        print(f"已输入值: {input_value}")

        if index < len(data) - 1:  # 不是最后一行时，等待一小时
            # 获取当前时间
            NowTime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"此刻现实时间:{NowTime}, 预计一小时后将再次输入...")
            print(f"剩余运行次数：{len(data) - index - 1} , 预计剩余运行时间：{((len(data) - index - 1) * 1)//24}天{((len(data) - index - 1) * 1)%24}小时", )
            time.sleep(2)  # 每隔一小时执行一次
    
    print("运行结束！！！！！")

if __name__ == "__main__":
    output_log = 'output.log'
    with StdoutTee(output_log, mode="a", buff=1):
        main()
