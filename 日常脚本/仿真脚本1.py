import time
import pyautogui
import pygetwindow as gw
import pandas as pd
from tee import StdoutTee

# 定义Excel文件路径
excel_file_path = './value.csv'
# 读取Excel数据
data = pd.read_csv(excel_file_path)

pyautogui.FAILSAFE = False # 关掉鼠标检测，防止锁屏丢失鼠标报错

window_title = 'Windows PowerShell'  # 替换为实际CMD窗口的标题


def focus_cmd_window(window_title):
    # 获取CMD窗口的焦点
    window = gw.getWindowsWithTitle(window_title)
    if window:
        try:
            if not window[0].isActive:
                window[0].activate()
            return True
            # 只有当指定窗口被激活时，才会执行下一步操作
        except gw.PyGetWindowException as e:
            # 有异常时，打印，然后进入循环，直到执行try块中的代码
            print(f"Error activating window: {e}")
            return False
    else:
        print(f"No window found with title: {window_title}")
        return False

def ensure_cmd_window(window_title):
    while not focus_cmd_window(window_title):
        print("Waiting for the specified window to become active...")
        time.sleep(1)  # Wait for 1 second before checking again

def pause_simulation():
    # 按下F键暂停仿真程序
    ensure_cmd_window(window_title)  # 确保CMD窗口是激活的
    pyautogui.press('f')
    ensure_cmd_window(window_title)  # 确保CMD窗口是激活的
    pyautogui.press('enter')
    time.sleep(1)  # 确保弹出

def input_power_value(value):
    # 每个字符之间都检测，这样基本会强制切换回来，就不会输入到别的地方
    for char in value + '\n':
        ensure_cmd_window(window_title)  # 确保CMD窗口是激活的
        pyautogui.typewrite(char)
        time.sleep(0.1)  # Small delay to simulate typing

def restart_simulation():
    # 按下R键恢复仿真程序
    time.sleep(0.5)  # 确保输入完成
    ensure_cmd_window(window_title)  # 确保CMD窗口是激活的
    pyautogui.press('r')
def main():
    
    for index, row in data.iterrows():
        input_value = f"1={row['value']}"  # 读取csv文件中value列
        
        ensure_cmd_window(window_title)  # 确保CMD窗口是激活的
        pause_simulation()  # 暂停仿真程序
        input_power_value(input_value)  # 输入数值
        restart_simulation()  # 重新启动仿真程序
        print(f"已输入值: {input_value}")

        if index < len(data) - 1:  # 不是最后一行时，等待一小时
            # 获取当前时间
            NowTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"此刻现实时间:{NowTime}, 预计一小时后将再次输入...")
            print(f"剩余运行次数：{len(data) - index - 1} , 预计剩余运行时间：{((len(data) - index - 1) * 1)//24}天{((len(data) - index - 1) * 1)%24}小时", )
            time.sleep(3600)  # 每隔一小时执行一次
    
    print("运行结束！！！！！")

if __name__ == "__main__":
    output_log = 'output.log'
    with StdoutTee(output_log, mode="a", buff=1):
        NowTime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"此刻现实时间:{NowTime}, 已输入第一个0.81，3600秒后输入第二个值：0.78...")
        time.sleep(3600)
        main()
