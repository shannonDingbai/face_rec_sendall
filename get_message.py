import socket
import tkinter as tk
import threading
import tkinter.messagebox
def receive_message():
    last_message = ''
    while True:
        message = client_socket.recv(1024)
        if not message:
            break
        # 弹出提示框，显示收到的消息
        message_str = message.decode()
        print(last_message)

        if message_str != last_message and message_str != 'unknown':
            last_message = message_str
            print(message_str, '............')
            tkinter.messagebox.showinfo("提示", message_str + '来了')
#关闭弹框
def close_messagebox():
    root.destroy()
# 创建TCP套接字
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 连接到服务器
server_ip = '服务器端口号'
port = 12345
client_socket.connect((server_ip, port))

# 创建主窗口
root = tk.Tk()
root.withdraw()  # 隐藏主窗口


# 在主窗口中添加按钮，点击按钮可以手动关闭提示框
button = tk.Button(root, text="Click and Quit", command=close_messagebox)
button.pack()

# 开启一个新的线程来接收消息
receive_thread = threading.Thread(target=receive_message)
receive_thread.start()

# 进入消息循环
root.mainloop()

# 关闭客户端连接
client_socket.close()

