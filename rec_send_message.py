'''
获取人脸识别结果
传给客户端
收到客户端的关闭消息之后关闭服务端
'''
import socket
import time
import dlib
import cv2
import torch
from PIL import Image
import os
from nets.facenet import Facenet as facenet
import numpy as np
import threading
import sys
def letterBox(old_img,size):
    #将传入的img转为rgb格式
    # print(type(old_img))
    image= Image.fromarray(cv2.cvtColor(old_img, cv2.COLOR_BGR2RGB))
    # image=old_img.convert("RGB")
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.Resampling.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    # if input_shape[-1]==1:
    #     new_image=new_image.convert("L")
    return new_image
def detect(img1):
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = facenet(backbone="mobilenet", mode="predict")
    model.load_state_dict(torch.load("model_data/facenet_mobilenet.pth", map_location=device), strict=False)
    net = model.eval()
    filePath='img'
    with torch.no_grad():
        all_l1s=[]
        fileList=os.listdir(filePath)
        img = letterBox(img1, [160, 160])
        for item in fileList:
            # print(item,'oooo')
            item_path=os.path.join(filePath,item)
            item_img=cv2.imread(item_path)
            img2=letterBox(item_img, [160, 160])
            photo_1 = torch.from_numpy(np.expand_dims(np.transpose(np.asarray(img).astype(np.float64) / 255, (2, 0, 1)), 0)).type(torch.FloatTensor)
            photo_2 = torch.from_numpy(np.expand_dims(np.transpose(np.asarray(img2).astype(np.float64) / 255, (2, 0, 1)), 0)).type(torch.FloatTensor)

            if 'cuda'==True:
                photo_1 = photo_1.cuda()
                photo_2 = photo_2.cuda()

            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            output1 = net(photo_1).cpu().numpy()
            output2 = net(photo_2).cpu().numpy()

            # ---------------------------------------------------#
            #   计算二者之间的距离
            # ---------------------------------------------------#
            l1 = np.linalg.norm(output1 - output2, axis=1)
            all_l1s.append(l1)
        min_l1=min(all_l1s)
        if min_l1<1:
            min_index = all_l1s.index(min_l1)
            min_img_name = fileList[min_index].split('.')[0]
        else:
            min_img_name='unknown'

    return min_img_name
def get_face(img,detector):
    # Dlib 预测器
    predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
    # Dlib 检测
    faces = detector(img, 1)
    person_name=''
    print('人脸数：', len(faces), '\n')
    if len(faces)!=0:
        for k, d in enumerate(faces):
            # 计算矩形大小
            # (x,y),(宽度width,高度height)
            pos_start = tuple([d.left(), d.top()])
            pos_end = tuple([d.right(), d.bottom()])

            # 计算矩形框大小
            height = d.bottom() - d.top()
            width = d.right() - d.left()

            # 根据人脸大小生成空的图象
            img_blank = np.zeros((height, width, 3), np.uint8)

            for i in range(height):
                for j in range(width):
                    img_blank[i][j] = img[d.top() + i][d.left() + j]
                # 存在本地
            person_name = detect(img_blank)
    return person_name
def draw_box(img,name,detector):
    face=detector(img)
    for k, d in enumerate(face):
        cv2.rectangle(img, tuple([d.left()+5, d.top()+5]), tuple([d.right(), d.bottom()]),
                      (255, 255, 255), 2)
        cv2.putText(img, name, tuple([d.left(), d.top() + 8]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                    2)
        # cv2.imshow('camera', img)
        #判断文件夹是否为空，为空时写入文件
        cv2.resize(img,(180,180))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        return img
def handle_client(client_socket):
    # video_path = 'test-video/liuyifei.mp4'
    cap = cv2.VideoCapture(0)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    det = dlib.get_frontal_face_detector()
    person_name = ''
    count = 0
    # 如果接收到命令为exit，则表示该用户退出，删除对应用户信息，关闭连接
    while True:
        # 循环发送消息，直到客户端关闭连接
        while cap.isOpened():
            # 读取每一帧图片
            ret, img = cap.read()
            if not ret:
                break
            if (count % fps == 0):
                # 搜寻相似度最高的人脸
                person_name = get_face(img, det)
                # 画框
                # face_img = draw_box(img, person_name, det)
                message = person_name
                if message:
                    client_socket.sendall(message.encode())
                time.sleep(2)
            count += 1
    cap.release()
    cv2.destroyAllWindows()

def get_time():
    """
    返回当前系统时间
    """
    return '[' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ']'
def send_message():
    # 创建TCP套接字
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 端口重用
    # 获取本地IP地址
    local_ip = socket.gethostbyname(socket.gethostname())

    # 设置端口号
    port = 12345

    # 绑定IP地址和端口号
    server_socket.bind((local_ip, port))
    client_sockets=[]
    # 开始监听
    server_socket.listen(10)
    print(get_time(), "系统：等待连接")
    while True:
        try:
            # 等待客户端连接
            client_socket, client_addr = server_socket.accept()
            # 将客户端套接字保存到列表中
            client_sockets.append(client_socket)
            print(f"客户端 {client_addr} 已连接")
        except KeyboardInterrupt:  # 按下ctrl+c会触发此异常
            server_socket.close()  # 关闭套接字
            sys.exit("\n" + get_time() + "系统：服务器安全退出！")  # 程序直接退出，不捕捉异常
        except Exception as e:
            print(e)
            client_sockets.remove(client_socket)
            continue
        thread = threading.Thread(target=handle_client(client_socket), args=(client_socket,))
        thread.start()

send_message()