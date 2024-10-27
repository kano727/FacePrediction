import logging
import os
import random
import shutil
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

"""
遍历val_data文件夹，展示图片并通过按键剪切到my_dataset文件夹，实现手动标签功能
"""

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def center_window(window, width, height):
    # 获取屏幕宽度和高度
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    # 计算位置
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)

    # 设置窗口大小和位置
    window.geometry(f"{width}x{height}+{x}+{y}")


def show_image(image_path):
    # 创建一个窗口
    window = tk.Tk()
    window.title("Image Viewer")

    # 设定窗口大小
    window_width = 600
    window_height = 500
    center_window(window, window_width, window_height)

    # 打开图片
    img = Image.open(image_path)

    # 将图片缩放到窗口大小
    img = img.resize((window_width, window_height - 40), Image.LANCZOS)  # 预留空间给文件名

    # 将PIL图像转换为Tkinter图像
    img_tk = ImageTk.PhotoImage(img)

    # 创建一个标签显示图片
    label = tk.Label(window, image=img_tk)
    label.pack()

    remaining_count = len(os.listdir(folder_path))

    # 获取文件名并显示
    file_name = os.path.basename(image_path)
    name_label = tk.Label(window, text=file_name + f"  Remaining images: {remaining_count}", font=("Arial", 14))
    name_label.pack()


    # 等待用户输入按键
    window.bind("<Key>", lambda e: handle_key(e, image_path, window))
    window.focus_force()  # 聚焦到窗口
    window.mainloop()

def handle_key(event, image_path, window):
    if event.keysym == "Return":  # 回车键
        logging.info(f"Deleted: {image_path}")
        os.remove(image_path)
        window.destroy()
    elif event.keysym == "1":
        dest_path = 'my_dataset/bad'  # 替换为你的目标文件夹路径
        shutil.copy(image_path, dest_path)
        logging.info(f"Copied to {dest_path}: {image_path}")
        os.remove(image_path)
        logging.info(f"Deleted: {image_path}")
        window.destroy()
    elif event.keysym == "2":
        dest_path = 'my_dataset/middle'  # 替换为你的目标文件夹路径
        shutil.copy(image_path, dest_path)
        logging.info(f"Copied to {dest_path}: {image_path}")
        os.remove(image_path)
        logging.info(f"Deleted: {image_path}")
        window.destroy()
    elif event.keysym == "3":
        dest_path = 'my_dataset/good'  # 替换为你的目标文件夹路径
        shutil.copy(image_path, dest_path)
        logging.info(f"Copied to {dest_path}: {image_path}")
        os.remove(image_path)
        logging.info(f"Deleted: {image_path}")
        window.destroy()
    elif event.keysym == "space":  # 空格键
        window.quit()  # 退出窗口
        os._exit(0)    # 终止程序

def main(folder_path):
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    random.shuffle(images)  # 随机打乱图片顺序
    for filename in images:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            show_image(image_path)

if __name__ == "__main__":
    folder_path = 'val_data/0'
    main(folder_path)
