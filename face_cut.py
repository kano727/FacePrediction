import cv2
import os
import numpy as np

# 加载 DNN 模型
model_file = 'res10_300x300_ssd_iter_140000.caffemodel'
config_file = 'deploy.prototxt'
net = cv2.dnn.readNetFromCaffe(config_file, model_file)

# 指定要处理的文件夹路径
input_folder = 'val_data/origin'  # 替换为你的文件夹路径
output_folder = 'val_data'

# 创建保存文件夹
os.makedirs(output_folder, exist_ok=True)

# 递归遍历文件夹
for root, dirs, files in os.walk(input_folder):
    for file in files:
        # 检查文件扩展名
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path)

            # 检查图像是否成功读取
            if image is None:
                print(f"Warning: Could not read image {image_path}. Skipping.")
                continue

            # 获取图像的高度和宽度
            h, w = image.shape[:2]

            # 创建输入blob并进行人脸检测
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            # 提取人脸并保存
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # 设定置信度阈值
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    face_img = image[startY:endY, startX:endX]
                    output_path = os.path.join(output_folder, f'{os.path.splitext(file)[0]}_face_{i}.jpg')
                    cv2.imwrite(output_path, face_img)

print(f'Detected and extracted faces from images in {input_folder}. Extracted faces saved to {output_folder}.')
