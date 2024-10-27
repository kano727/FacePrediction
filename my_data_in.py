import os
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom


def copy_images_and_create_xml(src_dirs, scores, dst_dir, xml_output_path):
    """
    遍历每个文件夹的图片，将文件名和评分记录到XML中，并将图片复制到目标文件夹。

    Args:
        src_dirs (list): 源文件夹的列表，每个文件夹包含一些图片。
        scores (list): 每个文件夹对应的评分列表，与src_dirs顺序对应。
        dst_dir (str): 目标文件夹路径，所有图片会复制到此文件夹。
        xml_output_path (str): 输出的 XML 文件路径。
    """
    # 创建目标文件夹
    os.makedirs(dst_dir, exist_ok=True)

    # 创建XML根元素
    root = ET.Element("average_scores")

    for src_dir, score in zip(src_dirs, scores):
        for filename in os.listdir(src_dir):
            if filename.endswith(('.jpg', '.png')):
                src_path = os.path.join(src_dir, filename)
                dst_path = os.path.join(dst_dir, filename)

                # 将图片复制到目标文件夹
                shutil.copy2(src_path, dst_path)

                # 添加XML元素
                image_element = ET.SubElement(root, "image", name=filename)
                ET.SubElement(image_element, "average_score").text = str(score)

    # 格式化并写入XML文件
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(xml_output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)


def delete_all_files_in_folder(folder_path):
    # 确保文件夹存在
    if os.path.exists(folder_path):
        # 遍历文件夹中的所有文件和子文件夹
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # 如果是文件，则删除
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
    else:
        print("The specified folder does not exist.")


# 使用示例
root_dirs = ['my_dataset/bad', 'my_dataset/middle', 'my_dataset/good']
labels = [0, 1, 2]
dst_dir = 'my_train_data'
xml_output_path = 'xml/my_average_scores.xml'

delete_all_files_in_folder(dst_dir)
os.remove('xml/my_average_scores.xml')
copy_images_and_create_xml(root_dirs, labels, dst_dir, xml_output_path)
