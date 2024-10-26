import pandas as pd
import os
import shutil
import xml.etree.ElementTree as ET


# 读取所有工作表
def read_xlsx_sheets(file_path):
    return pd.read_excel(file_path, sheet_name=0)

# 创建新的 XML 文件
def create_xml_file(xml_path):
    root = ET.Element("average_scores")
    tree = ET.ElementTree(root)
    tree.write(xml_path, encoding='utf-8', xml_declaration=True)

# 更新 XML 文件
def update_xml_file(xml_path, scores):
    if not os.path.exists(xml_path):
        create_xml_file(xml_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for file_name, score in scores.items():
        # image_element = root.find(f"./image[@name='{file_name}']")
        # if image_element is not None:
        #     # 更新已有评分
        #     score_element = image_element.find('average_score')
        #     if score_element is not None:
        #         score_element.text = str(score)
        # else:
            # 创建新的 image 元素并添加评分
            new_image_element = ET.SubElement(root, "image", name=file_name)
            new_score_element = ET.SubElement(new_image_element, "average_score")
            new_score_element.text = str(score)

    tree.write(xml_path, encoding='utf-8', xml_declaration=True)


def main():
    xlsx_file = 'SCUT-FBP5500_with_Landmarks/All_Ratings.xlsx'  # 替换为你的xlsx文件路径
    pt_folder = 'SCUT-FBP5500_with_Landmarks/Images'
    output_folder = 'train_data'
    xml_path = 'average_scores.xml'

    # 读取所有工作表
    df = read_xlsx_sheets(xlsx_file)

    scores = {}

    # 从所有工作表中提取评分
    for index, row in df.iterrows():
        file_name = row[1]  # 第二列
        score = float(row[2])  # 第三列

        if isinstance(file_name, str) and (file_name.startswith('ftw') or file_name.startswith('fty')):
            pt_file_path = os.path.join(pt_folder, file_name)
            if os.path.exists(pt_file_path):
                shutil.copy(pt_file_path, os.path.join(output_folder, file_name))
                if file_name not in scores:
                    scores[file_name] = score
                else:
                    scores[file_name] = (score + scores[file_name])/2

    scores = {key: round(value) - 1 for key, value in scores.items()}

    # 更新 XML 文件
    update_xml_file(xml_path, scores)


if __name__ == "__main__":
    main()
