import xml.etree.ElementTree as ET
import os
import math
# 类别映射（根据你的数据集修改）
class_mapping = {
    'car': 0,
    # 'person': 1,
    # 其他类别...
}

def rotate_point(cx, cy, x, y, angle):
    """绕中心点旋转一个点"""
    nx = (x - cx) * math.cos(angle) - (y - cy) * math.sin(angle) + cx
    ny = (x - cx) * math.sin(angle) + (y - cy) * math.cos(angle) + cy
    return nx, ny

def obb_to_points(cx, cy, w, h, angle_rad):
    """
    将旋转框转换为四个顶点坐标（按顺序：左上、右上、右下、左下）
    :return: [x1,y1,x2,y2,x3,y3,x4,y4]
    """
    # 未旋转的四个角点（相对于中心点）
    half_w = w / 2
    half_h = h / 2

    corners = [
        (-half_w, -half_h),  # 左上
        (half_w, -half_h),   # 右上
        (half_w, half_h),    # 右下
        (-half_w, half_h)    # 左下
    ]

    # 应用旋转
    rotated = []
    for dx, dy in corners:
        x = cx + dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
        y = cy + dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
        rotated.append((x, y))

    return [
        rotated[0][0], rotated[0][1],
        rotated[1][0], rotated[1][1],
        rotated[2][0], rotated[2][1],
        rotated[3][0], rotated[3][1]
    ]

def convert_xml_to_txt(xml_file, output_folder, img_folder):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text
    img_path = os.path.join(img_folder, filename)

    if not os.path.exists(img_path):
        print(f"图片不存在: {img_path}")
        return

    size = root.find('size')
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)

    label_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.txt')

    with open(label_path, 'w') as f:
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if cls_name not in class_mapping:
                continue  # 忽略未知类别
            cls_id = class_mapping[cls_name]

            robndbox = obj.find('robndbox')
            cx = float(robndbox.find('cx').text)
            cy = float(robndbox.find('cy').text)
            w = float(robndbox.find('w').text)
            h = float(robndbox.find('h').text)
            angle_rad = float(robndbox.find('angle').text)  # 已知是 radians

            # 转换为四点坐标，并归一化
            points = obb_to_points(cx, cy, w, h, angle_rad)
            normalized_points = [p / img_w if i % 2 == 0 else p / img_h for i, p in enumerate(points)]

            # 写入 YOLO OBB 四点坐标格式
            line = f"{cls_id} " + " ".join([f"{coord:.6f}" for coord in normalized_points])
            f.write(line + "\n")

    print(f"已生成标签文件: {label_path}")
# 示例用法
if __name__ == '__main__':
    xml_folder = 'UAV/UAV_ROD_Data/test/annotations'     # XML文件所在目录
    image_folder = 'yoloobbcarUAV/images/val'        # 图片文件所在目录
    output_folder = 'yoloobbcarUAV/labels/val'   # 输出txt文件目录

    os.makedirs(output_folder, exist_ok=True)

    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            full_xml_path = os.path.join(xml_folder, xml_file)
            convert_xml_to_txt(full_xml_path, output_folder, image_folder)