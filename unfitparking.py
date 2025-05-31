import cv2
import numpy as np
import json
from PIL import Image, ImageDraw
from ultralytics import YOLO
from shapely.geometry import Polygon
def parse_shapes_from_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = []
    id_counter = 1

    for shape in data['shapes']:
        if shape['label'] == 'parkingbox':
            points = shape['points']
            
            # 转换为整数坐标的元组列表
            polygon = [(x, y) for x, y in points]

        if shape['shape_type'] == 'rectangle' and len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            # 构造四角点
            polygon = [
                (x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2)
            ]

        # 生成ID：A1, B1,... Z1, AA1,...
        column = (id_counter - 1) // 1 + 1
        def num_to_col_letters(n):
            letters = ''
            while n > 0:
                n, rem = divmod(n - 1, 26)
                letters = chr(65 + rem) + letters
            return letters
        area_id = f"{num_to_col_letters(id_counter)}1"

        result.append({
            "id": area_id,
            "polygon": polygon
        })

        id_counter += 1

    return result
def is_point_in_polygon(point, polygon):
    """射线法判断点是否在多边形内"""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside



def calculate_outside_area(car_bbox, slot_poly):
    """
    计算车框超出停车框的面积。
    
    参数:
        car_bbox: list 或 tuple，表示车框的四个角点 [(x1,y1), (x2,y2), ..., (x4,y4)]
        slot_poly: list，表示停车位的多边形顶点 [(x1,y1), (x2,y2), ..., (xn,yn)]

    返回:
        outside_area: float，超出停车框的面积
        inside_area: float，位于停车框内的面积
        ratio_inside: float，在框内的占比
    """
    # 创建车框和停车框的多边形对象
    car_polygon = Polygon(car_bbox)
    parking_slot_polygon = Polygon(slot_poly)

    if not car_polygon.is_valid or not parking_slot_polygon.is_valid:
        raise ValueError("输入的坐标不是有效的多边形")

    # 求交集（车在停车位内的区域）
    intersection = car_polygon.intersection(parking_slot_polygon)
    inside_area = intersection.area if not intersection.is_empty else 0.0
    print('parking_slot_polygon.area',parking_slot_polygon.area)
    # 总车框面积
    total_car_area = car_polygon.area

    # 超出部分面积 = 总面积 - 交集面积
    outside_area = total_car_area - inside_area

    # 可选：计算车在停车框内的比例
    ratio_inside = inside_area / total_car_area if total_car_area > 0 else 0.0

    return outside_area, inside_area, ratio_inside

def calculate_iou(bbox, polygon):
    """计算旋转框与多边形的IoU（简化版：用外接矩形近似）"""
    rect_bbox = cv2.boundingRect(np.array(bbox, dtype=np.int32))
    rect_poly = cv2.boundingRect(np.array(polygon, dtype=np.int32))
    x1, y1, w1, h1 = rect_bbox
    x2, y2, w2, h2 = rect_poly
    # 计算交集和并集
    inter_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def check_parking_violations(cars, parking_slots):
    """检测未正确停放的车辆"""
    violations = []
    matched_slots = set()  # 已匹配的停车位
    
    for car in cars:
        car_id = car["id"]
        car_bbox = car["bbox"]
        best_match = None
        best_iou = 0
        match_slot= None
        for slot in parking_slots:
            slot_id = slot["id"]
            slot_poly = slot["polygon"]
            
            # 检查车辆是否在停车位内
            all_inside = all(is_point_in_polygon(p, slot_poly) for p in car_bbox)
            iou = calculate_iou(car_bbox, slot_poly)
            
            if all_inside or iou > 0.5:  # 匹配成功
                if iou > best_iou:
                    best_iou = iou
                    best_match = slot_id
                    match_slot = slot_poly
        if best_match is None:
            violations.append({"car_id": car_id, "reason": "未停在车位内"})
        else:
            matched_slots.add(best_match)
            # 进一步检查方向或压线（需补充具体逻辑）
            print(car_bbox, match_slot)
            outside_area, inside_area, ratio_inside = calculate_outside_area(car_bbox, match_slot)
            if ratio_inside < 0.9:
                violations.append({"car_id": car_id, "slot_id": best_match, "reason": "斜停或压线"})
            print(outside_area, inside_area, ratio_inside)
    # 检查未被占用的停车位（可选）
    unused_slots = [slot["id"] for slot in parking_slots if slot["id"] not in matched_slots]
    
    return violations, unused_slots
def transform_normalized_obb(obb_tensor, M):
    """
    输入：
        obb_tensor: shape (N, 4, 2)，归一化坐标 (0~1)
        M: 单应性矩阵 (3x3)
    输出：
        transformed_obbs: 变换后的 OBB 列表，格式同输入
    """
    # Step 1: 移动到 CPU 并转为 numpy
    obb_np = obb_tensor.detach().cpu().numpy()

    # Step 3: 对每个 OBB 执行 perspectiveTransform
    transformed_obbs = []
    for obb in obb_np:
        pts = obb.reshape(-1, 1, 2).astype(np.float32)  # shape (4, 1, 2)
        #print(pts)
        transformed_pts = cv2.perspectiveTransform(pts, M)  # 应用变换
        #print(transformed_pts)
        transformed_obbs.append(transformed_pts.reshape(4, 2))

    # 返回 NumPy 格式的结果（可选）
    return transformed_obbs

#读取JSON文件并解析停车位
refer_img='parkingdata/lake/20250528_193112_0138_W.jpg'
parking_slots = parse_shapes_from_json("parkingdata/lake/20250528_193112_0138_W.json")
#print(parking_slots)

#yolo

cars = []
car_id = 1  # 用于给每个检测到的 car 分配唯一 ID
model_y = YOLO('best.pt')#模型路径
img_name='parkingdata/lake/20250528_193153_0140_W.jpg'


#图像对齐
img1 = cv2.imread(img_name)  # 待对齐图像
img2 = cv2.imread(refer_img)  # 参考图像

# 使用 ORB 代替 SIFT
# orb = cv2.ORB_create(nfeatures=2000)  # 可以控制最大特征点数
# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
# 特征匹配（FLANN或BFMatcher）
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)  # 按距离排序
#仅保留距离最近的前50%匹配点
matches = matches[:int(len(matches)*0.8)]
# 提取匹配点坐标
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
# 计算相似变换矩阵（缩放 + 旋转 + 平移）
M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
aligned_img = cv2.warpAffine(img1, M, (img2.shape[1], img2.shape[0]))
cv2.imwrite("aligned_image.jpg", aligned_img)  # 保存对齐后的图像
M_homogeneous = np.eye(3)
M_homogeneous[:2, :] = M
print(M_homogeneous)
# 如果使用单应性矩阵（透视变换，适用于非平面场景）
# M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

results = model_y.predict(img_name, 
                save = False, 
                name = 'test_vis',
                device= 0,
                conf=0.7
                )
for result in results:
    obb = result.obb
    #print(obb.xyxyxyxy)
    bbox_points=transform_normalized_obb(obb.xyxyxyxy, M_homogeneous)
    #print(bbox_points)
    for box in bbox_points:
        coords = box.tolist()  # 转为 Python 列表
        #print(coords)
        # coords[0] 是四个角点的坐标 [[x1,y1], [x2,y2], ..., [x4,y4]]
        bbox_points = coords
        # 可选：四舍五入为整数像素坐标
        bbox_points = [(round(x), round(y)) for x, y in bbox_points]
        # 添加到结果中
        
        cars.append({
            "id": car_id,
            "bbox": bbox_points
        })
        car_id += 1
#print(cars)

#画停车框
original_image_size = (5472, 3078)  # (宽, 高)

img = cv2.imread("aligned_image.jpg")
height, width = img.shape[:2]
# 计算缩放比例
scale_w = width / original_image_size[0]
scale_h = height / original_image_size[1]
# 如果要保持纵横比一致，选择最小的缩放比例作为统一缩放因子


img = Image.open("aligned_image.jpg")
# 创建一个新的空白图像
#target_image = Image.new('RGB', target_image_size, color='white')
draw = ImageDraw.Draw(img)

for box in parking_slots:
    scaled_polygon = [(point[0]*scale_w, point[1]*scale_h) for point in box['polygon']]
    draw.polygon(scaled_polygon, outline="red", fill=None,width=3)  # 绘制多边形边框

# 保存或显示图像
img.save('withbox.jpg')  # 显示图像
# target_image.save('path/to/save/image.jpg')  # 或者保存到文件

# 示例数据
# cars = [
#     {"id": 1, "bbox": [(100, 100), (150, 100), (150, 150), (100, 150)]},
#     {"id": 2, "bbox": [(200, 200), (250, 200), (250, 250), (200, 250)]},  # 未匹配
# ]
# parking_slots = [
#     {"id": "A1", "polygon": [(80, 80), (180, 80), (180, 180), (80, 180)]},
#     {"id": "B1", "polygon": [(190, 190), (290, 190), (290, 290), (190, 290)]},
# ]

violations, unused_slots = check_parking_violations(cars, parking_slots)

#print("违规车辆:", violations)
#print("空闲车位:", unused_slots)
img = Image.open("aligned_image.jpg")
draw = ImageDraw.Draw(img)
for violation in violations:
    #print(f"车辆 {violation['car_id']} 违规原因: {violation['reason']}")
    if violation['reason']=='斜停或压线':
        draw.polygon(cars[violation['car_id']-1]['bbox'], outline="red", fill=None,width=3)
img.save('result.jpg')