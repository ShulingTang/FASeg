import os
import cv2
import numpy as np
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from concurrent.futures import ThreadPoolExecutor


def process_image(img_file, seg_file, img_dir, seg_dir, idx):
    record = {}

    # 图片路径和文件名
    file_name = os.path.join(img_dir, img_file)
    record["file_name"] = file_name
    record["image_id"] = idx
    seg_file_path = os.path.join(seg_dir, seg_file)
    record["sem_seg_file_name"] = seg_file_path

    # 读取图片获取其尺寸
    img = cv2.imread(file_name)
    height, width = img.shape[:2]
    record["height"] = height
    record["width"] = width

    # 不需要生成边界框和annotations，删除这些部分
    return record
# def process_image(img_file, seg_file, img_dir, seg_dir, idx):
#     record = {}
#
#     # 图片路径和文件名
#     file_name = os.path.join(img_dir, img_file)
#     record["file_name"] = file_name
#     record["image_id"] = idx
#     seg_file_path = os.path.join(seg_dir, seg_file)
#     record["sem_seg_file_name"] = seg_file_path
#
#     # 读取图片获取其尺寸
#     img = cv2.imread(file_name)
#     height, width = img.shape[:2]
#     record["height"] = height
#     record["width"] = width
#
#     # 处理标签（分割图）
#     segmentation = cv2.imread(seg_file_path, cv2.IMREAD_GRAYSCALE)
#
#     objs = []
#     unique_classes = np.unique(segmentation)  # 使用np.unique获取所有类别
#     for class_id in unique_classes:
#         if class_id == 0:
#             continue  # 通常0代表背景类，跳过
#
#         # 找出该类别的所有区域，并生成边界框
#         mask = segmentation == class_id
#         contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for contour in contours:
#             bbox = cv2.boundingRect(contour)
#             obj = {
#                 "bbox": bbox,
#                 "bbox_mode": BoxMode.XYWH_ABS,
#                 "category_id": class_id,
#                 "segmentation": [contour.flatten().tolist()],  # 每个类别的分割轮廓
#                 "iscrowd": 0
#             }
#             objs.append(obj)
#
#     record["annotations"] = objs
#     return record


def get_pixocial_24_dicts(img_dir, seg_dir):
    img_files = sorted(os.listdir(img_dir))
    seg_files = sorted(os.listdir(seg_dir))

    dataset_dicts = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_image, img_files[idx], seg_files[idx], img_dir, seg_dir, idx)
            for idx in range(len(img_files))
        ]
        for future in tqdm(futures):
            dataset_dicts.append(future.result())

    return dataset_dicts



PIXOCIAL_24_CATEGORIES = [
    {"color": [0, 0, 0], "id": 0, "isthing": 0, "name": "Background"},
    {"color": [0, 255, 255], "id": 1, "isthing": 1, "name": "Headwear"},
    {"color": [0, 97, 255], "id": 2, "isthing": 1, "name": "Hair"},
    {"color": [42, 42, 78], "id": 3, "isthing": 1, "name": "Glove"},
    {"color": [255, 0, 0], "id": 4, "isthing": 1, "name": "Eyeglasses"},
    {"color": [140, 199, 0], "id": 5, "isthing": 1, "name": "Tops"},
    {"color": [50, 0, 220], "id": 6, "isthing": 1, "name": "Dress"},
    {"color": [34, 139, 34], "id": 7, "isthing": 1, "name": "Coat"},
    {"color": [255, 0, 255], "id": 8, "isthing": 1, "name": "Socks"},
    {"color": [240, 32, 160], "id": 9, "isthing": 1, "name": "Pants"},
    {"color": [255, 255, 0], "id": 10, "isthing": 1, "name": "Skin"},
    {"color": [0, 255, 0], "id": 11, "isthing": 1, "name": "Scarf"},
    {"color": [80, 127, 255], "id": 12, "isthing": 1, "name": "Skirt"},
    {"color": [193, 255, 193], "id": 13, "isthing": 1, "name": "Face"},
    {"color": [140, 54, 170], "id": 14, "isthing": 1, "name": "Left_Shoe"},
    {"color": [45, 82, 160], "id": 15, "isthing": 1, "name": "Bag"},
    {"color": [213, 179, 60], "id": 16, "isthing": 1, "name": "Accessories"},
    {"color": [185, 40, 235], "id": 17, "isthing": 1, "name": "Jumpsuits"},
    {"color": [88, 196, 239], "id": 18, "isthing": 1, "name": "Dummy_Column"},
    {"color": [13, 135, 136], "id": 19, "isthing": 1, "name": "Left_Hand"},
    {"color": [66, 1, 122], "id": 20, "isthing": 1, "name": "Right_Hand"},
    {"color": [208, 92, 123], "id": 21, "isthing": 1, "name": "Left_Leg"},
    {"color": [255, 122, 0], "id": 22, "isthing": 1, "name": "Right_Leg"},
    {"color": [0, 64, 128], "id": 23, "isthing": 1, "name": "Right_Shoe"}
]


PIXOCIAL_24_CLASSES = [c["name"] for c in PIXOCIAL_24_CATEGORIES]
PIXOCIAL_24_COLORS = [c["color"] for c in PIXOCIAL_24_CATEGORIES]

_root = os.getenv("DETECTRON2_DATASETS", "datasets")

# 注册训练集和验证集
for d in ["train", "val"]:
    img_root = f"{_root}/pixocial_24/{d}_images"
    gt_root = f"{_root}/pixocial_24/{d}_segmentations"
    DatasetCatalog.register(
      f"pixocial_24_{d}",
      lambda img_dir=img_root, seg_dir=gt_root: get_pixocial_24_dicts(img_dir, seg_dir)
    )
    MetadataCatalog.get(f"pixocial_24_{d}").set(stuff_classes=PIXOCIAL_24_CLASSES)
    MetadataCatalog.get(f"pixocial_24_{d}").set(stuff_colors=PIXOCIAL_24_COLORS)
    MetadataCatalog.get(f"pixocial_24_{d}").set(evaluator_type="sem_seg")
    MetadataCatalog.get(f"pixocial_24_{d}").set(ignore_label=255)
    MetadataCatalog.get(f"pixocial_24_{d}").set(image_root=img_root)
    MetadataCatalog.get(f"pixocial_24_{d}").set(sem_seg_root=gt_root)

