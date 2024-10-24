import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from mask2former.data.datasets.utils import get_pixocial_dicts


_root = os.getenv("DETECTRON2_DATASETS", "datasets")

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


# 注册训练集和验证集
for d in ["train", "val"]:
    img_root = f"{_root}/pixocial_24/{d}_images"
    gt_root = f"{_root}/pixocial_24/{d}_segmentations"
    DatasetCatalog.register(
      f"pixocial_24_{d}",
      lambda img_dir=img_root, seg_dir=gt_root: get_pixocial_dicts(img_dir, seg_dir)
    )
    MetadataCatalog.get(f"pixocial_24_{d}").set(stuff_classes=PIXOCIAL_24_CLASSES)
    MetadataCatalog.get(f"pixocial_24_{d}").set(stuff_colors=PIXOCIAL_24_COLORS)
    MetadataCatalog.get(f"pixocial_24_{d}").set(evaluator_type="sem_seg")
    MetadataCatalog.get(f"pixocial_24_{d}").set(ignore_label=255)
    MetadataCatalog.get(f"pixocial_24_{d}").set(image_root=img_root)
    MetadataCatalog.get(f"pixocial_24_{d}").set(sem_seg_root=gt_root)

