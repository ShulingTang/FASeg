import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from utils import get_pixocial_dicts


_root = os.getenv("DETECTRON2_DATASETS", "datasets")

PIXOCIAL_LINING_CATEGORIES = [
    {"color": [0, 0, 0], "id": 0, "isthing": 0, "name": "background"},
    {"color": [0, 255, 255], "id": 1, "isthing": 1, "name": "tops"},
    {"color": [0, 97, 255], "id": 2, "isthing": 1, "name": "pants"},
    {"color": [42, 42, 78], "id": 3, "isthing": 1, "name": "overalls"},
    {"color": [255, 0, 0], "id": 4, "isthing": 1, "name": "tops_lining"},
    {"color": [140, 199, 0], "id": 5, "isthing": 1, "name": "pants_lining"},
    {"color": [50, 0, 220], "id": 6, "isthing": 1, "name": "overalls_lining"},
]


PIXOCIAL_LINING_CLASSES = [c["name"] for c in PIXOCIAL_LINING_CATEGORIES]
PIXOCIAL_LINING_COLORS = [c["color"] for c in PIXOCIAL_LINING_CATEGORIES]

for d in ["train", "val"]:
    img_root = f"{_root}/pixocial_lining/{d}_images"
    gt_root = f"{_root}/pixocial_lining/{d}_segmentations"
    DatasetCatalog.register(
      f"pixocial_24_{d}",
      lambda img_dir=img_root, seg_dir=gt_root: get_pixocial_dicts(img_dir, seg_dir)
    )
    MetadataCatalog.get(f"pixocial_24_{d}").set(stuff_classes=PIXOCIAL_LINING_CLASSES)
    MetadataCatalog.get(f"pixocial_24_{d}").set(stuff_colors=PIXOCIAL_LINING_COLORS)
    MetadataCatalog.get(f"pixocial_24_{d}").set(evaluator_type="sem_seg")
    MetadataCatalog.get(f"pixocial_24_{d}").set(ignore_label=255)
    MetadataCatalog.get(f"pixocial_24_{d}").set(image_root=img_root)
    MetadataCatalog.get(f"pixocial_24_{d}").set(sem_seg_root=gt_root)
