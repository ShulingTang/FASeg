import os
import cv2
import numpy as np
from tqdm import tqdm
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


def get_pixocial_dicts(img_dir, seg_dir):
    img_files = [image_name for image_name in os.listdir(img_dir) if image_name.endswith('.jpg')]
    img_files = sorted(img_files)
    seg_files = sorted(os.listdir(seg_dir))

    dataset_dicts = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_image, img_files[idx], seg_files[idx], img_dir, seg_dir, idx)
            for idx in range(len(img_files))
        ]
        for future in tqdm(futures):
            dataset_dicts.append(future.result())

    # 打乱dataset_dicts顺序
    np.random.shuffle(dataset_dicts)

    return dataset_dicts
