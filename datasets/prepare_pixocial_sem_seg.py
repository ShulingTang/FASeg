import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import tqdm
from PIL import Image


def convert(input_path, output_path):
    """
    读取输入路径的图像文件，进行像素值的转换，并保存到输出路径。
    :param input_path: 输入图像的路径
    :param output_path: 输出图像的路径
    """
    try:
        img = np.asarray(Image.open(input_path))
        assert img.dtype == np.uint8, f"图像数据类型应为uint8, 但得到的是{img.dtype}"
        img = img - 1  # 0 (ignore) becomes 255. others are shifted by 1
        Image.fromarray(img).save(output_path)
    except Exception as e:
        print(f"处理文件 {input_path} 时出错: {e}")


def process_directory(input_dir, output_dir, max_workers=4):
    """
    处理指定目录中的所有图像文件，使用并行处理来加速。
    :param input_dir: 输入文件目录
    :param output_dir: 输出文件目录
    :param max_workers: 并行处理的最大工作进程数
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取输入目录中的所有文件
    files = list(input_dir.iterdir())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(convert, file, output_dir / file.name): file for file in files}
        for future in tqdm.tqdm(as_completed(future_to_file), total=len(files), desc=f"Processing {input_dir.name}"):
            file = future_to_file[future]
            try:
                future.result()
            except Exception as e:
                print(f"文件 {file} 处理失败: {e}")


if __name__ == "__main__":
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "pixocial_24"
    for name in ["train", "val"]:
        annotation_dir = dataset_dir / f"{name}_segmentations"
        output_dir = dataset_dir / f"{name}_segmentations_detectron2"
        process_directory(annotation_dir, output_dir, max_workers=4)
