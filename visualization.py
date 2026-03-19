import cv2
import os

from PIL import Image

from utils import normalize
import numpy as np

def visualizer(pathes, anomaly_map, img_size, save_path, cls_name):
    for idx, path in enumerate(pathes):
        cls = path.split('/')[-2]
        filename = path.split('/')[-1]
        vis = cv2.cvtColor(cv2.resize(cv2.imread(path), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
        mask = normalize(anomaly_map[idx])
        vis = apply_ad_scoremap(vis, mask)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
        save_vis = os.path.join(save_path, 'imgs', cls_name[idx], cls)
        if not os.path.exists(save_vis):
            os.makedirs(save_vis)
        cv2.imwrite(os.path.join(save_vis, filename), vis)

def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

if __name__ == '__main__':
    # 图片目录和命名规则
    image_dir = './data/mvtec/hazelnut/test/hole/'
    prefix = 'anomaly_map_'
    extension = '.png'
    count = 8

    # 读取所有图片
    images = []
    for i in range(count):
        filename = f"{prefix}{i:03d}{extension}"
        filepath = os.path.join(image_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"图片不存在: {filepath}")
        img = Image.open(filepath)
        images.append(img)

    # 检查是否有图片
    if not images:
        raise ValueError("没有找到任何图片进行拼接")

    # 获取每张图片的尺寸
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)

    # 创建新图像（横向拼接）
    new_img = Image.new('RGB', (total_width, max_height))

    # 拼接图片
    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width

    # 保存结果
    output_path = os.path.join(image_dir, 'a_anomaly_all_no_hsf_hole.jpg')
    new_img.save(output_path)
    print(f"拼接完成，保存为: {output_path}")