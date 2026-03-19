# ------------------------------------------------------------
# 1. 环境准备
# ------------------------------------------------------------
# pip install matplotlib seaborn scikit-learn numpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn.cluster import KMeans

# ------------------------------------------------------------
# 2. 生成示例数据
# ------------------------------------------------------------
H, W = 128, 128          # 图像尺寸
N = 1024                 # Patch 数量
PATCH_SIZE = 4

# 随机 Patch 坐标 (x, y)
xy = np.column_stack((
    np.random.randint(0, W-PATCH_SIZE, N),
    np.random.randint(0, H-PATCH_SIZE, N)
))

# 随机 8-D 特征（例如 ViT 的 patch token）
features = np.random.randn(N, 8)

# 预计算的异常图（值越大越异常）
anomaly_map = np.random.rand(H, W) ** 3   # 人为制造高异常区

# ------------------------------------------------------------
# 3. KMeans++ 聚类
# ------------------------------------------------------------
K = 3
kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
labels = kmeans.fit_predict(features)
centroids = kmeans.cluster_centers_   # 仅用于可视化

# ------------------------------------------------------------
# 4. 计算每个聚类的异常分数
# ------------------------------------------------------------
scores = []
for k in range(K):
    mask_k = (labels == k)
    # 取该区域对应像素的异常均值
    coords = xy[mask_k]
    region_vals = [anomaly_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE].mean()
                   for x, y in coords]
    scores.append(np.mean(region_vals))
scores = np.array(scores)

# ------------------------------------------------------------
# 5. 绘图
# ------------------------------------------------------------
sns.set_style('white')
fig = plt.figure(figsize=(12, 4))

# 5.1 原图 + 聚类区域
ax1 = fig.add_subplot(1, 3, 1)
img_vis = np.zeros((H, W, 3))
palette = np.array(sns.color_palette('tab10'))
for k in range(K):
    mask_k = (labels == k)
    for x, y in xy[mask_k]:
        img_vis[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = palette[k]
ax1.imshow(img_vis, extent=[0, W, 0, H], origin='lower')
ax1.set_title('Clustered Regions')

# 5.2 异常图热力图
ax2 = fig.add_subplot(1, 3, 2)
sns.heatmap(anomaly_map, cmap='Reds', cbar=True, ax=ax2)
ax2.set_title('Anomaly Map')

# 5.3 异常分数条形图 + 高亮
ax3 = fig.add_subplot(1, 3, 3)
bars = ax3.bar(range(K), scores, color=palette[:K])
best = np.argmax(scores)
bars[best].set_edgecolor('red')
bars[best].set_linewidth(3)
ax3.set_xticks(range(K))
ax3.set_xticklabels([f'R{k}' for k in range(K)])
ax3.set_title('Region Anomaly Scores')
ax3.set_ylabel('Mean Anomaly')

# 5.4 在聚类图上用红框标出最高分区域
best_mask = (labels == best)
best_coords = xy[best_mask]
x_min, y_min = best_coords.min(axis=0)
x_max, y_max = best_coords.max(axis=0)
rect = Rectangle((x_min, y_min), x_max-x_min+PATCH_SIZE, y_max-y_min+PATCH_SIZE,
                 linewidth=2, edgecolor='red', facecolor='none')
ax1.add_patch(rect)
ax1.text(x_min, y_min-5, 'Best', color='red', fontsize=10, weight='bold')

# ------------------------------------------------------------
# 6. 保存
# ------------------------------------------------------------
plt.tight_layout()
plt.savefig('region_anomaly_pipeline.png', dpi=300, bbox_inches='tight')
plt.savefig('region_anomaly_pipeline.svg')
plt.show()