# 视觉 Prompt 与自适应双门控融合

## 创新点描述

本文在 AnomalyCLIP 的对象无关文本提示学习框架上，引入一种视觉 Prompt 增强与自适应双门控融合机制。首先，利用图像全局视觉特征与文本特征之间的跨模态注意力，构建文本引导的视觉 Prompt，从而获得更贴近异常语义的增强视觉表示。进一步地，考虑到不同样本对文本先验和局部异常聚类信息的依赖程度不同，本文设计了双门控自适应融合模块，分别对视觉 Prompt 分支和 HSF 聚类分支进行动态调节，使模型能够按样本自适应选择更合适的特征组合方式。

该方法具有两个优点。其一，视觉 Prompt 使全局图像特征获得来自文本异常语义的细粒度引导，从而提升图像级异常判别能力。其二，自适应双门控融合避免了固定加权策略对所有样本一视同仁的问题，使模型能够在“文本语义引导”和“局部异常区域聚合”之间自动平衡，有助于提升模型的鲁棒性与泛化能力。

## 方法公式

### 1. 视觉 Prompt 生成

给定视觉编码器输出的全局图像特征 $V \in \mathbb{R}^{D}$，以及文本编码器输出的文本特征集合 $T \in \mathbb{R}^{K \times D}$，采用跨模态注意力生成视觉 Prompt：

$$
\tilde{V} = \operatorname{softmax}\left(\frac{V T^{\top}}{\sqrt{D}}\right) \cdot T
$$

随后将原始视觉特征与视觉 Prompt 进行拼接，并通过 MLP 得到增强视觉表示：

$$
V' = \operatorname{MLP}(V \oplus \tilde{V})
$$

其中，$\oplus$ 表示特征拼接操作。

### 2. 第一阶段门控融合

为了自适应控制视觉 Prompt 对原始特征的影响强度，设计第一阶段门控：

$$
G_v = \sigma(\operatorname{MLP}(V \oplus V'))
$$

$$
V_g = G_v \odot V' + (1 - G_v) \odot V
$$

其中，$\sigma(\cdot)$ 表示 Sigmoid 函数，$\odot$ 表示逐元素乘法，$V_g$ 为门控后的视觉特征。

### 3. 第二阶段门控融合

对于 HSF 模块从高异常区域聚合得到的聚类特征 $F_c$，进一步设计第二阶段门控：

$$
G_c = \sigma(\operatorname{MLP}(V_g \oplus F_c))
$$

$$
V_{\text{final}} = G_c \odot F_c + (1 - G_c) \odot V_g
$$

其中，$V_{\text{final}}$ 为最终用于异常分类的图像表示。

### 4. 总体损失函数

训练时，总损失由图像级分类损失、像素级分割损失、知识蒸馏损失、视觉 Prompt 对齐损失以及门控正则项共同组成：

$$
\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{seg} + \mathcal{L}_{kd} + \lambda_1 \mathcal{L}_{vp} + \lambda_2 \mathcal{L}_{gate}
$$

其中，$\mathcal{L}_{vp}$ 用于约束视觉 Prompt 分支的表示稳定性，$\mathcal{L}_{gate}$ 用于抑制门控值过早塌缩到 0 或 1。

## 模块图说明

整体流程如下：

1. 输入图像经过 AnomalyCLIP 视觉编码器得到全局视觉特征 $V$ 和多层 patch 特征。
2. 文本 Prompt 经文本编码器得到正常/异常文本特征 $T$。
3. 通过跨模态注意力计算视觉 Prompt $\tilde{V}$，再经 MLP 得到增强视觉特征 $V'$。
4. 第一阶段门控对 $V$ 与 $V'$ 进行自适应融合，得到 $V_g$。
5. patch 特征与文本特征生成 anomaly map，并通过 HSF 聚合得到聚类特征 $F_c$。
6. 第二阶段门控对 $V_g$ 与 $F_c$ 进行自适应融合，得到最终图像特征 $V_{\text{final}}$。
7. 使用 $V_{\text{final}}$ 进行图像级异常分类，同时对 similarity map 进行像素级异常监督。

## 可直接用于论文的贡献表述

可以写成下面两点：

1. 提出一种文本引导的视觉 Prompt 增强机制，通过跨模态注意力将异常语义注入视觉表示，实现更细粒度的跨模态对齐。
2. 提出一种自适应双门控融合策略，能够按样本动态平衡视觉 Prompt 信息与异常聚类信息，相比固定加权方式具有更好的灵活性和鲁棒性。
