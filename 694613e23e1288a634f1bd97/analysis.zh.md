# 1. 论文基本信息

## 1.1. 标题
**Splatt3R: Zero-shot Gaussian Splatting from Uncalibrated Image Pairs**
（Splatt3R：基于未经校准图像对的零样本 3D 高斯泼溅技术）

## 1.2. 作者
Brandon Smart, Chuanxia Zheng, Iro Laina, Victor Adrian Prisacariul。
作者来自牛津大学的主动视觉实验室（Active Vision Lab）和视觉几何组（Visual Geometry Group, VGG）。

## 1.3. 发表期刊/会议
该论文目前发布于预印本平台 arXiv（2024年8月），考虑到作者背景及论文质量，其目标可能是计算机视觉领域的顶级会议（如 CVPR/ICCV/ECCV）。

## 1.4. 发表年份
2024 年

## 1.5. 摘要
本文介绍了 **Splatt3R**，这是一种无需相机姿态、前馈式（Feed-forward）的 3D 重建和新视角合成（Novel View Synthesis, NVS）方法。该方法仅需一对未经校准（Uncalibrated）的自然图像作为输入，即可预测 3D 高斯泼溅（3D Gaussian Splatting, 3D-GS）参数，而无需任何相机内参、外参或深度信息。Splatt3R 构建在 3D 几何重建模型 **MASt3R** 之上，通过扩展其架构来同时处理 3D 结构和外观。与传统的 3D-GS 不同，Splatt3R 首先通过优化 3D 点云几何损失进行训练，随后优化新视角合成目标，从而避免了从立体视图训练时容易陷入的局部最优解。实验表明，Splatt3R 在 **ScanNet++** 数据集上表现优异，能以 4 FPS 的速度重建场景，并支持实时渲染。

## 1.6. 原文链接
- **arXiv 链接:** [https://arxiv.org/abs/2408.13912](https://arxiv.org/abs/2408.13912)
- **PDF 链接:** [https://arxiv.org/pdf/2408.13912v2.pdf](https://arxiv.org/pdf/2408.13912v2.pdf)
- **发布状态:** 预印本 (Preprint)

  ---

# 2. 整体概括

## 2.1. 研究背景与动机
### 2.1.1. 核心问题
在计算机视觉中，从少量图像重建 3D 场景并合成新视角是一项挑战。现有的主流技术如 **NeRF (Neural Radiance Fields)** 或 **3D Gaussian Splatting (3D-GS)** 通常依赖以下条件：
1.  **密集的图像输入：** 需要几十甚至上百张图片。
2.  **精确的相机姿态：** 需要预先通过 **SfM (Structure from Motion)** 算法（如 COLMAP）计算相机的位置和角度。
3.  **昂贵的逐场景优化：** 每个新场景都需要数小时的训练。

### 2.1.2. 现有挑战与 Gap
对于“野外（In-the-wild）”环境下拍摄的两张照片（立体对），通常缺乏相机参数。现有的前馈式模型（如 pixelSplat）虽然能快速生成结果，但仍然假设相机内参和外参是已知的。如果相机参数未知，这些方法将失效。此外，仅从两张图片训练 3D 高斯模型极易陷入<strong>局部最优解 (Local Minima)</strong>，导致几何结构破碎。

### 2.1.3. 创新思路
Splatt3R 提出利用强大的几何先验模型 **MASt3R**。MASt3R 已经学到了如何从两张未校准图片预测 3D 点云。Splatt3R 在此基础上增加了一个“高斯头（Gaussian Head）”，直接预测每个像素点对应的高斯属性（颜色、不透明度、形状等），从而实现了<strong>无需姿态 (Pose-free)</strong> 的端到端 3D 重建。

## 2.2. 核心贡献/主要发现
1.  **Pose-free Reconstruction:** 首次实现了仅从一对未经校准的图像进行前馈式 3D 高斯泼溅重建。
2.  **MASt3R 架构扩展:** 将纯几何重建模型扩展为兼顾几何与外观的生成模型。
3.  <strong>损失掩膜策略 (Loss Masking Strategy):</strong> 提出了一种基于可见性的损失掩膜，解决了外推视角（Extrapolated viewpoints）训练不稳定的问题。
4.  **高效性能:** 实现 4 FPS 的重建速度，且生成的 3D 高斯模型可实时渲染。

    ---

# 3. 预备知识与相关工作

## 3.1. 基础概念
### 3.1.1. 3D 高斯泼溅 (3D Gaussian Splatting, 3D-GS)
这是一种 3D 场景表示方法。它将场景看作是由成千上万个彩色、半透明的 3D 椭球体（即高斯基元）组成的集合。
每个高斯球由以下参数定义：
*   <strong>均值 (Mean) $\pmb{\mu}$:</strong> 空间中的中心位置。
*   <strong>协方差 (Covariance) $\Sigma$:</strong> 定义形状和旋转。
*   <strong>不透明度 (Opacity) $\alpha$:</strong> 透明度。
*   <strong>球谐函数 (Spherical Harmonics, SH):</strong> 描述视角相关的颜色。

### 3.1.2. 未经校准 (Uncalibrated)
指输入图像没有任何关联的元数据，如焦距（内参）或拍摄时的相机位置和朝向（外参/姿态）。

## 3.2. 前人工作
### 3.2.1. DUSt3R 与 MASt3R
这是由 NAVER LABS Europe 提出的“基础模型”。
*   **DUSt3R:** 通过 Transformer 架构直接回归每像素的 3D 坐标。
*   **MASt3R:** 在 DUSt3R 基础上引入了局部特征匹配（Matching），提高了精度。
    它们不依赖相机初始化，而是直接在统一的坐标系中预测点云。

### 3.2.2. 前馈式高斯模型 (Feed-forward GS)
如 **pixelSplat** 和 **MVSplat**。这些模型使用神经网络一次性预测所有高斯参数，而不是逐场景优化。但它们通常需要已知的相机射线（即需要姿态）。

## 3.3. 技术演进与差异化分析
传统的 3D 视觉路径是：`匹配 -> 估计姿态 (SfM) -> 密集重建 (MVS) -> 渲染`。
Splatt3R 走的是：`端到端学习 -> 直接预测 3D 高斯`。
**核心区别：** Splatt3R 彻底去除了对“相机姿态估计”这一预处理步骤的依赖，真正做到了“拿起两张图就能重建”。

---

# 4. 方法论

## 4.1. 方法原理
Splatt3R 的核心逻辑是利用 **MASt3R** 提供的强大几何定位能力，将每个像素“泼溅”到 3D 空间。由于 MASt3R 已经能准确预测点的位置（即高斯的均值 $\mu$），Splatt3R 只需要额外预测让这些点看起来像真实物体的“外观属性”。

下图（原文 Figure 2）展示了 Splatt3R 的整体架构：

![该图像是示意图，展示了 Splatt3R 方法的模块结构，包括两个输入图像的 ViT 编码器和不同的特征匹配头。图中体现了点云生成和 3D Gaussian Splat 相关的损失计算过程。关键公式涉及 Gaussian 参数的描述。](images/2.jpg)
*该图像是示意图，展示了 Splatt3R 方法的模块结构，包括两个输入图像的 ViT 编码器和不同的特征匹配头。图中体现了点云生成和 3D Gaussian Splat 相关的损失计算过程。关键公式涉及 Gaussian 参数的描述。*

## 4.2. 核心方法详解 (逐层深入)

### 4.2.1. 基础架构：MASt3R 的集成
Splatt3R 使用 **ViT (Vision Transformer)** 作为主干网络（Backbone）。
1.  **编码阶段:** 输入两张图像 $\mathbf{I}^1, \mathbf{I}^2$，通过共享权重的 ViT 提取特征。
2.  **解码阶段:** 使用 Transformer 解码器进行交叉注意力（Cross-attention）处理，使两张图像的特征能够相互“交流”，从而理解深度和匹配关系。

### 4.2.2. 高斯头 (Gaussian Head) 的设计
原文在 MASt3R 已有的“点图分支”和“匹配分支”之外，引入了第三个并行分支——<strong>高斯预测头 (Gaussian Head)</strong>。该分支使用 **DPT (Dense Prediction Transformer)** 架构。
对于每个像素点，模型预测以下参数：
1.  <strong>协方差 ($\Sigma$):</strong> 通过旋转四元数 $q \in \mathbb{R}^4$ 和缩放尺度 $s \in \mathbb{R}^3$ 参数化。
2.  <strong>颜色 ($S$):</strong> 预测球谐函数系数或直接预测 RGB 颜色。
3.  <strong>不透明度 ($\alpha$):</strong> 使用 Sigmoid 激活函数确保值在 `[0, 1]` 之间。
4.  <strong>偏移量 ($\Delta \in \mathbb{R}^3$):</strong> 为了微调点的位置，最终的高斯均值定义为：
    $\mu = x + \Delta$
    其中 $x$ 是 MASt3R 预测的初始 3D 点位置。

### 4.2.3. 几何监督与训练目标
为了避免局部最优，Splatt3R 采用了两步走战略：
<strong>第一步：点云几何损失 ($L_{pts}$)</strong>
直接监督像素在 3D 空间的位置。对于每个有效像素 $i$ 和视图 $v \in \{1, 2\}$：
$$L_{pts} = \sum_{v \in \{1, 2\}} \sum_{i} C_i^v L_{regr}(v, i) - \gamma \log(C_i^v)$$
其中：
*   $C_i^v$ 是模型预测的<strong>置信度图 (Confidence map)</strong>，用于降低天空或透明物体的权重。
*   $\gamma$ 是平衡超参数。
*   回归损失定义为：
    $$L_{regr}(v, i) = \left| \frac{1}{z} X_i^v - \frac{1}{\bar{z}} \hat{X}_i^v \right|$$
    这里 $X$ 是真实 3D 点，$\hat{X}$ 是预测点，$z$ 和 $\bar{z}$ 是归一化因子。

<strong>第二步：新视角渲染损失 ($\mathcal{L}$)</strong>
在几何结构稳定的基础上，通过差分渲染器生成新视角图像 $\hat{\mathbf{I}}$，并与真实目标图像 $\mathbf{I}$ 对比。

### 4.2.4. 损失掩膜策略 (Loss Masking Strategy)
这是本文的关键创新。在广角基线（Wide baseline）下，目标视角可能会看到输入图像中完全被遮挡的区域。强行监督这些区域会导致模型性能崩坏。
Splatt3R 构造了一个掩膜 $M$，只对在输入视图中可见的像素计算损失。
下图（原文 Figure 3）展示了掩膜的生成逻辑：

![Figure 3. Our loss masking approach. Valid pixels are considered to be those that are: inside the frustum of at least one of the views, have their reprojected depth match the ground truth, and are considered valid pixels with valid depth in their dataset.](images/3.jpg)
*该图像是示意图，展示了通过目标视图和上下文视图计算有效像素的过程。这些有效像素需满足在视锥内、具有匹配深度，并被视为有效深度点。最后，生成的损失掩膜将用于后续的3D重建任务。*

最终的掩膜重构损失为：
$$\mathcal{L} = \lambda_{MSE} L_{MSE}(M \odot \hat{\mathbf{I}}, M \odot \mathbf{I}) + \lambda_{LPIPS} L_{LPIPS}(M \odot \hat{\mathbf{I}}, M \odot \mathbf{I})$$
其中：
*   $L_{MSE}$ 是均方误差。
*   $L_{LPIPS}$ 是感知相似度损失。
*   $\odot$ 表示元素逐像素相乘（Hadamard product）。

    ---

# 5. 实验设置

## 5.1. 数据集
*   **ScanNet++:** 包含 450 多个室内场景的高分辨率激光扫描数据。作者使用了官方的训练和验证集，涵盖了复杂的室内几何结构。
*   **In-the-wild Data:** 为了验证泛化性，作者还使用了手机拍摄的室外和物体照片。

## 5.2. 评估指标
论文使用了三个标准指标来衡量新视角合成的质量：
1.  <strong>PSNR (Peak Signal-to-Noise Ratio) - 峰值信噪比:</strong>
    *   **定义:** 衡量图像失真程度，值越高表示图像质量越接近原图。
    *   **公式:** $\mathrm{PSNR} = 10 \cdot \log_{10}\left(\frac{MAX_I^2}{MSE}\right)$。
2.  <strong>SSIM (Structural Similarity Index Measure) - 结构相似性:</strong>
    *   **定义:** 衡量两幅图像的亮度、对比度和结构的相似度，范围 `[0, 1]`，越接近 1 越好。
3.  <strong>LPIPS (Learned Perceptual Image Patch Similarity) - 学习感知图像块相似度:</strong>
    *   **定义:** 利用深度网络提取特征，衡量人眼感知上的相似度，值越低表示越真实。

## 5.3. 对比基线
*   **MASt3R (Point Cloud):** 直接渲染 MASt3R 生成的有色点云（无高斯属性）。
*   **pixelSplat:** 目前最先进的前馈式高斯泼溅模型。为了公平对比，测试了两种情况：使用真实相机姿态 (GT cams) 和使用由 MASt3R 估计的姿态 (MASt3R cams)。

    ---

# 6. 实验结果与分析

## 6.1. 核心结果分析
以下是原文 Table 1 的定量对比结果：

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="3">Close (φ = 0.9, ψ = 0.9)</th>
<th colspan="3">Medium (φ = 0.7, ψ = 0.7)</th>
<th colspan="3">Wide (φ = 0.5, ψ = 0.5)</th>
<th colspan="3">Very Wide (φ = 0.3, ψ = 0.3)</th>
</tr>
<tr>
<th>PSNR ↑</th>
<th>SSIM ↑</th>
<th>LPIPS ↓</th>
<th>PSNR ↑</th>
<th>SSIM ↑</th>
<th>LPIPS ↓</th>
<th>PSNR ↑</th>
<th>SSIM ↑</th>
<th>LPIPS ↓</th>
<th>PSNR ↑</th>
<th>SSIM ↑</th>
<th>LPIPS ↓</th>
</tr>
</thead>
<tbody>
<tr>
<td>Splatt3R (Ours)</td>
<td>**19.66**</td>
<td>**0.757**</td>
<td>**0.234**</td>
<td>**19.66**</td>
<td>**0.770**</td>
<td>**0.229**</td>
<td>**19.41**</td>
<td>**0.783**</td>
<td>**0.220**</td>
<td>**19.18**</td>
<td>**0.794**</td>
<td>**0.209**</td>
</tr>
<tr>
<td>MASt3R (Points)</td>
<td>18.56</td>
<td>0.708</td>
<td>0.278</td>
<td>18.51</td>
<td>0.718</td>
<td>0.259</td>
<td>18.73</td>
<td>0.739</td>
<td>0.245</td>
<td>18.44</td>
<td>0.758</td>
<td>0.242</td>
</tr>
<tr>
<td>pixelSplat (GT)</td>
<td>15.67</td>
<td>0.609</td>
<td>0.436</td>
<td>15.92</td>
<td>0.643</td>
<td>0.381</td>
<td>16.08</td>
<td>0.672</td>
<td>0.407</td>
<td>16.56</td>
<td>0.709</td>
<td>0.299</td>
</tr>
</tbody>
</table>

**分析：**
*   **全方位超越:** Splatt3R 在所有基线长度（从近到远）下均显著优于 pixelSplat。
*   **鲁棒性:** 即使 pixelSplat 使用了真实的相机姿态，其表现依然不如无需姿态的 Splatt3R。这说明 MASt3R 的几何先验比 pixelSplat 的概率深度估计更可靠。
*   **点云 vs 高斯:** Splatt3R 比单纯的 MASt3R 点云渲染效果更好（PSNR 提升约 1.1dB），证明了学习高斯属性对于填补空隙和改善外观的有效性。

## 6.2. 定性分析 (可视化)
下图（原文 Figure 4）展示了在 ScanNet++ 上的重建效果：

![Figure 4. Qualitative comparisons on ScanNet++. We compare different methods on ScanNet $^ { + + }$ testing examples. The two context camera views for each image are included in the first row of the table.](images/4.jpg)
*该图像是对比图，展示了不同方法在ScanNet++测试示例上的效果。第一行是每个图像的真实场景，下面依次是我们的方法、MASt3R（点云）和PixelSplat（MASt3R姿态）所生成的结果，展示了在不同视角下的重建效果。*

可以看到，Splatt3R 生成的图像比点云渲染更平滑，且比 pixelSplat 更完整。Figure 5 展示了该模型在野外图像上的极强泛化能力：

![Figure 5. Examples of Splatt3R generalizing to in-the-wild testing examples. The bottom row showcases examples with few direct pixel correspondences between the two context images.](images/5.jpg)
*该图像是展示Splatt3R在野外测试样例中的结果。底部行展示了两个上下文图像之间几乎没有直接像素对应的例子。*

## 6.3. 消融实验
原文 Table 2 指出：
*   <strong>Loss Masking (损失掩膜):</strong> 极其关键。如果不使用掩膜，高斯球的大小会无限膨胀，导致内存溢出。
*   **LPIPS Loss:** 显著提升了图像的视觉清晰度。
*   **Offsets ($\Delta$):** 对点位的小幅修正能进一步提升性能。

    ---

# 7. 总结与思考

## 7.1. 结论总结
Splatt3R 是 3D 重建领域的一个重要进展。它通过结合<strong>基础几何模型 (MASt3R)</strong> 和 <strong>3D 高斯泼溅 (3D-GS)</strong>，解决了稀疏、未校准视图下的重建难题。
其核心价值在于：**降低了 3D 重建的门槛**。用户不再需要复杂的摄影测量知识或昂贵的计算设备，只需两张照片，即可在几百毫秒内得到可交互的 3D 模型。

## 7.2. 局限性与未来工作
1.  **遮挡区域的“空洞”:** Splatt3R 遵循“不瞎猜”原则，使用了损失掩膜。这导致输入视角看不到的地方在渲染时是空白的。未来的方向可能是结合生成式模型（如 Diffusion Models）来幻觉（Hallucinate）出缺失的背面。
2.  **基线限制:** 虽然支持宽基线，但如果两张图完全没有重叠，重建依然会非常困难。
3.  **内存占用:** 预测每像素高斯参数对显存仍有一定要求，尽管论文中提到在 2080Ti 上即可运行。

## 7.3. 个人启发与批判
**启发:**
*   **基础模型的力量:** Splatt3R 的成功证明了“在一个强大的预训练几何模型上做加法”比“从零开始设计一个复杂的端到端 NVS 系统”更有效。
*   **简单即美:** 论文并没有设计复杂的注意力机制来预测高斯参数，而是直接在 MASt3R 后面接了一个 DPT 头，这种简洁的架构往往具有更好的泛化性。

**批判:**
*   论文在球谐函数（SH）的实验上遇到了过拟合问题。这反映出在极稀疏视图（仅两张图）下，区分“光照变化”和“物体颜色”依然是一个病态问题。
*   目前的模型主要针对室内场景（ScanNet++）训练，虽然在室外有展示，但对于大规模城市级别的重建能力尚待验证。