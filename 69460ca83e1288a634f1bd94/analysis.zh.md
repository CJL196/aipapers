# 1. 论文基本信息

## 1.1. 标题
**DUSt3R: Geometric 3D Vision Made Easy** (DUSt3R：让几何三维视觉变得简单)

## 1.2. 作者
Shuzhe Wang (阿尔托大学), Vincent Leroy, Yohann Cabon, Boris Chidlovskii, Jerome Revaud (Naver Labs Europe)。

## 1.3. 发表期刊/会议
该论文发表于 **CVPR 2024** (IEEE/CVF Conference on Computer Vision and Pattern Recognition)。CVPR 是计算机视觉领域的顶级国际会议，具有极高的影响力和学术声望。

## 1.4. 发表年份
2024年（预印本最早发布于2023年12月）。

## 1.5. 摘要
传统的多视角立体重建 (Multi-View Stereo, MVS) 算法依赖于预先估计相机的内参和外参，这一过程通常繁琐且容易出错。本文提出了 **DUSt3R**，一种全新的三维重建范式。它不需要任何关于相机校准或视角姿态的先验信息，而是将成对重建问题表述为<strong>点图 (Pointmaps)</strong> 的回归问题。该方法通过简单的全局对齐策略，可以无缝统一单目和双目重建，并扩展到多图场景。实验表明，DUSt3R 在深度估计和姿态估计等多项任务上达到了最先进的水平 (state-of-the-art)。

## 1.6. 原文链接
*   **PDF 链接:** [https://arxiv.org/pdf/2312.14132v3.pdf](https://arxiv.org/pdf/2312.14132v3.pdf)
*   <strong>代码仓库 (GitHub):</strong> [https://github.com/naver/dust3r](https://github.com/naver/dust3r)

    ---

# 2. 整体概括

## 2.1. 研究背景与动机
*   **核心问题:** 传统的三维重建（如 SfM 和 MVS）是一个复杂的流水线：特征检测 -> 特征匹配 -> 稀疏重建 -> 相机姿态估计 -> 稠密重建。在这个链条中，任何一步的错误都会累积到下一步。特别是相机姿态估计（SfM 阶段），在视角较少、物体表面缺乏纹理或相机运动不足时极易失败。
*   **重要性:** 三维重建是自动驾驶、机器人导航、考古和文化遗产保护的基础。目前的算法虽然强大，但“门槛”很高，需要精确的相机参数。
*   **创新思路:** 作者反其道而行之，提出：**为什么不直接从图像预测三维形状呢？** 如果我们能直接回归出图像每个像素对应的三维坐标，相机的参数（位置、焦距等）就可以作为副产品顺带计算出来。

## 2.2. 核心贡献/主要发现
1.  **端到端新范式:** 提出了首个从未经校准、未经定位的图像中进行端到端三维重建的流水线。
2.  <strong>点图 (Pointmap) 表示:</strong> 引入了点图作为场景表示，它打破了传统透视相机模型的硬约束，将几何信息隐式地包含在回归结果中。
3.  **全局对齐优化:** 提出了一种简单快速的 3D 空间对齐算法，能将多对图像生成的点图统一到一个全局坐标系中，取代了复杂的捆绑调整 (Bundle Adjustment, BA)。
4.  **性能卓越:** 仅用一个通用的模型，就在单目深度估计、多视角姿态估计等多个任务上打破了纪录。

    ---

# 3. 预备知识与相关工作

## 3.1. 基础概念
*   <strong>多视角立体重建 (Multi-View Stereo, MVS):</strong> 利用多张从不同角度拍摄的图像来恢复场景三维结构的技术。
*   <strong>运动恢复结构 (Structure-from-Motion, SfM):</strong> 从一系列二维图像中同时恢复相机姿态和稀疏三维点云的过程。
*   <strong>主干网络 (backbone):</strong> 模型中负责提取特征的基础网络结构，本文使用的是 <strong>视觉变换器 (Vision Transformer, ViT)</strong>。
*   <strong>词元 (token):</strong> 在 Transformer 架构中，图像被切分成小方块，每个方块被转化为一个向量，称为词元。

## 3.2. 前人工作与技术演进
传统的三维视觉依赖于**几何几何约束**（如对极几何）。
*   <strong>第一阶段（手工设计）:</strong> 依赖 SIFT 等特征点匹配，通过求解基本矩阵来恢复几何。
*   <strong>第二阶段（深度学习组件）:</strong> 使用神经网络替换特征提取（如 SuperPoint）或匹配（如 SuperGlue），但整体框架仍是传统的 SfM/MVS。
*   <strong>第三阶段（神经渲染/隐式表示）:</strong> 如 NeRF，虽然效果惊人，但通常需要已知的相机姿态作为输入。

    **DUSt3R 的差异化:** 相比于上述方法，DUSt3R 彻底抛弃了“先估计相机，再重建三维”的逻辑，改为“直接预测三维，再推导相机”。

## 3.3. 关键前置技术：CroCo
DUSt3R 的架构深受 **CroCo (Cross-view Completion)** 的影响。CroCo 是一种预训练任务：给模型两张有重叠的图像，掩盖其中一张的部分区域，让模型参考另一张图来补全。这种训练让模型学会了不同视角间的几何对应关系。

---

# 4. 方法论

## 4.1. 方法原理
DUSt3R 的核心思想是将两张输入图像 $I^1$ 和 $I^2$ 映射到两个对应的三维点图 $X^{1,1}$ 和 $X^{2,1}$。这两个点图都在第一个相机的坐标系中。这样，两张图的相对位置关系就被隐式地编码在了三维点的坐标中。

## 4.2. 核心方法详解

### 4.2.1. 点图 (Pointmap) 的定义
点图 $\mathbf{X} \in \mathbb{R}^{W \times H \times 3}$ 是一个与图像像素一一对应的矩阵，每个像素 `(i, j)` 存储其在三维空间中的坐标 `(x, y, z)`。
如果已知相机内参 $K$ 和深度图 $D$，点图可以通过下式获得：
$$
X_{i,j} = K^{-1} [i D_{i,j}, j D_{i,j}, D_{i,j}]^\top
$$
其中：
*   $K^{-1}$: 相机内参的逆矩阵，用于将图像平面坐标投影回空间。
*   `i, j`: 像素的横纵坐标。
*   $D_{i,j}$: 该像素对应的深度值。

    在 DUSt3R 中，我们不预先知道 $K$ 和 $D$，而是让模型直接输出 $X$。

### 4.2.2. 网络架构
下图（原文 Figure 2）展示了网络 $\mathcal{F}$ 的架构：

![Figure 2. Architecture of the network $\\mathcal { F }$ Two views of a scene $( I ^ { 1 } , I ^ { 2 } )$ are first encoded in a Siamese manner with a shared ViT encoder. The resulting token representations $F ^ { 1 }$ and $F ^ { 2 }$ are then passed to two transformer decoders that constantly exchange information via pointmaps are expressed in the same coordinate fameof thefrst e $I ^ { 1 }$ .The network $\\mathcal { F }$ is trained using a simple regression loss (Eq. (4))](images/2.jpg)
*该图像是示意图，展示了网络 $\mathcal{F}$ 的架构。两幅场景视图 $(I^1, I^2)$ 首先通过共享的 ViT 编码器进行编码，得到的标记表示分别为 $F^1$ 和 $F^2$。这两个表示随后被送入两个变换解码器，利用点图在相同坐标系中不断交换信息。整个网络通过简单的回归损失进行训练，以实现高效的 3D 重建及相机参数估计。*

1.  <strong>共享编码器 (Siamese ViT Encoder):</strong> 两张图像 $I^1, I^2$ 通过相同的 ViT 主干网络提取词元表示 $F^1, F^2$。
2.  <strong>交互解码器 (Transformer Decoder):</strong> 这是核心部分。解码器包含 $B$ 个块，每个块内两组词元会通过<strong>自注意力 (Self-attention)</strong> 关注自身视角，再通过<strong>交叉注意力 (Cross-attention)</strong> 关注另一视角。
    $$
    G_i^1 = \mathrm{DecoderBlock}_i^1 (G_{i-1}^1, G_{i-1}^2)
    $$
    $$
    G_i^2 = \mathrm{DecoderBlock}_i^2 (G_{i-1}^2, G_{i-1}^1)
    $$
    这里信息在两个分支间不断交换，使模型能通过“三角测量”的直觉来对齐三维空间。
3.  <strong>任务头 (Regression Heads):</strong> 解码后的词元被送入回归头，输出点图 $X^{v,1}$ 和<strong>置信图 (Confidence map)</strong> $C^{v,1}$。

### 4.2.3. 训练目标 (Loss Function)
模型使用**置信度加权的回归损失**进行训练。
首先定义单点回归损失（欧氏距离）：
$$
\ell_{\mathrm{regr}}(v, i) = \left\| \frac{1}{z} X_i^{v,1} - \frac{1}{\bar{z}} \bar{X}_i^{v,1} \right\|
$$
其中 $z$ 是预测点云的平均尺度因子，$\bar{z}$ 是真实标注数据 (Ground Truth) 的平均尺度。这样做是为了处理尺度不确定性。

最终的损失函数 $\mathcal{L}_{\mathrm{conf}}$ 为：
$$
\mathcal{L}_{\mathrm{conf}} = \sum_{v \in \{1, 2\}} \sum_{i \in \mathcal{D}^v} C_i^{v,1} \ell_{\mathrm{regr}}(v, i) - \alpha \log C_i^{v,1}
$$
符号解释：
*   $C_i^{v,1}$: 模型预测的该像素的置信度。
*   $\ell_{\mathrm{regr}}$: 三维坐标误差。
*   $\alpha$: 正则化超参数，防止模型给所有点都分配极低的置信度。
*   $\log C_i^{v,1}$: 惩罚低置信度，迫使模型在容易重建的区域提高精度。

### 4.2.4. 全局对齐 (Global Alignment)
当有 $N$ 张图时，模型会预测多对点图。为了将它们统一，作者提出了全局优化：
$$
\chi^* = \underset{\chi, P, \sigma}{\arg \min} \sum_{e \in \mathcal{E}} \sum_{v \in e} \sum_{i=1}^{HW} C_i^{v,e} \left\| \chi_i^v - \sigma_e P_e X_i^{v,e} \right\|
$$
符号解释：
*   $\chi_i^v$: 最终在全球坐标系中的三维点。
*   $X_i^{v,e}$: 图像对 $e$ 中模型预测的局部三维点。
*   $P_e, \sigma_e$: 图像对 $e$ 对应的旋转平移矩阵和尺度缩放。
*   $C_i^{v,e}$: 预测的置信度，用于过滤噪点。

    该优化直接在 3D 空间进行，比传统的重投影误差优化快得多。

---

# 5. 实验设置

## 5.1. 数据集
实验使用了极其丰富的数据混合体：
*   **静态场景:** MegaDepth (户外), ARKitScenes (室内), ScanNet++ (室内), Habitat (合成室内)。
*   **物体中心:** CO3D-v2 (各种常见物体)。
*   **自动驾驶:** Waymo (户外道路)。
*   **合成数据:** BlendedMVS。
    总计抽取了约 **850万个图像对** 进行训练。

## 5.2. 评估指标
1.  **AbsRel (Absolute Relative Error):** 绝对相对误差。
    $$
    \mathrm{AbsRel} = \frac{1}{N} \sum \frac{|y - \hat{y}|}{y}
    $$
    其中 $y$ 是真值深度，$\hat{y}$ 是预测深度。值越小越好。
2.  **$\delta_{1.25}$ (Threshold Accuracy):** 阈值精度。
    量化预测值与真实值之比在 $(1/1.25, 1.25)$ 范围内的比例。
    $$
    \max(\frac{\hat{y}}{y}, \frac{y}{\hat{y}}) < 1.25
    $$
    值越大越好。
3.  **mAA (mean Average Accuracy):** 平均准确度，常用于评价相机姿态估计的旋转和平移误差。

## 5.3. 对比基线
*   **视觉定位:** HLoc (基于匹配的经典 SOTA), DSAC* (基于坐标回归的 SOTA)。
*   **深度估计:** DPT (强单目模型), Monodepth2 (自监督模型)。
*   **多视角姿态:** PoseDiffusion, COLMAP+SPSG。

    ---

# 6. 实验结果与分析

## 6.1. 核心结果分析
DUSt3R 在几乎所有下游任务中都表现惊人，尤其是在不给定相机内参的情况下。

### 6.1.1. 多视角姿态估计
在 CO3Dv2 和 RealEstate10K 数据集上，DUSt3R 显著超过了此前的最强模型 PoseDiffusion。

以下是原文 **Table 2** 的部分结果对比：

<table>
<thead>
<tr>
<th rowspan="2">Methods (方法)</th>
<th colspan="3">Co3Dv2</th>
<th>RealEstate10K</th>
</tr>
<tr>
<th>RRA@15 ↑</th>
<th>RTA@15 ↑</th>
<th>mAA(30) ↑</th>
<th>mAA(30) ↑</th>
</tr>
</thead>
<tbody>
<tr>
<td>COLMAP+SPSG (经典 SfM)</td>
<td>36.1</td>
<td>27.3</td>
<td>25.3</td>
<td>45.2</td>
</tr>
<tr>
<td>PoseDiffusion (此前 SOTA)</td>
<td>80.5</td>
<td>79.8</td>
<td>66.5</td>
<td>48.0</td>
</tr>
<tr>
<td><strong>DUSt3R 512 (本文)</strong></td>
<td><strong>96.2</strong></td>
<td><strong>86.8</strong></td>
<td><strong>76.7</strong></td>
<td><strong>67.7</strong></td>
</tr>
</tbody>
</table>

**分析:** RRA@15 代表旋转误差小于15度的比例。DUSt3R 达到了 96.2%，远超经典方法。这证明了模型学习到了极强的三维形状先验，能应对大基线、少视角的极端情况。

### 6.1.2. 单目深度估计
即使只输入一张图（输入两次相同图），DUSt3R 的表现依然强劲。

以下是原文 <strong>Table 2 (Monocular Depth 部分)</strong> 的结果：

<table>
<thead>
<tr>
<th rowspan="2">Methods</th>
<th colspan="2">NYUD-v2 (室内)</th>
<th colspan="2">KITTI (户外)</th>
</tr>
<tr>
<th>AbsRel ↓</th>
<th>δ1.25 ↑</th>
<th>AbsRel ↓</th>
<th>δ1.25 ↑</th>
</tr>
</thead>
<tbody>
<tr>
<td>DPT-BEiT (有监督)</td>
<td>0.054</td>
<td>0.965</td>
<td>0.094</td>
<td>0.892</td>
</tr>
<tr>
<td>SlowTv (零样本)</td>
<td>0.115</td>
<td>0.872</td>
<td>0.068</td>
<td>0.561</td>
</tr>
<tr>
<td><strong>DUSt3R 512 (零样本)</strong></td>
<td><strong>0.065</strong></td>
<td><strong>0.940</strong></td>
<td><strong>0.107</strong></td>
<td><strong>0.866</strong></td>
</tr>
</tbody>
</table>

**分析:** 作为一个通用模型，DUSt3R 在未见过的测试集（零样本）上展现了接近专门训练的有监督模型 (DPT) 的水平。

---

# 7. 总结与思考

## 7.1. 结论总结
DUSt3R 证明了**直接回归 3D 坐标**比传统的“内参-外参-三角测量”链条更加鲁棒。它通过 Transformer 的交叉注意力机制，让网络在内部完成了三维对齐。这种“让几何视觉变简单”的承诺确实得到了履行：它统一了单目、双目和多目重建，且无需复杂的相机标定。

## 7.2. 局限性与未来工作
*   **计算开销:** 尽管全局对齐很快，但成对推理的次数随图像数量 $N$ 呈平方增长（$O(N^2)$），处理超大规模场景（如整个城市）仍具挑战。
*   **精度瓶颈:** 回归模型生成的点云虽然“看起来很对”，但在微小的几何精度上（如毫米级测量），可能仍逊色于传统的基于亚像素匹配和三角测量的算法。
*   **未来方向:** 将 DUSt3R 与隐式表示（如 3D Gaussian Splatting 或 NeRF）结合，利用 DUSt3R 提供的初始姿态和点云进行快速建模。

## 7.3. 个人启发与批判
*   **启发:** 这篇论文体现了“暴力美学”和“第一性原理”的结合。它不再纠结于如何优化复杂的几何公式，而是利用大数据和 Transformer 强大的表达能力直接去模拟几何逻辑。
*   **批判:** 尽管论文声称“不需要参数”，但实际上模型是在有参数的数据集上训练出来的，它通过大量数据“背”下了不同焦距和视角的几何规律。此外，对于完全透明或镜面反射的物体（违反了“每个射线对应一个点”的假设），其性能可能会显著下降。