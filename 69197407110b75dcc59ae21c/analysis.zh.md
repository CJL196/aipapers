# 1. 论文基本信息

## 1.1. 标题
VGGT: Visual Geometry Grounded Transformer (VGGT: 视觉几何基础 Transformer)

## 1.2. 作者
Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, David Novotny。
他们主要隶属于牛津大学视觉几何组 (Visual Geometry Group, University of Oxford) 和 Meta AI。

## 1.3. 发表期刊/会议
预印本 (Pre-print)，发布于 arXiv。

## 1.4. 发表年份
2025

## 1.5. 摘要
本文提出了 `VGGT` (Visual Geometry Grounded Transformer)，一个前馈神经网络，能够直接从一张、几张甚至数百张图像中推断出场景的所有关键 3D 属性，包括相机参数、点图 (point maps)、深度图 (depth maps) 和 3D 点轨迹 (3D point tracks)。这种方法是 3D 计算机视觉领域的一大进步，此前模型通常受限于特定任务且高度专业化。`VGGT` 简单高效，能在不到一秒的时间内完成图像重建，并且性能超越了需要通过视觉几何优化技术进行后处理的替代方法。该网络在多项 3D 任务中取得了最先进的 (state-of-the-art) 结果，包括相机参数估计、多视图深度估计、密集点云重建和 3D 点跟踪。作者还展示了将预训练的 `VGGT` 作为特征主干网络 (feature backbone) 可以显著增强下游任务，例如非刚性点跟踪和前馈式新视角合成 (feed-forward novel view synthesis)。代码和模型已公开。

## 1.6. 原文链接
- 论文原文链接: https://arxiv.org/abs/2503.11651
- PDF 链接: https://arxiv.org/pdf/2503.11651v1.pdf
- 发布状态: 该论文于 2025-03-14T17:59:47.000Z 在 arXiv 上发布，目前为预印本。

# 2. 整体概括

## 2.1. 研究背景与动机
传统的 3D 重建方法，如 `Structure-from-Motion` (SfM)，主要依赖于迭代优化技术，例如 `Bundle Adjustment` (BA)。虽然机器学习在特征匹配和单目深度预测等辅助任务中发挥了作用，但视觉几何仍然在 3D 重建中扮演核心角色，导致了高复杂度和计算成本。近年来，诸如 `DUSt3R` 和 `MASt3R` 等方法显示了直接用神经网络解决 3D 任务的潜力，但它们通常一次只能处理两张图像，并且仍需通过后处理（如融合成对重建）来处理更多图像。

本文的动机在于：随着神经网络能力的不断增强，是否能最终通过一个神经网络直接解决 3D 任务，几乎完全摒弃几何后处理？现有的深度 3D 模型（如 `DepthAnything`、`MoGe`、`LRM`）通常专注于单一 3D 任务，而非统一地预测所有相关的 3D 属性。因此，论文旨在开发一个简单、高效、多功能的神经网络，能够直接从多个视图中一次性推断出场景的所有关键 3D 属性。

## 2.2. 核心贡献/主要发现
本文提出了 `VGGT` (Visual Geometry Grounded Transformer)，其核心贡献和主要发现如下：

1.  **统一的多任务 3D 属性预测器：** 引入了 `VGGT`，一个大型前馈 `Transformer` 网络。它能够从一张、几张甚至数百张图像中，在数秒内预测场景的所有关键 3D 属性，包括相机内参 (intrinsics) 和外参 (extrinsics)、点图、深度图以及 3D 点轨迹。这标志着从单一任务、迭代优化方法向统一、前馈解决方案的转变。
2.  **超越传统方法的直接可用预测：** 论文证明了 `VGGT` 的预测结果可以直接使用，且在多项 3D 任务中具有高度竞争力，通常优于那些依赖耗时后处理优化技术（如 `Bundle Adjustment`）的最先进方法。这突显了其在效率和性能上的显著优势。
3.  **结合优化方法后的进一步性能提升：** `VGGT` 的预测结果作为高质量初始化，与 `Bundle Adjustment` 等视觉几何优化技术结合后，能在所有任务上实现最先进的 (state-of-the-art) 性能，甚至超越专门处理某个 3D 任务的方法，显著提升了质量。
4.  **强大的泛化能力和下游任务应用：** `VGGT` 作为特征主干网络，其学习到的特征能够显著增强下游任务的性能，例如动态视频中的点跟踪和前馈式新视角合成，证明了模型的通用性和鲁棒性。
5.  **创新性架构和训练策略：** `VGGT` 基于一个相对标准的 `Transformer` 架构，引入了 `Alternating-Attention` (交替注意力) 机制，在帧内和全局层面交替进行自注意力，实现了信息融合的平衡。同时，通过在大量多样化的 3D 标注数据集上进行多任务联合训练，使其能够学习到 3D 属性之间的内在联系。

# 3. 预备知识与相关工作

## 3.1. 基础概念
*   <strong>3D 属性 (3D Attributes):</strong>
    *   <strong>相机参数 (Camera Parameters):</strong> 描述相机如何捕捉 3D 场景并将其投影到 2D 图像上的属性。包括：
        *   <strong>内参 (Intrinsics):</strong> 描述相机内部几何特性，如焦距 (focal length)、主点 (principal point) 和畸变系数 (distortion coefficients)。这些参数固定了从相机坐标系到像素坐标系的投影方式。
        *   <strong>外参 (Extrinsics):</strong> 描述相机在世界坐标系中的位置和方向，通常由旋转 (rotation) 和平移 (translation) 组成。旋转可以用四元数 (quaternion) 表示，平移可以用三维向量表示。
    *   <strong>深度图 (Depth Map):</strong> 一张与输入图像尺寸相同的灰度图像，其中每个像素的值代表对应 2D 图像像素在 3D 场景中的深度信息（即从相机到场景点的距离）。
    *   <strong>点图 (Point Map):</strong> 一张与输入图像尺寸相同的图像，其中每个像素的值代表对应 2D 图像像素在 3D 场景中的三维坐标。在本文中，点图是视点不变的 (viewpoint invariant)，意味着所有 3D 点都定义在第一个相机的坐标系中，作为世界坐标系。
    *   <strong>3D 点轨迹 (3D Point Tracks):</strong> 描述一个 3D 场景点在不同图像帧中的 2D 对应位置序列。跟踪是识别和连接图像序列中同一物理点在不同时间或不同视角下的投影。
*   **Transformer:** 一种基于自注意力机制 (self-attention mechanism) 的神经网络架构，最初用于自然语言处理。
    *   <strong>自注意力 (Self-Attention):</strong> 允许模型在处理序列中的某个元素时，同时考虑序列中的所有其他元素，并根据它们的重要性分配不同的权重。其核心思想是通过计算查询 (Query, $Q$)、键 (Key, $K$) 和值 (Value, $V$) 矩阵之间的相似性来聚合信息。
        $$
        \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
        $$
        其中，$Q$ 是查询矩阵，$K$ 是键矩阵，$V$ 是值矩阵，$d_k$ 是键向量的维度，用于缩放点积以防止梯度过小。
    *   <strong>词元 (Token):</strong> 在 `Transformer` 模型中，输入数据（如图像块、单词）被离散化成一系列离散的单元，称为词元。
    *   <strong>主干网络 (Backbone):</strong> 指神经网络中负责提取通用特征的底层部分，通常是一个预训练的深度卷积网络或 `Transformer` 编码器。
*   <strong>视觉几何 (Visual Geometry):</strong> 计算机视觉的一个分支，关注从图像中提取 3D 信息，通常涉及相机模型、投影、几何变换等数学概念。
    *   **Bundle Adjustment (BA):** 束调整，一种非线性优化技术，用于在 3D 重建中精炼相机姿态 (camera poses) 和 3D 点坐标，通过最小化观测图像点与 3D 点重投影到图像上的点之间的误差。它同时优化相机参数和场景点，以获得全局一致的解。
    *   **Multi-View Stereo (MVS):** 多视图立体，旨在从多张已知相机参数的图像中重建场景的密集 3D 几何（例如，密集点云或网格）。
    *   **Perspective n-Point (PnP) problem:** 透视 $n$ 点问题，给定 $n$ 个 3D 空间点及其在 2D 图像上的投影，求解相机在世界坐标系中的姿态（旋转和平移）。
*   <strong>损失函数 (Loss Functions):</strong>
    *   <strong>Huber 损失 (Huber Loss):</strong> 一种在回归问题中使用的损失函数，结合了 L1 损失 (对异常值不敏感) 和 L2 损失 (在误差较小时平滑)。它在误差较小时是平方误差，误差较大时是线性误差。
        $$
        L_{\delta}(a) =
        \begin{cases}
            \frac{1}{2}a^2 & \text{if } |a| \le \delta \\
            \delta(|a| - \frac{1}{2}\delta) & \text{if } |a| > \delta
        \end{cases}
        $$
        其中 $a$ 是误差，$δ$ 是阈值。
    *   <strong>Aleatoric Uncertainty Loss (偶然不确定性损失):</strong> 一种考虑数据本身固有噪声的损失函数。它通过让模型预测任务输出的同时，也预测输出的不确定性（通常是方差），并在损失函数中对不确定性大的预测给予较小的惩罚，从而在训练中学习不确定性。
    *   <strong>通道广播元素级乘积 (Channel-Broadcast Element-wise Product) $\odot$:</strong> 在深度学习中，指一个较小维度的张量（例如，一个不确定性图）沿着某个轴扩展其维度，使其与另一个较大维度的张量（例如，一个预测误差图）的维度匹配，然后进行元素级的乘法。

## 3.2. 前人工作
*   **Structure from Motion (SfM):** 经典的计算机视觉问题，用于从一系列图像中估计相机参数并重建稀疏点云。传统的 `SfM` 管线包括图像匹配、三角测量 (triangulation) 和 `Bundle Adjustment`。`COLMAP` [94] 是一个基于传统管线的流行框架。近年来，深度学习改进了 `SfM` 管线的许多组件，如关键点检测和图像匹配。`VGGSfM` [125] 等方法探索了端到端可微分的 `SfM`，并在挑战性场景中超越了传统算法。
*   **Multi-view Stereo (MVS):** 旨在从多张重叠图像中密集重建场景几何，通常假设相机参数已知。`MVS` 方法分为传统手工特征方法、全局优化方法和基于学习的方法。`DUSt3R` [129] 和 `MASt3R` [62] 直接从一对视图中估计对齐的密集点云，无需相机参数。一些同期工作 [111, 127, 141, 156] 尝试用神经网络替代 `DUSt3R` 的测试时优化，但性能次优或相当。本文的 `VGGT` 旨在大幅超越 `DUSt3R` 和 `MASt3R`。
*   **Tracking-Any-Point (TAP):** 目标是跟踪视频序列中感兴趣的点，包括动态运动。`Particle Video` [91] 和 `PIPs` [44] 首次引入，`TAP-Vid` [23] 提出了基准。`TAPIR` [24]、`CoTracker` [55, 56]、`DOT` [60]、`TAPTR` [63] 和 `LocoTrack` [13] 等方法进一步发展了专门的点跟踪器。本文展示了 `VGGT` 的特征结合现有跟踪器也能实现最先进的性能。
*   <strong>大型 3D 神经网络 (Large 3D Neural Networks):</strong> 例如 `DepthAnything` [142]、`MoGe` [128] 和 `LRM` [49]。然而，这些模型通常只专注于单一 3D 任务，如单目深度估计或新视角合成。`VGGT` 的不同之处在于它使用共享主干网络同时预测所有相关的 3D 属性。

## 3.3. 技术演进
3D 重建领域的技术演进经历了从纯几何方法到结合机器学习，再到深度学习主导的变革：
1.  <strong>传统几何方法 (Traditional Geometry-based Methods):</strong> 以 `SfM` 和 `MVS` 为代表，依赖于手工特征、几何约束和迭代优化（如 `Bundle Adjustment`）。这些方法虽然精度高，但计算成本昂贵，且对初始值敏感。
2.  <strong>机器学习辅助几何 (ML-Assisted Geometry):</strong> 深度学习开始改进 `SfM` 管线中的特定组件，如关键点检测 (SuperPoint [21]) 和图像匹配 (SuperGlue [92])，提高了鲁棒性和效率。`VGGSfM` [125] 更进一步，将整个 `SfM` 管线构建为可微分框架，实现了端到端训练。
3.  <strong>学习密集 3D 几何 (Learning Dense 3D Geometry):</strong> `DUSt3R` [129] 和 `MASt3R` [62] 等方法尝试直接学习密集点云，但它们通常局限于双视图输入，并且仍需要额外的几何后处理来融合结果或处理多视图场景。
4.  <strong>端到端前馈 3D 模型 (End-to-End Feed-Forward 3D Models):</strong> 近期出现了专注于单一 3D 任务的大型神经网络，如 `DepthAnything` (深度估计) 和 `LRM` (新视角合成)。然而，它们缺乏对多项 3D 属性的统一预测。
5.  **`VGGT` 的位置:** `VGGT` 代表了这一演进的最新阶段，它旨在构建一个**纯前馈、多任务、多视图**的 `Transformer` 模型，直接预测所有关键 3D 属性，并尽量减少对几何后处理的依赖。它借鉴了大型语言模型 (LLM) 和视觉 `Transformer` (ViT) 的成功经验，用一个通用架构在大规模数据集上进行训练，以学习复杂 3D 几何关系。

## 3.4. 差异化分析
`VGGT` 与相关工作的主要区别和创新点体现在以下几个方面：

*   **前馈与多视图处理：** `VGGT` 是一个前馈网络，能够直接从一张、几张甚至数百张图像中一次性推断所有 3D 属性。这与 `DUSt3R` 和 `MASt3R` 等方法形成鲜明对比，后者通常只能处理双视图，并且需要耗时的全局对齐或迭代优化等后处理步骤来处理更多视图或获得可用结果。
*   **统一的多任务预测：** `VGGT` 使用一个共享主干网络同时预测相机参数、深度图、点图和 3D 点轨迹等多种 3D 属性。这与 `DepthAnything`、`MoGe` 和 `LRM` 等只专注于单一 3D 任务的大型模型不同。`VGGT` 认为学习这些相互关联的 3D 属性有助于提高整体准确性。
*   **超越优化方法的性能：** 论文强调 `VGGT` 的直接预测结果在许多情况下就已优于那些需要复杂几何优化后处理（如 `Bundle Adjustment` 或全局对齐）的最先进方法。这显著降低了计算成本和复杂性，使其适用于实时应用。
*   **最小化的 3D 归纳偏置：** `VGGT` 基于一个相对标准的、大规模的 `Transformer` 架构，除了引入 `Alternating-Attention` (交替注意力) 机制外，并未特别设计针对 3D 的归纳偏置。这与 `VGGSfM` 等通过可微分 `Bundle Adjustment` 紧密集成几何先验的方法形成对比。`VGGT` 依赖于大规模数据训练来学习 3D 几何。
*   **通用特征主干网络：** `VGGT` 预训练的特征可以作为强大的主干网络，显著增强多种下游任务（如非刚性点跟踪和新视角合成）的性能，体现了其通用性和作为基础模型 (foundation model) 的潜力。

# 4. 方法论

## 4.1. 方法原理
`VGGT` 的核心思想是利用一个大型 `Transformer` 神经网络，直接从一组输入图像中推断出场景的多种 3D 几何属性。该模型旨在通过前馈方式，在不依赖传统视觉几何迭代优化的情况下，实现高质量的 3D 重建。它通过 `Alternating-Attention` (交替注意力) 机制，平衡了帧内信息整合和跨帧信息聚合，从而有效地处理多视图输入。同时，`VGGT` 采取了多任务学习范式，同时预测相机参数、深度图、点图和跟踪特征，即使这些任务之间存在闭式关系，实验也表明联合学习能带来性能提升。

## 4.2. 核心方法详解
### 4.2.1. 问题定义与符号
输入是一系列 $N$ 个 RGB 图像 $(I_i)_{i=1}^N$，其中 $I_i \in \mathbb{R}^{3 \times H \times W}$。`Transformer` $f$ 将这个序列映射到一组相应的 3D 标注，每个帧对应一个：
$$
f \left( ( I _ { i } ) _ { i = 1 } ^ { N } \right) = ( \mathbf { g } _ { i } , D _ { i } , P _ { i } , T _ { i } ) _ { i = 1 } ^ { N } .
$$
其中：
*   $\mathbf{g}_i \in \mathbb{R}^9$ 是第 $i$ 幅图像的相机参数，采用 [125] 的参数化方式，表示为 $\mathbf{g} = [\mathbf{q}, \mathbf{t}, \mathbf{f}]$ 的拼接，其中：
    *   $\mathbf{q} \in \mathbb{R}^4$ 是旋转四元数 (rotation quaternion)。
    *   $\mathbf{t} \in \mathbb{R}^3$ 是平移向量 (translation vector)。
    *   $\mathbf{f} \in \mathbb{R}^2$ 是视野 (field of view) 参数，表示焦距。
    *   假设相机主点 (principal point) 位于图像中心。
*   $D_i \in \mathbb{R}^{H \times W}$ 是第 $i$ 幅图像的深度图。对于像素位置 $\mathbf{y} \in \mathcal{T}(I_i)$，其深度值 $D_i(\mathbf{y}) \in \mathbb{R}^+$ 表示从第 $i$ 个相机观察到的对应 3D 点的深度。
*   $P_i \in \mathbb{R}^{3 \times H \times W}$ 是第 $i$ 幅图像的点图。它将每个像素与对应的 3D 场景点 $P_i(\mathbf{y}) \in \mathbb{R}^3$ 相关联。点图是视点不变的，所有 3D 点都定义在第一个相机 $\mathbf{g}_1$ 的坐标系中，作为世界参考系。
*   $T_i \in \mathbb{R}^{C \times H \times W}$ 是 $C$ 维的密集跟踪特征图，用于点跟踪。

    `Transformer` $f$ 不直接输出点轨迹，而是输出跟踪特征 $T_i$。点跟踪任务由一个单独的跟踪模块 $\tau$ 完成。给定查询图像 $I_q$ 中的固定查询点 $\mathbf{y}_q$，网络输出一个轨迹 $\mathcal{T}^\star(\mathbf{y}_q) = (\mathbf{y}_i)_{i=1}^N$，其中 $\mathbf{y}_i \in \mathbb{R}^2$ 是在所有图像 $I_i$ 中与 $\mathbf{y}_q$ 对应的 2D 点。跟踪模块 $\tau$ 的定义为：
$$
\begin{array} { r } { \mathcal { T } ( ( \mathbf { y } _ { j } ) _ { j = 1 } ^ { M } , ( T _ { i } ) _ { i = 1 } ^ { N } ) = ( ( \hat { \mathbf { y } } _ { j , i } ) _ { i = 1 } ^ { N } ) _ { j = 1 } ^ { M } } \end{array}
$$
这个模块接收查询点 $\mathbf{y}_q$ 和由 `Transformer` $f$ 输出的密集跟踪特征 $T_i$，然后计算轨迹。`Transformer` $f$ 和跟踪模块 $\tau$ 联合进行端到端训练。

**预测顺序和坐标系：** 输入图像的顺序是任意的，但第一张图像被选为参考帧。网络架构对于除了第一帧之外的所有帧都是置换等变的 (permutation equivariant)。所有预测的相机、点图和深度图都在第一个相机 $\mathbf{g}_1$ 的坐标系中。因此，第一个相机的外参被设置为单位变换，即旋转四元数 $\mathbf{q}_1 = [0, 0, 0, 1]$ 和平移向量 $\mathbf{t}_1 = [0, 0, 0]$。

**超完备预测：** `VGGT` 同时预测相机参数、深度图和点图，即使这些量之间存在闭式关系（例如，相机参数可以从点图推断，深度图可以从点图和相机参数推断）。作者在实验中发现，在训练期间明确预测所有这些量可以带来显著的性能提升。然而，在推理期间，通过组合独立估计的深度图和相机参数来生成 3D 点比直接使用专门的点图头 (point map head) 更准确。

### 4.2.2. 特征主干网络 (Feature Backbone)
`VGGT` 模型 $f$ 被实现为一个大型 `Transformer`。每个输入图像 $I$ 首先通过 `DINO` [78] 被分块 (patchified) 为一组 $K$ 个词元 $\mathrm{t}^I \in \mathbb{R}^{K \times C}$。所有帧的图像词元集合 $\mathrm{t}^I = \bigcup_{i=1}^N \{\mathrm{t}_i^I\}$ 被送入主网络结构，该结构交替使用帧内自注意力 (frame-wise self-attention) 和全局自注意力 (global self-attention) 层。

<strong>交替注意力 (Alternating-Attention, AA)：</strong> `VGGT` 引入了 `Alternating-Attention` 机制来调整标准 `Transformer` 设计。
*   **帧内自注意力：** 独立地关注每个帧内的词元 $\mathrm{t}_k^I$。
*   **全局自注意力：** 联合地关注所有帧中的词元 $\mathrm{t}^I$。
    这种交替方式平衡了跨图像的信息整合和图像内词元的激活归一化。默认情况下，模型使用 $L=24$ 层全局注意力与帧内注意力交替。架构中不使用任何交叉注意力 (cross-attention) 层，只使用自注意力层。

### 4.2.3. 预测头 (Prediction Heads)
为了预测相机参数、深度图、点图和跟踪特征，模型首先对每个输入图像 $I_i$ 的图像词元 $\mathrm{t}_i^I$ 进行增强。
*   **词元增强：** 增加一个相机词元 $\mathbf{t}_i^{\mathbf{g}} \in \mathbb{R}^{1 \times C'}$ 和四个寄存器词元 (register tokens) $\mathrm{t}_i^R \in \mathbb{R}^{4 \times C'}$。所有这些词元被拼接起来，经过 `Alternating-Attention` `Transformer` 处理，输出精炼后的词元 $(\hat{\mathrm{t}}_i^I, \hat{\mathrm{t}}_i^{\mathbf{g}}, \hat{\mathrm{t}}_i^R)_{i=1}^N$。
*   **特殊词元：** 第一个帧的相机词元 $(\mathbf{t}_1^{\mathbf{g}} := \bar{\mathbf{t}}^{\mathbf{g}})$ 和寄存器词元 $(\mathrm{t}_1^R := \bar{\mathrm{t}}^R)$ 设置为与所有其他帧 $(i \in [2, \dots, N])$ 的词元 $(\mathbf{t}_i^{\mathbf{g}} := \bar{\bar{\mathbf{t}}}^{\mathbf{g}}, \mathrm{t}_i^R := \bar{\bar{\mathbf{t}}}^R)$ 不同的可学习词元。这使得模型能够区分第一帧，并将其作为 3D 预测的坐标参考帧。输出的寄存器词元 $\hat{\mathrm{t}}_i^R$ 被丢弃，而 $\hat{\mathrm{t}}_i^I, \hat{\mathrm{t}}_i^{\mathbf{g}}$ 用于预测。

**相机估计：**
相机的预测 $(\hat{\mathbf{g}}^i)_{i=1}^N$ 是通过将精炼后的相机词元 $(\hat{\mathrm{t}}_i^{\mathbf{g}})_{i=1}^N$ 输入到四个额外的自注意力层，然后接一个线性层来完成的。这形成了相机头 (camera head)，用于预测相机内参和外参。

**密集预测：**
输出的图像词元 $\hat{\mathrm{t}}_i^I$ 用于预测密集输出，即深度图 $D_i$、点图 $P_i$ 和跟踪特征 $T_i$。具体来说，$\hat{\mathrm{t}}_i^I$ 首先通过一个 `DPT` 层 [87] 转换为密集特征图 $F_i \in \mathbb{R}^{C'' \times H \times W}$。然后，每个 $F_i$ 通过一个 $3 \times 3$ 卷积层映射到相应的深度图 $D_i$ 和点图 $P_i$。`DPT` 头还输出密集特征 $T_i \in \mathbb{R}^{C \times H \times W}$，作为跟踪模块的输入。
同时，模型还会预测深度图和点图的不确定性图 (uncertainty maps) $\Sigma_i^D \in \mathbb{R}_+^{H \times W}$ 和 $\Sigma_i^P \in \mathbb{R}_+^{H \times W}$。这些不确定性图在损失函数中使用，并在训练后反映模型对预测的置信度。

**跟踪：**
为了实现跟踪模块 $\tau$，模型使用了 `CoTracker2` [57] 的架构。它接收密集跟踪特征 $T_i$ 作为输入。给定查询图像 $I_q$ 中的查询点 $\mathbf{y}_j$，跟踪头 $\tau$ 预测所有图像 $I_i$ 中与 $\mathbf{y}_j$ 对应的 2D 点集。具体步骤是，首先在查询点 $\mathbf{y}_j$ 处对查询图像 $T_q$ 的特征图进行双线性采样 (bilinearly sampled) 以获得其特征。然后，该特征与所有其他特征图 $T_i, i \neq q$ 进行关联，以获得一组相关图 (correlation maps)。这些相关图随后由自注意力层处理，以预测最终的 2D 对应点 $\hat{\mathbf{y}}_i$。该跟踪器不假设输入帧的时间顺序，因此可以应用于任何图像集。

### 4.2.4. 训练 (Training)
**训练损失：** `VGGT` 模型 $f$ 采用多任务损失进行端到端训练：
$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { c a m e r a } } + \mathcal { L } _ { \mathrm { d e p t h } } + \mathcal { L } _ { \mathrm { p m a p } } + \lambda \mathcal { L } _ { \mathrm { t r a c k } } . } \end{array}
$$
其中，相机损失 $\mathcal{L}_{\mathrm{camera}}$、深度损失 $\mathcal{L}_{\mathrm{depth}}$ 和点图损失 $\mathcal{L}_{\mathrm{pmap}}$ 的范围相似，无需相互加权。跟踪损失 $\mathcal{L}_{\mathrm{track}}$ 乘以权重因子 $\lambda = 0.05$。

*   **相机损失 $\mathcal{L}_{\mathrm{camera}}$：** 监督预测相机参数 $\hat{\mathbf{g}}_i$ 与真实标注 (Ground Truth) $\mathbf{g}_i$ 的差异，使用 `Huber` 损失 $| \cdot |_\epsilon$。
    $$
    \begin{array} { r } { \mathcal { L } _ { \mathrm { c a m e r a } } = \sum _ { i = 1 } ^ { N } \| \hat { \mathbf { g } } _ { i } - \mathbf { g } _ { i } \| _ { \epsilon } }
    \end{array}
    $$
    *   $\hat{\mathbf{g}}_i$: 第 $i$ 帧预测的相机参数。
    *   $\mathbf{g}_i$: 第 $i$ 帧真实标注 (Ground Truth) 的相机参数。
    *   $\|\cdot\|_\epsilon$: `Huber` 损失。
*   **深度损失 $\mathcal{L}_{\mathrm{depth}}$：** 遵循 `DUSt3R` [129]，采用 `aleatoric uncertainty loss` [59, 75]，根据预测的不确定性图 $\Sigma_i^D$ 对预测深度 $\hat{D}_i$ 与真实标注 (Ground Truth) 深度 $D_i$ 之间的差异进行加权。
    $$
    \begin{array} { r } { \mathcal { L } _ { \mathrm { d e p t h } } = \sum _ { i = 1 } ^ { N } \big ( \| \Sigma _ { i } ^ { D } \odot \big ( \hat { D } _ { i } - D _ { i } \big ) \| + \big \| \Sigma _ { i } ^ { D } \odot \big ( \nabla \hat { D } _ { i } - \nabla D _ { i } \big ) \big \| - \alpha \log \Sigma _ { i } ^ { D } \big ) } \end{array}
    $$
    *   $\hat{D}_i$: 第 $i$ 帧预测的深度图。
    *   $D_i$: 第 $i$ 帧真实标注 (Ground Truth) 的深度图。
    *   $\Sigma_i^D$: 第 $i$ 帧预测的深度不确定性图。
    *   $\odot$: 通道广播元素级乘积。
    *   $\nabla$: 梯度算子。
    *   $\alpha$: 权重参数，用于平衡不确定性项。
*   **点图损失 $\mathcal{L}_{\mathrm{pmap}}$：** 定义与深度损失类似，使用点图不确定性 $\Sigma_i^P$。
    $$
    \begin{array} { r } { \mathcal { L } _ { \mathrm { pmap } } = \sum _ { i = 1 } ^ { N } \big ( \| \Sigma _ { i } ^ { P } \odot ( \hat { P _ { i } } - P _ { i } ) \| + \| \Sigma _ { i } ^ { P } \odot ( \nabla \hat { P } _ { i } - \nabla P _ { i } ) \| - \alpha \log \Sigma _ { i } ^ { P } \big ) } \end{array}
    $$
    *   $\hat{P}_i$: 第 $i$ 帧预测的点图。
    *   $P_i$: 第 $i$ 帧真实标注 (Ground Truth) 的点图。
    *   $\Sigma_i^P$: 第 $i$ 帧预测的点图不确定性图。
*   **跟踪损失 $\mathcal{L}_{\mathrm{track}}$：**
    $$
    \begin{array} { r l } { \mathcal { L } _ { \mathrm { t r a c k } } } & { { } = } \end{array} \begin{array} { r } { \sum _ { j = 1 } ^ { M } \sum _ { i = 1 } ^ { N } \| { \bf y } _ { j , i } - \hat { { \bf y } } _ { j , i } \| } \end{array}
    $$
    *   $\mathbf{y}_j$: 查询图像 $I_q$ 中的真实标注 (Ground Truth) 查询点。
    *   $\mathbf{y}_{j,i}$: $\mathbf{y}_j$ 在图像 $I_i$ 中的真实标注 (Ground Truth) 对应点。
    *   $\hat{\mathbf{y}}_{j,i}$: 跟踪模块 $\tau$ 预测的对应点。
    *   $M$: 查询点的总数。
    *   除了上述 L1 损失，还应用了可见性损失 (visibility loss，二元交叉熵) 来估计点在给定帧中是否可见。

        <strong>真实标注坐标归一化 (Ground Truth Coordinate Normalization)：</strong> 为了消除 3D 重建结果的尺度和全局参考系歧义，作者对数据进行归一化。具体做法是：
1.  所有量都表示在第一个相机 $\mathbf{g}_1$ 的坐标系中。
2.  计算点图 $P$ 中所有 3D 点到原点的平均欧几里得距离，并用此尺度来归一化相机平移 $\mathbf{t}$、点图 $P$ 和深度图 $D$。
    重要的是，与 `DUSt3R` [129] 不同，`VGGT` 不对 `Transformer` 的预测输出应用这种归一化；相反，它强制模型从训练数据中学习到所选择的归一化方式。

<strong>实现细节 (Implementation Details)：</strong>
*   模型包含 $L=24$ 层全局和帧内注意力，总参数量约为 12 亿。
*   使用 `AdamW` 优化器训练 160K 次迭代，峰值学习率为 0.0002，预热 8K 次迭代，采用余弦学习率调度器。
*   每个批次随机采样 224 帧，来自一个随机训练场景。
*   输入帧、深度图和点图被调整大小，最大维度为 518 像素。宽高比在 0.33 到 1.0 之间随机化。
*   随机应用颜色抖动 (color jittering)、高斯模糊 (Gaussian blur) 和灰度增强 (grayscale augmentation)。
*   训练在 64 块 A100 GPU 上运行 9 天。
*   采用梯度范数裁剪 (gradient norm clipping) (阈值 1.0) 确保训练稳定性。
*   利用 `bfloat16` 精度和梯度检查点 (gradient checkpointing) 提高 GPU 内存和计算效率。

    <strong>训练数据 (Training Data)：</strong> 模型在大型多样化的数据集集合上进行训练，包括 `Co3Dv2` [88]、`BlendMVS` [146]、`DL3DV` [69]、`MegaDepth` [64]、`Kubric` [41]、`WildRGB` [135]、`ScanNet` [18]、`HyperSim` [89]、`Mapillary` [71]、`Habitat` [107]、`Replica` [104]、`MVS-Synth` [50]、`PointOdyssey` [159]、`Virtual KITTI` [7]、`Aria Synthetic Environments` [82]、`Aria Digital Twin` [82] 以及一个类似 `Objaverse` [20] 的艺术家创作资产合成数据集。这些数据集涵盖了室内外环境、合成和真实世界场景。3D 标注来源于直接传感器捕获、合成引擎或 `SfM` 技术 [95]。

# 5. 实验设置

## 5.1. 数据集
实验使用了多个数据集来评估 `VGGT` 在不同 3D 任务上的性能：

*   **Co3Dv2 [88]:** 用于相机姿态估计。这是一个 3D 重建数据集，包含从不同视角拍摄的物体集合。
*   **RealEstate10K [161]:** 用于相机姿态估计。这是一个大规模视频数据集，用于评估模型在未见 (unseen) 真实世界场景中的泛化能力。
*   **DTU [51]:** 用于多视图深度估计。这是一个经典的室内场景多视图数据集，提供高精度的深度图真值。
*   **ETH3D [97]:** 用于点图估计和消融研究。这是一个包含室内和室外场景的基准，提供高精度的 3D 几何真值。
*   **ScanNet-1500 [18, 92]:** 用于双视图匹配。这是一个大规模室内场景 RGB-D 数据集，常用于 3D 任务的评估。
*   **Image Matching Challenge (IMC) [54]:** 用于相机姿态估计，重点关注照片旅游 (phototourism) 数据。该基准包含具有挑战性的真实世界场景。
*   **GSO (Google Scanned Objects) [28]:** 用于新视角合成。这是一个包含高分辨率 3D 扫描物体的集合。
*   **TAP-Vid Benchmarks (Kinetics, RGB-S, DAVIS) [23]:** 用于动态点跟踪。这些基准包含具有挑战性的视频序列，用于评估模型在动态场景下跟踪点的能力。
*   **Kubric [41]:** 用于动态点跟踪任务的微调。这是一个用于生成合成视频的工具，提供丰富的 3D 真值。

    这些数据集涵盖了广泛的场景类型（室内、室外、合成、真实世界）、物体种类和任务难度，旨在全面验证 `VGGT` 的有效性、鲁棒性和泛化能力。

## 5.2. 评估指标
论文中使用的评估指标及其说明如下：

*   **AUC@30 (Area Under Curve at 30 degrees)**
    1.  <strong>概念定义 (Conceptual Definition):</strong> `AUC@T` 是相机姿态估计任务中常用的评估指标，用于衡量模型在不同误差阈值下的准确性。它通过计算相对旋转准确率 (Relative Rotation Accuracy, RRA) 和相对平移准确率 (Relative Translation Accuracy, RTA) 的最小值所形成曲线下的面积来综合评估相机姿态的精度。`RRA` 和 `RTA` 分别计算图像对之间的旋转和平移角误差，然后根据设定的阈值来判断准确性。`AUC@30` 特指当阈值最大为 30 度时的曲线下面积。值越高表示性能越好。
    2.  <strong>数学公式 (Mathematical Formula):</strong>
        $$
        \mathrm{AUC@T} = \int_0^T \min(\mathrm{Accuracy}_{\mathrm{RRA}}(\theta), \mathrm{Accuracy}_{\mathrm{RTA}}(\phi)) \, d\tau
        $$
        其中，本文中 $T=30$ 度。
    3.  <strong>符号解释 (Symbol Explanation):</strong>
        *   $\mathrm{AUC@T}$: 在阈值 $T$ 下的曲线下面积。
        *   $\mathrm{Accuracy}_{\mathrm{RRA}}(\theta)$: 相对旋转误差小于等于 $\theta$ 的图像对的比例。
        *   $\mathrm{Accuracy}_{\mathrm{RTA}}(\phi)$: 相对平移误差小于等于 $\phi$ 的图像对的比例。
        *   $\theta$: 旋转误差的角阈值。
        *   $\phi$: 平移误差的角阈值。
        *   $\tau$: 积分变量，表示误差阈值。

*   **Accuracy (Acc.), Completeness (Comp.), Overall (Chamfer distance)**
    1.  <strong>概念定义 (Conceptual Definition):</strong> 这些指标用于评估多视图深度估计和点图估计任务中预测几何的质量。
        *   <strong>Accuracy (精度):</strong> 衡量预测点云到真实标注点云的平均最近距离。它反映了预测点云与真实场景几何的贴合程度，值越低表示预测结果越精确。
        *   <strong>Completeness (完整度):</strong> 衡量真实标注点云到预测点云的平均最近距离。它反映了预测点云覆盖真实场景几何的程度，值越低表示预测结果越完整。
        *   <strong>Overall (总体):</strong> 是 `Accuracy` 和 `Completeness` 的平均值，也称为 `Chamfer distance` (倒角距离)。它是一个综合性的点云相似度度量，值越低表示预测点云与真实标注点云越接近。
    2.  <strong>数学公式 (Mathematical Formula):</strong> 假设 $P_{pred}$ 是预测点集，$P_{gt}$ 是真实标注 (Ground Truth) 点集。
        *   **Accuracy:**
            $$
            \mathrm{Accuracy}(P_{pred}, P_{gt}) = \frac{1}{|P_{pred}|} \sum_{x \in P_{pred}} \min_{y \in P_{gt}} \|x-y\|_2
            $$
        *   **Completeness:**
            $$
            \mathrm{Completeness}(P_{pred}, P_{gt}) = \frac{1}{|P_{gt}|} \sum_{y \in P_{gt}} \min_{x \in P_{pred}} \|x-y\|_2
            $$
        *   **Overall (Chamfer distance):**
            $$
            \mathrm{Overall}(P_{pred}, P_{gt}) = \frac{1}{2} \left( \mathrm{Accuracy}(P_{pred}, P_{gt}) + \mathrm{Completeness}(P_{pred}, P_{gt}) \right)
            $$
    3.  <strong>符号解释 (Symbol Explanation):</strong>
        *   $P_{pred}$: 预测点集。
        *   $P_{gt}$: 真实标注 (Ground Truth) 点集。
        *   $|P_{pred}|$: 预测点集中点的数量。
        *   $|P_{gt}|$: 真实标注 (Ground Truth) 点集中点的数量。
        *   $x$: 预测点集中的一个点。
        *   $y$: 真实标注 (Ground Truth) 点集中的一个点。
        *   $\|x-y\|_2$: 点 $x$ 和点 $y$ 之间的欧几里得距离。
        *   $\min_{y \in P_{gt}} \|x-y\|_2$: 从 $P_{gt}$ 中找到离 $x$ 最近的点的距离。
        *   $\min_{x \in P_{pred}} \|x-y\|_2$: 从 $P_{pred}$ 中找到离 $y$ 最近的点的距离。

*   **PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index Measure), LPIPS (Learned Perceptual Image Patch Similarity)**
    1.  <strong>概念定义 (Conceptual Definition):</strong> 这些指标用于评估新视角合成 (novel view synthesis) 任务中生成图像的质量。
        *   <strong>PSNR (峰值信噪比):</strong> 是一种基于像素差的客观质量评估指标，通过计算原始图像与生成图像之间的均方误差 (MSE) 来衡量重建图像的失真程度。PSNR 值越高，表示图像失真越小，生成图像质量越好。
        *   <strong>SSIM (结构相似性指数):</strong> 是一种更符合人类视觉感知的图像质量评估指标，它从亮度、对比度和结构三个方面衡量两幅图像的相似度。SSIM 值接近 1 表示两幅图像非常相似。
        *   <strong>LPIPS (感知图像块相似度):</strong> 利用预训练的深度神经网络提取图像特征，然后计算特征空间中的距离来衡量两幅图像之间的感知差异。LPIPS 值越低，表示两幅图像在人类感知上越相似。
    2.  <strong>数学公式 (Mathematical Formula):</strong>
        *   **PSNR:** 假设 $I$ 是原始图像，$K$ 是生成图像，$M \times N$ 是图像尺寸，$\mathrm{MAX}_I$ 是像素的最大可能值（例如，对于 8 位图像是 255）。
            $$
            \mathrm{MSE} = \frac{1}{MN} \sum_{i=0}^{M-1} \sum_{j=0}^{N-1} [I(i,j) - K(i,j)]^2
            $$
            $$
            \mathrm{PSNR} = 10 \cdot \log_{10} \left( \frac{\mathrm{MAX}_I^2}{\mathrm{MSE}} \right)
            $$
        *   **SSIM:** 假设 `x, y` 是两张图像的对应像素块，$μ_x, μ_y$ 是均值，$σ_x, σ_y$ 是标准差，$σ_{xy}$ 是协方差，$c_1, c_2$ 是用于稳定计算的常数。
            $$
            \mathrm{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}
            $$
        *   **LPIPS:** 假设 $x, x_0$ 是两张输入图像，$\phi_l$ 是在深度网络的第 $l$ 层提取的特征，$w_l$ 是在每个通道上调整特征的权重向量。
            $$
            \mathrm{LPIPS}(x, x_0) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} \|w_l \odot (\phi_l(x)_{h,w} - \phi_l(x_0)_{h,w})\|_2^2
            $$
    3.  <strong>符号解释 (Symbol Explanation):</strong>
        *   **PSNR:**
            *   `I(i,j)`: 原始图像在 `(i,j)` 处的像素值。
            *   `K(i,j)`: 生成图像在 `(i,j)` 处的像素值。
            *   `M, N`: 图像的宽和高。
            *   $\mathrm{MAX}_I$: 图像中可能的最大像素值。
        *   **SSIM:**
            *   $\mu_x, \mu_y$: 图像块 `x, y` 的平均值。
            *   $\sigma_x, \sigma_y$: 图像块 `x, y` 的标准差。
            *   $\sigma_{xy}$: 图像块 `x, y` 的协方差。
            *   $c_1 = (K_1 L)^2, c_2 = (K_2 L)^2$: 用于避免分母为零的小常数，其中 $L$ 是像素值的动态范围，$K_1, K_2 \ll 1$。
        *   **LPIPS:**
            *   $x, x_0$: 两张输入图像。
            *   $\phi_l$: 在深度网络的第 $l$ 层提取的特征。
            *   $w_l$: 在每个通道上调整特征的权重向量。
            *   $H_l, W_l$: 第 $l$ 层特征图的高度和宽度。
            *   $\odot$: 元素级乘法。
            *   $\|\cdot\|_2^2$: L2 范数的平方。

*   **Occlusion Accuracy (OA), $\delta_{\mathrm{avg}}^{\mathrm{vis}}$, Average Jaccard (AJ)**
    1.  <strong>概念定义 (Conceptual Definition):</strong> 这些指标用于评估动态点跟踪任务中点跟踪和可见性预测的准确性。
        *   <strong>Occlusion Accuracy (OA, 遮挡准确率):</strong> 衡量模型是否正确预测了跟踪点在给定帧中是可见还是被遮挡。
        *   <strong>$\delta_{\mathrm{avg}}^{\mathrm{vis}}$ (平均可见点跟踪准确率):</strong> 衡量在真实可见点中，有多少比例的点被模型准确跟踪到（即预测位置与真实标注位置之间的距离在一个预设像素阈值内）。值越高表示跟踪性能越好。
        *   <strong>Average Jaccard (AJ, 平均 Jaccard):</strong> 综合衡量跟踪和遮挡预测的准确性，通过计算每个跟踪点的 `Jaccard` 相似系数并在所有跟踪点和帧上取平均。它同时考虑了位置准确性和可见性预测的准确性。
    2.  <strong>数学公式 (Mathematical Formula):</strong>
        *   **Occlusion Accuracy (OA):**
            $$
            \mathrm{OA} = \frac{\text{正确预测可见/遮挡点的数量}}{\text{总点数}}
            $$
        *   **$\delta_{\mathrm{avg}}^{\mathrm{vis}}$:** 假设 $M$ 为跟踪点的总数，$N$ 为帧的总数，$v_{j,i}$ 为第 $j$ 个点在第 $i$ 帧的真实可见性（1为可见，0为遮挡），$\mathbf{y}_{j,i}$ 为真实位置，$\hat{\mathbf{y}}_{j,i}$ 为预测位置，$\tau$ 为像素阈值。
            $$
            \delta_{\mathrm{avg}}^{\mathrm{vis}} = \frac{1}{M} \sum_{j=1}^M \left( \frac{1}{\sum_{i=1}^N \mathbf{1}(v_{j,i})} \sum_{i=1}^N \mathbf{1}(v_{j,i} \land \| \mathbf{y}_{j,i} - \hat{\mathbf{y}}_{j,i} \|_2 < \tau) \right)
            $$
        *   **Average Jaccard (AJ):** 假设 `v'_{j,i}` 为第 $j$ 个点在第 $i$ 帧的预测可见性。
            $$
            \mathrm{AJ} = \frac{1}{M} \sum_{j=1}^M \frac{\sum_{i=1}^N \mathbf{1}(v_{j,i} \land \| \mathbf{y}_{j,i} - \hat{\mathbf{y}}_{j,i} \|_2 < \tau)}{\sum_{i=1}^N \mathbf{1}(v_{j,i} \lor (\|\mathbf{y}_{j,i} - \hat{\mathbf{y}}_{j,i}\|_2 < \tau \land v'_{j,i}))}
            $$
    3.  <strong>符号解释 (Symbol Explanation):</strong>
        *   $M$: 跟踪点的总数。
        *   $N$: 帧的总数。
        *   $v_{j,i}$: 真实标注 (Ground Truth) 中第 $j$ 个点在第 $i$ 帧的可见性（1表示可见，0表示遮挡）。
        *   $\mathbf{1}(\cdot)$: 指示函数，当括号内条件为真时为 1，否则为 0。
        *   $\mathbf{y}_{j,i}$: 真实标注 (Ground Truth) 中第 $j$ 个点在第 $i$ 帧的 2D 坐标。
        *   $\hat{\mathbf{y}}_{j,i}$: 预测的第 $j$ 个点在第 $i$ 帧的 2D 坐标。
        *   $\tau$: 像素阈值，用于判断跟踪是否准确。
        *   `v'_{j,i}`: 预测中第 $j$ 个点在第 $i$ 帧的可见性。

## 5.3. 对比基线
论文将 `VGGT` 与多项任务中的最先进方法进行了比较：

*   <strong>相机姿态估计 (Camera Pose Estimation):</strong>
    *   **传统/混合 SfM:** $Colmap+SPSG$ [92]、`PixSfM` [66]、`VGGSfM v2` [125]、`DFSfM (LoFTR)` [47]。
    *   **深度学习驱动方法:** `PoseDiff` [124]、`DUSt3R` [129]、`MASt3R` [62]、`MV-DUSt3R` [111]、`CUT3R` [127]、`FLARE` [156]、`Fast3R` [141]。

*   <strong>多视图深度估计 (Multi-view Depth Estimation):</strong>
    *   **已知真值相机的方法:** `Gipuma` [40]、`MVSNet` [144]、`CIDER` [139]、`PatchmatchNet` [121]、`MASt3R` [62]、`GeoMVSNet` [157]。
    *   **未知真值相机的方法:** `DUSt3R` [129]。

*   <strong>点图估计 (Point Map Estimation):</strong>
    *   `DUSt3R` [129]、`MASt3R` [62]。

*   <strong>图像匹配 (Image Matching):</strong>
    *   `SuperGlue` [92]、`LoFTR` [105]、`DKM` [32]、`CasMTR` [9]、`Roma` [33]。

*   <strong>新视角合成 (Novel View Synthesis):</strong>
    *   `LGM` [110]、`GS-LRM` [154]、`LVSM` [53]。

*   <strong>动态点跟踪 (Dynamic Point Tracking):</strong>
    *   `TAPTR` [63]、`LocoTrack` [13]、`BootsTAPIR` [26]、`CoTracker` [56]。

# 6. 实验结果与分析

## 6.1. 核心结果分析

### 6.1.1. 相机姿态估计 (Camera Pose Estimation)
在 `CO3Dv2` 和 `RealEstate10K` (未训练数据) 数据集上的实验结果（Table 1）表明，`VGGT` (前馈模式) 显著优于所有竞争方法。尤其是在 `RealEstate10K` 上，`VGGT` 的性能优势更为明显，验证了其卓越的泛化能力。与需要数秒甚至数十秒进行后优化的方法（如 `DUSt3R`、`MASt3R`、`VGGSfM`）相比，`VGGT` 仅需 0.2 秒即可完成重建。
同时，当 `VGGT` 与 `Bundle Adjustment` (BA) 结合时 (`Ours (with BA)`)，性能进一步提升，达到了 93.5 的 `AUC@30`，超越了所有基线，显示出 `VGGT` 提供的预测可以作为 `BA` 的良好初始化，大大加速了优化过程。

以下是原文 Table 1 的结果：

<table><tr><td>Methods</td><td>Re10K (unseen) AUC@30 ↑</td><td>CO3Dv2 AUC@30 ↑</td><td>Time</td></tr><tr><td>Colmap+SPSG [92] PixSfM [66]</td><td>45.2 49.4</td><td>25.3 30.1</td><td>∼ 15s &gt; 20s</td></tr><tr><td>PoseDiff [124]</td><td>48.0</td><td>66.5</td><td>∼ 7s</td></tr><tr><td>DUSt3R [129] MASt3R [62]</td><td>67.7 76.4</td><td>76.7 81.8</td><td>∼ 7s ∼ 9s</td></tr><tr><td>VGGSfM v2 [125]</td><td>78.9</td><td>83.4</td><td>∼ 10s</td></tr><tr><td>MV-DUSt3R [111] ‡</td><td>71.3</td><td>69.5</td><td>∼ 0.6s</td></tr><tr><td>CUT3R [127]</td><td>75.3</td><td>82.8</td><td>∼ 0.6s</td></tr><tr><td>FLARE [156] ‡</td><td>78.8</td><td>83.3</td><td>∼ 0.5s</td></tr><tr><td>Fast3R [141] ‡</td><td>72.7</td><td>82.5</td><td>∼ 0.2s</td></tr><tr><td>Ours (Feed-Forward)</td><td>85.3</td><td>88.2</td><td>∼ 0.2s</td></tr><tr><td>Ours (with BA)</td><td>93.5</td><td>91.8</td><td>∼ 1.8s</td></tr></table>

在 `Image Matching Challenge` (IMC) [54] 数据集上的结果（Table 10）进一步证实了 `VGGT` 的强大。`VGGT` (前馈模式) 的性能与最先进的 `VGGSfMv2` 相当，但在速度上快得多 (0.2s vs. 10s)。当结合 `BA` 时 (`VGGT + BA`)，模型在 `IMC` 上实现了最先进的性能，将 `AUC@10` 从 71.26 提高到 84.91，`AUC@3` 从 39.23 提高到 66.37。这表明 `VGGT` 提供的 3D 点可以直接作为 `BA` 的初始化，省去了三角测量和迭代细化的需要，使得整个过程更快。

以下是原文 Table 10 的结果：

<table><thead><tr><th>Method</th><th>Test-time Opt.</th><th>AUC@3</th><th>AUC@5</th><th>AUC@10</th><th>Runtime</th></tr></thead><tbody><tr><td>COLMAP (SIFT+NN) [94]</td><td>✓</td><td>23.58</td><td>32.66</td><td>44.79</td><td>&gt;10s</td></tr><tr><td>PixSfM (SIFT + NN) [66]</td><td>✓</td><td>25.54</td><td>34.80</td><td>46.73</td><td>&gt;20s</td></tr><tr><td>PixSfM (LoFTR) [66]</td><td>✓</td><td>44.06</td><td>56.16</td><td>69.61</td><td>&gt;20s</td></tr><tr><td>PixSfM (SP + SG) [66]</td><td>✓</td><td>45.19</td><td>57.22</td><td>70.47</td><td>&gt;20s</td></tr><tr><td>DFSfM (LoFTR) [47]</td><td>✓</td><td>46.55</td><td>58.74</td><td>72.19</td><td>&gt;10s</td></tr><tr><td>DUSt3R [129]</td><td>✓</td><td>13.46</td><td>21.24</td><td>35.62</td><td>∼ 7s</td></tr><tr><td>MASt3R [62]</td><td>✓</td><td>30.25</td><td>46.79</td><td>57.42</td><td>∼ 9s</td></tr><tr><td>VGGSfM [125]</td><td>✓</td><td>45.23</td><td>58.89</td><td>73.92</td><td>∼ 6s</td></tr><tr><td>VGGSfMv2 [125]</td><td>✓</td><td>59.32</td><td>67.78</td><td>76.82</td><td>∼ 10s</td></tr><tr><td>VGGT (ours)</td><td>X</td><td>39.23</td><td>52.74</td><td>71.26</td><td>0.2s</td></tr><tr><td>VGGT + BA (ours)</td><td>✓</td><td>66.37</td><td>75.16</td><td>84.91</td><td>1.8s</td></tr></tbody></table>

### 6.1.2. 多视图深度估计 (Multi-view Depth Estimation)
在 `DTU` 数据集上的结果（Table 2）显示，`VGGT` 在未知真实标注 (Ground Truth) 相机参数的情况下，大幅超越了 `DUSt3R` (将 `Overall` 分数从 1.741 降至 0.382)。更重要的是，`VGGT` 的性能与那些在测试时已知真实标注 (Ground Truth) 相机参数的方法（如 `MASt3R`、`GeoMVSNet` 等）相媲美。这归因于 `VGGT` 的多图像训练方案，使其能够原生推理多视图三角测量，而不是依赖于临时对齐过程。

以下是原文 Table 2 的结果：

<table><tr><td>Known GT camera</td><td>Method</td><td>Acc.↓</td><td>Comp.↓</td><td>Overall↓</td></tr><tr><td>:</td><td>Gipuma [40]</td><td>0.283</td><td>0.873</td><td>0.578</td></tr><tr><td></td><td>MVSNet [144]</td><td>0.396</td><td>0.527</td><td>0.462</td></tr><tr><td>✓</td><td>CIDER [139]</td><td>0.417</td><td>0.437</td><td>0.427</td></tr><tr><td>✓</td><td>PatchmatchNet [121]</td><td>0.427</td><td>0.377</td><td>0.417</td></tr><tr><td>✓</td><td>MASt3R [62]</td><td>0.403</td><td>0.344</td><td>0.374</td></tr><tr><td>✓</td><td>GeoMVSNet [157]</td><td>0.331</td><td>0.259</td><td>0.295</td></tr><tr><td>X</td><td>DUSt3R [129]</td><td>2.677</td><td>0.805</td><td>1.741</td></tr><tr><td>X</td><td>Ours</td><td>0.389</td><td>0.374</td><td>0.382</td></tr></table>

### 6.1.3. 点图估计 (Point Map Estimation)
在 `ETH3D` 数据集上的点图估计结果（Table 3）显示，即使 `DUSt3R` 和 `MASt3R` 进行了耗时的优化（全局对齐，约 10 秒/场景），`VGGT` (前馈模式) 仍以 0.2 秒/重建的速度显著优于它们。值得注意的是，通过将预测的深度图和相机参数进行反投影 (unprojecting) 来构建 3D 点云 (`Ours (Depth + Cam)`) 相比直接使用点图头 (`Ours (Point)`) 获得了更高的准确性。这表明将复杂任务分解为更简单的子问题（深度图和相机预测），即使这些子问题在训练时是联合监督的，也能带来性能上的优势。

以下是原文 Table 3 的结果：

<table><tr><td>Methods</td><td>Acc.↓</td><td>Comp.↓</td><td>Overall↓</td><td>Time</td></tr><tr><td>DUSt3R</td><td>1.167</td><td>0.842</td><td>1.005</td><td>~ 7s</td></tr><tr><td>MASt3R</td><td>0.968</td><td>0.684</td><td>0.826</td><td>∼ 9s</td></tr><tr><td>Ours (Point)</td><td>0.901</td><td>0.518</td><td>0.709</td><td>∼ 0.2s</td></tr><tr><td>Ours (Depth + Cam)</td><td>0.873</td><td>0.482</td><td>0.677</td><td>∼ 0.2s</td></tr></table>

下图（原文 Figure 4）展示了 `VGGT` 模型在多个场景中的应用效果，包含多个视图的 3D 重建结果，如斗兽场的重建，显示深度图和点云效果。该研究提供了比传统方法更高效的图像重建过程。

![该图像是插图，展示了 VGGT 模型在多个场景中的应用效果，包含多个视图的 3D 重建结果，如斗兽场的重建，显示深度图和点云效果。该研究提供了比传统方法更高效的图像重建过程。](images/4.jpg)
*该图像是插图，展示了 VGGT 模型在多个场景中的应用效果，包含多个视图的 3D 重建结果，如斗兽场的重建，显示深度图和点云效果。该研究提供了比传统方法更高效的图像重建过程。*

### 6.1.4. 图像匹配 (Image Matching)
在 `ScanNet-1500` 数据集上的双视图匹配结果（Table 4）显示，尽管 `VGGT` 的跟踪头并非专门为双视图匹配设计，它仍然在所有基线中取得了最高的 `AUC` 准确率，甚至超越了最先进的双视图匹配方法 `Roma`。这凸显了 `VGGT` 学习到的特征的通用性和强大性。

以下是原文 Table 4 的结果：

<table><tr><td>Method</td><td>AUC@5↑</td><td>AUC@10 ↑</td><td>AUC@20 ↑</td></tr><tr><td>SuperGlue [92]</td><td>16.2</td><td>33.8</td><td>51.8</td></tr><tr><td>LoFTR [105]</td><td>22.1</td><td>40.8</td><td>57.6</td></tr><tr><td>DKM [32]</td><td>29.4</td><td>50.7</td><td>68.3</td></tr><tr><td>CasMTR [9]</td><td>27.1</td><td>47.0</td><td>64.4</td></tr><tr><td>Roma a [33]</td><td>31.8</td><td>53.4</td><td>70.9</td></tr><tr><td>Ours</td><td>33.9</td><td>55.2</td><td>73.4</td></tr></table>

### 6.1.5. 新视角合成 (Novel View Synthesis)
在 `GSO` 数据集上的新视角合成结果（Table 7）表明，`VGGT` 的特征提取器在进行微调后，即使不需要输入图像的相机参数，并且使用更少的训练数据（仅 `Objaverse` 的 20%），也能达到与 `LVSM` 等方法竞争的性能。这再次强调了 `VGGT` 学习特征的有效性和泛化能力。

以下是原文 Table 7 的结果：

<table><tr><td>Method</td><td>Known Input Cam</td><td>Size</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>LGM [110]</td><td></td><td>256</td><td>21.44</td><td>0.832</td><td>0.122</td></tr><tr><td>GS-LRM [154]</td><td>:</td><td>256</td><td>29.59</td><td>0.944</td><td>0.051</td></tr><tr><td>LVSM [53]</td><td>✓</td><td>256</td><td>31.71</td><td>0.957</td><td>0.027</td></tr><tr><td>Ours-NVS*</td><td>×</td><td>224</td><td>30.41</td><td>0.949</td><td>0.033</td></tr></table>

下图（原文 Figure 6）展示了生成模型的效果。上排为输入图像，中排为真实图像，下排为模型预测的图像，呈现了从简单形状到企鹅的转换过程。

![Figure 6. Qualitative Examples of Novel View Synthesis. The top row shows the input images, the middle row displays the ground truth images from target viewpoints, and the bottom row presents our syn…](images/6.jpg)
*该图像是一个插图，展示了生成模型的效果。上排为输入图像，中排为真实图像，下排为模型预测的图像，呈现了从简单形状到企鹅的转换过程。*

### 6.1.6. 动态点跟踪 (Dynamic Point Tracking)
在 `TAP-Vid` 基准上的结果（Table 8）显示，将 `CoTracker` 的主干网络替换为 `VGGT` 预训练的特征主干网络后，`CoTracker` 的性能显著增强。例如，在 `TAP-Vid RGB-S` 数据集上，`VGGT` 的跟踪特征将 $\delta_{\mathrm{avg}}^{\mathrm{vis}}$ 提高了 5%。这表明 `VGGT` 的特征具有强大的泛化能力，即使在动态场景和非专门设计的情况下也能保持出色的性能。

以下是原文 Table 8 的结果：

<table><tr><td rowspan="2">Method</td><td colspan="2">Kinetics</td><td colspan="2">RGB-S</td><td colspan="2">DAVIS</td></tr><tr><td>δvis AJ</td><td>OA</td><td>AJ</td><td>Ovig OA</td><td>AJ</td><td>δvis OA</td></tr><tr><td>TAPTR [63]</td><td>49.0 64.4</td><td>85.2</td><td>60.8</td><td>76.2 87.0</td><td>63.0 76.1</td><td>91.1</td></tr><tr><td>LocoTrack [13]</td><td>52.9 66.8 85.3</td><td></td><td>69.7</td><td>83.2 89.5</td><td>62.9</td><td>75.3 87.2</td></tr><tr><td>BootsTAPIR [26]</td><td>54.6 68.4</td><td>86.5</td><td>70.8</td><td>83.0 89.9</td><td></td><td>61.4 73.6 88.7</td></tr><tr><td>CoTracker [56]</td><td>49.6 64.3 83.3 67.4 78.9</td><td></td><td></td><td>85.2</td><td></td><td>61.8 76.1 88.3</td></tr><tr><td>CoTracker + Ours 57.2 69.0 88.9 72.1 84.0</td><td></td><td></td><td></td><td></td><td></td><td>91.6 64.7 77.5 91.4</td></tr></table>

下图（原文 Figure 5）展示了 `VGGT` 方法与 `CoTracker` 的刚性和动态点追踪效果。上部分为 `VGGT` 的追踪模块输出，显示了多个关键点轨迹；下部分为 `CoTracker` 的处理效果，对比展示了两种方法的异同。

![Figure 5. Visualization of Rigid and Dynamic Point Tracking. Top: VGGT's tracking module $\\tau$ outputs keypoint tracks for an CoTracker \[56\], which processes sequential inputs.](images/5.jpg)
*该图像是一个示意图，展示了 VGGT 方法与 CoTracker 的刚性和动态点追踪效果。上部分为 VGGT 的追踪模块输出，显示了多个关键点轨迹；下部分为 CoTracker 的处理效果，对比展示了两种方法的异同。*

## 6.2. 消融实验/参数分析

### 6.2.1. 特征主干网络 (Feature Backbone)
为了验证所提出的 `Alternating-Attention` (AA) 设计的有效性，作者将其与两种替代注意力架构进行了比较：(a) 仅使用全局自注意力 (`Global Self-Attention Only`)，和 (b) 使用交叉注意力 (`Cross-Attention`)。所有模型变体保持相同的参数数量和 `2L` 注意力层。

以下是原文 Table 5 的结果：

<table><tr><td>ETH3D Dataset</td><td>Acc.↓</td><td>Comp.↓</td><td>Overall↓</td></tr><tr><td>Cross-Attention</td><td>1.287</td><td>0.835</td><td>1.061</td></tr><tr><td>Global Self-Attention Only</td><td>1.032</td><td>0.621</td><td>0.827</td></tr><tr><td>Alternating-Attention</td><td>0.901</td><td>0.518</td><td>0.709</td></tr></table>

结果（Table 5）表明，`Alternating-Attention` 架构在点图估计精度上明显优于这两种基线变体。这证明了 `AA` 在整合帧内和跨帧信息方面的有效性。此外，初步实验还显示，使用交叉注意力的架构通常不如仅使用自注意力的架构。

### 6.2.2. 多任务学习 (Multi-task Learning)
作者还验证了训练一个单一网络同时学习多个 3D 量的益处，即使这些输出可能存在重叠关系（例如，深度图和相机参数可以共同生成点图）。

以下是原文 Table 6 的结果：

<table><tr><td>w. Lcamera</td><td>W. Ldepth</td><td>W. Ltrack</td><td>Acc.↓</td><td>Comp.↓</td><td>Overall↓</td></tr><tr><td>×</td><td>✓</td><td>:</td><td>1.042</td><td>0.627</td><td>0.834</td></tr><tr><td></td><td>×</td><td></td><td>0.920</td><td>0.534</td><td>0.727</td></tr><tr><td>:</td><td>✓</td><td>×</td><td>0.976</td><td>0.603</td><td>0.790</td></tr><tr><td></td><td>✓</td><td>✓</td><td>0.901</td><td>0.518</td><td>0.709</td></tr></table>

结果（Table 6）显示，当训练过程中不包含相机、深度或跟踪损失时，点图估计的准确性会显著下降。特别是，引入相机参数估计 (w. `Lcamera`) 显著增强了点图的准确性，而深度估计 (w. `Ldepth`) 则贡献了边际改进。这支持了多任务学习范式，即使预测量存在相互依赖关系，联合监督也能提高整体性能。

### 6.2.3. 运行时和内存 (Runtime and Memory)
以下是原文 Table 9 的结果：

<table><tr><td>Input Frames</td><td>1</td><td>2</td><td>4</td><td>8</td><td>10</td><td>20</td><td>50</td><td>100</td><td>200</td></tr><tr><td>Time (s)</td><td>0.04</td><td>0.05</td><td>0.07</td><td>0.11</td><td>0.14</td><td>0.31</td><td>1.04</td><td>3.12</td><td>8.75</td></tr><tr><td>Mem. (GB)</td><td>1.88</td><td>2.07</td><td>2.45</td><td>3.23</td><td>3.63</td><td>5.58</td><td>11.41</td><td>21.15</td><td>40.63</td></tr></table>

表 9 评估了特征主干网络在处理不同数量输入帧时的推理运行时和峰值 GPU 内存使用情况。结果表明，`VGGT` 在单帧处理时非常高效（0.04秒，1.88GB），并且随着帧数增加，时间和内存消耗呈线性增长趋势。处理 200 帧仍能在 8.75 秒内完成，且内存使用为 40.63 GB。这证明了 `VGGT` 的高效性和可扩展性。

## 6.3. 定性示例

下图（原文 Figure 3）展示了 `VGGT` 在单视图、双视图和多视图场景下的重建效果。左侧为输入图像，中央为 `VGGT` 的输出，右侧为 `DUST3R` 的输出，响应时间均小于 0.1 秒，显示出 `VGGT` 的高效性能。

![该图像是图表，展示了VGGT在单视图、双视图和多视图场景下的重建效果。左侧为输入图像，中央为VGGT的输出，右侧为DUST3R的输出，响应时间均小于0.1秒，显示出VGGT的高效性能。](images/3.jpg)
*该图像是图表，展示了VGGT在单视图、双视图和多视图场景下的重建效果。左侧为输入图像，中央为VGGT的输出，右侧为DUST3R的输出，响应时间均小于0.1秒，显示出VGGT的高效性能。*

下图（原文 Figure 7）展示了多种场景的视觉表现，包括艺术展示、物体示例和交通状况。左侧的图像包括多个水壶和瓶子的艺术作品，而右侧则展示了交通繁忙的道路。整体图像展示了不同的视觉内容和风格。

![该图像是一个示意图，展示了多种场景的视觉表现，包括艺术展示、物体示例和交通状况。左侧的图像包括多个水壶和瓶子的艺术作品，而右侧则展示了交通繁忙的道路。整体图像展示了不同的视觉内容和风格。](images/7.jpg)
*该图像是一个示意图，展示了多种场景的视觉表现，包括艺术展示、物体示例和交通状况。左侧的图像包括多个水壶和瓶子的艺术作品，而右侧则展示了交通繁忙的道路。整体图像展示了不同的视觉内容和风格。*

# 7. 总结与思考

## 7.1. 结论总结
本文提出了 `VGGT` (Visual Geometry Grounded Transformer)，一个在 3D 计算机视觉领域具有里程碑意义的前馈神经网络。`VGGT` 能够直接从多视图输入中推断出场景的所有关键 3D 属性，包括相机参数、点图、深度图和 3D 点轨迹。它克服了传统方法对迭代优化和后处理的依赖，以及现有深度学习方法在多任务和多视图处理上的局限性。

`VGGT` 的核心贡献在于：
1.  **统一且高效的多任务 3D 属性预测：** 通过一个大型 `Transformer` 架构和 `Alternating-Attention` 机制，实现了从单视图到数百视图的高效、端到端 3D 重建。
2.  **卓越的性能：** 在相机姿态估计、多视图深度估计、密集点云重建和 3D 点跟踪等多个任务上取得了最先进的 (state-of-the-art) 结果，并且其前馈预测通常优于甚至超越了需要耗时优化步骤的竞争方法。
3.  **强大的泛化能力和作为特征主干网络的潜力：** 预训练的 `VGGT` 特征可以显著增强下游任务（如新视角合成和动态点跟踪）的性能，证明了其通用性和作为基础模型 (foundation model) 的价值。

    `VGGT` 的简单性和效率使其非常适用于实时应用，标志着 3D 计算机视觉从以几何为中心向以神经网络为中心范式转变的重要一步。

## 7.2. 局限性与未来工作
论文作者指出了 `VGGT` 的几个局限性：
*   **图像类型限制：** 当前模型不支持鱼眼或全景图像。
*   **极端姿态：** 在涉及极端输入旋转的条件下，重建性能会下降。
*   **非刚性运动：** 尽管模型可以处理轻微的非刚性运动，但在涉及大量非刚性变形的场景中会失效。

    作者认为，解决这些局限性可以通过在特定数据集上进行微调，且只需进行最小的架构修改即可实现，这体现了 `VGGT` 的灵活性和易于适应性。

未来可能的研究方向包括：
*   **特定数据集微调：** 通过在包含鱼眼/全景图像、极端旋转或非刚性运动的数据集上进行微调，扩展模型的能力。
*   <strong>可微分 `Bundle Adjustment` (BA)：</strong> 探索结合可微分 `BA` 进行大规模无监督训练，以利用 3D 标注缺失场景中的有效监督信号。尽管目前计算成本较高，但这仍是一个有前景的方向。
*   **部署优化：** 进一步优化 `Transformer` 的内存和运行时效率，例如采用 `Tensor Parallelism` 等 `LLM` 部署技术来加速多 GPU 推理。

## 7.3. 个人启发与批判
### 7.3.1. 个人启发
`VGGT` 的研究为我带来了以下几点启发：

*   **`Transformer` 在 3D 领域的巨大潜力：** 论文再次证明了 `Transformer` 架构的强大和通用性。即使在 3D 这种通常被认为需要强几何归纳偏置的领域，一个设计合理的 `Transformer` 也能在大量数据驱动下学习复杂的几何关系，并超越传统方法。这鼓励我们大胆探索 `Transformer` 在更多非传统领域的应用。
*   **多任务学习的协同效应：** `VGGT` 明确通过多任务学习同时预测多个相互关联的 3D 属性，并证明了这种联合训练即使对于存在闭式关系的量也能带来性能提升。这提示我们在设计模型时，可以考虑让模型同时学习多个相关任务，以利用任务间的隐式协同作用。
*   **前馈式解决方案的价值：** `VGGT` 强调前馈推理，显著提高了效率，使其适用于实时应用。在很多实际场景中，低延迟的系统是关键。`VGGT` 的成功表明，通过精心设计和大规模训练，纯前馈模型可以达到甚至超越迭代优化方法的性能。
*   <strong>基础模型 (Foundation Model) 思维在 3D 领域的应用：</strong> `VGGT` 作为预训练的特征主干网络，能够显著增强下游任务。这与 `GPT`、`CLIP` 等在 NLP 和 2D 视觉领域的基础模型思路一致。未来，在 3D 领域构建一个强大的、通用的基础模型，将极大地推动后续研究和应用。
*   **数据驱动的重要性：** `VGGT` 能够取得成功，离不开其在大量多样化的 3D 标注数据集上的训练。这再次强调了高质量、大规模数据在深度学习中的核心作用，尤其是在复杂感知任务中。

### 7.3.2. 批判与潜在问题
尽管 `VGGT` 取得了令人印象深刻的成果，但在我看来，仍存在一些潜在的问题或可以改进的地方：

*   <strong>“最小 3D 归纳偏置”</strong>的定义： 论文声称 `VGGT` 的设计具有“最小 3D 归纳偏置”，除了 `Alternating-Attention` 机制。然而，`Alternating-Attention` 本身就可以被视为一种特定的 3D 归纳偏置，因为它明确地将图像分为“帧内”和“全局”两种处理方式，这在多视图几何中是有意义的。如果完全无偏置，可能只是一个标准的 `ViT` 处理所有图像词元，而 `AA` 显然考虑了多视图输入的结构。
*   **对大规模数据的依赖：** `VGGT` 的成功高度依赖于大规模、多样化的 3D 标注数据集。对于那些难以获取大量高质量 3D 真值 (Ground Truth) 的特定或小众场景，这种方法的适用性可能会受限。数据标注成本可能成为瓶颈。
*   **非刚性运动的限制：** 模型目前难以处理大幅非刚性变形的场景。虽然作者提到可以通过微调解决，但这可能需要专门的架构修改或损失函数设计，而不仅仅是数据微调。在动态、可变形的世界中，这是一个重要的限制。
*   **计算资源的消耗：** 尽管前馈推理速度快，但训练过程消耗了 64 块 A100 GPU 长达 9 天，总参数量达 12 亿。这表明模型对计算资源的需求极高，对于资源有限的研究者或企业来说，其可复现性和可及性仍然是一个挑战。
*   **`Differentiable BA` 的权衡：** 论文提及 `differentiable BA` 在小规模实验中表现有前景，但因计算成本未被纳入最终模型。这提出一个疑问：如果能优化 `differentiable BA` 的效率，`VGGT` 是否能进一步减少对大规模监督数据的依赖，通过无监督或自监督的方式学习更强的几何一致性？这可能是模型进一步提升鲁棒性和泛化能力的关键。
*   **预测不确定性图的利用：** 模型预测了不确定性图，但在论文的评估中，这些不确定性图似乎主要用于损失函数，而非在推理阶段主动用于结果的后处理或置信度量化。未来可以探索如何更好地利用这些不确定性图来提高模型在实际应用中的决策能力。
*   **`Alternating-Attention` 的理论分析：** 虽然消融实验证明了 `Alternating-Attention` 的有效性，但其内在机理，即如何在信息聚合和归一化之间找到最佳平衡，以及其数学上的性质，可以进行更深入的理论分析。