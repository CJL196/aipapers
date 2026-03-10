# 1. 论文基本信息

## 1.1. 标题
论文标题为 **MegaSaM: Accurate, Fast, and Robust Structure and Motion from Casual Dynamic Videos**（MegaSaM：来自随意动态视频的准确、快速且鲁棒的结构与运动恢复）。该标题清晰地表明了系统的核心目标：从非专业拍摄的、包含动态物体的单目视频中，高精度地估计相机参数和场景三维结构。

## 1.2. 作者
论文作者包括 Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, 和 Noah Snavely。
*   **隶属机构：** 作者主要来自 **Google DeepMind**，部分作者来自 **UC Berkeley**（加州大学伯克利分校）和 **University of Michigan**（密歇根大学）。
*   **背景：** 这些作者多在计算机视觉、三维重建和深度学习领域有深厚背景，尤其是 Noah Snavely 和 Angjoo Kanazawa 在结构恢复结构（SfM）和神经渲染领域享有盛誉。

## 1.3. 发表期刊/会议
*   **发布状态：** 该论文目前为 <strong>预印本 (Preprint)</strong>，发布于 arXiv。
*   **发布时间：** 2024 年 12 月 5 日 (UTC)。
*   **影响力：** 虽然尚未正式发表在某特定顶会（如 CVPR, ICCV 等）的论文集上，但 arXiv 预印本在计算机科学领域具有极高的流通性和影响力，尤其是来自 Google DeepMind 等顶级实验室的工作，通常代表了该领域的最新前沿进展。

## 1.4. 摘要
论文提出了一种系统，能够从随意拍摄的单目动态视频中准确、快速且鲁棒地估计相机参数和深度图。传统的运动恢复结构（SfM）和单目 SLAM 技术通常假设场景是静态的且具有较大的相机视差，因此在缺乏这些条件时容易产生错误估计。近期的神经网络方法虽然试图克服这些挑战，但往往计算成本高或在动态视频中表现脆弱。本文展示了一种深度视觉 SLAM 框架的有效性：通过对训练和推理方案的仔细修改，该系统可以扩展到具有 unconstrained 相机路径的复杂动态场景真实世界视频，包括视差较小的视频。在合成和真实视频上的大量实验表明，与 prior 和 concurrent 工作相比，该系统在相机姿态和深度估计上显著更准确和鲁棒，且运行时间更快或相当。

## 1.5. 原文链接
*   **arXiv 链接：** https://arxiv.org/abs/2412.04463
*   **PDF 链接：** https://arxiv.org/pdf/2412.04463v2
*   **项目主页：** https://mega-sam.github.io/

# 2. 整体概括

## 2.1. 研究背景与动机
*   **核心问题：** 从图像集中提取相机参数和场景几何结构是计算机视觉中的基本问题，通常称为运动恢复结构（SfM）或同步定位与建图（SLAM）。然而，现有成熟算法主要针对具有大相机基线的静止场景。当应用于“随意单目视频”（casual monocular videos）时，这些方法往往会失败。
*   <strong>挑战与空白 (Gap)：</strong>
    1.  **动态场景：** 随意视频通常包含移动物体和场景动态，传统 SfM/SLAM 假设场景静态，动态物体会破坏几何约束。
    2.  **有限的相机视差：** 手持相机拍摄的视频往往相机运动有限（如近乎静止或纯旋转），导致几何结构难以观测（unobservable）。
    3.  **计算成本与鲁棒性：** 现有的神经网络方法要么计算昂贵（如需要测试时微调），要么在相机运动不受控或视场未知时表现脆弱。
*   **切入点：** 作者重新审视并扩展了先前的深度视觉 SLAM 框架（特别是 DROID-SLAM），通过引入单目深度先验、运动概率图和不确定性感知的全局光束法平差（BA），使其能够处理动态和视差受限的视频。

## 2.2. 核心贡献/主要发现
*   **MegaSaM 系统：** 提出了一套完整的流水线，能够从随意单目动态视频中实现准确、快速和鲁棒的相机跟踪和深度估计。
*   **深度视觉 SLAM 的扩展：** 证明了通过对训练和推理方案的修改，深度视觉 SLAM 框架可以有效处理动态视频。关键创新包括将单目深度先验和运动概率图集成到可微分的 SLAM 范式中。
*   **不确定性感知的全局 BA：** 分析了视频结构和相机参数的可观测性，引入了不确定性感知的全局光束法平差方案，在相机参数约束不足时提高了系统鲁棒性。
*   **一致视频深度优化：** 展示了如何在不需要测试时网络微调的情况下，准确且高效地获得一致的视频深度。
*   **性能优势：** 在合成和真实数据集上的实验表明，MegaSaM 在相机姿态和深度估计的准确性及鲁棒性上显著优于 prior 和 concurrent 基线，且运行速度具有竞争力。

    下图（原文 Figure 1）展示了 MegaSaM 系统的输入输出效果，直观体现了其从随意视频中恢复三维结构的能力：

    ![Figure 1. MegaSaM enables accurate, fast and robust estimation of cameras and scene structure from a casually captured monocular video of a dynamic scene. Top: input video frames (every tenth frame shown). Bottom: our estimated camera and 3D point clouds unprojected by predicted video depths without any postprocessing.](images/1.jpg)
    *该图像是插图，展示了MegaSaM系统如何从动态场景的视频中进行镜头及场景结构的估计。上方为输入视频的每十帧，底部为通过预测的深度无后处理的估计相机和3D点云。*

# 3. 预备知识与相关工作

## 3.1. 基础概念
为了理解本文，读者需要掌握以下基本概念：
*   <strong>运动恢复结构 (Structure from Motion, SfM)：</strong> 一种从二维图像序列中恢复三维场景结构和相机运动的技术。它通常通过特征匹配和光束法平差来实现。
*   <strong>同步定位与建图 (Simultaneous Localization and Mapping, SLAM)：</strong> 指移动设备（如机器人、相机）在未知环境中运动时，同时进行自身定位和环境地图构建的技术。视觉 SLAM (Visual SLAM) 特指使用相机作为传感器。
*   <strong>光束法平差 (Bundle Adjustment, BA)：</strong> SfM 和 SLAM 中的核心优化步骤。它通过最小化重投影误差（reprojection error）来联合优化相机姿态和三维点坐标。
*   <strong>视差 (Parallax)：</strong> 由于相机位置变化，同一物体在图像中位置的差异。足够的视差是恢复深度信息的关键；如果相机只旋转不平移，视差为零，深度无法恢复。
*   <strong>单目深度估计 (Monocular Depth Estimation)：</strong> 从单张 RGB 图像预测深度图的技术。通常存在尺度模糊（scale ambiguity），即只能预测相对深度。
*   <strong>可微分光束法平差 (Differentiable Bundle Adjustment)：</strong> 将 BA 优化过程嵌入神经网络中，使其可以通过反向传播进行端到端训练。

## 3.2. 前人工作
*   **视觉 SLAM 和 SfM：** 传统方法通过特征匹配或光度对齐估计对应关系，然后通过 BA 优化。深度视觉 SLAM（如 DROID-SLAM）使用神经网络估计对应关系和不确定性，并通过可微分 BA 层更新状态。但它们通常假设静态场景和足够的相机基线。
*   **动态场景处理：** 近期工作如 Robust-CVD 和 CasualSAM 通过优化空间变化样条或微调单目深度网络来联合估计相机和深度。Particle-SfM 和 LEAP-VO 通过长时轨迹推断运动掩码，在 BA 中降低动态特征权重。
*   **单目深度：** 单图像深度模型（如 DepthAnything）在单图上泛化性强，但在视频中存在时间不一致性。现有方法通过测试时优化或直接预测视频深度来解决，但往往计算昂贵。
*   **动态场景重建：** 一些工作使用时变辐射场（如 NeRF）进行动态重建，但它们通常需要相机参数或视频深度作为输入，而 MegaSaM 的输出可作为这些系统的输入。

## 3.3. 技术演进
该领域的技术演进从传统的几何方法（特征匹配 + BA）转向了深度学习方法（学习对应关系 + 可微分 BA）。早期的深度 SLAM 主要关注静态场景。随后的工作开始尝试处理动态场景，但往往牺牲了速度或鲁棒性。MegaSaM 处于这一演进的前沿，它结合了深度 SLAM 的效率和对动态场景的适应能力，通过引入外部先验（单目深度）和内部学习（运动概率）来解决长期存在的挑战。

## 3.4. 差异化分析
*   **与 DROID-SLAM 相比：** MegaSaM 扩展了 DROID-SLAM，增加了运动概率预测和单目深度初始化，使其能处理动态和弱视差场景。
*   **与 CasualSAM 相比：** MegaSaM 不需要耗时的单目深度网络微调，且通过固定相机参数进行深度优化，速度更快。
*   **与 MonST3R 相比：** 虽然 MonST3R 也处理动态场景，但 MegaSaM 在相机跟踪的鲁棒性和准确性上表现更好，尤其是在相机运动受限的情况下。

# 4. 方法论

## 4.1. 方法原理
MegaSaM 的核心思想是将深度视觉 SLAM 框架扩展以适应动态和随意拍摄的视频。其基本原理是利用可微分的光束法平差（BA）层迭代更新场景几何和相机姿态，同时引入单目深度先验来初始化视差，并学习运动概率图来降低动态物体在优化中的权重。此外，通过不确定性分析来决定何时引入深度正则化，以解决弱视差下的退化问题。

## 4.2. 核心方法详解

### 4.2.1. 深度视觉 SLAM 公式化
深度视觉 SLAM 系统（如 DROID-SLAM）的特点是采用可微分的、学习到的 BA 层。系统在处理视频时跟踪两个状态变量：每帧的低分辨率视差图 $\hat { \mathbf { d } } _ { i } \in \mathcal { R } ^ { \frac { H } { 8 } \times \frac { W } { 8 } }$ 和相机姿态 $\hat { \bf G } _ { i } \in S E ( 3 )$。这些变量通过可微分 BA 层迭代更新，该层操作于动态构建的帧图 $( I _ { i } , I _ { j } ) \in \mathcal { P }$ 上的图像对。

给定两个视频帧 `I _ { i }` 和 `I _ { j }`，DROID-SLAM 通过卷积门控循环单元迭代预测 2D 对应场 $\hat { \mathbf { u } } _ { i j }$ 和置信度 $\hat { \mathbf { w } } _ { i j }$。刚性运动对应场也可以从相机自运动和视差通过多视图约束推导：

$$
{ \bf u } _ { i j } = \pi \left( \hat { \bf G } _ { i j } \circ \pi ^ { - 1 } ( { \bf p } _ { i } , \hat { \bf d } _ { i } , K ^ { - 1 } ) , K \right) ,
$$

其中 $\mathbf { p } _ { i }$ 表示像素坐标网格，$\pi$ 表示投影函数，$\hat { \mathbf { G } } _ { i j } = \hat { \mathbf { G } } _ { j } \circ \hat { \mathbf { G } } _ { i } ^ { - 1 }$ 是 `I _ { i }` 和 `I _ { j }` 之间的相对相机姿态，$\boldsymbol { K } \in \mathcal { R } ^ { 3 \times 3 }$ 表示相机内参矩阵。

<strong>可微分光束法平差 (Differentiable Bundle Adjustment)：</strong>
DROID-SLAM 假设焦距已知，但随意视频的焦距通常未知。因此，MegaSaM 通过迭代最小化网络预测的当前流与从相机参数和视差推导的刚性运动流之间的加权重投影成本来优化相机姿态、焦距和视差：

$$
\mathcal { C } ( \hat { \mathbf { G } } , \hat { \mathbf { d } } , \hat { f } ) = \sum _ { ( i , j ) \in \mathcal { P } } | | \hat { \mathbf { u } } _ { i j } - \mathbf { u } _ { i j } | | _ { \Sigma _ { i j } } ^ { 2 }
$$

其中权重 $\begin{array} { r } { \Sigma _ { i j } = \mathrm { d i a g } ( \hat { \mathbf { w } } _ { i j } ) ^ { - 1 } \end{array}$。为了实现可微分的端到端训练，使用 Levenberg-Marquardt 算法优化上述公式：

$$
\left( \mathbf { J } ^ { T } \mathbf { W } \mathbf { J } + \lambda \mathrm { d i a g } ( \mathbf { J } ^ { T } \mathbf { W } \mathbf { J } ) \right) \boldsymbol { \Delta } \boldsymbol { \xi } = \mathbf { J } ^ { T } \mathbf { W } \mathbf { r }
$$

其中 $\Delta \pmb { \xi } = ( \Delta \mathbf G , \Delta \mathbf d , \Delta f ) ^ { T }$ 是状态变量的参数更新，$\mathbf { J }$ 是重投影残差关于参数的雅可比矩阵，$\mathbf { W }$ 是包含每对帧的 $\hat { \mathbf { w } } _ { i j }$ 的对角矩阵。$\lambda$ 是网络在每次 BA 迭代期间预测的阻尼因子。

可以通过将方程左侧的近似 Hessian 矩阵划分为以下块矩阵形式，将相机参数（包括姿态和焦距）和视差变量分开：

$$
\begin{array} { r } { \left[ \mathbf { H } _ { \mathbf { G } , f } \quad \mathbf { E } _ { \mathbf { G } , f } \right] \left[ \Delta \xi _ { \mathbf { G } , f } \right] = \binom { \tilde { r } _ { \mathbf { G } , f } } { \tilde { r } _ { \mathbf { d } } } } \\ { \mathbf { E } _ { \mathbf { G } , f } ^ { T } \quad \mathbf { H } _ { \mathbf { d } } } \end{array}
$$

由于方程 2 中的每对重投影项中只包含单个视差变量，方程 4 中的 $\mathbf { H _ { d } }$ 是一个对角矩阵，因此我们可以使用 Schur 补技巧高效计算参数更新，从而实现完全可微的 BA 更新：

$$
\begin{array} { r l r } & { \Delta \pmb { \xi } _ { \mathbf { G } , f } = \left[ \mathbf { H } _ { \mathbf { G } , f } - \mathbf { E } _ { \mathbf { G } , f } \mathbf { H } _ { \mathbf { d } } ^ { - 1 } \mathbf { E } _ { \mathbf { G } , f } { } ^ { T } \right] ^ { - 1 } \left( \tilde { r } _ { \mathbf { G } , f } - \mathbf { E } _ { \mathbf { G } , f } \mathbf { H } _ { \mathbf { d } } ^ { - 1 } \tilde { r } _ { \mathbf { d } } \right) } & \\ & { \qquad ( 5 ) } & \\ & { \Delta \mathbf { z } = \mathbf { H } _ { \mathbf { d } } ^ { - 1 } ( \tilde { r } _ { \mathbf { d } } - \mathbf { E } _ { \mathbf { G } , f } ^ { T } \Delta \pmb { \xi } _ { \mathbf { G } , f } ) } & { ( 6 ) } \end{array}
$$

**训练：** 流和不确定性预测是从静态场景的合成视频集合中进行端到端训练的：

$$
\mathcal { L } _ { \mathrm { s t a t i c } } = \mathcal { L } _ { \mathrm { c a m } } + w _ { \mathrm { f l o w } } \mathcal { L } _ { \mathrm { f l o w } }
$$

其中 $\mathcal { L } _ { \mathrm { c a m } }$ 和 ${ \mathcal { L } } _ { \mathrm { f l o w } }$ 是比较 BA 层估计的相机参数和自运动诱导流与相应真实标注数据 (Ground Truth) 的损失。

### 4.2.2. 扩展到随意动态视频
深度视觉 SLAM 在静态场景和足够相机平移的视频上表现良好，但在动态内容或有限视差的视频上性能下降。为了克服这些问题，作者提出了对原始训练和推理流水线的关键修改。

下图（原文 Figure 10）展示了流量、置信度和运动图预测器的架构，说明了网络 $F$ 和 $F_m$ 的分工：

![Figure 10. Architecture of fow, confidence and movement map predictor. The gray blocks belong to the nework $F$ for flow and confidence prediction, and the blue blocks belong to the network `F _ { m }` for object movement map prediction. In the first stage, we perform ego-motion pretraining for $F$ . In the second stage, we perform dynamic fine-tuning for `F _ { m }` while fixing the parameters of $F$ .](images/10.jpg)

<strong>学习运动概率 (Learning Motion Probability)：</strong>
为了扩展模型以处理动态场景，作者提出使用额外的网络 `F _ { m }` 迭代预测对象运动概率图 $\mathbf { m } _ { i } \in \mathcal { R } ^ { \frac { H } { 8 } \times \frac { w } { 8 } } = F _ { m } \left( \{ I _ { i } \} \cup \mathcal { N } ( i ) \right)$，该图以 `I _ { i }` 和一组相邻关键帧 $\mathcal { N } ( i ) = \{ I _ { j } | ( i , j ) \in \mathcal { P } \}$ 为条件。该运动图专门被监督以预测对应于动态内容的像素。在每次 BA 迭代期间，将成对流置信度 $\hat { \mathbf { w } } _ { i j }$ 与对象运动图 $\mathbf { m } _ { i }$ 结合以形成方程 2 中的最终权重：$\tilde { \mathbf { w } } _ { i j } = \hat { \mathbf { w } } _ { i j } \mathbf { m } _ { i }$。

此外，设计了一个两阶段训练方案，在静态和动态视频的混合数据上训练模型，以有效学习 2D 流以及运动概率图。
1.  **自运动预训练阶段：** 使用静态场景的合成数据训练原始深度 SLAM 模型 $F$，监督预测的流和置信度图。这有助于模型有效学习仅由自运动引起的成对流和相应置信度。
2.  **动态微调阶段：** 冻结 $F$ 的参数，在合成动态视频上微调 `F _ { m }`，在每次迭代期间以预训练的 $F$ 的特征为条件，预测运动概率图 $\mathbf { m } _ { i }$，并通过相机和交叉熵损失进行监督：

    $$
\mathcal { L } _ { \mathrm { d y n a m i c } } = \mathcal { L } _ { \mathrm { c a m } } + w _ { \mathrm { m o t i o n } } \mathcal { L } _ { \mathrm { C E } }
$$

下图（原文 Figure 3）展示了学习到的运动概率图示例，右侧的热力图显示了被识别为动态区域的概率：

![Figure 3. Learned movement map. Left: input video frame, right: corresponding learned motion probability map.](images/3.jpg)

<strong>视差和相机初始化 (Disparity and Camera Initialization)：</strong>
DROID-SLAM 通过将视差 $\hat { \mathbf { d } }$ 简单设置为常数值 1 来初始化。然而，作者发现这种初始化在相机基线有限和场景动态复杂的视频上无法执行准确的相机跟踪。受近期工作启发，在训练和推理阶段，通过集成单目深度先验执行数据驱动初始化。在训练期间，使用来自 DepthAnything 的视差初始化 $\breve { \mathbf { d } }$，并借用每个训练序列的真实标注数据 (Ground Truth) 深度的估计全局尺度和偏移。对于每个训练序列，将前两个相机姿态初始化为真实标注数据以消除规范模糊 (gauge ambiguity)，并通过随机扰动真实标注数据值 $2 5 \%$ 来初始化相机焦距。

### 4.2.3. 推理
推理流水线由两个组件组成：(i) 前端模块通过执行帧选择 followed by 滑动窗口 BA 来注册关键帧的相机。(ii) 后端模块通过对所有视频帧执行全局 BA 来细化估计。

**初始化和前端跟踪：**
类似于训练，将单目深度和焦距预测集成到推理流水线中。特别是，使用度量对齐的单目视差 $D _ { i } ^ { \mathrm { a l i g n } } = \hat { \alpha } D _ { i } ^ { \mathrm { r e l } } + \hat { \beta }$ 初始化每帧视差图 $\hat { \mathbf { d } } _ { i }$，其中 $D _ { i } ^ { \mathrm { r e l } }$ 是来自 [71] 的每帧仿射不变视差，每视频全局尺度和偏移参数 $( { \hat { \alpha } } , { \hat { \beta } } )$ 通过将 $D _ { i } ^ { \mathrm { r e l } }$ 与来自 UniDepth 的额外度量深度估计 $D _ { i } ^ { \mathrm { a b s } }$ 进行中值对齐来估计。UniDepth 模型还预测每帧的焦距；使用视频帧的中值估计获得初始焦距估计 $\hat { f }$，该估计在前端阶段是固定的。

为了初始化 SLAM 系统，积累具有足够成对运动的关键帧，直到拥有 $N _ { \mathrm { i n i t } } = 8$ 个活动图像集。通过执行仅相机运动的束调整同时固定视差变量 $\hat { \mathbf { d } } _ { i }$ 为对齐的单目视差 $D _ { i } ^ { \mathrm { a l i g n } }$ 来初始化这些关键帧的相机姿态。逐渐添加新关键帧，移除旧关键帧，并以滑动窗口方式执行局部 BA，其中每个关键帧视差也初始化为对齐的单目视差。在此阶段，BA 成本函数由重投影误差和单目深度正则化项组成：

$$
\mathcal { C } = \sum _ { ( i , j ) \in \mathcal { P } } | | \widehat { \mathbf { u } } _ { i j } - \mathbf { u } _ { i j } | | _ { \Sigma _ { i j } } ^ { 2 } + w _ { d } \sum _ { i } | | \widehat { \mathbf { d } } _ { i } - D _ { i } ^ { \mathrm { a l i g n } } | | ^ { 2 } .
$$

<strong>不确定性感知全局 BA (Uncertainty-aware Global BA)：</strong>
后端模块首先对所有关键帧执行全局 BA。然后执行位姿图优化以注册非关键帧的姿态。最后，后端模块通过对所有视频帧的全局 BA 细化整个相机轨迹。

一个关键问题是：是否（或何时）应将方程 9 中的单目深度正则化添加到全局束调整中？一方面，如果输入视频中有足够的相机基线，观察到不需要单目深度正则化，因为问题已经约束良好，事实上单目深度的误差可能会降低相机跟踪精度。另一方面，如果视频是由相机基线很小的旋转相机拍摄的，那么执行仅重投影的 BA 而没有额外约束可能会导致退化解，如下图（原文 Figure 2）所示：

![Figure 2. Ablation on our design choices. From left to right, we visualize cameras and reconstruction from our system (a) without mono-depth initialization, (b) without uncertainty-aware BA, (c) with full configuration. For these difficult near-rotational sequences, our full method produces much better camera and scene geometry.](images/2.jpg)
*该图像是图表，展示了我们系统设计选择的消融实验。左侧依次为：不使用单目深度初始化、未考虑不确定性BA、以及完整配置。对于这些难度较大的近旋转序列，我们的完整方法在相机和场景几何重建上表现更佳。*

为了理解原因，作者探索了方程 4 线性系统中的近似 Hessian 矩阵。正如 Goli 等人所示，给定后验 $p ( \boldsymbol { \theta } | \mathcal { T } )$，可以使用 Laplace 近似通过逆 Hessian 估计变量的协方差 $\Sigma$：$\Sigma _ { \theta } = - \mathbf { H } ( \theta ^ { * } ) ^ { - 1 }$，其中 $\theta ^ { * }$ 是参数的 MAP 估计，$\Sigma _ { \theta }$ 表示估计变量的认知不确定性 (epistemic uncertainty)。由于当输入帧数量较大时反转完整 Hessian 计算成本高，作者遵循 Ritter 等人并通过 Hessian 的对角线近似 $\Sigma _ { \theta }$：

$$
\Sigma _ { \theta } \approx \mathrm { d i a g } \left( - \mathbf { H } ( \theta ^ { * } ) \right) ^ { - 1 }
$$

直观地说，当考虑方程 2 中的重投影误差时，估计变量的雅可比 $\mathbf { J } _ { \theta }$ 表示如果扰动变量重投影误差会变化多少。因此，当扰动参数对重投影误差影响很小时，不确定性 $\Sigma _ { \theta }$ 很大。具体来说，考虑视差变量，并考虑输入视频由静态相机拍摄的极端情况。在这种情况下，成对重投影误差作为视差的函数将保持不变，意味着估计视差的不确定性很大；即，仅从视频中无法观测到视差。作者在下图（原文 Figure 4）中可视化了估计归一化视差 $\Sigma _ { d }$ 的空间不确定性：

![Figure 4. Visualization of epistemic uncertainty. From left to right, we visualize camera paths, reference image and corresponding epistemic uncertainty of disparity. The geometry is not observable from the top example with little camera parallax, as indicated by the larger uncertainty. The peak on the bottom uncertainty map corresponds to the epipole for forward moving motion.](images/4.jpg)
*该图像是图表，展示了相机路径、参考图像 I 及对应的不确定性 Σ_d。左侧上方为相机路径，左下方展示了动态场景中的两位行走者，右侧为 I 图像和 Σ_d 的不确定性可视化，揭示了不同位置的几何观测能力与不确定性大小的关系。*

这种不确定性量化为我们提供了相机和视差参数可观测性的度量，允许我们决定在哪里添加单目深度正则化（以及何时关闭相机焦距优化）。在实践中，作者发现简单地检查归一化视差的中值不确定性和归一化焦距的不确定性对所有测试视频都有效。特别是，在完成前端跟踪后，检索由所有关键帧形成的视差 Hessian 的对角线条目并计算其中值 $\mathrm { m e d } \left( \mathrm { d i a g } ( \mathbf { H } _ { \mathbf { d } } ) \right)$，以及共享焦距的 Hessian 条目 `H _ { f }`。然后根据中值视差 Hessian 设置单目深度正则化权重 $w _ { d } = \gamma _ { d } \exp \left( - \beta _ { d } \mathrm { m e d } \left( \mathrm { d i a g } ( \mathbf { H _ { d } } ) \right) \right)$。换句话说，如果由于相机运动视差有限导致相机姿态仅从输入视频中无法观测，则启用单目深度正则化。此外，如果 $H _ { f } < \tau _ { f }$，则禁用焦距优化，因为此条件表明焦距可能从输入中无法观测。

### 4.2.4. 一致深度优化
可选地，给定估计的相机参数，可以在比估计的低分辨率视差变量更高的分辨率下获得更准确和一致的视频深度。

作者遵循 CasualSAM 并沿视频深度以及每帧偶然不确定性图执行额外的一阶优化。目标由三个成本函数组成：

$$
\mathcal { C } _ { \mathrm { c v d } } = w _ { \mathrm { f l o w } } \mathcal { C } _ { \mathrm { f l o w } } + w _ { \mathrm { t e m p } } \mathcal { C } _ { \mathrm { t e m p } } + w _ { \mathrm { p r i o r } } \mathcal { C } _ { \mathrm { p r i o r } }
$$

其中 $\mathcal { C } _ { \mathrm { f l o w } }$ 表示成对 2D 流重投影损失，$\mathcal { C } _ { \mathrm { t e m p } }$ 是时间深度一致性损失，$\mathcal { C } _ { \mathrm { p r i o r } }$ 是尺度不变单目深度先验损失。从现成模块推导原始帧分辨率的 2D 光流。

与 CasualSAM 相比，设计有一些不同：(i) 不执行耗时的单目深度网络微调，而是构建和优化输入视频上的视差和不确定性变量序列；(ii) 在优化期间固定相机参数而不是联合优化相机和深度；(iii) 采用表面法线一致性和多尺度深度梯度匹配损失来替换 CasualSAM 中使用的深度先验损失。这些修改导致更快的优化时间以及更准确的视频深度估计。

# 5. 实验设置

## 5.1. 数据集
实验在以下数据集上进行：
*   **MPI Sintel：** 包含具有复杂物体运动和相机路径的动画视频序列。评估了数据集中的 18 个序列，每个序列包含 20-50 张图像。
*   **DyCheck：** 最初设计用于评估新视图合成任务，包含从手持相机捕获的动态场景的真实世界视频。每个视频包含 180-500 帧。使用 Shape of Motion 提供的细化相机参数和传感器深度作为真实标注数据 (Ground Truth)。
*   **In-the-wild：** 在随意动态视频上进一步评估。具体包括 DynIBaR 使用的 12 个随意视频。这些视频具有长时间持续时间（100-600 帧）、不受控制的相机路径和复杂的场景运动。通过实例分割构建真实标注数据运动掩码，在运行 COLMAP 获得可靠相机参数之前屏蔽移动物体。

## 5.2. 评估指标
*   **相机姿态估计：**
    1.  <strong>绝对平移误差 (Absolute Translation Error, ATE)：</strong> 衡量估计相机轨迹与真实标注数据轨迹之间的全局平移偏差。
    2.  <strong>相对平移误差 (Relative Translation Error, RTE)：</strong> 衡量局部帧间平移的准确性。
    3.  <strong>相对旋转误差 (Relative Rotation Error, RRE)：</strong> 衡量局部帧间旋转的准确性。
    *   **公式：** 这些通常计算为估计姿态与真实标注数据姿态对齐后的均方根误差 (RMSE)。对齐通过计算全局 $\mathrm { S i m ( 3 ) }$ 变换（Umeyama 对齐）完成。
*   **深度估计：**
    1.  <strong>绝对相对误差 (Absolute Relative Error, abs-rel)：</strong> 衡量预测深度与真实标注数据深度之间的相对差异。
        *   公式：$\frac{1}{N} \sum \frac{|D_{pred} - D_{gt}|}{D_{gt}}$
    2.  <strong>对数均方根误差 (log RMSE)：</strong> 在对数空间中衡量深度误差，对大深度值更敏感。
        *   公式：$\sqrt{\frac{1}{N} \sum (\log D_{pred} - \log D_{gt})^2}$
    3.  <strong>Delta 准确率 ($\delta _ { 1 . 2 5 }$)：</strong> 衡量预测深度在真实标注数据一定阈值范围内的像素比例。
        *   公式：$\frac{1}{N} \sum \mathbb{I}(\max(\frac{D_{pred}}{D_{gt}}, \frac{D_{gt}}{D_{pred}}) < 1.25)$
    *   **符号解释：** $D_{pred}$ 为预测深度，$D_{gt}$ 为真实标注数据深度，$N$ 为像素总数。

## 5.3. 对比基线
论文将 MegaSaM 与最近的相机姿态估计方法进行了比较，包括：
*   **ACE-Zero：** 基于场景坐标回归的最先进相机定位方法，设计用于静态场景。
*   **CasualSAM & RoDynRF：** 通过优化单目深度网络或 instant-NGP 联合估计相机参数和稠密场景几何。
*   **Particle-SfM & LEAP-VO：** 通过预测长时轨迹的运动分割来估计动态视频的相机，然后在标准视觉里程计或 SfM 流水线中屏蔽移动物体。
*   **MonST3R：** 并发工作，扩展 Dust3R 以处理动态场景，从输入帧对预测的全局 3D 点云估计相机参数。
*   **DepthAnything-V2：** 原始单目深度，用于完整性比较。

# 6. 实验结果与分析

## 6.1. 核心结果分析
数值结果表明，MegaSaM 在相机姿态估计的所有误差指标上（校准和未校准设置）均表现出显著改进并达到最佳相机跟踪精度，同时在运行时间上具有竞争力。值得注意的是，即使 MonST3R 采用了更新的全局 3D 点云表示，MegaSaM 在鲁棒性和准确性上也优于该并发工作。深度预测结果同样在所有指标上显著优于其他基线。

以下是原文 Table 1 的结果，展示了在 Sintel 数据集上的相机估计定量比较：

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="3">Calibrated</th>
<th colspan="4">Uncalibrated</th>
</tr>
<tr>
<th>ATE</th>
<th>RTE</th>
<th>RRE</th>
<th>ATE</th>
<th>RTE</th>
<th>RRE</th>
<th>Time</th>
</tr>
</thead>
<tbody>
<tr>
<td>CasualSAM [78]</td>
<td>0.041</td>
<td>0.023</td>
<td>0.17</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>1.3s</td>
</tr>
<tr>
<td>LEAP-VO [6]</td>
<td>0.036</td>
<td>0.013</td>
<td>0.20</td>
<td>0.067</td>
<td>0.019</td>
<td>0.47</td>
<td>1.6m</td>
</tr>
<tr>
<td>ACE-Zero [3]</td>
<td>0.053</td>
<td>0.028</td>
<td>1.26</td>
<td>0.065</td>
<td>0.028</td>
<td>0.30</td>
<td>10s</td>
</tr>
<tr>
<td>Particle-SfM [79]</td>
<td>0.062</td>
<td>0.032</td>
<td>1.92</td>
<td>0.057</td>
<td>0.038</td>
<td>1.64</td>
<td>21s</td>
</tr>
<tr>
<td>RoDynRF [34]</td>
<td>0.110</td>
<td>0.049</td>
<td>1.68</td>
<td>0.109</td>
<td>0.051</td>
<td>1.32</td>
<td>15m</td>
</tr>
<tr>
<td>MonST3R [76]</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>0.078</td>
<td>0.038</td>
<td>0.49</td>
<td>1.0s</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.018</b></td>
<td><b>0.008</b></td>
<td><b>0.04</b></td>
<td><b>0.023</b></td>
<td><b>0.008</b></td>
<td><b>0.06</b></td>
<td><b>1.0s</b></td>
</tr>
</tbody>
</table>

以下是原文 Table 2 的结果，展示了在 DyCheck 数据集上的相机估计定量比较：

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="3">Calibrated</th>
<th colspan="4">Uncalibrated</th>
</tr>
<tr>
<th>ATE</th>
<th>RTE</th>
<th>RRE</th>
<th>ATE</th>
<th>RTE</th>
<th>RRE</th>
<th>Time</th>
</tr>
</thead>
<tbody>
<tr>
<td>CasualSAM [78]</td>
<td>0.185</td>
<td>0.022</td>
<td>0.23</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>0.8s</td>
</tr>
<tr>
<td>LEAP-VO [6]</td>
<td>0.167</td>
<td>0.011</td>
<td>0.09</td>
<td>0.209</td>
<td>0.027</td>
<td>0.28</td>
<td>2.8m</td>
</tr>
<tr>
<td>ACE-Zero [3]</td>
<td>0.062</td>
<td>0.012</td>
<td>0.11</td>
<td>0.056</td>
<td>0.012</td>
<td>0.12</td>
<td>1.6s</td>
</tr>
<tr>
<td>Particle-SfM [79]</td>
<td>0.081</td>
<td>0.014</td>
<td>0.20</td>
<td>0.087</td>
<td>0.015</td>
<td>0.29</td>
<td>35s</td>
</tr>
<tr>
<td>RoDynRF [34]</td>
<td>0.548</td>
<td>0.074</td>
<td>0.70</td>
<td>0.562</td>
<td>0.087</td>
<td>0.90</td>
<td>6.6m</td>
</tr>
<tr>
<td>MonST3R [76]</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>0.690</td>
<td>0.078</td>
<td>0.54</td>
<td>1.0s</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.020</b></td>
<td><b>0.005</b></td>
<td><b>0.05</b></td>
<td><b>0.020</b></td>
<td><b>0.005</b></td>
<td><b>0.06</b></td>
<td><b>1.0s</b></td>
</tr>
</tbody>
</table>

以下是原文 Table 3 的结果，展示了在 In-the-Wild  footage 数据集上的相机估计定量比较：

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="3">Calibrated</th>
<th colspan="4">Uncalibrated</th>
</tr>
<tr>
<th>ATE</th>
<th>RTE</th>
<th>RRE</th>
<th>ATE</th>
<th>RTE</th>
<th>RRE</th>
<th>Time</th>
</tr>
</thead>
<tbody>
<tr>
<td>CasualSAM [78]</td>
<td>0.016</td>
<td>0.004</td>
<td>0.031</td>
<td>0.005</td>
<td>0.31</td>
<td>0.04</td>
<td>1.1m</td>
</tr>
<tr>
<td>LEAP-VO [6]</td>
<td>0.035</td>
<td>0.005</td>
<td>0.30</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>0.6s</td>
</tr>
<tr>
<td>ACE-Zero [3]</td>
<td>0.091</td>
<td>0.051</td>
<td>0.007</td>
<td>0.091</td>
<td>0.008</td>
<td>0.08</td>
<td>4.0s</td>
</tr>
<tr>
<td>Particle-SfM [79]</td>
<td>0.008</td>
<td>0.054</td>
<td>0.007</td>
<td>0.09</td>
<td>0.14</td>
<td>49s</td>
</tr>
<tr>
<td>RoDynRF [34]</td>
<td>0.116</td>
<td>0.021</td>
<td>0.34</td>
<td>0.112</td>
<td>0.031</td>
<td>0.39</td>
<td>7.6m</td>
</tr>
<tr>
<td>MonST3R [76]</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>0.073</td>
<td>0.014</td>
<td>0.18</td>
<td>1.7s</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.004</b></td>
<td><b>0.001</b></td>
<td><b>0.01</b></td>
<td><b>0.005</b></td>
<td><b>0.01</b></td>
<td><b>0.03</b></td>
<td><b>1.3s</b></td>
</tr>
</tbody>
</table>

*(注：Table 3 数据基于输入文本转录，部分数据可能存在排版对齐差异，以原文 PDF 为准)*

以下是原文 Table 4 的结果，展示了视频深度的定量比较：

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="3">Sintel [4]</th>
<th colspan="3">Dycheck [12]</th>
</tr>
<tr>
<th>abs-rel</th>
<th>log-rmse</th>
<th>δ1.25</th>
<th>abs-rel</th>
<th>log-rmse</th>
<th>δ1.25</th>
</tr>
</thead>
<tbody>
<tr>
<td>DA-v2 [72]</td>
<td>0.37</td>
<td>0.55</td>
<td>58.6</td>
<td>0.20</td>
<td>0.27</td>
<td>84.7</td>
</tr>
<tr>
<td>DepthCrafter [20]</td>
<td>0.27</td>
<td>0.50</td>
<td>68.2</td>
<td>0.22</td>
<td>0.29</td>
<td>83.7</td>
</tr>
<tr>
<td>CasualSAM [78]</td>
<td>0.31</td>
<td>0.49</td>
<td>64.2</td>
<td>0.21</td>
<td>0.30</td>
<td>78.4</td>
</tr>
<tr>
<td>MonST3R [76]</td>
<td>0.31</td>
<td>0.43</td>
<td>62.5</td>
<td>0.26</td>
<td>0.35</td>
<td>66.5</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.21</b></td>
<td><b>0.39</b></td>
<td><b>73.1</b></td>
<td><b>0.11</b></td>
<td><b>0.20</b></td>
<td><b>94.1</b></td>
</tr>
</tbody>
</table>

## 6.2. 消融实验/参数分析
作者进行了消融实验以验证相机跟踪和深度估计模块的主要设计选择。
下表（原文 Table 5）展示了在 Sintel 数据集上的消融研究结果：

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="3">Poses</th>
<th colspan="2">Depth</th>
</tr>
<tr>
<th>ATE</th>
<th>RTE</th>
<th>RRE</th>
<th>Abs-Rel</th>
<th>δ1.25</th>
</tr>
</thead>
<tbody>
<tr>
<td>Droid-SLAM [59]</td>
<td>0.030</td>
<td>0.022</td>
<td>0.50</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>w/o mono-init.</td>
<td>0.038</td>
<td>0.026</td>
<td>0.49</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>w/o m_i</td>
<td>0.032</td>
<td>0.127</td>
<td>0.14</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>w/o 2-stage train.</td>
<td>0.035</td>
<td>0.136</td>
<td>0.17</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>w/o u-BA</td>
<td>0.033</td>
<td>0.013</td>
<td>0.11</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>w/ ft-pose</td>
<td>0.041</td>
<td>0.018</td>
<td>0.33</td>
<td>0.23</td>
<td>71.2</td>
</tr>
<tr>
<td>w/o new C_prior</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>0.36</td>
<td>72.5</td>
</tr>
<tr>
<td><b>Full</b></td>
<td><b>0.019</b></td>
<td><b>0.008</b></td>
<td><b>0.04</b></td>
<td><b>0.21</b></td>
<td><b>73.1</b></td>
</tr>
</tbody>
</table>

结果显示，完整系统优于所有其他替代配置。特别是，移除单目深度初始化、运动图预测或两阶段训练都会导致相机姿态误差显著增加。

## 6.3. 定性比较
下图（原文 Figure 5）展示了估计相机轨迹的定性比较，MegaSaM 的估计（红色虚线）比所有其他基线更接近真实标注数据相机轨迹（蓝色实线）：

![Figure 5. Visualization of estimated camera trajectories. Due to scene dynamics, our camera estimate (red dash) deviates less from the ground truth camera trajectory (blue solid line) than all other baselines.](images/5.jpg)
*该图像是图表，展示了在不同场景下估计的相机轨迹，包括 alley 1、block、dog-running、ambush 4、pillow 和 girl-spinning。图中红色虚线表示我们的方法，蓝色实线为地面真实轨迹，其他颜色代表不同基线。由于场景动态，我们的方法在相机轨迹估计中表现出更小的偏差。*

下图（原文 Figure 6）展示了视频深度的视觉比较，MegaSaM 产生了更准确、详细且时间一致的视频深度：

![Figure 6. Visual comparisons of video depths. We compare video depth estimates from our approach and from CasualSAM \[78\] and MonST3R \[76\] by visualizing their depth maps (odd columns) and corresponding $x { - } t$ slices (even columns).](images/6.jpg)
*该图像是一个示意图，比较我们的深度估计方法与 CasualSAM 及 MonST3R 的视频深度估计。图中包含不同场景的输入图像（顶部），以及各自对应的深度图 (GT、CsAM、MonST 和 Ours) 在下方的可视化效果，展示了各自的深度估计差异。*

下图（原文 Figure 7）进一步对比了不同方法在具有挑战性的 DAVIS 示例上的重建和相机跟踪质量，MegaSaM 产生了更准确的相机和更一致的几何结构：

![该图像是示意图，展示了不同方法（CasualSaM、MonST3R 和本文提出的方法）的结构与运动估计效果。上方展示了输入动态视频及相应的深度图形，底部则对比了各方法在相同输入下的输出结果。可以观察到，本文方法在复杂动态场景中的表现更为准确和稳健。](images/7.jpg)
*该图像是示意图，展示了不同方法（CasualSaM、MonST3R 和本文提出的方法）的结构与运动估计效果。上方展示了输入动态视频及相应的深度图形，底部则对比了各方法在相同输入下的输出结果。可以观察到，本文方法在复杂动态场景中的表现更为准确和稳健。*

# 7. 总结与思考

## 7.1. 结论总结
MegaSaM 提出了一套从随意单目动态视频中生成准确相机参数和一致深度的流水线。该方法有效地扩展到具有不同时间持续时间、不受约束的相机路径和复杂场景动态的随意镜头。论文证明，通过仔细扩展，先前的深度视觉 SLAM 和 SfM 框架可以实现对广泛视频的强泛化能力，并显著优于最近的最先进 (state-of-the-art) 方法。

## 7.2. 局限性与未来工作
作者指出了以下局限性：
*   **极端挑战场景：** 如果移动物体主导整个图像或系统没有可靠跟踪的内容，相机跟踪可能会失败。
*   **相机运动与物体运动共线：** 在动态视频中，如果相机运动和物体运动共线，系统也会遇到困难。
*   **焦距变化与畸变：** 系统无法处理视频中焦距变化或强径向畸变的情况。
*   **未来方向：** 将当前视觉基础模型的更好先验集成到流水线中是一个值得探索的方向。

## 7.3. 个人启发与批判
*   **启发：** MegaSaM 展示了将传统几何优化（BA）与深度学习先验（单目深度、运动概率）相结合的强大潜力。它没有完全抛弃几何约束，而是用学习到的组件来增强几何方法在困难情况下的鲁棒性。这种“混合”范式可能是未来三维视觉系统的发展方向。
*   **批判性思考：**
    1.  **依赖外部先验：** 系统依赖 DepthAnything 和 UniDepth 等外部单目深度模型。如果这些基础模型在特定域上失败，MegaSaM 的性能可能会受到影响。
    2.  **计算复杂度：** 虽然声称快速，但涉及多个优化阶段（前端 BA、后端全局 BA、深度优化），在资源受限的边缘设备上部署可能仍有挑战。
    3.  **动态物体处理：** 虽然引入了运动概率图，但在极端动态场景（如人群密集）下，运动分割的准确性仍是瓶颈。
    4.  **泛化性：** 尽管在多个数据集上表现良好，但对于训练数据分布之外的极端光照或纹理缺乏场景，仍需进一步验证。

        总体而言，MegaSaM 是动态场景三维重建领域的重要进展，为随意视频的高质量三维理解提供了实用且高效的解决方案。