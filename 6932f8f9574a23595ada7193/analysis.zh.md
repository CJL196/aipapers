# 1. 论文基本信息

## 1.1. 标题
<strong>ScoreHOI: 通过分数引导扩散实现物理上合理的人-物交互重建 (ScoreHOI: Physically Plausible Reconstruction of Human-Object Interaction via Score-Guided Diffusion)</strong>

论文的核心主题是利用一种基于<strong>分数引导扩散 (Score-Guided Diffusion)</strong> 的新方法，来解决从单张图像中三维重建人与物体交互 (Human-Object Interaction, HOI) 的问题，并确保重建结果在物理上是合理和可信的。

## 1.2. 作者
*   **作者:** Ao Li, Jinpeng Liu, Yixuan Zhu, Yansong Tang*
*   **隶属机构:** 清华大学 (Tsinghua University) 和清华大学深圳国际研究生院 (Tsinghua Shenzhen International Graduate School)。
*   **研究背景:** 作者团队在计算机视觉和图形学领域有深入的研究，特别是在人体姿态估计、运动生成和三维重建等方面。Yansong Tang 是通讯作者，在该领域有多篇高水平论文发表。

## 1.3. 发表期刊/会议
*   **发布平台:** arXiv (预印本)
*   **说明:** 本文是一篇预印本，提交于 arXiv 平台。根据其引用的会议（如 CVPR, ECCV, ICCV）以及研究主题，该论文的目标投递会议很可能是计算机视觉领域的顶级会议之一。
*   **特殊说明:** 论文中提供的发表日期 (2025-09-09) 和部分参考文献的年份（如 ECCV 2025, ICLR 2025）均指向未来，这表明本文档中的信息可能是虚构或占位符，在分析时应将此作为背景信息。

## 1.4. 发表年份
根据 arXiv 编号和元数据，为 **2025** 年 (虚构日期)。

## 1.5. 摘要
论文旨在解决从单张图像中联合重建三维人体和交互物体姿态的挑战。现有的优化方法由于缺乏关于人-物交互的先验知识，往往难以生成物理上合理的结果。为了克服这一限制，本文提出了一种名为 **ScoreHOI** 的新型优化器，它基于扩散模型，并引入了<strong>扩散先验 (diffusion priors)</strong> 来精确恢复人-物交互。通过利用<strong>分数引导采样 (score-guided sampling)</strong> 的可控性，该扩散模型能够根据图像观测和物体特征，重建出人体和物体姿态的条件分布。在推理阶段，ScoreHOI 通过在去噪过程中引入特定的<strong>物理约束 (physical constraints)</strong>（如接触、碰撞）来有效提升重建结果的质量。此外，论文还提出了一种<strong>接触驱动的迭代优化 (contact-driven iterative refinement)</strong> 方法，以增强接触的合理性和重建精度。在标准基准数据集上的大量实验表明，ScoreHOI 的性能优于当前最先进的方法，证明了其在联合人-物交互重建任务中能够实现精确且鲁棒的改进。

## 1.6. 原文链接
*   **原文链接:** https://arxiv.org/abs/2509.07920
*   **PDF 链接:** https://arxiv.org/pdf/2509.07920v1.pdf
*   **发布状态:** 预印本 (Preprint)。

# 2. 整体概括

## 2.1. 研究背景与动机
*   **核心问题:** 从单张 RGB 图像中准确、逼真地重建三维人体和其交互物体的姿态、形状及空间关系。这是一个极具挑战性的<strong>不适定问题 (ill-posed problem)</strong>，因为单个二维图像损失了大量的深度和三维空间信息，可能对应无限多种三维场景。

*   **重要性:** 解决此问题对于理解人类与环境的复杂互动至关重要，在机器人学（如模仿学习）、虚拟/增强现实 (VR/AR)、游戏开发和人机交互等领域有广泛的应用前景。

*   <strong>现有挑战与空白 (Gap):</strong>
    1.  <strong>优化方法 (Optimization-based methods):</strong> 这类方法通过迭代优化来最小化物理约束（如接触、无穿透）的损失。但它们往往**过度依赖物理约束而忽略了图像特征**，导致结果虽然物理上可能合理，但与原始图像的对应关系较差，出现“漂移”现象。此外，这类方法通常**计算效率低下**。
    2.  <strong>回归方法 (Regression-based methods):</strong> 这类方法通过深度神经网络直接从图像中回归出人体和物体的参数。它们速度快，但通常是一步到位的预测，**鲁棒性差**，尤其是在面对严重遮挡或深度模糊的复杂场景时，难以保证物理合理性。
    3.  **缺乏强大的先验:** 无论是优化还是回归方法，都缺乏对“什么是自然的人-物交互”这一问题的强大先验知识。

*   **创新思路:** 本文的切入点是利用<strong>扩散模型 (Diffusion Models)</strong> 作为一种强大的生成先验。作者不将扩散模型用于从零生成，而是巧妙地将其改造为一个**优化器**。其核心思想是：一个好的 HOI 姿态应该位于真实 HOI 数据的分布流形上。因此，可以将一个粗糙的、可能不合理的初始估计，通过扩散模型的去噪过程“拉回”到这个合理的数据流形上，并在“拉回”的过程中，施加物理约束进行引导，从而得到一个既符合图像观测又物理合理的结果。

## 2.2. 核心贡献/主要发现
1.  **提出 ScoreHOI 框架:** 首次提出一个将<strong>分数引导的扩散模型 (score-guided diffusion model)</strong> 用作优化器的框架，专门用于改进人-物交互的三维重建结果。这为解决 HOI 重建问题提供了一个全新的、基于生成先验的优化范式。

2.  **物理约束引导的去噪采样:** 在扩散模型的去噪采样过程中，创新性地引入了**物理约束作为引导**。通过计算物理损失（接触、穿透等）关于潜变量的梯度，来修正去噪方向，使得最终生成的姿态不仅符合数据先验，还满足现实世界的物理规律。

3.  **接触驱动的迭代优化策略:** 设计了一种<strong>接触驱动的迭代优化 (contact-driven iterative refinement)</strong> 机制。该机制在多次优化步骤中，不仅优化人体和物体的姿态参数，还**动态更新接触区域的预测**，从而使物理引导更加精确，进一步提升了接触的真实性和整体重建精度。

4.  **卓越的性能表现:** 在 BEHAVE 和 InterCap 等标准数据集上取得了最先进的 (state-of-the-art) 性能。特别是在衡量接触质量的 **F-Score** 指标上，相比之前的方法取得了 **9%** 的显著提升，同时推理速度远超传统的优化方法。

# 3. 预备知识与相关工作

## 3.1. 基础概念
### 3.1.1. 人-物交互重建 (Human-Object Interaction Reconstruction)
<strong>人-物交互重建 (Human-Object Interaction, HOI) Reconstruction</strong> 是指根据图像或视频等输入，同时恢复三维空间中人体的姿态、形状和其交互物体的姿态、形状及两者之间空间关系的任务。例如，从一张人坐在椅子上的照片，重建出人体和椅子的三维模型，并确保人“坐”在椅子上，而不是悬空或穿过椅子。

### 3.1.2. SMPL 模型
**SMPL (Skinned Multi-Person Linear Model)** 是一种参数化的人体三维模型。它使用少量参数就能生成一个完整且带蒙皮的三维人体网格。其核心参数包括：
*   <strong>姿态参数 (pose parameters, $\theta \in \mathbb{R}^{24 \times 3}$):</strong> 控制身体 24 个关节点的旋转，决定了人体的动作姿态。
*   <strong>形状参数 (shape parameters, $\beta \in \mathbb{R}^{10}$):</strong> 控制人体的体型，如高矮、胖瘦等。
    通过一个可微分的函数 $\mathcal{M}(\theta, \beta)$，这些参数可以映射为一个包含约 7000 个顶点的三维人体网格。本文使用的是 **SMPL-H** 模型，它是在 SMPL 基础上扩展的，额外包含了精细的手部姿态参数，更适合进行抓握等交互动作的建模。

### 3.1.3. 分数引导的扩散模型 (Score-Guided Diffusion Models)
扩散模型是一类强大的生成模型，它包含两个过程：
1.  <strong>前向过程 (Forward Process):</strong> 在此过程中，逐步向真实数据（如一张图片或一组姿态参数）添加高斯噪声，直到数据完全变成纯噪声。这个过程是固定的、无须学习的。
2.  <strong>反向过程 (Reverse Process):</strong> 训练一个神经网络（通常是 U-Net 结构）来学习逆转上述过程，即从纯噪声中逐步去除噪声，最终恢复出原始数据。

    <strong>分数 (Score)</strong> 在这个背景下，指的是数据在某个噪声水平下的**对数概率密度的梯度**，即 $\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)$。它直观地指向了数据密度增加最快的方向。一个训练好的去噪网络 $\epsilon_{\phi}(\mathbf{x}_t, t)$ 所预测的噪声，与分数之间存在直接关系：
$$
\epsilon _ { \phi } ( \pmb { x } _ { t } , t ) \approx - \sqrt { 1 - \alpha _ { t } } \nabla _ { \pmb { x } _ { t } } \log p ( \pmb { x } _ { t } )
$$
其中 $\alpha_t$ 是与噪声水平相关的系数。

<strong>分数引导 (Score Guidance)</strong> 是一种在去噪过程中施加额外控制的技术。基于贝叶斯定理，我们可以修改这个分数，使其包含我们想要的条件或约束。例如，如果我们想让生成过程满足某个物理约束 $\mathcal{P}$，我们可以将原始分数 $\nabla \log p(\mathbf{x}_t)$ 修改为条件分数 $\nabla \log p(\mathbf{x}_t | \mathcal{P})$。这种引导使得模型在生成数据的同时，还能满足特定的外部要求，是实现可控生成的关键。

### 3.1.4. DDIM
**DDIM (Denoising Diffusion Implicit Models)** 是对传统扩散模型（如 DDPM）采样过程的一种加速。DDPM 的采样是马尔可夫链，每一步都依赖于前一步，采样步数通常很多（如 1000 步）。DDIM 提出了一种非马尔可夫的采样过程，允许以更大的步长进行采样，从而能在显著减少采样步数（如 20-50 步）的情况下，生成高质量的结果。DDIM 还引入了一个重要的特性：<strong>确定性反演 (deterministic inversion)</strong>，即可以将一张真实图像通过 DDIM 的反向过程精确地映射到一个潜在的噪声向量，然后再通过正向采样过程完美地恢复原图。这为图像编辑和本文所用的“优化”提供了基础。

## 3.2. 前人工作
*   <strong>优化方法 (e.g., CHORE [59]):</strong> 这类方法通常会有一个初始的姿态估计，然后定义一系列物理损失函数（如接触损失、穿透惩罚），并使用像 Adam 这样的标准优化器来迭代调整人体和物体的参数，直到损失最小。其缺点是优化过程缓慢，且容易陷入局部最优，有时为了满足物理约束而牺牲了与图像的匹配度。

*   <strong>回归方法 (e.g., CONTHO [37]):</strong> 这类方法设计了一个端到端的深度网络，直接从图像特征回归出人体和物体的参数。为了建模交互关系，它们通常会使用<strong>交叉注意力机制 (cross-attention)</strong>，让网络在编码人体和物体特征时能够相互感知。例如，在 CONTHO 中，注意力机制的计算公式类似于经典的 Transformer：
    $$
    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    $$
    其中 Q (Query), K (Key), V (Value) 可能分别来自人体和物体的特征，从而实现信息的交互。这类方法速度快，但单次前向传播的模式使其在复杂场景下不够鲁棒。

*   <strong>扩散模型用于人体重建 (e.g., DPoser [34]):</strong> 近期有工作开始将扩散模型用作人体姿态的先验。例如，训练一个扩散模型来学习自然的人体姿态分布。在重建时，可以将一个不自然或有噪声的姿态通过去噪过程进行“修正”。

## 3.3. 技术演进
该领域的技术演进路线大致如下：
1.  **独立重建:** 分别重建人体和物体，然后尝试将它们组合在一起，但难以保证交互的自然性。
2.  **联合回归:** 设计一个统一的网络，同时预测人体和物体的参数，并使用注意力等机制隐式地建模它们的关系。
3.  **联合优化:** 在回归的基础上，增加一个基于物理约束的优化步骤，以提升结果的物理真实性。
4.  <strong>基于生成先验的优化 (本文):</strong> 将强大的生成模型（扩散模型）作为先验，取代传统的优化器，在数据流形上进行引导式优化，寻求在数据真实性、物理真实性和图像一致性之间取得更好的平衡。

## 3.4. 差异化分析
与之前工作的核心区别在于：
*   **与优化方法的区别:** ScoreHOI 不使用 Adam 等通用优化器在参数空间进行优化，而是利用扩散模型的去噪过程在**数据分布的流形**上进行优化。这相当于利用了扩散模型学到的海量数据先验，使得优化过程更不容易偏离“合理”的姿态范围，同时速度更快。
*   **与回归方法的区别:** ScoreHOI 不是一次性预测，而是一个**迭代优化过程**，因此比单步回归方法更鲁棒。它从一个粗糙的回归结果出发，逐步精化，使其能够修正初始预测中的错误。
*   **与其他人体重建扩散模型的区别:** 之前的工作主要将扩散模型用作姿态的“修正器”或先验，而 ScoreHOI 则将其明确地构建为一个**受物理约束引导的优化器**，并设计了**接触驱动的迭代机制**来动态调整引导信号，这是其在 HOI 任务上的独特创新。

    下图（原文 Figure 1）直观地展示了这三种方法的区别：

    ![Figure 1. Comparison of current methods and the proposed ScoreHOI. (a) Optimization-based methods iteratively refine predicted outcomes with the physical objectives. (b) Regressionbased methods update predictions via a dual-branch forward process. (c) Our ScoreHOI integrates a score-based denoising module that incorporates physical constraints during the sampling process.](images/1.jpg)
    *该图像是示意图，比较了当前方法与提出的ScoreHOI。图中展示了三种方法，分别是(a) 基于优化的方法和通过物理约束进行联合优化，(b) 回归方法中的前向细化过程，以及(c) 我们的ScoreHOI，通过接触驱动的迭代精 refinement，结合噪声去除模块和物理约束，提升了重建精度。*

# 4. 方法论

本节将详细拆解 ScoreHOI 的技术方案。其整体推理流程如下图（原文 Figure 2）所示，主要包括两个阶段：(a) 初始参数估计 和 (b) 接触驱动的迭代优化。

![Figure 2. The inference procedure of ScoreHOI. (a) Given the input image $I$ the human and object segmented silhouette $S _ { \\mathrm { h } } , S _ { \\mathrm { o } }$ and the object template $P _ { \\mathrm { ~ o ~ } }$ , we initially extract the image feature $\\mathcal { F }$ and estimate the SMPL and object parameters $\\theta$ $\\beta$ , $R _ { \\mathrm { o } }$ and $t _ { \\mathrm { o } }$ . (b) Employing a contact-driven iterative refinement strategy, we refine these parameters $_ { \\textbf { \\em x } }$ through the execution of a DDIM inversion and guided sampling lo Due i s, hysl ctant uc pe $L _ { \\mathrm { p t } }$ and contact $L _ { \\mathrm { h o } } , L _ { \\mathrm { o f } }$ are actively supervised. Following each optimization iteration, the contact masks $\\{ \\mathbf { M } _ { i } \\} _ { i \\in \\{ \\mathrm { h , o , f } \\} }$ are updated to enhance the precision of the guidance.](images/2.jpg)
*该图像是示意图，展示了ScoreHOI的推理过程。图中分为(a) 和(b) 两部分，(a) 为“考虑可用性的回归器”，展示了如何从视觉输入 $I$ 和点云 $P_o$ 中提取特征并通过回归器进行SMPL拟合，估计人类和物体的参数；(b) 为“接触驱动的迭代精炼”，阐述了如何进行DDIM反演和指导采样以优化模型参数，并通过接触预测器更新接触掩码以提高指导的精确度。*

## 4.1. 方法原理
ScoreHOI 的核心思想是将扩散模型作为一个带有强大先验知识的优化器。传统的优化方法是在一个高维参数空间中搜索，容易迷失方向或陷入局部最优。ScoreHOI 则将优化问题转化为了一个在数据流形上的“导航”问题：
1.  首先，通过一个回归网络得到一个粗糙的初始解 $\mathbf{x}_0$。
2.  然后，使用 DDIM 反演技术，将这个解 $\mathbf{x}_0$ “加噪”到一个中间噪声水平 $\tau$，得到潜变量 $\mathbf{x}_{\tau}$。这相当于将初始解投影到扩散过程的中间状态。
3.  接着，从 $\mathbf{x}_{\tau}$ 开始执行引导式的去噪采样过程。在每一步去噪时，不仅利用扩散模型预测的“数据先验”方向，还额外计算一个指向“物理更合理”方向的梯度，并将两者结合，共同决定下一步的去噪方向。
4.  最终，经过若干步去噪，得到一个既符合数据先验（看起来自然）又满足物理约束（接触、不穿透）的优化结果。

## 4.2. 核心方法详解 (逐层深入)
### 4.2.1. 阶段一：可用性感知的回归器 (Affordance-Aware Regressor)
这是方法的第一步，旨在从输入图像中获取一个初始的人体和物体参数估计。
*   **输入:**
    *   裁切后的 RGB 图像 $I_{rgb}$。
    *   人体和物体的分割掩码 $S_h, S_o$。
    *   一个粗糙的物体模板点云 $P_o$。
*   **过程:**
    1.  使用一个图像主干网络 (backbone)，如 ResNet50，从输入图像中提取视觉特征 $\mathcal{F}$。
    2.  为了增强对物体几何形状和功能的理解，论文引入了<strong>可用性 (Affordance)</strong> 的概念。具体地，使用一个在大型 3D 物体数据集（如 ModelNet40）上预训练的**可用性感知网络**（如 PointNeXt），从物体模板 $P_o$ 中提取几何特征 $c_G$。这个特征编码了物体的功能属性（如“平面可坐”，“柱体可握”），有助于模型更好地理解交互。
    3.  将视觉特征 $\mathcal{F}$ 和几何特征结合，通过两个独立的预测头 (head)，分别回归出初始的人体参数和物体参数：
        *   人体参数：姿态 $\theta^0$ (SMPL-H, $52 \times 6 = 312$ 维) 和形状 $\beta^0$ (10 维)。
        *   物体参数：旋转 $R_o^0$ (6D 表示, 6 维) 和平移 $t_o^0$ (3 维)。
*   **输出:**
    一个初始的参数向量 $\mathbf{x}^0 = \{\theta^0, \beta^0, R_o^0, t_o^0\}$，总维度为 $312 + 10 + 6 + 3 = 331$ 维。这个 $\mathbf{x}^0$ 将作为后续扩散优化的起点。

### 4.2.2. 阶段二：通过物理引导进行优化 (Optimization with Physical Guidance)
这是方法的核心，即利用扩散模型进行引导式优化。
1.  <strong>DDIM 反演 (Inversion):</strong>
    从初始估计 $\mathbf{x}_0^{\text{init}}$ 出发，执行 DDIM 反演过程，将其映射到一个预设的中间噪声水平 $\tau$ 的噪声潜变量 $\mathbf{x}_{\tau}$。

2.  <strong>引导式去噪采样 (Guided Denoising Sampling):</strong>
    从 $\mathbf{x}_{\tau}$ 开始，逐步去噪直到 $\mathbf{x}_0$。在每一步去噪（从时间步 $t$ 到 $t-\Delta t$）中，核心是计算一个**修正后**的噪声 $\epsilon'_{\phi}$。
    根据贝叶斯法则，带有物理约束 $\mathcal{P}$ 的条件分数为：
    $$
    \nabla _ { \pmb { x } _ { t } } \log p ( \pmb { x } _ { t } | \pmb { c } , \mathcal { P } ) = \nabla _ { \pmb { x } _ { t } } \log p ( \pmb { x } _ { t } | \pmb { c } ) + \nabla _ { \pmb { x } _ { t } } \log p ( \mathcal { P } | \pmb { c } , \pmb { x } _ { t } )
    $$
    *   第一项 $\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t | \mathbf{c})$ 是扩散模型本身学到的数据先验，可以通过网络预测的噪声 $\epsilon_{\phi}(\mathbf{x}_t, t, \mathbf{c})$ 得到。
    *   第二项 $\nabla_{\mathbf{x}_t} \log p(\mathcal{P} | \mathbf{c}, \mathbf{x}_t)$ 是物理引导项，直接计算很困难。论文采用了[7, 22, 50]中的一个关键近似：<strong>物理约束的梯度是在对当前噪声数据 $\mathbf{x}_t$ 进行一步去噪得到的“干净”预测 $\hat{\mathbf{x}}_0(\mathbf{x}_t)$ 上计算的</strong>。
        $$
        \nabla _ { \pmb { x } _ { t } } \log p ( \mathcal { P } | \pmb { c } , \pmb { x } _ { t } ) \simeq \nabla _ { \pmb { x } _ { t } } \log p ( \mathcal { P } | \pmb { c } , \hat { \pmb { x } _ { 0 } } ( \pmb { x } _ { t } ) )
        $$
        其中，一步去噪预测 $\hat{\mathbf{x}}_0(\mathbf{x}_t)$ 的计算公式为：
        $$
        \hat{\pmb{x}_0}(\pmb{x}_t) = \frac{1}{\sqrt{\alpha_t}}(\pmb{x}_t - \sqrt{1 - \alpha_t}\epsilon_{\phi}(\pmb{x}_t, t))
        $$
    
    3.  **定义物理目标 $L_{\mathcal{P}}$:**
        物理约束损失 $L_{\mathcal{P}}$ 由三部分组成，它在预测的“干净”人体网格 $V_h$ 和物体网格 $V_o$（由 $\hat{\mathbf{x}}_0$ 生成）上计算：
        $$
        L _ { \mathcal { P } } = \lambda _ { \mathrm { h o } } L _ { \mathrm { h o } } + \lambda _ { \mathrm { o f } } L _ { \mathrm { o f } } + \lambda _ { \mathrm { p t } } L _ { \mathrm { p t } }
        $$
        *   <strong>人-物接触 ($L_{ho}$):</strong> $L _ { \mathrm { h o } } = \| ( \mathbf { M } _ { \mathrm { h } } + \mathbf { M } _ { \mathrm { o } } ) \odot | V _ { \mathrm { h } } - V _ { \mathrm { o } } | \| _ { 2 }$。该损失旨在最小化在预测的接触区域（由掩码 $\mathbf{M}_h$ 和 $\mathbf{M}_o$ 定义）上，人体和物体对应顶点之间的欧氏距离。
        *   <strong>物-地接触 ($L_{of}$):</strong> $L _ { \mathrm { o f } } = \| \mathbf { M } _ { \mathrm { f } } \odot | V _ { \mathrm { o } } | \| _ { 1 }$。该损失旨在让与地面接触的物体顶点（由掩码 $\mathbf{M}_f$ 定义）的高度趋近于零。
        *   <strong>穿透避免 ($L_{pt}$):</strong> $L_{pt} = - \mathbb{E}[|\Phi_0^-(V_h)|]$。该损失利用物体的<strong>符号距离函数 (Signed Distance Function, SDF)</strong> $\Phi_0$。当人体顶点 $V_h$ 进入物体内部时，SDF 值为负。通过最小化负的 SDF 值（即最大化其绝对值），来惩罚穿透现象。

    4.  **计算修正后的噪声:**
        将物理损失 $L_{\mathcal{P}}$ 对 $\mathbf{x}_t$ 求梯度，并加到原始预测噪声上，得到修正后的噪声 $\epsilon'_{\phi}$：
        $$
        \epsilon _ { \phi } ^ { \prime } = \epsilon _ { \phi } ( \pmb { x } _ { t } , t , \pmb { c } ) + \rho \sqrt { 1 - \alpha _ { t } } \nabla _ { \pmb { x } _ { t } } L _ { \mathcal { P } }
        $$
        其中 $\rho$ 是一个控制引导强度的超参数。这个梯度 $\nabla_{\mathbf{x}_t} L_{\mathcal{P}}$ 就是来自物理约束的“引导力”。

    5.  **DDIM 采样步骤:**
        使用修正后的噪声 $\epsilon'_{\phi}$ 来执行 DDIM 的一步采样，得到更“干净”的潜变量 $\mathbf{x}_{t-\Delta t}$。重复此过程，直到 $t=0$。

### 4.2.3. 接触驱动的迭代优化 (Contact-Driven Iterative Refinement)
物理引导的有效性高度依赖于接触区域掩码 $\mathbf{M}_h, \mathbf{M}_o, \mathbf{M}_f$ 的准确性。如果初始预测的接触区域是错的，那么物理引导也会是错的。为了解决这个问题，论文提出了一个外层迭代循环，如 **Algorithm 1** 所示：

```
Algorithm 1: Contact-Driven Iterative Refinement

1: Input: 初始参数 x_0^0, 扩散模型 ε_θ, 图像特征 F, 时间步 t, 条件 c
2: Output: 优化后的参数 x_0^N
3: for n = 0 to N-1 do
4:     // 基于当前参数 x_0^n 采样特征
5:     F_h^n, F_o^n = sample(x_0^n, F)
6:     // 基于新特征更新接触掩码
7:     {M_i}_{i∈{h,o,f}} = Contact(x_0^n, F_h^n, F_o^n)
8:     // 执行一次完整的DDIM反演+引导采样循环，得到优化的参数
9:     x_0^(n+1) ← DDIM_Loop(x_0^n, t, c, ε_θ, {M_i})
10: end for
11: return x_0^N
```

这个算法的核心在于第 7 行和第 9 行。在每次大的迭代 $n$ 中：
1.  首先，根据当前的参数估计 $\mathbf{x}_0^n$，通过一个<strong>接触预测器 (Contact Predictor)</strong>（一个 Transformer 结构）来**重新预测**接触掩码 $\{ \mathbf{M}_i \}$。
2.  然后，将这个**更新后**的、更准确的接触掩码用于下一次的 DDIM 引导采样循环中，以产生更精确的物理引导。
    这个“预测-优化-再预测”的循环迭代 $N$ 次，使得接触预测和姿态优化相互促进，逐步提高重建质量。

### 4.2.4. 扩散模型架构 (Diffusion Model Architecture)
*   **IG-Adapter:** 为了将外部条件有效地融入扩散模型，论文设计了一个名为 **IG-Adapter (Image and Geometry Adapter)** 的模块，如下图（原文 Figure 3）所示。

    ![Figure 3. The overview of IG-Adapter. We introduce an IGAdapter designed to integrate the image feature guidance $c _ { \\mathrm { I } }$ and the geometry feature guidance $c _ { \\mathrm { G } }$ into the diffusion model. The incorporation of observational and geometric awareness enhances the controllability of the model during the inference process.](images/3.jpg)
    *该图像是示意图，展示了IG-Adapter的结构。图中展示了如何将图像特征$c_I$与几何特征$c_G$通过线性层和交叉注意力机制整合到扩散模型中，增强模型在推理过程中的可控性。箭头指向输入和输出，分别标记为`xt`和`xt-1`，强调特征融合对重建过程的重要性。*

    该模块本质上是一个额外的交叉注意力层。它接收两个条件作为输入：
    *   <strong>图像特征条件 ($c_I$):</strong> 从图像主干网络提取的视觉特征 $\mathcal{F}$ 经过平均池化得到。
    *   <strong>几何特征条件 ($c_G$):</strong> 从预训练的可用性感知网络提取的物体几何特征。
        通过这个适配器，模型在去噪的每一步都能同时感知到“图像里看到了什么”和“这个物体是什么样的”，从而增强了生成的可控性和对输入的忠实度。
*   **训练目标:** 扩散模型的训练目标是标准的噪声预测损失，即让网络预测的噪声 $\epsilon_{\theta}$ 尽可能接近添加的真实噪声 $\epsilon$：
    $$
    L _ { \mathrm { DM } } = \mathbb { E } _ { { \pmb { x } } _ { 0 } , { \epsilon } , { t } , { \pmb { c } } _ { \mathrm { I } } , { \pmb { c } } _ { \mathrm { G } } } \| \epsilon - { \epsilon } _ { \theta } ( { \pmb { x } } _ { t } , { t } , { \pmb { c } } _ { \mathrm { I } } , { \pmb { c } } _ { \mathrm { G } } ) \| ^ { 2 }
    $$

# 5. 实验设置

## 5.1. 数据集
*   **BEHAVE [2]:** 这是一个大型的、在自然环境中的人-物交互数据集。它包含 8 名被试与 20 种不同物体进行交互的视频，提供了精确的 3D 人体、物体和接触标注。这是评估 HOI 重建质量的主要基准之一。
*   **InterCap [23]:** 该数据集包含 10 名被试与 10 种不同大小和功能的物体进行交互的数据。它特别关注手部和脚部的接触，提供了多视角 RGB-D 视频和标注。
*   **IMHD [72]:** 这是一个包含 15 名被试在 10 个不同交互场景下的数据集。论文**仅使用该数据集的训练集来训练扩散模型**，目的是利用其多样化的交互数据来增强模型的生成能力和先验知识，而不用于评估。

## 5.2. 评估指标
### 5.2.1. 倒角距离 (Chamfer Distance, CD)
1.  **概念定义:** 倒角距离是一种衡量两个点云之间相似度的度量。它计算从一个点云中的每个点到另一个点云中最近点的平均距离，并在两个方向上都计算后取平均。值越小，表示两个点云越接近，即重建结果越准确。在本文中，分别计算人体网格的倒角距离 `CD_human` 和物体网格的倒角距离 `CD_object`。
2.  **数学公式:**
    $$
    \text{CD}(S_1, S_2) = \frac{1}{|S_1|} \sum_{x \in S_1} \min_{y \in S_2} \|x-y\|_2^2 + \frac{1}{|S_2|} \sum_{y \in S_2} \min_{x \in S_1} \|x-y\|_2^2
    $$
3.  **符号解释:**
    *   $S_1, S_2$: 两个进行比较的点云集合。
    *   `x, y`: 分别是点云 $S_1$ 和 $S_2$ 中的点。
    *   $\min_{y \in S_2} \|x-y\|_2^2$: 点 $x$ 到点云 $S_2$ 中所有点的最小平方欧氏距离。

### 5.2.2. 接触的精确率、召回率和 F-Score (Precision, Recall, and F-Score for Contact)
1.  **概念定义:** 这些指标用于评估重建模型在预测**接触区域**方面的准确性。首先，通过一个阈值（如 5cm）将在物体附近的的人体顶点分类为“接触顶点”。然后将这个预测的接触顶点集合与<strong>真实标注数据 (Ground Truth)</strong> 的接触顶点集合进行比较。
    *   <strong>精确率 (Precision):</strong> 在所有被模型预测为“接触”的顶点中，有多少是真正接触的。高精确率意味着模型预测的接触点很可靠，很少误报。
    *   <strong>召回率 (Recall):</strong> 在所有真正发生接触的顶点中，有多少被模型成功地预测出来了。高召回率意味着模型能找到大部分真实的接触点，很少漏报。
    *   **F-Score:** 精确率和召回率的调和平均数，用于综合评估接触预测的整体性能。
2.  **数学公式:**
    $$
    \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
    $$
    $$
    \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
    $$
    $$
    \text{F-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
    $$
3.  **符号解释:**
    *   TP (True Positive): 真正接触且被预测为接触的顶点数。
    *   FP (False Positive): 非接触但被预测为接触的顶点数。
    *   FN (False Negative): 真正接触但被预测为非接触的顶点数。

## 5.3. 对比基线
*   **PHOSA [68]:** 一种较早的、从单张图像中感知 3D 人-物空间关系的回归方法。
*   **CHORE [59]:** 一种基于优化的 SOTA 方法，通过迭代优化来强制施加接触和物理约束。
*   **CONTHO [37]:** 一种基于回归的 SOTA 方法，使用 Transformer 来建模接触关系。
*   **VisTracker [61]:** 一种基于优化的视频 HOI 跟踪方法。

# 6. 实验结果与分析

## 6.1. 核心结果分析
### 6.1.1. 主要性能对比
以下是原文 Table 1 的结果，展示了 ScoreHOI 与其他 SOTA 方法在 BEHAVE 和 InterCap 数据集上的定量比较。

<table>
<thead>
<tr>
<th>Datasets</th>
<th>Methods</th>
<th>CD<sub>human</sub>↓</th>
<th>CD<sub>object</sub>↓</th>
<th>Contact<sub>rec</sub><sup>p</sup>↑</th>
<th>Contact<sub>rec</sub><sup>r</sup>↑</th>
<th>Contact<sub>rec</sub><sup>F-s</sup>↑</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4">BEHAVE</td>
<td>PHOSA [68]</td>
<td>12.17</td>
<td>26.62</td>
<td>0.393</td>
<td>0.266</td>
<td>0.317</td>
</tr>
<tr>
<td>CHORE [59]</td>
<td>5.58</td>
<td>10.66</td>
<td>0.587</td>
<td>0.472</td>
<td>0.523</td>
</tr>
<tr>
<td>CONTHO [37]</td>
<td>4.99</td>
<td>8.42</td>
<td>0.628</td>
<td>0.496</td>
<td>0.554</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>4.85</b></td>
<td><b>7.86</b></td>
<td><b>0.634</b></td>
<td><b>0.586</b></td>
<td><b>0.609</b></td>
</tr>
<tr>
<td rowspan="4">InterCap</td>
<td>PHOSA [68]</td>
<td>11.20</td>
<td>20.57</td>
<td>0.228</td>
<td>0.159</td>
<td>0.187</td>
</tr>
<tr>
<td>CHORE [59]</td>
<td>7.01</td>
<td>12.81</td>
<td>0.339</td>
<td>0.253</td>
<td>0.290</td>
</tr>
<tr>
<td>CONTHO [37]</td>
<td>5.96</td>
<td>9.50</td>
<td>0.661</td>
<td>0.432</td>
<td>0.522</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>5.56</b></td>
<td><b>8.75</b></td>
<td>0.627</td>
<td><b>0.590</b></td>
<td><b>0.578</b></td>
</tr>
</tbody>
</table>

*   **分析:**
    *   <strong>重建精度 (CD):</strong> ScoreHOI 在人体 (`CD_human`) 和物体 (`CD_object`) 的重建精度上均优于所有基线方法，说明其生成的网格与真实标注数据最接近。
    *   <strong>接触质量 (Contact F-Score):</strong> 最显著的优势体现在接触质量上。在 BEHAVE 数据集上，ScoreHOI 的 `Contact F-Score` 达到了 **0.609**，相比之前的最佳方法 CONTHO (0.554) **提升了约 9.9%**。这主要归功于其强大的物理引导和迭代优化机制。值得注意的是，召回率 ($Contact_rec^r$) 提升非常明显（从 0.496 提升到 0.586），表明 ScoreHOI 能更有效地找到并重建出真实的接触关系，而不仅仅是提高接触预测的准确性。

        下图（原文 Figure 4）的定性结果也直观地展示了 ScoreHOI 的优势。相比 CHORE 和 CONTHO，ScoreHOI 的重建结果（尤其从侧视图看）在物理上更合理，例如人真实地坐在了物体上，手也准确地接触了物体表面。

        ![该图像是多个输入图像与人-物体交互重建结果的对比示意图，展示了前视图和侧视图的不同展示效果，对比了 CHORE、CONTHO 和我们的 ScoreHOI 方法在重建过程中的效果。](images/4.jpg)
        *该图像是多个输入图像与人-物体交互重建结果的对比示意图，展示了前视图和侧视图的不同展示效果，对比了 CHORE、CONTHO 和我们的 ScoreHOI 方法在重建过程中的效果。*

### 6.1.2. 效率对比
以下是原文 Table 2 的结果，比较了不同优化方法的推理效率。

<table>
<thead>
<tr>
<th>Methods</th>
<th>CD<sub>human</sub>↓</th>
<th>CD<sub>object</sub>↓</th>
<th>FPS↑</th>
</tr>
</thead>
<tbody>
<tr>
<td>CHORE [59]</td>
<td>5.58</td>
<td>10.66</td>
<td>0.0035</td>
</tr>
<tr>
<td>VisTracker [61]</td>
<td>5.24</td>
<td>7.89</td>
<td>0.0359</td>
</tr>
<tr>
<td>Ours-Faster</td>
<td>4.87</td>
<td>7.95</td>
<td>2.0080</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>4.85</b></td>
<td><b>7.86</b></td>
<td><b>0.2895</b></td>
</tr>
</tbody>
</table>

*   **分析:**
    *   ScoreHOI 的推理速度（FPS，每秒处理帧数）为 0.2895，比基于 Adam 优化的 CHORE (0.0035 FPS) 快了**近两个数量级**。这证明了将扩散模型作为优化器在效率上的巨大优势。
    *   论文还测试了一个 `Ours-Faster` 版本（迭代次数 $N=2$），其 FPS 达到了 2.0，速度更快，而性能仅有轻微下降。这表明该方法在速度和精度之间有很好的可调性。

## 6.2. 消融实验/参数分析
消融实验用于验证模型中各个组件的有效性。所有实验均在 BEHAVE 数据集上进行。

### 6.2.1. 模块、条件和引导的有效性
以下是原文 Table 3 的结果。

<table>
<thead>
<tr>
<th>Methods</th>
<th>CD<sub>human</sub>↓</th>
<th>CD<sub>object</sub>↓</th>
<th>Contact<sub>rec</sub><sup>p</sup>↑</th>
<th>Contact<sub>rec</sub><sup>r</sup>↑</th>
<th>Contact<sub>rec</sub><sup>F-s</sup>↑</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="6">* Module</td>
</tr>
<tr>
<td>w/o diffusion</td>
<td>5.03</td>
<td>8.48</td>
<td>0.612</td>
<td>0.523</td>
<td>0.588</td>
</tr>
<tr>
<td>w/o CDIR</td>
<td>4.93</td>
<td>7.98</td>
<td>0.628</td>
<td>0.545</td>
<td>0.577</td>
</tr>
<tr>
<td colspan="6">* Condition</td>
</tr>
<tr>
<td>No condition</td>
<td>4.94</td>
<td>8.23</td>
<td>0.626</td>
<td>0.549</td>
<td>0.585</td>
</tr>
<tr>
<td>w/o c<sub>G</sub></td>
<td>4.87</td>
<td>7.99</td>
<td>0.628</td>
<td>0.559</td>
<td>0.591</td>
</tr>
<tr>
<td>w/o c<sub>I</sub></td>
<td>4.88</td>
<td>8.03</td>
<td>0.631</td>
<td>0.566</td>
<td>0.597</td>
</tr>
<tr>
<td colspan="6">* Guidance</td>
</tr>
<tr>
<td>No guidance</td>
<td>4.93</td>
<td>8.01</td>
<td>0.624</td>
<td>0.524</td>
<td>0.570</td>
</tr>
<tr>
<td>w/o L<sub>ho</sub></td>
<td>4.87</td>
<td>7.95</td>
<td>0.632</td>
<td>0.525</td>
<td>0.574</td>
</tr>
<tr>
<td>w/o L<sub>pt</sub></td>
<td>4.87</td>
<td>7.93</td>
<td>0.619</td>
<td>0.567</td>
<td>0.592</td>
</tr>
<tr>
<td>w/o L<sub>of</sub></td>
<td>4.89</td>
<td>7.95</td>
<td>0.631</td>
<td>0.577</td>
<td>0.602</td>
</tr>
<tr>
<td><b>Full model</b></td>
<td><b>4.85</b></td>
<td><b>7.86</b></td>
<td><b>0.634</b></td>
<td><b>0.586</b></td>
<td><b>0.609</b></td>
</tr>
</tbody>
</table>

*   **分析:**
    *   **模块有效性:** 去掉扩散优化 (`w/o diffusion`) 或去掉接触驱动的迭代优化 (`w/o CDIR`) 都会导致性能下降，证明了这两个核心模块的必要性。
    *   **条件有效性:** 不使用任何条件 (`No condition`) 或分别去掉几何条件 (`w/o cG`)、图像条件 (`w/o cI`) 都会损害性能，说明 `IG-Adapter` 引入的图像和几何感知能力对精确重建至关重要。
    *   **物理引导有效性:** 去掉任何一项物理引导（无引导 `No guidance`、无人-物接触 `w/o Lho`、无穿透 `w/o Lpt`、无物-地接触 `w/o Lof`）都会导致相应的指标下降。特别地：
        *   去掉 `L_ho` 会导致接触**召回率**大幅下降，说明模型不再主动寻求接触。
        *   去掉 `L_pt` 会导致接触**精确率**下降，因为会出现不合理的穿透和多余的接触。
    
            下图（原文 Figure 5）定性地展示了物理引导的作用。没有接触引导 `L_ho` 时，人和物体分离；没有穿透引导 `L_pt` 时，手会穿进物体。

            ![Figure 5. Qualitative results for ablation study. Upper row: the ablation study of $L _ { \\mathrm { { h o } } }$ . The inclusion of contact guidance between the $L _ { \\mathrm { h o } }$ . The absence of a penetration penalty results in a notable rise in the occurrence of unreasonable interactions.](images/5.jpg)
            *该图像是插图，展示了物体与人类交互的重建效果。在上半部分的“正面视图”和下半部分的“侧面视图”中，包含多个输入图像及其对应的重建模型。每个视图分别展示了在不使用接触损失（$L_{ho}$和$L_{pt}$）和完整模型的情况下，如何影响交互效果。图中可见不同条件下的小人和物体的相对位置，突显了接触引导在提高交互重建质量方面的重要性。*

### 6.2.2. 优化超参数分析
以下是原文 Table 4 的结果，分析了迭代次数 $N$、噪声水平 $τ$ 和 DDIM 步长 $Δt$ 的影响。

<table>
<thead>
<tr>
<th>N</th>
<th>τ</th>
<th>∆t</th>
<th>CD<sub>human</sub>↓</th>
<th>CD<sub>object</sub>↓</th>
<th>Contact<sub>rec</sub><sup>F-s</sup>↑</th>
</tr>
</thead>
<tbody>
<tr>
<td>10</td>
<td>50</td>
<td>25</td>
<td>5.80</td>
<td>10.70</td>
<td>0.539</td>
</tr>
<tr>
<td>10</td>
<td>50</td>
<td>10</td>
<td>5.22</td>
<td>8.72</td>
<td>0.584</td>
</tr>
<tr>
<td>10</td>
<td>50</td>
<td>5</td>
<td>4.99</td>
<td>8.16</td>
<td>0.569</td>
</tr>
<tr>
<td>2</td>
<td>50</td>
<td>2</td>
<td>4.87</td>
<td>7.95</td>
<td>0.604</td>
</tr>
<tr>
<td>5</td>
<td>50</td>
<td>2</td>
<td>4.86</td>
<td>7.87</td>
<td>0.607</td>
</tr>
<tr>
<td>20</td>
<td>50</td>
<td>2</td>
<td>4.87</td>
<td>7.89</td>
<td>0.610</td>
</tr>
<tr>
<td>10</td>
<td>25</td>
<td>2</td>
<td>4.87</td>
<td>7.95</td>
<td>0.606</td>
</tr>
<tr>
<td>10</td>
<td>100</td>
<td>2</td>
<td>4.88</td>
<td>8.07</td>
<td>0.615</td>
</tr>
<tr>
<td><b>10</b></td>
<td><b>50</b></td>
<td><b>2</b></td>
<td><b>4.85</b></td>
<td><b>7.86</b></td>
<td><b>0.609</b></td>
</tr>
</tbody>
</table>

*   **分析:**
    *   增加迭代次数 $N$ 或噪声水平 $τ$ 通常会提高接触 F-score，因为有更多的优化空间和更强的先验约束。
    *   减小 DDIM 步长 $Δt$（即增加采样步数）可以显著提高性能。
    *   最终，作者选择 $N=10, τ=50, Δt=2$ 作为在性能和效率之间的最佳平衡点。

# 7. 总结与思考

## 7.1. 结论总结
本文提出了一种名为 **ScoreHOI** 的创新框架，用于从单张图像中进行物理上合理的人-物交互三维重建。该方法的核心是将一个预训练的**扩散模型**巧妙地用作一个**基于分数的优化器**。通过在去噪采样过程中引入**物理约束作为引导**，并结合一种**接触驱动的迭代优化策略**，ScoreHOI 能够有效地将一个粗糙的初始估计精炼成一个既符合图像内容，又满足物理真实性（如精确接触、无穿透）的高质量重建结果。实验证明，ScoreHOI 在标准基准上不仅超越了现有最先进的方法，特别是在接触质量方面取得了巨大提升，而且其推理效率远高于传统的优化方法。

## 7.2. 局限性与未来工作
*   **局限性:** 作者坦诚，当前模型的一个主要局限性在于对**已知物体类别**的依赖。由于训练数据集中物体的<strong>标准姿态 (canonical pose)</strong> 是预定义的，当遇到一个训练时未见过、无法定义标准姿态的新物体时，模型可能难以进行有效的优化。
*   **未来工作:** 未来的研究方向将聚焦于解决<strong>无模板的未知物体 (template-free unseen objects)</strong> 的重建问题，进一步提升模型的泛化能力和实用性。

## 7.3. 个人启发与批判
*   **启发:**
    1.  **扩散模型作为优化器:** 这篇论文最亮眼的启发是将扩散模型从一个“生成器”的角色转变为一个“优化器”。这为许多计算机视觉和图形学的优化问题提供了一个全新的、强大的解决思路。凡是可以被一个粗糙估计+后续优化的任务，似乎都可以尝试引入扩散先验来指导优化过程，使其在数据流形上进行，从而避免陷入不合理的解空间。
    2.  **引导与迭代的结合:** 物理引导和迭代优化的结合非常精妙。通过迭代地更新引导信号（接触掩码），模型能够动态地修正自己的优化方向，形成一个正反馈循环。这种“边走边看边调整”的策略对于解决复杂的、多约束的优化问题极具借鉴意义。
    3.  **平衡多重目标:** 该框架优雅地平衡了三个核心目标：与图像的**视觉一致性**（通过图像条件 $c_I$）、与数据分布的**先验一致性**（通过扩散模型本身）以及**物理一致性**（通过物理引导 $L_{\mathcal{P}}$）。

*   **批判与思考:**
    1.  **对初始估计的敏感度:** 论文从一个初始回归结果出发进行优化，但没有深入探讨该方法对初始估计质量的敏感度。如果初始估计偏差极大（例如，人与物体完全分离或姿态完全错误），ScoreHOI 是否还能有效地将其“拉回”到正确的解？优化的“吸引域”有多大是一个值得探究的问题。
    2.  **泛化能力的挑战:** 正如作者所指出的，对已知物体类别和标准姿态的依赖是其主要软肋。虽然引入了“可用性感知网络”来提取通用几何特征，但这可能仍然不足以应对形状、拓扑结构千变万化的真实世界物体。实现真正的“in-the-wild”应用，需要解决对未知物体的零样本或少样本重建问题。
    3.  **物理约束的局限性:** 当前的物理约束（接触、穿透、地面）相对简单。更复杂的物理现象，如力学平衡（例如，人推箱子时，力和反作用力）、物体形变等，尚未被建模。将更丰富的物理引擎知识融入扩散引导过程，可能是未来一个有趣的方向。