# 1. 论文基本信息

## 1.1. 标题
<strong>通过自重采样实现自回归视频扩散的端到端训练 (End-to-End Training for Autoregressive Video Diffusion via Self-Resampling)</strong>

论文标题直接点明了研究的核心：
*   **研究对象:** 自回归视频扩散模型 (Autoregressive Video Diffusion Models)。
*   **核心目标:** 实现端到端训练 (End-to-End Training)，即无需依赖外部教师模型或分阶段训练。
*   **核心方法:** 通过一种名为“自重采样 (Self-Resampling)”的机制来实现这一目标。

## 1.2. 作者
*   **作者列表:** Yuwei Guo, Ceyuan Yang, Hao He, Yang Zhao, Meng Wei, Zhenheng Yang, Weilin Huang, Dahua Lin。
*   **隶属机构:**
    *   香港中文大学 (The Chinese University of Hong Kong)
    *   字节跳动 Seed 团队 (ByteDance Seed)
    *   字节跳动 (ByteDance)
*   **背景分析:** 作者团队来自顶尖学术机构和业界领先的人工智能实验室，尤其是在计算机视觉和生成模型领域有深厚积累。例如，第一作者 Yuwei Guo 和通讯作者 Ceyuan Yang 之前在 `AnimateDiff`、`Long Context Tuning for Video Generation` 等知名工作中已有合作，表明他们在视频生成领域具备丰富的研究经验。

## 1.3. 发表期刊/会议
论文正文中标注的发表日期为2025年，这表明它是一篇提交给未来某个顶级会议（如 CVPR, ICCV, NeurIPS, ICLR 等）的预印本 (preprint)。当前版本发布在 **arXiv** 上，这是一个广泛使用的学术论文预印本平台。在计算机科学领域，将工作发布到 arXiv 是为了在同行评审之前快速分享研究成果，这已成为一种标准做法。

## 1.4. 发表年份
*   **预印本发布日期:** 2025-12-17 (根据论文元数据)
*   **论文内标注日期:** December 18, 2025

## 1.5. 摘要
自回归视频扩散模型在世界模拟方面展现出巨大潜力，但它们容易受到因训练与测试不匹配而产生的<strong>暴露偏差 (exposure bias)</strong> 的影响。虽然近期工作通过后训练 (post-training) 方式解决此问题，但通常依赖于一个双向的教师模型或在线判别器。为了实现一个<strong>端到端 (end-to-end)</strong> 的解决方案，我们引入了 <strong>重采样强制 (Resampling Forcing)</strong>，这是一个无需教师模型的框架，能够从零开始大规模训练自回归视频模型。我们方法的核心是一种<strong>自重采样 (self-resampling)</strong> 方案，它在训练期间模拟了模型在历史帧上的推理时误差。在这些退化的历史帧条件下，一个稀疏的因果掩码强制了时间因果性，同时通过帧级扩散损失实现了并行训练。为了促进高效的长时域生成，我们进一步引入了<strong>历史路由 (history routing)</strong>，这是一种无参数机制，能为每个查询动态检索最相关的 top-k 个历史帧。实验表明，我们的方法取得了与基于蒸馏的基线相当的性能，并且由于在原生长度上进行训练，在较长视频上表现出更优的时间一致性。

## 1.6. 原文链接
*   **原文链接:** https://arxiv.org/abs/2512.15702
*   **PDF 链接:** https://arxiv.org/pdf/2512.15702v1.pdf
*   **发布状态:** 预印本 (Preprint)。

    ---

# 2. 整体概括

## 2.1. 研究背景与动机
### 2.1.1. 核心问题
论文旨在解决<strong>自回归 (Autoregressive)</strong> 视频生成模型中的一个根本性难题：<strong>暴露偏差 (Exposure Bias)</strong>。

*   **自回归生成范式:** 像语言模型一样，视频也可以一帧一帧地生成。当前帧的生成依赖于所有先前已生成的帧。这种模式非常符合物理世界的因果规律（未来由过去决定），因此在世界模拟、游戏仿真等领域潜力巨大。
*   **暴露偏差的产生:**
    1.  **训练阶段:** 为了高效并行训练，模型通常采用一种名为 <strong>教师强制 (Teacher Forcing)</strong> 的策略。即在预测第 $i$ 帧时，提供给模型的历史信息是 <strong>完全正确、无误差的真实视频帧 (Ground Truth)</strong>。
    2.  <strong>推理（生成）阶段:</strong> 模型在现实世界中生成视频时，它无法接触到未来的真实帧。因此，在生成第 $i$ 帧时，它依赖的历史信息是模型**自己**在前 `i-1` 步中生成的帧。
    3.  **训练-测试不匹配:** 由于模型不完美，其生成的帧必然存在或多或少的误差。当这些带有误差的帧被作为后续帧的输入时，误差会像滚雪球一样不断累积，最终导致视频质量急剧下降，甚至完全崩溃。这就是暴露偏差。

### 2.1.2. 现有研究的挑战与空白 (Gap)
现有的解决方案大多不是<strong>端到端 (end-to-end)</strong> 的，存在以下问题：

1.  <strong>依赖后训练 (Post-training):</strong> 像 `Self Forcing` [32] 这样的方法，需要先用一个基础的自回归模型生成完整视频，然后通过一个<strong>预训练好的、强大的双向教师模型 (bidirectional teacher model)</strong> 进行蒸馏，或者使用一个<strong>在线判别器 (online discriminator)</strong> 进行对抗性训练，来“校正”生成的视频分布。
2.  **可扩展性差:** 这些方法需要一个额外的、通常比学生模型大得多的教师模型，或者复杂的对抗训练流程，这使得从零开始大规模训练变得困难且昂贵。
3.  **破坏因果性:** 使用双向教师模型（可以同时看到过去和未来的帧）来指导一个只能看到过去的因果学生模型，可能会无意中将未来的信息“泄露”给学生，从而破坏了严格的时间因果性。
4.  **长视频处理效率低:** 传统的自回归模型在处理长视频时，需要关注所有历史帧，导致注意力计算的成本随视频长度线性增长，这在实际应用中是不可持续的。

### 2.1.3. 本文的切入点
论文的创新思路是：**与其在事后修复错误，不如在训练时就让模型学会如何应对自己的错误。**

具体而言，作者提出了一种名为 <strong>重采样强制 (Resampling Forcing)</strong> 的端到端训练框架。其核心思想是在训练过程中，**主动模拟模型在推理时会犯的错误**，并将这些“有瑕疵”的历史帧作为输入，然后要求模型依然能生成正确的下一帧。这样训练出来的模型，对输入中的噪声和误差具有更强的鲁棒性，从而在实际生成长视频时能有效抑制误差累积。

## 2.2. 核心贡献/主要发现
1.  **提出 `Resampling Forcing` 框架:** 这是一个全新的、无需教师模型的端到端训练框架，用于解决自回归视频扩散模型中的暴露偏差问题。它简单、可扩展，且能从零开始训练模型。
2.  **设计 `Self-Resampling` 机制:** 这是框架的核心。通过让模型在训练时对历史帧进行“再生成”（即部分加噪再用自身去噪），巧妙地模拟了推理时会产生的误差。这种自生成的“负样本”让模型学会了在不完美条件下进行稳健预测。
3.  **引入 `History Routing` 机制:** 为了解决长视频生成的效率问题，论文提出了一种无参数的动态注意力机制。它不依赖于固定的滑动窗口，而是根据内容相关性，为当前帧动态选择最重要的 $k$ 个历史帧进行关注，从而在保持长程依赖的同时，将注意力复杂度控制在常数级别。
4.  **优异的实验结果:**
    *   **性能媲美基线:** 在视频生成质量上，该方法达到了与依赖强大教师模型进行蒸馏的先进方法（如 `Self Forcing`）相当的水平。
    *   **更强的时间一致性:** 由于模型在完整的长视频上进行训练，相比于那些将长视频切片处理的方法，它在长时域下展现出更好的时间连贯性和因果性，例如物理现象（如倒水时液位上升）的正确模拟。

        ---

# 3. 预备知识与相关工作

## 3.1. 基础概念
### 3.1.1. 扩散模型 (Diffusion Models)
扩散模型是一类强大的深度生成模型。其核心思想分为两个过程：
*   <strong>前向过程 (Forward Process):</strong> 对一张真实的图像（或视频帧）逐步、多次地添加少量高斯噪声，直到它最终变成完全的随机噪声。这个过程是固定的，不需要学习。
*   <strong>反向过程 (Reverse Process):</strong> 训练一个神经网络（通常是 U-Net 或 Transformer 架构），让它学会如何“撤销”前向过程。即从一个纯噪声图像开始，逐步、多次地去除噪声，最终还原出一张清晰的图像。
    在视频生成中，这个过程被应用于每一帧，而模型在去噪时会额外接收文本、历史帧等作为条件信息。

### 3.1.2. 自回归模型 (Autoregressive Models)
自回归模型是一种序列生成模型，它将序列的联合概率分布分解为一系列条件概率的乘积。对于一个序列 $x_1, x_2, \dots, x_N$，其概率为：
$$
p(x_{1:N}) = p(x_1) \cdot p(x_2 | x_1) \cdot p(x_3 | x_{1:2}) \cdots p(x_N | x_{<N}) = \prod_{i=1}^{N} p(x_i | x_{<i})
$$
这意味着生成第 $i$ 个元素时，需要依赖所有在它之前的元素。这种严格的顺序和因果依赖性是其核心特征。

### 3.1.3. 教师强制 (Teacher Forcing)
这是训练自回归模型时常用的一种高效策略。在预测序列中的第 $i$ 个元素 $x_i$ 时，模型接收的输入条件是 <strong>真实的 (ground-truth)</strong> 前缀 $x_{<i}$，而不是模型自己在前几步生成的有偏差的输出。这样做的好处是：
*   **并行计算:** 可以在一个批次内并行地计算所有时间步的损失，因为每个时间步的输入都是已知的，无需等待前一个时间步的生成结果。
*   **训练稳定:** 始终使用正确的数据作为输入，可以提供稳定且准确的梯度信号，有助于模型快速收敛。
    然而，这也直接导致了前述的<strong>暴露偏差 (Exposure Bias)</strong> 问题。

### 3.1.4. 注意力机制 (Attention Mechanism)
注意力机制是 Transformer 模型的核心组件，它允许模型在处理序列数据时，动态地为输入序列的不同部分分配不同的“注意力权重”。其基本计算过程如下：
对于一个查询 (Query, $Q$)，以及一系列的键 (Key, $K$) 和值 (Value, $V$)，注意力输出的计算公式为：
$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
*   **`Q, K, V`**: 分别是查询、键、值的矩阵。在自注意力 (self-attention) 中，它们都由同一输入序列线性变换而来。
*   $QK^T$: 计算每个查询与所有键之间的相似度分数。
*   $\sqrt{d_k}$: 缩放因子，其中 $d_k$ 是键向量的维度。用于稳定梯度。
*   $\mathrm{softmax}$: 将相似度分数归一化为概率分布（即注意力权重）。
*   $V$: 将注意力权重应用于对应的值，得到加权和输出。

    在视频中，一个像素块（或称 patch）的 `Query` 会去和所有其他（或指定的）像素块的 `Key` 计算相似度，从而决定应该“关注”哪些区域的信息。

## 3.2. 前人工作
*   <strong>双向视频生成 (Bidirectional Video Generation):</strong> 这类模型（如 Sora、Lumiere）在生成视频时，每一帧都可以关注过去和未来的所有帧。它们通常能生成整体质量和一致性非常高的短视频，但由于破坏了时间因果性，不适用于需要严格按时间顺序预测未来的世界模拟任务。
*   <strong>自回归视频生成 (Autoregressive Video Generation):</strong>
    *   **早期方法:** 直接使用教师强制进行训练，导致长视频生成时质量下降严重 [20, 31, 84]。
    *   **噪声注入:** 一些工作尝试在历史帧中注入少量噪声来模拟推理时的不完美 [62, 67]，但这与模型真实的误差模式可能不匹配。
    *   **Self Forcing [32, 42]:** 这是最相关的先前工作。它采用一种**后训练**策略，先用自回归模型生成视频，然后用一个强大的**双向教师模型**进行<strong>蒸馏 (distillation)</strong>，或者用判别器进行对抗训练。这种方法有效，但如前所述，它不具备端到端、从零训练的可行性，且可能泄露未来信息。
*   <strong>高效注意力机制 (Efficient Attention for Video Generation):</strong>
    *   <strong>稀疏注意力掩码 (Sparse Attention Masks):</strong> 通过预先定义的规则，限制每个 token 只能关注一部分其他 token，例如只关注邻近的或按一定步长采样的 token [40, 59]。
    *   <strong>动态路由 (Dynamic Routing):</strong> 受大语言模型中 `Mixture of Experts` 或 `Mixture of Block Attention` [46] 的启发，让模型动态地为每个查询选择性地关注一部分键和值，而不是使用固定的模式。本文的 `History Routing` 就属于此类。

## 3.3. 技术演进
视频生成技术从早期的 GAN 和 VAE，发展到如今以扩散模型为主流的时代。在架构上，从基于卷积的 U-Net 演进到基于 Transformer 的 DiT 架构，实现了更好的可扩展性。在生成范式上，主要分为两条路线：
1.  **双向/联合生成:** 以 Sora 为代表，追求极致的短视频生成质量和时空一致性。
2.  **自回归/因果生成:** 追求严格的时间因果性，面向世界模型和交互式应用。
    本文的工作处于第二条路线，并致力于解决该路线的核心瓶颈——**暴露偏差和长时域效率**，其创新之处在于提出了一种**端到端**的解决方案，摆脱了对外部教师模型的依赖。

## 3.4. 差异化分析
与最相关的 `Self Forcing` 相比，本文的 `Resampling Forcing` 核心区别在于：

| 特性 | Self Forcing [32] | Resampling Forcing (本文) |
| :--- | :--- | :--- |
| **训练范式** | 后训练 (Post-training)，分阶段 | <strong>端到端 (End-to-end)</strong>，一体化 |
| **外部依赖** | **需要**一个预训练好的、强大的双向教师模型 | **无需**任何外部教师模型或判别器 |
| **误差来源** | 通过蒸馏，学习匹配教师模型的输出分布 | 通过**自重采样**，学习应对**自身**产生的误差 |
| **因果性** | 可能因双向教师的指导而**受损** | **严格保持**时间因果性 |
| **可扩展性** | 较低，受限于教师模型的获取和计算开销 | **更高**，训练流程简单，易于大规模部署 |

---

# 4. 方法论

## 4.1. 方法原理
本文方法的核心直觉是：**为了让模型在充满“错误”的推理环境中生存下来，就必须在训练时让它提前适应这种环境。**

传统的教师强制让模型在“无菌环境”（只有真实数据）中训练，导致其对推理时的“污染”（模型自身误差）毫无抵抗力。本文提出的 `Resampling Forcing` 则是在训练中主动、可控地制造这种“污染”，从而锻炼模型的鲁棒性。

这个过程可以类比为给一个人注射疫苗：疫苗（模拟的误差）本身是减毒或灭活的，但足以刺激免疫系统（模型）产生抗体（鲁棒性），从而在未来面对真正的病毒（推理时的误差累积）时能够有效应对。

## 4.2. 核心方法详解 (逐层深入)
### 4.2.1. 背景：自回归视频扩散模型

首先，论文回顾了自回归视频扩散模型的标准定义。给定一个条件 $c$（例如文本描述），一个包含 $N$ 帧的视频序列 $\pmb{x}^{1:N}$ 的联合概率分布可以被分解为：
$$
p ( \pmb { x } ^ { 1 : N } | c ) = \prod _ { i = 1 } ^ { N } p ( \pmb { x } ^ { i } | \pmb { x } ^ { < i } , c )
$$
**符号解释:**
*   $\pmb{x}^i$: 视频的第 $i$ 帧。
*   $\pmb{x}^{<i}$: 视频的前 `i-1` 帧，即历史帧。
*   $p(\cdot)$: 概率分布。

    这个公式表明，生成整个视频的过程被拆解为 $N$ 个独立的步骤，每一步都是在给定历史帧 $\pmb{x}^{<i}$ 和条件 $c$ 的情况下，生成当前帧 $\pmb{x}^i$。

对于每一帧的生成，模型采用扩散模型的反向过程。从一个标准高斯噪声 $\pmb{x}_1^i \sim \mathcal{N}(\mathbf{0}, I)$ 开始，通过求解一个常微分方程 (ODE) 来逐步去噪，最终得到清晰的帧 $\pmb{x}^i$ (即在时刻 $t=0$ 的 $\pmb{x}_0^i$)。这个过程的数学表达为：
$$
\pmb { x } ^ { i } = \pmb { x } _ { 1 } ^ { i } + \int _ { 1 } ^ { 0 } \pmb { v } _ { \theta } ( \pmb { x } _ { t } ^ { i } , \pmb { x } ^ { < i } , t , c ) \mathrm { d } t
$$
**符号解释:**
*   $\pmb{x}_t^i$: 第 $i$ 帧在去噪时间步 $t$ 时的带噪状态。$t=1$ 是纯噪声，$t=0$ 是清晰帧。
*   $\pmb{v}_\theta(\cdot)$: 一个由参数 $\theta$ 构成的神经网络（通常是 DiT），它预测在当前状态下应该朝哪个方向去噪（即速度场）。这个网络同时接收带噪的当前帧 $\pmb{x}_t^i$、**干净的历史帧** $\pmb{x}^{<i}$、时间步 $t$ 和条件 $c$ 作为输入。
*   $\int_1^0 \dots \mathrm{d}t$: 从时间 $t=1$ 到 $t=0$ 的积分，表示整个去噪过程。在实践中，通过数值求解器（如欧拉法）分步近似。

    在标准的教师强制训练中，模型的目标是学习预测正确的速度场。论文使用了<strong>流匹配 (Flow Matching)</strong> 的目标函数。首先，任意时刻 $t$ 的带噪样本 $\pmb{x}_t^i$ 是由真实帧 $\pmb{x}^i$ 和噪声 $\epsilon^i$ 线性插值得到的：
$$
\pmb { x } _ { t } ^ { i } = ( 1 - t ) \cdot \pmb { x } ^ { i } + t \cdot \pmb { \epsilon } ^ { i }
$$
**符号解释:**
*   $\epsilon^i \sim \mathcal{N}(0, I)$: 与真实帧 $\pmb{x}^i$ 同样大小的标准高斯噪声。
*   $t \in [0, 1]$: 时间步，表示噪声水平。

    此时，从 $\pmb{x}^i$ 到 $\epsilon^i$ 的理想速度场是 $\epsilon^i - \pmb{x}^i$。因此，训练的目标就是让网络 $\pmb{v}_\theta$ 的预测尽可能接近这个理想速度场。损失函数定义为：
$$
\mathcal{L} = \mathbb{E}_{i, t, \boldsymbol{x}, \boldsymbol{\epsilon}} \left[ \left\| (\boldsymbol{\epsilon}^i - \boldsymbol{x}^i) - \boldsymbol{v}_{\boldsymbol{\theta}} (\boldsymbol{x}_t^i, \boldsymbol{x}^{<i}, t, \boldsymbol{c}) \right\|_2^2 \right]
$$
**符号解释:**
*   $\mathbb{E}[\cdot]$: 期望值，表示在所有可能的视频、帧、时间步和噪声上取平均。
*   $\|\cdot\|_2^2$: L2 范数的平方，即均方误差损失。

    下图（原文 Figure 2）直观地展示了教师强制训练下，推理时误差累积的过程。

    ![Figure 2 Error Accumulation. Top: Models trained with ground truth input add and compound errors autoregressively. Bottom: We train the model on imperfect input with simulated model errors, stabilizing the long-horizon autoregressive generation. The gray circle represents the closest match in the ground truth distribution.](images/2.jpg)
    *该图像是示意图，展示了单步生成与自回归生成的差异。上部分展示了教师强制方法中的历史帧与预测的关系，以及随之产生的累积错误；下部分展示了本研究提出的错误模拟方法，显示了有界错误的生成过程。图中标识了推断步骤和模型误差。*

### 4.2.2. 核心方法：重采样强制 (Resampling Forcing)
为了解决误差累积问题，本文的核心思想是在训练时就让模型接触到“有瑕疵”的历史帧。

#### 4.2.2.1. 模拟推理时误差：自重采样 (Self-Resampling)

如何生成这些“有瑕疵”的历史帧？答案是：**让模型自己来生成**。这个过程被称为**自重采样**，具体步骤如下（参考下图 a 部分）：

![该图像是图示，展示了自回归重采样、并行训练和因果掩码的过程。图(a)说明了通过因果 DiT 进行自回归重采样的步骤，图(b)展示了并行训练中的扩散损失，而图(c)展示了因果掩码的结构。这些方法旨在改善视频扩散模型的训练和生成效果。](images/3.jpg)
*该图像是图示，展示了自回归重采样、并行训练和因果掩码的过程。图(a)说明了通过因果 DiT 进行自回归重采样的步骤，图(b)展示了并行训练中的扩散损失，而图(c)展示了因果掩码的结构。这些方法旨在改善视频扩散模型的训练和生成效果。*

对于一段训练视频中的第 $i$ 帧，它的“瑕疵版” $\tilde{\pmb{x}}^i$ 是这样生成的：
1.  **部分加噪:** 取出真实的、干净的第 $i$ 帧 $\pmb{x}^i$。随机选择一个模拟时间步 $t_s \in (0, 1)$。根据公式 (3) 将 $\pmb{x}^i$ 加噪到 $t_s$ 时刻，得到带噪的 $\pmb{x}_{t_s}^i$。
2.  **自回归去噪:** 使用**当前正在训练的模型** $\pmb{v}_\theta$，从 $\pmb{x}_{t_s}^i$ 开始，执行从 $t_s$ 到 `0` 的反向去噪过程。**关键点在于**，此过程依赖的历史帧是**之前已经生成好的“瑕疵版”历史** $\tilde{\pmb{x}}^{<i}$。

    这个自回归去噪生成 $\tilde{\pmb{x}}^i$ 的过程可以用以下公式精确描述：
$$
\tilde { \pmb { x } } ^ { i } = \pmb { x } _ { t _ { s } } ^ { i } + \int _ { t _ { s } } ^ { 0 } \pmb { v } _ { \theta } ( \pmb { x } _ { t } ^ { i } , \tilde { \pmb { x } } ^ { < i } , t , c ) \mathrm { d } t
$$
**符号解释:**
*   $\tilde{\pmb{x}}^i$: 第 $i$ 帧的“瑕疵”或“退化”版本。
*   $t_s$: 模拟时间步，控制了模拟误差的强度。$t_s$ 越大，噪声越多，模型自由发挥空间越大，产生的误差也可能越大。
*   $\tilde{\pmb{x}}^{<i}$: 前 `i-1` 帧的“瑕疵”版本集合。
*   **重要:** 整个公式 (5) 的计算过程<strong>不进行梯度反向传播 (detached from gradient backpropagation)</strong>。这是为了防止模型走捷径，例如学会生成一个非常容易被自己解码的中间状态，而不是真正学习对一般性误差的鲁棒性。

    通过这个过程，$\tilde{\pmb{x}}^i$ 就包含了模型在当前阶段会犯的两种典型错误：
*   <strong>帧内生成误差 (intra-frame errors):</strong> 来自不完美的去噪过程（从 $t_s$ 到 0）。
*   <strong>帧间累积误差 (inter-frame errors):</strong> 来自于它所依赖的、本身就有瑕疵的历史 $\tilde{\pmb{x}}^{<i}$。

#### 4.2.2.2. 模拟时间步的采样 (Sampling Simulation Timestep)
$t_s$ 的选择很关键。
*   如果 $t_s$ 太小（接近0），$\tilde{\pmb{x}}^i$ 会非常接近真实的 $\pmb{x}^i$，模拟的误差太弱，接近于教师强制，无法有效抑制误差累积。
*   如果 $t_s$ 太大（接近1），模型去噪的起点几乎是纯噪声，生成的 $\tilde{\pmb{x}}^i$ 可能会与原始内容相差甚远，导致内容漂移 (content drifting)。

    因此，作者选择从一个集中在中间值、抑制两极的分布中采样 $t_s$。他们使用了<strong>对数正态分布 (Logit-Normal Distribution)</strong>：
$$
\mathrm { l o g i t } ( t _ { s } ) \sim \mathcal { N } ( 0 , 1 )
$$
其中 `logit` 函数是 `sigmoid` 函数的反函数。然后，为了能更灵活地控制 $t_s$ 的分布偏向，引入了一个偏移参数 $s$ 对其进行变换：
$$
{ t _ { s } } \gets s \cdot { t _ { s } } / \left( { 1 + ( s - 1 ) \cdot { t _ { s } } } \right)
$$
在实验中，作者设置 $s < 1$，这会使得 $t_s$ 的采样更偏向于低噪声区域（即较小的 $t_s$ 值），鼓励模型在保持历史保真度的同时进行误差修正。

#### 4.2.2.3. 训练流程总结

完整的 `Resampling Forcing` 训练流程（如 Algorithm 1 所示）可以总结为：
1.  <strong>教师强制预热 (Warmup):</strong> 在训练初期，模型非常不稳定，产生的误差接近随机噪声。此时直接进行自重采样效果不好。因此，先用标准的教师强制训练一段时间，让模型具备基本的生成能力。
2.  <strong>切换到 <code>Resampling Forcing</code>:</strong>
    *   <strong>步骤 A (误差模拟):</strong> 对于一个训练视频 $\pmb{x}^{1:N}$，通过公式 (5) 的自回归重采样过程，生成一个完整的“瑕疵版”视频 $\tilde{\pmb{x}}^{1:N}$。此过程不计算梯度。
    *   <strong>步骤 B (模型训练):</strong> 使用“瑕疵版”视频 $\tilde{\pmb{x}}^{1:N}$ 作为历史条件，同时使用**原始的、干净的**视频 $\pmb{x}^{1:N}$ 作为预测目标，通过公式 (4) 计算损失并更新模型参数 $\theta$。这一步可以像教师强制一样并行计算所有帧。

        以下是论文中提供的算法伪代码：

**Algorithm 1 Resampling Forcing**

<table>
<tr><td><b>Require:</b> Video Dataset D</td></tr>
<tr><td><b>Require:</b> Shift Parameter s</td></tr>
<tr><td><b>Require:</b> Autoregressive Video Diffusion Model v<sub>θ</sub>(·)</td></tr>
<tr><td>1: <b>while</b> not converged <b>do</b></td></tr>
<tr><td>2: &nbsp;&nbsp;&nbsp;&nbsp; t<sub>s</sub> ∼ LogitNormal(0, 1) &nbsp;&nbsp;<span style="color:gray;">// sample simulation timestep</span></td></tr>
<tr><td>3: &nbsp;&nbsp;&nbsp;&nbsp; t<sub>s</sub> ← s · t<sub>s</sub> / (1 + (s − 1) · t<sub>s</sub>) &nbsp;&nbsp;<span style="color:gray;">// shift timestep (equation (7))</span></td></tr>
<tr><td>4: &nbsp;&nbsp;&nbsp;&nbsp; Sample video and condition (x<sup>1:N</sup>, c) ∼ D</td></tr>
<tr><td>5: &nbsp;&nbsp;&nbsp;&nbsp; <b>with</b> gradient disabled <b>do</b></td></tr>
<tr><td>6: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>for</b> i = 1 to N <b>do</b> &nbsp;&nbsp;<span style="color:gray;">// autoregressive resampling</span></td></tr>
<tr><td>7: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\tilde{x}^i \leftarrow x_{t_s}^i + \int_{t_s}^0 v_\theta(x_t^i, \tilde{x}^{<i}, t, c) dt$ &nbsp;&nbsp;<span style="color:gray;">// using numerical solver and KV cache (equation (5))</span></td></tr>
<tr><td>8: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>end for</b></td></tr>
<tr><td>9: &nbsp;&nbsp;&nbsp;&nbsp; <b>end with</b></td></tr>
<tr><td>10: &nbsp;&nbsp;&nbsp;&nbsp; Sample training timestep t<sup>i</sup></td></tr>
<tr><td>11: &nbsp;&nbsp;&nbsp;&nbsp; Sample $\epsilon^i \sim \mathcal{N}(0, I)$</td></tr>
<tr><td>12: &nbsp;&nbsp;&nbsp;&nbsp; $x_t^i \leftarrow (1-t^i) \cdot x^i + t^i \cdot \epsilon^i$</td></tr>
<tr><td>13: &nbsp;&nbsp;&nbsp;&nbsp; `L \leftarrow \frac{1}{N} \sum_{i=1}^N \|(\epsilon^i - x^i) - v_\theta(x_t^i, \tilde{x}^{<i}, t^i, c)\|_2^2` &nbsp;&nbsp;<span style="color:gray;">// parallel training with causal mask (equation (4))</span></td></tr>
<tr><td>14: &nbsp;&nbsp;&nbsp;&nbsp; Update θ with gradient descent</td></tr>
<tr><td>15: <b>end while</b></td></tr>
<tr><td>16: <b>return</b> θ</td></tr>
</table>

### 4.2.3. 高效长视频生成：历史路由 (History Routing)

为了解决自回归生成中历史上下文不断增长导致的计算瓶颈，论文提出了一种动态稀疏注意力机制——<strong>历史路由 (History Routing)</strong>。

其核心思想是，在为当前帧的某个 `token` 计算注意力时，不再关注所有历史帧，而是只选择**最相关**的 $k$ 个历史帧进行关注。如下图（原文 Figure 4）所示：

![Figure 4 History Routing Mechanism. Our routing mechanism dynamically selects the top- $k$ important frames to attend. In this illustration, we show a $k = 2$ example, where only the 1st and 3rd frames are selected for the 4th frame's query token $\\mathbf { q } _ { 4 }$ .](images/4.jpg)
*该图像是一个示意图，展示了历史路由机制。图中，Top-K Router 动态选择了与当前帧相关的前 $k$ 个重要帧，通过查询令牌 $q_4$ 进行注意力机制处理，以实现高效的视频生成。*

具体实现如下：
1.  **选择标准:** 对于第 $i$ 帧中的一个查询 `token` $\pmb{q}_i$，它与第 $j$ 个历史帧 $(j < i)$ 的相关性，通过 $\pmb{q}_i$ 与该历史帧所有 `Key` 向量的描述符 $\phi(\pmb{K}_j)$ 的点积来衡量。作者采用 `mean-pool` 作为描述符函数 $\phi(\cdot)$，即取该帧所有 `Key` 向量的平均值。这是一个无参数且高效的选择。
2.  **Top-k 选择:** 根据上述相关性得分，为 $\pmb{q}_i$ 选出得分最高的 $k$ 个历史帧。这个选择过程用公式表示为：
    $$
    \Omega ( \pmb { q } _ { i } ) = \arg \operatorname* { m a x } _ { \Omega ^ { * } } \sum _ { j \in \Omega ^ { * } } \left( \pmb { q } _ { i } ^ { \top } \phi ( \pmb { K } _ { j } ) \right)
    $$
    **符号解释:**
    *   $\Omega(\pmb{q}_i)$: 为查询 $\pmb{q}_i$ 选出的 $k$ 个历史帧的索引集合。
    *   $\phi(\pmb{K}_j)$: 第 $j$ 个历史帧的 `Key` 向量描述符。
3.  **稀疏注意力计算:** 最终的注意力计算只在选出的 $k$ 个帧上进行：
    $$
    \mathrm { A t t e n t i o n } ( q _ { i } , K _ { < i } , V _ { < i } ) = \mathrm { S o f t m a x } \left( \frac { q _ { i } K _ { \Omega ( q _ { i } ) } ^ { \top } } { \sqrt { d } } \right) \cdot V _ { \Omega ( q _ { i } ) }
    $$
    **符号解释:**
    *   $K_{\Omega(q_i)}$ 和 $V_{\Omega(q_i)}$: 分别是只包含被选中的 $k$ 个历史帧的 `Key` 和 `Value` 向量。

        这种机制将每个 `token` 的注意力计算复杂度从与历史长度 $L$ 相关的 $\mathcal{O}(L)$ 降低到了常数 $\mathcal{O}(k)$，极大地提升了长视频生成的效率。同时，由于选择是动态的、内容感知的，它比固定的滑动窗口更能保留重要的长程依赖。

---

# 5. 实验设置

## 5.1. 数据集
论文没有明确指定用于训练的公开数据集名称，但描述了其数据特征和处理方式：
*   **数据来源:** 训练数据为 5 秒和 15 秒的视频片段。
*   **分辨率:** 视频分辨率为 $480 \times 832$。
*   **评估数据:** 评估时，使用了 **VBench** [34] 基准测试。VBench 是一个全面的视频生成模型评估套件，它包含一系列精心设计的文本提示 (prompts)，覆盖了不同的场景、主体、动作和风格，用于生成视频并进行多维度评估。

    选择在不同长度（5秒和15秒）的视频上训练，是为了验证方法在处理和生成长视频方面的能力和优势。

## 5.2. 评估指标
论文使用了 VBench 提供的自动化评估指标，主要分为三类：`Temporal Quality` (时间质量), `Visual Quality` (视觉质量), 和 `Text Alignment` (文本对齐)。

### 5.2.1. 时间质量 (Temporal Quality)
*   **概念定义:** 该指标衡量视频在时间维度上的连贯性和流畅性，例如动作是否自然、帧与帧之间有无闪烁或突变。VBench 中主要通过 `Temporal Consistency` (时间一致性) 来量化，它计算视频帧之间特征表示的抖动程度。抖动越小，时间质量越高。
*   **数学公式:** VBench 使用 `CLIP` 特征空间中的 `Cumulative Matching and Motion Dissonance (CMMD)` 等指标。一个简化的概念性公式可以表示为帧间特征距离的变化：
    $$
    \text{Temporal Quality} \propto \frac{1}{N-1} \sum_{i=1}^{N-1} \text{distance}(f(\mathbf{x}_i), f(\mathbf{x}_{i+1}))
    $$
*   **符号解释:**
    *   $f(\mathbf{x}_i)$: 提取第 $i$ 帧特征的函数（如 CLIP 图像编码器）。
    *   $\text{distance}(\cdot, \cdot)$: 衡量两个特征向量之间距离的函数（如余弦距离）。
    *   这个指标越低表示一致性越好，论文中的分数经过处理，数值越高代表质量越好。

### 5.2.2. 视觉质量 (Visual Quality)
*   **概念定义:** 评估单帧画面的美学质量、清晰度、真实感和有无伪影。VBench 通过一个在美学评分数据集上训练的 `Image Quality Assessment (IQA)` 模型来打分。
*   **数学公式:** 通常是一个复杂的神经网络函数，可以概念化为：
    $$\text{Visual Quality} = \frac{1}{N} \sum_{i=1}^{N} \text{IQA\_Model}(\mathbf{x}_i)$$
*   **符号解释:**
    *   $\text{IQA\_Model}(\mathbf{x}_i)$: 对第 $i$ 帧进行美学或质量评分的预训练模型。
    *   分数越高代表视觉质量越好。

### 5.2.3. 文本对齐 (Text Alignment)
*   **概念定义:** 衡量生成的视频内容与给定的文本提示的匹配程度。VBench 使用 `CLIP` 模型来计算视频帧和文本提示之间的相似度得分。
*   **数学公式:**
    $$
    \text{Text Alignment} = \frac{1}{N} \sum_{i=1}^{N} \text{cosine_similarity}(\text{CLIP}_I(\mathbf{x}_i), \text{CLIP}_T(\text{text}))
    $$
*   **符号解释:**
    *   $\text{CLIP}_I(\mathbf{x}_i)$: CLIP 模型提取的第 $i$ 帧的图像特征。
    *   $\text{CLIP}_T(\text{text})$: CLIP 模型提取的文本提示的文本特征。
    *   $\text{cosine\_similarity}(\cdot, \cdot)$: 余弦相似度。
    *   分数越高表示对齐越好。

## 5.3. 对比基线
论文将自己的方法与一系列先进的自回归视频生成模型进行了比较：
*   **SkyReels-V2 [1]:** 一个片段级 (clip-level) 的自回归模型，它顺序生成 5 秒的视频片段。
*   **MAGI-1 [60]:** 一个大规模自回归模型，它放松了严格的因果约束，在当前块去噪完成前就开始下一个块的去噪。
*   **NOVA [17]:** 一个不使用矢量量化 (VQ) 的自回归视频生成模型。
*   **Pyramid Flow [35]:** 使用金字塔流匹配的高效视频生成模型。
*   **CausVid [77]:** 一个使用蒸馏来从双向模型中学习的因果视频生成器。
*   **Self Forcing [32]:** 最重要的基线之一，采用后训练蒸馏策略来弥合训练-测试差距。
*   **LongLive [73]:** 一个同期的工作，与 `Self Forcing` 原理类似，但专注于更长视频的生成，通过对长视频的子片段进行蒸馏。

    这些基线覆盖了不同的技术路线（严格因果、松弛因果、片段自回归、蒸馏等），使得比较非常全面。特别地，与 `CausVid`、`Self Forcing` 和 `LongLive` 的比较，能够突显本文方法在无需教师模型的情况下的竞争力。

---

# 6. 实验结果与分析

## 6.1. 核心结果分析
核心实验结果展示在论文的 Table 1 中。作者将生成的 15 秒视频切分为三个 5 秒的片段（0-5s, 5-10s, 10-15s）并分别评估，以考察模型性能随时间推移的变化。

以下是原文 Table 1 的结果：

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">#Param</th>
<th rowspan="2">Teacher Model</th>
<th colspan="3">Video Length = 0-5 s</th>
<th colspan="3">Video Length = 5-10 s</th>
<th colspan="3">Video Length = 10-15 s</th>
</tr>
<tr>
<th>Temporal</th>
<th>Visual</th>
<th>Text</th>
<th>Temporal</th>
<th>Visual</th>
<th>Text</th>
<th>Temporal</th>
<th>Visual</th>
<th>Text</th>
</tr>
</thead>
<tbody>
<tr>
<td>SkyReels-V2 [11]</td>
<td>1.3B</td>
<td>-</td>
<td>81.93</td>
<td>60.25</td>
<td>21.92</td>
<td>84.63</td>
<td>59.71</td>
<td>21.55</td>
<td>87.50</td>
<td>58.52</td>
<td>21.30</td>
</tr>
<tr>
<td>MAGI-1 [60]</td>
<td>4.5B</td>
<td>-</td>
<td>87.09</td>
<td>59.79</td>
<td>26.18</td>
<td>89.10</td>
<td>59.33</td>
<td>25.40</td>
<td>86.66</td>
<td>59.03</td>
<td>25.11</td>
</tr>
<tr>
<td>NOVA [17]</td>
<td>0.6B</td>
<td>-</td>
<td>87.58</td>
<td>44.42</td>
<td>25.47</td>
<td>88.40</td>
<td>35.65</td>
<td>20.15</td>
<td>84.94</td>
<td>30.23</td>
<td>18.22</td>
</tr>
<tr>
<td>Pyramid Flow [35]</td>
<td>2.0B</td>
<td>-</td>
<td>81.90</td>
<td>62.99</td>
<td>27.16</td>
<td>84.45</td>
<td>61.27</td>
<td>25.65</td>
<td>84.27</td>
<td>57.87</td>
<td>25.53</td>
</tr>
<tr>
<td>CausVid [77]</td>
<td>1.3B</td>
<td>WAN2.1-14B(5s)</td>
<td>89.35</td>
<td>65.80</td>
<td>23.95</td>
<td>89.59</td>
<td>65.29</td>
<td>22.90</td>
<td>87.14</td>
<td>64.90</td>
<td>22.81</td>
</tr>
<tr>
<td>Self Forcing [32]</td>
<td>1.3B</td>
<td>WAN2.1-14B(5s)</td>
<td>90.03</td>
<td>67.12</td>
<td>25.02</td>
<td>84.27</td>
<td>66.18</td>
<td>24.83</td>
<td>84.26</td>
<td>63.04</td>
<td>24.29</td>
</tr>
<tr>
<td>LongLive [73]</td>
<td>1.3B</td>
<td>WAN2.1-14B(5s)</td>
<td>81.84</td>
<td>66.56</td>
<td>24.41</td>
<td>81.72</td>
<td>67.05</td>
<td>23.99</td>
<td>84.57</td>
<td>67.17</td>
<td>24.44</td>
</tr>
<tr>
<td><b>Ours (75% sparsity)</b></td>
<td><b>1.3B</b></td>
<td>-</td>
<td><b>90.18</b></td>
<td><b>63.95</b></td>
<td><b>24.12</b></td>
<td><b>89.80</b></td>
<td><b>61.95</b></td>
<td><b>24.19</b></td>
<td><b>87.03</b></td>
<td><b>61.01</b></td>
<td><b>23.35</b></td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>1.3B</b></td>
<td>-</td>
<td><b>91.20</b></td>
<td><b>64.72</b></td>
<td><b>25.79</b></td>
<td><b>90.44</b></td>
<td><b>64.03</b></td>
<td><b>25.61</b></td>
<td><b>89.74</b></td>
<td><b>63.99</b></td>
<td><b>24.39</b></td>
</tr>
</tbody>
</table>

**分析:**
1.  **性能稳定，无明显衰减:** 观察 `Ours` (本文方法) 的三项指标在三个时间段内的变化，可以发现其分数非常稳定，没有出现随着时间推移而显著下降的趋势。例如，时间质量从 91.20 -> 90.44 -> 89.74，视觉质量从 64.72 -> 64.03 -> 63.99，衰减幅度很小。这有力地证明了 `Resampling Forcing` **有效抑制了误差累积**。
2.  **媲美蒸馏方法:** 与 `Self Forcing` 和 `LongLive` 等依赖一个强大的 14B 参数教师模型进行蒸馏的方法相比，本文方法 (`Ours`) 在**不使用任何教师模型**的情况下，取得了非常有竞争力的结果。尤其是在时间质量上，本文方法在所有时间段都显著优于这些蒸馏基线，这表明端到端的长视频训练对于学习时间连贯性至关重要。
3.  **高效的稀疏路由:** `Ours (75% sparsity)` 版本使用了 `History Routing` ($k=5$)，其注意力稀疏度高达 75%。与使用密集注意力的 `Ours` 版本相比，它的性能仅有微不足道的下降。这证明了 `History Routing` 可以在**大幅降低计算成本**的同时，几乎不牺牲生成质量，为高效长视频生成提供了可行的方案。
4.  **定性与因果性优势:** 如下图（原文 Figure 5）所示，本文方法不仅在长时域保持了视觉质量的稳定，还展现了更好的因果性。在“倒牛奶”的例子中，依赖双向教师模型蒸馏的 `LongLive` 出现了液体“先升后降”的违背物理规律的现象，这可能是因为教师模型泄露了未来信息。而本文方法由于严格的因果训练，正确地模拟了液位单调上升的过程。

    ![Figure 5 Qualitative Comparisons. Top: We compare with representative autoregressive video generation models, showin ur method's stable quality on long video generation.Bottom:Compared with LongLive \[73\] that istill from shor bcialacr, urmethoexhibts bettecusaliye usedashe ienoe thees liquid level, and red arrows to highlight the liquid level in each frame.](images/5.jpg)
    *该图像是展示了不同视频生成模型的定性比较，包括我们的模型与其他几种技术在长视频生成中的稳定性。上半部分展示了各模型的输出效果，下半部分则对比了在倒牛奶过程中液体的变化，红色箭头强调了每帧液体的水平变化。*

## 6.2. 消融实验/参数分析
作者通过一系列消融实验，验证了其方法设计的合理性。

### 6.2.1. 误差模拟策略
论文比较了三种不同的误差模拟策略：(1) <strong>噪声增强 (noise augmentation)</strong>：简单地在历史帧上添加高斯噪声；(2) <strong>并行重采样 (resampling - parallel)</strong>：所有历史帧并行地进行自重采样，不考虑帧间依赖；(3) <strong>自回归重采样 (resampling - autoregressive)</strong>：本文提出的方法，重采样过程是自回归的。

以下是原文 Table 2 的结果：

<table>
<thead>
<tr>
<th rowspan="2">Simulation Strategies</th>
<th colspan="3">Video Length = 0-15 s</th>
</tr>
<tr>
<th>Temporal</th>
<th>Visual</th>
<th>Text</th>
</tr>
</thead>
<tbody>
<tr>
<td>noise augmentation</td>
<td>87.15</td>
<td>61.90</td>
<td>21.44</td>
</tr>
<tr>
<td>resampling - parallel</td>
<td>88.01</td>
<td>62.51</td>
<td>24.51</td>
</tr>
<tr>
<td>resampling - autoregressive</td>
<td><b>90.46</b></td>
<td><b>64.25</b></td>
<td><b>25.26</b></td>
</tr>
</tbody>
</table>

**分析:** 结果清晰地表明，**自回归重采样**策略在所有指标上都显著优于其他两种。这说明，仅仅模拟帧内误差（并行重采样）或使用不匹配的噪声模式（噪声增强）是不够的，**模拟误差在时间上的累积过程**对于训练模型的鲁棒性至关重要。

### 6.2.2. 模拟时间步偏移因子 $s$ 的影响
下图（原文 Figure 6）展示了偏移因子 $s$ 对生成结果的影响。

![该图像是示意图，展示了红色气球在不同位移（Slight Shift、Moderate Shift 和 Large Shift）情况下的场景变化。图中包含三行，每行显示不同位移参数（0.1、0.6 和 3.0）下气球在街道环境中移动的效果，体现了模型在视频生成中的自适应能力。](images/6.jpg)
*该图像是示意图，展示了红色气球在不同位移（Slight Shift、Moderate Shift 和 Large Shift）情况下的场景变化。图中包含三行，每行显示不同位移参数（0.1、0.6 和 3.0）下气球在街道环境中移动的效果，体现了模型在视频生成中的自适应能力。*

*   <strong>小 $s$ (如 0.1):</strong> 模拟的误差强度太弱，模型依然倾向于累积错误，导致视频后半段质量下降（如气球模糊）。
*   <strong>大 $s$ (如 3.0):</strong> 模拟的误差强度太强，模型被允许大幅偏离历史信息，导致内容漂移（如气球在第一帧就偏离了初始位置）。
*   <strong>中等 $s$ (如 0.6):</strong> 在保持历史一致性和修正误差之间取得了最佳平衡。

### 6.2.3. 稀疏历史策略
下图（原文 Figure 7）比较了 `History Routing` 与传统的 `sliding-window` 注意力。

![该图像是插图，展示了不同的注意力机制在视频生成中的表现，包括密集因果注意力、滑动窗口和历史路由（top-1 和 top-5）。这些机制在长视频生成上展现了不同的特征与效果。](images/7.jpg)
**分析:**

*   **`top-5` vs. `dense`:** 路由到 top-5 历史帧（75% 稀疏度）的效果与密集注意力（关注所有历史）几乎没有差别。
*   **`top-1` vs. `sliding-window (size 1)`:** 即使在极高的 95% 稀疏度下，路由到 top-1 历史帧也比只关注前一帧的滑动窗口表现出更好的内容一致性（鱼的外观保持不变）。这证明了**动态、内容感知**的上下文选择比**静态、局部**的选择策略更为优越。

### 6.2.4. 历史路由频率
下图（原文 Figure 8）可视化了在生成第 21 帧时，前 20 个历史帧被选中的频率。

![Figure 8 History Routing Frequency. We visualize the beginning 20 frames' frequency of being selected when generating the 21st frame. For readability, the maximum bar is truncated and labeled with its exact value.](images/8.jpg)
**分析:**

可以观察到一个混合模式：模型最关注**最新的几帧**（如第 18、19、20 帧）和**最初的几帧**（如第 1 帧）。这种“首尾并重”的模式被称为 “attention-sink” 现象，即初始帧作为全局信息的“锚点”被持续关注。这解释了为什么动态路由比简单的滑动窗口效果好，因为它能同时捕捉局部动态和全局上下文。

---

# 7. 总结与思考

## 7.1. 结论总结
这篇论文提出了一种名为 **`Resampling Forcing`** 的创新性端到端训练框架，有效解决了自回归视频扩散模型中的**暴露偏差**问题。
*   **核心贡献:** 通过**自重采样**机制，在训练中巧妙地模拟并让模型适应其自身的推理时误差，从而在无需外部教师模型或复杂后训练的情况下，显著抑制了长视频生成中的误差累积。
*   **附加贡献:** 引入了**历史路由**机制，一种高效的动态稀疏注意力方案，能够在几乎不损失质量的前提下，将长视频生成的注意力复杂度降为常数级别。
*   **主要发现:** 实验证明，该方法不仅在各项指标上达到了与依赖大型教师模型进行蒸馏的SOTA方法相当的水平，更在长时域的时间一致性和因果性上表现出优越性，为构建可扩展、高效且遵循物理规律的视频世界模型铺平了道路。

## 7.2. 局限性与未来工作
*   **推理速度:** 作为一种基于扩散的模型，生成过程仍然需要多步迭代去噪，这限制了其实时生成能力。未来的工作可以探索结合少步蒸馏或更高效的采样器来加速推理。
*   **训练成本:** 训练时需要同时处理带噪的扩散样本和干净的历史序列，对计算资源有一定要求。架构上的优化（如 [42] 中提到的）可能有助于降低这一成本。

## 7.3. 个人启发与批判
1.  **思想的优雅与通用性:** `Resampling Forcing` 的核心思想——“**用自己的矛，攻自己的盾**”——非常巧妙。它将原本棘手的“模型误差”这一负面因素，转化为训练过程中的一种“数据增强”或“课程学习”信号。这种让模型在训练中直面并学会处理自身不完美性的思路，不仅适用于视频生成，也可能对其他自回归任务（如长文本生成、音频合成）中的暴露偏差问题有重要启发。

2.  **端到端训练的价值:** 本文有力地论证了端到端训练范式在因果模型中的重要性。相比于依赖一个可能“知晓未来”的外部教师进行“事后补救”，在严格的因果约束下从头训练，更能让模型内化正确的时间动态和物理规律。Figure 5 中倒牛奶的例子就是一个绝佳的证明。

3.  **对“从零开始”的审视:** 论文声称其方法可以“从零开始 (from scratch)”训练，这指的是其训练目标和流程是自洽的，不依赖外部教师。然而，实验本身是在一个强大的预训练视频模型 `WAN2.1-1.3B` 的基础上进行的微调。这虽然是当前领域研究的常规做法，但距离真正意义上从随机初始化开始训练一个大规模自回归视频模型，可能还有一定距离。未来的工作需要验证该方法在完全从零训练时的收敛性和稳定性。

4.  **长时域的挑战依然存在:** 尽管论文在 15 秒视频上取得了优异成果，并为更长视频的生成提供了高效的注意力机制，但要实现分钟级甚至小时级的连贯视频生成（即真正的“世界模型”），挑战依然巨大。`History Routing` 虽然降低了计算复杂度，但如何确保在极长的历史中依然能检索到数千帧之前的关键信息，仍是一个开放问题。当前 `top-k` 的机制可能在更长的时间尺度下面临信息瓶颈。