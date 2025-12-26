# 1. 论文基本信息

## 1.1. 标题
<strong>直接对齐完整扩散轨迹与细粒度人类偏好 (Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference)</strong>

论文的核心主题是提出一种新的方法，用于将<strong>扩散模型 (Diffusion Models)</strong> 的生成过程与人类偏好进行对齐。与以往的方法不同，该方法旨在优化<strong>整个扩散轨迹 (full diffusion trajectory)</strong>，而不仅仅是最后几个步骤，并且能够<strong>细粒度地 (fine-grained)</strong> 调整模型以符合特定的人类偏好，如“真实感”或“细节丰富度”。

## 1.2. 作者
- **Xiangwei Shen, Zhimin Li, Zhantao Yang, Shiyi Zhang, Yingfang Zhang, Donghao Li, Chunyu Wang, Qinglin Lu, Yansong Tang**
- **隶属机构:**
    1.  <strong>Hunyuan, Tencent (腾讯混元):</strong> 腾讯公司旗下专注于大模型研发的团队。
    2.  <strong>School of Science and Engineering, The Chinese University of Hong Kong, Shenzhen (香港中文大学（深圳）理工学院):</strong>
    3.  <strong>Shenzhen International Graduate School, Tsinghua University (清华大学深圳国际研究生院):</strong>
- **研究背景:** 作者团队主要来自腾讯混元团队，该团队在多模态大模型，特别是文生图模型（如混元DiT）方面有深入研究。这表明论文的研究成果很可能基于其内部强大的模型和计算资源，具有较强的实践背景。

## 1.3. 发表期刊/会议
- <strong>预印本 (Preprint):</strong> 论文目前发布在 arXiv 上，尚未经过同行评审，也未在正式的学术会议或期刊上发表。
- **arXiv:** 是一个广泛使用的学术论文预印本平台，允许研究人员在正式发表前分享他们的最新研究成果。这篇论文的发表日期较新，反映了该领域的快速发展。

## 1.4. 发表年份
- **2025年** (根据论文元数据中的`2025-09-08`，这应是预期的发表或提交日期，实际上传日期可能更早，但表明这是非常前沿的研究)。

## 1.5. 摘要
论文摘要概括了其核心工作：
*   **问题提出:** 现有的利用<strong>可微奖励 (differentiable reward)</strong> 对齐扩散模型的方法存在两大挑战：(1) 依赖计算成本高昂的<strong>多步去噪 (multistep denoising)</strong> 来评估奖励，导致优化仅限于少数几个扩散步骤；(2) 需要持续的<strong>离线 (offline)</strong> 奖励模型微调，以达到如照片真实感等特定美学要求。
*   **核心方法:**
    1.  **Direct-Align:** 提出了一种名为 `Direct-Align` 的方法，通过预定义一个<strong>噪声先验 (noise prior)</strong>，利用“扩散状态是噪声和目标图像的插值”这一原理，可以从任意时间步通过插值高效地恢复原始图像。这避免了在去噪后期阶段的<strong>过度优化 (over-optimization)</strong>。
    2.  <strong>语义相对偏好优化 (Semantic Relative Preference Optimization, SRPO):</strong> 提出 `SRPO`，将奖励信号构建为<strong>文本条件 (text-conditioned)</strong> 的信号。该方法通过对<strong>正向 (positive)</strong> 和<strong>负向 (negative)</strong> 提示词的增强，实现在线调整奖励，从而减少对离线奖励模型微调的依赖。
*   **主要结果:** 通过在 `FLUX` 模型上微调，该方法使其在人类评估的<strong>真实感 (realism)</strong> 和<strong>美学质量 (aesthetic quality)</strong> 上提升了超过 **3倍**。

## 1.6. 原文链接
- **官方来源:** [https://arxiv.org/abs/2509.06942](https://arxiv.org/abs/2509.06942)
- **PDF 链接:** [https://arxiv.org/pdf/2509.06942v3.pdf](https://arxiv.org/pdf/2509.06942v3.pdf)
- **发布状态:** 预印本 (Preprint)。

# 2. 整体概括

## 2.1. 研究背景与动机
- **核心问题:** 如何更高效、更灵活地让文生图模型（特别是扩散模型）的输出更符合人类的细粒度偏好（如“更真实”、“细节更丰富”、“光影效果更好”）。
- <strong>当前挑战 (Gap):</strong>
    1.  **效率瓶颈与局部优化:** 现有的基于<strong>在线强化学习 (Online-RL)</strong> 的对齐方法，如 `ReFL` 和 `DRaFT`，需要通过多步去噪过程来计算奖励并反向传播梯度。这个过程计算量巨大且不稳定，尤其是在噪声水平很高的早期扩散步骤。因此，这些方法被迫将优化范围限制在扩散过程的<strong>后期阶段 (late timesteps)</strong>。这种“局部优化”不仅效率低，还容易导致一个严重问题——<strong>奖励黑客 (Reward Hacking)</strong>，即模型学会了利用奖励模型的偏见来获得高分，但生成的图像质量却很差（例如，图像过度饱和、过于平滑、缺乏细节）。
    2.  **奖励模型的局限性与调整成本:** 现有的奖励模型（如 `HPSv2`、`PickScore`）本身存在偏见，且通常是为通用美学或图文对齐而训练的，无法满足特定、细粒度的偏好（如“照片真实感”）。为了达到特定效果，研究者通常需要花费大量成本<strong>离线 (offline)</strong> 收集高质量数据来微调奖励模型，或者设计复杂的奖励系统来平衡多个指标。这个过程缺乏灵活性，无法在训练中<strong>在线 (online)</strong> 调整。
- **创新切入点:**
    1.  **解决效率瓶颈:** 论文提出 `Direct-Align`，通过一个巧妙的设计——**预定义噪声先验**，使得模型可以从任何时间步<strong>一步 (single-step)</strong> 恢复出清晰图像。这完全绕开了昂贵的多步去噪采样，从而实现了对<strong>整个扩散轨迹 (full diffusion trajectory)</strong> 的高效优化，从根本上解决了局部优化问题。
    2.  **解决奖励调整难题:** 论文提出 `SRPO`，将奖励模型从一个固定的“打分器”变成一个可被文本**动态控制**的“偏好引导器”。通过给提示词加上“正向控制词”（如 `realistic photo`）和“负向控制词”（如 `CG render`），并优化它们之间的**相对差异**，模型可以在线地、细粒度地学习特定偏好，同时有效抑制奖励模型自身的固有偏见。

## 2.2. 核心贡献/主要发现
- **核心贡献:**
    1.  **提出 `Direct-Align` 框架:** 一种高效的单步图像恢复方法，能够对扩散模型的整个生成轨迹进行优化，有效缓解了**奖励黑客**问题。
    2.  **提出 `SRPO` 机制:** 一种新颖的在线奖励调整方法，将奖励信号重构为<strong>文本条件的偏好 (text-conditioned preferences)</strong>，通过正负向提示词增强，实现了对模型生成风格（如真实感、细节）的细粒度控制，减少了对离线微调奖励模型的依赖。
    3.  **实现了最先进的性能与效率:**
        *   **性能突破:** 在 `FLUX.1.dev` 模型上的实验表明，`SRPO` 方法在人类评估的**真实感**和**美学质量**上分别提升了约 **3.7倍** 和 **3.1倍**，达到了最先进的水平。
        *   **效率突破:** 该方法极为高效，仅需在 **32块 NVIDIA H20 GPU** 上训练 **10分钟** 即可收敛，训练效率远超同类方法（如比 `DanceGRPO` 快 **75倍**）。

- **关键发现:**
    1.  **全轨迹优化的重要性:** 仅优化扩散后期步骤确实会导致模型过度拟合奖励模型的偏见（奖励黑客），而对包括早期步骤在内的整个轨迹进行优化，能显著提升生成图像的质量和鲁棒性。
    2.  **奖励信号的相对性:** 与追求绝对奖励分数相比，优化一对<strong>语义相对 (semantic-relative)</strong> 的奖励信号（例如，“真实照片”得分 vs. “CG渲染”得分）是一种更有效的正则化手段，它能过滤掉奖励模型中与语义无关的偏见，使优化方向更精确。
    3.  **奖励模型的可控性:** 奖励模型不仅能打分，还能被文本“引导”。通过简单的提示词增强，就可以在线改变奖励模型的偏好，这是一个强大且低成本的对齐工具。

# 3. 预备知识与相关工作

## 3.1. 基础概念
### 3.1.1. 扩散模型 (Diffusion Models)
<strong>扩散模型 (Diffusion Models)</strong> 是一种强大的生成模型，其核心思想分为两个过程：
1.  <strong>前向过程 (Forward Process):</strong> 也称为扩散过程。从一张真实的、清晰的图像 $x_0$ 开始，逐步、重复地向其添加少量的高斯噪声，直到经过 $T$ 步后，图像变成完全的纯高斯噪声 $\epsilon$。这个过程可以被精确地数学描述，在任意时间步 $t$，带噪图像 $x_t$ 可以直接由初始图像 $x_0$ 和噪声 $\epsilon$ 通过插值得到：
    $$
    x_t = \alpha_t x_0 + \sigma_t \epsilon
    $$
    其中 $\alpha_t$ 和 $\sigma_t$ 是与时间步 $t$ 相关的预设系数（称为噪声调度表），随着 $t$ 从 0 增大到 $T$，$\alpha_t$ 从 1 减小到 0，而 $\sigma_t$ 从 0 增大到 1。

2.  <strong>反向过程 (Reverse Process):</strong> 也称为去噪过程。模型学习如何逆转上述加噪过程。它从一个纯噪声图像 $x_T$ 开始，通过一个神经网络（通常是 U-Net 结构）在每个时间步 $t$ 预测出应该被移除的噪声 $\epsilon_\theta(x_t, t, c)$（其中 $c$ 是条件，如文本提示词）。通过逐步减去预测的噪声，最终生成一张清晰的图像。

### 3.1.2. 在线强化学习 (Online Reinforcement Learning, Online-RL)
<strong>在线强化学习 (Online-RL)</strong> 是一种通过与环境实时交互来学习最优策略的机器学习范式。在文生图领域，这个过程可以被理解为：
*   <strong>智能体 (Agent):</strong> 扩散模型本身。
*   <strong>动作 (Action):</strong> 在每个去噪步骤中预测噪声。
*   <strong>状态 (State):</strong> 当前的带噪图像 $x_t$。
*   <strong>奖励 (Reward):</strong> 生成最终图像后，由一个<strong>奖励模型 (Reward Model)</strong> 对图像质量进行打分。这个分数就是奖励。
*   **学习过程:** 模型根据奖励信号调整其参数（即神经网络的权重），使得未来生成的图像能获得更高的奖励。所谓“在线”，是指模型的参数更新是基于当前批次生成的样本和奖励，而不是基于一个固定的、离线的数据集。

### 3.1.3. 可微奖励 (Differentiable Reward)
在传统的强化学习中，奖励通常是不可微的（例如，游戏得分），需要使用策略梯度等方法进行优化。但在文生图对齐中，奖励模型（如 `HPSv2`）本身是一个神经网络，其输出的奖励分数对于输入的图像是<strong>可微的 (differentiable)</strong>。这意味着我们可以将奖励函数直接连接到生成模型的计算图上，然后通过<strong>反向传播 (backpropagation)</strong> 将奖励的梯度直接传导给生成模型的参数。这种方法被称为<strong>直接偏好优化 (Direct Preference Optimization, DPO)</strong> 的一种形式，相比策略梯度，它通常更稳定、更高效。

## 3.2. 前人工作
论文主要围绕现有基于**可微奖励**的在线RL对齐方法展开，并对它们进行了改进。

1.  **ReFL (Reward-guided Fine-tuning via Large-scale Language-image model)**
    *   **核心思想:** `ReFL` 是一种典型的直接反向传播方法。它在扩散过程的某个后期时间步 $t$ 停止，然后进行一次**单步去噪预测**，得到预测的清晰图像 $\hat{x}_0$。接着，用奖励模型 $R$ 评估 $\hat{x}_0$ 的质量，并将奖励的梯度反向传播，以优化扩散模型。
    *   **局限性:**
        *   其单步预测公式 `\hat{x}_0 = (\mathbf{x_t} - \sigma_t \epsilon_\theta(\mathbf{x_t}, t, \mathbf{c})) / \alpha_t` 在早期时间步（噪声大时）非常不准确，生成的图像质量差，导致奖励信号不可靠。
        *   因此，`ReFL` 只能在**后期时间步**（例如 $t$ 接近0）进行优化，这限制了其优化范围，容易导致奖励黑客。

2.  **DRaFT (Directly Fine-tuning Diffusion Models on Differentiable Rewards)**
    *   **核心思想:** `DRaFT` 同样采用直接反向传播。但它不是进行单步预测，而是从某个中间状态 $x_k$ 开始，通过一个**可微的多步采样器**（如 DDIM）进行多步去噪，生成最终图像，然后计算奖励并反向传播。
    *   **局限性:**
        *   通过多步采样器进行反向传播的**计算成本极高**，且随着步数增加，梯度容易**爆炸或消失**，训练非常不稳定。
        *   为了维持计算可行性，`DRaFT` 同样只能在少数几个（原文报告不超过5步）后期步骤上进行优化，面临与 `ReFL` 相似的**局部优化**和**奖励黑客**问题。

3.  **DanceGRPO / Flow-GRPO**
    *   **核心思想:** 这类方法属于<strong>策略梯度 (Policy Gradient)</strong> 的范畴，它们通过比较一组生成样本的相对好坏来优化模型，而不是直接使用绝对奖励分数。
    *   **局限性:** 策略梯度方法通常比直接反向传播方法的**样本效率低**，需要更多的生成样本和训练步骤才能收敛。论文指出，其训练效率比本文提出的方法低得多（慢75倍）。

4.  <strong>奖励模型微调 (Refining Reward Models)</strong>
    *   **动机:** 现有奖励模型（如 `HPSv2`、`PickScore`）存在偏见，例如偏爱过饱和、低细节的图像。
    *   **方法:** 像 `ICTHP` 等工作通过收集大规模高质量数据集来离线微调奖励模型，以纠正这些偏见。
    *   **局限性:** 成本高昂，且缺乏灵活性。每次需要新的偏好时，都可能需要重新收集数据和微调。

## 3.3. 技术演进
文生图模型的对齐技术经历了从<strong>基于分类器引导 (Classifier Guidance)</strong> 到<strong>无分类器引导 (Classifier-Free Guidance, CFG)</strong>，再到更精细的<strong>基于人类反馈的强化学习 (RLHF)</strong> 的演变。RLHF 又分为两大流派：
1.  <strong>策略梯度方法 (Policy-based):</strong> 如 `DPOK`、`GRPO` 等，通过估计奖励来更新策略，相对稳健但样本效率较低。
2.  <strong>直接反向传播方法 (Gradient-based):</strong> 如 `ReFL`、`DRaFT`，利用可微奖励直接优化，效率高但面临训练不稳定和奖励黑客的挑战。

    本文的工作正是在**直接反向传播**这条技术路线上，针对其核心痛点——**局部优化**和**奖励调整**——提出的创新解决方案。

## 3.4. 差异化分析
- **与 `ReFL` 和 `DRaFT` 的区别:**
    - `ReFL` 和 `DRaFT` 都受限于后期步骤优化。`Direct-Align` 通过**预定义噪声先验**实现**全轨迹的单步恢复**，从根本上解决了这个问题，实现了**全局优化**，效率和效果都更优。
- **与 `DanceGRPO` 等策略梯度方法的区别:**
    - `SRPO` 依然采用计算效率更高的**直接反向传播**机制，但通过**语义相对偏好**的设计，吸收了 `GRPO` 类方法利用相对优势进行优化的思想，同时避免了其样本效率低的问题。
- **与 `ICTHP` 等奖励模型微调方法的区别:**
    - `ICTHP` 通过**离线**修改奖励模型来适应新偏好。`SRPO` 则通过**在线**的提示词工程来**动态引导**现有奖励模型，无需任何额外数据和模型微调，更加灵活和低成本。

# 4. 方法论

本论文的核心方法由两部分构成：`Direct-Align` 解决了**如何高效优化全扩散轨迹**的问题，而 `SRPO` 则解决了**如何灵活调整奖励信号以符合细粒度偏好**的问题。

## 4.1. 方法原理
### 4.1.1. Direct-Align: 全轨迹单步图像恢复
现有方法（如 `ReFL`）在去噪早期效果差，是因为它们的单步预测公式 $\hat{x}_0 = (\mathbf{x_t} - \sigma_t \epsilon_\theta) / \alpha_t$ 强依赖于模型预测的噪声 $\epsilon_\theta$，而当真实噪声 $\epsilon_{gt}$ 占比很大时，模型预测的微小误差会被 $1/\alpha_t$ 因子（在早期 $t$ 很大时，$\alpha_t$ 极小）急剧放大，导致结果崩坏。

`Direct-Align` 的核心直觉是：既然我们知道扩散过程的起点（清晰图像 $x_0$）和终点（纯噪声 $\epsilon_{gt}$），那么任何中间状态 $x_t$ 都是这两者的线性插值。如果我们能**预先知道**这个用于生成 $x_t$ 的具体噪声 $\epsilon_{gt}$，我们就可以**精确地**、**一步**从 $x_t$ 恢复出 $x_0$。

该方法通过一个巧妙的流程实现了这一点：**先合成，再恢复**。

### 4.1.2. SRPO: 语义相对偏好优化
`SRPO` 的核心直觉是：奖励模型（如 `HPSv2`）内部是一个类似 `CLIP` 的图文匹配模型，其奖励分数本质上是图像特征和文本特征的点积。这意味着，我们可以通过修改**文本**来改变奖励的“评判标准”。

例如，我们想让模型生成“真实的照片”。`SRPO` 不仅使用原始提示词 $p$，还构造了一对**语义相对**的提示：
*   <strong>正向提示 (Positive Prompt):</strong> `p_c_pos` = “realistic photo” + $p$
*   <strong>负向提示 (Negative Prompt):</strong> `p_c_neg` = “CG render” + $p$

    然后，`SRPO` 优化的目标不再是最大化单一奖励，而是**最大化正向奖励与负向奖励的差值**。这会迫使模型学习区分“真实照片”和“CG渲染”的细微特征，同时，由于两个奖励都来自同一张图片，奖励模型中与此语义差异无关的固有偏见（如对某种颜色的偏好）会在相减过程中被**抵消**掉。

## 4.2. 核心方法详解 (逐层深入)
### 4.2.1. Direct-Align 训练流程
`Direct-Align` 的完整训练流程如下图所示，可以分解为三个关键步骤：

![该图像是示意图，展示了直接对齐全扩散轨迹与细粒度人类偏好的方法。图中包含两个主要部分：Direct-Align 和语义相对偏好优化（SRPO）。Direct-Align部分说明了如何将高斯噪声注入图像并通过梯度去噪及逆高斯方法恢复图像，配合奖励模型（RM）进行评估。SRPO部分则描述了如何根据正负提示动态调整奖励与惩罚，从而优化人类评价的现实感和美学质量。](images/2.jpg)

<strong>Step 1: 注入预定义噪声 (Inject Predefined Noise)</strong>
首先，我们从一个清晰的图像 $x_0$ 开始（在训练中，这可以是数据集中的真实图像，或由模型生成的样本）。然后，我们生成一个标准高斯噪声向量 $\epsilon_{gt}$（这里的 `gt` 代表 `ground truth`，因为我们把它当作已知的真实噪声）。接着，根据扩散模型的正向过程公式，在任意一个时间步 $t$ 合成带噪图像 $x_t$：
$$
\mathbf{x_t} = \alpha_t \mathbf{x_0} + \sigma_t \epsilon_{gt}
\tag{4a}
$$
- $\mathbf{x_0}$: 原始清晰图像。
- $\epsilon_{gt}$: 随机采样并**固定**的高斯噪声，我们称之为<strong>噪声先验 (noise prior)</strong>。
- $\alpha_t, \sigma_t$: 时间步 $t$ 对应的噪声调度系数。
- $\mathbf{x_t}$: 合成的带噪图像。

<strong>Step 2: 单步去噪预测 (One-step Denoise)</strong>
现在，我们将 $x_t$ 输入到扩散模型中，让模型预测在时间步 $t$ 和条件 $c$ 下的噪声 $\epsilon_\theta(\mathbf{x_t}, t, \mathbf{c})$。

<strong>Step 3: 单步图像恢复 (Single-step Image Recovery)</strong>
这是 `Direct-Align` 的关键创新。我们不使用 `ReFL` 那种不稳定的恢复公式，而是利用我们在 Step 1 中已知的噪声先验 $\epsilon_{gt}$。从公式 (4a) 可以直接推导出 $x_0$ 的精确解析解：
$$
\mathbf{x_0} = \frac{\mathbf{x_t} - \sigma_t \epsilon_{gt}}{\alpha_t}
\tag{4b}
$$
这个公式告诉我们，只要知道 $x_t$ 和对应的 $\epsilon_{gt}$，就可以**完美地**恢复出 $x_0$。然而，我们的目标是优化模型 $\epsilon_\theta$，所以不能完全绕开它。

因此，论文提出了一种混合恢复策略。模型预测的噪声 $\epsilon_\theta$ 只用来替换一小部分真实噪声。具体来说，最终的奖励计算基于如下恢复的图像 $\hat{x}_0$：
$$
r = r \left( \frac { \mathbf { x_t } - \Delta \sigma_t \epsilon_ { \theta } ( \mathbf { x_t } , t , \mathbf { c } ) - ( \sigma_t - \Delta \sigma ) \epsilon_{gt} } { \alpha_t } \right)
\tag{5}
$$
- **公式解析:**
    - 这个公式本质上是对公式 (4b) 的修改。在理想情况下，模型预测的噪声 $\epsilon_\theta$ 应该等于真实噪声 $\epsilon_{gt}$。
    - 我们将真实噪声 $\sigma_t \epsilon_{gt}$ 分为两部分：一小部分 $\Delta \sigma_t \epsilon_{gt}$ 和剩余部分 $(\sigma_t - \Delta \sigma) \epsilon_{gt}$。
    - 我们用模型预测的噪声 $\Delta \sigma_t \epsilon_{\theta}$ 来**替代**那一小部分真实噪声。
    - $\Delta \sigma_t$ 是一个非常小的系数，表示我们只相信模型预测的一小部分。这意味着恢复的图像主要由**稳定的真实噪声**决定，而模型的预测只在其中施加一个微小的、可控的扰动。
    - **梯度来源:** 奖励 $r$ 的梯度会通过 $\epsilon_\theta$ 反向传播给模型参数。由于 $\Delta \sigma_t$ 很小，梯度被有效控制，避免了爆炸。同时，因为恢复的图像质量高（如下图 `Figure 3` 所示），奖励信号非常准确，即使在噪声很大的早期时间步也是如此。

      下图（原文 Figure 3）直观对比了 `ReFL` 的单步预测和 `Direct-Align` 的单步恢复在早期时间步（噪声占比95%）的效果。`Direct-Align` 恢复的图像结构清晰，而 `ReFL` 的预测则完全是伪影。

      ![Figure 3. Comparison on one-step prediction at early timestep The values 0.075 and 0.025 denote the weight of the model prediction term used for method, respectively. The earliest $5 \\%$ represent state with $9 5 \\%$ noise from an unshifted timestep. By constructing a Gaussian prior, our one-step sampling method achieves highquality results at early timesteps, even when the input image is highly noised.](images/3.jpg)
      *该图像是一个比较图表，展示了不同时间步长下的图像预测质量。上方为先前方法的结果，下方是我们的方法在两种不同权重参数（0.075和0.025）下的效果。每一列代表不同的噪声水平，从最早的5%到50%。该方法通过构建高斯先验，实现了在早期时间步长下的高质量图像重建。*

### 4.2.2. 奖励聚合与正则化
为了进一步提升稳定性和缓解奖励黑客，`Direct-Align` 还引入了两个机制：

1.  <strong>奖励聚合 (Reward Aggregation):</strong>
    在一次训练中，不只在一个时间步 $t$ 进行优化，而是在一个时间步区间 $\{k, k-1, ..., k-n\}$ 内进行多次恢复和奖励计算，然后将这些奖励梯度累积起来。这使得模型能同时感知多个时间步的偏好。
    $$
    r(\mathbf{x_t}) = \lambda(t) \cdot \sum_{i=k-n}^{k} r(\text{recover}(x_i), \mathbf{c})
    $$
    其中 $\lambda(t)$ 是一个衰减折扣因子，它会给<strong>后期时间步 (late timesteps)</strong> 分配较低的权重。这是因为后期步骤更容易发生奖励黑客，降低其权重可以有效抑制过拟合。

2.  <strong>逆向过程正则化 (Inversion-Based Regularization):</strong>
    `Direct-Align` 的恢复过程与模型的计算图方向解耦，因此可以同时在<strong>去噪 (denoising)</strong> 和<strong>加噪 (inversion)</strong> 两个方向上进行优化。
    *   **去噪奖励 $r_1$:** 目标是使模型预测的噪声能更好地恢复出符合偏好的图像。
        $$
        r_1 = r_1 \left( \frac { \mathbf{a} - \Delta \sigma_t \epsilon_ { \theta } \left( \mathbf { x_t } , t , \mathbf { c } \right) } { \alpha_t } \right)
        \tag{12}
        $$
        这里 $\mathbf{a}$ 是带噪图像，我们希望通过减去模型预测的噪声来得到高奖励图像，因此这是一个**梯度上升**过程。
    *   **加噪奖励 $r_2$:** 目标是惩罚那些不好的模式。
        $$
        r_2 = r_2 \left( \frac { \mathbf { b } + \Delta \sigma_t \epsilon_ { \theta } \left( \mathbf { x_t } , t , \mathbf { c } \right) } { \alpha_t } \right)
        \tag{13}
        $$
        这里 $\mathbf{b}$ 是另一个图像状态，我们通过**加上**模型预测的噪声来计算一个负向奖励，这是一个**梯度下降**过程，起到正则化作用。

### 4.2.3. Semantic-Relative Preference Optimization (SRPO)
`SRPO` 是一种新颖的奖励函数设计。标准的奖励函数 $RM(\mathbf{x}, \mathbf{p})$ 计算图像 $\mathbf{x}$ 和提示词 $\mathbf{p}$ 的匹配分数。`SRPO` 对其进行了扩展。

1.  <strong>语义引导偏好 (Semantic Guided Preference, SGP):</strong>
    `SGP` 的核心是将奖励函数视为一个受文本条件 $C$ 参数化的函数。通过在原始提示词 $\mathbf{p}$ 前面加上一个“控制词” $\mathbf{p_c}$，可以改变文本嵌入 $\mathbf{C}$，从而引导奖励模型的偏好。
    $$
    r_{SGP}(\mathbf{x}) = RM(\mathbf{x}, (\mathbf{p_c}, \mathbf{p})) \propto f_{img}(\mathbf{x})^T \cdot \mathbf{C}_{(\mathbf{p_c}, \mathbf{p})}
    \tag{7}
    $$
    - $f_{img}(\mathbf{x})$: 奖励模型的图像编码器输出的图像特征。
    - $\mathbf{C}_{(\mathbf{p_c}, \mathbf{p})}$: 奖励模型的文本编码器输出的文本特征，由控制词 $\mathbf{p_c}$ 和原始提示词 $\mathbf{p}$ 共同决定。

2.  <strong>语义相对偏好 (Semantic-Relative Preference, SRP):</strong>
    `SGP` 虽然可以引导偏好，但仍然会受到奖励模型自身偏见的影响。`SRP` 通过计算一对**相对奖励**来解决这个问题。它使用同一个图像 $\mathbf{x}$，但分别用一个正向控制词（如“realistic”）和一个负向控制词（如“cartoon”）来计算两个奖励，然后优化它们的差值。
    $$
    \begin{aligned}
    r_{SRP}(\mathbf{x}) &= r_1 - r_2 \\
    &= f_{img}(\mathbf{x})^T \cdot (\mathbf{C}_1 - \mathbf{C}_2)
    \end{aligned}
    \tag{8, 9}
    $$
    - $\mathbf{C}_1$: 正向提示词（例如，`"realistic photo", p`）对应的文本特征。
    - $\mathbf{C}_2$: 负向提示词（例如，`"CG render", p`）对应的文本特征。
    - **原理:** 优化这个差值，意味着梯度方向由 $(\mathbf{C}_1 - \mathbf{C}_2)$ 决定。图像特征 $f_{img}(\mathbf{x})$ 中与这个**语义差异**无关的部分（即奖励模型的固有偏见）在优化中被忽略，而与语义差异高度相关的部分则被强化。这使得优化方向更加精确，有效抑制了奖励黑客。

      论文还提出了一个类似<strong>无分类器引导 (CFG)</strong> 的变体形式，通过一个系数 $k$ 来权衡正负向奖励：
$$
r_{CFG}(\mathbf{x}) = f_{img}(\mathbf{x})^T \cdot ((1-k) \cdot \mathbf{C}_2 + k \cdot \mathbf{C}_1)
\tag{10}
$$

# 5. 实验设置

## 5.1. 数据集
- **训练数据集:**
    - **HPDv2 (Human Preference Dataset v2):** 这是一个包含人类偏好标注的数据集，常用于训练或评估对齐算法。论文使用该数据集的训练集进行模型微调。该数据集包含来自 **DiffusionDB** 的四个视觉概念。
- **评估数据集:**
    - **HPDv2 benchmark:** 包含 3,200 个提示词，用于自动指标评估。
    - **人类评估数据集:** 从 HPDv2 基准的四个子类别中各抽取前 125 个提示词，共 500 个提示词，用于进行详细的人类评估。
- **选择理由:** HPDv2 是一个公认的、用于评估文生图模型与人类偏好对齐程度的标准数据集，使用它可以方便地与其他方法进行公平比较。

## 5.2. 评估指标
论文使用了自动评估指标和人类评估两种方式。

### 5.2.1. 自动评估指标
1.  **Aesthetic Score v2.5 (Aes)**
    *   **概念定义:** 一个预训练的神经网络模型，专门用于评估图像的**美学质量**。它不关心图像内容是否与文本匹配，只从构图、色彩、光影等方面给出一个美学分数。分数越高，代表图像在普遍意义上越“好看”。
    *   **数学公式:** 该指标由一个深度神经网络 $f_{aes}$ 直接计算得出，没有简单的解析公式。其计算方式为：$S_{aes} = f_{aes}(\mathbf{x})$。
    *   **符号解释:** $S_{aes}$ 是美学分数，$\mathbf{x}$ 是输入的图像。

2.  **PickScore**
    *   **概念定义:** 一个基于 `CLIP` 的奖励模型，通过在一个大规模的人类偏好数据集 `Pick-a-Pic`上进行微调得到。它旨在预测一张图像相比于另一张（对于同一个提示词）是否更受人类青睐。分数越高，表示图像越符合人类偏好。
    *   **数学公式:** $S_{pick} = f_{pick}(\mathbf{x}, \mathbf{p})$。
    *   **符号解释:** $S_{pick}$ 是 PickScore 分数，$\mathbf{x}$ 是图像，$\mathbf{p}$ 是提示词。

3.  **ImageReward**
    *   **概念定义:** 另一个基于人类偏好训练的奖励模型。它专注于评估图文对齐度和生成质量，是 RLHF for T2I 领域的常用奖励模型之一。分数越高，表示图像质量和图文一致性越好。
    *   **数学公式:** $S_{ir} = f_{ir}(\mathbf{x}, \mathbf{p})$。
    *   **符号解释:** $S_{ir}$ 是 ImageReward 分数，$\mathbf{x}$ 是图像，$\mathbf{p}$ 是提示词。

4.  **HPSv2.1 (Human Preference Score v2.1)**
    *   **概念定义:** 本文训练中使用的核心奖励模型。它也是一个基于 `CLIP` 并在人类偏好数据上微调的模型，旨在量化图像与人类审美的对齐程度。分数越高，表示越符合人类偏好。
    *   **数学公式:** $S_{hps} = f_{hps}(\mathbf{x}, \mathbf{p})$。
    *   **符号解释:** $S_{hps}$ 是 HPSv2.1 分数，$\mathbf{x}$ 是图像，$\mathbf{p}$ 是提示词。

5.  **SGP-HPS (Semantic Guided Preference - HPS)**
    *   **概念定义:** 这是**本文提出的一个新指标**，专门用于衡量模型在“真实感”这个维度上的提升。它计算了使用正向提示（"Realistic photo"）和负向提示（"CG Render"）时，由 `HPSv2.1` 模型给出的奖励分数之差。差值越大，说明模型生成的图像在“真实感”维度上与“CG感”的区分度越高。
    *   **数学公式:**
        $$
        S_{sgp-hps} = f_{hps}(\mathbf{x}, \mathbf{p}_{pos}) - f_{hps}(\mathbf{x}, \mathbf{p}_{neg})
        $$
    *   **符号解释:** $\mathbf{p}_{pos}$ 是加了“Realistic photo”前缀的提示词，$\mathbf{p}_{neg}$ 是加了“CG Render”前缀的提示词。

6.  **GenEval**
    *   **概念定义:** 一个专注于评估<strong>图文对齐 (Text-to-Image Alignment)</strong> 的框架。它通过检测生成图像中是否包含了提示词中描述的<strong>对象 (objects)</strong> 来进行打分。分数越高，说明图像内容与文本描述的一致性越好。
    *   **数学公式:** 该指标基于复杂的视觉-语言模型，没有简单的解析公式。
    *   **符号解释:** N/A

7.  **DeQA (Degradation-aware Quality Assessment)**
    *   **概念定义:** 一个用于评估图像**退化程度**的指标。它主要关注图像中是否存在伪影、模糊、噪声等质量问题。分数越低通常表示图像质量越好，但在此表中可能是经过处理，分数越高表示退化越少。
    *   **数学公式:** 基于神经网络模型，没有简单的解析公式。
    *   **符号解释:** N/A

### 5.2.2. 人类评估
- **评估维度:**
    1.  <strong>文本-图像对齐 (Text-image alignment):</strong> 图像内容与提示词的匹配程度。
    2.  <strong>真实感与伪影 (Realism and artifact presence):</strong> 图像是否看起来真实，是否存在AI生成常见的变形、扭曲等问题。
    3.  <strong>细节复杂性与丰富度 (Detail complexity and richness):</strong> 图像的纹理、细节是否清晰丰富。
    4.  <strong>美学构图与吸引力 (Aesthetic composition and appeal):</strong> 图像的构图、色彩、光影是否美观。
- **评估等级:** 每个维度分为四个等级：<strong>优秀 (Excellent)</strong>, <strong>良好 (Good)</strong>, <strong>合格 (Pass)</strong>, <strong>失败 (Fail)</strong>。
- **评估人员:** 10名经过培训的标注员和3名领域专家。

## 5.3. 对比基线
- **FLUX:** 基线模型，即未经任何偏好对齐微调的 `FLUX.1.dev` 开源模型。
- **ReFL*:** 论文作者复现的 `ReFL` 方法。
- **DRaFT-LV*:** 论文作者复现的 `DRaFT` 方法的变体。
- **DanceGRPO:** 一种基于策略梯度的先进在线RL对齐方法。
- **Direct-Align:** 本文提出的优化框架，但未使用 `SRPO` 奖励，仅使用标准的 HPS 奖励。
- **SRPO:** 本文提出的完整方法，结合了 `Direct-Align` 框架和 `SRPO` 奖励机制。
- **FLUX.1.Krea:** 由 Krea AI 发布的最新版 `FLUX` 模型，作为一个强大的外部基线。

# 6. 实验结果与分析

## 6.1. 核心结果分析
论文的核心结果展示在 `Table 1` 和 `Figure 4` 中，通过自动指标和人类评估，全面对比了 `SRPO` 与其他方法的性能。

### 6.1.1. 自动评估结果（表格）
以下是原文 `Table 1` 的结果，该表格对比了不同在线RL方法在多个自动评估指标和人类评估上的表现。

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="5">Reward</th>
<th colspan="2">Other Metrics</th>
<th colspan="3">Human Eval</th>
<th rowspan="2">GPU hours(H20)</th>
</tr>
<tr>
<th>Aes</th>
<th>Pick</th>
<th>ImageReward</th>
<th>HPS</th>
<th>SGP-HPS</th>
<th>GenEval</th>
<th>DeQA</th>
<th>Real</th>
<th>Aesth</th>
<th>Overall</th>
</tr>
</thead>
<tbody>
<tr>
<td>FLUX</td>
<td>5.867</td>
<td>22.671</td>
<td>1.115</td>
<td>0.289</td>
<td>0.463</td>
<td>0.678</td>
<td>4.292</td>
<td>8.2</td>
<td>9.8</td>
<td>5.3</td>
<td>−</td>
</tr>
<tr>
<td>ReFL*</td>
<td>5.903</td>
<td>22.975</td>
<td>1.195</td>
<td>0.298</td>
<td>0.470</td>
<td>0.656</td>
<td>4.299</td>
<td>5.5</td>
<td>6.6</td>
<td>3.1</td>
<td>16</td>
</tr>
<tr>
<td>DRaFT-LV*</td>
<td>5.729</td>
<td>22.932</td>
<td>1.178</td>
<td>0.296</td>
<td>0.458</td>
<td>0.636</td>
<td>4.236</td>
<td>8.3</td>
<td>9.7</td>
<td>4.7</td>
<td>24</td>
</tr>
<tr>
<td>DanceGRPO</td>
<td>6.022</td>
<td>22.803</td>
<td>1.218</td>
<td>0.297</td>
<td>0.414</td>
<td>0.585</td>
<td>4.353</td>
<td>5.3</td>
<td>8.5</td>
<td>3.7</td>
<td>480</td>
</tr>
<tr>
<td>Direct-Align</td>
<td>6.032</td>
<td>23.030</td>
<td>1.223</td>
<td>0.294</td>
<td>0.448</td>
<td>0.668</td>
<td>4.373</td>
<td>5.9</td>
<td>8.7</td>
<td>3.9</td>
<td>16</td>
</tr>
<tr>
<td>SRPO</td>
<td>6.194</td>
<td>23.040</td>
<td>1.118</td>
<td>0.289</td>
<td>0.505</td>
<td>0.665</td>
<td>4.275</td>
<td>38.9</td>
<td>40.5</td>
<td>29.4</td>
<td>5.3</td>
</tr>
</tbody>
</table>

**表格分析:**
1.  **SRPO的优越性:** `SRPO` 在**所有人类评估指标**（`Real`真实感, `Aesth`美学, `Overall`总体偏好）上都取得了**压倒性**的优势。例如，在真实感方面，`SRPO` 的优秀率达到了 **38.9%**，远超基线 `FLUX` 的 8.2% 和其他所有方法（均低于10%）。这证明了 `SRPO` 在提升特定、细粒度偏好上的强大能力。
2.  **奖励黑客的证据:**
    *   `ReFL` 和 `Direct-Align`（未使用 `SRPO`）虽然在奖励指标（如 `PickScore`, `ImageReward`）上有所提升，但在人类评估的真实感上甚至**低于**基线 `FLUX`（`ReFL` 5.5% vs `FLUX` 8.2%）。这正是**奖励黑客**的典型表现：模型学会了刷高奖励分数，但实际图像质量变差了。
    *   `SRPO` 在 `HPS` 和 `ImageReward` 分数上与基线 `FLUX` 持平甚至略低，但人类评估分数却大幅领先。这说明 `SRPO` **没有过拟合**奖励模型，而是真正学习到了人类偏好的本质。
3.  **SGP-HPS 指标的有效性:** `SRPO` 在本文提出的 `SGP-HPS` 指标上得分最高（0.505），这表明它确实成功地拉大了“真实照片”和“CG渲染”之间的语义区分度，与人类评估的真实感提升结果一致。
4.  **训练效率:** `SRPO` 的训练时间仅为 **5.3 GPU小时**，比 `DanceGRPO`（480小时）快了近两个数量级，同时效果远超后者，展示了其极高的效率。

### 6.1.2. 人类评估结果（图表）
下图（原文 Figure 4）更直观地展示了人类评估中各项指标的“优秀率”对比。

![该图像是一个条形图，展示了不同模型在文本到图像对齐、现实性和 AI 生成内容伪影、清晰度与细节、美学质量以及整体偏好方面的评估结果。各模型的表现通过“优、良、中、差”四个等级进行分类显示。](images/4.jpg)

**图表分析:**
此图清晰地显示了 `SRPO` 在所有人类评估维度上的巨大优势。其他在线RL方法在“真实感”和“细节”上几乎没有提升，甚至有所下降，而 `SRPO` 则实现了飞跃式的进步。

### 6.1.3. 定性结果分析
论文中的 `Figure 5` 和 `Figure 8` 展示了生成图像的视觉对比。

![该图像是一个对比示意图，展示了FLUX、DanceGRPO和SRPO三种方法在风格保持、美学和摄影效果上的不同表现。每个方法的图像分别展示了在不同评价标准下生成的效果，可以看到不同方法在细节和画风上的差异。](images/5.jpg)
*该图像是一个对比示意图，展示了FLUX、DanceGRPO和SRPO三种方法在风格保持、美学和摄影效果上的不同表现。每个方法的图像分别展示了在不同评价标准下生成的效果，可以看到不同方法在细节和画风上的差异。*

- <strong>与 `DanceGRPO` 对比 (Figure 5):</strong> `DanceGRPO` 生成的图像虽然在某些美学指标上得分高，但常常带有明显的AI伪影，如不自然的“油光感”和物体边缘的“诡异高光”。相比之下，`SRPO` 生成的图像在真实感和细节纹理上表现出色，更自然、更可信。
- <strong>风格控制能力 (Figure 8):</strong> `SRPO` 可以通过简单的控制词（如 `dark`, `lighting`, `comic`, `cyberpunk`）实现对生成风格的细粒度控制。这证明了 `SRPO` 框架的灵活性和可扩展性。

  ![该图像是一个示意图，展示了不同风格下的图像生成效果，左侧为没有风格词的推断结果，右侧为带有风格词的推断结果。每一行包含不同主题，如黑暗、灯光、文艺复兴等，体现了风格词对生成图像的影响。](images/8.jpg)
  *该图像是一个示意图，展示了不同风格下的图像生成效果，左侧为没有风格词的推断结果，右侧为带有风格词的推断结果。每一行包含不同主题，如黑暗、灯光、文艺复兴等，体现了风格词对生成图像的影响。*

## 6.2. 消融实验/参数分析
论文通过一系列消融实验验证了其方法各组件的有效性。

### 6.2.1. 优化时间步的影响
`Figure 7` 的实验对比了在不同时间步区间（`Early` 早期, `All` 全部, `Late` 晚期）训练的效果。

![Figure 7. Comparison of Optimization Effects of Different timestpe Intervals & Comparison of Reward-System and SRPO on Direct-Align. (1) Hacking Rate: Annotators compare three outputs and select the one that is least detailed or most overprocessed, labeling it as hacking (2) The prompt is A young girl riding a gray wolf in a dark forest. Reward-System can only adjusts scale of rewards, resulting in trade-offs between two rewards effect. In contrast, SRPO penalizes out irrelevant directions from the reward, effectively preventing reward hacking and enhancing image texture.](images/7.jpg)
*该图像是图表，展示了不同优化时间步的效果比较及奖励系统与SRPO的效果。上半部分对比了早期、全部和晚期时间步下的输出，在时间步越晚时，黑客率显著增加，达到77%。下半部分展示了FLUX、HPS和SRPO等奖励系统的生成效果，SRPO系列展示了更佳的图像质量。*

- **结果:** 只在<strong>晚期 (Late)</strong> 步骤训练，<strong>“黑客率”</strong> (Hacking Rate) 高达 77%，生成的图像明显过于平滑、缺乏细节。在<strong>全部 (All)</strong> 步骤训练，黑客率依然很高。这强有力地证明了**仅优化后期步骤是导致奖励黑客的主要原因**。`Direct-Align` 能够优化全轨迹，是其成功的关键。

### 6.2.2. Direct-Align 和 SRPO 组件的有效性
`Figure 9 (d)` 的消融实验分析了 `Direct-Align` 的两个核心组件（早期优化、后期折扣）和 `SRPO` 的不同实现方式。

![Figure 8. Visualization of SRPO-controlled results for different style words](images/9.jpg)
*该图像是图表，展示了FLUX.1Krea和SRPO的比较结果，包括各项指标的评分，如现实感、细节和美学等。同时，图中也展示了关于与现实相关的提示的实验结果，以及与风格词的增强效果，最后包含了SRPO的消融实验结果。*

- **移除早期优化:** 如果去掉对早期时间步的优化（使其类似 `ReFL`），真实感和细节复杂度都显著下降，并出现过饱和等奖励黑客现象。
- <strong>移除后期折扣 $\lambda(t)$:</strong> 如果不降低后期时间步的优化权重，模型同样会倾向于过拟合奖励，导致纹理不自然。
- **`SRPO` 的不同实现:** 实验对比了基于<strong>逆向过程正则化 (Inversion)</strong> 的 `SRPO` 和直接构造奖励差值（如公式10）的 `SRPO`。虽然前者效果最好，但后者也取得了有竞争力的结果。这表明 `SRPO` 的核心思想（语义相对偏好）具有普适性，即使在不支持逆向优化的其他RL算法中也具备应用潜力。

# 7. 总结与思考

## 7.1. 结论总结
这篇论文针对现有基于可微奖励的扩散模型对齐方法的两大核心痛点——**效率低下/局部优化**和**奖励调整困难**，提出了一个完整且高效的解决方案。
- **贡献:**
    1.  **`Direct-Align` 框架**通过创新的**单步图像恢复**机制，实现了对**整个扩散轨迹**的高效优化，从根本上缓解了因局部优化导致的**奖励黑客**问题。
    2.  **`SRPO` 奖励机制**通过**在线的、语义相对的提示词增强**，将奖励模型从一个固定的“裁判”转变为一个可动态引导的“教练”，实现了对生成结果的**细粒度偏好控制**（如真实感），且无需昂贵的离线奖励模型微调。
- **意义:**
    - **性能与效率的双重突破:** 该方法在大幅提升图像真实感和美学质量（人类评估提升超3倍）的同时，将训练时间缩短至仅10分钟，为大规模、低成本地部署高质量文生图模型对齐提供了可能。
    - **首个系统性提升真实感的工作:** 据作者所称，这是首个系统性地、在不使用额外真实数据的情况下，显著提升大规模扩散模型生成图像**真实感**的工作。

## 7.2. 局限性与未来工作
作者在论文中指出了当前工作的两个主要局限性：
1.  **可控性局限:** `SRPO` 的控制效果依赖于奖励模型（如 `HPSv2`）对其“控制词”的理解能力。如果某个控制词（如 `Cyberpunk`）在奖励模型的训练数据中出现频率很低，那么控制效果就会减弱，甚至产生不期望的伪影。
2.  **可解释性局限:** 该方法依赖于视觉-语言模型（VLM）在隐空间中的相似度计算，某些控制文本在经过编码器映射后，其在隐空间中的方向可能与人类直觉中的RL优化方向不完全一致，缺乏完全的可解释性。

**未来工作方向:**
1.  **开发更系统的控制策略:** 例如引入可学习的控制词元（learnable tokens），而非依赖固定的文本。
2.  **微调一个对控制词更敏感的奖励模型:** 专门训练一个能够响应控制词和提示系统的VLM奖励模型，以增强可控性和可解释性。
3.  **扩展 `SRPO` 框架:** 将 `SRPO` 的思想应用到其他在线RL算法中，提升其灵活性。

## 7.3. 个人启发与批判
这篇论文给我带来了深刻的启发，其方法设计非常巧妙且直击痛点。

- **启发点:**
    1.  <strong>“跳出盒子”</strong>的思维: 面对多步采样反向传播的困境，作者没有在“如何优化这个过程”上死磕，而是通过“先合成再恢复”的思路，完全绕开了这个难题。这种<strong>“改变问题定义”</strong>的思路在科研中极具价值。
    2.  <strong>“相对论”</strong>的重要性: `SRPO` 的核心思想是优化**相对差异**而非**绝对值**。这在机器学习中是一个非常强大的理念。通过构建一对正负样本，可以有效过滤掉系统性偏差，让模型聚焦于学习真正的“差异性特征”。这个思想可以广泛应用于各种需要去偏或进行细粒度控制的任务。
    3.  **挖掘现有模型的潜力:** `SRPO` 并未训练新模型，而是通过巧妙的提示词工程“解锁”了现有奖励模型（`HPSv2`）内部蕴含的、但未被充分利用的语义控制能力。这展示了在资源有限的情况下，通过“软”方法（算法设计）而非“硬”方法（训练大模型）取得突破的可能性。

- **批判性思考与潜在问题:**
    1.  **对噪声先验的依赖:** `Direct-Align` 的成功依赖于在恢复图像时能够访问到用于加噪的 `ground truth` 噪声 $\epsilon_{gt}$。这在**训练阶段**是完全可行的，因为整个过程是人为构建的。但在<strong>推理（inference）阶段</strong>，模型是从一个纯随机噪声开始生成的，并不存在这样一个预知的 $\epsilon_{gt}$。论文提到推理时使用标准的50步采样，这意味着训练和推理之间可能存在一定的<strong>偏差 (discrepancy)</strong>。虽然实验结果表明效果很好，但这种偏差的理论影响值得进一步探讨。
    2.  **控制词的选择与泛化性:** `SRPO` 的效果强依赖于控制词的选择。如何系统性地发现有效、正交的控制词对？当面对训练集中未见过的、更复杂的偏好时，简单的提示词增强是否依然有效？这可能是该方法泛化性的一个潜在挑战。例如，如何表达“一种带有轻微忧郁情绪的日落光影”这种复杂的偏好？
    3.  <strong>“离线 SRPO”</strong>的讨论: 论文在附录中提到，将在线生成的样本换成离线的真实照片进行 `SRPO` 训练，也能提升真实感。这模糊了RLHF与监督微调（SFT）的界限。虽然作者强调这仍是RL方法，但其与在高质量真实照片上进行SFT的根本区别是什么？如果主要提升来自于数据本身，那么 `SRPO` 带来的增益有多少？这一点需要更清晰的界定和实验分析。