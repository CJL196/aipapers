# 1. 论文基本信息

## 1.1. 标题
**A Comprehensive Survey on World Models for Embodied AI** (体感人工智能世界模型的综合综述)

## 1.2. 作者
Xinqing Li, Xin He, Le Zhang, Member, IEEE, Min Wu, Senior Member, IEEE, Xiaoli Li, Fellow, IEEE, and Yun Liu

## 1.3. 发表期刊/会议
该论文尚未在特定期刊或会议上正式发表。根据提供的 `Published at (UTC)：2025-10-19T07:12:32.000Z` 和 `https://arxiv.org/abs/2510.16732` 信息，它目前是 arXiv 上的一个预印本（preprint），通常意味着它正在等待同行评审或已经提交给某个会议/期刊。考虑到作者中多位是 IEEE 会员或会士，该综述未来很可能在 IEEE 旗下期刊或顶级会议上发表。

## 1.4. 发表年份
2025年10月19日 (根据 arXiv 上的发布时间)

## 1.5. 摘要
体感人工智能 (Embodied AI) 要求智能体能够感知、行动并预测其行动如何改变未来的世界状态。<strong>世界模型 (World Models)</strong> 作为一种内部模拟器，能够捕捉环境的动态变化，从而实现对未来状态的<strong>前向推演 (forward rollouts)</strong> 和<strong>反事实推演 (counterfactual rollouts)</strong>，以支持感知、预测和决策。本综述提出了一个用于体感人工智能世界模型的统一框架。具体来说，我们形式化了问题设置和学习目标，并提出了一个三轴分类法，包括：
1.  <strong>功能性 (Functionality)</strong>：<strong>决策耦合型 (Decision-Coupled)</strong> 与 <strong>通用型 (General-Purpose)</strong>。
2.  <strong>时间建模 (Temporal Modeling)</strong>：<strong>序列模拟与推理 (Sequential Simulation and Inference)</strong> 与 <strong>全局差异预测 (Global Difference Prediction)</strong>。
3.  <strong>空间表示 (Spatial Representation)</strong>：<strong>全局潜向量 (Global Latent Vector)</strong>、<strong>词元特征序列 (Token Feature Sequence)</strong>、<strong>空间潜网格 (Spatial Latent Grid)</strong> 和 <strong>分解渲染表示 (Decomposed Rendering Representation)</strong>。

    我们系统化了机器人学、自动驾驶和通用视频设置中的数据资源和评估指标，涵盖了像素预测质量、状态级理解和任务性能。此外，我们对最先进的模型进行了定量比较，并提炼了关键的开放挑战，包括：统一数据集的稀缺性、需要评估物理一致性而非像素保真度的指标、模型性能与实时控制所需计算效率之间的权衡、以及在缓解误差累积的同时实现长时序一致性的核心建模难题。最后，我们在 `https://github.com/Li-Zn-H/AwesomeWorldModels` 维护了一个精选的参考文献列表。

## 1.6. 原文链接
- **arXiv 链接:** `https://arxiv.org/abs/2510.16732`
- **PDF 链接:** `https://arxiv.org/pdf/2510.16732v2.pdf`
- **发布状态:** 预印本 (preprint)，于2025-10-19T07:12:32.000Z 发布。

# 2. 整体概括

## 2.1. 研究背景与动机
<strong>体感人工智能 (Embodied AI)</strong> 是当前人工智能领域的一个热点方向，其核心在于使智能体能够像人类一样，在物理环境中进行感知、行动并理解其行动对未来世界状态的影响。为了实现这一目标，智能体需要一个强大的内部机制来模拟和预测环境动态，这就是<strong>世界模型 (World Models)</strong> 的作用。

论文指出，人类认知科学表明，人类通过整合感官输入构建内部世界模型，这些模型不仅预测和模拟未来事件，还塑造感知并指导行动。受此启发，早期的 <strong>人工智能 (AI)</strong> 研究中的世界模型植根于<strong>基于模型的强化学习 (model-based Reinforcement Learning, RL)</strong>，通过学习潜在的状态转移模型来提高样本效率和规划性能。Ha 和 Schmidhuber 的开创性工作首次明确提出了“世界模型”这一术语，并启发了 `Dreamer` 系列模型，强调了学习到的动态模型如何驱动基于想象的<strong>策略优化 (policy optimization)</strong>。

然而，随着大规模<strong>生成建模 (generative modeling)</strong> 和<strong>多模态学习 (multimodal learning)</strong> 的进步，世界模型已经超越了最初的策略学习范畴，发展成为能够进行高保真度未来预测的通用环境模拟器，例如 `Sora` 和 `V-JEPA 2`。这种扩展导致了功能角色、时间建模策略和空间表示的多样化，但也引入了跨子社区术语和分类法不一致的问题。

<strong>现有研究的挑战或空白 (Gap)：</strong>
1.  **多样化与不一致性：** 世界模型领域快速发展，但缺乏一个统一的分类框架来组织和理解这些多样化的方法。不同子社区在术语和分类上存在不一致。
2.  **动态捕捉的复杂性：** 忠实地捕捉环境动态需要同时解决状态的<strong>时间演化 (temporal evolution)</strong> 和场景的<strong>空间编码 (spatial encoding)</strong>。
    *   **时间维度：** 长时序<strong>推演 (rollouts)</strong> 容易导致<strong>误差累积 (error accumulation)</strong>，使得<strong>一致性 (coherence)</strong> 成为视频预测和策略想象中的核心挑战。
    *   **空间维度：** 粗糙或以二维为中心的布局不足以提供处理遮挡、物体永久性感知和几何感知规划所需的几何细节。而三维或体积占用表示（如<strong>神经场 (neural fields)</strong> 和<strong>结构化体素网格 (structured voxel grids)</strong>）则能更好地支持预测和控制。
3.  **缺乏统一分类法：** 现有综述虽然从功能或应用角度进行分类，但未能提供一个统一的、涵盖体感人工智能世界模型关键设计维度的分类法。

**本文的切入点或创新思路：**
本综述旨在通过引入一个围绕功能性、时间建模和空间表示这三个核心轴线的统一框架，来解决上述挑战。该框架旨在系统化现有方法，整合标准化的数据集和评估指标，促进定量比较，并为未来的研究提供一个全景的、可操作的知识图谱。它将世界模型定义为能够产生<strong>可操作预测 (actionable predictions)</strong> 的内部模拟器，从而将其与静态场景描述符或纯粹的生成视觉模型区分开来，后者不捕捉可控的动态。

## 2.2. 核心贡献/主要发现
这篇综述的核心贡献可以总结为以下几点：

1.  **提出统一的分类框架：**
    *   为了应对领域内术语和分类法的不一致，论文提出了一个基于<strong>功能性 (Functionality)</strong>、<strong>时间建模 (Temporal Modeling)</strong> 和<strong>空间表示 (Spatial Representation)</strong> 的三轴分类法。这是对现有世界模型研究进行系统化组织的重要贡献。
    *   **功能性**：区分了<strong>决策耦合型 (Decision-Coupled)</strong> 和<strong>通用型 (General-Purpose)</strong> 世界模型。
    *   **时间建模**：区分了<strong>序列模拟与推理 (Sequential Simulation and Inference)</strong> 和<strong>全局差异预测 (Global Difference Prediction)</strong>。
    *   **空间表示**：涵盖了<strong>全局潜向量 (Global Latent Vector)</strong>、<strong>词元特征序列 (Token Feature Sequence)</strong>、<strong>空间潜网格 (Spatial Latent Grid)</strong> 和<strong>分解渲染表示 (Decomposed Rendering Representation)</strong>。

2.  **形式化问题设置和学习目标：**
    *   论文在<strong>部分可观察马尔可夫决策过程 (Partially Observable Markov Decision Process, POMDP)</strong> 的数学框架下，形式化了世界模型的问题设置和学习目标，包括动态先验、滤波后验和重建目标，以及如何通过最大化 <strong>证据下界 (Evidence Lower Bound, ELBO)</strong> 进行学习。这为理解世界模型的数学基础提供了清晰的视角。

3.  **系统化数据资源和评估指标：**
    *   论文对机器人学、自动驾驶和通用视频设置中的广泛使用的数据资源进行了分类和总结，包括模拟平台、交互式基准、离线数据集和真实世界机器人平台。
    *   详细阐述了评估世界模型的三个抽象层次的指标：<strong>像素预测质量 (Pixel Prediction Quality)</strong> (如 `FID`, `FVD`, `SSIM`, `PSNR`, `LPIPS`, `VBench`)、<strong>状态级理解 (State-level Understanding)</strong> (如 `mIoU`, `mAP`, <strong>位移误差 (Displacement Error)</strong>, <strong>倒角距离 (Chamfer Distance)</strong>) 和<strong>任务性能 (Task Performance)</strong> (如<strong>成功率 (Success Rate)</strong>, <strong>样本效率 (Sample Efficiency)</strong>, <strong>奖励 (Reward)</strong>, <strong>碰撞 (Collision)</strong>)。

4.  **定量比较和开放挑战提炼：**
    *   对不同任务（如 `nuScenes` 上的视频生成、`Occ3D-nuScenes` 上的4D<strong>占用率预测 (Occupancy Forecasting)</strong>、`DMC` 和 `RLBench` 上的控制任务、`nuScenes` 上的规划任务）中的最先进模型进行了定量比较，并以表格形式呈现，突出了各方法的优缺点。
    *   提炼了当前世界模型领域面临的关键开放挑战，包括：
        *   **数据与评估：** 统一数据集的稀缺性，以及需要更侧重物理一致性而非像素保真度的评估指标。
        *   **计算效率：** 模型性能与实时控制所需计算效率之间的权衡。
        *   **建模策略：** 实现长时序一致性同时缓解误差累积的核心建模难题，以及在效率和表达能力之间进行有效空间表示的挑战。

5.  **维护精选参考文献列表：**
    *   提供了在线维护的 `GitHub` 仓库 `https://github.com/Li-Zn-H/AwesomeWorldModels`，作为一个持续更新的资源，方便研究人员追踪最新进展。

        这些贡献共同为体感人工智能世界模型领域提供了一个全面的概览、一个结构化的理解框架，并指明了未来的研究方向。

# 3. 预备知识与相关工作

## 3.1. 基础概念

为了理解这篇综述，特别是其对世界模型的形式化和分类，我们需要了解以下几个核心概念：

### 3.1.1. 体感人工智能 (Embodied AI)
<strong>体感人工智能 (Embodied AI)</strong> 关注的是使智能体（例如机器人、自动驾驶车辆）能够在物理世界中感知、行动、与环境互动，并学习如何通过这些互动实现目标。它强调智能体的身体（`embodiment`）和其所处环境之间的紧密联系，认为智能体的决策和学习过程应考虑物理约束、感官输入和行动的后果。世界模型在体感 AI 中至关重要，因为它允许智能体在实际行动前进行“思考”和“规划”。

### 3.1.2. 世界模型 (World Models)
<strong>世界模型 (World Models)</strong> 是智能体内部对环境动态的认知或计算模型。它不是真实环境本身，而是智能体从经验中学习到的、用于预测环境如何响应其行动以及时间如何演化的一个“模拟器”。世界模型使智能体能够：
*   <strong>前向推演 (Forward Rollouts)</strong>：预测在给定一系列行动后，环境的未来状态会如何发展。
*   <strong>反事实推演 (Counterfactual Rollouts)</strong>：想象在采取不同行动路径时，环境可能会如何发展，这对于规划和决策至关重要。
*   **感知和决策支持**：通过预测来指导智能体的感知（例如，预测接下来会看到什么）和决策（例如，选择能达到目标的行动序列）。

    与纯粹的<strong>生成视觉模型 (generative visual models)</strong> 区别在于，世界模型不仅生成视觉内容，更重要的是它捕捉了<strong>可控动态 (controllable dynamics)</strong>，即环境如何响应智能体的行动。

### 3.1.3. 部分可观察马尔可夫决策过程 (Partially Observable Markov Decision Process, POMDP)
**POMDP** 是<strong>马尔可夫决策过程 (Markov Decision Process, MDP)</strong> 的泛化，用于在智能体无法完全观察到环境真实状态的情况下进行决策。
*   **MDP**：假设智能体知道当前环境的精确状态。
*   **POMDP**：智能体只能通过观察（例如，感官输入如图像、声音）来推断环境的真实状态。由于观察可能是不完整、模糊或有噪声的，智能体需要维护一个关于当前真实状态的<strong>信念状态 (belief state)</strong> 或<strong>潜在状态 (latent state)</strong> 的分布。

    在世界模型中，智能体通常无法直接访问环境的真实状态 $s_t$，而是通过观察 $o_t$ 来推断一个<strong>学习到的潜在状态 (learned latent state)</strong> $z_t$，这就是 POMDP 框架的体现。

### 3.1.4. 证据下界 (Evidence Lower Bound, ELBO)
**ELBO** 是在<strong>变分推断 (Variational Inference)</strong> 中常用的一种目标函数，用于训练<strong>生成模型 (generative models)</strong>，特别是<strong>变分自编码器 (Variational Autoencoders, VAE)</strong>。由于直接最大化观测数据的对数似然 $\log p(o)$ 通常是不可行的，ELBO 提供了一个可以优化的下界。最大化 ELBO 相当于：
1.  **最大化重建质量**：使模型能够从潜在表示中忠实地重建观测数据。
2.  **最小化 KL 散度**：使推断出的后验分布 $q(z|o)$ 尽可能接近先验分布 `p(z)`，这有助于学习有意义且易于采样的潜在空间。

    在世界模型中，ELBO 用于学习潜在状态动态模型和观测重建模型，确保潜在状态既能编码足够的信息进行重建，又能遵循合理的动态演化。

## 3.2. 前人工作与技术演进

世界模型的研究历史可以追溯到基于模型的强化学习，但其现代形式深受深度学习和大规模生成模型的影响。

### 3.2.1. 早期模型：基于模型强化学习与潜在动态学习
*   <strong>模型基础 (Model-Based RL)</strong>：早期的世界模型主要用于提高强化学习的<strong>样本效率 (sample efficiency)</strong>。智能体学习一个环境模型，然后利用这个模型来规划行动或生成合成经验以训练<strong>策略 (policy)</strong>。
*   **Ha and Schmidhuber [9]**：在2018年提出了一个开创性的世界模型，将观测数据编码到<strong>潜在空间 (latent space)</strong>，并使用<strong>循环神经网络 (Recurrent Neural Networks, RNNs)</strong> 来建模潜在动态以进行策略优化。这奠定了现代世界模型的基础。
*   **`Dreamer` 系列 [10-12]**：在此基础上发展，如 `PlaNet` [38] 引入了<strong>循环状态空间模型 (Recurrent State-Space Model, RSSM)</strong>，融合了确定性记忆和随机成分，以实现鲁棒的<strong>长时序想象 (long-horizon imagination)</strong>。`Dreamer`、`DreamerV2`、`DreamerV3` 进一步完善了这一框架，通过在潜在空间中进行想象来优化策略。

### 3.2.2. 近期发展：大规模生成建模与多模态学习的融合
*   **生成模型扩展**：随着<strong>大规模生成模型 (large-scale generative modeling)</strong>（如<strong>扩散模型 (Diffusion Models)</strong>）和<strong>多模态学习 (multimodal learning)</strong> 的兴起，世界模型的应用范围和能力得到了极大扩展。
*   **`Sora` [13] 和 `V-JEPA 2` [14]**：这些模型展示了世界模型超越策略学习，成为高保真度未来预测的通用环境模拟器的潜力。它们能够生成逼真且长时序一致的视频，表明了在捕捉复杂环境动态方面的巨大进步。
*   **功能多样化**：这种扩展导致世界模型的功能角色、时间建模策略和空间表示变得更加多样化。例如，从最初的纯视觉输入到多模态输入（视觉、语言、触觉等），从简单潜在向量到结构化空间表示（如网格、神经场）。

### 3.2.3. 挑战与演进方向
*   **误差累积**：长时序预测中的<strong>误差累积 (error accumulation)</strong> 是一个核心挑战，促使研究人员探索更鲁棒的时间建模方法。
*   **空间表示**：为了更好地支持几何感知规划和处理遮挡，世界模型需要更精细的<strong>空间表示 (spatial representation)</strong>，从简单的二维到三维<strong>占用率 (occupancy)</strong> 或<strong>神经辐射场 (Neural Radiance Fields, NeRF)</strong>。
*   **统一性缺失**：随着技术的发展，不同社区和研究方向涌现出大量模型，但缺乏一个统一的分类和评估标准，使得领域内的比较和理解变得困难。

## 3.3. 差异化分析

本综述的工作与现有综述的主要区别在于其提出的<strong>三轴分类法 (three-axis taxonomy)</strong>。

*   <strong>功能导向型综述 (Function-oriented surveys)</strong>：例如 Ding et al. [4] 侧重于理解和预测这两个核心功能，Zhu et al. [19] 基于世界模型的核心能力构建框架。
*   <strong>应用驱动型综述 (Application-driven surveys)</strong>：例如 Guan et al. [20] 和 Feng et al. [21] 专注于自动驾驶领域的<strong>世界模型 (World Models)</strong> 技术。

    这些现有综述虽然有其价值，但都未能提供一个统一的、涵盖体感人工智能世界模型关键设计维度的分类法。

**本文的核心区别和创新点：**
本综述通过引入<strong>功能性 (Functionality)</strong>、<strong>时间建模 (Temporal Modeling)</strong> 和<strong>空间表示 (Spatial Representation)</strong> 这三个核心轴线，提供了一个更全面、更细致的分类框架。
*   **功能性**：区分<strong>决策耦合型 (Decision-Coupled)</strong>（为特定决策任务优化）和<strong>通用型 (General-Purpose)</strong>（任务无关的广泛预测）。
*   **时间建模**：区分<strong>序列模拟与推理 (Sequential Simulation and Inference)</strong>（自回归逐步预测）和<strong>全局差异预测 (Global Difference Prediction)</strong>（并行直接预测整个未来状态）。
*   **空间表示**：提供了从抽象的<strong>全局潜向量 (Global Latent Vector)</strong> 到结构化的<strong>词元特征序列 (Token Feature Sequence)</strong>、<strong>空间潜网格 (Spatial Latent Grid)</strong>，再到显式的<strong>分解渲染表示 (Decomposed Rendering Representation)</strong>（如<strong>神经场 (neural fields)</strong>）的渐进式分类。

    这个三轴框架不仅为组织现有方法提供了统一的结构，还集成了标准化的数据集和评估指标，从而促进了定量比较，并为未来的研究提供了全景式、可操作的知识图谱。它更深入地探讨了世界模型的设计选择如何影响其预测范围、物理保真度以及对体感智能体的下游性能。

# 4. 方法论

本综述的核心方法论是提出并应用一个<strong>三轴分类法 (three-axis taxonomy)</strong> 来系统地组织和分析体感人工智能中的世界模型。此外，它还形式化了世界模型的<strong>数学基础 (mathematical formulation)</strong>。

## 4.1. 核心概念与理论基础

在深入分类之前，综述首先明确了世界模型的核心概念和其数学形式化。

### 4.1.1. 核心概念
世界模型的功能基于以下三个方面：
*   <strong>模拟与规划 (Simulation &amp; Planning)</strong>：利用学习到的动态模型生成 plausible 的未来场景，允许智能体在不与真实世界互动的情况下，通过想象评估潜在行动。
*   <strong>时间演化 (Temporal Evolution)</strong>：学习编码状态如何演化，从而实现时间上一致的<strong>推演 (rollouts)</strong>。
*   <strong>空间表示 (Spatial Representation)</strong>：以适当的保真度编码场景几何，使用如<strong>潜在词元 (latent tokens)</strong> 或<strong>神经场 (neural fields)</strong> 等格式为控制提供上下文。

    这三个支柱为分类法奠定了概念基础。

### 4.1.2. 世界模型的数学形式化
综述将环境互动形式化为<strong>部分可观察马尔可夫决策过程 (Partially Observable Markov Decision Process, POMDP)</strong>。为了符号一致性，我们定义 $t=0$ 时的空初始动作 $a_0$，这使得动态可以统一地表示。在每个时间步 $t \geq 1$，智能体接收一个观测 $o_t$ 并采取一个动作 $a_t$，而真实状态 $s_t$ 保持不可观测。为了处理这种<strong>部分可观察性 (partial observability)</strong>，世界模型使用单步<strong>滤波后验 (filtering posterior)</strong> 来推断一个学习到的<strong>潜在状态 (latent state)</strong> $z_t$，其中假设先前的潜在状态 $z_{t-1}$ 总结了相关的历史信息。最后，$z_t$ 用于重建 $o_t$。

**世界模型的核心组成部分及其数学表示：**
$$
\begin{array} { l l } { \text { Dynamics Prior: } } & { p _ { \theta } ( z _ { t } \mid z _ { t - 1 } , a _ { t - 1 } ) } \\ { \text { Filtered Posterior: } } & { q _ { \phi } ( z _ { t } \mid z _ { t - 1 } , a _ { t - 1 } , o _ { t } ) } \\ { \text { Reconstruction: } } & { p _ { \theta } ( o _ { t } \mid z _ { t } ) } \end{array}
$$
其中：
*   $p_{\theta}(z_t \mid z_{t-1}, a_{t-1})$：<strong>动态先验 (Dynamics Prior)</strong>。这是一个基于模型参数 $\theta$ 的概率分布，描述了在给定前一时刻潜在状态 $z_{t-1}$ 和动作 $a_{t-1}$ 的情况下，当前时刻潜在状态 $z_t$ 的演化规律。它代表了世界模型对环境动态的预测能力。
*   $q_{\phi}(z_t \mid z_{t-1}, a_{t-1}, o_t)$：<strong>滤波后验 (Filtered Posterior)</strong>。这是一个基于推理模型参数 $\phi$ 的概率分布，描述了在给定前一时刻潜在状态 $z_{t-1}$、动作 $a_{t-1}$ 和当前观测 $o_t$ 的情况下，对当前时刻潜在状态 $z_t$ 的最佳估计。它通过结合当前观测来校正动态先验的预测。
*   $p_{\theta}(o_t \mid z_t)$：<strong>重建 (Reconstruction)</strong>。这是一个基于模型参数 $\theta$ 的概率分布，描述了在给定当前潜在状态 $z_t$ 的情况下，生成当前观测 $o_t$ 的能力。它衡量了潜在状态编码观测信息的能力。

<strong>联合分布的马尔可夫分解 (Markov Factorization of Joint Distribution)：</strong>
根据<strong>马尔可夫 (Markovian)</strong> 结构假设，观测和潜在状态的联合分布可以分解为：
$$
p _ { \theta } \big ( o _ { 1 : T } , z _ { 0 : T } \mid a _ { 0 : T - 1 } \big ) = p _ { \theta } \big ( z _ { 0 } \big ) \prod _ { t = 1 } ^ { T } p _ { \theta } \big ( z _ { t } \mid z _ { t - 1 } , a _ { t - 1 } \big ) p _ { \theta } \big ( o _ { t } \mid z _ { t } \big ) .
$$
其中：
*   $o_{1:T}$ 和 $z_{0:T}$ 分别表示从时间步 1 到 $T$ 的观测序列和从时间步 0 到 $T$ 的潜在状态序列。
*   $a_{0:T-1}$ 表示从时间步 0 到 `T-1` 的动作序列。
*   $p_{\theta}(z_0)$ 是初始潜在状态的先验分布。

<strong>变分后验近似 (Variational Posterior Approximation)：</strong>
为了推断潜在状态，我们需要近似不可处理的真实后验 $p_{\theta} \big ( z _ { 0 : T } \ \big | \ o _ { 1 : T } , a _ { 0 : T - 1 } \big )$。这通过一个时间分解的变分分布 $q_{\phi}$ 来实现：
$$
q _ { \phi } ( z _ { 0 : T } | o _ { 1 : T } , a _ { 0 : T - 1 } ) = q _ { \phi } ( z _ { 0 } | o _ { 1 } ) \prod _ { t = 1 } ^ { T } q _ { \phi } ( z _ { t } | z _ { t - 1 } , a _ { t - 1 } , o _ { t } ) ,
$$
当忽略动作输入 $a$ 时，这会简化为无动作情况。

<strong>优化目标：证据下界 (ELBO)：</strong>
直接最大化对数似然 $\log p _ { \theta } ( o _ { 1 : T } \mid a _ { 0 : T - 1 } )$ 是不可处理的。因此，我们使用近似后验 $q_{\phi}$ 来优化 <strong>证据下界 (ELBO)</strong>，它为学习模型参数提供了一个可处理的目标：
$$
\log p _ { \theta } ( o _ { 1 : T } \mid a _ { 0 : T - 1 } ) = \log \int p _ { \theta } \big ( o _ { 1 : T } , z _ { 0 : T } \mid a _ { 0 : T - 1 } \big ) d z _ { 0 : T } \geq \mathbb { E } _ { q _ { \phi } } \Big [ \log \frac { p _ { \theta } \big ( o _ { 1 : T } , z _ { 0 : T } \mid a _ { 0 : T - 1 } \big ) } { q _ { \phi } \big ( z _ { 0 : T } \mid o _ { 1 : T } , a _ { 0 : T - 1 } \big ) } \Big ] = : \mathcal { L } ( \theta , \phi ) .
$$
在 $p_{\theta}$ 和 $q_{\phi}$ 都满足马尔可夫分解的假设下，这个 ELBO 可以分解为<strong>重建目标 (reconstruction objective)</strong> 和 <strong>KL 正则化项 (KL regularization term)</strong>：
$$
\mathcal { L } ( \theta , \phi ) = \sum _ { t = 1 } ^ { T } \mathbb { E } _ { q _ { \phi } ( z _ { t } ) } \bigl [ \log p _ { \theta } ( o _ { t } \mid z _ { t } ) \bigr ] - D _ { \mathrm { K L } } \bigl ( q _ { \phi } ( z _ { 0 : T } \mid o _ { 1 : T } , a _ { 0 : T - 1 } ) \parallel p _ { \theta } ( z _ { 0 : T } \mid a _ { 0 : T - 1 } ) \bigr ) .
$$
其中：
*   $\sum _ { t = 1 } ^ { T } \mathbb { E } _ { q _ { \phi } ( z _ { t } ) } \bigl [ \log p _ { \theta } ( o _ { t } \mid z _ { t } ) \bigr ]$：这是<strong>重建项 (reconstruction term)</strong>。它鼓励模型忠实地预测观测 $o_t$。通过最大化这个项，模型学习从潜在状态 $z_t$ 生成逼真的观测。
*   $D _ { \mathrm { K L } } \bigl ( q _ { \phi } ( z _ { 0 : T } \mid o _ { 1 : T } , a _ { 0 : T - 1 } ) \parallel p _ { \theta } ( z _ { 0 : T } \mid a _ { 0 : T - 1 } ) \bigr )$：这是 <strong>KL 散度正则化项 (KL regularization term)</strong>。它衡量了近似后验 $q_{\phi}$ 和动态先验 $p_{\theta}$ 之间的差异。通过最小化这个项，模型迫使<strong>滤波后验 (filtered posterior)</strong> $q_{\phi} ( z _ { t } \mid z _ { t - 1 } , a _ { t - 1 } , o _ { t } )$ 与<strong>动态先验 (dynamics prior)</strong> $p_{\theta} ( z _ { t } \mid z _ { t - 1 } , a _ { t - 1 } )$ 对齐。这有助于学习一个有意义且可预测的潜在动态空间。

**现代世界模型的训练范式：**
现代世界模型因此采用<strong>重建-正则化训练范式 (reconstruction-regularization training paradigm)</strong>。学习到的潜在轨迹 $z_{1:T}$ 作为一个紧凑的、预测性的记忆，支持下游的<strong>策略优化 (policy optimization)</strong>、<strong>模型预测控制 (Model-Predictive Control, MPC)</strong> 和<strong>反事实推理 (counterfactual reasoning)</strong>。
这些世界模型可以通过<strong>循环模型 (recurrent models)</strong> [25-27]、<strong>基于 Transformer 的架构 (Transformer-based architectures)</strong> [28-30] 或<strong>基于扩散的解码器 (diffusion-based decoders)</strong> [31-35] 来实现。

## 4.2. 三轴分类法

本综述的核心方法论是其提出的三轴分类法，用于组织和理解体感人工智能中的世界模型。该分类法包括：
1.  <strong>功能性 (Functionality)</strong>：决策耦合型 vs. 通用型
2.  <strong>时间建模 (Temporal Modeling)</strong>：序列模拟与推理 vs. 全局差异预测
3.  <strong>空间表示 (Spatial Representation)</strong>：全局潜向量、词元特征序列、空间潜网格、分解渲染表示

    以下将详细解释每个轴。

### 4.2.1. 轴一：功能性 (Functionality)
这个维度区分了世界模型与其下游任务的耦合程度。
*   <strong>决策耦合型 (Decision-Coupled)</strong>：
    *   **定义**：这类模型是<strong>任务特定的 (task-specific)</strong>，其学习到的动态模型是为了优化特定的决策任务（例如，机器人操纵、自动驾驶中的路径规划）而设计的。
    *   **特点**：它们通常在特定任务的数据上进行训练，并可能直接包含任务相关的奖励或目标信息。优化目标往往与特定决策任务的性能紧密相关。
    *   **例子**：传统的基于模型的强化学习中的世界模型，例如 `Dreamer` 系列，它们学习环境动态是为了在潜在空间中进行规划并优化策略。

*   <strong>通用型 (General-Purpose)</strong>：
    *   **定义**：这类模型是<strong>任务无关的 (task-agnostic)</strong> 模拟器，专注于对环境进行广泛而高保真度的预测。它们旨在捕捉环境的基本物理和动态，以便能够泛化到各种下游应用中，而不仅仅是某个特定任务。
    *   **特点**：通常在大量多样化的、无标签的视频数据上进行<strong>预训练 (pretrain)</strong>，学习更一般的世界知识。它们可能不直接优化决策任务，但其学习到的强大动态模型可以作为各种下游任务的基础。
    *   **例子**：`Sora` 和 `V-JEPA 2` 等大规模视频生成模型，它们通过预测未来的像素或特征来模拟环境，为广泛的体感 AI 任务提供基础。

### 4.2.2. 轴二：时间建模 (Temporal Modeling)
这个维度关注世界模型如何处理时间序列数据并预测未来状态。
*   <strong>序列模拟与推理 (Sequential Simulation and Inference)</strong>：
    *   **定义**：这类模型以<strong>自回归 (autoregressive)</strong> 的方式模拟动态，一步一步地展开未来状态。它们通常在每个时间步预测下一个状态，并使用这个预测来生成更远的未来。
    *   **特点**：
        *   **预测方式**：$s_{t+1} = f(s_t, a_t)$，即未来的状态是当前状态和动作的函数。
        *   **误差累积**：由于每个预测都依赖于前一个预测，误差会随着时间步的增加而累积，导致<strong>长时序推演 (long-horizon rollouts)</strong> 的<strong>一致性 (coherence)</strong> 成为一个挑战。
        *   **效率**：通常计算效率较高，尤其是在短时序预测中。
    *   **例子**：<strong>循环神经网络 (RNNs)</strong>、<strong>循环状态空间模型 (RSSMs)</strong> 和<strong>基于 Transformer 的自回归模型 (autoregressive Transformer-based models)</strong> 属于此类。

*   <strong>全局差异预测 (Global Difference Prediction)</strong>：
    *   **定义**：这类模型尝试直接或并行地估计整个未来状态序列，而不是逐步生成。它们可能通过学习当前状态与未来状态之间的全局差异来实现。
    *   **特点**：
        *   **预测方式**：可能通过学习一个函数 $f(s_t) \rightarrow (s_{t+1}, s_{t+2}, ..., s_{t+H})$ 来直接预测未来 $H$ 个时间步，或者通过掩码建模 (masked modeling) 等方式。
        *   **误差累积**：由于不是严格的自回归，可以减轻<strong>误差累积 (error accumulation)</strong> 的问题，从而在<strong>长时序预测 (long-horizon prediction)</strong> 中提供更好的时间一致性。
        *   **效率**：通常需要更大的计算资源，尤其是在训练阶段，因为它们可能需要处理整个序列或预测大规模的结构。
    *   **例子**：某些<strong>扩散模型 (Diffusion Models)</strong> 或<strong>掩码自编码器 (Masked Autoencoders)</strong>，它们可以一次性生成或预测未来的多个帧或状态。

### 4.2.3. 轴三：空间表示 (Spatial Representation)
这个维度描述了世界模型如何编码和表示环境的<strong>空间信息 (spatial information)</strong>。
*   <strong>全局潜向量 (Global Latent Vector)</strong>：
    *   **定义**：将复杂的<strong>世界状态 (world states)</strong> 编码成一个紧凑的、低维的<strong>向量 (vector)</strong>。
    *   **特点**：
        *   <strong>紧凑性 (Compactness)</strong>：高度压缩信息，占用空间小。
        *   <strong>效率 (Efficiency)</strong>：便于高效的实时计算，尤其适用于物理设备上的部署。
        *   <strong>信息损失 (Information Loss)</strong>：可能丢失细粒度的空间和几何细节，导致难以处理遮挡、物体永久性感知等问题。
    *   **例子**：`PlaNet`、`Dreamer` 系列模型最初使用的潜在状态表示。

*   <strong>词元特征序列 (Token Feature Sequence)</strong>：
    *   **定义**：将<strong>世界状态 (world states)</strong> 建模为一系列<strong>离散词元 (discrete tokens)</strong>。这些词元可以代表图像块、语义概念、物体特征等。
    *   **特点**：
        *   <strong>依赖建模 (Dependency Modeling)</strong>：擅长捕捉复杂的空间、时间以及<strong>跨模态依赖 (cross-modal dependencies)</strong>。
        *   <strong>组合性 (Compositionality)</strong>：可以通过组合不同的词元来表示和生成复杂的场景。
        *   <strong>与 LLM 兼容 (LLM Compatibility)</strong>：与<strong>大语言模型 (Large Language Models, LLMs)</strong> 中使用的词元化方法兼容，便于集成。
    *   **例子**：<strong>基于 Transformer 的世界模型 (Transformer-based world models)</strong>，如 `iVideoGPT` [56]、`Genie` [50]，以及许多结合语言模型进行规划的模型。

*   <strong>空间潜网格 (Spatial Latent Grid)</strong>：
    *   **定义**：将<strong>空间信息 (spatial information)</strong> 编码在几何对齐的<strong>网格 (grids)</strong> 上，或融入显式的<strong>空间先验 (spatial priors)</strong>（例如<strong>鸟瞰图 (Bird's-Eye View, BEV)</strong> 特征或<strong>体素网格 (voxel grids)</strong>）。
    *   **特点**：
        *   <strong>局部性 (Locality)</strong>：保留了空间信息的局部结构，便于<strong>卷积 (convolutional)</strong> 或<strong>注意力 (attention-based)</strong> 更新。
        *   <strong>几何归纳偏置 (Geometric Inductive Bias)</strong>：利用几何先验（如 BEV 视角下物体间的相对位置关系），更好地处理三维场景。
        *   <strong>效率 (Efficiency)</strong>：支持高效的流式<strong>推演 (streaming rollouts)</strong>。
    *   **例子**：自动驾驶领域的 BEV 模型、4D 占用率预测模型，如 `OccWorld` [93]、`DriveWorld` [87]。

*   <strong>分解渲染表示 (Decomposed Rendering Representation, DRR)</strong>：
    *   **定义**：将 3D 场景分解为一组可学习的<strong>基元 (learnable primitives)</strong>，如**3D Gaussian Splatting (3DGS)** [36] 或<strong>神经辐射场 (Neural Radiance Fields, NeRF)</strong> [37]，然后使用<strong>可微分渲染 (differentiable rendering)</strong> 来实现高保真度的<strong>新颖视角合成 (novel view synthesis)</strong>。
    *   **特点**：
        *   <strong>视图一致性 (View-Consistency)</strong>：能够从任意视角生成一致的图像。
        *   <strong>物体级组合性 (Object-level Compositionality)</strong>：可以独立地操纵场景中的物体基元。
        *   <strong>物理集成 (Physics Integration)</strong>：易于与物理先验和<strong>数字孪生 (digital twins)</strong> 集成，支持长时序的<strong>推演 (rollouts)</strong>。
    *   **例子**：`ManiGaussian` [53]、`GaussianWorld` [103] 等。

        Figure 1 总结了论文的结构和分类法。

        ![该图像是一个示意图，展示了关于世界模型在体感人工智能中的框架。图中包括了核心概念、决策耦合、空间表示和时间建模等模块。每个模块下列出了相关的方法和代表性模型，以及它们在不同任务中的应用，如视频生成和全局预测。](images/1.jpg)
        *该图像是一个示意图，展示了关于世界模型在体感人工智能中的框架。图中包括了核心概念、决策耦合、空间表示和时间建模等模块。每个模块下列出了相关的方法和代表性模型，以及它们在不同任务中的应用，如视频生成和全局预测。*

以下是原文 `Figure 1` 的描述：
**Figure 1. 论文的结构和分类法概述。** 本图描绘了世界模型的核心概念、数学形式化和三轴分类法。顶部的三个核心概念——模拟与规划、时间演化和空间表示——通过数学形式化连接到世界模型。三轴分类法定义了世界模型在体感人工智能中的核心设计空间，将方法划分为决策耦合型与通用型、序列模拟与推理与全局差异预测、以及四种空间表示。这些分类映射到各种任务和领域，例如视频生成和全局预测。该图设计部分灵感来源于 [12]、[14]、[22]、[23]。

## 4.3. 分类法应用与代表性方法

综述随后应用此分类法对机器人学和自动驾驶领域的代表性工作进行了分类。这在 `Table I` 和 `Table II` 中有所体现。

### 4.3.1. 决策耦合型世界模型 (Decision-Coupled World Models)

这类模型为特定决策任务而优化。

#### 4.3.1.1. 序列模拟与推理 (Sequential Simulation and Inference)
*   <strong>全局潜向量 (Global Latent Vector)</strong>：
    *   **代表方法**：`PlaNet` [38]、`Dreamer` 系列 [10-12] 及其变体（`Dreaming` [110]、`DreamerPro` [111]、`HRSSM` [25]、`DisWM` [112]），以及针对<strong>迁移学习 (transferability)</strong> 的模型（`PreLAR` [52]、`SENSEI` [114]、`AdaWM` [117]）。
    *   **自动驾驶应用**：`MILE` [81]、`SEM2` [83] 等。
    *   **其他架构**：`TransDreamer` [28] 引入了<strong>Transformer 状态空间模型 (Transformer State-Space Model, TSSM)</strong>；`GLAM` [57] 结合了 `Mamba` 等<strong>状态空间模型 (State Space Models, SSMs)</strong>。
    *   <strong>逆动力学建模 (Inverse Dynamics Modeling, IDM)</strong>：`GLAMOR` [39]、`Iso-Dream` [40] 等。

*   <strong>词元特征序列 (Token Feature Sequence)</strong>：
    *   **代表方法**：`MWM` [41]、`NavMorph` [129]、`TWM` [29] 等。
    *   **结合 LLM**：`EvoAgent` [131]、`RoboHorizon` [132] 将 `LLM` 与 `RSSM` 结合。
    *   **自动驾驶应用**：`DrivingWorld` [133]、`Doe-1` [134]、`DrivingGPT` [23] 等。
    *   **对象中心方法**：`CarFormer` [142]、`Dyn-O` [69] 将场景表示为对象集合。
    *   **扩散模型**：`Epona` [148]、`Goff et al.` [149] 利用<strong>自回归扩散模型 (autoregressive diffusion models)</strong>。
    *   **显式推理**：`NavCoT` [58]、`ECoT` [54]、`MineDreamer` [79] 引入<strong>思维链 (Chain-of-Thought, CoT)</strong>。

*   <strong>空间潜网格 (Spatial Latent Grid)</strong>：
    *   **代表方法**：`DriveDreamer` [91]、`GenAD` [92] 使用基于 `GRU` 的动态模型；`DriveWorld` [87]、`Raw2Drive` [155] 在 `BEV` 词元上实例化 `RSSM` 动态。
    *   **4D 占用率预测**：`OccWorld` [93]、`RenderWorld` [156] 等。
    *   **LLM 结合**：`OccLLaMA` [18] 将占用率、动作和文本统一在单个词汇中。
    *   **机器人学应用**：`WMNav` [163]、`RoboOccWorld` [164]。

*   <strong>分解渲染表示 (Decomposed Rendering Representation)</strong>：
    *   **代表方法**：`GAF` [74]、`ManiGaussian` [53]、$ManiGaussian++$ [80] 利用 `3DGS` 预测未来状态。
    *   <strong>数字孪生 (Digital Twin)</strong>：`DreMa` [60]、`DexSim2Real2` [167] 等结合物理模拟器。

#### 4.3.1.2. 全局差异预测 (Global Difference Prediction)
*   <strong>词元特征序列 (Token Feature Sequence)</strong>：
    *   **代表方法**：`TOKEN` [95]、`GeoDrive` [170]。
    *   **扩散模型与 IDM**：`FLARE` [171]、`LaDi-WM` [172]、`villa-X` [76]、`VidMan` [55] 结合扩散模型和<strong>逆动力学模型 (IDM)</strong>。

*   <strong>空间潜网格 (Spatial Latent Grid)</strong>：
    *   **代表方法**：`EmbodiedDreamer` [173]、`TesserAct` [78] 结合可微分物理和视频扩散。
    *   **规划**：`3DFlowAction` [176]、`Imagine-2-Drive` [177]。

### 4.3.2. 通用型世界模型 (General-Purpose World Models)

这类模型预训练任务无关的动态模型以捕捉环境物理并生成未来场景。

#### 4.3.2.1. 序列模拟与推理 (Sequential Simulation and Inference)
*   <strong>词元特征序列 (Token Feature Sequence)</strong>：
    *   **代表方法**：`iVideoGPT` [56]、`Genie` [50]、`RoboScape` [180] 利用<strong>词元化潜在空间 (tokenized latent space)</strong> 进行预测和生成。
    *   **结合语言先验**：`EVA` [71]、`Owl-1` [183] 引入<strong>视觉语言模型 (Visual Language Models, VLMs)</strong>。
    *   **扩散模型**：`AdaWorld` [72]、`Vid2World` [184] 将视频扩散模型适应为可控世界模型。
    *   **几何保真度**：`Geometry Forcing` [185]、`DeepVerse` [64] 结合显式 3D 先验。
    *   **序列模型**：`Po et al.` [194]、`S2-SSM` [61] 使用<strong>状态空间模型 (SSMs)</strong>。

*   <strong>空间潜网格 (Spatial Latent Grid)</strong>：
    *   **代表方法**：`PhyDNet` [195]、`MindJourney` [73] 结合结构化网格和物理信息方法。
    *   **扩散预测**：`DOME` [94]、`Copilot4D` [82]、`LidarDM` [100] 利用扩散模型进行网格预测。

*   <strong>分解渲染表示 (Decomposed Rendering Representation)</strong>：
    *   **代表方法**：`GaussianWorld` [103]、`InfiniCube` [22] 利用 `3DGS` 或类似技术进行场景演化建模。

#### 4.3.2.2. 全局差异预测 (Global Difference Prediction)
*   <strong>词元特征序列 (Token Feature Sequence)</strong>：
    *   **代表方法**：`V-JEPA` [51]、`V-JEPA 2` [14]、`WorldDreamer` [45] 利用<strong>联合嵌入预测架构 (Joint-Embedding Predictive Architecture, JEPA)</strong> 和<strong>掩码建模 (masked modeling)</strong>。
    *   **扩散模型**：`Sora` [13]、`ForeDiff` [202] 利用扩散模型生成长而连贯的视频序列。

*   <strong>空间潜网格 (Spatial Latent Grid)</strong>：
    *   **代表方法**：`UniFuture` [206]、`HERMES` [108]、`BEVWorld` [207] 专注于统一场景理解和未来预测。
    *   **4D 生成**：`OccSora` [85]、`DynamicCity` [99] 使用词元化 4D 表示。

*   <strong>分解渲染表示 (Decomposed Rendering Representation)</strong>：
    *   **代表方法**：`DriveDreamer4D` [105]、`ReconDreamer` [106]、`MagicDrive3D` [84] 结合视频生成和<strong>高斯 splatting (Gaussian Splatting)</strong>。

## 4.4. 方法流程图示

论文 `Figure 1` 以图示方式概括了分类框架。

![该图像是一个示意图，展示了关于世界模型在体感人工智能中的框架。图中包括了核心概念、决策耦合、空间表示和时间建模等模块。每个模块下列出了相关的方法和代表性模型，以及它们在不同任务中的应用，如视频生成和全局预测。](images/1.jpg)
*该图像是一个示意图，展示了关于世界模型在体感人工智能中的框架。图中包括了核心概念、决策耦合、空间表示和时间建模等模块。每个模块下列出了相关的方法和代表性模型，以及它们在不同任务中的应用，如视频生成和全局预测。*

该图的中心是<strong>世界模型 (World Model)</strong>，其概念基础是<strong>模拟与规划 (Simulation &amp; Planning)</strong>、<strong>时间演化 (Temporal Evolution)</strong> 和<strong>空间表示 (Spatial Representation)</strong>。这些概念通过<strong>数学形式化 (Mathematical Formulation)</strong> 联系起来，包括<strong>动态先验 (Dynamics Prior)</strong>、<strong>滤波后验 (Filtered Posterior)</strong> 和<strong>重建 (Reconstruction)</strong>。

在世界模型的周围是其**三轴分类法**：
1.  <strong>功能性 (Functionality)</strong>：
    *   <strong>决策耦合型 (Decision-Coupled)</strong>
    *   <strong>通用型 (General-Purpose)</strong>
2.  <strong>时间建模 (Temporal Modeling)</strong>：
    *   <strong>序列模拟与推理 (Sequential Simulation and Inference)</strong>
    *   <strong>全局差异预测 (Global Difference Prediction)</strong>
3.  <strong>空间表示 (Spatial Representation)</strong>：
    *   <strong>全局潜向量 (Global Latent Vector)</strong>
    *   <strong>词元特征序列 (Token Feature Sequence)</strong>
    *   <strong>空间潜网格 (Spatial Latent Grid)</strong>
    *   <strong>分解渲染表示 (Decomposed Rendering Representation)</strong>

        这些分类法分支进一步细化为具体的<strong>世界模型 (World Models)</strong> 方法，例如：
*   **决策耦合型 + 序列模拟与推理 + 全局潜向量** 包含了 `Dreamer`、`PlaNet`、`RSSM` 等。
*   **通用型 + 全局差异预测 + 词元特征序列** 包含了 `Sora`、`V-JEPA` 等。

    图中的右侧展示了世界模型在不同领域的应用，如<strong>视频生成 (Video Generation)</strong> 和<strong>全局预测 (Global Prediction)</strong>，这些应用通过<strong>基准 (Benchmarks)</strong> 和<strong>评估指标 (Metrics)</strong> 进行衡量。

# 5. 实验设置

作为一篇综述，本文不进行自己的实验，而是系统地总结了现有研究中常用的<strong>数据集 (Data Resources)</strong> 和<strong>评估指标 (Metrics)</strong>，以建立一个统一的评估基础。

## 5.1. 数据资源

为了满足体感人工智能的多样化需求，综述将数据资源分为四类：<strong>模拟平台 (Simulation Platforms)</strong>、<strong>交互式基准 (Interactive Benchmarks)</strong>、<strong>离线数据集 (Offline Datasets)</strong> 和<strong>真实世界机器人平台 (Real-world Robot Platforms)</strong>。

### 5.1.1. 模拟平台
这些平台提供可控且可扩展的虚拟环境，用于训练和评估世界模型。

*   **MuJoCo [218]**：一个高度可定制的物理引擎，广泛用于机器人学和控制研究，以其高效的关节系统和接触动力学模拟而闻名。
*   **NVIDIA Isaac**：一个端到端、GPU 加速的模拟栈，包括 `Isaac Sim`、`Isaac Gym` [221] 和 `Isaac Lab` [222]。提供逼真的渲染和大规模强化学习能力。
*   **CARLA [219]**：一个基于 `Unreal Engine` 的开源城市自动驾驶模拟器，提供逼真的渲染、多样的传感器和闭环评估协议。
*   **Habitat [220]**：一个用于体感人工智能的高性能模拟器，专注于逼真的 3D 室内导航。
*   **Atari [223]**：一个像素级、离散动作的游戏套件，用于评估智能体的性能。

### 5.1.2. 交互式基准
这些基准提供标准化的任务套件和协议，用于可重现的闭环评估世界模型。

*   **DeepMind Control (DMC) [224]**：一个标准的基于 `MuJoCo` 的控制任务套件，为从状态或像素观测学习的智能体提供了一致的比较基础。
*   **Atari100k [239]**：`Atari` 套件的一个特例，通过限制交互步数为 100k 来评估样本效率。
*   **Meta-World [225]**：一个用于多任务和元强化学习的基准，包含 50 个多样化的机器人操纵任务。
*   **RLBench [226]**：提供 100 个模拟桌面操纵任务，具有稀疏奖励和丰富多模态观测。
*   **LIBERO [228]**：一个用于终身机器人操纵的基准，提供 130 个程序生成的任务和人类演示。
*   **nuPlan [227]**：一个用于自动驾驶规划的基准，使用轻量级闭环模拟器和超过 1500 小时的真实世界驾驶日志。

### 5.1.3. 离线数据集
这些数据集是预先收集的大规模轨迹，消除了交互式<strong>推演 (rollouts)</strong> 的需求，为世界模型的预训练和可重现评估提供了基础。

*   **RT-1 [233]**：一个用于机器人学习的真实世界数据集，包含 13 万多条轨迹，涵盖 700 多个任务。
*   **Open X-Embodiment (OXE) [235]**：一个聚合了 21 个机构、22 种机器人实体、527 种技能、超过 100 万条轨迹的大型语料库。
*   **Habitat-Matterport 3D (HM3D) [232]**：一个包含 1000 个室内重建的大规模数据集，扩展了体感人工智能模拟的范围。
*   **nuScenes [230]**：一个大规模多模态自动驾驶数据集，包含 1000 个 20 秒的场景，具有 360 度传感器套件和密集的 3D 标注。
*   **Waymo [231]**：另一个大规模多模态自动驾驶基准，包含 1150 个 20 秒的场景，具有 5 个激光雷达和 5 个摄像头。
*   **Occ3D [234]**：定义了从环视图像进行 3D <strong>占用率预测 (occupancy prediction)</strong>，提供体素标签。
*   **Something-Something v2 (SSv2) [229]**：一个用于细粒度动作理解的视频数据集，包含 22 万多段视频。
*   **OpenDV [90]**：`GenAD` 提出的最大规模自动驾驶视频-文本数据集，支持视频预测和世界模型预训练。
*   **VideoMix22M [14]**：`V-JEPA 2` 引入的大规模数据集，用于自监督预训练，包含 2200 万个样本。

### 5.1.4. 真实世界机器人平台
这些平台提供物理实体进行交互，实现闭环评估、高保真数据收集和真实世界约束下的<strong>模拟到现实 (Sim-to-Real, S2R)</strong> 验证。

*   **Franka Emika [236]**：一个 7-自由度协作机器人手臂，具有精确力控制能力。
*   **Unitree Go1 [237]**：一个经济高效且广泛采用的四足机器人，用于步态和体感人工智能研究。
*   **Unitree G1 [238]**：一个紧凑的人形机器人，用于研究，提供多达 43 个自由度。

    以下是原文 `Table III` 的数据资源总览：

    <table><thead><tr><th>Category</th><th>Name</th><th>Year</th><th>Task</th><th>Input</th><th>Domain</th><th>Scale</th><th>Protocol1</th></tr></thead><tbody><tr><td>Simulation Platforms</td><td>MuJoCo [218]</td><td>2012</td><td>Continuous control</td><td>Proprio.</td><td>Sim</td><td>-</td><td>-</td></tr><tr><td></td><td>CARLA [219]</td><td>2017</td><td>Driving simulation</td><td>RGB-D/Seg/LiDAR/Radar/GPS/IMU</td><td>Sim</td><td></td><td>✓</td></tr><tr><td></td><td>Habitat [220]</td><td>2019</td><td>Embodied navigation</td><td>RGB-D/Seg/GPS/Compass</td><td>Sim</td><td></td><td>✓</td></tr><tr><td></td><td>Isaac Gym [221]</td><td>2021</td><td>Continuous control</td><td>Proprio.</td><td>Sim</td><td></td><td>-</td></tr><tr><td></td><td>Isaac Lab [222]</td><td>2023</td><td>Robot learning suites</td><td>RGB-D/Seg/LiDAR/Proprio.</td><td>Sim</td><td>-</td><td></td></tr><tr><td></td><td>Atari [223]</td><td>2013</td><td>Discrete-action game</td><td>RGB/State</td><td>Sim</td><td>55+ Games</td><td>✓</td></tr><tr><td>Interactive Benchmarks</td><td>DMC [224]</td><td>2018</td><td>Continuous control</td><td>RGB/Proprio.</td><td>Sim</td><td>30+ Tasks</td><td>✓</td></tr><tr><td></td><td>Meta-World [225]</td><td>2019</td><td>Multi-task manipulation</td><td>RGB/Proprio.</td><td>Sim</td><td>50 tasks</td><td></td></tr><tr><td></td><td>RLBench [226]</td><td>2020</td><td>Robotic manipulation</td><td>RGB-D/Seg/Proprio.</td><td>Sim</td><td>100 tasks</td><td>✓</td></tr><tr><td></td><td>nuPlan [227]</td><td>2021</td><td>Driving planning</td><td>RGB/LiDAR/Map/Proprio.</td><td>Real</td><td>1.5k hours</td><td>✓</td></tr><tr><td></td><td>LIBERO [228]</td><td>2023</td><td>Lifelong manipulation</td><td>RGB/Text/Proprio.</td><td>Sim</td><td>130 tasks</td><td>✓</td></tr><tr><td>Offline Datasets</td><td>SSv2 [229]</td><td>2018</td><td>Video-action understanding</td><td>RGB/Text</td><td>Real</td><td>220k videos</td><td>169k/24k/27k</td></tr><tr><td></td><td>nuScenes [230]</td><td>2020</td><td>Driving perception</td><td>RGB/LiDAR/Radar/GPS/IMU</td><td>Real</td><td>1k scenes</td><td>700/150/150</td></tr><tr><td></td><td>Waymo [231]</td><td>2020</td><td>Driving perception</td><td>RGB/LiDAR</td><td>Real</td><td>1.15k scenes</td><td>798/202/150</td></tr><tr><td></td><td>HM3D [232]</td><td>2021</td><td>Indoor navigation</td><td>RGB-D</td><td>Real</td><td>1k scenes</td><td>800/100/100</td></tr><tr><td></td><td>RT-1 [233]</td><td>2022</td><td>Real-robot manipulation</td><td>RGB/Text</td><td>Real</td><td>130k+ trajectories</td><td></td></tr><tr><td></td><td>Occ3D [234]</td><td>2023</td><td>3D occupancy</td><td>RGB/LiDAR</td><td>Real</td><td>1.9k scenes</td><td>600/150/150; 798/202/-</td></tr><tr><td></td><td>OXE [235]</td><td>2024</td><td>Cross-embodiment pretraining</td><td>RGB-D/LiDAR/Text</td><td>Real</td><td>1M+ trajectories</td><td></td></tr><tr><td></td><td>OpenDV [90]</td><td>2024</td><td>Driving video pretraining</td><td>RGB/Text</td><td>Real</td><td>2k+ hours</td><td></td></tr><tr><td></td><td>VideoMix22M [14]</td><td>2025</td><td>Video pretraining</td><td>RGB</td><td>Real</td><td>22M+ samples</td><td></td></tr><tr><td>Real-world Robot Platforms</td><td>Franka Emika [236]</td><td>2022</td><td>Manipulation</td><td>Proprio.</td><td>Real</td><td></td><td></td></tr><tr><td></td><td>Unitree Go1 [237]</td><td>2021</td><td>Quadruped locomotion</td><td>RGB-D/LiDAR/Proprio.</td><td>Real</td><td></td><td></td></tr><tr><td></td><td>Unitree G1 [238]</td><td>2024</td><td>Humanoid manipulation</td><td>RGB-D/LiDAR/Proprio./Audio</td><td>Real</td><td></td><td></td></tr></tbody></table>

`Protocol1` 备注：对于交互式基准，勾号 $(\checkmark)$ 表示提供可用的评估协议。对于数据集，表示提供官方数据分割。

## 5.2. 评估指标

评估世界模型的能力涵盖了捕捉动态、泛化到未见场景以及随着额外资源扩展的能力。综述将指标分为三个抽象层次：<strong>像素预测质量 (Pixel Prediction Quality)</strong>、<strong>状态级理解 (State-level Understanding)</strong> 和<strong>任务性能 (Task Performance)</strong>。

### 5.2.1. 像素预测质量 (Pixel Prediction Quality)
这类指标评估世界模型重建感官输入和生成逼真序列的能力，衡量模型捕捉原始环境动态的程度。

*   **Fréchet Inception Distance (FID) [244]**
    *   **概念定义**：FID 用于评估生成图像的真实性和多样性。它通过比较真实图像和生成图像在特征空间中的分布来量化它们之间的差异。一个较低的 FID 值表示生成图像的质量更高，更接近真实图像。
    *   **数学公式**：
        $$
        \mathrm { F I D } ( x , y ) = \| \pmb { \mu _ { x } } - \pmb { \mu _ { y } } \| _ { 2 } ^ { 2 } + \operatorname { Tr } \left( \pmb { \Sigma _ { x } } + \pmb { \Sigma _ { y } } - 2 ( \pmb { \Sigma _ { x } } \pmb { \Sigma _ { y } } ) ^ { 1 / 2 } \right)
        $$
    *   **符号解释**：
        *   $x$：真实图像的特征分布。
        *   $y$：生成图像的特征分布。
        *   $\pmb{\mu_x}$：真实图像特征分布的均值向量。
        *   $\pmb{\mu_y}$：生成图像特征分布的均值向量。
        *   $\pmb{\Sigma_x}$：真实图像特征分布的协方差矩阵。
        *   $\pmb{\Sigma_y}$：生成图像特征分布的协方差矩阵。
        *   $\| \cdot \|_2^2$：L2 范数的平方。
        *   $\operatorname{Tr}(\cdot)$：矩阵的迹（对角线元素之和）。
        *   $(\cdot)^{1/2}$：矩阵的平方根。
        *   FID 的计算通常使用在 `ImageNet` [243] 上预训练的 `Inception-v3` [245] 网络的特征。

*   **Fréchet Video Distance (FVD) [246]**
    *   **概念定义**：FVD 是 FID 在视频领域的扩展，评估视频的逐帧质量和时间一致性。它使用一个运动感知的特征提取器（如 `I3D` 网络）来提取视频特征，并计算这些特征分布的 Fréchet 距离。较低的 FVD 值表示生成视频在外观和动态上更接近真实视频，且时间伪影较少。
    *   **数学公式**：与 FID 公式相同，但在视频特征空间中计算。
        $$
        \mathrm { FVD } ( x , y ) = \| \pmb { \mu _ { x } } - \pmb { \mu _ { y } } \| _ { 2 } ^ { 2 } + \operatorname { Tr } \left( \pmb { \Sigma _ { x } } + \pmb { \Sigma _ { y } } - 2 ( \pmb { \Sigma _ { x } } \pmb { \Sigma _ { y } } ) ^ { 1 / 2 } \right)
        $$
    *   **符号解释**：
        *   $x$：真实视频的特征分布。
        *   $y$：生成视频的特征分布。
        *   $\pmb{\mu_x}$：真实视频特征分布的均值向量。
        *   $\pmb{\mu_y}$：生成视频特征分布的均值向量。
        *   $\pmb{\Sigma_x}$：真实视频特征分布的协方差矩阵。
        *   $\pmb{\Sigma_y}$：生成视频特征分布的协方差矩阵。
        *   FVD 的计算通常使用在 `Kinetics-400` [248] 上预训练的 `I3D` [247] 网络的特征。

*   **Structural Similarity Index Measure (SSIM) [249]**
    *   **概念定义**：SSIM 是一种感知图像质量指标，它通过比较两幅图像的亮度、对比度和结构来评估它们的相似性。它的设计旨在更好地反映人眼对图像质量的感知。值越接近 1，表示图像相似度越高。
    *   **数学公式**：
        $$
        \mathrm { S S I M } ( x , y ) = \frac { ( 2 \pmb { \mu } _ { x } \pmb { \mu } _ { y } + C _ { 1 } ) ( 2 \pmb { \Sigma } _ { x y } + C _ { 2 } ) } { ( \pmb { \mu } _ { x } ^ { 2 } + \pmb { \mu } _ { y } ^ { 2 } + C _ { 1 } ) ( \pmb { \Sigma } _ { x } ^ { 2 } + \pmb { \Sigma } _ { y } ^ { 2 } + C _ { 2 } ) }
        $$
    *   **符号解释**：
        *   $x$：一个图像块（`patch`）。
        *   $y$：另一个图像块（通常是参考图像的对应块）。
        *   $\pmb{\mu_x}$：图像块 $x$ 的均值。
        *   $\pmb{\mu_y}$：图像块 $y$ 的均值。
        *   $\pmb{\Sigma_x^2}$：图像块 $x$ 的方差。
        *   $\pmb{\Sigma_y^2}$：图像块 $y$ 的方差。
        *   $\pmb{\Sigma_{xy}}$：图像块 $x$ 和 $y$ 之间的协方差。
        *   $C_1 = (K_1 L)^2$， $C_2 = (K_2 L)^2$：用于稳定除法的小常数，其中 $L$ 是像素值的动态范围（例如，8 位图像为 255），$K_1 \ll 1$ 和 $K_2 \ll 1$ 是小常数（通常 $K_1=0.01, K_2=0.03$）。
        *   最终的 SSIM 分数通常通过对滑动窗口内的局部 SSIM 值取平均来获得。

*   **Peak Signal-to-Noise Ratio (PSNR) [250]**
    *   **概念定义**：PSNR 是一种衡量图像（或视频）重建质量的客观指标。它通过比较重建图像和原始图像之间的<strong>均方误差 (Mean Squared Error, MSE)</strong> 来量化像素级的失真。PSNR 值越高，表示图像失真越小，重建质量越好。
    *   **数学公式**：
        $$
        \mathrm { M S E } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( x _ { i } - y _ { i } \right) ^ { 2 }
        $$
        $$
        \mathrm { P S N R } ( x , y ) = 1 0 \cdot \log _ { 1 0 } \left( \frac { \mathrm { M A X } ^ { 2 } } { \mathrm { M S E } } \right)
        $$
    *   **符号解释**：
        *   $\mathrm{MSE}$：均方误差。
        *   $N$：图像中的像素总数。
        *   $x_i$：原始图像中第 $i$ 个像素的值。
        *   $y_i$：重建图像中第 $i$ 个像素的值。
        *   $\mathrm{MAX}$：像素可能的最大值（例如，8 位 RGB 图像为 255，归一化图像为 1）。

*   **Learned Perceptual Image Patch Similarity (LPIPS) [251]**
    *   **概念定义**：LPIPS 是一种感知图像相似度指标，旨在更好地与人类的视觉判断相关联。它通过计算两幅图像在预训练深度网络（如 `AlexNet`、`VGG` 或 `ResNet`）的特征空间中的加权 L2 距离来衡量相似度。较低的 LPIPS 值表示图像感知上更相似。
    *   **数学公式**：
        $$
        \mathrm { L P I P S } ( x , y ) = \sum _ { l } \frac { 1 } { H _ { l } W _ { l } } \sum _ { h , w } \left\| w _ { l } \odot \left( \hat { f } _ { h , w , x } ^ { l } - \hat { f } _ { h , w , y } ^ { l } \right) \right\| _ { 2 } ^ { 2 }
        $$
    *   **符号解释**：
        *   $x$：输入图像。
        *   $y$：参考图像。
        *   $l$：深度网络的层索引。
        *   $\hat{f}_{h,w,x}^l$：在层 $l$、空间位置 `(h, w)` 处，对输入 $x$ 提取并单位归一化后的特征。
        *   $\hat{f}_{h,w,y}^l$：在层 $l$、空间位置 `(h, w)` 处，对输入 $y$ 提取并单位归一化后的特征。
        *   $H_l, W_l$：层 $l$ 的特征图的高度和宽度。
        *   $w_l$：对层 $l$ 的特征图进行通道加权的权重。
        *   $\odot$：元素乘法。
        *   $\| \cdot \|_2^2$：L2 范数的平方。

*   **VBench [252]**
    *   **概念定义**：VBench 是一个针对视频生成模型的综合基准套件，它通过评估 16 个维度来全面衡量模型的性能。这些维度分为两类：<strong>视频质量 (Video Quality)</strong>（如主体一致性、运动平滑度）和<strong>视频条件一致性 (Video-Condition Consistency)</strong>（如物体类别、人类动作）。它提供精心策划的提示套件和大规模人类偏好标注，以确保强大的感知对齐，从而实现对模型能力和局限性的细粒度评估。
    *   **数学公式**：`VBench` 本身不是一个单一的数学公式，而是一套评估指标的集合，其中每个子指标可能都有其特定的计算方法。由于其复杂性，这里不提供单一的统一公式。
    *   **符号解释**：`VBench` 的每个子维度（例如 `subject consistency`、`motion smoothness`）都有其自己的计算逻辑，通常涉及特征提取、相似度计算或人类评估得分。

### 5.2.2. 状态级理解 (State-level Understanding)
这类指标超越了像素保真度，评估模型是否捕捉到物体、布局和语义，并能预测它们的演化，强调对结构理解而非外观的评估。

*   **mean Intersection over Union (mIoU)**
    *   **概念定义**：mIoU 是评估语义分割任务的常用指标。它计算每个类别的<strong>交并比 (Intersection over Union, IoU)</strong>，然后对所有类别取平均。IoU 量化了预测与<strong>真实标注数据 (Ground Truth)</strong> 之间的重叠程度，并惩罚分割错误。mIoU 值越高，表示语义场景理解越精确。
    *   **数学公式**：对于类别 $c$，
        $$
        \mathrm { I o U } = \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F P } + \mathrm { F N } }
        $$
        数据集级别的分数为：
        $$
        \mathrm { m I o U } = { \frac { 1 } { | C | } } \sum _ { c \in C } \mathrm { I o U } _ { c }
        $$
    *   **符号解释**：
        *   $\mathrm{TP}$：<strong>真阳性 (True Positives)</strong>，正确预测为正的像素数。
        *   $\mathrm{FP}$：<strong>假阳性 (False Positives)</strong>，错误预测为正的像素数。
        *   $\mathrm{FN}$：<strong>假阴性 (False Negatives)</strong>，错误预测为负的像素数。
        *   $C$：数据集中所有类别的集合。
        *   $|C|$：类别的总数。

*   **mean Average Precision (mAP)**
    *   **概念定义**：mAP 用于评估目标检测和实例分割任务。它通过在不同的<strong>交并比 (IoU)</strong> 阈值下计算每个类别的<strong>平均精度 (Average Precision, AP)</strong>，然后对类别和阈值取平均。mAP 值越高，表示实例识别更准确、定位更精确、置信度估计更校准。
    *   **数学公式**：精度和召回率定义为：
        $$
        \mathrm { P r e c i s i o n } = \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F P } } , \quad \mathrm { R e c a l l } = \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F N } }
        $$
        设 $P_{c, \tau}(r)$ 为通过单调插值获得的类别 $c$ 在阈值 $\tau$ 下的精度-召回曲线。类别 $c$ 在阈值 $\tau$ 下的 AP 为：
        $$
        \mathrm { A P } _ { c , \tau } = \int _ { 0 } ^ { 1 } P _ { c , \tau } ( r ) \mathrm { d } r
        $$
        mAP 对所有类别 $C$ 和所有阈值 $T$ 的 AP 取平均：
        $$
        \mathrm { m A P } = { \frac { 1 } { | C | } } \sum _ { c \in C } \left( { \frac { 1 } { | T | } } \sum _ { \tau \in T } { \mathrm { A P } } _ { c , \tau } \right)
        $$
    *   **符号解释**：
        *   $\mathrm{TP}$：<strong>真阳性 (True Positives)</strong>，正确检测到的实例。
        *   $\mathrm{FP}$：<strong>假阳性 (False Positives)</strong>，错误检测到的实例。
        *   $\mathrm{FN}$：<strong>假阴性 (False Negatives)</strong>，未检测到的真实实例。
        *   $\mathrm{Precision}$：精度。
        *   $\mathrm{Recall}$：召回率。
        *   $P_{c, \tau}(r)$：类别 $c$ 在 IoU 阈值 $\tau$ 下的精度-召回曲线。
        *   $\mathrm{AP}_{c, \tau}$：类别 $c$ 在 IoU 阈值 $\tau$ 下的平均精度。
        *   $C$：所有类别的集合。
        *   $|C|$：类别的总数。
        *   $T$：所有 IoU 阈值的集合（例如，`COCO` 数据集通常使用 `0.5:0.05:0.95`）。

*   <strong>Displacement Error (位移误差)</strong>
    *   **概念定义**：位移误差指标通过测量关键点、物体中心和轨迹<strong>航点 (waypoints)</strong> 的空间准确性来评估<strong>状态级理解 (state-level understanding)</strong>。它通常计算预测轨迹与<strong>真实标注数据 (Ground Truth)</strong> 轨迹之间的欧几里得距离。较低的值表示定位越准确。
    *   **数学公式**：
        *   **Average Displacement Error (ADE)**：平均位移误差，计算预测轨迹上所有点与真实轨迹对应点的平均欧几里得距离。
            $$
            \mathrm { ADE } = \frac { 1 } { N_T } \sum _ { t = 1 } ^ { N_T } \| \mathbf { p } _ { t } - \mathbf { \hat { p } } _ { t } \| _ { 2 }
            $$
        *   **Final Displacement Error (FDE)**：最终位移误差，测量预测轨迹终点与真实轨迹终点之间的欧几里得距离。
            $$
            \mathrm { FDE } = \| \mathbf { p } _ { N_T } - \mathbf { \hat { p } } _ { N_T } \| _ { 2 }
            $$
    *   **符号解释**：
        *   $N_T$：轨迹中的时间步总数。
        *   $\mathbf{p}_t$：真实轨迹在时间步 $t$ 的位置向量。
        *   $\mathbf{\hat{p}}_t$：预测轨迹在时间步 $t$ 的位置向量。
        *   $\| \cdot \|_2$：L2 范数（欧几里得距离）。

*   **Chamfer Distance (CD) [253]**
    *   **概念定义**：CD 量化了两个点集（例如，预测的点云和<strong>真实标注数据 (Ground Truth)</strong> 点云）之间的几何相似性。它计算一个点集中每个点到另一个点集中最近点的距离平方和，然后将两个方向的距离相加。CD 能够捕捉表面、占用率、`BEV` 和 3D 结构，并且是可微分的，因此可用作训练损失和评估指标。
    *   **数学公式**：
        $$
        \mathrm { C D } ( S _ { 1 } , S _ { 2 } ) = \sum _ { x \in S _ { 1 } } \operatorname* { m i n } _ { y \in S _ { 2 } } \left\| x - y \right\| _ { 2 } ^ { 2 } + \sum _ { y \in S _ { 2 } } \operatorname* { m i n } _ { x \in S _ { 1 } } \left\| x - y \right\| _ { 2 } ^ { 2 }
        $$
    *   **符号解释**：
        *   $S_1$：第一个点集（例如，预测点云）。
        *   $S_2$：第二个点集（例如，真实点云）。
        *   $x$：点集 $S_1$ 中的一个点。
        *   $y$：点集 $S_2$ 中的一个点。
        *   $\operatorname{min}_{y \in S_2} \|x - y\|_2^2$：点 $x$ 到点集 $S_2$ 中最近点的欧几里得距离平方。
        *   $\operatorname{min}_{x \in S_1} \|x - y\|_2^2$：点 $y$ 到点集 $S_1$ 中最近点的欧几里得距离平方。
        *   $\| \cdot \|_2^2$：L2 范数的平方。

### 5.2.3. 任务性能 (Task Performance)
这类指标评估世界模型在支持有效决策方面的最终价值，关注在体感设置下实现目标的安全性与效率。

*   <strong>Success Rate (SR) (成功率)</strong>
    *   **概念定义**：SR 量化了满足预定义成功条件的评估<strong>情节 (episodes)</strong> 的比例。在导航和操纵任务中，成功条件通常是二元的（例如，到达目标或正确放置物体）。在自动驾驶中，要求更严格，例如在没有碰撞或重大违规的情况下完成路线。最终的 SR 报告为所有测试<strong>情节 (episodes)</strong> 的二元结果的平均值。
    *   **数学公式**：
        $$
        \mathrm { SR } = \frac { \text { Number of successful episodes } } { \text { Total number of evaluation episodes } } \times 100\%
        $$
    *   **符号解释**：
        *   `Number of successful episodes`：成功完成任务的情节数量。
        *   `Total number of evaluation episodes`：总共进行的评估情节数量。

*   <strong>Sample Efficiency (SE) (样本效率)</strong>
    *   **概念定义**：SE 量化了达到目标性能所需的样本（例如，环境互动步骤、数据点）数量。它可以通过固定预算基准（例如 `Atari-100k`）、数据-性能曲线或在机器人学中通过实现给定成功率所需的演示数量来评估。更高的样本效率意味着模型能够以更少的经验学习得更好。
    *   **数学公式**：`Sample Efficiency` 通常没有一个单一的数学公式，而是通过报告达到某个性能水平所需的交互步数、训练数据量或墙钟时间等来衡量。例如，对于 `Atari-100k`，SE 直接就是 100k 步内达到的性能。
    *   **符号解释**：根据具体的评估方法而定。

*   <strong>Reward (奖励)</strong>
    *   **概念定义**：在<strong>强化学习 (Reinforcement Learning, RL)</strong> 中，奖励是智能体在时间步 $t$ 从环境接收到的信号 $r_t$。<strong>回报 (Return)</strong> 是未来奖励的累积总和，通常经过折扣。奖励用于指导智能体的学习，使其最大化长期累积奖励。
    *   **数学公式**：<strong>折扣回报 (Discounted Return)</strong> $G_t$ 定义为：
        $$
        G _ { t } = \sum _ { k = 0 } ^ { \infty } \gamma ^ { k } r _ { t + k + 1 }
        $$
    *   **符号解释**：
        *   $G_t$：在时间步 $t$ 之后的累积折扣回报。
        *   $r_{t+k+1}$：在时间步 $t+k+1$ 获得的奖励。
        *   $\gamma$：<strong>折扣因子 (discount factor)</strong>，通常介于 0 和 1 之间，用于权衡即时奖励和未来奖励的重要性。
        *   $\infty$：表示无限长的时间步。
        *   在实践中，有时会使用<strong>平均回报 (average return)</strong> 或<strong>归一化回报 (normalized return)</strong> 进行跨任务比较。

*   <strong>Collision (碰撞)</strong>
    *   **概念定义**：安全性通过基于碰撞的指标进行评估。主要衡量标准是<strong>碰撞率 (collision rate)</strong>，即至少发生一次碰撞的评估<strong>情节 (episodes)</strong> 的比例，这在室内导航中很常见。在自动驾驶中，使用经过<strong>曝光归一化 (exposure-normalized)</strong> 的变体，例如每公里碰撞次数或每小时碰撞次数。较低的碰撞率表示更高的安全性。
    *   **数学公式**：
        $$
        \mathrm { Collision~Rate } = \frac { \text { Number of episodes with at least one collision } } { \text { Total number of evaluation episodes } } \times 100\%
        $$
    *   **符号解释**：
        *   `Number of episodes with at least one collision`：至少发生一次碰撞的情节数量。
        *   `Total number of evaluation episodes`：总共进行的评估情节数量。

# 6. 实验结果与分析

本综述通过任务目标和标准化基准对现有世界模型的性能进行比较，并提供了简明的表格来突出每个方法的优缺点。

## 6.1. 像素生成 (Pixel Generation)

### 6.1.1. `nuScenes` 上的生成 (Generation on nuScenes)
自动驾驶视频生成被视为一项世界模型任务，它在固定长度的剪辑中合成 plausible 的场景动态。典型的协议会生成短序列，并使用 `FID` (外观保真度) 和 `FVD` (时间一致性) 来评估质量。

以下是原文 `Table IV` 在 `nuScenes` 上的视频生成性能比较：

<table><thead><tr><th>Method</th><th>Pub.</th><th>Resolution</th><th>FID↓</th><th>FVD↓</th></tr></thead><tbody><tr><td>MagicDrive3D [84]</td><td>arXiv'24</td><td>224 × 400</td><td>20.7</td><td>164.7</td></tr><tr><td>Delphi [86]</td><td>arXiv'24</td><td>512 × 512</td><td>15.1</td><td>113.5</td></tr><tr><td>Drive-WM [88]</td><td>CVPR'24</td><td>192 × 384</td><td>15.8</td><td>122.7</td></tr><tr><td>GenAD [90]</td><td>CVPR'24</td><td>256 × 448</td><td>15.4</td><td>184.0</td></tr><tr><td>DriveDreamer [91]</td><td>ECCV'24</td><td>128 × 192</td><td>52.6</td><td>452.0</td></tr><tr><td>Vista [96]</td><td>NeurIPS'24</td><td>576 × 1024</td><td>6.9</td><td>89.4</td></tr><tr><td>DrivePhysica [214]</td><td>arXiv'24</td><td>256 × 448</td><td>4.0</td><td>38.1</td></tr><tr><td>DrivingWorld [133]</td><td>arXiv'24</td><td>512 × 1024</td><td>7.4</td><td>90.9</td></tr><tr><td>DriveDreamer-2 [97]</td><td>AAAI'25</td><td>256 × 448</td><td>11.2</td><td>55.7</td></tr><tr><td>UniFuture [206]</td><td>arXiv'25</td><td>320 × 576</td><td>11.8</td><td>99.9</td></tr><tr><td>MiLA [189]</td><td>arXiv'25</td><td>360 × 640</td><td>4.1</td><td>14.9</td></tr><tr><td>GeoDrive [170]</td><td>arXiv'25</td><td>480 × 720</td><td>4.1</td><td>61.6</td></tr><tr><td>LongDWM [188]</td><td>arXiv'25</td><td>480 × 720</td><td>12.3</td><td>102.9</td></tr><tr><td>MaskGWM [104]</td><td>CVPR'25</td><td>288 × 512</td><td>8.9</td><td>65.4</td></tr><tr><td>GEM [102]</td><td>CVPR'25</td><td>576 × 1024</td><td>10.5</td><td>158.5</td></tr><tr><td>Epona [148]</td><td>ICCV'25</td><td>512 × 1024</td><td>7.5</td><td>82.8</td></tr><tr><td>STAGE [198]</td><td>IROS'25</td><td>512 × 768</td><td>11.0</td><td>242.8</td></tr><tr><td>DriVerse [109]</td><td>ACMMM'25</td><td>480 × 832</td><td>18.2</td><td>95.2</td></tr></tbody></table>

**分析**：
*   <strong>最佳视觉保真度 (`FID`↓)</strong>：`DrivePhysica` [214] 取得了最低的 `FID` (4.0)，表明其生成的视频在外观上最接近真实。紧随其后的是 `MiLA` [189] 和 `GeoDrive` [170]，均为 4.1。
*   <strong>最强时间一致性 (`FVD`↓)</strong>：`MiLA` [189] 实现了最低的 `FVD` (14.9)，这表明它在生成视频的动态连贯性和时间稳定性方面表现最佳。`DrivePhysica` [214] (38.1) 和 `DriveDreamer-2` [97] (55.7) 也表现出色。
*   **趋势**：较早的模型如 `DriveDreamer` [91] (FID 52.6, FVD 452.0) 性能相对较差，而近期的模型则在分辨率和性能上都有显著提升。这反映了自动驾驶视频生成领域在过去几年中的快速发展，特别是<strong>扩散模型 (Diffusion Models)</strong> 和更先进的<strong>世界模型 (World Models)</strong> 架构的应用。
*   **权衡**：`DrivePhysica` 在 `FID` 上表现最好，但在 `FVD` 上略逊于 `MiLA`，这表明在追求极致的外观逼真度和时间动态一致性之间可能存在一定的权衡。

## 6.2. 场景理解 (Scene Understanding)

### 6.2.1. `Occ3D-nuScenes` 上的 4D 占用率预测 (4D Occupancy Forecasting on Occ3D-nuScenes)
4D <strong>占用率预测 (Occupancy Forecasting)</strong> 被视为一项代表性的世界模型任务。给定过去 2 秒的 3D 占用率，模型预测接下来 3 秒的场景动态。评估遵循 `Occ3D-nuScenes` 协议，并报告 `mIoU` 和分时段 `IoU`。

以下是原文 `Table V` 在 `Occ3D-nuScenes` 基准上 4D <strong>占用率预测 (Occupancy Forecasting)</strong> 的性能比较：

<table><thead><tr><td rowspan="2">Method</td><td rowspan="2">Input</td><td rowspan="2">Aux. Sup</td><td rowspan="2">Ego traj.</td><td colspan="5">mIoU (%) ↑</td><td colspan="5">IoU (%) ↑</td></tr><tr><td>Recon.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>Recon.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr></thead><tbody><tr><td>Copy &amp; Paste2</td><td>Occ</td><td>None</td><td>Pred.</td><td>66.38</td><td>14.91</td><td>10.54</td><td>8.52</td><td>11.33</td><td>62.29</td><td>24.47</td><td>19.77</td><td>17.31</td><td>20.52</td></tr><tr><td>OccWorld-O [93]</td><td>Occ</td><td>None</td><td>Pred.</td><td>66.38</td><td>25.78</td><td>15.14</td><td>10.51</td><td>17.14</td><td>62.29</td><td>34.63</td><td>25.07</td><td>20.18</td><td>26.63</td></tr><tr><td>OccLLaMA-O [18]</td><td>Occ</td><td>None</td><td>Pred.</td><td>75.20</td><td>25.05</td><td>19.49</td><td>15.26</td><td>19.93</td><td>63.76</td><td>34.56</td><td>28.53</td><td>24.41</td><td>29.17</td></tr><tr><td>RenderWorld-O [156]</td><td>Occ</td><td>None</td><td>Pred.</td><td>-</td><td>28.69</td><td>18.89</td><td>14.83</td><td>20.80</td><td>-</td><td>37.74</td><td>28.41</td><td>24.08</td><td>30.08</td></tr><tr><td>DTT-O [98]</td><td>Occ</td><td>None</td><td>Pred.</td><td>85.50</td><td>37.69</td><td>29.77</td><td>25.10</td><td>30.85</td><td>92.07</td><td>76.60</td><td>74.44</td><td>72.71</td><td>74.58</td></tr><tr><td>DFIT-OccWorld-O [174]</td><td>Occ</td><td>None</td><td>Pred.</td><td>-</td><td>31.68</td><td>21.29</td><td>15.18</td><td>22.71</td><td>-</td><td>40.28</td><td>31.24</td><td>25.29</td><td>32.27</td></tr><tr><td>COME-O [213]</td><td>Occ</td><td>None</td><td>Pred.</td><td>-</td><td>30.57</td><td>19.91</td><td>13.38</td><td>21.29</td><td></td><td>36.96</td><td>28.26</td><td>21.86</td><td>29.03</td></tr><tr><td>DOME-O [94]</td><td>Occ</td><td>None</td><td>GT</td><td>83.08</td><td>35.11</td><td>25.89</td><td>20.29</td><td>27.10</td><td>77.25</td><td>43.99</td><td>35.36</td><td>29.74</td><td>36.36</td></tr><tr><td>COME-O [213]</td><td>Occ</td><td>None</td><td>GT</td><td>-</td><td>42.75</td><td>32.97</td><td>26.98</td><td>34.23</td><td>-</td><td>50.57</td><td>43.47</td><td>38.36</td><td>44.13</td></tr><tr><td>OccWorld-T [93]</td><td>Camera</td><td>Semantic LiDAR</td><td>Pred.</td><td>7.21</td><td>4.68</td><td>3.36</td><td>2.63</td><td>3.56</td><td>10.66</td><td>9.32</td><td>8.23</td><td>7.47</td><td>8.34</td></tr><tr><td>OccWorld-S [93]</td><td>Camera</td><td>None</td><td>Pred.</td><td>0.27</td><td>0.28</td><td>0.26</td><td>0.24</td><td>0.26</td><td>4.32</td><td>5.05</td><td>5.01</td><td>4.95</td><td>5.00</td></tr><tr><td>RenderWorld-S [156]</td><td>Camera</td><td>None</td><td>Pred.</td><td>-</td><td>2.83</td><td>2.55</td><td>2.37</td><td>2.58</td><td>-</td><td>14.61</td><td>13.61</td><td>12.98</td><td>13.73</td></tr><tr><td>COME-S [213]</td><td>Camera</td><td>None</td><td>Pred.</td><td>-</td><td>25.57</td><td>18.35</td><td>13.41</td><td>19.11</td><td>-</td><td>45.36</td><td>37.06</td><td>30.46</td><td>37.63</td></tr><tr><td>OccWorld-D [93]</td><td>Camera</td><td>Occ</td><td>Pred.</td><td>18.63</td><td>11.55</td><td>8.10</td><td>6.22</td><td>8.62</td><td>22.88</td><td>18.90</td><td>16.26</td><td>14.43</td><td>16.53</td></tr><tr><td>OccWorld-F [93]</td><td>Camera</td><td>Occ</td><td>Pred.</td><td>20.09</td><td>8.03</td><td>6.91</td><td>3.54</td><td>6.16</td><td>35.61</td><td>23.62</td><td>18.13</td><td>15.22</td><td>18.99</td></tr><tr><td>OccLLaMA-F [18]</td><td>Camera</td><td>Occ</td><td>Pred.</td><td>37.38</td><td>10.34</td><td>8.66</td><td>6.98</td><td>8.66</td><td>38.92</td><td>25.81</td><td>23.19</td><td>19.97</td><td>22.99</td></tr><tr><td>DFIT-OccWorld-F [174]</td><td>Camera</td><td>Occ</td><td>Pred.</td><td>-</td><td>13.38</td><td>10.16</td><td>7.96</td><td>10.50</td><td>-</td><td>19.18</td><td>16.85</td><td>15.02</td><td>17.02</td></tr><tr><td>DTT-F [98]</td><td>Camera</td><td>Occ</td><td>Pred.</td><td>43.52</td><td>24.87</td><td>18.30</td><td>15.63</td><td>19.60</td><td>54.31</td><td>38.98</td><td>37.45</td><td>31.89</td><td>36.11</td></tr><tr><td>DOME-F [94]</td><td>Camera</td><td>Occ</td><td>GT</td><td>75.00</td><td>24.12</td><td>17.41</td><td>13.24</td><td>18.25</td><td>74.31</td><td>35.18</td><td>27.90</td><td>23.44</td><td>28.84</td></tr><tr><td>COME-F [213]</td><td>Camera</td><td>Occ</td><td>GT</td><td>-</td><td>26.56</td><td>21.73</td><td>18.49</td><td>22.26</td><td>-</td><td>48.08</td><td>43.84</td><td>40.28</td><td>44.07</td></tr></tbody></table>

**分析**：
*   **输入模态的影响**：
    *   使用 `Occ` (占用率) 作为输入的模型 (`OccWorld-O`、`OccLLaMA-O` 等) 通常优于仅使用 `Camera` (摄像头) 作为输入的模型（例如 `OccWorld-S`）。这表明直接的 3D 结构信息对<strong>占用率预测 (occupancy forecasting)</strong> 至关重要。
    *   在 `Occ` 输入组中，`COME-O` [213] (GT `Ego traj.`) 在 `Avg. mIoU` (34.23) 和 `Avg. IoU` (44.13) 上表现最佳，其次是 `DTT-O` [98] (Avg. mIoU 30.85, Avg. IoU 74.58)。`DTT-O` 在 `IoU` 指标上表现异常突出，尤其是在 `Recon.` 阶段，这可能归因于其<strong>三平面表示 (triplane representation)</strong> 和多尺度 Transformer 架构对增量变化的捕捉能力。
    *   在 `Camera` 输入组中，`COME-S` [213] (Pred. `Ego traj.`, None `Aux. Sup.`) 在 `Avg. mIoU` (19.11) 和 `Avg. IoU` (37.63) 上表现突出，显示了仅凭摄像头信息进行复杂场景理解的强大能力。而使用 `Occ` 作为辅助监督的 `Camera` 模型 (例如 `COME-F`) 进一步提升了性能，`COME-F` (GT `Ego traj.`) 达到了 `Avg. mIoU` 22.26 和 `Avg. IoU` 44.07。
*   <strong>辅助监督 (`Aux. Sup.`) 的作用</strong>：添加辅助监督（如 `Semantic LiDAR` 或 `Occ`）通常能提升性能，尤其是在摄像头输入的情况下。
*   <strong>自车轨迹 (`Ego traj.`) 的影响</strong>：使用<strong>真实标注数据 (Ground Truth, GT)</strong> 的自车轨迹而非预测轨迹，能够显著提高预测性能。例如，`COME-O` 从预测轨迹的 `Avg. mIoU` 21.29 提升到 `GT` 轨迹的 34.23。这凸显了精确自车定位对未来预测的重要性。
*   **长时序衰减**：所有方法在预测时间越长时，`mIoU` 和 `IoU` 都会显著下降（从 1 秒到 3 秒），这反映了<strong>误差累积 (error accumulation)</strong> 和预测不确定性在长时序预测中的固有挑战。

## 6.3. 控制任务 (Control Tasks)

### 6.3.1. `DMC` 上的评估 (Evaluation on DMC)
大多数研究在像素级设置下，使用 $64 \times 64 \times 3$ 的观测数据，在 `DMC` 上探索世界模型学习控制相关动态的能力。主要指标是<strong>情节回报 (Episode Return)</strong>，定义为 1000 步内的累积奖励。

以下是原文 `Table VI` 在 `DMC` 基准上的性能比较：

<table><thead><tr><td rowspan="2">Method</td><td rowspan="2">Step</td><td colspan="4">Episode Return↑</td><td rowspan="2">Avg. / Total</td></tr><tr><td>Reacher Easy</td><td>Cheetah Run</td><td>Finger Spin</td><td>Walker Walk</td></tr></thead><tbody><tr><td>PlaNet [38]</td><td>5M</td><td>469</td><td>496</td><td>495</td><td>945</td><td>333/20</td></tr><tr><td>Dreamer [10]</td><td>5M</td><td>935</td><td>895</td><td>499</td><td>962</td><td>823/20</td></tr><tr><td>Dreaming [110]</td><td>500k</td><td>905</td><td>566</td><td>762</td><td>469</td><td>610/12</td></tr><tr><td>TransDreamer [28]</td><td>2M</td><td>-</td><td>865</td><td>-</td><td>933</td><td>893/4</td></tr><tr><td>DreamerPro [111]</td><td>1M</td><td>873</td><td>897</td><td>811</td><td>-</td><td>857/6</td></tr><tr><td>MWM [41]</td><td>1M</td><td>-</td><td>670</td><td></td><td>-</td><td>690/7</td></tr><tr><td>HRSSM [25]</td><td>500k</td><td>910</td><td>-</td><td>960</td><td>-</td><td>938/3</td></tr><tr><td>DisWM [112]</td><td>1M</td><td>960</td><td>820</td><td>-</td><td>920</td><td>879/5</td></tr></tbody></table>

`Note`: DMC 上的性能比较。带下划线的数据表示从各自的奖励曲线中近似得分。平均分数 (`Avg.`) 作为粗略指标提供，因为任务难度不同。

**分析**：
*   <strong>样本效率 (`Step`)</strong>：
    *   `PlaNet` 和 `Dreamer` 在 5M 步下训练，`Dreamer` 实现了更高的平均回报 (823/20)。
    *   `Dreaming` [110] 和 `HRSSM` [25] 在 500k 步下达到了不错的回报，显著提高了<strong>样本效率 (sample efficiency)</strong>。
    *   `TransDreamer` [28] 在 2M 步下取得了较高的平均回报 (893/4)，而 `DreamerPro` [111] 和 `DisWM` [112] 在 1M 步下也表现出良好性能。
*   **任务表现**：
    *   `DisWM` [112] 在 `Reacher Easy` 任务上表现最佳 (960)。
    *   `Dreamer` 和 `TransDreamer` 在 `Cheetah Run` (895, 865) 和 `Walker Walk` (962, 933) 任务上表现出色。
    *   `HRSSM` [25] 在 `Finger Spin` 任务上达到了高分 (960)。
*   **趋势**：尽管评估协议和任务子集不一致，但结果表明近期模型在更少的训练步骤中实现了更强的性能，提升了<strong>数据效率 (data efficiency)</strong>。然而，构建一个能够跨任务、模态和数据集广泛迁移的模型仍然是一个开放挑战。

### 6.3.2. `RLBench` 上的评估 (Evaluation on RLBench)
`RLBench` 使用 7-自由度模拟 `Franka` 机械臂评估操纵任务，旨在测试世界模型捕捉任务相关动态和支持条件动作生成的能力。主要指标是<strong>成功率 (Success Rate)</strong>。

以下是原文 `Table VII` 在 `RLBench` 操纵任务上的性能比较：

<table><thead><tr><td rowspan="2" colspan="2">Criteria</td><td colspan="5">Methods</td></tr><tr><td>VidMan [55]</td><td>ManiGaussian [53]</td><td>ManiGaussian++ [80]</td><td>DreMa [60]</td><td>TesserAct [78]</td></tr></thead><tbody><tr><td rowspan="7">Specifications</td><td>Episode</td><td>125</td><td>25</td><td>25</td><td>250</td><td>100</td></tr><tr><td>Pixel</td><td>224</td><td>128</td><td>256</td><td>128</td><td>512</td></tr><tr><td>Depth</td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr><tr><td>Language</td><td>✓</td><td>✓</td><td>✓</td><td></td><td>✓</td></tr><tr><td>Proprioception</td><td>✓</td><td>✓</td><td>✓</td><td></td><td></td></tr><tr><td>Characteristic</td><td>IDM</td><td>GS</td><td>Bimanual</td><td>GS</td><td>DiT</td></tr><tr><td>Stack Blocks</td><td>48</td><td>12</td><td>-</td><td>12</td><td>-</td></tr><tr><td>Success Rate (%)</td><td>Close Jar</td><td>88</td><td>28</td><td>-</td><td>51</td><td>44</td></tr><tr><td></td><td>Open Drawer</td><td>94</td><td>76</td><td>-</td><td>-</td><td>80</td></tr><tr><td></td><td>Sweep to Dustpan</td><td>93</td><td>64</td><td>92</td><td>-</td><td>56</td></tr><tr><td></td><td>Slide Block</td><td>98</td><td>24</td><td>-</td><td>62</td><td>-</td></tr><tr><td></td><td>Avg. / Total</td><td>67/18</td><td>45/10</td><td>35/10</td><td>25/9</td><td>63/10</td></tr></tbody></table>

`Avg.`: 平均分数仅作为粗略指标报告，因为任务难度各异。

**分析**：
*   **任务覆盖范围**：`VidMan` [55] 在所有列出的任务中（除了 `Stack Blocks` 和 `Close Jar`）都展示了最高的成功率，并在最广泛的任务集（18 个任务）上获得了最高的平均成功率 (67%)，这表明<strong>逆动力学模型 (IDM)</strong> 作为一种架构方向具有潜力。
*   **输入模态和骨干网络**：近期方法越来越多地利用多模态输入（包括深度信息、语言和<strong>本体感受 (proprioception)</strong>）并采用更强的<strong>主干网络 (backbones)</strong>，如 `3DGS` 和 `DiT`。
*   **具体任务表现**：
    *   `VidMan` 在 `Close Jar` (88%)、`Open Drawer` (94%)、`Sweep to Dustpan` (93%) 和 `Slide Block` (98%) 等任务上表现非常出色。
    *   $ManiGaussian++$ [80] (35/10) 专注于<strong>双手操纵 (bimanual manipulation)</strong>。
    *   `TesserAct` [78] (63/10) 在 `Open Drawer` 任务上表现优异 (80%)。
*   **异构性**：尽管存在实施差异（如<strong>情节预算 (episode budgets)</strong>、分辨率和模态），但整体趋势是模型在操纵任务中表现出越来越好的性能。

## 6.4. 规划 (Planning)

### 6.4.1. `nuScenes` 验证集上的开环规划 (Open-loop Planning on nuScenes Validation Split)
开环规划被视为 `nuScenes` 验证集上的一项世界模型任务，其中模型根据有限的历史信息预测自车运动。方法观察过去 2 秒的轨迹，并预测未来 3 秒的 2D `BEV` <strong>航点 (waypoints)</strong>。评估报告在多个时间范围内的 `L2` 误差和<strong>碰撞率 (collision rate)</strong>。

以下是原文 `Table VIII` 在 `nuScenes` 验证集上开环规划的性能比较：

<table><thead><tr><td rowspan="2">Method</td><td rowspan="2">Input</td><td rowspan="2">Aux. Sup.2</td><td colspan="4">L2 (m) ↓</td><td colspan="4">Collision Rate (%) ↓</td></tr><tr><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr></thead><tbody><tr><td>UniAD [254]</td><td>Camera</td><td>Map &amp; Box &amp; Motion &amp; Tracklets &amp; Occ</td><td>0.48</td><td>0.96</td><td>1.65</td><td>1.03</td><td>0.05</td><td>0.17</td><td>0.71</td><td>0.31</td></tr><tr><td>UniAD+DriveWorld [87]</td><td>Camera</td><td>Map &amp; Box &amp; Motion &amp; Tracklets &amp; Occ</td><td>0.34</td><td>0.67</td><td>1.07</td><td>0.69</td><td>0.04</td><td>0.12</td><td>0.41</td><td>0.19</td></tr><tr><td>GenAD [92]</td><td>Camera</td><td>Map &amp; Box &amp; Motion</td><td>0.36</td><td>0.83</td><td>1.55</td><td>0.91</td><td>0.06</td><td>0.23</td><td>1.00</td><td>0.43</td></tr><tr><td>FSDrive [101]</td><td>Camera</td><td>Map &amp; Box &amp; QA</td><td>0.40</td><td>0.89</td><td>1.60</td><td>0.96</td><td>0.07</td><td>0.12</td><td>1.02</td><td>0.40</td></tr><tr><td>OccWorld-T [93]</td><td>Camera</td><td>Semantic LiDAR</td><td>0.54</td><td>1.36</td><td>2.66</td><td>1.52</td><td>0.12</td><td>0.40</td><td>1.59</td><td>0.70</td></tr><tr><td>Doe-1 [134]</td><td>Camera</td><td>QA</td><td>0.50</td><td>1.18</td><td>2.11</td><td>1.26</td><td>0.04</td><td>0.37</td><td>1.19</td><td>0.53</td></tr><tr><td>SSR [160]</td><td>Camera</td><td>None</td><td>0.24</td><td>0.65</td><td>1.36</td><td>0.75</td><td>0.00</td><td>0.10</td><td>0.36</td><td>0.15</td></tr><tr><td>OccWorld-S [93]</td><td>Camera</td><td>None</td><td>0.67</td><td>1.69</td><td>3.13</td><td>1.83</td><td>0.19</td><td>1.28</td><td>4.59</td><td>2.02</td></tr><tr><td>Epona [148]</td><td>Camera</td><td>None</td><td>0.61</td><td>1.17</td><td>1.98</td><td>1.25</td><td>0.01</td><td>0.22</td><td>0.85</td><td>0.36</td></tr><tr><td>RenderWorld [156]</td><td>Camera</td><td>None</td><td>0.48</td><td>1.30</td><td>2.67</td><td>1.48</td><td>0.14</td><td>0.55</td><td>2.23</td><td>0.97</td></tr><tr><td>Drive-OccWorld [157]</td><td>Camera</td><td>None</td><td>0.32</td><td>0.75</td><td>1.49</td><td>0.85</td><td>0.05</td><td>0.17</td><td>0.64</td><td>0.29</td></tr><tr><td>OccWorld-D [93]</td><td>Camera</td><td>Occ</td><td>0.52</td><td>1.27</td><td>2.41</td><td>1.40</td><td>0.12</td><td>0.40</td><td>2.08</td><td>0.87</td></tr><tr><td>OccWorld-F [93]</td><td>Camera</td><td>Occ</td><td>0.45</td><td>1.33</td><td>2.25</td><td>1.34</td><td>0.08</td><td>0.42</td><td>1.71</td><td>0.73</td></tr><tr><td>OccLLaMA-F [18]</td><td>Camera</td><td>Occ</td><td>0.38</td><td>1.07</td><td>2.15</td><td>1.20</td><td>0.06</td><td>0.39</td><td>1.65</td><td>0.70</td></tr><tr><td>DTT-F [98]</td><td>Camera</td><td>Occ</td><td>0.35</td><td>1.01</td><td>1.89</td><td>1.08</td><td>0.08</td><td>0.33</td><td>0.91</td><td>0.44</td></tr><tr><td>DFIT-OccWorld-V [174]</td><td>Camera</td><td>Occ</td><td>0.42</td><td>1.14</td><td>2.19</td><td>1.25</td><td>0.09</td><td>0.19</td><td>1.37</td><td>0.55</td></tr><tr><td>NeMo [161]</td><td>Camera</td><td>Occ</td><td>0.39</td><td>0.74</td><td>1.39</td><td>0.84</td><td>0.00</td><td>0.09</td><td>0.82</td><td>0.30</td></tr><tr><td>OccWorld-O [93]</td><td>Occ</td><td>None</td><td>0.43</td><td>1.08</td><td>1.99</td><td>1.17</td><td>0.07</td><td>0.38</td><td>1.35</td><td>0.60</td></tr><tr><td>OccLLaMA-O [18]</td><td>Occ</td><td>None</td><td>0.37</td><td>1.02</td><td>2.03</td><td>1.14</td><td>0.04</td><td>0.24</td><td>1.20</td><td>0.49</td></tr><tr><td>RenderWorld-O [156]</td><td>Occ</td><td>None</td><td>0.35</td><td>0.91</td><td>1.84</td><td>1.03</td><td>0.05</td><td>0.40</td><td>1.39</td><td>0.61</td></tr><tr><td>DTT-O [98]</td><td>Occ</td><td>None</td><td>0.32</td><td>0.91</td><td>1.76</td><td>1.00</td><td>0.08</td><td>0.32</td><td>0.51</td><td>0.30</td></tr><tr><td>DFIT-OccWorld-O [174]</td><td>Occ</td><td>None</td><td>0.38</td><td>0.96</td><td>1.73</td><td>1.02</td><td>0.07</td><td>0.39</td><td>0.90</td><td>0.45</td></tr></tbody></table>

`Aux. Sup.2`: Aux. Sup. 是辅助监督的缩写。

**分析**：
*   <strong>L2 误差 (`L2`↓)</strong>：
    *   $UniAD+DriveWorld$ [87] 在 `Map & Box & Motion & Tracklets & Occ` 等大量辅助监督下，实现了最低的 `Avg. L2` (0.69)，尤其在 3 秒预测上仅为 1.07。这表明结合多模态输入和丰富的辅助信息可以显著提高轨迹预测的精确性。
    *   `SSR` [160] (Camera, None Aux. Sup.) 在无辅助监督的情况下，取得了令人印象深刻的 `Avg. L2` (0.75)，并且在 1 秒预测时 `L2` 仅为 0.24，表现非常出色。
    *   `NeMo` [161] (Camera, Occ Aux. Sup.) 在 `Avg. L2` (0.84) 也表现良好，并且 1 秒预测的 `L2` 仅为 0.39。
*   <strong>碰撞率 (`Collision Rate`↓)</strong>：
    *   `SSR` [160] 实现了最低的 `Avg. Collision Rate` (0.15%)，在 1 秒预测时甚至为 0.00%，这表明其在安全性方面表现出众。
    *   $UniAD+DriveWorld$ [87] (0.19%) 和 `NeMo` [161] (0.30%) 也表现出较低的碰撞率。
*   **权衡**：在 `L2` 误差和<strong>碰撞率 (collision rate)</strong> 之间存在明显的权衡。例如，$UniAD+DriveWorld$ 拥有最低的 `L2` 误差，但 `SSR` 实现了最低的碰撞率且 `L2` 误差也很有竞争力。
*   **摄像头与占用率**：基于摄像头的模型现在能够超越使用<strong>特权占用率 (privileged occupancy)</strong> 信息的模型。例如，`SSR` 仅使用摄像头输入就达到了非常好的 `L2` 和碰撞率。这反映了<strong>端到端规划 (E2E planning)</strong> 的日益成熟。
*   **长时序预测**：所有模型的 `L2` 误差和<strong>碰撞率 (collision rate)</strong> 都会随着预测时间（从 1 秒到 3 秒）的增加而增加，这再次证实了<strong>长时序预测 (long-horizon prediction)</strong> 的固有挑战。

# 7. 总结与思考

## 7.1. 结论总结
本综述为体感人工智能领域的<strong>世界模型 (World Models)</strong> 建立了一个统一的框架，通过三轴分类法（功能性、时间建模、空间表示）对现有研究进行了系统化的组织和分析。论文形式化了世界模型的问题设置和学习目标，阐述了其数学基础，并对机器人学、自动驾驶和通用视频设置中的数据资源和评估指标进行了系统化。通过对最先进模型进行定量比较，总结了世界模型在像素生成、场景理解和控制任务中的最新进展。

核心发现包括：
*   **分类的重要性**：明确的分类法有助于理解领域的多样性和复杂性。
*   **多模态和结构化表示的兴起**：为了更好地捕捉动态和支持几何感知规划，模型正从简单的<strong>全局潜向量 (Global Latent Vector)</strong> 转向更复杂的<strong>词元特征序列 (Token Feature Sequence)</strong>、<strong>空间潜网格 (Spatial Latent Grid)</strong> 和<strong>分解渲染表示 (Decomposed Rendering Representation)</strong>。
*   **性能提升**：在各种基准上，世界模型的性能在持续提升，尤其是在<strong>样本效率 (sample efficiency)</strong> 和生成视频的逼真度方面。
*   **挑战依然存在**：尽管取得了进展，但在<strong>长时序一致性 (long-horizon consistency)</strong>、<strong>误差累积 (error accumulation)</strong>、<strong>计算效率 (computational efficiency)</strong> 和<strong>物理一致性 (physical consistency)</strong> 方面仍面临重大挑战。

## 7.2. 局限性与未来工作

论文作者指出了以下局限性并提出了未来的研究方向：

### 7.2.1. 数据与评估 (Data & Evaluation)
*   **挑战**：
    *   **统一数据集的稀缺性**：体感人工智能涵盖多个领域，但缺乏统一的大规模数据集，导致模型泛化能力受限。
    *   **评估指标的不足**：现有指标如 `FID` 和 `FVD` 过分强调<strong>像素保真度 (pixel fidelity)</strong>，而忽视了<strong>物理一致性 (physical consistency)</strong>、<strong>动态 (dynamics)</strong> 和<strong>因果关系 (causality)</strong>。新基准（如 `EWM-Bench` [255]）虽然引入了新度量，但仍是任务特定的，缺乏跨领域标准。
*   **未来方向**：
    *   **统一的多模态跨领域数据集**：应优先构建能够支持<strong>可迁移预训练 (transferable pretraining)</strong> 的统一数据集。
    *   **物理基础的评估框架**：需要开发超越感知真实感的评估框架，以评估物理一致性、<strong>因果推理 (causal reasoning)</strong> 和<strong>长时序动态 (long-horizon dynamics)</strong>。

### 7.2.2. 计算效率 (Computational Efficiency)
*   **挑战**：
    *   **实时控制需求**：`Transformer` 和<strong>扩散网络 (Diffusion Networks)</strong> 虽然性能强大，但其高昂的推理成本与机器人系统对实时控制的要求相冲突。
    *   **传统方法的局限**：`RNN` 和<strong>全局潜向量 (Global Latent Vector)</strong> 仍然被广泛使用，因为它们计算效率更高，但在捕捉长期依赖方面存在局限。
*   **未来方向**：
    *   **优化模型架构**：通过<strong>量化 (quantization)</strong>、<strong>剪枝 (pruning)</strong> 和<strong>稀疏计算 (sparse computation)</strong> 等技术优化模型，减少推理延迟。
    *   **探索新型时间方法**：研究<strong>状态空间模型 (State Space Models, SSMs)</strong>（如 `Mamba`），以在保持实时效率的同时增强<strong>长时序推理 (long-range reasoning)</strong> 能力。

### 7.2.3. 建模策略 (Modeling Strategy)
*   **挑战**：
    *   **长时序动态与空间表示**：世界模型仍在努力解决<strong>长时序时间动态 (long-horizon temporal dynamics)</strong> 和高效<strong>空间表示 (spatial representations)</strong> 的问题。
    *   **序列模拟与全局预测的权衡**：<strong>自回归设计 (autoregressive designs)</strong> 紧凑且<strong>样本效率高 (sample-efficient)</strong>，但会<strong>累积误差 (accumulate errors)</strong>；<strong>全局预测 (global prediction)</strong> 提高了多步<strong>一致性 (coherence)</strong>，但计算成本高且<strong>闭环交互 (closed-loop interactivity)</strong> 较弱。
    *   **空间效率瓶颈**：<strong>潜向量 (latent vectors)</strong>、<strong>词元序列 (token sequences)</strong> 和<strong>空间网格 (spatial grids)</strong> 在效率和表达能力之间存在权衡；<strong>分解渲染方法 (decomposed rendering approaches)</strong>（如 `NeRF` 和 `3DGS`）提供高保真度，但在动态场景中扩展性差。
    *   **闭环控制的困难**：在世界模型中实现<strong>闭环控制 (closed-loop control)</strong> 仍然具有挑战性。
*   **未来方向**：
    *   **集成自回归和全局预测的优势**：通过<strong>显式记忆 (explicit memory)</strong> 或<strong>分层规划 (hierarchical planning)</strong> 增强<strong>长时序预测 (long-horizon prediction)</strong> 的稳定性。
    *   **任务分解**：受<strong>思维链 (Chain-of-Thought, CoT)</strong> 启发，通过设置中间目标来提高<strong>时间一致性 (temporal consistency)</strong>。
    *   **统一架构**：未来的框架应优先优化<strong>长时序推理 (long-range reasoning)</strong>、<strong>计算效率 (computational efficiency)</strong> 和<strong>生成保真度 (generative fidelity)</strong>，并将时间建模和空间建模无缝集成到统一的架构中，以在效率、保真度和交互性之间取得有效平衡。

## 7.3. 个人启发与批判

这篇综述提供了一个极其全面的视角，揭示了<strong>世界模型 (World Models)</strong> 在<strong>体感人工智能 (Embodied AI)</strong> 领域的核心作用和发展路径。

**个人启发：**
1.  **统一框架的价值**：作者提出的三轴分类法极具洞察力。它不仅为理解现有方法提供了一个清晰的结构，也为未来研究指明了潜在的设计空间。作为研究者，我们可以利用这个框架来定位新工作的创新点，并更好地与现有工作进行比较。
2.  **长时序一致性是核心瓶颈**：无论是像素生成、占用率预测还是轨迹规划，`长时序预测`中的`误差累积`都是一个反复出现的、根本性的挑战。这意味着未来的突破可能不仅在于模型容量的提升，更在于如何设计能够内在抵抗或有效纠正这种误差的机制，例如通过引入更强的物理先验、更智能的记忆机制或更鲁棒的纠错循环。
3.  **多模态与多表示融合**：从`全局潜向量`到`分解渲染表示`的演进，反映了世界模型对环境理解的日益精细化和多维度化。未来的通用世界模型必然需要能够无缝融合视觉、语言、触觉甚至物理定律等多种模态和表示，以构建真正像人类一样全面的世界认知。
4.  **Sim-to-Real 的重要性**：文中多次提到`S2R`（模拟到现实）的挑战和相关工作。世界模型的最终目标是在真实世界中发挥作用，因此如何弥合模拟环境和真实环境之间的差距，将是衡量模型实用性的关键。

**批判与可以改进的地方：**
1.  **物理一致性评估的缺失**：虽然综述强调了现有评估指标（如`FID`/`FVD`）忽视`物理一致性`的问题，但其在`定量比较`部分仍然主要依赖这些像素级的指标。这反映了领域内缺乏成熟的、可广泛应用的`物理一致性`评估标准。未来的研究（包括综述本身）需要投入更多精力来定义和应用这些指标，而不仅仅是提出这个挑战。例如，可以更详细地讨论哪些现有工作已经尝试量化物理违反（如物体穿透、不合理的运动速度等），并如何在标准化的方式下进行比较。
2.  **计算效率的细化**：`计算效率`被列为主要挑战之一，但除了提及`RNN`/`Transformer`的权衡和`SSM`的潜力外，可以更深入地探讨具体的技术细节。例如，当今大规模`世界模型`的训练和推理成本到底有多高？不同`空间表示`（如`3DGS`与`NeRF`）在实时性方面的具体瓶颈是什么？如何量化和比较这些效率？
3.  **因果推理的探讨**：`因果推理`在世界模型中至关重要，因为它涉及理解“为什么”环境会以某种方式变化，而不仅仅是“如何”变化。综述在`数学形式化`中提及`因果分解`，并在`未来方向`中强调了`因果推理`，但可以更深入地探讨现有模型是如何尝试学习和利用`因果关系`的，以及目前离实现真正意义上的`因果世界模型`还有多远。
4.  **统一数据集的建设性方案**：虽然指出了统一数据集的稀缺性，但可以更具体地提出构建这类数据集的路线图或建议。例如，是否可以通过众包、联邦学习、或者利用现有大型但分散的数据集进行整合和标准化来解决？这需要社区共同努力。
5.  **LLM 与世界模型的深层融合**：虽然提到了`LLM`与世界模型的结合，但对二者如何进行更深层次的`融合`（超越`CoT`和`指令引导`）的探讨可以更进一步。例如，`LLM`是否能提供更高层次的抽象世界知识，来弥补`世界模型`在常识推理和长时序规划上的不足？如何将`语言`作为一种`通用接口`来`查询`和`操作`世界模型？

    总而言之，这篇综述为体感人工智能的`世界模型`领域提供了一个坚实的基石，其提出的框架和对挑战的洞察将对未来的研究产生深远影响。同时，领域内仍有许多基础和应用层面的问题等待解决。