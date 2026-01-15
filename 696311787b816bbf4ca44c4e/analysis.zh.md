# 1. 論文基本信息

## 1.1. 标题
Reward Forcing: Efficient Streaming Video Generation with Rewarded Distribution Matching Distillation
(中文：奖励强制：通过带奖励的分布匹配蒸馏实现高效的流式视频生成)

论文的核心主题是解决高效流式视频生成中的<strong>运动停滞 (motion stagnation)</strong> 问题。它提出了一种名为 `Reward Forcing` 的新框架，通过两个关键技术——`EMA-Sink` 和 `Re-DMD`——来提升生成视频的动态性和长期一致性，同时保持高效率。

## 1.2. 作者
- **作者列表:** Yunhong Lu, Yanhong Zeng, Haobo Li, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jiapeng Zhu, Hengyuan Cao, Zhipeng Zhang, Xing Zhu, Yujun Shen, Min Zhang.
- **隶属机构:** 论文作者来自多个顶尖学术和工业研究机构，包括浙江大学 (Zhejiang University)、蚂蚁集团 (Ant Group)、上海交通大学 (SJTU) 等。这种产学研结合的背景通常意味着研究既有学术深度，也关注实际应用价值。

## 1.3. 发表期刊/会议
论文的元数据显示其发表于 UTC 时间 2025-12-04，并提供了 arXiv 预印本链接。这表明在当前时间点（2026-01-11），它是一篇已公开发布在 arXiv 上的预印本 (preprint)，通常这类高质量的工作会投递到计算机视觉或机器学习领域的顶级会议，如 CVPR, ICCV, NeurIPS 等。

## 1.4. 发表年份
2025年 (根据 arXiv 提交信息)

## 1.5. 摘要
高效的流式视频生成对于模拟交互式和动态的世界至关重要。现有方法通过滑动窗口注意力机制来蒸馏少步视频扩散模型，并使用初始帧作为“注意力沉点”词元 (sink tokens) 来维持注意力性能并减少误差累积。然而，这导致视频帧过度依赖这些静态词元，从而产生初始帧被复制和运动动态减弱的问题。为解决此问题，我们引入了 `Reward Forcing`，一个包含两个关键设计的新颖框架。首先，我们提出了 `EMA-Sink`，它维护一组由初始帧初始化且大小固定的词元，并通过指数移动平均 (EMA) 方式，在旧词元滑出窗口时不断融合它们的信息来进行更新。`EMA-Sink` 在不增加额外计算成本的情况下，既能捕捉长期上下文，又能融入近期动态，从而在保持长时一致性的同时防止了初始帧复制。其次，为了更好地从教师模型中蒸馏运动动态，我们提出了一种新颖的<strong>带奖励的分布匹配蒸馏 (Rewarded Distribution Matching Distillation, Re-DMD)</strong>。传统的分布匹配方法平等对待每个训练样本，限制了模型优先学习动态内容的能力。`Re-DMD` 则通过优先处理由一个视觉语言模型评定为具有更强动态性的样本，来使模型的输出分布偏向于高奖励区域。`Re-DMD` 在保持数据保真度的同时，显著增强了运动质量。我们通过定量和定性实验表明，`Reward Forcing` 在标准基准上达到了最先进的性能，并能在单张 H100 GPU 上以 23.1 FPS 的速度实现高质量的流式视频生成。

## 1.6. 原文链接
- **原文链接:** https://arxiv.org/abs/2512.04678
- **PDF 链接:** https://arxiv.org/pdf/2512.04678v2.pdf
- **发布状态:** 预印本 (Preprint)

# 2. 整体概括

## 2.1. 研究背景与动机
- **核心问题:** 如何在保持<strong>高效率（实时流式生成）</strong>和**高质量**的前提下，生成**长时程、内容连贯且富含动态变化**的视频。
- **重要性:** 实时、高质量的视频生成是构建交互式虚拟世界、高级模拟器以及各种实时内容创作工具的关键技术。用户需要能够即时生成并与之互动的动态场景。
- <strong>现有挑战与空白 (Gap):</strong>
    1.  **效率与质量的矛盾:** 传统的视频扩散模型虽然质量高，但通常需要同时处理所有帧（双向注意力），计算成本高昂，不适合流式生成。
    2.  **误差累积:** 为了提高效率，研究转向自回归 (autoregressive) 模型，一帧一帧地生成。这类模型采用<strong>滑动窗口注意力 (sliding window attention)</strong> 来限制计算量，但这会导致误差累积，视频质量随时间推移而下降。
    3.  <strong>运动停滞 (Motion Stagnation):</strong> 为缓解误差累积，`LongLive` 等工作引入了<strong>注意力沉点 (attention sinks)</strong> 机制，即在滑动窗口中永久保留初始帧的词元。这虽然稳定了生成过程，却带来了新的严重问题：模型对静态的初始帧产生了**过度依赖**。其结果是，后续生成的画面倾向于复制第一帧的内容（“闪回”现象），导致视频缺乏运动和变化，场景显得僵硬。
- **切入点与创新思路:** 本文精准地抓住了“运动停滞”这一核心痛点。作者认为，问题的根源在于**静态的注意力沉点**和**无差别的蒸馏目标**。因此，他们提出了一个双管齐下的解决方案 `Reward Forcing`：
    1.  **动态化上下文:** 用一个动态更新的 `EMA-Sink` 代替静态的初始帧沉点，使其既能记住长期的历史信息，又能不断融入最新的动态变化。
    2.  **动态化学习目标:** 改进传统的分布匹配蒸馏 (DMD) 过程，引入一个奖励模型来评估生成样本的“动态性”，并在训练中赋予动态性强的样本更高的权重。这使得模型在学习（蒸馏）过程中，被“强制”去关注和生成更具动感的视频。

## 2.2. 核心贡献/主要发现
- **核心贡献:**
    1.  **提出 `EMA-Sink` 机制:** 这是一种新颖的、用于长视频生成的状态压缩方法。它通过指数移动平均 (EMA) 动态更新一个固定大小的“沉点”词元，优雅地解决了静态沉点导致的运动停滞问题，且几乎不增加计算开销。
    2.  <strong>提出 `Re-DMD` (Rewarded Distribution Matching Distillation) 框架:</strong> 这是一种创新的蒸馏方法，它将强化学习的思想与分布匹配相结合。通过一个外部奖励模型（视觉语言模型）来评估生成内容的运动质量，并利用该奖励分数来加权蒸馏损失，从而引导学生模型优先学习生成高动态性的内容。
- **主要发现:**
    1.  `Reward Forcing` 框架能够显著提升流式生成视频的**运动质量**和**长期一致性**，有效解决了先前方法中普遍存在的画面僵硬和内容重复问题。
    2.  该方法在短视频和长视频的标准基准测试 (VBench) 中均取得了<strong>最先进的 (state-of-the-art)</strong> 性能。
    3.  该方法实现了极高的**生成效率**，在单张 H100 GPU 上达到了 23.1 FPS 的实时生成速度，这对于交互式应用至关重要。

# 3. 预备知识与相关工作

## 3.1. 基础概念
- <strong>视频扩散模型 (Video Diffusion Models):</strong> 这是一类生成模型，其工作原理分为两个过程。**前向过程**：对一个真实的视频数据逐步、多次地添加高斯噪声，直到它变成完全的随机噪声。**反向过程**：训练一个神经网络（通常是基于 `Transformer` 或 `U-Net` 架构），让它学习如何从随机噪声开始，一步步地“去噪”，最终恢复出一个清晰、真实的视频。通过控制这个去噪过程（例如，通过文本提示），模型就能生成新的视频内容。
- <strong>自回归生成 (Autoregressive Generation):</strong> 这是一种序列生成范式。在生成视频时，不是一次性生成所有帧，而是一次只生成一帧（或一小段视频块），并且每一帧的生成都依赖于它之前已经生成的所有帧。其概率模型可以表示为 `p(\mathbf{x}^{1:N}) = \prod_{i=1}^N p(\mathbf{x}^i | \mathbf{x}^{<i})`。这种方式天然适合流式生成，因为未来的帧不需要在当前步骤被访问。
- <strong>滑动窗口注意力 (Sliding Window Attention):</strong> 在标准的 `Transformer` 模型中，每个词元 (token) 需要与序列中的所有其他词元计算注意力，这导致计算复杂度随序列长度二次方增长 ($O(L^2)$)，对于长视频来说是不可接受的。滑动窗口注意力是一种优化，它限制每个词元只与最近的 $w$ 个词元（即窗口内的词元）计算注意力，将计算复杂度降低到 $O(L \cdot w^2)$，使其与序列总长度 $L$ 呈线性关系，从而实现了高效的长序列处理。
- <strong>注意力沉点 (Attention Sinks):</strong> 这是 `Transformer` 模型在处理长序列时发现的一个有趣现象。研究表明，即使在使用滑动窗口时，只要**始终保留序列最开始的几个词元**在注意力计算的缓存中，模型的性能就能得到极大的维持，有效防止因窗口滑动导致的信息丢失和性能崩溃。在视频生成中，通常将视频的第一帧作为这个“沉点”。
- <strong>分布匹配蒸馏 (Distribution Matching Distillation, DMD):</strong> 这是一种模型压缩技术，旨在将一个强大但缓慢的“教师”模型（如需要数千步去噪的完整扩散模型）的知识迁移到一个轻量且快速的“学生”模型（如只需1-4步就能生成的模型）。其核心思想不是让学生模型在单个样本上模仿教师的输出，而是调整学生模型的参数，使其**生成的所有样本的整体分布**与教师模型生成的样本分布尽可能接近。
- <strong>强化学习 (Reinforcement Learning, RL):</strong> 这是一种机器学习范式，其中一个<strong>智能体 (agent)</strong>（这里指视频生成模型）通过与环境的交互来学习。智能体会采取行动（生成视频），然后从环境中获得一个<strong>奖励 (reward)</strong>（如视频质量得分）。智能体的目标是学习一个<strong>策略 (policy)</strong>（生成视频的方式），以最大化其获得的累积奖励。在生成模型中，RL常被用来根据人类偏好或特定指标（如美学、动态性）来微调模型。

## 3.2. 前人工作
- **自回归长视频生成:**
    - `CausVid`:  pioneering work that applied <strong>分布匹配蒸馏 (DMD)</strong> to reformulate a bidirectional diffusion model into a fast, causal (autoregressive) one for video generation.
    - `Self-Forcing`: Built upon `CausVid`. Its key innovation was to bridge the **train-test gap**. During training, it simulates the inference process by feeding the model its own previously generated outputs, rather than always using ground-truth frames. This makes the model more robust to its own errors during actual generation.
    - `LongLive`: Extended `Self-Forcing` for longer videos. It explicitly incorporated the <strong>注意力沉点 (attention sinks)</strong> mechanism by keeping the initial frame in the Key-Value (KV) cache, which stabilized long-horizon generation but led to the motion stagnation problem addressed by this paper.

- **强化学习用于视频生成:**
    - <strong>直接偏好优化 (Direct Preference Optimization, DPO):</strong> 一种无需显式训练奖励模型的强化学习方法。它直接从成对的偏好数据（例如，视频A比视频B更好）中学习，调整模型以生成更符合偏好的内容。
    - <strong>策略优化 (Policy Optimization):</strong> 如 `Self-Forcing++` 使用 `Flow-GRPO` 算法，将生成过程视为一个策略，并使用 RL 目标直接优化这个策略，以改善视频的长期时间平滑度等指标。

## 3.3. 技术演进
视频生成技术的发展脉络可以概括为：
1.  **高质量但慢速的生成:** 早期基于 `GAN` 和后来的扩散模型，专注于在短视频（几秒钟）上实现高保真度，但计算成本极高，无法实时。
2.  **追求效率的自回归方法:** 为了实现更长的视频和更快的生成，研究者转向自回归框架，并结合模型蒸馏技术 (如 `DMD`) 将多步模型压缩为少步模型，代表作是 `CausVid` 和 `Self-Forcing`。
3.  **解决长时一致性:** 当视频变长时，误差累积和风格漂移成为主要问题。`LongLive` 等工作通过引入注意力沉点来解决这个问题，但牺牲了动态性。
4.  **平衡一致性与动态性:** 本文 `Reward Forcing` 处在技术演进的最新阶段。它认识到静态沉点是导致运动停滞的罪魁祸首，并提出 `EMA-Sink` 和 `Re-DMD` 来同时解决长时一致性和动态性不足的问题，是对此前工作的一个精准而有效的改进。

## 3.4. 差异化分析
- **与 `LongLive` 的对比:** `LongLive` 使用**静态**的初始帧作为注意力沉点。`Reward Forcing` 的 `EMA-Sink` 则是**动态**的，它是一个不断吸收历史信息并演变的上下文摘要。这使得模型既能保持长期记忆，又不会被“锁死”在初始状态，从而根本上解决了运动停滞问题。
- **与 `CausVid` / `Self-Forcing` 的对比:** 这些工作使用<strong>标准的分布匹配蒸馏 (DMD)</strong>，该方法对所有生成样本一视同仁，无法区分“动态的”好样本和“静态的”坏样本。`Reward Forcing` 的 `Re-DMD` 则引入了**奖励信号**，它不再是盲目地匹配整个分布，而是有偏好地、加权地去匹配教师分布中**高动态性**的区域，使得学习目标更加明确和高效。

# 4. 方法论

## 4.1. 方法原理
`Reward Forcing` 的核心思想是通过**改造上下文机制**和**重塑学习目标**来解决流式视频生成中的运动停滞问题。
- <strong>改造上下文机制 (`EMA-Sink`):</strong> 传统方法要么因丢弃历史信息而导致不连贯，要么因固守初始帧而导致僵化。`EMA-Sink` 提出了一种折中方案：创建一个浓缩了全部历史信息的“摘要”词元。这个摘要是动态变化的，随着新帧的生成，它会不断地用一种平滑衰减的方式（指数移动平均）融入刚刚滑出窗口的帧的信息。这样，模型在生成新帧时，既能看到近期的高清细节（窗口内的帧），也能看到一个不断演变的、代表全局历史的低频信号（`EMA-Sink` 词元）。
- <strong>重塑学习目标 (`Re-DMD`):</strong> 传统蒸馏方法的目标是“让学生像老师”，但并未指明要学习老师的哪些方面。`Re-DMD` 引入了一个“评论家”（奖励模型），它会给学生生成的视频打分（重点是动态性得分）。然后，在蒸馏过程中，对于得分高的视频，模型会用更大的力度去匹配教师模型的分布；对于得分低的视频，则减小匹配力度。这相当于在训练中引入了一个明确的导向，即“不仅要像老师，更要学习老师生成高动态性视频的能力”。

## 4.2. 核心方法详解 (逐层深入)

### 4.2.1. EMA-Sink：长视频的状态封装机制
为了解决标准滑动窗口注意力机制丢弃历史信息的问题，`EMA-Sink` 被设计为一个高效的全局状态压缩与更新模块。

**1. 问题背景：滑动窗口的信息瓶颈**

在自回归生成第 $i$ 帧时，模型通常只关注一个大小为 $w$ 的局部窗口，即之前的 `w-1` 帧 $\bar{\mathcal{X}}^{i, w} = [\mathbf{x}^{i-w+1:i-1}]$。当窗口滑动到第 $i+1$ 帧时，最旧的帧 $\mathbf{x}^{i-w+1}$ 及其对应的键 (Key) 和值 (Value) 向量将从注意力缓存中被丢弃。这种信息的永久丢失是导致长期不一致性的根源。

**2. `EMA-Sink` 机制**

`EMA-Sink` 的核心思想是**不丢弃，而是压缩**。它维护一个固定大小的压缩全局状态（沉点词元） $\mathcal{S}_*^i$，该状态通过指数移动平均 (EMA) 来持续融合被“驱逐”出窗口的帧的信息。

*   **状态更新:** 当帧 $\mathbf{x}^{i-w}$ 从滑动窗口中移出时，其对应的键值对 $(K^{i-w}, V^{i-w})$ 会被用来更新沉点状态。更新过程如下面的公式所示：
    $$
    \begin{array} { r } { { \pmb { S } } _ { K } ^ { i } = \alpha \cdot { \pmb { S } } _ { K } ^ { i - 1 } + ( 1 - \alpha ) \cdot { \pmb { K } } ^ { i - w } , } \\ { { \pmb { S } } _ { V } ^ { i } = \alpha \cdot { \pmb { S } } _ { V } ^ { i - 1 } + ( 1 - \alpha ) \cdot { \pmb { V } } ^ { i - w } . } \end{array}
    $$
    **符号解释:**
    *   $\pmb{S}_K^i, \pmb{S}_V^i$: 分别表示在生成第 $i$ 帧时，更新后的沉点键 (Key) 状态和沉点值 (Value) 状态。
    *   $\pmb{S}_K^{i-1}, \pmb{S}_V^{i-1}$: 上一时刻的沉点状态。
    *   $\pmb{K}^{i-w}, \pmb{V}^{i-w}$: 刚被移出窗口的第 `i-w` 帧的键和值向量。
    *   $\alpha \in (0, 1)$: 动量衰减因子。一个接近 1 的 $\alpha$ 值意味着沉点状态变化缓慢，更多地保留长期历史信息；一个较小的 $\alpha$ 值则意味着沉点状态对近期被移出的帧更敏感。这提供了一个平滑的时间压缩机制，既保留了遥远历史的“记忆痕迹”，又让近期的信息占据主导。

*   **注意力计算:** 在计算注意力时，`EMA-Sink` 状态被拼接到当前局部窗口的键和值向量的前面，形成一个扩展的上下文。
    $$
    K _ { \mathrm { global } } ^ { i } = \left[ S _ { K } ^ { i } ; K ^ { i - w + 1 : i } \right]
    $$
    $$
    V _ { \mathrm { \ g l o bal } } ^ { i } = \left[ S _ { V } ^ { i } ; V ^ { i - w + 1 : i } \right]
    $$
    **符号解释:**
    *   $K_{\mathrm{global}}^i, V_{\mathrm{global}}^i$: 供当前帧查询 (Query) 使用的全局键和值。
    *   `[;]`: 表示拼接操作。
    *   $K^{i-w+1:i}, V^{i-w+1:i}$: 当前滑动窗口内所有帧的键和值向量。

        通过这种方式，每个新生成的帧都能够同时关注到**细粒度的局部上下文**（窗口内的帧）和**粗粒度的全局历史**（`EMA-Sink` 词元），从而在不增加计算和内存成本的情况下，打破了固定窗口大小的信息瓶颈。

下图（原文 Figure 2）直观地对比了 `EMA-Sink` 与传统方法的区别。

![Figure 2. Comparison of EMA Sink with Existing Methods. Long video generation models typically extrapolate beyond their training sequence length during inference. (a) Window Attention caches only recent tokens for efficient inference but suffers performance degradation. (b) Sliding Window with attention sinks retains initial tokens for stable attention computation and recent tokens for extrapolation. However, discarding intermediate frames causes over-reliance on the first frame, leading to "frame copying" and stiff transitions. (c) EMA Sink preserves full history through exponential moving average (EMA) updates of all historical frames, maintaining stable and consistent performance in long video extrapolation without increasing computational cost.](images/2.jpg)
*该图像是示意图，比较了三种长视频生成方法的注意力机制。左侧是窗口注意力，性能下降；中间是带有注意力沉 sink 的滑动窗口，导致对第一帧过度依赖；右侧是我们提出的 EMA-Sink，利用 EMA 更新实现历史帧的保留，保持稳定的性能而不增加计算成本。*

### 4.2.2. Re-DMD：带奖励的分布匹配蒸馏
`Re-DMD` 旨在解决标准 `DMD` 无法区分样本质量、平等对待所有生成结果的局限性。

**1. 问题背景：标准 `DMD` 的局限性**

标准 `DMD` 的目标是最小化学生模型生成分布 $p_{\mathrm{fake}}$ 与教师模型生成分布 $p_{\mathrm{real}}$ 之间的 KL 散度。其目标函数可写作：
$$
\mathcal { T } _ { \mathrm { D M D } } = \mathbb { E } _ { p ( c ) p _ { \mathrm { f a k e } } ( \pmb { x } _ { 0 } | c ) } \Big [ \log \frac { p _ { \mathrm { f a k e } } ( \pmb { x } _ { 0 } | \pmb { c } ) } { p _ { \mathrm { r e a l } } ( \pmb { x } _ { 0 } | \pmb { c } ) } \Big ]
$$
这个目标函数对分布中的所有区域（所有类型的样本）都一视同仁，导致模型可能学会生成视觉质量高但毫无动态的视频，因为这类样本在分布上离教师模型很近，不会产生大的损失。

**2. `Re-DMD` 机制**

`Re-DMD` 将强化学习中的<strong>奖励加权回归 (Reward-Weighted Regression)</strong> 思想融入 `DMD` 框架。其推导过程基于期望最大化 (EM) 算法。

*   **RL 目标:** 首先，引入一个通用的 RL 微调目标，它旨在最大化奖励，同时通过 KL 散度惩罚与原始模型 $q$ 的偏离，以保持保真度。
    $$
    \mathcal { T } _ { \mathrm { R L } } ( p , q ) = \mathbb { E } \Big [ \frac { r ( { \pmb x } _ { 0 } , { \pmb c } ) } { \beta } - \log \frac { p ( { \pmb x } _ { 0 } | { \pmb c } ) } { q ( { \pmb x } _ { 0 } | { \pmb c } ) } \Big ]
    $$
    **符号解释:**
    *   $p$: 我们希望学习到的优化后的新分布。
    *   $q$: 原始模型（学生模型当前）的分布。
    *   $r(\mathbf{x}_0, \mathbf{c})$: 奖励函数，用于评估生成样本 $\mathbf{x}_0$ 在给定条件 $\mathbf{c}$ 下的质量（如运动性）。
    *   $\beta$: 温度系数，控制奖励项的影响强度。$\beta$ 越小，奖励的影响越大。

*   <strong>E-步 (Expectation-step):</strong> 求解上述优化问题，可以得到最优分布 $p(\mathbf{x}_0|\mathbf{c})$ 的形式：
    $$
    p ( \pmb { x } _ { 0 } | \pmb { c } ) = \frac { 1 } { Z ( \pmb { c } ) } q ( \pmb { x } _ { 0 } | \pmb { c } ) \exp \Big ( \frac { r ( \pmb { x } _ { 0 } , \pmb { c } ) } { \beta } \Big )
    $$
    **符号解释:**
    *   $Z(\mathbf{c})$: 归一化常数，确保概率和为1。
        这个公式的直观含义是：最优的分布 $p$ 是在原始分布 $q$ 的基础上，通过一个与奖励值 $r$ 相关的指数项进行**重新加权**得到的。奖励越高的样本，其在 $p$ 分布中的概率密度就越大。

*   <strong>M-步 (Maximization-step):</strong> 我们将这个理论上的最优分布 $p$ 投射到我们参数化的学生模型上。具体做法是将 `DMD` 目标中的样本期望，替换为在 E-步得到的加权分布下的期望。这产生了 `Re-DMD` 的目标函数：
    $$
    \mathcal { T } _ { \mathrm { Re - D M D } } = \mathbb { E } _ { p ( c ) p _ { \mathrm { f a k e } } ^ { \prime } ( \pmb { x } _ { 0 } | c ) } \left[ \frac { \exp \left( r \left( \pmb { x } _ { 0 } , \pmb { c } \right) / \beta \right) } { Z ( \pmb { c } ) } \log \frac { p _ { \mathrm { f a k e } } \left( \pmb { x } _ { 0 } | \pmb { c } \right) } { p _ { \mathrm { r e a l } } \left( \pmb { x } _ { 0 } | \pmb { c } \right) } \right]
    $$
    这里的 $p'_{\mathrm{fake}}$ 是学生模型的原始输出分布。这个公式的核心是，原始的 `DMD` 损失 $\log \frac{p_{\mathrm{fake}}}{p_{\mathrm{real}}}$ 被一个权重 $\frac{\exp(r(\cdot)/\beta)}{Z(\mathbf{c})}$ 所调制。**高奖励样本会获得高权重，从而在梯度下降中发挥更大作用。**

*   **梯度计算:** 直接计算上述损失函数是困难的，但其梯度可以被有效地估计。通过对目标函数求导，并利用 score-matching 的思想，得到最终用于训练的梯度形式：
    $$
    \begin{array} { r l } & { \nabla _ { \theta } \mathcal { J } _ { \mathrm { Re - D M D } } \approx - \mathbb { E } _ { t } \Big ( \int \exp ( r ^ { c } ( \pmb { x } _ { t } ) / \beta ) \cdot \big ( s _ { \mathrm { r e al } } ( \Psi ( G _ { \theta } ( \epsilon ) , t ) , t ) } \\ & { \quad \quad \quad - s _ { \mathrm { f a k e } } \big ( \Psi ( G _ { \theta } ( \epsilon ) , t ) , t ) \big ) \frac { \mathrm { d } G _ { \theta } ( \epsilon ) } { \mathrm { d } \theta } \mathrm { d } \epsilon \Big ) . } \end{array}
    $$
    **符号解释:**
    *   $\nabla_\theta \mathcal{J}_{\mathrm{Re-DMD}}$: `Re-DMD` 损失关于模型参数 $\theta$ 的梯度。
    *   $\exp(r^c(\mathbf{x}_t)/\beta)$: **奖励权重**。这是与标准 `DMD` 相比的核心区别。它根据生成样本的奖励值来放大或缩小梯度。
    *   $s_{\mathrm{real}}, s_{\mathrm{fake}}$: 分别是教师模型（在真实数据上训练）和学生模型（在生成数据上训练）的<strong>分数函数 (score function)</strong>，在扩散模型中，它与去噪预测相关。
    *   $G_\theta(\epsilon)$: 学生生成器，输入噪声 $\epsilon$ 输出样本。
    *   $\frac{\mathrm{d}G_\theta(\epsilon)}{\mathrm{d}\theta}$: 生成器关于参数的雅可比矩阵。
    *   **重要实践:** 论文指出，在实践中，对带噪样本 $\mathbf{x}_t$ 的奖励 $r^c(\mathbf{x}_t)$ 是通过先将其去噪得到 $\mathbf{x}_0$，然后计算 $r^c(\mathbf{x}_0)$ 来估计的。这避免了对奖励模型求导的需要，大大简化了计算并稳定了训练。

        下图（原文 Figure 3）展示了 `Re-DMD` 的工作流程。

        ![该图像是示意图，展示了奖励强制（Reward Forcing）框架的结构，包括动态摩托车生成过程。图中展示了当前关键值缓存（Current KV cache）的更新过程，以及生成视频的关键元素如教师梯度和奖励函数。相关公式包括 EMA 更新，标记为 `EMA ext{ update}`。](images/3.jpg)
        *该图像是示意图，展示了奖励强制（Reward Forcing）框架的结构，包括动态摩托车生成过程。图中展示了当前关键值缓存（Current KV cache）的更新过程，以及生成视频的关键元素如教师梯度和奖励函数。相关公式包括 EMA 更新，标记为 `EMA ext{ update}`。*

# 5. 实验设置

## 5.1. 数据集
- **训练数据集:**
    - 论文使用了一个包含 16,000 个从基础模型（教师模型）采样的 ODE 解对进行初始训练。
    - 训练所用的文本提示 (prompts) 来自于 `VidProM` 数据集。该数据集经过了筛选，并由大型语言模型 (LLM) 进行了增强，以提高提示的质量和多样性。
- **评估数据集:**
    - **短视频评估:** 使用了 `VBench` 基准测试中的 946 个官方提示。这些提示经过了 `Qwen` 系列模型的改写，以增加测试的挑战性。
    - **长视频评估:** 使用了 `MovieGen` 数据集中的前 128 个提示，与 `CausVid` 等先前工作保持一致，生成时长为 60 秒的视频。
- **数据集选择理由:**
    - `VidProM` 是一个大规模、高质量的文本-视频提示数据集，适合用于训练。
    - `VBench` 和 `MovieGen` 是视频生成领域公认的权威评测基准，使用它们可以确保与现有最先进方法进行公平、全面的比较。

## 5.2. 评估指标
- **VBench / VBenchLong:** 这是一个综合性的视频生成评测套件，从多个维度评估视频质量。论文中提及的具体指标包括：
    - <strong>质量维度 (Quality):</strong>
        *   `Subject Consistency`: 主体一致性。
        *   `Background Consistency`: 背景一致性。
        *   `Motion Smoothness`: 运动平滑度。
        *   `Dynamic Degree`: 动态程度。
        *   `Aesthetic Quality`: 美学质量。
        *   `Imaging Quality`: 成像质量。
    - <strong>语义维度 (Semantic):</strong>
        *   `Object Class`: 物体类别。
        *   `Human Action`: 人类动作。
        *   `Color`: 颜色。
        *   ... 等等，评估视频内容与文本提示的匹配程度。

- <strong>Drift (漂移):</strong>
    - **概念定义:** 该指标用于量化长视频生成过程中的质量不稳定性。它计算视频中各个时间段的成像质量得分的标准差。一个低的漂移值意味着视频从头到尾保持了稳定的高质量，而高的漂移值则表示视频质量波动大或随时间下降。
    - **数学公式:**
      $$
        \operatorname { D r i f t } ( V _ { i } ) = { \sqrt { \frac { 1 } { M - 1 } \sum _ { j = 1 } ^ { M } ( s _ { i , j } - { \bar { s } } _ { i } ) } }
        $$
    - **符号解释:**
        *   $V_i$: 第 $i$ 个长视频。
        *   $M$: 视频被分割成的片段数量（实验中为30）。
        *   $s_{i,j}$: 视频 $V_i$ 的第 $j$ 个片段的成像质量分数。
        *   $\bar{s}_i$: 视频 $V_i$ 所有片段的平均成像质量分数。

- <strong>FPS (Frames Per Second, 每秒帧数):</strong>
    - **概念定义:** 衡量模型生成视频的速度。它表示在一秒钟内可以生成多少帧图像。这是评估模型是否能用于实时应用的关键指标。
    - **数学公式:**
      $$
        \text{FPS} = \frac{\text{Total Generated Frames}}{\text{Total Generation Time (seconds)}}
        $$
    - **符号解释:**
        *   `Total Generated Frames`: 生成的总帧数。
        *   `Total Generation Time`: 完成生成所花费的总时间。

- **Qwen3-VL Score:**
    - **概念定义:** 这是一种基于强大的视觉语言模型 (VLM) `Qwen3-VL` 的自动评估方法。研究者设计了一套详细的评分标准，让 VLM 像人类专家一样，从<strong>文本对齐度 (Text Alignment)</strong>、<strong>运动动态性 (Motion Dynamics)</strong> 和<strong>视觉质量 (Visual Quality)</strong> 三个维度为生成的视频打分（1-5分）。这种方法可以提供比传统指标更接近人类感知的评估结果。

## 5.3. 对比基线
论文将 `Reward Forcing` 与一系列具有代表性的开源视频生成模型进行了比较，涵盖了两大类：
- <strong>基于扩散 (Diffusion) 的模型:</strong>
    - `LTX-Video`
    - `Wan-2.1` (本文方法的教师模型)
- <strong>自回归 (Autoregressive) 模型:</strong>
    - `SkyReels-V2`
    - `MAGI-1`
    - `NOVA`
    - `Pyramid Flow`
    - `CausVid`
    - `Self Forcing`
    - `LongLive`
    - `Rolling Forcing`

      这些基线模型覆盖了从传统扩散模型到最新的高效自回归模型的演进路径，特别是 `CausVid`, `Self Forcing`, `LongLive` 是与本文方法技术路线最接近的直接竞争对手，因此将它们作为核心比较对象具有很强的说服力。

# 6. 实验结果与分析

## 6.1. 核心结果分析
实验结果有力地证明了 `Reward Forcing` 框架在提升视频动态性、保持长期一致性以及实现高效率方面的综合优势。

### 6.1.1. 短视频生成性能
以下是原文 Table 1 的结果，比较了模型在5秒短视频生成任务上的表现：

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Params</th>
<th rowspan="2">FPS↑</th>
<th colspan="3">VBench evaluation scores ↑</th>
</tr>
<tr>
<th>Total</th>
<th>Quality</th>
<th>Semantic</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="6"><strong>Diffusion</strong></td>
</tr>
<tr>
<td>LTX-Video [13]</td>
<td>1.9B</td>
<td>8.98</td>
<td>80.00</td>
<td>82.30</td>
<td>70.79</td>
</tr>
<tr>
<td>Wan-2.1 [72]</td>
<td>1.3B</td>
<td>0.78</td>
<td><u>84.26</u></td>
<td><u>85.30</u></td>
<td>80.09</td>
</tr>
<tr>
<td colspan="6"><strong>Autoregressive</strong></td>
</tr>
<tr>
<td>SkyReels-V2 [7]</td>
<td>1.3B</td>
<td>0.49</td>
<td>82.67</td>
<td>84.70</td>
<td>74.53</td>
</tr>
<tr>
<td>MAGI-1 [69]</td>
<td>4.5B</td>
<td>0.19</td>
<td>79.18</td>
<td>82.04</td>
<td>67.74</td>
</tr>
<tr>
<td>NOVA [13]</td>
<td>0.6B</td>
<td>0.88</td>
<td>80.12</td>
<td>80.39</td>
<td>79.05</td>
</tr>
<tr>
<td>Pyramid Flow [33]</td>
<td>2B</td>
<td>6.7</td>
<td>81.72</td>
<td>84.74</td>
<td>69.62</td>
</tr>
<tr>
<td>CausVid [89]</td>
<td>1.3B</td>
<td>17.0</td>
<td>82.88</td>
<td>83.93</td>
<td>78.69</td>
</tr>
<tr>
<td>Self Forcing [30]</td>
<td>1.3B</td>
<td>17.0</td>
<td>83.80</td>
<td>84.59</td>
<td>80.64</td>
</tr>
<tr>
<td>LongLive [82]</td>
<td>1.3B</td>
<td>20.7</td>
<td>83.22</td>
<td>83.68</td>
<td><u>81.37</u></td>
</tr>
<tr>
<td>Rolling Forcing [45]</td>
<td>1.3B</td>
<td>17.5</td>
<td>81.22</td>
<td>84.08</td>
<td>69.78</td>
</tr>
<tr>
<td><strong>Ours</strong></td>
<td><strong>1.3B</strong></td>
<td><strong>23.1</strong></td>
<td><strong>84.13</strong></td>
<td><strong>84.84</strong></td>
<td>81.32</td>
</tr>
</tbody>
</table>

- **分析:**
    - **综合性能:** `Ours` (本文方法) 的 VBench 总分达到了 84.13，在所有自回归模型中排名第一，非常接近甚至在某些子项上超过了速度慢得多的教师模型 `Wan-2.1`。这证明了 `Reward Forcing` 在蒸馏过程中很好地保持了视频的整体质量和语义对齐度。
    - **效率优势:** 本文方法的生成速度达到了 **23.1 FPS**，是所有对比模型中最快的。这比之前的 SOTA 模型 `LongLive` (20.7 FPS) 还要快 11.6%，比 `Self Forcing` (17.0 FPS) 快了 36%。这一速度使其非常适合实时交互应用。
    - **权衡:** 值得注意的是，本文方法是在使用了最小注意力窗口的情况下取得这些成绩的，这进一步凸显了 `EMA-Sink` 在小窗口下依然能有效维持长程依赖的能力。

### 6.1.2. 长视频生成性能
以下是原文 Table 2 的结果，比较了模型在60秒长视频生成任务上的表现：

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="7">VBench Long Evaluation Scores ↑</th>
<th rowspan="2">Drift↓</th>
<th colspan="3">Qwen3-VL Score ↑</th>
</tr>
<tr>
<th>Total</th>
<th>Subject</th>
<th>Background</th>
<th>Smoothness</th>
<th>Dynamic</th>
<th>Aesthetic</th>
<th>Imaging Quality</th>
<th>Visual</th>
<th>Dynamic</th>
<th>Text</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="12">Diffusion Forcing</td>
</tr>
<tr>
<td>SkyReels-V2 [7]</td>
<td>75.94</td>
<td>96.43</td>
<td>96.59</td>
<td>98.91</td>
<td>39.86</td>
<td>50.76</td>
<td>58.65</td>
<td>7.315</td>
<td>3.30</td>
<td>3.05</td>
<td>2.70</td>
</tr>
<tr>
<td colspan="12">Distilled Causal</td>
</tr>
<tr>
<td>CausVid [89]</td>
<td>77.78</td>
<td>97.92</td>
<td>96.62</td>
<td>98.47</td>
<td>27.55</td>
<td>58.39</td>
<td>67.77</td>
<td>2.906</td>
<td>4.66</td>
<td>3.16</td>
<td>3.32</td>
</tr>
<tr>
<td>Self Forcing [30]</td>
<td>79.34</td>
<td>97.10</td>
<td>96.03</td>
<td>98.48</td>
<td>54.94</td>
<td>54.40</td>
<td>67.61</td>
<td>5.075</td>
<td>3.89</td>
<td>3.44</td>
<td>3.11</td>
</tr>
<tr>
<td>LongLive [82]</td>
<td>79.53</td>
<td>97.96</td>
<td>96.50</td>
<td>98.79</td>
<td>35.54</td>
<td>57.81</td>
<td>69.91</td>
<td>2.531</td>
<td>4.79</td>
<td>3.81</td>
<td>3.98</td>
</tr>
<tr>
<td><strong>Ours</strong></td>
<td><strong>81.41</strong></td>
<td><strong>97.26</strong></td>
<td><strong>96.05</strong></td>
<td><strong>98.88</strong></td>
<td><strong>66.95</strong></td>
<td><strong>57.47</strong></td>
<td><strong>70.06</strong></td>
<td><strong>2.505</strong></td>
<td><strong>4.82</strong></td>
<td><strong>4.18</strong></td>
<td><strong>4.04</strong></td>
</tr>
</tbody>
</table>

- **分析:**
    - **动态性巨大提升:** 这是最显著的结果。本文方法的 `Dynamic` 分数达到了 **66.95**，远超所有基线。相比于 `LongLive` 的 35.54，提升了 **88.4%**。这直接、强有力地证明了 `Re-DMD` 机制在解决运动停滞问题上的决定性作用。
    - **长期一致性:** 本文方法的 `Drift` 值为 **2.505**，是所有模型中最低的，表明其在长达60秒的生成过程中保持了最佳的质量稳定性。这验证了 `EMA-Sink` 在维持长期一致性方面的有效性。
    - **VLM 评估:** 在更接近人类感知的 `Qwen3-VL` 评估中，本文方法在<strong>视觉质量 (4.82)</strong>、<strong>动态性 (4.18)</strong> 和<strong>文本对齐 (4.04)</strong> 三个维度上均取得了最高分，进一步证实了其综合性能的优越性。
    - **定性展示:** 原文 Figure 4 和 5 中的视觉对比也直观地展示了这些优势。本文方法生成的视频（如赛车）展现出更强的运动感和更连贯的场景变换，而基线模型则显得更为静止或出现不一致。

## 6.2. 消融实验/参数分析
消融实验旨在验证模型中每个新组件的必要性和有效性。以下是原文 Table 3 的结果：

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="4">VBench Evaluation Scores ↑</th>
<th rowspan="2">Drift↓</th>
</tr>
<tr>
<th>Background</th>
<th>Smooth</th>
<th>Dynamic</th>
<th>Quality</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="6"><strong>Improvement</strong></td>
</tr>
<tr>
<td>Ours</td>
<td>95.85</td>
<td>98.91</td>
<td><strong>64.06</strong></td>
<td>71.42</td>
<td>1.77</td>
</tr>
<tr>
<td>w/o Re-DMD</td>
<td>95.61</td>
<td>98.64</td>
<td>43.75</td>
<td>70.50</td>
<td>2.65</td>
</tr>
<tr>
<td>w/o Re-DMD w/o EMA</td>
<td>95.07</td>
<td>98.82</td>
<td>35.15</td>
<td>70.57</td>
<td>2.51</td>
</tr>
<tr>
<td>w/o Sink</td>
<td>94.94</td>
<td>98.56</td>
<td>51.56</td>
<td>69.92</td>
<td>5.08</td>
</tr>
<tr>
<td colspan="6"><strong>Impact of α</strong></td>
</tr>
<tr>
<td>α = 0.99</td>
<td>95.90</td>
<td>98.96</td>
<td>65.15</td>
<td>70.81</td>
<td>2.52</td>
</tr>
<tr>
<td>α = 0.9</td>
<td>95.80</td>
<td>99.09</td>
<td>63.15</td>
<td>71.37</td>
<td>3.23</td>
</tr>
<tr>
<td>α = 0.5</td>
<td>94.57</td>
<td>98.89</td>
<td>64.37</td>
<td>71.11</td>
<td>3.78</td>
</tr>
<tr>
<td colspan="6"><strong>Impact of β</strong></td>
</tr>
<tr>
<td>β = 1</td>
<td>95.14</td>
<td>98.31</td>
<td>54.68</td>
<td>71.73</td>
<td>2.63</td>
</tr>
<tr>
<td>β = 2/3</td>
<td>95.02</td>
<td>98.46</td>
<td>60.93</td>
<td>70.61</td>
<td>1.91</td>
</tr>
<tr>
<td>β = 1/3</td>
<td>94.94</td>
<td>98.43</td>
<td>58.59</td>
<td>69.29</td>
<td>2.02</td>
</tr>
<tr>
<td>β = 1/5</td>
<td>92.40</td>
<td>96.40</td>
<td>94.53</td>
<td>68.26</td>
<td>3.13</td>
</tr>
</tbody>
</table>

- **组件有效性分析:**
    - **`Re-DMD` 的作用:** 移除 `Re-DMD` (`w/o Re-DMD`) 后，`Dynamic` 分数从 64.06 **骤降至 43.75**。这清晰地表明，`Re-DMD` 是提升视频动态性的核心驱动力。
    - **`EMA-Sink` 的作用:** 在移除了 `Re-DMD` 的基础上再移除 `EMA-Sink`，`Dynamic` 分数进一步下降到 35.15，同时 `Smooth` 分数也出现波动。这说明 `EMA-Sink` 自身也能通过引入近期动态来改善运动性，并对维持平滑过渡至关重要。
    - **Sink Token 的作用:** 完全移除沉点词元 (`w/o Sink`) 后，`Drift` 值从 1.77 飙升至 5.08，表明质量稳定性急剧下降，验证了注意力沉点机制本身对于维持长时程生成的基础作用。

- **超参数影响分析:**
    - **EMA 更新权重 $\alpha$:** 实验显示 $\alpha$ 在平衡**运动平滑度**和<strong>长期一致性 (Drift)</strong> 之间起着关键作用。$\alpha=0.99$ 时漂移最小，但平滑度略逊于 $\alpha=0.9$；而 $\alpha=0.9$ 时平滑度最高，但漂移增大。这说明 $\alpha$ 控制着模型对历史的“遗忘”速度，需要在长期记忆和对近期变化的响应之间找到一个最佳平衡点。
    - **奖励权重 $\beta$:** $\beta$ 控制着奖励信号的强度。$\beta$ 过小 (如 1/5) 会使模型过度追求动态性，导致 `Dynamic` 分数异常高 (94.53)，但牺牲了背景一致性、平滑度和成像质量。$\beta$ 过大 (如 1) 则奖励信号太弱，动态性提升不明显 (54.68)。实验选择的 $\beta=1/2$ 是一个在各项指标间取得良好平衡的折中选择。

# 7. 总结与思考

## 7.1. 结论总结
本论文成功地识别并解决了高效流式视频生成中的一个核心痛点——**运动停滞**。作者提出的 `Reward Forcing` 框架，通过其两大创新支柱，为生成高质量、高动态性且长时一致的视频提供了强大而高效的解决方案。
- **`EMA-Sink`** 机制通过动态更新的全局上下文，巧妙地摆脱了对静态初始帧的过度依赖，既保证了长期的连贯性，又注入了必要的动态变化。
- **`Re-DMD`** 框架则通过引入奖励信号来加权蒸馏过程，首次实现了对“动态性”这一特定质量维度的定向优化，显著提升了生成视频的运动表现。

  实验结果充分证明，`Reward Forcing` 在短视频和长视频任务上均达到了最先进的性能，并在效率上刷新了纪录，为交互式、动态虚拟世界的实时模拟铺平了道路。

## 7.2. 局限性与未来工作
论文在附录中坦诚地讨论了当前方法的局限性和未来方向：
- **局限性:**
    1.  **奖励模型的偏见:** 方法的性能在很大程度上依赖于所使用的奖励模型 (`VideoAlign` 的运动质量评分)。如果奖励模型本身存在偏见（例如，偏好某种特定类型的运动），那么蒸馏出的学生模型也会继承这种偏见。
    2.  **评估指标与奖励的不完全对齐:** 奖励模型追求的目标（如动态性）与标准评测基准 (`VBench`) 的评分维度不完全一致。有时，奖励得分的提升可能不会成比例地体现在 `VBench` 总分上。
- **未来工作:**
    1.  **更先进的奖励模型:** 探索和设计更复杂、更全面的奖励模型是未来的一个重要方向。例如，可以引入物理规律、因果关系和语义常识作为奖励的先验知识，以生成更真实、更合理的动态世界。
    2.  **改进评估框架:** 随着生成能力的提升，需要发展更精细化的评估框架，以更准确地衡量视频的物理真实性和语义连贯性。

## 7.3. 个人启发与批判
- **个人启发:**
    1.  **问题导向的创新:** 本文是一个优秀的问题驱动研究范例。它没有泛泛地追求“更好”的模型，而是精准定位了现有SOTA方法 (`LongLive`) 的核心缺陷（运动停滞），并为此设计了两个高度针对性的解决方案。
    2.  **RL 与蒸馏的优雅结合:** `Re-DMD` 的思想非常具有启发性。它不是简单地用 RL 做后期的微调，而是将 RL 的奖励信号无缝地融入到了模型蒸馏这一核心训练阶段。这种“加权学习”的思想，即将外部知识（奖励）转化为训练中的注意力（权重），可以广泛应用于其他领域的模型蒸馏任务，例如，蒸馏出具有特定风格的图像模型，或更具创造力的语言模型。
    3.  **简单而有效:** `EMA-Sink` 的设计体现了工程上的智慧。它用一个非常简单的指数移动平均公式，在几乎零额外计算成本的情况下，解决了复杂的长期依赖问题，是“奥卡姆剃刀”原则的良好体现。

- **批判性思考:**
    1.  **对奖励模型的依赖:** 如作者所言，该方法的效果上限受限于奖励模型的质量。这引入了一个新的研究环节：如何设计和训练高质量的、无偏见的、多维度的视频奖励模型。这本身就是一个充满挑战的领域。
    2.  **超参数的敏感性:** 消融实验表明，模型对超参数 $\alpha$ 和 $\beta$ 的选择较为敏感，需要仔细调优才能在不同指标间达到平衡。这可能会增加方法在不同数据集或任务上复现和应用的难度。
    3.  **动态性的定义:** “动态性”本身是一个主观且多维度的概念。当前方法依赖于单一的奖励分数来量化它。但“好的动态”可能包含平滑的镜头移动、激烈的人物动作、微妙的表情变化等多种形式。未来的工作或许可以探索使用多头奖励模型，分别对不同类型的动态性进行建模和奖励，以生成内容更丰富的视频。