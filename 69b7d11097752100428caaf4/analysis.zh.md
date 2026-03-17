# 1. 论文基本信息

## 1.1. 标题
论文标题为 **DiffusionNFT: Online Diffusion Reinforcement with Forward Process**（DiffusionNFT：基于正向过程的在线扩散强化学习）。该标题直接揭示了论文的核心创新点：提出了一种名为 DiffusionNFT 的新方法，该方法在扩散模型（Diffusion Models）的<strong>正向过程（Forward Process）</strong>而非传统的反向采样过程中进行在线强化学习（Online Reinforcement Learning）。

## 1.2. 作者
论文作者团队来自清华大学（Tsinghua University）、英伟达（NVIDIA）和斯坦福大学（Stanford University）。主要作者包括 Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, 和 Ming-Yu Liu。其中 Jun Zhu 为通讯作者（Corresponding Author）。这表明该研究是由学术界顶尖高校与工业界领军企业合作完成的，兼具理论深度与工程落地能力。

## 1.3. 发表期刊/会议
该论文发布于 arXiv 预印本平台，发布时间为 2025 年 9 月 19 日（UTC）。arXiv 是计算机科学、物理学、数学等领域研究人员分享最新研究成果的主要平台。虽然截至本文分析时可能尚未经过同行评议正式发表在顶级会议（如 NeurIPS, ICML, CVPR 等）上，但鉴于作者团队的背景及研究内容的创新性，该工作具有极高的学术关注度和影响力潜力。

## 1.4. 发表年份
2025 年。

## 1.5. 摘要
论文旨在解决将在线强化学习（RL）应用于扩散模型时面临的似然不可计算（intractable likelihoods）的挑战。现有的方法（如 FlowGRPO）通过离散化反向采样过程来应用 GRPO 风格的训练，但存在求解器限制、正反向不一致以及难以集成无分类器引导（CFG）等缺陷。作者提出了 **Diffusion Negative-aware FineTuning (DiffusionNFT)**，这是一种新的在线 RL 范式，通过流匹配（flow matching）直接在正向扩散过程上优化扩散模型。DiffusionNFT 通过对比“正向”和“负向”生成样本定义隐式的策略改进方向，将强化信号自然地融入监督学习目标中。该方法支持任意黑盒求解器，无需似然估计，仅需干净图像而非采样轨迹。实验表明，DiffusionNFT 的效率比 FlowGRPO 最高高出 25 倍，且无需 CFG。例如，在 GenEval 任务上，DiffusionNFT 在 1k 步内将分数从 0.24 提升至 0.98，而 FlowGRPO 需要 5k 步以上且需额外使用 CFG。

## 1.6. 原文链接
*   **arXiv 摘要页:** https://arxiv.org/abs/2509.16117
*   **PDF 下载:** https://arxiv.org/pdf/2509.16117v2
*   **发布状态:** 预印本 (Preprint)

# 2. 整体概括

## 2.1. 研究背景与动机
### 2.1.1. 核心问题
在线强化学习（Online RL）在大语言模型（LLM）的后训练（post-training）中取得了巨大成功（如 RLHF），显著提升了模型的对齐和推理能力。然而，将这一成功复现到视觉生成领域的扩散模型（Diffusion Models）上却面临巨大挑战。核心难点在于扩散模型的<strong>似然函数（likelihood function）难以精确计算</strong>。策略梯度（Policy Gradient）算法通常假设模型似然是可计算的，但这在扩散模型中不成立，因为扩散过程涉及连续的随机微分方程（SDE）或常微分方程（ODE），其概率密度只能通过昂贵的近似方法来估计。

### 2.1.2. 现有研究的局限性
为了绕过似然计算障碍，最近的工作（如 FlowGRPO）尝试将扩散采样过程离散化，将其重构为多步决策问题，从而应用 GRPO 等算法。然而，作者指出这种基于<strong>反向过程（Reverse Process）</strong>的强化学习方法存在三个根本性缺陷：
1.  <strong>正反向不一致（Forward Inconsistency）：</strong> 仅关注反向采样过程破坏了与正向扩散过程的 adhere（遵循），可能导致模型退化为级联高斯分布，违背扩散模型的基本原理。
2.  <strong>求解器限制（Solver Restriction）：</strong> 数据收集过程依赖于一阶 SDE 采样器，无法充分利用对生成效率更有利的高阶 ODE 求解器。
3.  <strong>CFG 集成复杂（Complicated CFG Integration）：</strong> 扩散模型严重依赖无分类器引导（Classifier-Free Guidance, CFG）来提升生成质量，这通常需要训练条件模型和无条件模型。现有的 RL 实践通常在后期集成 CFG，导致复杂且低效的双模型优化方案。

### 2.1.3. 创新切入点
作者提出了一个关键问题：<strong>扩散强化学习能否在正向过程（Forward Process）而非反向过程上进行？</strong>
基于扩散策略只有一个正向（加噪）过程但有多个反向（去噪）过程（如不同采样器）的特性，作者提出了 **Diffusion Negative-aware FineTuning (DiffusionNFT)**。该方法直接在正向扩散过程上通过流匹配目标进行策略优化，通过对比高奖励（正向）和低奖励（负向）生成样本来定义隐式的策略改进方向。

## 2.2. 核心贡献/主要发现
1.  **新范式 DiffusionNFT：** 提出了一种基于正向过程的在线 RL 范式，无需似然估计，消除了对特定 SDE 采样器的依赖，支持任意黑盒求解器。
2.  <strong>负向感知微调（Negative-aware FineTuning）：</strong> 引入负向数据（低奖励样本）作为训练信号，通过对比正向和负向策略定义改进方向，将强化信号融入标准的监督学习（SL）目标中。
3.  **高效性与 CFG-Free：** 实验显示 DiffusionNFT 比 FlowGRPO 效率高 3 到 25 倍。它完全不需要 CFG 即可达到甚至超越使用 CFG 的基线模型性能。
4.  **性能提升：** 在多个基准测试（GenEval, OCR, PickScore 等）上，基于 SD3.5-Medium 的 DiffusionNFT 模型显著优于更大的模型（如 SD3.5-L, FLUX.1-Dev）及 FlowGRPO 基线。

# 3. 预备知识与相关工作

## 3.1. 基础概念
为了理解本文，读者需要掌握以下核心概念：

### 3.1.1. 扩散模型与流匹配 (Diffusion Models & Flow Matching)
扩散模型通过一个<strong>正向过程（Forward Process）</strong>逐渐向干净数据 $\pmb{x}_0$ 添加高斯噪声，直到变成纯噪声 $\pmb{x}_T$。然后通过学习一个<strong>反向过程（Reverse Process）</strong>从噪声中恢复数据。
<strong>流匹配（Flow Matching）</strong>是一种训练扩散模型的方法，它学习一个速度场（velocity field）$\pmb{v}_\theta(\pmb{x}_t, t)$，该场描述了数据点在从噪声到数据的路径上的切线方向。训练目标是最小化预测速度与真实速度之间的差异。
正向加噪过程通常可以参数化为：
$$
\pmb { x } _ { t } = \alpha _ { t } \pmb { x } _ { 0 } + \sigma _ { t } \pmb { \epsilon } , \pmb { \epsilon } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )
$$
其中 $\alpha_t$ 和 $\sigma_t$ 是噪声调度系数。

### 3.1.2. 强化学习与策略梯度 (Reinforcement Learning & Policy Gradient)
在强化学习中，<strong>智能体（Agent）</strong>通过与环境交互获得<strong>奖励（Reward）</strong>来优化其<strong>策略（Policy）</strong>。策略梯度算法（如 PPO, GRPO）通过计算奖励对策略参数的梯度来更新模型。
*   **GRPO (Group Relative Policy Optimization):** 一种无需价值函数（Value Function）的策略优化算法，它通过对一组采样输出的奖励进行归一化来计算优势（Advantage），常用于大语言模型的对齐。
*   <strong>似然（Likelihood）:</strong> 策略梯度算法通常依赖于计算给定动作（或生成序列）的概率（似然）。对于自回归模型（如 LLM），这是容易计算的；但对于扩散模型，由于涉及连续潜变量和积分，精确似然难以计算。

### 3.1.3. 无分类器引导 (Classifier-Free Guidance, CFG)
CFG 是一种在推理时提高扩散模型生成质量的技术。它通过结合条件模型（根据提示词生成）和无条件模型（无提示词生成）的输出来“引导”生成过程向条件分布靠近。公式上通常表现为速度场的线性组合：$\pmb{v}_{guided} = \pmb{v}_{cond} + s \cdot (\pmb{v}_{cond} - \pmb{v}_{uncond})$，其中 $s$ 是引导强度。然而，CFG 需要维护两个模型或两次前向传播，增加了计算成本。

## 3.2. 前人工作
*   **基于似然的方法：** 如 Diffusion-DPO，试图将直接偏好优化（DPO）适配到扩散模型，但需要额外的似然近似。
*   **基于策略梯度的方法：** 如 Black et al. (2023) 和 FlowGRPO (Liu et al., 2025)。FlowGRPO 通过将反向采样离散化为多步 MDP，使得每一步的转移概率变为可处理的高斯分布，从而应用 GRPO。但如前所述，这带来了求解器限制和正反向不一致问题。
*   **无似然方法：** 如奖励反向传播（Reward Backpropagation），但受限于可微奖励和内存成本。

## 3.3. 技术演进与差异化分析
下图（原文 Figure 2）直观展示了本文方法（NFT）与传统方法（GRPO）在流程上的根本区别：

![Figure 2: Comparison between Forward-Process RL (NFT) and Reverse-Process RL (GRPO). NFT allows using any solvers and does not require storing the whole sampling trajectory for optimization.](images/2.jpg)

**差异化分析：**
*   **优化对象：** FlowGRPO 优化反向去噪步骤的转移概率；DiffusionNFT 优化正向加噪过程的速度预测器。
*   **数据需求：** FlowGRPO 需要存储完整的采样轨迹（trajectory）来计算每一步的优势；DiffusionNFT 仅需最终的干净图像及其奖励，无需中间轨迹。
*   **求解器：** FlowGRPO 绑定于一阶 SDE 采样器；DiffusionNFT 解耦了训练与采样，允许使用任意高阶 ODE 求解器进行数据收集。
*   **CFG 依赖：** FlowGRPO 通常依赖 CFG 来保证质量；DiffusionNFT 通过 RL 后训练内化了引导能力，实现了 CFG-Free。

# 4. 方法论

本章将详细拆解 DiffusionNFT 的技术方案。作者的核心思想是不使用策略梯度，而是通过监督学习（SL）的目标，利用正向和负向样本的对比来隐式地优化策略。

## 4.1. 问题设定 (Problem Setup)
### 4.1.1. 在线强化学习设定
考虑一个预训练的扩散策略 $\pi^{\mathrm{old}}$ 和提示词数据集 $\{c\}$。在每次迭代中，对于每个提示词 $c$，采样 $K$ 张图像 $\pmb{x}_0^{1:K}$。
每个图像获得一个奖励 $r \in [0, 1]$，代表其最优性概率（optimality probability）：
$$
r ( \pmb { x } _ { 0 } , \pmb { c } ) : = p ( \mathbf { o } = 1 | \pmb { x } _ { 0 } , \pmb { c } )
$$
其中 $\mathbf{o}=1$ 表示样本是“最优”的。这个奖励将连续值映射为二元划分的概率。

### 4.1.2. 正负数据集划分
收集到的数据可以根据奖励随机划分为两个虚拟子集：正向数据集 $\mathcal{D}^+$ 和负向数据集 $\mathcal{D}^-$。图像 $\pmb{x}_0$ 落入 $\mathcal{D}^+$ 的概率为 $r$，落入 $\mathcal{D}^-$ 的概率为 `1-r`。
这两个子集背后的分布分别为：
$$
\pi ^ { + } ( x _ { 0 } | c ) : = \pi ^ { \mathrm { o l d } } ( x _ { 0 } | \mathbf { o } = 1 , c ) = \frac { r ( x _ { 0 } , c ) } { p _ { \pi ^ { \mathrm { o l d } } } ( \mathbf { o } = 1 | c ) } \pi ^ { \mathrm { o l d } } ( x _ { 0 } | c )
$$
$$
\pi ^ { - } ( \boldsymbol { x } _ { 0 } | \boldsymbol { c } ) : = \pi ^ { \mathrm { o l d } } ( \boldsymbol { x } _ { 0 } | \mathbf { o } = 0 , \boldsymbol { c } ) = \frac { 1 - r ( \boldsymbol { x } _ { 0 } , \boldsymbol { c } ) } { 1 - p _ { \pi ^ { \mathrm { o l d } } } ( \mathbf { o } = 1 | \boldsymbol { c } ) } \pi ^ { \mathrm { o l d } } ( \boldsymbol { x } _ { 0 } | \boldsymbol { c } )
$$
显然，策略的优劣关系为 $\pi^+ \succ \pi^{\mathrm{old}} \succ \pi^-$。传统的拒绝微调（RFT）仅训练 $\mathcal{D}^+$，但 DiffusionNFT 认为负向反馈 $\mathcal{D}^-$ 对策略改进至关重要。

## 4.2. 负向感知扩散强化 (Negative-aware Diffusion Reinforcement)
### 4.2.1. 强化引导 (Reinforcement Guidance)
作者定义了一个改进方向 $\Delta \in \mathbb{R}^n$，目标是优化速度预测器 $\pmb{v}_\theta$  towards 一个理想的目标速度 $\pmb{v}^*$：
$$
{ \boldsymbol v } ^ { * } ( { \boldsymbol x } _ { t } , c , t ) : = { \boldsymbol v } ^ { \mathrm { o l d } } ( { \boldsymbol x } _ { t } , c , t ) + \frac { 1 } { \beta } \Delta ( { \boldsymbol x } _ { t } , c , t )
$$
其中 $\beta$ 是引导强度超参数。这类似于 CFG 中的引导公式，但这里的 $\Delta$ 是<strong>强化引导（Reinforcement Guidance）</strong>。

### 4.2.2. 改进方向定理 (Theorem 3.1)
为了确定 $\Delta$ 的形式，作者分析了 $\pi^+, \pi^-, \pi^{\mathrm{old}}$ 分布之间的差异。
<strong>定理 3.1 (改进方向):</strong> 考虑对应于策略三元组 $\pi^+, \pi^-, \pi^{\mathrm{old}}$ 的扩散模型速度场 $\pmb{v}^+, \pmb{v}^-, \pmb{v}^{\mathrm{old}}$。这些模型之间的方向差异是成比例的：
$$
\begin{array} { r l r } & { } & { \Delta : = [ 1 - \alpha ( { \pmb x } _ { t } ) ] [ { \pmb v } ^ { o l d } ( { \pmb x } _ { t } , { \pmb c } , t ) - { \pmb v } ^ { - } ( { \pmb x } _ { t } , { \pmb c } , t ) ] } \\ & { } & { = \alpha ( { \pmb x } _ { t } ) [ { \pmb v } ^ { + } ( { \pmb x } _ { t } , { \pmb c } , t ) - { \pmb v } ^ { o l d } ( { \pmb x } _ { t } , { \pmb c } , t ) ] . } \end{array}
$$
其中 $0 \leq \alpha ( { \pmb x } _ { t } ) \leq 1$ 是一个标量系数：
$$
\alpha ( { \pmb x } _ { t } ) : = \frac { \pi _ { t } ^ { + } ( { \pmb x } _ { t } | { \pmb c } ) } { \pi _ { t } ^ { o l d } ( { \pmb x } _ { t } | { \pmb c } ) } \mathbb { E } _ { \pi ^ { o l d } ( { \pmb x } _ { 0 } | { \pmb c } ) } r ( { \pmb x } _ { 0 } , { \pmb c } )
$$
该定理表明，理想的引导方向 $\Delta$ 可以通过正向策略与旧策略的差，或旧策略与负向策略的差来表示。下图（原文 Figure 3）展示了这一改进方向的几何直觉：

![Figure 3: Improvement Direction.](images/3.jpg)
*该图像是示意图，展示了DiffusionNFT中的改进方向。图中对比了正向生成（$D^+$）和负向生成（$D^-$）的流动，通过引导向量（`v_ heta`）指示了优化过程的发展，利用奖励值$r$从0到1的变化反映了生成效果的提升。*

### 4.2.3. 策略优化目标 (Theorem 3.2)
基于上述理论，作者提出了一个直接优化 $\pmb{v}_\theta$  towards $\pmb{v}^*$ 的训练目标。
<strong>定理 3.2 (策略优化):</strong> 考虑训练目标：
$$
\mathcal { L } ( \theta ) = \mathbb { E } _ { c , \pi ^ { o l d } ( { \pmb x } _ { 0 } \mid c ) , t } r \| { \pmb v } _ { \theta } ^ { + } ( { \pmb x } _ { t } , { \pmb c } , t ) - { \pmb v } \| _ { 2 } ^ { 2 } + ( 1 - r ) \| { \pmb v } _ { \theta } ^ { - } ( { \pmb x } _ { t } , { \pmb c } , t ) - { \pmb v } \| _ { 2 } ^ { 2 }
$$
其中定义了<strong>隐式正向策略（Implicit positive policy）</strong>：
$$
\pmb { v } _ { \theta } ^ { + } ( \pmb { x } _ { t } , \pmb { c } , t ) : = ( 1 - \beta ) \pmb { v } ^ { o l d } ( \pmb { x } _ { t } , \pmb { c } , t ) + \beta \pmb { v } _ { \theta } ( \pmb { x } _ { t } , \pmb { c } , t )
$$
和<strong>隐式负向策略（Implicit negative policy）</strong>：
$$
\pmb { v } _ { \theta } ^ { - } ( \pmb { x } _ { t } , \pmb { c } , t ) : = ( 1 + \beta ) \pmb { v } ^ { o l d } ( \pmb { x } _ { t } , \pmb { c } , t ) - \beta \pmb { v } _ { \theta } ( \pmb { x } _ { t } , \pmb { c } , t )
$$
给定无限数据和模型容量，该目标的最优解满足：
$$
{ v } _ { \theta ^ { * } } ( { x } _ { t } , c , t ) = { v } ^ { o l d } ( { x } _ { t } , c , t ) + \frac { 2 } { \beta } \Delta ( { x } _ { t } , c , t )
$$
这个公式非常关键。它表明通过最小化上述损失函数 $\mathcal{L}(\theta)$，模型实际上是在学习向改进方向 $\Delta$ 移动。注意这里不需要训练两个独立的模型 $\pmb{v}^+$ 和 $\pmb{v}^-$，而是通过**隐式参数化技术**直接优化单个目标策略 $\pmb{v}_\theta$。下图（原文 Figure 4）展示了这一联合优化过程：

![Figure 4: DiffusionNFT jointly optimizes two dual diffusion objectives, on both positive $( r = 1 )$ and negative $( r = 0$ branches. Rather than training two independent models ${ \\boldsymbol { v } } _ { \\theta } ^ { + }$ and ${ \\boldsymbol v } _ { \\boldsymbol { \\theta } _ { } } ^ { - }$ , it adopts a implicit parameerization technique that directlyoptimizes single target poliy ${ \\pmb v } _ { \\theta }$ .](images/4.jpg)
*该图像是示意图，展示了DiffusionNFT在积极（$r = 1$）和消极（$r = 0$）传播目标上进行联合优化的过程。图中通过条件输入（如“可爱的小狗”）和添加噪声生成图像，采用隐式参数化技术来优化单一目标策略${\pmb v}_{\theta}$。优化过程考虑了最优性奖励$r^{1:K} \in [0, 1]$，并通过具体的损失函数设计实现目标策略的提升。*

## 4.3. 实际实现 (Practical Implementation)
作者提供了 DiffusionNFT 的伪代码（Algorithm 1），并阐述了几个关键设计选择。

### 4.3.1. 最优性奖励归一化
由于视觉强化学习中的奖励通常是无约束的连续标量，需要将其转换为 $r \in [0, 1]$ 的最优性概率。受 GRPO 启发，采用组内归一化：
$$
r ( \pmb { x } _ { 0 } , \pmb { c } ) : = \frac { 1 } { 2 } + \frac { 1 } { 2 } \mathrm { c } \mathrm { 1 } \mathrm { i } \mathrm { p } \left[ \frac { r ^ { \mathrm { r a w } } ( \pmb { x } _ { 0 } , \pmb { c } ) - \mathbb { E } _ { \pi ^ { \mathrm { o l d } } ( \cdot \vert \pmb { c } ) } r ^ { \mathrm { r a w } } ( \pmb { x } _ { 0 } , \pmb { c } ) } { Z _ { c } } , - 1 , 1 \right]
$$
其中 $Z_c$ 是归一化因子（如全局奖励标准差）。

### 4.3.2. 采样策略的软更新 (Soft Update)
由于 DiffusionNFT 是离策略（off-policy）的，采样策略 $\pi^{\mathrm{old}}$ 与训练策略 $\pi_\theta$ 解耦。作者采用指数移动平均（EMA）式的软更新，而非硬更新：
$$
\theta ^ { \mathrm { o l d } } \leftarrow \eta _ { i } \theta ^ { \mathrm { o l d } } + ( 1 - \eta _ { i } ) \theta
$$
其中 $\eta_i$ 控制学习速度与稳定性的权衡。完全在策略（$\eta=0$）会导致不稳定，完全离策略（$\eta \to 1$）会导致收敛慢。

### 4.3.3. 自适应损失加权 (Adaptive Loss Weighting)
为了加速训练，作者采用了一种自适应加权方案，将速度预测损失转换为自归一化的 $\pmb{x}_0$ 回归形式（受 DMD 方法启发）：
$$
w ( t ) \| v _ { \theta } ( x _ { t } , c , t ) - v \| _ { 2 } ^ { 2 } \rightarrow \frac { \| x _ { \theta } ( x _ { t } , c , t ) - x _ { 0 } \| _ { 2 } ^ { 2 } } { \mathrm { s g } ( \mathrm { m e a n } ( \mathrm { a b s } ( x _ { \theta } ( x _ { t } , c , t ) - x _ { 0 } ) ) ) }
$$
其中 $\mathrm{sg}$ 是停止梯度（stop-gradient）算子。

### 4.3.4. 无 CFG 优化 (CFG-Free Optimization)
作者将 CFG 视为一种离线的强化引导形式。在 DiffusionNFT 中，直接丢弃 CFG，仅初始化条件模型。实验表明，RL 后训练可以有效地学习或替代 CFG 的功能，从而实现单模型的高效推理。

# 5. 实验设置

## 5.1. 数据集
实验主要基于 **SD3.5-Medium** 模型（25 亿参数），分辨率为 $512 \times 512$。
*   **训练数据集:**
    *   **Pick-a-Pic:** 用于基于模型的奖励（PickScore, ClipScore, HPSv2.1）训练，以增强对齐和人类偏好。
    *   **GenEval 训练集:** 用于规则奖励 GenEval 训练。
    *   **OCR 训练集:** 用于规则奖励 OCR 训练。
*   **评估数据集:**
    *   **DrawBench:** 用于评估基于模型的奖励的泛化能力（Out-of-domain）。
    *   **GenEval/OCR 测试集:** 用于评估规则奖励任务的性能。

## 5.2. 评估指标
论文使用了多种指标来全面评估生成质量。以下是关键指标的定义：

### 5.2.1. GenEval
*   **概念定义:** 一个专注于评估文本到图像生成中对象组合能力的基准。它检查生成的图像是否包含提示词中要求的特定对象、颜色、位置和数量。
*   **数学公式:** 通常是一个准确率分数，即满足所有组合约束的图像比例。
    $$ \text{GenEval Score} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\text{Image}_i \text{ satisfies all constraints}) $$
*   **符号解释:** $N$ 是测试图像总数，$\mathbb{I}$ 是指示函数。

### 5.2.2. OCR (Optical Character Recognition) Score
*   **概念定义:** 评估模型在图像中渲染文本的准确性和可读性。
*   **数学公式:** 通常使用 OCR 工具识别图像中的文本，并与提示词中的目标文本计算编辑距离或匹配率。
    $$ \text{OCR Score} = \text{Similarity}(\text{OCR}(\text{Image}), \text{Target Text}) $$
*   **符号解释:** $\text{OCR}(\cdot)$ 表示 OCR 识别函数。

### 5.2.3. PickScore / ClipScore / HPSv2.1
*   **概念定义:** 基于深度学习模型的评分指标，用于衡量图像与文本提示的语义对齐程度（ClipScore, PickScore）或人类偏好（HPSv2.1, ImageReward）。
*   **数学公式:** 通常计算图像嵌入和文本嵌入之间的余弦相似度。
    $$ \text{Score} = \cos(E_{img}(\text{Image}), E_{text}(\text{Prompt})) $$
*   **符号解释:** $E_{img}, E_{text}$ 分别为图像和文本编码器。

## 5.3. 对比基线
*   **SD3.5-M (w/o CFG):** 未进行后训练的原始模型，不使用 CFG。
*   **SD3.5-M + CFG:** 原始模型，但在推理时使用 CFG 增强。
*   **FlowGRPO:** 当前最先进的基于反向过程的扩散强化学习基线。
*   **Larger Models:** SD3.5-L (8B), FLUX.1-Dev (12B)，用于验证小模型后训练能否超越大模型。

## 5.4. 训练细节
*   **微调技术:** 使用 LoRA ($\alpha=64, r=32$)。
*   **批次设置:** 每个 epoch 包含 48 组，组大小 $G=24$。
*   **采样步数:** 对比实验使用 10 步，多奖励训练使用 40 步以保证高质量数据收集。
*   **求解器:** 数据收集使用 40 步一阶 ODE 采样器（评估时），对比实验中使用 10 步。

# 6. 实验结果与分析

## 6.1. 核心结果分析
### 6.1.1. 多奖励联合训练性能
下表（原文 Table 1）展示了 DiffusionNFT 在多个基准测试上的综合表现。

以下是原文 Table 1 的结果：

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">#Iter</th>
<th colspan="2">Rule-Based</th>
<th colspan="6">Model-Based</th>
</tr>
<tr>
<th>GenEval</th>
<th>OCR</th>
<th>PickScore</th>
<th>ClipScore</th>
<th>HPSv2.1</th>
<th>Aesthetic</th>
<th>ImgRwd</th>
<th>UniRwd</th>
</tr>
</thead>
<tbody>
<tr>
<td>SD-XL‡</td>
<td></td>
<td>0.55</td>
<td>0.14</td>
<td>22.42</td>
<td>0.287</td>
<td>0.280</td>
<td>5.60</td>
<td>0.76</td>
<td>2.93</td>
</tr>
<tr>
<td>SD3.5-L‡</td>
<td></td>
<td>0.71</td>
<td>0.68</td>
<td>22.91</td>
<td>0.289</td>
<td>0.288</td>
<td>5.50</td>
<td>0.96</td>
<td>3.25</td>
</tr>
<tr>
<td>FLUX.1-Dev</td>
<td></td>
<td>0.66</td>
<td>0.59</td>
<td>22.84</td>
<td>0.295</td>
<td>0.274</td>
<td>5.71</td>
<td>0.96</td>
<td>3.27</td>
</tr>
<tr>
<td>SD3.5-M (w/o CFG)</td>
<td></td>
<td>0.24</td>
<td>0.12</td>
<td>20.51</td>
<td>0.237</td>
<td>0.204</td>
<td>5.13</td>
<td>-0.58</td>
<td>2.02</td>
</tr>
<tr>
<td>+ CFG</td>
<td>—</td>
<td>0.63</td>
<td>0.59</td>
<td>22.34</td>
<td>0.285</td>
<td>0.279</td>
<td>5.36</td>
<td>0.85</td>
<td>3.03</td>
</tr>
<tr>
<td>+ FlowGRPO†</td>
<td>&gt;5k</td>
<td>0.95</td>
<td>0.66</td>
<td>22.51</td>
<td>0.293</td>
<td>0.274</td>
<td>5.32</td>
<td>1.06</td>
<td>3.18</td>
</tr>
<tr>
<td rowspan="4">+ Ours</td>
<td>2k</td>
<td>0.66</td>
<td>0.92</td>
<td>22.41</td>
<td>0.290</td>
<td>0.280</td>
<td>5.32</td>
<td>0.95</td>
<td>3.15</td>
</tr>
<tr>
<td>4k</td>
<td>0.54</td>
<td>0.68</td>
<td>23.50</td>
<td>0.280</td>
<td>0.316</td>
<td>5.90</td>
<td>1.29</td>
<td>3.37</td>
</tr>
<tr>
<td>1.7k</td>
<td>0.94</td>
<td>0.91</td>
<td>23.80</td>
<td>0.293</td>
<td>0.331</td>
<td>6.01</td>
<td>1.49</td>
<td>3.49</td>
</tr>
</tbody>
</table>

**分析：**
1.  **超越 CFG：** 仅使用条件模型（w/o CFG）的 DiffusionNFT（1.7k 迭代）在 GenEval (0.94) 和 OCR (0.91) 上显著超越了使用 CFG 的原始 SD3.5-M (0.63, 0.59)。
2.  **超越大模型：** 2.5B 参数的 DiffusionNFT 模型在多个指标上超越了 8B 的 SD3.5-L 和 12B 的 FLUX.1-Dev。
3.  **效率优势：** 达到类似或更好性能，DiffusionNFT 仅需 1.7k 迭代，而 FlowGRPO 需要 >5k 迭代。

    下图（原文 Figure 1）展示了性能对比的可视化：

    ![Figure 1: Performance of DiffusionNFT. (a) Head-to-head comparison with FlowGRPO on the GenEval task. (b) By employing multiple reward models, DiffusionNFT significantly boosts the performance of SD3.5-Medium in every benchmark tested, while being fully CFG-free.](images/1.jpg)
    *该图像是一个图表，展示了DiffusionNFT与Flow-GRPO在GenEval任务上的性能对比（图(a)），以及在不同基准测试中，利用多个奖励模型显著提升SD3.5-Medium的表现（图(b)）。*

### 6.1.2. 头对头效率对比
在单奖励设置下，DiffusionNFT 与 FlowGRPO 进行了严格的效率对比。
*   **GenEval 任务：** DiffusionNFT 在 1k 步内将分数从 0.24 提升至 0.98，而 FlowGRPO 在 5k 步后仅达到 0.95 且需额外使用 CFG。
*   **时间效率：** 如下图所示（原文 Figure 6），DiffusionNFT 在墙钟时间（wall-clock time）上比 FlowGRPO 快 3 到 25 倍。

    ![Figure 5: Qualitative Comparison. The prompts are taken from GenEva1, OCR and DrawBench respectively, where we compare the corresponding FlowGRPO model with our model.](images/6.jpg)
    *该图像是图表，展示了DiffusionNFT与FlowGRPO在不同任务上的效率对比。图中横轴为训练时间（GPU小时），纵轴分别为OCR得分、PickScore和HPSv2.1得分。DiffusionNFT在各项测试中表现出显著的效率提升，最高达24倍。*

### 6.1.3. 定性对比
下图（原文 Figure 5）展示了生成样本的视觉质量对比。DiffusionNFT 生成的图像在遵循复杂提示（如物体位置、文本渲染）方面表现更好，且无需 CFG 即可保持高保真度。

![该图像是包含多幅插图的拼贴，展示了不同主题的城市景象，其中包括蓝色比萨、黄色棒球手套、涂鸦墙和纽约天际线等元素。画面色彩丰富，展现了城市文化的多样性和活力。](images/5.jpg)
*该图像是包含多幅插图的拼贴，展示了不同主题的城市景象，其中包括蓝色比萨、黄色棒球手套、涂鸦墙和纽约天际线等元素。画面色彩丰富，展现了城市文化的多样性和活力。*

## 6.2. 消融实验与参数分析
作者对关键设计选择进行了消融研究，验证了方法的有效性。

### 6.2.1. 负向损失的重要性
实验发现，如果移除负向策略损失（即仅使用正向数据 $\mathcal{D}^+$），奖励在在线训练期间几乎瞬间崩溃。这证明了<strong>负向信号（Negative Signals）</strong>在扩散 RL 中对于防止模式崩溃（Mode Collapse）至关重要，这与 LLM 中的观察（RFT 是强基线）不同。

### 6.2.2. 扩散采样器的选择
在线样本既用于奖励评估也用于训练数据，因此质量至关重要。下图（原文 Figure 7）显示 ODE 采样器优于 SDE 采样器，特别是在对噪声敏感的 PickScore 上。二阶 ODE 在 GenEval 上略优于一阶 ODE。

![该图像是一个示意图，展示了不同训练迭代下的 GenEval 分数变化。四条曲线分别对应于不同的超参数设置，表现出各自的收敛趋势，说明对于 $ u_i$ 值的调整对模型性能的影响。](images/7.jpg)
*该图像是一个示意图，展示了不同训练迭代下的 GenEval 分数变化。四条曲线分别对应于不同的超参数设置，表现出各自的收敛趋势，说明对于 $ u_i$ 值的调整对模型性能的影响。*

### 6.2.3. 自适应加权与软更新
*   **加权策略：** 下图（原文 Figure 9）显示，随着 $t$ 增大给予流匹配损失更高权重的自适应策略优于启发式选择（如 $w(t)=1-t$），后者会导致训练崩溃。
*   **软更新：** 下图（原文 Figure 8）表明，从较小的 $\eta$ 开始并逐渐增大，能在收敛速度和训练稳定性之间取得最佳平衡。完全在策略（$\eta=0$）会导致灾难性崩溃。

    ![Figure 9: Different time-dependent weighting strategies.](images/9.jpg)
    *该图像是图表，展示了不同时间依赖加权策略对GenEval得分和PickScore的影响。左侧图(a)显示了在训练迭代中，随着不同加权策略（如$w(t) = 1 - t$、$w(t) = 1$、$w(t) = t$和自适应权重）的变化，GenEval得分的变化曲线。右侧图(b)则显示了相应的PickScore变化。各条曲线的颜色和样式对应于不同的加权策略。*

    ![Figure 7: Different diffusion samplers for data collection.](images/8.jpg)
    *该图像是图表，展示了不同训练迭代下的GenEval得分和PickScore。左侧(a)图中，三种方法（1st-order SDE、1st-order ODE和2nd-order ODE）的得分随训练迭代增加的变化情况，以及它们的收敛趋势。右侧(b)图则展现了相同方法下的PickScore变化趋势。*

### 6.2.4. 引导强度 $\beta$
下图（原文 Figure 10）展示了引导强度 $\beta$ 的选择。$\beta$ 接近 1 时表现稳定，实践中选择 1 或 0.1（用于更快的奖励增长）。

![Figure 10: Choices of strength $\\beta$ .](images/10.jpg)
*该图像是一个示意图，展示了不同强度 `eta` 对 GenEval 分数的影响。随着训练迭代的增加，蓝色线（$eta = 0.01$）、橙色线（$eta = 1.0$）和绿色线（$eta = 10.0$）在 GenEval 分数上表现出不同的收敛趋势。*

# 7. 总结与思考

## 7.1. 结论总结
本文提出了 **DiffusionNFT**，一种基于正向过程的扩散模型在线强化学习新范式。
1.  **理论创新：** 通过在正向扩散过程上定义对比改进方向，将 RL 信号融入监督学习目标，避免了似然估计和反向过程离散化。
2.  **工程优势：** 支持任意黑盒求解器，无需存储采样轨迹，实现了 CFG-Free 的高效训练与推理。
3.  **实证效果：** 在效率上比 FlowGRPO 最高提升 25 倍，在性能上超越了更大的基础模型和 CFG 基线，证明了小模型通过后训练可以达到 SOTA 水平。

## 7.2. 局限性与未来工作
*   **奖励模型依赖：** 方法依赖于预训练的奖励模型（Reward Models）。如果奖励模型存在偏差（Bias），强化学习过程可能会放大这些偏差（Reward Hacking）。
*   **计算资源：** 尽管效率高于 FlowGRPO，但在线 RL 仍需大量的采样和推理计算，对于资源受限的场景仍具挑战。
*   **多模态扩展：** 目前主要在文生图任务上验证，未来可探索在视频生成、3D 生成等多模态任务上的应用。

## 7.3. 个人启发与批判
*   **正向过程的潜力：** 本文最大的启发在于重新审视了扩散模型的**正向过程**。传统 RL 方法过于执着于将生成过程建模为 MDP（反向去噪），而忽略了正向加噪过程本身包含的丰富分布信息。DiffusionNFT 证明了利用正向过程进行策略优化不仅可行，而且更高效。
*   **负向数据的价值：** 在 LLM 的 RLHF 中，负向样本往往被忽略或仅用于 DPO 的偏好对。本文强调了在扩散模型中显式利用负向数据（低奖励样本）来定义“避免方向”的重要性，这为防止生成模型的模式崩溃提供了新视角。
*   **CFG 的替代：** 实现 CFG-Free 是一个重要的工程里程碑。这意味着模型本身学会了“引导”，而不是依赖推理时的 tricks。这简化了部署流程，降低了推理延迟。
*   **潜在问题：** 论文中提到“负向损失移除会导致奖励瞬间崩溃”，这与 LLM 中的 RFT 表现不同。这暗示扩散模型的优化景观（Optimization Landscape）可能比自回归模型更敏感，更容易陷入局部最优。未来的工作可以进一步研究为何扩散模型对负向信号如此敏感，以及是否有更稳健的正则化方法。
*   **通用性：** 作者提到这是迈向统一监督学习和强化学习的一步。如果 DiffusionNFT 能推广到其他生成范式（如自回归视觉模型），将具有更广泛的意义。

    总体而言，DiffusionNFT 是一篇兼具理论深度和实用价值的优秀论文，为扩散模型的后训练提供了一条高效、 principled（有原则的）的新路径。