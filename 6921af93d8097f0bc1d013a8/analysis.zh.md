# 1. 论文基本信息

## 1.1. 标题
Flow-GRPO: Training Flow Matching Models via Online RL

## 1.2. 作者
Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Lil, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, Wanli Ouyang 等。
作者来自多个机构，包括香港中文大学 MMLab (MMLab, CUHK)、清华大学 (Tsinghua University)、快手科技 Kling Team (Kling Team, Kuaishou Technology)、南京大学 (Nanjing University) 和上海人工智能实验室 (Shanghai AI Laboratory)。

## 1.3. 发表期刊/会议
该论文为预印本 (preprint)，发布于 arXiv 平台。在人工智能和机器学习领域，arXiv 预印本是学术交流的重要组成部分，许多高质量的研究成果在正式发表前都会在此平台发布，具有较高的前瞻性。

## 1.4. 发表年份
2025年。

## 1.5. 摘要
本文提出了 Flow-GRPO，这是首个将在线策略梯度强化学习 (Reinforcement Learning, RL) 集成到流匹配 (Flow Matching) 模型中的方法。该方法采用两个关键策略：(1) 将确定性常微分方程 (Ordinary Differential Equation, ODE) 转换为等效的随机微分方程 (Stochastic Differential Equation, SDE)，使其在所有时间步长上匹配原始模型的边际分布 (marginal distribution)，从而为 RL 探索提供统计采样；(2) 采用去噪减少 (Denoising Reduction) 策略，在训练时减少去噪步骤，同时在推理时保留原始步骤，显著提高了采样效率而不牺牲性能。实验证明，Flow-GRPO 在多项文本到图像 (Text-to-Image, T2I) 任务中表现出色。在组合生成 (compositional generation) 任务中，经过 RL 微调的 SD3.5-M (Stable Diffusion 3.5 Medium) 在对象计数、空间关系和细粒度属性方面的 GenEval 准确率从 63% 提高到 95%。在视觉文本渲染 (visual text rendering) 任务中，准确率从 59% 提高到 92%，极大地增强了文本生成能力。Flow-GRPO 还在人类偏好对齐 (human preference alignment) 方面取得了显著进步。值得注意的是，该方法极少出现奖励欺骗 (reward hacking) 现象，即奖励的增加没有以图像质量或多样性明显下降为代价。

## 1.6. 原文链接
https://arxiv.org/abs/2505.05470
PDF 链接: https://arxiv.org/pdf/2505.05470v5.pdf
发布状态：预印本 (v5)。

# 2. 整体概括

## 2.1. 研究背景与动机
流匹配 (Flow Matching) 模型 [2, 3] 因其坚实的理论基础和在生成高质量图像方面的出色性能，已在图像生成领域占据主导地位 [4, 5]。然而，它们在处理涉及多个对象、属性和关系的复杂场景合成 [6, 7] 以及文本渲染 [8] 时，往往表现不佳。

与此同时，在线强化学习 (Online Reinforcement Learning, RL) [9] 在增强大型语言模型 (Large Language Models, LLMs) 的推理能力方面取得了显著成功 [10, 11]。虽然之前的研究主要集中于将 RL 应用于早期的扩散模型 (Diffusion Models) [12] 和针对流生成模型的离线 RL 技术（如直接偏好优化 (Direct Preference Optimization, DPO) [13, 14, 15]），但在线 RL 在推动流匹配生成模型方面的潜力仍未得到充分探索。

本研究旨在探索如何有效利用在线 RL 来改进流匹配模型。然而，将 RL 训练应用于流模型面临几个关键挑战：
1.  **确定性与随机性的冲突：** 流模型依赖基于常微分方程 (Ordinary Differential Equation, ODE) 的确定性生成过程 [3]，这意味着它们在推理过程中无法进行随机采样 (stochastic sampling)。相比之下，RL 依靠随机采样来探索环境，通过尝试不同的行动 (action) 并根据奖励 (reward) 进行改进来学习。RL 对随机性的需求与流匹配模型的确定性本质相冲突。
2.  **采样效率问题：** 在线 RL 依赖高效采样来收集训练数据，但流模型通常需要许多迭代步骤才能生成每个样本，这限制了效率。对于大型模型 [5, 4] 来说，这个问题更为突出。为了使 RL 在图像或视频生成等任务中具有实用性，提高采样效率至关重要。

## 2.2. 核心贡献/主要发现
为应对上述挑战，本文提出了 Flow-GRPO，该方法将 GRPO (Group Relative Policy Optimization) [16] 集成到用于文本到图像 (Text-to-Image, T2I) 生成的流匹配模型中，并采用了两个关键策略：

1.  <strong>ODE-to-SDE 转换 (ODE-to-SDE Conversion)：</strong> 该策略克服了原始流模型的确定性本质。通过将基于 ODE 的流转换为等效的随机微分方程 (Stochastic Differential Equation, SDE) 框架，在保持原始边际分布 (marginal distributions) 的同时引入了随机性。这使得模型能够进行 RL 探索所需的统计采样。
2.  <strong>去噪减少策略 (Denoising Reduction Strategy)：</strong> 为提高在线 RL 中的采样效率，该策略在训练期间减少去噪步骤 (denoising steps)，同时在推理 (inference) 时保留完整的去噪步骤。实验表明，使用较少的步骤可以保持性能，同时显著降低数据生成成本，从而大幅提高训练效率。

**核心发现和实证结果：**
*   **性能显著提升：** Flow-GRPO 在多个 T2I 任务中表现出显著效果。
    *   **组合生成：** RL 微调后的 SD3.5-M 在 GenEval 基准测试中的准确率从 63% 提高到 95%，能够生成近乎完美的物体计数、空间关系和细粒度属性，甚至超越了最先进的 GPT-4o [18] 模型。
    *   **视觉文本渲染：** 准确率从 59% 提高到 92%，显著增强了文本生成能力。
    *   **人类偏好对齐：** 在人类偏好对齐方面也取得了实质性进展。
*   **奖励欺骗抑制：** 显著的发现是，Flow-GRPO 训练过程中很少发生奖励欺骗 (reward hacking)，这意味着奖励的增加并没有以图像质量或多样性明显下降为代价。KL 散度 (Kullback-Leibler divergence, KL) 约束被证明能够有效防止这种现象。

    总结而言，Flow-GRPO 提供了一个简单而通用的框架，用于将在线 RL 应用于流基生成模型，为解决流模型在复杂生成任务中的局限性提供了有效途径。

# 3. 预备知识与相关工作

## 3.1. 基础概念
为理解 Flow-GRPO，我们需要了解以下几个核心概念：

### 3.1.1. 流匹配 (Flow Matching)
流匹配是一种生成建模技术，它学习从一个简单分布（通常是高斯噪声）到复杂数据分布（如图像）的连续变换。其核心思想是学习一个“速度场 (velocity field)”，它定义了数据点在时间 $t \in [0, 1]$ 上从噪声向数据转化的路径。
*   <strong>Rectified Flow (整流流) [3]:</strong> Flow-GRPO 建立在 Rectified Flow 框架之上。它定义了“加噪”数据 $\mathbf{x}_t$ 为原始数据 $\mathbf{x}_0$ 和噪声样本 $\mathbf{x}_1$ 的线性插值：
    $$
    \mathbf{x}_t = (1 - t) \mathbf{x}_0 + t \mathbf{x}_1, \quad \text{对于 } t \in [0, 1]
    $$
    其中，$\mathbf{x}_0 \sim X_0$ 是来自真实数据分布的样本，$\mathbf{x}_1 \sim X_1$ 是噪声样本（通常是标准正态分布）。
*   <strong>速度场 (Velocity Field) $\mathbf{v}_\theta(\mathbf{x}_t, t)$:</strong> 流匹配模型的目标是训练一个模型（通常是 Transformer）来直接回归 (regress) 这个速度场。这个速度场代表了数据点在每个时间步长上的瞬时变化方向和速度。训练目标是最小化 Flow Matching 目标函数：
    $$
    \mathcal{L}(\theta) = \mathbb{E}_{t, \mathbf{x}_0 \sim X_0, \mathbf{x}_1 \sim X_1} \left[ \epsilon \| \mathbf{v} - \mathbf{v}_\theta(\mathbf{x}_t, t) \|^2 \right]
    $$
    其中，$\mathbf{v} = \mathbf{x}_1 - \mathbf{x}_0$ 是真实的目标速度场。通过优化这个目标函数，模型学习到的 $\mathbf{v}_\theta(\mathbf{x}_t, t)$ 能够准确地描述从 $\mathbf{x}_0$ 到 $\mathbf{x}_1$ 的路径。
*   <strong>确定性采样 (Deterministic Sampling)：</strong> 一旦训练完成，生成样本可以通过求解一个常微分方程 (Ordinary Differential Equation, ODE) 来实现：
    $$
    \mathrm{d}\mathbf{x}_t = \mathbf{v}_\theta(\mathbf{x}_t, t) \mathrm{d}t
    $$
    从初始噪声 $\mathbf{x}_0 \sim X_1$ 开始，沿着速度场进行积分，即可得到最终的生成样本。这个过程是确定性的，即给定相同的初始噪声，总会生成相同的输出。

### 3.1.2. 强化学习 (Reinforcement Learning, RL)
强化学习是一种机器学习范式，其中一个智能体 (agent) 通过与环境 (environment) 交互来学习最佳行为策略。智能体的目标是最大化其在长时间内获得的累积奖励 (cumulative reward)。
*   <strong>马尔可夫决策过程 (Markov Decision Process, MDP):</strong> RL 问题通常被建模为 MDP，它由五个元素组成 $(S, \mathcal{A}, \rho_0, P, R)$：
    *   $S$: 状态空间 (state space)，智能体可能遇到的所有可能状态的集合。
    *   $\mathcal{A}$: 动作空间 (action space)，智能体可以执行的所有可能动作的集合。
    *   $\rho_0$: 初始状态分布 (initial state distribution)，智能体开始时的状态分布。
    *   $P$: 状态转移函数 (state transition function)，描述了智能体执行某个动作后，从一个状态转移到另一个状态的概率。形式为 $P(s' | s, a)$。
    *   $R$: 奖励函数 (reward function)，描述了智能体在某个状态执行某个动作后获得的即时奖励。
*   <strong>策略 (Policy) $\pi(a|s)$:</strong> 智能体在给定状态下选择动作的规则或概率分布。在本文中，策略是流模型预测去噪样本的概率分布。
*   <strong>在线强化学习 (Online RL):</strong> 智能体在学习过程中实时与环境交互、收集数据并更新其策略。这与离线 RL（使用预先收集的数据集进行训练）形成对比。
*   <strong>策略梯度 (Policy Gradient):</strong> 一类直接优化策略函数参数 $\theta$ 的 RL 算法，通过计算策略目标函数对 $\theta$ 的梯度并执行梯度上升来最大化期望奖励。
*   **GRPO (Group Relative Policy Optimization) [16]:** 一种策略梯度方法，它通过在同一“组”内生成的样本之间进行奖励比较来估计优势函数 (advantage function)，从而实现策略更新。GRPO 的一个优点是它不需要价值网络 (value network)，因此内存效率更高，计算开销更小。

### 3.1.3. 常微分方程 (ODE) 与随机微分方程 (SDE)
*   <strong>常微分方程 (Ordinary Differential Equation, ODE):</strong> 描述一个或多个变量的函数及其导数之间的关系的方程。在流匹配中，ODE 描述了数据从噪声到数据点的确定性轨迹。例如：$\mathrm{d}\mathbf{x}_t = \mathbf{v}_t \mathrm{d}t$。
*   <strong>随机微分方程 (Stochastic Differential Equation, SDE):</strong> 描述一个或多个变量的函数及其导数，并包含一个或多个随机项（通常是维纳过程 (Wiener process) 或布朗运动 (Brownian motion)）的方程。SDE 用于模拟受随机噪声影响的系统。例如：$\mathrm{d}\mathbf{x}_t = f(\mathbf{x}_t, t) \mathrm{d}t + g(t) \mathrm{d}\mathbf{w}$，其中 $\mathrm{d}\mathbf{w}$ 是维纳过程增量。
*   <strong>Euler-Maruyama 离散化 (Euler-Maruyama Discretization):</strong> 一种数值方法，用于近似 SDE 的解。它将连续时间的 SDE 转化为离散时间步长的迭代更新规则，其中包含了随机噪声项。

## 3.2. 前人工作
*   **LLMs 中的 RL：** 在线 RL 已在 LLMs 领域取得显著成功，例如 DeepSeek-R1 [10] 和 OpenAI-o1 [11] 都利用策略梯度方法（如 PPO [20] 或 GRPO [16]）来提高 LLMs 的推理能力。本文选择 GRPO，因其在去除价值网络后的内存效率优势。
*   **扩散模型与流匹配：** 扩散模型 [21, 22, 23] 通过逐步添加高斯噪声并学习逆过程来生成数据。流匹配 [2, 3] 通过直接匹配速度场来学习连续时间归一化流，以更少的 ODE 步骤实现高效的确定性采样，使其成为当前图像 [4, 5] 和视频 [24, 25, 26, 27] 生成模型的主流选择。最近有工作 [28, 29] 将扩散模型和流模型统一在 SDE/ODE 框架下，为 Flow-GRPO 提供了理论基础。
*   **T2I 模型的对齐：** 近期 T2I 模型与人类偏好的对齐工作主要有五种方向：
    1.  <strong>可微分奖励直接微调 (Direct fine-tuning with differentiable rewards) [30, 31, 32, 33]:</strong> 直接使用奖励模型反向传播梯度。
    2.  <strong>奖励加权回归 (Reward Weighted Regression, RWR) [34, 35, 36, 37]:</strong> 根据奖励对样本进行加权，然后执行监督学习。
    3.  <strong>直接偏好优化 (Direct Preference Optimization, DPO) 及其变体 [38, 39, 14, 40, 41, 42, 43, 44, 45, 46]:</strong> 通过比较偏好对样本对进行优化。
    4.  <strong>PPO 风格策略梯度 (PPO-style policy gradients) [47, 48, 49, 50, 51, 52]:</strong> 将 RL 策略梯度应用于生成模型。
    5.  <strong>免训练对齐方法 (Training-free alignment methods) [53, 54, 55]:</strong> 无需额外训练即可实现对齐。
        这些方法已成功将 T2I 模型与人类偏好对齐，改善了美学和语义一致性。Flow-GRPO 在此基础上，将 GRPO 引入作为当前最先进 T2I 系统主干的流匹配模型。

## 3.3. 差异化分析
本文 Flow-GRPO 与现有工作的核心区别和创新点在于：
1.  <strong>首次将在线策略梯度 RL (GRPO) 集成到流匹配模型：</strong> 多数现有工作关注扩散模型或离线 RL。Flow-GRPO 填补了在线 RL 在流匹配模型应用上的空白。
2.  **独创的 ODE-to-SDE 转换策略：** 这是 Flow-GRPO 的关键创新。现有流匹配模型本质上是确定性的，无法直接进行 RL 探索。Flow-GRPO 通过将 ODE 转换为等效 SDE，在保留模型原有边际分布的同时引入了必要的随机性，从而使 RL 探索成为可能。这与并发工作 [56] 通过预测均值和方差引入随机性的方式不同，后者需要重新训练预训练模型。
3.  **高效的去噪减少策略：** 针对在线 RL 数据收集成本高的问题，Flow-GRPO 提出了在训练时减少去噪步骤、推理时保持完整步骤的策略，显著提高了训练效率，而不牺牲最终性能。
4.  **奖励欺骗的有效抑制：** 通过 KL 散度正则化，Flow-GRPO 能够有效防止奖励欺骗，确保模型在提升任务性能的同时，不损害图像质量和多样性。

# 4. 方法论

本节将详细阐述 Flow-GRPO 的方法论，包括其核心原理、如何将 GRPO 应用于流匹配模型，以及两大创新策略：ODE-to-SDE 转换和去噪减少。

## 4.1. 方法原理
Flow-GRPO 的核心思想是解决流匹配模型固有的确定性生成过程与在线强化学习所需的随机探索之间的矛盾，并提高强化学习训练的效率。通过引入 SDE 框架，Flow-GRPO 允许模型在生成过程中进行随机采样，从而为 GRPO 算法提供探索机制。同时，去噪减少策略优化了训练期间的数据收集成本，使得在线 RL 在计算密集型的图像生成任务中变得可行。

## 4.2. GRPO 在流匹配中的应用
强化学习的目标是学习一个策略 (policy) $\pi_\theta$，以最大化预期累积奖励。这通常通过优化一个正则化的目标函数来表示：
$$
\operatorname*{max}_{\theta} \mathbb{E}_{(s_0, a_0, \ldots, s_T, a_T) \sim \pi_\theta} \left[ \sum_{t=0}^T \left( R(s_t, a_t) - \beta D_{\mathrm{KL}}(\pi_\theta(\cdot \mid s_t) || \pi_{\mathrm{ref}}(\cdot \mid s_t)) \right) \right]
$$
其中，
*   $\pi_\theta$ 是由参数 $\theta$ 定义的当前策略。
*   $(s_0, a_0, \ldots, s_T, a_T)$ 是一条由策略 $\pi_\theta$ 生成的轨迹 (trajectory)，其中 $s_t$ 是状态 (state)，$a_t$ 是动作 (action)。
*   $R(s_t, a_t)$ 是在状态 $s_t$ 执行动作 $a_t$ 获得的奖励。
*   $D_{\mathrm{KL}}(\pi_\theta(\cdot \mid s_t) || \pi_{\mathrm{ref}}(\cdot \mid s_t))$ 是当前策略 $\pi_\theta$ 与参考策略 (reference policy) $\pi_{\mathrm{ref}}$ 在状态 $s_t$ 上的 KL 散度 (Kullback-Leibler divergence)，用于限制策略更新的幅度，防止策略偏离过远。
*   $\beta$ 是 KL 散度项的权重超参数。

    与 PPO (Proximal Policy Optimization) 等其他基于策略的方法不同，GRPO (Group Relative Policy Optimization) [16] 提供了一种轻量级的替代方案，它引入了组相对 (group relative) 公式来估计优势函数 (advantage function)，从而无需价值网络 (value network)。

正如第 3 节预备知识中提到的，流匹配模型的去噪过程可以被视为一个马尔可夫决策过程 (MDP)。给定一个提示 (prompt) $\mathbf{c}$，流模型 $p_\theta$ 生成一组 $G$ 张独立图像 $\{ \mathbf{x}_0^i \}_{i=1}^G$，以及相应的逆时间轨迹 (reverse-time trajectories) $\{ (\mathbf{x}_T^i, \mathbf{x}_{T-1}^i, \ldots, \mathbf{x}_0^i) \}_{i=1}^G$。

然后，第 $i$ 张图像的优势函数 (advantage function) $\hat{A}_t^i$ 通过对组级别奖励进行归一化来计算：
$$
\hat{A}_t^i = \frac{R(\mathbf{x}_0^i, \mathbf{c}) - \mathrm{mean}(\{R(\mathbf{x}_0^i, \mathbf{c})\}_{i=1}^G)}{\mathrm{std}(\{R(\mathbf{x}_0^i, \mathbf{c})\}_{i=1}^G)}
$$
其中，
*   $R(\mathbf{x}_0^i, \mathbf{c})$ 是根据生成的最终图像 $\mathbf{x}_0^i$ 和提示 $\mathbf{c}$ 计算的奖励。
*   $\mathrm{mean}(\{R(\mathbf{x}_0^i, \mathbf{c})\}_{i=1}^G)$ 是组内所有图像奖励的平均值。
*   $\mathrm{std}(\{R(\mathbf{x}_0^i, \mathbf{c})\}_{i=1}^G)$ 是组内所有图像奖励的标准差。
    这种组相对的优势估计使得模型能够学习到在当前组内表现更好的策略。

GRPO 通过最大化以下目标函数来优化策略模型：
$$
\begin{array} { r } { \mathcal { T } _ { \mathrm { F l o w - G R P O } } ( \theta ) = \mathbb { E } _ { c \sim \mathcal { C } , \{ { \boldsymbol x } ^ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \vert c ) } f ( r , \hat { A } , \theta , \varepsilon , \beta ) , } \end{array}
$$
其中，
$$
\begin{array} { c } { ^ { \mathrm { \tiny { ~ r } } } ( r , \hat { A } , \theta , \varepsilon , \beta ) = \displaystyle \frac 1 G \sum _ { i = 1 } ^ { G } \frac 1 T \sum _ { t = 0 } ^ { T - 1 } \left( \operatorname* { m i n } \left( r _ { t } ^ { i } ( \theta ) \hat { A } _ { t } ^ { i } , \ \mathrm { c l i p } \Big ( r _ { t } ^ { i } ( \theta ) , 1 - \varepsilon , 1 + \varepsilon \Big ) \hat { A } _ { t } ^ { i } \right) - \beta D _ { \mathrm { K L } } ( \pi _ { \theta } | | \pi _ { \mathrm { r e f } } ) \right) , } \\  ^ { \mathrm { \tiny { ~ r } } _ { t } ^ { i } ( \theta ) = \displaystyle \frac { p \theta ( x _ { t - 1 } ^ { i } \mid x _ { t } ^ { i } , c ) } { p _ { \theta _ { \mathrm { d d } } } \big ( { x } _ { t - 1 } ^ { i } \mid x _ { t } ^ { i } , c \big ) } . } \end{array}
$$
这里，
*   $\pi_{\theta_{\mathrm{old}}}$ 是旧策略，用于收集样本。
*   $r_t^i(\theta)$ 是策略比率 (policy ratio)，它衡量了当前策略 $p_\theta$ 相对于旧策略 $p_{\theta_{\mathrm{old}}}$ 在特定状态下采取动作的相对概率。
*   $\mathrm{clip}(\cdot)$ 函数用于将策略比率限制在一个区间 $[1-\varepsilon, 1+\varepsilon]$ 内，以避免过大的策略更新，这是 PPO 算法中常用的技术，GRPO 也借鉴了这一思想。$\varepsilon$ 是裁剪超参数。
*   $D_{\mathrm{KL}}(\pi_\theta || \pi_{\mathrm{ref}})$ 是当前策略 $\pi_\theta$ 与参考策略 $\pi_{\mathrm{ref}}$ 之间的 KL 散度，用于正则化策略更新，防止奖励欺骗和模型偏离预训练模型过远。

## 4.3. 从 ODE 到 SDE 的转换
GRPO 算法依赖于随机采样来生成多样化的轨迹，以便进行优势函数估计和探索。然而，流匹配模型通常基于确定性的 ODE 进行采样：
$$
\mathrm{d}\mathbf{x}_t = \mathbf{v}_t \mathrm{d}t \quad \text{(Eq. 6)}
$$
其中 $\mathbf{v}_t$ 是通过 Flow Matching 目标（如 Eq. 2）学习到的速度场。这种确定性方法无法满足 GRPO 策略更新的两个关键要求：
1.  **概率计算困难：** $r_t^i(\theta)$（Eq. 5）的计算需要 $p(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{c})$。在确定性动力学下，由于散度估计，这变得计算成本极高。
2.  **探索性不足：** 更重要的是，RL 依赖于探索。如第 5.3 节所述，减少探索性会降低训练效率。确定性采样（除了初始种子外没有随机性）尤其成问题。

    为了解决这个限制，Flow-GRPO 将确定性的 Flow-ODE 从 Eq. 6 转换为一个等效的 SDE，该 SDE 在所有时间步长上都匹配原始模型的边际概率密度函数 (marginal probability density function)。

以下是转换的关键步骤，详细推导在附录 A 中给出：
首先，我们考虑一个通用 SDE 的形式：
$$
\mathrm{d}\mathbf{x}_t = f_{\mathrm{SDE}}(\mathbf{x}_t, t) \mathrm{d}t + \sigma_t \mathrm{d}\mathbf{w} \quad \text{(Appendix Eq. 11)}
$$
其中 $f_{\mathrm{SDE}}(\mathbf{x}_t, t)$ 是漂移系数 (drift coefficient)，$\sigma_t$ 是扩散系数 (diffusion coefficient)，$\mathrm{d}\mathbf{w}$ 是维纳过程增量。

该 SDE 的边际概率密度 $p_t(\mathbf{x})$ 遵循 Fokker-Planck 方程 [74]：
$$
\partial_t p_t(\mathbf{x}) = - \nabla \cdot [ f_{\mathrm{SDE}}(\mathbf{x}_t, t) p_t(\mathbf{x}) ] + \frac{1}{2} \nabla^2 [ \sigma_t^2 p_t(\mathbf{x}) ] \quad \text{(Appendix Eq. 12)}
$$
同时，原始 ODE（Eq. 6）的边际概率密度演化为：
$$
\partial_t p_t(\mathbf{x}) = - \nabla \cdot [ \mathbf{v}_t(\mathbf{x}_t, t) p_t(\mathbf{x}) ] \quad \text{(Appendix Eq. 13)}
$$
为了确保 SDE 与原始 ODE 具有相同的边际分布，我们让它们的 Fokker-Planck 方程相等。经过推导（详见 Appendix A），SDE 的漂移系数 $f_{\mathrm{SDE}}$ 可以表示为：
$$
f_{\mathrm{SDE}} = \boldsymbol{v}_t(\boldsymbol{x}_t, t) + \frac{\sigma_t^2}{2} \nabla \log p_t(\boldsymbol{x}) \quad \text{(Appendix Eq. 16)}
$$
因此，前向 SDE (forward SDE) 为：
$$
\mathrm{d}\mathbf{x}_t = \left( \mathbf{v}_t(\mathbf{x}_t) + \frac{\sigma_t^2}{2} \nabla \log p_t(\mathbf{x}_t) \right) \mathrm{d}t + \sigma_t \mathrm{d}\mathbf{w} \quad \text{(Appendix Eq. 17)}
$$
根据前向 SDE 和反向 SDE 之间的关系 [75, 23]，当 $f(\mathbf{x}_t, t)$ 是前向 SDE 的漂移项，$\sigma_t$ 是扩散项时，对应的反向 SDE (reverse-time SDE) 为：
$$
\mathrm{d}\mathbf{x}_t = \left( f(\mathbf{x}_t, t) - \sigma_t^2 \nabla \log p_t(\mathbf{x}_t) \right) \mathrm{d}t + \sigma_t \mathrm{d}\overline{\mathbf{w}} \quad \text{(Appendix Eq. 19)}
$$
将 $f_{\mathrm{SDE}}$ 代入，并简化后，我们得到最终的反向 SDE 形式：
$$
\boxed{ \mathrm{d}\mathbf{x}_t = \left( \mathbf{v}_t(\mathbf{x}_t) - \frac{\sigma_t^2}{2} \nabla \log p_t(\mathbf{x}_t) \right) \mathbf{d}t + \sigma_t \mathbf{d}\mathbf{w} } \quad \text{(Appendix Eq. 21)}
$$
一旦有了分数函数 (score function) $\nabla \log p_t(\mathbf{x}_t)$，就可以直接模拟这个过程。对于 Rectified Flow，分数函数与速度场 $\mathbf{v}_t$ 隐式关联。
对于线性插值 $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$，分数函数可以推导为：
$$
\nabla \log p_t(\mathbf{x}) = - \frac{\mathbf{x}}{t} - \frac{1-t}{t} \mathbf{v}_t(\mathbf{x}) \quad \text{(Appendix Eq. 27)}
$$
将此分数函数代入反向 SDE (Appendix Eq. 21)，得到了 Rectified Flow 的 SDE 形式：
$$
\mathrm{d}\mathbf{x}_t = \left[ \mathbf{v}_t(\mathbf{x}_t) + \frac{\sigma_t^2}{2t} \left( \mathbf{x}_t + (1-t) \mathbf{v}_t(\mathbf{x}_t) \right) \right] \mathrm{d}t + \sigma_t \mathrm{d}\mathbf{w} \quad \text{(Eq. 8, Appendix Eq. 28)}
$$
最后，应用 Euler-Maruyama 离散化 (Euler-Maruyama discretization) 得到最终的更新规则：
$$
\boxed{ \mathbf{x}_{t+\Delta t} = \mathbf{x}_t + \left[ \mathbf{v}_\theta(\mathbf{x}_t, t) + \frac{\sigma_t^2}{2t} \big( \mathbf{x}_t + (1-t) \mathbf{v}_\theta(\mathbf{x}_t, t) \big) \right] \Delta t + \sigma_t \sqrt{\Delta t} \epsilon } \quad \text{(Eq. 9)}
$$
其中，
*   $\mathbf{v}_\theta(\mathbf{x}_t, t)$ 是流模型学习到的速度场。
*   $\Delta t$ 是时间步长。
*   $\epsilon \sim \mathcal{N}(0, I)$ 是注入的随机性，服从标准正态分布。
*   $\sigma_t$ 是扩散系数，用于控制生成过程中的随机性水平。本文中设置为 $\sigma_t = a \sqrt{\frac{t}{1-t}}$，其中 $a$ 是一个标量超参数，控制噪声水平。

    通过这种 SDE 转换，策略 $\pi_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{c})$ 变为一个各向同性高斯分布 (isotropic Gaussian distribution)，因此 KL 散度 $D_{\mathrm{KL}}(\pi_\theta || \pi_{\mathrm{ref}})$ 可以很容易地计算出闭合形式：
$$
D_{\mathrm{KL}}(\pi_\theta || \pi_{\mathrm{ref}}) = \frac{|| \overline{\mathbf{x}}_{t+\Delta t, \theta} - \overline{\mathbf{x}}_{t+\Delta t, \mathrm{ref}} ||^2}{2\sigma_t^2 \Delta t} = \frac{\Delta t}{2} \left( \frac{\sigma_t(1-t)}{2t} + \frac{1}{\sigma_t} \right)^2 \| \mathbf{v}_\theta(\mathbf{x}_t, t) - \mathbf{v}_{\mathrm{ref}}(\mathbf{x}_t, t) \|^2 \quad \text{(Eq. 10)}
$$
其中 $\overline{\mathbf{x}}_{t+\Delta t, \theta}$ 和 $\overline{\mathbf{x}}_{t+\Delta t, \mathrm{ref}}$ 分别是当前策略和参考策略在 $t+\Delta t$ 时刻的均值项。

## 4.4. 去噪减少 (Denoising Reduction)
生成高质量图像的流模型通常需要许多去噪步骤 (denoising steps)，这使得在线 RL 的数据收集成本很高。然而，本文发现，在线 RL 训练期间并不需要大量的时间步长。
Flow-GRPO 采用去噪减少策略：在训练期间使用显著更少的去噪步骤来生成样本（例如，训练时设定 $T=10$ 步），而在推理时保留原始的去噪步骤数量（例如，SD3.5-M 的默认 $T=40$ 步）以获得高品质样本。实验表明，这种方法可以在不牺牲测试时图像质量的情况下，显著加速训练过程，降低数据生成成本。

# 5. 实验设置

本节详细介绍 Flow-GRPO 的实验设置，包括评估任务、奖励设计、评估指标、对比基线、模型规格、超参数和计算资源。

## 5.1. 数据集
Flow-GRPO 在多个文本到图像 (T2I) 任务上进行了评估，这些任务使用了不同的数据集和奖励设计。

### 5.1.1. 组合图像生成 (Compositional Image Generation)
*   **评估基准：** GenEval [17]。GenEval 是一个评估 T2I 模型在复杂组合提示下（如对象计数、空间关系和属性绑定）性能的基准测试。它包含六个具有挑战性的组合图像生成任务。
*   **训练提示生成：** 使用官方 GenEval 脚本生成，这些脚本通过应用模板和随机组合来构建提示数据集。
*   **测试集：** 经过严格去重，确保提示仅在对象顺序上不同（例如，“一张 A 和 B 的照片”与“一张 B 和 A 的照片”）被视为相同，并从训练集中移除这些变体。
*   **奖励：** 规则 기반 (rule-based)。
    *   <strong>计数 (Counting):</strong>
        $$
        r = 1 - |N_{\mathrm{gen}} - N_{\mathrm{ref}}| / \bar{N_{\mathrm{ref}}}
        $$
        其中 $N_{\mathrm{gen}}$ 是生成图像中检测到的对象数量，$N_{\mathrm{ref}}$ 是提示中引用的对象数量，$\bar{N_{\mathrm{ref}}}$ 是参考对象数量的平均值。
    *   <strong>位置 (Position) / 颜色 (Color):</strong> 如果对象计数正确，则给予部分奖励；当预测的位置或颜色也正确时，再给予剩余的奖励。
*   **提示比例：** 根据基础模型在六个任务上的初始准确率，将训练提示比例设置为：Position : Counting : Attribute Binding : Colors : Two Objects : Single Object = 7:5:3:1:1:0。

### 5.1.2. 视觉文本渲染 (Visual Text Rendering) [8]
*   **任务：** 评估 T2I 模型准确渲染提示中指定文本的能力。文本在海报、书籍封面和表情包等图像中很常见。
*   **提示模板：** 每个提示遵循模板：“A sign that says "text"”，其中 `"text"` 是应出现在图像中的精确字符串。
*   **训练/测试数据：** 使用 GPT-4o 生成 20K 训练提示和 1K 测试提示。
*   **奖励/指标：** 使用以下公式衡量文本保真度：
    $$
    r = \mathrm{max}(1 - N_{\mathrm{e}} / N_{\mathrm{ref}}, 0)
    $$
    其中 $N_{\mathrm{e}}$ 是渲染文本与目标文本之间的最小编辑距离 (minimum edit distance)，$N_{\mathrm{ref}}$ 是提示中引号内字符的数量。此奖励也作为文本准确率的度量标准。

### 5.1.3. 人类偏好对齐 (Human Preference Alignment) [19]
*   **任务：** 旨在使 T2I 模型生成的图像与人类偏好保持一致。
*   **奖励模型：** 使用 PickScore [19] 作为奖励模型。PickScore 基于大规模的人类标注配对比较（针对相同提示生成的图像），为每对图像和提示提供一个综合得分，评估图像与提示的对齐度及其视觉质量。

## 5.2. 评估指标
除了任务特定的准确率指标外，为检测奖励欺骗 (reward hacking)（即奖励增加但图像质量或多样性下降），本文还评估了以下四种自动化图像质量指标。所有指标均在 DrawBench [1] 上计算，DrawBench 是一个包含多样化提示的综合 T2I 模型基准测试。
*   <strong>美学评分 (Aesthetic Score) [59]:</strong> 一个基于 CLIP (Contrastive Language-Image Pre-training) 的线性回归器，用于预测图像的美学分数。美学分数越高，表示图像在人类感知中越美观。
*   <strong>DeQA 评分 (DeQA Score) [60]:</strong> 一个基于多模态大语言模型 (Multimodal Large Language Model, MLLM) 的图像质量评估 (Image Quality Assessment, IQA) 模型。它量化了失真、纹理损坏和其他低级伪影对感知质量的影响。得分越高表示图像质量越好。
*   **ImageReward [32]:** 一个通用的 T2I 人类偏好奖励模型，它能够捕捉文本-图像对齐、视觉保真度和无害性等多个标准。分数越高表示图像越符合人类偏好。
*   **UnifiedReward [61]:** 最近提出的一个用于多模态理解和生成的统一奖励模型，目前在人类偏好评估排行榜上取得了最先进的性能。分数越高表示图像质量和偏好对齐度越高。

## 5.3. 对比基线
为了全面评估 Flow-GRPO 的性能，论文将其与多种对齐方法进行了比较：
*   <strong>监督微调 (Supervised Fine-Tuning, SFT):</strong> 从每个组中选择奖励最高的图像并对其进行微调。
*   <strong>奖励加权回归 (Reward Weighted Regression, RWR) / Flow-RWR [14, 76]:</strong> 在每个组中对奖励应用 softmax，然后执行奖励加权似然最大化。
*   <strong>直接偏好优化 (Direct Preference Optimization, DPO) / Flow-DPO [14, 39]:</strong> 将每个组中奖励最高的图像作为选定样本 (chosen sample)，奖励最低的作为拒绝样本 (rejected sample)，然后应用 DPO 损失进行训练。
*   **在线与离线变体：** SFT、RWR 和 DPO 都提供了在线 (online) 和离线 (offline) 变体。离线变体使用固定的预训练模型进行数据收集，而在线变体则每 40 步更新其数据收集模型。
*   **DDPO [12]:** 这是一个为扩散模型开发的 RL 训练方法，本文将其通过 ODE-to-SDE 转换适配到流匹配模型进行对比。
*   **ReFL [32]:** 一个直接微调扩散模型的策略，通过将奖励模型分数视为人类偏好损失，并反向传播梯度到随机选择的晚期时间步 $t$。
*   **ORW [35]:** 一种在线奖励加权回归方法，通过 Wasserstein-2 正则化防止策略崩溃和保持多样性。

## 5.4. 模型规格
*   <strong>基础模型 (Base Model):</strong> SD3.5-M (Stable Diffusion 3.5 Medium) [4]。
*   <strong>奖励模型 (Reward Models):</strong>
    *   美学评分 (Aesthetic Score) [59]: https://github.com/LAION-AI/aesthetic-predictor
    *   PickScore [19]: https://huggingface.co/yuvalkirstain/PickScore_v1
    *   DeQA 评分 (DeQA score) [60]: https://huggingface.co/zhiyuanyou/DeQA-Score-Mix3
    *   ImageReward [32]: https://huggingface.co/THUDM/ImageReward
    *   UnifiedReward [61]: https://huggingface.co/CodeGoat24/UnifiedReward-7b-v1.5

## 5.5. 超参数规范
除 KL 比率 $\beta$ 外，GRPO 的超参数在所有任务中均保持固定。
*   <strong>采样时间步长 (Sampling Timestep):</strong> 训练时 $T = 10$，评估时 $T = 40$。
*   <strong>组大小 (Group Size):</strong> $G = 24$。
*   <strong>噪声水平 (Noise Level):</strong> $a = 0.7$。
*   <strong>图像分辨率 (Image Resolution):</strong> 512。
*   **KL 比率 $\beta$:**
    *   GenEval 和文本渲染：$\beta = 0.04$。
    *   Pickscore：$\beta = 0.01$。
*   <strong>LoRA (Low-Rank Adaptation) 参数:</strong> $\alpha = 64$ 和 $r = 32$。

## 5.6. 计算资源规范
模型使用 24 块 NVIDIA A800 GPU 进行训练。

# 6. 实验结果与分析

本节详细分析 Flow-GRPO 在各项任务上的实验结果，并对关键设计选择进行消融研究。

## 6.1. 核心结果分析

### 6.1.1. 组合图像生成 (GenEval)
Flow-GRPO 在组合图像生成任务中表现出色。图 1 (a) 和表 1 展示了 Flow-GRPO 在训练过程中 GenEval 性能的稳步提升，并最终超越了 GPT-4o。

![Figure 1: (a) GenEval performance rises steadily throughout Flow-GRPO's training and outperforms GPT-4o. (b) Image quality metrics on DrawBench \[1\] remain essentially unchanged. (c) Human Preference Scores on DrawBench improves after training. Results show that Flow-GRPO enhances the desired capability while preserving image quality and exhibiting minimal reward-hacking.](images/1.jpg)
*该图像是图表，展示了Flow-GRPO训练过程中的GenEval性能（a）、图像质量（b）和偏好评分（c）。在GenEval表现中，SD3.5-Medium结合Flow-GRPO的得分达到0.95，明显优于其他模型。图像质量在美学和Deqa方面分别为5.39和4.07，偏好评分也显著提升，显示出该方法在能力提升的同时保持了图像质量。*

Figure 1: (a) GenEval performance rises steadily throughout Flow-GRPO's training and outperforms GPT-4o. (b) Image quality metrics on DrawBench [1] remain essentially unchanged. (c) Human Preference Scores on DrawBench improves after training. Results show that Flow-GRPO enhances the desired capability while preserving image quality and exhibiting minimal reward-hacking.

以下是原文 Table 1 的结果：

<table><tr><td>Model</td><td>Overall</td><td>Single Obj.</td><td>Two Obj.</td><td>Counting</td><td>Colors</td><td>Position</td><td>Attr. Binding</td></tr><tr><td colspan="8">Diffusion Models</td></tr><tr><td>LDM [62]</td><td>0.37</td><td>0.92</td><td>0.29</td><td>0.23</td><td>0.70</td><td>0.02</td><td>0.05</td></tr><tr><td>SD1.5 [62]</td><td>0.43</td><td>0.97</td><td>0.38</td><td>0.35</td><td>0.76</td><td>0.04</td><td>0.06</td></tr><tr><td>SD2. 62]</td><td>0.50</td><td>0.98</td><td>0.51</td><td>0.44</td><td>0.85</td><td>0.07</td><td>0.17</td></tr><tr><td>SD-XL [63]</td><td>0.55</td><td>0.98</td><td>0.74</td><td>0.39</td><td>0.85</td><td>0.15</td><td>0.23</td></tr><tr><td>DALLE-2 [64]</td><td>0.52</td><td>0.94</td><td>0.66</td><td>0.49</td><td>0.77</td><td>0.10</td><td>0.19</td></tr><tr><td>DALLE-3 [65</td><td>0.67</td><td>0.96</td><td>0.87</td><td>0.47</td><td>0.83</td><td>0.43</td><td>0.45</td></tr><tr><td colspan="8">Autoregressive Models</td></tr><tr><td>Show-o [66]</td><td>0.53</td><td>0.95</td><td>0.52</td><td>0.49</td><td>0.82</td><td>0.11</td><td>0.28</td></tr><tr><td>Emu3-Gen [67]</td><td>0.54</td><td>0.98</td><td>0.71</td><td>0.34</td><td>0.81</td><td>0.17</td><td>0.21</td></tr><tr><td>JanusFlow [68]</td><td>0.63</td><td>0.97</td><td>0.59</td><td>0.45</td><td>0.83</td><td>0.53</td><td>0.42</td></tr><tr><td>Janus-Pro-7B [69]</td><td>0.80</td><td>0.99</td><td>0.89</td><td>0.59</td><td>0.90</td><td>0.79</td><td>0.66</td></tr><tr><td>GPT-4o [18]</td><td>0.84</td><td>0.99</td><td>0.92</td><td>0.85</td><td>0.92</td><td>0.75</td><td>0.61</td></tr><tr><td colspan="8">Flow Matching Models</td></tr><tr><td>FLUX.1 Dev [5]</td><td>0.66</td><td>0.98</td><td>0.81</td><td>0.74</td><td>0.79</td><td>0.22</td><td>0.45</td></tr><tr><td>SD3.5-L [4]</td><td>0.71</td><td>0.98</td><td>0.89</td><td>0.73</td><td>0.83</td><td>0.34</td><td>0.47</td></tr><tr><td>SANA-1.5 4.8B [70]</td><td>0.81</td><td>0.99</td><td>0.93</td><td>0.86</td><td>0.84</td><td>0.59</td><td>0.65</td></tr><tr><td>SD3.5-M [4]</td><td>0.63</td><td>0.98</td><td>0.78</td><td>0.50</td><td>0.81</td><td>0.24</td><td>0.52</td></tr><tr><td>SD3.5-M+Flow-GRPO</td><td>0.95</td><td>1.00</td><td>0.99</td><td>0.95</td><td>0.92</td><td>0.99</td><td>0.86</td></tr></table>

表 1: GenEval 结果。最佳分数以蓝色标出，次佳分数以绿色标出。除 SD3.5-M 以外的模型结果来自 [7] 或其原始论文。Obj.: 对象；Attr.: 属性。

从表 1 可以看出，SD3.5-M (基础模型) 在 GenEval 上的总体准确率为 0.63。经过 Flow-GRPO 训练后，SD3.5-M+Flow-GRPO 的总体准确率显著提高到 0.95。这一性能超越了所有其他基线模型，包括 GPT-4o (0.84) 和大型流匹配模型 SD3.5-L (0.71) 及 SANA-1.5 4.8B (0.81)。在各个子任务中，Flow-GRPO 也取得了显著进步，特别是在 `Counting` (计数)、`Position` (位置) 和 `Attr. Binding` (属性绑定) 等复杂组合能力上，分别从 0.50、0.24、0.52 提高到 0.95、0.99、0.86，展现出 Flow-GRPO 在精确控制生成内容方面的强大能力。

图 3 展示了定性比较，进一步印证了 Flow-GRPO 在计数、颜色、属性绑定和位置方面的卓越性能。

![Figure 3: Qualitative Comparison on the GenEval Benchmark. Our approach demonstrates superior performance in Counting, Colors, Attribute Binding, and Position.](images/3.jpg)
*该图像是一个示意图，展示了不同模型在GenEval基准上的定性比较，包括计数、颜色、属性绑定和位置等性能。模型包括FLUX.1 Dev、GPT-4o、SD-3.5-M，以及结合Flow-GRPO的SD-3.5-M，分别展示了各自生成的图像效果。*

Figure 3: Qualitative Comparison on the GenEval Benchmark. Our approach demonstrates superior performance in Counting, Colors, Attribute Binding, and Position.

### 6.1.2. 视觉文本渲染 (OCR Accuracy)
Flow-GRPO 同样提升了视觉文本渲染任务的准确性。表 2 中的 OCR Accuracy (OCR 准确率) 指标显示，SD3.5-M 的原始准确率为 0.59，而经过 Flow-GRPO 训练后，准确率提升至 0.92。

### 6.1.3. 人类偏好对齐 (PickScore)
在人类偏好对齐任务中，Flow-GRPO 也实现了显著提升。表 2 中的 PickScore (任务指标) 显示，SD3.5-M 的原始 PickScore 为 21.72，而经过 Flow-GRPO (w/ KL) 训练后，PickScore 提升至 23.31。

以下是原文 Table 2 的结果：

<table><tr><td rowspan="2">Model</td><td colspan="3">Task Metric</td><td colspan="2">Image Quality</td><td colspan="3">Preference Score</td></tr><tr><td>GenEval</td><td>OCR Acc.</td><td>PickScore</td><td>Aesthetic</td><td>DeQA</td><td>ImgRwd</td><td>PickScore</td><td>UniRwd</td></tr><tr><td>SD3.5-M</td><td>0.63</td><td>0.59</td><td>21.72</td><td>5.39</td><td>4.07</td><td>0.87</td><td>22.34</td><td>3.33</td></tr><tr><td colspan="9">Compositional Image Generation</td></tr><tr><td>Flow-GRPO (w/o KL)</td><td>0.95</td><td></td><td></td><td>4.93</td><td>2.77</td><td>0.44</td><td>21.16</td><td>2.94</td></tr><tr><td>Flow-GRPO (w/KL)</td><td>0.95</td><td></td><td></td><td>5.25</td><td>4.01</td><td>1.03</td><td>22.37</td><td>3.51</td></tr><tr><td colspan="9">Visual Text Rendering</td></tr><tr><td>Flow-GRPO (w/o KL)</td><td></td><td>0.93</td><td></td><td>5.13</td><td>3.66</td><td>0.58</td><td>21.79</td><td>3.15</td></tr><tr><td>Flow-GRPO (w/KL)</td><td></td><td>0.92</td><td></td><td>5.32</td><td>4.06</td><td>0.95</td><td>22.44</td><td>3.42</td></tr><tr><td colspan="9">Human Preference Alignment</td></tr><tr><td>Flow-GRPO (w/o KL)</td><td></td><td></td><td>23.41</td><td>6.15</td><td>4.16</td><td>1.24</td><td>23.56</td><td>3.57</td></tr><tr><td>Flow-GRPO (w/ KL)</td><td></td><td></td><td>23.31</td><td>5.92</td><td>4.22</td><td>1.28</td><td>23.53</td><td>3.66</td></tr></table>

表 2: 在组合图像生成、视觉文本渲染和人类偏好基准上的性能，通过测试提示上的任务性能以及 DrawBench 提示上的图像质量和偏好分数进行评估。ImgRwd: ImageReward；UniRwd: UnifiedReward。

### 6.1.4. 奖励欺骗分析
Table 2 也展示了 Flow-GRPO 在保持图像质量和多样性方面做得很好，即很少发生奖励欺骗。
*   **组合图像生成任务：** Flow-GRPO (w/o KL) 在 GenEval 任务指标上达到了 0.95，与 Flow-GRPO (w/ KL) 相同。然而，其在 DrawBench 上的图像质量指标（Aesthetic 4.93，DeQA 2.77）和偏好分数（ImgRwd 0.44，PickScore 21.16，UniRwd 2.94）均显著低于 SD3.5-M 基础模型和 Flow-GRPO (w/ KL)。这表明，如果没有 KL 正则化，模型虽然提高了任务特定指标，但损害了通用图像质量和人类偏好。相比之下，Flow-GRPO (w/ KL) 在 GenEval 达到 0.95 的同时，保持了与基础模型相似甚至略高的图像质量和偏好分数，有效抑制了奖励欺骗。
*   **视觉文本渲染任务：** 结果相似，Flow-GRPO (w/o KL) 在 OCR Acc. 上达到了 0.93，但图像质量和偏好分数有下降，而 Flow-GRPO (w/ KL) 在 OCR Acc. 达到 0.92 的同时，保持了良好的图像质量和偏好分数。
*   **人类偏好对齐任务：** Flow-GRPO (w/o KL) 提高了 PickScore (任务指标) 到 23.41，但作者指出，虽然图像质量指标未显著下降，但视觉多样性出现了崩溃，生成结果趋于单一风格。Flow-GRPO (w/ KL) 则在提高 PickScore 的同时，保持了视觉多样性。图 6 定性地展示了 KL 正则化对防止质量下降和多样性衰退的有效性。

    ![Figure 6: Effect of KL Regularization. The KL penalty effectively suppresses reward hacking preventing Quality Degradation (for GenEval and OCR) and Diversity Decline (for PickScore).](images/6.jpg)
    *该图像是一个示意图，展示了KL正则化的效果。左侧的‘Quality Degradation’部分对比了不同模型生成的苹果图像质量，右侧的‘Diversity Decline’部分则展示了不同模型生成的林肯演讲图像多样性。采用KL正则化的图像在质量与多样性上均表现优异。*

Figure 6: Effect of KL Regularization. The KL penalty effectively suppresses reward hacking preventing Quality Degradation (for GenEval and OCR) and Diversity Decline (for PickScore).

### 6.1.5. Flow-GRPO 与其他对齐方法的比较
图 4 比较了 Flow-GRPO 与 SFT、Flow-DPO 及其在线变体在组合生成任务中的表现。

![Figure 4: Comparison with Other Alignment Methods on the Compositional Generation Task.](images/4.jpg)
*该图像是图表，展示了不同对齐方法在组合生成任务中的 GenEval 评分对比。随着训练提示数量的增加，Flow-GRPO 方法的 GenEval 评分显著提高，最高达到 `0.9` 以上，而其他方法的表现有所不同。*

Figure 4: Comparison with Other Alignment Methods on the Compositional Generation Task.

结果显示，Flow-GRPO 持续显著优于所有基线方法。在线 DPO 优于其离线版本，这与 [15] 的发现一致。对在线 DPO 的关键参数 $\beta$ 进行超参数搜索发现，过小的 $\beta$ 值可能导致训练崩溃。

在 Appendix C.1 中，进一步比较了 Flow-GRPO 与其他对齐方法，包括 DDPO [12] 和 ReFL [32]，以及 ORW [35]。
*   **与 DDPO 比较：** 图 8 (左侧) 显示，Flow-GRPO 的奖励增长更快，并在训练后期保持稳定，而 DDPO 的奖励增长较慢，最终在后期崩溃。这表明 Flow-GRPO 训练更稳定且持续改进。
*   **与 ReFL 比较：** 图 8 (右侧) 表明，当奖励可微分时，GRPO 的性能优于 ReFL。更重要的是，GRPO 不需要可微分奖励，可以直接使用先进的视觉语言模型 (VLM) 作为奖励提供者，这带来了更复杂、通用、且面向未来升级的奖励设计。

    ![Figure 8: Comparison of Flow-GRPO and Other Alignment Methods on the Human Preference Alignment task. Since methods like DPO use different tuned batch sizes from Flow-GRPO, we use the number of training prompts on the $\\mathbf { X }$ -axis for a fair comparison across these methods.](images/8.jpg)
    *该图像是图表，展示了Flow-GRPO与其他对齐方法在人工偏好对齐任务中的比较。图中显示了不同训练提示数量下的PickScore评估值变化，Flow-GRPO表现优越，第二张图则展示了与ReFL和DDPO在训练步骤上的比较。*

Figure 8: Comparison of Flow-GRPO and Other Alignment Methods on the Human Preference Alignment task. Since methods like DPO use different tuned batch sizes from Flow-GRPO, we use the number of training prompts on the $\mathbf { X }$ -axis for a fair comparison across these methods.

*   **与 ORW 比较：** 以下是原文 Table 5 和 Table 6 的结果：

    <table><tr><td>Method</td><td>Step 0</td><td>Step 240</td><td>Step 480</td><td>Step 720</td><td>Step 960</td></tr><tr><td>SD3.5-M + ORW</td><td>28.79</td><td>29.05</td><td>29.15</td><td>27.58</td><td>23.05</td></tr><tr><td>SD3.5-M + Flow-GRPO</td><td>28.79</td><td>29.10</td><td>29.17</td><td>29.51</td><td>29.89</td></tr></table>

表 5: 测试集上奖励分数随训练步骤的变化。

<table><tr><td>Method</td><td>CLIP Score ↑</td><td>Diversity Score ↑</td></tr><tr><td>SD3.5-M</td><td>27.99</td><td>0.96</td></tr><tr><td>SD3.5-M + ORW</td><td>28.40</td><td>0.97</td></tr><tr><td>SD3.5-M + Flow-GRPO</td><td>30.18</td><td>1.02</td></tr></table>

表 6: 不同微调方法的 CLIP 和多样性分数比较。

表 5 显示，Flow-GRPO 在训练步骤中持续提升奖励分数，而 ORW 在后期训练中出现了奖励下降。表 6 显示 Flow-GRPO 在 CLIP Score 和 Diversity Score 上均优于 ORW，进一步证明其在保持图像质量和多样性方面的优势。

## 6.2. 消融实验与参数分析

### 6.2.1. 去噪减少 (Denoising Reduction) 的影响
图 7 (a) 强调了去噪减少策略对训练加速的显著影响。

![Figure 7: Ablation studies on our critical design choices. (a) Denoising Reduction: Fewer denoising steps accelerate convergence and yield similar performance. (b) Noise Level: Moderate noise level b $a = 0 . 7$ ) maximises OCR accuracy, while too little noise hampers exploration.](images/7.jpg)
*该图像是图表，展示了去噪减少对GenEval得分和噪声水平消融对OCR准确度的影响。图(a)显示不同去噪步骤在GPU训练时间中的表现，图(b)显示不同噪声水平$a$对OCR准确度的影响，最佳噪声水平为$a = 0.7$。*

Figure 7: Ablation studies on our critical design choices. (a) Denoising Reduction: Fewer denoising steps accelerate convergence and yield similar performance. (b) Noise Level: Moderate noise level b $a = 0 . 7$ ) maximises OCR accuracy, while too little noise hampers exploration.

实验在没有 KL 约束的情况下进行，以探索不同时间步长对优化的影响。将数据收集时间步长从 40 减少到 10，在所有三个任务上实现了超过 4 倍的加速，同时不影响最终奖励。进一步减少到 5 步未能持续提高速度，有时甚至减慢了训练。因此，选择 10 步作为后续实验的设置。附录 C.2 中的图 9 展示了视觉文本渲染和人类偏好对齐任务的去噪减少消融结果。

![Figure 9: Effect of Denoising Reduction](images/9.jpg)
*该图像是图表，展示了 Flow-GRPO 在视觉文本渲染和人类偏好对齐方面的训练效果。左侧图表显示 OCR 评估准确率随着训练时间的变化，右侧图表呈现 PickScore 的变化趋势。不同步骤数的效果被标记，显示了训练效率的提升。*

Figure 9: Effect of Denoising Reduction

### 6.2.2. 噪声水平 (Noise Level) 的影响
SDE 中的 $\sigma_t$ 越高，图像多样性和探索性越强，这对 RL 训练至关重要。通过噪声水平超参数 $a$ (Eq. 9) 来控制这种探索。图 7 (b) 展示了 $a$ 对性能的影响。
*   较小的 $a$ (如 0.1) 限制了探索，减缓了奖励的提升。
*   增加 $a$ (最高到 0.7) 能够促进探索并加速奖励的获得。
*   超过 0.7 (例如从 0.7 到 1.0)，进一步增加 $a$ 不会带来额外的好处，因为探索已经足够。
*   然而，注入过多的噪声（通过进一步增加 $a$）会降低图像质量，导致奖励为零并使训练失败。这表明存在一个最佳的噪声水平。

### 6.2.3. 组大小 (Group Size) 的影响
图 5 展示了使用 PickScore 作为奖励函数时，组大小 $G$ 的影响。

![Figure 5: Ablation Studies on Different Group Size $G$ Higher group size performs better.](images/5.jpg)
*该图像是图表，展示了不同组大小 $G$ 对 Flow-GRPO 训练步骤的影响。可以看到，组大小为 24 时的评估分数最高，而组大小为 6 时的评估效果明显下降，表明更高的组大小带来了更好的性能。*

Figure 5: Ablation Studies on Different Group Size $G$ Higher group size performs better.

当组大小减小到 $G=12$ 和 $G=6$ 时，训练变得不稳定并最终崩溃。而 $G=24$ 在整个过程中保持稳定。作者观察到，较小的组大小会产生不准确的优势估计 (advantage estimates)，增加了方差，并导致训练崩溃，这与 [71, 72] 中的报道一致。这表明 GRPO 对组大小有一定的敏感性，需要足够大的组来获得稳定的优势估计。

### 6.2.4. 奖励欺骗 (Reward Hacking) 的分析
正如 6.1.4 节所讨论，KL 正则化对于防止奖励欺骗至关重要。表 2 和图 6 清楚地表明，移除 KL 约束会导致图像质量和多样性下降。虽然在人类偏好对齐任务中，移除 KL 似乎不影响图像质量指标，但会导致视觉多样性崩溃。KL 正则化能够有效阻止这种崩溃，并维持多样性。附录 C.5 中的图 12 展示了有无 KL 的学习曲线，强调了 KL 正则化并非简单地等同于提前停止 (early stopping)，而是通过限制策略与参考策略之间的偏离来保持模型质量，尽管这可能需要更长的训练时间。

![Figure 12: Learning Curves with and without KL. KL penalty slows early training yet effectively suppresses reward hacking.](images/12.jpg)
*该图像是图表，展示了在训练步骤中，使用和不使用 KL 的情况下在多个任务中的评估结果。左侧(a)为图像生成的 GenEval 分数，中间(b)为视觉文本渲染的 OCR 准确率，右侧(c)为人类偏好对齐的 PickScore。通过 KL 惩罚能有效抑制奖励黑客行为。*

Figure 12: Learning Curves with and without KL. KL penalty slows early training yet effectively suppresses reward hacking.

### 6.2.5. 泛化能力分析 (Generalization Analysis)
Flow-GRPO 在 GenEval 的未见场景中表现出强大的泛化能力（表 4）。具体来说，它能够捕捉对象数量、颜色和空间关系，并从 2-4 个对象的训练泛化到生成 5-6 个甚至 12 个对象。

以下是原文 Table 4 的结果：

<table><tr><td rowspan="2">Method</td><td colspan="7">Unseen Objects</td><td colspan="2">Unseen Counting</td></tr><tr><td>Overall</td><td>Single Obj.</td><td>Two Obj.</td><td>Counting</td><td>Colors</td><td>Position</td><td>Attr. Binding</td><td>5-6 Objects</td><td>12 Objects</td></tr><tr><td>SD3.5-M</td><td>0.64</td><td>0.96</td><td>0.73</td><td>0.53</td><td>0.87</td><td>0.26</td><td>0.47</td><td>0.13</td><td>0.02</td></tr><tr><td>SD3.5-M+Flow-GRPO</td><td>0.90</td><td>1.00</td><td>0.94</td><td>0.86</td><td>0.97</td><td>0.84</td><td>0.77</td><td>0.48</td><td>0.12</td></tr></table>

表 4: Flow-GRPO 展示了强大的泛化能力。未见对象 (Unseen Objects): 在 60 种对象类别上训练，在 20 种未见类别上评估。未见计数 (Unseen Counting): 训练生成 2、3 或 4 个对象，并在两种设置下评估：生成 5 或 6 个对象，以及生成 12 个对象。

从表 4 可以看出，在 `Unseen Objects` 任务中，Flow-GRPO 将总体准确率从 SD3.5-M 的 0.64 提高到 0.90，尤其是在 `Counting` 和 `Position` 等属性上表现出色。在 `Unseen Counting` 任务中，Flow-GRPO 也显著提高了生成 5-6 个对象和 12 个对象的准确率，分别从 0.13 和 0.02 提高到 0.48 和 0.12。

此外，表 3 显示 Flow-GRPO 在 T2I-CompBench++ [6, 73] 上也取得了显著进展。T2I-CompBench++ 是一个针对开放世界组合 T2I 生成的综合基准测试，其对象类别和关系与模型在 GenEval 风格训练数据中遇到的显著不同。

以下是原文 Table 3 的结果：

<table><tr><td>Model</td><td>Color</td><td>Shape</td><td>Texture</td><td>2D-Spatial</td><td>3D-Spatial</td><td>Numeracy</td><td>Non-Spatial</td></tr><tr><td>Janus-Pro-7B [69]</td><td>0.5145</td><td>0.3323</td><td>0.4069</td><td>0.1566</td><td>0.2753</td><td>0.4406</td><td>0.3137</td></tr><tr><td>EMU3 [67]</td><td>0.7913</td><td>0.5846</td><td>0.7422</td><td></td><td>—</td><td></td><td>—</td></tr><tr><td>FLUX.1 Dev [5]</td><td>0.7407</td><td>0.5718</td><td>0.6922</td><td>0.2863</td><td>0.3866</td><td>0.6185</td><td>0.3127</td></tr><tr><td>SD3.5-M [4]</td><td>0.7994</td><td>0.5669</td><td>0.7338</td><td>0.2850</td><td>0.3739</td><td>0.5927</td><td>0.3146</td></tr><tr><td>SD3.5-M+Flow-GRPO</td><td>0.8379</td><td>0.6130</td><td>0.7236</td><td>0.5447</td><td>0.4471</td><td>0.6752</td><td>0.3195</td></tr></table>

表 3: T2I-CompBench++ 结果。此评估使用与表 1 中相同的模型，该模型在 GenEval 生成的数据集上进行了训练。最佳分数以蓝色标出。

在 T2I-CompBench++ 上，SD3.5-M+Flow-GRPO 在 `Color` (颜色)、`Shape` (形状)、`2D-Spatial` (2D空间)、`3D-Spatial` (3D空间) 和 `Numeracy` (数字概念) 等维度上均优于 SD3.5-M 基础模型，例如 `2D-Spatial` 从 0.2850 提升到 0.5447，`Numeracy` 从 0.5927 提升到 0.6752。这进一步证明了 Flow-GRPO 训练的模型具有强大的泛化能力，能够处理其训练数据中未直接出现的新组合和关系。

### 6.2.6. 初始噪声 (Initial Noise) 的影响 (Appendix C.3)
附录 C.3 中的图 10 展示了初始噪声对训练过程的影响。用不同的随机噪声初始化每次推演 (rollout) 可以增加探索多样性。实验表明，使用不同初始噪声的 Flow-GRPO 变体在训练过程中持续获得高奖励，而使用相同初始噪声的变体则表现不佳。这强调了在 RL 训练中引入足够多样性初始状态的重要性。

![Figure 10: Effect of Initial Noise](images/10.jpg)
*该图像是一个图表，展示了在训练步骤与 PickScore 评估之间的关系，比较了使用不同初始噪声和相同初始噪声的 Flow GRPO 方法的效果。随着训练步骤的增加，两条曲线显示出明显的上升趋势。*

Figure 10: Effect of Initial Noise

### 6.2.7. FLUX.1-Dev 上的额外结果 (Appendix C.4)
附录 C.4 中的图 11 和表 7 展示了将 Flow-GRPO 应用于 FLUX.1-Dev [5] 模型的结果。即使在另一个先进的流匹配模型上，Flow-GRPO (使用 PickScore 作为奖励) 也显示出奖励曲线在训练过程中稳步上升，且没有明显的奖励欺骗。这进一步证明了 Flow-GRPO 框架的通用性和鲁棒性。

![Figure 11: Additional Results on FLUX.1-Dev](images/11.jpg)
*该图像是图表，展示了在 FLUX.1 Dev 数据集上使用 Flow-GRPO 方法的训练步骤与 PickScore 评估的关系。随着训练步骤的增加，PickScore 评估值逐渐上升，最终达到 23.43，明显高于未使用 Flow-GRPO 方法时的 21.94。*

Figure 11: Additional Results on FLUX.1-Dev

<table><tr><td>Model</td><td>Aesthetic</td><td>DeQA</td><td>ImageReward</td><td>PickScore</td><td>UnifiedReward</td></tr><tr><td>FLUX.1-Dev</td><td>5.71</td><td>4.31</td><td>0.85</td><td>22.62</td><td>3.65</td></tr><tr><td>FLUX.1-Dev + Flow-GRPO</td><td>6.02</td><td>4.24</td><td>1.32</td><td>23.97</td><td>3.81</td></tr></table>

表 7: FLUX.1-Dev 和 Flow-GRPO 微调模型的比较。

从表 7 可以看出，FLUX.1-Dev + Flow-GRPO 在 Aesthetic (美学评分)、ImageReward、PickScore 和 UnifiedReward 等偏好评分上均高于 FLUX.1-Dev 基础模型，DeQA 评分基本持平。这表明 Flow-GRPO 成功提升了 FLUX.1-Dev 的人类偏好对齐能力，同时保持了图像质量。

## 6.3. 定性结果
附录 C.6 中的图 13、14 和 15 定性地比较了 SD3.5-M 及其经过 Flow-GRPO 增强的版本（有无 KL 正则化）在 GenEval、OCR 和 PickScore 奖励下的生成结果。这些图像直观地展示了 Flow-GRPO 在提高目标能力方面的有效性，例如更准确地生成对象数量、空间关系、文本内容等。同时，带有 KL 正则化的 Flow-GRPO 版本在保持图像质量和最小化奖励欺骗方面表现更好，而没有 KL 约束的版本则可能导致图像质量和多样性的显著下降。

![Figure 13: Additional Qualitative comparison between the SD3.5-M and SD3.5-M $^ +$ Flow-GRPO trained with GenEval reward.](images/13.jpg)
*该图像是图表，展示了SD3.5-M模型与Flow-GRPO模型在生成图像方面的比较。左侧列显示SD3.5-M生成的多种对象图像，而右侧列展示Flow-GRPO生成的对应图像，包含交通信号灯、消防栓及其它场景，展现了不同处理方式对生成效果的影响。*

Figure 13: Additional Qualitative comparison between the SD3.5-M and SD3.5-M $^ +$ Flow-GRPO trained with GenEval reward.

![该图像是四个商店的外观展示，均悬挂着 "WORLD'S BEST DELI" 的招牌，展示了不同的店面风格与布局。整体上，商店的设计色调略有差异，令人印象深刻。](images/22.jpg)
*该图像是四个商店的外观展示，均悬挂着 "WORLD'S BEST DELI" 的招牌，展示了不同的店面风格与布局。整体上，商店的设计色调略有差异，令人印象深刻。*

Figure 14: Additional Qualitative comparison between the SD3.5-M and SD3.5-M $^ +$ Flow-GRPO trained with OCR reward.

![Figure 15: Additional Qualitative comparison between the SD3.5-M and SD3.5-M $^ +$ Flow-GRPO trained with PickScore reward.](images/23.jpg)
*该图像是图表，展示了SD3.5-M、Flow-GRPO以及Flow-GRPO(w/o KL)在不同生成任务中的效果对比。每一列分别展示不同风格的图像生成，包括油画、卡通和图形设计等，体现了Flow-GRPO在生成质量和风格多样性方面的提升。*

Figure 15: Additional Qualitative comparison between the SD3.5-M and SD3.5-M $^ +$ Flow-GRPO trained with PickScore reward.

这些定性结果进一步强化了定量指标的发现，即 Flow-GRPO 能够显著提高生成模型在特定任务上的表现，并且 KL 正则化在确保生成质量和多样性方面发挥着关键作用。

## 6.4. 训练样本可视化与去噪减少 (Appendix D)
附录 D 中的图 19 展示了在不同推理设置下训练样本的可视化，直观地说明了去噪减少策略。

![Figure 19: Visualization of training samples under difference inference settings.](images/29.jpg)
*该图像是一个示意图，展示了在不同推理设置下的训练样本。左上角为 ODE 采样步驟为 40，右上角为 SDE 采样步驟为 40，左下角为 SDE 采样步驟为 10，右下角为 SDE 采样步驟为 5，表现出不同的图像质量和细节。*

Figure 19: Visualization of training samples under difference inference settings.

*   图 19 (a) 和 (b) 分别展示了 40 步 ODE 采样和 40 步 SDE 采样的图像。两者在视觉上难以区分，这验证了 SDE 采样器在引入随机性时能够保持图像质量。
*   图 19 (c) 和 (d) 分别展示了 10 步和 5 步 SDE 采样的图像。这些图像显示出明显的伪影，例如颜色漂移和细节模糊，质量较低。
*   尽管低质量样本似乎可能阻碍优化，但 Flow-GRPO 发现它们仍然能够提供有用的奖励信号。由于 Flow-GRPO 依赖于相对偏好，即使样本质量较低，它也能从中提取有价值的相对排序信息。同时，更短的轨迹显著减少了训练时间。因此，采用去噪减少策略的 Flow-GRPO 在布局导向的基准测试（如 GenEval）和质量导向的指标（如 PickScore）上都能更快地收敛，而最终性能没有牺牲。

# 7. 总结与思考

## 7.1. 结论总结
本文提出了 Flow-GRPO，这是首个将在线策略梯度强化学习 (Reinforcement Learning, RL) 集成到流匹配 (Flow Matching) 模型中的方法。该方法通过以下两大创新解决了流匹配模型在 RL 训练中的核心挑战：
1.  **ODE-to-SDE 转换：** 成功地将确定性的常微分方程 (ODE) 采样过程转换为等效的随机微分方程 (SDE) 采样过程。这在保持原始模型边际分布的同时，为 RL 探索引入了必要的随机性，从而使得流匹配模型能够利用在线 RL 进行优化。
2.  **去噪减少策略：** 通过在训练期间减少去噪步骤，同时在推理时保留原始步骤，Flow-GRPO 显著提高了在线 RL 的采样效率和训练速度，而没有牺牲最终的图像生成性能。

    实证结果表明，Flow-GRPO 在多项文本到图像 (T2I) 任务中取得了显著的性能提升：
*   在组合生成任务中，将 SD3.5-M 的 GenEval 准确率从 63% 提升至 95%，超越了最先进的 GPT-4o。
*   在视觉文本渲染任务中，准确率从 59% 提高到 92%。
*   在人类偏好对齐方面也取得了实质性进展。
    值得注意的是，Flow-GRPO 通过 KL 散度 (Kullback-Leibler divergence, KL) 正则化，有效抑制了奖励欺骗 (reward hacking) 现象，确保了模型在提升任务性能的同时，图像质量和多样性没有明显下降。Flow-GRPO 提供了一个简单、通用且强大的框架，为将在线 RL 应用于流基生成模型开辟了新途径。

## 7.2. 局限性与未来工作
尽管 Flow-GRPO 取得了显著成就，作者也指出了其当前的一些局限性并提出了未来的研究方向：
1.  <strong>奖励设计 (Reward Design)：</strong> 目前的奖励设计，例如使用对象检测器或跟踪器作为规则 기반 (rule-based) 奖励，虽然可以鼓励物理真实感和时间一致性，但仍显简单。未来需要开发更先进、更精细的奖励模型，以捕捉更复杂的生成质量和用户偏好。特别是在视频生成等领域，需要更复杂的奖励信号。
2.  <strong>平衡多重奖励 (Balancing Multiple Rewards)：</strong> 视频生成等任务通常需要优化多个目标，例如真实感、平滑度、连贯性等。平衡这些相互竞争的目标仍然是一个挑战，需要仔细的调整和更智能的优化策略。
3.  <strong>可扩展性 (Scalability)：</strong> 视频生成任务比 T2I 任务在计算资源上密集得多。将 Flow-GRPO 扩展到视频生成等更大规模的应用需要更高效的数据收集和训练流水线。
4.  **防止奖励欺骗：** 尽管 KL 正则化在抑制奖励欺骗方面表现出色，但它可能导致训练时间延长，并且在某些特定提示下，奖励欺骗仍有可能发生。未来需要探索更优的方法来完全杜绝奖励欺骗，确保模型在所有情境下都能保持高质量的生成。

## 7.3. 个人启发与批判

### 7.3.1. 个人启发
这篇论文提供了一些重要的启发：
*   **弥合确定性与随机性鸿沟：** ODE-to-SDE 转换是一个非常优雅且通用的解决方案。它成功地将确定性生成模型（如流匹配）与需要探索的在线强化学习范式连接起来。这种转换思路不仅限于图像生成，未来可能应用于其他基于 ODE 的确定性生成模型，使其也能从在线 RL 中受益。
*   **效率与性能的平衡：** 去噪减少策略是 RL 训练中一个非常实用的创新。在计算密集型任务中，如何高效地收集数据始终是一个瓶颈。作者通过理论分析和实验证明，即使在训练时使用低质量的样本（较少的去噪步骤），模型仍能从相对奖励信号中有效学习，从而大幅加速训练。这种“训练时低保真，推理时高保真”的策略值得在其他生成模型训练中借鉴。
*   **KL 正则化的重要性：** 论文强调了 KL 正则化在防止奖励欺骗方面的关键作用。这提醒我们，在优化特定指标时，必须警惕模型可能出现的“钻空子”行为，而正则化是维护模型通用质量和多样性的有效手段。它表明，模型的“能力”与“品质”之间需要细致的平衡，而 KL 正则化提供了一种实现这种平衡的机制。
*   **GRPO 的潜力：** GRPO 作为一种轻量级的策略梯度方法，无需价值网络，在内存和计算效率上具有优势。这使得它非常适合与大型生成模型结合，降低了 RL 微调的门槛和资源需求。

### 7.3.2. 批判与可改进之处
*   **SDE 引入随机性的影响深入分析：** 论文虽然阐述了 SDE 如何引入随机性，但关于这种随机性对最终生成结果的细节影响可以更深入。例如，引入随机性是否会增加生成的多样性（而非简单的噪声），或者在某些情况下是否可能导致模式崩溃 (mode collapse)？SDE 的参数 $\sigma_t$ 和 $a$ 的选择对生成质量和探索效率的理论权衡可以进一步探讨。
*   **奖励模型独立性：** 论文强调了 GRPO 不需要可微分奖励，可以利用 VLMs 作为奖励模型。然而，VLM 奖励模型本身可能存在自身的偏见或局限性。如何确保奖励模型的鲁棒性和公平性，以及 VLM 奖励模型与人类真实偏好之间的一致性，是一个持续的挑战。未来的工作可以探索更自洽、更少依赖外部预训练奖励模型的方法。
*   **奖励欺骗的深层机制：** 尽管 KL 正则化有效，但论文提到它可能导致训练时间延长，且某些提示下仍可能发生奖励欺骗。这表明奖励欺骗是一个复杂现象，需要更深层次的理解和更鲁棒的解决方案，而不仅仅是正则化。例如，探索更复杂的奖励函数设计，或者在 RL 优化目标中直接整合多样性指标。
*   **泛化能力的边界：** 尽管论文展示了 Flow-GRPO 在未见对象和计数上的良好泛化能力，但对于更抽象、更复杂的组合概念，其泛化能力边界如何？例如，在概念组合或领域外泛化方面的表现如何？这需要更具挑战性的基准测试来评估。
*   **低质量样本的学习机制：** 论文指出，在去噪减少策略下，低质量的训练样本（如 5 步 SDE 采样）仍能提供有用的奖励信号。这背后的具体学习机制值得进一步研究。RL 智能体是如何从模糊或有伪影的图像中有效提取“相对偏好”信息的？这对于理解 RL 在噪声数据环境中的学习能力具有理论意义。
*   **计算成本的进一步优化：** 尽管去噪减少策略提高了效率，但 24 块 A800 GPU 的配置仍显示出训练大型生成模型所需的巨大资源。未来的工作可以探索更轻量级的模型架构、更高效的训练算法或更优化的硬件利用，以降低训练门槛。