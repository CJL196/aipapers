# DIffusionNfT：在线扩散增强与前向过程

Kaiwen Zheng1,2,* Huayu Chen $^{1, 2, *}$ Haotian $\mathbf{Y e}^{2, 3}$ Haoxiang Wang2 Qinsheng Zhang2 Kai Jiang1 Hang $\mathbf{S u}^{1}$ Stefano Ermon³ Jun Zhu1,† Ming-Yu Liu2 *同等贡献 通讯作者 1清华大学 2英伟达 3斯坦福大学 https://research.nvidia.com/labs/dir/DiffusionNFT

# 摘要

在线强化学习（RL）在训练后语言模型中占据核心地位，但其在扩展到扩散模型时仍面临挑战，主要是由于无法处理的似然问题。近期的工作通过离散化反向采样过程来实现 GRPO 风格的训练，但继承了一些固有的缺陷，包括求解器限制、正向-反向不一致性以及与无分类器引导（CFG）的复杂集成。我们提出了扩散负感知微调（DiffusionNFT），这是一种新的在线 RL 范式，通过流匹配直接在正向过程中优化扩散模型。DiffusionNFT 对比正向和负向生成，定义一个隐式的策略改进方向，自然地将增强信号融入监督学习目标中。这一格式允许使用任意黑箱求解器进行训练，消除了对似然估计的需求，并且只需干净的图像，而不需要为策略优化采样轨迹。在正面对比中，DiffusionNFT 的效率提高了多达 $2 5 \times$，同时不需要 CFG。例如，在 1k 步骤内，DiffusionNFT 将 GenEval 分数从 0.24 提升至 0.98，而 FlowGRPO 在超过 $5 \mathrm { k }$ 步骤和附加 CFG 的情况下只达到了 0.95。通过利用多种奖励模型，DiffusionNFT 显著提高了 SD3.5-Medium 在每个测试基准上的性能。

![](images/1.jpg)  

Figure 1: Performance of DiffusionNFT. (a) Head-to-head comparison with FlowGRPO on the GenEval task. (b) By employing multiple reward models, DiffusionNFT significantly boosts the performance of SD3.5-Medium in every benchmark tested, while being fully CFG-free.

# 1 引言

在线强化学习在大语言模型的后训练中发挥了关键作用，推动了最近在大语言模型的对齐和推理能力方面的进展。然而，在视觉生成中复制扩散模型相似的成功并不简单。

![](images/2.jpg)  

Figure 2: Comparison between Forward-Process RL (NFT) and Reverse-Process RL (GRPO). NFT allows using any solvers and does not require storing the whole sampling trajectory for optimization.

策略梯度算法假设模型的似然性是可以精确计算的。这一假设适用于自回归模型，但在扩散模型中则本质上被违背，在扩散模型中，似然性只能通过昂贵的概率常微分方程（ODE）或随机微分方程（SDE）变分界限来近似（Song et al., 2021）。近期的研究通过离散化反向采样过程来规避这一障碍，将扩散生成重新定义为一个多步骤决策问题（Black et al., 2023）。这使得相邻步骤之间的转换变为可处理的高斯分布，从而可以直接将现有的强化学习算法（如GRPO）应用于扩散领域（Xue et al., 2025；Liu et al., 2025）。尽管取得了一些有希望的进展，我们认为基于GRPO的扩散强化学习仍面临根本性限制：（1）前向不一致。仅关注反向采样过程打破了对前向扩散过程的遵循，存在模型退化为级联高斯的风险。（2）求解器限制。数据收集过程依赖于一阶SDE采样器，限制了ODE或高阶求解器的全面利用，这些求解器通常用于流模型且对生成效率有利。（3）复杂的无分类指导（CFG）集成。扩散模型严重依赖无分类器指导（CFG）（Ho & Salimans, 2022），这需要同时训练条件和无条件模型。目前的RL实践通常在后训练中结合CFG，导致复杂且低效的双模型优化方案。我们旨在解开数据收集，消除求解器限制，并与标准监督预训练保持一致。由于扩散策略仅包含单一的前向（噪声）过程但有多个反向（去噪）过程（例如，不同的采样器），一个自然的问题是：扩散强化是否可以在前向过程中而非反向过程中进行？本文提出了一种名为扩散负向感知微调（Diffusion Negative-aware FineTuning，DiffusionNFT）的新在线强化学习范式。DiffusionNFT并不基于传统的策略梯度框架，而是通过流匹配目标直接在前向扩散过程中执行策略优化。直观地说，它定义了在“正”样本和“负”样本（通过奖励信号分割）上学习的两个隐式策略之间的对比改进方向，并向正策略进行优化，而不修改采样过程。前向过程强化学习的公式提供了若干实际好处（图2）。首先，DiffusionNFT允许使用任意黑盒求解器进行数据收集，而不是依赖一阶SDE采样器。其次，它消除了存储整个采样轨迹的需要，仅需干净图像用于策略优化。第三，它与标准扩散训练完全兼容，仅需对现有代码进行最小修改。最后，它是一个原生的离策略算法，自然允许去耦的训练和采样策略，而无需重要性采样。我们通过对多个奖励模型进行后续训练SD3.5-Medium（Esser et al., 2024）来评估DiffusionNFT。整个训练过程故意在无CFG设置下进行。尽管这导致初始化性能显著降低，但我们发现DiffusionNFT在域内和域外奖励上显著提高性能，迅速超过CFG和GRPO基线。在单奖励设置下，我们还与FlowGRPO进行了一对一比较。在测试的四个任务中，DiffusionNFT的效率始终表现为$3 \times$至$25 \times$，且取得了更好的最终得分。例如，它在1k步内将GenEval得分从0.24提高到0.98，而FlowGRPO在超过$5 \text{ k }$步和额外CFG使用的情况下仅达到0.95。DiffusionNFT是对传统策略梯度方法的直接强化学习替代方案，将负向感知微调（NFT）范式（Chen et al., 2025c）引入扩散领域。在监督学习基础上，我们相信这一范式为跨各种模态实现通用、统一的原生离策略强化学习提供了有效路径。

# 2 背景

# 2.1 扩散与流动模型

扩散模型（Ho等，2020；Song等，2020b）通过根据前向过程逐渐扰动干净数据 $\pmb { x } _ { 0 } \sim \pi _ { 0 } = p _ { \mathrm { d a t a } }$ 来学习连续数据分布。然后，通过学习逆转该过程可以生成数据。前向噪声过程具有封闭形式的转换核 $\pi _ { t | 0 } ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) = \mathcal { N } ( \alpha _ { t } \pmb { x } _ { 0 } , \sigma _ { t } ^ { 2 } \mathbf { I } )$，具有特定的噪声调度 $\alpha _ { t } , \sigma _ { t }$，使得重参数化成为可能。

$$
\begin{array} { r } { \pmb { x } _ { t } = \alpha _ { t } \pmb { x } _ { 0 } + \sigma _ { t } \pmb { \epsilon } , \pmb { \epsilon } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) . } \end{array}
$$

学习扩散模型的一种方法是采用速度参数化 ${ \pmb v } _ { \theta } ( { \pmb x } _ { t } , t )$（Zheng 等，2023b），该参数用于预测轨迹的切线，通过最小化目标速度 $\textbf { { v } }$ 来训练，该目标速度由调度的时间导数定义为 ${ \pmb v } = \dot { \alpha } _ { t } { \pmb x } _ { 0 } + \dot { \sigma } _ { t } { \pmb \epsilon }$，在记号 $\bar { \dot { f } } _ { t } : = \mathrm { d } f _ { t } / \dot { \mathrm { d } { t } }$ 下，$w ( t )$ 是某种加权函数。反向采样通常遵循 h $\begin{array} { r } { \frac { \mathrm { d } \pmb { x } _ { t } } { \mathrm { d } t } = \pmb { v } _ { \theta } ( \pmb { x } _ { t } , t ) } \end{array}$ 使用 ${ \pmb v } _ { \theta }$。这种形式被称为流匹配（Lipman 等，2022），其中简单的欧拉离散化作为有效的常微分方程求解器，相当于 DDIM（Song 等，2020a）。

$$
\begin{array} { r } { \mathbb { E } _ { t , { \boldsymbol { x } } _ { 0 } \sim \pi _ { 0 } , { \boldsymbol { \epsilon } } \sim { \mathcal { N } } ( \mathbf { 0 } , \mathbf { I } ) } [ w ( t ) \| { \boldsymbol { v } } _ { \theta } ( { \boldsymbol { x } } _ { t } , t ) - { \boldsymbol { v } } \| _ { 2 } ^ { 2 } ] , } \end{array}
$$

修正流（Liu 等, 2022）可以视为上述讨论的扩散模型的特例，其中 $\alpha _ { t } = 1 - t , \sigma _ { t } = t$，这将速度目标简化为 ${ \pmb v } = { \pmb \epsilon } - { \pmb x } _ { 0 }$。

# 2.2 扩散模型的策略梯度算法

为了将策略梯度算法如PPO（Schulman等，2017）或GRPO（Shao等，2024）应用于扩散模型，最近的研究（Black等，2023；Fan等，2023；Liu等，2025；Xue等，2025）将扩散采样公式化为一个多步马尔可夫决策过程（MDP）。这可以通过离散化扩散模型的反向采样过程来实现。虽然流模型自然通过常微分方程（ODE）允许简单高效的采样，但缺乏随机性限制了GRPO的应用。FlowGRPO（Liu等，2025）通过在速度参数化 ${ \pmb v } _ { \theta }$ 下使用随机微分方程（SDE）形式（Song等，2020b）来解决这个问题（见附录B.1）：

$$
\mathrm { d } x _ { t } = \Big [ v _ { \theta } ( x _ { t } , t ) + \frac { g _ { t } ^ { 2 } } { 2 t } \big ( x _ { t } + ( 1 - t ) v _ { \theta } ( x _ { t } , t ) \big ) \Big ] \mathrm { d } t + g _ { t } \mathrm { d } w _ { t }
$$

其中 $\begin{array}{r} g_{t} = a \sqrt{\frac{t}{1 - t}} \end{array}$

$$
\pi _ { \boldsymbol { \theta } } ( \mathbf { x } _ { t - \Delta t } \mid \mathbf { x } _ { t } ) = \mathcal { N } \Big ( \mathbf { x } _ { t } + \Big [ v _ { \boldsymbol { \theta } } ( \mathbf { x } _ { t } , t ) + \frac { g _ { t } ^ { 2 } } { 2 t } ( \mathbf { x } _ { t } + ( 1 - t ) v _ { \boldsymbol { \theta } } ( \mathbf { x } _ { t } , t ) ) \Big ] \Delta t , \ g _ { t } ^ { 2 } \Delta t \mathbf { I } \Big ) .
$$

这使相邻步骤之间的转移核成为可处理的高斯分布，从而能够直接应用现有的策略梯度算法，例如 GRPO。

# 3 通过负相关微调的扩散强化学习

# 3.1 问题设置

在线强化学习。考虑一个预训练的扩散策略 $\pi ^ { \mathrm { o l d } }$ 和提示数据集 $\{ c \}$。在每次迭代中，我们为提示 $^ c$ 采样 $K$ 张图像 $\pmb { x } _ { 0 } ^ { 1 : \hat { K } }$，其最优性概率 $r \in [ 0 , 1 ]$ 表示为 $r ( \pmb { x } _ { 0 } , \pmb { c } ) : = p ( \mathbf { o } = 1 | \pmb { x } _ { 0 } , \pmb { c } )$ （Levine, 2018）。该最优性作为从连续值奖励到二元划分的桥梁。收集到的数据可以随机分为两个虚拟子集。一幅图像 $\scriptstyle { \pmb x } _ { 0 }$ 将有概率 $r$ 落入正数据集 $\mathcal { D } ^ { + }$，否则则落入负数据集 $\mathcal { D } ^ { - }$。在无限样本的情况下，这两个子集的潜在分布分别为

$$
\pi ^ { + } ( x _ { 0 } | c ) : = \pi ^ { \smash { \mathrm { o l d } } } ( x _ { 0 } | \mathbf { o } = 1 , c ) = \frac { p ( \mathbf { o } = 1 | x _ { 0 } , c ) \pi ^ { \mathrm { o l d } } ( x _ { 0 } | c ) } { p _ { \pi ^ { \mathrm { o l d } } } ( \mathbf { o } = 1 | c ) } = \frac { r ( x _ { 0 } , c ) } { p _ { \pi ^ { \mathrm { o l d } } } ( \mathbf { o } = 1 | c ) } \pi ^ { \mathrm { o l d } } ( x _ { 0 } | c ) 
$$

$$
\pi ^ { - } ( \boldsymbol { x } _ { 0 } | \boldsymbol { c } ) : = \pi ^ { \mathrm { o l d } } ( \boldsymbol { x } _ { 0 } | \mathbf { o } = 0 , \boldsymbol { c } ) = \frac { p ( \mathbf { o } = 0 | \boldsymbol { x } _ { 0 } , \boldsymbol { c } ) \pi ^ { \mathrm { o l d } } ( \boldsymbol { x } _ { 0 } | \boldsymbol { c } ) } { p _ { \pi ^ { \mathrm { o l d } } } ( \mathbf { o } = 0 | \boldsymbol { c } ) } = \frac { 1 - r ( \boldsymbol { x } _ { 0 } , \boldsymbol { c } ) } { 1 - p _ { \pi ^ { \mathrm { o l d } } } ( \mathbf { o } = 1 | \boldsymbol { c } ) } \pi ^ { \mathrm { o l d } } ( \boldsymbol { x } _ { 0 } | \boldsymbol { c } )
$$

强化学习要求在每次迭代中进行策略改进。优化后的策略 $\pi ^ { * }$ 满足

$$
\mathbb { E } _ { \pi ^ { * } ( \cdot | c ) } r ( { \boldsymbol x } _ { 0 } , c ) > \mathbb { E } _ { \pi ^ { \mathrm { o l d } } ( \cdot | c ) } r ( { \boldsymbol x } _ { 0 } , c ) \qquad ( \mathrm { d e n o t e d ~ a s } \quad \pi ^ { * } \succ \pi ^ { \mathrm { o l d } } )
$$

基于正数据的策略改进。可以容易地证明 $\pi ^ { + } \succ \pi ^ { \mathrm { o l d } } \succ \pi ^ { - }$ 始终成立，因此 $\pi ^ { \mathrm { o l d } }$ 的直接改进可以是 $\pi ^ { * } = \pi ^ { + }$。为了实现这一点，之前的工作（Lee et al., 2023）仅在 $\mathcal { D } ^ { + }$ 上进行扩散训练，称为拒绝微调（Rejection FineTuning，RFT）。尽管简单，RFT 不能有效利用 $\mathcal { D } ^ { - }$ 中的负数据（Chen et al., 2025c）。强化指导。我们认为，负反馈对策略改进至关重要，尤其是在扩散模型中。我们不是将 $\pi ^ { + }$ 视为优化点，而是利用负数据和正数据来推导改进方向 $\Delta \in \mathbb { R } ^ { n }$。训练目标定义为，其中 $\textbf { { v } }$ 是扩散模型的速度预测器，$\beta$ 是超参数。这个定义形式上类似于扩散指导，如无分类器指导（Classifier-Free Guidance，CFG）（Ho & Salimans, 2022）。我们将 $\Delta ( \pmb { x } _ { t } , \pmb { c } , t ) \in \mathbb { R } ^ { n }$ 称为强化指导，$\textstyle { \frac { 1 } { \beta } } \in \mathbb { R }$ 表示指导强度。

$$
{ \boldsymbol v } ^ { * } ( { \boldsymbol x } _ { t } , c , t ) : = { \boldsymbol v } ^ { \mathrm { o l d } } ( { \boldsymbol x } _ { t } , c , t ) + \frac { 1 } { \beta } \Delta ( { \boldsymbol x } _ { t } , c , t ) .
$$

在3.2节中，我们讨论了两个挑战：1. 什么样的$\Delta$形式能够实现策略改进？2. 如何利用收集的数据集$\mathcal { D } ^ { + }$和$\mathcal { D } ^ { - }$直接优化${ \pmb v } _ { \theta } { \pmb v } ^ { * }$？

# 3.2 负向意识扩散强化学习与前向过程

在公式(3)中，$\Delta$ 表示改进策略与原始策略之间的分布差异。为此，我们首先研究 $\pi ^ { + } \stackrel { \cdot } { \succ } \pi ^ { \mathrm { o l d } } \succ \pi ^ { - }$ 之间的分布差异。定理 3.1（改进方向）。考虑策略三元组 $\pi ^ { + } , \pi ^ { - }$ 和 $\pi ^ { o l d }$ 的扩散模型 $v ^ { + }$ 、$v ^ { - }$ 和 $v ^ { o l d }$，这些模型之间的方向差异是成比例的：

$$
\begin{array} { r l r } & { } & { \Delta : = [ 1 - \alpha ( { \pmb x } _ { t } ) ] [ { \pmb v } ^ { o l d } ( { \pmb x } _ { t } , { \pmb c } , t ) - { \pmb v } ^ { - } ( { \pmb x } _ { t } , { \pmb c } , t ) ] } \\ & { } & { = \alpha ( { \pmb x } _ { t } ) [ { \pmb v } ^ { + } ( { \pmb x } _ { t } , { \pmb c } , t ) - { \pmb v } ^ { o l d } ( { \pmb x } _ { t } , { \pmb c } , t ) ] . } \end{array}
$$

其中 $0 \leq \alpha ( { \pmb x } _ { t } ) \leq 1$ 是一个标量系数：

$$
\alpha ( { \pmb x } _ { t } ) : = \frac { \pi _ { t } ^ { + } ( { \pmb x } _ { t } | { \pmb c } ) } { \pi _ { t } ^ { o l d } ( { \pmb x } _ { t } | { \pmb c } ) } \mathbb { E } _ { \pi ^ { o l d } ( { \pmb x } _ { 0 } | { \pmb c } ) } r ( { \pmb x } _ { 0 } , { \pmb c } )
$$

方程 (4) 表示了一个理想的指导方向 $\Delta$，用于改进 $v^{\mathrm{o l d}}$。在适当的指导强度下，可以确保策略的改进。例如，在方程 (3) 中设定 $\beta = \alpha(\pmb{x}_t)$，我们有 ${\pmb v}^{*}({\pmb x}_t, {\pmb c}, t) = {\pmb v}^{\mathrm{o l d}}({\pmb x}_t, {\pmb c}, t) + \alpha(\pmb{x}_t)\Delta(\pmb{x}_t, {\pmb c}, t) = v^{+}(\pmb{x}_t, {\pmb c}, t)$，因此 $\pi^{*} = \pi^{+} \succ \pi^{\mathrm{o l d}}$ 成立。图 3 显示了改进方向 $\Delta$ 的示意图。

![](images/3.jpg)  

Figure 3: Improvement Direction.

结合公式 (3) 和 (4)，我们现在引入一个训练目标，直接优化 ${ \pmb v } _ { \theta }$ 朝向 $v ^ { * }$：

![](images/4.jpg)  

Figure 4: DiffusionNFT jointly optimizes two dual diffusion objectives, on both positive $( r = 1 )$ and negative $( r = 0$ branches. Rather than training two independent models ${ \boldsymbol { v } } _ { \theta } ^ { + }$ and ${ \boldsymbol v } _ { \boldsymbol { \theta } _ { } } ^ { - }$ , it adopts a implicit parameerization technique that directlyoptimizes single target poliy ${ \pmb v } _ { \theta }$ .

定理 3.2（策略优化）。考虑训练目标：

$$
\mathcal { L } ( \theta ) = \mathbb { E } _ { c , \pi ^ { o l d } ( { \pmb x } _ { 0 } \mid c ) , t } r \| { \pmb v } _ { \theta } ^ { + } ( { \pmb x } _ { t } , { \pmb c } , t ) - { \pmb v } \| _ { 2 } ^ { 2 } + ( 1 - r ) \| { \pmb v } _ { \theta } ^ { - } ( { \pmb x } _ { t } , { \pmb c } , t ) - { \pmb v } \| _ { 2 } ^ { 2 } ,
$$

其中 $\begin{array} { r } { \pmb { v } _ { \theta } ^ { + } ( \pmb { x } _ { t } , \pmb { c } , t ) : = ( 1 - \beta ) \pmb { v } ^ { o l d } ( \pmb { x } _ { t } , \pmb { c } , t ) + \beta \pmb { v } _ { \theta } ( \pmb { x } _ { t } , \pmb { c } , t ) , \quad ( } \end{array}$ 隐式正政策）和 $\begin{array} { r } { \pmb { v } _ { \theta } ^ { - } ( \pmb { x } _ { t } , \pmb { c } , t ) : = ( 1 + \beta ) \pmb { v } ^ { o l d } ( \pmb { x } _ { t } , \pmb { c } , t ) - \beta \pmb { v } _ { \theta } ( \pmb { x } _ { t } , \pmb { c } , t ) . } \end{array}$ （隐式负政策）。在数据和模型容量无限的情况下，方程 (5) 的最优解满足

$$
{ v } _ { \theta ^ { * } } ( { x } _ { t } , c , t ) = { v } ^ { o l d } ( { x } _ { t } , c , t ) + \frac { 2 } { \beta } \Delta ( { x } _ { t } , c , t ) .
$$

定理 3.2 提出了一个新的离线策略强化学习范式（图 4）。它不是应用策略梯度，而是采用监督学习（SL）目标，同时在在线负数据 $\mathcal { D } ^ { - }$ 上进行训练。这使得该算法具有高度的灵活性，能够与现有的 SL 方法兼容。我们称我们的方法为扩散负样本感知微调（DiffusionNFT），突出了其负样本感知的 SL 特性以及与语言模型中并行算法 NFT 的概念相似性（Chen et al., 2025c）。下面，我们讨论 DiffusionNFT 的几个显著优势。

1. 前向一致性。与根据逆扩散过程构建强化学习的策略梯度方法（例如，FlowGRPO）不同，DiffusionNFT 在前向过程中定义了典型的扩散损失。这保持了我们所称的前向一致性——扩散模型的基础概率密度遵循 Fokker-Planck 方程（Øksendal, 2003；Song et al., 2020b），确保学习到的模型对应于有效的前向过程（即，$\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ 正确地通过联合分布 $\pi _ { \boldsymbol { \theta } } ( \pmb { x } _ { t } , \pmb { x } _ { 0 } ) = \pi _ { \boldsymbol { \theta } } ( \pmb { x } _ { 0 } ) \pi _ { t | 0 } ( \pmb { x } _ { t } | \pmb { x } _ { 0 } )$ 与 $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ 相关联）。 2. 解算器灵活性。DiffusionNFT 完全解耦了策略训练和数据采样。这使得在采样过程中可以充分利用任何黑箱解算器，而不必依赖于一阶 SDE 采样器。它还消除了在数据收集过程中存储整个采样轨迹的需要，仅需干净的图像及其相关奖励供训练使用。 3. 隐式指导集成。直观地说，DiffusionNFT 定义了一个指导方向 $\Delta$ 并将该指导应用于旧策略 $v ^ { \mathrm { o l d } }$（公式（6））。然而，与其学习一个单独的指导模型 $\Delta _ { \theta }$ 并采用引导采样，不如采用隐式参数化技术，使强化指导能够直接集成到学习到的策略中。这一技术受到了最近无指导训练进展的启发（Chen et al., 2025a），使我们能够在单个策略模型上持续进行强化学习，这对在线强化至关重要。 4. 无似然性公式。以往的扩散强化学习方法在根本上受限于对似然性近似的依赖。无论是通过变分界限近似边际数据似然并应用 Jensen 不等式以降低损失计算成本（Wallace et al., 2024），还是将逆过程离散化以估计序列似然（Black et al., 2023），它们不可避免地在扩散后训练中引入系统估计偏差。相比之下，DiffusionNFT 本质上是无似然性的，避免了这种妥协。

# 3.3 实际实现

我们在算法1中提供了DiffusionNFT的伪代码。接下来，我们将详细阐述关键设计选择。

# 算法 1 扩散负面意识微调 (DiffusionNFT)

需求：重定义 ${ \pmb v } ^ { \mathrm { r e f } }$，原始奖励函数 $r ^ { \mathrm { r a w } } ( \cdot ) \in \mathbb { R }$，提示数据集 $\{ c \}$。初始化：数据收集策略 ${ \pmb v } ^ { \mathrm { o l d } } { \pmb v } ^ { \mathrm { r e f } }$，训练策略 ${ \pmb v } _ { \theta } { \pmb v } ^ { \mathrm { r e f } }$，数据缓冲区 $\mathcal { D } \emptyset$ 1: 对于每次迭代 $i$ 进行 2: 对于每个采样提示 $^ c$ 进行 // 推演步骤，数据收集 3: $K$ $\pmb { x } _ { 0 } ^ { 1 : K }$ $\{ r ^ { \mathrm { r a w } } \} ^ { 1 : K }$ 4: 归一化组内原始奖励：$r ^ { \mathrm { n o m } } : = r ^ { \mathrm { r a w } } - \mathsf { m e a n } ( \{ r ^ { \mathrm { r a w } } \} ^ { 1 : K } )$。 5: 定义最优概率 $r = 0 . 5 + 0 . 5 * \mathrm { c } \mathrm { 1 } \mathrm { i } \mathrm { p } \{ r ^ { \mathrm { n o r m } } / Z _ { c } , - 1 , 1 \}$。 6: $\mathcal { D } \{ c , \ x _ { 0 } ^ { 1 : K } , r ^ { \bar { 1 } : K } \in [ 0 , 1 ] \}$ 7: 结束循环 8: 对于每个小批量 $\{ c , \pmb { x } _ { 0 } , r \} \in \mathcal { D }$ 进行 // 梯度步骤，策略优化 9: 正向扩散过程：$\begin{array} { r } { \pmb { x } _ { t } = \alpha _ { t } \pmb { x } _ { 0 } + \sigma _ { t } \pmb { \epsilon } ; \pmb { v } = \dot { \alpha } _ { t } \pmb { x } _ { 0 } + \dot { \sigma } _ { t } \pmb { \epsilon } , } \end{array}$。 10: 隐式正向速度：$\boldsymbol { v } _ { \theta } ^ { + } ( \boldsymbol { x } _ { t } , \boldsymbol { c } , t ) : = ( 1 - \beta ) \boldsymbol { v } ^ { \mathrm { o l d } } ( \boldsymbol { x } _ { t } , \boldsymbol { c } , t ) + \beta \boldsymbol { v } _ { \theta } ( \boldsymbol { x } _ { t } , \boldsymbol { c } , t )$。 11: 隐式负向速度：$\boldsymbol { v } _ { \theta } ^ { - } ( \boldsymbol { x } _ { t } , \boldsymbol { c } , t ) : = ( 1 + \beta ) \boldsymbol { v } ^ { \mathrm { o l d } } ( \boldsymbol { x } _ { t } , \boldsymbol { c } , t ) - \beta \boldsymbol { v } _ { \theta } ( \boldsymbol { x } _ { t } , \boldsymbol { c } , t )$。 12: . $\theta \theta - \lambda \nabla _ { \theta } [ r \| v _ { \theta } ^ { + } ( x _ { t } , c , t ) - v \| _ { 2 } ^ { 2 } + ( 1 - r ) \| v _ { \theta } ^ { - } ( x _ { t } , c , t ) - v \| _ { 2 } ^ { 2 } ]$。(方程 (5)) 13: 结束循环 14: 更新数据收集策略 $\theta ^ { \mathrm { o l d } } \eta _ { i } \theta ^ { \mathrm { o l d } } + ( 1 - \eta _ { i } ) \theta$，并清空缓冲区 $\mathcal { D } \emptyset$ // 在线更新 15: 结束循环 输出：$v _ { \theta }$ 最优奖励。在大多数视觉强化学习设置中，奖励表现为无约束的连续标量，而不是二元最优信号。受现有 GRPO 实践的启发（Shao et al., 2024; Liu et al., 2025; Xue et al., 2025），我们首先将原始奖励 $r ^ { \mathrm { { r a w } } }$ 转换为 $r \in [ 0 , 1 ]$，表示最优概率：

$$
r ( \pmb { x } _ { 0 } , \pmb { c } ) : = \frac { 1 } { 2 } + \frac { 1 } { 2 } \mathrm { c } \mathrm { 1 } \mathrm { i } \mathrm { p } \left[ \frac { r ^ { \mathrm { r a w } } ( \pmb { x } _ { 0 } , \pmb { c } ) - \mathbb { E } _ { \pi ^ { \mathrm { o l d } } ( \cdot \vert \pmb { c } ) } r ^ { \mathrm { r a w } } ( \pmb { x } _ { 0 } , \pmb { c } ) } { Z _ { c } } , - 1 , 1 \right] .
$$

$Z _ { c } > 0$ 是某个归一化因子，可以采用全局奖励的形式。我们对于每个提示 $^ c$ 采样 $K$ 张图像，通过 $\mathbb { E } _ { \pi ^ { \mathrm { o l d } } ( \cdot | \boldsymbol { c } ) } r ^ { \mathrm { r a w } } ( \pmb { x } _ { 0 } , \pmb { c } )$ 来估计每个提示的结果。采样策略的软更新。DiffusionNFT 的离策略特性将采样策略 $\bar { \pi } ^ { \mathrm { o l d } }$ 与训练策略 $\pi _ { \theta }$ 解耦。这消除了每次迭代后对 "硬" 更新 $\pi ^ { \mathrm { o l d } } \pi ^ { \theta }$ 的需求。相反，我们利用这一特性采用 "软" 指数移动平均更新：

$$
\theta ^ { \mathrm { o l d } }  \eta _ { i } \theta ^ { \mathrm { o l d } } + ( 1 - \eta _ { i } ) \theta
$$

其中 $i$ 是迭代次数，参数 $\eta$ 控制学习速度和稳定性之间的权衡。严格的在线策略 $( \eta = 0 )$ 导致初期进展迅速，但容易发生严重的不稳定，导致灾难性崩溃。相反，几乎离线的方法 $\langle \eta \to 1 \rangle$ 具有稳健的稳定性，但收敛速度过于缓慢，难以实现（见图 8）。

自适应损失权重。典型的扩散损失包括一个时间依赖的权重 $w ( t )$ (公式 (1))。我们采用自适应权重方案，而不是手动调节。速度预测器 ${ \pmb v } _ { \theta }$ 可以等效转换为 $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ 预测器，记作 $\scriptstyle { \mathbf { { x } } } \theta$（例如，在校正流调度下，${ \mathbf { } } x _ { \theta } = x _ { t } - t { \mathbf { } } v _ { \theta }$）。我们通过自归一化的 $\scriptstyle { \mathbf { { \mathit { x } } } _ { 0 }$ 回归形式来替代权重，这一方法受到扩散蒸馏方法 DMD (Yin et al., 2024) 的启发：

$$
w ( t ) \| v _ { \theta } ( x _ { t } , c , t ) - v \| _ { 2 } ^ { 2 }  \frac { \| x _ { \theta } ( x _ { t } , c , t ) - x _ { 0 } \| _ { 2 } ^ { 2 } } { \mathrm { s g } ( \mathrm { m e a n } ( \mathrm { a b s } ( x _ { \theta } ( x _ { t } , c , t ) - x _ { 0 } ) ) ) }
$$

其中 $S \mathbb { { g } }$ 是停止梯度操作符。我们发现这通常能加快训练速度（图 9）。无类优化。无分类器引导（CFG）（Ho & Salimans, 2022）是一种在推理时提高生成质量的默认技术，但它会使后期训练变得复杂并降低效率。从概念上看，我们将 CFG 理解为强化引导的一种离线形式（方程（4）），其中条件模型和无条件模型分别对应正向信号和负向信号。基于这一理解，我们在算法设计中抛弃 CFG，策略仅通过条件模型初始化。尽管这种初始化看似不理想，但我们观察到性能迅速上升并迅速超过 CFG 基线（图 1）。这表明 CFG 的功能可以通过 RL 后期训练有效学习或替代，与最近的研究相呼应，后期训练实现了无需 CFG 的强大性能（Chen et al., 2025b;a; Zheng et al., 2025）。

Table 1: Evaluation Results. Gray-colored: In-domain reward. † Evaluated on official checkpoints. ‡Evaluated under $1 0 2 4 \times 1 0 2 4$ resolution. Bold: best; Underline: second best.   

<table><tr><td rowspan="2">Model</td><td rowspan="2">#Iter</td><td colspan="2">Rule-Based</td><td colspan="6">Model-Based</td></tr><tr><td>GenEval</td><td>OCR</td><td>PickScore</td><td>ClipScore</td><td>HPSv2.1</td><td>Aesthetic</td><td>ImgRwd</td><td>UniRwd</td></tr><tr><td>SD-XL‡</td><td></td><td>0.55</td><td>0.14</td><td>22.42</td><td>0.287</td><td>0.280</td><td>5.60</td><td>0.76</td><td>2.93</td></tr><tr><td>SD3.5-L‡</td><td></td><td>0.71</td><td>0.68</td><td>22.91</td><td>0.289</td><td>0.288</td><td>5.50</td><td>0.96</td><td>3.25</td></tr><tr><td>FLUX.1-Dev</td><td></td><td>0.66</td><td>0.59</td><td>22.84</td><td>0.295</td><td>0.274</td><td>5.71</td><td>0.96</td><td>3.27</td></tr><tr><td>SD3.5-M (w/o CFG) + CFG</td><td></td><td>0.24</td><td>0.12</td><td>20.51</td><td>0.237</td><td>0.204</td><td>5.13</td><td>-0.58</td><td>2.02</td></tr><tr><td></td><td>—</td><td>0.63</td><td>0.59</td><td>22.34</td><td>0.285</td><td>0.279</td><td>5.36</td><td>0.85</td><td>3.03</td></tr><tr><td>+ FlowGRPO†</td><td>&gt;5k</td><td>0.95</td><td>0.66</td><td>22.51</td><td>0.293</td><td>0.274</td><td>5.32</td><td>1.06</td><td>3.18</td></tr><tr><td></td><td>2k</td><td>0.66</td><td>0.92</td><td>22.41</td><td>0.290</td><td>0.280</td><td>5.32</td><td>0.95</td><td>3.15</td></tr><tr><td></td><td>4k</td><td>0.54</td><td>0.68</td><td>23.50</td><td>0.280</td><td>0.316</td><td>5.90</td><td>1.29</td><td>3.37</td></tr><tr><td>+ Ours</td><td>1.7k</td><td>0.94</td><td>0.91</td><td>23.80</td><td>0.293</td><td>0.331</td><td>6.01</td><td>1.49</td><td>3.49</td></tr></table>

# 4 实验

我们从三个角度展示了DiffusionNFT的潜力：（1）多奖励联合训练以实现强大的CFG自由性能，（2）与FlowGRPO在单奖励上的正面对比，以及（3）对关键设计选择的消融研究。

# 4.1 实验设置

我们的实验基于 SD3 . 5-Medium (Esser 等, 2024)，分辨率为 $5 1 2 \times 5 1 2$，大多数设置与 FlowGRPO (Liu 等, 2025) 对齐。奖励模型。(1) 基于规则的奖励，包括用于组合图像生成的 GenEva1 (Ghosh 等, 2023) 和用于视觉文本呈现的 OCR，其中部分奖励分配策略遵循 FlowGRPO。(2) 基于模型的奖励，包括 PickScore (Kirstain 等, 2023)、ClipScore (Hessel 等, 2021)、HPSv2.1 (Wu 等, 2023)、美学 (Schuhmann, 2022)、ImageReward (Xu 等, 2023) 和统一奖励 (Wang 等, 2025)，这些方法用于评估图像质量、图像-文本对齐和人类偏好。提示数据集。对于 GenEval 和 OCR，我们使用 FlowGRPO 的相应训练集和测试集。对于其他奖励，我们在 Pick-a-Pic (Kirstain 等, 2023) 上进行训练，并在 DrawBench (Saharia 等, 2022) 上进行评估。训练与评估。我们使用 LoRA 进行微调，其中 $\alpha = 6 4$，$r = 3 2$。每个周期由 48 组组成，组大小为 $G = 2 4$。我们使用 10 次推演采样步骤进行对比实验和消融研究，在多奖励训练中使用 40 次步骤以获得最佳视觉质量。评估通过 40 次步骤的一阶 ODE 采样器进行。附录 C 中提供了更多详细信息。

# 4.2 多奖励联合训练

我们首先评估 DiffusionNFT 在全面提升基础模型方面的有效性。从无 CFG 的 SD3.5-M（25 亿参数）开始，我们共同优化五个奖励：GenEva1、OCR、PickScore、ClipScore 和 HP Sv2。由于这些奖励基于不同的提示，我们首先训练基于模型的奖励的 Pick-a-Pic，以增强对齐和人类偏好，随后使用基于规则的奖励（GenEval、OCR）。在美学、图像奖励和统一奖励上进行域外评估。如表1所示，我们最终的无 CFG 模型不仅在域内和域外指标上超越了 CFG，并且在单一奖励拟合的情况下与 FlowGRPO 相匹配，同时还超越了基于 CFG 的更大模型，如 $\mathrm { S D } 3 . 5 \mathrm { - L }$（80 亿参数）和 FLUX.1-Dev（120 亿参数）（Labs, 2024）。图5中的定性比较展示了我们方法的卓越视觉质量。

# 4.3 正面比较

我们对单次训练奖励进行了与 FlowGRPO 的正面比较。如图 1(a) 和图 6 所示，相比之下，我们的方法在墙时效率上提高了 $3 \times$ 到 $25 \times$。

![](images/5.jpg)  
FlowGRPO

DiffusionNFT在仅仅${ \sim } 1 \mathrm { k }$次迭代中实现了0.98的GenEval评分。这表明，无需条件前缀的模型能够在我们的框架下快速适应特定的奖励环境。

![](images/6.jpg)  

Figure 5: Qualitative Comparison. The prompts are taken from GenEva1, OCR and DrawBench respectively, where we compare the corresponding FlowGRPO model with our model.   

Figure 6: Head-to-head comparison between DiffusionNFT with FlowGRPO on single rewards.

# 4.4 消融研究

![](images/7.jpg)

![](images/8.jpg)  

Figure 7: Different diffusion samplers for data collection.   

Figure 8: Soft-update strategies.

我们分析了核心设计选择的影响：负损失。负感知组件在DiffusionNFT中至关重要。如果没有对${ \boldsymbol { v } } _ { \boldsymbol { \theta } } ^ { - }$的负策略损失，我们发现在线训练期间奖励几乎瞬间崩溃，突显了负信号在扩散强化学习中的重要作用。这一现象与在大语言模型中的观察有所不同，在那里，RFT仍然是一个强有力的基线（Xiong等，2025；Chen等，2025c）。

![](images/9.jpg)  

Figure 9: Different time-dependent weighting strategies.

![](images/10.jpg)  

Figure 10: Choices of strength $\beta$ .

扩散采样器。在DiffusionNFT中，在线样本用于奖励评估和训练数据，因此质量至关重要。图7显示，ODE采样器的表现优于SDE采样器，尤其在对噪声敏感的PickScore上。二阶ODE在GenEval上略优于一阶ODE，而在PickScore上表现相当。自适应加权。当流匹配损失在较大$t$时被给予更高的权重时，我们发现稳定性有所提高，而逆向策略（例如，$w(t) = 1 - t$）则会导致崩溃（图9）。我们的自适应调度始终与启发式选择相匹配或超出。软更新。我们在图8中比较了不同的$\eta_{i}$调度用于软更新。完全基于策略的$b\gamma_{i} = 0$加快了早期进展，但使得训练不稳定，而过于偏离策略的$(\eta = 0.9)$则减缓了收敛。我们发现从一个较小的$\eta$开始，逐渐增加到较大值，在收敛速度和训练稳定性之间达成了有效的平衡。引导强度。如图10所示，引导参数$\beta$也在稳定性和收敛速度之间进行权衡。我们发现$\beta$接近1时表现稳定，并在实践中选择$\beta$为1或0.1（以便快速提高奖励）。

# 5 相关工作

强化学习算法从离散自回归（AR）模型过渡到连续扩散模型面临着一个核心挑战：扩散模型计算精确模型似然的内在困难（Song et al., 2021），而这些似然对于强化学习至关重要（Chen et al., 2023；Liu et al., 2025）。为了解决这个挑战，现有的努力包括：

无似然方法：（1）奖励反向传播（Xu 等，2023；Prabhudesai 等，2023；Clark 等，2023；Prabhudesai 等，2024）被证明非常有效，但仅限于可微分奖励，并且由于长去噪链展开时的内存成本和梯度爆炸，只能调整低噪声时间步。（2）奖励加权回归（RWR）（Lee 等，2023）是一种离线微调方法，但缺乏负策略目标来惩罚低奖励生成。（3）策略引导。这包括能量引导（Janner 等，2022；Lu 等，2023）和CFG风格引导（Frans 等，2025；Jin 等，2025）。这些方法都需要结合多个模型进行引导采样，从而使在线优化变得复杂。（4）基于得分的强化学习。这些方法试图直接在得分而非似然场上执行强化学习（Zhu 等，2025）。

基于似然的方法：（1）Diffusion-DPO（Wallace et al.，2024；Yang et al.，2024；Liang et al.，2024；Yuan et al.，2024；Li et al.，2025a）将DPO适配于扩散用于成对的人类偏好数据，但与自回归方法相比，需要额外的似然和损失近似；DDO（Zheng et al.，2025）使用高质量数据集作为正信号，自生成样本作为负信号，从而避免成对数据的需求，在视觉生成中实现了最先进的无条件生成图像FID，同时仍然依赖于扩散案例的似然近似。（2）策略梯度方法，从PPO风格开始（Black et al.，2023；Fan et al.，2023），逐步分解轨迹似然，而不考虑前向一致性。近期的GRPO扩展（Liu et al.，2025；Xue et al.，2025）证明在扩散强化学习中有效且可扩展，但它们将训练损失与随机微分方程采样器耦合，面临效率瓶颈。MixGRPO（Li et al.，2025b）通过混合随机微分方程和常微分方程提高效率，但耦合问题和前向不一致性问题仍然存在。

# 6 结论

我们引入了扩散负向意识微调（Diffusion Negative-aware FineTuning, DiffusionNFT），这是一种针对扩散模型的在线强化学习新范式，直接作用于前向过程。通过将策略改进表述为正向生成和负向生成之间的对比，DiffusionNFT 将强化信号无缝整合到标准的扩散目标中，消除了对似然估计和基于随机微分方程的反向过程的依赖。实证结果表明，DiffusionNFT 展示了强大而高效的奖励优化能力，其效率比 Flow-GRPO 高出 $2.5 \times$，同时生成的单一模型在多种领域内外的奖励上均优于 CFG 基线。我们相信这项工作代表了将监督学习与强化学习在扩散中的统一的一步，并强调前向过程作为可扩展、高效及理论上有根据的扩散强化学习的有前景基础。关于大型语言模型（LLMs）的使用，我们仅将大型语言模型作为写作助手，用于语言润色和提升表达清晰度。LLMs 并未参与研究构思、方法设计、实验执行或结果分析。所有科学贡献和实质性写作均由作者完成。

# 致谢

我们感谢 Cheng Lu、Hanzi Mao、Zekun Hao、Tao Yang、Zhanhao Liang、Shuhuai Ren、Tenglong Ao、Xintao Wang、Haoqi Fan、Jiajun Liang、Yuji Wang 和 Hongzhou Zhu 的宝贵讨论。

# REFERENCES

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.

Huayu Chen, Cheng Lu, Chengyang Ying, Hang Su, and Jun Zhu. Offline reinforcement learning via high-fidelity generative behavior modeling. In The Eleventh International Conference on Learning Representations, 2023.

Huayu Chen, Kai Jiang, Kaiwen Zheng, Jianfei Chen, Hang Su, and Jun Zhu. Visual generation without guidance. Forty-second international conference on machine learning, 2025a.

Huayu Chen, Hang Su, Peize Sun, and Jun Zhu. Toward guidance-free ar visual generation via condition contrastive alignment. In ICLR, 2025b.

Huayu Chen, Kaiwen Zheng, Qinsheng Zhang, Ganqu Cui, Yin Cui, Haotian Ye, Tsung-Yi Lin, Ming-Yu Liu, Jun Zhu, and Haoxiang Wang. Bridging supervised learning and reinforcement learning in math reasoning. arXiv preprint arXiv:2505.18116, 2025c.

Kevin Clark, Paul Vicol, Kevin Swersky, and David J Fleet. Directly fine-tuning diffusion models on differentiable rewards. arXiv preprint arXiv:2309.17400, 2023.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models. Advances in Neural Information Processing Systems, 36:7985879885, 2023.

Kevin Frans, Seohong Park, Pieter Abbeel, and Sergey Levine. Diffusion guidance is a controllable policy improvement operator. arXiv preprint arXiv:2505.23458, 2025.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36: 5213252152, 2023.

Martin Gonzalez, Nelson Fernandez Pinto, Thuy Tran, Hatem Hajri, Nader Masmoudi, et al. Seeds: Exponential sde solvers for fast high-quality sampling from diffusion models. Advances in Neural Information Processing Systems, 36:6806168120, 2023.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718, 2021.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:68406851, 2020.

Marlis Hochbruck and Alexander Ostermann. Exponential integrators. Acta Numerica, 19:209286, 2010.

Chin-Wei Huang, Jae Hyun Lim, and Aaron C Courville. A variational perspective on diffusionbased generative models and score matching. Advances in Neural Information Processing Systems, 34:2286322876, 2021.

Michael Janner, Yilun Du, Joshua Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis. In International Conference on Machine Learning, 2022.

Luozhijie Jin, Zijie Qiu, Jie Liu, Zijie Diao, Lifeng Qiao, Ning Ding, Alex Lamb, and Xipeng Qiu. Inference-time alignment control for diffusion models with reinforcement learning guidance. arXiv preprint arXiv:2508.21016, 2025.

Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. Advances in neural information processing systems, 34:2169621707, 2021.

Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Picka-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:3665236663, 2023.

Black Forest Labs. Flux. https://github.com/black-forest-labs/flux,2024.

Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, and Shixiang Shane Gu. Aligning text-to-image models using human feedback. arXiv preprint arXiv:2302.12192, 2023.

Sergey Levine. Reinforcement learning and control as probabilistic inference: Tutorial and review. arXiv preprint arXiv:1805.00909, 2018.

Binxu Li, Minkai Xu, Meihua Dang, and Stefano Ermon. Divergence minimization preference optimization for diffusion model alignment. arXiv preprint arXiv:2507.07510, 2025a.

Junzhe Li, Yutao Cui, Tao Huang, Yinping Ma, Chun Fan, Miles Yang, and Zhao Zhong. Mixgrpo: Unlocking flow-based grpo efficiency with mixed ode-sde. arXiv preprint arXiv:2507.21802, 2025b.

Zhanhao Liang, Yuhui Yuan, Shuyang Gu, Bohan Chen, Tiankai Hang, Ji Li, and Liang Zheng. Step-aware preference optimization: Aligning preference with denoising performance at each step. arXiv preprint arXiv:2406.04314, 2(5):7, 2024.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in neural information processing systems, 35:57755787, 2022a.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver $^ { + + }$ : Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022b.

Cheng Lu, Huayu Chen, Jianfei Chen, Hang Su, Chongxuan Li, and Jun Zhu. Contrastive energy prediction for exact energy-guided diffusion sampling in offline reinforcement learning. arXiv preprint arXiv:2304.12824, 2023.

Bernt Øksendal. Stochastic differential equations. In Stochastic differential equations: an introduction with applications, pp. 3850. Springer, 2003.

Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, and Katerina Fragkiadaki. Aligning text-toimage diffusion models with reward backpropagation. 2023.

Mihir Prabhudesai, Russell Mendonca, Zheyang Qin, Katerina Fragkiadaki, and Deepak Pathak. Video diffusion alignment via reward gradients. arXiv preprint arXiv:2407.08737, 2024.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:3647936494, 2022.

Christoph Schuhmann. Laion-aesthetics. https://laion.ai/blog/ laion-aesthetics/,2022.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020a.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020b.

Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of scorebased diffusion models. In Advances in Neural Information Processing Systems, volume 34, pp. 14151428, 2021.

Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 82288238, 2024.

Feng Wang and Zihao Yu. Coefficients-preserving sampling for reinforcement learning with flow matching. arXiv preprint arXiv:2509.05952, 2025.

Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025.

Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score: Better aligning text-to-image models with human preference. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 20962105, 2023.

Wei Xiong, Jiarui Yao, Yuhui Xu, Bo Pang, Lei Wang, Doyen Sahoo, Junnan Li, Nan Jiang, Tong Zhang, Caiming Xiong, et al. A minimalist approach to llm reasoning: from rejection sampling to reinforce. arXiv preprint arXiv:2504.11343, 2025.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:1590315935, 2023.

Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.

Kai Yang, Jian Tao, Jiafei Lyu, Chunjiang Ge, Jiaxin Chen, Weihan Shen, Xiaolong Zhu, and Xiu Li. Using human feedback to fine-tune diffusion models without any reward model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 89418951, 2024.

Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 66136623, 2024.

Huizhuo Yuan, Zixiang Chen, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning of diffusion models for text-to-image generation. Advances in Neural Information Processing Systems, 37: 7336673398, 2024.

Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. arXiv preprint arXiv:2204.13902, 2022.

Kaiwen Zheng, Cheng Lu, Jianfei Chen, and Jun Zhu. Dpm-solver-v3: Improved diffusion ode solver with empirical model statistics. In Thirty-seventh Conference on Neural Information Processing Systems, 2023a.

Kaiwen Zheng, Cheng Lu, Jianfei Chen, and Jun Zhu. Improved techniques for maximum likelihood estimation for diffusion odes. In International Conference on Machine Learning, pp. 42363 42389. PMLR, 2023b.

Kaiwen Zheng, Guande He, Jianfei Chen, Fan Bao, and Jun Zhu. Diffusion bridge implicit models. arXiv preprint arXiv:2405.15885, 2024.

Kaiwen Zheng, Yongxin Chen, Huayu Chen, Guande He, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Direct discriminative optimization: Your likelihood-based visual generative model is secretly a gan discriminator. In ICML, 2025.

Huaisheng Zhu, Teng Xiao, and Vasant G Honavar. Dspo: Direct score preference optimization for diffusion model alignment. In The Thirteenth International Conference on Learning Representations, 2025.

# A Proof Of Theorems

Lemma A.1 (Distribution Split). Consider the distribution triplet $\pi ^ { + }$ , $\pi ^ { - }$ , and $\pi ^ { o l d }$ , as defined in Section 3.1:

$$
\begin{array} { l } { { \pi ^ { + } ( { \pmb x } _ { 0 } | c ) : = \pi ^ { o l d } ( { \pmb x } _ { 0 } | { \bf o } = 1 , c ) = \displaystyle \frac { p ( { \bf o } = 1 | { \pmb x } _ { 0 } , c ) \pi ^ { o l d } ( { \pmb x } _ { 0 } | c ) } { p _ { \pi ^ { o l d } } ( { \bf o } = 1 | c ) } = \displaystyle \frac { r ( { \pmb x } _ { 0 } , c ) } { p _ { \pi ^ { o l d } } ( { \bf o } = 1 | c ) } \pi ^ { o l d } ( { \pmb x } _ { 0 } | c ) \quad \mathrm { ~ o ~ r ~ } \quad ( 1 \leq \alpha \leq \alpha ) } } \\ { { \pi ^ { - } ( { \pmb x } _ { 0 } | c ) : = \pi ^ { o l d } ( { \pmb x } _ { 0 } | { \bf o } = 0 , c ) = \displaystyle \frac { p ( { \bf o } = 0 | { \pmb x } _ { 0 } , c ) \pi ^ { o l d } ( { \pmb x } _ { 0 } | c ) } { p _ { \pi ^ { o l d } } ( { \bf o } = 0 | c ) } = \displaystyle \frac { 1 - r ( { \pmb x } _ { 0 } , c ) } { 1 - p _ { \pi ^ { o l d } } ( { \bf o } = 1 | c ) } \pi ^ { o l d } ( { \pmb x } _ { 0 } | c ) \quad \mathrm { ~ o ~ r ~ } \quad ( 1 \leq \alpha \leq \alpha ) } } \end{array}
$$

$\pi ^ { o l d } ( \pmb { x } _ { 0 } | \pmb { c } )$ is as a linear combination between its positive slt $\pi ^ { + } ( { \pmb x } _ { 0 } | { \pmb c } )$ and negative split $\pi ^ { - } \left( \pmb { x } _ { 0 } | \pmb { c } \right)$ .

$$
\pi ^ { o l d } ( x _ { 0 } | c ) = p _ { \pi ^ { o l d } } ( \mathbf { o } = 1 | c ) \pi ^ { + } ( x _ { 0 } | c ) + [ 1 - p _ { \pi ^ { o l d } } ( \mathbf { o } = 1 | c ) ] \pi ^ { - } ( x _ { 0 } | c )
$$

Proof. The result follows directly from Eq.(7) and Eq.(8).

Lemma A.2 (Posterior Split). The diffusion posteriors for distribution triplet $\pi ^ { + } , \pi ^ { - }$ , and $\pi ^ { o l d }$ satisfy:

$$
\begin{array} { r l } & { { \pi ^ { o l d } } ( x _ { 0 } | x _ { t } , c ) = \alpha ( { x _ { t } } ) \pi ^ { + } ( { x _ { 0 } } | x _ { t } , c ) + [ 1 - \alpha ( { x _ { t } } ) ] \pi ^ { - } ( { x _ { 0 } } | { x _ { t } } , c ) } \\ & { \qquad w h e r e \qquad \alpha ( { x _ { t } } ) : = \frac { \pi _ { t } ^ { + } ( { x _ { t } } | c ) } { \pi _ { t } ^ { o l d } ( { x _ { t } } | c ) } \mathbb { E } _ { \pi ^ { o l d } ( { x _ { 0 } } | c ) } r ( { x _ { 0 } } , c ) } \end{array}
$$

Proof. Leveraging Bayes' Rule:

$$
\pi ^ { \mathrm { o l d } } ( { \pmb x } _ { 0 } | { \pmb c } ) = \frac { \pi _ { t } ^ { \mathrm { o l d } } ( { \pmb x } _ { t } | { \pmb c } ) \pi _ { 0 | t } ^ { \mathrm { o l d } } ( { \pmb x } _ { 0 } | { \pmb x } _ { t } , { \pmb c } ) } { \pi ( { \pmb x } _ { t } | { \pmb x } _ { 0 } ) }
$$

Replacing all distributions in Eq. (9) (Lemma A.1) we get

$$
\begin{array} { r l } & { \frac { \pi _ { t } ^ { \mathrm { o d d } } ( x _ { t } | c ) \pi _ { 0 | t } ^ { \mathrm { o d d } } ( x _ { 0 } | x _ { t } , c ) } { \pi ( x _ { t } | x _ { 0 } ) } = p _ { \pi ^ { \mathrm { s i o } } } ( \mathbf { o } = 1 | c ) \frac { \pi _ { t } ^ { + } ( x _ { t } | c ) \pi _ { 0 | t } ^ { - } ( x _ { 0 } | x _ { t } , c ) } { \pi ( x _ { t } | x _ { 0 } ) } } \\ & { \qquad + \left[ 1 - p _ { \pi ^ { \mathrm { s i d } } } ( \mathbf { o } = 1 | c ) \right] \frac { \pi _ { t } ^ { - } ( x _ { t } | c ) \pi _ { 0 | t } ^ { - } ( x _ { 0 } | x _ { t } , c ) } { \pi ( x _ { t } | x _ { 0 } ) } } \\ & { \qquad \Rightarrow \pi _ { 0 | t } ^ { \mathrm { o d d } } ( x _ { 0 } | x _ { t } , c ) = p _ { \pi ^ { \mathrm { s i d } } } ( \mathbf { o } = 1 | c ) \frac { \pi _ { t } ^ { + } ( x _ { t } | c ) } { \pi _ { t } ^ { \mathrm { o d d } } ( x _ { t } | c ) } \pi _ { 0 | t } ^ { + } ( x _ { 0 } | x _ { t } , c ) } \\ & { \qquad + \left[ 1 - p _ { \pi ^ { \mathrm { s i d } } } ( \mathbf { o } = 1 | c ) \right] \frac { \pi _ { t } ^ { - } ( x _ { t } | c ) } { \pi _ { t } ^ { \mathrm { o d d } } ( x _ { t } | c ) } \pi _ { 0 | t } ^ { - } ( x _ { 0 } | x _ { t } , c ) } \end{array}
$$

Diffuse both sides of Eq. (9), we have

$$
\begin{array} { r l } & { \pi _ { t } ^ { \mathrm { o l d } } ( { \pmb x } _ { t } | { \pmb c } ) = p _ { \pi ^ { \mathrm { o l d } } } ( { \bf o } = 1 | { \pmb c } ) \pi _ { t } ^ { + } ( { \pmb x } _ { t } | { \pmb c } ) + [ 1 - p _ { \pi ^ { \mathrm { o l d } } } ( { \bf o } = 1 | { \pmb c } ) ] \pi _ { t } ^ { - } ( { \pmb x } _ { t } | { \pmb c } ) } \\ & { \qquad p _ { \pi ^ { \mathrm { o l d } } } ( { \bf o } = 1 | { \pmb c } ) \frac { \pi _ { t } ^ { + } ( { \pmb x } _ { t } | { \pmb c } ) } { \pi _ { t } ^ { \mathrm { o l d } } ( { \pmb x } _ { t } | { \pmb c } ) } + [ 1 - p _ { \pi ^ { \mathrm { o l d } } } ( { \bf o } = 1 | { \pmb c } ) ] \frac { \pi _ { t } ^ { - } ( { \pmb x } _ { t } | { \pmb c } ) } { \pi _ { t } ^ { \mathrm { o l d } } ( { \pmb x } _ { t } | { \pmb c } ) } = 1 } \end{array}
$$

Note that

$$
p _ { \pi ^ { \mathrm { o l d } } } ( \mathbf { o } = 1 | \pmb { c } ) = \mathbb { E } _ { \pi ^ { \mathrm { o l d } } ( \pmb { x } _ { 0 } | \pmb { c } ) } r ( \pmb { x } _ { 0 } , \pmb { c } )
$$

We have

$$
\pi _ { 0 | t } ^ { \mathrm { o l d } } ( \pmb { x } _ { 0 } | \pmb { x } _ { t } , \pmb { c } ) = \alpha ( \pmb { x } _ { t } ) \pi _ { 0 | t } ^ { + } ( \pmb { x } _ { 0 } | \pmb { x } _ { t } , \pmb { c } ) + [ 1 - \alpha ( \pmb { x } _ { t } ) ] \pi _ { 0 | t } ^ { - } ( \pmb { x } _ { 0 } | \pmb { x } _ { t } , \pmb { c } )
$$

Theorem A.3 (Improvement Direction). Consider diffusion models ${ \mathbf { } } v ^ { + } , v ^ { - }$ , and $v ^ { o l d }$ for the distribution triplet $\pi ^ { + } , \pi ^ { - }$ , and $\pi ^ { o l d }$ The directional differences between these models are parallel:

$$
\begin{array} { r l } & { \Delta : = [ 1 - \alpha ( { \pmb x } _ { t } ) ] \left[ v ^ { o l d } ( { \pmb x } _ { t } , { \pmb c } , t ) - v ^ { - } ( { \pmb x } _ { t } , { \pmb c } , t ) \right] \quad ( R e i n f o r c e m e n t G u i d a n c e ) } \\ & { \quad = \quad \alpha ( { \pmb x } _ { t } ) \qquad [ v ^ { + } ( { \pmb x } _ { t } , { \pmb c } , t ) - v ^ { o l d } ( { \pmb x } _ { t } , { \pmb c } , t ) ] . } \end{array}
$$

where $0 \leq \alpha ( { \pmb x } _ { t } ) \leq 1$ is a scalar coefficient:

$$
\alpha ( { \pmb x } _ { t } ) : = \frac { \pi _ { t } ^ { + } ( { \pmb x } _ { t } | { \pmb c } ) } { \pi _ { t } ^ { o l d } ( { \pmb x } _ { t } | { \pmb c } ) } \mathbb { E } _ { \pi ^ { o l d } ( { \pmb x } _ { 0 } | { \pmb c } ) } r ( { \pmb x } _ { 0 } , { \pmb c } )
$$

Proof. According to the relationship between the optimal velocity predictor and the posterior mean of $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ (i.e., the optimal $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ predictor) (Zheng et al., 2023b):

$$
\begin{array} { r } { \pmb { v } ^ { \mathrm { o l d } } ( \pmb { x } _ { t } , \pmb { c } , t ) = a _ { t } \pmb { x } _ { t } + b _ { t } \mathbb { E } _ { \pi ^ { \mathrm { o l d } } ( \pmb { x } _ { 0 } | \pmb { x } _ { t } , \pmb { c } ) } [ \pmb { x } _ { 0 } ] } \\ { \pmb { v } ^ { + } ( \pmb { x } _ { t } , \pmb { c } , t ) = a _ { t } \pmb { x } _ { t } + b _ { t } \mathbb { E } _ { \pi ^ { + } ( \pmb { x } _ { 0 } | \pmb { x } _ { t } , \pmb { c } ) } [ \pmb { x } _ { 0 } ] } \\ { \pmb { v } ^ { - } ( \pmb { x } _ { t } , \pmb { c } , t ) = a _ { t } \pmb { x } _ { t } + b _ { t } \mathbb { E } _ { \pi ^ { - } ( \pmb { x } _ { 0 } | \pmb { x } _ { t } , \pmb { c } ) } [ \pmb { x } _ { 0 } ] } \end{array}
$$

where $\begin{array} { r } { a _ { t } = \frac { \dot { \sigma } _ { t } } { \sigma _ { t } } , b _ { t } = \dot { \alpha } _ { t } - \frac { \dot { \sigma } _ { t } \alpha _ { t } } { \sigma _ { t } } } \end{array}$ Based on Lemma A.2 we have

$$
v ^ { \mathrm { o l d } } ( x _ { t } , c , t ) = \alpha ( x _ { t } ) v ^ { + } ( x _ { t } , c , t ) + [ 1 - \alpha ( x _ { t } ) ] v ^ { - } ( x _ { t } , c , t )
$$

Rearranging the equation, we complete the proof.

Theorem A.4 (Reinforcement Guidance Optimization). Consider the training objective:

$$
\mathcal { L } ( \theta ) = \mathbb { E } _ { c , \pi ^ { o l d } ( { \pmb x } _ { 0 } \mid c ) , t } r \| { \pmb v } _ { \theta } ^ { + } ( { \pmb x } _ { t } , { \pmb c } , t ) - { \pmb v } \| _ { 2 } ^ { 2 } + ( 1 - r ) \| { \pmb v } _ { \theta } ^ { - } ( { \pmb x } _ { t } , { \pmb c } , t ) - { \pmb v } \| _ { 2 } ^ { 2 } ,
$$

where $\begin{array} { r } { \pmb { v } _ { \theta } ^ { + } ( \pmb { x } _ { t } , \pmb { c } , t ) : = ( 1 - \beta ) \pmb { v } ^ { o l d } ( \pmb { x } _ { t } , \pmb { c } , t ) + \beta \pmb { v } _ { \theta } ( \pmb { x } _ { t } , \pmb { c } , t ) , } \end{array}$ (Implicit positive policy) and $v _ { \theta } ^ { - } ( x _ { t } , c , t ) : = ( 1 + \beta ) v ^ { o l d } ( x _ { t } , c , t ) - \beta v _ { \theta } ( x _ { t } , c , t )$ . (Implicit negative policy) Given unlimited data and model capacity, the optimal solution of Eq. (10) satisfies

$$
{ \pmb v } _ { \theta ^ { * } } ( { \pmb x } _ { t } , c , t ) = { \pmb v } ^ { o l d } ( { \pmb x } _ { t } , c , t ) + \frac { 2 } { \beta } \Delta ( { \pmb x } _ { t } , c , t ) .
$$

Proof.

$$
\begin{array} { r l } & { \mathrel { \mathop : } ( \theta ) = \mathbb { E } _ { c , t , \pi _ { t } ^ { \mathrm { o l d } } ( { \boldsymbol x } _ { t } | c ) \pi _ { 0 | t } ^ { \mathrm { o l d } } ( { \boldsymbol x } _ { 0 } | { \boldsymbol x } , c ) } r ( { \boldsymbol x } _ { 0 } , c ) \| v _ { \theta } ^ { + } ( { \boldsymbol x } _ { t } , c , t ) - v \| _ { 2 } ^ { 2 } + [ 1 - r ( { \boldsymbol x } _ { 0 } , c ) ] \| v _ { \theta } ^ { - } ( { \boldsymbol x } _ { t } , c , t ) - v \| _ { 2 } ^ { 2 } } \\ & { \qquad = \mathbb { E } _ { c . t , \pi _ { t } ^ { \mathrm { o l d } } ( { \boldsymbol x } _ { t } | c ) } \{ \mathbb { E } _ { \pi _ { 0 | t } ^ { \mathrm { o l d } } ( { \boldsymbol x } _ { 0 } | { \boldsymbol x } , c ) } r ( { \boldsymbol x } _ { 0 } , c ) \| v _ { \theta } ^ { + } ( { \boldsymbol x } _ { t } , c , t ) - v \| _ { 2 } ^ { 2 } } \\ & { \qquad + \mathbb { E } _ { \pi _ { 0 | t } ^ { \mathrm { o l d } } ( { \boldsymbol x } _ { 0 } | { \boldsymbol x } _ { t } , c ) } [ 1 - r ( { \boldsymbol x } _ { 0 } , c ) ] \| v _ { \theta } ^ { - } ( { \boldsymbol x } _ { t } , c , t ) - v \| _ { 2 } ^ { 2 } \} } \end{array}
$$

From Lemma A.1 we have $r ( \pmb { x } _ { 0 } , \pmb { c } ) \pi ^ { \mathrm { o l d } } ( \pmb { x } _ { 0 } | \pmb { c } ) = p _ { \pi ^ { \mathrm { o l d } } } ( \mathbf { o } = 1 | \pmb { c } ) \pi ^ { + } ( \pmb { x } _ { 0 } | \pmb { c } )$ , therefore:

$$
\begin{array} { r l } & { r ( { \boldsymbol x } _ { 0 } , c ) \pi _ { 0 | t } ^ { \mathrm { o l d } } ( { \boldsymbol x } _ { 0 } | { \boldsymbol x } _ { t } , c ) = r ( { \boldsymbol x } _ { 0 } , c ) \frac { \pi ^ { \mathrm { o l d } } ( { \boldsymbol x } _ { 0 } | c ) \pi ( { \boldsymbol x } _ { t } | { \boldsymbol x } _ { 0 } ) } { \pi _ { t } ^ { \mathrm { o l d } } ( { \boldsymbol x } _ { t } | c ) } } \\ & { \quad \quad \quad \quad \quad = p _ { \pi ^ { \mathrm { o l d } } } ( \mathbf { o } = 1 | c ) \frac { \pi _ { t } ^ { + } ( { \boldsymbol x } _ { t } | c ) } { \pi _ { t } ^ { \mathrm { o l d } } ( { \boldsymbol x } _ { t } | c ) } \frac { \pi ^ { + } ( { \boldsymbol x } _ { 0 } | c ) \pi ( { \boldsymbol x } _ { t } | { \boldsymbol x } _ { 0 } ) } { \pi _ { t } ^ { + } ( { \boldsymbol x } _ { t } | c ) } } \\ & { \quad \quad \quad \quad = p _ { \pi ^ { \mathrm { o l d } } } ( \mathbf { o } = 1 | c ) \frac { \pi _ { t } ^ { + } ( { \boldsymbol x } _ { t } | c ) } { \pi _ { t } ^ { \mathrm { o l d } } ( { \boldsymbol x } _ { t } | c ) } \pi _ { 0 | t } ^ { + } ( { \boldsymbol x } _ { 0 } | { \boldsymbol x } _ { t } , c ) } \\ & { \quad \quad \quad \quad = \alpha ( { \boldsymbol x } _ { t } ) \pi _ { 0 | t } ^ { + } ( { \boldsymbol x } _ { 0 } | { \boldsymbol x } _ { t } , c ) } \end{array}
$$

Similarly,

$$
[ 1 - r ( { \pmb x } _ { 0 } , { \pmb c } ) ] \pi _ { 0 | t } ^ { \mathrm { o l d } } ( { \pmb x } _ { 0 } | { \pmb x } _ { t } , { \pmb c } ) = [ 1 - \alpha ( { \pmb x } _ { t } ) ] \pi _ { 0 | t } ^ { - } ( { \pmb x } _ { 0 } | { \pmb x } _ { t } , { \pmb c } )
$$

Then,

$$
\begin{array} { r l } & { \mathcal { L } ( \theta ) = \mathbb { E } _ { c , t , \pi _ { t } ^ { \mathrm { o l d } } ( { \boldsymbol { \alpha } } _ { t } \mid c ) } \{ \alpha ( { \boldsymbol { x } } _ { t } ) \mathbb { E } _ { \pi _ { 0 | t } ^ { + } ( { \boldsymbol { \alpha } } _ { 0 } \mid { \boldsymbol { x } } _ { t } , c ) } \} \| v _ { \theta } ^ { + } ( { \boldsymbol { x } } _ { t } , c , t ) - v \| _ { 2 } ^ { 2 } } \\ & { \qquad + \left[ 1 - \alpha ( { \boldsymbol { x } } _ { t } ) \right] \mathbb { E } _ { \pi _ { 0 | t } ^ { - } ( { \boldsymbol { \alpha } } _ { 0 } \mid { \boldsymbol { x } } _ { t } , c ) } \| v _ { \theta } ^ { - } ( { \boldsymbol { x } } _ { t } , c , t ) - v \| _ { 2 } ^ { 2 } \} } \\ & { \qquad = \mathbb { E } _ { c , t , \pi _ { t } ^ { \mathrm { o l d } } ( { \boldsymbol { \alpha } } _ { t } \mid c ) } \{ \alpha ( { \boldsymbol { x } } _ { t } ) \| v _ { \theta } ^ { + } ( { \boldsymbol { x } } _ { t } , c , t ) - \mathbb { E } _ { \pi _ { 0 | t } ^ { + } ( { \boldsymbol { \alpha } } _ { 0 } \mid { \boldsymbol { x } } _ { t } , c ) } [ v ] \| _ { 2 } ^ { 2 } } \\ & { \qquad + \left[ 1 - \alpha ( { \boldsymbol { x } } _ { t } ) \right] \| v _ { \theta } ^ { - } ( { \boldsymbol { x } } _ { t } , c , t ) - \mathbb { E } _ { \pi _ { 0 | t } ^ { - } ( { \boldsymbol { x } } _ { 0 } \mid { \boldsymbol { x } } _ { t } , c ) } [ v ] \| _ { 2 } ^ { 2 } \} + C _ { 1 } } \\ &  \qquad = \mathbb { E } _ { c , t , \pi _ { t } ^ { \mathrm { o l d } } ( { \boldsymbol { \alpha } } _ { t } \mid c ) } \{ \alpha ( { \boldsymbol { x } } _ { t } ) \| v _ { \theta } ^ { + } \end{array}
$$

Combining Theorem A.3, we observe that

$$
\begin{array} { l } { v _ { \theta } ^ { + } ( x _ { t } , c , t ) - v ^ { + } ( x _ { t } , c , t ) = ( 1 - \beta ) v ^ { \mathrm { o l d } } ( x _ { t } , c , t ) + \beta v _ { \theta } ( x _ { t } , c , t ) - v ^ { + } ( x _ { t } , c , t ) } \\ { \displaystyle \qquad = \beta [ v _ { \theta } - v ^ { \mathrm { o l d } } - \frac { 1 } { \beta } \frac { \Delta } { \alpha ( x _ { t } ) } ] } \\ { v _ { \theta } ^ { - } ( x _ { t } , c , t ) - v ^ { - } ( x _ { t } , c , t ) = ( 1 + \beta ) v ^ { \mathrm { o l d } } ( x _ { t } , c , t ) - \beta v _ { \theta } ( x _ { t } , c , t ) - v ^ { - } ( x _ { t } , c , t ) } \\ { \displaystyle \qquad = - \beta [ v _ { \theta } - v ^ { \mathrm { o l d } } - \frac { 1 } { \beta } \frac { \Delta } { 1 - \alpha ( x _ { t } ) } ] } \end{array}
$$

Substituting these results into $\mathcal { L } ( \boldsymbol { \theta } )$ :

$$
\begin{array} { r l } { \mathcal { L } ( \theta ) = \mathbb { E } _ { \epsilon , 1 , r _ { i } ^ { \mathrm { s a l } } ( x _ { 1 } , \epsilon ) } \{ \alpha ( x _ { 1 } ) \beta ^ { 2 } \| v _ { \theta } - v ^ { \mathrm { o d d } } - \frac { 1 } { \beta } \frac { \Delta } { \alpha ( x _ { k } ) } \| _ { 2 } ^ { 2 } } & { } \\  + \left[ 1 - \alpha ( x _ { k } ) \right] \beta ^ { 2 } \| v _ { \theta } - v ^ { \mathrm { o d d } } - \frac { 1 } { \beta } \frac { \Delta } { 1 - \alpha ( x _ { k } ) } \| _ { 2 } ^ { 2 } \} & { } \\ { = \beta ^ { 2 } \mathbb { E } _ { \epsilon , t , r _ { i } ^ { \mathrm { s a l } } ( x _ { 1 } , \epsilon ) } \{ \alpha ( x _ { 1 } ) \| v _ { \theta } - ( v ^ { \mathrm { o d d } } + \frac { 1 } { \beta } \frac { \Delta } { \alpha ( x _ { k } ) } ) \| _ { 2 } ^ { 2 } } & { } \\ { + \left[ 1 - \alpha ( x _ { k } ) \right] \| v _ { \theta } - ( v ^ { \mathrm { o d d } } + \frac { 1 } { \beta } \frac { \Delta } { 1 - \alpha ( x _ { k } ) } ) \| _ { 2 } ^ { 2 } \} + C _ { 1 } } & { } \\ { - \beta ^ { 2 } \mathbb { E } _ { \epsilon , t , r ^ { \mathrm { s a l } } ( x _ { 1 } , \epsilon ) } \| v _ { \theta } - \alpha ( x _ { k } ) ( v ^ { \mathrm { o d d } } + \frac { 1 } { \beta } \frac { \Delta } { \alpha ( x _ { k } ) } ) - \left[ 1 - \alpha ( x _ { k } ) \right] ( v ^ { \mathrm { a d d } } + \frac { 1 } { \beta } \frac { \Delta } { 1 - \alpha ( x _ { k } ) } ) \| _ { 2 } ^ { 2 } + C _ { 1 } } & { } \\  = \beta ^ { 2 } \mathbb { E } _ { \epsilon , t , r ^ { \mathrm { s a l } } ( x _ { 1 } , \epsilon ) } \| v _   \end{array}
$$

from which it isvious that the tal $\theta ^ { * }$ satisfies $\begin{array} { r } { { v _ { \theta ^ { * } } } ( { x _ { t } } , c , t ) = { v } ^ { \mathrm { o l d } } ( { x _ { t } } , c , t ) + { \frac { 2 } { \beta } } \Delta ( { x _ { t } } , c , t ) . } \end{array}$

# B Theoretical Discussions

# B.1 FLOW SDE

As flow models are a special case of diffusion models under the rectified schedule $\alpha _ { t } = 1 - t , \sigma _ { t } = t$ the earliest results on diffusion SDEs (Song et al., 2020b) can be directly applied without difficulty. FlowGRPO (Liu et al., 2025) and DanceGRPO (Xue et al., 2025) derive the flow SDE with unexplained hyperparameters $\begin{array} { r } { g _ { t } = a \sqrt { \frac { t } { 1 - t } } } \end{array}$ or additional complexity. We provide a simpler and more principled perspective based solely on the diffusion model framework.

To leverage the diffusion SDE formulation in Song et al. (2020b), we need to match its forward SDE $\mathrm { d } \pmb { x } _ { t } = f ( t ) \pmb { x } _ { t } \mathrm { d } t + g ( t ) \mathrm { d } \pmb { w } _ { t }$ with the forward transition kernel ${ \pmb x } _ { t } = \alpha _ { t } { \pmb x } _ { 0 } + \sigma _ { t } { \pmb \epsilon }$ As noted in the first two arXiv versions of the VDM paper (Kingma et al., 2021), $f ( t ) , g ( t )$ are related to $\alpha _ { t } , \sigma _ { t }$ by $\begin{array} { r } { f ( t ) = \frac { \mathrm { d } \log \alpha _ { t } } { \mathrm { d } t } } \end{array}$ d logtα , g2(t) = $\begin{array} { r } { g ^ { 2 } ( t ) = \frac { \mathrm { d } \sigma _ { t } ^ { 2 } } { \mathrm { d } t } - 2 \frac { \mathrm { d } \log { \alpha _ { t } } } { \mathrm { d } t } \sigma _ { t } ^ { 2 } } \end{array}$ Setting $\alpha _ { t } = 1 - t , \sigma _ { t } = t$ we have

$$
f ( t ) = - { \frac { 1 } { 1 - t } } , \quad g ^ { 2 } ( t ) = { \frac { 2 t } { 1 - t } }
$$

for rectified flow. According to (Huang et al., 2021), the generalized reverse SDE takes the form:

$$
\mathrm { d } \pmb { x } _ { t } = \left[ f ( t ) \pmb { x } _ { t } - \frac { 1 + \lambda _ { t } ^ { 2 } } { 2 } g ^ { 2 } ( t ) \nabla _ { \pmb { x } _ { t } } \log \pi _ { t } ( \pmb { x } _ { t } ) \right] \mathrm { d } t + \lambda _ { t } g ( t ) \mathrm { d } \bar { \pmb { w } } _ { t }
$$

where $\lambda _ { t } \in [ 0 , 1 ]$ . Equivalently, it amounts to introducing Langevin dynamics on top of the diffusion ODE, with $\lambda _ { t } = 0$ corresponding to ODE, and $\lambda _ { t } = 1$ corresponding to the maximum variance SDE in Song et al. (2020b). The score function ${ \pmb s } _ { \theta } ( { \pmb x } _ { t } , t ) \approx \nabla _ { { \pmb x } _ { t } } \log \pi _ { t } ( { \pmb x } _ { t } )$ , noise predictor $\epsilon _ { \theta } ( x _ { t } , t )$ , data predictor ${ \pmb x } _ { \theta } ( { \pmb x } _ { t } , t )$ and velocity predictor ${ \pmb v } _ { \theta } ( { \pmb x } _ { t } , t )$ are interconvertible under general noise schedules (Zheng et al., 2023b):

$$
{ \bf \nabla } _ { \theta } ( { \bf x } _ { t } , t ) = - \sigma _ { t } s _ { \theta } ( { \bf x } _ { t } , t ) , \quad { \bf x } _ { \theta } ( { \bf x } _ { t } , t ) = \frac { { \bf x } _ { t } - \sigma _ { t } \epsilon _ { \theta } ( { \bf x } _ { t } , t ) } { \alpha _ { t } } , \quad { \bf v } _ { \theta } ( { \bf x } _ { t } , t ) = \dot { \alpha } _ { t } x _ { \theta } ( { \bf x } _ { t } , t ) + \dot { \sigma } _ { t } \epsilon _ { \theta } ( { \bf x } _ { t } , t )
$$

Applying these relations to the rectified flow schedule, we can derive:

$$
\mathbf { \boldsymbol { s } } _ { \theta } ( \mathbf { \boldsymbol { x } } _ { t } , t ) = - \frac { \mathbf { \boldsymbol { x } } _ { t } + ( 1 - t ) \mathbf { \boldsymbol { v } } _ { \theta } ( \mathbf { \boldsymbol { x } } _ { t } , t ) } { t }
$$

Substituting Eq. (11) and Eq. (14) into Eq. (12), we have the diffusion SDE under rectified flow:

$$
\mathrm { d } \pmb { x } _ { t } = \left[ ( 1 + \lambda _ { t } ^ { 2 } ) \pmb { v } _ { \theta } ( \pmb { x } _ { t } , t ) + \frac { \lambda _ { t } ^ { 2 } } { 1 - t } \pmb { x } _ { t } \right] \mathrm { d } t + \lambda _ { t } \sqrt { \frac { 2 t } { 1 - t } } \mathrm { d } \pmb { w }
$$

$\begin{array} { r } { g _ { t } = \lambda _ { t } \sqrt { \frac { 2 t } { 1 - t } } } \end{array}$ from the interpolation parameter $\lambda _ { t } \in [ 0 , 1 ]$ to the variance parameter $g _ { t }$ . This also explains the choice $g _ { t } =$ $a { \sqrt { \frac { t } { 1 - t } } }$ in FlowGRPO, where $a = \sqrt { 2 } \lambda _ { t }$ is a scaled version of $\lambda _ { t }$ with $a = { \sqrt { 2 } }$ corresponding to the maximum variance SDE. In comparison, DanceGRPO adopts a fixed variance $g _ { t }$ across timesteps, which is less effective on image models while more stable on video models.

FlowGRPO and DanceGRPO directly take the Euler discretization of the flow SDE. In principle, there are more accurate ways, such as utilizing the idea of diffusion implicit models (Song et al., 2020a; Zheng et al., 2024), which is equivalent to the first-order discretization after applying exponential integrators (Hochbruck & Ostermann, 2010; Zhang & Chen, 2022; Gonzalez et al., 2023). Specifically, the sampling step from $t$ to $s < t$ can be derived as:

$$
\begin{array} { r } { \mathbf { r } _ { s } = \left[ ( 1 - s ) + \sqrt { s ^ { 2 } - \rho _ { t } ^ { 2 } } \right] x _ { t } - \left[ ( 1 - s ) t - \sqrt { s ^ { 2 } - \rho _ { t } ^ { 2 } } ( 1 - t ) \right] v _ { \theta } ( x _ { t } , t ) + \rho _ { t } \epsilon , \quad \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) } \end{array}
$$

where $\begin{array} { r } { \rho _ { t } = \eta _ { t } s \sqrt { 1 - \frac { s ^ { 2 } ( 1 - t ) ^ { 2 } } { t ^ { 2 } ( 1 - s ) ^ { 2 } } } } \end{array}$ $\eta _ { t } \in [ 0 , 1 ]$ SDE. Compared to the Euler discretization, the DDIM-style discretization avoids singularities at boundaries and is expected to reduce sampling errors. However, we did not observe notable advantages by replacing the SDE sampler with stochastic DDIM. Concurrent work (Wang & Yu, 2025) improves the SDE sampler through the Coefficients-Preserving Sampling (CPS) principle.

# B.2 HIgH-ORDER FLOW ODE SAMPLER

We implement the 2nd-order ODE sampler for flow models based on the DPM-Solver series (Lu et al., 2022a;b; Zheng et al., 2023a), which uses the multistep method and half the log signal-to-noise ratio (SNR) $\lambda _ { t } = \log ( \alpha _ { t } / \sigma _ { t } )$ for time discretization. Specifically, for three consecutive timesteps $t _ { i } < t _ { i - 1 } < t _ { i - 2 }$ , where ${ \pmb x } _ { t _ { i - 1 } } , { \pmb x } _ { t _ { i - 2 } }$ are already obtained, the update rule for $\mathbf { x } _ { t _ { i } }$ is:

$$
x _ { t _ { i } } = \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } x _ { t _ { i - 1 } } - \alpha _ { t _ { i } } ( e ^ { - h _ { i } } - 1 ) \left[ \left( 1 + \frac { 1 } { 2 r _ { i } } \right) x _ { \theta } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) - \frac { 1 } { 2 r _ { i } } x _ { \theta } ( x _ { t _ { i - 2 } } , t _ { i - 2 } ) \right]
$$

where $\begin{array} { r } { h _ { i } = \lambda _ { t _ { i } } - \lambda _ { t _ { i - 1 } } , r _ { i } = \frac { h _ { i - 1 } } { h _ { i } } } \end{array}$ and the data predictor ${ \pmb x } _ { \theta } = { \pmb x } _ { t } - t { \pmb v } _ { \theta }$ for rectified flow. Highorder solvers are also adopted in MixGRPO (Li et al., 2025b) but only for certain steps. Adopting the 2nd-order solver throughout the entire sampling process is infeasible, as $\lambda _ { t }$ will be infinity at boundaries $t = 0$ or $t = 1$ . Following common practices, the first and last steps degrade to the first-order solver, which is the default Euler discretization for flow models.

# B.3 INTUITION BEHIND THE FLOWGRPO OBJECTIVE

We provide some insight into reverse-process diffusion RL by inspecting the FlowGRPO objective in a sampler-agnostic manner. For any first-order SDE sampler, the reverse sampling step from $t$ to $s < t$ can be expressed as

$$
\pmb { x } _ { s } = l ( s , t ) \pmb { x } _ { t } - m ( s , t ) \pmb { v } _ { \theta } ( \pmb { x } _ { t } , t ) + n ( s , t ) \pmb { \epsilon } , \quad \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )
$$

where $l ( s , t ) , m ( s , t ) , n ( s , t )$ depend only on $s , t$ and the sampler. Consider the on-policy case and the branching strategy in MixGRPO. Starting from a shared $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ , a group of $N$ noises $\epsilon ^ { ( 1 ) } , \dots , \epsilon ^ { ( N ) }$ aresampled and incorporated into the reverse step to produce multiple samples $\pmb { x } _ { s } ^ { ( 1 ) } , \ldots , \pmb { x } _ { s } ^ { ( N ) }$ .

They go through further sampling, yielding $N$ clean samples and corresponding advantages $A ^ { ( 1 ) } , \dotsc , A ^ { ( N ) }$ On-policy GRPO minimizes the negative advantage-weighted log likelihoods:

$$
\mathcal { L } ( \theta ) = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } A ^ { ( i ) } \log p _ { \theta } ( x _ { s } ^ { ( i ) } | \pmb { x } _ { t } )
$$

where

$$
\begin{array} { c } { { \log p _ { \theta } ( x _ { s } ^ { ( i ) } | x _ { t } ) = - \frac { \| x _ { s } ^ { ( i ) } - ( l ( s , t ) x _ { t } - m ( s , t ) v _ { \theta } ( x _ { t } , t ) ) \| _ { 2 } ^ { 2 } } { 2 n ^ { 2 } ( s , t ) } + C } } \\ { { = - \frac { \| m ( s , t ) v _ { \theta } ( x _ { t } , t ) - m ( s , t ) v _ { \mathrm { s g } ( \theta ) } ( x _ { t } , t ) + n ( s , t ) \epsilon ^ { ( i ) } \| _ { 2 } ^ { 2 } } { 2 n ^ { 2 } ( s , t ) } + C } } \end{array}
$$

erge bee the pes $\pmb { x } _ { s } ^ { ( 1 ) } , \ldots , \pmb { x } _ { s } ^ { ( N ) }$ log likelihood w.r.t. can be surprisingly reduced to a simple form:

$$
\nabla _ { \boldsymbol { \theta } } \log { p _ { \boldsymbol { \theta } } ( \mathbf { x } _ { s } ^ { ( i ) } | \mathbf { x } _ { t } ) } = - \frac { m ( s , t ) } { n ( s , t ) } \nabla _ { \boldsymbol { \theta } } ( ( \epsilon ^ { ( i ) } ) ^ { \top } \mathbf { v } _ { \boldsymbol { \theta } } ( \mathbf { x } _ { t } , t ) )
$$

and

$$
\nabla _ { \theta } \mathcal { L } ( \theta ) = \frac { m ( s , t ) } { n ( s , t ) } \nabla _ { \theta } \left[ \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( A ^ { ( i ) } \epsilon ^ { ( i ) } ) ^ { \top } \pmb { v } _ { \theta } ( \pmb { x } _ { t } , t ) \right]
$$

Therefore, FlowGRPO essentially aligns the velocity field with the advantage-weighted noise, while $\textstyle { \frac { m ( s , t ) } { n ( s , t ) } }$ across sampling steps. In the following, we show a further conclusion that FlowGRPO can be viewed as $a$ gradient estimation of reward backpropagation.

Denote $r _ { t } ( \pmb { x } _ { t } )$ as the implicit gradient-free function that solves the PF-ODE from $t$ to 0 and fetches the reward on the cleaned sample. The rewards can be expressed as

$$
r ^ { ( i ) } = r _ { s } \Big ( l ( s , t ) \pmb { x } _ { t } - m ( s , t ) \pmb { v } _ { \theta } ( \pmb { x } _ { t } , t ) + n ( s , t ) \pmb { \epsilon } ^ { ( i ) } \Big )
$$

According to Stein's identity, we have

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } r ^ { ( i ) } \epsilon ^ { ( i ) } \approx \mathbb { E } _ { \epsilon \sim \mathcal { N } ( \mathbf { 0 } , I ) } \left[ r _ { s } \left( l ( s , t ) x _ { t } - m ( s , t ) v _ { \theta } ( x _ { t } , t ) + n ( s , t ) \epsilon \right) \epsilon \right] } \\ & { \quad \quad \quad \quad = n ( s , t ) \mathbb { E } _ { \epsilon \sim \mathcal { N } ( \mathbf { 0 } , I ) } \left[ \nabla r _ { s } \left( l ( s , t ) x _ { t } - m ( s , t ) v _ { \theta } ( x _ { t } , t ) + n ( s , t ) \epsilon \right) \right] } \end{array}
$$

Therefore,

$$
\begin{array} { r l } & { \quad \nabla _ { \theta } \left[ \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( A ^ { ( i ) } \epsilon ^ { ( i ) } ) ^ { \top } v _ { \theta } ( x _ { t } , t ) \right] } \\ & { \approx \frac { n ( s , t ) } { \sigma } \mathbb { E } _ { \epsilon \sim \mathcal { N } ( \mathbf { 0 } , I ) } \left[ \nabla r _ { s } \left( l ( s , t ) x _ { t } - m ( s , t ) v _ { \theta } ( x _ { t } , t ) + n ( s , t ) \epsilon \right) \nabla _ { \theta } v _ { \theta } ( x _ { t } , t ) \right] } \\ & { = - \displaystyle \frac { n ( s , t ) } { m ( s , t ) \sigma } \mathbb { E } _ { \epsilon \sim \mathcal { N } ( \mathbf { 0 } , I ) } \left[ \nabla _ { \theta } r _ { s } \left( l ( s , t ) x _ { t } - m ( s , t ) v _ { \theta } ( x _ { t } , t ) + n ( s , t ) \epsilon \right) \right] } \end{array}
$$

where $\sigma$ is the global std used in GRPO normalization. Therefore, the GRPO loss gradient is

$$
\nabla _ { \theta } \mathcal { L } ( \theta ) \approx - \frac { 1 } { \sigma } \mathbb { E } _ { \epsilon \sim \mathcal { N } ( \mathbf { 0 } , I ) } \left[ \nabla _ { \theta } r _ { s } \left( l ( s , t ) x _ { t } - m ( s , t ) v _ { \theta } ( x _ { t } , t ) + n ( s , t ) \epsilon \right) \right]
$$

From the above gradient, GRPO optimizes the reverse transition $t  s$ when the remaining trajectory $s  0$ is gradient-free. Compared to works like ReFL (Xu et al., 2023), which conduct direct gradient backpropagation and approximate $s \to 0$ with a single forward pass ( ${ \bf \delta x } _ { 0 }$ -prediction), GRPO introduces higher estimation variance but avoids backpropagation through the $s \to 0$ process, allowing larger $s$ and a longer sampling chain for $s \to 0$ .

# C Experiment Details

Training Configurations. Our setup largely follows FlowGRPO, adopting the same number of groups per epoch (48), group size (24), LoRA configuration $( \alpha = 6 4 , r = 3 2 )$ , and learning rate $( 3 e \mathrm { ~ - ~ } 4 )$ . For each collected clean image, forward noising and loss computation are performed exactly on the corresponding sampling timesteps. We employ a 2nd-order ODE sampler for data collection and enable adaptive time weighting by default.

Single-Reward. For a head-to-head comparison with FlowGRPO under single-reward settings, we fix the number of sampling steps to 10 to ensure fairness. By default, we set $\beta = 1$ and $\eta _ { i } ~ =$ $\operatorname* { m i n } ( 0 . 0 0 1 i , 0 . 5 )$ , which work stably for most reward models. In the case of OCR, the reward rapidly approaches 1 within 100 iterations but suffers from instability. To address this, we adopt a more conservative soft-update strategy with $\eta _ { \mathrm { m a x } } = 0 . 9 9 9$ .

Multi-Reward. To comprehensively improve the base model across multiple rewards, we adopt a multi-stage training scheme. The training setup involves three categories of rewards and datasets: (1) PickScore, CLIPScore, and HPSv2.1 rewards on the Pick-a-Pic dataset; (2) GenEval reward with the three rewards above on the GenEval dataset; and (3) OCR reward with the three rewards above on the OCR dataset. Since the initial CFG-free generation is of low quality, we first train on (1) for 800 iterations to enhance image quality, followed by (2) for 300 iterations, (1) for 200 iterations, (2) for 200 iterations, and finally (3) for 100 iterations. All rewards are equally weighted, with PickScore divided by 26 for normalization to [0, 1]. By default, we use $\beta = 0 . 1$ and $\eta _ { i } = \operatorname* { m i n } ( 0 . 0 0 1 i , 0 . 5 )$ , while setting $\eta _ { \mathrm { m a x } } = 0 . 9 5$ for OCR to stabilize training. The number of sampling steps is fixed to 40 to ensure high-fidelity data collection.

# D ADDITIONAL RESULTS

Table 2: Evaluation results of FlowGRPO and DiffusionNFT trained on single rewards, both initialized from CFG-free base model.Gray-colored: In-domain reward. We observe that training exclusively on the OCR reward impairs generalization to other metrics; to compensate this, we enable CFG when evaluating non-OCR rewards for OCR-trained models.   

<table><tr><td rowspan="2">Model</td><td rowspan="2">#Iter</td><td colspan="2">Rule-Based</td><td colspan="6">Model-Based</td></tr><tr><td>GenEval</td><td>OCR</td><td>PickScore</td><td>ClipScore</td><td>HPSv2.1</td><td>Aesthetic</td><td>ImgRwd</td><td>UniRwd</td></tr><tr><td>SD3.5-M (w/o CFG)</td><td></td><td>0.24</td><td>0.12</td><td>20.51</td><td>0.237</td><td>0.204</td><td>5.13</td><td>-0.58</td><td>2.02</td></tr><tr><td>+ CFG</td><td></td><td>0.63</td><td>0.59</td><td>22.34</td><td>0.285</td><td>0.279</td><td>5.36</td><td>0.85</td><td>3.03</td></tr><tr><td>+ FlowGRPO</td><td>4k</td><td>0.97</td><td>0.30</td><td>21.78</td><td>0.277</td><td>0.248</td><td>5.15</td><td>0.74</td><td>2.87</td></tr><tr><td rowspan="5">+ Ours</td><td>1k</td><td>0.66</td><td>0.96</td><td>21.94</td><td>0.280</td><td>0.257</td><td>5.18</td><td>0.31</td><td>2.86</td></tr><tr><td>4k</td><td>0.54</td><td>0.60</td><td>23.62</td><td>0.257</td><td>0.295</td><td>6.42</td><td>1.17</td><td>3.17</td></tr><tr><td>1k</td><td>0.98</td><td>0.36</td><td>21.92</td><td>0.271</td><td>0.251</td><td>5.33</td><td>0.68</td><td>2.91</td></tr><tr><td>150</td><td>0.54</td><td>0.97</td><td>21.63</td><td>0.281</td><td>0.246</td><td>5.19</td><td>0.37</td><td>2.81</td></tr><tr><td>2k</td><td>0.53</td><td>0.64</td><td>24.03</td><td>0.270</td><td>0.315</td><td>6.17</td><td>1.29</td><td>3.40</td></tr></table>

We provide more qualitative comparison between the base model, FlowGRPO and our multi-reward optimized model in Figure 11, Figure 12 and Figure 13.

![](images/11.jpg)  
a photo of a brown hot dog and a purple pizza   
Figure 11: Qualitative comparison between FlowGRPO and our model on GenEval prompts.

![](images/12.jpg)  
A close-po amedicine bottle with a prominent warning label that reads "Consul Doctor", set agaist a neutral background, emphasizing the clarity and visibility of the text.

![](images/13.jpg)  
A courtroom scene with a judge's gavel resting on a wooden plaque that reads "Orderin the Cour", s against the backdrop of a quiet, solemn courtroom.

![](images/14.jpg)  
A realistic photo of a tech campus courtyard at night, featuring a glowing "AI Training Zone" hologram fl the futuristic atmosphere.

![](images/15.jpg)  
Anqu ypewrir wi hee  pape nser proe isplayngheyped wors "Chap1 It Wa Dark NighThe en  dimy e ud wi glesk lacg ao over the typewriter.

![](images/16.jpg)  
A ba  a coru e wihher "LieBe  ,

Figure 12: Qualitative comparison between FlowGRPO and our model on OCR prompts.

SD3.5-M SD3.5-M +FlowGRPO +DiffusionNFT (w/o CFG) (w/ CFG) (w/ CFG) (w/o CFG)

![](images/17.jpg)  
A side view of an owl sitting in a field.   
Figure 13: Qualitative comparison between FlowGRPO and our model on DrawBench prompts.