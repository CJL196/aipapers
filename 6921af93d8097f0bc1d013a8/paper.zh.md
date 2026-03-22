# Flow-GRPO：通过在线强化学习训练流匹配模型

刘杰1,3,5\* 刘工野2,3\* 梁佳俊3 李阳光 刘嘉恒4 王新涛3 瓦鹏飞3 张迪3 欧阳万里1,5 1香港中文大学 MMLab 2清华大学 3快手科技 Kling团队 4南京大学 5上海人工智能实验室 jieliu@link.cuhk.edu.hk wlouyang@ie.cuhk.edu.hk 代码： https://github.com/yifan123/flow_grpo

# 摘要

我们提出了 Flow-GRPO，这是第一个将在线策略梯度强化学习 (RL) 集成到流匹配模型中的方法。我们的方法使用了两个关键策略：(1) ODE 转 SDE 的转换，将确定性的常微分方程 (ODE) 转换为等效的随机微分方程 (SDE)，该 SDE 在所有时间步与原始模型的边际分布相匹配，从而实现 RL 探索的统计采样；(2) 去噪减缩策略，减少训练去噪步骤的同时保留原始推理步骤的数量，显著提高采样效率而不牺牲性能。实证结果表明，Flow-GRPO 在多项文本到图像的任务中表现有效。在组合生成方面，RL 调优的 SD3.5-M 的生成准确度从 $6 3 \%$ 提高到 $9 5 \%$。在视觉文本渲染中，准确度从 $5 9 \%$ 提高到 $9 2 \%$，极大提升了文本生成效果。Flow-GRPO 在人类偏好对齐方面也取得了显著进展。值得注意的是，几乎没有奖励操控的现象发生，这意味着奖励的增加并没有以可观的图像质量或多样性下降为代价。

# 1 引言

流匹配模型已经成为图像生成领域的主流，这得益于其坚实的理论基础和在生成高质量图像方面的强大表现。然而，它们在处理涉及多个对象、属性和关系的复杂场景时往往面临困难，同时在文本渲染方面也存在挑战。同时，在线强化学习已被证明在增强大型语言模型的推理能力方面非常有效。尽管之前的研究主要集中在将强化学习应用于早期的基于扩散的生成模型和离线强化学习技术（如直接偏好优化）用于基于流的生成模型，但在线强化学习在推动流匹配生成模型方面的潜力仍然很大程度上尚未被探索。在本研究中，我们探讨了如何利用在线强化学习有效地提升流匹配模型。利用强化学习训练流模型面临几个关键挑战：(1) 流模型依赖于基于常微分方程的确定性生成过程，这意味着在推理过程中不能进行随机采样。而强化学习依赖于随机采样来探索环境，通过尝试不同的动作并根据奖励进行改进。这种对随机性的需求与流匹配模型的确定性特性相矛盾。(2) 在线强化学习依赖于高效采样以收集训练数据，但流模型通常需要多次迭代步骤来生成每个样本，从而限制了效率。这个问题在大型模型中表现得更加明显。为了使强化学习在图像或视频生成等任务中具有实际应用，提升采样效率是至关重要的。

![](images/1.jpg)  

Figure 1: (a) GenEval performance rises steadily throughout Flow-GRPO's training and outperforms GPT-4o. (b) Image quality metrics on DrawBench [1] remain essentially unchanged. (c) Human Preference Scores on DrawBench improves after training. Results show that Flow-GRPO enhances the desired capability while preserving image quality and exhibiting minimal reward-hacking.

为了解决这些挑战，我们提出了Flow-GRPO，将GRPO [16]集成到文本到图像（T2I）生成的流匹配模型中，采用了两项关键策略。首先，我们采用ODE到SDE策略，以克服原始流模型的确定性特性。通过将基于ODE的流转换为等效的随机微分方程（SDE）框架，我们在保持原始边际分布的同时引入了随机性。其次，为了提高在线强化学习中的采样效率，我们应用了去噪减少策略，该策略在训练过程中减少去噪步骤，同时在推理时保持完整的调度。我们的实验表明，使用更少的步骤能保持性能，同时显著降低数据生成成本。

我们在不同奖励类型的T2I任务上评估了Flow-GRPO。(1) 可验证奖励，使用GenEval基准和视觉文本渲染任务。GenEval包括组合图像生成任务（例如，生成特定的物体计数、颜色和空间关系），这些任务可以通过物体检测方法自动评估。Flow-GRPO将Stable Diffusion 3.5 Medium (SD3.5-M)的准确率从$63\%$提升到$95\%$，超越了最先进的GPT-4o模型。在视觉文本渲染方面，SD3.5-M的准确率从$59\%$提高到$92\%$，大大增强了其文本生成能力。(2) 基于模型的奖励，例如人类偏好Pickscore奖励。这些结果表明我们的框架与任务无关，展示了其可推广性和鲁棒性。重要的是，所有的改进都是在很少的奖励干扰下实现的，如图1所示。总结来说，Flow-GRPO的贡献如下：•我们首次通过将确定性常微分方程采样转换为随机微分方程采样，将GRPO引入流匹配模型，展示了在线强化学习在T2I任务中的有效性。Flow-GRPO将SD3.5-M的准确率从$63\%$提升到$95\%$，而未明显降低图像质量。•我们发现流匹配模型的在线强化学习不需要标准的长时间步进行训练样本收集。通过在训练过程中使用较少的去噪步骤，并在测试过程中保留原始步骤，我们可以显著加快训练过程。•我们证明，Kullback-Leibler (KL)约束有效防止奖励干扰，其中奖励在牺牲图像质量或多样性的情况下增加。KL正则化在经验上并不等同于提前停止。通过合适的KL项，我们可以在保持良好图像质量的同时匹配KL-free版本的高奖励，尽管训练时间更长。

# 2 相关工作

针对大语言模型的强化学习。在线强化学习有效地提升了大语言模型的推理能力，例如 DeepSeek-R1 和 OpenAI-o1，采用了像 PPO 或无值网络的 GRPO 这样的策略梯度方法。GRPO 通过去除对值网络的需求，更加节省内存，因此我们在本研究中采用了它。PPO 也可以类似地应用于流匹配。扩散与流匹配。扩散模型通过向数据添加高斯噪声并训练神经网络以反转该过程。采样使用离散的 DDPM 步骤或概率流 SDE 解算器生成高保真输出。流匹配通过直接匹配速度场学习连续时间归一化流，允许仅用少量 ODE 步骤进行高效确定性采样。它在去噪步骤远少于扩散的情况下实现了具有竞争力的 FID，使其成为最近图像和视频生成模型的主流选择。最近的工作在 SDE/ODE 框架下统一了扩散和流模型。我们的工作建立在他们的理论基础上，并将 GRPO 引入基于流的模型中。

T2I的对齐。最近，旨在将预训练的T2I模型与人类偏好对齐的努力主要有五个方向：（1）使用可微奖励进行直接微调 [30, 31, 32, 33]；（2）奖励加权回归 (RWR) [34, 35, 36, 37]；（3）直接偏好优化 (DPO) 及其变体 [38, 39, 14, 40, 41, 42, 43, 44, 45, 46]；（4）基于PPO的策略梯度 [47, 48, 49, 50, 51, 52]；（5）无训练对齐方法 [53, 54, 55]。这些方法已成功地将T2I模型与人类偏好对齐，改善了美学和语义一致性。基于这一进展，我们引入了用于流匹配模型的GRPO，这是当前最先进的T2I系统的主干网络。并行工作 [56] 将GRPO应用于文本到语音流模型，但他们并不是将常微分方程转换为随机微分方程以注入随机性，而是通过估计高斯分布（预测速度的均值和方差）来重构速度预测，这需要对预训练模型进行重新训练。另一项研究 [57] 也探索了基于SDE的随机性，但重点关注推理时间的缩放。

# 3 前言

在本节中，我们引入流匹配的数学公式，并描述去噪过程如何映射为多步骤马尔可夫决策过程（MDP）。流匹配。设 $x _ { 0 } \sim X _ { 0 }$ 是来自真实分布的数据样本，$x _ { 1 } \sim X _ { 1 }$ 表示一个噪声样本。最近先进的图像生成模型（例如，文献[4，5]）和视频生成模型（例如，文献[24，26，25，27]）采用了整流流框架[3]，该框架定义了“有噪声”的数据 $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ 为 $t \in [ 0 , 1 ]$。然后，通过最小化流匹配目标[2，3]，训练一个变换器模型以直接回归速度场 ${ \pmb v } _ { \theta } ( { \pmb x } _ { t } , t )$。

$$
{ \pmb x } _ { t } = \left( 1 - t \right) { \pmb x } _ { 0 } \ + \ t { \pmb x } _ { 1 } ,
$$

$$
\mathcal { L } ( \theta ) = \mathbb { E } _ { t , \ : x _ { 0 } \sim X _ { 0 } , \ : x _ { 1 } \sim X _ { 1 } } \left[ \mathbf { \epsilon } \| \pmb { v } \mathrm { ~ - ~ } \pmb { v } _ { \theta } ( \pmb { x } _ { t } , t ) \| ^ { 2 } \right] ,
$$

目标速度场为 ${ \pmb v } = { \pmb x } _ { 1 } - { \pmb x } _ { 0 }$ 。

去噪作为马尔可夫决策过程（MDP）。如文献[12]所示，流匹配模型中的迭代去噪过程可以被形式化为一个马尔可夫决策过程（MDP）$( S , { \mathcal { A } } , \rho _ { 0 } , P , R )$。在第$t$步的状态为$\pmb { s } _ { t } \triangleq ( \pmb { c } , t , \pmb { x } _ { t } )$，动作为模型预测的去噪样本$\mathbf { \Phi } _ { { \pmb { a } } _ { t } } \triangleq { \pmb { x } } _ { t - 1 }$，而策略则为$\pi ( \mathbf { a } _ { t } \ | \ \mathbf { \beta } _ { s _ { t } } ) \ \triangleq \ p _ { \boldsymbol \theta } ( \mathbf { x } _ { t - 1 } \ | \ \mathbf { \beta } \mathbf { x } _ { t } , { \boldsymbol \mathsf { c } } )$。状态转移是确定性的：$P ( \pmb { s } _ { t + 1 } \mid \pmb { s } _ { t } , \pmb { a } _ { t } ) \triangleq$ $( \delta _ { c } , \delta _ { t - 1 } , \delta _ { { \pmb x } _ { t - 1 } } )$，初始状态分布为$\rho _ { 0 } ( \pmb { \mathscr { s } } _ { 0 } ) \triangleq ( p ( \pmb { c } ) , \delta _ { T } , \mathcal { N } ( \pmb { 0 } , \mathbf { I } ) )$，其中$\delta _ { y }$是以$y$为中心的狄拉克δ分布。奖励仅在最后一步给予：$R ( \pmb { s } _ { t } , \pmb { a } _ { t } ) \triangleq r ( \pmb { x } _ { 0 } , \pmb { c } )$当且仅当$t = 0$，否则为0。

# 4 流动-广义随机最优控制

在本节中，我们介绍了 Flow-GRPO，它通过在线强化学习增强流模型。我们首先回顾 GRPO [16] 的核心思想，并将其调整为流匹配。随后，我们展示如何将确定性常微分方程采样器转换为具有相同边际分布的随机微分方程采样器，引入应用 GRPO 所需的随机性。最后，我们介绍降噪减少，这是一种实用的采样策略，可以显著加快训练速度而不降低性能。

![](images/2.jpg)  

Figure 2: Overview of Flow-GRPO. Given a prompt set, we introduce an ODE-to-SDE strategy to enable stochastic sampling for online RL. With Denoising Reduction (only $\mathrm { T } = 1 0$ steps), we efficiently gather low-quality but still informative trajectories. Rewards from these trajectories feed the GRPO loss, which updates the model online and yields an aligned policy.

GRPO 在流匹配中的应用。强化学习的目标是学习一个策略，以最大化期望累计奖励。这通常被表述为优化一个带有正则化目标的策略 $\pi _ { \theta }$：

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { ( s _ { 0 } , a _ { 0 } , \ldots , s _ { T } , a _ { T } ) \sim \pi _ { \theta } } \left[ \sum _ { t = 0 } ^ { T } \left( R ( s _ { t } , a _ { t } ) - \beta D _ { \mathrm { K L } } ( \pi _ { \theta } ( \cdot \mid s _ { t } ) | | \pi _ { \mathrm { r e f } } ( \cdot \mid s _ { t } ) ) \right) \right] .
$$

与其他基于策略的方法如 PPO [20] 不同，GRPO [16] 提供了一种轻量级的替代方案，它引入了一种组相对形式来估计优势。回忆一下，去噪过程可以被表述为一个马尔可夫决策过程（MDP），如第 3 节所示。给定一个提示 $^ c$，流模型 $p _ { \theta }$ 采样出一组 $G$ 个独立图像 $\{ \boldsymbol { x } _ { 0 } ^ { i } \} _ { i = 1 } ^ { G }$ 以及相应的时间轨迹 $\{ ( \pmb { x } _ { T } ^ { i } , \pmb { x } _ { T - 1 } ^ { i } , \cdot \cdot \cdot , \pmb { x } _ { 0 } ^ { i } ) \} _ { i = 1 } ^ { G }$。然后，通过如下方式对组级奖励进行归一化，以计算第 $i$ 张图像的优势：

$$
\hat { A } _ { t } ^ { i } = \frac { R ( { \pmb x } _ { 0 } ^ { i } , { \pmb c } ) - \mathrm { m e a n } ( \{ R ( { \pmb x } _ { 0 } ^ { i } , { \pmb c } ) \} _ { i = 1 } ^ { G } ) } { \mathrm { s t d } ( \{ R ( { \pmb x } _ { 0 } ^ { i } , { \pmb c } ) \} _ { i = 1 } ^ { G } ) } .
$$

GRPO通过最大化以下目标来优化策略模型：

$$
\begin{array} { r } { \mathcal { T } _ { \mathrm { F l o w - G R P O } } ( \theta ) = \mathbb { E } _ { c \sim \mathcal { C } , \{ { \boldsymbol x } ^ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \vert c ) } f ( r , \hat { A } , \theta , \varepsilon , \beta ) , } \end{array}
$$

where /hwєә/ adv. 在哪里 pron. 哪里 n. 地点 Definition: pron. & conj. Whether. adv. At or in what place; hence, in what situation, position, or circumstances; -- used interrogatively. adv. At or in which place; at the place in which; hence, in the case or instance in which; -- used relatively. adv. To what or which place; hence, to what goal, result, or issue; whither; -- used interrogatively and relatively; as, where are you going? conj. Whereas. n. Place; situation.

$$
\begin{array} { c } { ^ { \mathrm { \tiny { ~ r } } } ( r , \hat { A } , \theta , \varepsilon , \beta ) = \displaystyle \frac 1 G \sum _ { i = 1 } ^ { G } \frac 1 T \sum _ { t = 0 } ^ { T - 1 } \left( \operatorname* { m i n } \left( r _ { t } ^ { i } ( \theta ) \hat { A } _ { t } ^ { i } , \ \mathrm { c l i p } \Big ( r _ { t } ^ { i } ( \theta ) , 1 - \varepsilon , 1 + \varepsilon \Big ) \hat { A } _ { t } ^ { i } \right) - \beta D _ { \mathrm { K L } } ( \pi _ { \theta } | | \pi _ { \mathrm { r e f } } ) \right) , } \\  ^ { \mathrm { \tiny { ~ r } } _ { t } ^ { i } ( \theta ) = \displaystyle \frac { p \theta ( x _ { t - 1 } ^ { i } \mid x _ { t } ^ { i } , c ) } { p _ { \theta _ { \mathrm { d d } } } \big ( { x } _ { t - 1 } ^ { i } \mid x _ { t } ^ { i } , c \big ) } . } \end{array}
$$

从常微分方程 (ODE) 到随机微分方程 (SDE)。GRPO依赖于公式4和公式5中的随机采样来生成多样化的轨迹，以进行优势估计和探索。扩散模型自然支持这一点：正向过程逐步添加高斯噪声，反向过程通过方差逐渐减小的马尔可夫链近似基于得分的SDE求解器。相比之下，流匹配模型在正向过程中使用确定性常微分方程 (ODE)：

$$
\mathrm { d } { \pmb { x } } _ { t } = { \pmb { v } } _ { t } \mathrm { d } t ,
$$

其中 ${ \mathbf { } } v _ { t }$ 是通过方程 2 中的流匹配目标学习得出的。一个常见的采样方法是对这个常微分方程（ODE）进行离散化，从而在连续时间步之间建立一一映射。这种确定性方法在满足 GRPO 策略更新要求方面存在两个关键问题：（1）方程 5 中的 $r _ { t } ^ { i } ( \theta )$ 需要计算 $p ( \pmb { x } _ { t - 1 } \mid \pmb { x } _ { t } , \pmb { c } )$，在确定性动力学下，由于发散估计而变得计算开销巨大。（2）更重要的是，强化学习依赖于探索。如第 5.3 节所示，减少的噪声会显著降低训练效率。确定性采样没有超出初始种子的随机性，尤其具有问题。为了解决这一限制，我们将方程 6 中的确定性流-常微分方程转换为一个等效的随机微分方程（SDE），使其在所有时间步上与原始模型的边际概率密度函数相匹配。我们在这里概述关键过程。详细证明见附录 A。遵循 [23, 28, 29]，我们构建了一个逆时间 SDE 形式，保留了边际分布：

$$
\mathrm { d } { \pmb x } _ { t } = \bigg ( { \pmb v } _ { t } ( { \pmb x } _ { t } ) - \frac { \sigma _ { t } ^ { 2 } } { 2 } \nabla \log p _ { t } ( { \pmb x } _ { t } ) \bigg ) \mathrm { d } t + \sigma _ { t } \mathrm { d } { \pmb w } ,
$$

其中 $\mathrm{d} w$ 表示维纳过程增量，$\sigma_{t}$ 控制生成过程中的随机性水平。对于修正流，公式 7 被指定为：

$$
\mathrm { d } \pmb { x } _ { t } = \left[ \pmb { v } _ { t } ( \pmb { x } _ { t } ) + \frac { \sigma _ { t } ^ { 2 } } { 2 t } \left( \pmb { x } _ { t } + ( 1 - t ) \pmb { v } _ { t } ( \pmb { x } _ { t } ) \right) \right] \mathrm { d } t + \sigma _ { t } \mathrm { d } \pmb { w } .
$$

应用欧拉-马鲁亚马离散化得到最终更新规则：

$$
\boxed { x _ { t + \Delta t } = x _ { t } + \left[ v _ { \theta } ( x _ { t } , t ) + \frac { \sigma _ { t } ^ { 2 } } { 2 t } \big ( x _ { t } + ( 1 - t ) v _ { \theta } ( x _ { t } , t ) \big ) \right] \Delta t + \sigma _ { t } \sqrt { \Delta t } \epsilon }
$$

其中 $\epsilon \sim \mathcal{N}(0, I)$ 引入了随机性。我们在本文中使用 $\sigma_{t} = a \sqrt{\frac{t}{1 - t}}$，其中 $a$ 是一个标量超参数，用以控制噪音水平（参见第 5.3 节以了解其对性能的影响）。方程 9 显示策略 $\pi_{\boldsymbol{\theta}}(\mathbf{x}_{t - 1} \mid \mathbf{x}_{t}, \mathbf{c})$ 是一个各向同性的高斯分布。我们可以轻松地计算方程 5 中的策略 $\pi_{\theta}$ 与参考策略 $\pi_{\mathrm{ref}}$ 之间的 KL 散度，其形式为：

$$
D _ { \mathrm { K L } } ( \pi _ { \theta } | | \pi _ { \mathrm { r e f } } ) = \frac { | | \overline { { x } } _ { t + \Delta t , \theta } - \overline { { x } } _ { t + \Delta t , \mathrm { r e f } } | | ^ { 2 } } { 2 \sigma _ { t } ^ { 2 } \Delta t } = \frac { \Delta t } { 2 } \left( \frac { \sigma _ { t } ( 1 - t ) } { 2 t } + \frac { 1 } { \sigma _ { t } } \right) ^ { 2 } \| v _ { \theta } ( x _ { t } , t ) - v _ { \mathrm { r e f } } ( x _ { t } , t ) \| ^ { 2 }
$$

去噪降噪。为了生成高质量图像，流模型通常需要许多去噪步骤，这使得在线强化学习的数据收集成本高昂。然而，我们发现在线强化学习训练过程中大时间步并非必要。在样本生成过程中，我们可以使用显著更少的去噪步骤，同时在推理过程中保留原始去噪步骤以获得高质量样本。注意，我们在训练中将时间步$T$设置为10，而推理时间步$T$则设置为原始默认设置$T = 40$（针对SD3.5-M）。我们的实验表明，这种方法实现了快速训练而不牺牲测试时的图像质量。

# 5 实验

本节通过实证评估Flow-GRPO在三个任务上改善流匹配模型的能力。（1）组合图像生成：该任务要求对物体进行精确的排列和属性控制。我们报告GenEval上的结果。（2）视觉文本渲染：一个基于规则的任务，评估按提示中指定的文本的准确渲染。（3）人类偏好对齐：该任务旨在使T2I模型与人类偏好对齐。

# 5.1 实验设置

我们介绍了三个任务，详细说明了它们各自的提示和奖励定义。有关超参数详情和计算资源规范，请参见附录 B.3 和附录 B.4。

组合图像生成。GenEval [17] 在六个复杂的组合图像生成任务中评估 T2I 模型，这些任务涉及对象计数、空间关系和属性绑定等复杂组合提示。我们使用其官方评估管道，该管道检测对象边界框和颜色，然后推断它们的空间关系。训练提示是通过官方 GenEval 脚本生成的，这些脚本使用模板和随机组合构建提示数据集。测试集严格去重：仅对象顺序不同的提示（例如，$"\mathtt{a}$ 一张 A 和 $\mathbb{B}^{\mathfrak{n}}$ 的照片" 与 "一张 B 和 A 的照片"）被视为相同，因此这些变体会从训练集中移除。根据基础模型在六个任务中的初始准确率，我们设置提示比例为 位置 : 计数 : 属性绑定 : 颜色 : 两个对象 : 单个对象 $= 7 : 5 : 3 : 1 : 1 : 0$。奖励基于规则：(1) 计数: $r = 1 - | N_{\mathrm{gen}}^{-} - N_{\mathrm{ref}} | / \bar{N_{\mathrm{ref}}}.$ (2) 位置 / 颜色: 如果对象计数正确，将分配部分奖励；当预测的位置或颜色也正确时，余下的奖励将被授予。

视觉文本渲染 [8]。文本在海报、书籍封面和表情包等图像中十分常见，因此在生成的图像中能够精确且连贯地放置文本对 T2I 模型至关重要。在我们的设定中，我们定义了一个文本渲染任务，其中每个提示遵循模板 $^{\mathfrak{c}\mathfrak{c}} \mathtt{A}$，表示“文本”。具体来说，占位符“文本”是应在图像中出现的确切字符串。我们使用 GPT4o 生成了 20K 训练提示和 1K 测试提示。根据 [58]，我们用奖励 $r = \mathrm{m a x} (1 - N_{\mathrm{e}} / N_{\mathrm{r e f}}, 0)$ 测量文本忠实度，其中 $N_{\mathrm{e}}$ 是渲染文本与目标文本之间的最小编辑距离，$N_{\mathrm{r e f}}$ 是提示中引号内的字符数。该奖励还作为我们的文本准确性指标。人类偏好对齐 [19]。该任务旨在将 T2I 模型与人类偏好对齐。我们使用 PickScore [19] 作为我们的奖励模型，该模型基于大规模人类注释的同一提示生成图像的成对比较。对于每个图像和提示对，PickScore 提供一个总体评分，评估多个标准，例如图像与提示的一致性及其视觉质量。图像质量评估指标。由于 T2I 模型的训练目标是最大化预定义奖励，它容易受到奖励黑客攻击，其结果是奖励增加但图像质量或多样性下降。本研究旨在使在线强化学习在 T2I 生成中有效，同时不明显影响质量或多样性。为了检测超出任务特定精度的奖励黑客行为，我们评估了四个自动图像质量指标：美学评分 [59]、DeQA [60]、ImageReward [32] 和 UnifiedReward [61]（详见附录 B.1）。所有指标是在 DrawBench [1] 上计算的，DrawBench 是一个包含多样提示的 T2I 模型综合基准。

# 5.2 主要结果

图1和表1显示Flow-GRPO在训练过程中生成评估性能稳步提升，最终超越了GPT-4o。这一过程在保持图像质量指标和DrawBench上的偏好评分的同时进行，DrawBench是一个具有多样且全面提示的基准，用于评估模型的整体能力。图3提供了定性比较。除了组合图像生成外，表2详细介绍了对视觉文本渲染和人类偏好任务的评估。Flow-GRPO提高了文本渲染能力，并在DrawBench上保持了图像质量指标和偏好评分。有关相关定性示例，请参见附录C.6中的图13、14和15。对于人类偏好任务，图像质量在没有KL正则化的情况下没有下降。然而，我们发现省略KL导致视觉多样性的崩塌，这是一种在第5.3节中进一步讨论的奖励黑客形式。这些结果表明，Flow-GRPO在提升所需能力的同时，对图像质量或视觉多样性的下降影响极小。Flow-GRPO与其他对齐方法的比较。我们将Flow-GRPO与几种对齐方法进行比较：监督微调（SFT）、Flow-DPO [14, 39] 及其在线变体。Flow-GRPO始终在所有基准中显著优于其他方法。在每一步中，我们使用与Flow-GRPO相同的组大小生成一组图像。唯一的区别在于更新规则：SFT：选择每组中最高奖励的图像并对其进行微调。Flow-DPO：将每组中最高奖励的图像作为选定样本，最低的作为被拒绝的样本，然后应用DPO损失。

Table 1: GenEval Result. Best scores are inblue, second-best ingreenResults for models other than SD3.5-M are from [7] or their original papers. Obj.: Object; Attr.: Attribution.   

<table><tr><td>Model</td><td>Overall</td><td>Single Obj.</td><td>Two Obj.</td><td>Counting</td><td>Colors</td><td>Position</td><td>Attr. Binding</td></tr><tr><td colspan="8">Diffusion Models</td></tr><tr><td>LDM [62]</td><td>0.37</td><td>0.92</td><td>0.29</td><td>0.23</td><td>0.70</td><td>0.02</td><td>0.05</td></tr><tr><td>SD1.5 [62]</td><td>0.43</td><td>0.97</td><td>0.38</td><td>0.35</td><td>0.76</td><td>0.04</td><td>0.06</td></tr><tr><td>SD2. 62]</td><td>0.50</td><td>0.98</td><td>0.51</td><td>0.44</td><td>0.85</td><td>0.07</td><td>0.17</td></tr><tr><td>SD-XL [63]</td><td>0.55</td><td>0.98</td><td>0.74</td><td>0.39</td><td>0.85</td><td>0.15</td><td>0.23</td></tr><tr><td>DALLE-2 [64]</td><td>0.52</td><td>0.94</td><td>0.66</td><td>0.49</td><td>0.77</td><td>0.10</td><td>0.19</td></tr><tr><td>DALLE-3 [65</td><td>0.67</td><td>0.96</td><td>0.87</td><td>0.47</td><td>0.83</td><td>0.43</td><td>0.45</td></tr><tr><td colspan="8">Autoregressive Models</td></tr><tr><td>Show-o [66]</td><td>0.53</td><td>0.95</td><td>0.52</td><td>0.49</td><td>0.82</td><td>0.11</td><td>0.28</td></tr><tr><td>Emu3-Gen [67]</td><td>0.54</td><td>0.98</td><td>0.71</td><td>0.34</td><td>0.81</td><td>0.17</td><td>0.21</td></tr><tr><td>JanusFlow [68</td><td>0.63</td><td>0.97</td><td>0.59</td><td>0.45</td><td>0.83</td><td>0.53</td><td>0.42</td></tr><tr><td>Janus-Pro-7B [69]</td><td>0.80</td><td>0.99</td><td>0.89</td><td>0.59</td><td>0.90</td><td>0.79</td><td>0.66</td></tr><tr><td>GPT-4o [18]</td><td>0.84</td><td>0.99</td><td>0.92</td><td>0.85</td><td>0.92</td><td>0.75</td><td>0.61</td></tr><tr><td colspan="8">Flow Matching Models</td></tr><tr><td>FLUX.1 Dev [5]</td><td>0.66</td><td>0.98</td><td>0.81</td><td>0.74</td><td>0.79</td><td>0.22</td><td>0.45</td></tr><tr><td>SD3.5-L [4]</td><td>0.71</td><td>0.98</td><td>0.89</td><td>0.73</td><td>0.83</td><td>0.34</td><td>0.47</td></tr><tr><td>SANA-1.5 4.8B [70]</td><td>0.81</td><td>0.99</td><td>0.93</td><td>0.86</td><td>0.84</td><td>0.59</td><td>0.65</td></tr><tr><td>SD3.5-M [4]</td><td>0.63</td><td>0.98</td><td>0.78</td><td>0.50</td><td>0.81</td><td>0.24</td><td>0.52</td></tr><tr><td>SD3.5-M+Flow-GRPO</td><td>0.95</td><td>1.00</td><td>0.99</td><td>0.95</td><td>0.92</td><td>0.99</td><td>0.86</td></tr></table>

![](images/3.jpg)  

Figure 3: Qualitative Comparison on the GenEval Benchmark. Our approach demonstrates superior performance in Counting, Colors, Attribute Binding, and Position.

离线变体使用固定的预训练模型进行数据收集，而在线变体则每40步更新其数据收集模型。如图4所示，Flow-GRPO的表现优于所有基线。在线DPO的性能也超越了其离线对应的结果，这与文献[15]一致。对于第二优秀的在线DPO，对其关键参数$\beta$的超参数搜索揭示出较小的值并不总是最佳，过小的$\beta$值可能导致训练崩溃。附录 $\textrm{C}$ 提供了更多涵盖额外方法和任务的全面比较。

Table 2: Performance on Compositional Image Generation, Visual Text Rendering, and Human Preference benchmarks, evaluated by task performance on test prompts, and by image quality and preference scores on DrawBench prompts. ImgRwd: ImageReward; UniRwd: UnifiedReward.   

<table><tr><td rowspan="2">Model</td><td colspan="3">Task Metric</td><td colspan="2">Image Quality</td><td colspan="3">Preference Score</td></tr><tr><td>GenEval</td><td>OCR Acc.</td><td>PickScore</td><td>Aesthetic</td><td>DeQA</td><td>ImgRwd</td><td>PickScore</td><td>UniRwd</td></tr><tr><td>SD3.5-M</td><td>0.63</td><td>0.59</td><td>21.72</td><td>5.39</td><td>4.07</td><td>0.87</td><td>22.34</td><td>3.33</td></tr><tr><td colspan="9">Compositional Image Generation</td></tr><tr><td>Flow-GRPO (w/o KL)</td><td>0.95</td><td></td><td></td><td>4.93</td><td>2.77</td><td>0.44</td><td>21.16</td><td>2.94</td></tr><tr><td>Flow-GRPO (w/KL)</td><td>0.95</td><td></td><td></td><td>5.25</td><td>4.01</td><td>1.03</td><td>22.37</td><td>3.51</td></tr><tr><td colspan="9">Visual Text Rendering</td></tr><tr><td>Flow-GRPO (w/o KL)</td><td></td><td>0.93</td><td></td><td>5.13</td><td>3.66</td><td>0.58</td><td>21.79</td><td>3.15</td></tr><tr><td>Flow-GRPO (w/KL)</td><td></td><td>0.92</td><td></td><td>5.32</td><td>4.06</td><td>0.95</td><td>22.44</td><td>3.42</td></tr><tr><td colspan="9">Human Preference Alignment</td></tr><tr><td>Flow-GRPO (w/o KL)</td><td></td><td></td><td>23.41</td><td>6.15</td><td>4.16</td><td>1.24</td><td>23.56</td><td>3.57</td></tr><tr><td>Flow-GRPO (w/ KL)</td><td></td><td></td><td>23.31</td><td>5.92</td><td>4.22</td><td>1.28</td><td>23.53</td><td>3.66</td></tr></table>

![](images/4.jpg)  

Figure 4: Comparison with Other Alignment Methods on the Compositional Generation Task.

![](images/5.jpg)  

Figure 5: Ablation Studies on Different Group Size $G$ Higher group size performs better.

# 5.3 分析

本节呈现了若干分析，以更好地理解Flow-GRPO的行为和鲁棒性。我们检视了奖励黑客、去噪减少与噪声水平的影响、群体规模的效应以及模型的泛化能力等问题。我们在附录C中提供了更多的分析。 奖励黑客。我们使用KL正则化来缓解奖励黑客，通过调整KL系数，使发散在训练过程中保持较小且近乎恒定，从而使模型保持接近其预训练权重。这允许针对特定任务的奖励优化，而不会损害整体性能。如表2所示，去除组合图像生成和视觉文本渲染的KL约束显著降低了DrawBench上的图像质量和偏好得分。相比之下，适当调整的KL可以在保持质量的同时，在特定任务的度量上获得相似的提升。在人类偏好对齐任务中，去除KL并不影响图像质量，可能是由于PickScore与评估指标之间的重叠，但却导致视觉多样性的崩溃。输出趋向于单一风格，不同随机种子生成几乎相同的结果。KL正则化防止了这种崩溃并维持了多样性。有关训练曲线，请参见附录C.5中的图12，更多示例见图6。 去噪减少的影响。图7(a)强调了去噪减少在加速训练方面的重要影响。为了探讨不同时间步长如何影响优化，这些实验是在没有KL约束的情况下进行的。将数据收集时间步长从40减少到10，使所有三个任务的速度提升超过$4 \times$，而不影响最终奖励。进一步减少到5并不能一致地提高速度，有时会减慢训练，因此我们为后续实验选择10个时间步长。对于其他两个任务，奖励与训练时间的学习曲线见附录C.2中的图9。

![](images/6.jpg)  

Figure 6: Effect of KL Regularization. The KL penalty effectively suppresses reward hacking preventing Quality Degradation (for GenEval and OCR) and Diversity Decline (for PickScore).

噪声水平的影响。在随机微分方程中，较高的 $\sigma _{t}$ 增强了图像多样性和探索性，这对于强化学习训练至关重要。我们通过噪声水平 $a$（方程 9）来控制这种探索。图 7 (b) 显示了 $a$ 对性能的影响。较小的 $a$（例如 0.1）限制了探索，并减缓了奖励的提高。增加 $a$（最高可达 0.7）会增强探索并加快奖励增长。超过这个点（例如，从 0.7 增加到 1.0），进一步增加将没有额外的好处，因为探索已经足够。我们还观察到，进一步增加 $a$ 也会导致注入过多噪声，降低图像质量，从而导致零奖励和训练失败。

![](images/7.jpg)  

Figure 7: Ablation studies on our critical design choices. (a) Denoising Reduction: Fewer denoising steps accelerate convergence and yield similar performance. (b) Noise Level: Moderate noise level b $a = 0 . 7$ ) maximises OCR accuracy, while too little noise hampers exploration.

群体规模的影响。图5展示了使用PickScore作为奖励函数时群体规模$G$的影响。当群体规模缩减到$G = 1$和$G = 6$时，训练变得不稳定并最终崩溃，而$G = 2$和$G = 4$在整个过程中保持稳定。我们观察到，较小的群体规模产生了不准确的优势估计，增加了方差并导致训练崩溃，这一现象也在文献[71, 72]中有所报告。泛化分析。Flow-GRPO在GenEval未见场景中表现出强泛化能力（表4）。具体而言，它能够捕捉物体的数量、颜色和空间关系，从$2$到$4$个物体进行泛化生成$5$、$6$或$12$个物体。此外，表3显示Flow-GRPO在T2I-CompBench $^{++}$ [6, 73]上取得了显著提升。该开放世界组成性T2I生成的综合基准测试涉及的物体类别和关系与我们模型的GenEval风格训练数据有显著差异。

Table 3: T2I-CompBench $^ { + + }$ Result. This evaluation uses the same model presented in Table 1, which was trained on the GenEval-generated dataset. The best score is inblue   

<table><tr><td>Model</td><td>Color</td><td>Shape</td><td>Texture</td><td>2D-Spatial</td><td>3D-Spatial</td><td>Numeracy</td><td>Non-Spatial</td></tr><tr><td>Janus-Pro-7B [69]</td><td>0.5145</td><td>0.3323</td><td>0.4069</td><td>0.1566</td><td>0.2753</td><td>0.4406</td><td>0.3137</td></tr><tr><td>EMU3 [67]</td><td>0.7913</td><td>0.5846</td><td>0.7422</td><td></td><td>—</td><td></td><td>—</td></tr><tr><td>FLUX.1 Dev [5]</td><td>0.7407</td><td>0.5718</td><td>0.6922</td><td>0.2863</td><td>0.3866</td><td>0.6185</td><td>0.3127</td></tr><tr><td>SD3.5-M [4]</td><td>0.7994</td><td>0.5669</td><td>0.7338</td><td>0.2850</td><td>0.3739</td><td>0.5927</td><td>0.3146</td></tr><tr><td>SD3.5-M+Flow-GRPO</td><td>0.8379</td><td>0.6130</td><td>0.7236</td><td>0.5447</td><td>0.4471</td><td>0.6752</td><td>0.3195</td></tr></table>

Table 4: Flow-GRPO demonstrates strong generalization. Unseen Objects: Trained on 60 object classes, evaluated on 20 unseen classes. Unseen Counting: Trained to render 2, 3, or 4 objects, and evaluated in two settings: rendering 5 or 6 objects, and rendering 12 objects.   

<table><tr><td rowspan="2">Method</td><td colspan="7">Unseen Objects</td><td colspan="2">Unseen Counting</td></tr><tr><td>Overall</td><td>Single Obj.</td><td>Two Obj.</td><td>Counting</td><td>Colors</td><td>Position</td><td>Attr. Binding</td><td>5-6 Objects</td><td>12 Objects</td></tr><tr><td>SD3.5-M</td><td>0.64</td><td>0.96</td><td>0.73</td><td>0.53</td><td>0.87</td><td>0.26</td><td>0.47</td><td>0.13</td><td>0.02</td></tr><tr><td>SD3.5-M+Flow-GRPO</td><td>0.90</td><td>1.00</td><td>0.94</td><td>0.86</td><td>0.97</td><td>0.84</td><td>0.77</td><td>0.48</td><td>0.12</td></tr></table>

# 6 结论

我们提出了Flow-GRPO，这是第一个将在线策略梯度强化学习整合到流匹配模型中的方法。通过将确定性常微分方程转化为随机微分方程，并在训练过程中减少去噪步骤，Flow-GRPO实现了高效的基于强化学习的优化，同时在图像质量或多样性上几乎没有显著妥协。我们的方法显著提高了组合生成、文本渲染和人类偏好对齐的性能，同时最小化了奖励黑客的影响。Flow-GRPO为将在线强化学习应用于基于流的生成模型提供了一个简单而通用的框架。 局限性与未来工作。虽然本工作侧重于图像到图像的任务，但Flow-GRPO在视频生成方面具有潜力，这引发了几个未来的研究方向：（1）奖励设计：简单的启发式方法，如使用物体检测器或追踪器作为基于规则的奖励，可以鼓励物理现实性和时间一致性，但仍需要更高级的奖励模型。（2）平衡多个奖励：视频生成需要优化多个目标，包括现实性、平滑性和一致性。平衡这些相互竞争的目标仍然具有挑战性，需要仔细调优。（3）可扩展性：视频生成比图像到图像的生成资源消耗更大，因此在大规模应用Flow-GRPO时需要更高效的数据收集和训练流程。此外，探索更好的防止奖励黑客的方法也值得关注。虽然KL正则化显著有助于改善情况，但它需要更长的训练时间，并且在某些提示下偶尔会出现奖励黑客现象。

# 致谢

本研究部分由香港马会慈善信托资助的JC STEM人工智能科学与工程实验室以及香港研究资助局（项目编号：CUHK14213224）支持。我们衷心感谢郑明武对证明的深刻讨论，以及周展辉对本论文清晰度提升的宝贵意见。

# References

[1] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:3647936494, 2022.   
[2] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.   
[3] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.   
[4] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.   
[5] Black Forest Labs. Flux. https://github. com/black-forest-labs/flux, 2024.   
[6] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. Advances in Neural Information Processing Systems, 36:7872378747, 2023.   
[7] Zhiyuan Yan, Junyan Ye, Weijia Li, Zilong Huang, Shenghai Yuan, Xiangyang He, Kaiqing Lin, He Cu He, n an. tal  ao gpt4o in image generation. arXiv preprint arXiv:2504.02782, 2025.   
[8] Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, and Furu Wei. Textdiffuser: Diffusion models as text painters. Advances in Neural Information Processing Systems, 36:9353 9387, 2023.   
[9] Richard S Sutton, Andrew G Barto, et al. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.   
[10] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.   
[11] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.   
[12] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.   
[13] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:5372853741, 2023.   
[14] Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Wenyu Qin, Menghan Xia, et al. Improving video generation with human feedback. arXiv preprint arXiv:2501.13918, 2025.   
[15] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Juncheng Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengchen Ma, et al. Skyreels-v2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025.   
[16] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.   
[17] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:5213252152, 2023.   
[18] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.   
[19] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pic-a-pic:An open dataset of user preferences for text-to-image generation.Advances in Neural Information Processing Systems, 36:3665236663, 2023.

[20] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[21] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:68406851, 2020.

[22] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

[23] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

[24] Kuaishou. Kling ai. https://klingai.kuaishou.com/, 2024.

[25] Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[26] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 1:8, 2024.

[27] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

[28] Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797, 2023.

[29] Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, and Ricky TQ Chen. Adjoint matching: Fine-tuning flow and diffusion generative models with memoryless stochastic optimal control. arXiv preprint arXiv:2409.08861, 2024.

[30] Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, and Katerina Fragkiadaki. Aligning textto-image diffusion models with reward backpropagation. arXiv preprint arXiv:2310.03739, 2023.

[31] Kevin Clark, Paul Vicol, Kevin Swersky, and David J Fleet. Directly fine-tuning diffusion models on differentiable rewards. arXiv preprint arXiv:2309.17400, 2023.

[32] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36, 2024.

[33] Mihir Prabhudesai, Russell Mendonca, Zheyang Qin, Katerina Fragkiadaki, and Deepak Pathak. Video diffusion alignment via reward gradients. arXiv preprint arXiv:2407.08737, 2024.

[34] Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019.

[35] Jiajun Fan, Shuaike Shen, Chaoran Cheng, Yuxin Chen, Chumeng Liang, and Ge Liu. Online reward-weighted fine-tuning of flow matching with wasserstein regularization. In The Thirteenth International Conference on Learning Representations, 2025.

[36] Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, and Shixiang Shane Gu." Aligning text-to-image models using human feedback. arXiv preprint arXiv:2302.12192, 2023.

[37] Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. Raft: Reward ranked finetuning for generative foundation model alignment. arXiv preprint arXiv:2304.06767, 2023.

[38] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.   
[39] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 82288238, 2024.   
[40] Kai Yang, Jian Tao, Jiafei Lyu, Chunjiang Ge, Jiaxin Chen, Weihan Shen, Xiaolong Zhu, and Xiu Li. Using human feedback to fine-tune diffusion models without any reward model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 89418951, 2024.   
[41] Zhanhao Liang, Yuhui Yuan, Shuyang Gu, Bohan Chen, Tiankai Hang, Ji Li, and Liang Zheng. Step-aware preference optimization: Aligning preference with denoising performance at each step. arXiv preprint arXiv:2406.04314, 2024.   
[42] Huizhuo Yuan, Zixiang Chen, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning of diffusion models for text-to-image generation. arXiv preprint arXiv:2402.10210, 2024.   
[43] Runtao Liu, Haoyu Wu, Zheng Ziqiang, Chen Wei, Yingqing He, Renjie Pi, and Qifeng Chen. Videodpo: Omni-preference alignment for video diffusion generation. arXiv preprint arXiv:2412.14167, 2024.   
[44] Jiacheng Zhang, Jie Wu, Weifeng Chen, Yatai Ji, Xuefeng Xiao, Weilin Huang, and Kai Han. Onlinevpo: Align video diffusion model with online video-centric preference optimization. arXiv preprint arXiv:2412.15159, 2024.   
[5] Hiroki Furuta, Heiga Zen, Dale Schuurans, Aleksandra Fust, Yutak Matso, Pery Lg, and Sherry Yang. Improving dynamic object interactions in text-to-video generation with ai feedback. arXiv preprint arXiv:2412.02617, 2024.   
[46] Zhanhao Liang, Yuhui Yuan, Shuyang Gu, Bohan Chen, Tiankai Hang, Mingxi Cheng, Ji Li, and Liang Zheng. Aesthetic post-training diffusion models from generic preferences with step-by-step preference optimization. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1319913208, 2025.   
[47] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.   
[48] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.   
[49] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammd Ghavazdeh, Kangwok Lee, and Kimin Lee. Reinforcement earning for fine-tuning text-to-image diffusion models. Advances in Neural Information Processing Systems, 36, 2024.   
[50] Shashank Gupta, Chaitanya Ahuja, Tsung-Yu Lin, Sreya Dutta Roy, Harrie Oosterhuis, Maarten de Rijke, and Satya Narayan Shukla. A simple and effective reinforcement learning method for text-to-image diffusion fine-tuning. arXiv preprint arXiv:2503.00897, 2025.   
[51] Zichen Miao, Jiang Wang, Ze Wang, Zhengyuan Yang, Lijuan Wang, Qiang Qiu, and Zicheng Liu. Training diffusion models towards diverse image generation with reinforcement learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1084410853, 2024.   
[52] Hanyang Zhao, Haoxian Chen, Ji Zhang, David D Yao, and Wenpin Tang. Score as action: Fine-tuning diffusion generative models by continuous-time reinforcement learning. arXiv preprint arXiv:2502.01819, 2025.   
[53] Po-Hung Yeh, Kuang-Huei Lee, and Jun-Cheng Chen. Training-free diffusion model alignment with sampling demons. arXiv preprint arXiv:2410.05760, 2024.   
[54] Zhiwei Tang, Jiangweizhi Peng, Jiasheng Tang, Mingyi Hong, Fan Wang, and Tsung-Hui Chang. Tuning-free alignment of diffusion models with direct noise optimization. arXiv preprint arXiv:2405.18881, 2024.   
[55] Jiaming Song, Qinsheng Zhang, Hongxu Yin, Morteza Mardani, Ming-Yu Liu, Jan Kautz, Yongxin Chen, and Arash Vahdat. Loss-guided diffusion models for plug-and-play controllable generation. In International Conference on Machine Learning, pages 3248332498. PMLR, 2023.   
[56] Xiaohui Sun, Ruitong Xiao, Jianye Mo, Bowen Wu, Qun Yu, and Baoxun Wang. F5r-tts: Improving flow matching based text-to-speech with group relative policy optimization. arXiv preprint arXiv:2504.02407, 2025.   
[57] Jaihoon Kim, Taehoon Yoon, Jisung Hwang, and Minhyuk Sung. Inference-time scaling for flow models via stochastic generation and rollover budget forcing. arXiv preprint arXiv:2503.19385, 2025.   
[58] Lixue Gong, Xiaoxia Hou, Fanshi Li, Liang Li, Xiaochen Lian, Fei Liu, Liyang Liu, Wei Liu, Wei Lu, Yichun Shi, et al. Seedream 2.0: A native chinese-english bilingual image generation foundation model. arXiv preprint arXiv:2503.07703, 2025.   
[59] Chrisoph Schuhmann. Laion aesthetics, Aug 2022.   
[60] Zhiyuan You, Xin Cai, Jinjin Gu, Tianfan Xue, and Chao Dong. Teaching large language models to regress accurate image quality scores using score distribution. arXiv preprint arXiv:2501.11561, 2025.   
[61] Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025.   
[62] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1068410695, 2022.   
[63] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.   
[64] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.   
[65] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Jun Zu Joyc Le Yu Guo l Ipoim e wh betapns. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.   
[66] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.   
[67] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.   
[68] Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Liang Zhao, et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. arXiv preprint arXiv:2411.07975, 2024.   
[69] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.   
[70] Enze Xie, Junsong Chen, Yuyang Zhao, Jincheng Yu, Ligeng Zhu, Yujun Lin, Zhekai Zhang, Muyang Li, Junyu Chen, Han Cai, et al. Sana 1.5: Efficient scaling of training-time and inference-time compute in linear diffusion transformer. arXiv preprint arXiv:2501.18427, 2025.   
[71] Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, and Yi Dong. Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models. arXiv preprint arXiv:2505.24864, 2025.   
[72] Yang Chen, Zhuolin Yang, Zihan Liu, Chankyu Lee, Peng Xu, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Acereason-nemotron: Advancing math and code reasoning through reinforcement learning. arXiv preprint arXiv:2505.16400, 2025.   
[73] Kaiyi Huang, Chengqi Duan, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2icompbench $^ { + + }$ An enhanced and comprehensive benchmark for compositional text-to-image generation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.   
[74] Bernt Øksendal and Bernt Øksendal. Stochastic differential equations. Springer, 2003.   
[75] Brian DO Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3):313326, 1982.   
[76] Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019.

# Appendix of Flow-GRPO: Training Flow Matching Models via Online RL

A Mathematical Derivations for Stochastic Sampling using Flow Models 17

# B Further Details on the Experimental Setup 18

B.1 Quality Metrics . . . 18   
B.2 Model Specification 19   
B.3 Hyperparameters Specification 19   
B.4 Compute Resources Specification 19

# C Extended Experimental Results 19

C.1 Flow-GRPO vs. Other Alignment Methods 19   
C.2 Effect of Denoising Reduction 21   
C.3 Effect of Initial Noise 21   
C.4 Additional Results on FLUX.1-Dev 22   
C.5 Learning Curves with or without KL 22   
C.6 Additional Qualitative Results 22   
C.7 Evolution of Evaluation Images During Flow-GRPO Training 22

# Training Sample Visualization with Denoising Reduction 22

Our Appendix consists of 4 sections. Readers can click on each section number to navigate to the corresponding section:

Section A provides detailed derivations of stochastic sampling in flow matching models.   
Section B presents details about our experimental setup.   
Section C offers some additional experimental results, including 1) the comparison with other alignment methods, 2) ablation of denoising reduction on OCR accuracy and pickscore, 3) ablation of initial noise, 4) additional results on FLUX.1-Dev, 5) the learning curves of FlowGRPO on three tasks, 6) additional qualitative results, and 7) evolution of evaluation images during training.   
Section D provides a visualization of training samples under the denoising reduction strategy.

In addition to this Appendix, we also provide more visualization results, see this website. We encourage the readers to consult this HTML page for a more intuitive assessment of the improvements brought by Flow-GRPO.

# A Mathematical Derivations for Stochastic Sampling using Flow Models

We present a detailed proof here. To compute $p _ { \theta } ( \pmb { x } _ { t - 1 } \mid \pmb { x } _ { t } , \pmb { c } )$ in Equation 5 during forward sampling, we adapt flow models to a stochastic differential equation (SDE). While flow models normally follow a deterministic ODE:

$$
\mathrm { d } \pmb { x } _ { t } = \pmb { v } _ { t } \mathrm { d } t
$$

We consider its stochastic counterpart. Inspired by the derivation from SDE to its probability flow ODE in SGMs [23], we aim to construct a forward SDE with specific drift and diffusion coefficients so that its marginal distribution matches that of Eq. 10. We begin with the generic form of SDE:

$$
\mathrm { d } \pmb { x } _ { t } = f _ { \mathrm { S D E } } ( \pmb { x } _ { t } , t ) \mathrm { d } t + \sigma _ { t } \mathrm { d } \pmb { w } ,
$$

Its marginal probability density $p _ { t } ( \pmb { x } )$ evolves according to the FokkerPlanck equation [74], i.e.,

$$
\partial _ { t } p _ { t } ( x ) = - \nabla \cdot [ f _ { \mathrm { S D E } } ( { \pmb x } _ { t } , t ) p _ { t } ( { \pmb x } ) ] + \frac { 1 } { 2 } \nabla ^ { 2 } [ \sigma _ { t } ^ { 2 } p _ { t } ( { \pmb x } ) ]
$$

Similarly, the marginal probability density associated with Eq. 10 evolves:

$$
\partial _ { t } p _ { t } ( { \pmb x } ) = - \nabla \cdot [ { \pmb v } _ { t } ( { \pmb x } _ { t } , t ) p _ { t } ( { \pmb x } ) ]
$$

To ensure that the stochastic process shares the same marginal distribution as the ODE, we impose:

$$
- \nabla \cdot [ f _ { \mathrm { S D E } } p _ { t } ( { \pmb x } ) ] + \frac { 1 } { 2 } \nabla ^ { 2 } [ \sigma _ { t } ^ { 2 } p _ { t } ( { \pmb x } ) ] = - \nabla \cdot [ { \pmb v } _ { t } ( { \pmb x } _ { t } , t ) p _ { t } ( { \pmb x } ) ]
$$

Observing that

$$
\begin{array} { r l } & { \nabla ^ { 2 } [ \sigma _ { t } ^ { 2 } p _ { t } ( { \pmb x } ) ] = \sigma _ { t } ^ { 2 } \nabla ^ { 2 } p _ { t } ( { \pmb x } ) } \\ & { \qquad = \sigma _ { t } ^ { 2 } \nabla \cdot ( \nabla p _ { t } ( { \pmb x } ) ) } \\ & { \qquad = \sigma _ { t } ^ { 2 } \nabla \cdot ( p _ { t } ( { \pmb x } ) \nabla \log p _ { t } ( { \pmb x } ) ) } \end{array}
$$

Substituting Eq. 15 to Eq. 14, we arrive at the drift coefficients of the target forward SDE:

$$
f _ { \mathrm { S D E } } = \boldsymbol { v } _ { t } ( \boldsymbol { x } _ { t } , t ) + \frac { \sigma _ { t } ^ { 2 } } { 2 } \nabla \log p _ { t } ( \boldsymbol { x } )
$$

Hence, we can rewrite the forward SDE in Eq. 11 as:

$$
\mathrm { d } { \pmb x } _ { t } = \bigg ( { \pmb v } _ { t } ( { \pmb x } _ { t } ) + \frac { \sigma _ { t } ^ { 2 } } { 2 } \nabla \log p _ { t } ( { \pmb x } _ { t } ) \bigg ) \mathrm { d } t + \sigma _ { t } \mathrm { d } { \pmb w } ,
$$

where dw denotes Wiener process increments, and $\sigma _ { t }$ is the diffusion coefficient controlling the level of stochasticity during sampling.

The relationship between forward and reverse-time SDEs has been established in [75, 23]. Specifically, if the forward SDE takes the form

$$
\mathrm { d } \pmb { x } _ { t } = f ( \pmb { x } _ { t } , t ) \mathrm { d } t + g ( t ) \mathrm { d } \pmb { w } ,
$$

then the corresponding reverse-time SDE is

$$
\mathrm { d } \pmb { x } _ { t } = \left[ f ( \pmb { x } _ { t } , t ) - g ^ { 2 } ( t ) \nabla \log p _ { t } ( \pmb { x } _ { t } ) \right] \mathrm { d } t + g ( t ) \mathrm { d } \overline { { \pmb { w } } } .
$$

Setting $g ( t ) = \sigma _ { t }$ , we obtain the reverse-time SDE corresponding to Eq. 17 as

$$
\mathrm { d } \pmb { x } _ { t } = \bigg [ \pmb { v } _ { t } ( \pmb { x } _ { t } ) + \frac { \sigma _ { t } ^ { 2 } } { 2 } \nabla \log p _ { t } ( \pmb { x } _ { t } ) - \sigma _ { t } ^ { 2 } \nabla \log p _ { t } ( \pmb { x } _ { t } ) \bigg ] \mathrm { d } t + \sigma _ { t } \mathrm { d } \pmb { \overline { { w } } } .
$$

We thus arrive at the final form of the reverse-time SDE:

$$
\boxed { \mathrm { d } \pmb { x } _ { t } = \left( \pmb { v } _ { t } ( \pmb { x } _ { t } ) - \frac { \sigma _ { t } ^ { 2 } } { 2 } \nabla \log p _ { t } ( \pmb { x } _ { t } ) \right) \mathbf { d } t + \sigma _ { t } \mathbf { d } \pmb { w } , }
$$

Once the score function $\nabla \log p _ { t } ( { \pmb x } _ { t } )$ is available, the process can be simulated directly. For flow matching, this score is implicitly linked to the velocity field ${ \mathbf { } } v _ { t }$ .

Specifically, let $\dot { \alpha _ { t } } \equiv \partial \alpha _ { t } / \partial t$ . All expectations are over $x _ { 0 } \sim X _ { 0 }$ and $\pmb { x } _ { 1 } \sim \mathcal { N } ( 0 , \pmb { I } )$ , where $X _ { 0 }$ is the data distribution.

For the linear interpolation ${ \pmb x } _ { t } = \alpha _ { t } { \pmb x } _ { 0 } + \beta _ { t } { \pmb x } _ { 1 }$ , we have:

$$
p _ { t | 0 } ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) = \mathcal { N } \left( \pmb { x } _ { t } \ | \ \alpha _ { t } \pmb { x } _ { 0 } , \beta _ { t } ^ { 2 } \pmb { I } \right) ,
$$

yielding the conditional score:

$$
\nabla \log p _ { t | 0 } ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) = - \frac { \pmb { x } _ { t } - \alpha _ { t } \pmb { x } _ { 0 } } { \beta _ { t } ^ { 2 } } = - \frac { \pmb { x } _ { 1 } } { \beta _ { t } } .
$$

The marginal score becomes:

$$
\begin{array} { r l } & { \nabla \log p _ { t } ( \pmb { x } _ { t } ) = \mathbb { E } \left[ \nabla \log p _ { t | 0 } ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) \mid \pmb { x } _ { t } \right] } \\ & { \qquad = - \displaystyle \frac { 1 } { \beta _ { t } } \mathbb { E } [ \pmb { x } _ { 1 } \mid \pmb { x } _ { t } ] . } \end{array}
$$

For the velocity field ${ \pmb v } _ { t } ( { \pmb x } _ { t } )$ , we derive:

$$
\begin{array} { l } { { v _ { t } ( x ) = \mathbb { E } \left[ \dot { \alpha } _ { t } { x _ { 0 } } + \dot { \beta } _ { t } { x _ { 1 } } \mid { x _ { t } } = x \right] } } \\ { { \ = \dot { \alpha } _ { t } \mathbb { E } [ { x _ { 0 } } \mid { x _ { t } } = x ] + \dot { \beta } _ { t } \mathbb { E } [ { x _ { 1 } } \mid { x _ { t } } = x ] } } \\ { { \ = \dot { \alpha } _ { t } \mathbb { E } \left[ \frac { { x _ { t } } - \dot { \beta } _ { t } { x _ { 1 } } } { \alpha _ { t } } \mid { x _ { t } } = x \right] + \dot { \beta } _ { t } \mathbb { E } [ { x _ { 1 } } \mid { x _ { t } } = x ] } } \\ { { \ = \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } { x - \frac { \dot { \alpha } _ { t } \beta _ { t } } { \alpha _ { t } } \mathbb { E } [ { x _ { 1 } } \mid { x _ { t } } = x ] + \dot { \beta } _ { t } \mathbb { E } [ { x _ { 1 } } \mid { x _ { t } } = x ] } } } \\ { { \ = \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } { x - \left( \dot { \beta } _ { t } { \beta _ { t } } - \frac { \dot { \alpha } _ { t } { \beta _ { t } ^ { 2 } } } { \alpha _ { t } } \right) \nabla \log { p _ { t } } ( x ) } , } } \end{array}
$$

Substituting $\alpha _ { t } = 1 - t$ and $\beta _ { t } = t$ simplifies Equation 25 to:

$$
\pmb { v } _ { t } ( \pmb { x } ) = - \frac { \pmb { x } } { 1 - t } - \frac { t } { 1 - t } \nabla \log p _ { t } ( \pmb { x } ) .
$$

Solving for the score yields:

$$
\nabla \log p _ { t } ( { \pmb x } ) = - \frac { { \pmb x } } { t } - \frac { 1 - t } { t } { \pmb v } _ { t } ( { \pmb x } ) .
$$

Substituting Equation 27 into 21 gives the final SDE:

$$
\mathrm { d } \pmb { x } _ { t } = \left[ \pmb { v } _ { t } ( \pmb { x } _ { t } ) + \frac { \sigma _ { t } ^ { 2 } } { 2 t } \left( \pmb { x } _ { t } + ( 1 - t ) \pmb { v } _ { t } ( \pmb { x } _ { t } ) \right) \right] \mathrm { d } t + \sigma _ { t } \mathrm { d } \pmb { w } .
$$

Applying Euler-Maruyama discretization yields the update rule:

$$
\left| x _ { t + \Delta t } = x _ { t } + \left[ v _ { \theta } ( x _ { t } , t ) + \frac { \sigma _ { t } ^ { 2 } } { 2 t } \big ( x _ { t } + ( 1 - t ) v _ { \theta } ( x _ { t } , t ) \big ) \right] \Delta t + \sigma _ { t } \sqrt { \Delta t } \epsilon , \right.
$$

where $\epsilon \sim \mathcal { N } ( 0 , I )$ injects stochasticity.

# B Further Details on the Experimental Setup

# B.1 Quality Metrics

The details of quality metrics are as follows:

Aesthetic score [59]: a CLIP-based linear regressor that predicts an image's aesthetic score.   
•DeQA score [60]: a multimodal large language model based image-quality assessment (IQA) model that quantifies how distortions, texture damage, and other low-level artefacts affect perceived quality.   
• ImageReward [32]: a general purpose T2I human preference reward model that captures textimage alignment, visual fidelity, and harmlessness.   
•UnifiedReward [61]: a recently proposed unified reward model for multimodal understanding and generation that currently achieves state-of-the-art performance on the human preference assessment leaderboard.

# B.2 Model Specification

The following table lists the base model and the reward models and their corresponding links.

<table><tr><td>Models</td><td>Links</td></tr><tr><td>SD3.5-M [4]</td><td>https://huggingface.co/stabilityai/stable-diffusion-3.5-medium</td></tr><tr><td>Aesthetic Score [59]</td><td>https://github.com/LAION-AI/aesthetic-predictor</td></tr><tr><td>PickScore [19]</td><td>https://huggingface.co/yuvalkirstain/PickScore_v1</td></tr><tr><td>DeQA score [60]</td><td>https://huggingface.co/zhiyuanyou/DeQA-Score-Mix3</td></tr><tr><td>ImageReward [32]</td><td>https://huggingface.co/THUDM/ImageReward</td></tr><tr><td>UnifiedReward [61]</td><td>https://huggingface.co/CodeGoat24/UnifiedReward-7b-v1.5</td></tr></table>

# B.3 Hyperparameters Specification

Except for $\beta$ , GRPO hyperparameters are fixed across tasks. We use a sampling timestep $T = 1 0$ and an evaluation timestep $T = 4 0$ .Other settings include a group size $G = 2 4$ , an noise level $a = 0 . 7$ and an image resolution of 512. The KL ratio $\beta$ is set to 0.04 for GenEval and Text Rendering, and 0.01 for Pickscore. We use Lora with $\alpha = 6 4$ and $r = 3 2$ .

# B.4 Compute Resources Specification

We train our model using 24 NVIDIA A800 GPUs. The learning curves in Appendix C.5 provide details on the specific GPU hours.

# C Extended Experimental Results

# C.1 Flow-GRPO vs. Other Alignment Methods

We compare Flow-GRPO with several alignment methods: supervised fine-tuning (SFT), rewardweighted regression (Flow-RWR [14, 76]), Flow-DPO [14], and their online variants. Flow-GRPO consistently outperforms all baselines by a significant margin. At each step, we generate a group of images using the same group size as in Flow-GRPO. The only difference lies in the update rule:

SFT: Select the highest-reward image in each group and fine-tune on it.   
Flow-RWR [14, 76]: Apply a softmax over rewards in each group and perform reward-weighted likelihood maximization.   
Flow-DPO [14, 39]: Use the highest-reward image in each group as the chosen sample and the lowest as the rejected, then apply the DPO loss.

Offine variants use a fixed pretrained model for data collection, while online variants update their data collection model every 40 steps. As shown in Figure 8, Flow-GRPO outperforms all other methods. The figure also indicates that DPO and SFT improve over time. In contrast, RWR does not, which aligns with experimental findings on RWR in [12]. Additionally, Online DPO surpasses offline DPO, aligning with [15]'s finding that online DPO performs better. For the second-best online DPO, a hyperparameter search on its key parameter $\beta$ revealed that smaller values are not always optimal; excessively small $\beta$ values can cause training collapse.

![](images/8.jpg)  
Figure 8: Comparison of Flow-GRPO and Other Alignment Methods on the Human Preference Alignment task. Since methods like DPO use different tuned batch sizes from Flow-GRPO, we use the number of training prompts on the $\mathbf { X }$ -axis for a fair comparison across these methods.

DDPO. DDPO [12] was originally developed for diffusion-based backbones, so we adapted it to flow-matching models via our ODE-to-SDE conversion. Using SD3.5-M as the base model and PickScore as the reward signal, we track the evaluation reward throughout the entire training process in Figure 8. We find that DDPO's reward increases more slowly than Flow-GRPO's and eventually collapses in the later stages, whereas Flow-GRPO trains stably and continues to improve consistently over time.

ReFL. ReFL [32] directly fine-tunes diffusion models by viewing reward model scores as human preference losses and back-propagating gradients to a randomly-picked late timestep $t$ Following ImageReward [32], we back-propagate gradients to a randomly chosen late timestep $t \in [ 3 0 , 4 0 ]$ during denoising. Figure 8 shows that GRPO surpasses ReFL when the reward is differentiable, indicating that GRPO maintains strong performance in settings where ReFL applies. More importantly, GRPO does not require differentiable rewards, enabling direct use of state-of-the-art Vision-Language Models (VLMs) as reward providers. This offers two key advantages:

•Sophisticated, General-Purpose Rewards: VLMs can conduct human-like evaluations through a structured reasoning process. Given a prompt, a VLM can decompose it into key criteria, reason step by step to verify each aspect in the generated image, and then provide a comprehensive overall score. This enables a single, unified reward model to handle diverse tasks, from text-to-image generation to complex instruction-based image editing.

•Future-Proof and Cost-Free Upgrades: The field of VLMs is advancing at a breathtaking pace. By using a VLM as the reward source, our framework automatically benefits from these improvements. As VLMs become more capable, the reward model becomes stronger without any additional training data or computational cost.

ORW. ORW [35] is an online reward-weighted regression method that guides the model to prioritize high-reward regions. Unlike KL regularization, it employs Wasserstein-2 regularization to prevent policy collapse and maintain diversity. To ensure a fair comparison, we adopt the same experimental setup as in our Human Preference Alignment task. For ORW, we set $\beta = 0 . 5$ and $\alpha = 1$ (lower values led to unstable training). The steps_per_epoch parameter, which controls how frequently the data-collecting policy is updated, was chosen from 20, 40, 100, 400 based on best performance. Table 5 reports reward scores on the test set across training steps. Following ORW's Table 1, we randomly sampled 50 DrawBench prompts and generated 64 images per prompt to compute CLIP and Diversity scores. As shown in Table 6, Flow-GRPO outperforms ORW on both metrics.

Table 5: Reward scores on the test set over training steps.   

<table><tr><td>Method</td><td>Step 0</td><td>Step 240</td><td>Step 480</td><td>Step 720</td><td>Step 960</td></tr><tr><td>SD3.5-M + ORW</td><td>28.79</td><td>29.05</td><td>29.15</td><td>27.58</td><td>23.05</td></tr><tr><td>SD3.5-M + Flow-GRPO</td><td>28.79</td><td>29.10</td><td>29.17</td><td>29.51</td><td>29.89</td></tr></table>

Table 6: Comparison of CLIP and diversity scores across different fine-tuning methods.   

<table><tr><td>Method</td><td>CLIP Score ↑</td><td>Diversity Score ↑</td></tr><tr><td>SD3.5-M</td><td>27.99</td><td>0.96</td></tr><tr><td>SD3.5-M + ORW</td><td>28.40</td><td>0.97</td></tr><tr><td>SD3.5-M + Flow-GRPO</td><td>30.18</td><td>1.02</td></tr></table>

# C.2 Effect of Denoising Reduction

We show the extended Denoising Reduction ablations of Visual Text Rendering and Human Preference Alignment tasks in Figure 9.

![](images/9.jpg)  
Figure 9: Effect of Denoising Reduction

# C.3 Effect of Initial Noise

We initialize each rollout with difference random noise to increase exploratory diversity during RL training. We perform an additioanl ablation to confirm this claim. With SD3.5-M as the base model and PickScore as the reward, we compare Flow-GRPO with different initial noise against Flow-GRPO with the same initial noise. Figure 10 shows the variant with different noise consistently achieved high rewards during the training process.

![](images/10.jpg)  
Figure 10: Effect of Initial Noise

![](images/11.jpg)  
Figure 11: Additional Results on FLUX.1-Dev

# C.4 Additional Results on FLUX.1-Dev

We run Flow-GRPO on FLUX.1-Dev [5] using PickScore as the reward signal. The reward curve rises steadily throughout training without noticeable reward hacking. Figure 11 shows the reward values over the training process, and Table 7 compares FLUX.1-Dev with FLUX.1-Dev $^ +$ Flow-GRPO on DrawBench.

Table 7: Comparison of FLUX.1-Dev and Flow-GRPO fine-tuned models.   

<table><tr><td>Model</td><td>Aesthetic</td><td>DeQA</td><td>ImageReward</td><td>PickScore</td><td>UnifiedReward</td></tr><tr><td>FLUX.1-Dev</td><td>5.71</td><td>4.31</td><td>0.85</td><td>22.62</td><td>3.65</td></tr><tr><td>FLUX.1-Dev + Flow-GRPO</td><td>6.02</td><td>4.24</td><td>1.32</td><td>23.97</td><td>3.81</td></tr></table>

# C.5 Learning Curves with or without KL

Figure 12 shows learning curves for three tasks, with and without KL. These results emphasize that KL regularization is not empirically equivalent to early stopping. Adding appropriate KL can achieve the same high reward as the KL-free version and maintain image quality, though it requires longer training.

# C.6 Additional Qualitative Results

Figures 13, 14 & 15 qualitatively compare SD3.5-M with its Flow-GRPO enhanced versions (with and without KL regularization) using GenEval, OCR and PickScore rewards, respectively. FlowGRPO with KL regularization improves the target capability while maintaining image quality and minimizing reward-hacking. Conversely, removing the KL constraint significantly degrades image quality and diversity.

# C.7 Evolution of Evaluation Images During Flow-GRPO Training

To better understand the training dynamics of our proposed Flow-GRPO framework, we visualize the evolution of generated samples corresponding to fixed evaluation prompts at regular intervals during training in Figure 16, 17 & 18. For consistency, all visualizations are produced using a 40-step ODE-based sampling schedule. These qualitative results provide a visual representation of how the model progressively improves its generation quality and alignment with task objectives over time.

# D Training Sample Visualization with Denoising Reduction

In this section, we compare images obtained with SDE sampling at various steps against those produced by ODE sampling, and offer an intuitive view of the denoising reduction strategy. Figure 19 presents SD3.5-Medium samples under four inference settings: (a) ODE sampling with 40 steps; (b) SDE sampling with 40 steps; () SDE sampling with 10 steps; (d) SDE sampling with 5 steps.

![](images/12.jpg)  
Figure 12: Learning Curves with and without KL. KL penalty slows early training yet effectively suppresses reward hacking.

The 40-step ODE and SDE runs yield visually indistinguishable images, confirming that our SDE sampler preserves quality. Shortening the SDE schedule to 10 and 5 steps introduces conspicuous artifacts, like color drift and fine details blur. Contrary to expectation that such low-quality samples might hinder optimization. it actually do just the opposite and accelerate optimization. Because Flow-GRPO relies on relative preferences, it still extracts a useful reward signal, while the shorter trajectories signifactly cut wall-clock time. Consequently, Flow-GRPO with denoising reduction strategy converges more quickly on both layout-oriented benchmarks such as GenEval and qualityfocused metrics such as PickScore, without sacrificing final performance.

![](images/13.jpg)  
Figure 13: Additional Qualitative comparison between the SD3.5-M and SD3.5-M $^ +$ Flow-GRPO trained with GenEval reward.

![](images/14.jpg)  
Flow-GRPO

![](images/15.jpg)  
Flow-GRPO(w/o KL)

![](images/16.jpg)

![](images/17.jpg)

futuristic buildings and greenery, with soft ambient lighting enhancing the futuristiatmosphere.

![](images/18.jpg)  
sunny sky.

![](images/19.jpg)

of desert landscape in the background.

![](images/20.jpg)  
beyond.

![](images/21.jpg)

![](images/22.jpg)  
surrounded by vibrant window displays and happy customers.   
Figure 14: Additional Qualitative comparison between the SD3.5-M and SD3.5-M $^ +$ Flow-GRPO trained with OCR reward.

![](images/23.jpg)  
Figure 15: Additional Qualitative comparison between the SD3.5-M and SD3.5-M $^ +$ Flow-GRPO trained with PickScore reward.

# Training Process on GenEval Task

![](images/24.jpg)  
a photo of a blue pizza and a yellow baseball glove.

Figure 16: We visualize the generated samples across successive training iterations during the optimization of SD3.5-Medium on the GenEval task.

# Training Process on OCR Task

![](images/25.jpg)  
a laboratory setting with a mouse cage prominently displayed. the cage label reads " caution: telepathic subjects " in bold letters, with a warning symbol. the environment is sterile and clcal emphasizing the unusual nature of the experiment.

![](images/26.jpg)  
a weathered cave explorer's journal page, with the phrase " lost city near" prominently written in faded ink, surrounded by sketches of ancient ruins and cryptic symbols, under a dim, mystical light.

![](images/27.jpg)

a realistic photograph of a fast food drive - thru menu board at dusk, featuring a bold and colorful advertisement that reads " try our new burger " with an appetizing image of the burger below, set against the backdrop of a busy suburban street.

Figure 17: We visualize the generated samples across successive training iterations during the optimization of SD3.5-Medium on the OCR task.

# Training Process on PickScore Task

![](images/28.jpg)  
a woman on top of a horse   
Figure 18: We visualize the generated samples across successive training iterations during the optimization of SD3.5-Medium on the PickScore task.

![](images/29.jpg)  
Figure 19: Visualization of training samples under difference inference settings.