# Iso-Dream：在世界模型中隔离和利用不可控的视觉动态

铸造 潘\* 项明朱\* 云博王† 晓康杨 人工智能部 MoE 关键实验室，上海交通大学 {panmt53, xmzhu76, yunbow, xkyang}@sjtu.edu.cn

# 摘要

世界模型学习基于视觉的交互系统中动作的后果。然而，在诸如自主驾驶等实际场景中，通常存在与动作信号无关的不可控动态，使得学习有效的世界模型变得困难。为了解决这个问题，我们提出了一种新的强化学习方法，称为Iso-Dream，该方法在两个方面改进了Dream-to-Control框架[22]。首先，通过优化逆动态，我们鼓励世界模型在孤立状态转移分支上学习可控和不可控的时空变化源。其次，我们优化智能体在世界模型的解耦潜在想象上的行为。具体而言， 为了估计状态值，我们将不可控状态推演到未来并将其与当前可控状态关联。通过这种方式，动态源的隔离可以大大有利于智能体的长期决策，例如一辆能够通过预测其他车辆运动来避免潜在风险的自驾车。实验表明，Iso-Dream在解耦混合动态方面有效，并在广泛的视觉控制和预测领域显著优于现有方法。

# 1 引言

人类通过观察和与环境互动，可以推断和预测真实世界的动态。受到此启发，许多前沿的人工智能智能体采用自监督学习或强化学习技术从周围环境中获取知识。其中，世界模型在机器人视觉控制领域受到了广泛关注，并推动了基于模型的强化学习的最新进展。一个典型的方法是使用强化学习智能体收集的观察和控制信号的轨迹，学习环境的可微分模拟器，即世界模型，然后通过优化世界模型的潜在想象中的行为来更新强化学习智能体。然而，由于观察序列是高维的、非平稳的，并且通常受到多种物理动态源的驱动，如何在复杂视觉场景中学习有效的世界模型仍然是一个未解决的问题。在自主驾驶等现实场景中，我们通常可以将系统中的时空动态分为两个部分：可控部分，它们完全响应动作信号；以及不可控部分，例如其他车辆的运动和其他外部变化。可控状态和不可控状态的区分可以在两个方面提升基于模型的强化学习：模块化表示提高了智能体在具有噪声的非平稳环境中的泛化能力，例如我们修改后的DeepMind Control Suite中的时变背景。

![](images/1.jpg)  

Figure 1: Probabilistic graph of Iso-Dream. It learns to decouple complex visual dynamics into controllable states $\left( { { s _ { t } } } \right)$ and noncontrollable states $\left( { z _ { t } } \right)$ by optimizing the inverse dynamics (Red dashed arrows). On top of the disentangled states, it performs model-based reinforcement learning by explicitly considering the predicted noncontrollable component of future dynamics (Blue arrows).

更重要的是，它提升了在长期强化学习任务中基于对未来不可控动态预测做出决策的优势。例如，在自动驾驶中，通过预测其他车辆的运动可以更好地避免潜在风险。我们提出了Iso-Dream，一种新颖的模型基础强化学习框架，能够学习解耦和利用可控和不可控状态转移。因此，它从两个方面改进了原始的Dreamer：[22] (i) 一种新的世界模型表示形式以及 (ii) 一种新的演员-评论家算法，从世界模型中推导行为。如图1所示，解耦世界模型的基础是将混合潜在状态分离为一个基于行动的分支和一个不依赖于行动的分支，这两个分支可以独立地传递不同来源的视觉动态。这些组件共同训练，以最大化变分下界。为了进一步隔离可控状态，基于行动的分支还通过逆动态进行优化，即推理推动相邻时间步之间状态转移的动作。Iso-Dream的另一个贡献是发现解开物理动态可以通过更准确地预测环境中的固有变化，极大地惠及下游决策任务。直观地说，人类可以根据对周围未来变化的预期，决定在每个时刻如何与环境互动。为了做出更具前瞻性的决策，如图1中的蓝色箭头所示，策略网络通过注意力机制集成当前的可控状态和多个步骤的预测不可控状态。这使得智能体能够全面考虑与环境可能的未来交互。我们在以下领域评估Iso-Dream：修改后的DeepMind Control Suite，带有噪声视频背景；CARLA自动驾驶环境，其中其他车辆可以自然地视为不可控组件；现实世界的BAIR机器人数据集和RoboNet数据集，这些数据集有助于验证世界模型在解耦中的有效性。在所有基准测试中，Iso-Dream均显著优于现有方法。

# 2 相关工作

基于动作的视频预测。解决视觉控制问题的一种直接深度学习方案是学习基于动作的视频预测模型 [38, 14, 8, 53]，然后对可用行为执行蒙特卡洛重要性采样和优化算法，例如交叉熵方法 [15, 12, 29]。视频预测的热门话题主要包括长期和高保真未来帧生成 [44, 43, 51, 5, 52, 50, 54, 41, 40, 36, 56, 28, 2]、动态不确定性建模 [1, 10, 48, 31, 7, 16, 55]、以物体为中心的场景分解 [47, 27, 18, 58, 3] 以及时空解耦 [49, 27, 19, 6]。相应的技术改进主要涉及使用更有效的神经网络结构、新颖的概率建模方法以及具体形式的视频表示。解耦方法与Iso-Dream中的世界模型密切相关。它们通常将视觉动态分为内容和运动向量，或长期和短期状态。相比之下，Iso-Dream旨在学习一个基于可控性的解耦世界模型，这对下游行为学习过程贡献更大。

视觉MBRL。在视觉控制任务中，智能体需要直接从高维观测中学习动作策略。它们大致可分为两类，即无模型方法和基于模型的方法。其中，MBRL方法明确建模状态转移，并且通常比无模型方法具有更高的样本效率。Ha和Schmidhuber提出的世界模型首先以自监督的方式学习环境的压缩潜在状态，然后在由世界模型生成的潜在状态上训练智能体。遵循两阶段训练程序，PlaNet使用基于动作条件的递归状态空间模型（RSSM）作为世界模型，并在递归状态上使用交叉熵方法优化动作策略。在Dreamer和DreamerV2中，智能体通过优化RSSM中预测的潜在状态的期望值来学习行为。InfoPower优先考虑来自视觉观测的功能相关信息，以获得更强的MBRL表示。值得注意的是，Iso-Dream在两个方面与InfoPower非常不同。首先，我们明确建模可控和不可控动态的状态转移，从而可以根据特定领域的先验知识选择是否将不可控状态纳入行为学习。其次，我们提出了一种新的行为学习方法，极大地受益于解耦的世界模型，使我们能够在当前做出决策之前预览不可控模式的可能未来状态。

# 3 方法

在本节中，我们首先介绍Iso-Dream的基本假设与总体框架，以实现可控与不可控动态的解耦与利用（第3.1节）。对于表示学习，我们引入了三分支世界模型及其反向动态的训练目标（第3.2节）。在行为学习方面，我们提出了一种演员-评论家方法，该方法在解耦世界模型的潜在状态的想象中进行训练，使得智能体能够考虑不可控动态的可能未来状态（第3.3节）。最后，我们讨论了Iso-Dream如何部署以与环境进行交互（第3.4节）。

# 3.1 Iso-Dream的基本假设

如图1所示，当智能体接收到一系列视觉观察$O_{1:T}$时，潜在的时空动态可以定义为$u_{1:T}$。我们的目标是通过将$u_{1:T}$解耦为在时空中变化的可控制潜在状态$s_{1:T}$和不可控制潜在状态$z_{1:T}$，来理解不同动态之间的内在关系，使得：

$$
u _ { 1 : T } \sim ( s , z ) _ { 1 : T } , \quad s _ { t + 1 } \sim p ( s _ { t + 1 } \mid s _ { t } , a _ { t } ) , \quad z _ { t + 1 } \sim p ( z _ { t + 1 } \mid z _ { t } ) ,
$$

其中 $a _ { t }$ 是行动信号。为了实现长期预测，我们将 $s _ { t }$ 和 $z _ { t }$ 互相隔离，并分别建模其状态转移 $p ( s _ { t + 1 } \mid s _ { t } , \bar { a _ { t } } )$ 和 $\mathbf { \bar { \rho } } _ { p \left( z _ { t + 1 } \mid z _ { t } \right) }$。根据我们对环境的先验知识，我们可以选择是否将不可控状态推演并在行为学习中考虑它们。对于可被视为时间变化噪声的不可控组件的任务，我们简单地通过 $a _ { t } \sim \pi ( a _ { t } \mid s _ { t } )$ 来推导行动策略。可控状态的隔离提高了智能体对非平稳系统的泛化能力。对于如自动驾驶等任务，行为是通过计算 $s _ { t }$ 和想象的不可控状态在时间范围 $\tau$ 内的关系来推导的。它假设在特定的长时间任务中，智能体能够极大地受益于预测外部不可控力量的后果。

$$
a _ { t } \sim \pi ( a _ { t } \mid s _ { t } , z _ { t : t + \tau } ) ,
$$

# 3.2 可控与非可控动态的表征学习

受到之前研究的启发 [37, 17]，这些研究表明模块化结构在解耦学习中是有效的，我们利用三分支架构将 $u _ { t }$ 解耦为可控动态状态 $s _ { t }$、不可控动态状态 $z _ { t }$ 和背景的时间不变表示。如图 2(a) 所示，基于动作的分支建模 $p ( s _ { t + 1 } \mid s _ { t } , a _ { t } )$。它遵循 PlaNet [23] 的 RSSM 架构，使用递归神经网络 $\mathtt { G R U } _ { s } ( \cdot )$、确定性隐藏状态 $h _ { t }$ 和随机状态 $s _ { t }$ 来形成过渡模型，其中 GRU 保持可控动态的历史信息。无动作分支用类似的网络结构建模 $p ( z _ { t + 1 } \mid z _ { t } )$。具有独立参数的过渡模型可以写成如下形式：

$$
\begin{array} { r l } & { p ( \widetilde { s } _ { t } \mid s _ { < t } , a _ { < t } ) = p ( \widetilde { s } _ { t } \mid h _ { t } ) , \quad \mathrm { w h e r e } h _ { t } = \mathtt { G R U } _ { s } ( h _ { t - 1 } , s _ { t - 1 } , a _ { t - 1 } ) , } \\ & { \qquad p ( \widetilde { z } _ { t } \mid z _ { < t } ) = p ( \widetilde { z } _ { t } \mid h _ { t } ^ { \prime } ) , \quad \mathrm { w h e r e } h _ { t } ^ { \prime } = \mathtt { G R U } _ { z } ( h _ { t - 1 } ^ { \prime } , z _ { t - 1 } ) . } \end{array}
$$

![](images/2.jpg)  

Figure 2: The overall architecture of the world model and the behavior learning algorithm in IsoDream. (a) World model with three branches to explicitly disentangle controllable, noncontrollable, and static components from visual data, where the action-conditioned branch learns controllable state transitions by modeling inverse dynamics. (b) The agent optimizes the behaviors in imaginations of the world model through a future state attention mechanism.

我们在此使用 $\tilde { s } _ { t }$ 和 $\tilde { z } _ { t }$ 来表示先验表示。我们利用从 $s _ { t } \sim q ( s _ { t } \mid h _ { t } , o _ { t } )$ 和 $z _ { t } \sim q ( z _ { t } \mid h _ { t } ^ { \prime } , o _ { t } )$ 得到的后验表示来优化转换模型。我们通过共享编码器 $\mathtt { E n c } _ { \theta }$ 和后续的特定分支编码器 $\mathtt { E n c } _ { \phi _ { 1 } }$ 和 $\mathtt { E n c } _ { \phi _ { 2 } }$ 学习 $\backslash \dot { o } _ { t } \in \mathbb { R } ^ { 3 \times H \times W }$。为了增强与控制信号对应的解耦表示学习，我们引入逆动力学的训练目标。因此，我们设计了一个2层多层感知器（MLP）的逆单元，以推断导致可控状态某些转换的动作：

$$
\tilde { \boldsymbol { a } } _ { t - 1 } = \mathtt { M L P } \big ( \boldsymbol { s } _ { t - 1 } , \boldsymbol { s } _ { t } \big ) ,
$$

输入是动作条件分支中的后验表示。通过学习回归真实行为 $a_{t-1}$，逆元胞促使动作条件分支孤立出可控动态的表示。为了避免训练崩溃的情况，即动作条件分支捕获了大部分有用信息，而无动作分支几乎没有学习到任何东西，在图像重建过程中，我们分别使用先验状态 $\tilde{s}_t$ 和后验状态 $z_t$ 来生成可控的视觉组件 $\hat{\sigma}_t^{s} \in \bar{\mathbb{R}}^{3 \times H \times W}$，以及掩模 $M_t^{s} \in \dot{\mathbb{R}}^{1 \times H \times W}$，同时生成 $\hat{o}_t^{z} \in \mathbb{R}^{3 \times H \times W}$ 与 $\backslash M_t^{z} \in \mathbb{R}^{1 \times H \times W}$。通过进一步整合从前 $K$ 帧提取的时间不变信息，我们得到了

$$
\hat { o } _ { t } = M _ { t } ^ { s } \odot \hat { o } _ { t } ^ { s } + M _ { t } ^ { z } \odot \hat { o } _ { t } ^ { z } + ( 1 - M _ { t } ^ { s } - M _ { t } ^ { z } ) \odot \hat { o } ^ { b } , \quad \mathrm { w h e r e ~ } \hat { o } ^ { b } = \mathtt { D e c } _ { \varphi _ { 3 } } \big ( \mathrm { E n c } _ { \theta , \phi _ { 3 } } \big ( o _ { 1 : K } \big ) \big ) \big ) .
$$

对于奖励建模，我们有两个与无动作分支相关的选项。在一种情况下，无法控制的动态可以被视为与任务无关的噪声，因此 $z_{t}$ 在想象过程中不再有用。换句话说，策略和预测的奖励仅与可控状态相关。在另一种情况下，未来不可控状态会影响智能体的决策方式，我们在行为学习中考虑无动作组件。为此，我们学习替代的奖励模型 $p(r_{t} \mid s_{t})$ 或 $p(r_{t} \mid s_{t}, {\bar{z}}_{t})$，形式为多层感知机（MLP）。对于从重放缓冲区在训练期间采样的序列 $(o_{t}, a_{t}, r_{t})_{t=1}^{T}$，世界模型可以使用以下损失函数进行优化，其中 $\alpha$，$\beta_{1}$ 和 $\beta_{2}$ 是超参数：

$$
\begin{array} { r l } & { \mathcal { L } = \mathrm { E } \{ \displaystyle \sum _ { t = 1 } ^ { T } \frac { - \ln p \left( o _ { t } \mid h _ { t } , s _ { t } , h _ { t } ^ { \prime } , z _ { t } \right) } { \mathrm { i n a g e l o s s } } \frac { - \ln p \left( r _ { t } \mid h _ { t } , s _ { t } , h _ { t } ^ { \prime } , z _ { t } \right) } { \mathrm { r e w a r d l o g ~ l o s s } } \frac { - \ln p \left( \gamma _ { t } \mid h _ { t } , s _ { t } , h _ { t } ^ { \prime } , z _ { t } \right) } { \mathrm { d i s c o u n t l o g l o s s } } } \\ & { \quad \quad + \underbrace { \alpha \ell _ { 2 } ( a _ { t } , \tilde { a } _ { t } ) } _ { \mathrm { a c t i o n ~ l o s s } } + \underbrace { \beta _ { 1 } \mathrm { K L } [ q ( s _ { t } \mid h _ { t } , o _ { t } ) \mid p ( s _ { t } \mid h _ { t } ) ] + \beta _ { 2 } \mathrm { K L } [ q ( z _ { t } \mid h _ { t } ^ { \prime } , o _ { t } ) \mid p ( z _ { t } \mid h _ { t } ^ { \prime } ) ] } _ { \mathrm { K L 4 i v e r g e n c e } } \} . } \end{array}
$$

Aunu 1；i-Dcam（mmmt. Uu mumal t ao & pm uement）1 超参数：$L$：想象范围；$\tau$：未来状态注意力的窗口大小。2 用随机序列初始化重放缓冲区 $\boldsymbol { B }$。3 当未收敛时，执行以下操作：4 对于更新步骤 $c = 1 \ldots C$，执行以下操作：5 从 $\mathcal { B }$ 中抽取数据序列 $\left\{ \left( o _ { t } , a _ { t } , r _ { t } \right) \right\} _ { t = 1 } ^ { T }$。6 // 表示学习 7 使用公式（6）计算世界模型损失并更新模型参数。8 // 行为学习 9 从 $z _ { t }$ 中推演 $\tilde { z } _ { i }$ 的轨迹 $\left\{ \tilde { z } _ { i } \right\} _ { i = t + 1 } ^ { t + L + \tau }$。10 对于时间步 $j = i \ldots i + L$，执行以下操作：11 使用公式（7）计算潜在状态 $e _ { j } \sim$ Attention $( \tilde { s } _ { j } , \tilde { z } _ { j : j + \tau } )$。12 想象一个动作 $a _ { j } \sim \pi ( a _ { j } | e _ { j } )$。13 仅使用基于动作的分支预测下一个可控状态 $\tilde { s } _ { j + 1 } \sim p ( \tilde { s } _ { j } , a _ { j } )$。14 结束。15 使用估计的奖励和价值更新公式（8）中的策略和值模型。16 结束。17 // 环境交互 18 $o _ { 1 }$ env.reset()。19 对于时间步 $t = 1 \dots T$，执行以下操作：20 计算后验表示 $s _ { t } \sim q \left( s _ { t } \mid h _ { t } , o _ { t } \right) , z _ { t } \sim q \left( z _ { t } \mid h _ { t } ^ { \prime } , o _ { t } \right)$。21 仅通过无动作分支推演不可控状态 $\tilde { z } _ { t + 1 : t + \tau }$。22 使用未来状态注意力生成 $a _ { t } \sim \pi ( a _ { t } \mid s _ { t } , z _ { t } , { \tilde { z } } _ { t + 1 : t + \tau } )$，参见公式（7）。23 $r _ { t } , o _ { t + 1 } \gets \mathsf { e n v }$.step $( a _ { t } )$。24 结束。25 将经验添加到重放缓冲区 $\boldsymbol { B } \gets \boldsymbol { B } \cup \{ ( o _ { t } , a _ { t } , r _ { t } ) _ { t = 1 } ^ { T } \}$。26 结束。世界模型训练方法可以部分定制以适应不同环境。在行为学习确实涉及不可控状态的情况下，最小化 ELBO 目标可以保持 $\tilde { z } _ { t }$ 的语义。否则，如果无动作特征仅用于防止干扰影响 Iso-Dream 的训练过程，而不是用于行为学习，我们可以仅使用重构损失来训练无动作分支。

# 3.3 在解耦想象中的行为学习

由于解耦的世界模型，我们可以优化智能体的行为，自适应地考虑可用动作与不可控动态的可能未来状态之间的关系。一实际例子是自动驾驶，其中其他车辆的运动可以自然地视为不可控但可预测的组成部分。如图2(b)所示，我们在此提出了一种改进的演员-评论家学习算法，该算法 1) 使无动作分支能够预见未来，而不是依赖于动作条件分支，2) 利用不可控动态的预测未来信息做出更具前瞻性的决策。假设我们在想象时期的时间步 $t$ 上做出决策。来自原始 Dreamer 方法的一个直接解决方案是基于孤立的可控状态 $\tilde { s } _ { t } \in \mathbb { R } ^ { 1 \times d }$ 学习一个动作模型和一个价值模型。然而，我们注意到通过采用注意力机制，我们可以获得 $\tilde { z } _ { t : t + \tau } \in \mathbb { R } ^ { \tau \times d }$，其中 $\tau$ 是从现在开始的滑动窗口长度。未来状态注意力为：$e _ { t } = \mathrm { s o f t m a x } \big ( \tilde { s } _ { t } \tilde { z } _ { t : t + \tau } ^ { T } \big ) \tilde { z } _ { t : t + \tau } + \tilde { s } _ { t }.$ 这样，$\tilde { s } _ { t }$ 演变为更“具前瞻性”的表示 $e _ { t } \in \mathbb { R } ^ { 1 \times d }$。我们对 Dreamer [22]中的动作模型和价值模型更新如下：

$$
\mathrm { A c t i o n \ m o d e l : } \quad a _ { t } \sim \pi ( a _ { t } \mid e _ { t } ) , \quad \mathrm { V a l u e \ m o d e l : } \quad v _ { \xi } ( e _ { t } ) \approx \mathbb { E } _ { \pi ( \cdot \mid e _ { t } ) } \sum _ { k = t } ^ { t + L } \gamma ^ { k - t } r _ { k } ,
$$

其中 $L$ 是想象时间范围。如算法 1 所示，在想象过程中，我们首先使用无动作的转换模型获取长度为 $L + \tau$ 的不可控状态序列，记作 $\{ \tilde { z } _ { i } \} _ { i = t } ^ { i + L + \tau }$。然后从隐含想象中预测下一个可控状态 $s _ { j + 1 }$。我们遵循 DreamerV2 [24] 的方法训练动作模型，以最大化 $\lambda$-回报 [45]，并训练价值模型以回归 $\lambda$-回报。

Table 1: Performance of visual control tasks in the DMC Suite. The agents are trained and evaluated in environments with video_easy dynamic background. We report the mean and std of final performance over 3 seeds and 5 trajectories. $\ast _ { \mathrm { W e } }$ use a different setup from that in the paper of DBC.   

<table><tr><td>TASK</td><td>SVEA</td><td>CURL</td><td>DBC*</td><td>DreameRV2</td><td>Iso-Dream</td></tr><tr><td>WAlkeR WALK</td><td>826 ± 65</td><td>443 ± 206</td><td>32 ± 7</td><td>655 ± 47</td><td>911 ± 50</td></tr><tr><td>CHEETAH RUN</td><td>178 ± 64</td><td>269 ± 24</td><td>15 ± 5</td><td>475 ± 159</td><td>659 ± 62</td></tr><tr><td>FINGER SPIN</td><td>562 ± 22</td><td>280 ± 50</td><td>1 ± 2</td><td>755 ± 92</td><td>800 ± 59</td></tr><tr><td>HOPPER STAND</td><td>6 ± 8</td><td>451 ± 250</td><td>5 ± 9</td><td>260 ± 366</td><td>746 ± 312</td></tr></table>

# 3.4 通过推演不可控动态进行策略部署

如上所述，在非可控动态与控制任务无关的情况下，与环境交互时，我们仅使用可控动态的状态在每个时间步 $t$ 生成策略。然而，对于非可控动态应与智能体行为密切相关的情况，如算法1的第21-22行所示，无动作分支连续预测从当前后验状态 $z _ { t }$ 开始的下一个 $\tau - 1$ 个非可控状态 $\tilde { z } _ { t + 1 : t + \tau }$。与行为学习过程中的公式 (7) 类似，这里我们使用学习到的未来状态注意力网络自适应整合 $s _ { t } , z _ { t }$ 和 $\tilde { z } _ { t + 1 : t + \tau }$。基于整合特征 $e _ { t }$，Iso-Dream 智能体从动作模型中抽取 $a _ { t }$ 来与环境进行交互。

# 4 实验

# 4.1 实验设置

基准测试。我们在两个强化学习环境中对Iso-Dream进行定量和定性评估，即DeepMind Control Suite和CARLA，以及两个针对动作条件视频预测的真实数据集，即BAIR机器人推送和RoboNet。视频预测实验能够提供更直观的解耦学习可视化。比较方法。在视觉控制任务中，我们将我们的方法与五个基线进行比较，包括基于模型和无模型的方法，即DreamerV2、CURL、SVEA、SAC和DBC。在动作条件视频预测中，我们主要将我们解耦的世界模型与三种方法进行比较，即SVG、SA-ConvLSTM和PhyDNet。

# 4.2 DeepMind 控制套件

实现细节。为了验证通过解耦不同组件在复杂视觉动态下对 Iso-Dream 的增强效果，我们在 DMC 泛化基准的环境中评估 Iso-Dream。不同于在原始 DeepMind 控制套件环境中训练，智能体在带有自然视频背景（即 video_easy 环境）下进行训练和测试。在此环境中，由于背景被随机替换为真实世界的视频，背景的不可控运动可能会影响智能体的动态学习和行为学习过程。因此，为了获得更好的决策策略并避免嘈杂背景的干扰，智能体可能会在时空中解耦不可控表示（即动态背景）和可控表示，并仅使用可控表示进行控制。为此，我们简单地用仅重建损失训练无动作分支，并在想象和策略部署中丢弃该分支。我们在四个不同领域的四个任务中对我们的模型与基线进行评估。环境步骤的数量限制为 $5 0 0 \mathrm { k }$ 。定量结果。为了评估性能，我们在带有视频背景的环境中训练和测试智能体。如表 1 所示，Iso-Dream 在所有任务中的表现超过了 DreamerV2 和其他基线，表明三分支结构可以有效学习与任务相关的视觉表示，并减轻视觉数据中复杂背景的干扰。

![](images/3.jpg)  

Figure 3: Video prediction results on the DMC (left) and CARLA (right) benchmarks of Iso-Dream. For each sequence, we use the first 5 images as context frames. Iso-Dream successfully disentangles controllable and noncontrollable components.

定性结果。我们利用Iso-Dream完成视频预测任务于video_easy环境中。测试回合中帧和动作序列是随机收集的。模型接收前5帧，并仅根据动作输入预测接下来的45帧。为了展示定性结果，我们可视化了来自于动作条件分支和无动作分支的掩码和视觉解耦组件。整体可视化结果如图3（左）所示。从该预测结果中，我们可以发现Iso-Dream具有预测长期序列和从video_easy环境中的图像中解耦可控与不可控动态的能力。如图3中动作条件分支输出的第三和第四行所示，可控表征已成功隔离且与其掩码相匹配。此外，在此可视化中，背景视频中的无动作组件是海浪的运动，这在动作无条件分支输出的第五和第六行中得以体现。

# 4.3 CARLA 自动驾驶环境

实现细节。在自动驾驶任务中，我们在自我车辆的车顶使用一个视角为60度的摄像头，获取$6 4 \times 6 4$像素的图像。遵循DBC [59]中的设置，为了鼓励高速公路行驶并惩罚碰撞，奖励被制定为：$r _ { t } = v _ { e g o } ^ { T } \hat { u } _ { h } \cdot \Delta t - \bar { \xi } _ { 1 } \cdot \mathbb { I } - \bar { \xi } _ { 2 } \cdot \hat { | } s t e e r |$，其中$v _ { e g o }$投影到高速公路的单位向量$\hat { u } _ { h }$上，并乘以时间离散化$\Delta t = 0 . 0 5$以测量以米为单位的高速公路行驶进度。冲击$\mathbb { I } \in \bar { \mathbb { R } } ^ { + }$由碰撞引起，转向惩罚$s t e e r \in [ - 1 , 1 ]$有助于保持车道。超参数$\xi _ { 1 }$和$\xi _ { 2 }$分别设置为$1 0 ^ { - 4 }$和1。我们在公式(6)中使用$\beta _ { 1 } = 1$，$\beta _ { 2 } = 1$和$\alpha = 1$，在公式(7)中使用$\tau = 5$。

定量结果。如图 4(a) 所示，Iso-Dream 相较于其他基线表现出显著优势，并且大幅超过 DreamerV2。此外，我们还进行了消融实验，以确认逆动态建模和不可控状态的推演策略的有效性。图 $4 ( \mathbf { b } )$ 显示，当移除逆细胞时，性能下降，表明建模逆动态以将可控和不可控成分从整体动态中隔离的重要性。为了验证所提注意力机制的有效性，我们进行了实验来评估 Iso-Dream，其中策略网络直接将当前可控状态与不可控状态拼接作为输入。比较蓝色曲线和绿色曲线，我们观察到在无动作分支中推演不可控状态可以显著提高智能体的决策结果。红色曲线显示，在缺少捕捉静态信息的单独网络分支时，Iso-Dream 的性能下降约 $15\%$。

![](images/4.jpg)  

Figure 4: Performance with 3 seeds on the CARLA driving task. (a) Comparison of existing methods, in which Iso-Dream outperforms DreamerV2 by a large margin. (b) Ablation studies that can show the respective impact of optimizing the inverse dynamics (orange), rolling out noncontrollable states (green), and modeling the time-invariant information with a separate network branch (red).

Table 2: Video prediction results on BAIR and RoboNet datasets with bouncing balls. We use the first 2 frames as input to predict the next 28 frames on BAIR and the next 18 frames on RoboNet.   

<table><tr><td rowspan="2">MODEL</td><td colspan="2">BAIR</td><td colspan="2">RoboNET</td></tr><tr><td>PSNR ↑</td><td>SSIM↑</td><td>PSNR ↑</td><td>SSIM↑</td></tr><tr><td>SVG [10]</td><td>18.12</td><td>0.712</td><td>19.86</td><td>0.708</td></tr><tr><td>SA-CONvLSTM [35]</td><td>18.28</td><td>0.677</td><td>19.30</td><td>0.638</td></tr><tr><td>PhyDNet [19]</td><td>18.91</td><td>0.743</td><td>20.89</td><td>0.727</td></tr><tr><td>Iso-Dream</td><td>19.51</td><td>0.768</td><td>21.71</td><td>0.769</td></tr></table>

定性结果。图3（右列）展示了在CARLA环境中预测的重建结果。在CARLA中，我们观察到当主车（即智能体）上的摄像头移动时，智能体的动作可能会影响观测中的所有像素值。因此，我们将其他车辆的视觉动态视为可控制状态与不可控制状态的结合。因此，我们的模型能够通过学习基于动作的和无动作的分支的注意力掩码（值在0和1之间）来确定哪个组件是主导的。“无动作掩码”在其他车辆周围显示了热点，而“基于动作掩码”的相应区域的注意力值仍大于0。智能体通过推演不可控制组件，可以预览其他车辆可能的未来状态，从而避免碰撞。我们在补充材料中包含了更多不同车辆数量的展示。

# 4.4 BAIR 与 RoboNet 的基于动作的视像预测

实现细节。为了在更复杂的环境中评估我们的世界模型的有效性，我们在BAIR和RoboNet数据集上测试了所提结构的视频预测能力。此外，我们在原始观测中添加了与控制信号无关的可预测视觉动态，即相同大小和速度的弹跳球。在训练阶段，我们训练模型从2个观测值预测未来10帧。在测试中，我们使用前2帧作为输入，在BAIR数据集中预测接下来的28帧，在RoboNet数据集中预测接下来的18帧。所有训练和测试的输入都调整为 $64 \times 64$ 的大小。考虑到弹跳球的简单性和可预测性，在无动作分支中，我们使用与DMC实验中类似的结构。此外，我们在两个分支中都用两层ST-LSTM单元替换了GRU单元 [52]。优化目标包括图像重建损失和逆元件的动作重建损失。SSIM和PSNR被作为评估指标。

![](images/5.jpg)  

Figure 5: Showcases of video prediction results on the BAIR robot pushing dataset. We display every 3 frames in the prediction horizon. The generated masks show that each branch of Iso-Dream captures coarse localisation of controllable representations and noncontrollable representations.

定量结果。表 2 显示了在 BAIR 和 RoboNet 数据集上，包含弹球的训练和测试阶段的定量结果。与其他模型相比，Iso-Dream 在这两个数据集上表现出竞争力。在 PSNR 指标上，Iso-Dream 在 BAIR 数据集上比 SVG 提高了 $7 . 7 \%$，在 RoboNet 数据集上提高了 ${ \bar { 9 } } . 3 \%$。与同样在两个分支中解耦特征的 PhyDNet 相比，Iso-Dream 在 PSNR 和 SSIM 上都取得了更好的表现。这表明，Iso-Dream 拥有更强的解耦学习能力，以实现长期预测。定性结果。我们在图 5 中可视化了 BAIR 数据集中包含弹球的预测帧序列。具体来说，提供了两个分支的输出及其相应的蒙版。从这些展示中可以看出，Iso-Dream 的世界模型在建模未来动态以实现长期预测方面更加准确。它表明，无动作分支学习的是不可控动态，而有动作条件的分支学习的是与输入动作相关的可控动态。

# 5 结论

在本文中，我们提出了一种名为 Iso-Dream 的 MBRL 框架，主要解决在复杂视觉动态下基于视觉的预测和控制的困难。我们的方法对世界模型表示学习和相关 MBRL 算法有两个新的贡献。首先，它通过模块化网络结构和逆动态学习来解耦可控和不可控的潜在状态转换。此外，它通过将不可控动态推演到未来来进行长期决策，并学习其对当前行为的影响。Iso-Dream 在 CARLA 自动驾驶任务上取得了竞争性结果，其他车辆可以自然地视为不可控组件，这表明借助解耦的潜在状态，智能体能够通过预览可能的未来状态在无动作网络分支中做出更具前瞻性的决策。此外，Iso-Dream 显示出有效改善修改后的 DeepMind Control Suite 中的视觉控制任务，以及在 BAIR 机器人推送数据集和 RoboNet 数据集上的视觉预测任务。Iso-Dream 的一个局限性是计算效率。与 DreamerV2 相比，由于行为学习中的状态转换更为密集，它每集所需的训练时间更长。但幸运的是，从图 4(a) 显示，Iso-Dream 的样本效率高于现有的 MBRL 方法。另一个局限性是对不同环境的特殊处理。在我们的初步实验中，我们尝试对所有测试基准使用相同的模型架构。然而，我们观察到不同基准对网络结构有特定要求，这与我们对环境的先验知识有关。

# 致谢

本研究得到了中国自然科学基金（U19B2035, 62106144）、上海市科委重大科技项目（2021SHZDZX0102）和上海市航浦计划（21Z510202133）的支持。

# References

[1] Mohammad Babaeizadeh, Chelsea Finn, Dumitru Erhan, Roy H Campbell, and Sergey Levine. Stochastic variational video prediction. In ICLR, 2018.   
[2] Nadine Behrmann, Jurgen Gall, and Mehdi Noroozi. Unsupervised video representation learning by bidirectional feature prediction. In WACV, pages 16701679, 2021.   
[3] Xinzhu Bei, Yanchao Yang, and Stefano Soatto. Learning semantic-aware dynamics for video prediction. In CVPR, pages 902912, 2021.   
[4] Homanga Bharadhwaj, Mohammad Babaeizadeh, Dumitru Erhan, and Sergey Levine. Information prioritization through empowerment in visual model-based RL. In ICLR, 2022.   
[5] Prateep Bhattacharjee and Sukhendu Das. Temporal coherency based criteria for predicting video frames using deep multi-stage generative adversarial networks. In NeurIPS, pages 42714280, 2017.   
[6] Navaneeth Bodla, Gaurav Shrivastava, Rama Chellappa, and Abhinav Shrivastava. Hierarchical video prediction using relational layouts for human-object interactions. In CVPR, 2021.   
[7] Lluis Castrejon, Nicolas Ballas, and Aaron Courville. Improved conditional VRNNs for video prediction. In ICCV, pages 76087617, 2019.   
[8] Silvia Chiappa, Sébastien Racaniere, Daan Wierstra, and Shakir Mohamed. Recurrent environment simulators. In ICLR, 2017.   
[9] Sudeep Dasari, Frederik Ebert, Stephen Tian, Suraj Nair, Bernadette Bucher, Karl Schmeckpeper, Siddarth Singh, Sergey Levine, and Chelsea Fin. Robonet: Large-scale multi-robot learning. arXiv preprint arXiv:1910.11215, 2019.   
[10] Emily Denton and Rob Fergus. Stochastic video generation with a learned prior. In ICML, pages 11741183. PMLR, 2018.   
[11] Alexey Dosovitskiy, Germán Ros, Felipe Codevilla, Antonio M. López, and Vladlen Koltun. CARLA: an open urban driving simulator. In CoRL, volume 78, pages 116. PMLR, 2017.   
[12] Frederik Ebert, Chelsea Finn, Sudeep Dasari, Annie Xie, Alex Lee, and Sergey Levine. Visual foresight: Model-based deep reinforcement learning for vision-based robotic control. arXiv preprint arXiv:1812.00568, 2018.   
[13] Frederik Ebert, Chelsea Finn, Alex X Lee, and Sergey Levine. Self-supervised visual planning with temporal skip connections. In CoRL, pages 344356, 2017.   
[14] Chelsea Finn, Ian Goodfellow, and Sergey Levine. Unsupervised learning for physical interaction through video prediction. In NeurIPS, pages 6472, 2016.   
[15] Chelsea Finn and Sergey Levine. Deep visual foresight for planning robot motion. In ICRA, pages 27862793. IEEE, 2017.   
[16 Jean-Yves Francehi, Edouard Delasalles, Mickaë Chen, ylvain Lamprier, and Patrick Gallinari. tochastic latent residual video prediction. In ICML, pages 32333246, 2020.   
[17] Anirudh Goyal, Alex Lamb, Jordan Hoffmann, Shagun Sodhani, Sergey Levine, Yoshua Bengio, and Bernhard Schölkopf. Recurrent independent mechanisms. In ICLR, 2021.   
[18] Klaus Greff, Raphaël Lopez Kaufman, Rishabh Kabra, Nick Watters, Christopher Burgess, Daniel Zoran, Loic Matthey, Matthew Botvinick, and Alexander Lerchner. Multi-object representation learning with iterative variational inference. In ICML, pages 24242433, 2019.   
[19] Vincent Le Guen and Nicolas Thome. Disentangling physical dynamics from unknown factors for unsupervised video prediction. In CVPR, pages 1147411484, 2020.   
[20] David Ha and Jürgen Schmidhuber. Recurrent world models facilitate policy evolution. In NeurIPS, 2018.   
[21] Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, et al. Soft actor-critic algorithms and applications. arXiv preprint arXiv:1812.05905, 2018.   
[2 a Hafr, Timoy Lil, Ji Ba, nd Moha oozi Dre  cnto: L behaviors by latent imagination. In ICLR, 2020.   
[23] Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In ICML, pages 25552565. PMLR, 2019.   
[24] Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, and Jimmy Ba. Mastering atari with discrete world models. arXiv preprint arXiv:2010.02193, 2020.   
[25] Nicklas Hansen, Hao Su, and Xiaolong Wang. Stabilizing deep q-learning with convnets and vision transformers under data augmentation. In NeurIPS, 2021.   
[26] Nicklas Hansen and Xiaolong Wang. Generalization in reinforcement learning by soft data augmentation. In ICRA, 2021.   
[27Jun-Ting Hsieh, Bingbin Liu, De-An Huang, Li F Fei-Fei, and Juan Carlos Niebles. Learning to decompose and disentangle representations for video prediction. In NeurIPS, pages 517526, 2018.   
[28] Beibei Jin, Yu Hu, Qiankun Tang, Jingyu Niu, Zhiping Shi, Yinhe Han, and Xiaowei Li. Exploring spatial-temporal multi-frequency analysis for high-fidelity and temporal-consistency video prediction. In CVPR, pages 45544563, 2020.   
[29] Minju Jung, Takazumi Matsumoto, and Jun Tani.Goal-directed behavior under variational predictive coding: Dynamic organization of visual attention and working memory. In IROS, pages 10401047. IEEE, 2019.   
[30] Lukasz Kaiser, Mohammad Babaeizadeh, Piotr Milos, Blazej Osinski, Roy H Campbell, Konrad Czechowski, Dumitru Erhan, Chelsea Finn, Piotr Kozakowski, Sergey Levine, et al. Model-based reinforcement learning for Atari. In ICLR, 2020.   
[31] Taesup Kim, Sungjin Ahn, and Yoshua Bengio. Variational temporal abstraction. In NeurIPS, volume 32, pages 1157011579, 2019.   
[32] Ilya Kostrikov, Denis Yarats, and Rob Fergus. Image augmentation is all you need: Regularizing deep reinforcement learning from pixels. arXiv preprint arXiv:2004.13649, 2020.   
[33] Michael Laskin, Kimin Lee, Adam Stooke, Lerrel Pinto, Pieter Abbeel, and Aravind Srinivas. Reinforcement learning with augmented data. arXiv preprint arXiv:2004.14990, 2020.   
[34] Michael Laskin, Aravind Srinivas, and Pieter Abbeel. CURL: contrastive unsupervised representations fr rinormet learnng. In ICML, volume 119 of Procding of Machine Learin Research, pges 56395650. PMLR, 2020.   
[35] Zhihui Lin, Maomao Li, Zhuobin Zheng, Yangyang Cheng, and Chun Yuan. Self-attention convlstm for spatiotemporal prediction. In AAAI, volume 34, pages 1153111538, 2020.   
[36] Wenqian Liu, Abhishek Sharma, Octavia Camps, and Mario Sznaier. Dyan: A dynamical atoms-based network for video prediction. In ECCV, pages 170185, 2018.   
[37] Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob e ik   o Neural Information Processing Systems, 33:1152511538, 2020.   
[38] Junhyuk Oh, Xiaoxiao Guo, Honglak Lee, Richard Lewis, and Satinder Singh. Action-conditional video prediction using deep networks in atari games. arXiv preprint arXiv:1507.08750, 2015.   
[39] Junhyuk Oh, Satinder Singh, and Honglak Lee. Value prediction network. In NeurIPS, 2017.   
[40] Marc Oliu, JaverSelv, and Sergio Escalera. Foldedrecurrent neural networks orfuture video preicion. In ECCV, pages 716731, 2018.   
[41] Fitsum A Reda, Guilin Liu, Kevin JShih, Robert Kirby, Jon Barker, David Tarjan, Andrew Tao, and Bryan Catanzaro. Sdc-net: Video prediction using spatially-displaced convolution. In ECCV, pages 718733, 2018.   
[42] Ramanan Sekar, Oleh Rybkin, Kostas Daniilidis, Pieter Abbeel, Danijar Hafner, and Deepak Pathak. Planning to explore via self-supervised world models. In ICML, pages 85838592, 2020.   
[43] Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-Kin Wong, and Wang-chun Woo. Convolual TM network A macie ear aprac or prepitation owstin. In Neur, page 802810, 2015.   
[44] Nitish Sivastava, Ean Mnsiov, nd Ruslan Salakhudiv. Unsupevis eari  video rerenations using lstms. In ICML, pages 843852. PMLR, 2015.   
[45] Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press, 2018.   
[46] Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las Casas, David Budden, Abbas Abdolmaleki, Josh Merel, Andrew Lefrancq, et al. Deepmind control suite.arXiv preprint arXiv:1801.00690, 2018.   
[4Sjoerd van Steenkiste, Michael Chang, Klaus Gre and Jürgen Schmidhuber. Relational neural expecaton maximization: Unsupervised discovery of objects and their interactions. In ICLR, 2018.   
[48] Ruben Villegas, Arkanath Pathak, Harini Kannan, Dumitru Erhan, Quoc V Le, and Honglak Lee. High fidelity video prediction with large stochastic recurrent neural networks. In NeurIPS, pages 8191, 2019.   
[49] Ruben Villegas, Jimei Yang, Seunghoon Hong, Xunyu Lin, and Honglak Lee. Decomposing motion and content for natural video sequence prediction. In ICLR, 2017.   
[50] Ruben Villegas, Jimei Yang, Yuliang Zou, Sungryul Sohn, Xunyu Lin, and Honglak Lee. Learning to generate long-term future via hierarchical prediction. In ICML, pages 35603569, 2017.   
[51] Carl Vondrick, Hamed Pirsiavash, and Antonio Torralba. Generating videos with scene dynamics. In NeurIPS, pages 613621, 2016.   
[52] Yunbo Wang, Mingsheng Long, Jianmin Wang, Zhifeng Gao, and Philip S Yu. Predrnn: Recurrent neural networks for predictive learning using spatiotemporal Istms. In NeurIPS, pages 879888, 2017.   
[53] Yunbo Wang, Haixu Wu, Jianjin Zhang, Zhifeng Gao, Jianmin Wang, Philip S Yu, and Mingsheng Log. Predrnn: A recurrent neural network for spatiotemporal predictive learning. arXi preprint arXiv:2103.09504, 2021.   
[54] Nevan Wichers, Ruben Villegas, Dumitru Erhan, and Honglak Lee. Hierarchical long-term video prediction without supervision. In ICML, pages 60386046, 2018.   
[55] Bohan Wu, Suraj Nair, Roberto Martin-Martin, Li Fei-Fei, and Chelsea Finn. Greedy hierarchical variational autoencoders for large-scale video prediction. In CVPR, pages 23182328, 2021.   
[56] Jingwei Xu, Bingbing Ni, ZefanLi, Shuo Cheng, and Xiaokang Yang. Structure preserving video prediction. In CVPR, pages 14601469, 2018.   
[57] Denis Yarats, Amy Zhang, Ilya Kostrikov, Brandon Amos, Joelle Pineau, and Rob Fergus. Improving sample efficiency in model-free reinforcement learning from images. In AAAI, pages 1067410681, 2021.   
[58] Polina Zablotskaia, Edoardo A Dominici, Leonid Sigal, and Andreas M Lehrmann. Unsupervised video decomposition using spatio-temporal iterative inference. arXiv preprint arXiv:2006.14727, 2020.   
[59] Amy Zhang, Rowan McAllister, Roberto Calandra, Yarin Gal, and Sergey Levine. Learning invariant representations for reinforcement learning without reconstruction. In ICLR, 2021.

# A Benchmarks

We quantitatively and qualitatively evaluate Iso-Dream on the following two environments for visual control and two real-world datasets for action-conditioned video prediction.

•DeepMind control suite [46]: A set of stable, well-tested continuous control tasks that are easy to use and modify. For vision-based control, we use a modified version of the DeepMind control suite in DMControl Generalization Benchmark [26] to evaluate Iso-Dream. In this environment, agents are trained to complete different tasks with random natural video as backgrounds, namely video_easy and video_hard benchmarks. We use 4 tasks to test our Iso-Dream, i.e., Finger Spin, Cheetah Run, Walker Walk, Hopper Stand.

•CARLA [11]: An open-source simulator with more complex and realistic visual observations for autonomous driving research. In our experiments, we evaluate Iso-Dream in a first-person highway driving task in "Town04". The agent's goal is to drive as far as possible in 1000 time steps without colliding with the 30 other moving vehicles or barriers.

•BAIR robot pushing [13]: An action-conditioned video prediction dataset composed of hours of self-supervised learning with the robotic arm Sawyer. In each video, a random moving robotic arm pushes a variety of objects on similar tables with a static background. Each video also has recorded actions taken by the robotic arm which correspond to the commanded gripper pose.

RoboNet [9]: A large-scale dataset contains action-conditioned videos of seven robotic arms interacting with a variety of objects from four different research laboratories, i.e., Berkeley, Google, Penn, and Stanford.

# B Compared Methods

For visual MBRL, we compare our method with the following baselines and existing approaches:

•DreamerV2 [24]: A model-based RL method that learns directly from latent variables in world models. The latent representation enables agents to imagine thousands of trajectories in parallel.   
•CURL [34]: A model-free RL method that extracts high-level features from raw pixels using contrastive learning, maximizing agreement between augmented versions of the same observation.   
•SVEA [25]: A framework for data augmentation in deep Q-learning algorithms that improves stability and generalization on off-policy RL.   
SAC [21] A model-free actor-critic method that optimizes a stochastic policy in an off-policy way.   
•DBC [59]: It learns a bisimulation metric representation without reconstruction loss, which are invariant to different task-irrelevant details in the observation.

For video prediction, we compare the proposed world model with the following approaches:

•SVG [10]: This model introduces random variables into latent space, which ensures that the future trajectory is inherently random.   
•SA-ConvLSTM [35]: Based on the self-attention mechanism, this model uses the self-attention memory to capture long-term spatial dependency.   
•PhyDNet [19]: This model uses a two-branch architecture to disentangle PDE dynamics from unknown complementary information.

# C Additional Visualization in DMC and CARLA

DeepMind Control suite. In Figure 6, more showcases on the DeepMind Control are presented with different noisy backgrounds. We show the visualization of the masks and decoupled components from three branches of Iso-Dream.

CARLA autonomous driving simulator. In Figure 7, we visualize the video prediction results on the CARLA environment with different numbers of vehicles. We train Iso-Dream with 30 vehicles and test with 10 vehicles and 20 vehicles respectively.

![](images/6.jpg)  
Figure 6: Video prediction results with different noisy backgrounds on the DMC. For each sequence, we use the first 5 images as context frames.

# D Additional Results on the BAIR Robot Pushing Dataset

Figure 8 shows an interesting result of the different training sets (i.e., BAIR, BAIR $+$ bouncing balls) and the same testing set (i.e., BAIR). Iso-Dream is the only approach that achieves improvements when training on noisy data with bouncing balls, as shown in Figure 8(red bars). In this training setup, it performs best on the standard test set without balls. Iso-Dream is built on a more efficient architecture than the baseline models. It provides a general framework that can be easily extended to other backbones.

Ablation study. In Table 3, the first row shows the results of removing the action-free branch in the world model of Iso-Dream. The performance has decreased from 21.43 to 20.47 and from 19.51 to 18.51 in PSNR for predicting the next 18 frames and next 28 frames respectively, indicating that modular network structures are effective for predictive learning by decoupling the controllable and noncontrollable representations. Comparing the second row and third row in the Table 3, we observe that modeling inverse dynamics can improve the performance by learning more deterministic state transitions given particular actions in the action-conditioned branch.

# E Network Architectures for Different Environments

The networks and hyper-parameters used for different environments are shown in Table 4.

![](images/7.jpg)  
Figure 7: Video prediction results with 10 vehicles (left) and 20 vehicles (right) on the CARLA environment. For each sequence, we use the first 5 images as context frames.

![](images/8.jpg)  
Figure 8: The results of models trained on BAIR (blue) and BAIR $^ +$ bouncing balls (red), and tested on BAIR. We use the first 2 frames as input to predict the next 18 frames. The horizontal axis represents the different models, and the vertical axes represent test results of PSNR and SSIM.

Table 3: Ablation study for each component of Iso-Dream for video prediction on BAIR with bouncing balls. Lines 1-2 show the results of removing the action-free branch and Inverse cell, respectively. We use the first 2 frames as input to predict the next 18 frames and the next 28 frames.   

<table><tr><td>MODEL</td><td>PReDict 18 FRamES PSNR ↑</td><td>SSIM ↑</td><td>Predict 28 Frames PSNR ↑ SSIM ↑</td></tr><tr><td>Iso-Dream w/o action-free Branch</td><td>20.47</td><td>0.795</td><td>18.51 0.690</td></tr><tr><td>Iso-Dream w/o Inverse CeLL</td><td>21.42</td><td>0.829</td><td>19.34 0.759</td></tr><tr><td>Iso-Dream</td><td>21.43</td><td>0.832</td><td>19.51 0.768</td></tr></table>

Table 4: An overview of layers and hyper-parameters used for three environments.   

<table><tr><td rowspan=1 colspan=1>Name</td><td rowspan=1 colspan=1>DMC</td><td rowspan=1 colspan=1>CARLA</td><td rowspan=1 colspan=1>BARI / RoboNet</td></tr><tr><td rowspan=1 colspan=1>Encθ</td><td rowspan=1 colspan=1>conv3-32</td><td rowspan=1 colspan=1>conv3-32</td><td rowspan=1 colspan=1>conv3-64</td></tr><tr><td rowspan=1 colspan=4>Action-conditioned branch</td></tr><tr><td rowspan=1 colspan=1>Encφ1</td><td rowspan=1 colspan=1>conv3-64</td><td rowspan=1 colspan=1>conv3-64</td><td rowspan=1 colspan=1>conv3-64</td></tr><tr><td rowspan=1 colspan=1>GRUs</td><td rowspan=1 colspan=1>hidden size = 200</td><td rowspan=1 colspan=1>hidden size = 200</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=1 colspan=1>ST-LSTM</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>hidden size = 64</td></tr><tr><td rowspan=1 colspan=1>Dec1</td><td rowspan=1 colspan=1>conv3-4</td><td rowspan=1 colspan=1>conv3-4</td><td rowspan=1 colspan=1>conv3-4</td></tr><tr><td rowspan=1 colspan=1>α</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.0001</td></tr><tr><td rowspan=1 colspan=1>β1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=1 colspan=3>Action-free branch</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Encφ2</td><td rowspan=1 colspan=1>conv3-64</td><td rowspan=1 colspan=1>conv3-64</td><td rowspan=1 colspan=1>conv3-64</td></tr><tr><td rowspan=1 colspan=1>GRUz</td><td rowspan=1 colspan=1>hidden size = 200</td><td rowspan=1 colspan=1>hidden size = 200</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=1 colspan=1>ST-LSTM</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>hidden size = 64</td></tr><tr><td rowspan=1 colspan=1>Decφ2</td><td rowspan=1 colspan=1>conv3-4</td><td rowspan=1 colspan=1>conv3-4</td><td rowspan=1 colspan=1>conv3-4</td></tr><tr><td rowspan=1 colspan=1>β2</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=1 colspan=4>Static branch</td></tr><tr><td rowspan=1 colspan=1>Encφ3</td><td rowspan=1 colspan=1>conv3-64</td><td rowspan=1 colspan=1>conv3-64</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=1 colspan=1>Dec3</td><td rowspan=1 colspan=1>conv3-3</td><td rowspan=1 colspan=1>conv3-3</td><td rowspan=1 colspan=1>-</td></tr></table>