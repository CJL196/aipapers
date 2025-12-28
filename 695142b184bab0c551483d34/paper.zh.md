# DanceGRPO：在视觉生成中释放 GRPO 的潜力

Zeyue $\mathbf { \boldsymbol { x } } \mathbf { \boldsymbol { u } } \mathbf { \Theta } { \mathbf { \Xi } } ^ { 1 , 2 }$，Jie ${ \pmb { \mathsf { W } } } { \pmb { \mathsf { u } } } ^ { 1 \ddag }$，Yu Gao1，Fangyuan Kong1，Lingting $\mathbf { z h u ^ { 2 } }$，Mengzhao Chen²，Zhiheng Liu2，Wei Liu1，Qiushan Guo1，Weilin Huang1†，Ping Luo2† 1字节跳动种子团队，香港大学 †通讯作者， ‡项目负责人

# 摘要

近期生成性人工智能的进展彻底改变了视觉内容创作，但使模型输出与人类偏好对齐仍然是一个关键挑战。虽然强化学习（RL）作为微调生成模型的有希望的方式崭露头角，但现有方法如 DDPO 和 DPOK面临根本的限制，特别是在扩展到大规模和多样化的提示集时无法保持稳定的优化，严重限制了其实用性。本文提出了 DanceGRPo 框架，通过对组相对策略优化（GRPO）在视觉生成任务中的创新适配来解决这些限制。我们的关键见解是，GRPO的固有稳定机制使其能够独特地克服困扰先前基于RL的方法在视觉生成中的优化挑战。DanceGRPO 设立了几项重要的进展：首先，它在多个现代生成范式中展示了一致且稳定的策略优化，包括扩散模型和修正流。其次，它在扩展到复杂的现实场景时保持了强健的性能，涵盖了三个关键任务和四个基础模型。第三，它在优化捕捉五种不同奖励模型以评估图像/视频美学、文本-图像对齐、视频运动质量和二元反馈等多样化人类偏好时展示了显著的灵活性。我们的综合实验表明，DanceGRPO 在多个已建立基准测试中超过了基线方法，提升幅度高达 $1 8 1 \%$，包括 HPS-v2.1、CLIP Score、VideoAlign 和 GenEval。我们的结果确立了 DanceGRPO 作为一个稳健且多功能的解决方案，能够扩展基于人类反馈的强化学习（RLHF）任务在视觉生成中的应用，为和谐强化学习与视觉合成提供了新的见解。日期：2025年5月1日 项目页面：https://dancegrpo.github.io/ 代码：https://github.com/XueZeyue/DanceGRPO

# 1 引言

最近在生成模型方面的进展——特别是扩散模型和校正流——通过提高图像和视频生成的输出质量和多样性，彻底改变了视觉内容的创作。尽管预训练建立了基础数据分布，但在训练过程中整合人类反馈对于使输出与人类偏好和美学标准对齐至关重要。现有方法面临显著的局限性：ReFL依赖于可微分奖励模型，这在视频生成中引入了显存效率低下，并需要大量的工程工作，而DPO变种（扩散-DPO、流-DPO、在线VPO）仅实现了边际的视觉质量提高。基于强化学习的方法，通过优化作为黑箱目标的奖励，提供了潜在的解决方案，但引入了三个未解决的挑战：（1）基于常微分方程的校正流模型的采样与马尔可夫决策过程的表述相冲突；（2）先前的策略梯度方法在超越小型数据集（例如<100个提示）时表现出不稳定性；（3）现有方法在视频生成任务上未经验证。本研究通过通过随机微分方程重新构造扩散模型和校正流的采样，并应用群体相对政策优化来稳定训练过程，解决了这些差距。在本文中，我们开创性地通过DanceGRPO框架将GRPO适配于视觉生成任务，建立GRPO与视觉生成任务之间的“和谐舞蹈”。我们的关键见解是，GRPO的架构稳定性特性为解决限制早期基于RL的方法在视觉生成中的优化不稳定性提供了一个有原则的解决方案。我们扩展了DanceGRPO的系统研究，评估其在生成范式（扩散模型、校正流）和任务（文本到图像、文本到视频、图像到视频）中的性能。我们的分析采用各种基础模型和奖励指标，以评估美学质量、对齐度和运动动态。此外，通过提出的框架，我们发现了关于推演初始化噪声、奖励模型兼容性、最佳N推理缩放、时间步选择和在二元反馈上的学习的见解。我们的贡献可以总结如下：•稳定性和开创性。我们首次发现GRPO的内在稳定机制有效解决了持续阻碍先前基于RL的视觉生成方法的核心优化挑战。通过仔细重构随机微分方程、选择适当的优化时间步、初始化噪声和噪声尺度，我们实现了GRPO与视觉生成任务之间的无缝集成。通用性和可扩展性。我们所知，DanceGRPO是第一个基于RL的统一应用框架，能够无缝适配多样的生成范式、任务、基础模型和奖励模型。与先前的RL算法不同，后者主要在小规模数据集上的文本到图像扩散模型上得到验证，DanceGRPO在大规模数据集上表现出强大的性能，展示了可扩展性和实用性。高效性。我们的实验表明，DanceGRPO在视觉生成任务中实现了显著的性能提升，超越基准达到了$181\%$的增幅，包括HPSv2.1、CLIP评分、VideoAlign和GenEval。值得注意的是，DanceGRPO还使模型能够学习最佳N推理缩放中的去噪轨迹。我们还进行了一些初步尝试，使模型能够捕捉二元（$0/1$）奖励模型的分布，展示了其捕捉稀疏阈值反馈的能力。

# 2 方法

# 2.1 初步研究

扩散模型 [1]。扩散过程在时间步 $t$ 上逐渐破坏观察到的数据点 $\mathbf{x}$，通过将数据与噪声混合，扩散模型的正向过程可以定义为：

$$
\mathbf { z } _ { t } = \alpha _ { t } \mathbf { x } + \sigma _ { t } \mathbf { \epsilon } , \mathrm { ~ w h e r e ~ } \epsilon \sim \mathcal { N } ( 0 , \mathbf { I } ) ,
$$

且 $\alpha _{t}$ 和 $\sigma _{t}$ 表示噪声调度。噪声调度的设计方式确保 $\mathbf{z}_{0}$ 接近干净数据，而 $\mathbf{z}_{1}$ 接近高斯噪声。为了生成新的样本，我们初始化样本 $\mathbf{z}_{1}$ 并定义扩散模型的样本方程，该方程基于去噪模型在时间步 $t$ 的输出 $\hat{\epsilon}$：

$$
{ \bf z } _ { s } = \alpha _ { s } \hat { \bf x } + \sigma _ { s } \hat { \epsilon } ,
$$

其中 $\hat { \bf x }$ 可以通过方程 (1) 推导得出，从而达到更低的噪声水平 $s$。这也是一个 DDIM 采样器 [28]。在整流流动 [6] 中，我们将正向过程视为数据 $\mathbf { x }$ 与噪声项 $\epsilon$ 之间的线性插值：

$$
\mathbf { z } _ { t } = ( 1 - t ) \mathbf { x } + t { \boldsymbol { \epsilon } } ,
$$

其中 $\epsilon$ 始终定义为高斯噪声。我们将 $\mathbf { u } = \epsilon - \mathbf { x }$ 定义为“速度”或“向量场”。与扩散模型类似，给定去噪模型在时间步 $t$ 的输出 $\hat { \bf { u } }$，我们可以通过以下方式达到更低的噪声水平 $s$：

$$
\mathbf { z } _ { s } = \mathbf { z } _ { t } + \hat { \mathbf { u } } \cdot ( s - t ) .
$$

分析。尽管扩散模型与校正流有着不同的理论基础，但在实践中，它们是同一个事物的两个方面，如下公式所示：

$$
\tilde { \mathbf { z } } _ { s } = \tilde { \mathbf { z } } _ { t } + \mathrm { N e t w o r k ~ o u t p u t } \cdot ( \eta _ { s } - \eta _ { t } ) .
$$

对于 $\epsilon$-预测（即扩散模型），根据公式 (2)，我们有 $\tilde { \mathbf { z } } _ { s } = \mathbf { z } _ { s } / \alpha _ { s }$，$\tilde { \mathbf { z } } _ { t } = \mathbf { z } _ { t } / \alpha _ { t }$，$\eta _ { s } = \sigma _ { s } / \alpha _ { s }$，以及 $\eta _ { t } = \sigma _ { t } / \alpha _ { t }$。对于修正流，我们有 $\tilde { \mathbf { z } } _ { s } = \mathbf { z } _ { s }$，$\tilde { \mathbf { z } } _ { t } = \mathbf { z } _ { t }$，$\eta _ { s } = s$，以及 $\eta _ { t } = t$，来自于公式 (4)。

# 2.2 DanceGRPO 在本节中，我们首先将扩散模型和校正流的采样过程表述为马尔可夫决策过程。然后，我们介绍采样的随机微分方程（SDEs）及DanceGRPO算法。

将去噪建模为马尔可夫决策过程。根据 DDPO [18]，我们将扩散模型和修正流的去噪过程表述为一个马尔可夫决策过程（MDP）：

$$
\begin{array}{c} \begin{array} { r l } & { \mathbf { s } _ { t } \triangleq ( \mathbf { c } , t , \mathbf { z } _ { t } ) , \quad \pi ( \mathbf { a } _ { t } \mid \mathbf { s } _ { t } ) \triangleq p ( \mathbf { z } _ { t - 1 } \mid \mathbf { z } _ { t } , \mathbf { c } ) , \quad P ( \mathbf { s } _ { t + 1 } \mid \mathbf { s } _ { t } , \mathbf { a } _ { t } ) \triangleq \left( \delta _ { \mathbf { c } } , \delta _ { t - 1 } , \delta _ { \mathbf { z } _ { t - 1 } } \right) } \\ & { \mathbf { a } _ { t } \triangleq \mathbf { z } _ { t - 1 } , \quad R ( \mathbf { s } _ { t } , \mathbf { a } _ { t } ) \triangleq \left\{ \begin{array} { l l } { r ( \mathbf { z _ { 0 } } , \mathbf { c } ) , } & { \mathrm { i f ~ } t = 0 } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} , \quad \rho _ { 0 } ( \mathbf { s } _ { 0 } ) \triangleq ( p ( \mathbf { c } ) , \delta _ { T } , \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) ) \right.} \end{array}  ,  \end{array}
$$

其中 $\mathbf{c}$ 是提示，$\pi(\mathbf{a}_t \mid \mathbf{s}_t)$ 是从 $z_t$ 到 $z_{t-1}$ 的概率。$\delta_y$ 是狄拉克 delta 分布，仅在 $y$ 处具有非零密度。轨迹由 $T$ 个时间步组成，之后 $P$ 导致一个终止状态。$r(\mathbf{z}_0, \mathbf{c})$ 是奖励模型，总是由视觉-语言模型（如 CLIP [26] 和 Qwen-VL [29]）参数化。采样 SDEs 的公式化。由于 GRPO 需要通过多个轨迹样本进行随机探索，策略更新依赖于轨迹概率分布及其相关的奖励信号，我们将扩散模型和校正流的采样过程统一为 SDEs 的形式。对于扩散模型，如 [30, 31] 中所示，正向 SDE 表示为：$\mathrm{d} \mathbf{z}_t = f_t \mathbf{z}_t \mathrm{d}t + g_t \mathrm{d} \mathbf{w}$。对应的反向 SDE 可以表示为：

$$
\mathrm { d } \mathbf { z } _ { t } = \left( f _ { t } \mathbf { z } _ { t } - \frac { 1 + \varepsilon _ { t } ^ { 2 } } { 2 } g _ { t } ^ { 2 } \nabla \log p _ { t } ( \mathbf { z _ { t } } ) \right) \mathrm { d } t + \varepsilon _ { t } g _ { t } \mathrm { d } \mathbf { w } ,
$$

其中 \( dw \) 是布朗运动，\(\varepsilon _ { t }\) 在采样过程中引入了随机性。类似地，整流流的前向常微分方程为：\(\mathrm { d } \mathbf { z } _ { t } = \mathbf { u } _ { t } \mathrm { d } t\)。生成过程在时间上反转了常微分方程。然而，这种确定性表述无法提供 GRPO 所需的随机探索。受到 [3234] 的启发，我们在反向过程中引入如下的随机微分方程案例：

$$
\mathrm { d } \mathbf { z } _ { t } = ( \mathbf { u } _ { t } - \frac { 1 } { 2 } \varepsilon _ { t } ^ { 2 } \nabla \log p _ { t } ( \mathbf { z } _ { t } ) ) \mathrm { d } t + \varepsilon _ { t } \mathrm { d } \mathbf { w } ,
$$

其中 $\varepsilon _ { t }$ 也在采样过程中引入了随机性。给定一个正态分布 $p _ { t } ( \mathbf { z } _ { t } ) = \mathcal { N } ( \mathbf { z } _ { t } \mid \alpha _ { t } \mathbf { x } , \sigma _ { t } ^ { 2 } I )$ ，我们有 $\nabla \log p _ { t } ( \mathbf { z } _ { t } ) = - ( \mathbf { z } _ { t } - \alpha _ { t } \mathbf { x } ) / \sigma _ { t } ^ { 2 }$ 。我们可以将此插入上述两个随机微分方程中，并获得 $\pi ( \mathbf { a } _ { t } \mid \mathbf { s } _ { t } )$ 。更多理论分析见附录 B。算法。受到 Deepseek-R1 [20] 的启发，给定一个提示 $\mathbf { c }$ ，生成模型将从模型 $\pi _ { \theta _ { o l d } }$ 中采样一组输出 $\left\{ \mathbf { o } _ { 1 } , \mathbf { o } _ { 2 } , . . . , \mathbf { o } _ { G } \right\}$ ，并通过最大化以下目标函数来优化策略模型 $\pi _ { \theta }$ ：

$$
\mathcal { I } ( \theta ) = \mathbb { E } _ { \{ \mathbf { o } _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot | \mathbf { c } ) } \bigg [ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \operatorname* { m i n } \Bigg ( \rho _ { t , i } A _ { i } , \mathrm { c l i p } \big ( \rho _ { t , i } , 1 - \epsilon , 1 + \epsilon \big ) A _ { i } \Bigg ) \bigg ] ,
$$

其中 $\rho _ { t , i } = \frac { \pi _ { \theta } \left( \mathbf { a } _ { t , i } \left| \mathbf { s } _ { t , i } \right. \right) } { \pi _ { \theta _ { o l d } } \left( \mathbf { a } _ { t , i } \left| \mathbf { s } _ { t , i } \right. \right) }$ 以及 $\pi _ { \boldsymbol { \theta } } ( \mathbf { a } _ { t , i } | \mathbf { s } _ { t , i } )$ 是在时间步 $t$ 的 $\mathbf { o } _ { i }$ 。$\epsilon$ 是一个超参数，$A _ { i }$ 是优势函数，使用一组奖励 $\{ r _ { 1 } , r _ { 2 } , . . . , r _ { G } \}$ 计算，该奖励对应于每组中的输出：

$$
A _ { i } = { \frac { r _ { i } - \operatorname * { m e a n } ( \{ r _ { 1 } , r _ { 2 } , \cdots , r _ { G } \} ) } { \operatorname * { s t d } ( \{ r _ { 1 } , r _ { 2 } , \cdots , r _ { G } \} ) } } .
$$

由于实际中奖励稀疏，我们在优化过程中对所有时间步应用相同的奖励信号。虽然传统的GRPO公式采用KL正则化来防止奖励过度优化，但我们在实证中观察到省略此部分时性能差异很小。因此，我们默认省略KL正则化项。完整算法见算法1。我们还介绍了如何在附录C中使用无分类器引导（CFG）进行训练。总之，我们将扩散模型和校正流的采样过程形式化为MDP，使用SDE采样方程，采用GRPO风格的目标，并推广到文本到图像、文本到视频和图像到视频的生成任务。 初始化噪声。在DanceGRPO框架中，初始化噪声构成了一个关键组件。以前的基于RL的方法，如DDPO，使用不同的噪声向量来初始化训练样本。然而，如附录F中的图8所示，将不同的噪声向量分配给具有相同提示的样本总是会导致视频生成中的奖励黑客现象，包括训练不稳定。因此，在我们的框架中，我们为来自相同文本提示的样本分配共享的初始化噪声。 时间步选择。虽然去噪过程可以在MDP框架内严格形式化，但经验观察表明，在去噪轨迹中可以省略某些时间步而不影响性能。这种计算步骤的减少提高了效率，同时保持输出质量，更多分析见第3.6节。 整合多个奖励模型。在实践中，我们采用多个奖励模型以确保更稳定的训练和更高质量的视觉结果。如附录中的图9所示，专门使用HPS-v2.1奖励训练的模型趋向于生成不自然的（“油腻”）输出，而整合CLIP分数有助于保持更真实的图像特征。我们不是直接结合奖励，而是聚合优势函数，因为不同的奖励模型通常在不同的尺度上操作。这种方法可以稳定优化并导致更平衡的生成。 扩展最佳N推断缩放。如第3.6节所述，我们的方法优先使用高效样本——具体而言，与基于最佳N采样选择的前k和后k候选者相关的样本。这种选择性采样策略通过关注高奖励和关键低奖励区域来优化训练效率。我们使用穷举搜索来生成这些样本。虽然其他方法，如树搜索或贪婪搜索，仍然是进一步探索的有希望途径，但我们将其系统整合推迟到未来研究中。

# 2.3 应用到不同任务的不同奖励

我们在两种生成范式（扩散/修正流）和三个任务（文本到图像、文本到视频、图像到视频）中验证了我们算法的有效性。为此，我们选择了四个基础模型（Stable Diffusion [2]、HunyuanVideo [22]、FLUX [23]、SkyReels-I2V [24]）进行实验。这些方法在其采样过程中都可以在MDP框架内精确构建。这使我们能够统一这些任务的理论基础，并通过DanceGRPO进行改进。据我们所知，这是首次将统一框架应用于多样的视觉生成任务。

# 算法 1 DanceGRPO 训练算法

需求：初始策略模型 $\pi_\theta$；奖励模型 $\{ R_{k} \}_{k=1}^{K}$；提示数据集 $\mathcal{D}$；时间步选择比例 $\tau$；总采样步骤 $T$

确保：优化的策略模型 $\pi_\theta$ 1: 对于训练迭代 $= 1$ 至 $M$ 2: 从数据集 $\mathcal{D}$ 中抽样批次 $\mathcal{D}_b$，作为提示的批次 3: 更新旧策略：$\pi_{\theta_{\mathrm{old}}} \pi_\theta$ 4: 对于每个提示 $\mathbf{c} \in \mathcal{D}_b$，使用 $G$ 生成 $\{ \mathbf{o}_i \}_{i=1}^{G} \sim \pi_{\theta_{\mathrm{old}}}(\cdot | \mathbf{c})$ 6: 使用每个 $R_k$ 计算奖励 $\{ r_i^k \}_{i=1}^{G}$ 7: 对于每个样本： 8: 计算 $\begin{array}{r} A_i \gets \sum_{k=1}^{K} \frac{r_i^k - \mu^k}{\sigma^k} \end{array}$ 9: 结束循环 10: 子采样 $\lceil \tau T \rceil$ 个时间步 $\mathcal{T}_{\mathrm{sub}} \subset \{ 1 . . T \}$ 11: 对于 $t \in \mathcal{T}_{\mathrm{sub}}$： 12: 通过梯度上升更新策略：$\theta \gets \theta + \eta \nabla_\theta \mathcal{I}$ 13: 结束循环 14: 结束循环 15: 结束循环 表1：关于关键能力的对齐方法比较。VideoGen：视频生成泛化。可扩展性：对具有大量提示的数据集的可扩展性。奖励 $\uparrow$ 表示显著的奖励提升。RFs：适用于校正流。无差别奖励：不需要可微分的奖励模型。

<table><tr><td>Method</td><td>RL-based</td><td>VideoGen</td><td>Scalability</td><td>Reward ↑</td><td>RFs</td><td>No Diff-Reward</td></tr><tr><td>DDPO/DPOK</td><td>√</td><td>X</td><td>X</td><td>√</td><td>X</td><td>√</td></tr><tr><td>ReFL</td><td>X</td><td>X</td><td>√</td><td>√</td><td>√</td><td>X</td></tr><tr><td>DPO</td><td>X</td><td>√</td><td>√</td><td>X</td><td>V</td><td>√</td></tr><tr><td>Ours</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

我们使用五个奖励模型来优化视觉生成质量： (1) 图像美学通过在人工评估数据上微调的预训练模型来量化视觉吸引力[25]； (2) 文本-图像对齐采用 CLIP [26] 最大化提示和输出之间的跨模态一致性； (3) 视频美学质量将图像评估扩展到时间域，使用 VLMs [14, 29] 评估帧质量和连贯性； (4) 视频运动质量通过物理感知的 VLM [14] 对轨迹和变形的分析来评估运动的真实感； (5) 阈值二元奖励采用受[20] 启发的二元机制，通过固定阈值对奖励进行离散化（超过阈值的值获得 1，其他为 0），专门设计用来评估生成模型在基于阈值优化下学习突变奖励分布的能力。

# 2.4 与 DDPO、DPOK、ReFL 和 DPO 的比较

正如表1中的综合能力矩阵所示，DanceGRPO为扩散模型对齐设定了新的标准。我们的方法在所有评估维度上都实现了全方位的优越性：(1) 无缝视频生成，(2) 大规模数据集扩展性，(3) 显著的奖励改进，(4) 与校正流的原生兼容性，以及(5) 独立于可微分奖励。这一综合能力特征是任何单一基线方法（DDPO/DPOK/ReFL/DPO）无法实现的，它使得在多个生成领域内的同时优化成为可能，同时保持训练的稳定性。更多比较可以在附录D中找到。

# 3 实验

# 3.1 一般设置

文本到图像生成。我们采用稳定扩散模型 v1.4、FLUX 和 HunyuanVideo-T2I（使用 HunyuanVideo 中的一个潜在帧）作为基础模型，HPS-v2.1 [25] 和 CLIP 分数 [26] 及其二元奖励作为奖励模型。我们使用经过策划的提示数据集，平衡多样性和复杂性，以指导优化。对于评估，我们选择 1,000 个测试提示来评估 CLIP 分数和 Pick-a-Pic 性能（在第 3.2 节中）。我们使用 GenEval 和 HPS-v2.1 基准的官方提示。 文本到视频生成。我们的基础模型是 HunyuanVideo [22]，其奖励信号源自 VideoAlign [14]。使用 VidProM [36] 数据集策划提示，并筛选出另外 1,000 个测试提示以评估第 3.3 节中的 VideoAlign 分数。 图像到视频生成。我们使用 SkyReels-I2V [24] 作为我们的基础模型。VideoAlign [14] 作为主要的奖励指标，而通过 ConsisID [37] 构建的提示数据集配对由 FLUX [23] 合成的参考图像，以确保条件保真度。筛选出额外的 1,000 个测试提示以评估第 3.4 节中的 VideoAlign 分数。 实验设置。我们在适合任务复杂性的规模计算资源上实现了所有模型：32 个 H800 GPU 用于基于流的文本到图像模型，8 个 H800 GPU 用于稳定扩散变体，64 个 H800 GPU 用于文本到视频生成系统，以及 32 个 H800 GPU 用于图像到视频转换架构。我们基于 FastVideo [38, 39] 开发了我们的框架。附录 A 中详细列出了全面的超参数配置和训练协议。我们总是使用超过 10,000 个提示来优化模型。本文中呈现的所有奖励曲线采用移动平均法绘制，以实现更平滑的可视化。我们使用基于常微分方程的采样器进行评估和可视化。u 是基础模型，() 是使用 HPS 分数训练的模型，(3) 是同时优化 HPS 和 CLIP 分数的模型。为了评估，我们报告 HPS-v2.1 和 GenEval 分数，使用官方提示，同时 CLIP 分数和 Pick-a-Pic 指标基于我们的 1,000 个测试提示集计算。

<table><tr><td>Models</td><td>HPS-v2.1 [25]</td><td>CLIP Score [26]</td><td>Pick-a-Pic [40]</td><td>GenEval [27]</td></tr><tr><td>Stable Diffusion</td><td>0.239</td><td>0.363</td><td>0.202</td><td>0.421</td></tr><tr><td>Stable Diffusion with HPS-v2.1</td><td>0.365</td><td>0.380</td><td>0.217</td><td>0.521</td></tr><tr><td>Stable Diffusion with HPS-v2.1&amp;CLIP Score</td><td>0.335</td><td>0.395</td><td>0.215</td><td>0.522</td></tr></table>

表3 FLUX的结果。在此表中，我们展示了FLUX的结果，包括使用HPS评分训练的FLUX以及同时使用HPS评分和CLIP评分训练的FLUX。我们使用与表2相同的评估提示。

<table><tr><td>Models</td><td>HPS-v2.1 [25]</td><td>CLIP Score [26]</td><td>Pick-a-Pic [40]</td><td>GenEval [27]</td></tr><tr><td>FLUX</td><td>0.304</td><td>0.405</td><td>0.224</td><td>0.659</td></tr><tr><td>FLUX with HPS-v2.1</td><td>0.372</td><td>0.376</td><td>0.230</td><td>0.561</td></tr><tr><td>FLUX with HPS-v2.1&amp;CLIP Score</td><td>0.343</td><td>0.427</td><td>0.228</td><td>0.687</td></tr></table>

表4 不同方法的比较，基于表5 HunyuanVideo在Videoalign和扩散训练的结果，以及仅使用HPS分数和基于VideoAlign VQ&MQ训练的VisionReward的结果。“BaseCLIP分数。”“基线”指的是HunyuanVideo的原始结果。我们使用VisionReward的概率加权和。在附录E中可以找到与DDPO的更多比较。

<table><tr><td>Approach</td><td>Baseline</td><td>Ours</td><td>DDPO</td><td>ReFL</td><td>DPO</td></tr><tr><td>HPS-v2.1</td><td>0.239</td><td>0.365</td><td>0.297</td><td>0.357</td><td>0.241</td></tr><tr><td>CLIP Score</td><td>0.363</td><td>0.421</td><td>0.381</td><td>0.418</td><td>0.367</td></tr></table>

<table><tr><td>Benchmarks</td><td>VQ</td><td>MQ</td><td>TA</td><td>VisionReward</td></tr><tr><td>Baseline</td><td>4.51</td><td>1.37</td><td>1.75</td><td>0.124</td></tr><tr><td>Ours</td><td>7.03 (+56%)</td><td>3.85 (+181%)</td><td>1.59</td><td>0.128</td></tr></table>

# 3.2 文本生成图像

稳定扩散。稳定扩散 v1.4 是一个基于扩散的文本生成图像框架，包含三个核心组件：用于迭代去噪的 UNet 架构、基于 CLIP 的文本编码器用于语义条件化，以及用于潜在空间建模的变分自编码器（VAE）。如表 2 和图 1(a) 所示，我们提出的方法 DanceGRPO 在奖励指标上实现了显著提升，使得 HPS 分数从 0.239 提高到 0.365，CLIP 分数从 0.363 上升到 0.395。我们还采用了 Pick-a-Pic [40] 和 GenEval [27] 等指标来评估我们的方法。结果确认了我们方法的有效性。此外，如表 4 所示，我们的方法在各项指标上相较于其他方法表现最佳。我们参考 [15] 实现了 DPO 的在线版本。在基于基于规则的奖励模型（如 DeepSeek-R1）获得的见解基础上，我们进行了使用二元奖励公式的初步实验。通过将连续 HPS 奖励阈值设置为 0.28——为超过该阈值的奖励赋值 1（CLIP 分数为 0.39），否则赋值 0，我们构建了一个简化的奖励模型。图 3(a) 表示，尽管阈值化方法本质上简单，DanceGRPO 仍有效适应了这种离散化的奖励分布。结果表明，在视觉生成任务中，我们的奖励模型是有效的。未来，我们也将努力探索更强大的视觉奖励模型，例如，通过多模态大语言模型进行判断。最优 N 个推断扩展。我们通过最优 N 个推断扩展，以稳定扩散探索样本效率，如第 2.2 节所述。通过在从越来越大的样本池中选择的 16 个样本的子集（选择奖励最高 8 个和最低 8 个）上训练模型（每个提示包括 16、64 和 256 个样本），我们评估样本策展对稳定扩散收敛动态的影响。图 4(a) 显示，最优 N 个扩展显著加速了收敛。这强调了战略性样本选择在减少训练开销的同时保持性能的实用性。对于其他方法，如树搜索或贪婪搜索，我们将其系统集成推迟到未来研究。FLUX。FLUX.1-dev 是一个基于流的文本生成图像模型，在多个基准测试中推进了最先进的水平，利用比稳定扩散更复杂的架构。为了优化性能，我们整合了两种奖励模型：HPS 分数和 CLIP 分数。如图 1(b) 和表 3 所示，所提出的训练范式在所有奖励指标上实现了显著改善。HunyuanVideo-T2I。HunyuanVideo-T2I 是 HunyuanVideo 框架的文本生成图像适配，通过将潜在帧的数量减少到一个来重新配置。此修改将原始的视频生成架构转变为基于流的图像合成模型。我们进一步使用公开可用的 HPS-v2.1 模型进行系统优化，这是一个受人类偏好驱动的视觉质量度量。如图 1(c) 所示，该方法将平均奖励得分从大约 0.23 提高到 0.33，反映出与人类审美偏好的更好一致性。

![](images/1.jpg)  
Fgure1We visualize he reward curve Stable Diffusion,FLUX.1-dev, and Hunyuanideo-T2I n HPS score rom left o right. After applying CLIP score, the HPS score decreases, but the generated images become more natural (Figure 9 in Appendix), and the CLIP score improves (Tables 2 and 3).

# 3.3 文本到视频生成

HunyuanVideo。与文本到图像框架相比，优化文本到视频生成模型面临显著更大的挑战，主要是由于训练和推理过程中的计算成本提高，以及收敛速度较慢。在预训练方案中，我们始终采用渐进策略：初期训练集中于文本到图像生成，其次是低分辨率视频合成，最后以高分辨率视频优化为结束。然而，实证观察表明，仅依赖以图像为中心的优化会导致视频生成结果不理想。为了解决这个问题，我们的实现使用分辨率为480 $\times$ 480像素合成的训练视频样本，但我们可以以更大的像素可视化这些样本。

![](images/2.jpg)  
Ful r qalh qual quality on SkyReels-I2V.

此外，构建有效的视频奖励模型以训练对齐面临重大困难。我们的实验评估了几个候选模型：Videoscore [41] 模型表现出不稳定的奖励分布，使其在优化中不实用，而 Visionreward-Video [42] 作为一个 29 维度的指标，产生了语义一致的奖励，但在各个维度上存在不准确的问题。因此，我们采用了 VideoAlign [14]，这是一个多维框架，评估三个关键方面：视觉美学质量、运动质量和文本-视频对齐。值得注意的是，文本-视频对齐维度显示出显著的不稳定性，促使我们将其排除在最终分析之外。我们还增加了每秒采样帧的数量，以提高 VideoAlign 的训练稳定性。如图 2(a) 和图 2(b) 所示，我们的方法在视觉和运动质量指标上分别实现了 $5.6\%$ 和 $181\%$ 的相对提升。扩展的定性结果见表 5。

# 3.4 图像到视频生成

SkyReels-2V。SkyReels-I2V代表了最先进的开源图像到视频（I2V）生成框架，该框架于2025年2月在本研究开始时建立。该模型源自HunyuanVideo架构，通过将图像条件整合到输入连接过程中进行微调。我们研究的一个核心发现是，I2V模型仅允许优化运动质量，包括运动一致性和美学动态，因为视觉保真度和文本视频对齐在本质上受到输入图像属性的限制，而不是模型的参数空间。因此，我们的优化协议利用了VideoAlign奖励模型中的运动质量指标，在这一维度上实现了$1 1 8 \%$的相对改进，如图2(c)所示。我们必须启用CFG训练，以确保在RLHF训练过程中采样质量。

![](images/3.jpg)  
aWesalz herai crv ia wars.W ho e u valua resu FLUX (T2I), HunyuanVideo (T2V), and SkyReel (I2V), respectively.

# 3.5 人类评估

我们展示了使用内部提示和参考图像进行的人类评估结果。在文本到图像生成方面，我们对FLUX进行了240个提示的评估。在文本到视频生成方面，我们对HunyuanVideo进行了200个提示的评估，对于图像到视频生成，我们对SkyReels-I2V进行了200个与其对应参考图像配对的提示测试。如图3(b)所示，人类艺术家始终偏好经过RLHF优化的输出。更多可视化结果可以在附录的图10和附录F中找到。

# 3.6 消融研究

时间步选择的消融实验。如第2.2节所述，我们研究时间步选择对HunyuanVideo-T2I模型训练动态的影响。我们在三种实验条件下进行消融研究：(1) 仅在来自噪声的前30%的时间步上训练，(2) 在随机抽样的30%时间步上训练，(3) 在输出前最后40%的时间步上训练，(4) 在随机抽样的60%时间步上训练，(5) 在抽样的100%时间步上训练。如图4(b)所示，实证结果表明，初始的30%时间步对学习基础生成模式至关重要，这在模型性能的不成比例贡献中得到了证实。然而，限制仅在这一区间内训练会导致与全序列训练相比性能下降，关键在于对后期精炼动态暴露不足。为了平衡计算效率与模型保真度之间的权衡，我们在训练过程中始终实施40%的随机时间步丢弃策略。该方法在所有阶段随机屏蔽40%的时间步，同时保持潜在扩散过程中的时间连续性。研究结果表明，战略性时间步子抽样可以优化基于流的生成框架中的资源利用。关于噪声水平$\varepsilon_{t}$的消融实验。我们系统地研究噪声水平$\varepsilon_{t}$在FLUX训练期间的影响。我们的分析表明，降低$\varepsilon_{t}$会导致显著的性能下降，如图4(c)所定量展示的那样。值得注意的是，使用替代噪声衰减调度（例如，DDPM中使用的那些）的实验与我们的基线配置相比，输出质量没有统计显著差异。此外，噪声水平大于0.3有时在RLHF训练后会导致图像噪声。

![](images/4.jpg)  
FurThi ur hows herwar  Bes calnhlatitection,h ablationose evelspectivey.WhilBes ifernce scaln cnsstentlproves perorance wim samples, it reduces sampling efficiency. Therefore, we leave Best-of-N as an optional extension.

# 4 相关工作

大型语言模型的对齐。大型语言模型（LLMs）通常通过人类反馈的强化学习（RLHF）进行对齐。RLHF涉及基于模型输出的比较数据来训练一个奖励函数，以捕捉人类偏好，然后在强化学习中利用该奖励函数来对齐策略模型。某些方法利用策略梯度法，而其他方法则专注于直接策略优化（DPO）。策略梯度法已证明有效，但计算成本高昂，并且需要广泛的超参数调优。相比之下，DPO提供了更具成本效益的替代方案，但与策略梯度法相比，性能始终较差。最近，DeepSeek-R1展示了大规模强化学习与格式化和仅结果的奖励函数的应用可以引导LLMs朝向自我涌现的思维过程，使其能够进行人类般的复杂链式推理。这种方法在复杂推理任务中取得了显著优势，展现了推动大型语言模型推理能力的巨大潜力。AligDiff与人类反馈的对齐有所不同，但相比于LLMs，其探索仍然处于原始阶段。该领域的主要方法包括：（1）类似DPO的直接策略优化方法，（2）结合奖励信号的直接反向传播，如ReFL，以及（3）基于策略梯度的方法，包括DPOK和DDPO。然而，生产级模型主要依赖于DPO和ReFL，因为以前的策略梯度方法在大规模设置下表现出不稳定性。我们的工作解决了这一局限性，提供了一种增强稳定性和可扩展性的稳健解决方案。我们还希望我们的工作能为跨不同模态（例如图像和文本）统一优化范式的潜力提供见解。

# 5 结论与未来工作

本研究开创性地将群体相对策略优化（GRPO）集成到视觉生成中，建立了DanceGRPO作为一个统一框架，以增强扩散模型和校正流在文本到图像、文本到视频以及图像到视频任务中的表现。通过弥合语言和视觉模态之间的差距，我们的方法解决了先前方法的关键局限性，通过有效对齐人类偏好并强有力地扩展到复杂的多任务环境，实现了卓越的性能。实验表明在视觉真实感、运动质量和文本-图像对齐方面有显著的提升。未来的工作将探讨GRPO在多模态生成中的扩展，进一步统一生成性人工智能中的优化范式。

# References

information processing systems, 33:68406851, 2020.   
[2] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Börn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1068410695, 2022.   
[3] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.   
[4] Zeyue Xue, Guanglu Song, Qiushan Guo, Boxiao Liu, Zhuofan Zong, Yu Liu, and Ping Luo. Raphael: Textto-mage generation via large mixture of diffusion paths.Advances in Neural Information Processing Systems, 36:4169341706, 2023.   
[5] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.   
[ Xihao L, Chengyue Gong,and Qiang Liu. Flo traight and ast:Learing ogenerate nd transr ata with rectified flow. arXiv preprint arXiv:2209.03003, 2022.   
[ atric Esser, Smi Kulal, Andres Blatta, RaiEntai, Jnas Müller, Harry Saini Yam Levi Do Lorez Axel Sauer, FrederBoesel, e alScalng rectifed fotransormers or hig-resolution mage sythes. In Forty-first international conference on machine learning, 2024.   
[8] Lixue Gong, Xioia Hou, Fanshi Li, Liang Li, Xiaochn Lian, Fei Liu, Lyang Liu, Wei Liu, Wei Lu, Yichun S e al.Seedream 2.0:A native chinee-english bilngual imagegeneration foundation modelarXiv preprint arXiv:2503.07703, 2025.   
[9]J u, XiaL YuceWu,YTog, Qk  i ng, JieTang nuxo I: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:1590315935, 2023.   
[0] Jache Zhang, Jie Wu,Yuxi Ren, Xin Xia, Hua Kuag,Pan Xie, Jashi Li Xuee Xiao, We HuaShil We e alUniImprve latent diffus model vinid eedbackleararXiv preprnt arXiv:2404.05595, 2024.   
[11] Ming Li Taojann Yang, Huafeg Kuang, Jie Wu, Zhng Wang, Xueeng Xiao, and Chen Chen. Controlne++: i plus _plus. In European Conference on Computer Vision, pages 129147. Springer, 2024.   
[12] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, StefanoErmon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 82288238, 2024.   
[13] Ziyu Guo, Renrui Zhang, Chengzhuo Tong, Zhizheng Zhao, Peng Gao, Hongsheng Li, and Pheng-Ann Heng. Can we generate images with cot? let's verify and reinforce image generation step by step. arXiv preprint arXiv:2501.13926, 2025.   
[14] Jie Liu, Gongye Liu, Jiajun Liang, Zyag Yuan, Xiokun Liu, Ming Zheng, Xiele Wu, Qiulin Wang, Wenu Qin, Menghan Xia, et al. Improving video generation with human feedback. arXiv preprint arXiv:2501.13918, 2025.   
[15] Jiacheng Zhang, Jie Wu, Weifeng Chen, Yatai Ji, Xuefeng Xiao, Weilin Huang, and Kai Han. Onlinevpo:Align video diffusion model with online video-centric preference optimization. arXiv preprint arXiv:2412.15159, 2024.   
[16] Richard S Sutton, Andrew G Barto, et al. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.   
[17] John Schulman, Flip Wolski, PrafulaDhariwal, AlecRadord, and Oleg Klimov. roximal polcotiiation algorithms. arXiv preprint arXiv:1707.06347, 2017.   
[18] Kevin Black, Michael Janner, Yilun Du, ya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.   
[19] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models. Advances in Neural Information Processing Systems, 36:7985879885, 2023.   
[20] Daya Guo, Dejan Yang, Haowei Zhang, Junxio Song, Ruoyu Zhang, Runxin Xu, Qiao Zhu, Shirong Ma, Peyi Wan, Xiao Bi, et al.Deepseek-r Incentivizing reasoning capability in lls via reinorcement learning.arXiv preprint arXiv:2501.12948, 2025.   
[21] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxio Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.   
[22] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Janwei Zhang, et al. Hunyuanvideo:A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.   
[23] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.   
[24]SkyReels-AI.Skyreels v: Human-centricvideofoundationmodelhttps://github.com/SkyworkAI/SkyReels-V1, 2025.   
[25] Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hongsheng Li. Better aligning text-to-image models with human preference. arXiv preprint arXiv:2303.14420, 1(3), 2023.   
[26] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Aske Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 87488763. PmLR, 2021.   
[7] Drub hosh, Han Hajshirzi, and Ld Scmi. Geneval: Abject-oc fmeork orevalti text-to-image alignment. Advances in Neural Information Processing Systems, 36:5213252152, 2023.   
[28] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.   
[9] eng Wang, Shuai Bai, Sinan Tan, Shij Wang, Zhio an, Jinze Bai, KeqinChen, Xuejng Lu, Jial Wan, Wenbn Ge, et al. Qwen2-v: Enhancing vision-language model's perception of the world at any resolution.arXiv preprint arXiv:2409.12191, 2024.   
[0 Ya S n enrmo.Genetiveli y tiatiadient the daistrbutio.A in neural information processing systems, 32, 2019.   
[31] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic diferential equations. arXiv preprint arXiv:2011.13456, 2020.   
[32] Michael S Albergo and Eric Vanden-Eijnden. Building normalizing fows with stochastic interpolants.arXiv preprint arXiv:2209.15571, 2022.   
[33] Michael Albergo, Nicholas  Bof and EricVanden-Eijnden. Stochastic interpolants: unifying framework for flows and diffusions, 2023. URL https://arxiv. org/abs/2303.08797, 3.   
[34] Ruiq Gao, Emiel Hoogeboo, Jonathan Heek, Valentin DeBortoli, Kevin P. Murphy, and Tim Salimans. Diffuion meets flow matching: Two sides of the same coin. 2024.   
[35] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.   
[36] Wenhao Wang and Yi Yang.Vidprom:A miion-scale real prompt-gallery dataset for text-to-video diffusion models. arXiv preprint arXiv:2403.06098, 2024.   
[37SenYuan, Jina Hua Xiany He,Yuyag Ge, Yuju hi Lu Chen JieboLuo,and LiYuan. enypreserving text-to-video generation by frequency decomposition. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1297812988, 2025.   
[8] Hangliang Ding, Dachng Li, Runng u, Pey Zhang, Zhij Deng, Ion Stoica, and Hao ZhangEft-t: Efficient video diffusion transformers with attention tile, 2025.   
[39] Peiyuan Zhang, Yongqi Chen, Runlong Su, Hangliang Ding, Ion Stoica, Zhenghong Liu, and Hao Zhang. Fast video generation with sliding tile attention, 2025.   
[40] Yuval Kirstain Adam Polyak, Uriel Singer, Shahuland Matiana, JePena, and Omer Levy. Pck-a-pi:An oe dataset of user preferences for text-to-image generation.Advances in Neural Information Processing Systems, 36:3665236663, 2023.   
[41] Xuan He, Dongu Jang, Ge Zhang, Max Ku, Achint Soni, Sherman Siu, Haonan Chen, Abhranil Chandra, Ziyan JAarn Arul Videor:Buildiatoaic meri sulate nerai um ba video generation. arXiv preprint arXiv:2406.15252, 2024.   
[42] Jiazheng Xu, Yu Huang, Jiale Cheng, Yuanming Yang, Jiajun Xu, Yuan Wang, Wenbo Duan, Shen Yang, Qunin Jin, Shurun Li, et al. Visionreward: Finegrained multi-dimensional human preference learning for image and video generation. arXiv preprint arXiv:2412.21059, 2024.   
[43] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ige Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.   
[44] Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al. A survey on evaluation of large language models. ACM transactions on intelligent systems and technology, 15(3):145, 2024.   
[45] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:18771901, 2020.   
[46] Aaron Grattafori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The lama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.   
[47] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Lu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.   
[48] Long Ouyng, Jeffrey Wu, Xu Jiang, DiogAmeida, Carrol Wainright, Pamea Mishkin, Chong Zhang, Sanii Agarwal, KatariSlamaAl Ray, ealTraii angage modelstooostrucions wihuan eac. Advances in neural information processing systems, 35:2773027744, 2022.   
[49] Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Ren Lu, Thomas Mesnard, Johan Ferret, Colton Bishop, Ethan Hall, Victor Carbune, and Abhinav Rastogi. Rlaif:Scalng reinforcement learning from human feedback with ai feedback. 2023.   
[50] Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward modeloveroptimization. In International Conference on Machine Learning, pages 1083510866. PMLR, 2023.   
[51] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav F Dee Gangul Tom Henighan,  l Traig a helpful and hares assistant wih reinorcmet l from human feedback. arXiv preprint arXiv:2204.05862, 2022.   
[52] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direc preference optimization:Your language model is secrety a rewardmodel.Advances in Neural Information Processing Systems, 36:5372853741, 2023.   
[] Zhan LiangYuhuiYuan, Shuyng Gu, BohanChen, TiankaiHang, Ji Li, and Liang ZhengStep-awae eereoptimization: Aligning preference with denoising performance at eac step. arXiv preprint arXiv:2406.04314, 2(5):7, 2024.   
[54] Tao Zhang, Cheng Da, Kun Ding, Kun Jin, Yan Li, Tingting Gao, Di Zhang, Shiming Xiang, and Chunhong PDiffsn model as a oise-aware latent rewar model or step-evel preference optiization.arXiv preint arXiv:2502.01051, 2025.   
[55] Mihir Prabhudesai, Russel Mendonca, Zheyang Qin, Katerina Fragkiadaki, and Deepak Pathak. Video diffusion alignment via reward gradients. arXiv preprint arXiv:2407.08737, 2024.   
[56] Yyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiy Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Liang Zhao, et al. Janusflow: Harmonizing autoregression and rectified fow for unifed multimodal understanding and generation. arXiv preprint arXiv:2411.07975, 2024.

# Appendix

# A Experimental Settings

We provide detaile experimental settigs nTable6, whicapplyexclusively totraii without lassiree guidance (CFG). When enabling CFG, we configure one gradient update per iteration. Additionally, the sampling steps vary by model: we use 50 steps for Stable Diffusion, and 25 steps for FLUX and HunyuanVideo.

Table 6 Our Hyper-paramters.   

<table><tr><td>Learning rate</td><td>1e-5</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Gradient clip norm</td><td>1.0</td></tr><tr><td>Prompts per iteration</td><td>32</td></tr><tr><td>Images per prompt</td><td>12</td></tr><tr><td>Gradient updates per iteration</td><td>4</td></tr><tr><td>Clip range €</td><td>1e-4</td></tr><tr><td>Noise level εt</td><td>0.3</td></tr><tr><td>Timestep Selection T</td><td>0.6</td></tr></table>

# B More Analysis

# B.1 Stochastic Interpolants

The stochastic interpolant framework, introduced by [33], offrs a unifying perspective on generative models like rectified fows and score-based diffusion models. It achieves this by constructing a continuous-time stochastic process that bridges any two arbitrary probability densities, $\rho _ { 0 }$ and $\rho _ { 1 }$ .

In our work, we connect with a specic type of stochasticinterpolant known as spatially linear interpolants, defined in Section 4 of [33]. Given densities $\rho _ { 0 } , \rho _ { 1 } : \mathbb { R } ^ { d }  \mathbb { R } _ { \ge 0 }$ , a spatially linear stochasticinterpolant process $x _ { t }$ is defined as:

$$
\mathbf { z } _ { t } = \alpha ( t ) \mathbf { z } _ { 0 } + \beta ( t ) \mathbf { z } _ { 1 } + \gamma ( t ) \mathbf { \epsilon } , \quad t \in [ 0 , 1 ] ,
$$

where $\mathbf { z } _ { 0 } \sim \rho _ { 0 }$ , $\mathbf { z } _ { 1 } \sim \rho _ { 1 }$ , and $\mathbf { \epsilon } \gets \mathcal { N } ( 0 , I )$ is a standard Gaussian random variable independent of $\mathbf { z } _ { 0 }$ and $\mathbf { z } _ { 1 }$ . The functions $\alpha , \beta , \gamma : [ 0 , 1 ]  \mathbb { R }$ are sufficiently smooth and satisfy the boundary conditions:

$$
\alpha ( 0 ) = \beta ( 1 ) = 1 , \quad \alpha ( 1 ) = \beta ( 0 ) = \gamma ( 0 ) = \gamma ( 1 ) = 0 ,
$$

with the additional constraint that $\gamma ( t ) \geq 0$ for $t \in ( 0 , 1 )$ . The term $\gamma ( t ) \mathbf { z }$ introduces latent noise, smoothing the path between densities.

Specific choices within this framework recover familiar models:

Rectified Flow (RF): Setting $\gamma ( t ) = 0$ (removing the latent noise), $\alpha ( t ) = 1 - t$ , and $\beta ( t ) = t$ yields the linear interpolation ${ \bf z } _ { t } = ( 1 - t ) { \bf z } _ { 0 } + t { \bf z } _ { 1 }$ used in Rectified Flow [6, 33]. The dynamics are typically governed by an ODE $\mathrm { d } { \bf z } _ { t } = { \bf u } _ { t } \mathrm { d } t$ , where $\mathbf { u } _ { t }$ is the learned velocity field.

•Score-Based Diffusion Models (sBDM): The framework connects to SBDMs via one-sided linear interpolants (Section 4.4 of [33]), where $\rho _ { 1 }$ is typically Gaussian. The interpolant takes the form $\mathbf { z } _ { t } = \alpha ( t ) \mathbf { z } _ { 0 } + \beta ( t ) \mathbf { z } _ { 1 }$ . The VP-SDE formulation [31] corresponds to choosing $\alpha ( t ) = \sqrt { 1 - t ^ { 2 } }$ and $\beta ( t ) = t$ after a time reparameterization.

A pivotal insight is that: "The law of the interpolant $z _ { t }$ at any time $t \in [ 0 , 1 ]$ can be realized by many different processes, including an ODE and forward and backward SDEs whose drifts can be learned from data."

The stochastic interpolant framework provides a probability flow ODE for RF:

$$
\mathrm { d } { \mathbf { z } } _ { t } = { \mathbf { u } } _ { t } \mathrm { d } t .
$$

The backward SDE associated with the interpolant's density evolution is given by:

$$
\mathrm { d } { \mathbf { z } } _ { t } = { \mathbf { b } } _ { B } ( t , { \mathbf { x } } _ { t } ) \mathrm { d } t + \sqrt { 2 \epsilon ( t ) } \mathrm { d } { \mathbf { z } } ,
$$

where $\mathbf { b } _ { B } ( t , \mathbf { x } ) = \mathbf { u } _ { t } - \epsilon ( t ) \mathbf { s } ( t , \mathbf { x } )$ is the backward drift, $\underline { { \mathbf { s } ( t , \mathbf { x } ) } }$ is the score function, and $\epsilon ( t ) \geq 0$ is a tunable diffusion coefficient (noise schedule). If we set $\varepsilon _ { t } = \sqrt { 2 \epsilon ( t ) }$ , the backward SDE becomes:

$$
\mathrm { d } \mathbf { z } _ { t } = \left( \mathbf { u } _ { t } - \frac { \varepsilon _ { t } ^ { 2 } } { 2 } \nabla \log p _ { t } ( \mathbf { z } _ { t } ) \right) \mathrm { d } t + \varepsilon _ { t } \mathrm { d } \mathbf { z } ,
$$

which is the same as [34].

# B.2 Connections between Rectified Flows and Diffusion Models

Ne aim to demonstrate the equivalence between certain formulations of diffusion models and fow matching specifically, stochastic interpolants) by deriving the hyperparameters of one model from the other.

The forward process of a diffusion model is described by an SDE:

$$
\mathrm { d } \mathbf { z } _ { t } = f _ { t } \mathbf { z } _ { t } \mathrm { d } t + g _ { t } \mathrm { d } \mathbf { w } ,
$$

where dw is a Brownian motion, and $f _ { t } , g _ { t }$ define the noise schedule.

The corresponding generative (reverse) process SDE is given by:

$$
\mathrm { d } \mathbf { z } _ { t } = \left( f _ { t } \mathbf { z } _ { t } - \frac { 1 + \eta _ { t } ^ { 2 } } { 2 } g _ { t } ^ { 2 } \nabla \log p _ { t } ( \mathbf { z } _ { t } ) \right) \mathrm { d } t + \eta _ { t } g _ { t } \mathrm { d } \mathbf { w } ,
$$

where $p _ { t } ( \mathbf { z } _ { t } )$ is the marginal probability density of $\mathbf { z } _ { t }$ at time $t$

For flow matching, we consider an interpolant path between data $\mathbf { x } = \mathbf { z } _ { 0 }$ and noise $\epsilon$ (typically $\epsilon \sim \mathcal { N } ( 0 , \bf { I } )$

$$
\mathbf { z } _ { t } = \alpha _ { t } \mathbf { x } + \sigma _ { t } \epsilon .
$$

This path satisfies the ODE:

$$
\mathrm { d } \mathbf { z } _ { t } = \mathbf { u } _ { t } \mathrm { d } t , \quad \mathrm { w h e r e } ~ \mathbf { u } _ { t } = \dot { \alpha } _ { t } \mathbf { x } + \dot { \sigma } _ { t } \epsilon .
$$

This can be generalized to a stochastic interpolant SDE:

$$
\mathrm { d } \mathbf { z } _ { t } = ( \mathbf { u } _ { t } - \frac { 1 } { 2 } \varepsilon _ { t } ^ { 2 } \nabla \log p _ { t } ( \mathbf { z } _ { t } ) ) \mathrm { d } t + \varepsilon _ { t } \mathrm { d } \mathbf { w } .
$$

The core idea is to match the marginal distributions $p _ { t } ( \mathbf { z } _ { t } )$ generated by the forward diffusion process Eq.(16) with those implied by the interpolant path Eq.(18). We will derive $f _ { t }$ and $g _ { t }$ from this requirement, and then relate the noise terms of the generative SDEs Eq.(17) and Eq.(20) to find $\eta _ { t }$ .

Deriving $E [ \mathbf { z } _ { t } ] = \alpha _ { t } \mathbf { x }$ $f _ { t }$ by Matching Means. From Eq.(18), assuming The mean $\mathbf m _ { t } = E [ \mathbf z _ { t } ]$ q is fixed and $\mathbf { z } _ { 0 } = \mathbf { x }$ satisfies the ODE $E [ \epsilon ] = \mathbf { 0 }$ , the mean of $\begin{array} { r } { \frac { \mathrm { d } \mathbf { m } _ { t } } { \mathrm { d } t } = f _ { t } \mathbf { m } _ { t } } \end{array}$ $\mathbf { z } _ { t }$ is We require $\mathbf { m } _ { t } = \alpha _ { t } \mathbf { x }$ for all $t$ . Substituting into the mean ODE:

$$
\frac { \mathrm { d } } { \mathrm { d } t } ( \alpha _ { t } \mathbf { x } ) = f _ { t } ( \alpha _ { t } \mathbf { x } ) , \qquad \dot { \alpha } _ { t } \mathbf { x } = f _ { t } \alpha _ { t } \mathbf { x } .
$$

Assuming this holds for any $\mathbf { x }$ and $\alpha _ { t } \neq 0$ , we divide by $\alpha _ { t } \mathbf { x }$ .

$$
f _ { t } = { \frac { { \dot { \alpha } } _ { t } } { \alpha _ { t } } }
$$

Using the identity $\textstyle { \frac { \mathrm { d } } { \mathrm { d } t } } \log ( y ) = { \dot { y } } / y$ , we get:

$$
f _ { t } = \partial _ { t } \log ( \alpha _ { t } )
$$

Deriving $g _ { t } ^ { 2 }$ by Matching Variances. From Eq.(18), assuming $\mathbf { x }$ is fixed and $V a r ( \epsilon ) = \mathbf { I }$ (identity matrix for standard Gaussian noise), the variance (covariance matrix) of $\mathbf { z } _ { t }$ is $V a r ( \mathbf { z } _ { t } ) = V a r ( \alpha _ { t } \mathbf { x } + \sigma _ { t } \epsilon ) = \sigma _ { t } ^ { 2 } V a r ( \epsilon ) = \sigma _ { t } ^ { 2 } \mathbf { I }$ Let $V _ { t } = \sigma _ { t } ^ { 2 }$ be the scalar variance magnitude. The variance $V _ { t } = \mathrm { T r } ( V a r ( \mathbf { z } _ { t } ) ) / d$ for the process Eq.(16) oig $\begin{array} { r } { \frac { \mathrm { d } V _ { t } } { \mathrm { d } t } = 2 f _ { t } V _ { t } + g _ { t } ^ { 2 } } \end{array}$ (Here, $g _ { t } ^ { 2 }$ n injection rate). We require $V _ { t } = \sigma _ { t } ^ { 2 }$ .Substitute $V _ { t } = \sigma _ { t } ^ { 2 }$ and $f _ { t } = \dot { \alpha } _ { t } / \alpha _ { t }$ into the variance evolution equation:

$$
\frac { \mathrm { d } } { \mathrm { d } t } ( \sigma _ { t } ^ { 2 } ) = 2 \left( \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \right) \sigma _ { t } ^ { 2 } + g _ { t } ^ { 2 } , \qquad 2 \sigma _ { t } \dot { \sigma } _ { t } = 2 \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \sigma _ { t } ^ { 2 } + g _ { t } ^ { 2 } .
$$

Solving for $g _ { t } ^ { 2 }$

$$
g _ { t } ^ { 2 } = 2 \sigma _ { t } \dot { \sigma } _ { t } - 2 \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \sigma _ { t } ^ { 2 } = \frac { 2 } { \alpha _ { t } } ( \alpha _ { t } \sigma _ { t } \dot { \sigma } _ { t } - \dot { \alpha } _ { t } \sigma _ { t } ^ { 2 } ) = \frac { 2 \sigma _ { t } } { \alpha _ { t } } ( \alpha _ { t } \dot { \sigma } _ { t } - \dot { \alpha } _ { t } \sigma _ { t } )
$$

U $\begin{array} { r } { \partial _ { t } ( \sigma _ { t } / \alpha _ { t } ) = \frac { \alpha _ { t } \dot { \sigma } _ { t } - \dot { \alpha } _ { t } \sigma _ { t } } { \alpha _ { t } ^ { 2 } } } \end{array}$ , which implies $\alpha _ { t } \dot { \sigma } _ { t } - \dot { \alpha } _ { t } \sigma _ { t } = \alpha _ { t } ^ { 2 } \partial _ { t } ( \sigma _ { t } / \alpha _ { t } )$ Then we get:

$$
g _ { t } ^ { 2 } = \frac { 2 \sigma _ { t } } { \alpha _ { t } } \left( \alpha _ { t } ^ { 2 } \partial _ { t } \left( \frac { \sigma _ { t } } { \alpha _ { t } } \right) \right) = 2 \alpha _ { t } \sigma _ { t } \partial _ { t } \left( \frac { \sigma _ { t } } { \alpha _ { t } } \right)
$$

Thus, we have:

$$
g _ { t } ^ { 2 } = 2 \alpha _ { t } \sigma _ { t } \partial _ { t } \left( \frac { \sigma _ { t } } { \alpha _ { t } } \right)
$$

Deriving $\eta _ { t }$ by Matching Noise Terms in Generative SDEs. We compare the coeicients of the Brownian motion term (dw) in the reverse diffusion SDE Eq.(17) and the stochastic interpolant SDE Eq.(20). The diffusion coefficient (magnitude of the noise term) is $D _ { \mathrm { d i f f } } = \eta _ { t } g _ { t }$ . The diffusion coefficient is $D _ { \mathrm { i n t } } = \varepsilon _ { t }$ . To match the noise structure in these specific SDE forms, we set $D _ { \mathrm { d i f f } } = D _ { \mathrm { i n t } }$ : $\eta _ { t } g _ { t } = \varepsilon _ { t }$ . Solving for $\eta _ { t }$ (assuming $\begin{array} { r } { g _ { t } \neq 0 ) { : \eta _ { t } } = \frac { \varepsilon _ { t } } { g _ { t } } } \end{array}$ Substitute $g _ { t } = \sqrt { g _ { t } ^ { 2 } }$ using the result from Eq.(7):

$$
\eta _ { t } = \frac { \varepsilon _ { t } } { \sqrt { 2 \alpha _ { t } \sigma _ { t } \partial _ { t } ( \sigma _ { t } / \alpha _ { t } ) } }
$$

Summary of Results. By requiring the forward diffusion process Eq.(16) to match the marginal mean and variance of the interpolant path Eq.(18) at all times $t$ , we derived:

$$
f _ { t } = \partial _ { t } \log ( \alpha _ { t } ) , \quad g _ { t } ^ { 2 } = 2 \alpha _ { t } \sigma _ { t } \partial _ { t } ( \sigma _ { t } / \alpha _ { t } ) , \quad \eta _ { t } = \varepsilon _ { t } / ( 2 \alpha _ { t } \sigma _ { t } \partial _ { t } ( \sigma _ { t } / \alpha _ { t } ) ) ^ { 1 / 2 } .
$$

Thes relationships establish the equivalence between the parameters  the tworameworks under the specid conditions.

# • Classifier-Free Guidance (CFG) Training

Classifier-Free Guidance (CFG) [35] is a widely adopted technique for generating high-quality samples in conditional generative modeling. However, in our settings, integrating CFG into training pipelines introduces instability during optimization. To mitigate this, we empirically recommend disabling CFG during the sampling phase for models with high sample fidelity, such as HunyuanVideo and FLUX, as it reduces gradient oscillation while preserving output quality.

For CFG-dependent models like SkyReels-I2V and Stable Diffusion, where CFG is critical for reasonable sampl ualiy dentiykeystabiliy:raiexclusivelhendiialjectiveeaddiv optimization trajectories. This necessitates the joint optimization of both conditional and unconditional outputs, effectively doubling VRAM consumption due to dual-network computations. Morever, we propose reducing the frequency of parameter updates per training iteration. For instance, empirical validation shows that limiting updates to one per iteration significantly enhances training stability for SkyReels-I2V, with minimal impact on convergence rates.

# D Advantages over DDPO and DPOK

Our approach differs from prior RL-based methods for text-to-image diffusion models (e.g., DDPO, DPOK) in three key aspects: (1) We employ a GRPO-style objective function, (2) we compute advantages within prompt-level groups rather than globally, (3) we ensure noise consistency across samples from the same prompt, (4) we generalize these improvements beyond diffusion models by applying them to rectified flows and scaling to video generation tasks.

# E Inserting DDPO into Rectified Flow SDEs

We also insert DDPO-style objective function into rectified fow SDEs, but it always diverges, as shown in Figure 5, which demonstrates the superiority of DanceGRPO.

![](images/5.jpg)  
Figure 5 We visualize the results of DDPO and Ours. DDPO always diverges when applied to rectified fow SDEs

# F More Visualization Results

We provide more visualization results on FLUx, Stable Diffusion, and HunyuanVideo as shown in Figure 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, and 18.

![](images/6.jpg)  
Prompt: Generate a picture of a blue sports car parked on the road, metal texture   
FuWevisualize he results by selecti FLUX tiiz wit the HPS scorea rations 0, 60,120, 180,4 and The tizoutputs te exhibt rghertoesanrierdetailsHoweveincoorati I regularization is crucial, as demonstrated in Figures 11, 12, and 13.

![](images/7.jpg)  
Prompt: a basketball player wearing a jersey with the number _23_

![](images/8.jpg)  
Prpt: A boy wih yellow ha s cclng n a county road, realisic full shot, war color palette

![](images/9.jpg)  
Figur7Visulization  the iversiy themodel beore nd afterRLHF. Diffent ee tend  ete images after RLHF.   
hciz zHla waa r radiant reflections, sunlight, sparkle

![](images/10.jpg)  
gu This ure emostrate he pacf theLIP score The prop is  photf cup. We dthat the moel traine solely with HPS-v2.1 rewars tends toproduceunatural (oily)outputs, whil incorporatinCLIP scores helps maintain more natural image characteristics.

![](images/11.jpg)  
Figure 10Overall visualization. We visualize the results before and after RLHF of FLUX and HunyuanVido.

![](images/12.jpg)  
We e elutpt LUX,iv yhe P enhanced by both the HPS and CLIP scores.

![](images/13.jpg)  
FWe  eu LUX,v e P enhanced by both the HPS and CLIP scores.

![](images/14.jpg)  
W  uLUX,v  P enhanced by both the HPS and CLIP scores.

![](images/15.jpg)  
FigurWe preent herigialutputs HuyuanVideo-T alongsi tiizations drive solly by he HPS score and those enhanced by both the HPS and CLIP scores.

![](images/16.jpg)  
u  lu  a n   iole h and those enhanced by both the HPS and CLIP scores.

![](images/17.jpg)  
Figure 16 Visualization results of HunyuanVideo.

![](images/18.jpg)  
  
Figure 17 Visualization results of HunyuanVideo.

![](images/19.jpg)  
Prompt: Man walking in an abandoned city in the rain

![](images/20.jpg)  
Prompt: A man leaning on a tree at the beach

![](images/21.jpg)  
: h   
Figure 18 Visualization results of HunyuanVideo.

After RLHF

![](images/22.jpg)  
Prompt: A black Porsche drifting in the desert

RLHF

![](images/23.jpg)  
Prompt: Large light blue ice dragon breathing blue fire on town

![](images/24.jpg)  
Prompt: A young black girl walking down a street with a lot of huge trees

![](images/25.jpg)  
Figure 19 Visualization results of SkyReels-I2V.

Prompt: a cinematic shot of a group of ultra marathon runners, 1980s, a street surrounded by fields, morning hours, anamorphic lens, Kodak Filmstock