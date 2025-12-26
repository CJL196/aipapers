# 自回归视频扩散的端到端训练通过自重采样

Yuwei $\pmb{G u o}^{1,*}$，Ceyuan Yang²,†，Hao $\mathsf{H} \ @^{1}$，Yang Zhao²²，Meng Wei³，Zhenheng Yang³，Weilin Huang²，Dahua Lin¹ ¹香港中文大学，ByteDance Seed，³字节跳动 * 在字节跳动 Seed 完成的工作，† 通讯作者

# 摘要

自回归视频扩散模型在世界模拟中展现出潜力，但由于训练与测试之间的不匹配，易受到曝光偏差的影响。尽管近期的研究通过后训练方法解决了这一问题，但通常依赖于双向教师模型或在线鉴别器。为了实现端到端的解决方案，我们提出了重采样强制（Resampling Forcing），这是一个无教师的框架，能够从零开始大规模训练自回归视频模型。我们方法的核心是一个自重采样方案，该方案在训练过程中模拟推理时模型在历史帧上的错误。在这些降质的历史帧条件下，一个稀疏的因果掩码强制执行时间因果性，同时允许采用帧级扩散损失进行并行训练。为了促进高效的长时间生成，我们进一步引入历史路由，这是一种无参数机制，能够动态检索每个查询最相关的前 $k$ 个历史帧。实验表明，我们的方法在性能上与基于蒸馏的基线相当，并且由于原生长度训练，在较长视频上的时间一致性更优。日期：2025年12月18日 项目页面：[https://guoyww.github.io/projects/resampling-forcing](https://guoyww.github.io/projects/resampling-forcing)

# 1 引言

最近在生成视频模型方面的进展展示了其在世界建模中的强大潜力，通过近似物理动力学并预测基于当前观察的未来状态。实现这一愿景需要一种自回归视频生成范式，该范式根据过去的上下文预测下一帧，从而反映出物理世界的严格因果性质。除了世界模拟，这种范式还支持一系列多样的应用，涵盖游戏模拟、互动内容创作和时序推理。尽管其概念优雅，自回归视频生成仍面临重大挑战。主要问题是曝光偏差：在教师强制（teacher forcing）下，模型在训练时基于真实标注历史进行学习，但在推理时必须依赖自己生成的输出。这一训练和测试的不匹配可能导致错误累积，其中模型预测中的小错误在自回归推理中被放大，可能导致灾难性碰撞。此外，在自回归生成中扩展上下文加剧了注意力复杂性，为长时间预测的训练和推理带来了实际障碍。

![](images/1.jpg)  

FWeouRepliForg，一个用于视频重建的框架，旨在接收不同模式。顶部：教师通过累积误差来进行视频合成。为了在长视频上保持稳定质量，最近的工作采用了后训练策略，以减轻训练和测试间的不匹配，旨在将生成的视频分布与真实数据对齐。例如，自我强制（Self Forcing）首先自回归地生成完整视频，随后应用蒸馏或对抗目标来强制分布匹配。然而，依赖双向教师或在线鉴别器阻碍了从零开始可扩展的自回归视频模型的训练。双向教师还可能泄露未来信息，妨碍学生模型的严格时间因果关系。此外，扩展到更长序列的模型通常使用简单的滑动窗口注意力机制，忽视历史上下文的重要性变化，这可能削弱长期一致性。

在本研究中，我们提出了重采样强制，这是一种针对自回归视频扩散模型的端到端训练框架。我们从大语言模型中的下一词预测目标中获得灵感，将每帧的条件作用于其干净历史，并通过因果掩蔽在每帧的扩散损失下进行并行训练。我们认为，为了减轻错误传播和放大，模型必须训练成对输入扰动具有鲁棒性，同时保持干净的预测目标。为此，我们方法的核心是自重采样机制：模型首先在历史帧中诱导错误，然后利用这一降级历史来条件下一帧的预测。为了模拟推理时模型的错误，我们自回归地重采样每帧去噪轨迹的后半段，采用在线模型权重。这个过程与梯度反向传播分离，以避免捷径学习。此外，我们引入了一种历史路由机制，通过无参数的路由器动态检索最相关的前 $k$ 个历史帧，在长时间推演中保持接近恒定的注意力复杂度。实证结果表明，重采样强制有效减轻了自回归视频扩散模型中的错误积累，实现了与最先进的蒸馏模型相当的生成质量。利用本地长视频训练，我们的方法在长视频生成中超过了外推基线。此外，与蒸馏基线相比，我们的模型在因果依赖性方面表现出更严格的遵循。我们还证明了我们的历史路由机制在几乎没有质量损失的情况下达到了稀疏上下文，为长期生成提供了一种可行的记忆设计。我们期待这项工作能够推动未来视频世界模型的可扩展训练和长期记忆的发展。

# 2 相关工作

双向视频生成。双向视频生成指的是非因果模型，它共同合成所有帧，使得每一帧可以同时关注过去和未来的上下文。早期的工作利用GANs或改进的基于UNet的文本到图像扩散模型。受SoRA启发，该领域转向3D自编码器和可扩展的扩散变换器（DiT），其中所有视频词元通过自注意力进行交互，文本条件通过MMDiT风格的融合或独立的交叉注意力层注入。最先进的系统包括Veo、Seedance、Kling等商业模型，以及开源模型如CogVideoX、LTX-Video、HunyuanVideo和WAN。我们的方法基于DiT的视频扩散主干网络。自回归视频生成。最近，自回归视频生成由于其在世界和游戏模拟中的潜力而获得了显著的关注。它在因果分解下顺序生成视频，将每一帧的条件基于其历史上下文。自回归视频扩散的一个关键挑战是训练-测试不匹配导致的错误累积。早期直接使用教师强制的方法随着视频长度的增加而出现质量下降。为了解决这个问题，以前的工作在历史帧中注入小噪声，以近似推断时的降级。另一个方向采用扩散强制，给每一帧分配一个独立的噪声水平，以便在自回归推理过程中以任意噪声进行条件化。其他工作探索了通过滚动去噪框架放宽严格因果性。在这种设置中，滑动窗口内的视频帧保持非递减的噪声水平，并在前一帧达到目标时间步时启动生成。一些工作探索了计划插值策略，首先生成一个未来关键帧，然后插值中间帧。最近，自强制作为一种有前景的训练后解决方案出现在训练-测试对齐中。它首先自回归地生成整个视频，然后计算整体分布匹配损失。然而，它对在线鉴别器的依赖使得对抗损失或对预训练双向教师的蒸馏在可扩展性和训练可行性上面临挑战。我们的办法在推断模拟的洞察下，独特地支持端到端训练，无需借助辅助模型。基于模型预测的条件化。在自回归系统中，基于模型自身输出的条件化策略广泛应用于减轻暴露偏差。调度抽样在语言模型中例证了这种方法，通过用模型预测的词元替代地面真实序列的部分内容。在扩散模型中，自条件化通过将当前去噪步骤基于先前的估计来改进样本质量。对于视频生成，稳定视频无限采用了一种错误回收策略，以减少自回归长视频推断中的漂移。

高效注意力用于视频生成。作为高维时空信号，视频表示需要大量的词元，使得平方复杂度的注意力成为计算瓶颈。为了减轻复杂性，许多研究探索了视频生成的高效注意力设计。一类工作采用线性复杂度的注意力。其他研究利用注意力得分中的稀疏性，通过定义的注意力掩码修剪较少激活的词元。例如，Radical Attention观察到一个时空衰减并提出了一种稀疏掩码，随着时间距离的增加而缩小重要性。最近的一些工作还从大型语言模型中适配了先进的稀疏注意力设计。例如，MoC和VMoBA将混合块注意力集成到DiT模块中，其中注意力中的键和值通过top-$k$路由器动态选择。我们的工作集成了一种类似的路由机制，专门针对自回归视频生成中的长历史处理。

# 3 方法

我们首先回顾自回归视频扩散模型的背景，并在3.1节中分析曝光偏差问题。接着，我们在3.2节中介绍我们的重抽样强制算法以实现端到端训练，并在3.3节中阐述动态历史路由以实现高效的长时间注意力机制。

# 3.1 背景

定义。自回归（AR）视频扩散模型将视频生成分解为帧间自回归和帧内扩散 [30, 56]。具体而言，给定条件 $c$ ，$N$ 帧视频序列 $\pmb { x } ^ { 1 : N }$ 的联合分布表示为

$$
p ( \pmb { x } ^ { 1 : N } | c ) = \prod _ { i = 1 } ^ { N } p ( \pmb { x } ^ { i } | \pmb { x } ^ { < i } , c ) .
$$

为了从每个条件分布中采样，第 $i$ 帧 $\mathbf { x } ^ { i }$（在时间步 $t = 0$ 时记为 $\mathbf { \Delta } \mathbf { x } _ { 0 } ^ { i }$）是通过从高斯噪声 $\pmb { x } _ { 1 } ^ { i } \sim \mathcal { N } ( \mathbf { 0 } , I )$ 在时间步 $t = 1$ 开始解决逆时间常微分方程 (ODE) 合成的，可以通过数值求解器（如欧拉法）来计算。这里，神经网络 $\pmb { v } _ { \theta } ( \cdot )$ 具有参数 $\theta$，它参数化速度场 $\mathrm { d } \pmb { x } _ { t } ^ { i } / \mathrm { d } t$，并以历史帧 $\mathbf { \boldsymbol { x } } ^ { < \imath }$ 为条件，即 $\mathrm { d } \pmb { x } _ { t } ^ { i } / \mathrm { d } t = \pmb { v } _ { \theta } ( \pmb { x } _ { t } ^ { i } , \pmb { x } ^ { < i } , t , c )$。现代扩散模型通常采用扩散变压器 (DiT) [49] 架构，其中视频经过块划分并通过注意力机制处理 [63]。在实践中，通常每个自回归步骤生成一段帧。为简化起见，本文中将每段称为一帧。

$$
\pmb { x } ^ { i } = \pmb { x } _ { 1 } ^ { i } + \int _ { 1 } ^ { 0 } \pmb { v } _ { \theta } ( \pmb { x } _ { t } ^ { i } , \pmb { x } ^ { < i } , t , c ) \mathrm { d } t ,
$$

教师强迫。在训练这样的序列模型时，一种常见的方法是教师强迫，在这种情况下，模型的训练是为了在给定其真实标注历史 $\mathbf { \boldsymbol { x } } ^ { < i }$ 的情况下预测当前帧 $\mathbf { x } ^ { i }$。在流匹配 [43] 中，时间步 $t$ 的样本 $\mathbf { \Delta } \mathbf { x } _ { t } ^ { i }$ 是高斯噪声 $\epsilon ^ { i } \sim \mathcal { N } ( 0 , I )$ 和干净帧 $\mathbf { x } ^ { i }$ 之间的插值。

$$
\pmb { x } _ { t } ^ { i } = ( 1 - t ) \cdot \pmb { x } ^ { i } + t \cdot \pmb { \epsilon } ^ { i } .
$$

然后，网络 $\pmb { v } _ { \theta } ( \cdot )$ 被训练以回归速度 $\mathrm { d } \pmb { x } _ { t } ^ { i } / \mathrm { d } t = \epsilon ^ { i } - \pmb { x } ^ { i }$，通过最小化

$$
\begin{array} { r } { \mathcal { L } = \mathbb { E } _ { i , t , { \boldsymbol { x } } , { \boldsymbol { \epsilon } } } \left[ \| ( { \boldsymbol { \epsilon } } ^ { i } - { \boldsymbol { x } } ^ { i } ) - { \boldsymbol { v } } _ { \boldsymbol { \theta } } ( { \boldsymbol { x } } _ { t } ^ { i } , { \boldsymbol { x } } ^ { < i } , t , { \boldsymbol { c } } ) \| _ { 2 } ^ { 2 } \right] . } \end{array}
$$

在这里，$\pmb { v } _ { \theta } ( \cdot )$ 接受两个独立的序列作为输入：噪声帧 $\mathbf { \Delta } \mathbf { x } _ { t } ^ { i }$ 作为扩散样本和无噪声的真实标注帧 $\mathbf { \boldsymbol { x } } ^ { < i }$。因果掩码限制每个帧只能关注其干净的历史，从而实现所有帧的并行训练（见图 3 (b,c)）。在推理过程中，一旦生成了一个帧，其干净特征可以被缓存并用于后续帧的生成（KV 缓存）。因此，注意力查询的数量保持不变，而键和值随着视频长度的增加而增长。

![](images/2.jpg)  
Figure 2 Error Accumulation. Top: Models trained with ground truth input add and compound errors autoregressively. Bottom: We train the model on imperfect input with simulated model errors, stabilizing the long-horizon autoregressive generation. The gray circle represents the closest match in the ground truth distribution.

错误累积。在教师强迫下，每一帧的生成依赖于其真实标注历史。然而，在推理过程中，模型的预测不可避免地存在缺陷。换句话说，模型生成总是与真实分布存在非零差异，我们将其称为模型错误。在完美输入上训练的模型将通过自回归循环传播和积累这些错误，导致质量下降，并最终在长期推演中失败。（见图2顶部）。

# 3.2 提高错误鲁棒性

如上所述，教师强制的失败源于训练输入与推理输入之间的分布不匹配，这种情况受限于不可减少的模型错误。由于模型容量和训练样本的有限性，消除这些不易处理的错误并不可行。相反，我们提出在条件降解的情况下训练模型，同时保持无误差的预测目标。这使得模型不再严格遵守输入条件，虽然预测仍然存在不完美，但错误不再累积。相反，错误在自回归过程中稳定在近乎恒定的水平。为了模拟这一过程，我们必须在输入条件上考虑时间错误。我们追求一种端到端的、无教师的方法来实现这一点，优先考虑简单性和可扩展性。在自回归扩散中，驱动分布迁移的主要因素有两个：（1）来自不完美评分估计和离散化的帧内生成误差，主要影响高频细节；（2）通过自回归循环传播的帧间累积误差。

![](images/3.jpg)  
FRepligForTulat e-ie moeleo  osevios timestep $t _ { s }$ , then use the online model weights to autoregressively complete the remaining denoising steps. (b) The its clean history frames.

为了模拟来自两个方面的错误，我们引入了基于历史条件的自回归自重采样。为了模拟帧内错误，我们对后续去噪轨迹进行重采样，其中高频细节通常被合成。具体而言，我们通过公式（3）将真实视频帧 $\mathbf { x } ^ { i }$ 破坏到一个采样时间步 $t _ { s } \in ( 0 , 1 )$，以获得 $\mathbf { \Delta } \mathbf { x } _ { t _ { s } } ^ { i }$。随后，我们使用在线模型 $\pmb { v } _ { \theta } ( \cdot )$ 完成剩余的去噪步骤，生成包含模型误差的降级无噪声帧 $\tilde { \mathbf { x } } ^ { i }$。时间步 $t _ { s }$ 控制 $\tilde { \mathbf { x } } ^ { i }$ 与其真实版本 $\mathbf { x } ^ { i }$ 的接近程度。为了模拟帧间误差积累，我们对每一帧进行自回归重采样，基于降级历史帧 $\widetilde { \pmb { x } } ^ { < i }$ 进行条件采样（见图 3 (a)），即，

$$
\tilde { \pmb { x } } ^ { i } = \pmb { x } _ { t _ { s } } ^ { i } + \int _ { t _ { s } } ^ { 0 } \pmb { v } _ { \theta } ( \pmb { x } _ { t } ^ { i } , \tilde { \pmb { x } } ^ { < i } , t , c ) \mathrm { d } t .
$$

利用模式通过不断学习来修正模型当前的缺陷。梯度在此过程中是独立的，以防止捷径学习。在实践中，这个过程可以通过 KV 缓存高效实现。

采样仿真时间步。方程（5）中的时间步 $t _ { s }$ 决定了历史忠实性与误差修正灵活性之间的权衡。较小的 $t _ { s }$ 导致低重采样强度，使得样本 $\tilde { \mathbf { x } } ^ { i }$ 与其真实值 $\mathbf { x } ^ { i }$ 非常相似。这鼓励模型保持对历史帧的忠实，但会增加错误累积的风险（教师强制是一个极限情况，其中 $t _ { s } = 0$）。另一方面，较大的 $t _ { s }$ 给予更大的误差修正灵活性，但提高了内容漂移的风险，因为模型被允许显著偏离历史上下文。因此，$t _ { s }$ 的分布应集中在中间值，同时抑制极端值。为了建模这一点，我们选择从满足上述属性的对数正态分布 LogitNormal $( 0 , 1 )$ 中采样 $t _ { s }$：

$$
\mathrm { l o g i t } ( t _ { s } ) \sim \mathcal { N } ( 0 , 1 ) .
$$

通常，更强大的模型会引入更少的错误，从而允许在低重采样强度上有更大的侧重，反之亦然。为了手动偏置 $t _ { s }$ 的分布，受到 [18] 的启发，我们在从标准逻辑正态分布采样 $t _ { s }$ 之后，应用带参数 $s$ 的时间步移位。

$$
{ t _ { s } } \gets s \cdot { t _ { s } } / \left( { 1 + ( s - 1 ) \cdot { t _ { s } } } \right) .
$$

在实现过程中，我们设置 $s < 1$ 以在低噪声部分施加更多权重。重采样后，我们使用退化视频 $\tilde { \pmb { x } } ^ { 1 : N }$ 作为历史条件，并将真实视频 $\pmb { x } ^ { 1 : N }$ 作为训练目标。重采样强制的完整伪代码如算法 1 所示。教师强制热身。在初始训练阶段，模型尚未收敛到因果架构，无法自回归地生成有意义的内容。模型在这方面的误差

# 算法 1 重采样强制阶段主要受随机初始化的影响，而非特定的帧内缺陷或帧间累积。因此，进行历史自适应重采样可能导致不规范的学习信号，并阻碍收敛。因此，我们首先使用教师强制对模型进行热身。一旦模型获得基本的自回归能力（尽管不完美），我们便转向重采样强制并继续训练。

<table><tr><td>Require: Video Dataset D</td><td></td><td></td></tr><tr><td></td><td>Require: Shift Parameter s</td><td></td></tr><tr><td></td><td>Require: Autoregressive Video Diffusion Model vθ(·)</td><td></td></tr><tr><td></td><td>1: while not converged do</td><td></td></tr><tr><td>2:</td><td>ts ∼ LogitNormal(0, 1)</td><td> sample simulation timestep</td></tr><tr><td>3:</td><td>ts ← s · ts/ (1 + (s − 1) · ts)</td><td> shift timestep (equation (7))</td></tr><tr><td>4:</td><td>Sample video and condition (x1:N , c) ∼ D</td><td></td></tr><tr><td>5:</td><td></td><td></td></tr><tr><td>6:</td><td>with gradient disabled do</td><td></td></tr><tr><td>7:</td><td>for i = 1 to N do</td><td> autoregressive resampling</td></tr><tr><td>8:</td><td>L0 vθ(xt, x&lt;i, t, c) dt</td><td>&gt; using numerical solver and KV cache (equation (5))</td></tr><tr><td>9:</td><td>Jts end for</td><td></td></tr><tr><td>10:</td><td>end with</td><td></td></tr><tr><td>11:</td><td>Sample training timestep ti</td><td></td></tr><tr><td>12:</td><td>Sample i ∼ N (0, I)</td><td></td></tr><tr><td>13:</td><td>xti ← (1 − ti) · xi + ti · i</td><td></td></tr><tr><td>14:</td><td>N 1 L ← ∑I ∥(i − xi) − vθ(xi, x&lt;i, ti, c)k2 N</td><td>&gt; parallel training with causal mask (equation (4))</td></tr><tr><td>15:</td><td>i=1 Update θ with gradient descent</td><td></td></tr><tr><td></td><td>16: end while</td><td></td></tr><tr><td></td><td>17: return θ</td><td></td></tr></table>

# 3.3 路由历史上下文

在自回归生成中，随着视频长度的增加，历史帧的数量也逐渐增加。这使得后续帧的生成速度减缓，因为密集因果注意力需要关注整个历史上下文。为了解决这个问题，一个常见的解决方案是限制注意力接受域至一个局部滑动窗口。然而，这种方法妥协了长期依赖性，牺牲了全局一致性，并加剧了漂移问题。为了保持稳定的注意力复杂度，我们选择可选地用动态路由机制替换密集因果注意力，这受到大型语言模型中先进稀疏注意力的启发。具体来说，对于第 $i$ 帧的查询词元 $\pmb q _ { i }$，我们动态检索并关注与之最相关的 $k$ 个历史帧，而不是关注整个历史（见图 4），即 $\Omega ( \pmb q _ { i } )$ 是与查询 $\pmb { q } _ { i }$ 选定的 $k$ 个历史帧的索引集合。作为选择度量，我们使用 $\pmb { q } _ { i }$ 和帧描述符 $\phi ( \pmb { K } _ { j } )$（对于第 $j$ 帧）的点积，即：

![](images/4.jpg)  
Figure 4 History Routing Mechanism. Our routing mechanism dynamically selects the top- $k$ important frames to attend. In this illustration, we show a $k = 2$ example, where only the 1st and 3rd frames are selected for the 4th frame's query token $\mathbf { q } _ { 4 }$ .

$$
\mathrm { A t t e n t i o n } ( q _ { i } , K _ { < i } , V _ { < i } ) = \mathrm { S o f t m a x } \left( \frac { q _ { i } K _ { \Omega ( q _ { i } ) } ^ { \top } } { \sqrt { d } } \right) \cdot V _ { \Omega ( q _ { i } ) } ,
$$

$$
\Omega ( \pmb { q } _ { i } ) = \arg \operatorname* { m a x } _ { \Omega ^ { * } } \sum _ { j \in \Omega ^ { * } } \left( \pmb { q } _ { i } ^ { \top } \phi ( \pmb { K } _ { j } ) \right) .
$$

根据文献 [8, 46, 69]，我们将均值池化用作描述符变换 $\phi ( \cdot )$，因为它符合注意力得分计算，并且不需要参数。这将每个 token 的注意力复杂度从线性 $\mathcal { O } ( L )$ 降低到常数 ${ \mathcal { O } } ( k )$ ，随着历史帧数 $L$ 的增加，达到了 $1 - k / L$ 的注意力稀疏性。值得注意的是，虽然可以将 $k$ 设置得很小以获得高稀疏性，但路由机制以头和 token 为单位运作，这意味着跨不同注意力头和空间位置的 tokens 可以路由到不同的历史混合，从而共同产生一个显著大于 $k$ 帧的感受野。在实现中，遵循 MoBA [46]，我们采用高效的两分支注意力，将一个框架内的注意力路径融合，仅针对其自身框架内的 tokens。在历史分支中，我们为每个查询 token 选择最多前 $k$ 个相关的历史帧。两个分支都通过 FlashAttention [15, 16] 的 flash_attn_varlen_func() 接口高效实现。输出通过对齐其对数和指数项进行组合，得到等同于对键的联合进行单一 softmax 的结果。

# 4 实验

模型。我们基于WAN2.1-1.3B架构构建了我们的方法，并加载其预训练权重以加快收敛速度。原始模型使用双向注意力并生成5秒的视频（81帧），分辨率为$480 \times 832$。我们修改了时间步条件以支持每帧的噪声水平，并实现了图3(c)中的稀疏因果注意力，采用torch.flex_attention()，不增加额外参数。按照文献[14, 32, 73]，我们使用3个潜在帧的块大小作为自回归单元。训练。在切换到因果注意力后，模型在5秒视频上使用教师强迫目标进行了10K步的热身，然后过渡到重采样强迫，并在$5 \mathrm{s}$和15秒（249帧）视频上分别顺序训练15K和5K步。随后，我们在15秒视频上启用稀疏历史路由进行了1.5K次迭代的微调。训练批次大小为64，AdamW优化器的学习率为$5e-5$。我们将时间步偏移因子设置为$s = 0.6$（第3.2节），并在top-$k$历史路由中设置$k = 5$（第3.3节）。为了提高效率，我们对历史重采样使用1步Euler求解器（公式(5)）。推理。我们使用一致的推理设置生成所有视频帧。我们使用32步的Euler采样器，时间步偏移因子为5.0。所有帧的无分类器引导比例为5.0。

# 4.1 比较

基准。我们将我们的方法与近期的自回归视频生成基准进行比较，包括 SkyReelsV2 [1]、MAGI-1 [60]、NOVA [17]、Pyramid Flow [35]、CausVid [77]、Self Forcing [32] 以及一项同时进行的工作 LongLive [73]。值得注意的是，SkyReel-V2 作为一个片段级自回归模型，按顺序生成 5 秒的视频片段。MAGI-1 放宽了严格的因果约束，在当前片段生成完成之前就开始下一个片段的去噪。LongLive 采用与 Self Forcing 相同的原则，但生成更长的视频，然后截取 5 秒的子片段并与教师模型计算蒸馏损失。 定性比较。我们在图 5 中提供了不同方法的视觉定性比较，所有模型都被提示生成 15 秒的视频。在上方面板中，我们比较了不同时间点的视觉质量，观察到大多数严格的自回归模型（例如 Pyramid Flow [35]、CausVid [77] 和 Self Forcing [32]）表现出错误积累，表现为颜色、纹理和整体清晰度的逐渐下降。具体而言，CausVid 趋向于过饱和，而 Pyramid Flow 和 Self Forcing 则显示出颜色和纹理畸变。相比之下，放宽严格因果关系的方法（MAGI-1 [60]）或使用较大自回归片段的方法（SkyReels-V2 [11]）可以减轻长时间水平的降解。然而，这些放宽的设置妥协了严格自回归的内在优势，例如逐帧交互性和未来状态预测的真实因果依赖性。

![](images/5.jpg)  
Figure 5 Qualitative Comparisons. Top: We compare with representative autoregressive video generation models, showin ur method's stable quality on long video generation.Bottom:Compared with LongLive [73] that istill from shor bcialacr, urmethoexhibts bettecusaliye usedashe ienoe thees liquid level, and red arrows to highlight the liquid level in each frame.

在严格的自回归范式下，我们的方法在长期视觉质量上表现出比基线更强的稳健性。

在下方面板中，我们进一步与同时期的自回归模型LongLive进行比较，该模型首先生成长视频，然后通过短时教师进行子片段蒸馏。尽管LongLive在长程视觉质量方面表现强劲，但我们发现从短双向教师进行的蒸馏无法确保严格的时间因果关系，即使使用时间因果学生架构。在图5中的“持续倒入”示例中，LongLive产生的液位在持续倒入的情况下先上升后下降，这违反了物理法则。相比之下，我们的模型维持了严格的时间因果关系：液位单调上升，同时源容器在排空。我们将这种非因果行为归因于两个因素。首先，双向教师本质上是非因果的，允许未来的信息通过注意力机制影响早期帧，从而在蒸馏过程中泄露未来的上下文。其次，子片段蒸馏强调局部外观质量而忽视全局因果关系。相反，我们的训练严格排除了来自未来的信息泄露。

定量比较。我们使用 VBench 提供的自动评估指标来评估方法。所有模型均被提示生成 15 秒的视频，我们将其分为三个部分并分别评估，以更好地评估长期质量。结果在表中总结，如表中所示，该方法在所有视频长度上保持了可比的视觉质量和优越的时间质量，相较于基线模型。在较长的视频长度上，我们方法的性能也与长视频蒸馏基线 LongLive 相匹配。鉴于蒸馏基的方法（即 CausVid、Self Forcing 和 LongLive）需要一个预训练的 140 亿参数双向教师，我们的方法为训练自回归视频模型提供了实质性的效率和实用性。此外，历史路由机制在仅产生相对于密集注意基线的微不足道的下降的同时，实现了 $75\%$ 的注意力稀疏性，展示了其在受限计算和内存预算下的长期生成潜力。表1 定量比较。我们将生成的15秒视频分为三部分，分别为 5 秒、10 秒和 15 秒，并使用 VBench 进行单独评估。

<table><tr><td rowspan="2">Method</td><td rowspan="2">#Param</td><td rowspan="2">Teacher Model</td><td colspan="3">Video Length = 015 s Temporal</td><td colspan="3">Video Length = 510 s</td><td colspan="3">Video Length = 1015 s</td></tr><tr><td></td><td>Visual</td><td>Text</td><td>Temporal</td><td>Visual</td><td>Text</td><td>Temporal</td><td>Visual</td><td>Text</td></tr><tr><td>SkyReels-V2 [11]</td><td>1.3B</td><td></td><td>81.93</td><td>60.25</td><td>21.92</td><td>84.63</td><td>59.71</td><td>21.55</td><td>87.50</td><td>58.52</td><td>21.30</td></tr><tr><td>MAGI-1 [60]</td><td>4.5B</td><td></td><td>87.09</td><td>59.79</td><td>26.18</td><td>89.10</td><td>59.33</td><td>25.40</td><td>86.66</td><td>59.03</td><td>25.11</td></tr><tr><td>NOVA [17]</td><td>0.6B</td><td>-</td><td>87.58</td><td>44.42</td><td>25.47</td><td>88.40</td><td>35.65</td><td>20.15</td><td>84.94</td><td>30.23</td><td>18.22</td></tr><tr><td>Pyramid Flow [ [35]</td><td>2.0B</td><td></td><td>81.90</td><td>62.99</td><td>27.16</td><td>84.45</td><td>61.27</td><td>25.65</td><td>84.27</td><td>57.87</td><td>25.53</td></tr><tr><td>CausVid [77]</td><td>1.3B</td><td>WAN2.1-14B(5s)</td><td>89.35</td><td>65.80</td><td>23.95</td><td>89.59</td><td>65.29</td><td>22.90</td><td>87.14</td><td>64.90</td><td>22.81</td></tr><tr><td>Self Forcing [32]</td><td>1.3B</td><td>WAN2.1-14B(5s)</td><td>90.03</td><td>67.12</td><td>25.02</td><td>84.27</td><td>66.18</td><td>24.83</td><td>84.26</td><td>63.04</td><td>24.29</td></tr><tr><td>LongLive [ [73]</td><td>1.3B</td><td>WAN2.1-14B(5s)</td><td>81.84</td><td>66.56</td><td>24.41</td><td>81.72</td><td>67.05</td><td>23.99</td><td>84.57</td><td>67.17</td><td>24.44</td></tr><tr><td>Ours (75% sparsity)</td><td>1.3B</td><td>-</td><td>90.18</td><td>63.95</td><td>24.12</td><td>89.80</td><td>61.95</td><td>24.19</td><td>87.03</td><td>61.01</td><td>23.35</td></tr><tr><td>Ours</td><td>1.3B</td><td></td><td>91.20</td><td>64.72</td><td>25.79</td><td>90.44</td><td>64.03</td><td>25.61</td><td>89.74</td><td>63.99</td><td>24.39</td></tr></table>

# 4.2 分析性研究

在本节中，我们对设计组件进行消融实验，并分析模型行为。误差模拟策略。在3.2节中，我们假设在训练过程中将模型暴露于不完美的历史上下文可以减轻误差累积，并提出自回归自重采样以模拟模型错误。我们与两种替代方案进行比较：噪声增强和并行重采样。在第一种方案中，向历史帧中添加小的高斯噪声以提高质量。表2 显示了误差模拟策略。自回归重采样实现了最佳效果。

<table><tr><td>Simulation Strategies</td><td colspan="3">Video Length = 015 s</td></tr><tr><td></td><td>Temporal</td><td>Visual</td><td>SText</td></tr><tr><td>noise augmentation</td><td>87.15</td><td>61.90</td><td>21.44</td></tr><tr><td>resampling - parallel</td><td>88.01</td><td>62.51</td><td>24.51</td></tr><tr><td>resampling - autoregressive</td><td>90.46</td><td>64.25</td><td>25.26</td></tr></table>

对推理错误的鲁棒性。在第二种方法中，所有历史帧都是并行重采样，而不是自回归重采样。如表 2 所示，自回归重采样策略的质量最高，其次是并行重采样和噪声增强。我们将此归因于加性噪声与模型推理时错误模式之间的不匹配，以及并行重采样仅捕获逐帧降级而忽略了时间轴上的自回归积累。

仿真时间步移动。我们分析了偏置时间步$t_{s}$分布的移动因子$s$。在公式（7）中定义的小$s$使$t_{s}$集中在低噪声区域，而大的$s$则将$t_{s}$向高噪声区域移动。相应地，小$s$对应于较弱的历史重采样，鼓励与过去内容的一致性，而大的$s$则强制进行更强的重采样，促进可能导致漂移的内容修改。我们观察到模型性能对$s$的选择具有鲁棒性；因此，我们在这次消融实验中采用极端值以更好地可视化移动因子的影响。在图6中，使用小$s$训练的模型表现出误差积累和质量下降，而非常大的$s$则降低了与历史的一致性，增加了初始内容漂移的风险。因此，适中的$s$值对于在减轻误差积累和防止漂移之间取得平衡至关重要。

![](images/6.jpg)  

提示：一只红色气球漂浮在一条废弃的街道上。.. 图6 比较时间步移动。对重采样时间步 $t _ { s }$ 进行适度的移动尺度是必要的，以平衡误差累积和内容漂移之间的关系。

稀疏历史策略。我们比较了三种在自回归生成中利用历史上下文的机制：稠密因果注意力、动态历史路由和滑动窗口注意力 [32, 42]。我们在图7中展示了稠密注意力、前5和前1路由以及滑动窗口注意力的定性结果。如图所示，前5个历史帧的路由产生了 $7 5 \%$ 的稀疏性，同时其质量可与稠密注意力相媲美。从前5到前1的减少（ $9 5 \%$ 的稀疏性）仅导致轻微的质量下降，证明了路由机制的稳健性。我们进一步对比了前1路由与大小为1的滑动窗口。尽管稀疏性相同，路由机制在鱼的外观保持了一致性上表现更优。我们假设滑动窗口注意力的固定和局部感受野加剧了漂移的风险。相比之下，我们的动态路由使每个查询词元能够选择多样的历史上下文组合，共同产生更大的有效感受野，更好地保持全球一致性。

![](images/7.jpg)  

热带鱼在五彩斑斓的珊瑚礁中游动… 图7 稀疏历史策略。我们对密集因果注意力、动态历史路由和滑动窗口注意力在外观一致性方面进行了比较。

历史路由频率。为了更深入地了解历史路由，我们对 $k = 1 , 3 , 5 , 7$ 进行了实验，并可视化了在当前帧生成过程中每个历史帧的选择频率。如图 8 所示，选择频率呈现出一种混合的“滑动窗口”和“注意力汇聚”模式：路由器优先考虑最初的帧以及紧接目标之前的最新帧。在极端稀疏性下（$k = 1$），这一效果最为明显，随着稀疏性的降低（$k = 1 7$），选择频率变得更加分散，涵盖了更广泛的中间帧。该观察结果为最近推出的将“帧汇聚”与滑动窗口结合的注意力设计提供了实证支持，适用于长视频的推演 [73]，可以视为我们方法的一种特例。我们的结果表明了一种替代注意力稀疏的方法：用动态的、内容感知的路由机制替代启发式掩码，能够探索更大范围的上下文组合空间。

![](images/8.jpg)  
Figure 8 History Routing Frequency. We visualize the beginning 20 frames' frequency of being selected when generating the 21st frame. For readability, the maximum bar is truncated and labeled with its exact value.

# 5 讨论

我们提出了重采样强制，这是一个端到端、无教师框架，用于训练自回归视频扩散模型。通过识别误差累积的根本原因，我们提出了一种历史自重采样策略，有效减轻了这一问题，确保了长期生成的稳定性。此外，我们引入了一种历史路由机制，旨在保持近乎恒定的注意力复杂度，尽管历史上下文不断增加。实验结果表明我们的方法在视觉质量和高稀疏性注意力下的稳健性上表现优越。局限性：作为一种基于扩散的方法，我们的模型在推理时需要迭代去噪步骤。实现实时延迟可能需要后续加速，例如少步骤蒸馏或改进的采样器。此外，我们的训练涉及处理双序列（扩散样本和干净历史），这可以通过与建筑优化类似的方式进一步改进。

# References

[1] Eloi Alonso, Adam Jelley, Vincent Michel, Anssi Kanervisto, Amos J Storkey, Tim Pearce, and Françis Fleuret. Diffusionfor world modeling:Visual details matter in atari.Advances in Neural Information Processing Systems, 37:5875758791, 2024.   
[2] Phil J.Ball, Jakob Bauer, Frank Belltt Bethanie Brownel, Ariel Ephrat, Shlomi Fruchter, Agrim Gupta, Kristian Holsheimer, Aleksander Holynski, Jiri Hron, Christos Kaplanis, Marjorie Limont, Matt McGill, Yanko Olveira, Jack Parkr-Holder, Frank Perbet,Guy Scully, Jremy Shar, Stephen Spencer, Omer Tov, Ruben Vlleas, Emma Wang, Jessica Yung, CipBaetu, Jordi Berbel, David Bridson, Jake Bruce, GavinButtimore, Sarah Chakera, Bilva Chandra, Paul Collins, Alex Cullum, Bogdan Damoc, Vibha Dasagi, Maxime Gazeau, Charles Gbadamosi, Woohyun Han, Ed Hirst, Ashyana Kachra, Lucie Kerley, Kristian Kjems, Eva Knoepfel, Vika Koriakin, Jessica Lo, Cong Lu, Zeb Mehring, Alex Moufarek, Henna Nandwani, Valeria Oliveira, Fabio Pardo, Jane Park, Andrew Pierson, Ben Poole, Helen Ran, Tim Salimans, Manuel Sanchez, Igor Saprykin, Amy Shen, Sailesh Sidhwani, Duncan Smith, Joe Stanton, Hamish Tomlinson, Dimple Vijaykumar, Luyu Wang, Piers Wingfield, Nat Wong, Keyang Xu, Christopher Yew, Nick Young, Vadim Zubov, Douglas Eck, Dumitru Erhan, Koray Kavukcuoglu, Demis Hassabis, Zoubin Gharamani, Raia Hadsell, Aäron van den Oord, Inbar Mosseri, Adrian Bolton, Satinder Singh, and Tim Rocktäschel. Genie 3: A new frontier for world models, 2025.   
[3] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, et al. Lumiere: A space-time diffusion model for video generation. In SIGGRAPH Asia 2024 Conference Papers, pages 111, 2024.   
[4] Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. Advances in neural information processing systems, 28, 2015.   
[5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi Zion English, Vikram Voleti, Adam Letts, et alStable vidodiffusion: Scalnglatent videodiffusion odels to large datasets. arXiv preprint arXiv:2311.15127, 2023.   
[6] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2256322575, 2023.   
[7] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators, 2024. URL https://openai.com/research/video-generation-models-as-world-simulators.   
[8] Shengqu Cai, Ceyuan Yang, Lvmin Zhang, Yuwei Guo, Junfei Xiao, Ziyan Yang, Yinghao Xu, Zhenheng Yang, AlanYuille, Leonidas Guibas alMixture contexts orlongvidogeneration.rXiv prerin arXiv:2508.1058, 2025.   
[9] Zhepeng Cen, Yao Liu, Silang Zeng, Pratik Chaudhari, Huzfa Rangwala, George Karypis, and Raso Fakoor. Briging thetraining-inference gap in ls by leveraging sel-generated tokensarXiv preprintarXiv:2410.1655, 2024.   
[10] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diusion.Advances inNeural Information Procesing Systems, 37:2408124125, 2025.   
[11] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, et al. Skyreels-v2: Infinite-length flm generative model. arXiv preprint arXiv:2504.13074, 2025.   
[12] Juns Chen, Yuyang Zhao, Jinchengu, Ruihahu, Junyu Chen, Shuai Yang, Xianba Wng, Yicheg Pan, Daquan Zhou, Huan Ling, et al. Sana-video: Effiient video generation with block linear diffusion transformer. arXiv preprint arXiv:2509.24695, 2025.   
[13] Ting Chen, Ruixing Zhang, nd Geoffy Hinton.Analog bits: Generating discte data using diuion oels with self-conditioning. arXiv preprint arXiv:2208.04202, 2022.   
[14] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hieh. Self-forcing++: Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283, 2025.   
[15] Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In Interational Conference on Learning Representations (ICLR), 2024.   
[ Ti Dao DanFu, Stamn, AiRudra an Crisher lashAtte: astne-ent exact attention with IO-awareness. In Advances in Neural Information Processing Systems (NeurIPS), 2022.   
[1] Haoge Deng, Ting Pan, Haiwen Diao, Zhenxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, and Xinlong Wang. Autoregressive video generation without vector quantization. arXiv preprint arXiv:2412.14169, 2024.   
[18] Patric Essr, Smi KulaAndres Blata RhiEntear Jonas Mü, Harry Sai YmLevi D Lore, AxelSauer, FredericBoesel, e alScalng rectife fowtransormers or hig-resolution mage syntheis. In Forty-first international conference on machine learning, 2024.   
[9] Fbian Falck, TeodoraPandeva, Kiarash Zahiria, Rachel Lawrece, Richard Turr, Edward Meeds, JaviZazo, and Sushrut Karmalkar.A fourier space perspective on diffusion models.arXiv preprint arXiv:2505.11278, 2025.   
[20] Kaifeng Gao, Jiaxin Shi, Hanwang Zhang, Chunping Wang, Jun Xiao, and Long Chen. Ca2-vdm: Efficient autoregressive video diffusion model with causal generation and cache sharing.arXiv preprint arXiv:2411.16375, 2024.   
[21] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025.   
[2] Ian Goodelow, Jean Pouget-Abadie,Mehi Mirza,Bing Xu, David Warde-Farley, Sherjl Ozair Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139144, 2020.   
[23] Google DeepMind. Veo, 2025. URL https://deepmind.google/models/veo.   
[24] Jiatao Gu, Ying Shen, Tianrong Chen, Laurent Dinh, Yuyang Wang, Miguel Angel Bautista, David Berthelot, Josh Susskind, and Shuangfei Zhai. Starflow-v: End-to-end video generative modeling with normalizing flow. arXiv preprint arXiv:2511.20462, 2025.   
[25] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.   
[26] Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhije Lin, Zhenheg Yang, Dahua Lin, and Lu Jiang. Long context tuning for video generation. arXiv preprint arXiv:2503.10589, 2025.   
[27] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shira, Nir Zabari Ori Gordon, et al. Ltx-videoRealtime video latent diffusionarXiv preprint arXiv:2501.00103, 2024.   
[28] Horan He, Yang Zhang, Liang Lin, Zhongwen Xu, and Ling Pan. Pre-trained video generative models as world simulators. arXiv preprint arXiv:2502.07825, 2025.   
[29] Roberto Henschel, Levon Khachatryan, Hayk Poghosyan, Daniil Hayrapetyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Streamingt2v: Consistent, dynamic, and extendable long video generation from text.In Procedings ofthe ComputerVision and PatternRecognition Conference, pages 25682577, 2025.   
[30] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:68406851, 2020.   
[31] Jinyi Hu, Shengding Hu, Yuxuan Song, Yufei Huang, Mingxuan Wang, Hao Zhou, Zhiyuan Liu, Wei-Ying Ma, and Maosong Sun. Acdit: Interpolating autoregressive conditional modeling and diffusion transformer. arXiv preprint arXiv:2412.07720, 2024.   
[32] Xun Huan, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechman. Sef forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.   
[3] Yushi Huang, Xingtong Ge, Ruihao Gong, Chengtao Lv, and Jun Zhang. Linvideo: A post-training framework towards o (n) attention in efficient video generation. arXiv preprint arXiv:2510.08318, 2025.   
[34] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2180721818, 2024.   
[35] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954, 2024.   
[36] Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Gao Huang, and Jiashi Feng. How ar is video generation from world model: A physical law perspective. arXiv preprint arXiv:2411.02385, 2024.   
[37] Weije Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Janwi Zhang, t al. Hunyuanvideo:A systematic framework for large video generative models.arXiv preprint arXiv:2412.03603, 2024.   
[38] Kuaishou. Kling video model. https://kling.kuaishou.com/en, 2024.   
[39] Wuyang Li, WentaoPan, Po-Chien Luan, Yang Gao, andAlexande AlahiStable videoifniy: Ifniteeh video generation with error recycling. arXiv preprint arXiv:2510.09212, 2025.   
[40] Xingyang Li, Muyang Li, Tianle Cai, Haocheng Xi, Shuo Yang, Yujun Lin, Lvmin Zhang, Songln Yang, Jinbo Hu, Kelly Peng, et al. Radial attention: ${ \mathcal { O } } ( n \log n )$ sparse attention with energy decay for long video generation. arXiv preprint arXiv:2506.19852, 2025.   
[1 Zny Li,Shuj Hu,Shuj i, Long Zhou, Jeo hoi, L Meng, Xun uo, Jinyu i, Hefei Lig and Furu Wei. Arlon: Boosting diffusion transformers with autoregressive models for long video generation. arXiv preprint arXiv:2410.20502, 2024.   
[42] Shanchuan Lin, Ceyuan Yang, Hao He, Jianwen Jiang, Yuxi Ren, Xin Xia, Yang Zhao, Xuefeng Xiao, and Lu Jiang. Autoregressive adversarial post-training for real-time interactive video generation. arXiv preprint arXiv:2506.09350, 2025.   
[43] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.   
[44] Hozhe Liu, Shikun Liu, Zijan Zhou, Mengmeng Xu, Yanping Xie, Xiao Han, Juan Pérez, Ding Liu, Kumara Kahatapitiya, Menglin Jia, et al.Mardini: Masked autoregressive diffusion for video generation at scale.arXiv preprint arXiv:2410.20280, 2024.   
[45] Jinlai Liu, Jian Han, Bin Yan, Hui Wu, Fenga Zhu, Xing Wang, Yi Jiang, Bingyue Peng, and Zehuan Yuan. Infinitystar: Unifed spacetime autoregressive modeling for visual generation. arXiv preprint arXiv:2511.04675, 2025.   
[6zheLu, Zheu Jiang, ingn Lu, Yuluu,Tao Jang,Chao Hong, ShaeLi, Wera e Euan, Yuzhi Wang, et al. Moba: Mixture of block attention for long-context llms. arXiv preprint arXiv:2502.13189, 2025.   
[47] Tsvetomila Mihaylova and André FT Martins. Scheduled sampling for transformers. arXiv preprint arXiv:1906.07651, 2019.   
[8 Ma Mi  Jn u,AbrSal,analruuEu e i diffusion models. arXiv preprint arXiv:2308.15321, 2023.   
[49] Wllam Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 41954205, 2023.   
[50] Ryan Po, Yotam Nitzan, Richard Zhang, Berlin Chen, Tri Dao, Eli Shechtman, Gordon Wetzstein, and Xun Huang. Long-context state-space video world models. arXiv preprint arXiv:2505.20171, 2025.   
[51] Sucheng Ren, Chen Chen, Zhenbang Wang, Liangchen Song, Xianin Zhu, Alan Yuile, Yinfei Yang, and Jiasen Lu. Autoregressive video generation beyond next frames prediction. arXiv preprint arXiv:2509.24081, 2025.   
[52] David Ruhe, Jonathan Heek, Tim Salimans, and Emiel Hoogeboom. Rolling diffusion models, 2024. URL https://arxiv.org/abs/2402.09470.   
[53] Flori Schmit.Genelization generatioA clse lookat exposure basarXipreprintarXiv:1910.02, 2019.   
[54] Joonghyuk Shin, Zhengqi Li, Richard Zhang, Jun-Yan Zhu, Jaesik Park, Eli Schechtman, and Xun Huang. Motionstream: Real-time video generation with interactive motion controls. arXiv preprint arXiv:2511.01266, 2025.   
[55] Ivan Skorokhodov, Sergey Tulyakov, and Mohamed EhoseinyStylegan-A continuous videogenerator with the price, image quality and perks of stylegan2. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 36263636, 2022.   
[6] Jascha Sohl-Dickstein, Eric Wess, Niru Mahewaranathan, and Sury Gangul. Deep unsupervised larnig sing nonequilibrium thermodynamics. In International conference on machine learning, pages 22562265. pmlr, 2015.   
[57] Kiwhan Song, Boyuan Chen, Max Simchowitz, Yilun Du, Russ Tedrake, and Vincent Sitzman. History-guided video diffusion. arXiv preprint arXiv:2502.06764, 2025.   
[58] Minghen Sun, Weing Wang, Gen Li, Jiawei Liu, Jiahu Sun, Wanquan Feng, Shanshan Lao, SiYu Zhou, Qian He, and Jing Liu. Ar-diffusion: Asynchronous video generation with auto-regressive diffusion. In Proceedings the Computer Vision and Pattern Recognition Conference, pages 73647373, 2025.   
[59] Wenho Sun, Rong-Cheng Tu,Yifu Ding Zhao Jin, Jingyi Lio, Shuu Liu, and Dach Tao.Vort:Et video diffusion via routing sparse attention. arXiv preprint arXiv:2505.18809, 2025.   
[60] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Minqu Tang, Shuai Han, Tiang Zhag, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025.   
[61] Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, and Jan Kautz. Mocogan: Decomposing motion and content for video generation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 15261535, 2018.   
[62] Dan Valevski, Yaniv Leviathan, Moab Arar, and ShlomiFruchter. Difusn models are real-time game engnes. arXiv preprint arXiv:2408.14837, 2024.   
[63] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan  Gome, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.   
[] Team Wan, Ang Wang,Baole Ai Bin Wen, Chaoje Mao,Chen-Wei Xie, DiChen, Feiw u, Haii Zhao, Jaxio Yang, et al. Wan Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.   
[65] Hongjie Wang, Chih-Yao Ma, Yen-Cheng Liu, Ji Hou, Tao Xu, Jialiang Wang, Felix Juefei-Xu, Yaqio Luo, Peizo Zhang, Tingbo Hou, et al. Lingen: Towards high-resolution minute-length text-to-videogeneration with linear computational complexity. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 25782588, 2025.   
[66] Jing Wang, Fngzhuo Zhang, Xiaoli Li, Vincent YF Tan, Tianyu Pang, Chao Du, Aixin Sun, and Zhurn Yan. Error analyses of auto-regressive video diffusion models: A unified framework. arXiv preprint arXiv:2503.10704, 2025.   
[67] Wenming Weng, Ruoyu Feng, Yanhui Wang, Qi Dai, Chunyu Wang, Dacheng Yin, Zhiyuan Zhao, Kai Qiu, Jianmin Bao, Yuhui Yuan, et al. Art-v: Auto-regressive text-to-video generation with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 73957405, 2024.   
[8 Thadäus Wiedemer, Yuxuan Li, Paul Vicol, Shixian Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, rank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025.   
[69] Jianzog Wu, Liang Hou, Haotian Yang, Xin Tao, Ye Tian, Pengei Wan, Di Zhang, and Yunhai Tong. moba: Mixture-of-block attention for video diffusion models. arXiv preprint arXiv:2506.23858, 2025.   
[70] Haocheng Xi, Shuo Yang, Yilong Zhao, Chenfeng Xu, Muyang Li, Xiuyu Li, Yujun Lin, Han Cai, Jintao Zhang, Dacheng Li, et al. Sparse videogen:Accelerating video diffusion transformers with spatial-temporal sparsity. arXiv preprint arXiv:2502.01776, 2025.   
[71]Yifei Xia, Suhan Ling, Fangcheng Fu, Yuje Wang, Huixia Li, Xuefeng Xiao, and Bin Cui.Traini-free nd adaptive sparse attention for efficient long video generation. arXiv preprint arXiv:2502.21079, 2025.   
[72] Desai Xie, Zhan Xu, Yicong Hong, Hao Tan, Difan Liu, Feng Liu, Arie Kauman, and Yang Zhou. Progresive autoregressive video diffusion models.In Procedingsf theComputerVision and Pattern Recognition Conference, pages 63226332, 2025.   
[73] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcon Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025.   
[74] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaoan Zhang, Guanyu Feng, et al.Cogvideox:Text-to-video diffusion models with an expert transormer.arXiv preprint arXiv:2408.06072, 2024.   
[75] TanweiYin, Micha Gharbi Tesg ark, Richard Zhang, EiShechman, Fred Durand,and BilFreeman. Improved distribution matchingdistillationforfast image synthess.Advancesinneuralinformation procesing systems, 37:4745547487, 2024.   
[76] Tanei Yin, Michal Gharbi, Richard Zhang, Eli Shechtn, Fredo Durand, Wilm TFreman, and Taesng Park. One-step diffsion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 66136623, 2024.   
[77] Tanwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast causal video generators. arXiv e-prints, pages arXiv2412, 2024.   
[78] Jiwen Yu, Yiran Qin, Xintao Wang, Pengei Wan, Di Zhang, and Xihui Liu. Gameacory: Creating new games with generative interactive videos. arXiv preprint arXiv:2501.08325, 2025.   
[79] Hange Yuan, Weiua Chen, Jun Cen, Hu Yu, Jngun Liang, Shu Chang, Zhiui Lin, Tao Feng, Pwei Lu Jiazheng Xing, et al. Lumos-1: On autoregressive video generation from a unified model perspective.arXiv preprint arXiv:2507.08801, 2025.   
[80] Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Yuxing Wei, L WZhip XtivpartteHaralnatiea ar In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2307823097, 2025.   
[81] Jinto Zhang, Chendog Xiang Haofeg Huang, Jia Wei, Hocheng Xi, Jun Zhu, and Jiani Chen. Spargatt: Accurate sparse attention accelerating any model inference. arXiv preprint arXiv:2502.18137, 2025.   
[82] Lvmin Zhang and Maneesh Agrawala. Packing input frame context in next-frame prediction models for video generation. arXiv preprint arXiv:2504.12626, 2(3):5, 2025.   
[83] Peiyn Zhang Yongqi Chen, Haog Huag Wil Lin, Zhengong Liu, Ion Stoica, Eric Xing, and Hao Zhan. Vsa: Faster video diffusion with trainable sparse attention. arXiv preprint arXiv:2505.13389, 2025.   
[84] Tianyuan Zhang, Sai Bi, Yicong Hong, Kai Zhang, Fujun Luan, Songlin Yang, Kalyan Sunkaval, Wiliam T Freeman, and Hao Tan. Test-time training done right. arXiv preprint arXiv:2505.23884, 2025.   
[85] Yan Zhang, Chunli Peng, Boyang Wang, Puyi Wang, Qingcheng Zhu, Fei Kang, Biao Jiang, Zedong Gao, Eric Li, Yang Liu, et al. Matrix-game: Interactive world foundation model. arXiv preprint arXiv:2506.18701, 2025.   
[86] Yuan Zhang, Jacheg Jang, GuoigMa,Zhiying Lu, Haoyag Huang, JianlogYuan, and Nan Duan. Generativ pre-trained autoregressive diffusion transformer. arXiv preprint arXiv:2505.07344, 2025.   
[87] Mingyuan Zhou, Huange Zheng, Yi Gu, Zhendong Wang, and Hai Huang.Adversarial score identiy istiaton: Rapidly surpassing the teacher in one step. arXiv preprint arXiv:2410.14919, 2024.   
[88] Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang.Score identity distilation: Exponentially stdistillation pretrained diffusion model forone-steeneration. InForty-irs Interaial Conference on Machine Learning, 2024.