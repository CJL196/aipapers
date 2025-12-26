# WorldPlay：面向实时交互世界建模的长期几何一致性

文强 孙¹³、海玉 张²³、浩源 王³、军塔 吴³、泽涵 王³、振伟 王³、云鸿 王²、君 张†¹、腾飞 王†³、淳超 郭³ ¹香港科技大学 ²北京航空航天大学 ³腾讯云原

![](images/1.jpg)

# 摘要

本文介绍了WorldPlay，这是一种流媒体视频扩散模型，能够实现实时、互动的世界建模，并保持长期的几何一致性，解决了当前方法在速度和内存之间的权衡。WorldPlay依靠三个关键创新获得强大功能。1) 我们使用双重动作表示，以实现对用户键盘和鼠标输入的稳健动作控制。2) 为了确保长期一致性，我们的再构建上下文记忆根据过去的帧动态重建上下文，并使用时间重构技术使几何重要但已过时的帧保持可访问，有效缓解了内存衰减。3) 我们还提出了上下文强制，这是一种针对内存敏感模型的新型蒸馏方法。对齐教师和学生之间的记忆上下文，保持学生使用长距离信息的能力，实现实时速度，同时防止错误漂移。综合而言，WorldPlay以24帧每秒的速度生成720p长时段流媒体视频，并且在一致性上优于现有技术，在各种场景中表现出强大的泛化能力。项目页面和在线演示可以在以下链接找到：https://3d-models.hunyuan.tencent.com/world/ 和 https://3d.hunyuan.tencent.com/sceneTo3D。

# 1. 引言

世界模型正在推动计算智能的关键转型，从以语言为中心的任务转向视觉和空间推理。通过模拟动态三维环境，这些模型使智能体能够感知和与复杂环境互动，为具身机器人和游戏开发开辟了新可能性。世界建模的前沿是实时交互视频生成，其目标是自回归预测未来视频帧（或片段），以便对每个用户的键盘命令提供即时视觉反馈。尽管取得了重大进展，但一个根本性挑战仍然存在：如何在交互式世界建模中同时实现实时生成（速度）和长期几何一致性（记忆）。有一类方法优先考虑速度，通过蒸馏但忽略了记忆，导致在重访时场景出现不一致。另一类方法通过显式或隐式记忆维护一致性，但复杂的记忆使得蒸馏变得复杂。如表1所总结，低延迟和高一致性同时实现仍然是一个未解决的问题。为了解决这一挑战，我们开发了WorldPlay，这是一种用于一般场景的实时和长期一致的世界模型。我们将这个问题视为一个下一个片段（16帧）预测任务，用于生成基于用户操作的流视频。在自回归扩散模型的基础上，WorldPlay 从以下三个关键要素中汲取力量。第一个是双重动作表示，用于控制智能体和相机的运动。以往的研究通常依赖于离散的键盘输入（例如W、A、S、D）作为动作信号，这种方法虽然能够提供合理的、适应规模的移动，但在需要重访确切位置的记忆检索时会存在歧义。相反，连续的相机位姿 $( R , T )$ 提供了空间位置，但由于训练数据中场景尺度的变化，造成训练不稳定。为了结合两者的优点，我们将动作信号转换为连续的相机位姿和离散的按键，从而实现强健的控制和准确的位置缓存。第二个关键设计是重构上下文记忆，以保持长期几何一致性。我们通过两阶段的过程主动重构记忆，超越简单的检索。首先，通过依据空间和时间接近度查询过去的帧动态重建上下文集。为了解决长距离衰减（在变换器中远离token的影响逐渐减弱），我们提出了时间重构，将这些检索帧的位置嵌入重写。该操作有效地将几何上重要但时间久远的记忆“拉”近时间，使模型把它们视为近期的信息。这个过程保持了相关长程信息的影响，允许强健的自由外推，同时保持强的几何一致性。最后一个关键要素是上下文强制，这是一种针对记忆感知模型的创新蒸馏方法，以实现实时生成。现有的蒸馏方法无法保留长期记忆，因为存在根本的分布不匹配：训练一个记忆感知的自回归学生来模仿一个无记忆的双向教师。即使为教师添加记忆，记忆上下文的不匹配也会导致分布的偏离。我们通过在蒸馏过程中对教师和学生的记忆上下文进行对齐来解决这个问题。这样的对齐促进了有效的分布匹配，使得在不中断记忆的情况下实现实时速度，同时减轻长序列下的误差累积。综上所述，WorldPlay 在用户控制下以24 FPS（720p）的速度实现实时交互视频生成，同时保持长期几何一致性。该模型构建在一个大型、精心策划的320K真实和合成视频的数据集上，并配备定制的渲染和处理平台。如图1所示，WorldPlay 显示出卓越的生成质量和在包括第一人称和第三人称真实和风格化世界的多样场景中显著的泛化能力，并支持从3D重建到可提示事件的各种应用。

# 2. 相关工作

视频生成。扩散模型已成为视频生成建模的最先进方法。采用潜在扩散模型（LDM）在潜在空间中学习视频分布，实现了高效的视频生成。最近，自回归视频生成模型理论上能够生成无限长度的视频，为世界模型奠定基础。随着强大架构和复杂数据流程的进展，经过网络规模数据集训练的模型展现了新兴的零样本能力，以感知、建模和操控视觉世界，从而使模拟物理世界变得可行。

交互式一致性世界模型。世界模型旨在根据当前和过去的观察与行动预测未来状态。研究如 [13, 16, 29, 31, 33, 46, 48, 5961, 64, 75] 采用离散或连续的动作信号，以使智能体能够在虚拟环境中导航和互动。后续旨在实现几何一致性的工作可以分为两类：显式3D重建和隐式条件化。[4, 32, 42, 52, 73, 76, 77] 通过显式重建3D表示并从这些表示呈现条件帧来确保空间一致性。然而，它们在很大程度上依赖于重建质量，使得维持长期一致性变得具有挑战性。其他一些近期工作 [23] 显式构建3D世界模型，而不依赖视频生成模型。尽管实现了有前景的3D生成结果，但这些方法不能在实时使用案例中执行。相反，[67, 74] 通过利用视场 (FOV) 从历史帧中检索相关上下文实现隐式条件化，展现出良好的可扩展性。然而，开发一个维持几何一致性的实时世界模型仍然是一个未解的问题。

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Oasis [9]</td><td rowspan=1 colspan=1>Matrix-Game2.0 [17]</td><td rowspan=1 colspan=1>GameGenX [5]</td><td rowspan=1 colspan=1>GameCraft [31]</td><td rowspan=1 colspan=1>WorldMem [67]</td><td rowspan=1 colspan=1>VMem [32]</td><td rowspan=1 colspan=1>WorldPlay</td></tr><tr><td rowspan=1 colspan=1>Resolution</td><td rowspan=1 colspan=1>360P</td><td rowspan=1 colspan=1>360p</td><td rowspan=1 colspan=1>720p</td><td rowspan=1 colspan=1>720p</td><td rowspan=1 colspan=1>360P</td><td rowspan=1 colspan=1>576p</td><td rowspan=1 colspan=1>720p</td></tr><tr><td rowspan=1 colspan=1>Action Space</td><td rowspan=1 colspan=1>Discrete</td><td rowspan=1 colspan=1>Discrete</td><td rowspan=1 colspan=1>Discrete</td><td rowspan=1 colspan=1>Continuous</td><td rowspan=1 colspan=1>Discrete</td><td rowspan=1 colspan=1>Continuous</td><td rowspan=1 colspan=1>Continuous +Discrete</td></tr><tr><td rowspan=1 colspan=1>Real-time</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>v</td></tr><tr><td rowspan=1 colspan=1>Long-termConsistency</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>v</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Long-Horizon</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>v</td></tr><tr><td rowspan=1 colspan=1>Domain</td><td rowspan=1 colspan=1>Minecraft</td><td rowspan=1 colspan=1>General</td><td rowspan=1 colspan=1>General</td><td rowspan=1 colspan=1>General</td><td rowspan=1 colspan=1>Minecraft</td><td rowspan=1 colspan=1>Static Scene</td><td rowspan=1 colspan=1>General</td></tr></table>

T 蒸馏。实时能力也是世界模型的一个重要特性。对于视频扩散模型，现有的方法通常采用蒸馏技术来实现少步推理，从而实现实时生成。例如，一些研究采用对抗训练策略来实现少步推理，但它们往往会遭遇训练不稳定和模式崩溃的问题。另一部分研究利用变分评分蒸馏（VSD）来实现出色的少步生成性能。此外，CausVid 提出了从双向教师扩散模型蒸馏因果学生模型，以实现实时自回归生成。此外，自我强制方法通过优化 CausVid 的推演策略来解决曝光偏差。我们的方法提出了上下文强制，以在实现实时生成的同时保持互动性和几何一致性。

# 3. 方法

我们的目标是构建一个几何一致且实时互动的世界模型 $N _ { \theta } ( x _ { t } | O _ { t - 1 } , A _ { t - 1 } , a _ { t } , c )$，其参数为 $\theta$，能够根据过去的观察 $O _ { t - 1 } = \{ x _ { t - 1 } , . . . , x _ { 0 } \}$、动作序列 $A _ { t - 1 } = \{ a _ { t - 1 } , . . . , a _ { 0 } \}$ 以及当前动作 $a _ { t }$ 生成下一个片段 $x _ { t }$（一个片段包含几帧）。这里，$c$ 是描述世界的文本提示或图像。为简化符号，后文中我们省略 $A , a , c$。我们首先在第 3.1 节介绍相关的基本概念。在第 3.2 节，我们讨论控制的动作表示。第 3.3 节描述了我们重新构建的上下文记忆，以确保长期几何一致性，紧接着第 3.4 节讨论我们的上下文强制，旨在减少曝光偏差并实现少步生成，同时保持长期一致性。最后，第 3.5 节详细说明了实时流式生成的额外优化。该管道如图 2 所示。

# 3.1. 基础知识

全序列视频扩散模型。当前的视频扩散模型通常由因果3D变分自编码器（VAE）和扩散变换器（DiT）组成，其中每个DiT块由3D自注意力、交叉注意力和前馈网络（FFN）构成。扩散时间步通过位置嵌入（PE）和多层感知机（MLP）进行处理，以调节DiT块。该模型采用流匹配进行训练。具体而言，在给定由3D VAE编码的视频潜变量$z _ { \mathrm { 0 } }$、随机噪声$z _ { 1 } ~ \sim ~ \mathcal { N } ( 0 , I )$以及扩散时间步$k ~ \in ~ [ 0 , 1 ]$的情况下，通过线性插值获得中间潜变量$z _ { k }$。模型的训练目标是预测速度$v _ { k } = z _ { 0 } - z _ { 1 }$。

$$
\mathcal { L } _ { \mathrm { F M } } ( \theta ) = \mathbb { E } _ { k , z _ { 0 } , z _ { 1 } } \bigg \| N _ { \theta } ( z _ { k } , k ) - v _ { k } \bigg \| ^ { 2 } .
$$

逐块自回归生成。然而，完整序列的视频扩散模型是一种非因果架构，这限制了其在无限长度交互生成中的能力。受到扩散强制[6]的启发，我们将其微调为逐块自回归视频生成模型。具体而言，对于视频潜变量 $z _ { 0 } \in \mathbb { R } ^ { C \times T \times H \times W }$，我们将其划分为 $\textstyle { \frac { T } { 4 } }$ 个块 $\{ z _ { 0 } ^ { i } \in \mathbb { R } ^ { C \times 4 \times H \times W } | i = 0 , . . . , \frac { T } { 4 } - 1 \}$，因此每个块（4个潜变量）可以解码为16帧。在训练期间，我们为每个块添加不同的噪声水平 $k _ { i }$，并将完整序列自注意力修改为块因果注意力。训练损失与等式1类似。

![](images/2.jpg)  
context memory from past chunks to enforce long-term temporal and geomeric consistency.

![](images/3.jpg)  
Figure 3. Detailed architecture of our autoregressive diffusion transformer. The discrete key is incorporated with time embedding, while the continuous camera pose is injected into causal selfattention through PRoPE [33].

# 3.2. 控制的双重动作表示

现有方法使用键盘和鼠标输入作为动作信号，并通过多层感知机（MLP）或注意力模块注入动作控制。这使得模型能够在具有不同尺度（例如，极大和极小场景）的场景中学习物理上合理的移动。然而，它们在提供准确的先前位置信息以便进行空间记忆检索方面存在困难。相比之下，相机姿态（旋转矩阵和位移向量）提供精确的空间位置，便于进行精确控制和记忆检索，但仅使用相机姿态进行训练面临训练稳定性挑战，因为训练数据中的尺度变化。为此，我们提出了一种双重动作表示，结合了两个领域的优点，如图 3 所示。该设计不仅为我们在第 3.3 节的记忆模块缓存空间位置，而且还实现了稳健和精确的控制。具体而言，我们采用位置编码（PE）和一个零初始化的 MLP 来编码离散键，并将其纳入时间步嵌入中，随后用于调制 DiT 模块。对于连续相机姿态，我们采用相对位置编码，即 PRoPE，这比常用的射线图具有更好的普适性，从而将完整的相机视锥注入自注意力模块。原始自注意力计算如下，其中 $R$ 代表视频潜变量的 3D 旋转位置编码（RoPE）。为了编码相机之间的视锥关系，我们利用了额外的注意力计算，这里 $D ^ { p r o j }$ 来源于相机的内外参数，如 [33] 所述。最后，每个自注意力模块的结果为 $A t t n _ { 1 } + z e r o \_ i n i t ( A t t n _ { 2 } )$ 。

$$
A t t n _ { 1 } = A t t n ( R ^ { \top } \odot Q , R ^ { - 1 } \odot K , V ) ,
$$

$$
\begin{array} { c } { A t t n _ { 2 } = D ^ { p r o j } \odot A t t n ( ( D ^ { p r o j } ) ^ { \top } \odot Q , } \\ { ( D ^ { p r o j } ) ^ { - 1 } \odot K , ( D ^ { p r o j } ) ^ { - 1 } \odot V ) , } \end{array}
$$

# 3.3. 重新构建上下文记忆以确保一致性

保持长期几何一致性需要回忆过去的帧，确保在返回以前位置时内容保持不变。然而，简单地将所有过去的帧作为上下文（图 4a）对于长序列来说在计算上是不可行且冗余的。为了解决这一问题，我们为每个新块 $x _ { t }$ 从过去的片段 $O _ { t - 1 }$ 重建记忆上下文 $C _ { t }$。我们的方法超越了以前的工作[67, 74]，结合了短期时间线索和长范围空间参照：1）时间记忆 $( C _ { t } ^ { T } )$ 包含 $L$ 个最近的块 $\{ x _ { t - L } , . . . , x _ { t - 1 } \}$ 以确保短期运动平滑性。2）空间记忆 $( C _ { t } ^ { S } )$ 从不相邻的过去帧中采样，以防止长序列中的几何漂移，其中 $C _ { t } ^ { S } \subseteq O _ { t - 1 } - C _ { t } ^ { T }$。该采样由几何相关性评分引导，考虑了视场重叠和相机距离。

![](images/4.jpg)  
Figure 4. Memory mechanism comparisons. The red and blue blocks represent the memory and current chunk, respectively. The number in each block represents the temporal index in RoPE. For simplicity of illustration, each chunk only contains one frame.

一旦重建了记忆上下文，挑战就转向如何应用这些上下文以确保一致性。有效利用检索到的上下文需要克服位置编码中的一个基本缺陷。使用标准的相对位置编码（RoPE）时（图 4b），当前片段与过去记忆之间的距离会随着时间的推移而不断增长。这种不断增长的相对距离最终可能会超过 RoPE 中训练的插值范围，从而导致外推伪影 [58]。更重要的是，感知到的与这些远古空间记忆的距离的增长会削弱它们对当前预测的影响。为了解决这个问题，我们提出了时间重塑（Temporal Reframing，图 4c）。我们丢弃绝对时间索引，并动态重新分配新的位置编码给所有上下文帧，为当前帧建立一个固定的小相对距离，而不考虑它们之间的实际时间间隔。这个操作有效地“拉近”了重要的过去帧，确保它们保持影响力，并为长期一致性提供稳健的外推能力。

# 3.4. 上下文强制

自回归模型在长视频生成过程中常常会遭遇误差累积，从而导致视觉质量随时间下降。此外，扩散模型的多步骤去噪对于实时交互而言速度过慢。近期的方法通过将强大的双向教师扩散模型提炼成快速、少步骤的自回归学生，来应对这些挑战。这些技术强制学生的输出分布 $p _ { \theta } ( x _ { 0 : t } )$ 与教师的输出分布对齐，从而通过采用分布匹配损失来提高生成质量。

![](images/5.jpg)  
Figure 5. Context forcing is a novel distillation method that employs memory-augmented self-rollout and memory-augmented bidirectional video diffusion to preserve long-term consistency, enable real-time interaction, and mitigate error accumulation.

$$
\nabla _ { \theta } \mathcal { L } _ { D M D } = \mathbb { E } _ { k } \big ( \nabla _ { \theta } \mathrm { K L } \big ( p _ { \theta } ( x _ { 0 : t } ) \big | \big | p _ { d a t a } ( x _ { 0 : t } ) \big ) \big ) ,
$$

反向 $\mathrm{KL}$ 的梯度可以通过从教师模型导出的得分差异进行近似。然而，这些方法与具有记忆意识的模型不兼容，原因是一种关键的分布不匹配。标准的教师扩散模型是在短片段上训练的，并且本质上是无记忆的。即使教师模型被增强了记忆，其双向特性也不可避免地与学生的因果自回归过程不同。这意味着如果没有精心设计的记忆上下文来缓解这一差距，记忆上下文的差异将导致其条件分布 $p ( x | C )$ 对齐失败，从而导致分布匹配失败。因此，我们提出了如图 5 所示的上下文强制，它缓解了教师与学生之间的记忆上下文不对齐问题。在学生模型中，我们基于记忆上下文自我推演 4 个片段，条件为 $p _ { \theta } ( x _ { j : j + 3 } | x _ { 0 : j - 1 } ) = \prod _ { i = j } ^ { j + 3 } p _ { \theta } ( x _ { i } | C _ { i } )$。为了构建我们的教师模型 $V _ { \beta }$，我们在标准双向扩散模型中增强记忆，并通过在学生的记忆上下文中屏蔽 $x _ { j : j + 3 }$ 来构造其上下文，其中 $C _ { j : j + 3 }$ 表示与学生自我推演 $x _ { j : j + 3 }$ 对应的所有上下文记忆块。通过将记忆上下文与学生模型对齐，我们强制教师所表示的分布尽可能接近学生模型，从而更有效地实现分布匹配。此外，这避免了在长视频和冗余上下文上训练 $V _ { \beta }$，促进了长期视觉分布的学习。通过上下文强制，我们在实时生成中保持长期一致性，进行 4 次去噪步骤，并减轻错误累积。

$$
p _ { d a t a } ( x _ { j : j + 3 } | x _ { 0 : j - 1 } ) = p _ { \beta } ( x _ { j : j + 3 } | C _ { j : j + 3 } - x _ { j : j + 3 } ) ,
$$

![](images/6.jpg)  
F boxes) and visual qualityacross diverse scenes, icludingboth frstand thir-personreal and stylizedworlds.

# 3.5. 实时延迟下的流式生成

我们通过一系列优化来增强上下文强制，以最小化延迟，在 ${8 \times} \mathrm{H} 800$ GPU 上实现每秒 $24 \mathrm{F} P S$ 和 $720 \mathrm{p}$ 分辨率的交互式流媒体体验。针对 DiT 和 VAE 的混合并行方法。与复制整个模型或在时间维度上调整序列并行的方法不同，我们的并行方法结合了序列并行和注意力并行，将每个整体块的词元在设备之间进行分区。该设计确保生成每个块的计算工作负载均匀分配，大幅降低每个块的推理时间，同时保持生成质量。流式部署和渐进解码。为了最小化首次帧到达时间并实现无缝交互，我们采用基于 NVIDIA Triton 推理框架的流式部署架构，并实施渐进式多步 VAE 解码策略，该策略将帧解码并流式处理为较小批次。在从 DiT 生成潜在表示后，帧会逐步解码，使用户能够在后续帧仍在处理时观察生成的内容。该流式管道确保在不同计算负载下实现平滑、低延迟的交互。量化与高效注意力。此外，我们采用了一整套全面的量化策略。具体而言，我们采用 Sage Attention、浮点量化和矩阵乘法量化来提升推理性能。此外，我们为注意力模块使用 KV-cache 机制，以消除自回归生成过程中的冗余计算。

<table><tr><td></td><td colspan="6">Short-term (61 frames)</td><td colspan="4">Long-term (≥ 250 frames)</td><td></td></tr><tr><td></td><td>Real-time</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>Rdist ↓</td><td>Tdist↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>Rdist ↓</td><td>Tdist ↓</td></tr><tr><td>CameraCtrl [16]</td><td>X</td><td>17.93</td><td>0.569</td><td>0.298</td><td>0.037</td><td>0.341</td><td>10.09</td><td>0.241</td><td>0.549</td><td>0.733</td><td>1.117</td></tr><tr><td>SEVA [80]</td><td></td><td>19.84</td><td>0.598</td><td>0.313</td><td>0.047</td><td>0.223</td><td>10.51</td><td>0.301</td><td>0.517</td><td>0.721</td><td>1.893</td></tr><tr><td>ViewCrafter [77]</td><td>× ×</td><td>19.91</td><td>0.617</td><td>0.327</td><td>0.029</td><td>0.543</td><td>9.32</td><td>0.277</td><td>0.661</td><td>1.573</td><td>3.051</td></tr><tr><td>Gen3C [52]</td><td>X</td><td>21.68</td><td>0.635</td><td>0.278</td><td>0.024</td><td>0.477</td><td>15.37</td><td>0.431</td><td>0.483</td><td>0.357</td><td>0.979</td></tr><tr><td>VMem [64]</td><td>X</td><td>19.97</td><td>0.587</td><td>0.316</td><td>0.048</td><td>0219</td><td>12.77</td><td>0.335</td><td>0.542</td><td>0.748</td><td>1.547</td></tr><tr><td>Matrix-Game-2.0 [17]</td><td>v</td><td>17.26</td><td>0.505</td><td>0.383</td><td>0.287</td><td>0.843</td><td>9.57</td><td>0.205</td><td>0.631</td><td>2.125</td><td>2.742</td></tr><tr><td>GameCraft [31]</td><td>X</td><td>21.05</td><td>0.639</td><td>0.341</td><td>0.151</td><td>0.617</td><td>10.09</td><td>0.287</td><td>0.614</td><td>2.497</td><td>3.291</td></tr><tr><td>Ours (w/o Context Forcing)</td><td>X</td><td>21.27</td><td>0.669</td><td>0.261</td><td>0.033</td><td>0.157</td><td>16.27</td><td>0.425</td><td>0.495</td><td>0.611</td><td>0.991</td></tr><tr><td>Ours (full)</td><td>v</td><td>21.92</td><td>0.702</td><td>0.247</td><td>0.031</td><td>0.121</td><td>18.94</td><td>0.585</td><td>0.371</td><td>0.332</td><td>0.797</td></tr></table>

抱歉，我无法处理您提供的内容。请提供清晰的英文文本以便翻译。

# 4. 实验

数据集。WorldPlay在一个全面的数据集上进行训练，该数据集包含大约32万个高质量视频样本，这些样本源自真实世界的录像和合成环境。对于真实世界视频，我们从公开可用的真实视频资源开始[36，40]，并去除短小、低质量的片段，以及包含水印、用户界面、密集人群或不稳定相机运动的样本。为了减轻原始视频中常见的单调运动，我们采用3D高斯散射[25]对经过筛选的视频进行3D重建。然后，我们从这些3D场景中使用新的重访轨迹渲染定制视频。这些渲染结果通过Difix3D $^ +$ [66]进一步优化，以修复漂浮伪影，生成额外的10万个高质量真实视频片段。对于合成数据，我们收集了数百个虚幻引擎场景，并通过渲染复杂的定制轨迹生成5万个视频片段。此外，我们建立了一个游戏录制平台，邀请数十名玩家从设计轨迹的第一人称/第三人称AAA游戏中收集170K样本。我们将每个视频分割成片段，并使用视觉-语言模型[81]生成文本注释。对于没有动作注释的视频，我们使用VIPE[20]进行标记。评估协议。我们的测试集包含从DL3DV、游戏视频和AI生成图像中提取的600个案例，涵盖多种风格。在短期设置中，我们利用测试视频中的相机轨迹作为输入姿态。生成的视频帧直接与真实标注数据（GT）帧进行比较，以评估视觉质量和相机姿态准确性。在长期设置中，我们使用设计用以强制重访的各种自定义循环相机轨迹测试长期一致性。每个模型沿着自定义轨迹生成帧，然后沿相同路径返回，通过将生成的帧与初始传递过程中生成的相应帧进行比较，来评估返回路径上的指标。我们使用LPIPS、PSNR和SSIM来测量视觉质量，$R _ { \mathrm { d i s t } }$和$T _ { \mathrm { d i s t } }$来量化动作准确性。基准测试。我们与多种基准进行了全面比较，这些基准主要分为两类：1) 没有记忆的动作控制扩散模型：CameraCtrl [16]，SEVA [80]，ViewCrafter [77]，Matrix-Game 2.0 [17]和GameCraft [31]；2) 有记忆的动作控制扩散模型：Gen3C [52]和VMem [32]。更多评估结果可以在我们的附录中找到。

# 4.1. 主要结果

定量结果。如表2所示，在短期情况下，我们的方法实现了更优的视觉保真度，并保持了竞争力的控制精度。虽然利用显式三维表示的方法（如ViewCrafter [77]、Gen3C [52]）能够实现更准确的旋转，但在进行移动转换时，它们遭遇了不准确的深度估计和不一致的比例等问题。对于更具挑战性的长期场景，动作精度通常降低，我们的方法仍然保持更稳定，并取得最佳表现。在长期几何一致性方面，Matrix-Game2.0 [17]和GameCraft [31]由于缺乏记忆机制表现不佳。虽然VMem [32]和Gen3C [52]采用显式三维缓存来维持一致性，但由于受到深度准确性和对齐的限制，难以实现稳健的长期一致性。得益于重构上下文记忆，我们实现了长期一致性的提升。此外，通过上下文强制，我们进一步防止了误差累积，从而提高了视觉质量和动作精度。至关重要的是，WorldPlay同时实现了沉浸式仿真的实时交互所需的性能。定性结果。我们在图6中提供了与基线的定性比较。Gen3C [52]中使用的显式三维缓存对中间输出质量非常敏感，并受到深度估计准确性的限制。相反，我们的重构上下文记忆保证了更强健的隐式先验的长期一致性，实现了更优的场景泛化能力。Matrix-Game2.0 [17]和GameCraft [31]因缺乏记忆而无法支持自由探索。此外，它们在第三人称场景中无法很好地泛化，难以控制场景中的智能体，限制了它们的适用性。相比之下，WorldPlay成功地将其效能扩展到这些场景，并保持了高视觉保真度和长期几何一致性。

Table 3. Ablation for action representation. We conduct validation using the bidirectional model.   

<table><tr><td>Action</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>Rdist ↓</td><td>Tdist </td></tr><tr><td>Discrete</td><td>21.47</td><td>0.661</td><td>0.248</td><td>0.103</td><td>0.615</td></tr><tr><td>Continuous</td><td>21.93</td><td>0.665</td><td>0.231</td><td>0.038</td><td>0.287</td></tr><tr><td>Full</td><td>22.09</td><td>0.687</td><td>0.219</td><td>0.028</td><td>0.113</td></tr></table>

![](images/7.jpg)  
Figure 7. RoPE design comparisons. Upper: Our reframed RoPE avoids exceeding the the positional range in standard RoPE, alleviating error accumulation. Bottom: By maintaining a small relative distance to long-range spatial memory, it achieves better long-term consistency.

# 4.2. 消融实验

动作表征。表3验证了所提双重动作表征的有效性。当仅使用离散键作为动作信号时，模型难以实现细粒度控制，例如移动距离或旋转角度，导致在 $R _ { \mathrm { d i s t } }$ 和 $T _ { \mathrm { d i s t } }$ 指标上的表现不佳。使用连续相机姿态可以获得更好的结果，但由于尺度变化而更难收敛。通过采用双重动作表征，我们实现了最佳的整体控制性能。

RoPE设计。表4展示了在记忆机制中不同RoPE设计的定量结果，显示了重新构建的rope优于简单的对应物，尤其是在视觉指标上。如图7的上半部分所示，RoPE更容易出现误差累积。这也由于绝对时间索引增加了记忆和预测片段之间的距离，导致几何一致性变弱，如图7的下半部分所示。上下文强制。为了验证记忆对齐的重要性，我们按照[74]训练教师模型，其中记忆是在潜在层级而非片段层级选择的。虽然这可能减少教师模型中的记忆上下文数量，但也引入了教师与学生模型之间的上下文不对齐，导致如图8a所示的崩溃结果。此外，对于过去的片段$x_{0:j-1}$，我们尝试遵循推理时间的流程，自我推演历史片段作为上下文，参考[68]中的方法。然而，这可能导致双向扩散模型提供不准确的评分估计，因为它是在使用干净片段作为记忆的情况下训练的。因此，这种不一致导致了如图8b所示的伪影。我们通过从真实视频中采样获得历史片段，生成的结果优于如图8c所示的结果。

Table 4. Ablation for positional encoding design in memory. The results are evaluated on the long-term test data.   

<table><tr><td></td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>Rdist ↓</td><td>Tdist </td></tr><tr><td>RoPE</td><td>14.03</td><td>0.358</td><td>0.534</td><td>0.805</td><td>1.341</td></tr><tr><td>Reframed RoPE</td><td>16.27</td><td>0.425</td><td>0.495</td><td>0.611</td><td>0.991</td></tr></table>

![](images/8.jpg)  
Figure 8. Ablation for context forcing. a) When the teacher and student have misaligned context, it leads to distillation failure, resulting in collapsed outputs. b) Self-rollout historical context can introduce artifacts. Zoom in for details.

![](images/9.jpg)  
Figure 9. Promptable event. Our method supports text-based manipulation during streaming.

# 4.3. 应用

三维重建。得益于长期的几何一致性，我们可以整合一个三维重建模型 [44] 以生成高质量的点云，如图 1 (d) 中所示。可提示事件。除了导航控制，WorldPlay 还支持基于文本的交互来触发动态世界事件。如图 9 和图 1 (e) 所示，用户可以随时提示以响应性地改变正在进行的流。

# 5. 结论

WorldPlay 是一个强大的世界模型，具有实时互动和长期几何一致性。它使用户能够从单张图像或文本提示中自定义独特的世界。尽管专注于导航控制，其架构显示出可支持更丰富互动潜力的迹象，例如动态、基于文本触发的事件。通过提供一个系统化的控制、记忆和提炼框架，WorldPlay 标志着创建一致且互动的虚拟世界向前迈出了重要一步。将其扩展到生成具有多智能体互动和复杂物理动态的更长视频将是富有前景的未来方向。

# References

[1] Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos J Storkey, Tim Pearce, and François Fleuret. Diffusion for world modeling: Visual details matter in atari. Advances in Neural Information Processing Systems, 37: 5875758791, 2024. 2   
[2] Sherwin Bahmani, Ivan Skorokhodov, Guocheng Qian, Aliaksandr Siarohin, Willi Menapace, Andrea Tagliasacchi, David B Lindell, and Sergey Tulyakov. Ac3d: Analyzing and improving 3d camera control in video diffusion transformers. In CVPR, pages 2287522889, 2025.   
[3] Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, and Yann LeCun. Navigation world models. In CVPR, pages 1579115801, 2025. 2   
[4] Chenjie Cao, Jingkai Zhou, Shikai Li, Jingyun Liang, Chaohui Yu, Fan Wang, Xiangyang Xue, and Yanwei Fu. Uni3c: Unifying precisely 3d-enhanced camera and human motion controls for video generation. arXiv preprint arXiv:2504.14899, 2025. 2   
[5] Haoxuan Che, Xuanhua He, Quande Liu, Cheng Jin, and Hao Chen. Gamegen-x: Interactive open-world game video generation. arXiv preprint arXiv:2411.00769, 2024. 3   
[6] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:2408124125, 2024. 2, 3   
[7] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In CVPR, pages 73107320, 2024. 2   
[8] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Selfforcing $^ { + + }$ Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283, 2025. 5   
[9] Etched Decart. Oasis: A universe in a transformer. https : //oasis-model.github.io/,2024.2,3,4   
10] Google Deepmind. Veo3 video model, 2025. ht tps : / / deepmind.google/models/veo/.2   
11] Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, and Jiajun Wu. Worldscore: A unified evaluation benchmark for world generation. arXiv preprint arXiv:2504.00983, 2025. 6   
12] Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step diffusion via shortcut models. arXiv preprint arXiv:2410.12557, 2024. 3   
[15] ru Gao, Haoyuan Guo, 1uyen Hoang, wellin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025.2   
[14] Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. arXiv preprint arXiv:2505.13447, 2025. 3   
[15] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-toimage diffusion models without specific tuning. In ICLR, 2024. 2   
[16] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. In ICLR, 2025. 2, 7   
[17] Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, et al. Matrix-game 2.0: An open-source, real-time, and streaming interactive world model. arXiv preprint arXiv:2508.13009, 2025. 2, 3, 4, 7   
[18] Roberto Henschel, Levon Khachatryan, Hayk Poghosyan, Daniil Hayrapetyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Streamingt2v: Consistent, dynamic, and extendable long video generation from text. In CVPR, pages 25682577, 2025. 2   
[19] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:68406851, 2020. 2   
[20] Jiahui Huang, Qunjie Zhou, Hesam Rabeti, Aleksandr Korovko, Huan Ling, Xuanchi Ren, Tianchang Shen, Jun Gao, Dmitry Slepichev, Chen-Hsuan Lin, et al. Vipe: Video pose engine for 3d geometric perception. arXiv preprint arXiv:2508.10934, 2025. 7, 2   
[21] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the traintest gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025. 2, 3, 5, 1   
[22] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2180721818, 2024. 5, 6   
[23] Team HunyuanWorld. Hunyuanworld 1.0: Generating immersive, explorable, and interactive 3d worlds from words or pixels. arXiv preprint, 2025. 3   
[24] Minguk Kang, Richard Zhang, Connelly Barnes, Sylvain Paris, Suha Kwak, Jaesik Park, Eli Shechtman, Jun-Yan Zhu, and Taesung Park. Distilling diffusion models into conditional gans. In ECCV, pages 428447. Springer, 2024. 3   
[25] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):1391, 2023. 7   
[26] Jihwan Kim, Junoh Kang, Jinyoung Choi, and Bohyung Han. Fifo-diffusion: Generating infinite videos from text without training. Advances in Neural Information Processing Systems, 37:8983489868, 2024. 2   
[27] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 3   
[28] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 2, 3, 1   
[29] Xin Kong, Shikun Liu, Xiaoyang Lyu, Marwan Taher, Xiaojuan Qi, and Andrew J Davison. Eschernet: A generative model for scalable view synthesis. In CVPR, pages 9503 9513, 2024. 2   
[30] Kuaishou. Kling video model, 2024. https:// klingai.com/global/.2   
[31] Jiaqi Li, Junshu Tang, Zhiyong Xu, Longhuang Wu, Yuan Zhou, Shuai Shao, Tianbao Yu, Zhiguo Cao, and Qinglin Lu. Hunyuan-gamecraft: High-dynamic interactive game video generation with hybrid history condition. arXiv preprint arXiv:2506.17201, 2025. 2, 3, 7   
[32] Runjia Li, Philip Torr, Andrea Vedaldi, and Tomas Jakab. Vmem: Consistent interactive video scene generation with surfel-indexed view memory. In ICCV, 2025. 2, 3, 7   
[33] Ruilong Li, Brent Yi, Junchen Liu, Hang Gao, Yi Ma, and Angjoo Kanazawa. Cameras as relative positional encoding. arXiv preprint arXiv:2507.10496, 2025. 2, 4   
[34] Shenggui Li, Fuzhao Xue, Chaitanya Baranwal, Yongbin Li, and Yang You. Sequence parallelism: Long sequence training from system perspective. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 23912404, Toronto, Canada, 2023. Association for Computational Linguistics. 6   
[35] Xinyang Li, Tengfei Wang, Zixiao Gu, Shengchuan Zhang, Chunchao Guo, and Liujuan Cao. Flashworld: Highquality 3d scene generation within seconds. arXiv preprint arXiv:2510.13678, 2025. 3   
[36] Zhen Li, Chuanhao Li, Xiaofeng Mao, Shaoheng Lin, Ming Li, Shitian Zhao, Zhaopan Xu, Xinyue Li, Yukang Feng, Jianwen Sun, et al. Sekai: A video dataset towards world exploration. arXiv preprint arXiv:2506.15675, 2025. 7, 2   
[37] Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxllightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929, 2024. 3   
[38] Shanchuan Lin, Xin Xia, Yuxi Ren, Ceyuan Yang, Xuefeng Xiao, and Lu Jiang. Diffusion adversarial post-training for one-step video generation. 2025.   
[39] Shanchuan Lin, Ceyuan Yang, Hao He, Jianwen Jiang, Yuxi Ren, Xin Xia, Yang Zhao, Xuefeng Xiao, and Lu Jiang. Autoregressive adversarial post-training for real-time interactive video generation. arXiv preprint arXiv:2506.09350, 2025. 3   
[40] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. D13dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In CVPR, pages 2216022169, 2024. 7, 2   
[41] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In ICLR, 2023. 2, 3   
[42] Fangfu Liu, Wenqiang Sun, Hanyang Wang, Yikai Wang, Haowen Sun, Junliang Ye, Jun Zhang, and Yueqi Duan. Reconx: Reconstruct any scene from sparse views with video diffusion model. arXiv preprint arXiv:2408.16767, 2024. 2   
[43] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161, 2025. 5   
[44] Yifan Liu, Zhiyuan Min, Zhenwei Wang, Junta Wu, Tengfei Wang, Yixuan Yuan, Yawei Luo, and Chunchao Guo. Worldmirror: Universal 3d world reconstruction with any-prior prompting. arXiv preprint arXiv:2510.10726, 2025. 8   
[45] Yanzuo Lu, Yuxi Ren, Xin Xia, Shanchuan Lin, Xing Wang, Xuefeng Xiao, Andy J Ma, Xiaohua Xie, and Jian-Huang Lai. Adversarial distribution matching for diffusion distillation towards efficient image and video synthesis. In ICCV, pages 1681816829, 2025. 3   
[46] Xiaofeng Mao, Shaoheng Lin, Zhen Li, Chuanhao Li, Wenshuo Peng, Tong He, Jiangmiao Pang, Mingmin Chi, Yu Qiao, and Kaipeng Zhang. Yume: An interactive world generation model. arXiv preprint arXiv:2507.17744, 2025. 2   
[47] Minimax. Hailuo video model, 2024. https : / / hailuoai.video.2   
[48] Takeru Miyato, Bernhard Jaeger, Max Welling, and Andreas Geiger. Gta: A geometry-aware attention mechanism for multi-view transformers. In ICLR, 2024. 2   
[49] Jack Parker-Holder, Philip Ball, Jake Bruce, Vibhavari Dasagi, Kristian Holsheimer, Christos Kaplanis, Alexandre Moufarek, Guy Scully, Jeremy Shar, Jimmy Shi, Stephen Spencer, Jessica Yung, Michael Dennis, Sultan Kenjeyev, Shangbang Long, Vlad Mnih, Harris Chan, Maxime Gazeau, Bonnie Li, Fabio Pardo, Luyu Wang, Lei Zhang, Frederic Besse, Tim Harley, Anna Mitenkova, Jane Wang, Jeff Clune, Demis Hassabis, Raia Hadsell, Adrian Bolton, Satinder Singh, and Tim Rocktäschel. Genie 2: A large-scale foundation world model. 2024. 2   
[50] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, pages 41954205, 2023. 2, 3   
[51] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 2   
[52] Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Müller, Alexander Keller, Sanja Fidler, and Jun Gao. Gen3c: 3d-informed world-consistent video generation with precise camera control. In CVPR, pages 61216132, 2025. 2, 7   
[53] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10684 10695, 2022. 2   
[54] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022. 3   
[55] Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast highresolution image synthesis with latent adversarial diffusion distillation. In SIGGRAPH Asia, pages 111, 2024. 3   
[56] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In ECCV, pages 87103. Springer, 2024. 3   
[57] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021. 2   
[58] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. 2, 4, 5   
[59] Wenqiang Sun, Shuo Chen, Fangfu Liu, Zilong Chen, Yueqi Duan, Jun Zhang, and Yikai Wang. Dimensionx: Create any 3d and 4d scenes from a single image with controllable video diffusion. arXiv preprint arXiv:2411.04928, 2024. 2   
[60] Wenqiang Sun, Fangyun Wei, Jinjing Zhao, Xi Chen, Zilong Chen, Hongyang Zhang, Jun Zhang, and Yan Lu. From virtual games to real-world play. arXiv preprint arXiv:2506.18901, 2025.   
[61] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines. In ICLR, 2025. 2   
[62] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 3, 1   
[63] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in neural information processing systems, 36: 84068441, 2023. 3   
[64] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH, pages 111, 2024. 2,7   
[65] Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025. 2   
[66] Jay Zhangjie Wu, Yuxuan Zhang, Haithem Turki, Xuanchi Ren, Jun Gao, Mike Zheng Shou, Sanja Fidler, Zan Gojcic, and Huan Ling. Difix3d+: Improving 3d reconstructions with single-step diffusion models. In CVPR, pages 26024 26035, 2025. 7, 2   
[67] Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. Worldmem: Longterm consistent world simulation with memory. arXiv preprint arXiv:2504.12369, 2025. 2, 3, 4, 5   
[68] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025. 5, 8, 6   
[69] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In ICLR, 2024. 2   
[70] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and Bill Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:4745547487, 2024. 2, 3, 5   
[71] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In CVPR, pages 66136623, 2024. 3   
[72] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In CVPR, pages 2296322974, 2025. 3, 5   
[73] Hong-Xing Yu, Haoyi Duan, Charles Herrmann, William T Freeman, and Jiajun Wu. Wonderworld: Interactive 3d scene generation from a single image. In CVPR, pages 59165926, 2025.2   
[74] Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Context as memory: Scene-consistent interactive long video generation with memory retrieval. arXiv preprint arXiv:2506.03141, 2025. 2, 3, 4, 5, 8   
[75] Jiwen Yu, Yiran Qin, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Gamefactory: Creating new games with generative interactive videos. In ICCV, 2025. 2   
[76] Mark YU, Wenbo Hu, Jinbo Xing, and Ying Shan. Trajectorycrafter: Redirecting camera trajectory for monocular videos via diffusion models. In ICCV, 2025. 2   
[77] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048, 2024. 2, 7   
[78] Haiyu Zhang, Xinyuan Chen, Yaohui Wang, Xihui Liu, Yunhong Wang, and Yu Qiao. Accvideo: Accelerating video diffusion model with synthetic dataset. arXiv preprint arXiv:2503.19462, 2025. 3   
[79] Jintao Zhang, Jia Wei, Pengle Zhang, Jun Zhu, and Jianfei Chen. Sageattention: Accurate 8-bit attention for plug-andplay inference acceleration. In ICLR, 2025. 6   
[80] Jensen Zhou, Hang Gao, Vikram Voleti, Aaryaman Vasishta, Chun-Han Yao, Mark Boss, Philip Torr, Christian Rupprecht, and Varun Jampani. Stable virtual camera: Generative view synthesis with diffusion models. arXiv preprint arXiv:2503.14489, 2025. 7   
[81] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internv13: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 7

# WorldPlay: Towards Long-Term Geometric Consistency for Real-Time Interactive World Modeling

Supplementary Material

# A. Training and Inference Details

We adopt the pretrained DiT-based video diffusion models [28, 62] as the backbone. For the chunk-wise autoregressive diffusion transformer, we group 4 latents into a chunk. For the memory context, we set the temporal memory length to 3 chunks and the spatial memory length to 1 chunk. For the bidirectional teacher model $V _ { \beta }$ , we also adopt the dual-action representation and construct the memory context as described in Sec.3.4. The training consists of three stages.

Stage One: Action Control. In the first stage, we focus on injecting action control into the pretrained model. We employ the dual action representation to the pretrained model and train the bidirectional action model for 30K iterations. Then, we replace the 3D self-attention with block causal attention and train for an additional 30K iterations as our AR action model. We find that this enables the AR action model to converge more easily. In this stage, the model is trained on 61 frames (4 chunks) using the Adam optimizer with a learning rate of $1 e - 5$ and a batch size of 64.

Stage Two: Memory. In the second stage, we train the bidirectional action model and the AR action model with context memory as described in Sec.3.3 and Sec.3.4, respectively. The training is performed on longer videos, while other settings remain the same as in the first stage.

Stage Three: Context Forcing. In the final stage, we use the bidirectional model as the teacher and the AR model as the student for distillation. To stabilize the distillation process, we employ a progressive training strategy that gradually increases the maximum length of the generated latents. For the student model, the learning rate is set to $1 e - 6$ , while for the bidirectional model, which is used to compute the fake score, the learning rate is set to $2 e - 7$ The models are trained for 2K iteration with a batch size of 64. All other hyperparameters follow [21]. For the details of context forcing, see Algorithm 1.

Finally, our AR model can produce multiple chunks in a streaming fashion with KV cache as shown in Algorithm 2. When the user provides only camera poses, we first compute the relative translations and rotations between consecutive poses, and then apply a thresholding mechanism to identify and convert them into discrete actions. Conversely, when only discrete actions are available, we use the predefined relative translations and rotations associated with each action to convert them into camera poses.

# Algorithm 1 Context Forcing Training

Require: Number of denoising timesteps $d$ and chunks $n = 4$   
Require: Dataset $D$ (encoded by 3D VAE)   
Require: AR diffusion model $N _ { \theta }$   
Require: Bidirectional diffusion model $V _ { \beta } ^ { f a k e }$ and Vreal 1: loop 2: Progressively increase maximum chunk length $m$ 3: Sample chunk length $j \sim \operatorname { U n i f o r m } ( 0 , 1 , \dots , m )$ 4: Sample context $x _ { 0 : j - 1 } \sim D$ 5: for $i = j , \ldots , j + n - 1$ do 6: Initialize $x _ { i } ^ { i n i t } \sim \mathcal { N } ( 0 , I )$ 7: Reconstitute context memory $C _ { i } \subseteq \{ x _ { 0 } , \ldots , x _ { i - 1 } \}$ 8: Sample $s \sim \mathrm { U n i f o r m } ( 1 , 2 , \dots , d )$   
9: Self-rollout $x _ { i }$ using $N _ { \theta }$ with $C _ { i }$ and $s$ denoising steps   
10: end for   
11: Align context memory $C ^ { t e a } \gets C _ { j : j + n - 1 } - x _ { j : j + n - 1 }$   
12: Sample diffusion timestep $k \sim [ 0 , 1 ]$   
13: $\hat { x } _ { j : j + n - 1 }  A d d N o i s e ( x _ { j : j + n - 1 , } k )$   
14: Compute fke score $S ^ { f a k e } \gets V _ { \beta } ^ { f a k e } ( \hat { x } _ { j : j + n - 1 } , C ^ { t e a } , k )$   
15: Compute real score $S ^ { f a k e } \gets V ^ { r e a l } ( \hat { x } _ { j : j + n - 1 } , C ^ { t e a } , k )$   
16: Update $\theta$ via distribution matching loss   
17: Update $\beta$ via flow matching loss as in [21]

# Algorithm 2 Inference with KV Cache

Require: Number of inference chunks $n _ { c }$   
Require: Denoise timesteps $\{ k _ { 1 } , \ldots , k _ { d } \}$   
Require: Number of inference chunks $n _ { c }$   
Require: AR diffusion model $N _ { \theta }$ (returns KV embeddings via $\mathsf { \bar { N } } _ { \theta } ^ { \mathrm { K V } } ,$   
1: Initialize model output $X _ { \theta } \gets [ ]$   
Initialize KV cache $\mathbf { K V } \gets [ ]$   
3: for $i = 0 , \ldots , n _ { c } - 1$ do   
4: Initialize $x _ { i } \sim \mathcal { N } ( 0 , I )$   
5: Reconstitute context memory $C _ { i } \subseteq \{ x _ { 0 } , \ldots , x _ { i - 1 } \}$   
6: for $s = d , \ldots , 1$ do   
7: if $s = d$ and $i > 1$ then   
8: Reset $\mathbf { K V }  N _ { \theta } ^ { \mathrm { K V } } ( C _ { i } , 0 )$   
9: end if   
10: Denoise $x _ { i } \gets N _ { \theta } ( x _ { i } , \mathbf { K } \mathbf { V } , k _ { s } )$   
11: end for   
12: Add output $X _ { \theta }$ .append $( x _ { i } )$   
13:end for   
14: return $X _ { \theta }$

# B. Dataset

Table 5 provides a comprehensive breakdown of our dataset. We deliberately curate a diverse and high-quality

<table><tr><td>Category</td><td>Data Source</td><td>Annotation (discrete, continuous)</td><td># Clips</td><td>Ratios</td></tr><tr><td>Real-World Dynamics</td><td>Sekai [36]</td><td>(x,x)</td><td>40K</td><td>12.5%</td></tr><tr><td>Real-World 3D Scene</td><td>DL3DV [40]</td><td>(V.v)</td><td>60K</td><td>18.75%</td></tr><tr><td>Synthetic 3D Scene</td><td>UE Rendering</td><td>(x,V)</td><td>50K</td><td>15.625%</td></tr><tr><td>Simulation Dynamics</td><td>Game Video Recordings</td><td>(v,x)</td><td>170K</td><td>53.125%</td></tr></table>

continuous), the number of clips, and their corresponding ratio in the final dataset.

![](images/10.jpg)  
Figure 10. Camera trajectories included in our collected dataset.

collection, encompassing data from the simulation engine and real world, as well as static and dynamic environments, to guarantee the strong generalization of our model.

For Real-World Dynamics, we employ the Sekai dataset [36]. However, the original videos often suffer from scene clutter and high dynamics. To address these issues, we implement a rigorous filtering pipeline. Specifically, we apply a state-of-the-art object detection model (YOLO [51]) to identify the presence of crowds and vehicles. By setting an empirical threshold, we filter out clips with high densities of moving objects, thereby ensuring annotation accuracy and stable training.

Regarding the Real-World 3D Scene data (DL3DV [40]), the original videos lack diversity in camera movement speed and trajectory complexity. To overcome this, we implement a sophisticated processing workflow: 3D Scene Reconstruction Customized Trajectory Rendering $ \mathrm { V i } \cdot$ . sual Quality Filtering Video Repair Post-processing (using Difix3D $^ +$ [66]). This procedure yields additional 60K high-quality real video clips featuring balanced movement speed. During the customized trajectory rendering stage, we deliberately design diverse revisit trajectories to facilitate the learning of long-term geometric consistency. The discrete actions and continuous camera poses in these rendered data are highly accurate, which helps the model learn well-structured action patterns.

For Synthetic 3D Scene (UE Rendering) data, we collect hundreds of UE scenes and obtain 50K video clips by rendering complex, customized trajectories. For Simulation Dynamics (Game Video Recordings), we establish a dedicated game recording platform and invite players to record 170K video clips from 1st/3rd-person AAA games.

We segment the original long videos into 30 to 40 seconds clips and employ a vision-language model to produce descriptive text annotations for every clip. Subsequently, we leverage VIPE [20] to generate high-quality camera poses for clips without camera annotations. However, given the long duration and high scene diversity of our dataset, we observe that pose estimation could be inaccurate, i.e., pose collapse. Therefore, we filter out videos whose adjacent frames exhibit erratic camera positions or rotation angles. Finally, for clips lacking discrete action annotations, we derive them from the continuous camera poses: we project the rotation and translation components onto the $x , y , z$ axes and apply a threshold to map these continuous values into corresponding discrete action states.

Fig. 10 illustrates the camera trajectories. Our dataset contains complex and diverse trajectories, including a large number of revisit trajectories, which enables our model to learn precise action control and long-term geometric consistency.

# C. Additional Experimental Results

# C.1. More Qualitative Results

Fig. 11 illustrates the results of WorldPlay under various actions and virtual environments. As shown in the first three rows, we can interact with complex composite actions, e.g., various combinations of movements. Moreover, WorldPlay can follow intricate trajectories, such as complex rotations and alternating sequences of rotations and movements as demonstrated in the middle six rows. This enhanced control capability is enabled by our dual action representation, which allows for more precise and reliable action guidance.

![](images/11.jpg)  
Figure 11. More qualitative results.

Furthermore, WorldPlay exhibits strong generalization, enabling it to control different types of agents, e.g., human or animals, to roam within the scenes as shown in the last two rows. For more intuitive perspectives, please refer to the supplementary videos.

# C.2. Long Video Generation

Fig. 12 presents long video generation results from WorldPlay, we maintain long-term consistency, e.g., frame 1 and frame 252 in the top two examples, and preserve high visual quality throughout the entire sequence. Moreover, our

![](images/12.jpg)  
Figure 12. Long video generation.

L

Figure 13. Visualization of different models under context forcing.   
Table 6. Comparison of Models under Context Forcing. The results are evaluated on the long-term test data. Student (AR) denotes the AR model before distillation, Teacher (bidirectional) refers to the memory-augmented bidirectional video diffusion model, and Final (distilled) represents the AR model after distillation. NFE denotes the number of function evaluations.   

<table><tr><td></td><td>NFE</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>Rdist ↓</td><td>Tdist ↓</td></tr><tr><td>Student (AR)</td><td>100</td><td>16.27</td><td>0.425</td><td>0.495</td><td>0.611</td><td>0.991</td></tr><tr><td>Teacher (Bidirectional)</td><td>100</td><td>19.31</td><td>0.599</td><td>0.383</td><td>0.209</td><td>0.717</td></tr><tr><td>Final (Distilled)</td><td>4</td><td>18.94</td><td>0.585</td><td>0.371</td><td>0.332</td><td>0.797</td></tr></table>

context memory ensures that the generation time for each chunk remains constant and does not increase as the video length grows, enabling real-time interactivity and enhancing the user's immersive experience.

# C.3. Comparison of Models under Context Forcing

We provide a comprehensive comparison of different models under context forcing in Table 6 and Fig. 13. The teacher model exhibits better control capability and visual quality due to the bidirectional nature, which provides reliable guidance during distillation. However, this limits its realtime interactivity. Through context forcing, we mitigate error accumulation while maintaining and even surpassing long-term consistency of the student model, yielding improved overall performance. In addition, context forcing reduces the student model's inference steps, enabling realtime interaction.

# C.4. Ablation for Memory Size

Table 7 evaluates the effect of different memory sizes. Using a larger spatial memory size leads to slightly better PSNR metric, while a larger temporal memory size better preserves the pretrained model's temporal continuity, resulting in better overall performance. Moreover, a larger spatial memory size may significantly increase the teacher model's memory size, as the spatial memory of adjacent chunks may completely differ, while their temporal memory overlaps. This not only increases the difficulty of training the teacher model but also poses challenges for distillation.

# C.5. Evaluation on VBench

We evaluate our model on VBench [22] across diverse metrics. For each baseline, we provide the same image and action to generate long-horizon videos. The results presented in Fig. 14 demonstrate the superior performance of WorldPlay. Notably, our method achieves outstanding results in key aspects such as consistency, motion smoothness, and scene generalizability.

Table 7. Ablation for memory size. Spa. and Tem. denote the number of chunks in spatial memory and temporal memory, respectively.   

<table><tr><td>Spa.</td><td>Tem.</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>Rdist ↓</td><td>Tdist ↓</td></tr><tr><td>3</td><td></td><td>16.41</td><td>0.418</td><td>0.502</td><td>0.634</td><td>1.054</td></tr><tr><td></td><td>3</td><td>16.27</td><td>0.425</td><td>0.495</td><td>0.611</td><td>0.991</td></tr></table>

![](images/13.jpg)  
Figure 14. VBench evaluation.

![](images/14.jpg)  
Figure 15. Human evaluation.

# D. User Study

We conduct a comprehensive user study across multiple dimensions, including visual quality, control accuracy, and long-term consistency. In our setup, users are presented with two videos, generated from the same initial image and action inputs, and asked to select their preference based on the specified criteria. To ensure the robustness of our evaluation, we select 300 cases from diverse benchmarks such as VBench [22] and WorldScore [11], and 300 customized trajectories. The final results are then evaluated by a panel of 30 assessors. As shown in Fig. 15, compared to other baselines, our distilled model achieves superior generation quality across all aforementioned evaluation metrics, clearly demonstrating our model's capability for both realtime interaction and long-term consistency.

![](images/15.jpg)  
Figure 16. Visualization of promptable event and video continuation.

![](images/16.jpg)  
Figure 17. 3D reconstruction results.

# E. Additional Applications

# E.1. 3D Reconstruction

Fig. 17 presents additional 3D reconstruction results. With our reconstituted context memory, we maintain temporal consistency and ensure long-term geometric consistency, which is essential for reliable 3D reconstruction. This is further validated by the consistency and compactness in the reconstructed point clouds. By generating diverse 3D scenes, this provides the potential to augment the scarce 3D datasets.

# E.2. Promptable Event

Due to the autoregressive nature of WorldPlay, we can modify the text prompt at any time to control the subsequent generated content. Specifically, inspired by LongLive [68], we employ a KV-recache technique to refresh the cached key—value states whenever the text prompt is modified. This effectively erases residual information from the previous prompt while preserving the motion and visual cues necessary to maintain temporal continuity. As shown in the upper part of Fig. 16, we can change the weather and trigger a fire eruption, or introduce new objects and characters. Through promptable event, we can generate various complex and uncommon scenarios, which can benefit agent learning by enabling agents to handle these unexpected situations.

# E.3. Video Continuation

As shown at the bottom of Fig. 16, WorldPlay can generate follow-up content that remains highly consistent with a given initial video clip in terms of motion, appearance, and lighting. This enables stable video continuation, effectively extending the original video while preserving spatialtemporal consistency and content coherence, which opens up new possibilities in creative video generation and virtual environment construction.

# F. Limitations

While WorldPlay demonstrates strong performance, extending the framework to generate videos with longer durations, multi-agent interactions, and more complex physical dynamics still requires further investigation. Moreover, Expanding the action types to a broader set is another promising direction. These challenges remain open for future research.