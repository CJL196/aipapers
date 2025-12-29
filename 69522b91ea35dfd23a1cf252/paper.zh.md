# StoryMem：基于记忆的多轮长视频讲故事

Kaiwen Zhang $^ { 1 , 2 , * }$ , Liming Jiang²†, Angtian Wang², Jacob Zhiyuan Fang², Tiancheng $\mathbf { z h i ^ { 2 } }$ , Qing Yan²², Hao Kang², Xin Lu²², Xingang Pan¹,³ 1 南洋理工大学 S-Lab, 智能创意, 字节跳动 * 在字节跳动实习期间的工作 † 项目负责人 § 通讯作者

# 摘要

视觉叙事需要生成具有电影质量和长距离一致性的多镜头视频。受到人类记忆的启发，我们提出了StoryMem，一个将长篇视频叙事重新定义为基于明确视觉记忆的迭代镜头合成的范式，将预训练的单镜头视频扩散模型转变为多镜头讲述者。这通过一种新颖的记忆到视频（M2v）设计来实现，该设计保持了一个紧凑且动态更新的记忆库，存储来自历史生成镜头的关键帧。然后，通过潜在级联和负RoPE偏移，仅使用LoRA微调的方法，将存储的记忆注入单镜头视频扩散模型。语义关键帧选择策略以及美学偏好过滤进一步确保生成过程中的信息稳定记忆。此外，所提出的框架自然适应平滑的镜头过渡和定制的故事生成应用。为了便于评估，我们引入了ST-Bench，一个多镜头视频叙事的多样化基准。大量实验表明，StoryMem在跨镜头一致性上优于以前的方法，同时保持高美学质量和提示遵循，标志着朝着连贯的分钟级视频叙事迈出了重要一步。日期：2025年12月23日 项目页面：https://kevin-thu.github.io/StoryMem

# 1 引言

讲故事是人类创造力的核心表达，从洞穴艺术到电影跨越了悠久的历史。近年来，视频扩散模型的显著进展使得合成单次短视频能够达到接近电影级的视觉真实度。一个自然的后续步骤是超越孤立的片段，使视频模型能够构建连贯的视觉叙事。然而，真正的讲故事需要在镜头和场景之间实现多层次的连贯性，从角色和环境的低级一致性到视觉风格和叙事流程的高级对齐。实现这样长度达到一分钟的多镜头叙事生成仍然是一个重大挑战。

现有的解决方案主要分为两类。理论模型通常与基于信号的视频扩散模型联合工作。例如，LCT [8] 采用全注意力机制结合交错的3D RoPE [39] 来捕捉镜头间的依赖关系，但随着序列长度的增加，训练和推理的开销呈二次方增长。后续方法通过 token 压缩 [47] 或稀疏注意力机制 [17, 31] 来提高效率，但仍然需要对多镜头视频进行大量重新训练，并且通常在视觉质量上低于其预训练的单镜头基础模型。第二类方法采用基于关键帧的解耦管线 [13, 29, 45, 53, 56]：每个镜头的第一帧通过上下文图像模型 [15, 40, 54] 生成，然后扩展成一个由预训练的图像到视频 (I2V) 模型生成的剪辑。尽管这种方法高效并利用了高质量的单镜头模型，但这种独立镜头的设计缺乏时间感知，即没有上下文在镜头间传播，导致视觉细节不一致、场景演变不连贯以及过渡生硬。因此，诸如新角色的出现或摄像机视角的变化等变动在整个长视频中无法保持一致。

![](images/1.jpg)  
Figur1 Given a story sript with per-shot text descriptions, StoryMem generates appealing minute-ong, multi-shot iv  uayTro o generation using a memory-conditioned single-shot video diffusion model.

这两条研究反映了叙事视频生成中的一个核心难题：联合训练需要大量计算资源和稀缺的高质量多镜头数据，而解耦方法则面临不一致性的问题。在本研究中，我们探索了一条第三路径，既实现高一致性又具备高效率。我们的关键见解是，高质量的预训练单镜头视频扩散模型可以有效适应长期连贯叙事，前提是结合视觉记忆。受到人类记忆选择性保留显著视觉信息的启发，我们引入了StoryMem，这是一种将长篇视频叙事重新构思为基于过去关键帧和每镜头描述的迭代镜头合成的新范式。该范式的核心是我们的记忆到视频（M2v）设计，一个轻量框架，通过将显式视觉记忆注入单镜头视频扩散模型来强制执行跨镜头一致性。StoryMem并不是在长视频数据上训练一个单一的整体模型，而是通过维护一个存储关键视觉上下文的紧凑记忆库逐镜头生成故事，例如角色、场景及之前生成镜头的风格提示。然后，这些记忆作为全局条件信号，引导后续的生成朝着一致的视觉语义和流畅的场景演变（见图1）。

为实现记忆条件生成，我们引入了一种简单而有效的设计，该设计结合了记忆潜在拼接和负向旋转位置编码偏移，自然地扩展了预训练的图像到视频（I2V）模型，以捕捉长期依赖并保持上下文级一致性。我们进一步提出了一种基于 CLIP 特征的语义关键帧选择策略，并通过 HPSv3 进行美学偏好过滤，以保持一个既信息丰富又可靠的紧凑记忆库。在生成过程中，记忆动态提取、更新并注入模型，以指导每个新镜头。值得注意的是，StoryMem 仅需在语义一致的短视频剪辑上进行 LoRA 微调，实现了强大的跨镜头一致性，同时保持了预训练单镜头视频扩散模型的高视觉质量。除了文本驱动的故事生成，StoryMem 作为一个多功能框架，适用于更广泛的范式，当与 I2V 控制结合时，可实现自然场景转换，并通过将参考图像融入记忆中来支持定制的参考到视频（R2V）生成。为了支持评估和比较，我们提出了 ST-Bench，一种涵盖多种复杂叙事的多场景、多镜头视频讲故事基准。大量实验证明，StoryMem 在保持高视觉保真度和强提示一致性的同时，实现了优越的跨镜头一致性，超越了最先进的方法。我们的方法标志着向连贯的时长一分钟视频讲故事迈出了重要一步，缩短了单镜头视频扩散与长格式视觉叙事之间的差距。

# 2 相关工作

基于关键帧的故事生成。先前的视觉叙事研究主要集中在故事图像生成上。例如，StoryDiffusion引入了一致性自注意力机制到文本生成图像（T2I）模型中，以产生角色一致的分镜头。目前，随着视频生成模型的出现，后续工作通过基于关键帧的代理框架将图像处理扩展到视频。这些方法首先使用故事图像模型或上下文图像编辑模型生成关键帧，然后利用图像到视频（I2V）模型将其扩展为视频片段。然而，一致性仅在关键帧级别得到保证，各个镜头基本上是独立的，镜头之间的过渡往往显得生硬。相比之下，我们引入了一种记忆到视频（M2V）框架，以逐镜头的方式生成连贯的故事视频，同时以最小的计算开销保留镜头间的上下文信息。

多镜头长视频生成。除了基于关键帧的叙事，另一条研究方向针对一般的多镜头长视频生成，通过微调预训练的单镜头模型（例如，Wan [42]）。LCT [8] 在这一方向上开创性地通过全注意力机制 [41] 和交织的三维RoRE [39] 联合建模多镜头视频，实现了强大的跨镜头一致性，但引入了平方级的计算成本。最近，一系列与LCT框架相关的研究通过词元压缩 [47] 或稀疏注意力 [7, 31] 提高了效率。然而，它们需要大规模的多镜头视频数据训练，通常在质量上较其预训练的高质量单镜头基础模型存在退化。相比之下，我们的M2V框架只需对短视频片段进行轻量级的LoRA [12] 微调，从而在保持基础模型视觉保真度的同时，无缝适应其他范式，如图像到视频（I2V）和参考到视频（R2V）进行定制化故事生成。视频生成中的记忆机制。记忆是智能系统的一项基本能力，已在大型语言模型中得到广泛研究 [4, 46, 52]。相较之下，视频生成的记忆机制仍然在很大程度上未被探索。现有的尝试通过推理时训练 [11]、三维建模 [14, 24] 或基于相机的记忆帧检索与交叉注意力 [48, 51] 将记忆引入视频世界模型。然而，它们主要针对在可控的世界模拟下的空间一致性，并依赖于动作或相机姿态等辅助控制输入，从而限制了其在通用视频生成中的适用性。

![](images/2.jpg)  
Figure 2 Overview of StoryMem. StoryMem generates each shot conditioned on a memory bank that stores keams fom previusy nrate shots. Duri eneration, the seecmemory ame are ncoded b 3D VAE, fused with noisy video latents and binary masks, and fed into a LoRA-finetuned memory-conditioned Video DiT to apivo narrative progression. Byerativey enerating shots wihmemory updatesStoryMem produce coherent minue-n, multi-shot story videos.

据我们所知，我们首次在通用视频生成模型中引入了显式记忆机制。这使得模型能够在不断演变的生成过程中记忆重要的角色和背景场景，从而在一分钟的生成中保持一致性。

# 3 方法论

在本节中，我们介绍了StoryMem，这是一种基于单次视频扩散模型的新型管道，旨在解决连贯的多镜头故事生成挑战。我们首先在3.1节中介绍基础知识。接着，在3.2节中正式化问题，并在3.3节中介绍我们提出的记忆到视频（M2V）机制，该机制通过基于记忆的视频生成来强制实现跨镜头一致性。3.4节进一步展示了我们的记忆提取和更新策略，以将微调后的M2V模型应用于长篇多镜头讲故事。最后，3.5节将M2V扩展为MI2V和MR2V，以实现自然场景转换和自定义控制。StoryMem的概述见图2。

# 3.1 初步研究

视频扩散模型。我们的方法建立在潜在视频扩散模型 [10, 38, 42] 之上。扩散过程作用于视频潜变量 $\boldsymbol { z } _ { 0 } = \boldsymbol { \mathcal { E } } ( \boldsymbol { v } ) \in \mathbb { R } ^ { c \times f \times h \times w }$，通过 3D VAE [19] 编码器 $\mathcal { E }$ 编码 RGB 视频 $\boldsymbol { v }$。扩散模型通过学习将噪声样本 $z _ { 1 } = \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ 转换为目标数据样本 $z _ { 0 }$ 的微分方程来学习视频潜变量的分布：

$$
d z _ { t } = v _ { \Theta } ( z _ { t } , t ) d t , \quad t \in [ 0 , 1 ] ,
$$

速度场 $\boldsymbol { v } _ { \Theta }$ 由神经网络 $\Theta$ 参数化。该网络在修正流 [26, 28] 形式下，通过速度预测损失进行训练：

$$
\begin{array} { r } { z _ { t } = ( 1 - t ) z _ { 0 } + t \epsilon , \quad \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) , \quad } \\ { \mathcal { L } _ { R F } = \mathbb { E } _ { z _ { 0 } , \epsilon , t } \left[ \left\| v _ { \Theta } ( z _ { t } , t ) - ( z _ { 0 } - \epsilon ) \right\| ^ { 2 } \right] . } \end{array}
$$

我们采用基于最新技术的单次视频扩散模型 Wan2.2-I2V [42] 作为基础模型，该模型使用扩散变换器（DiT）[35] 作为速度预测网络 $\Theta$。每个 DiT 模块包含自注意力机制，负责视频内部依赖关系建模，以及跨注意力机制，用于文本条件设置，并使用 3D 旋转位置嵌入（RoPE）[39] 来编码空间和时间坐标。Wan-I2V 模型进一步依据第一帧图像进行条件设置，以指导视频合成。该图像与填充零的帧拼接，并通过 3D VAE 编码为条件潜变量 $\boldsymbol { z } _ { c } \in \mathbb { R } ^ { c \times f \times h \times w }$。一个二进制掩码 $M \in \{ 0 , 1 \} ^ { s \times f \times h \times w }$ 表示哪些帧是保留的或生成的，其中 $s$ 表示 3D VAE 的时间步幅，$F = s \times f$ 为原始视频的帧数。在扩散过程中，噪声潜变量 $z _ { t }$、条件潜变量 $z _ { c }$ 和掩码 $M$ 沿通道维度拼接，并送入 DiT。我们的 M2V 设计利用了这种掩码引导的条件扩散架构，将条件扩展到基于关键帧的记忆上下文，以实现多次生成。

# 3.2 问题表述

给定一组文本分段 $\mathcal { T } = \{ t _ { i } \} _ { i = 1 } ^ { N }$，我们的目标是生成一个连贯的多镜头叙事视频 $\mathcal { V } = \{ v _ { i } \} _ { i = 1 } ^ { N }$。这个任务可以表示为学习条件分布：

$$
p _ { \Theta } ( \mathcal { V } \mid \mathcal { T } ) = p _ { \Theta } ( v _ { 1 : N } \mid t _ { 1 : N } ) ,
$$

该方法同时捕捉文本提示之间的语义关系和生成镜头之间的视觉一致性。为了利用现有单镜头视频扩散模型的强大能力，我们以自回归的方式分解联合分布，这与故事的自然进展相一致：

$$
p _ { \Theta } ( \mathcal { V } \mid \mathcal { T } ) = \prod _ { i = 1 } ^ { N } p _ { \Theta } ( v _ { i } \mid v _ { 1 : i - 1 } , \mathcal { T } ) .
$$

然而，由于视频帧包含大量时间冗余，直接条件化于所有过去的视频镜头 $v_{1:i-1}$ $\mathcal{K} = \{ k_{i} \}_{i=0}^{N}$，通过一个一致的图像生成器 $p_{\psi}$，然后独立地使用图像到视频模型合成每个镜头，即，

$$
p _ { \Theta , \psi } ( \mathcal { V } , K \mid \mathcal { T } ) \approx p _ { \psi } ( \mathcal { K } \mid \mathcal { T } ) \prod _ { i = 1 } ^ { N } p _ { \Theta } ( v _ { i } \mid t _ { i } , k _ { i } ) .
$$

然而，这种表述缺乏时间意识，固有地受到有限视频上下文和僵化过渡的影响，未能捕捉不同镜头之间不断发展的叙事。受到人类记忆的启发，人类记忆选择性地保留关键视觉印象而非完整体验，我们引入了一种明确的基于关键帧的记忆机制，其中压缩记忆 $m_{i}$ 存储高达镜头 $i$ 的关键视觉上下文。令 $\mathcal{M} = \{ m_{i} \}_{i=0}^{N}$，$m_{0}$ 是空的，第一个镜头仅根据其文本描述 $t_{1}$ 生成，但我们的框架也支持可选的替代初始化。联合分布 $p_{\Theta, \phi}(\mathcal{V}, \mathcal{M} \mid \mathcal{T})$ 可以写成：

$$
p _ { \phi } ( m _ { 0 } ) \prod _ { i = 1 } ^ { N } p _ { \Theta } \big ( v _ { i } \mid t _ { i } , m _ { i - 1 } \big ) p _ { \phi } \big ( m _ { i } \mid m _ { i - 1 } , v _ { i } , t _ { i } \big ) ,
$$

其中 $p _ { \Theta }$ 参数化了镜头级视频生成器，而 $p _ { \phi }$ 表示记忆更新机制。在我们的框架中，$m _ { i }$ 被实现为从之前镜头中提取的小集合关键帧，为后续生成提供明确的视觉锚点。记忆更新作为一个确定性函数实现：

$$
m _ { i } = f _ { \phi } ( m _ { i - 1 } , v _ { i } ) ,
$$

得到简化的因式分解：

$$
p _ { \Theta } ( \mathcal { V } \mid \mathcal { T } ) \approx \prod _ { i = 1 } ^ { N } p _ { \Theta } ( v _ { i } \mid t _ { i } , m _ { i - 1 } ) .
$$

该公式实现了基于记忆的多次生成，其中每次生成不仅依赖于其文本描述，还依赖于一个不断演变的记忆，该记忆总结了来自先前生成的角色、场景和风格信息，从而在整个视频中确保跨镜头的一致性和叙事连贯性。

# 3.3 记忆到视频 (M2V)

基于上述公式，我们提出了记忆到视频（M2V）机制，通过带有负RoPE偏移的记忆条件视频扩散模型实现条件分布 $p _ { \Theta } ( v _ { i } \mid t _ { i } , m _ { i - 1 } )$。

记忆作为条件。为了实现基于记忆的视频生成，我们借鉴了预训练的图像条件视频模型 Wan-I2V 的架构设计。我们首先使用与基础模型相同的 3D VAE 编码器 $\varepsilon$ 将记忆帧编码为记忆潜变量 $\boldsymbol { z } _ { m } \in \mathbb { R } ^ { c \times f _ { m } \times h \times w }$，其中每个潜变量对应一帧记忆图像，没有时间压缩。这些记忆潜变量与沿时间维度从零填充帧编码得到的潜变量串联在一起，形成条件潜变量 $z _ { c }$。一个二进制掩码 $M \in \{ 0 , 1 \} ^ { s \times ( f _ { m } + f ) \times h \times w }$ 被引入 DiT 以仅生成未被掩盖的帧。根据 I2V 模型设计，将噪声潜变量、记忆条件潜变量以及掩码在通道维度上串联在一起，并输入 DiT 进行速度预测。

负 RoPE 移位。将记忆帧融入视频序列引发了位置编码问题：记忆帧与当前镜头在时间上并不连续，而是代表摘要过去内容的离散关键帧。为了解决这个问题，我们利用 3D RoPE 的可推广性，并通过将负帧索引分配给记忆潜变量来扩展它。具体来说，对于一个具有 $f$ 个潜在帧和 $f _ { m }$ 个记忆帧的镜头，时间位置索引定义为 $\{ - f _ { m } S , - ( f _ { m } - 1 ) S , \ldots , - S , 0 , 1 , \ldots , f - 1 \} ,$ 其中 $S$ 是一个固定的偏移量，表示记忆潜变量与视频潜变量之间的时间间隔。这种负 RoPE 移位自然地将记忆帧嵌入为前置事件，保持当前视频从零开始的原始时间编码，使得 DiT 能够在统一的时间空间内无缝关注先前和当前的上下文。 数据整理与训练。与依赖于多镜头长序列的联合建模方法不同，我们的模型可以有效地在单镜头视频上进行训练。为了整理记忆视频一致性数据，我们从以下来源收集训练数据：（1）通过镜头级相似性将来自电影单镜头视频数据集的视觉相关短片进行分组。在每个被选中的组内，目标镜头被选择，而其他镜头作为记忆进行采样。然后模型训练以重建基于提供的记忆帧的目标视频，遵循 I2V 模型中使用的纠正流损失（方程 3）。在推理时，记忆部分被丢弃，只有新生成的片段被解码为视频镜头。

# 3.4 内存提取与更新

经过训练，微调后的 M2V 模型学习根据提示指令从记忆库中提取相关上下文，生成具有连贯角色和背景的多样化镜头。为了将我们的 M2V 模型应用于多镜头长视频生成，我们进一步设计了一个记忆更新函数 $f _ { \phi }$，该函数模拟人类记忆，通过保留具有代表性和意义的时刻并丢弃冗余信息。为此，我们提出了一种记忆帧提取策略，选择语义上重要且美学上可靠的关键帧作为记忆。 语义关键帧选择。对于每个生成的镜头，我们旨在识别一小组独特的关键帧作为记忆。为了捕捉逐帧语义，我们计算所有帧的 CLIP [36] 嵌入，并依次选择关键帧：第一帧固定，每个后续帧通过余弦相似度与最新的关键帧进行比较。当相似度低于动态阈值时，添加新的关键帧，该阈值开始时较低，如果选定帧的数量超过预设的上限，则逐渐增加。这个自适应策略在保留多样化视觉上下文的同时去除冗余。 美学偏好过滤。虽然语义选择有效地识别了不同的内容，但并不能保证图像质量良好，具有较大运动的镜头可能错误地被选为关键帧，提供有限的上下文指导。我们通过进一步使用 HPSv3 [30] 作为美学奖励模型进行过滤，以确保记忆清晰且在视觉上可靠。

![](images/3.jpg)  
Figure 3 Qualitative comparison. Our StoryMem generates coherent multi-scene, multi-shot story videos aligned w per-ho eptions.n ontrast,he reraimode ankeambasbaselnes  preeveo character and scene consistency, while HoloCine [31] exhibits noticeable degradation in visual quality.

多镜头长视频生成。整合上述组件，我们使用 M2V 模型进行多镜头生成，并通过一个动态内存库在镜头之间更新关键帧。在生成每个镜头 $v _ { i }$ 后，我们将其提取的关键帧与 CLIP [36] 空间中的现有记忆进行比较，仅添加语义上不同的关键帧，将 $m _ { i - 1 }$ 更新为 $m _ { i }$。为了防止内存库的 uncontrolled growth，我们采用混合记忆-接收器 $^ +$ 滑动窗口策略：早期关键帧作为长期锚点固定，保持全局一致性，而最近的关键帧形成一个短期窗口，捕捉局部依赖关系。如果达到容量限制，则丢弃最旧的短期记忆。如果需要，人工创作者或大型视觉语言模型可以进一步审查和优化所选关键帧以实现更细致的故事特定控制，尽管我们结果中并未使用此选项。

# 3.5 扩展到 MI2V 和 MR2V

如第3.2节所述，我们的框架提供了灵活的设计空间，并且可以与其他正交的视频生成范式无缝适配。

![](images/4.jpg)  
Figur4User study.Our method is consistently prefered over all baselines in most dimensions, highlighting its supuloeaivcehaus preethe Tie indicates no significant preference, and Lose indicates that users prefer the baseline.

一个重要的设计选择是将 M2V 与 I2V 结合起来。尽管基于记忆的范式有效地解决了跨镜头一致性的问题，但当将多个镜头拼接成一个长视频时，镜头之间的过渡可能仍然显得不自然，有时会导致非因果运动。一种实用的解决方案是加入场景切换指示器，让脚本创作者（LLM 或人类）决定是否在每两个相邻镜头之间进行场景切换。如果未明确指定切换，模型将重用前一个镜头的最后一帧作为下一个镜头的第一帧，从而实现更平滑和更自然的连贯性。另一个应用是个性化初始化记忆状态 $m _ { 0 }$。例如，用户可以提供角色或背景参考图像作为初始记忆，从而实现定制化的多镜头视频生成。

# 4 实验

# 4.1 实现细节

我们的框架建立在最先进的开源视频生成模型 Wan2.2-I2V-A14B [42] 之上，该模型具有 140 亿个活跃参数。我们使用 rank-128 的 LoRA 对 DiT 模块中的线性层进行微调，新增约 0.7 亿个活跃参数。M2V 模型在一个精心策划的数据集上训练，该数据集包含 40 万个五秒单次拍摄视频，每个剪辑与 15 个语义一致的视频相匹配。在训练过程中，记忆长度随机从 1 到 10 帧中采样，负 RoPE 移动偏移 $S$ 设置为 5。在推理过程中，我们使用 3 帧的记忆池大小，每次拍摄的关键帧限制为 3，初始语义相似性阈值为 0.9，以及美学评分阈值为 3.0。

# 4.2 ST-基准

多镜头长视频叙事仍然是一个相对较新且未被充分探索的研究领域，尚无标准的评价基准。最相关的开放基准 ViStoryBench 专注于故事板图像生成，不适用于我们的 视频生成任务。为了全面评估我们的方法，我们建立了一个新的多场景、多镜头故事视频生成基准，称为 ST-Bench。我们提示 GPT-5 创作 30 个涵盖多种风格的长篇故事剧本，每个剧本包含故事概述、812 个镜头级文本提示、相应的首帧提示（仅适用于基于两阶段关键帧的方法）以及场景切换指示器。总的来说，ST-Bench 提供了 300 个详细的视频提示，描述了角色、场景、动态事件、镜头类型以及可能的摄影机运动。关于 ST-Bench 的更多细节见附录 B。我们希望这个基准能够促进未来长篇故事视频生成的研究。

![](images/5.jpg)  
FuR ults.SorMemnable si soviatinus eeae  h memory. The real-person reference images were used with proper consent from the individuals involved.

# 4.3 评估

基线设置。为了展示StoryMem的优越性，我们遵循HoloCine [31] 的评估协议，将StoryMem与三种多镜头长视频生成的代表性范式进行比较：（1）独立应用于每个镜头的预训练视频扩散模型Wan2.2-T2V-A14B，作为单镜头质量的基准；（2）基于关键帧的两阶段方法，StoryDiffusion [54] 和 IC-LoRA [15]，关键帧通过Wan2.2-I2V-A14B进行扩展；（3）最先进的联合多镜头模型HoloCine [31]，通过微调Wan2.2-T2V-A14B进行整体一分钟视频生成。为了兼容，我们将ST-Bench中的提示转换为HoloCine所需的格式，使用GPT-5。 定性结果。我们提供了定性比较，以说明我们方法的优越性。如图3所示，我们的StoryMem生成了一致的多镜头故事视频，这些视频紧密跟随每个镜头的描述，同时保持高视觉保真度。角色的身份、外观和服装在不同场景和镜头之间保持一致。值得注意的是，在[镜头5]中，我们的模型有效地从[镜头2]中检索上下文信息，产生高度一致的街景，即使在多次镜头转换之后也是如此。相比之下，预训练模型和基于关键帧的基线未能保持长期一致性（例如，角色身份不匹配、服装变化、街景外观不一致），而HoloCine在视觉质量上明显退化。附录A提供了更多定性示例。 定量结果。根据之前的研究 [8, 31]，我们从三个方面评估所有方法：（1）美学质量通过VBench [16] 中采用的LAION美学预测器进行测量，反映包括色彩和谐、真实感和自然感在内的视觉吸引力。（2）提示遵循通过ViCLIP [44] 进行评估，比较生成的视频与故事脚本；全球评分计算整个多镜头视频与故事概述之间的余弦相似度，而单镜头评分则比较每个镜头与其对应的提示。（3）跨镜头一致性计算所有镜头对的平均ViCLIP相似度。由于不同镜头可能描绘不同的角色或场景，我们还报告了一个前10一致性评分，通过对与其提示特征相似度选择的十对最相关镜头进行平均。

![](images/6.jpg)  
[Shot 3] The student   
Figur6Ablation study. Top:our method ectively preserves newly merged characer consisency.Bottom:u method can maintain long-term visual fidelity. The selected keyframe in [Shot 1] is highlighted in blue box.

表1报告了所有方法在ST-Bench上的定量结果。我们的方法在跨镜头一致性指标上大幅超越了所有基准，比基础模型高出28.7%，比之前的最先进方法HoloCine高出9.4%的整体一致性。这突显了我们基于记忆的设计在保持跨镜头长程连贯性方面的优势。我们的方法在美学质量和提示遵循方面也取得了强劲的表现，达到了所有一致性视频生成方法中最高的美学评分和最佳的全局语义对齐。尽管单镜头的提示遵循评分稍低，但在我们的MI2设置下，这种现象是可以预期的，因为该设置引入了额外的控制，以实现更平滑和更自然的镜头过渡，并可能稍微限制了单镜头的表现。 用户研究。视频叙事是一项复杂且以人为中心的任务。现有的指标无法全面捕捉角色级别的跨镜头一致性或镜头过渡的自然性。因此，我们还进行了全面的用户研究，让人类评估者将我们的生成结果与每个基准在成对设置中进行比较。如图4所示，在大多数评估维度上，我们的结果受到强烈偏爱。尽管预训练的Wan2.2模型由于其独立的镜头生成设置在单镜头质量上表现良好，但它缺乏一致性机制，在其他维度上表现显著较差。 扩展应用。得益于我们基于记忆的框架的灵活设计，StoryMem还可以通过将参考图像视为初始记忆来支持定制化故事生成，如图所示。与传统的基于参考的方法相比，我们的方法通过同时保留新生成的场景和有选择地利用存储在记忆中的相关上下文，能够实现自然的叙事进展。我们的框架还与其他专门的保留参考技术兼容，但这并不是我们工作的重点。

# 4.4 消融研究

我们进行消融研究以验证我们所提出的四种技术的有效性。为了评估我们的方法在关键帧选择上的表现，我们将其与一种简单策略进行比较，该策略始终将每个镜头生成的第一个帧作为记忆。正如图6的顶部示例所示，这种简单选择未能捕捉到视频镜头中新增的人物（例如，教授），导致外观和服装的不一致。为了评估表2消融研究，我们研究了每种在StoryMem中提出的技术的有效性。

<table><tr><td rowspan="2">Method</td><td rowspan="2">Aesthetic Quality↑</td><td colspan="2">Prompt Following↑</td><td colspan="2">Cross-shot Consistency↑</td></tr><tr><td>Global</td><td>Single-shot</td><td>Overall</td><td>Top-10 Pairs</td></tr><tr><td>w/o Semantic Selection</td><td>0.6076</td><td>0.2257</td><td>0.2295</td><td>0.4878</td><td>0.5287</td></tr><tr><td>w/o Aesthetic Filtering</td><td>0.6018</td><td>0.2251</td><td>0.2313</td><td>0.4844</td><td>0.5330</td></tr><tr><td>w/o Memory Sink</td><td>0.6093</td><td>0.2277</td><td>0.2330</td><td>0.4891</td><td>0.5241</td></tr><tr><td>Ours</td><td>0.6133</td><td>0.2289</td><td>0.2313</td><td>0.5065</td><td>0.5337</td></tr></table>

谢尔顿坐在一个充满年长孩子的教室里，穿着 bow tie 和格子衬衫。他举起手... 谢尔顿自信地在黑板上用工整的书写解释答案... 谢尔顿独自坐在树下，阅读一本物理书，而其他孩子在踢足球...

![](images/7.jpg)

![](images/8.jpg)  
Improves consistency when character details are explicitly specified in each shot prompt

![](images/9.jpg)  
Figure7 Limitation. Top: Our method may struggle to preserve consistency in complex multi-character scenarios where pure visual memory becomes ambiguous.Bottom: Explicitly providing character details in each shot prompt will mitigate the problem.

审美过滤的重要性，我们在记忆提取过程中去除了这一步。图6中的底部示例显示，没有审美过滤，语义选择变得对噪声敏感，并且可能包含低质量的规范性帧，导致最低的审美评分。最后，为了检查记忆回退机制，一旦记忆库满，我们用完整的滑动窗口策略进行替换。这导致远程视觉保真度的下降，而我们的方法在微观生成中保持高质量和一致性。总体而言，如表2所示，我们的方法在大多数指标上实现了最佳性能，由于额外的初始记忆约束，单次提示随后的表现略有下降。

# 5 结论

我们提出了StoryMem，这是一种连贯的多镜头长视频讲述范式。为此，我们引入了记忆到视频（M2V）框架，通过隐式拼接、负RoPE位移和轻量级LoRA微调，增强了单镜头扩散模型的显式视觉记忆。结合精细的记忆提取与更新策略，StoryMem实现了逐镜头合成分钟级叙事视频。大量实验表明，StoryMem在跨镜头连贯性方面优于最先进的方法，同时保持了美学质量和提示一致性。尽管有效，StoryMem在复杂的多角色场景中可能仍会面临纯视觉记忆模糊的问题，当相邻镜头存在较大运动差异时，也难以实现完全平滑的过渡。未来的工作将探索更结构化、实体感知的记忆表示和改进的过渡建模，以更好地支持丰富而流畅的叙事。

参考文献 [1] A. Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, 和 Dominik Lorenz. 稳定视频扩散：将潜在视频扩散模型扩展到大规模数据集. ArXiv, abs/2311.15127, 2023. URL https://api.semanticscholar.org/CorpusID:265312551. [2] Shengqu Cai, Ceyuan Yang, Lvmin Zhang, Yuwei Guo, Junfei Xiao, Ziyan Yang, Yinghao Xu, Zhenheng Yang, Alan L. Yuille, Leonidas J. Guibas, Maneesh Agrawala, Lu Jiang, 和 Gordon Wetzstein. 用于长视频生成的上下文混合. ArXiv, abs/2508.21058, 2025. URL https://api.semanticscholar.org/CorpusID: 280950315. [3] David Dinkevich, Matan Levy, Omri Avrahami, Dvir Samuel, 和 Dani Lischinski. Story2board：一种无训练的表达性故事板生成方法. ArXiv, abs/2508.09983, 2025. URL https://apiemanticschoar.org/CorpusID:280642114. [4] Yiming Du, Wenyu Huang, Danna Zheng, Zhaowei Wang, Sbastien Montella, Mirella Lapata, Kam-Fai Wong, 和 Jeff Z. Pan. 重新思考人工智能中的记忆：分类法、操作、主题和未来方向. ArXiv, abs/2505.00675, 2025. URL https://api.semanticscholar.org/CorpusID:278237720. [5] Patrick Esser, Sumith Kulal, A. Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam Levi, Domink Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, 和 Robin Rombach. 扩展整流流变换器以进行高分辨率图像合成. ArXiv, abs/2403.03206, 2024. URL https://api.semanticscholar.org/CorpusID:268247980. [6] Google Deepmind. Gemini 2.5 快速图像（纳米香蕉）：使用Gemini创建和编辑图像. https://deepmind.google/models/gemini/image, 2025. [7] Google Deepmind. Veo3视频模型. https://deepmind.google/models/veo/, 2025. [8] Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, 和 Lu Jiang. 用于视频生成的长上下文调优. ArXiv, abs/2503.10589, 2025. URL https://api.semanticscholar.org/CorpusID:276961453. [9] Huiguo He, Huan Yang, Zixi Tuo, Yuan Zhou, Qiuyue Wang, Yuhang Zhang, Zeyu Liu, Wenhao Huang, Hongyang Chao, 和 Jian Yin. Dreamstory：由LLM引导的多主题一致扩散的开放域故事可视化. IEEE 模式分析与机器智能学报, PP, 2024. URL https://api.semanticscholar.org/CorpusID:271270047. [10] Jonathan Ho, Ajay Jain, 和 P. Abbeel. 去噪扩散概率模型. ArXiv, abs/2006.11239, 2020. URL https://api.semanticscholar.org/CorpusID:219955663. [11] Yinig Hong, Beide Liu, Maxine Wu, Yuanho Zhai, Kai-Wei Chang, Lingjie Li, Kevin Qinghong Lin, Chung-Ching Lin, Jiang Wang, Zhengn Yng Yin Wu 和 Lijun Wan. 驱动长视频生成的多模态一致性. ArXiv, abs/2410.23277, 2024. URL https://api.semanticscholar.org/CorpusID:273696012. [12] J. Edward Hu, Yelong Shen, Phillip Walls, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, 和 Weizhu Chen. LoRA：大语言模型的低秩适配. ArXiv, abs/2106.09685, 2021. URL https://api.semanticscholar.org/CorpusID:235458009. [13] Panwen Hu, Jin Jiang, Jianqi Chen, Minei Han, Shengai Liao, Xiaoju Chang, 和 Xiaodan Liang. Storagnt：通过多智能体协作进行定制化故事讲述视频生成. ArXiv, abs/2411.04925, 2024. URL https://api.semanticscholar.org/CorpusID:273878062. [14] Junc Huang, Xintng Hu, Boyo Han, Shaosuai Shi, Zhuoo Tian, Tianyu He, 和 Li Jiang. Memyorcin：用于Minecraft上连续场景生成的时空记忆, 2025. URL https://arxiv.org/abs/2510.03198. [15] Lianghua Huang, Wei Wang, Zhigang Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, 和 Jingren Zhou. 扩散变换器中的上下文LoRA. ArXiv, abs/2410.23775, 2024. URL https://api.semanticscholar.org/CorpusID:273707547. [16] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, 和 Ziwei.

刘。VBench：视频生成模型的全面基准测试套件。载于IEEE/CVF计算机视觉与模式识别会议论文集，2024。[17] 贾伟南，卢犇，黄梦琪，王华量，黄宾源，陈南，刘穆，姜继东，毛振东。Moga：端到端长视频生成的组混合注意力机制，2025。网址 https://arxiv.org/abs/2510.18692。[18] 毛聚聚，黄晓克，谢云飞，常源琦，惠慕德，徐冰杰，周宇尹。Story-adapter：一种无需训练的长故事可视化迭代框架。ArXiv, abs/2410.06244, 2024。网址 https://api.semanticscholar.org/CorpusID:273227855。[19] Diederik P. Kingma 和 Max Welling。自编码变分贝叶斯。ICLR, abs/1312.6114, 2014。网址 https://api.semanticscholar.org/CorpusID:216078090。[20] 孔维杰，田琦，张子健，敏罗克，戴左卓，周进，熊佳良，李鑫，吴博，张建伟，吴凯瑟，林沁，袁俊坤，龙彦新，王阿拉丁，王安东，李长霖，黄多军，杨帆，谭浩，王红梅，宋杰，白嘉望，吴建兵，薛金宝，王乔伊，王凯，刘梦扬，李鹏宇，李帅，王伟彦，余文清，邓希，李扬，陈怡，崔宇博，彭振，余珍，何志宇，徐志勇，周子翔，徐哲，杨丹涛，陆青林，刘松涛，周大权，王鸿发，杨永，王迪，刘宇弘，姜杰，和蔡塞萨尔。Hunyuanvideo：一个系统化的大型视频生成模型框架。ArXiv, abs/2412.03603, 2024。网址 https://api.semanticscholar.org/CorpusID:274514554。[21] 快手。Kling视频模型。https://kling.kuaishou.com，2025。[22] 黑森林实验室。Flux。https://github.com/black-forest-labs/flux，2024。[3] 黑森林实验室，Stephe Batiol，A. Blattn，Frederc Boesel，阿克沙姆·孔苏，阿基尔·迪亚涅，提姆·道克，杰克·英格利什，齐昂·英格利什，帕特里克·埃瑟，苏米特·库拉尔，凯尔·莱西，雅姆·莱维，李成，多米尼克·洛伦茨，乔纳斯·穆勒，达斯汀·波德尔，罗宾·荣巴赫，哈利·赛尼，阿克塞尔·索尔，和卢克·史密斯。Flux.1 kontext：潜在空间中的上下文图像生成与编辑的流匹配。ArXiv, abs/2506.15742, 2025。网址 https://api.semanticscholar.org/CorpusID:279464475。[ ] 林润佳，Philip H. S. Torr，安德烈亚·维达尔，和托马斯·雅卡布。Vmem：基于Surfels索引的视图记忆一致性交互视频场景生成。ArXiv, abs/2506.18903, 2025。网址 https://api.semanticscholar.org/CorpusID:279999602。[25] 李怡彤，甘哲，沈烨龙，刘靖国，程煜，吴悦新，劳伦斯·卡林，大卫·爱德温·卡森，和高剑锋。Storygan：一种用于故事可视化的顺序条件生成对抗网络。CVPR，2019。[26] 亚伦·利普曼，里基·T. Q. 陈，赫莉·本-哈穆，马西米利安·尼克尔，和马特·李。生成建模的流匹配。ArXiv, abs/2210.02747, 2022。网址 https://api.semanticscholar.org/CorpusID:252734897。[27] 刘畅，吴浩宁，钟宇杰，张晓宇，和谢维迪。智能格林：通过潜在扩散模型进行开放式视觉叙事。CVPR，2024。网址 https://api.semanticscholar.org/CorpusID:258999141。[8] 曹熙林，切尔龙，甘根卡尔，流动重构。ArXiv, abs/2209.03003, 2022。网址 https://api.semanticscholar.org/CorpusID:25211177。[29] 龙富辰，邱朝帆，姚婷，和梅涛。Vidostudio：生成一致内容和多场景视频。载于ECCV，2024。网址 https://api.semanticscholar.org/CorpusID:266725702。[30] 马宇航，吴晓诗，孙可强，和李洪生。Hpsv3：迈向广谱人类偏好评分，2025。网址 https://arxiv.org/abs/2508.03789。[31] 孟轶浩，欧阳浩，余悦，王秋宇，王文，郑家梁，王汉林，李煜轩，陈成，曾艳红，沈煜俊，和曲华敏。Holocine：电影化多镜头长视频叙事的整体生成。arXiv预印本 arXiv:2510.20822，2025。[32] OpenAI。视频生成模型作为世界模拟器。https://openai.com/research/video-generation-models-as-world-simulators，2024。[33] OpenAI。GPT-5模型。https://openai.com/gpt-5，2025。[34] OpenAI。Sora2视频模型。https://openai.com/research/sora-2，2025。

[35] William S. Peebles 和 Saining Xie. 基于变换器的可扩展扩散模型. ICCV, 页码 4172-4182, 2023. URL https://api.semanticscholar.org/CorpusID:254854389. [36] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, AmndAskel Pamel Mishki, Jack Clark, Gretche Krueer 和 Sutskever. 从自然语言监督中学习可转移的视觉模型. 2021. URL https://api.semanticscholar.org/CorpusID:231591445. [Tanzila Rahmn, Hsin-Ying Lee, Jan Ren, S. Tulyakov, Shweta Mahajan, 和 Leonid Sigal. 制作故事：视觉记忆条件的一致故事生成. CVPR, 2023. [38] Robin Rombach, A. Blattmann, Dominik Lorenz, Patrick Esser, 和 Björn Ommer. 基于潜在扩散模型的高分辨率图像合成. CVPR, 页码 10674-10685, 2022. URL https://api.semanticscholar.org/CorpusID:245335280. [39] Jani, Yu Lu, She an, Bo We, 和 un LRoo:ho roa 嵌入. ArXiv, abs/2104.09864, 2021. URL https://api.semanticscholar.org/CorpusID:233307138. [40] Yoad Tewel, Omri Kaduri, Rinon Gal, Yoni Kasten, Lior Wolf, Gal Chechik, 和 Yuval Atzmon. 无需训练的一致文本到图像生成. ACM 图形学研究论文 (TOG), 43:1-18, 2024. URL https://api.semanticscholar.org/CorpusID:267412997. [41] Shish Vswani Noa. Shazeer, Niki Parmar, Jakob Uszkort, Lion Joe Aidan, Luk Ki, 和 Illia Polosukhin. 关注即是全部 2017. URL https://api.semanticscholar.org/CorpusID:13756489. [42] Wan 团队, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jiaxio Yang, Jianyuan Zeng, Jiayu Wang, Jingeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xio Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijng Liu, Yimi Wang, Yingya Zhang, Ytong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, 和 Ziyu Liu. Wan：开放且先进的大规模视频生成模型. arXiv 预印本 arXiv:2503.20314, 2025. [43] Mengyu Wang, Henghui Ding, Jianing Peng, Yao Zhao, Yunpeng Chen, 和 Yunchao Wei. Characonsist：细粒度一致角色生成. ICCV, 2025. URL https://api.semanticscholar.org/CorpusID:280296701. [44] Yi Wang, Yinan He, Yizuo Li, Kunchang Li, Jashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, 等. Internvid：用于多模态理解和生成的大规模视频-文本数据集. 在第十二届国际学习表示会议上, 2023. [45] Weija Wu, Zeyu Zhu, 和 Mike Zheng Shou. 通过多智能体协同规划进行自动化电影生成. ArXiv, abs/2503.07314, 2025. URL https://api.semanticscholar.org/CorpusID:276929150. [46] Yaxiong Wu, Sheng Liang, Chen Zhang, Yichao Wang, Yongye Zhang, Huifeng Guo, Ruimig Tang, 和 Yong Liu. 从人类记忆到人工智能记忆：关于大语言模型时代记忆机制的调查. ArXiv, abs/2504.15965, 2025. URL https://api.semanticscholar.org/CorpusID:277993681. [47] Junfei Xiao, Ceyuan Yang, Lvmin Zhang, Shengqu Cai, Yang Zhao, Yuwei Guo, Gordon Wetzstein, Maneesh Agrawala, Alan L. Yuille, 和 Lu Jiang. 电影长官：朝向短电影生成. ArXiv, abs/2507.1834, 2025. URL https://api.semanticscholar.org/CorpusID:280017397. [48] Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, 和 Xingang Pan. Worldmem：具有记忆的长期一致世界模拟. ArXiv, abs/2504.12369, 2025. URL https://api.semanticscholar.org/CorpusID:277857150. [49] Shuai Yang, Yuying Ge, Yang Li, Yukang Chen, Yixiao Ge, Ying Shan, 和 Yingcong Chen. Seed-story：使用大型语言模型的多模态长篇故事生成. ArXiv, abs/2407.08683, 2024. URL https://api.semanticscholar.org/CorpusID:271097376. [50] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Xiaotao Gu, Yuxuan Zhang, Weihan Wang, Yean Cheng, Ting Liu, Bin Xu, Yuxio Dong, 和 Jie Tang. Cogvideox：带有专家变换器的文本到视频扩散模型. ArXiv, abs/2408.06072, 2024. URL https://api.semanticscholar.org/CorpusID:271855655. [51] Jiwen Yu, Jianhong Bai, Yiran in, Quande Liu, Xint Wang, Pengei Wan, Di Zhang, 和 Xihui Liu. 记忆作为背景：具有记忆检索的场景一致交互长视频生成. ArXiv, abs/2506.03141, 2025. URL https://api.semanticscholar.org/CorpusID:279119340. [52] Zeyu Zhang, Quanyu Dai, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Jieming Zhu, Zhenhua Dong, 和 Ji-Rong Wen. 基于大语言模型的智能体记忆机制调查. ACM 信息系统交易, 43:1-47, 2024. URL https://api.semanticscholar.org/CorpusID:269293320. [53] Canyu Zhao, Mingyu Liu, Wen Wang, Jian wei Yuan, Hao Chen, Bo Zhang, 和 Chunhua Shen. Moviedreamer：连贯长视觉序列的分层生成. ArXiv, abs/2407.16655, 2024. URL https://api.semanticscholar.org/CorpusID:271334414. [54] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, 和 Qibin Hou. Storydiffusion：长范围图像和视频生成的一致自注意力. abs/2405.01434, 2024. URL https://api.semanticscholar.org/CorpusID:269502120. [55] Cailn Zhuang, Ailn Huang, Wei Cheng, Jinwei Wu, Yoqi Hu, Jiaqi Liao, Honguan Wang, Xinyo Liao, Weiei Cai, Hengyuan Xu, 等. Vistorybench：故事可视化的综合基准套件. arXiv 预印本 arXiv:2505.24862, 2025. [56] Shaobin Zhuang, Kunchang Li, Xinyuan Chen, Yaohui Wang, Ziwei Liu, Yu Qiao, 和 Yali Wang. Vlogger：实现你的梦成视频博客. CVPR, 页码 8806-8817, 2024. URL https://api.semanticscholar.org/CorpusID:267028492.

# 附录

# 更加定性的结果

在此，我们提供额外的定性结果，以进一步证明StoryMem的有效性。图8提供了一个额外的示例，说明我们的方法在不同视觉风格中的优越性。图9展示了我们方法生成的更多结果，进一步突显其在故事视频生成中的多样性。

# B ST-Bench的更多细节

在这里，我们提供有关 ST-Bench 构建的更多细节。如主论文所述，ST-Bench 由 30 个使用 GPT-5 生成的长篇故事脚本组成，总共生成 300 个视频提示。每个故事遵循统一的 JSON 架构，其中包括短故事概述、1-4 个场景、8-12 个镜头级视频提示、对应的首帧提示（仅针对基于关键帧的两阶段方法），以及剪辑指示，指定某个镜头是否以硬场景转换开始或平滑地从前一个镜头延续。为了确保高质量且易于模型使用的提示，我们设计了一个结构化系统指令，限制每个镜头描述为 1-4 个简明句子，强调角色外观、简单动作、场景布局、情绪和轻量级相机引导，如图 10 所示。最终的基准涵盖了各种风格，从现实主义到童话故事，从古代到现代环境，以及从西方到东方的文化美学，确保不同的生成范式接收标准化和结构良好的输入。图 11 中提供了一个完整的示例故事脚本（不包括首帧提示），以说明 ST-Bench 中使用的格式。所有故事脚本将被发布，以支持该领域未来的研究。

# C 限制与未来工作

我们视觉记忆机制的一大局限性是模糊检索问题。受基础模型Wan2.2的架构设计限制，该模型采用基于跨注意力的DiT，而非更灵活的MMDiT，因此我们的记忆纯粹是视觉的，未包含文本元信息。换句话说，记忆更新函数$m _ { i } = f _ { \phi } ( m _ { i - 1 } , v _ { i } )$（公式8）没有包括标准公式中所需的文本信息$t _ { i }$（公式7）。因此，在复杂的多角色场景中，模型可能无法根据当前的镜头提示正确检索记忆中的上下文，导致角色在不同镜头间出现不一致，如图7顶部所示。一个简单的缓解方法是在每个镜头提示中提供更明确的角色描述，帮助模型匹配预期的记忆，如图7底部所示。未来，我们计划探索更结构化、实体感知的记忆表示，以更根本地解决这个局限性。另一个小局限性在于实现完全平滑的镜头切换。尽管我们的自回归镜头生成过程和MI2V设计显著缓解了以往基于关键帧的方法中的僵硬切换，但单帧连接机制未能传达视频速度信息。因此，当两个相邻的MI2V连接镜头在运动速度上存在显著差异时，切换仍可能显得不自然，如某些提供的视频结果所示。未来的工作中，我们计划通过在连续镜头间重叠更多帧，而不打算进行场景切换，实现更平滑的过渡。

# 猴王的故事

![](images/10.jpg)  
Figure 8 Additional qualitative comparison.

![](images/11.jpg)  
Figure 9 More qualitative results.

抱歉，我无法提供您所要求的信息。

# 领域指令：

故事概述：对整个故事的简洁总结。 • 场景编号：场景的顺序索引。 • 切换：该提示是否以场景切换开始： - "True"：一个新的切换。 "False"：平滑地从上一个提示的最后一帧继续。必须确保两个相邻的提示可以自然地连接成一个流畅的连续片段。 - 故事中的第一个提示必须始终为 "True"。视频提示：构成场景内故事节点的文本到视频提示列表。提示应反映自然、流畅且合乎逻辑的故事进展。

# 每个视频提示应描述的内容（如适用）：

{ "character": { "appearance": "外貌特征", "attire": "服装", "age": "年龄", "style": "风格" }, "actions_and_interactions": { "motion": "动作", "gestures": "手势", "expressions": "表情", "eye_contact": "眼神交流", "simple_physical_actions": "简单的身体动作" }, "scene_and_background": { "location": "室内/室外场所", "props": "道具", "layout": "布局", "lighting": "照明", "environment_details": "环境细节" }, "atmosphere_and_mood": { "emotional_tone": "情感基调", "colors": "颜色", "aesthetic_feeling": "美学感觉" }, "camera_and_editing": { "shot_type": "镜头类型（例如，特写/中景/广角）", "simple_camera_movement": "简单的镜头运动", "transitions": "切换", } }

![](images/12.jpg)  
Figure 11 Example of story script in ST-Bench.