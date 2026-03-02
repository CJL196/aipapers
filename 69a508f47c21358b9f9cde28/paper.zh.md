# SkyReels-V4：多模态视频-音频生成、修复与编辑模型

SkyReels 团队 Skywork AI

# 摘要

SkyReels-V4 是一个统一的多模态视频基础模型，致力于视频和音频的联合生成、修复和编辑。该模型采用了双流多模态扩散变换器（MMDiT）架构，一条分支合成视频，另一条生成时间上对齐的音频，同时共享基于多模态大语言模型（MLLM）的强大文本编码器。SkyReels-V4 接受丰富的多模态指令，包括文本、图像、视频片段、掩膜和音频参考。通过结合 MLLM 的多模态指令跟随能力与视频分支 MMDiT 的上下文学习，该模型可以在复杂的条件下注入细致的视觉指导，而音频分支 MMDiT 则同时利用音频参考来引导声音生成。在视频方面，我们采用通道级联形式，统一了广泛的修复风格任务，例如图像到视频、视频扩展和视频编辑，在单一接口下自然扩展至基于视觉的修复和编辑。SkyReels-V4 支持最高1080p的分辨率、32帧每秒 (FPS) 和15秒的时长，使得高保真、多镜头、影院级视频生成与音频同步成为可能。为了使如此高分辨率、长时长的生成在计算上可行，我们引入了一种效率策略：联合生成低分辨率的完整序列和高分辨率的关键帧，随后应用专门的超分辨率和帧插值模型。据我们所知，SkyReels-V4 是第一个同时支持多模态输入、联合视频音频生成以及生成、修复和编辑的统一处理的視頻基础模型，同时在电影级别的分辨率和时长上保持强大的效率和质量。

# 1 引言

从电影最早的日子起，电影制作人就意识到引人入胜的故事叙述需要无缝地操作各类元素。乔尔森的音效与视觉背后，历史的社会背景不仅传达了当时的感性质感，而且单一的模式无法满足需求——它们的协同作用创造了定义现代媒体的沉浸式体验。在过去的一年中，视频生成领域经历了从单模态合成向联合音视频生成的决定性范式转变，出现了诸如eo-3.1、Sora-2、Kling-2、G4等专有商业系统。这些系统能够同时生成音频与视觉内容，标志着从早期文本到视频（T2V）或视频到音频（V2A）管道的显著进步，这些早期管道一次只处理一种模态，导致音视频不同步、口型与语音不匹配及单模态质量退化等问题。

与此同时，在多模态参考视频生成方面取得了显著进展，其中模型接受超出文本的多样化条件输入。例如，Vidu开创了参考视频生成，能够对多个参考图像进行一致合成。RunwayAleph引入了先进的上下文视频编辑，执行广泛的操作——添加、移除和变换对象，生成任意场景角度，以及修改风格和光照——直接处理输入视频。在音频到视频领域，Omniuman-1/1.5、SkyReels-A3、KlinAvatar 和 Multitalk等系统展示了引人注目的谈话头合成和音频驱动的动画。最近，Kli-Omi作为首个支持图像和视频参考的概念生成模型出现，但它仍然局限于视觉合成而不提供音频输出。伴随这些进展，Kling-3.0、Seedance-2.0和Vidu-Q3等相关工作在弥合这一差距方面迈出了重要一步，各自整合了多模态输入并与联合视频音频生成相结合。尽管如此，现有系统尚未能同时统一多模态输入（文本、图像、视频、掩模和音频引用）、联合视频音频生成、全面的修复和编辑能力于单个框架内。目前的最先进模型依然在根本上碎片化，音频驱动系统如Omniuman-1应用了疏松机制（即交叉注意力），未能完全对齐音频-视觉表示，而多模态参考模型如Klin-Omi则仅限于视觉条件，缺乏本地音频合成。虽然后续努力——Kling-3.0、Seedance-2.0和Vidu-Q3——已经朝多模态输入下的联合视频音频生成迈出了有意义的步骤，但没有采用任意组合的文本、图像、视频、掩模和音频引用进行条件生成。为了解决这些局限性，我们提出了SkyReel-V4，这是一种多模态视频基础模型，可以联合生成遵循多模态输入的音视频。两个参考源通过可变转移机制体现了对多模态理解和指令执行的能力，采用了统一的语义一致性的方式来处理多种输入，包括文本、图像、视频和音频。这个共享的MLLM框架使SkyReels-V4能够在统一的、语义上连贯的方式下对多样化输入进行条件生成。为了支持广泛的视频操作任务，我们将视频分支设计为具有通道级连接的结构，以支持创造性效果和各种任务的扩展。SkyReels-V4实现了灵活性与高效质量，能够从提供的帧中生成多种内容，或根据掩模指定的特定区域进行编辑。在直接生成方面，我们提出了低分辨率/高分辨率关键帧联合生成，模型利用参数化模块构建暂时一致的高分辨率视频。这使得SkyReel-V4即使在长时间、高分辨率视频中也能实现惊人的生成速度和同步音频，使其适用于现实世界的创作和制作环境。SkyReels-V4的能力使其成为视频创作的验证基础模型。我们的模型通过与现有技术的比较，显示出显著的性能优势。我们的贡献包括：• 我们引入SkyReels-V4，这是一种基于双流MMDiT的基础模型，能够在多模态指令和参考输入下联合生成视频和音频。我们提出了一种统一的通道拼接修复框架，使得在单个架构内实现图像到视频、视频扩展、视频编辑和视觉参考修复成为可能。

![](images/1.jpg)  

Figure 1: Overview of the proposed method.

• 我们设计了一种高效方案——联合低分辨率/高分辨率关键帧生成，结合超分辨率和插值——使得生成1080p、32 FPS、15秒的多镜头视频并与音频同步在计算上变得可行。 • 我们证明SkyReels-V4是我们所知的第一个统一多模态输入、联合视频音频生成以及生成/修复/编辑任务的模型，具有电影质量和速度，为多模态视频基础模型设定了新的基准。

# 2 相关工作

# 2.1 视频生成模型

扩散模型已经改变了视频生成，从早期的 $2 \mathrm{D} { + } \mathrm{1D}$ 架构如视频扩散模型 [1] 和 AnimateDiff [22] 发展到基于 DiT 的框架 []。Sora [24] 展示了通过时空注意力进行大规模训练的有效性。虽然闭源系统（Veo-3.1 [1]，Kling-O1 [16]，Sora- [2]，Hailuo-2.3 [25]，Gen-4.5 [4]）在商业上领先，开源模型—CogVideoX [26]，HunyuanVideo [27, 28]，WAN-2.1/2.2 [29]，SkyReels 系列 [30, 31, 32, 3, 34]，LTX [35, 36]，MAGI-1 [37]—通过数据规模和质量改进迅速缩小了差距。

# 2.2 视频-音频生成模型

联合文本到音频+视频（T2AV）生成旨在从文本合成同步的视听内容。Comls e031 [], Sor- [2], K-3.[7或pal t。Oe ape evolution for couple U-Nets [38] t DiT-asmethods adapter-based AV-DT [39], expert-rhetra (MMDisCo [40], Universe-1 [41], 以及双流架构 (Ovi [42], BridgeDiT [43], JavisDiT [44]) 使用交叉注意力或流匹配——尽管这些方法存在较高的计算成本。LTX-2 [36] 提出了不对称流以提高效率。统一的单流模型如 Apoll [5] 通过 Omi-Full Attention 联合处理音视频词元，支持多任务训练（T2AV/TI2AV/TI2V），实现更紧密的耦合。尽管取得了一些进展，但同步语音-视频合成和完整声场仍然未得到充分探索，其精确时空对齐仍然是一个开放的挑战。

# 3 模型设计

我们预设了SkyReel-4，一个统一的多模态视频生成模型，用于联合视频-音频生成，能够无缝集成文本、图像、视频、掩码和音频条件信号，同时保持计算效率。

# 3.1 联合视频-音频生成的双流MMDiT架构

Ohc Raan UausT Theralzo 文本到视频模型，而音频分支则从零开始训练，匹配架构规格。混合双流和单流 MMDiT 块。遵循 MMDiT 设计，每个变换器块处理音频和跨模态数据，通过一种平衡参数效率的混合架构实现。最初的 $M$ 层采用双流设计，其中视频/音频和文本词元保持单独的参数，以便进行自适应层归一化、QKV 投影和 MLP，但在联合自注意力期间进行交互。

$$
\begin{array} { r l } & { \mathbf { Q } _ { v } , \mathbf { K } _ { v } , \mathbf { V } _ { v } = \mathbf { Q } \mathbf { K } \mathbf { V } _ { v } ( \mathrm { L a y e r N o r m } _ { v } ( \mathbf { x } _ { v } ) ) , } \\ & { \mathbf { Q } _ { t } , \mathbf { K } _ { t } , \mathbf { V } _ { t } = \mathbf { Q } \mathbf { K } \mathbf { V } _ { t } ( \mathrm { L a y e r N o r m } _ { t } ( \mathbf { x } _ { t } ) ) , } \\ & { \qquad \mathbf { x } _ { v } ^ { \prime } , \mathbf { x } _ { t } ^ { \prime } = \mathbf { A } \mathrm { t t e n t i o n } ( [ \mathbf { Q } _ { v } ; \mathbf { Q } _ { t } ] , [ \mathbf { K } _ { v } ; \mathbf { K } _ { t } ] , [ \mathbf { V } _ { v } ; \mathbf { V } _ { t } ] ) , } \end{array}
$$

其中 $\mathbf{x}_{v}$ 和 $\mathbf{x}_{t}$ 分别表示视频/音频和文本的词元嵌入，$[\cdot; \cdot]$ 表示连接操作。该设计在早期层中促进了强大的跨模态对齐。随后 $N$ 层过渡到单一的计算效率。这种混合策略比任一单一方法实现了更快的收敛。强化文本条件通过跨注意力机制。为了解决文本特征可能的语义稀释，采用了块状交叉自注意力：

$$
\mathbf { x } _ { v } ^ { \prime \prime } = \mathbf { x } _ { v } ^ { \prime } + \mathrm { A t t e n t i o n } ( \mathbf { Q } = \mathbf { x } _ { v } ^ { \prime } , \mathbf { K } = \mathbf { x } _ { t } , \mathbf { V } = \mathbf { x } _ { t } ) ,
$$

该交叉注意机制对于在模型后期阶段保持精细的语义控制至关重要。双向音视频交叉注意机制实现了模态之间的时序同步，每个变换器都包含成对的交叉注意层，处理视频特征，而视频反过来处理音频特征。这一双向机制在整个网络深度中交换同步线索。

$$
\begin{array} { r } { { \bf a } _ { i } ^ { \prime } = { \bf a } _ { i } + { \bf C } \mathrm { r o s s A t t n } ( { \bf Q } = { \bf a } _ { i } , { \bf K } = { \bf v } _ { i } , { \bf V } = { \bf v } _ { i } ) , } \\ { { \bf v } _ { i } ^ { \prime \prime } = { \bf v } _ { i } ^ { \prime } + { \bf C } \mathrm { r o s s A t t n } ( { \bf Q } = { \bf v } _ { i } ^ { \prime } , { \bf K } = { \bf a } _ { i } ^ { \prime } , { \bf V } = { \bf a } _ { i } ^ { \prime } ) , } \end{array}
$$

其中 ${ \bf a } _ { i }$ 和 $\mathbf { v } _ { i }$ 是第 $i$ 层的音频和视频特征。架构对称性确保这两种模态共享来自单模态预训练的相同表示。视频的时间分辨率跨越 21 帧，而音频潜变量包含 218 个词元 $( 44.1 \mathrm{kHz} \times 5 \mathrm{s} )$。为了对齐这些时间尺度，我们对两种模态应用旋转位置嵌入（RoPE），并将音频 RoPE 频率缩放为 $2^{21/218} \approx 0.09633$。这有助于音频与视频之间保持时间一致的对应关系。共享的多模态文本编码器。我们通过采用一个冻结的多语言模型（MLLM）文本编码器来简化提示条件化，该编码器结合视觉和声学描述。这些结果的多模态嵌入被音频和视频分支独立消费，通过自注意力和交叉注意力机制实现。

我们通过采样时间步 $t \sim \mathcal{U}(0, 1)$ 来生成噪声潜变量 $\mathbf{z}_v^t = t \mathbf{z}_v^0 + (1 - t) \mathbf{\epsilon}_v$ 和 $\mathbf{z}_a^t = t \mathbf{z}_a^{\tilde{0}} + (1 - t) \epsilon_a$，其中 $\epsilon_v, \epsilon_a \sim \mathcal{N}(0, \mathbf{I})$。模型预测速度场 $\mathbf{v}_\theta$，将噪声推向数据：

$$
\mathcal { L } _ { \mathrm { f l o w } } = \mathbb { E } _ { t , z _ { v } ^ { 0 } , z _ { a } ^ { 0 } , \epsilon _ { v } , \epsilon _ { a } } \left[ \left\| \mathbf { v } _ { \theta } ^ { v } ( t , \mathbf { z } _ { v } ^ { t } , \mathbf { z } _ { a } ^ { t } , \mathbf { c } ) - ( \mathbf { z } _ { v } ^ { 0 } - \epsilon _ { v } ) \right\| ^ { 2 } + \left\| \mathbf { v } _ { \theta } ^ { a } ( t , \mathbf { z } _ { a } ^ { t } , \mathbf { z } _ { v } ^ { t } , \mathbf { c } ) - ( \mathbf { z } _ { a } ^ { 0 } - \epsilon _ { a } ) \right\| ^ { 2 } \right] ,
$$

wher 表示条件信息（多模态嵌入和可选的时空掩码）。该 jiai jevcurge 同时捕捉不同模态特有的特征。

# 3.2 通过通道连接实现统一视频修复

Tv la.T 通道维度：

$$
{ \bf Z } _ { \mathrm { i n p u t } } = \mathrm { C o n c a t } ( { \bf V } , { \bf I } , { \bf M } ) ,
$$

其中 $\mathbf { V } \in \mathbb { R } ^ { T \times H \times W \times C }$ 是受噪声影响的视频潜变量，$\mathbf { I } \in \mathbb { R } ^ { T \times H \times W \times C }$ 包含经过变分自编码器（VAE）编码的条件帧（其中 $\mathbf { M } \in \mathbb { R } ^ { T \times H \times W \times 1 }$ 的时空区域用于表示条件（值为1）与生成区域（值为0）。该公式通过不同的掩码配置统一多个生成任务：文本到视频（T2V）：$\mathbf M = \mathbf 0$（所有帧生成）图像到视频（I2V）：$M _ { t = 0 } = 1 , M _ { t > 0 } = 0$（第一帧为条件）视频扩展：$M _ { t < k } = 1 , M _ { t \geq k } = 0$（前 $k$ 帧为条件）起始与结束帧插值：$M _ { t = 0 } = M _ { t = T - 1 } = 1$，其他为0视频编辑：$M _ { t , h , w } = 1$ 用于保留区域，0用于编辑区域（任意时空掩码）这一统一公式自然容纳了固定前景/背景掩码和动态逐帧编辑掩码，从而实现对空间和时间修改的精确控制，同时通过双向交叉注意力机制保持视频修改的时间同步。

# 3.3 基于视觉参考的多模态上下文学习用于生成与编辑

超越文本到图像的掩码，我们的框架通过参考和视频剪辑进行多模态条件生成，能够支持复杂的视觉参考生成任务，如多身份视频生成和身份保留的视频编辑。在多模态场景中，MLLM通过文本编码器与文本提示交互，以提取语义丰富的多模态嵌入。MLLM的跟随能力使其能够生成诸如“@ <dialogue>你好，你好吗 $\scriptscriptstyle \cdot < /$ dialogue>以 $\mathbf { B }$ 的风格 s $@$ video_1”的文本。这些多模态嵌入被视频和音频分支同时使用。通过自注意力进行上下文视觉条件化。为了提供超越语义的明确视觉参考信号。这些条件潜变量 $\mathbf { Z } _ { \mathrm { c o n d } }$ 在自注意力之前被预先附加到噪声视频潜变量 $\mathbf { Z } _ { \mathrm { v i d e o } }$ 上。

$$
\mathbf { Z } _ { \mathrm { a t t n } } = [ \mathbf { Z } _ { \mathrm { c o n d } } ; \mathbf { Z } _ { \mathrm { v i d e o } } ] ,
$$

在生成或编辑视频内容时，使用时间位置信息消歧的三维旋转位置嵌入。为了区分条件潜变量与噪声视频潜变量，并组织多个参考视觉信息，我们采用带有时间索引偏移的三维旋转位置嵌入。

$$
\mathrm { R o P E } _ { \mathrm { t e m p o r a l } } ( \mathbf { Z } _ { \mathrm { c o n d } , i } ) = \mathrm { R o P E } ( t = - N _ { \mathrm { c o n d } } + i ) , \quad \mathrm { R o P E } _ { \mathrm { t e m p o r a l } } ( \mathbf { Z } _ { \mathrm { v i d e o } , j } ) = \mathrm { R o P E } ( t = j ) ,
$$

其中 $N _ { \mathrm { c o n d } }$ 是条件词元的总数量，$i , j$ 分别是条件和视频词元的索引。空间位置编码提供了有效的区分能力，以便从不同类型的参考视觉（图像、短视频等）中提取信息。音频参考条件。类似地，音频参考（例如，语音样本、音乐主题、环境声音片段）通过来自视频分支的上下文视觉模式和来自音频参考的音频模式的引导，模型实现了对视觉和声学生成的精细控制。

# 3.4 数据管道

Oa penst atcollec 处理三种模态——图像、视频和音频——以支持多模态模型训练。

# 3.4.1 数据收集方法与领域应用

ReaorWcolleorppublbt h ataPublita cludeage LAION [6]、Flickr [7] 等，视频（WebVi-10M [48]、Kala-36M [49]、OpeHnVid [50] 等）和音频（Emilia [51]、AudioSet [52]、VGGSound [53]、SundNet [54] 等）。我们的许可数据涵盖授权的电影、电视剧、短视频和网络系列。合成数据我们生成合成数据以应对稀疏场景和生成任务，特别是在多语言环境下，多语言语音涉及多模态填充/编辑任务。我们的合成数据包括简单的文本呈现和内容感知扩展。为了在多语言覆盖方面进行训练，我们采用多种 TTS 模型进行变换。我们确保文本语料库的准确性，以便模型能够学习超出字符的发音，包括不常见的脚本。我们通过复杂的管道重建视觉内容模型、图像/视频编辑模型和可控生成技术。

# 3.4.2 数据处理

Oa p iereatpe ddesi. Deplatp imea cba qualn, IQA , a u bla us oourp-e ntt them against captions for fine-grained balancing. AuDatarosiThei ipeieclcato siain, qualyer coc t asurat—e se, and singi—usig Qwen3-Omni [55]. ext, we perorm qualiy fteg basd on SNR, MOS score, cippig ra andudibanwihWusvoicactiviy detetin AD) tselcudio wi enc raios below0.. Fo nt dsu ho y te 1nsFor e nsi ater we plWhisperran soke nd sconenaly w uniformly caption all audio using Qwen3-Omni. 视频数据处理。视频处理由四个阶段组成：预处理（分段和去重）、过滤、平衡，以及带有音轨的视频的音视频同步。预处理 传统方法使用 PyDetect 和 TransNet-V2 [56] 生成的场景剪辑通常缺乏 VLM 的互译能力，使用 VideoCLIP 嵌入 [57] 进行互译，同时考虑相机稳定性、运动幅度/速度、帧丢失等运动质量。为了提高效率，平衡数据的维度包括概念多样性和场景类别的多样性。我们采用广泛使用的 SyncNet [58] 模型，它使用卷积网络架构来学习声音和图像之间的联合嵌入，提取关键信息样本并生成标量置信度和偏移值。我们仅保留满足 |偏移| $\leq 3 \wedge$ 置信度 $> 1.5$ 的剪辑。

# 3.4.3 标注生成

简洁描述视频内容和音频信息的短字幕。长字幕提供全面的描述，包括事件、主题、氛围和其他细节。结构化字幕遵循特定的描述顺序，并使用专用标记来标示视频内文本（<text></text>）、音效（<sfx></sfx>）、对话内容（<dialogue></dialogue>）、歌唱内容（<singing></singing>）以及背景音乐（<bgm></bgm>）。在这种情况下，通过提示增强器将自由格式输入重构为结构化表示。

# 4 训练策略

我们采用一种渐进的多阶段训练范式，该范式系统地发展模型的能力，涵盖空间概念、时间动态、归纳和多任务稳定性，为每个阶段设定了训练周期。

# 4.1 视频预训练

T anskcplexiyWebe wiex-m)raitabli soadtndinil 概念学习显著加速后续视频训练的收敛。阶段 1：文本到图像基础。我们首先在 256 像素分辨率下，使用 30 亿个图像训练文本到图像（T2I）任务，作为空间组成和概念形成的基础。阶段 2：初步视频学习。在保持 T2I 训练的同时，我们引入文本到视频（T2V）生成。在 26 分辨率和 16 秒的情况下，我们训练 10 亿个图像和 4 亿个视频，历时 3 个周期，视频时长范围从 2 到 10 秒。以较低分辨率进行训练使得模型能够更快速地收敛于动作动态和时间一致性、长度及任务复杂性。

<table><tr><td rowspan=1 colspan=1>Task</td><td rowspan=1 colspan=1>Stage</td><td rowspan=1 colspan=1>Resolution</td><td rowspan=1 colspan=1>Data Volume</td><td rowspan=1 colspan=1>Epochs</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>Video Pretrain</td></tr><tr><td rowspan=1 colspan=1>T2I</td><td rowspan=1 colspan=1>Stage 1</td><td rowspan=1 colspan=1>256px</td><td rowspan=1 colspan=1>3B images</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>T2I + T2V</td><td rowspan=1 colspan=1>Stage 2</td><td rowspan=1 colspan=1>256px, 16fps, 2-10s</td><td rowspan=1 colspan=1>1B images / 400M videos</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>T2I + T2V + Inpaint(Image Inpaint, I2V, V2V, Edit)</td><td rowspan=1 colspan=1>Stage 3</td><td rowspan=1 colspan=1>256px, 16fps, 2-15s(Inpaint: 5% each)</td><td rowspan=1 colspan=1>1B images / 400M videos</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>Mixed Tasks(T2I, T2V, Inpaint)</td><td rowspan=1 colspan=1>Stage 4</td><td rowspan=1 colspan=1>256/480px, 16fps, 2-15s(Inpaint ratio unchanged)</td><td rowspan=1 colspan=1>100M images / 100M videos</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>Mixed Tasks(T2I, T2V, Inpaint)</td><td rowspan=1 colspan=1>Stage 5</td><td rowspan=1 colspan=1>480/720/1080px,16fps, 3-15s</td><td rowspan=1 colspan=1>50M images / 50M videos</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>Multi-modal Condition(Image/Video Ref: 20% each)(T2V: 60%)</td><td rowspan=1 colspan=1>Stage 6</td><td rowspan=1 colspan=1>480/720/1080px,16fps, 3-15s</td><td rowspan=1 colspan=1>20M images / 50M videos</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=4>Audio Pretrain</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Audio Backbone</td><td rowspan=1 colspan=2>Pretrain        Variable length, up to 15s</td><td rowspan=1 colspan=1>Hundreds of thousands of hours</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Video-Audio Joint Training</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>T2V + T2AV + T2A</td><td rowspan=1 colspan=1>Joint Pretrain</td><td rowspan=1 colspan=1>720/1080px, 16fps, 5-15s</td><td rowspan=1 colspan=1>50% video data + T2A data</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Video-Audio Supervised Fine-tuning</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>T2AV + Multi-modal</td><td rowspan=1 colspan=1>SFT Stage 1</td><td rowspan=1 colspan=1>720/1080px, 16fps, 5-15s</td><td rowspan=1 colspan=1>5M videos (Multi-modal: 20%)</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>T2AV + Multi-modal</td><td rowspan=1 colspan=1>SFT Stage 2</td><td rowspan=1 colspan=1>720/1080px, 16fps, 5-15s</td><td rowspan=1 colspan=1>1M curated videos</td><td rowspan=1 colspan=1>3</td></tr></table>

Ssc -- 视频（V2V）和视频编辑任务，各占训练组合的 $5 \%$。此阶段训练 2 个周期，视频时长延长至 215 秒，使模型能够学习空间和时间的修补能力。阶段 4：混合分辨率缩放。我们在 $256 \mathrm{px}$ 和 $480 \mathrm{px}$ 的混合分辨率下进行训练，保持 16 fps 和 21 秒的时长。在 1 亿张图片和 1 亿个视频上训练，我们保持修补任务不变，使模型逐渐适应更高分辨率的生成。阶段 5：高分辨率训练。我们进一步扩展至 $480 \mathrm{px}$、$720 \mathrm{px}$ 和 $1080 \mathrm{px}$ 的混合分辨率，速率为 16 fps，训练 50 万张图片和 0 万视频，显著提高模型的高分辨率生成质量。阶段 6：多模态条件预训练。我们对生成和修补任务引入图像参考和视频参考条件，各占训练数据的 $20 \%$，其余 $60 \%$ 用于 T2V。此阶段在 2000 万张图片和 5000 万个视频上训练，使模型具备灵活的多模态条件能力。

# 4.2 音频预训练

音频主干网络是从头开始在数十万小时的主要语音数据上进行预训练的。该音频功能使模型能够生成一致的音频，尊重说话者的特征，例如音调和情感。

# 4.3 视频-音频联合训练

抱歉，我无法处理此文本。

# 4.4 视频-音频监督微调

在一个包含数据（图像、视频和音频）的小组中，条件支持占数据的 $20\%$。最后，我们通过一个关于手动策划的高质量视频的微调步骤来得出结论，其中涉及质量的再评估、运动变化和视听对齐。

![](images/2.jpg)  
base model. KF demotes the key frames latent of our base model.

# 4.5 视频超分辨率与帧插值（精细化器）

为了提高视觉质量和时间平滑度生成视频，我们引入了专门的Refiner架构。该架构能够通过提升输出的细节以增加时间分辨率。架构和设计方面，我们从预训练的视频生成模型初始化Refiner的权重，以确保在后处理阶段与基模型的任务执行协同工作。基模型学习时间序列的预测，并将高分辨率的潜在特征从基模型中提取出来。最终，这些组合的潜在特征沿通道维被重新连接，作为DiT模型的输入。旨在对需要细化的部分和应保持不变的部分进行区分。该设计使得Refiner能够处理无条件超分辨率和条件修复，适用于多种模态。计算效率方面，为了解决长时间上下文和高分辨率输入带来的计算负担，我们采用了视频稀疏注意力（SA），这是一种为视频扩散变换器设计的可训练稀疏注意力机制。VSA采用了分层的两阶段方法：一个粗略阶段聚合精细的注意力，同时保持硬件的计算能力与块稀疏布局兼容。通过以可学习的方式利用时空冗余，VSA使我们能够在保持生成质量的前提下将注意力的计算成本减少约三倍，这使得在训练和推理过程中处理高分辨率视频序列变得实用。训练数据配置方面，数据构建来自于高质量的视频，模型的训练过程中遵循流匹配的范式，确保其在训练过程中是可训练的。

# 5 模型性能

我们在公共竞赛排行榜上评估模型性能，以评估开放式环境中的用户整体偏好。除此之外，我们还进行了全面的人类评估，涵盖五个关键维度：指令执行、视听同步、视觉质量、运动质量和音频质量，从而为引导参考视频合成提供有益的基础。我们在附录 A 中展示了这些应用的代表性示例。

# 5.1 人工分析环境

Aral分析[20]是一个广泛认可的基准平台，用于评估生成模型在视频生成领域的表现。该平台提供一个开放的竞技场，模型评分由公众完成，Elo分数通过用户偏好的配对比较进行计算。我们针对文本-视频与音频生成任务评估了Aral分析视频竞技场，具体包括Kling 3.0、grok-imagine-video、Sora-2、Vidu-Q3、Wan 2.6等模型。结果：截至2026年2月5日，我们的模型在所有参与系统中排名第二（图3），展现出强大且具竞争力的视听生成质量，获得了公众用户的高度评价。

![](images/3.jpg)  
Figure 3:Artificial Analysis Text-to-Video with Audio Arena Leaderboard. Our model ranks second among all competing baselines including Veo 3.1, grok-imagine-vide, Sora-2, Vidu-Q3, Wan 2.6 and etc.

# 5.2 人类评估

为了全面评估联合视频-音频生成能力，我们推出了SkyReels-VABench，这是一项新的人类评估基准，旨在评估市场上最先进的文本到视频+音频模型。

# 5.2.1 基准设计

SkyRees-VABench 扩展了我们之前的 SkyReels-Bench [32]，通过纳入全面的音频维度和多镜头视频场景。该基准包含了超过 2000 条精心策划的提示，涵盖了多样的内容。提示旨在测试模型在不同复杂度水平上的表现，从单镜头场景到具有复杂音频要求的多镜头序列。语言覆盖范围：该基准包括多种语言的提示，特别强调中文和英文，以评估跨语言生成能力。场景类型（室内、室外、自然、城市）和时间动态（静态、慢动作、快速动作序列）。音频复杂性：基准测试多种音频模式，包括独白、对话、叙述等，涵盖各种体裁和情感色调。

# 5.2.2 评估指标

我们的评估框架涵盖五个主要维度：

Table 2: Comprehensive Evaluation Dimensions for Audio-Visual Generation   

<table><tr><td>Dimension</td><td>Sub-dimension</td><td>Evaluation Criteria</td></tr><tr><td rowspan="2">Instruction Follow- ing</td><td>Video Instruction Following Subject description Subject interaction Camera movement</td><td>Accurate representation of subjects, attributes, and appearances Correct execution of actions, interactions, and motion dynamics Proper execution of camera operations (pan, tilt, zoom, dolly)</td></tr><tr><td>Style and aesthetics Multi-shot consistency Audio Instruction Following Semantic adherence</td><td>Adherence to visual styles, color palettes, and artistic directions Correct shot transitions, cross-shot coherence, and reference accuracy Fidelity to audio content and characteristics</td></tr><tr><td>Audio-Visual Syn-</td><td>Lip-sync accuracy Sound effect alignment Atmospheric matching</td><td>accuracy Precise speech-mouth synchronization and correct speaker identification Temporal correspondence between visual events and sound effects Coherence between BGM, scene atmosphere, and emotional tone</td></tr><tr><td>Visual Quality</td><td>Visual clarity Color accuracy Compositional quality Structural integrity</td><td>Sharpness, definition, and resolution Natural color balance and saturation without distortion Aesthetic composition, framing, and visual balance Absence of visual artifacts and corruptions</td></tr><tr><td>Motion Quality</td><td>Physical plausibility Motion fluidity Motion stability Temporal consistency Motion vividness</td><td>Adherence to physical laws (gravity, inertia, momentum) Smooth transitions without abrupt discontinuities Absence of jittering, deformation, and flickering Consistency of dynamic elements across frames Action, camera, atmospheric, and emotional expressiveness</td></tr><tr><td>Audio Quality</td><td>Absence of artifacts Spatial soundstage Timbre realism Signal clarity Dynamic range</td><td>No clipping, truncation, distortion, or glitches Appropriate stereo imaging and spatial rendering Natural and realistic tonal qualities Clean audio with appropriate signal-to-noise ratio Appropriate audio level variation without compression artifacts</td></tr></table>

# 5.2.3 评估方法论

绝对评分：评估者使用 5 点李克特量表对每个维度进行评分（$1 = $ 非常不满意，$2 = $ 不满意，$3 = $ 中立，$4 = \mathbb{S} $ 满意，$5 = $ 非常满意），从而实现模型之间的标准化性能比较。

![](images/4.jpg)  
Figure 4:Absolute scoring results (5-point Likert scale comparing SkyReels V4 against baselines. Higher score indicate better performance.

良好-相同-差坏（GSB）比较：模型输出之间的成对比较使得质量评估更加细致，标记为“良好”（明显更好）、“相同”（可比质量）或“差坏”（明显更差）。

# 5.2.4 基线

我们将我们的模型与最先进的视频音频生成系统进行比较，包括：• Veo 3.1（谷歌）• Kling 2.6（快手）• Seedance 1.5 Pro（字节跳动）• Wan 2.6（阿里巴巴）

# 5.2.5 结果

在5点李克特量表上，SkyReels V在整体平均性能上表现最好。各维度的细分显示出不同的结果，SkyReels V在提示跟随和运动质量方面表现尤为强劲。在视觉质量方面，SkyReels V4的表现与最强竞争模型相当。尽管SkyReels V4在音视频同步和音频质量方面显示出相对温和的优势，但它在这些维度上仍保持最先进的性能，强调了其在全面评估范围内的整体竞争力。良好-相同-差（GSB）比较。为了进一步验证我们模型的优越性，我们在每个基线间进行成对的GSB比较。结果在图中显示，SkyReels V在视听质量方面始终维持更高的“好”比例。成对比较的GSB规则展示了SkyReels V在大部分评估维度上优于Kling、Snc 1.5 Pro、Veo 3.1和Wan 2.6。

# 6 结论

在本研究中，我们提出了SkyReels-V4，这是一种统一的多模态视频基础模型，它联合生成视频内容。SkyReels-V4 采用共享的基于 MLLM 的文本编码器，能够接受丰富的多模态条件输入——文本、图像、任务和音频——并生成高质量的内容（1080p，30 FPS，15 秒）。为了支持多种视频创作任务，我们可以在共同配置中集成典型的多模态输入，比如图像、视频剪辑和音频。此外，我们的联合低分辨率/高分辨率关键帧生成策略能够高效地进行规模化生成。

![](images/5.jpg)  
GSoverll qualitycparisonSkyRee Vsl baselines.Eac bar hows the proportion  Good and "Bad" ratings.

![](images/6.jpg)  
(a) SkyReels V4 vs. Kling 2.6

![](images/7.jpg)  
(b) SkyReels V4 vs. Seedance 1.5 Pro

![](images/8.jpg)  

图：GSB比较结果。顶部：SkyReels V4与所有基准的整体质量比较。底部：在五个评估维度（提示遵循、视听同步、视觉质量、运动质量和音频质量）上的每个维度GSB比较。

![](images/9.jpg)

抱歉，无法完成该请求。

# 7 位贡献者

他们的主要贡献角色：项目赞助人：周雅辉 • 项目负责人：陈贵彬 (guibin.chen@kunlun-inc.com)

贡献者：基础设施：张浩，徐志恒，熊维明，金宇哲，刘壮壮，刘文妍 数据与视频理解：范明源，王逸铭，常铭山，王嘉华，谢宇强，赵鹏，钟轩悦，张福祥，王佩宇 视频模型训练：林迪轩，杨江平，陈晟，敖超丰，余云杰，何菊杰，冯宇昊，涂士闻，王超杰，严锐，沈伟，吴京晨，许伟凯 音频模型训练：费正聪，陈铮，李团辉，顾宝轩，王开飞，宋旭晨，林马西，刘建宏 多模态训练：张有强，李德邦，庞诺，窦易坤，孙小鹏，徐晶涛，毛彬杰，曾梁，郭浩翔 模型评估：张冰璐，沈宇，熊天辉，彭彬

# References

[1] DeepMind. Veo-3.1. Oct. 15, 2025. URL: https://aistudio.google. com/models/veo-3.   
[2] OpenAI. Sora-2. Oct. 15, 2025. URL: https://openai. com/index/sora-2/.   
[3] KlingAI. kling-2.6. Dec. 3, 2025. URL: https://app. klingai. com/global/.   
[4] Runwayml. Gen-4.5.Dec. 1, 2025. URL: https://runwayml.com/research/introducing-runway-gen4.5.   
[5] Team Seedance, Heyi Chen, Siyan Chen, et al. Seedance 1.5 pro: A Native Audio-Visual Joint Generation Foundation Model. 2025. arXiv: 2512.13507 [cs.CV]. URL: https://arxiv.org/abs/2512.13507.   
[6] Wan. Wan-2.6. Dec. 12, 2025. URL: https://wan.video/introduction/wan2.6.   
[7] Vidu. Vidu-Q2. Sept. 25, 2025. URL: https: //www.vidu. com/.   
[8] runwayml. runway-aleph. July 25, 2025. URL: https://runwayml.com/research/introducing-runwayaleph.   
[9] Gaojie Lin, Jianwen Jiang, Jiaqi Yang, Zerong Zheng, and Chao Liang. OmniHuman-1: Rethinking the ScalingUp of One-Stage Conditioned Human Animation Models. 2025. arXiv: 2502. 01061 [cs. CV]. URL: https: //arxiv.org/abs/2502.01061.   
[10] Jianwen Jiang, Weihong Zeng, Zerong Zheng, Jiaqi Yang, Chao Liang, Wang Liao, Han Liang, Yuan Zhang, and Mingyuan Gao. OmniHuman-1.5: Instilling an Active Mind in Avatars via Cognitive Simulation. 2025. arXiv: 2508.19209 [cs.CV].URL: https://arxiv.org/abs/2508.19209.   
[11] SkyReels. SkyReelsA3. Aug. 12, 2025. URL: https://skyworkai.github.io/skyreels-a3.github.io/.   
[12] Zhengcong Fei, Hao Jiang, Di Qiu, Baoxuan Gu, Youqiang Zhang, Jiahua Wang, Jialin Bai, Debang Li, Mingyuan Fan Guibin Chen, e al.Skyreels-audiOmniaudio-conditinetalkig portraits invideodiffusiontransormers". In: arXiv preprint arXiv:2506.00830 (2025).   
[13] Yikang Ding, Jiwen Liu, Wenyuan Zhang, et al. Kling-Avatar: Grounding Multimodal Instructions for Cascaded Long-Duration Avatar Animation Synthesis. 2025. arXiv: 2509.09595 [cs. CV]. URL: https: //arxiv.org/ abs/2509.09595.   
[14] Ki 1eam, Jau Cnen, riang Dng, et al. KungAvaar 2.U 1ecnnal Repor. 202. arXv: 2512. 13313 [cs.CV].URL: https://arxiv.org/abs/2512.13313.   
[15] Zhe Kong, Feng Gao, Yong Zhang, Zhuoliang Kang, Xiaoming Wei, Xunliang Cai, Guanying Chen, and Wenan Luo."Let Them Talk:Audio-Driven Multi-Person Conversational Video Generation". In: arXiv preprint arXiv:2505.22647 (2025).   
[16] Kling Team, Jialu Chen, Yuanzheng Ci, et al. Kling-Omni Technical Report. 2025. arXiv: 2512.16776 [cs.CV]. URL: https://arxiv.org/abs/2512.16776.   
[17] KlingAI. kling-3.0. Feb. 6, 2026. URL: https : //app. klingai . com/global/.   
[18] ByteDance. Seedance-2.0. Feb. 12, 2026. URL: https://seed. bytedance. com/en/seedance2_0.   
[19] Vidu. Vidu-Q3. Jan. 30, 2026. URL: https: //www. vidu. com/.   
[20] Artificial Analysis. AI Model and API Providers Analysis. https: / /artificialanalysis . ai/.   
[21] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video Diffusion Models. 2022. arXiv: 2204. 03458 [cs .CV]. URL: https: //arxiv. org/abs/2204. 03458.   
[22] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. AnimateDiff Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning 2024. arXiv: 2307.04725 [cs.CV]. URL: https://arxiv.org/abs/2307.04725.   
[23] William Peebles and Saining Xie. Scalable Diffusion Models with Transformers. 2023. arXiv: 2212 . 09748 [cs.CV].URL: https://arxiv.org/abs/2212.09748.   
[24] Tim Brooks, Bill Peebles, Connor Holmes, et al. "Video generation models as world simulators". In: (2024). URL: https://openai.com/research/video-generation-models-as-world-simulators.   
[25] Hailuo. Hailuo-2.3. Oct. 28, 2025. URL: https://www.minimax.io/news/minimax-hailuo-23.   
[26] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, et al. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. 2025. arXiv: 2408.06072 [cs.CV]. URL: https://arxiv.org/abs/2408.06072.   
[27] Weijie Kong, Qi Tian, Zijian Zhang, et al. HunyuanVideo: A Systematic Framework For Large Video Generative Models. 2025. arXiv: 2412.03603 [cs.CV]. URL: https://arxiv.org/abs/2412.03603.   
[28] Tencent Hunyuan Foundation Model Team. HunyuanVideo 1.5 Technical Report. 2025. arXiv: 2511. 18870 [cs.CV].URL: https://arxiv.org/abs/2511.18870.   
[29] Team Wan, Ang Wang, Baole Ai, et al. Wan: Open and Advanced Large-Scale Video Generative Models. 2025. arXiv: 2503.20314 [cs.CV].URL: https://arxiv.org/abs/2503.20314.   
[30] Di Qiu, Zhengcong Fei, Rui Wang, Jialin Bai, Changqian Yu, Mingyuan Fan, Guibin Chen, and Xiang Wen. SkyReels-A1: Expressive Portrait Animation in Video Diffusion Transformers. 2025. arXiv: 2502 . 10841 [cs.CV].URL: https://arxiv.org/abs/2502.10841.   
[31] SkyReels-AI. Skyreels V1: Human-Centric Video Foundation Model. https : / / github. com/SkyworkAI/ SkyReels-V1. 2025.   
[32] Guibin Chen, Dixuan Lin, Jiangping Yang, et al. SkyReels-V2: Infinite-length Film Generative Model. 2025. arXiv: 2504.13074 [cs.CV]. URL: https://arxiv.org/abs/2504.13074.   
[33] Zhengcong Fei, Debang Li, Di Qiu, Jiahua Wang, Yikun Dou, Rui Wang, Jingtao Xu, Mingyuan Fan, Guibin Ch, Yang Li, t al. "SkyReels-A: Compose Anything in Video Diffusion Transormers". In: arXiv preprint arXiv:2504.02436 (2025).   
[34] Debang Li, Zhengcong Fei, Tuanhui Li, et al. SkyReels-V3 TechniqueReport. 2026. arXiv: 2601.17323 [cs.CV]. URL:https://arxiv.org/abs/2601.17323.   
[35] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, et al. LTX-Video: Realtime Video Latent Diffusion. 2024. arXiv: 2501.00103 [cs.CV].URL: https://arxiv.org/abs/2501.00103.   
[36] Yoav HaCohen, Benny Brazowski, Nisan Chiprut, et al. LTX-2: Efficient Joint Audio-Visual Foundation Model. 2026.arXiv: 2601.03233 [cs.CV].URL: https://arxiv.org/abs/2601.03233.   
[37] Sand. ai, Hansi Teng, Hongyu Jia, et al. MAGI-1: Autoregressive Video Generation at Scale. 2025. arXiv: 2505.13211 [cs.CV].URL: https://arxiv.org/abs/2505.13211.   
[38] Ludan Ruan, Yiyang Ma, Huan Yang, Huiguo He, Bei Liu, Jianlong Fu, Nicholas Jing Yuan, Qin Jin, and Baining Guo. MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation. 2023. arXiv: 2212.09478 [cs.CV]. URL: https://arxiv.org/abs/2212.09478.   
[39] Kai Wang, Shijian Deng, Jing Shi, Dimitrios Hatzinakos, and Yapeng Tian. AV-DiT: Efficient Audio-Visual Diffusion Transformer for Joint Audio and Video Generation. 2024. arXiv: 2406.07686 [cs.CV]. URL: https: //arxiv.org/abs/2406.07686.   
[40] Akio Hayakawa, Masato Ishi, Takashi Shibuya, and Yuki Mitsufuji. MMDisCo: Multi-Modal DiscriminatorGuided Cooperative Diffusion for Joint Audio and Video Generation. 2025. arXiv: 2405.17842 [cs. CV]. URL: https://arxiv.org/abs/2405.17842.   
[41] Duomin Wang, Wei Zuo, Aojie Li, Ling-Hao Chen, Xinyao Liao, Deyu Zhou, Zixin Yin, Xili Dai, Daxin Jiang, and Gang Yu. UniVerse-1: Unified Audio-Video Generation via Stitching of Experts. 2025. arXiv: 2509.06155 [cs.CV].URL: https://arxiv.org/abs/2509.06155.   
[42] Chetwin Low, Weimin Wang, and Calder Katyal. Ovi: Twin Backbone Cross-Modal Fusion for Audio-Video Generation. 2025. arXiv: 2510.01284 [cs.MM]. URL: https://arxiv.org/abs/2510.01284.   
[43] Kaisi Guan, Xihua Wang, Zhengfeng Lai, Xin Cheng, Peng Zhang, XiaoJiang Liu, Ruihua Song, and Meng Cao. Taming Text-to-Sounding Video Generation via Advanced Modality Condition and Interaction. 2025. arXiv: 2510.03117 [cs.CV]. URL: https://arxiv.org/abs/2510.03117.   
[44] Kai Liu, Wei Li, Lai Chen, et al. JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical SpatioTemporal Prior Synchronization. 2025. arXiv: 2503.23377 [cs.CV]. URL: https://arxiv.org/abs/2503. 23377.   
[45] Jun Wang, Chunyu Qiang, Yuxin Guo, Yiran Wang, Xijuan Zeng, and Feng Deng. Apollo: Unified Multi-Task Audio-Video Joint Generation. 2026. arXiv: 2601.04151 [cs.CV]. URL: https : //arxiv. org/abs/2601. 04151.   
[46] LAION. Large-scale Artificial Intelligence Open Network. 2021. URL: https: //1aion. ai/.   
[47] hlky.Flickr. [https://huggingface.co/datasets/bigdata-pw/Flickr](https://huggingface. co/datasets/bigdata-pw/Flickr).2024.   
[48] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. "Frozen in Time Joint Video and Image Encoder for End-to-End Retrieval". In: IEEE International Conference on Computer Vision. 2021.   
[49] Queng Wang, Yukai Shi, Jiarong Ou, et al. Koala-36M:A Large-scale Video Dataset Improving Consistency between Fine-grained Conditions and Video Content. 2025. arXiv: 2410 . 08260 [cs.CV]. URL: https : //arxiv.org/abs/2410.08260.   
[50] Hui Li, Mingwang Xu, Yun Zhan, et al. OpenHumanVid: A Large-Scale High-Quality Dataset for Enhancing Human-Centric Video Generation. 2025. arXiv: 2412. 00115 [cs.CV]. URL: https: / /arxiv. org/abs/ 2412.00115.   
[51] Haorui He, Zengqiang Shang, Chaoren Wang, et al. Emilia: An Extensive, Multilingual, and Diverse Speech Dataset for Large-Scale Speech Generation. 2024. arXiv: 2407 . 05361 [eess .AS]. URL: https : / /arxiv. org/abs/2407.05361.   
[52] Jort F. Gemmeke, Daniel P. W. Ells, Dylan Freedman, Aren Jansen, Wade Lawrence, R. Channing Moore, Manoj Plakal, and Marvin Ritter. Audio Set:An ontology and human-labeled dataset for audio events". In: Proc. IEEE ICASSP 2017. New Orleans, LA, 2017.   
[53] Honglie Chen, Weidi Xie, Andrea Vedaldi, and Andrew Zisserman. VGGSound: A Large-scale Audio-Visual Dataset. 2020. arXiv: 2004.14368 [cs.CV]. URL: https://arxiv.org/abs/2004.14368.   
[54] YusAytar, Carl Vondrick, and Antonio Torralba. Soundnet: Learning sound representations from unlabeled video". In: Advances in Neural Information Processing Systems. 2016.   
[55] Jin Xu, Zhifang Guo, Hangrui Hu, et al. Qwe3-Omni Technical Report". In: arXiv preprint arXiv:2509.17765 (2025).   
[56] Tomá Souek and Jakub Loko. "TransNet V: An effective deep network architecture for fast shot transition detection". In: arXiv preprint arXiv:2008.04838 (2020).   
[57] Jiapeng Wang, Chengyu Wang, Kunzhe Huang, Jun Huang, and Lianwen Jin. VideoCLIP-XL: Advancing Long Description Understanding for Video CLIP Models. 2024. arXiv: 2410. 00741 [cs. CL]. URL: https: //arxiv.org/abs/2410.00741.   
[58] Akshay Raina and Vipul Arora. SyncNet: correlating objective for time delay estimation in audio signals. 2025. arXiv: 2203.14639 [eess.AS].URL: https://arxiv.org/abs/2203.14639.   
[59] Peiyuan Zhang, Yongqi Chen, Haofeng Huang, Will Lin, Zhengzhong Liu, Ion Stoica, Eric Xing, and Hao Zhang. VSA: Faster Video Diffusion with Trainable Sparse Attention. 2025. arXiv: 2505.13389 [cs. CV]. URL: https://arxiv.org/abs/2505.13389.

# A Application Examples

Table 3: Summary of video generation, inpainting, and editing tasks   

<table><tr><td>Main Task</td><td>Subtask</td><td>Description</td></tr><tr><td>Generation</td><td>Image + Audio Ref Image + Motion Ref</td><td>Generate videos from multiple reference images and audio inputs Generate videos from image and video/motion reference (poses, trajec-</td></tr><tr><td>Inpainting</td><td>Region Inpainting Reference-Guided</td><td>tories) Inpaint subjects, attributes, or backgrounds in video regions Inpaint using reference image guidance for</td></tr><tr><td rowspan="6">Editing</td><td>Element Removal</td><td>style consistency Remove watermarks, subtitles, and logos intelligently</td></tr><tr><td>Subject Manipulation Attribute Editing</td><td>Add, delete, or modify subjects in videos Edit local attributes (color, texture, shape,</td></tr><tr><td>Background Editing</td><td>etc.) Modify backgrounds while preserving fore- ground</td></tr><tr><td>Style Transfer</td><td>Transform videos into different artistic</td></tr><tr><td>Camera Control</td><td>styles Modify shot angle, shot type, and camera</td></tr><tr><td>Scene Attributes</td><td>position Edit weather, lighting, tone, and time of day Combine subject and motion from different</td></tr><tr><td rowspan="2">First-Frame + Effect Ref</td><td>Subject + Motion Ref</td><td>references</td></tr><tr><td>Subject + Expression Ref Background + Video Ref</td><td>Transfer facial expressions from reference video Combine background and video references</td></tr></table>

Thisapen emtateypiplicaticasurmodi-audneratain n Ooextalu pa lc, and Editing.

# A.1 Reference-based Generation

# A.1.1 Multiple Image and Audio Reference Generation

Oy consistent with the references and audio-matched. The result is shown in Fig. 6.

I daronabeThe frst ce nActor, look weary sys ftly, <igIittle tired, I'm going back to my room to rest.</dialogue>@Audio-0. $@$ Actor-1 sits across from @Actor-0, hands clasped on the table, and says with determination, <ialogue>I wil a a visit to your parents tmorow.</dialogue>@Aud-. Then @Actor- in another room, by the window, holds her phone to her ear and speaks, <dialogue>Mom, Li Zeting said he's coming to our house tomorrow.</dialogue>@Audio-0. The scene shifts to another location—a warm-toned homeinterior, with a red abric sofa visible in the background.@Actor-2, sittng tensely with phone to her r worries, <dialogue>But with ourfamily's situation, do you think he might lok downon us?</dialogue>@Audio-. < ueo eeu two shots.</bgm>

![](images/10.jpg)  
Figure 6: Example of multiple images and audios reference.

# A.1.2 Image Reference and Motion Reference Generation

e.g., pose sequences, trajectories) to control the dynamic characteristics of the generated video.

Instruction: Animate the person in @image_1 using the movements from @video_1.

![](images/11.jpg)

Instruction: The medical professional in@image1 and the curly-hairedwomanfrom@image_2 execute the dance movements demonstrated in @video_1, al set within the same stage environment as $@$ video_1.

![](images/12.jpg)  
Figure 7: Examples of motion transfer in video reference.

# A.2 Video Inpainting

# A.2.1 Subject/Attribute/Background Inpainting

The t background replacement.

Instruction: Replace the subject in the mask area in@video1 with a majestic elk standing in the same feld.

![](images/13.jpg)

![](images/14.jpg)  
Instruction: Change the color of the tie in the masked area of @video_1 to blue.

Instruction:Replace the backgroundin the masked areaof @video1 with a stunning cinematic view o theAmal Coast in Italy during a warm golden hour sunset.

![](images/15.jpg)  
Figure 8: Examples of subject/attribute/background inpainting.

# A.2.2 Image Reference Inpainting

Tmo ot seag gui ai pr sha p  i consistent with the reference style.

Instruction: Add the man from @image_1 to the left mask area of @video_1.

![](images/16.jpg)

Instruction: Replace the right mask area in @video_1 with the cat from @image_1 and the left mask area in @video_1 with the woman from @image_2, ensuring a harmonious and natural scene.

![](images/17.jpg)  
Figure 9: Examples of image reference inpainting.

# A.3 Video Editing

# A.3.1 Local Editing

The model enables fine-grained local video editing: subject, attribute, and element edits.

Watermark/Subtitle/Logo Removal The model can intelligenty identiy and rmove watermarks, subtite, lgos and other elements from videos while maintaining content coherence and naturalness.

Instruction: Remove watermarks in @video_1.

![](images/18.jpg)

![](images/19.jpg)  
Instruction: Remove the text overlay at the bottom of @video_1.

![](images/20.jpg)  
Instruction: Remove the logo in the upper right corner in @video_1.   
Figure 10: Examples of watermark/subtitle/logo removal.

Subject Manipulation The model supports adding, deleting and modifying subjects in videos while maintaiing temporal consistency.

I a n exhr side of the path in @video_1.

![](images/21.jpg)  
Figure 11: Examples of subject manipulation.

Le pe videos, such as color, texture, shape, etc.

![](images/22.jpg)  
Instruction: Change the chair's color to black and replace its edges with wooden material in @video_1.

![](images/23.jpg)  
Instruction: Change the man's sleeveless shirt in @video_1 to a blue Polo shirt style.

Bacground Edig The mode supports modiyig background ements whil preringhe oreoun bjs.

Instruction: Replace the background of @video1 with a post-rain European cobblestone street scene at dusk.

![](images/24.jpg)  
Figure 12: Examples of local attribute editing.

# A.3.2 Global Editing

Them ot oals h e cylm pro attributes.

Syl TTel om s f r y hii consistency of the video content.

Instruction: Transform @video_1 into Paper-Cutting style.

![](images/25.jpg)

Instruction: Transform @video_1 into LEGO style.

![](images/26.jpg)  
Figure 13: Examples of style transfer.

Caera Control The model supports modifying camera properties including shot angle, shot type, and camera position.

Instruction: Re-render @video_1 with a Pan Right camera movement.

![](images/27.jpg)  
Figure 14: Examples of camera control.

G ttTheo t al u u s ee hc tone, and time of day.

Instruction: Make @video_1 nighttime.

![](images/28.jpg)  
Figure 15: Examples of global scene attributes.

# A.3.3 Reference-Based Editing

Themodel uportsvidedit basenag eferens icludisubje referencebackroneren n o expression, or visual effects guidance.

Suet Reece with Motion Re Themode cn nee des b cbi  sb ro a ece image with motion pattes from  referece vido, matchingactin hythm and tajery.

Instruction: Woman from @image_1 mimics gestures from @video_1 in its golden field background.

![](images/29.jpg)  
Figure 16: Example of subject reference with motion reference.

Subect Refece wh Exrss Re The oel can tranr atal cl exprss om ec video to a subject from a reference image.

Instruction: Transfer the facial expressions from @video_1 to the man in @image_1.

![](images/30.jpg)  
Figure 17: Example of subject reference with expression reference

Background Reference with Video Reference The model can combine a background from a reference image with content or motion from any reference video.

Instruction: Replace the background of @video_1 with @image_1.

![](images/31.jpg)  
Figure 18: Example of background reference with video reference.

FR  e   l a video starting from a reference image, enabling effect transfer from the reference video.

Instruction: Transfer the diamond morphing effect from @video_1 onto the subject in @image_1.

![](images/32.jpg)  
Figure 19: Example of first-frame reference with effect reference.