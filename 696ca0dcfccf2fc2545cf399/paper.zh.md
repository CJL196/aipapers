# 利用结构化多视频协同推理增强视频大语言模型

何志豪 $^ { 1 , 2 }$ , 何天耀 $^ { 1 }$ , 徐云 $^ { 1 }$ , 陈铁源 $^ { 1 }$ , 刘华滨 $^ { 1 }$ , 甘超凡 $^ { 1 }$ , 吴祖轩 $^ { 2 , 3 }$ , 林伟尧 $^ { 1 }$ * 上海交通大学。 $^ 2$ 上海创新研究院。 $^ 3$ 复旦大学。 *通讯作者。电子邮箱：wylin@sjtu.edu.cn； 贡献作者：ziho_he@sjtu.edu.cn；

# 摘要

尽管视频语言模型蓬勃发展，但当前对于全面视频推理的追求受到个别视频固有的时空不完整性阻碍，导致幻觉和不准确性。一个有前景的解决方案是通过多段相关视频增强推理性能。然而，视频令牌数量众多，且包含冗余信息，因此直接将相关视频数据输入大型语言模型以增强响应可能适得其反。为解决这一挑战，我们提出了一种视频语言模型的多视频协作框架。为了高效灵活地表示视频，我们建立了一个视频结构模块，将视频知识表示为时空图。基于结构化的视频表示，我们设计了图融合模块，将结构化知识和来自相关视频的有价值信息融合到增强的图节点令牌中。最后，我们构建了一个复杂的多视频结构化提示，将图、视觉和文本令牌整合为大型语言模型的输入。大量实验验证了我们框架的有效性，展示了其作为推进视频语言模型的有前景的途径的潜力。代码已开源于 https://github.com/ziHoHe/SMV-CR。关键词：结构化多视频、协作推理、视频结构模块、图融合模块、结构化提示工程。

# 1 引言

大型语言模型（LLMs）的显著进展促进了视频语言模型（VLMs）的繁荣。凭借LLMs的强大通用知识，视频语言模型展现出理解复杂视频内容的令人印象深刻的潜力。这些模型有效地弥合了视频与语言之间的语义鸿沟，促进了视觉感知与语言处理的细致融合。然而，实现可靠和全面的视频推理仍然是一个难以捉摸的目标。具体而言，从单个视频中捕获的时空知识往往不完整，原因在于有限的视觉感知和理解能力。妨碍视觉感知的因素包括稀疏采样，空间方面则存在来自单个视频的不完整和冗余信息。这些挑战阻止了视频语言模型抓取有价值的知识，并导致在回答给定问题时出现幻觉和失败。引入多个在同一领域高度相关的视频可以补偿缺失的信息，使视频语言模型能够归纳并提供可靠的答案。

![](images/1.jpg)  
Fig. 1: Video question-answering pipeline under different video collaboration strategies. (a) Single-video reasoning pipeline; (b) Direct multi-video collaboration pipeline: concatenate multiple video's visual tokens, which is burdensome; (c) Structured multi-video collaboration pipeline (ours).

尽管存在明显的优势，但来自多个视频的增量信息并不一定能在实际案例中带来更好的推理。如图1(b)所示，直接的多视频联合策略是在传递给视频语言模型之前，将多个视频的视觉特征连接在提示中，这可能会导致视频语言模型面临大量的词元[46]。在处理长上下文时，实证证据[20, 21]证明，大型语言模型倾向于仅关注输入的特定和有限片段，从而忽视可能至关重要的信息。此外，视频内容的高维性、冗余性和模糊性与语言的结构化、清晰和规则性特征形成鲜明对比。这种时间视觉复杂性使得多视频知识的综合和提炼特别具有挑战性。遮挡[12, 13]和视角切换[14-17]等因素使得理解能力主要受视频内容复杂性[18, 19]和有限有效上下文长度[20, 21]的限制。由于视频语言模型大多源自大型语言模型[22-24]，并借助视觉指令微调[25]，视频语言模型在面对知识不完整时只能依赖语言先验[26, 27]。这种倾向带来了虚假的幻觉和与视觉事实的不一致[28, 29]。主流的视频语言模型遵循单视频推理流程（如图1(a)所示）。受到多数据协作研究[30-32]和检索增强生成研究[33-35]的启发，解决这一限制的一个有希望的想法是整合来自多个相关视频的知识。我们在图2中展示了一些视频问答的例子。正如这些例子所示，单视频推理在本文中，我们提出了一种结构化的多视频协同推理框架。如图1(c)所示，我们的框架在图结构表示层面上实现了多视频信息的对齐和完整性。为了应对视频信息的冗余性和复杂性，我们首先设计了视频结构模块（VSM）。该模块逐步分析关键目标并提取其时空关系，从而构建高效的数据结构化视频表示。在获得结构化表示后，我们提出了图融合模块（GFM）以生成适合大型语言模型的图词元。GFM通过图注意力网络（GAT）[36]将结构信息整合到节点特征中，然后基于交叉图注意力（CGA）融合多视频知识。最后，我们精心设计提示，排列融合的图词元、目标视频词元和文本词元，以增强视频语言模型的多视频结构知识。总体而言，我们的贡献有三个方面：

![](images/2.jpg)  
Fig. 2: Video question answering examples from video language models. Single-video reasoning (Left): In video-1, the environmental visual cues are hard for the model to perceive, leading to a 'sports team'hallucination based on only the textual query and linguistic priors. In Video 2, ice-related visuals are missing, and limited bartending knowledge causes the model to skip the question. Multi-video reasoning (Right): Introducing relevant videos allows the model to complete and summarize domain-specific knowledge (such as environmental protection or bartending in this case), leading to more reliable and accurate answers.

我们探讨了一个可行的结构化推理框架，以实现视频语言模型下的多视频协作。在这个框架中，我们开发了视频结构化模块，以获取视频信息的数据高效时空图。然后，我们进一步设计了图融合模块，以将时空关系和跨视频知识融合到图词元中。大量实验表明，我们的框架通过多视频结构化和协作，可以增强可靠性和准确性。

# 2 相关工作

# 2.1 视觉语言模型

在视觉理解领域，伴随着大语言模型的发展，取得了显著进展。通过特征对齐和视觉指令微调，视觉-语言模型学习视觉和语言的联合表示，利用大规模训练大型语言模型所带来的通用能力，从而在开放世界环境中理解图像和视频。其中，Video-LLaMA和BLIP-2采用Q-former提取有价值的视觉信息，将其转换为紧凑且适合大型语言模型的视觉词元进行对齐。相反，Flamingo和BLIP-3利用可扩展的感知重采样器获取可学习的视觉词元。LLaVA、Video-LLaVA、LLaVA-OneVision、Qwen2/2.5-VL使用基于MLP的投影器将视觉和文本输入映射到共享特征空间。此外，由于图像和视频输入中包含复杂的信息，许多研究旨在减少视觉词元的数量，以减轻大型语言模型在计算和理解上的负担。虽然当前的方法重点关注单视频推理，但我们尝试探索一个结构化的视频推理框架，有效应对多视频输入带来的冗余和整合挑战。

# 2.2 多数据协作

虽然大多数深度学习任务遵循单一数据处理管道，但多数据协作提供了一种有前景的方向，可以通过多个样本之间的内在对应关系来提高性能。目前的研究大致可分为两类，即内容相关协作和任务相关协作。内容相关协作通过对多个相关数据的比较来帮助模型关注关键内容。例如，共同分割任务探讨如何通过总结和共享相同对象相关特征在不同场景中分割相同对象。少样本图像分类、动作识别和细粒度分类方法则通过多数据比较关注关键差异，以实现准确分类。检索增强生成也是一种有前景的内容相关协作，它向大型语言模型提供来自检索的相关数据的支持信息。在任务相关协作中，模型通过观察多个样本执行相同任务的方式来学习如何完成任务。例如，多视频摘要研究旨在通过多数据互补和精炼，从一组视频中生成摘要。上下文学习利用带有任务指导和回答的例子，向大型语言模型展示如何完成任务。

![](images/3.jpg)  
Fig. 3: Multi-video collaborative reasoning framework. Together with the target video, $N$ related videos are retrieved to facilitate the reasoning process. First, we design the Video Structuring Module to obtain the structured video representation. Then, the Graph Fusion Module fuses the structure information and the related videos' information to get the video graph tokens. Finally, according to the designed prompts, the graph tokens, visual tokens, and text tokens are arranged as input to the large language model for question answering.

对于视频语言模型，目前的多视频协作策略是直接连接多个输入，导致冗余负担过重和协作困难。在这些进展的基础上，我们提出了多视频推理任务，通过多视频协作实现关键时空信息的补偿和精炼。

# 3 方法

# 3.1 概述

# 3.1.1 多视频推理的设置

在本文中，我们探讨了利用多视频信息进行视频理解的潜力。具体而言，我们采用多视频设置，其中目标视频 $V _ { 0 }$ 伴随着 $N$ 个相关视频 $\{ V _ { 1 } , V _ { 2 } , \ldots , V _ { N } \}$。为了检索相关视频，预先构建它们的特征向量以实现高效的视频检索。第 4.4 节讨论了不同的视频向量化方法。最后，借助 $N$ 个检索到的视频，方法需要回答有关目标视频的问题。

# 3.1.2 框架概述

我们的多视频协作推理框架如图 3 所示。首先，在第 3.2 节介绍视频结构化模块，以获得结构化视频表示。基于获得的视频结构，我们在第 3.3 节设计图融合模块，以融合结构化视频表示并将有用的相关视频信息转换为结果图标记。最后，我们根据设计的多视频推理提示安排所有图标记、视觉标记和文本标记，并将它们发送到大型语言模型。

# 3.2 视频结构模块

高效的结构化视频表示为后续多视频知识的整合铺平了道路。给定视频及其配对的密集字幕，我们的 video structuring module (VSM) 的过程如图 3 右下方所示，具体步骤如下。步骤 1：场景检测。为减少视频中的时间冗余，我们采用轻量级的基于内容的场景检测器 Autoshot [55]，将视频分割为不同的场景。从每个检测到的场景中，我们提取其中间帧作为关键帧，后续的视频结构化过程将以此为输入。将视频 $V _ { N }$ 的 $M$ 个关键帧记作 $\mathcal { F } _ { N } = \{ F _ { 1 } , F _ { 2 } , \ldots , F _ { M } \}$。步骤 2：密集视频字幕生成。为了为后续的结构化流程做准备，我们需要提取详细且细粒度的文本概念。为此，我们利用视频大语言模型生成输入视频的全面描述，使用的提示设计详见图 4。步骤 3：文本场景图解析。然后，我们使用 SceneGraphParser [56] 从密集视频字幕中提取文本场景图 $\mathcal G ^ { \mathrm { T e x t } }$，将其大语言模型替换为 Qwen3-30-A3B [57]。文本场景图由若干个三元组组成 $\tau _ { i } = \{ s _ { i } , p _ { i } , o _ { i } \}$，每个三元组 $\tau _ { i }$ 代表视频中的第 $i$ 次交互或事件。这里，$s _ { i }$、$p _ { i }$ 和 $o _ { i }$ 分别表示主语、谓语和宾语。每个三元组的格式为主语 - 谓语 - 宾语，作为视频中捕获的关系和动态的基础表示。

步骤 4：图信息过滤。为提高数据质量，我们应用主动过滤机制，剔除无关或冗余的三元组。具体而言，我们使用图像级分类器来验证场景中三元组相关对象或主体的存在。这是通过制定简单的二分类任务，结合定制的提示，并利用 SigLIP [58] 进行分类来实现的。例如，我们设计提示，如“与 {对象/主体} 相关的物体在图像中。”作为正样本，以及“与 {对象/主体} 相关的物体不在图像中。”作为负样本。根据分类结果，我们确定是否保留或丢弃对象-主体对。如果图像中存在独立的对象或主体，我们将构建格式为 $\{ s _ { i } , * , s _ { i } \}$ 或 $\{ o _ { i } , * , o _ { i } \}$ 的三元组，以在下一步骤中建立节点之间的自连接。相反，如果对象和主体没有同时出现，则相应的三元组将被丢弃。这个过程生成过滤后的三元组，记作 $\hat { \mathcal { G } } ^ { \mathrm { T e x t } }$ 。

步骤5：视频图谱建立。在经过筛选的文本场景图 $\hat { \mathcal { G } } ^ { \mathrm { T e x t } }$ 和对应每个三元组的关键帧 $\mathcal { F } _ { \{ 0 , \ldots , N \} }$ 的基础上，我们为目标视频及相关视频建立基于图结构的视频表示。具体而言，该图由节点、帧内边和帧间边组成。节点表示视频中物体或主体的特征。对于节点级特征，我们首先利用 Qwen3-Embedding [59] 从 $\hat { \mathcal { G } } ^ { \mathrm { T e x t } }$ 中提取文本特征 $\mathbf { T }$，将文本转换为每个物体和主体的特征表示，然后引入池化注意力机制，根据文本特征和关键帧 $\mathcal { F } _ { \{ 0 , . . . , N \} }$ 提取有效的视觉特征，并与原始文本特征进行自适应加权融合，以获得更稳健的节点级特征表示。有关此过程的进一步细节将在第3.3节中提供。帧内边由每个三元组的谓词表示，表示同一帧内物体与物体之间的空间和交互关系。同时，帧间边链接跨不同帧的共享相同主体和物体的物体，从而建模它们的时间关系。通过上述步骤，我们可以为目标视频及其相关视频建立基于图结构的视频表示，以便进行进一步的协作。

![](images/4.jpg)  
Fig. 4: Video captioning prompts. We refer to the design outlined in [54] to create the prompts used to extract captions from videos. The prompts are divided into two parts: the system prompt and the user message. In the system prompt, we define the task of video captioning and provide corresponding guidelines along with a standardized output format. For the output format, the program randomly selects contents in green font as the normalized format for reference during each process of captioning. For the user message, we utilize $< V I D E O _ { - } T O K E N S >$ as the video tokens, and we provide a concise instruction to the model, then generate a detailed description for the video.

# 3.3 图融合模块

图融合模块（GFM）由三元嵌入模块（TEM）和用于图信息处理的多层堆叠架构组成。该架构的每一层都集成了两个基本组件：层次框架图注意网络（HFGAT）和交叉图注意（CGA）机制。首先，在TEM中，我们引入类别嵌入（CE）来增强GFM区分目标视频图与相关视频图的能力。CE定义如下：

$$
\begin{array} { r l } & { \mathrm { C E } _ { t a r } = \sigma ( \pmb { \alpha } ) , } \\ & { \mathrm { C E } _ { r e l } = 1 - \sigma ( \pmb { \alpha } ) , } \end{array}
$$

其中 $\pmb { \alpha } \in \mathbb { R } ^ { d }$ 表示跨帧共享的可学习参数，$\sigma$ 表示 sigmoid 函数。计算得到的类别嵌入随后被直接应用于 GFM 的输入，对于目标视频中的文本特征 $\mathbf { T } _ { t a r }$ 使用 $\mathrm { C E } _ { t a r }$，对于来自相关视频的其他文本特征 ${ \bf T } _ { r e l }$ 使用 $\mathrm { C E } _ { r e l }$。该集成过程的公式如下：

$$
\begin{array} { r } { \mathbf { T } _ { t a r } = \mathbf { T } _ { t a r } + \mathbf { C } \mathbf { E } _ { t a r } , \quad } \\ { \mathbf { T } _ { r e l } = \mathbf { T } _ { r e l } + \mathbf { C } \mathbf { E } _ { r e l } , \quad } \\ { \mathbf { T } = [ \mathbf { T } _ { t a r } , \mathbf { T } _ { r e l } ] , \quad \quad } \end{array}
$$

![](images/5.jpg)  
Fig. 5: Structured multi-video prompts. We properly integrate the multi-modal tokens, together with the prompt guidance, to form an LLM-friendly input.

其中 [,] 表示对目标视频和相关视频中的三元组进行串联操作。通过利用交叉熵，GFM 可以隐式学习区分目标视频和相关视频。此外，为了有效整合与三元组对应的关键帧的视觉信息，我们将在 TEM 中集成如公式 (3) 所定义的池化注意力，从而能够直接聚合由文本特征引导的关键帧的视觉特征。

$$
\begin{array} { r l } & { \mathbf { Q } = \mathbf { T } \mathbf { W } _ { Q } \in \mathbb { R } ^ { 1 \times d } , } \\ & { \mathbf { K } = \mathbf { I } \mathbf { W } _ { K } \in \mathbb { R } ^ { ( H _ { p } \times W _ { p } ) \times d } , } \\ & { \mathbf { V } = \mathbf { I } \mathbf { W } _ { V } \in \mathbb { R } ^ { ( H _ { p } \times W _ { p } ) \times d } , } \\ & { \tilde { \mathbf { I } } = \mathrm { s o f t m a x } ( \mathbf { Q } \mathbf { K } ^ { \top } / \sqrt { d } ) \mathbf { V } \in \mathbb { R } ^ { 1 \times d } , } \end{array}
$$

其中 $\mathbf { W } _ { Q } \ \in \ \mathbb { R } ^ { d \times d } , \mathbf { W } _ { K } \ \in \ \mathbb { R } ^ { d \times d } , \mathbf { W } _ { V } \ \in \ \mathbb { R } ^ { d \times d }$ 是与注意力机制的查询 $\mathbf { Q }$、键 $\mathbf { K }$ 和值 $\mathbf { V }$ 相关的可学习权重矩阵 [60]，而 $\textbf { I } \in \ \mathbb { R } ^ { ( H _ { p } \times W _ { p } ) \times d }$ 表示从VLM的视觉编码器提取的视觉特征，该特征利用多个词元来表示关键帧。$( H _ { p } \times W _ { p } )$ 表示从视觉编码器提取后的视觉特征的长度。通过应用池化注意力，我们以文本特征为指导聚合视觉特征，从而获得更强健的特征表示。随后，我们使用自适应权重 $\beta \in \mathbb { R } ^ { d }$ 融合从Qwen3-Embedding [59] 提取的原始文本特征 $\mathbf { T }$ 和池化后的视觉特征 $\tilde { \mathbf { I } }$，定义如下：

$$
\hat { \mathbf { T } } = \sigma ( \beta ) \odot \mathbf { T } + ( 1 - \sigma ( \beta ) ) \odot \tilde { \mathbf { I } } ,
$$

这里 $\odot$ 表示哈达玛乘积（逐元素相乘）。该融合操作自适应地平衡文本和视觉特征的贡献，生成最终的稳健表示 $\hat { \mathbf { T } }$ 。

然后，我们将处理好的三元组特征 $\hat { \mathbf { T } }$ 输入到多层架构中以处理图形信息。特征首先传递到 HF-GAT，它专门用于融合单个视频中的图结构数据。传统的图注意力网络（GATs）[36] 主要用于节点分类和关系预测等任务，其中节点之间的关系是明确定义的。相比之下，对于帧间和帧内上下文，这些关系通常是隐式或缺失的。为了解决这个挑战，如第 3.2 节所述，我们首先将原始视觉模态数据转换为基于图的结构化表示，使用视觉结构模型（VSM）。在构建的图中，节点代表主体或物体的特征，特征最初是从第 3.2 节提取的，随后由 TEM 处理。对于帧内边，我们利用基于三元组的关系，其中从 $s _ { i }$ 到 $o _ { i }$ 形成一个有向链接。对于帧间边，我们利用步骤中过滤后的结果 $\hat { \mathcal { G } } ^ { \mathrm { T e x t } }$（见第 3 节），链接 $s _ { i } ^ { t - 1 }$、$o _ { i } ^ { t - 1 }$、$s _ { i } ^ { t }$ 和 $o _ { i } ^ { t }$ 的当前帧。此外，我们为帧间连接引入了双向链接，因为这增强了系统理解视频内容的能力 [61]。

一旦通过 HF-GAT 提取了单个视频的结构化特征，下一步是识别并融合视频之间最相关的信息。为此，我们引入了跨图注意力机制，通过带有自定义位置 ID 的自注意力机制进行实现。为了实现基于三元组特征的多视频协同推理，我们确定了三个关键原则：1）在单个三元组中，主体与客体之间的关系是不可交换的。2）在单个视频内，三元组之间的位置关系是无序的。3）在通过相关性排序检索的多个视频之间，三元组的顺序是不可交换的。对于原则 1）和 2），HF-GAT 提供的结构化视频表示本质上捕捉了单个视频内的位置编码，因为 HF-GAT 基于相应视频的图连接关系聚合并传递信息，从而在表示中隐式编码了位置信息。因此，我们的主要关注点是解决原则 3），同时确保遵循原则 1）和 2）。为了处理原则 3），我们在每个视频内分配一致的位置 ID，并根据检索相关性动态调整跨检索视频的位置 ID。例如，目标视频的三元组特征始终分配位置 ID 为 0，而相关视频的位置 ID 则根据其检索相关性动态确定。这些位置 ID 随后通过 RoPE [62] 整合，以有效地在跨图注意力机制中编码位置信息。此外，我们还为 HF-GAT 和 CGA 组件应用残差连接和预先归一化。具体而言，我们为预先归一化应用 LayerNorm [63]，该方法常用于视觉变换器（ViT）[58]。值得注意的是，我们在层中排除了前馈网络（FFN），以保持来自视觉编码器的对齐视觉特征的不变性，最小化过度特征移动，并确保 GFM [64] 输入和输出之间的线性关系。然后，各视频的基于图的结构化视频表示通过 GFM 处理，以构建图令牌，每个令牌对应于与结构化信息融合后的主体或客体的节点特征。

# 3.4 结构化多视频提示

在获得融合的多视频图令牌后，我们需要将图、视频和文本令牌整合在一起，以创建适合大语言模型的输入。因此，我们提出了结构化的多视频提示，如图5所示。我们的提示源自之前视频语言模型的提示设计[6]。对于目标视频，我们保持目标视频的 视频令牌 $< V I D E O _ { - } T O K E N S >$，以保留细粒度和详细的视觉信息。同时，我们还附加其基于图的结构化数据 $< G R A P H _ { - } T O K E N S >$，以指示关键对象和时空关系。对于 $N$ 个相关视频，我们仅保留简洁且数据高效的基于图的结构化数据 $< G R A P H _ { - } T O K E N S >$。在这一背景下，我们进一步指明目标视频与相关视频之间的关系，以及大语言模型如何利用这些相关的多视频结构化数据。通过以这种方式构建提示，我们使视频语言模型能够有效地利用多视频信息，从而增强模型对视频内容相关查询的推理能力和回答能力。

# 4 实验

# 4.1 实验设置

数据集描述。在训练阶段，我们基于 LLaVA-Video-178K 数据集构建了一个包含结构化视频信息的 GFM 训练数据集 [54]。具体而言，我们按照第 3.2 节中的说明对数据集进行了逐步预处理。值得注意的是，对于 LLaVA-Video-178K 中已经包含字幕的视频数据，我们保留原始字幕而不做进一步修改，以简化预处理工作流。关于相关视频的视频向量化和检索机制，我们利用 Qwen3-Embedding-8B [59] 从视频字幕中提取查询嵌入（用于检索）和文档嵌入（用于存储）。这一方法在检索任务中被广泛使用 [59]。具体而言，文档嵌入是通过直接输入字幕生成的，而查询嵌入则通过设计的提示生成，以准备嵌入模型的输入，如下所示：指示：这是视频的字幕。请提供一个搜索查询，以检索其他相关视频的字幕表示。 \n 查询：{caption}。最后，我们构建了一个包含大约 87K 样本的训练数据集。尽管与用于训练其他大语言模型的数据集相比（例如，87K 对比 9.36M [40]）相对较小，但我们的方法实现了有效的性能提升，如表 2 所示。这个结果凸显了我们的方法能够无缝整合到现有模型框架中，并通过对紧凑数据集的简单高效训练来提升性能。在评估方面，我们在不同的视频问答基准上测试我们的方法，包括 ActivityNet-QA [65]、NExT-QA [66]、EgoSchema [67] 和 Video-MME [68]。这些基准涵盖了短视频和长视频理解任务，为我们的方法提供了全面的评估。此外，为了提高消融研究的实验效率，我们在训练中使用了大约 $10\%$ 原始训练数据集。同样，在评估过程中，我们选择了 NExT-QA 和 EgoSchema 数据集的子集，每个子集包含大约 0.5K 样本。

实施细节。我们提出的结构化多视频协作框架适应于一般的视频语言模型。为验证我们所提方法的有效性，我们在 A6000 48GB GPU 上对不同参数规模的各种模型进行了实验，包括 LLaVA-OneVision-0.5B [40] 和 LLaVA-Video-7B [54]。对于图融合模块（GFM），隐藏状态大小配置与相应视觉编码器的输出维度相匹配。我们使用预训练权重初始化视频语言模型（VLM），并通过在我们构建的数据集上训练，进一步增强其有效理解基于图的视频表示的能力。为了高效优化我们的模型，我们采用标准的两阶段训练策略 [6, 64]。在第一阶段，我们冻结视觉编码器、投影器和语言模型，专注于训练 GFM 以对齐输入到语言模型。在第二阶段，我们解冻投影器和语言模型，同时保持视觉编码器冻结，并对语言模型应用 LoRA [69]，然后同时微调投影器、GFM 和语言模型。详细的训练方案和超参数配置见表 1。

Table 1: The training recipe for VLM in our experiments.   

<table><tr><td></td><td>Stage-1</td><td>Stage-2</td></tr><tr><td>Trainable</td><td>GFM</td><td>GFM, Projector, LLM</td></tr><tr><td>Batch size</td><td>128</td><td>64</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Warmup ratio</td><td>0.03</td><td>0.03</td></tr><tr><td>Learning rate schedule</td><td>Cosine decay</td><td>Cosine decay</td></tr><tr><td>LR: φgFM</td><td>1e-3</td><td>1e-4</td></tr><tr><td>LR: φProj.</td><td>-</td><td>1e-5</td></tr><tr><td>LR: φLLM</td><td>-</td><td>1e-5</td></tr></table>

# 4.2 视频问答

我们在视频问答任务上评估了先进的视频语言模型，包括 ActivityNet-QA、NExT-QA、EgoSchema 和 Video-MME，这些任务共同覆盖了多种视频理解任务。我们遵循 Video-LLaVA 提出的 методологии，使用 ChatGPT-Assistant 报告 ActivityNet-QA 数据集的开放式回答准确度。然而，由于在 ActivityNet-QA 原始评估流程中使用的 gpt-3.5-turbo-0613 模型已被弃用，为了公平比较，我们选择使用开源的大语言模型 Qwen3-235B-A22B 对结果进行重新评估。作为开源模型，它比 GPT 系列更容易获取和使用。此外，Qwen3-235B-A22B 相较于 gpt-3.5-turbo 系列展现出更强的语言能力，使其成为评估开放式回答正确性的理想替代方案。因此，为了确保基线的公平和一致的比较，我们使用 Qwen3-235B-A22B 对 LLaVA-OneVision-0.5B 和 LLaVA-Video-7B 的 ActivityNet-QA 数据集进行重新评估。我们进一步比较了几种先进的视频语言模型的性能，它们均在单个视频的基础上进行视频问答。与这些传统方法不同，我们的框架通过生成基于目标视频和多个检索到的相关视频的答案，扩展了推理能力。实验结果如表 2 所示，突显了我们的方法相较于基线 LLaVA-OneVision-0.5B 和 LLaVA-Video-7B 的优越性。通过引入多视频协作推理的概念，我们的方法提高了在多种任务中的平均准确率，包括开放式回答、多项选择问答和涵盖不同时长视频的视频理解任务。这些结果表明，我们的方法在紧凑数据集上高效训练，整合了多视频知识，并提供了更可靠的答案。

Table 2: Video question answering performances on different large video-language models The wavy lines indicate the re-evaluated results.   

<table><tr><td rowspan=1 colspan=2>ModelTaskDuration</td><td rowspan=1 colspan=1>Params</td><td rowspan=1 colspan=1>Frames</td><td rowspan=1 colspan=1>ActivityNet-QAOpen-EndedShort</td><td rowspan=1 colspan=1>NExT-QAMulti-ChoiceShort</td><td rowspan=1 colspan=1>EgoSchemaMulti-ChoiceLong</td><td rowspan=1 colspan=1>Video-MMEMulti-ChoiceLong</td><td rowspan=1 colspan=1>AverageAcc. (%)</td></tr><tr><td rowspan=2 colspan=2>Video-LLaVA [6]LLaMA-VID [43]</td><td rowspan=1 colspan=1>7B</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>45.30</td><td rowspan=1 colspan=1>62.60</td><td rowspan=1 colspan=1>38.40</td><td rowspan=1 colspan=1>40.40</td><td rowspan=1 colspan=1>46.68</td></tr><tr><td rowspan=1 colspan=1>7B</td><td rowspan=1 colspan=1>1fps</td><td rowspan=1 colspan=1>47.40</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>38.50</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=4 colspan=2>PLLaVA [70]VideoChat2 [71]LLaVA-NeXT-Video [72]Qwen2-VL [41]</td><td rowspan=1 colspan=1>7B</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>56.30</td><td rowspan=1 colspan=1>68.17</td><td rowspan=1 colspan=1>45.16</td><td rowspan=1 colspan=1>44.25</td><td rowspan=1 colspan=1>53.47</td></tr><tr><td rowspan=1 colspan=1>7B</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>54.40</td><td rowspan=1 colspan=1>47.90</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=1 colspan=1>7B</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>53.50</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>43.90</td><td rowspan=1 colspan=1>46.50</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=3 colspan=2>Qwen2-VL [41]Qwen2.5-VL [42]Qwen2.5-VL [42]</td><td rowspan=1 colspan=1>7B</td><td rowspan=1 colspan=1>2fps</td><td rowspan=1 colspan=1>57.40</td><td rowspan=1 colspan=1>77.20</td><td rowspan=1 colspan=1>66.70</td><td rowspan=1 colspan=1>63.30</td><td rowspan=1 colspan=1>66.15</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3B</td><td rowspan=1 colspan=1>2fps</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>64.80</td><td rowspan=1 colspan=1>61.50</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=4 colspan=2>Qwen2.5-VL [42]VideoLLaMA2 [44]VideoLLaMA2.1 [44]VideoLLaMA3 [73]</td><td rowspan=1 colspan=1>7B</td><td rowspan=1 colspan=1>2fps</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>65.00</td><td rowspan=1 colspan=1>65.10</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=1 colspan=1>7B</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>50.20</td><td rowspan=1 colspan=1>75.60</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>47.90</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=1 colspan=1>7B</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>53.00</td><td rowspan=1 colspan=1>75.60</td><td rowspan=1 colspan=1>53.10</td><td rowspan=1 colspan=1>54.90</td><td rowspan=1 colspan=1>59.15</td></tr><tr><td rowspan=1 colspan=1>2B</td><td rowspan=1 colspan=1>180</td><td rowspan=1 colspan=1>58.20</td><td rowspan=1 colspan=1>81.10</td><td rowspan=1 colspan=1>58.50</td><td rowspan=1 colspan=1>59.60</td><td rowspan=1 colspan=1>64.35</td></tr><tr><td rowspan=3 colspan=2>InternVL2 [74]InternVL2.5 [74]NVILA [45]</td><td rowspan=1 colspan=1>8B</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>55.00</td><td rowspan=1 colspan=1>54.00</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=1 colspan=1>8B</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>58.90</td><td rowspan=1 colspan=1>85.00</td><td rowspan=1 colspan=1>51.50</td><td rowspan=1 colspan=1>64.20</td><td rowspan=1 colspan=1>64.90</td></tr><tr><td rowspan=1 colspan=1>8B</td><td rowspan=1 colspan=1>256</td><td rowspan=1 colspan=1>60.90</td><td rowspan=1 colspan=1>82.20</td><td rowspan=1 colspan=1>54.30</td><td rowspan=1 colspan=1>64.20</td><td rowspan=1 colspan=1>65.40</td></tr><tr><td rowspan=2 colspan=2>LLaVA-OneVision[40]+Ours</td><td rowspan=2 colspan=1>0.5B0.5B</td><td rowspan=2 colspan=1>3232</td><td rowspan=2 colspan=1>45.6546.46</td><td rowspan=1 colspan=1>57.20</td><td rowspan=1 colspan=1>26.80</td><td rowspan=1 colspan=1>44.00</td><td rowspan=1 colspan=1>43.41</td></tr><tr><td rowspan=1 colspan=1>58.71</td><td rowspan=1 colspan=1>28.38</td><td rowspan=1 colspan=1>43.74</td><td rowspan=1 colspan=1>44.32</td></tr><tr><td rowspan=1 colspan=2>LLaVA-Video [54]+Ours</td><td rowspan=1 colspan=1>7B7B</td><td rowspan=1 colspan=1>6464</td><td rowspan=1 colspan=1>60.5561.25</td><td rowspan=1 colspan=1>83.2084.00</td><td rowspan=1 colspan=1>557.3061.76</td><td rowspan=1 colspan=1>63.3064.37</td><td rowspan=1 colspan=1>66.0967.84</td></tr></table>

# 4.3 消融研究

对所提组件的消融研究。我们的框架由两个组件组成：视频结构化和多视频协作。实验结果分别在表 3 中展示了 LLaVA-OneVision-0.5B [40] 的结果，在表 4 中展示了 LLaVA-Video-7B [54] 的结果。首先，我们通过禁用视频结构化模块并测试常见的多视频融合策略来评估我们的框架。"多视频词元"策略涉及将所有视频词元连接作为上下文的输入，而"多视频字幕"策略则将所有视频的字幕输入上下文。最初，我们使用模型中关于相关视频的默认帧数，但这在推理过程中导致了内存溢出（OOM）问题。这凸显了直接利用"多视频词元"方法在实际推理场景中的不实用性。我们通过将相关视频的帧数减少到 8，而保持目标视频的默认帧数来解决此问题。尽管进行了这一调整，"多视频词元"策略仍然向上下文中引入了过多的词元，挑战了模型的理解能力，并导致明显的性能下降（LLaVA-OneVision-0.5B下降了 $-9.6\%$，LLaVA-Video-7B下降了 $7.2\%$）。相反，"多视频字幕"方法显示出轻微的性能提升（LLaVA-OneVision-0.5B 增加了 +0.4%），但 LLaVA-Video-7B 的性能保持不变。启用视频结构化模块提供了结构化和更清晰的视频表示，有助于大型语言模型（LLMs）实现更好的内容理解。这导致所有模型的一致性性能提升（LLaVA-OneVision-0.5B 增加了 +0.6%，LLaVA-Video-7B 增加了 $+3.79\%$）。最后，应用多视频图融合策略使得框架能够有效提取相关视频中的有价值信息。这实现了显著的性能提升（LLaVA-OneVision-0.5B 增加了 $+3.8\%$，LLaVA-Video-7B 增加了 $+4.4\%$），并且仅需额外 0.2K 的词元，相较于单视频处理的开销非常小，同时利用视频结构化模块提升推理能力。

Table 3: Ablation study on video structuring and multi-video fusion components on NExT-QA, conducted using the baseline model LLaVA-OneVision-0.5B [40].   

<table><tr><td>Struct</td><td>Multi-video</td><td>context L</td><td>NExT-QA</td></tr><tr><td rowspan="4"></td><td>single video</td><td>6.5K</td><td>61.4</td></tr><tr><td>multi-video tokens (32)</td><td>38K</td><td>OOM</td></tr><tr><td>multi-video tokens (8)</td><td>15K</td><td>51.8</td></tr><tr><td>multi-video captions</td><td>9.3K</td><td>61.8</td></tr><tr><td>✓</td><td>single video</td><td>7.3K</td><td>62.0</td></tr><tr><td></td><td>graph fusion module</td><td>7.5K</td><td>65.2</td></tr></table>

Table 4: Ablation study on video structuring and multi-video fusion components on NExT-QA, conducted using the baseline model LLaVA-Video-7B [54].   

<table><tr><td>Struct</td><td>Multi-video</td><td>context L</td><td>NExT-QA</td></tr><tr><td></td><td>single video multi-video tokens (64) multi-video tokens (8) multi-video captions</td><td>13K 73K 22K</td><td>79.8 OOM 72.6</td></tr><tr><td>✓</td><td>single video</td><td>16K 13.8K</td><td>79.8 83.6</td></tr><tr><td></td><td>graph fusion module</td><td>14K</td><td>84.2</td></tr></table>

Table 5: Ablation on the design of GFM. PA refers to Pooling Attention.   

<table><tr><td>HF-GAT</td><td>PA</td><td>CGA</td><td>FFN</td><td>NExT-QA</td><td>EgoSchema</td></tr><tr><td></td><td></td><td></td><td></td><td>61.4</td><td>26.4</td></tr><tr><td></td><td></td><td></td><td></td><td>64.2</td><td>28.0</td></tr><tr><td>·</td><td>✓</td><td></td><td></td><td>64.4</td><td>28.2</td></tr><tr><td>✓</td><td>√</td><td>✓</td><td></td><td>65.0</td><td>28.6</td></tr><tr><td>✓</td><td>√</td><td>V</td><td>V</td><td>64.4</td><td>27.6</td></tr></table>

Table 6: Ablations on different video retrieval strategies.   

<table><tr><td rowspan=1 colspan=1>Video Retrieval Strategy</td><td rowspan=1 colspan=1>NExT-QA</td><td rowspan=1 colspan=1>EgoSchema</td></tr><tr><td rowspan=3 colspan=1>video vector-based retrievalrestricted retrievalcaption vector-based retrieval</td><td rowspan=1 colspan=1>63.8</td><td rowspan=1 colspan=1>27.6</td></tr><tr><td rowspan=1 colspan=1>63.6</td><td rowspan=1 colspan=1>27.6</td></tr><tr><td rowspan=1 colspan=1>65.0</td><td rowspan=1 colspan=1>28.6</td></tr></table>

图融合模块设计的消融研究。我们进一步对图融合模块的设计进行消融研究，该模块包括三个组成部分：HF-GAT、CGA 和 TEM 中的池化注意力。为了加快训练和评估，我们使用 LLaVA-OneVision0.5B 模型，利用原始训练和评估数据集的子集来验证所提组件的有效性。由于我们在第 3.3 节中讨论了 FFN 的使用，因此我们也将其纳入考虑，并对这三个组件进行消融研究。消融实验的结果如表 5 所示。在第一行中，图结构特征词元被直接发送到多模态投影层以获取图词元。在第二行中，我们结合 HF-GAT 以将结构信息传播到每个词元，这使得 NExT-QA 数据集的性能提升了 $2.8\%$，EgoSchema 数据集的性能提升了 $1.6\%$。我们还将池化注意力整合到 TEM 中，以嵌入来自特定场景的结构信息。结果证明，融入这一丰富信息相比原始文本结构信息略有改善（NExT-QA 数据集提高 $+3\%$，EgoSchema 数据集提高 $+1.8\%$）。此外，通过引入 CGA，多视频知识被融入图词元，进一步在 NExT-QA 数据集上提升了 3.6% 的性能，EgoSchema 数据集上提升了 $2.2\%$。有趣的是，采用

![](images/6.jpg)  
Fig. 6: Comparative analysis of accuracy $( \% )$ and context length (K) for NExT-QA across different models under varying numbers of related videos.

FFN 并未提高推理准确性，进一步支持了第 3.3 节得出的结论。

# 4.4 对检索到的视频内容的讨论

多个视频如何影响性能？多视频数据有助于实现更全面的推理。在这一部分，我们讨论相关视频数量的影响。如图6所示，当检索视频数量从1增加到8时，准确率最初上升，并在5个视频时达到峰值，然后逐渐下降。重要的是，这一准确度趋势伴随着总词元数量的仅微小增加。视频相关性如何影响性能？在上述实验中，我们为每次迭代安排与目标视频最相关的视频。在这一部分，我们重新排列具有不同较低相似度（以检索特征之间的余弦相似度进行测量）的视频，以评估视频相关性如何影响性能。正如图7所示，随着视频相关性的降低，推理性能下降，但性能仍与基线相当。视频检索策略如何影响性能？直观地，不同的检索策略是我们协同推理方法的潜在影响因素。因此，为了进行更详细的讨论，我们对三种视频检索策略进行了消融研究：• 基于视频向量的检索。该策略适用于大多数情况。它使用SigLIP视觉编码器生成从采样帧中提取的类别词元特征，然后计算平均特征作为每个视频的特征向量，并构建视频向量数据集。在推理期间，基于特征向量之间的最高余弦相似度检索$N$个相关视频。 • 基于字幕向量的检索。该策略适用于配有相应字幕的视频数据集。它使用文本编码器从每个视频的字幕中提取特征向量，并构建字幕向量数据集。在推理期间，通过寻找字幕特征向量之间的最高余弦相似度来检索$N$个相关视频。 • 限制性检索。该策略适用于人工划分的视频数据集。具体来说，在推理期间，检索过程限制在测试集内的视频，检索方法遵循与基于字幕向量的检索相同的程序。我们实现了这三种视频检索策略，并在评估数据集上进行了测试。结果在表6中展示，基于字幕向量的检索实现了最佳性能，这归因于高质量的提示构建（见图4）和Qwen3-Embedding的出色检索能力。因此，我们在本研究中采用基于字幕向量的检索策略。然而，其他策略也表现出具有竞争力的性能，表明推理过程仅受到检索策略选择的轻微影响。总体而言，我们的框架在不同视频检索策略中表现出稳健的性能。

![](images/7.jpg)  
Fig. 7: Comparative analysis of accuracy $( \% )$ for NExT-QA across different models under varying relevance of related videos.

结论。使用更相关的视频可以提高性能，因为关键在于检索包含关键信息的视频。添加更多无关或相关性较低的视频不可避免地引入噪声，但我们的方法在一定程度上能够过滤掉无关信息。此外，我们的框架表现出显著的鲁棒性，采用不同的视频检索策略时仍能维持强大的性能。

# 4.5 可视化

我们在图 8 中可视化了我们的多视频协作框架的推理过程。查询是“视频中的滑板手在进行什么活动？”，这需要高层次和领域相关的知识。基线模型未能提供详细的回答，仅给出了通用的描述。相比之下，我们的框架将视频表示为图结构数据，保留了关键的时空信息。在图 8 中，颜色块对应于匹配颜色的三元组，突出显示通过池化注意力识别的关键兴趣区域。这些区域强调了理解场景所需的最重要的视觉特征，基于其关联的三元组。此外，虚线展示了这些三元组之间的关系，表明结构化视频表示是如何基于不同帧和视频之间的关系信息进行聚合和传递的。通过图 8 中展示的跨图注意力机制，来自相关视频的子图为当前视频的图贡献了有用的关系结构。通过融合来自相关视频的子图特征，我们的模型建立了复杂场景的连贯理解，从而得出准确且详细的回应。

# 4.6 更多视频问答结果

我们在图9中展示了额外的视频问答结果，说明了我们框架在提供准确、详细和具有上下文精确性的答案方面的优势，超越了基线方法。例如，我们的模型有效地整合了领域知识，正确解读独特活动，如准确识别身穿绿色衣服的人是在“雕刻西瓜制作南瓜灯”，而不仅仅是“用刀切西瓜”。同样，它提供了关于使用双杠的更详细和上下文相关的答案，识别为“两个高度不同的平行杠”，并具体说明该男子“正在利用这些杠进行他的日常活动”。此外，我们的框架可能在一定程度上减轻了幻觉问题，例如在评估保龄球视频的安全性时，它得出的结论是“保龄球是一项安全运动，因为没有显示受伤”，而不是生成虚构的细节。在视频中，滑板者正在进行什么活动？

![](images/8.jpg)  

基线：滑板运动员正在道路上滑行。

![](images/9.jpg)  

图8：我们结构化的多视频协作推理的可视化。我们展示了一个代表性的视频问答示例，来自我们的结构化多视频协作管道，展示了池化注意力可视化、结构化结果和跨图注意力图，以及应用我们框架前后的生成答案。在每个场景中，颜色块对应着配色相匹配的三元组，突出显示通过池化注意力识别的兴趣区域，而虚线则表示场景内三元组之间的关系。这些示例展示了我们结构化多视频协作推理框架的鲁棒性和可靠性，相较于基线提供了精确、准确和上下文感知的答案，适用于多样化的查询。

# 5 结论

在本研究中，我们提出了一种开创性的框架，通过结构化的多视频协作推理来增强视频大语言模型。我们首先设计了视频结构模块，将视频建模为时空图。随后，图融合模块将相关视频信息整合成增强的图词元，这些词元与视觉词元和文本词元结合，形成多视频结构化提示，作为语言模型的输入。大量实验表明，我们的方法在理解复杂视频内容和准确回答查询方面的有效性和鲁棒性。我们希望我们的工作能够为可靠的视频理解提供见解，并激发更多研究兴趣。

# 6 语句和声明

# 6.1 数据可用性声明

原始视频源及相应注释可以通过公开可获取的开源数据集合法访问[54]。此外，我们计划在不久的将来公开发布用于数据处理管道的代码以及用于GFM训练的训练数据集。

# 6.2 致谢与竞争利益

本文部分由中国国家自然科学基金资助（编号：62325109，62561160155，U21B2013），部分由上海“一带一路”青年学者交流基金资助（编号：24510742000）。

# 穿绿色衣服的人在做什么？

![](images/10.jpg)  
Fig. 9: Visualization of video question answering examples.

# References

[1] Li, Y., Wang, C., Jia, J.: Llama-vid: An image is worth 2 tokens in large language models. arXiv preprint arXiv:2311.17043 (2023)

[2] Yang, A., Nagrani, A., Seo, P.H., Miech, A., Pont-Tuset, J., Laptev, I., Sivic, J., Schmid, C.: Vid2seq: Large-scale pretraining of a visual language model for dense video captioning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1071410726 (2023)

[3] Zhang, H., Li, X., Bing, L.: Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858 (2023)

[4] Li, K., He, Y., Wang, Y., Li, Y., Wang, W., Luo, P., Wang, Y., Wang, L., Qiao, Y.:

Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355 (2023)

[5] Maaz, M., Rasheed, H., Khan, S., Khan, F.S.: Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424 (2023)

[6] Lin, B., Ye, Y., Zhu, B., Cui, J., Ning, M., Jin, P., Yuan, L.: Video-llava: Learning united visual representation by alignment before projection. In: Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 59715984 (2024)

[7] Luo, R., Zhao, Z., Yang, M., Dong, J., Qiu, M., Lu, P., Wang, T., Wei, Z.: Valley: Video assistant with large language model enhanced ability. arXiv preprint arXiv:2306.07207 (2023)

[8] Liu, H., Lv, W., See, J., Lin, W.: Taskadaptive spatial-temporal video sampler for few-shot action recognition. In: Proceedings of the 30th ACM International Conference on Multimedia, pp. 62306240 (2022)

[9] Yang, A., Miech, A., Sivic, J., Laptev, I., Schmid, C.: Tubedetr: Spatio-temporal video grounding with transformers. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16442- 16453 (2022)

[10] Yu, W., Zheng, H., Li, M., Ji, L., Wu, L., Xiao, N., Duan, N.: Learning from inside: Self-driven siamese sampling and reasoning for video question answering. Advances in Neural Information Processing Systems 34, 2646226474 (2021)

[11] Piergiovanni, A., Kuo, W., Angelova, A.: Rethinking video vits: Sparse video tubes for joint image and video learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22142224 (2023)

[12] Jin, K.-M., Lee, G.-H., Lee, S.-W.: Otpose: occlusion-aware transformer for pose estimation in sparsely-labeled videos. In: 2022 IEEE International Conference on Systems, Man, and Cybernetics (SMC), pp. 32553260 (2022). IEEE

[13] Dong, N., Zhang, L., Yan, S., Tang, H., Tang, J.: Erasing, transforming, and noising defense network for occluded person re-identification. IEEE Transactions on Circuits and Systems for Video Technology (2023)

[14] Wang, Z., Zhu, Y.: Video key frame monitoring algorithm and virtual reality display based on motion vector. IEEE Access 8, 159027159038 (2020)

[15] Feng, J., Xiao, X.: Multiobject tracking of wildlife in videos using few-shot learning. Animals 12(9), 1223 (2022)

[16] Volino, M., Casas, D., Collomosse, J.P.,

Hilton, A.: Optimal representation of multiview video. In: Proceedings of BMVC 2014- British Machine Vision Conference (2014). BMVC

[17] Zhang, W., Li, Y., Lu, W., Xu, X., Liu, Z., Ji, X.: Learning intra-video difference for person re-identification. IEEE Transactions on Circuits and Systems for Video Technology 29(10), 30283036 (2018)

[18] Xia, B., He, J., Zhang, Y., Wang, Y., Tian, Y., Yang, W., Van Gool, L.: Structured sparsity learning for efficient video superresolution. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2263822647 (2023)

[19] Ji, J., Krishna, R., Fei-Fei, L., Niebles, J.C.: Action genome: Actions as compositions of spatio-temporal scene graphs. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1023610247 (2020)

[20] Liu, N.F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., Liang, P.: Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics 12, 157173 (2024)

[21] Hsieh, C.-P., Sun, S., Kriman, S., Acharya, S., Rekesh, D., Jia, F., Ginsburg, B.: Ruler: What's the real context size of your longcontext language models? arXiv preprint arXiv:2404.06654 (2024)

[22] Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al.: Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023)

[23] Jiang, A.Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D.S., Casas, D.d.l., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al.: Mistral 7b. arXiv preprint arXiv:2310.06825 (2023)

[24] He, Z., Yu, H., Gong, Z., Liu, S., Li, J.,

Lin, W.: Rodimus\*: Breaking the accuracyefficiency trade-off with efficient attentions. In: The Thirteenth International Conference on Learning Representations (2025). https://openreview.net/forum?id=IIVYiJ1ggK [25] Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. Advances in neural information processing systems 36 (2024)

[26] Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., Parikh, D.: Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 69046913 (2017)

[27] Lin, Z., Chen, X., Pathak, D., Zhang, P., Ramanan, D.: Visualgptscore: Visiolinguistic reasoning with multimodal generative pre-training scores. arXiv preprint arXiv:2306.01879 (2023)

[28] Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W.X., Wen, J.-R.: Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355 (2023)

[29] Gunjal, A., Yin, J., Bas, E.: Detecting and preventing hallucinations in large vision language models. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, pp. 1813518143 (2024)

[30] Wang, X., Zhang, X., Cao, Y., Wang, W., Shen, C., Huang, T.: Seggpt: Segmenting everything in context. arXiv preprint arXiv:2304.03284 (2023)

[31] Zhang, Y., Zhou, K., Liu, Z.: What makes good examples for visual in-context learning? Advances in Neural Information Processing Systems 36 (2024)

[32] Wu, J., Zhong, S.-h., Liu, Y.: Dynamic graph convolutional network for multi-video summarization. Pattern Recognition 107, 107382 (2020)

[33] Lewis, P., Perez, E., Piktus, A., Petroni,

F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., et al.: Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems 33, 94599474 (2020)

[34] Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., Larson, J.: From local to global: A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130 (2024)

[35] Guo, Z., Xia, L., Yu, Y., Ao, T., Huang, C.: Lightrag: Simple and fast retrievalaugmented generation. arXiv preprint arXiv:2410.05779 (2024)

[36] Velikovi, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., Bengio, Y.: Graph attention networks. In: International Conference on Learning Representations (2018). https:/ /openreview.net/forum?id ${ } = 1$ JXMpikCZ [37] Li, J., Li, D., Savarese, S., Hoi, S.: Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In: International Conference on Machine Learning, pp. 1973019742 (2023). PMLR

[38] Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al.: Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems 35, 2371623736 (2022)

[39] Xue, L., Shu, M., Awadalla, A., Wang, J., Yan, A., Purushwalkam, S., Zhou, H., Prabhu, V., Dai, Y., Ryoo, M.S., et al.: xgen-mm (blip-3): A family of open large multimodal models. arXiv preprint arXiv:2408.08872 (2024)

[40] Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., et al.: Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326 (2024)

[41] Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., et al.: Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution. arXiv preprint arXiv:2409.12191 (2024)

[42] Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al.: Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923 (2025)

[43] Li, Y., Wang, C., Jia, J.: Llama-vid: An image is worth 2 tokens in large language models. In: European Conference on Computer Vision, pp. 323340 (2024). Springer

[44] Cheng, Z., Leng, S., Zhang, H., Xin, Y., Li, X., Chen, G., Zhu, Y., Zhang, W., Luo, Z., Zhao, D., et al.: Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476 (2024)

[45] Liu, Z., Zhu, L., Shi, B., Zhang, Z., Lou, Y., Yang, S., Xi, H., Cao, S., Gu, Y., Li, D., et al.: Nvila: Efficient frontier visual language models. In: Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 41224134 (2025)

[46] Hou, R., Chang, H., Ma, B., Shan, S., Chen, X.: Cross attention network for few-shot classification. Advances in neural information processing systems 32 (2019)

[50] Wu, J., Zhong, S.-H., Liu, Y.: Mvsgcn: A novel graph convolutional network for multivideo summarization. In: Proceedings of the 27th ACM International Conference on Multimedia, pp. 827835 (2019)

learning for fine-grained visual categorization. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 82428251 (2019)

[47] Wang, X., Zhang, S., Qing, Z., Tang, M. Zuo, Z., Gao, C., Jin, R., Sang, N.: Hybrid relation guided set matching for few-shot action recognition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1994819957 (2022)

[48] Cao, K., Ji, J., Cao, Z., Chang, C.-Y., Niebles, J.C.: Few-shot video classification via temporal alignment. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10618 10627 (2020)

[51] Panda, R., Mithun, N.C., Roy-Chowdhury, A.K.: Diversity-aware multi-video summarization. IEEE Transactions on Image Processing 26(10), 47124724 (2017)

[52] Wang, X., Wang, W., Cao, Y., Shen, C., Huang, T.: Images speak in images: A generalist painter for in-context visual learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 68306839 (2023)

[53] Bar, A., Gandelsman, Y., Darrell, T., Globerson, A., Efros, A.: Visual prompting via image inpainting. Advances in Neural Information Processing Systems 35, 2500525017 (2022)

[54] Zhang, Y., Wu, J., Li, W., Li, B., Ma, Z., Liu, Z., Li, C.: Video Instruction Tuning With Synthetic Data (2024). https://arxiv. org/abs/2410.02713

[55] Zhu, W., Huang, Y., Xie, X., Liu, W., Deng, J., Zhang, D., Wang, Z., Liu, J.: Autoshot: A short video dataset and state-of-the-art shot boundary detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2238 2247 (2023)

[56] Wu, H., Mao, J., Zhang, Y., Jiang, Y., Li, L., Sun, W., Ma, W.-Y.: Unified visual-semantic embeddings: Bridging vision and language with structured meaning representations. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 66096618 (2019)

[49] Luo, W., Yang, X., Mo, X., Lu, Y., Davis, L.S., Li, J., Yang, J., Lim, S.-N.: Cross-x [57] Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C.,

Lv, C., et al.: Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025)

[58] Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L.: Sigmoid loss for language image pretraining. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1197511986 (2023)

[59] Zhang, Y., Li, M., Long, D., Zhang, X., Lin, H., Yang, B., Xie, P., Yang, A., Liu, D., Lin, J., et al.: Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176 (2025)

[60] Vaswani, A.: Attention is all you need. Advances in Neural Information Processing Systems (2017)

[61] Simonyan, K., Zisserman, A.: Two-stream convolutional networks for action recognition in videos. Advances in neural information processing systems 27 (2014)

[62] Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., Liu, Y.: Roformer: Enhanced transformer with rotary position embedding. Neurocomputing 568, 127063 (2024)

[63] Ba, J.L., Kiros, J.R., Hinton, G.E.: Layer normalization. arXiv preprint arXiv:1607.06450 (2016)

[64] Gao, L., Zhong, Y., Zeng, Y., Tan, H., Li, D., Zhao, Z.: Linvt: Empower your image-level large language model to understand videos. arXiv preprint arXiv:2412.05185 (2024)

[65] Yu, Z., Xu, D., Yu, J., Yu, T., Zhao, Z., Zhuang, Y., Tao, D.: Activitynet-qa: A dataset for understanding complex web videos via question answering. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33, pp. 91279134 (2019)

[66] Xiao, J., Shang, X., Yao, A., Chua, T.-S.: Next-qa: Next phase of question-answering to explaining temporal actions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9777- 9786 (2021)

[67] Mangalam, K., Akshulakov, R., Malik, J.: Egoschema: A diagnostic benchmark for very long-form video language understanding. Advances in Neural Information Processing Systems 36, 4621246244 (2023)

[68] Fu, C., Dai, Y., Luo, Y., Li, L., Ren, S., Zhang, R., Wang, Z., Zhou, C., Shen, Y., Zhang, M., et al.: Video-mme: The firstever comprehensive evaluation benchmark of multi-modal llms in video analysis. In: Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 24108 24118 (2025)

[69] Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. ICLR 1(2), 3 (2022)

[70] Xu, L., Zhao, Y., Zhou, D., Lin, Z., Ng, S.K., Feng, J.: Pllava: Parameter-free llava extension from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994 (2024)

[71] Li, K., Wang, Y., He, Y., Li, Y., Wang, Y., Liu, Y., Wang, Z., Xu, J., Chen, G., Luo, P., et al.: Mvbench: A comprehensive multimodal video understanding benchmark. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2219522206 (2024)

[72] Zhang, Y., Li, B., Liu, h., Lee, Y.j., Gui, L., Fu, D., Feng, J., Liu, Z., Li, C.: LLaVANeXT: A Strong Zero-shot Video Understanding Model (2024). https://llava-vl. github.io/blog/2024-04-30-llava-next-video/ [73] Zhang, B., Li, K., Cheng, Z., Hu, Z., Yuan, Y., Chen, G., Leng, S., Jiang, Y., Zhang, H., Li, X., et al.: Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106 (2025)

[74] Chen, Z., Wang, W., Cao, Y., Liu, Y., Gao, Z., Cui, E., Zhu, J., Ye, S., Tian, H., Liu, Z., et al.: Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint