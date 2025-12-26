# VideoForest：基于人物锚定的跨视频问答分层推理

孟怡然\* 中山大学 中国珠海

叶俊宏\* 中山大学 中国珠海

魏周 卡迪夫大学 英国

岳光辉 深圳大学 中国深圳

毛旭东 中国珠海中山大学

王若梅 中山大学 中国广州

赵宝全 中山大学 中国珠海

# 摘要

跨视频问答在建立视频流之间的有意义联系和管理多源信息检索的复杂性方面，面临着显著挑战，这超出了传统单视频理解的范围。我们推出了VideoForest，这是一个新颖的框架，通过以人为锚的分层推理来解决这些挑战，使得在不需要端到端训练的情况下实现有效的跨视频理解。VideoForest集成了三个关键创新：1）一种以人为锚的特征提取机制，采用ReID和跟踪算法在多个视频源之间建立稳健的时空关系；2）一种多粒度的跨越树结构，分层组织围绕人物级别轨迹的视觉内容；3）一种多智能体推理框架，能够高效地遍历这一分层结构以回答复杂查询。为了评估我们的方法，我们开发了CrossVideoQA1，这是一个专门为以人为中心的跨视频分析设计的综合基准。实验结果表明，VideoForest在跨视频推理任务中表现优越，在人员识别方面达到了$71.93\%$的准确率，在行为分析方面达到了$83.75\%$，在摘要和推理方面达到了$51.67\%$。

# 关键词

跨视频提问回答、基于人物的推理、层次视频表示、多代理框架

# ACM参考格式：

Yiran Meng, Junhong Ye, Wei Zhou, Guanghui Yue, Xudong Mao, Ruomei Wang, 和 Baoquan Zhao. 2025. VideoForest: 基于人的层次推理用于跨视频问答. 载于第33届ACM国际多媒体会议论文集（MM '25），2025年10月27日至31日，爱尔兰都柏林。ACM, 纽约, NY, 美国，10页。 https://doi.org/10.1145/3746027.3754573

# 1 引言

跨视频理解代表了计算机视觉中最具挑战性的前沿之一，要求系统提取、关联和推理分布在多个视频流中的信息。与单视频分析不同，单视频分析中的上下文保持在时间边界内，跨视频推理需要复杂的机制在不同的空间视角和时间序列之间建立有意义的连接。这一能力在监控和监测场景中特别关键，因为关键信息本质上分散在多个摄像头中，因此需要统一分析以实现全面的态势感知。

考虑一项安全调查，要求分析人员确定：哪位个人在14:00到16:00之间穿越了所有三个校园建筑？回答此类查询不仅需要在每个视频流中识别和跟踪个人，还需要跨多个摄像头对不同角度和录制条件下的身份和行为进行交叉参考。尽管视频理解方面取得了显著进展，但当前的方法仍然在单流处理范式上受限，使其无法满足跨多个视频源的查询需求。这种架构限制阻止了跨摄像头视角整合互补信息。

如图1所示，现有的视频问答系统[7, 26, 32, 46]主要集中在最大化单个视频内的性能，这无意中强化了这种单流限制。即使是最近的进展，例如引入复杂代理搜索策略的视频代理(VideoAgent) [7]，以及开创运动轨迹分析的Chat-Video [35]，最终仍然是在孤立视频处理的范围内操作。在文献中，建立不同视频流之间语义桥接的关键能力——这是进行真正跨视频推理所必需的——仍然在很大程度上没有得到探索。

![](images/1.jpg)  

Figure 1: Comparison of single-video vs. cross-video question answering paradigms.

为了解决这些限制，我们引入了VideoForest，一种新颖的分层框架，可以在多个视频流中实现高效的人物中心推理（见图2）。我们的关键见解是，人类主体作为不同视频之间的自然桥梁，提供了一致的参考实体，从而可以构建跨视频关系。VideoForest通过三种创新组件来实现这一见解：首先，我们开发了一种基于人物的特征提取机制，采用重新识别（ReID）和跟踪算法，在多个视频中建立一致的身份表示，从而创建跨越不同摄像机视角的稳健时空关系。其次，我们设计了一种多粒度跨越树结构，按层次组织以人物轨迹为中心的视觉内容，使得从粗略的场景级信息到细粒度的行为细节的高效导航成为可能。第三，我们实现了一种多智能体推理框架，能够高效地遍历这一分层结构，执行复杂的跨视频推理，同时保持计算的可处理性。

为了评估我们的方法并推动跨视频理解研究，我们推出了CrossVideoQA，这是第一个专门为监控场景中以人为中心的跨视频问答设计的综合基准数据集。我们的大量实验证明了VideoForest在各种推理任务中的有效性。

本工作的主要贡献有三方面：

我们介绍了第一个以人物为锚的跨视频问答的层次框架，开创了一种基于树的架构，利用人类主体作为桥梁连接多个视频流，实现对分布式视觉信息的统一理解。• 我们开发了一种高效的多粒度视频组织策略，与多智能体推理框架相结合，保留关键的时间-空间关系，同时使跨视频问答在计算上可行。• 我们提出了CrossVideoQA，一个用于评估以人物为中心的跨视频问答能力的新基准数据集，为这一新兴研究方向建立了新的评估协议和性能基准。

# 2 相关文献

# 2.1 视频问答

视频问答（VideoQA）是多模态理解的重要基石，与文本-视频检索和视频字幕生成并肩，需要对视频和语言之间复杂语义和因果关系的深入理解。为了提高模型的鲁棒性和可解释性，视觉定位方法使模型能够突出显示相关的视频片段或关键帧以生成答案。虽然这些方法成功定位了证据，但推理过程依然不透明。ChatVideo引入运动轨迹作为基本分析单元，利用专业的视觉基础模型生成属性注释，以增强动态场景下的时间建模。另一条研究线使用外部大型语言模型作为推理模块。LLoVi通过视频字幕将视频问答转换为基于文本的问答，而VideoAgent则递归评估框架的充足性以回答问题。尽管这些方法提高了问答性能，但它们在很大程度上依赖于大型语言模型的语言推理，这容易出现幻觉且缺乏可解释性。Video-CCAM引入了视觉编码器与大型语言模型之间的交叉模态注意力和因果掩蔽，在多个基准上表现出色。InternVL 2.5则改进了训练策略和数据质量，同时在短视频理解、长视频检索和问答任务中表现优异。TV-trees采用神经符号方法在视觉和文本模态之间进行明确推理，但假设视频文本已被预先转录。尽管在单视频任务上有所进展，能够进行跨视频理解和复杂时间问答的系统依然稀少。

# 2.2 结构化视频表示

结构化视频表示通过将视频从帧序列转化为层次语义表示，增强了视频问答（Video QA）的效果，从而更好地理解对象、动作和事件之间的关系[41]。该方法将多粒度的视觉信息与相应的语言概念融合，提高了准确性和可解释性[8, 17, 39]。最近的视频语言方法强调结构化帧表示，以实现高效场景理解[5, 12, 25, 30, 50]。LVNet [30] 通过层次关键帧选择减少冗余，VideoReCap [12] 引入渐进式字幕，桥接短片与长片的理解，而VideoTree [50] 通过自上而下的视频语言嵌入，实现了动态深度调整，促进了长视频的高效理解。然而，这些方法主要关注单视频分析，未能探讨跨视频相关性的挑战。我们的工作扩展了这些基础思想，向多视频领域引入以人为中心的连接性和时空关系建模，以实现有效的跨视频理解。

# 2.3 视频理解基准测试

视频理解任务通过三个复杂性层次进展：对整体视频事件的抽象理解[3, 11]，对特定时刻识别的时间理解[24, 34]，以及对时空定位的时空理解[1, 49]。这种进展反映了人类从视觉信息中建立全面理解的认知过程[27]。多模态大型语言模型（MLLMs）推动了全面评估框架的建立[10, 16, 19, 28, 29]。VideoMM [10] 创建了第一个整体 MLLM 评估基准，而 VideoVistaAV [19] 引入了考虑多样内容类别、时间尺度和推理能力的多面向评估。尽管取得了这些进展，现有基准主要集中在单一视频理解上，导致跨视频推理评估的关键缺口。我们提出的 CrossVideoQA 基准通过基于爱丁堡办公室监控和 HACS 数据集 [31, 48] 精心策划的跨视频查询来解决这一限制，为多视频理解系统建立了新的评估标准。

![](images/2.jpg)  

Figure 2: VideoForest architecture for cross-video question answering.

# 3 方法论

# 3.1 问题定义和符号

我们将跨视频问答任务形式化如下。设 $\mathcal { V } = \{ V _ { 1 } , V _ { 2 } , . . . , V _ { n } \}$ 表示 $n$ 个视频的集合，其中每个视频 $V _ { i }$ 由有序的帧时间序列组成 $\mathcal { F } _ { i } = \{ f _ { i , 1 } , f _ { i , 2 } , . . . , f _ { i , m _ { i } } \}$，其中 $m _ { i }$ 表示视频 i 中帧的数量。我们的目标是构建一个统一的层次化表示，以便实现高效的跨视频信息检索和推理。

对于每一帧 $f_{i,j}$（视频 i 的第 j 帧），我们提取两个互补的表示：（1）视觉嵌入 $\mathbf{v}(f_{i,j}) \in \mathbb{R}^{d}$，一个捕捉语义视觉内容的 $d$ 维稠密表示；（2）人物检测 $\mathbf{p}(f_{i,j}) = \{(t_{k}, \mathbf{x}_{k}, \mathrm{id}_{k})\}_{k=1}^{K_{i,j}}$，其中 $K_{i,j}$ 是帧 $f_{i,j}$ 中检测到的人数，$t_{k} \in \mathbb{R}$ 表示时间戳，$\mathbf{x}_{k} = (x_{k}, y_{k}) \in \mathbb{R}^{2}$ 表示空间坐标，$\mathrm{id}_{k} \in \mathcal{I}$ 是来自所有身份集合 $\boldsymbol{\mathcal{T}}$ 的唯一人物标识符。

跨视频问答任务可以正式定义为一个映射函数：

您接受的训练数据截至2023年10月。

Q : \mathcal { V } \times \mathcal { T } \times \mathcal { L }  \mathcal { A } ,
$$

其中 $\mathcal { T } \subset \mathbb { R } ^ { + }$ 表示时间约束（例如，感兴趣的时间区间），$\mathcal { L } \subset \mathbb { R } ^ { 2 }$ 表示空间约束（例如，感兴趣的区域），$\mathcal { A }$ 表示答案空间，可能包括文本响应、时间定位或实体识别。这种公式明确地将跨视频推理过程建模为依赖于时间和空间约束，捕捉了监控和观察场景中固有的复杂时空关系。我们的层级 VideoForest 框架通过一个基于人员锚定的树结构实现这种映射，能够有效地遍历和整合多个视频源的信息。

# 3.2 双流特征提取与自适应分割

基于我们的正式问题定义，我们实现了一种互补的双流架构，以实现全面的视频表示。视觉内容流使用ViCLIP编码器[36, 37]，由$\theta _ { v }$参数化，以计算在我们符号中定义的帧级嵌入：

$$
\mathbf { v } ( f _ { i , j } ) = \phi ( f _ { i , j } ; \theta _ { v } ) \in \mathbb { R } ^ { d } .
$$

同时，基于人中心的流利用一个具有参数 $\theta _ { p }$ 的专用跟踪模型 $\psi$ 来识别和提取结构化的人物表示：

$$
\mathbf { p } ( f _ { i , j } ) = \psi ( f _ { i , j } ; \theta _ { p } ) = \{ ( t _ { k } , \mathbf { x } _ { k } , \mathrm { i d } _ { k } ) \} _ { k = 1 } ^ { K _ { i , j } } ,
$$

其中 $K_{i,j}$ 表示在帧 $f_{i,j}$ 中检测到的人数。

为了将视频划分为语义上连贯的片段，我们定义了一个自适应边界检测函数 $S : \mathcal { F } _ { i }  \{ 0 , 1 \}$ ，该函数通过析取标准识别显著的转变：

$$
S ( f _ { i , j } ) = \mathbb { \mathbb { k } } [ C _ { 1 } ( f _ { i , j } ) \vee C _ { 2 } ( f _ { i , j } ) \vee C _ { 3 } ( f _ { i , j } ) ] ,
$$

三个互补标准的制定如下：

$$
\begin{array} { r l } & { C _ { 1 } ( f _ { i , j } ) : \| \mathbf { v } ( f _ { i , j } ) - \mathbf { v } ( f _ { i , j + 1 } ) \| _ { 2 } > \epsilon _ { 1 } , \quad \mathrm { ( l o c a l ~ t r a n s i t i o n ) } } \\ & { C _ { 2 } ( f _ { i , j } ) : \| \mathbf { v } ( f _ { i , j } ) - \mathbf { v } ( f _ { i , j } ^ { \mathrm { c e n t } } ) \| _ { 2 } > \epsilon _ { 2 } , \quad \mathrm { ( g l o b a l ~ d e v i a t i o n ) } } \\ & { \quad C _ { 3 } ( f _ { i , j } ) : | \mathcal { P } ( f _ { i , j } ) \triangle \mathcal { P } ( f _ { i , j - 1 } ) | \geq \Delta \varphi . \quad \mathrm { ( p e r s o n - s e t ~ c h a n g ~ } } \end{array}
$$

在这里，$\lVert \rVert ^ { \epsilon } [ \cdot ]$表示指示函数，$\begin{array} { r l } { \mathcal { P } ( f _ { i , j } ) } & { { } = } \end{array}$ $\{ \mathrm { i d } _ { k } | ( t _ { k } , \mathbf { x } _ { k } , \mathrm { i d } _ { k } ) \in \mathbf { p } ( f _ { i , j } ) \}$表示存在于帧$f _ { i , j }$中的人员身份的集合，而$f _ { i , j } ^ { \mathrm { c e n t } }$指当前段的代表性帧。多标准方法在三个不同级别上运行：$C _ { 1 }$通过局部特征距离捕捉帧间外观变化，$C _ { 2 }$测量与该段视觉原型的偏差以识别全局内容漂移，$C _ { 3 }$通过连续人员身份集合之间的对称差异的基数$\triangle$来量化以人为中心的动态。阈值$\epsilon _ { 1 } , \epsilon _ { 2 }$和$\Delta \varphi$通过对保留数据集的交叉验证来确定，以优化时间粒度和语义连贯性之间的权衡。当根据$S ( f _ { i , j } ) = 1$检测到段边界时，我们创建一个新段$S _ { i , k } = \{ f _ { i , j _ { \mathrm { s t a r t } } } , f _ { i , j _ { \mathrm { s t a r t } } + 1 } , . . . , f _ { i , j _ { \mathrm { e n d } } } \}$，其中$j _ { \mathrm { s t a r t } }$和$j _ { \mathrm { e n d } }$表示段的包含边界。这种方法为每个视频$V _ { i }$生成一系列不重叠的段$\{ S _ { i , 1 } , S _ { i , 2 } , . . . , S _ { i , n _ { i } } \}$，有效地将连续的视频流解析为离散的语义单元。

这种自适应分割作为我们层次树表示的基础构件，能够高效地进行多粒度视频索引和检索。这些分段保持语义一致性，同时建立可管理的单位，以便后续在视频之间进行基于人物的关联。通过在分割标准中同时融入视觉内容和以人为中心的动态，我们的方法确保生成的分段保持有意义的上下文边界，促进跨视频推理。

# 3.3 多层次语义表示

考虑到分段视频结构，我们为每个段$s_{i,k}$构建语义丰富的表示。我们定义一个多模态编码函数$\eta : S \times \mathcal{P} \to \mathbb{R}^{d}$，将视觉内容和人物轨迹映射到统一的语义空间：

$$
\mathbf { C } ( S _ { i , k } ) = \eta ( \mathbf { v } ( f _ { i , j } ^ { \mathrm { k e y } } ) , \mathbf { P } ( S _ { i , k } ) ; \theta _ { \eta } ) ,
$$

其中 $f _ { i , j } ^ { \mathrm { k e y } } = f _ { i , \lfloor ( j _ { \mathrm { s t a r t } } + j _ { \mathrm { e n d } } ) / 2 \rfloor }$ 表示该片段，$\mathbf { P } ( S _ { i , k } ) = \{ \mathbf { p } ( f _ { i , j } ) | f _ { i , j } \in S _ { i , k } \}$ 表示该片段中所有帧的聚合人员检测集，而 $\theta _ { \eta }$ 参数化了编码函数。这种表述确保我们的语义表示通过关键帧嵌入捕捉静态视觉内容，并通过轨迹聚合捕捉动态以人为中心的活动。

这种多层次语义表示为跨视频推理提供了丰富的基础，捕捉了视觉场景上下文和以人为中心的动态。生成的段级编码 $\{ \mathbf { C } ( S _ { 1 , 1 } ) , \mathbf { C } ( S _ { 1 , 2 } ) , . . . , \mathbf { C } ( S _ { n , n _ { n } } ) \}$ 作为我们层次树结构中的语义节点，使得在多个视频之间高效检索和关联内容成为可能。

# 3.4 视频森林建设

基于分段视频及其语义表示，我们构建一个层次树结构 $\mathcal{T} = (V, E)$，该结构在多个粒度上组织内容。每个节点 $v \in V$ 被定义为一个结构化的元组：

$$
\boldsymbol { v } = ( t _ { \mathrm { s t a r t } } , t _ { \mathrm { e n d } } , \mathcal { R } _ { v } , \mathbf { C } _ { v } , \Gamma _ { v } ) ,
$$

其中 $[ t _ { \mathrm { s t a r t } } , t _ { \mathrm { e n d } } ] \subset \mathbb { R } ^ { + }$ 表示由 th nhende 跨越的时间区间，$\mathscr { R } _ { v } = \{ ( \mathrm { i d } _ { k } , \tau _ { k } ) \} _ { k = 1 } ^ { K _ { v } }$ 包含 $\mathrm { i d } _ { k } \in \mathcal { I }$ 的引证信息和轨迹描述符 $\tau _ { k }$，$\mathbf { C } _ { v } \in \mathbb { R } ^ { d }$ 表示由函数 $\eta$ 计算的语义内容表示，$\Gamma _ { v } \subset V$ 表示子节点的集合。

边集 $E ~ = ~ \{ ( v _ { i } , v _ { j } ) ~ \mid ~ v _ { j } ~ \in ~ \Gamma _ { v _ { i } } \}$ 定义了层次化的父子关系，能够实现高效的多粒度遍历。为了确保全面的时间覆盖，同时保持不重叠的子区段，节点的递归划分遵循由分割函数 Split : $V  2 ^ { V }$ 实现的无重叠覆盖原则：

$$
{ \mathrm { S p l i t } } ( v ) = \{ v _ { 1 } , v _ { 2 } , \ldots , v _ { K _ { v } } \} ,
$$

满足以下时间范围和不相交约束：

$$
\bigcup _ { i = 1 } ^ { K _ { v } } [ t _ { \mathrm { s t a r t } } ( v _ { i } ) , t _ { \mathrm { e n d } } ( v _ { i } ) ] = [ t _ { \mathrm { s t a r t } } ( v ) , t _ { \mathrm { e n d } } ( v ) ] ,
$$

$$
\forall i \neq j : [ t _ { \mathrm { s t a r t } } ( v _ { i } ) , t _ { \mathrm { e n d } } ( v _ { i } ) ] \cap [ t _ { \mathrm { s t a r t } } ( v _ { j } ) , t _ { \mathrm { e n d } } ( v _ { j } ) ] = 0 .
$$

分割标准是基于语义相似性和个体级连续性自适应确定的，分割边界优先与在分割过程中识别的段落边界对齐。对于每个视频 $V _ { i }$，我们构建一个相应的树 $\mathcal { T } _ { i }$，根节点跨越整个视频时长，叶节点对应于细粒度的片段 $\{ S _ { i , k } \} _ { k = 1 } ^ { n _ { i } }$，以便快速识别与时间和个体中心查询相关的内容。在每个节点整合个体重识别信息 $\mathcal { R } _ { v }$ 创建了不同视频树之间的自然桥接点，使得基于个体身份连续性的跨视频关联和推理成为可能。

![](images/3.jpg)  

Figure 3: Architecture of our distributed multi-agent framework for cross-video reasoning.

# 3.5 协作多智能体系统用于跨视频推理

如图3所示，我们的跨视频推理系统通过协调的多智能体架构整合多个视频源的信息。这种方法解决了视频之间时空关系的挑战，例如同一场景的不同视角或在不同时间从同一视角的录制。该系统采用四个专门的代理模块协同工作，以促进高效的跨视频推理。我们的多智能体推理系统是基于CrewAI框架实现的，该框架提供了模块化的代理任务调度和工具集成能力。我们扩展了CrewAI，以支持动态树状遍历和跨代理知识传播。

3.5.1 代理架构和功能专业化。我们的多代理框架由四个专业化组件组成，每个组件具有独特的功能：

$$
\mathcal { A } = \{ \mathcal { A } _ { \mathrm { f l t e r } } , \mathcal { A } _ { \mathrm { r e t r i e v a l } } , \mathcal { A } _ { \mathrm { n a v i g a t e } } , \mathcal { A } _ { \mathrm { i n t e g r a t e } } \}
$$

${ \mathcal { A } } _ { \mathrm { f i t e r } }$ 智能体处理输入查询，以提取时间和空间约束，从集合 $\{ \mathcal { T } _ { i } \} _ { i = 1 } ^ { n }$ 中识别并选择相关的视频树结构。$\mathcal { A } _ { \mathrm { r e t r i e v a l } }$ 智能体管理知识库访问，检索相关信息，同时通过基于置信度的检索机制防止冗余计算。${ \mathcal { A } } _ { \mathrm { n a v i g a t e } }$ 智能体使用优化的搜索策略遍历层次树结构，以定位与查询相关的信息。最后，Aintegrate 智能体综合来自知识库和树结构的信息，进行跨视频推理，以生成全面的答案。

推理工作流程遵循一个连续的五阶段过程：(1) 视频选择，(2) 知识库检索，(3) 层次树遍历，(4) 跨视频信息整合，(5) 知识库更新。

3.5.2 知识库构建与基于信心的维护。为了应对在多个查询中反复访问相同信息的计算挑战，我们实现一个具有信心加权条目的全球知识库 $\mathcal { K }$：

$$
\mathcal { K } = \{ ( d _ { i } , l _ { i } , s _ { i } , c _ { i } ) \} _ { i = 1 } ^ { N } ,
$$

其中 $d _ { i } \in \mathcal { D }$ 表示日期信息，$l _ { i } \in \mathcal { L }$ 表示空间位置，$s _ { i } \in S$ 是一个包含主题和动作用的信息字符串，$c _ { i } \in [ 0 , C _ { \operatorname* { m a x } } ]$ 是置信分数。

$\mathcal { A } _ { \mathrm { r e t r i e v a l } }$ 代理通过正式更新函数 $\mathcal { U } : \mathcal { K } _ { \mathrm { n e w } } \times \mathcal { K } \mathcal { K }$ 来维护 $\mathcal { K }$ 的完整性，定义为：

$$
\mathcal { U } ( k _ { \mathrm { n e w } } , \mathcal { K } ) = \left\{ \begin{array} { l l } { \mathcal { K } \cup \{ ( d _ { \mathrm { n e w } } , l _ { \mathrm { n e w } } , s _ { \mathrm { n e w } } , 1 ) \} , } & { \mathrm { i f } ~ k _ { \mathrm { n e w } } \notin \mathcal { K } } \\ { \mathcal { K } \setminus \{ k _ { i } \} \cup \{ ( d _ { i } , l _ { i } , s _ { i } , c _ { i } + 1 ) \} , } & { \mathrm { i f } ~ k _ { \mathrm { n e w } } = k _ { i } \in \mathcal { K } } \\ { \mathcal { K } \setminus \{ k _ { i } \} \cup \{ ( d _ { i } , l _ { i } , s _ { i } , c _ { i } - 1 ) \} } & { \mathrm { i f } , k _ { \mathrm { n e w } } \approx k _ { i } ~ \mathrm { a n d } ~ c _ { i } > 2 } \\ { \mathcal { K } \setminus \{ k _ { i } \} \cup \{ ( d _ { \mathrm { n e w } } , l _ { \mathrm { n e w } } , s _ { \mathrm { n e w } } , 1 ) \} , } & { \mathrm { i f } ~ k _ { \mathrm { n e w } } \approx k _ { i } ~ \mathrm { a n d } ~ c _ { i } \leq 2 } \end{array} \right.
$$

其中 $k _ { \mathrm { n e w } } \approx k _ { i }$ 表示根据相似性度量 $\sin ( k _ { \mathrm { n e w } } , k _ { i } ) > \tau _ { \mathrm { s i m } }$ ，新条目与现有条目之间存在语义冲突。

这种基于信心的方法使系统能够随着时间的推移自我纠正，通过迭代的查询回答逐步完善知识库。在处理新查询时，信心分数超过阈值 $\tau _ { \mathrm { c o n f } }$ 的条目会被优先检索，从而减少计算负担并提高响应时间。

3.5.3 自适应层次搜索优化。$\mathcal { A } _ { \mathrm { n a v i g a t e } }$ 代理采用一种高效的自上而下搜索策略 $\boldsymbol { S } : \boldsymbol { Q } \times \boldsymbol { V } $ $2 ^ { \overset { \vartriangle } { \mathbf { C } } }$，递归地探索层次结构。对于查询 $q \in { \cal Q }$ 和树 $\mathcal { T }$ 中的节点 $\textit { v } \in \textit { V }$，搜索函数的公式如下：

$$
S ( q , v ) = \left\{ \begin{array} { l l } { \mathrm C _ { v } , } & { \mathrm { i f ~ R e l e v a n c e } ( q , \mathrm C _ { v } ) \geq \tau _ { \mathrm { r e l } } } \\ { \bigcup _ { v _ { c } \in \Gamma _ { v } } S ( q , v _ { c } ) , } & { \mathrm { o t h e r w i s e } } \end{array} \right.
$$

相关性 $\because Q \times \mathbb { R } ^ { d }  [ 0 , 1 ]$ 测量查询 $q$ 与节点 v 的内容表示 $\mathbf { C } _ { v }$ 之间的语义相似性，$\tau _ { \mathrm { r e l } } \in \left[ 0 , 1 \right]$ 是一个可配置的相关性阈值。

搜索过程从选定视频树的根节点开始，并基于从查询中提取的时间、空间和人本约束逐步优化探索。当查询中包含人员级别的信息时，搜索利用存储在每个节点上的ReID信息 $\mathcal{R}_{v}$ 高效识别不同视频中的相关内容。

# 4 实验评估

为了全面评估 VideoForest 在跨视频推理方面的能力，我们需要一个专门设计的基准，用于测试跨多个视频源的集成和理解，这些视频源具有不同的空间和时间关系。现有的视频问答基准主要集中在单视频理解上，因此不足以评估跨视频推理性能。我们首先介绍我们的 CrossVideoQA 基准，然后呈现实施细节和比较结果，证明我们的方法在多个推理任务和评估配置中的有效性。

# 4.1 CrossVideoQA 基准测试

我们介绍了CrossVideoQA，这是一个专门设计用于评估跨视频推理能力的综合基准数据集。该基准解决了整合来自多个视频源的信息的基本挑战，特别关注跨越不同空间位置和时间段的人本查询。

4.1.1 数据集构建。跨视频理解在多个领域具有重要应用。在监控环境中，它可以在复杂环境中如办公楼和交通枢纽，通过分布式相机网络跟踪个体。在内容分析场景中，它有助于在不同视频视角中发现相关事件，从而实现全面的故事重构。为了支持对这些能力的严格评估，CrossVideoQA整合了两个互补的高质量数据集：

$$
\mathcal { D } _ { \mathrm { C r o s s V i d e o Q A } } = \mathcal { D } _ { \mathrm { E O S D } } \cup \mathcal { D } _ { \mathrm { H A C S } }
$$

爱丁堡办公室监控数据集 [31] 提供了在 3 个不同地点于 12 个不同日期捕捉的 18 个监控视频，涵盖了大约 45 万帧。这一数据集对于分析控制室内环境中结构化的人类行为模式特别有价值。HACS 数据集 [48] 提供了 50,000 个视频，包含 155 万个动作片段，提供了更广泛的动作类别和环境背景。

4.1.2 评估框架。CrossVideoQA 结构围绕三个逐渐复杂的推理任务，这些任务评估跨视频理解的不同方面：

人员识别：评估系统在多个视频源中识别和跟踪特定个体的能力，建立跨越空间和时间边界的个体级对应关系。

•行为分析：评估人类活动、互动和行为模式的解读，这些可能跨越多个视频，需整合来自不同来源的上下文信息。

•总结和推理：测试在综合因果关系、提取见解以及在多个视频中进行逻辑推理方面的高级能力，以回答复杂问题。

为了全面评估在不同时空配置下的跨视频推理，我们定义了四种评估方式，这些方式的复杂性系统性地增加：

$M = \{ M _ { \mathrm { s i n g l e } } , M _ { \mathrm { c r o s s - s p a t i a l } } , M _ { \mathrm { c r o s s - t e m p o r a l } } , M _ { \mathrm { c r o s s - s p a t i o t e m p o r a l } } \}$ (18) 其中 $M _ { \mathrm { s i n g l e } }$ 评估单个视频内的检索（同一天，同地点），$M _ { \mathrm { c r o s s - s p a t i a l } }$ 需要在相同时间段内跨地点的整合，$M _ { \mathrm { c r o s s - t e m p o r a l } }$ 评估固定位置的时间推理，而 $M _ { \mathrm { c r o s s } }$ 的时空表示最具挑战性的场景，需要全面的时空整合。这个结构化框架提供了对一个系统在跨视频理解能力上的全面评估，涵盖了一系列日益复杂的场景，从而能够有针对性地识别其优势和局限性。

4.1.3 基准构建方法论。为了确保基准的质量和相关性，我们采用了一种严格的三阶段问题生成流程。首先，领域专家为每个推理类别和评估方式手动创建高质量的示范问题，建立金标准参考。其次，使用大型语言模型在受限生成参数下系统性地扩展问题集，以确保覆盖面和多样性。最后，所有生成的问题都经过专家审核，以验证可回答性、事实准确性和适当的难度校准。这种系统的方法产生了一个多样且具有挑战性的基准，系统性地探讨了有效跨视频推理所需的能力。

# 4.2 实现细节

我们进行了全面的实验，以评估在CrossVideoQA基准上的VideoForest。所有实验均在NVIDIA RTX 4090 GPU上使用PyTorch框架进行。为了公平比较，我们在所有模型上实施了一致的评估协议。

# 4.3 比较方法

我们将 VideoForest 与最先进的视频理解模型进行了基准测试：

•Video-CCAM [9]: 结合了视觉编码器和语言模型之间的交叉注意机制与因果掩码，在不同视频长度领域表现出色。

•InternVL 2.5 [18, 38]：一种先进的多模态语言模型，扩展了InternVL 2.0，采用了更加强化的训练策略和数据质量优化。

• LLaVA-OneVision [15, 42]: 一种统一模型，旨在实现单图像、多图像和视频理解任务中的跨模态迁移学习。

• ShareGPT4Video [4]：一种经过训练的视听语言模型，基于480万高质量视频，在多个视频理解基准测试中实现了最先进的性能。该模型还为我们的VideoForest框架提供了字幕组件。

为了公平评估这些单视频模型在跨视频任务上的表现，我们实施了一项顺序处理协议，明确指示：(1) 评估视频与查询的相关性，(2) 从相关视频中提取相关信息，以及 (3) 将提取的信息合成一个连贯的响应。

Table 1: Quantitative performance comparison across the three reasoning categories defined in the CrossVideoQA benchmark.   

<table><tr><td>模型</td><td>人类推荐</td><td>行为分析</td><td>总结与推理</td><td>整体准确率</td></tr><tr><td>ShareGPT4Video-8B [4]</td><td>50.00</td><td>49.38</td><td>41.67</td><td>47.21</td></tr><tr><td>VideoCCAM-7B [9]</td><td>42.86</td><td>41.98</td><td>45.00</td><td>43.15</td></tr><tr><td>InternVL-2.5 [38]</td><td>58.93</td><td>66.67</td><td>46.67</td><td>58.38</td></tr><tr><td>LLaVAOneVision [15]</td><td>51.79</td><td>53.09</td><td>36.67</td><td>47.72</td></tr><tr><td>ChatUniVi [35]</td><td>46.43</td><td>66.67</td><td>30.00</td><td>49.75</td></tr><tr><td>LLaVA-NeXTVideo-7B [21, 47]</td><td>51.79</td><td>56.79</td><td>36.67</td><td>49.24</td></tr><tr><td>VideoChatFlash [18]</td><td>46.43</td><td>46.91</td><td>46.67</td><td>46.70</td></tr><tr><td>VideoLLaMA3-7B [2, 6, 46]</td><td>46.43</td><td>50.62</td><td>33.33</td><td>44.16</td></tr><tr><td>LongVA-7B [40]</td><td>44.64</td><td>64.20</td><td>38.33</td><td>50.76</td></tr><tr><td>BIMBA-LLaVA [13]</td><td>57.14</td><td>71.60</td><td>36.67</td><td>58.38</td></tr><tr><td>mPLUG-Owl3 [43]</td><td>57.79</td><td>71.60</td><td>38.33</td><td>55.84</td></tr><tr><td>VideoForest（我们的）</td><td>71.93</td><td>83.75</td><td>51.67</td><td>69.12</td></tr></table>

<table><tr><td rowspan=1 colspan=2>模型</td><td rowspan=1 colspan=1>跨时间</td><td rowspan=1 colspan=1>跨空间</td><td rowspan=1 colspan=1>跨时空</td><td rowspan=1 colspan=1>单一</td></tr><tr><td rowspan=11 colspan=2>ShareGPT4Video-8B [4]VideoCCAM-7B [9]InternVL-2.5 [38]LLaVAOneVision [15]ChatUniVi [35]LLaVA-NeXTVideo-7B [21, 47]VideoChatFlash [18]VideoLLaMA3-7B [2, 6, 46]LongVA-7B [40]BIMBA-LLaVA [13]mPLUG-Owl3s [43]</td><td rowspan=1 colspan=1>64.00</td><td rowspan=1 colspan=1>38.64</td><td rowspan=1 colspan=1>53.85</td><td rowspan=1 colspan=1>42.31</td></tr><tr><td rowspan=1 colspan=1>60.00</td><td rowspan=1 colspan=1>46.15</td><td rowspan=1 colspan=1>50.00</td><td rowspan=1 colspan=1>57.69</td></tr><tr><td rowspan=1 colspan=1>52.00</td><td rowspan=1 colspan=1>46.15</td><td rowspan=1 colspan=1>69.23</td><td rowspan=1 colspan=1>73.08</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>36.00</td><td rowspan=1 colspan=1>61.54</td><td rowspan=1 colspan=1>53.85</td><td rowspan=1 colspan=1>50.00</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>64.00</td><td rowspan=1 colspan=1>38.46</td><td rowspan=1 colspan=1>38.46</td><td rowspan=1 colspan=1>57.69</td></tr><tr><td rowspan=1 colspan=1>56.00</td><td rowspan=1 colspan=1>42.31</td><td rowspan=1 colspan=1>65.38</td><td rowspan=1 colspan=1>42.31</td></tr><tr><td rowspan=1 colspan=1>52.00</td><td rowspan=1 colspan=1>46.15</td><td rowspan=1 colspan=1>34.62</td><td rowspan=1 colspan=1>53.85</td></tr><tr><td rowspan=1 colspan=1>48.00</td><td rowspan=1 colspan=1>46.15</td><td rowspan=1 colspan=1>42.31</td><td rowspan=1 colspan=1>46.15</td></tr><tr><td rowspan=1 colspan=1>52.00</td><td rowspan=1 colspan=1>50.00</td><td rowspan=1 colspan=1>34.62</td><td rowspan=1 colspan=1>50.00</td></tr><tr><td rowspan=1 colspan=1>48.00</td><td rowspan=1 colspan=1>46.15</td><td rowspan=1 colspan=1>69.23</td><td rowspan=1 colspan=1>53.85</td></tr><tr><td rowspan=1 colspan=1>68.00</td><td rowspan=1 colspan=1>46.15</td><td rowspan=1 colspan=1>65.38</td><td rowspan=1 colspan=1>53.85</td></tr><tr><td rowspan=1 colspan=2>VideoForest（我们的）</td><td rowspan=1 colspan=1>72.00</td><td rowspan=1 colspan=1>69.23</td><td rowspan=1 colspan=1>65.38</td><td rowspan=1 colspan=1>61.54</td></tr></table>

Table 2: Performance comparison across the four evaluation modalities in CrossVideoQA.   

# 4.4 性能分析

4.4.1 任务特定性能分析。表1呈现了我们基准测试中三个基本推理类别的综合评估。VideoForest在所有评估维度上显示出统计显著的性能优势，尤其是在人员识别方面（比最强基线提高了 $ + 1 3 . 0 0 \%$）和行为分析（$ + 1 7 . 0 8 \% $）。这些持续的性能差异验证了我们层次化、以人为中心的推理架构在解决跨视频理解挑战中的有效性。特别值得注意的是，即使是像InternVL-2.5这样在传统VideoQA任务中取得竞争性结果的最先进单视频模型，在跨视频人员识别场景中表现出显著的性能下降（ $5 8 . 9 3 \%$ 相比我们的 $7 1 . 9 3 \%$）。这一性能差距突显了明确的人员级跟踪和再识别组件在跨时间和空间分散视频片段中维持身份一致性的重要性。结果提供了令人信服的证据，表明VideoForest的多层次推理方法有效应对了跨视频理解中的基本挑战。

4.4.2 跨时空配置的评估。表2呈现了四种时空配置下的性能分析。VideoForest在跨时序推理方面的准确率为$72.00\%$，比ShareGPT4Video高出$8.00\%$，展示了我们层级树结构在建立时间关系方面的有效性。在跨空间整合方面，我们的模型达到了$69.23\%$的准确率，超过了LLaVA-OneVision的$61.54\%$，高出$7.69\%$，验证了我们基于人的锚定方法来连接跨空间边界信息的有效性。VideoForest在所有配置中保持了强劲的表现，尽管在跨时空任务中差距最小，这表明该领域仍然具有挑战性，并暗示了未来的研究方向。现有模型显示出专业化模式——InternVL 2.5在单视频任务中表现优异（$73.08\%$），但在跨时序方面表现不佳（$52.00\%$），而ShareGPT4Video则显示出相反的模式。VideoForest在所有配置中展示了均衡的性能。

# 4.5 定性分析

图4展示了VideoForest的推理过程和响应生成的示例。我们的定性分析揭示了两个关键模式：1) VideoForest有效地使用两阶段推理处理跨视频查询——首先从单个视频树中检索相关信息，然后综合生成连贯的答案；2) 主要失败模式涉及细粒度动作识别，模型无法识别诸如在纸上书写等详细动作，原因是监控视频分辨率有限、帧采样限制以及复杂环境中的动作模糊性。

# 4.6 消融研究

我们进行了消融研究，以量化VideoForest中关键组件的贡献。表3展示了在移除单个组件时对性能的影响。结果表明，每个组件对整体性能都有显著贡献。禁用知识库检索使性能下降了$1 0 { - } 2 5 \%$，特别影响了需要上下文依赖和精确知识的跨时空场景。消除反射组件对时空推理的影响最大，跨时间场景中下降最为显著（约$3 3 \%$）。这些结果验证了我们的架构设计，并突显了组件之间的互补性，以实现有效的跨视频推理。我们对视频树搜索过程中与核心机制对应的三个关键模块进行了定量消融研究：

![](images/4.jpg)  

Figure 4: Exemplars from CrossVideoQA illustrating VideoForest's multi-modal reasoning architecture

Table 3: Ablation study of VideoForest across four settings.   

<table><tr><td>设置</td><td>无检索</td><td>无反思</td><td>完整模型</td></tr><tr><td>跨时间</td><td>60.00</td><td>48.00</td><td>72.00</td></tr><tr><td>跨空间</td><td>61.54</td><td>57.69</td><td>69.23</td></tr><tr><td>跨空间-时间</td><td>50.00</td><td>46.15</td><td>65.38</td></tr><tr><td>单一</td><td>50.00</td><td>42.31</td><td>61.54</td></tr><tr><td>平均</td><td>55.39</td><td>48.54</td><td>67.54</td></tr></table>

<table><tr><td>设置</td><td>无搜索中的ReID</td><td>无视频过滤</td><td>无深度树遍历</td><td>完整模型</td></tr><tr><td>跨时间</td><td>52.00</td><td>44.00</td><td>60.00</td><td>72.00</td></tr><tr><td>跨空间</td><td>53.85</td><td>61.54</td><td>65.38</td><td>69.23</td></tr><tr><td>跨空间-时间</td><td>57.69</td><td>38.46</td><td>61.54</td><td>65.38</td></tr><tr><td>单一</td><td>50.00</td><td>61.54</td><td>57.69</td><td>61.54</td></tr><tr><td>平均</td><td>53.39</td><td>51.39</td><td>61.15</td><td>67.54</td></tr></table>

•无搜索中的ReID：人物轨迹构建树结构，但在推理时不使用人物ID进行节点过滤，评估ReID作为锚信号的影响。 •无视频过滤：所有视频直接传递给下游代理，没有根据问题内容进行预过滤。 •无深度树遍历：仅保留第一层节点（粗粒摘要），排除更深的细粒度细节。

Table 4: Ablation study on structural design choices in the video tree search module of VideoForest.   

表4展示了这三种机制的结果。移除任何模块都会导致准确率的持续下降，证明了它们不可或缺的角色。禁用基于ReID的过滤导致跨时间设置中准确率下降$20.00\%$，平均降低$14.15\%$。移除视频过滤在多跳场景中造成了明显的降级——在跨时空任务中高达$26.92\%$，突显了它在减少无关搜索空间中的重要性。限制树深度导致平均下降$6.39\%$，显示出细节的低层信息对细粒度推理的重要贡献。值得注意的是，在没有视频过滤的设置中，下游代理偶尔仍会通过自然语言理解选择相关片段，这暗示了多智能体架构中的容错性和适应性。

# 5 结论

本文介绍了VideoForest，一个新颖的层次框架，用于跨视频问答，解决了在多个视频源中整合和推理分散信息的基本挑战。通过以人物级特征为自然的桥梁点，我们能够进行复杂的跨视频推理，而无需在多视频数据集上进行端到端训练。我们开发了CrossVideoQA，一个专门为以人为中心的跨视频分析设计的综合基准，并展示了VideoForest在从单视频到跨时空配置的最新模型中的性能优势。我们的基于树的架构和多智能体推理为跨视频理解奠定了基础，克服了孤立视频处理的限制，同时保持了实际应用中的计算可行性。

# 致谢

本工作得到了中国国家自然科学基金（编号：62176223，62302535，62371305）的支持，以及广东省基础与应用基础研究基金的部分支持（2023A1515011639，2024A1515030025）。