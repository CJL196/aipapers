# PaLM-E：一种具身的多模态语言模型

Danny Driess 1 2 Fei Xia 1 Mehdi S. M. Sajjadi 3 Corey Lynch 1 Aakanksha Chowdhery 3 Brian Ichter 1 Ayzaan Wahid 1 Jonathan Tompson 1 Quan Vuong 1 Tianhe ${ \bf { V } } { \bf { u } } ^ { 1 }$ Wenlong Huang 1 Yevgen Chebotar1 Pierre Sermanet 1 Daniel Duckworth 3 Sergey Levine 1 Vincent Vanhoucke1 Karol Hausman 1 Marc Toussaint 2 Klaus Greff 3 Andy Zeng 1 Igor Mordatch 3 Pete Florence 1 1谷歌机器人 2柏林工业大学 3谷歌研究 https://palm-e.github.io

![](images/1.jpg)  
l green and blue) are inserted alongside text tokens (in orange) as input to an LLM, trained end-to-end.

# 摘要

# 1. 引言

大型语言模型已被证明能够执行复杂任务。然而，在现实世界中实现一般推理，特别是在机器人问题上，提出了基础问题的挑战。我们提出了一种具身语言模型，直接将现实世界的连续传感器模态融入语言模型，从而建立词语与感知之间的联系。我们具身语言模型的输入是多模态句子，这些句子交织了视觉、连续状态估计和文本输入编码。我们对这些编码进行端到端训练，与预训练的大型语言模型结合，应用于多个具身任务，包括顺序机器人操作规划、视觉问答和图像描述。我们的评估显示，PaLM-E，一个单一的大型具身多模态模型，可以应对多种具身推理任务，涵盖多种观察模态，并可在多种具身表现上展现积极的迁移：该模型受益于跨互联网规模的语言、视觉和视觉-语言领域的多样联合训练。我们最大的模型PaLM-E-562B具有562亿参数，除了在机器人任务上进行训练外，还是一款具有视觉-语言通用能力的模型，在OK-VQA上表现出最先进的性能，并随着规模的增加保持通用语言能力。

大型语言模型（LLM）在包括对话（Glaese et al., 2022；Thoppilan et al., 2022）、逐步推理（Wei et al., 2022；Kojima et al., 2022）、数学问题解决（Lewkowycz et al., 2022；Polu et al., 2022）和代码编写（Chen et al., 2021a）等多个领域显示出强大的推理能力。然而，这些模型在现实世界推理中的一个局限性是建构问题：虽然在大量文本数据上训练LLM可能导致与我们物理世界相关的表征，但将这些表征与现实世界的视觉和物理传感器模态连接起来，对于解决计算机视觉和机器人领域更广泛的实际问题至关重要（Tellex et al., 2020）。之前的研究（Ahn et al., 2022）将LLM的输出与学习的机器人策略和可赐性函数进行接口，以作出决策，但其局限在于LLM本身仅接收文本输入，这对于许多需要场景几何配置的重要任务而言是不够的。此外，在我们的实验中，我们表明当前在典型视觉-语言任务（如视觉问答（VQA））上训练的最先进视觉-语言模型无法直接解决机器人推理任务。本文提出了具身语言模型，

# 视觉条件笑话 $^ +$ 少量提示

![](images/2.jpg)

I see (image omitted). Description: A cow sitting on a rug. Joke: Why is the cow sitting on a rug? Because it wants to be a cow-ch! .. I see <img>. Description: a donkey is carrying a dog, cat, and rooster. Joke: what do you call a donkey with a rooster on his back. A rooster booster.

# 零样本多模态推理链

Given <img>. Q: Can I go down this street on a bicycle, yes or no? A: Let's think step by step. 1. do not enter. 2. except bicycles. 3. do not entry except bicycles. 4. yes.

# 机器人计划

![](images/3.jpg)

Given <img>. Q: If a robot wanted to be useful here, what steps should it take? A: clean table..pick up trash.. pick up chairs.. wipe chairs.. put chairs down.

![](images/4.jpg)

![](images/5.jpg)

Given <img>. Q: How many championship rings did the player on the left win in his career A: Let's think step by step. 1. kobe bryant is on the left. 2. kobe bryant has won 5 championship rings.

零样本：多图像关系 P5

# 机器人视觉感知、对话与规划

在这个场景中，<img1> 在炊具上，炊具上放着一个锅。

Given <img>. Q: Who are the two teams playing in this photo, and which was the last to win a championship, and which year did they win, and who was their star player that year? A: Let's think step by step. 1. team in white is new

![](images/6.jpg)

1. 纽约尼克斯。 2. 绿色球队是波士顿凯尔特人。 3. 最近赢得冠军的是波士顿凯尔特人。 4. 年份是2008年。 5. 明星球员是保罗·皮尔斯。 6. 我不太确定这是否正确。我不是篮球迷，我只是谷歌搜索了一下。

Robot: I am a robot operating in a kitchen. Given <img>, what do I see? a bowl of eggs, a bowl of flour, and a bowl of sugar.. and a spatula.. and a spoon.. and a bag of flour.. and a bag of sugar. When a human asks me to do a task, I will respond with the sequence of actions I would do to accomplish the task with only the items I see. Human: Use all of the ingredients you see to make a cake batter. 1. crack egg. 2. put egg in bowl. 3. put flour in bowl.  put sugar in bowl. 5.mix.  put in pan.7. bake. 8.eat. clean up.

# 物理预测

![](images/7.jpg)

Given <img> Q: What will the robot do next? A: fall.

![](images/8.jpg)

# 空间基础

Given <img>. Q: What flavor is the doughnut on the right? Strawberry.. Q: What flavor is the doughnut on the left? A blueberry.

![](images/9.jpg)

照片1：<img1>。照片2：<img2>。问：照片1中有什么，而照片2中没有？答：太阳镜。照片1：<img1>。照片：<img2>。：照片1中有什么，而照片2中没有？答：让我们一步一步思考并解释。照片1有折叠衣物上的太阳镜。照片2没有折叠衣物上的太阳镜。因此，太阳镜在照片1中而不在照片2中。端到端自我中心问答，对话。

# 无需OCR的数学推理 ocReate 你自己的 kiz asparagu

# Given <img>.

问：我和我的朋友只点了两份定制披萨。我总共应该支付多少钱？答：让我们一步步来。第一步：2份定制披萨。第二步：2 x \$9.99。第三步：\$19.98。照片1，上午10:30：<img1>。照片2，下午12:45：<img2>。照片3，下午3:45：<img3>。问：我忘了，午餐吃了什么，是什么时候？答：让我们一步步来。1. 你午餐吃了三明治。2. 时间是下午12:45。

![](images/10.jpg)

直接整合来自具身智能体的传感器模态的连续输入，从而使语言模型能够为现实世界的顺序决策提供更为扎实的推理。图像和状态估计等输入与语言词元嵌入到同一潜在嵌入中，并由基于Transformer的语言模型的自注意力层以与文本相同的方式处理。我们从一个预训练的语言模型开始，通过编码器注入连续输入。这些编码器被端到端训练，以自然文本的形式输出顺序决策，能够通过调节低级策略或回答具身问题让具身智能体理解。我们在多种设置中评估该方法，比较不同的输入表示（例如，标准与以对象为中心的视觉输入ViT编码），冻结与微调语言模型时的编码器训练，以及调查多任务联合训练是否能够实现迁移。为了探讨该方法的广度，我们在三个机器人操作领域进行评估（其中两个是在现实世界的闭环中），标准的视觉-语言任务如视觉问答和图像描述，以及语言任务。我们的结果表明，相较于单任务训练，多任务训练能提升性能。我们展示了跨任务的迁移能够在机器人任务中实现高数据效率，例如，通过少量的训练实例显著提高学习成功率，甚至展示出对新物体组合或未见物体的一次性或零次性泛化能力。

我们将PaLM-E扩展至562亿参数，整合了540亿的PaLM（Chowdhery et al., 2022）大语言模型和22亿的视觉变换器（ViT）（Dehghani et al., 2023），成为截至我们所知的最大视觉-语言模型。PaLM-E-562B在OK-VQA（Marino et al., 2019）基准上实现了最先进的性能，而不依赖于特定任务的微调。尽管这并不是我们实验的重点，我们还发现（图2）PaLM-E-562B表现出广泛的能力，包括零-shot多模态推理链（CoT）、few-shot提示、无OCR数学推理和多图像推理，尽管它仅在单图像示例上进行了训练。零-shot CoT（Kojima et al., 2022），最初是一个仅限语言的概念，已经在具有任务特定程序的多模态数据上得到验证（Zeng et al., 2022），但我们所知的并未通过端到端模型实现。总结我们的主要贡献，我们（1）提出并展示了一种通用、迁移学习的多体决策智能体可以通过将具身数据融入多模态大语言模型的训练中进行训练。我们展示了，（2）尽管当前最先进的通用视觉-语言模型在零-shot状态下未能很好地解决具身推理问题，但可以训练出一个称职的通用视觉-语言模型，同时也是一个高效的具身推理者。在研究如何最好地训练此类模型时，我们（3）引入了新颖的架构理念，如神经场景表示和实体标记多模态词元。最后，除了专注于PaLM-E作为具身推理者之外，我们（4）展示了PaLM-E在视觉和语言方面也是量化能力强的通才，并且（5）证明扩展语言模型的规模可以实现多模态微调，同时减少灾难性遗忘。

# 2. 相关工作

通用视觉-语言建模。基于在大型语言模型（Brown et al., 2020; Devlin et al., 2018）和视觉模型（Dosovitskiy et al., 2020）上的成功，近年来对大型视觉-语言模型（VLMs）的兴趣日益增长（Li et al., 2019; Lu et al., 2019; Hao et al., 2022; Gan et al., 2022）。与其前身不同，VLM能够同时理解图像和文本，并可以应用于诸如视觉问答（Zhou et al., 2020; Zellers et al., 2021b）、图像描述（Hu et al., 2022）、光学字符识别（Li et al., 2021）和目标检测（Chen et al., 2021b）等任务。图像整合的方法有所不同。例如，Alayrac et al.（2022）通过直接关注单个上下文图像的机制增强了预训练语言模型。相比之下，PaLM-E将图像和文本表示为潜在向量的“多模态句子”，使其能够在句子的任何部分灵活处理多个图像。与我们的工作更为相关的是Frozen（Tsimpoukelli et al., 2021），其中视觉编码器参数通过反向传播通过一个冻结的LLM（Lu et al., 2021）进行优化。受到这项工作的启发，我们通过引入替代输入模态（例如神经场景表示）在更广泛的范围内研究设计，我们提出的方法在VQAv2基准上实证性能超过Frozen超过$45\%$。更重要的是，我们证明了PaLM-E不仅适用于感知任务，还适用于体现任务。

动作输出模型。以往的研究集中于在具身环境中结合视觉和语言输入，旨在进行直接的动作预测（Guhur et al., 2022; Shridhar et al., 2022b;a; Zhang & Chai, 2021; Silva et al., 2021; Jang et al., 2022; Nair et al., 2022; Lynch et al., 2022; Brohan et al., 2022）。在这些方法中，VIMA（Jiang et al. 2022）探索了类似于PaLM-E的多模态提示。在这些研究中，语言的角色或许最准确地被描述为任务规范。相比之下，PaLM-E生成高层次的文本指令；通过这样做，模型能够自然地依赖于自己的预测，并直接利用嵌入在其参数中的世界知识。这不仅赋能具身推理，还能实现问答功能，如我们实验中所展示的。在输出动作的研究中，与之最为相似的方法可能是Gato中提出的（Reed et al., 2022），该方法与PaLM-E一样，是一个通用的多具身智能体。与Gato不同，我们展示了不同任务之间的正向迁移，模型得益于在多个领域的多样化联合训练。

在具身任务规划中使用大型语言模型（LLMs）。已经提出几种方法来利用LLMs在具身领域中。虽然许多研究工作重点在于理解自然语言目标（Lynch & Sermanet, 2020；Shridhar et al., 2022a；Nair et al., 2022；Lynch et al., 2022），但考虑自然语言作为规划表示的研究相对较少。本研究的重点就在于此。LLMs包含大量关于世界的内化知识（Bommasani et al., 2021），但没有具体依据生成的计划可能无法执行。一条研究方向采用提示从LLM中直接引出一系列指令，方法包括利用LLM生成与一组合适指令之间的语义相似性（Huang et al., 2022b）、整合可供性函数（Ahn et al., 2022）、视觉反馈（Huang et al., 2022c）、生成世界模型（Nottingham et al., 2023；Zellers et al., 2021a）、在图和地图上规划（Shah et al., 2022；Huang et al., 2022a）、视觉解释（Wang et al., 2023）、程序生成（Liang et al., 2022；Singh et al., 2022），或者向提示中注入信息（Zeng et al., 2022）。相较之下，PaLM-E被训练为直接生成计划，而不依赖于辅助模型进行具体化。这反过来使得可以将预训练LLMs中存储的丰富语义知识直接整合到规划过程中。除少数例外，许多研究中使用的LLMs的参数都未经过进一步训练而直接应用。在LID（Li et al., 2022）中，这一限制得以放宽，LLM参数经过微调以生成用于产生高层指令的规划网络。$\mathrm { ( S L ) ^ { 3 } }$（Sharma et al., 2021）则更加复杂地同时微调两个LLM：一个用于生成高层指令的规划网络和一个选择动作的低层策略网络。借助PaLM-E，我们的研究兴趣是独特且互补的：我们研究一个通用的多具身模型，跨越多个模态。

# 3. PaLM-E：一个具身多模态语言模型

PaLM-E 的主要架构思想是将连续的、具身的观察数据，如图像、状态估计或其他传感器模态，注入到预训练语言模型的语言嵌入空间中。这是通过将连续观察数据编码为与语言标记的嵌入空间相同维度的向量序列来实现的。因此，连续信息以类似于语言标记的方式被注入到语言模型中。PaLM-E 是一个仅解码的语言模型，能够在给定前缀或提示时自回归生成文本补全。我们称我们的模型为 PaLM-E，因为我们使用 PaLM（Chowdhery et al., 2022）作为预训练语言模型，并使其具身化。

The inputs to PaLM-E consist of text and (multiple) continuous observations. The multimodal tokens corresponding to these observations are interleaved with the text to form multi-modal sentences. An example of such a multi-modal sentence is $\mathsf Q$ : What happened between <img. $\beth$ and $< \mathrm { i } \mathrm { m } 9 . 2 > ?$ where $< \mathrm { i } \mathrm { m } 9 . \dot { \imath } >$ represents an embedding of an image. The output of PaLM-E is text generated auto-regressively by the model, which could be an answer to a question, or a sequence of decisions produced by PaLM-E in textual form that should be executed by a robot. When PaLM-E is tasked with producing decisions or plans, we assume that there exists a low-level policy or planner that can translate these decisions into low-level actions. Prior work has discussed a variety of ways to train such low-level policies (Lynch & Sermanet, 2020; Brohan et al., 2022), and we use these prior methods directly without modification. In the following, we describe our approach more formally.

仅解码器大语言模型。仅解码器的大语言模型（LLMs）是生成模型，旨在预测一段文本 $w _ { 1 : L }$ 的概率 $p ( w _ { 1 : L } )$，该文本表示为一个词元序列 $w _ { 1 : L } = ( w _ { 1 } , \dots , w _ { L } )$，其中 $w _ { i } \in \mathcal W$。典型的神经网络结构通过将其分解为 $p _ { \mathrm { L M } }$，这是一种大型变换器网络。

$$
p ( w _ { 1 : L } ) = \prod _ { l = 1 } ^ { L } p _ { \mathrm { L M } } ( w _ { l } | w _ { 1 : l - 1 } ) ,
$$

前缀解码器专用的大语言模型。由于该大语言模型是自回归的，因此可以在不改变架构的情况下，对预训练模型进行前缀 $w _ { 1 : n }$ 的条件设置。

$$
p ( w _ { n + 1 : L } | w _ { 1 : n } ) = \prod _ { l = n + 1 } ^ { L } p _ { \mathrm { L M } } ( w _ { l } | w _ { 1 : l - 1 } ) .
$$

前缀或提示 $w _ { 1 : n }$ 提供了上下文，LLM 基于此继续预测后续的词元 $w _ { n + 1 : L }$。这通常用于推理，以引导模型的预测。例如，提示可以包含 LLM 应该解决的任务描述或类似任务的期望文本完成示例。

词元嵌入空间。词元 $w _ { i }$ 是固定词汇表 $\mathcal { W }$ 的元素，该词汇表是一个对应于自然语言中的（子）词的离散有限集合。在内部，LLM 将 $w _ { i }$ 嵌入到词元嵌入空间 $\mathcal { X } \subset \mathbb { R } ^ { k }$ 中，通过映射 $\gamma : \mathcal { W } \to \mathcal { X }$，即 $p _ { \mathrm { L M } } ( w _ { l } | \boldsymbol { x } _ { 1 : l - 1 } )$，其中 $x _ { i } = \gamma ( w _ { i } ) \in \mathbb { R } ^ { k }。映射 $\gamma$ 通常表示为一个大小为 $k \times | \mathcal { W } |$ 的大型嵌入矩阵，并进行端到端训练。在我们的案例中，$| \mathcal { W } | = 256000$（Chowdhery 等，2022）。多模态句子：连续观察的注入。可以通过跳过离散词元级别，将多模态信息（例如图像观察）直接映射到语言嵌入空间 $\mathcal { X }$ 中，从而将其注入到 LLM 中。为此，我们训练一个编码器 $\phi : \mathcal { O } \to \mathcal { X } ^ { q }$，将（连续）观察空间 $\mathcal { O }$（详细信息见第4节）映射为一系列 $q$ 个向量在 $\mathcal { X }$ 中。这些向量随后与正常嵌入的文本词元交错，以形成 LLM 的前缀。这意味着前缀中的每个向量 $x _ { i }$ 是由词元嵌入器 $\gamma$ 或编码器 $\phi _ { i }$ 所形成的。

$$
x _ { i } = \left\{ \begin{array} { l l } { \gamma ( w _ { i } ) } & { \mathrm { i f ~ } i \mathrm { ~ a ~ i s ~ t e x t ~ t o k e n , ~ o r ~ } } \\ { \phi _ { j } ( O _ { j } ) _ { i } } & { \mathrm { i f ~ } i \mathrm { ~ c o r r e s p o n d s ~ t o ~ o b s e r v a t i o n ~ } O _ { j } . } \end{array} \right.
$$

注意，单个观察 $O _ { j }$ 通常被编码为多个嵌入向量。可以在前缀的不同位置交错不同的编码器 $\phi _ { i }$，以结合来自不同观察空间的信息。通过这种方式将连续信息注入LLM中，重新利用其现有的位置信息编码。与其他视觉语言模型（例如，Chen 等，2022）的方法不同，观察嵌入并不是以固定的位置插入，而是动态地放置在周围文本中。将输出体现：PaLM-E 在机器人控制回路中。PaLM-E 是一种生成模型，根据多模态句子作为输入生成文本。为了将模型的输出与体现连接起来，我们区分两种情况。如果任务仅通过输出文本即可完成，例如在体现式问答或场景描述任务中，则模型的输出被直接视为任务的解决方案。或者，如果 PaLM-E 被用于解决体现式规划或控制任务，则它生成的文本用于条件低级命令。特别地，我们假设可以访问能够从某个（小）词汇表中执行低级技能的策略，并且来自 PaLM-E 的成功计划必须由这些技能的序列组成。请注意，PaLM-E 必须根据训练数据和提示自行确定哪些技能可用，且不使用其他机制来约束或过滤其输出。尽管这些策略是语言条件的，但它们无法解决长时间跨度的任务或处理复杂指令。因此，PaLM-E 被集成到一个控制回路中，其预测的决策通过机器人执行低级策略，从而产生新的观察，PaLM-E 能够根据需要重新规划。在这个意义上，PaLM-E 可以被理解为一种高级策略，它对低级策略进行排序和控制。

# 4. 不同传感器模态的输入与场景表示

在本节中，我们描述了纳入 PaLM-E 的个体模态以及如何设置它们的编码器。我们为每个编码器 $\phi : \mathcal { O } \to \mathcal { X }$ 提出了不同的架构选择，以将相应的模态映射到语言嵌入空间。我们研究了状态估计向量、用于 2D 图像特征的视觉变换器（ViTs）（Dosovitskiy et al., 2020; Chen et al., 2022; Ry00 et al., 2021）以及 3D 认知物体场景表示变换器（OSRT）（Sajjadi et al., 2022a）。除了表示输入场景全局的编码器外，我们还考虑了对象中心的表示，它将观测划分为表示场景中各个对象的词元。状态估计向量。状态向量，例如来自机器人或对象的状态估计，或许是最简单输入到 PaLM-E 的形式。设 $\boldsymbol { s } \in \mathbb { R } ^ { S }$ 为描述场景中对象状态的向量。例如，$s$ 可以包含这些对象的姿态、大小、颜色等信息。然后，MLP $\phi _ { \mathrm { s t a t e } }$ 将 $s$ 映射到语言嵌入空间。

视觉变换器（ViT）。ViT $\tilde { \phi } _ { \mathrm { V i T } }$（Dosovitskiy 等，2020）是一种将图像 $I$ 映射为一系列词元嵌入 $\tilde { x } _ { 1 : m } \ = \ \tilde { \phi } _ { \mathrm { v i T } } ( \bar { I } ) \ \in$ $\mathbb { R } ^ { m \times \tilde { k } }$ 的变换器架构。我们考虑几种变体，包括来自 Chen 等（2022） 的 40 亿参数模型，称为 ViT-4B，以及一个相似的 220 亿参数模型 ViT22B（Dehghani 等，2023），两者均已在图像分类上进行了预训练。我们进一步研究 ViT 词元学习器架构 $\mathrm { ( V i T + T L ) }$（Ryoo 等，2021），该模型从头开始全程端到端训练。值得注意的是，ViT 嵌入的维度 $\tilde { k }$ 不一定与语言模型的维度相同。因此，我们将每个嵌入投影为 $x _ { i } = \overset { \vartriangle } { \phi _ { \mathrm { V i T } } } ( I ) _ { i } = \psi ( \widetilde { \phi } _ { \mathrm { V i T } } ( I ) _ { i } )$，其中 $\psi$ 是学习的仿射变换。面向对象的表示。与语言不同，视觉输入并不是预先结构化为有意义的实体和关系：尽管 ViT 可能捕捉语义，但其表示的结构更像是静态网格，而不是对象实例的集合。这对与已经在符号上进行预训练的 LLM 的接口以及解决需要与物理对象交互的具体推理构成挑战。因此，我们还探讨了结构化编码器，旨在在将视觉输入注入 LLM 之前将其分离为不同的对象。给定真实标注的对象实例掩码 $M _ { j }$，我们可以将 ViT 的表示分解为 $x _ { 1 : m } ^ { j } = \phi _ { \mathrm { V i T } } \bar { ( M _ { j } \circ I ) }$，用于对象 $j$。

对象场景表示变换器（OSRT）。一种不需要真实标注分割的替代方案是OSRT（Sajjadi等，2022a）：它不依赖于关于对象的外部知识，而是通过架构中的归纳偏差以无人监督的方式发现这些对象（Locatello等，2020）。基于SRT（Sajjadi等，2022b），OSRT通过一种新颖的视图合成任务在领域内数据上学习3D中心的神经场景表示。其场景表示由对象槽组成 $o _ { j } = \bar { \phi } _ { 0 \mathrm { S R T } } ( I _ { 1 : v } ) _ { j } \in \bar { \mathbb R } ^ { \bar { k } }$。我们将每个槽投影到 $x _ { 1 : m } ^ { j } = \psi ( \bar { \phi } _ { \mathrm { O S R T } } ( I _ { 1 : v } ) _ { j } )$，其中$\psi$是一个多层感知器（MLP）。注意，单个对象始终被标记为多个嵌入，即$\psi : \mathbb { R } ^ { \bar { k } } \mathbb { R } ^ { m \times k }$，对于OSRT映射到$m$个嵌入。

实体引用。对于具身规划任务，PaLM-E 必须能够在生成的计划中引用对象。在许多情况下，包括我们大多数实验中，场景中的对象可以通过其一些独特属性用自然语言进行标识。然而，也存在一些情境，在这些情境下对象不容易用简洁的语言进行识别，例如，当桌子上有多个同色的积木分布在不同位置时。对于以对象为中心的表示，如 OSRT，我们为输入提示中对应对象的多模态词元标记如下：对象 1 是 ${ < } \mathrm { o b j - } 1 >$ ... 对象 $j \mathrm { i } s < \infty \mathrm { j } \lrcorner j >$。这使得 PaLM-E 能够通过特殊的形式为 obj.$j$ 的词元在其生成的输出句子中引用对象。在这种情况下，我们假设低级策略也在这些词元上操作。

# 5. 训练方案

PaLM-E的训练数据集形式为 $\begin{array} { r l } { D } & { { } = } \end{array}$ $\left\{ \left( I _ { 1 : u _ { i } } ^ { i } , w _ { 1 : L _ { i } } ^ { i } , n _ { i } \right) \right\} _ { i = 1 } ^ { N }$，每个样本$i$包含$u _ { i }$个连续观测$I _ { j } ^ { i }$、文本$w _ { 1 : L _ { i } } ^ { i }$和索引$n _ { i }$。尽管是仅解码器模型，文本的前缀部分直到索引$n _ { i }$ 是由多模态句子构成的，而预测目标仅包含文本词元。因此，损失函数是基于各个非前缀词元$ni+1:Li$的交叉熵损失的平均值。为了在模型中形成多模态句子，我们在文本中使用特殊词元，这些词元将在文本中被编码器的位置的嵌入向量所替代。我们基于预训练的8B、62B和540B参数版本的PaLM，作为仅解码器的大语言模型，在其输入编码器中注入连续观测。这些编码器可以是预训练的，也可以是从头开始训练的，见第4节。我们将8B的大语言模型与4B的视觉变换器结合称为PaLM-E12B，将62B的大语言模型与22B的视觉变换器结合称为PaLM-E-84B，将540B的大语言模型与22B的视觉变换器结合称为PaLM-E-562B。

模型冻结的变体。我们的大多数架构由三个部分组成：编码器 $\tilde { \phi }$ ，投影器 $\psi$ 和LLM $p _ { \mathrm { L M } }$。在训练PaLM-E时，一种方法是更新所有这些组件的参数。然而，LLM在提供合适的提示时表现出令人印象深刻的推理能力（Wei et al., 2022）。因此，我们探讨冻结LLM并仅训练输入编码器是否可行，以及不同模态编码器的比较。在这种情况下，编码器必须生成嵌入向量，使得冻结的LLM能够基于观察进行推理，并且能够向LLM传递关于具体实例能力的信息。训练这种编码可以被理解为一种输入条件的软提示形式（Tsimpoukelli et al., 2021），与正常的软提示（Lester et al., 2021）相关。在对$\phi _ { \mathrm { O S R T } }$的实验中，我们还冻结了槽表示，即我们仅更新作为OSRT与LLM之间接口的小型投影器$\psi$。跨任务共训练。在我们的实验中，我们研究在各种多样的数据上共训练模型的效果。“完整混合”，参见附录A，主要由一组多样的互联网规模的视觉和语言数据组成，来自各种任务。采样频率设置为仅$8 . 9 \%$的完整混合为具象数据，并且每个具象都有几个任务。

# 6. 实验

我们的实验考虑了三种不同机器人形态的多样化机器人（移动）操控任务，包括仿真和两种不同的真实机器人。我们参考 https://palm-e.github.io 以观看 PaLM-E 在这些任务中的能力展示。尽管这不是我们工作的重点，我们也在视觉问答（VQA）、图像描述和已建立的语言建模任务等通用视觉语言任务上评估了 PaLM-E。

我们将实验研究分为两个大类。首先，我们比较第4节中不同输入表示在性能、泛化能力和数据效率方面的表现。第二类实验关注于一个架构，即主要的PaLM-E版本，它由预训练的ViT和PaLM语言模型构成，接受原始图像作为连续输入。我们在这里展示了一个模型，训练在多种数据集的混合上，跨越不同任务和机器人体现，可以在所有这些任务上同时实现高性能。关键是，我们研究在这些数据集上的共同训练是否能够实现迁移（图3）：尽管任务和体现不同，但通过训练任务的混合，个别任务的性能得以提升。我们研究了共同训练策略和模型参数大小对性能、泛化能力和数据效率的影响。最后，我们考虑冻结大型语言模型（LLM），仅训练将视觉注入到LLM中的ViT是否是一条可行的路径

![](images/11.jpg)  

Figure 3: Overview of transfer learning demonstrated by PaLME: across three different robotics domains, using PaLM and ViT pretraining together with the full mixture of robotics and general visual-language data provides a significant performance increase compared to only training on the respective in-domain data. See Tab. 1, Fig. 4, Tab. 2, Tab. 4 for additional data in each domain.

# 6.1. 机器人环境/任务

我们的三个机器人环境（图1）包括一个任务与运动规划（TAMP）领域，在该领域中，机器人需要操控（抓取和堆叠）物体，一个桌面推送环境，以及一个移动操控领域。在每个领域中，PaLM-E 都是基于该领域的专家数据进行训练的。在许多情况下，每个任务的可用数据量是稀疏的。TAMP 任务涉及对可能的计划进行大量组合，并且许多决策序列是不可行的。PaLM-E 必须生成由多个步骤组成的计划，这些步骤具有复杂的决策边界。多物体桌面推送环境来源于公开可用的 Language-Table 数据集（Lynch 等，2022），该环境具有挑战性，因为它包含多个物体、大量语言元素和复杂的推送动态。对于 TAMP 和 Language-Table 环境，PaLM-E 必须推理物体的姿态。仅仅知道哪些物体在桌子上或者大致了解它们之间的关系是不够的，关于场景几何的更细致细节对于解决任务至关重要。最后，我们考虑一个类似于 SayCan（Ahn 等，2022）的移动操控领域，其中机器人需要在厨房环境中解决多种任务，包括在抽屉中寻找物体、抓取它们并将其带给人类。对于所有领域，我们都考虑了这些环境中的规划和视觉问答（VQA）任务。对于移动操控和 Language-Table 环境，PaLM-E 被集成到控制循环中，以在现实世界中执行计划，并必须在面对外部干扰或低级控制策略失败时调整计划。

# 6.2. TAMP 环境

附录中的表7显示了TAMP环境的规划成功率和VQA性能。在这些实验中，大语言模型（LLM）是冻结的（针对预训练的LLM）。表7中报告的结果中，输入表示是在包含96,000个仅针对TAMP环境的训练场景的数据集上训练的，即没有其他数据包含在混合中。对于场景中有3-5个物体的情况，数量与训练集相同，大多数输入表示的表现都相似。然而，当增加物体数量时，结果表明使用预训练的LLM显著提高了性能，特别是在实体引用方面。此外，我们显示62B的LLM相比于8B变体在分布外泛化方面表现更好，而未预训练的LLM基本上没有分布外泛化能力。SayCan基线（Ahn等，2022）使用了oracle可行性函数，但在解决该环境时存在困难，因为可行性函数仅限于当前可能的操作，但对于LLM在TAMP环境中构建长期计划并没有足够的信息。

表 1 显示了在 $1\%$ 数据集上训练 3-5 个对象的结果，这对应于每个规划任务仅有 320 个示例。我们可以看到输入表示之间存在显著差异，尤其是在规划任务中。首先，在低数据场景下，预训练大语言模型对状态输入是有益的。其次，两个 ViT 变体 $\mathrm { \Delta V i T + T L }$ 和 ViT-4B 在处理这些少量数据的规划任务时表现不佳。然而，如果我们同时在其他机器人环境以及一般的视觉-语言数据集上进行联合训练（ViT-4B 通用型），则 ViT-4B 的性能超过了两倍。这表明不同机器人形态和任务之间存在显著的迁移效应。最后，使用 OSRT 作为输入表示在这里导致了最佳性能，展示了 3D 感知对象表示的优势。我们还观察到这里的另一个迁移实例：当我们移除 TAMP VQA 数据，仅在 640 个规划任务示例上训练时，性能出现（轻微）下降。未在机器人数据上训练的最先进视觉-语言模型 PaLI（Chen 等，2022）无法解决这些任务。我们仅在 $\mathrm { { q _ { 2 } } }$（桌上物体的左右/中间位置）和 $\mathrm { { q } _ { 3 } }$（垂直物体关系）上对其进行了评估，因为这两者最类似于典型的 VQA 任务。

# 6.3. 语言-表格环境

表2报告了来自Language-Table环境（Lynch et al., 2022）上长时间跨度任务的成功率。PaLM-E整合于一个控制环中，该控制环以长时间跨度任务和当前图像作为输入，并为低级策略输出指令。我们发现，在互联网规模的视觉与语言联合训练下，模型在机器人规划方面变得更加有效，尤其是在只有每个任务10个示例的少量样本条件下。将12B模型扩展到84B模型在3个任务中的2个上取得了改进。与TAMP环境一样，SayCan和零-shot PaLI都不具备有效性，无法解决测试中最简单的任务。

![](images/12.jpg)  

Figure 4: Planning success results in the TAMP environment $1 \%$ data) for PaLM-E-12B, comparing of the effects of PaLM-E models (i) using the full training mixture, (ii) pre-training (ViT and PaLM), and (iii) freezing or finetuning the language model. Transfer from full mixture is particularly effective. Note that full mixture contains only $1 \%$ of the training data (320 examples each) for the tasks evaluated here. Shown is the mean of tasks $\mathsf { p } _ { 1 } , \mathsf { p } _ { 2 }$ .

真实机器人结果与少样本泛化。在图7的a)中，我们可以看到PaLM-E能够指导一台真实机器人完成多阶段的桌面操作任务，同时保持对抗干扰的鲁棒性。在给定观测图像和长期目标的情况下，例如“按颜色将积木分类到角落”，PaLM-E以$1 \ \mathrm{Hz}$的频率输出语言子目标给Lynch等人(2022)的策略，这些策略以$5 \ \mathrm{Hz}$的频率输出低级机器人动作。之前的工作(Lynch等人，2022)则是在循环中引入人类，进行交互式指导子目标和修正。在图5的b)中，我们看到PaLM-E具备一-shot和零-shot学习的能力。在这里，我们在100个不同的长期任务上微调了PaLM-E，每个任务只有一个训练样本，例如“将所有积木放在中心”，“将蓝色积木从队列中移除”。我们还观察到，PaLM-E能够在零-shot情况下泛化到涉及新对象对的任务（图7的c)）以及涉及在原始机器人数据集或微调数据集中未见过的对象的任务，例如一个玩具海龟（图5的d)）。

# 6.4. 移动操作环境

我们展示了PaLM-E在具有挑战性和多样化的移动操控任务上的表现。我们在很大程度上遵循Ahn等人（2022）中的设置，其中机器人需要根据人类的指令规划一系列导航和操作动作。例如，给定指令“我洒了饮料，你能给我带点东西来清理一下吗？”，机器人需要规划一个包含“1. 找到海绵，2. 拿起海绵，3. 把它带给用户，4. 放下海绵。”的动作序列。受到这些任务的启发，我们开发了三个用例来测试PaLM-E的具身推理能力：可供性预测、故障检测和长时间规划。低层策略来自RT-1（Brohan等人，2022），这是一种接受RGB图像和自然语言指令的变换器模型，并输出末端执行器控制命令。

![](images/13.jpg)  
one-shot: "Move the remaining blocks to the group"   
zero-shot: "Move the green blocks to the turtle"   
a kitchen, and one-shot / zero-shot generalization with a tabletop manipulation robot.

<table><tr><td rowspan="2"></td><td rowspan="2">Object- centric</td><td rowspan="2">LLM pre-train</td><td colspan="4">Embodied VQA</td><td colspan="2">Planning</td></tr><tr><td>q1</td><td>q2</td><td>q3</td><td>q4</td><td>p1</td><td>P2</td></tr><tr><td>SayCan (oracle afford.) (Ahn et al., 2022)</td><td></td><td>✓</td><td></td><td>-</td><td>-</td><td></td><td></td><td>38.7 33.3</td></tr><tr><td>PaLI (zero-shot) (Chen et al., 2022)</td><td></td><td>✓</td><td></td><td>0.0</td><td>0.0</td><td></td><td>-</td><td>-</td></tr><tr><td>PaLM-E (ours) w/ input enc:</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>State</td><td>(GT)</td><td>X</td><td></td><td>99.4 89.8</td><td>90.3</td><td></td><td></td><td>88.3 45.0 46.1</td></tr><tr><td>State</td><td>(GT)</td><td>✓</td><td></td><td>100.0 96.3</td><td>95.1</td><td></td><td></td><td>93.1 55.9 49.7</td></tr><tr><td>ViT + TL</td><td>(GT)</td><td>✓</td><td>34.7</td><td>54.6</td><td>74.6</td><td></td><td></td><td>91.6 24.0 14.7</td></tr><tr><td>ViT-4B single robot</td><td>X</td><td>✓</td><td>-</td><td>45.9</td><td>78.4</td><td></td><td></td><td>92.2 30.6 32.9</td></tr><tr><td>ViT-4B full mixture</td><td>X</td><td></td><td>-</td><td>70.7</td><td>93.4</td><td></td><td></td><td>92.1 74.1 74.6</td></tr><tr><td>OSRT (no VQA)</td><td>✓</td><td></td><td>-</td><td>-</td><td>-</td><td>-</td><td></td><td>71.9 75.1</td></tr><tr><td>OSRT</td><td>✓</td><td>✓</td><td>99.7</td><td>98.2 100.0 93.7 82.5 76.2</td><td></td><td></td><td></td><td></td></tr></table>

Table 1: Comparison of different input representations on TAMP environment (in terms of success rates), where data from TAMP constitutes only $1 \%$ (i.e., 320 samples for ${ \tt p } _ { 1 }$ , ${ \tt p } _ { 2 }$ each) of total training data size. PaLM-E outperforms both PaLI and SayCan on embodied VQA and planning tasks. Cross-domain transfer is observed, since the PaLM-E with ViT-4B trained on our full data mixture improves planning performance. OSRT, despite using no large-scale data, provides the most effective input encodings for learning. (GT) means ground-truth object-centric information provided. In all experiments, the LLM is frozen. The non-object centric ViT-4B variant utilizes color to reference objects, hence ${ \bf q } _ { 1 }$ cannot be evaluated here. The LLM is frozen in these experiments (except for the case where it is not pre-trained). Sec. B.1 describes the tasks $\mathbf { q } _ { 1 } \mathbf { - q } _ { 4 }$ , ${ \tt p } _ { 1 }$ , $\mathrm { \mathsf { q } _ { 2 } }$ .

Affordance prediction. We investigate PaLM-E's performance at affordance prediction, i.e. whether a sk ill of the low-level policy can be executed in the current environment. This can be formulated as the VQA problem Given <img>. $\mathsf Q$ : Is it possible to <skill> here?. PaLM-E outperforms PaLI (zero-shot), as well as thresholding on value functions trained with QT-OPT (Tab. 4).

Failure detection. For a robot to do closed-loop planning, it is also important to detect failures, as is shown in (Huang et al., 2022c). The multi-modal prompt is Given <img>. Q: Was <skill> successful?. Tab. 4 shows that PaLM-E outperforms PaLI (zero-shot), as well as a finetuned version of CLIP on this dataset. PaLM-E also outperforms the algorithm proposed in Xiao et al. (2022) that leverages two CLIP models trained with hindsight relabeled data. This method has access to more information than our method, and was specifically designed to just solve failure detection on this dataset.

Real robot results: Long-horizon planning. Finally, we use PaLM-E to perform embodied planning end-to-end for mobile manipulation tasks. The prompt structure for this taskis Human: <instruction> Robot: <step history>. I see <img>. PaLM-E is trained to generate the next step of the plan, conditioned on the history of taken steps and the current image observation of the scene. After each step is decoded, we map them to a low-level policy as defined in Ahn et al. (2022). This process is done in an autoregressive manner, until PaLM-E outputs "terminate". We train the model by using the runs from (Ahn et al., 2022), which contains 2912 sequences. We qualitatively evaluated the model in a real kitchen and found the model can carry out long-horizon mobile manipulation tasks, even under adversarial disturbances (Fig. 5).

# 6.5. 在通用视觉语言任务上的表现

虽然这不是我们工作的重点，但我们在表5中报告了关于通用视觉-语言任务的结果，包括OKVQA（Marino等，2019），VQA v2（Goyal等，2017）和COCO标题生成（Chen等，2015）。一个单一的通用模型

Table 2: Results on planning tasks in the simulated environment from Lynch et al. (2022).   

<table><tr><td colspan="7">Zero-shot Baselines</td><td colspan="3">Task 1</td><td colspan="3">Task 2</td><td colspan="3">Task 3</td></tr><tr><td colspan="5">SayCan (oracle afford.) (Ahn et al., 2022) PaLI (Chen et al., 2022)</td><td colspan="3">0.0 0.0</td><td colspan="5"></td><td colspan="2">- -</td></tr><tr><td colspan="5">trained</td><td>Task</td><td colspan="5"># Demos</td><td colspan="5"></td></tr><tr><td>PL-E-</td><td>on</td><td>from scratch</td><td>LLM+ViT pretrain</td><td>LLM frozen</td><td>finetune</td><td>10</td><td>20</td><td>40</td><td>10</td><td>20</td><td>40</td><td>10</td><td>20</td><td></td><td>80</td></tr><tr><td>12B</td><td>Single robot</td><td>✓</td><td>X</td><td>n/a</td><td>✓</td><td>20.0</td><td>30.0</td><td>50.0</td><td>2.5</td><td>6.3</td><td>2.5</td><td>11.3</td><td>16.9</td><td></td><td>28.3</td></tr><tr><td>12B</td><td>Full mixture</td><td>X</td><td>✓</td><td>✓</td><td>X</td><td>-</td><td>-</td><td>20.0</td><td>-</td><td></td><td></td><td>36.3</td><td>-</td><td>-</td><td>29.4</td></tr><tr><td>12B</td><td>Full mixture</td><td>X</td><td>✓</td><td>X</td><td>X</td><td>-</td><td></td><td>80.0</td><td></td><td></td><td></td><td>57.5</td><td></td><td></td><td>50.0</td></tr><tr><td>12B</td><td>Full mixture</td><td>X</td><td>✓</td><td>X</td><td>✓</td><td>70.0</td><td>80.0</td><td>80.0</td><td>31.3</td><td>58.8</td><td></td><td>58.8</td><td>57.5</td><td>54.4</td><td>56.3</td></tr><tr><td>84B</td><td>Full mixture</td><td>X</td><td>✓</td><td>X</td><td>X</td><td>-</td><td>-</td><td>90.0</td><td></td><td></td><td></td><td>53.8</td><td>-</td><td>-</td><td>64.4</td></tr></table>

<table><tr><td>Task 1. Q: There is a block that is closest to {i.e., top right corner}. Push that block to the other block of the same color.</td></tr><tr><td>Task 2. Q: How to sort the blocks by colors into corners?</td></tr><tr><td>Task 3. Q: How to push allthe blocks that are on the {left/right} side together, without bringing over any of the blocks that are on the {right/left} side?</td></tr></table>

Table 4: Mobile manipulation environment: failure detection and affordance prediction (F1 score).   

<table><tr><td colspan="3">Baselines</td><td>Failure det.</td><td>Affordance</td></tr><tr><td colspan="3">PaLI (Zero-shot) (Chen et al., 2022)</td><td>0.73</td><td>0.62</td></tr><tr><td colspan="3">CLIP-FT (Xiao et al., 2022)</td><td>0.65</td><td>-</td></tr><tr><td colspan="3">CLIP-FT-hindsight (Xiao et al., 2022)</td><td>0.89</td><td>-</td></tr><tr><td colspan="3">QT-OPT (Kalashnikov et al., 2018)</td><td>-</td><td>0.63</td></tr><tr><td>PaLM-E-12B trained on</td><td>from scratch</td><td>LLM+ViT pretrain</td><td>LLM frozen</td><td></td></tr><tr><td>Single robot</td><td>✓</td><td>X</td><td>n/a ✓</td><td>0.54 0.91</td></tr><tr><td>Single robot</td><td>X</td><td>✓</td><td></td><td>0.78 0.87</td></tr><tr><td>Full mixture</td><td>X</td><td>✓</td><td>✓</td><td>0.91</td></tr><tr><td>Full mixture</td><td>X</td><td>✓</td><td>X</td><td>0.77</td></tr></table>

Table 5: Results on general visual-language tasks. For the generalist models, they are the same checkpoint across the different evaluations, while task-specific finetuned models use differentfinetuned models for the different tasks. COCO uses Karpathy splits. $\dagger$ is 32-shot on OK-VQA (not finetuned).   

<table><tr><td>Model</td><td colspan="2">VQAv2 test-dev test-std</td><td>OK-VQA val</td><td>COCO Karpathy test</td></tr><tr><td colspan="5">Generalist (one model)</td></tr><tr><td>PaLM-E-12B</td><td>76.2</td><td></td><td>55.5</td><td>135.0</td></tr><tr><td>PaLM-E-562B</td><td>80.0</td><td>-</td><td>66.1</td><td>138.7</td></tr><tr><td colspan="5">Task-specific finetuned models</td></tr><tr><td>Flamingo (Alayrac et al., 2022)</td><td>82.0</td><td>82.1</td><td>57.8†</td><td>138.1</td></tr><tr><td>PaLI (Chen et al., 2022)</td><td>84.3</td><td>84.3</td><td>64.5</td><td>149.1</td></tr><tr><td>PaLM-E-12B</td><td>77.7</td><td>77.9</td><td>60.1</td><td>136.0</td></tr><tr><td>PaLM-E-66B</td><td>-</td><td>-</td><td>62.9</td><td>-</td></tr><tr><td>PaLM-E-84B</td><td>80.5</td><td>-</td><td>63.3</td><td>138.0</td></tr><tr><td colspan="5">Generalist (one model), with frozen LLM</td></tr><tr><td>(Tsimpoukelli et al., 2021)</td><td>48.4</td><td></td><td>-</td><td>-</td></tr><tr><td>PaLM-E-12B frozen</td><td>70.3</td><td></td><td>51.5</td><td>128.0</td></tr></table>

PaLM-E-562B模型在OK-VQA上取得了最高的报告成绩，包括超越了专门针对OK-VQA微调的模型。与（Tsimpoukelli等，2021）相比，PaLM-E在VQA v2上以固定的LLM实现了最高的表现，据我们所知。这表明PaLM-E不仅在机器人任务中是一个具身推理者，同时也是一个具有竞争力的视觉语言通才。

# 6.6. 在通用语言任务上的表现

表 8 报告了 PaLM-E 在 21 项通用语言基准测试中针对自然语言理解（NLU）和自然语言生成（NLG）任务的平均表现。显著的趋势是，随着模型规模的增加，语言能力的灾难性遗忘显著减少。如图 6 所示，对于最小的模型（PaLM ），

![](images/14.jpg)  

Table 3: Task prompts for Tab. 2.   

Figure 6: Results on general language tasks ( $\mathbf { N L G } =$ natural language generation): increasing scale leads to less catastrophic forgetting between a corresponding PaLM-E model and its inherited PaLM model. See full suite of tasks and results in Tab. 8.

E-12B) 模型在多模态训练期间，其自然语言生成（NLG）性能下降了 $8 7 . 3 \%$（相对），而对于最大的模型（PaLM-E-562B），仅下降了 $3 . 9 \%$。

# 7. 实验总结与讨论

通用模型与专业模型的迁移。如图3所示，我们展示了多个迁移的实例，这意味着在相同时间内对不同任务和数据集进行训练的PaLM-E，相较于分别仅在不同任务上训练的模型，性能显著提升。在图4中，对“完整混合”进行共同训练的性能实现了翻倍。在表9中，若加入LLM/ViT的预训练，且对完整混合数据进行训练而非仅对移动操作数据进行训练，性能显著改善。对于表2中的语言表实验，我们观察到了类似的行为。数据效率。与可用的大规模语言或视觉-语言数据集相比，机器人数据显著稀缺。正如在最后一段中讨论的，我们的模型展示了迁移能力，帮助PaLM-E在机器人领域通过极少的训练样本解决机器人任务，例如语言表任务的样本数在10到80之间，TAMP任务的样本数为320。OSRT结果通过使用几何输入表示展示了另一种数据效率实例。未来工作的一个有希望的方向是将其与受益于大规模视觉数据的方法结合起来。保持语言能力。我们展示了在多模态训练过程中保持模型语言能力的两种路径作为另一种途径，当整个模型进行端到端训练时，随着模型规模的增加，模型保持了更高的原始语言性能（图6）。

# 8. 结论

我们提出通过将图像等多模态信息注入预训练的大语言模型的嵌入空间，构建一个具身语言模型。实验表明，现成的最先进的视觉-语言模型在一般视觉问答和图像描述任务上训练，但对于具身推理任务并不够充分，同时也揭示了最近针对通过可供性来支撑语言模型的提案的局限性。为了解决这些局限性，我们提出了PaLM-E，一个能够在模拟环境和现实世界中控制不同机器人，同时在一般视觉问答和图像描述任务上定量表现良好的单一模型。特别是，将神经场景表示（即，OSRT）引入模型的新颖架构思想尤其有效，即便在没有大规模数据的情况下也是如此。PaLM-E在多种机器人具身和一般视觉-语言任务的多样任务组合上进行训练。重要的是，我们已经证明这种多样化的训练为从视觉-语言领域到具身决策制定开辟了多个迁移通道，使得机器人规划任务能够高效利用数据。我们的结果表明，冷冻语言模型是朝向通用具身多模态模型的可行路径，这些模型完全保留其语言能力，但我们也提出了一条不同的路径，即通过解冻模型：扩大语言模型的规模可以显著减少在成为具身智能体过程中的灾难性遗忘。我们最大的模型PaLM-E-562B展现了新兴能力，如多模态推理链以及对多张图像进行推理的能力，尽管其训练仅基于单图像提示。

# 致谢

作者感谢以下人士的建议、帮助和支持：Xi Chen、Etienne Pot、Sebastian Goodman、Maria Attarian、Ted Xiao、Keerthana Gopalakrishnan、Kehang Han、Henryk Michalewski、Neil Houlsby、Basil Mustafa、Justin Gilmer、Yonghui Wu、Erica Moreira、Victor Gomes、Tom Duerig、Henning Meyer 和 Kendra Byrne。

# References

Ahn, M., Brohan, A., Brown, N., Chebotar, Y., Cortes, O., David, B., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., et al. Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691, 2022.

Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al. Flamingo: a visual language model for few-shot learning. arXiv preprint arXiv:2204.14198, 2022.

Bommasani, R., Hudson, D. A., Adeli, E., Altman, R., Arora, S., von Arx, S., Bernstein, M. S., Bohg, J., Bosselut, A., Brunskill, E., et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021.

Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Dabis, J., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., Hsu, J., et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in neural information processing systems, 33: 18771901, 2020.

Changpinyo, S., Kukliansky, D., Szpektor, I., Chen, X., Ding, N., and Soricut, R. All you may need for vqa are image captions, 2022. URL https: //arxiv.org/ abs/2205.01883.

Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. d. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021a.

Chen, T., Saxena, S., Li, L., Fleet, D. J., and Hinton, G. Pix2seq: A language modeling framework for object detection. arXiv preprint arXiv:2109.10852, 2021b.

Chen, X., Fang, H., Lin, T., Vedantam, R., Gupta, S., Dollár, P., and Zitnick, C. L. Microsoft COCO captions: Data collection and evaluation server. CoRR, abs/1504.00325, 2015.

Chen, X., Wang, X., Changpinyo, S., Piergiovanni, A., Padlewski, P., Salz, D., Goodman, S., Grycner, A. Mustafa, B., Beyer, L., et al. Pali: A jointly-scaled multilingual language-image model. arXiv preprint arXiv:2209.06794, 2022.

Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H. W., Sutton, C., Gehrmann, S., et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.

Dehghani, M., Djolonga, J., Mustafa, B., Padlewski, P., Heek, J., Gilmer, J., Steiner, A., Caron, M., Geirhos, R., Alabdulmohsin, I., et al. Scaling vision transformers to 22 billion parameters. arXiv preprint arXiv:2302.05442, 2023.

Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Driess, D., Ha, J.-S., and Toussaint, M. Deep visual reasoning: Learning to predict action sequences for task and motion planning from an initial scene image. In Proc. of Robotics: Science and Systems (R:SS), 2020.

Gan, Z., Li, L., Li, C., Wang, L., Liu, Z., Gao, J., et al. Vision-language pre-training: Basics, recent advances, and future trends. Foundations and Trends® in Computer Graphics and Vision, 14(34):163352, 2022.

Glaese, A., McAleese, N., Trebacz, M., Aslanides, J., Firoiu, V., Ewalds, T., Rauh, M., Weidinger, L., Chadwick, M., Thacker, P., et al. Improving alignment of dialogue agents via targeted human judgements. arXiv preprint arXiv:2209.14375, 2022.

Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., and Parikh, D. Making the V in VQA matter: Elevating the role of image understanding in Visual Question Answering. In Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

Guhur, P.-L., Chen, S., Garcia, R., Tapaswi, M., Laptev, I., and Schmid, C. Instruction-driven history-aware policies for robotic manipulations. arXiv preprint arXiv:2209.04899, 2022.

Hao, Y., Song, H., Dong, L., Huang, S., Chi, Z., Wang, W. Ma, S., and Wei, F. Language models are general-purpose interfaces. arXiv preprint arXiv:2206.06336, 2022.

Hu, X., Gan, Z., Wang, J., Yang, Z., Liu, Z., Lu, Y., and Wang, L. Scaling up vision-language pre-training for image captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1798017989, 2022.

Huang, C., Mees, O., Zeng, A., and Burgard, W. Visual language maps for robot navigation. arXiv preprint arXiv:2210.05714, 2022a.

Huang, W., Abbeel, P., Pathak, D., and Mordatch, I. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. arXiv preprint arXiv:2201.07207, 2022b.

Huang, W., Xia, F., Xiao, T., Chan, H., Liang, J., Florence, P., Zeng, A., Tompson, J., Mordatch, I., Chebotar, Y., et al. Inner monologue: Embodied reasoning through planning with language models. arXiv preprint arXiv:2207.05608, 2022c.

Jang, E., Irpan, A., Khansari, M., Kappler, D., Ebert, F., Lynch, C., Levine, S., and Finn, C. Bc-z: Zero-shot task generalization with robotic imitation learning. In Conference on Robot Learning, pp. 9911002. PMLR, 2022.

Jiang, Y., Gupta, A., Zhang, Z., Wang, G., Dou, Y., Chen, Y., Fei-Fei, L., Anandkumar, A., Zhu, Y., and Fan, L. Vima: General robot manipulation with multimodal prompts. arXiv preprint arXiv:2210.03094, 2022.

Kalashnikov, D., Irpan, A., Pastor, P., Ibarz, J., Herzog, A., Jang, E., Quillen, D., Holly, E., Kalakrishnan, M., Vanhoucke, V., et al. Scalable deep reinforcement learning for vision-based robotic manipulation. In Conference on Robot Learning, pp. 651673. PMLR, 2018.

Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., and Iwasawa, Y. Large language models are zero-shot reasoners. arXiv preprint arXiv:2205.11916, 2022.

Lester, B., Al-Rfou, R., and Constant, N. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691, 2021.

Lewkowycz, A., Andreassen, A., Dohan, D., Dyer, E., Michalewski, H., Ramasesh, V., Slone, A., Anil, C., Schlag, I., Gutman-Solo, T., et al. Solving quantitative reasoning problems with language models. arXiv preprint arXiv:2206.14858, 2022.

Li, L. H., Yatskar, M., Yin, D., Hsieh, C.-J., and Chang, K.-W. Visualbert: A simple and performant baseline for vision and language. arXiv preprint arXiv:1908.03557, 2019.

Li M., Lv, T., Chen, J., Cui, L., Lu, Y., Flrecio, D., Zha, C., Li, Z., and Wei, F. Trocr: Transformer-based optical character recognition with pre-trained models. arXiv preprint arXiv:2109.10282, 2021.

Li, S., Puig, X., Du, Y., Wang, C., Akyurek, E., Torralba, A., Andreas, J., and Mordatch, I. Pre-trained language models for interactive decision-making. arXiv preprint arXiv:2202.01771, 2022.

Liang, J., Huang, W., Xia, F., Xu, P., Hausman, K., Ichter, B., Florence, P., and Zeng, A. Code as policies: Language model programs for embodied control. arXiv preprint arXiv:2209.07753, 2022.

Locatello, F., Weissenborn, D., Unterthiner, T., Mahendran, A., Heigold, G., Uszkoreit, J., Dosovitskiy, A., and Kipf, T. Object-centric learning with slot attention. Advances in Neural Information Processing Systems, 33:11525 11538, 2020.

Lu, J., Batra, D., Parikh, D., and Lee, S. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. Advances in neural information processing systems, 32, 2019.

Lu, K., Grover, A., Abbeel, P., and Mordatch, I. Pretrained transformers as universal computation engines. arXiv preprint arXiv:2103.05247, 1, 2021.

Lynch, C. and Sermanet, P. Language conditioned imitation learning over unstructured data. arXiv preprint arXiv:2005.07648, 2020.

Lynch, C., Wahid, A., Tompson, J., Ding, T., Betker, J., Baruch, R., Armstrong, T., and Florence, P. Interactive language: Talking to robots in real time. arXiv preprint arXiv:2210.06407, 2022.

Marino, K., Rastegari, M., Farhadi, A., and Mottaghi, R. Okvqa: A visual question answering benchmark requiring external knowledge. In Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

Nair, S., Mitchell, E., Chen, K., Savarese, S., Finn, C., et al. Learning language-conditioned robot behavior from offline data and crowd-sourced annotation. In Conference on Robot Learning, pp. 13031315. PMLR, 2022.

Nottingham, K., Ammanabrolu, P., Suhr, A., Choi, Y., Hajishirzi, H., Singh, S., and Fox, R. Do embodied agents dream of pixelated sheep?: Embodied decision making using language guided world modelling. arXiv preprint arXiv:2301.12050, 2023.

Piergiovanni, A., Kuo, W., and Angelova, A. Pre-training image-language transformers for open-vocabulary tasks, 2022. URL https://arxiv.org/abs/2209. 04372.

Polu, S., Han, J. M., Zheng, K., Baksys, M., Babuschkin, I. and Sutskever, I. Formal mathematics statement curriculum learning. arXiv preprint arXiv:2202.01344, 2022.

Reed, S., Zolna, K., Parisotto, E., Colmenarejo, S. G., Novikov, A., Barth-Maron, G., Gimenez, M., Sulsky, Y., Kay, J., Springenberg, J. T., et al. A generalist agent. arXiv preprint arXiv:2205.06175, 2022.

Ryoo, M. S., Piergiovanni, A., Arnab, A., Dehghani, M., and Angelova, A. Tokenlearner: What can 8 learned tokens do for images and videos? arXiv preprint arXiv:2106.11297, 2021.

Sajjadi, M. S. M., Duckworth, D., Mahendran, A., van Steenkiste, S., Paveti, F., Lui, M., Guibas, L. J., Greff, K., and Kipf, T. Object Scene Representation Transformer. NeurIPS, 2022a. URL https: //osrt-paper.github.io/.

Sajjadi, M. S. M., Meyer, H., Pot, E., Bergmann, U., Greff, K., Radwan, N., Vora, S., Lui, M., Duckworth, D., Dosovitskiy, A., et al. Scene representation transformer: Geometry-free novel view synthesis through set-latent scene representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 62296238, 2022b.

Shah, D., Osinski, B., Ichter, B., and Levine, S. Lmnav: Robotic navigation with large pre-trained models of language, vision, and action. arXiv preprint arXiv:2207.04429, 2022.

Sharma, P., Ding, N., Goodman, S., and Soricut, R. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of ACL, 2018.

Sharma, P., Torralba, A., and Andreas, J. Skill induction and planning with latent language. arXiv preprint arXiv:2110.01517, 2021.

Shridhar, M., Manuelli, L., and Fox, D. Cliport: What and where pathways for robotic manipulation. In Conference on Robot Learning, pp. 894906. PMLR, 2022a.

Shridhar, M., Manuelli, L., and Fox, D. Perceiver-actor: A multi-task transformer for robotic manipulation. arXiv preprint arXiv:2209.05451, 2022b.

Silva, A., Moorman, N., Silva, W., Zaidi, Z., Gopalan, N., and Gombolay, M. Lancon-learn: Learning with language to enable generalization in multi-task manipulation. IEEE Robotics and Automation Letters, 7(2):16351642, 2021.

Singh, I., Blukis, V., Mousavian, A., Goyal, A., Xu, D., Tremblay, J., Fox, D., Thomason, J., and Garg, A. ProgPrompt: Generating situated robot task plans using large language models. arXiv preprint arXiv:2209.11302, 2022.

Tellex, S., Gopalan, N., Kress-Gazit, H., and Matuszek, C. Robots that use language. Annual Review of Control, Robotics, and Autonomous Systems, 3:2555, 2020.

Thoppilan, R., De Freitas, D., Hall, J., Shazeer, N., Kulshreshtha, A., Cheng, H.-T., Jin, A., Bos, T., Baker, L.,

Du, Y., et al. Lamda: Language models for dialog applications. arXiv preprint arXiv:2201.08239, 2022.

Tsimpoukelli, M., Menick, J. L., Cabi, S., Eslami, S., Vinyals, O., and Hill, F. Multimodal few-shot learning with frozen language models. Advances in Neural Information Processing Systems, 34:200212, 2021.

Wang, Z., Cai, S., Liu, A., Ma, X., and Liang, Y. Describe, explain, plan and select: Interactive planning with large language models enables open-world multi-task agents. arXiv preprint arXiv:2302.01560, 2023.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Chi, E., Le, Q., and Zhou, D. Chain of thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903, 2022.

Xiao, T., Chan, H., Sermanet, P., Wahid, A., Brohan, A., Hausman, K., Levine, S., and Tompson, J. Robotic skill acquisition via instruction augmentation with visionlanguage models. arXiv preprint arXiv:2211.11736, 2022.

Zellers, R., Holtzman, A., Peters, M., Mottaghi, R., Kembhavi, A., Farhadi, A., and Choi, Y. Piglet: Language grounding through neuro-symbolic interaction in a 3d world. arXiv preprint arXiv:2106.00188, 2021a.

Zellers, R., Lu, X., Hessel, J., Yu, Y., Park, J. S., Cao, J. Farhadi, A., and Choi, Y. Merlot: Multimodal neural script knowledge models. Advances in Neural Information Processing Systems, 34:2363423651, 2021b.

Zeng, A., Wong, A., Welker, S., Choromanski, K., Tombari, F., Purohit, A., Ryoo, M., Sindhwani, V., Lee, J., Vanhoucke, V., et al. Socratic models: Composing zeroshot multimodal reasoning with language. arXiv preprint arXiv:2204.00598, 2022.

Zhang, Y. and Chai, J. Hierarchical task learning from language instructions with unified transformers and selfmonitoring. arXiv preprint arXiv:2106.03427, 2021.

Zhou, L., Palangi, H., Zhang, L., Hu, H., Corso, J., and Gao, J. Unified vision-language pre-training for image captioning and vqa. In Proceedings of the AAAI Conference on Artificial Intelligence, 2020.

![](images/15.jpg)  
PaLM-E guiding a real robot through long horizon tasks   
to adversarial disturbances. We find evidence that PaLM- is capableof one-shot and zero shot generalization.

# A. Data Mixture

lata distribution is general vision-language tasks, with less than $10 \%$ robot data.

# B. Environment Details

# B.1. Task and Motion Planning (TAMP)

joses. Fig. 8 show an example test scene that contains 6 objects.

In the global version, we consider the following three VQA tasks:

Table 6: Dataset sampling frequency and ratio for the "full mixture" referred to in experiments.   

<table><tr><td>Dataset in full mixture</td><td>Sampling frequency</td><td>%</td></tr><tr><td>Webli (Chen et al., 2022)</td><td>100</td><td>52.4</td></tr><tr><td>VQ2A (Changpinyo et al., 2022)</td><td>25</td><td>13.1</td></tr><tr><td>VQG (Changpinyo et al., 2022)</td><td>10</td><td>5.2</td></tr><tr><td>CC3M (Sharma et al., 2018)</td><td>25</td><td>13.1</td></tr><tr><td>Object Aware (Piergiovanni et al., 2022)</td><td>10</td><td>5.2</td></tr><tr><td>OKVQA (Marino et al., 2019)</td><td>1</td><td>0.5</td></tr><tr><td>VQAv2 (Goyal et al., 2017)</td><td>1</td><td>0.5</td></tr><tr><td>COCO (Chen et al., 2015)</td><td>1</td><td>0.5</td></tr><tr><td>Wikipedia text</td><td>1</td><td>0.5</td></tr><tr><td>(robot) Mobile Manipulator, real</td><td>6</td><td>3.1</td></tr><tr><td>(robot) Language Table (Lynch et al., 2022), sim and real</td><td>8</td><td>4.2</td></tr><tr><td>(robot) TAMP, sim</td><td>3</td><td>1.6</td></tr></table>

![](images/16.jpg)  
.

: $\mathrm { q _ { 2 } }$ : object-table relation. Example prompt: Given <img>. Q: Is the red object left, right or center of the table?.Target: A: The red object is in the center of the table.

: ${ \mathrm { q } } _ { 3 }$ : object-object relations. Example prompt: Given <img>. Q: Is the yellow object below the blue object?.Target: A: No, the yellow object is not below the blue object.

: $\mathrm { q _ { 4 } }$ : plan feasibility. Example prompt: Given <img>. Q: Is it possible to first grasp th blue object, then place it on the yellow object, and then grasp the yellow object?.Target: A: No, this is not possible.

as well as the two planning tasks

: ${ \tt p } _ { 1 }$ : grasping. Example prompt: Given <img>. Q: How to grasp the green object?. Target: A: First grasp the orange object and place it on the table, then grasp the green object. : ${ \tt p } _ { 2 }$ : stacking. Example prompt: Given <img>. Q: How to stack the white object on top of the red object?. Target: A: First grasp the green object and place it on the table, then grasp the white object and place it on the red object.

F $=$ obj 1 is <ob $\dot { ] } _ { 1 } >$ . . . . Obj j is ${ < \mathrm { o b j } _ { j } > }$ . , and the VQA task $\mathbf { q } _ { 1 }$ is about the color of an object. The other tasks (except with the different prefix, and entity referrals), remain the same.

obtained with the method of Driess et al. (2020).

been referenced by their special tokens $\mathrm { o b j } _ { j }$ jeThe Cble, affordance functions.   

<table><tr><td></td><td>φ</td><td>LLM pre-trained</td><td>q1</td><td>q2</td><td>q3</td><td>q4</td><td>P1</td><td>P2</td></tr><tr><td rowspan="10"></td><td>SayCan (w/ oracle affordances)</td><td>✓</td><td>-</td><td>-</td><td>-</td><td>-</td><td>38.7</td><td>33.3</td></tr><tr><td>state</td><td>X</td><td>100.0</td><td>99.3</td><td>98.5</td><td>99.8</td><td>97.2</td><td>95.5</td></tr><tr><td>state</td><td>(unfrozen)</td><td>100.0</td><td>98.8</td><td>100.0</td><td>97.6</td><td>97.7</td><td>95.3</td></tr><tr><td>state</td><td></td><td>100.0</td><td>98.4</td><td>99.7</td><td>98.5</td><td>97.6</td><td>96.0</td></tr><tr><td>state (w/o entity referrals)</td><td></td><td>100.0</td><td>98.8</td><td>97.5</td><td>98.1</td><td>94.6</td><td>90.3</td></tr><tr><td>ViT + TL (obj. centric)</td><td></td><td>99.6</td><td>98.7</td><td>98.4</td><td>96.8</td><td>9.2</td><td>94.5</td></tr><tr><td>ViT + TL (global)</td><td></td><td>-</td><td>60.7</td><td>90.8</td><td>94.3</td><td>70.7</td><td>69.2</td></tr><tr><td>ViT-4B (global)</td><td></td><td>-</td><td>98.2</td><td>99.4</td><td>99.0</td><td>96.0</td><td>93.4</td></tr><tr><td>ViT-4B generalist</td><td></td><td></td><td>97.1</td><td>100.0</td><td>98.9</td><td>97.5</td><td>95.2</td></tr><tr><td>OSRT</td><td>✓</td><td>99.6</td><td>99.1</td><td>100.0</td><td>98.8</td><td>98.1</td><td>95.7</td></tr><tr><td rowspan="3">6 objects</td><td>state</td><td>X</td><td>20.4</td><td>39.2</td><td>71.4</td><td>85.2</td><td>56.5</td><td>34.3</td></tr><tr><td>state</td><td>✓</td><td>100.0</td><td>98.5</td><td>94.0</td><td>89.3</td><td>95.3</td><td>81.4</td></tr><tr><td>state (w/o entity referrals)</td><td>✓</td><td>77.7</td><td>83.7</td><td>93.6</td><td>91.0</td><td>81.2</td><td>57.1</td></tr><tr><td rowspan="3">8 objects</td><td>state</td><td>X</td><td>18.4</td><td>27.1</td><td>38.1</td><td>87.5</td><td>24.6</td><td>6.7</td></tr><tr><td>state</td><td>✓</td><td>100.0</td><td>98.3</td><td>95.3</td><td>89.8</td><td>91.3</td><td>89.3</td></tr><tr><td>state (w/o entity referrals)</td><td>✓</td><td>60.0</td><td>67.1</td><td>94.1</td><td>81.2</td><td>49.3</td><td>49.3</td></tr><tr><td rowspan="3">6 objects + OOD tasks</td><td>state (8B LLM)</td><td>X</td><td>-</td><td>0</td><td>0</td><td>72.0</td><td>0</td><td>0</td></tr><tr><td>state (8B LLM)</td><td></td><td>-</td><td>49.3</td><td>89.8</td><td>68.5</td><td>28.2</td><td>15.7</td></tr><tr><td>state (62B LLM)</td><td>✓</td><td>-</td><td>48.7</td><td>92.5</td><td>88.1</td><td>40.0</td><td>30.0</td></tr></table>

# B.2. Interactive Language Table

uab 2022).

a as in Lynch et al. (2022). The policy executes 40 steps ( $1 0 \mathrm { H z }$ for 4 seconds) before requiring another command from the The data collection procedure for the real world experiments are the same as in simulation.

Trai nvalationtraeetevos heodels, werai preraie LM-mdelr,00 ale sigtly ifeent erins  theul mixtureFo Tasks  ndation, eplment uat from the prompt alone.

C. Natural Language Generation and Understanding Results   

<table><tr><td>1-shot evals</td><td>PaLM-8B</td><td>PaLM-E-12B (unfrozen)</td><td>PaLM-62B</td><td>PaLM-E-84B (unfrozen)</td><td>PaLM-540B</td><td>PaLM-E-562B (unfrozen)</td><td>Category</td></tr><tr><td>TriviaQA (wiki) (EM)</td><td>48.5</td><td>10.1</td><td>72.7</td><td>31.8</td><td>81.4</td><td>74.6</td><td>NLG</td></tr><tr><td>Natural Questions (EM)</td><td>10.6</td><td>1.6</td><td>23.1</td><td>7.6</td><td>29.3</td><td>27.2</td><td>NLG</td></tr><tr><td>WebQuestions (EM)</td><td>12.6</td><td>3.4</td><td>19.8</td><td>7.9</td><td>22.6</td><td>21.8</td><td>NLG</td></tr><tr><td>Lambada</td><td>57.8</td><td>1.4</td><td>75.5</td><td>26.1</td><td>81.8</td><td>83.3</td><td>NLG</td></tr><tr><td>HellaSwag</td><td>68.2</td><td>48.4</td><td>79.7</td><td>75.3</td><td>83.6</td><td>83.5</td><td>NLU</td></tr><tr><td>StoryCloze</td><td>78.7</td><td>68.7</td><td>83.8</td><td>83.9</td><td>86.1</td><td>86.3</td><td>NLU</td></tr><tr><td>Winograd</td><td>82.4</td><td>71.8</td><td>85.3</td><td>86.4</td><td>87.5</td><td>89.0</td><td>NLU</td></tr><tr><td>Winogrande</td><td>68.3</td><td>55.3</td><td>76.8</td><td>72.5</td><td>83.7</td><td>83.0</td><td>NLU</td></tr><tr><td>RACE-M</td><td>57.7</td><td>43.2</td><td>64.1</td><td>57.4</td><td>69.3</td><td>70.3</td><td>NLU</td></tr><tr><td>RACE-H</td><td>41.6</td><td>33.2</td><td>48.7</td><td>42.3</td><td>52.1</td><td>52.8</td><td>NLU</td></tr><tr><td>PIQA</td><td>76.1</td><td>68.1</td><td>80.9</td><td>78.2</td><td>83.9</td><td>84.9</td><td>NLU</td></tr><tr><td>ARC-e</td><td>71.3</td><td>53.4</td><td>78.9</td><td>71.4</td><td>85.0</td><td>86.3</td><td>NLU</td></tr><tr><td>ARC-c</td><td>42.3</td><td>30.9</td><td>51.8</td><td>46.7</td><td>60.1</td><td>62.6</td><td>NLU</td></tr><tr><td>OpenBookQA</td><td>47.4</td><td>41.4</td><td>51.2</td><td>51.6</td><td>53.6</td><td>55.8</td><td>NLU</td></tr><tr><td>BoolQ</td><td>64.7</td><td>61.6</td><td>83.1</td><td>81.6</td><td>88.7</td><td>89.4</td><td>NLU</td></tr><tr><td>Copa</td><td>82.0</td><td>77.0</td><td>93.0</td><td>91.0</td><td>91.0</td><td>93.0</td><td>NLU</td></tr><tr><td>RTE</td><td>57.8</td><td>54.9</td><td>71.5</td><td>59.6</td><td>78.7</td><td>75.1</td><td>NLU</td></tr><tr><td>Wic</td><td>50.6</td><td>50.0</td><td>48.6</td><td>50.2 75.8</td><td>63.2 86.3</td><td>64.1</td><td>NLU</td></tr><tr><td>WSC</td><td>81.4 87.8</td><td>68.4 71.2</td><td>84.9 91.0</td><td>78.5</td><td>92.8</td><td>85.6 92.5</td><td>NLU</td></tr><tr><td>ReCoRD CB</td><td>41.1</td><td>37.5</td><td>55.4</td><td>73.2</td><td>83.9</td><td>80.3</td><td>NLU NLU</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Avg NLU Avg NLG</td><td>64.7 32.4</td><td>55.0 4.1</td><td>72.3 47.8</td><td>69.2 18.4</td><td>78.2 53.8</td><td>78.5 51.7</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>-4.3%</td><td></td><td></td><td></td></tr><tr><td>NLU delta (%, relative) NLG delta (%, relative)</td><td></td><td>-15.0% -87.3%</td><td></td><td>-61.6%</td><td></td><td>+0.4% -3.8%</td><td></td></tr></table>

D. Additional Data for Affordance and Success Detection   
Tabl 9:Mobile manipulation environment: failure detection, showing individual precision and recall scores.   

<table><tr><td colspan="3">Model</td><td></td><td>Precision</td><td>Recall</td><td>F1-score</td></tr><tr><td colspan="3">PaLI (Zero-shot) (Chen et al., 2022)</td><td></td><td>0.59</td><td>0.98</td><td>0.73</td></tr><tr><td colspan="3">CLIP-FT (Xiao et al., 2022) CLIP-FT-hindsight (Xiao et al., 2022)</td><td></td><td>0.50</td><td>0.95</td><td>0.65</td></tr><tr><td colspan="3">PaLM-E-12B from</td><td>LLM</td><td>1.0</td><td>0.80</td><td>0.89</td></tr><tr><td>trained on</td><td>scratch</td><td>pretrain</td><td>frozen</td><td></td><td></td><td></td></tr><tr><td>Single robot</td><td>✓</td><td>X ✓</td><td>n/a</td><td>0.52 0.91</td><td>0.55</td><td>0.54</td></tr><tr><td>Single robot</td><td>X</td><td>✓</td><td>✓</td><td>0.89</td><td>0.92</td><td>0.91 0.91</td></tr><tr><td>Full mixture</td><td>X</td><td></td><td>✓</td><td></td><td>0.93</td><td></td></tr><tr><td>Full mixture</td><td>X</td><td>✓</td><td>X</td><td>0.66</td><td>0.91</td><td>0.77</td></tr></table>

<table><tr><td colspan="4">Model</td><td>Precision</td><td>Recall</td><td>F1-score</td></tr><tr><td colspan="4">PaLI (Zero-shot) (Chen et al., 2022) QT-OPT (Kalashnikov et al., 2018)</td><td>0.57 0.60</td><td>0.69</td><td>0.62 0.63</td></tr><tr><td>PLM-E-12B</td><td>from scratch</td><td>LLM+ViT pretrain</td><td>LLM</td><td></td><td>0.67</td><td></td></tr><tr><td>trained on Single robot</td><td>✓</td><td></td><td>frozen</td><td></td><td></td><td></td></tr><tr><td>Single robot</td><td>X</td><td>× ✓</td><td>n/a</td><td>0.67 0.90</td><td>0.35 0.69</td><td>0.46 0.78</td></tr><tr><td>Full mixture</td><td>X</td><td>✓</td><td>✓ ✓</td><td>0.95</td><td>0.80</td><td>0.87</td></tr><tr><td>Full mixture</td><td>X</td><td>✓</td><td>X</td><td>0.92</td><td>0.88</td><td>0.91</td></tr></table>

Tab 10obilmanipulation eviromen:fordance predictio, howindividual preisionanrea res.

# E. Image Attribution

The image of the New York Knicks and Boston Celtics in Figure 2 is under the terms CC-by-2.0 (https: / / creativecommons.org/licenses/by/2.0/),andwaspostedtoFlickrbykowarskiathttps://www.flickr. com/photos/27728232@N00/8666371367. The egocentric video images are from https://youtu.be/ -UXKmqBPk 1w, as in (Zeng et al., 2022), via permission from creator Cody Wanner.