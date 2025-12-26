# 基于智能体的视频剪辑

Lingfeng $\mathrm { Y a n g ^ { 1 \dagger } }$ , Zhenyuan 陈2†, Xiang $_ { \mathrm { L i ^ { 3 , 2 * } } }$ , Peiyang Jia4, Liangqu Long4, Jian 杨1\* 1南京科技大学, VCIP, 计算机科学, 南开大学, 3NKIARI, 深圳福田, 4Insta360 {yanglfnjust, csjyang}@njust.edu.cn, {zhenyuanchen, xiang.li.implus@}@nankai.edu.cn jiapeiyang@insta360.com, liangqu.long@gmail.com

# 摘要

随着信息变得更加可获取，用户生成的视频时长日益增加，这给观众带来了筛选大量内容以提取有价值见解的负担。这一趋势凸显了高效提取关键视频信息算法的必要性。尽管在高亮检测、时刻检索和视频摘要等领域取得了显著进展，当前的方法主要集中在选择特定时间区间，往往忽视了片段之间的相关性及片段安排的潜力。本文提出了一项新任务——视频修剪（Video Trimming, VT），该任务着重于检测废弃画面、选择有价值片段，并将其编排成具有连贯故事的最终视频。为了解决这一任务，我们提出了基于智能体的视频修剪（Agent-based Video Trimming, AVT），分为三个阶段：视频结构化、片段过滤和故事构建。具体而言，我们使用视频字幕智能体将视频片段转换为结构化文本描述，采用过滤模块根据每个片段的结构化信息动态剔除低质量镜头，并利用视频排列智能体选择并编排有效片段构成连贯的最终叙述。为了评估，我们开发了视频评估智能体来评估修剪后的视频，同时与人工评估进行对比。此外，我们从互联网收集原始用户视频，策划了一个用于视频修剪的新基准数据集。因此，AVT在用户研究中获得了更为积极的评估，并在YouTube Highlights、TVSum以及我们自己数据集的高亮检测任务上表现出更高的mAP和精度。代码和模型可在https://ylingfeng.github.io/AVT获取。

# 1. 引言

随着视觉信息和日常视频内容的日益增加，迫切需要能够进行视频理解的算法，以有效提取冗余内容中的关键信息。

![](images/1.jpg)  
Figure 1. A comparison between our new task and existing video tasks: (a) Highlight Detection retrieves clips above a saliency threshold. (b) Moment Retrieval identifies the start and end for intervals related to a given query. (c) Video Summarization extracts keyframes for each theme of the video. (d) Video Trimming addresses more than just a retrieval task by also filtering wasted footage and logically composing the selected segments.

理解视频对于减少过量内容至关重要。在视频理解领域取得了显著的进展。基于这些基础，重点检测的方法集中在预测显著性分数，以识别和提取视频中的重要片段，从而减少冗余信息的数量。时刻检索旨在识别与给定查询对应的视频中特定时刻，而视频摘要则编译关键帧以捕捉视频中检测到的主题。然而，当前的方法仅关注内容提取和检索，而未考虑片段之间的关系和连贯性。为了克服这一局限性，我们首次提出了一项新任务：视频修剪（VT）。该任务不仅涉及选择高显著性片段，还包括过滤掉无用镜头并排列剩余剪辑，最终生成一个逻辑结构清晰且连贯的视频输出。现有任务的比较见于图1。

![](images/2.jpg)  
(b) discards defective clips, and finally (c) organizes the remaining clips into a coherent final cut.

为了为此任务建立一个可行的基线，我们考虑利用智能体的概念。最近的观察表明，多模态大语言模型（MLLMs）在上下文交流和格式化交互方面展现出了强大的能力，使其成为无训练视频理解任务（如视频字幕生成和问答）的有效智能体。基于智能体的方法利用具有特别设计提示流程的MLLMs，将视频总结为文本，并根据给定输入组织这些总结。或者，这些模型也可以作为控制器，协调各种执行者，如跟踪和字幕生成系统，以解决复杂的多模态视频任务。尽管在视频理解任务中广泛使用基于智能体的方法，本研究旨在利用其能力开发首个视频剪辑算法。通过整合MLLMs，我们的方法针对长达半小时或更长的长视频进行编辑，将其剪辑为更短的可观看最终片段。具体而言，基于智能体的视频剪辑（AVT）在三个关键阶段展示了创新：视频结构化、片段过滤和故事构成。在视频结构化阶段，视频被划分为更小的单元。视频字幕生成智能体然后将这些单元转换为结构化文本描述，使每个片段的详细语义分析成为可能。因此，后续过程不涉及视觉内容，从而提高了效率和速度。除了生成基本描述外，我们还整合了标志视频片段缺陷的属性，如遮挡、抖动、过曝和无意义内容，以评估帧的质量。片段过滤阶段利用动态模块通过分析结构化文本描述选择有用片段，以区分有价值和无关的内容。在故事构成阶段，视频安排智能体将选择的片段组合成一个连贯的最终视频，确保叙述流畅且引人入胜。此外，我们设计了一个视频评估智能体来评估剪辑视频的质量。我们还创建了一个新的视频剪辑基准，其中包括带有废弃和高亮标签的原始用户视频。为了进行评估，我们在三个基准上进行用户研究和关于零-shot高亮检测的量化实验：YouTube Highlights、TVSum和我们的数据集。我们的贡献总结如下：据我们所知，我们是首个引入视频剪辑（VT）这一新任务的研究，旨在从用户拍摄的长视频中提取关键意图，以生成连贯叙事的缩减视频。为建立基线，我们提出了AVT算法，该算法将视频内容转换为结构化描述，过滤废弃片段，并将选定的片段安排成一个连贯的最终叙事视频。我们通过整合网络视频并使用视频评估智能体评估视频质量，与人工评估相结合，提出了一个新的视频剪辑基准。我们的方法在视频剪辑和零-shot高亮检测方面表现优异，用户研究和多个基准皆有证明。

# 2. 相关工作

# 2.1. 视频理解

借助大规模语言模型（MLLMs），开展了一系列研究，以推动视频理解。现有方法涵盖一个或多个任务，例如视频问答 [24, 31, 50, 60, 71]、长视频理解 [54, 54] 和时刻定位 [67]。InternVid [51] 通过对比学习和多阶段预训练构建了一个大规模的视频-文本模型。InternVideo [50, 52] 利用多模态数据扩大视频理解。像 LaViLa [71] 和 Valley [31] 通过微调来提升基于视频的指令理解。Merlin [66] 和 MovieChat [40] 增强视频问答和长视频理解。PaLM-E [11] 将现实世界的感知数据整合到语言模型中，而 SeViLA [67] 则通过关键帧定位进行事件预测。Vid2Seq [60] 和 VideoChat [24] 通过微调实现了以聊天为中心的视频理解。最近，像 LongVLM [54] 和 VTimeLLM [19] 的模型通过分段和识别时刻改进了长视频的理解。与解决通用视频任务不同，我们的方法专注于视频剪辑，利用基础模型作为核心组件。

# 2.2. 视频智能体

现有的视频智能体方法主要分为两种发展类型。第一种类型利用大型语言模型（LLMs）与外部工具和代码执行器相结合。DoraemonGPT 通过使用符号记忆来增强 VQA 任务，以实现更好的检索和摘要。同样，InternGPT 通过互动投票提高推理能力，而 MM-ReAct 将 REACT 机制扩展到多模态任务。Video ChatCaptioner 通过多智能体迭代投票加强视频理解。第二种针对特定的视频理解任务，将视频内容转换为文本语料库以进行后续分析。AssistGPT 通过规划、执行、检查和学习的循环来提升 VQA 和时刻检索。ChatVideo 将视频内容结构化为文本数据库，以便于高效查询，而 LLoVi 则专注于利用字幕进行细粒度 VQA 和区间检索。MM-Vid 将多模态信息视为文本数据，VideoAgent 通过迭代投票和相似性匹配来改善时刻检索。第一类视频智能体在视频剪辑方面表现不足，因为它们缺乏专门的模型或工具来解决此任务。对于第二类，尽管视频检索智能体可以根据查询提取片段，但它们常常忽视关键的全局内容，从而影响视频的一致性。相较之下，我们的方法首次通过创建创新的视频处理管道，结合视频智能体系统来应对这一挑战。

# 2.3. 视频时间定位

该任务旨在将视频中的目标片段进行时序定位，以连续区间或离散关键帧的形式，包括亮点检测、时刻检索和视频摘要。一方面，亮点检测 (Highlight Detection) 预测显著性得分以提取亮点片段，捕捉关键视觉或情境时刻。然而，这些基于亮点的方法缺乏连贯视频剪辑所需的时间背景和事件关系。另一方面，时刻检索 (Moment Retrieval) 基于给定查询选择时刻。像 DiDeMo、ActivityNet Caption 和 Charades-STA 等数据集为视频片段提供区域字幕，以促进检索任务。此外，Moment-DETR、QD-DETR、TR-DETR、UniVTG 和 UVCOM 等方法旨在通过互补模块设计解决时刻预测和显著性预测的问题。然而，检索到的片段往往缺乏全面的视频覆盖和背景，并且需要事先的用户查询。最后，视频摘要通过选择最能代表原始视频内容的关键镜头来压缩视频。通用摘要仅仅依赖视觉线索，而基于查询的摘要则允许用户通过指定文本关键词来定制摘要。尽管摘要能够压缩视频，但所选片段是离散的，无法生成连贯且可观看的视频。总之，视频时序定位任务仅专注于片段选择，往往忽略叙事流。而我们提出的视频剪辑任务强调片段选择与构成的结合，在缩短视频时长的同时保持叙事的一致性。

# 3. 方法

在本节中，我们介绍视频剪辑（VT）任务，该任务不仅包括亮点选择，还包括过滤冗余镜头和创建连贯、叙事一致的视频输出。为了为该任务建立基线，我们提出了基于智能体的视频剪辑（AVT），这是一种利用多模态大型语言模型（MLLMs）作为无训练智能体的算法，分为三个阶段：视频结构化、剪辑过滤和故事构建。最后，我们提出了一种基于智能体的评估指标来评估最终剪辑，并辅之以用户研究。

![](images/3.jpg)  
Figure 3. Keyframes from a mountain biking video. Clips marked with red boxes are discarded due to higher defect scores, while clips with green boxes are selected despite minor shaking, as they highlight the dynamic scene of cycling on a mountain path.

# 3.1. 视频结构化

最近的多模态视频任务通过从视觉上下文中提取信息来实现视频理解，以推导语义特征 [7, 24, 25, 32, 69]，或直接生成描述性文本 [26, 46, 49]。在我们的案例中，我们采用后者，以确保与多模态智能体如GPT-4 [2] 或 Gemini [45] 的兼容性。值得注意的是，一旦视频被处理为文本，后续操作就与视觉内容无关，这通过仅处理文本输入来提升处理速度并降低计算成本。

为了结构化视频内容，我们将帧划分为剪辑，每个剪辑的默认持续时间为3秒。对于每个剪辑，按每秒采样一帧以提供视觉输入。与以前仅仅得出视频内容通用描述的工作不同，我们旨在根据拍摄特征评估每个剪辑的质量。从经验来看，手持录制可能包含有缺陷的镜头，例如目标被障碍物遮挡或过度抖动的相机，这两者都会影响观看体验。这些缺陷与视觉线索中描绘的事件之间的关系较弱，因此通常不会被一般总结所捕捉。因此，我们需要专门提取缺陷属性，以便对剪辑质量进行全面评估。为此，我们识别出四种主要缺陷：遮挡、抖动、过曝和无意义的内容，如图2（b）所示。具体而言，无意义的内容是指通过简单过渡、缺乏实质性信息的空镜头所特征化的场景。

# 算法 1 动态滤波算法

键的列表 keys，数字的列表 nums 输出：(filter_flag, highlightflag, highlight_score) 1: 初始化 $s c o r e \gets 0$ $m a x \_ k e y \gets ]$ None 2: 对于每个 (key, num) 在 zip(keys, nums) 中 3: 如果 $n u m \ge s c o r e$ 4: $s c o r e \gets n u m$ 5: $m a x _ { - } k e y \gets k e y$ 6: 结束 7: 结束循环 8: 如果 max_key $=$ [Highlight] 9: 返回 (False, True, score) 10: 否则 11: 返回 (score = 0, False, score) 12: 结束 此外，为了处理冗长的原始视频，有必要消除冗余片段，同时保留亮点和引人入胜的部分。尽管原始字幕提供了详细的信息，但在片段修剪方面显得不足，因为相似的视觉内容可能产生不同的文本描述。为了解决这个问题，我们引入了上下文属性，从四个维度总结视频内容：什么、在哪里、何时和谁，提供关于活动、地点、时间和潜在人物的简要见解。此外，我们设计了一个“亮点”属性，用于衡量每个片段的整体兴奋水平。为了获得上述属性，我们利用多语言大模型（MLLMs）作为视频字幕生成器，为每个片段提取原始字幕、上下文属性和缺陷属性，如图 2(a) 所示。片段 ID 遵循与视频长度相关的自然顺序。我们期望结构化的文本信息由简短的句子或短语组成，除了“亮点”属性和所有缺陷属性外，后者应返回一个范围为 0 到 1 的浮动值，表示一个片段是高光还是表现出特定缺陷的程度。如果值为 0，则表示为负属性。这些分数将在下一部分中用于动态过滤掉无效片段。

# 3.2. 剪辑过滤

与现有的时刻检索方法不同，后者基于与特定查询的一致性对视频进行评分，我们专注于视觉属性来评估画质。我们收集缺陷属性和来自视频标注智能体输出的高亮评分，格式为字符串：“[遮挡]: 0.8; [抖动]: 0.7; [过曝]: 0.0; [无意义]: 0.0; [高亮]: $\it 0 . 9 ^ { * }$。”该输出包括四个缺陷指示符和一个高亮评分，用于过滤机制的输入。具体而言，一种常见做法是过滤掉所有片段

# 用户提示

请提供需要翻译的内容。

# 故事创作

# 任务介绍

您是专业的视频编辑，专注于视频剪辑。根据一系列视频剪辑的描述，您的目标是选择合适的剪辑，将其组合成具有完整故事的视频，包括开头、发展和结尾。

# 组成步骤

# 1. 一般理解

# 剪辑信息

# 片段 ID: 1

一位女性在城市天际线前介绍她的 vlog，然后在她的厨房继续讲述。

总体理解：该视频集展示了一位女性介绍她的 vlog，参与日常厨房活动，并照顾她的宠物狗。根据主题，可以分为：[2]：vlog 介绍；[3, 39, 40, 41, 42, 49, 50, 51, 54]：厨房活动与宠物照顾；[67, 68, 70, 72, 75, 81]：城市探险与互动；[125, 126, 127, 129]：客厅的游戏时间。开头是一位女性在城市天际线前介绍她的 vlog，并在厨房里交谈；选择剪辑 ID：[2] 发展部分跟随总结的主题；选择剪辑 ID：[3, 39, 40, 41, 42, 49, 50, 51, 54, 70, 7, 8, 1, 12, 127] 结尾是一只小白狗在客厅中活跃地玩耍，以生动愉快的场景结束该序列；选择剪辑 ID：[129] 代理输出任何负缺陷分数大于零。然而，这并不总是实用的。在第一人称视角拍摄的视频中，镜头抖动是不可避免的，代理会将剪辑标记为抖动，从而导致有用内容的排除。为了解决这个问题，我们引入了一个正向指标，以平衡负向指标。假设是确保算法更关注丰富的视频内容本身，而不是小的拍摄缺陷。基于这一策略，只有当剪辑的“亮点”分数超过所有缺陷分数时，该剪辑才被选为最终合成的有效内容。这个机制被称为动态过滤器，平衡内容丰富性与拍摄缺陷。如图 3 所示，我们可视化了来自剪辑的多样帧，以展示这一过滤规则的应用。

![](images/4.jpg)

![](images/5.jpg)

为了详细理解，我们在算法1中呈现字符串处理算法。该算法通过将结构化数据解析为属性-分值对来处理数据。对于返回值，缺陷标志决定是否应过滤某个片段，而高亮标志和分值则在故事创作的下一阶段进一步使用。

# 3.3. 故事创作

在本节中，我们介绍一个故事创作的智能体，它将过滤后的视频片段安排成连贯的顺序。对于用户提示，我们向视频智能体介绍任务，并结合思维链 [53]（CoT）生成视频创作步骤，考虑全局概念、片段选择和排版安排（见图4）。我们将整个用户提示记作 $P$。然后，假设我们获得了 $M$ 个有效片段，其索引为 $C = \{ C _ { 1 } , C _ { 2 } , \dots , C _ { M } \}$，我们通过连接结构化信息来格式化用户输入 $I$，具体如下：

$$
\begin{array} { c } { { I = \{ \{ C l i p I D \} _ { k } , \{ H i g h l i g h t F l a g \} _ { k } ( \{ S c o r e \} _ { k } ) , } } \\ { { \{ C o n t e x t u a l A t t r i b u t e s \} _ { k } , } } \\ { { \{ R a w C a p t i o n \} _ { k } \} \mid _ { C l i p - k } \{ C _ { 1 } \sim C _ { M } \} , } } \end{array}
$$

高亮标志及相应的得分来源于过滤阶段，而其余信息则来自于结构化阶段。接下来，我们通过 $P$ 和 $I$ 向视频排列智能体发出提示，期望输出的故事线由首选序列构成，该序列是通过将每个句子映射到其相应的剪辑索引以合乎逻辑的顺序生成的。在处理后，我们期待输出包含复合剪辑的序列，记作 $C^{t}$，以及叙述和其组织背后的推理，如图 2 (c) 所示。合成阶段可以反复进行，直到达到所需的视频长度。值得注意的是，一次处理过多剪辑可能导致模糊输出，因为大语言模型在长上下文中面临困难，并易于分心。因此，我们将剪辑分组，并并行调用智能体进行初始处理。随着剪辑数量的减少，所有信息被整合到最终合成中。随后，我们将最终剪辑索引映射回其对应的视频时长，并将其组装成最终视频。值得注意的是，并非所有剪辑都会被选中，剪辑的顺序可能并不严格遵循时间顺序；相反，它们将根据智能体组织的故事线进行排列。有关提示设计的详细信息，请参见补充材料。

Table 1. User study through blind testing, using a scale from 1 to 10, of different methods on the video trimming dataset, comprising 30 final cuts from 42 raw videos and involving 17 participants.   

<table><tr><td>Method</td><td>Richness</td><td>Appeal</td><td>Excitement</td><td>Wasted</td><td>Overall</td><td> Agent</td></tr><tr><td>UniVTG [27]</td><td>6.41</td><td>7.15</td><td>4.74</td><td>6.04</td><td>6.30</td><td>3.03</td></tr><tr><td>UVCOM [57]</td><td>6.15</td><td>7.12</td><td>4.69</td><td>6.47</td><td>6.23</td><td>2.91</td></tr><tr><td>AVT (ours)</td><td>7.21</td><td>7.78</td><td>5.57</td><td>6.72</td><td>7.15</td><td>3.32</td></tr></table>

现有的方法主要集中在图像与文本的匹配上，强调检索到的视频片段的准确性，并旨在最大化它们的整体显著性。然而，在视频剪辑的背景下，这并不是唯一的目标。一个讲述得当的故事应该优先考虑最突出的活动，同时也要包含引入和结尾的部分。尽管开头和结尾可能看起来不如主要内容显著，但它们是至关重要的。我们强调画面构图的重要性，这不仅通过剪辑的排列实现，还通过融入稍微不那么突出的部分来实现。这些部分可以有效地作为过渡，连接前后的内容。

# 3.4. 最终剪辑评估

如 G-Eval [29] 所示，基于大型语言模型（LLM）的评估器具备评估自然语言生成质量的能力。我们将这种自动评估扩展到多模态任务，通过利用 LLM 作为视频评估智能体来评估最终视频。直接提示多模态大型语言模型进行美学评估往往与人类评估不太一致 [12, 47]。为了提高评估的准确性，我们定义了评估标准并为视频评估智能体创建了链式推理（CoT）指令。评估标准按 1 到 5 分评级，包括材料丰富度，评估多样性和叙事连贯性；吸引力，测量参与度、时长和娱乐性；精彩片段，评估亮点质量和频率；以及浪费镜头的数量，考虑无关内容，分数越高表示干扰越少、观看体验越好。视频评估智能体仅使用视频内容作为输入，并为每个指标输出分数及其依据。例如：“[材料丰富度]: {原因} (2.5); [吸引力]: {原因} (3.0); [精彩片段]: {原因} (3.5); [浪费镜头的数量]: $\{ 原因 \}$ (2.0);。”我们计算所有分数的平均值，以确定视频剪辑的最终评级。

# 4. 实验

在本节中，我们首先介绍数据集和实施细节。接着，我们比较修剪视频的质量，并进行精彩片段检测的定量实验。然后，展示有关AVT主要组件的消融研究。最后，我们提供结果可视化和案例研究，以便进行进一步讨论。

# 4.1. 数据集

现有数据集。YouTube Highlights [43] 和 TVSum [41] 数据集是评估视频时间定位任务的两个成熟基准。我们在数据的 $20 \%$ 上进行了测试，按照 [27, 28] 中相同的划分。对于评估指标，我们遵循 [28] 中描述的方法，使用 mAP 来评估 YouTube Highlights 数据集，使用 Top-5 mAP 来评估 TVSum 数据集。视频剪辑数据集。此外，我们从 YouTube 收集网络爬取的视频，并专门构建了一个用于视频剪辑的基准。我们将常见的用户视频类型分为三类：日常生活、体育和旅游 vlog。在每个类别中，我们选择 10 位视频上传者，并选择一或多个围绕一致事件拍摄的视频，这意味着算法可能需要从多个源视频中组合视频剪辑。总共编制了 30 个主题，共 42 个视频，每个视频平均时长 10 分钟。我们为每个视频标注了四个等级的分数：0 表示废弃，1 表示模糊，2 表示正常，3 表示精彩片段。详细说明见补充材料。

# 4.2. 实现细节

为了增强多模态交互能力并确保输出格式的限制，我们在我们的AVT算法中使用GPT-4o模型实现所有智能体。对于视觉输入，视频被划分为3秒的片段，默认每秒抽取一个关键帧。所有帧图像的短边调整为512像素。文本输入包括我们设计的提示指令以及结构化的视频字幕和属性。该配置对于单个10分钟的原始视频产生约153,000个输入图像词元、100,000个输入文本词元以及20,000个输出文本词元，使用API的成本约为0.83美元。输出视频的时长限制为约一分钟，以确保公平比较。更多细节请参见补充材料。

# 4.3. 视频裁剪的比较

人类评估。我们基于构建的视频剪辑数据集进行了用户研究。我们设计了一项盲测，随机打乱来自每种方法的输出视频的顺序。对于现有的高亮检测方法，我们利用其预训练模型生成显著性评分，并通过连接得分最高的视频片段来获得最终视频。十七位参与者被要求在五个方面对这些视频进行评分，这些方面与视频评估代理的标准相似：素材丰富性、吸引力、精彩片段、浪费镜头的数量，以及总体感知评分。表1显示，我们的AVT在最终剪辑质量上取得了整体改善，得益于对浪费镜头的过滤和片段组合过程。我们还在最右侧列出了代理评估得分，这与人类评估的排名一致。

Table 2. Evaluation agent performance of different methods on the validation set of YouTube Highlights and TVSum dataset.   

<table><tr><td>Dataset</td><td>| Method</td><td>Richness</td><td></td><td>Appeal Excitement</td><td>Wasted</td><td>Average</td></tr><tr><td rowspan="4">e</td><td> UMT [28]</td><td>2.70</td><td>3.08</td><td>3.40</td><td>3.44</td><td>3.15</td></tr><tr><td>UniVTG [27]</td><td>2.67</td><td>3.06</td><td>3.35</td><td>3.39</td><td>3.12</td></tr><tr><td>UVCOM [57]</td><td>2.72</td><td>3.10</td><td>3.45</td><td>3.45</td><td>3.18</td></tr><tr><td>AVT (ours)</td><td>2.79</td><td>3.17</td><td>3.53</td><td>3.44</td><td>3.23</td></tr><tr><td rowspan="5">unsaL</td><td>| PGL-SUM [4]</td><td>2.75</td><td>3.05</td><td>3.10</td><td>3.10</td><td>3.00</td></tr><tr><td>UniVTG [27]</td><td>2.65</td><td>2.95</td><td>2.85</td><td>3.15</td><td>2.90</td></tr><tr><td>UVCOM [57]</td><td>2.50</td><td>2.80</td><td>2.70</td><td>3.30</td><td>2.83</td></tr><tr><td>AVT (ours)</td><td>3.15</td><td>3.35</td><td>3.25</td><td>3.70</td><td>3.36</td></tr></table>

智能体评估。根据第3.4节的内容，我们使用设计的视频评估智能体评估生成视频的质量。我们在YouTube Highlights和TVSum数据集的验证集上进行实验，分别使用150个和10个视频。我们将我们的方法与三种相关的先前方法进行比较。表2显示，我们的AVT在指标上优于现有方法。值得注意的是关于与人工评估一致性的消融研究将在第4.5节中进一步阐述。

# 4.4. 高亮检测的比较

在本节中，我们将我们的方法与之前的高亮检测和视频摘要方法进行比较。我们将 AVT 的显著性评分定义如下：

$$
S _ { i } = { \left\{ \begin{array} { l l } { S _ { h } } & { { \mathrm { i f } } \ S _ { h } > m a x ( S _ { d } ) , } \\ { S _ { h } - m a x ( S _ { d } ) } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. }
$$

其中 $i$ 代表 $C l i p ~ I D$，$S _ { h }$ 和 $S _ { d }$ 表示结构化剪辑描述中的高亮和缺陷评分。我们首先在 YouTube Highlights 和 TVSum 上开展实验。我们报告完全/弱监督方法的原始论文结果。接下来，我们在零样本迁移的方面比较这些方法。值得注意的是，对于 UniVTG [27]，我们直接复制他们的零样本结果。对于其他方法，我们使用在最大规模数据集上预训练的模型进行推理，例如 QVHighlights [22] 和 CharadesSTA [15]。我们遵循 [27, 28] 的验证划分在 YouTube Highlights 数据集上进行比较。由于 TVSum 的规模较小，其验证集仅包含 10 个视频，这可能导致不一致的评分，因此我们在零样本设置中对最多 50 个视频进行测量。如表 3 和表 4 所示，我们的 AVT 在零样本迁移下实现了最先进的高亮检测性能，并与部分监督方法进行训练后可相媲美。

![](images/6.jpg)  
Figure 5. Highlight detection results of mAP and precision on our collected video trimming dataset.

接下来，使用我们构建的视频剪辑数据集，我们展示了现有方法在高亮检测中的平均精确度（mAP）以及最终视频中选定剪辑的高亮片段的准确性。在图5中，我们观察到，与之前的方法相比，AVT视频中选择的废素材更少，因为它们并不专注于素材过滤。此外，我们的方法提取了更多的高亮片段。

# 4.5. 消融研究

AVT的组件。在这一部分，我们分析AVT中每个组件的有效性，包括结构化阶段、滤波阶段和动态模块。此外，我们比较了用时间顺序简单双视频片段拼接替代故事构建阶段的结果。对于所有没有构建阶段的实验，选择了具有最高显著性评分的片段。对于控制条件，在所有组件禁用的情况下，我们随机选择视频片段。我们进行了用户研究，并对废料/亮点片段比例进行了定量精度测量，以评估生成视频的质量。表5显示，片段过滤显著减少了废料，而构建过程提升了整体视频表现。值得注意的是，尤其是在体育内容中，如果没有动态过滤模块，亮点片段可能会因为缺陷属性而被丢弃。进一步的分析详见补充材料。评估智能体的人类相关性。根据GEval，我们采用三种元评估指标：Pearson $( r )$ , Spearman $( \rho )$ 和Kendall-Tau $( \tau )$ ，以测量我们的评估智能体与人类偏好之间的相关性。我们进行了消融实验，研究请求智能体同时提供评分原因的影响，以及使用第3.4节所述的多样化标准的效果。在同时激活这两种策略时，我们的平均相关性达到0.5247，如表6所示。

# 4.6. 可视化

我们在图6中可视化了每种方法的显著性得分和选择的时间区间。由于现有方法不包括组合阶段，它们的最终视频是通过串联高显著性画面构建的，导致观看体验不一致且内容丰富性有限。AVT在选择更多精彩画面和更少浪费画面的同时，超越了以往的工作，并保持了与原始视频一致的叙事线。

<table><tr><td>Method</td><td>Sup</td><td>Dog</td><td>Gym.</td><td>Par.</td><td>Ska.</td><td>Ski.</td><td>Sur.</td><td>Avg.</td></tr><tr><td>LSVM [43]</td><td>FS</td><td>60.0</td><td>41.0</td><td>61.0</td><td>62.0</td><td>36.0</td><td>61.0</td><td>53.6</td></tr><tr><td>Trailer [48]</td><td>FS</td><td>63.3</td><td>82.5</td><td>62.3</td><td>52.9</td><td>74.5</td><td>79.3</td><td>69.1</td></tr><tr><td>SL-Module [59]</td><td>FS</td><td>70.8</td><td>53.2</td><td>77.2</td><td>72.5</td><td>66.1</td><td>76.2</td><td>69.3</td></tr><tr><td>Joint-VA† [5]</td><td>FS</td><td>64.5</td><td>71.9</td><td>80.8</td><td>62.0</td><td>73.2</td><td>78.3</td><td>71.8</td></tr><tr><td>UMT† [28]</td><td>FS</td><td>65.9</td><td>75.2</td><td>81.6</td><td>71.8</td><td>72.3</td><td>82.7</td><td>74.9</td></tr><tr><td>UniVTG [27]</td><td>FS</td><td>74.3</td><td>79.0</td><td>74.4</td><td>84.9</td><td>75.1</td><td>83.9</td><td>78.6</td></tr><tr><td>UVCOM [57]</td><td>FS</td><td>73.8</td><td>77.1</td><td>75.7</td><td>75.3</td><td>74.0</td><td>82.7</td><td>76.4</td></tr><tr><td>LIM-S [58]</td><td>WS</td><td>57.9</td><td>41.7</td><td>67.0</td><td>57.8</td><td>48.6</td><td>65.1</td><td>56.4</td></tr><tr><td>MINI-Net† [18]</td><td>WS</td><td>58.2</td><td>61.7</td><td>70.2</td><td>72.2</td><td>58.7</td><td>65.1</td><td>64.4</td></tr><tr><td>TCG† [65]</td><td>WS</td><td>55.4</td><td>62.7</td><td>70.9</td><td>69.1</td><td>60.1</td><td>59.8</td><td>63.0</td></tr><tr><td>RRAE [61]</td><td>ZS</td><td>49.0</td><td>35.0</td><td>50.0</td><td>25.0</td><td>22.0</td><td>49.0</td><td>38.3</td></tr><tr><td>UniVTG [27]</td><td>ZS</td><td>48.8</td><td>57.5</td><td>59.4 39.7</td><td></td><td>57.4</td><td>49.1</td><td>52.0</td></tr><tr><td>UVCOM [57]</td><td>ZS</td><td>46.6</td><td>67.4</td><td>61.4</td><td>57.2</td><td>63.5</td><td>60.9</td><td>59.5</td></tr><tr><td>AVT (ours)</td><td>ZS</td><td>58.0</td><td>62.1</td><td>76.1</td><td>32.0</td><td>67.1</td><td>67.9</td><td>60.5</td></tr></table>

<table><tr><td>Method</td><td>Sup</td><td>VT</td><td>VU</td><td>GA</td><td>MS</td><td>PK</td><td>PR</td><td>FM</td><td>BK</td><td>BT</td><td>DS</td><td>Avg.</td></tr><tr><td>sLSTM [70]</td><td>FS</td><td>41.1</td><td>46.2</td><td>46.3</td><td>47.7</td><td>44.8</td><td>46.1</td><td>45.2</td><td>40.6</td><td>47.1</td><td>45.5</td><td>45.1</td></tr><tr><td>Trailer [48]</td><td>FS</td><td>61.3</td><td>54.6</td><td>65.7</td><td>60.8</td><td>59.1</td><td>70.1</td><td>58.2</td><td>64.7</td><td>65.6</td><td>68.1</td><td>62.8</td></tr><tr><td>SL-Module [59]</td><td>FS</td><td>86.5</td><td>68.7</td><td>74.9</td><td>86.2</td><td>79.0</td><td>63.2</td><td>58.9</td><td>72.6</td><td>78.9</td><td>64.0</td><td>73.3</td></tr><tr><td>Joint-VA† [5]</td><td>FS</td><td>83.7</td><td>57.3</td><td>78.5</td><td>86.1</td><td>80.1</td><td>69.2</td><td>70.0</td><td>73.0</td><td>97.4</td><td>67.5</td><td>76.3</td></tr><tr><td>UMT [28]</td><td>FS</td><td>87.5</td><td>81.5</td><td>88.2</td><td>78.8</td><td>81.5</td><td>87.0</td><td>76.0</td><td>86.9</td><td>84.4</td><td>79.6</td><td>83.1</td></tr><tr><td>UniVTG [27]</td><td>FS</td><td>92.0</td><td>77.8</td><td>89.8</td><td>83.8</td><td>82.2</td><td>85.8</td><td>74.3</td><td>91.8</td><td>90.5</td><td>77.6</td><td>84.6</td></tr><tr><td>UVCOM [57]</td><td>FS</td><td>87.6</td><td>91.6</td><td>91.4</td><td>86.7</td><td>86.9</td><td>86.9</td><td>76.9</td><td>92.3</td><td>87.4</td><td>75.6</td><td>86.3</td></tr><tr><td>LIM-S [58]</td><td>WS</td><td>55.9</td><td>42.9</td><td>61.2</td><td>54.0</td><td>60.4</td><td>47.5</td><td>43.2</td><td>66.3</td><td>69.1</td><td>62.6</td><td>56.3</td></tr><tr><td>MINI-Net† [18]</td><td>WS</td><td>80.6</td><td>68.3</td><td>78.2</td><td>81.8</td><td>78.1</td><td>65.8</td><td>57.8</td><td>75.0</td><td>80.2</td><td>65.5</td><td>73.2</td></tr><tr><td>TCG† [65]</td><td>WS</td><td>85.0</td><td>71.4</td><td>81.9</td><td>78.6</td><td>80.2</td><td>75.5</td><td>71.6</td><td>77.3</td><td>78.6</td><td>68.1</td><td>76.8</td></tr><tr><td>SG [33]</td><td>ZS</td><td>42.3</td><td>47.2</td><td>47.5</td><td>48.9</td><td>45.6</td><td>47.3</td><td>46.4</td><td>41.7</td><td>48.3</td><td>46.6</td><td>46.2</td></tr><tr><td>UniVTG [27]</td><td>ZS</td><td>52.0</td><td>48.1</td><td>50.9</td><td>56.9</td><td>51.6</td><td>43.3 60.0 64.0 59.2 54.9</td><td></td><td></td><td></td><td></td><td>54.1</td></tr><tr><td>UVCOM [57]</td><td>ZS</td><td>63.4</td><td>44.5</td><td>50.6</td><td>67.6</td><td>55.1</td><td>42.0</td><td>47.5</td><td>56.9</td><td>58.6</td><td>39.3</td><td>52.5</td></tr><tr><td>AVT (ours)</td><td>ZS</td><td>76.6</td><td>75.9</td><td>62.4</td><td>63.9</td><td>76.6</td><td>68.8</td><td>39.4</td><td>45.6</td><td>43.4</td><td></td><td>62.9 61.6</td></tr></table>

![](images/7.jpg)  
Table 3. Highlight detection results of mAP on YouTube Highlights. $\dagger$ denotes using audio modality.   
Table 4. Highlight detection results of Top-5 mAP on TVSum. $^ \dagger$ denotes using audio modality. FS: Fully supervised. WS: Weakly supervised. ZS: Zero-shot.   
footage and less wasted footage.

Table 5. Ablation study on the effectiveness of AVT components. VS: Video Structuring. CF: Clip Filtering. DF: Dynamic Filter. SC: Story Composition.   

<table><tr><td>Method</td><td>VS</td><td>CF</td><td>DF</td><td>SC</td><td>User ↑</td><td>Waste ↓</td><td>Highlight ↑</td></tr><tr><td rowspan="2">UniVTG [27] UVCOM [57]</td><td></td><td>-</td><td>-</td><td>-</td><td>6.30</td><td>0.276</td><td>0.066</td></tr><tr><td></td><td>-</td><td></td><td></td><td>6.23</td><td>0.175</td><td>0.066</td></tr><tr><td rowspan="6">AVT (ours)</td><td></td><td></td><td>-</td><td>-</td><td>3.70</td><td>0.337</td><td>0.083</td></tr><tr><td></td><td></td><td></td><td>-</td><td>6.19</td><td>0.135</td><td>0.110</td></tr><tr><td></td><td></td><td>-</td><td></td><td>6.45</td><td>0.165</td><td>0.096</td></tr><tr><td></td><td></td><td></td><td></td><td>6.70</td><td>0.141</td><td>0.109</td></tr><tr><td></td><td></td><td></td><td>L</td><td>5.23</td><td>0.199</td><td>0.107</td></tr><tr><td></td><td></td><td>V</td><td>L</td><td>7.15</td><td>0.083</td><td>0.108</td></tr></table>

Table 6. Pearson $( r )$ , Spearman $( \rho )$ , and Kendall-Tau $( \tau )$ correlations of different metrics on video trimming benchmark.   

<table><tr><td>Output Reason Diverse Criteria</td><td></td><td>r</td><td>ρ</td><td>τ</td><td>Avg.</td></tr><tr><td>-</td><td>-</td><td>0.2675</td><td>0.2451</td><td>0.1723</td><td>0.2283</td></tr><tr><td>-</td><td>✓</td><td>0.4082</td><td>0.4119</td><td>0.3067</td><td>0.3756</td></tr><tr><td>✓</td><td>-</td><td>0.5260</td><td>0.4990</td><td>0.3738</td><td>0.4663</td></tr><tr><td>✓</td><td>✓</td><td>0.5616</td><td>0.5667</td><td>0.4457</td><td>0.5247</td></tr></table>

# 5. 结论

在本文中，我们引入了一项新的任务——视频剪辑（VT），该任务聚焦于片段选择和叙事保存，以从冗余内容中提取有意义的见解。为了解决这一任务，我们提出了基于智能体的视频剪辑（AVT），这是一个基线框架，包含三个关键阶段：视频结构化，在此阶段，视频字幕智能体提供片段描述；剪辑过滤，动态选择剪辑的过滤模块；以及故事构成，在该阶段，视频排列智能体创建 cohesive narrative。进一步地，我们设计了一个视频评估智能体用于评估视频质量。我们构建了一个标注的视频剪辑任务基准。我们的方案在高光检测方面优于现有方法，并在用户研究中展现出优越的人类偏好。

# References

[1] Claude 3. Introducing the next generation of claude. https://www.anthropic.com/news/claude-3- family, 2024. 2, 3 [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,   
2023. 2, 3, 4 [3] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In ICCV, 2017. 1,   
3 [4] Evlampios Apostolidis, Georgios Balaouras, Vasileios Mezaris, and Ioannis Patras. Combining global and local attention with positional encoding for video summarization. In ISM, 2021. 1, 7 [5] Taivanbat Badamdorj, Mrigank Rochan, Yang Wang, and Li Cheng. Joint visual and audio learning for video highlight detection. In ICCV, 2021. 3, 8 [6] Taivanbat Badamdorj, Mrigank Rochan, Yang Wang, and Li Cheng. Contrastive learning for unsupervised video highlight detection. In CVPR, 2022. 3 [7] Guo Chen, Yin-Dong Zheng, Jiahao Wang, Jilan Xu, Yifei Huang, Junting Pan, Yi Wang, Yali Wang, Yu Qiao, Tong Lu, et al. Videollm: Modeling video sequence with large language models. arXiv preprint arXiv:2305.13292, 2023. 4 [8] Jun Chen, Deyao Zhu, Kilichbek Haydarov, Xiang Li, and Mohamed Elhoseiny. Video chatcaptioner: Towards enriched spatiotemporal descriptions. arXiv preprint arXiv:2304.04227, 2023. 3 [9] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024. 2, 3   
10] Sandra Eliza Fontes De Avila, Ana Paula Brandao Lopes, Antonio da Luz Jr, and Arnaldo de Albuquerque Araújo. Vsumm: A mechanism designed to produce static video summaries and a novel evaluation method. Pattern recognition letters, 2011. 13   
11] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palme: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023. 3   
[12] Jinlan Fu, See-Kiong Ng, Zhengbao Jiang, and Pengfei Liu. Gptscore: Evaluate as you desire. arXiv preprint arXiv:2302.04166, 2023. 6   
[13] Valentin Gabeur, Chen Sun, Karteek Alahari, and Cordelia Schmid. Multi-modal transformer for video retrieval. In ECCV, 2020. 1   
[14] Difei Gao, Lei Ji, Luowei Zhou, Kevin Qinghong Lin, Joya Chen, Zihan Fan, and Mike Zheng Shou. Assistgpt: A general multi-modal assistant that can plan, execute, inspect, and learn. arXiv preprint arXiv:2306.08640, 2023. 2, 3   
[15] Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. Tall: Temporal activity localization via language query. In ICCV, 2017. 3, 7, 13   
[16] Michael Gygli, Helmut Grabner, Hayko Riemenschneider, and Luc Van Gool. Creating summaries from user videos. In ECCV, 2014. 1, 3, 13   
[17] Michael Gygli, Yale Song, and Liangliang Cao. Video2gif: Automatic generation of animated gifs from video. In CVPR, 2016.3   
[18] Fa-Ting Hong, Xuanteng Huang, Wei-Hong Li, and WeiShi Zheng. Mini-net: Multiple instance ranking network for video highlight detection. In ECCV, 2020. 3, 8   
[19] Bin Huang, Xin Wang, Hong Chen, Zihan Song, and Wenwu Zhu. Vtimellm: Empower llm to grasp video moments. In CVPR, 2024. 3   
[20] Hao Jiang and Yadong Mu. Joint video summarization and moment localization by cross-task sample transfer. In CVPR, 2022. 3   
[21] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In ICCV, 2017. 1, 3   
[22] Jie Lei, Tamara L Berg, and Mohit Bansal. Detecting moments and highlights in videos via natural language queries. NeurIPS, 2021. 1, 3, 7   
[23] Chenliang Li, Haiyang Xu, Junfeng Tian, Wei Wang, Ming Yan, Bin Bi, Jiabo Ye, Hehong Chen, Guohai Xu, Zheng Cao, et al. mplug: Effective and efficient vision-language learning by cross-modal skip-connections. arXiv preprint arXiv:2205.12005, 2022. 2   
[24] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 3, 4   
[25] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023. 1, 4   
[26] Kevin Lin, Faisal Ahmed, Linjie Li, Chung-Ching Lin, Ehsan Azarnasab, Zhengyuan Yang, Jianfeng Wang, Lin Liang, Zicheng Liu, Yumao Lu, et al. Mm-vid: Advancing video understanding with gpt-4v (ision). arXiv preprint arXiv:2310.19773, 2023. 2, 3, 4   
[27] Kevin Qinghong Lin, Pengchuan Zhang, Joya Chen, Shraman Pramanick, Difei Gao, Alex Jinpeng Wang, Rui Yan, and Mike Zheng Shou. Univtg: Towards unified videolanguage temporal grounding. In ICCV, 2023. 3, 6, 7, 8, 17   
[28] Ye Liu, Siyuan Li, Yang Wu, Chang-Wen Chen, Ying Shan, and Xiaohu Qie. Umt: Unified multi-modal transformers for joint video moment retrieval and highlight detection. In CVPR, 2022. 6, 7, 8   
[29] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. G-eval: Nlg evaluation using gpt-4 with better human alignment. arXiv preprint arXiv:2303.16634, 2023. 6, 7   
[30] Zhaoyang Liu, Yinan He, Wenhai Wang, Weiyun Wang, Yi Wang, Shoufa Chen, Qinglong Zhang, Zeqiang Lai, Yang Yang, Qingyun Li, et al. Interngpt: Solving vision-centric tasks by interacting with chatgpt beyond language. arXiv preprint arXiv:2305.05662, 2023. 2, 3   
[31] Ruipu Luo, Ziwang Zhao, Min Yang, Junwei Dong, Da Li, Pengcheng Lu, Tao Wang, Linmei Hu, Minghui Qiu, and Zhongyu Wei. Valley: Video assistant with large language model enhanced ability. arXiv preprint arXiv:2306.07207, 2023. 3   
[32] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In ACL, 2024. 1, 4   
[33] Behrooz Mahasseni, Michael Lam, and Sinisa Todorovic. Unsupervised video summarization with adversarial lstm networks. In CVPR, 2017. 3, 8   
[34] Niluthpol Chowdhury Mithun, Sujoy Paul, and Amit K RoyChowdhury. Weakly supervised video moment retrieval from text queries. In CVPR, 2019. 1, 3   
[35] WonJun Moon, Sangeek Hyun, SangUk Park, Dongchan Park, and Jae-Pil Heo. Query-dependent video representation for moment retrieval and highlight detection. In CVPR, 2023. 1, 3   
[36] Saiteja Nalla, Mohit Agrawal, Vishal Kaushal, Ganesh Ramakrishnan, and Rishabh Iyer. Watch hours in minutes: Summarizing videos with user intent. In ECCV, 2020. 3   
[37] OpenAI. Gpt-4o. https://openai.com/index/ hello-gpt-40, 2024. 6, 12   
[38] Aidean Sharghi, Jacob S Laurel, and Boqing Gong. Queryfocused video summarization: Dataset, evaluation, and a memory network based approach. In CVPR, 2017. 3   
[39] Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H Chi, Nathanael Schärli, and Denny Zhou. Large language models can be easily distracted by irrelevant context. In ICML, 2023. 5   
[40] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, et al. Moviechat: From dense token to sparse memory for long video understanding. In CVPR, 2024. 3   
[41] Yale Song, Jordi Vallmitjana, Amanda Stent, and Alejandro Jaimes. Tvsum: Summarizing web videos using titles. In CVPR, 2015. 1, 2, 3, 6, 13   
[42] Hao Sun, Mingyao Zhou, Wenjing Chen, and Wei Xie. Trdetr: Task-reciprocal transformer for joint moment retrieval and highlight detection. arXiv preprint arXiv:2401.02309, 2024. 3   
[43] Min Sun, Ali Farhadi, and Steve Seitz. Ranking domainspecific highlights by analyzing edited videos. In ECCV, 2014. 1, 2, 3, 6, 8, 13   
[44] Dídac Surís, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. In ICCV, 2023. 1, 2   
[45] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 2, 3, 4   
[46] Junke Wang, Dongdong Chen, Chong Luo, Xiyang Dai, Lu Yuan, Zuxuan Wu, and Yu-Gang Jiang. Chatvideo: A tracklet-centric multimodal and versatile video understanding system. arXiv preprint arXiv:2304.14407, 2023. 2, 3, 4   
[47] Jiaan Wang, Yunlong Liang, Fandong Meng, Zengkui Sun, Haoxiang Shi, Zhixu Li, Jinan Xu, Jianfeng Qu, and Jie Zhou. Is chatgpt a good nlg evaluator? a preliminary study. arXiv preprint arXiv:2303.04048, 2023. 6   
[48] Lezi Wang, Dong Liu, Rohit Puri, and Dimitris N Metaxas. Learning trailer moments in full-length movies with cocontrastive attention. In ECCV, 2020. 8   
[49] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena YeungLevy. Videoagent: Long-form video understanding with large language model as agent. ECCV, 2024. 2, 3, 4   
[50] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, et al. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191, 2022. 2, 3   
[51] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942, 2023. 3, 17   
[52] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Jilan Xu, Zun Wang, et al. Internvideo2: Scaling video foundation models for multimodal video understanding. ECCV, 2024. 3, 17   
[53] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. NeurIPS, 2022. 5   
[54] Yuetian Weng, Mingfei Han, Haoyu He, Xiaojun Chang, and Bohan Zhuang. Longvlm: Efficient long video understanding via large language models. ECCV, 2024. 3   
[55] Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. Visal chatt: Talkig, drawing and editing with visual foundation models. arXiv preprint arXiv:2303.04671, 2023. 2   
[56] Guande Wu, Jianzhe Lin, and Claudio T Silva. Intentvizor: Towards generic query guided interactive video summarization. In CVPR. 2022. 3

[72] Kaiyang Zhou, Yu Qiao, and Tao Xiang. Deep reinforcement learning for unsupervised video summarization with diversity-representativeness reward. In AAAI, 2018. 1

[57] Yicheng Xiao, Zhuoyan Luo, Yong Liu, Yue Ma, Hengwei X  : A unified video comprehension framework for moment retrieval and highlight detection. In CVPR, 2024. 3, 6, 7, 8,   
17 [58] Bo Xiong, Yannis Kalantidis, Deepti Ghadiyaram, and Kristen Grauman. Less is more: Learning highlight detection from video duration. In CVPR, 2019. 1, 3, 8 [59] Minghao Xu, Hang Wang, Bingbing Ni, Riheng Zhu, Zhenbang Sun, and Changhu Wang. Cross-category video highlight detection via set-based learning. In ICCV, 2021. 8 [60] Antoine Yang, Arsha Nagrani, Paul Hongsuck Seo, Antoine Miech, Jordi Pont-Tuset, Ivan Laptev, Josef Sivic, and Cordelia Schmid. Vid2seq: Large-scale pretraining of a visual language model for dense video captioning. In CVPR,   
2023. 1, 2, 3 [61] Huan Yang, Baoyuan Wang, Stephen Lin, David Wipf, Minyi Guo, and Baining Guo. Unsupervised extraction of video highlights via robust recurrent auto-encoders. In ICCV,   
2015.8 [62] Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381, 2023. 2, 3 [63] Zongxin Yang, Guikun Chen, Xiaodi Li, Wenguan Wang, and Yi Yang. Doraemongpt: Toward understanding dynamic scenes with large language models. arXiv preprint arXiv:2401.08392, 2024. 3 [64] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In ICLR, 2023. 3 [65] Qinghao Ye, Xiyue Shen, Yuan Gao, Zirui Wang, Qi Bi, Ping Li, and Guang Yang. Temporal cue guided video highlight detection with low-rank audio-visual fusion. In ICCV, 2021.   
8 [66] En Yu, Liang Zhao, Yana Wei, Jinrong Yang, Dongming Wu, Lingyu Kong, Haoran Wei, Tiancai Wang, Zheng Ge, Xiangyu Zhang, et al. Merlin: Empowering multimodal llms with foresight minds. ECCV, 2024. 3 [67] Shoubin Yu, Jaemin Cho, Prateek Yadav, and Mohit Bansal. Self-chained image-language model for video localization and question answering. NeurIPS, 2023. 2, 3 [68] Ce Zhang, Taixi Lu, Md Mohaiminul Islam, Ziyang Wang, Shoubin Yu, Mohit Bansal, and Gedas Bertasius. A simple llm framework for long-range video question-answering. EMNLP, 2024. 3 [69] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 1, 4 [70] Ke Zhang, Wei-Lun Chao, Fei Sha, and Kristen Grauman. Video summarization with long short-term memory. In ECCV, 2016. 8 [71] Yue Zhao, Ishan Misra, Philipp Krähenbühl, and Rohit Girdhar. Learning video representations from large language models. In CVPR, 2023. 1, 3

# Agent-based Video Trimming

Supplementary Material

# A. Video Trimming Dataset

We collect user-generated videos from YouTube and construct a benchmark for video trimming specifically. The data collection process adheres to three key principles:

1. The current dataset predominantly consists of videos already edited by their creators. In contrast, for video trimming scenarios, we aim to use raw, unedited videos. These raw videos, typically filmed by individuals, often contain imperfections such as occlusion, jitter, or overexposure, reflecting real-world filming conditions.

2. The raw videos selected should be in long form, with durations exceeding 5 minutes, rather than short, pre-edited montages.

3.In practical video trimming tasks, input videos may originate from multiple sources. As a result, algorithms are expected to generate cuts from various videos, whereas existing datasets typically focus on a single video per topic.

Following these principles, we curated a collection of 42 videos uploaded by 30 different users. A comparison with existing datasets is shown in Tab. S2, which highlights that our dataset boasts the longest average video duration, approximately 10 minutes per video.

To ensure diversity in trimming scenarios, we selected videos spanning a range of topics, categorized into three groups: daily life, sports, and travel vlogs. For each category, we chose 10 video uploaders and included one or more videos that revolve around a consistent event. Detailed topics and corresponding YouTube IDs are listed in Tab. S1.

Additionally, we annotated each video with 10 annotators using four ranking levels to evaluate footage quality: 0 for wasted, 1 for ambiguous, 2 for normal, and 3 for highlight footage. Examples of these ground-truth annotations can be found in Sec E.

# B. Prompt Design

This section provides the detailed prompts for AVT, utilizing GPT-4o [37]. It covers prompts for video structuring, story composition, and video evaluation. Figure S1 illustrates the structuring prompt, which includes general captions, defect attributes, and contextual attributes.

Next, we describe the story composition process, which occurs in two stages. The first stage focuses on grouping clips to prevent overwhelming input lengths, prioritizing the selection of highlight segments (Fig. S2). In the second stage, the selected clips are gathered and arranged into a coherent final video, with an emphasis on narrative flow (Fig. S3). Finally, Figure S4 presents the prompt used by GPT to evaluate video quality.

Table S1. Our curated video trimming dataset includes 42 YouTube videos, contributed by 30 users, and spans a wide variety of topics.   

<table><tr><td rowspan=1 colspan=1>Class</td><td rowspan=1 colspan=1>Sub-Class</td><td rowspan=1 colspan=1> User</td><td rowspan=1 colspan=1>YouTube ID(s)</td></tr><tr><td rowspan=5 colspan=1>daily life</td><td rowspan=1 colspan=1>family</td><td rowspan=1 colspan=1>CapperCoolCooperEarls Family VlogsJason BoonThe Semps</td><td rowspan=1 colspan=1>KRqR6eLSoP8MyLwV1V19WYPcxuFef17PYYoIkzzpQjKM</td></tr><tr><td rowspan=1 colspan=1>food</td><td rowspan=1 colspan=1>Soon Films </td><td rowspan=1 colspan=1>McjBGfacfc</td></tr><tr><td rowspan=1 colspan=1>friend</td><td rowspan=1 colspan=1>Shawn Roscoe</td><td rowspan=1 colspan=1>JOnA4VgnoCo</td></tr><tr><td rowspan=1 colspan=1>light show</td><td rowspan=1 colspan=1>World In Nature</td><td rowspan=1 colspan=1>nFPJMj0tq9G</td></tr><tr><td rowspan=1 colspan=1>pets</td><td rowspan=1 colspan=1>Cats with GoProGone to the Snow DogsMs Kendall G</td><td rowspan=1 colspan=1>a98Ra7PaTeEiISfWRiDelgBhxk-O1Y7Ho</td></tr><tr><td rowspan=10 colspan=1>sports</td><td rowspan=1 colspan=1>badminton</td><td rowspan=1 colspan=1>SHUTTLE&amp;MORE</td><td rowspan=1 colspan=1>lyhfZy7tShU</td></tr><tr><td rowspan=1 colspan=1>basketball</td><td rowspan=1 colspan=1>June young Kim</td><td rowspan=1 colspan=1>UNemjRp6YJg</td></tr><tr><td rowspan=2 colspan=1>cycling</td><td rowspan=1 colspan=1>Erkan Sakallioglu</td><td rowspan=1 colspan=1>A06p0jlOd6UiwCSAYGwPq4xLhHU8uo2aY</td></tr><tr><td rowspan=1 colspan=1>Richard Whittle</td><td rowspan=1 colspan=1>g--B5HBR1Y</td></tr><tr><td rowspan=1 colspan=1>motorcycle</td><td rowspan=1 colspan=1>Skaily Production</td><td rowspan=1 colspan=1>MXAzBe7PZOQ</td></tr><tr><td rowspan=1 colspan=1>skateboard</td><td rowspan=1 colspan=1>TOW TRUCK BOB</td><td rowspan=1 colspan=1>VVK1KkIKCYQ</td></tr><tr><td rowspan=1 colspan=1>skating</td><td rowspan=1 colspan=1>HC+</td><td rowspan=1 colspan=1>a8M-5nTrl18bA3CxZllhsIdZ3i-HuhQXM</td></tr><tr><td rowspan=3 colspan=1>skiing</td><td rowspan=1 colspan=1>Alex E</td><td rowspan=1 colspan=1>E50qGoDzNtgdFrfsgW1M98ddxw58h5YAfB8zm1hTvgAh7aeRrf-m-8mqNZjVDZcfYpuAxGH6aWMYxnsTcvtttfY</td></tr><tr><td rowspan=1 colspan=1>Emerson Nishi</td><td rowspan=1 colspan=1>WL4TA--CVcA</td></tr><tr><td rowspan=1 colspan=1>ecosnowsportsTV</td><td rowspan=1 colspan=1>5FE871Ij1DQ</td></tr><tr><td rowspan=10 colspan=1>travel</td><td rowspan=1 colspan=1>amusement park</td><td rowspan=1 colspan=1>Informative Topics Vlogs</td><td rowspan=1 colspan=1>hmImxd681YI</td></tr><tr><td rowspan=1 colspan=1>city walk</td><td rowspan=1 colspan=1>Jahaar views</td><td rowspan=1 colspan=1>tp912d19x4E</td></tr><tr><td rowspan=1 colspan=1>hiking</td><td rowspan=1 colspan=1>thePOVchannel</td><td rowspan=1 colspan=1>I1gGZa4-h_U</td></tr><tr><td rowspan=1 colspan=1>luge</td><td rowspan=1 colspan=1>Travel&amp;Adventure Junkies</td><td rowspan=1 colspan=1>dxz00qnhrPo</td></tr><tr><td rowspan=1 colspan=1>mountaineering</td><td rowspan=1 colspan=1>stivn</td><td rowspan=1 colspan=1>iUMBQugUtVQ</td></tr><tr><td rowspan=1 colspan=1>rafting</td><td rowspan=1 colspan=1>All About Shenzhen</td><td rowspan=1 colspan=1>A7Ys5d-Zwro</td></tr><tr><td rowspan=1 colspan=1>road trip</td><td rowspan=1 colspan=1>Mojo Rides</td><td rowspan=1 colspan=1>ePKyNYP7uNg</td></tr><tr><td rowspan=2 colspan=1>show</td><td rowspan=1 colspan=1>HetfieldMustaine22</td><td rowspan=1 colspan=1>McC9gB5Cr60</td></tr><tr><td rowspan=1 colspan=1>KriyaLv</td><td rowspan=1 colspan=1>10LM0_Jzt5M</td></tr><tr><td rowspan=1 colspan=1>water park</td><td rowspan=1 colspan=1>Gezen Adam</td><td rowspan=1 colspan=1>3iz5SmEQj9AWgbe-WTp_QI</td></tr></table>

Table S2. Comparisons of existing datasets with our video trimming dataset.   

<table><tr><td>Dataset</td><td>#Video</td><td>#User</td><td>Content</td><td>Annotation type</td><td>Query</td><td>Duration (Min, Max, Avg)</td></tr><tr><td>YouTube Highlights [43]</td><td>423</td><td>5</td><td>Web videos</td><td>Frame-level scores</td><td>Title</td><td>7s, 1483s, 102s</td></tr><tr><td>SumMe [16]</td><td>25</td><td>15~18</td><td>User-generated videos</td><td>Frame-level scores</td><td>N/A</td><td>32s, 324s, 146s</td></tr><tr><td>TVSum [41]</td><td>50</td><td>20</td><td>Web videos</td><td>Frame-level scores</td><td>Title</td><td>83s, 647s, 235s</td></tr><tr><td>Charades-STA [15]</td><td>9,848</td><td>267</td><td>Web videos</td><td>Time intervals</td><td>Local caption</td><td>2s, 194s, 30s</td></tr><tr><td>OVP [10]</td><td>50</td><td>5</td><td>Various genre videos</td><td>Time intervals</td><td>N/A</td><td>83s, 647s, 235s</td></tr><tr><td>YouTube [10]</td><td>39</td><td>5</td><td>Web videos</td><td>Time intervals</td><td>N/A</td><td>83s, 647s, 235s</td></tr><tr><td>Video Trimming (ours)</td><td>42</td><td>10</td><td>Web videos</td><td>Frame-level scores</td><td>N/A</td><td>141s, 1483s, 556s</td></tr></table>

# system

ul detailed answers.

# user:

y well as purposeful tracking shots, should not be considered as obstructions.

considered as jitter. Shots with clear actions or behaviors should not be considered shaky.   
colored horizontal stripes, colored vertical stripes, green fringing, pink screen, or purple screen.   
on the top, bottom, or sides of the frame, it indicates that the video has been edited.   
walls are ineffective and meaningless.

considered highlights.

# # Useful attribute:

longer descriptions. Above all, don't use general, non-discriminative descriptions.   
[What]: Describe the main actions or events occurring in the scene.   
phrases such as 'outdoor'   
[When]: Determine the time of day, season, or any relevant time period depicted.   
rather than ambiguous phrases such as 'person'.

Py summarize the content of the video clip. It should not exceed two sentences.

# Answer given questions with the following restrictions.   
(1) If you are not sure about the answer, say you do not know honestly.   
(2) Do not imagine any contents that are Not in the video.   
( Do not add information.   
(4) Do not describe each frame individually and do not mention the frame.   
(5) Do not summarize negative or uncertain answers. # Output format constraints   
The overall output format is as follows:   
{"atibuteuseless":Uselesattribute, "attributeuseful:Useul attribute, "racaption":Vido captin}   
- Useless attribute output format constraints:   
.    ] [Meaningless]: 0.8; [Highlight]: 0.9;   
exist, while a score of 1 indicates absolute reliability."   
- Useful attribute output format constraints:   
Each scene should contain the above four attributes.   
It is recommended to use one word or one short phrase to summarize each attribute.   
o ul  a ge, h y nd nse couple"}   
- Video caption output format constraints:   
summarize the content of the video clip. It should not exceed two sentences.

system # IDENTITY and PURPOSE Y <clip id>, <highlight score>, <clip caption>, <clip attribute>. Y segments. The <clip id> represents the temporal sequence of the original video. Think step-by-step about how to achieve the best possible results by following the steps below. # STEPS onsiderptianriuheideonszhmacoplthemh score> ranging from 0 to 1, with higher scores being prioritized. structure of beginning, development, and ending. # RULES - Include segments of the beginning and the end, focus on choosing continuous brilliant clips. Avoid duplicate clips or clips with similar sceneries. rather than fixed scenes. - The number of selected clips should be no less than half of the inputs clip length. more closely indexed clips should be considered for merging first. # OUTPUT INSTRUCTIONS Only output Markdown. Do not imagine any contents that are Not in the clip captions. Do not output the markdown code syntax, only the content. Do not use bold or italics formatting in the markdown output. Do not list clip id in HIGHLIGHTS. You use the following format in exciting video collection: [<clip id>: sentence], .…, [<clip id>: sentence], where each houlbewiten gsndae wit nc sntehunotehcio - Do not repeat ideas, quotes, facts, or resources. Do not start items with the same opening words. Ensure you follow ALL these instructions when creating your output.

ser: # INPUT INPUT:

# system

# IDENTITY and PURPOSE   
attributes and descriptions provided for each clip.   
series of segments that capture multiple complete action scenes.   
The final merged video needs to consider the input video sequence and satisfy logical rationality.   
Think step by step about how to achieve the best possible results by following the steps below.   
Select clip from the start and the end of the input <clip id> as beginning and ending.

# STEPS

clips according to each theme.   
2. The clips of each theme should contain the development of the event.   
the temporal sequence of the original video.

# RULE

Avoid duplicate clips or clips with similar sceneries. - The selected clips should all of the themes and ensure content diversity. -The chosen <clip id> should cover the clips from the start, middle, end of the inputs sequence. rather than fixed scenes - Ensure that the selected segments for the final story generation do not exceed 20 and no less than 15. narrative integrity. critical and should be selected. These segments should be consistently included in the final output.

# OUTPUT INSTRUCTIONS   
- Only output Markdown. Do not imagine any contents that are Not in the clip captions. Do not output the markdown code syntax, only the content.   
- Do not use bold or italics formatting in the markdown output.   
- You use the following format in output: [<clip id>: sentence], [<clip id>: sentence], where each sentence should be   
written in English and wrapped with [] and each sentence should note the clip id coming from. Do not repeat ideas, quotes, facts, or resources. Do not start items with the same opening words. Ensure you follow ALL these instructions when creating your output.

user: # INPUT INPUT:

# system

detailed answers.

# user:

highest score (best).

should contain specific event content. Scores range from 1 to 5, with 5 being the highest score (best).

h talking to the camera, static scenes, ec.Scores range from 1 to , with 5 being the highest score (best).

# Answer given questions with the following restrictions.   
(1) If you are not sure about the answer, say you do not know honestly.   
(2) Do not imagine any contents that are Not in the video.   
(3) Do not add information.   
(4) Do not describe each frame individually and do not mention the frame.   
(5) Do not summarize negative or uncertain answers.

# Output format constraints.

T ; [Content of Exciting Segments]: Reason (3.5); [Amount of Waste Footage]: Reason (2.0);

Table S3. Ablation study on the impact of sampling ratio and prompt design on performance and cost.   

<table><tr><td>Frame Sampling Ratio</td><td>Prompt</td><td>Input Image Token</td><td>Input Text Token</td><td>Output Text Token</td><td>API Cost</td><td>Agent Metric</td></tr><tr><td>4/1s</td><td>Isolated</td><td>1,836,000</td><td>100,000</td><td>20,000</td><td>$5.04</td><td>3.34</td></tr><tr><td>4 /1s</td><td>Unified</td><td>612,000</td><td>100,000</td><td>20,000</td><td>$1.98</td><td>3.33</td></tr><tr><td>1/1s</td><td>Isolated</td><td>459,000</td><td>100,000</td><td>20,000</td><td>$1.60</td><td>3.34</td></tr><tr><td>1 /1s</td><td>Unified</td><td>153,000</td><td>100,000</td><td>20,000</td><td>$0.83</td><td>3.32</td></tr></table>

Table S4. Comparison of the fidelity between the final videos and the raw videos.   

<table><tr><td>Method</td><td>ViCLIP</td><td>InternVideo2</td><td>Avg.</td></tr><tr><td>UniVTG [27]</td><td>0.877</td><td>0.941</td><td>0.909</td></tr><tr><td>UVCOM [57]</td><td>0.852</td><td>0.928</td><td>0.890</td></tr><tr><td>AVT (ours)</td><td>0.906</td><td>0.951</td><td>0.929</td></tr></table>

# C. Implementation and Efficiency

We analyze the efficiency and cost of different implementations by varying the video sampling ratio and prompt design. Specifically, we compare a sampling ratio of 1 frame per second (fps) with 4 fps. As shown in Fig. S1, three components: raw captions, defect attributes, and contextual attributes, are typically generated together using a unified prompt. Alternatively, these components can be extracted separately using isolated prompts, which require processing three times the visual content.

The current GPT API pricing is $\$ 2.50$ per million input tokens and $\$ 10.00$ per million output tokens. Each sampled keyframe resized to $5 1 2 \times 5 1 2$ generates approximately 255 tokens in GPT-4o. Metrics are evaluated using the Video Evaluation Agent. Tab. S3 highlights that adopting a 1 fps sampling ratio and a unified prompt reduces the cost of processing a 10-minute video from $\$ 5.04$ to $\$ 0.83$ while maintaining comparable performance to configurations with higher sampling rates and isolated prompts.

# D. Fidelity Evaluation

For the quantitative experiments on video trimming, we also introduce a fidelity evaluation to assess the visual content similarity between the generated videos from different methods. For previous methods, we directly concatenate intervals with the highest saliency scores. In this experiment, we measure the feature similarity between the final video and the raw videos. A well-trimmed video should preserve the full content of the original video while maintaining its narrative coherence.

We utilize two benchmarks, leveraging video features extracted by ViCLIP [51] and InternVideo2 [52]. For both raw and trimmed videos from each method, an equal number of keyframes are sampled and processed through vision encoders. The feature similarity between the raw and trimmed videos is subsequently evaluated. As shown in Tab. S4, our method consistently improves content fidelity across various feature extraction models.

# E. More Visualization

In the main paper, we present visualizations of saliency scores and selected intervals for each method, demonstrating the effectiveness of our waste filtering operation and composition phase. In this supplementary section, we expand the analysis by incorporating additional visualizations and conducting case studies to highlight the significance of the AVT module designs.

# E.1. Clip selection

We present a case study that visualizes the clip selection results when incorporating different AVT modules. Figure S5 illustrates the impact of AVT's clip filtering module by comparing performance with and without it. Without filtering, story composition is applied to all intervals, resulting in a full row of light green segments in the visualization. This lack of candidate narrowing leads to the inclusion of more wasted footage in the final video. Figure S6 highlights the consequences of omitting the dynamic filtering module. Without this module, the clip filter discards most segments, especially in sports content, where intense activity often introduces jitter or other visual defects. As a result, highlight segments are misclassified as defects and excluded from the composition. The second row in the visualization shows significantly fewer filtered clips (light green) compared to the first row, emphasizing the importance of the dynamic filtering module. The joint design of the AVT modules substantially enhances the viewing experience and enriches the content. By selecting more highlight footage and minimizing wasted footage, AVT not only outperforms prior approaches but also preserves a coherent storyline that aligns with the raw video material.

# E.2. Storyline

We create the final video by constructing a corresponding storyline that outlines the rationale behind selecting each clip as the beginning, development, or ending, referred to as clip-wise captions. Additionally, we generate clustered themes, each representing a group of selected segments, as outlined in the story composition prompts. Ultimately, this results in a global storyline that captures the entire content of the trimmed videos. These captions, presented at various levels, are visualized in Fig. S7.

# E.3. More Visualization with Existing Methods

We present additional visualizations of the saliency scores and selected intervals for each method in our video trimming dataset, as shown in Fig. S8. Overall, AVT outperforms previous approaches by selecting more highlight footage, reducing wasted footage, and maintaining a consistent storyline, ultimately enhancing the viewing experience.

For instance, AVT excels at retrieving dynamic scenes like mountain biking, while existing methods tend to select more mundane clips. In another scenario, for plain vlogs such as food videos or dolphin shows, AVT efficiently trims the complete story across the entire timeline of the source video, while other methods may overlook key content.

![](images/8.jpg)  
Figure S5. Effect of clip filtering on visualization of trimmed videos.

![](images/9.jpg)  
Figure S6. Effect of dynamic filter module on visualization of trimmed videos.

![](images/10.jpg)  
Figure S7. Visualization of the multi-level storyline of the trimmed final video.

T c

![](images/11.jpg)

![](images/12.jpg)  
Figure S8. Visualization of trimmed videos on the video trimming dataset.