# 自链式图像-语言模型用于视频定位与问答

Shoubin Yu Jaemin Cho Prateek Yadav Mohit Bansal 北卡罗来纳大学教堂山分校 {shoubin, jmincho, praty, mbansal}@cs.unc.edu

# 摘要

最新研究表明，利用大型预训练的图像语言模型进行视频问答取得了有希望的结果。虽然这些图像语言模型可以有效地引导视频语言模型的表示学习，但它们通常仅通过均匀采样的视频帧作为视觉输入，而没有显式的语言感知和时间建模。当视频输入的某一部分与语言查询相关时，这种均匀帧采样往往会导致重要视觉线索的缺失。尽管人类常常会专注于视频的某一时刻并回放该时刻以回答问题，但训练一个查询感知的视频时刻定位器通常需要昂贵的标注和高计算成本。为了解决这个问题，我们提出了自链式视频定位与回答框架（SEVILA），它利用单个图像语言模型（BLIP2）同时应对视频的时间关键帧定位和问答任务。SEVILA框架由两个模块组成：LocALIzER和AnswERER，两个模块都是高效微调自BLIP-2。我们提出了两种方式将这些模块串联用于级联推理和自我精炼。首先，在前向链中，LocALizER在视频中找到多个语言感知的关键帧，AnswERER则利用这些关键帧来预测答案。其次，在反向链中，AnswERER生成关键帧伪标签以精炼LocALIzER，减轻对昂贵视频时刻定位标注的需求。我们的SeViLA框架在五个具有挑战性的视频问答和事件预测基准上超越了众多强基线/先前工作，并在微调（NExT-QA和STAR）及零-shot（NExT-QA、STAR、How2QA和VLEP）设置中达到了最先进的水平。我们展示了对框架的全面分析，包括LocALIzER的影响、LocALIZER与其他时间定位模型的比较、LocALIzER的预训练/自我精炼以及关键帧数量的变化。

# 1 引言

大型预训练语言模型的最近成功导致了多模态视觉与语言模型的迅猛发展，这些模型能够共同理解视觉（图像/视频）和语言数据。然而，由于更高的计算和标注成本，视频语言模型（video-LMs）在模型和数据规模方面比图像语言模型（image-LMs）更具挑战性。因此，最近的研究通过利用预训练的图像语言模型探索了视频语言模型的高效训练。尽管这种热启动策略促进了视频语言模型的视觉表征学习，但它们通常以均匀/随机采样的视频帧作为视觉输入，而没有显式的语言感知和时间建模。然而，这种简单的均匀/随机帧采样可能导致重要视觉线索的丧失，从而使视频语言模型集中在与语言查询无关或不重要的帧上。

![](images/1.jpg)  

Figure 1: Self-Chained Video Localization-Answering $( \mathbf { S E V I L A } )$ consists of a LocALIzER and an AnswEreR. Left: Forward chain for language-aware temporal keyframe localization and question answering. Right: Reverse chain for LocALizER self-refinement with keyframe pseudo-labels.

为了解决这一问题，我们引入了自链视频定位问答（SEViLA），这是一个新的视频语言框架，我们采用一个单一的图像语言模型来处理视频中的时间定位和问答，同时避免昂贵的语言感知时间标注（以及两个模块之间的自我优化）。我们的SEViLA框架通过对最新的最先进图像语言模型BLIP-2进行参数高效的微调，获得了两个模块，定位器（LocALIzer）和回答器（AnswERer）。SEViLA通过将定位器的输出链入回答器的输入（正向链，图1左）来处理视频语言任务，同时回答器给出反馈以优化定位器（反向链，图1右）。在正向链中，定位器利用BLIP-2主干的原始图像语言理解，通过定位提示“该帧中的信息是否提供了解答所需的细节？”选择重要的语言感知视频关键帧。然后，回答器将选择的关键帧的拼接作为视觉输入，预测视频级别的答案。在反向链中，我们生成关键帧伪标签来优化定位器，当回答器能够使用该帧输出正确答案时，我们将视频帧标记为关键帧。这一自我优化提高了语言感知时间定位的准确性，并减轻了对昂贵关键帧标注的需求。我们在五个具有挑战性的视频问答和事件预测基准（NExT-QA，STAR，How2QA，TVQA和VLEP）上展示了SEViLA框架的有效性，其中$\mathrm { S E V I L A }$超越了多个强基线/前期工作，在微调设置（NExT-QA和STAR）和零样本设置（NExT-QA、STAR、How2QA和VLEP）上均达到了最先进水平。我们还表明，定位器可以作为一个强大的独立时刻检索模型。我们提供了全面的分析，以阐述所提框架的设计选择，包括时间定位的影响、自我优化过程（反向链）的影响以及关键帧数量的变化。我们的贡献总结如下： •一个新的视频语言框架SEViLA，其中定位器和回答器从单一图像语言模型初始化，分别处理视频中的时间定位和问答。 •一种新的自我优化方法，用于语言感知的时间关键帧定位，其中回答器生成关键帧伪标签以优化定位器，而无需昂贵的时间标注。 •在多个视频语言基准上具有强大的经验性能，达到最先进水平。 •全面分析阐述所提框架的设计选择。

# 2 相关工作

图像-语言预训练模型。随着跨模态应用需求的不断增长，图像-语言预训练研究受到了极大的关注和成功。图像-语言预训练模型在模型和预训练数据规模方面，发展速度超过了视频-语言预训练模型（更详细的模型大小和预训练数据规模比较见附录）。这归因于图像数据获取的便利性以及相对简单的数据结构，使得图像-语言学习的扩展更为容易。在我们的论文中，SEViLA是基于最新的最先进图像语言模型BLIP-2构建的，并扩展以支持视频输入用于视频-语言任务。我们还将我们的SEViLA框架与当前最先进的视频语言模型InternVideo进行比较，以展示纳入关键帧定位的较大图像-语言模型的优越性。

图像到视频的迁移学习。图像和视频语言模型之间的差距激发了许多有用的方法，集中于图像到视频的迁移学习，这些方法利用有限数量的视频帧来增强学习效率。Luo等人将预训练的CLIP主干网络调整应用于视频片段检索。Yang等人扩展了冻结的双向语言模型，通过整合多幅图像并应用额外的视频级预训练来促进模型适应。Wang等人将多幅图像转换为层次化字幕，并通过时间顺序提示进行排列，以帮助语言模型理解视频级事件。然而，这些工作采用了一种非语言感知的均匀采样策略。这可能导致时间建模中的关键视觉线索丢失，甚至使模型承担无关信息。在本文中，我们提出了一种LocALizeR，以为视频语言任务提供语言感知的视觉信息。

语言感知关键帧定位。许多方法 [42, 3, 18, 54, 24, 42, 41, 73, 9] 已被提出以解决语言感知关键帧定位的挑战。Buch 等人 [3] 通过使用答案标签优化了端到端的流程，以便为下游任务选择单个关键帧。Lu 等人 [42] 使用独立的图像和语言模型选择帧，并通过具有多个训练目标的问答模型回答问题。Qian 等人 [54] 设计了一个具有预定义范围的视频剪辑提议模型，并在与问答模型的迭代训练中优化它。Kim 等人 [24] 利用半参数检索器根据帧和语言特征的相似性获取关键帧。我们采用一个大型图像语言模型作为我们的定位器 (LocALIzER)，并将其与答案生成器 (AnswERER) 进行串联。我们的定位器可以帮助在正向链中微调答案生成器，并在反向链中通过伪标签进行优化。

# 3 方法：SEVILA

在本节中，我们介绍了自链式视频定位问答（SEViLA）框架的方法细节。首先，我们提供了BLIP-2的基础知识，这为我们的框架奠定了基础。接着，我们详细阐述了BLIP-2 LocALIzER和BLIP-2 AnswERER的设计，以实现视频的时间定位和问答。最后，我们展示了SeViLA框架在正向和反向链中的训练与推理过程。

# 3.1 预备知识：BLIP-2

我们采用 BLIP-2 作为我们 SEVILA 框架的主干。BLIP-2 是一个最新的最先进的预训练图像-语言模型（图像-LM），包括：（1）一个冻结的图像编码器；（2）一个冻结的大型语言模型（LLM）；（3）一个 Q-Former，它是一个可训练的变换器模块，连接图像编码器和 LLM，类似于充当适配器。它的输入为来自图像编码器的视觉特征 $h$ 和可学习的查询嵌入 $q$ ，输出固定长度的视觉特征 $v$ 。BLIP-2 的 Q-Former 经过两个阶段的预训练。首先，它连接到图像编码器进行图像-文本的预训练。这个阶段使 Q-Former 能够提取文本所需的最有信息量的视觉信息，并去除 $v$ 中的任何无关细节。随后，Q-Former 连接到 LLM，以利用其生成语言的能力。这是通过使用全连接层将查询嵌入投影到 LLM 的维度上，并结合图像-文本的预训练来实现的。因此，这些查询特征作为 LLM 的软视觉提示。凭借经过两阶段预训练的 Q-Former 和 LLM，BLIP-2 在各种图像-语言任务上展现出先进的性能。在我们的 SEVILA 框架中，我们将 BLIP-2 作为视频时间定位和问答模块的基本构建块。在训练过程中，我们通过保持图像编码器和 LLM 的冻结状态来保留它们。在这种情况下，仅在 LocALIzER 和 AnswERER 训练期间更新这两个 Q-Former。

![](images/2.jpg)  

Figure 2: In SEVILA framework, LocALIzer (top) selects top-K video frames, which guides AnswErer (bottom) to focus on important language-aware video moments and predict answers. Both LocALIzER and AnswERER are initialized from a single pre-trained BLIP-2 model, where only Q-formers and a linear layer $2 . 5 \%$ of total parameters) are tuned for each module. We omit the linear layer after the Q-former for simplicity.

# 3.2 自链视频定位与回答

将 BLIP-2 适配于视频的时间定位和问答。如图 2 所示，我们的 SEViLA 框架采用 BLIP-2 来解决视频的时间定位和问答问题。我们通过使用不同的 Q-formers 将 BLIP-2 分配为定位器（LocALizer）和回答者（AnswERer）两个角色。我们首先详细阐述我们的定位器和回答者如下：

LocALIzER。我们首先通过冻结的图像编码器ViT [16] 提取帧特征，记为 $E _ { v }$。给定视频，我们均匀采样 $n$ 帧 $\{ f _ { 1 } , . . . , \bar { f _ { n } } \}$。然后，我们得到第 $i _ { t h }$ 帧特征 $h _ { i }$，其表达为 $h _ { i } = E _ { v } ( f _ { i } )$。最后，我们将视频表示为帧特征集合 $V = \{ h _ { 1 } , . . . , h _ { n } \}$。这些特征提取一次后保存，供 LocALIzER 和 AnswERER 后续重用。LocALIzER 的主要目标是从 $V$ 中选择 $k$ 个具有语言感知的关键帧特征，其中 $k$ 通常远小于 $n$。如图 2（顶部）所示，我们随后独立地通过 Q-Former $Q _ { l o c }$ 从 $V$ 中的原始帧特征提取视觉查询特征 $v _ { i }$。接下来，视觉查询特征 $v _ { i }$ 与语言上下文 $L$ 连接并输入到 LLM（Flan-T5 [7]）中，我们通过结合问题、选项和定位提示“帧中的信息是否提供了准确回答给定问题所需的细节？”来创建 $L$。LocALizer 输出每帧的得分 $s _ { i }$，即在给定视觉特征 $v _ { i }$ 和语言上下文 $L$ 的情况下生成词“yes”的概率：$s _ { i } = L L M ( c o n c a t ( v _ { i } , L ) )$。我们可以基于最高帧得分定位具有语言感知的关键帧 $K = \{ v _ { 1 } ^ { k } , . . . , v _ { K } ^ { k } \}$。我们的 LocALize 可以表述为：

$$
K = \operatorname { L o c a L I Z E R } ( V , L ) , \quad | K | = k \ll n
$$

AnswereR。通过LocALIzER获得的关键帧集 $K$，如图2（下部）所示，可以生成使用As的答案。我们首先通过 $Q _ { a n s }$ 处理关键帧查询 $v _ { i }$，遵循与LocALIzER相同的程序。接下来，我们通过连接所有查询特征和语言上下文，向大语言模型（LLM）输入这些数据，获得视频级答案 $\boldsymbol { a } = \dot { L } L \dot { \boldsymbol { M } } ( \text{concat} ( v _ { 1 } ^ { k } , . . . , \mathbf { \bar { \boldsymbol { v } } } _ { K } ^ { k } , \mathbf { \bar { \boldsymbol { L } } } ) ) ^ { 2 }$，并以多个帧输入进行建模。我们的AnswERER可以表述为：

$$
a = \operatorname { A N S W E R E R } ( K , L )
$$

![](images/3.jpg)  

Figure 3: Top: In the forward chain, the LocALizer finds multiple language-aware keyframes, then the Answerer utilizes these keyframes to predict answers. We use the forward chain for both inference and AnswErer fine-tuning. Bottom: In the reverse chain, we generate keyframe pseudo-labels by using the AnSWERer to refine the LocALIZER.

# 3.3 通过自链训练 AnswERER 和 LoCALIZER

在正向链中微调AnswErer。如图3（上）所示，我们使用LocALizEr生成的关键帧，通过正向链对AnswErer进行下游任务的微调。AnswErer接受由LocALizEr生成的关键帧。我们在附录中将默认设置与其他设置（例如，输入帧均匀选择）进行比较。 在反向链中精炼LocALizEr。我们在反向链中采用伪标注方法来解决代价高昂的帧级定位标注问题。我们使用二进制伪标注，当AnswErer能够使用某帧生成正确答案时，我们将该视频帧标记为关键帧。如图3（下）所示，冻结的AnswErer首先由问答任务提示进行提示，并生成帧级答案，然后通过将此预测与真实答案进行比较来获得伪标注。LocALizEr被训练以定位语言感知的伪标注关键帧。 使用时刻检索标签对LocALizEr进行预训练。为增强我们的LocALizEr，我们通过预训练从视频时刻检索/定位任务中进行迁移学习。我们使用来自QVHighlights的视频、查询和视频级时间跨度标签，并通过将其时间戳与跨度注释进行比较，为每帧分配二进制定位标签。有关预训练的更多细节，请参见附录。

# 4 实验

在本节中，我们首先概述我们的实验设置（第4.1节）。然后，我们在微调（第4.2节）和零样本（第4.3节）设置中展示SeVILA框架在5个具有挑战性的长视频问答和事件预测基准上的优越性。我们还对$\mathrm{S E V I L A}$框架进行消融研究，以展示其各个组件在下游任务中的有效性（第4.4节）。接下来，我们报告LocALizeR在视频时刻检索中的表现（第4.5节）。最后，我们对LocALizeR进行深入的定量和定性分析，以展示我们在时间关键帧定位设计上的效果（第4.6节和附录）。关于单帧与多帧LocALizER、预训练策略、迭代自我精炼、计算成本以及扩展到另一个Image-LM模型的更多结果在附录中。

# 4.1 实验设置

基准测试。我们在三个视频语言任务上评估我们的 SEViLA 框架，包括多选视频问答（NExT-QA [77]、STAR [75]、How2QA [36]、TVQA [27]）、视频事件预测（VLEP [28]）和时刻检索（QVHighlights [30]）。有关详细信息，请参见附录。

Table 1: Fine-tuning results on video question answering (NExT-QA, STAR, How2QA, TVQA) and video event prediction (VLEP). We gray out the methods take extra speech input or use dense frames. We bold the best numbers, and underlined the second-best numbers. dense/1fps: the model takes dense (1fps) video frames instead of a fixed number of frames. $3 2  4$ : our LocALizer selects 4 keyframes from 32 frames. \* represents the results tested by ourselves. $\mathbf { S } \mathbf { E } \mathbf { V } \mathbf { I } \mathbf { L } \mathbf { A } ^ { \dagger }$ uses the zero-shot LocALizer without refining on pseudo-labels via the reverse chain.   

<table><tr><td rowspan="2">Model (# Frames)</td><td colspan="4">NExT-QA</td><td colspan="4">STAR</td><td rowspan="2" colspan="4">How2QA TVQA VLEP</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>Tem. Cau. Des. Avg. Int. Seq. Pre. Fea. Avg.</td><td></td></tr><tr><td colspan="9">(w/ speech input or use dense frames)</td><td></td><td></td><td></td></tr><tr><td>HERO (dense/1fps) [36]</td><td>-</td><td>-</td><td>-</td><td></td><td></td><td></td><td></td><td>-</td><td>73.8</td><td>73.6</td><td>-</td></tr><tr><td>JustAsk (20) [84]</td><td>51.4</td><td>49.6</td><td>63.1</td><td>52.3</td><td></td><td></td><td>-</td><td></td><td>84.4</td><td>-</td><td>-</td></tr><tr><td>FrozenBiLM (10) [85]</td><td>-</td><td></td><td></td><td></td><td></td><td></td><td>-</td><td>-</td><td>86.7</td><td>82.0</td><td>-</td></tr><tr><td>VidIL 4-shot (12) [72]</td><td>-</td><td></td><td></td><td></td><td></td><td></td><td>-</td><td>-</td><td>-</td><td>-</td><td>72.0</td></tr><tr><td>T+T (dense/1fps) [40]</td><td></td><td></td><td></td><td></td><td></td><td></td><td>-</td><td>-</td><td>92.4</td><td>-</td><td>-</td></tr><tr><td>T+T (+ASR, dense/1fps) [40]</td><td></td><td></td><td></td><td>-</td><td></td><td></td><td>-</td><td>-</td><td>93.2</td><td></td><td>-</td></tr><tr><td>Flamingo-80B 32-shot (30) [1]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FrozenBiLM (10) [85]</td><td>-</td><td></td><td>-</td><td></td><td></td><td></td><td></td><td>42.2 -</td><td>- 81.5</td><td>57.5</td><td></td></tr><tr><td>All-in-One (32) [67]</td><td>48.6</td><td></td><td></td><td></td><td></td><td></td><td></td><td>48.0 63.2 50.6 47.5 50.8 47.7 44.0 47.5</td><td>-</td><td>-</td><td></td></tr><tr><td>Temp[ATP] (32) [3]</td><td>49.3</td><td>48.6</td><td></td><td></td><td></td><td></td><td></td><td>65.0 51.5 50.6 52.8 49.3 40.6 48.3</td><td>-</td><td>-</td><td></td></tr><tr><td>VGT (32) [78]</td><td>55.0</td><td>52.2</td><td>64.0</td><td>55.0</td><td></td><td></td><td></td><td>44.2</td><td></td><td></td><td></td></tr><tr><td>MIST (32) [18]</td><td>56.6</td><td>54.6</td><td>66.9</td><td></td><td></td><td></td><td></td><td>57.1 55.5 54.2 54.2 44.4 51.1</td><td>-</td><td></td><td></td></tr><tr><td>VFC (32) [50]</td><td>53.3</td><td>57.6</td><td>72.8</td><td>58.6</td><td>-</td><td>-</td><td></td><td>-</td><td>-</td><td></td><td></td></tr><tr><td>CoVGT (32) [79]</td><td>57.4</td><td>58.8</td><td>69.3</td><td>60.0</td><td></td><td>-</td><td></td><td>45.9</td><td>-</td><td></td><td></td></tr><tr><td>SeViTFiD (10) [24]</td><td>-</td><td></td><td></td><td>60.6</td><td></td><td></td><td></td><td>-</td><td></td><td>-</td><td></td></tr><tr><td>HiTeA (16) [87]</td><td>58.3</td><td>62.4</td><td>75.6</td><td>63.1</td><td>-</td><td>-</td><td></td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>InternVideo* (8) [71]</td><td>58.5</td><td>62.5</td><td>75.8</td><td>63.2</td><td></td><td></td><td></td><td>62.7 65.6 54.9 51.9 58.7</td><td>79.0</td><td>57.2</td><td>63.9</td></tr><tr><td>BLIP-2voting (4)</td><td>65.2</td><td>70.1</td><td>80.1</td><td>70.1</td><td></td><td></td><td></td><td>52.3 54.8 49.0 51.2 51.8</td><td>79.6</td><td>54.5</td><td>67.0</td></tr><tr><td>BLIP-2concat (ANSWERER) (4)</td><td>68.1</td><td>72.9</td><td>81.2</td><td>72.6</td><td></td><td></td><td></td><td>65.4 69.0 59.7 54.2 62.0</td><td>82.2</td><td>59.8</td><td>68.6</td></tr><tr><td>SEVILA† (32 → 4)</td><td>68.8</td><td>73.4</td><td>83.5</td><td>73.4</td><td></td><td>63.2 66.6 61.3 60.0 62.7</td><td></td><td></td><td>83.7</td><td>59.7</td><td>69.0</td></tr><tr><td>SeViLA (32 → 4)</td><td>69.4</td><td>74.2</td><td>81.3</td><td>73.8</td><td></td><td>63.7 70.4 63.1 62.4 64.9</td><td></td><td></td><td>83.6</td><td>61.6</td><td>68.9</td></tr></table>

基线。我们将我们的 $\mathrm { S E V I L A }$ 框架与最先进的视频-语言预训练模型 InternVideo [71] 以及我们的主干网络 BLIP-2 [35] 进行比较。我们扩展 BLIP-2 以适应两种视频设置：（1）BLIP $2 ^ { \mathrm { v o t i n g } }$，该模型独立处理每个均匀采样的帧，并通过对所有帧级答案进行多数投票获得最终答案；（2）BLIP $2 ^ { \mathrm { c o n c a t } }$，其中 Q-former 处理每帧，Flan-T5 将视觉特征的拼接作为前缀。BLIP $2 ^ { \mathrm { c o n c a t } }$ 的实现细节。SEVILA 框架采用 BLIP-2 [35]，这是一个拥有 41 亿参数的图像-语言模型，总共在 1.29 亿张图像上进行预训练，包括 COCO [39]、Visual Genome [25]、CC12M [59]、SBU [52] 和来自 LAION400M [57] 的 1.15 亿张图像。详细信息见附录。

# 4.2 在视频问答和事件预测任务中，与最先进方法的微调比较

我们将我们的SEViLA框架与最近的最先进模型在4个视频问答基准和1个视频事件预测数据集上进行了比较。结果展示在表1中，我们的发现总结如下：（a）时间建模非常重要。BLIP. $2 ^ { \mathrm { v o t i n g } }$在STAR、How2QA、TVQA和VLEP上表现不如我们的BLIP. $2 ^ { \mathrm { c o n c a t } }$（ANSwEReR）和其他视频语言模型。特别是在需要强大时间理解的任务STAR-Sequence上，我们的BLIP. $2 ^ { \mathrm { c o n c a t } }$（AnswERER）显著超过BLIP-2voting，提升幅度为$1 3 . 1 \%$（$6 9 . 0 \%$ 对比 $5 4 . 8 \%$）。由于BLIP $2 ^ { \mathrm { v o t i n g } }$独立处理帧并缺乏帧间的时间建模，这一结果表明，时间建模对于处理视频语言任务至关重要，且我们的时间建模设计是有效的。

（b）关键帧选择具有帮助作用。我们的 $\mathbf { S } \mathbf { E } \mathbf { V } \mathbf { I } \mathbf { L } \mathbf { A } ^ { \dagger }$ 框架，采用零-shot LocALIzER，在所有任务中领先，平均优势为 $5 . 3 \%$，超过了顶级视频语言模型（InternVideo）。它还超越了在 NeXT-QA 上使用均匀帧采样的 BLP $2 ^ { \mathrm { c o n c a t } }$（AnswereR），在 NeXT-QA ($ + 1 . 2 \% )$，STAR ($ + 0 . 7 \% )$，How2QA ($ + 1 . 5 \% )$ 和 VLEP ($ + 0 . 4 \% )$ 上均有提高。这突显了在视频语言任务中关键帧选择的重要性，即使在使用零-shot LocALIzER 的情况下。

Table 2: Zero-shot results on video question answering and video event prediction.   

<table><tr><td rowspan="2">Model (# Frames)</td><td colspan="4">NExT-QA</td><td colspan="4">STAR</td><td rowspan="2" colspan="4">How2QA TVQA VLEP</td></tr><tr><td></td><td>Tem. Cau. Des. Avg. Int. Seq. Pre. Fea. Avg.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>(w/ speech input or use dense frames)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>JustAsk (20) [84]</td><td>-</td><td>-</td><td></td><td></td><td></td><td>-</td><td></td><td></td><td>-</td><td>51.1</td><td>-</td><td>-</td></tr><tr><td>FrozenBiLM (10) [85]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>-</td><td>58.4</td><td>59.2</td><td>-</td></tr><tr><td>ViperGPT (dense/1fps) [63]</td><td></td><td>-</td><td>-</td><td>60.0</td><td>-</td><td></td><td></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Flamingo-80B (30) [1]</td><td></td><td></td><td>-</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FrozenBiLM (10) [85]</td><td>-</td><td>- -</td><td>-</td><td>- -</td><td></td><td></td><td></td><td>-</td><td>39.7</td><td>- 41.9</td><td>- 29.7</td><td>-</td></tr><tr><td>VFC (32) [50]</td><td>- 45.4</td><td></td><td>51.6 64.1 51.5</td><td></td><td></td><td>-</td><td>-</td><td></td><td>- -</td><td>-</td><td>-</td><td>- -</td></tr><tr><td>InternVideo* (8) [71]</td><td>43.4</td><td>48.0</td><td>65.1</td><td>49.1</td><td></td><td></td><td>43.8 43.2 42.3 37.4 41.6</td><td></td><td></td><td>62.2</td><td>35.9</td><td>58.7</td></tr><tr><td>BLIP-2voting (4)</td><td>59.1</td><td>61.3</td><td>74.9</td><td>62.7</td><td>41.8</td><td></td><td>39.7 40.2 39.5 40.3</td><td></td><td></td><td>69.8</td><td>35.7</td><td>63.8</td></tr><tr><td>BLIP-2concat (AnswereR) (4)</td><td>59.7</td><td>60.8</td><td>73.8</td><td>62.4</td><td></td><td></td><td>45.5 41.8 41.8 40.0 42.2</td><td></td><td></td><td>70.8</td><td>36.6</td><td>64.0</td></tr><tr><td>SEVILA† (32 → 4)</td><td>61.3</td><td></td><td>61.5 75.6 63.6 48.3 45.0 44.4 40.8 44.6</td><td></td><td></td><td></td><td></td><td></td><td></td><td>72.3</td><td>38.2</td><td>64.4</td></tr></table>

(c) 自我精炼提高了时间定位。对于 $\mathrm{S E V I L A}$，我们使用伪标签对 LocALizeR 进行了精炼（见第 3.3 节）。与 $\mathbf{S}\mathbf{E}\mathbf{V}\mathbf{I}\mathbf{L}\mathbf{A}^{\dagger}$ 相比，SEVILA 进一步提高了 NExT-QA （0.4%）、STAR （+2.2%）和 TVQA （+1.9%）的性能。SEVILA 框架在仅使用视觉和语言模态的情况下，在 NExT-QA、STAR、TVQA 和 VLEP 上达到了新一流的微调性能。这说明了时间定位的重要性以及我们自我精炼方法在关键帧定位中的有效性。

# 4.3 在视频问答和事件预测上的零-shot与最先进技术的比较

我们进一步将我们的SEViLA框架与最近的最先进模型在零样本设置下进行比较。 我们在表2中展示了零样本结果，然后在接下来讨论这些发现。

(a) Image-LM 在没有视频预训练的情况下超过了 Video-LM。令人惊讶的是，BLIP $2^{\mathrm{voting}}$ 在没有帧间时间建模的情况下，在多个需要时间推理的数据集上超过了之前最先进的视频语言模型 InternVideo。BLIP $2^{\mathrm{voting}}$ 在 NExT-QA（增加 13.6%）、How2QA（增加 7.6%）和 VLEP（增加 5.1%）等任务上均超过了 InternVideo。在 How2QA 上，BLIP $2^{\mathrm{voting}}$ 甚至超过了进行了额外语音和视频预训练的 FrozenBiLM，增幅达到 11.4%。这突显了 Image-LM 在视频语言任务中的潜力，得益于其模型规模和充分的预训练。

关键帧选择比均匀采样更有效。我们的 $\mathbf { S } \mathbf { E } \mathbf { V } \mathbf { I } \mathbf { L } \mathbf { A } \dagger$ 框架结合了零-shot LocALIzeR 和零-shot AnswEReR，在 NExT-QA $( + 1 . 2 \% )$、STAR $( + 2 . 4 \% )$、How2QA $( + 1 . 5 \% )$、TVQA $( + 1 . 6 \% )$ 和 VLEP $( + 0 . 4 \% )$ 的表现优于使用均匀采样帧的 BLIP-2concat (AnswERER)，在 NExT-QA、STAR、How2QA 和 VLEP 上实现了新的最先进的零-shot 性能，而在仅使用视觉和语言模态的情况下，在 TVQA 上也创下了新的最先进水平。在 STAR 上，我们的 $\mathrm { S E V I L A ^ { \dagger } }$ 框架甚至比参数为 80B 的零-shot Flamingo [1] 提高了 $4 . 9 \%$。这一结果证明了我们的 SEViLA 框架在适应视频-语言任务方面的有效性，以及语言感知关键帧选择的重要性。

# 4.4 SEVILA框架的消融研究

我们对我们的 $\mathrm { S E V I L A }$ 框架进行消融研究，评估 LocALIzER 和 AnswERER 的有效性。结果见表 3。我们总结的发现如下：(a) 稀疏帧的表现优于密集帧：我们观察到，当 AnswERER 采用更多帧时性能下降（A 对比 B），这证实了稀疏帧表现更佳的原因在于原始的 Image-LM 模型有限的时间建模能力，而过于密集的帧可能会分散模型的注意力。

关键帧优于均匀采样帧：我们比较了 AnswERER 和 LocALIZER（SEViLA 框架）以及使用均匀采样帧的 AnswERER。在零-shot AnswERER 设置中，我们观察到当使用零-shot LocALIZER $( \mathbf { B } \ \nu . s .$ C) 时，性能显著提升，在 NExT-QA 上提升 $ (+ 1 . 0 \% )$，在 STAR 上提升 $ (+ 2 . 4 \% )$，在 How2QA 上提升 $ (+ 1 . 5 \% )$，在 TVQA 上提升 $ (+ 1 . 6 \% )$，在 VLEP 上提升 $ (+ 0 . 4 \% )$。并且对 LocALIZER 的伪标签精化进一步使所有任务的性能平均提高了 $2 . 1 \%$（B 与 D 对比）。在微调的 AnswERER 设置中，LocALIZER 的优势依然显著。我们的 SEVILA 框架利用了关键帧。

Table 3: Ablation studies on SEViLA framework. 'uniform' refers to the uniform sampling of video frames. LocALIzER† refers to the zero-shot LocALIzER without refining on pseudo-labels.   

<table><tr><td rowspan="2"></td><td colspan="2">AnSwERER</td><td rowspan="2">Keyframe</td><td>NExT-QA</td><td>STAR</td><td rowspan="2">How2QA TVQA VLEP</td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td colspan="2"># frame finetuned?</td><td></td><td>Tem. Cau. Des. Avg. Int. Seq. Pre. Fea. Avg.</td></tr><tr><td></td><td>32</td><td>X</td><td>uniform</td><td></td><td>54.7 56.7 67.8 57.7 46.2 43.6 40.7 41.042.8</td><td>67.0</td><td>33.2</td><td>54.0</td></tr><tr><td>B.</td><td>4</td><td>×</td><td>uniform</td><td></td><td>59.7 60.8 73.8 62.4 45.5 41.8 41.8 40.0 42.2</td><td>70.8</td><td>36.6</td><td>64.0</td></tr><tr><td>C.</td><td>4</td><td>×</td><td>LocaLizeR†</td><td></td><td>61.3 61.5 75.6 63.6 48.3 45.0 44.4 40.8 44.6</td><td>72.3</td><td>38.2</td><td>64.4</td></tr><tr><td>D.</td><td>4</td><td>X</td><td>LocaLizer</td><td></td><td>62.3 63.1 74.9 64.6 49.0 46.4 45.2 41.6 45.5</td><td>72.9</td><td>39.1</td><td>64.6</td></tr><tr><td>E.</td><td>4</td><td>√</td><td>uniform</td><td></td><td>68.1 72.9 81.2 72.6 65.4 69.0 59.7 54.2 62.0</td><td>82.2</td><td>59.8</td><td>68.6</td></tr><tr><td>F.</td><td>4</td><td>L</td><td>LOCAlizeR†</td><td></td><td>68.8 73.4 83.5 73.4 63.2 66.6 61.3 60.062.7</td><td>83.7</td><td>59.7</td><td>69.0</td></tr><tr><td>G.</td><td>4</td><td>4</td><td>LOcaLIzER</td><td></td><td>69.4 74.2 81.3 73.8 63.7 70.4 63.1 62.4 64.9</td><td>83.6</td><td>61.6</td><td>68.9</td></tr></table>

Table 4: Comparison on QVHighlights test split. We aggregate frame-level results of our LocALIZER for video-level evaluation (see Appendix).   

<table><tr><td>Model</td><td>R1@0.5</td><td>R1@0.7</td><td>mAP</td></tr><tr><td>CAL [13]</td><td>25.4</td><td>11.5</td><td>9.8</td></tr><tr><td>XML [29]</td><td>41.8</td><td>30.3</td><td>32.1</td></tr><tr><td>Moment-DETR [30]</td><td>52.8</td><td>33.0</td><td>30.7</td></tr><tr><td>QD-DETR [51]</td><td>62.4</td><td>44.9</td><td>39.8</td></tr><tr><td>LocALizeR (Ours)</td><td>54.5</td><td>36.5</td><td>32.3</td></tr></table>

Table 5: The impact of QVHighlights PreTraining (PT) and Self-Refinement (SR) for our LoCALIzER in Sec. 3.3.   

<table><tr><td rowspan="2">PT SR</td><td colspan="3">NExT-QA</td><td rowspan="2">How2QA</td></tr><tr><td>Tem. Cau.</td><td></td><td>Des. Avg.</td></tr><tr><td>-</td><td>-</td><td>60.4 61.0</td><td>74.6 62.9</td><td>70.7</td></tr><tr><td>✓</td><td>-</td><td>61.3 61.5</td><td>75.6 63.6</td><td>72.3</td></tr><tr><td>-</td><td>✓</td><td>62.1 62.6</td><td>75.1 64.3</td><td>72.8</td></tr><tr><td>✓</td><td>V</td><td>62.3 63.1</td><td>74.9 64.6</td><td>72.9</td></tr></table>

LocALIzER 在各任务中平均超越了 4 帧 AnswERER $0.7\%$。伪标签精炼在这种情况下仍然有效，为所有任务提供了平均 $1.5\%$ 的提升。这些结果表明，关键帧选择对零样本和微调设置下的视频语言任务的显著改进具有贡献。

# 4.5 关于视频片段检索的最先进技术的比较

在本节中，我们评估了我们的LocALizer在视频瞬时检索任务上的表现。我们在QVHighlights [30] 数据集上对LocALizer进行了预训练，如3.3节中讨论的，随后在同一数据集上评估其性能。为了在QVHighlights上进行测试，我们首先以每秒0.5帧的速度提取视频帧，遵循Moment-DETR [30] 的方法，并将这些帧传递给我们的LocALizer以获得二进制帧级预测，指示某个帧是否与查询句子匹配。接下来，我们将这些预测合并为视频级时间跨度预测。我们通过合并相邻的正预测，并确保间隔不超过某一阈值，将帧级预测聚合为视频级跨度。这些合并结果随后被整合为一个单一的视频级跨度。关于聚合过程的更多信息可以在附录中找到。有趣的是，如表4所示，我们的LocALizer在没有时间建模/训练的情况下，并且仅在帧级上操作，超越了许多之前采用复杂时间建模和视频数据训练的方法 [13, 30, 29]。这表明我们的LocALizer可以作为某些任务的独立模型进一步工作。这也暗示了具有时间设计的大型图像语言模型可能是视频瞬时检索的一个有前景的研究方向。这一点从我们的LocALizer相较于Moment-DETR的优越表现中得到了证实，尽管没有时间建模。

# 4.6 本地化器的详细分析

在本节中，我们首先分析了预训练和自我优化对我们的LocALIZER的影响。然后，我们在零样本和微调设置中将我们的LocALizer与其他关键帧选择方法进行比较。接下来，我们在LocALIZE中实验了不同的关键帧选择范围和数量，以评估时间定位对整体性能的影响。我们进一步展示了LocALIzeR在AnswEReR微调中的影响。我们还基于BLIP-2和oracle关键帧定位提供了上限分析。最后，我们提供了LocALIzER的可视化结果。附录中包含了其他实验。关于LocALizER的预训练和自我优化的消融实验，我们在零样本的4帧AnswERER上进行这些消融研究。如表5所示，未经过训练的BLIP-2 LocALIZER对AnswEReR的提升仅微乎其微。此外，QVHighlights的预训练和通过逆向链进行的自我优化均独立提供了显著的性能提升。当同时应用预训练和自我优化时，能取得最佳结果。这进一步证明了我们的方法在关键帧时间定位方面具有高效性。

与其他关键帧选择方法的比较。在表6中，我们将我们的LocALIzER与不同的关键帧定位方法进行比较，包括CLIP [55]、Moment-DETR [30]（这两者均为零-shot方法），以及ATP [3]、Differentiable Top-K [8]（后者经过答案标签的微调）。我们将这些关键帧定位方法与我们的零-shot AnswERER结合。上述方法从32幅均匀采样的帧中选择4个关键帧。我们发现来自CLIP和Moment-DETR的关键帧都无法帮助AnswERER。这可能是由于它们的CLIP预训练在图像和简短的陈述句上，未能生成对问题敏感的视觉特征，并使AnswERER受到无关特征的干扰。相反，我们的零-shot LocALIzER† 在NExT-QA上提高了平均$1.2\%$。此外，我们在伪标签上进行优化的LocALIzER的表现优于微调的ATP和Differentiable Top $\mathbf{\nabla} \cdot \mathbf{K}$，在所有问题类型中平均提高了$2.2\%$。总体而言，我们的LocALizer在这两种设置中显示出卓越的有效性。

Table 6: Comparison of our LocALIzeR with other keyframe localization methods.   

<table><tr><td>Method</td><td>NExT-QA</td></tr><tr><td>AnswERER</td><td>Tem. Cau. Des. Avg. 59.7 60.8 73.7 62.4</td></tr><tr><td>(zero-shot)</td><td>60.0 72.5 61.8</td></tr><tr><td>+ CLIP [55] + Moment-DETR [30] + Localizer</td><td>59.2 59.5 60.6 72.1 62.0 61.3 61.5 75.6 63.6</td></tr><tr><td>(fine-tuning)</td><td></td></tr><tr><td>+ ATP [3]</td><td>60.4 61.3 73.4 62.8</td></tr><tr><td>+ Differentiable Top-K [8] 59.5 59.7 72.7 61.6</td><td></td></tr><tr><td>+ LocaliZeR</td><td>62.3 63.1 74.9 64.6</td></tr></table>

关键帧选择范围和数量的影响。在表7中，我们评估了在零样本设置下使用各种关键帧选择范围和数量的时间关键帧定位。即使只选择一个关键帧，我们的LocALIZER在基于8个帧级答案的多数投票中，相比于BLIP $2 ^ { \mathrm { v o t i n g } }$ 显示了显著的提升：NExT-QACausal $( + 2 . 4 \% )$，NExT-QA-Description $( + 3 . 6 \% )$，以及How2QA $( + 2 . 6 \% )$。这突显了LocALIZE的有效性在于选择性关键帧的定位。我们还注意到，多个关键帧对NExT-QA-Temporal问题有益，但更密集的帧则导致性能下降。这支持了我们在第4.4节中的发现，即使用过于密集的帧可能会干扰图像语言模型。

Table 7: Ablation of different numbers of input frames and output keyframes.   

<table><tr><td rowspan="2">Settings</td><td colspan="3">NExT-QA</td><td rowspan="2">How2QA</td></tr><tr><td>Tem.</td><td>Cau.</td><td>Des. Avg.</td></tr><tr><td>BLIP-2voting (8)</td><td>59.9</td><td>60.2</td><td>72.4 62.0</td><td>69.8</td></tr><tr><td>8→1</td><td>59.8</td><td>61.1</td><td>76.0 62.9</td><td>72.4</td></tr><tr><td>16→1</td><td>59.2</td><td>62.6</td><td>74.9 63.4</td><td>73.2</td></tr><tr><td>16→4</td><td>60.7</td><td>61.5</td><td>75.8 63.4</td><td>72.4</td></tr><tr><td>32→4</td><td>61.3</td><td>61.5</td><td>75.6 63.6</td><td>72.3</td></tr><tr><td>32→8</td><td>59.4</td><td>60.9</td><td>74.7 62.5</td><td>71.3</td></tr><tr><td>64→8</td><td>58.9</td><td>60.9</td><td>74.0 62.2</td><td>71.8</td></tr></table>

不同帧采样对AnswERer微调的影响。在第3.3节中，我们讨论了SEVILA框架中的AnswERER如何可以在前向链中进一步微调，使用来自LocALizER的关键帧。表8比较了不同帧采样策略下的微调，并表明SEVILA框架在同时利用LocALizER进行AnswERER训练和评估时效果最佳。这可能是由于提供了更具信息量的关键帧，以及训练与评估之间的领域转变较为温和。对oracle关键帧的上限性能分析。我们进一步探讨了在假设“完美”定位器的前提下的上限性能，该定位器始终能够为Answerer提供正确的关键帧。为此，我们均匀地从每个视频中采样四帧，分别输入BLIP-2，并生成四个帧级答案。上限性能通过oracle准确率表示，考虑到如果四帧中的至少一帧给出了正确答案，则问题被视为正确回答。如表9所示，在两种设置中，BLIP-2的多数投票与oracle准确率之间存在显著差距。这些差距强调了在时间定位方面需要更多未来的工作，以有效利用image-LM处理视频语言任务。对LocALizER的定性分析。在图4中，我们展示了NExT-QA中的一个示例（更多内容见附录），展示了QA对、我们的LocALizER结果以及我们手动注释的与任务相关的视频片段的真实标注。我们的LocALizER与人工标注的匹配度更高。

<table><tr><td colspan="2">Frame Sampling</td><td colspan="2">NExT-QA</td></tr><tr><td>Training</td><td>Inference</td><td>Temp. Cau.</td><td>Des. Avg.</td></tr><tr><td>Random</td><td>Uniform</td><td>68.1 72.9</td><td>81.2 72.6</td></tr><tr><td>Random</td><td>LoCAlizeR†</td><td>67.6 73.4</td><td>84.0 73.1</td></tr><tr><td>LOCALiZeR</td><td>Uniform</td><td>68.2 72.7</td><td>80.0 72.3</td></tr><tr><td>LocalizeR†</td><td>LOCalizeR</td><td>68.8 73.4</td><td>83.5 73.4</td></tr></table>

<table><tr><td rowspan="2">Datasets</td><td colspan="2">BLIP-2voting (Oracle)</td></tr><tr><td>Zero-Shot</td><td>Fine-tuned</td></tr><tr><td>NExT-QA (Avg.)</td><td>62.7 (70.1)</td><td rowspan="3">70.1 (79.7) 51.8 (72.2)</td></tr><tr><td>STAR (Avg.)</td><td>40.3 (52.9)</td></tr><tr><td>How2QA</td><td>69.8 (77.8) 79.6 (86.4)</td></tr><tr><td>TVQA</td><td>35.7 (45.4)</td><td>54.5 (69.0)</td></tr><tr><td>VLEP</td><td>63.8 (70.5)</td><td>67.0 (79.1)</td></tr></table>

Table 8: Comparing different frame sampling during ANSwERER fine-tuning. The LocALIzER† is frozen during fine-tuning. We use 4 frames for AnswERER training, while the LoCALIZER $^ { \dagger }$ is the default $3 2 {  } 4$ .

Table 9: BLIP. $2 ^ { \mathrm { v o t i n g } }$ and oracle (in brackets) performance analysis across datasets. We use 4 frames for each video question. Oracle: at least 1 of 4 frames can give the right answer.

两个女士为什么把手放在眼睛上方凝视外面？

![](images/4.jpg)  

Figure 4: Visualization of our LocALIzER. We use zero-shot AnswERER with different frame sampling (uniform v.s. LocALizeR) to answer the question. Red options are answered wrongly with uniformly sampled frames. Green options are answered correctly with our LocALizeR. Best viewed in color.

均匀选择。这种准确的定位使得回答者能够正确回答问题，而均匀选择则导致错误的回答。这些结果表明，我们的LocALIZER能够有效地在视频中定位与任务相关的关键帧，从而有利于下游任务。

# 5 结论与未来工作

在本文中，我们提出了SEViLA，一个新颖的视频语言框架。SEViLA适配了图像语言模型，以获得两个模块：（1）LocALIzER，用于语言感知的时间定位；（2）AnswEREr，用于关键帧上的问答。SEViLA通过将LocALIZER的输出与AnswERER的输入连接起来（前链）来处理视频语言任务，同时AnswERER可以通过伪标注向LocALIzER提供反馈，以进行精细调整（后链）。所提出的时间定位允许对视频进行更为深入的理解，提高视频语言任务的准确性，而伪标注过程减轻了对昂贵的语言感知关键帧注释的需求。与最先进的基线模型相比，SEViLA在五个视频问答和事件预测基准上取得了具有竞争力或更好的性能。我们还提供了全面的分析，阐述了所提出的双阶段自链设计。我们的研究鼓励未来的工作来改善视频理解中的时间定位。局限性与更广泛的影响。关于局限性和更广泛影响的讨论见附录。

# 致谢

我们感谢评审人以及Archiki Prasad、Hao Tan、Yi-Lin Sung和Jie Lei对本文的宝贵反馈和意见。本研究得到了ARO奖项W911NF2110220、DARPA KAIROS资助FA8750-19-2-1004、DARPA MCS资助N66001-19-2-4031和NSF-AI Engage研究所DRL211263的支持。文章中所包含的观点、意见和/或研究结果仅代表作者个人，而不代表资助机构的观点。