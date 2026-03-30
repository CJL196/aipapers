# DeltaVLM：通过指令引导的差异感知进行交互式遥感影像变化分析

邓培†，周文谦†，IEEE 学生会员，吴涵林，IEEE 会员

摘要—准确解读多时相卫星影像中的土地覆被变化对于实际场景至关重要。然而，现有方法通常只提供一次性变化掩码或静态标题，限制了它们支持交互式、查询驱动分析的能力。在本工作中，我们提出了遥感影像变化分析（RSICA）作为一种新范式，结合了变化检测和视觉问答的优势，以实现对双时相遥感影像变化的多轮、指导性探究。为支持这一任务，我们构建了ChangeChat-105k，这是一个大型的遵循指令的数据集，通过混合的基于规则和GPT辅助的过程生成，涵盖六种交互类型：变化描述、分类、量化、定位、开放式问答和多轮对话。基于该数据集，我们提出了DeltaVLM，这是一种专为交互式RSICA定制的端到端架构。DeltaVLM具有三项创新：(1) 一种微调的双时相视觉编码器，用于捕捉时间差异；(2) 一种具有交叉语义关系测量（CSRM）机制的视觉差异感知模块，用于解读变化；(3) 一种指导指令的Q-former，有效提取与查询相关的视觉变化差异信息，并将其与文本指令对齐。我们在ChangeChat-105k上训练DeltaVLM，使用冻结的大型语言模型，仅调整视觉和对齐模块以优化效率。大量实验和消融研究表明，DeltaVLM在单轮标题和多轮交互变化分析中均实现了最先进的性能，超越了现有的多模态大型语言模型和遥感视觉语言模型。代码、数据集和预训练权重可在 https://github.com/hanlinwu/DeltaVLM 获得。关键词—遥感影像变化分析、视觉问答、视觉语言模型、遵循指令

# I. 引言

地球观测卫星持续获取大量数据，为通过遥感图像（RSI）监测我们动态的星球开辟了机遇。这些图像能够提取和解读时间变化，给灾害管理[1]、森林砍伐监测[2]和环境监测[3]等应用带来了显著价值。然而，分析RSI相较于自然图像处理面临着独特的挑战，特别是在解释多时相变化方面。大气变化、传感器差异和几何畸变[4]等因素使得准确检测和解读随时间变化的情况变得复杂。早期对RS影像时间变化的分析主要依赖于变化检测技术[5]，如像素级或基于对象的方法。尽管这些方法在定位变化方面有效，但通常无法提供对这些变化性质的深入洞察。

![](images/1.jpg)  
Fig. 1. The performance of DeltaVLM against state-of-the-art VLMs on five RS change analysis tasks. Each axis corresponds to a task-specific metric: captioning (BLEU-1), classification (precision), quantification (inverted Road's-MAE), localization (F1-score), and open-ended QA (BLEU-1).

将自然语言处理（NLP）技术融入遥感影像（RSI）的解释中，缩小了原始视觉数据与人类理解之间的差距。遥感影像描述生成（RSIC）旨在为单一观测生成描述性文字。为了实现更互动的探索，遥感视觉问答（RSVQA）被引入，允许用户使用自然语言问题查询遥感影像，并获得相应的文本响应。然而，RSIC和RSVQA仅限于单幅影像分析，无法捕捉时间变化。为了解决这一局限性，遥感影像变化描述生成（RS-ICC）将RSIC扩展到双时相分析，生成时空差异的文本描述。

最近，大型语言模型（LLMs）[9] 的出现及其向视觉-语言模型（VLMs）[10] 的演变，为遥感（RS）解释引入了交互能力，能够处理超出静态标题的后续查询。然而，由于数据分布差异显著，这些主要在自然场景上训练的模型在遥感任务上的表现有限，如图1所示。为了解决这一领域差距，最近的研究通过在特定领域数据集上进行指令微调，将VLMs适配到遥感任务中 [11]。这种方法有效地将 VLM 的推理能力转移到遥感解释中，在相对较少的训练样本下实现了在复杂开放任务上的强大表现。例如，RSGPT [12] 通过领域自适应微调扩展 LLMs 到遥感任务，如遥感影像分类（RSIC）和遥感视觉问答（RSVQA）。类似地，基于 LLaVA [15] 框架构建的GeoChat [13] 和 RS-LLaVA [14] 在单图像基于区域的问答（QA）和视觉定位方面表现出色。然而，这些模型局限于单图像分析，无法从双时相数据中生成特定指令的变化描述。此外，目前大多数 RS-VLMs 主要集中在微调 LLM 主干，而忽视了在多时相分析中 RS 特有的视觉挑战，包括大气变化、季节变化和传感器噪声，这些因素可能掩盖有意义的时序差异 [16]。这一限制在缺乏大规模的交互式双时相遥感分析的指令遵循数据集的情况下进一步加剧。为了解决这些挑战，我们提出了遥感图像变化分析（RSICA），这是一项新任务，将变化描述的语义基础与 VQA 的交互和推理能力相结合。RSICA 实现了多回合多任务对话，允许用户动态探索和解释双时相遥感影像中的变化。为支持该任务，我们精心构建了 ChangeChat- $1 0 5 \mathrm { k }$，这是一个包含 105,107 对指令-响应对的大规模指令遵循数据集，通过结合基于规则的方法和 ChatGPT 的上下文学习能力 [17] 生成。我们提出的数据集覆盖了交互式变化分析的多种指令类型，包括：1）变化描述，2）二元变化检测，3）类别特定变化量化，4）变化定位，5）开放式问答，以及 6）多回合对话。

基于 ChangeChat- $1 0 5 \mathrm { k }$ 数据集，我们提出了 DeltaVLM，这是一种创新的端到端架构，用于查询驱动的遥感图像变化分析（RSICA）。与现有的视觉语言模型（VLM）针对单一图像输入不同，DeltaVLM 将传统的三阶段变化描述流程扩展为一个专门的视觉语言框架，能够进行指令引导的多任务变化解读，处理双时相遥感影像。其核心组件包括：1）一个双时相视觉编码器（Bi-VE），处理双时相遥感影像以提取和比较特征，捕捉多个尺度的时间差异；2）一个指令引导的差异感知模块（IDPM），结合跨语义关系测量（CSRM）机制和 Q-former，感知细微的视觉变化，过滤上下文中的无关语义和噪声，并动态地将这些差异与用户特定的指令对齐；3）一个大语言模型（LLM），将对齐后的差异信息解码为上下文相关的语言响应。为了验证 DeltaVLM 的有效性，我们在提出的 RSICA 任务上进行综合实验和消融研究，使用 ChangeChat- $1 0 5 \mathrm { k }$ 数据集。结果表明，DeltaVLM 实现了最先进的性能，优于现有的通用大型 VLM 和遥感变化描述模型，在交互式变化分析场景中表现突出。本文的主要贡献可以总结为如下几点：我们引入了 RSICA，这是一项新颖的任务，将变化描述和视觉问答（VQA）整合到一个统一的、交互式的、用户驱动的框架中，以分析双时相遥感影像。2）我们提出了 ChangeChat-105k，这是一个大规模的遥感指令跟随数据集，涵盖多种变化相关任务，包括描述、分类、计数、定位、开放式问答和多轮对话。我们提出的 DeltaVLM 是一种针对 RSICA 定制的创新 VLM 架构，集成了双时相视觉编码器、具有 CSRM 的视觉差异感知模块和指令引导的 Q-former，用于动态、上下文相关的多任务和多轮交互。4）我们进行了综合评估，证明了 DeltaVLM 相比现有基线在 RSICA 任务上的优越性能，验证了其在应对复杂变化分析挑战中的有效性。

# II. 相关工作

我们在RSICA中的工作基于几个关键思想，包括变化检测、变化描述、视觉问答和视觉语言模型。在本节中，我们将简要回顾这些领域。

# A. 变化检测

遥感中的变化检测旨在识别和定位不同时间拍摄的卫星图像中同一区域的变化。我们将这些方法分为两大类：传统方法和基于深度学习（DL）的方法。1) 传统变化检测方法：早期的基于代数的变化检测方法，如图像差分、比率法和变化矢量分析[18]，虽然计算简单，但对噪声和辐射不一致性敏感。为了提高鲁棒性，开发了基于指数[19]、基于变换[20]和基于分类[21]的方法，各自解决特定的局限性，但仍然缺乏语义理解。基于对象的图像分析[22]引入了空间上下文，改善了高分辨率遥感图像中的变化检测准确性，但在很大程度上依赖于准确的图像分割。尽管这些传统方法贡献了有价值的概念，但它们通常依赖于手工特征和统计模型。2) 基于DL的变化检测方法：深度学习通过使模型能够直接从双时相遥感图像中学习空间和时间表示，改变了遥感中的变化检测。基于CNN的早期方法，如孪生网络[23]和U-Net结构[24]，通过成对比较或端到端分割提高了变化定位的准确性。近年来，基于Transformer的架构[25]，[26]通过利用自注意力机制处理多尺度特征和建模时间依赖性表现出更强的性能。与传统CNN相比，它们提供了更多的全球上下文。然而，这些模型通常需要大规模的标注数据集和高计算资源。为了解决对标记数据的需求，自监督学习逐渐受到关注，使用对比学习[27]和掩模图像建模[28]等策略。为了更好的泛化，还发展了少样本学习和零样本学习方法[29]。此外，多模态融合已应用于增强在云层覆盖或雨天等不利条件下的检测稳定性[30]。有关变化检测方法的更全面调查可参考文献[5]。

# B. 变更标题生成

虽然变化检测主要输出二值掩模，变化描述则致力于生成对比两时相遥感影像中观察到的变化的自然语言描述。1）基于编码器-解码器的方法：早期的变化描述方法主要依赖于编码器-解码器架构：卷积神经网络（CNN）被用作视觉编码器，从配对图像中提取特征，而递归神经网络（RNN）作为语言解码器，生成突出变化的描述[31]。为了提高描述质量，后续研究将注意力机制融入解码器，使得模型能够动态关注显著变化区域[32]。尽管这些方法为变化描述奠定了基础，但它们在复杂的遥感场景中往往未能有效建模全球语义关系。

2) 三阶段方法：基于早期的编码器-解码器框架，最近的深度学习变化描述方法发展为三阶段架构：视觉编码、双时间特征融合和语言解码。在视觉编码阶段，广泛使用如 ResNet 或视觉变换器（ViTs）等预训练主干网络从输入图像中提取高层特征。接下来，在时间特征融合阶段，提出了多种特定任务的策略，以有效整合这些特征并强调与变化相关的信息。在最终的语言解码阶段，基于变换器的解码器逐渐取代了 RNN，因为它们在全局上下文建模方面表现更佳，能够生成更准确和更具描述性的标题。在这个框架内，出现了几种代表性方法。刘等人提出了 RSICCFormer，使用 ResNet-101 作为视觉编码器，并采用双分支变换器解码器以增强变化表示。类似地，孙等人提出了稀疏聚焦变换器（SFT），同样利用 ResNet-101，但引入了一种稀疏注意机制，以有选择性地关注变化区域，同时降低计算成本。相比之下，基于 ViT 的方法如 PSNet 采用 ViT-B/32 作为编码器，并优化其变换器解码器以进行多尺度特征融合，从而实现更详细和准确的变化描述。 3) 多任务和语义对齐方法：尽管架构的改进推动了变化描述的发展，但它们通常未能捕捉变化描述与相关任务之间的内在语义关系。为了解决这个问题，最近的方法探索了任务级的创新，特别是在多任务学习和语义对齐方面。这些方法通常涉及将变化描述与辅助任务进行联合建模，例如变化检测、任务分解或语义引导，以增强生成描述的语义质量。例如，刘等人推出了 PromptCC，將变化描述任务分解为两个子任务：二元变化分类和细致变化感知。与直接从原始双时间 RSI 生成相比，这一两步过程提升了标题生成的性能。在此基础上，朱等人提出了 Semantic-CC，该方法结合来自辅助变化检测任务的像素级语义指导，以生成更准确和上下文对齐的标题。

4) 基于大语言模型的方法：尽管多任务和语义对齐方法已取得显著进展，但它们通常针对特定任务和数据集进行定制，缺乏对多样化遥感任务的灵活性。为了解决这些局限性，近期研究转向了大语言模型，利用其强大的生成与上下文推理能力来增强变化描述的鲁棒性。基于大语言模型的方法通常不从零开始训练模型，而是利用预训练的视觉编码器提取高级语义特征，然后与大语言模型集成以生成具有上下文感知的标题。通常采用两种主要策略来提升标题质量：（1）对具有特定领域注释的遥感数据集进行大语言模型的微调，以及（2）通过特定任务的提示或中间输出（例如，变化图）来指导标题生成过程，帮助模型集中关注相关变化。例如，Noman 等人开发了 CDChat ，该模型对 Vicuna-v1.5 在如 SYSU-CD 和 LEVIR-CD 等数据集上进行微调，通过监督学习对视觉和文本表示进行对齐。这使得标题生成的准确性和可解释性得到了提高。总体而言，基于大语言模型的方法代表了变化描述的新方向。它们在任务间的泛化能力和对多样化遥感场景的适应性为未来的研究和实际应用提供了巨大的潜力。

# C. RSVQA

尽管变化检测和变化描述在遥感图像理解方面取得了显著进展，但仍受到预定义信息提取的限制，通常仅限于单轮交互和静态输出。为了弥补这一差距并实现更灵活的用户驱动分析，研究人员借鉴了自然图像领域的 VQA [41]，开发了 RSVQA。RSVQA [7] 允许用户提出针对特定分析需求的自然语言问题，VQA 系统则基于遥感图像的视觉内容生成上下文相关的答案。Lobry 等人 [7] 建立了两个基准数据集：RSVQA-LR 和 RSVQA-HR，包含来自农业和城市场景的大规模图像-问题-答案三元组，涵盖低和高空间分辨率，为未来更加动态和以用户为中心的遥感应用研究奠定了基础。大多数 RSVQA 方法遵循编码-融合-解码架构。为了提高细粒度的多模态推理，提出了几种增强融合模块的方法。例如，空间层次推理网络 [42] 结合基于哈希的多尺度注意力机制和语义分割先验，以更好地捕捉空间层次结构，而语义对象感知方法 [43] 通过对象级关系建模强化空间推理。相较之下，Chappuis 等人 [44] 提出了 Prompt-RSVQA，作为双模态融合的替代方案，通过将视觉信息转换为文本提示，且融合在基于变换器的语言模型中隐式发生。RSVQA 方法已从早期的全局特征融合演变为高级的基于注意力的架构，结合了面向对象的推理和基于变换器的交互。尽管在静态图像分析方面取得了这些进展，但现有的 RSVQA 方法常常忽视了遥感数据地理空间变化分析中的时间动态 [45]。变化感知 VQA [46] 是解决这一问题的早期尝试，提出了一种基于多时相航空图像的变化感知视觉问答方法。然而，它采用了分类器来选择答案，而不是生成自然语言，这在响应灵活性上显著低于 VLMs。

Type 1: Change Captioning   
<Img> < Img> Please briefly describe the changes in these two images.   

Type 2: Binary Change Classification   
<Img> <Img> Please judge whether these two images have changed. Please answer yes or no.   

类型 3：类别特定变化量化 < Img> < Img> 请确定有多少条道路和建筑物发生了变化？ 类型 4：变化定位 < Img> < Img> 请用 $3 \times 3$ 网格指明建筑物和道路发生变化的位置。

Type 5: Open-ended QA   
< Img> <Img> Are there any changes in roads or lanes?   
< Img> < Img> What new features has appeared in the images? <Img> <Img> Is there any new construction visible in the images?   

Type 6: Multi-turn Conversation   
<Img> <Img>   

Q1: 请判断这两张图片是否发生了变化。请回答是或否。 Q2: 如果发生了变化，请分别统计道路和建筑物的变化数量。 Q3: 根据上述分析，请详细描述这两张图片的变化情况。

# D. VLMs

VLMs [15], [47], [48] 将视觉编码器与大语言模型（LLMs）结合，以执行多模态任务，如图像描述、跨模态检索和视觉问答（VQA）。与依赖于特定任务架构并拥有独立视觉和语言模块的传统 VQA 系统不同，VLMs 利用统一的 Transformer 架构并进行大规模预训练，以应对多样化的多模态任务。大语言模型在语言理解和推理能力上的进展也推动了 VLMs 的发展。值得注意的实例包括 GPT-4o [49]、Qwen-VL-Plus [50]、GLM-4V-Plus [51] 以及 Gemini-1.5-Pro [52]。尽管这些通用 VLMs 在零-shot 推理中表现良好，但由于领域特定的语义差异以及诸如气象噪声和尺度变化等挑战，它们在遥感（RS）领域的表现不佳。为了应对 VLMs 在 RS 领域应用的挑战，一些研究人员构建了特定于 RS 的大规模视觉语言数据集 [53]，对预训练的 VLMs 进行微调，并取得了显著进展。例如，Kuckreja 等 [13] 开发了 GeoChat，这是一种开创性的模型，可实现区域级对话式问答。类似地，RSGPT [12] 在 InstructBLIP [54] 的基础上构建了一个注释感知的 Q-former 模块，该模块将区域级视觉特征映射到指令条件的查询嵌入，从而改善了区域级对话式问答中的视觉文本对齐。进一步扩展这一思路，Zhan 等 [55] 提出了 SkyEyeGPT，将空间定位的视觉特征进一步与遵循指令的大语言模型对齐。SkyEyeGPT 建立了一个多任务对话框架，包括分割、检测、定位和对话式问答。尽管 RS 领域的 VLMs 取得了重大进展，但它们设计用于单时序分析，无法提供对双时序遥感图像的动态变化的用户互动分析。最近，ChangeChat [56] 开发了一个遵循指令的数据集，并对预训练的 VLMs 进行了微调，以应对时序感知的地理空间任务。然而，ChangeChat 未能根据变化的指令有针对性地提取相关视觉特征，留下了进一步提升响应质量的空间。为了解决这一问题，我们扩展了 ChangeChat 数据集，并重新设计了 DeltaVLM，以提取基于指令指导的差异信息，从而实现多任务、多回合的对话，以便用户驱动的地理空间变化解读。

# III. CHANgECHAT-105K 数据集

最近在遥感变化检测和描述方面的进展产生了基准数据集，如 LEVIR-CC 和 LEVIR-MCI。LEVIR-CC 提供了 10,077 对双时相图像，每对图像有五个人工撰写的描述，支持变化描述，但缺乏对物体数量或精确位置的细粒度注释。LEVIR-MCI 基于 LEVIR-CC，提供像素级变化图以支持二元变化检测，但缺乏对变化信息的深入探索，无法支持交互式分析。为了支持遥感智能交互分析，我们引入了 ChangeChat-105k，这是一个大型数据集，包含 105,107 对从 LEVIR-CC 和 LEVIR-MCI 派生的指令-响应对。我们采用结合规则基础方法和大型语言模型生成的混合管道，利用 ChatGPT 的上下文学习。对于诸如物体计数和定位等结构化任务，我们使用规则基础方法结合 LEVIR-MCI 的像素级变化数据。

# (b) 少样本种子示例

# 输出上下文

# 预期输出 一排排建筑物在底部被建造。沿着道路修建了一排排别墅，取代了底部的树木。

# 变更计数：

{"道路": 1, "建筑物": 10}

# 变更轮廓：

{"道路": [[[1.0, 0.84], ...]], "建筑物": [[[0.95, 0.99], ...]]}

# 修改标题：

问题：请确定有多少条道路和建筑物发生了变化？回答：有1条新路和10栋新房。在道路上方可以清楚地看到8栋房子，另外两栋位于左下角和道路下方，只能看到一点点。

# （c）生成对话数据示例

类型 I：来自更改标题的一般问答 问题：该区域发生的主要变化是什么？

# 类型 II：基于轮廓和计数信息的细粒度问答

引文：i 基于我们的方法生成的图像底部部分，临近新建道路。使用地图和基于OpenCV的轮廓检测提取精确信息。对于开放性任务，我们使用经过筛选的提示和种子示例，通过ChatGPT生成多样化的指令。数据集包括六种指令类型，从结构化信息提取到开放性推理，详见图2，使得对多任务、互动变化分析能力进行全面评估成为可能。1) 变化描述：在此任务中，我们将原始LEVIR-CC三元组 $\boldsymbol { \mathit { I } } _ { t _ { 1 } } , \ \boldsymbol { \mathit { I } } _ { t _ { 2 } } , \boldsymbol { \mathit { C } } )$ 扩展为指令-响应格式。对于每对双时间图像 $( I _ { t _ { 1 } } , I _ { t _ { 2 } } )$ 及其相应的变化描述 $( C )$，我们设计固定指令 $Q$ 为：“请简要描述这两幅图像中的变化。”指令-响应对格式如下：2) 二元变化分类：在此任务中，我们生成指令，要求DeltaVLM判断是否发生变化，期待得到一个二元“是”或“否”的回答。指令设计模板为：“请判断这两幅图像是否发生了变化。回答是或否。”每对图像的真实标注来自LEVIR-MCI中的变化图。3) 类别特定变化量化：我们创建指令指导DeltaVLM量化特定类别的变化，如计算新建建筑物或道路的数量。这些与数量相关的指令基于模板生成，使用OpenCV库的轮廓检测器计算物体数量。4) 变化定位：为在空间上定位变化，我们设计指令，要求DeltaVLM返回$3 \times 3$网格中的变化区域，单元格标记为：

$$
P = \{ \mathrm { T L } , \mathrm { T C } , \mathrm { T R } , \mathrm { C L } , \mathrm { C C } , \mathrm { C R } , \mathrm { B L } , \mathrm { B C } , \mathrm { B R } \} ,
$$

其中 TL=左上角，TC=中上，...，BR=右下角。真实标注数据是通过将每个变化图分割为九个区块获得的；任何像素变化超过 5% 的区块被标记为已变化。5) 开放式问答：为了生成更多样化的指令跟随数据，我们利用 ChatGPT 的上下文学习能力自动生成指令-响应对。如图 3 所示，我们首先向 ChatGPT 提供了系统信息以引导其响应。然后，我们手动设计了一些每种任务类型的示例，以帮助其理解所需的输出结构。具体而言，它生成了两种类型的对话数据：i) 来自变化描述的问答对；以及 ii) 融合提取的轮廓和量化信息的细粒度查询。值得注意的是，我们没有向 ChatGPT 提供任何视觉信息。所有问题和答案均基于我们构建的五个描述、变化轮廓和从变化图中提取的计数信息构造的提示生成，如图 3 (b) 所示。

![](images/2.jpg)  
Fig. 4. An overview of our proposed DeltaVLM.

多轮对话：我们设计了多轮对话，以鼓励DeltaVLM使用链式思维（CoT）方法进行变化分析。指令的难度逐步增加，首先是简单的二元变化分类，然后是变化对象和数量识别，最后是复杂且详细的变化描述任务。

# IV. 方法论

在本节中，我们详细解释了DeltaVLM的架构。

# A. 概述

如图4所示，DeltaVLM 是一个专为交互式 RSICA 设计的端到端框架，包括三个关键步骤：(1) 双时态视觉特征编码，(2) 基于指令的差异特征提取，以及 (3) 基于大型语言模型的语言解码。首先，双时态视觉编码器 (Bi-VE) 从成对输入图像 $I _ { t _ { 1 } }$ 和 $I _ { t _ { 2 } }$ 中提取特征，

$$
F _ { t _ { 1 } } , F _ { t _ { 2 } } = \Phi _ { \mathrm { B i \mathrm { - } V E } } ( I _ { t _ { 1 } } , I _ { t _ { 2 } } ) .
$$

然后，IDPM 通过 CSRM 机制增强双时间特征 $F _ { t _ { 1 } } , F _ { t _ { 2 } }$，接着由 Q-former 将增强后的特征与用户指令 $P$ 和可学习查询 $Q$ 对齐，生成指令引导的差异表示 $\hat { F } _ { \mathrm { d i f f } }$。

$$
\begin{array} { r l } & { F _ { t _ { 1 } } ^ { \prime } , F _ { t _ { 2 } } ^ { \prime } = \Phi _ { \mathrm { e n h a n c e r } } ( F _ { t _ { 1 } } , F _ { t _ { 2 } } ) } \\ & { \qquad \hat { F } _ { \mathrm { d i f f } } = \Phi _ { \mathrm { Q - f o r m e r } } ( [ F _ { t _ { 1 } } ^ { \prime } , F _ { t _ { 2 } } ^ { \prime } ] ; P , Q ) . } \end{array}
$$

最后，$\hat { F } _ { \mathrm { d i f f } }$ 由大型语言模型解码为自然语言响应 $T$，并以指令 $P$ 为条件。

$$
T = \Phi _ { \mathrm { L L M } } ( \hat { F } _ { \mathrm { d i f f } } , P ) .
$$

# B. 双时态视觉编码

为了利用大规模预训练的力量，我们采用 EVA-ViT $\cdot \mathrm { g } / 1 4$ [58] 作为我们的 Bi-VE 主干网络。为了将其适应 RSICA 同时减少灾难性遗忘，我们采用选择性微调：前 37 层 Transformer 被冻结，仅对最后两个模块进行微调。

给定一对双时空遥感图像 $I _ { t _ { 1 } } , I _ { t _ { 2 } } \in \mathbb { R } ^ { H \times W \times 3 }$，其中 $H$ 和 $W$ 分别表示高度和宽度，$\Phi _ { \mathrm { B i - V E } }$ 独立处理每幅图像，以避免过早融合时域信息并防止初始特征提取中的偏差。每幅图像首先被划分为一系列 $16 \times 16$ 的补丁嵌入，然后通过 Transformer 编码层传递，以捕捉复杂的视觉模式。特征从倒数第二层提取（绕过分类头），以获取特定任务的语义，得到 $F _ { t _ { 1 } } , F _ { t _ { 2 } } \in \mathbb { R } ^ { N \times D }$，其中 $N$ 是补丁数量，$D$ 是隐藏维度。从数学上讲，Bi-VE 的操作可表示为：

$$
\begin{array} { r l } & { F _ { t _ { 1 } } = \Phi _ { \mathrm { V i T } } ( I _ { t _ { 1 } } ; \Theta _ { \mathrm { f i n e - t u n e d } } ) \in \mathbb { R } ^ { \frac { H } { 1 6 } \times \frac { W } { 1 6 } \times D } } \\ & { F _ { t _ { 2 } } = \Phi _ { \mathrm { V i T } } ( I _ { t _ { 2 } } ; \Theta _ { \mathrm { f i n e - t u n e d } } ) \in \mathbb { R } ^ { \frac { H } { 1 6 } \times \frac { W } { 1 6 } \times D } , } \end{array}
$$

其中 $\Phi _ { \mathrm { V i T } }$ 是 EVA-ViT- $\mathrm { g } / 1 4$ 编码器，$\Theta _ { \mathrm { f i n e - t u n e d } }$ 表示微调层的参数。由于基于图块的处理，$F _ { t _ { 1 } }$ 和 $F _ { t _ { 2 }$ 的空间分辨率降低了 16 倍。这些特征图随后被输入到 IDPM。

# $C .$ 指导性指令差异感知

给定双时态特征 $F _ { t _ { 1 } } , F _ { t _ { 2 } }$，我们首先计算原始视觉差异，其中 $F _ { \mathrm { d i f f } } \ \in \ \mathbb { R } ^ { N \times D }$ 捕捉所有像素级的变化。直接将 $F _ { \mathrm { d i f f } }$ 解码为语言可能会引入干扰信息，例如传感器差异、光照或季节变化。为了解决这个问题，我们首先通过 CSRM 机制探索 $F _ { \mathrm { d i f f } }$ 和 $F _ { t _ { 1 } }$、$F _ { t _ { 2 } }$ 之间的语义关系，以消除无关变化干扰。随后，通过指令引导的 Q-former 实现跨模态对齐。

$$
F _ { \mathrm { d i f f } } = F _ { t _ { 2 } } - F _ { t _ { 1 } } ,
$$

1) 跨语义关系测量：CSRM机制分为三个步骤：上下文化、门控和过滤。上下文化。为了理解变化如何与每个时间状态相关，我们通过将差异特征与原始特征融合来计算上下文向量，其中 $[ \cdot ; \cdot ]$ 表示通道维度的拼接，$W _ { c } , W _ { c } ^ { \prime } \in$ $\mathbb { R } ^ { D \times 2 D ^ { \vee } }$ 和 $b _ { c } , b _ { c } ^ { \prime } \in \mathbb { R } ^ { D }$ 是可学习的权重和基。在这个过程中，线性投影将拼接特征转换到一个新的空间，强调语义连接，同时，对输出施加tanh激活函数将其限制在 $[ - 1 , 1 ]$ 范围内。

$$
\begin{array} { r l } & { C _ { t _ { 1 } } = \operatorname { t a n h } ( W _ { c } [ F _ { \mathrm { d i f f } } ; F _ { t _ { 1 } } ] + b _ { c } ) } \\ & { C _ { t _ { 2 } } = \operatorname { t a n h } ( W _ { c } ^ { \prime } [ F _ { \mathrm { d i f f } } ; F _ { t _ { 2 } } ] + b _ { c } ^ { \prime } ) , } \end{array}
$$

门控机制。通过上下文化，这些上下文向量捕捉变化与上下文之间的关系。为了进一步根据每个检测到的变化与其语义相关性加权，我们采用了一种门控机制，灵感来源于门控递归单元（GRU）[59]。此步骤通过sigmoid激活函数生成门向量 $G _ { t _ { 1 } }$ 和 $G _ { t _ { 2 } $。

$$
\begin{array} { r l } & { G _ { t _ { 1 } } = \sigma ( W _ { \mathrm { g } } [ F _ { \mathrm { d i f f } } ; F _ { t _ { 1 } } ] + b _ { \mathrm { g } } ) } \\ & { G _ { t _ { 2 } } = \sigma ( W _ { \mathrm { g } } ^ { \prime } [ F _ { \mathrm { d i f f } } ; F _ { t _ { 2 } } ] + b _ { \mathrm { g } } ^ { \prime } ) , } \end{array}
$$

其中 $\sigma$ 是产生 $(0, 1)$ 范围内的相关性分数的 sigmoid 函数，$W_{g}, W_{g}^{\prime} \in \bar{\mathbb{R}}^{\bar{D} \times 2D}$，$b_{g}, b_{g}^{\prime} \in \mathbb{R}^{D}$ 是可学习的权重和偏置。过滤。最后，我们通过门控向量与相应上下文向量之间的逐元素相乘 $(\odot)$ 有选择性地保留语义相关的信息：

$$
\begin{array} { r l } & { F _ { t _ { 1 } } ^ { \prime } = G _ { t _ { 1 } } \odot C _ { t _ { 1 } } } \\ & { F _ { t _ { 2 } } ^ { \prime } = G _ { t _ { 2 } } \odot C _ { t _ { 2 } } . } \end{array}
$$

该乘法在 $G_{t_{1}}$ 和 $G_{t_{2}}$ 的指导下，对 $C_{t_{1}}$ 和 $C_{t_{2}}$ 进行了精细化，通过抑制具有低门值的无关成分（例如噪声），同时保留具有高门值的重要变化（例如新的结构或土地覆盖变化）。因此，得到的过滤特征 $F_{t_{1}}^{\prime}$ 和 $F_{t_{2}}^{\prime}$ 仅保留语义相关的变化，然后传递给后续的 Q-former 模块。2) $Q$ former 用于跨模态对齐：受到 InstructBLIP [54] 的启发，我们的 Q-former 模块专门设计用于生成与给定指令 $P$ 对齐的变化感知特征。该过程以一组可学习的查询嵌入 $Q \in \mathbb{R}^{L \times d}$ 开始，其中 $L = 32$ 是查询的数量，$d$ 是与 LLM 输入空间匹配的特征维度。这些查询首先通过自注意力层进行精细化：最后，指令感知的变化特征通过前馈网络，生成最终的紧凑输出：

$$
\hat { F } _ { \mathrm { d i f f } } = \mathrm { F F N } ( Q _ { \mathrm { C A } } ) \in \mathbb { R } ^ { 3 2 \times d } .
$$

通过整合这些步骤，Q-former 确保提取的特征有效捕捉与指令相关的变化，同时通过查询瓶颈保持计算效率。

# D. 基于大语言模型的语言解码器

我们选择了 Vicuna-7B [60] 作为我们的语言解码器，这是一个强大的仅解码的 LLM，经过 LLaMA [61] 的微调。我们的解码器将视觉特征 $\hat { F } _ { \mathrm { d i f f } }$ 和指令提示 $P$ 作为输入，生成特定于指令的变化描述。该过程首先对指令提示 $P$ 进行分词和嵌入。

$$
E = \Phi _ { \mathrm { e m b e d d i n g } } ( P ) ,
$$

其中 $\Phi _ { \mathrm { e m b e d d i n g } }$ 表示分词器和嵌入函数，将原始文本转换为适合语言模型的嵌入序列 $E$。这些嵌入通过将单词或子词映射到高维向量，捕捉提示的语义本质。学习到的与提示对齐的特征 $\hat { F } _ { \mathrm { d i f f } }$ 和嵌入提示 $E$ 一起作为语言解码器的输入，生成描述性标题 $T = \{ t _ { 1 } , \dots , t _ { N } \}$，以总结双时间变化：

$$
T = \Phi _ { \mathrm { L L M } } ( \hat { F } _ { \mathrm { d i f f } } , E ) \in \mathcal { C } ^ { N } ,
$$

其中 $\Phi _ { \mathrm { L L M } }$ 表示 Vicuna-7B 解码器，$T$ 是来自词汇表 $\mathcal { C }$ 的 $N$ 个词元的序列。这个过程有效地解读了用户查询背景下的视觉差异，生成特定任务的变化描述。

# $E$ 训练目标

与典型的 RSICC 方法生成固定描述的 RS 图像对不同，DeltaVLM 是在指令条件数据上进行训练的。为此，我们通过指令提示 $P _ { j }$ 来扩充数据集，创建 $D _ { \mathrm { t r a i n } } \ =$ $\{ ( I _ { 1 } , P _ { 1 } , T _ { 1 } ) , \cdot \cdot \cdot , ( I _ { M } , P _ { M } , T _ { M } ) \}$，其中每个 $P _ { j }$ 是与双时相图像对 $I _ { j }$ 及其目标描述 $T _ { j }$ 对应的用户查询，从而使模型能够适应多样化的用户指令。模型使用交叉熵损失函数进行训练：

$$
Q _ { \mathrm { S A } } = \mathrm { S e l f A t t e n t i o n } ( Q ) .
$$

接下来，精炼查询 $Q _ { \mathrm { S A } }$ 通过交叉注意机制同时关注于连接的视觉特征和指令提示：

$$
\begin{array} { r } { Q _ { \mathrm { C A } } = \mathrm { C r o s s A t t e n t i o n } ( Q _ { \mathrm { S A } } , [ F _ { t _ { 1 } } ^ { \prime } ; F _ { t _ { 2 } } ^ { \prime } ] , P ) . } \end{array}
$$

此步骤动态地将变化特征与任务特定指令对齐，针对用户的查询进行定制。

$$
\mathcal { L } _ { \mathrm { t r a i n } } = - \frac { 1 } { K } \sum _ { i = 1 } ^ { K } w _ { i } \log ( \hat { w } _ { i } ) ,
$$

其中 $K$ 表示目标描述中的总词元数量，$w_{i}$ 是位置 $i$ 处的独热编码真实词元，而 $\hat{w}_{i}$ 是 DeltaVLM 对第 $i$ 个词元的预测概率。通过在扩增数据集 $D_{\mathrm{train}}^{\prime}$ 上最小化该损失，训练 DeltaVLM 生成准确且与用户特定指令相关的描述。

<table><tr><td>Instruction Type</td><td>Source Data</td><td>Generation Method</td><td>Response Format</td><td>Training Set</td><td>Test Set</td></tr><tr><td>Change Captioning</td><td>LEVIR-CC</td><td>Rule-based</td><td>Descriptive Text</td><td>34,075</td><td>1,929</td></tr><tr><td>Binary Change Classification</td><td>LEVIR-MCI</td><td>Rule-based</td><td>Yes/No Response</td><td>6,815</td><td>1,929</td></tr><tr><td>Category-specific Change Quantification</td><td>LEVIR-MCI</td><td>Rule-based</td><td>Object Count</td><td>6,815</td><td>1,929</td></tr><tr><td>Change Localization</td><td>LEVIR-MCI</td><td>Rule-based</td><td>Grid Location</td><td>6,815</td><td>1,929</td></tr><tr><td>Open-ended QA</td><td>Derived (LEVIR-CC/MCI)</td><td>GPT-assisted</td><td>Q&amp;A Pair</td><td>26,600</td><td>7,527</td></tr><tr><td>Multi-turn Conversation</td><td>Derived (LEVIR-MCI)</td><td>Rule-based</td><td>Multi-turn Dialogue</td><td>6,815</td><td>1,929</td></tr><tr><td>Total</td><td>−</td><td>−</td><td>−</td><td>87,935</td><td>17,172</td></tr></table>

# V. 实验与分析

在本节中，我们展示了全面的实验来评估 DeltaVLM 在 RSICA 中的有效性。我们首先描述实验设置，然后报告多个任务的定量结果，最后通过消融研究分析关键组件。

# A. 实验设置

1) 数据集：我们在 ChangeChat $1 0 5 \mathrm { k }$ 数据集上评估了我们提出的 DeltaVLM，该数据集包含 105,107 对与双时间图像补丁（大小为 $2 5 6 \times 2 5 6$，空间分辨率为 $0 . 5 \mathrm { m } / \mathrm { p i x e l }$）对齐的指令-响应对。每对图像都标注了多种任务类型：二元变化检测、物体计数、变化定位和变化描述。为了评估，我们将数据集分为训练集和测试集，任务和子集间指令-响应对的详细分布见表 I。 2) 实施细节：所有实验均在使用 NVIDIA L20 GPU 的 Ubuntu 20.04 上通过 PyTorch 框架进行。为了进行数据增强，我们首先应用随机裁剪，去除 $0 - 5 \%$ 的图像内容，然后在 $[ - 1 5 ^ { \circ } , + 1 5 ^ { \circ } ]$ 范围内进行随机旋转。增强后的图像被调整为 $2 2 4 \times 2 2 4$ 像素，以符合 ViT $\mathrm { { g } / 1 4 }$ 主干网络的补丁嵌入要求。我们采用了 AdamW 优化器 [62]，权重衰减设置为 0.05 以进行正则化。初始学习率设置为 $1 \times 1 0 ^ { - 5 }$，批量大小为 24，最大训练周期为 30。 3) 评估指标：为了全面评估 DeltaVLM 的多任务能力，我们采用了该领域广泛使用的任务特定指标，以提供 DeltaVLM 效能的整体视角。 • 变化描述：我们采用 BLEU-N $\mathrm { { N = 1 } }$，2，3，4) [63]、METEOR [64]、ROUGE-L [65] 和 CIDEr [66] 来评估生成的变化描述的质量。这些指标分别评估了 n-gram 重叠、语义相似性、句子结构和人类共识对齐。 • 二元变化分类：我们使用准确率、精确率、召回率和 F1-score 来衡量分类性能。F1-score 提供了一个平衡的度量，对于变化/不变化的分布尤为重要。 • 类别特定变化量化：我们使用平均绝对误差 (MAE) 和均方根误差 (RMSE) 来评估计数准确性，其中 MAE 捕获平均偏差，RMSE 则惩罚较大的错误。 • 变化定位：变化定位任务要求返回 $3 \times 3$ 网格格式中变化的位置，该任务属于多类分类任务。我们使用精确率、召回率、F1-score、Jacard 相似性和子集准确率来评估定位质量。

# B. 与基准模型的比较

我们将DeltaVLM与各种任务的最先进基线进行比较：变化描述、二元变化分类、特定类别变化量化和变化定位。这些基线分为两组：（1）针对遥感（RS）特定的变化描述模型，专注于适应遥感图像的领域适配架构，包括RS-ICCFormer [8]、Prompt-CC [35]、PSNet [34]和SFT [33]；（2）通用的大型视觉语言模型，包括GPT-4o [49]、Qwen-VL-Plus [50]、GLM-4V-Plus [51]和Gemini-1.5-Pro [52]。

1) 更改字幕：我们首先在ChangeChat-105k测试集上评估DeltaVLM与专用RS更改字幕和通用VLM基准的表现。如表II所示，我们在大多数指标上达到了最先进的性能，展示了指导性变更分析的有效性。在通用VLM中，所有模型在所有指标上的性能明显较低，BLEU-4分数范围从13.85（GLM-4V-Plus）到22.95（Qwen-VL-Plus），而专用模型则超过62。这一显著差距突显了在没有领域特定微调的情况下将通用VLM应用于RS变更分析的挑战，因为它们难以捕捉卫星图像中覆盖地物变化或结构变化等细微细节。相比之下，专用于RS的模型表现更强，利用了针对RS任务优化的架构。虽然SFT在BLEU-4、METEOR和CIDEr上略占优势，这可能归因于其在微调过程中对n-gram一致性的优化。DeltaVLM在早期n-gram指标上的优越表现表明其在词级和短语级匹配上更为准确，这表明我们的指导性方法生成的更改描述与用户查询更加上下文相关。表II 在ChangeChat-105K数据集上与最先进的方法在更改字幕任务中的比较。

<table><tr><td>Category</td><td>Method</td><td>BLEU-1</td><td>BLEU-2</td><td>BLEU-3</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE-L</td><td>CIDEr-D</td></tr><tr><td rowspan="4">VLMs</td><td>GPT-4o [49]</td><td>46.03</td><td>33.09</td><td>24.66</td><td>18.05</td><td>22.50</td><td>56.49</td><td>90.92</td></tr><tr><td>Qwen-VL-Plus [50]</td><td>41.31</td><td>33.19</td><td>27.96</td><td>22.95</td><td>18.04</td><td>51.24</td><td>92.99</td></tr><tr><td>GLM-4V-Plus [51]</td><td>35.59</td><td>24.26</td><td>18.54</td><td>13.85</td><td>20.13</td><td>54.39</td><td>93.16</td></tr><tr><td>Gemini-1.5-Pro [52]</td><td>45.68</td><td>33.59</td><td>25.53</td><td>19.01</td><td>22.64</td><td>56.25</td><td>91.37</td></tr><tr><td rowspan="4">RS Change Captioning Models</td><td>RSICCFormer [8]</td><td>84.72</td><td>76.27</td><td>68.87</td><td>62.77</td><td>39.61</td><td>74.12</td><td>134.12</td></tr><tr><td>PromptCC [35]</td><td>83.66</td><td>75.73</td><td>69.10</td><td>63.54</td><td>38.82</td><td>73.72</td><td>136.44</td></tr><tr><td>SNet [34]</td><td>83.86</td><td>75.13</td><td>67.89</td><td>62.11</td><td>38.80</td><td>73.60</td><td>132.62</td></tr><tr><td>SFT 33]</td><td>84.56</td><td>75.87</td><td>68.64</td><td>62.87</td><td>39.93</td><td>74.69</td><td>137.05</td></tr><tr><td rowspan="4"></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>85.78</td><td>77.15</td><td>69.24</td><td>62.51</td><td>39.47</td><td>75.01</td><td>136.72</td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

表 III 二元变化分类的结果

<table><tr><td>Method</td><td>Accuracy (%)</td><td>Precision (%)</td><td>Recall (%)</td><td>F1 (%)</td></tr><tr><td>GPT-4o [49]</td><td>84.81</td><td>83.58</td><td>86.62</td><td>85.07</td></tr><tr><td>Qwen-VL-Plus s [50]</td><td>58.22</td><td>73.65</td><td>25.52</td><td>37.90</td></tr><tr><td>GLM-4V-Plus [51]</td><td>79.83</td><td>88.38</td><td>68.67</td><td>77.29</td></tr><tr><td>Gemini-1.5-Pro [52]</td><td>83.83</td><td>84.03</td><td>83.51</td><td>83.77</td></tr><tr><td>Ours</td><td>93.99</td><td>96.29</td><td>91.49</td><td>93.83</td></tr></table>

二元变化分类：对于二元变化分类，基于用户指令判断双时域遥感影像之间是否发生变化，我们将DeltaVLM与大型视觉语言模型进行比较，结果在所有指标上表现出色，准确率为$93.99\%$，精确率为$96.29\%$，召回率为$91.49\%$，F1-score为$93.83\%$。在基准模型中，性能也存在显著差异。GPT-4o和Gemini-1.5-Pro在基础变化检测任务中表现出竞争力，F1-score分别为$85.07\%$和$83.77\%$，显示出它们在基本变化检测任务中的能力。然而，像Qwen-VL-Plus这样的模型表现明显滞后，仅有$25.52\%$的召回率和$37.90\%$的F1-score，表明其在预测“无变化”案例时存在明显偏向。此外，DeltaVLM的卓越表现使其F1-score相比最佳基准（GPT-4o）提高了$8.76\%$，在精确率$(+7.91\%)$和召回率$(+4.87\%)$上均有显著提升。平衡的精确率-召回率权衡$(96.29\%$ vs $91.49\%)$ 表明我们的模型在正确识别变化区域和最小化假阳性方面均有所增强。表 IV 变化量化的结果。

<table><tr><td>Method</td><td colspan="2">Roads</td><td colspan="2">Buildings</td></tr><tr><td></td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td>GPT-4o [49]</td><td>0.49</td><td>1.00</td><td>1.86</td><td>4.57</td></tr><tr><td>Qwen-VL-Plus [50]</td><td>0.90</td><td>1.50</td><td>4.41</td><td>9.03</td></tr><tr><td>GLM-4V-Plus [51]</td><td>0.82</td><td>1.62</td><td>2.05</td><td>4.61</td></tr><tr><td>Gemini-1.5-Pro [52]</td><td>0.58</td><td>1.25</td><td>2.56</td><td>8.71</td></tr><tr><td>Ours</td><td>0.24</td><td>0.70</td><td>1.32</td><td>2.89</td></tr></table>

3) 分类特定变化量化：我们不仅检测变化，还评估模型在两个关键基础设施类别（道路和建筑）中计数特定物体变化的能力，使用MAE和RMSE指标来衡量计数准确性。如表IV所示，DeltaVLM在这两个类别中均实现了最低错误，分别为道路的MAE/RMSE为$0 . 2 4 / 0 . 7 0$，建筑的MAE/RMSE为$1 . 3 2 / 2 . 8 9$。在道路计数方面，我们的模型相比于表现最佳的基线模型GPT-4o，MAE减少了$5 1 \%$，显示出对线性基础设施变化的精确检测。低RMSE表明在计数时表现一致，没有大的计数错误。由于建筑物的大小和遮挡模式各异，建筑量化对所有模型来说更具挑战性，但DeltaVLM仍实现了相较于GPT-4o的$2 9 \%$ MAE减少。DeltaVLM在所有指标上平均提高了$3 5 \%$，这证明了我们的CSRM模块在关注相关物体特定变化的同时过滤不相关变异的有效性。有趣的是，所有模型在量化建筑变化时的表现普遍不如道路变化，表IV中建筑的MAE和RMSE值更高。我们将此归因于建筑的复杂性，建筑结构的多样性对准确量化构成了重大挑战，相较之下，道路结构则相对统一和线性。尽管面临这些挑战，DeltaVLM在两个类别中始终优于所有基线模型，在性能上保持明显优势。这一优越表现凸显了我们模型在应对多样化变化量化任务复杂性方面的强大能力。 4) 变化定位：我们进一步评估了DeltaVLM定位变化位置的能力，重点关注道路和建筑，结果见表V。如表V所示，DeltaVLM优于所有大型VLM。与表现最佳的基线模型Gemini-1.5-Pro相比，DeltaVLM在F1-score上实现了$2 6 . 2 \%$的显著提升，同时在精确率和召回率方面分别提高了$2 6 . 6 2 \%$和$2 5 . 7 7 \%$。这些结果突显了缺乏专门变化检测机制的模型在此任务中面临的挑战。在建筑定位方面，尽管城市结构带来了更大的复杂性，我们的模型在所有指标上始终保持并且甚至放大了其性能优势。在道路和建筑定位任务中，DeltaVLM在变化定位方面相较于通用大型VLM的一致且显著的性能提升凸显了我们定制架构和领域特定训练在解决遥感影像复杂空间需求方面的有效性。表V 道路和建筑变化定位结果。

<table><tr><td>Category</td><td>Method</td><td>Prec.1</td><td>Rec.2</td><td>F13</td><td>J. Sim.4</td><td>S. Acc.5</td></tr><tr><td rowspan="5">Roads</td><td>GPT-4o [49]</td><td>30.44</td><td>27.01</td><td>28.62</td><td>7.80</td><td>33.85</td></tr><tr><td>Qwen-VL-Plus [50]</td><td>15.42</td><td>1.40</td><td>2.56</td><td>0.25</td><td>67.19</td></tr><tr><td>GLM-4V-Plus [51]</td><td>21.99</td><td>33.32</td><td>26.49</td><td>7.93</td><td>6.79</td></tr><tr><td>Gemini-1.5-Pro [52]</td><td>43.01</td><td>40.55</td><td>41.74</td><td>9.62</td><td>48.63</td></tr><tr><td>Ours</td><td>69.63</td><td>66.32</td><td>67.94</td><td>14.00</td><td>70.92</td></tr><tr><td rowspan="5">Buildings</td><td>GPT-4o [49]</td><td>55.63</td><td>33.70</td><td>41.98</td><td>14.09</td><td>41.47</td></tr><tr><td>Qwen-VL-Plus [50]</td><td>22.23</td><td>20.78</td><td>21.48</td><td>6.52</td><td>7.26</td></tr><tr><td>GLM-4V-Plus [51]</td><td>38.98</td><td>57.83</td><td>46.57</td><td>17.93</td><td>17.11</td></tr><tr><td>Gemini-1.5-Pro [52]</td><td>65.71</td><td>51.75</td><td>57.90</td><td>18.62</td><td>45.62</td></tr><tr><td>Ours</td><td>77.79</td><td>80.22</td><td>78.99</td><td>23.15</td><td>65.53</td></tr></table>

1 精确率 $( \% )$ 2 召回率 $( \% )$ 3 F1 值 $( \% )$ 4 Jaccard 相似度 $( \% )$ 5 子集准确率 $( \% )$ 表 VI 开放式问答结果

<table><tr><td>Method</td><td>B-11</td><td>B-21</td><td>B-31</td><td>B-41</td><td>MTR2</td><td>R-L3</td><td>C-D4</td></tr><tr><td>GPT-4o [49]</td><td>33.08</td><td>21.08</td><td>14.06</td><td>9.68</td><td>22.24</td><td>35.53</td><td>72.58</td></tr><tr><td>Qwen-VL-Plus [50]</td><td>24.75</td><td>12.55</td><td>6.70</td><td>3.88</td><td>16.69</td><td>27.74</td><td>27.22</td></tr><tr><td>GLM-4V-Plus [51]</td><td>34.27</td><td>22.38</td><td>15.66</td><td>11.43</td><td>22.48</td><td>37.11</td><td>100.66</td></tr><tr><td>Gemini-1.5-Pro [52]</td><td>32.90</td><td>20.44</td><td>13.38</td><td>9.06</td><td>21.85</td><td>35.19</td><td>68.64</td></tr><tr><td>Ours</td><td>36.67</td><td>27.09</td><td>20.62</td><td>16.21</td><td>17.85</td><td>32.60</td><td>127.38</td></tr></table>

BLEU-1/2/3/4 2 METEOR。3ROUGE-L 4 CIr-D。5) 开放式问题回答：为了评估DeltaVLM在超越变化图像说明的语义理解能力，我们对其在开放式问题回答中的表现进行了评估。此任务比变化图像说明更具挑战性，因为它要求模型推断并回应多样化的用户特定查询。如表VI所示，尽管GLM-4V-Plus在METEOR和ROUGE-L上表现竞争力，但在CIDEr上的表现较弱，表明其与人类撰写答案的对齐程度较低。GPT-4o和Gemini-1.5-Pro表现中等，而Qwen-VL-Plus整体表现最差——尤其是在CIDEr上——突显其处理复杂多模态RS输入的局限性。这些结果展示了DeltaVLM在RSICA任务中理解和回应多样化开放式指令的强大能力。

# C. 消融研究

为了验证我们提出的Bi-VE微调方法和跨语义关系测量机制对遥感变化分析任务性能的影响，我们对变化描述和二元变化分类任务进行了消融研究。表II比较了DeltaVLM在不同条件下的表现：没有Bi-VE微调（标记为“w/o BiVE FT”）和没有CSRM（标记为“w/o CSRM”）。没有CSRM机制的模型在各项指标上显著下降。在固定Bi-VE参数后，模型在所有指标上有所改善。然而，与完整模型相比，仍然存在不足之处。 表VII 变化描述任务的消融分析

<table><tr><td>Method</td><td>B-11</td><td>B-21</td><td>B-31</td><td>B-41</td><td>MTR2</td><td>R-L3</td><td>C-D4</td></tr><tr><td>w/o CSRM</td><td>64.42</td><td>56.52</td><td>53.08</td><td>51.40</td><td>29.31</td><td>60.54</td><td>101.92</td></tr><tr><td>w/o Bi-VE FT</td><td>84.24</td><td>75.62</td><td>67.91</td><td>61.40</td><td>39.29</td><td>74.73</td><td>134.76</td></tr><tr><td>DeltaVLM</td><td>85.78</td><td>77.15</td><td>69.24</td><td>62.51</td><td>39.47</td><td>75.01</td><td>136.72</td></tr></table>

BLEU-1/2/3. 2 METEOR. 3 ROUGE-L. 4 CIDEr-D. 表 VIII 二元变化分类任务的消融分析

<table><tr><td>Method</td><td>Accuracy (%)</td><td>Precision (%)</td><td>Recall (%)</td><td>F1 (%)</td></tr><tr><td>w/o CSRM</td><td>50.13</td><td>75.00</td><td>0.31</td><td>0.62</td></tr><tr><td>w/o Bi-VE FT</td><td>90.57</td><td>99.49</td><td>81.54</td><td>89.62</td></tr><tr><td>DeltaVLM</td><td>93.99</td><td>96.29</td><td>91.49</td><td>93.83</td></tr></table>

DeltaVLM模型。这些实验结果表明，CSRM模块对DeltaVLM至关重要，因为其缺失会导致模型难以识别视觉差异信息。此外，微调视觉编码器最后两层的参数确实增强了其在遥感场景的适应性，提高了模型对标题生成的能力。在二元变化分类任务中，与没有Bi-VE微调条件相比，微调Bi-VE显著提高了F1分数。缺少CSRM模块同样导致所有评估指标的性能较差。然而，该模型对预测“无变化”类别表现出强烈偏向，这表明在没有CSRM提供的语义过滤的情况下，模型无法检测到有意义的差异。

# D. 质性结果

为了进一步展示DeltaVLM的交互能力，我们评估了其在多轮对话场景中的表现。图5展示了示例交互，其中模型基于双时间遥感指数（RSIs）响应一系列用户查询。这些示例说明了DeltaVLM在维持对话上下文、分析图像内容和生成连贯反应方面的能力，涵盖了变化检测、描述性文字生成、量化和定位等多种查询类型。模型一致且具上下文意识的回复突显了其在遥感变化分析的多轮对话场景中实际应用的潜力。

# VI. 结论

在本文中，我们介绍了交互式遥感图像变化分析（RSICA），这是一项通过用户中心的、基于查询的方式重新定义遥感图像分析的任务。为了解决这一新挑战，我们提出了DeltaVLM，这是一种创新的端到端框架，集成了一种选择性微调的双时序视觉编码器、一种具有CSRM机制的IDPM、用于跨模态对齐的指令引导Q-former，以及用于生成上下文感知响应的LLM解码器。为了支持这一任务，我们构建了一个大规模数据集，ChangeChat-$10^5 \mathrm{k}$。在该数据集上的广泛实验表明，DeltaVLM在多种交互式RSICA子任务中表现优越。与现有的多模态视觉语言模型相比，DeltaVLM在交互性和复杂场景中表现出色。消融研究进一步验证了各个组件的有效性。在未来的工作中，我们将探索统一多模态输出的创新架构，同时增强模型的推理能力和提高响应效率。

![](images/3.jpg)  
Fig. 5. Demonstration of multi-round dialogue capability of DeltaVLM.

# REFERENCES

[1] C. Van Westen, "Remote sensing for natural disaster management," Int. Arch. Photogramm. Remote Sens. Spat. Inf. Sci., vol. 33, no. B7/4; PART 7, pp. 16091617, 2000.   
[2] R. R. Chowdhury, "Driving forces of tropical deforestation: The role of remote sensing and spatial models," Singap. J. Trop. Geogr., vol. 27, no. 1, pp. 82101, 2006.   
[3] R. R. Navalgund, V. Jayaraman, and P. Roy, "Remote sensing applications: An overview," Curr. Sci., pp. 17471766, 2007.   
[4] A. Bannari, D. Morin, G. Bénié, and F. Bonn, "A theoretical review of different mathematical models of geometric corrections applied to remote sensing images," Remote sensing reviews, vol. 13, no. 1-2, pp. 2747, 1995.   
[5] L. Ding, D. Hong, M. Zhao, H. Chen, C. Li, J. Deng, N. Yokoya, L. Bruzzone, and J. Chanussot, "A survey of sample-efficient deep learning for change detection in remote sensing: Tasks, strategies, and challenges," IEEE Geosci. Remote Sens. Mag., pp. 227, 2025.   
[6] B. Qu, X. Li, D. Tao, and X. Lu, "Deep semantic understanding of high resolution remote sensing image," in 2016 Int. Conf. Comput. Inf. Telecommun. Syst. (CITS). IEEE, 2016, pp. 15.   
[7] S. Lobry, D. Marcos, J. Murray, and D. Tuia, "RSVQA: Visual question answering for remote sensing data," IEEE Trans. Geosci. Remote Sens., vol. 58, no. 12, pp. 85558566, 2020.   
[8] C. Liu, R. Zhao, H. Chen, Z. Zou, and Z. Shi, "Remote sensing image change captioning with dual-branch transformers: A new method and a large scale dataset," IEEE Trans. Geosci. Remote Sens., vol. 60, pp. 120, 2022.   
[9] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever et al., "Language models are unsupervised multitask learners," OpenAI blog, vol. 1, no. 8, p. 9, 2019.   
10] J. Lu, D. Batra, D. Parikh, and S. Lee, "ViLBERT: Pretraining taskagnostic visiolinguistic representations for vision-and-language tasks," in Proc. Adv. Neural Inf. Process. Syst., 2019.   
[11] J. Wei, M. Bosma, V. Y. Zhao, K. Guu, A. W. Yu, B. Lester, N. Du, A. M. Dai, and Q. V. Le, "Finetuned language models are zero-shot learners," arXiv preprint arXiv:2109.01652, 2021.   
[12] Y. Hu, J. Yuan, C. Wen, X. Lu, Y. Liu, and X. Li, "RSGPT: A remote sensing vision language model and benchmark," ISPRS J. Photogramm. Remote Sens., vol. 224, pp. 272286, 2025.   
[13] K. Kuckreja, M. S. Danish, M. Naseer, A. Das, S. Khan, and F. S. Khan, "Geochat: Grounded large vision-language model for remote sensing." in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2024, pp. 27 831 27 840.   
[14] Y. Bazi, L. Bashmal, M. M. Al Rahhal, R. Ricci, and F. Melgani, "RSllava: A large vision-language model for joint captioning and question answering in remote sensing imagery," Remote Sens., vol. 16, no. , p. 1477, 2024.   
[15] H. Liu, C. Li, Q. Wu, and Y. J. Lee, "Visal instruction tunig," Proc. Adv. Neural Inf. Process. Syst., vol. 36, pp. 34 89234 916, 2023.   
[16] B. Rasti, P. Scheunders, P. Ghamisi, G. Licciardi, and J. Chanussot, "Noise reduction in hyperspectral imagery: Overview and application," Remote Sens., vol. 10, no. 3, p. 482, 2018.   
[17] OpenAI, "ChatGPT: Optimizing language models for dialogue," 2022, accessed: 2025-05-19. [Online]. Available: https://openai.com/blog/ chatgpt   
[18] A. Singh, "Review article digital change detection techniques using remotely-sensed data," Int. J. Remote Sens., vol. 10, no. 6, pp. 989 1003, 1989.   
[19] R. F. Nelson, "Detecting forest canopy change due to insect activity ans,"   e Sns. vol . 13031314, 1983.   
[20 A. A. Nielsen, K. Conradsen, and J. J. Simpson, "Multivariate alteration detection (mad) and maf postprocessing in multispectral, bitemporal image data: New approaches to change detection studies," Remote Sens. Environ., vol. 64, no. 1, pp. 119, 1998.   
[21] P. Serra, X. Pons, and D. Sauri, "Post-classification change detection with data from different sensors: some accuracy considerations," Int. J. Remote Sens., vol. 24, no. 16, pp. 33113340, 2003.   
[22] T. Blaschke, "Object based image analysis for remote sensing," ISPRS J. Photogramm. Remote Sens., vol. 65, no. 1, pp. 216, 2010.   
[23] R. C. Daudt, B. Le Saux, and A. Boulch, "Fully convolutional siamese networks for change detection," in Proc. Int. Conf. Image Process. IEEE, 2018, pp. 40634067.   
[24] O. Ronneberger, P. Fischer, and T. Brox, "U-net: Convolutional networks for biomedical image segmentation," in Int. Conf. Med. Image Comput. Comput.-Assist. Interv. (MICCAI). Springer, 2015, pp. 234241.   
[25] H. Chen, Z. Qi, and Z. Shi, "Remote sensing image change detection with transformers," IEEE Trans. Geosci. Remote Sens., vol. 60, pp. 114, 2021.   
[26] W. G. C. Bandara and V. M. Patel, "A transformer-based siamese network for change detection," in Proc. IEEE Int. Geosci. Remote Sens. Symp. IEEE, 2022, pp. 207210.   
[27] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, "A simple framework for contrastive learning of visual representations," in roc. Int. Con. Mach. Learn. PmLR, 2020, pp. 15971607.   
.  .  F. transformers," arXiv preprint arXiv:2106.08254, 2021.   
[29] M. Palatucci, D. Pomerleau, G. E. Hinton, and T. M. Mitchell, "Zeroshot learning with semantic output codes," Proc. Adv. Neural Inf. Prcess. Syst., vol. 2, 2009   
[30] N. Longbotham, F. Pacifici, T. Glenn, A. Zare, M. Volpi, D. Tuia, EChristophe, J. Michel, J. Inglada, J.Chanussot et al., "Multi-odal change detection, application to the detection of flooded areas: Outcome of the 20092010 data fusion contest," IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens., vol. 5, no. 1, pp. 331342, 2012.   
[31] G. Hoxha, S. Chouaf, F. Melgani, and Y. Smara, "Change captioning: A new paradigm for multitemporal remote sensing image analysis," IEEE Trans. Geosci. Remote Sens., vol. 60, pp. 114, 2022.   
[32] Q. You, H. Jin, Z. Wang, C. Fang, and J. Luo, "Image captioning with semantic attention," in roc. IEE Conf. Comput. Vis. Pattern Recognit., 2016, pp. 46514659.   
[33] D. Sun, Y. Bao, J. Liu, and X. Cao, "A lightweight sparse focus m  m n h pt"  . Top. Appl. Earth Obs. Remote Sens., 2024.   
[34] C. Liu, J. Yang, Z. Qi, Z. Zou, and Z. Shi, "Progressive scale-aware network for remote sensing image change captioning," in Proc. IEEE Int. Geosci. Remote Sens. Symp. IEEE, 2023, pp. 66686671.   
[ .Liu R.Zhao, J.hen, Z., Z. Zou and Z.Shi, Ad paradigm with prompt learning for remote sensing image change captioning," IEEE Transactions on Geoscience and Remote Sensing, 2023.   
[36] Y. Zhu, L. Li, K. Chen, C. Liu, F. Zhou, and Z. X. Shi, "Semanticcc: Boosting remote sensing image change captioning via foundational knowledge and semantic guidance," IEEE Trans. Geosci. Remote Sens., vol. 62, pp. 116, 2024.   
[37] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Ne y . . ar few-shot learers," ro.Av. eural In rocess. st. vol 3, p. 18771901, 2020.   
[38] M. Noman, N. Ahsan, M. Naseer, H. Cholakkal, R. M. Anwer, S. Khan, change description," arXiv preprint arXiv:2409.16261, 2024.   
[39] Q. Shi, M. Liu, S. Li, X. Liu, F. Wang, and L. Zhang, "A deeply supervised attention metric-based network and an open aerial image dataset for remote sensing change detection," IEEE Trans. Geosci. Remote Sens., vol. 60, pp. 116, 2021.   
[40] H. Chen and Z. Shi, "A spatial-temporal attention-based method and a new dataset for remote sensing image change detection," Remote Sens., vol. 12, no. 10, p. 1662, 2020.   
[41] S. Antol, A. Agrawal, J. Lu, M. Mitchell, D. Batra, C. L. Zitnick, and D. Parikh, "Vqa: Visual question answering," in Proc. Int. Conf. Comput. Vis., 2015, pp. 24252433.   
[42] Z. Zhang, L. Jiao, L. Li, X. Liu, P. Chen, F. Liu, Y. Li, and Z. Guo, "A spatial hierarchical reasoning network for remote sensing visual question answering," IEEE Trans. Geosci. Remote Sens., vol. 61, pp. 115, 2023.   
[43] J. Wang, Z. Zheng, Z. Chen, A. Ma, and Y. Zhong, "Earthvqa: Towards queryable earth via relational reasoning-based remote sensing visual question answering," in Proc. AAAI Conf. Artif. Intell, vol. 38, no. 6, 2024, pp. 54815489.   
[4] C.Chappuis, V. Zermatten, S. Lobry, B. Le Saux, and D. Tuia, "Promptrsvqa: Prompting visual context to a language model for remote sensing visual question answering," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2022, pp. 13721381.   
[45] C. Liu, J. Zhang, K. Chen, M. Wang, Z. Zou, and Z. Shi, "Remote sensing temporal vision-language models: A comprehensive survey," arXiv preprint arXiv:2412.02573, 2024.   
[46] Z. Yuan, L. Mou, and X. X. Zhu, "Change-aware visual question answering," in Proc. IEEE Int. Geosci. Remote Sens. Symp. IEEE, 2022, pp. 227230.   
. .H B pre-training for unified vision-language understanding and generation," in Proc. Int. Conf. Mach. Learn. PMLR, 2022, pp. 12 88812900.   
[48] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds et al., "Flamingo: a visual language model for few-shot learning," Proc. Adv. Neural Inf. Process. Syst., vol. 35, pp. 2371623736, 2022.   
[49] A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes, A. Radford et al., "Gpt-4o system card," arXiv preprint arXiv:2410.21276, 2024.   
[50] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, XL Wag . t l "we Enhavisin-a model's perception of the world at any resolution," arXiv preprint arXiv:2409.12191, 2024.   
[51] T. GLM, A. Zeng, B. Xu, B. Wang, C. Zhang, D. Yin, D. Zhang, D. Rojas, G. Feng, H. Zhao et al., "Chatglm: A family of large language models from glm-130b to glm-4 all tools," arXiv preprint arXiv:2406.12793, 2024.   
[52] G. Team, P. Georgiev, V. I. Lei, R. Burnell, L. Bai, A. Gulati, G. Tanzer, D. Vincent, Z. Pan, S. Wang et al., "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context," arXiv preprint arXiv:2403.05530, 2024.   
[53] Z. Zhang, T. Zhao, Y. Guo, and J. Yin, "RS5M and GeoRSCLIP: A large scale vision-language dataset and a large vision-language model for remote sensing," IEEE Trans. Geosci. Remote Sens., 2024.   
[54] W. Dai, J. Li, D. Li, A. Tiong, J. Zhao, W. Wang, B. Li, P. N. Fung, and S. Hoi, "Instructblip: Towards general-purpose vision-language models with instruction tuning," Proc. Adv. Neural Inf. Process. Syst., vol. 36, pp. 49 25049 267, 2023.   
[5] . Zhan, Z. Xing, and Y. Yuan, "Skyeye: Unifyig remot sesig vision-language tasks via instruction tuning with large language model," ISPR J. Photogramm. Remote Sens., vol. 221, pp. 6477, 2025.   
[56] P. Deng, W. Zhou, and H. Wu, "Changechat: An interactive model for remote sensing change analysis via multimodal instruction tuning," in ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2025, pp. 15.   
[57] C. Liu, K. Chen, H. Zhang, Z. Qi, Z. Zou, and Z. Shi, "Change-agent: Towards interactive comprehensive remote sensing change interpretation and analysis," IEEE Trans. Geosci. Remote Sens., 2024.   
[58] Y. Fang, W. Wang, B. Xie et al., "EVA: Exploring the limits of masked l    al   . Vis. Pattern Recognit., 2023, pp. 19 35819 369.   
[59] K. Cho, B. van Merrienboer, Çaglar Gülehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, "Learning phrase representations using rnn encoderdecoder for statistical machine translation," in Conf. Empir. Methods Nat. Lang. Process., 2014.   
[60] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. Xing et al., "Judging llm-as-a-judge with mt-bench and chatbot arena," Proc. Adv. Neural Inf. Process. Syst., vol. 36, pp. 46 595 46623, 2023.   
[61] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar et al., L Ope i gols,"X arXiv:2302.13971, 2023.   
[62] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," arXiv preprint arXiv:1711.05101, 2017.   
[63] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, "BLEU: a method for automatic evaluation of machine translation," in Proc. 40th Annu. Meet. Assoc. Comput. Linguist., 2002, pp. 311318.   
[64] S. Banerjee and A. Lavie, "METEOR: An automatic metric for mt evaluation with improved correlation with human judgments," in Proc. ACL Workshop Intrinsic Extrinsic Eval. Mach. Transl. Summ., 2005, pp. 6572.   
[65] C.-Y. Lin, "Rouge: A package for automatic evaluation of summaries," in Proc. Workshop Text Summ. Branches Out, 2004, pp. 7481.   
[66] R. Vedantam, C. L. Zitnick, and D. Parikh, "CIDEr: Consensus-based image description evaluation," Proc. IEEE Conf. Comput. Vis. Pattern Recognit., p. 45664575, 2014.