# OmniVTLA：视觉-触觉-语言-动作模型与语义对齐的触觉感知

Zhenxue Cheng1Yiqian Zhang，Wenkang Zhan，Haoyu Li，Keyu Wang，Liong，Hengdi Za 1Paxini科技公司。2上海交通大学 †通讯作者

近期的视觉-语言-动作（VLA）模型建立在视觉-语言基础之上，取得了令人鼓舞的成果，并展现了在机器人操作中的任务泛化潜力。然而，由于触觉传感器的异质性和获取触觉数据的难度，目前的VLA模型显著忽视了触觉感知的重要性，并在接触丰富的任务中失败。为了解决这一问题，本文提出了OmniVTLA，这是一种涉及触觉感知的新型架构。具体而言，我们的贡献主要有三点。首先，OmniVTLA具备一个双路径触觉编码器框架。该框架通过使用预训练的视觉变换器（ViT）和语义对齐的触觉ViT（SA-ViT），增强了不同视觉基础和基于力的触觉传感器之间的触觉感知。其次，我们引入了ObjTac，这是一个全面的基于力的触觉数据集，捕捉了56种物体在10个类别中的文本、视觉和触觉信息。ObjTac包含135K三模态样本，补充了现有的视觉触觉数据集。第三，利用该数据集，我们训练了一个语义对齐的触觉编码器，以学习统一的触觉表示，为OmniVTLA提供更好的初始化。真实世界的实验表明，相较于最先进的VLA基线，取得了显著的改进，在抓取任务中使用抓手的成功率达到了96.9%（比基线高21.9%），而灵巧手的成功率达到了100%（比基线高6.2%）。此外，与现有的VLA相比，OmniVTLA显著缩短了任务完成时间，并通过触觉感知生成了更平滑的轨迹。我们的ObjTac数据集可以在 https://readerek.github.io/Objtac.github.io 找到。日期：2025年8月25日

# 1 引言

触觉感知对于人类的灵巧性至关重要，使得从穿针到处理易碎物品等复杂任务能够以显著的精确性和适应性完成。尽管视觉提供了全局的空间上下文，但触觉感知则提供了互补的优势：直接测量接触动态（例如，压力分布、纹理、坚韧性、视觉线索和低频反馈控制（Dahiya等，2009））。这一生物证据突显了视觉-触觉整合在需要物理交互的复杂操作任务中的关键作用。在机器人领域，视觉与触觉感知的整合已经成为增强操作能力的一个有前景的方向（Cui和Trinkle，2021）。早期的研究（Calandra等，2018；Li等，2018；Qi等，2023；Huan等，2024）专注于结合视觉和触觉特征的小规模模型，用于特定任务，如滑动检测或抓握稳定性预测。尽管这些方法展示了多模态感知的价值，但它们的范围有限，往往针对狭窄的应用，缺乏在多样化场景中的推广性。

近期在视觉-语言-动作（VLA）模型（Brohan 等，2023a；Kim 等，2024a；Black 等，2024；Team 等，2025）方面的进展彻底改变了机器人操控。这些模型利用大规模预训练的视觉-语言模型（VLMs）（Liu 等，2023；Li 等，2024；Zhang 等，2025a；Bai 等，2025）来理解自然语言指令和视觉观察，展现出良好的泛化和智能潜力。然而，这些模型主要依赖视觉和语言，忽视了触觉传感提供的丰富语义和物理反馈。现有尝试（Zhang 等，2025b；Huang 等，2025；Yu 等，2025）将触觉融合到 VLA 框架中，通常将触觉数据视为低层信号，未能在语义上与视觉和语言上下文对齐。

![](images/1.jpg)  

Fu1 左侧：基础 VLA 模型，其中图像编码器通常继承经过对比学习训练的预训练 CLIP/SigLIP 主干网络，以实现潜在空间语义分割。右侧：VTLA 模型。重要的是，设计针对语义理解的多模态系统，在视觉、语言和触觉模态方面极少被满足。

为了弥补这一差距，我们提出了 OmniVTLA（视觉-触觉-语言-动作模型），这是一种将视觉、触觉与语言统一到共享语义空间中的新颖架构，如图1所示。VTLA 借助对比学习将高分辨率触觉信号与视觉和语言概念对齐，使机器人能够在视听上下文中“理解”它们所感受到的内容。具体来说，我们为触觉数据引入了一条双编码器路径，以解决异构性问题，利用预训练的视觉变换器（ViT）和语义对齐的触觉 ViT（SA-ViT）。其次，我们构建了 ObjTac，一个综合性数据集，捕捉了56个对象、10个类别的文本视觉和基于力的触觉数据，总共135K三模态样本。第三，我们使用跨传感器数据训练了一个语义对齐的触觉编码器，以学习统一的触觉表示，为 OmniVTLA 提供更好的初始化。大量实验证明 VTLA 超越了 VLA 基线。在抓取和放置任务中，VTLA 的成功率提高了 $2 1 . 9 \%$，达到 $9 6 . 9 \%$，在灵巧手上提高成功率 $6 . 2 \%$，达到 $1 0 0 \%$。此外，它生成的轨迹更平滑，遵循“在清晰时快速移动，只有在接触靠近时减速”的直观原则。我们的贡献总结如下：我们提出了 OmniVTLA，一个建模视觉、触觉和语言的全端接触丰富操作任务的新框架。OmniVTLA 利用双编码器路径克服不同触觉传感器的异构性。• 我们引入了 ObjTac，一个综合性的触觉数据集，为56个对象收集了135K三模态样本。基于此，我们为 OmniVTLA 训练了一个语义对齐的触觉编码器。 • 现实世界实验表明 OmniVTLA 相比典型的 VLA 模型具有更优的性能，成功率提高了多达 $2 1 . 9 \%$。此外，它减少了完成时间，并使生成的轨迹更平滑。

# 2 相关工作

我们提出的 VTLA 与其他 VLA 模型的区别汇总于表 1。

触觉传感器感知任务。早期的研究主要集中在处理物理信号（例如，力、振动、变形）以完成特定的感知任务，如抓取稳定性预测（Calandra et al 2018，Cui et al 2020）和滑动检测（Li et al. 2018）。相关工作致力于学习通用的触觉表示，以实现跨任务、传感器和模态的可转移性。通过数据集构建（Fu et al. 2024；Cheng et al. 2025）、共享嵌入空间（Yang et al. 2024）、可转移架构（Zhao et al. 2024）和统一建模框架（Feng et al. 2025），这些研究展示了跨模态对齐和可泛化表示在触觉感知中的重要性。尽管这些方法提高了触觉感知能力，但它们与行动策略生成仍然脱钩，限制了其在实时机器人控制中的适用性。此外，大多数现有工作利用基于视觉的触觉数据（如GelSight（Yuan et al. 2017；Johnson and Adelson, 2009），但在很大程度上忽视了基于力的触觉数据，而后者在机器人策略学习中也被广泛使用。表1 不同VLA模型的比较。L：语言；V：视觉；T：触觉；A：行动。

<table><tr><td>Model Type</td><td>Methods</td><td>Input</td><td></td><td>Output Semantic-Aligned</td></tr><tr><td>VA</td><td>Diffusion Policy (Chi et al., 2023)</td><td>V</td><td>A</td><td>✓</td></tr><tr><td>VTA</td><td>RDP (Xue et al., 2025)</td><td>V + T</td><td>A</td><td>X</td></tr><tr><td>VLA</td><td>OpenVLA (Kim et al., 2024a), π0(Black et al., 2024)</td><td>V + L</td><td>A</td><td>✓</td></tr><tr><td>TLA</td><td>TLA (Hao et al., 2025)</td><td>T + L</td><td>A</td><td>X</td></tr><tr><td>VTLA</td><td>VTLA (Zhang et al., 2025b), Tactile-VLA (Huang et al., 2025)</td><td>V + T +L</td><td>A</td><td>X</td></tr><tr><td>OmniVTLA</td><td>Ours</td><td>V + T + L</td><td>A</td><td>√</td></tr></table>

视觉-触觉融合在操作中的应用。最近在视觉-触觉策略学习方面的进展已经展示了在接触丰富的操控中的显著进展。强化学习框架有效地结合了视觉和触觉输入，用于组装任务（Lee et al., 2020；Hansen et al., 2022）和灵巧的手内部操控（Hu et al., 2025）。近年来，该领域越来越多地采用模仿学习范式（u al 2023；Lin et al 2024；Huan et al, 2024；Xue et al 202；Liu et al 2025），探索细粒度操控的视觉-触觉表征和系统架构。尽管这些方法在特定任务性能上取得了令人印象深刻的成果，但与视觉-语言-行动模型相比，它们在语义推理和概括能力方面仍然有限，这仍然是我们希望通过视觉-触觉语义融合来解决的一个巨大差距。

视觉-语言-动作模型。视觉语言动作（VLA）模型已成为通用机器人策略的一种强大范式。Brohan 等人（2023b）在这一方向上开创性地将机器人动作表示为语言词元，使知识可以从网络规模的预训练中转移。Kim 等人（2024b）通过 LoRA 微调提供了一种开源替代方案，以实现高效迁移。后续工作（Team 等，2024；Black 等，202Bjorcka 025exan）的研究集中在基于动作生成的能力（Chi 等，2023）。可扩展性方面的努力（Wen 等，2025；Team 等，2025；Shukor 等，2025）以及推理机制（Zhao 等，2025；Lin 等，2025）和 3D 扩展（Zhen 等，2024；Qu 等，2025）进一步增强了适用性。尽管 VLA 模型在开放世界泛化方面表现出色，但其对视觉和语言的单一依赖在需要精确物理交互的接触丰富任务中限制了性能。新兴的触觉增强方法通过基于语言的传感器融合（Jones 等，2025）、涉及触觉的 VLA 学习（Hao 等，2025；Zhang 等，2025b）和低维力感知控制（Huang 等，2025；Yu 等，2025）解决了这些限制。然而，这些方法尚未充分探讨触觉编码器的设计。我们的 OmniVTLA 框架通过建立双编码器路径以适应触觉，基于统一的跨模态表示学习，从根本上推动了这一范式的发展。

# 3 种方法

# 3.1 问题表述

正式地，行动模型的目标是建模分布 $p ( \mathbf { A } _ { t } | \mathbf { o } _ { t } )$，其中 $\mathbf { A } _ { t } = \left\{ a _ { t } , a _ { t + 1 } , \dotsc , a _ { t + H - 1 } \right\}$ 表示相应的动作序列（$\mathrm { H }$ 是块长度），而 $\mathbf { o } _ { t }$ 表示当前时刻的观察值。对于典型的 VLA 模型，观察值由几张 RGB 图像、一个语言提示以及机器人本体状态组成，则该模型可以正式表示为：

$$
O _ { t } = \mathbf { M } _ { \mathrm { V L A } } \big ( \mathbf { A } _ { t } \mid f _ { \phi } ( \mathbf { I } _ { t } ^ { i } ) , l _ { t } \big ) ,
$$

![](images/2.jpg)  
a heeeybt viulnact ata  we s tacorsTherst i v pre-trained visual encoder to inherit rich semantic representations from large-scale image data.The second ViT (SATv an textualmodalits.Thisdualencoder desienablesfecivenowleeranseranconsistent reprati learning across diverse sensory inputs.

其中 $\mathbf { I } _ { t } ^ { i }$ 表示第 $i^{\mathrm{th}}$ 图像，例如第三视图图像和机器人手腕图像，$l _ { t }$ 是一系列语言词元。通常，图像 $\mathbf { I } _ { t } ^ { i }$ 使用基于视觉变换器（Vision Transformers, ViT）的对比图像编码器 $f _ { \phi }$（例如 CLIP, SigLIP）进行编码，然后投影到与文本词元对应的潜在嵌入空间。同时，我们的研究目标是将 TLA 模型的战术数据纳入 NPU，如图 2 所示。VTLA 模型的表达如下。

$$
o _ { t } = \mathbf { M } _ { \mathrm { V T L A } } \big ( \mathbf { A } _ { t } \ \lvert \ f _ { \phi } ( \mathbf { I } _ { t } ^ { i } ) , f _ { \theta } ( \mathbf { T } _ { t } ^ { j } ) , l _ { t } \big ) ,
$$

其中 $\mathbf{T}_{t}^{j}$ 表示第 $j$ 个触觉数据，例如附加在两指夹持器上的触觉传感器或灵巧手的多个手指和手掌。$f_{\phi}$ 表示触觉编码器。从直观上讲，触觉数据可以重新映射为张量，并利用类似 ViT 的结构作为图像编码器进行编码。但在本研究中，我探讨了不同的触觉编码器及其相应的训练策略，以验证 VTLA 的最佳架构。

# 3.2 具有双编码器路径的整体架构

所提出的 OmniVTLA，如图 2 所示，基于 $\pi 0$（Black et al., 2024）构建。它由三个核心组件组成：分词器、主干网络和动作头。分词器处理：1) 通过 PaliGemma 分词器处理语言指令 $l _ { t }$（词汇量：257,152），2) 使用 SigLiP 模型处理图像观测 $\mathbf { I } t ^ { i }$（Zhai et al., 2023），3) 处理触觉观测 $\mathbf { T } t ^ { j }$，将所有模态投影为潜在词元。具体来说，对于包括第三视角和手腕的图像，我们将原始捕获图像调整为 $224 \times 224$，每幅图像产生 256 个词元。对于触觉数据，我们将数据范围归一化到 int8，并将多传感器输入拼接成单幅图像，然后通过 ViT 类编码器处理调整到 $224 \times 224$ 的输入，以生成 256 个词元。Gemma-2B 主干网络处理连接的词元以生成动作词元，这些词元通过经过训练的动作头解码，使用与 $\pi 0$ 相一致的流匹配损失。动作表示根据末端执行器的不同而变化。对于双指夹持器，表示为 10 个词元（3 个相对位置，6 个相对角度，1 个夹持器状态）。对于四指手，表示为 25 个词元（3 个相对位置，6 个相对角度，16 个绝对关节位置）。现有工作充分解决了触觉编码器的设计，主要由于两种形式的异质性：（1）触觉与视觉数据之间，以及（2）不同触觉传感器之间（如图 2 左上部分所示）。这一挑战因触觉数据集的不同特征而加剧，例如 Touch and Go (TAG)（Yang et al., 2022）、SSVTP（Kerr et al., 2023）、ObjectFolder（Gao et al., 2021），使得统一编码器设计变得复杂。因此，四种不同的触觉编码器值得探索，详细结果将在第 4.2 节中讨论。

![](images/3.jpg)  
Fo data pairoachieve semantic-level algnment. We alsovisualize par of tactil images after the normalization.

LA-FS：触觉编码器从头开始训练，仅依赖有限的远程操作触觉数据。 VLA-re：触觉编码器从大规模数据集中的预训练视觉编码器初始化，并在少量远程操作数据上进行微调。 VTLA-SA：触觉编码器首先通过跨模态对比学习进行训练，以实现语义层级对齐（第3.3节），然后在少量数据上进行调优。 OmniVTLA：双编码器路径，其中一条路径为VTLA-Pre，另一条路径为VTLA-SA。触觉异质性源于不同的传感原理：视觉触觉传感器（例如GelSight (Yuan, 2017; Johnson and Adelson, 2009)）捕捉表面几何形状，而其他传感器（例如Paxini Gen2 (Paxini, 2025)）测量力。值得注意的是，视觉触觉传感器通常捕捉更高的空间分辨率，但时间分辨率较低，最多可达30Hz；而基于力的传感器虽然空间分辨率较低，但可以捕捉更高的时间分辨率，以更好地表征事件。因此，基于力的传感器可以更好地补充视觉模态。为了应对不同触觉传感器的异质性，我们提出了双ViT编码器，其连接的词元实现了跨传感器理解，作为我们提出的OmniVTLA模型的触觉编码器。

# 3.3 语义对齐触觉编码器

现有研究（Fen等人，025）探讨了统一的视觉-触觉传感器表征，但仍然无法推广到基于力的触觉感知。如表2所示，预训练的AnyTouch编码器在基于力的数据集中仅实现了$4 0 . 2 1 \%$的材料分类准确率，表明跨传感器迁移存在严重的局限性。为了应对这一问题，我们收集了一个带有对齐文本的视频触觉感知数据集，命名为TacVis。该数据集包含10种物体类型（即塑料、玻璃、木材、砖块、金属、织物、皮革、陶瓷、纸张及其他），按表面粗糙度（粗糙与光滑）和材料硬度（刚性与柔软）进行分类。我们的

<table><tr><td rowspan="2">Method</td><td rowspan="2">Our data in training set</td><td colspan="3">Touch and Go</td><td colspan="3">Our Collected Dataset</td></tr><tr><td>Material</td><td>Roughness</td><td>Hardness</td><td>Material</td><td>Roughness</td><td>Hardness</td></tr><tr><td>AnyTouch</td><td>X</td><td>79.39</td><td>86.32</td><td>95.16</td><td>40.21</td><td>68.01</td><td>90.11</td></tr><tr><td>SA-ViT (Ours)</td><td>✓</td><td>74.90</td><td>85.46</td><td>92.10</td><td>70.44</td><td>82.21</td><td>93.91</td></tr></table>

Tabvalati 是一个基于模型的技术，用于支持分类任务的头部。

![](images/4.jpg)  

左图：实验设置的硬件和环境。右图：涉及操作任务的物体。收集的数据集将很快发布。数据收集和处理流程如下所述。对于每个物体，我们进行了5次交互试验，每次试验持续1060秒（以$60 ~ \mathrm { H z }$的频率采样）。这产生了270,000条力数据记录。我们还以720P分辨率和30帧每秒的速度捕捉第一人称视角的视觉记录，结果生成了252个视频序列，平均时长为18秒。总的来说，我们收集了135K个样本，配有触觉和视觉数据。2）我们为语言模态添加了物体级注释，包括物体名称、材料类型、粗糙度类别、硬度类别、视频级元数据和文本描述。3）通过时间戳进行了时间同步，以对齐视觉和触觉模态。

为了训练更好的语义对齐编码器，我们将自己收集的数据集添加到现有数据集中，并采用AnyTouch（Fenget al., 2025）的第二阶段训练流程，以实现多模态和跨传感器对齐。由于我们的数据集包含三模态数据对，对于新增数据，我们直接使用总损失函数 $\begin{array} { r } { \mathcal { L } _ { a l i g n } = \alpha _ { V L } * \frac { \mathcal { L } _ { V L } + \mathcal { L } _ { T V } } { 2 } + \alpha _ { V T } * \frac { \mathcal { L } _ { V T } + \mathcal { L } _ { T V } } { 2 } + \alpha _ { T L } * \frac { \mathcal { L } _ { T L } + \mathcal { L } _ { L T } } { 2 } } \end{array}$ 其中 $\mathcal { L } _ { V L }$、$\alpha _ { V L }$、$\alpha _ { V T }$ 和 $\alpha _ { T L }$ 是超参数。此外，还将二进制交叉熵的跨传感器匹配损失添加到总损失中。通过整合我们的数据集ObjTac，这种语义对齐触觉编码器能够更好地适应已实现的触觉传感器和语义表示，并在基于力的触觉数据集上提高准确性，同时在基于视觉的触觉数据集Touch和Go中维持接近基线的表现。

# 4 实验

# 4.1 实验设置

基线和训练细节 我们将VTLA模型与两个模型进行了比较，其中Diffusion Policy (DP) (Chi et al., 2023)作为非VLM基线，$\pi 0$ (Black et al., 2024)作为VLA基线。我们按照代码库中指定的默认设置训练DP和$\pi 0$模型，但对于DP，我们将动作步长设置为64。对于我们的OmniVTLA模型，我们增加了触觉图像输入。更多训练细节可以在附录中找到。 实现与任务设置 我们的机器人系统包括一只UR5机械臂、一只配备两个触觉传感器和一个腕部摄像头的夹爪、一只配备腕部摄像头的11个触觉传感器的灵巧手，以及一个基座摄像头（见图4）。我们对四个物体（短罐、方形咖啡瓶、雨衣罐、牛奶盒）进行抓取和放置任务，使用夹爪完成其一，使用灵巧手完成两个物体（咖啡瓶和牛奶盒）（见图4），每个物体收集40个远程操作演示集，以$30 \ \mathrm{Hz}$的频率采集。塑料瓶和方形瓶作为未见物体用于泛化评估。我们将触觉数据处理为3通道图像表示，通过最大-最小力归一化，并重塑为3通道张量。

![](images/5.jpg)  
FOffevaliai resultf ifftmodes usi iffet bjet whereur proo OmiTLv the lowest MSE between predicted trajectory and GT trajectory.

<table><tr><td rowspan="2">Model</td><td colspan="3">Tactile Enc.</td><td colspan="4">SR (%) ↑</td><td rowspan="2"></td><td colspan="5">CT (step) ↓</td></tr><tr><td>FS</td><td>Pre</td><td>SA</td><td>Can</td><td>Bottle</td><td>Milk</td><td>Tin</td><td>Avg Can</td><td>Bottle</td><td>Milk</td><td>Tin</td><td>Avg</td></tr><tr><td>VLA (π0)</td><td>X</td><td></td><td></td><td>62.5</td><td>37.5</td><td>100</td><td>100</td><td>75.0</td><td>981</td><td>562</td><td>648</td><td>436</td><td>657</td></tr><tr><td>VTLA-FS</td><td>✓</td><td>×</td><td></td><td>75.0</td><td>50.0</td><td>100</td><td>100</td><td>81.2</td><td>677</td><td>549</td><td>498</td><td>423</td><td>537</td></tr><tr><td>VTLA-Pre</td><td>X</td><td>✓</td><td></td><td>62.5</td><td>75.0</td><td>100</td><td>100</td><td>84.4</td><td>847</td><td>526</td><td>540</td><td>429</td><td>586</td></tr><tr><td>VTLA-SA</td><td>X</td><td>X</td><td>2</td><td>87.5</td><td>62.5</td><td>100</td><td>100</td><td>87.5</td><td>524</td><td>553</td><td>455</td><td>405</td><td>484</td></tr><tr><td>OmniVTLA</td><td>X</td><td>✓</td><td></td><td>100</td><td>87.5</td><td>100</td><td>100</td><td>96.9</td><td>535</td><td>537</td><td>527</td><td>393</td><td>498</td></tr></table>

能够3 真实世界实验结果，使用两指夹持器对不同模型进行评估。基线 VLA 模型为 $\pi 0$。粗体字表示最佳性能， underline 字体表示第二最佳性能。

<table><tr><td rowspan="2">Model</td><td colspan="5">SR (%) ↑</td><td colspan="5">CT (step) ↓</td></tr><tr><td>Bottle</td><td>Milk</td><td>Plastic</td><td>Square†</td><td>Avg</td><td>Bottle</td><td>Milk</td><td>lastic</td><td>Square†</td><td>Avg</td></tr><tr><td>VLA (π0)</td><td>100</td><td>100</td><td>87.5</td><td>87.5</td><td>93.8</td><td>312</td><td>324</td><td>369</td><td>368</td><td>343</td></tr><tr><td>OmniVTLA</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>307</td><td>305</td><td>339</td><td>335</td><td>322</td></tr></table>

TabReal-真实世界实验结果表明，使用我们设计的扩展手部模型基础的VLA模型是$\pi 0$。方框表示方形咖啡瓶。加粗字体表示最佳性能。为了研究触觉反馈在任务执行中的作用，我们设计了一种多阶段抓取协议。与传统方法不同，我们的方法包含多达三次增量抓取尝试。具体来说，当抓取物体接近目标时，手部会逐步闭合，通过三个阶段，在第三次尝试中成功抓取。抓取后，手部将物体运输到预定目标位置。为了公平评估，我们使用网格地图标准化初始物体姿态，并在32次抓取尝试中测试每个模型，对于灵巧手则测试16次抓取尝试（每个初始状态在4个网格位置上进行2次试验）。每次试验的最大评估步骤设置为1500。

评估指标 我们通过两种互补的方法评估了我们的方法：离线验证和真实世界实验。在离线验证中，我们计算均方误差（MSE） $\begin{array} { r } { \mathrm { M S E } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \| x _ { t } - \hat { x } _ { t } \| ^ { 2 } } \end{array}$，其中 $T$ 表示总时间步，$x _ { t }$（真实值）和 $\hat { x } _ { t }$（预测值）表示 10 维或 25 维状态向量，包括末端执行器位置（xyz）、6D旋转表示（Zhou et al., 2018），以及 1 个夹持器开口或 16 个灵巧手的绝对关节。对于真实世界评估，我们采用了三个指标：（1）成功率（SR），衡量在结束时间戳时成功放置物体的比例；（2）完成时间（CT），从任务启动到成功放置物体并夹持器打开的时间；（3）运动平滑度，计算末端执行器沿轨迹的运动方差。

<table><tr><td rowspan="2">Model</td><td rowspan="2">Tactile Enc.</td><td colspan="5">SR (%) ↑</td><td colspan="5">CT (step) ↓</td></tr><tr><td>Can</td><td>Bottle</td><td>Milk</td><td>Tin</td><td>Avg.</td><td>Can</td><td>Bottle</td><td>Milk</td><td>Tin</td><td>Avg.</td></tr><tr><td>VA (DP)</td><td></td><td>75.0</td><td>75.0</td><td>50.0</td><td>37.5</td><td>59.4</td><td>767</td><td>989</td><td>1010</td><td>638</td><td>851</td></tr><tr><td>VTA (Ours)</td><td>×</td><td>100</td><td>75.0</td><td>75.0</td><td>62.5</td><td>78.1</td><td>695</td><td>658</td><td>783</td><td>593</td><td>682</td></tr></table>

TabRealworpeal sulhepac i tac usi twfin. 基准模型为 DP Chi 等（2023），所有参数均从头开始训练。粗体字表示最佳性能。

<table><tr><td>Model</td><td colspan="3">Tactile Enc.</td><td colspan="5">Smoothness (×10−4) ↓</td></tr><tr><td></td><td>FS</td><td>Pre</td><td>SA</td><td>Can</td><td>Bottle</td><td>Milk</td><td>Tin</td><td>Avg</td></tr><tr><td>VLA (π0)</td><td>X</td><td>X</td><td>X</td><td>29.3</td><td>0.78</td><td>6.24</td><td>1.95</td><td>9.57</td></tr><tr><td>VTLA-FS</td><td>✓</td><td>X</td><td>X</td><td>2.57</td><td>0.69</td><td>1.54</td><td>1.69</td><td>1.62</td></tr><tr><td>VTLA-Pre</td><td>X</td><td>✓</td><td>X</td><td>1.95</td><td>0.97</td><td>5.09</td><td>2.63</td><td>2.66</td></tr><tr><td>VTLA-SA</td><td>X</td><td>X</td><td>✓</td><td>1.12</td><td>0.45</td><td>0.92</td><td>1.68</td><td>1.04</td></tr><tr><td>OmniVTLA</td><td>X</td><td>✓</td><td>✓</td><td>1.33</td><td>1.37</td><td>1.90</td><td>1.22</td><td>1.46</td></tr></table>

表6 使用触觉编码器生成的轨迹的平滑度，针对模型 $\pi 0$ 。为了进行公正比较，进行了三种设置，其中 "From Scratch" 表示从头开始训练，"Pre" 表示预训练模型；"S" 代表我们提出的语义对齐触觉编码器。

# 4.2 评估结果

离线验证结果和基于遥操作的验证数据展示了 OmniVTLA 在不同物体上的优越预测性能。如图 5 所示，OmniVTLA 在所有模型中实现了最低的均方误差（MSE），其平均值为 $1.40 \times 10^{-4}$。这一趋势在大多数物体上均得以保持：对于短罐，OmniVTLA 相比于 VLA 将 MSE 降低了 $7.8\%$；对于瓶子，降低幅度达到了 $23.3\%$。VTLA-FS 的异常结果可能源于过拟合，强调了使用大规模触觉数据的重要性，而不仅仅依赖于遥操作驱动的数据。结果强调了语义对齐（SA）触觉编码器有效整合触觉信号与视觉及语言线索，从而实现更准确的状态预测——这是精确操作所必需的。

真实世界结果 真实世界实验验证了OmniVTLA在抓取和放置任务中优于$\pi 0$和DP的性能。对于使用夹爪的$\pi 0$（见表3），VTLA-SA在使用最多一个触觉解码器的情况下表现优于其他设计，达到了$87.5\%$的平均成功率（SR），比从零开始（FS）编码器高出6.3%，比预训练（Pre）编码器高出$3.1\%$。当将Pre和SA编码器结合在提出的OmniVTLA中时，获得了Outstanding的$96.9\%$的平均成功率，展示了双重触觉解码器设计的优越性。在完成时间（CT）方面，SA编码器将平均步骤数减少了$26.3\%$，相较于VLA基线（从657步减少到484步），证明了触觉反馈优化了操作。我们提出的OmniVTLA实现了第二好的性能，将完成时间减少了$24.2\%$（从657步减少到498步）。对于使用四指灵巧手的$\pi 0$（见表4），我们的OmniVTLA将成功率提高了6.2%（从$93.8\%$增加到$100\%$），将完成时间减少了$6\%$（从343步减少到322步）。特别是对于未见物体Plastic和Square，我们的成功率达到了$100\%$，而VLA仅达到了$87.5\%$。对于DP基线（见表5），结合触觉感知使平均成功率提高了18.7%（从$59.4\%$增加到$78.1\%$），平均完成时间减少了$19.9\%$（从851步减少到682步）。这确认了触觉信号普遍增强了性能，无论基础是什么。 轨迹的平滑度 触觉感知显著提高了运动的平滑度，如表6所量化。SA编码器实现了最低的平均平滑度指标$(1.04 \times 10^{-4})$，比VLA基线低$89.6\%$。这与“清晰时快动，接触时慢下”的直观原理一致。语义对齐的触觉反馈使得机器人能够更智能、更微妙地调整夹爪的动作，减少完成时间，同时避免在接触过程中出现突兀的运动——这对于处理易碎物体至关重要。

![](images/6.jpg)  
Figure6 Visualization of several failed cases for VLA, VTLA-FS, VTLA-Pre, VTLA-SA due toinsufficient contac awareneo contac gulcntact, andurproos OmiTLA hivesuul raspianble contact owing to full tactile sensing.

定性结果 为了理解触觉感知的有效性，我们展示了一些现实世界实验的定性结果。语言提示为“拾起短罐并将其移动到盘子上”，我们可视化了VLA、VTLA-Pre 和 OmniVTLA 模型的失败或成功案例（图6）。VLA 模型由于缺乏足够的触觉反馈经常无法抓取物体，而 VTLA-Pre 在没有成功提起的情况下持续进行手势调整。相比之下，OmniVTLA 利用语义触觉提示来稳定抓握并执行预期轨迹，如成功提起短罐的案例所示，使用了抓手和瓶身在灵巧手中。

# 5 结论与未来工作

我们提出了 OmniVTLA，一种新颖的视觉-触觉-语言-动作模型，并提出了一种具有视觉和语言模态的语义对齐触觉编码器。我们提出了一种双编码器路径，以解决触觉数据的异质性。此外，我们引入了 ObjTac 数据集，用于跨模态对比学习框架，使机器人能够在任务相关的上下文中解释触觉数据。实验结果表明，与最先进的 VLA 基线相比，成功率提高了 21.9%（使用两指夹持器）和 6.2%（使用四指灵巧手）。此外，OmniVTLA 将完成时间减少了约 24.2%，并通过触觉引导学习实现了更平滑的轨迹。尽管目前评估的任务和机器人仍然有限，但我们的 OmniVTLA 为触觉感知的机器人操作奠定了重要基础。未来的工作将探索更复杂的任务、更高效的触觉表示以及时间动态融合架构。

参考文献 Shuai Bai, Keqin Chen, Xuejing Liu, Jialn Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shije Wang, Jun Tang 等. Qwen2. 5-vl 技术报告. arXiv 预印本 arXiv:2502.13923, 2025. Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang 等. Gr00t n1: 一种用于通用人形机器人的开放基础模型. arXiv 预印本 arXiv:2503.14734, 2025. Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter 等, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, 和 Ury Zhilinsky. $\pi _ { 0 }$ : 一种用于通用机器人控制的视觉-语言-行动流模型, 2024. https://arxiv.org/abs/2410.24164. Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn 等. Rt-: 视觉-语言-行动模型转移网络知识以进行机器人控制. arXiv 预印本 arXiv:2307.15818, 2023a. Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn 等. Rt- 视觉-语言-行动模型转移网络知识以进行机器人控制. arXiv 预印本 arXiv:2307.15818, 2023b.

罗伯托·卡兰德拉，安德鲁·欧文斯，迪内什·贾亚拉曼，贾斯廷·林，袁文珍，吉滕德拉·马利克，爱德华·哈德尔森，谢尔盖·莱维。 《利用视觉和触觉进行抓取与重新抓取》。IEEE机器人与自动化快报, 3(4): 3300-3307, 2018。 宁诚，徐济南，管昌浩，高晶，王韦豪，李优，孟繁东，周杰，方斌，和韩文娟。 《Touc100k：一个大规模触觉语言视觉数据集用于触觉中心的多模态表示》。信息融合, 页码 103305, 2025。 陈情，许振熙，邢松，埃里克，尤李，班杰明，巴赫，恩·谢尔氏。 《扩散策略：通过动作扩散的视觉-运动策略学习》。国际机器人研究杂志, 页码 02783649241273668, 2023。 崔金达和杰夫·特林克。 《迈向下一代学习机器人操作》。科学机器人, 6(5): eab946, 2021。 崔少伟，王睿，魏军航，李繁荣，和王硕。 《可变形物体的抓取状态评估使用视觉-触觉感知》。在2020 IEEE国际机器人与自动化会议（ICRA）上，页码 538-544。IEEE, 2020。 拉文德·达希亚，乔治·梅塔，毛里齐奥·瓦莱，和朱利奥·桑迪尼。 《触觉传感：从人类到类人机器人》。IEEE机器人学报, 26(1): 120, 2009。 阿列克斯·多索维茨基，卢卡斯·贝耶，亚历山大·科列斯尼科夫，迪尔克·魏塞博尔，白晓华，托马斯·温特纳，莫斯塔法·德赫加尼，马蒂亚斯·敏德尔，乔治·海戈尔德，西尔万·杰利，等人。 《一张图值得 16×16 个单词：大规模图像识别的变换器》。arXiv 预印本 arXiv:2010.11929, 2020。 傅如，胡建宇，夏文科，高天梭，阿申，孙宇，方斌，和胡迪。 《Ayt：跨多个视觉-触觉传感器学习统一状态动态表示》。arXiv 预印本 arXiv:2502.12191, 2025。 傅乐霖，达塔·高拉，关黄炫，威尔·钟，杰米·德雷克，何塞·罗德里戈，穆萨·穆卡姆，麦克·兰贝塔，罗伯托·卡兰德拉，和肯·戈德堡。 《一个包含触觉、视觉和语言的数据集用于多模态对齐》。arXiv 预印本 arXiv:2402.13232, 2024。 阮高燕，肖维·马，李飞飞，和吴佳钧。 《从视觉、听觉和触觉表示学习的状态机模型》，2021。https://arxiv.org/abs/2109.07991。 约瑟夫·汉斯，弗朗西斯·霍根，德米特里·里夫金，大卫·摩根，迈克尔·詹金，和格雷戈里·杜德克。 《视觉-触觉增强学习：利用深度强化学习学习多模态操作策略》。在2022国际机器人与自动化会议（ICRA）上，页码 8298-8304。IEEE, 2022。 彭浩，张超凡，李丁哲，曹晓戈，郝小帅，崔少伟，和王硕。 《TLA：接触丰富操作的触觉语言-行动模型》。arXiv 预印本 arXiv:2503.08548, 2025。 胡文斌，黄必丹，李王维，杨思成，郑宇，和李志斌。 《触Dexterous in-hand 操作通过深度强化学习与触觉感知》。机器人与自动化系统，186: 104904, 2025。 黄斌，王意勋，英欣怡，罗意月，和李云哲。 《3D-ITA：采用视觉-触觉感知的学习抓取操作》。arXiv 预印本 arXiv:2410.24091, 2024。 贾华，舒王，范琦·李，韩昊，樊琛文，和杨高。 《触觉：解锁图像行动模型的物理知识以实现触觉泛化》。arXiv 预印本 arXiv:2507.09160, 2025。 米迦·K·约翰逊和爱德华·哈德尔森。 《用于表面纹理和形状测量的反图形感知》。在2009 IEEE计算机视觉与模式识别会议上，页码 1070-1077, 2009。doi: 10.1109/CVPR.2009.5206534。 乔舒亚·琼斯，奥耶·梅斯，卡梅尔·塞拉扎，凯尔·斯塔霍维茨，皮特·阿贝尔，和谢尔盖·莱维。 《超出视线：通过语言对接的异构传感器的一般ist机器人策略》。arXiv 预印本 arXiv:2501.069, 2025。 贾斯廷·凯尔，黄黄，阿尔伯特·威尔科克斯，瑞安·霍克，杰弗里·伊赫诺夫斯基，罗伯托·卡兰德拉，和肯·戈德堡。 《超级视角。》。arXiv 预印本 arXiv:213042。 金穆，卡尔·佩尔奇，西达特·卡拉姆切提，泰德·肖，阿什温·巴拉克里希纳，苏拉杰·奈尔，拉法尔·拉法伊洛维奇，伊桑·福特，格蕾丝·林，帕纳格·桑凯提，等人。 《OpenVLA：一个开源的视觉-语言-行动模型》。arXiv 预印本 arXiv:2406.09246, 2024a。 金穆，卡尔·佩尔奇，西达特·卡拉姆切提，泰德·肖，阿什温·巴拉克里希纳，苏拉杰·奈尔，拉法尔·拉法伊洛维奇，伊桑·福特，格蕾丝·林，帕纳格·桑凯提，等人。 《OpenVLA：一个开源的视觉-语言-行动模型》。arXiv 预印本 arXiv:2406.09246, 2024b。 米切尔·李，余克·朱，彼得·扎卡里，马修·谭，克里希南·辛维萨，西尔维奥·萨瓦雷，李飞飞，安妮梅·G 和珍·博·马克。 《视觉与触觉：学习多模态表示以应对复杂任务》。IEEE机器人学报, 36(3): 582-596, 2020。 李博，张宇涵，郭东，张任瑞，李峰，张浩，张开晨，张佩元，李彦伟，刘子维，等人。 《Llava-One Vision：轻松的视觉任务迁移》。arXiv 预印本 arXiv:2408.03326, 2024。 李建华，邢东，和爱德华·阿德尔森。 《通过结合触觉和视觉信息进行滑动检测》。在2018 IEEE国际机器人与自动化会议（ICRA）上，页码 7772-7777。IEEE, 2018。 林范琦，赖瑞超，胡影冬，尤家成，赵俊铭，和杨高。 《OntwoVLA：一个集成视觉语言-行动模型，具备自适应推理能力》。arXiv 预印本 arXiv:2505.11917, 2025。 托尔·林，张宇，克万，霍兹，布雷特·S·列维，和吉腾德拉·马利克。 《用两只多指手学习技能》。arXiv 预印本 arXiv:2404.16823, 2024。 刘芳晨，李川宇，秦宜华，安基特·肖，徐晶，皮特·阿贝尔，和陈睿。 《维生素：通过无源机器人视觉-触觉操作界面学习接触丰富任务》。arXiv 预印本 arXiv:2504.06156, 2025。 华李，涌_kwuan。 《QWuanYong Je Leisal 阶段处理系统》，36: 3489234916, 2023。 刘松铭，吴凌轩，李邦国，谎凯，陈华煜，郑宜，徐榕，苏航，和朱俊。 《RDT-1B：一个用于双手操作的扩散基础模型》。arXiv 预印本 arXiv:2410.07864, 2024。 Paxini。 《PX-6AX：触觉处理单元》，2025。https://paxini.com/ax/gen2。 贾海志，布伦特·依，苏达尔尚·苏雷什，麦克·兰贝塔，易马，罗伯托·卡兰德拉，和吉腾德拉·马利克。 《利用视触觉进行一般性手持物体旋转》。在机器人学习会议上，页码 2549-2564。PMLR, 2023。 邱德琳，宋浩明，陈齐志，姚元启，叶心怡，丁燕，王志刚，谷静，赵斌，王多等。 《空间VL：探索用于视觉-语言-行动模型的空间表示》。arXiv 预印本 arXiv:2501.15830, 2025。 阿列克·拉德福德，金钟旭，克里斯·哈拉西，阿迪提亚·拉梅什，加布里埃尔·戈赫，桑迪尼·阿尔瓦尔，吉里什·萨斯特里，阿曼达·阿斯克，帕梅拉·米什基，杰克·克拉克，等。 《在自然语言理解上学习具有可调性的视觉模型》。在国际机器学习会议上，页码 8748-8763。PMLR, 2021。 穆斯塔法·舒克，达娜·奥巴基罗娃，弗朗西斯科·卡普阿诺，佩平·库伊曼斯，史蒂文·帕尔马，阿迪尔·祖伊丁，米歇尔·阿尔拉金，卡罗琳·帕斯卡尔，马丁诺·鲁西，安德烈斯·马拉菲奥蒂，西蒙·阿利贝尔，马修·科尔，托马斯·沃尔夫，和雷米·卡登。 《SMOLVLA：一个负担得起且高效的视觉-语言-行动模型》。arXiv 预印本 arXiv:2506.01844, 2025。 双子机器人团队，萨闵达·阿贝鲁安，乔舒亚·安斯利，尚·巴普蒂斯特·阿莱拉克，蒙特塞拉特·冈萨雷斯·阿雷纳斯，特拉维斯·阿姆斯特朗，阿什温·巴拉克里希纳，罗伯特·巴鲁赫，玛利亚·鲍沙，米希尔·布洛克泽伊尔，等。 《双子机器人：将人工智能带入物理世界》。arXiv 预印本 arXiv:2503.20020, 2025。 八角模型团队，迪比亚·戈什，霍默·沃尔克，卡尔·佩尔奇，凯文·布莱克，奥耶·梅斯，苏迪普·达萨里，乔伊·海纳，托巴斯·克雷曼，查尔斯·徐，等。 《八角：一个开源通用机器人政策》。arXiv 预印本 arXiv:2405.12213, 2024。 文俊，朱宜晨，李吉妮，朱敏杰，唐志彬，吴琨，徐志云，刘宁，程然，沈璨等。 《TinyVLA：迈向快速、高效的数据视-语-行动模型用于机器人操作》。IEEE机器人与自动化快报，2025。 韩雪，任杰佳，陈文迪，张古，方源，顾颖，徐华哲，和陆册武。 《反应扩散策略：用于接触操作的视觉-触觉策略》。arXiv 预印本 arXiv:2503.0281, 2025。 方宇阳，马晨阳，张家程，朱晶，袁文珍，和安德鲁·欧文斯。 《触觉和出发：学习从人类收集的视觉和触觉信息》，2022。https://arxiv.org/abs/2211.12498。 冯宇阳，冯超，陈子扬，朴亨世，丹尼尔·王，杜一鸣，曾子耀，陈新，瑞特·甘戈巴迪亚，安德鲁·欧文斯，等。 《将触觉绑定到一切：学习统一的多模态触觉表示》。在IEEE/CVF计算机视觉与模式识别会议论文集中，页码 26340-26353, 2024。 余家雯，刘海若，余巧军，任杰佳，何聪，丁海彤，黄光宇，黄国范，宋燕，蔡盼盼等。 《ForceVLA：增强接触丰富操作的触觉感知模型》。arXiv 预印本 arXiv:2505.22159, 2025。

Keln Yu、Yuhai Han、Qixian Wang、Vaiav Saxea、Dani Xu 和 Zhao Mimicouc：利用多模态人类触觉示范进行接触丰富的操控。arXiv 预印本 arXiv:2310.16917，2023年。 WeYuan SynDonganEdward H.Adelson.Gelsigh：高分辨率机器人触觉传感器的几何和力学。传感器，17(12)，2017年。ISSN 1424-8220。doi: 10.3390/s17122762。https://www.mdpi.com/1424-8220/17/12/2762。 Xia Zhai BsilMustaa Alexaner Kolesiov 和 Lucas Beyer：语言预训练的模型。在IEEE/CVF国际计算机视觉会议论文集，页码 11975-11986，2023年。 Boqang Zhang、Kehan Li、Zesen Cheng、Zhiqiang Hu、Yuqian Yuan、Guanzheng Chen、Sicong Leng、Yuming Jiang 和 HZha XilVm：推理轨迹模型。arXiv 预印本 arXiv:2501.13106，2025年a。 Chaon Zhang、eng Hao、Xiage o、Xiaoshuai Hao、Shaowei ui 和 Shuo Wangtla：具有偏好学习的视觉-触觉-语言操作模型，用于插入操作。arXiv 预印本 arXiv:2505.09577，2025年b。 Jalia Zhao、Yuxian Ma、LirWang 和 Edward Adelso：跨多种传感器和任务的预训练学习的触觉变换器。arXiv 预印本 arXiv:2406.13640，2024年。 Qinqig Zhao、Yao Lu、Moo Jin Kim、Zipeng Fu、Zhuoyang Zhang、Yecheng Wu、Zhaoshuo Li、Qianli Ma、Song Han 和 CelFi Co-Visl：链状模型在计算机视觉与模式识别会议中的研究，页码 1702-1713，2025年。 Hay Zhen、Xiw Qiu、Peiho Chen、Jinchn Yang、Xin Yan、Yilun Du、Yinig Hong 和 Chuang Gan：3dvl：一个3D视觉-语言-动作生成世界模型。arXiv 预印本 arXiv:2403.09631，2024年。 Yi Zhou、Connelly Barnes、Jngwan Lu、Jime Yang 和 HaoLi：神经网络中旋转表示的连续性。CoRR，abs/1812.07035，2018年。http://arxiv.org/abs/1812.07035。

# 6 附录

# 6.1 数据集与训练细节

数据集对象列表表 7 提供了我们 ObjTac 数据集的完整对象清单，包含 56 个对象，分为十个类别。数据收集过程 数据收集由两个过程组成：触摸和抓取。在触摸过程中，每个对象被触摸 25 次，单次交互持续 10 到 60 秒（以 $60 \mathrm{Hz}$ 采样）。一个 Python 脚本记录手指触觉传感器数据及精确的时间戳，同时一个 Intel RealSense2 摄像头以 720p 分辨率（30 FPS）捕捉同步的第一人称 RGB 视频。在所有 56 个对象中，此过程产生了 252 个视频录制（平均每个 18 秒）、135,000 帧视频和 270,000 个力数据点。抓取过程旨在研究对象操作动态。当抓取过程重新开始时，将系统性测试抓取成功/失败条件及抓取后的稳定性（滑移检测）。试验将包括成功的抓取、失败的尝试、稳定持握阶段和受控释放动作，导致滑移事件。所有试验将保持与触摸过程一致的数据格式，包括同步的 720p 视频和传感器录制。训练细节 表 8 列出了模型的训练细节。

# 6.2 更多结果

Ablatin 研究 Chukig Szes 图 7 显示了从 10 步到 50 步的行动跨度对不同模型均方误差 (MSE) 的影响。无论分块大小如何，OmniVTLA 始终表现出最低的 MSE，突显了其在处理顺序行动依赖方面的稳健性。总体趋势表明，建模更长的动作序列使 VTLA 更好地预测接触动态，这与 VLA 不同，后者在分块长度增加 (从 30 到 50) 时略有降低。动作轨迹比较 图 7 右侧显示了 OmniVTLA 和 VLA 之间的动作轨迹比较。结果清晰地表明 OmniVTLA 在富含触觉的操控任务中具有明显的性能优势。具体而言，与基线 VLA 相比，OmniVTLA 在 Pick&Place 任务中完成的动作步骤约减少 $50\%$，这表明大大提高了操作效率。更重要的是，OmniVTLA 在整个过程中展现了卓越的运动平滑性，在一次尝试中成功完成任务，无需进行纠正调整。相比之下，VLA 的轨迹更加不稳定，存在明显的不稳定性和偶尔的掉落。这些结果表明，融入触觉反馈显著提高了 VLA 在富含触觉的任务中的表现，从而导致更稳定和可靠的抓取行为。

![](images/7.jpg)  

图7 左：VLA、VTLA-Pre、VTLA-FS、VTLA-SA 和 OmniVTLA 模型在不同动作块长度下均方误差 (MSE) 的比较。右：OmniVTLA 和 VLA 的动作轨迹，其中更高的垂直值表示夹具闭合程度更大。

<table><tr><td rowspan=1 colspan=1>Material</td><td rowspan=1 colspan=1>Corresponding Items (~56)</td></tr><tr><td rowspan=1 colspan=1>Plastic</td><td rowspan=1 colspan=1>Plastic bulb, Beverage bottle 1,Beverage bottle 2, Remote control,Phone case, Plastic cup lid, Plastic goblet</td></tr><tr><td rowspan=1 colspan=1>Glass</td><td rowspan=1 colspan=1>Glass bottle, Glass 1, Glass 2</td></tr><tr><td rowspan=1 colspan=1>Wood</td><td rowspan=1 colspan=1>Wooden board</td></tr><tr><td rowspan=1 colspan=1>Brick</td><td rowspan=1 colspan=1>Stone 1, Stone 2, Stone 3, Pebble 1, Pebble 2, Pebble 3</td></tr><tr><td rowspan=1 colspan=1>Metal</td><td rowspan=1 colspan=1>Vice, Metal box, Thermos cup, Laptop, Fountain pen, Adapter</td></tr><tr><td rowspan=1 colspan=1>Fabric</td><td rowspan=1 colspan=1>Pure cotton fabric 1, Pure cotton fabric 2, Pure cotton fabric 3,Jeans, Pillowcase, Linen pants, Nylon shirt, Sweater,Sponge 1, Sponge 2, Canvas peaked cap,Plush toy 1, Plush toy 2, Plush toy 3, Plush toy 4</td></tr><tr><td rowspan=1 colspan=1>Leather</td><td rowspan=1 colspan=1>Leather bag 1, Leather bag 2, Leather bag 3</td></tr><tr><td rowspan=1 colspan=1>Ceramic</td><td rowspan=1 colspan=1>Ceramic bowl, Ceramic tile 1,Ceramic tile 2, Ceramic tile 3, Ceramic tile 4</td></tr><tr><td rowspan=1 colspan=1>Paper</td><td rowspan=1 colspan=1>Toilet paper, Newspaper, Writing paper,Business card, Corrugated paper, Paper shopping bag</td></tr><tr><td rowspan=1 colspan=1>Others</td><td rowspan=1 colspan=1>Apple, Frosted glass, Mouse pad, Notebook cover</td></tr></table>

表7 我们数据集中项目列表

<table><tr><td>Parameter</td><td>VTLA &amp; PiO</td><td>VTA &amp; DP</td></tr><tr><td>GPU</td><td>NVIDIA A100 (80 VRAM)</td><td>NVIDIA A100 (80 VRAM)</td></tr><tr><td>training method</td><td>fine-tune 2.5e-5 peak LR</td><td>train from scratch</td></tr><tr><td>learning rate</td><td>(1K steps linear warmup, 29K steps cosine decay to 2.5e-6)</td><td>0.0001</td></tr><tr><td>total batch size train steps</td><td>32 30K</td><td>32 200K</td></tr><tr><td>input image type</td><td>1 third-person camera image, 1 wrist-mounted camera image</td><td>1 third-person camera image, 1 wrist-mounted camera image</td></tr><tr><td>action chunk size</td><td>1 tactile image (VTLA) 50 steps</td><td>1 tactile image (VTA) 64 steps</td></tr><tr><td>input image size</td><td>224x224</td><td>480x640</td></tr><tr><td>observation history robot state</td><td>no yes (use EEF)</td><td>yes (2-step history)</td></tr></table>

表8 VTLA、Pi0、VTA和DP的训练细节。