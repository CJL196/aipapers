# RT-1: 用于大规模真实世界控制的机器人变换器

1 安东尼·布罗汉\*, 诺亚·布朗\*, 贾斯提斯·卡巴哈尔\*, 叶夫根尼·切博塔尔\*, 约瑟夫·达比斯\*, 切尔西 $\mathbf { F i n n } ^ { * }$ , 基尔萨娜·戈帕拉克里希南\*, 卡罗尔·豪斯曼\*, 亚历克斯·赫尔佐格†, 茉莉·徐\*, 朱利安·伊巴兹\*, 布莱恩·伊赫特\*, 亚历克斯·伊尔潘\*, 托马斯·杰克逊\*, 萨莉·杰斯芒特\*, 尼基尔·J·乔希\*, 瑞安·朱利安\*, 德米特里·卡拉什尼科夫\*, 余恒·阮\*, 伊莎贝尔·利亚尔\*, 郎光辉, 谢尔盖·列温\*, 姚 $\mathbf { L } \mathbf { u } ^ { * }$ , 乌茨瓦·马拉\*, 迪克莎·曼君纳特\*, 伊戈尔·莫达奇‡, 奥菲尔·纳胡姆‡, 卡罗利娜·帕拉达\*, 乔迪林·佩拉尔塔\*, 艾米莉·佩雷斯\*, 卡尔·佩尔奇\*, 乔内尔·基安包\*, 卡尼什卡·拉奥\*, 迈克尔·刘\*, 格雷西亚·萨拉萨尔\*, 潘纳格·桑克提\*, 凯文·赛义德\*, 贾斯皮亚·辛格\*, 苏梅德·松塔克‡, 奥斯汀·斯通\*, 克莱顿 $\mathbf { T a n } ^ { * }$ , 黄·陈\*, 文森特·范霍克\*, 史蒂夫·维加\*, 冯·阮\*, 费 $\mathbf { X i a ^ { * } }$ , 特德·肖\*, 彭 $\mathbf { X } \mathbf { u } ^ { * }$ , 司春 $\mathbf { X } \mathbf { u } ^ { * }$ , 天河 $\mathbf { V } \mathbf { u } ^ { * }$ , 布里安娜·齐特科维奇\* \*谷歌机器人, $^ { \dagger }$ 日常机器人, $^ \ddag$ 谷歌研究，Brain Team

# 摘要

通过从大型、多样化、无任务特定的数据集中转移知识，现代机器学习模型可以在零样本或使用小型任务特定数据集的情况下，以高水平的性能解决特定的下游任务。尽管这一能力已在计算机视觉、自然语言处理和语音识别等其他领域得到证明，但在机器人技术领域仍有待展示，因为模型的泛化能力在收集现实世界机器人数据的难度下显得尤为重要。我们认为，成功构建此类通用机器人模型的关键之一在于开放式的无任务特定训练，结合能够吸收所有多样化机器人数据的高容量架构。本文提出了一种被称为机器人变换器的模型类，展现出可扩展的模型特性。我们通过对不同模型类别的研究以及它们在数据大小、模型大小和数据多样性方面的泛化能力进行验证，基于大型数据集收集真实机器人执行现实任务的数据。该项目的网站和视频可以在 robotics-transformer.github.io 找到。

# 1 引言

端到端的机器人学习，无论是模仿学习还是强化学习，通常涉及在单任务（Kalashnikov et al., 2018；Zhang et al., 2018）或多任务（Kalashnikov et al., 2021b；Jang et al., 2021）环境中收集针对特定任务的数据，这些数据被精确调校以适应机器人应执行的任务。这一工作流程与其他领域（如计算机视觉和自然语言处理）中监督学习的经典方法相似：在这些领域中，通常会收集、标注并部署任务特定的数据集，以解决单个任务，任务之间几乎没有相互作用。近几年来，计算机视觉、自然语言处理及其他领域经历了一场变革，逐渐从孤立的小规模数据集和模型转向基于广泛、大规模数据集预训练的大型通用模型。这些模型成功的关键在于开放式、任务无关的训练，结合可以吸收大规模数据集中所有知识的高容量架构。如果一个模型能够“吸取”经验，以学习语言或感知中的一般模式，那么它可以更高效地将这些模式运用于单个任务。虽然在监督学习中，消除对大型任务特定数据集的需求通常是颇具吸引力的，但在机器人领域这一点显得更为关键，因为数据集可能需要工程密集型的自主操作或昂贵的人类演示。因此，我们提出问题：我们能否在由各种机器人任务组成的数据上训练一个单一的、强大的、大型多任务主干模型？这样的模型是否享有在其他领域观察到的优势，能够实现对新任务、新环境和新对象的零样本泛化？在机器人领域构建这样的模型并非易事。尽管近年来文献中提出了几种大型多任务机器人策略（Reed et al., 2022；Jang et al., 2021），但这些模型往往在现实世界任务的广度上有所限制，比如Gato（Reed et al., 2022），或专注于训练任务，而非对新任务的泛化，比如最近的指令跟随方法（Shridhar et al., 2021；2022），或者在新任务上表现相对较低（Jang et al., 2021）。

![](images/1.jpg)

RT-1 接受图像和自然语言指令，并输出离散化的底部和手臂动作。尽管其规模为 3500 万参数，但仍以每秒 3 赫兹的速度运行，这要归功于其高效且高容量的架构：一个以条件化的 FiLM（Perez 等，2018）、EffcientNet（Tan & Le，2019）、TokenLearner（Ry00 等，2021）和 Transformer（Vaswani 等，2017）为基础的模型。

![](images/2.jpg)  

Figure 1: A high-level overview of RT-1's architecture, dataset, and evaluation.

RT 的大规模、真实世界训练（130,000 个示例）和评估（3000 次真实世界试验）显示出令人印象深刻的泛化能力、鲁棒性和从多样化数据中学习的能力。

两个主要挑战在于组装正确的数据集和设计合适的模型。虽然数据收集和整理常常是许多大规模机器学习项目的“无名英雄”（Radford et al., 2021; Ramesh et al., 2021），但在机器人领域尤为如此，因为数据集往往是特定于机器人并且是手动收集的（Dasari et al., 2019; Ebert et al., 2021）。正如我们在评估中所展示的，良好的泛化能力需要结合规模和广度的数据集，涵盖多种任务和环境。同时，数据集中的任务应当具有足够的关联性，以便使模型能够发现结构相似任务之间的模式，并以新颖的方式执行结合这些模式的新任务。我们收集了与机器人相关的数据集，包含约${ \sim } 1 3 0 \mathrm { k }$个经历和超过700个任务，并在评估中剖析了该数据集的各个方面。第二个挑战在于模型本身的设计。有效的机器人多任务学习需要高容量模型，而Transformer（Vaswani et al., 2017）模型在这方面表现出色，特别是在需要根据语言指令学习多种任务的情况下。然而，机器人控制器还必须足够高效，以便实时运行，这对Transformer尤其构成了重大挑战。我们提出了一种新颖的架构，称为RT-1（Robotics Transformer 1），通过将高维输入和输出（包括相机图像、指令和电机命令）编码为紧凑的词元表示，以供Transformer使用，从而实现高效推理，使实时控制成为可能。

我们的贡献是RT-1模型以及在大规模且广泛的真实世界机器人任务数据集上对该模型的实验。我们的实验不仅展示了RT-1相较于先前技术显著提高的泛化能力和鲁棒性，还评估并消融了模型及训练集构成中的许多设计选择。我们的结果显示，RT-1在超过700个训练指令中实现了$9 7 \%$的成功率，并且在新任务、干扰物和背景上的泛化能力分别比下一个最佳基线提高了$2 5 \%$、$36 \%$和$18 \%$。这样的性能水平使我们能够在SayCan（Ahn等，2022）框架内执行最长达50个阶段的任务。我们进一步展示了RT-1能够结合来自仿真或其他机器人类型的数据，在保持原始任务性能的同时提升对新场景的泛化能力。RT-1能力的简要概述见图$\bar { 1 6 ^ { 2 }$。

# 2 相关工作

最近的一些研究提出了基于 Transformer 的机器人控制策略。与 RT-1 一样，几项研究使用经过 Transformer 处理的语言指令作为指定和推广到新任务的稳健框架（Zhang & Chai, 2021；Pashevich et al., 2021；Silva et al., 2021；Jang et al., 2021；Ahn et al., 2022；Nair et al., 2022）。我们的研究进一步推动了 Transformer 的应用，将语言和视觉观测到机器人动作的映射视为序列建模问题，使用 Transformer 学习这种映射。这个想法直接受到游戏玩法成功的启发（Chen et al., 2021；Lee et al., 2022a），以及模拟机器人导航（Fang et al., 2019）、运动（Janner et al., 2021；Gupta et al., 2022）和操控（Jiang et al., 2022）环境的启发。我们注意到，这些研究中的几项超越了单纯的文本条件，使用 Transformer 在机器人形态（例如，Gupta et al. (2022)）和其他模态的任务规范间进行推广（例如，Jang et al. (2021)；Jiang et al. (2022)）。这些扩展是 RT-1 未来有前景的发展方向。除了基于 Transformer 的策略外，我们的工作重点是实现可推广和稳健的实际机器人操控，且在规模上有所突破。现有关于实际基于 Transformer 的机器人操控的研究关注从每个任务的一组演示中高效学习任务（Shridhar et al., 2022）。Beavir Trar Shaulla 0a Gat（Re l vocate 在大规模机器人和非机器人数据集上学习单一模型。然而，这些研究在实际机器人任务中的应用有限；例如，Gato 有效地学习了单一任务（彩色块叠放），但没有评估其对新任务或各种实际场景的推广能力。在技术层面，我们的研究探讨了如何构建基于 Transformer 的策略，以结合高容量和推广能力，以及实时控制所需的计算效率。

尽管使用高容量Transformer模型来学习机器人控制策略还是一个相对较新的创新，但机器人技术在多任务和语言条件学习方面有着悠久的历史，而RT-1正是在这些基础上构建的。有大量相关工作涉及学习机器人抓取的策略和预测模型，旨在实现对新物体的泛化。早期的研究通过将语言解析、视觉和机器人控制结合的管道方法来解决机器人语言理解问题，也采用了端到端的方法。多任务机器人学习还从学习达到目标的角度进行探讨，以及学习能够在离散集合或其他参数化形式中执行任务的策略。许多早期的机器人研究还专注于收集包含不同任务演示或试验的数据集。我们的工作进一步证实了多任务、语言条件机器人学习的有效性，展示了在更大规模和更丰富的行为、物体和场景下的实验结果，并提出了新的架构和设计选择，使机器人学习能够在显著更大规模上进行。

# 3 准备工作

机器人学习。我们的目标是学习机器人策略，以解决基于语言的视觉任务。正式来说，我们考虑一个顺序决策环境。在时间步 $t = 0$ 时，策略 $\pi$ 接收到语言指令 $i$ 和初始图像观测 $x_{0}$。策略生成一个动作分布 $\pi(\cdot \mid i, x_{0})$，从中采样出一个动作 $a_{0}$ 并应用于机器人。这个过程持续进行，策略通过从学习到的分布 $\pi(\cdot \mid i, \{ x_{j} \}_{j=0}^{t})$ 中采样，迭代地产生动作 $a_{t}$ 并将这些动作应用于机器人。交互在机器人完成指令时结束。从起始步骤 $t = 0$ 到终止步骤 $T$ 的轨迹 $\{ i, \{ ( x_{j}, a_{j} ) \}_{j=0}^{T} \} \}$ 被称为一个回合。在每个回合结束时，智能体将获得一个二元奖励 $r \in \{ 0, 1 \}$，指示机器人是否执行了指令 $i$。目标是学习一个策略 $\pi$，以在指令、初始状态 $x_{0}$ 和转移动态的分布上期望最大化平均奖励。

Transformer。RT-1使用Transformer（Vaswani等，2017）来参数化策略$\pi$ GenerT $\{ \xi _ { h } \} _ { h = 0 } ^ { H }$到输出序列$\{ y _ { k } \} _ { k = 0 } ^ { K }$。虽然Transformer最初是为文本序列设计的，其中每个输入$\xi _ { j }$和输出$y _ { k }$表示一个文本词元，但它们已扩展到图像（Parmar等，2018）以及其他模态（Lee等，2022；Reed等，2022）。如下一节所述，我们通过首先将输入$i$和$\{ x _ { j } \} _ { j = 0 } ^ { t }$映射到序列$\{ \xi _ { h } \} _ { h = 0 } ^ { H }$，并将动作输出$a _ { t }$映射到序列$\{ y _ { k } \} _ { k = 0 } ^ { K }$来参数化$\pi$，即$\{ \xi _ { h } \} _ { h = 0 } ^ { H } \to \{ y _ { k } \} _ { k = 0 } ^ { K }$。模仿学习。模仿学习方法在演示数据集$\mathcal { D }$上训练策略$\pi$（即$\mathcal { D } = \{ ( i ^ { ( n ) } , \{ ( x _ { t } ^ { ( n ) } , \bar { a } _ { t } ^ { ( n ) } ) \} _ { t = 0 } ^ { T ^ { ( n ) } } ) \} _ { n = 0 } ^ { N }$，最终奖励为1）。我们使用行为克隆（Pomerleau，1988）来学习$\pi$，通过最小化给定图像和语言指令的动作$a _ { t }$的负对数似然来优化$\pi$。

# 4 系统概述

本工作的目标是构建并展示一个通用的机器人学习系统，该系统能够吸收大量数据并有效地进行泛化。我们使用来自Everyday Robots的移动操控器，这些操控器具有7个自由度的臂、一只两指夹持器和一个移动底座（参见图2（d））。为了收集数据并评估我们的方法，我们使用三个基于厨房的环境：两个真实办公室厨房和一个基于这些真实厨房构建的训练环境。训练环境如图2（a）所示，由部分台面构成，用于大规模数据收集。两个真实环境如图2（b）和（c）所示，与训练环境的台面相似，但在照明、背景和完整的厨房几何结构上有所不同（例如，可能有一个橱柜而不是抽屉，或者一个水槽可能是可见的）。我们在这些不同环境中评估我们策略的性能，测量策略的表现及其泛化能力。我们的训练数据由人类提供的演示组成，我们为每一集注释了机器人刚刚执行的指令的文本描述。这些指令通常包含一个动词和一个或多个名词来描述目标对象。为了将这些指令进行分类，我们将它们拆分成多个技能（例如，动词如“拾取”、“放置”或“竖放”）和对象（例如，名词如“可乐罐”、“苹果”或“抽屉”）。我们在第5.2节中描述了我们的大规模数据收集策略的详细信息。我们最大的数据库包含超过$1 3 0 \mathrm { k }$个单独的演示，构成700多个不同的任务指令，使用各种各样的对象（参见图2（f））。我们在第5.2节中详细描述了收集的数据。

我们系统的主要贡献之一是网络架构，称为机器人变换器1（RT-1），这是一个高效的模型，能够吸收大量数据，有效泛化，并以实时速率输出机器人控制动作。RT-1以一段短序列图像和自然语言指令作为输入，并在每个时间步输出一个机器人动作。为此，架构（如图1a所示）利用了几个元素：首先，图像和文本通过一个在ImageNet上预训练的卷积网络（Tan & Le, 2019）进行处理，该网络根据指令的预训练嵌入通过FiLM（Perez et al., 2018）进行条件化，接着使用Token Learner（Ryoo et al., 2021）计算一组紧凑的词元，最后通过变换器（Vaswani et al., 2017）对这些词元进行关注，并产生离散化的动作词元。这些动作包括用于手臂运动的七个维度（x, y, z, roll, pitch, yaw，夹爪的开合），用于底座运动的三个维度（x, y, yaw）以及一个离散维度用于在三种模式之间切换：控制手臂、底座或终止回合。RT-1执行闭环控制，以$3 \mathrm{ Hz}$的频率指挥动作，直到产生“终止”动作或达到预设的时间步限制。

# 5 RT-1：机器人转换器

在本节中，我们描述了如何对图像、文本和动作进行词元化，然后讨论 RT-1 模型架构。接着，我们描述了如何实现实时控制所需的运行速度。最后，我们描述了数据收集过程以及我们数据集中的技能和指令。

![](images/3.jpg)  

Figure 2: (a) Robot classroom where we collect data at scale; (b) a real office kitchen, one of the two realistic environments used for evaluation (named Kitchen1 in the rest of the paper); (c) a different office kitchen used for evaluation (named Kitchen2 in the rest of the paper); (d) mobile manipulator used throughout the paper; (e) a set of objects used for most of the skills to expand skill diversity; (f) a more diverse set of objects used mostly to expand object diversity of the picking skill.

# 5.1 模型

我们的模型基于 Transformer 架构（Vaswani 等，2017）构建，接受图像和任务描述的历史作为输入，并直接输出标记化的动作，如图 1a 所示，图 3 中有详细说明。接下来，我们将按照图 3 中的自上而下顺序描述模型的各个组件。有关大规模模型选择的更多细节见附录 C.3。指令和图像标记化。RT-1 架构依赖于高效且紧凑的图像和语言指令标记化。RT-1 通过将图像传入经过 ImageNet 预训练的 EfficientNet-B3（Tan & Le，2019）模型来标记化 6 张图像，该模型接受分辨率为 $300 \times 300$ 的 6 张图像作为输入，并从最后的卷积层输出形状为 $9 \times 9 \times 512$ 的空间特征图。与 Reed 等（2022）不同的是，我们在将图像输入到 Transformer 主干之前，并不将其切分成视觉标记。我们将 EfficientNet 输出的特征图展平为 81 个视觉标记，传递给网络的后续层。

为了包含语言指令，我们将图像标记器条件化于以预训练语言嵌入形式呈现的自然语言指令上，从而允许在早期提取与任务相关的图像特征，并提高RT-1的性能。指令首先通过通用句子编码器（Cer et al., 2018）进行嵌入。该嵌入随后作为输入传递给身份初始化的FiLM层（Perez et al., 2018），这些层被添加到预训练的EfficientNet中，以条件化图像编码器。通常，将FiLM层插入预训练网络内部会干扰中间激活，并消除使用预训练权重的好处。为了解决这个问题，我们将产生FiLM仿射变换的密集层的权重$( f _ { c }$和$h _ { C }$)初始化为零，从而使FiLM层最初作为身份运作，并保留预训练权重的功能。我们发现，身份初始化的FiLM在以从零开始初始化的EfficientNet进行训练时也能产生更好的结果，而不需图像网预训练，但其表现仍不及上述初始化。图像标记器的架构如图3所示。RT-1通过FiLM EfficientNet-B3进行的图像和指令标记总共有1600万个参数，包含26层MBConv块和FiLM层，输出81个视觉-语言令牌。TokenLearner。为了进一步压缩RT-1需要关注的令牌数量，从而加快推理速度，RT-1使用TokenLearner（Ryoo et al., 2021）。TokenLearner是一个逐元素注意力模块，能够学习将大量令牌映射到少量令牌。这使我们能够根据信息软选择图像令牌，仅将重要的令牌组合传递给后续的Transformer层。TokenLearner的引入将来自预训练FiLM-EfficientNet层的81个视觉令牌下采样为8个最终令牌，这些令牌随后被传递给我们的Transformer层。

![](images/4.jpg)  

Figure 3: The architecture diagram of RT-1. The instruction is transformed into a USE embedding and used to condition a pre-trained EffcientNet via FiLM layers. The resulting vision-language tokens are reduced by the TokenLearner and fed into a decoder-only Transformer, which outputs tokenized actions.

Transformer。每个图像的这8个词元被拼接到历史中的其他图像中，形成48个总词元（加上位置编码），输入到RT-1的Transformer主干网络中。该Transformer是一个仅解码器的序列模型，具有8个自注意力层和1900万个总参数，输出动作词元。动作词元化。为了对动作进行词元化，RT-1中的每个动作维度被离散化为256个区间。如前所述，我们考虑的动作维度包括七个与手臂运动相关的变量（$ x , y , z $，滚转、俯仰、偏航、夹爪开合）、三个与底座运动相关的变量（$ x , y $，偏航）以及一个用于在三种模式之间切换的离散变量：控制手臂、底座或终止情节。对于每个变量，我们将目标映射到256个区间中的一个，其中区间在每个变量的范围内均匀分布。损失。我们使用标准的分类交叉熵目标和因果掩蔽，这在之前的基于Transformer的控制器中得到了应用（Reed等, 2022；Lee等, 2022a）。推理速度。与许多大模型应用（如自然语言或图像生成）相比，实时运行在真实机器人上的模型的独特需求之一是快速且一致的推理速度。考虑到我们在本研究中测量的执行指令的人类速度（约为2到4秒），我们希望模型的速度不显著慢于此。基于我们的实验，这一要求对应于至少3 Hz的控制频率，并且在考虑到系统中其他延迟的情况下，模型的推理时间预算应少于100毫秒。这个要求限制了我们可以使用的模型大小。我们在实验中进一步探讨模型大小对推理速度的影响。我们采用两种技术来加快推理速度：（i）通过使用TokenLearner（Ryoo等，2021）减少预训练EffcientNet模型生成的词元数量，（ii）仅计算这些词元一次并在未来的推理中重用它们重叠的窗口。这两者分别使我们的模型推理速度提高了2.4倍和1.7倍。关于模型推理的更多细节在附录C.1中。

# 5.2 数据

Table 1: The list of skills collected for RT-1 together with their descriptions and example instructions.   

<table><tr><td>Skill</td><td>Count</td><td>Description</td><td>Example Instruction</td></tr><tr><td>Pick Object</td><td>130</td><td>Lift the object off the surface</td><td>pick iced tea can</td></tr><tr><td>Move Object Near Object</td><td>337</td><td>Move the first object near the second</td><td>move pepsi can near rxbar blueberry</td></tr><tr><td>Place Object Upright</td><td>8</td><td>Place an elongated object upright</td><td>place water bottle upright</td></tr><tr><td>Knock Object Over</td><td>8</td><td>Knock an elongated object over</td><td>knock redbull can over</td></tr><tr><td>Open Drawer</td><td>3</td><td>Open any of the cabinet drawers</td><td>open the top drawer</td></tr><tr><td>Close Drawer</td><td>3</td><td>Close any of the cabinet drawers</td><td>close the middle drawer</td></tr><tr><td>Place Object into Receptacle 84</td><td></td><td>Place an object into a receptacle</td><td>place brown chip bag into white bowl</td></tr><tr><td>Pick Object from Receptacle 162 and Place on the Counter</td><td></td><td>Pick an object up from a location and then pick green jalapeno chip bag from paper place it on the counter</td><td>bowl and place on counter</td></tr><tr><td>Section 6.3 and 6.4 tasks</td><td>9</td><td>Skills trained for realistic, long instructions</td><td>open the large glass jar of pistachios pull napkin out of dispenser grab scooper</td></tr><tr><td>Total</td><td>744</td><td></td><td></td></tr></table>

我们的目标是构建一个具有高性能、对新任务的泛化能力以及对干扰和背景的鲁棒性的系统。因此，我们旨在收集一个大型、多样化的机器人轨迹数据集，包含多个任务、物体和环境。我们的主要数据集包括约 130,000 个机器人演示数据，这些数据是通过一组 13 台机器人在 17 个月内收集的。我们在一系列办公室厨房区域进行这个大规模的数据收集，我们称之为机器人课堂，如图 2 所示。有关数据收集的更多细节可见附录 C.2。技能和指令。虽然文献中对任务的定义尚不一致，但在本工作中，我们计算系统可以执行的语言指令数量，其中指令对应于一个动词和一个或多个名词的组合，例如“将水瓶竖直放置”、“将可乐罐移到绿色薯片袋上”或“打开抽屉”。RT-1能够在我们在实验中详细评估和描述的多个真实办公厨房环境中执行超过 700 个语言指令。为了对评估进行分组并得出系统性能的结论，我们按动词将指令分组，我们称之为技能。表 1 显示了更详细的指令列表，包括示例和每项技能的指令数量。

当前的技能集包括提取、放置、开关抽屉、从抽屉中取出和放入物品、将长物体竖立放置、将其碰倒、抽取餐巾纸和打开罐子。这些技能的选择旨在展示多种行为与多个物体的相互作用（见图2(e)），以测试RT-1在泛化新指令和执行多项任务能力等方面的表现。然后，我们大幅扩展了“提取”技能的物体多样性，以确保技能能够泛化到各种物体（见图2(f)中扩展的物体集合）。在进行消融实验时，我们进一步扩展了技能集，包含了表1最后一行添加的指令，这些指令用于第6.4节和第6.3节中描述的实验。这些额外的技能专注于办公室厨房中的现实长时间指令。添加任务和数据的整个过程在附录C.4中进行了描述。由于在添加新指令时我们并不对特定技能做出任何假设，因此系统易于扩展，我们能够不断提供更多样化的数据以提升其能力。

# 6 实验

我们的实验旨在回答以下问题：1. RT-1 能否学习执行大量指令，并在零样本情况下对新任务、物体和环境进行泛化？（第 6.2 节） 2. 通过结合异构数据源（例如模拟数据或来自不同机器人的数据），我们能否进一步提升所得到的模型？（第 6.3 节） 3. 各种方法在长时间跨度的机器人场景中如何进行泛化？（第 6.4 节） 4. 在数据量和数据多样性变化的情况下，泛化指标如何变化？（第 6.5 节） 5. 在模型设计中有哪些重要且实用的决策，它们如何影响性能和泛化？（附录第 D.4 节）

在本节中，我们将比较两种最先进的基线架构，Gato（Reed 等，2022）和 BC-Z（Jang 等，2021）。重要的是，这两者都是在我们在第 5.2 节中详细描述的数据上训练的（这是我们系统的重要组成部分），因为这些出版物中的原始模型不会表现出我们评估任务所需的泛化能力。Gato 与 RT-1 相似，基于 Transformer 架构，但在多个方面与 RT-1 有所不同。首先，它在没有语言概念的情况下计算图像令牌，并且每个图像令牌嵌入是为每个图像块单独计算的，而不是像我们模型中的早期语言融合和全局图像嵌入。其次，它不使用预训练的文本嵌入来编码语言字符串。它还不包括第 5.1 节中讨论的对实际机器人来说必要的推理时间考虑因素，例如 TokenLearner 和去除自回归动作。为了以足够高的频率在实际机器人上运行 Gato，我们还限制了模型的大小，相较于原始发布的 12 亿参数（导致机器人推理时间为 1.9 秒），使其与 RT-1 的大小相当（Gato 为 3700 万参数，而 RT-1 为 3500 万参数）。BC-Z 基于 ResNet 架构，曾在 SayCan（Ahn 等，2022）中使用。BC-Z 与 RT-1 的不同之处在于，它是一个前馈模型，不使用以前的时间步，并且它使用连续的动作而不是离散的动作令牌。除了原始的 BC-Z 模型大小外，我们还将我们的方法与一个更大的 BC-Z 版本进行比较，该版本与 RT-1 的参数数量相似，称之为 BC-Z XL。我们在附录 D.4 和 D.5 中研究和分析每个设计决策如何改变性能。我们通过实验评估成功率，以测量训练指令的性能、对未见指令的泛化能力、对背景和干扰项的鲁棒性以及在长时间视野场景中的表现，具体如下。在本节中，我们通过超过 3000 次实际试验评估我们的方法和基线，使其成为迄今为止规模最大的机器人学习系统评估之一。

# 6.1 实验设置

如第4节所述，我们在三个环境中使用来自Everyday Robots的一组移动操控器评估RT-1：两个真实的办公室厨房和一个基于这些真实厨房建模的训练环境。训练环境如图2（a）所示，由部分台面构成，而两个真实环境如图2（b, c）所示，则具有与训练环境相似的台面，但在照明、背景和完整厨房几何形状上有所不同（例如，可能有橱柜而不是抽屉，或可能可见水槽）。评估政策时重点关注训练任务的性能、对新任务的泛化能力、对未见环境的鲁棒性，以及在长期任务中的连锁性能，具体如下。已见任务性能。为了评估在已见指令上的性能，我们对从训练集中抽取的指令进行性能评估。然而，需要注意的是，此评估仍然涉及对象的位置和其他设置因素的变化（例如，时间、机器人位置），要求技能能够对环境中的实际变异进行泛化。总体而言，我们在此评估中测试了超过200个任务：36个对象拾取任务，35个对象击打任务，35个物体竖直放置任务，48个物体移动任务，18个打开和关闭各种抽屉的任务，以及36个从抽屉中取出和放置物体的任务。未见任务泛化。为了评估对未见任务的泛化能力，我们测试了21个新颖的未见指令。这些指令分布在技能和对象上。这确保了训练集中至少存在每个对象和技能的一些实例，但它们会以新颖的方式组合。例如，如果“捡起苹果”被排除在外，则还有其他训练指令中包含该苹果。所有未见指令的列表可以在附录D.1中找到。鲁棒性。为了评估鲁棒性，我们执行了30个真正的任务以测试干扰物鲁棒性，和22个任务以测试背景鲁棒性。背景鲁棒性是通过在新厨房中进行评估（这些厨房具有不同的照明和背景视觉效果）以及使用不同的台面表面（例如，图案桌布）来测试的。鲁棒性评估场景的示例配置如图4所示。长期场景。我们还评估了对更真实的长期场景的泛化，这些场景各自需要执行一系列技能。本次评估的目标是结合多个泛化维度，例如新任务、对象、环境，并测试在现实环境中的整体泛化能力。这些评估由两个真实厨房中的15个长期指令组成，这些指令需要执行包含约10个不同步骤的技能序列，每个步骤的范围大致与训练指令相当。这些步骤是通过使用SayCan系统（Ahn等，2022）从更高级别的指令自动获得，例如“How would you throw away all the items on the table?”，具体细节请见第6.4节和附录D.3。

![](images/5.jpg)  

Figure 4: Evaluation scenarios for distractors (first row), from left to right: easy (0-5 distractors), medium (9 distractors), hard (9 distractors and occluded object); background (second row), from left to right: original environment, patterned table cloth, new kitchen; and realistic scenarios in the real kitchen (third row), generalization levels from left to right: $L 1$ , $L 2$ and $L 3$ .

# 6.2 CAN RT-1 学会执行大量指令，并能对新任务、物体和环境进行泛化吗？

为了回答我们的第一个问题，我们分析了RT-1的整体性能、泛化能力和稳健性，相较于以往提出的模型。具体而言，我们比较了Gato（Reed et al., 2022）和BC-Z（Jang et al., 2021）所使用的模型架构，以及BC-Z的一个更大版本，我们称之为BC-Z XL。然而，需要注意的是，所有模型都是在与RT-1相同的数据上训练的，评估仅比较模型架构，而不是任务集、数据集或整体机器人系统。RT-1的能力在很大程度上取决于数据集和任务集，我们相信它相较于之前的工作有了显著提升（例如，BC-Z使用100个任务，而原始的Gato模型训练了一个具有不同形状的堆叠任务），因此这一比较应被视为较有利于之前的模型，这些模型同样受益于我们收集的大规模多样化的数据集和任务集。

结果如表 2 所示。在每个类别中，我们发现 RT-1 显著优于之前的模型。在已见任务上，RT-1 能够成功执行超过 200 条指令中的 $97\%$，比 BC-Z 多 $25\%$，比 Gato 多 $32\%$。在未见任务上，RT-1 展示了其对新指令的泛化能力，成功执行 $76\%$ 的前所未见的指令，比下一个最佳基线多 $24\%$。虽然这种对新指令的泛化得益于策略的自然语言条件化，使得策略能够理解之前见过的概念的新组合，但所有基线也同样基于自然语言条件化，原则上享有相同的好处。我们在下一部分进一步消融 RT-1 的不同组件，以更好地理解我们的方法中哪些方面对这种差异贡献最大。在干扰物和背景任务上，我们发现 RT-1 具有相当高的鲁棒性，成功执行 $83\%$ 的干扰物鲁棒性任务和 $59\%$ 的背景鲁棒性任务，分别比下一个最佳替代方案高出 $36\%$ 和 $18\%$。总体而言，我们发现 RT-1 具有高通用性能，同时展现出令人印象深刻的泛化和鲁棒性。我们在图 5 中展示了 RT-1 智能体的示例轨迹，包括涵盖不同技能、环境和对象的指令。我们还在附录中提供了不同泛化测试的附加轨迹示例，包括背景（图 10）和干扰物（图 12）。

<table><tr><td>Model</td><td>Seen Tasks</td><td></td><td></td><td>Unseen Tasks Distractors Backgrounds</td></tr><tr><td>Gato (Reed et al., 2022)</td><td>65</td><td>52</td><td>43</td><td>35</td></tr><tr><td>BC-Z (Jang et al., 2021)</td><td>72</td><td>19</td><td>47</td><td>41</td></tr><tr><td>BC-ZXL</td><td>56</td><td>43</td><td>23</td><td>35</td></tr><tr><td>RT-1 (ours)</td><td>97</td><td>76</td><td>83</td><td>59</td></tr></table>

![](images/6.jpg)

Table 2: Overall performance of RT-1 and baselines across seen tasks, generalization to unseen tasks, and robustness to distractors and backgrounds.

对现实指令的推广。接下来，我们测试我们的方法是否足够全面地泛化到之前评估的所有不同维度，以便在真实厨房中部署，因为真实厨房同时面临着多个分布转变，例如新的任务组合、物体干扰以及新的环境。

为了在真实厨房中评估我们的算法，我们构建了一系列任务序列，以实现多个现实目标。机器人在抽屉中补充多种零食，整理打翻的调料瓶，关闭人类遗留的打开抽屉，使用橙子和餐巾准备小吃，并从厨房的多个地方找回失去的太阳镜和章鱼玩具。用于这些场景的详细指令列在附录 D.1 中。办公室厨房与训练环境有显著变化，我们将这些场景中的任务按一般化程度进行分类：$L 1$ 为针对新的台面布局和光照条件的一般化，$L 2$ 为针对未知干扰物体的附加一般化，$L 3$ 为针对全新的任务设置、新任务对象或在未见位置（如靠近水槽）中的对象的进一步一般化。与真实厨房中的三项任务（补充零食、准备小吃和找回失物）对应的三个级别在图 4 的最后一行中进行了描绘。不同级别的示例轨迹在附录图 11 中呈现。我们在这些现实场景中报告每项任务的成功率以及不同的一般化水平，并在表 3 中发现 RT-1 在所有级别中表现最为稳健。Gato 在第一级别的表现相当不错，但在更困难的一般化场景中的表现显著下降。BC-Z 及其 XL 版本在 $L 2$ 级别的表现相当不错，并且在 $L 3$ 的表现优于 Gato，但仍未达到 RT-1 的一般化水平。

# 6.3 我们能否通过结合异构数据源，如模拟数据或来自不同机器人的数据，进一步提升模型的结果？

接下来，我们探索RT-1在利用高度异质数据方面的局限性。我们展示了RT-1如何整合和学习来自不同数据源的数据，并在不牺牲其在这些数据固有的各种任务中的原始任务表现的情况下，从这些数据中改进。为此，我们进行两项实验：（1）RT-1在真实数据和模拟数据上进行训练和测试，以及（2）

![](images/7.jpg)  

Figure 5: Example evaluation trajectories for RT-1 across various instructions.

<table><tr><td></td><td colspan="4">Generalization Scenario Levels</td></tr><tr><td>Models</td><td>All</td><td>L1</td><td>L2</td><td>L3</td></tr><tr><td>Gato Reed et al. (2022)</td><td>30</td><td>63</td><td>25</td><td>0</td></tr><tr><td>BC-Z Jang et al. (2021)</td><td>45</td><td>38</td><td>50</td><td>50</td></tr><tr><td>BC-ZXL</td><td>55</td><td>63</td><td>75</td><td>38</td></tr><tr><td>RT-1 (ours)</td><td>70</td><td>88</td><td>75</td><td>50</td></tr></table>

![](images/8.jpg)  

Table 3: Realistic generalization scenarios:we compare model success rate in a realistic Google kitchen scenarios across three levels of generalization: $L 1$ for generalization to the new counter-top layout and lighting conditions, $L 2$ for additionally generalization to unseen distractor objects, $L 3$ for additionally generalization to drastically new task settings, new task objects or in unseen locations like near a sink.

RT-1 在由不同机器人原始收集的大规模多任务数据集上进行训练。有关每个数据集的更多信息请参见附录 D.2。

吸收仿真数据。表4显示了RT-1及基线模型吸收真实和仿真数据的能力。为了进行测试，我们使用所有真实演示数据，同时提供额外的仿真数据，其中包含机器人在现实世界中从未见过的物体。具体而言，我们指定了不同的泛化场景：对于使用真实物体的已见技能，训练数据包含该指令的真实数据（即，在已见任务上的表现）；对于使用仿真物体的已见技能，训练数据包含该指令的仿真数据（例如，“拾取一个仿真物体”，该物体在仿真中存在）；而对于使用仿真物体的未见技能，训练数据包含该物体的仿真数据，但在仿真或现实中都没有描述该物体技能的指令示例（例如，“将仿真物体移动到苹果旁”，尽管机器人只练习过拾取该仿真物体，而没有在靠近其他物体时移动它）。所有评估均在现实世界中进行，但为了限制评估指令的数量，我们专注于拾取和移动技能。

![](images/9.jpg)

Table 4: Experimental results for incorporating simulation data in RT-1. Adding simulation data does not impact the performance on real objects, while significantly improving real performance on objects that were only introduced in simulation $( + 6 4 \% )$ . It also improves real-world generalization on simulated objects used with skills seen only in the real world $( + 2 6 \% )$ , e.g. "move $\mathrm { X }$ to $\mathbf { Y } ^ { \prime \prime }$ where X only appeared in simulated "pick $X ^ { \ast }$ task.   

<table><tr><td></td><td></td><td>Real Objects</td><td colspan="2">Sim Objects (not seen in real)</td></tr><tr><td></td><td>Models Training Data</td><td>Seen Skill w/ Objects</td><td>Seen Skill w/ Objects</td><td>Unseen Skill w/ Objects</td></tr><tr><td>RT-1</td><td>Real Only</td><td>92</td><td>23</td><td>7</td></tr><tr><td>RT-1</td><td>Real + Sim</td><td>90(-2)</td><td>87(+64)</td><td>33(+26)</td></tr></table>

在表4中，我们发现对于RT-1，添加模拟数据并没有导致性能的下降，与仅使用真实数据集相比。然而，我们确实看到在仅在模拟中见过的物体和任务上，性能显著提高（从23%提升到87%），接近真实数据中的表现，显示出显著的领域转移能力。我们还观察到在未见过的指令上，性能从7%提升到33%；考虑到所涉及的物体在真实环境中从未见过，且指令也完全未曾接触，这一结果非常令人印象深刻。总体而言，我们发现RT-1能够有效吸收新数据，即使来自非常不同的领域。 吸收来自不同机器人的数据。为了推动RT-1的数据吸收极限，我们进行了一系列额外实验，将来自不同机器人的两个数据源结合起来：Kuka IIWA以及迄今为止实验中使用的Everyday Robots移动操纵器。Kuka数据包含在QT-Opt（Kalashnikov等，2018）中收集的所有成功示例，共209k个回合，其中机器人在一个箱子中无差别地抓取物体（查看表5中的Kuka示例）。为了测试RT-1是否能够有效吸收这两个非常不同的数据集，我们将其称为标准的“课堂评估”，以及在Kuka数据中反映箱子抓取设置的新构建任务的性能，称为“箱子抓取评估”（见图6）。我们想强调这个设置的难度，指出数据集之间的主要差异。收集数据的机器人在外观和动作空间上不仅不同，而且它们所部署的环境在外观和动态上也不同。此外，QT-Opt数据展示了完全不同的动作分布，因为它是由RL智能体收集的，而我们数据集中则是人类演示。

结果在表5中呈现。我们观察到，将RT-1数据与Kuka数据混合的模型在原始任务的性能上（即课堂评估）只有微小下降，即$2 \%$。更重要的是，在Bin-picking评估中，我们观察到在多机器人数据上训练的模型表现为$39 \%$，而仅在RT-1数据上训练的模型表现为$22 \%$。这之间存在$17 \%$的性能差异（几乎是$2 \mathbf{x}$）。此外，RT-1在Kuka bin-picking数据上训练，并在与Everyday Robots (EDR)机器人进行bin-picking任务时，表现为$0 \%$的性能，这确认了从另一种机器人形态转移行为的困难。然而，将两种机器人数据混合，使RT-1能够推断EDR机器人的正确动作，即使面临Kuka机器人观察到的状态。这是在没有对EDR机器人进行bin-picking的明确演示的情况下实现的，并利用了Kuka机器人收集的过去经验。这些结果表明，RT-1的吸收特性还包括通过观察其他机器人的经验来获取新技能的能力，并展示了一个令人兴奋的未来工作方向，我们可以结合更多的多机器人数据集以增强机器人的能力。

![](images/10.jpg)  

Figure 6: In Table 5, RT-1 is trained with data from two robotics platforms and learns to generalize across them.

![](images/11.jpg)

Table 5: Experimental results for mixing data from two different robots. Incorporating Kuka binpicking data from QT-Opt (Kalashnikov et al., 2018) in RT-1 minimally impacts the standard classroom evaluation performance and results in almost a $2 \mathbf { x }$ improvement in generalization to the Binpicking evaluation (that is similar to the setup in the Kuka data) on the Everyday Robots manipulator. This demonstrates an effective transfer across two different robot morphologies.   

<table><tr><td>Models</td><td>Training Data</td><td>Classroom eval Bin-picking eval</td><td></td></tr><tr><td>RT-1</td><td>Kuka bin-picking data + EDR data</td><td>90(-2)</td><td>39(+17)</td></tr><tr><td>RT-1</td><td>EDR only data</td><td>92</td><td>22</td></tr><tr><td>RT-1</td><td>Kuka bin-picking only data</td><td>0</td><td>0</td></tr></table>

# 6.4 各种方法如何概括长时间跨度的机器人场景？

在接下来的实验中，我们评估我们的方法是否具有足够的泛化能力，以便在长期实际厨房环境中使用。为了回答这个问题，我们在两个不同的真实厨房中执行RT-1和各种基线，使用SayCan（Ahn等，2022）框架。由于SayCan将许多低级指令结合起来以执行高级指令，因此可能的高级指令数量随着技能的增加而呈组合爆炸式增长，因此RT-1的技能广度得以充分体现（有关SayCan算法的更多细节，请参考Ahn等（2022））。长期任务的成功率也随着任务长度的增加呈指数下降，因此在操作技能中的高成功率尤其重要。此外，由于移动操作任务同时需要导航和操作，因此策略对基础位置的鲁棒性至关重要。更多细节见附录D.3。

表6展示了我们的结果（见附录表12中的指令）。除原始SayCan外，所有方法的规划成功率均为87%，而RT-1表现最佳，在Kitchen1中的执行成功率为67%。Kitchen2构成了一个更具挑战性的泛化场景，因为机器人课堂训练场景是基于Kitchen1构建的（见图2中的厨房图片）。由于这种泛化困难，结合Gato的SayCan无法完成任何长时间任务，而结合BC-Z的SayCan成功率为13%。原始SayCan论文并未评估在新厨房中的性能。令人惊讶的是，在我们的算法中，从Kitchen1到Kitchen2的操作表现没有明显下降。在补充视频中，我们展示了这使我们能够操作Kitchen2中未见过的抽屉，并且可以使用SayCan-RT1规划和执行超长时间任务，步骤最多可达50步。

Table 6: SayCan style long horizon tasks in Kitchen1 and Kitchen2. (\*Original SayCan eval uses a slightly different prompt so the planning success rate is lower.)   

<table><tr><td rowspan="2"></td><td colspan="2"></td><td colspan="2">SayCan tasks in Kitchen1 SayCan tasks in Kitchen2</td></tr><tr><td>Planning</td><td>Execution</td><td>Planning</td><td>Execution</td></tr><tr><td>Original SayCan (Ahn et al., 2022)*</td><td>73</td><td>47</td><td>-</td><td>-</td></tr><tr><td>SayCan w/ Gato (Reed et al., 2022)</td><td>87</td><td>33</td><td>87</td><td>0</td></tr><tr><td>SayCan w/ BC-Z (Jang et al., 2021)</td><td>87</td><td>53</td><td>87</td><td>13</td></tr><tr><td>SayCan w/ RT-1 (ours)</td><td>87</td><td>67</td><td>87</td><td>67</td></tr></table>

# 6.5 泛化度量如何随着数据量和数据多样性的变化而变化？

虽然之前的研究已经展示了基于变换器模型的模型参数数量的扩展能力（Lee et al., 2022a；Reed et al., 2022；Jiang et al., 2022），但在许多机器人领域，模型规模往往不是主要瓶颈，其最大规模受限于在真实机器人上运行这些模型的延迟要求。相反，在本研究中，我们重点探讨数据集规模和多样性对模型性能的影响，因为这些因素在传统数据受限的机器人学习领域中扮演着重要角色。由于对真实机器人进行数据收集特别昂贵，因此量化我们的模型需要什么样的数据以实现特定的性能和泛化能力非常重要。因此，我们的最后一个问题关注于RT-1在不同数据特性下的扩展属性。

<table><tr><td></td><td></td><td></td><td></td><td colspan="4">Generalization</td></tr><tr><td>Models</td><td>% Tasks % Data</td><td></td><td>Seen Tasks</td><td></td><td></td><td>All Unseen Tasks Distractors Backgrounds</td><td></td></tr><tr><td>Smaller Data</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RT-1 (ours)</td><td>100</td><td>100</td><td>97</td><td>73</td><td>76</td><td>83</td><td>59</td></tr><tr><td>RT-1</td><td>100</td><td>51</td><td>71</td><td>50</td><td>52</td><td>39</td><td>59</td></tr><tr><td>RT-1</td><td>100</td><td>37</td><td>55</td><td>46</td><td>57</td><td>35</td><td>47</td></tr><tr><td>RT-1</td><td>100</td><td>22</td><td>59</td><td>29</td><td>14</td><td>31</td><td>41</td></tr><tr><td>Narrower Data</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RT-1 (ours)</td><td>100</td><td>100</td><td>97</td><td>73</td><td>76</td><td>83</td><td>59</td></tr><tr><td>RT-1</td><td>75</td><td>97</td><td>86</td><td>54</td><td>67</td><td>42</td><td>53</td></tr></table>

![](images/12.jpg)  

Table 7: Various data ablations of RT-1 across seen tasks, generalization to unseen tasks, and robustness to distractors and backgrounds. Data diversity has a higher impact on the performance and generalization than data quantity.

在表7中，我们展示了RT-1的性能、泛化能力和鲁棒性，随着数据集规模（数据的百分比）和数据集多样性（任务的百分比）的减少。为了分隔数据集规模和多样性的轴，我们通过从数据量最大的任务中移除数据来创建具有相同任务多样性的小型数据集，将每个任务的示例数量限制为200（结果为51%的数据）、100（37%的数据）和50（22.5%的数据）。为了创建一个更窄的数据集，我们移除了数据最少的任务，从而保留了97%的整体数据，但仅保留了75%的任务。随着数据集规模的减小，我们观察到性能和泛化能力普遍下降的趋势，而泛化能力的下降趋势更为明显。当我们使数据集更窄时，我们看到性能急剧下降，尤其是在泛化方面。实际上，移除25%的任务，同时保留97%的数据，相当于将数据集规模减少多达49%的相应泛化性能。因此，我们的主要启示是数据多样性比数据数量更为重要。

# 7 结论、局限性与未来工作

我们提出了机器人变压器1（Robotics Transformer 1, RT-1），这是一种能够有效吸收大量数据并随着数据数量和多样性进行扩展的机器人学习方法。我们在一个包含超过130,000个示范回合的大型数据集上训练了RT-1，该数据集是在17个月内与13个机器人收集的。在我们广泛的实验中，我们展示了我们的方法能够在97%的成功率下执行超过700条指令，并且有效地对新任务、物体和环境进行泛化，优于之前发表的基准。我们还展示了RT-1能够成功吸收来自仿真和其他机器人形态的异构数据，而不会牺牲原始任务的性能，同时改善对新场景的泛化。最后，我们展示了这种性能和泛化水平如何使我们能够在SayCan（Ahn等，2022）框架中执行非常长的任务，最多可达到50个步骤。尽管RT-1在大规模机器人学习中展示了一个有前景的模型吸收步骤，但它也存在一些局限性。首先，它是一种模仿学习方法，继承了该类方法的挑战，例如它可能无法超越示范者的性能。其次，对新指令的泛化仅限于以前见过的概念组合，RT-1尚无法对未见过的全新动作进行泛化。最后，我们的方法在一组大型但不够灵活的操作任务上进行了展示。我们计划继续扩展RT-1能够使能和泛化的指令集，以应对这一挑战。在探索该项工作的未来方向时，我们希望通过开发允许非专家通过定向数据收集和模型提示来训练机器人的方法，更快地扩大机器人技能的数量。尽管当前版本的RT-1在处理干扰物体时相当强健，但其在背景和环境方面的鲁棒性仍可以通过大幅提高环境多样性来进一步改善。我们还希望通过可扩展的注意力机制和记忆来提高RT-1的反应速度和上下文保持能力。为了使研究界能够在此项工作基础上进行更深入的研究，我们已将RT$1^{4}$的代码开源，希望能为研究人员提供一个宝贵的资源，以推动机器人学习的扩展。

# 致谢

我们衷心感谢 Aleksandra Faust、Andy Christiansen、Chuyuan Fu、Daniel Kappler、David Rendleman、Eric Jang、Jessica Gomez、Jessica Lin、Jie Tan、Josh Weaver、Justin Boyd、Krzysztof Choromanski、Matthew Bennice、Mengyuan Yan、Mrinal Kalakrishnan、Nik Stewart、Paul Wohlhart、Peter Pastor、Pierre Sermanet、Wenlong Lu、Zhen Yu Song、Zhuo Xu，以及 Google 机器人团队和 Everyday Robots 的广大团队，感谢他们的反馈和贡献。

# REFERENCES

Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, et al. Do as I can, not as I say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691, 2022.

Daniel Cer, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St John, Noah Constant, Mario Guajardo-Cespedes, Steve Yuan, Chris Tar, et al. Universal sentence encoder. arXiv preprint arXiv:1803.11175, 2018.

Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Misha Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. Decision transformer: Reinforcement learning via sequence modeling. Advances in neural information processing systems, 34:1508415097, 2021.

Michael Jae-Yoon Chung, Abram L Friesen, Dieter Fox, Andrew N Meltzoff, and Rajesh PN Rao. A bayesian developmental approach to robotic goal-based imitation learning. PloS one, 10(11): e0141965, 2015.

Sudeep Dasari, Frederik Ebert, Stephen Tian, Suraj Nair, Bernadette Bucher, Karl Schmeckpeper, Siddharth Singh, Sergey Levine, and Chelsea Finn. Robonet: Large-scale multi-robot learning. In Conference on Robot Learning, 2019.

Marc Peter Deisenroth, Peter Englert, Jan Peters, and Dieter Fox. Multi-task policy search for robotics. In 2014 IEEE international conference on robotics and automation (ICRA), pp. 3876 3881. IEEE, 2014.

Coline Devin, Abhishek Gupta, Trevor Darrell, Pieter Abbeel, and Sergey Levine. Learning modular neural network policies for multi-task and multi-robot transfer. In 2017 IEEE international conference on robotics and automation (ICRA), pp. 21692176. IEEE, 2017.

Miroslav Dudík, John Langford, and Lihong Li. Doubly robust policy evaluation and learning. arXiv preprint arXiv:1103.4601, 2011.

Frederik Ebert, Yanlai Yang, Karl Schmeckpeper, Bernadette Bucher, Georgios Georgakis, Kostas Daniilidis, Chelsea Finn, and Sergey Levine. Bridge data: Boosting generalization of robotic skills with cross-domain datasets. arXiv preprint arXiv:2109.13396, 2021.

Kuan Fang, Alexander Toshev, Li Fei-Fei, and Silvio Savarese. Scene memory transformer for embodied agents in long-horizon tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 538547, 2019.

Roy Fox, Ron Berenstein, Ion Stoica, and Ken Goldberg. Multi-task hierarchical imitation learning for home automation. In 2019 IEEE 15th International Conference on Automation Science and Engineering (CASE), pp. 18. IEEE, 2019.

Abhinav Gupta, Adithyavairavan Murali, Dhiraj Prakashchand Gandhi, and Lerrel Pinto. Robot learning in homes: Improving generalization and reducing dataset bias. Advances in neural information processing systems, 31, 2018.

Agrim Gupta, Linxi Fan, Surya Ganguli, and Li Fei-Fei. Metamorph: Learning universal controllers with transformers. arXiv preprint arXiv:2203.11931, 2022.

Josiah P Hanna, Peter Stone, and Scott Niekum. Bootstrapping with models: Confidence intervals for off-policy evaluation. In Thirty-First AAAI Conference on Artificial Intelligence, 2017.

Daniel Ho, Kanishka Rao, Zhuo Xu, Eric Jang, Mohi Khansari, and Yunfei Bai. RetinaGAN: An object-aware approach to sim-to-real transfer, 2020. URL https : / /arxiv. org/abs/ 2011.03148.

De-An Huang, Yu-Wei Chao, Chris Paxton, Xinke Deng, Li Fei-Fei, Juan Carlos Niebles, Animesh Garg, and Dieter Fox. Motion reasoning for goal-based imitation learning. In 2020 IEEE International Conference on Robotics and Automation (ICRA), pp. 48784884. IEEE, 2020.

Alexander Irpan, Kanishka Rao, Konstantinos Bousmalis, Chris Harris, Julian Ibarz, and Sergey Levine. Off-policy evaluation via off-policy classification. Advances in Neural Information Processing Systems, 32, 2019.

Stephen James, Zicong Ma, David Rovick Arrojo, and Andrew J Davison. RLBench: The robot learning benchmark & learning environment. IEEE Robotics and Automation Letters, 5(2):3019 3026, 2020.

Eric Jang, Alex Irpan, Mohi Khansari, Daniel Kappler, Frederik Ebert, Corey Lynch, Sergey Levine, and Chelsea Finn. Bc-z: Zero-shot task generalization with robotic imitation learning. In Conference on Robot Learning, pp. 9911002. PMLR, 2021.

Michael Janner, Qiyang Li, and Sergey Levine. Reinforcement learning as one big sequence modeling problem. In ICML 2021 Workshop on Unsupervised Reinforcement Learning, 2021.

Yunfan Jiang, Agrim Gupta, Zichen Zhang, Guanzhi Wang, Yongqiang Dou, Yanjun Chen, Li FeiFei, Anima Anandkumar, Yuke Zhu, and Linxi Fan. Vima: General robot manipulation with multimodal prompts. arXiv preprint arXiv:2210.03094, 2022.

Tom Jurgenson, Or Avner, Edward Groshev, and Aviv Tamar. Sub-goal trees a framework for goalbased reinforcement learning. In International Conference on Machine Learning, pp. 50205030. PMLR, 2020.

Dmitry Kalashnikov, Alex Irpan, Peter Pastor, Julian Ibarz, Alexander Herzog, Eric Jang, Deirdre Quillen, Ethan Holly, Mrinal Kalakrishnan, Vincent Vanhoucke, et al. Scalable deep reinforcement learning for vision-based robotic manipulation. In Conference on Robot Learning, pp. 651 673. PMLR, 2018.

Dmitry Kalashnikov, Jacob Varley, Yevgen Chebotar, Benjamin Swanson, Rico Jonschkowski, Chelsea Finn, Sergey Levine, and Karol Hausman. Mt-opt: Continuous multi-task robotic reinforcement learning at scale. arXiv preprint arXiv:2104.08212, 2021a.

Dmitry Kalashnikov, Jake Varley, Yevgen Chebotar, Ben Swanson, Rico Jonschkowski, Chelsea Finn, Sergey Levine, and Karol Hausman. MT-opt: Continuous multi-task robotic reinforcement learning at scale. arXiv, 2021b.

Thomas Kollar, Stefanie Tellex, Deb Roy, and Nicholas Roy. Toward understanding natural language directions. In 2010 5th ACM/IEEE International Conference on Human-Robot Interaction (HRI), pp. 259266. IEEE, 2010.

Kuang-Huei Lee, Ofir Nachum, Mengjiao Yang, Lisa Lee, Daniel Freeman, Winnie Xu, Sergio Guadarrama, Ian Fischer, Eric Jang, Henryk Michalewski, et al. Multi-game decision transformers. arXiv preprint arXiv:2205.15241, 2022a.

Kuang-Huei Lee, Ted Xiao, Adrian Li, Paul Wohlhart, Ian Fischer, and Yao Lu. PI-QT-Opt: Predictive information improves multi-task robotic reinforcement learning at scale. arXiv preprint arXiv:2210.08217, 2022b.

Ian Lenz, Honglak Lee, and Ashutosh Saxena. Deep learning for detecting robotic grasps. The International Journal of Robotics Research, 34(4-5):705724, 2015.

Corey Lynch and Pierre Sermanet. Language conditioned imitation learning over unstructured data. arXiv preprint arXiv:2005.07648, 2020.

Matt MacMahon, Brian Stankiewicz, and Benjamin Kuipers. Walk the talk: Connecting language, knowledge, and action in route instructions. Def, 2(6):4, 2006.

Hongyuan Mei, Mohit Bansal, and Matthew R Walter. Listen, attend, and walk: Neural mapping of navigational instructions to action sequences. In Thirtieth AAAI Conference on Artificial Intelligence, 2016.

Suraj Nair, Eric Mitchell, Kevin Chen, Silvio Savarese, Chelsea Finn, et al. Learning languageconditioned robot behavior from offline data and crowd-sourced annotation. In Conference on Robot Learning, pp. 13031315. PMLR, 2022.

Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander $\mathrm { K u }$ , and Dustin Tran. Image transformer. In International conference on machine learning, pp. 4055 4064. PMLR, 2018.

Alexander Pashevich, Cordelia Schmid, and Chen Sun. Episodic transformer for vision-andlanguage navigation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1594215952, 2021.

Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1), Apr. 2018. doi: 10.1609/aaai.v32i1.11671. URL https : / /ojs .aaai. org/index.php/AAAI/article/view/11671.

Lerrel Pinto and Abhinav Gupta. Supersizing self-supervision: Learning to grasp from $5 0 \mathrm { k }$ tries and 700 robot hours. In 2016 IEEE international conference on robotics and automation (ICRA), pp. 34063413. IEEE, 2016.

Dean A Pomerleau. Alvinn: An autonomous land vehicle in a neural network. Advances in neural information processing systems, 1, 1988.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pp. 87488763. PMLR, 2021.

Antonin Raffin, Ashley Hill, René Traoré, Timothée Lesort, Natalia Díaz-Rodríguez, and David Filliat. Decoupling feature extraction from policy learning: assessing benefits of state representation learning in goal based robotics. arXiv preprint arXiv: 1901.08651, 2019.

Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pp. 88218831. PMLR, 2021.

Scott Reed, Konrad Zolna, Emilio Parisotto, Sergio Gomez Colmenarejo, Alexander Novikov, Gabriel Barth-Maron, Mai Gimenez, Yury Sulsky, Jackie Kay, Jost Tobias Springenberg, et al. A generalist agent. arXiv preprint arXiv:2205.06175, 2022.

Michael Ryoo, AJ Piergiovanni, Anurag Arnab, Mostafa Dehghani, and Anelia Angelova. Tokenlearner: Adaptive space-time tokenization for videos. Advances in Neural Information Processing Systems, 34:1278612797, 2021.

Ashutosh Saxena, Justin Driemeyer, Justin Kearns, and Andrew Ng. Robotic grasping of novel objects. Advances in neural information processing systems, 19, 2006.

Nur Muhammad Mahi Shafiullah, Zichen Jeff Cui, Ariuntuya Altanzaya, and Lerrel Pinto. Behavior transformers: Cloning $k$ modes with one stone. arXiv preprint arXiv:2206.11251, 2022.

Pratyusha Sharma, Lekha Mohan, Lerrel Pinto, and Abhinav Gupta. Multiple interactions made easy (mime): Large scale demonstrations data for imitation. In Conference on robot learning, pp. 906915. PMLR, 2018.

Mohit Shridhar, Lucas Manuelli, and Dieter Fox. Cliport: What and where pathways for robotic manipulation. In Proceedings of the 5th Conference on Robot Learning (CoRL), 2021.

Mohit Shridhar, Lucas Manuelli, and Dieter Fox. Perceiver-actor: A multi-task transformer for robotic manipulation. arXiv preprint arXiv:2209.05451, 2022.

Andrew Silva, Nina Moorman, William Silva, Zulfiqar Zaidi, Nakul Gopalan, and Matthew Gombolay. Lancon-learn: Learning with language to enable generalization in multi-task manipulation. IEEE Robotics and Automation Letters, 7(2):16351642, 2021.

Avi Singh, Eric Jang, Alexander Irpan, Daniel Kappler, Murtaza Dalal, Sergey Levinev, Mohi Khansari, and Chelsea Finn. Scalable multi-task imitation learning with autonomous improvement. In 2020 IEEE International Conference on Robotics and Automation (ICRA), pp. 2167 2173. IEEE, 2020.

Simon Stepputtis, Joseph Campbell, Mariano Phielipp, Stefan Lee, Chitta Baral, and Heni Ben Amor. Language-conditioned imitation learning for robot manipulation tasks. Advances in Neural Information Processing Systems, 33:1313913150, 2020.

Mingxing Tan and Quoc Le. EfficientNet: Rethinking model scaling for convolutional neural networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 61056114. PMLR, 09-15 Jun 2019. URL https: //proceedings.mlr. press/v97/tan19a.html.

Stefanie Tellex, Thomas Kollar, Steven Dickerson, Matthew Walter, Ashis Banerjee, Seth Teller, and Nicholas Roy. Understanding natural language commands for robotic navigation and mobile manipulation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 25, pp. 15071514, 2011.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Ulrich Viereck, Andreas Pas, Kate Saenko, and Robert Platt. Learning a visuomotor controller for real world robotic grasping using simulated depth images. In Conference on robot learning, pp. 291300. PMLR, 2017.

Ted Xiao, Eric Jang, Dmitry Kalashnikov, Sergey Levine, Julian Ibarz, Karol Hausman, and Alexander Herzog. Thinking while moving: Deep reinforcement learning with concurrent control. arXiv preprint arXiv:2004.06089, 2020.

Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on robot learning, pp. 10941100. PMLR, 2020.

Tianhao Zhang, Zoe McCarthy, Owen Jow, Dennis Lee, Xi Chen, Ken Goldberg, and Pieter Abbeel. Deep imitation learning for complex manipulation tasks from virtual reality teleoperation. In 2018 IEEE International Conference on Robotics and Automation (ICRA), pp. 56285635. IEEE, 2018.

Yichi Zhang and Joyce Chai. Hierarchical task learning from language instructions with unified transformers and self-monitoring. arXiv preprint arXiv:2106.03427, 2021.

# APPENDIX

# A AutHoR COntributIONS

• Evaluations (ablations, designing procedures, implementations, and running ablations): Yevgen Chebotar, Keerthana Gopalakrishnan, Karol Hausman, Julian Ibarz, Brian Ichter, Alex Irpan, Isabel Leal, Kuang-Huei Lee, Yao Lu, Ofir Nachum, Kanishka Rao, Sumedh Sontakke, Austin Stone, Quan Vuong, Fei Xia, Ted Xiao, and Tianhe Yu.   
• Network Architecture (tokenizer, training, inference): Yevgen Chebotar, Keerthana Gopalakrishnan, Julian Ibarz, Alex Irpan, Kuang-Huei Lee, Yao Lu, Karl Pertsch, Kanishka Rao, Michael Ryoo, Sumedh Sontakke, Austin Stone, and Quan Vuong.   
•Developed Infrastructure (data, training, collect, simulation, evaluations, storage, and operations): Anthony Brohan, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Yao Lu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, and Tianhe Yu.   
•Leadership (managed or advised on the project): Chelsea Finn, Karol Hausman, Julian Ibarz, Sally Jesmonth, Sergey Levine, Yao Lu, Igor Mordatch, Carolina Parada, Kanishka Rao, Pannag Sanketi, Vincent Vanhoucke.   
• Paper (figures, vizualizations, writing): Keerthana Gopalakrishnan, Karol Hausman, Brian Ichter, Sergey Levine, Ofir Nachum, Karl Pertsch, Kanishka Rao, Austin Stone, Fei Xia, and Ted Xiao.   
•Data collection and evaluations: Noah Brown, Justice Carbajal, Joseph Dabis, Tomas Jackson, Utsav Malla, Deeksha Manjunath, Jodily Peralta, Emily Perez, Jornell Quiambao, Grecia Salazar, Kevin Sayed, Jaspiar Singh, Clayton Tan, Huong Tran, Steve Vega, and Brianna Zitkovich.

B MODEL CARD

We present the Model Card for RT-1 in Fig. 7.

# C MOdEL ANd Data

C.1 MODEL INFERENCE

In addition to the inference speed requirement, we need to ensure that our system outputs actions at a consistent frequency, avoiding jitter. To accomplish this, we introduce a fixed-time waiting mechanism that waits a certain amount of time (280ms, the max observed latency of all components) after the state, that was used to compute the next action, has been captured, but before applying the action, similarly to the procedure described by Xiao et al. (2020).

# C.2 DATA COLLECTION AT SCALE.

Each of the robots autonomously approaches its station at the beginning of the episode and communicates to the operator the instruction that they should demonstrate to the robot. To ensure a balanced dataset as well as randomization of the scene, we created a software module responsible for sampling the instructions to be demonstrated as well as the randomization of the background configuration. Each of the robots tells the demonstrator how to randomize the scene and which instruction to demonstrate.

Demonstrations are collected with direct line-of-sight between operator and robot using 2 virtual reality remotes. We map remote controls onto our policy action space to preserve consistency of the transition-dynamics. 3D position and rotational displacements of the remote are mapped to 6d displacements of the robot tool. The x, y position of the joystick is mapped to a turning angle and driving distance of the mobile base. We compute and track trajectories to the target poses that we obtain from the joystick commands.

# Model Card for RT-1 (Robotics Transformer)

# Model Details

•Developed by researchers at Robotics at Google and Everyday Robots, 2022, v1. •Transformer-based model, built upon a FiLM-conditioned EfficientNet (Tan & Le, 2019), a TokenLearner (Ryoo et al., 2021), and a Transformer (Vaswani et al., 2017). Trained with imitation learning with inputs of natural language tasks and images and output robot actions.

# Intended Use

•Intended to be used for controlling an Everyday Robot for manipulation tasks.   
•Unclear suitability as a learned representation for different robotic embodiments, environments, or significantly varied downstream tasks.   
Not suitable for interaction with humans.

# Factors

•Factors include varying backgrounds, lighting, scenes, base position, and novel natural language tasks. Hardware factors include camera and robot embodiment.

# Metrics

•Evaluation metrics include seen task performance, unseen task performance, robustness to backgrounds and distractors, and performance in long-horizon scenarios. Each measures the success rate of the model performing natural language specified tasks with randomized objects and object locations and varying scenes.

# Training Data

•Trained on 130k tele-operation demonstrations over 13 robots and 744 tasks.   

<table><tr><td>Skill</td><td>Count</td><td>Description</td><td>Example Instruction</td></tr><tr><td>Pick Object</td><td>130</td><td>Lift the object off the surface</td><td>pick iced tea can</td></tr><tr><td>Move Object Near Object</td><td>337</td><td>Move the first object near the second</td><td>move pepsi can near rxbar blueberry</td></tr><tr><td>Place Ob ject Upright</td><td>8</td><td>Place an elongated object upright</td><td>place water bottle upright</td></tr><tr><td>Knock Object Over</td><td>8</td><td>Knock an elongated object over</td><td>knock redbull can over</td></tr><tr><td>Open / Close Drawer</td><td>6</td><td>Open or close any of the cabinet drawers</td><td>open the top drawer</td></tr><tr><td>Place Object into Receptacle 84</td><td></td><td>Place an object into a receptacle</td><td>place brown chip bag into white bowl</td></tr><tr><td>Pick Object from Receptacle 162 and Place on the Counter</td><td></td><td></td><td>Pick an object up from a location and then pick green jalapeno chip bag from paper</td></tr><tr><td></td><td></td><td>place it on the counter</td><td>bowl and place on counter</td></tr><tr><td>Additional tasks</td><td>9</td><td>Skills trained for realistic, long instructions</td><td>pull napkin out of dispenser</td></tr><tr><td>Total</td><td colspan="2">744</td><td></td></tr></table>

# Evaluation Data

Evaluated on real-world randomized scenes and over 3000 total rollouts in the environment it was trained on as well as two new office kitchen environments.

# Quantitative Analyses

•RT-1 shows high-performance and robustness and can learn from heterogenous data.

![](images/13.jpg)

# Ethical Considerations

•Early research, model has not yet been evaluated for suitability to use outside of its current research setting.

# Caveats and Recommendations

•While the current model covers only a small portion of possible robotic manipulation tasks, it presents a recipe for scalable robotic learning and an architecture that shows favorable generalization and data absorption properties.

Figure 7: Model Card for RT-1.

As robot learning systems become more capable and the number of instructions they can handle increases, evaluation of these models becomes difficult (Kalashnikov et al., 2021a; Jang et al., 2021). This is an important consideration not only for evaluating different model classes and data distributions during the development process, but also for selecting the most performant model checkpoints for a particular training run. While there have been a number of proposed solutions to this problem (Dudík et al., 2011; Irpan et al., 2019; Hanna et al., 2017), mostly known in the offline reinforcement learning literature as "off-policy evaluation", it still remains an open research challenge to evaluate multi-task robot learning systems at scale.

In this work, we propose leveraging simulation for "real to sim" transfer as a scalable tool that provides an approximate estimate of model performance during training across many real tasks. We run policies trained from real data in a simulator to test the ful rollout performance. Note that all of our training data comes from the real world (except the experiment in Section 6.3), and the simulator is used only for model selection. To accomplish this, we expand the simulation environment proposed by Lee et al. (2022b) to support 551 of the tasks described in Section 5.2. For each of these tasks, we define a set of scene setup randomizations, robot pose randomizations, and success detection criteria. To bridge the visual distribution shift between the real world and the simulation, we train a RetinaGAN (Ho et al., 2020) model that transforms simulated images into realistic looking images. Then, we deploy policies trained on real data directly into these simulation environments by applying RetinaGAN visual transformations at each timestep and measuring rollout simulated task success rates.

While models trained only on real world data perform better in the real world than they do in simulation, we find that the simulation success rates of high-performing real world policies are higher than the smulatio success rates of lowperorming real wor polics. In other words, the orri of simulation policy success rates are informative for predicting the ordering of real world policy success rates. We note that in this real-to-sim evaluation setting, we have a less strict requirement for simulation accuracy compared to sim-to-real settings; as long as simulation success rates are directionally correlated with real success rates, we can accept a moderate or even high gap between real and simulation success rates.

We present example camera images from simulation as well as their RetinaGAN-based transformations in Fig. 8.

![](images/14.jpg)  
Figure 8: Example camera images showcasing raw simulation, simulation with RetinaGAN applied, and the real world.

Figure 9 shows the growth of data, number of tasks, and the success rate of the policy over time. The number of tasks/instructions that our system is capable of grows over time as more data is collected. The same is true with the performance of seen tasks. One of the important aspects of the future work is develop techniques that allow us to grow the data as well as the robots performance and general capabilities at a faster rate.

![](images/15.jpg)  
Figure 9: The growth of data, number of tasks, and seen instruction performance over time.

# D EXPERIMENTS

# D.1 Evaluation Details

In Section 6.2, we study the zero-shot generalization capabilities of RT-1 to difficult scenarios not present in the training dataset. To fairly evaluate different ablations of RT-1 as well as baseline policies, we design standardized evaluation procedures that cover a range of incremental difficulty levels.

Seen tasks. We evaluate on 744 tasks present in the training dataset. The breakdown between 12 skills is shown in Table 1. For all "Seen" evaluations, we use the same classroom setting used for data collection as described in Section 5.2. For each policy, we report a single representative metric that takes a skill-weighted average across individual skill evaluations.

Unseen tasks. We evaluate policy performance on 53 tasks that are held out during training. While the unseen instructions' specific combinations of skills and objects are not seen during training, other combinations of the same skills and objects are present in the training set. We evaluate these unseen tasks in the same environment and the same randomization procedure as the Seen tasks. A full list of these unseen tasks is shown in Table 8.

Distractor robustness. We test three tasks ("pick coke can", "place coke can upright", "move coke can near green rice chip bag") with incrementally more distractor objects added to the scene. The easy setting includes 0, 2, or 5 distractor objects. The medium setting includes 9 distractor objects, but the coke can is never obscured. The hard setting includes 9 distractor objects, but the scene is more crowded and the coke can is partially occluded. Both the medium are hard setting are more difficult than scenarios in the training dataset, which contained between 0 and 4 distractors. Examples of these difficulty settings and policy evaluation rollouts are shown in Figure 12.

Background robustness. We test six tasks ("pick coke can", "move blue chip bag near orange, "knock redbull can over", "pick green jalapeno chip bag", "move sponge near brown chip bag","place redbull can upright") with incrementally more challenging backgrounds and counter textures. In the easy setting, we utilize the same background environments and counter textures as the training dataset. In the medium setting, we utilize the same background environment but add a patterned tablecloth to change the counter texture. In the hard setting, we utilize a brand new kitchen environment with a new countertop; this changes the counter texture, drawer material and color, and background visuals. Examples of these diffculty settings and policy evaluation rollouts are shown in Figure 10.

Realistic instructions. To study how RT-1 performs in more realistic scenarios, we propose an evaluation setting in a real office kitchen that is a dramatic shift from the original training classroom environment. We propose a variety of skills that combine aspects of the previous zero-shot evaluations, including adding new distractors, including new backgrounds, and new combinations of objects with skills. We refer to the easiest scenario as $L 1$ generalization, which introduces a new countertop and lighting condition but keeps the skills and objects the same. Next, $L 2$ generalization additionally adds novel distractor objects such as kitchen jar containers. Finally, $L 3$ generalization adds new objects or new locations such as near a sink. While some of these distribution shifts are tested in Section 6.2, these realistic instructions aim to test multiple dimensions simultaneously. Examples of these instructions are presented in Fig. 11.

![](images/16.jpg)  
Figure 10: "Backgrounds" evaluations focus on testing the performance of RT-1 on settings with different table textures and different backgrounds, such as those found in kitchens never trained on. These visual differences are quite pronounced, which in the most challenging case entails a new kitchen with different counter texture, different lighting conditions, different counter material, and a different background.

![](images/17.jpg)  
Figure 11: "Realistic instructions" evaluations propose realistic scenarios multiple distribution shifts that incrementally increase in difficulty. $L 1$ generalization introduces a new real office kitchen with new lighting conditions. $L 2$ generalization additionally adds unseen distractor objects. Finally, $L 3$ generalization includes new objects or objects in new locations, such as next to a sink.

# D.2 HETErogeneous Data

We also explore the limits of RT-1 for utilizing highly heterogeneous data. We demonstrate how RT1 can incorporate and learn from vastly different data sources and improve from such data without

#

<table><tr><td>Instruction pick coke can from top drawer and place on counter pick green can from top drawer and place on counter pick green rice chip bag from middle drawer and place on counter</td></tr><tr><td>pick redbull can from top drawer and place on counter</td></tr><tr><td>place 7up can into bottom drawer place brown chip bag into top drawer</td></tr><tr><td>place green can into middle drawer move 7up can near redbull can move apple near green rice chip bag</td></tr></table>

![](images/18.jpg)  
Figure 12: "Distractors" evaluations focus on diversifying initial scene configurations well beyond the distributions contained in the training dataset, which contain between 2 and 4 distractor objects. In the most challenging scenarios, the scene is extremely cluttered and contains occlusions for the objects of interest.

sacrificing its original-tasks performance across the varied tasks inherent in this data. To this end, we conduct two experiments: (1) RT-1 trained and tested on both real data and simulation data and (2) RT-1 trained across large datasets of different tasks, originally collected by different robots.

Absorbing simulation data. Table 9 shows the ability of RT-1, and baselines, to absorb both real and simulation data. To test this, we take all of the real demonstration data but we also provide additional simulation data that includes objects that the robot has never seen in the real world. We add a set of sim objects and only show them on a subset of tasks, specifically the picking tasks, in simulation. To accomplish this, we run our real2sim method described in Sec. C.3 to bootstrap a simulation policy from the real world policy that is then trained with multi-task RL (Kalashnikov et al., 2021a) with additional objects in simulation. From this process, we extract $5 1 8 \mathrm { k }$ successful trajectories of picking new objects and mix them with the real data that was used in the previous experiments. The goal of this experiment is to demonstrate that by expanding the dataset of simulation trajectories, we can benefit RT-1's generalization capabilities without sacrificing the original training performance  a desired property of an absorbent model.

To evaluate the properties of this model, we specify different generalization scenarios: for seen skills with real objects the training data has real data of that instruction (i.e., performance on seen tasks), for seen skills with sim objects the training data has sim data of that instruction (e.g. "pick up a sim object", which was present in sim), and for unseen skills with sim objects the training data has sim data of that object but there are no examples of the instruction describing the skill with that object either in sim or in real (e.g., "move a sim object to apple", even though the robot has only practiced in picking that sim object and not moving it near other objects). All evaluations are done in the real world but to limit the number of instructions evaluated, we focus on pick and move-to skills.

We find in Table 9 that for RT-1, we do not lose performance adding simulation data compared to the Real Only dataset. We do however, see a significant increase in performance (from $23 \%$ to $87 \%$ on objects and tasks seen only in simulation, to approximately the performance of the those in real, demonstrating an impressive degree of domain transfer. We also see a significant increase in performance on unseen instructions from $7 \%$ to $33 \%$ ; impressive given the object in question has never been seen in real and the instruction never seen at all. Overall, we find that RT-1 is able to efficiently "sponge up" new data, even from a very different domain.

Table 9: Experimental results for incorporating simulation data in RT-1. Adding simulation data does not impact the performance on real objects, while significantly improving real performance on objects that were only introduced in simulation.   

<table><tr><td></td><td></td><td></td><td colspan="2">Real Objects Sim Objects (not seen in real)</td></tr><tr><td>Models</td><td>Training Data</td><td>Seen Skill w/ Objects</td><td>Seen Skill w/ Objects</td><td>Unseen Skill w/ Objects</td></tr><tr><td>RT-1</td><td>Real Only</td><td>92</td><td>23</td><td>7</td></tr><tr><td>RT-1</td><td>Real + Sim</td><td>90</td><td>87</td><td>33</td></tr></table>

![](images/19.jpg)

Absorbing data from different robots. To push the data absorption limits of RT-1, we conduct an additional set of experiments where we combine two data sources that originate from different robots: Kuka IIWA as well as the Everyday Robots mobile manipulators used in the experiments so far. The Kuka data contains all the successful examples collected in QT-Opt (Kalashnikov et al., 2018), which corresponds to 209k episodes, where the robot was indiscriminately grasping objects in a bin (see an example of a Kuka episode in Table. 10). Our goal in this experiment is to analyze whether the performance on the RT-1 tasks drops when adding the additional data and, more importantly, whether we can observe any transfer from data collected by a different robot morphology.

We would like to emphasize the difficulty of this setting by noting the major differences between the datasets. Not only are the robots that collected the data different in appearance and action space, but also the environment they were deployed in has different appearance and dynamics. In addition the QT-Opt data presents a completely different action distribution  it was collected by an RL agent as opposed to human demonstrations present in our dataset.

To mix the Kuka data together with the RT-1 data, we first transform the original Kuka 4-DOF action space into the same action space as RT-1, namely we set the rolland pitch to 0, while keeping the yaw values that were present in the original Kuka data. In addition, we transform the binary gripper-close command into a continuous gripper-closedness command that is present in the RT-1 data. We also need text instructions corresponding to the task performed and since the Kuka data does not contain the name of the object that was grasped, we relabel al the data to the "pick anything" instruction. With these modifications, we mix both datasets with the 2:1 (RT-1 data : Kuka data) ratio and train RT-1 to obtain the final model.

To test whether RT-1 can effectively absorb these two very different datasets, we evaluate the performance on the original RT-1 tasks (in this case, we also focus on "pick" and "move to" skills), which we refer to as the standard "Classroom eval", as well as the performance on the newly constructed tasks that reflect the bin-picking setup present in the Kuka data, which we refer to as the "Bin-picking eval". For the Bin-picking eval to be close to the original dataset, we put in the same looking bin for the objects as well as modify the robot to be similar to the Kuka manipulators by addng extra wires and coloring the gripper gray. For al of the evaluations we use the Everyday Robots robot with the picking commands and evaluate it based on 72 grasping trials.

The results are presented in Table 10. We observe that the model that mixes the RT-1 data and the Kuka data has only a minimal decrease in the original tasks' performance (i.e. Classroom eval), i.e. $2 \%$ .Even more importantly, in the Bin-picking eval, we observe that the model trained on multirobot data performs at $3 9 \%$ compared to the $2 2 \%$ of the model that was trained only on the RT-1 data. This is a $1 7 \%$ performance difference (almost $2 \mathbf { x }$ ). Additionally, RT-1 trained on Kuka bin-picking data and evaluated on the bin-picking tasks with the Everyday Robots (EDR) robot achieves $0 \%$ performance, confirming that it is difficult to transfer a behavior from another robot morphology. However, mixing the data from both robots allows RT-1 to infer the correct actions of the EDR robot even when faced with the states observed by Kuka robots. This is achieved without explicit demonstrations of bin-picking on EDR robot and by taking advantage of past experiences collected by Kuka robots. These results indicate that RT-1's absorption properties also include the ability to

<table><tr><td>Models</td><td>Training Data</td><td>Classroom eval Bin-picking eval</td><td></td></tr><tr><td>RT-1</td><td>Kuka bin-picking data + EDR data</td><td>90</td><td>39</td></tr><tr><td>RT-1</td><td>EDR only data</td><td>92</td><td>22</td></tr><tr><td>RT-1</td><td>Kuka bin-picking only data</td><td>0</td><td>0</td></tr></table>

![](images/20.jpg)

Table 10: Experimental results for mixing data from two different robots. Incorporating Kuka bin-picking data from QT-Opt (Kalashnikov et al., 2018) in RT-1 minimally impacts the standard classroom evaluation performance and results in almost a $2 \mathbf { x }$ improvement in generalization to the Bin-picking evaluation (that is similar to the setup in the Kuka data) on the Everyday Robots manipulator. This demonstrates an effective transfer across two different robot morphologies.

acquire new skills through observing other robots' experiences and present an exciting avenue of future work where we combine many more multi-robot datasets to enhance the robot capabilities.

# D.3 Long-Horizon Evaluation Details

In addition to short-horizon individual skill evaluations shown in previous sections, we also evaluate how RT-1 performs in a long-horizon realistic kitchen setting that chains multiple manipulation and navigation skills to accomplish natural language instructions within the SayCan framework (Ahn et al., 2022). A list of long-horizon instructions used for these evaluations is listed in Table 12.

The success rate of long-horizon tasks decreases exponentially with the length of the task, so high success rates in manipulation skill are particularly important. Furthermore, as mobile manipulation tasks require both navigation and manipulation, the policies ability to be robust to base position is crucial. Since SayCan combines many low-level instructions to perform high-level instructions, the number of possible high-level instructions increases combinatorially with instructions, so the skill-breadth of RT-1 can be fully seen.

SayCan works by grounding language models in robotic affordances and it leverages few-shot prompting to break down a long horizon task expressed in natural language to a sequence of low level skills. An example of long horizon task would be "Bring me two different sodas", and one feasible plan would be "1. find a coke, 2. pick up the coke, 3. bring it to you, 4. put down the coke, nd a pepsi, 6. pick up the pepsi, 7. bring it to you, 8. put down the pepsi, 9. done." To obtain the affordance function we use value functions trained with MT-OPT (Kalashnikov et al., 2021a). For a detailed description of SayCan algorithm please refer to (Ahn et al., 2022).

Since the focus of this paper is acquisition of many generalizable skills, we focus our evaluation on one subset of tasks presented in Ahn et al. (2022). It is the long-horizon family of tasks, involving 15 instructions, each instruction requires an average of 9.6 steps to complete, and involves an average of 2.4 manipulation skills per instruction. A full list of the instructions can be found in Table 12.

We compare against 3 baselines. 1) SayCan with BC-Z, which uses SayCan planning algorithm with BC-Z as manipulation policy, 2) SayCan with Gato, which uses SayCan planning algorithm with Gato as manipulation policy, 3) Originally reported SayCan results, which use SayCan planning algorithm with BC-Z, but since it uses a slightly different prompt, the planning success rate is lower. We reimplemented 3) in 1) for a fair comparison.

As shown in Table 11, except for original SayCan, all methods get $87 \%$ as planning success rate, and RT-1 performs the best, with $67 \%$ execution success rate in Kitchen1. Kitchen2 constitutes a much more challenging generalization scene, since the Robot Classroom training scenes are modeled after Kitchen1 (see the pictures of the kitchens in Fig. 2). Due to this generalization difficulty, SayCan with Gato is not able to finish any long horizon task, and SayCan with BC-Z is able to achieve a success rate of $13 \%$ . The original SayCan paper did not evaluate performance in a new kitchen. Surprisingly, the manipulation performance does not see a visible drop from Kitchen1 to Kitchen2

for our method. In the supplementary video, we show that this enables us to operate unseen drawers in Kitchen2, and that we can use SayCan-RT1 to plan and execute ultra-long horizon tasks, with as many as 50 steps.   
Table 11: SayCan style long horizon tasks in Kitchen1 and Kitchen2. (\*Original SayCan eval uses a slightly different prompt so the planning success rate is lower.)   

<table><tr><td></td><td colspan="2"></td><td colspan="2">SayCan tasks in Kitchen1 SayCan tasks in Kitchen2</td></tr><tr><td></td><td>Planning</td><td>Execution</td><td>Planning</td><td>Execution</td></tr><tr><td>Original SayCan (Ahn et al., 2022)*</td><td>73</td><td>47</td><td>-</td><td>-</td></tr><tr><td>SayCan w/ Gato (Reed et al., 2022)</td><td>87</td><td>33</td><td>87</td><td>0</td></tr><tr><td>SayCan w/ BC-Z (Jang et al., 2021)</td><td>87</td><td>53</td><td>87</td><td>13</td></tr><tr><td>SayCan w/ RT-1 (ours)</td><td>87</td><td>67</td><td>87</td><td>67</td></tr></table>

# D.4 MODEL ABLATIONS

# What are the important and practical decisions in the design of the model and how do they affect performance and generalization?

To answer this question, we perform a set of ablations over different design decisions in RT-1. We aim to test a number of hypotheses that will help us disambiguate where the benefits of our method come from. Possible hypotheses about the source of improvement include: (i) the capacity and expressiveness of our model, which we verify by ablating the model size, trying other architectures (e.g., by removing the Transformer component); (i) the particular action representation, which makes it easy to represent complex multi-modal action distributions, which we test by switching to continuous (normally distributed) actions, as well as by ablating the auto-regressive action representation; (ii) the ImageNet pre-trained initialization of the components, which we test by initializing the model's weights randomly; and (iv) access to the short history, which we test by excluding observation history. More concretely, we ablate our model by (1) decreasing the model size (from 35M to 21M parameters), (2) removing the Transformer architecture (using a pre-trained EfficientNet instead), (3) using a continuous instead of discrete action space (using an MSE loss and multivariate normal output), (4) auto-regressively conditioning on actions, (5) removing ImageNet pre-training of the FiLM EffcientNet, and (6) removing history (reducing the sequence of six images as input to a single image). For each ablation we compare on the axes of performance on seen tasks, performance on unseen tasks, as well as inference speed and robustness to distractors and backgrounds (with a more detailed description of each category in Section 6.1 and Appendix D.1).

Table 13 shows the results of each ablation and the delta performance compared to the full RT-1. RT-1 achieves impressive performance on tasks and new environments, and particularly outperforms baselines on the most challenging robustness problems. We also find that each design decision is important, though at varying levels. We first evaluate a model that replaces the per-dimension discretized action representation in our model with a more standard continuous Gaussian distribution. We observe a significant decline in performance from this modification. The per-dimension discretization allows our model to represent complex multi-modal distributions, while the Gaussian distribution captures only a single mode. These results suggest that this standard and popular choice is highly suboptimal with the more complex and diverse demonstration data used by our system. ImageNet pre-training is particularly important for model generalization and robustness, decreasing the unseen task performance rate by $33 \%$ , as a result of the large and diverse visuals of the ImageNet dataset. Adding history has an impact primarily on generalization to distractors, while removing the Transformer component has a uniform but small negative impact across the seen tasks, unseen tasks and distractors. In order to keep the ImageNet pre-training while reducing the model size, we reduce the number of parameters only by $40 \%$ (from 31M to 25M). Resulting performance drops across training and generalization tasks but not as much as in other ablations. Finally, autoregressively conditioning on actions, as used in (Reed et al., 2022; Chen et al., 2021; Lee et al., 2022a), did not benefit performance and slowed inference by more than $2 \mathbf { x }$ .

As described in Sec. 5.1, in order to run large Transformer models on real robots, we require a model that supports fast inference for real-time operation. Note that in order to achieve our target control rate of $\mathrm { 3 H z }$ (described in Sec. 5.1), we also need to consider other sources of latency in the pipeline, such as the camera latency and communication overhead. However, these factors will be constant for all the models, and therefore we focus our evaluation on just the network inference time. The last column of Table 13 shows the inference speed of all the models. RT-1 is almost an order of magnitude faster than Gato with a similar number of parameters, but it is also considerably slower than a ResNet-based BC-Z. In terms of the different ablations of our model, we observe that the biggest slow-down is caused by including auto-regressive actions ( ${ \sim } 2 \mathbf { x }$ slow-down), and since this does not significantly influence the performance, the final version of RT-1 does not generate actions auto-regressively.

Table 12: List of SayCan instructions evaluated in Sec. 6.4   

<table><tr><td colspan="2">Instruction</td></tr><tr><td colspan="2">How would you put an energy bar and water bottle on the table How would you bring me a lime soda and a bag of chips Can you throw away the apple and bring me a coke How would you bring me a 7up can and a tea? How would throw away all the items on the table? How would you move an multigrain chips to the table and an apple to the far counter? How would you move the lime soda, the sponge, and the water bottle to the table? How would you bring me two sodas? How would you move three cokes to the trash can? How would you throw away two cokes? How would you bring me two different sodas? How would you bring me an apple, a coke, and water bottle? I spilled my coke on the table, how would you throw it away and then bring me something to help clean?</td></tr></table>

<table><tr><td></td><td></td><td></td><td colspan="3">Distractors</td><td>Backgrounds</td><td></td><td></td></tr><tr><td>Model</td><td>Seen Tasks Unseen Tasks</td><td></td><td>All</td><td>Easy</td><td>Medium</td><td>Hard</td><td>All</td><td>Inference Time (ms)</td></tr><tr><td>Gato (Reed et al., 2022)</td><td>65 (-32)</td><td>52 (-24)</td><td>43 (-40)</td><td>71</td><td>44</td><td>29</td><td>35 (-24)</td><td>129</td></tr><tr><td>BC-Z (Jang et al., 2021)</td><td>72 (-25)</td><td>19 (-57)</td><td>47 (-36)</td><td>100</td><td>67</td><td>7</td><td>41 (-18)</td><td>5.3</td></tr><tr><td>BC-ZXL</td><td>56 (-41)</td><td>43 (-33)</td><td>23 (-60)</td><td>57</td><td>33</td><td>0</td><td>35 (-24)</td><td>5.9</td></tr><tr><td>RT-1 (ours)</td><td>97</td><td>76</td><td>83</td><td>100</td><td>100</td><td>64</td><td>59</td><td>15</td></tr><tr><td>RT-1 w/o big model</td><td>89 (-8)</td><td>62 (-14)</td><td>77 (-6)</td><td>100</td><td>100</td><td>50</td><td>53 (-6)</td><td>13.5</td></tr><tr><td>RT-1 w/o pre-training</td><td>84 (-13)</td><td>43 (-33)</td><td>60 (-23)</td><td>100</td><td>67</td><td>36</td><td>41 (-18)</td><td>15</td></tr><tr><td>RT-1 w/ continuous actions</td><td>68 (-29)</td><td>43 (-33)</td><td>37 (-46)</td><td>71</td><td>67</td><td>0</td><td>35 (-24)</td><td>16</td></tr><tr><td>RT-1 w/ auto-regressive actions</td><td>85 (-12)</td><td>71 (-5)</td><td>67 (-16)</td><td>100</td><td>78</td><td>43</td><td>65 (+6)</td><td>36</td></tr><tr><td>RT-1 w/o history</td><td>82 (-15)</td><td>62 (-14)</td><td>50 (-33)</td><td>71</td><td>89</td><td>14</td><td>59 (+0)</td><td>15</td></tr><tr><td>RT-1 w/o Transformer</td><td>86 (-13)</td><td>62 (-14)</td><td>67 (-16)</td><td>100</td><td>100</td><td>29</td><td>59 (+0)</td><td>26</td></tr></table>

![](images/21.jpg)  
Table 13: Various model ablations of RT-1 across seen tasks, generalization to unseen tasks, and robustness to distractors and backgrounds.

# D.5 SUMMARY AND ANALYSIS

In this section, we summarize some of our findings and propose intuition for RT-1's high performance, generalization, and robustness. First, ImageNet pretraining (along with Universal Sentence Encoder language embedding) has a large impact particularly on unseen tasks. We observe that RT-1 inherits some of the knowledge that results from the generality and diversity of the datasets these models were trained on. Second, continuous actions have a large impact across all aspects of performance. This has been previously observed and may be due to the ability to represent more complex action distributions  the per-dimension discretization allows our model to represent complex multi-modal distributions, while the Gaussian distribution captures only a single mode. Third, given such expressive multitask models, data diversity has a larger impact than data size. Indeed, even datasets collected in simulated environments or from different robotic embodiments can be leveraged by RT-1, opening avenues for new regimes of data collection.

Finally, RT-1 fuses language into the image pipeline early via FiLM conditioning, compared to e.g., Gato's late fusion. This enables image tokens that focus only on relevant features for the instruction at hand, which may be the cause of poor distractor performance for Gato. Figure 13 visualizes the attention during rollouts of RT-1. We see that the attention is focused on relevant features and particularly on interaction between the gripper and the object of interest. The bottleneck of attention layers such as these results in a compact representation which effectively ignores distractors and varying backgrounds.

![](images/22.jpg)  
Layer 2, Head 6   
pick green jalapeno chip bag from middle drawer and place on counter"   
place rxbar blueberry in bottom drawer"   
open middle drawer"   
Figure 13: In this figure we show the attention map of the RT-1 policy. Different layers and heads generally focus on different part of the image. Most commonly, they focus on the parts of the scene with the richest interaction affordances, such as graspable objets. For example, Layer 2 Head 6 focuses on the jalapeno chips and pepsi can in grasping tasks; and Layer 4 Head 2 focuses on the drawer in drawer opening tasks.