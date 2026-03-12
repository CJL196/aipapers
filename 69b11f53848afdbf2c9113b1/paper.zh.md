# 视觉-语言-动作 (VLA) 模型：概念、进展、应用与挑战

兰詹·萨普科塔、杨·曹、康斯坦丁·I·鲁梅利奥蒂斯、马诺杰·卡尔基 康奈尔大学，生物与环境工程系，美国纽约伊萨卡 香港科技大学，计算机科学与工程系，香港 佩洛波尼萨大学，信息学与电信系，希腊

# 摘要

Viec，VLMs 项目代码库可在 GitHub 上获取（源链接）关键词：视觉-语言-动作、动作标记化、人工智能、机器人技术、视觉-语言模型

# . 引言

在视觉-语言-动作（VLA）模型开发之前，机器人技术和人工智能的进展主要集中在不同的领域：能够获取、解释和识别图像的视觉系统，能够理解和生成文本的语言系统，以及能够控制运动的动作系统。这些孤立的系统在各自的领域中表现良好，但在协同工作、推广到新场景或适应现实世界挑战的复杂性和不可预测性方面存在困难。

如图1所示，传统计算机视觉模型主要基于卷积神经网络（CNN），专为狭窄指定的任务设计，例如物体检测或分类，这些模型需要大量标注数据集，并且在环境或目标稍有改变时需要繁琐的重训练。这些视觉模型能够“看到”（例如，如图1所示识别果园中的苹果），但缺乏对语言的理解能力，也无法将视觉洞察转化为期望的行动。语言模型，尤其是大语言模型（LLMs），彻底改变了基于文本的理解和生成；然而，它们仍然局限于处理语言，无法感知或推理物理世界的能力（图1中的“果园里的熟苹果”就是这种局限性的例证）。与此同时，基于动作的机器人系统，依赖于手工设计的策略或强化学习，使得特定行为（如物体操作）得以实现，但需要繁琐的工程设计，并无法推广到专门设计场景以外的情况。

![](images/1.jpg)  

Figure 1: Evolution from isolated modalities to unified Vision-Language-Action models. Integrated perception, language, and action enable adaptive, generalizable embodied intelligence.

尽管通用视觉语言模型取得了显著的多模态理解的进展，通过将视觉和语言结合在一起，但仍然存在明显的整合缺口：无法基于多模态输入生成或执行连贯的动作。正如图 1 所示， most AI 系统专注于一种或两种模态，例如视觉-语言、视觉-动作或语言-动作，难以将三者完全整合到统一的端到端框架中。因此，机器人可以视觉上识别物体（“苹果”）、理解相应的文本指令（“拿起苹果”）或执行预定义的运动动作（抓握），但将这些能力整合并表现为流畅、适应性强的行为却缺失。最终导致的结果是一个无法灵活适应新任务或环境的流程，造成脆弱的泛化能力和劳动密集型的工程工作。这一局限性突显了具身 AI 的一个关键瓶颈：缺乏能够共同感知、理解和行动的系统，使得智能自主行为仍然是一个具有挑战性的目标。

![](images/2.jpg)  

Figure 2: Mind map of core VLA concepts. Each color-coded branch highlights a foundational dimension: definitions (foundation), historical evolution, multimodal integration, tokenization and encoding, learning paradigms, and adaptive execution in embodied settings.

迫切需要弥合这些差距催生了视觉语言行动（VLA）模型。VLA 模型于 2021-2022 年间构思，由谷歌 DeepMind 的机器人变压器 2（RT-2）等先驱性努力推动，推出了一种转换架构，将感知、推理和控制统一在一个框架内。作为对图 2 中概述的局限性的解决方案，VLA 集成了视觉输入、语言理解和运动控制能力，使具身智能体能够感知周围环境、理解复杂指令并动态执行适当的动作。早期的 VLA 方法通过扩展视觉语言模型以包括动作词元（即机器人运动指令的数字或符号表示）实现了这种集成，从而使模型能够从配对的视觉、语言和轨迹数据中学习。这种方法创新显著提高了机器人在未知物体上的概括能力、解释新语言指令的能力，以及在非结构化环境中进行多步推理的能力。

VLA 模型代表了统一多模态智能发展的变革性步骤，克服了长期以来将视觉、语言和行动视为独立领域的局限性。通过利用整合视觉、语言和行为信息的互联网规模数据集，VLA 使机器人不仅能够识别和描述其环境，还能在复杂动态环境中进行情境推理并执行适当的行动。从孤立的视觉、语言和行动系统到集成的 VLA 范式的进展（如图 2 和图 3 所示）捕捉了朝向真正适应性和可推广的具身智能体发展的根本转变。考虑到这一范式的变革潜力，对现有文献进行全面且批判性的信息评审是及时且必要的。首先，此类评审对于澄清 VLA 与其前身之间的基础概念和架构原则是必须的。其次，它提供了对该领域快速进展和关键里程碑的结构化叙述，使研究人员和从业者能够理解算法和技术进步的轨迹。第三，深入的评审对于绘制从家用机器人到工业自动化和辅助技术等多种现实应用的广泛范围至关重要，VLA 在这些领域已经展现出变革潜力。此外，通过批判性地审视当前面临的挑战，如数据效率、安全性、泛化能力和伦理考量，该评审识别出必须解决的障碍，以便实现广泛部署。最后，综合这些见解有助于向更广泛的人工智能和机器人社区传达新兴研究方向和实际考虑，促进合作与创新。在本评审中，我们系统地分析 VLA 模型的基础原则。此外，我们讨论其发展进程和技术挑战。我们的目标是整合对 VLA 的当前理解和应用，同时识别其局限性并提出未来发展的方向。评审以对关键概念基础的详细审查开始，包括 VLA 模型的定义、历史演变、多模态集成机制以及跨视觉、语言和行动的一体化词元化和表示策略。这些概念描述为理解 VLA 在不同模态中的结构和功能奠定了基础。在此描述的基础上，我们呈现了最近进展和训练效率策略的统一视角。这包括在 VLA 模型中采用和扩展的架构创新，以及最初在更广泛的机器学习和机器人领域中开发的数据高效学习框架、参数高效建模技术和模型加速策略。这些进展对于将 VLA 系统扩展到现实应用至关重要。

![](images/3.jpg)

接下来，我们将全面讨论当前VLA系统所面临的局限性（图3），这些局限性不仅反映了具身人工智能和机器人领域的更广泛挑战，而且由于视觉、语言和行动的紧密整合而以独特而复杂的形式出现。讨论的局限性包括推理瓶颈、安全问题、高计算需求、有限的泛化能力和伦理影响。我们不仅强调这些紧迫的挑战，还提供了针对性解决方案的分析讨论。这三幅图共同提供了可视化的框架，支持了本综述中呈现的文本分析。通过概述概念景观、最近的创新和开放挑战，本研究旨在指导未来的研究，并推动更为强大、高效和具有伦理基础的VLA系统的发展。

图4总结了本综述的整体结构和逻辑流程，并展示了手稿的组织方式，以提供对VLA研究的全面和系统的分析。如图所示，论文从基础概念开始，逐步过渡到最新进展、应用、挑战和未来研究方向，确保各部分之间叙述的一致性。为构建这一架构，采用了主要关键词“VisionLanguage—Action”和“VisionLanguage Models”进行广泛而严格的文献检索，并结合常用缩写“VLA”。这些关键词用于从主要的学术和技术数据库中检索备选研究，包括Hugging Face、arXiv、ScienceDirect、Nature、IEEE Xplore、Wiley和Springer Nature。通过人工筛选，进一步精炼了筛选出的文献，以确保其相关性、技术深度和与本综述范围的一致性。仅保留直接有助于理解VLA概念、方法进展、应用领域和开放挑战的文章。这一多阶段的过滤过程使得文章能够平衡基础研究和最前沿成果的覆盖，同时避免外围或松散相关的研究。因此，图4不仅反映了论文的主题组织，也体现了其基本的综述方法论。本文遵循结构化的层次组织，系统地阐述VLA模型的基础、演变和影响，如图4所示。介绍部分动机在于探讨具身智能与VLA的出现，随后是建立核心原则的概念部分，包括多模态融合、词元化、学习范式和实时控制。在这些基础上，进展部分考察了架构创新、训练和效率提升以及参数高效加速策略。应用部分将这些发展与真实世界领域结合，包括类人机器人、自动驾驶汽车、医疗保健、农业、工业系统和交互式AR导航。接下来是对关键挑战的专注分析，包括实时推理、安全性、泛化、系统集成和伦理考虑。论文最后列出了未来的路线图，提炼出在持续学习、可扩展性、可解释性和具身智能等方面的跨领域研究方向。

![](images/4.jpg)  

Figure 4: Flow and structure of this paper, from conclusion and future roadmap back through applications, progress, concepts, to introduction.

# 2. 视觉-语言-动作模型的概念

VLA模型代表了一类智能系统，它们共同处理视觉输入，解析自然语言指令，并生成可在动态环境中物理机器人硬件上实现的可执行动作表示。从技术上讲，VLA将视觉编码器（例如CNN、ViT）、语言模型（例如LLM、变换器）以及策略模块或规划器结合起来，以实现任务条件控制。这些模型通常基于视觉-语言模型中建立的多模态融合技术，如交叉注意力、连接嵌入或词元统一，并扩展它们以对齐感知观察、语言指令和动作表示。与传统的视觉运动管道不同，VLA支持语义基础 [154]，使得上下文感知推理 [228]、可供性检测 [79] 和时间规划 [141] 成为可能。一个典型的VLA模型通过摄像头或传感器数据观察环境，解析以语言表达的目标（例如“捡起红色苹果”）（图5），并输出可以由自动化系统实施的低级或高级动作序列，以执行该动作。最近的进展整合了模仿学习、强化学习或检索增强模块，以提高样本效率和泛化能力。本综述考察了VLA模型如何从基础的融合架构演变为能够在机器人技术、导航和人机合作中进行现实世界部署的通用智能体。VLA模型是多模态人工智能系统，将视觉感知、语言理解和物理动作生成统一为一个单一框架。这些模型使机器人或AI智能体能够解析感知输入（例如图像、文本），理解上下文含义，并在现实世界环境中自动执行任务——这一切都是通过端到端的学习和动作，而不是孤立的子系统。如图5所示，VLA模型弥合了早期机器人和AI系统中视觉识别、语言理解和/或动作执行之间的历史脱节，从而扩展了其能力。

# 2.1. 演变与时间线

从2022年到2025年，VLA模型的快速发展展示了三个不同的进化阶段：1. 基础整合（2022-2023）。早期的VLA通过多模态融合架构建立了基本的视觉运动协调。 [202] 首次将CLIP嵌入与运动原语结合，而 [181] 展示了在604个任务上的通用能力。 [19] 通过规模化模仿学习达到了 $9 7 \%$ 的操作成功率， [112] 通过基于变换器的规划引入了时间推理。到2023年， [299] 实现了视觉链式思维推理， [40] 通过扩散过程推进了随机动作预测。这些基础解决了低级控制问题，但缺乏组合推理的能力，即将复杂任务分解为可重用的、具有语义基础的子动作，并在新环境中重新组合，促使后续在可供性基础方面的创新 [287, 100]。

![](images/5.jpg)  
Learning Paradigms, and Adaptive Control and Real-Time Execution.

2. 专业化与具身推理（2024）。第二代视觉语言模型（VLA）融入了领域特定的归纳偏置。[269] 通过检索增强训练提高了少量样本适应能力，而 [280] 通过三维场景图整合优化了导航。[48] 引入了可逆架构以提高内存效率，且 [239] 通过物理信息感知注意力解决了部分可观察性问题。与此同时， [5] 通过物体中心的解耦改善了组合理解， [293] 通过多模态传感器融合将应用扩展至自动驾驶。然而，这些进步需要新的基准测试方法论 [254]。3. 泛化与安全关键部署（2025）。最新系统优先考虑鲁棒性和人类对齐。[274] 整合了形式验证用于风险感知决策，而 [53] 通过层次化的 VLA 展示了全身控制。[20] 优化了嵌入式部署的计算效率，[131] 则结合了神经-符号推理以进行因果推断。[129] 的可用性链和 [14] 的仿真到真实转移学习等新兴范式解决了跨具身挑战，而 [139] 通过自然语言嵌入将 VLA 与人机交互接口连接起来。图6展示了2022年至2025年间开发的45个 VLA 模型的发展演变的全面时间轴。最早的 VLA 系统包括 CLIPort [202]，

Gato [181]、RT-1 [19] 和 VIMA [112] 通过将预训练的视觉-语言表示与任务条件策略相结合，奠定了操作与控制的基础。这些早期的 VLA 系统之后，出现了 ACT [287]、RT2 [299] 和 VoxPoser [100]，这些系统集成了视觉连锁思维推理和效能基础。诸如扩散策略 [40] 和 Octo [218] 的模型引入了随机建模和可扩展的数据管道。在 2024 年，Deer-VLA [269]、ReVLA [48] 和 Uni-NaVid [280] 等系统增加了领域专门化和内存高效设计，而 Occllama [239] 和 ShowUI [139] 则解决了部分可观察性和用户交互。该轨迹继续向机器人技术集中，出现了 Quar-VLA [54] 和 RoboMamba [143]。近期创新强调了泛化和部署：SafeVLA [274]、Humanoid-VLA [53] 和 MoManipVLA [246] 集成了验证、全身控制和记忆系统。诸如 $\mathrm { G r 0 0 t } \ : \mathrm { N 1 }$ [14] 和 SpatialVLA [175] 的模型进一步架起了从模拟到真实转移和空间基础的桥梁。这个时间线展示了 VLAs 如何从模块化学习发展到通用、安全和具身的智能。

# 2.2. 多模态融合：从孤立管道到统一智能体

VLA模型出现的一个核心进展在于其能够进行多模态集成，即在统一架构内共同处理视觉、语言和动作。传统的机器人系统将感知、自然语言理解和控制视为离散模块，通常通过手动定义的接口或数据转换连接。例如，经典的基于管道的框架要求感知模型输出符号标签，这些标签随后被规划器映射到特定动作，通常需要领域特定的手工工程。这些方法缺乏适应性，在模糊或未知环境中表现不佳，无法将指令泛化到预定义模板之外。

![](images/6.jpg)  
grouping.

与此相比，现代视觉语言算法（VLA）利用大规模预训练编码器和基于变换器的架构实现端到端的模态融合。这一转变使模型能够在同一计算空间内解释视觉观察和语言指令，从而实现灵活且具上下文感知的推理。例如，在“选择成熟的苹果”这一任务中，视觉编码器——通常为视觉变换器（ViT）或ConvNeXt——解析场景以定位和分类相关物体（如水果、叶子、背景），并根据学习到的纹理、形状和上下文特征推断与成熟度相关的视觉线索，而不是依赖于固定的颜色假设。同时，语言模型，通常为T5、GPT或BERT的变体，将指令编码为高维嵌入。这些表示通过跨注意力或联合词元化方案进行融合，生成一个统一的潜在空间，为动作策略提供信息。这种多模态协同在CLIPort中首次得到有效展示，它以桌面场景的RGB图像和自然语言指令（例如“把蓝色方块放在红色方块上”）作为输入，利用CLIP进行语义定位编码，并通过卷积传输解码器输出像素级的取放动作分布。通过直接将视觉运动策略条件化于语言嵌入，CLIPort消除了显式的语言解析，实现了端到端的语言条件操控。类似地，VIMA通过使用变换器编码器共同处理以物体为中心的视觉词元和指令词元，推动了这一方法的进展，使其能够在空间推理任务中实现少量样本的泛化。最近的进展进一步推动了这种融合，通过引入时间和空间的基础。例如，VoxPoser利用体素级推理来解决3D物体选择中的歧义，通过组合预训练的视觉-语言模型和经典的运动规划器，显著实现了零-shot操控，无需特定任务的训练数据。而RT2则在一个统一的变换器中融合视觉-语言词元和动作表示，经过大规模互联网视觉-语言语料和来自RT-1数据集中超过10万个真实机器人示例的共同训练，使其能够实现对未见指令的零-shot泛化。另一个值得注意的贡献是Octo，它引入了一种记忆增强型变换器，该变换器在Open XEmbodiment数据集中收集的超过四百万条机器人轨迹上进行训练，支持长时间决策，并展示了联合感知-语言-动作学习的可扩展性。至关重要的是，VLA为现实世界中的基础对接挑战提供了稳健的解决方案。例如，Occllama通过基于注意力的机制处理遮挡物体引用，而ShowUI展示了自然语言接口，使非专业用户能够通过语音或输入命令操控智能体。这些能力之所以能够实现，是因为融合不仅限于表面级别，而是捕捉到模态间的语义、空间和时间的对齐。

# 2.3. 词元化与表征：视觉语言模型如何编码世界

VLA模型的核心创新在于其基于词元的表示框架，这使得它能够在感知、语言和物理动作空间之间进行整体推理。受自回归生成模型如变换器的启发，现代VLA通过离散词元对世界进行编码，将所有模态（视觉、语言、状态和动作）统一到一个共享的嵌入空间中。这使得模型不仅能够理解“需要做什么”（语义推理），还能够以完全可学习和可组合的方式理解“如何去做”（控制策略执行）。 前缀词元：编码上下文和指令：前缀词元作为VLA模型的上下文骨架。这些词元将环境场景（通过图像或视频）和伴随的自然语言指令编码成紧凑的嵌入，从而引导模型的内部表示。例如，如图7所示，在“将绿色方块放在红色托盘上”的任务中，杂乱的桌面图像通过视觉编码器（如ViT或ConvNeXt）进行处理，而指令则由大型语言模型（如T5或LLaMA）嵌入。这些词元随后被转换成一系列前缀词元，建立模型对目标和环境布局的初步理解。这种共享表示实现了跨模态的基础支持，使系统能够解析空间参考（如“在左侧”、“在蓝杯旁边”）和对象语义（“绿色方块”）。

![](images/7.jpg)  

Figure 7: A diagram illustrating the end-to-end tokenization and representation process in VLA models. Visual input (e.g., cluttered tabletop) is encoded by a vision encoder (e.g., ViT), while natural language instructions (e.g., "stack the green blocks") are processed by a language encoder (e.g., T5). The system fuses prefix, state, and action tokens through a transformer and autoregressively predicts motor actions.

状态 token：嵌入机器人的配置：除了感知外部刺激，变形体（VLA）还必须意识到其内部物理状态 [242, 143]。这通过使用状态 token 来实现，状态 token 编码了智能体配置的关节位置、力矩读数、抓手状态、末端执行器姿态，甚至附近物体的位置的实时信息 [126]。这些 token 对于确保情境意识和安全至关重要，尤其是在操作或移动过程中 [211, 105]。

图8展示了VLA模型如何利用状态词元在操作和导航场景中实现动态的、上下文感知的决策。在图8a中，一个机械臂部分伸展在一个脆弱物体附近。在这种情况下，状态词元通过编码实时的本体感知信息（如关节角度、夹持器姿态和末端执行器的接近度）发挥着关键作用。这些词元与视觉和基于语言的前缀词元不断融合，使变换器能够推理物理约束。因此，模型可以推断出碰撞即将发生，并相应调整电机指令，例如重新规划机械臂轨迹或调节输出力。在移动机器人平台中，如图8b所示，状态词元封装了空间特征，如里程计、激光雷达扫描和惯性传感器数据。这些对于地形感知的移动和障碍物规避至关重要。变换器模型将这种状态表示与环境和指令上下文相结合，以生成能够动态适应变化环境的导航动作。无论是在杂乱环境中抓取物体，还是在不平坦地形中自主导航，状态词元为情境意识提供了一种结构化机制，使自回归解码器能够生成精确、上下文敏感的动作序列，反映内部机器人配置和外部感知数据。

![](images/8.jpg)  

Figure 8: Illustrating how VLA models utilize prefix, state, and action tokens in real-world scenarios. In robotic manipulation, state tokens detect arm extension near fragile objects, enabling path adjustment. In navigation, they represent LiDAR and odometry data. The apple-picking task shows how prefix tokens guide goal understanding, while action tokens generate motion sequences for targeted grasping and execution.

• 动作词元：自回归控制生成：VLA 词元管道的最后一层涉及动作词元，这些词元由模型自回归生成，以代表运动控制中的下一步。每个词元对应于一个低级控制信号，例如关节角度更新、扭矩值、轮子速度或高级运动原语。在推理过程中，模型逐步解码这些词元，依赖于前缀和状态词元，有效地将 VLA 模型转变为语言驱动的策略生成器。这种形式化允许与现实世界的执行系统无缝集成，支持可变长度的动作序列，并通过强化学习或模仿学习框架实现模型微调。值得注意的是，像 RT-2 和 PaLM-E 这样的模型体现了这种设计，其中感知、指令和体现融合成统一的词元流。例如，在图 9 描绘的苹果采摘任务中，模型可能接收包含果园图像和文本指令的前缀词元。状态词元描述机器人的当前手臂姿态以及抓手是打开还是关闭。然后，动作词元逐步预测，引导机器人手臂朝向苹果，调整抓手取向，并以适当的力度执行抓取。这种方法的美在于，它使得传统上用于文本生成的变换器可以类似于生成句子那样生成物理动作序列，只不过在这里，句子是运动。

![](images/9.jpg)  

Figure 9: Ilustrating the process of how VLAs Encode the World. VLAs encode the world by converting vision, language, and sensor inputs into tokens, fusing them through cross-attention, predicting action sequences via transformers, and executing tasks with real-time feedback - enabling robots to interpret scenes, follow instructions, and adapt actions dynamically.

为了在机器人领域实现 VLA 范式，我们在图 9 中展示了一个结构化的管道，演示了多模态信息，特别是视觉、语言和自我感知状态，如何被编码、融合，并转化为可执行的动作序列。这个端到端循环使机器人能够理解复杂任务，例如“在绿色叶子附近摘取成熟的苹果”，并执行精确的、上下文敏感的操作。系统首先从多模态输入获取开始，收集三种不同的数据流：视觉观测（例如，RGB-D 帧）、自然语言命令和实时机器人状态信息（例如，关节角度或速度）。这些数据被独立地使用预训练模块进行词元化，转化为离散的嵌入。如图所示，图像通过视觉变换器（ViT）主干网络处理，以生成视觉词元，指令通过语言模型（如 BERT 或 T5）解析，生成语言词元，状态输入通过轻量级 MLP 编码器转化为紧凑的状态词元。这些词元随后使用跨模态注意力机制进行融合，在此过程中模型共同推理对象语义、空间布局和物理约束。融合后的表示形成了决策的上下文基础。在图 9 中，这被标记为多模态融合步骤。融合的嵌入传递到一个自回归解码器，通常是一个变换器，生成一系列动作词元。这些词元可能对应于关节位移、抓手力量调节或高级运动原语（例如，“移动到抓取姿势”、“旋转手腕”）。预测的动作词元随后被转化为低级控制命令，并由一个外部的、依赖硬件的执行循环执行，该循环与机器人控制器接口，以通过反馈更新的状态观测来闭合感知-动作循环，供下一个 VLA 推理步骤使用。这个闭环机制使模型能够实时动态适应扰动、物体位移或遮挡。

为提供清晰具体的实现细节，算法 1 形式化了 VLA 词元化过程。给定 RGB-D 帧 $I$、自然语言指令 $T$ 和关节角度向量 $\theta$，该算法生成一组可以按顺序执行的动作词元。图像 $I$ 通过 ViT 处理生成 $V$，即一组 400 个视觉词元。同时，指令 $T$ 由 BERT 模型编码为 $L$，即一序列 12 个语义语言词元。同时，机器人状态 $\theta$——包含关节角、末端执行器姿态和本体感知信号——通过多层感知机编码为一个紧凑的 64 维状态嵌入 $s$，为模型提供对机器人配置和物理约束的实时感知，从而在动作生成过程中保持同步。这些词元随后通过交叉注意力模块融合，生成一个共享的 512 维表示 $F$，捕捉必要的语义、意图和情境意识，以支持落实的动作。最后，诸如 FAST [171] 等策略解码器将融合特征映射为 50 个离散的动作词元，然后可以解码为电机指令 $\tau_{1:N}$。解码过程采用基于变换器的架构实现，如代码片段“动作预测代码”所示。一个 12 层的变换器解码器被实例化，模型维度为 512，注意力头数为 8。融合的多模态词元作为上下文提供，解码器自回归地逐步预测动作词元，每个预测的词元代表下一个控制决策，条件是完整的多模态上下文和所有先前生成的动作。生成的动作词元序列随后被去词元化为连续的电机指令轨迹以便执行。该实现类似于 LLM 中的文本生成，但这里的“句子”是一条运动轨迹，代表了自然语言生成技术在物理动作合成中的新颖再利用。图 9、算法 1 和伪代码共同示范了 VLAs 如何在一个连贯且可解释的词元空间中统一感知、指令和表现。该模块化设计使框架能够跨任务和机器人形态进行泛化，从而加速在现实世界应用中的部署，如苹果采摘、家务任务和移动导航。重要的是，词元化步骤的清晰性和可分离性使得架构具有可扩展性，能够进一步研究在 VLA 系统中的词元学习、分层规划或符号基础等内容。算法 1 VLA 词元化管道

<table><tr><td>1: Input: RGB-D frame I, text command T, joint angles θ</td></tr><tr><td>2: V ← ViT(I) 400 vision tokens 3: L ← BERT(T) &gt; 12 language tokens</td></tr><tr><td>4: S ← MLP(θ) 64-dim state encoding</td></tr><tr><td>5: F ← CrossAttention(V, L, S ) &gt; 512-dim fused token</td></tr><tr><td>6: A ← FAST(F) 50 action tokens</td></tr><tr><td>7: Output: Motor commands T1:N</td></tr></table>

# 动作预测代码

# 类似Python的伪代码 def predict_actions(融合词元): 变压器 $=$ 变压器( 层数 $= 1 2$ , 特征维度 $= 5 1 2$ , 头数 $^ { = 8 }$ ) 行动词元 $=$ 变压器.decode( 融合词元, 记忆 $=$ 融合词元 ) return 反词元化(行动词元)

# 2.4. 学习范式：数据来源与训练策略

训练视觉语言模型（VLA）需要一种混合学习范式，整合来自网络的语义知识和来自机器人数据集的任务基础信息[35]。如前面部分所示，VLA的多模态架构必须接触到支持语言理解、视觉识别和运动控制的多样数据形式。这通常通过两种主要数据源来实现。首先，如图10所示，大规模互联网衍生的语料库构成模型语义先验的主干。这些数据集包括图像-标题对（例如，COCO，LAION400M）、指令跟随数据集（例如，HowTo100M，WebVid）以及视觉问答语料库（例如，VQA，

![](images/10.jpg)  

Figure 10: Learning Paradigms: Data Sources and Training Strategies for VLAs.

这样的数据集使得视觉和语言编码器的预训练成为可能，帮助模型获取对象、动作和概念的一般表示。这一阶段通常使用对比或掩蔽建模目标，例如 CLIP 风格的对比学习或语言建模损失，以在共享嵌入空间中对齐视觉和语言模态。重要的是，这个阶段让视觉语言代理（VLA）具备了基本的“世界理解”，促进了组合泛化、物体基础以及零样本迁移。然而，单靠语义理解不足以执行物理任务。因此，第二阶段聚焦于将模型扎根于具身经验中。从现实机器人或高保真模拟器采集的机器人轨迹数据集被用于教导模型语言和感知如何转化为行动。这些数据集包括 RoboNet、BridgeData 和 RT-X，提供视频-动作对、关节轨迹以及在自然语言指令下的环境互动。演示数据可能来自动觉教学、远程操作或脚本化策略。该阶段通常采用监督学习（例如行为克隆）、强化学习（RL）或模仿学习来训练自回归策略解码器，根据融合的视觉-语言-状态嵌入预测动作词元。最近的研究越来越多地采用多阶段或多任务训练策略。例如，模型通常在视觉-语言数据集上使用掩蔽语言建模进行预训练，然后在机器人演示数据上用词元级自回归损失进行微调。其他方法使用课程学习，其中较简单的任务（例如物体推动）在复杂任务（例如多步骤操作）之前。一些方法进一步利用领域自适应，例如在 OpenVLA 中，或通过模拟到现实的迁移，弥合合成与现实世界分布之间的差距。通过将语义先验与任务执行数据统一，这些学习范式允许 VLA 模型跨任务、领域和具身进行泛化，成为可扩展的、遵循指令的智能体的基础，能够在现实世界中稳健操作。通过共同微调，这些数据集得到了对齐。模型学习将视觉和语言输入映射到适当的行动序列。这种训练范式不仅帮助模型理解物体的可用性（例如，苹果可以被抓取）和行动结果（例如，抬起需要的力量和轨迹），还促进了对新场景的泛化。在厨房操作任务上训练的模型可能能够推断如何在户外果园中摘苹果，如果它学习了物体定位、抓取和遵循语言指令的一般原则。近期的架构，例如谷歌 DeepMind 的 RT-2（机器人变换器2），在实际应用中展示了这一原则。RT-2 将动作生成视为一种文本生成形式，其中每个动作词元对应于机器人的控制空间中的一个离散命令。由于模型在网络规模的多模态数据和成千上万的机器人演示上进行了训练，它能够灵活地解释新指令并对新物体和任务进行零样本泛化，这在传统控制系统或早期多模态模型中几乎是不可能实现的。

# 2.5. 自适应控制与实时执行

VLAs 的另一优势在于其执行自适应控制的能力，通过实时传感器反馈动态调节行为。这在动态、非结构化环境中尤为重要，例如果园、住宅或医院，因为意外变化（例如风吹动苹果、光线变化、人类出现）可能会改变任务参数。在执行过程中，状态词元实时更新，反映传感器输入和关节反馈。模型随后可以相应地修订其计划行动。例如，在苹果采摘场景中，如果目标苹果稍微移动或另一个苹果进入视野，模型会动态地重新解释场景并调整抓取轨迹。这种能力模拟了人类的适应性，是 VLA 系统相较于基于管道的机器人系统的核心优势。

# 3. 视觉-语言-行动模型的进展

VLA 模型的起源受到基于变换器的 LLM（大型语言模型）显著成功的推动，特别是 2022 年 11 月发布的 ChatGPT，该模型展示了前所未有的语义推理能力。这一突破启发了研究人员将语言模型扩展到多模态领域，集成感知和机器人行动。到 2023 年，GPT-4 通过同时处理文本和图像引入了多模态能力，这促使后续努力将以语言为中心的多模态基础模型扩展到纳入物理行动表示和控制接口。同时，像 CLIP（2022）和 Flamingo（2022）等 VLM（视觉语言模型）通过对比学习建立了强大的视觉文本对齐能力，使得零-shot 物体识别成为可能，并为 VLM 模型（如 CLIP）奠定了基础。这些模型利用大规模标注数据集来对齐图像与文本描述，这是整合行动的重要前提。

一个重要的发展是大型机器人数据集的创建，例如RT-1的130,000个演示，这些提供了对齐动作的数据，对于共同训练视觉、语言和动作组件至关重要。这些数据集捕捉了多样的任务和环境，使模型能够学习可泛化的行为。随后在2023年，谷歌推出的RT-2带来了架构上的突破，它是一个具有里程碑意义的视觉-语言-动作（VLA）模型，将视觉、语言和动作词元统一起来，将机器人控制视为自回归序列预测任务。RT-2使用离散余弦变换（DCT）压缩和字节对编码（BPE）对动作进行离散化，在新对象上的性能提高了63%。多模态融合技术，例如交叉注意力变换器，将Vision Transformer（ViT）处理的图像（例如，400个补丁词元）与语言嵌入集成，使机器人能够执行复杂的命令，如“拿起碗左边的红色杯子”。此外，加州大学伯克利分校的Octo模型（2023年）推出了一种开源方法，具有9300万个参数和扩散解码器，基于来自OpenX-Embodiment数据集的800,000个机器人演示进行训练，进一步拓宽了研究的广度。

# 3.1. VLA模型的架构创新

从2023年到2024年，VLA模型经历了显著的架构进步和训练方法的改进。双系统架构作为一项关键创新涌现，以NVIDIA的GR00T N1（2025）为例，该模型结合了系统1（具有10毫秒延迟的快速扩散策略，用于低级控制）和系统2（基于LLM的规划器，用于高级任务分解）。这种分离使战略规划与实时执行之间的协调变得高效，从而提升了在动态环境中的适应能力。其他模型，如斯坦福的OpenVLA（2024），引入了一个7B参数的开源VLA，基于970,000个真实世界机器人演示进行训练，使用了双视觉编码器（DINOv2和SigLIP）以及Llama 2语言模型，超越了像RT-2-X（55B）这样较大的模型。训练范式演变为利用网络规模视觉-语言数据（如LAION-5B）和机器人轨迹数据（如RT-X）进行协同微调，将语义知识与物理约束对齐。像UniSim这样的合成数据生成工具通过创建具有光真实感的场景（例如遮挡物体）来解决数据稀缺问题，这对于稳健训练至关重要。通过低秩自适应（LoRA）适配器来增强参数效率，允许在不完全重训练的情况下进行领域自适应，从而使GPU时数减少了70%。如Physical Intelligence的pi 0模型（2024）中所示，引入基于扩散的策略，提供了改进的动作多样性，但需要显著的计算资源。这些进步使VLA技术更加普及，促进了协作和加速创新。最近的VLA模型朝向三种主要架构范式收敛，这三种范式在效率、模块化和鲁棒性之间取得了平衡：早期融合模型、双系统架构和自校正框架。这些创新针对真实世界机器人系统中的基础、泛化和动作可靠性等具体挑战。

早期融合模型：一种VLA方法类别专注于在输入阶段融合视觉和语言表示，然后再将其传递给策略模块。黄等人提出的EF-VLA模型[96]，在2025年国际学习表征大会（ICLR）上展示， exemplifies 这一趋势，通过保留CLIP[202]建立的表示对齐。EF-VLA接受图像-文本对，利用CLIP的冻结编码器对其进行编码，并在动作预测之前在变换器主干网络中早期融合生成的嵌入。这一设计确保了在CLIP预训练过程中学到的语义一致性得以保留，从而减少过拟合并增强泛化能力。值得注意的是，EF-VLA在组合操作任务上表现出$20 \%$的性能提升，并在先前未见的目标描述中达到了$85 \%$的成功率。通过保持视觉-语言主干网络的冻结，该方法保持了计算效率，避免了灾难性遗忘，同时领域特定的训练仅限于轻量级的策略或动作模块，使任务适应成为可能，而不牺牲模型的通用视觉-语义表示。

2. 双系统架构：受到人类认知的双过程理论启发，NVIDIA 的 GR00T N1（2025年）实现了两个互补的子系统：一个快速反应模块（系统1）和一个慢速推理规划器（系统2）。系统1 包括一个基于扩散的控制策略，具有 $1 0 ~ \mathrm { m s }$ 的延迟，适用于细粒度的低级控制，如末端效应器的稳定或自适应抓取。相比之下，系统2 使用大语言模型进行任务规划、技能组合和高层次序列安排。规划器将长期目标（如“清理桌子”）解析为原子子任务，而低级控制器确保实时执行。这种分解使得多时间尺度推理成为可能，并提高了安全性，特别是在需要快速反应和深思熟虑同时共存的环境中。在多阶段家庭操作的基准测试中，GR00T N1 的成功率比 RT-1、RT-2 和 OpenVLA 等单一模型提高了 $17 \%$，并将碰撞失败率降低了 $28 \%$。

3. 自我纠正框架：第三种架构演变是自我纠正 VLA 模型的出现，它通过显式的故障检测和恢复机制增强了传统推理管道。SC-VLA（2024）保留了类似于早期端到端或层次化 VLA 设计的标准快速推理路径，但引入了一条额外的较慢的纠正路径，该路径在检测到执行故障或不一致时被选择性激活，以重新评估决策并生成恢复动作。在这个框架中，默认行为是直接从融合嵌入中使用轻量级变换器预测姿态或动作。当检测到故障时，例如抓取失败或障碍物碰撞，模型会调用一个二级过程进行链式思维推理[281, 270]。该路径查询内部 LLM（或外部专家系统）以诊断故障模式并生成纠正策略[59]。例如，如果机器人反复错误地识别被遮挡的物体，LLM 可能会建议进行主动视角变化或夹爪重定向。在闭环实验中，SC-VLA 将任务失败率降低了 $3 5 \%$，并显著提高了在杂乱和对抗环境中的恢复能力。 4. VLA 模型的架构设计空间 VLA 模型展示了丰富多样的架构设计和功能侧重点，可以沿着端到端与模块化管道、层次化与扁平策略结构、以及低级控制与高级规划之间的平衡进行系统性组织（见表 1）。端到端 VLA，如 CLIPort [202]、RT-1 [19] 和 OpenVLA [122]，通过单一统一网络直接处理原始传感器输入为动作指令。相比之下，专注于组件的模型如 VLATest [237] 和 Chain-of-Affordance [129] 解耦了感知、语言基础和动作模块，使得对单个子模块的针对性改进成为可能。层次化架构被提出以应对复杂的长时间任务，通过将战略决策与反应控制分开。例如，CogACT [131] 和 $\mathrm { N a V _ { \mathrm { - } } }$ ILA [38] 采用两层层次结构，其中基于 LLM 的规划者向低级控制器发布子目标，从而结合了系统 2 推理与系统 1 执行的优势。类似地，ORION [69] 在一个连贯的框架中集成了用于长期上下文聚合的 QT-Former 和生成轨迹规划器。强调低级策略的模型以基于扩散的控制器为典型（例如 Pi-0 [15]、DexGraspVLA [291]），它们在产生平滑和多样的运动分布方面表现出色，但通常会带来更高的计算成本。相反，高级规划者（例如 FAST Pi-0 Fast [171]、CoVLA [5]）专注于快速生成子目标或粗略轨迹预测，将细粒度控制委托给专门模块或传统运动规划器。端到端双系统模型如 HybridVLA [142] 和 Helix [217] 通过同时训练两个组件，同时保持模块的可解释性，模糊了这些区别。表 1 进一步强调了最近的 VLA 如何平衡这些权衡。像 OpenDriveVLA [293] 和 CombatVLA [34] 的模型优先在动态、安全关键领域进行层次规划，而轻量级、面向边缘的系统如 Edge VLA [20] 和 TinyVLA [242] 则强调实时低级策略，代价是高级推理的牺牲。这个分类框架不仅阐明了 VLA 的设计空间，还通过精准定位未被充分探讨的组合，指导未来的发展，例如完全端到端的层次模型，经过优化用于嵌入式部署，承诺推动 VLA 系统在机器人、自主驾驶等领域的能力和适用性。表 1 中的分类具有重要意义，因为它为比较多样化的 VLA 架构提供了一个清晰的框架，突出了设计选择（如端到端集成与层次分解）如何影响任务性能、可扩展性和适应性。通过沿着低级政策执行和高级规划等维度对模型进行分类，研究人员可以更清楚地识别现有方法的优势和局限，并发现架构创新的机会。例如，农业机器人任务如高速果实采摘或精准喷雾受益于强调快速、反应性低级控制器的架构，而果园导航、多行覆盖规划或长时间作物监测等应用则需要更强的高级规划和推理能力。因此，这一分类法有助于为特定用例选择合适的 VLA 架构，并指导未来的研发朝向平衡响应性与认知规划的混合系统，最终加速具身 AI 的进步。

此外，为了汇总近年来VLA模型的进展，表2呈现了2022年至2025年间开发的显著系统的总结。这些模型基于早期融合、双系统处理和自校正反馈回路等架构创新，结合了多样的设计理念和训练策略。每个表格条目明确列出了模型的架构组件——即视觉编码器、语言编码器和动作解码器，以及用于基础和评估模型能力的主要训练数据集。像CLIPort [202]和RT-2 [299]这样的模型通过将语义嵌入与动作策略对齐奠定了早期基础，而像$P i$ -Zero、CogACT [131]和GR00T N1 [14]等更近期的框架则引入了可扩展的架构，采用基于扩散或高频控制器的设计。一些模型利用互联网规模的视觉-语言语料库和机器人轨迹数据集进行多模态预训练，从而增强了泛化能力和零-shot能力 [297, 291, 289, 257]。此表格比较为研究人员提供了一个参考点，以理解VLA设计在真实和模拟环境中的功能多样性、领域适用性以及新兴趋势。

<table><tr><td>Model Name</td><td>Year</td><td>End-to- End</td><td>Hie rarc hical</td><td>Comp onent Focused</td><td>Low-Level Policy</td><td>High-Level Planner</td></tr><tr><td>CLIPort [202]</td><td>2022</td><td></td><td>X</td><td>X</td><td></td><td>X</td></tr><tr><td>RT-1 [19]</td><td>2022</td><td></td><td>X</td><td>X</td><td></td><td>X</td></tr><tr><td>Gato [181]</td><td>2022</td><td></td><td>X</td><td>X</td><td></td><td>X</td></tr><tr><td>VIMA [112]</td><td>2022</td><td></td><td>X</td><td>X</td><td></td><td>X</td></tr><tr><td>Diffusion Policy [40]</td><td>2023</td><td></td><td>X</td><td>X</td><td>2</td><td>X</td></tr><tr><td>ACT [287]</td><td>2023</td><td>2</td><td>X</td><td>X</td><td>✓</td><td>X</td></tr><tr><td>VoxPoser [100]</td><td>2023</td><td></td><td>X</td><td>X</td><td>✓</td><td>X</td></tr><tr><td>Seer [80]</td><td>2023</td><td>✓</td><td>X</td><td>X</td><td>✓</td><td>X</td></tr><tr><td>Octo [218]</td><td>2024</td><td>✓</td><td>X</td><td>X</td><td>✓</td><td>X</td></tr><tr><td>OpenVLA [122]</td><td>2024</td><td>✓</td><td>X</td><td>X</td><td>✓</td><td>X</td></tr><tr><td>CogACT [131]</td><td>2024</td><td>X</td><td>✓</td><td>X</td><td>✓</td><td>✓</td></tr><tr><td>VLATest [237]</td><td>2024</td><td>X</td><td></td><td>✓</td><td>×</td><td>X</td></tr><tr><td>NaVILA [38]</td><td>2024</td><td>X</td><td>✗</td><td>X</td><td>✓</td><td>✓</td></tr><tr><td>RoboNurse-VLA [132]</td><td>2024</td><td>✓</td><td>X</td><td>X</td><td>✓</td><td>X</td></tr><tr><td>Mobility VLA [42]</td><td>2024</td><td>X</td><td>✓</td><td>X</td><td>✓</td><td>✓</td></tr><tr><td>RevLA [48]</td><td>2024</td><td>X</td><td>X</td><td>✓</td><td>X</td><td>X</td></tr><tr><td>Uni-NaVid [280]</td><td>2024</td><td>X</td><td>✓</td><td>X</td><td>✓</td><td>✓</td></tr><tr><td>RDT-1B [144]</td><td>2024</td><td>✓</td><td>X</td><td>X</td><td>✓</td><td>X</td></tr><tr><td>RoboMamba [143]</td><td>2024</td><td></td><td>X</td><td>*&gt;&gt;</td><td>✓</td><td>X</td></tr><tr><td>Chain-of-Affordance [129]</td><td>2024</td><td>&gt;xx</td><td>X</td><td></td><td></td><td>X</td></tr><tr><td>Edge VLA [20]</td><td>2024</td><td></td><td>X</td><td></td><td></td><td>X</td></tr><tr><td>ShowUI-2B [139]</td><td>2024</td><td>✓</td><td>X</td><td>X</td><td>✓</td><td>X</td></tr><tr><td>Pi-0 [15]</td><td>2024</td><td>✓</td><td>X</td><td>X</td><td>✓</td><td>X</td></tr><tr><td>FAST (Pi-0 Fast) [171]</td><td>2025</td><td>X</td><td>X</td><td>✓</td><td>✓</td><td>X</td></tr><tr><td>OpenVLA-OFT [121]</td><td>2025</td><td>✓</td><td>X</td><td>X</td><td>✓</td><td>X</td></tr><tr><td>CoVLA [5]</td><td>2025</td><td>X</td><td>✓</td><td>X</td><td>✓</td><td>✓</td></tr><tr><td>OpenDriveVLA [293]</td><td>2025</td><td>X</td><td></td><td>X</td><td>✓</td><td>✓</td></tr><tr><td>ORION [69]</td><td>2025</td><td>X</td><td>-</td><td>X</td><td>✓</td><td>✓</td></tr><tr><td>UAV-VLA [191]</td><td>2025</td><td>X</td><td></td><td>X</td><td>✓</td><td>✓</td></tr><tr><td>CombatVLA [34]</td><td>2025</td><td>✓</td><td>X</td><td>X</td><td></td><td>X</td></tr><tr><td>HybridVLA [142]</td><td>2025</td><td>X</td><td>✓</td><td>X</td><td>✓</td><td>✓</td></tr><tr><td>NORA [103]</td><td>2025</td><td>✓</td><td>X</td><td>X</td><td>✓</td><td>X</td></tr><tr><td>SpatialVLA [175]</td><td>2025</td><td>X</td><td>X</td><td>✓</td><td>✓</td><td>X</td></tr><tr><td>MoLe-VLA [283]</td><td>2025</td><td>X</td><td>X</td><td>✓</td><td>✓</td><td>X</td></tr><tr><td>JARVIS-VLA [130]</td><td>2025</td><td>✓</td><td>X</td><td>X</td><td>✓</td><td>X</td></tr><tr><td>UP-VLA [279]</td><td>2025</td><td>&gt;×x</td><td>X</td><td>*&gt;</td><td></td><td>X</td></tr><tr><td>Shake-VLA [120]</td><td>2025</td><td></td><td>X</td><td></td><td></td><td>X</td></tr><tr><td>DexGraspVLA [291]</td><td>2025</td><td></td><td>✓</td><td>X</td><td></td><td>✓</td></tr><tr><td>DexVLA [241]</td><td>2025</td><td>X</td><td>✓</td><td>X</td><td></td><td>✓</td></tr><tr><td>Humanoid-VLA [53]</td><td>2025</td><td></td><td>X</td><td>X</td><td>✓</td><td>X</td></tr><tr><td>ObjectVLA [297]</td><td>2025</td><td>✓</td><td>X</td><td>X</td><td>✓</td><td>X</td></tr><tr><td>Long-VLA [64]</td><td>2025</td><td></td><td>X</td><td>X</td><td>✓</td><td>X</td></tr><tr><td>RetoVLA [123]</td><td>2025</td><td>X</td><td>X</td><td>✓</td><td>✓</td><td>X</td></tr><tr><td>Vlaser [256]</td><td>2025</td><td></td><td>√</td><td>X</td><td>✓</td><td>✓</td></tr><tr><td>Discrete Diffusion VLA [138]</td><td>2025</td><td></td><td>X</td><td>X</td><td>✓</td><td>X</td></tr><tr><td>Being-H0 [152]</td><td>2025</td><td></td><td>X</td><td>X</td><td>✓</td><td>X</td></tr><tr><td>EgoVLA [258]</td><td>2025</td><td></td><td>X</td><td>X</td><td>✓</td><td>X</td></tr><tr><td>StereoVLA [47]</td><td>2025</td><td>✓</td><td>X</td><td>X</td><td>✓</td><td>X</td></tr><tr><td>GeoVLA [212]</td><td>2025</td><td>X</td><td>✓</td><td>X</td><td>✓</td><td>✓</td></tr><tr><td>EfficientVLA [262]</td><td>2025</td><td>X</td><td>X</td><td>✓</td><td>✓</td><td>X</td></tr></table>

Table 2: Compact summary of representative Vision-Language-Action (VLA) models. Each row reports the primary encoders/decoders, training data, and the main distinctive capability.   

<table><tr><td>Model (Ref.)</td><td>Architecture (vision / language / action)</td><td>Training data</td><td>Key strength / uniqueness</td></tr><tr><td>CLIPort [202]</td><td>CLIP-ResNet50 + Transporter-ResNet / CLIP- Self-collected [SC] GPT / LingUNet</td><td></td><td>Aligns semantic CLIP features with Transporter spatial rea- soning for precise SE(2) manipulation.</td></tr><tr><td>RT-1 [19]</td><td>EfficientNet / Universal Sentence Encoder / RT-1-Kitchen [SC] Transformer (discretized actions)</td><td></td><td>Early large-scale transformer policy for multi-task kitchen manipulation with tokenized actions.</td></tr><tr><td>RT-2 [299]</td><td>ViT-22B or ViT-4B / PaLI-X or PaLM-E / VQA + RT-1-Kitchen symbol-tuning (action tokens)</td><td></td><td>Co-finetunes internet-scale VQA with robot data, yielding emergent generalization for embodied tasks.</td></tr><tr><td>Gato [181]</td><td>ViT / SentencePiece / Transformer (unified to- Self-collected [SC] ken stream)</td><td></td><td>Generalist agent unifying robotics, language, and Atari via shared tokenization and a single transformer.</td></tr><tr><td>VIMA [112]</td><td>ViT + Mask R-CNN / T5 / Transformer</td><td>VIMA-Data [SC]</td><td>Prompt-driven VL grounding across multiple composi- tional task types (six prompt modalities).</td></tr><tr><td>ACT [287]</td><td>ResNet-18 / —/ CVAE-Transformer</td><td>ALOHA [SC]</td><td>Temporal ensembling enables smooth bimanual imitation with fine control precision.</td></tr><tr><td>Octo [218]</td><td>CNN / T5-base / Diffusion Transformer</td><td>Open X-Embodiment (OXE)</td><td>Large multi-robot policy trained on 4M+ trajectories span- ning many robot embodiments.</td></tr><tr><td>VoxPoser [100]</td><td>ViLD + MDETR/GPT-4 / MPC (LLM-guided Zero-shot planning)</td><td></td><td>Composes LLM+VLM for constraint-aware motion plan- ning without task-specific training.</td></tr><tr><td>Diffusion Policy [40]</td><td>ResNet-18 / — / U-Net or Transformer diffu- Self-collected [SC] sion</td><td></td><td>Diffusion modeling captures multimodal action distribu- tions for robust visuomotor control.</td></tr><tr><td>OpenVLA [122]</td><td>DINOv2 + SigLIP / Prismatic-7B / symbol- OXE + DROID tuning</td><td></td><td>Open-source RT-2-like VLA; supports efficient LoRA adaptation and broad generalization.&quot;</td></tr><tr><td>π0 (Pi-Zero) [15]</td><td>PaliGemma VLM / PaliGemma (multimodal) / Pi-Cross-Embodiment 300M diffusion action model</td><td></td><td>Lightweight general robot controller (reported ~3B total) with strong cross-robot, open-world generalization and bi- manual skills.</td></tr><tr><td>π0-Fast [171]</td><td>PaliGemma VLM / PaliGemma / autoregres- Pi-Cross-Embodiment sive transformer with FAST tokenization</td><td></td><td>High-frequency real-time control via compressed frequency-space action tokens (reported up to 15× faster inference).</td></tr><tr><td>OpenVLA-OFT [121]</td><td>SigLIP + DINOv2 (multi-view) / Llama-2 7B LIBERO; bimanual ALOHA / parallel decoding + action chunking (L1 re- gression)</td><td></td><td>Fine-tuning recipe with parallel decoding and chunked ac- tions; reported 97.1% LIBERO success and 26× faster in- ference for high-frequency bimanual control.</td></tr><tr><td>RDT-1B [144]</td><td>Multi-view RGB encoder / transformer lan- 46 datasets (&gt;1M episodes) + guage module / Diffusion Transformer (unified ALOHA fine-tune action space)</td><td></td><td>1.2B diffusion foundation model for dexterous bimanual manipulation with strong language conditioning and zero- shot transfer.</td></tr><tr><td>Helix1</td><td>System 2: open-source VLM for multimodal Figure reasoning (79 Hz) / integrated semantics / System 1: transformer visuomotor policy (200 Hz, full upper-body)</td><td>robot E2E (pixels+language→actions)</td><td>Humanoid-focused dual-rate VLA enabling real-time high- DoF control, dexterity, and collaborative multi-robot ma- nipulation with zero-shot generalization.</td></tr><tr><td>CogACT [131]</td><td>/ Llama-2 via Prismatic-7B / DiT-Base (300M tasks</td><td>DINOv2 ViT-L/14 + SigLIP ViT-So400M/14 OXE subset; Realman &amp; Franka</td><td>Componentized VLA with diffusion action transformer; reported +59.1% real-world success vs. OpenVLA and</td></tr><tr><td>Chain-of-Affordance</td><td>diffusion) Affordance-aware visual encoder / transformer LIBERO; real+sim manipulation reasoning prompts / autoregressive + diffusion</td><td></td><td>strong adaptation to unseen robots/objects. Sequential affordance reasoning (object→grasp→spatial→motion) improves spatial plan-</td></tr><tr><td></td><td>policy (affordance-conditioned)</td><td></td><td>ning and obstacle avoidance; reported stronger LIBERO performance than OpenVLA.</td></tr><tr><td>Edge VLA (EVLA) [20]</td><td>SigLIP + DINOv2 / Qwen2 (0.5B) / non- Bridge; OXE; 1.2M textimage pairs autoregressive joint control prediction</td><td></td><td>Edge-optimized VLA (e.g., Jetson-class) with reported 30 50 Hz inference and OpenVLA-comparable performance under low power.</td></tr><tr><td>ShowUI-2B [139]</td><td>UI-guided visual token selection / interleaved 256K GUI instruction-following V-L-A streaming / transformer GUI action predictor</td><td></td><td>Compact 2B VLA for digital automation; strong screenshot grounding and GUI/web navigation with efficient token se- lection.</td></tr><tr><td>GR00T N1 [14]</td><td>NVIDIA Eagle-2 VLM / integrated high-level Human demos + robot trajectories + planning / diffusion transformer (DiT)</td><td>simulation + internet video</td><td>Generalist humanoid dual-system design combining plan- ning and diffusion execution for dexterous multi-step con- trol and broad embodiment generalization.</td></tr><tr><td>Seer [80]</td><td>Grounding-optimized visual backbone / trans- LIBERO former language / autoregressive action head</td><td></td><td>Strong visual grounding for manipulation; competitive on LIBERO but typically below newer fine-tuned variants (e.g., OpenVLA-OFT).</td></tr><tr><td>DiffusionVLA [240]</td><td>Transformer visual encoder / autoregressive reasoning / diffusion action head bin-picking</td><td>LIBERO; factory sorting; zero-shot</td><td>Diffusion control improves robustness and interpretability;</td></tr></table>

<table><tr><td rowspan=1 colspan=16>Continued from previous page</td></tr><tr><td rowspan=1 colspan=16>Model (Ref.)                Architecture (vision / language / action)    Training data                  Key strength / uniqueness</td></tr><tr><td rowspan=1 colspan=8>ChatVLA [295]              Phase-aligned vision encoder / Prismatic MoE Unified chat-action (web+robot)LLM / unified V-L-A planner</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=4>PointVLA [125]</td><td rowspan=2 colspan=2></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>CLIP + 3D point cloud fusion / LLaMA-2 / Few-shot spatial tasks (real+sim)</td><td rowspan=2 colspan=7>CLIP + 3D point cloud fusion / LLaMA-2 / Few-shot spatial tasks (real+sim)</td><td></td></tr><tr><td rowspan=1 colspan=6></td><td rowspan=1 colspan=2>transformer with spatial token fusion</td><td></td></tr><tr><td rowspan=3 colspan=8>VLA-Cache [252]            SigLIP + token memory buffer / Prismatic-7B ALOHA + sim/real fusion/ transformer with dynamic token reuseHybridVLA [142]            CLIP + DINOv2 / LLaMA-2 / hybrid diffusion RT-X + synthetic fusion+ autoregressive ensemble</td><td rowspan=1 colspan=7>SigLIP + token memory buffer / Prismatic-7B ALOHA + sim/real fusion</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=7></td><td></td></tr><tr><td rowspan=2 colspan=8>HybridVLA [142]            CLIP + DINOv2 / LLaMA-2 / hybrid diffusion RT-X + synthetic fusion+ autoregressive ensembleMoLe-VLA [283]            Multi-stage ViT + STAR router / CogKD-</td><td rowspan=2 colspan=7>RLBench + real-world manipulation</td><td></td></tr><tr><td rowspan=1 colspan=2>Multi-stage ViT + STAR router / CogKD-</td><td></td></tr><tr><td rowspan=2 colspan=6></td><td rowspan=1 colspan=2>enhanced transformer / sparse transformer (dy-</td><td rowspan=4 colspan=7>ViT for aerial imagery / GPT instruction pars Satellie + UAV imagery instructions </td><td></td></tr><tr><td rowspan=1 colspan=2>namic routing)</td><td></td></tr><tr><td rowspan=2 colspan=6>UAV-VLA [191]</td><td rowspan=1 colspan=2>ViT for aerial imagery / GPT instruction pars Satellie + UAV imagery instructions </td><td></td></tr><tr><td rowspan=1 colspan=2>ing / transformer path planner</td><td></td></tr><tr><td rowspan=1 colspan=6>DexGraspVLA [291]</td><td rowspan=1 colspan=2>Object-centric spatial ViT / transformer grasp</td><td rowspan=1 colspan=7>Dexterousgrasping benchmark</td><td></td></tr><tr><td rowspan=3 colspan=6>GraspVLA [46]</td><td rowspan=1 colspan=2>reasoning / diffusion grasp controller</td><td rowspan=1 colspan=7>(sim+real)</td><td></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=7></td><td></td></tr><tr><td rowspan=1 colspan=2>Multi-view DINOv2 + SigLIP / VLM pre-</td><td rowspan=1 colspan=7>SynGrasp-1B; GRIT</td><td></td></tr><tr><td rowspan=1 colspan=6></td><td rowspan=1 colspan=2>dicts boxes+grasps / flow-matching action ex-</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=6></td><td rowspan=1 colspan=2>pert (PAG)</td><td rowspan=1 colspan=7></td><td></td></tr><tr><td rowspan=1 colspan=6>Interleave-VLA [62]</td><td rowspan=1 colspan=2>InternVL2.5 + OWLv2 / Qwen2.5 / continu-</td><td rowspan=1 colspan=7>Open Interleaved X-Embodiment</td><td></td></tr><tr><td rowspan=2 colspan=6></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>ous action predictor (OpenVLA+0-style, dif-</td><td rowspan=1 colspan=4>(210k eps., 11 datasets)</td><td></td></tr><tr><td rowspan=1 colspan=2>fusion controller)</td><td rowspan=1 colspan=7></td><td rowspan=8 colspan=1>from sketches and novel multimodal prompts.Targets long-horizon execution by explicitly separating&quot;moving&quot; vs. &quot;interaction&quot; phases, improving subtaskcompatibility and robustness over extended task horizons.Lightweight spatial-reasoning upgrade by repurposing reg-ister tokens; reported sizable success-rate gains on complexmanipulation with minimal architectural overhead.Bridges embodied reasoning and control: strong perfor-</td></tr><tr><td rowspan=3 colspan=6>Long-VLA [64]</td><td rowspan=1 colspan=2>End-to-end VLA for long-horizon tasks /</td><td rowspan=1 colspan=7>Long-horizon multi-step robot ma-</td></tr><tr><td rowspan=1 colspan=2>phase-aware input masking + transformer pol-</td><td rowspan=1 colspan=7>nipulation demonstrations (task se-</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2>icy (subtask phase segmentation)</td><td rowspan=1 colspan=7>quences)</td></tr><tr><td rowspan=4 colspan=6>RetoVLA [123]Vlaser [256]</td><td rowspan=1 colspan=2>VLM-based policy / reuses register tokens as</td><td rowspan=1 colspan=7>Real-robot manipulation on a 7-DoF</td></tr><tr><td rowspan=2 colspan=2>spatial context for action prediction</td><td rowspan=1 colspan=7>arm + task-specific demonstrations</td></tr><tr><td rowspan=1 colspan=7></td></tr><tr><td rowspan=1 colspan=2>VLM-to-VLA pipeline / synergistic embodied</td><td rowspan=1 colspan=7>Vlaser-6M embodied reasoning</td></tr><tr><td rowspan=2 colspan=4></td><td rowspan=2 colspan=3></td><td rowspan=1 colspan=4>reasoning + policy learning (reasoning-aware</td><td rowspan=1 colspan=4>dataset + VLA fine-tuning data</td><td rowspan=19 colspan=1>mance across embodied grounding/QA/planning while im-proving transfer to policy learning under domain shift.Unifies diffusion-style refinement with discrete token inter-faces: adaptive decoding order, error correction via remask-ing, and reduced autoregressive bottlenecks with strongbenchmark success rates.Scales dexterous manipulation by leveraging diversehuman video data; improves generalization to noveltasks/scenes compared with small teleop-only robotdatasets.Uses abundant egocentric videos for scalable pretraining,then aligns embodiments via unified action space to enablepractical robot transfer with limited robot data.Explicitly exploits stereo geometry to improve spatial pre-cision (depth-sensitive grasping/manipulation) and robust-ness to viewpoint/camera variations.Practical deployability: speeds up and reduces com-pute/memory of large VLA policies with minimal accuracydrop, enabling closer-to-real-time inference on constrainedhardware.</td></tr><tr><td rowspan=1 colspan=2>VLA fine-tuning)</td><td rowspan=1 colspan=3>(robot demonstrations)</td><td rowspan=1 colspan=6>(robot demonstrations)</td></tr><tr><td rowspan=1 colspan=6>Discrete Diffusion VLA [138]</td><td rowspan=1 colspan=2>Single-transformer VLA / discretized action</td><td rowspan=1 colspan=7>LIBERO + SimplerEnv (Frac-</td><td rowspan=1 colspan=1>Unifies diffusion-sty</td></tr><tr><td rowspan=10 colspan=6>Being-H0 [152]EgoVLA [258]</td><td rowspan=1 colspan=2>chunks + discrete diffusion refinement (CE</td><td rowspan=1 colspan=7>tal/Bridge) style VLA benchmarks</td></tr><tr><td rowspan=4 colspan=2>training; remasking)Dexterous VLA pretrained from human videos/ explicit hand-motion modeling + VL ground-</td><td rowspan=3 colspan=4>Large-scale human manipulation</td><td rowspan=2 colspan=5></td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=2 colspan=2></td></tr><tr><td rowspan=1 colspan=3>uman mani</td><td rowspan=1 colspan=3>manipulation</td></tr><tr><td rowspan=1 colspan=7>videos (egocentric/third-person) +</td></tr><tr><td rowspan=1 colspan=2>ing for action</td><td rowspan=1 colspan=7>transfer to robot control</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=7></td></tr><tr><td rowspan=1 colspan=2>VLM pretraining on egocentric human manip-</td><td rowspan=2 colspan=7>Large-scale egocentric humanvideos + small set of robot demon-</td></tr><tr><td rowspan=1 colspan=2>ulation / unified human-robot action space +</td></tr><tr><td rowspan=1 colspan=2>robot fine-tuning</td><td rowspan=1 colspan=7>strations</td></tr><tr><td rowspan=3 colspan=6>StereoVLA [47]</td><td rowspan=1 colspan=2>Stereo-enhanced VLA / geometric-semantic</td><td rowspan=2 colspan=7>Stereo robot manipulation demon-strations(stereoRGB with</td></tr><tr><td rowspan=1 colspan=2>fusion from stereo pairs (+ auxiliary depth</td><td rowspan=1 colspan=7>strations(stereoRGB with</td></tr><tr><td rowspan=1 colspan=2>cues) for action decoding</td><td rowspan=1 colspan=7>language-conditioned actions)</td><td rowspan=1 colspan=1>b</td></tr><tr><td rowspan=3 colspan=6>EfficientVLA [262]</td><td rowspan=2 colspan=2>Training-free VLA acceleration/compression /structured redundancy removal across VLApipeline (token/interface optimizations)</td><td rowspan=1 colspan=7>Applies to existing VLAs (eval-</td></tr><tr><td rowspan=1 colspan=7>benchmarks; e.g., SIMPLR-style</td></tr><tr><td></td><td></td><td rowspan=1 colspan=7>settings)</td></tr></table>

# 3.2. 视觉语言中的训练效率进展—动作模型

VLA模型在训练和优化技术方面取得了快速进展，以调和多模态输入、减少计算需求，并实现实时控制。关键进展领域包括：

# 数据高效学习。

在大规模视觉-语言语料库（例如 LAION-5B）和机器人轨迹集合（例如 Open X-Embodiment）上进行共同微调，使语义理解与运动技能相一致。OpenVLA（70亿参数）取得了比55亿参数的RT-2变体高出16.5%的成功率，展示了共同微调能够在参数更少的情况下实现强大的泛化能力，优于仅仅扩大模型规模。通过UniSim生成的合成数据创建了包括遮挡和动态光照的真实照片场景，以增强稀有边缘案例场景，在杂乱环境中提高模型的鲁棒性超过20%。自监督预训练采用对比目标（类似CLIP）在动作微调之前学习联合视觉文本嵌入，从而减少对任务特定标签的依赖。Qwen2-VL利用自监督对齐加速下游抓取与放置的收敛速度，提高了12%。参数高效适配。低秩适配（LoRA）在冻结的变换器层中插入轻量级适配器矩阵，使可训练参数减少高达70%，同时保持性能。Pi-0快速变体仅在静态主干上使用10万适配器参数，以实现连贯的200Hz控制，几乎没有精度损失。

# 推理加速。

压缩动作词元（FAST）和双系统框架中的并行解码（例如 GR00T N1）使策略推理速度提高了最多 $2 . 5 \times$，将每步延迟降低到低于 $5 \mathrm { m s }$，相较于在实时操控和人形控制基准上评估的标准自回归解码的单一策略 VLA。这种加速的代价是适度的轨迹平滑性损耗，表现为在高频控制下，动作离散化误差略微增加以及细粒度运动连续性降低 [14, 209]。总体而言，这些方法将 VLA 转变为能够处理语言条件、视觉引导任务的实用智能体，适用于动态的真实环境。

# 3.3. VLA模型中的参数高效方法与加速技术

除了数据高效学习策略外，VLA 模型的另一个研究方向专注于减少模型大小、内存占用和推理延迟，以便在计算和电力资源有限的真实机器人平台上进行部署。与之前讨论的以训练为中心的效率方法不同，本小节中的技术主要针对适应时的参数效率和策略推理时的运行时加速。 1. 低秩模块的参数高效适应：许多 VLA 采用的参数高效适应机制，如低秩适应（LoRA），不仅限于完全微调，此前已经在训练效率的背景下引入。在本节中，我们强调其在部署期间减少有效参数占用的作用。例如，OpenVLA 在一个冻结的 7B 参数主干上使用轻量级 LoRA 适配器（大约 $2 0 \mathbf { M }$ 参数），在最小内存开销下实现任务适应，避免跨任务重复完整模型权重。这种设计允许多个任务专属策略在资源受限的系统中共存，同时保留在预训练过程中学习的通用视觉语义表示。 2. 边缘部署的量化：模型量化降低数值精度以提高推理吞吐量和内存效率。对嵌入平台如 NVIDIA Jetson Orin 的 OpenVLA 实验表明，INT8 量化在抓取与放置基准测试中保持了约 $97 \%$ 的全精度任务成功率，仅在细致熟练的操作中有轻微降级。训练后量化和每通道校准进一步减小了在高动态范围传感器输入下的精度损失。这些优化使得在典型的移动机器人严格功率预算下维持高达 $3 0 \mathrm { H z }$ 的控制频率成为可能。 3. 模型剪枝和架构整体瘦身：结构化剪枝删除冗余架构组件，如注意力头或前馈子层，以减少内存和计算需求。尽管在 VLA 中的研究不如独立视觉或语言模型那样深入，但早期对基于扩散的视觉运动策略的研究表明，剪除多达 $20 \%$ 的卷积视觉编码器对抓取稳定性基本没有影响。将类似的剪枝策略应用于基于变换器的 VLA（如 RDT-1B）可以减少大约 $25 \%$ 的内存占用，同时任务成功率下降不足 $2 \%$，实现小于 4 GB 的部署。 4. 压缩动作标记化：为了解决长时间控制序列导致的推理瓶颈，提出了压缩动作表示。FAST 将连续的动作轨迹重新表述为紧凑的频域标记，显著减少解码长度。Pi-0 Fast 变种通过将 $1 0 0 0 \mathrm { m s }$ 的动作窗口压缩为 16 个离散标记，实现了高达 $1 5 \times$ 的推理速度提升，使得在桌面 GPU 上的控制速率高达 $2 0 0 \mathrm { H z }$。该方法以最小的轨迹粒度换取了巨大的速度提升，非常适合高频、迅速反应的操作任务。 5. 并行解码和动作块化：标准的自回归 VLA 顺序解码动作，导致累积延迟。采用双系统架构（如 GR00T N1）的并行解码策略同时生成时空动作标记组，在以 $1 0 0 \mathrm { H z }$ 运行的 7 自由度机器人手臂上，端到端推理延迟减少约 $2 . 5 \times$。动作块化进一步将多步例程（如抓取与放置）抽象为单个高级标记，使得在长时间的操作任务（如厨房工作流程）中推理步骤减少多达 $40 \%$。 6. 硬件感知编译和运行时优化：最后，硬件感知优化利用编译器级图重写、内核融合和加速器特定原语来最大化吞吐量。像 TensorRT-LLM 这样的框架利用张量核心、融合的注意力内核和管道化内存传输来加速变换器推理和扩散采样。在 OpenVLA-OFT 中，这些优化将推理延迟降低约 $30 \%$，并在 RTX 类 GPU 上将每次推理的能耗降低 $25 \%$，与标准 PyTorch 执行相比。这些系统级优化对在移动机器人、空中平台和具有严格功率限制的人形系统上部署实时 VLA 至关重要。 讨论：参数高效适应与推理加速技术共同推动了 VLA 部署的普及： • LoRA 和量化使较小实验室及研发项目能够在消费级硬件上微调并运行十亿参数的 VLA，从而为机器人解锁前沿的语义理解。 • 剪枝和 FAST 标记化压缩模型与动作表示，支持小于 4 GB 和小于 5 毫秒的控制循环，而不牺牲在灵巧任务中的精度。 • 并行解码和动作块化克服了自回归策略的顺序瓶颈，支持 $1 0 0 {-} 2 0 0 ~ \mathrm { H z }$ 的决策速率，这对于敏捷操作和腿部运动至关重要。 • 混合强化学习和监督学习训练在复杂环境中稳定探索，而硬件感知编译确保在边缘加速器上实现实时性能。 这些进展使得在工业操纵器、助力型无人机和消费级机器人中嵌入 VLA 模型变得切实可行，搭建了从研究原型到现实自主性的桥梁。

# 3.4. 视觉-语言-动作模型的应用

VLA模型迅速崛起，成为具身智能的基础构件，整合了感知、自然语言理解和运动控制于统一架构中。通过将视觉和语言模态编码为共享语义空间并生成具有上下文基础的动作，VLA模型实现了智能体与其环境之间的无缝互动。其多模态能力使VLA在广泛的现实应用中发挥了变革性作用。在类人机器人领域，像Helix和RoboNurse-VLA这样的系统结合了视觉、语言和灵巧操作，协助完成家庭任务和外科手术，展示了实时推理和安全意识控制。在自动驾驶汽车中，OpenDriveVLA和ORION等模型处理动态视觉流和自然语言指令，以便在复杂城市环境中做出透明的自适应驾驶决策。工业部署利用VLA架构进行高精度的装配、检验和协作制造。在农业领域，基于VLA的机器人系统能够实现视觉引导的水果采摘、植物监测和异常检测，从而减少对劳动力的依赖并提高可持续性。此外，近期在互动增强现实系统中的进展利用VLA模型进行实时的语言条件空间导航，基于语音或视觉提示引导用户在室内和室外环境中移动。在这些领域中，VLA提供了一个统一的框架，用于稳健、适应性强且语义一致的任务执行，标志着向具身通用智能体的重大转变。表3总结了近期VLA模型，概述了它们的方法论、应用领域和关键创新。接下来的小节将按时间顺序深入探讨这些应用领域，如图11所示。

# 3.4.1. 类人机器人

类人机器人旨在模仿人类身体的形态和功能，代表了部署视觉语言行动（VLA）模型最具挑战性但影响深远的领域之一。这些平台必须能够无缝地感知复杂环境，理解口语或书面自然语言，并以人类水平的灵巧性执行复杂的物理任务。VLA模型的核心优势在于其将感知、认知和控制统一为一个单一的、可端到端训练的框架，使类人机器人能够解读视觉输入（例如，杂乱场景的RGB-D图像）、理解语言指令（例如，“把勺子放到抽屉里”）并生成精确的运动轨迹。比较代表性的视觉-语言-行动（VLA）方法、应用领域和关键...

<table><tr><td rowspan=1 colspan=19>Reference (Year)     VLA methodology                         Application area             Strength / key innovation</td></tr><tr><td rowspan=1 colspan=5></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>mani</td><td rowspan=1 colspan=9>manipulation evaluation.</td><td rowspan=1 colspan=1>ing (robustness/reliability).</td></tr><tr><td rowspan=4 colspan=6>NaVILA [38] (2024)RoboNurse-VLA [132]</td><td rowspan=4 colspan=4>Two-level VLA: high-level vision-language gener-ates mid-level navigation commands; RL locomo-tion executes.Vision module (SAM2) + language module Surgical assistance</td><td rowspan=1 colspan=8>Two-level VLA: high-level vision-language gener-</td><td rowspan=1 colspan=1>Legged navigation from natural</td></tr><tr><td rowspan=2 colspan=8>scenes.</td><td></td></tr><tr><td rowspan=1 colspan=4>scenes.</td><td></td></tr><tr><td rowspan=1 colspan=8>(instrument</td><td></td></tr><tr><td rowspan=2 colspan=6>(2024)Mobility VLA [42]</td><td rowspan=1 colspan=4>(Llama2) with real-time voice-to-action pipeline.</td><td rowspan=2 colspan=8>grasp and handover).</td><td></td></tr><tr><td rowspan=1 colspan=4>Hierarchical VLA with long-context VLM for goal Multimodal instruction navigation</td><td></td></tr><tr><td rowspan=2 colspan=6>(2024)CoVLA [5] (2025)</td><td rowspan=1 colspan=4>localization and topological graph navigation.</td><td rowspan=2 colspan=8>with demonstration tours (MINT).Autonomous driving (dataset +</td><td></td></tr><tr><td rowspan=1 colspan=4>CLIP-based vision, Llama-2 language, trajectory</td><td></td></tr><tr><td rowspan=1 colspan=6>OpenDriveVLA [293]</td><td rowspan=1 colspan=4>prediction for action.Hierarchical alignment of 2D/3D visual tokens</td><td rowspan=1 colspan=8>VLA training).End-to-end autonomous driving.</td><td></td></tr><tr><td rowspan=2 colspan=6>(2025)ORION [69] (2025)</td><td rowspan=1 colspan=4>and language embeddings; autoregressive agent</td><td rowspan=2 colspan=8>Holistic end-to-end autonomous</td><td></td></tr><tr><td rowspan=1 colspan=4>environment—ego modeling.QT-Former for history context, LLM reasoning, 1</td><td rowspan=1 colspan=3>environment—ego modeling.</td><td></td></tr><tr><td rowspan=2 colspan=6>QUAR-VLA   [54]</td><td rowspan=1 colspan=4>generative planner for trajectory prediction.</td><td rowspan=1 colspan=8>driving.</td><td></td></tr><tr><td rowspan=1 colspan=2>QUAR-VLA</td><td rowspan=1 colspan=3>[54]</td><td rowspan=1 colspan=4>QUART-based fusion of vision and language for</td><td rowspan=1 colspan=3>Q</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>navigat</td><td rowspan=1 colspan=3>Quadruped navigation, manipula-</td><td></td></tr><tr><td rowspan=1 colspan=6>(2025)</td><td rowspan=1 colspan=4>action generation.</td><td rowspan=1 colspan=3>tion, whol</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>body ta</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=6>TinyVLA [242] (2025)UAV-VLA    [191]</td><td rowspan=1 colspan=4>Compact multimodal backbone with diffusion-</td><td rowspan=1 colspan=8>Fast, data-efficient manipulation</td><td></td></tr><tr><td rowspan=1 colspan=4>policy decoder.</td><td rowspan=1 colspan=8>control.</td><td></td></tr><tr><td rowspan=1 colspan=6>UAV-VLA    [191](2025)</td><td rowspan=1 colspan=4>Modular pipeline: GPT for goal extraction, VLMfor object search, GPT for action generation.</td><td rowspan=2 colspan=8>UAV mission planning from lan-guage + satellite imagery.Bimanual household manipula-</td><td></td></tr><tr><td rowspan=3 colspan=6>Bi-VLA [74] (2025)ChatVLA [295] (2025)</td><td rowspan=2 colspan=4>Multimodal transformer linking vision, language,</td><td rowspan=2 colspan=8>Bimanual household manipula-</td><td></td></tr><tr><td rowspan=1 colspan=1>al-world bimanu</td></tr><tr><td rowspan=1 colspan=4>and bimanual action modules.Phased alignment training with Mixture-of-Experts</td><td rowspan=1 colspan=8>tion.Unified multimodal understanding</td><td></td></tr><tr><td rowspan=1 colspan=6></td><td rowspan=1 colspan=4>for VL-A integration.</td><td rowspan=1 colspan=8>and robot control.</td><td></td></tr><tr><td rowspan=1 colspan=2>RoboMamba</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>[143]</td><td rowspan=1 colspan=4>Mamba/SSM-based VLA with co-trained vision</td><td rowspan=1 colspan=8>Efficient robotic reasoning and</td><td></td></tr><tr><td rowspan=1 colspan=2>(2025)</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=4>encoder and SE(3) action modeling.</td><td rowspan=1 colspan=7>manipulation.</td><td></td></tr><tr><td rowspan=1 colspan=6>OTTER [97] (2025)</td><td rowspan=1 colspan=4>Text-aware feature extraction using frozen pre-</td><td rowspan=1 colspan=8>Manipulation with zero-shot gen-</td><td></td></tr><tr><td rowspan=1 colspan=6></td><td rowspan=1 colspan=4>trained VLMs.</td><td rowspan=1 colspan=8>eralization.</td><td></td></tr><tr><td rowspan=1 colspan=6>PointVLA    [125]</td><td rowspan=1 colspan=4>Injects 3D point-cloud features into frozen VLA</td><td rowspan=1 colspan=8>Spatial reasoning; few-shot and</td><td></td></tr><tr><td rowspan=1 colspan=6>(2025)</td><td rowspan=1 colspan=4>via modular skip-blocks.</td><td rowspan=1 colspan=8>long-horizon manipulation.</td><td></td></tr><tr><td rowspan=1 colspan=6>VLA-Cache   [252]</td><td rowspan=1 colspan=4>Adaptive token caching with selective reuse of</td><td rowspan=1 colspan=8>Real-time efficient manipulation</td><td></td></tr><tr><td rowspan=1 colspan=6>(2025)</td><td rowspan=1 colspan=4>static visual tokens.</td><td rowspan=1 colspan=8>inference.</td><td></td></tr><tr><td rowspan=2 colspan=6>CombatVLA   [34](2025)</td><td rowspan=2 colspan=4>Video-action AoT training with truncated AoT forfast inference.</td><td rowspan=1 colspan=8>Real-time combatdecision-</td><td></td></tr><tr><td rowspan=1 colspan=8>making in 3D games.</td><td></td></tr><tr><td rowspan=1 colspan=6>HybridVLA   [142]</td><td rowspan=1 colspan=4>Unified LLM with collaborative diffusion and au-</td><td rowspan=1 colspan=8>Single-/dual-arm manipulation</td><td></td></tr><tr><td rowspan=1 colspan=6>(2025)</td><td rowspan=1 colspan=4>toregressive action policies.</td><td rowspan=2 colspan=8>across sim and real tasks.</td><td></td></tr><tr><td rowspan=1 colspan=6>NORA [103] (2025)</td><td rowspan=1 colspan=4>3B-parameter VLA using Qwen-2.5-VL-3B back-</td><td></td></tr><tr><td rowspan=1 colspan=6></td><td rowspan=1 colspan=4>bone with FAST+ tokenizer.</td><td rowspan=1 colspan=8>+ real).</td><td></td></tr><tr><td rowspan=1 colspan=6>SpatialVLA   [175]</td><td rowspan=1 colspan=4>Ego3D position encoding and adaptive action grids</td><td rowspan=2 colspan=8>manipulation.</td><td></td></tr><tr><td rowspan=1 colspan=6>(2025)</td><td rowspan=1 colspan=4>for spatially-aware VLA.</td><td rowspan=1 colspan=1>transfer and generali</td></tr><tr><td rowspan=1 colspan=6>MoLe-VLA   [283]</td><td rowspan=1 colspan=4>Mixture-of-Layers with dynamic layer-skipping</td><td rowspan=1 colspan=8>Efficient manipulation on RL- </td><td></td></tr><tr><td rowspan=1 colspan=6>(2025)</td><td rowspan=1 colspan=4>via router and distillation.</td><td rowspan=1 colspan=8>Bench + real robots.</td><td></td></tr><tr><td rowspan=1 colspan=6>JARVIS-VLA [130</td><td rowspan=1 colspan=4>Post-trained large VLMs with VL guidance and an</td><td rowspan=1 colspan=8>Open-world visual games (e.g.,</td><td></td></tr><tr><td rowspan=1 colspan=6>(2025)</td><td rowspan=1 colspan=4>action head for keyboard/mouse control.</td><td rowspan=1 colspan=4>Minecraft), 1k+ tasks.</td><td rowspan=3 colspan=4>Minecraft), 1k+ tasks.Embodied manipulation with pre-cise spatial reasoning.</td><td></td></tr><tr><td rowspan=1 colspan=6>UP-VLA [279] (2025)</td><td rowspan=1 colspan=4>Unified VLA with joint multimodal understanding</td><td rowspan=1 colspan=4>Embodied man</td><td></td></tr><tr><td rowspan=2 colspan=4>Shake-VLA</td><td rowspan=2 colspan=1>[120]</td><td></td><td rowspan=1 colspan=4>and future prediction objectives.</td><td rowspan=1 colspan=4>cise spatial reasoning.</td><td></td></tr><tr><td rowspan=2 colspan=6>Shake-VLA   [120](2025)</td><td rowspan=1 colspan=4>Modular stack with vision, speech-to-text, RAG,</td><td rowspan=1 colspan=7>Bimanual cocktail mixing in clut-</td><td rowspan=1 colspan=1>clut-</td><td></td></tr><tr><td rowspan=1 colspan=4>anomaly detection, and bimanual arms.</td><td rowspan=3 colspan=8>ter/noise.Quadruped multi-task locomotion,navigation, manipulation.</td><td></td></tr><tr><td rowspan=2 colspan=6>MoRE [285] (2025)DexGraspVLA [291]</td><td rowspan=3 colspan=4>Sparse MoE with LoRA modules and RL-based Q-function training.Hierarchical planner (pre-trained VL) + diffusion General dexterous grasping in di-low-level controller.</td><td></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>navigation, manipulation.</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=2>verse</td><td rowspan=1 colspan=2>e conditi</td><td rowspan=1 colspan=6>verse conditions.</td><td></td></tr></table>

![](images/11.jpg)  

Figure 11: Mind-map of application domains for VisionLanguage-Action models, with Humanoid Robotics positioned at the top and remaining domains arranged clockwise to match the order of discussion in this section.

近期的进展显著加速了视觉语言模型（VLA）在类人机器人中的部署。例如，由Figure AI开发的类人机器人Helix2，利用完全集成的VLA模型高频率执行全身操作，实时控制手臂、手、躯干，甚至精细的手指动作。该架构遵循双系统设计：多模态变换器处理语言命令和视觉流等输入，而实时运动策略以200 Hz的频率输出密集的行动向量。这样，Helix能够在未知的物体和任务之间进行泛化，灵活适应变化的环境，而无需特定任务的重新训练。VLA在类人系统中的主要优势在于其能够利用共享表征在多样化任务上进行扩展。与传统的依赖于任务特定编程或模块化管道的机器人系统不同，基于VLA的类人机器人在统一的基于标记的框架下工作。视觉输入通过预训练的视觉语言模型（如DINOv2或SigLIP）进行编码，而指令则利用如LLaMA或GPT风格的编码器进行处理。这些表征被融合为前缀标记，捕捉场景和任务的完整上下文。行动标记随后通过自回归生成，类似于语言解码，但代表机器人关节和末端执行器的运动指令。此能力使类人机器人能够在以人为中心的空间中有效操作，如家庭、医院和零售环境。在家庭环境中，基于VLA的机器人可以通过解读语音命令进行表面清洁、准备简单餐食或整理物品。在医疗领域，像RoboNurse-VLA这样系统已展示通过实时语音和视觉线索向外科医生进行精确工具交接的能力。在零售环境中，配备VLA的类人平台可以帮助处理客户咨询、补货和导航商店布局，而无需明确预编程。现代类人VLA的显著特点是其能在嵌入式低功耗硬件上运行，使现实世界的部署成为可能。例如，TinyVLA和MoManipVLA等系统展示了在Jetson级GPU上高效推理的能力，使移动部署不影响性能。这些模型利用扩散基础政策、基于LoRA的微调和动态标记缓存等技术，最大限度减少计算成本，同时保持高精度和泛化能力。在物流和制造领域，启用VLA的类人机器人已经产生了商业影响。像Figure 01这样的机器人被部署在仓库中，与人类工人一起执行挑拣、分类和上架等重复且物理上密集的任务。通过持续学习和强健的多模态基础，这些机器人具备处理新物体类别和动态变化场景的能力。随着VLA模型在多样行动生成、空间推理和实时适应能力上的持续进步，类人机器人正在成为家庭、工业环境和公共空间中高度有能力的助手。它们的优势在于能够通过共享的基于标记的架构，实现感知、语言理解和运动控制的统一，进而在非结构化的人类环境中实现无缝、上下文感知的行为。

例如，如图12所示，考虑名为“Helix”的最先进人形机器人，它配备了下一代VLA模型。当被口头指示“请从冰箱里拿水瓶”时，Helix激活其集成感知系统，其中一个基础视觉语言模型（例如，SigLIP或DINOv2）对视觉场景进行分割，以识别冰箱、把手和瓶子。语言输入由语言模型（如LLaMA-4）处理，该模型对指令进行词元化，并将其与视觉上下文融合。这个融合后的表征被传递给一个层次控制器：高层策略规划任务顺序（定位把手、打开门、识别瓶子、抓取），而中层规划器定义运动原语，例如抓取类型和关节轨迹。低层VLA控制器，通常基于扩散策略网络，以亚秒级延迟执行这些动作。当遇到变化（例如，倾斜的瓶子或滑动的抓握）时，Helix的智能体AI模块实时进行微观策略优化，根据反馈调整其抓握方式。这个例子展示了VLA驱动的人形机器人具有变革潜力。从厨房到诊所，这些系统不仅能解释复杂的指令并灵活执行物理任务，同时也能适应环境的不确定性。通过嵌入智能推理和安全对齐机制，现代VLA驱动的人形机器人正从狭窄任务执行者转变为通用和可信赖的协作伙伴。随着像TinyVLA和MoManipVLA这样的能效模型逐渐成熟，移动低功耗平台上的部署变得愈加实用，开启了一个具身的、社会对齐的AI新纪元。

![](images/12.jpg)  
H VLA VLA-based generalist robotics with dynamic task adaptation and safe, semantically grounded manipulation.

# 3.4.2. 自主车辆系统

自主车辆（AV），包括自动驾驶汽车、卡车和无人机，代表了VLA模型的前沿应用领域，其中安全关键决策需要紧密结合感知、语义理解和实时行动生成。与传统的模块化AV管道明确区分感知、规划和控制不同，VLA框架通过在统一模型内共同处理视觉观察、高层语义线索和内部状态表示，探索更紧密的架构耦合。虽然这种端到端的形式在模拟和受控基准测试中对指令条件下的导航和推理显示出良好效果，但大型商业系统（例如，特斯拉自动驾驶仪）仍然依赖模块化或混合管道，最近的行业努力集中于整合视觉语言推理组件，而不是在安全关键驾驶中完全部署VLA风格的行动生成。VLA模型使自主车辆能够理解超越像素级对象识别的复杂环境。例如，一辆在城市环境中行驶的自动驾驶汽车必须检测交通标志、理解行人行为，并解释导航指令如“在加油站之后向右转第二个路口”。这些任务涉及融合视觉和语言信号以理解空间关系、预测意图，并生成上下文敏感的驾驶行动。VLA通过基于词元的表示来编码这些信息，其中视觉编码器（例如ViT、CLIP）、语言模型（例如LLaMA-4）和轨迹解码器在统一的语义空间中工作，使得车辆能够推理高层目标并将其转化为低层运动。一个值得注意的贡献是CoVLA，它提供了一个综合数据集，将超过80小时的真实驾驶视频与同步的传感器数据（如LiDAR、里程计）配对，以及详细的自然语言注释和高分辨率驾驶轨迹。该数据集使得VLA模型能够对感知和语言特征与物理动作进行对齐的训练。CoVLA采用CLIP进行视觉定位，LLaMA-2进行指令嵌入，以及轨迹解码器做运动预测。这种配置使得AV能够解读口头提示（例如，“给救护车让路”）和环境条件（例如，合并流量），以做出透明和安全的驾驶决策。OpenDriveVLA通过将2D/3D多视角视觉词元与自然语言输入进行层次对齐，推动了VLA建模的进展。其架构利用自我中心的空间感知和外部场景理解来构建动态的智能体-环境-自我交互模型。通过自回归解码，OpenDriveVLA生成可被人类解读的行动计划（例如，转向角度、加速度）和轨迹可视化。其端到端框架在公共自动驾驶基准测试中取得领先表现，包括nuScenes和Waymo Open Motion数据集上的规划和轨迹预测任务，以及驾驶场景的视觉语言问答基准，展现了在城市导航和行为预测方面的强健性。另一个开创性模型，ORION，通过结合QT-Former以保留长时程的视觉上下文，一个LLM用于推理交通叙事和生成轨迹规划者，推动了闭环自动驾驶的边界。ORION在将视觉-语言模型的离散推理空间与AV运动的连续控制空间对齐方面表现出色。这种统一优化导致了准确的视觉问答（VQA）和轨迹规划，对于涉及模糊人类指令或遮挡障碍物的场景（例如，“在红色卡车后面出口”）至关重要。

例如，如图13所示，考虑一款名为“AutoNav”的自主配送车辆，在密集城市环境中使用下一代视觉语言架构（VLA）进行操作。当AutoNav接收到云端指令“在面包店旁红色遮阳篷附近投递包裹，然后避免施工区域返回基地”时，其车载视觉语言模型（VLM，例如CLIP或SigLIP）从多个摄像头解析视觉流，识别动态地标如面包店招牌、红色遮阳篷和交通锥。同时，基于LLaMA-4的语言模型模块解码指令，并将其与实时感知上下文相融合，包括激光雷达（LiDAR）、全球定位系统（GPS）和惯性测量。一个层次化的控制栈通过自回归的VLA解码器处理这些多模态信号，整合自我中心视角和世界中心地图来规划自适应路径。当车辆接近投递地点时，意外的行人活动促使一个智能子模块触发轨迹重规划，使用一种基于强化学习的策略优化例程。同时，AutoNav以声音警告行人，并重新校准速度以维持安全边际。这种语义理解、感知定位和自适应控制的相互作用展示了基于VLA的系统在安全关键场景中实现可解释、与人类行为一致的能力。这个场景说明了紧密集成的VLA架构如何超越传统的感知-规划-控制流程，通过实现端到端的语义推理、快速跨模块适应和可解释的决策制定。与模块化管道不同，后者的感知输出、规划更新和控制调整由松散耦合的组件处理，语义反馈有限，基于VLA的系统则共同推理语言意图、视觉上下文和实体状态，允许其动态重新规划轨迹、向人类传达与安全相关的意图，并实时调整控制策略。因此，系统表现出更大的自主性、通过人类可解释输出提高的透明度，以及在安全关键环境中更敏捷的决策能力。在航空机器人领域，VLA增强了无人机或无人驾驶飞行器（UAV）在配送和其他任务中的能力。像UAVVLA这样的模型结合了卫星影像、自然语言任务描述和车载传感器以执行高层指令（例如，“将包裹投递到带有蓝色遮布的屋顶平台”）。这些系统采用模块化VLA架构，其中视觉语言规划器解析全球上下文，而飞行控制器执行精准航点，支持物流、灾后响应和军事侦察等应用。

![](images/13.jpg)  

Figure 13: This illustration depicts an autonomous delivery vehicle powered by a VLA system, integrating VLMs for visual grounding, LLMs for instruction parsing, and a VLA decoder for path planning. Agentic AI enables adaptive trajectory refinement in dynamic environments, exemplifying how multi-modal integration drives safe, interpretable, and autonomous decision-making in realworld navigation tasks.

随着自主系统越来越多地在非结构化环境中运行，变换学习算法（VLA）提供了一种可扩展的、可解释的且数据高效的替代传统流程的方法。通过从大规模多模态数据集中学习，并将决策建模为词元预测，VLA 将人类水平的语义与机器人的运动对齐，为更安全、更智能的自主驾驶和导航技术铺平了道路。

# 3.4.3. 工业机器人

工业机器人正在经历范式转变，结合了VLA模型，使新一代智能机器人能够进行高级推理、灵活任务执行以及与人类操作员的自然沟通[32, 7]。传统工业机器人通常在高度结构化的环境中运行，使用刚性编程，当适应新的装配线或产品变种时，常常需要 extensive 重新配置和手动干预[6, 182]。这些系统缺乏现代动态制造环境所需的语义基础和适应性。相比之下，VLA模型提供了更人性化和可泛化的框架。通过视觉输入（例如，组件布局或输送带状态）、自然语言指令（例如，“拧紧红色模块上的螺丝”）和机器人状态的联合嵌入，VLA可以推断上下文并实时执行适当的控制命令[135, 72, 156]。视觉变换器（例如，ViT, DINOv2）、大语言模型（例如，LLaMA-4）以及自回归或扩散式动作解码器构成了这些系统的核心，使机器人能够解析多模态指令并执行基于其环境的动作。在这一领域最重要的贡献之一是CogACT[131]，这是一个专为工业机器人操作设计的模块化VLA框架。与依赖冻结语言-视觉嵌入并直接进行动作量化的早期VLA不同，CogACT引入了一种基于扩散的动作变换器，能够更稳健和自适应地建模动作序列。该系统使用视觉-语言编码器（例如，Prismatic-7B）提取高级场景和指令嵌入，随后将其传递给扩散变换器（DiT-Base）以生成细粒度的运动动作。这种模块化分离使得对未见过的工具、部件和布局具有更好的泛化能力，同时在现实世界约束下保持可解释性和鲁棒性。此外，CogACT通过高效的微调展示了在不同机器人实现（如6自由度臂或双手系统）之间的快速适应，使其适合在异质工厂环境中部署[131]。经验评估表明，CogACT在实际任务成功率上超过了之前的模型，如OpenVLA，提升幅度超过$28 \%$[122]，特别是在复杂的高精度任务上，如多步骤装配、螺丝固定和零件分类[131, 262]。随着制造业向工业4.0范式的转变，VLA有望减少编程开销，支持语音指令编程，并促进实时人机协作的混合主动任务。尽管执行精度、安全保障和延迟优化仍然是活跃的研究领域，但VLA模型的使用标志着朝着自主、智能和可适应机器人转变工厂迈出了重要一步。

# 3.4.4. 医疗与医疗机器人技术

医疗保健和医疗机器人领域具有高度风险，精确、安全和适应性是 VLA 模型日益能够提供的关键特性。传统医疗机器人系统高度依赖远程操作或预编程行为，这限制了它们在动态手术或护理环境中的自主性和响应能力。相比之下，VLA 模型提供了一种灵活的框架，整合了实时视觉感知、语言理解和精细运动控制，使医疗机器人能够理解高层次指令并自主执行复杂的程序或辅助任务。

在外科机器人领域，VLAs 可以显著增强微创手术的能力。这些系统可以将腹腔镜视频流、解剖图以及语音命令融合成一个统一的词元表示，使用视觉编码器（如 ViT、SAM-2）和语言模型（如 LLaMA、T5）。例如，如图14a所示，在“对左冠状动脉施加缝合”的任务中，视觉模块识别解剖目标，而语言模块为指令提供上下文。然后，动作解码器将融合的语义嵌入转换为具有亚毫米精度的逐步运动指令。这种视觉感知、基于语言的意图和动作级控制的闭环融合，使机器人能够自适应地重新定位工具、施加动态力反馈并避免关键解剖结构，从而减少外科医生的微观管理需求，降低人为错误的风险。

![](images/14.jpg)  

Figure 14: a) This figure illustrates a VLA surgical system executing the task "apply a suture to the left coronary artery." The vision module identifies anatomical targets, the language model interprets the instruction, and the action decoder generates precise motor commands, enabling adaptive tool control, real-time feedback, and safe autonomous operation; b) A VLA-powered assistive robot perceives patient behavior, processes verbal requests (e.g., "bring my walker"), and autonomously executes context-aware motion plans, enabling real-time assistance in eldercare, rehabilitation, and hospital logistics without relying on predefined scripts or manual oversight.

在手术室之外，VLA模型正在推动新一代患者辅助机器人在老年护理、康复和医院物流中的应用。这些系统可以自主感知患者行为，理解口头或手势输入，并执行响应任务，例如获取药物、引导助行器或在紧急情况下通知护理人员。举例来说，如图14b所示，一种启用VLA的机器人可以视觉检测到患者试图从床上起身，解读诸如“把我的助行器带来”的口头请求，并生成适合上下文的运动计划，以在没有预定义脚本或持续监督的情况下提供帮助。最近的VLA框架，如RoboNurse-VLA，突显了这一方法在现实世界中的可行性。RoboNurse采用SAM-2进行语义场景分割，并使用LLaMA2进行命令理解，整合到实时语音到行动的管道中，使机器人能够在手术室协助传递手术工具。该系统对多种工具、不同光照条件和嘈杂环境表现出较强的鲁棒性，这是临床环境中的常见挑战。此外，VLA架构在可解释性和可审计性方面具有优势，这在受监管的医疗领域中至关重要。场景定位和轨迹预测可以在事后可视化和审查，这可能促进临床信任并启用类似FDA的验证流程。基于LoRA的微调允许在极少的数据和计算基础设施下，适应特定医院环境或程序性工作流。重要的是，VLA模型的多模态基础使得跨域可转移性成为可能：同一模型在手术工具操作上的训练可以适应患者移动任务，且只需适度的再培训。这种模块化显著减少了与任务特定自动化系统相比的发展时间和成本。随着医疗机器人从远程操作援助转向半自主和协作系统，VLA模型在这一转型中处于核心地位。如前所述，VLA在将高级语义理解与低级控制相结合的能力，对于提供可扩展、与人类对齐和适应的机器人医疗解决方案至关重要。随着医疗系统面临日益增长的需求和人力资源短缺，VLA驱动的机器人将在提高医疗精准性、操作效率和以患者为中心的护理方面发挥关键作用。

# 3.4.5. 精准与自动化农业

如图15所示，视觉语言模型（VLA）正成为精准农业和自动化农业中具有变革性的工具，为劳动密集型任务提供智能、适应性的解决方案，适用于各种农田景观。与依赖于刚性传感器驱动管道且需手动重新编程以适应每个任务或环境变化的传统农业自动化系统不同，VLA在统一框架内集成了多模态感知、自然语言理解和实时行动生成。这种统一的多模态集成使得自主地面机器人和无人机能够理解复杂的田野场景，遵循口头或基于文本的农业指令，并生成如选择性采摘水果或自适应灌溉等上下文感知的动作。VLA能够动态调整以应对遮挡、地形不规则、光照变化或不同作物类型，并结合在合成的、真实感的数据集上训练，使其能够在不同的作物类型、地理位置和季节之间进行泛化。通过利用行动词元化、基于变换器的策略生成和类似LoRA微调的技术，这些系统正在重新定义农业机器人在可持续和精准驱动农业中的可扩展性和智能。在现代果园和其他作物田地中，VLA能够处理来自RGB-D相机、多光谱传感器或无人机的视觉输入，以监测植物生长、检测病害和识别营养缺乏。视觉变换器（例如ConvNeXt、DINOv2）编码来自视觉场景的空间和语义信息，而大语言模型（例如T5、LLaMA）解析自然语言指令，如“检查东侧地块是否有白粉病”或“在灌溉沟附近收获成熟的苹果”。通过词元融合，这些模态在共享表示空间中对齐，使机器人能够高精度地执行细粒度、上下文感知的动作。例如，在果摘任务中，如图15所示，配备VLA的地面机器人可以利用基于图像的成熟度线索识别成熟的农产品，解释用户指定的标准，如“仅选择A级水果”，并通过控制其末端执行器的行动词元执行运动序列。这种方法确保了农作物的损害最小化，优化了采摘率，并能够实时适应遮挡或地形变化等意外变量。在灌溉管理中，由VLA模型引导的无人机可以解读田野地图和口头指令，以选择性地灌溉受压区域，减少用水量。除了即时任务执行外，VLA模型还预计将通过闭环反馈机制支持动态重配置和终身学习。特别是在部署期间收集的执行结果、传感器观察和任务成功信号可以被记录，并定期纳入离线或增量更新的VLA策略中。当结合来自作物环境（例如，3D果园渲染）的真实感仿真生成的合成训练数据时，这种基于反馈的适应可能使模型在无需大量手动标注的情况下，逐步提高对新作物品种、病虫害条件和季节变化的鲁棒性。预计像LoRA适配器和基于扩散的策略细化等参数高效技术将在促进这种持续适应的同时限制计算开销。总体而言，VLA模型与农业工作流程的集成预计将带来若干长期益处，包括减少对熟练人工劳动力的依赖、通过有针对性的干预提高产量，以及通过优化输入使用增强环境可持续性。随着全球食品系统日益面临气候变化和资源约束，基于VLA的农业技术预计将为可扩展、智能且上下文感知的农业实践做出贡献，从而更好地适应现实世界的复杂性。

# 3.4.6. 基于视觉-语言-行动模型的互动增强现实导航

交互式增强现实（AR）导航代表了一个前沿领域，在该领域中，视觉语言模型（VLA）能够通过实时提供智能的、上下文感知的指导，显著增强人类与环境的互动。在这一范式中，VLA从增强现实设备（如智能眼镜或智能手机）中处理持续流动的视觉数据，同时结合自然语言查询，以在用户的物理世界视野中直接生成动态导航提示。与依赖于僵化地图和有限用户输入的传统GPS系统不同，基于VLA的AR智能体能够解释复杂的视觉场景（如交叉路口、室内走廊、标识牌），并响应自由格式的指令，例如“带我去最近有轮椅坡道的药店”或“显示前往会议室的最安静路线”。从技术角度来看，这些模型整合了视觉编码器（例如，ViT，DINOv2），用于从RGB摄像头帧中提取场景表示，一个语言编码器（例如，T5或LLaMA）处理用户提示或语音命令，以及一个动作解码器，预测词元化的导航提示，如方向覆盖、路线点或语音指令。基于变换器的架构融合了这些不同模态，以推理空间布局和语义意图，使AR智能体能够自适应地突出显示路径、地标和危害，直接在用户的视野内。例如，如图16所示，在一个拥挤的机场，VLA智能体可以在视图中识别出扶梯、登机口或行李领取，同时理解诸如“我如何在不走楼梯的情况下到达22号登机口？”的查询，并根据实时的占用情况和障碍调整路线。

![](images/15.jpg)  
agricultural automation.

期望虚拟语言助手（VLA）支持交互式指令循环，用户可以首先发出高级命令（例如，“导航到药房”），随后通过添加额外约束来进一步细化指令，例如“避免繁忙区域”或“选择景观优美的路线”。通过上下文感知反馈和迭代澄清，这些交互模式可以改善视觉障碍者或认知挑战者的可及性和可用性。在物流和室内导航中，这些系统可以与物联网传感器和数字孪生集成，为仓库工人、维修团队或送货机器人在复杂环境中提供指导。此外，可以通过持续微调实现个性化导航，让VLA模型随着时间的推移学习用户偏好和当地空间布局。

![](images/16.jpg)  

Figure 16: Showing how VLA models enable interactive AR navigation by fusing real-time visual perception, language understanding, and action planning. In dynamic environments such as airports, VLAs interpret user queries like "avoid stairs to Gate 22," analyze visual scenes (e.g., detecting escalators), and adjust navigational paths accordingly, supporting personalized, accessible, and context-aware mobility guidance.

随着增强现实硬件变得更加实惠并融入日常生活，基于视觉定位算法的导航系统将实现无缝的空间理解、多模态交互和在公共、工业及辅助环境中的自主引导，从而重新定义人类对物理空间的感知、探索和互动方式。

# 4. 视觉-语言-动作模型的挑战与局限性

VLA模型面临一系列相互关联的挑战，这些挑战阻碍了它们从研究原型向稳健的现实世界系统的转化。首先，实现实时、资源感知的推理仍然困难：DeeR-VLA等模型利用动态早期退出架构在操控基准上减少计算量$5{-}6 \times$，同时保持准确性，但在复杂场景中的收益减小。同样，UniNaVid将自我中心视频词元压缩至$5 \ \mathrm{H z}$导航，但在高度模糊的指令和较长的时间范围内仍然表现不佳。此外，这些以效率为驱动的设计往往暴露出计算速度与表示覆盖之间的权衡。在快速压缩或早期退出约束下操作时，即便是先进的混合视觉语言基础方法也表现出有限的物体泛化能力；例如，ObjectVLA仅对$64\%$的新物体进行泛化，突显了实时优化可能加剧开放世界鲁棒性差距的问题。其次，使用最小监督调整VLA模型并确保在稀疏、嘈杂数据下的稳定策略更新并非易事。ConRFT结合了行为克隆和Q学习，以及人机协同微调，在八个接触丰富的任务上快速收敛到$96.3\%$的成功率，但它在很大程度上依赖于专家的干预和奖励调整。像Hi Robot这样的分层框架将高层推理与低层执行解耦，以提高指令的保真度，但协调这些模块与理清模糊反馈仍然具有挑战性。同样，触觉语言动作模型将触觉流与语言命令融合，在未见过的peg-in-hole任务上实现了超过$85\%$的成功率，但数据集的广度和实时多步解码仍限制了更广泛的泛化。

此外，在动态环境中确保安全性、泛化能力和端到端可靠性需要新的建模和评估标准。占用语言动作模型，如 OccLLaMA，将三维场景理解与行动规划相结合，但它们需要扩展到更丰富的场景动态和跨模态的语义一致性。RaceVLA 通过量化的迭代控制循环推进高速无人机导航；然而，与更大规模的 VLAs 和专用推理模型相比，其有限的视觉-物理泛化能力在未见或快速变化的环境中引发了安全隐患。${ \mathrm{R e V L A} }$ 中的模型合并策略恢复了失去的域外视觉鲁棒性，使定向物体检测（OOD）抓取成功率提高了 $77 \%$，但也引入了额外的计算和复杂性。最后，SafeVLA 通过约束马尔可夫决策过程制定约束条件，将不安全行为减少超过 $80 \%$，然而，为多样化的现实世界任务定义全面且不具限制性的安全规则仍然是一个未解问题。解决这些交叉限制对 VLA 模型在复杂的实际机器人环境中实现可靠的自主操作至关重要。

基于上述关键限制，必须将每个挑战映射到针对性的缓解策略，并评估其系统级影响。表4展示了这一映射，识别核心限制、基于最近进展的潜在技术解决方案，并阐述了其在现实世界VLA部署中的预期效益。例如，解决实时推理限制利用并行解码和量化变压器管道，结合硬件加速（例如，TensorRT），以维持无人机和操作臂中的控制回路速度。通过混合扩散-自回归策略解决多模态动作表示，增强模型产生多样化、上下文相关的运动指令以应对复杂任务的能力。为确保在开放世界中的安全性，可以集成动态风险评估模块和自适应规划层，从而在不可预测的环境中确保强健的紧急停止行为。同样，通过策划去偏见语料库和先进的对比微调，可以最小化数据集偏见和定位，增强在新对象和场景中推广时的公平性和语义忠实度。这些策略以及其他涉及模拟到现实转移、触觉集成和节能架构的方法，构成了将VLA研究转变为可靠、可扩展自治的全面路线图。本节的其余部分组织为五个集中小节，每个小节考察文献中识别出的VLA挑战的不同集群。首先，我们分析实时推理限制及其新兴解决方法。接下来，我们探索开放世界环境中的多模态动作表示和安全保障。然后，我们讨论数据集偏见、定位策略及其对未见任务的概括，随后探讨系统集成复杂性和计算需求。最后，我们考虑在现实世界应用中部署VLA的鲁棒性和伦理影响。

# 4.1. 实时推理限制

尽管近期取得了一些进展，在对延迟敏感的场景中部署视觉语言模型（VLA）依然受到实时推理要求的制约，特别是在机器人操作、自动驾驶和空中控制等应用中。VLA 通常依赖自回归解码策略，该策略根据先前的预测顺序生成动作词元。尽管这种方法在许多任务中有效，但自回归解码范式显著限制了推理速度，通常在标准 GPU 研究平台（例如，单个高端消费级或数据中心 GPU）上进行端到端 VLA 推理时，仅能达到 $3 { - } 5 \mathrm { H z }$ 的速率。这一速度远低于高效且稳定的机器人操作所需的控制频率，后者通常在数十赫兹（用于高层规划）到更高更新率（用于低层反馈控制）之间变动，具体取决于任务和硬件平台。例如，当机器人手臂操作精细物体时，频繁的位置信息更新对于保持准确性和防止损坏是至关重要的。像 OpenVLA 和 Pi-0 这样的模型在这一顺序生成词元的方法上面临固有挑战，从而限制了它们在动态环境中的有效性。

<table><tr><td colspan="2">Challenge / limitation</td><td>Potential solution</td><td>Expected impact</td></tr><tr><td colspan="2">Real-time inference straints</td><td>Parallel decoding, quantized transformers, and hardware acceleration I (e.g., TensorRT) [129, 122]; reduce autoregressive overhead [75, 142].</td><td>Enables real-time control and deployment in latency-critical domains [251, 191] (e.g., UAVs, manipulators).</td></tr><tr><td colspan="2">Multi-modal action represen- tation</td><td>Hybrid tokenization combining diffusion and autoregressive policies [171]; train on diverse demonstrations and multi-modal outputs [156].</td><td>Improves performance on complex, dynamic manipulation with mul- tiple valid solution modes [74].</td></tr><tr><td colspan="2">Safety assurance in open worlds</td><td>Dynamic risk assessment modules [183, 233]; low-latency emergency- stop and adaptive planning layers [113].</td><td>Improves reliability and safety in unpredictable settings (homes, fac- tories, healthcare); enhances user acceptability.</td></tr><tr><td colspan="2">Dataset bias and grounding</td><td>Curate diverse/debiased datasets [185]; stronger grounding (e.g., CLIP fine-tuning with hard negatives) [282, 17].</td><td>Improves fairness and semantic fidelity [109], and enhances general- ization to novel real-world inputs [227, 288, 175].</td></tr><tr><td colspan="2">Limited 3D perception and reasoning</td><td>clouds with visionlanguage features.</td><td>complex environments [129].</td></tr><tr><td colspan="2">Cross-embodiment general- ization</td><td>Train across diverse morphologies; learn embodiment-agnostic action abstractions; apply cross-domain adaptation [266].</td><td>Facilitates policy transfer across robot platforms and configurations [279, 122].&quot;</td></tr><tr><td colspan="2">Annotation complexity and cost</td><td>Weak supervision, active learning, and synthetic data generation to re- duce manual labeling [148].</td><td>Lowers development cost and accelerates scaling to new tasks/domains [233, 286].</td></tr><tr><td colspan="2">Sim-to-real transfer gap</td><td>Domain adaptation, sim-to-real fine-tuning, and real-world calibration [210, 134].</td><td>Improves reliability and consistency when deploying beyond simula- tion [4, 66].</td></tr><tr><td colspan="2">Integration of physical knowl- edge</td><td>  training pipelines [53].</td><td>[106].</td></tr><tr><td colspan="2">Multi-modal integration (tac- tile, audio)</td><td>Fuse tactile/audio with vision and language [114]; extend multimodal transformer fusion.</td><td>Improves robustness under occlusion/ambiguity and expands the task repertoire [76, 136, 91].</td></tr><tr><td colspan="2">Long-horizon multi-stage tasks</td><td>Hierarchical policies, memory-augmented networks, and trajectory planning modules [136].</td><td>Improves sequential planning, memory, and compositional execution [131, 286, 227, 175].</td></tr><tr><td colspan="2">System integration complexity</td><td>Unified transformer backbones [289]; temporal alignment and sim-to- real transfer strategies [163, 277].</td><td>Enables tighter planning-control coordination and more robust trans- fer to physical robots [187, 208].</td></tr><tr><td colspan="2">Energy and compute demands</td><td>tors.</td><td></td></tr><tr><td colspan="2">Generalization to unseen taks</td><td> , agnostic pretraining [173, 146].</td><td>242, 286].</td></tr><tr><td colspan="2">Robustness to environmental variability</td><td>online recalibration [156].</td><td>ics [285, 242].</td></tr><tr><td colspan="2">Ethical and societal implica- tions</td><td>Privacy via on-device processing/anonymization [149, 198, 252, 34]; fairness audits; regulatory and trust frameworks.</td><td>Promotes equitable and trustworthy adoption across social, medical, and labor domains [160, 180, 217, 172].</td></tr></table>

新兴解决方案，如并行解码，以NVIDIA的GR00T N1模型为例，旨在通过同时预测多个词元来加速推理。GR00T N1相较于传统解码方法实现了约$2.52 \times$的加速；然而，这种并行性往往会引入轨迹平滑性的权衡，导致机器人运动的次优表现。这种运动在手术机器人等敏感应用中是不可取的，因为精确性、适应性和灵活性至关重要。因此，在不妥协输出质量的情况下实现快速推理仍然是一个开放的挑战。此外，硬件限制加剧了实时推理的约束。例如，处理高维视觉嵌入通常涉及超过400个词元，每个词元维度为512，需要约$1.2 \mathrm{GB}/\mathrm{s}$的内存带宽。这一需求显著超出了当前嵌入式系统或边缘AI硬件（如NVIDIA Jetson平台）的容量，从而限制了实际部署。即使使用高效的量化技术来降低浮点运算的精度以缓解内存限制，模型在执行要求亚毫米精度的任务（如双手机器人操作或医疗机器人）时，通常仍会出现精度下降。

# 4.2. 多模态动作表示与安全保障

多模态动作表示：当前视觉语言模型（VLA）的一个显著局限性在于准确表示多模态动作，尤其是在需要连续和细致控制的场景中。传统的离散标记化方法，如将动作划分为256个不同的区间，固有地缺乏精确度，在精细任务中如精细的机器人抓取或复杂的外科手术中会造成 substantial errors。例如，在组装任务中的精准机器人操作中，离散表示可能导致动作错位或不精确，从而削弱性能和可靠性。另一方面，基于连续多层感知机（MLP）的方法面临模式崩溃的风险，即模型尽早收敛到单一动作轨迹，尽管有多条可行路径可供选择。这降低了在高度动态环境中适应性决策所需的灵活性。新兴的基于扩散的策略，例如 Pi-Zero 和 RDT-1B 模型，提供了更丰富的多模态动作表示，能够捕捉多样的动作可能性。然而，它们的计算开销相对较大，大约是传统基于变换器解码器的三倍，使其在实时部署中变得不切实际。因此，VLA 模型在处理复杂动态任务时目前面临挑战，例如在拥挤空间中的机器人导航或复杂的双手操作，其中多种战略动作可能同样有效且依赖于上下文。 开放世界中的安全保障：另一个面对视觉语言模型的关键挑战是确保在动态、不可预测的环境中保持强健的安全性，这是现实世界场景的特征。许多当前的实现高度依赖于预定义的、硬编码的力量和扭矩阈值，这显著限制了它们在遭遇不可预见或新颖条件下的适应能力，例如意外障碍物或突发的环境变化。用于碰撞预测的模型在杂乱和动态空间中的准确率通常仅为约82%，在诸如仓库物流或家庭机器人等应用中，安全余量极小，存在严重风险。此外，诸如紧急停止等基本安全机制通常会引入 substantial latency，往往在200到500毫秒之间，主要由于全面的安全验证。这种延迟虽然看起来微不足道，但在高速操作或关键干预中，例如自动驾驶或紧急机器人响应时，可能会变得危险。

# 4.3. 数据集偏差、基础性和对未见任务的泛化

限制视觉语言模型（VLA）效果的一个重要障碍是数据集偏差和语义对齐缺陷的普遍存在。目前的训练数据集主要来源于网络爬取的资源，常常表现出固有的偏见。研究表明，标准数据集中的大约 $17 \%$ 关联偏向于刻板印象的解读，例如不成比例地将“医生”等术语与男性形象关联。这些偏见通过训练传播，导致 VLA 在多样环境中部署时，产生语义不一致或上下文不恰当的响应。例如，OpenVLA 等模型在新环境中忽视约 $23 \%$ 的物体引用，这在准确解释指令至关重要的真实应用中，显著限制了其实用性。这种语义对齐问题还扩展到组合泛化的挑战，VLA 在遇到稀有或非常规组合时往往失效，例如由于训练语料库中的代表性不足，无法理解“黄色马”等短语。这些短处突显出对精心策划、平衡且全面的领域特定数据集的迫切需求，并需要结合先进的语义对齐算法，以减轻偏见并增强不同上下文之间的语义一致性。除数据集偏见造成的挑战外，更广泛的问题是对未见任务的泛化，这对 VLA 的实际部署构成了关键障碍。尽管现有模型在熟悉环境或与训练场景相似的任务中表现良好，但在面对全新任务或不熟悉变体时，其性能大幅下降，通常下降幅度可达 $40 \%$。例如，专门训练用于家庭任务的 VLA 在进入工业或农业环境时可能会遇到困难或失败，这主要是由于物体类型、环境动态和操作限制的差异。这一限制主要来源于对狭隘训练分布的过拟合以及对多样任务表示的曝光不足。因此，目前的 VLA 在零-shot 或 few-shot 学习场景中表现出有限的泛化能力，阻碍了其适应性和可扩展性。

# 4.4. 系统集成复杂性与计算需求

将 VLA 模型整合到双系统架构中，即结合高层次认知规划（系统 2）和实时物理控制（系统 1），在机器人应用中展现出显著的复杂性。一个主要挑战源自这两个系统之间的时间不匹配。通常，系统 2 利用大型语言模型（如 GPT 或 LLaMA-4）进行复杂任务的分解和战略规划。这些模型由于其巨大的计算需求，通常在标准的基于 GPU 的推理平台（例如，单个高端消费级或数据中心 GPU）上执行时，推理延迟在 ${ \sim } 800 \mathrm { ms }$ 或更长时间。相反，负责低层次运动执行的系统 1 组件通常运行在实时 CPU、微控制器或专用机器人控制器上，实施紧密约束的控制循环，更新间隔通常在几毫秒的量级，这取决于平台和任务。这种操作节奏的明显差异导致同步困难，造成延迟和可能的次优执行轨迹。例如，NVIDIA 的 GR00T N1 模型展现了这两个系统的有效整合，但仍因异步交互时而出现运动不流畅，突显了这一内在挑战。此外，高维视觉编码器（如视觉变换器（ViT））与低维动作解码器之间的特征空间不对齐加剧了整合的复杂性。在试图调和这些不同的嵌入时，感知理解与可操作命令之间的连贯性可能显著下降。OpenVLA 和 RoboMamba 利用基于变换器的视觉处理和后续的动作解码，展示了这些整合挑战，导致从仿真环境转向物理硬件部署时性能下降。这些差异可能导致性能降低，主要是由于模拟动态与现实世界传感器噪声或校准问题之间的不匹配。能源和计算需求构成了 VLA 部署的另一个重要障碍，尤其是在典型的自主无人机、移动机器人和穿戴式机器人系统的边缘计算环境中。高级 VLA 通常具有的巨量参数（例如，拥有超过 70 亿个参数的模型）在其原生形式下需要的计算资源往往超过 28 GB 的显存。这些需求远高于大多数当前面向边缘的处理器和 GPU 的能力，限制了复杂 VLA 在专用高资源环境之外的实际应用。

# 4.5. VLA部署中的鲁棒性和伦理挑战

VLA模型在实际应用中的一个核心障碍是其对环境变化的鲁棒性有限，这引发了重要的伦理和安全考虑。环境鲁棒性是指系统在动态变化和部分可观察条件下维持可靠感知、推理和动作生成的能力。在实践中，真实环境通过光照波动、不利天气、传感器噪声和物体遮挡等因素引入显著的不确定性。实证证据强调了这些限制在多个VLA组件中的表现。例如，系统如OpenDriveVLA [293]中使用的视觉模块在低对比度或以阴影为主的场景中经历约$20{-}30\%$的准确性下降，反映出当前视觉编码器对挑战性光照条件的敏感性。同样，在如CoVLA [5]等VLA系统中，语言理解在声学噪声或语义模糊的环境中恶化，指令误解可能传播到不正确的动作执行。在以操作为中心的场景中，像RoboMamba [143]这样的VLA驱动机器人系统在杂乱环境中表现不佳，常常错误估计部分遮挡物体的姿态或方向，从而降低任务成功率。这些鲁棒性限制在安全关键的应用中具有直接的伦理影响，因为在实际环境变化下的性能下降可能导致意外行为、可靠性降低和用户信任丧失。因此，解决鲁棒性问题不仅是技术挑战，也是在人本环境中负责任和伦理部署VLA系统的前提。

# 5. 讨论

如图17所示，视觉-语言-行动（VLA）模型面临一系列多方面的挑战，涉及算法、计算和伦理维度。首先，由于自回归解码器的顺序特性和多模态输入的高维性，在资源受限的硬件上实现实时推断仍然很困难。其次，将视觉、语言和行动融合成一致的策略在遇到意料之外的环境变化时引入安全漏洞。第三，数据集偏见和定位错误削弱了模型的泛化能力，常常导致模型在分布外任务上失败。第四，整合多样的组件——感知、推理和控制——生成复杂的架构，使得优化和维护变得困难。第五，大型VLA系统的能源和计算需求限制了其在嵌入式或移动平台上的部署。最后，有限的对环境变化的鲁棒性可能导致不安全或不可靠的行为，这进而引发与安全保障、问责制、隐私和偏见缓解相关的伦理和监管问题。总体而言，这些局限性制约了VLA模型在现实世界中的机器人技术、自治系统和交互应用中的实际应用。关于这些挑战的潜在解决方案将在下面讨论。

# 5.1. 潜在解决方案

实时推理约束。未来的研究必须开发协调延迟、吞吐量和任务特定准确性的超大规模架构。一种有前景的方向是整合专用硬件加速器，如基于FPGA的视觉处理器和优化稀疏矩阵运算的张量核心，以在亚毫秒级别执行卷积层和变换层。模型压缩技术如低秩适应（LoRA）可以将参数数量缩减高达$90\%$，在保持基准任务上超过$95\%$原始性能的同时，减少内存占用和推理时间。渐进量化策略结合混合精度算术（例如，FP16/INT8）与按块校准，可以进一步将计算量减少$2{-}4$倍，几乎不损失准确性。自适应推理架构可以根据输入复杂度动态调整网络深度或宽度，类似于DeeR-VLA中的早期退出分支，通过选择性绕过变换层来减少平均计算，当视觉场景或语言指令较简单时。最后，有效的词元化方案利用子词补丁嵌入和动态词汇分配，可以将视觉和语言输入压缩为紧凑的表示，最小化词元数量而不牺牲语义丰富性。这些创新可以在普通边缘GPU上实现亚$50~\mathrm{ms}$的端到端推理，为自适应无人机飞行、实时遥控和协作制造等对延迟敏感的应用铺平道路。

多模态动作表示与安全保障。解决多模态动作表示和稳健安全性需求的端到端框架，需要在严格的安全约束下统一感知、推理和控制。结合基于扩散的低级运动原语采样的混合策略架构 [40] 和自回归高级规划器 [242]，能够实现多样化动作轨迹的紧凑随机表示，提高动态环境下的适应性。安全性可以通过实时风险评估模块来强化，该模块接收包括视觉、深度和本体感觉数据在内的多传感器融合流，以预测碰撞概率和关节应力阈值，当预定义的安全边界被突破时触发紧急停止电路 [183, 233]。增强约束优化的强化学习算法（例如，SafeVLA中的拉格朗日方法 [274]）可以学习最大化任务成功的策略，同时严格遵守安全约束。在线模型适应技术，如基于规则的强化学习（GRPO）和直接偏好优化（DPO），进一步在新环境条件下优化动作选择，确保在各种场景下保持一致的安全性能 [113]。关键是嵌入形式化验证层，在执行前对规划器输出进行符号分析，从而保证即便对于基于神经网络的控制器也能遵守安全不变性。集成这些方法论将产生能够执行复杂多模态动作且在非结构化现实环境中具备可证明安全性的VLA系统。

![](images/17.jpg)

![](images/18.jpg)  

Figure 18: This conceptual illustration presents "Eva," a future humanoid assistant powered by Vision-Language Models (VLMs), VLA frameworks, and agentic AI systems. VLMs enable semantic scene understanding and object affordance prediction, while VLAs translate language-grounded instructions into hierarchical motor plans. Agentic AI modules ensure adaptive learning, selfrefinement, and interactive decision-making in open-ended environments. Together, these components represent a foundational blueprint for Artificial General Intelligence (AGI) in robotics, where perception, language understanding, planning, and safe autonomous behavior converge in real-world, socially aware tasks.

数据集偏差、基础与未见任务的泛化。稳健的泛化要求数据多样性扩展和先进的学习范式。策划大规模的去偏见多模态数据集，将网络规模的图像—文本语料（如LAION-5B [194]）与以机器人为中心的轨迹档案（如Open XEmbodiment [227]）结合，为公平的语义基础打下基础。硬负样本采样和视觉语言主干网络（如CLIP变体）的对比微调可以减轻虚假相关性并增强语义保真性 [17, 282]。元学习框架通过学习跨任务家族的共享先验，使其能够快速适应新任务，在视觉语言机器人导航模型中得到了验证 [175]。具有重放缓冲区和正则化策略的持续学习算法在整合新概念时保留旧知识，解决了VLA模型中的灾难性遗忘问题 [48]。来自3D感知领域的迁移学习（如在3D-VLA [288]中的点云推理）可以为模型赋予更强的空间归纳偏置，从而提高对异常分布场景的稳健性。最后，进行模拟到现实（sim2real）的微调，结合领域随机化和实际校准（如动态光照、纹理和物理变化），确保在合成环境中学习到的策略能够有效迁移到物理机器人上 [4, 66]。这些综合策略将使视觉语言模型能够自信地泛化到现实部署中的未见物体、场景和任务。

系统集成复杂性和计算需求。为了在紧张的计算预算下管理多模态管道的复杂协调，研究人员必须采用模型模块化和硬件—软件协同设计。低秩适配（LoRA）适配器可以注入到预训练的变换器层中，实现针对特定任务的微调，而无需修改核心权重。通过知识蒸馏，将大型“教师”视觉语言模型（VLA）转移到轻量级“学生”网络，采用基于互信息的目标指导，鼓励学生匹配教师的中间表示和动作分布，生成参数减少$5 \mathrm { - } 1 0 \times$的紧凑模型，同时保留$9 0 { - } 9 5 \%$的任务性能。混合精度量化结合量化感知训练可以将权重压缩到48位，降低内存带宽和能耗超过$60 \%$。为支持稀疏张量运算、动态令牌路由和融合视觉语言内核而定制的硬件加速器，能够在$2 0 { - } 3 0 \mathrm { ~ W ~ }$功率范围内提供持续的$^ { 1 0 0 + }$ TOPS吞吐量，满足嵌入式机器人平台的需求。像TensorRT-LLM和TVM这样的工具链可以针对特定边缘设备优化端到端VLA图，融合层并预计算静态子图。新兴架构如TinyVLA展示了参数少于1亿的VLA能够在操作基准上达到接近最先进的性能，并支持实时推理，为资源受限环境的广泛部署铺平了道路。• VLA部署中对环境变异性的鲁棒性。确保VLA在现实环境中的鲁棒性能需要针对性的技术干预，以应对环境不确定性和长期系统漂移。领域随机化和合成数据增强管道，如UniSim的闭环传感器模拟器，生成光照、遮挡和传感器噪声的光照真实变化，从而提高对分布变化的适应性。此外，能够根据实时反馈动态调整感知阈值和控制增益的自适应重标定模块，可以减轻由传感器老化或操作条件变化引起的性能下降。综上，这些方法旨在提高VLA系统在多样化和不断演变的部署场景下的稳定性和可靠性。

# •大语言模型中的伦理、隐私和社会考量

部署。除了技术上的稳健性，VLA系统的部署还带来了重要的伦理和社会挑战，这些挑战需要以治理为导向的解决方案。需要偏见审计工具来识别训练数据中偏斜的人口或语义分布，并制定相应的纠正策略，如对抗性去偏和反事实数据增强。此外，隐私保护推理机制，包括设备端处理、敏感数据流的同态加密以及训练过程中的差分隐私，对于保护用户数据在医疗和智能家居等领域至关重要。此外，透明的影响评估、利益相关者参与和劳动力技能提升计划有助于管理社会经济影响，而监管框架和行业标准对确保问责制和负责任的VLA采纳至关重要。

# 5.2. 未来路线图

基于视觉语言模型（VLA）系统的未来预计将在日益强大的多模态基础、代理推理和具身持续学习的交汇处发展。在未来十年，我们预见到几种汇聚趋势将推动VLA从能力强大但脆弱的任务专才朝着可靠的普适机器人智能转变。然而，这一发展轨迹将受到先前强调的持续制约/限制的影响：（i）闭环控制中的实时推理瓶颈，（ii）不完整的多模态行动表征和薄弱的安全保障，（iii）数据集偏差和分布转移下的基础失败，（iv）感知-记忆-推理-控制的集成复杂性，（v）高计算和能量需求限制边缘部署，以及（vi）开放世界环境中的鲁棒性、透明性和伦理问题。为全面解决这些问题，图19总结了系统级的研究路线图，图18则直观地展示了视觉语言模型（VLM）、VLA架构和代理人工智能模块如何共同发展，朝着机器人中的具身人工通用智能（AGI）演进。

多模态基础模型作为具身感知的“皮层”：当前的视觉语言架构通常依赖于视觉语言主干网络与任务特定策略头的结合，这限制了通用知识的重用并增加了跨领域再训练的成本。一个合理的下一步是训练一个统一的多模态基础模型，该模型基于网络规模的图像、视频、文本和交互/可用性轨迹，作为一个共享的“皮层”，不仅编码静态语义，还编码动态、接触先验和常识物理知识[295, 289, 272]。这样的皮层可以通过将语言基础于以对象为中心的表征和持久的场景结构，减少由浅层关联造成的失败模式[206]。如图18所强调的，这种基础模型皮层将使机器人能够将环境分割为可行动实体（对象、区域、可用性），并为下游规划者和控制器提供稳定的语义锚定。然而，为了防止过度自信和幻觉性基础，这些模型必须结合校准的不确定性和与证据相关的推理，以确保感知驱动的计划在遮挡、杂乱和模棱两可的指令下仍然可验证[145]。

![](images/19.jpg)  
s ehi orEfit De ate, e,  Re $\boldsymbol { \mathcal { E } } _ { \mathcal { F } }$ Safe Intelligence (grounding, uncertainty. safety assurance), and (iii) Unified Systems $\boldsymbol { \mathcal { E } } _ { \mathcal { F } }$ Governance (2D-temporal-3D integration, transfer, evaluation, and responsible deployment).

智能、 自我监督、终身学习与持续适应：当前视觉语言助手（VLA）的一个明确局限性是其静态特性：训练一次的策略在面临非平稳环境时并未发生变化。未来的VLA应采用智能学习循环，使模型能够提出探索目标、假设结果，并通过模拟和真实推演进行自我修正，从而在数月或数年中实现技能的持续增长。这一方向预计能够缓解分布偏移、数据集偏见和长期脆弱性，允许模型随时间适应；然而，持续的策略更新也带来了新的风险。尤其是，反复的在线或增量学习可能会覆盖先前获得的能力（灾难性遗忘），引发非预期的行为退化，并增加对嘈杂、对抗性或非意图环境反馈的敏感性，这些反馈可能会损害已学习的策略。因此，终身学习必须与重放和安全意识更新、模块适配器及基于验证的策略修订相结合。在图19的路线图中，这种智能终身学习范式自然位于可靠与安全智能与统一系统与治理的交集处，其中持续学习被视为一个被控制、可审计的生命周期过程，而不是临时微调的一步。 可扩展性和可解释性的层次化神经符号规划：从低级运动原语扩展到长期目标需要明确的层次结构。下一代VLA系统可能会使用基于语言的规划器（微调用于可用性和约束的LLM风格模块），将目标分解为结构化的子任务，随后进行中级技能策略和低级控制器的设计，以确保合规运动。这种神经符号融合有助于通过施加更易于调试、监控和验证的接口，来缩小集成复杂度。重要的是，层次结构还支持选择性验证：可以检查高级计划是否违反约束（不安全步骤、禁止区域），而低级轨迹则可以通过控制障碍函数、模型预测控制（MPC）和运行时安全监控来保护。这些组件在“可靠与安全智能”的支柱下与图19对齐，其中安全性通过规划时间约束和执行时间保护来强制，而不仅仅依赖于事后评估。

通过世界模型和物理/因果推理进行实时适应：在非结构化环境中稳健部署要求视觉语言智能体（VLA）保持内部的对象、接触和动态的预测模型。能够预测近期状态转变和失败概率的世界模型可以支持反事实评估（“如果我在这里推，会发生什么碰撞？”）以及在现实偏离预期时（例如，抓握滑动、意外摩擦）迅速采取纠正措施[203]。这一能力对于安全的操作、导航和人机交互至关重要，因为小错误会迅速累积[156]。然而，世界模型必须足够高效以便在设备上使用，并与多传感器证据保持一致。因此，未来一个重要方向是硬件感知的、内存高效的预测建模，例如时序词元压缩[267，216]、事件驱动的状态更新[250，226]以及将可微分物理模拟器与学习到的动态相结合的混合物理学习模型，以实现与控制相关的更新速率[102，52]。在图19中，这些需求共同出现在高效部署支柱（实时约束）和可靠与安全智能支柱（用于扎根决策的物理/因果关系）中。

效率与可扩展性：将通用性与边缘部署相结合。大规模多模态主干网络的计算资源需求与闭环控制的延迟/能量限制之间的不匹配仍然是VLA采纳的一个主要障碍。未来的VLA应优先考虑参数高效设计（结构稀疏、低秩适配、模块化专家），在降低推理成本的同时保持泛化能力。除了训练时的效率，任何时间/提前退出策略都可以自适应分配计算，确保安全关键步骤保持高保真度，同时常规步骤使用更低成本的路径。同样重要的是动作空间效率：紧凑的动作词元化和分块控制表示缩短自回归范围，使得在不牺牲时间平滑性的情况下实现更高的控制速率。这些模型级选择必须与基于硬件的编译相结合，涵盖GPU/NPU/边缘加速器，包括量化感知调度和内存优化注意力内核。计算感知缓存和情节记忆进一步减少冗余的前向传递，提高长时间任务的响应能力。这些方向共同落实了图19中的高效部署支柱，并直接解决了之前强调的计算/能量限制。 跨机体转移和形态无关的技能表示：为每种机器人形态训练独立的VLA的方法不太可能扩展。未来的关键主题是与机体无关的策略学习，技能在抽象的动作空间中表达（例如，接触目标、适配点操作、任务空间约束），这些技能在轮式平台、四足机器人和类人机器人之间转移。元学习和少样本校准可以使在新的机器人上根据几分钟的数据快速启动，而不是几周的训练。这一方向还通过在不同机体和环境之间强制不变性减轻了数据集偏见，但它需要标准化表示、共同接口和可重复的评估协议。在图19中，这一与机体无关的技能学习范式位于统一系统 $\boldsymbol { \mathcal { E } } _ { \mathcal { F } }$ 的治理之下，连接了架构统一与原则性的转移和基准。 超越任务成功的评估：安全、恢复和资源感知指标。VLA的进展需要反映部署现实的测量。单靠任务成功并不能清晰地表示失败的严重性、时间一致性、不安全的近失和能量低效。未来的基准应该量化安全违规、确定性校准、恢复行为、时间一致性、能量消耗和在人的约束下的下游效用。此外，评估应报告计算预算、数据集组成和部署条件，以便进行公平比较并诊断偏见驱动的收益。这种测量不仅仅是科学的卫生：它是审计和治理的基础，决定了系统是否能够负责任地大规模部署。这一动机构成了图19中的评估分支，并补充了图18中的概念部署叙述。 安全、伦理和以人为本的对齐作为首要设计目标：随着VLA获得自主性，内置的安全性和价值对齐变得至关重要。未来的系统应集成实时风险评估工具，在执行高风险操作之前评估潜在危害，在模糊情况下请求自然语言确认，并保持透明的日志以便问责。隐私意识感知、偏见审计和人机协作监督必须嵌入生命周期中，特别是在辅助机器人和安全关键自主系统中。与监管一致的评估协议和标准化工作对于将VLA进展转化为可信的现实世界系统至关重要。这一治理视角在图19中明确体现，并在图18中通过社会意识的类人机器人设置隐含反映。

跨领域主题：持续学习、故障恢复、互动和控制精度：在图19的所有支柱中，预期有几个跨领域主题将塑造未来十年的VLA研究。首先，持续和终身学习必须是安全的、可审计的，并且具有抗遗忘能力，以支持长期适应而不 destabilize 部署系统。其次，故障检测和恢复应被视为一流能力，结合自省监测、考虑不确定性的感知以及在执行偏离预期结果时的结构化恢复行为。第三，提高动作生成的精确性和可靠性仍然至关重要：虽然基于VLA的规划者可以实现灵活的、以语言为条件的决策，但其轨迹精度和控制稳定性目前仍落后于传统分析方法，如模型预测控制、基于采样的运动规划和反馈线性控制器。因此，结合VLA驱动的高层规划与经典或学习的低层控制器的混合架构预计将在实现语义灵活性和控制级精确性方面发挥核心作用。最后，人机对齐和互动需要意图澄清、共享自主性和可解释的行动理由机制，以支持在多种真实世界环境中的信任和可用性。总之，图19强调，缩小实验室演示与稳健现实世界部署之间的差距将需要在效率、安全性、数据和泛化、系统集成、评估和治理方面的协调进展。补充地，图18展示了这些进展如何在移动机器人、机械臂、辅助系统和类人的多个平台上汇聚成通用的具身智能体，其中多模态感知、层级规划、持续适应和人机对齐的安全性集成在统一的智能堆栈中。共同处理这些方向预计将把VLA从有前景的研究原型转变为可靠、广泛适用的具身系统，而非仅限于任何单一机器人形态的解决方案。

# 6. 结论

在这篇综合评述中，我们系统地评估了过去三年发布的视觉-语言-行动（VLA）模型的最新发展、方法论和应用。我们的分析始于VLA的基础概念，定义其作为统一视觉感知、自然语言理解和在物理或模拟环境中生成行动的多模态系统。我们追溯了其演变和时间线，详细列出了标志性里程碑，这些里程碑标志着从孤立的感知-行动模块到完全统一的、能遵循指令的机器人智能体的转变。我们强调了多模态集成如何从松散耦合的管道发展到基于变换器的架构，从而实现模态之间的无缝协调。接下来，我们研究了词元化和表示技术，重点探讨了VLA如何编码视觉和语言信息，包括行动原语和空间语义。

我们探讨了学习范式，详细叙述了监督学习、模仿学习、强化学习及多模态预训练等训练策略和数据集，这些都塑造了视觉语言模型（VLA）的性能。在“自适应控制和实时执行”部分，我们讨论了现代VLA如何优化以适应动态环境，分析了支持延迟敏感任务的策略。随后，我们对主要的架构创新进行了分类，调查了50多个近期的VLA模型。这部分讨论包括了模型设计、内存系统和交互保真度的进展。我们进一步研究了提高训练效率的策略，包括像LoRA这样的参数高效方法、量化和模型剪枝，以及像并行解码和硬件感知推理这样的加速技术。我们对现实世界应用的分析突显了VLA模型在六个领域的潜力和当前局限性：类人机器人、自主车辆、工业自动化、医疗保健、农业和增强现实（AR）导航。在这些环境中，VLA展现出了强大的高层语义推理、指令跟随和任务概括能力，特别是在结构化或部分受控的环境中。然而，与传统的分析规划和控制流程相比，它们的有效性常常受到实时推理延迟、环境变异下的有限鲁棒性以及长时间跨度或安全关键控制时的精度降低的限制。此外，应用特定的适配和广泛的数据整理常常是实现可靠性能所必需的，这突显了可扩展性和部署方面的挑战。这些发现表明，虽然VLA非常适合语义决策和灵活任务规范，但将VLA推理与经典或学习的低级控制器整合的混合架构在实际的世界操作中仍然至关重要。面对挑战和限制，我们集中在五个核心领域：实时推理、多模态动作表示和安全性、偏见与概化、系统集成与计算约束，以及伦理部署。我们提出了基于当前文献的潜在解决方案，包括模型压缩、跨模态基础、领域适配和智能学习框架。最后，我们的讨论和未来路线图阐述了视觉语言模型、VLA架构和智能AI系统的融合如何引导机器人朝向人工通用智能（AGI）的发展。此次综述提供了对VLA进展的统一理解，识别了未解决的挑战，并为未来开发智能化、具身化和与人类对齐的智能体描绘了一条结构化的前进路线。

# 资金声明

本研究部分得到了国家科学基金会（NSF）和美国农业部（USDA）、国家食品与农业研究所（NIFA）的支持，通过“农业人工智能（AI）研究所”项目，奖金编号为AWD003473和AWD004595，以及USDA-NIFA接收编号1029004，项目标题为“使用柔性操作器的机器人花朵疏摘”。额外支持来自USDA/NIFA赠款编号2024-67022-41788，接收编号为1031712，项目为“将UCF的人工智能研究扩展到新颖农业工程应用（PARTNER）”。

# 声明

作者声明不存在利益冲突。

# 关于人工智能写作辅助的声明

ChatGPT 和 Perplexity 被用于提高语法准确性和优化句子结构；所有 AI 生成的修订均经过仔细审查和编辑，以确保相关性。

# References

[1] Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al., 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 .   
[2] Agarwal, L., Verma, B., 2024. From methods to datasets: A survey on image-caption generators. Multimedia Tools and Applications 83, 2807728123.   
[3] Alayrac, J.B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al., 2022. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems 35, 2371623736.   
[4] Anderson, P., Shrivastava, A., Truong, J., Majumdar, A., Parikh, D., Batra, D., Lee, S., 2021. Sim-to-real transfer for vision-and-language navigation, in: Conference on Robot Learning, PMLR. pp. 671681.   
[5] Arai, H., Miwa, K., Sasaki, K., Watanabe, K., Yamaguchi, Y., Aoki, S., Yamamoto, I., 2025. Covla: Comprehensive vision-language-action dataset for autonomous driving, in: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), IEEE. pp. 19331943.   
[6] Asif, S., Bueno, M., Ferreira, P., Anandan, P., Zhang, Z., Yao, Y., Ragunathan, G., Tinkler, L., Sotoodeh-Bahraini, M., Lohse, N., et al., 2025. Rapid and automated configuration of robot manufacturing cells. Robotics and Computer-Integrated Manufacturing 92, 102862.   
[7] Assres, G., Bhandari, G., Shalaginov, A., Gronli, T.M., Ghinea, G., 2025. State-of-the-art and challenges of engineering ml-enabled software systems in the deep learning era. ACM Computing Surveys .   
[8] Asuzu, K., Singh, H., Idrissi, M., 2025. Humanrobot interaction through joint robot planning with large language models. Intelligent Service Robotics , 117.   
[9] Ayaz, M., Khan, M., Saqib, M., Khelifi, A., Sajjad, M., Elsadik, A. 0Medvlm: Medical vision-agge model for consumer devices. IEEE Consumer Electronics Magazine .   
[10] Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., Lin, J., 2025. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923   
[11] Bartoccioni, F., Ramzi, E., Besnier, V., Venkataramanan, S., Vu, T.H., Xu, Y., Chambon, L., Gidaris, S., Odabas, S., Hurych, D., et al., 2025. Vavim and vavam: Autonomous driving through video generative modeling. arXiv preprint arXiv:2502.15672 .   
[12] Bathula, N.V., Paleti, I., Pagidi, S., Akkumahanthi, S.S., Guduru, N.T., 2024. Policy learning-based image captioning with vision transformer, in: 2024 IEEE International Students' Conference on Electrical, Electronics and Computer Science (SCEECS), IEEE. pp. 16.   
[13] Belkhale, S., Ding, T., Xiao, T., Sermanet, P., Vuong, Q., Tompson, J., Chebotar, Y., Dwibedi, D., Sadigh, D., 2024. Rt-h: Action hierarchies using language. arXiv preprint arXiv:2403.01823 .   
[14] Bjorck, J., Castañeda, F., Cherniadev, N., Da, X., Ding, R., Fan, L., Fang, Y., Fox, D., Hu, F., Huang, S., et al., 2025. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734   
[15] Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K. Ichter, B., et al., 2024. Pi-0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164 .   
[16] Bolya, D., Huang, P.Y., Sun, P., Cho, J.H., Madotto, A., Wei, C., Ma, T., Zhi, J., Rajasegaran, J., Rasheed, H., et al., 2025. Perception encoder: The best visual embeddings are not at the output of the network. arXiv preprint arXiv:2504.13181 .   
[17] Bordes, F., Pang, R.Y., Ajay, A., Li, A.C., Bardes, A., Petryk, S., Mañas, O., Lin, Z., Mahmoud, A., Jayaraman, B. al. Ani visi-a modeling. arXiv preprint arXiv:2405.17247 .   
[18] Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Chen, X., Choromanski, K., Ding, T., Driess, D., Dubey, A., Finn, C., et al., 2023. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818 .

[19] Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Dabis, J., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., Hsu, J., et al., 2022. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817 .

[20] Budzianowski, P., Maa, W., Freed, M., Mo, J., Xie, A., Tipnis, V., Bolte, B., 2024. Edgevla: Efficient visionlanguage-action models. environments 20, 3.

[21] Cangelosi, A., Metta, G., Sagerer, G., Nolfi, S., Nehaniv, C., Fischer, K., Tani, J., Belpaeme, T., Sandini, G., Nori, F., et al., 2010. Integration of action and language knowledge: A roadmap for developmental robotics. IEEE Transactions on Autonomous Mental Development 2, 167195.

[22] Cao, J., Gan, Z., Cheng, Y., Yu, L., Chen, Y.C., Liu, J., 2020. Behind the scene: Revealing the secrets of pre-trained vision-and-language models, in: Computer VisionECCV 2020: 16th European Conference, Glasgow, UK, August 2328, 2020, Proceedings, Part VI 16, Springer. pp. 565580.

[23] Cao, L., 2024. Ai robots and humanoid ai: Review, perspectives and directions. arXiv preprint arXiv:2405.15775 .

[24] Cao, Y., Ju, Y., Xu, D., 2024a. 3dgs-det: Empower 3d gaussian splatting with boundary guidance and boxfocused sampling for 3d object detection. arXiv preprint arXiv:2410.01647 .

[25] Cao, Y., Zeng, Y., Xu, H., Xu, D., 2023. Coda: Collaborative novel box discovery and cross-modal alignment for open-vocabulary 3d object detection, in: NeurIPS.

[26] Cao, Y., Zeng, Y., Xu, H., Xu, D., 2024b. Collaborative novel object discovery and box-guided crossmodal alignment for open-vocabulary 3d object detection. arXiv preprint arXiv:2406.00830 .

[27] Chang, C., Shi, Y., Cao, D., Yang, W., Hwang, J., Wang, H., Pang, J., Wang, W., Liu, Y., Peng, W.C., et al., 2025. A survey of reasoning and agentic systems in time series with large language models. arXiv preprint arXiv:2509.11575 .

[28] Chang, Y., Wang, X., Wang, J., Wu, Y., Yang, L., Zhu, K., Chen, H., Yi, X., Wang, C., Wang, Y., et al., 2024. A survey on evaluation of large language models. ACM transactions on intelligent systems and technology 15, 145.

[29] Chatzopoulos, D., Bermejo, C., Huang, Z., Hui, P., 2017. Mobile augmented reality survey: From where we are to where we go. Ieee Access 5, 69176950.

[30] Chen, B., Xu, Z., Kirmani, S., Ichter, B., Sadigh, D., Guibas, L., Xia, F., 2024a. Spatialvlm: Endowing

vision-language models with spatial reasoning capabilities, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14455 14465.   
[31] Chen, H., Hou, L., Wu, S., Zhang, G., Zou, Y., Moon, S., Bhuiyan, M., 2024b. Augmented reality, deep learning and vision-language query system for construction worker safety. Automation in Construction 157, 105158.   
[32] Chen, H., Li, S., Fan, J., Duan, A., Yang, C., NavarroAlarcon, D., Zheng, P., 2025a. Human-in-the-loop robot learning for smart manufacturing: A human-centric perspective. IEEE Transactions on Automation Science and Engineering .   
[33] Chen, H., Liu, B., Wang, S., Wang, X., Han, W., Zhu, Y., Wang, X., Bi, Y., 2025b. Language modulates vision: Evidence from neural networks and human brain-lesion models. arXiv preprint arXiv:2501.13628   
[34] Chen, P., Bu, P., Wang, Y., Wang, X., Wang, Z., Guo, J., Zhao, Y., Zhu, Q., Song, J., Yang, S., et al., 2025c. Combatvla: An efficient vision-language-action model for combat tasks in 3d action role-playing games. arXiv preprint arXiv:2503.09527 .   
[35] Chen, X., Xu, W., Kan, S., Zhang, L., Jin, Y., Cen, Y., Li, Y., 2025d. Vision-semantics-label: A new twostep paradigm for action recognition with large language model. IEEE Transactions on Circuits and Systems for Video Technology .   
[36] Chen, Y., Tian, S., Liu, S., Zhou, Y., Li, H., Zhao, D., 2025e. Conrft: A reinforced fine-tuning method for vla models via consistency policy. arXiv preprint arXiv:2502.05450 .   
[37] Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, Q., Zhu, X., Lu, L., et al., 2024c. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2418524198.   
[38] Cheng, A.C., Ji, Y., Yang, Z., Gongye, Z., Zou, X., Kautz, J., B1yk, E., Yin, H., Liu, S., Wang, X., 2024a. Navila: Legged robot vision-language-action model for navigation. arXiv preprint arXiv:2412.04453 .   
[39] Cheng, H., Xiao, E., Yu, C., Yao, Z., Cao, J., Zhang, Q., W, J. Su M., Xu K. Gu, J, l - ulation facing threats: Evaluating physical vulnerabilities in end-to-end vision language action models. arXiv preprint arXiv:2409.13174 .   
[40] Chi, C., Xu, Z., Feng, S., Cousineau, E., Du, Y., Burchfiel, B., Tedrake, R., Song, S., 2023. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research , 02783649241273668.

[1] Chi, H., Gao, H.a. Liu, Z., Liu, J., Liu, C., Li, J.,ang, K., Yu, Y., Wang, Z., Li, W., et al., 2025. Impromptu vla: Open weights and open data for driving vision-languageaction models. arXiv preprint arXiv:2505.23757 .

[42] Chiang, H.T.L., Xu, Z., Fu, Z., Jacob, M.G., Zhang, T., Lee, T.W.E., Yu, W., Schenck, C., Rendleman, D., Shah, D., et al., 2024. Mobility vla: Multimodal instruction navigation with long-context vlms and topological graphs. arXiv preprint arXiv:2407.07775 .

[43] Chowa, S.S., Alvi, R., Rahman, S.S., Rahman, M.A., Raiaan, M.A.K., Islam, M.R., Hussain, M., Azam, S., 2026. From language to action: a review of large language models as autonomous agents and tool users. Artificial Intelligence Review .

[44] Dang, R., Yuan, Y., Zhang, W., Xin, Y., Zhang, B., Li, L., Wang, L., Zeng, Q., Li, X., Bing, L., 2025. Ecbench: Can multi-modal foundation models understand the egocentric world? a holistic embodied cognition benchmark. arXiv preprint arXiv:2501.05031 .

[45] Dasari, S., Ebert, F., Tian, S., Nair, S., Bucher, B., Schmeckpeper, K., Singh, S., Levine, S., Finn, C., 2019. Robonet: Large-scale multi-robot learning. arXiv preprint arXiv:1910.11215 .

[46] Deng, S., Yan, M., Wei, S., Ma, H., Yang, Y., Chen, J., Zhang, Z., Yang, T., Zhang, X., Cui, H., Zhang, Z., Wang, H., 2025a. Graspvla: a grasping foundation model pre-trained on billion-scale synthetic action data. URL: https://arxiv.org/abs/2505.03233, arXiv:2505.03233.

[47] Deng, S., Yan, M., Zheng, Y., Su, J., Zhang, W., Zhao, X., Cui, H., Zhang, Z., Wang, H., 2025b. Stereovla: Enhancing vision-language-action models with stereo vision. arXiv preprint arXiv:2512.21970 .

[48] Dey, S., Zaech, J.N., Nikolov, N., Van Gool, L., Paudel, D.P., 2024. Revla: Reverting visual domain limitation of robotic foundation models. arXiv preprint arXiv:2409.15250 .

[49] Din, M.U., Akram, W., Saoud, L.S., Rosell, J., Hussain, I., 2025. Vision language action models in robotic manipulation: A systematic review. arXiv preprint arXiv:2507.10672 .

[50] Ding, D., Yao, T., Luo, R., Sun, X., 2025a. Visual question answering in robotic surgery: A comprehensive review. IEEE Access .

[51] Ding, J., Zhang, Y., Shang, Y., Zhang, Y., Zong, Z. Feng, J., Yuan, Y., Su, H., Li, N., Sukiennik, N., et al., 2024a. Understanding world or predicting future? a comprehensive survey of world models. arXiv preprint arXiv:2411.14499 .

[52] Ding, M., Chen, Z., Du, T., Luo, P., Tenenbaum, J., Gan, C., 2021. Dynamic visual reasoning by learning differentiable physics models from video and language. Advances in Neural Information Processing Systems 34, 887899.

[53] Ding, P., Ma, J., Tong, X., Zou, B., Luo, X., Fan, Y., Wang, T., Lu, H., Mo, P., Liu, J., et al., 2025b. Humanoid-vla: Towards universal humanoid control with visual integration. arXiv preprint arXiv:2502.14795 [54] Ding, P., Zhao, H., Zhang, W., Song, W., Zhang, M., Huang, S., Yang, N., Wang, D., 2024b. Quar-vla: Visionlanguage-action model for quadruped robots, in: European Conference on Computer Vision, Springer. pp. 352367.

[55] Donahue, J., Anne Hendricks, L., Guadarrama, S., Rohrbach, M., Venugopalan, S., Saenko, K., Darrell, T., 2015. Long-term recurrent convolutional networks for visual recognition and description, in: Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 26252634.

[56] Dong, H., Liu, M., Zhou, K., Chatzi, E., Kannala, J., Stachniss, C., Fink, O., 2025. Advances in multimodal adaptation and generalization: From traditional approaches to foundation models. arXiv preprint arXiv:2501.18592 .

[57] Doveh, S., Arbelle, A., Harary, S., Schwartz, E., Herzig, R., Giryes, R., Feris, R., Panda, R., Ullman, S., Karlinsky, L., 2023. Teaching structured vision & language concepts to vision & language models, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26572668.

[58] Driess, D., Xia, F., Sajjadi, M.S., Lynch, C., Chowdhery, A., Wahid, A., Tompson, J., Vuong, Q., Yu, T., Huang, W., et al., 2023. Palm-e: An embodied multimodal language model. Openreview .

[59] Duan, J., Pumacay, W., Kumar, N., Wang, Y.R., Tian, S., Yuan, W., Krishna, R., Fox, D., Mandlekar, A., Guo, Y., 2024. Aha: A vision-language-model for detecting and reasoning over failures in robotic manipulation. arXiv preprint arXiv:2410.00371 .

[60] Duarte, N.F., Rakovi, M., Tasevski, J., Coco, M.I., Billard, A., Santos-Victor, J., 2018. Action anticipation: Reading the intentions of humans and robots. IEEE Robotics and Automation Letters 3, 41324139.

[61] Ebert, F., Yang, Y., Schmeckpeper, K., Bucher, B., Georgakis, G., Daniilidis, K., Finn, C., Levine, S., 2021. Bridge data: Boosting generalization of robotic skills with cross-domain datasets. arXiv preprint arXiv:2109.13396 .

[62] Fan, C., Jia, X., Sun, Y., Wang, Y., Wei, J., Gong, Z., Zhao, X., Tomizuka, M., Yang, X., Yan, J., Ding, M., 2025a. Interleave-vla: Enhancing robot manipulation with interleaved image-text instructions. URL: https://arxiv.org/abs/2505.02152, arXiv:2505.02152.

[63] Fan, L., Chen, K., Xu, Z., Yuan, M., Huang, P., Huang, W., 2024. Language reasoning in vision-language-action model for robotic grasping, in: 2024 China Automation Congress (CAC), IEEE. pp. 66566661.

[64] Fan, Y., Ding, P., Bai, S., Tong, X., Zhu, Y., Lu, H., Dai, F., Zhao, W., Liu, Y., Huang, S., et al., 2025b. Long-vla: Unleashing long-horizon capability of vision language action model for robot manipulation. arXiv preprint arXiv:2508.19958 .

[5] Fang, H., Liu, Y., Du, Y., Du, L., Yang, H., 2025a. Sqapvla: A synergistic quantization-aware pruning framework for high-performance vision-language-action models. arXiv preprint arXiv:2509.09090 .

[66] Fang, Y., Yang, Y., Zhu, X., Zheng, K., Bertasius, G., Szafir, D., Ding, M., 2025b. Rebot: Scaling robot learning with real-to-sim-to-real robotic video synthesis. arXiv preprint arXiv:2503.14526 .

[67] Firoozi, R., Tucker, J., Tian, S., Majumdar, A., Sun, J., Liu, W., Zhu, Y., Song, S., Kapoor, A., Hausman, K., et al., 2023. Foundation models in robotics: Applications, challenges, and the future. The International Journal of Robotics Research , 02783649241281508.

[68] Foster, D.J., Block, A., Misra, D., 2024. Is behavior cloning all you need? understanding horizon in imitation learning. arXiv preprint arXiv:2407.15007 .

[69] Fu, H., Zhang, D., Zhao, Z., Cui, J., Liang, D., Zhang, C., Zhang, D., Xie, H., Wang, B., Bai, X., 2025. Orion: A holistic end-to-end autonomous driving framework by vision-language instructed action generation. arXiv preprint arXiv:2503.19755 .

[70] Gao, B., Liu, Y., Li, Y., Li, H., Li, M., He, W., 2025a. A vision-language model for predicting potential distribution land of soybean double cropping. Frontiers in Environmental Science 12, 1515752.

[71] Gao, C., Liu, Z., Chi, Z., Huang, J., Fei, X., Hou, Y., Zhang, Y., Lin, Y., Fang, Z., Jiang, Z., et al., 2025b. Vlaos: Structuring and dissecting planning representations and paradigms in vision-language-action models. arXiv preprint arXiv:2506.17561 .

[72] Gao, J., Belkhale, S., Dasari, S., Balakrishna, A., Shah, D., Sadigh, D., 2025c. A taxonomy for evaluating generalist robot policies. arXiv preprint arXiv:2503.01238 [73] Gao, S.H., Cheng, M.M., Zhao, K., Zhang, X.Y., Yang, M.H., Torr, P., 2019. Res2net: A new multi-scale backbone architecture. IEEE TPAMI .

[74] Gbagbe, K.F., Cabrera, M.A., Alabbas, A., Alyunes, O., Lykov, A., Tsetserukou, D., 2024. Bi-vla: Vision-language-action model-based system for bimanual robotic dexterous manipulations, in: 2024 IEEE International Conference on Systems, Man, and Cybernetics (SMC), IEEE. pp. 28642869.

[75] Geens, R., 2024. Bringing generative ai to edge devices through interoperable compute cores, in: Flanders AI Research Day, Location: Ghent.

[76] Ghosh, A., Acharya, A., Saha, S., Jain, V., Chadha, A., 2024. Exploring the frontier of vision-language models: A survey of current methodologies and future directions. arXiv preprint arXiv:2404.07214 .

[77] Gu, J., Wang, Z., Kuen, J., Ma, L., Shahroudy, A., Shuai, B., Liu, T., Wang, X., Wang, G., Cai, J., et al., 2018. Recent advances in convolutional neural networks. Pattern recognition 77, 354377.

[78] Gu, Q., Ju, Y., Sun, S., Gilitschenski, I., Nishimura, H., Itkina, M., Shkurti, F., 2025a. Safe: Multitask failure detection for vision-language-action models. arXiv preprint arXiv:2506.09937 .

[79] Gu, Q., Su, J., Yuan, L., 2021. Visual affordance detection using an efficient attention convolutional neural network. Neurocomputing 440, 3644. doi:https: //doi.org/10.1016/j.neucom.2021.01.018.

[80] Gu, X., Wen, C., Ye, W., Song, J., Gao, Y., 2023. Seer: Language instructed video prediction with latent diffusion models. arXiv preprint arXiv:2303.14897 .

[81] Gu, Z., Li, J., Shen, W., Yu, W., Xie, Z., McCrory, S., Cheng, X., Shamsah, A., Griffin, R., Liu, C.K., et al., 2025b. Humanoid locomotion and manipulation: Current progress and challenges in control, planning, and learning. arXiv preprint arXiv:2501.02116 .

[82] Guan, W., Hu, Q., Li, A., Cheng, J., 2025. Efficient vision-language-action models for embodied manipulation: A systematic survey. arXiv preprint arXiv:2510.17111 .

[83] Guo, Y., Zhang, J., Chen, X., Ji, X., Wang, Y.J., Hu, Y., Chen, J., 2025. Improving vision-language-action model with online reinforcement learning. arXiv preprint arXiv:2501.16664 .

[84] Guruprasad, P., Sikka, H., Song, J., Wang, Y., Liang, P.P., 2024. Benchmarking vision, language, & action models on robotic learning tasks. arXiv preprint arXiv:2411.05821 .

[85] Haldar, S., Peng, Z., Pinto, L., 2024. Baku: An efficient transformer for multi-task policy learning. arXiv preprint arXiv:2406.07539 .

[86] Han, S., Wang, M., Zhang, J., Li, D., Duan, J., 2024. A review of large language models: Fundamental architectures, key technological evolutions, interdisciplinary technologies integration, optimization and compression techniques, applications, and challenges. Electronics 13, 5040.

[87] Hanson, A., Riseman, E., 2014. The visions imageunderstanding system, in: Advances in Computer Vision. Psychology Press, pp. 1114.

[88] Hao, P., Zhang, C., Li, D., Cao, X., Hao, X., Cui, S., Wang, S., 2025. Tla: Tactile-language-action model for contact-rich manipulation. arXiv preprint arXiv:2503.08548 .

[89] He, K., Zhang, X., Ren, S., Sun, J., 2016. Deep residual learning for image recognition, pp. 770778.

[90] Holldack, F., Banh, L., Strobel, G., 2026. Agentic information systems. Electronic Markets 36, 5.

[91] Hong, Y., 2025. Building 3D Foundation Models for the Embodied Minds. Ph.D. thesis. University of California, Los Angeles.

[92] Hou, Z., Zhang, T., Xiong, Y., Duan, H., Pu, H., Tong, R., Zhao, C., Zhu, X., Qiao, Y., Dai, J., et al., 2025. Dita: Scaling diffusion transformer for generalist vision-language-action policy. arXiv preprint arXiv:2503.19757 .

[93] Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y. Wang, S., Wang, L., Chen, W., et al., 2022. Lora: Lowrank adaptation of large language models. ICLR 1, 3.

[94] Hu, Y., Tang, J., Gong, X., Zhou, Z., Zhang, S., Elvitigala, D.S., Mueller, F., Hu, W., Quigley, A.J., 2025. Vision-based multimodal interfaces: A survey and taxonomy for enhanced context-aware system design. arXiv preprint arXiv:2501.13443 .

[95] Huang, G., Liu, Z., Van Der Maaten, L., Weinberger, K.Q., 2017. Densely connected convolutional networks, in: Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 47004708.

[96] Huang, H., Liu, F., Fu, L., Wu, T., Mukadam, M., Malik, J., Goldberg, K., Abbeel, P., 2024. Early fusion helps vision language action models generalize better, in: 1st Workshop on X-Embodiment Robot Learning.

[97] Huang, H., Liu, F., Fu, L., Wu, T., Mukadam, M., Malik, J., Goldberg, K., Abbeel, P., 2025a. Otter: A visionlanguage-action model with text-aware visual feature extraction. arXiv preprint arXiv:2503.03734 .

[98] Huang, S., Dong, L., Wang, W., Hao, Y., Singhal, S., Ma, S., Lv, T., Cui, L., Mohammed, O.K., Patra, B., et al., 2023a. Language is not all you need: Aligning perception with language models. Advances in Neural Information Processing Systems 36, 7209672109.

[99] Huang, W., Gu, Q., Ye, N., 2025b. Decision spikeformer: Spike-driven transformer for decision making. arXiv preprint arXiv:2504.03800 .

[100] Huang, W., Wang, C., Zhang, R., Li, Y., Wu, J., FeiFei, L., 2023b. Voxposer: Composable 3d value maps for robotic manipulation with language models. arXiv preprint arXiv:2307.05973 .

[101] Huang, Y., Hua, H., Zhou, Y., Jing, P., Nagireddy, M., Padhi, I., Dolcetti, G., Xu, Z., Chaudhury, S., Rawat, A., et al., 2025c. Building a foundational guardrail for general agentic systems via synthetic data. arXiv preprint arXiv:2510.09781 .

[102] Huang, Z., Chen, F., Pu, Y., Lin, C., Su, H., Gan, C., 2023c. Diffvl: Scaling up soft body manipulation using vision-language driven differentiable physics. Advances in Neural Information Processing Systems 36, 29875 29900.

[103] Hung, C.Y., Sun, Q., Hong, P., Zadeh, A., Li, C., Tan, U., Majumder, N., Poria, S., et al., 2025. Nora: A small open-sourced generalist vision language action model for embodied tasks. arXiv preprint arXiv:2504.19854 .

[104] Ikeda, B., Gramopadhye, M., Nekervis, L., Szafir, D., 2025. Marcer: Multimodal augmented reality for composing and executing robot tasks, in: 2025 20th ACM/IEEE International Conference on Human-Robot Interaction (HRI), IEEE. pp. 529539.

[105] Imran, A., Gopalakrishnan, K., 2025. Foundation models in robotics, in: AI for Robotics: Toward Embodied and General Intelligence in the Physical World. Springer, pp. 139210.

[106] Intelligence, P., Black, K., Brown, N., Darpinian, J., Dhabalia, K., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., et al., 2025. pi0.5: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054 .

[107] Jeong, H., Lee, H., Kim, C., Shin, S., 2024. A survey of robot intelligence with large language models. Applied Sciences 14, 8868.

[108] Jha, K., Doshi, A., Patel, P., Shah, M., 2019. A comprehensive review on automation in agriculture using artificial intelligence. Artificial Intelligence in Agriculture 2, 112.

[109] Jiang, J., Xiao, W., Lin, Z., Zhang, H., Ren, T., Gao, Y., Lin, Z., Cai, Z., Yang, L., Liu, Z., 2024. Solami: Social vision-language-action modeling for immersive interaction with 3d autonomous characters. arXiv preprint arXiv:2412.00174 .

[110] Jiang, J., Xiao, W., Lin, Z., Zhang, H., Ren, T., Gao, Y., Lin, Z., Cai, Z., Yang, L., Liu, Z., 2025a. Solami: Social vision-language-action modeling for immersive interaction with 3d autonomous characters, in: Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 2688726898.

[111] Jiang, S., Huang, Z., Qian, K., Luo, Z., Zhu, T., Zhong, Y., Tang, Y., Kong, M., Wang, Y., Jiao, S., et al., 2025b. A survey on vision-language-action models for autonomous driving. arXiv preprint arXiv:2506.24044 .

[112] Jiang, Y., Gupta, A., Zhang, Z., Wang, G., Dou, Y., Chen, Y., Fei-Fei, L., Anandkumar, A., Zhu, Y., Fan, L., 2022. Vima: General robot manipulation with multimodal prompts. arXiv preprint arXiv:2210.03094 2, 6.

[113] Jiang, Y., Zhang, R., Wong, J., Wang, C., Ze, Y., Yin, H., Gokmen, C., Song, S., Wu, J., Fei-Fei, L., 2025c. Behavior robot suite: Streamlining real-world whole-body manipulation for everyday household activities. arXiv preprint arXiv:2503.05652 .

[114] Jones, J., Mees, O., Sferrazza, C., Stachowicz, K., Abbeel, P., Levine, S., 2025. Beyond sight: Finetuning generalist robot policies with heterogeneous sensors via language grounding. arXiv preprint arXiv:2501.04693 .

[115] Karamcheti, S., Zhai, A.J., Losey, D.P., Sadigh, D., 2021. Learning visually guided latent actions for assistive teleoperation, in: Learning for dynamics and control, PMLR. pp. 12301241.

[116] Karli, U.B., Kurumisawa, T., Fitzgerald, T., 2025. Ask before you act: Token-level uncertainty for intervention in vision-language-action models, in: Second Workshop on Out-of-Distribution Generalization in Robotics at RSS 2025.

[117] Katiyar, N., 2023. A Model-Driven Framework for Domain-Specific Adaptation of Time Series Forecasting Pipeline. McGill University (Canada).

[118] Kawaharazuka, K., Oh, J., Yamada, J., Posner, I., Zhu, Y., 2025. Vision-language-action models for robotics: A review towards real-world applications. IEEE Access .

[119] Kelly, C., Hu, L., Yang, B., Tian, Y., Yang, D., Yang, C., Huang, Z., Li, Z., Hu, J., Zou, Y., 2024. Visiongpt: Vision-language understanding agent using generalized multimodal framework. arXiv preprint arXiv:2403.09027 .

[120] Khan, M.H., Asfaw, S., Iarchuk, D., Cabrera, M.A., Moreno, L., Tokmurziyev, I., Tsetserukou, D., 2025. Shake-vla: Vision-language-action model-based system for bimanual robotic manipulations and liquid mixing. arXiv preprint arXiv:2501.06919 .

[121] Kim, M.J., Finn, C., Liang, P., 2025. Fine-tuning visionlanguage-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645 .

[122] Kim, M.J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E., Lam, G., Sanketi, P., et al., 2024. Openvla: An opensource vision-language-action model. arXiv preprint arXiv:2406.09246 .

[123] Koo, J., Cho, T., Kang, H., Pyo, E., Oh, T.G., Kim, T., Choi, A.J., 2025. Retovla: Reusing register tokens for spatial reasoning in vision-language-action models. arXiv preprint arXiv:2509.21243 .

[124] Lee, N., Bang, Y., Lovenia, H., Cahyawijaya, S., Dai, W., Fung, P., 2023. Survey of social bias in vision-language models. arXiv preprint arXiv:2309.14381 .

[125] Li, C., Wen, J., Peng, Y., Peng, Y., Feng, F., Zhu, Y., 2025a. Pointvla: Injecting the 3d world into vision-language-action models. arXiv preprint arXiv:2503.07511 .

[126] Li, D., Jin, Y., Sun, Y., Yu, H., Shi, J., Hao, X., Hao, P., Liu, H., Sun, F., Zhang, J., et al., 2024a. What foundation models can bring for robot learning in manipulation: A survey. arXiv preprint arXiv:2404.18201 .

[127] Li, J., Skinner, G., Yang, G., Quaranto, B.R., Schwaitzberg, S.D., Kim, P.C., Xiong, J., 2024b. Llava-surg: towards multimodal surgical assistant via structured surgical video learning. arXiv preprint arXiv:2408.07981 [128] Li, J., Wei, P., Han, W., Fan, L., 2023. Intentqa: Contextaware video intent reasoning, in: Proceedings of the IEEE/CVF international conference on computer vision, pp. 1196311974.

[129] Li, J., Zhu, Y., Tang, Z., Wen, J., Zhu, M., Liu, X., Li, C., Cheng, R., Peng, Y., Feng, F., 2024c. Improving visionlanguage-action models via chain-of-affordance. arXiv preprint arXiv:2412.20451 .

[130] Li, M., Wang, Z., He, K., Ma, X., Liang, Y., 2025b. Jarvis-vla: Post-training large-scale vision language models to play visual games with keyboards and mouse. arXiv preprint arXiv:2503.16365 .

[131] Li, Q., Liang, Y., Wang, Z., Luo, L., Chen, X., Liao, M., Wei, F., Deng, Y., Xu, S., Zhang, Y., et al., 2024d. Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation. arXiv preprint arXiv:2411.19650 .

[132] Li, S., Wang, J., Dai, R., Ma, W., Ng, W.Y., Hu, Y., Li, Z., 2024e. Robonurse-vla: Robotic scrub nurse system based on vision-language-action model. arXiv preprint arXiv:2409.19590 .

[133] Li, S., Wu, H., Shao, J., Ma, Y., Gan, Y., Luo, Y., Wang, Y., Nie, D., Wang, L., Wu, W., et al., 2025c. Seeing the forest and the trees: Query-aware tokenizer for long-video multimodal language models. arXiv preprint arXiv:2511.11910 .

[134] Li, Y., Deng, Y., Zhang, J., Jang, J., Memmel, M., Yu, R., Garrett, C.R., Ramos, F., Fox, D., Li, A., et al., 2025d. Hamster: Hierarchical action models for open-world robot manipulation. arXiv preprint arXiv:2502.05485 .

[135] Li, Y., Gong, Z., Li, H., Huang, X., Kang, H., Bai, G., Ma, X., 2025e. Robotic visual instruction. arXiv preprint arXiv:2505.00693 .

[136] Li, Y., Lai, Z., Bao, W., Tan, Z., Dao, A., Sui, K., Shen, J., Liu, D., Liu, H., Kong, Y., 2025f. Visual large language models for generalized and specialized applications. arXiv preprint arXiv:2501.02765 .

[137] Li, Z., Wu, X., Du, H., Nghiem, H., Shi, G., $2 0 2 5 \mathrm { g }$ . Benchmark evaluations, applications, and challenges of large vision language models: A survey. arXiv preprint arXiv:2501.02189 1.

[138] Liang, Z., Li, Y., Yang, T., Wu, C., Mao, S., Nian, T., Pei, L., Zhou, S., Yang, X., Pang, J., et al., 2025. Discrete diffusion vla: Bringing discrete diffusion to action decoding in vision-language-action policies. arXiv preprint arXiv:2508.20072 .

[139] Lin, K.Q., Li, L., Gao, D., Yang, Z., Wu, S., Bai, Z., Lei, W., Wang, L., Shou, M.Z., 2024. Showui: One vision-language-action model for gui visual agent. arXiv preprint arXiv:2411.17465 .

[140] Lin, Y., Zhou, H., Chen, M., Min, H., 2019. Automatic sorting system for industrial robot with 3d visual perception and natural language interaction. Measurement and Control 52, 100115.

[141] Liu, H., Yao, R., Liu, W., Huang, Z., Shen, S., Ma, J. 2025a. Codrivevlm: Vlm-enhanced urban cooperative dispatching and motion planning for future autonomous mobility on demand systems. URL: https://arxiv. org/abs/2501.06132,arXiv:2501.06132.

[142] Liu, J., Chen, H., An, P., Liu, Z., Zhang, R., Gu, C., Li, X., Guo, Z., Chen, S., Liu, M., et al., 2025b. Hybridvla: Collaborative diffusion and autoregression in a unified vision-language-action model. arXiv preprint arXiv:2503.10631 .

[143] Liu, J., Liu, M., Wang, Z., An, P., Li, X., Zhou, K., Yang, S., Zhang, R., Guo, Y., Zhang, S., 2024a. Robomamba:

Efficient vision-language-action model for robotic reasoning and manipulation. Advances in Neural Information Processing Systems 37, 4008540110.

[144] Liu, S., Wu, L., Li, B., Tan, H., Chen, H., Wang, Z., Xu, K., Su, H., Zhu, J., 2024b. Rdt-1b: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864 .

[145] Liu, S., Yang, S., Fang, D., Jia, S., Tang, Y., Su, L., Peng, R., Yan, Y., Zou, X., Hu, X., 2026. Vision-language introspection: Mitigating overconfident hallucinations in mllms via interpretable bi-causal steering. arXiv preprint arXiv:2601.05159 .

[146] Liu, Y., Cao, X., Chen, T., Jiang, Y., You, J., Wu, M., Wang, X., Feng, M., Jin, Y., Chen, J., 2025c. From screens to scenes: A survey of embodied ai in healthcare. arXiv preprint arXiv:2501.07468 .

[147] Liu, Y., Cao, X., Chen, T., Jiang, Y., You, J., Wu, M., Wang, X., Feng, M., Jin, Y., Chen, J., 2025d. A survey of embodied ai in healthcare: Techniques, applications, and opportunities. arXiv preprint arXiv:2501.07468 .

[148] Liu, Z., Liang, H., Huang, X., Xiong, W., Yu, Q., Sun, L., Chen, C., He, C., Cui, B., Zhang, W., 2024c. Synthvlm: High-efficiency and high-quality synthetic data for vision language models. arXiv preprint arXiv:2407.20756 [149] Lu, H., Li, H., Shahani, P.S., Herbers, S., Scheutz, M., 2025. Probing a vision-language-action model for symbolic states and integration into a cognitive architecture. arXiv preprint arXiv:2502.04558 .

[150] Lu, J., Clark, C., Lee, S., Zhang, Z., Khosla, S., Marten, R., Hoiem, D., Kembhavi, A., 2024. Unifiedio 2: Scaling autoregressive multimodal models with vision language audio and action, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2643926455.

[151] Lu, Y., Liao, Z., 2023. Towards happy housework: Scenario-based experience design for a household cleaning robotic system. EAI Endorsed Transactions on Scalable Information Systems 10.

[152] Luo, H., Feng, Y., Zhang, W., Zheng, S., Wang, Y., Yuan, H., Liu, J., Xu, C., Jin, Q., Lu, Z., 2025. Being-hO: vision-language-action pretraining from large-scale human videos. arXiv preprint arXiv:2507.15597 .

[153] Luo, J., Xu, C., Wu, J., Levine, S., 2024. Precise and dexterous robotic manipulation via humanin-the-loop reinforcement learning. arXiv preprint arXiv:2410.21845 .

[154] Lyre, H., 2024. "understanding ai": Semantic grounding in large language models. URL: https: //arxiv. org/ abs/2402.10992, arXiv:2402.10992.

[155] Lyu, J., Li, Z., Shi, X., Xu, C., Wang, Y., Wang, H., 2025. Dywa: Dynamics-adaptive world action model for generalizable non-prehensile manipulation. arXiv preprint arXiv:2503.16806 .

[156] Ma, Y., Song, Z., Zhuang, Y., Hao, J., King, I., 2024. A survey on vision-language-action models for embodied ai. arXiv preprint arXiv:2405.14093 .

[157] Misra, I., Girdhar, R., Joulin, A., 2021. An end-to-end transformer model for 3d object detection, in: ICCV.

[158] Mohammed, M.Q., Chung, K.L., Chyi, C.S., 2020. Review of deep reinforcement learning-based object grasping: Techniques, open challenges, and recommendations. Ieee Access 8, 178450178481.

[159] Moroncelli, A., Soni, V., Shahid, A.A., Maccarini, M., Forgione, M., Piga, D., Spahiu, B., Roveda, L., 2024. Integrating reinforcement learning with foundation models for autonomous robotics: Methods and perspectives. arXiv preprint arXiv:2410.16411 .

[160] Mumuni, A., Mumuni, F., 2025. Large language models for artificial general intelligence (agi): A survey of foundational principles and approaches. arXiv preprint arXiv:2501.03151 .

[161] Ni, F., Hao, J., Wu, S., Kou, L., Yuan, Y., Dong, Z., Liu, J., Li, M., Zhuang, Y., Zheng, Y., 2024. Peria: Perceive, reason, imagine, act via holistic language and vision planning for manipulation. Advances in Neural Information Processing Systems 37, 1754117571.

[162] Nie, Y., Li, L., Gan, Z., Wang, S., Zhu, C., Zeng, M., Liu, Z., Bansal, M., Wang, L., 2021. Mlp architectures for vision-and-language modeling: An empirical study. arXiv preprint arXiv:2112.04453 .

[163] Noorani, E., Serlin, Z., Price, B., Velasquez, A., 2025. From abstraction to reality: Darpa's vision for robust sim-to-real autonomy. arXiv preprint arXiv:2503.11007 [164] Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al., 2023. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 .

[165] Pang, J., Zheng, P., Fan, J., Liu, T., 2025. Towards cognition-augmented human-centric assembly: A visual computation perspective. Robotics and ComputerIntegrated Manufacturing 91, 102852.

[166] Pantalone, D., Faini, G.S., Cialdai, F., Sereni, E., Bacci, S., Bani, D., Bernini, M., Pratesi, C., Stefano, P., Orzalesi, L., et al., 2021. Robot-assisted surgery in space: pros and cons. a review from the surgeon's point of view. npj Microgravity 7, 56.

[167] Park, S., Kim, H., Kim, S., Jeon, W., Yang, J., Jeon, B., Oh, Y., Choi, J., 2025. Saliency-aware quantized imitation learning for efficient robotic control, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1314013150.

[168] Park, S.M., Kim, Y.G., 2023. Visual language integration: A survey and open challenges. Computer Science Review 48, 100548.

[169] Pasas-Farmer, S., Jain, R., 2025. From discovery to delivery: Governance of ai in the pharmaceutical industry. Green Analytical Chemistry 13, 100268.

[170] Patel, D., Eghbalzadeh, H., Kamra, N., Iuzzolino, M.L., Jain, U., Desai, R., 2023. Pretrained language models as visual planners for human assistance, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1530215314.

[171] Pertsch, K., Stachowicz, K., Ichter, B., Driess, D., Nair, S., Vuong, Q., Mees, O., Finn, C., Levine, S., 2025. Fast: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747 .

[172] Plaat, A., van Duijn, M., van Stein, N., Preuss, M., van der Putten, P., Batenburg, K.J., 2025. Agentic large language models, a survey. arXiv preprint arXiv:2503.23037 .

[173] Polubarov, A., Lyubaykin, N., Derevyagin, A., Zisman, I., Tarasov, D., Nikulin, A., Kurenkov, V., 2025. Vintix: Action model via in-context reinforcement learning. arXiv preprint arXiv:2501.19400 .

[174] Qi, C.R., Litany, O., He, K., Guibas, L.J., 2019. Deep hough voting for 3d object detection in point clouds, in: ICCV.

[175] Qu, D., Song, H., Chen, Q., Yao, Y., Ye, X., Ding, Y., Wang, Z., Gu, J., Zhao, B., Wang, D., et al., 2025. Spatialvla: Exploring spatial representations for visual-language-action model. arXiv preprint arXiv:2501.15830 .

[176] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al., 2021. Learning transferable visual models from natural language supervision, in: ICML, PMLR.

[177] Radford, A., Narasimhan, K., Salimans, T., Sutskever, I., et al., 2018. Improving language understanding by generative pre-training .

[178] Rawal, P.K., 2025. An Intelligent Versatile Pipeline for 6D Localization of Industrial Components in a Production Environment. Ph.D. thesis. Fraunhofer Verlag.

[179] Ray, P.P., 2023. Chatgpt: A comprehensive review on background, applications, key challenges, bias, ethics, limitations and future scope. Internet of Things and Cyber-Physical Systems 3, 121154.

[180] Raza, S., Qureshi, R., Zahid, A., Fioresi, J., Sadak, F., Saeed, M., Sapkota, R., Jain, A., Zafar, A., Hassan, M.U., et al., 2025. Who is responsible? the data, models, users or regulations? responsible generative ai for a sustainable future. arXiv preprint arXiv:2502.08650 .

[181] Reed, S., Zolna, K., Parisotto, E., Colmenarejo, S.G., Novikov, A., Barth-Maron, G., Gimenez, M., Sulsky, Y., Kay, J., Springenberg, J.T., et al., 2022. A generalist agent. arXiv preprint arXiv:2205.06175 .

[182] Rodriguez-Guerra, D., Sorrosal, G., Cabanes, I., Calleja, C., 2021. Human-robot interaction review: Challenges and solutions for modern industrial environments. Ieee Access 9, 108557108578.

[183] Rodriguez-Juan, J., Ortiz-Perez, D., Garcia-Rodriguez, J., Tomás, D., Nalepa, G.J., 2025. Integrating advanced vision-language models for context recognition in risks assessment. Neurocomputing 618, 129131.

[184] Roychoudhury, A., Khorshidi, S., Agrawal, S., Bennewitz, M., 2023. Perception for humanoid robots. Current Robotics Reports 4, 127140.

[185] Sahili, Z.A., Patras, I., Purver, M., 2025. Scaling for fairness? analyzing model size, data composition, and multilinguality in vision-language bias. arXiv preprint arXiv:2501.13223 .

[186] Sameni, S., Kafle, K., Tan, H., Jenni, S., 2024. Building vision-language models on solid foundations with masked distillation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1421614226.

[187] Samson, M., Muraccioli, B., Kanehiro, F., 2025. Scalable, training-free visual language robotics: a modular multi-model framework for consumer-grade gpus, in: 2025 IEEE/SICE International Symposium on System Integration (SII), IEEE. pp. 193198.

[188] Sanyal, S., Roy, K., 2025. Asma: An adaptive safety margin algorithm for vision-language drone navigation via scene-aware control barrier functions. IEEE Robotics and Automation Letters .

[189] Sapkota, R., Karkee, M., 2025. Object detection with multimodal large vision-language models: An in-depth review. Available at SSRN 5233953 .

[190] Sapkota, R., Roumeliotis, K.I., Cheppally, R.H., Calero, M.F., Karkee, M., 2025. A review of 3d object detection with vision-language models. arXiv preprint arXiv:2504.18738 .

[191] Sautenkov, O., Yaqoot, Y., Lykov, A., Mustafa, M.A., Tadevosyan, G., Akhmetkazy, A., Cabrera, M.A., Martynov, M., Karaf, S., Tsetserukou, D., 2025. Uav-vla: Vision-language-action system for large scale aerial mission generation. arXiv preprint arXiv:2501.05014 .

[192] Schakkal, A., Zandonati, B., Yang, Z., Azizan, N., 2025. Hierarchical vision-language planning for multi-step humanoid manipulation. arXiv preprint arXiv:2506.22827 [193] Schmidgall, S., Cho, J., Zakka, C., Hiesinger, W., 2024. Gp-vls: A general-purpose vision language model for surgery. arXiv preprint arXiv:2407.19305 .

[194] Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., et al., 2022. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems 35, 2527825294.

[195] Serpiva, V., Lykov, A., Myshlyaev, A., Khan, M.H., Abdulkarim, A.A., Sautenkov, O., Tsetserukou, D., 2025. Racevla: Vla-based racing drone navigation with human-like behaviour. arXiv preprint arXiv:2503.02572 [196] Shadab Siddiqui, M., Rabbi, M., Islam, M.J., Ahmed, R.U., 2025. Comparison of different controller architectures for autonomous driving and recommendations for robust and safe implementations. Journal of Advanced Transportation 2025, 9995539.

[197] Shao, R., Li, W., Zhang, L., Zhang, R., Liu, Z., Chen, R., Nie, L., 2025. Large vlm-based vision-languageaction models for robotic manipulation: A survey. arXiv preprint arXiv:2508.13073 .

[198] Sharshar, A., Khan, L.U., Ullah, W., Guizani, M., 2025. Vision-language models for edge networks: A comprehensive survey. arXiv preprint arXiv:2502.07855 .

[199] Shi, L.X., Ichter, B., Equi, M., Ke, L., Pertsch, K. Vuong, Q., Tanner, J., Walling, A., Wang, H., Fusai, N., et al., 2025. Hi robot: Open-ended instruction following with hierarchical vision-language-action models. arXiv preprint arXiv:2502.19417 .

[200] Shin, H.C., Roth, H.R., Gao, M., Lu, L., Xu, Z., Nogues, I., Yao, J., Mollura, D., Summers, R.M., 2016. Deep convolutional neural networks for computer-aided detection: Cnn architectures, dataset characteristics and transfer learning. IEEE transactions on medical imaging 35, 12851298.

[201] Shivadekar, S., 2025. Artificial Intelligence for Cognitive Systems: Deep Learning, Neuro-symbolic Integration, and Human-Centric Intelligence. Deep Science Publishing.

[202] Shridhar, M., Manuelli, L., Fox, D., 2022. Cliport: What and where pathways for robotic manipulation, in: Conference on robot learning, PMLR. pp. 894906.

[203] Shukor, M., Aubakirova, D., Capuano, F., Kooijmans, P., Palma, S., Zouitine, A., Aractingi, M., Pascal, C., Russi, M., Marafioti, A., et al., 2025. Smolvla: A vision-language-action model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844 .

[204] Si, W., Wang, N., Yang, C., 2021. A review on manipulation skill acquisition through teleoperation-based learning from demonstration. Cognitive Computation and Systems 3, 116.

[205] Simonyan, K., Zisserman, A., 2014. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556 .

[206] Singh, G., 2025. Neural Object-Centric Scene Representation and Generation. Ph.D. thesis. Rutgers The State University of New Jersey, School of Graduate Studies.

[207] Singh, S., Singh, J., Shah, B., Sehra, S.S., Ali, F., 2022. Augmented reality and gps-based resource efficient navigation system for outdoor environments: Integrating device camera, sensors, and storage. Sustainability 14, 12720.

[208] Song, M., Deng, X., Zhou, Z., Wei, J., Guan, W., Nie, L., 2025a. A survey on diffusion policy for robotic manipulation: Taxonomy, analysis, and future directions. Authorea Preprints .

[209] Song, W., Chen, J., Ding, P., Zhao, H., Zhao, W., Zhong, Z., Ge, Z., Ma, J., Li, H., 2025b. Accelerating visionlanguage-action model integrated with action chunking via parallel decoding. arXiv preprint arXiv:2503.02310 .

[210] Sun, H., Wang, H., Ma, C., Zhang, S., Ye, J., Chen, X., Lan, X., 2025a. Prism: Projection-based reward integration for scene-aware real-to-sim-to-real transfer with few demonstrations. arXiv preprint arXiv:2504.20520 .

[211] Sun, J., Mao, P., Kong, L., Wang, J., 2025b. A review of embodied grasping. Sensors (Basel, Switzerland) 25, 852.

[212] Sun, L., Xie, B., Liu, Y., Shi, H., Wang, T., Cao, J., 2025c. Geovla: Empowering 3d representations in vision-language-action models. arXiv preprint arXiv:2508.09071 .

[213] Sutskever, I., Martens, J., Hinton, G.E., 2011. Generating text with recurrent neural networks, in: Proceedings of the 28th international conference on machine learning (ICML-11), pp. 10171024.

[214] Szot, A., Mazoure, B., Agrawal, H., Hjelm, R.D., Kira, Z., Toshev, A., 2024. Grounding multimodal large language models in actions. Advances in Neural Information Processing Systems 37, 2019820224.

[215] Taherin, A., Lin, J., Akbari, A., Akbari, A., Zhao, P., Chen, W., Kaeli, D., Wang, Y., 2025. Cross-platform scaling of vision-language-action models from edge to cloud gpus. arXiv preprint arXiv:2509.11480 .

[216] Tan, X., Yang, Y., Ye, P., Zheng, J., Bai, B., Wang, X., Hao, J., Chen, T., 2025. Think twice, act once: Tokenaware compression and action reuse for efficient inference in vision-language-action models. arXiv preprint arXiv:2505.21200 .

[217] Team, G.R., Abeyruwan, S., Ainslie, J., Alayrac, J.B., Arenas, M.G., Armstrong, T., Balakrishna, A., Baruch, R., Bauza, M., Blokzijl, M., et al., 2025. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020 .

[218] Team, O.M., Ghosh, D., Walke, H., Pertsch, K., Black, K., Mees, O., Dasari, S., Hejna, J., Kreiman, T., Xu, C., et al., 2024. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213 .

[219] Tellex, S., Gopalan, N., Kress-Gazit, H., Matuszek, C., 2020. Robots that use language. Annual Review of Control, Robotics, and Autonomous Systems 3, 2555.

[220] Tian, H., Wang, T., Liu, Y., Qiao, X., Li, Y., 2020. Computer vision technology in agricultural automation—a review. Information processing in agriculture 7, 119.

[221] Tian, K., Jiang, Y., Yuan, Z., Peng, B., Wang, L., 2024. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems 37, 8483984865.

[222] Torres, N., Ulloa, C., Araya, I., Ayala, M., Jara, S., 2024. A comprehensive analysis of gender, racial, and promptinduced biases in large language models. International Journal of Data Science and Analytics , 138.

[223] Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al., 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 .

[224] Trivedi, C., Bhattacharya, P., Prasad, V.K., Patel, V., Singh, A., Tanwar, S., Sharma, R., Aluvala, S., Pau, G., Sharma, G., 2024. Explainable ai for industry 5.0: vision, architecture, and potential directions. IEEE Open Journal of Industry Applications .

[225] Verbaan, L., 2024. Perception and control with large language models in robotic manipulation. TU Delft Library [226] Vinod, K., Ramesh, P.J., Chakravarthi, B., et al., 2025. Sebvs: Synthetic event-based visual servoing for robot navigation and manipulation. arXiv preprint arXiv:2508.17643 .

[227] Vuong, Q., Levine, S., Walke, H.R., Pertsch, K., Singh, A., Doshi, R., Xu, C., Luo, J., Tan, L., Shah, D., et al., 2023. Open x-embodiment: Robotic learning datasets and rt-x models, in: Towards Generalist Robots: Learning Paradigms for Scalable Skill Acquisition $@$ CoRL2023.

[228] Waite, J.R., Hasan, M.Z., Liu, Q., Jiang, Z., Hegde, C., Sarkar, S., 2025. Rls3: R1-based synthetic sample selection to enhance spatial reasoning in vision-language models for indoor autonomous perception, in: Proceedings of the ACM/IEEE 16th International Conference on Cyber-Physical Systems (with CPS-IoT Week 2025), Association for Computing Machinery, New York, NY, USA. doi:10.1145/3716550.3722033.

[229] Wang, G., Bai, L., Nah, W.J., Wang, J., Zhang, Z., Chen, Z., Wu, J., Islam, M., Liu, H., Ren, H., 2024a. Surgicallvlm: Learning to adapt large vision-language model for grounded visual question answering in robotic surgery. arXiv preprint arXiv:2405.10948 .

[230] Wang, H., Xing, Z., Wu, W., Yang, Y., Tang, Q., Zhang, M., Xu, Y., Zhu, L., 2024b. Non-invasive to invasive: Enhancing ffa synthesis from cfp with a benchmark dataset and a novel network, in: Proceedings of the 1st International Workshop on Multimedia Computing for Health and Medicine, pp. 715.

[231] Wang, J., Guo, D., Liu, H., 2025a. Where to learn: Embodied perception learning planned by vision-language models. IEEE Transactions on Cognitive and Developmental Systems .

[232] Wang, S., 2025. Roboflamingo-plus: Fusion of depth and rgb perception with vision-language models for enhanced robotic manipulation. arXiv preprint arXiv:2503.19510 .

[233] Wang, T., Han, C., Liang, J.C., Yang, W., Liu, D., Zhang, L.X., Wang, Q., Luo, J., Tang, R., 2024c. Exploring the adversarial vulnerabilities of vision-language-action models in robotics. arXiv preprint arXiv:2411.13587 .

[234] Wang, Y., Liu, Q., Jiang, Z., Wang, T., Jiao, J., Chu, H., Gao, B., Chen, H., 2025b. Rad: Retrieval-augmented decision-making of meta-actions with vision-language models in autonomous driving, in: Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 38383848.

[235] Wang, Y., Niu, X., Ba, J., Su, Z., Du, L., 2025c. Navigating embodied intelligence: Enabling technologies, security and privacy, and emerging trends. IEEE Internet of Things Journal .

[236] Wang, Y., Wu, S., Zhang, Y., Yan, S., Liu, Z., Luo, J., Fei, H., 2025d. Multimodal chain-of-thought reasoning: A comprehensive survey. arXiv preprint arXiv:2503.12605 .

[237] Wang, Z., Zhou, Z., Song, J., Huang, Y., Shu, Z., Ma, L., 2024d. Towards testing and evaluating vision-languageaction models for robotic manipulation: An empirical study. arXiv preprint arXiv:2409.12894 .

[238] Wei, C., Guo, C., Zhang, J., Shan, H., Xu, Y., Zhang, Z. Liu, Y., Wang, Q., Zhou, C., Li, H., et al., 2025. Focus: A streaming concentration architecture for efficient visionlanguage models. arXiv preprint arXiv:2512.14661 .

[239] Wei, J., Yuan, S., Li, P., Hu, Q., Gan, Z., Ding, W., 2024. Occllama: An occupancy-language-action generative world model for autonomous driving. arXiv preprint arXiv:2409.03272 .

[240] Wen, J., Zhu, M., Zhu, Y., Tang, Z., Li, J., Zhou, Z., Li, C., Liu, X., Peng, Y., Shen, C., et al., 2024. Diffusion-vla: Scaling robot foundation models via unified diffusion and autoregression. arXiv preprint arXiv:2412.03293 .

[241] Wen, J., Zhu, Y., Li, J., Tang, Z., Shen, C., Feng, F., 2025a. Dexvla: Vision-language model with plug-in diffusion expert for general robot control. arXiv preprint arXiv:2502.05855 .

[242] Wen, J., Zhu, Y., Li, J., Zhu, M., Tang, Z., Wu, K., Xu, Z., Liu, N., Cheng, R., Shen, C., et al., 2025b. Tinyvla: Towards fast, data-efficient vision-language-action models for robotic manipulation. IEEE Robotics and Automation Letters .

[243] Woo, S., Debnath, S., Hu, R., Chen, X., Liu, Z., Kweon, I.S., Xie, S., 2023. Convnext v2: Co-designing and scaling convnets with masked autoencoders, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1613316142.

[244] Wu, J., Zhong, M., Xing, S., Lai, Z., Liu, Z., Chen, Z., Wang, W., Zhu, X., Lu, L., Lu, T., et al., 2024a. Visionllm v2: An end-to-end generalist multimodal large language model for hundreds of vision-language tasks. Advances in Neural Information Processing Systems 37, 6992569975.

[245] Wu, W., Feng, X., Gao, Z., Kan, Y., 2024b. Smart: scalable multi-agent real-time motion generation via nexttoken prediction. Advances in Neural Information Processing Systems 37, 114048114071.

[246] Wu, Z., Zhou, Y., Xu, X., Wang, Z., Yan, H., 2025. Momanipvla: Transferring vision-language-action models for general mobile manipulation. arXiv preprint arXiv:2503.13446 .

[247] Xiang, T.Y., Jin, A.Q., Zhou, X.H., Gui, M.J., Xie, X.L., Liu, S.Q., Wang, S.Y., Duang, S.B., Wang, S.C., Lei, Z., et al., 2025. Vla model-expert collaboration for bi-directional manipulation learning. arXiv preprint arXiv:2503.04163 .

[248] Xiong, J., Liu, G., Huang, L., Wu, C., Wu, T., Mu, Y., Yao, Y., Shen, H., Wan, Z., Huang, J., et al., 2024. Autoregressive models in vision: A survey. arXiv preprint arXiv:2411.05902 .

[249] Xu, D., Chen, Y., Wang, J., Huang, Y., Wang, H., Jin, Z., Wang, H., Yue, W., He, J., Li, H., et al., 2024a. Mlevlm: Improve multi-level progressive capabilities based on multimodal large language model for medical visual question answering, in: Findings of the Association for Computational Linguistics ACL 2024, pp. 4977 4997.

[250] Xu, F., Zhai, G., Kong, X., Fu, T., Gordon, D.F., An, X., Busam, B., 2025a. Stare-vla: Progressive stageaware reinforcement for fine-tuning vision-languageaction models. arXiv preprint arXiv:2512.05107 .

[251] Xu, J., Sun, Q., Han, Q.L., Tang, Y., 2025b. When embodied ai meets industry 5.0: human-centered smart manufacturing. IEEE/CAA Journal of Automatica Sinica 12, 485501.

[252] Xu, S., Wang, Y., Xia, C., Zhu, D., Huang, T., Xu, C., 2025c. Vla-cache: Towards efficient vision-languageaction model via adaptive token caching in robotic manipulation. arXiv preprint arXiv:2502.02175 .

[253] Xu, Y., Liu, G., Kompella, R.R., Hu, S., Huang, T., I1- han, F., Tekin, S.F., Yahn, Z., Liu, L., 2025d. Languagevision planner and executor for text-to-visual reasoning. arXiv preprint arXiv:2506.07778 .

[254] Xu, Z., Wu, K., Wen, J., Li, J., Liu, N., Che, Z., Tang, J., 2024b. A survey on robotics with foundation models: toward embodied ai. arXiv preprint arXiv:2402.02385 .

[255] Xue, H., Ren, J., Chen, W., Zhang, G., Fang, Y., Gu, G., Xu, H., Lu, C., 2025. Reactive diffusion policy: Slowfast visual-tactile policy learning for contact-rich manipulation. arXiv preprint arXiv:2503.02881 .

[256] Yang, G., Zhang, T., Hao, H., Wang, W., Liu, Y., Wang, D., Chen, G., Cai, Z., Chen, J., Su, W., et al., 2025a. Vlaser: Vision-language-action model with synergistic embodied reasoning. arXiv preprint arXiv:2510.11027 .

[257] Yang, R., Chen, G., Wen, C., Gao, Y., 2025b. Fp3: A 3d foundation policy for robotic manipulation. arXiv preprint arXiv:2503.08950 .

[260] Yang, Y., Huang, W., Wei, Y., Peng, H., Jiang, X., Jiang, H., Wei, F., Wang, Y., Hu, H., Qiu, L., et al., 2023a. Attentive mask clip, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2771 2781.

[261] Yang, Y., Sun, J., Kou, S., Wang, Y., Deng, Z., 2025e. Lohovla: A unified vision-language-action model for long-horizon embodied tasks. arXiv preprint arXiv:2506.00411 .

[262] Yang, Y., Wang, Y., Wen, Z., Zhongwei, L., Zou, C., Zhang, Z., Wen, C., Zhang, L., 2025f. Efficientvla: Training-free acceleration and compression for vision-language-action models. arXiv preprint arXiv:2506.10100 .

[263] Yang, Y., Zhou, J., Ding, X., Huai, T., Liu, S., Chen, Q., Xie, Y., He, L., 2025g. Recent advances of foundation language models-based continual learning: A survey. ACM Computing Surveys 57, 138.

[264] Yang, Z., Chen, Y., Wang, J., Manivasagam, S., Ma, W.C., Yang, A.J., Urtasun, R., 2023b. Unisim: A neural closed-loop sensor simulator, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13891399.

[258] Yang, R., Yu, Q., Wu, Y., Yan, R., Li, B., Cheng, A.C., Zou, X., Fang, Y., Cheng, X., Qiu, R.Z., et al., 2025c. Egovla: Learning vision-language-action models from egocentric human videos. arXiv preprint arXiv:2507.12440 .

[265] Yang, Z., Garrett, C., Fox, D., Lozano-Pérez, T., Kaelbling, L.P., 2025h. Guiding long-horizon task and motion planning with vision language models, in: 2025 IEEE International Conference on Robotics and Automation (ICRA), IEEE. pp. 1684716853.

[259] Yang, S., Li, Y.L., Wang, S., 2025d. Upl-net: Uncertainty-aware prompt learning network for semisupervised action recognition. Neurocomputing 619, 129126.

[266] Ye, S., Jang, J., Jeon, B., Joo, S., Yang, J., Peng, B., Mandlekar, A., Tan, R., Chao, Y.W., Lin, B.Y., et al., 2024. Latent action pretraining from videos. arXiv preprint arXiv:2410.11758 .

[267] Ye, Y., Ma, J., Cen, J., Lu, Z., 2025. Token expand-merge: Training-free token compression for vision-language-action models. arXiv preprint arXiv:2512.09927 .

[268] Yu, Z., Wang, B., Zeng, P., Zhang, H., Zhang, J., Gao, L., Song, J., Sebe, N., Shen, H.T., 2025. A survey on efficient vision-language-action models. arXiv preprint arXiv:2510.24795 .

[269] Yue, Y., Wang, Y., Kang, B., Han, Y., Wang, S., Song, S., Feng, J., Huang, G., 2024. Deer-vla: Dynamic inference of multimodal large language models for efficient robot execution. Advances in Neural Information Processing Systems 37, 5661956643.

[270] Zawalski, M., Chen, W., Pertsch, K., Mees, O., Finn, C., Levine, S., 2024. Robotic control via embodied chainof-thought reasoning. arXiv preprint arXiv:2407.08693

[271] Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L., 2023. Sigmoid loss for language image pre-training, in: Proceedings of the IEEE/CVF international conference on computer vision, pp. 1197511986.   
[] Zhan, Z.Cen, Y. Zou, J., Lv, Q. Liu, H. K., Lin, L., Wang, G., 2026. Stable language guidance for vision-language-action models. arXiv preprint arXiv:2601.04052 .   
[273] Zhang, B., Li, J., Shen, J., Cai, Y., Zhang, Y., Chen, Y., Dai, J., Ji, J., Yang, Y., 2025a.Va-area:An opensource framework for benchmarking vision-languageaction models. arXiv preprint arXiv:2512.22539   
[274] Zhang, B., Zhang, Y., Ji, J., Lei, Y., Dai, J., Chen, Y., Yang, Y., 2025b. Safevla: Towards safety alignment of vision-language-action model via safe reinforcement learning. arXiv preprint arXiv:2503.03480 .   
[275] Zhang, D., Sun, J., Hu, C., Wu, X., Yuan, Z., Zhou, R., Shen, F., Zhou, Q., 2025c. Pure vision language action (vla) models: A comprehensive survey. arXiv preprint arXiv:2509.19012 .   
[276] Zhang, H., Ding, P., Lyu, S., Peng, Y., Wang, D., 2025d. Gevrm: Goal-expressive video generation model for robust visual manipulation. arXiv preprint arXiv:2502.09268 .   
[277] Zhang, H., Yu, H., Zhao, L., Choi, A., Bai, Q., Yang, B., Xu, W., 2025e. Slim: Sim-to-real legged instructive manipulation via long-horizon visuomotor learning. arXiv preprint arXiv:2501.09905 .   
[78] ag, H. Z, N.Ka, P. , Z.g, J., Wang, W., 2024a. Vla-3d: A dataset for 3d semantic scene understanding and navigation. arXiv preprint arXiv:2411.03540 .   
[279] Zhang, J., Guo, Y., Hu, Y., Chen, X., Zhu, X., Chen, J., 2025f. Up-vla: A unified understanding and prediction model for embodied agent. arXiv preprint arXiv:2501.18867 .   
[280] Zhang, J., Wang, K., Wang, S., Li, M., Liu, H., Wei, S., Wang, Z., Zhang, Z., Wang, H., 2024b. Uninavid: A video-based vision-language-action model for unifying embodied navigation tasks. arXiv preprint arXiv:2412.06224 .   
[281] Zhang, K., Yin, Z.H., Ye, W., Gao, Y., 2024c. Learning manipulation skills through robot chain-ofthought with sparse failure guidance. arXiv preprint arXiv:2405.13573 .   
[282] Zhang, K., Yun, P., Cen, J., Cai, J., Zhu, D., Yuan, H., Zhao, C., Feng, T. Wang, M.Y., Chen, Q., et al., 25. Generativertificial intelligencn robotimanipulation: A survey. arXiv preprint arXiv:2503.03464 .   
[283] Zhang, R., Dong, M., Zhang, Y., Heng, L., Chi, X., Dai, G., Du, L., Wang, D., Du, Y., Zhang, S., 2025h. Mole-vla: Dynamic layer-skipping vision language action model via mixture-of-layers for efficient robot manipulation. arXiv preprint arXiv:2503.20384 .   
[284] Zhang, Z., Bao, C., Pan, X., Chang, C.M., Igarashi, T., Zhang, G., 2025i. Through the lens of privacy: Exploring privacy protection in vision-language model interactions on smart glasses, in: Proceedings of the Extended Abstracts of the CHI Conference on Human Factors in Computing Systems, pp. 18.   
[285] Zhao, H., Song, W., Wang, D., Tong, X., Ding, P.Cheng, X. Ge, Z., 2025a. More: Unlocin scalability in reinforcement learning for quadruped vision-language-action models. arXiv preprint arXiv:2503.08007 .   
[286] Zhao, Q., Lu, Y., Kim, M.J., Fu, Z., Zhang, Z., Wu, Y., Li, Z., Ma, Q., Han, S., Finn, C., et al., 2025b. Cot-vla: Visual chain-of-thought reasoning for vision-languageaction models. arXiv preprint arXiv:2503.22020 .   
[287] Zhao, T.Z., Kumar, V., Levine, S., Finn, C., 2023. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705 .   
[288] Zhen, H., Qiu, X., Chen, P., Yang, J., Yan, X., Du, Y., Hong, Y., Gan, C., 2024. 3d-vla: A 3d visionlanguage-action generative world model. arXiv preprint arXiv:2403.09631 .   
[289] Zheng, J., Li, J., Liu, D., Zheng, Y., Wang, Z., Ou, Z., Liu, Y., Liu, J., Zhang, Y.Q., Zhan, X., 2025. Universal actions for enhanced embodied foundation models. arXiv preprint arXiv:2501.10105 .   
[290] Zheng, J., Shi, C., Cai, X., Li, Q., Zhang, D., Li, C., Yu, D Ma, Q model based agents: A roadmap. IEEE Transactions on Pattern Analysis and Machine Intelligence .   
[1] Zhong, Y., Huang, X., Li, R., Zhang, C., Liang, Y., Yang, Y., Chen, Y., 2025. Dexgraspvla: A vision-languageaction framework towards general dexterous grasping. arXiv preprint arXiv:2502.20900 .   
[292] Zhou, D.W., Zhang, Y., Wang, Y., Ning, J., Ye, H.J., Zhan, D.C., Liu, Z., 2025a. Learning without forgetting for vision-language models. IEEE Transactions on Pattern Analysis and Machine Intelligence .   
[293] Zhou, X., Han, X., Yang, F., Ma, Y., Knoll, A.C., 2025b. Opendrivevla: Towards end-to-end autonomous driving with large vision language action model. arXiv preprint arXiv:2503.23463 .   
[294] Zhou, Z., Cai, T., Zhao, S.Z., Zhang, Y., Huang, Z., Zhou, B., Ma, J., 2025c. Autovla: A vision-languageaction model for end-to-end autonomous driving with

adaptive reasoning and reinforcement fine-tuning. arXiv preprint arXiv:2506.13757 .

[295] Zhou, Z., Zhu, Y., Zhu, M., Wen, J., Liu, N., Xu, Z., Meng, W., Cheng, R., Peng, Y., Shen, C., et al., 2025d. Chatvla: Unified multimodal understanding and robot control with vision-language-action model. arXiv preprint arXiv:2502.14420 .

[296] Zhu, D.H., Chang, Y.P., 2020. Robot with humanoid hands cooks food better? effect of robotic chef anthropomorphism on food quality prediction. International Journal of Contemporary Hospitality Management 32, 1367 1383.

[297] Zhu, M., Zhu, Y., Li, J., Zhou, Z., Wen, J., Liu, X., Shen, C., Peng, Y., Feng, F., 2025. Objectvla: End-to-end open-world object manipulation without demonstration. arXiv preprint arXiv:2502.19250 .

[298] Zhu, Y., Zhou, Y., Wang, C., Cao, Y., Han, J., Hou, L., Xu, H., 2024. Unit: Unifying image and text recognition in one vision encoder, in: NeurIPS.

[299] Zitkovich, B., Yu, T., Xu, S., Xu, P., Xiao, T., Xia, F., Wu, J., Wohlhart, P., Welker, S., Wahid, A., et al., 2023. Rt-2: Vision-language-action models transfer web knowledge to robotic control, in: Conference on Robot Learning, PMLR. pp. 21652183.