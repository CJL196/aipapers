# DreamVLA：一个具备综合世界知识的视觉-语言-行动模型

Wenyao Zhang124\* Hongsi Liu27\* Zekun Qi34\* Yunnan Wang12\* Xinqiang Yu4 Jiazhao Zhang45 Runpei Dong6 Jiawei He4 Fan Lu7 He Wang45 Zhizheng Zhang4 Li Yi3 Wenjun Zeng2 Xin Jin2‡ 1SJTU 2EIT 3THU 4Galbot 5PKU 6UIUC 7USTC A项目页面 Q代码 Hugging Face

# 摘要

最近在视觉-语言-行动（VLA）模型方面的进展显示出将图像生成与行动预测相结合以提高机器人操作的泛化和推理能力的前景。然而，现有的方法限于具有挑战性的基于图像的预测，这受到冗余信息的影响，并且缺乏全面和关键的世界知识，包括动态、空间和语义信息。为了解决这些局限性，我们提出了DreamVLA，这是一个新颖的VLA框架，它整合了全面的世界知识预测，以实现逆动态建模，从而建立一个用于操作任务的感知-预测-行动循环。具体而言，DreamVLA引入了动态区域导向的世界知识预测，结合空间和语义线索，为行动规划提供紧凑而全面的表示。这一设计与人类如何与世界互动的方式相一致，即在行动之前先形成抽象的多模态推理链。为了在训练过程中减轻动态、空间和语义信息之间的干扰，我们采用了一种块状结构注意机制，该机制掩盖了它们之间的相互注意，防止信息泄漏，并保持每个表示的清晰和分离。此外，为了建模未来行动的条件分布，我们采用了一种基于扩散的变换器，将行动表示与共享潜在特征分离。对真实世界和模拟环境的广泛实验表明，DreamVLA在真实机器人任务上实现了$76.7\%$的成功率，在CALVIN ABC-D基准测试中平均长度为4.44。

# 1 介绍

机器人学习的演变展示了在训练能够在各种环境中执行多样化任务的策略方面的显著进展[111]。一个有前景的方向是视觉-语言-动作（VLA）模型，它利用预训练的多模态大型语言模型（MMLMs）[2629]的丰富理解能力，直接将自然语言指令和视觉观察映射到机器人动作[15, 1, 12]。尽管这些方法[30 32, 13, 1, 3342]取得了显著成果，但它们从观察到动作的直接映射缺乏人类在理解和推理未来环境知识时所具有的闭环预测能力。

为了将未来知识预测纳入VLA，大多数现有方法 [43, 5, 4455] 利用副驾驶生成模型生成未来帧/关键点，然后基于目标图像预测动作序列。一些方法 [5661] 将像素级图像预测与动作预测整合在一个框架中，这利用了预测和规划之间的协同作用，并将预测视为类似于大型语言模型（LLMs）[62] 中使用的中间推理步骤的过程 [58]。尽管在整合密集视觉预测方面取得了早期成功，但这些方法自然存在局限性：（1）冗余像素信息：预测图像与当前观察之间存在显著重叠，使得预测效率和有效性降低。（2）缺乏空间信息：缺乏环境的显式三维知识 [6366, 22]。（3）缺乏高层次知识预测：缺失对未来状态的高层次理解，例如语义信息。因此，我们认为现有方法（图1（a-c））不足以在世界级未来知识的背景下预测未来状态，以实现更全面的预测-动作循环。

![](images/1.jpg)  

Figure 1: (a) Vanilla VLA directly maps visual observations and language instructions to actions. (b) Models leveraging separate image/video generation or copilot models to generate future frames or trajectories, subsequently guiding an action head. (c) VLA variants explicitly predict a subgoal image as an intermediate visual reasoning step prior to action generation. (d) Our proposed DreamVLA, which explicitly predicts dynamic regions, depth map, semantics (DINOv2 and SAM) knowledge, significantly enhances the model's action reasoning and generalization.

为了解决这些问题，我们提出了DreamVLA，一个将全面的世界知识预测纳入视觉-语言-行动模型的新框架，从而建立了用于操控任务的感知-预测-行动循环。如图1(d)所示，我们提出的方法不是直接生成整个未来帧，而是引入世界嵌入来预测与机器人执行高度相关的全面世界知识，例如动态区域、深度和高级语义特征。这种方法与人类与世界互动的方式相一致，强调相关的变化和世界知识。通过梦境/预测环境中这些有针对性的方面，我们旨在为模型提供简洁且相关的中间表示，从而促进更有效的行动规划。

为了获得全面的世界知识，我们的方法包含三个关键特征：(1) 动态区域预测。我们利用现成的光流预测模型[67, 68]来识别场景中的动态区域，使模型能够集中于运动区域，这些区域对于任务执行至关重要，而不是冗余的帧重建。(2) 深度感知预测。我们采用深度估计技术[63]生成每帧的深度图，提供有价值的空间上下文，有助于理解环境的三维结构。(3) 高级基础特征。我们结合与视觉基础模型如DINOv2[69]和SAM[70]对齐的语义特征。通过这种方式，DreamVLA为模型提供了更全面和有效的规划与执行路径。此外，我们采用了一种块级结构的注意机制，屏蔽了它们之间的相互注意，防止信息泄漏，保持每个表示的清晰和解耦。由于世界和动作嵌入占据相同的潜在空间并共享相似的统计特征，简单的多层感知器（MLP）头无法解耦特定模态的信息或利用它们的跨模态相关性。我们采用了一种基于扩散的变换器，解耦动作表示与共享潜在特征以推理动作。

通过对公共基准进行广泛的实验，我们发现融入世界知识预测会显著提升性能。我们的方法在CALVIN基准上达到了最先进的性能（平均长度为4.44），并且我们分析了世界知识成分的影响，发现它们在不同方面都有所改善。具体而言，综合消融实验表明，单独预测动态区域带来了最大的收益，而深度和语义线索则提供了较小且大致相等的好处。更糟糕的是，当单独使用深度或语义预测时，不仅无法提供帮助，反而可能降低性能。在模拟和真实世界的广泛实验中证明了我们方法的有效性。

我们工作的主要贡献总结如下：

我们将视觉-语言-行动模型重新构建为感知预测-行动模型，使得模型显式预测一组紧凑的动态、空间和高级语义信息，为规划提供简洁而全面的前瞻提示。我们引入了一种基于块的结构注意机制，结合扩散变换解码器，以抑制来自跨类型知识泄漏的表示噪声，从而实现一致的多步骤行动推理。

•DreamVLA在CALVIN ABC-D基准测试上设立了新的艺术标准（4.44平均任务长度），在模拟平台上超越了之前的方法，提升幅度高达$3.5\%$，并将现实世界的成功率提升至$76.7\%$。消融研究确认了每个组件的贡献。

# 2 相关研究文献

# 2.1 视觉-语言-行动模型

最早的VLA [16, 71, 2, 7274] 通过将预训练的视觉-语言表示与任务条件的策略结合，为操控和控制奠定了基础。受大型语言模型 [7578] 和多模态大型语言模型 [28, 26, 79, 65, 80] 最近进展的启发，以及大规模机器人数据集 [12, 8183] 的出现，VLA 已成为机器人学习的一种趋势。RT系列 [2, 84, 85] 是首次尝试在机器人演示数据集上微调 MLLM 的先锋，取得了强大的准确性和泛化能力。在此基础上，许多先进的技术 [30, 32, 13, 1, 33, 34, 73, 3537, 8688, 38, 89] 被开发出来以提升性能。同时，考虑到扩散模型在建模多峰方面的优势，一些研究人员 [9094] 采用不同的架构从基于观察、任务指令和机器人先验知识的噪声中采样动作。考虑到这种直接将观察和指令映射到动作的方式缺乏类似LLM的推理步骤 [62]，大多数现有的方法 [43, 5, 4449] 利用共助图像/视频生成模型生成未来帧，然后基于目标图像预测动作序列。然而，上述方法仍需要额外的生成模型，这会引入推理时间和计算负担。因此，几个方法 [5661] 在一个单一框架中将像素级预测与动作预测整合，利用预测与规划的协同作用。尽管取得了一定的成功，但这些方法自然存在冗余重构的局限性 [95]，并且缺乏空间和语义信息。

# 2.2 机器人知识预测

学习未来世界知识以便对机器人进行训练已越来越受到关注，以实现行动预测循环的策略。早期的尝试基于现成的视频生成模型来实现这一点，并将目标图像或状态输入到策略模型中进行逆动力学。这种两阶段的训练策略易于实现，但受限于视频生成模型的性能和延迟。更先进的解决方案通过要求策略生成明确的预测来将预测与控制结合在一起，除了行动外。这些工作具体要求策略输出（i）高层次的子任务/选项序列或语言计划，以分解长期目标，（ii）潜在的未来嵌入/潜在行动，以紧凑编码即将到来的运动意图，（iii）完整的子目标图像或短期视觉滚动，以预测场景如何演变，以及（iv）以物体为中心的信号（例如，边界框），以捕捉与操控相关的动力学。这一系列工作展示了更好的性能和泛化能力。然而，未来状态受到冗余视觉信息或单调状态的限制。与之前的工作相比，DreamVLA提议以一种高效（动态区域）和有效（全面知识）的方式预测未来知识，展示了强大的性能和泛化能力。

# 3 方法论

# 3.1 问题定义和符号

我们的目标是通过利用丰富的世界知识作为指导原则来提升机器人执行能力。在这个背景下，我们将视觉语言—行动推理构建为一个逆动力学问题 [103, 56, 49]，将未来世界知识预测视为机器人控制的中间推理，从而充分释放预测与执行的协同效应。在每个时刻 $t$ ，机器人接收三种异构信号：自然语言指令 $l$ ，原始视觉帧 $o _ { t }$ ，以及其本体状态 $s _ { t }$ 。为了注入前瞻性推理，我们定义了一组特殊的令牌，称为 <dream> 查询 [79]，并将所有输入连接成一个序列。一个统一模型 $\mathcal { M }$ 将这些输入映射为一个紧凑的潜在表示，我们称之为世界嵌入：

![](images/2.jpg)  

Figure 2: Framework Overview. Given the current robot state $s _ { t }$ , observation $o _ { t }$ , and language instruction, DreamVLA encodes multimodal inputs via frozen text, visual encoders and a tunable state encoder. These tokens, together with a learnable set of <dream> queries, are processed by a large language model to produce world embedding. Three lightweight decoders then project each corresponding element of this embedding into the dynamics region $\hat { f } _ { t + n }$ , monocular depth $\hat { d } _ { t + n }$ and high-level semantics $\hat { c } _ { t + n }$ . A separate <action> query draws a latent action embedding, which conditions a diffusion transformer that refines Gaussian noise into an $n$ -step action sequence $\hat { a } _ { t : t + n - 1 }$ . The dashed box highlights prediction heads that are used only during training; inference skips these heads and operates directly on the world embedding.

$$
\mathbf { w } _ { t + n } = \mathcal { M } \left( l , o _ { t } , s _ { t } \middle | < \middle \mathrm { d } \mathbf { r } \mathbf { e } \mathbf { a m } > \right) .
$$

接下来，世界嵌入预测结合运动线索、空间细节和高级语义的综合世界知识。具体来说，一组预测器 $\mathcal { P }$ 预测 $n$ 步。

$$
\begin{array} { r } { \hat { p } _ { t + n } = \mathcal { P } \big ( \mathbf { w } _ { t + n } \big ) = \big [ \hat { f } _ { t + n } , \hat { d } _ { t + n } , \hat { c } _ { t + n } \big ] , } \end{array}
$$

其中 $\hat { f } _ { t + n }$ 标记动态区域，$\hat { d } _ { t + n }$ 编码单目深度，$\hat { c } _ { t + n }$ 可选地存储高级语义特征（例如 DINOv2 [69]，SAM [70]）。

给定世界嵌入 $\mathbf { w } _ { t + n }$，<action> 查询被统一模型 $\mathcal { M }$ 分配到潜在动作嵌入，以聚合相关的动作信息。去噪扩散变换器 $\mathcal { D }$ 基于潜在特征制定了一个 $n$ 步动作：

$$
\hat { a } _ { t : t + n - 1 } = \mathcal { D } \big ( \mathcal { M } \big ( l , o _ { t } , s _ { t } , < \mathtt { d r e a m } > \vert < \mathtt { a c t i o n } > \big ) \big ) ,
$$

因此完成了一个在训练和推理中都相同的感知-预测-行动循环。本章其余部分详细介绍了系统组件——编码器、世界知识预测器和基于扩散的动作生成器——它们实现了上述公式。

# 3.2 模型架构

如图2所示，我们的DreamVLA框架由三个核心模块组成，这些模块在统一的变压器架构中运行。首先，异构输入——包括自然语言$l$、视觉观察$o_{t}$和本体状态$s_{t}$——分别通过特定于模态的编码器进行处理。我们使用CLIP [101]文本嵌入对语言指令进行编码，通过掩蔽自编码器[104]对视觉帧进行编码，以获取时空补丁表示，并通过几个卷积层和全连接层对本体信号进行编码。在编码后，一组可学习的查询被附加到这些多模态嵌入上，称为<梦>和<行动>，其中<梦>包含三个子查询（动态、深度和语义），可用于特定知识的预测。随后，我们利用基于GPT-2 [105]的大型语言模型，通过精心结构化的因果和非因果注意机制（图4）在模态和查询之间进行集成和关注。这有效地将低级感知信号融合成紧凑且语义连贯的世界状态表示。

![](images/3.jpg)  

Figure 3: Visualization of dynamic regions over time. We show the static camera (left) and wrist-mounted camera (right) observations alongside the corresponding dynamic masks generated by our method at multiple time steps. The masks highlight dynamic regions by leveraging optical flow trajectories extracted via CoTracker [68, 67]. Compared to the original observations, our method objects and end-effector), enabling more structured and efficient action reasoning.

最后，专门的轻量级输出头由浅层卷积层组成，将世界嵌入解码为明确的预测：重建预期的动态区域、单目深度和语义特征。在推理过程中，DreamVLA完全跳过解码器，从而节省了大量计算资源。相反，模型输出一个世界嵌入，封装了未来动态、深度和语义的预测，而不进行像素级重建，从而保留了未来状态推理的准确性提升，同时保持低延迟。与此同时，我们采用去噪扩散变换器 [90] 将潜在动作嵌入解码为可执行的机器人动作序列。这些组件共同使DreamVLA能够以端到端的方式执行稳健的预测视觉语言—动作推理。

# 3.3 综合世界知识预测

预测未来的重要性比仅仅重现原始未来图像更有价值。DreamVLA明确预测了未来世界知识中最相关的内容，以便进行操作，包括 (i) 运动中心的动态区域，(ii) 3D深度几何，以及 (iii) 高层次语义。这些互补信号为原始像素提供了一个紧凑的、结构化的替代品，并为策略提供了逆向动力规划的前瞻性上下文。

以运动为中心的动态区域重建。预测动态区域可以告诉机器人场景中哪些部分即将移动，使模型能够捕捉当前场景、语言指令和实现预测运动所需动作之间的统计联系。如图3所示，DreamVLA既不预测密集光流，也不合成整个未来帧。相反，我们首先应用CoTracker [67, 68]来提取动态区域，即与机器人末端执行器或其他可移动物体一起移动的像素，然后训练DreamVLA仅重建这些区域。此外，使用不对称标记器生成重建目标可以进一步提升性能[104]。从离散变分自编码器(dVAE) [106109]的角度看，整体优化是最大化对数似然$\mathrm { P } ( \bar { x } _ { i } | \tilde { x } _ { i } )$的证据下界(ELBO) [110 112, 66]。令$x$表示原始图像，$\tilde { x }$为掩码运动区域，$z$为重建目标。生成建模可以描述为：

$$
\sum _ { ( z _ { i } , \bar { z } _ { i } ) \in \mathcal { D } } \log \mathrm { P } ( x _ { i } | \tilde { x } _ { i } ) \geq \sum _ { ( x _ { i } , \bar { x } _ { i } ) \in \mathcal { D } } \left( \mathbb { E } _ { z _ { i } \sim \mathrm { Q } _ { \phi } \left( \mathbf { z } \mid x _ { i } \right) } \left[ \log \mathrm { P } _ { \psi } ( x _ { i } | z _ { i } ) \right] - D _ { \mathrm { K L } } \left[ z , \mathrm { P } _ { \theta } ( \mathbf { z } | \hat { z } _ { i } ) \right] \right) ,
$$

其中 $\mathrm { P } _ { \psi } ( x | z )$ 是用于恢复原始数据的分词解码器，$\hat { z } _ { i } = \mathrm { Q } _ { \phi } ( \mathbf { z } | \tilde { x } _ { i } )$ 表示来自掩蔽数据的掩蔽运动区域标记，而 $\mathrm { P } _ { \theta } ( z | \hat { z } _ { i } )$ 以自编码方式重建掩蔽标记。在这里，$\mathrm { P } _ { \theta } ( z | \hat { z } _ { i } )$ 为零，动态区域预测损失可以表述为：

$$
\mathcal { L } _ { \mathrm { d y n } } = \frac { 1 } { | \mathcal { D } | } \sum _ { x _ { i } \in \mathcal { D } } \mathbb { E } _ { z \sim Q _ { \phi } ( z | x _ { i } ) } \Big [ - \log \mathrm { P } _ { \psi } \big ( ( x _ { i } ) _ { \mathcal { M } } \mid z \big ) \Big ] .
$$

深度预测。预测深度场的演变可以告诉机器人下一步应该如何移动，使其朝向自由空间，远离即将到来的障碍物。如果可用深度传感器，我们会用真实地图来监督DreamVLA；在没有深度传感的低成本平台上，我们则通过单一RGB流来幻觉未来的几何形状。为此，我们将Depth-Anything [63, 64]的预测视为自我监督的教师，并训练一个专门的深度查询来回归对齐的未来地图 $\hat { d } _ { t + n }$。目标是一个尺度归一化的均方误差，

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { d e p t h } } = \frac { 1 } { H W } \displaystyle \sum _ { i , j } \big ( \hat { d } _ { t + n } ^ { ( i , j ) } - \alpha { d } _ { t + n } ^ { ( i , j ) } \big ) ^ { 2 } , } \\ & { \quad \quad \alpha = \frac { \sum _ { i , j } \hat { d } _ { t + n } ^ { ( i , j ) } { d } _ { t + n } ^ { ( i , j ) } } { \sum _ { i , j } { d } _ { t + n } ^ { ( i , j ) } } , } \end{array}
$$

其中 $\alpha$ 消除了单目方法无法解决的全局尺度模糊性。在实际操作中，这个简单的损失是足够的：教师提供有度量意义的深度，尺度归一化合成和碰撞检查同时忽略任何任意的全球尺度变化。

对比语义预测。预测未来语义教会机器人哪些物体或区域对于任务是重要的，提供一个高层次的上下文（例如，物体身份和可用性），指导目标选择和抓取选择。为了学习这些语义，DreamVLA 使用 InfoNCE 损失 [113, 66] 预测未来的 DINOv2 [69] 和 SAM [70] 特征 $\hat { c } _ { t + n }$：真实特征为正样本，而空間偏移特征作为负样本。这鼓励具有区分性的信息预测，模型必须在 plausible 但错误的未来中选择正确的物体语义：

$$
\mathcal { L } _ { \mathrm { s e m } } = - \log \frac { \exp \left( \hat { c } _ { t + n } ^ { \top } c _ { t + n } / \tau \right) } { \sum _ { k } \exp \left( \hat { c } _ { t + n } ^ { \top } c _ { k } / \tau \right) } ,
$$

其中 $k$ 代表空间中的标记数量，$\tau$ 表示温度。

跨类型知识解缠的结构化注意力。为了保持清晰的跨类型知识边界，<dream> 被分解为三个子查询（动态、深度和语义）。如果这些子查询能够自由地相互关注，高频流动细节将会污染深度推理，语义线索可能渗入运动特征，导致嘈杂的混合表征。因此，我们屏蔽它们的相互注意力：每个子查询仅关注共享的视觉、语言和状态标记，而三者之间的直接链接被禁用，保持其潜在特征的解缠和无交叉干扰。如图4所示，<dream> 和 <action> 查询也采用限制在过去上下文中的因果注意力，这保持了时间因果关系。这种有组织的模式反映了Mixture-of-Experts (MoE) 网络中使用的专业路由[114]。通过避免跨模态泄漏，结构化注意力为动作预测提供了清晰的未来世界知识，提高了鲁棒性，并保持了时间一致性。

![](images/4.jpg)  

Figure 4: Block-wise structured attention.

# 3.4 通过去噪扩散变换器的逆动力学

给定两个有序观测 $o_{t}$ 和 $o_{t+1}$，经典的逆动态推断中间动作 $\hat{a}_{t}$。我们通过预测一个完整的动作序列 $\hat{a}_{t:t+n-1}$ 来扩展这一表述，该序列以当前观测 $o_{t}$ 和未来潜在世界嵌入 ${\bf w}_{t+n}$ 为条件。具体来说，DreamVLA 首先通过专用的动作查询和模型的因果注意力，将这一已有预测未来动态、深度和语义丰富的潜在嵌入聚合成一个紧凑的动作嵌入。由于世界和动作嵌入占据相同的潜在空间并共享相似的统计特性，一个简单的 MLP 头部无法分离特定模态的信息或利用它们的跨模态相关性。因此，我们采用去噪扩散变换器 (DiT) [90, 115] 作为动作头。以动作嵌入为条件，DiT 采用迭代自注意力和去噪，将感知预测与控制先验融合，并将高斯噪声转化为 $n$ 步轨迹 $a_{t:t+n-1}$，产生连贯、多样且物理上合理的动作序列。动作预测的损失可以表述为：

Table 1: CALVIN ABC-D results. We present the average success computed over 1000 rollouts for each task and the average number of completed tasks to solve 5 instructions consecutively (Avg. Len.). DreamVLA shows significant superiority over baselines. The best results are bolded.   

<table><tr><td rowspan="2">方法</td><td colspan="6">连续完成的任务</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>平均长度 ↑</td></tr><tr><td>Roboflamingo [30]</td><td>82.4</td><td>61.9</td><td>46.6</td><td>33.1</td><td>23.5</td><td>2.47</td></tr><tr><td>Susie [118]</td><td>87.0</td><td>69.0</td><td>49.0</td><td>38.0</td><td>26.0</td><td>2.69</td></tr><tr><td>GR-1 [14]</td><td>85.4</td><td>71.2</td><td>59.6</td><td>49.7</td><td>40.1</td><td>3.06</td></tr><tr><td>3D Diffusor Actor [93]</td><td>92.2</td><td>78.7</td><td>63.9</td><td>51.2</td><td>41.2</td><td>3.27</td></tr><tr><td>OpenVLA [1]</td><td>91.3</td><td>77.8</td><td>62.0</td><td>52.1</td><td>43.5</td><td>3.27</td></tr><tr><td>RoboDual [119]</td><td>94.4</td><td>82.7</td><td>72.1</td><td>62.4</td><td>54.4</td><td>3.66</td></tr><tr><td>UNIVLA [120]</td><td>95.5</td><td>85.8</td><td>75.4</td><td>66.9</td><td>56.5</td><td>3.80</td></tr><tr><td>Pi0 [32]</td><td>93.8</td><td>85.0</td><td>76.7</td><td>68.1</td><td>59.9</td><td>3.92</td></tr><tr><td>CLOVER [121]</td><td>96.0</td><td>83.5</td><td>70.8</td><td>57.5</td><td>45.4</td><td>3.53</td></tr><tr><td>UP-VLA [57]</td><td>92.8</td><td>86.5</td><td>81.5</td><td>76.9</td><td>69.9</td><td>4.08</td></tr><tr><td>Robovlm [37]</td><td>98.0</td><td>93.6</td><td>85.4</td><td>77.8</td><td>70.4</td><td>4.25</td></tr><tr><td>Seer [56]</td><td>96.3</td><td>91.6</td><td>86.1</td><td>80.3</td><td>74.0</td><td>4.28</td></tr><tr><td>VPP [49]</td><td>95.7</td><td>91.2</td><td>86.3</td><td>81.0</td><td>75.0</td><td>4.29</td></tr><tr><td>DreamVLA</td><td>98.2</td><td>94.6</td><td>89.5</td><td>83.4</td><td>78.1</td><td>4.44</td></tr></table>

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { { D i T } } } = \mathbb { E } _ { \tau , \varepsilon } \big \| \varepsilon - \varepsilon _ { \theta } \big ( \sqrt { \bar { \alpha } _ { \tau } } a _ { t : t + n - 1 } + \sqrt { 1 - \bar { \alpha } _ { \tau } } \varepsilon , \tau , \mathbf { c } \big ) \big \| _ { 2 } ^ { 2 } , } \end{array}
$$

其中 $\varepsilon _ { \theta }$ 是 DiT 去噪器，$\varepsilon \sim \mathcal { N } ( 0 , I )$，$\bar { \alpha } _ { \tau }$ 遵循余弦噪声调度，$\mathbf { c }$ 是从大型语言模型中获得的潜在动作嵌入。推理通过抽取高斯样本并运行学习到的逆扩散进行，产生多样化且在物理上合理的轨迹，从而闭合感知-预测—动作循环。

# 4 实验

# 4.1 实施细节

所有模型均在PyTorch中实现，并在NVIDIA 8 A800 GPU上训练。我们使用AdamW优化器，初始学习率为$1 \times 10^{-3}$，权重衰减为$1 \times 10^{-4}$，并采用余弦学习率调度，线性预热为$5\%$。批量大小设置为64，每种模态的查询长度设置为9，DiT中的扩散步骤设置为10。我们将动态区域、深度和分割预测损失的权重分别设置为$\lambda_{\mathrm{dyn}}=0.1$、$\lambda_{\mathrm{depth}}=0.0001$、$\lambda_{\mathrm{sem}}=0.1$，以及动作损失的权重为$\lambda_{\mathrm{DiT}}=1$。我们首先在CALVIN的无语言分割和完整的DROID数据集上预训练DreamVLA。对于LIBERO基准，我们首先在LIBERO-90上预训练DreamVLA，然后在每个轨道上进行微调。该模型预测整个帧而不是综合知识，从而保持存储和计算要求在可管理范围内。然后，我们使用综合世界知识预测目标在每个目标数据集上微调DreamVLA。所有模型训练20个周期，并选择验证成功率（SR）最高的检查点进行最终评估。

# 4.2 模拟基准实验

仿真设置。我们在CALVIN [117] 和LIBERO [122] 基准上评估DreamVLA。CALVIN是一个为学习长期、基于语言的机器人操作策略而设计的仿真基准。它包括四个不同的操作环境，每个环境有超过六小时的遥控播放数据，捕获自多个传感器，包括静态和夹具安装的RGB-D相机、触觉图像和本体感觉读取。我们报告每个轨道的成功率和5个任务的平均长度。此外，还在LIBERO [122]上进行评估，LIBERO是一个涵盖四个套件（LIBERO-Spatial/-Object/-Goal/-Long）的仿真基准。每个套件包含10个任务，辅以50个由人类遥控操作的演示，针对空间推理、以物体为中心的操作和目标完成。

Table 2: The extended LIBERO experiments. DreamVLA achieves the best or competitive performance across all tracks compared to previous approaches. The best results are bolded.   

<table><tr><td rowspan="2">方法</td><td colspan="4">得分（%）</td><td rowspan="2">平均</td></tr><tr><td>空间</td><td>对象</td><td>目标</td><td>长</td></tr><tr><td>扩散策略 [90]</td><td>78.3</td><td>92.5</td><td>68.3</td><td>50.5</td><td>72.4</td></tr><tr><td>Octo [13]</td><td>78.9</td><td>85.7</td><td>84.6</td><td>51.1</td><td>75.1</td></tr><tr><td>OpenVLA [1]</td><td>84.7</td><td>88.4</td><td>79.2</td><td>53.7</td><td>76.5</td></tr><tr><td>SpatialVLA [36]</td><td>88.2</td><td>89.9</td><td>78.6</td><td>55.5</td><td>78.1</td></tr><tr><td>CoT-VLA [58]</td><td>81.1</td><td>87.5</td><td>91.6</td><td>87.6</td><td>69.0</td></tr><tr><td>DreamVLA</td><td>97.5</td><td>94.0</td><td>89.5</td><td>89.5</td><td>92.6</td></tr></table>

结果。如表1所示，DreamVLA在ABC-D任务上取得了最高的性能。我们的方法超越了Roboflamingo [30]、3D Diffusor Actor [93]、OpenVLA [1]、RoboDual [119]、UNIVLA [120]、Robovlm [37]和GR1 [14]，这些方法直接将RGB/深度图像投影到动作信号，如图1(a)所示。与使用副驾驶模型生成子目标图像作为输入的方法相比，如Susie [118]和CLOVER [121]，如图1(b)所示，我们的模型显著实现了更准确的控制。DreamVLA的表现超越了UP-VLA [57]、Seer [56]和VPP [49]等方法，如图1(c)所示，这些方法将整个子目标图像的前瞻合并为一个VLA，以利用更集成的设计和联合优化，表明我们的方法在模拟任务中具有更好的多任务学习和泛化能力。对于LIBERO基准测试 [122]，DreamVLA在所有轨道上表现出更好或可比的能力，相比于之前的方法，通过未来世界知识预测，如表2所示。

# 4.3 现实世界实验

为了评估我们方法在现实世界中的有效性，我们使用Franka Panda机械臂进行夹具抓取的实地实验。在我们的设置中，两个RealSense D415相机捕捉RGB图像。其中一个在第三人称视角，另一个位于机械臂的末端，如图5所示。我们收集了四类物体进行两个任务：拾取和放置。此外，我们还进行抽屉开关任务的实验，如补充材料所示。根据[56]，我们在DROID [82]上对DreamVLA进行了预训练，该数据集包含多种场景中Franka机器人的大规模轨迹。为了公平比较，我们在收集的演示数据集上对Diffusion Policy [90]、Octo-Base [13]、OpenVLA [1]和DreamVLA进行了微调，每个任务包含100条轨迹。

![](images/5.jpg)  

Figure 5: Real-world experiment setup.

在实验设置中，每次试验允许最多进行20次连续尝试。在抓取实验中，物体被随机放置在桌面上。如果机器人手臂在预定义的尝试限制内成功抓取目标物体，则该次试验被认为是成功的。在放置实验中，机器人需要在允许的尝试次数内完成抓取和放置操作。在抽屉操作任务中，抽屉随机放置在机器人手臂面前。如果抽屉位移超过10厘米，则实验被认为是成功的，这表明有效的交互。结果如表3所示，表明我们的方法优于其他方法。更多现实世界实验的可视化结果展示在补充部分。

Table 3: Real-world evaluation with the Franka Robot across three tasks.   

<table><tr><td rowspan="2">方法</td><td colspan="3">捡取</td><td colspan="3">放置</td><td colspan="3">抽屉</td><td>总任务</td></tr><tr><td>瓶子</td><td>玩偶</td><td>平均</td><td>香蕉</td><td>辣椒</td><td>平均</td><td>打开</td><td>关闭</td><td>平均</td><td>平均</td></tr><tr><td>扩散策略[90]</td><td>50.0</td><td>70.0</td><td>60.0</td><td>65.0</td><td>45.0</td><td>55.0</td><td>15.0</td><td>60.0</td><td>37.5</td><td>50.8</td></tr><tr><td>Octo-Base [13]</td><td>50.0</td><td>60.0</td><td>55.0</td><td>40.0</td><td>50.0</td><td>45.0</td><td>20.0</td><td>50.0</td><td>35.0</td><td>45.0</td></tr><tr><td>OpenVLA [1]</td><td>50.0</td><td>40.0</td><td>45.0</td><td>20.0</td><td>30.0</td><td>25.0</td><td>40.0</td><td>30.0</td><td>35.0</td><td>35.0</td></tr><tr><td>DreamVLA</td><td>85.0</td><td>80.0</td><td>82.5</td><td>80.0</td><td>80.0</td><td>80.0</td><td>70.0</td><td>65.0</td><td>67.5</td><td>76.7</td></tr></table>

<table><tr><td rowspan="2">方法</td><td colspan="6">连续完成的任务</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>平均长度↑</td></tr><tr><td>香草VLA*</td><td>93.0</td><td>82.4</td><td>72.3</td><td>62.6</td><td>53.3</td><td>3.64</td></tr><tr><td>+ 动态区域</td><td>97.6</td><td>92.6</td><td>87.5</td><td>80.4</td><td>73.7</td><td>4.32</td></tr><tr><td>+ 深度</td><td>98.3</td><td>94.3</td><td>88.5</td><td>82.0</td><td>77.2</td><td>4.40</td></tr><tr><td>+ 语义</td><td>98.2</td><td>94.6</td><td>89.5</td><td>83.4</td><td>78.1</td><td>4.44</td></tr></table>

Table 4: Performance comparison between predicting the optical flow and dynamic region. Notably the \* denotes that this result is from [56].   

# 4.4 消融研究

在本节中，我们设计实验以探讨以下问题。

# 问题1：每个模态特征的贡献是什么？

DreamVLA的核心动机是使模型能够预测未来的全面视觉知识，以增强行动推理。然而，并非所有类型的知识对后续执行的贡献是相同的。我们考虑四种类型的预测知识：动态区域、深度和从SAM和DINO衍生的语义分割特征。如图6所示，我们首先独立地用每种知识进行模型训练。绿色虚线表示不使用知识预测的Vanilla VLA基线的性能。在所有类型中，预测动态区域被证明是最有益的，因为这些掩码明确标示了即将改变的像素，因此几乎与策略的行动语义完美对齐。相比之下，仅用深度图、DINO或SAM特征监督网络不仅无助于提升性能，反而往往会导致性能下降。我们分析认为，这一差距源于每个辅助目标与下游目标的匹配程度：动态区域标签提供的梯度强化了行动头，而深度回归和高维特征匹配（DINO/SAM）却注入了大量噪声损失，这些损失主导了优化。在有限的模型注意力预算下，这些竞争性梯度稀释了与任务相关的特征，并推动主干模型朝向次优解，导致观察到的性能下降低于虚线基线。

接下来，我们同时训练所有五个知识头（All），并进行消融研究（All-X），在此过程中，我们一次性去除一个知识信号以评估其贡献。去除Flea导致性能显著下降，确认其关键作用。有趣的是，去除DINO的结果却表现出类似甚至更好的性能，这表明并非所有语义信号在预测结果时都是同等有帮助或稳定的，因此我们在后续的消融实验中仅使用来自SAM的语义特征。表4显示了所有消融实验中明显且递减的回报模式。

# Q2: 辅助任务与未来知识预测：哪一个推动了改进？

表5对比了两种训练方案：预测完整的世界知识和进行辅助重建，显示前者明显优越。在我们的消融实验中，每种预测策略都被单独替换为其重建对应物，但每次替换一致降低了性能：仅训练重新绘制当前RGB、深度、语义或DINOv2特征的VLA能够处理前几个动作，但很快就失去了连贯性，而训练预测下一个动态区域、深度图和语义的网络在整个轨迹中保持准确性，并在失败之前可以完成更多任务。原因在于预测提供了一种更丰富的、以行动为导向的信号，引导学习朝向将推动即将到来的决策的像素，而重建仅仅是重新审视控制策略实际上并不需要的背景细节。

Q3：我们为什么使用光流作为掩码，而不是直接预测它？

![](images/6.jpg)  

Figure 6: CALVIN ABC-D performance with respect to different combinations of knowledge prediction. $\mathbf { A l l = a l l }$ of five models, and All- $\mathbf { \nabla } \cdot \mathbf { X } =$ taking X out of All.

Table 5: Performance comparison between cotraining with auxiliary tasks and predicting the comprehensive world knowledge.   

<table><tr><td rowspan="2">方法</td><td colspan="6">连续完成的任务</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>平均长度</td></tr><tr><td>辅助</td><td>97.7</td><td>92.3</td><td>85.6</td><td>79.5</td><td>74.2</td><td>4.14</td></tr><tr><td>预测</td><td>98.2</td><td>94.6</td><td>89.5</td><td>83.4</td><td>78.1</td><td>4.44</td></tr></table>

<table><tr><td rowspan="2">方法</td><td colspan="6">连续完成的任务</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>平均长度</td></tr><tr><td>光学</td><td>97.6</td><td>92.4</td><td>86.8</td><td>81.7</td><td>75.4</td><td>4.23</td></tr><tr><td>动态</td><td>98.2</td><td>94.6</td><td>89.5</td><td>83.4</td><td>78.1</td><td>4.44</td></tr></table>

Table 6: Performance comparison between predicting the optical flow and dynamic region.   

为了证明我们选择采用以运动为中心的动态区域而非直接流量预测的合理性，我们在相同设置下实现了这两种变体（表6）。在光流设置中，模型必须预测完整的未来流场和子目标图像，这显著增加了训练的复杂性。这额外的负担在多步成功率上表现得十分明显，相比之下，我们的动态区域方法仅仅利用预训练的流量模型来获取一个二进制掩码，专注于“相关运动发生在哪里”，带来了显著的改进。

# Q4：DreamVLA中结构化注意力的有效性。

为了证明我们提出的结构注意机制在图4中的有效性，我们用一个普通的因果掩码替换了它，同时保持其他一切不变。在这个设置中，每个<dream>查询，包括用来捕捉语义的查询，都可以读取同一步骤中产生的流和深度标记；额外的交叉查看混合了无关信号，增加了梯度噪声，并迅速降低了长时间控制的效果。我们的掩码去除了所有查询之间的边缘，因此<action>查询仅咨询过去的语言、状态和多模态预测，绝不涉及它们的兄弟查询。表7展示了结果：因果变体对普通VLA带来了边际改进，而块稀疏版本在整个过程中保持了高成功率，确认了阻止步内泄漏的重要性。

# Q5: 我们可以使用共享查询来预测综合世界知识吗？

与其为动态区域、深度和语义特征分配单独的查询，不如让一组共享查询来预测所有信号。为了测试这个想法，我们将每个世界嵌入向量分成四个相等的子空间，每个四分之一用于传递不同的模态。表8显示，共享查询设计会影响动作表现：在同一个查询中混合模态会引入交叉干扰，因此扩散头接收到噪声特征。相比之下，为每个模态单独分配查询能够保持表示的分离，从而带来明显的性能提升。

# Q6: <dream> 查询中每种方式的查询计数影响。

每个 <dream> 查询包含三个元素组：动态、深度和语义，每组分配 $K$ 个查询。我们将 $K \in \{ 4 , \bar { 9 } , 1 6 \}$ 进行变化，以检查其影响。当 $K = 4$ 时，有限的容量阻止模型编码细粒度的运动、几何和语义，因此即使内存使用最少，准确性也会下降。

<table><tr><td rowspan="2">数量</td><td colspan="6">连续完成的任务</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td> 平均长度</td></tr><tr><td>4</td><td>97.2</td><td>92.6</td><td>86.4</td><td>80.7</td><td>75.1</td><td>4.32</td></tr><tr><td>9</td><td>98.2</td><td>94.6</td><td>89.5</td><td>83.4</td><td>78.1</td><td>4.44</td></tr><tr><td>16</td><td>98.1</td><td>93.0</td><td>86.9</td><td>81.0</td><td>73.9</td><td>4.33</td></tr></table>

当 $K = 9$ 时，每种模态都有足够的带宽而不会过载主干，产生最佳的成功率和最长的连续任务执行时间。增加到 $K = 16$ 引入了多余的标记，这些标记争夺注意力并增加 GPU 内存，却没有带来额外的收益，并且略微降低了泛化能力。

Table 9: Performance comparison between different numbers of <dream> queries.   

Table 7: Performance comparison between vanilla causal and our structured attention.   

<table><tr><td rowspan="2">方法</td><td colspan="6">连续完成的任务</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>平均长度</td></tr><tr><td>因果</td><td>94.2</td><td>86.5</td><td>78.4</td><td>71.3</td><td>62.7</td><td>3.75</td></tr><tr><td>结构</td><td>98.2</td><td>94.6</td><td>89.5</td><td>83.4</td><td>78.1</td><td>4.44</td></tr></table>

<table><tr><td rowspan="2">方法</td><td colspan="6">连续完成的任务</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>平均长度</td></tr><tr><td>共享</td><td>95.5</td><td>90.1</td><td>83.8</td><td>76.9</td><td>70.4</td><td>4.17</td></tr><tr><td>分离</td><td>98.2</td><td>94.6</td><td>89.5</td><td>83.4</td><td>78.1</td><td>4.44</td></tr></table>

Table 8: Performance comparison between shared and seprated queries.   

# 5 限制与未来工作

虽然DreamVLA在视觉-语言-动作方面表现出色，并在CALVIN [117] 上达到了最先进的性能，但其当前的范围仍然较窄：它主要练习平行夹具的操作，依赖于RGB中心的数据，并且在几何和材料多样性有限的场景中进行训练。因此，我们计划 (i) 添加具有丰富接触注释的灵巧手演示 [123, 124]，(ii) 引入3D点云 [125, 126, 102, 66, 127, 128, 65, 129] 和空间信息 [22, 130]，触觉，并将其融合为体积世界状态，以及 (iii) 扩展数据收集和政策在职微调，以增强泛化能力和长时间稳定性。

# 6 结论

我们提出了DreamVLA，一个新颖的视觉-语言-动作框架，通过全面的世界知识预测实现逆动力学建模，支持操作任务的感知-预测-动作循环。DreamVLA利用动态区域引导的知识预测，结合空间和语义线索，生成紧凑且信息丰富的表示用于动作规划。我们引入了一种块状结构注意力机制，并配合扩散变换解码器，以抑制跨类型知识泄漏所带来的表示噪声，从而实现连贯的多步骤动作推理。在真实和模拟环境中进行的广泛实验表明了DreamVLA的有效性，在真实世界的机器人任务中达到了$76.7\%$的成功率，并在CALVIN ABC-D基准测试中超越了之前的方法。