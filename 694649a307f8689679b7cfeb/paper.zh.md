# PERCEIVER-ACTOR：用于机器人操作的多任务变换器

Mohit Shridhar 1, \* Lucas Manuelli 2 Dieter Fox 1,2 1华盛顿大学 2NVIDIA mshr@cs.washington.edu lmanuelli@nvidia.com fox@cs.washington.edu peract.github.io

摘要：变压器在视觉和自然语言处理领域的革命性突破源于其在大规模数据集上的扩展能力。但在机器人操控中，数据既有限又昂贵。那么，操控是否还能在正确的问题表述下从变压器中受益？我们通过 PERACT 进行探讨，该系统是一个用于多任务6自由度操控的语言条件行为克隆智能体。PERACT 利用 Perceiver Transformer [1] 对语言目标和 RGB-D 体素观测进行编码，并通过“检测下一个最佳体素动作”输出离散化动作。与在二维图像上操作的框架不同，体素化的三维观测和动作空间为高效学习6自由度动作提供了强大的结构先验。通过这种表述，我们为18个 RLBench 任务（共249种变体）和7个现实世界任务（共18种变体）训练了一个单一的多任务变压器，仅使用每个任务的少数示例。我们的结果表明，PERACT 在广泛的桌面任务中显著优于无结构的图像到动作智能体和3D卷积网络基线。 关键词：变压器、语言基础、操控、行为克隆

# 1 引言

Transformers [2] 在自然语言处理和计算机视觉中变得越来越普遍。通过将问题表述为序列建模任务，并在大量多样化的数据上进行训练，Transformers 在多个领域取得了突破性的成果 [3, 4, 5, 6]。即使在那些传统上不涉及序列建模的领域 [7, 8]，Transformers 也被作为一种通用架构 [9] 进行了采纳。但在机器人操作中，数据既有限又昂贵。我们是否仍然能够通过正确的问题表述，将 Transformers 的强大能力引入 6-自由度（6-DoF）操作？

语言模型操作于词元序列，视觉变换器操作于图像补丁序列。尽管存在像素变换器，但它们在数据效率上不如采用卷积或补丁的方法，后者能够有效利用图像的二维结构。因此，尽管变换器可能是领域无关的，但它们仍然需要正确的问题表述以提高数据效率。在直接将二维图像映射到六自由度动作的行为克隆智能体中，类似的效率问题也显而易见。像Gato和BC-Z这样的智能体展现了令人印象深刻的多任务能力，但它们需要数周甚至数月的数 据收集。相比之下，最近在强化学习中的研究，例如C2FARM，通过构建体素化的观察和动作空间，有效学习三维动作的视觉表征，采用了3D卷积网络。同样，在本研究中，我们旨在利用体素补丁的三维结构，实现高效的六自由度行为克隆，配合变换器（类似于视觉变换器如何利用图像补丁的二维结构）。

为此，我们提出了PERACT（全称为PERCEIVER-ACTOR），这是一个语言条件的BC智能体，能够通过每个任务仅需少量演示即学会模仿多种6自由度操作任务。PERACT对一系列RGB-D体素补丁进行编码，并预测离散化的平移、旋转和夹具动作，这些动作通过一个运动规划器在观察-行动循环中执行。PERACT本质上是一个通过监督学习训练的分类器，用于检测类似于先前工作如CLIPort的动作，然而我们的观察和动作是用3D体素表示而不是2D图像像素。由于高维输入的扩大问题，体素网格在端到端的BC方法中不如图像普遍。但是在PERACT中，我们使用Perceiver2 Transformer来编码高达100万个体素的高维输入，并仅需一小组潜在向量。这种基于体素的表述提供了强大的结构先验，具有多个好处：自然的方法融合多视角观察，学习稳健的以动作为中心的表示，以及在6自由度中实现数据增强，这些都通过关注多样而非狭窄的多任务数据来帮助学习可泛化的技能。

![](images/1.jpg)  
agent trained with just 53 demonstrations. See the supplementary video for simulated and real-world rollouts.

为了研究这种公式的有效性，我们在 RLBench [15] 环境中进行大规模实验。我们训练了一个单一的多任务智能体，该智能体在 18 个多样化任务上进行训练，这些任务有 249 种变体，涉及一系列的抓取和非抓取行为，例如将葡萄酒瓶放置在架子上和用棍子拖动物体（见图 1 a-j）。每个任务还包括几个姿态和语义变体，物体在放置、颜色、形状、大小和类别上各不相同。我们的结果表明，PERACT 显著优于图像到动作的智能体（提升 $3 4 \times$）和 3D 卷积网络基线（提升 $2 . 8 \times$），且未使用任何实例分割、物体姿态、记忆或符号状态的显式表示。我们还在 Franka Panda 上验证了我们的方法，使用从零开始训练的多任务智能体，涵盖 7 个真实世界任务，总共仅有 53 次示范（见图 1 k-o）。总结来说，我们的贡献如下：• 一种用于感知、行动和指定目标的新颖问题公式，基于变换器。 一个高效的以动作为中心的框架，用于将语言与 6 自由度动作相结合。 实证结果针对一系列模拟和真实世界任务研究多任务智能体。代码和预训练模型将发布在 peract . github . io。

# 2 相关工作

操控的愿景。传统上，机器人感知的方法使用明确的“物体”表示，如实例分割、物体类别、姿态等。这类方法在处理如布料和豆类等可变形和颗粒状物体时表现不佳，因为这些物体难以用几何模型或分割来表示。相比之下，近年来的方法通过学习以行动为中心的表示，而没有任何“物体性”假设，但它们仅限于简单的自上而下的2D环境和简单的拾取和放置基本操作。在3D领域，James等人提出了C2FARM，这是一种以行动为中心的强化学习代理，采用粗到细的3D-UNet主干网络。粗到细的方案具有有限的感受野，无法在最精细的层面上查看整个场景。相比之下，PERACT通过Transformer主干网络学习具有全局感受野的以行动为中心的表示。此外，PERACT采用行为克隆而非强化学习，这使我们能够通过语言目标为条件，轻松训练多任务代理以完成多个任务。

端到端操控方法 [28, 29, 30, 31] 对对象和任务做出的假设最少，但通常被表述为图像到动作的预测任务。直接在RGB图像上训练6自由度任务通常效率较低，通常需要多个示例或回合才能学习基本技能，例如重新排列物体。相比之下，PERACT使用了体素化的观测和动作空间，这在6自由度环境中显著提高了效率和鲁棒性。虽然其他6自由度抓取的研究 [32, 33, 34, 35, 36, 37] 使用了RGB-D和点云输入，但它们并未应用于顺序任务或与语言条件结合。另一类研究通过使用预训练的图像表示 [16, 38, 39] 来解决数据效率问题，以启动行为克隆（BC）。尽管我们的框架是从零开始训练的，但这样的预训练方法在未来的工作中可以整合在一起，以实现更高的效率和对未见对象的更好泛化。

用于智能体和机器人的变换器。变换器已成为多个领域的主流架构。从自然语言处理开始 [2, 3, 40]，最近在视觉领域 [4, 41]，甚至强化学习 [8, 42, 43] 中也得到了应用。在机器人技术中，变换器已被应用于辅助远程操作 [44]、四足 locomotion [45]、路径规划 [46, 47]、模仿学习 [48, 49]、形态控制 [50]、空间重排列 [51] 和抓取 [52]。变换器在多领域设置中也取得了令人瞩目的结果，例如在 Gato 中 [9]，单个变换器在 16 个领域上进行了训练，如标注、语言锚定、机器人控制等。然而，Gato 依赖于非常大的数据集，例如用于块堆叠的 15K 轮次和 Meta-World [53] 任务的 94K 轮次。我们的方法可能会补充像 Gato 这样的智能体，这些智能体可以利用我们的 3D 形式来提高效率和稳健性。

语言基础的操控。已有多项研究提出了将语言与机器人动作相结合的方法。然而，这些方法使用了解耦的感知和动作管道，语言主要用于引导感知。最近，提出了一些端到端的方法，用于在语言指令的条件下训练行为克隆（BC）智能体。这些方法需要成千上万的人类示例或在几天甚至几个月内收集的自主回合。相比之下，PERAcT仅需几分钟的训练数据即可学习稳健的多任务策略。为了进行基准测试，存在多个仿真环境，但我们使用RLBench，因为它具有丰富的6自由度任务和生成带有模板语言目标的演示的便利性。

# 3 感知者-行为者

PERACT 是一个语言条件下的行为克隆智能体，用于 6 自由度操控。其核心思想是学习基于语言目标的动作感知表征。在对场景进行体素化重建后，我们使用 Perceiver Transformer [1] 来学习每个体素的特征。尽管输入空间极其庞大 $( 1 0 0 ^ { 3 } )$，Perceiver 通过一小组潜在向量对输入进行编码。每个体素的特征随后被用于预测每个时间步中离散化的平移、旋转和夹爪状态下的下一个最佳动作。PERACT 纯粹依赖当前观察来确定在顺序任务中接下来要做什么。见图 2 以获取概述。第 3.1 节和第 3.2 节描述了我们的数据集设置。第 3.3 节描述了我们使用 PERACT 的问题表述，第 3.4 节提供了训练 PERACT 的细节。有关进一步的实现细节，请参见附录 B。

![](images/2.jpg)  
.

# 3.1 演示

我们假设可以访问一个数据集 $\mathcal{D} = \{ \zeta_{1}, \zeta_{2}, \ldots, \zeta_{n} \}$，包含 $n$ 个专家示范，每个示范都与英语语言目标 $\mathcal{G} = \{ \boldsymbol{\mathrm{l}}_{1}, \boldsymbol{\mathrm{l}}_{2}, \ldots, \boldsymbol{\mathrm{l}}_{n} \}$ 配对。这些示范由专家在运动规划者的帮助下收集，以达到中间姿态。每个示范 $\zeta$ 是一系列连续动作 $\bar{\mathcal{A}} = \{ a_{1}, a_{2}, \ldots, a_{t} \}$，与观察 $\mathcal{O} = \{ \tilde{o}_{1}, \tilde{o}_{2}, \dots, \tilde{o}_{t} \}$ 配对。一个动作 $a$ 由 6 自由度的姿态、夹爪的开启状态以及运动规划者是否使用了碰撞避免来达到中间姿态组成：$\boldsymbol{a} = \{ a_{\mathrm{pose}}, a_{\mathrm{open}}, a_{\mathrm{collide}} \}$。一个观察 $\tilde{o}$ 由任意数量摄像头的 RGB-D 图像组成。在模拟实验中，我们使用四个摄像头 $\tilde{\sigma}_{\mathrm{sim}} = \{ \sigma_{\mathrm{front}}, o_{\mathrm{left}}, o_{\mathrm{right}}, o_{\mathrm{wrist}} \}$，而在现实世界实验中，仅使用一个摄像头 $\tilde{o}_{\mathrm{real}} \stackrel{\sim}{=} \{ o_{\mathrm{front}} \}$。

# 3.2 关键帧与体素化

根据James等人的先前工作[14]，我们通过关键帧提取和体素化构建了一个结构化的观察和动作空间。训练我们的智能体直接预测连续动作是低效且噪声较大的。因此，对于每个演示$\zeta$，我们提取一个关键帧动作集合$\left\{ \mathbf { k } _ { 1 } , \mathbf { k } _ { 2 } , \ldots , \mathbf { k } _ { m } \right\} \subset { \mathcal { A } }$，使用一个简单的启发式方法来捕捉动作序列中的瓶颈末端执行器姿态[71]：如果（1）关节速度接近零且（2）夹爪打开状态没有改变，则该动作为关键帧。然后，演示$\zeta$中的每个数据点可以被视为一个“预测下一个（最佳）关键帧动作”的任务[14, 72, 73]。有关该过程的示意图，请参见附录图F。

为了学习以动作为中心的三维表征，我们使用体素网格来表示观察和动作空间。观察体素 $\mathbf { v }$ 是通过三角测量从已知相机外参和内参融合的 RGB-D 观测 $\tilde { o }$ 重建而来的。默认情况下，我们使用一个 $1 \mathrm { { 0 0 ^ { 3 } } }$ 的体素网格，对应于公制尺度下的体积为 $\mathrm { 1 . 0 m ^ { 3 } }$。关键帧动作 $\mathbf { k }$ 被离散化，以便将我们的行为克隆（BC）智能体的训练公式化为“下一个最佳动作”分类任务。平移是指与夹爪手指中心最近的体素。旋转被离散为每个旋转轴 5 度的区间。夹爪打开状态是一个二元值。碰撞也是一个二元值，表示运动规划器是否应该避免体素网格中的所有物体或完全不避开；在这两种碰撞避免模式之间的切换至关重要，因为任务通常涉及接触型（例如，拉开抽屉）和非接触型运动（例如，抓取手柄而不碰撞任何物体）。

# 3.3 PERACT 智能体

PERACT 是一个基于 Transformer 的智能体，它接受体素观察和语言目标 $( \mathbf { v } , \mathbf { l } )$，并输出离散化的平移、旋转和夹持器打开动作。该动作通过运动规划器执行，然后该过程重复，直到达到目标。语言目标 1 使用预训练语言模型进行编码。我们使用 CLIP 的语言编码器，但任何预训练语言模型都可以满足需求。选择 CLIP 为未来的工作开辟了使用与语言对齐的预训练视觉特征的可能性，从而更好地泛化到未见的语义类别和实例。体素观察 $v$ 被分割成大小为 $5 ^ { 3 }$ 的三维块（类似于 ViT 等视觉变换器）。在实现中，这些块是通过具有大小和步幅均为 5 的三维卷积层提取的，然后被展平为一系列体素编码。语言编码经过线性层微调后，附加到体素编码上以形成输入序列。我们还将学习到的位置嵌入添加到序列中，以整合体素和词元的位置。

输入的语言和体素编码序列极其冗长。标准Transformer具有 $\bar { \mathcal { O } } ( n ^ { 2 } )$ 自注意力连接，并且输入为 $( 1 0 0 / 5 ) ^ { 3 } = \mathrm { \bar { 8 } 0 0 0 }$ 补丁，难以适应普通GPU的内存。因此，我们使用Perceiver [1] Transformer。Perceiver是一种潜在空间Transformer，其中不直接对整个输入进行自注意力计算，而是首先计算输入与一组较小的潜在向量之间的交叉注意力（这些向量是随机初始化并经过训练的）。这些潜在向量通过自注意力层进行编码，最终输出时，潜在向量再次与输入进行交叉注意力，以匹配输入大小。参见附录图6，尺寸为 $5 1 2 : \mathbb { R } ^ { 2 0 4 8 \times 5 1 ^ { 2 } }$，但在附录G中我们尝试了不同的潜在尺寸。Perceiver Transformer使用6个自注意力层来编码潜在向量，并从输出交叉注意力层输出补丁编码序列。这些补丁编码通过3D卷积层和三线性上采样进行上采样，以解码64维体素特征。解码器包含一个来自编码器的跳跃连接（如UNet [77]）。然后，每个体素特征用于预测离散化动作 [14]。对于平移，体素特征被重塑为原始体素网格 $( 1 0 0 ^ { 3 } )$，以形成一个3D $\mathcal { Q }$ -函数的动作值。对于旋转、夹具打开和碰撞，这些特征经过最大池化，然后通过线性层解码，形成各自的 $\mathcal { Q }$ -函数。最佳动作 $\tau$ 通过简单地最大化 $\mathcal { Q }$ -函数选择。

$$
\begin{array} { r l r } { \mathcal { T } _ { \mathrm { t r a n s } } = \underset { ( x , y , z ) } { \mathrm { a r g m a x } } \ Q _ { \mathrm { t r a n s } } ( ( x , y , z ) \mid \mathbf { v } , \mathbf { l } ) , } & { \quad } & { \mathcal { T } _ { \mathrm { r o t } } = \underset { ( \psi , \theta , \phi ) } { \mathrm { a r g m a x } } \ Q _ { \mathrm { r o t } } ( ( \psi , \theta , \phi ) \mid \mathbf { v } , \mathbf { l } ) , } \\ { \mathcal { T } _ { \mathrm { o p e n } } = \underset { \omega } { \mathrm { a r g m a x } } \ Q _ { \mathrm { o p e n } } ( \omega \mid \mathbf { v } , \mathbf { l } ) , } & { \quad } & { \mathcal { T } _ { \mathrm { c o l l i d e } } = \underset { \kappa } { \mathrm { a r g m a x } } \ Q _ { \mathrm { c o l l i d e } } ( \kappa \mid \mathbf { v } , \mathbf { l } ) , } \end{array}
$$

其中 $( x , y , z )$ 是网格中的体素位置，$( \psi , \theta , \phi )$ 是欧拉角的离散旋转，$\omega$ 是夹爪的开启状态，$\kappa$ 是碰撞变量。有关 $\mathcal { Q }$ 预测的示例，请参见图 5。

# 3.4 训练细节

PERAcT 通过监督学习，使用来自演示数据集的离散时间输入-行动元组进行训练。这些元组由体素观察、语言目标和关键帧动作组成 $\{ ( \mathbf { v } _ { 1 } , \mathbf { l } _ { 1 } , \mathbf { k } _ { 1 } ) , ( \mathbf { v } _ { 2 } , \mathbf { l } _ { 2 } ^ { - } , \mathbf { k } _ { 2 } ) , \ldots \}$ 。在训练过程中，我们随机采样一个元组，并指导智能体在给定观察和目标 $( \mathbf { v } , \mathbf { l } )$ 的情况下预测关键帧动作 $\mathbf { k }$。对于平移，$Y _ { \mathrm { t r a n s } } : \mathbb { R } ^ { \hat { H } \times \hat { W } \times D }$。旋转也通过每个旋转轴的一次性编码表示，具有 $R$ 旋转区间 $Y _ { \mathrm { r o t } } : \mathbb { R } ^ { ( 3 6 0 / R ) \times 3 }$ （所有实验的 $R = 5$ 度）。类似地，打开和碰撞变量是二进制一次性向量 $Y _ { \mathrm { o p e n } } : \mathbb { R } ^ { 2 }$ ; $Y _ { \mathrm { c o l l i d e } } : \mathbb { R } ^ { 2 }$。智能体通过交叉熵损失进行训练，类似于分类器：

$$
\mathcal { L } _ { \mathrm { t o t a l } } = - \mathbb { E } _ { Y _ { \mathrm { t r a n s } } } [ \log \mathcal { V } _ { \mathrm { t r a n s } } ] - \mathbb { E } _ { Y _ { \mathrm { r o t } } } [ \log \mathcal { V } _ { \mathrm { r o t } } ] - \mathbb { E } _ { Y _ { \mathrm { o p e n } } } [ \log \mathcal { V } _ { \mathrm { o p e n } } ] - \mathbb { E } _ { Y _ { \mathrm { c o l i d e } } } [ \log \mathcal { V } _ { \mathrm { c o l l i d e } } ] ,
$$

其中 $\nu _ { \mathrm { t r a n s } } = \mathrm { s o f t m a x } { \left( \mathcal { Q } _ { \mathrm { t r a n s } } { \left( ( x , y , z ) | \mathbf { v } , \mathbf { l } \right) } \right) }$ , rot = softmax( $\mathcal { Q } _ { \mathrm { r o t } } \big ( \big ( \psi , \theta , \phi \big ) | { \bf v } , { \bf l } \big ) \big )$ , $\begin{array} { r l } { \mathcal { V } _ { \mathrm { o p e n } } } & { { } = } \end{array}$ softmax $\left( \mathcal { Q } _ { \mathrm { o p e n } } ( \omega | \mathbf { v } , \mathbf { l } ) \right)$ , $\mathcal { V } _ { \mathrm { c o l l i d e } } = \mathrm { s o f t m a x } ( \mathcal { Q } _ { \mathrm { c o l l i d e } } ( \kappa | \mathbf { v } , \mathbf { l } ) )$ 。 为了提高鲁棒性，我们还用平移和旋转扰动增强了 $\mathbf { v }$ 和 $\mathbf { k }$ 。 具体细节见附录 E 。 默认情况下，我们使用 $1 0 0 ^ { 3 }$ 的体素网格大小 。 我们通过重放专家演示并离散化动作进行了验证测试，以确保 $1 0 0 ^ { 3 }$ 是执行的足够分辨率 。 智能体在 8 张 NVIDIA V100 GPU 上以 16 的批量大小训练了 16 天（600K 轮次）。 我们使用 LAMB [78] 优化器，以 Perceiver [1] 为参考 。 对于多任务训练，我们只是从数据集中所有任务中抽样输入-动作元组。 为了确保在抽样过程中不会过度代表较长时间跨度的任务，每个批次都包含任务的均匀分布。 即，我们首先均匀抽样一组长度为批量大小的任务，然后为每个抽样任务随机选择一个输入-动作元组。 用此策略，较长时间跨度的任务需要更多的训练步骤来完全覆盖输入-动作对，但所有任务在梯度更新过程中被给予相等的权重。

# 4 结果

我们进行实验以回答以下问题：(1) 与非结构化的图像到动作框架及标准架构（如3D卷积神经网络）相比，PERACT的有效性如何？影响PERACT性能的因素有哪些？(2) 与具有局部感受野的方法相比，变换器的全局感受野是否确实具有优势？(3) PERACT能否在含有噪声数据的真实任务上进行训练？

# 4.1 仿真设置

我们在环境中进行实验。模拟设置在CoppelaSim中，并通过PyRep进行接口。所有实验均使用一台配备平行夹爪的Franka Panda机器人。输入观测数据通过四个RGB-D相机捕获，这些相机分别位于正前方、左肩、右肩和手腕上，如附录图7所示。所有相机均无噪声，分辨率为$128 \times 128$。

语言条件任务。我们在18个RLBench任务上进行训练和评估。有关示例，请参见 peract.github.io，附录A提供了各个任务的详细信息。每个任务包括多个变体，范围从2到60个可能性，例如，在堆积积木任务中，“堆2个红色积木”和“堆4个紫色积木”是两个变体。这些变体在数据生成过程中随机抽样，但在评估期间保持一致以进行一对一比较。一些RLBench任务经过修改，以包含额外的变体，以对多任务和语言基础能力进行压力测试。18个任务总共有249个变体，提取的关键帧数量在2到17之间。每个回合的所有关键帧都有相同的语言目标，该目标基于模板构建（但对于现实世界任务是人工标注的）。请注意，在所有实验中，我们不测试对未见物体的泛化，即我们的训练和测试对象相同。然而，在测试时，智能体必须处理新颖的物体姿态、随机抽样的目标以及随机抽样的场景，这些场景在物体颜色、形状、大小和类别上具有不同的语义实例化。这里的重点是评估一个经过所有任务和变体训练的单一多任务智能体的性能。评估指标。每个多任务智能体在所有18个任务上独立评估。评估的分数为失败得0分或完全成功得100分。没有部分得分。我们报告每个任务的25个评估回合的平均成功率 $( 2 5 \times 1 8 = 4 5 0$ 总回合) ，对于每个任务训练的智能体数量为$n = 1 0 $，每个任务100个示范。在评估期间，智能体会持续采取行动，直到神谕指示任务完成或达到最大25步。

# 4.2 仿真结果

表1报告了在所有18个任务上训练的多任务智能体的成功率。由于训练18个独立智能体的资源限制，我们无法研究单任务智能体。

基准方法。我们通过与两个语言条件基线进行对比，研究我们问题表述的有效性：Image-BC 和 C2FARM-BC。Image-BC 是一种图像到动作的智能体，类似于 BC-Z [12]。按照 BC-Z 的方法，我们使用 FiLM [81] 结合 CLIP [76] 语言特征进行条件化，但视觉编码器输入的是 RGB-D 图像而不仅仅是 RGB。我们还研究了 CNN 和 ViT 视觉编码器。C2FARM-BC 是 James 等人 [14] 提出的一个 3D 全卷积网络，在 RLBench 任务上取得了最先进的成果。与我们的智能体类似，C2FARM-BC 也在体素化空间中检测动作，但它使用一种粗到细的方案在两个级别的体素化中检测动作： $\mathrm { \dot { 3 } 2 ^ { 3 } }$ 体素与 $1 ^ { 3 } \mathrm { m }$ 网格，以及从第一级“放大”后 $3 2 ^ { 3 }$ 体素与 $0 . 1 5 ^ { 3 } \mathrm { m }$ 网格。注意，在最细的级别，C2FARM-BC 的分辨率 $\left( 0 . 4 7 \mathrm { c m } \right)$ 高于 PERACT（1cm）。我们使用与 James 等人 [14] 相同的 3D ConvNet 架构，但不是用 RL 进行训练，而是采用交叉熵损失进行 BC（来自第 3.4 节）。我们在瓶颈处用 CLIP [76] 语言特征进行条件化，类似于 LingUNets [82, 16] 的方法。

多任务性能。表1比较了Image-BC和C2FARM-BC相对于PERACT的性能。在示例不足的情况下，Image-BC在大多数任务上的性能接近于零。Image-BC在单视角观察中处于劣势，必须从头学习手眼协调。相比之下，PERACT的体素基础表述自然允许整合多视角观察，学习6自由度的动作表示，以及在3D中进行数据增强，这些在基于图像的方法中都是不易实现的。C2FARM-BC是最具竞争力的基线，但由于粗到细的方案和部分由于仅使用卷积的架构，其感受野有限。在表1的36个评估中，PERACT在25项评估中超过了C2FARM-BC，平均提升分别为$\mathbf { 1 . 3 3 \times }$（10个示例）和$\mathbf { 2 . 8 3 \times }$（100个示例）。对于某些任务而言，C2FARM-BC在 rohkem示例下的表现实际上更差，可能是由于容量不足。由于额外的训练示例包括优化的额外任务变体，它们可能会最终影响性能。

<table><tr><td></td><td colspan="2">open drawer</td><td colspan="2">slide block</td><td colspan="2">sweep to dustpan</td><td colspan="2">meat off grill</td><td colspan="2">turn tap</td><td colspan="2">put in drawer</td><td colspan="2">close jar</td><td colspan="2">drag stick</td><td colspan="2">stack blocks</td></tr><tr><td>Method</td><td>10</td><td>100</td><td>10</td><td>100</td><td>10</td><td>100</td><td>10</td><td>100</td><td>10</td><td>100</td><td>10</td><td>100</td><td>10</td><td>100</td><td>10</td><td>100</td><td>10</td><td>100</td></tr><tr><td>Image-BC (CNN)</td><td>4</td><td>4</td><td>4</td><td>0</td><td>0</td><td>0</td><td></td><td>0</td><td>20</td><td>8</td><td>0</td><td>8</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Image-BC (ViT</td><td>16</td><td>0</td><td>8</td><td>0</td><td></td><td>0</td><td></td><td></td><td>24</td><td>16</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>C2FARM-BC</td><td>28</td><td>20</td><td>12</td><td>16</td><td></td><td>0</td><td></td><td></td><td></td><td>68</td><td>12</td><td>4</td><td>28</td><td>24</td><td>72</td><td>24</td><td>4</td><td>0</td></tr><tr><td>PERACT (w/o Lang)</td><td>20</td><td>28</td><td>8</td><td>12</td><td>20</td><td>16</td><td>40 40</td><td>20 48</td><td>60 36</td><td>60</td><td>16</td><td>16</td><td>16</td><td>12</td><td>48</td><td>60</td><td>0</td><td>0</td></tr><tr><td>PERACT</td><td>68</td><td>80</td><td>32</td><td>72</td><td>72</td><td>56</td><td>68</td><td>84</td><td>72</td><td>80</td><td>16</td><td>68</td><td>32</td><td>60</td><td>36</td><td>68</td><td>12</td><td>36</td></tr><tr><td rowspan="3"></td><td colspan="2">screw</td><td colspan="2">put in</td><td colspan="2">place</td><td colspan="2">put in</td><td colspan="2">sort</td><td colspan="2">push</td><td colspan="2">insert</td><td colspan="2">stack</td><td colspan="2">place</td></tr><tr><td>bulb</td><td></td><td>safe</td><td></td><td>wine</td><td></td><td>cupboard</td><td></td><td>shape</td><td></td><td>buttons</td><td></td><td>peg</td><td></td><td>cups</td><td></td><td>cups</td><td></td></tr><tr><td>10</td><td>100</td><td>10</td><td>100</td><td>10</td><td>100</td><td>10</td><td>100 0</td><td>10</td><td>100</td><td>10</td><td>100</td><td>10</td><td>100</td><td>10</td><td>100</td><td>10</td><td>100</td></tr><tr><td>Image-BC (CNN)</td><td>0</td><td>0</td><td>0</td><td>4</td><td>0 4</td><td>0</td><td>0</td><td>0</td><td>0 0</td><td>0</td><td>4</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Image-BC (ViT</td><td>0</td><td>0</td><td>0</td><td>0</td><td></td><td>0</td><td></td><td></td><td></td><td>0</td><td>16</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>C2FARM-BC</td><td>12</td><td>8</td><td>0</td><td>12</td><td>36</td><td>8</td><td></td><td></td><td></td><td>8</td><td>88</td><td>72</td><td>0</td><td>4</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>PERACT (w/o Lang)</td><td>0</td><td>24</td><td>8</td><td>20</td><td>8</td><td>20</td><td>0</td><td>0</td><td>0</td><td>0</td><td>60</td><td>68</td><td>4</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>PERACT</td><td>28</td><td>24</td><td>16</td><td>44</td><td>20</td><td>12</td><td>0</td><td>16</td><td>16</td><td>20</td><td>56</td><td>48</td><td>4</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></table>

Table 1. Multi-Task Test Results. Success rates (mean $\%$ ) of various multi-task agents tasks trained with either 10 or 100 demonstrations per C2FARM-BC [14], the most competitive baseline, with an average improvement of $1 . 3 3 \times$ with 10 demos and $2 . 8 3 \times$ with 100 demos.

通常情况下，对于开放抽屉等变化有限的任务（3种变化），10次演示足以让PERACT实现超过65%的成功率。但是，对于变化更多的任务，如堆叠积木（60种变化），则需要大大更多的数据，有时甚至需要涵盖所有可能的概念，例如“蓝绿色积木”，这些概念可能在训练数据中没有出现。请参阅补充视频中的模拟推演，以了解这些评估的复杂性。对于三个任务：插入 peg、堆叠杯子和放置杯子，所有智能体的成功率接近零。这些任务的精度要求非常高，偏差几厘米或几度可能导致无法恢复的失败。但在附录H中，我们发现专门针对这三个任务训练的单一任务智能体会稍微缓解这个问题。消融实验。表1报告了不使用语言条件的PERACT，说明该智能体没有任何语言目标，因此对底层任务没有了解，表现为偶然性。此外，我们还在图3中报告了开放抽屉任务的其他消融结果。总结这些结果： (1) 跳跃连接有助于智能体稍快地训练， (2) Perceiver Transformer 对于利用全局感受野实现良好性能至关重要， (3) 提取良好的关键帧动作对于监督训练非常重要，因为随机选择或固定间隔的关键帧会导致零性能。

![](images/3.jpg)  
Figure 3. Ablation Experiments. Success rate of PERACT after ablating key components.

敏感性分析。在附录 G 中，我们研究了影响 PERACT 性能的因素：感知器潜变量的数量、体素化分辨率和数据增强。我们发现，更多的潜向量通常提高了智能体对更多任务建模的能力，但对于简单的短时间任务，较少的潜向量即可满足需求。同样，对于不同的体素化分辨率，一些任务可以通过像 $3 2 ^ { 3 }$ 这样的粗体素网格解决，但某些高精度任务需要完整的 $1 0 0 ^ { 3 }$ 网格。最后，数据增强中的旋转扰动通常有助于提高鲁棒性，主要是通过让智能体接触到更多物体的旋转变化。

# 4.3 全局感受野与局部感受野

为了进一步研究我们Transformer智能体的全局感受野，我们在开启抽屉任务上进行了额外实验。开启抽屉任务有三个变体：“打开上层抽屉”、“打开中层抽屉”和“打开下层抽屉”，由于感受野有限，很难区分抽屉把手，它们在视觉上都是相同的。图4报告了在100个示例上训练的PERACT和C2FARM-BC智能体。尽管开启抽屉任务可以用更少的示例解决，但在这里我们想确保数据不足不是一个问题。我们包含了几个不同体素化方案的C2FARM-BC版本。例如， [16, 16]表示两个层次的$1 6 ^ { 3 }$体素网格，分别在$\mathrm { 1 m ^ { 3 } }$和$\mathrm { 0 . 1 5 m ^ { 3 } }$。而[64]表示一个单层的$6 4 ^ { 3 }$体素网格，没有粗到细的方案。PERAcT是唯一一个成功率超过$> 7 0 \%$的智能体，而所有C2FARM-BC版本的表现则与随机几率相当，约为$\mathrm { \sim 3 3 \% }$，这表明Transformer的全局感受野对解决该任务至关重要。

![](images/4.jpg)  
Figure 4. Global vs. Local Receptive Field Experiments. Success rates of PERACT against various C2FARM-BC [14] baselines

![](images/5.jpg)  
Figure 5. Q-Prediction Examples: Qualitative examples of translation $\mathcal { Q }$ -Predictions from PERACT along with expert actions, highlighted

# 4.4 真实机器人结果

我们还通过在Franka Emika Panda机器人上进行真实机器人实验验证了我们的结果。有关设置的详细信息，请参见附录D。在没有任何仿真到现实的迁移或预训练的情况下，我们从头开始训练了一个多任务PERACT智能体，在仅53个演示中完成了7个任务（包含18种独特变体）。请参见补充视频以获取展示任务多样性和对场景变化的鲁棒性的定性结果。表2报告了小规模评估的成功率。与仿真结果类似，我们发现PERACT能够在简单的短期任务（如从少量演示中按压洗手液）上实现超过65%的成功率。最常见的失败涉及预测错误的夹具打开动作，这通常导致智能体进入未见状态。未来的工作可以通过使用HG-DAgger风格的方法来纠正智能体，从而解决这个问题。其他问题还包括智能体利用数据集中的偏差，类似于以往的工作。这可以通过扩大专家数据并增加更多多样化的任务和任务变体来解决。

<table><tr><td>Task</td><td># Train</td><td># Test</td><td>Succ. %</td></tr><tr><td>Press Handsan</td><td>5</td><td>10</td><td>90</td></tr><tr><td>Put Marker</td><td>8</td><td>10</td><td>70</td></tr><tr><td>Place Food</td><td>8</td><td>10</td><td>60</td></tr><tr><td>Put in Drawer</td><td>8</td><td>10</td><td>40</td></tr><tr><td>Hit Ball</td><td>8</td><td>10</td><td>60</td></tr><tr><td>Stack Blocks</td><td>10</td><td>10</td><td>40</td></tr><tr><td>Sweep Beans</td><td>8</td><td>5</td><td>20</td></tr></table>

Table 2. Success rates (mean $\%$ ) of a multitask model trained an evaluated 7 realworld tasks (see Figure 1).

# 5 限制与结论

我们提出了PERACT，这是一种基于Transformer的多任务智能体，用于6自由度操作。我们在模拟和现实任务中的实验表明，正确的问题表述，即检测体素动作，对数据效率和鲁棒性有显著影响。虽然PERACT相当强大，但将其扩展到灵巧的连续控制仍然是一个挑战。PERACT依赖基于采样的运动规划器来执行离散化动作，并且不易扩展到多指手等N自由度执行器。有关PERACT局限性的详细讨论，请参见附录L。但总体而言，我们对通过关注多样化而非狭窄的多任务数据来扩大机器人学习与Transformers的结合感到兴奋。

# 致谢

我们感谢 Selest Nashef 和 Karthik Desingh 在华盛顿大学的 Franka 设备设置方面的帮助。感谢 Stephen James 在 RLBench 和 ARM 问题上的支持。我们还要感谢 Valts Blukis、Zoey Chen、Markus Grotz、Aaron Walsman 和 Kevin Zakka 对初稿的反馈。同时感谢 Shikhar Bahl 进行的初步讨论。本研究部分由海军研究办公室（ONR）资助，资助编号为 #1140209-405780。Mohit Shridhar 获得 NVIDIA 研究生奖学金支持，并在整个项目期间担任 NVIDIA 的兼职实习生。

参考文献 [1] A. Jaegle, S. Borgeaud, J.-B. Alayrac, C. Doersch, C. Ionescu, D. Ding, S. Koppula, D. Zoran. arXiv 预印本 arXiv:2107.14795, 2021. [2] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, 和 I. Polosukhin. 注意力机制是你所需的一切。载于神经信息处理系统进展 (NeuRIPS), 2017. [3] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell 等. 语言模型是少量学习者。神经信息处理系统 (NeurIPS), 2020. [4] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly 等. 一幅图像价值16x16个单词：用于大规模图像识别的变换器。载于国际学习表征会议 (ICLR), 2020. [5] J. Jumper, R. Evans, A. Pritzel, T. Green, M. Figurnov, O. Ronneberger, K. Tunyasuvunakool, R. Bates, A. Zídek, A. Potapenko 等. 通过 Alphafold 进行高精度蛋白质结构预测。自然, 2021. [6] O. Vinyals, I. Babuschkin, J. Chung, M. Mathieu, M. Jaderberg, W. M. Czarnecki, A. Dudzik, A. Huang, P. Georgiev, R. Powell 等. Alphastar：在星际争霸 II 中掌握实时策略。DeepMind 博客, 2019. [7] T. Chen, S. Saxena, L. Li, D. J. Fleet, 和 G. Hinton. Pix2seq：用于目标检测的语言建模框架。arXiv 预印本 arXiv:2109.10852, 2021. [8] L. Chen, K. Lu, A. Rajeswaran, K. Lee, A. Grover, M. Laskin, P. Abbeel, A. Srinivas, 和 I. Mordatch. 决策变换器：通过序列建模的强化学习。载于神经信息处理系统 (NeurIPS), 2021. [9] S. Reed, K. Zolna, E. Parisotto, S. G. Colmenarejo, A. Novikov, G. Barth-Maron, M. Gimenez, Y. Sulsky, J. Kay, J. T. Springenberg 等. 一种通用智能体。arXiv 预印本 arXiv:2205.06175, 2022. [10] J. Devlin, M. W. Chang, K. Lee, 和 K. Toutanova. BERT：用于语言理解的深度双向变换器预训练。载于北美计算语言学协会会议 (NAACL), 2018. [11] A. Jaegle, F. Gimeno, A. Brock, O. Vinyals, A. Zisserman, 和 J. Carreira. Perceiver：通用感知与迭代注意力。载于国际机器学习会议 (ICML), 2021. [12] E. Jang, A. Irpan, M. Khansari, D. Kappler, F. Ebert, C. Lynch, S. Levine, 和 C. Finn. Bc-z：借助机器人模仿学习进行零样本任务泛化。载于机器人学习会议 (CoRL), 2021. [13] M. Ahn, A. Brohan, N. Brown, Y. Chebotar, O. Cortes, B. David, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog 等. 照做我所能，而不是我所说：将语言与机器人功能对接。arXiv 预印本 arXiv:2204.01691, 2022. [14] S. James, K. Wada, T. Laidlow, 和 A. J. Davison. 粗到细 Q-注意力：通过离散化高效学习视觉机器人操作。载于计算机视觉与模式识别会议 (CVPR), 2022. [15] S. James, Z. Ma, D. R. Arrojo, 和 A. J. Davison. RLBench：机器人学习基准与学习环境。IEEE 机器人与自动化快报 (RA-L), 2020. [16] M. Shridhar, L. Manueli, 和 D. Fox. Cliport：机器人操作的“什么”和“在哪里”路径。载于机器人学习会议 (CoRL), 2021. [17] A. Zeng, P. Florence, J. Tompson, S. Welker, J. Chien, M. Attarian, T. Armstrong, I. Krasin, D. Duong, V. Sindhwani, 和 J. Lee. 运输网络：为机器人操作重新排列视觉世界。载于机器人学习会议 (CoRL), 2020. [18] J. J. Gibson. 生态视觉感知方法：经典版。心理学出版社, 2014. [19] R. A. Brooks. 机器人新方法。科学, 1991. [20] K. He, G. Gkioxari, P. Dolár, 和 R. Girshick. Mask R-CNN。载于计算机视觉与模式识别会议 (CVPR), 2017. [21] Y. Xiang, T. Schmidt, V. Narayanan, 和 D. Fox. PoseCNN：在杂乱场景中进行6D物体姿态估计的卷积神经网络。载于机器人：科学与系统会议 (RSS), 2018. [22] M. Zhu, K. G. Derpanis, Y. Yang, S. Brahmbhatt, M. Zhang, C. Phillips, M. Lecce, 和 K. Danilidis. 单幅图像3D物体检测与抓取姿态估计。载于2014 IEEE国际机器人与自动化会议 (ICRA), 2014. [23] A. Zeng, K.-T. Yu, S. Song, D. Suo, E. Walker, A. Rodriguez, 和 J. Xiao. 亚马逊拣选挑战中的多视角自监督深度学习进行6D姿态估计。载于2017 IEEE国际机器人与自动化会议 (ICRA), 2017. [24] X. Deng, Y. Xiang, A. Mousavian, C. Eppner, T. Bretl, 和 D. Fox. 自监督6D物体姿态估计用于机器人操作。载于2020 IEEE国际机器人与自动化会议 (ICRA), 2020. [25] C. Xie, Y. Xiang, A. Mousavian, 和 D. Fox. 两种模式的最佳结合：分别利用RGB和深度进行未知物体实例分割。载于机器人学习会议 (CoRL), 2020. [26] A. Zeng, S. Song, K.-T. Yu, E. Donlon, F. R. Hogan, M. Bauza, D. Ma, O. Taylor, M. Liu, E. Romo 等. 在杂乱环境中进行新物体的机器人拾取和放置，使用多功能抓取与跨域图像匹配。国际机器人研究期刊 (IJRR), 2019. [27] E. Stengel-Eskin, A. Hundt, Z. He, A. Murali, N. Gopalan, M. Gombolay, 和 G. Hager. 使用自然语言指令指导多步骤重排任务。载于机器人学习会议 (CoRL), 2022. [28] D. Kalashnikov, A. Irpan, P. Pastor, J. Ibarz, A. Herzog, E. Jang, D. Quillen, E. Holly, M. Kalakrishnan, V. Vanhoucke 等. QT-OPT：基于视觉的机器人操作可扩展深度强化学习。载于机器人学习会议 (CoRL), 2018. [29] Y. Wu, W. Yan, T. Kurutach, L. Pinto, 和 P. Abbeel. 学习在没有示范的情况下操作可变形物体。载于机器人：科学与系统会议 (RSS), 2020. [30] S. Levine, C. Finn, T. Darrell, 和 P. Abbeel. 深度视觉运动策略的端到端训练。机器学习研究期刊, 17(1):1334-1373, 2016. [31] C. Finn 和 S. Levine. 深度视觉前瞻规划机器人运动。载于2017 IEEE国际机器人与自动化会议 (ICRA), 2017. [32] S. Song, A. Zeng, J. Lee, 和 T. Funkhouser. 在真实环境中的抓取：从低成本示范中学习6D闭环抓取。IEEE 机器人与自动化快报 (RA-L), 2020. [33] A. Murali, A. Mousavian, C. Eppner, C. Paxton, 和 D. Fox. 在杂乱环境中针对目标驱动的物体操作进行6D抓取。载于国际机器人与自动化会议 (ICRA), 2020. [34] A. Mousavian, C. Eppner, 和 D. Fox. 6D抓取网络：用于物体操作的变分抓取生成。载于国际计算机视觉会议 (ICCV), 2019. [35] Z. Xu, H. Zhanpeng, 和 S. Song. UMPNet：用于关节物体的通用操作策略网络。机器人与自动化快报 (RA-L), 2022. [36] S. Agrawal, Y. Li, J.-S. Liu, S. K. Feiner, 和 S. Song. 场景编辑作为遥操作：6D装配的案例研究。arXiv 预印本 arXiv:2110.04450, 2021. [37] A. Simeonov, Y. Du, A. Tagliasacchi, J. B. Tenenbaum, A. Rodriguez, P. Agrawal, 和 V. Sitzmann. 神经描述子场：用于操作的SE(3)-协变物体表示。arXiv 预印本 arXiv:2112.05124, 2021. [38] S. Nair, A. Rajeswaran, V. Kumar, C. Finn, 和 A. Gupta. R3M：一种用于机器人操作的通用视觉表示。arXiv 预印本 arXiv:2203.12601, 2022. [39] W. Yuan, C. Paxton, K. Desingh, 和 D. Fox. SORNet：用于序列操作的空间对象中心表示。载于机器人学习会议 (CoRL), PMLR, 2021. [40] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, 和 V. Stoyanov. RoBERTa：一种稳健优化的BERT预训练方法。arXiv 预印本 arXiv:1907.11692, 2019. [41] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, 和 B. Guo. Swin 变换器：使用移动窗口的分层视觉变换器。载于国际计算机视觉会议 (ICCV), 2021. [42] M. Janner, Q. Li, 和 S. Levine. 离线强化学习作为一个巨大序列建模问题。神经信息处理系统 (NeurIPS), 2021. [43] K.-H. Lee, O. Nachum, M. Yang, L. Lee, D. Freeman, W. Xu, S. Guadarrama, I. Fischer, E. Jang, H. Michalewski 等. 多游戏决策变换器。arXiv 预印本 arXiv:2205.15241, 2022. [44] H. M. Clever, A. Handa, H. Mazhar, K. Parker, O. Shapira, Q. Wan, Y. Narang, I. Akinola, M. Cakmak, 和 D. Fox. 助益遥控：利用变换器收集机器人任务示范。arXiv 预印本 arXiv:2112.05129, 2021. [45] R. Yang, M. Zhang, N. Hansen, H. Xu, 和 X. Wang. 学习基于视觉的四足机器人运动控制的端到端方法，使用跨模态变换器。arXiv 预印本 arXiv:2107.03996, 2021. [46] D. S. Chaplot, D. Pathak, 和 J. Malik. 使用变换器的可微空间规划。载于国际机器学习会议 (ICML), 2021. [47] J. J. Johnson, L. Li, A. H. Qureshi, 和 M. C. Yip. 动作规划变换器：一个模型规划所有。arXiv 预印本 arXiv:2106.02791, 2021. [48] S. Dasari 和 A. Gupta. 用于单次视觉模仿的变换器。arXiv 预印本 arXiv:2011.05970, 2020. [49] H. Kim, Y. Ohmura, 和 Y. Kuniyoshi. 基于变换器的深度模仿学习用于双臂机器人操作。载于智能机器人与系统国际会议 (IROS), IEEE, 2021. [50] A. Gupta, L. Fan, S. Ganguli, 和 L. Fei-Fei. Metamorph：使用变换器学习通用控制器。arXiv 预印本 arXiv:2203.11931, 2022. [51] W. Liu, C. Paxton, T. Hermans, 和 D. Fox. 结构变换器：学习空间结构以进行语言引导的语义重排新物体。载于国际机器人与自动化会议 (ICRA), 2022. [52] Y. Han, R. Batra, N. Boyd, T. Zhao, Y. She, S. Hutchinson, 和 Y. Zhao. 通过变换器学习可变形物体的可推广视觉触觉机器人抓取策略。arXiv 预印本 arXiv:2112.06374, 2021. [53] T. Yu, D. Quillen, Z. He, R. Julian, K. Hausman, C. Finn, 和 S. Levine. Meta-World：多任务和元强化学习的基准和评估。载于机器人学习会议 (CoRL), 2020. [54] M. Shridhar 和 D. Hsu. 针对人机交互的指示表达的交互式视觉归属。载于机器人：科学与系统会议 (RSS), 2018. [55] C. Matuszek, L. Bo, L. Zettlemoyer, 和 D. Fox. 从无脚本的指示手势与语言中学习人机交互。载于人工智能协会会议 (AAAI)，第28卷，2014.

[56] M. Bollini, S. Tellex, T. Thompson, N. Roy, 和 D. Rus. 使用烹饪机器人解释和执行食谱. 在 《实验机器人学》, 第 481-495 页. 施普林格, 2013. [57] D. K. Misra, J. Sung, K. Lee, 和 A. Saxena. 告诉我，戴夫：自然语言到操作指令的上下文敏感映射. 《国际机器人研究杂志》 (IJRR), 2016. [58] Y. Bisk, D. Yuret, 和 D. Marcu. 与机器人的自然语言交互. 在《北美计算语言学协会会议》(NAACL), 2016. [] J. Thao, S. Zhang, R. J. Mooey, 和 P.Stoe. 通过人机对话学习和解析自然语言命令. 在《第二十四届国际人工智能联合会议》(IJCAI), 2015. [60] J. Hatori, Y. Kikuchi, S. Kobayashi, K. Takahashi, Y. Tsuboi, Y. Unno, W. Ko, 和 J. Tan. 通过无约束的口语指令交互式地选择现实世界中的对象. 在《国际机器人与自动化会议》(ICRA), 2018. [61] Y. Chen, R. Xu, Y. Lin, 和 P. A. Vela. 一种基于自然语言命令的抓取检测联合网络. arXiv:2104.00492 [cs], 2021年4月. [62] V. Blukis, R. A. Knepper, 和 Y. Artzi. 基于少量样本的对象定位，将自然语言指令映射到机器人控制. 在《机器人学习会议》(CoRL), 2020. [63] C. Paxton, Y. Bisk, J. Thomason, A. Byravan, 和 D. Fox. 前瞻性：通过预测未来从语言中生成可解释的计划. 在《国际机器人与自动化会议》(ICRA), 2019. [64] S. Tellex, T. Kollar, S. Dickerson, M. Walter, A. Banerjee, S. Teller, 和 N. Roy. 理解用于机器人导航和移动操作的自然语言命令. 在《美国人工智能协会会议》(AAAI), 2011. [65] T. Nguyen, N. Gopalan, R. Patel, M. Corsaro, E. Pavlick, 和 S. Tellex. 使用上下文自然语言查询进行机器人对象检索. arXiv预印本 arXiv:2006.13253, 2020. [66] Y. Bisk, A. Holtzman, J. Thomason, J. Andreas, Y. Bengio, J. Chai, M. Lapata, A. Lazaridou, J. May, A. Nisnevich, N. Pinto, 和 J. Turian. 经验基础的语言构建. 在《自然语言处理中的实证方法会议》(EMNLP), 2020. [67] S. Nair, E. Mitchell, K. Chen, S. Savarese, C. Finn, 等. 从离线数据和众包注释中学习语言条件下的机器人行为. 在《机器人学习会议》(CoRL), 2022. [68] O. Mees, L. Hermann, 和 W. Burgard. 在非结构化数据上语言条件下的机器人模仿学习中什么是重要的. 《IEEE机器人与自动化快报》(RA-L), 2022. [69] C. Lynch 和 P. Sermanet. 在游戏中定位语言. 在《计算语言学协会会议》(ACL), 2022. [70] O. Mees, L. Hermann, E. Rosete-Beas, 和 W. Burgard. Calvin：一个用于语言条件策略学习的基准，以处理长时间机器人操作任务. arXiv预印本 arXiv:2112.03227, 2021. [71] E. Johns. 粗到细模仿学习：基于单个演示的机器人操作. 在《国际机器人与自动化会议》(ICRA), 2021. [72] S. James 和 A. J. Davison. Q-注意力：为基于视觉的机器人操作启用高效学习. 《IEEE机器人与自动化快报》(RA-L), 7(2):1612-1619, 2022. [73] S. Liu, S. James, A. J. Davison, 和 E. Johns. Auto-lambda：解耦动态任务关系. 《机器学习研究交易》, 2022. [74] H. Moravec. 通过立体视觉和三维证据网格进行机器人空间感知. 《感知》，1996. [75] Y. Roth-Tabak 和 R. Jain. 使用深度信息构建环境模型. 《计算机》，22(6):85-90, 1989. [76] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, 和 I. Sutskever. 从自然语言监督中学习可转移的视觉模型. arXiv:2103.00020, 2021. [77] O. Ronneberger, P. Fischer, 和 T. Brox. U-net：生物医学图像分割的卷积网络. 在《医学图像计算与计算机辅助干预国际会议》，第 234-241 页. 施普林格，2015. [78] Yu, J. Li. Ri J Hse S. S.Bhp X., J., 和 C.-J. Hsieh. 深度学习的大批量优化：76分钟内训练BERT. arXiv预印本 arXiv:1904.00962, 2019. [79] E. Rohmer, S. P. N. Singh, 和 M. Freese. V-rep：一种多功能和可扩展的机器人仿真框架. 在《智能机器人与系统国际会议》(IROS), 2013. [80] S. James, M. Freese, 和 A. J. Davison. Pyrep：将 V-rep 引入深度机器人学习. arXiv预印本 arXiv:1906.11176, 2019. [81] E. Perez, F. Strub, H. De Vries, V. Dumoulin, 和 A. Courvill. Film：具有通用条件层的视觉推理. 在《美国人工智能协会会议》，2018. [82] D. Misra, A. Bennett, V. Blukis, E. Niklasson, M. Shatkhin, 和 Y. Artzi. 在三维环境中映射指令到动作，结合视觉目标预测. 在2019年《自然语言处理中的实证方法会议》(EMNLP), 2018. [83] Z. Mandi, P. Abbeel, 和 S. James. 微调与元强化学习的有效性. arXiv预印本 arXiv:2206.03271, 2022. [84] S. Sodhani, A. Zhang, 和 J. Pineau. 基于上下文表示的多任务强化学习. 在 M. Meila 和 T. Zhang 编辑的《国际机器学习会议》(ICML), 2021. [85] M. Shridhar, X. Yuan, M.-A. Côté, Y. Bisk, A. Trischler, 和 M. Hausknecht. ALFWorld：将文本与体现环境对齐以实现互动学习. 在《国际表示学习会议》(ICLR), 2021. [86] A. Zeng, A. Wong, S. Welker, K. Choromanski, F. Tombari, A. Purohit, M. Ryoo, V. Sindhwani, J. Lee, V. Vanhoucke, 等. 苏格拉底模型：组合零-shot多模态推理与语言. arXiv预印本 arXiv:2204.00598, 2022. [87] W. Huang, P. Abbeel, D. Pathak, 和 I. Mordatch. 语言模型作为零-shot计划者：为体现智能体提取可执行知识. arXiv预印本 arXiv:2201.07207, 2022. [88] L. P. Kaelbling 和 T. Lozano-Pérez. 在信念空间中的集成任务和运动规划. 《国际机器人研究杂志》，2013. [89] C. R. Garrett, R. Chitnis, R. Holladay, B. Kim, T. Silver, L. P. Kaelbling, 和 T. Lozano-Pérez. 集成任务和运动规划. 年度控制、机器人和自主系统回顾, 2021. [90] G. Konidaris, L. P. Kaelbling, 和 T. Lozano-Perez. 从技能到符号：学习用于抽象高级规划的符号表示. 人工智能研究杂志, 2018. [91] J. Mao, Y. Xue, M. Niu, H. Bai, J. Feng, X. Liang, H. Xu, 和 C. Xu. 用于三维对象检测的体素变压器. 在《国际计算机视觉会议》(ICCV), 2021. [92] C. He, R. Li, S. Li, 和 L. Zhang. 体素集变压器：一种基于点云的三维对象检测集合到集合的方法. 在《计算机视觉与模式识别会议》(CVPR), 第 8417-8427 页, 2022. [93] K. Zheng, R. Chitnis, Y. Sung, G. Konidaris, 和 S. Tellex. 朝向最佳关联对象搜索. 在《国际机器人与自动化会议》(ICRA), 2022. [94] V. Blukis, C. Paxton, D. Fox, A. Garg, 和 Y. Artzi. 持久空间语义表示. 在《机器人学习会议》(CoRL), 第 706-717 页. PMLR, 2022. [95] R. Corona, S. Zhu, D. Klein, 和 T. Darrell. 基于体素的信息语言定位. 在《计算语言学协会会议》(ACL), 2022. [96] V. Sitzmann, J. Thies, F. Heide, M. NieBner, G. Wetzstein, 和 M. Zollhofer. Deepvoxels：学习持久三维特征嵌入. 在《计算机视觉与模式识别会议》(CVPR), 2019. [97] T. Müller, A. Evans, C. Schied, 和 A. Keller. 具有多分辨率哈希编码的即时神经图形原语. 《ACM图形交易》(ToG), 2022. [98] Sara Fridovich-Keil 和 Alex Yu, M. Tancik, Q. Chen, B. Recht, 和 A. Kanazawa. Plenoxels：没有神经网络的辐射场. 在 CVPR, 2022. [99] S. Lal, M. Prabhudesai, I. Mediratta, A. W. Harley, 和 K. Fragkiadaki. Coconets：连续对比三维场景表示. 在《IEEE/CVF计算机视觉与模式识别会议》的论文中, 2021. [100] H.-Y. F. Tung, Z. Xian, M. Prabhudesai, S. Lal, 和 K. Fragkiadaki. 3D-oes：视角不变的对象分解环境模拟器. arXiv预印本 arXiv:2011.06464, 2020. [101] M. NieBner, M. Zollhöfer, S. Izadi, 和 M. Stamminger. 使用体素哈希进行实时三维重建. 《ACM图形交易》(ToG), 2013. [102] Y. Tay, M. Dehghani, D. Bahri, 和 D. Metzler. 高效变压器：综述. 《ACM计算调查》(CSUR), 2020. [103] A. Goyal, A. R. Didolkar, A. Lamb, K. Badola, N. R. Ke, N. Rahaman, J. Binas, C. Blundell, M. C. Mozer, 和 Y. Bengio. 通过共享的全球工作空间协调神经模块. 在《国际表示学习会议》(ICLR), 2021. [104] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, 和 S. Zagoruyko. 使用变压器的端到端对象检测. 在《欧洲计算机视觉会议》(ECCV), 2020. [105] F. Locatello, D. Weissenborn, T. Unterthiner, A. Mahendran, G. Heigold, J. Uszkoreit, A. Dosovitskiy, 和 T. Kipf. 使用插槽注意力的面向对象学习. 《神经信息处理系统》(NeurIPs), 2020. [106] R. Ranftl, A. Bochkovskiy, 和 V. Koltun. 用于密集预测的视觉变压器. 在《国际计算机视觉会议》(ICCV), 第 12179-12188 页, 2021. [107] D. P. Kingma 和 J. Ba. Adam：一种随机优化方法. arXiv预印本 arXiv:1412.6980, 2014. [108] S. James 和 P. Abbeel. 粗到细的Q-注意力与学习的路径排序. arXiv预印本 arXiv:2204.01571, 2022. [109] A. Zeng, S. Song, J. Lee, A. Rodriguez, 和 T. Funkhouser. Tossingbot：学习用残余物理扔任意物体. 《IEEE机器人学报》(T-RO), 2020. [110] A. Kamath, M. Singh, Y. LeCun, I. Misra, G. Synnaeve, 和 N. Carion. MDetr调节的检测以实现端到端多模态理解. arXiv预印本 arXiv:2104.12763, 2021. [] A. Birhane, V. U. Prabhu, 和 E. Kahembwe. 多模态数据集：厌女、色情和恶性刻板印象. arXiv预印本 arXiv:2110.01963, 2021. [112] E. M. Bender, T. Gebru, A. McMillan-Major, 和 S. Shmitchell. 随机鹦鹉的危险：语言模型能否过大？ 在2021年ACM公正性、责任和透明度会议，第610-623页，2021。 [113] Y. LeCun, S. Chopra, R. Hadsell, M. Ranzato, 和 F. Huang. 关于基于能量的学习的教程. 预测结构化数据, 1(0), 2006. [114] P. Florence, C. Lynch, A. Zeng, O. A. Ramirez, A. Wahid, L. Downs, A. Wong, J. Lee, I. Mordatch, 和 J. Tompson. 隐式行为克隆. 在《机器人学习会议》(CoRL), 2022.

# 任务详情

Table 3. Language-Conditioned Tasks in RLBench [15].   

<table><tr><td>Task</td><td>Variation Type</td><td># of Variations</td><td>Avg. Keyframes</td><td>Language Template</td></tr><tr><td>open drawer</td><td>placement</td><td>3</td><td>3.0</td><td>&quot;open the — drawer&quot;</td></tr><tr><td>slide block</td><td>color</td><td>4</td><td>4.7</td><td>&quot;slide the block to — target&quot;</td></tr><tr><td>sweep to dustpan</td><td>size</td><td>2</td><td>4.6</td><td>&quot;sweep dirt to the — dustpan&quot;</td></tr><tr><td>meat off grill</td><td>category</td><td>2</td><td>5.0</td><td>&quot;take the — off the grill&quot;</td></tr><tr><td>turn tap</td><td>placement</td><td>2</td><td>2.0</td><td>&quot;turn — tap&quot;</td></tr><tr><td>put in drawer</td><td>placement</td><td>3</td><td>12.0</td><td>&quot;put the item in the — drawer&quot;</td></tr><tr><td>close jar</td><td>color</td><td>20</td><td>6.0</td><td>&quot;close the _— jar&quot;</td></tr><tr><td>drag stick</td><td>color</td><td>20</td><td>6.0</td><td>&quot;use the stick to drag the cube onto the — target&quot;</td></tr><tr><td>stack blocks</td><td>color, count</td><td>60</td><td>14.6</td><td>&quot;stack — — blocks&quot;</td></tr><tr><td>screw bulb</td><td>color</td><td>20</td><td>7.0</td><td>&quot;screw in the — light bulb&quot;</td></tr><tr><td>put in safe</td><td>placement</td><td>3</td><td>5.0</td><td>&quot;put the money away in the safe on the — shelf&quot;</td></tr><tr><td>place wine</td><td>placement</td><td>3</td><td>5.0</td><td>&quot;stack the wine bottle to the — of the rack&quot;</td></tr><tr><td>put in cupboard</td><td>category</td><td>9</td><td>5.0</td><td>&quot;put the — in the cupboard&quot;</td></tr><tr><td>sort shape</td><td>shape</td><td>5</td><td>5.0</td><td>&quot;put the — in the shape sorter&quot;</td></tr><tr><td>push buttons</td><td>color</td><td>50</td><td>3.8</td><td>&quot;push the — button, [then the — button]&quot;</td></tr><tr><td>insert peg</td><td>color</td><td>20</td><td>5.0</td><td>&quot;put the ring on the — spoke&quot;</td></tr><tr><td>stack cups</td><td>color</td><td>20</td><td>10.0</td><td>&quot;stack the other cups on top of the — cup&quot;</td></tr><tr><td>place cups</td><td>count</td><td>3</td><td>11.5</td><td>&quot;place — cups on the cup holder&quot;</td></tr></table>

设置。我们的模拟实验设定在 RLBench [15] 中。我们选择了 100 个任务中的 18 个，涉及至少两个或多个变体，以评估智能体的多任务能力。虽然 PERACT 可以很容易地应用于更多 RLBench 任务，但在我们的实验中，我们特别关注的是对多样化语言指令的理解，而不是针对像 “[始终] 拿掉平底锅盖” 这样的单一变体任务学习一次性策略。一些任务经过修改，以包括额外的变体。有关概述，请参见表 3。我们报告了从第 3.2 节中所述方法提取的平均关键帧。

变化。任务的变化包括随机采样的颜色、大小、形状、数量、位置和物体类别。颜色集合包括20种实例：颜色 $= \{ { \bf r } { \tt e d }$ ，栗色，青柠，绿色，蓝色，海军蓝，黄色，青色，品红，银色，灰色，橙色，橄榄色，紫色，水绿色，天蓝色，紫罗兰，玫瑰色，黑色，白色\}。大小集合包括2种实例：$\mathsf { s i z e s } = \{ \mathsf { s h o r t } , \mathsf { t a l l } \}$。形状集合包括5种实例：形状 $=$ \{立方体，圆柱体，三角形，星星，月亮\}。数量集合包括3种实例：数量 $= \{ 1 , 2 , 3 \}$。位置和物体类别具体到每个任务。例如，打开抽屉有3个放置位置：顶部，中部和底部，而放入橱柜包括9个YCB物体。除了这些语义变化，物体还以随机姿势放置在桌面上。一些大物体如抽屉的姿势变化是受限的，以确保与Franka臂的操作在运动学上是可行的。在接下来的章节中，我们将详细描述这18个任务。我们将重点突出那些从原始RLBench代码库[15] 修改的任务，并描述具体修改了哪些内容。

# A.1 打开抽屉

文件名：open_drawer.py 任务：打开三个抽屉中的一个：上、中或下。 修改：无。 对象：1个抽屉。 成功标准：指定抽屉的棱柱关节完全伸展。

# A.2 滑块

文件名：slide_block_to_color_target.py 任务：将积木滑动到一个带颜色的方形目标上。目标颜色仅限于红色、蓝色、粉色和黄色。修改：是的。原始的 slide_block_to_target.py 任务仅包含一个目标。增加了三个其他目标，使总共有四种变体。对象：1个积木和4个彩色目标方块。成功标准：积木的某个部分位于指定的目标区域内。

# A.3 扫至簸箕

文件名：sweep_to_dustpan_of_size.py 任务：将灰尘颗粒扫入短灰斗或高灰斗。 已修改：是。原始的sweep_to_dustpan.py任务仅包含一个灰斗。添加了另一个灰斗，总共有2种变体。 对象：5个灰尘颗粒和2个灰斗。 成功指标：所有5个灰尘颗粒都位于指定的灰斗内。

# A.4 从烤架上取肉

文件名：meat_off_grill.py 任务：将鸡肉或牛排从烤架上移开并放在一旁。修改：无。对象：1片鸡肉，1片牛排，1个烤架 成功标准：指定的肉类放在一旁，远离烤架。

# A.5 转动开关

文件名：turn_tap.py 任务：旋转水龙头的左侧或右侧把手。左侧和右侧是相对于水龙头的方向定义的。修改状态：无。对象：1个带有2个把手的水龙头。成功指标：指定把手的旋转关节相对于起始位置至少偏离 $90^{\circ}$。

# A.6 放入抽屉

文件名：put_item_in_drawer.py 任务：将方块放入三个抽屉之一：上层、中层或下层。已修改：否。对象：1 个方块和 1 个抽屉。成功标准：方块在指定的抽屉内。 A.7 关闭罐子 文件名：close_jar.py 任务：将指定颜色的盖子放在罐子上，并旋紧盖子。罐子的颜色从20种颜色实例的完整集合中抽样。已修改：否。对象：1 个方块和 2 个彩色罐子。成功标准：盖子旋紧且颜色正确。

# A.8 拖拽杆

文件名：reach_and_drag.py 任务：抓住棒子并用它将立方体拖到指定的彩色目标方块上。目标颜色从完整的20种颜色实例中抽样。修改：是的。原始的reach_and_drag.py任务仅包含一个目标。增加了三个随机颜色的目标。对象：1个立方体，1个棒子，和4个彩色目标方块。成功指标：立方体的某一部分位于指定目标区域内。

# A.9 堆叠块

文件名：stack_blocks.py 任务：在绿色平台上堆叠出指定颜色的 $N$ 块积木。始终有 4 块指定颜色的积木和 4 块其他颜色的干扰积木。积木颜色从 20 种颜色实例的完整集合中随机抽取。已修改：否。物体：8 块颜色积木（其中 4 块为干扰物），以及 1 个绿色平台。成功标准：$N$ 块积木位于绿色平台的区域内。

# A.10 螺口灯泡

文件名：light_bulb_in.py 任务：从指定的灯座上取下灯泡，并将其旋入灯架。灯座的颜色从20种颜色实例的完整集合中采样。场景中总会有两个灯座，一个是指定的，一个是干扰灯座。已修改：无。对象：2个灯泡，2个灯座和1个灯架。成功标准：来自指定灯座的灯泡位于灯架的插槽内。

# A.11 放入安全区域

文件名：put money_in_safe.py 任务：将一叠钱放入指定架子上的保险箱内。架子有三个放置位置：上、中、下。修改状态：无。对象：1叠钱，1个保险箱。成功标准：钱叠在保险箱内指定架子上。

# A.12 放置酒类

文件名：place_wine_at_rack_location.py 任务：抓取酒瓶并将其放在木架的三个指定位置之一：左侧、中间、右侧。位置的定义是相对于木架的朝向。 修改：是。原始的stack_wine.py任务只有一个放置位置。增加了两个其他位置，总共增加到三个变体。 对象：1个酒瓶和1个木架。 成功标准：酒瓶位于木架上指定的放置位置。

# A.13 放入橱柜

文件名：put_groceries_in_cupboard.py 任务：抓取指定物体并将其放入上方的橱柜中。场景中始终包含9个随机放置在桌面上的YCB物体。已修改：否。物体：9个YCB物体和1个悬浮在空中的橱柜（如魔法一般）。成功指标：指定物体在橱柜内。

# A.14 排序形状

文件名：place_shape_in_shape_sorter.py 任务：拾取指定形状并将其放入分类器中的正确孔内。场景中始终有4个干扰形状和1个正确形状。已修改：形状和分类器的大小已扩大，以便在RGB-D输入中容易区分。对象：5个形状和1个分类器。成功指标：指定形状在分类器内。

# A.15 按钮

文件名：push_buttons.py 任务：按指定顺序按下彩色按钮。按钮颜色从完整的20种颜色实例中采样。场景中始终有三个按钮。修改：无。对象：3个按钮。成功指标：所有指定的按钮均已被按下。

# A.16 插入钉子

文件名：insert_onto_square_peg.py 任务：拾起方块并将其放在指定颜色的插头上。插头颜色从20种颜色实例的完整集合中抽取。 修改：无。 物体：1个方块和1个带有三个颜色插头的平台。 成功指标：方块在指定的插头上。

# A.17 堆叠杯子

文件名：stack_cups.py 任务：将所有杯子堆叠在指定颜色的杯子上。杯子颜色从20种颜色实例的完整集合中抽样。场景中始终包含三个杯子。修改：无。对象数量：3个高杯。成功标准：所有其他杯子均在指定杯子内。

# A.18 放置杯子

文件名：place_cups.py 任务：将 $N$ 个杯子放置在杯架上。这是一个要求极高精度的任务，杯子的把手必须与杯架的辐条完全对齐，才能成功放置。修改：无。对象：3 个带把手的杯子和 1 个带有三个辐条的杯架。成功标准：$N$ 个杯子在杯架上，每个都放置在一个独立的辐条上。

# B PERACT 详细信息

在本节中，我们提供了 PERACT 的实现细节。请参阅此 Colab 教程以获取 PyTorch 实现。

输入观测。根据 James 等人的研究，我们的输入体素观测是一个 $100^3$ 体素网格，包含 10 个通道：$\mathbb{R}^{100 \times 100 \times 100 \times 10}$ T 的 PyTorch 的 scatter_ 函数。10 个通道由以下组成：3 个 RGB 值、3 个点值、1 个占用值和 3 个位置索引值。RGB 值经过标准化，符合零均值分布。点值是机器人坐标系中的笛卡尔坐标。占用值表示一个体素是被占用还是空的。位置索引值表示该体素相对于 $100^3$ 网格的 3D 位置。除了体素观测外，输入还包括具有 4 个标量值的本体感觉数据：抓手开合状态、左手指关节位置、右手指关节位置和时间步（动作序列的时间步）。输入语言。语言目标通过 CLIP 的语言编码器进行编码。我们使用 CLIP 的分词器对句子进行预处理，这始终会生成一个 77 个词元的输入序列（并进行零填充）。这些词元通过语言编码器被编码，生成维度为 $R^{77 \times 512}$ 的序列。

预处理。体素网格通过一个 $1 \times 1$ 核心的 3D 卷积层进行编码，将通道维度从 10 上采样到 64。同样，自我感知数据通过一个线性层进行编码，将输入维度从 4 上采样到 64。编码后的体素网格通过一个核大小和步幅均为 5 的 3D 卷积层被划分为 $5 ^ { 3 }$ 块，这样得到一个维度为 $\mathbb { R } ^ { 20 \times 20 \times 20 \times 64} $ 的块张量，维度为 $\mathbb { R } ^ { 20 \times 20 \times 20 \times 128 }$ 和 $\mathbb { R } ^ { 8000 \times 128 }$ 的特征通过线性层从 512 降采样到 128 维，然后被添加到张量中，形成最终输入序列到 Perceiver Transformer，维度为 $\mathbb { R } ^ { 8077 \times 128 }$。我们还将学习到的位置嵌入添加到输入序列中。这些嵌入用可训练的 nn 参数表示，在 PyTorch 中进行处理。

Perceiver Transformer 是一种潜在空间变换器，它使用一小组潜在向量来编码极长的输入序列。参见图 6 以获取该过程的示意图。Perceiver 首先计算输入序列与维度为 $\mathbb { R } ^ { 2 0 4 8 \times 5 1 2 }$ 的潜在向量集之间的交叉注意力。这些潜在向量是随机初始化的，并进行端到端训练。潜在向量通过 6 层自注意力层进行编码，然后与输入进行交叉注意以输出与输入维度匹配的序列。该输出经过 3D 卷积层和三线性上采样进行上采样，以形成一个具有 64 个通道的体素特征网格：$\mathbb { R } ^ { 1 0 0 \times 1 0 0 \times 1 0 0 \times 6 4 }$。这个特征网格与处理阶段初始的 64 维特征网格进行连接，作为跳跃连接到编码层。最后，使用 $1 \times 1$ 核心的 3D 卷积层将通道从 128 维下采样回 64 维。我们的 Perceiver 实现基于一个现有的开源库。

![](images/6.jpg)  
Figure 6. Perceiver Transformer Architecture. Perceiver is a latent-space transformer. Q, K, V represent queries, keys, and values, respectively. We use 6 selfattention layers in our implementation.

解码。在翻译过程中，体素特征网格通过一个 $1 \times 1$ 的三维卷积层进行解码，以将通道维度从 64 下采样到 1。该张量是维度为 $\mathbb { R } ^ { 1 0 \mathbf { \dot { 0 } } \times 1 0 0 \times 1 0 0 \times 1 }$ 的翻译 $\mathcal { Q } $ -函数，体素特征网格在三维维度上进行最大池化，以形成维度为 $\mathbb { R } ^ { 1 \times 6 4 }$ 的向量。该向量通过三个独立的线性层解码，以形成各自的 $\mathcal { Q } $ -函数，分别用于旋转、夹持器打开和防碰撞。旋转线性层输出维度为 $\mathbb { R } ^ { 2 1 6 }$ 的 logits，其他层输出维度为 $\mathbb { R } ^ { 2 }$ 的 logits。我们的代码库基于 James 等人的 ARM 仓库构建。

# C 评估工作流程

# C.1 模拟

第4.2节中的模拟实验遵循四个阶段的工作流程：(1) 生成一个数据集，包含训练集、验证集和测试集，分别包含100、25和25个示例。(2) 在训练集上训练智能体，并每10K次迭代保存一次检查点。(3) 在验证集上评估所有保存的检查点，并标记表现最佳的检查点。(4) 在测试集上评估表现最佳的检查点。虽然该工作流程遵循监督学习中的标准训练-验证-测试范式，但对于实际机器人环境而言，并不是最可行的工作流程。在实际机器人中，收集验证集并评估所有检查点可能非常昂贵。

# C.2 真实机器人

在第4.4节的真实机器人实验中，我们简单地选择训练中的最后一个检查点。我们通过可视化 $\mathcal { Q }$ 预测在交换或修改语言目标的训练示例上检查智能体是否经过充分训练。在评估训练好的智能体时，智能体会不断执行动作，直到人类用户停止执行。我们还实时可视化 $\mathcal { Q }$ 预测，以确保智能体即将采取的行动是安全可执行的。

# D 机器人设置

# D.1 仿真

所有模拟实验均采用图7所示的四个摄像头设置。前置摄像头、左肩摄像头和右肩摄像头为静态，而手腕摄像头则随末端效应器移动。我们没有修改RLBench [15] 的默认摄像头姿态。这些姿态最大限度地覆盖了桌面，同时最小化了移动手臂造成的遮挡。手腕摄像头尤其能够提供小物体如把手的高分辨率观测。

![](images/7.jpg)  
Figure 7. Simulated Setup. The four camera setup: front, left shoulder, right shoulder, and on the wrist.

# D.2 真实机器人

硬件设置。真实机器人实验使用一台配有并联夹爪的Franka Panda操作手。为了实现感知，我们使用安装在三脚架上的Kinect-2 RGB-D相机，倾斜角度朝向桌面。参考图D。我们尝试设置多个Kinect以进行多视图观察，但无法解决多个飞行时间传感器引发的干扰问题。Kinect-2以$5 1 2 \times 4 2 4$的分辨率在$3 0 \mathrm { H z }$下提供RGB-D图像。相机与机器人基坐标系之间的外参使用easy handeye软件包进行标定。我们在夹爪上安装了ARUCO标记，以帮助标定过程。数据收集。我们使用HTC Vive控制器进行演示收集。该控制器是一个6自由度追踪器，能够提供相对于静态基站的准确位姿。这些位姿作为标记在$\mathbf { R } \mathbf { V i z } ^ { 1 0 }$上显示，同时显示来自Kinect2的实时RGB-D点云。用户使用标记和点云作为参考来指定目标位姿。这些目标位姿由运动规划器执行。我们使用Franka ROS和MoveIt，默认使用RRT-Connect规划器。训练与执行。我们从头开始训练一个PERACT智能体，使用53个演示样本。训练样本通过$\pm 0 . 1 2 5 \mathrm { m }$的平移扰动和$\pm 4 5 ^ { \circ }$的偏航旋转扰动进行增强。我们在8个NVIDIA P100 GPU上训练了2天。在评估期间，我们只选择了训练中的最后一个检查点（因为我们没有收集用于优化的验证集）。推理在单个Titan X GPU上进行。

![](images/8.jpg)  
Figure 8. Real-Robot Setup with Kinect-2 and Franka Panda.

# E 数据增强

PERACT 的基于体素的公式自然允许通过 SE(3) 变换进行数据增强。在训练过程中，体素化观察样本 $\mathbf { v }$ 及其对应的关键帧动作 $\mathbf { k }$ 会受到随机平移和旋转的扰动。平移扰动的范围为 $[ \pm 0 . 1 2 5 \mathrm { { \bar { m } } , \pm 0 . 1 2 5 \mathrm { { m } , \pm 0 . 1 2 5 \mathrm { { m } } ] } }$。旋转扰动仅限于航向轴，范围为 $[ 0 ^ { \circ } , 0 ^ { \circ } , \pm 4 5 ^ { \circ } ]$。$4 5 ^ { \circ }$ 的限制确保了扰动后的旋转不会超出 Franka 臂的运动学可达范围。我们也尝试过俯仰和滚转扰动，但这会显著延长训练时间。任何将离散化动作推到观察体素网格外的扰动都会被丢弃。有关数据增强的示例，请参见图 10 的底部行。

# F 演示增强

根据 James 等人的研究，我们将每个演示中的数据点视为“预测下一个（最佳）关键帧动作”任务。该过程的示例见图 9。在该示例中，$\mathbf { k } _ { 1 }$ 和 $\mathbf { k } _ { 2 }$ 是从第 3.2 节中描述的方法提取的两个关键帧。橙色圆圈表示与下一个关键帧动作配对的 RGB-D 观测数据点。

![](images/9.jpg)  
Figure 9. Keyframes and Demo Augmentation.

# G 灵敏度分析

在表 4 中，我们考察了影响 PERACT 性能的三个因素：旋转数据增强、Perceiver 潜在变量的数量和体素化分辨率。所有多任务智能体均以每个任务 100 个示例进行训练，并在每个任务上评估 25 个回合。简要总结这些结果：（1）$4 5 ^ { \circ }$ 的偏航扰动在处理旋转变化较大的任务（如堆叠积木）时提高了性能，但在处理旋转受限的任务（如放置酒瓶）时反而降低了性能。（2）仅使用 512 个潜在变量的 PERACT 在性能上与默认的 2048 个潜在变量的智能体相当（有时甚至更好），这展示了 Perceiver 架构的压缩能力。（3）像 $3 2 ^ { 3 }$ 这样的粗网格对于某些任务是足够的，但高精度任务（如排序形状）需要更高分辨率的体素化。（4）较大的补丁大小降低了内存使用，但可能会影响需要子补丁精度的任务。

Table 4. Sensitivity Analysis. Success rates (mean $\%$ ) of various PERACT agents trained with 100 demonstrations per task. We   

<table><tr><td></td><td></td><td>open drawer</td><td>slide block</td><td>Sweep to dustpan</td><td>meat off grill</td><td>turn tap</td><td>put in drawer</td><td>close jar</td><td>drag stick</td><td>stack blocks</td></tr><tr><td>PERACT</td><td></td><td>80</td><td>72</td><td>56</td><td>84</td><td>80</td><td>68</td><td>60</td><td>68</td><td>36</td></tr><tr><td></td><td>PeRACT w/o Rot Aug</td><td>92</td><td>72</td><td>56</td><td>92</td><td>96</td><td>60</td><td>56</td><td>100</td><td>8</td></tr><tr><td>PERACT</td><td>4096 latents</td><td>84</td><td>88</td><td>44</td><td>68</td><td>84</td><td>48</td><td>48</td><td>84</td><td>12</td></tr><tr><td>PERACT</td><td>1024 latents</td><td>84</td><td>48</td><td>52</td><td>84</td><td>84</td><td>52</td><td>32</td><td>92</td><td>12</td></tr><tr><td>PERACT</td><td>512 latents</td><td>92</td><td>84</td><td>48</td><td>100</td><td>92</td><td>32</td><td>32</td><td>100</td><td>20</td></tr><tr><td>PERACT</td><td>643 voxels</td><td>88</td><td>72</td><td>80</td><td>60</td><td>84</td><td>36</td><td>40</td><td>84</td><td>32</td></tr><tr><td>PERACT</td><td>323 voxels</td><td>28</td><td>44</td><td>100</td><td>60</td><td>72</td><td>24</td><td>0</td><td>24</td><td>0</td></tr><tr><td>PERACT</td><td>73 patches</td><td>72</td><td>48</td><td>96</td><td>92</td><td>76</td><td>76</td><td>36</td><td>96</td><td>32</td></tr><tr><td>PERACT</td><td>93 patches</td><td>68</td><td>64</td><td>56</td><td>52</td><td>96</td><td>56</td><td>36</td><td>92</td><td>20</td></tr><tr><td></td><td></td><td>screw bulb</td><td>put in safe</td><td>place wine</td><td>put in cupboard</td><td>sort shape</td><td>push buttons</td><td>insert peg</td><td>stack cups</td><td>place cups</td></tr><tr><td>PERACT</td><td></td><td>24</td><td>44</td><td>12</td><td>16</td><td>20</td><td>48</td><td>0</td><td>0</td><td>0</td></tr><tr><td></td><td>PerAct w/o Rot Aug</td><td>20</td><td>32</td><td>48</td><td>8</td><td>8</td><td>56</td><td>8</td><td>4</td><td>0</td></tr><tr><td>PERACT</td><td>4096 latents</td><td>32</td><td>44</td><td>52</td><td>8</td><td>12</td><td>72</td><td>4</td><td>4</td><td>0</td></tr><tr><td>PERACT</td><td>1024 latents</td><td>24</td><td>32</td><td>36</td><td>8</td><td>20</td><td>40</td><td>8</td><td>4</td><td>0</td></tr><tr><td>PERACT</td><td>512 latents</td><td>48</td><td>40</td><td>36</td><td>24</td><td>16</td><td>32</td><td>12</td><td>0</td><td>4</td></tr><tr><td>PERACT</td><td>643 voxels</td><td>24</td><td>48</td><td>44</td><td>12</td><td>4</td><td>32</td><td>0</td><td>4</td><td>0</td></tr><tr><td>PERACT</td><td>323 voxels</td><td>12</td><td>20</td><td>52</td><td>0</td><td>0</td><td>60</td><td>0</td><td>0</td><td>0</td></tr><tr><td>PERACT</td><td>73 patches</td><td>8</td><td>48</td><td>76</td><td>0</td><td>12</td><td>16</td><td>0</td><td>0</td><td>0</td></tr></table>

# H 高精度任务

在表1中，PERACT在三个高精度任务上表现为零性能：放置杯子、堆叠杯子和插入销钉。为了调查多任务优化是否是影响性能的因素之一，我们为每个任务训练了3个独立的单任务智能体。我们发现单任务智能体能够达到非零性能，这表明更好的多任务优化方法可能会提高某些任务的性能。

Table 5. Success rates (mean $\%$ of multi-task and single-task PERACT agents trained with 100 demos and evaluated on 25 episodes.   

<table><tr><td></td><td>Multi</td><td>Single</td></tr><tr><td>place cups</td><td>0</td><td>24</td></tr><tr><td>stack cups</td><td>0</td><td>32</td></tr><tr><td>insert peg</td><td>0</td><td>16</td></tr></table>

# I 相关工作补充

在这一部分，我们简要讨论一些在第2节中未提及的其他工作。并行工作。最近，Mandi等人[83]发现，在多任务（但单一变体）环境中，对新任务进行预训练和微调与元学习方法在RLBench任务上的表现具有竞争力，甚至更好。这种预训练和微调的范式可能直接适用于PERACT，其中一个经过预训练的PERACT智能体可以迅速适应新任务，而无需明确使用元学习算法。多任务学习。在RLBench的背景下，Auto-λ[73]提出了一个多任务优化框架，超越了第3.4节中的统一任务加权。该方法根据验证损失动态调整任务权重。未来的PERACT工作可以用Auto-λ替代统一任务加权，以实现更好的多任务性能。在Meta-World[53]的背景下，Sodhani等人[84]发现语言条件化在50个任务变体的多任务强化学习中带来了性能提升。基于语言的规划。本文仅探讨了语言指令在整个回合中不变的单一目标设置。然而，语言条件自然允许以顺序方式组合多个指令[69]。因此，一些先前的工作[85, 13, 86, 87]已将语言作为高层动作规划的媒介，然后可以使用预训练的低层技能来执行这些动作。未来的工作可以结合基于语言的规划，以实现更抽象目标的落地，如“做晚餐”。任务与运动规划。在任务与运动规划（TAMP）[88, 89]的子领域中，Konidaris等人[90]提出了一种以动作为中心的符号规划方法。给定一组预定义的动作技能，智能体与环境互动以构建一组符号，这些符号随后可用于规划。

体素表示。基于体素的表示已被用于多个特别受益于三维理解的领域。比如在物体检测、物体搜索和视觉-语言基础构建中，体素图被用来构建持久的场景表示。在神经辐射场（NeRF）中，体素特征网格大幅减少了训练和渲染时间。类似地，其他机器人领域的研究也使用了体素化表示，以嵌入驾驶和操作的视点不变性。Perceiver中潜在向量的使用与计算机图形学中的体素哈希有广泛关系。PerceiverIO不是使用基于位置的哈希函数将体素映射到固定大小的内存，而是使用交叉注意力将输入映射到固定大小的潜在向量，这些向量是端到端训练的。另一大区别是对未占用空间的处理。在图形学中，未占用空间对渲染没有影响，但在PERACT中，未占用空间是许多“动作检测”发生的地方。因此，未占用空间和已占用空间之间的关系，即场景、物体和机器人，对于学习动作表示至关重要。长上下文和潜在空间变换器。为扩展变换器到更长的上下文长度，提出了几种方法。使用固定大小潜在向量而不是完整上下文的潜在空间变换器就是其中一种。关于速度、内存和性能之间的权衡，没有明确的赢家。然而，潜在空间方法在物体检测和基于槽的物体发现中取得了令人信服的结果。

# J 额外的 Q 预测示例

图10展示了训练后的PERACT智能体的额外$\mathcal { Q }$预测示例。传统的以对象为中心的表征，如姿态和实例分割，难以高精度地表示豆堆或番茄藤。而以行动为中心的智能体，如PERACT，专注于学习动作的感知表征，这就要求从业者定义什么应该被视为对象（这是一项更困难的问题，且通常特定于任务和具体的实现）。

![](images/10.jpg)  
Figure 10. Additional Q-Prediction Examples. Translation $\mathcal { Q }$ -Prediction examples from PERACT. The top two rows are from simulated

# K 失败的事情

在本节中，我们描述了一些尝试，但并未成功或在实践中引起了问题。真实世界的多摄像头设置。我们尝试设置多个 Kinect-2 以实现真实世界的多视角观察，但未能解决多个飞行时间传感器之间的干扰问题。具体来说，深度帧变得非常嘈杂，并且有很多空洞。未来的工作可以尝试快速开关相机，或使用干扰最小的更好的飞行时间摄像头。傅里叶特征作为位置嵌入。我们也尝试将傅里叶特征与输入序列串联，而不是使用学习到的位置嵌入，就像某些 Perceiver 模型中所做的那样。傅里叶特征导致性能大幅下降。预训练视觉特征。继 CLIPort 之后，我们尝试使用来自 CLIP 的预训练视觉特征，而不是原始 RGB 值，以启动学习并提高对未见对象的泛化。我们在 4 个 RGB 帧上运行了 CLIP 的 ResNet50，并在 UNet 的方式中使用共享解码层进行上采样。但是我们发现这非常慢，尤其是因为 ResNet50 和解码层需要在 4 个独立的 RGB 帧上运行。在这种额外开销下，训练多任务智能体所需的时间远远超过 16 天。未来的工作可以尝试在辅助任务上进行解码层的预训练和预提取特征，以加快训练速度。在多个自注意力层上进行上采样。受到密集预测变换器（DPT）的启发，我们尝试在 Perceiver 变换器的多个自注意力层上进行特征上采样。然而这根本没有效果；也许 Perceiver 的潜在空间自注意力层与 ViT 和 DPT 的全输入自注意力层截然不同。极端旋转增强。除了偏航旋转扰动外，我们还尝试扰动俯仰角和横滚。虽然 PERACT 仍然能够学习策略，但训练所需的时间显著延长。使用 Adam 而非 LAMB。我们尝试使用 Adam 优化器训练 PERACT，而不是 LAMB，但在模拟和真实世界实验中均导致性能下降。

# L 限制与风险

尽管PERACT相当强大，但它也存在一些局限性。以下部分讨论了这些局限性及其在实际应用中的潜在风险。 基于采样的运动规划器。PERACT依赖于基于采样的运动规划器来执行离散化的动作。这使得PERACT受到随机规划器的影响，以达到特定的姿态。尽管这一问题在我们实验中的任务中并未造成重大问题，但许多其他任务对到达姿态的路径非常敏感。例如，将水倒入杯子需要为适当地倾斜水容器提供一条平滑的路径。未来的工作可以通过结合学习的和采样的运动路径来解决这一问题。 动态操控。另一问题是离散时间的离散化动作不易应用于需要实时闭环操控的动态任务。这可以通过一个独立的视觉伺服机制来解决，该机制能够通过闭环控制到达目标姿态。或者，PERACT可以扩展为预测一系列离散化的动作，而不仅仅是一个动作。在这里，基于Transformer的架构可能特别有利。此外，智能体还可以被训练为预测其他物理参数，如目标速度。 灵巧操控。使用离散化动作对多自由度机器人（如多指手）也并非易事。具体来说，对于多指手，PERACT可以被修改为预测可以通过逆运动学求解器达到的 fingertip 姿态，但对于像多指手这样的欠驱动系统，这种方法的可行性和鲁棒性尚不明确。 推广至新实例和物体。在图11中，我们报告了关于打开抽屉任务的小规模扰动实验结果。我们观察到，改变把手的形状不会影响性能。然而，具有随机纹理和颜色的把手使智能体感到困惑，因为在训练过程中它只见过一种颜色和纹理的抽屉。超越这种一次性设置，在多个抽屉实例上进行训练可能会提高泛化性能。尽管我们没有明确研究对未见物体的泛化能力，但训练PERACT的动作检测器以适应广泛的物体，并评估其处理新物体的能力，可能是可行的，这类似于语言条件的实例分割器和物体检测器的使用。或者，可以使用来自多模态编码器（如CLIP或R3M）的预训练视觉特征进行引导学习。

![](images/11.jpg)

语言基础的范围。与之前的工作类似，PERACT 对动词-名词短语的理解紧密依赖于示范和任务。例如，“用簸箕清理桌上的豆子”中的“清理”特指将豆子放入簸箕的动作序列，而不是一般意义上的“清理”，后者可以应用于其他任务，例如用布清洁桌子。任务完成预测。对于现实世界和模拟评估，一个预言机指示是否达到了期望目标。这个预言机可以被一个成功分类器替代，该分类器可以预训练以从 RGB-D 观察中预测任务完成情况。历史和部分可观察性。PERACT 仅依赖当前观察来预测下一个动作。因此，需要历史信息的任务，如计数或排序，是不可行的，除非伴随有任务完成预测器。同样，对于涉及部分可观察性的任务，例如逐一检查抽屉寻找特定物品，PERACT 不会记录之前看到的内容。未来的工作可以包括来自前一时间步的观察，或者附加 Perceiver 潜变量，或训练一个循环神经网络以跨时间步编码潜变量。基于运动学可行性的数据增强。第 E 节中描述的数据增强方法并未考虑用 Franka 手臂达到扰动动作的运动学可行性。未来的工作可以预先计算出在离散化动作空间中不可达的姿态，并丢弃任何将动作推入不可达区域的增强扰动。平衡的数据集。由于 PERACT 仅用少量示范进行训练，因此偶尔会利用训练数据中的偏差。例如，如果“将蓝色方块放在黄色方块上”的示例在训练数据中过于常见，PERACT 可能倾向于始终这样做。通过扩展数据集以包含更多样化的物体和属性示例，这些问题可能会得到解决。此外，数据可视化方法可用于识别和修正这些偏差。多任务优化。第 3.4 节中提出的均匀任务采样策略有时可能会影响性能。由于所有任务的权重相等，针对某些具有共同元素的任务（例如，移动方块）进行优化，可能对其他不同任务（例如，扭转水龙头）的性能产生不利影响。未来的工作可以采用动态任务加权方法，如 Auto $\lambda$ 来更好地进行多任务优化。部署风险。PERACT 是一个用于 6-DoF 操作的端到端框架。与任务和运动规划中的某些方法不同，后者有时可以提供任务完成的理论保证，PERACT 是一个纯粹反应式系统，其性能只能通过经验方法进行评估。此外，与之前的工作不同，我们不使用可能包含有害偏见的互联网预训练视觉编码器。即便如此，在部署前彻底研究和减轻任何偏见是明智的。因此，对于现实世界的应用，在训练和测试期间与人类保持互动可能会有所帮助。不建议在安全关键系统中使用未见过的物体和有人员的观察。

# M 新兴属性

在本节中，我们展示了关于 PERACT 新兴特性的初步发现。

# M.1 目标跟踪

尽管PERACT并不是专门针对6自由度物体跟踪进行训练的，但我们的动作检测框架可以用来在杂乱场景中定位物体。在这个视频中，我们展示了一个智能体，它仅使用5个“按压洗手液”的示例在一个洗手液实例上进行了训练，然后在一个未见过的洗手液实例上进行了跟踪评估。PERACT不需要构建完整的洗手液表示，仅需学习如何按压它们。我们的实现以每秒2.23帧（每帧0.45秒）的推理速度运行，允许近实时的闭环行为。

![](images/12.jpg)

![](images/13.jpg)  
Figure 12. Object Tracker. Tracking an unseen hand sanitizer instance.   
Figure 13. Examples of Multi-Modal Predictions.

# M.2 多模态动作

PERAcT的问题表述允许对多模态动作分布进行建模，即在特定目标下多个动作都是有效的场景。图13展示了从PERAcT中选取的一些多模态动作预测示例。由于有多个“黄色方块”和“杯子”可以选择，$\mathcal { Q }$预测分布有多个模态。在实际操作中，我们观察到智能体倾向于偏好某些物体实例（如图13中的前面杯子），这是由于训练数据集中的偏好偏置造成的。我们还注意到，第3.4节中的基于交叉熵的训练方法与能量基模型（EBMs）紧密相关。在某种程度上，交叉熵损失推动了专家的6自由度动作，同时压制了离散动作空间中的其他所有动作。在测试时，我们简单地最大化所学得的$\mathcal { Q }$预测，而不是通过优化来最小化能量函数。未来的工作可以探讨EBM [114] 的训练和推理方法，以实现更好的泛化和执行性能。