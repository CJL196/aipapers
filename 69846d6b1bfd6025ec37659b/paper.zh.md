# 探索驱动的生成交互环境

内德科·萨沃夫1，纳塞尔·卡泽米尔·穆罕默德·马赫迪1，丹达·帕尼·保杜尔1，西·王1,2,3，卢克·范·古尔1 1 索非亚大学“圣克里门特·奥赫里德斯基”INSAIT 2 苏黎世联邦理工学院 3 慕尼黑工业大学

# 摘要

现代世界模型需要昂贵且耗时的大量视频数据集，这些数据集包含人类或特定环境智能体的动作示范。为了简化训练，我们专注于使用许多虚拟环境来获取便宜的、自动收集的交互数据。Genie [5] 是一个最近的多环境世界模型，展示了许多共享行为的环境的仿真能力。不幸的是，训练他们的模型需要昂贵的示范。因此，我们提出一个训练框架，仅使用虚拟环境中的随机智能体进行训练。虽然以这种方式训练的模型表现出良好的控制能力，但它受到随机探索可能性的限制。为了克服这一限制，我们提出了AutoExplore Agent——一种完全依赖于世界模型的不确定性的探索智能体，提供多样化的数据，以便其能够学习最佳策略。我们的智能体完全独立于特定环境的奖励，因此能够轻松适应新环境。通过这种方法，预训练的多环境模型能够快速适应新环境，提升视频的真实感和可控性。为了自动获取大规模的交互数据集进行预训练，我们对具有相似行为和控制的环境进行了分组。为此，我们对974个虚拟环境的行为和控制进行了标注——这个数据集被命名为RetroAct。为了构建我们的模型，我们首先创建了Genie的开放实现版本——GenieRedux，并在我们的版本GenieRedux-G中进行了增强和调整。我们的代码和数据可在https://github.com/insait-institute/GenieRedux获取。

# 1. 引言

从交互环境中学习使我们能够理解和表征规则、可能的行动及其所带来的后果。作为耗时的手动编写合成模拟器的替代方案，世界模型已作为深度学习工具出现，能够完全根据观察（通常是观察到的环境的图像）对真实环境进行建模[1, 5, 37, 65]。

![](images/1.jpg)  
Figure 1. Our proposed world model training framework. It consists of a pretrained multi-environment world model on random agent data, and a new AutoExplore Agent that explores an environment and delivers diverse data for fine-tuning.

之前的研究如 [19, 23, 33] 使用轻量级世界模型来支持具有目标导向的智能体，提供目标特定的状态表示。重点在于粗略的未来预测，而非其高视觉质量。相比之下，近期的世界模型的目标是根据过去的观察和行动实现高质量的未来预测。这些最新模型能够提供真实的行动执行，甚至实现与人类的实时互动 [1, 58]。这一切得益于扩散模型、变换器 [14, 59] 和状态空间模型 [17] 的兴起，以及借鉴视频生成管道的架构选择 [51, 60]。通常，这些生成模型的设计目的是紧密匹配单一选定的环境。一个最先进的模型Genie，在许多具有相似动态的视觉多样化环境中进行训练，从而在新视觉上展现出泛化能力。构建这些高质量的统计模拟器需要对环境和要模拟的行动进行多样的观察。有些研究通过昂贵的视频数据集收集和人类示范行动的整理来获得这些数据 [1, 5, 63]。如果行动不可用，则需要设计一个额外的组件来预测它们，这可能会引入与真实标注相比的不确定性 [5, 37]。在这种情况下，扩展到具有新类型行动的新环境是困难的，因为这又需要一个昂贵的数据收集过程。其他一些研究，如 [58] 探索了使用环境特定的智能体来检索数据，在他们的案例中是游戏Doom。

在本研究中，我们提出了一个在多个环境中可访问且无需付出太多努力的世界模型训练框架。为此，我们首先构建了RetroAct——一个经过注释和整理的大型复古游戏环境数据集（基于Stable Retro [48]的环境）。我们根据行为标签和控制描述对这些环境进行分组。这种分组使我们能够生成具有相似行为的大规模交互数据集。接下来，我们使用随机智能体对多环境世界模型GenieRedux进行预训练，这是我们对Genie [5]的开放实现。与[66]报告的通过预训练改善智能体行为不同，我们的目标是改善世界模型。为此，我们对GenieRedux进行了适应虚拟环境的调整，并实施了架构和训练过程的增强，最终得到GenieRedux-G模型。我们观察到，仅通过对从RetroAct中自动收集的200个环境和50个具有映射控制的环境的随机交互训练GenieRedux-G，我们能够获得控制行为（50个环境中的0.450 ∆PSNR）和合理的视觉保真度（50个环境中的26.36 PSNR）。

由于随机动作在探索环境方面的能力有限，我们开发了一种获取更丰富的交互数据的方法，以提升模型的控制行为和视觉逼真度。为此，受到[53]的启发，我们开发了自己的环境无关奖励函数，使得智能体能够探索不同的环境，完全不依赖于预定义的环境奖励。虽然他们的目标是高性能的目标驱动智能体，但我们的设计基于改进底层的大世界模型，以提高环境模拟的视觉逼真度和可控性。图1提供了图形说明。我们探索驱动智能体的目标是最大化世界模型的不确定性，这一不确定性通过在GenieRedux-G的观察预测阶段获得的分类熵来估计。一旦获得丰富的数据，我们会对GenieRedux-G进行微调。我们表明，与随机智能体预训练相比，该方法在视觉（提高高达7.4 PSNR）和控制（提高高达$1 . 4 \Delta \mathrm { P S N R } $）上都有显著改善。我们的贡献如下：• 基于我们世界模型不确定性训练探索智能体的廉价数据收集框架。• GenieRedux和GenieRedux-G的实现与发布——基于[5]的开源Pytorch模型。• 基于我们的分词器表示研究，对模型进行的架构和损失变化，从而提高了视觉逼真度。• 为多环境世界模型训练准备的大规模环境数据集。

# 2. 相关工作

世界模型。最初作为辅助强化学习（RL）智能体的粗略想象模型[10, 19, 21, 24, 53]，世界模型已经发展为以动作为条件的独立真实视频生成模型[9, 39, 50, 64]。它们通过提供环境的预测表示来促进特定任务的智能体训练。受到[20]的启发，Ha 和 Schmidhuber [18] 使用变分自编码器（VAE）将视觉观测编码为潜在状态，并利用多维条件随机神经网络（MDNRNN）基于先前状态、动作和 VAE 输出预测未来状态，以促进策略学习。DreamerV2 [22] 引入了一种 RL 智能体，达到了人类水平的 Atari 游戏表现。它使用卷积神经网络（CNN）对图像进行编码，并利用递归状态计算后验和先验随机状态。不过与我们的工作不同，它并未评估智能体对世界模型改进的影响，也没有跨不同环境推广任务奖励。世界模型还旨在生成以动作为条件的真实视频[27, 37, 65]。Genie [5] 训练了一个视频分词器和一个潜在动作模型（LAM）用于动态下一帧生成。GAIA-1 [27] 通过将多模态输入编码为统一表示，并基于先前输入预测图像词元，处理非结构化环境中的自动驾驶问题，使用自回归变换器。Menapace 等人 [37] 采用编码器-解码器架构，其中预测的动作标签作为瓶颈，允许用户通过离散动作控制生成的视频。这些工作中的关键差距是自动数据收集，而这是我们的方法所解决的。高效探索。在 RL 中高效探索的重要性被[28]突显。早期方法通过添加噪声 [16, 34] 或使用熵正则化 [40] 来增强探索，但它们存在动作空间限制，并且在复杂动态情况下常常失败，因为多样的动作并不总能推动有意义的探索。一种更直接的方法是使用异质智能体 [26, 29, 52] 采用多样的探索策略以增强环境探索。贝叶斯方法 [54, 57] 也被引入来创建基于不确定性的探索获取函数 [2, 38, 41-44]，但通常在对高维输入（如图像）进行泛化时面临困难。

![](images/2.jpg)  
agent. The reward is solely based on the classification uncertainty of our model.

最近的探索方法强调状态新颖性，鼓励智能体仅在访问状态后评估新颖性。相较之下，我们的方法受到启发，使用模型不一致性主动引导智能体前往具有最高潜力的状态，而不依赖环境目标驱动的奖励。提议通过状态转移中的不确定性和简单特征提取器驱动的探索智能体。相反，我们提出了一种旨在改善不建模状态的世界模型的探索智能体。Plan2Explore使智能体能够使用最大化RSSM模型状态熵的奖励来寻求新颖状态。尽管Plan2Explore通过其框架改善了目标导向的智能体，但我们使用基于词元不确定性的独特探索奖励改进了现代变压器世界模型。EX2学习了一个分类器来区分访问过的状态，为分类器难以区分的状态提供内在奖励，而不是依赖于世界模型。基于KL散度的方法通过比较分布来引导探索，例如，SMM计算策略引导的状态分布与均匀目标之间的KL散度。Tao等人提出了一种基于状态与其在低维特征空间中最近邻之间距离的内在奖励。然而，低维度导致信息损失，限制了对完整状态空间的探索——这是我们通过使用世界模型来解决的问题。

# 3. RetroAct 数据集

我们首先通过构建一个框架，以低成本获取多环境交互数据，来解决可接入的多环境世界模型训练问题。特别是，我们旨在收集许多环境中相似动作的交互数据。我们并不依赖于昂贵的人类交互，而是获取和整理一组虚拟环境。作为数据来源，我们使用了Stable Retro框架，该框架包含了多个平台上的复古游戏，并附带有起始状态。我们不使用已定义的奖励。我们获取几乎所有支持的游戏（974个）。

![](images/3.jpg)  
Figure 3. RetroAct Annotation. Description of environments in Ret roAct by annotated attribute. Better viewed zoomed.

这个原始数据集包含了非常不同的视觉和行为混合环境。然而，在我们学习相似动态的设置中，需要建立环境行为之间的对应关系。我们进行标注，对每个环境的三个方面进行分类。运动风格分类了控制下移动的对象及其方式的整体风格，密切相关于游戏类型；摄像机视角；控制轴描述玩家可以移动的方向。标签分布显示在图3中。在表1中，我们将我们的RetroAct与其他相关数据集进行了比较。RetroAct通过提供行为和控制标注，同时保持较高的环境数量，具有独特性。我们发现，数据集中最常见的环境类型是平台游戏，共483个标题。作为最大的子集，我们仅提取这些游戏进行进一步使用，因为需要许多展现相似控制的环境。我们为模型定义了五个运动动作 - 向左移动、向右移动、向上移动、向下移动和跳跃。每个游戏都有自己按钮与动作的映射。因此，我们为每个483个标题中的5个选定动作生成一个短视频片段，并构建一个标注工具来观察和标注执行的动作。最终，我们标注了2,925个行为标签和2,898个控制标签。

Table 1. Comparison of RetroAct dataset to others.   

<table><tr><td>Dataset</td><td>Type</td><td>#Environments</td><td>Diverse Behaviors</td><td>Open</td><td>Behavior Annotation</td><td>Control Annotation</td></tr><tr><td>Coinrun [13]</td><td>Environments</td><td>1</td><td>X</td><td>√</td><td>X</td><td>×</td></tr><tr><td>ALE [4]</td><td>Environments</td><td>57</td><td>✓</td><td>✓</td><td>X</td><td>X</td></tr><tr><td>Stable Retro [48]</td><td>Environments</td><td>1003</td><td>✓</td><td>✓</td><td>X</td><td>X</td></tr><tr><td>Platformers [5]</td><td>Videos</td><td>Unknown</td><td>✓</td><td>X</td><td>X</td><td>X</td></tr><tr><td>RetroAct(Ours)</td><td>Environments</td><td>974</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr></table>

经过实验，我们观察到在环境数量较多的情况下，模型需要更多的训练，因此我们定义了两个子集以应对计算成本：一个是由483个经过行为过滤的游戏中的前200个游戏组成的子集，用于预训练；另一个是使用RetroAct的动作标签，从50个随机选择的动作一致性游戏中组成的子集，用于微调。

我们通过在所有环境中启动随机智能体来收集大规模数据集，收集动作和观测。从200个游戏集中，我们构建了平台游戏数据集 Platformers-200，包含10,000个回合（每个游戏50个回合），每个回合最多500帧，生成了460万张图像。从50个游戏集中，我们获得了Platformers-5000数据集，包含5000个回合（每个游戏100个回合），每个回合长度最多为1000，生成了480万张图像。在我们的协议中，我们将每个环境的1%会话取作验证集。我们展示了使用随机智能体已经足以学习一定程度的可控性，然后在此基础上构建我们设计的探索智能体。为了验证我们的GenieRedux实现，我们实现了CoinRun案例研究。利用上述协议，我们获得了一个包含10,000个回合的数据库，最大长度为500，生成了400万张图像。

# 4. 多环境世界模型

考虑到虚拟环境，我们的第一个目标是自动获取图像序列数据集 $I _ { 1 } , . . . , I _ { N }$ 和相应的动作 $a _ { 1 } , . . . , a _ { N - 1 }$ 。给定序列 $I _ { 1 } , . . . , I _ { N }$ 及过去和未来的动作 $a _ { 1 } , . . . , a _ { N + T - 1 }$ ，我们的世界模型旨在预测未来的 $T$ 帧 $I _ { N + 1 } , . . . , I _ { T }$ ，这些帧对应于已执行的动作。GenieRedux。由于作者未提供 Genie [5] ，我们创建了一个开源实现，并称之为 GenieRedux。我们将在第 5 节和补充材料 F 中对我们的实现进行定量和定性验证。它由三个组件组成。一个视频

分词器将输入帧序列编码为时空词元：$e _ { 1 } , . . . , e _ { N } \ = \ T _ { e n c } ( I _ { 1 } , . . . , I _ { N } )$ ，并解码回图像：${ { I } _ { 1 } } , . . . , { { I } _ { N } } = { { T } _ { d e c } } ( { { e } _ { 1 } } , . . . , { { e } _ { N } } )$ 。潜在动作模型将输入帧序列编码为时空词元：$\begin{array} { r l } { a _ { 1 } , . . . , a _ { N - 1 } } & { { } = } \end{array}$ $L A M _ { e n c } ( I _ { 1 } , . . . , I _ { N - 1 } )$ ，并将其解码以重构未来预测 $I _ { 2 } , . . . , I _ { N } = L A M _ { d e c } ( a _ { 1 } , . . . , a _ { N - 1 } )$ 。动态模块根据部分被遮蔽的帧词元和动作预测下一个帧：$I _ { 2 } , . . . , I _ { N + T - 1 } =$ $D ( e _ { 1 } , . . . , e _ { N } , . . . , e _ { N + T - 1 } ; a _ { 1 } , . . . , a _ { N + T - 1 } )$ ，其中在推理过程中 $e _ { N } , . . , e _ { N + T - 1 }$ 被遮蔽。我们严格遵循Genie的规范来实现这些组件。所有组件使用因果空间时间变换网络（STTN）[62]。我们使用位置编码生成器（PEG）[12]进行空间和时间注意力，使用带线性偏置的注意力（ALiBi）[49]进行时间注意力。我们以序列大小16帧和分辨率 $6 4 \mathrm { x } 6 4$ 训练我们的模型，以应对计算限制。我们在50K数据样本上训练基于U-Net的超分辨率网络，将输出放大到 $2 5 6 \times 2 5 6$ 。（补充材料B）GenieRedux-G。在基础模型的基础上，我们提供一种变体 - GenieRedux-G，适用于虚拟环境，并包含架构和训练改进。虽然GenieRedux使用必不可少的LAM模型来获取动作，但我们放弃了它，因为我们的智能体提供了真实标注的动作。相反，onehot动作被连接到动态模块的每一层进行条件处理。通过这种方式，我们避免了预测的不确定性。动态模块由ST-ViViT编码器组成，后接MaskGIT架构[8]，根据计划在训练期间预测在分词器词典中随机遮蔽输入词元的索引。由于使用了标准交叉熵，令牌分类的缺点是对任何不同于真实标注的预测都进行同等惩罚。然而，词典中相近的词元比远处的词元变化显著更少，正如第5节所示。为了在 $N _ { E }$ 个词元的分类中使这种词元之间的距离概念得以实现，我们设计了一种词元距离交叉熵（TDCE）损失：

$$
T D C E ( x , y ) = ( y ^ { T } K ) \cdot s o f t m a x ( x ) + C E ( x , y )
$$

这里 $\boldsymbol { x } \in \mathcal { R } ^ { N _ { E } }$ 是预测的 logits，$y \in \mathcal { R } ^ { N _ { E } }$ 是真实的 one-hot 类别。$K \in \mathcal { R } ^ { N _ { E } \times N _ { E } }$ 是在训练开始时预计算的所有词元之间的余弦距离表；$C E ( . )$ 表示标准交叉熵损失。当一个错误的词元类别被赋予概率时，它会根据与真实类别的距离受到惩罚。MaskGIT 的设计是以可学习的嵌入为输入，这些嵌入由 Tokenizer 预测的词元进行索引。它们是随机初始化的，因此不包含任何词元的内容。鉴于编码本身和词元之间的距离能够影响动态模块的性能，我们通过将嵌入添加到词元本身来增加一个跳过连接，这提高了模型的视觉保真度和可控性。自动探索智能体 我们扩展了框架，增加了一个探索智能体，通过深入环境获取数据。我们将其命名为自动探索智能体。智能体的奖励完全基于世界模型的性能，且在没有任何环境奖励的情况下运行。因此，它可以在各种环境中进行训练，而无需调整其特定参数或依赖于奖励定义。

我们的奖励设计基于 GenieRedux-G 使用分类进行词元预测的事实。每个词元是通过从代码本的类别分布中抽样来预测的。我们首先通过让 GenieRedux-G-50 从我们想要估计奖励的当前观测 $I _ { c }$ 向后运行 5 步，获得所有 $N _ { T }$ 的词元预测分布。我们提供 2 张图像 $I _ { c - 4 } , I _ { c - 3 }$ ，预测 3 张图像 - $I _ { c - 2 } , . . . , I _ { c }$ ，并提取 $I _ { c }$ 的预测词元分布以获得 $x =$ $[ x _ { 1 } , . . . , x _ { t } , . . . , x _ { N _ { T } } ]$ 。我们通过计算类别分布的熵来评估每个预测词元的预期不确定性 $u _ { t }$ ，并将其归一化到 [0, 2] 的范围内：

$$
u _ { t } = \frac { 2 \cdot \sum _ { i } ^ { N _ { T } } x _ { i } \cdot l o g ( x _ { i } ) } { N _ { e } }
$$

研究Tokenizer表示的属性，我们发现一种普遍的词元被学习以表示环境的静态部分。只有变化的部分会产生较高的不确定性，因此我们取整个不确定性集合$S = \{ u _ { t } \}$中不确定性最高的$25\%$子集$S_{top}$。奖励，如公式3所示，设定了智能体的目标，收集数据以最大化世界模型的不确定性。

$$
\begin{array} { r } { S _ { 2 5 \% } = \{ u \in S \mid u \geq Q _ { 7 5 } ( S ) \} } \\ { R ( I _ { c } ) = \frac { 1 } { | S _ { 2 5 \% } | } \underset { u \in S _ { 2 5 \% } } { \sum } u } \end{array}
$$

我们的智能体是一个行动-评价模型，采用策略梯度方法训练。在智能体架构方面，我们参考了[39]。它由一个卷积神经网络（CNN）编码器和接下来的长短期记忆网络（LSTM）组成。作为强化学习的标准做法，4帧图像被堆叠、进行最大池化，结果作为智能体单个时间步的输入。探索驱动的世界模型训练。我们最初在平台游戏环境$- 2 0 0$上预训练GenieRedux-G，然后在平台游戏环境-50上进行微调，以获得模型GenieRedux-G-50。接着，我们使用GenieRedux-G-50作为奖励来源训练AutoExplore Agent。智能体训练的详细信息见附录A.3。在选定环境中运行经过训练的探索智能体，我们获得了一个新的多样化数据集，其中包含在未见场景下的动作示范。我们首先对Tokenizer的解码器进行1,000次迭代的微调，以适应新的未见场景。然后，对GenieReduxG的动态模块在新数据上进行微调，以在新条件下实现更高的视觉逼真性和可控性。为了构建评估我们方法的测试集，我们针对每个探索的环境训练了一个Agent-57模型，利用可用的环境奖励。关于测试设置的更多详细信息请参见附录A.2。

为了评估视觉保真度，我们使用 FID（Fréchet 发生距离）Heusel et al. [25]、PSNR（信噪比）和 SSIM（结构相似性指标）Wang et al. [61]。为了评估可控性，我们使用最近提出的 $\Delta _ { t } \mathrm { P S N R }$ 指标 [5]，该指标比较真实标注动作 $( \hat { x } _ { t } )$ 与随机动作 $( \hat { x } _ { t } ^ { \prime } )$ 的视觉效果：$\Delta _ { t } \mathrm { P S N R } = \mathrm { P S N R } ( x _ { t } , \hat { x } _ { t } ) - \mathrm { P S N R } ( x _ { t } , \hat { x } _ { t } ^ { \prime } )$，其中 $x _ { t }$ 是时间 $t$ 时的真实标注帧。更高的 $\Delta _ { t } \mathrm { P S N R }$ 表示更高的可控性。与 Bruce et al. [5] 一致，对于所有实验，我们报告 $t = 4$ 时的 $\Delta _ { t } \mathrm { P S N R }$。

# 5. 实验

比较GenieRedux和GenieRedux-G。我们执行了原始CoinRun案例研究，使用随机智能体，正如[5]所建议的，以验证和比较GenieRedux与LAM，以及使用智能体提供行动的GenieRedux-G。在本研究中，模型之间唯一的区别是LAM的存在。我们首先在由随机智能体收集的数据集上进行训练。视觉保真度结果见表2。我们的GenieRedux实现展示了高视觉质量，匹配了所有七个CoinRun环境行动，以及环境动态进程（详见补充材料F）。然而，正如指标所示，GenieRedux-G显示出更优的视觉保真度和可控性（详见补充材料F），因为它避免了LAM预测的不确定性。本研究表明，即使使用随机智能体也能在世界模型中产生行动性能能力。接下来，我们在环境奖励上使用PPO训练一个演员-评论家智能体，参考[13]收集数据并训练GenieRedux-TA和GenieRedux-G-TA。表3显示了由训练智能体收集的测试集上的评估结果。

Table 2. Comparison of GenieRedux and GenieRedux-G on Basic Test Set. Peformed on a test set, collected from the Coinrun environment with randomly sampled actions.   

<table><tr><td rowspan="2">Model</td><td colspan="3">Basic Test Set</td></tr><tr><td>FID↓</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>Tokenizer</td><td>18.14</td><td>38.25</td><td>0.96</td></tr><tr><td>LAM</td><td>37.01</td><td>33.97</td><td>0.92</td></tr><tr><td>GenieRedux</td><td>21.88</td><td>25.51</td><td>0.77</td></tr><tr><td>GenieRedux-G</td><td>18.88</td><td>33.41</td><td>0.92</td></tr></table>

Table 3. Comparison of GenieRedux and GenieRedux-G on Diverse Test Set. The models are trained with data collected by random agent and trained agent (-TA), and tested on data collected by a trained agent from the Coinrun environment.   

<table><tr><td rowspan="2">Model</td><td colspan="3">Diverse Test Set</td></tr><tr><td>FID↓</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>Tokenizer</td><td>19.13</td><td>35.85</td><td>0.94</td></tr><tr><td>Tokenizer-TA</td><td>11.63</td><td>40.62</td><td>0.97</td></tr><tr><td>GenieRedux</td><td>23.97</td><td>23.82</td><td>0.73</td></tr><tr><td>GenieRedux-G</td><td>19.51</td><td>31.66</td><td>0.90</td></tr><tr><td>GenieRedux-TA</td><td>12.57</td><td>31.97</td><td>0.90</td></tr><tr><td>GenieRedux-G-TA</td><td>12.40</td><td>34.44</td><td>0.92</td></tr></table>

GenieRedux-G 在所有设置上均优于 GenieRedux。此外，使用多样化代理收集的数据训练的模型在视觉表现上优于使用随机代理训练的模型。GenieRedux-G-TA 的 $\Delta { \sf P S N R }$ 为 1.89，而 GenieRedux-G 为 0.70，显示出多样化数据训练在可控性上的优越性。（更多内容见 Sup.Mat. F） 多环境模型。在这里，我们评估最初在 RetroAct 中的多个环境上训练的模型。GenieRedux-G-200 在 Platformers $- 2 0 0$ 数据集上进行 180k 次迭代的预训练。在验证集上，我们获得了 23.32 的 PSNR 和 17.12 的 FID。以该模型为基础，GenieRedux-G-50 在 Platformers-50 上进行训练。其在从所选 50 个环境单独生成的 $1 0 \mathrm { k }$ 会话测试集上的定量评估见表 4 开头。由于这 50 个环境具有相应的动作控制，我们观察到了预测质量的提升。图 4 展示了 GenieRedux-G 成功执行指令动作。由于上移动作很少使用，它更多地作为无操作动作存在。（更多内容见 Sup.Mat C.1） 消融研究。在这项实验中，我们评估了 GenieRedux-G 中每项改进的附加收益——附加的词元输入以及使用词元距离交叉熵损失进行训练。消融实验是在生成的 $1 0 \mathrm { k }$ 会话测试集上进行的，每个会话长 500 帧。

Table 4. Ablation study on improvements in GenieRedux-G.   

<table><tr><td>Model GenieRedux-G-200</td><td>FID↓ 22.31</td><td>PSNR↑ 25.11</td><td>SSIM↑ 0.80</td></tr><tr><td colspan="2">GenieRedux-G-50 + Token Input + TDCE Loss Autoregressive</td><td colspan="2">23.80 26.36 22.96 26.65 22.95 27.06 22.11 28.07</td><td>0.84 0.84 0.85</td></tr><tr><td colspan="2">Input Right Left</td><td>Up</td><td>Down</td><td>0.88 Jump</td></tr><tr><td>K</td><td></td><td></td><td></td><td></td></tr><tr><td>|</td><td></td><td></td><td></td><td></td></tr></table>

数据是通过在“平台者-50”环境中使用随机动作策略收集的。视觉保真度评估见表4。可以看出，每个组件在视觉保真度方面都为我们的模型带来了益处。最后，我们对最佳模型进行自回归评估，以实现我们的最高得分。

标记器表示研究。该实验提供了对 GenieRedux-G 内部工作的深入见解，以激励我们提出的更改。由于动态模块完全基于标记表示进行操作，我们对此进行了深入研究。图 5 显示了输入序列的重构（第一行）和可视化的标记表示（最后一行），其中每个预测的标记索引被分配了不同的颜色。第一帧的视觉特征由各种标记捕获。从第二帧开始，表示发生了显著变化 - 一个标记专门用于表示静态帧区域，而所有运动区域则更新为新的内容。我们观察到视觉上相似的补丁预测相同或类似的标记，因此我们将每个预测的标记替换为其在代码本中最接近的标记。我们只保持特殊背景标记不变。在图 5 的第二行中，我们展示了结果重构 - 虽然出现了一些模糊，图像仍然保持大体不变。相反，将每个标记替换为其在代码本中最远的标记（第三行）会导致图像有显著的不同。这一特性 - 更接近的标记具有更相似的外观 - 激励我们提出标记距离交叉熵损失，该损失惩罚预测远离真实标注的标记。图 6 可视化了 GenieRedux-G-50 对其动态模块每个预测标记的气不确定性，该不确定性度量是对 1024 个代码本标记的分类熵。与运动相关的标记具有最高的不确定性；其他区域大多数被分类为“静态”标记。因此，最小的角色移动会产生低不确定性，而向前运动则增加不确定性。这促使我们基于这一不确定性构建 AutoExplore Agent 的奖励。

<table><tr><td>Environment</td><td>Strategy</td><td>Model</td><td>FID↓</td><td>PSNR↑</td><td>SSIM↑</td><td>∆PSNR↑</td></tr><tr><td rowspan="4">Adventure Island II</td><td>Random</td><td>GenieRedux-G-50 GenieRedux-G-50-ft</td><td>41.99 42.34</td><td>26.32 27.04</td><td>0.81 0.81</td><td>0.83 1.19</td></tr><tr><td>Exploration</td><td>Tokenizer-ft GenieRedux-G</td><td>11.01</td><td>38.95</td><td>0.98</td><td>-</td></tr><tr><td></td><td>GenieRedux-G-50-ft</td><td>11.94 12.77</td><td>28.33 30.60</td><td>0.88 0.90</td><td>0.37 1.47</td></tr><tr><td>Random Autoregressive Exploration Autoregressive</td><td>GenieRedux-G-50-ft GenieRedux-G-50-ft</td><td>41.55 11.33</td><td>27.82 33.61</td><td>0.83 0.94</td><td>1.24 2.09</td></tr><tr><td rowspan="2">Super Mario Bros</td><td>Random</td><td>GenieRedux-G-50 GenieRedux-G-50-ft Tokenizer</td><td>29.83 30.13</td><td>34.24 34.54</td><td>0.94 0.94</td><td>0.56 0.54</td></tr><tr><td>Exploration Random Autoregressive</td><td>GenieRedux-G GenieRedux-G-50-ft</td><td>8.09 9.56 9.55</td><td>42.00 34.00 36.13</td><td>0.99 0.95 0.97</td><td>- 0.09 0.57</td></tr><tr><td rowspan="5">Smurfs</td><td>Exploration Autoregressive</td><td>GenieRedux-G-50-ft GenieRedux-G-50-ft</td><td>30.84 9.33</td><td>34.85 37.77</td><td>0.95 0.97</td><td>0.57 0.76</td></tr><tr><td>Random</td><td>GenieRedux-G-50 GenieRedux-G-50-ft</td><td>79.51 80.61</td><td>21.47 21.83</td><td>0.69</td><td>0.47</td></tr><tr><td rowspan="2">Exploration</td><td>Tokenizer</td><td>17.86</td><td>35.61</td><td>0.70 0.98</td><td>0.65 -</td></tr><tr><td>GenieRedux-G</td><td>20.43</td><td>35.42</td><td>0.80</td><td>0.85</td></tr><tr><td>Random Autoregressive</td><td>GenieRedux-G-50-ft</td><td>20.01</td><td>27.45</td><td>0.85</td><td>1.55</td></tr><tr><td></td><td>Exploration Autoregressive</td><td>GenieRedux-G-50-ft GenieRedux-G-50-ft</td><td>80.16 18.97</td><td>22.16 29.53</td><td>0.71 0.90</td><td>0.69 2.06</td></tr></table>

![](images/4.jpg)  
Figure 5. Tokenizer Representation. Reconstruction images from the tokenizer, and the effect of replacing each token with its closest and furthest in the codebook. Lastly, we visualize the indices of the predicted tokens.

基于探索的训练。我们展示了对GenieRedux-G的基于探索的训练。我们在三个环境中进行该过程——AdventureIslandII，提供了一个简单的设置供智能体学习（开始时无敌人的单个平台）；SuperMarioBros在开始后不久提供了敌人和障碍物；Smurfs则提供了更复杂的背景图像和不同的动作动态。对于每个环境，我们训练了一种自动探索智能体。我们观察到智能体学会向前移动并避开障碍以最大化奖励。（更多信息见补充材料D）

![](images/5.jpg)  
Figure 6. Dynamics Uncertainty. Shown is the uncertainty per token predicted for each image of an example sequence. Uncertainty is generated in the regions of motion.

我们使用预训练的GenieRedux-G-50模型作为基线，并在两种设置中对其进行微调：一种是由随机智能体在选定环境中收集的数据集，另一种是由我们的AutoExplorer智能体收集的数据集。每个数据集包含10,000个会话，每个会话长700帧。我们在10,000次迭代中对GenieRedux-G-50进行微调（GenieRedux-G-50-ft），并选择表现最佳的模型。在我们的比较中，我们还包括一个从头开始在多样性探索数据集上训练的GenieRedux-G模型，该模型进行1,5000次迭代，以展示预训练的效果。我们对所有模型执行单遍生成，并对微调模型在随机数据和AutoExplorer智能体的数据集上执行计算负载更重的自回归评估。表5展示了每个环境的视觉保真度和可控性指标，证实了我们探索方法的有效性。基于AutoExplorer智能体数据微调的模型在视觉保真度方面始终优于那些在随机动作上训练的模型。基于探索的微调也提高了可控性。对于小字符和均匀背景的环境，所有模型的学习难度可能更大。然而，在这种情况下，可控性的提升在自回归评估中仍然显著可见。图7展示了我们方法的卓越质量。此外，我们观察到多环境预训练相比于非预训练模型在研究的两个方面都带来了显著的提升。（更多内容见补充材料C）

![](images/6.jpg)  
Figure 7. AutoExplore Agent vs Random Agent Qualitative Comparison. We show that AutoExplore exhibits better visual quality and avoids losing track of the agent.

Table 6. Comparison of AutoExplore Agent with others.   

<table><tr><td>Agent</td><td colspan="3">SuperMarioBros AdventureIslandII PSNR↓ SSIM↓ ΔPSNR↓P</td></tr><tr><td>RF</td><td>28.58 0.94</td><td>0.181</td><td>|PSNR↓ SSIM↓ ∆PSNR↓ 24.82 0.78 0.44</td></tr><tr><td>VAE</td><td>24.40 0.86</td><td>0.087</td><td>16.57 0.5</td></tr><tr><td>Ours 23.81</td><td>0.85</td><td>0.065</td><td>0.072 15.20 0.41 0.070</td></tr></table>

AutoExplore代理评估。我们将AutoExplore代理与文献[6]中的基于探索的方法进行比较。我们在GenieRedux上基于RF和VAE特征的SSE训练代理，并在表6中与我们的结果进行比较。AutoExplore代理的奖励结果显示最大世界模型视觉和可控性误差（基于1k次代理动作的实验），实现了其在我们框架中的预期角色。用户研究。为了验证最终结果的质量，我们进行了一项用户研究，请人们分别对由基于随机代理数据和AutoExplore代理数据训练的GenieRedux-G生成的样本质量进行1到5的评分。我们的研究中的每个样本由两个16帧的片段以同步方式播放组成——真实片段和我们的GenieRedux-G-50-ft重建片段，给定两个初始帧并自回归地生成其余部分。我们向用户提供共120个样本——每种模型40个样本和两个真实样本的40个样本，以建立规模。从两个选定的游戏——超级马里奥兄弟和冒险岛II中各提供20个样本。我们收集了19名参与者的反馈。结果如图8所示。基于AutoExplore代理数据训练的模型显然更接近真实值，证明了我们方法的质量。

![](images/7.jpg)  
Figure 8. User study results. Our user study on two games shows that our model trained with AutoExplore Agent's data is consistently rated higher.

通过第二项用户研究，我们评估生成帧的动作准确性。我们使用模糊的单输入案例（角色在空中起始）并在 AdventureIslandII 上生成 60 个包含 3 个动作的剪辑。用户更倾向于我们的探索训练模型，对其评分为 $\mathbf { 0 . 7 5 \ : \pm { \ : 0 . 0 1 9 } }$，评分范围为 0（随机偏好）到 1（探索偏好）。（更多内容见 Sup.Mat. E.2）

# 6. 结论

随着世界模型发展为具备出色仿真特性的庞大模型，它们需要大量的交互数据集，包括多样化的观察和动作。Genie 在多个环境中训练展示了令人印象深刻的能力，然而，它需要收集大量视频数据集和一个用于推断动作的模型。在本研究中，我们通过构建一个新的框架，从大量虚拟环境中收集交互数据，从而解决数据收集和整理的沉重负担。我们首先构建了 Genie 的开放实现——GenieRedux，并将其增强为 GenieRedux-G 版本。我们通过在大量虚拟环境上进行预训练，获得了表现出控制力的模型。我们提出了 AutoExplore Agent，作为一种完全独立于环境奖励的智能体，旨在最大化 GenieRedux-G 的不确定性，从而解决随机数据收集策略的过拟合限制。在对探索过的环境进行微调后，我们的模型在视觉逼真度和可控性方面的提升显著优于仅依靠随机智能体数据进行训练的结果。在多个环境中展示这一点后，我们证明了我们的框架使下一代世界模型的训练变得更易获取、更具成本效益和省力的潜力。

# 7. 致谢

INSAIT，索非亚大学“圣克利门特·奥赫里德斯基”。部分资助来自保加利亚教育与科学部对INSAIT的支持，作为保加利亚国家科研基础设施路线图的一部分。本项目得到了谷歌云平台（GCP）提供的计算资源支持。

# References

[1] Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos J Storkey, Tim Pearce, and François Fleuret. Diffusion for world modeling: Visual details matter in atari. Advances in Neural Information Processing Systems, 37: 5875758791, 2024.   
[2] Kamyar Azizzadenesheli, Emma Brunskill, and Animashree Anandkumar. Efficient exploration through bayesian deep q-networks. In 2018 Information Theory and Applications Workshop (ITA), pages 19. IEEE, 2018.   
[3] Marc Bellemare, Sriram Srinivasan, Georg Ostrovski, Tom Schaul, David Saxton, and Remi Munos. Unifying countbased exploration and intrinsic motivation. Advances in neural information processing systems, 29, 2016.   
[4] M. G. Bellemare, Y. Naddaf, J. Veness, and M. Bowling. The arcade learning environment: An evaluation platform for general agents. Journal of Artificial Intelligence Research, 47:253279, 2013.   
[5] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024.   
[6] Yuri Burda, Harri Edwards, Deepak Pathak, Amos Storkey, Trevor Darrell, and Alexei A Efros. Large-scale study of curiosity-driven learning. In International Conference on Learning Representations, 2019.   
[7] Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by random network distillation. In International Conference on Learning Representations, 2019.   
[8] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1131511325, 2022.   
[9] Chang Chen, Yi-Fu Wu, Jaesik Yoon, and Sungjin Ahn. Transdreamer: Reinforcement learning with transformer world models. arXiv preprint arXiv:2202.09481, 2022.   
10] Silvia Chiappa, Sébastien Racaniere, Daan Wierstra, and Shakir Mohamed. Recurrent environment simulators. In International Conference on Learning Representations, 2017.   
11] Leshem Choshen, Lior Fox, and Yonatan Loewenstein. Dora the explorer: Directed outreaching reinforcement actionselection. arXiv preprint arXiv:1804.04012, 2018.   
12] Xiangxiang Chu, Zhi Tian, Bo Zhang, Xinlong Wang, and Chunhua Shen. Conditional positional encodings for vision transformers. In The Eleventh International Conference on Learning Representations. 2023.   
[13] Karl Cobbe, Oleg Klimov, Chris Hesse, Taehoon Kim, and John Schulman. Quantifying generalization in reinforcement learning. In International conference on machine learning, pages 12821289. PMLR, 2019.   
[14] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2020.   
[15] Justin Fu, John Co-Reyes, and Sergey Levine. Ex2: Exploration with exemplar models for deep reinforcement learning. Advances in neural information processing systems, 30, 2017.   
[16] Scott Fujimoto, Herke Hoof, and David Meger. Addressing function approximation error in actor-critic methods. In International conference on machine learning, pages 1587 1596. PMLR, 2018.   
[17] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.   
[18] David Ha and Jürgen Schmidhuber. Recurrent world models facilitate policy evolution. Advances in neural information processing systems, 31, 2018.   
[19] David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018.   
[20] David Ha, Jonas Jongejan, and Ian Johnson. Draw together with a neural network. Retrieved Oct, 5:2021, 2017.   
[21] Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In International conference on machine learning, pages 25552565. PMLR, 2019.   
[22] Danijar Hafner, Timothy P Lillicrap, Mohammad Norouzi, and Jimmy Ba. Mastering atari with discrete world models. In International Conference on Learning Representations, 2021.   
[23] Danijar Hafner, Timothy P Lillicrap, Mohammad Norouzi, and Jimmy Ba. Mastering atari with discrete world models. In International Conference on Learning Representations, 2021.   
[24] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023.   
[25] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.   
[26] Dan Horgan, John Quan, David Budden, Gabriel BarthMaron, Matteo Hessel, Hado van Hasselt, and David Silver. Distributed prioritized experience replay. In International Conference on Learning Representations, 2018.   
[27] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023.   
[28] Sham Kakade and John Langford. Approximately optimal approximate reinforcement learning. In Proceedings of the Nineteenth International Conference on Machine Learning, pages 267274, 2002.   
[29] Steven Kapturowski, Georg Ostrovski, John Quan, Remi Munos, and Will Dabney. Recurrent experience replay in distributed reinforcement learning. In International conference on learning representations, 2018.   
[30] Youngjin Kim, Wontae Nam, Hyunwoo Kim, Ji-Hoon Kim, and Gunhee Kim. Curiosity-bottleneck: Exploration by distilling task-specific novelty. In International conference on machine learning, pages 33793388. PMLR, 2019.   
[31] Martin Klissarov, Riashat Islam, Khimya Khetarpal, and Doina Precup. Variational state encoding as intrinsic motivation in reinforcement learning. In Task-Agnostic Reinforcement Learning Workshop at Proceedings of the International Conference on Learning Representations, pages 16 32, 2019.   
[32] Lia Le, Benjamn ysebach, milo Parisotto, Ei Xig, Sergey Levine, and Ruslan Salakhutdinov. Efficient exploration via state marginal matching. arXiv preprint arXiv:1906.05274, 2019.   
[33] Ian Lenz, Ross A Knepper, and Ashutosh Saxena. Deepmpc: Learning deep latent features for model predictive control. In Robotics: Science and Systems, page 25. Rome, Italy, 2015.   
[34] Ryan Lowe, Yi I Wu, Aviv Tamar, Jean Harb, OpenAI Pieter Abbeel, and Igor Mordatch. Multi-agent actor-critic for mixed cooperative-competitive environments. Advances in neural information processing systems, 30, 2017.   
[35] Marlos C Machado, Marc G Bellemare, and Michael Bowling. Count-based exploration with the successor representation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 51255133, 2020.   
[36] Jarryd Martin, Suraj Narayanan Sasikumar, Tom Everitt, and Marcus Hutter. Count-based exploration in feature space for reinforcement learning. arXiv preprint arXiv:1706.08090, 2017.   
[37] Willi Menapace, Stephane Lathuiliere, Sergey Tulyakov, Aliaksandr Siarohin, and Elisa Ricci. Playable video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1006110070, 2021.   
[38] Alberto Maria Metelli, Amarildo Likmeta, and Marcello Restelli. Propagating uncertainty in reinforcement learing via wasserstein barycenters. Advances in Neural Information Processing Systems, 32, 2019.   
[39] Vincent Micheli, Eloi Alonso, and François Fleuret. Transformers are sample-efficient world models. In The Eleventh International Conference on Learning Representations, 2023.   
[40] Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International conference on machine learning, pages 19281937. PmLR, 2016.   
[41] Ian Osband and Benjamin Van Roy. Bootstrapped thompson sampling and deep exploration. arXiv preprint arXiv:1507.00300, 2015.   
[42] Ian Osband, Charles Blundell, Alexander Pritzel, and Benjamin Van Roy. Deep exploration via bootstrapped dqn. Advances in neural information processing systems, 29, 2016.   
[43] Ian Osband, Benjamin Van Roy, and Zheng Wen. Generalization and exploration via randomized value functions. In International Conference on Machine Learning, pages 23772386. PMLR, 2016.   
[44] Ian Osband, John Aslanides, and Albin Cassirer. Randomized prior functions for deep reinforcement learning. Advances in Neural Information Processing Systems, 31, 2018.   
[45] Georg Ostrovski, Marc G Bellemare, Aäron Oord, and Rémi Munos. Count-based exploration with neural density models. In International conference on machine learning, pages 27212730. PMLR, 2017.   
[46] Deepak Pathak, Pulkit Agrawal, Alexei A Efros, and Trevor Darrell. Curiosity-driven exploration by self-supervised prediction. In International conference on machine learning, pages 27782787. PMLR, 2017.   
[47] Deepak Pathak, Dhiraj Gandhi, and Abhinav Gupta. Selfsupervised exploration via disagreement. In International conference on machine learning, pages 50625071. PMLR, 2019.   
[48] Mathieu Poliquin. Stable retro, a maintained fork of openai's gym-retro. https://github.com/FaramaFoundation/stable-retro,2024.   
[49] Ofir Press, Noah Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. In International Conference on Learning Representations, 2022.   
[50] Jan Robine, Marc Höftmann, Tobias Uelwer, and Stefan Harmeling. Transformer-based world models are happy with 100k interactions. In The Eleventh International Conference on Learning Representations, 2023.   
[51] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1068410695, 2022.   
[52] Tom Schaul. Prioritized experience replay. arXiv preprint arXiv:1511.05952, 2015.   
[53] Ramanan Sekar, Oleh Rybkin, Kostas Daniilidis, Pieter Abbeel, Danijar Hafner, and Deepak Pathak. Planning to explore via self-supervised world models. In International conference on machine learning, pages 85838592. PMLR, 2020.   
[54] Niranjan Srinivas, Andreas Krause, Sham Kakade, and Matthias Seeger. Gaussian process optimization in the bandit setting: no regret and experimental design. In Proceedings of the 27th International Conference on International Conference on Machine Learning, pages 10151022, 2010.   
[55] Haoran Tang, Rein Houthooft, Davis Foote, Adam Stooke, OpenAI Xi Chen, Yan Duan, John Schulman, Filip DeTurck, and Pieter Abbeel. # exploration: A study of count-based exploration for deep reinforcement learning. Advances in neural information processing systems, 30, 2017.   
[56] Ruo Yu Tao, Vincent François-Lavet, and Joelle Pineau. Novelty search in representational space for sample efficient expiorauion. Aavances in Ieural injormation rrocessing Systems, 33:81148126, 2020.   
[57] William R Thompson. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika, 25(3-4):285294, 1933.   
[58] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines. arXiv preprint arXiv:2408.14837, 2024.   
[59] A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems, 2017.   
[60] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual descriptions. In International Conference on Learning Representations, 2022.   
[61] Zhou Wang, Alan Conrad Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13:600612, 2004.   
[62] Mingxing Xu, Wenrui Dai, Chunmiao Liu, Xing Gao, Weiyao Lin, Guo-Jun Qi, and Hongkai Xiong. Spatialtemporal tansorme etorks for taf fowfoeas. arXiv preprint arXiv:2001.02908, 2020.   
[63] Sherry Yang, Yilun Du, Seyed Kamyar Seyed Ghasemipour, Jonathan Tompson, Leslie Pack Kaelbling, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. In The Twelfth International Conference on Learning Representations, 2023.   
[64] Sherry Yang, Jacob Walker, Jack Parker-Holder, Yilun Du, Jake Bruce, Andre Barreto, Pieter Abbeel, and Dale Schuurmans. Video as the new language for real-world decision making. arXiv preprint arXiv:2402.17139, 2024.   
[65] Ze Yang, Yun Chen, Jingkang Wang, Sivabalan Manivasagam, Wei-Chiu Ma, Anqi Joyce Yang, and Raquel Urtasun. Unisim: A neural closed-loop sensor simulator. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13891399, 2023.   
[66] Lixuan Zhang, Meina Kan, Shiguang Shan, and Xilin Chen. Prelar: World model pre-training with learnable action representation. In European Conference on Computer Vision, pages 185201, 2024.   
[67] Tianjun Zhang, Paria Rashidinejad, Jiantao Jiao, Yuandong Tian, Joseph E Gonzalez, and Stuart Russell. Made: Exploration via maximizing deviation from explored regions. Advances in Neural Information Processing Systems, 34:9663 9680, 2021.

# Exploration-Driven Generative Interactive Environments

Supplementary Material

# Table of Contents

# A Experimental Protocol S1

A.1. Training Protocol of GenieRedux and GenieRedux-G. S1   
A.2 Testing Protocol of GenieRedux and GenieRedux-G. S2   
A.3 Training Protocol of AutoExplore Agent S2

# B Super-Resolution Network S2

# C Multi-Environment Models Additional Experiments 1

C.1. Qualitative Results of GenieRedux-G-50 S2   
C.2 Autoregressive Evaluation S2   
C.3. Multi-Environment Fine-tuning . . . . S4   
C.4. Qualitative Evaluation of Fine-tuned   
Models. . S4

# D AutoExplore Agent Behavior Study S6

# E User Studies S6

E.1. General Quality User Study Details . . S6   
E.2. Action Quality User Study Details . . S6

# F. GenieRedux Evaluation on CoinRun Case Study ST

F.1. CoinRun Case Study Details S7   
F.2. Prediction Horizon Evaluations . . . . S7   
F.3. GenieRedux-G Qualitative Evaluation S7   
F.4. GenieRedux-TA Qualitative Evaluation S7   
F.5. Jafar Qualitative Comparison . . . . . S7   
F.6. Additional GenieRedux-G-TA Quali  
tative Results and Limitations . . . . S8

# A. Experimental Protocol

# A.1. Training Protocol of GenieRedux and GenieRedux-G.

The architecture and training parameters of the Tokenizer and the Dynamics module of GenieRedux-G are shown respectively in the Tab. 7 and Tab. 8. GenieRedux shares those choices, with the addition of LAM defined as in Tab. 9. For the purpose of the case study, we use 7 latent actions. Training parameters can be seen in Tab. 10.

Table 7. Tokenizer hyperparameters   

<table><tr><td>Component</td><td>Parameter</td><td>Value</td></tr><tr><td>Encoder</td><td>num_blocks</td><td>8</td></tr><tr><td></td><td>d_model</td><td>512</td></tr><tr><td></td><td>num_heads</td><td>8</td></tr><tr><td>Decoder</td><td>num_block</td><td>8</td></tr><tr><td></td><td>d_model</td><td>512</td></tr><tr><td></td><td>num_heads</td><td>8</td></tr><tr><td>Codebook</td><td>num_codes</td><td>1024</td></tr><tr><td></td><td>latent_dim</td><td>32</td></tr></table>

Table 8. Dynamics hyperparameters   

<table><tr><td>Component</td><td>Parameter</td><td>Value</td></tr><tr><td>Architecture</td><td>num_blocks</td><td>12</td></tr><tr><td rowspan="5">Sampling</td><td>d_model</td><td>512</td></tr><tr><td>num_heads</td><td>8</td></tr><tr><td>temperature</td><td>1.0</td></tr><tr><td>maskgit_steps</td><td>25</td></tr><tr><td></td><td></td></tr></table>

Table 9. LAM hyperparameters   

<table><tr><td>Component</td><td>Parameter</td><td>Value</td></tr><tr><td>Encoder</td><td>num_blocks</td><td>8</td></tr><tr><td></td><td>d_model</td><td>512</td></tr><tr><td></td><td>num_heads</td><td>8</td></tr><tr><td>Decoder</td><td>num_blocks</td><td>8</td></tr><tr><td></td><td>d_model</td><td>512</td></tr><tr><td></td><td>num_heads</td><td>8</td></tr><tr><td>Codebook</td><td>num_codes</td><td>7</td></tr><tr><td></td><td>latent_dim</td><td>32</td></tr></table>

Table 10. Optimizer Hyperparameters   

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>max_lr</td><td>1 × 10−4</td></tr><tr><td>min_lr</td><td>5 × 10-5</td></tr><tr><td>β1</td><td>0.9</td></tr><tr><td>β2</td><td>0.99</td></tr><tr><td>weight_decay</td><td>1 × 10−4</td></tr><tr><td>linear_warmup_start_factor</td><td>0.5</td></tr><tr><td>warmup_steps</td><td>5000</td></tr></table>

We train the Tokenizer on 8 A100 GPUs for $7 2 \mathrm { k }$ iterations, with batch size 112 and patch size 4, on a dataset of all 483 environments (50 sessions per environment obtained with a random agent).

Our Dynamics module is trained with sequences of

16 frames, processed by the pretrained tokenizer. Dynamics module is trained with batch size 80 on 8 A100 GPUs for $1 8 5 \mathrm { k }$ iterations on Platformers $- 2 0 0$ (GenieRedux-G-200), and fine-tuned for 80k iterations on Plat formers-50 (GenieRedux-G-50), batch size 160.

For an agent (random or AutoExplore Agent), we obtain a dataset of $1 0 \mathrm { k }$ sequences of length 800. We finetune GenieRedux-G-50 on a set for 10k iterations to obtain GenieRedux-G-50-ft, with batch size 160.

We always use the Adam optimizer with a linear warmup and cosine annealing strategy.

We note that GenieRedux has $\sim 3 5 0 \mathrm { M }$ total parameters, broken down as follows: Tokenizer (100M), LAM (170M), and Dynamics (80M). Meanwhile, GenieRedux-G has $\mathord { \sim } 1 8 0 \mathrm { M }$ total parameters: Tokenizer (100M) and Dynamics (80M).

# A.2. Testing Protocol of GenieRedux and GenieRedux-G.

For our test set, we train Agent57 per environment, using the available environment reward. In order to have many diverse episodes in our datasets and all the actions to be represented, we mix, using an $\epsilon$ -greedy approach, the agent's actions with random actions. We collect 1000 episodes as the test set (with episode length 700) and evaluate on sequences of size 12 with step size 20 in two settings. While our model can handle a single frame as input, for a fair evaluation, we choose to provide two, as a single frame does not provide motion information and there are multiple valid solutions (see Sect. F.6). We provide two frames and predict the next 10, given all actions. In the usual case, we perform MaskGIT inference with 25 iterations for all 10 images at once. We obtain much fewer artifacts and higher level of control if we adapt an autoregressive approach - iteratively generating 2 frames at a time given all previous tokens in the sequence, each with 25 iterations. However, as this is computationally heavy, we provide autoregressive results for our best models only in our evaluations.

# A.3. Training Protocol of AutoExplore Agent

For each of the environments, we train for 300 epochs with the following schedule for each epoch: 1. Run the current agent for 200 steps in 8 running environments in parallel to collect data in the replay buffer. Actions are sampled with temperature 1.0, with an epsilon-greedy algorithm with $\epsilon$ starting from 0.1 and linearly decaying to 0.01 over the course of 150 epochs. 2. Train the agent for 200 steps, sampling from the buffer, with batch size 128. Actor-critic loss is used with entropy regularization over the actions to prevent greedy behavior. In the end, we choose the agent with the highest evaluation return throughout training.

# B. Super-Resolution Network

We upscale the outputs of GenieRedux-G from 64x64 to 256x256 by a U-Net based super-resolution network, with MSE loss for both training and evaluation. The training data consists of $2 5 6 \times 2 5 6$ images that we captured from the original environments. Three configurations were tested: (1) a small U-Net with feature channel dimensions [64,128,256,512] and approximately 31 million parameters, trained on 16,000 images (12,000 for training and 4,000 for testing), achieving a test loss of 0.0081; (2) the same small U-Net trained on a larger dataset of 50,000 images (45,000 for training and 5,000 for testing), achieving a test loss of 0.0047; and (3) a larger U-Net with feature channel dimensions [128,256,512,1024] and 124 million parameters, trained on the same 50,000-image dataset, achieving a test loss of 0.0029. All models were trained with a batch size of 128, a learning rate of 0.0001, and a step-based scheduler (step_size $= 2 5$ , gamma $_ { = 0 . 5 }$ for 300 epochs, using Adam optimizer.

# C. Multi-Environment Models Additional Experiments

# C.1. Qualitative Results of GenieRedux-G-50

In Fig. 9 we show examples of 10-frame predictions from GenieRedux-G-50. They are sampled from the test set and the actions that resulted in the ground truth sequence were given to the model to produce the predictions. As seen, the model was able to produce outcomes from the actions that are close to the ground truth. In Fig. 10 is shown the developments of 3 actions over time for GenieRedux-G-50, showing a smooth trajectory and the action being executed.

In our experiments, we test the ability of our models to simulate multiple environments in virtual environments already observed by the models. For new unseen environments, our models show limited generalization abilities, characterized by pausing motion and visual artifacts. We believe that generalizability can be improved by training our tokenizer on a larger video dataset, however, with care taken to preserve the learned background token strategy learned, as it brings important properties to our exploration reward and the Dynamics module.

# C.2. Autoregressive Evaluation

For one of the environments - SuperMarioBros, we provide comparison of all our models using autoregressive evaluation. This evaluation is more computationally heavy, so we originally compare them with a single-pass evaluation, and evaluate autoregressively only the fine-tuned models on both strategies - with a random agent and AutoExplore Agent. As for small characters and uniform backgrounds, single-pass evaluation appears to produce close results between all models in terms of controllability, we choose to perform full autoregressive evaluation on SuperMarioBros (where these conditions are present) to show the benefit of our approach. Results are shown in Tab. 11. The newly autoregressively evaluated models are GenieRedux-G-50 and GenieRedux. It can be concluded that, with our exploration approach, we obtain significantly better results in terms of visual fidelity and controllability.

![](images/8.jpg)  
for comparison.

Table 11. SuperMarioBros Autoregressive Quantitative Evaluation.   

<table><tr><td>Environment</td><td>Strategy</td><td>Model</td><td>FID↓</td><td>PSNR↑</td><td>SSIM↑</td><td>ΔPSNR↑</td></tr><tr><td rowspan="3">Super Mario Bros.</td><td>Random Autoregressive</td><td>GenieRedux-G-50 GenieRedux-G-50-ft</td><td>30.48 30.84</td><td>34.59 34.85</td><td>0.94 0.95</td><td>0.55 0.57</td></tr><tr><td></td><td>Tokenizer-ft</td><td>8.08</td><td>42.00</td><td>0.99</td><td>-</td></tr><tr><td rowspan="2">Exploration Autoregressive</td><td rowspan="2">GenieRedux-G</td><td>9.46</td><td>34.38</td><td>0.95</td><td>0.07</td></tr><tr><td>GenieRedux-G-50-ft</td><td>9.33</td><td>37.77 0.97</td><td>0.76</td></tr></table>

TaRxp (Exploration). GenieRedux-G denotes a non-fine-tuned model, trained with the exploration data.   

<table><tr><td>Environment</td><td>Strategy</td><td>Model</td><td>FID↓</td><td>PSNR↑</td><td>SSIM↑</td><td>ΔPSNR↑</td></tr><tr><td rowspan="6">Combined Environments</td><td rowspan="2">Random</td><td>GenieRedux-G-50</td><td>43.57</td><td>27.55</td><td>0.82</td><td>0.65</td></tr><tr><td>GenieRedux-G-50-ft</td><td>43.98</td><td>27.74</td><td>0.82</td><td>0.78</td></tr><tr><td rowspan="2">Exploration</td><td>Tokenizer-ft</td><td>14.02</td><td>37.98</td><td>0.98</td><td>-</td></tr><tr><td>GenieRedux-G</td><td>14.88</td><td>28.91</td><td>0.88</td><td>0.25</td></tr><tr><td>Random Autoregressive</td><td>GenieRedux-G-50-ft</td><td>14.61</td><td>31.29</td><td>0.91</td><td>1.09</td></tr><tr><td></td><td>GenieRedux-G-50-ft</td><td>43.69</td><td>28.19</td><td>0.83</td><td>0.79</td></tr><tr><td></td><td>Exploration Autoregressive</td><td>GenieRedux-G-50-ft</td><td>14.49</td><td>33.14</td><td>0.93</td><td>1.46</td></tr></table>

![](images/9.jpg)  
development of the actions.

# C.3. Multi-Environment Fine-tuning

In this experiment, we take the diverse datasets, collected from the three environments we have studied - AdventureIslandII, SuperMarioBros and Smur f s, and fine-tune GenieRedux-G-50 on all of them together. In this way, we evaluate the effect of our method on multi-environment training. Results are shown in Tab. 12.

Using AutoExplore Agent's data, the model has improved its visual fidelity and controllability across the test set, containing all three environments (equal number of samples each). This shows that our method is applicable for improving multi-environment training as well.

# C.4. Qualitative Evaluation of Fine-tuned Models

In Fig. 11 are shown examples per environment of predictions from a model, fine-tuned on a random agent versus a model fine-tuned on AutoExplore Agent's data. The model, trained on AutoExplore Agent's data, exhibits much higher visual quality and less artifacts. We also note that the tokenizer plays a role in improving visual quality. After exploration, the tokenizer is able to fit better to new visuals of the environment, which reduces visual artifacts.

In Fig. 12 we show AutoExplore Agent's data helping to achieve better controllability compared to the random agent. As typical for controllability evaluation, we give a single frame as input. We observe that in cases where the motion is ambiguous (e.g. where a character might be going up or down), fine-tuning with exploration data leads to more confidence and hence realistic sequence generation. In contrast, models trained on random data cannot resolve the situation and copy the frame multiple times.

![](images/10.jpg)  
p .

![](images/11.jpg)  
lbi GRupetu like this where the agent can be going up or down, exploration data shows to improve performance.

![](images/12.jpg)  
Figure 13. AutoExplore Agent Behavior. We show the behavior of our AutoExplore Agent on the three environments studied. It can be seen that the agent learned to progress by moving right, jumping over obstacles and dealing with enemies.

# D. AutoExplore Agent Behavior Study

The agent was trained with the five actions that the world model was trained with. While initially in training the agent learns simpler strategies like jumping, eventually it achieves better returns by learning to move forwards in an environment (and reveal new scenes). To progress even further, the agent learns to overcome obstacles and enemies. In Fig. 13, we show the behavior of the agent on the three environments used after training. The agent is observed to move forward in the environment, to overcome enemies, jump over obstacles. Interestingly, the strategy in Smurfs was to sometimes wait to be attacked by an enemy, which caused the player to disappear and the camera to move before spawning. This seems to cause an increase in world model uncertainty in that environment. In other cases (flying enemies), the agent tries to avoid. In Smurfs, there is an action of entering a door. We observe that sometimes the agent enters a door that causes the character to reappear from a different side on the screen.

# E. User Studies

# E.1. General Quality User Study Details

We provide extra details about our user study to evaluate the models fine-tuned on data from a random agent and from AutoExplore Agent. In Fig. 14 we show the interface for a single sample given to the user. A clip is shown with two parts that the user should compare and rate. The order of the samples is random. The instructions given to the users at the start of the study are provided below.

Thank you for participating in our study! You will watch a total of 120 video samples. Each sample consists of two clips:

Top clip: Reference Bottom clip: Comparison clip

![](images/13.jpg)  
Figure 14. General User Study Sample.

![](images/14.jpg)  
Figure 15. Action Quality User Study Sample.

Please compare the two clips in each sample and rate how closely they match in terms of visual quality and content. Use the scale provided:

1 : The two clips completely differ in terms of visual quality and/or content   
5 : The two clips closely match in terms of visual quality and content   
Submit your rating for each sample through this form. Your feedback is important and greatly appreciated!

# E.2. Action Quality User Study Details

We conduct a second user study to specifically evaluate the gains in action quality of our model fine-tuned on data from the AutoExplore agent over the baseline. Observing that our model is particularly beneficial in scenarios with ambiguous initial frames, we use this user study to test this.

Table 13. Visual Fidelity of TA models.   

<table><tr><td rowspan="2">Model</td><td colspan="3">Basic Test Set</td></tr><tr><td>FID↓</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>Tokenizer-TA</td><td>12.10</td><td>39.53</td><td>0.97</td></tr><tr><td>LAM-TA</td><td>47.73</td><td>28.24</td><td>0.85</td></tr><tr><td>GenieRedux-TA</td><td>13.26</td><td>25.47</td><td>0.82</td></tr><tr><td>GenieRedux-G-TA</td><td>13.01</td><td>32.09</td><td>0.94</td></tr></table>

We use single initial frames with the agent in mid-jump. Participants interact with an interface shown in Fig.15. The user is shown pairs of synchronized videos, generated by the baseline and the exploration model (left/right position is randomized). Both videos depicted the same action — left, jump, or right—which was explicitly labeled in bold red below them. Participants were instructed to assess the quality of the action performed, disregarding any differences related purely to visual quality, and select one of the following options:

•Left: The left clip depicts the action more accurately.   
•No Difference: Both clips depict the action equally well.   
•Right: The right clip depicts the action more accurately.

# F. GenieRedux Evaluation on CoinRun Case Study

In this section, we qualitatively evaluate our Genie implementation - GenieRedux and its variant GenieRedux-G on the CoinRun case study. We also quantitatively and qualitatively study the effect of using data from a trained agent in the Coinrun environment (GenieRedux and GenieReduxG). We study the behavior and limitations of the model and compare our implementation with a concurrent one.

# F.1. CoinRun Case Study Details

We train the Tokenizer and the Dynamics module on CoinRun environment datasets, one obtained from a random agent, and one obtained from a trained agent using environment reward.

For training the agent for exploration, we enable velocity maps on CoinRun. These maps also need to be enabled for the agent during data collection. When evaluating models trained on different datasets (random agent vs. trained agent), to be fair, we exclude the velocity map regions by setting their pixels to black on all sets.

Throughout the training, we use a batch size of 84 and a patch size of 4 for all components. We use the Adam Optimizer with a linear warm-up and cosine annealing strategy.

We refer to the test set obtained from a random agent as Basic Test Set and to the one obtained from a trained agent as Diverse Test Set.

![](images/15.jpg)  
Figure 16. GenieRedux-G-TA Controllability Across Horizons.   
Figure 17. GenieRedux Quantitative Evaluation. We present a few sequences from the test set with predictions from GenieRedux. On the example at the top we show a successful jump action. On the example at the bottom we show a successful motion progression.

# F.2. Prediction Horizon Evaluations

We evaluate the controllability of our best model (at $5 0 \mathrm { k }$ iterations) over varying prediction horizons in Fig. 16. As expected, predictions become more challenging further into the future. The first prediction is also difficult due to insufficient motion information - we obtain O0. $4 \ \Delta _ { t } \mathrm { P S N R }$ for $t = 1$ . To address this issue, we provide the model with 4 frames and actions (predicting 10), and observe an improvement of our best model (GenieRedux-G-TA) from 34.79 PSNR (12.75 FID) in our results in the main paper to 38.31 PSNR (12.29 FID) on Diverse Test Set.

# F.3. GenieRedux-G Qualitative Evaluation

In Fig. 17 we show quantitative results demonstrating that GenieRedux-G can perform motion progression and action execution.

# F.4. GenieRedux-TA Qualitative Evaluation

In Fig. 18 we demonstrate that GenieRedux-TA is able to execute actions and complete motion. In Fig. 19 we show that the model is capable of executing all actions of the environment.

# F.5. Jafar Qualitative Comparison

We compare with Jafar [68] - a concurrent with our implementation of Genie (in JAX). We obtain and train their model as instructed. We train GenieRedux with Jafar's model parameters and like them separate LAM from Dynamics in training. The latter significantly worsened GenieRedux's action representation. Despite that, GenieRedux shows significantly better visual fidelity metrics, achieving 17.91 PSNR (46.12 FID), compared to Jafar's 12.66 PSNR (154.12 FID). GenieRedux does not exhibit Jafar's artifacts or the reported problematic "hole digging" behavior. Moreover, we observe that Jafar lacks causality, which we find problematic.

![](images/16.jpg)  
Figure 18. GenieRedux-TA Qualitative Comparison. We present a few samples from the test set with various actions. We demonstrate that GenieRedux-TA performs the actions correctly.

![](images/17.jpg)  
Figure 19. GenieRedux-TA Controllability. We show predictions for all environment actions of GenieRedux-TA.

In Fig. 20 we show Jafar's reconstruction of 10 frames into the future, given the first frame and a sequence of actions. The results are on the validation set after training. We observe an abundance of artifacts. We note that if we provide the images instead of providing the first frame, we get much less artifacts. This seems to hint that Jafar relies on future images to make predictions for the current frame, which might be an inherent problem of the model not being

![](images/18.jpg)  
Figure 20. Jafar Qualitative Results. The results are on the validation set. We give only a single image and actions and predict 15 frames in the future.

![](images/19.jpg)  
Figure 21. GenieRedux with Jafar's Parameters Qualitative Results. We show 15 frames into the future given actions and an initial frame of our model.

causal.

We additionally report test set results for Jafar - 0.48 SSIM and for GenieRedux (with Jafar parameters) - 0.62 SSIM.

In addition, we show the version of GenieRedux that we trained to match Jafar in Fig. 21. While it can be noticed that the model prefers inaction when encountering actions, it successfully progresses motion - e.g. moving a character through the air. We also notice fairly good visual quality.

# F.6. Additional GenieRedux-G-TA Qualitative Results and Limitations

We provide additional visuals of our best performing GenieRedux-G-TA in Fig. 22 and Fig. 23. We see that our model performs well under different actions and scenarios.

Next, we discuss the limitations of GenieRedux-G-TA and visualize the known cases in Fig. 24. One possible failure case occurs whenever the environment state or the actions suggest that a major exploration of the environment will unfold - for example, when falling down from midjump. As the agent is only given a single frame and cannot possibly know the layout of the level, it attempts to reconstruct something that is not guaranteed to be the actual level. Often, the agent exhibits uncertainty in these cases, as shown in the results.

Another possible weakness occurs whenever on the first frame a motion is already in progress - for example, in progress of jumping. In that case the model observes a single frame with the agent in the air and has no information about which direction the agent is heading - going up or going down. In that case, the model could exhibit uncertainty in the form of artifacts suggesting that the agent is both landing and jumping up, or alternatively not perform an action at all. This is a state from which the agent often recovers in a few steps. Still, we find that it can be avoided by providing more input frames to the model that can give motion information.

![](images/20.jpg)  
Figure 22. GenieRedux-G-TA Extra Qualitative Results. More sampled sequences from the test set, showing good match with the ground truth when enacting actions.

![](images/21.jpg)  
Figure 23. GenieRedux-G-TA Controllability Demonstration. We show that GenieRedux-G is able to perform all Coinrun environment actions.

![](images/22.jpg)  
Figure 24. GenieRedux-G-TA Limitations. Two failure cases of GenieRedux-G-TA - whenever a sizeable new unknown part of the environment is revealed; whenever an in-progress motion is ambiguous.

# References for Supplementary Material

[68] Timon Willi, Matthew Thomas Jackson, and Jakob Nicolaus Foerster. Jafar: An open-source genie reimplemention in jax. In First Workshop on Controllable Video Generation @ ICML 2024, 2024.