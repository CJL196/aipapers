# 计划、姿态与出发：面向开放世界的文本到运动生成

刘进鹏1\* 戴文勋1\* 王春宇2\* 程宜杰1 唐燕松1\* 童鑫2 1清华大学深圳国际研究生院 2微软亚洲研究院 {liujp22@mails., daiwx23@mails., cyj22@mails., tang.yansong@sz.}tsinghua.edu.cn {chnuwa, xtong}@microsoft.com

![](images/1.jpg)  
such as "Jump on one foot" and "Experiencing a profound sense of joy".

# 摘要

传统的文本到运动生成方法通常是在有限的文本-运动对上训练的，这使得它们很难推广到开放世界场景。一些工作使用CLIP模型对齐运动空间和文本空间，旨在使得能够从自然语言运动描述中生成运动。然而，它们仍然受到限制，只能生成有限且不真实的原地运动。为了解决这些问题，我们提出了一种称为PRO-Motion的分治框架，该框架由运动规划器、姿势扩散器和目标扩散器三个模块组成。运动规划器指导大型语言模型生成一系列脚本，描述目标运动中的关键姿势。与自然语言不同，这些脚本可以使用非常简单的文本模板描述所有可能的姿势。这显著降低了姿势扩散器的复杂性，它将脚本转换为姿势，为开放世界生成铺平了道路。最后，目标扩散器作为另一个扩散模型，估计所有姿势的全身平移和旋转，产生真实的运动。实验结果显示我们的方法优于其他对比方法，并展示了其从复杂的开放世界提示（如“体验深刻的快乐”）生成多样且真实的运动的能力。项目页面可访问https://moonsliu.github.io/Pro-Motion/。

# 1. 引言

文本到动作生成引起了越来越多的关注，这在虚拟现实、视频游戏和电影行业等许多应用中扮演着重要角色。以前的模型通常从成对的文本-动作数据中训练生成对抗网络（GANs）、变分自编码器（VAEs）和扩散模型，并且在文本提示类似于训练集中的情况下取得了合理的生成结果。图2（a）展示了这种范式。然而，它们难以处理超出现有数据集的开放世界文本提示，这是一个必须解决的核心挑战。否则，它们只能生成有限的“玩具式”动作。一些最近的工作提出增强模型处理超出训练数据的自然语言动作描述的能力。为此，它们利用预训练的视觉语言模型CLIP，以将训练动作中的姿势与动作描述对齐，希望能够从自然语言生成姿势。图2（b）对此进行了描述。然而，CLIP学习的文本空间与动作描述在很大程度上是不同的，使得将自然语言与动作连接的效果不佳。因此，这些方法仍然受到限制，仅能从有限的文本提示生成动作。此外，由于CLIP缺乏时间先验，这些方法在生成具有正确时间顺序的动作时存在困难。因此，它们只能生成不真实的原地动作。在本文中，我们提出了一个称为PRO-Motion的分治框架，包含“计划”、“姿态”和“执行”三个步骤，用于开放世界文本到动作生成，如图2（c）所示。在第一阶段“计划”中，我们引入了一个动作规划器，将复杂的自然语言动作描述转换为一系列姿态脚本，这些脚本描述身体部位之间的关系，遵循简单的模板，例如“男人直立，躯干垂直。他的左脚稍微离地，双臂放松在身体两侧。”这这一点通过利用大型语言模型中的运动常识来实现，并通过上下文示例进一步增强。重要的是要理解，尽管这些脚本简单并且限于一个小空间，但由于其组合性质，它们能够覆盖所有可能的姿势。动作规划器弥合了自然语言与姿势描述之间的差距，有效解决了分布外问题。得益于上述优点，在第二阶段“姿态”中，我们可以训练一个生成模型，通过仅使用相对较小的带标签数据集来实现脚本到姿态的生成。我们推测并证明该模型具有强大的泛化能力，能够包含广泛的姿态和脚本，因为新的姿势或脚本可以分解为多个熟悉的身体部位。在实现中，我们在一个公共数据集上开发了一个基于扩散的模型，称为姿态扩散器，它感知结构化姿势描述与身体部位之间的联系，从而产生多样和逼真的姿势。然后，我们进一步利用姿态规划模块选择关键姿势，考虑到相邻姿势的一致性以及文本与姿势之间的语义对齐。此外，在最后的“执行”阶段，我们观察到通过分析多个连续的身体姿势，我们可以同时预测平移和旋转。例如，在一个序列中，初始姿势为站立姿势，第二个姿势为左脚步进，第三个姿势为右脚步进，我们可以估计前方的平移。此外，在相邻的关键姿势之间学习插值是简单的，仅需少量的动作数据便可有效捕捉这样的先验。因此，我们设计了一个基于变换器的“执行扩散器”模块，以捕捉关键姿势之间的内在联系。为了验证我们提出的PRO-Motion的有效性，我们在多种数据集上进行了实验。定量和定性结果都显示出我们的方法相对于当前最先进的方法在开放世界文本到动作生成上的优势，并展示了其从复杂提示生成多样和逼真动作的能力，例如“单脚跳”和“体验深刻的快乐”。

# 2. 相关工作

文本到运动生成。基于现有的标注运动捕捉数据集，现有研究探索了各种用于文本驱动运动生成的生成模型，如 GANs、VAEs 和扩散模型。然而，这些方法受到对有限的文本-运动配对数据集的重度依赖的限制。为了解决这一问题，一些研究尝试利用当前强大的大规模预训练模型，即 CLIP，来克服数据限制，实现开放词汇运动生成。AvatarCLIP 针对给定文本描述生成运动，通过在线匹配和优化。然而，匹配无法生成超出分布的候选姿势，从而限制了生成复杂运动的能力，在线优化则耗时且不稳定。OOHMG 利用 CLIP 图像特征生成候选姿势，并通过掩膜学习进行运动生成。然而，由于 CLIP 缺乏时间先验，该方法无法捕捉动作的时间顺序，导致生成的运动不准确甚至完全相反。我们的方法采取不同的步骤，探讨大规模语言模型中人体姿势和运动的强大先验知识，以增强文本-运动对齐能力，并实现开放世界运动生成。

基于关键帧的运动生成。考虑到运动可以视为一系列姿态的组合，基于关键帧的运动生成引起了广泛的关注。运动预测涉及在提供一个或多个动画关键帧作为上下文时生成不受限制的运动延续。早期的研究[6, 13, 20, 23, 24, 38, 53, 60]采用循环神经网络（RNN）来建模人类运动序列，这得益于其在捕捉时间动态方面的强大能力。除了RNN，其他网络架构如卷积神经网络（CNN）[47, 61]和图卷积网络（GCN）[16]也被提出，以增强时间和运动关系的建模。变换器（Transformers）[4, 10]的出现进一步促进了运动序列中长距离依赖的建模。与我们的方法接近的是运动插值，它受到过去和未来关键帧的限制。早期的方法包括基于物理的方法[55, 74, 90]，这些方法涉及优化问题的求解，以及统计模型[11, 54, 89]。最近，一些基于神经网络的方法如RNN[28, 82, 99]、CNN[30, 40, 102]和变换器[18, 54, 67]在该领域占据了主导地位。与明确提供平移和旋转的运动插值方法不同，我们通过让模型学习相邻关键姿态之间的先验知识来实现平移和旋转的预测，以及姿态插值。

![](images/2.jpg)  
Figure 2. Comparison of different paradigms for text-to-motion generation. (a) Most existing models leverage the generative models [22, 33, 41] to construct the relationship between text and motion based on text-motion pairs. (b) Some methods render 3D poses to images and employ the image space of CLIP to align text with poses. Then they reconstruct the motion in the local dimension based on the poses. (c) Conversely, we decompose motion descriptions into structured pose descriptions. Then we generate poses based on corresponding pose descriptions. Finally, we reconstruct the motion in local and global dimensions. "Gen.", "Decomp.", "Desc.", "Rec." stand for "Generative model", "Decompose", "Pose Description" and "Reconstruction" respectively.

大型语言模型辅助视觉内容生成。近年来，由于在语言生成、推理、世界知识和上下文学习等任务中的卓越表现，大型语言模型（LLMs）在自然语言处理（NLP）和人工通用智能（AGI）领域引起了广泛关注。[7, 27] 将大型语言模型与基于扩散的生成模型[33, 72] 结合，旨在生成更高质量图像生成所需的提示。[35, 43] 利用大型语言模型规划视觉内容生成并识别关键动作，从而实现复杂动态视频的生成。另一些研究工作，包括[42, 48, 78, 91, 93]，提出将视觉API与语言模型集成，以便基于视觉信息进行决策或规划，进一步连接视觉和语言模型。与我们的方法接近的研究工作是利用LLMs作为有形智能体的规划者[2, 36, 37, 51, 79, 81, 92, 100]，以生成可在现实环境中执行的计划。不同于专注于机器人的研究，我们引入LLMs来操控运动的关键姿态生成，实现细粒度控制。

# 3. 方法

# 3.1. 基础知识

去噪扩散概率模型（DDPMs）。DDPMs，如[33, 56, 80]中详细描述，涉及两个马尔可夫链：一个将数据扩散到噪声的正向链，以及一个将噪声转化回数据的反向链。形式上，扩散正向过程迭代地向原始信号$x _ { 0 }$添加高斯噪声，以生成一系列噪声样本，即$\{ x _ { t } \} _ { t = 1 } ^ { T }$，其公式化如下：

$$
q ( x _ { t } | x _ { t - 1 } ) = N ( x _ { t } ; \sqrt { 1 - \beta _ { t } } x _ { t - 1 } , \beta _ { t } I ) ,
$$

其中 $q \big ( x _ { t } | x _ { t - 1 } \big )$ 表示条件概率分布，即在给定 $x _ { t - 1 }$ 的情况下的 $x _ { t }$ 的概率分布，$\beta _ { t } \in ( 0 , 1 )$ 表示时间步骤 $t$ 的噪声水平。当 $T$ 足够大时，我们可以假设 $x _ { T } \sim \mathcal { N } ( 0 , I )$。在条件合成设置中，去噪器旨在通过从先验分布 $\mathcal { N } ( 0 , \mathrm { I } )$ 采样随机噪声，然后通过反向扩散过程将其转换回 $x _ { 0 }$ 来建模分布 $p ( x _ { 0 } | c )$。在我们的设置中，我们遵循 [69]，直接预测原始样本，即 $\hat { x } _ { 0 } = f _ { \theta } ( x _ { t } , t , c )$，其简单目标为 [33]：

$$
\operatorname* { m i n } _ { \theta } \mathcal { L } = \mathbb { E } _ { x _ { 0 } , t , c } \left[ | | x _ { 0 } - f _ { \theta } ( x _ { t } , t , c ) | | _ { 2 } ^ { 2 } \right] .
$$

根据[33]，在每个时间步$t$，我们预测原始样本$\hat{x}_{0} = f_{\theta}(x_{t}, t, c)$并将其扩散回$x_{t-1}$。从$t = T$开始，迭代地从$p(x_{0} | c)$中进行采样，直到获得$x_{0}$。为了训练我们的扩散模型$f_{\theta}$，我们采用无分类器引导[32]，通过将$c$随机设置为$\emptyset$来处理10\%的样本。这使得$f_{\theta}$能够学习有条件和无条件分布，从而使得$f_{\theta}(x_{t}, t, \theta)$逼近$p(x_{0})$。我们使用系数$w$来平衡样本多样性与保真度之间的权衡：

![](images/3.jpg)

$$
f _ { \theta } ^ { w } ( x _ { t } , t , c ) = f _ { \theta } ( x _ { t } , t , \mathcal { O } ) + w \cdot ( f _ { \theta } ( x _ { t } , t , c ) - f _ { \theta } ( x _ { t } , t , \mathcal { O } ) ) .
$$

# 3.2. 动态规划器

如图3所示，当接收到用户提示，例如“随意跳舞”时，我们利用GPT-3.5 [57] 创建一个基于关于参与运动的身体部位的先验知识来描述关键姿势的计划。为了确保GPT-3.5在生成这些关键姿势的描述时保持运动的一致性，我们向GPT-3.5提供一个用户提示，表明预期的运动以及一个任务描述，以保证时间一致性和对各种运动属性（如每秒帧数（FPS）和帧数）的控制。除了管理整体运动外，我们还建立了五条基本规则，指导GPT-3.5描述关键姿势：（1）表征身体部位弯曲的程度，例如，使用‘左肘’这样的描述符，如‘完全弯曲’、‘轻微弯曲’、‘笔直’。 （2）分类不同身体部位之间的相对距离，例如，两只手为‘靠近’、‘肩宽’、‘张开’或‘远离’。 （3）描述不同身体部位的相对位置，例如，‘左髋’和‘左膝’，使用‘在后面’、‘在下面’或‘在右侧’等术语。 （4）确定身体部位的方向是‘垂直’还是‘水平’，例如，‘右膝’。 （5）识别身体部位是否与地面接触，例如‘左膝’和‘右脚’。此外，我们还为GPT-3.5提供一些参考姿势描述，以指导其生成过程。有关提示设计的更多详细信息，请参阅补充材料。

# 3.3. 姿态扩散器

在这一部分，我们介绍了我们的姿态扩散模块，旨在生成与运动规划器在第3.2节中提供的局部身体部位描述相一致的关键姿态。如图4（a）所示，我们利用了一种去噪扩散模型，该模型由$N$个相同层叠加而成。每层包含两个子块。第一个是残差块，其中结合了通过两层前馈网络处理的正弦时间嵌入生成的时间嵌入。第二个是交叉模态变换器块，它通过标准的交叉注意机制集成条件信号，即文本。中间的残差姿态特征作为查询向量，而从冻结的DistillBERT中提取的文本嵌入作为鍵和值向量。此外，我们随机掩蔽文本嵌入以实现无分类器学习。该模块使我们能够生成与姿态描述准确对齐的关键姿态。由于去噪扩散概率模型（DDPMs）的采样多样性，姿态扩散模块能够为每个姿态描述生成多个合理的姿态，我们引入了姿态规划模块，旨在从候选姿态中选择最合理的关键姿态。我们提出了两个目标：（1）最小化相邻帧之间的姿态差异；（2）最大化姿态与相应描述之间的相似性。为了实现这些目标，我们设计了两个编码器：一个由单层双向GRU组成的文本编码器$\Phi$，以及一个使用VPoser编码器的姿态编码器$\Theta$。这两个编码器产生L2范数嵌入，用于计算相似性。我们采用维特比算法搜索最合理的姿态路径。具体来说，假设我们有一组$F$个姿态描述，记为$\{ t _ { i } \} _ { i = 1 } ^ { F }$，并且对于每个姿态描述$t _ { i }$，我们有一组$L$个生成的候选姿态，表示为$\{ p _ { j } ^ { i } \} _ { j = 1 } ^ { K }$，其中第$i$帧$(i > 1)$的转移概率矩阵$A ^ { i }$用于第一个目标，其中相邻帧姿态的选择应优先考虑相似性较高的配对，如下所示：

$$
A _ { j k } ^ { i } = \frac { \exp { \left( \Theta { ( p _ { j } ^ { i - 1 } ) } ^ { T } \Theta ( p _ { k } ^ { i } ) \right) } } { \sum _ { l = 1 } ^ { L } \exp { \left( \Theta { ( p _ { j } ^ { i - 1 } ) } ^ { T } \Theta ( p _ { l } ^ { i } ) \right) } } .
$$

同样，第 $i$ 帧（$i \geq 1$）的发射概率矩阵 $E^{i}$ 需满足第二个目标，其中当前帧的关键姿态选择应优先考虑与以下描述具有较高匹配度的姿态：

$$
E _ { j } ^ { i } = \frac { \exp \Big ( \Phi ( t _ { i } ) ^ { T } \Theta ( p _ { j } ^ { i } ) \Big ) } { \sum _ { l = 1 } ^ { L } \exp \Big ( \Phi ( t _ { i } ) ^ { T } \Theta ( p _ { l } ^ { i } ) \Big ) } .
$$

算法的总体目标是生成一个姿态路径 $G = \{ g _ { i } \} _ { i = 1 } ^ { F }$，以最大化联合概率：

$$
\underset { G } { \arg \operatorname* { m a x } } P ( G ) = \prod _ { i = 1 } ^ { F } P ( g _ { i } | g _ { i - 1 } ) = E _ { g _ { 1 } } ^ { 1 } \prod _ { i = 2 } ^ { F } E _ { g _ { i } } ^ { i } A _ { g _ { i - 1 } g _ { i } } ^ { i } .
$$

# 3.4. Go-Diffuser

为了插值和预测第3.3节获得的关键姿态的全局信息，如平移和旋转，我们在这一部分引入了Go-Diffuser模块。我们的模块基于扩散模型，如图4(b)所示。由于变压器结构在运动生成领域已被证明是有效的，我们在实现中采用了变压器编码器架构。该模块接收一个时间步$t$下的带噪运动序列$\boldsymbol { x } _ { t } ^ { 1 : N }$，以及$t$本身和条件即关键姿态$\{ p _ { g _ { i } } ^ { i } \} _ { i = 1 } ^ { F }$，为了更好地捕捉全局信息，我们将它们视为离散的词元，而不是统一的特征。在实践中，关键姿态被投影并随机遮挡以进行分类学习。带噪输入$\boldsymbol { x } _ { t } ^ { 1 : N }$被投影并整合位置信息。变压器编码器的输出被投影回原始运动维度，生成预测的运动序列$\hat { x } _ { 0 } ^ { 1 : N }$。该模块使我们能够平滑地插值关键姿态并将全局属性分配给动作。

# 4. 实验

# 4.1. 数据集

在我们的实验中，我们利用来自AMASS、PoseScript、Motion-X和HumanML3D的姿势数据、运动数据和文本描述。AMASS统一了多种基于光学标记的动作捕捉数据集，提供超过40小时的运动数据，但没有文本描述。PoseScript由从AMASS提取的静态3D人体姿势组成，结合细粒度的语义人写注释描述（PoseScript-H）和自动生成的标题（PoseScript-A）。HumanML3D是一个广泛使用的运动语言数据集，为来自AMASS的运动数据提供标题。Motion-X是一个大规模的3D表现性全身运动数据集，具有详细的语言描述。我们从Motion-X中选择两个子集进行开放世界的文本到动作生成实验。具体而言，我们采用句子转换器计算IDEA-400中文本描述的相似度，这是Motion-X内一个高质量的运动语言子集，并与HumanML3D中的文本描述进行比较。我们筛选出相似度高于指定阈值α（例如，0.45）的配对，从而得到一个包含368对文本-动作的运动语言数据集，作为我们的第一测试数据集。此外，我们选择Motion的功夫子集作为我们的第二测试数据集。

![](images/4.jpg)  
Figure 4. Illustration of our Dual-Diffusion model. (a) PostureDiffuser module is designed to predict the original pose conditioned by the pose description. The model consists of $N$ identical layers, with each layer featuring a residual block for incorporating time step information and a cross-modal transformer block for integrating the condition text. (b) $G o$ -Diffuser module serves the function of obtaining motion with translation and rotation from discrete key poses without global information. In this module, the key poses obtained from Sec. 3.3 are regarded as independent tokens. We perform attention operations [87] between these tokens and noised motion independently, which can significantly improve the perception ability between every condition pose and motion sequence.

抱歉，我无法处理该文本。请提供清晰的内容以供翻译。

<table><tr><td rowspan="2"></td><td colspan="4">Text-motion</td><td rowspan="2">FID ↓</td><td rowspan="2">MultiModal Dist ↓</td><td rowspan="2">Smooth →</td></tr><tr><td>R@10 ↑</td><td>R@20 ↑</td><td>R@30 ↑</td><td>MedR ↓</td></tr><tr><td>&quot;test on ood368 subset&quot;</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MDM [85]</td><td>17.81</td><td>34.06</td><td>48.75</td><td>31.20</td><td>3.500541</td><td>2.613644</td><td>0.000114</td></tr><tr><td>MotionCLIP [84]</td><td>16.25</td><td>35.62</td><td>52.81</td><td>28.90</td><td>2.227522</td><td>2.288905</td><td>0.000073</td></tr><tr><td>Codebook+Interpolation [34]</td><td>15.62</td><td>31.25</td><td>46.56</td><td>32.80</td><td>4.084785</td><td>2.516041</td><td>0.000146</td></tr><tr><td>AvatarCLIP [34]</td><td>15.31</td><td>31.56</td><td>47.19</td><td>32.60</td><td>4.181952</td><td>2.449695</td><td>0.000146</td></tr><tr><td>OOHMG [44]</td><td>15.62</td><td>34.06</td><td>48.75</td><td>29.80</td><td>3.982753</td><td>2.149275</td><td>0.000758</td></tr><tr><td>Ours</td><td>20.25</td><td>36.56</td><td>53.14</td><td>26.10</td><td>1.488678</td><td>1.534521</td><td>0.001312</td></tr><tr><td>&quot;test on kungfu subset&quot;</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MDM [85]</td><td>12.50</td><td>29.69</td><td>42.19</td><td>37.50</td><td>12.060187</td><td>3.725436</td><td>0.000735</td></tr><tr><td>MotionCLIP [84]</td><td>15.62</td><td>29.69</td><td>46.88</td><td>32.50</td><td>17.414746</td><td>4.297871</td><td>0.000123</td></tr><tr><td>Codebook+Interpolation [34]</td><td>10.94</td><td>20.31</td><td>29.69</td><td>37.50</td><td>2.521690</td><td>2.764137</td><td>0.000138</td></tr><tr><td>AvatarCLIP [34]</td><td>15.62</td><td>31.25</td><td>46.88</td><td>32.50</td><td>1.966764</td><td>2.497678</td><td>0.000171</td></tr><tr><td>OOHMG [44]</td><td>14.06</td><td>32.81</td><td>48.44</td><td>32.50</td><td>4.904853</td><td>2.471666</td><td>0.000847</td></tr><tr><td>Ours</td><td>20.31</td><td>34.38</td><td>50.00</td><td>31.00</td><td>4.124218</td><td>2.374380</td><td>0.001559</td></tr></table>

# 4.2. 开放世界运动生成

在本节中，我们首先介绍监督学习基线[84, 85]、开放世界基线[34, 44, 84]以及评估指标[26, 62]。接着，我们讨论与这些基线的对比实验结果。MDM基线：MDM[85]采用一种利用扩散模型的监督学习方法。然而，当应用于新的设置时，即开放词汇运动生成，其性能常常退化。我们在带有人体ML3D标注的AMASS数据的SMPL-H[49, 59, 73]版本上进行了训练。MotionCLIP基线：MotionCLIP[84]是一种在AMAS数据上进行训练的监督开放词汇方法，并且使用BABEL[66]标注。我们使用作者提供的预训练模型，并在开放词汇设置下测试模型的性能。Codebook+插值基线：在姿态生成阶段，我们利用VPoserCodebook[59]作为姿态生成器，并在姿态生成阶段选择最相似的姿态。在运动生成阶段，我们仅使用插值方法生成运动。AvatarCLIP基线：AvatarCLIP[34]是一种基于优化的方法。它还包含一个通过匹配进行文本到姿态的阶段，并利用匹配的姿态在训练于AMASS数据集的运动变分自编码器[41]的潜在空间中搜索最相关的运动。评估指标采用自[26, 62]，包括R精度、Frechet嵌入距离（FID）和多模态距离。为了进行定量评估，使用对比损失训练运动特征提取器和文本特征提取器，以生成几何上接近的特征向量用于匹配的文本-运动对。有关上述指标以及文本和运动特征提取器设计的更多细节，请参考补充材料。考虑R精度：对于每个生成的运动，其真实标注文本描述和从测试数据集中随机选择的不匹配描述形成一个描述池，然后计算并排名运动特征与池中每个描述的文本特征之间的欧几里得距离。同时，多模态距离计算为每个生成运动的运动特征与其在测试数据集中对应描述的文本特征之间的平均欧几里得距离。

如图5所示，对于动作描述“弯腰”，基于监督的方法MDM [85]生成了合理的动作序列，展示了从直立到俯卧的过程。而OOHMG /citeoohmg则涉及更多的腿部动作，这与描述并不相符。其他利用CLIP [68]的方法，如MotionCLIP [84]、Codebook+Interpolation [34]和AvatarCLIP [34]错误地判断了关键姿势序列，它们的脊椎首先弯曲到一定程度，然后弯曲程度减小。由于CLIP缺乏时间先验，第一阶段生成的姿势序列没有按照正确的顺序进行。此外，对于动作描述“埋头哭泣，最后蹲下”，基于监督学习范式的MDM [85]常常在类似情况下失败，并且无法生成未见过的动作。由于动作描述和图像描述之间的差距，通过CLIP的语言空间匹配文本和动作并不有效。它们在处理细节和旋转时显得力不从心。平均全局表示了身体关节和全局平移的表现。

![](images/5.jpg)  
Figure 5. Comparation of our methods with previous text-to-motion generation methods.

<table><tr><td rowspan="2">Methods</td><td colspan="4">Average Positional Error ↓</td><td colspan="4">Average Variance Error ↓</td></tr><tr><td>root joint</td><td>global traj.</td><td>mean local</td><td>mean global</td><td>root joint</td><td>global traj.</td><td>mean local</td><td>mean global</td></tr><tr><td>Regression</td><td>5.878673</td><td>5.53344</td><td>0.642252</td><td>5.919954</td><td>35.387340</td><td>35.386562</td><td>0.147606</td><td>35.483219</td></tr><tr><td>Baseline[62]</td><td>0.384152</td><td>0.373394</td><td>0.183978</td><td>0.469322</td><td>0.114308</td><td>0.113845</td><td>0.015207</td><td>0.126049</td></tr><tr><td>Ours</td><td>0.365327</td><td>0.354685</td><td>0.128763</td><td>0.418265</td><td>0.111131</td><td>0.110855</td><td>0.008708</td><td>0.118334</td></tr></table>

精确的运动描述。在表1中，定量指标展示了我们模型在语义一致性和运动合理性方面优于其他方法的优势。

# 4.3. 消融研究

姿态扩散器与零样本开放词汇文本到姿态生成方法 [34, 44] 相比，我们首先利用大型语言模型将姿态描述翻译为局部身体部位描述，然后将其输入到第三节 3.3 的姿态生成器中，以精确生成姿态。如图 6 所示，匹配方法在姿态生成结果上优于优化和VPoser优化，表明直接使用CLIP进行匹配比通过复杂流程的优化更有效。然而，“匹配”未能为多样化文本生成更精确的姿态，展现出在生成姿态时保留文本信息的限制。例如，在“哭”和“祈祷”等情况下，“匹配”对于意义不同的文本生成了相同的姿态。当生成需要对身体部位进行更精确控制的姿态时，如“跳华尔兹”或“踢足球”，无论是OOHMG还是“匹配”都无法达到令人满意的结果。相反，通过使用大型语言模型精确描述预期姿态，我们实现了对姿态生成的准确控制，从而能够更有效地进行基于文本的开放世界姿态合成。

Go-Diffuser。这是我们所知首个以零样本本地姿态驱动方式预测运动空间信息的尝试，我们已经开发出适当的基线方法来评估位移和旋转的感知能力。根据我们的观察，通过分析相邻姿态之间身体部位的变化，可以估计人们的位移和旋转。我们将从关键姿态估计全局信息的过程视为重建任务。因此，我们设计了一个简单的基于多层感知器（MLP）的网络来首先实现这一目标。变换器结构在运动生成领域已被证明是正确的，因此我们通过提取姿态序列特征作为注入扩散模型的条件来设计基线方法。如图7所示，从左上角的四幅图像可以观察到，简单的MLP网络在某种程度上能够预测运动位移信息。正如左下角图像所示，使用现有运动编码器提取姿态序列作为特征可能会忽略姿态序列内部关系，从而导致细节混淆。在右上角图像中，我们的方法在捕捉膝关节屈伸等细节方面表现出更好的保真度。此外，右下角的四幅图像显示，当处理相似的相邻姿态时，我们的模型展现出更精细的感知能力，从而为数字头像赋予适当的运动趋势。如表2所示，我们的方法在全局轨迹、旋转和局部姿态关节的平均位移误差和平均方差误差上都取得了最先进的结果。

![](images/6.jpg)  
Figure 6. Comparison of our method with previous text-to-pose generation methods.

![](images/7.jpg)

# 5. 结论

在本文中，我们介绍了PRO-Motion，一个旨在解决开放世界文本到动作生成任务的模型。它由三个模块组成：动作规划器、姿态扩散器和运动扩散器。动作规划器指导大型语言模型生成一系列脚本，描述目标动作中的关键姿态。姿态扩散器将脚本转换为姿态，为开放世界生成铺平道路。最后，运动扩散器估计所有姿态的全身平移和旋转，产生多样且逼真的动作。实验结果表明，与其他方法相比，我们的方法具有明显的优势。

# References

[1] Hyemin Ahn, Timothy Ha, Yunho Choi, Hwiyeon Yoo, and Songhwai Oh. Text2action: Generative adversarial synthesis from language to action. In ICRA, pages 59155920, 2018. 2   
[2] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, et al. Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691, 2022. 3   
[3] Chaitanya Ahuja and Louis-Philippe Morency. Language2pose: Natural language grounded pose forecasting. In 3DV, 2019. 16   
[4] Emre Aksan, Manuel Kaufmann, Peng Cao, and Otmar Hilliges. A spatio-temporal transformer for 3d human motion prediction. In 3DV, pages 565574, 2021. 3   
[5] Nikos Athanasiou, Mathis Petrovich, Michael J Black, and Gül Varol. Teach: Temporal action composition for 3d humans. In 3DV, pages 414423, 2022. 2   
[6] Emad Barsoum, John Kender, and Zicheng Liu. Hp-gan: Probabilistic 3d human motion prediction via gan. In CVPRW, pages 14181427, 2018. 3   
[7] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, pages 1839218402, 2023. 3   
[8] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. NIPS, pages 18771901, 2020. 3   
[9] Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712, 2023. 3   
10] Yujun Cai, Lin Huang, Yiwei Wang, Tat-Jen Cham, Jianfei Cai, Junsong Yuan, Jun Liu, Xu Yang, Yiheng Zhu, Xiaohui Shen, et al. Learning progressive joint propagation for human motion prediction. In ECCV, pages 226242, 2020. 3   
11] Jinxiang Chai and Jessica K Hodgins. Constraint-based motion optimization using a statistical dynamic model. In SIGGRAPH, pages 8es. 2007. 3   
12] Xin Chen, Biao Jiang, Wen Liu, Zilong Huang, Bin Fu, Tao Chen, and Gang Yu. Executing your commands via motion diffusion in latent space. In CVPR, pages 1800018010, 2023.2   
[13] Hsu-kuang Chiu, Ehsan Adeli, Borui Wang, De-An Huang, and Juan Carlos Niebles. Action-agnostic human pose forecasting. In WACV, pages 14231432, 2019. 3   
[14] Kyunghyun Cho, Bart Van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078, 2014. 4   
[15] Rishabh Dabral, Muhammad Hamza Mughal, Vladislav Golyanik, and Christian Theobalt. Mofusion: A framework for denoising-diffusion-based motion synthesis. In CVPR, pages 97609770, 2023. 2   
[16] Lingwei Dang, Yongwei Nie, Chengjiang Long, Qing Zhang, and Guiqing Li. Msr-gcn: Multi-scale residual graph convolution networks for human motion prediction. In ICCV, pages 1146711476, 2021. 3   
[17] Ginger Delmas, Philippe Weinzaepfel, Thomas Lucas, Francesc Moreno-Noguer, and Grégory Rogez. Posescript: 3d human poses from natural language. In ECCV, pages 346362, 2022. 2, 5, 13, 14   
[18] Yinglin Duan, Tianyang Shi, Zhengxia Zou, Yenan Lin, Zhehui Qian, Bohan Zhang, and Yi Yuan. Single-shot motion completion with transformer. arXiv preprint arXiv:2103.00776, 2021. 3   
[19] G David Forney. The viterbi algorithm. IEEE, 61(3):268 278, 1973. 4   
[20] Katerina Fragkiadaki, Sergey Levine, Panna Felsen, and Jitendra Malik. Recurrent network models for human dynamics. In ICCV, pages 43464354, 2015. 3   
[21] Anindita Ghosh, Noshaba Cheema, Cennet Oguz, Christian Theobalt, and Philipp Slusallek. Synthesis of compositional animations from textual descriptions. In ICCV, 2021. 16   
[22] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139144, 2020. 3   
[23] Anand Gopalakrishnan, Ankur Mali, Dan Kifer, Lee Giles, and Alexander G Ororbia. A neural temporal model for human motion prediction. In CVPR, pages 1211612125, 2019. 3   
[24] Liang-Yan Gui, Yu-Xiong Wang, Xiaodan Liang, and José MF Moura. Adversarial geometry-aware human motion prediction. In ECCV, pages 786803, 2018. 3   
[25] Chuan Guo, Xinxin Zuo, Sen Wang, Shihao Zou, Qingyao Sun, Annan Deng, Minglun Gong, and Li Cheng. Action2motion: Conditioned generation of 3d human motions. In ACM MM, pages 20212029, 2020. 2   
[26] Chuan Guo, Shihao Zou, Xinxin Zuo, Sen Wang, Wei Ji, Xingyu Li, and Li Cheng. Generating diverse and natural 3d human motions from text. In CVPR, pages 51525161, 2022. 2, 5, 6   
[27] Yaru Hao, Zewen Chi, Li Dong, and Furu Wei. Optimizing prompts for text-to-image generation. arXiv preprint arXiv:2212.09611, 2022. 3 [28] Félix G Harvey and Christopher Pal. Recurrent transition networks for character locomotion. In SIGGRAPH Asia, pages 14. 2018. 3 [29] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016. 14 [30] Alejandro Hernandez, Jurgen Gall, and Francesc MorenoNoguer. Human motion prediction via spatio-temporal inpainting. In ICCV, pages 71347143, 2019. 3 [31] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. NeurIPS, 2017. 16 [32] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 3 [33] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NIPS, 33:68406851, 2020. 3, 4,   
14 [34] Fangzhou Hong, Mingyuan Zhang, Liang Pan, Zhongang Cai, Lei Yang, and Ziwei Liu. Avatarclip: Zero-shot textdriven generation and animation of 3d avatars. arXiv preprint arXiv:2205.08535, 2022. 2, 6, 7 [35] Susung Hong, Junyoung Seo, Sunghwan Hong, Heeseong Shin, and Seungryong Kim. Large language models are frame-level directors for zero-shot text-to-video generation. arXiv preprint arXiv:2305.14330, 2023. 3 [36] Chenguang Huang, Oier Mees, Andy Zeng, and Wolfram Burgard. Visual language maps for robot navigation. In ICRA, pages 1060810615, 2023. 3 [37] Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. In ICML, pages 91189147, 2022. 3 [38] Ashesh Jain, Amir R Zamir, Silvio Savarese, and Ashutosh Saxena. Structural-rnn: Deep learning on spatio-temporal graphs. In CVPR, pages 53085317, 2016. 3 [39] Yanli Ji, Feixiang Xu, Yang Yang, Fumin Shen, Heng Tao Shen, and Wei-Shi Zheng. A large-scale rgb-d database for arbitrary-view human action recognition. In ACM MM, pages 15101518, 2018. 2 [40] Manuel Kaufmann, Emre Aksan, Jie Song, Fabrizio Pece, Remo Ziegler, and Otmar Hilliges. Convolutional autoencoders for human motion infilling. In 3DV, pages 918927,   
2020. 3 [41] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 3,   
6 [42] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 3 [43] Han Lin, Abhay Zala, Jaemin Cho, and Mohit Bansal. Videodirectorgpt: Consistent multi-scene video generation via llm-guided planning. arXiv preprint arXiv:2309.15091,   
2023. 3 [44] Junfan Lin, Jianlong Chang, Lingbo Liu, Guanbin Li, Liang Lin, Qi Tian, and Chang-wen Chen. Being comes from not-being: Open-vocabulary text-to-motion generation with wordless training. In CVPR, pages 2322223231, 2023. 2, 6, 7   
[45] Jing Lin, Ailing Zeng, Shunlin Lu, Yuanhao Cai, Ruimao Zhang, Haoqian Wang, and Lei Zhang. Motion-x: A largescale 3d expressive whole-body human motion dataset. NeurIPS, 2023. 2, 5, 6, 14   
[46] Xiao Lin and Mohamed R Amer. Human motion modeling using dvgans. arXiv preprint arXiv:1804.10652, 2018. 2   
[47] Xiaoli Liu, Jianqin Yin, Jin Liu, Pengxiang Ding, Jun Liu, and Huaping Liu. Trajectorycnn: a new spatio-temporal feature learning network for human motion prediction. TCSVT, 31(6):21332146, 2020. 3   
[48] Zhaoyang Liu, Yinan He, Wenhai Wang, Weiyun Wang, Yi Wang, Shoufa Chen, Qinglong Zhang, Yang Yang, Qingyun Li, Jiashuo Yu, et al. Internchat: Solving vision-centric tasks by interacting with chatbots beyond language. arXiv preprint arXiv:2305.05662, 2023. 3   
[49] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. SMPL: A skinned multiperson linear model. SIGGRAPH Asia, 2015. 6, 13   
[50] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 14   
[51] Yujie Lu, Weixi Feng, Wanrong Zhu, Wenda Xu, Xin Eric Wang, Miguel Eckstein, and William Yang Wang. Neurosymbolic procedural planning with commonsense prompting. arXiv preprint arXiv:2206.02928, 2022. 3   
[52] Naureen Mahmood, Nima Ghorbani, Nikolaus F Troje, Gerard Pons-Moll, and Michael J Black. Amass: Archive of motion capture as surface shapes. In ICCV, pages 5442 5451, 2019. 2, 5, 7   
[53] Julieta Martinez, Michael J Black, and Javier Romero. On human motion prediction using recurrent neural networks. In CVPR, pages 28912900, 2017. 3   
[54] Jianyuan Min, Yen-Lin Chen, and Jinxiang Chai. Interactive generation of human animation with deformable motion models. TOG, 29(1):112, 2009. 3   
[55] J Thomas Ngo and Joe Marks. Spacetime constraints revisited. In SIGGRAPH, pages 343350, 1993. 3   
[56] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In ICML, pages 81628171, 2021. 3, 4   
[57] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. NIPS, pages 2773027744, 2022. 3, 4   
[58] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed AA Osman, Dimitrios Tzionas, and Michael J Black. Expressive body capture: 3d hands, face, and body from a single image. In CVPR, pages 10975 10985, 2019. 4   
[59] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, and Michael J. Black. Expressive body capture: 3d hands, face, and body from a single image. In CVPR, 2019. 6   
[60] Dario Pavllo, David Grangier, and Michael Auli. Quaternet: A quaternion-based recurrent model for human motion. arXiv nrenrint arXiv:1805 06485 20183   
[61] Dario Pavllo, Christoph Feichtenhofer, Michael Auli, and David Grangier. Modeling human motion with quaternionbased neural networks. IJCV, 128:855872, 2020. 3   
[62] Mathis Petrovich, Michael J Black, and Gül Varol. Actionconditioned 3d human motion synthesis with transformer vae. In ICCV, pages 1098510995, 2021. 2, 5, 6, 7, 8   
[63] Mathis Petrovich, Michael J Black, and Gül Varol. Temos: Generating diverse human motions from textual descriptions. In ECCV, pages 480497, 2022. 2, 8, 13, 16   
[64] Mathis Petrovich, Michael J. Black, and Gül Varol. TMR: Text-to-motion retrieval using contrastive 3D human motion synthesis. In ICCV, 2023. 5   
[65] Matthias Plappert, Christian Mandery, and Tamim Asfour. The kit motion-language dataset. Big data, 4(4):236252, 2016.2   
[66] Abhinanda R Punnakkal, Arjun Chandrasekaran, Nikos Athanasiou, Alejandra Quiros-Ramirez, and Michael J Black. Babel: Bodies, action and behavior with english labels. In CVPR, pages 722731, 2021. 2, 6   
[  , Yuy Zeg nd un Zhou nvia two-stage transformers. T0G, 41(6):116, 2022. 3   
[68] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 87488763, 2021. 2, 6   
[69] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022. 3, 4   
[70] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing, 2019. 5, 14   
[71] Nils Reimers and Iryna Gurevych. Making monolingual sentence embeddings multilingual using knowledge distillation. In EMNLP, 2020. 5, 14   
[72] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 1068410695, 2022. 3   
[73] Javier Romero, Dimitrios Tzionas, and Michael J. Black. Embodied hands: Modeling and capturing hands and bodies together. SIGGRAPH Asia, 36(6), 2017. 6, 13   
[74] Charles Rose, Brian Guenter, Bobby Bodenheimer, and Michael F Cohen. Efficient generation of motion transitions using spacetime constraints. In SIGGRAPH, pages 147154, 1996. 3   
[75] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108, 2019. 4   
[76] Yonatan Shafir, Guy Tevet, Roy Kapon, and Amit H Bermano. Human motion diffusion as a generative prior. arXiv preprint arXiv:2303.01418, 2023. 2   
[77] Amir Shahroudy, Jun Liu, Tian-Tsong Ng, and Gang Wang. Ntu rgb $^ +$ A la a atat  u iviy analvsis. In CVPR. pages 10101019. 2016. 2   
[78] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving ai tasks with chatgpt and its friends in huggingface. arXiv preprint arXiv:2303.17580, 2023. 3   
[79] Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, and Animesh Garg. Progprompt: Generating situated robot task plans using large language models. In ICRA, pages 1152311530, 2023. 3   
[80] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, pages 2256 2265, 2015. 3, 4   
[81] Chan Hee Song, Jiaman Wu, Clayton Washington, Brian M Sadler, Wei-Lun Chao, and Yu Su. Llm-planner: Few-shot grounded planning for embodied agents with large language models. In ICCV, pages 29983009, 2023. 3   
[82] Xiangjun Tang, He Wang, Bo Hu, Xu Gong, Ruifan Yi, Qilong Kou, and Xiaogang Jin. Real-time controllable motion transition for characters. T0G, 41(4):110, 2022. 3   
[83] Yansong Tang, Jinpeng Liu, Aoyang Liu, Bin Yang, Wenxun Dai, Yongming Rao, Jiwen Lu, Jie Zhou, and Xiu Li. Flag3d: A 3d fitness activity dataset with language instruction. In CVPR, pages 2210622117, 2023. 2   
[84] Guy Tevet, Brian Gordon, Amir Hertz, Amit H Bermano, and Daniel Cohen-Or. Motionclip: Exposing human motion generation to clip space. In ECCV, pages 358374, 2022. 2, 6   
[85] Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, and Amit H Bermano. Human motion diffusion model. arXiv preprint arXiv:2209.14916, 2022. 2, 5,6   
[86] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 3   
[87] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. NIPS, 2017. 2, 4, 5   
[88] Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, and James Hays. Composing text and image for image retrieval-an empirical odyssey. In CVPR, pages 64396448, 2019. 14   
[89] Jack M Wang, David J Fleet, and Aaron Hertzmann. Gaussian process dynamical models for human motion. TPAMI, 30(2):283298, 2007. 3   
[90] Andrew Witkin and Michael Kass. Spacetime constraints. SIGGRAPH, 22(4):159168, 1988. 3   
[91] Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. Visual chatgpt: Talking, drawing and editing with visual foundation models. arXiv preprint arXiv:2303.04671, 2023. 3   
[92] Zeqi Xiao, Tai Wang, Jingbo Wang, Jinkun Cao, Wenwei Zhang, Bo Dai, Dahua Lin, and Jiangmiao Pang. Unified human-scene interaction via prompted chain-of-contacts. arVi nranrint arVi.2200 07018, 20222   
[93] Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381, 2023. 3   
[94] Ye Yuan, Jiaming Song, Umar Iqbal, Arash Vahdat, and Jan Kautz. Physdiff: Physics-guided human motion diffusion model. In ICCV, 2023. 2   
[95] Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, et al. Glm-130b: An open bilingual pre-trained model. arXiv preprint arXiv:2210.02414, 2022. 3   
[96] Mingyuan Zhang, Zhongang Cai, Liang Pan, Fangzhou Hong, Xinying Guo, Lei Yang, and Ziwei Liu. Motiondiffuse: Text-driven human motion generation with diffusion model. arXiv preprint arXiv:2208.15001, 2022. 2   
[97] Mingyuan Zhang, Xinying Guo, Liang Pan, Zhongang Cai, Fangzhou Hong, Huirong Li, Lei Yang, and Ziwei Liu. Remodiffuse: Retrieval-augmented motion diffusion model. arXiv preprint arXiv:2304.01116, 2023. 2   
[98] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022. 3   
[99] Xinyi Zhang and Michiel van de Panne. Data-driven autocompletion for keyframe animation. In SIGGRAPH, pages 111, 2018. 3   
[100] K Zheng, K Zhou, J Gu, Y Fan, J Wang, Z Li, X He, and XE Wang. Jarvis: a neuro-symbolic commonsense reasoning framework for conversational embodied agents. arXiv preprint arXiv:2208.13266, 2022. 3   
[101] Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. On the continuity of rotation representations in neural networks. In CVPR, 2019. 13   
[102] Yi Zhou, Jingwan Lu, Connelly Barnes, Jimei Yang, Sitao Xiang, et al. Generative tweening: Long-term inbetweening of 3d human motions. arXiv preprint arXiv:2005.08891, 2020.3

# Appendix

In this supplementary material, we provide additional details and experiments not included in the main paper due to limitations in space.

•Appendix A: Qualitative results of precise control for motion generation in our formulation.   
Appendix B: Details of the motion representations.   
•Appendix C: Details of the prompt engineering and the entire system message template.   
•Appendix D: Details of our dual-diffusion models.   
•Appendix E: Details of the evaluation metrics.

Note: Blue characters denote the main paper's reference.

# A. Precise Pose Control

As shown in Fig. 8, our Posture-Diffuser module can accurately generate poses from edited pose descriptions, which becomes particularly beneficial when users are not content with the key pose descriptions produced by the Motion Planner. In such cases, users have the option to manually edit the description of specific body parts to gain precise control over the generated poses. For example, in the Control $\# 1$ of Pose Description $\# 1$ , we replace the description of "their left arm is behind the right with their right elbow bent at 90 degrees." with "the left hand is beside the right arm and the right hand is to the left of the left hand". It's obvious the position of the hands changes correctly. As well, when we delete the description of "The left foot is stretched backwards and is behind the right foot" in the Control $\# 1$ of Pose Description $\# 2$ , the position of the left foot changes correctly. Furthermore, we could find that although the scripts are simple and limited to a small space, they are expressive to cover all possible postures due to their compositional nature.

# B. Motion Representations

Skinned Multi-Person Linear model (SMPL)[49][73] is a skinned vertex-based model that accurately represents various body shapes in natural human poses. It's widely utilized in the human motion generation field. So we adopted the SMPL-based human model as well. SMPL begins with an artist-created mesh with $\Nu = 6 8 9 0$ vertices and $\mathrm { K } = 2 3$ joints. The mesh has the same topology for men and women, spatially varying resolution, a clean quad structure, a segmentation into parts, initial blend weights, and a skeletal rig. We follow the SMPL-based representation of TEMOS [63] and construct the feature vector for SMPL data.

Translation. It consists of two parts. The first part is the velocity of the root joint in the global coordinate system. The second part is the position of the root joint for the Z axis. The dimension of the translation is 3. Root Orientation. It contains one rotation, and we utilize the 6D continuous representation [101] to store it. So the dimension of the root orientation is $1 \ast 6 = 6$ .

•Pose Body. The motion data is from the SMPL-H [49, 73] version of AMASS. Because we focus on the movement of the human body, we removed all rotations on the hands, resulting in 21 rotations). The same as root orientation, we utilize the 6D continuous representation [101]. So the dimension of the pose body is $2 1 ^ { * } 6 = 1 2 6$ .

So the final dimension of the feature is $3 + 1 * 6 + 2 1 * 6 =$ 135. In our experiments, we remove all the motion sequences that are less than 64 frames, and the motion sequences longer than 64 frames are processed as 64 frames. The data sample will be in R64\*135.

# read motion 2 motion_parms $= \quad \left\{ \begin{array} { r l r l } \end{array} \right.$ 3 'trans': motion[:, :3], # controls the global body position 4 'root_orient': motion[:, 3 : $3 + 6 ]$ , # controls the global root orientation 5 'pose_body': motion[:, 9 $= \begin{array} { l l l l l } { { 9 } } & { { + } } & { { 2 1 } } & { { \star } } & { { 6 } } \end{array} ] ,$ # controls the body 6

When we perform the translation and rotation prediction, we only know the local attributes of the adjacent poses, but not their global location information. So we must normalize the pose position. Following [63], the translation of neighboring poses is subtracted and represented by the instantaneous velocity as the translation attribute of the current frame.

# extract the root gravity axis   
2 # for smpl it is the last coordinate   
3 root $=$ trans[..., 2]   
4 trajectory $=$ trans[..., [0, 1]]   
5   
6 # Comoute the difference of trajectory (for X and Y axis)   
7 vel_trajectory $=$ torch.diff(trajectory, dim $= - 2$   
$8 \# 0$ for the first one ${ } = > { }$ keep the dimentionality   
9 vel_trajectory $=$ torch.cat( $( \mathrm { ~ 0 ~ } \star$ vel_trajectory [..., [0], :], vel_trajectory), dim $_ { 1 = - 2 }$

# C. Prompt Design

Fig. 9 illustrates the complete prompt used by our Motion Planner in Sec. 3.2. We first define the overall objective and task requirements and then establish five fundamental rules of body parts including bending degree, relative distance, relative position, orientation, and ground contact. Additionally, we specify the body parts to which each rule applies. Through this rule-based approach, we can guide the LLM to generate precise key pose descriptions, achieving fine-grained control over poses. Next, we define the output format of the LLM and provide examples of pose descriptions for the LLM to reference. These examples are sourced from Posescript [17]. Finally, a user prompt of motion description is provided to instruct the LLM to generate the key pose descriptions. By harnessing the powerful LLM, the user prompts are no longer confined to specific formats but are open-world, allowing for expressions like "Jump on one foot" and "Experiencing a profound sense of joy".

![](images/8.jpg)

# D. Detailed Implementations

Posture-diffuser. We employ an AdamW [50] optimizer for 1000 epochs with a batch size of 512 and a learning rate of 1e-4. The number of layers is $N = 3$ and the latent dim is set to 512. Following [33], we use the linear schedule, where $\beta _ { s t a r t } = 0 . 0 0 0 8 5$ and $\beta _ { e n d } = 0 . 0 1 2$ Our model is trained using poses with automatically generated captions (PoseScript-A [17]) and adopts the last checkpoint as our final model. For the two encoders used in Posture Planning module, we use the pretrained text-to-pose retrieval model from [17], which is trained on PoseScript-A using the Batch

Based Classification loss [88].

Go-diffuser. We employ an AdamW [50] optimizer with a batch size of 64 and a learning rate of 1e-4. The latent dim is 512, the number of transformer layers is 8, and the number of heads is 4. We utilize GELU [29] as the activation function and the dropout is 0.1. The mask probability of the model's condition is O.1. We use the cosine schedule and the number of diffusion steps is 100.

# E. Evaluation details

We will detail our evaluation metrics on the Experiments part in this section. In open-world text-to-motion experiments, we employ sentence transformers [70, 71] to compute the similarity between the text descriptions in IDEA-400 [45], a high-quality motion language subset within Motion-X, and the text descriptions in HumanML3D. We filter out pairs with similarity greater than a specified threshold $\alpha$ , e.g., 0.45, yielding a motion language dataset comprising 368 text-motion pairs as our first test dataset.

<table><tr><td></td></tr><tr><td>Th as possible.Before you write each description, you must follow these instructions. These are primary rules: Rule 1. Characterize the degree of bending of body parts.</td></tr><tr><td>  &#x27;partially bent&#x27;, &#x27;slightly bent&#x27;, &#x27;straight&#x27;]. 1.2 You should select the body part form the list: [left knee, right knee, left elbow, right elbow].</td></tr><tr><td>Rule 2. Classify the relative distances between different body parts. o u    rar . 2.2 You must compare the distances between these body part pairs as much as possible:</td></tr><tr><td>(left elbow, right elbow), (left hand, right hand), (left knee, right knee), (left foot, right foot),</td></tr><tr><td></td></tr><tr><td>(eand let houler, (let hand riht houler (riht hand let hour) (riht hand rght houer),</td></tr><tr><td>(    ,</td></tr><tr><td>,</td></tr><tr><td>(left hand, left foot), (left hand, right foot), (right hand, left foot), (right hand, right foot)</td></tr><tr><td></td></tr><tr><td>Rule 3. Describe the relative positions of different body parts. 3.1 For he ront-back direction you should select thedeition wor rom the ist: [behind, n on of].</td></tr><tr><td>For the up-down direction, you should select the description word from the list: [&#x27;below&#x27;, &#x27;above&#x27;].</td></tr><tr><td>F  y u       3.2 You must compare the relative positioning between these body part pairs as much as possible:</td></tr><tr><td>, ,</td></tr><tr><td>(ran our, efot,ip,  ot,t ip, ewr neck, ri wr,</td></tr><tr><td> </td></tr><tr><td></td></tr><tr><td>Rule 4. Determine whether a body part is oriented &#x27;vertical&#x27; or &#x27;horizontal&#x27;.</td></tr><tr><td>4.1 You should select the description word from the list: [&#x27;vertical&#x27;, &#x27;horizontal&#x27;]. 4.2 You need to determine as much as possible whether the body limb formed by the following pairs</td></tr><tr><td>of body parts is &#x27;vertical&#x27; or &#x27;horizontal&#x27;:</td></tr><tr><td>(left hip, left knee), (right hip, right knee), (left knee, left ankle), (right knee, right ankle),</td></tr><tr><td>(lehoulder, t elbow), (riht houlde, right ebow), (le ebw, let wrist), (righ ebow, righ wrst),</td></tr><tr><td>(pelvis, left shoulder), (pelvis, right shoulder), (pelvis, neck)</td></tr><tr><td>Rule 5. Identify whether a body part is in contact with the ground.</td></tr><tr><td>5.1 You should select the description word from the list: [&#x27;on the ground&#x27;]. 5.2 You should select the body part form the list: [left knee, right knee, left foot, right foot].</td></tr><tr><td></td></tr><tr><td>You should write all the pose description together. The response should follow the format: {&quot;F1&quot;pose description&quot;,n&quot;F2&quot;pose desciption&quot;,n&quot;F3&quot;pose description&quot;,n&quot;F4&quot;:&quot;pose description&quot;,</td></tr><tr><td>&quot;F5&quot;pose description&quot;,\n&quot;F6&quot;:&quot;pose description&quot;,n&quot;F7&quot;:&quot;pose description&quot;,\n&quot;F8&quot;pose description}</td></tr><tr><td>Some sample pose descriptions are as follows: </td></tr><tr><td>ar,the let leg, the torso and the right thigh are straightened up while the right elbow is bent a bt. &quot; </td></tr><tr><td> </td></tr><tr><td>further than shoulder width apart from the other. &quot;</td></tr><tr><td></td></tr><tr><td></td></tr></table>

# calculate the similarity of motion descriptions and filter out pairs with similarity greater than a specified threshold 2import torch 3import torch.nn as nn 4import torch.nn.functional as F 5from transformers import AutoTokenizer, AutoModel , util 6 def mean_pooling(model_output, attention_mask): token_embeddings $=$ model_output[0] # First element of model_output contains all token embeddings input_mask_expanded $=$ attention_mask. unsqueeze(-1).expand(token_embeddings.size()) .float() 10 return torch.sum(token_embeddings $\star$ input_mask_expanded, 1) / torch.clamp( input_mask_expanded.sum(1), min $= 1 \in - 9$ 11 12 # calculate the features of descriptions 13class TextToSen(nn.Module): 14 def _init_(self, device): 15 super().__init_() 16 self.device $=$ device 17 self.tokenizer $=$ AutoTokenizer. from_pretrained('sentence-transformers/allmpnet-base-v2') 18 self.model $=$ AutoModel.from_pretrained(' sentence-transformers/all-mpnet-base-v2'). eval().to(self.device) 19 20 def forward(self, sentences): 21 encoded_input $=$ self.tokenizer(sentences, padding=True, truncation $\equiv$ True, return_tensors $= ^ { \prime }$ pt').to(self.device) 22 23 # Compute token embeddings 24 with torch.no_grad(): 25 model_output $=$ self.model( $\star \star$ encoded_input) 26 27 # Perform pooling 28 sentence_embeddings $=$ mean_pooling( model_output, encoded_input['attention_mask' ]) 29 30 # Normalize embeddings 31 sentence_embeddings $\begin{array} { r l } { \mathbf { \Psi } } & { { } = \mathbf { \Psi } \mathbf { \Psi } \mathbf { \Psi } \mathbf { \Psi } } \end{array}$ .normalize( sentence_embeddings, $\mathrm { p } { = } 2$ , dim $^ { 1 = 1 }$ 32 33 return sentence_embeddings 34 35# calculate the similarity 36similarity_score $=$ util.pytorch_cos_sim( embeddings1, embeddings2) 37 38if similarity_score $>$ threshold: 39 False

tances between the motion and text embeddings, given one motion sequence and K text descriptions (1 ground-truth and $K - 1$ randomly selected mismatched descriptions).

R-Precision is determined by ranking the Euclidean dis

Frechet Inception Distance (FID) measures the distributional difference between the generated and real motion by applying FID [31] to the extracted motion features derived from that text.

Multimodal Distance (MM-Dist) is calculated as the average Euclidean distance between each text feature and the corresponding generated motion feature.

In go-diffuser experiments, we utilize Average Position Error (APE) and Average Variance Error (AVE) to evaluate our methods [63]. We report the listed four metrics in Tab. 2.

Global trajectory errors. Tak3 only the X and Y coordinates of the root joint. It is the red trajectory on the ground in the visualizations of Fig. 7.

Mean local errors. Average the joint errors in the body's local coordinate system,

Mean global errors. Average the joint errors in the global coordinate system.

As in JL2P [3], Ghosh et al. [21] and TEMOS [63], the APE for a specific joint $\mathrm { j }$ is determined by computing the mean of the L2 distances between the generated and ground truth joint positions across the frames (F) and samples (N):

$$
A P E [ j ] = \frac { 1 } { N F } \sum _ { n \in N } \sum _ { f \in F } \left\| \boldsymbol { H } _ { f } \left[ j \right] - \hat { \boldsymbol { H } } _ { f } \left[ j \right] \right\| _ { 2 }
$$

As introduced in Ghosh et al [21] and TEMOS [63], the Average Variance Error (AVE), quantifies the distinction in variations. This metric is defined as the mean of the L2 distances between the generated and ground truth variances for the joint j.

$$
A V E [ j ] = { \frac { 1 } { N } } \sum _ { n \in N } \left\| \delta \left[ j \right] - \hat { \delta } \left[ j \right] \right\| _ { 2 }
$$

where,

$$
\delta \left[ j \right] = \frac { 1 } { F - 1 } \sum _ { f \in F } \left( H _ { f } \left[ j \right] - \tilde { H } _ { f } \left[ j \right] \right) ^ { 2 } \in R ^ { 3 }
$$

denotes the variance of the joint j.