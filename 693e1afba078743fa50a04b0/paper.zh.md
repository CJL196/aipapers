# 一步扩散与分布匹配蒸馏

天威·尹1 米哈伊尔·加尔比2 理查德·张2 埃利·谢赫特曼2 弗雷多·杜兰1 威廉·T·弗里曼1 太成·朴2 1麻省理工学院 2Adobe研究 https://tianweiy.github.io/dmd/

![](images/1.jpg)  
F     a  [ -

# 摘要

扩散模型生成高质量图像，但需要进行数十次前向传播。我们引入了分布匹配蒸馏（DMD），这一过程旨在将扩散模型转化为一个一步图像生成器，对图像质量的影响最小化。我们通过最小化近似的KL散度，强制一步图像生成器在分布层面上与扩散模型匹配，其梯度可表示为两个评分函数之间的差异，一个是目标分布的评分函数，另一个是由我们的一步生成器生成的合成分布的评分函数。评分函数被参数化为分别在每个分布上独立训练的两个扩散模型。结合一个简单的回归损失，以匹配多步扩散输出的大规模结构，我们的方法在所有已发布的少步扩散方法中表现优越，在ImageNet $6 4 \times 6 4$ 上达到2.62 FID，在零样本COCO-30k上达到11.49 FID，性能可与稳定扩散相媲美，但速度快几个数量级。利用FP16推理，我们的模型能够在现代硬件上以20 FPS的速度生成图像。

# 1. 引言

扩散模型已彻底改变了图像生成，凭借稳定的训练流程达到了前所未有的逼真性和多样性。然而，与生成对抗网络和变分自编码器相比，它们的采样过程是一种缓慢的迭代过程，通过逐步去噪将高斯噪声样本转化为复杂的图像。这通常需要数十到数百次昂贵的神经网络评估，限制了在使用生成管道作为创意工具时的互动性。为了加速采样速度，之前的方法将原始多步扩散采样发现的噪声图像映射提炼为单次通过的学生网络。然而，拟合如此高维、复杂的映射无疑是一项艰巨的任务。一个挑战是运行完整去噪轨迹的高昂成本，仅仅是为了实现学生模型的一次损失计算。最近的方法通过逐步增加学生的采样距离来缓解这一问题，而无需运行原始扩散的完整去噪序列。然而，提炼模型的性能仍然落后于原始的多步扩散模型。

与其强制噪声与扩散生成图像之间的对应关系，我们只是强制学生生成的图像与原始扩散模型看起来无法区分。从高层次来看，我们的目标与其他分布匹配生成模型的动机相似，例如 GMMN 或 GANs。然而，尽管它们在生成真实图像方面取得了显著成功，但在一般文本到图像数据上扩展模型一直很具挑战性。在本研究中，我们通过从一个已经在大规模文本到图像数据上训练的扩散模型开始，绕过了这个问题。具体而言，我们对预训练的扩散模型进行微调，以学习不仅是数据分布，还有由我们的蒸馏生成器产生的假分布。由于已知扩散模型能够近似扩散分布上的评分函数，我们可以将去噪后的扩散输出解释为使图像“更真实”或如果扩散模型是在假图像上学习的，则为“更假”的梯度方向。最后，生成器的梯度更新规则被设定为两者之间的差异，推动合成图像朝向更高的真实感和更低的假感。之前的研究显示，在一种称为变分评分蒸馏的方法中，用预训练的扩散模型建模真实与假分布对于三维对象的测试时优化也是有效的。我们的见解是，类似的方法可以训练整个生成模型。此外，我们发现预计算适度数量的多步骤扩散采样结果，并对我们的一步生成施加简单的回归损失，可以在存在分布匹配损失时充当有效的正则化器。此外，回归损失确保我们的一步生成器与教师模型对齐，展示了实时设计预览的潜力。我们的方法汲取了 VSD、GANs 和 pix2pix 的灵感和见解，表明通过 (1) 用扩散模型建模真实与假分布，以及 (2) 使用简单的回归损失匹配多步骤扩散输出，我们能够训练一个高保真度的一步生成模型。

我们评估了使用我们的分布匹配蒸馏程序（DMD）训练的模型在各项任务中的表现，包括在 CIFAR-10 和 ImageNet $6 4 \times 6 4$ 上的图像生成，以及在 MS COCO $5 1 2 \times 5 1 2$ 上的零样本文本到图像生成。在所有基准测试中，我们的一步生成器显著超越了所有已发布的少步扩散方法，如渐进蒸馏，修正流，以及一致性模型。在 ImageNet 上，DMD 达到了 2.62 的 FID，相较于一致性模型提高了 $2 . 4 \times$。采用与稳定扩散相同的去噪器架构，DMD 在 MS-COCO 2014-$3 0 \mathrm { k }$ 上达到了 11.49 的竞争性 FID。我们的定量和定性评估表明，我们模型生成的图像在质量上与高成本的稳定扩散模型生成的图像非常相似。重要的是，我们的方法在保持这种图像保真度的同时，实现了神经网络评估的 $1 0 0 \times$减少。这种效率使得 DMD 在使用 FP16 推理时以 20 FPS 的速度生成 $5 1 2 \times 5 1 2$ 的图像，为互动应用开辟了广泛的可能性。

# 2. 相关工作

扩散模型 扩散模型已成为一种强大的生成建模框架，在图像生成、音频合成和视频生成等多个领域取得了前所未有的成功。这些模型通过逆扩散过程逐步将噪声转化为一致的结构。尽管达到了最先进的结果，但扩散模型固有的迭代过程在实时应用中往往伴随高昂且难以承受的计算成本。我们的工作基于领先的扩散模型，并提出了一种简单的蒸馏管道，将多步骤生成过程简化为单次前向传递。我们的方法可以普遍适用于任何具有确定性采样的扩散模型。

扩散加速 在扩散模型推理过程中的加速一直是该领域的关键焦点，促进了两类方法的发展。第一类方法是快速扩散采样器 [31, 41, 45, 46, 91]，可以显著减少预训练扩散模型所需的采样步骤，从一千步减少到仅20-50步。然而，进一步减少步骤往往会导致性能的灾难性下降。另一方面，扩散蒸馏作为进一步提升速度的有希望的途径应运而生 [3, 16, 42, 47, 51, 65, 75, 83, 92]。它们将扩散蒸馏框架视为知识蒸馏 [19]，在此过程中，训练学生模型以将原始扩散模型的多步输出提炼为单步输出。Luhman等 [47] 和DSNO [93] 提出了一个简单的方法，即预计算去噪轨迹，并使用像素空间中的回归损失来训练学生模型。然而，一个显著的挑战是对于每次损失函数的计算而言，运行完整去噪轨迹的代价非常高。为了解决这个问题，Progressive

![](images/2.jpg)  
Figure 2. Method overview. We train one-step generator $G _ { \theta }$ to map random noise $z$ into a realistic image. To match the multi-step distribution matching gradient $\nabla _ { \boldsymbol { \theta } } \overline { { \boldsymbol { D } _ { K L } } }$ to the fake image to enhance realism. We inject a random amount of noise to the fake image and the one-step generator.

蒸馏（PD）[51, 65] 训练一系列学生模型，这些模型的采样步骤数是先前模型的一半。InstaFlow [42, 43] 逐步学习更直接的流，使得一步预测能够在更长的距离上保持准确性。一致性蒸馏（CD）[75]、TRACT [3] 和BOOT [16] 训练学生模型，以匹配其在ODE流中不同时间步的自身输出，这反过来又被要求匹配其在另一个时间步的自身输出。相比之下，我们的方法表明，一旦将分布匹配作为训练目标，Luhman等人和DSNO预计算扩散输出的简单方法是足够的。

分布匹配 最近，几类生成模型在扩展到复杂数据集方面取得了成功，通过恢复由于预定义机制（如噪声注入或令牌掩蔽）而受损的样本。在另一方面，存在一些生成方法并不依赖于样本重构作为训练目标。相反，它们在分布层面上匹配合成样本和目标样本，例如GMMD或GANs。其中特别是GANs在现实主义方面展现了前所未有的质量，特别是当GAN损失可以与特定任务的辅助回归损失结合以减轻训练不稳定性时，这种方法在成对图像翻译到非成对图像编辑等应用中都表现出优势。然而，GAN在文本引导合成中并不是一个热门选择，因为需要精心设计架构以确保在大规模下的训练稳定性。

最近，几项研究[1, 12, 82, 86]探讨了基于评分的模型与分布匹配之间的联系。特别是，ProlificDreamer [80] 提出了变分评分蒸馏（VSD），利用预训练的文本到图像扩散模型作为分布匹配损失。由于VSD可以利用大规模的预训练模型进行无配对设置[17, 58]，因此在基于粒子的文本条件3D合成优化中表现出色。我们的方法对VSD进行了改进和扩展，以训练深度生成神经网络以蒸馏扩散模型。此外，受到生成对抗网络（GANs）在图像转换中成功的启发，我们通过回归损失增强训练的稳定性。因此，我们的方法在像LAION [69]这样的复杂数据集上成功实现了高现实感。我们的方法与最近将GAN与扩散相结合的工作[68, 81, 83, 84]不同，因为我们的公式并不以GAN为基础。我们的方法与同时进行的研究[50, 85]有相似的动机，均利用VSD目标训练生成器，但不同于那些方法的是，我们通过引入回归损失专门针对扩散蒸馏，且在文本到图像任务上显示出最先进的结果。

# 3. 分布匹配蒸馏

我们的目标是将给定的预训练扩散去噪模型——基础模型 $\mu _ { \mathrm { b a s e } }$ ——提炼成一个快速的“一步”图像生成器 $G _ { \theta }$，该生成器能够在无需耗时的迭代采样过程（见第3.1节）中生成高质量图像。虽然我们希望从相同的分布中生成样本，但我们并不一定寻求重现精确的映射。

![](images/3.jpg)  
Figure 3. Optimizing various objectives starting from the same configuration (left) leads to different outcomes. (a) Maximizing the real score only, the fake samples all collapse to the closest mode of the real distribution. (b) With our distribution matching objective but not regression loss, the generated fake data covers more of the real distribution, but only recovers the closest mode, missing the second mode entirely. (c) Our full objective, with the regression loss, recovers both modes of the target distribution.

通过类比生成对抗网络（GAN），我们将蒸馏模型的输出称为假样本，与训练分布中的真实图像相对。我们在图2中说明了我们的方法。我们通过最小化两个损失的总和来训练快速生成器：一个分布匹配目标（第3.2节），其梯度更新可以表示为两个评分函数的差值，以及一个回归损失（第3.3节），该损失鼓励生成器在一组固定的噪声-图像对上匹配基模型输出的大尺度结构。关键是，我们使用两个扩散去噪器分别建模真实和假分布的评分函数，并用不同幅度的高斯噪声进行扰动。最后，在第3.4节中，我们展示了如何通过无分类器指导来调整我们的训练过程。

# 3.1. 预训练基础模型与一步生成器

我们的蒸馏过程假设给定一个预训练的扩散模型 $\mu _ { \mathrm { b a s e } }$。扩散模型旨在逆转一个高斯扩散过程，该过程逐渐向来自真实数据分布 $x _ { 0 } \sim p _ { \mathrm { r e a l } }$ 添加噪声，将其转变为白噪声 $x _ { T } ~ \sim ~ \mathcal { N } ( 0 , \mathbf { I } )$，共经过 $T$ 个时间步 [21, 71, 74]；我们使用 $T = 1 0 0 0$。我们将扩散模型表示为 $\mu _ { \mathrm { b a s e } } ( x _ { t } , t )$。从高斯样本 $x _ { T }$ 开始，模型在时间步 $t \in \{ 0 , 1 , . . . , T - 1 \}$ （或噪声级别）条件下，迭代地去噪一个运行中的噪声估计 $x _ { t }$，以生成目标数据分布的样本。扩散模型通常需要 10 到 100 步来生成逼真的图像。我们的推导使用扩散的均值预测形式以简化 [31]，但在变量变换 [33] 的情况下，与 $\epsilon$ 预测 [21, 63] 的效果一致（见附录 H）。我们的实现使用来自 EDM [31] 和 Stable Diffusion [63] 的预训练模型。一步生成器。我们的一步生成器 $G _ { \theta }$ 具有基础扩散去噪器的架构，但没有时间条件。我们在训练前用基础模型初始化其参数 $\theta$，即 $G _ { \theta } ( z ) = \mu _ { \mathrm { b a s e } } ( z , T - 1 ) , \forall z$。

# 3.2. 分布匹配损失

理想情况下，我们希望我们的快速生成器能够生成与真实图像无法区分的样本。受到ProlificDreamer的启发，我们最小化真实图像分布$p_{ \mathrm{real} }$与假图像分布$p_{ \mathrm{fake} }$之间的Kullback-Leibler (KL) 散度：

$$
\begin{array} { r l } & { D _ { K L } \left( p _ { \mathrm { f a k e } } \parallel p _ { \mathrm { r e a l } } \right) = \underset { x \sim p _ { \mathrm { f a k e } } } { \mathbb { E } } \left( \log \left( \frac { p _ { \mathrm { f a k e } } ( x ) } { p _ { \mathrm { r e a l } } ( x ) } \right) \right) } \\ & { \qquad = \underset { z \sim \mathcal { N } ( 0 ; \mathbf { I } ) } { \mathbb { E } } - \left( \log \mathbf { \delta } p _ { \mathrm { r e a l } } ( x ) - \log \mathbf { \delta } p _ { \mathrm { f a k e } } ( x ) \right) . } \\ & { \qquad \quad x = G _ { \theta } ( z ) } \end{array}
$$

计算概率密度以估计该损失通常是不可处理的，但我们只需要相对于 $\theta$ 的梯度来通过梯度下降训练我们的生成器。使用近似得分进行梯度更新。对式（1）相对于生成器参数的梯度进行求解：

$$
\nabla _ { \theta } D _ { K L } = \operatorname* { l } _ { z \sim \mathcal { N } ( 0 ; \mathbf { I } ) } \Big [ - \big ( s _ { \mathrm { r e a l } } ( x ) - s _ { \mathrm { f a k e } } ( x ) \big ) \frac { d G } { d \theta } \Big ] ,
$$

其中 $s _ { \mathrm { r e a l } } ( x ) = \nabla _ { x } \mathrm { l o g } \ p _ { \mathrm { r e a l } } ( x )$，$s _ { \mathrm { f a k e } } ( x ) = \nabla _ { x } \log p _ { \mathrm { f a k e } } ( x )$ 是各自分布的评分。直观上，$s _ { \mathrm { r e a l } }$ 使 $x$ 向 $p _ { \mathrm { r e a l } }$ 的峰值移动，而 $- s _ { \mathrm { f a k e } }$ 则将其分开，如图 3(a, b) 所示。计算这个梯度仍然具有挑战性，原因有二：首先，对于低概率样本，评分会发散——特别是对于伪样本，$p _ { \mathrm { r e a l } }$ 消失；其次，我们用于估计评分的工具，即扩散模型，仅提供扩散分布的评分。Score-SDE [73, 74] 针对这两个问题提供了应对方案。通过对数据分布施加不同标准差的随机高斯噪声，我们创建了一系列在环境空间上完全支持的“模糊”分布，因此它们重叠，使得式 (2) 中的梯度是明确定义的（图 4）。Score-SDE 进一步表明，训练过的扩散模型近似于扩散分布的评分函数。因此，我们的策略是使用一对扩散去噪器来建模高斯扩散后真实和伪分布的评分。稍微滥用符号，我们分别将其定义为 $s _ { \mathrm { r e a l } } ( x _ { t } , t )$ 和 $s _ { \mathrm { f a k e } } ( x _ { t } , t )$。扩散样本 $x _ { t } \sim q ( x _ { t } | x )$ 是通过在扩散时间步 $t$ 上将噪声添加到生成器输出 $x = G _ { \theta } ( z )$ 获得的：

$$
q _ { t } ( x _ { t } | x ) \sim \mathcal { N } ( \alpha _ { t } x ; \sigma _ { t } ^ { 2 } \mathbf { I } ) ,
$$

其中 $\alpha _ { t }$ 和 $\sigma _ { t }$ 来源于扩散噪声调度。真实得分。真实分布是固定的，对应于基础扩散模型的训练图像，因此我们使用预训练扩散模型的固定副本 $\mu _ { \mathrm { b a s e } } ( x , t )$ 来建模其得分。给定扩散模型的得分由 Song 等人 [74] 给出：

$$
s _ { \mathrm { r e a l } } ( x _ { t } , t ) = - \frac { x _ { t } - \alpha _ { t } \mu _ { \mathrm { b a s e } } ( x _ { t } , t ) } { \sigma _ { t } ^ { 2 } } .
$$

![](images/4.jpg)  
Figure 4. Without perturbation, the real/fake distributions may not overlap (a). Real samples only get a valid gradient from the real score, and fake samples from the fake score. After diffusion (b), our distribution matching objective is well-defined everywhere.

动态学习的虚假评分。我们以与真实评分相同的方式推导虚假评分函数：

$$
s _ { \mathrm { f a k e } } ( x _ { t } , t ) = - \frac { x _ { t } - \alpha _ { t } \mu _ { \mathrm { f a k e } } ^ { \phi } ( x _ { t } , t ) } { \sigma _ { t } ^ { 2 } } .
$$

然而，由于我们生成样本的分布在训练过程中不断变化，我们动态地调整伪扩散模型 $\mu _ { \mathrm { f a k e } } ^ { \phi }$ 以跟踪这些变化。我们将伪扩散模型初始化为预训练的扩散模型 $\mu _ { \mathrm { b a s e } }$ ，在训练过程中通过最小化标准去噪目标 [21, 77] 来更新参数 $\phi$ ：

$$
\mathcal { L } _ { \mathrm { d e n o i s e } } ^ { \phi } = | | \mu _ { \mathrm { f a k e } } ^ { \phi } ( x _ { t } , t ) - x _ { 0 } | | _ { 2 } ^ { 2 } ,
$$

其中 $\mathcal { L } _ { \mathrm { d e n o i s e } } ^ { \phi }$ 根据扩散时间步 $t$ 进行加权，采用与基本扩散模型训练期间相同的加权策略 [31, 63]。分布匹配梯度更新。我们的最终近似分布匹配梯度是通过在公式 (2) 中用在扰动样本 $x _ { t }$ 上由两个扩散模型定义的精确得分进行替换，并对扩散时间步求期望得到的：

$$
\nabla _ { \theta } D _ { K L } \simeq \operatorname * { \mathbb { \Gamma } } _ { z , t , x , x _ { t } } \left[ w _ { t } \alpha _ { t } \left( s _ { \mathrm { f a k e } } ( x _ { t } , t ) - s _ { \mathrm { r e a l } } ( x _ { t } , t ) \right) \frac { d G } { d \theta } \right] ,
$$

其中 $z \sim \mathcal { N } ( 0 ; \mathbf { I } ) , x = G _ { \theta } ( z ) , t \sim \mathcal { U } ( T _ { \operatorname* { m i n } } , T _ { \operatorname* { m a x } } ) ,$ 并且 $x _ { t } \sim q _ { t } ( x _ { t } | x )$。我们在附录 F 中包括推导过程。在这里，$w _ { t }$ 是一个时间相关的标量权重，我们添加它以改善训练动态。我们设计权重因子以归一化不同噪声水平下梯度的大小。具体而言，我们计算去噪图像与输入之间的空间和通道维度的平均绝对误差，其中 $S$ 是空间位置的数量，$C$ 是通道的数量。在第 4.2 节中，我们展示了该权重因子优于之前的设计 [58, 80]。我们将 $T _ { \mathrm { m i n } } =$ $0 . 0 2 T$ 和 $T _ { \mathrm { m a x } } = 0 . 9 8 T$，遵循 DreamFusion [58]。

$$
\begin{array} { r } { w _ { t } = \frac { \sigma _ { t } ^ { 2 } } { \alpha _ { t } } \frac { C S } { \lvert \lvert \mu _ { \mathrm { b a s e } } ( x _ { t } , t ) - x \rvert \rvert _ { 1 } } , } \end{array}
$$

# 3.3. 回归损失与最终目标

上节介绍的分布匹配目标在 $t \gg 0$ 的情况下是明确的，即当生成样本被大量噪声污染时。然而，对于少量噪声，$s _ { \mathrm { r e a l } } ( x _ { t } , t )$ 通常变得不可靠，因为 $p _ { \mathrm { r e a l } } ( x _ { t } , t )$ 会趋近于零。此外，由于分数 $\nabla _ { \boldsymbol { x } } \log ( p )$ 对概率密度函数 $p$ 的缩放是保持不变的，因此优化容易受到模式崩溃/丢失的影响，导致虚假分布在某些模式上分配更高的总体密度。为了解决这个问题，我们使用额外的回归损失来确保所有模式都被保留；见图 3(b), (c)。

该损失度量了在相同输入噪声下生成器与基础扩散模型输出之间的逐点距离。具体而言，我们构建了一个成对的数据集 $\mathcal { D } = \{ \boldsymbol { z } , \boldsymbol { y } \}$，由随机高斯噪声图像 $z$ 和相应的输出 $y$ 组成，这些输出是通过使用确定性常微分方程求解器从预训练的扩散模型 $\mu _ { \mathrm { b a s e } }$ 采样获得的 [31, 41, 72]。在我们的 CIFAR-10 和 ImageNet 实验中，我们使用来自 EDM [31] 的 Heun 求解器，对于 CIFAR-10 使用 18 步，对于 ImageNet 使用 256 步。对于 LAION 实验，我们使用 PNDM [41] 求解器，设定为 50 次采样步骤。我们发现即使是少量的噪声—图像对，在 CIFAR10 的情况下，比如使用不到 $1 \%$ 的训练计算量生成的，也能作为有效的正则化器。我们的回归损失如下所示：

$$
\mathcal { L } _ { \mathrm { r e g } } = \underset { ( z , y ) \sim \mathcal { D } } { \mathbb { E } } \ell ( G _ { \theta } ( z ) , y ) .
$$

我们使用学习感知图像块相似性（LPIPS）[89] 作为距离函数 $\ell$，遵循 InstaFlow [43] 和一致性模型 [75]。

最终目标网络 $\dot { \mu } _ { \mathrm { f a k e } } ^ { \phi }$ 的损失函数为 $\mathcal { L } _ { \mathrm { d e n o i s e } } ^ { \phi }$，其梯度为 $\nabla _ { \boldsymbol { \theta } } D _ { K L }$，最终目标为 $D _ { K L } + \lambda _ { \mathrm { r e g } } \mathcal { L } _ { \mathrm { r e g } }$，其中 $\lambda _ { \mathrm { r e g } } = 0.25$，除非另有说明。梯度 $\nabla _ { \boldsymbol { \theta } } D _ { K L }$ 在公式 (7) 中计算，梯度 $\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { r e g } }$ 则通过自动微分从公式 (9) 中计算。我们将这两种损失应用于不同的数据流：未配对的伪样本用于分布匹配梯度，配对示例如第 3.3 节所述用于回归损失。算法 1 概述了最终的训练过程。附录 B 提供了更多详细信息。

# 3.4. 无分类器引导的蒸馏

无分类器引导 [20] 被广泛用于提高文本到图像扩散模型的图像质量。我们的方法同样适用于使用无分类器引导的扩散模型。我们首先通过从引导模型中采样生成相应的噪声输出对，以构建回归损失所需的配对数据集 $\mathcal { L } _ { \mathrm { r e g } }$ 。在计算分布匹配梯度 $\nabla _ { \boldsymbol { \theta } } D _ { K L }$ 时，我们用从均值导出的真实得分进行替代。

# 算法 1：DMD 训练过程

预训练的真实扩散模型 $\mu _ { \mathrm { r e a l } }$，配对数据集 $\mathcal { D } = \{ z _ { \mathrm { r e f } } , y _ { \mathrm { r e f } } \}$

训练生成器 $G$ 。 1 // 初始化生成器及假分数估计器 来自预训练模型 2 $G \gets$ coWeights $( \mu _ { \mathrm { r e a l } } )$ , µfake coWeights $_ { ( \mu _ { \mathrm { r e a l } } ) }$ 3 当训练期间 do 4 // 生成图像 5 采样批次 $z \sim \mathcal { N } ( 0 , \mathbf { I } ) ^ { B }$ 和 $( z _ { \mathrm { r e f } } , y _ { \mathrm { r e f } } ) \sim \mathcal { D }$ 6 $x G ( z )$ , $x _ { \mathrm { r e f } } G ( z _ { \mathrm { r e f } } )$ 7 $x =$ concat $( x , x _ { \mathrm { r e f } } )$ 如果数据集是 LAION 否则 $_ x$ 8 9 // 更新生成器 10 $\mathcal { L } _ { \mathrm { K L } } \gets$ distributionMatchingLoss( $\mu _ { \mathrm { r e a l } }$ , µfake, x) // Eq 7 11 $\mathcal { L } _ { \mathrm { r e g } } \mathrm { L P I P S } ( x _ { \mathrm { r e f } } , y _ { \mathrm { r e f } } ) / / \textrm { \texttt { E q } } 9$ 12 $\mathcal { L } _ { G } \mathcal { L } _ { \mathrm { K L } } + \lambda _ { \mathrm { r e g } } \mathcal { L } _ { \mathrm { r e g } }$ 13 G ← update $\left( G , \mathcal { L } _ { G } \right)$ 14 15 // 更新假分数估计模型 16 采样时间步 $t \sim \mathcal { U } ( 0 , 1 )$ 17 $x _ { t } \gets$ forwardDiffusion $( \operatorname { s t o p g r a d } ( x ) , t )$ 18 Ldenoise denoisingLoss( $\mu _ { \mathrm { f a k e } } ( x _ { t } , t )$ , stopgrad(x)) / / Eq 6 19 $\mu _ { \mathrm { f a k e } } $ update( $\mu _ { \mathrm { f a k e } }$ , ${ \mathcal { L } } _ { \mathrm { d e n o i s e } }$ 20 结束当训练期间，指导模型的预测。与此同时，我们不修改假分数的公式。我们用固定的指导比例训练我们的单步生成器。

# 4. 实验

我们通过多个基准测试评估我们方法的能力，包括在CIFAR-10和ImageNet上的类别条件生成。我们使用Fréchet Inception Distance (FID)来测量图像质量，并使用CLIP Score评估文本到图像的对齐。首先，我们在ImageNet上进行直接比较（第4.1节），我们的分布匹配蒸馏显著优于使用相同基础扩散模型的对比蒸馏方法。其次，我们进行详细的消融研究，以验证我们提出的模块的有效性（第4.2节）。第三，我们在LAION-Aesthetic- $^ { 6 . 2 5 + }$ 数据集上训练文本到图像模型，使用无分类器引导比例为3（第4.3节）。在这一阶段，我们蒸馏Stable Diffusion v1.5，并展示我们的蒸馏模型的FID与原始模型相当，同时提供了 $3 0 \times$ 的加速。最后，我们在LAION-Aesthetic- $^ { 6 + }$ 上训练另一文本到图像模型，利用更高的引导值8（第4.3节）。该模型旨在增强视觉质量，而不是优化FID指标。定量和定性分析确认，使用我们的分布匹配蒸馏程序训练的模型能够生成与Stable Diffusion相媲美的高质量图像。我们在附录中描述了额外的训练和评估细节。

# 4.1. 类条件图像生成

我们在类别条件的 ImageNet $6 4 \times 6 4$ 数据集上训练我们的模型，并与竞争方法进行基准测试。结果如表 1 所示。我们的模型超越了成熟的 GAN，如 BigGAN-deep [4]，以及最近的扩散蒸馏方法，包括一致性模型 [75] 和 TRACT [3]。我们的方法显著缩小了保真度差距，达到了与原始扩散模型几乎相同的 FID 分数（within 0.3），同时速度提升了 512 倍。在 CIFAR-10 上，我们的类别条件模型达到了竞争力的 FID 分数 2.66。我们在附录中包含了 CIFAR-10 的结果。

Table 1. Sample quality comparison on ImageNet- $6 4 \times 6 4$ Baseline numbers are derived from Song et al. [75]. The upper section of the table highlights popular diffusion and GAN approaches [4, 9]. The middle section includes a list of competing diffusion distillation methods. The last row shows the performance of our teacher model, EDM† [31].   

<table><tr><td>Method</td><td># Fwd Pass (↓)</td><td>FID (↓)</td></tr><tr><td>BigGAN-deep [4] ADM [9]</td><td>1 250</td><td>4.06 2.07</td></tr><tr><td>Progressive Distillation [65]</td><td>1</td><td>15.39</td></tr><tr><td>DFNO [92]</td><td>1</td><td>7.83</td></tr><tr><td>BOOT [16]</td><td>1</td><td>16.30</td></tr><tr><td>TRACT [3]</td><td>1</td><td>7.43</td></tr><tr><td>Meng et al. [51]</td><td>1</td><td>7.54</td></tr><tr><td>Diff-Instruct [50]</td><td>1</td><td>5.57</td></tr><tr><td>Consistency Model [75]</td><td>1</td><td>6.20</td></tr><tr><td>DMD (Ours)</td><td>1</td><td>2.62</td></tr><tr><td>EDM (Teacher) [31]</td><td>512</td><td>2.32</td></tr></table>

# 4.2. 消融研究

我们首先将我们的方法与两个基线进行比较：一个省略了分布匹配目标，另一个缺少了我们框架中的回归损失。表 2（左）总结了结果。在缺乏分布匹配损失的情况下，我们的基线模型生成的图像缺乏真实感和结构完整性，如图 5 的顶部部分所示。同样，省略回归损失导致训练不稳定，容易出现模式崩溃，从而减少生成图像的多样性。这一问题在图 5 的底部部分中得到了说明。表 2（右）展示了我们提出的样本加权策略的优势（第 3.2 节）。我们与 $\sigma _ { t } / \alpha _ { t }$ 和 $\sigma _ { t } ^ { 3 } / \alpha _ { t }$ 这两种 DreamFusion [58] 和 ProlificDreamer [80] 使用的流行加权方案进行了比较。我们的加权策略实现了 0.9 的 FID 提升，因为它规范了不同噪声水平下的梯度幅值，从而稳定了优化过程。

![](images/5.jpg)  
DMD (ours)

![](images/6.jpg)  
without distribution matching

(a) 我们的模型（左）与不包含分布匹配目标的基线模型（右）之间的定性比较。基线模型生成的图像在真实感和结构完整性方面有所妥协。图像均来自相同的随机种子。

![](images/7.jpg)

(b) 我们的模型（左）与省略回归损失的基线模型（右）之间的定性比较。基线模型倾向于出现模式崩溃和缺乏多样性，灰色汽车的主导出现即为证据（用红色方框突出显示）。图像是从相同的随机种子生成的。

Figure 5. Ablation studies of our training loss, including the distribution matching objective (top) and the regression loss (bottom).   

<table><tr><td>Training loss</td><td></td><td></td><td>CIFAR ImageNet Sample weighting CIFAR</td></tr><tr><td>w/o Dist. Matching</td><td>3.82</td><td>9.21</td><td>σt/αt [58]</td></tr><tr><td>w/o Regress. Loss</td><td>5.58</td><td>5.61</td><td>3.60 σ/αt [58, 80] 3.71</td></tr><tr><td>DMD (Ours)</td><td>2.66</td><td>2.62</td><td>Eq. 8 (Ours) 2.66</td></tr></table>

Table 2. Ablation study. (left) We ablate elements of our training loss. We show the FID results on CIFAR-10 and ImageNet$6 4 \times 6 4$ (right) We compare different sample weighting strategies for the distribution matching loss.

# 4.3. 文本到图像生成

我们使用零样本 MS COCO 来评估我们模型在文本到图像生成任务上的性能。我们通过对 LAION-Aesthetics- $^{6.25+}$ 上进行稀疏 Distillation，训练了一个文本到图像模型，基于 Stable Diffusion v1.5 [63]。我们使用指导比例为 3，这在基础的 Stable Diffusion 模型中产生了最佳的 FID 值。训练大约需要 36 小时，使用 72 个 A100 GPU 的集群。表 3 比较了我们的模型与最先进的方案。我们的方法展现出优越的性能，超过了 StyleGAN-T [67]，超越了所有其他扩散加速方法，包括高级扩散求解器 [46, 91]，以及扩散蒸馏技术，如潜在一致性模型 [48, 49]、UFOGen [84] 和 InstaFlow [43]。我们大幅缩小了稀疏模型与基础模型之间的差距，FID 值达到了离 Stable Diffusion v1.5 仅 2.7 的水平，同时运行速度约快 $30 \times$。使用 FP16 进行推理时，我们的模型以每秒 20 帧的速度生成图像，使得交互式应用成为可能。

Table 3. Sample quality comparison on zero-shot text-toimage generation on MS COCO-30k. Baseline numbers are derived from GigaGAN [26]. The dashed line indicates that the result is unavailable. †Results are evaluated by us using the released models. LCM-LoRA is trained with a guidance scale of 7.5. We use a guidance scale of 3 for all the other methods. Latency is measured with a batch size of 1.   

<table><tr><td>Family</td><td>Method</td><td colspan="3">Resolution (↑) Latency (↓) FID (↓)</td></tr><tr><td rowspan="9">Original, unaccelerated</td><td>DALL·E [60]</td><td>256</td><td></td><td>27.5</td></tr><tr><td>DALL·E 2 [61]</td><td>256</td><td>-</td><td>10.39</td></tr><tr><td>Parti-750M [87]</td><td>256</td><td>-</td><td>10.71</td></tr><tr><td>Parti-3B [87]</td><td>256</td><td>6.4s</td><td>8.10</td></tr><tr><td>Make-A-Scene [13]</td><td>256</td><td>25.0s</td><td>11.84</td></tr><tr><td>GLIDE [52]</td><td>256</td><td>15.0s</td><td>12.24</td></tr><tr><td>LDM [63]</td><td>256</td><td>3.7s</td><td>12.63</td></tr><tr><td>Imagen [64]</td><td>256</td><td>9.1s</td><td>7.27</td></tr><tr><td>eDiff-I [2]</td><td>256</td><td>32.0s</td><td>6.95</td></tr><tr><td rowspan="3">GANs</td><td>LAFITE [94]</td><td>256</td><td>0.02s</td><td>26.94</td></tr><tr><td>StyleGAN-T [67]</td><td>512</td><td>0.10s</td><td>13.90</td></tr><tr><td>GigaGAN [26]</td><td>512</td><td>0.13s</td><td>9.09</td></tr><tr><td rowspan="6">Accelerated diffusion</td><td>DPM++ (4 step) [46]†</td><td>512</td><td>0.26s</td><td>22.36</td></tr><tr><td>UniPC (4 step) [91]†</td><td>512</td><td>0.26s</td><td>19.57</td></tr><tr><td>LCM-LoRA (4 step)[49]†</td><td>512</td><td>0.19s</td><td>23.62</td></tr><tr><td>InstaFlow-0.9B [43]</td><td>512</td><td>0.09s</td><td>13.10</td></tr><tr><td>UFOGen [84]</td><td>512</td><td>0.09s</td><td>12.78</td></tr><tr><td>DMD (Ours)</td><td>512</td><td>0.09s</td><td>11.49</td></tr><tr><td>Teacher</td><td>SDv1.5† [63]</td><td>512</td><td>2.59s</td><td>8.78</td></tr></table>

高引导尺度扩散蒸馏。在文本生成图像的任务中，扩散模型通常在高引导尺度下运行，以提高图像质量 [57, 63]。为了在这个高引导尺度的范围内评估我们的蒸馏方法，我们训练了一个额外的文本到图像模型。该模型在 LAION-Aesthetics- $^ { 6 + }$ 数据集上使用 8 的引导尺度对 SD v1.5 进行了蒸馏 [69]。表 4 将我们的方法与各种扩散加速方法进行基准对比 [46, 49, 91]。与低引导模型类似，我们的一步生成器显著优于竞争方法，即使它们采用了四步采样过程。与竞争方法及基础扩散模型的定性比较见于图 6。

# 5. 局限性

尽管我们的结果令人鼓舞，但我们的单步模型与更精细的扩散采样路径离散化（例如，进行100次或1000次神经网络评估的模型）之间仍存在轻微的质量差异。此外，我们的框架微调了虚假评分函数和生成器的权重，导致训练期间显著的内存使用。像 LORA 这样的技术为解决此问题提供了潜在的解决方案。

![](images/8.jpg)  
.

Table 4. FID/CLIP-Score comparison on MS COCO-30K. †Results are evaluated by us. LCM-LoRA is trained with a guidance scale of 7.5. We use a guidance scale of 8 for all the other methods. Latency is measured with a batch size of 1.   

<table><tr><td colspan="3">Method Latency (↓) FID (↓) CLIP-Score (↑)</td></tr><tr><td>DPM++ (4 step)[46]†</td><td>0.26s 22.44</td><td>0.309</td></tr><tr><td>UniPC (4 step)[91]†</td><td>0.26s 23.30</td><td>0.308</td></tr><tr><td>LCM-LoRA (1 step) [49]†</td><td>0.09s 77.90</td><td>0.238</td></tr><tr><td>LCM-LoRA (2 step) [49]†</td><td>0.12s 24.28</td><td>0.294</td></tr><tr><td>LCM-LoRA (4 step) [49]†</td><td>0.19s 23.62</td><td>0.297</td></tr><tr><td>DMD (Ours)</td><td>0.09s 14.93</td><td>0.320</td></tr><tr><td>SDv1.5† (Teacher) [63]</td><td>2.59s 13.45</td><td>0.322</td></tr></table>

# 致谢

这项工作在 TY 担任 Adobe Research 实习生期间开始。我们对与徐怡伦、肖广轩和姜铭国的深入讨论表示感谢。本研究部分得到了 NSF 资助 2105819、1955864 和 2019786（IAIFI）、新加坡 DSTA 资助（DST00OECI20300823，视觉的新表征），以及来自 GIST 和亚马逊的资助。

# References

[1] Siddarth Asokan, Nishanth Shetty, Aadithya Srikanth, and Chandra Sekhar Seelamantula. Gans settle scores! arXiv preprint arXiv:2306.01654, 2023. 3   
[2] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, et al. ediffi: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022. 2, 7   
[3] David Berthelot, Arnaud Autef, Jierui Lin, Dian Ang Yap, Shuangfei Zhai, Siyuan Hu, Daniel Zheng, Walter Talbot, and Eric Gu. Tract: Denoising diffusion models with transitive closure time-distillation. arXiv preprint arXiv:2303.04248, 2023. 2, 3, 6, 14   
[4] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. In ICLR, 2019. 3, 6, 14   
[5] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Textto-image generation via masked generative transformers. In ICML, 2023. 3   
[6] Nanxin Chen, Yu Zhang, Heiga Zen, Ron J Weiss, Mohammad Norouzi, and William Chan. Wavegrad: Estimating gradients for waveform generation. In ICLR, 2021. 2   
[7] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174, 2016. 13   
[8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009. 2, 6   
[9] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In NeurIPS, 2021. 6   
[10] Gintare Karolina Dziugaite, Daniel M Roy, and Zoubin Ghahramani. Training generative neural networks via maximum mean discrepancy optimization. In UAI, 2015. 3   
[11] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In CVPR, 2023. 2   
[12] Jean-Yves Franceschi, Mike Gartrell, Ludovic Dos Santos, Thibaut Issenhuth, Emmanuel de Bézenac, Mickaël Chen, and Alain Rakotomamonjy. Unifying gans and score-based diffusion as generative particle models. In NeurIPS, 2023. 3   
[13] Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Make-a-scene: Scenebased text-to-image generation with human priors. In ECCV, 2022. 7   
[14] Xinyu Gong, Shiyu Chang, Yifan Jiang, and Zhangyang Wang. Autogan: Neural architecture search for generative adversarial networks. In ICCV, 2019. 14   
[15] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative Adversarial Nets. In NIPS, 2014. 1, 2, 3   
[16] Jiatao Gu, Shuangfei Zhai, Yizhe Zhang, Lingjie Liu, and Joshua M Susskind. Boot: Data-free distillation of denoising diffusion models with bootstrapping. In ICML 2023 Workshop on Structured Probabilistic Inference & Generative Modeling, 2023. 2, 3, 6 [17] Amir Hertz, Kfir Aberman, and Daniel Cohen-Or. Delta denoising score. In ICCV, 2023. 3 [18] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017. 6 [19] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. In NeurIPS 2014 Deep Learning Workshop, 2015. 2 [20] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In arXiv preprint arXiv:2207.12598, 2022. 5 [21] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 1, 2, 3, 4,   
5 [22] Jonathan Ho, William Chan, Chitwan Saharia, Jy Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 2 [23] Aapo Hyvärinen and Peter Dayan. Estimation of nonnormalized statistical models by score matching. JMLR,   
2005.2 [24] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In CVPR, 2017. 2, 3 [25] Yifan Jiang, Shiyu Chang, and Zhangyang Wang. Transgan: Two pure transformers can make one strong gan, and that can scale up. In NeurIPS, 2021. 14 [26] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In CVPR, 2023. 2, 3, 7, 13 [27] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of gans for improved quality, stability, and variation. In ICLR, 2018. 2 [28] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In CVPR, 2019. 3 [29] Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Training generative adversarial networks with limited data. In NeurIPS, 2020. 14 [30] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In CVPR, 2020. 2, 3, 14 [31] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In NeurIPS, 2022. 2, 4, 5, 6, 12, 14 [32] Sergey Kastryulin, Jamil Zakirov, Denis Prokopenko, and Dmitry V. Dylov. Pytorch image quality: Metrics for image quality assessment, 2022. 12, 13 [33] Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. In NeurIPS, 2021. 4 [34] Diederik P Kingma and Max Welling. Auto-encoding variational baves, In ICIR 2014 1   
[35] Zhiteng Kong, We1 Ping, J1aj1 Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. In ICLR, 2021. 2   
[36] Alex Krizhevsky et al. Learning multiple layers of features from tiny images. 2009. 2, 6   
[37] Hsin-Ying Lee, Hung-Yu Tseng, Qi Mao, Jia-Bin Huang, Yu-Ding Lu, Maneesh Singh, and Ming-Hsuan Yang. Drit++: Diverse image-to-image translation via disentangled representations. IJCV, 2020. 3   
[38] Kwonjoon Lee, Huiwen Chang, Lu Jiang, Han Zhang, Zhuowen Tu, and Ce Liu. Vitgan: Training gans with vision transformers. In ICLR, 2022. 14   
[39] Yujia Li, Kevin Swersky, and Rich Zemel. Generative moment matching networks. In ICML, 2015. 2, 3   
[40] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 2   
[41] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. In ICLR, 2022. 2, 5, 13   
[42] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR, 2023. 1, 2, 3, 14   
[43] Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, and Qiang Liu. Instaflow: One step is enough for high-quality diffusion-based text-to-image generation. arXiv prent arXiv:2309.06380, 2023. 1, 2, 3, 5, 7   
[44] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019. 12, 13   
[45] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. In NeurIPS, 2022. 2, 14   
[46] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver $^ { + + }$ : Fast solver for guided sampling of diffusion probabilistic models. In arXiv preprint arXiv:2211.01095, 2022. 2, 7, 8, 13, 14   
[47] Eric Luhman and Troy Luhman. Knowledge distillation in iterative generative models for improved sampling speed. arXiv preprint arXiv:2101.02388, 2021. 1, 2, 14   
[48] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. 1, 2, 7, 13   
[49] Simian Luo, Yiqin Tan, Suraj Patil, Daniel Gu, Patrick von Platen, Apolinário Passos, Longbo Huang, Jian Li, and Hang Zhao. Lcm-lora: A universal stable-diffusion acceleration module. arXiv preprint arXiv:2310.04378, 2023. 7, 8, 13   
[50] Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, and Zhihua Zhang. Diff-instruct: A universal approach for transferring knowledge from pre-trained diffusion models. arXiv preprint arXiv:2305.18455, 2023. 3, 6, 14   
[51] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In CVPR, 2023. 1 2 3 6 14   
[52] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In ICML, 2022. 7   
[53] Ollin. Tiny autoencoder for stable diffusion. ht tps : / / github.com/madebyollin/taesd,2023. 13   
[54] Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and Jun-Yan Zhu. Semantic image synthesis with spatially-adaptive normalization. In CVPR, 2019. 3   
[55] Taesung Park, Jun-Yan Zhu, Oliver Wang, Jingwan Lu, Eli Shechtman, Alexei Efros, and Richard Zhang. Swapping autoencoder for deep image manipulation. In NeurIPS, 2020. 3   
[56] Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu. On aliased resizing and surprising subtleties in gan evaluation. In CVPR, 2022. 14   
[57] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 7   
[58] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In ICLR, 2023. 3, 5, 6, 7   
[59] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 6   
[60] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, 2021. 3, 7   
[61] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022. 1, 2, 3, 7   
[62] Scott Reed, Zeynep Akata, Xinchen Yan, Lajanugen Logeswaran, Bernt Schiele, and Honglak Lee. Generative adversarial text to image synthesis. In ICML, 2016. 2   
[63] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 1, 2, 4, 5, 7, 8, 13   
[64] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, 2022. 1, 2, 3, 7   
[65] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In ICLR, 2022. 1, 2, 3, 6, 14   
[66] Axel Sauer, Katja Schwarz, and Andreas Geiger. Styleganxl: Scaling stylegan to large diverse datasets. In SIGGRAPH, 2022. 14   
[67] Axel Sauer, Tero Karras, Samuli Laine, Andreas Geiger, and Timo Aila. Stylegan-t: Unlocking the power of gans for fast large-scale text-to-image svnthesis. ICML. 2023. 3. 7   
[68] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. arXiv preprint arXiv:2311.17042, 2023. 3   
[69] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, etl. Laion-An open largescale dataset for traii next generation image-text models. In NeurIPS, 2022. 3, 6, 7, 13   
[70] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792, 2022. 2   
[71] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2015. 1, 2, 4   
[72] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. 2, 5, 14   
[73] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In NeurIPS, 2019. 2, 4   
[74] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021. 1, 2, 4   
[75] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In ICML, 2023. 1, 2, 3, 5, 6, 12, 13, 14   
[76] Yuan Tian, Qin Wang, Zhiwu Huang, Wen Li, Dengxin Dai, Minghao Yang, Jun Wang, and Olga Fink. Off-policy reinforcement learning for efficient and effective gan architecture search. In ECCV, 2020. 14   
[77] Pascal Vincent. A connection between score matching and denoising autoencoders. Neural Computation, 2011. 5   
[78] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, and Thomas Wolf. Diffusers: State-of-the-art diffusion models. https://github.com/huggingface/ diffusers, 2022. 13   
[79] Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Andrew Tao, Jan Kautz, and Bryan Catanzaro. High-resolution image synthesis and semantic manipulation with conditional gans. In CVPR, 2018. 3   
[80] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. arXiv preprint arXiv:2305.16213, 2023. 2, 3, 4, 5, 6,   
[81] Zhendong Wang, Huangjie Zheng, Pengcheng He, Weizhu Chen, and Mingyuan Zhou. Diffusion-gan: Training gans with diffusion. In ICLR, 2023. 3, 14   
[82] Romann M Weber. The score-difference flow for implicit generative modeling. arXiv preprint arXiv:2304.12906, 2023. 3   
[83] Zhisheng Xiao, Karsten Kreis, and Arash Vahdat. Tackling the generative learning trilemma with denoising diffusion gans. In ICLR, 2022. 2, 3, 14   
[84] Yanwu Xu, Yang Zhao, Zhisheng Xiao, and Tingbo Hou. Ufogen: You forward once large scale text-to-image generation via diffusion gans. arXiv preprint arXiv:2311.09257, 2023. 3, 7   
[85] Senmao Ye and Fei Liu. Score mismatching for generative modeling. arXiv preprint arXiv:2309.11043, 2023. 3, 14   
[86] Mingxuan Yi, Zhanxing Zhu, and Song Liu. Monoflow: Rethinking divergence gans via the perspective of wasserstein gradient flows. In ICML, 2023. 3   
[87] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022. 3, 7   
[88] Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, and Dimitris N Metaxas. Stackgan++: Realistic image synthesis with stacked generative adversarial networks. TPAMI, 2018. 2   
[89] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 3, 5, 14   
[90] Shengyu Zhao, Jonathan Cui, Yilun Sheng, Yue Dong, Xiao Liang, Eric I Chang, and Yan Xu. Large scale image completion via co-modulated generative adversarial networks. In ICLR, 2021. 3   
[91] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. Unipc: A unified predictor-corrector framework for fast sampling of diffusion models. arXiv preprint arXiv:2302.04867, 2023. 1, 2, 7, 8, 13   
[92] Hongkai Zheng, Weili Nie, Arash Vahdat, Kamyar Azizzadenesheli, and Anima Anandkumar. Fast sampling of diffusion models via operator learning. In ICML, 2023. 1, 2, 6, 14   
[93] Hongkai Zheng, Weili Nie, Arash Vahdat, Kamyar Azizzadenesheli, and Anima Anandkumar. Fast sampling of diffusion models via operator learning. In ICML, 2023. 2   
[94] Yufan Zhou, Ruiyi Zhang, Changyou Chen, Chunyuan Li, Chris Tensmeyer, Tong Yu, Jiuxiang Gu, Jinhui Xu, and Tong Sun.Towards language-ree training for text-to-mage generation. In CVPR, 2022. 7   
[95] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycleconsistent adversarial networks. In ICCV, 2017. 3

# Algorithm 2: distributionMatchingLoss

# mu_real, mu_fake: denoising networks for real and fake distribution   
# x: fake sample generated by our one-step generator   
# min_dm_step, max_dm_step: timestep intervals for computing distribution matching loss   
# bs: batch size

# random timesteps

timestep $=$ randint(min_dm_step, max_dm_step, [bs]) noise $=$ randn_like(x)

# Diffuse generated sample by injecting noise # e.g. noise_x = x + noise \* sigma_t (EDM) noisy_x $=$ forward_diffusion(x, noise, timestep)

$\#$ denoise using real and fake denoiser   
with_grad_disabled(): pred_fake_image $=$ mu_fake(noisy_x, timestep) pred_real_image $=$ mu_real(noisy_x, timestep)   
# The weighting_factor diverges slightly from our   
# paper's equation, adapting to accomodate the mean   
# prediction scheme we use here.   
weighting_factor $=$ abs(x - pred_real_image).mean( dim $=$ [1, 2, 3], keepdim=True)   
grad $=$ (pred_fake_image - pred_real_image) / weighting_factor

# the loss that would enforce above grad loss $= \phantom { - } 0 . 5 \phantom { - } \star$ mse_loss(x, stopgrad(x - grad))

# Algorithm 3: denoisingLoss

# pred_fake_image: denoised output by mu_fake on x_t # x: fake sample generated by our one-step generator # weight: weighting strategy(SNR+1/0.5^2 for EDM, SNR for SDvl.5) loss $=$ mean(weight $\star$ (pred_fake_image - x)\*\*2)

# Appendix

# A. Qualitative Speed Comparison

In the accompanying video material, we present a qualitative speed comparison between our one-step generator and the original stable diffusion model. Our one-step generator achieves comparable image quality with the Stable Diffusion model while being around $3 0 \times$ faster.

# B. Implementation Details

For a comprehensive understanding, we include the implementation specifics for constructing the KL loss for the generator $G$ in Algorithm 2 and training the fake score estimator parameterized by $\mu _ { \mathrm { f a k e } }$ in Algorithm 3.

# B.1. CIFAR-10

We distill our one-step generator from EDM [31] pretrained models, specifically utilizing "edm-cifar10-32x32- cond-vp" for class-conditional training and "edm-cifar10- $3 2 \mathrm { x } 3 2 $ -uncond-vp" for unconditional training. We use $\sigma _ { \mathrm { m i n } } = 0 . 0 0 2$ and $\sigma _ { \mathrm { m a x } } = 8 0$ and discretize the noise schedules into $1 0 0 0 { \mathrm { ~ b i n s } }$ . To create our distillation dataset, we generate 100,000 noise-image pairs for class-conditional training and 500,000 for unconditional training. This process utilizes the deterministic Heun sampler (with $S _ { \mathrm { c h u r n } } =$ 0) over 18 steps [31]. For the training phase, we use the AdamW optimizer [44], setting the learning rate at 5e-5, weight decay to 0.01, and beta parameters to (0.9, 0.999). We use a learning rate warmup of 500 steps. The model training is conducted across 7 GPUs, achieving a total batch size of 392. Concurrently, we sample an equivalent number of noise-image pairs from the distillation dataset to calculate the regression loss. Following Song et al. [75], we incorporate the LPIPS loss using a VGG backbone from the PIQ library [32]. Prior to input into the LPIPS network, images are upscaled to a resolution of $2 2 4 \times 2 2 4$ using bilinear upsampling. The regression loss is weighted at 0.25 $( \lambda _ { \mathrm { { r e g } } } = 0 . 2 5 )$ for class-conditional training and at 0.5 $( \lambda _ { \mathrm { { r e g } } } ~ = ~ 0 . 5 )$ for unconditional training. The weights for the distribution matching loss and fake score denoising loss are both set to 1. We train the model for 300,000 iterations and use a gradient clipping with a L2 norm of 10. The dropout is disabled for all networks following consistency model [75].

# B.2. ImageNet ${ \bf \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { 6 4 }$

We distill our one-step generator from EDM [31] pretrained models, specifically utilizing "edm-imagenet-64x64-condadm" for class-conditional training. We use a $\sigma _ { \mathrm { m i n } } = 0 . 0 0 2$ and $\sigma _ { \mathrm { m a x } } = 8 0$ and discretize the noise schedules into 1000 bins. Initially, we prepare a distillation dataset by generating 25,000 noise-image pairs using the deterministic Heun sampler (with $S _ { \mathrm { c h u r n } } = 0 $ )over 256 steps [31]. For the training phase, we use the AdamW optimizer [44], setting the learning rate at 2e-6, weight decay to 0.01, and beta parameters to (0.9, 0.999). We use a learning rate warmup of 500 steps. The model training is conducted across 7 GPUs, achieving a total batch size of 336. Concurrently, we sample an equivalent number of noise-image pairs from the distillation dataset to calculate the regression loss. Following Song et al. [75], we incorporate the LPIPS loss using a VGG backbone from the PIQ library [32]. Prior to input into the LPIPS network, images are upscaled to a resolution of $2 2 4 \times 2 2 4$ using bilinear upsampling. The regression loss is weighted at 0.25 $( \lambda _ { \mathrm { r e g } } = 0 . 2 5 )$ , and the weights for the distribution matching loss and fake score denoising loss are both set to 1. We train the models for 350,000 iterations. We use mixed-precision training and a gradient clipping with a L2 norm of 10. The dropout is disabled for all networks following consistency model [75].

# B.3. LAION-Aesthetic $\mathbf { 6 . 2 5 + }$

We distill our one-step generator from Stable Diffusion v1.5 [63]. We use the LAION-Aesthetic $6 . 2 5 +$ [69] dataset, which contains around 3 million images. Initially, we prepare a distillation dataset by generating 500,000 noiseimage pairs using the deterministic PNMS sampler [41] over 50 steps with a guidance scale of 3. Each pair corresponds to one of the first 500,000 prompts of LAIONAesthetic $6 . 2 5 +$ For the training phase, we use the AdamW optimizer [44], setting the learning rate at 1e-5, weight decay to 0.01, and beta parameters to (0.9, 0.999). We use a learning rate warmup of 500 steps. The model training is conducted across 72 GPUs, achieving a total batch size of 2304. Simultaneously, noise-image pairs from the distillation dataset are sampled to compute the regression loss, with a total batch size of 1152. Given the memory-intensive nature of decoding generated latents into images using the VAE for regression loss computation, we opt for a smaller VAE network [53] for decoding. Following Song et al. [75], we incorporate the LPIPS loss using a VGG backbone from the PIQ library [32]. The regression loss is weighted at 0.25 $\lambda _ { \mathrm { { r e g } } } ~ = ~ 0 . 2 5 )$ , and the weights for the distribution matching loss and fake score denoising loss are both set to 1. We train the model for 20,000 iterations. To optimize GPU memory usage, we implement gradient checkpointing [7] and mixed-precision training. We also apply a gradient clipping with a L2 norm of 10.

# B.4. LAION-Aesthetic $\mathbf { 6 + }$

We distill our one-step generator from Stable Diffusion v1.5 [63]. We use the LAION-Aesthetic $^ { 6 + }$ [69] dataset, comprising approximately 12 million images. To prepare the distillation dataset, we generate 12,000,000 noise-image pairs using the deterministic PNMS sampler [41] over 50 steps with a guidance scale of 8. Each pair corresponds to a prompt from the LAION-Aesthetic $^ { 6 + }$ dataset. For training, we utilize the AdamW optimizer [44], setting the learning rate at 1e-5, weight decay to 0.01, and beta parameters to (0.9, 0.999). We use a learning rate warmup of 500 steps. To optimize GPU memory usage, we implement gradient checkpointing [7] and mixed-precision training. We also apply a gradient clipping with a L2 norm of 10. The training takes two weeks on approximately 80 A100 GPUs. During this period, we made adjustments to the distillation dataset size, the regression loss weight, the type of VAE decoder, and the maximum timestep for the distribution matching loss computation. A comprehensive training log is provided in Table 5. We note that this training schedule, constrained by time and computational resources, may not be the most efficient or optimal.

<table><tr><td>Version #Reg. Pair</td><td>Reg. Weight</td><td></td><td>Max DM Step</td><td>VAE-Type</td><td>DM BS</td><td>Reg. BS</td><td>Cumulative Iter.</td><td>FID</td></tr><tr><td>V1</td><td>2.5M</td><td>0.1</td><td>980</td><td>Small</td><td>32</td><td>16</td><td>5400</td><td>23.88</td></tr><tr><td>V2</td><td>2.5M</td><td>0.5</td><td>980</td><td>Small</td><td>32</td><td>16</td><td>8600</td><td>18.21</td></tr><tr><td>V3</td><td>2.5M</td><td>1</td><td>980</td><td>Small</td><td>32</td><td>16</td><td>21100</td><td>16.10</td></tr><tr><td>V4</td><td>4M</td><td>1</td><td>980</td><td>Small</td><td>32</td><td>16</td><td>56300</td><td>16.86</td></tr><tr><td>V5</td><td>6M</td><td>1</td><td>980</td><td>Small</td><td>32</td><td>16</td><td>60100</td><td>16.94</td></tr><tr><td>V6</td><td>9M</td><td>1</td><td>980</td><td>Small</td><td>32</td><td>16</td><td>68000</td><td>16.76</td></tr><tr><td>V7</td><td>12M</td><td>1</td><td>980</td><td>Small</td><td>32</td><td>16</td><td>74000</td><td>16.80</td></tr><tr><td>V8</td><td>12M</td><td>1</td><td>500</td><td>Small</td><td>32</td><td>16</td><td>80000</td><td>15.61</td></tr><tr><td>V9</td><td>12M</td><td>1</td><td>500</td><td>Large</td><td>16</td><td>4</td><td>127000</td><td>15.33</td></tr><tr><td>V10</td><td>12M</td><td>0.75</td><td>500</td><td>Large</td><td>16</td><td>4</td><td>149500</td><td>15.51</td></tr><tr><td>V11</td><td>12M</td><td>0.5</td><td>500</td><td>Large</td><td>16</td><td>4</td><td>162500</td><td>15.05</td></tr><tr><td>V12</td><td>12M</td><td>0.25</td><td>500</td><td>Large</td><td>16</td><td>4</td><td>165000</td><td>14.93</td></tr></table>

Table 5. Training Logs for the LAION-Aesthetic $^ { 6 + }$ Dataset: 'Max DM step' denotes the highest timestep for noise injection in computing the distribution matching loss. "VAE-Type small" corresponds to the Tiny VAE decoder [53], while "VAE-Type large" indicates the standard VAE decoder used in SDv1.5. "DM BS" denotes the batch size used for the distribution matching loss while "Reg. BS" represents the batch size used for the regression loss.

# C. Baseline Details

# C.1. w/o Distribution Matching Baseline

This baseline adheres to the training settings outlined in Sections B.1 and B.2, with the distribution matching loss omitted.

# C.2. w/o Regression Loss Baseline

Following the training protocols from Sections B.1 and B.2, this baseline excludes the regression loss. To prevent training divergence, the learning rate is adjusted to 1e-5.

# C.3. Text-to-Image Baselines

We benchmark our approach against a variety of models, including the base diffusion model [63], fast diffusion solvers [46, 91], and few-step diffusion distillation baselines [48, 49].

Stable Diffusion We employ the StableDiffusion v1.5 model available on huggingface3, generating images with the PNMS sampler [41] over 50 steps.

Fast Diffusion Solvers We use the UniPC [91] and DPMSolver $^ { + + }$ [46] implementations from the diffusers library [78], with all hyperparameters set to default values.

LCM-LoRA We use the LCM-LoRA SDv1.5 checkpoints hosted on Hugging Face4. As the model is pre-trained with guidance, we do not apply classifier-free guidance during inference.

# D. Evaluation Details

For zero-shot evaluation on COCO, we employ the evaluation code from GigaGAN $[ 2 6 ]$ .Specifically, we generate 30,000 images using random prompts from the MSCOCO2014 validation set. We downsample the generated images from $5 1 2 \times 5 1 2$ to $2 5 6 \times 2 5 6$ using the PIL.Lanczos resizer. These images are then compared with 40,504 real images from the same validation set to calculate the FID metric using the clean-fid [56] library. Additionally, we employ the OpenCLIP-G backbone to compute the CLIP score. For ImageNet and CIFAR-10, we generate 50,000 images for each and calculate their FID using the EDM's evaluation code $[ 3 1 ] ^ { 6 }$ .

# E. CIFAR-10 Experiments

Following the setup outlined in Section B.1, we train our models on CIFAR-10 and conduct comparisons with other competing approaches. Table 6 summarizes the results.

Table 6. Sample quality comparison on CIFAR-10. Baseline numbers are derived from Song et al. [75]. †Methods that use classconditioning.   

<table><tr><td>Family</td><td>Method</td><td># Fwd Pass (↓)</td><td>FID ()</td></tr><tr><td rowspan="10">GAN</td><td>BigGAN† [4]</td><td>1</td><td>14.7</td></tr><tr><td>Diffusion GAN [83]</td><td>1</td><td>14.6</td></tr><tr><td>Diffusion StyleGAN [81]</td><td>1</td><td>3.19</td></tr><tr><td>AutoGAN [14]</td><td>1</td><td>12.4</td></tr><tr><td>E2GAN [76]</td><td>1</td><td>11.3</td></tr><tr><td>ViTGAN [38]</td><td>1</td><td>6.66</td></tr><tr><td>TransGAN [25]</td><td>1</td><td>9.26</td></tr><tr><td>StylegGAN2 [30]</td><td>1</td><td>6.96</td></tr><tr><td>StyleGAN2-ADA† [29]</td><td>1</td><td>2.42</td></tr><tr><td>StyleGAN-XL† [66]</td><td>1</td><td>1.85</td></tr><tr><td rowspan="5">Diffusion + Samplers</td><td>DDIM [72]</td><td>10</td><td>8.23</td></tr><tr><td>DPM-solver-2 [45]</td><td>10</td><td>5.94</td></tr><tr><td>DPM-solver-fast [45]</td><td>10</td><td>4.70</td></tr><tr><td>3-DEIS [92]</td><td>10</td><td>4.17</td></tr><tr><td>DPM-solver++ [46]</td><td>10</td><td>2.91</td></tr><tr><td rowspan="10">Diffusion + Distillation</td><td>Knowledge Distillation [47]</td><td>1</td><td>9.36</td></tr><tr><td>DFNO [92]</td><td>1</td><td>3.78</td></tr><tr><td>1-Rectified Flow (+distill) [42]</td><td>1</td><td>6.18</td></tr><tr><td>2-Rectified Flow (+distill) [42]</td><td>1</td><td>4.85</td></tr><tr><td>3-Rectified Flow (+distill) [42]</td><td>1</td><td>5.21</td></tr><tr><td>Progressive Distillation [65]</td><td>1</td><td>8.34</td></tr><tr><td>Meng et al. [51]†</td><td>1</td><td>5.98</td></tr><tr><td>Diff-Instruct [50]†</td><td>1</td><td>4.19</td></tr><tr><td>Score Mismatching [85]</td><td>1</td><td>8.10</td></tr><tr><td>TRACT [3]</td><td>1 1</td><td>3.78</td></tr><tr><td>DMD (Ours)</td><td>Consistency Model [75]</td><td></td><td>3.55</td></tr><tr><td></td><td></td><td>1</td><td>3.77</td></tr><tr><td></td><td>DMD-conditional (Ours)†</td><td>1</td><td>2.66</td></tr><tr><td>Diffusion</td><td>EDM† (Teacher) [31]</td><td>35</td><td>1.84</td></tr></table>

# F. Derivation for Distribution Matching Gradient

We present the derivation for Equation 7 as follows:

$$
\begin{array} { r l } & { \nabla _ { \theta } D _ { K L } \simeq \underset { z , t , x , x \ t } { \mathbb { E } } \left[ w _ { t } \big ( s _ { \mathrm { f a k e } } ( x _ { t } , t ) - s _ { \mathrm { r e a l } } ( x _ { t } , t ) \big ) \frac { \partial x _ { t } } { \partial \theta } \right] } \\ & { \quad \quad \quad = \underset { z , t , x , x \ t } { \mathbb { E } } \left[ w _ { t } \big ( s _ { \mathrm { f a k e } } ( x _ { t } , t ) - s _ { \mathrm { r e a l } } ( x _ { t } , t ) \big ) \frac { \partial x _ { t } } { \partial G _ { \theta } ( z ) } \frac { \partial G _ { \theta } ( z ) } { \partial \theta } \right] } \\ & { \quad \quad \quad = \underset { z , t , x , x \ t } { \mathbb { E } } \left[ w _ { t } \big ( s _ { \mathrm { f a k e } } ( x _ { t } , t ) - s _ { \mathrm { r e a l } } ( x _ { t } , t ) \big ) \frac { \partial x _ { t } } { \partial x } \frac { \partial G _ { \theta } ( z ) } { \partial \theta } \right] } \\ & { \quad \quad \quad = \underset { z , t , x , x \ t } { \mathbb { E } } \left[ w _ { t } \alpha _ { t } \big ( s _ { \mathrm { f a k e } } ( x _ { t } , t ) - s _ { \mathrm { r e a l } } ( x _ { t } , t ) \big ) \frac { d G } { d \theta } \right] } \end{array}
$$

# G. Prompts for Figure 1

We use the following prompts for Figure 1. From left to right:

•A DSLR photo of a golden retriever in heavy snow.   
•A Lightshow at the Dolomities.   
• A professional portrait of a stylishly dressed elderly woman wearing very large glasses in the style of Iris Apfel, with highly detailed features.   
•Medium shot side profile portrait photo of a warrior chief, sharp facial features, with tribal panther makeup in blue on red, looking away, serious but clear eyes, $5 0 \mathrm { m m }$ portrait, photography, hard rim lighting photography.   
•A hyperrealistic photo of a fox astronaut; perfect face, artstation.

# H. Equivalence of Noise and Data Prediction

The noise prediction model $\boldsymbol { \epsilon } ( x _ { t } , t )$ and data prediction model $\mu ( x _ { t } , t )$ could be converted to each other according to the following rule [31]

$$
\mu ( x _ { t } , t ) = \frac { x _ { t } - \sigma _ { t } \epsilon ( x _ { t } , t ) } { \alpha _ { t } } , \quad \epsilon ( x _ { t } , t ) = \frac { x _ { t } - \alpha _ { t } \mu ( x _ { t } , t ) } { \sigma _ { t } } .
$$

# I. Further Analysis of the Regression Loss

DMD utilizes a regression loss to stabilize training and mitigate mode collapse (Sec. 3.3). In our paper, we mainly adopt the LPIPS [89] distance function, as it has been commonly adopted in prior works. For further analysis, we experiment with a standard L2 distance to train our distilled model on the CIFAR-10 dataset. The model trained using L2 loss achieves an FID score of 2.78, compared to 2.66 with LPIPS, demonstrating the robustness of our method to different loss functions.

# J. More Qualitative Results

We provide additional qualitative results on ImageNet (Fig. 7), LAION (Fig. 8, 9, 10, 11), and CIFAR10 (Fig. 12, 13).

![](images/9.jpg)  
Figure 7. One-step samples from our class-conditional model on ImageNet $\mathrm { F I D } { = } 2 . 6 2$

![](images/10.jpg)  
.

![](images/11.jpg)

![](images/12.jpg)  
"a dog of the german shepherd breed, wearing a space suit, in a post-pocalyptic world, among debris, rubble, dramatic colors, cinematic lighting, vivid colors, sparkling colors, full colors/vector"

![](images/13.jpg)  
"druid portrait by mandy jurgens and warhol, ernst haeckel, james jean, artstation"   
speed $3 0 \times$ faster.

![](images/14.jpg)  
speed $3 0 \times$ faster.

![](images/15.jpg)  
water droplets, bright volumetric lighting,Nikon 200mm lens, 1/8000 sec shutter speed"

![](images/16.jpg)  
  
speed $3 0 \times$ faster.

![](images/17.jpg)  
Figure 12. One-step samples from our class-conditional model on CIFAR-10 $\mathrm { F I D } { = } 2 . 6 6 )$ .

![](images/18.jpg)  
Figure 13. One-step samples from our unconditional model on CIFAR-10 $\mathrm { F I D } { = } 3 . 7 7 $ .