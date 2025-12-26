# 可扩展的带有变换器的扩散模型

威廉·皮布尔斯* 加州大学伯克利分校 谢星宁 纽约大学

![](images/1.jpg)  
F of our class-conditional DiT-XL/2 models trained on ImageNet at $5 1 2 \times 5 1 2$ and $2 5 6 \times 2 5 6$ resolution, respectively.

# 摘要

# 1. 引言

我们探索了一种基于变换器架构的新类扩散模型。我们训练图像的潜在扩散模型，采用变换器替代常用的 U-Net 主干网络，该变换器在潜在补丁上进行操作。我们通过 Gflops 衡量的前向传播复杂性分析了我们的扩散变换器（DiTs）的可扩展性。我们发现，具有更高 Gflops 的 DiTs——通过增加变换器的深度/宽度或增加输入词元的数量——一致地表现出较低的 FID。除了具有良好的可扩展性特征，我们最大的 DiT-XL/2 模型在类条件 ImageNet $5 I 2 \times 5 I 2$ 和 $2 5 6 \times 2 5 6$ 基准测试中超越了所有先前的扩散模型，在后者上达到了最先进的 FID 值 2.27。机器学习正在经历由变换器推动的复兴。在过去五年中，自然语言处理 [8, 42]、视觉 [10] 以及其他多个领域的神经架构在很大程度上被变换器所取代 [60]。尽管如此，许多图像级生成模型的类别仍然拒绝这一趋势——虽然变换器在自回归模型 [3,6,43,47] 中的使用广泛，但在其他生成建模框架中的采用却相对较少。例如，扩散模型在最近图像级生成模型的进展中处于前沿 [9,46]；然而，它们都将卷积 U-Net 架构作为事实上的主干选择。

![](images/2.jpg)  
FID-0 urXL/ Ue MDM.

Ho等人的开创性工作首次为扩散模型引入了U-Net主干网络。在像素级自回归模型和条件GAN中初见成功之后，U-Net继承自Pixel$\mathrm { C N N + + }$，并进行了一些调整。该模型是卷积网络，主要由ResNet模块组成。与标准U-Net相比，在较低分辨率下穿插了额外的空间自注意力模块，这些模块是变换器中的重要组件。Dhariwal和Nichol对U-Net的几个架构选择进行了消融实验，例如使用自适应归一化层注入条件信息及卷积层的通道数。然而，Ho等人提出的U-Net的高层设计基本保持不变。我们的研究旨在揭示扩散模型中架构选择的重要性，并为未来的生成建模研究提供经验基线。我们表明，U-Net的归纳偏差对于扩散模型的性能并不关键，可以很容易地用标准设计如变换器进行替代。因此，扩散模型有望从最近的架构统一趋势中受益，例如，通过继承其他领域的最佳实践和训练配方，并保留可扩展性、鲁棒性和效率等良好特性。标准化架构也将为跨领域研究开辟新可能。本文关注基于变换器的新类扩散模型。我们称之为扩散变换器（Diffusion Transformers），简称DiTs。DiTs遵循视觉变换器（Vision Transformers, ViTs）的最佳实践，已被证明在视觉识别中比传统卷积网络（如ResNet）更有效地扩展。更具体地，我们研究了变换器在网络复杂度与样本质量之间的扩展行为。通过在潜在扩散模型（Latent Diffusion Models, LDMs）框架下构建和基准测试DiT设计空间，我们可以成功地用变换器替代U-Net主干网络。我们进一步表明，DiTs是扩散模型的可扩展架构：网络复杂度（以Gflops为度量）与样本质量（以FID为度量）之间存在强相关性。通过简单地扩大DiT的规模并训练具有高容量主干网络（118.6 Gflops）的LDM，我们在类别条件$2 5 6 \times 2 5 6$ ImageNet生成基准上实现了2.27 FID的最先进结果。

# 2. 相关工作

变换器。变换器在语言、视觉、强化学习和元学习等领域已经取代了特定领域的架构。它们在语言领域的模型规模、训练计算和数据不断增加的情况下展示了显著的扩展性，作为通用自回归模型和视觉变换器（ViTs）。除了语言，变换器还被训练用于自回归地预测像素。它们还在离散代码本上进行训练，作为自回归模型和掩码生成模型；前者在参数达到200亿时展现了优秀的扩展性。最后，变换器在去噪扩散概率模型（DDPMs）中被探索，以合成非空间数据，例如，在DALL·E 2中生成CLIP图像嵌入。在本文中，我们研究变换器作为图像扩散模型主干网络时的扩展性。

![](images/3.jpg)

去噪扩散概率模型（DDPMs）。扩散[19, 54]和基于评分的生成模型[22, 56]在图像生成领域取得了特别成功[35,46,48, 50]，在许多情况下超越了之前最先进的生成对抗网络（GANs）[12]。过去两年中，DDPM 的改进主要得益于采样技术的提升[19, 27, 55]，特别是无分类器引导[21]、将扩散模型重新构造成预测噪声而不是像素[19]，以及使用级联的 DDPM 管道，在此管道中，低分辨率的基础扩散模型与上采样器并行训练[9, 20]。对于上述所有扩散模型，卷积 U-Net[49]是事实上的主干架构选择。相关工作[24]提出了一种基于注意力的新型高效架构用于 DDPM；我们探讨纯变换器。架构复杂性。在评估图像生成文献中的架构复杂性时，使用参数计数是一种相当常见的做法。一般而言，参数计数可能不是图像模型复杂性良好的代理，因为它们并未考虑到例如图像分辨率对性能的显著影响[44, 45]。相反，本文中大部分模型复杂性分析是通过理论 Gflops 的视角进行的，这使我们与架构设计文献保持一致，在该文献中，Gflops 被广泛用于衡量复杂性。在实践中，理想的复杂性指标仍然存在争议，因为它往往依赖于特定的应用场景。Nichol 和 Dhariwal 改进扩散模型的开创性工作[9, 36]与我们最相关——在那里，他们分析了 U-Net 架构类的可扩展性和 Gflop 性能。本文中，我们集中于变换器类。

# 3. 扩散变换器

# 3.1. 初步研究

扩散模型。在介绍我们的架构之前，我们简要回顾一些理解扩散模型（DDPMs）所需的基本概念 [19, 54]。高斯扩散模型假设一种前向添加噪声的过程，该过程逐渐对真实数据 $x _ { 0 }$ 应用噪声：$q ( x _ { t } | x _ { 0 } ) = \mathcal { N } ( x _ { t } ; \sqrt { \bar { \alpha } _ { t } } x _ { 0 } , ( 1 - \bar { \alpha } _ { t } ) \mathbf { I } )$，其中常数 $\bar { \alpha } _ { t }$ 是超参数。通过应用重参数化技巧，我们可以采样 $x _ { t } = \sqrt { \bar { \alpha } _ { t } } x _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon _ { t }$，其中 $\epsilon _ { t } \sim \mathcal { N } ( 0 , \bf { I } )$。

扩散模型的训练目标是学习逆向过程，以逆转前向过程的破坏：$p _ { \theta } ( x _ { t - 1 } | x _ { t } ) \ = \mathcal { N } ( \mu _ { \theta } ( x _ { t } ) , \Sigma _ { \theta } ( x _ { t } ) )$，其中神经网络用于预测$p _ { \theta }$的统计特性。逆向过程模型通过对$x _ { 0 }$的对数似然的变分下界进行训练[30]，其简化为${ \mathcal { L } } ( \theta ) = - p ( x _ { 0 } | x _ { 1 } ) + \begin{array} { r l } { \sum _ { t } \mathcal { D } _ { K L } \big ( q ^ { * } ( x _ { t - 1 } | x _ { t } , x _ { 0 } ) \big | \big | p _ { \theta } \big ( x _ { t - 1 } | x _ { t } \big ) \big ) } & { { } } \end{array}$，不包括对训练无关的附加项。由于$q ^ { * }$和$p _ { \theta }$均为高斯分布，$\mathcal { D } _ { K L }$可以使用两个分布的均值和协方差进行评估。通过将$\mu _ { \theta }$重新参数化为噪声预测网络$\epsilon _ { \theta }$，模型可以通过预测的噪声$\epsilon _ { \theta } ( x _ { t } )$和真实采样高斯噪声$\epsilon _ { t }$之间的简单均方误差进行训练：$\mathcal { L } _ { s i m p l e } ( \theta ) = | | \epsilon _ { \theta } ( x _ { t } ) - \epsilon _ { t } | | _ { 2 } ^ { 2 }$。但是，为了使用学习得到的逆向过程协方差$\Sigma _ { \theta }$训练扩散模型，必须优化整个$\mathcal { D } _ { K L }$项。我们遵循Nichol和Dhariwal的方法[36]：用$\mathcal { L } _ { s i m p l e }$训练$\epsilon _ { \theta }$，并用完整的$\mathcal { L }$训练$\Sigma _ { \theta }$。一旦$p _ { \theta }$被训练，新图像可以通过初始化$x _ { t _ { \operatorname* { m a x } } } \ \sim \ N ( 0 , \mathbf { I } )$并通过重新参数化技巧采样$x _ { t - 1 } \sim p _ { \theta } ( x _ { t - 1 } | x _ { t } )$来生成。

无分类器引导。条件扩散模型将额外信息作为输入，例如类标签 $c$。在这种情况下，反向过程变成 $p _ { \theta } ( x _ { t - 1 } | x _ { t } , c )$，其中 $\epsilon _ { \theta }$ 和 $\Sigma _ { \theta }$ 以 $c$ 为条件。在这种设置中，可以使用无分类器引导来鼓励采样过程找到使得 $\log p ( c | x )$ 较高的 $x$ [21]。根据贝叶斯定理，$\log p ( c | x ) \propto \log p ( x | c ) - \log p ( x )$，因此 $\nabla _ { x } \log p ( c | x ) \propto \nabla _ { x } \log p ( x | c ) - \nabla _ { x } \log p ( x )$。通过将扩散模型的输出解释为分数函数，DDPM 采样过程可以通过以下公式引导以采样具有高 $p ( x | c )$ 的 $x$：$\hat { \epsilon } _ { \theta } ( x _ { t } , c ) ~ = ~ \epsilon _ { \theta } ( x _ { t } , \emptyset ) + s ~ .$ $\nabla _ { \boldsymbol { x } } \log { p ( \boldsymbol { x } | \boldsymbol { c } ) } \propto \epsilon _ { \theta } ( x _ { t } , \emptyset ) + s \cdot ( \epsilon _ { \theta } ( x _ { t } , \boldsymbol { c } ) - \epsilon _ { \theta } ( x _ { t } , \emptyset ) )$，其中 $s > 1$ 表示引导的尺度（注意 $s = 1$ 恢复标准采样）。通过在训练期间随机去掉 $c$，并用学习到的“空”嵌入 $\varnothing$ 替换它，来评估 $c = \emptyset$ 的扩散模型。众所周知，无分类器引导相比通用采样技术显著提高了样本质量 [21, 35, 46]，这一趋势在我们的 DiT 模型中也得到了验证。潜在扩散模型。直接在高分辨率像素空间中训练扩散模型可能会计算上不可行。潜在扩散模型（LDMs）[48] 通过两阶段的方法解决了这个问题：（1）学习一个自动编码器，将图像压缩成更小的空间表示，使用学习到的编码器 $E$；（2）训练一个对表示 $z = E ( x )$ 的扩散模型，而不是对图像 $x$ 的扩散模型（$E$ 被冻结）。然后可以通过从扩散模型中采样表示 $z$，并随后使用学习到的解码器将其解码为图像 $x = D ( z )$ 来生成新图像。如图 2 所示，LDMs 在使用像 ADM 这样的像素空间扩散模型的一部分 Gflops 的情况下，取得了良好的性能。由于我们关注计算效率，这使得它们成为架构探索的吸引起点。在本文中，我们将 DiTs 应用于潜在空间，尽管它们也可以无修改地应用于像素空间。这使我们的图像生成管道成为一种混合基础的方法；我们使用现成的卷积变分自编码器和基于变压器的 DDPMs。

# 3.2. 扩散变换器设计空间

我们引入了扩散变换器（Diffusion Transformers, DiTs），这是一种新的扩散模型架构。我们的目标是尽可能忠实于标准变换器架构，以保留其扩展特性。由于我们关注的是图像的扩散决策过程模型（特别是图像的空间表示），DiT 基于视觉变换器（Vision Transformer, ViT）架构，该架构对图像块序列进行操作 [10]。DiT 保留了 ViT 的许多最佳实践。图 3 显示了完整 DiT 架构的概述。在本节中，我们将描述 DiT 的前向传播过程，以及 DiT 类设计空间的组成部分。

![](images/4.jpg)  
Figure 4. Input specifications for DiT. Given patch size $p \times p$ a spatial representation (the noised latent from the VAE) of shape $I \times I \times C$ is "patchified" into a sequence of length $T = ( I / p ) ^ { 2 }$ with hidden dimension $d$ A smaller patch size $p$ results in a longer sequence length and thus more Gflops.

Patchify。DiT 的输入是一个空间表示 $z$（对于形状为 $2 5 6 \times 2 5 6 \times 3$ 的图像，$z$ 的形状为 $3 2 \times 3 2 \times 4$）。DiT 的第一层是“patchify”，它通过线性嵌入输入中的每个 patch，将空间输入转换为 $T$ 个每个维度为 $d$ 的标记序列。经过 patchify 后，我们对所有输入标记应用标准 ViT 的频率基础位置嵌入（正弦-余弦版本）。通过 patchify 创建的标记数量 $T$ 是由 patch 大小超参数 $p$ 决定的。如图 4 所示，将 $p$ 减半将导致 $T$ 增加四倍，因此至少会将总的 transformer Gflops 增加四倍。尽管这对 Gflops 有显著影响，但需要注意的是，更改 $p$ 对下游参数计数没有实质性影响。我们将 $p = 2, 4, 8$ 添加到 DiT 设计空间中。DiT 块设计。在 patchify 之后，输入标记通过一系列 transformer 块进行处理。除了带噪声的图像输入外，扩散模型有时还会处理额外的条件信息，如噪声时间步 $t$、类别标签 $c$、自然语言等。我们探索了四种不同方式处理条件输入的 transformer 块变体。这些设计对标准 ViT 块设计进行了小但重要的修改。所有块的设计如图 3 所示。上下文条件。在输入序列中，我们直接将 $t$ 和 $c$ 的向量嵌入作为两个额外的标记附加，处理时与图像标记没有区别。这类似于 ViT 中的 ${ \mathsf { C l } }{ \mathsf { S } }$ 标记，它使我们能够使用标准的 ViT 块而无需修改。经过最后一个块后，我们从序列中移除条件标记。这种方法对模型几乎没有引入新的 Gflops。

![](images/5.jpg)  
Figure 5. Comparing different conditioning strategies. adaLNZero outperforms cross-attention and in-context conditioning at all stages of training.

交叉注意力块。我们将$t$和$c$的嵌入拼接成一个长度为二的序列，与图像词元序列分开。变换器块经过修改，在多头自注意力块后添加了一个额外的多头交叉注意力层，类似于Vaswani等人的原始设计，亦和LDM用于类标签条件的设计相似。交叉注意力为模型增加了最多的Gflops，约为$15\%$的开销。自适应层归一化（adaLN）块。在GAN和UNet主干的扩散模型中，自适应归一化层的广泛使用推动了该技术的应用。我们探索用自适应层归一化（adaLN）替换变换器块中的标准层归一化层。我们并不是直接学习逐维缩放和平移参数$\gamma$和$\beta$，而是从$t$和$c$的嵌入向量总和中回归它们。在我们探索的三种块设计中，adaLN增加的Gflops最少，因此是计算效率最高的方案。它也是唯一一个限制在所有词元上应用相同函数的条件机制。adaLN-Zero块。先前对ResNet的研究发现，将每个残差块初始化为恒等映射是有益的。例如，Goyal等人发现，在每个块中将最终批归一化缩放因子$\gamma$进行零初始化，加速了大规模训练。在监督学习设置中，这种方法表现突出。扩散U-Net模型采用了类似的初始化策略，在任何残差连接之前，对每个块中的最终卷积层进行零初始化。我们探索对adaLN DiT块的修改，进行相同的处理。在回归$\gamma$和$\beta$的同时，我们还回归逐维缩放参数$\alpha$，这些参数在DiT块的任何残差连接之前立即应用。

Table 1. Details of DiT models. We follow ViT [10] model configurations for the Small (S), Base (B) and Large (L) variants; we also introduce an XLarge (XL) config as our largest model.   

<table><tr><td>Model</td><td>Layers N</td><td>Hidden size d</td><td>Heads</td><td>Gflops (I=32, p=4)</td></tr><tr><td>DiT-S</td><td>12</td><td>384</td><td>6</td><td>1.4</td></tr><tr><td>DiT-B</td><td>12</td><td>768</td><td>12</td><td>5.6</td></tr><tr><td>DiT-L</td><td>24</td><td>1024</td><td>16</td><td>19.7</td></tr><tr><td>DiT-XL</td><td>28</td><td>1152</td><td>16</td><td>29.1</td></tr></table>

我们初始化多层感知机（MLP）以对所有 $\alpha$ 输出零向量；这将完整的 DiT 块初始化为恒等函数。与原始 adaLN 块一样，adaLNZero 为模型增加的 Gflops 微乎其微。我们在 DiT 设计空间中包含了上下文中的交叉注意力、自适应层标准化和 adaLN-Zero 块。模型大小。我们应用一系列 $N$ 个 DiT 块，每个块的隐藏维度大小为 $d$。遵循 ViT，我们使用标准变换器配置，这些配置联合缩放 $N$、$d$ 和注意力头[10, 63]。具体而言，我们使用四个配置：DiT-S、DiT-B、DiT-L 和 DiT-XL。这些配置涵盖了从 0.3 到 118.6 Gflops 的广泛模型大小和计算资源分配，使我们能够评估缩放性能。表 1 给出了这些配置的详细信息。我们将 B、S、L 和 XL 配置添加到 DiT 设计空间中。变换器解码器。在最后一个 DiT 块之后，我们需要将图像词元序列解码为输出噪声预测和输出对角协方差预测。这两种输出的形状与原始空间输入相等。我们使用标准线性解码器来完成这一任务；我们应用最后的层标准化（如果使用 adaLN 则为自适应）并将每个词元线性解码为一个 $p \times p \times 2 C$ 张量，其中 $C$ 是输入到 DiT 的空间输入的通道数。最后，我们将解码后的词元重新排列为它们原始的空间布局，以获得预测的噪声和协方差。我们探索的完整 DiT 设计空间包括块大小、变换器块架构和模型大小。

# 4. 实验设置

我们探索DiT设计空间，并研究我们模型类别的扩展属性。我们的模型名称根据其配置和潜在补丁大小 $p$ 进行命名；例如，DiT-XL/2 指的是 XLarge 配置和 $p = 2$。训练。我们在ImageNet数据集上以 $256 \times 256$ 和 $512 \times 512$ 的图像分辨率训练条件分类的潜在DiT模型，这是一个高度竞争的生成建模基准。我们将最终线性层初始化为零，其余部分使用ViT的标准权重初始化技术。我们对所有模型使用AdamW进行训练。

![](images/6.jpg)  
F transformer backbone yields better generative models across all model sizes and patch sizes.

我们使用的学习率为 $1 \times 10^{-4}$，没有权重衰减，批量大小为 256。我们唯一使用的数据增强是水平翻转。与之前许多关于 ViTs 的研究不同，我们发现对于训练 DiTs 达到高性能而言，不需要学习率预热或正则化。即使没有这些技术，训练在所有模型配置下都非常稳定，我们没有观察到在训练变换器时常见的损失尖峰。遵循生成建模文献中的常见做法，我们对训练中的 DiT 权重保持一个衰减率为 0.9999 的指数移动平均（EMA）。所有报告的结果均使用 EMA 模型。我们在所有 DiT 模型大小和补丁大小上使用相同的训练超参数。我们的训练超参数几乎完全保留自 ADM。我们没有调整学习率、衰减/预热计划、Adam 的 $\beta_{1}/\beta_{2}$ 或权重衰减。

扩散。我们使用来自稳定扩散的现成预训练变分自编码器（VAE）模型[30][48]。VAE编码器的下采样因子为8—给定一个形状为$256 \times 256 \times 3$的RGB图像$x$， $z = E ( x )$的形状为 $32 \times 32 \times 4$。在本节的所有实验中，我们的扩散模型在这个$\mathcal{Z}$空间中操作。经过从我们的扩散模型中采样一个新的潜变量后，我们使用VAE解码器将其解码为像素 $x = D ( z )$。我们保留了来自ADM[9]的扩散超参数；具体来说，我们使用$t _ { \mathrm { m a x } } = 1000$的线性方差调度，从$1 \times 10 ^ { - 4 }$到$2 \times 10 ^ { - 2 }$，ADM对协方差 $\Sigma _ { \theta }$的参数化及其嵌入输入时间步和标签的方法。评估指标。我们使用Fréchet Inception Distance (FID) [18]来衡量扩展性能，这是评估图像生成模型的标准指标。我们在与先前工作的比较中遵循惯例，并报告使用250个DDPM采样步骤的FID-50K。FID对小的实现细节非常敏感[37]；为确保准确的比较，本文中报告的所有值均通过导出样本并使用ADM的TensorFlow评估套件[9]获得。本节报告的FID数字除非另有说明，否则不使用无分类器引导。我们还报告Inception Score [51]、sFID [34]和精确率/召回率[32]作为附加指标。计算。我们在JAX [1]中实现所有模型，并使用TPU-v3集群进行训练。我们最计算密集的模型DiT-XL/2在具有256的全局批量大小的TPU v3-256集群上训练速度约为5.7次迭代/秒。

# 5. 实验

DiT 块设计。我们训练了四个最高 Gflop 的 DiT-XL/2 模型，每个模型使用不同的块设计——上下文设计（119.4 Gflops）、交叉注意力（137.6 Gflops）、自适应层归一化（adaLN，118.6 Gflops）或 adaLN-zero（118.6 Gflops）。我们在训练过程中测量了 FID。图 5 显示了结果。adaLN-Zero 块的 FID 值低于交叉注意力和上下文条件，同时具有最高的计算效率。在 400K 次训练迭代时，adaLN-Zero 模型的 FID 值几乎是上下文模型的一半，证明了条件机制对模型质量的关键影响。初始化同样重要——adaLN-Zero 将每个 DiT 块初始化为恒等函数，显著优于普通的 adaLN。本文其余部分的所有模型将使用 adaLN-Zero DiT 块。

![](images/7.jpg)

![](images/8.jpg)  
Figure 8. Transformer Gflops are strongly correlated with FID. We plot the Gflops of each of our DiT models and each model's FID-50K after 400K training steps.

模型规模和补丁大小的扩展。我们训练了 12 个 DiT 模型，遍历不同的模型配置（S、B、L、XL）和补丁大小（8、4、2）。请注意，DiT-L 和 DiT-XL 在相对 Gflops 上的距离显著小于其他配置。图 2（左）概述了每个模型的 Gflops 及其在 400K 训练迭代后的 FID。在所有情况下，我们发现增加模型规模和减小补丁大小显著改善了扩散模型。图 6（顶部）展示了在增加模型规模且保持补丁大小不变时 FID 的变化。在所有四个配置中，通过加深和加宽变换器，在训练的各个阶段都获得了显著的 FID 改进。同样，图 6（底部）显示了在保持模型规模不变时补丁大小减小时的 FID 变化。我们再次观察到，通过简单地扩展 DiT 处理的词元数量，同时保持参数大致固定，在整个训练过程中 FID 也有显著改善。

DiT的Gflops对于提升性能至关重要。图6的结果表明，参数数量并不能唯一决定DiT模型的质量。在模型尺寸保持不变的情况下，降低补丁尺寸，变换器的总参数实际上没有明显变化（实际上，总参数略微减少），而只有Gflops增加。这些结果表明，扩大模型的Gflops实际上是提升性能的关键。为了进一步研究，我们在图8中绘制了在400K训练步骤下FID-50K与模型Gflops的关系。结果表明，当不同DiT配置的总Gflops相似时，它们的FID值也接近（例如，DiT-S/2和DiT-B/4）。我们发现模型Gflops与FID-50K之间存在很强的负相关关系，这表明额外的模型计算能力是提升DiT模型的重要因素。在附录的图12中，我们发现这一趋势在其他指标（如Inception Score）上也成立。

![](images/9.jpg)  
Figure 9. Larger DiT models use large compute more efficiently. We plot FID as a function of total training compute.

更大的 DiT 模型在计算效率上更具优势。在图 9 中，我们绘制了所有 DiT 模型的 FID 与总训练计算量的关系。我们通过模型 Gflops $\cdot$ 批量大小 $\cdot$ 训练步数 · 3 来估算训练计算量，其中 3 的系数大致将反向传播的计算量估算为前向传播的两倍。我们发现，即使小型 DiT 模型训练时间更长，相较于训练步数更少的大型 DiT 模型，最终也变得计算效率较低。同样，我们发现除了补丁大小外完全相同的模型，即使在控制训练 Gflops 的情况下，也有不同的性能表现。例如，$\mathrm { X L } / 4$ 在大约 $1 0 ^ { 1 0 }$ Gflops 后被 XL/2 超越。可视化缩放。我们在图 7 中可视化了缩放对样本质量的影响。在 400K 训练步数时，我们使用相同的初始噪声 $x _ { t _ { \operatorname* { m a x } } }$、采样噪声和类别标签从我们的 $1 2 \mathrm { D i T }$ 模型中抽样一张图像。这使我们能够直观地解释缩放如何影响 DiT 的样本质量。实际上，同时扩大模型大小和词元数量显著提高了视觉质量。

# 5.1. 最先进的扩散模型 $2 5 6 \times 2 5 6$ ImageNet。根据我们的规模分析，我们继续训练最高 Gflop 模型 DiT-XL/2，训练 7M 步。我们在图 1 中展示了模型的样本，并与最先进的类条件生成模型进行了比较。我们在表 2 中报告结果。当使用无分类器引导时，DiT-XL/2 超越了所有之前的扩散模型，将 LDM 实现的之前最佳 FID-50K 3.60 降低至 2.27。图 2（右）显示，DiT-XL/2（118.6 Gflops）相较于潜空间 U-Net 模型如 LDM-4（103.6 Gflops）计算效率高，且显著高于像 ADM（1120 Gflops）或 ADM-U（742 Gflops）这样的像素空间 U-Net 模型。

Table 2. Benchmarking class-conditional image generation on ImageNet $2 5 6 \times 2 5 6 .$ DiT-XL/2 achieves state-of-the-art FID.   

<table><tr><td colspan="6">Class-Conditional ImageNet 256×256</td></tr><tr><td>Model</td><td>FID↓</td><td>sFID↓</td><td>IS↑</td><td>Precision↑</td><td>Recall↑</td></tr><tr><td>BigGAN-deep [2]</td><td>6.95</td><td>7.36</td><td>171.4</td><td>0.87</td><td>0.28</td></tr><tr><td>StyleGAN-XL [53]</td><td>2.30</td><td>4.02</td><td>265.12</td><td>0.78</td><td>0.53</td></tr><tr><td>ADM [9]</td><td>10.94</td><td>6.02</td><td>100.98</td><td>0.69</td><td>0.63</td></tr><tr><td>ADM-U</td><td>7.49</td><td>5.13</td><td>127.49</td><td>0.72</td><td>0.63</td></tr><tr><td>ADM-G</td><td>4.59</td><td>5.25</td><td>186.70</td><td>0.82</td><td>0.52</td></tr><tr><td>ADM-G, ADM-U</td><td>3.94</td><td>6.14</td><td>215.84</td><td>0.83</td><td>0.53</td></tr><tr><td>CDM [20]</td><td>4.88</td><td>-</td><td>158.71</td><td>-</td><td>-</td></tr><tr><td>LDM-8 [48]</td><td>15.51</td><td>-</td><td>79.03</td><td>0.65</td><td>0.63</td></tr><tr><td>LDM-8-G</td><td>7.76</td><td>-</td><td>209.52</td><td>0.84</td><td>0.35</td></tr><tr><td>LDM-4</td><td>10.56</td><td>-</td><td>103.49</td><td>0.71</td><td>0.62</td></tr><tr><td>LDM-4-G (cfg=1.25)</td><td>3.95</td><td>-</td><td>178.22</td><td>0.81</td><td>0.55</td></tr><tr><td>LDM-4-G (cfg=1.50)</td><td>3.60</td><td>-</td><td>247.67</td><td>0.87</td><td>0.48</td></tr><tr><td>DiT-XL/2</td><td>9.62</td><td>6.85</td><td>121.50</td><td>0.67</td><td>0.67</td></tr><tr><td>DiT-XL/2-G (cfg=1.25)</td><td>3.22</td><td>5.28</td><td>201.77</td><td>0.76</td><td>0.62</td></tr><tr><td>DiT-XL/2-G (cfg=1.50)</td><td>2.27</td><td>4.60</td><td>278.24</td><td>0.83</td><td>0.57</td></tr></table>

Table 3. Benchmarking class-conditional image generation on ImageNet ${ \bf 5 1 2 } \times { \bf 5 1 2 }$ Note that prior work [9] measures Precision and Recall using 1000 real samples for $5 1 2 \times 5 1 2$ resolution; for consistency, we do the same.   

<table><tr><td colspan="6">Class-Conditional ImageNet 512×512</td></tr><tr><td>Model</td><td>FID↓</td><td>sFID↓</td><td>IS↑</td><td>Precision↑</td><td>Recall↑</td></tr><tr><td>BigGAN-deep [2]</td><td>8.43</td><td>8.13</td><td>177.90</td><td>0.88</td><td>0.29</td></tr><tr><td>StyleGAN-XL [53]</td><td>2.41</td><td>4.06</td><td>267.75</td><td>0.77</td><td>0.52</td></tr><tr><td>ADM [9]</td><td>23.24</td><td>10.19</td><td>58.06</td><td>0.73</td><td>0.60</td></tr><tr><td>ADM-U</td><td>9.96</td><td>5.62</td><td>121.78</td><td>0.75</td><td>0.64</td></tr><tr><td>ADM-G</td><td>7.72</td><td>6.57</td><td>172.71</td><td>0.87</td><td>0.42</td></tr><tr><td>ADM-G, ADM-U</td><td>3.85</td><td>5.86</td><td>221.72</td><td>0.84</td><td>0.53</td></tr><tr><td>DiT-XL/2</td><td>12.03</td><td>7.12</td><td>105.25</td><td>0.75</td><td>0.64</td></tr><tr><td>DiT-XL/2-G (cfg=1.25)</td><td>4.64</td><td>5.77</td><td>174.77</td><td>0.81</td><td>0.57</td></tr><tr><td>DiT-XL/2-G (cfg=1.50)</td><td>3.04</td><td>5.02</td><td>240.82</td><td>0.84</td><td>0.54</td></tr></table>

我们的方法在所有先前的生成模型中达到了最低的 FID，包括之前的最先进模型 StyleGANXL [53]。最后，我们还观察到，与 LDM-4 和 LDM-8 相比，DiT-XL/2 在所有测试的无分类器引导比例下获得了更高的召回值。在仅训练 2.35M 步（类似于 ADM）的情况下，XL/2 仍然以 2.55 的 FID 超越了所有先前的扩散模型。

${ \bf 5 1 2 } \times { \bf 5 1 2 }$ ImageNet。我们在 $5 1 2 \times 5 1 2$ 分辨率上使用相同的超参数训练了一个新的 DiT-XL/2 模型，共计 3M 迭代，数据集为 ImageNet。该 XL/2 模型的补丁大小为 2，在对 $6 4 \times 6 4 \times 4$ 的输入潜变量进行补丁化后，总共处理了 1024 个词元（524.6 Gflops）。表 3 展示了与最先进方法的比较。在该分辨率下，XL/2 再次超越了所有先前的扩散模型，将 ADM 实现的最佳 FID 从 3.85 改进至 3.04。尽管词元数量增加，XL/2 依然保持计算效率。例如，ADM 使用 1983 Gflops，而 ADM-U 使用 2813 Gflops；而 XL/2 仅使用 524.6 Gflops。我们在图 1 和附录中展示了高分辨率 XL/2 模型的样本。

![](images/10.jpg)  
Figure 10. Scaling-up sampling compute does not compensate for a lack of model compute. For each of our DiT models trained for 400K iterations, we compute FID-10K using [16, 32, 64, 128, 256, 1000] sampling steps. For each number of steps, we plot the FID as well as the Gflops used to sample each image. Small models cannot close the performance gap with our large models, even if they sample with more test-time Gflops than the large models.

# 5.2. 模型扩展与计算采样

扩散模型的独特之处在于，它们在训练后可以通过增加生成图像时的采样步骤数量来使用额外的计算资源。鉴于模型的Gflops对样本质量的影响，本节研究较小模型的计算资源（DiTs）是否能够通过使用更多的采样计算超越较大的模型。我们在400K训练步骤后计算了所有12个DiT模型的FID，使用每张图像的[16, 32, 64, 128, 256, 1000]个采样步骤。主要结果见图10。考虑使用1000个采样步骤的DiT-L/2与使用128个步骤的DiT-XL/2。在这种情况下，L/2每张图像使用80.7 Tflops进行采样；而XL/2每张图像只使用$5 \times$更少的计算—15.2 Tflops进行采样。然而，XL/2的FID-10K更优（23.7对25.9）。一般来说，增加采样计算无法弥补模型计算的不足。

# 6. 结论

我们介绍了扩散变换器（Diffusion Transformers，DiTs），这是一个基于变换器的简单主干网络，能够超越以前的U-Net模型，并继承了变换器模型类别的优良扩展特性。鉴于本文中令人鼓舞的扩展结果，未来的研究应继续将DiTs扩展到更大规模的模型和词元数量。DiT还可以作为文本到图像模型（如DALL·E 2和Stable Diffusion）的直接替代主干进行探索。致谢。我们感谢Kaiming He、Ronghang Hu、Alexander Berg、Shoubhik Debnath、Tim Brooks、Ilija Radosavovic和Tete Xiao的有益讨论。William Peebles得到了NSF GRFP的支持。

# References

[1] James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. 6 [2] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale GAN training for high fidelity natural image synthesis. In ICLR, 2019. 5, 9 [3] Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In NeurIPS, 2020. 1 [4] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In CVPR, pages 1131511325, 2022. 2 [5] Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Misha Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. Decision transformer: Reinforcement learning via sequence modeling. In NeurIPS, 2021. 2 [6] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In ICML, 2020. 1, 2 [7] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019. 2 [8] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL-HCT, 2019.   
1 [9] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In NeurIPS, 2021. 1, 2, 3, 5,   
6, 9, 12 [10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2020. 1, 2,   
4,5 [11] Patrick Esser, Robin Rombach, and Björn Ommer. Taming transformers for high-resolution image synthesis, 2020. 2 [12] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In NIPS, 2014.   
3 [13] Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv:1706.02677, 2017.   
5 [14] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In CVPR, pages 1069610706, 2022. 2 [15] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR,   
2016.2   
[16] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016. 12   
[17] Tom Henighan, Jared Kaplan, Mor Katz, Mark Chen, Christopher Hesse, Jacob Jackson, Heewoo Jun, Tom B Brown, Prafulla Dhariwal, Scott Gray, et al. Scaling laws for autoregressive generative modeling. arXiv preprint arXiv:2010.14701, 2020. 2   
[18] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. 2017. 6   
[19] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 2, 3   
[20] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. arXiv:2106.15282, 2021. 3, 9   
[21] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. 3, 4   
[22] Aapo Hyvärinen and Peter Dayan. Estimation of nonnormalized statistical models by score matching. Journal of Machine Learning Research, 6(4), 2005. 3   
[23] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 11251134, 2017. 2   
[24] Allan Jabri, David Fleet, and Ting Chen. Scalable adaptive computation for iterative generation. arXiv preprint arXiv:2212.11972, 2022. 3   
[25] Michael Janner, Qiyang Li, and Sergey Levine. Offline reinforcement learning as one big sequence modeling problem. In NeurIPS, 2021. 2   
[26] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv:2001.08361, 2020. 2, 13   
[27] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In Proc. NeurIPS, 2022. 3   
[28] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In CVPR, 2019. 5   
[29] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015. 5   
[30] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 3, 6   
[31] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In NeurIPS, 2012. 5   
[32] Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. In NeurIPS, 2019. 6   
[33] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv:1711.05101. 2017. 5   
[34] Charlie Nash, Jacob Menick, Sander Dieleman, and Peter W Battaglia. Generating images with sparse representations. arXiv preprint arXiv:2103.03841, 2021. 6   
[35] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv:2112.10741, 2021. 3, 4   
[36] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In ICML, 2021. 3   
[37] Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu. On aliased resizing and surprising subtleties in gan evaluation. In CVPR, 2022. 6   
[38] Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In International conference on machine learning, pages 40554064. PMLR, 2018. 2   
[39] William Peebles, Ilija Radosavovic, Tim Brooks, Alexei Efros, and Jitendra Malik. Learning to learn with generative models of neural network checkpoints. arXiv preprint arXiv:2209.12892, 2022. 2   
[40] Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. In AAAI, 2018. 2, 5   
[41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 2   
[42] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. 2018. 1   
[43] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. 2019. 1   
[44] Ilija Radosavovic, Justin Johnson, Saining Xie, Wan-Yen Lo, and Piotr Dollár. On network design spaces for visual recognition. In ICCV, 2019. 3   
[45] Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, and Piotr Dollár. Designing network design spaces. In CVPR, 2020. 3   
[46] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv:2204.06125, 2022. 1, 2, 3, 4   
[47] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, 2021. 1, 2   
[48] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2, 3, 4, 6, 9   
[49] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234241. Springer, 2015. 2, 3 [50] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. Photorealistic text-toimage diffusion models with deep language understanding. arXiv:2205.11487, 2022. 3 [51] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen, and Xi Chen. Improved techniques for training GANs. In NeurIPS, 2016. 6 [52] Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P Kingma. PixelCNN++: Improving the pixelcnn with discretized logistic mixture likelihood and other modifications. arXiv preprint arXiv:1701.05517, 2017. 2 [53] Axel Sauer, Katja Schwarz, and Andreas Geiger. Styleganxl: Scaling stylegan to large diverse datasets. In SIGGRAPH,   
2022. 9 [54] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2015. 3 [55] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv:2010.02502, 2020. 3 [56] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In NeurIPS, 2019.   
3 [57] Andreas Steiner, Alexander Kolesnikov, Xiaohua Zhai, Ross Wightman, Jakob Uszkoreit, and Lucas Beyer. How to train your ViT? data, augmentation, and regularization in vision transformers. TMLR, 2022.6 [58] Aaron Van den Oord, Nal Kalchbrenner, Lasse Espeholt, Oriol Vinyals, Alex Graves, et al. Conditional image generation with pixelcnn decoders. Advances in neural information processing systems, 29, 2016. 2 [59] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 2 [60] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Eukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 1,   
2,5 [61] Tete Xiao, Piotr Dollar, Mannat Singh, Eric Mintun, Trevor Darrell, and Ross Girshick. Early convolutions help transformers see better. In NeurIPS, 2021. 6 [62] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv:2206.10789, 2022. 2 [63] Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers. In CVPR, 2022. 2,   
5

![](images/11.jpg)  
Figure 11. Additional selected samples from our ${ \bf 5 1 2 } \times { \bf 5 1 2 }$ and $2 5 6 \times 2 5 6$ resolution DiT-XL/2 models. We use a classifier-free guidanc scale of 6.0 for the $5 1 2 \times 5 1 2$ model and 4.0 for the $2 5 6 \times 2 5 6$ model. Both models use the ft-EMA VAE decoder.

# A. Additional Implementation Details

We include detailed information about all of our DiT models in Table 4, including both $2 5 6 \times 2 5 6$ and $5 1 2 \times 5 1 2$ models. In Figure 13, we report DiT training loss curves. Finally, we also include Gflop counts for DDPM U-Net models from ADM and LDM in Table 6.

DiT model details. To embed input timesteps, we use a 256-dimensional frequency embedding [9] followed by a two-layer MLP with dimensionality equal to the transformer's hidden size and SiLU activations. Each adaLN layer feeds the sum of the timestep and class embeddings into a SiLU nonlinearity and a linear layer with output neurons equal to either $4 \times$ (adaLN) or $6 \times$ (adaLN-Zero) the transformer's hidden size. We use GELU nonlinearities (approximated with tanh) in the core transformer [16].

Classifier-free guidance on a subset of channels. In our experiments using classifier-free guidance, we applied guidance only to the first three channels of the latents instead of all four channels. Upon investigating, we found that threechannel guidance and four-channel guidance give similar results (in terms of FID) when simply adjusting the scale factor. Specifically, three-channel guidance with a scale of $( 1 + x )$ appears reasonably well-approximated by fourchannel guidance with a scale of $( 1 + { \frac { 3 } { 4 } } x )$ (e.g., threechannel guidance with a scale of 1.5 gives an FID-50K of 2.27, and four-channel guidance with a scale of 1.375 gives an FID-50K of 2.20). It is somewhat interesting that applying guidance to a subset of elements can still yield good performance, and we leave it to future work to explore this phenomenon further.

# B. Model Samples

We show samples from our two DiT-XL/2 models at $5 1 2 \times 5 1 2$ and $2 5 6 \times 2 5 6$ resolution trained for 3M and 7M steps, respectively. Figures 1 and 11 show selected samples from both models. Figures 14 through 33 show uncurated samples from the two models across a range of classifierfree guidance scales and input class labels (generated with 250 DDPM sampling steps and the ft-EMA VAE decoder). As with prior work using guidance, we observe that larger scales increase visual fidelity and decrease sample diversity.

<table><tr><td>Model</td><td>Image Resolution</td><td>Flops (G)</td><td>Params (M)</td><td>Training Steps (K)</td><td>Batch Size</td><td>Learning Rate</td><td>DiT Block</td><td>FID-50K (no guidance)</td></tr><tr><td>DiT-S/8</td><td>256 × 256</td><td>0.36</td><td>33</td><td>400</td><td>256</td><td>1 × 10−4</td><td>adaLN-Zero</td><td>153.60</td></tr><tr><td>DiT-S/4</td><td>256× 256</td><td>1.41</td><td>33</td><td>400</td><td>256</td><td>1 × 10−4</td><td>adaLN-Zero</td><td>100.41</td></tr><tr><td>DiT-S/2</td><td>256 × 256</td><td>6.06</td><td>33</td><td>400</td><td>256</td><td>1 × 10−4</td><td>adaLN-Zero</td><td>68.40</td></tr><tr><td>DiT-B/8</td><td>256 × 256</td><td>1.42</td><td>131</td><td>400</td><td>256</td><td>1 × 10−4</td><td>adaLN-Zero</td><td>122.74</td></tr><tr><td>DiT-B/4</td><td>256× 256</td><td>5.56</td><td>130</td><td>400</td><td>256</td><td>1 × 10−4</td><td>adaLN-Zero</td><td>68.38</td></tr><tr><td>DiT-B/2</td><td>256 × 256</td><td>23.01</td><td>130</td><td>400</td><td>256</td><td>1 × 10−4</td><td>adaLN-Zero</td><td>43.47</td></tr><tr><td>DiT-L/8</td><td>256 × 256</td><td>5.01</td><td>459</td><td>400</td><td>256</td><td>1 × 10−4</td><td>adaLN-Zero</td><td>118.87</td></tr><tr><td>DiT-L/4</td><td>256 × 256</td><td>19.70</td><td>458</td><td>400</td><td>256</td><td>1 × 10-4</td><td>adaLN-Zero</td><td>45.64</td></tr><tr><td>DiT-L/2</td><td>256 × 256</td><td>80.71</td><td>458</td><td>400</td><td>256</td><td>1 × 10−4</td><td>adaLN-Zero</td><td>23.33</td></tr><tr><td>DiT-XL/8</td><td>256 × 256</td><td>7.39</td><td>676</td><td>400</td><td>256</td><td>1 × 10−4</td><td>adaLN-Zero</td><td>106.41</td></tr><tr><td>DiT-XL/4</td><td>256× 256</td><td>29.05</td><td>675</td><td>400</td><td>256</td><td>1 × 10−4</td><td>adaLN-Zero</td><td>43.01</td></tr><tr><td>DiT-XL/2</td><td>256× 256</td><td>118.64</td><td>675</td><td>400</td><td>256</td><td>1 × 10−4</td><td>adaLN-Zero</td><td>19.47</td></tr><tr><td>DiT-XL/2</td><td>256 × 256</td><td>119.37</td><td>449</td><td>400</td><td>256</td><td>1 × 10−4</td><td>in-context</td><td>35.24</td></tr><tr><td>DiT-XL/2</td><td>256 × 256</td><td>137.62</td><td>598</td><td>400</td><td>256</td><td>1 × 10−4</td><td>cross-attention</td><td>26.14</td></tr><tr><td>DiT-XL/2</td><td>256 × 256</td><td>118.56</td><td>600</td><td>400</td><td>256</td><td>1 × 10−4</td><td>adaLN</td><td>25.21</td></tr><tr><td>DiT-XL/2</td><td>256 × 256</td><td>118.64</td><td>675</td><td>2352</td><td>256</td><td>1 × 10−4</td><td>adaLN-Zero</td><td>10.67</td></tr><tr><td>DiT-XL/2</td><td>256× 256</td><td>118.64</td><td>675</td><td>7000</td><td>256</td><td>1 × 10−4</td><td>adaLN-Zero</td><td>9.62</td></tr><tr><td>DiT-XL/2</td><td>512 × 512</td><td>524.60</td><td>675</td><td>1301</td><td>256</td><td>1 × 10−4</td><td>adaLN-Zero</td><td>13.78</td></tr><tr><td>DiT-XL/2</td><td>512 × 512</td><td>524.60</td><td>675</td><td>3000</td><td>256</td><td>1 × 10−4</td><td>adaLN-Zero</td><td>11.93</td></tr></table>

T encoder and decoder. For both the $2 5 6 \times 2 5 6$ and $5 1 2 \times 5 1 2$ DT-XL/2 models, we never observed FID saturate and continued training them as long as possible. Numbers reported in this table use the ft-MSE VAE decoder.

# C. Additional Scaling Results

Impact of scaling on metrics beyond FID. In Figure 12, we show the effects of DiT scale on a suite of evaluation metrics—FID, sFID, Inception Score, Precision and Recall. We find that our FID-driven analysis in the main paper generalizes to the other metrics—across every metric, scaled-up DiT models are more compute-efficient and model Gflops are highly-correlated with performance. In particular, Inception Score and Precision benefit heavily from increased model scale.

Impact of scaling on training loss. We also examine the impact of scale on training loss in Figure 13. Increasing DiT model Gflops (via transformer size or number of input tokens) causes the training loss to decrease more rapidly and saturate at a lower value. This phenomenon is consistent with trends observed with language models, where scaledup transformers demonstrate both improved loss curves as well as improved performance on downstream evaluation suites [26].

# D. VAE Decoder Ablations

We used off-the-shelf, pre-trained VAEs across our experiments. The VAE models (ft-MSE and ft-EMA) are finetuned versions of the original LDM "f8" model (only the decoder weights are fine-tuned). We monitored metrics for our scaling analysis in Section 5 using the ft-MSE decoder, and we used the ft-EMA decoder for our final metrics reported in Tables 2 and 3. In this section, we ablate three different choices of the VAE decoder; the original one used by LDM and the two fine-tuned decoders used by Stable Diffusion. Because the encoders are identical across models, the decoders can be swapped-in without retraining the diffusion model. Table 5 shows results; XL/2 continues to outperform all prior diffusion models when using the LDM decoder.

Table 5. Decoder ablation. We tested different pre-trained VAE decoder weights available at https : / /huggingface. co/ stabilityai/sd-vae-ft-mse. Different pre-trained decoder weights yield comparable results on ImageNet $2 5 6 \times 2 5 6$ .   

<table><tr><td colspan="6">Class-Conditional ImageNet 256× 256, DiT-XL/2-G (cfg=1.5)</td></tr><tr><td>Decoder</td><td>FID↓</td><td>sFID↓</td><td>IS↑</td><td>Precision↑</td><td>Recall↑</td></tr><tr><td>original</td><td>2.46</td><td>5.18</td><td>271.56</td><td>0.82</td><td>0.57</td></tr><tr><td>ft-MSE</td><td>2.30</td><td>4.73</td><td>276.09</td><td>0.83</td><td>0.57</td></tr><tr><td>ft-EMA</td><td>2.27</td><td>4.60</td><td>278.24</td><td>0.83</td><td>0.57</td></tr></table>

Table 6. Gflop counts for baseline diffusion models that use UNet backbones. Note that we only count Flops for DDPM components.   

<table><tr><td colspan="5">Diffusion U-Net Model Complexities</td></tr><tr><td>Model</td><td>Image Resolution</td><td>Base Flops (G)</td><td>Upsampler Flops (G)</td><td>Total Flops (G)</td></tr><tr><td>ADM</td><td>128 × 128</td><td>307</td><td></td><td>307</td></tr><tr><td>ADM</td><td>256 × 256</td><td>1120</td><td></td><td>1120</td></tr><tr><td>ADM</td><td>512 × 512</td><td>1983</td><td>-</td><td>1983</td></tr><tr><td>ADM-U</td><td>256 × 256</td><td>110</td><td>632</td><td>742</td></tr><tr><td>ADM-U</td><td>512 × 512</td><td>307</td><td>2506</td><td>2813</td></tr><tr><td>LDM-4</td><td>256 × 256</td><td>104</td><td>-</td><td>104</td></tr><tr><td>LDM-8</td><td>256 × 256</td><td>57</td><td>-</td><td>57</td></tr></table>

![](images/12.jpg)

![](images/13.jpg)  
mean-squared error and $\mathcal { D } _ { K L }$ ). We also highlight early training behavior. Note that scaled-up DiT models exhibit lower training losses.

![](images/14.jpg)  
Figure 14. Uncurated $5 1 2 \times 5 1 2$ DiT-XL/2 samples. Classifier-free guidance scale $= 4 . 0$ Class label $=$ "arctic wolf" (270)

![](images/15.jpg)  
Figure 15. Uncurated $5 1 2 \times 5 1 2$ DiT-XL/2 samples. Classifier-free guidance scale $= 4 . 0$ Class label $=$ "volcano" (980)

![](images/16.jpg)  
Figure 16. Uncurated $5 1 2 \times 5 1 2$ DiT-XL/2 samples. Classifier-free guidance scale $= 4 . 0$ Class label $=$ "husky" (250)

![](images/17.jpg)  
Figure 17. Uncurated $5 1 2 \times 5 1 2$ DiT-XL/2 samples. Classifier-free guidance scale $= 4 . 0$ Class label $=$ "sulphur-crested cockatoo" (89)

![](images/18.jpg)  
Figure 18. Uncurated $5 1 2 \times 5 1 2$ DiT-XL/2 samples. Classifier-free guidance scale $= 4 . 0$ Class label $=$ "cliff drop-off" (972)

![](images/19.jpg)  
Figure 19. Uncurated $5 1 2 \times 5 1 2$ DiT-XL/2 samples. Classifier-free guidance scale $= 4 . 0$ Class label $=$ "balloon" (417)

![](images/20.jpg)  
Figure 20. Uncurated $5 1 2 \times 5 1 2$ DiT-XL/2 samples. Classifier-free guidance scale $= 4 . 0$ Class label $=$ "lion" (291)

![](images/21.jpg)  
Figure 21. Uncurated $5 1 2 \times 5 1 2$ DiT-XL/2 samples. Classifier-free guidance scale $= 4 . 0$ Class label $=$ "otter" (360)

![](images/22.jpg)  
Figure 22. Uncurated $5 1 2 \times 5 1 2$ DiT-XL/2 samples. Classifier-free guidance scale $= 2 . 0$ Class label $=$ "red panda" (387)

![](images/23.jpg)  
Figure 23. Uncurated $5 1 2 \times 5 1 2$ DiT-XL/2 samples. Classifier-free guidance scale $= 2 . 0$ Class label $=$ "panda" (388)

![](images/24.jpg)  
Figure 24. Uncurated $5 1 2 \times 5 1 2$ DiT-XL/2 samples. Classifier-free guidance scale $= 1 . 5$ Class label $=$ "coral reef" (973)

![](images/25.jpg)  
Figure 25. Uncurated $5 1 2 \times 5 1 2$ DiT-XL/2 samples. Classifier-free guidance scale $= 1 . 5$ Class label $=$ "macaw" (88)

![](images/26.jpg)  
Figure 26. Uncurated $2 5 6 \times 2 5 6$ DiT-XL/2 samples. Classifier-free guidance scale $= 4 . 0$ Class label $=$ "macaw" (88)

![](images/27.jpg)  
Figure 27. Uncurated $2 5 6 \times 2 5 6$ DiT-XL/2 samples. Classifier-free guidance scale $= 4 . 0$ Class label $=$ "dog sled" (537)

![](images/28.jpg)  
Figure 28. Uncurated $2 5 6 \times 2 5 6$ DiT-XL/2 samples. Classifier-free guidance scale $= 4 . 0$ Class label $=$ "arctic fox" (279)

![](images/29.jpg)  
Figure 29. Uncurated $2 5 6 \times 2 5 6$ DiT-XL/2 samples. Classifier-free guidance scale $= 4 . 0$ Class label $=$ "loggerhead sea turtle" (33)

![](images/30.jpg)  
Figure 30. Uncurated $2 5 6 \times 2 5 6$ DiT-XL/2 samples. Classifier-free guidance scale $= 2 . 0$ Class label $=$ "golden retriever" (207)

![](images/31.jpg)  
Figure 31. Uncurated $2 5 6 \times 2 5 6$ DiT-XL/2 samples. Classifier-free guidance scale $= 2 . 0$ Class label $=$ "lake shore" (975)

![](images/32.jpg)  
Figure 32. Uncurated $2 5 6 \times 2 5 6$ DiT-XL/2 samples. Classifier-free guidance scale $= 1 . 5$ Class label $=$ "space shuttle" (812)

![](images/33.jpg)  
Figure 33. Uncurated $2 5 6 \times 2 5 6$ DiT-XL/2 samples. Classifier-free guidance scale $= 1 . 5$ Class label $=$ "ice cream" (928)