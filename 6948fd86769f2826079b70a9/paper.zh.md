# 扩展整流流变压器以实现高分辨率图像合成

帕特里克·埃瑟* 苏米特·库拉尔 安德烈亚斯·布拉特曼 拉希姆·恩特扎里 乔纳斯·穆勒 哈里·萨伊尼 雅恩·莱维 多米尼克·洛伦茨 阿克塞尔·绍尔 弗雷德里克·博泽尔 达斯汀·波德尔 蒂姆·多克霍恩 锡安·英格利什 凯尔·莱西 亚历克斯·古德温 雅尼克·马雷克 罗宾·罗姆巴赫 * 稳定AI

![](images/1.jpg)  
and spatial reasoning, attention to fine details, and high image quality across a wide variety of styles.

# 摘要

扩散模型通过反转数据向噪声的前向路径，从噪声中生成数据，已成为高维感知数据（如图像和视频）的强大生成建模技术。校正流是一种近期提出的生成模型形式，通过直线连接数据和噪声。尽管它在理论属性和概念上更为简单，但尚未被确定为标准实践。在本研究中，我们通过将现有的噪声采样技术偏向于感知相关的尺度，改进了校正流模型的训练方法。通过大规模研究，我们展示了该方法相较于现有扩散形式在高分辨率文本到图像合成中的优越性能。此外，我们提出了一种基于变换器的新架构，用于文本到图像生成，该架构对两种模态使用独立的权重，并能够实现图像和文本词元之间的信息双向流动，从而改善文本理解、排版和人类偏好评分。我们证明该架构遵循可预测的扩展趋势，并将更低的验证损失与各种指标和人类评估下的改善文本到图像合成关联起来。我们最大的模型超越了最先进的模型，且将公开我们的实验数据、代码和模型权重。

# 1. 引言

扩散模型从噪声中生成数据（Song et al., 2020）。它们被训练以反转数据的前向路径朝向随机噪声，因此，在神经网络的近似和泛化特性支持下，可以生成训练数据中不存在但遵循训练数据分布的新数据点（Sohl-Dickstein et al., 2015；Song & Ermon, 2020）。这个生成建模技术已被证明在建模高维感知数据（如图像）方面非常有效（Ho et al., 2020）。近年来，扩散模型已成为从自然语言输入生成高分辨率图像和视频的公认方法，展现出令人印象深刻的泛化能力（Saharia et al., 2022b；Ramesh et al., 2022；Rombach et al., 2022；Podell et al., 2023；Dai et al., 2023；Esser et al., 2023；Blattmann et al., 2023b；Betker et al., 2023；Blattmann et al., 2023a；Singer et al., 2022）。由于其迭代特性及相关的计算成本，以及推理过程中较长的采样时间，对这些模型更高效训练和/或更快采样的研究逐渐增多（Karras et al., 2023；Liu et al., 2022）。虽然为数据到噪声指定前向路径可以实现高效训练，但这也带来了选择哪个路径的问题。这个选择对采样可能产生重要影响。例如，无法从数据中去除所有噪声的前向过程会导致训练和测试分布之间的不一致，从而产生灰色图像样本等伪影（Lin et al., 2024）。重要的是，前向过程的选择也影响学习到的反向过程，从而影响采样效率。虽然弯曲路径需要多个积分步骤来模拟过程，但直线路径可以通过单个步骤进行模拟，且不易造成误差积累。由于每个步骤对应于神经网络的评估，这对采样速度有直接影响。前向路径的一个特定选择是所谓的修正流（Rectified Flow）（Liu et al., 2022；Albergo & Vanden-Eijnden, 2022；Lipman et al., 2023），它在直线上连接数据与噪声。尽管这一模型类别在理论上具有更好的特性，但在实践中尚未得到决定性的确立。到目前为止，在小型和中型实验中已实证证明了一些优势（Ma et al., 2024），但这些大多仅限于类条件模型。在本研究中，我们通过引入修正流模型中的噪声尺度重加权，类似于噪声预测扩散模型（Ho et al., 2020），改变了这一情况。通过大规模研究，我们将新的公式与现有的扩散公式进行比较，并展示其优点。我们表明，广泛使用的文本到图像合成方法，其中固定的文本表示直接输入模型（例如，通过交叉注意力（Vaswani et al., 2017；Rombach et al., 2022）），并非理想，并提出了一种新架构，结合了用于图像和文本词元的可学习流，从而实现了它们之间的信息双向流动。我们将其与改进的修正流公式相结合，调查其可扩展性。我们在验证损失中演示了可预测的扩展趋势，并表明较低的验证损失与改进的自动和人工评估高度相关。我们的最大模型在定量评估（Ghosh et al., 2023）提示理解和人类偏好评分上超越了最先进的开放模型，如SDXL（Podell et al., 2023）、SDXL-Turbo（Sauer et al., 2023）、Pixart- $\alpha$（Chen et al., 2023），以及闭源模型如DALL-E 3（Betker et al., 2023）。

我们工作的核心贡献包括：(i) 我们对不同的扩散模型和修正流的公式进行了大规模、系统性的研究，以识别最佳设置。为此，我们引入了新的噪声采样器，用于修正流模型，从而在性能上优于之前已知的采样器。(ii) 我们设计了一种新颖、可扩展的文本到图像合成架构，允许网络内部文本和图像词元流之间的双向混合。我们展示了与已建立的主干网络（如 UViT (Hoogeboom et al., 2023) 和 DiT (Peebles & Xie, 2023)）相比的优势。最后，我们 (iii) 对我们的模型进行扩展研究，证明它遵循可预测的扩展趋势。我们展示了较低的验证损失与通过 T2I-CompBench (Huang et al., 2023)、GenEval (Ghosh et al., 2023) 及人类评分等指标评估的文本到图像性能改善之间的强相关性。我们将结果、代码和模型权重公开发布。

# 2. 无需模拟的流训练

我们考虑生成模型，这些模型通过普通微分方程（ODE）定义了从噪声分布 $p _ { 1 }$ 的样本 $x _ { 1 }$ 到数据分布 $p _ { 0 }$ 的样本 $x _ { 0 }$ 的映射，其中速度 $v$ 由神经网络的权重 $\Theta$ 参数化。Chen 等人（2018）的先前工作建议通过可微分的 ODE 求解器直接求解方程（1）。然而，这个过程计算开销较大，特别是对于那些参数化 $v _ { \Theta } ( y _ { t } , t )$ 的大规模网络架构。一个更高效的替代方案是直接回归一个向量场 $u _ { t }$，该向量场在 $p _ { 0 }$ 和 $p _ { 1 }$ 之间生成概率路径。为了构建这样的 $u _ { t }$，我们定义一个前向过程，对应于在 $p _ { 0 }$ 和 $p _ { 1 } = \mathcal { N } ( 0 , 1 )$ 之间的概率路径 $p _ { t }$，如下所示：

$$
d y _ { t } = v _ { \Theta } ( y _ { t } , t ) d t \ ,
$$

$$
z _ { t } = a _ { t } x _ { 0 } + b _ { t } \epsilon \quad \mathrm { w h e r e } \ \epsilon \sim { \mathcal N } ( 0 , I ) .
$$

对于 $a_{0} = 1, b_{0} = 0, a_{1} = 0$ 和 $b_{1} = 1$，边缘概率与数据和噪声分布是一致的。

$$
p _ { t } ( z _ { t } ) = \mathbb { E } _ { \epsilon \sim \mathcal { N } ( 0 , I ) } p _ { t } ( z _ { t } | \epsilon ) ~ ,
$$

为了表达 $z _ { t }$、$x _ { 0 }$ 和 $\epsilon$ 之间的关系，我们引入 $\psi _ { t }$ 和 $u _ { t }$，定义为

$$
\begin{array} { r } { \psi _ { t } ( \cdot | \epsilon ) : x _ { 0 } \mapsto a _ { t } x _ { 0 } + b _ { t } \epsilon } \\ { u _ { t } ( z | \epsilon ) : = \psi _ { t } ^ { \prime } ( \psi _ { t } ^ { - 1 } ( z | \epsilon ) | \epsilon ) } \end{array}
$$

由于 $z _ { t }$ 可以表示为常微分方程的解 $z _ { t } ^ { \prime } = u _ { t } ( z _ { t } | \epsilon )$，初始条件为 $z _ { 0 } ~ = ~ x _ { 0}$，因此 $u _ { t } ( \cdot | \epsilon )$ 生成了 $p _ { t } ( \cdot | \epsilon )$。值得注意的是，可以构造一个边际向量场 $u _ { t }$ 来生成边际概率路径 $p _ { t }$（Lipman et al., 2023）（见 B.1），此向量场使用条件向量场 $u _ { t } ( \cdot | \epsilon )$。

$$
u _ { t } ( z ) = \mathbb { E } _ { \epsilon \sim \mathcal { N } ( 0 , I ) } u _ { t } ( z | \epsilon ) \frac { p _ { t } ( z | \epsilon ) } { p _ { t } ( z ) }
$$

由于公式6中的边际化，直接使用流匹配目标回归 $u _ { t }$ 是不可行的。条件流匹配（见 B.1）使用条件向量场 $u _ { t } ( z | \epsilon )$ 提供了一个等效但可行的目标。

$$
\mathcal { L } _ { F M } = \mathbb { E } _ { t , p _ { t } ( z ) } | | v _ { \Theta } ( z , t ) - u _ { t } ( z ) | | _ { 2 } ^ { 2 } .
$$

$$
\mathcal { L } _ { C F M } = \mathbb { E } _ { t , p _ { t } ( z | \epsilon ) , p ( \epsilon ) } | | v _ { \Theta } ( z , t ) - u _ { t } ( z | \epsilon ) | | _ { 2 } ^ { 2 } ,
$$

为了将损失转换为显式形式，我们插入 $\psi _ { t } ^ { \prime } ( x _ { 0 } | \epsilon ) = a _ { t } ^ { \prime } x _ { 0 } + b _ { t } ^ { \prime } \epsilon$ 和 $\begin{array} { r } { \psi _ { t } ^ { - 1 } ( z | \dot { \epsilon } ) = \frac { z - b _ { t } \epsilon } { a _ { t } } } \end{array}$ 到 (5) 中。

$$
z _ { t } ^ { \prime } = u _ { t } ( z _ { t } | \epsilon ) = \frac { a _ { t } ^ { \prime } } { a _ { t } } z _ { t } - \epsilon b _ { t } ( \frac { a _ { t } ^ { \prime } } { a _ { t } } - \frac { b _ { t } ^ { \prime } } { b _ { t } } ) .
$$

现在考虑 $\lambda _ { t } : = \log { \frac { a _ { t } ^ { 2 } } { b _ { t } ^ { 2 } } }$，并且 $\lambda _ { t } ^ { \prime } = 2 ( \frac { a _ { t } ^ { \prime } } { a _ { t } } - \frac { b _ { t } ^ { \prime } } { b _ { t } } )$。

$$
u _ { t } ( z _ { t } | \epsilon ) = \frac { a _ { t } ^ { \prime } } { a _ { t } } z _ { t } - \frac { b _ { t } } { 2 } \lambda _ { t } ^ { \prime } \epsilon
$$

接下来，我们使用方程 (10) 将方程 (8) 重新参数化为噪声预测目标：

$$
\begin{array} { r } { \mathcal { L } _ { C F M } = \mathbb { E } _ { t , p _ { t } ( z | \epsilon ) , p ( \epsilon ) } | | v _ { \Theta } ( z , t ) - \frac { a _ { t } ^ { \prime } } { a _ { t } } z + \frac { b _ { t } } { 2 } \lambda _ { t } ^ { \prime } \epsilon | | _ { 2 } ^ { 2 } } \\ { = \mathbb { E } _ { t , p _ { t } ( z | \epsilon ) , p ( \epsilon ) } \left( - \frac { b _ { t } } { 2 } \lambda _ { t } ^ { \prime } \right) ^ { 2 } | | \epsilon _ { \Theta } ( z , t ) - \epsilon | | _ { 2 } ^ { 2 } } \end{array}
$$

我们定义 $\begin{array} { r } { \epsilon _ { \Theta } : = \frac { - 2 } { \lambda _ { t } ^ { \prime } b _ { t } } ( v _ { \Theta } - \frac { a _ { t } ^ { \prime } } { a _ { t } } z ) } \end{array}$，可以推导出各种加权损失函数，这些损失函数为期望解提供了信号，但可能会影响优化轨迹。为了对不同的方法进行统一分析，包括经典扩散公式，我们可以将目标写成以下形式（参考 Kingma & Gao (2023)）：

$$
\mathcal { L } _ { w } ( x _ { 0 } ) = - \frac { 1 } { 2 } \mathbb { E } _ { t \sim \mathcal { U } ( t ) , \epsilon \sim \mathcal { N } ( 0 , I ) } \left[ w _ { t } \lambda _ { t } ^ { \prime } \| \epsilon _ { \Theta } ( z _ { t } , t ) - \epsilon \| ^ { 2 } \right] ,
$$

其中 $w_{t} = - \textstyle { \frac { 1 } { 2 } } \lambda_{t}^{\prime} b_{t}^{2}$ 对应于 $\mathcal{L}_{C F M}$。

# 3. 流动轨迹

在本研究中，我们考虑上述形式主义的不同变体，以下是简要描述。修正流 修正流（RFs）（Liu 等，2022；Albergo & Vanden-Eijnden，2022；Lipman 等，2023）将前向过程定义为数据分布与标准正态分布之间的直线路径，即，并使用 $\mathcal { L } _ { C F M }$，该部分对应于 $\begin{array} { r } { w _ { t } ^ { \mathrm { R F } } = \frac { t } { 1 - t } } \end{array}$ 网络输出直接参数化速度 $v _ { \Theta }$。

$$
z _ { t } = ( 1 - t ) x _ { 0 } + t \epsilon ,
$$

EDM（Karras 等，2022）使用如下形式的前向过程，其中 （Kingma & Gao，2023） $b _ { t } = \exp { F _ { \mathcal { N } } ^ { - 1 } ( t | P _ { m } , P _ { s } ^ { 2 } ) }$，$F _ { \mathcal { N } } ^ { - 1 }$ 表示均值为 $P _ { m }$ 和方差为 $P _ { s } ^ { 2 }$ 的分布。请注意，这一选择会导致

$$
z _ { t } = x _ { 0 } + b _ { t } \epsilon
$$

请注意，当引入时间依赖加权时，上述目标的最优解并不会改变。因此，

$$
\lambda _ { t } \sim \mathcal { N } ( - 2 P _ { m } , ( 2 P _ { s } ) ^ { 2 } ) \quad \mathrm { f o r } t \sim \mathcal { U } ( 0 , 1 )
$$

网络通过 $\mathbf { F }$ -预测进行参数化（Kingma & Gao, 2023；Karras 等，2022），损失可以表示为 $\mathcal { L } _ { w _ { t } ^ { \mathrm { E D M } } }$，其中

$$
w _ { t } ^ { \mathrm { E D M } } = \mathcal { N } ( \lambda _ { t } | - 2 P _ { m } , ( 2 P _ { s } ) ^ { 2 } ) ( e ^ { - \lambda _ { t } } + 0 . 5 ^ { 2 } )
$$

Cosine（Nichol & Dhariwal, 2021）提出了一种形式的前向过程

$$
z _ { t } = \cos \big ( { \frac { \pi } { 2 } } t \big ) x _ { 0 } + \sin \big ( { \frac { \pi } { 2 } } t \big ) \epsilon \ .
$$

结合$\epsilon$-参数化和损失，这对应于加权$w_{t} = \mathrm{sech}(\lambda_{t} / 2)$。当与$\mathbf{v}$-预测损失（Kingma & Gao, 2023）结合时，加权为$w_{t} = e^{-\lambda_{t} / 2}$。

(LDM-)线性LDM（Rombach等，2022）使用了DDPM调度的修改（Ho等，2020）。两者都是保持方差的调度，即 $b _ { t } = \sqrt { 1 - a _ { t } ^ { 2 } }$，并且为离散时间步 $t = 0 , \ldots , T - 1$ 定义 $a _ { t }$，其形式与扩散系数 $\beta _ { t }$ 相关，表示为 $\begin{array} { r } { a _ { t } \ = \ ( \prod _ { s = 0 } ^ { t } ( 1 - \beta _ { s } ) ) ^ { \frac { 1 } { 2 } } } \end{array}$。对于给定的边界值 $\beta _ { 0 }$ 和 $\beta _ { T - 1 }$，DDPM 使用 $\begin{array} { r c l } { \beta _ { t } } & { = } & { \beta _ { 0 } + \frac { t } { T - 1 } ( \beta _ { T - 1 } - \beta _ { 0 } ) } \end{array}$，而LDM使用 $\begin{array} { r l } { \beta _ { t } } & { { } = } \end{array}$ $\begin{array} { r } { \left( \sqrt { \beta _ { 0 } } + \frac { t } { T - 1 } ( \sqrt { \beta _ { T - 1 } } - \sqrt { \beta _ { 0 } } ) \right) ^ { : 2 } } \end{array}$。

# 3.1. 针对射频模型的定制化信噪比采样器

RF损失在$[0, 1]$内的所有时间步上均匀训练速度$v _ { \Theta }$。然而，直观上，得到的速度预测目标$\epsilon - x _ { 0 }$在$[0, 1]$中间的$t$上更难，因为对于$t = 0$，最佳预测是$p _ { 1 }$的均值，对于$t = 1$，最佳预测是$p _ { 0 }$的均值。一般来说，将$t$的分布从常用的均匀分布$\mathcal { U } ( t )$更改为具有密度$\pi ( t )$的分布，相当于一个加权损失$\mathcal { L } _ { w _ { t } ^ { \pi } }$。

$$
w _ { t } ^ { \pi } = \frac { t } { 1 - t } \pi ( t )
$$

因此，我们旨在通过更频繁地采样中间时间步来给予更多权重。接下来，我们描述用于训练模型的时间步密度 $\pi ( t )$。Logit-Normal 采样 将中间步骤赋予更多权重的一种分布选项是 Logit-Normal 分布（Atchison & Shen, 1980）。其密度为 $\textstyle \log \mathrm { i t } ( t ) = \log { \frac { t } { 1 - t } }$ ，具有位置参数 $m$ 和尺度参数 $s$ 。位置参数使我们能够将训练时间步偏向数据 $p _ { 0 }$ （负 $m _ { \cdot }$）或噪声 $p _ { 1 }$ （正 $m$）。如图11所示，尺度参数控制分布的宽度。

$$
\pi _ { \ln } ( t ; m , s ) = \frac { 1 } { s \sqrt { 2 \pi } } \frac { 1 } { t ( 1 - t ) } \exp \Bigl ( - \frac { ( \mathrm { l o g i t } ( t ) - m ) ^ { 2 } } { 2 s ^ { 2 } } \Bigr ) ,
$$

在实践中，我们从正态分布中抽样随机变量 $u$，$u \sim \mathcal{N}(u; m, s)$，并通过标准逻辑函数进行映射。具有重尾的模态抽样 逻辑正态密度在端点 0 和 1 处始终消失。为了研究这是否对性能产生不利影响，我们还使用在 $[0, 1]$ 上具有严格正的密度的时间步抽样分布。对于尺度参数 $s$，我们定义

$$
f _ { \mathrm { m o d e } } ( u ; s ) = 1 - u - s \cdot \Bigl ( \cos ^ { 2 } \bigl ( \frac { \pi } { 2 } u \bigr ) - 1 + u \Bigr ) .
$$

对于 $\begin{array} { r } { - 1 \le s \le \frac { 2 } { \pi - 2 } } \end{array}$ ，该函数是单调的，我们可以使用它来从隐含密度中采样 $\pi _ { \mathrm { m o d e } } ( t ; s ) =$ $\textstyle { \left| { \frac { d } { d t } } f _ { \mathrm { m o d e } } ^ { - 1 } ( t ) \right| }$ 控制在采样过程中中点（正的 $s \mathrm { \lrcorner }$）或端点（负的 $s { \dot { } }$）的偏好程度。这个公式还包括一个均匀权重 $\pi _ { \mathrm { m o d e } } ( t ; s = 0 ) = \mathcal { U } ( t )$ 适用于 $s = 0$，在之前关于整流流（Rectified Flows）的工作中得到了广泛应用（Liu et al., 2022; Ma et al., 2024）。最后，我们还考虑了在 RF 设置中的余弦调度（Nichol & Dhariwal, 2021），尤其是我们正在寻找一个映射 $f : u \mapsto f ( u ) = t$ , $u \in [ 0 , 1 ]$，使得对数信噪比与余弦调度相匹配：$\begin{array} { r } { 2 \log { \frac { \cos ( { \frac { \pi } { 2 } } u ) } { \sin ( { \frac { \pi } { 2 } } u ) } } = 2 \log { \frac { 1 - f ( u ) } { f ( u ) } } } \end{array}$ 通过求解 $f$，我们得到 $u \sim \mathcal { U } ( u )$，从中得到密度。

$$
t = f ( u ) = 1 - \frac { 1 } { \tan ( \frac { \pi } { 2 } u ) + 1 } ,
$$

$$
\pi _ { \mathrm { C o s M a p } } ( t ) = \left| { \frac { d } { d t } } f ^ { - 1 } ( t ) \right| = { \frac { 2 } { \pi - 2 \pi t + 2 \pi t ^ { 2 } } } .
$$

# 4. 文本生成图像架构

对于文本条件图像采样，我们的模型必须考虑文本和图像这两种模态。我们使用预训练模型来推导合适的表示，然后描述我们的扩散主干网络的架构。这一概述如图2所示。我们的总体设置遵循LDM（Rombach et al., 2022）在预训练自编码器的潜在空间中训练文本到图像模型。与将图像编码为潜在表示类似，我们也遵循之前的方法（Saharia et al., 2022b；Balaji et al., 2022），使用预训练的冻结文本模型对文本条件 $c$ 进行编码。详细信息可在附录B.2中找到。 多模态扩散主干 我们的架构基于DiT（Peebles & Xie, 2023）架构。DiT 只考虑类别条件的图像生成，并使用调制机制使网络同时对扩散过程的时间步和类别标签进行条件化。类似地，我们使用时间步 $t$ 和 $c_{ \mathrm{v e c} }$ 的嵌入作为调制机制的输入。然而，由于池化文本表示仅保留了关于文本输入的粗粒度信息（Podell et al., 2023），网络还需要来自序列表示 $c_{ \mathrm{c t x t} }$ 的信息。我们构建一个由文本和图像输入的嵌入组成的序列。具体而言，我们添加位置编码并将潜在像素表示 $\boldsymbol{x} \in \mathbb{R}^{h \times w \times c}$ 的 $2 \times 2$ 裂块展平为长度为 $\textstyle{ \frac{1}{2} } \cdot h \cdot { \frac{1}{2} } \cdot w$ 的裂块编码序列。在将此裂块编码与文本编码 $c_{ \mathrm{c t x t} }$ 嵌入到共同维度后，我们将这两个序列进行连接。然后，我们遵循DiT，并应用一系列调制注意力和多层感知机（MLP）。

![](images/2.jpg)  
Figure 2. Our model architecture. Concatenation is indicated by $\odot$ and element-wise multiplication by $^ *$ The RMS-Norm for $Q$ and $K$ can be added to stabilize training runs. Best viewed zoomed in.

由于文本和图像的嵌入在概念上差异较大，我们为这两种模态使用了两个独立的权重集。如图 2b 所示，这相当于为每种模态拥有两个独立的变压器，但将两种模态的序列合并进行注意力操作，从而使得这两个表示可以在各自的空间中工作，同时考虑到对方。在我们的规模实验中，我们通过设置隐藏层大小为 $6 4 \cdot d$（在 MLP 块中扩展为 $4 \cdot 6 4 \cdot d$ 个通道）来将模型的大小参数化为模型的深度 $d$，即注意力块的数量，并且将注意力头的数量设置为 $d$。此外，不同方法的损失不可比，也未必与输出样本的质量相关；因此我们需要评估指标，以便能够比较不同的方法。我们在 ImageNet（Russakovsky 等，2014）和 CC12M（Changpinyo 等，2021）上训练模型，并在训练过程中使用验证损失、CLIP 分数（Radford 等，2021；Hessel 等，2021）和 FID（Heusel 等，2017）对模型的训练和 EMA 权重进行评估，评估是在不同采样器设置下进行的（不同的引导尺度和采样步骤）。我们根据 (Sauer 等，2021) 提出的方案在 CLIP 特征上计算 FID。所有指标均在 COCO-2014 验证集上进行评估（Lin 等，2014）。关于训练和采样超参数的完整细节请参见附录 B.3。

# 5.1.1. 结果

# 5. 实验

# 5.1. 改进校正流

我们旨在理解在方程1中，对于无模拟训练归一化流的不同方法，哪种是最有效的。为了使不同方法之间的比较可行，我们控制了优化算法、模型架构、数据集和采样器。我们对61种不同的模型在两个数据集上进行了训练。我们包括第3节中以下变体：同时使用线性（eps/linear，v/linear）和余弦（$\mathtt { e p s / c o s }$ $\scriptstyle { \mathtt { V } } / \subset \bigcirc { S } $）调度的$\epsilon \mathrm { - }$和$\mathbf { v }$预测损失。使用$\pi _ { \mathrm { m o d e } } ( t ; s )$的RF损失（rf/mode(s)），$s$的7个值均匀选择在$- 1$到1.75之间，另外$s = 1.0$和$s = 0$对应于均匀时间步采样（$\tt r f / m o d e$）。

Table 1. Global ranking of variants. For this ranking, we apply non-dominated sorting averaged over EMA and non-EMA weights, two datasets and different sampling settings.   

<table><tr><td rowspan="2"></td><td colspan="3">rank averaged over</td></tr><tr><td>all</td><td>5 steps</td><td>50 steps</td></tr><tr><td>variant rf/lognorm(0.00, 1.00)</td><td>1.54</td><td>1.25</td><td>1.50</td></tr><tr><td>rf/lognorm(1.00, 0.60)</td><td>2.08</td><td>3.50</td><td>2.00</td></tr><tr><td>rf/lognorm(0.50, 0.60)</td><td>2.71</td><td>8.50</td><td>1.00</td></tr><tr><td>rf/mode(1.29)</td><td>2.75</td><td>3.25</td><td>3.00</td></tr><tr><td>rf/lognorm(0.50, 1.00)</td><td>2.83</td><td>1.50</td><td>2.50</td></tr><tr><td>eps/linear</td><td>2.88</td><td>4.25</td><td>2.75</td></tr><tr><td>rf/mode(1.75)</td><td>3.33</td><td>2.75</td><td>2.75</td></tr><tr><td>rf/cosmap</td><td>4.13</td><td>3.75</td><td>4.00</td></tr><tr><td>edm(0.00, 0.60)</td><td>5.63</td><td>13.25</td><td>3.25</td></tr><tr><td>rf</td><td>5.67</td><td>6.50</td><td>5.75</td></tr><tr><td>v/linear</td><td>6.83</td><td></td><td>7.75</td></tr><tr><td></td><td></td><td>5.75</td><td></td></tr><tr><td>edm(0.60, 1.20)</td><td>9.00</td><td>13.00</td><td>9.00</td></tr><tr><td>v/cos</td><td>9.17</td><td>12.25</td><td>8.75</td></tr><tr><td>edm/cos</td><td>11.04</td><td>14.25</td><td>11.25</td></tr><tr><td>edm/rf</td><td>13.04</td><td>15.25</td><td>13.25</td></tr><tr><td>edm(-1.20, 1.20)</td><td>15.58</td><td>20.25</td><td>15.00</td></tr></table>

Table 2. Metrics for different variants. FID and CLIP scores of different variants with 25 sampling steps. We highlight the best, second best, and third best entries.   

<table><tr><td rowspan="2">variant</td><td colspan="2">ImageNet</td><td colspan="2">CC12M</td></tr><tr><td>CLIP</td><td>FID</td><td>CLIP</td><td>FID</td></tr><tr><td>rf</td><td>0.247</td><td>49.70</td><td>0.217</td><td>94.90</td></tr><tr><td>edm(-1.20, 1.20)</td><td>0.236</td><td>63.12</td><td>0.200</td><td>116.60</td></tr><tr><td>eps/linear</td><td>0.245</td><td>48.42</td><td>0.222</td><td>90.34</td></tr><tr><td>v/cos</td><td>0.244</td><td>50.74</td><td>0.209</td><td>97.87</td></tr><tr><td>v/linear</td><td>0.246</td><td>51.68</td><td>0.217</td><td>100.76</td></tr><tr><td>rf/lognorm(0.50, 0.60)</td><td>0.256</td><td>80.41</td><td>0.233</td><td>120.84</td></tr><tr><td>rf/mode(1.75)</td><td>0.253</td><td>44.39</td><td>0.218</td><td>94.06</td></tr><tr><td>rf/lognorm(1.00, 0.60)</td><td>0.254</td><td>114.26</td><td>0.234</td><td>147.69</td></tr><tr><td>rf/lognorm(-0.50, 1.00)</td><td>0.248</td><td>45.64</td><td>0.219</td><td>89.70</td></tr><tr><td>rf/lognorm(0.00, 1.00)</td><td>0.250</td><td>45.78</td><td>0.224</td><td>89.91</td></tr></table>

RF损失与$\pi _ { \mathrm { l n } } ( t ; m , s )$（rf/lognorm $( \mathfrak { m } , \mathrm { ~ \textbf ~ { ~ s ~ } ~ } )$），在网格中以30个值选择 $( m , s )$，其中 $m$ 均匀分布在 $- 1$ 到 $1$ 之间，$s$ 均匀分布在 $0.2$ 到 $2.2$ 之间。RF损失与$\pi _ { \mathrm { C o s M a p } } ( t )$（rf/cosmap）。• EDM（edm $( P _ { m } , P _ { s } )$）使用15个值，$P _ { m }$ 均匀选取在 $- 1 . 2$ 和 $1.2$ 之间，$P _ { s }$ 均匀选取在 $0.6$ 和 $1.8$ 之间。注意，$P _ { m } , P _ { s } = ( - 1 . 2 , 1 . 2 )$ 对应于(Karras et al., 2022)中的参数。EDM的调度与 $: \pm \ : ( \mathrm { e d m } / \mathrm { r f } )$ 的对数信噪比加权相匹配，同时与v/cos的对数信噪比加权相匹配 (edm/ cos)。对于每次实验，我们选择在使用EMA权重评估时具有最小验证损失的步骤，然后收集CLIP分数和FID，这些是在6种不同采样设置下获得的，既包括EMA权重也不包括EMA权重。在所有24种采样设置、EMA权重和数据集选择的组合中，我们使用非支配排序算法对不同的形式进行排名。为此，我们反复计算根据CLIP和FID分数得到的帕累托最优变体，给这些变体分配当前迭代索引，移除这些变体，然后继续处理剩余变体，直到所有变体都被排名。最后，我们对这24种不同控制设置的排名取平均。我们在表1中展示了结果，其中仅显示经过不同超参数评估的表现最好的两个变体。我们还显示了对采样设置限制在5步和50步的平均排名。我们观察到，rf/lognorm(0.00, 1.00)始终取得良好的排名。它在均匀时间步采样（r f）的整流流形式中表现优于，从而确认了我们关于中间时间步更重要的假设。在所有变体中，只有具有修改时间步采样的整流流形式的表现优于之前使用的LDM-Linear (Rombach et al., 2022) 形式（eps/linear）。我们还观察到，某些变体在某些设置下表现良好，但在其他设置下表现较差，例如rf/lognorm (0 .50, 0.60 )在50个采样步骤中是最佳变体，但在5个采样步骤中的表现大幅下降（平均排名为8.5）。我们在表2中观察到类似的行为。第一组显示了代表性变体及其在两个数据集上的度量，均为25个采样步骤。下一组显示了实现最佳CLIP和FID分数的变体。除了rf /mode (1. 75 )，这些变体通常在一个度量上表现很好，但在另一个度量上相对较差。相比之下，我们再次观察到rf/lognorm (0 . 00, 1.00)在各个度量和数据集上表现良好，在四次中有两次获得第三好的分数，并且一次获得第二好的表现。最后，我们在图3中展示了不同形式的定性行为，在该图中我们用不同颜色表示不同的形式组（edm、rf、eps和v）。整流流形式通常表现良好，并且与其他形式相比，当减少采样步骤数量时，其性能下降较少。

# 5.2. 改进模态特定表示

在前一节中，我们找到了一个公式，使得校正流模型不仅可以与诸如 LDM-Linear（Rombach 等，2022）或 EDM（Karras 等，2022）等成熟的扩散公式竞争，甚至能够超越它们。现在我们转向将该公式应用于高分辨率文本到图像的合成。因此，我们算法的最终性能不仅依赖于训练公式，还依赖于通过神经网络的参数化以及我们使用的图像和文本表示的质量。在接下来的部分中，我们将描述如何改善所有这些组件，然后在第 5.3 节中扩展我们的最终方法。

![](images/3.jpg)  
Figure 3. Rectified flows are sample efficient. Rectified Flows perform better then other formulations when sampling fewer steps. For 25 and more steps, only rf/1ognorm (0.00, 1.00) remains competitive to eps/linear.

Table 3. Improved Autoencoders. Reconstruction performance metrics for different channel configurations. The downsampling factor for all models is $f = 8$ .   

<table><tr><td>Metric</td><td>4 chn</td><td>8 chn</td><td>16 chn</td></tr><tr><td>FID (↓)</td><td>2.41</td><td>1.56</td><td>1.06</td></tr><tr><td>Perceptual Similarity (↓)</td><td>0.85</td><td>0.68</td><td>0.45</td></tr><tr><td>SSIM(↑)</td><td>0.75</td><td>0.79</td><td>0.86</td></tr><tr><td>PSNR(↑)</td><td>25.12</td><td>26.40</td><td>28.62</td></tr></table>

# 5.2.1. 改进型自编码器

潜在扩散模型通过在预训练自编码器的潜在空间中操作，实现了高效性（Rombach等，2022），该自编码器将输入的RGB $X \in \mathbb { R } ^ { H \times W \times 3 }$映射到更低维的空间$x = E ( X ) \in \mathbb { R } ^ { h \times w \times d }$。这个自编码器的重建质量为潜在扩散训练后可达到的图像质量提供了上限。与Dai等（2023）类似，我们发现增加潜在通道数$d$显著提升了重建性能，见表3。直观上，预测更高$d$的潜在变量是一项更困难的任务，因此具有更大容量的模型对于更大的$d$应能够获得更好的性能，最终实现更高的图像质量。我们在图10中确认了这一假设，我们看到$d = 16$的自编码器在样本FID方面表现出更好的可扩展性。因此，在本文剩余部分中，我们选择$d = 16$。

# 5.2.2. 改进的标注

Betker 等（2023）证明，合成生成的标题可以显著改善大规模训练的文本到图像模型。这是因为大规模图像数据集中人类生成的标题往往较为简单，过于关注图像主题，通常忽略描述背景或场景构成的细节，或者在适用的情况下省略显示的文本（Betker 等，2023）。我们遵循他们的方法，使用一款现成的、最先进的视觉-语言模型 $C o g V L M$（Wang 等，2023）为我们的大规模图像数据集创建合成注释。由于合成标题可能导致文本到图像模型忘记 VLM 知识库中不存在的某些概念，我们使用 $50 \%$ 原创标题和 $50 \%$ 合成标题的比例。

Table 4. Improved Captions. Using a 50/50 mixing ratio of synthetic (via $\mathrm { C o g V L M }$ (Wang et al., 2023)) and original captions improves text-to-image performance. Assessed via the GenEval (Ghosh et al., 2023) benchmark.   

<table><tr><td rowspan="2"></td><td>Original Captions</td><td>50/50 Mix</td></tr><tr><td>success rate [%]</td><td>success rate [%]</td></tr><tr><td>Color Attribution</td><td>11.75</td><td>24.75</td></tr><tr><td>Colors</td><td>71.54</td><td>68.09</td></tr><tr><td>Position</td><td>6.50</td><td>18.00</td></tr><tr><td>Counting</td><td>33.44</td><td>41.56</td></tr><tr><td>Single Object</td><td>95.00</td><td>93.75</td></tr><tr><td>Two Objects</td><td>41.41</td><td>52.53</td></tr><tr><td>Overall score</td><td>43.27</td><td>49.78</td></tr></table>

为了评估训练对这个字幕混合的影响，我们训练了两个 $d = 1 5 M M { - } D i T$ 模型，共计 250k 步，其中一个模型只使用原始字幕，另一个模型则使用 $5 0 / 5 0 \mathrm { m i x }$ 。我们使用 GenEval 基准（Ghosh 等，2023）对训练好的模型进行评估，如表 4 所示。结果表明，添加合成字幕进行训练的模型明显优于仅使用原始字幕的模型。因此，我们将 50/50 合成/原始字幕混合用于本研究的其余部分。

# 5.2.3. 改进的文本到图像主干网络

在本节中，我们比较了现有的基于变压器的扩散主干与我们新提出的多模态变压器基础扩散主干MM-DiT的性能，MM-DiT在第4节中介绍。MM-DiT 专门设计用于处理不同领域，此处指文本和图像词元，使用（两个）不同的可训练模型权重集。更具体地说，我们遵循第5.1节中的实验设置，并比较DiT、CrossDiT（DiT，但对文本词元进行交叉关注而不是按序列进行拼接（Chen等，2023））和我们的MM-DiT在CC12M上的文本到图像性能。对于MM-DiT，我们比较了两个权重集和三个权重集的模型，后者分别处理CLIP（Radford等，2021）和T5（Raffel等，2019）词元（参见第4节）。值得注意的是，DiT（与第4节中的文本和图像词元拼接）可以被解释为MM-DiT的一个特例，即所有模态共享一个权重集。最后，我们将UViT（Hoogeboom等，2023）结构视为广泛使用的UNet和变压器变体之间的混合体。

![](images/4.jpg)  
a space elevator, cinematic scifi art

![](images/5.jpg)  
A cheeseburger with juicy beef patties and melted cheese sits on top of a toilet that looks like a throne and stands in the middle of the royal chamber.

![](images/6.jpg)  
a hole in the floor of my bathroom with small gremlins living in it

![](images/7.jpg)  
a small office made out of car parts

![](images/8.jpg)  
This dreamlike digital art captures a vibrant, kaleidoscopic bird in a lush rainforest.

![](images/9.jpg)  
human life depicted entirely out of fractals

![](images/10.jpg)  
an origami pig on fire in the middle of a dark room with a pentagram on the floor

![](images/11.jpg)  
an old rusted robot wearing pants and a jacket riding skis in a supermarket.

![](images/12.jpg)  
smiling cartoon dog sits at a table, coffee mug on hand, as a room goes up in fames. "This is fine," the dog assures himself.

![](images/13.jpg)  
Awhisica fantasy.

![](images/14.jpg)  
Figure 4. Training dynamics of model architectures. Comparative analysis of DiT, CrossDiT, UViT, and MM-DiT on CC12M, focusing on validation loss, CLIP score, and FID. Our proposed MM-DiT performs favorably across all metrics.

我们分析了图4中这些架构的收敛行为：基本 DiT 表现不如 UViT。跨注意力 DiT 变体 CrossDiT 的性能优于 UViT，尽管 UViT 在最初似乎学习得更快。我们的 MM-DiT 变体显著优于跨注意力和基本变体。我们观察到使用三个参数集相比于两个仅带来了小幅提升（因此增加了参数数量和 VRAM 使用），所以对于本研究的其余部分，我们选择前者。

# 5.3. 大规模训练

在扩大规模之前，我们对数据进行过滤和预编码，以确保安全高效的预训练。然后，之前对扩散形式、架构和数据的所有考虑在最后一节中汇聚，我们将模型规模扩大到80亿参数。

# 5.3.1. 数据预处理

预训练的缓解措施 训练数据显著影响生成模型的能力。因此，数据过滤在限制不良能力方面是有效的（Nichol, 2022）。在销售前，我们对数据进行以下类别的过滤：(i) 性内容：我们使用 NSFW 检测模型过滤显式内容。(ii) 美学：我们移除评分系统预测低分的图像。(iii) 重复：我们使用基于聚类的去重方法，移除训练数据中的感知和语义重复；详见附录 E.2。 预计算图像和文本嵌入 我们的模型使用多个预训练、冻结网络的输出作为输入（自编码器潜变量和文本编码器表示）。由于这些输出在训练期间是恒定的，我们为整个数据集预先计算一次。我们在附录 E.1 中详细讨论了我们的方法。

![](images/15.jpg)  
Figure 5. Effects of QK-normalization. Normalizing the Q- and K-embeddings before calculating the attention matrix prevents the attention-logit growth instability (left), which causes the attention entropy to collapse (right) and has been previously reported in the discriminative ViT literature (Dehghani et al., 2023; Wortsman et al., 2023). In contrast with these previous works, we observe this instability in the last transformer blocks of our networks. Maximum attention logits and attention entropies are shown averaged over the last 5 blocks of a 2B $\mathrm { \Delta } \mathrm { d } = 2 4$ model.

# 5.3.2. 在高分辨率下微调

QK归一化 通常，我们在低分辨率图像（大小为 $256^2$ 像素）上预训练所有模型。接下来，我们在更高分辨率和混合纵横比的图像上对模型进行微调（具体细节见下一段）。我们发现，当转向高分辨率时，混合精度训练可能会变得不稳定，损失出现发散。这可以通过切换到全精度训练来解决——但与混合精度训练相比，性能下降约 $1 \sim 2 \times$。文献中提出了一种更高效的替代方案：Dehghani 等人（2023）观察到，大型视觉变换器模型的训练发散，是因为注意力熵失控增长。为避免这种情况，Dehghani 等人（2023）建议在注意力操作之前对 Q 和 K 进行归一化。我们遵循这一方法，并在 MMDiT 架构的两个流中使用带可学习尺度的 RMSNorm（Zhang & Sennrich，2019），见图 2。如图 5 所示，额外的归一化防止了注意力 logit 增长的不稳定性，验证了 Dehghani 等人（2023）和 Wortsman 等人（2023）的研究结果，并在与 AdamW（Loshchilov & Hutter，2017）优化器中的 $\epsilon = 10^{-15}$ 结合时，支持在 bf16-mixed（Chen et al.，2019）精度下进行高效训练。该技术也可以应用于在预训练过程中没有使用 qk 归一化的预训练模型：模型可以快速适应额外的归一化层并更稳定地训练。最后，我们想指出，尽管这种方法可以在一般情况下帮助稳定大型模型的训练，但它并不是通用的配方，可能需要根据具体的训练设置进行调整。 不同纵横比的位置信息编码 在固定 $256 \times 256$ 分辨率上训练后，我们旨在 (i) 增加分辨率，并且 (ii) 支持灵活纵横比的推理。由于我们使用2D位置频率嵌入，因此必须根据分辨率进行调整。在多纵横比设置中，像 (Dosovitskiy 等，2020) 中那样直接插值嵌入将无法正确反映边长。相反，我们使用扩展和插值位置网格的组合，并随后进行频率嵌入。

![](images/16.jpg)  
Figure 6. Timestep shifting at higher resolutions. Top right: Human quality preference rating when applying the shifting based on Equation (23). Bottom row: A $5 1 2 ^ { 2 }$ model trained and sampled with $\sqrt { m / n } = 1 . 0$ (top) and $\sqrt { m / n } = 3 . 0$ (bottom). See Section 5.3.2.

对于目标分辨率为 $S^{2}$ 像素的情况，我们使用分桶采样（NovelAI，2022；Podell等，2023），使得每个批次由同质尺寸的图像组成 $H \times W$，其中 ${\cal H} \cdot {\cal W} \approx S^{2}$。对于最大和最小训练长宽比，这导致最大宽度 $W_{\mathrm{max}}$ 和最大高度 $H_{\mathrm{max}}$ 的值。设 $h_{\operatorname*{max}} = H_{\operatorname*{max}} / 16, w_{\operatorname*{max}} = W_{\operatorname*{max}} / 16$，以及 $s = S / 16$ 为对应的潜在空间中的尺寸（因子8），在分块后（因子2）。基于这些值，我们构建一个垂直位置网格，其值为 $((p \cdot h_{\operatorname*{max}} - s) \begin{array}{r l} { { \big ( \big ( p - \frac{h_{\operatorname*{max}} - s}{2} \big ) \cdot \frac{256}{S} \big ) _{p = 0}^{h_{\operatorname*{max}} - 1}} } \end{array}$，并相应地构建水平位置网格。然后我们从所得到的二维位置网格中进行中心裁剪，再进行嵌入。

分辨率相关的时间步调度的移动 直观上，由于更高的分辨率拥有更多的像素，因此我们需要更多的噪声来破坏它们的信号。假设我们在一个分辨率为 $n = H \cdot W$ 的像素图像上工作。现在，考虑一个“常量”图像，即每个像素的值均为 $c$。前向过程生成 $z_{t} = (1 - t) c \mathbb{1} + t \epsilon$，其中 $\mathbb{1}$ 和 $\epsilon \in \mathbb{R}^{n}$。因此，$z_{t}$ 提供了关于随机变量 $Y = (1 - t) c + t \eta$ 的 $n$ 组观测值，其中 $c$ 和 $\eta$ 在 $\mathbb{R}$ 中，且 $\eta$ 服从标准正态分布。因此，$\mathbb{E}(Y) = (1 - t) c$，而 $\sigma(Y) = t$。我们因此能够通过 $c = -t$ 恢复 $c$，而 $c$ 与其样本估计 $\hat{c} = \frac{1}{1 - t} \sum_{i=1}^{n} z_{t,i}$ 之间的误差具有标准偏差 $\sigma(t, n) = \frac{t}{1 - t} \sqrt{\frac{1}{n}}$（因为 $Y$ 的偏差为 $\frac{t}{\sqrt{n}}$）。由于图像 $z_{0}$ 在其像素中是常量，$\sigma(t, n)$ 表示对 $z_{0}$ 的不确定性程度。例如，我们立刻可以看到，在任何给定时间 $0 < t < 1$，宽度和高度的倍增导致不确定性减半。但是，我们现在可以通过 Ansatz $\sigma(t_{n}, n) = \sigma(t_{m}, m)$ 将分辨率 $n$ 的时间步 $t_{n}$ 映射到分辨率 $m$ 的时间步 $t_{m}$，从而产生相同的不确定性程度。解关于 $t_{m}$ 的方程得到：

![](images/17.jpg)  
Figure 7. Human Preference Evaluation against currrent closed and open SOTA generative image models. Our 8B model compares favorable against current state-of-the-art text-to-image models when evaluated on the parti-prompts (Yu et al., 2022) across the categories visual quality, prompt following and typography generation.

$$
t _ { m } = \frac { \sqrt { \frac { m } { n } } t _ { n } } { 1 + ( \sqrt { \frac { m } { n } } - 1 ) t _ { n } }
$$

我们在图6中可视化了这个偏移函数。注意，恒定图像的假设并不现实。为了在推理过程中找到良好的偏移值 $\alpha := \sqrt{\frac{m}{n}}$，我们将其应用于在 $1024 \times 1024$ 分辨率下训练的模型的采样步骤，并进行了一项人类偏好研究。图6中的结果显示，对于偏移值大于1.5的样本具有较强的偏好，但在较高偏移值之间的差异不明显。因此，在后续实验中，我们在 $1024 \times 1024$ 分辨率下的训练和采样过程中使用偏移值 $\alpha = 3.0$。图6中可以找到经过8k训练步骤后有无这种偏移的样本的定性比较。最后，注意方程23暗示了 $\log{\frac{n}{m}}$ 的对数信噪比偏移，类似于（Hoogeboom等，2023）。

$$
\begin{array} { l } { { \lambda _ { t _ { m } } = 2 \log \displaystyle \frac { 1 - t _ { n } } { \sqrt { \frac { m } { n } } t _ { n } } } } \\ { { = \lambda _ { t _ { n } } - 2 \log \alpha = \lambda _ { t _ { n } } - \log \displaystyle \frac { m } { n } \ : . } } \end{array}
$$

在分辨率 $1 0 2 4 \times 1 0 2 4$ 进行移位训练后，我们使用附录 C 中描述的直接偏好优化（DPO）对模型进行对齐。

# 5.3.3. 结果

在图8中，我们研究了在规模上训练我们的MM-DiT的效果。对于图像，我们进行了一项大规模的扩展研究，并在分辨率为$256^{2}$像素的情况下，使用预编码数据进行不同参数数量的模型训练，共进行了$500k$步，批量大小为4096。我们在$2 \times 2$的补丁上进行训练（Peebles & Xie, 2023），并每50k步报告一次在CoCo数据集（Lin et al., 2014）上的验证损失。特别地，为了减少验证损失信号中的噪声，我们在$t \in (0, 1)$中等距采样损失水平，并分别计算每个水平的验证损失。然后，我们对除了最后一个$\mathit{t} = 1$水平之外的所有水平的损失进行平均。

同样，我们对我们的 MM-DiT 进行了一项初步的可扩展性研究，重点是视频。为此，我们从预训练的图像权重开始，并额外使用 $2 \mathbf { x }$ 时间补丁。我们遵循 Blattmann 等人（2023b）的方法，通过将时间维度折叠到批次轴，将数据输入预训练模型。在每个注意力层中，我们在视觉流中重新排列表示，并在空间注意力操作之后，在最终的前馈层之前对所有时空令牌执行全注意力。我们的视觉模型在包含 $16$ 帧和 $256 ^ { 2 }$ 像素的视频上训练了 $140 \mathrm { k }$ 步，批次大小为 $512$。我们每 $5 \mathrm { k }$ 步报告 Kinetics 数据集（Carreira & Zisserman, 2018）的验证损失。请注意，图 8 中我们报告的视频训练 FLOPs 仅包含视频训练的 FLOPs，不包括图像预训练的 FLOPs。在图像和视频领域中，我们观察到随着模型大小和训练步数的增加，验证损失平滑下降。我们发现验证损失与全面评估指标（CompBench（Huang et al., 2023）、GenEval（Ghosh et al., 2023））和人类偏好高度相关。这些结果支持验证损失作为模型性能的简单而通用的衡量标准。我们的结果显示，无论是图像模型还是视频模型均未出现饱和现象。图 12 说明了训练更大模型更长时间如何影响样本质量。表 5 显示了 GenEval 的完整结果。当应用第 5.3.2 节中提出的方法并提高训练图像分辨率时，我们最大的模型在大多数类别中表现出色，并在整体评分上超越了在提示理解方面的当前最先进的 DALLE 3（Betker et al., 2023）。我们的 $d = 38$ 模型在对人类偏好的评估中超越了当前的专有（Betker et al., 2023；ide, 2024）和开放（Sauer et al., 2023；pla, 2024；Chen et al., 2023；Pernias et al., 2023）生成图像模型。

Table 5. GenEval comparisons. Our largest model (depth $^ { 1 = 3 8 }$ ) outperforms all current open models and DALLE-3 (Betker et al., 2023) on GenEval (Ghosh et al., 2023). We highlight the best, second best, and third best entries. For DPO, see Appendix C.   

<table><tr><td></td><td colspan="2">Objects</td><td colspan="2"></td><td></td><td></td><td>Color</td></tr><tr><td>Model</td><td>Overall</td><td>Single Two</td><td></td><td>Counting</td><td></td><td></td><td>Colors Position Attribution</td></tr><tr><td>minDALL-E</td><td>0.23</td><td>0.73</td><td>0.11</td><td>0.12</td><td>0.37</td><td>0.02</td><td>0.01</td></tr><tr><td>SD v1.5</td><td>0.43</td><td>0.97</td><td>0.38</td><td>0.35</td><td>0.76</td><td>0.04</td><td>0.06</td></tr><tr><td>PixArt-alpha</td><td>0.48</td><td>0.98</td><td>0.50</td><td>0.44</td><td>0.80</td><td>0.08</td><td>0.07</td></tr><tr><td>SD v2.1</td><td>0.50</td><td>0.98</td><td>0.51</td><td>0.44</td><td>0.85</td><td>0.07</td><td>0.17</td></tr><tr><td>DALL-E 2</td><td>0.52</td><td>0.94</td><td>0.66</td><td>0.49</td><td>0.77</td><td>0.10</td><td>0.19</td></tr><tr><td>SDXL</td><td>0.55</td><td>0.98</td><td>0.74</td><td>0.39</td><td>0.85</td><td>0.15</td><td>0.23</td></tr><tr><td>SDXL Turbo</td><td>0.55</td><td>1.00</td><td>0.72</td><td>0.49</td><td>0.80</td><td>0.10</td><td>0.18</td></tr><tr><td>IF-XL</td><td>0.61</td><td>0.97</td><td>0.74</td><td>0.66</td><td>0.81</td><td>0.13</td><td>0.35</td></tr><tr><td>DALL-E 3</td><td>0.67</td><td>0.96</td><td>0.87</td><td>0.47</td><td>0.83</td><td>0.43</td><td>0.45</td></tr><tr><td>Ours (depth=18), 5122</td><td>0.58</td><td>0.97</td><td>0.72</td><td>0.52</td><td>0.78</td><td>0.16</td><td>0.34</td></tr><tr><td>Ours (depth=24), 5122</td><td>0.62</td><td>0.98</td><td>0.74</td><td>0.63</td><td>0.67</td><td>0.34</td><td>0.36</td></tr><tr><td>Ours (depth=30), 5122</td><td>0.64</td><td>0.96</td><td>0.80</td><td>0.65</td><td>0.73</td><td>0.33</td><td>0.37</td></tr><tr><td>Ours (depth=38), 5122</td><td>0.68</td><td>0.98</td><td>0.84</td><td>0.66</td><td>0.74</td><td>0.40</td><td>0.43</td></tr><tr><td>Ours (depth=38), 5122 w/DPO</td><td>0.71</td><td>0.98</td><td>0.89</td><td>0.73</td><td>0.83</td><td>0.34</td><td>0.47</td></tr><tr><td>Ours (depth=38), 10242 w/DPO</td><td>0.74</td><td>0.99</td><td>0.94</td><td>0.72</td><td>0.89</td><td>0.33</td><td>0.60</td></tr></table>

<table><tr><td rowspan="3"></td><td colspan="4">relative CLIP score decrease [%]</td></tr><tr><td>5/50 steps</td><td>10/50 steps</td><td>20/50 steps</td><td>path length</td></tr><tr><td>depth=15</td><td>4.30</td><td>0.86</td><td>0.21</td><td>191.13</td></tr><tr><td>depth=30</td><td>3.59</td><td>0.70</td><td>0.24</td><td>187.96</td></tr><tr><td>depth=38</td><td>2.71</td><td>0.14</td><td>0.08</td><td>185.96</td></tr></table>

Table 6. Impact of model size on sampling efficiency. The table shows the relative performance decrease relative to CLIP scores evaluated using 50 sampling steps at a fixed seed. Larger models can be sampled using fewer steps, which we attribute to increased robustness and better fitting the straight-path objective of rectified flow models, resulting in shorter path lengths. Path length is calculated by summing up $\| v _ { \theta } \cdot d t \|$ over 50 steps.

Parti-prompts 基准（Yu et al., 2022）在视觉美学、提示跟随和排版生成三个类别中进行评估，$c . f$ . 图 7。为了评估人类在这些类别中的偏好，评审员展示了来自两个模型的成对输出，并被要求回答以下问题：提示跟随：哪个图像更能代表上述文本并忠实遵循？视觉美学：给定提示，哪个图像质量更高且在美学上更令人愉悦？排版：哪个图像更准确地显示/呈现了上述描述中的文本？更准确的拼写更受欢迎！忽略其他方面。最后，表 6 突出了一个有趣的结果：不仅较大的模型表现更好，而且它们达到峰值性能所需的步骤也更少。

灵活的文本编码器 虽然使用多个文本编码器的主要动机是提高整体模型性能（Balaji 等，2022），但我们现在展示这种选择在推理过程中还增加了基于 MM-DiT 的修正流的灵活性。如附录 B.3 所述，我们的模型使用三个文本编码器进行训练，每个编码器的 dropout 率为 $4 6 . 3 \%$。因此，在推理时，我们可以使用所有三个文本编码器的任意子集。这为在提升内存效率的同时权衡模型性能提供了手段，这对于需要大量 VRAM 的 4.7B 参数的 T5-XXL（Raffel 等，2019）特别相关。值得注意的是，当仅使用两个基于 CLIP 的文本编码器来处理文本提示，并将 T5 嵌入替换为零时，我们观察到性能下降有限。我们在图 9 中提供了定性可视化。只有在涉及高度详细的场景描述或大量书面文本的复杂提示中，使用所有三个文本编码器时才发现显著的性能提升。这些观察结果也在图 7 的人类偏好评估结果中得到了验证（我们的结果不包括 T5）。移除 T5 对美学质量评分没有影响（胜率 $5 0 \%$），对提示遵循性的影响也很小（胜率 $4 6 \%$），而它对生成书面文本能力的贡献则更为显著（胜率 $3 8 \%$）。

![](images/18.jpg)  
hyperparameters throughout. An exception is depth $^ { 1 = 3 8 }$ , where learning rate adjustments at $3 \times 1 0 ^ { 5 }$ steps were necessary to prevent c validation loss and human preference, column 4. .

![](images/19.jpg)  
Figure 9. Impact of T5. We observe T5 to be important for complex prompts e.g. such involving a high degree of detail or longer spelled text (rows 2 and 3). For most prompts, however, we find that removing T5 at inference time still achieves competitive performance.

# 6. 结论

在本研究中，我们对文本到图像合成的修正流模型进行了尺度分析。我们提出了一种新颖的时步采样方法用于修正流训练，改进了之前的潜在扩散模型的扩散训练公式，并保持了修正流在少步采样条件下的优良特性。我们还展示了基于变换器的MM-DiT架构的优势，该架构考虑了文本到图像任务的多模态特性。最后，我们进行了这一组合的规模研究，其模型规模达到8亿参数和$5 \times 10^{22}$的训练浮点运算。我们证明了验证损失的改善与现有的文本到图像基准以及人类偏好评估之间的相关性。结合我们在生成建模和可扩展多模态架构方面的改善，使我们的性能与最先进的专有模型具有竞争力。规模趋势未显示饱和迹象，这让我们对未来继续改善模型性能充满信心。

# 更广泛的影响

本文展示的研究旨在推动机器学习领域，尤其是图像合成的进展。我们的工作可能会产生许多潜在的社会影响，但我们认为不必在此特别强调。有关扩散模型一般影响的详细讨论，我们建议感兴趣的读者参考（Po et al., 2023）。

# References

Ideogram v1.0 announcement, 2024. URL ht tps : / / ab out.ideogram.ai/1.0.

Playground v2.5 announcement, 2024. URL ht t ps : / /b1 og.playgroundai.com/playground-v2-5/.

Albergo, M. S. and Vanden-Eijnden, E. Building normalizing flows with stochastic interpolants, 2022.

Atchison, J. and Shen, S. M. Logistic-normal distributions: Some properties and uses. Biometrika, 67(2):261272, 1980.

autofaiss. autofaiss, 2023. URL https: / /github.c om/criteo/autofaiss.

Balaji, Y., Nah, S., Huang, X., Vahdat, A., Song, J., Zhang, Q., Kreis, K., Aittala, M., Aila, T., Laine, S., Catanzaro, B., Karras, T., and Liu, M.-Y. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers, 2022.

Betker, J., Goh, G., Jing, L., Brooks, T., Wang, J., Li, L., Ouyang, L., Zhuang, J., Lee, J., Guo, Y., et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3), 2023.

Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M., Lorenz, D., Levi, Y., English, Z., Voleti, V., Letts, A., et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023a.

Blattmann, A., Rombach, R., Ling, H., Dockhorn, T., Kim, S. W., Fidler, S., and Kreis, K. Align your latents: Highresolution video synthesis with latent diffusion models, 2023b.

Brooks, T., Holynski, A., and Efros, A. A. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1839218402, 2023.

Carlini, N., Hayes, J., Nasr, M., Jagielski, M., Sehwag, V., Tramer, F., Balle, B., Ippolito, D., and Wallace, E. Extracting training data from diffusion models. In 32nd

USENIX Security Symposium (USENIX Security 23), pp.   
52535270, 2023.

Carreira, J. and Zisserman, A. Quo vadis, action recognition? a new model and the kinetics dataset, 2018.

Changpinyo, S., Sharma, P. K., Ding, N., and Soricut, R. Conceptual $1 2 \mathrm { m }$ : Pushing web-scale image-text pretraining to recognize long-tail visual concepts. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 35573567, 2021. URL https://api.semanticscholar.org/Corp usID:231951742.

Chen, D., Chou, C., Xu, Y., and Hseu, J. Bfloat16: The secret to high performance on cloud tpus, 2019. URL https://cloud.google.com/blog/produc ts/ai-machine-learning/bfloat16-the-s ecret-to-high-performance-on-cloud-t pus?hl $=$ en.

Chen, J., Yu, J., Ge, C., Yao, L., Xie, E., Wu, Y., Wang, Z., Kwok, J., Luo, P., Lu, H., and Li, Z. Pixart-a: Fast training of diffusion transformer for photorealistic textto-image synthesis, 2023.

Chen, T. Q., Rubanova, Y., Bettencourt, J., and Duvenaud, D. K. Neural ordinary differential equations. In Neural Information Processing Systems, 2018. URL https : //api.semanticscholar.org/CorpusID:49 310446.

Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., Schuhmann, C., Schmidt, L., and Jitsev, J. Reproducible scaling laws for contrastive language-image learning. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023. doi: 10.1109/cvpr52729.2023.00276. URL http://dx.doi.org/10.1109/CVPR52729.2 023.00276.

Dai, X., Hou, J., Ma, C.-Y., Tsai, S., Wang, J., Wang, R., Zhang, P., Vandenhende, S., Wang, X., Dubey, A., Yu, M., Kadian, A., Radenovic, F., Mahajan, D., Li, K., Zhao, Y. Petrovic, V., Singh, M. K., Motwani, S., Wen, Y., Song, Y., Sumbaly, R., Ramanathan, V., He, Z., Vajda, P., and Parikh, D. Emu: Enhancing image generation models using photogenic needles in a haystack, 2023.

Dao, Q., Phung, H., Nguyen, B., and Tran, A. Flow matching in latent space, 2023.

Dehghani, M., Djolonga, J., Mustafa, B., Padlewski, P., Heek, J., Gilmer, J., Steiner, A., Caron, M., Geirhos, R., Alabdulmohsin, I., Jenatton, R., Beyer, L., Tschannen, M., Arnab, A., Wang, X., Riquelme, C., Minderer, M., Puigcerver, J., Evci, U., Kumar, M., van Steenkiste, S.,

Elsayed, G. F., Mahendran, A., Yu, F., Oliver, A., Huot, F., Bastings, J., Collier, M. P., Gritsenko, A., Birodkar, V., Vasconcelos, C., Tay, Y., Mensink, T., Kolesnikov, A., Paveti, F., Tran, D., Kipf, T., Lui, M., Zhai, X., Keysers, D., Harmsen, J., and Houlsby, N. Scaling vision transformers to 22 billion parameters, 2023.

Dhariwal, P. and Nichol, A. Diffusion models beat gans on image synthesis, 2021.

Dockhorn, T., Vahdat, A., and Kreis, K. Score-based generative modeling with critically-damped langevin diffusion. arXiv preprint arXiv:2112.07068, 2021.

Dockhorn, T., Vahdat, A., and Kreis, K. Genie: Higherorder denoising diffusion solvers, 2022.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2020.

Esser, P., Chiu, J., Atighehchian, P., Granskog, J., and Germanidis, A. Structure and content-guided video synthesis with diffusion models, 2023.

Euler, L. Institutionum calculi integralis. Number Bd. 1 in Institutionum calculi integralis. imp. Acad. imp. Saènt. 1768. URL https://books.google.de/book s?id $\underline { { \underline { { \mathbf { \Pi } } } } } =$ Vg8OAAAAQAAJ.

Fischer, J. S., Gui, M., Ma, P., Stracke, N., Baumann, S. A., and Ommer, B. Boosting latent diffusion with flow matching. arXiv preprint arXiv:2312.07360, 2023.

Ghosh, D., Hajishirzi, H., and Schmidt, L. Geneval: An object-focused framework for evaluating text-to-image alignment. arXiv preprint arXiv:2310.11513, 2023.

Gupta, A., Yu, L., Sohn, K., Gu, X., Hahn, M., Fei-Fei, L., Essa, I., Jiang, L., and Lezama, J. Photorealistic video generation with diffusion models, 2023.

Hessel, J., Holtzman, A., Forbes, M., Le Bras, R., and Choi, Y. Clipscore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 2021. doi: 10.18653/v1/2021.emnlp-main.595. URL ht tp : / / dx .doi.org/10.18653/v1/2021.emnlp-main. 595.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S. Gans trained by a two time-scale update rule converge to a local nash equilibrium, 2017.

Ho, J. and Salimans, T. Classifier-free diffusion guidance, 2022.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models, 2020.

Ho, J., Chan, W., Saharia, C., Whang, J., Gao, R., Gritsenko, A., Kingma, D. P., Poole, B., Norouzi, M., Fleet, D. J., and Salimans, T. Imagen video: High definition video generation with diffusion models, 2022.

Hoogeboom, E., Heek, J., and Salimans, T. Simple diffusion: End-to-end diffusion for high resolution images, 2023.

Huang, K., Sun, K., Xie, E., Li, Z., and Liu, X. T2icompbench: A comprehensive benchmark for open-world compositional text-to-image generation. arXiv preprint arXiv:2307.06350, 2023.

Hyvärinen, A. Estimation of non-normalized statistical models by score matching. J. Mach. Learn. Res., 6:695 709,2005.URL https://api.semanticschola r.org/CorpusID:1152227.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models, 2020.

Karras, T., Aittala, M., Aila, T., and Laine, S. Elucidating the design space of diffusion-based generative models. ArXiv, abs/2206.00364, 2022. URL https : / / api . se manticscholar.org/CorpusID:249240415.

Karras, T., Aittala, M., Lehtinen, J., Hellsten, J., Aila, T., and Laine, S. Analyzing and improving the training dynamics of diffusion models. arXiv preprint arXiv:2312.02696, 2023.

Kingma, D. P. and Gao, R. Understanding diffusion objectives as the elbo with simple data augmentation. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

Lee, K., Ippolito, D., Nystrom, A., Zhang, C., Eck, D., Callison-Burch, C., and Carlini, N. Deduplicating training data makes language models better. arXiv preprint arXiv:2107.06499, 2021.

Lee, S., Kim, B., and Ye, J. C. Minimizing trajectory curvature of ode-based generative models, 2023.

Lin, S., Liu, B., Li, J., and Yang, X. Common diffusion noise schedules and sample steps are flawed. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 54045411, 2024.

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., and Zitnick, C. L. Microsoft COCO: Common Objects in Context, pp. 740755. Springer International Publishing, 2014. ISBN 9783319106021. doi:

10.1007/978-3-319-10602-1_48. URL http : / /dx . d oi.0rg/10.1007/978-3-319-10602-1_48.

Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net /forum?id $\underline { { \underline { { \mathbf { \Pi } } } } } =$ PqvMRDCJT9t.

Liu, X., Gong, C., and Liu, Q. Flow straight and fast: Learning to generate and transfer data with rectified flow, 2022.

Liu, X., Zhang, X., Ma, J., Peng, J., and Liu, Q. Instaflow: One step is enough for high-quality diffusion-based textto-image generation, 2023.

Loshchilov, I. and Hutter, F. Fixing weight decay regularization in adam. ArXiv, abs/1711.05101, 2017. URL https://api.semanticscholar.org/Corp usID:3312944.

Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., and Zhu, J. Dpmsolver $^ { + + }$ : Fast solver for guided sampling of diffusion probabilistic models, 2023.

Ma, N., Goldstein, M., Albergo, M. S., Boffi, N. M., VandenEijnden, E., and Xie, S. Sit: Exploring flow and diffusionbased generative models with scalable interpolant transformers, 2024.

Nichol, A. Dall-e 2 pre-training mitigations. ht t ps : //openai.com/research/dall-e-2-pre-t raining-mitigations,2022.

Nichol, A. and Dhariwal, P. Improved denoising diffusion probabilistic models, 2021.

NovelAI. Novelai improvements on stable diffusion, 2022. URL https://blog.novelai.net/novelai -improvements-on-stable-diffusion-e10 d38db82ac.

Peebles, W. and Xie, S. Scalable diffusion models with transformers. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2023. doi: 10.1109/iccv51070.2023.00387. URL http : / / dx. d oi.0rg/10.1109/ICCV51070.2023.00387.

Pernias, P., Rampas, D., Richter, M. L., Pal, C. J., and Aubreville, M. Wuerstchen: An efficient architecture for large-scale text-to-image diffusion models, 2023.

Pizzi, E., Roy, S. D., Ravindra, S. N., Goyal, P., and Douze, M. A self-supervised descriptor for image copy detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1453214542, 2022.

Po, R., Yifan, W., Golyanik, V., Aberman, K., Barron, J. T., Bermano, A. H., Chan, E. R., Dekel, T., Holynski, A. Kanazawa, A., et al. State of the art on diffusion models for visual computing. arXiv preprint arXiv:2310.07204, 2023.

Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., and Rombach, R. Sdxl: Improving latent diffusion models for high-resolution image synthesis, 2023.

Pooladian, A.-A., Ben-Hamu, H., Domingo-Enrich, C., Amos, B., Lipman, Y., and Chen, R. T. Q. Multisample flow matching: Straightening flows with minibatch couplings, 2023.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision, 2021.

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., and Finn, C. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. arXiv:2305.18290, 2023.

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer, 2019.

Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., and Chen, M. Hierarchical text-conditional image generation with clip latents, 2022.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2022. doi: 10.1109/cvpr52688.2022.01042. URL http://dx.doi.org/10.1109/CVPR52688.2 022.01042.

Ronneberger, O., Fischer, P., and Brox, T. U-Net: Convolutional Networks for Biomedical Image Segmentation, pp. 234241. Springer International Publishing, 2015. ISBN 9783319245744. doi: 10.1007/978-3-319-24574-4_28. URLhttp://dx.doi.org/10.1007/978-3-3 19-24574-4_28.

Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M. S., Berg, A. C., and Fei-Fei, L. Imagenet large scale visual recognition challenge. International Journal of Computer Vision, 115:211 - 252, 2014. URL https : //api.semanticscholar.org/CorpusID:29 30547.

Saharia, C., Chan, W., Chang, H., Lee, C., Ho, J., Salimans, T. Fleet, D., and Norouzi, M. Palette: Image-to-image diffusion models. In ACM SIGGRAPH 2022 Conference Proceedings, pp. 110, 2022a.

Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E., Ghasemipour, S. K. S., Ayan, B. K., Mahdavi, S. S., Lopes, R. G., Salimans, T., Ho, J., Fleet, D. J., and Norouzi, M. Photorealistic text-to-image diffusion models with deep language understanding, 2022b.

Saharia, C., Ho, J., Chan, W., Salimans, T., Fleet, D. J., and Norouzi, M. Image super-resolution via iterative refinement. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4):47134726, 2022c.

Sauer, A., Chitta, K., Müller, J., and Geiger, A. Projected gans converge faster. Advances in Neural Information Processing Systems, 2021.

Sauer, A., Lorenz, D., Blattmann, A., and Rombach, R. Adversarial diffusion distillation. arXiv preprint arXiv:2311.17042, 2023.

Sheynin, S., Polyak, A., Singer, U., Kirstain, Y., Zohar, A., Ashual, O., Parikh, D., and Taigman, Y. Emu edit: Precise image editing via recognition and generation tasks. arXiv preprint arXiv:2311.10089, 2023.

Singer, U., Polyak, A., Hayes, T., Yin, X., An, J., Zhang, S., Hu, Q., Yang, H., Ashual, O., Gafni, O., Parikh, D., Gupta, S., and Taigman, Y. Make-a-video: Text-to-video generation without text-video data, 2022.

Sohl-Dickstein, J. N., Weiss, E. A., Maheswaranathan, N., and Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics. ArXiv, abs/1503.03585, 2015. URL https://api.semanticscholar. org/CorpusID:14888175.

Somepalli, G., Singla, V., Goldblum, M., Geiping, J., and Goldstein, T. Diffusion art or digital forgery? investigating data replication in diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 60486058, 2023a.

Somepalli, G., Singla, V., Goldblum, M., Geiping, J., and Goldstein, T. Understanding and mitigating copying in diffusion models. arXiv preprint arXiv:2305.20086, 2023b.

Song, J., Meng, C., and Ermon, S. Denoising diffusion implicit models, 2022.

Song, Y. and Ermon, S. Generative modeling by estimating gradients of the data distribution, 2020.

Song, Y., Sohl-Dickstein, J. N., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. ArXiv, abs/2011.13456, 2020. URL https : / /api . semant icscholar.org/CorpusID:227209335.

Tong, A., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Fatras, K., Wolf, G., and Bengio, Y. Improving and generalizing flow-based generative models with minibatch optimal transport, 2023.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is all you need, 2017.

Villani, C. Optimal transport: Old and new. 2008. URL https://api.semanticscholar.org/Corp usID:118347220.

Vincent, P. A connection between score matching and denoising autoencoders. Neural Computation, 23:1661 1674,2011. URL https://api.semanticscho lar.org/CorpusID:5560643.

Wallace, B., Dang, M., Rafailov, R., Zhou, L., Lou, A., Purushwalkam, S., Ermon, S., Xiong, C., Joty, S., and Naik, N. Diffusion Model Alignment Using Direct Preference Optimization. arXiv:2311.12908, 2023.

Wang, W., Lv, Q., Yu, W., Hong, W., Qi, J., Wang, Y., Ji, J., Yang, Z., Zhao, L., Song, X., et al. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079, 2023.

Wortsman, M., Liu, P. J., Xiao, L., Everett, K., Alemi, A., Adlam, B., Co-Reyes, J. D., Gur, I., Kumar, A., Novak, R., Pennington, J., Sohl-dickstein, J., Xu, K., Lee, J., Gilmer, J., and Kornblith, S. Small-scale proxies for large-scale transformer training instabilities, 2023.

Yu, J., Xu, Y., Koh, J. Y., Luong, T., Baid, G., Wang, Z., Vasudevan, V., Ku, A., Yang, Y., Ayan, B. K., et al. Scaling Autoregressive Models for Content-Rich Text-to-Image Generation. arXiv:2206.10789, 2022.

Zhai, X., Kolesnikov, A., Houlsby, N., and Beyer, L. Scaling vision transformers. In CVPR, pp. 1210412113, 2022.

Zhang, B. and Sennrich, R. Root mean square layer normalization, 2019.

# Supplementary

# A. Background

Mo Dick l So l 00Hl e ata y ha  hTvn ivehaal&Nic  Shl a  ; a  H Ba 2015and scorematching Hyvärinen, 2005;Vincent, 201 Song &Ermon, 2020), variusformulations  forwarand v  So00Dc 0 at Ho  00H&S ; a  H   ; themial workKin&Go (203)and Karras l. (20)have prooniulans nt l However, despit thes provements, the trajectoricomon ODE involve partlysgifiant mont e l  E trajectories.

RFlowMoel u l 0;Albergo&Vane-Ejnen, 0; Li al, 0 eiv aas  olzo el018s  s. CopareFsRectFlows nd ochastcnterolants havevantage hat hey ot reqi te uraidelhe nesul E ta te pbil owODE Son 20ass wioelsNeveheles, heyonot resul x e improved performance.

NLP (Kapla al00)ancviiask Dsk al00 Zhaial .Forife, U-N eeoH e al 2.Whil e ret works xploiffuirar bacoes Peeble& Xie 0Chel ; Ma et al., 2024), scaling laws for text-to-image diffusion models remain unexplored.

![](images/20.jpg)  
Detailed pen and ink drawing of a happy pig butcher selling meat in its shop.

![](images/21.jpg)  
a massive alien space ship that is shaped like a pretzel.

![](images/22.jpg)  
A kangaroo holding a beer, wearing ski goggles and passionately singing silly songs.

![](images/23.jpg)  
An entire universe inside a bottle sitting on the shelf at walmart on sale.

![](images/24.jpg)  
A cheesburger surfing the vibe wave at night

![](images/25.jpg)  
A swamp ogre with a pearl earring by Johannes Vermeer

![](images/26.jpg)  
A car made out of vegetables.

![](images/27.jpg)  
heat death of the universe, line art

![](images/28.jpg)  
A crab made of cheese on a plate

![](images/29.jpg)  
Dystopia of thousand of workers picking cherries and feeding them into a machine that runs on steam and is as large as a skyscraper. Written on the side of the machine: "SD3 Paper"

![](images/30.jpg)  
translucent  sd  all .

![](images/31.jpg)  
Film still of a long-legged cute big-eye anthropomorphic cheeseburger wearing sneakers relaxing on the couch in a sparsely decorated living room.

![](images/32.jpg)  
detailed pen and ink drawing of a massive complex alien space ship above a farm in the middle of nowhere.

![](images/33.jpg)  
photo of a bear wearing a suit and tophat in a river in the middle of a forest holding a sign that says "I cant bear it".

![](images/34.jpg)  
  
an anthropomorphic fractal person behind the counter at a fractal themed restaurant

![](images/35.jpg)  
drn elu ysv

![](images/36.jpg)  
an anthopomorphic pink donut with a mustache and cowboy hat standing by a log cabin in a forest with an old 1970s orange truck in the driveway

![](images/37.jpg)

![](images/38.jpg)  
beautiful oil painting of a steamboat in a river in the afternoon. On the side of the river is a large brick building with a sign on top that says D3.   
fox sitting in front of a computer in a messy room at night. On the screen is a 3d modeling program with a line render of a zebra.

# B. On Flow Matching

# B.1. Details on Simulation-Free Training of Flows

Following (Lipman et al., 2023), to see that $u _ { t } ( z )$ generates $p _ { t }$ we no hatent proviy and sufficient condition (Villani, 2008):

Therefore it suffices to show that

$$
\begin{array} { r l } { - \nabla \cdot [ u _ { t } ( z ) p _ { t } ( z ) ] = - \nabla \cdot [ \mathbb { E } _ { \epsilon \sim \mathcal { N } ( 0 , I ) } u _ { t } ( z | \epsilon ) \frac { p _ { t } ( z | \epsilon ) } { p _ { t } ( z ) } p _ { t } ( z ) ] } & { } \\ { = \mathbb { E } _ { \epsilon \sim \mathcal { N } ( 0 , I ) } - \nabla \cdot [ u _ { t } ( z | \epsilon ) p _ { t } ( z | \epsilon ) ] } & { } \\ { = \mathbb { E } _ { \epsilon \sim \mathcal { N } ( 0 , I ) } \frac { d } { d t } p _ { t } ( z | \epsilon ) = \frac { d } { d t } p _ { t } ( z ) , } \end{array}
$$

/here we used the continuity equation Equation (2) fr $u _ { t } ( z | \epsilon )$ in line Equation (28) to Equation (29) since $u _ { t } ( z | \epsilon )$ enerates $p _ { t } ( z | \epsilon )$ and the definition of Equation (6) in line Equation (27)

The equivalence of objectives $\mathcal { L } _ { F M } \backslash = \mathcal { L } _ { C F M }$ (Lipman et al., 2023) follows from

$$
\begin{array} { r l } & { \mathcal { L } _ { F M } ( \Theta ) = \mathbb { E } _ { t , p _ { t } ( z ) } | | v _ { \Theta } ( z , t ) - u _ { t } ( z ) | | _ { 2 } ^ { 2 } } \\ & { \qquad = \mathbb { E } _ { t , p _ { t } ( z ) } | | v _ { \Theta } ( z , t ) | | _ { 2 } ^ { 2 } - 2 \mathbb { E } _ { t , p _ { t } ( z ) } \langle v _ { \Theta } ( z , t ) | u _ { t } ( z ) \rangle + c } \\ & { \qquad = \mathbb { E } _ { t , p _ { t } ( z ) } | | v _ { \Theta } ( z , t ) | | _ { 2 } ^ { 2 } - 2 \mathbb { E } _ { t , p _ { t } ( z | \epsilon ) , p ( \epsilon ) } \langle v _ { \Theta } ( z , t ) | u _ { t } ( z | \epsilon ) \rangle + c } \\ & { \qquad = \mathbb { E } _ { t , p _ { t } ( z | \epsilon ) , p ( \epsilon ) } | | v _ { \Theta } ( z , t ) - u _ { t } ( z | \epsilon ) | | _ { 2 } ^ { 2 } + c ^ { \prime } = \mathcal { L } _ { C F M } ( \Theta ) + c ^ { \prime } } \end{array}
$$

where $c , c ^ { \prime }$ do not depend on $\Theta$ and line Equation (31) to line Equation (32) follows from:

$$
\begin{array} { r l } { \mathbb { E } _ { p _ { t } ( z | \epsilon ) , p ( \epsilon ) } \langle v _ { \Theta } ( z , t ) | u _ { t } ( z | \epsilon ) \rangle = \displaystyle \int \mathrm { d } z \displaystyle \int \mathrm { d } \epsilon p _ { t } ( z | \epsilon ) p ( \epsilon ) \langle v _ { \Theta } ( z , t ) | u _ { t } ( z | \epsilon ) \rangle } & { { } } \\ { \displaystyle } & { { } = \displaystyle \int \mathrm { d } z p _ { t } ( z ) \langle v _ { \Theta } ( z , t ) | \displaystyle \int \mathrm { d } \epsilon \frac { p _ { t } ( z | \epsilon ) } { p _ { t } ( z ) } p ( \epsilon ) u _ { t } ( z | \epsilon ) \rangle } \\ { \displaystyle } & { { } = \displaystyle \int \mathrm { d } z p _ { t } ( z ) \langle v _ { \Theta } ( z , t ) | u _ { t } ( z ) \rangle = \mathbb { E } _ { p _ { t } ( z ) } \langle v _ { \Theta } ( z , t ) | u _ { t } ( z ) \rangle } \end{array}
$$

where we extended with pt(z) qu qqu Equation (36).

# B.2. Details on Image and Text Representations

Lat ImageRepreentation Wfollo LDM (Rombac  l 22)andus  prtrautcderrepretRGB images $X \in \mathbb { R } ^ { H \times W \times 3 }$ in a smaller latent space $x = E ( X ) \in \mathbb { R } ^ { h \times w \times d }$ Weus  spatial downsampig cor  8, u that $\begin{array} { r } { h = \frac { H } { 8 } } \end{array}$ and $\begin{array} { r } { w = \frac { W } { 8 } } \end{array}$ $d$ from Equation 2 in the latent space, and when sampling a representation via Equation 1, we decode it back into pixel space $X = D ( x )$ via the decoder $D$ . We follow Rombach et al. (2022) and normalize the latents by their mean and standard for different $d$ evolves as a function of model capacity, as discussed in Section 5.2.1.

Tex Repreentation Smilr the ecodin  mag latent epreentations welsofollow previus appe (Saharia et al., 2022b; Balaji et al., 2022) and encode the text conditioning $c$ using pretrained, frozen text models. In p Specifically, we encode $c$ with the text encoders of both a CLIP L/14 model of Radford et al. (2021) as well as an OpenCLIP bmoeCelae he pouus 768peivey a vector conditioning $c _ { \mathrm { v e c } } \in \mathbb { R } ^ { 2 0 4 8 }$ W ceh puthitat haeLIP context conitioning $c _ { \mathrm { c t x t } } ^ { \mathrm { C L I P } } \in \mathbb { R } ^ { 7 7 \times 2 0 4 8 }$ Next, we encode $c$ $c _ { \mathrm { c t x t } } ^ { \mathrm { T 5 } } \in \mathbb { R } ^ { 7 7 \times 4 0 9 6 }$ ,of the $c _ { \mathrm { c t x t } } ^ { \mathrm { C L I P } }$ $c _ { \mathrm { c t x t } } ^ { \mathrm { T 5 } }$ $c _ { \mathrm { c t x t } } \in \mathbb { R } ^ { 1 5 4 \times 4 0 9 6 }$ $c _ { \mathrm { v e c } }$ and $c _ { \mathrm { c t x t } }$

![](images/39.jpg)  
16-channel autoencoder space needs more model capacity to achieve similar performance. At depth $d = 2 2$ , the gap between 8-chn and 16-cn becomes negligibleWe opt for the 16-chn model as we ultimately aim to scale to much larger model sizes.

# B.3. Preliminaries for the Experiments in Section 5.1.

aW e ea uosk  tabexo captns of the form  photo of aclassname"to mages, whereasnameis randomly chosen fromne te roir eas labsoealext--ta euh (Changpinyo et al., 2021) for training.

Optization In this experiment, we trainl moels using a global batch siz  1024 using theAdamWotizer (Loshchilov & Hutter, 2017) with a learning rate of $1 0 ^ { - 4 }$ and 1000 linear warmup steps. We use mixed-precision training and keee hi  aae  o (A of the three text encoders independently to zero with a probability of $4 6 . 4 \%$ , such that we roughly train an unconditional model in $1 0 \%$ of all steps.

va e. eLI o aoale y luring training on the C0CO-2014 validation split (Lin et al., 2014).

ight equally spaced values in the time interval [0, 1].

Toanalyzeowfet pes behavnder ifft sp sett we pro smp o e s LIP L Ror l 021neCLP L1e u

# B.4. Improving SNR Samplers for Rectified Flow Models

As described in Section 2, we introduce novel densities $\pi ( t )$ for the timesteps that we use to train our rectified flow models. 0t a ucDM(Kar0nDM l.

![](images/40.jpg)  
.

![](images/41.jpg)  
200k, 350k, 500k) and model sizes (top to bottom: depth $_ { = 1 5 }$ ,30, 38) on PartiPrompts, highlighting the influence of training duration and model complexity.

# C. Direct Preference Optimization

![](images/42.jpg)  
samples with better spelling

, thismeas beeapt preexifo Wal lIn Wl   aR  e learble Low-Ranaptain (LoRAmari rank1) r llearayers  is  praciWeee e Figure 13 shows samples of the respective base models and DPO-finetuned models.

# D. Finetuning for instruction-based image editing

Ac prcorraistr basmdinealimage-to-mgeifusn moels  t f he au e paha

![](images/43.jpg)  
models for both prompt following and general quality.

<table><tr><td>Model</td><td>Mem [GB]</td><td>FP [ms]</td><td>Storage [kB]</td><td>Delta [%]</td></tr><tr><td>VAE (Enc)</td><td>0.14</td><td>2.45</td><td>65.5</td><td>13.8</td></tr><tr><td>CLIP-L</td><td>0.49</td><td>0.45</td><td>121.3</td><td>2.6</td></tr><tr><td>CLIP-G</td><td>2.78</td><td>2.77</td><td>202.2</td><td>15.6</td></tr><tr><td>T5</td><td>19.05</td><td>17.46</td><td>630.7</td><td>98.3</td></tr></table>

T $[ \% ]$ is how much longer a training step takes, when adding this into the loop for the 2B MMDiT-Model (568ms/it).

t  u training a SDXL-based (Podell et al., 2023) editing model on the same data.

# E. Data Preprocessing for Large-Scale Text-to-Image Training

# E.1. Precomputing Image and Text Embeddings

Ou epoch, see Tab. 7.

T   o during training $_ { c , f }$ Tab. 7). We save the embeddings of the language models in half precision, as we do not observe a deterioration in performance in practice.

# E.2. Preventing Image Memorization

In y scan our training dataset for duplicated examples and remove them.

![](images/44.jpg)  
Figure 15. Zero Shot Text manipulation and insertion with the 2B Edit model

r , clustering and otherdownstream tasks.We also decided to follow Nichol (2022) to decide on a number of clusters $N$ .For our experiments, we use $N = 1 6 , 0 0 0$ .

W data, such as image embeddings.

Altra pW eento  ve f C hreshol s owngu1Base  theseresults weelece fourhrehol o heal u igu

# E.3. Assessing the Efficacy of our Deduplication Efforts

C t taamp bu heremanit rkeyemeoiz han o-uplicatexale (Somepalli et al., 2023a;a; Lee et al., 2021).

To e rk  r0 a uG tsehicrao (2023). Note that we run this techniques two times; one for $\mathrm { S D } { - 2 . 1 }$ model with only exact dedup removal as baseline, and fom wie .uutaixacupneauplatsC et al., 2022).

W0ua o.a00aa xt op c kendzatT intuition is that for diffusion models, with high probability $G e n ( p ; r _ { 1 } ) \approx _ { d } G e n ( p ; r _ { 2 } )$ for two different random initial seeds $r _ { 1 } , r _ { 2 }$ . On the other hand, if $G e n ( p ; r _ { 1 } ) \approx _ { d } G e n ( p ; r _ { 2 } )$ under some distance measure d, it is likely that these generated samples are memorized examples. To compute the distance measure $d$ between two images, we use a modified Euclidean $l _ { 2 }$ dia.In parlar,  nd at may is w  surs mr $l _ { 2 }$ distance (e.g., they all had gray backgrounds). We therefore instead divide each image into 16 non-overlapping $1 2 8 \times 1 2 8$ tiles and measure the maximum of the $l _ { 2 }$ distance between any pair of image tiles between the two images. Figure 17 shows the comparison betweenumber f memorized sample, before and after using D with the threshold of 0.5 to oe -uplicat sampes.Car  al. (0 markimages withi clique iz 0 a memoriz samples. e we alep iz  qorqreolblantuhe at ${ \mathrm { S S C D } } { = } 0 . 5$ show a $5 \times$ reduction in potentially memorized examples.

# Algorithm 1 Finding Duplicate Items in a Cluster

Ris  gusLis   AI   
for similarity search within the cluster, thresh  Threshold for determining duplicates   
Output: dups  Set of duplicate item IDs   
1: dups new set()   
2: for $i \gets 0$ to length(vecs) − 1 do   
3: $\mathsf { q s } \gets \mathsf { v e c s } [ i ]$ {Current vector}   
4: qid items[] {Current item ID}   
5: lims, $D , I $ index.range_search(qs, thresh)   
6: if qid $\in$ dups then   
7: continue   
8: end if   
9: start ← lims[0]   
10: end lims[1]   
11: duplicate_indices $ I [ s t a r t : e n d ]$   
12: duplicate_ids new list()   
13: for $j$ in duplicate_indices do   
14: if items $[ j ] \neq$ qid then   
15: duplicate_ids.append(items[j])   
16: end if   
17: end for   
18: dups.update(duplicate_ids)   
19end for   
20: Return dups {Final set of duplicate $\mathrm { I D s } \}$

(a) Final result of SSCD deduplication over the entire dataset

![](images/45.jpg)  
SSCD: Analysis of % removed images 1000 radomly selected clusters

![](images/46.jpg)  
(b) Result of SSCD deduplication with various thresholds over 1000 random clusters   
Figure 16. Results of deduplicating our training datasets for various filtering thresholds.

# Algorithm 2 Detecting Memorization in Generated Images

Require: Set of prompts $P$ , Number of generations per prompt $N$ , Similarity threshold $\epsilon = 0 . 1 5$ , Memorization threshold   
T   
Ensure: Detection of memorized images in generated samples   
1: Initialize $D$ to the set of most-duplicated examples   
: for each prompt $p \in P$ do   
3: for $i = 1$ to $N$ do   
4: Generate image ${ \mathrm { G e n } } ( p ; r _ { i } )$ with random seed $r _ { i }$   
5: end for   
6end for   
7 for each pair of generated images $x _ { i } , x _ { j }$ do   
8: if distance $d ( x _ { i } , x _ { j } ) < \epsilon$ then   
9: Connect $x _ { i }$ and $x _ { j }$ in graph $G$   
10: end if   
11end for   
12: for each node in $G$ do   
13: Find largest clique containing the node   
14: if size of clique $\geq T$ then   
15: Mark images in the clique as memorized   
16: end if   
17: end for

![](images/47.jpg)  
F ba lipevza o Cok models on the deduplicated training samples cut off at ${ \mathrm { S S C D } } { = } 0 . 5$ show a $5 \times$ reduction in potentially memorized examples.