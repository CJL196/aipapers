# ScoreHOI: 基于评分引导扩散的物理合理人-物交互重建

李傲^1,2 刘金鹏^1,2 朱奕璇^2 唐彦松^1,2* ^1清华大学深圳国际研究生院 ^2清华大学

# 摘要

人机交互的联合重建标志着理解人类与其周围环境之间复杂关系的重要里程碑。然而，以往的优化方法常常由于缺乏关于人机交互的先验知识而难以实现物理上合理的重建结果。本文介绍了ScoreHOI，这是一种有效的基于扩散的优化器，通过引入扩散先验实现人机交互的精确恢复。通过利用基于得分的采样中的可控性，扩散模型能够在给定图像观察和物体特征的情况下重建人类与物体姿态的条件分布。在推理过程中，ScoreHOI通过用特定的物理约束引导去噪过程，有效改善了重建结果。此外，我们提出了一种基于接触驱动的迭代优化方法，以增强接触的合理性并提高重建精度。对标准基准的广泛评估表明，ScoreHOI在性能上优于最先进的方法，突显了其在联合人机交互重建中的精确和稳健的改进能力。代码可在 https://github.com/RammusLeo/ ScoreHOI.git 访问。

# 1. 引言

联合三维人类-物体交互重建的任务涉及从图像或视频中恢复人类身体网格和交互物体的姿态。在最近几十年中，人类网格恢复（HMR）因其广泛的应用范围而在研究界引起了显著关注。同时，越来越多的研究强调重建人类与物体之间的交互的重要性，而不仅仅关注人类的重建。这种方法在多个应用领域具有巨大的潜力，例如机器人数据收集、虚拟现实/增强现实和游戏开发。

![](images/1.jpg)  
Figure 1. Comparison of current methods and the proposed ScoreHOI. (a) Optimization-based methods iteratively refine predicted outcomes with the physical objectives. (b) Regressionbased methods update predictions via a dual-branch forward process. (c) Our ScoreHOI integrates a score-based denoising module that incorporates physical constraints during the sampling process.

尽管取得了一些进展，但从单目图像直接重建人类和物体网格仍然是一项艰巨的挑战。这一困难源于忽略了许多人类与物体之间的交互模式，例如抓取、提起或坐下，这些交互模式构成了恢复任务的重要先验知识。为了应对这一问题，现有的方法已经纳入了细化模块，旨在将粗略估计转化为更准确的结果。如图1所示，目前的方法可以大致分为基于优化和基于回归的两种方式。基于优化的技术采用诸如Adam的联合优化器进行迭代细化，通过纳入物理目标（如接触和碰撞约束）来改进结果。然而，过分强调物理约束而忽视图像级特征往往会导致显著的重建不准确。此外，这些方法通常效率低下，需要大量计算时间。相比之下，基于回归的方法设计了一种考虑接触的交叉注意机制，以在整合图像特征的前向过程中重新建立人机交互关系。然而，参数的单步前向细化往往缺乏鲁棒性，特别是在严重遮挡或深度模糊等场景下。最近，扩散模型已经证明其能够从高斯噪声中恢复数据分布。这些模型通过对齐隐含数据分布的对数密度梯度，推断底层数据分布的隐式先验。利用贝叶斯定理，去噪过程可以通过额外的真实引导项得到增强，从而使优化过程更接近观察到的数据。由于其丰富的先验知识和引导采样的能力，扩散模型已被用于提高HMR任务中的估计质量。更近期的方法尝试将扩散模型融入人机重建中。然而，这些方法主要旨在利用扩散技术生成点云，而我们则专注于利用扩散先验来细化人机交互。为了克服先前方法的上述局限性，我们提出了ScoreHOI，一个有效的框架，利用基于得分的扩散模型来细化粗略估计结果，如图1下方所示。我们的设计考虑了两个主要挑战：（1）将人机交互的先验知识整合到优化过程中，以及（2）用合理的物理约束来监督采样过程。为了解决这些问题，我们提出了一种接触驱动的迭代细化方法，利用扩散生成模型作为一种具有广泛先验知识的强大优化器。在推理阶段，我们首先通过DDIM反演将初始回归估计反转为噪声潜表示。随后，我们通过DDIM采样来去噪该潜表示，并增加接触和碰撞约束等交互引导。为了增强可控性，我们引入了物体几何形状和图像特征作为条件。此外，我们迭代细化接触掩码，以提高物理 plausibility。我们在标准基准上进行了广泛的实验，包括BEHAVE和InterCap，基于各种初始回归结果。我们的ScoreHOI展现出了强大的鲁棒性和有效性，超越了以前的最先进方法，并显著提高了准确性。值得注意的是，我们在BEHAVE基准上实现了9%的接触F分数改进，从而证明了我们方法的优化能力。此外，我们进行了全面的消融研究，以强调我们管道和我们设计的细化的有效性和效率。

# 2. 相关工作

人机交互重建。建模人类与物体之间的三维交互是一个重要挑战。尽管最近的研究在使用高维数据和图像建模人类运动方面表现出了强大的性能，但这些方法仅限于手与物体的交互，无法扩展到全身交互，这更加复杂。一些方法，如PROX，可以适应场景约束来拟合三维人类模型，从多个视角捕捉交互，或基于人类与场景的交互重建三维场景。近年来，研究还扩展到人类间交互和自我接触。尽管有这些进展，现有的方法在单张图像中共同重建人机接触方面仍然存在困难。一些研究尝试从图像和视频中推断三维接触，但未能实现从单一RGB图像重建全身和物体。Weng等人分别预测三维人类和场景布局，并利用场景约束来优化人类重建。他们在SMPL-X顶点上应用预定义接触权重，这可能导致不准确。PHOSA分别拟合SMPL和物体网格，并依赖于预定义的接触对推理交互，但这些启发式方法不具有可扩展性，且常常缺乏准确性。我们的工作通过接触驱动的迭代细化注入多样的物理约束，从而使物体和人类网格的位置更加合理。

基于分数的扩散模型。扩散模型已成为表示复杂概率分布的强大工具，在文本生成图像方面显示出了卓越的性能。除了传统的去噪技术之外，引入了分数的概念，以表征扰动数据分布的时间相关梯度场。基于分数的采样应用使得去噪过程能够通过多种形式的指导来引导，从而促进了适用于广泛任务的方法的发展，包括超分辨率和图像修复。最近，像一些方法探讨了基于分数的扩散模型在人体物体重建任务中的潜力，通过观察对结果进行优化。然而，在三维人体物体重建的背景下，基于分数的优化尚未得到充分探索。我们的研究旨在通过将扩散先验整合到该领域中来填补这一空白，这有望提高重建的准确性和交互的物理合理性。

![](images/2.jpg)  
Figure 2. The inference procedure of ScoreHOI. (a) Given the input image $I$ the human and object segmented silhouette $S _ { \mathrm { h } } , S _ { \mathrm { o } }$ and the object template $P _ { \mathrm { ~ o ~ } }$ , we initially extract the image feature $\mathcal { F }$ and estimate the SMPL and object parameters $\theta$ $\beta$ , $R _ { \mathrm { o } }$ and $t _ { \mathrm { o } }$ . (b) Employing a contact-driven iterative refinement strategy, we refine these parameters $_ { \textbf { \em x } }$ through the execution of a DDIM inversion and guided sampling lo Due i s, hysl ctant uc  pe $L _ { \mathrm { p t } }$ and contact $L _ { \mathrm { h o } } , L _ { \mathrm { o f } }$ are actively supervised. Following each optimization iteration, the contact masks $\{ \mathbf { M } _ { i } \} _ { i \in \{ \mathrm { h , o , f } \} }$ are updated to enhance the precision of the guidance.

# 3. 方法

在本节中，我们介绍了ScoreHOI，这是一种有效的优化器，旨在利用先验知识和物理约束进行人类与物体交互重建。我们将首先回顾人体模型和基于评分的扩散模型的背景。接着，我们将首次介绍从考虑可供性回归器到基于接触的迭代优化的完整推理流程，如图2所示。随后，我们将展示扩散模型的架构及其他细节。

# 3.1. 准备工作

人体模型。SMPL [33] 是一种复杂的参数化模型，旨在三维表示人类身体。该模型由两组参数特征：姿态参数 $\theta \in \mathbb { R } ^ { 2 4 \times 3 }$，用于捕捉身体的方向和配置，以及形状参数 $\beta \in \mathbb { R } ^ { 1 0 }$ ，定义身体的物理特征，例如大小和比例。该模型建立了一个映射函数 ${ \mathcal { M } } ( \theta , \beta )$，将这些参数转换为详细的身体网格 $\mathcal { M } \in \mathbb { R } ^ { N \times 3 }$，其中 $ { N _ { \mathrm { ~ \scriptsize ~ = ~ } } } 6 9 8 0$ 表示组成网格的总顶点数量。SMPL-H [33, 39, 45] 模型基于参数化表示，结合了 SMPL [33] 身体模型和 MANO [45] 手部模型。这允许生成逼真的人类形象，可应用于各种场景，包括用手操控物体和与环境互动。

基于评分的扩散模型。扩散模型，如 DDPM [20]，包含两个过程：正向扩散和反向去噪。在正向扩散中，数据 $\scriptstyle { \mathbf { { \vec { x } } } } _ { 0 }$ 在 $t ~ = ~ T$ 时间步长内使用具有方差调度 $\{ \zeta _ { t } \}$ 的高斯核被转换为含噪数据 $\mathbf { \nabla } _ { \mathbf { x } _ { T } }$，定义为 $\ v { q } ( \mathbf { x } _ { t } | \mathbf { x } _ { 0 } ) = \mathcal { N } ( \sqrt { \alpha _ { t } } \mathbf { x } _ { 0 } , ( 1 - \alpha _ { t } ) \mathbf { I } )$，其中 $\begin{array} { r } { \alpha _ { t } : = \prod _ { s = 1 } ^ { t } ( 1 - \zeta _ { s } ) } \end{array}$。在反向去噪过程中，训练一个去噪模型 $\epsilon _ { \phi }$ 以通过最小化以下损失函数来预测噪声：$\mathcal { L } ( \phi ) = \mathbb { E } _ { \pmb { x } _ { 0 } , t , \epsilon } | | \epsilon _ { \phi } ( \pmb { x } _ { t } , t ) - \epsilon | | ^ { 2 }$，其中 $t$ 是从 $\{ 1 , . . , T \}$ 均匀抽样得到的，$\epsilon$ 是添加到 $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ 的噪声。根据 [49]，模型在时间步长 $t$ 处预测的噪声与模型的评分相关联：

$$
\begin{array} { r } { \epsilon _ { \phi } ( \pmb { x } _ { t } , t ) = - \sqrt { 1 - \alpha _ { t } } \nabla _ { \pmb { x } _ { t } } \log p ( \pmb { x } _ { t } ) , } \end{array}
$$

其中 $\nabla _ { \pmb { x } _ { t } } \log p ( \pmb { x } _ { t } )$ 指的是评分，它描述了扰动数据分布的时间依赖梯度场。对于加速采样，可以使用 DDIM [47]，它采用非马尔可夫过程。基于 1，

# 算法 1 基于接触的迭代优化

输入：初始参数 $\pmb { x } _ { 0 } ^ { 0 }$，扩散模型 $\epsilon _ { \theta }$，图像特征 $\mathcal { F }$，时间步 $t$，以及条件 $^ c$ 输出：优化后的参数 $\pmb { x } _ { 0 } ^ { N }$ 对于 $n = 0$ 到 $N - 1$ 做 $\mathcal { F } _ { \mathrm { h } } ^ { n } , \mathcal { F } _ { \mathrm { o } } ^ { n } \bf { 采样 } ( \pmb { x } _ { 0 } ^ { n } , \mathcal { F } )$ $\{ \mathbf { M } _ { i } \} _ { i \in \{ \mathrm { h , o , f } \} } \mathbf { 连接 } ( x _ { 0 } ^ { n } , \mathcal { F } _ { \mathrm { h } } ^ { n } , \mathcal { F } _ { \mathrm { o } } ^ { n } )$ $\mathbf { \boldsymbol { x } } _ { 0 } ^ { n + 1 } \gets \mathbf { \boldsymbol { D } } \mathbf { \boldsymbol { D } } \mathbf { \boldsymbol { I } } \mathbf { \boldsymbol { M } } \mathbf { \boldsymbol { L 0 0 p } } ( \mathbf { \boldsymbol { x } } _ { 0 } ^ { n } , t , c , \epsilon _ { \theta } , \{ \mathbf { M } _ { i } \} _ { i \in \{ \mathrm { h , o , f } \} } )$ 结束循环 返回 $\pmb { x } _ { 0 } ^ { N }$ 的去噪结果 $\hat { x _ { 0 } } ( \pmb { x } _ { t } )$ 来自 $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ 的计算为：

$$
\begin{array} { l } { \displaystyle \hat { \pmb { x } _ { 0 } } ( \pmb { x } _ { t } ) = \frac { 1 } { \sqrt { \alpha _ { t } } } ( \pmb { x } _ { t } - \sqrt { 1 - \alpha _ { t } } \epsilon _ { \phi } ( \pmb { x } _ { t } , t ) ) } \\ { \displaystyle \simeq \frac { 1 } { \sqrt { \alpha _ { t } } } ( \pmb { x } _ { t } + ( 1 - \alpha _ { t } ) \nabla _ { \pmb { x } _ { t } } \log p ( \pmb { x } _ { t } ) ) . } \end{array}
$$

该框架也支持条件分布。

# 3.2. 考虑可供性回归器

对于联合人类-物体交互重建任务，我们的主要目标是精确估计人体手部姿态 $\theta \in \mathbb { R } ^ { 5 2 \times 6 }$，人体形状参数 $\beta \in \mathbb { R } ^ { 1 0 }$，以及物体旋转 $R _ { \mathrm { o } } \in \mathbb { R } ^ { 6 }$ 和物体平移 $t _ { \mathrm { o } } ~ \in \ \mathbb { R } ^ { 3 }$。我们采用六维旋转格式，遵循文献 [73] 对 $\theta$ 和 $\scriptstyle { \mathbf { } } _ { R _ { \mathrm { o } } }$ 的表示，以增强预测结果的稳定性。我们使用裁剪后的图像 $I _ { r g b } \in \mathbf { \bar { \mathbb { R } } ^ { H \times \mathbf { \bar { W } } \times 3 }}$ 及其对应的人类和物体分割部分 $S _ { \mathrm { h } } ~ \in ~ \mathbb { R } ^ { H \times \bar { W } \times 3 }$ 和 $S _ { \mathrm { o } } ~ \in ~ \mathbb { R } ^ { H \times \bar { W } \times 3 }$ 作为视觉输入。根据之前的工作 [37, 59]，给定一个粗略的物体模板 $ { P _ { \mathrm { ~ o ~ } } } \in \ \mathbb { R } ^ { 6 4 \times 3 }$ 用于初始化物体的形状。然而，在一般情况下，物体的形状有许多变种，而我们的训练数据中的物体形状是有限的。之前的工作如 [37] 嵌入类标识以注入物体类别信息，但当物体形状超出训练集时，这种方法不可用。

尽管物体形状各异，但我们有一个共同假设，即同一类别的物体具有相似的外观，例如，桌子有平坦的表面，通常有支撑在地面上的腿。因此，为了增强泛化能力，我们引入了受到[40]启发的可供性概念。可供性是指物体的感知或实际属性，这些属性暗示了物体的使用方式，可以通过预训练的可供性感知网络构建。预训练模型从大规模三维物体数据集中生成丰富的先验知识，这对重建非常有价值。借助可供性感知，我们提取图像特征 $\mathcal{F}$，然后应用两个独立的头部来初步估计人体手部的SMPL-H参数 $\theta^{0}$ 和 $\beta^{0}$，以及物体的旋转 $R_{\mathrm{o}}^{0}$ 和平移 $t_{\mathrm{o}}^{0}$。

# 3.3. 物理引导下的优化

在获得初始参数后，我们将优化目标定义为人体姿态、人身体形状、物体旋转和物体平移的连接，表示为 $\pmb { x } = \{ \theta , \beta , R _ { 0 } , t _ { 0 } \} \in \mathbb { R } ^ { 3 3 1 }$。我们从初始的估计结果 $\pmb { x } ^ { \mathrm { i n i t } }$ 开始，利用 DDIM 逆过程获得噪声潜变量 $\scriptstyle { \mathbf { 2 } } \left( { \mathbf { 2 } } \right)$，其中 $\tau$ 是噪声水平：

$$
\pmb { x } _ { t + 1 } = \sqrt { \alpha _ { t + 1 } } \hat { \pmb { x } _ { 0 } } ( \pmb { x } _ { t } ) + \sqrt { 1 - \alpha _ { t + 1 } } \epsilon _ { \phi } ( \pmb { x } _ { t } , t , \pmb { c } ) ,
$$

其中 $^ c$ 是图像特征条件 $c _ { \mathrm { I } }$ 和几何特征条件 $c _ { \mathrm { G } }$ 的组合，定义于第 3.5 节。

在获得 ${ \bf { x } } _ { \tau }$ 后，可以通过前向 DDIM 采样过程检索 $\pmb { x } ^ { \mathrm { i n i t } }$。然而，目的是通过引入物理约束来增强 $\pmb { x } ^ { \mathrm { i n i t } }$。我们不使用原始分数 $\nabla _ { \pmb { x } _ { t } } \log p ( \pmb { x } _ { t } | \pmb { c } )$，而是应用具有物理目标的条件分数 $\nabla _ { \pmb { x } _ { t } } \log p ( \pmb { x } _ { t } | \pmb { c } , \mathcal { P } )$。根据贝叶斯法则，分数可以写成 $\nabla _ { \pmb { x } _ { t } } \log p ( \pmb { x } _ { t } | \pmb { c } , \mathcal { P } ) = \nabla _ { \pmb { x } _ { t } } \log p ( \pmb { x } _ { t } | \pmb { c } ) + \nabla _ { \pmb { x } _ { t } } \log p ( \mathcal { P } | \pmb { c } , \pmb { x } _ { t } )$，其中 $\nabla _ { \pmb { x } _ { t } } \log _ { \pmb { p } } ( \pmb { x } _ { t } | \pmb { c } )$ 是扩散模型的输出。然而，从 $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ 直接计算 $\nabla _ { \pmb { x } _ { t } } \log p ( \mathcal { P } | \pmb { c } , \pmb { x } _ { t } )$ 是困难的。受到 [7, 22, 50] 的启发，我们可以做一个假设：

$$
\nabla _ { \pmb { x } _ { t } } \log p ( \mathcal { P } | \pmb { c } , \pmb { x } _ { t } ) \simeq \nabla _ { \pmb { x } _ { t } } \log p ( \mathcal { P } | \pmb { c } , \hat { \pmb { x } _ { 0 } } ( \pmb { x } _ { t } ) )
$$

其中 $\hat { \pmb { x } _ { 0 } } ( \pmb { x } _ { t } )$ 表示来自 $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ 的去噪结果。因此，预测的人体网格 $\hat { V } _ { 0 } ^ { \mathrm { h } } ( V _ { t } ^ { \mathrm { h } } )$ 和物体网格 $\hat { V } _ { 0 } ^ { \mathrm { o } } ( V _ { t } ^ { \mathrm { o } } )$ 可以从 $\hat { \pmb { x } _ { 0 } } ( \pmb { x } _ { t } )$ 中推导出来，以阐明交互关系，这构成了计算物理约束的基本要素。现在我们旨在定义物理目标，包括人-物接触、物-地面接触和三维穿透，参见[22, 30, 60]：

$$
L _ { \mathcal { P } } = \lambda _ { \mathrm { h o } } L _ { \mathrm { h o } } + \lambda _ { \mathrm { o f } } L _ { \mathrm { o f } } + \lambda _ { \mathrm { p t } } L _ { \mathrm { p t } } ,
$$

个人约束条件如下所示：

人-物接触：$L _ { \mathrm { h o } } = | | ( \mathbf { M } _ { \mathrm { h } } + \mathbf { M } _ { \mathrm { o } } ) \odot | V _ { \mathrm { h } } -$ $V _ { \mathrm { o } } | \ | | _ { 2 }$。在人类接触区域 $\mathbf { M } _ { \mathrm { h } }$ 和物体接触区域 $\mathbf { M } _ { \mathrm { o } }$ 处，人类网格 $V _ { \mathrm { h } }$ 与物体网格 $V _ { \mathrm { o } }$ 之间的欧几里得距离理想情况下应为零。物-地面接触：$L _ { \mathrm { o f } } = | | \mathbf { M } _ { \mathrm { f } } \odot | V _ { \mathrm { o } } | \ | | _ { 1 }$。 $\mathbf { M } _ { \mathrm { f } }$ 表示与地面表面接触的物体顶点集合。对于此集合中的顶点，其高度应为零，表示与地面的直接接触。穿透避免：$L _ { \mathrm { p t } } = - \mathbb { E } [ | \Phi _ { 0 } ^ { - } ( V _ { \mathrm { h } } ) | ]$。我们通过利用物体的符号距离函数 (SDF) $\Phi _ { 0 } ^ { - }$ 为穿透场景引入惩罚。当发生重叠时，$L _ { \mathrm { p t } }$ 的值增加。最后，带有物理约束的修正噪声预测表示为：

$$
\epsilon _ { \phi } ^ { \prime } = \epsilon _ { \phi } ( \pmb { x } _ { t } , t , \pmb { c } ) + \rho \sqrt { 1 - \alpha _ { t } } \nabla _ { \pmb { x } _ { t } ^ { n } } L _ { \mathcal { P } }
$$

其中 $\rho$ 是用于控制引导尺度的权重。

# 3.4. 基于接触的迭代优化

在由物理引导的去噪过程中，我们发现接触区域的精确预测是一个关键因素。这些区域的预测来自于人类和物体姿态参数 $_ { \textbf { \em x } }$ 的网格采样特征 $\mathcal { F } _ { \mathrm { h } }$ 和 $\mathcal { F } _ { \mathrm { o } }$。然而，通过单次前向过程直接估计接触掩模容易出现缺陷，尤其是在严重遮挡的情况下。为了解决这个问题，我们实施了一种基于接触的迭代精炼策略，以增强推理阶段接触估计的准确性，如算法 1 所示。从第 $n$ 次迭代的初始参数 $\pmb { x } _ { 0 } ^ { n }$ 开始，我们首先从图像特征 $\mathcal { F }$ 中采样 $\mathcal { F } _ { \mathrm { h } }$ 和 $\mathcal { F } _ { \mathrm { o } }$。随后，人对物体接触掩模 $\mathbf { M } _ { \mathrm { h } }$、物体对人接触掩模 $\mathbf { M } _ { \mathrm { h } }$ 和物体对地面接触掩模 $\mathbf { M } _ { \mathrm { f } }$ 将通过接触预测器进行更新。接着，如第 3.3 节所述，我们通过引导采样获得有噪声的 $\pmb { x } _ { \tau } ^ { n }$，并通过 DDIMn $\pmb { x } _ { 0 } ^ { n + 1 }$，其中接触掩模和估计的人类及物体顶点被纳入其中。假设我们优化 $N$ 步，最终迭代将产生 $\pmb { x } _ { 0 } ^ { N }$。继 [37] 之后，我们应用双分支掩模变换器 [53] 来最终增强网格重建的质量。

# 3.5. 扩散模型架构

主要扩散模型由 3 个交叉注意力层构成。然而，在没有任何指导的情况下，生成结果无法控制，更不用说在重建任务中。为了解决这个问题，我们提出了一种 IG-Adapter，可以融入物体几何先验知识和视觉观察作为指导。我们引入了两个条件，包括由 $\mathcal { F }$ 的平均池化所得的图像特征条件 $c _ { \mathrm { I } }$ 和从预训练的感知意识网络中提取的几何特征条件 $c _ { \mathrm { o } }$。受到 [65] 的启发，我们训练了一个额外的交叉注意力块及其相关的融合线性头，如图 3 所示。通过结合几何先验和观察信息的注意力值，模型具备了互动意识和视觉意识。对于人形参数 $\beta$ ，我们采用相同的由 3 个交叉注意力层组成的分支。此外，在处理过程中，我们将 $\beta$ 标准化到 $[-1, 1]$。关于时间步，根据 [50]，输入通过缩放和平移操作以时间步 $t$ 为条件，定义为 ${ \pmb x } _ { t } = t _ { s } { \pmb x } + t _ { b }$，其中参数 $t _ { s }$ 和 $t _ { b }$ 来源于 MLP 编码器。在指导 $c _ { \mathrm { I } }$ 和 $\mathbf { c _ { G } }$ 的帮助下，我们可以以以下目标训练扩散模型：

$$
L _ { \mathrm { D M } } = \mathbb { E } _ { { \pmb { x } } _ { 0 } , { \epsilon } , { t } , { \pmb { c } } _ { \mathrm { I } } , { \pmb { c } } _ { \mathrm { G } } } | | \epsilon - { \epsilon } _ { \theta } ( { \pmb { x } } _ { t } , { t } , { \pmb { c } } _ { \mathrm { I } } , { \pmb { c } } _ { \mathrm { G } } ) | | ^ { 2 }
$$

# 3.6. 实现

Pytorch [38] 被用于实现。对于图像特征提取器，我们采用 ResNet50 [19] 作为主干网络。为了初步估计 SMPL-H 参数，我们采用 Hand4Whole 框架。关于预训练的基于能力感知的网络，我们使用 PointNeXt [42]，并在训练期间冻结模型以保持物体特征提取的稳定性。在推理阶段，执行 DDIM 采样，步长 $\Delta t = 2$，接触驱动的迭代优化重复进行 $N = 10$ 次。选择的中间噪声水平 $\tau$ 为 0.05。有关超参数的更多细节请参见第 4.3 节。

![](images/3.jpg)  
Figure 3. The overview of IG-Adapter. We introduce an IGAdapter designed to integrate the image feature guidance $c _ { \mathrm { I } }$ and the geometry feature guidance $c _ { \mathrm { G } }$ into the diffusion model. The incorporation of observational and geometric awareness enhances the controllability of the model during the inference process.

# 4. 实验

为了验证我们提出的 ScoreHOI 的有效性和效率，我们在标准基准上进行了全面的实验和消融研究。我们将介绍实验设置，并通过详细的比较和消融研究提供全面的分析。

# 4.1. 实验设置

数据集。BEHAVE [2] 数据集是自然环境中人类与物体交互的最大数据集，包含三维人类、物体和接触标注。该数据集包括：（1）8 名受试者与 20 个物体在 5 个自然环境中进行交互；（2）总共记录了 321 个视频序列，使用 4 台 Kinect RGB-D 相机；（3）20 个物体的纹理扫描重建。InterCap [23] 包含 10 名受试者（5 名男性和 5 名女性）与 10 个各种大小和使用特性的物体进行交互，包含手或脚的接触。总计，InterCap 拥有 223 个 RGBD 视频，生成 67,357 帧多视角图像，每帧包含 6 张 RGB-D 图像。IMHD 数据集 [72] 包含 15 名受试者（13 名男性，2 名女性）参与 10 种不同的交互场景。它还为每次捕捉提供了序列级的文本指导。每次拆分持续时间为 30 秒到 1 分钟不等。它包含具有实例级分割的 32 视图 RGB 视频。IMHD 利用 ViTPose [64] 和 MediaPipe 对 2D 和 3D 人体关键点进行标注。

![](images/4.jpg)  
F    HO effectively addresses this ill-posed problem, achieving superior reconstruction fidelity.

我们仅在扩散训练阶段使用 IMHD 训练集来增强生成能力。

训练细节。训练阶段分为两个部分。最初，我们在第一阶段训练图像主干网络、接触预测器和顶点精炼，遵循文献[37]中的损失函数，包括对SMPL-H和物体姿态参数、接触掩码以及2D/3D顶点的监督等。我们在BEHAVE [2]和InterCap [23]训练集上训练这些基础模块。随后，使用从冻结的图像主干网络中提取的图像特征训练扩散模型，结合IMHD [72]以增强生成能力。权重更新通过Adam优化器[27]进行，第一阶段的迷你批次大小为32，第二阶段为256。在两个阶段中，采用缩放、旋转和颜色抖动等数据增强技术，以增强数据集的多样性。第一阶段的训练总共进行50个epoch，初始学习率为$1 0 ^ { - 4 }$，在30个epoch后减少10倍。在第二阶段，参数收敛更快，因此只需以学习率$1 0 ^ { - 4 }$进行30个epoch的训练。模型使用4进行训练。

# NVIDIA RTX 4090 GPU 总共约使用 1.5 天。

评估细节。根据之前的研究 [37, 59, 68]，我们利用两种指标来提高模型性能：Chamfer距离、精度和召回率。Chamfer距离 $\mathbf { \hat { C D } _ { h u m a n } }$ 和 $\mathbf { C D _ { o b j e c t } }$。给定预测的3D人类和物体网格，我们对结合的3D人类和物体网格与真实标定（GT）3D人类和物体网格进行Procrustes对齐。通过对齐后的3D人类和物体网格，我们分别在3D人类 $\mathrm { ( C D _ { h u m a n } ) }$ 和3D物体 $\mathrm { ( C D _ { o b j e c t } ) }$ 上测量Chamfer距离，单位为厘米。接触的精度、召回率和F-Score来自重构 $\mathbf { ( C o n t a c t _ { p } ^ { r e c } }$，Contactrec，Contactre-s)。精度衡量预测的正例中有多少是真正的正例。高精度意味着当模型预测为正时，它很可能是正确的。召回率表示模型正确识别了多少实际的正例。高召回率意味着模型成功识别了大多数的正例。然而，实现高精度与高召回率的平衡是一项重大挑战。因此，我们引入F-Score $= 2 \times \mathbf { p } \times \mathbf { r } / ( \mathbf { p } + \mathbf { r } )$ 作为衡量这两者平衡的指标。我们采用这些指标来评估重构的3D人类和物体网格，特别是在接触方面。我们通过对人类顶点与物体网格的距离在5cm以内进行分类来获得接触图，参考文献 [37]。然后，我们测量人类接触图与真实标定之间的精度、召回率和F-Score。

Table 1. Quantitative comparison of 3D human and object reconstruction with state-of-the-art methods on BEHAVE [2] and InterCap [23]. Our ScoreHOI model demonstrates superior accuracy, particularly in reconstructing contact interactions.   

<table><tr><td>Datasets</td><td>Methods</td><td>CDhuman↓</td><td>CDobject</td><td>Contactec↑</td><td>Contactrec↑</td><td>Contactres↑</td></tr><tr><td rowspan="4">BEHAVE</td><td>PHOSA [68]</td><td>12.17</td><td>26.62</td><td>0.393</td><td>0.266</td><td>0.317</td></tr><tr><td>CHORE [ [59]</td><td>5.58</td><td>10.66</td><td>0.587</td><td>0.472</td><td>0.523</td></tr><tr><td>CONTHO [37]</td><td>4.99</td><td>8.42</td><td>0.628</td><td>0.496</td><td>0.554</td></tr><tr><td>Ours</td><td>4.85</td><td>7.86</td><td>0.634</td><td>0.586</td><td>0.609</td></tr><tr><td rowspan="4">InterCap</td><td>PHOSA [68]</td><td>11.20</td><td>20.57</td><td>0.228</td><td>0.159</td><td>0.187</td></tr><tr><td>CHORE [59]</td><td>7.01</td><td>12.81</td><td>0.339</td><td>0.253</td><td>0.290</td></tr><tr><td>CONTHO [37]</td><td>5.96</td><td>9.50</td><td>0.661</td><td>0.432</td><td>0.522</td></tr><tr><td>Ours</td><td>5.56</td><td>8.75</td><td>0.627</td><td>0.590</td><td>0.578</td></tr></table>

Table 2. Quantitative comparison of optimization efficiency. While achieving superior performance, our ScoreHOI maintains a higher inference efficiency compared with previous methods.   

<table><tr><td>Methods</td><td>CDhuman</td><td>CDobject↓</td><td>FPS↑</td></tr><tr><td>CHORE [59]</td><td>5.58</td><td>10.66</td><td>0.0035</td></tr><tr><td>VisTracker [61]</td><td>5.24</td><td>7.89</td><td>0.0359</td></tr><tr><td>Ours-Faster</td><td>4.87</td><td>7.95</td><td>2.0080</td></tr><tr><td>Ours</td><td>4.85</td><td>7.86</td><td>0.2895</td></tr></table>

# 4.2. 与最先进方法的比较

性能比较。我们与之前的最先进方法进行比较，采用两个实验协议，BEHAVE [2] 和 InterCap [23]。定量结果记录在表1中。我们的方法在大多数指标上实现了最佳性能。为了协调接触的精确度和召回率，并且目标是最大化这两个值，我们引入接触F评分作为评估接触重建质量的指标。由于全面的物理约束，我们在接触F评分上超过了之前的最先进方法$9\%$。此外，优化结果可视化于图4中，展示了我们的ScoreHOI在重建接触关系时具有更高的合理性和物理真实性。值得注意的是，仅凭单个单目图像，侧面视图中常常出现重建不准确的问题。我们的ScoreHOI有效地通过利用生成模型中的先验知识来缓解这一病态问题。 效率比较。为了评估各种优化方法的推理效率，我们在单个Nvidia RTX 4090 GPU上测量每秒帧数（FPS）。我们的ScoreHOI与两种已建立的方法CHORE [59]和VisTracker [61]进行基准测试，这两种方法分别使用Adam优化器重建图像和视频输入。评估结果如表2所示，表明我们的方法的效率超过了对比方法的一个到两个数量级，同时提供了更优的性能。此外，我们评估了一个更高效的框架，$N=2$，如表4第4行中详细说明的。该配置在几乎不妥协性能的情况下实现了七倍的效率提升。

Table 3. Ablation studies of modules, conditions and guidance. The CDIR refers to the contact-driven iterative refinement. All of the modules, conditions and guidance strategies we design contribute to the improvement of our model.   

<table><tr><td>Methods</td><td>CDhuman</td><td>Cobect Conacrec↑ Contcec↑ Contctrec ↑</td><td></td><td></td><td></td></tr><tr><td>* Module</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/o diffusion</td><td>5.03</td><td>8.48</td><td>0.612</td><td>0.523</td><td>0.588</td></tr><tr><td>w/o CDIR</td><td>4.93</td><td>7.98</td><td>0.628</td><td>0.545</td><td>0.577</td></tr><tr><td>* Condition</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>No condition</td><td>4.94</td><td>8.23</td><td>0.626</td><td>0.549</td><td>0.585</td></tr><tr><td>w/o cG</td><td>4.87</td><td>7.99</td><td>0.628</td><td>0.559</td><td>0.591</td></tr><tr><td>/ </td><td>4.88</td><td>8.03</td><td>0.631</td><td>0.566</td><td>0.597</td></tr><tr><td>* Guidance</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>No guidance</td><td>4.93</td><td>8.01</td><td>0.624</td><td>0.524</td><td>0.570</td></tr><tr><td>w/o Lho</td><td>4.87</td><td>7.95</td><td>0.632</td><td>0.525</td><td>0.574</td></tr><tr><td>w/o Lpt</td><td>4.87</td><td>7.93</td><td>0.619</td><td>0.567</td><td>0.592</td></tr><tr><td>w/o Lof</td><td>4.89</td><td>7.95</td><td>0.631</td><td>0.577</td><td>0.602</td></tr><tr><td>Full model</td><td>4.85</td><td>7.86</td><td>0.634</td><td>0.586</td><td>0.609</td></tr></table>



我们在 BEHAVE [2] 测试集上进行所有消融研究，作为公平比较的标准基准。优化模块的有效性。我们评估扩散采样循环和提出的基于接触的迭代精细化方法的影响，如表 3 上半部分所示。利用具有能力感知的回归器，我们的主干网络能够独立于扩散模型实现令人满意的结果。然而，仍然存在提升的潜力，特别是在接触召回方面。此外，我们还进行了一个不包含 CDIR 的消融研究，其中在采样循环期间不更新接触图。结果表明，CDIR 有助于提高重建精度和接触质量的改善。条件的有效性。如中央部分所示。

![](images/5.jpg)  
Figure 5. Qualitative results for ablation study. Upper row: the ablation study of $L _ { \mathrm { { h o } } }$ . The inclusion of contact guidance between the $L _ { \mathrm { h o } }$ . The absence of a penetration penalty results in a notable rise in the occurrence of unreasonable interactions.

Table 4. Ablation studies of optimization hyper-parameters. We evaluate the performance across various optimization hyperparameter configurations. The optimal results measured by chamfer-distance are selected to establish our baseline.   

<table><tr><td>N</td><td>τ</td><td>∆t</td><td>CDhuman</td><td>CDobject</td><td>Contactre ↑</td></tr><tr><td>10</td><td>50</td><td>25</td><td>5.80</td><td>10.70</td><td>0.539</td></tr><tr><td>10</td><td>50</td><td>10</td><td>5.22</td><td>8.72</td><td>0.584</td></tr><tr><td>10</td><td>50</td><td>5</td><td>4.99</td><td>8.16</td><td>0.569</td></tr><tr><td>2</td><td>50</td><td>2</td><td>4.87</td><td>7.95</td><td>0.604</td></tr><tr><td>5</td><td>50</td><td>2</td><td>4.86</td><td>7.87</td><td>0.607</td></tr><tr><td>20</td><td>50</td><td>2</td><td>4.87</td><td>7.89</td><td>0.610</td></tr><tr><td>10</td><td>25</td><td>2</td><td>4.87</td><td>7.95</td><td>0.606</td></tr><tr><td>10</td><td>100</td><td>2</td><td>4.88</td><td>8.07</td><td>0.615</td></tr><tr><td>10</td><td>50</td><td>2</td><td>4.85</td><td>7.86</td><td>0.609</td></tr></table>

优化超参数分析。我们进行了一项消融分析，以研究优化超参数的影响，包括迭代次数 $N$、中间噪声水平 $\tau$ 和 DDIM 步长 $\Delta t$，如表 4 所示。我们的研究结果表明，$N$ 的增加导致接触 F-Score 的提高，这归因于更多的细化步骤增强了接触的可信度。在 $\tau$ 上也观察到类似的趋势，即较高的值对应于改进的接触关系。关于 $\Delta t$，更多的采样步骤被观察到可以产生显著更好的结果。综上所述，我们的结果揭示了重建质量与交互可信度之间的平衡，最终选择 $N = 10, \tau = 50, \Delta t = 2$ 作为最佳基准参数组合。

# 5. 结论与未来工作 表3的结论显示，若从扩散模型中排除图像引导或几何引导，重建性能会降低。相比于第一行中排除扩散模块的设置，我们的接触驱动迭代优化方法即使在缺乏额外条件的情况下，也展现了提高性能的能力。

物理指导的有效性。我们研究了在DDIM采样过程中不同物理约束的影响，如表3底部所示。实验结果表明，纳入三项不同目标可以增强重建精度和接触性能。省略 $L _ { \mathrm { { h o } } }$ 会导致接触图回忆率显著下降，从而减少输出中接触交互的发生。相反，缺少 $L _ { \mathrm { p t } }$ 会导致接触精度下降，这归因于穿透增多和无关部分的冗余接触。为了证实我们的发现，定性结果也在图5中展示。在本文中，我们介绍了ScoreHOI，这是一个创新且有效的框架，旨在实现物理上合理的人机交互重建。它整合了基于扩散的先验知识以进行分数导向采样精细化，并结合物理约束以增强现实感。我们还提出了一种基于接触的迭代精细化算法，以改善接触模式的恢复。广泛的评估表明，其重建精度优越，物理合理性增强，推理速度快于基于优化的方法。然而，我们承认我们的模型在泛化能力上存在局限性，因为训练数据集中已知对象的预定义典型姿势使得无法优化具不确定典型姿势的对象参数。我们未来的研究将集中在解决无模板未见对象的问题上。我们期待我们的工作能为人机交互的联合重建领域贡献一种新颖视角，促进该领域的进一步研究和发展。

# 6. 致谢

本研究得到了广东省杰出青年学者自然科学基金（编号：2025B1515020012）和深圳市科技计划（编号：JCYJ20240813111903006）的支持。

# References

[1] Mykhaylo Andriluka, Leonid Pishchulin, Peter Gehler, and Bernt Schiele. 2D human pose estimation: New benchmark and state of the art analysis. In CVPR, 2014. 2   
[2] Bharat Lal Bhatnagar, Xianghui Xie, Ilya A Petrov, Cristian Sminchisescu, Christian Theobalt, and Gerard Pons-Moll. BEHAVE: Dataset and method for tracking human object interactions. In CVPR, 2022. 2, 5, 6, 7   
[3] Samarth Brahmbhatt, Ankur Handa, James Hays, and Dieter Fox. Contactgrasp: Functional multi-finger grasp synthesis from contact. In IROS, 2019. 2   
[4] Samarth Brahmbhatt, Chengcheng Tang, Christopher D Twigg, Charles C Kemp, and James Hays. Contactpose: A dataset of grasps with object contact and hand pose. In ECCV, 2020. 2   
[5] Zhe Cao, Hang Gao, Karttikeya Mangalam, Qi-Zhi Cai, Minh Vo, and Jitendra Malik. Long-term human motion prediction with scene context. In ECCV, 2020. 2   
[6] Yixin Chen, Siyuan Huang, Tao Yuan, Siyuan Qi, Yixin Zhu, and Song-Chun Zhu. Holistic $^ { + + }$ scene understanding: Single-view 3d holistic scene parsing and human pose estimation with human-object interaction and physical commonsense. In ICCV, 2019. 2   
[7] Hyungjin Chung, Jeongsol Kim, Michael T Mccann, Marc L Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. ICLR, 2023. 3, 4   
[8] Hyungjin Chung, Byeongsu Sim, Dohoon Ryu, and Jong Chul Ye. Improving diffusion models for inverse problems using manifold constraints. NeurIPS, 2022. 3   
[9] Enric Corona, Albert Pumarola, Guillem Alenya, Francesc Moreno-Noguer, and Grégory Rogez. Ganhand: Predicting human grasp affordances in multi-object scenes. In CVPR, 2020. 2   
[10] Wenxun Dai, Ling-Hao Chen, Jingbo Wang, Jinpeng Liu, Bo Dai, and Yansong Tang. Motionlcm: Real-time controllable motion generation via latent consistency model. In ECCV, 2024. 2   
[11] Kiana Ehsani, Shubham Tulsiani, Saurabh Gupta, Ali u to predict physical forces by simulating effects. In CVPR, 2020.2   
[12] Mihai Fieraru, Mihai Zanfir, Elisabeta Oneata, Alin-Ionut Popa, Vlad Olaru, and Cristian Sminchisescu. Threedimensional reconstruction of human interactions. In CVPR, 2020. 2   
[13] Mihai Fieraru, Mihai Zanfir, Elisabeta Oneata, Alin-Ionut Popa, Vad Olaru, and Cristian Sminchisescu. Learning complex 3d human self-contact. In AAAI, 2021. 2   
[14] Lin Geng Foo, Jia Gong, Hossein Rahmani, and Jun Liu. Distribution-aligned diffusion for human mesh recovery. In ICCV, 2023. 2   
[15] Shubham Goel, Georgios Pavlakos, Jathushan Rajasegaran, Reconstructing and tracking humans with transformers. In ICCV, 2023. 1   
[16] Mohamed Hassan, Vasileios Choutas, Dimitrios Tzionas, and Michael J Black. Resolving 3d human pose ambiguities with 3d scene constraints. In ICCV, 2019. 2   
[17] Mohamed Hassan, Partha Ghosh, Joachim Tesch, Dimitrios Tzionas, and Michael J Black. Populating 3d scenes by learning human-scene interaction. In CVPR, 2021. 2   
[18] Yana Hasson, Gul Varol, Dimitrios Tzionas, Igor Kalevatykh, Michael J Black, Ivan Laptev, and Cordelia Schmid. Learning joint reconstruction of hands and manipulated objects. In CVPR, 2019. 2   
[19] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.5   
[20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 2020. 2, 3   
[21] Chun-Hao P Huang, Hongwei Yi, Markus Höschle, Matvey Safroshkin, Tsvetelina Alexiadis, Senya Polikovsky, Daniel Scharstein, and Michael J Black. Capturing and inferring dense full-body human-scene contact. In CVPR, 2022. 2   
[22] Siyuan Huang, Zan Wang, Puhao Li, Baoxiong Jia, Tengyu Liu, Yixin Zhu, Wei Liang, and Song-Chun Zhu. Diffusionbased generation, optimization, and planning in 3d scenes. In CVPR, 2023. 4   
[23] Yinghao Huang, Omid Taheri, Michael J Black, and Dimitrios Tzionas. InterCap: Joint markerless 3D tracking of humans and objects in interaction. In GCPR, 2022. 2, 5, 6, 7   
[24] Yuheng Jiang, Suyi Jiang, Guoxing Sun, Zhuo Su, Kaiwen Guo, Minye Wu, Jingyi Yu, and Lan Xu. Neuralhofusion: Neural volumetric rendering under human-object interactions. In CVPR, 2022. 2   
[25] Angjoo Kanazawa, Michael J Black, David W Jacobs, and Jitendra Malik. End-to-end recovery of human shape and pose. In CVPR, 2018. 1   
[26] Korrawe Karunratanakul, Jinlong Yang, Yan Zhang, Michael JBlack, Krikamol Muandet, and Sy Tang. Grasping field: Learning implicit representations for human grasps. In 3DV, 2020. 2   
[27] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2014. 2, 6   
[28] Nikos Kolotouros, Georgios Pavlakos, Michael J Black, and Kostas Daniilidis. Learning to reconstruct 3d human pose and shape via model-fitting in the loop. In ICCV, 2019.1   
[29] Nikos Kolotouros, Georgios Pavlakos, Dinesh Jayaraman, and Kostas Daniilidis. Probabilistic modeling for human mesh recovery. In ICCV, 2021.   
[30] Jiaman Li, Alexander Clegg, Roozbeh Mottaghi, Jiajun Wu, Xavier Puig, and C Karen Liu. Controllable human-object interaction synthesis. In ECCV, 2025. 4   
[31] Zongmian Li, Jiri Sedlar, Justin Carpentier, Ivan Laptev, Nicolas Mansard, and Josef Sivic. Estimating 3d motion and forces of person-object interactions from monocular video. In CVPR, 2019. 2   
[32] Jinpeng Liu, Wenxun Dai, Chunyu Wang, Yiji Cheng, Yansong Tang, and Xin Tong. Plan, posture and go: Towards open-vocabulary text-to-motion generation. In ECCV, 2024. 2   
[33] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard   
1 Ul-l, alu il J D. il . A lu - person linear model. ACM TOG, 2015. 3 [34] Junzhe Lu, Jing Lin, Hongkun Dou, Ailing Zeng, Yue Deng, Yulun Zhang, and Haoqian Wang. Dposer: Diffusion model as robust 3d human pose prior, 2024. 2 [35] Aron Monszpart, Paul Guerrero, Duygu Ceylan, Ersin Yumer, and Niloy J Mitra. imapper: interaction-guided scene mapping from monocular videos. TOG, 2019. 2 [36] Lea Muller, Ahmed AA Osman, Siyu Tang, Chun-Hao P Huang, and Michael J Black. On self-contact and human pose. In CVPR, 2021. 2 [37] Hyeongjin Nam, Daniel Sungho Jung, Gyeongsik Moon, and Kyoung Mu Lee. Joint reconstruction of 3d human and object via contact-based refinement transformer. In CVPR,   
2024. 2, 4, 5, 6, 7, 1 [38] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. NeurIPS, 2019. 5 [39] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed AA Osman, Dimitrios Tzionas, and Michael J Black. Expressive body capture: 3d hands, face, and body from a single image. In CVPR, 2019. 2, 3 [40] Xiaogang Peng, Yiming Xie, Zizhao Wu, Varun Jampani, Deqing Sun, and Huaizu Jiang. Hoi-diff: Text-driven synthesis of 3d human-object interactions using diffusion models. arXiv preprint arXiv:2312.06553, 2023. 4 [41] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. NeurIPS, 2017. 1 [42] Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai, Hasan Hammoud, Mohamed Elhoseiny, and Bernard Ghanem. Pointnext: Revisiting pointnet $^ { + + }$ with improved training and scaling strategies. In NeurIPS, 2022. 5, 1 [43] Davis Rempe, Leonidas J Guibas, Aaron Hertzmann, Bryan Russell, Ruben Villegas, and Jimei Yang. Contact and human dynamics from monocular video. In ECCV, 2020. 2 [44] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2 [45] Javier Romero, Dimitris Tzionas, and Michael J Black. Embodied hands: Modeling and capturing hands and bodies together. TOG, 2017. 3 [46] Manolis Savva, Angel X Chang, Pat Hanrahan, Matthew Fisher, and Matthias NieBner. Pigraphs: learning interaction snapshots from observations. TOG, 2016. 2 [47] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. ICLR, 2021. 2, 3 [48] Jiaming Song, Arash Vahdat, Morteza Mardani, and Jan Kautz. Pseudoinverse-guided diffusion models for inverse problems. In ICLR, 2023. 3 [49] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. ICLR, 2021. 2, 3 [50] Anastasis Stathopoulos, Ligong Han, and Dimitris Metaxas. Score-guided diffusion for 3d human recovery. In CVPR,   
2024. 2, 3, 4, 5 [51] Guoxing Sun, Xin Chen, Yizhang Chen, Anqi Pang, Pei Lin, Yuheng Jiang, Lan Xu, Jingyi Yu, and Jingya Wang. Neural free-viewpoint performance rendering under complex human-object interactions. In ACM MM, 2021. 2   
[52] Omid Taheri, Nima Ghorbani, Michael J. Black, and Dimitrios Tzionas. GRAB: A dataset of whole-body human grasping of objects. In ECCV, 2020. 2   
[53] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 5   
[54] Yiqin Wang, Haoji Zhang, Yansong Tang, Yong Liu, Jiashi Feng, Jifeng Dai, and Xiaojie Jin. Hierarchical memory for long video qa. arXiv preprint arXiv:2407.00603, 2024. 2   
[55] Zhenzhen Weng and Serena Yeung. Holistic 3d human and scene mesh estimation from single view images. In CVPR, 2021.2   
[56] Zhirong Wu, Shuran Song, Aditya Khosla, Fisher Yu, Linguang Zhang, Xiaoou Tang, and Jianxiong Xiao. 3d shapenets: A deep representation for volumetric shapes. In CVPR, 2015. 1   
[57] Xianghui Xie, Bharat Lal Bhatnagar, Jan Eric Lenssen, and Gerard Pons-Moll. Template free reconstruction of humanobject interaction with procedural interaction generation. In CVPR, 2024. 2   
[58] Xianghui Xie, Bharat Lal Bhatnagar, Jan Eric Lenssen, and Gerard Pons-Moll. Template free reconstruction of humanobject interaction with procedural interaction generation. In CVPR, 2024. 2, 1   
[59] Xianghui Xie, Bharat Lal Bhatnagar, and Gerard Pons-Moll. CHORE: Contact, human and object reconstruction from a single RGB image. In ECCV, 2022. 2, 4, 6, 7   
[60] Xianghui Xie, Bharat Lal Bhatnagar, and Gerard Pons-Moll. Visibility aware human-object interaction tracking from single RGB camera. In CVPR, 2023. 2, 4   
[61] Xianghui Xie, Bharat Lal Bhatnagar, and Gerard Pons-Moll. Visibility aware human-object interaction tracking from single rgb camera. In CVPR, 2023. 7   
[62] Xianghui Xie, Jan Eric Lenssen, and Gerard Pons-Moll. Intertrack: Tracking human object interaction without object templates. 2024. 1   
[63] Yiming Xie, Varun Jampani, Lei Zhong, Deqing Sun, and Huaizu Jiang. Omnicontrol: Control any joint at any time for human motion generation. In ICLR, 2024. 2   
[64] Yufei Xu, Jing Zhang, Qiming Zhang, and Dacheng Tao. ViTPose: Simple vision transformer baselines for human pose estimation. In NeurIPS, 2022. 5   
[65] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models, 2023. 5   
[66] Hongwei Yi, Chun-Hao P Huang, Dimitrios Tzionas, Muhammed Kocabas, Mohamed Hassan, Siyu Tang, Justus Thies, and Michael J Black. Human-aware object placement for visual environment reconstruction. In CVPR, 2022. 2   
[67] Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi Feng, and Xiaojie Jin. Flash-vstream: Efficient realtime understanding for long video streams. arXiv preprint arXiv:2506.23825, 2025. 2   
[68] Jason Y Zhang, Sam Pepose, Hanbyul Joo, Deva Ramanan, Jitendra Malik, and Angjoo Kanazawa. Perceiving 3D human-obiect spatial arrangements from a single image in the wild. In ECCV, 2020. 1, 6, 7   
[69] Jason Y. Zhang, Sam Pepose, Hanbyul Joo, Deva Ramanan, Jitendra Malik, and Angjoo Kanazawa. Perceiving 3d human-object spatial arrangements from a single image in the wild. In ECCV, 2020. 2   
[70] Siwei Zhang, Yan Zhang, Qianli Ma, Michael J Black, and Siyu Tang. Place: Proximity learning of articulation and contact in 3d environments. In 3DV, 2020. 2   
[71] Xiaohan Zhang, Bharat Lal Bhatnagar, Sebastian Starke, Vladimir Guzov, and Gerard Pons-Moll. Couch: Towards controllable human-chair interactions. In ECCV, 2022. 2   
[72] Chengfeng Zhao, Juze Zhang, Jiashen Du, Ziwei Shan, Junye Wang, Jingyi Yu, Jingya Wang, and Lan Xu. I'm hoi: Inertia-aware monocular capture of 3d human-object interactions. In CVPR, 2024. 5, 6   
[73] Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. On the continuity of rotation representations in neural networks. In CVPR, 2019. 4   
[74] Yixuan Zhu, Ao Li, Yansong Tang, Wenliang Zhao, Jie Zhou, and Jiwen Lu. Dpmesh: Exploiting diffusion prior for occluded human mesh recovery. In CVPR, 2024. 1, 2   
[75] Yixuan Zhu, Haolin Wang, Ao Li, Wenliang Zhao, Yansong Tang, Jingxuan Niu, Lei Chen, Jie Zhou, and Jiwen Lu. Instarevive: One-step image enhancement via dynamic score matching. In ICLR, 2025. 3   
[76] Yixuan Zhu, Wenliang Zhao, Ao Li, Yansong Tang, Jie Zhou, and Jiwen Lu. Flowie: Efficient image enhancement via rectified flow. In CVPR, 2024. 3

# ScoreHOI: Physically Plausible Reconstruction of Human-Object Interaction via Score-Guided Diffusion

Supplementary Material

In this appendix, we describe the DDIM sampling loop in our methods in pseudo-code in Section A. We also provide additional detailed implementations and qualitative comparisons in Section B and Section D. Furthermore, we append rendered 3D object models and the source code along with this appendix.

# A. DDIM Refinement Loop

For a deeper understanding of the DDIM sampling loop, we illustrate a pseudo-code implementation in Algorithm A. The algorithm describe the DDIM Loop in detail.

# B. Detailed Implementations

We employ the PointNeXt [42] model, pre-trained on the ModelNet40 [56] dataset, as our affordance-aware regressor. ModelNet40 is a extensively utilized benchmark dataset for 3D shape classification and retrieval tasks, comprising 12,311 CAD models across 40 distinct object categories, including airplanes, chairs, tables, and cars. PointNeXt maintains the fundamental hierarchical architecture of PointNet $^ { - + }$ [41] while integrating contemporary deep learning techniques to augment performance and efficiency. Through the incorporation of point cloud awareness, the model is capable of comprehending object geometry across a variety of human-object interaction scenarios. For the contact predictor and mesh regressor, we leverage the contact estimation transformer and the contact-based refinement methodology proposed by [37]. The contact predictor receives human and object feature tokens, along with estimated human and object mesh vertices, as inputs to generate contact masks through two symmetrical 4-layer transformer blocks. During our contact-driven iterative refinement process, we iteratively update the human and object meshes to enhance contact interactions, thereby refining the contact prediction results with each iteration. After $N$ iterations of optimization, we obtain the $\pmb { x } _ { 0 } ^ { N }$ for final refinement. The mesh regressor, also constructed with two analogous 4-layer transformer branches, takes the updated human and object feature tokens and contact masks as input to further refine the mesh results.

# C. Ablation Studies

About Diffusion Priors. We demonstrate the generation results starting from different noise levels without physical guidance, as shown in the Figure A. The IG-Adapter successfully constructs plausible human-object interaction patterns, confirming that the diffusion model has acquired a valid prior distribution.

Table A. Efficiency comparision with template-free methods.   

<table><tr><td></td><td>HDM</td><td>InterTrack</td><td>ScoreHOI</td><td>ScoreHOI-F</td></tr><tr><td>FPS↑</td><td>0.0047</td><td>0.0012</td><td>0.2895</td><td>2.0080</td></tr></table>

![](images/6.jpg)  
Input Image τ = T τ = 0.5T τ = 0.25 τ = 0.05 Input Image τ = T τ = 0.5 τ = 0.25 τ = 0.05T Figure A. Generation results starting from different noise levels.

Comparison with template-free methods. We compared ScoreHOI with template-free methods, including HDM [58] and InterTrack [62]. As shown in the Table A, ScoreHOI surpasses these methods in efficiency. Additionally, these methods output point clouds, which are less practical for applications like robotic data collection due to the additional time and uncertainty introduced when fitting SMPL parameters.

# D. More Qualitative Results

To further substantiate the performance of our ScoreHOI, we present additional qualitative results of human-object interaction reconstruction in Figure B. Our methodology exhibits superior performance across diverse interaction patterns, such as sitting, carrying, grasping, and lifting. By incorporating physical constraints, our ScoreHOI achieves a higher degree of accuracy and physical fidelity in the reconstruction of human and object meshes.

# E. Geometry Model Demos

The demo 3D geometry mesh results are available under the demo directory. Reviewers are able to assess the performance from any perspective utilizing software such as MeshLab or Blender.

# Algorithm A Score-Guided DDIM refinement loop

1: Input: latent parameters $\pmb { x } _ { 0 } ^ { n }$ at step $n$ , denoising model $\epsilon _ { \phi }$ , image features $c _ { \mathrm { I } }$ , geometry features $\mathbf { c _ { G } }$ , gradient step size $\rho$ noise level $\tau$ , DDIM step size $\Delta t$ , estimated caontact masks $\{ \mathbf { M } _ { i } \} _ { i \in \{ \mathrm { h , o , f } \} }$ Output: latent parameters $\pmb { x } _ { 0 } ^ { n + 1 }$ for next sampling step $n + 1$ 3: $\pmb { x } _ { \tau } = \mathrm { D D I M I n v e r t } ( \pmb { x } _ { 0 } ^ { n } , \pmb { c } _ { \mathrm { I } } , \pmb { c } _ { \mathrm { G } } )$ Run DDIM inversion until noise level $\tau$ 4: for $t = \tau$ to $\Delta t$ with step size $\Delta t$ do 5: $\tilde { \epsilon } \gets \epsilon _ { \phi } ( \mathbf { { x } } _ { t } ^ { n } , t , \mathbf { { c } } _ { \mathrm { { I } } } , { { c } } _ { \mathrm { { G } } } )$ Predict noise 6: Initialize computational graph for $\mathbf { \Delta } \mathbf { x } _ { t } ^ { n }$ 7: $\begin{array} { r l } & { \dot { x ^ { n } } _ { 0 } \gets \frac { 1 } { \sqrt { \alpha _ { t } } } \big ( \dot { x } _ { t } ^ { n } - \sqrt { 1 - \alpha _ { t } } \big ) \tilde { \epsilon } } \\ & { L _ { \mathcal { P } } \gets \mathrm { P h y s i c a l G u i d a n c e } ( \hat { x } _ { 0 } ^ { n } , \{ \mathbf { M } _ { i } \} _ { i \in \{ \mathrm { h , o , f } \} } \big ) } \\ & { \tilde { \epsilon } ^ { ' } \gets \tilde { \epsilon } + \rho \sqrt { 1 - \alpha _ { t } } \nabla _ { x _ { t } ^ { n } } L _ { \mathcal { P } } } \\ & { \dot { x ^ { n } } _ { 0 } ^ { ' } \gets \frac { 1 } { \sqrt { \alpha _ { t } } } ( x _ { t } ^ { n } - \sqrt { 1 - \alpha _ { t } } ) \tilde { \epsilon } ^ { ' } } \\ & { x _ { t - \Delta t } ^ { n } \gets \sqrt { \alpha _ { t - \Delta t } } \hat { x } _ { 0 } ^ { n ^ { ' } } + \sqrt { 1 - \alpha _ { t - \Delta t } } \tilde { \epsilon } ^ { ' } } \end{array}$ $\triangleright$ Predict one-step denoised result 8: $\triangleright$ Compute physical guidance loss 9: Compute modified noise after score-guidance 10: Predict one-step denoised result with modified noise 11: . DDIM sampling step 12:end for 13 xn+1 $\pmb { x } _ { 0 } ^ { n + 1 }  \hat { \pmb { x } } _ { 0 } ^ { n ^ { \prime } }$ Update $\pmb { x } _ { 0 } ^ { n }$ for next generation 14: return $\pmb { x } _ { 0 } ^ { n + 1 }$

![](images/7.jpg)  
Figure B. Extra Qualitative comparisons. We highlight the contact interaction within each picture.