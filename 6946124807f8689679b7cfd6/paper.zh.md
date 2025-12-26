# 基于 MASt3R 的 3D 图像匹配基础

文森特·勒鲁瓦 约翰·卡丰 杰罗姆·雷沃 NAVER LABS 欧洲 https://github.com/naver/mast3r

![](images/1.jpg)  

Figure 1: Dense Correspondences. MASt3R extends DUSt3R as it predicts dense correspondences, even in regions where camera motion significantly degrades the visual similarity. Focal length can be derived from the predic 3geymakiu pr  adalonmethocmcaliratin, ce pos t an 3 heah challenging benchmarks.

# 摘要

图像匹配是所有表现最佳的三维视觉算法和流程的核心组成部分。尽管匹配从根本上是一个与相机姿态和场景几何 intrinsically 相关的三维问题，但通常将其视为二维问题。这是有道理的，因为匹配的目标是建立二维像素场之间的对应关系，但这似乎也是一个潜在的危险选择。在本研究中，我们采取不同的立场，建议将匹配视为一个三维任务，使用基于 Transformers 的最新强大三维重建框架 DUSt3R。基于点图回归，这种方法在匹配极端视角变化的视图时表现出令人印象深刻的鲁棒性，但准确性有限。我们在此旨在提高这种方法的匹配能力，同时保持其鲁棒性。因此，我们建议通过增加一个新的头部来增强 DUSt3R 网络，该头部输出密集的局部特征，并通过额外的匹配损失进行训练。我们进一步解决了密集匹配的二次复杂性问题，如果不加以仔细处理，对于下游应用而言，速度会变得极慢。我们引入了一种快速的双向匹配方案，这不仅将匹配速度提高了几个数量级，而且具有理论保证，并最终带来了更好的结果。大量实验表明，我们的方法 MASt3R 在多个匹配任务上显著超越了当前最先进的技术。特别是在极具挑战性的无地图定位数据集上，它在 VCRE AUC 指标上比最佳发表方法提高了 $30\%$（绝对提升）。

# 1. 引言

能够在同一场景的不同图像之间建立像素之间的对应关系，称为图像匹配，是所有三维视觉应用的核心组成部分，涵盖了映射、定位、导航、摄影测量以及普遍的自主机器人技术。最先进的视觉定位方法，例如，在离线映射阶段，通常依赖于图像匹配，例如使用COLMAP，以及在在线定位步骤中，通常使用PnP。在本文中，我们专注于这一核心任务，旨在给定两幅图像生成一份配对对应关系列表，称为匹配。特别地，我们希望输出高度准确且密集的匹配，这些匹配对视点和光照变化具有鲁棒性，因为这些最终是现实世界应用的限制因素。过去，匹配方法传统上被分为三个步骤：首先提取稀疏且可重复的关键点，然后用局部不变特征对其进行描述，最后通过比较特征空间中的距离来配对离散的关键点集合。该流程具有若干优点：在低至中等照明和视点变化下，关键点检测器较为精确，而关键点的稀疏性使得该问题在计算上易于处理，从而能够在毫秒内实现非常精确的匹配，只要图像在类似条件下被查看。这也解释了SIFT在COLMAP等三维重建流程中成功和持久的原因。

不幸的是，基于关键点的方法通过将匹配简化为一个关键点包的问题，忽略了对应任务的全局几何上下文。这使得它们在重复模式或低纹理区域的情况下尤其容易出错，这实际上对于局部描述符来说是个病态问题。一种解决方法是在配对步骤中引入全局优化策略，通常利用一些关于匹配的学习先验，这一点在SuperGlue等类似方法中得到了成功实施。然而，如果关键点及其描述符未能编码足够的信息，在匹配过程中利用全局上下文可能已经为时已晚。因此，另一个方向是考虑密集整体匹配，即完全避免关键点，直接一次性匹配整个图像。这在全球注意力机制出现后成为可能。这样的方式，如LoFTR，因此将图像视为一个整体，产生的对应集是密集的，并且对重复模式和低纹理区域更具鲁棒性。这在一些最具挑战性的基准上产生了新的最先进结果，例如无地图定位基准。

然而，即使是像 LoFTR [82] 这样表现优异的方法，在无地图定位基准测试中也仅取得了相对令人失望的 VCRE 精度 $3 4 \%$。我们认为，这是因为到目前为止，几乎所有的匹配方法都将匹配视为图像空间中的二维问题。实际上，匹配任务的表述本质上是一个三维问题：对应的像素是观察同一个三维点的像素。事实上，二维像素对应关系与三维空间中的相对相机姿态是同一枚硬币的两面，因为它们通过极线矩阵 [36] 直接相关。另一个证据是，当前在无地图基准上表现最优的方法是 DUSt3R [102]，该方法最初是为了三维重建而设计，而匹配只是三维重建的副产品。然而，从这一三维输出中天真地获得的对应关系目前在无地图基准上超越了所有其他基于关键点和匹配的方法。在本文中，我们指出，尽管 DUSt3R [102] 确实可以用于匹配，但其相对不精确，尽管对视角变化非常鲁棒。为了解决这个缺陷，我们提出附加一个第二个头部，以回归稠密局部特征图，并使用 InfoNCE 损失进行训练。所得到的架构 MASt3R，即“匹配和立体三维重建”，在多个基准测试中超越了 DUSt3R。为了获得像素级准确的匹配，我们提出一个粗到精的匹配方案，在此过程中在多个尺度上进行匹配。每个匹配步骤涉及从稠密特征图中提取互相匹配的特征，这一点可能和直觉相反，其耗时远远超过计算稠密特征图本身。我们提出的解决方案是一种更快的互相匹配算法，其速度几乎快两个数量级，同时提高了姿态估计的质量。总之，我们提出三项主要贡献。首先，我们提出 MASt3R，一种基于最近发布的 DUSt3R 框架的三维感知匹配方法。它输出能够实现高度精确且极其鲁棒的匹配的局部特征图。其次，我们提出了与快速匹配算法相关联的粗到精的匹配方案，使其能够处理高分辨率图像。第三，MASt3R 在多个绝对和相对姿态定位基准上显著超越了最先进的技术。

# 相关工作

基于关键点的匹配一直是计算机视觉的重要基石。匹配过程分为三个不同的阶段：关键点检测、局部不变描述和描述符空间中的最近邻搜索。与早期的手工制作方法如 SIFT 不同，现代方法正逐渐转向基于学习的数据驱动方案，以检测关键点、描述它们，或同时进行这两项工作。总体而言，基于关键点的方法在许多基准测试中占据主导地位，突显了其在需要高精度和速度的任务中的持久价值。然而，一个显著的问题是，它们将匹配简化为局部问题，即忽视了其整体性。因此，SuperGlue 和类似的方法提出在最后的配对步骤中利用更强的先验进行全局推理，以指导匹配，但仍然将检测和描述局部化。尽管取得了一定成功，但仍然受到关键点局部性质的限制，并且无法在强视角变化下保持不变。 稠密匹配。与基于关键点的方法相比，半稠密和稠密方法提供了一种不同的图像对应建立范式，考虑所有可能的像素关联。这与光流方法非常相似，通常采用粗到细的方案以降低计算复杂性。总体而言，这些方法旨在从全局视角考虑匹配，但代价是增加了计算资源。稠密匹配在需要详细空间关系和纹理对理解场景几何至关重要的场景中证明了其有效性，在许多基准测试中表现出色，而这些测试对于关键点来说在视角或光照变化极端时尤其具有挑战性。这些方法仍将匹配视为一个二维问题，这限制了它们在视觉定位中的使用。 相机姿态估计技术差异很大，但最成功的策略在速度、准确性和鲁棒性之间找到平衡，基本上基于像素匹配。匹配方法的不断改进促使了更多具有挑战性的相机姿态估计基准的引入，如 Aachen Day-Night、InLoc、CO3D 或 Map-free，这些基准均具有强视角和/或光照变化。其中最具挑战性的是 Map-free，一个定位数据集，提供了一张单一的参考图像，但没有地图，视角变化可达 $1 8 0 ^ { \circ }$。

在这些具有挑战性的条件下，将三维基础匹配视为一种关键的必要性，而经典的基于二维的匹配则完全无法满足需求。利用对场景物理属性的先验知识以提高精度或鲁棒性在过去被广泛研究，但大多数之前的工作仅仅利用了极线约束进行半监督学习对应关系，而没有进行根本性的改变。Toft等人则提出通过使用来自现成的单目深度预测器获得的透视变换来校正图像，从而改善关键点描述符。最近，尽管严格来说并不是匹配方法，姿态扩散或光线扩散通过将三维几何约束纳入其姿态估计公式中展现了有希望的表现。最后，最近的DUSt3R探讨了从无校准图像进行三维重建这一$a$ -先验更困难任务中恢复对应关系的可能性。尽管该方法并未明确针对匹配进行训练，但其结果令人鼓舞，在无地图排行榜中名列前茅。我们的贡献是追求这一思路，通过回归局部特征并明确对其进行成对匹配的训练。

# 3. 方法

给定两个图像 $I ^ { 1 }$ 和 $I ^ { 2 }$，分别由两个参数未知的相机 $C ^ { 1 }$ 和 $C ^ { 2 }$ 捕获，我们希望恢复一组像素对应关系 $\{ ( i , j ) \}$，其中 $i , j$ 为像素，$i = ( u _ { i } , \nu _ { i } ) , j = ( u _ { j } , \nu _ { j } ) \in \{ 1 , \dots , W \} \times \{ 1 , \dots , H \}$，$W$ 和 $H$ 分别是图像的宽度和高度。为了简化起见，我们假设它们具有相同的分辨率，但并不影响一般性。最终的网络可以处理具有可变纵横比的一对图像。我们的方法，如图 2 所示，旨在针对两张输入图像联合执行 3D 场景重建和匹配。该方法基于 Wang 等人最近提出的 DUSt3R 框架 [102]，我们首先在第 3.1 节中回顾该框架，然后在第 3.2 节中提出我们所提议的匹配头及其相应的损失。随后，我们在第 3.3 节中介绍了一种专门设计的优化匹配方案，以处理密集特征图，并在第 3.4 节中将其用于粗到细的匹配。

# 3.1. DUSt3R框架

DUSt3R [102] 是一种新近提出的方法，它共同解决了仅通过图像进行标定和三维重建的问题。基于变换器的网络根据两幅输入图像预测局部三维重建，以两个稠密的三维点云 $X ^ { 1 , 1 }$ 和 $X ^ { 2 , 1 }$ 的形式表示，以下简称为点图。

![](images/2.jpg)  
Fiur  Overvi  the propos aproac.Given twoinput mages matc, ur network regresss image and eac input pixel a 3D point, a conidence value and a local feature. Plugging either 3D points or lol features into our fast reciprocal  matcer (3.3) yiels robust correspondences. Compared t the DUSt3R framework which we build upon, our contributions are highlighted in blue.

一个点映射 $X^{a,b} \in \mathbb{R}^{H \times W \times 3}$ 表示每个像素 $i = (u, \nu)$ 与图像 $I^{a}$ 之间的稠密 2D 到 3D 映射， $X_{u, \nu}^{a,b} \in \mathbb{R}^{3}$ 表达在相机 $C^{b}$ 的坐标系中。通过回归两个在相机 $C^{1}$ 的相同坐标系中表达的点映射 $X^{1,1}$ 和 $X^{2,1}$，DUSt3R 有效地解决了联合标定和 3D 重建问题。当提供多于两张图像时，第二步的全局对齐将所有点映射合并到相同的坐标系中。请注意，在本文中，我们不使用这一步，限制于双目情况。我们现在详细解释推理过程。两幅图像首先以 Siamese 方式编码，使用 ViT[23]，生成两个表示 $H^{1}$ 和 $H^{2}$：

$$
\begin{array} { r } { \boldsymbol { H } ^ { 1 } = \operatorname { E n c o d e r } ( \boldsymbol { I } ^ { 1 } ) , } \\ { \boldsymbol { H } ^ { 2 } = \operatorname { E n c o d e r } ( \boldsymbol { I } ^ { 2 } ) . } \end{array}
$$

然后，两个相互交织的解码器联合处理这些表征，通过交叉注意力交换信息，以“理解”不同视点之间的空间关系和场景的全球三维几何结构。这些增强了空间信息的新表征记作 $H ^ { 1 }$ 和 $H ^ { 2 }$：

$$
H ^ { \prime 1 } , H ^ { \prime 2 } = \operatorname { D e c o d e r } ( H ^ { 1 } , H ^ { 2 } ) .
$$

最后，两个预测头从编码器和解码器输出的连接表示中回归最终的点图和置信度图：

$$
\begin{array} { r } { X ^ { 1 , 1 } , C ^ { 1 } = \mathrm { H e a d } _ { 3 \mathrm { D } } ^ { 1 } ( [ H ^ { 1 } , H ^ { \prime 1 } ] ) , } \\ { X ^ { 2 , 1 } , C ^ { 2 } = \mathrm { H e a d } _ { 3 \mathrm { D } } ^ { 2 } ( [ H ^ { 2 } , H ^ { \prime 2 } ] ) . } \end{array}
$$

回归损失。DUSt3R 采用简单的回归损失以全监督的方式进行训练，其中 $\upsilon \in \{ 1 , 2 \}$ 为视图，i 为一个像素，其真实的 3D 点 $\hat { X } ^ { \nu , 1 } ~ \in ~ \mathbb { R } ^ { 3 }$ 被定义。在原始公式中，引入了归一化因子 $z , \hat { z }$ 以使重建对尺度不变。这些因子简单地定义为所有有效 3D 点到原点的平均距离。

$$
\ell _ { \mathrm { r e g r } } ( \nu , i ) = \left\| \frac { 1 } { z } X _ { i } ^ { \nu , 1 } - \frac { 1 } { \hat { z } } \hat { X } _ { i } ^ { \nu , 1 } \right\| ,
$$

度量预测。在本研究中，我们指出尺度不变性并不一定是可取的，因为某些潜在的应用场景如无地图视觉定位需要度量尺度的预测。因此，我们修改了回归损失，以在已知真实点图为度量时忽略对预测点图的归一化。也就是说，只有在真实值为度量时，我们才设置 $z \ : = \ \hat { z }$，因此在这种情况下 $\ell _ { \mathrm { r e g r } } ( \nu , i ) =$ $| | X _ { i } ^ { \upsilon , 1 } - \hat { X } _ { i } ^ { \upsilon , 1 } | | / \hat { z }$。如同DUSt3R [102]，最终的置信度感知回归损失定义为

$$
\mathcal { L } _ { \mathrm { c o n f } } = \sum _ { \nu \in \{ 1 , 2 \} } \sum _ { i \in \mathcal { V } ^ { \nu } } C _ { i } ^ { \nu } \ell _ { \mathrm { r e g r } } ( \nu , i ) - \alpha \log C _ { i } ^ { \nu } .
$$

# 3.2. 匹配预测头和损失

为了从点图中获得可靠的像素对应关系，一个标准解决方案是在某种不变特征空间中寻找互惠匹配。虽然这样的方案在DUSt3R回归的点图（即在一个三维空间中）下，即使在极端视角变化的情况下也表现得相当出色，但我们注意到结果对应关系相当不精确，导致精度不理想。这是一个相当自然的结果，因为（i）回归本质上受噪声的影响，并且（ii）因为DUSt3R从未明确针对匹配进行训练。匹配头。基于这些原因，我们建议增加一个第二个头，输出两个密集特征图 $D^{1}$ 和 $D^{2} \in \mathbb{R}^{H \times W \times d}$，维度为 $d$：

$$
\begin{array} { l } { { D ^ { 1 } = \mathrm { H e a d } _ { \mathrm { d e s c } } ^ { 1 } ( [ H ^ { 1 } , H ^ { \prime 1 } ] ) , } } \\ { { D ^ { 2 } = \mathrm { H e a d } _ { \mathrm { d e s c } } ^ { 2 } ( [ H ^ { 2 } , H ^ { \prime 2 } ] ) . } } \end{array}
$$

我们将头部实现为一个简单的两层多层感知机，并交替使用非线性GELU激活函数。最后，我们将每个局部特征标准化为单位模。更多细节可以在补充材料中找到。匹配目标。我们希望鼓励来自一幅图像的每个局部描述符与另一幅图像中表示同一场景中三维点的最多一个描述符匹配。为此，我们利用真实对应关系集上的infoNCE损失 $\hat { \cal M } = \{ ( i , j ) | \hat { X } _ { i } ^ { 1 , 1 } = \hat { X } _ { j } ^ { 2 , 1 } \}$。

$$
\mathcal { L } _ { \mathrm { m a t c h } } = - \sum _ { ( i , j ) \in \hat { \mathcal { M } } } \log \frac { s _ { \tau } ( i , j ) } { \sum _ { k \in \mathcal { P } ^ { 1 } } s _ { \tau } ( k , j ) } + \log \frac { s _ { \tau } ( i , j ) } { \sum _ { k \in \mathcal { P } ^ { 2 } } s _ { \tau } ( i , k ) } ,
$$

$$
\mathrm { w i t h } s _ { \tau } ( i , j ) = \mathrm { e x p } \left[ - \tau D _ { i } ^ { 1 \top } D _ { j } ^ { 2 } \right] .
$$

这里，$\mathcal { P } ^ { 1 } = \{ i | ( i , j ) \in \hat { \mathcal { M } } \}$ 和 $\mathcal { P } ^ { 2 } \ = \ \{ j | ( i , j ) \ \in \ \hat { M } \}$ 表示在每幅图像中考虑的像素子集，$\tau$ 是一个温度超参数。需要注意的是，这个匹配目标本质上是一个交叉熵分类损失：与公式 (6) 中的回归不同，只有当网络正确预测像素时，才会给予奖励，而不是邻近像素。这强烈鼓励网络实现高精度匹配。最后，将回归损失和匹配损失结合起来以获得最终的训练目标：

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { c o n f } } + \beta \mathcal { L } _ { \mathrm { m a t c h } }
$$

# 3.3. 快速互惠匹配

给定两个预测特征图 $D^{1}$ 和 $D^{2} \in \mathbb{R}^{H \times W \times d}$，我们的目标是提取一组可靠的像素对应关系，即彼此的相互最近邻：

$$
\begin{array} { r } { \mathcal { M } = \{ ( i , j ) \mid j = \mathrm { N N } _ { 2 } ( D _ { i } ^ { 1 } ) \mathrm { ~ a n d ~ } i = \mathrm { N N } _ { 1 } ( D _ { j } ^ { 2 } ) \} , } \\ { \mathrm { w i t h ~ N N } _ { A } ( D _ { j } ^ { B } ) = \arg \operatorname* { m i n } _ { i } \left\| D _ { i } ^ { A } - D _ { j } ^ { B } \right\| . } \end{array}
$$

不幸的是，朴素的互反匹配实现具有高计算复杂度 $O ( W ^ { 2 } H ^ { 2 } )$，因为必须将图像中的每个像素与另一图像中的每个像素进行比较。虽然优化最近邻（NN）搜索是可能的，例如使用 K-d 树 [1]，但这种优化在高维特征空间中通常效率非常低下，并且在所有情况下，输出 $D ^ { 1 }$ 和 $D ^ { 2 }$ 的 MASt3R 推理时间相对慢上几个数量级。快速匹配。因此，我们提出了一种基于子采样的更快方法。该方法基于一个迭代过程，从初始稀疏集合 $k$ 个像素 $U ^ { 0 } = \{ U _ { n } ^ { 0 } \} _ { n = 1 } ^ { k }$ 开始，来处理第一幅图像 $I ^ { 1 }$。然后，每个像素被映射到其在 $I ^ { 2 }$ 上的最近邻，得到 $V ^ { 1 }$，并且结果像素以同样的方式重新映射回图像 $I ^ { 1 }$：

$$
U ^ { t } \longmapsto [ \mathrm { N N } _ { 2 } ( D _ { u } ^ { 1 } ) ] _ { u \in U ^ { t } } \equiv V ^ { t } \longmapsto [ \mathrm { N N } _ { 1 } ( D _ { \nu } ^ { 2 } ) ] _ { v \in V ^ { t } } \equiv U ^ { t + 1 }
$$

互惠匹配集（形成一个循环的匹配，即 $\mathcal { M } _ { k } ^ { t } = \{ ( U _ { n } ^ { t } , V _ { n } ^ { t } ) \ | \ U _ { n } ^ { t } = U _ { n } ^ { t + 1 } \} $）被收集。在下一次迭代中，已经收敛的像素会被过滤掉，即更新 $\boldsymbol { U } ^ { t + 1 } : = \boldsymbol { U } ^ { t + 1 } \setminus \boldsymbol { U } ^ { t }$。同样，从 $t = 1$ 开始，我们也会验证并过滤 $V ^ { t + 1 }$，以类似的方式与 $V ^ { t }$ 进行比较。如图3（左）所示，该过程会以固定次数迭代，直到大多数对应关系收敛到稳定（互惠）对。在图3（中），我们展示了未收敛点的数量 $| U ^ { t } |$ 在几次迭代后迅速减少到零。最终，输出的对应关系集由所有互惠对的连接组成 $\mathcal { M } _ { k } = \bigcup _ { t } \mathcal { M } _ { k } ^ { t }$。理论保证。快速匹配的总复杂度为 $O ( k W H )$ ，比简单方法 $a l l$ 快 $W H / k \gg 1$ 倍，如图3（右）所示。值得指出的是，我们的快速匹配算法提取了完整集 $M$ 的一个子集，其大小受限于 $| { \mathcal { M } } _ { k } | \leq k$。我们在补充材料中研究了该算法的收敛保证以及它如何体现异常值过滤特性，这解释了为什么最终的准确性实际上高于使用完整对应集 $M$ 的情况，见图3（右）。

# 3.4. 粗到精匹配

由于注意力机制在输入图像区域 $( W \times H )$ 上的二次复杂性，MASt3R 仅处理最大边长为 512 像素的图像。更大的图像将需要显著更多的计算能力来进行训练，而且 ViTs 目前尚未在更大测试时间分辨率上具备良好的泛化能力。因此，高分辨率图像（例如 1M 像素）需要被缩小以便匹配，之后将得到的对应关系重新放大回原始图像分辨率。这可能导致一定的性能损失，有时不足以造成定位精度或重建质量的显著下降。

粗到细的匹配是一种标准技术，用于保留高分辨率图像与低分辨率算法匹配的优势。我们因此探索了这一理念在MASt3R中的应用。我们的过程首先在两幅图像的降采样版本上进行匹配。我们将通过子采样获得的粗匹配集合记为 $\mathcal { M } _ { k } ^ { 0 }$ 。接下来，我们在每幅全分辨率图像上独立生成重叠窗口裁剪的网格 $W ^ { 1 }$ 和 $W ^ { 2 } \in \mathbb { R } ^ { w \times 4 }$。每个窗口裁剪在其最大维度上为512像素，且连续窗口重叠50%。然后，我们可以枚举所有窗口对的集合 $( w _ { 1 } , w _ { 2 } ) \in W ^ { 1 } \times W ^ { 2 }$ ，从中选择一个子集以覆盖大部分粗匹配 $\mathcal { M } _ { k } ^ { 0 }$。具体来说，我们采用贪婪方式逐一添加窗口对，直到覆盖90%的匹配。最后，我们对每个窗口对独立进行匹配：

![](images/3.jpg)  
lh of pixels $U ^ { 0 }$ and propagating it iteratively using NN search. Searching for cycles (blue arrows) detect reciprocal correspondences and allows to accelerate the subsequent steps, by removing points that converged. Center: Average number of remaining points in $U ^ { t }$ at iteration $t = 1 \ldots 6$ After only 5 iterations, nearly all points have yconverg t  reciproalmat.RighPeroraneversus-ime tadeofo the Maprea.

$$
\begin{array} { r l } & { D ^ { w _ { 1 } } , D ^ { w _ { 2 } } = \mathrm { M A S t 3 R } ( I _ { w _ { 1 } } ^ { 1 } , I _ { w _ { 2 } } ^ { 2 } ) } \\ & { \mathrm { ~ } \mathcal { M } _ { k } ^ { w _ { 1 } , w _ { 2 } } = \mathrm { f a s t } \underline { { \mathrm { ~ r e c i p r o c a l } \_ \mathrm { N N } ( D ^ { w _ { 1 } } , D ^ { w _ { 2 } } ) } } } \end{array}
$$

从每对窗口中获得的对应关系最终被映射回原始图像坐标并串联在一起，从而提供了密集的全分辨率匹配。

# 4. 实验结果

在4.1节中，我们详细描述了MASt3R的训练过程。接下来，我们在多个任务上进行评估，每次与最先进的技术进行比较，首先是基于视觉的相机位姿估计，使用Map-Free Relocalization Benchmark [5]（4.2节）、CO3D和RealEstate数据集（4.3节）以及其他标准视觉定位基准（4.4节）。最后，我们在4.5节中利用MASt3R进行稠密多视图立体重建。

# 4.1. 训练

训练数据。我们使用14个数据集的混合来训练我们的网络：Habitat，ARKitScenes，Blended MVS，MegaDepth，Static Scenes 3D，ScanNet $^{++}$，CO3D-v2，Waymo，Mapfree，WildRgb，VirtualKitti，Unreal4K，TartanAir和一个内部数据集。这些数据集具有多样的场景类型：室内、室外、合成、真实世界、以物体为中心等。其中，有10个数据集具有度量真实标注数据。当数据集未直接提供图像对时，我们基于[104]中描述的方法提取图像对。具体而言，我们利用现成的图像检索和点匹配算法来匹配和验证图像对。

训练。我们的模型架构基于公开的 DUSt3R 模型 [102]，并使用相同的主干网络（ViTLarge 编码器和 ViT-Base 解码器）。为了充分利用 DUSt3R 的 3D 匹配能力，我们将模型权重初始化为公开可用的 DUSt3R 检查点。在每个周期中，我们随机从所有数据集中均匀抽样 $6 5 0 \mathrm { k }$ 对。其中，我们使用余弦调度训练网络，共计 35 个周期，初始学习率设置为 0.0001。与 [102] 类似，我们在训练时随机化图像纵横比，确保最大图像尺寸为 512 像素。我们将局部特征维度设置为 $d = 2 4$，匹配损失权重设置为 $\beta = 1$。在训练时让网络看到不同的尺度非常重要，因为粗到细的匹配从缩小的图像开始，然后放大细节（见第 3.4 节）。因此，我们在训练期间进行了积极的数据增强，以随机裁剪的形式进行。图像裁剪经过单应性变换，以保持主点的中心位置。对应采样。为了生成匹配损失所需的真实对应关系（等式 (10)），我们仅需在真实的 3D 点图 $\hat { X } ^ { 1 , 1 } \hat { X } ^ { 2 , 1 }$ 之间找到互为对应的关系。然后，我们随机从每对图像中抽样 4096 个对应关系。如果找不到足够的对应关系，我们会填充随机虚假对应关系，以保持找到真实匹配的可能性不变。快速最近邻。对于第 3.3 节中的快速互为匹配，我们根据 $x$ 的维度不同而实现最近邻函数 $\operatorname { N N } ( x )$ （等式 (14)）。当匹配 3D 点 $x \in \mathbb { R } ^ { 3 }$ 时，我们使用 K-d 树 [56] 实现 $\operatorname { N N } ( x )$。然而，当匹配具有 $d = 2 4$ 的局部特征时，由于维度诅咒 [25] 的影响，K-d 树变得非常低效。因此，在这种情况下，我们依赖优化过的 FAISS 库 [24, 45]。

# 4.2. 无需地图的定位

数据集描述。我们从无地图重定位基准测试[5]开始实验，这是一个非常具有挑战性的数据集，旨在给定单个参考图像在度量空间中定位摄像头，而无需任何地图。该数据集包括460个训练场景、65个验证场景和130个测试场景，每个场景均包含两个视频序列。依据该基准测试，我们通过虚拟对应重投影误差（VCRE）和相机位姿精度进行评估，详细信息请参见[5]。

子采样的影响。我们并未对该数据集采用粗到细的匹配，因为图像分辨率已接近MASt3R的工作分辨率 $( 7 2 0 \times 5 4 0$ 与 $5 1 2 \times 3 8 4$ 分别)。如第3.3节所述，即使使用优化代码进行最近邻搜索，计算密集的双向匹配也极为缓慢。因此，我们选择对子集中的双向对应关系进行子采样，从完整集合 $M$ 中最多保留 $k$ 个对应关系（等式（13））。图3（右）展示了子采样在AUC（VCRE）性能和时间上的影响。令人惊讶的是，对于中间值的子采样，性能显著提升。使用 $k = 3 0 0 0$ 时，我们可以将匹配加速64倍，同时显著提高性能。我们在补充材料中提供了有关这一现象的见解。除非另有说明，否则后续实验中我们保持 $k = 3 0 0 0$。关于损失和匹配模式的消融实验。我们在表1中报告了不同变体的验证集结果：DUSt3R匹配3D点（I）；MASt3R也匹配3D点（II）或局部特征（III、IV、V）。对于所有方法，我们从用预测匹配集估计的本质矩阵 [36] 计算相对位姿（PnP的表现相似）。度量场景尺度是通过用在KITTI上微调的现成DPT提取的深度推断出来的（I-IV）或通过MASt3R直接输出的深度（V）。首先，我们注意到所有提出的方法在性能上显著优于DUSt3R基线，这可能是由于MASt3R的训练时间更长且数据量更多。其他条件相同的情况下，匹配描述子的性能明显优于匹配3D点（II对比IV）。这证实了我们最初的分析，回归本质上不适合计算像素对应关系，参见第3.2节。

我们还研究了仅使用单一匹配目标 $\scriptstyle \sum _ { \mathrm { m a t c h } }$（来自公式（10），III）进行训练的影响。在这种情况下，与同时使用3D和匹配损失（IV）进行训练相比，整体性能下降，特别是在姿态估计精度方面（例如，(III) 的中位旋转为 $10.8^{\circ}$，而 (IV) 为 $3.0^{\circ}$）。我们指出，尽管解码器现在有更多的能力来执行单一任务，而不是在进行3D重建时执行两个任务，但这并未提升性能，这表明在3D中进行匹配确实对提高匹配效果至关重要。最后，我们观察到，当直接使用 MASt3R 输出的度量深度时，性能大幅提升。这表明，与匹配任务一样，深度预测任务与3D场景理解高度相关，两者相互促进。测试集上的比较结果见表2。总体而言，MASt3R以较大幅度超越所有最先进的方法，VCRE AUC 超过 $93\%$。相比于第二好的已发布方法 LoFTR $^ +$ KBR [81, 82] 的 $63.4\%$ AUC，这是 $30\%$ 的绝对提升。同时，中位平移误差大幅降低至 $36 \mathrm{ cm}$，而最先进方法的平移误差约为 $2 \mathrm{ m}$。当然，改进的很大一部分是由于 MASt3R 预测的度量深度，但请注意，我们通过 DPT-KITTI 利用深度的变体（因此纯粹基于匹配）也超越了所有最先进的方法。我们还提供了使用 MASt3R 进行直接回归的结果，即不进行匹配，简单地使用 PnP 在第二张图像的点图 $X^{2,1}$ 上。这些结果与我们基于匹配的变体不相上下，尽管未使用参考相机的真实标定。如我们在下文所示，这在其他定位数据集上并不成立，通常通过已知内参进行匹配（例如使用 $\mathrm{PnP}$ 或本质矩阵）进行姿态计算似乎更安全。定性结果。我们在图4中展示了一些具有强视点变化（最多 $180^{\circ}$）的配对匹配结果。我们还通过插图突出显示了一些特定区域，这些区域尽管经历了剧烈的外观变化，仍然被 MASt3R 正确匹配。我们相信，这些对应关系在基于2D的匹配方法中几乎是不可能获得的。相比之下，将匹配基于3D能够相对简单地解决此问题。

Table 1: Results on the validation set of the Map-free dataset. (First and second best)   

<table><tr><td rowspan="2" colspan="2"></td><td rowspan="2">20</td><td rowspan="2">depth</td><td colspan="3">VCRE (&lt;90px)</td><td colspan="4">Pose Error</td></tr><tr><td>Reproj. ↓</td><td>Prec. ↑</td><td>AUC ↑</td><td>Med. Err. (m,) ↓</td><td></td><td>Precision ↑</td><td>AUC ↑</td></tr><tr><td>(I)</td><td>DUSt3R</td><td>3d</td><td>DPT</td><td>125.8 px</td><td>45.2%</td><td>0.704</td><td>1.10m</td><td>9.4°</td><td>17.0%</td><td>0.344</td></tr><tr><td>(II)</td><td>MASt3R</td><td>3d</td><td>DPT</td><td>112.0 px</td><td>49.9%</td><td>0.732</td><td>0.94m</td><td>3.6°</td><td>21.5%</td><td>0.409</td></tr><tr><td>(III)</td><td>MASt3R-M</td><td>feat</td><td>DPT</td><td>107.7 px</td><td>51.7%</td><td>0.744</td><td>1.10m</td><td>10.8°</td><td>19.3%</td><td>0.382</td></tr><tr><td>(IV)</td><td>MASt3R</td><td>feat</td><td>DPT</td><td>112.9 px</td><td>51.5%</td><td>0.752</td><td>0.93m</td><td>3.0°</td><td>23.2%</td><td>0.435</td></tr><tr><td>(V)</td><td>MASt3R</td><td>feat</td><td>(auto)</td><td>57.2 px</td><td>75.9%</td><td>0.934</td><td>0.46m</td><td>3.0°</td><td>51.7%</td><td>0.746</td></tr></table>

Table 2: Comparison with the state of the art on the test set of the Map-free dataset.   

<table><tr><td></td><td></td><td colspan="3">VCRE (&lt;90px)</td><td colspan="4">Pose Error</td></tr><tr><td></td><td>depth</td><td>Reproj. ↓</td><td>Prec. ↑</td><td>AUC ↑</td><td>Med. Err. (m,) ↓</td><td></td><td>Precision ↑</td><td>AUC ↑</td></tr><tr><td>RPR [5]</td><td>DPT</td><td>147.1 px</td><td>40.2%</td><td>0.402</td><td>1.68m</td><td>22.5°</td><td>6.0%</td><td>0.060</td></tr><tr><td>SIFT [52]</td><td>DPT</td><td>222.8 px</td><td>25.0%</td><td>0.504</td><td>2.93m</td><td>61.4°</td><td>10.3%</td><td>0.252</td></tr><tr><td>SP+SG [72]</td><td>DPT</td><td>160.3 px</td><td>36.1%</td><td>0.602</td><td>1.88m</td><td>25.4°</td><td>16.8%</td><td>0.346</td></tr><tr><td>LoFTR [82]</td><td>KBR</td><td>165.0 px</td><td>34.3%</td><td>0.634</td><td>2.23m</td><td>37.8°</td><td>11.0%</td><td>0.295</td></tr><tr><td>DUSt3R [102]</td><td>DPT</td><td>116.0 px</td><td>50.3%</td><td>0.697</td><td>0.97m</td><td>7.1°</td><td>21.6%</td><td>0.394</td></tr><tr><td>MASt3R</td><td>DPT</td><td>104.0 px</td><td>54.2%</td><td>0.726</td><td>0.80m</td><td>2.2°</td><td>27.0%</td><td>0.456</td></tr><tr><td>MASt3R</td><td>(auto)</td><td>48.7 px</td><td>79.3%</td><td>0.933</td><td>0.36m</td><td>2.2°</td><td>54.7%</td><td>0.740</td></tr><tr><td>MASt3R (direct reg.)</td><td></td><td>53.2 px</td><td>79.1%</td><td>0.941</td><td>0.42m</td><td>3.1°</td><td>53.0%</td><td>0.777</td></tr></table>

# 4.3. 相对姿态估计

数据集和协议。接下来，我们在 CO3Dv2 [67] 和 RealEstate10k [121] 数据集上评估相对姿态估计任务。CO3Dv2 包含从大约 37k 视频中提取的 600 万帧，覆盖了 51 个 MS-COCO 类别。真实相机位姿是通过 COLMAP [75] 从每段视频中的 200 帧获得的。RealEstate10k 是一个室内/室外数据集，包含 80K 个 YouTube 视频片段，总计 1000 万帧，摄像机位姿通过 SLAM 和束调整获得。按照 [100]，我们在 CO3Dv2 的 41 个类别和 RealEstate10k 测试集中的 1.8K 视频片段上评估 MASt3R。每个序列长 10 帧，我们评估所有可能的 45 对之间的相对相机位姿，不使用真实的焦距。基准和指标。与之前一样，使用 MASt3R 获得的匹配用于估计基本矩阵和相对姿态。请注意，我们的预测始终是成对进行的，与所有采用多个视角的方法（除了 DUSt3RPnP）相对。我们与一些近期的数据驱动方法进行比较，如 RelPose [115]、${ \mathrm { R e l P o s e } } ++$ [115]、PoseReg 和 PoseDiff [100]，以及最近的 RayDiff [116] 和 DUSt3R [102]。我们还报告了更多传统的 SfM 方法的结果，例如结合 SuperPoint [21] 和 SuperGlue [72] 的 PixSFM [50] 和 COLMAP [76] $\mathrm { ( C O L M A P + S P S G ) }$。与 [100] 类似，我们报告每对图像的相对旋转精度 (RRA) 和相对平移精度 (RTA) 以评估相对姿态误差，并选择阈值 $\tau = 15$ 以报告 $\mathrm { R T A @ 15 }$ 和 RRA $@ 15$。此外，我们计算平均准确率 (mAA30)，定义为在 $min( \mathrm { R R A @ 30 }, \mathrm { R T A @ 30 })$ 的角差准确度曲线下面积。结果。如表 3 所示，SfM 方法在该任务上表现明显较差，主要由于视觉支持不足。这是因为图像通常只观察一个小物体，再加上许多对之间的基线宽广，有时达到 $180^{\circ}$。相反，基于 3D 的方法如 RayDiffusion、DUSt3R 和 MASt3R 是该数据集中两个最具竞争力的方法，后者在两个数据集上的平移和 mAA 上均处于领先地位。值得注意的是，在 RealEstate 中，我们的 mAA 得分比最佳多视角方法至少提高 8.7 分，比成对的 DUSt3R 提高 15.2 分。这展示了我们方法在少输入视角设置下的准确性和鲁棒性。

# 4.4. 视觉定位

数据集。我们接着评估MASt3R在绝对位姿估计任务上的表现，所用数据集包括Aachen Day-Night[118]和InLoc[84]。Aachen数据集包括4,328张用手持相机拍摄的参考图像，以及824张白天和98张夜间的查询图像，这些图像是在德国亚琛老城区使用手机拍摄的。InLoc[84]是一个室内数据集，具有在$9 , 9 7 2 \mathrm { R G B - D } + 6 \mathrm { D O F }$位姿数据库图像和329张使用iPhone 7拍摄的查询图像之间的显著外观变化挑战。

![](images/4.jpg)  
ualiape heae atat.To air wi srvot hangs.Th spots in close-up. These regions could hardly be matched by local keypoints. See text for details.

评估指标。我们报告在三个阈值下成功定位图像的百分比：$(0.25 \mathrm{ m}, 2^{\circ})$，$(0.5 \mathrm{ m}, 5^{\circ})$和$(5 \mathrm{ m}, 10^{\circ})$用于Aachen，以及$(0.25 \mathrm{ m}, 10^{\circ})$，$(0.5 \mathrm{ m}, 10^{\circ})$，$(1 \mathrm{ m}, 10^{\circ})$用于InLoc。结果详见表4。我们研究了MASt3R在不同数量的检索图像下的性能。如预期，检索图像数量越多（top40），性能越好，在Aachen上取得了具有竞争力的表现，并在InLoc上显著超过了现有技术。有趣的是，即使只有一张检索图像（top1），我们的方法仍表现良好，展示了3D基础匹配的鲁棒性。我们还包括了直接回归的结果，这些结果相对较差，显示出数据集规模对定位误差的显著影响，即小场景受到的影响要小得多（见4.2节中的Map-free结果）。这确认了特征匹配在估计可靠姿态中的重要性。

# 4.5. 多视图三维重建

我们最终通过三角测量获得的匹配来执行多视图立体重建（MVS）。请注意，匹配是在不事先了解相机的情况下以全分辨率进行的，后者仅用于在真实标注数据参考框架中三角测量匹配。我们通过几何一致性后处理去除虚假三维点[99]。数据集和度量标准。我们在DTU [3]数据集上评估我们的预测。与所有竞争的学习方法相反，我们在零-shot设置中应用我们的网络，即我们不在DTU训练集上进行训练或微调，而是按原样应用我们的模型。在表3中，我们报告了基准作者提供的平均准确性、完整性和Chamfer距离误差度量。重建形状上一个点的准确性定义为到真实标注数据的最小欧几里得距离，而真实标注数据上一个点的完整性则是到重建形状的最小欧几里得距离。总体Chamfer距离是前两项度量的平均值。结果。基于数据驱动的方法在该领域的表现显著优于手工制作的方法，Chamfer误差减少了一半。据我们所知，我们是在零-shot设置中首次得出这样的结论。MASt3R不仅超越了DUSt3R基线，还与最佳方法展开竞争，所有这些都无需利用相机标定或位姿进行匹配，也没有见过这种相机设置。

Table 3: Left: Multi-view pose regression on the CO3Dv2 [67] and RealEstate10K [121] with 10 random frames. Parthe denot etho that  ot report resul nhe 10viws , e report the bstrcmn (8 vis). We distinguish between (a) multi-view and (b) pairwise methods. Right: Dense MVS results on the DTUdatase, in m. Handcraftmethods () perfor wors than learnig-bas apprache () that trai th em  h e zhoet ht reasonable performance.   

<table><tr><td rowspan="2"></td><td rowspan="2">Methods</td><td colspan="3">Co3Dv2</td><td rowspan="2">RealEstate10K mAA(30)</td></tr><tr><td>RRA@15</td><td>RTA@15</td><td>mAA(30)</td></tr><tr><td rowspan="8">(a)</td><td>Colmap+SG [21,72]</td><td>36.1</td><td>27.3</td><td>25.3</td><td>45.2</td></tr><tr><td>PixSfM [50]</td><td>33.7</td><td>32.9</td><td>30.1</td><td>49.4</td></tr><tr><td>RelPose [115]</td><td>57.1</td><td>-</td><td>-</td><td>-</td></tr><tr><td>PosReg [100]</td><td>53.2</td><td>49.1</td><td>45.0</td><td>-</td></tr><tr><td>PoseDiff [100]</td><td>80.5</td><td>79.8</td><td>66.5</td><td>48.0</td></tr><tr><td>RelPose++ [49]</td><td>(85.5)</td><td>-</td><td>-</td><td>-</td></tr><tr><td>RayDiff [116]</td><td>(93.3)</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DUSt3R-GA [102]</td><td>96.2</td><td>86.8</td><td>76.7</td><td>67.7</td></tr><tr><td rowspan="2">(b)</td><td>DUSt3R [102]</td><td>94.3</td><td>88.4</td><td>77.2</td><td>61.2</td></tr><tr><td>MASt3R</td><td>94.6</td><td>91.9</td><td>81.8</td><td>76.4</td></tr></table>

<table><tr><td></td><td>Methods</td><td>Acc.↓</td><td>Comp.↓</td><td>Overall↓</td></tr><tr><td rowspan="4">(c)</td><td>Camp [13]</td><td>0.835</td><td>0.554</td><td>0.695</td></tr><tr><td>Furu [31]</td><td>0.613</td><td>0.941</td><td>0.777</td></tr><tr><td>Tola [90]</td><td>0.342</td><td>1.190</td><td>0.766</td></tr><tr><td>Gipuma [32]</td><td>0.283</td><td>0.873</td><td>0.578</td></tr><tr><td rowspan="8">(d)</td><td>MVSNet [110]</td><td>0.396</td><td>0.527</td><td>0.462</td></tr><tr><td>CVP-MVSNet [109]</td><td>0.296</td><td>0.406</td><td>0.351</td></tr><tr><td>UCS-Net [17]</td><td>0.338</td><td>0.349</td><td>0.344</td></tr><tr><td>CER-MVS [55]</td><td>0.359</td><td>0.305</td><td>0.332</td></tr><tr><td>CIDER [107]</td><td>0.417</td><td>0.437</td><td>0.427</td></tr><tr><td>PatchmatchNet [99]</td><td>0.427</td><td>0.277</td><td>0.352</td></tr><tr><td>GeoMVSNet [119]</td><td>0.331</td><td>0.259</td><td>0.295</td></tr><tr><td></td><td></td><td></td><td>1.741</td></tr><tr><td rowspan="2">(e)</td><td>DUSt3R [102]</td><td>2.677</td><td>0.805</td><td></td></tr><tr><td>MASt3R</td><td>0.403</td><td>0.344</td><td>0.374</td></tr></table>

TablVisual 本地化结果在昼夜和内部场景上。我们报告了不同检索数据库图像的结果（前N）。

<table><tr><td rowspan="2">Methods</td><td colspan="2">AachenDayNight[118]</td><td colspan="2">InLoc[84]</td></tr><tr><td>Day</td><td>Night</td><td>DUC1</td><td>DUC2</td></tr><tr><td>Kapture+R2D2 [41]</td><td>91.3/97.0/99.5</td><td>78.5/91.6/100</td><td>41.4/60.1/73.7</td><td>47.3/67.2/73.3</td></tr><tr><td>SP+SuperGlue [72]</td><td>89.8/96.1/99.4</td><td>77.0/90.6/100</td><td>49.0/68.7/80.8</td><td>53.4/77.1/82.4</td></tr><tr><td>SP+LightGlue [51]</td><td>90.2/96.0/99.4</td><td>77.0/91.1/100</td><td>49.0/68.2/79.3</td><td>55.0/74.8/79.4</td></tr><tr><td>LoFTR [82]</td><td>88.7/95.6/99.0</td><td>78.5/90.6/99.0</td><td>47.5/72.2/84.8</td><td>54.2/74.8/85.5</td></tr><tr><td>DKM [27]</td><td></td><td></td><td>51.5/75.3/86.9</td><td>63.4/82.4/87.8</td></tr><tr><td>DUSt3R top1 [102]</td><td>72.7/89.6/98.1</td><td>59.7/80.1/93.2</td><td>36.4/55.1/66.7</td><td>27.5/42.7/49.6</td></tr><tr><td>DUSt3R top20 [102]</td><td>79.4/94.3/99.5</td><td>74.9/91.1/99.0</td><td>53.0/74.2/89.9</td><td>61.8/77.1/84.0</td></tr><tr><td>MASt3R top1</td><td>79.6/93.5/98.7</td><td>70.2/88.0/97.4</td><td>41.9/64.1/73.2</td><td>38.9/55.7/62.6</td></tr><tr><td>MASt3R top20</td><td>83.4/95.3/99.4</td><td>76.4/91.6/100</td><td>55.1/77.8/90.4</td><td>71.0/84.7/89.3</td></tr><tr><td>MASt3R top40</td><td>82.2/93.9/99.5</td><td>75.4/91.6/100</td><td>56.1/79.3/90.9</td><td>71.0/87.0/91.6</td></tr><tr><td>MASt3R direct reg. top1</td><td>1.5/4.5/60.7</td><td>1.6/4.2/47.6</td><td>13.1/32.3/58.1</td><td>10.7/26.0/38.2</td></tr></table>

# 5. 结论

基于MASt3R在三维中进行图像匹配，显著提高了在许多公共基准上的相机姿态和定位任务的标准。我们通过匹配成功改进了DUSt3R，达到了两全其美的效果：增强了鲁棒性，并实现了甚至超越仅依靠像素匹配所能达到的效果。我们引入了一个快速的相互匹配器和一个粗到细的方法以实现高效处理，使用户能够在准确性和速度之间取得平衡。MASt3R能够在少视角情况下（甚至在top1中）执行，我们相信这将大大增强定位的多样性。

# 附录

在本附录中，我们首先在附录 A 中呈现各种任务的额外定性示例，接着在附录 B 中提供快速互配算法收敛性的证明以及相关性能提升的深入研究。最后，我们在附录 C 中展示了关于粗到细匹配影响的消融研究。

# A. 额外的定性结果

我们在此提供了有关DTU [3]、InLoc [84]、Aachen Day-Night数据集 [118] 和无地图基准 [5] 的额外定性结果。DTU上的MVS。我们在图5中展示了后处理后的输出点云，这些点云根据50个最近邻的切平面计算得到的近似法线进行了着色。我们再一次强调，点云是通过对MASt3R的粗到细匹配进行三角测量得到的原始值。匹配采用一对多的策略，这意味着我们没有利用来自GT相机的极线约束，这与现有所有MVS方法形成了鲜明对比。MASt3R特别精确且鲁棒，提供了清晰且密集的细节。即使在对比度低的均匀区域，例如蔬菜表面或电源侧面，重建结果也很完整。匹配对于不同纹理或材料也很稳健，并且对兰伯特假设的违反（即蔬菜、塑料表面或白色雕塑上的镜面反射）也表现出较好的鲁棒性。

定性匹配结果。我们展示了一些匹配示例，图6为无地图基准[5]，图7为InLoc [84]数据集，图8为亚琛昼夜数据集[118]。所提出的MASt3R方法对极端视角变化具有鲁棒性，并且在这种情况下仍能提供大致正确的对应关系（图6中无地图右侧的配对），即使是面对面的视角（InLoc 7中的咖啡桌或走廊配对）。这让人想起了DUSt3R的能力，它在这种情况下提供了前所未有的鲁棒性。类似地，我们的方法能够处理大规模差异（例如图6中的无地图），重复和模糊的模式，以及环境和昼夜照明的变化（图8）。有趣的是，MASt3R输出的对应关系的准确性在视角基线增加时优雅地下降。即使在对应关系非常粗略估计的极端情况下，仍然可以恢复大致正确的相对相机姿态。得益于这些能力，MASt3R在多个基准测试中在零样本设置下达到了最先进的性能或接近于此。我们希望这项工作能够促进针对于多种视觉任务的点图回归研究的进展，其中鲁棒性和准确性至关重要。

![](images/5.jpg)  

Figure 5: Qualitative MVS results on the DTU dataset [3] simply obtained by triangulating the dense matches from MASt3R.

![](images/6.jpg)  

Figure 6: Qualitative examples of matching on Map-free localization benchmark.

![](images/7.jpg)  

Figure 7: Qualitative examples of matching on the InLoc localization benchmark.

![](images/8.jpg)  

Figure 8: Qualitative examples o matching onthe Aachen Day-Night localization benchmark. Pairs from the day subset are on the left column, and pairs from the night subset are on the right column.

# B. 快速互逆匹配

# B.1. 理论研究

我们在此详细说明主论文第3.3节中提出的快速倒数匹配算法的收敛性理论证明。与传统的二分图匹配形式 [18] 不同，后者使用完整图进行匹配，我们希望通过仅计算其较小的一部分来降低计算复杂性。如主论文的公式 (14) 所述，考虑两个预测特征集 $D^{1}$ 和 $D^{2} \in \mathbb{R}^{H \times W \times d}$ ，部分倒数匹配归结为找到互为最近邻的部分倒数对应关系，即互为最近邻（NN）：

$$
\begin{array} { r } { \mathcal { M } = \{ ( i , j ) \mid j = \mathrm { N N } _ { 2 } ( D _ { i } ^ { 1 } ) \mathrm { ~ a n d ~ } i = \mathrm { N N } _ { 1 } ( D _ { j } ^ { 2 } ) \} , } \\ { \mathrm { w i t h ~ N N } _ { A } ( D _ { j } ^ { B } ) = \arg \operatorname* { m i n } _ { i } \left\| D _ { i } ^ { A } - D _ { j } ^ { B } \right\| . } \end{array}
$$

我们在这里回顾算法的行为：初始的 $k$ 个 $I^{1}$ 的像素集合 ${\cal U}^{0} = \{ {\cal U}_{n}^{0} \}_{n=1}^{k}$，其中 $k \ll WH$，被映射到它们在 $I^{2}$ 中的最近邻，生成 $V^{1}$，然后这些 $V^{1}$ 被映射回 $I^{1}$ 中它们的最近邻：

$$
U ^ { t } \longmapsto [ \mathrm { N N } _ { 2 } ( D _ { u } ^ { 1 } ) ] _ { u \in U ^ { t } } \equiv V ^ { t } \longmapsto [ \mathrm { N N } _ { 1 } ( D _ { \nu } ^ { 2 } ) ] _ { v \in V ^ { t } } \equiv U ^ { t + 1 }
$$

经过这种反复的映射，互为匹配的样本（即形成循环的样本）被恢复并从 $U ^ { t + 1 }$ 中移除。剩余的“活跃”样本被映射回 $I ^ { 2 }$ 并再次检查互反性。我们对这个过程进行若干轮迭代。在足够的迭代后，我们会丢弃任何剩余的活跃样本。需要注意的是，我们使用的最近邻算法是确定性的，在其他图像中有多个描述符共享相同的最小距离（或最大相似度）时，它始终返回相同的索引，尽管这种情况非常不常见，因为描述符是实值的。

![](images/9.jpg)  

Figure 9: Illustration of the iterative FRM algorithm. Starting from 5 pixels in $I ^ { 1 }$ at $t = 0$ , the FRM connects them to their Nearest Neighbors (NN) in $I ^ { 2 }$ , and maps them back to their NN in $I ^ { 1 }$ I they go back to their starting p  pk  elat) i eteOhe t)hr ieanetarmpehexala . We howang he art pointbas ioe urap whic te will converge towards the same cycle. For clarity, all edges of $\mathcal { G }$ were not drawn.

收敛证明。根据设计，快速倒数匹配（FRM）在$I^{1}$和$I^{2}$之间的最近邻有向二部图$\mathcal{G}$上运行。$\mathcal{G}$包含有向边$\varepsilon$。所有节点，即像素，均属于$\mathcal{G}$，因为我们为每个像素的最近邻添加了一条边，但需注意，并非所有像素可以到达其他像素。例如，$I^{1}$和$I^{2}$中两个相互对应的像素仅彼此相连，而不与其他任何像素相连。这意味着$\mathcal{G}$可能由多个不相交的子图$\mathcal{G}^{i}$（$1 \leq i \leq HW$）构成，具有有向边$\mathcal{E}^{i}$（见图9）。命题B.1. 每个子图$\mathcal{G}^{i}$中只能存在一个循环。证明。这是一个相当简单的事实，因为我们构建$\mathcal{G}$时使得每个节点只能有一条边出发。如果沿着子图$\mathcal{G}^{i}$的路径前进，一旦到达属于循环的节点，则没有边可以离开循环，因为唯一的出边已是循环的一部分。因此，在$\mathcal{G}^{i}$中不能存在第二个（或更多）循环。引理B.2. 每个子图$\mathcal{G}^{i}$要么是一个单一循环，要么是一个特殊的树形结构，即一个有向图，其中从任何节点到根循环仅存在一条路径。

证明。前者自然而然地源于之前的解释：因为在 $\mathcal { G } ^ { i }$ 中只能存在一个循环，所以它自然可以形成一个循环。我们现在证明后者，即当 $\mathcal { G } ^ { i }$ 不是平凡循环时。我们从一个任意节点 $a$ 开始沿着 $\mathcal { G } ^ { i }$ 行进，该节点附加了描述符 $D _ { a } ^ { 1 }$ 。从该节点唯一的边缘指向其最近邻 $N N _ { 2 } ( D _ { a } ^ { 1 } ) = b$ 。现在在节点 $b$ ，我们做同样的事情，沿着唯一的边缘返回到 $I ^ { 1 }$ ：$N N _ { 1 } ( D _ { b } ^ { 2 } ) = c$。在 $I ^ { 1 }$ 和 $I ^ { 2 }$ 之间交替，我们得到 $N N _ { 2 } ( D _ { c } ^ { 1 } ) = d$ ，$N N _ { 1 } ( D _ { d } ^ { 2 } ) = e$ 等等。我们将 $s ( u , \nu ) = D _ { u } ^ { 1 \top } D _ { \nu } ^ { 2 }$ 表示为节点 $u$ 和 $\nu$ 之间边缘的相似度评分，其中 $( u , \nu ) \in \mathcal { E } ^ { i }$ 。由于边缘是最近邻，我们注意到 $s ( a , b ) \ \leq \ s ( c , b )$ 这是显然的，因为如果 $s ( c , b ) < s ( a , b )$ ，则 $b$ 的最近邻将不再是 $c$ 而至少是 $a$ 。将这一性质扩展到沿 $\mathcal { G } ^ { i }$ 的路径，可以得出：

$$
s ( a , b ) \leq s ( c , b ) \leq s ( c , d ) \leq s ( e , d ) . . .
$$

这意味着相似度分数沿着图的走向单调递增。$\mathcal { G } ^ { i }$ 中的节点数量是有限的，因此这个序列达到了上界相似度值 $s ( u , \upsilon )$。因为 $s ( u , \nu )$ 是 $\mathcal { G } ^ { i }$ 中的最大相似度，这确保了 $N N _ { 2 } ( D _ { u } ^ { 1 } ) = \nu$ 和 $N N _ { 1 } ( D _ { \nu } ^ { 2 } ) = u$ 形成了至少两个节点的循环。这意味着在 $\mathcal { G } ^ { i }$ 中，总是存在一个循环，位于最大相似度对之间。根据命题 B.1，我们可以得出结论：在 $\mathcal { G } ^ { i }$ 中不存在其他循环，因此每个起始点都必然通过单一路径指向根节点，形成一个以根节点为处的树形结构。

![](images/10.jpg)  
FIlut theienca ensy wheusenemathisel fast reciprocal matching with $k = 3 0 0 0$ .Fast reciprocal matching samples correspondences with a bias for large convergence basins, resulting in a more unifor coverage of the mages.Coverage can be measured in ters the mean and standard deviation $\sigma$ of the point matches in each density map, plotted as colored ellipses (red, green and blue correspond respectively to $1 \sigma , 1 . 5 \sigma$ and $2 \sigma$ .

请注意，根循环可以由超过两个节点组成，如果等式（21）中的多个最大相似度完全相等且最近邻算法生成了更大的循环。由于 $\mathcal { G }$ 是一个二分图，$\mathcal { G } ^ { i }$ 也是二分图，这意味着最终循环由偶数个节点组成。然而，在实践中，我们使用的是 24 维浮点描述符。为了存在更大的循环，例如 由 4 个节点 $a , b , c , d$ 组成的循环，相似度必须满足日益苛刻的约束，例如 $s ( a , b ) = s ( c , b ) = s ( c , d ) = s ( a , d )$。在实值距离下这种情况极不可能出现，因此我们认为其影响可以忽略不计。推论 B.3. 无论在 $\mathcal { G } ^ { i }$ 中的起始点是什么，FRM 算法始终会收敛到互为匹配的点。这从上面的论述中自然得出：我们并没有对这次遍历的起始点或其所属的子图做任何假设。对于图中的任何起始点，即对于所有初始像素 $U$，FRM 算法的设计将遵循最近邻的子图，最终会导致根循环，而根循环根据定义是一个互为匹配的点。

我们在图9中展示了这种行为。在上部（粉色）中，起始点 $u_{0}$ 直接位于一个包含两个节点 $u_{0}$ 和 $\upsilon_{0}$ 的循环中，算法在第一个循环验证后于步骤 $t=1$ 停止。下部展示了一个更复杂的收敛盆地案例，其中多个起始点 $u_{1}, u_{2}, u_{3}, u_{4}$ 分别导致两个节点 $\upsilon_{1}$ 和 $\upsilon_{2}$ 在 $I^{2}$ 中。沿着树状图的根路径，随着更新 $U$ 和 $V$，该算法在时间步 $t=1$ 时找到了 $u_{1}$ 和 $\upsilon_{1}$ 之间的循环。从5个初始像素位置中，算法返回了一个唯一的双向对应关系。请注意，可以人工构建一个最大化最近邻查询数量的图，从而影响计算效率，但如主论文图2（中间部分）所示，这在实际中非常不可能。活跃样本的数量，例如未达到循环的样本，在仅经过6次迭代后迅速降至0，从而显著加快了计算速度（右侧）。

![](images/11.jpg)  
ur Iratn ebass  theag .a basl wi (r eahi  ve when applying the fast reciprocal matching algorithm.

命题 B.4. 从 $k \ll H W$ 个样本开始，FRM 算法恢复一个所有可能的互反对应关系的子集 $\mathcal { M } _ { k }$，其基数为 $| { \cal M } _ { k } | = j \le k$。证明。这个事实显然源于 $k$ 个稀疏初始样本 $U$。如前所述，$\mathcal { G }$ 最多由 HW 个子图 $\mathcal { G } ^ { i }$ 组成。由于我们用 $k \ll H W$ 个种子初始化算法，这些种子最多可以覆盖 $k$ 个子图，每个子图导向一个互反匹配。由于可能存在收敛盆，如图 9 所示，样本可以沿着路径合并到其根循环，从而减少最终的互反数量，并解释了不等式 $j \leq k$。

# B.2. 性能随着快速匹配而提高

如主论文图2所示，FRM显著提高了性能。在我们在图9中提供的最小示例中，明显可以看到FRM提供了一个偏向于找到具有大基池的互匹配的采样（底部），因为相比于小基池（顶部），更多的初始样本可以落在它们上。请注意，基池的大小与互匹配的最大密度成反比。有趣的是，使用FRM，这导致互匹配的分布比全匹配更加均匀（即空间覆盖），如图10所示。由于空间覆盖更加均匀，RANSAC能够比在小图像区域内聚集许多点时更好地估计极线，这反过来提供了更好且更稳定的位姿估计。

为了展示盆地偏倚采样的效果，我们提出计算完整的对应集 $M$（公式（18））并进行两种采样：首先，我们随机进行简单采样，以达到与 FRM 相同的互补数量。其次，我们计算每个盆地的大小（如图 11 所示），并利用这些大小对采样进行偏倚。我们在图 12 中报告了这个实验的结果。虽然随机采样导致性能的灾难性下降，但盆地偏倚采样实际上提高了与使用完整图相比的性能（最右侧数据点）。正如预期的那样，FRM 算法提供的性能与偏倚采样密切相关，但相比于需要计算所有互补匹配以测量盆地大小的盆地偏倚采样，其计算量仅为其一小部分。重要的是，这些观察结果对再投影误差和姿态准确性均有效，不论用于估计相对姿态的 RANSAC 变体是什么。

# C. 粗到精

在本节中，我们展示了粗到细策略的重要优点。我们将其与仅采用粗匹配的方法进行比较，该方法只是计算输入图像在降采样到网络分辨率后的对应关系。

视觉定位在亚琛日夜数据集上[118]。对于该任务，输入图像的分辨率为 $1600 \times 1200$ 和 $1024 \times 768$，在横向和纵向模式下均缩放至 $512 \times 384 / 384 \times 512$。我们报告在三个阈值下成功定位图像的百分比：$(0.25 \mathrm{ m}, 2^{\circ})$、$(0.5 \mathrm{ m}, 5^{\circ})$ 和 $(5 \mathrm{ m}, 10^{\circ})$，详见表5（左）。我们观察到仅使用粗匹配时性能显著下降，夜间数据分割的top1准确率下降幅度可达 $15\%$。

![](images/12.jpg)  
Fr Compariso  the perforanc  the Map-ree bencmark validation ) or different subsaplig ae'ieoe enisi teial u  romate; ' denotes the proposed fast reciprocal matching; and 'basin'denotes random subsampling weighted by the size ocven asThe stn asrat peo lary hes aivspi catastrophic results.

Table 5:Coarse matching compared to Coarse-to-Fine for the tasksof visual localization on Aachen Day-Night (left) and MVS reconstruction on the DTU dataset (right).   

<table><tr><td>Methods</td><td>Coarse-to-Fine</td><td>Day</td><td>Night</td></tr><tr><td>MASt3R top1</td><td>×</td><td>74.9/90.3/98.5</td><td>55.5/82.2/95.8</td></tr><tr><td>MASt3R top1</td><td>✓</td><td>79.6/93.5/98.7</td><td>70.2/88.0/97.4</td></tr><tr><td>MASt3R top20</td><td>×</td><td>80.8/93.8/99.5</td><td>74.3/92.1/100</td></tr><tr><td>MASt3R top20</td><td>✓</td><td>83.4/95.3/99.4</td><td>76.4/91.6/100</td></tr></table>

MVS。DTU数据集[3]的输入图像分辨率为$1200 \times 1600$，缩小至$384 \times 512$。如主论文所述，我们在此报告了使用MASt3R获得的三角测量匹配结果的精度、完整性和Chamfer距离，分别在粗糙匹配和粗到细设置下，见表5（右侧）。虽然粗糙匹配仍然优于DUSt3R的直接回归，但我们看到在所有指标上重建质量明显下降，重建误差几乎翻倍。

# D. 详细实验设置

在我们的实验中，我们将置信损失权重 $\alpha = 0.2$ 设定为 [102] 中的值，匹配损失权重 $\beta = 1$，局部特征维度 $d = 24$，以及 InfoNCE 损失中的温度 $\tau = 0.07$。我们在表 6 中报告了用于训练 MASt3R 的详细超参数设置。

# References

[1] Scipy. https://docs.scipy.org/doc/scipy. 5   
[2] RGBD Objects in the Wild: Scaling Real-World 3D Object Learning from RGB-D Videos, 2024. arXiv:2401.12592 [cs]. 6   
[3] Henrik Aanæs, Rasmus Ramsbøl Jensen, George Vogiatzis, Engin Tola, and Anders Bjorholm Dahl. Largescale data for multiple-view stereopsis. IJCV, 2016. 9, 11, 17

Table 6: Detailed hyper-parameters for the training   

<table><tr><td>Methods</td><td>Acc.↓</td><td>Comp.↓</td><td>Overall</td></tr><tr><td>DUSt3R [102]</td><td>2.677</td><td>0.805</td><td>1.741</td></tr><tr><td>MASt3R Coarse</td><td>0.652</td><td>0.592</td><td>0.622</td></tr><tr><td>MASt3R</td><td>0.403</td><td>0.344</td><td>0.374</td></tr></table>

<table><tr><td>Hyper-parameters</td><td>fine-tuning</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Base learning rate</td><td>1e-4</td></tr><tr><td>Weight decay</td><td>0.05</td></tr><tr><td>Adam β</td><td>(0.9, 0.95 )</td></tr><tr><td>Pairs per Epoch</td><td>650k</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Epochs</td><td>35</td></tr><tr><td>Warmup epochs</td><td>7</td></tr><tr><td>Learning rate scheduler</td><td>Cosine decay</td></tr><tr><td rowspan="3">Input resolutions</td><td>512×384, 512×336</td></tr><tr><td>512x288, 512x256</td></tr><tr><td>512×160</td></tr><tr><td>Image Augmentations</td><td>Random crop, color jitter</td></tr><tr><td>Initialization</td><td>DUSt3R [102]</td></tr></table>

[4] Howard Addison, Trulls Eduard, etru1927, Yi Kwang Moo, old ufo, Dane Sohier, and Jin Yuhe. Image matching challenge 2022, 2022. 3

[5] Eduardo Arnold, Jamie Wynn, Sara Vicente, Guillermo Garcia-Hernando, Áron Monszpart, Victor Adrian Prisacariu, Daniyar Turmukhambetov, and Eric Brachmann. Map-free visual relocalization: Metric pose relative to a single image. In ECCV, 2022. 2, 3, 6, 7, 8, 11

[6] Chow Ashley, Trulls Eduard, HCL-Jevster, Yi

Kwang Moo, lcmrll, old ufo, Dane Sohier, tanjigou, WastedCode, and Sun Weiwei. Image matching challenge 2023, 2023. 3 [7] Vassileios Balntas, Karel Lenc, Andrea Vedaldi, and Krystian Mikolajczyk. HPatches: A benchmark and evaluation andatd n earne ocal decp tors. In CVPR, 2017. 3 [8] Axel Barroso-Laguna, Edgar Riba, Daniel Ponsa, and Krystian Mikolajczyk. Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters. In ICCV, 2019. 3 [9] Yash Bhalgat, João F. Henriques, and Andrew Zisserman. A light touch approach to teaching transformers multi-view geometry. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, 2023. 3   
[10] Aritra Bhowmik, Stefan Gumhold, Carsten Rother, and Eric Brachmann. Reinforced feature points: Optimizing feature detection and description for a high-level task. In CVPR, 2020. 3   
[11] Georg Bökman and Fredrik Kahl. A case for using rotation invariant features in state of the art feature matchers. In CVPRW, 2022. 3   
[12] Yohann Cabon, Naila Murray, and Martin Humenberger. Virtual KITTI 2. CoRR, abs/2001.10773, 2020. 6   
[13] Neill D. F. Campbell, George Vogiatzis, Carlos Hernández, and Roberto Cipolla. Using multiple hypotheses to improve depth-maps for multi-view stereo. In ECCV, 2008. 10   
[14] Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M. M. Montiel, and Juan D. Tardós. Orbslam3: An accurate open-source library for visual, visualinertial, and multimap slam. IEEE Transactions on Robotics, 2021. 2   
[15] Devendra Singh Chaplot, Dhiraj Gandhi, Saurabh Gupta, Abhinav Gupta, and Ruslan Salakhutdinov. Learning to explore using active neural slam. arXiv preprint arXiv:2004.05155, 2020. 2   
[16] Hongkai Chen, Zixin Luo, Lei Zhou, Yurun Tian, Mingmin Zhen, Tian Fang, David McKinnon, Yanghai Tsin, and Long Quan. Aspanformer: Detector-free image matching with adaptive span transformer. European Conference on Computer Vision (ECCV), 2022. 3   
[17] Shuo Cheng, Zexiang Xu, Shilin Zhu, Zhuwen Li, Li Erran Li, Ravi Ramamoorthi, and Hao Su. Deep stereo using adaptive thin volume representation with uncertainty awareness. In CVPR, 2020. 10   
[18] Minsu Cho, Jungmin Lee, and Kyoung Mu Lee. Reweighted random walks for graph matching. In ECCV, 2010. 13   
[19] Gabriela Csurka, Christopher R. Dance, and Martin Humenberger. From Handcrafted to Deep Local Invariant Features. arXiv, 1807.10254, 2018. 3   
[20] Afshin Dehghan, Gilad Baruch, Zhuoyuan Chen, Yuri Feigin, Peter Fu, Thomas Gebauer, Daniel Kurz, Tal Dimry, Brandon Joffe, Arik Schwartz, and Elad Shulman. ARKitScenes: A diverse real-world dataset for 3d indoor scene understanding using mobile RGB-D data. In NeurIPS Datasets and Benchmarks, 2021. 6   
[21] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superpoint: Self-supervised Interest Point Detection and Description. In CVPR, 2018. 3, 8, 10   
[22] Qiaole Dong, Chenjie Cao, and Yanwei Fu. Rethinking optical flow from geometric matching consistent perspective. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. 3   
[23] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021. 4   
[24] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. The faiss library. 2024. 7   
[25] Richard Duda, Peter Hart, and David G.Stork. Pattern Classification. 2001. 7   
[26] Mihai Dusmanu, Ignacio Rocco, Tomás Pajdla, Marc Pollefeys, Josef Sivic, Akihiko Torii, and Torsten Sattler. D2-net: A trainable CNN for joint description and detection of local features. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, pages 80928101. Computer Vision Foundation / IEEE, 2019. 4   
[27] Johan Edstedt, Ioannis Athanasiadis, Mårten Wadenbäck, and Michael Felsberg. DKM: Dense kernelized feature matching for geometry estimation. In IEEE Conference on Computer Vision and Pattern Recognition, 2023. 3, 10   
[28] Johan Edstedt, Qiyu Sun, Georg Bökman, Märten Wadenbäck, and Michael Felsberg. RoMa: Robust Dense Feature Matching. arXiv preprint arXiv:2305.15404, 2023. 3   
[29] Ufuk Efe, Kutalmis Gokalp Ince, and Aydin Alatan. Dfm: A performance baseline for deep feature matching. In CVPRW, 2021. 3   
[30] Martin A. Fischler and Robert C. Bolles. Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography. Commun. ACM, 24(6):381395, 1981. 2   
[31] Yasutaka Furukawa and Jean Ponce. Accurate, dense, and robust multiview stereopsis. PAMI, 2010. 10   
[32] Silvano Galliani, Katrin Lasinger, and Konrad Schindler. Massively parallel multiview stereopsis by surface normal diffusion. In ICCV, 2015. 10   
[33] Hugo Germain, Guillaume Bourmaud, and Vincent Lepetit. S2DNet: Learning image features for accurate sparse-to-dense matching. In ECCV, 2020. 3   
[34] Leonardo Gomes, Olga Regina Pereira Bellon, and Luciano Silva. 3d reconstruction methods for digital preservation of cultural heritage: A survey. Pattern Recognit. Lett., 2014. 2   
[35] Lars Hammarstrand, Fredrik Kahl, Will Maddern, Tomas Pajdla, Marc Pollefeys, Torsten Sattler, Josef Sivic, Erik Stenborg, Carl Toft, and Akihiko Torii. Long-Term Visual Localization Benchmark. https : //www.visuallocalization.net/.3   
[36] Richard Hartley and Andrew Zisserman. Multiple View Geometry in Computer Vision. Cambridge University Press, 2004. 2, 7   
[37] Kun He, Yan Lu, and Stan Sclaroff. Local descriptors optimized for average precision. In CVPR, 2018. 3   
[38] Yihui He, Rui Yan, Katerina Fragkiadaki, and Shoou-I Yu. Epipolar transformers. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020, 2020. 3   
[39] Dan Hendrycks and Kevin Gimpel. Bridging nonlinearities and stochastic regularizers with gaussian error linear units. CoRR, abs/1606.08415, 2016. 5   
[40] Zhaoyang Huang, Xiaoyu Shi, Chao Zhang, Qiang Wang, Ka Chun Cheung, Hongwei Qin, Jifeng Dai, and Hongsheng Li. Flowformer: A transformer architecture for optical flow. In ECCV, 2022. 3   
[41] Martin Humenberger, Yohann Cabon, Nicolas Guerin, Julien Morat, Jérôme Revaud, Philippe Rerole, Noé Pion, Cesar de Souza, Vincent Leroy, and Gabriela Csurka. Robust image retrieval-based visual localization using kapture, 2020. 2, 10   
[42] Shihao Jiang, Dylan Campbell, Yao Lu, Hongdong Li, and Richard I. Hartley. Learning to estimate hidden motions with global motion aggregation. 2021. 3   
[43] Wei Jiang, Eduard Trulls, Jan Hosang, Andrea Tagliasacchi, and Kwang Moo Yi. COTR: Correspondence Transformer for Matching Across Images. In ICCV, 2021. 2, 3   
[44] Yuhe Jin, Dmytro Mishkin, Anastasiia Mishchuk, Jií Matas, Pascal Fua, Kwang Moo Yi, and Eduard Trulls. Image Matching across Wide Baselines: From Paper to Practice. IJCV, 2020. 3   
[45] Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 7(3):535547, 2019. 7   
[46] Ni Junjie, Li Yijin, Huang Zhaoyang, Li Hongsheng, Bao Hujun, Cui Zhaopeng, and Zhang Guofeng. Pats: Patch area transportation with subdivision for local feature matching. In CVPR, 2023. 3   
[47] Dominik A. Kloepfer, João F. Henriques, and Dylan Campbell. SCENES: Subpixel Correspondence Estimation With Epipolar Supervision, 2024. 3   
[48] Zhengqi Li and Noah Snavely. Megadepth: Learning single-view depth prediction from internet photos. In CVPR, pages 20412050, 2018. 6   
[49] Amy Lin, Jason Y. Zhang, Deva Ramanan, and Shubham Tulsiani. Relpose $^ { + + }$ : Recovering 6d poses from sparse-view observations. CoRR, abs/2305.04926, 2023. 10   
[50] Philipp Lindenberger, Paul-Edouard Sarlin, Viktor Larsson, and Marc Pollefeys. Pixel-perfect structure-frommotion with featuremetric refinement. In ICCV, 2021. 8, 10 [51] Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Pollefeys. Lightglue: Local feature matching at light speed. In ICCV, 2023. 2, 3, 10 [52] David G. Lowe. Distinctive Image Features from Scaleinvariant Keypoints. IJCV, 2004. 2, 3, 8 [53] Zixin Luo, Lei Zhou, Xuyang Bai, Hongkai Chen, Jiahui Zhang, Yao Yao, Shiwei Li, Tian Fang, and Long Quan. Aslfeat: Learning local features of accurate shape and localization. In CVPR, 2020. 3 [54] Jiayi Ma, Xingyu Jiang, Aoxiang Fan, Junjun Jiang, and Junchi Yan. Image matching from handcrafted to deep features: A survey. IJCV, 2021. 3 [55] Zeyu Ma, Zachary Teed, and Jia Deng. Multiview stereo with cascaded epipolar raft. In ECCV, 2022. 10 [56] Songrit Maneewongvatana and David M. Mount. Analysis of approximate nearest neighbor searching with clustered point sets. In DIMACS, 1999. 7 [57] N. Mayer, E. Ilg, P. Häusser, P. Fischer, D. Cremers, A. Dosovitskiy, and T. Brox. A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation. In CVPR, 2016. 6 [58] Iaroslav Melekhov, Aleksei Tiulpin, Torsten Sattler, Marc Pollefeys, Esa Rahtu, and Juho Kannala. DGCNet: Dense geometric correspondence network. In Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV), 2019. 3 [59] Dmytro Mishkin, Jiri Matas, Michal Perdoch, and Karel Lenc. Wxbs: Wide baseline stereo generalizations. In BMVC, 2015. 3 [60] Dmytro Mishkin, Filip Radenovic, and Jiri Matas. Repeatability is not enough: Learning affine regions via discriminability. In ECCV, 2018. 3 [61] Raul Mur-Artal, Jose Maria Martinez Montiel, and Juan D Tardos. Orb-slam: a versatile and accurate monocular slam system. IEEE transactions on robotics,   
2015.2 [62] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael G. Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 5 [63] Onur Özyeil, Vladislav Voroninski, Ronen Basri, and Amit Singer. A survey of structure from motion\*. Acta Numerica, 26:305364, 2017. 2 [64] MV Peppa, JP Mills, KD Fieber, I Haynes, S Turner, A Turner, M Douglas, and PG Bryan. Archaeological feature detection from archive aerial photography with a sfm-mvs and image enhancement pipeline. The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 42:869875,   
2018. 2   
[65] René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In ICCV, 2021. 5, 7   
[66] Anurag Ranjan and Michael J. Black. Optical flow estimation using a spatial pyramid network. In CVPR, 2017. 5   
[67] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In ICCV, 2021. 3, 6, 8, 10   
[68] Jerome Revaud, Philippe Weinzaepfel, Zaid Harchaoui, and Cordelia Schmid. EpicFlow: Edge-Preserving Interpolation of Correspondences for Optical Flow. In CVPR, 2015. 2   
[69] Jérôme Revaud, Philippe Weinzaepfel, Zaid Harchaoui, and Cordelia Schmid. DeepMatching: Hierarchical deformable dense matching. IJCV, 2016. 2   
[70] Jerome Revaud, Philippe Weinzaepfel, César Roberto de Souza, and Martin Humenberger. R2D2: repeatable and reliable detector and descriptor. In NIPS, 2019. 3   
[71] Ethan Rublee, Vincent Rabaud, Kurt Konolige, and Gary R. Bradski. ORB: an efficient alternative to SIFT or SURF. In ICCV, 2011. 3   
[72] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. SuperGlue: Learning feature matching with graph neural networks. In CVPR, 2020. 2, 3, 8, 10   
[73] Paul-Edouard Sarlin, Cesar Cadena, Roland Siegwart, and Marcin Dymczyk. From coarse to fine: Robust hierarchical localization at large scale. In CVPR, 2019. 3   
[74] Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, Devi Parikh, and Dhruv Batra. Habitat: A platform for embodied ai research. In ICCV, 2019. 6   
[75] Johannes Lutz Schönberger and Jan-Michael Frahm. Structure-from-motion revisited. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 2, 3,8   
[76] Johannes Lutz Schönberger, Enliang Zheng, Marc Pollefeys, and Jan-Michael Frahm. Pixelwise view selection for unstructured multi-view stereo. In ECCV, 2016. 8   
[77] Johannes L. Schönberger, Hans Hardmeier, Torsten Sattler, and Marc Pollefeys. Comparative Evaluation of Hand-Crafted and Learned Local Features. In CVPR, 2017. 3   
[78] Ishwar K. Sethi and Ramesh C. Jain. Finding trajectories of feature points in a monocular image sequence. IEEE TPAMI, 1987. 4   
[79] Xiaoyu Shi, Zhaoyang Huang, Weikang Bian, Dasong Li, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, Jifeng Dai, and Hongsheng Li. Videoflow: Exploiting temporal cues for multi-frame optical flow estimation. In ICCV, 2023. 3   
[80] Xiaoyu Shi, Zhaoyang Huang, Dasong Li, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, Jifeng Dai, and Hongsheng Li. Flowformer $^ { + + }$ : Masked cost volume autoencoding for pretraining optical flow estimation. In CVPR, 2023. 3   
[81] Jaime Spencer, Chris Russell, Simon Hadfield, and Richard Bowden. Kick back & relax $^ { + + }$ : Scaling beyond ground-truth depth with slowtv & cribstv. In ArXiv Preprint, 2024. 7   
[82] Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and Xiaowei Zhou. LoFTR: Detector-free local feature matching with transformers. CVPR, 2021. 2, 3, 7, 8, 10   
[83] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott Ettinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Yu Zhang, Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov. Scalability in perception for autonomous driving: Waymo open dataset. In CVPR, 2020.6   
[84] H. Taira, M. Okutomi, T. Sattler, M. Cimpoi, M. Pollefeys, J. Sivic, T. Pajdla, and A. Torii. InLoc: Indoor Visual Localization with Dense Matching and View Synthesis. PAMI, 2019. 3, 8, 9, 10, 11   
[85] Shitao Tang, Jiahui Zhang, Siyu Zhu, and Ping Tan. Quadtree attention for vision transformers. ICLR, 2022. 3   
[86] Zachary Teed and Jia Deng. RAFT: recurrent all-pairs field transforms for optical flow. In ECCV, 2020. 3, 5   
[87] Sebastian Thrun. Probabilistic robotics. Communications of the ACM, 45(3):5257, 2002. 2   
[88] Yurun Tian, Xin Yu, Bin Fan, Fuchao Wu, Huub Heijnen, and Vassileios Balntas. Sosnet: Second order similarity regularization for local descriptor learning. In CVPR, 2019. 3   
[89] Carl Toft, Daniyar Turmukhambetov, Torsten Sattler, Fredrik Kahl, and Gabriel J. Brostow. Single-image depth prediction makes feature matching easier. In ECCV, 2020. 3   
[90] Engin Tola, Christoph Strecha, and Pascal Fua. Efficient large-scale multi-view stereo for ultra highresolution image sets. Mach. Vis. Appl., 2012. 10   
[91] Fabio Tosi, Yiyi Liao, Carolin Schmitt, and Andreas Geiger. Smd-nets: Stereo mixture density networks. In Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 6   
[92] Prune Truong, Martin Danelljan, and Radu Timofte. GLU-Net: Global-local universal network for dense flow and correspondences. In CVPR, 2020. 3   
[93] Prune Truong, Martin Danelljan, Luc Van Gool, and Radu Timofte. Learning accurate dense correspondences and when to trust them. In CVPR, 2021. 3   
[94] Prune Truong, Martin Danelljan, Radu Timofte, and Luc Van Gool. Pdc-net+: Enhanced probabilistic dense correspondence network. IEEE TPAMI. 2023. 3 [95] Aäron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learningwithcontrastive predictiv cding. CoRR, abs/1807.03748, 2018. 5 [96] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 2 [97] Yannick Verdie, Kwang Moo Yi, Pascal Fua, and Vincent Lepetit. TILDE: A temporally invariant learned detector. In CVPR, 2015. 3 [98] Bing Wang, Changhao Chen, Zhaopeng Cui, Jie Qin, Chris Xiaoxuan Lu, Zhengdi Yu, Peijun Zhao, Zhen Dong, Fan Zhu, Niki Trigoni, and Andrew Markham. P2-net: Joint description and detection of local features for pixel and point matching. In ICCV, 2021.   
3 [99] Fangjinhua Wang, Silvano Galliani, Christoph Vogel, Pablo Speciale, and Marc Pollefeys. Patchmatchnet: Learned multi-view patchmatch stereo. In CVPR, pages   
1419414203, 2021. 9, 10 [100] Jianyuan Wang, Christian Rupprecht, and David Novotny. PoseDiffusion: Solving pose estimation via diffusion-aided bundle adjustment. 2023. 3, 8, 10 [101] Qianqian Wang, Xiaowei Zhou, Bharath Hariharan, Noa Savly. La Feat Depor sn Camera Pose Supervision. In ECCV, 2020. 3 [102] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric   
3d vision made easy, 2023. 2, 3, 4, 6, 8, 10, 17 [103] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. Tartanair: A dataset to push the limits of visual slam. 2020.6 [104] Philippe Weinzaepfel, Thomas Lucas, Vincent Leroy, Yohann Cabon, Vaibhav Arora, Romain Brégier, Gabriela Csurka, Leonid Antsfeld, Boris Chidlovskii, and Jérôme Revaud. CroCo v2: Improved Cross-view Completion Pre-training for Stereo Matching and Op tical Flow. In ICCV, 2023. 6 [105] Changchang Wu. VisualSFM: A Visual Structure from Motion System. http://ccwu.me/vsfm/, 2011. 3 [106] Hao Wu, Aswin C. Sankaranarayanan, and Rama Chellappa. Cvpr. 2007. 4 [107] Qingshan Xu and Wenbing Tao. Learning inverse depth regression for multi-view stereo with correlation cost volume. In AAAI, 2020. 10 [108] Guandao Yang, Tomasz Malisiewicz, and Serge J. Belongie. Learning data-adaptive interest points through epipolar adaptation. In CVPR Workshops, 2019. 3 [109] Jiayu Yang, Wei Mao, José M. Álvarez, and Miaomiao Liu. Cost volume pyramid based depth inference for multi-view stereo. In CVPR, pages 48764885, 2020.   
10 [110] Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, and Long Quan. Mvsnet: Depth inference for unstructured multiview stereo. In ECCV, 2018. 10   
[111] Yuan Yao, Yasamin Jafarian, and Hyun Soo Park. MONET: multiview semi-supervised keypoint detection via epipolar divergence. In ICCV, 2019. 3   
[112] Yao Yao, Zixin Luo, Shiwei Li, Jingyang Zhang, Yufan Ren, Lei Zhou, Tian Fang, and Long Quan. Blendedmvs: A large-scale dataset for generalized multi-view stereo networks. In CVPR, 2020. 6   
[113] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Niener, and Angela Dai. Scannet $^ { + + }$ : A high-fidelity dataset of 3d indoor scenes. In Proceedings of the International Conference on Computer Vision (ICCV), 2023. 6   
[114] Wang Yifan, Carl Doersch, Relja Arandjelovic, João Carreira, and Andrew Zisserman. Input-level inductive biases for 3d reconstruction. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, 2022. 3   
[115] Jason Y. Zhang, Deva Ramanan, and Shubham Tulsiani. Relpose: Predicting probabilistic relative rotation for single objects in the wild. In ECCV, 2022. 8, 10   
[116] Jason Y Zhang, Amy Lin, Moneish Kumar, Tzu-Hsuan Yang, Deva Ramanan, and Shubham Tulsiani. Cameras as rays: Pose estimation via ray diffusion. In International Conference on Learning Representations (ICLR), 2024. 3, 8, 10   
[117] Xu Zhang, Felix X. Yu, Svebor Karaman, and Shih-Fu Chang. Learning discriminative and transformation covariant local feature detectors. In CVPR, 2017. 3   
[118] Zichao Zhang, Torsten Sattler, and Davide Scaramuzza. Reference pose generation for long-term visual localization via learned features and view synthesis. IJCV, 2021. 3, 8, 10, 11, 16   
[119] Zhe Zhang, Rui Peng, Yuxi Hu, and Ronggang Wang. Geomvsnet: Learning multi-view stereo with geometry perception. In CVPR, 2023. 10   
[120] Qunjie Zhou, Torsten Sattler, and Laura Leal-Taixe. Patch2pix: Epipolar-guided pixel-level correspondences. In CVPR, 2021. 3   
[121] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. SIGGRAPH, 2018. 8, 10   
[122] Shengjie Zhu and Xiaoming Liu. Pmatch: Paired masked image modeling for dense geometric matching. In CVPR, 2023. 3