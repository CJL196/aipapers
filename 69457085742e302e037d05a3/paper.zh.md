# GaussianDreamer：通过桥接二维和三维扩散模型从文本快速生成三维高斯分布

Taoran 余¹，贾敏 方²‡，俊杰 王²，冠军 吴³，灵溪 谢²，晓鹏 张²，文宇 刘¹，齐 天²，兴刚 王¹# 1华中科技大学电子信息学院 2华为公司 3华中科技大学计算机学院

![](images/1.jpg)  
F and achieve real-time rendering.

# 摘要

最近，基于文本提示生成3D资产的技术取得了显著成果。2D和3D扩散模型都可以根据提示生成不错的3D对象。3D扩散模型具有良好的3D一致性，但其质量和泛化能力有限，因为可用于训练的3D数据昂贵且难以获取。2D扩散模型则具备强大的泛化能力和细致生成能力，但难以保证3D一致性。本文试图通过最近提出的明确而高效的$3D$高斯点云表示来架起这两类扩散模型之间的桥梁。我们提出了一种快速的3D对象生成框架，名为GaussianDreamer，其中3D扩散模型提供初始化的先验，2D扩散模型则丰富了几何形状和外观。引入了噪声点生长和颜色扰动操作，以增强初始化的高斯分布。我们的GaussianDreamer能够在一台GPU上在15分钟内生成高质量的3D实例或3D头像，速度远快于之前的方法，同时生成的实例可以实时渲染。示例和代码可在https://taoranyi.com/gaussiandreamer/获取。

# 1. 引言

3D资产生成在传统流程中一直是一项成本高昂且专业性强的工作。最近，扩散模型在生成高质量和逼真的2D图像方面取得了巨大成功。许多研究工作尝试将2D扩散模型的强大能力转移到3D领域，以简化和辅助3D资产的创建过程，例如最常见的文本到3D任务。

实现该目标的主要有两条路线：（i）使用3D数据训练新的扩散模型[12, 15, 25, 51]（即3D扩散模型），以及（ii）将2D扩散模型提升至3D[2, 6, 9, 34, 55, 62, 72, 75, 81, 82, 86, 95]。前者实现直接，具备强大的3D一致性，但在扩展到大型生成领域时遇到困难，因为3D数据通常难以获取且成本高昂。目前3D数据集的规模远小于2D数据集。这导致生成的3D资产在处理复杂文本提示和生成复杂/精细的几何形状及外观方面表现不足。后者则得益于2D扩散模型的大数据领域，能够处理各种文本提示并生成高细节和复杂的几何形状及外观。然而，由于2D扩散模型无法感知相机视角，生成的3D资产在形成几何一致性方面困难重重，特别是对于结构复杂的实例。

本文提出使用最近的3D高斯点云技术来连接前述两种方法，同时兼具3D扩散模型的几何一致性和2D扩散模型的丰富细节。3D高斯是一种高效且明确的表示方法，由于其点云状结构，固有地享有几何先验。具体而言，我们使用两种3D扩散模型中的一种：文本到3D和文本到运动扩散模型，例如在我们的实现中使用的Shap-E和MDM，生成粗略的3D实例。基于粗略的3D实例，初始化一组3D高斯。我们引入了噪声点生长和颜色扰动两个操作，以补充初始化的高斯，为后续丰富3D实例提供支持。随后，通过得分蒸馏采样损失（SDS）与2D扩散模型交互，从而改进和优化3D高斯。由于来自3D扩散模型和3D高斯点云本身的几何先验，训练过程可以在很短的时间内完成。生成的3D资产可以在实时渲染中使用，而无需通过点云过程转换为网格等结构。我们的贡献可以总结如下：• 我们提出了一种名为GaussianDreamer的文本到3D方法，通过高斯分割连接3D和2D扩散模型，享受3D一致性和丰富生成细节。 • 引入噪声点生长和颜色扰动，以补充初始化的3D高斯，实现进一步内容丰富化。 • 整体方法简单且非常有效。在一台GPU上可以在15分钟内生成一个3D实例，速度远快于以前的方法，并可以直接进行实时渲染。

# 2. 相关工作

3D 预训练扩散模型。最近，基于扩散模型的文本到 3D 资产生成取得了巨大的成功。目前，主要分为将 2D 扩散模型提升到 3D 以及 3D 预训练扩散模型，二者的区别在于所使用的训练数据是 2D 还是 3D。3D 预训练扩散模型 [12, 15, 25, 51]，在我们论文中称为 3D 扩散模型，是在文本-3D 对上预训练的模型。经过预训练后，它们可以仅通过推理生成 3D 资产，如 Point-E [51] 和 Shape-E [25] 等模型可以在几分钟内生成 3D 资产。除了从文本生成 3D 资产，还有一些方法，其中 3D 扩散模型 [1, 7, 10, 27, 58, 68, 77, 93, 94, 96] 基于文本-运动数据生成运动序列。通过在文本-运动对上进行预训练，这些模型能够为不同的文本生成合理的运动序列。生成的运动序列可以基于网格表示转换为 SMPL（Skinned MultiPerson Linear）模型 [41]，但不包括纹理信息。在我们的方法中，可以通过使用不同的文本提示对转换后的 SMPL 进行绘图。

将2D扩散模型提升到3D。在文本到3D资产生成方法中，除了使用3D预训练扩散模型外，将2D扩散模型提升到3D是一种无训练的方法。此外，由于2D图像数据的丰富性，该方法能够生成多样性和保真度更高的资产。一些单图像到3D的方法也采用了类似的思路。DreamFusion首次提出SDS（评分蒸馏采样）方法，通过使用2D扩散模型更新3D表示模型。随后提出的SJC（评分雅各布链）方法则是将2D扩散模型提升到3D。后续方法基于DreamFusion进一步提高了3D生成的质量。其中，生成的3D资产可能受到多面问题的影响。为了解决这个问题，一些方法强化了不同视角的语义，并使用多视角信息来缓解这些问题。还有一些模型采用CLIP来将3D表示模型的每个视角与文本对齐。3D表示方法。近年来，神经辐射场（NeRF）在3D表示方面取得了令人瞩目的结果，许多文本到3D资产生成的方法也采用了NeRF或其变体作为表示方法。一些方法使用显式可优化的网格表示方法，如DMTET，以降低渲染成本并进一步提高分辨率。除此之外，还有一些生成方法利用点云和网格作为3D表示。最近，3D高斯喷涂被引入作为3D场景的表示方法，可以实现与基于NeRF的方法相媲美的渲染效果，并支持实时渲染。两个并行的工作也使用3D高斯喷涂构建3D表示。DreamGaussian使用单幅图像作为条件生成3D资产。

![](images/2.jpg)  
F clouds. In this case, we take text-to-3D and text-to-motion diffusion models as examples.

GEN [9] 实现了从文本到 3D 的高质量生成。我们的方法采用了类似的思路，使用 3D 高斯点云作为表示方法，这显著降低了提升分辨率的成本，并与可优化网格表示方法相比，实现了更快的优化速度。我们可以基于提示文本在非常短的时间内生成详细的 3D 资产。

# 3. 方法

在本节中，我们首先回顾二维和三维扩散模型以及三维表示方法——三维高斯溅射 [26]。我们在第3.2节中概述整个框架。然后，在第3.3节中，我们描述在三维扩散模型的辅助下初始化三维高斯的过程。在第3.4节中，我们描述使用二维扩散模型进一步优化三维高斯的过程。

# 3.1. 准备工作

DreamFusion。DreamFusion [55] 是将 2D 扩散模型提升到 3D 的最具代表性的方法之一，提出通过预训练的 2D 扩散模型 $\phi$ 通过得分蒸馏采样（SDS）损失优化 3D 表示。具体而言，它采用 MipNeRF [3] 作为 3D 表示方法，并优化其参数 $\theta$。设渲染方法为 $g$，则渲染后的图像结果为 $\mathbf{x} = g(\theta)$。为了使渲染图像 $\mathbf{x}$ 与从扩散模型 $\phi$ 获得的样本相似，DreamFusion 使用了一个评分估计函数：$\hat{\epsilon}_{\phi}(\mathbf{z}_{t}; y, t)$，该函数根据噪声图像 $\mathbf{z}_{t}$、文本嵌入 $y$ 和噪声水平 $t$ 预测采样噪声 $\hat{\epsilon}_{\phi}$。通过测量添加到渲染图像 $\mathbf{x}$ 的高斯噪声 $\epsilon$ 与预测噪声 $\hat{\epsilon}_{\phi}$ 之间的差异，该评分估计函数能够提供更新参数 $\theta$ 的方向。计算梯度的公式为，其中 $w(t)$ 是加权函数。

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { S D S } } ( \boldsymbol { \phi } , \mathbf { x } = g ( \boldsymbol { \theta } ) ) \triangleq \mathbb { E } _ { t , \epsilon } \left[ w ( t ) \left( \hat { \epsilon } _ { \boldsymbol { \phi } } ( \mathbf { z } _ { t } ; \boldsymbol { y } , t ) - \epsilon \right) \frac { \partial \mathbf { x } } { \partial \boldsymbol { \theta } } \right] ,
$$

3D高斯渲染。3D高斯渲染[26]（3DGS）是一种用于新视角合成的最新突破性方法。与基于体积渲染的隐式表示方法（例如NeRF [47]）不同，3D-GS通过渲染点（splatting）[89]实现图像的快速渲染，达到实时速度。具体来说，3D-GS通过一组各向异性的高斯函数来表示场景，定义包括其中心位置$\mu \in \mathbb { R } ^ { 3 }$、协方差$\Sigma \ \in \ \mathbb { R } ^ { 7 }$、颜色$c \in \mathbb { R } ^ { 3 }$和不透明度$\boldsymbol { \alpha } \in \mathbb { R } ^ { 1 }$。3D高斯函数可以通过以下方式查询：

$$
G ( x ) = e ^ { - \frac { 1 } { 2 } ( x ) ^ { T } \Sigma ^ { - 1 } ( x ) } ,
$$

其中 $x$ 表示 $\mu$ 与查询点之间的距离。为了计算每个像素的颜色，它使用了典型的基于神经网络的点渲染方法 [28, 29]。从相机中心发出一条光线 $r$，并沿着光线计算光线与 3D 高斯体相交的颜色和密度。渲染过程如下：

$$
C ( r ) = \sum _ { i \in \mathcal { N } } c _ { i } \sigma _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \sigma _ { j } ) , \quad \sigma _ { i } = \alpha _ { i } G ( x _ { i } ) ,
$$

其中 $\mathcal{N}$ 表示光线 $r$ 上的采样点数量，$c_{i}$ 表示第 $i$ 个高斯的颜色，$\alpha_{i}$ 表示第 $i$ 个高斯的不透明度，$x_{i}$ 是点与第 $i$ 个高斯之间的距离。

# 3.2. 全局框架

我们的整体框架由两个部分组成：使用3D扩散模型先验进行初始化，以及使用2D扩散模型进行优化，如图2所示。在3D扩散模型先验的初始化中，我们利用文本到3D和文本到动作的扩散模型构建的3D扩散模型 $F _ { 3 D }$ ，根据文本提示 $y$ 生成三角网格 $m$ ，可以表示为 $m = F _ { 3 D } ( y )$ 。生成的一组点云是从网格 $m$ 转换而来的。然后，经过噪声点扩展和颜色扰动后，初始化3D高斯 $\theta _ { b }$ 。为了获得更好的质量，我们利用2D扩散模型 $F _ { 2 D }$ 通过SDS [55] 进一步优化初始化的3D高斯 $\theta _ { b }$ ，以提示 $y$ 生成最终的3D高斯 $\theta _ { f }$ 。通过对生成的高斯进行点云渲染，可以实时渲染目标实例。

# 3.3. 基于三维扩散模型先验的高斯初始化

在本节中，我们主要讨论如何使用 3D 扩散模型先验来初始化 3D 高斯分布。首先，我们使用 3D 扩散模型 $F _ { 3 D }$ 根据提示 $y$ 生成 3D 资产。然后，我们将 3D 资产转换为点云，并使用转换后的点云来初始化 3D 高斯分布。我们采用了两种类型的 3D 扩散模型来生成 3D 资产。接下来，我们解释如何使用这些模型初始化 3D 高斯分布。

# 3.3.1 文本到三维扩散模型

在使用基于文本的3D生成模型时，生成的3D资产使用多层感知器（MLPs）来预测SDF值和纹理颜色。为了构建三角网格$m$，我们在MLP中沿着一个尺寸为$1 2 8 ^ { 3 }$的规则网格查询顶点的SDF值。然后，我们在$m$的每个顶点查询纹理颜色。我们将$m$的顶点和颜色转换为点云，记作$pt_{m}(p_{m}, c_{m})$，其中$\pmb{p}_{m} \in \mathbb{R}^{3}$表示点云的位置，与$m$的顶点坐标相等，$\boldsymbol{c}_{m} \in \mathbb{R}^{3}$表示点云的颜色，与$m$的颜色相同。然而，获得的颜色$c_{m}$相对简单，而位置$\pmb{p}_{m}$则较为稀疏。

噪声点生长和颜色扰动。我们不直接使用生成的点云 ${ \pmb p } { \pmb t } _ { m }$ 来初始化 3D 高斯分布。为了提高初始化的质量，我们在点云 ${ p t } _ { m }$ 周围执行噪声点生长和颜色扰动。首先，我们计算 ${ p t } _ { m }$ 上表面的包围盒（BBox），然后在 BBox 内均匀增长点云 $p t _ { r } ( p _ { r } , c _ { r } )$。${ \pmb p } _ { r }$ 和 $c _ { r }$ 表示 ${ \mathbf { \Omega } } _ { p t _ { r } }$ 的位置和颜色。为了实现快速搜索，我们使用位置 ${ \pmb p } _ { m }$ 构建 KDTree [4] $K _ { m }$。根据位置 $\pmb { p } _ { r }$ 和在 KDTree $K _ { m }$ 中找到的最近点之间的距离，我们决定保留哪些点。在此过程中，我们选择距离（归一化后）为 0.01 内的点。对于噪声点云，我们使其颜色 $c _ { r }$ 类似于 $c _ { m }$，并添加一些扰动：

![](images/3.jpg)  
Figure 3. The process of noisy point growing and color perturbation. "Grow&Pertb." denotes noisy point growing and color perturbation.

$$
\begin{array} { r } { \pmb { c } _ { r } = \pmb { c } _ { m } + \pmb { a } , } \end{array}
$$

其中 $^{ a }$ 的值在 0 到 0.2 之间随机采样。我们合并 ${ p t } _ { m }$ 和 ${ p t } _ { r }$ 的位置和颜色，以获得最终的点云。

$$
p t ( p _ { f } , c _ { f } ) = ( p _ { m } \oplus p _ { r } , c _ { m } \oplus c _ { r } ) ,
$$

其中 $\oplus$ 是连接操作。图 3 说明了噪声点增长和颜色扰动的过程。最后，我们使用最终点云 $_{ p t }$ 的位置 ${ \pmb p } _ { f }$ 和颜色 $c _ { f }$ 初始化 3D 高斯 $\theta _ { b } ( \mu _ { b } , c _ { b } , \Sigma _ { b } , \alpha _ { b } )$ 的位置 $\mu _ { b }$ 和颜色 $c _ { b }$。3D 高斯的透明度 $\alpha _ { b }$ 初始化为 0.1，协方差 $\Sigma _ { b }$ 计算为最近两个点之间的距离。算法 1 显示了具体的算法流程图。

# 3.3.2 文本到运动扩散模型

我们使用文本生成一系列人体运动，并选择最符合给定文本的人体姿态。接着，我们将该姿态的关键点转换为SMPL模型[41]，该模型由三角网格$m$表示。然后，我们将网格$m$转换为点云$p t _ { m } ( p _ { m } , c _ { m } )$，其中点云中每个点的位置${ \pmb p } _ { m }$对应于$m$的顶点。至于${ \pmb p } { \pmb t } _ { m }$的颜色$c _ { m }$，由于这里使用的SMPL模型没有纹理，我们随机初始化$c _ { m }$。为使${ \pmb p } { \pmb t } _ { m }$接近原点，我们计算点云${ \pmb p } _ { m }$的中心点$\pmb { p } _ { c } \in \mathbb { R } ^ { 3 }$，并从中心点${ \pmb p } _ { c }$中减去点云${ p t } _ { m }$的位置。

$$
\begin{array} { r } { p t ( { p } _ { f } , \pmb { c } _ { f } ) = p t _ { m } ( { p } _ { m } - { p } _ { c } , \pmb { c } _ { m } ) , } \end{array}
$$

最后，我们使用点云$_{pt}$来初始化3D高斯分布，类似于第3.3.1节中所描述的内容。为了改善运动序列的生成，我们通过保留与运动相关的部分来简化文本，并添加一个主体。例如，如果文本提示是“钢铁侠用左腿踢”，我们在生成运动序列时将其转化为“某人用左腿踢”。

<table><tr><td>Algorithm 1 The 3D Gaussian Initialization. ptm (pm, cm): Point clouds generated from F3D. ptr (pr, cr): Growing point clouds within the BBox. pt(pf , cf ): Point clouds used for initializing the 3D Gaus- sians. θb(µb, cb, ∑b, αb): Initialized 3D Gaussians.</td></tr><tr><td>Stage 1: Generate points in the BBox. Km ← BuildKDTree(pm) KDTree BBox ← pm Positions bounding box. Low, High ← BBox.MinBound, BBox.MaxBound Boundary of bounding box. psu ← Uniform(Low, High, size = (NumPoints, 3)) Points in the BBox.</td></tr><tr><td>Stage 2: Keep the points that meet the distance require- ment. pr, Cr = [], []</td></tr><tr><td>for all pu in psu do pun, i ← Km.SearchNearest(pu) Nearest point and its index in Km. if |pun − pu| &lt; 0.01 then pr.append(pu) c.append(cm [i] + 0.2×Random(size = 3)) Color of the nearest point plus perturbation. end if end for</td></tr></table>

# 3.4. 使用二维扩散模型进行优化

为了丰富细节并提高3D资产的质量，我们在使用3D扩散模型先验初始化3D高斯$\theta _ { b }$之后，利用2D扩散模型$F _ { 2 D }$来优化这些高斯。我们采用SDS（得分蒸馏采样）损失来优化3D高斯。首先，我们使用3D高斯喷溅方法[26]获得渲染图像$\mathbf { x } = g ( \theta _ { i } )$。这里，$g$表示如公式3所示的喷溅渲染方法。然后，我们使用公式1计算梯度，以便使用2D扩散模型$F _ { 2 D }$更新高斯参数$\theta _ { i }$。在使用2D扩散模型$F _ { 2 D }$进行短期优化后，最终生成的3D实例$\theta _ { f }$在3D扩散模型$F _ { 3 D }$提供的3D一致性基础上实现了高质量和高保真度。

Table 1. Quantitative comparisons on $\mathrm { T } ^ { 3 }$ Bench [17].   

<table><tr><td>Method</td><td>Time†</td><td colspan="4">Single Obj. Single w/ Surr. Multi Obj. Average</td></tr><tr><td>SJC [81]</td><td></td><td>24.7</td><td>19.8</td><td>11.7</td><td>18.7</td></tr><tr><td>DreamFusion [55]</td><td>6 hours</td><td>24.4</td><td>24.6</td><td>16.1</td><td>21.7</td></tr><tr><td>Fantasia3D [6]</td><td>6 hours</td><td>26.4</td><td>27.0</td><td>18.5</td><td>24.0</td></tr><tr><td>LatentNeRF [45]</td><td>15 minutes</td><td>33.1</td><td>30.6</td><td>20.6</td><td>28.1</td></tr><tr><td>Magic3D [34]</td><td>5.3 hours</td><td>37.0</td><td>35.4</td><td>25.7</td><td>32.7</td></tr><tr><td>ProlificDreamer [82]</td><td>∼10 hours</td><td>49.4</td><td>44.8</td><td>35.8</td><td>43.3</td></tr><tr><td>Ours</td><td>15 minutes</td><td>54.0</td><td>48.6</td><td>34.5</td><td>45.7</td></tr></table>

上述分数是两个指标（质量和对齐）的平均值。† GPU 时间在他们的论文中计入。

# 4. 实验

在本节中，我们首先在第4.1节中介绍实现细节。接着，在第4.2节中，我们展示定量比较结果。然后，在第4.3节中，我们展示我们方法的可视化结果，并与其他方法进行比较。在第4.4节中，我们进行了一系列消融实验，以验证我们方法的有效性。最后，我们讨论我们方法的局限性。

# 4.1. 实现细节

我们的方法是在 PyTorch [54] 上实现的，基于 ThreeStudio [14]。我们方法中使用的 3D 扩散模型是 Shap-E [25] 和 MDM [77]，并且我们加载了在 Objaverse [11] 上微调的 Shap-E 模型于 Cap3D [44]。对于 2D 扩散模型，我们使用 stabilityai/stablediffusion-2-1-base [62]，引导比例设为 100。我们使用的时间戳均匀采样自 0.02 到 0.98，在 500 次迭代之前，500 次迭代后变更为 0.02 到 0.55。对于 3D 高斯，透明度 $\alpha$ 和位置 $\mu$ 的学习率分别为 $1 \times 10^{-2}$ 和 $5 \times 10^{-5}$。3D 高斯的颜色 $c$ 由 sh 系数表示，度数设为 0，学习率设为 $1.25 \times 10^{-2}$。3D 高斯的协方差被转化为缩放和旋转进行优化，学习率分别为 $1 \times 10^{-3}$ 和 $1 \times 10^{-2}$。我们用于渲染的相机半径范围为 1.5 到 4.0，方位角范围为 -180 到 180 度，俯仰角范围为 -10 到 60 度。总的训练迭代次数为 1200。所有实验在单个 RTX 3090 上以批处理大小为 4 完成需时不超过 15 分钟。我们用于渲染的分辨率为 $1024 \times 1024$，在使用 2D 扩散模型优化时缩放至 $512 \times 512$。我们可以以 $512 \times 512$ 的分辨率实时渲染。所有代码将会公开。

# 4.2. 定量评估

我们按照 $\mathrm { T ^ { 3 } }$ Bench [17] 评估质量和一致性，该基准为文本到三维生成提供了全面的评测。设计了三种文本类别用于三维生成，复杂性逐渐增加，包括单个物体、带有环境的单个物体和多个物体。在表1中，我们的方法在生成时间较短的同时优于其他对比方法。

![](images/4.jpg)  
A 3D model of an adorable cottage with a thatched roof.

![](images/5.jpg)  
Fi is measured on RTX 3090, and our method is measured on RTX 3090.   

魔法匕首，神秘的，古老的……一张放大缩小的单反相机拍摄的狮鬃水母照片 图5. 我们的GaussianDreamer生成的更多样本。每个样本展示了两个视角。

# 4.3. 可视化结果

在本节中，我们展示了使用两种不同的3D扩散模型（文本到3D和文本到运动扩散模型）初始化3D高斯的结果。

使用文本到3D扩散模型进行初始化。我们在图4中展示了与DreamFusion [55]、Magic3D [34]、Fantasia3D [6]和ProlificDreamer [82]的对比结果。除了我们的方法外，其他方法的图像均来自ProlificDreamer的论文。当遇到涉及多个对象组合的提示时，例如提示“一个堆满巧克力饼干的盘子”，Magic3D、Fantasia3D和ProlificDreamer生成的结果并不包括盘子。相比之下，我们生成的结果可以有效地将盘子和巧克力饼干结合在一起。此外，与DreamFusion相比，我们生成的盘子具有更好的图案。我们的方法在质量上表现相当，同时生成时间节省了$2 1 - 2 4$倍。此外，我们的方法生成的3D高斯可以直接实现实时渲染，无需进一步转换为网状结构。图5展示了我们的GaussianDreamer根据各种提示生成的更多样本，这些样本在具有高质量细节的同时显示出良好的3D一致性。

使用文本到运动扩散模型的初始化。在图6中，我们展示了与DreamFusion [55]、DreamAvatar [5]、DreamWaltz [20] 和 AvatarVerse [91] 的比较结果。除了我们的方法，其他方法的图像是从AvatarVerse的论文中下载的。值得注意的是，我们的提示语为“蜘蛛侠/风暴兵张开双臂”，而其他方法的提示语为“蜘蛛侠/风暴兵”。这是因为在使用文本到运动扩散模型作为初始化生成运动时，我们需要更具体的动作描述。与其他方法相比，我们的方法实现了 $4 - 2 4$ 倍的速度提升，同时保持了可比的质量。此外，我们的方法还允许生成具有指定体姿的3D虚拟形象。在图7中，我们提供了使用不同人体姿势生成的更多结果。我们首先使用文本到运动3D扩散模型生成与文本提示相匹配的一系列动作，然后用运动中选定姿势的SMPL初始化3D高斯分布。我们的方法可以生成任意所需姿势的3D虚拟形象。

![](images/6.jpg)  
Stormtrooper 0 .

![](images/7.jpg)  
Figure 7. More generated 3D avatars by our GaussianDreamer initialized with the different poses of SMPL [41]. Here, the different poses of SMPL are generated using a text-to-motion diffusion model.

# 4.4. 消融研究与分析

初始化的作用。如图8所示，我们首先对3D高斯分布的初始化进行消融实验，以验证初始化可以改善3D一致性。第一列是使用NeRF [47]作为3D表示的Shap-E [25]的渲染结果。第二列是使用SDS损失优化随机初始化在立方体内的3D高斯分布的结果，第三列是我们的方法。我们在3个示例上展示了初始化效果。在第一行和第二行中，ShapE的生成结果较好，而我们的方法提供了更复杂的几何形状和更真实的外观。与随机初始化相比，在第一行中，我们方法的细节更佳。在第二行中，随机初始化生成的3D资产存在多头问题，而我们的做法没有出现这种问题。3D扩散模型的初始化可以避免不合理的几何形状。在第三行中，Shap-E的生成结果与给定文本提示相差甚远，而我们的方法通过2D扩散模型使3D资产更接近提示。我们的方法可以扩展Shap-E提示的领域，使其能够基于更广泛的提示生成3D资产。 噪声点生长和颜色扰动。图9展示了噪声点生长和颜色扰动的消融结果。应用噪声点生长和颜色扰动后，第一行展示了狙击步枪的细节有所改善。此外，第二列生成的毛绒摩托车更好地与提示中提到的毛绒玩具风格特征对齐，相较于没有噪声点生长和颜色扰动的情况。 使用不同的文本到3D扩散模型的初始化。我们选择了两个文本到3D生成模型，ShapE [25]和Point-E [51]，来验证我们框架的有效性。我们在Cap3D [44]中加载了在Objaverse [11]上微调的Point-E模型。图10中，我们展示了使用两种文本到3D生成模型之一初始化3D高斯分布后的生成结果。可以看出，两种初始化都产生了良好的生成结果。但是，考虑到Shap-E基于NeRF和SDF生成3D资产，提供的真实感比Point-E使用的点云表示更高，因此在图10的第一行中，使用Shap-E初始化的飞机几何形状更佳。

![](images/8.jpg)  
ur Alai sudie  thelization f the  Gaussans. The Shap- [5] renerig roluton $2 5 6 \mathbf { x } 2 5 6 .$

![](images/9.jpg)  
Figure 9. Ablation studies of noisy point growing and color perturbation. "Grow&Pertb." denotes noisy point growing and color perturbation.

![](images/10.jpg)  
Figure 10. Ablation studies of initialization with different text-to3D diffusion models: Point-E [51] and Shap-E [25].

# 4.5. 限制性因素

我们的方法生成的3D资产边缘并不总是锐利的，物体表面可能存在多余的3D高斯分布。如何过滤这些点云将是一个可能的改进方向。我们的方法利用了3D扩散模型先验，这在很大程度上缓解了多面体问题。然而，在几何差异最小而物体前后外观差异显著的场景中，例如背包，仍然有小概率遇到多面体问题。利用3D感知扩散模型可能能够解决此问题。此外，我们的方法在生成大规模场景（如室内场景）方面的有效性有限。

# 5. 结论

我们提出了一种快速的文本到3D方法GaussianDreamer，通过高斯点云表示将3D和2D扩散模型的能力结合起来。GaussianDreamer能够生成详细而真实的几何形状和外观，同时保持3D一致性。3D扩散模型的先验与来自3D高斯分布的几何先验有效地促进了收敛速度。每个样本可以在一块GPU上在15分钟内生成。我们相信，结合3D和2D扩散模型的方法可能是高效生成3D资产的一个有前途的方向。

# 致谢

本研究得到了中国国家自然科学基金的支持（编号：62376102）。

# References

[1] Tenglong Ao, Zeyi Zhang, and Libin Liu. Gesturediffuclip: Gesture diffusion model with clip latents. arXiv preprint arXiv:2303.14613, 2023. 2   
[2] Mohammadreza Armandpour, Huangjie Zheng, Ali Sadeghian, Amir Sadeghian, and Mingyuan Zhou. Reimagine the negative prompt algorithm: Transform 2d diffusion into 3d, alleviate janus problem and beyond. arXiv preprint arXiv:2304.04968, 2023. 1, 2   
[3] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In ICCV, pages 58555864, 2021. 2, 3   
[4] Jon Louis Bentley. Multidimensional binary search trees used for associative searching. Communications of the ACM, 18(9):509517, 1975. 4   
[5] Yukang Cao, Yan-Pei Cao, Kai Han, Ying Shan, and KwanYee K Wong. Dreamavatar: Text-and-shape guided 3d human avatar generation via diffusion models. arXiv preprint arXiv:2304.00916, 2023. 6, 7   
[6] Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation. arXiv preprint arXiv:2303.13873, 2023. 1, 2, 5, 6   
[7] Xin Chen, Biao Jiang, Wen Liu, Zilong Huang, Bin Fu, Tao Chen, and Gang Yu. Executing your commands via motion diffusion in latent space. In CVPR, pages 1800018010, 2023.2   
[8] Yiwen Chen, Chi Zhang, Xiaofeng Yang, Zhongang Cai, Gang Yu, Lei Yang, and Guosheng Lin. It3d: Improved textto-3d generation with explicit view synthesis. arXiv preprint arXiv:2308.11473, 2023. 2   
[9] Zilong Chen, Feng Wang, and Huaping Liu. Text-to-3d using gaussian splatting. arXiv preprint arXiv:2309.16585, 2023. 1, 2, 3   
[10] Rishabh Dabral, Muhammad Hamza Mughal, Vladislav Golyanik, and Christian Theobalt. Mofusion: A framework for denoising-diffusion-based motion synthesis. In CVPR, pages 97609770, 2023. 2   
[11] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In CVPR, pages 13142 13153, 2023. 5, 7   
[12] Jun Gao, Tianchang Shen, Zian Wang, Wenzheng Chen, Kangxue Yin, Daiqing Li, Or Litany, Zan Gojcic, and Sanja Fidler. Get3d: A generative model of high quality 3d textured shapes learned from images. NeurIPS, 35:31841 31854, 2022. 1, 2   
[13] Jiatao Gu, Alex Trevithick, Kai-En Lin, Joshua M Susskind, Christian Theobalt, Lingjie Liu, and Ravi Ramamoorthi. Nerfdiff: Single-image view synthesis with nerf-guided distillation from 3d-aware diffusion. In ICML, pages 11808 11826. PMLR, 2023. 2, 14   
[14] Yuan-Chen Guo, Ying-Tian Liu, Ruizhi Shao, Christian Laforte, Vikram Voleti, Guan Luo, Chia-Hao Chen, ZiXin Zou. Chen Wang. Yan-Pei Cao. and Song-Hai Zhang. threestudio: A unified framework for 3d content generation. https://github.com/threestudio-project/ threestudio, 2023. 5, 12   
[15] Anchit Gupta, Wenhan Xiong, Yixin Nie, Ian Jones, and Barlas Ouz. 3dgen: Triplane latent diffusion for textured mesh generation. arXiv preprint arXiv:2303.05371, 2023. 1, 2   
[16] Xiao Han, Yukang Cao, Kai Han, Xiatian Zhu, Jiankang Deng, Yi-Zhe Song, Tao Xiang, and Kwan-Yee K Wong. Headsculpt: Crafting 3d head avatars with text. In NeurIPS, 2023.2   
[17] Yuze He, Yushi Bai, Matthieu Lin, Wang Zhao, Yubin Hu, Jenny Sheng, Ran Yi, Juanzi Li, and Yong-Jin Liu. T3bench: Benchmarking current progress in text-to-3d generation. arXiv preprint arXiv:2310.02977, 2023. 5   
[18] Fangzhou Hong, Mingyuan Zhang, Liang Pan, Zhongang Cai, Lei Yang, and Ziwei Liu. Avatarclip: Zero-shot textdriven generation and animation of 3d avatars. arXiv preprint arXiv:2205.08535, 2022. 2   
[19] Shoukang Hu, Fangzhou Hong, Tao Hu, Liang Pan, Haiyi Mei, Weiye Xiao, Lei Yang, and Ziwei Liu. Humanliff: Layer-wise 3d human generation with diffusion model. arXiv preprint arXiv:2308.09712, 2023. 2   
[20] Yukun Huang, Jianan Wang, Ailing Zeng, He Cao, Xianbiao Qi, Yukai Shi, Zheng-Jun Zha, and Lei Zhang. Dreamwaltz: Make a scene with complex 3d animatable avatars. arXiv preprint arXiv:2305.12529, 2023. 6, 7   
[21] Yangyi Huang, Hongwei Yi, Yuliang Xiu, Tingting Liao, Jiaxiang Tang, Deng Cai, and Justus Thies. Tech: Text-guided reconstruction of lifelike clothed humans. arXiv preprint arXiv:2308.08545, 2023. 2   
[22] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, 2021. If you use this software, please cite it as below. 12   
[23] Ajay Jain, Ben Mildenhall, Jonathan T Barron, Pieter Abbeel, and Ben Poole. Zero-shot text-guided object generation with dream fields. In CVPR, pages 867876, 2022. 2   
[24] Ruixiang Jiang, Can Wang, Jingbo Zhang, Menglei Chai, Mingming He, Dongdong Chen, and Jing Liao. Avatarcraft: Transforming text into neural human avatars with parameterized shape and pose control. arXiv preprint arXiv:2303.17606, 2023. 2   
[25] Heewoo Jun and Alex Nichol. Shap-e: Generating conditional 3d implicit functions. arXiv preprint arXiv:2305.02463, 2023. 1, 2, 5, 7, 8, 12, 13   
[26] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42 (4), 2023. 2, 3, 5, 13   
[27] Jihoon Kim, Jiseob Kim, and Sungjoon Choi. Flame: Freeform language-based motion synthesis & editing. In AAAI, pages 82558263, 2023. 2   
[28] Georgios Kopanas, Julien Philip, Thomas Leimkühler, and George Drettakis. Point-based neural rendering with perview optimization. In Computer Graphics Forum, pages 29   
43. Wiley Online Library, 2021. 3 [29] Georgios Kopanas, Thomas Leimkühler, Glles Rainer, Clément Jambon, and George Drettakis. Neural point catacaustics for novel-view synthesis of reflections. ACM Transactions on Graphics (T0G), 41(6):115, 2022. 3 [30] Jiabao Lei, Yabin Zhang, Kui Jia, et al. Tango: Text-driven photorealistic and robust 3d stylization via lighting decomposition. NeurIPS, 35:3092330936, 2022. 2 [31] Gang Li, Heliang Zheng, Chaoyue Wang, Chang Li, Changwen Zheng, and Dacheng Tao. 3ddesigner: Towards photorealistic 3d object generation and editing with text-guided diffusion models. arXiv preprint arXiv:2211.14108, 2022. 2,   
14 [32] Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, and Sai Bi. Instant3d: Fast text-to-3d with sparse-view generation and large reconstruction model. arXiv preprint arXiv:2311.06214, 2023. 12, 13 [33] Weiyu Li, Rui Chen, Xuelin Chen, and Ping Tan. Sweetdreamer: Aligning geometric priors in 2d diffusion for consistent text-to-3d. arXiv preprint arXiv:2310.02596, 2023.   
2 [34] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In CVPR, pages 300309, 2023.   
1, 2, 5, 6 [35] Yukang Lin, Haonan Han, Chaoqun Gong, Zunnan Xu, Yachao Zhang, and Xiu Li. Consistent123: One image to highly consistent 3d asset using case-aware diffusion priors. arXiv preprint arXiv:2309.17261, 2023. 2 [36] Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Zexiang Xu, Hao Su, et al. One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. arXiv preprint arXiv:2306.16928, 2023. [37] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to  
3: Zero-shot one image to 3d object. arXiv preprint arXiv:2303.11328, 2023. 8 [38] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. Syncdreamer: Generating multiview-consistent images from a single-view image. arXiv preprint arXiv:2309.03453, 2023. 2 [39] Zhen Liu, Yao Feng, Michael J Black, Derek Nowrouzezahrai, Liam Paull, and Weiyang Liu. Meshdiffusion: Score-based generative 3d mesh modeling. arXiv preprint arXiv:2303.08133, 2023. 2 [40] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3d: Single image to 3d using cross-domain diffusion. arXiv preprint arXiv:2310.15008, 2023. 2 [41] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. Smpl: A skinned multiperson linear model. In Seminal Graphics Papers: Pushing the Roundaries Volume 2 nages 851866 2023 2 4 7   
[42] Jonathan Lorraine, Kevin Xie, Xiaohui Zeng, Chen-Hsuan Lin, Towaki Takikawa, Nicholas Sharp, Tsung-Yi Lin, Ming-Yu Liu, Sanja Fidler, and James Lucas. Att3d: Amortized text-to-3d object synthesis. arXiv preprint arXiv:2306.07349, 2023. 2   
[43] Shitong Luo and Wei Hu. Diffusion probabilistic models for 3d point cloud generation. In CVPR, pages 28372845, 2021.2   
[44] Tiange Luo, Chris Rockwell, Honglak Lee, and Justin Johnson. Scalable 3d captioning with pretrained models. arXiv preprint arXiv:2306.07279, 2023. 5, 7   
[45] Gal Metzer, Elad Richardson, Or Patashnik, Raja Giryes, and Daniel Cohen-Or. Latent-nerf for shape-guided generation of 3d shapes and textures. In CVPR, pages 1266312673, 2023. 5   
[46] Oscar Michel, Roi Bar-On, Richard Liu, Sagie Benaim, and Rana Hanocka. Text2mesh: Text-driven neural stylization for meshes. In CVPR, pages 1349213502, 2022. 2   
[47] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren $\mathrm { N g }$ Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, pages 405421, 2020. 2, 3, 7   
[48] Nasir Mohammad Khalid, Tianhao Xie, Eugene Belilovsky, and Tiberiu Popa. Clip-mesh: Generating textured meshes from text using pretrained image-text models. In SIGGRAPH Asia 2022 conference papers, pages 18, 2022. 2   
[49] Norman Müller, Yawar Siddiqui, Lorenzo Porzi, Samuel Rota Bulo, Peter Kontschieder, and Matthias NieBner. Diffrf: Rendering-guided 3d radiance field diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 43284338, 2023. 14   
[50] Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Trans. Graph., 41(4):102:1 102:15, 2022. 2   
[51] Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela Mishkin, and Mark Chen. Point-e: A system for generating 3d point clouds from complex prompts. arXiv preprint arXiv:2212.08751, 2022. 1, 2, 7, 8   
[52] Yichen Ouyang, Wenhao Chai, Jiayi Ye, Dapeng Tao, Yibing Zhan, and Gaoang Wang. Chasing consistency in text-to-3d generation from a single image. arXiv preprint arXiv:2309.03599, 2023. 2   
[53] Jangho Park, Gihyun Kwon, and Jong Chul Ye. Ed-nerf: Efficient text-guided editing of 3d scene using latent space nerf. arXiv preprint arXiv:2310.02712, 2023. 2   
[54] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. NeurIPS, 32, 2019. 5   
[55] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 1, 2, 3, 4, 5, 6, 7, 12, 13   
[56] Zekun Qi, Muzhou Yu, Runpei Dong, and Kaisheng Ma. Vpp: Efficient conditional 3d generation via voxel-point progressive representation. arXiv preprint arXiv:2307.16605, 2023. 2   
[57] Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, Hsin-Ying Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, et al. Magic123: One image to high-quality 3d object generation using both 2d and 3d diffusion priors. arXiv preprint arXiv:2306.17843, 2023.2   
[58] Sigal Raab, Inbal Leibovitch, Guy Tevet, Moab Arar, Amit H Bermano, and Daniel Cohen-Or. Single motion diffusion. arXiv preprint arXiv:2302.05905, 2023. 2   
[59] Alec Radford, Jong Wook Kim, Chris Hallacy, A. Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021. 12   
[60] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 87488763. PMLR, 2021. 2, 12, 13   
[61] Amit Raj, Srinivas Kaza, Ben Poole, Michael Niemeyer, Nataniel Ruiz, Ben Mildenhall, Shiran Zada, Kfir Aberman, Michael Rubinstein, Jonathan Barron, et al. Dreambooth3d: Subject-driven text-to-3d generation. arXiv preprint arXiv:2303.13508, 2023. 2   
[62] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10684 10695, 2022. 1, 5, 14   
[63] Aditya Sanghi, Hang Chu, Joseph G Lambourne, Ye Wang, Chin-Yi Cheng, Marco Fumero, and Kamal Rahimi Malekshan. Clip-forge: Towards zero-shot text-to-shape generation. In CVPR, pages 1860318613, 2022. 2   
[64] Kyle Sargent, Zizhang Li, Tanmay Shah, Charles Herrmann, Hong-Xing Yu, Yunzhi Zhang, Eric Ryan Chan, Dmitry Lagun, Li Fei-Fei, Deqing Sun, et al. Zeronvs: Zero-shot 360-degree view synthesis from a single real image. arXiv preprint arXiv:2310.17994, 2023. 2   
[65] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa R Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5b: An open large-scale dataset for training next generation image-text models. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022. 12   
[66] Hoigi Seo, Hayeon Kim, Gwanghyun Kim, and Se Young Chun. Ditto-nerf: Diffusion-based iterative text to omnidirectional 3d model. arXiv preprint arXiv:2304.02827, 2023.2   
[67] Junyoung Seo, Wooseok Jang, Min-Seop Kwak, Jaehoon Ko, Hyeonsu Kim, Junho Kim, Jin-Hwa Kim, Jiyoung Lee, and Seungryong Kim. Let 2d diffusion model know 3dconsistency for robust text-to-3d generation. arXiv preprint -V:…2202 07027, 20222   
[68] Yonatan Shafir, Guy Tevet, Roy Kapon, and Amit H Bermano. Human motion diffusion as a generative prior. arXiv preprint arXiv:2303.01418, 2023. 2   
[69] Tianchang Shen, Jun Gao, Kangxue Yin, Ming-Yu Liu, and Sanja Fidler. Deep marching tetrahedra: a hybrid representation for high-resolution 3d shape synthesis. NeurIPS, 34: 60876101, 2021. 2   
[70] Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. Zero $^ { 1 2 3 + + }$ a single image to consistent multi-view diffusion base model. arXiv preprint arXiv:2310.15110, 2023. 2   
[71] Yukai Shi, Jianan Wang, He Cao, Boshi Tang, Xianbiao Qi, Tanyu Yang, Yuku Huang, Shilong Lu, Lei Zhang, and Heung-Yeung Shum. Toss: High-quality text-guided novel view synthesis from a single image. arXiv preprint arXiv:2310.10644, 2023. 2   
[72] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512, 2023. 1, 2, 8   
[73] Liangchen Song, Liangliang Cao, Hongyu Xu, Kai Kang, Feng Tang, Junsong Yuan, and Yang Zhao. Roomdreamer: Text-driven 3d indoor scene synthesis with coherent geometry and texture. arXiv preprint arXiv:2305.11337, 2023. 2   
[74] Jingxiang Sun, Bo Zhang, Ruizhi Shao, Lizhen Wang, Wen Liu, Zhenda Xie, and Yebin Liu. Dreamcraft3d: Hierarchical 3d generation with bootstrapped diffusion prior. arXiv preprint arXiv:2310.16818, 2023. 2   
[75] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653, 2023. 1, 2, 14   
[76] Junshu Tang, Tengfei Wang, Bo Zhang, Ting Zhang, Ran Yi, Lizhuang Ma, and Dong Chen. Make-it-3d: High-fidelity 3d creation from a single image with diffusion prior. arXiv preprint arXiv:2303.14184, 2023. 2   
[77] Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, and Amit H Bermano. Human motion diffusion model. arXiv preprint arXiv:2209.14916, 2022. 2, 5   
[78] Christina Tsalicoglou, Fabian Manhardt, Alessio Tonioni, Michael Niemeyer, and Federico Tombari. Textmesh: Generation of realistic 3d meshes from text prompts. arXiv preprint arXiv:2304.12439, 2023. 2   
[79] Arash Vahdat, Francis Williams, Zan Gojcic, Or Litany, Sanja Fidler, Karsten Kreis, et al. Lion: Latent point diffusion models for 3d shape generation. NeurIPS, 35:10021 10039, 2022. 2   
[80] Can Wang, Menglei Chai, Mingming He, Dongdong Chen, and Jing Liao. Clip-nerf: Text-and-image driven manipulation of neural radiance fields. In CVPR, pages 38353844, 2022. 2   
[81] Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A Yeh, and Greg Shakhnarovich. Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In CVPR, pages 1261912629, 2023. 1, 2, 5   
[82] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. arXiv preprint arXiv:2305.16213, 2023. 1, 2, 5, 6, 12, 13   
[83] Haohan Weng, Tianyu Yang, Jianan Wang, Yu Li, Tong Zhang, CL Chen, and Lei Zhang. Consistent123: Improve consistency for one image to 3d object synthesis. arXiv preprint arXiv:2310.08092, 2023. 2   
[84] Zhenzhen Weng, Zeyu Wang, and Serena Yeung. Zeroavatar: Zero-shot 3d avatar generation from a single image. arXiv preprint arXiv:2305.16411, 2023. 2   
[85] Jinbo Wu, Xiaobo Gao, Xing Liu, Zhengyang Shen, Chen Zhao, Haocheng Feng, Jingtuo Liu, and Errui Ding. Hdfusion: Detailed text-to-3d generation leveraging multiple noise estimation. arXiv preprint arXiv:2307.16183, 2023. 2   
[86] Jiale Xu, Xintao Wang, Weihao Cheng, Yan-Pei Cao, Ying Shan, Xiaohu Qie, and Shenghua Gao. Dream3d: Zero-shot text-to-3d synthesis using 3d shape prior and text-to-image diffusion models. In CVPR, pages 2090820918, 2023. 1, 2   
[87] Jiayu Yang, Ziang Cheng, Yunfei Duan, Pan Ji, and Hongdong Li. Consistnet: Enforcing 3d consistency for multiview images diffusion. arXiv preprint arXiv:2310.10343, 2023.2   
[88] Jianglong Ye, Peng Wang, Kejie Li, Yichun Shi, and Heng Wang. Consistent-1-to-3: Consistent image to 3d view synthesis via geometry-aware diffusion models. arXiv preprint arXiv:2310.03020, 2023. 2   
[89] Wang Yifan, Felice Serena, Shihao Wu, Cengiz Öztireli, and Olga Sorkine-Hornung. Differentiable surface splatting for point-based geometry processing. ACM Transactions on Graphics (TOG), 38(6):114, 2019. 3   
[90] Chaohui Yu, Qiang Zhou, Jingliang Li, Zhe Zhang, Zhibin Wang, and Fan Wang. Points-to-3d: Bridging the gap between sparse points and shape-controllable text-to-3d generation. In ACM MM, pages 68416850, 2023. 2   
[91] Huichao Zhang, Bowen Chen, Hao Yang, Liao Qu, Xu Wang, Li Chen, Chao Long, Feida Zhu, Kang Du, and Min Zheng. Avatarverse: High-quality & stable 3d avatar creation from text and pose. arXiv preprint arXiv:2308.03610, 2023. 6, 7   
[92] Longwen Zhang, Qiwei Qiu, Hongyang Lin, Qixuan Zhang, Cheng Shi, Wei Yang, Ye Shi, Sibei Yang, Lan Xu, and Jingyi Yu. Dreamface: Progressive generation of animatable 3d faces under text guidance. arXiv preprint arXiv:2304.03117, 2023. 2   
[93] Mingyuan Zhang, Xinying Guo, Liang Pan, Zhongang Cai, Fangzhou Hong, Huirong Li, Lei Yang, and Ziwei Liu. Remodiffuse: Retrieval-augmented motion diffusion model. arXiv preprint arXiv:2304.01116, 2023. 2   
[94] Mengyi Zhao, Mengyuan Liu, Bin Ren, Shuling Dai, and Nicu Sebe. Modiff: Action-conditioned 3d motion generation with denoising diffusion probabilistic models. arXiv preprint arXiv:2301.03949, 2023. 2   
[95] Minda Zhao, Chaoyi Zhao, Xinyue Liang, Lincheng Li, Zeng Zhao, Zhipeng Hu, Changjie Fan, and Xin Yu. Efficientdreamer: High-fidelity and robust 3d creation via orthogonal-view diffusion prior. arXiv preprint arXiv:2308 13223 2023 1 2 8   
[96] Zixiang Zhou and Baoyuan Wang. Ude: A unified driving engine for human motion generation. In CVPR, pages 5632 5641, 2023. 2   
[97] Joseph Zhu and Peiye Zhuang. Hifa: High-fidelity textto-3d with advanced diffusion guidance. arXiv preprint arXiv:2305.18766, 2023. 2

![](images/11.jpg)  
Figure 11. Visual comparisons with Instant3D [32].

# A. Appendix

# A.1. More Results

Quantitative Comparisons. In Tab. 2, we use CLIP [60] similarity to quantitatively evaluate our method. The results of other methods in the table come from the concurrent Instant3D [32] paper. The results of Shap-E [25] come from the official source, while DreamFusion [55] and ProlificDreamer [82] results come from implementation by threestudio [14]. The implementation version of DreamFusion is shorter in time than the official report we mention in the main text. During the evaluation, we use a camera radius of 4, an elevation of 15 degrees, and select 120 evenly spaced azimuth angles from -180 to 180 degrees, resulting in 120 rendered images from different viewpoints. We follow the Instant3D settings, randomly selecting 10 from the 120 rendered images. We calculate the similarity between each selected image and the text and then compute the average for 10 selected images. It's worth noting that when other methods are evaluated, 400 out of DreamFusion's 415 prompts are selected. This is because some generations failed, so our method is disadvantaged during evaluation on all 415 prompts from DreamFusion. We use two models, ViT-L/14 from OpenAI [59] 1 and ViT-bigG-14 from OpenCLIP [22, 65] 2, to calculate CLIP similarity. Our method is superior to all methods except ProlificDreamer, but it is 40 times faster than ProlificDreamer in generation speed. As shown in Fig 11, our method shows notably better quality and details than a concurrent work Instant3D but the CLIP similarity increases marginally.

Table 2. Quantitative comparisons on CLIP [60] similarity with other methods.   

<table><tr><td>Methods</td><td>ViT-L/14 ↑</td><td>ViT-bigG-14 ↑</td><td>Generation Time ↓</td></tr><tr><td>Shap-E [25]</td><td>20.51</td><td>32.21</td><td>6 seconds</td></tr><tr><td>DreamFusion [55]</td><td>23.60</td><td>37.46</td><td>1.5 hours</td></tr><tr><td>ProlificDreamer [82]</td><td>27.39</td><td>42.98</td><td>10 hours</td></tr><tr><td>Instant3D [32]</td><td>26.87</td><td>41.77</td><td>20 seconds</td></tr><tr><td>Ours</td><td>27.23 ± 0.06</td><td>41.88 ± 0.04</td><td>15 minutes</td></tr></table>

![](images/12.jpg)  
Figure 12. Results of generation with ground.

![](images/13.jpg)  
Figure 13. Results of the diversity of our method.

Generation with Ground. When initializing, we add a layer of point clouds representing the ground at the bottom of the generated point clouds. The color of the ground is randomly initialized. Then, we use the point clouds with the added ground to initialize the 3D Gaussians. Fig. 12 shows the results of the final 3D Gaussian Splatting [26].

Diversity. In Fig. 13, we demonstrate the diversity of our method in generating 3D assets by using different random seeds for the same prompt.

Generation with More Fine-grained Prompts. More refined prompts are used to generate 3D assets, as shown in Fig. 14. It can be seen that Shap-E [25] generates similar results when given different descriptions of the word "axe" in the prompt. However, our method produces 3D assets that better match the prompt.

Automatically Select A Human Model. As shown in Fig 15, we attempt to use CLIP to guide the selection of the initialized human body model, by computing the similarities between images rendered from the generated SMPL models and the text prompt. We can achieve good rendering effects on various human body models. It would also be a promising direction to extend the assets to dynamic ones with the sequence of generated human body models.

![](images/14.jpg)  
Figure 14. Results of generation with more fine-grained prompts.

# A.2. More Ablation Studies

2D Diffusion Model During the process of optimizing 3D Gaussians with a 2D diffusion model, we perform ablation on the 2D diffusion models we use, specifically stabilityai/stable-diffusion-2-1-base [62] 3 and DeepFloyd/IF-I-XL-v1.0 4. Fig. 16 shows the results of the ablation experiment, where it can be seen that the 3D assets generated using the stabilityai/stable-diffusion-2-1- base have richer details.

Box Size in Point Growth In Fig 17, we conduct an ablation experiment on the box size, where a larger box leads to a fatter asset along with a more blurry appearance.

# A.3. More Discussions

Limitations Introduced by the 3D Datasets. Fig 18 shows the generation results of complex prompts. The domain-limited 3D diffusion model can only generate parts of the desired object with rough appearances. Our method completes the remaining part and provides finer details by bridging the domain-abundant 2D diffusion model.

![](images/15.jpg)  
Figure 15. Avatar generation.

![](images/16.jpg)  
Figure 16. Ablation studies of optimizing 3D Gaussians with different 2D diffusion models.

Recent Works. We discuss with more related work. Our focus is to connect the 3D and 2D diffusion models, fusing the data capacity from both types of diffusion models and generating 3DGS-based assets directly from text. DreamGaussian [75] finally generates mesh-based 3D assets from an image or an image generated from text, which can be orthogonal to our method. There is a possibility of a combination in the future. NerfDiff [13] uses a 3D-aware conditional diffusion to enhance details. DiffRF [49] employs 3D-Unet to operate directly on the radiation field, achieving truthful 3D geometry and image synthesis. 3DDesigner [31] proposes a two-stream asynchronous diffusion module, which can improve 3D consistency.

![](images/17.jpg)  
Viking axe, fantasy, weapon...   
Figure 17. Ablation on the size of the box.

![](images/18.jpg)  
Figure 18. Generation with complex prompts.