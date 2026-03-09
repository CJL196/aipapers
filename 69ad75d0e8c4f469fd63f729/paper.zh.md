# 生成对抗网络

伊恩·杰·古德费洛，尚·普吉-阿巴迪，梅赫迪·米尔扎，冯·徐，大卫·瓦德-法利，舍吉尔·奥扎尔，亚伦·库尔维尔，约书亚·本吉奥‡ 蒙特利尔大学计算机与运筹学系 加拿大魁北克省蒙特利尔市 H3C 3J7

# 摘要

我们提出了一种通过对抗过程估计生成模型的新框架，其中我们同时训练两个模型：一个生成模型 $G$ ，用于捕捉数据分布；一个判别模型 $D$ ，用于估计样本来自训练数据的概率，而非来自 $G$ 。$G$ 的训练过程是最大化 $D$ 出错的概率。该框架对应于一个极小极大双人游戏。在任意函数的空间中，存在唯一解，$G$ 恢复训练数据分布，而 $D$ 在各处等于 $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ 。当 $G$ 和 $D$ 由多层感知机定义时，整个系统可以通过反向传播进行训练。在训练或生成样本的过程中，无需使用任何马尔可夫链或展开的近似推理网络。实验通过定性和定量评估生成的样本，展示了该框架的潜力。

# 1 引言

深度学习的承诺在于发现丰富的分层模型，这些模型表示在人工智能应用中遇到的数据类型的概率分布，如自然图像、包含语音的音频波形和自然语言语料库中的符号。目前，深度学习中最显著的成功涉及判别模型，通常是将高维、丰富的感知输入映射到类别标签的模型。这些显著的成功主要基于反向传播和 dropout 算法，使用分段线性单元，这些单元具有特别良好的梯度表现。深度生成模型的影响相对较小，主要是由于在最大似然估计和相关策略中出现的许多难以处理的概率计算的近似困难，以及在生成背景中利用分段线性单元的好处的困难。我们提出了一种新的生成模型估计程序，规避了这些困难。在提出的对抗网络框架中，生成模型与对手进行较量：一个判别模型，用于学习确定样本是否来自模型分布或数据分布。生成模型可以被视为一组伪钞制造者，试图生产伪币并在不被发现的情况下使用它，而判别模型则类似于警察，试图检测伪钞。这场竞争促使双方团队改进其方法，直到伪币与真币无可区分。该框架可以为多种模型和优化算法提供具体的训练算法。本文探索了生成模型通过将随机噪声传递通过多层感知器生成样本的特殊情况，并且判别模型也是一个多层感知器。我们将这个特殊情况称为对抗网络。在这种情况下，我们可以仅使用成功的反向传播和 dropout 算法来训练这两个模型，并仅使用前向传播从生成模型中采样。无需近似推断或马尔可夫链。

# 2 相关工作

具有潜变量的有向图模型的替代方案是具有潜变量的无向图模型，例如限制玻尔兹曼机（RBM）、深度玻尔兹曼机（DBM）及其众多变种。这类模型中的相互作用以非标准化势函数的乘积形式表示，通过对所有随机变量状态的全局求和/积分进行标准化。这个量（分区函数）及其梯度对于除了最简单的情况外是不可处理的，尽管可以通过马尔可夫链蒙特卡洛（MCMC）方法进行估计。混合对依赖于MCMC的学习算法构成了显著问题。深度置信网络（DBN）是包含单个无向层和多个有向层的混合模型。尽管存在快速近似的逐层训练标准，但DBN面临着与无向和有向模型相关的计算困难。

还提出了一些不近似或界定对数似然的替代标准，例如得分匹配和噪声对比估计（NCE）。这两者都要求所学习的概率密度在一个归一化常数的基础上被解析地指定。值得注意的是，在许多有多个潜在变量层级的有趣生成模型中（如深度置信网络和深度玻尔兹曼机），甚至无法推导出一份可处理的非标准化概率密度。某些模型，例如去噪自编码器和收缩自编码器，具有与施加在受限玻尔兹曼机上的得分匹配非常相似的学习规则。在NCE中，如同本研究，采用了一种判别训练标准来拟合生成模型。然而，与拟合单独的判别模型不同，生成模型本身被用来区分生成的数据和固定噪声分布的样本。由于NCE使用固定的噪声分布，学习在模型甚至对观察变量的小子集学习到一个大致正确的分布后会显著减慢。

最后，一些技术并不涉及明确地定义概率分布，而是训练一个生成模型从期望分布中抽样。这种方法的优点在于，这类模型可以被设计为通过反向传播进行训练。该领域最近的显著工作包括生成随机网络（GSN）框架[5]，它扩展了广义去噪自编码器[4]：两者都可以被视为定义了一个参数化的马尔可夫链，即学习一个执行生成马尔可夫链一步的模型的参数。相比于GSN，生成对抗网络框架不需要马尔可夫链来进行抽样。由于生成对抗网络在生成过程中不需要反馈回路，因此能够更好地利用分段线性单元[19，9，10]，这改善了反向传播的性能，但在使用反馈回路时会遇到无界激活问题。最近的通过反向传播训练生成模型的示例包括对自编码变分贝叶斯[20]和随机反向传播[24]的研究。

# 3 对抗网络

对抗建模框架最容易应用于两个模型均为多层感知机的情况。为了学习生成器的分布 $p _ { g }$ 关于数据 $_ { \textbf { \em x } }$，我们定义输入噪声变量的先验分布 $p _ { z } ( z )$，然后将映射到数据空间表示为 $G ( z ; \theta _ { g } )$，其中 $G$ 是由参数 $\theta _ { g }$ 表示的多层感知机的可微分函数。我们还定义了第二个多层感知机 $D ( \pmb { x } ; \dot { \theta } _ { d } )$，其输出一个标量。$D ( { \pmb x } )$ 表示 $_ { \textbf { \em x } }$ 来自数据而非 $p _ { g }$ 的概率。我们训练 $D$ 以最大化分配正确标签给训练例子和来自 $G$ 的样本的概率。同时，我们训练 $G$ 以最小化 $\log ( 1 - D ( G ( z ) ) )$：换句话说，$D$ 和 $G$ 进行以下两人对抗博弈，价值函数为 $V ( G , D )$：

$$
\operatorname* { m i n } _ { G } \operatorname* { m a x } _ { D } V ( D , G ) = \mathbb { E } _ { { \pmb x } \sim p _ { \mathrm { d a t a } } ( { \pmb x } ) } [ \log D ( { \pmb x } ) ] + \mathbb { E } _ { { \pmb z } \sim p _ { \pmb z } ( { \pmb z } ) } [ \log ( 1 - D ( G ( { \pmb z } ) ) ) ] .
$$

在下一部分中，我们对对抗网络进行了理论分析，基本上展示了训练准则允许在 $G$ 和 $D$ 具有足够容量的情况下恢复数据生成分布，即在非参数极限下。有关该方法更不正式、更多教学性质的解释，请参见图 1。在实践中，我们必须使用迭代数值方法来实现该游戏。在训练的内循环中，完全优化 $D$ 在计算上是不可行的，并且在有限数据集上会导致过拟合。相反，我们在优化 $D$ 的 $k$ 步与优化 $G$ 的一步之间交替。这使得 $D$ 保持在其最优解附近，只要 $G$ 的变化足够缓慢。这一策略类似于 SML/PCD [31, 29] 训练在学习的每一步之间保持来自马尔可夫链的样本，以避免在学习的内循环中燃烧马尔可夫链。该过程在算法 1 中正式呈现。在实践中，方程 1 可能无法为 $G$ 提供足够的梯度以良好学习。在学习的早期，当 $G$ 表现不佳时，$D$ 可以高信心地拒绝样本，因为它们显然与训练数据不同。在这种情况下，$\log ( 1 - D ( G ( z ) ) )$ 会饱和。我们可以训练 $G$ 来最大化 $\arg D ( G ( z ) )$，而不是训练 $G$ 来最小化 $\log ( 1 - \bar { D ( G ( z ) ) } )$。这个目标函数导致 $G$ 和 $D$ 的动态具有相同的固定点，但在学习早期提供了更强的梯度。

![](images/1.jpg)  
FigureGenerativeadversarial nets are trained by simultaneously updating the discriminative distribution $D$ , blue, dashed line) so that it discriminates between samples from the data generating distribution (black, dotted line) $p _ { x }$ from those of the generative distribution $p _ { g }$ (G) (green, solid line). The lower horizontal line is the domain from which $_ z$ is sampled, in this case uniformly. The horizontal line above is part of the domain of $_ { \textbf { \em x } }$ . The upward arrows show how the mapping $x = G ( z )$ imposes the non-uniform distribution $p _ { g }$ on transformed samples. $G$ contracts in regions of high density and expands in regions of low density of $p _ { g }$ . (a) Consider an adversarial pair near convergence: $p _ { g }$ is similar to $p \mathrm { d a t a }$ and $D$ is a partially accurate classifier. (b) In the inner loop of the algorithm $D$ is trained to discriminate samples from data, converging to $D ^ { \ast } ( { \pmb x } ) =$ $\frac { p _ { \mathrm { d a t a } } ( \pmb { x } ) } { p _ { \mathrm { d a t a } } ( \pmb { x } ) + p _ { g } ( \pmb { x } ) }$ $G$ $D$ $G ( z )$ to be classified as data. (d) After several steps of training, if $G$ and $D$ have enough capacity, they will reach a point at which both cannot improve because $p _ { g } = p _ { \mathrm { d a t a } }$ . The discriminator is unable to differentiate between the two distributions, i.e. $\begin{array} { r } { D ( \pmb { x } ) = \frac { 1 } { 2 } } \end{array}$

# 4 理论结果

生成器 $G$ 隐式定义了概率分布 $p _ { g }$，该分布是当 $z \sim p _ { z }$ 时获得的样本 $G ( z )$ 的分布。因此，我们希望算法 1 能够收敛到 $p _ { \mathrm { d a t a } }$ 的良好估计器，只要有足够的容量和训练时间。本节的结果是在非参数设置下完成的，例如，我们通过研究概率密度函数空间中的收敛性来表示具有无限容量的模型。我们将在第 4.1 节中展示这个极小极大博弈在 $p _ { g } = p _ { \mathrm { d a t a } }$ 时具有全局最优解。然后我们将在第 4.2 节中展示算法 1 优化了 $\mathrm { E q } 1$，从而获得所需的结果。算法 1 生成对抗网络的迷你批量随机梯度下降训练。应用于鉴别器的步骤数 $k$ 是一个超参数。在我们的实验中，我们使用了 $k = 1$，这是成本最低的选项。对于每个训练迭代的次数：对 $k$ 步执行：从噪声先验 $p _ { g } ( z )$ 中抽样 $m$ 个噪声样本 $\{ z ^ { ( 1 ) } , \dots , z ^ { ( m ) } \}$；从数据生成分布 $p _ { \mathrm { d a t a } } ( \pmb { x } )$ 中抽样 $m$ 个样本 $\{ \pmb { x } ^ { ( 1 ) } , \ldots , \pmb { x } ^ { ( m ) } \}$；•通过上升其随机梯度来更新鉴别器：

$$
\nabla _ { \theta _ { d } } \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left[ \log D \left( { \pmb x } ^ { ( i ) } \right) + \log \left( 1 - D \left( G \left( { \pmb z } ^ { ( i ) } \right) \right) \right) \right] .
$$

# 结束

从噪声先验 $p _ { g } ( z )$ 中抽取的 $m$ 个噪声样本的样本小批量 $\{ z ^ { ( 1 ) } , \dots , z ^ { ( m ) } \}$ • 通过下降其随机梯度来更新生成器：

$$
\nabla _ { \theta _ { g } } \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \log \left( 1 - D \left( G \left( z ^ { ( i ) } \right) \right) \right) .
$$

# 结束

基于梯度的更新可以使用任何标准的基于梯度的学习规则。我们在实验中使用了动量法。

# 4.1 $p _ { g } = p _ { \mathbf { d a t a } }$ 的全局最优性

我们首先考虑在给定生成器 $G$ 的情况下，最优鉴别器 $D$。命题 1：对于固定的 $G$，最优鉴别器 $D$ 是

$$
D _ { G } ^ { * } ( { \pmb x } ) = \frac { p _ { d a t a } ( { \pmb x } ) } { p _ { d a t a } ( { \pmb x } ) + p _ { g } ( { \pmb x } ) }
$$

证明。当给定任意生成器 $G$ 时，判别器 $\mathbf{D}$ 的训练标准是最大化数量 $V(G, D)$。

$$
\begin{array} { l } { { \displaystyle V ( G , D ) = \int _ { x } p _ { \mathrm { d a t a } } ( { \pmb x } ) \log ( D ( { \pmb x } ) ) d x + \int _ { z } p _ { z } ( z ) \log ( 1 - D ( g ( z ) ) ) d z } \ ~ } \\ { { \displaystyle ~ = \int _ { x } p _ { \mathrm { d a t a } } ( { \pmb x } ) \log ( D ( { \pmb x } ) ) + p _ { g } ( { \pmb x } ) \log ( 1 - D ( { \pmb x } ) ) d x } } \end{array}
$$

对于任意的 \( ( a , b ) \in \mathbb{R}^{2} \setminus \{ 0 , 0 \} \)，函数 \( y \to a \log ( y ) + b \log ( 1 - y ) \) 在区间 \( [ 0 , 1 ] \) 上的最大值出现在 \( \frac { a } { a + b } \) 处，得出证据。请注意，\( D \) 的训练目标可以理解为最大化条件概率 \( P ( \boldsymbol{Y} = \boldsymbol{y} | \mathbf{x} ) \) 的对数似然估计，其中 \( Y \) 表示 \( \textbf{ \em x } \) 来自于 \( p_{\mathrm{data}} \)（即 \( y = 1 \)）还是来自于 \( p_{g} \)（即 \( y = 0 \)）。方程 1 中的最小最大博弈现在可以重述为：

$$
\begin{array} { r l } & { C ( G ) = \underset { D } { \operatorname* { m a x } } V ( G , D ) } \\ & { \qquad = \mathbb { E } _ { \boldsymbol { x } \sim p _ { \mathrm { d a t } } } [ \log D _ { G } ^ { * } ( \boldsymbol { x } ) ] + \mathbb { E } _ { \boldsymbol { z } \sim p _ { z } } [ \log ( 1 - D _ { G } ^ { * } ( G ( \boldsymbol { z } ) ) ) ] } \\ & { \qquad = \mathbb { E } _ { \boldsymbol { x } \sim p _ { \mathrm { d a t } } } [ \log D _ { G } ^ { * } ( \boldsymbol { x } ) ] + \mathbb { E } _ { \boldsymbol { x } \sim p _ { g } } [ \log ( 1 - D _ { G } ^ { * } ( \boldsymbol { x } ) ) ] } \\ & { \qquad = \mathbb { E } _ { \boldsymbol { x } \sim p _ { \mathrm { d a t } } } \left[ \log \frac { p _ { \mathrm { d a t } } ( \boldsymbol { x } ) } { P _ { \mathrm { d a t } } ( \boldsymbol { x } ) + p _ { g } ( \boldsymbol { x } ) } \right] + \mathbb { E } _ { \boldsymbol { x } \sim p _ { g } } \left[ \log \frac { p _ { g } ( \boldsymbol { x } ) } { p _ { \mathrm { d a t } } ( \boldsymbol { x } ) + p _ { g } ( \boldsymbol { x } ) } \right] } \end{array}
$$

定理 1. 当且仅当 $p _ { g } = p _ { d a t a }$ 时，虚拟训练标准 $C ( G )$ 达到全局最小值。在此时，$C ( G )$ 的值为 $- \log 4$。

证明。当 $p_{g} = p_{\mathrm{data}}$ 时，$\begin{array}{r} D_{G}^{*}(\pmb{x}) = \frac{1}{2} \end{array}$ （参考方程2）。因此，通过检查方程4在 $\begin{array}{r} D_{G}^{*}(\pmb{x}) = \frac{1}{2} \end{array}$ 时的情况，我们发现 $C(G) = \log \textstyle{\frac{1}{2}} + \log \textstyle{\frac{1}{2}} = -\log 4$。为了 $h$ 与 $C(G)$ 的关系，仅在 $p_{g} = p_{\mathrm{data}}$ 时达到，观察到通过从 $C(G) = V(D_{G}^{*}, G)$ 中减去此表达式，我们得到：

$$
\mathbb { E } _ { { \pmb x } \sim p _ { \mathrm { d a t a } } } \left[ - \log 2 \right] + \mathbb { E } _ { { \pmb x } \sim p _ { g } } \left[ - \log 2 \right] = - \log 4
$$

$$
C ( G ) = - \log ( 4 ) + K L ( p _ { \mathrm { d a t a } } | | \frac { p _ { \mathrm { d a t a } } + p _ { g } } { 2 }  ) + K L ( p _ { g } | | \frac { p _ { \mathrm { d a t a } } + p _ { g } } { 2 }  ) 
$$

其中 $\mathrm{KL}$ 是 Kullback-Leibler 散度。我们在之前的表达式中认识到模型分布与数据生成过程之间的 Jensen-Shannon 散度：

$$
C ( G ) = - \log ( 4 ) + 2 \cdot J S D \left( p _ { \mathrm { d a t a } } \| p _ { g } \right)
$$

由于两个分布之间的杰森-香农散度总是非负的，并且仅在它们相等时为零，我们已证明 $C ^ { * } = - \log ( 4 )$ 是 $C ( G )$ 的全局最小值，唯一的解为 $p _ { g } = p _ { \mathrm { d a t a } }$，即生成模型完美地复制数据生成过程。

# 4.2 算法 1 的收敛性

命题 2. 如果 $G$ 和 $D$ 具备足够的能力，并且在算法 1 的每一步中，鉴别器在给定 $G$ 的情况下被允许达到其最优状态，同时 $p_{g}$ 被更新以改善标准，则 $p_{g}$ 收敛到 $P_{data}$。

$$
\mathbb { E } _ { { \pmb x } \sim p _ { d a t a } } [ \log D _ { G } ^ { * } ( { \pmb x } ) ] + \mathbb { E } _ { { \pmb x } \sim p _ { g } } [ \log ( 1 - D _ { G } ^ { * } ( { \pmb x } ) ) ]
$$

证明。考虑 $V ( G , D ) = U ( p _ { g } , D )$ 作为 $p _ { g }$ 的函数，如上述标准所示。注意 $U ( p _ { g } , D )$ 在 $p _ { g }$ 上是凸的，凸函数的子导数上确界包含在达到最大值点的函数导数。换句话说，如果 $f ( x ) = \textstyle \operatorname* { s u p } _ { \alpha \in { \mathcal { A } } } f _ { \alpha } ( x )$ 且对于每个 $\alpha$，$f _ { \alpha } ( x )$ 在 $x$ 上是凸的，则 $\partial f _ { \beta } ( x ) \in \partial f$ 若 $\beta = \arg \operatorname* { s u p } _ { \alpha \in \mathcal { A } } f _ { \alpha } ( x )$。这等价于在给定相应 $G$ 的最优 $D$ 下计算 $p _ { g }$ 的梯度下降更新，$\operatorname* { s u p } _ { D } U ( p _ { g } , D )$ 在 $p _ { g }$ 上是凸的并具有唯一的全局最优解，如定理 1 所证明，因此在 $p _ { g }$ 足够小的更新下，$p _ { g }$ 收敛到 $p _ { x }$ ，从而结束证明。□ 在实践中，对抗网络通过函数 $G ( z ; \theta _ { g } )$ 表示有限的 $p _ { g }$ 分布，且我们优化的是 $\theta _ { g }$ 而不是 $p _ { g }$ 本身。使用多层感知器来定义 $G$ 会在参数空间中引入多个临界点。然而，多层感知器在实践中的优秀表现表明，尽管缺乏理论保证，它们仍然是一个合理的模型。

# 5 实验

我们在一系列数据集上训练了对抗网络，包括 MNIST [23]、多伦多人脸数据库 (TFD) [28] 和 CIFAR-10 [21]。生成器网络采用了线性整流激活函数 [19, 9] 和 sigmoid 激活函数的混合，而判别器网络使用了 maxout [10] 激活函数。在训练判别器网络时应用了 dropout [17]。尽管我们的理论框架允许在生成器的中间层使用 dropout 和其他噪声，我们仅在生成器网络的最底层输入噪声。我们通过对生成的样本使用 $G$ 来拟合高斯 Parzen 窗口，估算测试集数据在 $p _ { g }$ 下的概率，并报告该分布下的对数似然。高斯的 $\sigma$ 参数通过在验证集上的交叉验证获得。这个程序在 Breuleux 等人 [8] 中首次介绍，并用于多种生成模型，这些模型的确切似然无法处理 [25, 3, 5]。结果见表 1。这个估算似然的方法具有较高的方差，并且在高维空间中表现不佳，但据我们所知，它是可用的最佳方法。生成模型的进展能够进行采样但无法直接估算似然，进一步激励了如何评估这些模型的研究。

<table><tr><td>Model</td><td>MNIST</td><td>TFD</td></tr><tr><td>DBN [3]</td><td>138 ± 2</td><td>1909 ± 66</td></tr><tr><td>Stacked CAE [3]</td><td>121 ± 1.6</td><td>2110 ± 50</td></tr><tr><td>Deep GSN [6]</td><td>214 ± 1.1</td><td>1890 ± 29</td></tr><tr><td>Adversarial nets</td><td>225 ± 2</td><td>2057 ± 26</td></tr></table>

Table 1: Parzen window-based log-likelihood estimates. The reported numbers on MNIST are the mean loglikelihood of sample on test set, with the standard error of the mean computed across examples. On TD, we computed the standard error across folds of the dataset, with a different $\sigma$ chosen using the validation set of each fold. On TFD, $\sigma$ was cross validated on each fold and mean log-likelihood on each fold were computed. For MNIST we compare against other models of the real-valued (rather than binary) version of dataset.

在图2和图3中，我们展示了训练后从生成网络中抽取的样本。虽然我们并不声称这些样本优于现有方法生成的样本，但我们相信这些样本至少与文献中更优秀的生成模型具有竞争力，并突出 adversarial 框架的潜力。

![](images/2.jpg)  

图：模型样本的可视化。最右列展示了邻近样本的最近训练示例，以演示模型并没有记忆训练集。样本是公平的随机抽样，而不是精挑细选的。与大多数其他深度生成模型的可视化不同，这些图像显示的是模型分布的实际样本，而不是给定隐含单元样本的条件均值。此外，这些样本是无相关的，因为采样过程不依赖于马尔科夫链混合。a) MNIST b) TFD c) CIFAR-10（全连接模型）d) CIFAR-10（卷积判别器和“反卷积”生成器） I 1 1 5 5 S S 5 5 S 7 7 9 9 9 9 7 / /7

Figure 3: Digits obtained by linearly interpolating between coordinates in $_ { z }$ space of the full model.   

表格：生成建模中的挑战：针对不同深度生成建模方法在涉及模型的主要操作时所遇到的困难总结。

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Deep directedgraphical models</td><td rowspan=1 colspan=1>Deep undirectedgraphical models</td><td rowspan=1 colspan=1>Generativeautoencoders</td><td rowspan=1 colspan=1>Adversarial models</td></tr><tr><td rowspan=1 colspan=1>Training</td><td rowspan=1 colspan=1>Inference neededduring training.</td><td rowspan=1 colspan=1>Inference neededduring training.MCMC needed toapproximatepartition functiongradient.</td><td rowspan=1 colspan=1>Enforced tradeoffbetween mixingand power ofreconstructiongeneration</td><td rowspan=1 colspan=1>Synchronizing thediscriminator withthe generator.Helvetica.</td></tr><tr><td rowspan=1 colspan=1>Inference</td><td rowspan=1 colspan=1>Learnedapproximateinference</td><td rowspan=1 colspan=1>Variationalinference</td><td rowspan=1 colspan=1>MCMC-basedinference</td><td rowspan=1 colspan=1>Learnedapproximateinference</td></tr><tr><td rowspan=1 colspan=1>Sampling</td><td rowspan=1 colspan=1>No difficulties</td><td rowspan=1 colspan=1>Requires Markovchain</td><td rowspan=1 colspan=1>Requires Markovchain</td><td rowspan=1 colspan=1>No difficulties</td></tr><tr><td rowspan=1 colspan=1>Evaluating p(x)</td><td rowspan=1 colspan=1>Intractable, may beapproximated withAIS</td><td rowspan=1 colspan=1>Intractable, may beapproximated withAIS</td><td rowspan=1 colspan=1>Not explicitlyrepresented, may beapproximated withParzen densityestimation</td><td rowspan=1 colspan=1>Not explicitlyrepresented, may beapproximated withParzen densityestimation</td></tr><tr><td rowspan=1 colspan=1>Model design</td><td rowspan=1 colspan=1>Nearly all modelsincur extremedifficulty</td><td rowspan=1 colspan=1>Careful designneeded to ensuremultiple properties</td><td rowspan=1 colspan=1>Any differentiablefunction istheoreticallypermitted</td><td rowspan=1 colspan=1>Any differentiablefunction istheoreticallypermitted</td></tr></table>

# 6 个优缺点

这一新框架相较于以往的建模框架具有优缺点。缺点主要在于没有明确表示 $p _ { g } ( \pmb { x } )$，并且在训练过程中 $D$ 必须与 $G$ 保持良好的同步（特别是 $G$ 在没有更新 $D$ 的情况下不能训练过多，以避免出现“Helvetica场景”，即 $G$ 将太多的 $\mathbf { z }$ 值压缩到相同的 $\mathbf { x }$ 值上，导致无法充分建模 $p _ { \mathrm { d a t a } }$），这和玻尔兹曼机在学习步骤之间必须保持负链更新一致性相似。其优势在于永远不需要马尔可夫链，仅使用反向传播获取梯度，学习过程中不需要推断，并且可以将多种函数纳入模型。表 2 总结了生成对抗网络与其他生成建模方法的比较。上述优势主要是计算上的。对抗模型还可能从生成器网络不直接用数据示例进行更新，而仅通过流经判别器的梯度中获得某种统计优势。这意味着输入的组成部分不会直接复制到生成器的参数中。对抗网络的另一个优势是能够表示非常尖锐甚至退化的分布，而基于马尔可夫链的方法要求分布略微模糊，以便链能在不同模式之间进行混合。

# 7 结论与未来工作

该框架允许多种直接扩展：1. 通过将 $^ c$ 作为输入同时添加到 $G$ 和 $D$ 中，可以获得条件生成模型 $p ( { \pmb x } \mid { \pmb c } )$。2. 可以通过训练辅助网络来预测 $_ { z }$ 给定 $_ { \textbf { \em x }$ 的方式进行近似推断。这类似于通过觉醒-睡眠算法 [15] 训练的推断网络，但其优点在于可以在生成网络训练完成后为固定的生成网络训练推断网络。3. 可以通过训练一系列共享参数的条件模型来近似建模所有条件 $p ( \pmb { x } _ { S } \ | \ \pmb { x } _ { \mathcal { S } } )$，其中 $S$ 是 $_ { \textbf { \em x } }$ 的索引子集。本质上，可以使用对抗网络实现确定性 MP-DBM [11] 的随机扩展。4. 半监督学习：在标记数据有限的情况下，来自判别器或推断网络的特征可以提高分类器的性能。5. 效率改进：通过设计更好的方法来协调 $G$ 和 $D$ 或在训练期间确定更好的分布以从中采样 $\mathbf { z }$，可以大大加快训练过程。本文展示了对抗建模框架的可行性，表明这些研究方向可能会带来有价值的收获。

# 致谢

我们要感谢 Patrice Marcote、Olivier Delalleau、Kyunghyun Cho、Guillaume Alain 和 Jason Yosinski 的有益讨论。Yann Dauphin 与我们分享了他的 Parzen 窗口评估代码。我们特别感谢 Pylearn2 [12] 和 Theano [7, 1] 的开发者，尤其是 Frédéric Bastien，他迅速添加了一个 Theano 功能以促进这个项目。Arnaud Bergeron 提供了紧急支持，以便进行 $\mathrm { I A T } _ { \mathrm { E } } \mathrm { X }$ 排版。我们还要感谢 CIFAR 和加拿大研究主席计划的资助，以及 Compute Canada 和 Calcul Québec 提供的计算资源。Ian Goodfellow 由 2013 年谷歌深度学习奖学金支持。最后，我们要感谢 Les Trois Brasseurs 刺激了我们的创造力。

# References

[] Bastien, F., Lamblin, P. Pascanu, R., Bergstra, J. Godfelow, I. J., Bergeron, A., Bouchard, N. and Bengio, Y. (2012). Theano: new features and speed improvements. Deep Learning and Unsupervised Feature Learning NIPS 2012 Workshop.   
[2] Bengio, Y. (2009). Learning deep architectures for AI. Now Publishers.   
[3] Bengio, Y., Mesnil, G., Dauphin, Y., and Rifai, S. (2013a). Better mixing via deep representations. In ICML'13.   
[ Bengio, Y. Yao, L., Alain, G., and Vincent, P. (2013b). Generalized denoisng auto-encoders as eneative models. In NIPS26. Nips Foundation.   
[5] Bengio, Y. Thibodeau-Laufer, E., and Yosinski, J. (2014a). Deep enerative stochastic networks trainable by backprop. In ICML'14.   
[6] Bengio, Y., Thibodeau-Laufer, E., Alain, G., and Yosinski, J. (2014b). Deep generative stochastic networks trainable by backprop. In Proceedings of the 30th International Conference on Machine Learning (ICML'14).   
[Berstra, J., Breulux, O. Bastin, F. Lblin, P.Pasnu, R. Desjardins, G. Turian, J. Ware-Far, D., and Bengio, Y. (2010). Theano: a CPU and GPU math expression compiler. In Proceedings of the Python for Scientific Computing Conference (SciPy). Oral Presentation.   
[8] Breuleux, O., Bengio, Y., and Vincent, P. (2011). Quickly generating representative samples from an RBM-derived process. Neural Computation, 23(8), 20532073.   
[9] Glorot, X., Bordes, A., and Bengio, Y. (2011). Deep sparse rectifier neural networks. In AISTATS'2011.   
[10] Goodfellow, I. J, Warde-Farley, D., Mirza, M., Courvill, A., and Bengio, Y. (2013a). Maxout networks. In ICML'2013.   
[] Goodfellow, I. J., Mirza, M., Courville, A., and Bengio, Y. (2013b). Multi-prediction deep Boltzmann machines. In NIPS'2013.   
[2] Goodfellow, I. J., Warde-Farley, D., Lamblin, P., Dumoulin, V., Mirza, M., Pascanu, R., Bergstra, J., Bastien, F., and Bengio, Y. (2013c). Pylearn2: a machine learning research library. arXiv preprint arXiv:1308.4214.   
[13] Gutmann, M. and Hyvarinen, A. (2010). Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In AISTATS'2010.   
[14] Hinton, G., Deng, L., Dahl, G. E., Mohamed, A., Jaitly, N., Senior, A., Vanhoucke, V., Nguyen, P, Sainath, T., and Kingsbury, B. (2012a). Deep neural networks for acoustic modeling in speech recognition. IEEE Signal Processing Magazine, 29(6), 8297.   
[5Hinton, G. E, Dayan, P., Frey, B. J., and Neal, R. M. (1995). The wake-sleep algorith or unsupeisd neural networks. Science, 268, 15581161.   
[16] Hinton, G. E., Osindero, S., and Teh, Y. (2006). A fast learning algorithm for deep belief nets. eural Computation, 18, 15271554.   
[ Hinton, G. E., Srivastava, N., rizhevsky, A. Sutskever, I., and Salakutinov, R. (012).Impi neural networks by preventing co-adaptation of feature detectors. Technical report, arXiv:1207.0580.   
[18] Hyvärinen, A. (2005). Estimation of non-normalized statistical models using score matching. J. Machine Learning Res., 6.   
[ Jarret, K. Kavgu, K. Ro, M. andLCun, Y.(9). Wha s hebest stahe for object recognition? In Proc. International Conference on Computer Vision (ICCV'09), pages 21462153. IEEE.   
[20] Kingma, D. P. and Wellng, M. (2014). Auto-encoding variational bayes. In Proceedings of the International Conference on Learning Representations (ICLR).   
[21] Krizhevsky, A. and Hinton, G. (2009). Learning multiple layers of features from tiny images. Technical report, University of Toronto.   
[22] Krizhevsky, A., Sutskever, I., and Hinton, G. (2012). ImageNet classification with deep convolutional neural networks. In NIPS'2012.   
[3] LeCun, Y., Bottou, L., Bengio, Y., and Haffer, P. (1998). Gradient-based learning applied todocument recognition. Proceedings of the IEEE, 86(11), 22782324.   
[24] Rezende, D. J., Mohamed, S., and Wierstra, D. (2014). Stochastic backpropagation and approximate inference in deep generative models. Technical report, arXiv:1401.4082.   
[ Rifai, S., Bengi, Y. Duphi, Y. anVincet, P. (012).A gnrative proce or smpl onive auto-encoders. In ICML'12.   
[26] Salakhutdinov, R. and Hinton, G. E. (2009). Deep Boltzmann machines. In AISTATS'2009, pages 448 455.   
[27] Smolensky, P. (1986). Information processing in dynamical systems: Foundations of harmony theory. In D. E. Rumelhart and J. L. McClelland, editors, Parallel Distributed Processing, volume 1, chapter 6, pages 194281. MIT Press, Cambridge.   
[28] Susskind, J., Anderson, A., and Hinton, G. E. (2010). The Toronto face dataset. Technical Report UTML TR 2010-001, U. Toronto.   
[29] Tieleman, T. (2008). Training restricted Boltzmann machines using approximations to the likelihood gradient. In W. W. Cohen, A. McCallum, and S. T. Roweis, editors, ICML 2008, pages 10641071. ACM.   
[30] Vincent, P., Larochelle, H., Bengio, Y., and Manzagol, P.-A. (2008). Extracting and composing robust features with denoising autoencoders. In ICML 2008.   
[31] Younes, L. (19). On the convergence of Markovian stochastic algorithms with rapidly decreasing ergodicity rates. Stochastics and Stochastic Reports, 65(3), 177228.