# 基于张量低秩表示的半监督子空间聚类

贾宇恒, 陆冠兴, 刘辉, 侯俊辉, IEEE 高级会员

摘要——在本文中，我们提出了一种新颖的半监督子空间聚类方法，能够同时增强初始监督信息并构建判别亲和矩阵。通过将有限的监督信息表示为成对约束矩阵，我们观察到用于聚类的理想亲和矩阵与理想的成对约束矩阵具有相同的低秩结构。因此，我们将这两个矩阵堆叠成一个三维张量，在此基础上施加全局低秩约束，以促进亲和矩阵的构建并同步增强初始成对约束。此外，我们利用输入样本的局部几何结构来补充全局低秩先验，以实现更好的亲和矩阵学习。提出的模型被表述为带拉普拉斯图正则化的凸低秩张量表示问题，进而使用交替迭代算法进行求解。此外，我们提出通过增强的成对约束来细化亲和矩阵。在八个常用基准数据集上的全面实验结果表明，我们的方法优于最先进的方法。代码已在 https://github.com/GuanxingLu/Subspace-Clustering 上公开发布。关键词——张量低秩表示，半监督学习，子空间聚类，成对约束。

# 一、引言

高维数据广泛存在于图像处理、DNA 微阵列技术等多个领域。高维数据通常可以通过一组线性子空间很好地近似，但某个样本的子空间归属是未知的。子空间聚类旨在将数据样本划分为不同的子空间，这是建模高维数据的重要工具。目前最先进的子空间聚类方法基于自表达性，通过自身的线性组合来表示高维数据，并对自表示矩阵施加子空间保持先验。表示系数捕获样本的全局几何关系，并可以作为亲和矩阵，子空间分割可以通过谱方法获得。本工作得到了中国国家自然科学基金（资助号 62106044）、江苏省自然科学基金（资助号 BK20210221）、香港大学资助委员会（资助号 UGC/FDS11/E02/22）、东南大学知善青年学者计划（资助号 2242022R40015）及CCF-滴滴 GAIA青年学者合作研究基金的部分支持。通讯作者：刘辉。贾宇在东南大学计算机科学与工程学院工作，南京 210096，中国；卢光在东南大学邱成熙学院工作，南京 211102，中国；刘辉在香港明爱学院计算机信息科学学院工作；侯茗在香港城市大学计算机系工作，九龙，香港（电子邮件：yhjia@seu.edu.cn；guanxing@seu.edu.cn；hliu99-c@my.cityu.edu.hk；jh.hou@cityu.edu.hk）。

![](images/1.jpg)  
Fig. 1: Illustration of the proposed method, which adaptively learns the affinity and enhances the pairwise constraints simultaneously by using their identical global low-rank structure.

在生成的亲和矩阵上进行聚类。最著名的子空间聚类方法包括稀疏子空间聚类 [3] 和低秩表示 [4]。

在许多实际应用中，某些监督信息是可用的，例如数据集的标签信息以及两个样本之间的关系。一般而言，这些监督信息可以通过两种对偶约束来表示，即“必须链接”约束和“不能链接”约束，这指示两个样本是否属于同一类别。由于监督信息广泛存在并提供样本的判别性描述，许多半监督子空间聚类方法被提出来整合这些信息。根据监督信息的类型，我们大致将这些方法分为三类。第一类方法包括“必须链接”约束。例如，[5]、[6]将“必须链接”作为硬约束整合，限制拥有“必须链接”的样本具有完全相同的表示。第二类方法整合“不能链接”约束。例如，[7]要求两个具有“不能链接”的样本之间的亲和关系为0。Liu等人[8]首先通过图拉普拉斯项增强初始的“不能链接”，然后抑制两个具有“不能链接”样本之间的亲和度。第三类方法可以同时整合“必须链接”和“不能链接”，通常假设具有“必须链接”的两个样本之间应具有更高的亲和度，而具有“不能链接”的样本之间应具有较低的亲和度[9][13]。然而，上述半监督子空间聚类方法从局部的视角利用了监督信息，但忽视了对偶约束的全局结构，这对于半监督亲和矩阵学习也至关重要。换句话说，之前的方法在某种程度上低估了监督信息的使用。

为此，我们提出了一种新的半监督子空间聚类方法，如图1所示，该方法从全局角度探索监督信息。具体而言，在理想情况下，成对约束矩阵是低秩的，因为如果样本的所有成对关系都可用，我们可以将成对约束矩阵编码为一个二进制低秩矩阵。同时，理想的相似度矩阵也是低秩的，因为一个样本应该仅由同一类的样本表示。更重要的是，它们共享相同的低秩结构。基于这样的观察，我们将它们堆叠成一个三维张量，并在形成的张量上施加全局低秩约束。通过寻找张量的低秩表示，我们可以利用可用的成对约束来细化相似度矩阵，同时用学习到的相似度矩阵增强初始的成对约束。此外，我们编码数据样本的局部几何结构，以补充全局低秩先验。所提出的模型被公式化为一个凸优化模型，可以高效求解。最后，我们使用增强后的成对约束矩阵进一步细化相似度矩阵。在8个数据集上进行的大量实验（涉及2个指标）表明，我们的方法在很大程度上优于现有最先进的半监督子空间聚类方法。

# II. 初步研究

在此信中，我们用粗体欧拉脚本字母表示张量，例如 $\mathcal { A }$ ，用粗体大写字母表示矩阵，例如 A，用粗体小写字母表示向量，例如 a，以及用小写字母表示标量，例如 ( $\boldsymbol { \imath } . \parallel \cdot \parallel _ { 2 , 1 } , \parallel \cdot \parallel _ { \infty } , \parallel \cdot \parallel _ { F }$ 和 $\| \cdot \| _ { * }$ 分别表示矩阵的 $\ell _ { 2 , 1 }$ 范数、无穷范数和弗罗贝纽斯范数，以及核范数（即奇异值的总和）。 ${ \bf X } = [ { \bf x } _ { 1 } , { \bf x } _ { 2 } , . . . , { \bf x } _ { n } ] \in \mathbb { R } ^ { d \times n }$ 是数据矩阵，其中 $\mathbf { x } _ { i } \in \mathbb { R } ^ { d \times 1 }$ 指定第 $i$ 个样本的 $d$ 维向量表示，$n$ 是样本数量。令 $\Omega _ { m } = \{ ( i , j ) \mid \mathbf { x } _ { i }$ 和 $\mathbf { x } _ { j }$ 属于同一类} 以及 $\pmb { \Omega } _ { c } = \{ ( i , j ) \mid \mathbf { x } _ { i }$ 和 $\mathbf { x } _ { j }$ 属于不同类} 分别表示可用的必须链接集和不能链接集。我们可以将成对约束编码为矩阵 $\mathbf { B } \in \mathbb { R } ^ { n \times n }$ :

# 算法 1 通过 ADMM 求解方程 (4) 输入：数据 $\mathbf { X }$，成对约束 $\Omega _ { m } , \Omega _ { c }$，超参数 $\lambda , \beta$。 1: 初始化：$\mathcal { C } ^ { ( 0 ) } = \mathcal { V } _ { 2 } ^ { ( 0 ) } = 0$，${ \bf B } ^ { ( 0 ) } = { \bf Z } ^ { ( 0 ) } = { \bf D } ^ { ( 0 ) } = { \bf E } ^ { ( 0 ) } = { \bf Y } _ { 1 } ^ { ( 0 ) } = { \bf Y } _ { 3 } ^ { ( 0 ) } = 0$，$\rho = 1.1$，$\mu ^ { ( 0 ) } = \mathrm { 1 e } - 3$，$\mu _ { \mathrm { m a x } } = \mathrm { 1 e 1 0 }.$ 2: 重复 3: 更新 $\mathcal { C }$ 为 $\begin{array} { r } { \mathcal { C } ^ { ( k + 1 ) } = \mathcal { S } _ { \frac { 1 } { \mu _ { * } ^ { ( k ) } } } ( \mathcal { M } ^ { ( k ) } { + } \mathcal { y } _ { 2 } ^ { ( k ) } / \mu ^ { ( k ) } ) } \\ { . } \end{array}$，其中 $s$ 是张量奇异值阈值操作符 [14]； 4: 更新 $\mathbf { Z }$ 为 $\mathbf { Z } ^ { ( k + 1 ) } = \left( \mathbf { I } + \mathbf { X } ^ { \top } \mathbf { X } \right) ^ { - 1 } \left( \mathbf { X } ^ { \top } ( \mathbf { X } - \mathbf { E } ^ { ( k ) } ) + \right.$ $\mathcal C ^ { ( k ) } ( : , : , 2 ) + ( \mathbf X ^ { \top } \mathbf Y _ { 1 } ^ { ( k ) } - \mathcal y _ { 2 } ^ { ( k ) } ( : , : , 2 ) ) / \mu ^ { ( k ) } )$； 5: 更新 $\mathbf { B }$ 为 $\mathbf { B } ^ { ( k + 1 ) } = ( \boldsymbol { \mu } ^ { ( k ) } ( \mathcal { C } ^ { ( k ) } ( : , : , 2 ) + \mathbf { D } ^ { ( k ) } ) - ( \mathcal { V } _ { 2 } ^ { ( k ) } ( :$ ${ \bf \Xi } , { \bf \Xi } , 2 ) + { \bf Y } _ { 3 } ^ { ( k ) } ) ) / ( \beta ( { \bf L } + { \bf L } ^ { \top } ) + 2 \mu ^ { ( k ) } { \bf I } )$； 6: 更新 $\mathbf { D }$ 为 $\mathbf { D } _ { i j } ^ { ( k + 1 ) } = \left\{ \begin{array} { l l } { s , ~ \mathrm { 如果 } ~ ( i , j ) \in \Omega _ { m } } \\ { - s , ~ \mathrm { 如果 } ~ ( i , j ) \in \Omega _ { c } } \\ { \mathbf { B } _ { i j } ^ { ( k ) } + \mathbf { Y } _ { 3 i j } ^ { ( k ) } / \mu ^ { ( k ) } , ~ \mathrm { 其他情况； } } \end{array} \right.$

7: 通过更新 $\mathbf { E }$ 进行。

$$
\mathbf { e } _ { j } ^ { ( k + 1 ) } = \left\{ \begin{array} { l l } { \displaystyle \frac { \left\| \mathbf { q } _ { j } ^ { ( k ) } \right\| _ { 2 } - \lambda / \mu ^ { ( k ) } } { \left\| \mathbf { q } _ { j } ^ { ( k ) } \right\| _ { 2 } } \mathbf { q } _ { j } ^ { ( k ) } , } & { \mathrm { i f } \left\| \mathbf { q } _ { j } ^ { ( k ) } \right\| _ { 2 } \ge \lambda / \mu ^ { ( k ) } } \\ { 0 , } & { \mathrm { o t h e r w i s e } ; } \end{array} \right.
$$

8：通过 9 更新 $\mathbf { Y } _ { 1 } , \mathbf { \mathcal { Y } } _ { 2 } , \mathbf { \mathcal { Y } } _ { 3 }$ 和 $\mu$ 直至收敛

$$
\left\{ \begin{array} { l } { \mathbf { Y } _ { 1 } ^ { ( k + 1 ) } = \mathbf { Y } _ { 1 } ^ { ( k ) } + \boldsymbol { \mu } ^ { ( k ) } \left( \mathbf { X } - \mathbf { X } \mathbf { Z } ^ { ( k + 1 ) } - \mathbf { E } ^ { ( k + 1 ) } \right) } \\ { \mathcal { Y } _ { 2 } ^ { ( k + 1 ) } ( : , : , 1 ) = \mathcal { Y } _ { 2 } ^ { ( k ) } ( : , : , 1 ) + \boldsymbol { \mu } ^ { ( k ) } \left( \mathbf { Z } ^ { ( k + 1 ) } - \mathcal { C } ^ { ( k + 1 ) } ( : , : , 1 ) \right) } \\ { \mathcal { Y } _ { 2 } ^ { ( k + 1 ) } ( : , : , 2 ) = \mathcal { Y } _ { 2 } ^ { ( k ) } ( : , : , 2 ) + \boldsymbol { \mu } ^ { ( k ) } \left( \mathbf { B } ^ { ( k + 1 ) } - \mathcal { C } ^ { ( k + 1 ) } ( : , : , 2 ) \right) } \\ { \mathbf { Y } _ { 3 } ^ { ( k + 1 ) } = \mathbf { Y } _ { 3 } ^ { ( k ) } + \boldsymbol { \mu } ^ { ( k ) } \left( \mathbf { B } ^ { ( k + 1 ) } - \mathbf { D } ^ { ( k + 1 ) } \right) } \\ { \boldsymbol { \mu } ^ { ( k + 1 ) } = \operatorname* { m i n } \left( \boldsymbol { \rho } \boldsymbol { \mu } ^ { ( k ) } ; \boldsymbol { \mu } _ { \operatorname* { m a x } } \right) ; } \end{array} \right.
$$

$$
\mathbf { B } _ { i j } = \left\{ \begin{array} { l l } { \mathrm { 1 , ~ i f ~ } ( i , j ) \in \Omega _ { m } } \\ { - 1 , \mathrm { ~ i f ~ } ( i , j ) \in \Omega _ { c } . } \end{array} \right.
$$

子空间聚类旨在将一组样本划分为多个子空间。特别地，基于自表达的子空间聚类方法引起了广泛关注，这些方法通过学习自表示矩阵作为相似度。例如，Liu 等人 [4] 提出了通过优化来学习低秩相似度矩阵，其中 $\mathbf { Z } \in \mathbb { R } ^ { n \times n }$ 是表示矩阵，$\textbf { E } \in \mathbb { R } ^ { d \times n }$ 表示重构误差，$\lambda ~ > ~ 0$ 是一个超参数。

$$
\operatorname* { m i n } _ { \mathbf { Z } , \mathbf { E } } \quad \| \mathbf { Z } \| _ { * } + \lambda \| \mathbf { E } \| _ { 2 , 1 } \mathrm { s . t . } \mathbf { X } = \mathbf { X } \mathbf { Z } + \mathbf { E } ,
$$

最近，许多半监督子空间聚类方法通过引入成对约束得以提出[5][13]。如何纳入监督信息是半监督子空间聚类的关键。通常，现有方法从局部视角引入成对约束，即在 $\mathbf { x } _ { i }$ 和 $\mathbf { x } _ { j }$ 存在必须连接（resp. 不能连接）时扩展（resp. 缩减） $\mathbf { Z } _ { i j }$ 的值。

# III. 提出的方法

# A. 模型公式化

如前所述，现有的半监督子空间聚类方法通常以简单的逐元素方式对相似度矩阵施加成对约束，这在某种程度上未充分利用监督信息。正如之前的研究所述，理想的相似度矩阵 $\mathbf{Z}$ 是低秩的，因为样本应仅由同类样本重构。同时，理想的成对约束矩阵 $\mathbf{B}$ 也是低秩的，因为它记录了样本之间的成对关系。此外，我们观察到它们的低秩结构应该是相同的。因此，如果我们将它们叠加形成一个 3-D 张量 $\mathcal{C} \in \mathbb{R}^{n \times n \times 2}$，即 ${\mathcal{C}}(:,:,: ,1) = \mathbf{Z}$，以及 $\mathcal{C}(:,:,: ,2) = \mathbf{B}$，则形成的张量 $\mathcal{C}$ 理想情况下应该是低秩的。因此，我们使用全局张量低秩范数来利用这一先验，初步将问题表述为

$$
\begin{array} { r l } { \underset { \mathcal { C } , \mathbf { E } , \mathbf { B } , \mathbf { Z } } { \mathrm { m i n } } } & { \| \mathcal { C } \| _ { \mathfrak { P } } + \lambda \| \mathbf { E } \| _ { 2 , 1 } } \\ { \mathrm { s . t . } } & { \mathbf { X } = \mathbf { X } \mathbf { Z } + \mathbf { E } , \mathcal { C } ( : , : , 1 ) = \mathbf { Z } , \mathcal { C } ( : , : , 2 ) = \mathbf { B } , } \\ & { \mathbf { B } _ { i j } = s , ( i , j ) \in \Omega _ { m } , \mathbf { B } _ { i j } = - s , ( i , j ) \in \Omega _ { c } . } \end{array}
$$

![](images/2.jpg)

![](images/3.jpg)

表 I：在$30 \%$初始标签下的准确率和NMI的详细比较。

<table><tr><td>Accuracy</td><td>ORL</td><td>YaleB</td><td>COIL20</td><td>Isolet</td><td>MNIST</td><td>Alphabet</td><td>BF0502</td><td>Notting-Hill</td><td>Average</td></tr><tr><td>LRR</td><td>0.7405</td><td>0.6974</td><td>0.6706</td><td>0.6699</td><td>0.5399</td><td>0.4631</td><td>0.4717</td><td>0.5756</td><td>0.6036</td></tr><tr><td>DPLRR</td><td>0.8292</td><td>0.6894</td><td>0.8978</td><td>0.8540</td><td>0.7442</td><td>0.7309</td><td>0.5516</td><td>0.9928</td><td>0.7862</td></tr><tr><td>SSLRR</td><td>0.7600</td><td>0.7089</td><td>0.7159</td><td>0.7848</td><td>0.6538</td><td>0.5294</td><td>0.6100</td><td>0.7383</td><td>0.6876</td></tr><tr><td>L-RPCA</td><td>0.6568</td><td>0.3619</td><td>0.8470</td><td>0.6225</td><td>0.5662</td><td>0.5776</td><td>0.4674</td><td>0.3899</td><td>0.5612</td></tr><tr><td>CP-SSC</td><td>0.7408</td><td>0.6922</td><td>0.8494</td><td>0.7375</td><td>0.5361</td><td>0.5679</td><td>0.4733</td><td>0.5592</td><td>0.6445</td></tr><tr><td>SC-LRR</td><td>0.7535</td><td>0.9416</td><td>0.8696</td><td>0.8339</td><td>0.8377</td><td>0.6974</td><td>0.7259</td><td>0.9982</td><td>0.8322</td></tr><tr><td>CLRR</td><td>0.8160</td><td>0.7853</td><td>0.8217</td><td>0.8787</td><td>0.7030</td><td>0.6837</td><td>0.7964</td><td>0.9308</td><td>0.8020</td></tr><tr><td>Proposed Method</td><td>0.8965</td><td>0.9742</td><td>0.9761</td><td>0.9344</td><td>0.8747</td><td>0.8355</td><td>0.8697</td><td>0.9934</td><td>0.9193</td></tr><tr><td>NMI</td><td>ORL</td><td>YaleB</td><td>COIL20</td><td>Isolet</td><td>MNIST</td><td>Alphabet</td><td>BF0502</td><td>Notting-Hill</td><td>Average</td></tr><tr><td>LRR</td><td>0.8611</td><td>0.7309</td><td>0.7742</td><td>0.7677</td><td>0.4949</td><td>0.5748</td><td>0.3675</td><td>0.3689</td><td>0.6175</td></tr><tr><td>DPLRR</td><td>0.8861</td><td>0.7205</td><td>0.9258</td><td>0.8853</td><td>0.7400</td><td>0.7477</td><td>0.5388</td><td>0.9748</td><td>0.8024</td></tr><tr><td>SSLRR</td><td>0.8746</td><td>0.7409</td><td>0.7986</td><td>0.8337</td><td>0.6373</td><td>0.6070</td><td>0.4810</td><td>0.5949</td><td>0.6960</td></tr><tr><td>L-RPCA</td><td>0.8038</td><td>0.3914</td><td>0.9271</td><td>0.7834</td><td>0.5805</td><td>0.6590</td><td>0.4329</td><td>0.2294</td><td>0.6009</td></tr><tr><td>CP-SSC</td><td>0.8705</td><td>0.7224</td><td>0.9583</td><td>0.8127</td><td>0.5516</td><td>0.6459</td><td>0.4453</td><td>0.4733</td><td>0.6850</td></tr><tr><td>SC-LRR</td><td>0.8924</td><td>0.9197</td><td>0.9048</td><td>0.8362</td><td>0.7803</td><td>0.7316</td><td>0.7068</td><td>0.9931</td><td>0.8456</td></tr><tr><td>CLRR</td><td>0.9028</td><td>0.7895</td><td>0.8568</td><td>0.8892</td><td>0.6727</td><td>0.7091</td><td>0.6970</td><td>0.8293</td><td>0.7933</td></tr><tr><td>Proposed Method</td><td>0.9337</td><td>0.9548</td><td>0.9716</td><td>0.9218</td><td>0.7825</td><td>0.8107</td><td>0.7693</td><td>0.9771</td><td>0.8902</td></tr></table>

在公式 (3) 中，我们采用在 tensor SVD 上定义的核范数 $\lVert \cdot \rVert _ { \circledast }$ [14] 来寻求低秩表示，其他类型的张量低秩范数也适用，例如 [15]。我们引入一个标量 $s$ 来约束 $\mathbf { B }$ 的最大值和最小值，促使 $\mathbf { B }$ 与 $\mathbf { Z }$ 具有类似的规模。经验上，$s$ 被设置为通过 LRR 学习得到的最大学习亲和度元素。通过求解公式 (3)，亲和矩阵 $\mathbf { Z }$ 和成对约束矩阵 $\mathbf { B }$ 根据 $\mathcal { C }$ 上的核范数共同优化，即监督信息被转移到 $\mathbf { Z }$，同时，学习到的亲和矩阵也可以从全局角度增强初始的成对约束。表 II：消融研究。

<table><tr><td>Percentage</td><td>Accuracy</td><td>ORL</td><td>YaleB</td><td>COIL20</td><td>Isolet</td><td>MNIST</td><td>Alphabet</td><td>BF0502</td><td>Notting-Hill</td><td>Average</td></tr><tr><td rowspan="5">10</td><td>SSLRR</td><td>0.7223</td><td>0.6965</td><td>0.6874</td><td>0.6107</td><td>0.5121</td><td>0.4278</td><td>0.4150</td><td>0.5747</td><td>0.5808</td></tr><tr><td>CLRR</td><td>0.7193</td><td>0.7032</td><td>0.6309</td><td>0.7424</td><td>0.5435</td><td>0.5120</td><td>0.5165</td><td>0.6728</td><td>0.6301</td></tr><tr><td>Eq. (3)</td><td>0.7298</td><td>0.7838</td><td>0.6744</td><td>0.8599</td><td>0.5224</td><td>0.5022</td><td>0.5786</td><td>0.8079</td><td>0.6824</td></tr><tr><td>Eq. (4)</td><td>0.7298</td><td>0.7838</td><td>0.8708</td><td>0.8424</td><td>0.7659</td><td>0.6640</td><td>0.5779</td><td>0.9573</td><td>0.7740</td></tr><tr><td>Eq. (5)</td><td>0.7523</td><td>0.8696</td><td>0.9171</td><td>0.8665</td><td>0.7879</td><td>0.6862</td><td>0.5915</td><td>0.9576</td><td>0.8036</td></tr><tr><td rowspan="5">20</td><td>SSLRR</td><td>0.7390</td><td>0.6998</td><td>0.6966</td><td>0.6651</td><td>0.5308</td><td>0.4672</td><td>0.4750</td><td>0.6363</td><td>0.6137</td></tr><tr><td>CLRR</td><td>0.7808</td><td>0.7130</td><td>0.6971</td><td>0.8176</td><td>0.6401</td><td>0.6064</td><td>0.6863</td><td>0.8598</td><td>0.7251</td></tr><tr><td>Eq. (3)</td><td>0.7860</td><td>0.9194</td><td>0.8101</td><td>0.9012</td><td>0.6661</td><td>0.6443</td><td>0.7554</td><td>0.9378</td><td>0.8025</td></tr><tr><td>Eq. 4</td><td>0.7860</td><td>0.9194</td><td>0.9364</td><td>0.9065</td><td>0.8366</td><td>0.7511</td><td>0.8077</td><td>0.9817</td><td>0.8657</td></tr><tr><td>Eq. (5)</td><td>0.8325</td><td>0.9548</td><td>0.9569</td><td>0.9078</td><td>0.8439</td><td>0.7772</td><td>0.8223</td><td>0.9831</td><td>0.8848</td></tr><tr><td rowspan="5">30</td><td>SSLRR</td><td>0.7600</td><td>0.7089</td><td>0.7159</td><td>0.7848</td><td>0.6538</td><td>0.5294</td><td>0.6100</td><td>0.7383</td><td>0.6876</td></tr><tr><td>CLRR</td><td>0.8160</td><td>0.7853</td><td>0.8217</td><td>0.8787</td><td>0.7030</td><td>0.6837</td><td>0.7964</td><td>0.9308</td><td>0.8020</td></tr><tr><td>Eq. (3)</td><td>0.8893</td><td>0.9664</td><td>0.9096</td><td>0.9222</td><td>0.8370</td><td>0.7671</td><td>0.8083</td><td>0.9661</td><td>0.8832</td></tr><tr><td>Eq. (4)</td><td>0.8893</td><td>0.9664</td><td>0.9710</td><td>0.9300</td><td>0.8745</td><td>0.8244</td><td>0.8631</td><td>0.9917</td><td>0.9138</td></tr><tr><td>Eq. (5)</td><td>0.8965</td><td>0.9742</td><td>0.9761</td><td>0.9344</td><td>0.8747</td><td>0.8355</td><td>0.8697</td><td>0.9934</td><td>0.9193</td></tr></table>

此外，如果两个样本 ${ \bf { x } } _ { i }$ 和 $\mathbf { x } _ { j }$ 在特征空间中彼此接近，我们可以预期它们具有相似的成对关系，即 $\mathbf { B } ( : , i )$ 接近于 $\mathbf { B } ( : , j )$ 。为了编码这一先验，我们首先构建一个 $k \mathbf { N N }$ 图 $\mathbf { W } \in \mathbb { R } ^ { n \times n }$ 来捕捉样本的局部几何结构，并使用局部拉普拉斯正则化 $\mathrm { T r } ( \mathbf { B L B } ^ { \top } )$ 来补充全局低秩项，其中 $ { \mathbf { L } } = { \mathbf { D } } - { \mathbf { W } }$ 是拉普拉斯矩阵，且 $\textstyle \mathbf { D } _ { i i } = \sum _ { j } \mathbf { W } _ { i j }$ [16]。因此，我们的模型最终表示为

$$
\begin{array} { r l } { \underset { \mathcal { C } , \mathbf { E } , \mathbf { B } , \mathbf { Z } } { \operatorname* { m i n } } } & { \| \mathcal { C } \| _ { \mathfrak { S } } + \lambda \| \mathbf { E } \| _ { 2 , 1 } + \beta \operatorname { T r } ( \mathbf { B } \mathbf { L } \mathbf { B } ^ { \top } ) } \\ { \mathrm { s . t . } } & { \mathbf { X } = \mathbf { X } \mathbf { Z } + \mathbf { E } , \mathcal { C } ( : , : , 1 ) = \mathbf { Z } , \mathcal { C } ( : , : , 2 ) = \mathbf { B } , } \\ & { \mathbf { B } _ { i j } = s , ( i , j ) \in \Omega _ { m } , \mathbf { B } _ { i j } = - s , ( i , j ) \in \Omega _ { c } . } \end{array}
$$

在解决公式 (4) 后，我们首先将 $\mathbf { Z }$ 的每一列归一化到 [0, 1]，并通过 $\mathbf { B } \gets \mathbf { B } / s$ 归一化 $\mathbf { B }$。然后，我们使用增广的成对约束矩阵 $\mathbf { B }$ 修复 $\mathbf { Z }$，即：

$$
\mathbf { Z } _ { i j }  \{ \begin{array} { l l } { 1 - ( 1 - \mathbf { B } _ { i j } ) ( 1 - \mathbf { Z } _ { i j } ) , \mathrm { i f } \mathbf { B } _ { i j } \geq 0 } \\ { ( 1 + \mathbf { B } _ { i j } ) \mathbf { Z } _ { i j } , \mathrm { i f } \mathbf { B } _ { i j } < 0 . } \end{array} 
$$

当 $\mathbf { B } _ { i j }$ 大于 0 时，$\mathbf { x } _ { i }$ 和 $\mathbf { x } _ { j }$ 很可能属于同一类，方程 (5) 会增加 $\mathbf { Z }$ 中相应元素的值。同样，当 $\mathbf { B } _ { i j }$ 小于 0 时，$\mathbf { Z } _ { i j }$ 的值会减小。因此，方程 (5) 通过增加的成对约束进一步增强了亲和矩阵。最后，我们在 $\mathbf { W } = ( | \mathbf { Z } | + | \mathbf { Z } ^ { \top } | ) / 2$ 上应用谱聚类 [17] 以获得子空间分割。

# B. 优化算法

由于方程（4）包含多个变量和约束，我们采用交替方向乘子法（ADMM）解决它。[18]算法1总结了整个伪代码。由于页面限制，详细的推导过程可以在补充材料中找到。

算法 1 的计算复杂度主要由步骤 3-5 决定。具体来说，步骤 3 求解一个 $n { \times } n { \times } 2$ 张量的 t-SVD，其复杂度为 $\mathcal { O } ( 2 n ^ { 2 } \log 2 \mathrm { + } 2 n ^ { 3 } )$ [14]。步骤 4-5 涉及矩阵求逆和矩阵乘法操作，复杂度为 $\mathcal { O } ( n ^ { 3 } )$。注意，在步骤 4 中，需要求逆的矩阵 $\left( \mathbf { I } + \mathbf { X } ^ { \top } \mathbf { X } \right)$ 是不变的，只需提前计算一次。综上所述，算法 1 在一次迭代中的整体计算复杂度为 $\mathcal { O } ( n ^ { 3 } )$。

![](images/4.jpg)  
Fig. 4: Visual comparison of the affinity matrices learned by different methods on MNIST. The learned affinity matrices were normalized to [0,1]. Zoom in the figure for a better view.

![](images/5.jpg)  
Fig. 5: Influence of the hyper-parameters on clustering performance.

# 四、实验

在本节中，我们在8个常用基准数据集上评估了所提模型，这些数据集包括ORL、YaleB、COIL20、Isolet、MNIST、Alphabet、BF0502和Notting-Hill1。这些数据集涵盖了人脸图像、物体图像、数字图像、口语字母和视频。为了生成弱监督信息，遵循[7]的做法，我们从随机选择的标签推断了一对一约束。

我们将我们的方法与LRR [4]以及六种最先进的半监督子空间聚类方法进行了比较，包括DPLRR [8]、SSLRR [7]、L-RPCA [19]、CP-SSC [20]、SCLRR [9]和CLRR [5]。我们在所有方法上执行了标准谱聚类 [17]，以生成聚类结果。我们采用聚类准确率和归一化互信息（NMI）来评估它们的性能。对于这两个指标，数值越大越好。为了确保公平比较，我们通过穷举网格搜索仔细调整了所比较方法的超参数，以获得最佳结果。为了全面评估不同的方法，对于每个数据集，我们随机选择了初始标签的不同百分比 $\left( \left\{ 5 \% , 1 0 \% , 1 5 \% , 2 0 \% , 2 5 \% , 3 0 \% \right\} \right)$ 以推断成对约束。在每种情况下，我们对所有比较的方法使用相同的标签信息。为了减少随机选择的影响，我们对随机选择的标签进行了10次实验，并报告了平均性能。

![](images/6.jpg)  
Cvegenc beavcaron  ifent to hatasetsTheondnal isora

![](images/7.jpg)  
Fig. 7: Running time comparisons of different methods on eight datasets.

# A. 聚类准确性比较

图 2-3 比较了在不同成对约束数量下不同方法的聚类准确率和归一化互信息（NMI），表 I 显示了在每个数据集上使用 $30\%$ 初始标签作为监督信息的不同方法的聚类性能。从这些图表中，我们可以得出以下结论。1) 随着成对约束数量的增加，所有半监督子空间聚类方法的表现普遍提升，这表明在子空间聚类中包含监督信息的有效性。2) 所提出的方法显著优于其他方法。例如，我们的方法在 MNIST 数据集上的准确率从 0.61 提高到 0.78，在 YaleB 数据集上的 NMI 值从 0.72 提升至 0.89，与最佳对比方法相比。根据表 I，所提出的方法将最佳对比方法的平均聚类准确率从 0.83 提高到 0.92。此外，所提出的方法几乎总是能够在不同的监督信息下取得最佳聚类性能。3) 这些对比方法可能对不同数据集敏感（例如，SC-LRR 在 YaleB 和 MNIST 上取得第二好成绩，但在 ORL 和 COIL20 上表现相对较差），并对多样的聚类指标敏感（例如，CP-SSC 在 NMI 上表现良好，但在聚类准确率上较差）。相反，所提出的方法对不同数据集和指标具有鲁棒性。此外，我们在图 4 中可视化了在 MNIST 上不同方法学习得到的相似性矩阵，结果表明我们的方法生成了更密集且正确的连接，导致了最显著的块对角相似性。这得益于所使用的全局张量低秩正则化，进一步解释了图 2-3 和表 I 中报告的良好聚类结果。

# B. 超参数分析

图5展示了两个超参数 $\lambda$ 和 $\beta$ 如何影响我们的方法在 COIL20、MNIST 和 Alphabet 上的表现。可以看出，所提出的模型对接近最优的超参数具有相对稳健性。具体来说，我们建议将 $\lambda$ 设置为 0.01 和 $\beta$ 设置为 10。

# C. 收敛速度

图6展示了所有比较算法在所有数据集上的收敛行为。注意，所有方法的收敛标准相同，即两个连续步骤中变量的残差误差小于1e—3。我们可以得出结论，L-PRCA通常在所有方法中收敛速度最快。而所提算法与SSLRR和DPLRR等其他方法相比也表现出较快的收敛速度。此外，所提算法在所有八个数据集上均在130次迭代内收敛。图7比较了所有八种方法在每个数据集上的平均运行时间。注意，我们在一台配备2.90 GHz Intel(R) i5-10400F CPU和16.0 GB内存的Windows桌面上使用MATLAB实现了所有方法。我们可以观察到，所提方法在所有八个数据集上的平均运行时间为95.97秒。这略高于LRR，但显著低于SSLRR。同时，我们还需指出，所提方法在聚类性能方面明显优于所比较的方法。

# D. 消融实验

我们通过比较公式 (3)-(5) 的聚类精度，研究了模型中所涉及的先验/过程的有效性。比较的方法包括两个知名的逐元素半监督子空间聚类方法 SSLRR 和 CLRR。如表 II 所示，公式 (3) 的结果在所有数据集上显著优于 SSLRR 和 CLRR，显示了全局张量低秩先验相比逐元素融合策略的优势。此外，公式 (4) 和 (5) 逐步提高了所提模型的性能，这表明图正则化和后处理都对我们的模型有促进作用。

# V. 结论

我们提出了一种新颖的半监督子空间聚类模型。我们首先将相似度矩阵和成对约束矩阵堆叠成一个张量，然后在其上施加张量低秩先验，以同时学习相似度矩阵并增强成对约束。除了全局张量低秩项外，我们还添加了拉普拉斯正则化项来建模潜在的局部几何结构。此外，学习到的相似度矩阵通过增强的成对约束进行精细化。所提模型被构造为一个凸问题，并通过交替方向乘子法（ADMM）求解。实验结果表明，我们的模型显著优于其他半监督子空间聚类方法。在未来，我们将研究如何将我们的工作与现有的半监督学习神经网络结合。例如，我们可以将提出的成对约束增强作为损失函数，以端到端的方式训练神经网络。此外，我们将通过解决噪声成对约束问题来改进我们的方法。

# REFERENCES

[1] R. Vidal, "Subspace clustering," IEEE SPM, vol. 28, no. 2, pp. 5268, 2011.   
[2] X. Peng, Y. Li, I. W. Tsang, H. Zhu, J. Lv, and J. T. Zhou, "Xai beyond cass Itepeabneul luste. op. 2022.   
[3] E. Elhamifar and R. Vidal, "Sparse subspace clustering: Algorithm, theory, and applications," IEEE TPAMI, vol. 35, no. 11, pp. 27652781, 2013.   
[4] G. Liu, Z. Lin, S. Yan, J. Sun, Y. Yu, and Y. Ma, "Robust recovery o subspace structures by low-rank representation," IEEE TPAMI, vol. 35, no. 1, pp. 171184, 2013.   
[5] J. Wang, X. Wang, F. Tian, C. H. Liu, and H. Yu, "Constrained low- rank representation for robust subspace clustering," IEEE TCYB, vol. 47, no. 12, pp. 45344546, 2017.   
[6] C. Yang, M. Ye, S. Tang, T. Xiang, and Z. Liu, "Semi-supervised lowrank representation for image classification," VP, vol. 11, no. 1, pp. 7380, 2017.   
[L. Zhua, Z. Zhou, S. Gao, J. Yin, Z. Lin, andY. Ma, "Label iormin guided graph construction for semi-supervised learning," IEEE TIP, vol. 26, no. 9, pp. 41824192, 2017.   
[8] H. Liu, Y. Jia, J. Hou, and Q. Zhang, "Learning low-rank graph with enhanced supervision," IEEE TCSVT, vol. 32, no. 4, pp. 25012506, 2022.   
[9] K. Tang, R. Liu, Z. Su, and J. Zhang, "Structure-constrained low-rank representation," IEEE TNNLS, vol. 25, no. 12, pp. 21672179, 2014.   
C.Zhou . Z X.  .hi  X. o, i 16.   
[11] C.-G. Li and R. Vidal, "Structured sparse subspace clustering: A unified optimization framework," in Proc. IE CVPR, 2015, pp. 277286.   
[12] Z. Zhang, Y. Xu, L. Shao, and J. Yang, "Discriminative block-diagonal representation learning for image recognition," IEEE TNNLS, vol. 29, no. 7, pp. 31113125, 2018. with side-information," in Proc. IEEE ICPR, 2018, pp. 20932099.   
[14] C. Lu, J. Feng, Y. Chen, W. Liu, Z. Lin, and S. Yan, "Tensor robust principal component analysis with a new tensor nuclear norm," IEEE TPAMI, vol. 42, no. 4, pp. 925938, 2020.   
[15] S. Wang, Y. Chen, L. Zhang, Y. Cen, and V. Voronin, "Hyper-laplacian regularized nonconvex low-rank representation for multi-view subspace clustering," IEEE TSIPN, vol. 8, pp. 376388, 2022.   
[16] Y. Chen, X. Xiao, and Y. Zhou, "Multi-view subspace clustering via smultaneousy learnng the representation tensor and afnity matrix," PR, vol. 106, p. 107441, 2020.   
[17] B. Peng, J. Lei, H. Fu, C. Zhang, T.-S. Chua, and X. Li, "Unsupervised video action clustering via motion-scene interaction constraint," IEEE TCSVT, vol. 30, no. 1, pp. 131144, 2020.   
[18 Y.-P. Zhao, L. Chen, and C. L. P. hen, "Laplacian regularized noative representation for clustering and dimensionality reduction," IEEE T, vol. 31, no. 1, pp. 114, 2021.   
[19] D. Zeng, Z. Wu, C. Ding, Z. Ren, Q. Yang, and S. Xie, "Labeled-robust regression: Simultaneous data recovery and classification," IEEE TCYB, vol. 52, no. 6, pp. 50265039, 2022.   
[20] K. Somandepalli and S. Narayanan, "Reinforcing self-expressive representation with constraint propagation for ace clustering in movies," in Proc. IEEE ICASSP, 2019, pp. 40654069.   
[21] W. Zhang, Q. M. J. Wu, and Y. Yang, "Semisupervised manifold regularization via a subnetwork-based representation learning model," IEEE TCYB, pp. 114, 2022.   
[22] H. Zhao, J. Zheng, W. Deng, and Y. Song, "Semi-supervised broad learning system based on manifold regularization and broad network," IEEE TCÁS-I, vol. 67, no. 3, pp. 983994, 2020.   
[23] Y. Li, P. Hu, Z. Liu, D. Peng, J. T. Zhou, and X. Peng, "Contrastive clustering," in Proc. AAAI, vol. 35, no. 10, 2021, pp. 85478555.   
[24] Y. Li, M. Yang, D. Peng, T. Li, J. Huang, and X. Peng, "Twin contrastive learning for online clustering," IJCV, vol. 130, no. 9, pp. 22052221, 2022