# 长序列推荐模型需要解耦嵌入

冯宁雅1 潘俊伟2；吴加龙 $\mathbf { W } \mathbf { u } ^ { 1 }$ • 陈白旭1，王希梅2，李乾 $\mathbf { L i } ^ { 2 }$ ，胡显 $\mathbf { H } \mathbf { u } ^ { 2 }$ 姜杰2，龙明晟1 1 清华大学 软件学院，BNRist，中国 2 腾讯公司，中国 fny21@mails.tsinghua.edu.cn, jonaspan@tencent.com, wujialong0229@gmail.com mingsheng@tsinghua.edu.cn

# 摘要

终身用户行为序列对于捕捉用户兴趣和预测用户响应在现代推荐系统中至关重要。通常采用一个两阶段的范式来处理这些长序列：第一阶段通过注意力机制从原始长序列中搜索一 subset 相关行为，然后在第二阶段将其与目标项目聚合以构建用于预测的区分性表示。在这项工作中，我们首次识别并描述了现有长序列推荐模型中的一个被忽视的缺陷：单一的嵌入集难以同时学习注意力和表示，导致这两个过程之间的干扰。最初尝试用一些常见方法（例如，线性投影——借用自语言处理的技术）来解决这个问题，但证明效果不佳，突显了推荐模型面临的独特挑战。为了解决此问题，我们提出了解耦注意力与表示嵌入（DARE）模型，其中两个独立的嵌入表被初始化并单独学习，完全解耦注意力和表示。大量实验和分析表明，DARE在相关行为搜索上更加准确，并在公共数据集上实现了高达 $9 \text{‰}$ 的 AUC 增益，且在腾讯的广告平台上有所显著提升。此外，解耦嵌入空间使我们能够降低注意力嵌入维度，并加速搜索过程达 $50 \%$，而不会显著影响性能，从而实现更高效的高性能在线服务。实验用的 PyTorch 代码，包括模型分析，可在 https://github.com/thuml/DARE 获取。

# 1 引言

在推荐系统中，内容提供者必须为不同的用户提供合适的项目。为了增强用户参与度，提供的项目应与用户的兴趣相一致，这可以通过他们的点击行为来证明。因此，目标项目的点击率（CTR）预测已成为一项基础任务。准确的预测在很大程度上依赖于有效捕捉用户历史行为中反映的用户兴趣。先前的研究表明，较长的用户历史有助于提供更加准确的预测（Pi et al., 2020）。因此，长序列推荐模型近年来引起了显著的研究兴趣（Chen et al., 2021；Cao et al., 2022）。在在线服务中，系统响应延迟可能严重扰乱用户体验，因此在有限时间内有效处理长序列至关重要。一种通用范式采用两阶段过程（Pi et al., 2020）：搜索（即通用搜索单元）和序列建模（即精确搜索单元）。该方法依赖于两个核心模块：注意力模块，测量目标与行为之间的相关性，以及表示模块，生成行为的可区分表示。搜索阶段使用注意力模块来检索前k个相关行为，从原始长行为序列中构建较短的子序列。序列建模阶段依赖于两个模块，通过根据注意力聚合子序列中的行为表示来预测用户反应，从而提取可区分的表示。现有的工作广泛采用这一范式（Pi et al., 2020；Chang et al., 2023；Si et al., 2024）。

![](images/1.jpg)  

Figure 1: Overview of our work. During search, only a limited number of important behaviors are retrieved according to their attention scores. During sequence modeling, the selected behaviors are aggregated into a discriminative representation for prediction. Our DARE model decouples the embeddings used in attention calculation and representation aggregation, effectively resolving their conflict and leading to improved performance and faster inference speed.

注意力在长序列推荐中至关重要，因为它不仅为序列建模建模了每个行为的重要性，更重要的是决定了在搜索阶段选择哪些行为。然而，在大多数现有工作中，注意力和表示模块共享相同的嵌入，尽管它们服务于不同的功能——一个学习相关性评分，另一个学习区分性表示。我们首次从多任务学习（MTL）的角度分析这两个模块（Caruana, 1997）。采用MTL中常用的梯度分析方法（Yu et al., 2020；Liu et al., 2021），我们揭示不幸的是，这些共享嵌入的梯度被表示主导，更令人关注的是，两个模块的梯度方向往往彼此冲突。梯度的主导和冲突是任务间干扰的典型现象，影响模型在两个任务上的性能。我们的实验证据与理论见解一致：注意力未能准确捕捉行为的重要性，导致关键行为在搜索阶段被错误过滤（如第4.3节所示）。此外，梯度冲突还降低了表示的可区分性（如第4.4节所示）。受到原始自注意力机制中使用独立的查询、键（用于注意力）和值（用于表示）投影矩阵的启发（Vaswani et al., 2017），我们在推荐模型中实验了特定于注意力和表示的投影，旨在解决这两个模块之间的冲突。然而，这种方法没有产生积极的结果。我们还尝试了三种其他候选方法，但不幸的是，它们都未有效工作。通过深刻的实证分析，我们假设失败是由于推荐模型中的投影矩阵的容量显著较低（即参数较少），相比于自然语言处理（NLP）中的投影矩阵。这一限制难以克服，因为它来源于交互崩溃理论所施加的低嵌入维度（Guo et al., 2024）。为了解决这些问题，我们提出了去耦合注意力和表示嵌入（DARE）模型，该模型通过使用两个独立的嵌入表——一个用于注意力，另一个用于表示，完全将这两个模块在嵌入层面上解耦。这种去耦合使我们能够充分优化注意力以捕捉相关性和表示以增强可区分性。此外，通过分离嵌入，我们可以通过将注意力嵌入维度减半，快速加速搜索阶段，提升达 $50 \%$，对性能影响最小。在公共的淘宝和天猫长序列数据集上，DARE在所有嵌入维度上均优于最先进的TWIN模型，AUC提升高达 $9 \text{‰}$。在腾讯的广告平台进行的在线评估（全球最大的广告平台之一）中实现了 $1 . 4 7 \%$ 的GMV（总商品价值）提升。我们的贡献可以总结如下： • 我们识别了现有长序列推荐模型中注意力与表示学习间干扰的问题，并证明了常见方法（例如，借用自NLP的线性投影）未能有效解耦这两个模块。 • 我们提出了DARE模型，使用模块特定的嵌入完全解耦注意力和表示。我们的综合分析表明，我们的模型显著提高了注意力准确性和表示可区分性。 • 我们的模型在两个公共数据集上实现了最先进的结果，并在全球最大的推荐系统之一中实现了 $1 . 4 7 \%$ 的GMV提升。此外，我们的方法在减少去耦注意力嵌入大小的同时，显著加速了搜索阶段。

# 2 对注意力机制和表示的深入分析

在这一部分，我们首先回顾长序列推荐的一般公式。然后，我们分析共享嵌入的训练，重点讨论来自注意力和表示模块的梯度的主导性和冲突。最后，我们探讨为何简单的方法（例如，使用模块特定的投影矩阵）无法解决这个问题。

# 2.1 初步知识

问题表述。我们考虑基础任务，即点击率（CTR）预测，旨在根据用户的行为历史预测用户是否会点击特定的目标项。这通常被表述为二分类问题，学习一个预测函数 $f : \mathcal { X } \mapsto [ 0 , 1 ]$，给定训练数据集 $\mathcal { D } = \{ ( \mathbf { x } _ { 1 } , y _ { 1 } ) , \dotsc , ( \mathbf { x } _ { | \mathcal { D } | } , y _ { | \mathcal { D } | } ) \}$，其中 $\mathbf { x }$ 包含一系列表示行为历史的项和一个表示目标的单个项。

长序列推荐模型。为了满足在线服务中严格限制的推理时间，目前的长序列推荐模型一般通过检索与之相关的前 $\mathbf { \nabla } \cdot \mathbf { k }$ 个行为，首先构建一个短序列。注意力得分通过行为和目标嵌入的缩放点积来计算。形式上，第 $i$ 个历史行为和目标 $t$ 被嵌入为 $e _ { i }$ 和 $\boldsymbol { v } _ { t } \in \mathbb { R } ^ { \breve { d } }$，并且不失一般性，有 $1 , 2 , \dotsc , K = \mathrm { T o p - K } ( \langle e _ { i } , { \pmb v } _ { t } \rangle , i \in [ 1 , N ] )$，其中 $\langle \cdot , \cdot \rangle$ 代表点积。然后，每个行为的权重 $w _ { i }$ 通过softmax函数计算：$\begin{array} { r } { w _ { i } = \frac { e ^ { \langle { \pmb { e } _ { i } } , { \pmb { v } _ { t } } \rangle / \sqrt { d } } } { \sum _ { j = 1 } ^ { K } e ^ { \langle { \pmb { e } _ { j } } , { \pmb { v } _ { t } } \rangle / \sqrt { d } } } } \end{array}$ 将通过精湛的工业优化转化为性能。

# 2.2 统治与冲突的梯度分析

注意力模块和表示模块可以看作是两个任务：前者专注于学习行为的相关性评分，而后者则专注于在高维空间中学习可区分的（即可分离的）表示。然而，目前的方法对这两个任务使用了共享的嵌入，这可能导致类似于多任务学习（MTL）中的“任务冲突”现象（Yu等，2020；Liu等，2021），并阻止它们中的任何一个被完全实现。为了验证这一假设，我们分析了从两个模块到共享嵌入的梯度。实验验证。遵循MTL中的方法，我们实证观察到从注意力模块和表示模块反向传播到嵌入的梯度。比较其梯度范数，我们发现来自表示的梯度大约是来自注意力的五倍，主导了后者，如图2所示。在观察它们的梯度方向时，我们进一步发现，近三分之二的情况下，梯度角的余弦值为负，这表明它们之间存在冲突，如图3所示。主导和冲突是任务干扰的两种典型现象，暗示了在充分学习这些任务时面临的挑战。

![](images/2.jpg)  

Figure 2: The magnitude of embedding gradients from the attention and representation modules.

总之，注意力模块和表示模块在训练过程中以不同的方向和强度优化嵌入表，导致注意力丧失相关性准确性，而表示失去可区分性。值得注意的是，由于主导效应，这种影响对注意力的影响更加严重，正如第4.3节中所指出的类别之间学习的相关性较差。尽管在多任务学习中一些常用的技术可能缓解冲突，但我们倾向于寻求一种进一步解决冲突的优化模型结构。

![](images/3.jpg)  

Figure 3: Cosine angles of gradients.

发现 1. 在顺序推荐系统中，表示模块的梯度往往与注意力模块的梯度发生冲突，并通常主导嵌入梯度。

![](images/4.jpg)  

Figure 4: Illustration and evaluation for adopting linear projections. (a-b) The attention module in the original TWIN and after adopting linear projections. (c) Performance of TWIN variants. Adopting linear projections causes an AUC drop of nearly $2 \%$ on Taobao.

# 2.3 推荐模型需要更强大的解耦方法

正常的解耦方法无法解决冲突。为了应对这种冲突，一种简单的方法是为注意力和表示使用单独的投影，将原始嵌入映射到两个新的解耦空间。这在标准的自注意力机制中被采用（Vaswani 等，2017），该机制引入了查询、键（用于注意力）和数值投影矩阵（用于表示）。受到此启发，我们提出了一种变体 TWIN，利用线性投影来解耦注意力和表示模块，命名为 TWIN（w/ proj.）。与原始 TWIN 结构的比较如图 4a 和 4b 所示。令人惊讶的是，在线性投影在自然语言处理中的效果良好，但在推荐系统中却失去了效果，导致了负面的性能影响，如表 4c 所示。我们还尝试了三种其他候选方法（基于 MLP 的投影、增强线性投影的能力以及梯度归一化），共形成了八个模型，但没有一种有效地解决冲突。有关这些模型的结构和更多细节，请参见附录 C。

更大的嵌入维度使得线性投影在自然语言处理（NLP）中有效。引入投影矩阵的失败让我们产生疑问，为何它在NLP中效果良好而在推荐系统中却不然。一个可能的原因是，相较于NLP中的词元数量，投影矩阵的相对容量通常很强。例如，在LLaMA3.1中，嵌入维度为4096（Dubey et al., 2024），每个投影矩阵具有大约1600万个参数（$4096 \times 4096 = 16,777,216$），而仅需映射128,000个词汇中的词元。为了验证我们的假设，我们使用nanoGPT（Andrej）在NLP中进行了一项合成实验，采用莎士比亚数据集。具体而言，我们将其嵌入维度从128降至2，并检查带有/不带有投影矩阵的两个模型之间的性能差距。如图5所示，我们观察到当矩阵具有足够的容量时，即嵌入维度大于16时，投影导致显著更小的损失。然而，当矩阵容量进一步降低时，这一差距消失。我们的实验表明，仅当容量足够时，使用投影矩阵才有效。

![](images/5.jpg)  

Figure 5: The influence of linear projections with different embedding dimensions in NLP.

![](images/6.jpg)  

Figure 6: Architecture of the proposed DARE model. One embedding is responsible for attention, learning the correlation between the target and history behaviors, while another embedding is responsible for representation, learning discriminative representations for prediction. Decoupling these two embeddings allows us to resolve the conflict between the two modules.

受限的嵌入维度导致线性投影在推荐系统中失效。相反，由于交互崩溃理论（Guo 等，2024），推荐中的嵌入维度通常不超过 200，这使得每个矩阵最多只有 40000 个参数来映射数百万到数十亿的 ID。因此，推荐系统中的投影矩阵从未获得足够的容量，导致它们无法解耦注意力和表示。在这种情况下，附录 C 中提到的其他常规解耦方法也遭遇了容量不足的问题。发现 2. 由于嵌入维度造成的有限容量，线性投影等常规方法在顺序推荐模型中无法解耦注意力和表示。

# 3 DaRE: 解耦注意力与表示嵌入

由于附录 C 中展示的所有八种正常解耦模型均未能成功，基于我们的分析，我们寻求具有足够能力的方法，希望能够彻底解决冲突。为此，我们提议在嵌入层面解耦这两个模块。也就是说，我们采用两个嵌入表，一个用于注意力 $( E ^ { \mathrm { A t t } } )$，另一个用于表示 $( E ^ { \mathrm { R e p r } } )$。通过将梯度反向传播到不同的嵌入表，我们的方法有潜力彻底解决这两个模块之间的梯度主导和冲突。本节将具体介绍我们的模型，并在下一节通过实验展示其优势。

# 3.1 注意力嵌入

注意力度量历史行为与目标之间的相关性（Zhou et al., 2018）。遵循常见做法，我们使用缩放点积函数（Vaswani et al., 2017）。在数学上，第 $i$ 个历史行为 $i$ 和目标 $t$ 被嵌入到 $e _ { i } ^ { \mathrm { A t t } } , v _ { t } ^ { \mathrm { A t t } } \sim E ^ { \mathrm { A t t } }$ 中，其中 $E ^ { \mathrm { A t t } }$ 是注意力嵌入表。检索后 $1 , 2 , \ldots , K = \mathrm { T o p } { \mathrm { - } } \dot { \mathrm { K } } ( \langle e _ { i } , { \pmb v } _ { t } \rangle , i \in [ 1 , N ] )$，它们的权重 $w _ { i }$ 被形式化为：

$$
w _ { i } = \frac { { e ^ { \langle { { e _ { i } ^ { \mathrm { A t t } } } , { \bf { v } } _ { t } ^ { \mathrm { A t t } } } \rangle / \sqrt { \lvert { \cal E } ^ { \mathrm { A t t } } } \rvert } } } { { \sum _ { j = 1 } ^ { K } { e ^ { \langle { { e _ { j } ^ { \mathrm { A t t } } } , { \bf { v } } _ { t } ^ { \mathrm { A t t } } } \rangle / \sqrt { \lvert { \cal E } ^ { \mathrm { A t t } } } \rvert } } } } ,
$$

其中 $\langle \cdot , \cdot \rangle$ 代表点积，$| E ^ { \mathrm { A t t } } |$ 代表嵌入维度。

# 3.2 表示嵌入

在表示部分，使用了另一个嵌入表 $E ^ { \mathrm { R e p r } }$，其中 $i$ 和 $t$ 被嵌入到每个检索行为的 $x$ 中，然后与目标的嵌入拼接，作为多层感知机（MLP）的输入：$\textstyle \left[ \sum _ { i } w _ { i } e _ { i } , { \boldsymbol { v } } _ { t } \right]$。然而，已有研究证明 MLP 难以有效学习显式交互（Rendle 等，2020；Zhai 等，2023）。为了增强 $e _ { i } ^ { \mathrm { R e p r } } \odot v _ { t } ^ { \mathrm { R e p r } }$ 的辨别能力，在我们后续论文中将其称为 TR（参见第 4.4 节关于可辨别性的实证评价）。我们模型的整体结构如图 6 所示。正式地，用户历史 $h$ 被压缩为：$\begin{array} { r } { \pmb { h } = \sum _ { i = 1 } ^ { K } w _ { i } \cdot ( \pmb { e } _ { i } ^ { \mathrm { R e p r } } \odot \pmb { v } _ { t } ^ { \mathrm { R e p r } } ) } \end{array}$

# 3.3 推理加速

通过解耦注意力和表示嵌入表，注意力嵌入的维度 $E ^ { \mathrm { A t t } }$ 和表示嵌入的维度 $E ^ { \mathrm { R e p r } }$ 具有更大的灵活性。特别是，我们可以减少 $E ^ { \mathrm { A t t } }$ 的维度，同时保持 $E ^ { \mathrm { R e p r } }$ 的维度，以加快对原始长序列的搜索，同时不影响模型的性能。第 4.5 节的实验证明，我们的模型有潜力在性能影响很小的情况下将搜索速度提高 $50 \%$，甚至在可接受的性能损失下提高 $75 \%$。

# 3.4 讨论

考虑到将注意力和表示嵌入解耦的优越性，自然会提出一个想法：我们可以进一步在注意力（和表示）模块中解耦历史和目标的嵌入，即形成一种称为 TWIN-4E 的四个嵌入方法，包含注意力历史（在 NLP 中称为键）$e _ { i } ^ { \mathrm { A t t } } \in E ^ { \mathrm { A t t - h } }$，注意力目标（在 NLP 中称为查询）$\pmb { v } _ { t } ^ { \mathrm { A t t } } \in \pmb { E } ^ { \mathrm { A t t - t } }$，表示历史（在 NLP 中称为值）$e _ { i } ^ { \mathrm { R e p r } } \in E ^ { \mathrm { R e p r - h } }$，以及表示目标$v _ { t } ^ { \mathrm { R e p r } } \in E ^ { \mathrm { R e p r - t } }$。 TWIN-4E 的结构如图 7 所示。与我们的 DARE 模型相比，TWIN-4E 进一步解耦了行为和目标，意味着同一类别或项目具有两个完全独立的嵌入作为行为和目标。这与推荐系统中的两项先验知识强烈对立。1. 无论哪个是目标，哪个是历史，两个行为的相关性是相似的。2. 同一类别的行为应具有更高的相关性，这在 DARE 中是合理的，因为向量与自身的点积通常更大。

![](images/7.jpg)  

Figure 7: Illustration of the TWIN-4E model.

# 4 实验

# 4.1 设置

数据集和任务。我们使用公开可用的淘宝（Zhu et al., 2018; 2019; Zhuo et al., 2020）和天猫（Tianchi, 2018）数据集，这些数据集提供了用户在其平台上特定时间段内的行为数据。每个数据集包括用户点击的商品，这些商品用商品 ID 和相应的类别 ID 表示。因此，用户的历史被建模为商品 ID 和类别 ID 的序列。模型的输入由用户终身历史中的最近连续子序列以及一个目标商品组成。对于正样本，目标商品是用户下一次实际点击的商品，模型预期输出“是”。对于负样本，目标商品是随机抽样的，模型应输出“否”。除了这些公共数据集，我们还在全球最大的在线广告平台之一验证了我们的性能。关于数据集和训练/验证/测试划分的更多细节请参见附录 B。

基线对比。我们与多种推荐模型进行比较，包括 ETA（Chen et al., 2021）、SDIM（Cao et al., 2022）、DIN（Zhou et al., 2018）、TWIN（Chang et al., 2023）及其变体，以及 TWIN-V2（Si et al., 2024）。如第 3.2 节所讨论，通过交叉 $e _ { i } ^ { \mathrm { R e p r } } \odot v _ { t } ^ { \mathrm { R e p r } }$ 实现的目标感知表示显著提高了表示的区分性，因此我们将其纳入基线中以确保公平性。TWIN-4E 指的是第 3.4 节中介绍的模型，而 TWIN（w/ proj.）指的是第 2.3 节中描述的模型。TWIN（hard）表示在搜索阶段使用“硬搜索”的变体，意味着它只从与目标相同类别中检索行为。TWIN（w/o TR）指的是没有目标感知表示的原始 TWIN 模型，即将用户历史表示为 $\begin{array} { r } { \pmb { h } = \sum _ { i } { w _ { i } } \cdot \pmb { e } _ { i } } \end{array}$ 而不是 $\begin{array} { r } { \pmb { h } = \sum _ { i } w _ { i } ( \pmb { e } _ { i } \hat { \odot } \pmb { v } _ { t } ) } \end{array}$ 。

Table 1: Overall comparison reported by the means and standard deviations of AUC. The best results are highlighted in bold, while the previous best model is underlined. Our model outperforms all existing methods with obvious advantages, especially with small embedding dimensions.   

<table><tr><td>Setup</td><td colspan="2">Embedding Dim. = 16</td><td colspan="2">Embedding Dim. = 64</td><td colspan="2">Embedding Dim. = 128</td></tr><tr><td>Dataset</td><td>Taobao</td><td>Tmall</td><td>Taobao</td><td>Tmall</td><td>Taobao</td><td>Tmall</td></tr><tr><td>ETA (2021)</td><td>0.91326 (0.00338)</td><td>0.95744</td><td>0.92300 0.00079)</td><td>0.96658 (0.00042)</td><td>0.92480 (0.00032)</td><td>0.96956 (0.00039)</td></tr><tr><td>SDIM (2022)</td><td>0.90430</td><td>0.00108) 0.93516</td><td>0.90854</td><td>0.94110</td><td>0.91108</td><td>0.94298</td></tr><tr><td>DIN (2018)</td><td>(0.0103) 0.90442</td><td>(0.00069) 0.95894</td><td>(0.00085) 0.90912</td><td>(0.00093) 0.96194</td><td>(0.00119) 0.91078</td><td>(0.00081) 0.96428</td></tr><tr><td>TWIN (2023)</td><td>(0.00060) 0.91688</td><td>0.0037) 0.95812</td><td>(0.00092) 0.92636</td><td>(0.00033) 0.96684</td><td>(0.00054) 0.93116</td><td>(0.00013) 0.97060</td></tr><tr><td>TWIN (hard)</td><td>(0.00211) 0.91002</td><td>(0.00073) 0.96026</td><td>(0.00052) 0.91984</td><td>(0.00039) 0.96448</td><td>(0.00056) 0.91446</td><td>(0.00005) 0.96712</td></tr><tr><td>TWIN (w/ proj.)</td><td>(0.00053) 0.89642</td><td>(0.00024) 0.96152</td><td>(0.00048) 0.87176</td><td>(0.00042) 0.95570</td><td>(0.00055) 0.87990</td><td>(0.00019) 0.95724</td></tr><tr><td>TWIN (w/o TR)</td><td>(0.00351) 0.90732</td><td>(0.00088) 0.96170</td><td>(0.00437) 0.91590</td><td>(0.00403) 0.96320</td><td>(0.02022) 0.92060</td><td>(0.00194) 0.96366</td></tr><tr><td>TWIN-V2 (2024)</td><td>(0.00063) 0.89434</td><td>(0.00057) 0.94714</td><td>(0.00083) 0.90170</td><td>(0.00032) 0.95378</td><td>(0.00084) 0.90586</td><td>(0.00103) 0.95732</td></tr><tr><td>TWIN-4E</td><td>(0.00077) 0.90414</td><td>(0.00110) 0.96124</td><td>(0.00063) 0.90356</td><td>(0.00037) 0.96372</td><td>(0.00059) 0.90946</td><td>(0.00045) 0.96016</td></tr><tr><td>DARE (Ours)</td><td>(0.01329) 0.92568 (0.00025)</td><td>(0.00026) 0.96800 (0.00024)</td><td>(0.01505) 0.92992 (0.00046)</td><td>(0.0004) 0.97074 (0.00012)</td><td>(0.01508) 0.93242 (0.00045)</td><td>(0.01048) 0.97254 (0.00016)</td></tr></table>

# 4.2 整体性能

在推荐系统中，人们普遍认识到即使将AUC提高$1 \text{‰}$到$2 \text{‰}$也足以带来在线收益。如表1所示，我们的模型在不同嵌入大小的各种设置下，相比当前最先进的方法实现了$1 \text{‰}$和$9 \text{‰}$的AUC提升。特别是在Taobao和Tmall数据集中，嵌入维度为16时，分别观察到了$9 \text{‰}$和$6 \text{‰}$的显著AUC提升。同时，还有一些显著的发现。TWIN在大多数情况下优于TWIN (不包含TR)，证明了第4.4节所显示的$\text{tarewa eion } e_{i}^{\mathrm{Rep r}} \odot v_{t}^{\mathrm{Rep r}}$的有效性。我们的DARÉ模型明显优于TWIN-4E，证实了第3.4节讨论的先验知识非常适合推荐系统。基于TWIN并致力于以牺牲性能为代价加速搜索阶段的ETA和SDM，表现出可理解的较低AUC分数。针对视频推荐进行优化的领域特定方法TWIN-V2在我们的设置中效果较差。

# 4.3 注意力准确性

互信息捕捉了两个变量之间的共享信息，是理解数据关系的强大工具。我们计算行为与目标之间的互信息作为真实标注数据的相关性，遵循（Zhou et al., 2024）。学习到的注意力得分反映了模型对每个行为重要性的度量。因此，我们在图8中比较了注意力分布与互信息。特别是，图8a展示了目标类别与前10个类别的行为之间的互信息及其目标相关位置（即，行为在时间上与目标的接近程度）。我们观察到强烈的语义-时间相关性：来自与目标同类别的行为（第5行）通常相关性更高，并展现出明显的时间衰减模式。图8b展示了TWIN学习到的注意力得分，显示出良好的时间衰减模式，但高估了不同类别行为之间的语义相关性，使其对近期行为过于敏感，即使是来自不相关类别的行为。相比之下，我们提出的DARE能够有效捕捉时间衰减与语义模式。检索阶段完全依赖于注意力得分。因此，我们进一步调查了在测试数据集上的检索，这提供了更直观的注意力质量反映。具有前k个互信息的行为被视为最佳检索，我们使用归一化折扣累积增益（NDCG）（Järvelin & Kekäläinen, 2002）评估模型性能。结果以及案例研究显示在图9中（更多示例见附录E.4）。我们发现：

![](images/8.jpg)  

Figure 8: The ground truth (GT) and learned correlation between history behaviors of top-10 frequent categories (y-axis) at various positions ( $\mathbf { \dot { X } }$ -axis), with category 15 as the target. Our correlation scores are noticeably closer to the ground truth.

![](images/9.jpg)  

Figure 9: Retrieval in the search stage. (a) Our model can retrieve more correlated behaviors. (b-c) Two showcases where the $\mathbf { X }$ -axis is the categories of the recent ten behaviors.

• DARE 在检索方面表现显著更优。如图 9a 所示，我们模型的 NDCG 值明显高于所有基线，较 TWIN 提高了 $4 6 . 5 \%$ （0.8124 对比 0.5545），较 DIN 提高了 $2 7 . 3 \%$ （0.8124 对比 0.6382）。 • TWIN 对时间信息过于敏感。如前所述，TWIN 倾向于选择最新的行为，而不考虑其类别，这与真实标注数据相悖，原因在于不同类别之间的相关性被高估，如图 9b 和 9c 所示。 • 其他方法表现不稳定。对于其他方法，它们在很多情况下过滤掉了一些重要行为，并检索出不相关的行为，这解释了它们表现不佳的原因。 结果 1. DARE 成功捕捉了行为与目标之间的语义-时间相关性，在搜索阶段保留了更多相关行为。

# 4.4 表示区分能力

我们分析了学习表示的可区分性。在测试数据集上，我们取样本 com$\begin{array} { r } { \pmb { h } = \sum _ { i = 1 } ^ { K } \pmb { w } _ { i } \cdot \left( \pmb { e } _ { i } \odot \pmb { v } _ { t } \right) } \end{array}$。使用K均值算法，我们对这些向量进行量化，将每个向量映射到一个聚类 $Q(h)$。离散变量 $Q(h)$ 与标签 $Y$（目标是否被点击）之间的互信息（MI）可以反映表示的可区分性：可区分性 $(h, Y) = {\bf MI} \bar{(} Q(h), Y )$。如图10a所示，在不同的聚类数量下，我们的DARE模型优于最先进的TWIN模型，证明解耦提高了表示的可区分性。还有其他显著发现。尽管DIN在搜索阶段实现了更精确的检索（如图9a中更高的NDCG所示），但其表示的可区分性明显低于TWIN，特别是在淘宝数据集上，这解释了其整体表现较低。TWIN-4E的可区分性与我们的DARE模型相当，进一步确认其较差表现是由于缺乏推荐特定先验知识造成的不准确注意力。为了充分展示 $e_{i} \odot v_{t}$ 的有效性，我们将其与经典的拼接方式 $[ \Sigma_{i} . e_{i}, \dot{ \mathbf{v} }_{t} ]$ 进行了比较。如图10c所示，目标感知表示造成了巨大的差距（橙色），而较小的差距（蓝色和绿色）则是解耦造成的。值得注意的是，即使使用拼接，我们的DARE模型也优于TWIN。

![](images/10.jpg)  

Figure 10: Representation discriminability of different models, measured by the mutual information between the quantized representations and labels.

![](images/11.jpg)  

Figure 11: Efficiency during training and inference. (a-b) Our model performs obviously better with fewer training data. (c-d) Reducing the search embedding dimension, a key factor of online inference speed, has little influence on our model, while TWIN suffers an obvious performance loss.

结果 2. 在 DARE 模型中，目标感知表示的形式和嵌入解耦显著提高了表示的可区分性。

# 4.5 收敛性与效率

训练期间更快速的收敛。在推荐系统中，更快的学习速度意味着模型能够在更少的训练数据上达到良好的性能，这对于在线服务尤其重要。我们在训练过程中跟踪验证数据集上的准确率，如图11a所示。我们的DARE模型收敛明显更快。例如，在Tmall数据集上，TWIN在超过1300次迭代后达到90%的准确率。相比之下，我们的DARE模型仅需约450次迭代即可达到可比较的性能——这是TWIN所需时间的三分之一。

推理阶段中的高效搜索。通过解耦注意力嵌入空间 $e_{i}, v_{t} \in \mathbb{R}^{K_{A}}$ 和表示嵌入空间 $e_{i}, \mathbf{\boldsymbol{v}}_{t} \in \dot{\mathbb{R}}^{K_{R}}$，我们可以为这两个空间分配不同的维度。从经验上看，我们发现注意力模块在较小嵌入维度下的表现相当良好，这使我们能够减小注意力空间的大小 $(K_{A} \ll K_{R})$ 并显著加速搜索阶段，因为其复杂度为 $O(K_{A} N)$，其中 $N$ 是用户历史的长度。以 $K_{A}=128$ 为基线（“1”），我们对较小嵌入维度的复杂度进行了归一化。图 11c 显示我们的模型可以在对性能影响很小的情况下加速搜索速度 $50\%$，并且在可接受的性能损失情况下，速度提升可达 $75\%$，为实际应用提供了更灵活的选择。相比之下，TWIN 在减少嵌入维度时会经历显著的 AUC 下降。结果 3. 嵌入解耦可以在不显著影响 AUC 的情况下，实现模型训练收敛更快和至少 $50\%$ 的推理加速。

# 4.6 在线A/B测试与部署

我们将我们的方法应用于腾讯的广告平台。由于用户在广告上的行为稀疏，使得序列长度相对于内容推荐场景较短，我们引入了用户在我们文章中的行为序列以及微视频推荐场景。具体来说，将过去两年用户的广告行为和内容行为纳入考虑。在搜索之前，广告和内容序列的最大长度分别为4000和6000，平均为170和1500。采用DARE算法搜索后，序列长度减少到500以下。关于序列特征（辅助信息），我们选择了类别ID、行为类型ID、场景ID，以及两个目标感知的时间编码，即相对目标的位置和相对目标的时间间隔（经过离散化处理）。每天大约有10亿个训练样本。在2024年9月的为期5天的在线A/B测试中，所提出的DARE方法实现了$0 . 5 7 \%$的成本降低和$1 . 4 7 \%$的GMV（总商品价值）提升，相较于TWIN的生产基线。这将导致每年数亿美元的收入增长。

# 4.7 附录中的补充实验结果

检索阶段的检索数量。DARE的优势在于较少的检索数量，更加明确地证明了DARE对重要行为的选择更加准确（附录D.1）。序列长度与短序列建模。DARE在处理较长序列时始终受益，而在短序列建模中表现出的优势则相对较小（附录D.2）。GAUC和Logloss。除了AUC外，我们还在GAUC和Logloss下评估了DARE及所有基线模型。DARE显示出持续的优越性，证明了我们结果的可靠性（附录E.1）。

# 5 相关工作

点击率预测与长序列建模。点击率（CTR）预测在推荐系统中是基础，因为用户兴趣往往通过其点击行为反映出来。深度兴趣网络（DIN）(Zhou et al., 2018) 引入了目标感知注意力，利用多层感知机（MLP）学习与特定目标相关的每个历史行为的关注权重。该框架通过像 DIEN (Zhou et al., 2019)、DSIN (Feng et al., 2019) 和 BST (Chen et al., 2019) 等模型得到了扩展，以更好地捕捉用户兴趣。研究已证明，较长的用户历史会导致更准确的预测，从而使长序列建模成为焦点。SIM (Pi et al., 2020) 引入了搜索阶段（GSU），大大加速了序列建模阶段（ESU）。像 ETA (Chen et al., 2021) 和 SDIM (Cao et al., 2022) 等模型进一步改善了这一框架。特别值得注意的是，TWIN (Chang et al., 2023) 和 TWIN-V2 (Si et al., 2024) 统一了两个阶段中使用的目标感知注意力度量，显著提高了搜索质量。然而，如第 2.2 节所指出，在所有这些方法中，注意力学习通常被表征学习主导，造成了学习到的与实际行为相关性的显著差距。注意力。注意力机制在变换器（Transformers）中最为知名 (Vaswani et al., 2017)，已被证明极为有效，并广泛用于相关性测量。变换器使用 Q、K（注意力投影）和 V（表征投影）矩阵为每个项生成查询、键和值。查询和键的缩放点积作为相关性评分，而值则作为表征。这种结构在许多领域广泛应用，包括自然语言处理 (Brown et al., 2020) 和计算机视觉 (Dosovitskiy et al., 2021)。然而，在推荐系统中，鉴于 Guo et al. (2024) 指出的交互崩溃理论，小嵌入维度会使线性投影完全失效，如第 2.3 节所讨论。因此，在这个特定领域需要适当的调整。

# 6 结论

本文聚焦于长序列推荐，首先分析了嵌入的梯度主导性和冲突。随后，我们提出了一种新颖的解耦注意力与表示嵌入（DARE）模型，该模型利用独立的嵌入表完全解耦了注意力和表示。离线和在线实验均展示了DARE的潜力，综合分析突显了其在注意力精度、表示可分辨性以及更快推理速度方面的优势。

# 可重复性声明

为了确保可复现性，我们在附录A中提供了超参数和基线实现细节，附录B中包含数据集信息。我们已经在 https : / /github. com/ thuml/DARE 发布了完整的代码，包括数据集处理、模型训练和分析实验。

# 致谢

本研究得到了中国国家自然科学基金（62021002）、BNRist项目、腾讯创新基金以及国家大数据软件工程研究中心的支持。