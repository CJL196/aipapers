# 一分钟视频生成与测试时训练

Karan Dalal\*4 Daniel Koceja\*2 Gashon Hussein\*2 Jiarui ${ \mathrm { X u } } ^ { * 1 , 3 }$ 岳赵†5 宥金宋‡2 施浩 韩1 查俊昌1 Jan Kautz1 Carlos Guestrin2 辰典桥本2 Sanmi Koyejo2 Yejin Choi1 Yu Sun1,2 Xiaolong Wang1,3 1NVIDIA 2斯坦福大学 3加州大学圣地亚哥分校 4加州大学伯克利分校 5德克萨斯大学奥斯汀分校

![](images/1.jpg)

![](images/2.jpg)  
Jerrhapp Toic poloy. Je oriv Tomaet e hee andhe tw eo oeth, henh .

j /dʒei/ n. 字母j Definition: n. the 10th letter of the Roman alphabet

# 摘要

如今，变换器在生成一分钟视频时仍面临挑战，因为自注意力层对于长距离上下文效率较低。Mamba 层等替代方案在处理复杂的多场景故事时也遇到困难，因为它们的隐藏状态表现力较弱。我们实验了测试时训练（TTT）层，其隐藏状态本身可以是神经网络，因此更具表现力。将 TTT 层添加到预训练的变换器中，使其能够从文本故事板生成一分钟的视频。作为概念验证，我们组建了一个基于《猫和老鼠》卡通的数据集。与 Mamba 2、Gated DeltaNet 和滑动窗口注意力层等基准进行比较，TTT 层生成的视频更连贯，能够讲述复杂的故事，在每种方法的 100 个视频的人类评估中领先 34 个 Elo 点。尽管前景看好，但结果仍然包含伪影，这可能是由于预训练的 5B 模型的能力有限。我们实现的效率也有待提高。由于资源限制，我们仅实验了一分钟的视频，但该方法可以扩展到更长的视频和更复杂的故事。示例视频、代码和注释可在以下网址获取： https://test-time-training.github.io/video-dit

# 1. 引言

尽管在视觉和物理真实感方面取得了显著进展，最先进的视频Transformer仍主要生成单一场景的短片，而缺乏复杂的故事情节。截至2025年3月，公开视频生成API的最长时长为Sora（OpenAI）20秒，MovieGen（Meta）16秒，Ray 2（Luma）10秒，以及Veo 2（Google）8秒。这些API均无法自主生成复杂的多场景故事。

![](images/3.jpg)  
the hidden state itself a model $f$ with weights $W$ $\ell$ Therefore, updating the hidden state on a test sequence is equivalent to training the model $f$ at test time. This process, known as Test-Time Training (TTT), is programmed into TTT layers. Figure and caption taken from [43].

这些技术限制背后的一个基本挑战是长上下文，因为 Transformer 中自注意力层的计算成本随着上下文长度的增加而呈平方级增长。对于动态运动的视频生成，这一挑战尤其明显，因为其上下文无法被分词器轻易压缩。使用标准分词器，每个我们的一分钟视频需要超过 $3 0 0 \mathrm { k }$ 的上下文词元。使用自注意力生成一段一分钟的视频所需的时间是生成 20 段 3 秒视频的 $1 1 \times$，而训练所需的时间是 $1 2 \times$。为了解决这一挑战，近期关于视频生成的研究探讨了 RNN 层作为自注意力的高效替代方案，因为它们的计算成本随上下文长度线性增加 [47]。现代 RNN 层，特别是线性注意力的变体 [23, 37]，如 Mamba [8, 12] 和 DeltaNet [35, 53]，在自然语言任务中展示了令人印象深刻的结果。然而，我们尚未看到由 RNN 生成的复杂故事或动态运动的长视频。[47] 中的视频（链接）是高分辨率且长达一分钟，但仅包含单一场景和慢动作，更不用说复杂故事了。我们认为这些 RNN 层生成的视频不够复杂，因为它们的隐状态表达能力较弱。RNN 层只能将过去的词元存储到固定大小的隐状态中，而对于线性注意力变体如 Mamba 和 DeltaNet，这仅是一个矩阵。将数十万向量压缩到只有数千秩的矩阵中本质上是具有挑战性的。因此，这些 RNN 层难以记住远距离词元之间的深层关系。我们实验了一种替代类别的 RNN 层，其隐状态本身可以是神经网络。具体来说，我们使用两层 MLP，其隐单元数是线性（矩阵）隐状态的 $2 \times$，并包含更丰富的非线性特征。由于神经网络隐状态甚至在测试序列上也会通过训练进行更新，这些新层被称为测试时训练（TTT）层 [43]。我们从一个预训练的扩散 Transformer（CogVideo-X 5B [19]）开始，该模型只能以 16 fps 生成 3 秒的短片（或以 8 fps 生成 6 秒的短片）。然后，我们添加从零开始初始化的 TTT 层，并对该模型进行微调，以从文本故事板生成一分钟的视频。我们将自注意力层限制在 3 秒的片段，以便其成本保持在可管理范围内。在仅进行了初步系统优化的情况下，我们的训练运行相当于 $2 5 6 \mathrm { H } 1 0 0 \mathrm { s }$ 上的 50 小时。我们基于约 7 小时的《汤姆和杰瑞》卡通以及人工标注的故事板，策划了一个文本到视频的数据集。我们故意将研究范围限制在这一特定领域，以便快速迭代研究。作为概念验证，我们的数据集强调复杂的多场景和长程故事以及动态运动，其中仍需进展；而在视觉和物理真实感方面，我们强调较少，因为在这方面已经取得了显著进展。我们相信，对于这一特定领域的长上下文能力的改进会转移到通用的视频生成中。与 Mamba 2 [8]、Gated DeltaNet [53] 和滑动窗口注意力层等强基准相比，TTT 层生成更连贯的视频，可以讲述复杂的故事并具有动态运动，在每种方法的 100 个视频的人工评估中领先 34 Elo 点。作为参考，GPT-4o 在 LMSys Chatbot Arena [6] 中比 GPT-4 Turbo 高出 29 Elo 点。样本视频、代码和注释可在以下链接获取：https://test-time-training.github.io/video-dit

# 2. 测试时训练层

根据标准实践 [44, 54]，每个视频被预处理成一个 $T$ 词元的序列，其中 $T$ 由其持续时间和分辨率决定。本节回顾了用于一般序列建模的测试时训练（TTT）层，使用了文献 [43] 第2节的一些阐述。我们首先讨论如何以因果方式（按时间顺序）处理一般输入序列。第3节讨论如何通过反向调用递归神经网络（RNN）层在非因果的主干中使用它们。

# 2.1. TTT作为更新隐状态

所有 RNN 层将历史上下文压缩为固定大小的隐藏状态。这种压缩有两个后果。一方面，将输入词元 $x _ { t }$ 映射到输出词元 $z _ { t }$ 是高效的，因为更新规则和输出规则每个词元的计算时间都是常量。另一方面，RNN 层记忆长上下文的能力受到其隐藏状态所能存储的信息量的限制。[43] 的目标是设计具有表达能力的隐藏状态的 RNN 层，以压缩大量上下文。作为灵感，他们观察到自监督学习可以将大量训练集压缩到机器学习模型的权重中。[43] 的关键思想是利用自监督学习将历史上下文 $x _ { 1 } , \ldots , x _ { t }$ 压缩为隐藏状态 $W _ { t }$，通过将上下文视为一个未标记的数据集，而隐藏状态为机器学习模型 $f$ 的权重。更新规则如图 2 所示，是关于某些自监督损失 $\ell$ 的梯度下降步骤：

$$
W _ { t } = W _ { t - 1 } - \eta \nabla \ell ( W _ { t - 1 } ; x _ { t } ) ,
$$

学习率 $\eta$。直观上，输出的词元就是根据 $f$ 使用更新后的权重 $W _ { t }$ 对 $x _ { t }$ 的预测：

$$
z _ { t } = f ( x _ { t } ; W _ { t } ) .
$$

一个选择$\ell$是重构$x_{t}$本身。为了使学习问题变得非平凡，可以先将$x_{t}$处理成一个损坏的输入$\tilde{x}_{t}$（见第2.2小节），然后进行优化：

$$
\ell ( W ; x _ { t } ) = \| f ( \tilde { x } _ { t } ; W ) - x _ { t } \| ^ { 2 } .
$$

类似于去噪自编码器 [46]，$f$ 需要发现 $x _ { t }$ 各维度之间的相关性，以便从部分信息 $\tilde { x } _ { t }$ 中重构出它。与其他 RNN 层和自注意力机制一样，该算法将输入序列 $x _ { 1 } , \ldots , x _ { T }$ 映射到输出序列 $z _ { 1 } , \dots , z _ { T }$，可以被编程到序列建模层的前向传播中。即便在测试时，该层仍然为每个输入序列训练一组不同的权重序列 $W _ { 1 } , \dots , W _ { T }$。因此，它被称为测试时训练（TTT）层。从概念上讲，对 $\nabla \ell$ 调用反向传播意味着对梯度进行梯度计算，这是一种在元学习中广泛探讨的技术。TTT 层与 RNN 层和自注意力层具有相同的接口，因此可以在任何更大网络架构中替换。[43] 将训练更大网络称为外循环，而在每个 TTT 层中训练 $W$ 称为内循环。

# 2.2. 为TTT学习自监督任务

可以说，TTT中最重要的部分是由$\ell$指定的自监督任务。与其根据人类先验手工制作自监督任务，[43]采取了更端到端的方法，将其作为外部循环的一部分进行学习。从方程3中的 naive reconstruction 任务开始，他们使用一个低秩投影 $\tilde { x } _ { t } = \theta _ { K } x _ { t }$，其中$\theta _ { K }$是一个可以在外部循环中学习的矩阵。此外，也许并非所有的信息在$x _ { t }$中都是值得记忆的，因此重建标签也可以是低秩投影 $\theta _ { V } x _ { t }$ 而不是$x _ { t }$。总之，[43]中的自监督损失为：

$$
\ell ( W ; x _ { t } ) = \| f ( \theta _ { K } x _ { t } ; W ) - \theta _ { V } x _ { t } \| ^ { 2 } .
$$

最后，由于 $\theta_{K} x_{t}$ 的维度少于 $x_{t}$，因此 [43] 无法再使用方程 2 中的输出规则。所以他们进行了另一个投影 $\theta_{Q} x_{t}$，并将输出规则更改为：

$$
z _ { t } = f \left( \theta _ { Q } x _ { t } ; W _ { t } \right) .
$$

请注意，在内循环中，只有 $W$ 被优化，因此将其写为 $\ell$ 的一个参数；$\theta \mathrm { s }$ 是该内循环损失函数的“超参数”。$\theta _ { K } , \theta _ { V } , \theta _ { Q }$ 在外循环中被优化，类似于自注意力机制中的查询、键和值参数。

# 2.3. TTT-MLP 实例化

根据文献[43]，我们将内循环模型 $f$ 实例化为 $f _ { \mathsf { M L P } }$ 的包装器：一个与 Transformers 中类似的双层 MLP。具体来说，隐藏层维度是输入维度的 $4 \times$，后跟一个 GELU 激活函数 [16]。为了在 TTT 过程中获得更好的稳定性，$f$ 始终包含层归一化和残差连接。也就是说，

$$
f ( x ) = x + \mathsf { L N } ( f _ { \mathsf { M L P } } ( x ) ) .
$$

具有该 $f$ 的 TTT 层称为 TTT-MLP，这是本文中默认的实例。在第 4 节中，我们还实例化了 TTT-Linear（上述 $f$ 包装在一个线性模型中）作为基线。

# 3. 方法

在高层次上，我们的方法只是向预训练的扩散变换器添加 TTT 层，并在带有文本注释的长视频上进行微调。在实际操作中，使这种方法有效涉及许多设计选择。

# 3.1. 架构

预训练扩散变换器。我们添加 TTT 层然后进行微调的方法原理上可以与任何主干架构配合使用。我们选择扩散变换器作为初步示范，因为它是视频生成中最流行的架构。由于在视频上预训练扩散变换器的成本过高，我们从一个名为 CogVideo-X 5B 的预训练检查点开始。

![](images/4.jpg)  
selattention layers locally over segments and TTT layers gobally over the entire sequence.See Subsection 3..

门控。给定一个输入序列 $X = ( x _ { 1 } , \dots , x _ { T } )$，其中每个词元 $x _ { t } \in \mathbb { R } ^ { d }$，TTT 层生成一个输出序列 $Z = ( z _ { 1 } , . . . , z _ { T } ) = \mathsf { T T T } ( X )$。每个 $z _ { t } \in \mathbb { R } ^ { d }$ 遵循第二节中方程 1、4 和 5 描述的递推关系。将 TTT 层天真地插入到预训练网络中，会在微调开始时显著恶化其预测，因为此时 TTT 层是随机初始化的。为避免这种退化，我们按照标准做法用一个学习得到的向量 $\boldsymbol { \alpha } \in \mathbb { R } ^ { d }$ 对 TTT 进行门控：[1]

$$
\mathtt { g a t e } ( \mathsf { T T T } , X ; \alpha ) = \operatorname { t a n h } ( \alpha ) \otimes \mathsf { T T T } ( X ) + X ,
$$

其中 $\operatorname{tanh}(\alpha) \in (-1, 1)^{d}$ 与 $Z = \mathsf{TTT}(X)$ 中的每个 $z_{t}$ 元素逐项相乘。我们将 $\alpha$ 中的所有值初始化为 0.1，因此在微调开始时 $\operatorname{tanh}(\alpha)$ 中的值接近 0（约 0.1）。这种 $\alpha$ 的初始化允许 TTT 仍然对 $\mathtt{gate(TTT,} X; \alpha)$ 产生贡献，而不会显著覆盖 $X$。双向。扩散模型，包括 CogVideo-X，是非因果的，这意味着输出 token $z_{t}$ 可以基于所有的 $x_{1}, \ldots, x_{T}$ 而不仅仅是过去的 token $x_{1}, \ldots, x_{t}$ 进行条件化。为了以非因果的方式使用 TTT 层，我们应用一个叫做双向的标准技巧 [30]。给定一个算子 $\mathsf{rev}(X) = (x_{T}, \ldots, x_{1})$，它在时间上反转 $\boldsymbol{X} = (x_{1}, \ldots, x_{T})$，我们定义

$$
{ \mathsf { T T T } } ^ { \prime } ( X ) = { \mathsf { r e v } } ( { \mathsf { T T T } } ( { \mathsf { r e v } } ( X ) ) ) .
$$

由于 rev 被应用了两次，$\mathsf { T T T } ^ { \prime } ( X )$ 仍然保持时间顺序。但其中的 TTT 层现在以逆时间顺序扫描 $X$。修改后的架构。标准 Transformer，包括 CogVideo-X，包含交错的序列建模块和 MLP 模块。具体而言，标准序列建模块接受输入序列 $X$，并生成，其中 LN 是层 $\mathrm { { N o r m } ^ { 1 }$，$X ^ { \prime } + X$ 形成一个残差连接。我们仅修改序列建模块，其余架构保持不变。每个修改的块，在图 3 的左侧面板中说明，继续使用方程 8 中的 $X ^ { \prime }$ 并生成

$$
\begin{array} { c } { { X ^ { \prime } = \mathsf { s e l f } _ { - } \mathsf { a t t n } ( \mathsf { L N } ( X ) ) } } \\ { { Y = X ^ { \prime } + X , } } \end{array}
$$

$$
\begin{array} { r c l } { { } } & { { } } & { { Z = \tt g a t e ( T I T , } X ^ { \prime } ; \alpha ) , }  \\ { { } } & { { } } & { { Z ^ { \prime } = \tt g a t e ( T T T ^ { \prime } , } Z ; \beta ) , }  \\ { { } } & { { } } & { { Y = Z ^ { \prime } + X . } } \end{array}
$$

注意到 $\mathsf { T } \mathsf { T } \mathsf { T } ^ { \prime }$ 只是再次调用 TTT，因此它们共享相同的基本参数 $\theta _ { K } , \theta _ { V } , \theta _ { Q }$。但对于门控，方程 10 和 11 使用了不同的参数 $\alpha$ 和 $\beta$。

# 3.2. 总体流程

在本小节中，我们讨论如何为我们的架构创建输入的词元序列，以及每个序列如何分段处理。除了接下来讨论的前两种文本格式外，所有内容均适用于微调和推理。我们的流程在图3的右侧面板中进行了说明。场景与段落。我们将视频结构化为多个场景，而每个场景包含一个或多个3秒的段落。我们将3秒的段落用作文本到视频配对的基本单元，原因有三：● 原始预训练的CogVideo-X的最大生成长度为3秒。● 《猫和老鼠》每集中的大多数场景至少为3秒。● 给定3秒段，构建具有多个阶段的数据集（见3.3小节）最为便利。文本提示格式。在推理时，用户可以使用以下三种格式中的任何一种为长视频编写文本提示，按细节递增的顺序排列。每种格式的示例见附录中的图8。● 格式1：情节的简短总结，5-8句话。一些示例在图1中展示。● 格式2：更详细的情节，大约20句话，每句话大致对应一个3秒段。句子可以被标记为属于特定场景或场景组，但这些标签仅被视为建议。● 格式3：故事板。每个3秒段由3-5句话的段落描述，包含背景颜色和镜头运动等细节。一个或多个段落的组合严格归属于特定场景，并使用关键词<scene start>和<scene end>进行标识。在微调和推理过程中，我们文本分词器的实际输入始终采用格式3。格式之间的转换由Claude 3.7 Sonnet按照$1 2 3$的顺序进行。对于微调，我们的人类标注已经是格式3，如3.3小节所述。从文本到序列。在原始CogVideo-X对每个视频的输入文本进行分词后，它将文本词元与噪声视频词元连接起来，形成Transformer的输入序列。为了生成一个长视频，我们对每个3秒段独立应用相同的程序。具体而言，给定一个包含$n$段落的格式3故事板，我们首先生成$n$个序列段落，每个段落包含从相应段落提取的文本词元，后跟视频词元。然后，我们将所有$n$个序列段落连接在一起，形成输入序列，现已交错了文本和视频词元。本地注意力，全球TTT。CogVideo-X使用自注意力层为每个最大长度为3秒的视频全局处理整个输入序列，但对于长视频而言，全球注意力变得低效。为了避免增加自注意力层的上下文长度，我们使其局限于每个3秒段，独立地关注每个$n$个序列段落。TTT层则对整个输入序列进行全局处理，因为它们在长上下文中是高效的。

# 3.3. 微调方案与数据集

多阶段上下文扩展。遵循大型语言模型的标准做法，我们将修改后的架构的上下文长度扩展到一分钟，共分五个阶段。首先，我们在《汤姆与杰瑞》的3秒片段上微调整个预训练模型，以使其适应这一领域。在这一阶段，新参数（具体来说是TTT层和门控的参数）被赋予更高的学习率。在接下来的四个阶段中，我们分别在9秒、18秒、30秒和最终的63秒视频上进行微调。为避免过多遗忘预训练中的世界知识，我们仅微调TTT层、门控和自注意力层，并在这四个阶段中使用较低的学习率。详细的步骤见附录A。原始视频的超分辨率。我们使用1940年至1948年间发布的81集《汤姆与杰瑞》作为起始数据。每集约5分钟，所有集数合计约7小时。原始视频的分辨率各不相同，这在现代标准下普遍较差。我们在原始视频上运行一个视频超分辨率模型，生成在我们的数据集中共享分辨率为 $720 \times 480$ 的视觉增强视频。多阶段数据集。遵循3.2节中讨论的结构，我们首先让人工标注者将每一集分割成多个场景，然后从每个场景提取3秒的片段。接下来，由人工标注者为每个3秒片段撰写详细的段落。第一阶段直接在这些片段上进行微调。为了创建最后四个阶段的数据，我们将连续的3秒片段串联成分别为9秒、18秒、30秒和63秒的视频，同时附上其文本注释。场景边界由3.2节中的相同关键词标记。因此，所有训练视频的注释均采用格式3。

# 3.4. 非因果序列的并行化

第2节中讨论的更新规则无法在序列中的词元上进行简单的并行化，因为计算 $W _ { t }$ 需要 $\nabla \ell ( W _ { t - 1 } ; x _ { t } )$ ，而这又需要 $W _ { t - 1 }$ 。为了实现并行化，我们在每次更新中同时处理 $b$ 个词元，这在文献[43]中称为内循环小批量。本文中，我们设置 $b = 64$ 。具体来说，对于小批量 $i = 1 , \dots , T / b$ （假设 $T$ 是 $b$ 的整数倍），

$$
{ \cal W } _ { i b } = { \cal W } _ { ( i - 1 ) b } - \frac { \eta } { b } \sum _ { t = ( i - 1 ) b + 1 } ^ { i b } \nabla \ell \bigl ( W _ { ( i - 1 ) b } ; x _ { t } \bigr ) .
$$

由于该序列是非因果的，我们使用 $W _ { i b }$ 生成小批量 $i$ 中所有时间步的输出词元：

$$
z _ { t } = f ( W _ { i b } ; x _ { t } ) , \qquad \mathrm { f o r } \ t = ( i - 1 ) b + 1 , \dots , i b .
$$

注意到 $W _ { ( i - 1 ) b + 1 } , \dots , W _ { i b - 1 }$ 不再需要。经过此修改后，$f$ 可以并行处理一个（内循环）小批量的词元，类似于常规的多层感知器处理一个（外循环）训练数据的小批量。作为附带好处，我们观察到在词元之间平均梯度可以减少方差并稳定每次对 $W$ 的更新。

![](images/5.jpg)  
the hidden state $\mathbf { \hat { W } } ^ { ( 1 ) }$ and $W ^ { ( 2 ) }$ across SMs, transferring them between HBM and SMEM only during initial loading and final output. intermediate activations among SMs.

# 3.5. 芯片内张量并行

# 4. 评估

在GPU上高效实现TTT-MLP需要特殊设计，以利用其内存层次结构。GPU上的一个芯片称为流处理器（SM），类似于CPU上的内核。GPU上的所有SM共享一个相对较慢但容量大的全局内存，称为HBM，每个SM则具有一个快速但较小的片上内存，称为SMEM。在GPU上，SMEM与HBM之间频繁的数据传输会显著影响整体效率。Mamba和自注意力层（Flash Attention [9]）的高效实现通过内核融合来最小化此类传输。这些实现的高层次思路是将输入和初始状态加载到每个SMEM中，完全在片上进行计算，并仅将最终输出写回HBM。然而，TTT-MLP的隐藏状态，即双层MLP $f$ 的权重 $W ^ { ( 1 ) }$ 和 $W ^ { ( 2 ) }$，对于单个SM的SMEM来说过于庞大（与输入和激活一起考虑）。为了减少每个SM所需的内存，我们使用张量并行性 [39] 来将 $\mathbf { \bar { \boldsymbol { W } } } ^ { ( 1 ) }$ 和 $W ^ { ( 2 ) }$ 在SM之间分片，如图4所示。与大型MLP层如何在多个GPU的HBM之间分片和训练类似，我们现在将相同的思路应用于多个SM的SMEM之间，视每个SM为GPU的类比。我们利用NVIDIA Hopper GPU架构上的DSMEM特性，在SM之间实现AllReduce。有关我们内核的更多细节，请参见附录B。我们的实现显著提高了效率，因为隐藏状态和激活现在仅在初始加载和最终输出时从HBM读取和写入。作为一般原则，如果模型架构 $f$ 可以在GPU之间通过标准张量并行性分片，则当 $f$ 被用作隐藏状态时，相同的分片策略可以应用于SM之间。我们对TTT-MLP和五个基线（包括局部注意力、TTT-Linear、Mamba 2、门控DeltaNet和滑动窗口注意力层）进行了多维基准的人类评估，这些基线均具有线性复杂度。

# 4.1. 基线

除了局部注意力，所有基线都是通过第3.1小节中的方法添加到相同的预训练CogVideo-X 5B模型中；它们的修改后的架构均具有7.2B参数。所有基线都使用相同的微调方案，详见第3.3小节和附录A。接下来，我们将详细讨论这些基线。

• 局部注意力：对原始架构没有修改，独立对每个 3 秒的片段执行自注意力。 • TTT-Linear [43]：一个 TTT 层，实例化为 $f ( x ) = x + \mathsf { L N } ( f _ { \mathsf { L i n e a r } } ( x ) )$，其中 $f _ { \mathsf { L i n e a r } }$ 是一个线性模型。 • Mamba 2 [8]：一个现代的 RNN 层，具有一个矩阵隐藏状态，其规模大约是 TTT-Linear 中隐藏状态的 4 倍，但大约是 TTT-MLP 中隐藏状态的 2 倍。 • 门控 DeltaNet [53]：DeltaNet [52] 和 Mamba 2 的扩展，带有改进的更新规则。 • 滑动窗口注意力 [3]：具有固定 8192 词元窗口的自注意力（大约 1.5 秒的视频）。

# 4.2. 评估轴与协议

在MovieGen [44]的六个评估维度中，我们选择与我们领域相关的四个维度进行人工评估。

![](images/6.jpg)

抱歉，我无法理解您的请求。请提供更多上下文或明确的问题。

![](images/7.jpg)

GatD

![](images/8.jpg)

抱歉，我无法处理您的请求。

<table><tr><td></td><td>Text following</td><td>Motion naturalness</td><td>Aesthetics</td><td>Temporal consistency</td><td>Average</td></tr><tr><td>Mamba 2</td><td>985</td><td>976</td><td>963</td><td>988</td><td>978</td></tr><tr><td>Gated DeltaNet</td><td>983</td><td>984</td><td>993</td><td>1004</td><td>991</td></tr><tr><td>Sliding window</td><td>1016</td><td>1000</td><td>1006</td><td>975</td><td>999</td></tr><tr><td>TTT-MLP</td><td>1014</td><td>1039</td><td>1037</td><td>1042</td><td>1033</td></tr></table>

改进幅度最大的是场景一致性 $( + 3 8 )$ 和运动平滑性 $( + 3 9 )$。作为对比，GPT-4 在 Chatbot Arena 中比 GPT-3.5 Turbo 高出 46 个 Elo 分，而 GPT-4o 比 GPT-4 Turbo 高出 29 个 Elo 分 [6]。

![](images/9.jpg)  
Figure 6. For 63-second videos, inference with full attention (over 300k tokens) would have taken $1 1 \times$ longer than local attention, and training $1 2 \times$ longer, as discussed in Section 1. TTT-MLP takes $2 . 5 \times$ and $3 . 8 \times$ respectively  significantly more efficient than full attention, but still less efficient than, for example, Gated DeltaNet, which takes $1 . 8 \times$ longer than local attention in both inference and training.

与提供的提示对齐。 • 动作自然性：自然的肢体动作、面部表情以及遵循物理法则。看起来不自然或怪异的动作将受到惩罚。美学：有趣且引人入胜的内容、光照、色彩和摄像效果。 • 时间一致性：在场景内和跨场景的一致性。引用的描述来自 MovieGen [44]。我们的评估基于盲比较中的成对偏好，因为直接对长视频进行评分或同时对多个视频进行排序是具有挑战性的。具体而言，评估员会随机选择上述四个维度中的一个和一对共享相同情节的随机视频，然后要求指出在该维度上更佳的视频。为了收集视频池，我们首先使用 Claude 3.7 Sonnet 随机抽取 100 个情节（以子章节 3.2 中讨论的格式 $1 \ 2 \ 3$），然后为每个情节生成每种方法对应的一段视频。生成视频的方法始终对评估员未知。我们的评估员是在 prolific.com 上招募的，筛选条件为：居住在美国，英语为母语，年龄在 18 至 35 岁之间，至少有 100 次提交记录以及至少 $98 \%$ 的通过率。评估员的背景信息在网站上披露如下：性别： $50.78\%$ 男性， $47.66\%$ 女性， $1.56\%$ 其他。种族： $57.03\%$ 白人， $23.44\%$ 黑人， $10.94\%$ 混血， $5.47\%$ 亚裔， $3.12\%$ 其他。基于这些信息，我们认为我们的评估员代表了美国人口的一个样本。

# 4.3. 结果

我们在 LMSys Chatbot Arena 中使用 Elo 系统聚合成对偏好 [6]。Elo 分数如表 1 所示。

TTT-MLP 在平均上超过第二名方法 34 Elo 点。作为背景，GPT-4 相较于 GPT-3.5 Turbo 高出 46 Elo 点（1163 对比 1117），而 GPT-40 则在 GPT-4 Turbo 上高出 29 点（1285 对比 1256），这一切发生在 LMSys Chatbot Arena，因此我们提高 34 点在实际应用中是有意义的。图 5 比较了 TTT-MLP 和基线生成的样本视频帧。图 5 中展示的视频可在项目网站上访问：https://test-time-training.github.io/video-dit 18 秒的淘汰赛。请注意，局部注意力和 TTT-Linear 并未出现在表 1 中。为了避免在每个方法上评估更长视频的更高成本，我们首先进行了 18 秒视频的淘汰赛，遵循了 4.2 小节中讨论的相同程序。本轮淘汰了表现最差的局部注意力以及表现不如 TTT-MLP 的 TTT-Linear。淘汰赛的结果见附录中的表 3。

![](images/10.jpg)  

时间一致性：这些框在同一场景的3秒片段之间变形。

![](images/11.jpg)

运动自然性：奶酪悬浮在空中，而不是自然地落到地面。

![](images/12.jpg)

美学：当汤姆转身时，厨房的灯光变得异常明亮。

# 4.4. 局限性

短暂的背景。在上述的18秒淘汰赛中，Gated DeltaNet的平均表现最佳，比Mamba 2高出27个Elo点，比TTT-MLP高出28个Elo点（见附录中的表3）。对于18秒的视频，上下文长度大约为$1 0 0 \mathrm { k }$ 词元。这一评估显示，采用线性（矩阵）隐状态的RNN层（如Gated DeltaNet和Mamba 2）仍然是最有效的。此外，针对18秒和63秒视频的评估结果表明，Gated DeltaNet相比Mamba 2有显著提升。视频伪影。生成的63秒视频展示了作为概念验证的明确潜力，但仍存在显著的伪影，特别是在运动自然性和美学方面。图7展示了与我们三个评估轴相对应的伪影示例。我们观察到，这些伪影并非TTT-MLP特有，而是在所有方法中普遍存在。伪影的产生可能是源于预训练CogVideo-X 5B模型的有限能力。例如，原始CogVideo-X生成的视频（链接）似乎也在运动自然性和美学方面受到限制。壁-clock时间。即使在应用了我们在3.4节和3.5节中的改进后，TTT-MLP的效率仍然低于Gated DeltaNet和Mamba 2。这一局限性在图6中得到了强调，其中TTT-MLP的推理和训练速度分别比Gated DeltaNet慢$1 . 4 \times$和$2 . 1 \times$。第6节讨论了我们TTT-MLP内核的两个潜在改进，以提高效率。请注意，在我们的应用中，训练效率并不是一个重要问题，因为RNN层是在预训练后集成的，这构成了整体训练预算的大部分。RNN层的训练效率仅在微调阶段相关，而微调本身只是预算的一小部分。相比之下，推理效率则更具意义。

# 5. 相关工作

现代RNN层，尤其是线性注意力变体，如Mamba和DeltaNet，在自然语言任务中展示了出色的性能。受到它们成功的启发，以及Fast Weight Programmers的理念，提出了可扩展且实用的方法，使隐藏状态更大且非线性，从而更具表现力。最近的研究更进一步，开发了更大更非线性的隐藏状态，并使用更复杂的优化技术进行更新。相关工作部分详细讨论了TTT层的灵感来源。另有工作对RNN层的最新进展进行了良好的综述。 长视频建模。一些早期工作通过训练GAN来生成长视频，预测基于当前帧和运动向量的下一帧。由于自回归和基于扩散的方法的最新进展，生成质量显著提高。TATS提出在Transformer上使用滑动窗口注意力生成超过训练长度的视频。Phenaki以类似的自回归方式工作，但每帧是通过MaskGIT生成的。预训练的扩散模型可以通过级联、流式处理和添加过渡来扩展，以生成更长的视频。故事合成方法生成与文本故事中各个句子相对应的一系列图像或视频。例如，Craft通过检索生成复杂场景的视频，而StoryDiffusion利用扩散技术提高帧间过渡的平滑度。尽管与文本到视频生成有关，但故事合成方法通常需要额外的组件来维持场景之间的一致性，这些组件并非端到端处理。

# 6. 未来工作

我们概述了未来工作的几个有前景的方向。更快的实现。目前我们的TTT-MLP核受到寄存器溢出和异步指令的非最佳排序的瓶颈。通过最小化寄存器压力和开发更具编译器感知的异步操作实现，效率可能会进一步提高。更好的集成。使用双向和学习门仅是将TTT层集成到预训练模型中的一种可能策略。更好的策略应进一步提高生成质量并加速微调。其他视频生成主干网络，如自回归模型，可能需要不同的集成策略。生成较长视频与更大隐状态。我们的方法有可能扩展到生成具有线性复杂度的更长视频。实现这一目标的关键在于，我们相信，是将隐状态实例化为比我们的两层MLP更大的神经网络。例如，$f$本身可以是一个Transformer。致谢。我们感谢Hyperbolic Labs的计算支持，感谢Yuntian Deng在实验运行方面的帮助，以及Aaryan Singhal、Arjun Vikram和Ben Spector在系统问题上的帮助。Yue Zhao要感谢Philipp Krähenbühl的讨论和反馈。Yu Sun要感谢他的博士导师Alyosha Efros关于在机器学习工作中关注像素的深刻建议。关于作者身份的说明。Gashon Hussein和Youjin Song在该项目的初始版本提交给CVPR后加入团队，并对最终版本做出了重大贡献。由于CVPR不允许在提交后添加作者，他们的名字未能出现在OpenReview和会议网页上。然而，我们一致认为正式的作者列表应包括他们的名字，如我们发布的PDF所示。没有他们的工作，这个项目是不可能实现的。

# References

[1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. NeurIPS, 2022. 4   
[2] Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to memorize at test time. arXiv preprint arXiv:2501.00663, 2024. 9   
[3] Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020. 6   
[4] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In CVPR, 2022. 10   
[5] Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Seine: Short-to-long video diffusion model for generative transition and prediction. In ICLR, 2023. 10   
[6] Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Banghua Zhu, Hao Zhang, Michael Jordan, Joseph E Gonzalez, et al. Chatbot arena: An open platform for evaluating llms by human preference. In ICML, 2024. 2, 8   
[7] Kevin Clark, Kelvin Guu, Ming-Wei Chang, Panupong Pasupat, Geoffrey Hinton, and Mohammad Norouzi. Metalearning fast weight language models. EMNLP, 2022. 9   
[8] Tri Dao and Albert Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. In ICML, 2024. 2, 6, 9   
[9] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. In NeurIPS, 2022. 6   
10] Songwei Ge, Thomas Hayes, Harry Yang, Xi Yin, Guan Pang, David Jacobs, Jia-Bin Huang, and Devi Parikh. Long video generation with time-agnostic vqgan and timesensitive transformer. In ECCV, 2022. 10   
11] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 2020. 10   
12] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In COLM, 2024. 2, 9   
13] Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Fei-Fei Li, Irfan Essa, Lu Jiang, and José Lezama. r lltalsue vutu gellalo wiu ullusiol oui. II ECCV, 2024. 10   
[14] Tanmay Gupta, Dustin Schwenk, Ali Farhadi, Derek Hoiem, and Aniruddha Kembhavi. Imagine this! scripts to compositions to videos. In ECCV, 2018. 10   
[15] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation. arXiv preprint arXiv:2211.13221, 2022. 10   
[16] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016. 3   
[17] Roberto Henschel, Levon Khachatryan, Daniil Hayrapetyan, Hayk Poghosyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Streamingt2v: Consistent, dynamic, and extendable long video generation from text. arXiv preprint arXiv:2403.14773, 2024. 10   
[18] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 1   
[19] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. In ICLR, 2023. 2, 3   
[20] Ting-Hao Huang, Francis Ferraro, Nasrin Mostafazadeh, Ishan Misra, Aishwarya Agrawal, Jacob Devlin, Ross Girshick, Xiaodong He, Pushmeet Kohli, Dhruv Batra, et al. Visual storytelling. In NAACL, 2016. 10   
[21] Kazuki Irie, Imanol Schlag, Róbert Csordás, and Jürgen Schmidhuber. Going beyond linear transformers with recurrent fast weight programmers. NeurIPS, 2021. 9   
[22] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In CVPR, 2020. 10   
[23] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In ICML, 2020. 2, 9   
[24] Louis Kirsch and Jürgen Schmidhuber. Meta learning backpropagation and improving it. NeurIPS, 34:1412214134, 2021. 9   
[25] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jn Zhou, Jiag Xiog, XinL Bo Wu, Jiai a Kathrina Wu, Qin Lin, Junkun Yuan, Yanxin Long, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yi Chen, Yutao Cui, Yuan ng Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Dax Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, Jie Jiang, and Caesar Zhong. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv 2412.03603, 2025. 10   
[26] Yitong Li, Zhe Gan, Yelong Shen, Jingjing Liu, Yu Cheng, Yuexin Wu, Lawrence Carin, David Carlson, and Jianfeng Gao. Storygan: A sequential conditional gan for story visualization. In CVPR, 2019. 10   
[27] Shanchuan Lin, Bingchen Liu, Jiashi Li, and Xiao Yang. Common diffusion noise schedules and sample steps are flawed. In WACV, 2024. 1   
[28] Chang Liu, Haoning Wu, Yujie Zhong, Xiaoyun Zhang, Yanfeng Wang, and Weidi Xie. Intelligent grimm-open-ended visual storytelling via latent diffusion models. In CVPR, 2024. 10   
[29] Adyasha Maharana, Darryl Hannan, and Mohit Bansal. Storydall-e: Adapting pretrained text-to-image transformers for story continuation. In ECCV, 2022. 10   
[30] Shentong Mo and Yapeng Tian. Scaling diffusion mamba with bidirectional ssms for efficient image and video generation. arXiv preprint arXiv:2405.15881, 2024. 4   
[31] Xichen Pan, Pengda Qin, Yuhong Li, Hui Xue, and Wenhu Chen. Synthesizing coherent story with auto-regressive latent diffusion models. In WACV, 2024. 10   
[32] William Peebles and Saining Xie. Scalable diffusion models with transformers. In CVPR, 2023. 3, 4   
[33] Tanzila Rahman, Hsin-Ying Lee, Jian Ren, Sergey Tulyakov, Shweta Mahajan, and Leonid Sigal. Make-a-story: Visual memory conditioned consistent story generation. In CVPR, 2023. 10   
[34] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In ICLR, 2022. 1   
[35] Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. Linear transformers are secretly fast weight programmers. In ICML, 2021. 2, 9   
[36] Jürgen Schmidhuber. Learning to control fast-weight memories: An alternative to dynamic recurrent networks. Neural Computation, 4(1):131139, 1992. 9   
[37] Jürgen Schmidhuber. Learning to control fast-weight memories: An alternative to dynamic recurrent networks. Neural Computation, 4(1):131139, 1992. 2, 9   
[38] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. Flashattention-3: Fast and accurate attention with asynchrony and low-precision, 2024. 1   
[39] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatronlm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019. 6   
[40] Ivan Skorokhodov, Sergey Tulyakov, and Mohamed Elhoseiny. Stylegan-v: A continuous video generator with the price, image quality and perks of stylegan2. In CVPR, 2022. 10   
[41] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. 1   
[42] Benjamin F Spector, Simran Arora, Aaryan Singhal, Daniel Y Fu, and Christopher Ré. Thunderkittens: Simple, fast, and adorable ai kernels. In ICLR, 2025. 1   
[43] Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, and Carlos Guestrin. Learning to (learn at test time): Rnns with expressive hidden states. arXiv preprint arXiv:2407.04620, 2024. 2, 3, 5, 6, 9, 1   
[44] The Movie Gen team. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024. 2, 6, 8, 10   
[43] Kuen viegas, monammau Bavaerzauen, rieter-Jan Kndermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual description. In ICLR, 2023. 10   
[46] Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In ICML, 2008. 3   
[47] Hongjie Wang, Chih-Yao Ma, Yen-Cheng Liu, Ji Hou, Tao Xu, Jialiang Wang, Felix Juefei-Xu, Yaqiao Luo, Peizhao Zhang, Tingbo Hou, Peter Vajda, Niraj K. Jha, and Xiaoliang Dai. Lingen: Towards high-resolution minute-length text-to-video generation with linear computational complexity, 2024. 2   
[48] Ke Alexander Wang, Jiaxin Shi, and Emily B Fox. Testtime regression: a unifying framework for designing sequence models with associative memory. arXiv preprint arXiv:2501.12352, 2025. 9   
[49] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In ICCVW, 2021. 5   
[50] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. IJCV, 2024. 10   
[51] Wenhan Xiong, Jingyu Liu, Igor Molybog, Hejia Zhang, Prajjwal Bhargava, Rui Hou, Louis Martin, Rashi Rungta, Karthik Abinav Sankararaman, Barlas Oguz, et al. Effective long-context scaling of foundation models. In NAACL, 2024. 5   
[52] Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transformers with the delta rule over sequence length. In NeurIPS, 2024. 6, 9   
[53] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. In ICLR, 2025. 2, 6   
[54] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In ICLR, 2025. 2, 10, 1   
[55] Shengming Yin, Chenfei Wu, Huan Yang, Jianfeng Wang, Xiaodong Wang, Minheng Ni, Zhengyuan Yang, Linjie Li, Shuguang Liu, Fan Yang, et al. Nuwa-xl: Diffusion over diffusion for extremely long video generation. arXiv preprint arXiv:2303.12346, 2023. 10   
[56] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent selfattention for long-range image and video generation. In NeurIPS, 2024. 10

T J parameters are fine-tuned at reduced learning rates.   

<table><tr><td>Video len.</td><td>Ctx. len</td><td>Trainable parameters</td><td>Learning rate</td><td>Schedule</td><td>Steps</td></tr><tr><td>3 sec</td><td>18048</td><td>TTT / Pre-trained Params</td><td>1 × 10−4 / 1 × 10−5</td><td>Cosine / Constant</td><td>5000</td></tr><tr><td>9 sec</td><td>51456</td><td>TTT + Local Attn (QKVO)</td><td>1 × 10−5</td><td>Constant</td><td>5000</td></tr><tr><td>18 sec</td><td>99894</td><td>TTT + Local Attn (QKVO)</td><td>1 × 10-5</td><td>Constant</td><td>1000</td></tr><tr><td>30 sec</td><td>168320</td><td>TTT + Local Attn (QKVO)</td><td>1 × 10−5</td><td>Constant</td><td>500</td></tr><tr><td>63 sec</td><td>341550</td><td>TTT + Local Attn (QKVO)</td><td>1 × 10-5</td><td>Constant</td><td>250</td></tr></table>

Table 3. Human evaluation results for 18-second videos, discussed in Subsection 4.3 and 4.4.   

<table><tr><td></td><td>Text following</td><td>Motion naturalness</td><td>Aesthetics</td><td>Temporal consistency</td><td>Average</td></tr><tr><td>Local Attention</td><td>965</td><td>972</td><td>969</td><td>944</td><td>962</td></tr><tr><td>TTT-Linear</td><td>1003</td><td>995</td><td>1007</td><td>1001</td><td>1001</td></tr><tr><td>Mamba 2</td><td>1023</td><td>987</td><td>1008</td><td>1004</td><td>1005</td></tr><tr><td>Gated DeltaNet</td><td>1020</td><td>1039</td><td>1044</td><td>1026</td><td>1032</td></tr><tr><td>SWA</td><td>995</td><td>1004</td><td>993</td><td>980</td><td>993</td></tr><tr><td>TTT-MLP</td><td>994</td><td>1002</td><td>1002</td><td>1019</td><td>1004</td></tr></table>

# A. Experiment Details

Diffusion schedule. Following CogVideoX [54], we finetune our model using v-prediction [34], which includes a diffusion noise schedule with 1000 steps and ZeroSNR [27] enforced at the final step.

Training configurations. We use the following hyperparameters for all stages of training:

Optimizer: AdamW with $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 5 )$   
Learning Rate: Linear warmup over $2 \%$ of training steps   
Batch Size: 64   
• Gradient Clipping: 0.1   
Weight Decay: $1 0 ^ { - 4 }$ applied to all params except biases and normalization layers   
VAE Scale Factor: 1.0   
Dropout: Zero-out text prompt with probability 0.1   
•Precision: Mixed Precision with PyTorch FSDP2

TTT configurations. A key hyperparameter for TTT layers is the inner-loop learning rate $\eta$ ,which we set $\eta = 1 . 0$ for TTT-Linear and $\eta = 0 . 1$ for TTT-MLP.

Sampling schedule. We follow the DDIM sampler [41] with 50 steps, applying dynamic classifier-free guidance (CFG) [18] that increases CFG magnitude from 1 to 4 and utilizing negative prompts to further enhance video quality.

# B. On-Chip Tensor Parallel Details

We use ThunderKittens [42] to implement the TTT-MLP kernel, described in Subsection 3.5.

Hidden state sharding. We follow the standard strategy for Tensor Parallel, sharding the first layer column-wise and the second layer row-wise. As the GeLU non-linearity is elementwise, the forward pass of the TTT-layer requires a single reduction for computing the inner loss used to update the hidden state.

Further latency optimizations. We incorporate several techniques from FlashAttention-3 [38] to further reduce I/O latency on NVIDIA Hopper GPUs. In particular, we implement a multi-stage pipelining scheme that asynchronously prefetches future mini-batches from HBM, overlapping data transfers with computation on the current mini-batch. This approach, known as producer-consumer asynchrony, involves dedicating specialized warpgroups to either data loading (producer) or computation (consumer).

Gradient checkpointing. We integrate gradient checkpointing along the sequence dimension [43] directly into our fused kernel. To reduce I/O-induced stalls and CUDA thread workloads, we use the Tensor Memory Accelerator (TMA) to perform asynchronous memory stores.

# Format 1

T tn Tom is about to catch Jerry, Jerry makes it through the mouse hole and Tom slams into the wall.

# Format 2

Seent 1-: Tom walks into the kitchen carrying an apple pi. He sits at the table and begins eating.

ST watches Tom eating the pie, and eagerly rubs his tummy. He then darts off-screen to the right.

S  u J  o s     qu.

The story continues..

# Format 3

<arThenas   all his, nd wit-n-hie le T a c camera smoothly follows Tom from left to right, clearly showing each of his movements.

Thea H os coT   b

<arThenas   walls whbi n  wind-whiecku lettT vhshi  l ak behind the alt shaker.The cmer captures Jerry s he merges rom behind the salt shaker and stands o the countertop.

Thea v remains in position slightly to the side of Jerry, capturing his hungry expression.

Thea t  r

o c

The story continues...

tions of the segments, and (3) a detailed storyboard.