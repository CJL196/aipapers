# 通过联合学习对齐与翻译的神经机器翻译

兹米特里·巴赫达瑙 雅各布大学，德国 金炫燮 约书亚·本吉奥* 蒙特利尔大学

# 摘要

神经机器翻译是一种近期提出的机器翻译方法。与传统的统计机器翻译不同，神经机器翻译旨在构建一个单一的神经网络，通过联合调优以最大化翻译性能。近期提出的神经机器翻译模型通常属于编码器-解码器架构的一类，将源句子编码为固定长度的向量，从中解码器生成翻译。本文推测，使用固定长度向量是提升该基本编码器-解码器架构性能的瓶颈，并提出通过允许模型自动（软）搜索与预测目标词相关的源句子部分，而不必明确地将这些部分形成硬段落来进行扩展。采用这种新方法，我们在英语-法语翻译任务上的翻译性能达到了与现有最先进短语系统相当的水平。此外，定性分析表明，该模型发现的（软）对齐效果与我们的直觉非常一致。

# 1 引言

神经机器翻译是一种新兴的机器翻译方法，最近由Kalchbrenner和Blunsom（2013年）、Sutskever等人（2014年）以及Cho等人（2014b年）提出。与传统的基于短语的翻译系统（参见，例如，Koehn等人，2003年）不同，它由多个单独调节的小组件组成，神经机器翻译试图构建和训练一个大型单一神经网络，能够接收一个句子并输出正确的翻译。大多数提出的神经机器翻译模型属于编码器-解码器的范畴（Sutskever等人，2014年；Cho等人，2014a年），每种语言都有一个编码器和解码器，或者涉及一个针对每个句子应用的语言特定编码器，其输出被进行比较（Hermann和Blunsom，2014年）。编码器神经网络读取并将源句子编码为固定长度的向量。解码器随后从编码向量中输出翻译。整个编码器-解码器系统，由一对语言的编码器和解码器组成，共同训练以最大化给定源句子的正确翻译的概率。这个编码器-解码器方法可能遇到的一个问题是，神经网络需要能够将源句子的所有必要信息压缩到一个固定长度的向量中。这可能使得神经网络难以处理长句子，尤其是那些比训练语料库中的句子还要长的句子。Cho等人（2014b年）显示，基本的编码器-解码器的性能确实会随着输入句子长度的增加而迅速下降。为了解决这个问题，我们引入了一种编码器-解码器模型的扩展，它学习联合对齐和翻译。每当提议的模型生成翻译中的一个单词时，它会（软）搜索源句子中与之最相关信息集中所在的一组位置。模型然后基于与这些源位置相关的上下文向量以及所有先前生成的目标单词来预测一个目标单词。这种方法与基本编码器-解码器的最重要区别在于，它并不试图将整个输入句子编码为一个单一固定长度的向量。相反，它将输入句子编码为一系列向量，并在解码翻译时自适应地选择这些向量的一个子集。这使得神经翻译模型不必将源句子的所有信息， независимо于其长度，压缩到一个固定长度的向量中。我们展示了这种方法使模型更好地处理长句子。在本文中，我们表明，提议的联合学习对齐和翻译的方法相较于基本的编码器-解码器方法在翻译性能上有显著提升。改善在较长句子中更为明显，但在任何长度的句子中都可以观察到。在英法翻译任务中，提议的方法通过一个单一模型实现了与传统基于短语的系统相当或接近的翻译性能。此外，定性分析显示，提议的模型发现了源句子与对应目标句子之间在语言上合理的（软）对齐。

# 2 背景：神经机器翻译

从概率的角度来看，翻译相当于寻找一个目标句子 y，以最大化给定源句子 $\mathbf { x }$ 的条件概率，即 $\arg \operatorname* { m a x } _ { \mathbf { y } } p ( \mathbf { y } \mid \mathbf { x } )$。在神经机器翻译中，我们拟合一个参数化模型，以最大化句子对的条件概率，使用并行训练语料库。一旦翻译模型学习到条件分布，给定源句子可以通过寻找最大化条件概率的句子生成相应的翻译。最近，一些论文提出使用神经网络直接学习这种条件分布（参见，例如，Kalchbrenner 和 Blunsom，2013；Cho 等，2014a；Sutskever 等，2014；Cho 等，2014b；Forcada 和 eco，1997）。这种神经机器翻译方法通常由两个组件组成，第一个组件对源句子 $\mathbf { x }$ 进行编码，第二个组件将其解码为目标句子 y。例如，(Cho 等，2014a) 和 (Sutskever 等，2014) 使用了两个递归神经网络（RNN）将可变长度的源句子编码为固定长度的向量，并将该向量解码为可变长度的目标句子。尽管这是一种相对较新的方法，神经机器翻译已经显示出良好的结果。Sutskever 等（2014）报道，基于长短期记忆（LSTM）单元的 RNN 的神经机器翻译在英法翻译任务上达到了接近传统基于短语机器翻译系统的最先进性能。将神经组件添加到现有翻译系统中，例如，为短语表中的短语对打分（Cho 等，2014a）或重新排序候选翻译（Sutskever 等，2014），使得超越了之前的最先进性能水平。

# 2.1 RNN 编码器-解码器

在这里，我们简要描述基础框架，即RNN编码器-解码器，由Cho等人（2014a）和Sutskever等人（2014）提出，我们基于此构建了一种新颖的架构，能够同时学习对齐和翻译。在编码器-解码器框架中，编码器读取输入句子，一个向量序列$\mathbf x = \left( x_{1}, \cdots, x_{T_{x}} \right)$，将其转换为向量$c$。最常用的方法是使用RNN，其中$\boldsymbol{h}_{t} \in \mathbb{R}^{n}$是时刻$t$的隐状态，而$c$是从隐状态序列生成的向量。$f$和$q$是一些非线性函数。例如，Sutskever等人（2014）使用LSTM作为$f$，而$q \left( \{ h_{1}, \cdots, h_{T} \} \right) = \bar{h}_{T}$。

$$
h _ { t } = f \left( x _ { t } , h _ { t - 1 } \right)
$$

$$
c = q \left( \{ h _ { 1 } , \cdots , h _ { T _ { x } } \} \right) ,
$$

解码器通常被训练为预测下一个单词 $y _ { t ^ { \prime } }$，给定上下文向量 $c$ 和所有先前预测的单词 $\{ y _ { 1 } , \cdot \cdot \cdot , y _ { t ^ { \prime } - 1 } \}$。换句话说，解码器通过将联合概率分解为有序的条件概率，定义了翻译 $\mathbf { y }$ 的概率：

$$
p ( \mathbf { y } ) = \prod _ { t = 1 } ^ { T } p ( y _ { t } \mid \{ y _ { 1 } , \cdot \cdot \cdot , y _ { t - 1 } \} , c ) ,
$$

其中 $\mathbf { y } = \left( y _ { 1 } , \cdot \cdot \cdot , y _ { T _ { y } } \right)$。在 RNN 中，每个条件概率被建模为，其中 $g$ 是一个非线性的、可能是多层的函数，它输出 $y _ { t }$ 的概率，而 $s _ { t }$ 是 RNN 的隐含状态。值得注意的是，还可以使用其他架构，例如 RNN 与去卷积神经网络的混合结构（Kalchbrenner 和 Blunsom, 2013）。

$$
p ( y _ { t } \mid \left\{ y _ { 1 } , \cdot \cdot \cdot , y _ { t - 1 } \right\} , c ) = g ( y _ { t - 1 } , s _ { t } , c ) ,
$$

# 3 学习对齐与翻译

在本节中，我们提出了一种新颖的神经机器翻译架构。该新架构由一个双向递归神经网络作为编码器（第3.2节）和一个在解码翻译时模拟源句子搜索的解码器组成（第3.1节）。

# 3.1 解码器：一般描述

在一种新模型架构中，我们将公式 (2) 中的每个条件概率定义为：

$$
p ( y _ { i } | y _ { 1 } , \dots , y _ { i - 1 } ,  { \mathbf { x } } ) = g ( y _ { i - 1 } , s _ { i } , c _ { i } ) ,
$$

其中 $s_{i}$ 是在时间 $i$ 计算的 RNN 隐状态，具体计算方式为

$$
s _ { i } = f ( s _ { i - 1 } , y _ { i - 1 } , c _ { i } ) .
$$

需要注意的是，与现有的编码-解码方法不同（见公式 (2)），此处概率是基于每个目标词 $y _ { i }$ 的独特上下文向量 $c _ { i }$ 而定。上下文向量 $c _ { i }$ 依赖于编码器将输入句子映射到的一系列注释 $( h _ { 1 } , \cdots , h _ { T _ { x } } )$。每个注释 $h _ { i }$ 包含有关整个输入序列的信息，尤其关注输入序列中第 $i$ 个词周围的部分。我们将在下一节中详细解释注释的计算方法。

![](images/1.jpg)  

Figure 1: The graphical illustration of the proposed model trying to generate the $t$ -th target word $y _ { t }$ given a source sentence $( x _ { 1 } , x _ { 2 } , \dots , x _ { T } )$ .

上下文向量 $c_{i}$ 然后被计算为这些注释 $h_{i}$ 的加权和：

$$
c _ { i } = \sum _ { j = 1 } ^ { T _ { x } } \alpha _ { i j } h _ { j } .
$$

每个标注 $h _ { j }$ 的权重 $\alpha _ { i j }$ 是通过以下方式计算的，其中有一个对齐模型用来评估输入中位置 $j$ 附近的内容与位置 $i$ 的输出之间匹配的程度。评分基于 RNN 的隐藏状态 $s _ { i - 1 }$（在发出 $y _ { i }$ 之前，参见公式 (4)）和输入句子的第 $j$ 个标注 $h _ { j }$。

$$
\alpha _ { i j } = \frac { \exp { \left( e _ { i j } \right) } } { \sum _ { k = 1 } ^ { T _ { x } } \exp { \left( e _ { i k } \right) } } ,
$$

$$
e _ { i j } = a ( s _ { i - 1 } , h _ { j } )
$$

我们将对齐模型 $a$ 参数化为一个前馈神经网络，并与所提议系统的所有其他组件进行联合训练。注意，与传统机器翻译不同的是，对齐并不被视为潜在变量。相反，对齐模型直接计算软对齐，这允许成本函数的梯度通过反向传播。这个梯度可以用来联合训练对齐模型和整个翻译模型。我们可以将对所有注释进行加权求和的方法理解为计算期望注释，其中期望是基于可能的对齐。设 $\alpha _ { i j }$ 为目标词 $y _ { i }$ 与源词 $x _ { j }$ 之间对齐或翻译的概率。那么，第 $i$ 个上下文向量 $c _ { i }$ 为 $\alpha _ { i j }$。概率 $\alpha _ { i j }$ 或其相关能量 $e _ { i j }$ 反映了注释 $h _ { j }$ 在决定下一个状态 $s _ { i }$ 和生成 $y _ { i }$ 时相对于先前隐藏状态 $s _ { i - 1 }$ 的重要性。本质上，这实现了解码器中的注意力机制。解码器决定关注源句子的哪些部分。通过让解码器具有注意力机制，我们减轻了编码器将源句子的所有信息编码到固定长度向量中的负担。采用这种新方法，信息可以分散到整个注释序列中，解码器可以相应地选择性地检索。

# 3.2 EncodeR：用于序列标注的双向 RnN

通常的循环神经网络（RNN），如公式（1）所述，按照顺序读取输入序列 $\mathbf { x }$，从第一个符号 $x _ { 1 }$ 到最后一个符号 $x _ { T _ { x } }$。然而，在本方案中，我们希望每个词的标注能够总结不仅是前面的词，还有后面的词。因此，我们建议使用双向循环神经网络（BiRNN，Schuster 和 Paliwal，1997），这种网络在语音识别中最近被成功应用（见 Graves 等，2013）。正向 RNN $\vec { \boldsymbol { f } }$ 按照顺序读取输入序列（从 $x _ { 1 }$ 到 $x _ { T _ { x } }$），并计算出一系列正向隐藏状态 $( \vec { h } _ { 1 } , \cdots , \stackrel { \cdot } { \vec { h } _ { T _ { x } } } )$。反向 RNN $\overleftarrow { f }$ 以相反顺序读取序列（从 $x _ { T _ { x } }$ 到 $x _ { 1 }$），生成一系列反向隐藏状态 $( \overleftarrow { h } _ { 1 } , \cdots , \overleftarrow { h } _ { T _ { x } } )$。通过将正向隐藏状态 $\overrightarrow { h } _ { j }$ 和反向隐藏状态 $\overleftarrow { h } _ { j }$ 进行连接，我们为每个词 $x _ { j }$ 得到一个标注，即 $h _ { j } = { \left[ \overrightarrow { h } _ { j } ^ { \top } ; \overleftarrow { h } _ { j } ^ { \top } \right] } ^ { \top }$。通过这种方式，标注 $h _ { j }$ 同时包含了前面词汇和后面词汇的总结。由于 RNN 在表示近期输入时的倾向，标注 $h _ { j }$ 将更加关注围绕 $x _ { j }$ 的词汇。这一系列的标注在后续被解码器和对齐模型用于计算上下文向量（公式（5）（6））。请参见图 1 以获取所提模型的图形示例。

# 4 实验设置

我们在英法翻译任务上评估所提出的方法。我们使用ACL WMT '14提供的双语平行语料库。作为对比，我们还报告了Cho等人（2014a）最近提出的RNN编码解码器的性能。我们对这两种模型使用相同的训练程序和相同的数据集。

# 4.1 数据集

WMT '14包含以下英法平行语料库：欧洲议会（61M词），新闻评论（5.5M），联合国（421M）以及两个爬取的语料库，分别为90M和272.5M词，总共为850M词。按照Cho等人（2014a）所描述的程序，我们使用Axelrod等人（2011）的数据选择方法将合并语料库的大小减少到348M词。我们不使用除上述平行语料库外的任何单语数据，尽管可能使用更大的单语语料库进行编码器的预训练。我们将news-test 2012和news-test-2013连接起来，形成一个开发（验证）集，并在WMT '14的测试集（news-test-2014）上评估模型，该测试集包含3003个不在训练数据中的句子。

![](images/2.jpg)  

Figure 2: The BLEU scores of the generated translations on the test set with respect to the lengths of the sentences. The results are on the full test set which includes sentences having unknown words to the models.

在常规的词元化之后，我们使用每种语言中30,000个最频繁单词的短名单来训练我们的模型。任何不在短名单中的单词都被映射到一个特殊的词元（[UNK]）。我们不对数据应用其他特殊的预处理，如小写或词干提取。

# 4.2 模型

我们训练了两种类型的模型。第一种是RNN编码解码器（RNNencdec，Cho等，2014a），另一种是我们提出的模型，称为RNNsearch。我们对每个模型进行了两次训练：第一次使用长度不超过30个单词的句子（RNNencdec-30，RNNsearch-30），然后使用长度不超过50个单词的句子（RNNencdec-50，RNNsearch-50）。RNNencdec的编码器和解码器各有1000个隐含单元。RNNsearch的编码器由一个前向和一个后向递归神经网络（RNN）组成，各自具有1000个隐含单元。其解码器也有1000个隐含单元。在两种情况下，我们使用具有单个maxout（Goodfellow等，2013）隐含层的多层网络来计算每个目标词的条件概率（Pascanu等，2014）。我们使用带有Adadelta（Zeiler，2012）的迷你批量随机梯度下降（SGD）算法来训练每个模型。每次SGD更新方向都是使用80个句子的迷你批量计算的。我们大约训练了每个模型5天。模型训练完成后，我们使用束搜索找到一个大约最大化条件概率的翻译（参见，例如，Graves，2012；Boulanger-Lewandowski等，2013）。Sutskever等（2014）使用这种方法从他们的神经机器翻译模型生成翻译。有关模型架构和实验中使用的训练过程的更多细节，请参见附录A和B。

# 5 结果

# 5.1 定量结果

在表1中，我们列出了通过BLEU得分测量的翻译性能。显然，从表中可以看出，在所有情况下，提出的RNNsearch的表现优于传统的RNNencdec。更重要的是，当仅考虑由已知单词组成的句子时，RNNsearch的性能达到了传统短语翻译系统（Moses）的水平。这是一个重要的成就，考虑到Moses除了我们用于训练RNNsearch和RNNencdec的平行语料外，还使用了一个单独的单语语料库（418M字）。

![](images/3.jpg)  

Figure 3: Four sample alignments found by RNNsearch-50. The $\mathbf { X }$ -axis and y-axis of each plot correspond to the words in the source sentence (English) and the generated translation (French), respectively. Each pixel shows the weight $\alpha _ { i j }$ of the annotation of the $j$ -th source word for the $i$ -th target word (see Eq. (6)), in grayscale (0: black, 1: white). (a) an arbitrary sentence. (bd) three randomly selected samples among the sentences without any unknown words and of length between 10 and 20 words from the test set.

提出方法的一个动机是基本编码解码器方法中使用固定长度的上下文向量。我们猜测这一限制可能导致基本编码解码器在处理长句子时表现不佳。在图 2 中，我们看到随着句子长度的增加，RNNencdec 的性能显著下降。另一方面，RNNsearch-30 和 RNNsearch-50 对句子长度的变化更加鲁棒。尤其是 RNNsearch-50，甚至在长度为 50 或更长的句子上也没有表现下降。提出模型在性能上优于基本编码解码器的事实进一步得到了证实，即 RNNsearch-30 甚至超越了 RNNencdec-50（见表 1）。

<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>All</td><td rowspan=1 colspan=1>No UNK</td></tr><tr><td rowspan=1 colspan=1>RNNencdec-30RNNsearch-30</td><td rowspan=1 colspan=1>13.9321.50</td><td rowspan=1 colspan=1>24.1931.44</td></tr><tr><td rowspan=1 colspan=1>RNNencdec-50RNNsearch-50</td><td rowspan=1 colspan=1>17.8226.75</td><td rowspan=1 colspan=1>26.7134.16</td></tr><tr><td rowspan=1 colspan=1>RNNsearch-50*</td><td rowspan=1 colspan=1>28.45</td><td rowspan=1 colspan=1>36.15</td></tr><tr><td rowspan=1 colspan=1>Moses</td><td rowspan=1 colspan=1>33.30</td><td rowspan=1 colspan=1>35.63</td></tr></table>

Table 1: BLEU scores of the trained models computed on the test set. The second and third columns show respectively the scores on all the sentences and, on the sentences without any unknown word in themselves and in the reference translations. Note that RNNsearch $5 0 ^ { \star }$ was trained much longer until the performance on the development set stopped improving. (o) We disallowed the models to generate [UNK] tokens when only the sentences having no unknown words were evaluated (last column).

# 5.2 定性分析

# 5.2.1 对齐

所提方法提供了一种直观的方式来检查生成翻译中词语与源句子中词语之间的（软）对齐。这是通过可视化公式(6)中的注释权重 $\alpha _ { i j }$ 实现的，如图3所示。每个图中矩阵的每一行表示与注释相关的权重。由此我们可以看到在生成目标词时，源句中哪些位置被认为更重要。我们可以从图3中的对齐情况来看，英语与法语之间的词汇对齐大体上是单调的。我们在每个矩阵的对角线上看到强权重。然而，我们也观察到一些非平凡的、非单调的对齐。形容词和名词在法语和英语中的顺序通常不同，我们在图3（a）中看到一个例子。从这个图中，我们看到模型将短语[European Economic Area]正确翻译为[zone économique européen]。RNNsearch能够正确将[zone]与[Area]对齐，跳过了两个词（[European]和[Economic]），然后每次向回查看一个词，以完成整个短语[zone économique européenne]。

软对齐相较于硬对齐的优势显而易见，例如在图3(d)中。考虑源短语[the man]，它被翻译为[' homme]。任何硬对齐都会将[the]映射到[1']，将[man]映射到[homme]。这种映射对翻译并没有帮助，因为必须考虑[the]后面的词，以确定它应该翻译为[le]、[la]、[les]还是[']。我们的软对齐自然解决了这个问题，让模型同时关注[the]和[man]，在这个例子中，我们看到模型能够正确地将[the]翻译为[1']。我们在图3中所有呈现的案例中观察到了类似的行为。软对齐的另一个好处是，它能够自然处理长度不同的源短语和目标短语，而无需以一些直观上不合理的方式将某些词映射到或从无处（[NULL]）中（参见2010年Koehn的第4章和第5章）。

# 5.2.2 长句子

如图2所示，所提出的模型（RNNsearch）在翻译长句子方面明显优于传统模型（RNNencdec）。这可能是因为RNNsearch不要求将长句子完美编码为固定长度的向量，而只需准确编码特定单词周围的输入句子部分。例如，考虑来自测试集的源句子：一个接纳特权是医生根据其在医院作为医疗工作者的身份，接纳病人进入医院或医疗中心进行诊断或手术的权利。RNNencdec-50将该句子翻译为：Un privilège d'admission est le droit d'un médecin de reconnaitre un patient à l'hôpital ou un centre médical d'un diagnostic ou de prendre un diagnostic en fonction de son état de santé。RNNencdec-50正确翻译了源句子直到[un centre médical]。然而，从那里开始（下划线部分），它偏离了源句子的原始含义。例如，它将源句中的[based on his status as a health care worker at a hospital]替换为[en fonction de son état de santé]（“根据他的健康状况”）。另一方面，RNNsearch-50生成了以下正确翻译，保留了输入句子的整体含义，未遗漏任何细节：Un privilège d'admission est le droit d'un médecin d'admettre un patient à un hôpital ou un centre médical pour effectuer un diagnostic ou une procédure, selon son statut de travailleur des soins de santé à l'hôpital。让我们考虑测试集中的另一句：这种体验是迪士尼努力“延长其系列的生命周期并通过日益重要的数字平台与观众建立新关系”的一部分，他补充道。RNNencdec-50的翻译为Ce type d'expérience fait partie des initiatives du Disney pour "prolonger la durée de vie de ses nouvelles et de développer des liens avec les lecteurs numériques qui deviennent plus complexes。与之前的例子一样，RNNencdec在生成大约30个单词后开始偏离源句子的实际含义（见下划线部分）。在那之后，翻译质量恶化，出现基本错误，例如缺少结束引号。同样，RNNsearch-50能够正确翻译这句长句：Ce genre d'expérience fait partie des efforts de Disney pour "prolonger la durée de vie de ses séries et créer de nouvelles relations avec des publics via des plateformes numériques de plus en plus importantes", a-t-il ajouté。结合已经展示的定量结果，这些定性观察证实了我们的假设，即RNNsearch架构比标准RNNencdec模型能够提供更可靠的长句翻译。在附录C中，我们提供了RNNencdec-50、RNNsearch-50和谷歌翻译生成的几句长源句子的样本翻译以及参考翻译。

# 6 相关工作

# 6.1 学习对齐

最近，Graves（2013）在手写合成的背景下提出了一种将输出符号与输入符号对齐的类似方法。手写合成是一个任务，模型需要生成给定字符序列的手写文本。在他的研究中，他使用了一种高斯核的混合体来计算标注的权重，其中每个核的位置信息、宽度和混合系数是从对齐模型中预测的。更具体地说，他的对齐限制在预测位置时要求位置单调递增。与我们的方法最大的不同在于，在（Graves，2013）中，标注权重的模式仅向一个方向移动。在机器翻译的背景下，这是一个严重的限制，因为通常需要进行（远距离）重排序以生成语法正确的翻译（例如，从英语到德语）。另一方面，我们的方法需要为翻译中的每个单词计算源句中每个单词的标注权重。虽然这个缺陷在大多数输入和输出句子只有1540个单词的翻译任务中并不是很严重，但这可能限制该方案在其他任务中的适用性。 6.2 机器翻译的神经网络 自从Bengio等人（2003）提出了一种神经概率语言模型，该模型利用神经网络来建模给定固定数量前置单词条件下某个单词的条件概率以来，神经网络已广泛应用于机器翻译。然而，神经网络的作用在很大程度上仅限于为现有的统计机器翻译系统提供单一特征或对现有系统提供的候选翻译列表进行重新排序。例如，Schwenk（2012）提议使用前馈神经网络来计算源短语和目标短语对的分数，并将该分数作为短语基础统计机器翻译系统中的附加特征。最近，Kalchbrenner和Blunsom（2013）以及Devlin等人（2014）报告了神经网络作为现有翻译系统子组件的成功应用。传统上，训练为目标侧语言模型的神经网络已被用于重新评分或重新排序候选翻译列表（参见，例如，Schwenk等人，2006）。尽管上述方法已被证明能改善翻译性能优于最先进的机器翻译系统，但我们更关注的是设计一个基于神经网络的全新翻译系统的更为雄心勃勃的目标。因此，本文考虑的神经机器翻译方法与这些早期工作有着根本性的不同。我们的模型不作为现有系统的一部分，而是独立运作，直接从源句生成翻译。

# 7 结论

传统的神经机器翻译方法称为编码-解码方法，将整个输入句子编码为一个固定长度的向量，从中解码出翻译。我们推测，基于Cho等人（2014b）和Pouget-Abadie等人（2014）的最新实证研究，使用固定长度的上下文向量在翻译长句时存在问题。本文提出了一种新颖的架构，解决了这一问题。我们通过让模型在生成每个目标词时（软）搜索一组输入词或编码器计算的它们的注释来扩展基本的编码-解码结构。这使得模型不必将整个源句子编码为固定长度的向量，并使模型能够专注于与生成下一个目标词相关的信息。这对神经机器翻译系统在长句上的良好表现产生了重大积极影响。与传统机器翻译系统不同，翻译系统的所有部分，包括对齐机制，都是联合训练以产生更高的正确翻译对数概率。我们在英语到法语的翻译任务上测试了所提出的模型，称为RNNsearch。实验结果显示，所提出的RNNsearch显著优于传统的编码-解码模型（RNNencdec），无论句子长度如何，并且对源句子的长度更加稳健。通过定性分析，我们调查了RNNsearch生成的（软）对齐后得出结论，模型能够正确地将每个目标词与源句中相关词或它们的注释对齐，从而生成正确的翻译。或许更重要的是，所提出的方法在翻译表现上可与现有的基于短语的统计机器翻译相媲美。考虑到所提出的架构及整个神经机器翻译系列今年刚刚提出，这一结果令人瞩目。我们相信，这里提议的架构是朝着更好机器翻译的有希望一步，并且有助于更好地理解自然语言。未来面临的挑战之一是更好地处理未知或稀有词汇。为模型更广泛使用并在所有上下文中与当前最先进的机器翻译系统的表现相匹配，这将是必要的。

# 致谢

作者感谢Theano的开发者（Bergstra et al., 2010; Bastien et al., 2012）。我们感谢以下机构对研究经费和计算支持的支持：NSERC、Calcul Québec、Compute Canada、加拿大研究主席和CIFAR。Bahdanau感谢Planet Intelligent Systems GmbH的支持。我们还要感谢Felix Hill、Bart van Merriënboer、Jean Pouget-Abadie、Coline Devin和Tae-Ho Kim。

# REFERENCES

Axelrod, A., He, X., and Gao, J. (2011). Domain adaptation via pseudo in-domain data selection. In Proceedings of the ACL Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 355362. Association for Computational Linguistics.

Bastien, F., Lamblin, P., Pascanu, R., Bergstra, J., Goodfellow, I. J., Bergeron, A., Bouchard, N., and Bengio, Y. (2012). Theano: new features and speed improvements. Deep Learning and Unsupervised Feature Learning NIPS 2012 Workshop.

Bengio, Y., Simard, P., and Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. IEEE Transactions on Neural Networks, 5(2), 157166.

Bengio, Y., Ducharme, R., Vincent, P., and Janvin, C. (2003). A neural probabilistic language model. J. Mach. Learn. Res., 3, 11371155.

Bergstra, J., Breuleux, O., Bastien, F., Lamblin, P., Pascanu, R., Desjardins, G., Turian, J., WardeFarley, D., and Bengio, Y. (2010). Theano: a CPU and GPU math expression compiler. In Proceedings of the Python for Scientific Computing Conference (SciPy). Oral Presentation.

Boulanger-Lewandowski, N., Bengio, Y., and Vincent, P. (2013). Audio chord recognition with recurrent neural networks. In ISMIR.

Cho, K., van Merrienboer, B., Gulcehre, C., Bougares, F., Schwenk, H., and Bengio, Y. (2014a). Learning phrase representations using RNN encoder-decoder for statistical machine translation. In Proceedings of the Empiricial Methods in Natural Language Processing (EMNLP 2014). to appear.

Cho, K., van Merriënboer, B., Bahdanau, D., and Bengio, Y. (2014b). On the properties of neural machine translation: EncoderDecoder approaches. In Eighth Workshop on Syntax, Semantics and Structure in Statistical Translation. to appear.

Devlin, J., Zbib, R., Huang, Z., Lamar, T., Schwartz, R., and Makhoul, J. (2014). Fast and robust neural network joint models for statistical machine translation. In Association for Computational Linguistics.

Forcada, M. L. and Neco, R. P. (1997). Recursive hetero-associative memories for translation. In J. Mira, R. Moreno-Díaz, and J. Cabestany, editors, Biological and Artificial Computation: From Neuroscience to Technology, volume 1240 of Lecture Notes in Computer Science, pages 453462. Springer Berlin Heidelberg.

Goodfellow, I., Warde-Farley, D., Mirza, M., Courville, A., and Bengio, Y. (2013). Maxout networks. In Proceedings of The 30th International Conference on Machine Learning, pages 1319 1327.

Graves, A. (2012). Sequence transduction with recurrent neural networks. In Proceedings of the 29th International Conference on Machine Learning (ICML 2012).

Graves, A. (2013). Generating sequences with recurrent neural networks. arXiv:1308 . 0850 [cs.NE].

Graves, A., Jaitly, N., and Mohamed, A.-R. (2013). Hybrid speech recognition with deep bidirectional LSTM. In Automatic Speech Recognition and Understanding (ASRU), 2013 IEEE Workshop on, pages 273278.

Hermann, K. and Blunsom, P. (2014). Multilingual distributed representations without word alignment. In Proceedings of the Second International Conference on Learning Representations (ICLR 2014).

Hochreiter, S. (1991). Untersuchungen zu dynamischen neuronalen Netzen. Diploma thesis, Institut für Informatik, Lehrstuhl Prof. Brauer, Technische Universität München.

Hochreiter, S. and Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 17351780.

Kalchbrenner, N. and Blunsom, P. (2013). Recurrent continuous translation models. In Proceedings of the ACL Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 17001709. Association for Computational Linguistics.

Koehn, P. (2010). Statistical Machine Translation. Cambridge University Press, New York, NY, USA.

Koehn, P., Och, F. J., and Marcu, D. (2003). Statistical phrase-based translation. In Proceedings of the 2003 Conference of the North American Chapter of the Association for Computational Linguistics on Human Language Technology - Volume 1, NÁACL '03, pages 4854, Stroudsburg, PA, USA. Association for Computational Linguistics.

Pascanu, R., Mikolov, T., and Bengio, Y. (2013a). On the difficulty of training recurrent neural networks. In ICML'2013.

Pascanu, R., Mikolov, T., and Bengio, Y. (2013b). On the difficulty of training recurrent neural networks. In Proceedings of the 30th International Conference on Machine Learning (ICML 2013).

Pascanu, R., Gulcehre, C., Cho, K., and Bengio, Y. (2014). How to construct deep recurrent neural networks. In Proceedings of the Second International Conference on Learning Representations (ICLR 2014).

Pouget-Abadie, J., Bahdanau, D., van Merrinboer, B., Cho, K., and Bengio, Y. (2014). Overcoming the curse of sentence length for neural machine translation using automatic segmentation. In Eighth Workshop on Syntax, Semantics and Structure in Statistical Translation. to appear.

Schuster, M. and Paliwal, K. K. (1997). Bidirectional recurrent neural networks. Signal Processing, IEEE Transactions on, 45(11), 26732681.

Schwenk, H. (2012). Continuous space translation models for phrase-based statistical machine translation. In M. Kay and C. Boitet, editors, Proceedings of the 24th International Conference on Computational Linguistics (COLIN), pages 10711080. Indian Institute of Technology Bombay.

Schwenk, H., Dchelotte, D., and Gauvain, J.-L. (2006). Continuous space language models for statistical machine translation. In Proceedings of the COLING/ACL on Main conference poster sessions, pages 723730. Association for Computational Linguistics.

Sutskever, I., Vinyals, O., and Le, Q. (2014). Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems (NIPS 2014).

Zeiler, M. D. (2012). ADADELTA: An adaptive learning rate method. arXiv:1212.5701 [cs.LG].

# A MODEL ARCHITECTURE

# A.1 ARCHITECtuRAL CHoICES

The proposed scheme in Section 3 is a general framework where one can freely define, for instance, the activation functions $f$ of recurrent neural networks (RNN) and the alignment model $a$ .Here, we describe the choices we made for the experiments in this paper.

# A.1.1 RECurrENT NEurAL NETWork

For the activation function $f$ of an RNN, we use the gated hidden unit recently proposed by Cho et al. (2014a). The gated hidden unit is an alternative to the conventional simple units such as an element-wise tanh. This gated unit is similar to a long short-term memory (LSTM) unit proposed earlier by Hochreiter and Schmidhuber (1997), sharing with it the ability to better model and learn long-term dependencies. This is made possible by having computation paths in the unfolded RNN for which the product of derivatives is close to 1. These paths allow gradients to flow backward easily without suffering too much from the vanishing effect (Hochreiter, 1991; Bengio et al., 1994; Pascanu et al., 2013a). It is therefore possible to use LSTM units instead of the gated hidden unit described here, as was done in a similar context by Sutskever et al. (2014).

The new state $s _ { i }$ of the RNN employing $n$ gated hidden units8 is computed by

$$
s _ { i } = f ( s _ { i - 1 } , y _ { i - 1 } , c _ { i } ) = ( 1 - z _ { i } ) \circ s _ { i - 1 } + z _ { i } \circ \tilde { s } _ { i } ,
$$

where $\circ$ is an element-wise multiplication, and $z _ { i }$ is the output of the update gates (see below). The proposed updated state $\tilde { s } _ { i }$ is computed by

$$
\tilde { s } _ { i } = \operatorname { t a n h } \left( W e ( y _ { i - 1 } ) + U \left[ r _ { i } \circ s _ { i - 1 } \right] + C c _ { i } \right) ,
$$

where $e ( y _ { i - 1 } ) \in \mathbb { R } ^ { m }$ is an $m$ -dimensional embedding of a word $y _ { i - 1 }$ , and $r _ { i }$ is the output of the reset gates (see below). When $y _ { i }$ is represented as a 1-of- $K$ vector, $e ( y _ { i } )$ is simply a column of an embedding matrix $E \in \mathbb { R } ^ { m \times K }$ Whe ossle ei bs ter  make  euas les cluttered.

The update gates $z _ { i }$ allow each hidden unit to maintain its previous activation, and the reset gates $r _ { i }$ control how much and what information from the previous state should be reset. We compute them by

$$
\begin{array} { r } { z _ { i } = \sigma \left( W _ { z } e ( y _ { i - 1 } ) + U _ { z } s _ { i - 1 } + C _ { z } c _ { i } \right) , } \\ { r _ { i } = \sigma \left( W _ { r } e ( y _ { i - 1 } ) + U _ { r } s _ { i - 1 } + C _ { r } c _ { i } \right) , } \end{array}
$$

where $\sigma \left( \cdot \right)$ is a logistic sigmoid function.

At each step of the decoder, we compute the output probability (Eq. (4)) as a multi-layered function (Pascanu et al., 2014). We use a single hidden layer of maxout units (Goodfellow et al., 2013) and normalize the output probabilities (one for each word) with a softmax function (see Eq. (6)).

# A.1.2 ALIGNMENT MODEL

The alignment model should be designed considering that the model needs to be evaluated $T _ { x } \times T _ { y }$ times for each sentence pair of lengths $T _ { x }$ and $T _ { y }$ . In order to reduce computation, we use a singlelayer multilayer perceptron such that

$$
a ( s _ { i - 1 } , h _ { j } ) = v _ { a } ^ { \top } \operatorname { t a n h } \left( W _ { a } s _ { i - 1 } + U _ { a } h _ { j } \right) ,
$$

where $W _ { a } \in \mathbb { R } ^ { n \times n } , U _ { a } \in \mathbb { R } ^ { n \times 2 n }$ and $v _ { a } \in \mathbb { R } ^ { n }$ are the weight matrices. Since $U _ { a } h _ { j }$ does not depend on $i$ , we can pre-compute it in advance to minimize the computational cost.

# A.2 Detailed Description of the Model

# A.2.1 EnCOdER

In this section, we describe in detail the architecture of the proposed model (RNNsearch) used in the experiments (see Sec. 45). From here on, we omit all bias terms in order to increase readability.

The model takes a source sentence of 1-of-K coded word vectors as input

$$
\mathbf { x } = ( x _ { 1 } , \ldots , x _ { T _ { x } } ) , x _ { i } \in \mathbb { R } ^ { K _ { x } }
$$

and outputs a translated sentence of 1-of-K coded word vectors

$$
\mathbf { y } = ( y _ { 1 } , \dots , y _ { T _ { y } } ) , y _ { i } \in \mathbb { R } ^ { K _ { y } } ,
$$

where $K _ { x }$ and $K _ { y }$ are the vocabulary sizes of source and target languages, respectively. $T _ { x }$ and $T _ { y }$ respectively denote the lengths of source and target sentences.

First, the forward states of the bidirectional recurrent neural network (BiRNN) are computed:

$$
\begin{array} { r } { \overrightarrow { h } _ { i } = \left\{ \begin{array} { l l } { ( 1 - \overrightarrow { z } _ { i } ) \circ \overrightarrow { h } _ { i - 1 } + \overrightarrow { z } _ { i } \circ \overrightarrow { \underline { { h } } } _ { i } } & { , \mathrm { i f } i > 0 } \\ { 0 } & { , \mathrm { i f } i = 0 } \end{array} \right. } \end{array}
$$

where

$$
\begin{array} { r l } & { \vec { \underline { { h } } } _ { i } = \operatorname { t a n h } ( \overrightarrow { W } \overline { { E } } x _ { i } + \overrightarrow { U } [ \overrightarrow { r } _ { i } \circ \stackrel {  } { h } _ { i - 1 } ] ) } \\ & { \vec { \ z } _ { i } = \sigma ( \overrightarrow { W } _ { z } \overline { { E } } x _ { i } + \overrightarrow { U } _ { z } \overrightarrow { h } _ { i - 1 } ) } \\ & { \vec { \ r } _ { i } = \sigma ( \overrightarrow { W } _ { r } \overline { { E } } x _ { i } + \overrightarrow { U } _ { r } \overrightarrow { h } _ { i - 1 } ) . } \end{array}
$$

$\overline { { E } } \in \mathbb { R } ^ { m \times K _ { x } }$ is the word embedding matrix. $\overrightarrow { W } , \overrightarrow { W } _ { z } , \overrightarrow { W } _ { r } \in \mathbb { R } ^ { n \times m } , \overrightarrow { U } , \overrightarrow { U } _ { z } , \overrightarrow { U } _ { r } \in \mathbb { R } ^ { n \times n }$ are weight matrices. $m$ and $n$ are the word embedding dimensionality and the number of hidden units, respectively. $\sigma ( \cdot )$ is as usual a logistic sigmoid function.

The backward states $( \overleftarrow { h } _ { 1 } , \cdot \cdot \cdot , \overleftarrow { h } _ { T _ { x } } )$ $\overline { E }$ between the forward and backward RNNs, unlike the weight matrices.

We concatenate the forward and backward states to to obtain the annotations $( h _ { 1 } , h _ { 2 } , \cdots , h _ { T _ { x } } )$ where

$$
h _ { i } = \left[ \begin{array} { l } { \overrightarrow { h } _ { i } } \\ { \overleftarrow { h } _ { i } } \end{array} \right]
$$

# A.2.2 DECODER

The hidden state $s _ { i }$ of the decoder given the annotations from the encoder is computed by

$$
s _ { i } = ( 1 - z _ { i } ) \circ s _ { i - 1 } + z _ { i } \circ \tilde { s } _ { i } ,
$$

where

$$
\begin{array} { r l } & { \tilde { s } _ { i } = \operatorname { t a n h } \left( W E y _ { i - 1 } + U \left[ r _ { i } \circ s _ { i - 1 } \right] + C c _ { i } \right) } \\ & { z _ { i } = \sigma \left( W _ { z } E y _ { i - 1 } + U _ { z } s _ { i - 1 } + C _ { z } c _ { i } \right) } \\ & { r _ { i } = \sigma \left( W _ { r } E y _ { i - 1 } + U _ { r } s _ { i - 1 } + C _ { r } c _ { i } \right) } \end{array}
$$

$E$ is the word embedding matrix for the target language. $W , W _ { z } , W _ { r } \in \mathbb { R } ^ { n \times m } , U , U _ { z } , U _ { r } \in \mathbb { R } ^ { n \times n }$ and $C , C _ { z } , C _ { r } \ \in \ \mathbb { R } ^ { n \times 2 \overline { { n } } }$ are weights. Again, $m$ and $n$ are the word embedding dimensionality and the number of hidden units, respectively. The initial hidden state $s _ { 0 }$ is computed by $s _ { 0 } ~ =$ $\operatorname { t a n h } \left( W _ { s } \overleftarrow { h } _ { 1 } \right)$ , where $W _ { s } \in \mathbb { R } ^ { n \times n }$ .

The context vector $c _ { i }$ are recomputed at each step by the alignment model:

$$
c _ { i } = \sum _ { j = 1 } ^ { T _ { x } } \alpha _ { i j } h _ { j } ,
$$

<table><tr><td>Model</td><td>Updates (×105)</td><td>Epochs</td><td>Hours</td><td>GPU</td><td>Train NLL</td><td>Dev. NLL</td></tr><tr><td>RNNenc-30</td><td>8.46</td><td>6.4</td><td>109</td><td>TITAN BLACK</td><td>28.1</td><td>53.0</td></tr><tr><td>RNNenc-50</td><td>6.00</td><td>4.5</td><td>108</td><td>Quadro K-6000</td><td>44.0</td><td>43.6</td></tr><tr><td>RNNsearch-30</td><td>4.71</td><td>3.6</td><td>113</td><td>TITAN BLACK</td><td>26.7</td><td>47.2</td></tr><tr><td>RNNsearch-50</td><td>2.88</td><td>2.2</td><td>111</td><td>Quadro K-6000</td><td>40.7</td><td>38.1</td></tr><tr><td>RNNsearch-50*</td><td>6.67</td><td>5.0</td><td>252</td><td>Quadro K-6000</td><td>36.7</td><td>35.2</td></tr></table>

Table 2: Learning statistics and relevant information. Each update corresponds to updating the parameters once using a single minibatch. One epoch is one pass through the training set. NLL is the average conditional log-probabilities of the sentences in either the training set or the development set. Note that the lengths of the sentences differ.

where

$$
\begin{array} { l } { \displaystyle \alpha _ { i j } = \frac { \exp \left( e _ { i j } \right) } { \sum _ { k = 1 } ^ { T _ { x } } \exp \left( e _ { i k } \right) } } \\ { \displaystyle e _ { i j } = v _ { a } ^ { \top } \operatorname { t a n h } \left( W _ { a } s _ { i - 1 } + U _ { a } h _ { j } \right) , } \end{array}
$$

and $h _ { j }$ is the $j$ -th annotation in the source sentence (see Eq. (7)). $v _ { a } \in \mathbb { R } ^ { n ^ { \prime } } , W _ { a } \in \mathbb { R } ^ { n ^ { \prime } \times n }$ and $U _ { a } ~ \in \overline { { \mathbb { R } ^ { n ^ { \prime } \times 2 n } } }$ are weight matrices. Note that the model becomes RNN Encoder-Decoder (Cho et al., 2014a), if we fix $c _ { i }$ to $\vec { h } _ { \ : T _ { x } }$ .

With the decoder state $s _ { i - 1 }$ , the context $c _ { i }$ and the last generated word $y _ { i - 1 }$ , we define the probability of a target word $y _ { i }$ as

$$
p ( y _ { i } | s _ { i } , y _ { i - 1 } , c _ { i } ) \propto \exp \left( y _ { i } ^ { \top } W _ { o } t _ { i } \right) ,
$$

where

$$
\boldsymbol { t } _ { i } = \left[ \operatorname* { m a x } \left\{ \tilde { t } _ { i , 2 j - 1 } , \tilde { t } _ { i , 2 j } \right\} \right] _ { j = 1 , \ldots , l } ^ { \top }
$$

and $\tilde { t } _ { i , k }$ is the $k$ -th element of a vector $\tilde { t } _ { i }$ which is computed by

$$
\tilde { t } _ { i } = U _ { o } s _ { i - 1 } + V _ { o } E y _ { i - 1 } + C _ { o } c _ { i } .
$$

$W _ { o } \in \mathbb { R } ^ { K _ { y } \times l } , U _ { o } \in \mathbb { R } ^ { 2 l \times n } , V _ { o } \in \mathbb { R } ^ { 2 l \times m }$ and $C _ { o } \in \mathbb { R } ^ { 2 l \times 2 n }$ are weight matrices. This can be understood as having a deep output (Pascanu et al., 2014) with a single maxout hidden layer (Goodfellow et al., 2013).

# A.2.3 MODEL SIZE

For allthe models used in this paper, the size of a hidden layer $n$ is 1000, the word embedding dimensionality $m$ is 620 and the size of the maxout hidden layer in the deep output $l$ is 500. The number of hidden units in the alignment model $n ^ { \prime }$ is 1000.

# B TRAINInG PROCEDURE

# B.1 PARAMETER INITIALIZATION

Wi $U , U _ { z } , U _ { r } , \overleftarrow U , \overleftarrow U _ { z } , \overleftarrow U _ { r } , \overrightarrow U , \overrightarrow U _ { z }$ and $\smash { \vec { U } _ { r } }$ as random orthogonal matrices. For $W _ { a }$ and $U _ { a }$ , we initialized them by sampling each element from the Gaussian distribution of mean 0 and variance $0 . 0 0 1 ^ { 2 }$ . All the elements of $V _ { a }$ and all the bias vectors were initialized to zero. Any other weight matrix was initialized by sampling from the Gaussian distribution of mean 0 and variance $0 . 0 1 ^ { 2 }$ .

# B.2 TRAINING

We used the stochastic gradient descent (SGD) algorithm. Adadelta (Zeiler, 2012) was used to automatically adapt the learning rate of each parameter $\epsilon = 1 0 ^ { - 6 }$ and $\rho = 0 . 9 5$ We explicitly normalized the $L _ { 2 }$ -norm of the gradient of the cost function each time to be at most a predefined threshold of 1, when the norm was larger than the threshold (Pascanu et al., 2013b). Each SGD update direction was computed with a minibatch of 80 sentences.

At each update our implementation requires time proportional to the length of the longest sentence in a minibatch. Hence, to minimize the waste of computation, before every 20-th update, we retrieved 1600 sentence pairs, sorted them according to the lengths and split them into 20 minibatches. The training data was shuffled once before training and was traversed sequentially in this manner.

In Tables 2 we present the statistics related to training all the models used in the experiments.

# C Translations of Long Sentences

Table 3: The translations generated by RNNenc-50 and RNNsearch-50 from long source sentences (30 words or more) selected from the test set. For each source sentence, we also show the goldstandard translation. The translations by Google Translate were made on 27 August 2014.   

<table><tr><td rowspan=1 colspan=1>Source</td><td rowspan=1 colspan=1>An admitting privilege is the right of a doctor to admit a patient to a hospital or a medical centreto carry out a diagnosis or a procedure, based on his status as a health care worker at a hospital.</td></tr><tr><td rowspan=1 colspan=1>Reference</td><td rowspan=1 colspan=1>Le privilège d&#x27;admission est le droit d&#x27;un médecin, en vertu de son statut de membre soignantd&#x27;un hôpital, d&#x27;admettre un patient dans un hôpital ou un centre médical afin d&#x27;y délivrer undiagnostic ou un traitement.</td></tr><tr><td rowspan=1 colspan=1>RNNenc-50</td><td rowspan=1 colspan=1>Un privilège d&#x27;admission est le droit d&#x27;un médecin de reconnaitre un patient à l&#x27;hôpital ou uncentre médical d&#x27;un diagnostic ou de prendre un diagnostic en fonction de son état de santé.</td></tr><tr><td rowspan=1 colspan=1>RNNsearch-50</td><td rowspan=1 colspan=1>Un privilège d&#x27;admission est le droit d&#x27;un médecin d&#x27;admettre un patient à un hôpital ou uncentre médical pour effectuer un diagnostic ou une procédure, selon son statut de travailleur dessoins de santé à l&#x27;hôpital.</td></tr><tr><td rowspan=1 colspan=1>GoogleTranslate</td><td rowspan=1 colspan=1>Un privilège admettre est le droit d&#x27;un médecin d&#x27;admettre un patient dans un hôpital ou uncentre médical pour effectuer un diagnostic ou une procédure, fondée sur sa situation en tantque travailleur de soins de santé dans un hôpital.</td></tr><tr><td rowspan=1 colspan=1></td><td></td></tr><tr><td rowspan=1 colspan=1>Source</td><td rowspan=1 colspan=1>This kind of experience is part of Disney&#x27;s efforts to &quot;extend the lifetime of its series and buildnew relationships with audiences via digital platforms that are becoming ever more important,&quot;he added.</td></tr><tr><td rowspan=1 colspan=1>Reference</td><td rowspan=1 colspan=1>Ce type d&#x27;expérience entre dans le cadre des efforts de Disney pour &quot;´étendre la durée devie de ses séries et construire de nouvelles relations avec son public grâce à des plateformesnumériques qui sont de plus en plus importantes&quot;, a-t-il ajouté.</td></tr><tr><td rowspan=1 colspan=1>RNNenc-50</td><td rowspan=1 colspan=1>Ce type d&#x27;expérience fait partie des initiatives du Disney pour &quot;prolonger la durée de vie deses nouvelles et de développer des liens avec les lecteurs numériques qui deviennent plus com-plexes.</td></tr><tr><td rowspan=1 colspan=1>RNNsearch-50</td><td rowspan=1 colspan=1>Ce genre d&#x27;expérience fait partie des efforts de Disney pour &quot;prolonger la durée de vie de sesséries et créer de nouvelles relations avec des publics via des plateformes numériques de plusen plus importantes&quot;, a-t-il ajouté.</td></tr><tr><td rowspan=1 colspan=1>GoogleTranslate</td><td rowspan=1 colspan=1>Ce genre d&#x27;expérience fait partie des efforts de Disney à &quot;étendre la durée de vie de sa série etconstruire de nouvelles relations avec le public par le biais des plates-formes numériques quideviennent de plus en plus important&quot;, at-il ajouté.</td></tr><tr><td rowspan=1 colspan=1></td><td></td></tr><tr><td rowspan=1 colspan=1>Source</td><td rowspan=1 colspan=1>In a press conference on Thursday, Mr Blair stated that there was nothing in this video that mightconstitute a &quot;reasonable motive&quot; that could lead to criminal charges being brought against themayor.</td></tr><tr><td rowspan=1 colspan=1>Reference</td><td rowspan=1 colspan=1>En conférence de presse, jeudi, M. Blair a affrmé qu&#x27;il n&#x27;y avait rien dans cette vidéo qui puisseconstituer des &quot;motifs raisonnables&quot; pouvant mener au dépôt d&#x27;une accusation criminelle contrele maire.</td></tr><tr><td rowspan=1 colspan=1>RNNenc-50</td><td rowspan=1 colspan=1>Lors de la conférence de presse de jeudi, M. Blair a dit qu&#x27;il n&#x27;y avait rien dans cette vidéo quipourrait constituer une &quot;motivation raisonnable&quot; pouvant entrainer des accusations criminellesportées contre le maire.</td></tr><tr><td rowspan=1 colspan=1>RNNsearch-50</td><td rowspan=1 colspan=1>Lors d&#x27;une conférence de presse jeudi, M. Blair a déclaré qu&#x27;il n&#x27;y avait rien dans cette vidéo quipourrait constituer un &quot;motif raisonnable&quot; qui pourrait conduire à des accusations criminellescontre le maire.</td></tr><tr><td rowspan=1 colspan=1>GoogleTranslate</td><td rowspan=1 colspan=1>Lors d&#x27;une conférence de presse jeudi, M. Blair a déclaré qu&#x27;il n&#x27;y avait rien dans cette vidoqui pourrait constituer un &quot;motif raisonnable&quot; qui pourrait mener à des accusations criminellesportes contre le maire.</td></tr></table>