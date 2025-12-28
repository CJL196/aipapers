# 结合检索的生成模型用于知识密集型自然语言处理任务

帕特里克·刘易斯†‡, 伊桑·佩雷斯\*, 亚历山德拉·皮克图斯†, 法比奥·佩特罗尼†, 弗拉基米尔·卡尔普欣†, 纳曼·戈亚尔†, 海因里希·库特勒†, 迈克·刘易斯†,文涛 $\mathbf { V i h } ^ { \dagger }$ , 蒂姆·罗克塔斯切尔† ‡, 塞巴斯蒂安·里德尔† ‡, 道威·基拉† †Facebook AI 研究; ‡伦敦大学学院; \*纽约大学; plewis@fb.com

# 摘要

大型预训练语言模型已被证明能够在其参数中存储事实知识，并在下游自然语言处理任务微调时取得最先进的结果。然而，它们获取和精确操作知识的能力仍然有限，因此在知识密集型任务上的表现不及特定任务的架构。此外，提供决策来源和更新世界知识仍然是开放的研究问题。到目前为止，具有可微分访问机制的预训练模型，仅在提取型下游任务中进行了研究。我们探索了一种通用的微调方法用于检索增强生成（RAG）——结合预训练参数记忆和非参数记忆用于语言生成的模型。我们引入的RAG模型中，参数记忆是一个预训练的序列到序列模型，而非参数记忆是维基百科的稠密向量索引，通过预训练的神经检索器进行访问。我们比较了两种RAG形式，一种是对生成序列中的相同检索段进行条件限制，另一种则允许每个词元使用不同的段落。我们对一系列知识密集型自然语言处理任务进行了微调和评估，并在三个开放域问答任务上取得了最先进的成果，超越了参数序列到序列模型和特定任务的检索与提取架构。在语言生成任务中，我们发现RAG模型生成的语言比最先进的仅基于参数的序列到序列基准更加具体、多样和符合事实。

# 1 引言

预训练神经语言模型已被证明能够从数据中学习大量深层知识。它们能够做到这一点而无需访问外部记忆，因为其本质上是一个参数化的隐式知识库。尽管这一发展令人振奋，但此类模型确实存在缺点：它们无法轻松扩展或修订其记忆，无法直接提供对其预测的深入见解，并且可能产生“幻觉”。结合参数记忆与非参数（即基于检索）记忆的混合模型可以解决部分这些问题，因为知识可以直接被修订和扩展，访问的知识可以被检查和解释。REALM和ORQA这两个最近提出的模型结合了掩码语言模型与可微分检索器，已显示出良好的效果，但仅探索了开放域的抽取式问答。在这里，我们将混合的参数记忆和非参数记忆引入到“自然语言处理的工作马”——即序列到序列（seq2seq）模型中。

![](images/1.jpg)  

Figure 1: Overview of our approach. We combine a pre-trained retriever (Query Encoder $^ +$ Document Index) with a pre-trained seq2seq model (Generator) and fine-tune end-to-end. For query $x$ , we use Maximum Inner Product Search (MIPS) to find the top-K documents $z _ { i }$ For final prediction $y$ ,we treat $z$ as a latent variable and marginalize over seq2seq predictions given different documents.

我们通过一种通用的微调方法为预训练的参数记忆生成模型赋予了非参数记忆，我们称之为检索增强生成（RAG）。我们构建了RAG模型，其中参数记忆是一个预训练的seq2seq转换器，而非参数记忆是维基百科的稠密向量索引，通过预训练的神经检索器访问。我们在一个端到端的概率模型中结合了这些组件（见图1）。检索器（稠密段落检索器 [26]，以下简称DPR）根据输入提供潜在文档，然后seq2seq模型（BART [32]）结合这些潜在文档和输入生成输出。我们使用top-K近似方法对潜在文档进行边际化处理，可以是基于每个输出（假设同一文档负责所有词元）或者基于每个词元（不同文档负责不同词元）。像T5 [51]或BART一样，RAG可以在任何seq2seq任务上进行微调，其中生成器和检索器共同学习。之前有许多工作提出了通过非参数记忆来丰富系统的架构，这些架构是为特定任务从零开始训练的，例如记忆网络 [64, 55]、堆栈增强网络 [25] 和记忆层 [30]。相较之下，我们探讨了一种设置，在该设置中，参数和非参数记忆组件均为预训练且预先加载了丰富的知识。重要的是，通过使用预训练的访问机制，获取知识的能力得以实现，而无需额外训练。

我们的研究结果突显了将参数化和非参数化记忆与生成相结合在知识密集型任务中的优势——这些任务是人类无法在没有外部知识源访问的情况下合理完成的。我们的检索增强生成（RAG）模型在开放自然问题（Natural Questions）、WebQuestions 和 CuratedTrec 上达到了最先进的结果，并且在 TriviaQA 上显著超越了使用专业预训练目标的最新方法。尽管这些任务是抽取式的，我们发现无约束生成的表现超过了之前的抽取式方法。在知识密集型生成方面，我们对 MS-MARCO 和 Jeopardy 问题生成进行了实验，发现我们的模型生成的回答在事实性、具体性和多样性上均优于 BART 基线。在 FEVER 事实验证中，我们的成绩达到了当前最先进流程模型的 $4.3 \%$ 的范围，这些模型使用了强有力的检索监督。最后，我们展示了非参数化记忆可以被替代，以便随着世界的变化更新模型的知识。

# 2 方法

我们探讨了RAG模型，该模型利用输入序列$x$来检索文本文档$z$，并在生成目标序列$y$时将其作为额外上下文。如图1所示，我们的模型利用了两个组件：(i) 检索器$p _ { \eta } ( z | x )$，其参数为$\eta$，根据查询$x$返回文本段落的(top-K截断)分布，以及(ii) 生成器$p _ { \theta } ( y _ { i } | x , z , y _ { 1 : i - 1 } )$，其参数为$\theta$，根据前$i - 1$个令牌$y _ { 1 : i - 1 }$、原始输入$x$和检索到的文段$z$生成当前令牌。为了端到端地训练检索器和生成器，我们将检索到的文档视为潜变量。我们提出了两种模型，以不同的方式对潜在文档进行边际化，从而产生生成文本的分布。在一种方法中，RAG-Sequence，模型使用相同的文档来预测每个目标令牌。第二种方法，RAG-Token，可以基于不同的文档预测每个目标令牌。接下来，我们将正式介绍这两种模型，并描述$p _ { \eta }$和$p _ { \theta }$组件，以及训练和解码过程。

# 2.1 模型

RAG-Sequence 模型使用相同的检索文档来生成完整序列。从技术上讲，它将检索到的文档视为一个单一的潜变量，通过 top-K 近似方法来获取 seq2seq 概率 $p ( y | x )$。具体来说，使用检索器获取前 $\mathbf{K}$ 个文档，生成器为每个文档生成输出序列概率，然后进行边缘化处理。

$$
p _ { \mathrm { R A G . S e q u e n c e } } ( y | x ) \approx \sum _ { z \in \mathrm { t o p } \cdot k ( p ( z | x ) ) } p _ { \theta } ( y | x , z ) \ = \ \sum _ { z \in \mathrm { t o p } \cdot k ( p ( \cdot | x ) ) } p _ { \eta } ( z | x ) \prod _ { i } ^ { N } p _ { \theta } ( y _ { i } | x , z , y _ { 1 : i - 1 } )
$$

RAG-Token模型 在RAG-Token模型中，我们可以为每一个目标词元抽取不同的潜在文档，并相应地进行边际化。这使得生成器在生成答案时可以从多个文档中选择内容。具体而言，使用检索器检索前K个文档，然后生成器为每个文档产生下一个输出词元的分布，在进行边际化之前，并在生成下一个输出词元时重复这一过程。形式上，我们定义：

$$
p _ { \mathtt { R A G - I o k e n } } ( y | x ) \approx \prod _ { i } ^ { N } \sum _ { z \in \mathrm { t o p } \cdot k ( p ( \cdot | x ) ) } p _ { \eta } ( z | x ) p _ { \theta } ( y _ { i } | x , z , y _ { 1 : i - 1 } )
$$

最后，我们注意到 RAG 可以通过将目标类别视为长度为一的目标序列来用于序列分类任务，此时 RAG-Sequence 和 RAG-Token 是等价的。

# 2.2 检索器：DPR

检索组件 $p _ { \eta } ( z | x )$ 基于DPR [26]。DPR遵循双编码器架构：

$$
p _ { \eta } ( z | x ) \propto \exp \left( \mathbf { d } ( z ) ^ { \top } \mathbf { q } ( x ) \right) \qquad \mathbf { d } ( z ) = \mathrm { B E R T } _ { d } ( z ) , \ \mathbf { q } ( x ) = \mathrm { B E R T } _ { q } ( x )
$$

其中 $\mathbf { d } ( z )$ 是由 BERTBAsE 文档编码器生成的文档的稠密表示 [8]，$\mathbf { q } ( x )$ 是由同样基于 BERTBAsE 的查询编码器生成的查询表示。计算前 $\cdot \mathbf { k } ( p _ { \eta } ( \cdot | x ) )$，即具有最高先验概率 $p _ { \eta } ( z | x )$ 的 $k$ 个文档 $z$ 的列表，是一个最大内积搜索（MIPS）问题，可以在次线性时间内近似求解 [23]。我们使用从 DPR 预训练的双编码器来初始化我们的检索器并构建文档索引。该检索器经过训练以检索包含 TriviaQA [24] 问题和自然问题 [29] 答案的文档。我们将文档索引称为非参数记忆。

# 2.3 生成器：BART

生成器组件 $p _ { \theta } ( y _ { i } | x , z , y _ { 1 : i - 1 } )$ 可以使用任何编码器-解码器模型进行建模。我们使用 BART-large [32]，这是一个具有 4 亿参数的预训练序列到序列变换器 [58]。为了在从 BART 生成时将输入 $x$ 与检索到的内容 $z$ 结合起来，我们简单地将它们连接在一起。BART 是使用去噪目标和多种不同的噪声函数进行预训练的。在一系列多样的生成任务中，它取得了最先进的结果，并且优于大小相当的 T5 模型 [32]。我们将 BART 生成器参数 $\theta$ 统称为参数化记忆。

# 2.4 训练

我们联合训练检索器和生成器组件，而不对应当检索哪个文档进行任何直接监督。给定一组输入/输出对的微调训练语料库 $( x _ { j } , y _ { j } )$，我们最小化每个目标的负边际对数似然 $\Sigma _ { j } - \log p ( y _ { j } | x _ { j } )$，采用随机梯度下降法与 Adam 优化器 [28]。在训练过程中更新文档编码器 $\mathrm { B E R T } _ { d }$ 是代价高昂的，因为它需要定期更新文档索引，如 REALM 在预训练期间所做的那样 [20]。我们认为这一步对强性能并不是必要的，因此保持文档编码器（和索引）不变，仅微调查询编码器 $\mathsf { B E R T } _ { q }$ 和 BART 生成器。

# 2.5 解码

在测试阶段，RAG-Sequence 和 RAG-Token 需要不同的方法来近似 $\operatorname*{arg \ max} _ { y } p ( y | x )$。

RAG-Token模型可以看作是一个标准的自回归序列到序列生成器，其转移概率为：$\begin{array} { r } { p _ { \theta } ^ { \prime } ( y _ { i } | x , y _ { 1 : i - 1 } ) = \sum _ { z \in \mathrm { t o p } \cdot k ( p ( \cdot | x ) ) } p _ { \eta } ( z _ { i } | \overline { { x } } ) p _ { \theta } ( y _ { i } | x , \widehat { z } _ { i } , \chi _ { 1 : i - 1 } ^ { \sim } ) } \end{array}$ 为了解码，我们可以将 $p _ { \theta } ^ { \prime } ( y _ { i } | x , y _ { 1 : i - 1 } )$ 插入到标准束搜索解码器中。

RAG-Sequence 对于 RAG-Sequence，似然 $p ( y | x )$ 无法分解为传统的逐词似然，因此我们无法通过单次束搜索来解决它。相反，我们对每个文档 $z$ 进行束搜索，使用 $p _ { \theta } ( y _ { i } | x , z , y _ { 1 : i - 1 } )$ 对每个假设进行评分。这产生了一组假设 $Y$，其中一些可能并未出现在所有文档的束中。为了估计假设 $y$ 的概率，我们对每个未出现在束中的文档 $z$ 进行额外的前向传播，将生成器的概率与 $p _ { \eta } ( z | x )$ 相乘，然后对边际概率在束中求和。我们将这一解码过程称为“彻底解码”。对于更长的输出序列，$| Y |$ 可能会变得很大，导致需要进行多次前向传播。为实现更高效的解码，我们可以做进一步的近似，即 $\dot { p } _ { \theta } ( y | \dot { x } , z _ { i } ) \dot { \approx } 0$，其中 $y$ 并未在从 $x , z _ { i }$ 进行束搜索时生成。这避免了一旦候选集 $Y$ 生成后需要进行额外前向传播的需求。我们将这一解码过程称为“快速解码”。

# 3 实验

我们在广泛的知识密集型任务中实验了 RAG。在所有实验中，我们使用单个维基百科数据集作为我们的非参数知识源。按照 Lee 等人 [31] 和 Karpukhin 等人 [26] 的方法，我们使用 2018年12月的数据集。每篇维基百科文章被分割成不重叠的 100 字块，总共生成 2100 万个文档。我们使用文档编码器计算每个文档的嵌入，并使用 FAISS [23] 构建一个单一的 MIPS 索引，该索引采用了层次可导航的小世界近似算法以实现快速检索 [37]。在训练过程中，我们为每个查询检索前 $k$ 个文档。我们考虑 $k \in \{ 5 , 10 \}$ 用于训练，并使用开发数据设定测试时的 $k$。接下来，我们讨论每个任务的实验细节。

# 3.1 开放域问答

开放域问答（QA）是一项重要的现实应用，也是知识密集型任务的常用测试平台。我们将问题和答案视为输入-输出文本对 $( x , y )$，并通过直接最小化答案的负对数似然来训练 RAG。我们将 RAG 与流行的抽取式 QA 模式进行比较，在这一模式中，答案是从检索文档中提取的文本片段，主要依赖于非参数知识。我们还与“闭卷问答”方法进行比较，这些方法与 RAG 类似，都生成答案，但不利用检索，而是纯粹依赖于参数化知识。我们考虑四个流行的开放域 QA 数据集：自然问题（NQ）、TriviaQA（TQA）、WebQuestions（WQ）和 CuratedTrec（CT）。由于 CT 和 WQ 较小，我们遵循 DPR 的方法，通过使用我们的 NQ RAG 模型来初始化 CT 和 WQ 模型。我们使用与之前的研究相同的训练/开发/测试划分，并报告精确匹配（EM）分数。对于 TQA，为了与 T5 进行比较，我们还在 TQA Wiki 测试集上进行了评估。

# 3.2 抽象问答

RAG 模型可以超越简单的抽取式问答，使用自由形式的抽象文本生成来回答问题。为了测试 RAG 在知识密集型环境下的自然语言生成（NLG），我们使用 MSMARCO NLG 任务 v2.1 [43]。该任务由问题、从搜索引擎为每个问题检索的十个金标准段落以及从检索到的段落中标注的完整句子答案组成。我们不使用提供的段落，仅使用问题和答案，将 MSMARCO 视为开放域的抽象问答任务。MSMARCO 中有一些问题无法在没有访问金标准段落的情况下以与参考答案相匹配的方式回答，例如“加利福尼亚火山的天气如何？”因此，不使用金标准段落时性能会较低。我们还注意到，有些 MSMARCO 问题仅使用维基百科无法回答。在这方面，RAG 可以依靠参数知识生成合理的响应。

# 3.3 难题生成

为了评估RAG在非问答设置中的生成能力，我们研究开放领域的问题生成。与标准开放领域问答任务中通常包含的简短简单问题不同，我们提出了更具挑战性的生成《危险边缘》问题的任务。《危险边缘》是一种独特的格式，旨在根据关于某个实体的事实来猜测该实体。例如，“世界杯”是“1986年墨西哥作为第二个举办这个国际体育比赛的国家而进球”的问题的答案。由于《危险边缘》问题是精确的事实陈述，因此基于答案实体生成《危险边缘》问题构成了一项具有挑战性的知识密集型生成任务。

我们使用来自 SearchQA 的数据拆分，包括 100K 训练集、14K 开发集和 27K 测试集样本。鉴于这是一个新任务，我们训练了一个 BART 模型进行比较。按照 [67] 的方法，我们使用 SQuAD 调优的 Q-BLEU-1 评分指标进行评估 [42]。Q-BLEU 是 BLEU 的一种变体，它对匹配实体赋予更高的权重，并且与人的评判在问题生成方面的相关性高于标准指标。我们还进行了两次人工评估，一次用于评估生成内容的真实性，另一次用于特异性。我们将真实性定义为一个陈述是否可以被可信的外部来源证实，而特异性则定义为输入与输出之间的高度相互依赖 [33]。我们遵循最佳实践，采用成对比较评估 [34]。评估者会看到一个答案和两个生成的问题，一个来自 BART，一个来自 RAG。随后，他们被要求选择四个选项之一——问题 A 更好、问题 B 更好、两个都不错，或者两个都不好。

# 3.4 事实验证

FEVER 要求对自然语言声明进行分类，以确定该声明是否被维基百科支持或反驳，或者是否信息不足以做出决定。该任务需要从维基百科中检索与声明相关的证据，然后基于这些证据进行推理，以判断声明是否仅通过维基百科能够验证其真实性、虚假性或不可验证性。FEVER 是一个检索问题，结合了具有挑战性的蕴涵推理任务。它还为探索 RAG 模型处理分类而非生成的能力提供了适当的测试平台。我们将 FEVER 的类别标签（支持、反驳或信息不足）映射到单一输出词元，并直接使用声明-类别对进行训练。关键的是，与大多数其他 FEVER 方法不同，我们不对检索到的证据使用监督。在许多实际应用中，检索监督信号并不可用，而不需要这种监督的模型将适用于更广泛的任务。我们探索了两种变体：标准的三分类任务（支持/反驳/信息不足）和 Thorne 和 Vlachos 研究的二分类任务（支持/反驳）。在这两种情况下，我们报告标签准确率。

# 4 结果

# 4.1 开放域问答

表1展示了RAG与最先进模型的结果。在所有四个开放域问答任务中，RAG创造了新的最先进记录（仅在TQA的T5可比拆分上）。RAG结合了“闭卷”（仅限参数）方法的生成灵活性和“开卷”基于检索的方法的性能。与REALM和$\mathrm { T } 5 { + } \mathrm { S S M }$不同，RAG在没有昂贵的专业“显著跨度掩码”预训练的情况下也能取得优异结果[20]。值得注意的是，RAG的检索器使用DPR的检索器进行初始化，后者在Natural Questions和TriviaQA上使用检索监督。RAG与DPR问答系统相比表现良好，后者使用基于BERT的“交叉编码器”进行文档的重新排序，并配有一个抽取式阅读器。RAG证明了在实现最先进性能时，既不需要重新排序器也不需要抽取式阅读器。即使在可以提取答案的情况下，生成答案也有几个优势。文档中包含答案线索但并未逐字包含答案的情况，仍然可以有助于生成正确答案，而这在标准的抽取式方法中是不可能的，从而导致对文档的更有效边际化。此外，RAG能够在没有任何检索文档中找到正确答案的情况下生成正确答案，对于NQ这样的案例，其准确率达到$11.8 \%$，而抽取式模型的得分为$0 \%$。

Table 1: Open-Domain QA Test Scores. For TQA, left column uses the standard test set for OpenDomain QA, right column uses the TQA-Wiki test set. See Appendix D for further details.   

<table><tr><td>Model</td><td></td><td>NQ TQA</td><td>WQ</td><td>CT</td></tr><tr><td>Closed Book</td><td>T5-11B [52] T5-11B+SSM[52]</td><td>34.5 36.6</td><td>- /50.1 - /60.5</td><td>37.4 - 44.7 -</td></tr><tr><td>Open</td><td>REALM [20]</td><td>40.4 -/-</td><td>40.7</td><td>46.8</td></tr><tr><td>Book</td><td>DPR [26]</td><td>41.5</td><td>57.9/ -</td><td>41.1 50.6</td></tr><tr><td></td><td>RAG-Token</td><td>44.1</td><td>55.2/66.1</td><td>45.5 50.0</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>RAG-Seq.</td><td>44.5</td><td>56.8/68.0</td><td>45.2 52.2</td></tr></table>

Table 2: Generation and classification Test Scores. MS-MARCO SotA is [4], FEVER-3 is [68] and FEVER-2 is [57] \*Uses gold context/evidence. Best model without gold access underlined.   

<table><tr><td rowspan="2">Model</td><td colspan="2">Jeopardy B-1QB-1</td><td colspan="2">MSMARCO</td><td colspan="2">FVR3 FVR2 Label Acc.</td></tr><tr><td></td><td></td><td>R-L 49.8*</td><td>B-1 49.9*</td><td>76.8</td><td>92.2*</td></tr><tr><td>SotA BART</td><td>- 15.1</td><td>- 19.7</td><td>38.2</td><td>41.6</td><td>64.0</td><td>81.1</td></tr><tr><td>RAG-Tok. RAG-Seq.</td><td>17.3 14.7</td><td>22.2 21.4</td><td>40.1 40.8</td><td>41.5 44.2</td><td>72.5</td><td>89.5</td></tr></table>

# 4.2 抽象式问答

如表2所示，RAG-Sequence在Open MS-MARCO NLG上比BART高出2.6个Bleu分和2.6个Rouge-L分。RAG方法接近最先进的模型性能，这一成就令人印象深刻，因为（i）这些模型访问了包含生成参考答案所需特定信息的黄金段落，（ii）许多问题如果没有黄金段落无法回答，以及（iii）并非所有问题都能仅从维基百科得到回答。表3展示了我们模型生成的一些答案。在质性分析中，我们发现RAG模型的幻想现象较少，生成事实正确文本的频率高于BART。随后，我们还将展示RAG生成的文本相比于BART生成的文本在多样性上更高（见$\ S 4 . 5 $）。

# 4.3 危险游戏问题生成

表2显示，RAG-Token在Jeopardy问题生成上优于RAG-Sequence，且两个模型在Q-BLEU-1上均优于BART。表4展示了对来自BART和RAG-Token的452对生成结果的人类评估结果。评估者指出，BART在仅$7.1\%$的情况下比RAG更具事实性，而RAG在$42.7\%$的情况下更具事实性，且RAG和BART在另外$17\%$的情况下均具有事实性，清晰地展示了RAG在该任务中的有效性，超过了最先进的生成模型。评估者还发现RAG生成的结果在具体性上有较大优势。表3展示了每个模型的典型生成结果。

Jeopardy 问题通常包含两部分信息，而 RAG-Token 可能表现最佳，因为它可以生成结合多个文档内容的响应。图 2 显示了一个例子。当生成“Sun”时，文档 2 的后验概率较高，因为它提到“太阳照常升起”。类似地，当生成“A Farewell to Arms”时，文档 1 的后验概率占主导地位。值得注意的是，当每本书的第一个词元生成后，文档后验概率趋于平坦。这一观察表明，生成器能够在不依赖特定文档的情况下完成标题。换句话说，模型的参数知识足以完成这些标题。我们通过将部分解码“太阳”输入仅使用 BART 基线来找到这一假设的证据。BART 完成生成：“太阳照常升起”是这位作者的小说，“太阳照常升起”的标题存储在 BART 的参数中。类似地，BART 将完成部分解码“太阳照常升起”是这位作者的小说，后面接“A Farewell to Arms”。这个例子展示了参数性和非参数性记忆如何协同工作——非参数性组件有助于指导生成，提取存储在参数性记忆中的特定知识。

# 4.4 事实核查

表2展示了我们在FEVER上的结果。对于三类分类，RAG的得分与最先进模型的差距在$4.3\%$以内，而这些模型是复杂的管道系统，具有特定领域的架构和大量工程设计，并且使用了中间检索监督，而RAG并不需要这些。文档1：他的作品被认为是美国文学的经典……他的战时经历构成了他小说《永别了，武器》（1929）的基础……文档2：……1920年代“迷失一代”侨民社区的艺术家。他的处女作《太阳照常升起》于1926年出版。

![](images/2.jpg)  

Figure 2: RAG-Token document posterior $p ( z _ { i } | x , y _ { i } , y _ { - i } )$ for each generated token for input "Hemingway" for Jeopardy generation with 5 retrieved documents. The posterior for document 1 is high when generating "A Farewell to Arms" and for document 2 when generating "The Sun Also Rises".

Table 3: Examples from generation tasks. RAG models generate more specific and factually accurate responses. ?' indicates factually incorrect responses, \* indicates partially correct responses.   

<table><tr><td>Task</td><td>Input</td><td>Model</td><td>Generation</td></tr><tr><td rowspan="4">MS- MARCO</td><td rowspan="2">define middle ear</td><td>BART RAG-T</td><td>?The middle ear is the part of the ear between the middle ear and the nose. The middle ear is the portion of the ear internal to the eardrum.</td></tr><tr><td>RAG-S</td><td>The middle ear includes the tympanic cavity and the three ossicles.</td></tr><tr><td rowspan="2">what currency needed in</td><td>BART</td><td></td></tr><tr><td>RAG-T</td><td>The currency needed in Scotland is Pound sterling. Pound is the currency needed in Scotland.</td></tr><tr><td rowspan="4">Jeopardy Question</td><td rowspan="2">scotland Washington</td><td>RAG-S</td><td>The currency needed in Scotland is the pound sterling.</td></tr><tr><td>BART</td><td>?This state has the largest number of counties in the U.S.</td></tr><tr><td rowspan="2"></td><td>RAG-T RAG-S</td><td>It&#x27;s the only U.S. state named for a U.S. president It&#x27;s the state where you&#x27;ll find Mount Rainier National Park</td></tr><tr><td></td><td></td></tr><tr><td rowspan="4">Gener -ation</td><td rowspan="2">The Divine Comedy</td><td>BART</td><td>*This epic poem by Dante is divided into 3 parts: the Inferno, the Purgatorio &amp; the Purgatorio</td></tr><tr><td>RAG-T</td><td>Dante&#x27;s &quot;Inferno&quot; is the first part of this epic poem</td></tr><tr><td></td><td></td><td>This 14th century work is divided into 3 sections: &quot;Inferno&quot;, &quot;Purgatorio&quot; &amp; &quot;Paradiso&quot;</td></tr><tr><td></td><td>RAG-S</td><td></td></tr></table>

对于二分类任务，我们与 Thorne 和 Vlachos [57] 进行比较，他们训练 RoBERTa [35] 将声明分类为真或假，前提是提供金标准的证据句。尽管仅提供声明并检索自身的证据，RAG 的准确率仍在该模型的 $2.7\%$ 以内。我们还分析 RAG 检索的文档是否与 FEVER 中标注为金证据的文档相对应。我们计算 RAG 检索的前 $k$ 篇文档与金证据标注之间的文章标题重叠情况。我们发现，在 $71\%$ 的情况下，检索到的顶级文档来自金文章，而在 $90\%$ 的情况下，前 10 篇检索到的文章中有金文章。

# 4.5 附加结果

生成多样性 第4.3节显示，RAG模型在生成《 jeopardy 》问答时比BART更具事实性和具体性。根据近期关于促进多样性解码的研究，我们还通过计算不同模型生成的独特n-gram与总n-gram的比例来调查生成多样性。表5显示，RAG-Sequence的生成结果比RAG-Token更具多样性，两者的多样性显著高于BART，且无需任何促进多样性的解码。检索消融 RAG的一个关键特性是学习为任务检索相关信息。为了评估检索机制的有效性，我们进行消融实验，在训练期间冻结检索器。如表6所示，学习到的检索对所有任务的结果都有所提升。我们将RAG的稠密检索器与基于单词重叠的BM25检索器进行比较。在这里，我们用固定的BM25系统替换RAG的检索器，在计算$p ( z | x )$时使用BM25的检索得分作为逻辑值。表6显示了结果。对于FEVER，BM25表现最佳，可能是因为FEVER声明高度以实体为中心，因此非常适合基于单词重叠的检索。可微分检索在其他所有任务上都提高了结果，特别是在开放域问答中，这一点至关重要。

索引热插拔 非参数记忆模型如 RAG 的一个优势是知识可以在测试时轻松更新。仅使用参数的模型如 T5 或 BART 需要进一步训练来更新其随世界变化而改变的行为。为了证明这一点，我们使用 2016 年 12 月的 DrQA [5] 维基百科数据集构建了一个索引，并将使用该索引的 RAG 输出与我们主要结果中的更新索引（2018 年 12 月）进行比较。我们准备了一份82位在这些日期之间发生变化的世界领导者的名单，并使用模板“谁是 {职位}？”（例如“谁是秘鲁总统？”）向我们的 NQ RAG 模型查询每个索引。对于 2016 年的世界领导者，RAG 使用 2016 年索引的正确率为 $70 \%$，使用 2018 年索引的正确率为 $68 \%$。使用不匹配的索引时，准确率较低：2018 年索引与 2016 年领导者的准确率为 $12 \%$，2016 年索引与 2018 年领导者的准确率为 $4 \%$。这表明我们可以通过简单更换其非参数记忆来更新 RAG 的世界知识。

Table 4: Human assessments for the Jeopardy Question Generation Task.   

<table><tr><td colspan="2">Factuality</td><td>Specificity</td></tr><tr><td>BART better</td><td>7.1%</td><td>16.8%</td></tr><tr><td>RAG better</td><td>42.7 %</td><td>37.4%</td></tr><tr><td>Both good</td><td>11.7%</td><td>11.8%</td></tr><tr><td>Both poor</td><td>17.7%</td><td>6.9%</td></tr><tr><td>No majority</td><td>20.8%</td><td>20.1%</td></tr></table>

Table 5: Ratio of distinct to total tri-grams for generation tasks.   

<table><tr><td></td><td>MSMARCO</td><td>Jeopardy QGen</td></tr><tr><td>Gold</td><td>89.6%</td><td>90.0%</td></tr><tr><td>BART</td><td>70.7%</td><td>32.4%</td></tr><tr><td>RAG-Token</td><td>77.8%</td><td>46.8%</td></tr><tr><td>RAG-Seq.</td><td>83.5%</td><td>53.8%</td></tr></table>

Table 6: Ablations on the dev set. As FEVER is a classification task, both RAG models are equivalent.   

<table><tr><td>Model</td><td>NQ</td><td>TQA</td><td>WQ</td><td>CT</td><td colspan="2">Jeopardy-QGen</td><td colspan="2">MSMarco</td><td colspan="2">FVR-3 FVR-2 Label Accuracy</td></tr><tr><td></td><td></td><td>Exact Match</td><td></td><td></td><td>B-1</td><td>QB-1</td><td>R-L</td><td>B-1</td><td></td><td></td></tr><tr><td>RAG-Token-BM25 RAG-Sequence-BM25</td><td>29.7</td><td>41.5</td><td>32.1</td><td>33.1</td><td>17.5</td><td>22.3</td><td>55.5</td><td>48.4</td><td rowspan="2">75.1</td><td rowspan="2">91.6</td></tr><tr><td></td><td>31.8</td><td>44.1</td><td>36.6</td><td>33.8</td><td>11.1</td><td>19.5</td><td>56.5</td><td>46.9</td></tr><tr><td>RAG-Token-Frozen</td><td>37.8</td><td>50.1</td><td>37.1</td><td>51.1</td><td>16.7</td><td>21.7</td><td>55.9</td><td>49.4</td><td rowspan="2">72.9</td><td rowspan="2">89.4</td></tr><tr><td>RAG-Sequence-Frozen</td><td>41.2</td><td>52.1</td><td>41.8</td><td>52.6</td><td>11.8</td><td>19.6</td><td>56.7</td><td>47.3</td></tr><tr><td>RAG-Token</td><td>43.5</td><td>54.8</td><td>46.5</td><td>51.9</td><td>17.9</td><td>22.6</td><td>56.2</td><td>49.4</td><td rowspan="2">74.5</td><td rowspan="2">90.6</td></tr><tr><td>RAG-Sequence</td><td>44.0</td><td>55.8</td><td>44.9</td><td>53.4</td><td>15.3</td><td>21.5</td><td>57.2</td><td>47.5</td></tr></table>

检索更多文档的影响 模型分别使用5个或10个检索到的潜在文档进行训练，两者在性能上没有显著差异。我们在测试时可以灵活调整检索文档的数量，这可能会影响性能和运行时间。图3（左）显示，在测试时检索更多文档单调提高了RAG-Sequence的开放域问答结果，但RAG-Token的性能在10个检索文档时达到峰值。图3（右）显示，检索更多文档导致RAG-Token的Rouge-L提高，但以降低Bleu-1为代价，而对于RAG-Sequence，这一效果不那么明显。

![](images/3.jpg)  

Figure 3: Left: NQ performance as more documents are retrieved. Center: Retrieval recall performance in NQ. Right: MS-MARCO Bleu-1 and Rouge-L as more documents are retrieved.

# 5 相关工作

单任务检索 先前的研究表明，检索在不少 NLP 任务中有效提升性能，尤其是当其被孤立考虑时。这些任务包括开放域问答、事实核查、事实补全、长篇问答、维基百科文章生成、对话、翻译和语言建模等。我们的工作统一了将检索整合到各个任务中的成功经验，展示了一个单一的基于检索的架构能够在多个任务中实现强的性能。 NLP 任务通用架构 先前关于 NLP 任务通用架构的研究在不使用检索的情况下取得了巨大成功。单一的预训练语言模型在经过微调后，已经在 GLUE 基准上实现了各种分类任务的优异表现。GPT-2 后来展示了一个单一的、从左到右的预训练语言模型能够在判别和生成任务上都取得强的性能。为了进一步提升，BART 和 T5 提出了一个单一的预训练编码器-解码器模型，利用双向注意力在判别和生成任务上获得更好的性能。我们的工作旨在通过学习一个检索模块来增强预训练的生成语言模型，从而扩大可以使用单一统一架构的任务范围。 学习检索 在信息检索中，学习检索文档的工作非常丰富，最近也有类似于我们研究的预训练神经语言模型的研究。一些工作优化了检索模块，以帮助特定的下游任务，例如问答，使用搜索、强化学习或潜变量方法。与之不同的是，我们的研究展示了一个单一的基于检索的架构可以经过微调来适应多样的任务，并获得良好的性能。 基于内存的架构 我们的文档索引可以视为神经网络检索的大型外部内存，类似于记忆网络。相关工作学习为输入中的每个实体检索一个训练好的嵌入，而不是像我们的方法那样检索原始文本。其他研究则通过对事实嵌入的关注提升对话模型生成事实文本的能力。我们记忆的一个关键特征是它由原始文本组成，而不是分布式表示，这使得内存既具有人类可读性，提供了一种可解释性，又具有人类可写性，使我们能够通过编辑文档索引动态更新模型的内存。这种方法在知识密集型对话中也得到了应用，其中生成器直接条件于检索到的文本，尽管是通过 TF-IDF 获取，而非端到端学习的检索。 检索与编辑方法 我们的方法与检索与编辑风格的方法有一些相似之处，这些方法为给定输入检索类似的训练输入-输出对，然后进行编辑以提供最终输出。这些方法在多个领域中取得了成功，包括机器翻译和语义解析。我们的做法有几个区别，重点不在于对检索到的项目进行轻微编辑，而是在多个检索内容中聚合信息，学习潜在检索，以及检索证据文档而非相关的训练对。尽管如此，RAG 技术在这些设置中可能表现良好，并可能代表值得关注的未来工作。

# 6 讨论

在本研究中，我们提出了具有参数化和非参数化记忆的混合生成模型。我们展示了我们的RAG模型在开放领域问答中获得了最先进的结果。我们发现，人们更倾向于选择RAG的生成结果，而不是纯参数化的BART，因为RAG被认为更加事实性和具体。我们对学习到的检索组件进行了深入调查，验证了其有效性，并说明了检索索引如何能够热插拔，以便在不需要任何重新训练的情况下更新模型。在未来的工作中，探讨这两个组件是否可以共同从头预训练可能会是有益的，预训练的目标可以类似于BART的去噪目标或其他目标。我们的工作为参数化和非参数化记忆如何相互作用以及如何最有效地将它们结合打开了新的研究方向，展示了在广泛的自然语言处理任务中应用的潜力。

# 更广泛的影响

这项工作相比于之前的研究提供了多项积极的社会效益：它更加强烈地基于真实的事实知识（在本例中为维基百科），这使得其生成的内容减少“幻觉”，生成的事实性更强，并提供了更多的控制和可解释性。RAG 可以在多种场景中使用，直接为社会带来好处，例如为其赋予医疗索引，并就该主题提出开放领域的问题，或者帮助人们更加高效地完成工作。但这些优势也带来了潜在的缺点：维基百科或任何潜在的外部知识来源可能永远不会完全真实，也可能完全不受偏见影响。由于 RAG 可以被用作语言模型，因此与 GPT-2 [50] 的类似担忧在这里同样有效，尽管可以说程度较轻，包括可能被用于生成滥用、伪造或误导性的新闻内容或社交媒体上的内容；模仿他人；或自动生成垃圾邮件/钓鱼内容 [54]。先进的语言模型也可能在未来几十年内导致各种工作的自动化 [16]。为了降低这些风险，可以采用人工智能系统来对抗误导性内容和自动化的垃圾邮件/钓鱼。

# 致谢

作者感谢评审们对本文的深思熟虑和建设性反馈，以及感谢 HuggingFace 在开源运行 RAG 模型代码方面的支持。作者还要感谢 Kyunghyun Cho 和 Sewon Min 进行的富有成效的讨论和建议。EP 感谢 NSF 研究生奖学金的支持。PL 得到了 FAIR 博士生项目的资助。

# References

[1] Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang. MS MARCO: A Human Generated MAchine Reading COmprehension Dataset. arXiv:1611.09268 [cs], November 2016. URL http: //arxiv.org/abs/1611.09268. arXiv: 1611.09268.   
[2] Petr Baudi and Jan edivy. Modeling of the question answering task in the yodaqa system. In International Conference of the Cross-Language Evaluation Forum for European Languages, pages 222228. Springer, 2015. URL https://1ink.springer.com/chapter/10.1007% 2F978-3-319-24027-5_20.   
[3] Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. Semantic Parsing on Freebase from Question-Answer Pairs. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 15331544, Seattle, Washington, USA, October 2013. Association for Computational Linguistics. URL http://www.aclweb. org/anthology/ D13-1160.   
[4] Bin Bi, Chenliang Li, Chen Wu, Ming Yan, and Wei Wang. Palm: Pre-training an autoencoding&autoregressive language model for context-conditioned generation. ArXiv, abs/2004.07159, 2020. URL https://arxiv.org/abs/2004.07159.   
[5] Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. Reading Wikipedia to Answer Open-Domain Questions. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 18701879, Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1171. URL https://www.aclweb.org/anthology/P17-1171.   
[6] Eunsol Choi, Daniel Hewlett, Jakob Uszkoreit, Illia Polosukhin, Alexandre Lacoste, and Jonathan Berant. Coarse-to-fine question answering for long documents. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 209220, Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1020. URL https://www.aclweb.org/anthology/P17-1020.

[7] Christopher Clark and Matt Gardner. Simple and Effective Multi-Paragraph Reading Comprehension. arXiv:1710.10723 [cs], October 2017. URL http: //arxiv . org/abs/1710.10723. arXiv: 1710.10723.

[8] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 41714186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URLhttps://www.aclweb.org/anthology/N19-1423.

[9] Emily Dinan, Stephen Roller, Kurt Shuster, Angela Fan, Michael Auli, and Jason Weston. Wizard of wikipedia: Knowledge-powered conversational agents. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id $\underset { . } { = }$ r1173iRqKm.

[10] Matthew Dunn, Levent Sagun, Mike Higgins, V. Ugur Guney, Volkan Cirik, and Kyunghyun Cho. SearchQA: A New Q&A Dataset Augmented with Context from a Search Engine. arXiv:1704.05179 [cs], April 2017. URL http://arxiv.org/abs/1704.05179. arXiv: 1704.05179.

[11] Angela Fan, Mike Lewis, and Yann Dauphin. Hierarchical neural story generation. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 889898, Melbourne, Australia, July 2018. Association for Computational Linguistics. doi: 10.18653/v1/P18-1082. URL https://www.aclweb.org/anthology/ P18-1082.

[12] Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. ELI5: Long form question answering. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 35583567, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1346. URL https : //www .aclweb . org/ anthology/P19-1346.

[13] Angela Fan, Claire Gardent, Chloe Braud, and Antoine Bordes. Augmenting transformers with KNN-based composite memory, 2020. URL https://openreview.net/forum?id= H1gx1CNKPH.

[14] Thibault Févry, Livio Baldini Soares, Nicholas FitzGerald, Eunsol Choi, and Tom Kwiatkowski. Entities as experts: Sparse memory access with entity supervision. ArXiv, abs/2004.07202, 2020. URL https://arxiv.org/abs/2004.07202.

[15] Marjan Ghazvininejad, Chris Brockett, Ming-Wei Chang, Bill Dolan, Jianfeng Gao, Wen tau Yih, and Michel Galley. A knowledge-grounded neural conversation model. In AAAI Conference on Artificial Intelligence, 2018. URL https: / /www .aaai . org/ocs/index . php/ AAAI/AAAI18/paper/view/16710.

[16] Katja Grace, John Salvatier, Allan Dafoe, Baobao Zhang, and Owain Evans. When will AI exceed human performance? evidence from AI experts. CoRR, abs/1705.08807, 2017. URL http://arxiv.org/abs/1705.08807.

[17] Jiatao Gu, Yong Wang, Kyunghyun Cho, and Victor O.K. Li. Search engine guided neural machine translation. In AAAI Conference on Artificial Intelligence, 2018. URL https: //www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17282.

[18] Jiatao Gu, Yong Wang, Kyunghyun Cho, and Victor O.K. Li. Search engine guided neural machine translation. In 32nd AAAI Conference on Artificial Intelligence, AAAI 2018, 32nd AAAI Conference on Artificial Intelligence, AAAI 2018, pages 51335140. AAAI press, 2018. 32nd AAAI Conference on Artificial Intelligence, AAAI 2018 ; Conference date: 02-02-2018 Through 07-02-2018.

[19] Kelvin Guu, Tatsunori B. Hashimoto, Yonatan Oren, and Percy Liang. Generating sentences by editing prototypes. Transactions of the Association for Computational Linguistics, 6:437450, 2018. doi: 10.1162/tacl_a_00030. URL https://www.aclweb.org/anthology/Q18-1031.

[20] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. REALM: Retrieval-augmented language model pre-training. ArXiv, abs/2002.08909, 2020. URL https : //arxiv.org/abs/2002.08909.

[21] Tatsunori B Hashimoto, Kelvin Guu, Yonatan Oren, and Percy S Liang. A retrieve-and-edit framework for predicting structured outputs. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems 31, pages 10052- 10062. Curran Associates, Inc., 2018. URL http://papers.nips.cc/paper/ 8209-a-retrieve-and-edit-framework-for-predicting-structured-outputs. pdf.

[22] Nabil Hossain, Marjan Ghazvininejad, and Luke Zettlemoyer. Simple and effective retrieveedit-rerank text generation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 25322538, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.228. URL https : //www . aclweb . org/ anthology/2020.acl-main.228.

[23] Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with gpus. arXiv preprint arXiv:1702.08734, 2017. URL https://arxiv.org/abs/1702.08734.

[24] Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 16011611, Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1147. URL https://www.aclweb.org/anthology/P17-1147.

[25] Armand Joulin and Tomas Mikolov. Inferring algorithmic patterns with stackaugmented recurrent nets. In Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 1, NIPS'15, page 190-198, Cambridge, MA, USA, 2015. MIT Press. URL https://papers.nips.cc/paper/ 5857-inferring-algorithmic-patterns-with-stack-augmented-recurrent-nets.

[26] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. arXiv preprint arXiv:2004.04906, 2020. URL https://arxiv. org/abs/2004.04906.

[27] Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization through memorization: Nearest neighbor language models. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id $\ c =$ HklBjCEKvH.

[28] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Yoshua Bengio and Yann LeCun, editors, 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015. URL http://arxiv.org/abs/1412.6980.

[29] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Matthew Kelcey, Jacob Devlin, Kenton Lee, Kristina N. Toutanova, Llion Jones, Ming-Wei Chang, Andrew Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural Questions: a Benchmark for Question Answering Research. Transactions of the Association of Computational Linguistics, 2019. URL https://tomkwiat.users.x20web.corp.google.com/papers/ natural-questions/main-1455-kwiatkowski.pdf.

[30] Guillaume Lample, Alexandre Sablayrolles, Marc' Aurelio Ranzato, Ludovic Denoyer, and Herve Jegou. Large memory layers with product keys. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d' Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems 32, pages 85488559. Curran Associates, Inc., 2019. URL http: //papers.nips.cc/paper/9061-large-memory-layers-with-product-keys.pdf.

[31] Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. Latent retrieval for weakly supervised open domain question answering. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 60866096, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1612. URL https : //www. aclweb. org/ anthology/P19-1612.

[32] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. arXiv preprint arXiv:1910.13461, 2019. URL https://arxiv.org/abs/1910.13461.

[33] Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, and Bill Dolan. A diversity-promoting objective function for neural conversation models. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 110119, San Diego, California, June 2016. Association for Computational Linguistics. doi: 10.18653/v1/N16-1014. URL https://www.aclweb.org/anthology/ N16-1014.

[34] Margaret Li, Jason Weston, and Stephen Roller. Acute-eval: Improved dialogue evaluation with optimized questions and multi-turn comparisons. ArXiv, abs/1909.03087, 2019. URL https://arxiv.org/abs/1909.03087.

[35] Hairong Liu, Mingbo Ma, Liang Huang, Hao Xiong, and Zhongjun He. Robust neural machine translation with joint textual and phonetic embedding. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 30443049, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1291. URL https://www.aclweb.org/anthology/P19-1291.

[36] Peter J. Liu\*, Mohammad Saleh\*, Etienne Pot, Ben Goodrich, Ryan Sepassi, Lukasz Kaiser, and Noam Shazeer. Generating wikipedia by summarizing long sequences. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum? id=HygOvbWC-.

[37] Yury A. Malkov and D. A. Yashunin. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE Transactions on Pattern Analysis and Machine Intelligence, 42:824836, 2016. URL https : //arxiv . org/abs/1603. 09320.

[38] Gary Marcus. The next decade in ai: four steps towards robust artificial intelligence. arXiv preprint arXiv:2002.06177, 2020. URL https: //arxiv.org/abs/2002.06177.

[39] Luca Massarelli, Fabio Petroni, Aleksandra Piktus, Myle Ott, Tim Rocktäschel, Vassilis Plachouras, Fabrizio Silvestri, and Sebastian Riedel. How decoding strategies affect the verifiability of generated text. arXiv preprint arXiv:1911.03587, 2019. URL https: //arxiv.org/abs/1911.03587.

[40] Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, and Hao Wu. Mixed precision training. In ICLR, 2018. URL https://openreview.net/forum?id $=$ r1gs9JgRZ.

[41] Nikita Moghe, Siddhartha Arora, Suman Banerjee, and Mitesh M. Khapra. Towards exploiting background knowledge for building conversation systems. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 23222332, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1255. URL https://www.aclweb.org/anthology/D18-1255.

[42] Preksha Nema and Mitesh M. Khapra. Towards a better metric for evaluating question generation systems. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 39503959, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1429. URL https : //www. aclweb. org/ anthology/D18-1429.

[43] Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. MS MARCO: A human generated machine reading comprehension dataset. In Tarek Richard Besold, Antoine Bordes, Artur S. d'Avila Garcez, and Greg Wayne, editors, Proceedings of the Workshop on Cognitive Computation: Integrating neural and symbolic approaches 2016 co-located with the 30th Annual Conference on Neural Information Processing Systems (NIPS 2016), Barcelona, Spain, December 9, 2016, volume 1773 of CEUR Workshop Proceedings. CEUR-WS.org, 2016. URL http://ceur-ws.org/Vol-1773/CoCoNIPS_ 2016_paper9.pdf.

[44] Rodrigo Nogueira and Kyunghyun Cho. Passage re-ranking with BERT. arXiv preprint arXiv:1901.04085, 2019. URL https://arxiv.org/abs/1901.04085.

[45] Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. fairseq: A fast, extensible toolkit for sequence modeling. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations), pages 4853, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-4009. URL https : //www . aclweb. org/anthology/N19-4009.

[46] Ethan Perez, Siddharth Karamcheti, Rob Fergus, Jason Weston, Douwe Kiela, and Kyunghyun Cho. Finding generalizable evidence by learning to convince q&a models. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 24022411, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1244. URL https://www.aclweb.org/anthology/D19-1244.

[47] Fabio Petroni, Tim Rocktäschel, Sebastian Riedel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, and Alexander Miller. Language models as knowledge bases? In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 24632473, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/ D19-1250. URL https://www.aclweb.org/anthology/D19-1250.

[48] Fabio Petroni, Patrick Lewis, Aleksandra Piktus, Tim Rocktäschel, Yuxiang Wu, Alexander H. Miller, and Sebastian Riedel. How context affects language models' factual predictions. In Automated Knowledge Base Construction, 2020. URL https://openreview.net/forum? id=025X0zPfn.

[49] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving Language Understanding by Generative Pre-Training, 2018. URL https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/ language-unsupervised/language_understanding_paper.pdf.

[50] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners, 2019. URL https://d4mucfpksywv.cloudfront.net/better-language-models/language_ models_are_unsupervised_multitask_learners.pdf.

[51] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv e-prints, 2019. URL https: //arxiv . org/abs/1910 . 10683.

[52] Adam Roberts, Colin Raffel, and Noam Shazeer. How much knowledge can you pack into the parameters of a language model? arXiv e-prints, 2020. URL https : //arxiv . org/abs/ 2002.08910.

[53] Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: Bm25 and beyond. Found. Trends Inf. Retr., 3(4):333389, April 2009. ISSN 1554-0669. doi: 10.1561/ 1500000019. URL https://doi.org/10.1561/1500000019.

[54] Irene Solaiman, Miles Brundage, Jack Clark, Amanda Askell, Ariel Herbert-Voss, Jeff Wu, Alec Radford, and Jian-Bing Wang. Release strategies and the social impacts of language models. ArXiv, abs/1908.09203, 2019.

[55] Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-to-end memory networks. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems 28, pages 24402448. Curran Associates, Inc., 2015. URLhttp://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf.

[56] James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. FEVER: a large-scale dataset for fact extraction and VERification. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 809819, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. doi: 10.18653/v1/N18-1074. URL https://www.aclweb.org/anthology/N18-1074.

[57] James H. Thorne and Andreas Vlachos. Avoiding catastrophic forgetting in mitigating model biases in sentence-pair classification with elastic weight consolidation. ArXiv, abs/2004.14366, 2020. URL https://arxiv.org/abs/2004.14366.

[58] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems 30, pages 59986008. Curran Associates, Inc., 2017. URL http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf.

[59] Ashwin Vijayakumar, Michael Cogswell, Ramprasaath Selvaraju, Qing Sun, Stefan Lee, David Crandall, and Dhruv Batra. Diverse beam search for improved description of complex scenes. AAAI Conference on Artificial Intelligence, 2018. URL https : / /www .aaai . org/ocs/index. php/AAAI/AAAI18/paper/view/17329.

[60] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pages 353355, Brussels, Belgium, November 2018. Association for Computational Linguistics. doi: 10.18653/v1/W18-5446. URL https : //www . aclweb . org/ anthology/W18-5446.

[61] Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. SuperGLUE: A Stickier Benchmark for GeneralPurpose Language Understanding Systems. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d\textquotesingle Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems 32, pages 32613275. Curran Associates, Inc., 2019. URL https : // arxiv.org/abs/1905.00537.

[62] Shuohang Wang, Mo Yu, Xiaoxiao Guo, Zhiguo Wang, Tim Klinger, Wei Zhang, Shiyu Chang, Gerry Tesauro, Bowen Zhou, and Jing Jiang. $\mathtt { R } ^ { 3 }$ :Reinorced ranker-reader for open-domain question answering. In Sheila A. Mcllraith and Kilian Q. Weinberger, editors, Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications of Artificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018, pages 5981-5988. AAAI Press, 2018. URL https://www.aaai.org/ocs/index. php/AAAI/AAAI18/paper/view/16712.

[63] Shuohang Wang, Mo Yu, Jing Jiang, Wei Zhang, Xiaoxiao Guo, Shiyu Chang, Zhiguo Wang, Tim Klinger, Gerald Tesauro, and Murray Campbell. Evidence aggregation for answer reranking in open-domain question answering. In ICLR, 2018. URL https :/ /openreview. net/forum?id=rJl3yM-Ab.

[64] Jason Weston, Sumit Chopra, and Antoine Bordes. Memory networks. In Yoshua Bengio and Yann LeCun, editors, 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015. URL http://arxiv.org/abs/1410.3916.

[65] Jason Weston, Emily Dinan, and Alexander Miller. Retrieve and refine: Improved sequence generation models for dialogue. In Proceedings of the 2018 EMNLP Workshop SCAI: The 2nd International Workshop on Search-Oriented Conversational AI, pages 8792, Brussels, Belgium, October 2018. Association for Computational Linguistics. doi: 10.18653/v1/W18-5713. URL https://www.aclweb.org/anthology/W18-5713.

[66] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Huggingface's transformers: State-of-the-art natural language processing. ArXiv, abs/1910.03771, 2019.

[67] Shiyue Zhang and Mohit Bansal. Addressing semantic drift in question generation for semisupervised question answering. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJNLP), pages 24952509, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1253. URL https://www.aclweb.org/anthology/D19-1253.

[68] Wanjun Zhong, Jingjing Xu, Duyu Tang, Zenan Xu, Nan Duan, Ming Zhou, Jiahai Wang, and Jian Yin. Reasoning over semantic-level graph for fact checking. ArXiv, abs/1909.03745, 2019. URLhttps://arxiv.org/abs/1909.03745.

# Appendices for Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

# A Implementation Details

For Open-domain QA we report test numbers using 15 retrieved documents for RAG-Token models. For RAG-Sequence models, we report test results using 50 retrieved documents, and we use the Thorough Decoding approach since answers are generally short. We use greedy decoding for QA as we did not find beam search improved results. For Open-MSMarco and Jeopardy question generation, we report test numbers using ten retrieved documents for both RAG-Token and RAG-Sequence, and we also train a BART-large model as a baseline. We use a beam size of four, and use the Fast Decoding approach for RAG-Sequence models, as Thorough Decoding did not improve performance.

# B Human Evaluation

![](images/4.jpg)  
Figure 4:Annotation interface for human evaluation of factuality. A pop-out for detailed instructions and a worked example appear when clicking "view tool guide".

Figure 4 shows the user interface for human evaluation. To avoid any biases for screen position, which model corresponded to sentence A and sentence B was randomly selected for each example. Annotators were encouraged to research the topic using the internet, and were given detailed instructions and worked examples in a full instructions tab. We included some gold sentences in order to assess the accuracy of the annotators. Two annotators did not perform well on these examples and their annotations were removed from the results.

# C Training setup Details

We train all RAG models and BART baselines using Fairseq [45].2 We train with mixed precision floating point arithmetic [40], distributing training across 8, 32GB NVIDIA V100 GPUs, though training and inference can be run on one GPU. We find that doing Maximum Inner Product Search with FAISS is sufficiently fast on CPU, so we store document index vectors on CPU, requiring $\sim 1 0 0$ GB of CPU memory for all of Wikipedia. After submission, We have ported our code to HuggingFace Transformers $[ 6 6 ] ^ { 3 }$ , which achieves equivalent performance to the previous version but is a cleaner and easier to use implementation. This version is also open-sourced. We also compress the document index using FAISS's compression tools, reducing the CPU memory requirement to 36GB. Scripts to run experiments with RAG can be found at https : //github. com/huggingface/transformers/ blob/master/examples/rag/README. md and an interactive demo of a RAG model can be found at https://huggingface.co/rag/

# D Further Details on Open-Domain QA

For open-domain QA, multiple answer annotations are often available for a given question. These answer annotations are exploited by extractive models during training as typically all the answer annotations are used to find matches within documents when preparing training data. For RAG, we also make use of multiple annotation examples for Natural Questions and WebQuestions by training the model with each $( q , a )$ pair separately, leading to a small increase in accuracy. For TriviaQA, there are often many valid answers to a given question, some of which are not suitable training targets, such as emoji or spelling variants. For TriviaQA, we filter out answer candidates if they do not occur in top 1000 documents for the query.

CuratedTrec preprocessing The answers for CuratedTrec are given in the form of regular expressions, which has been suggested as a reason why it is unsuitable for answer-generation models [20]. To overcome this, we use a pre-processing step where we first retrieve the top 1000 documents for each query, and use the answer that most frequently matches the regex pattern as the supervision target. If no matches are found, we resort to a simple heuristic: generate all possible permutations for each regex, replacing non-deterministic symbols in the regex nested tree structure with a whitespace.

TriviaQA Evaluation setups The open-domain QA community customarily uses public development datasets as test datasets, as test data for QA datasets is often restricted and dedicated to reading compehension purposes. We report our results using the datasets splits used in DPR [26], which are consistent with common practice in Open-domain QA. For TriviaQA, this test dataset is the public TriviaQA Web Development split. Roberts et al. [52] used the TriviaQA official Wikipedia test set instead. Févry et al. [14] follow this convention in order to compare with Roberts et al. [52] (See appendix of [14]). We report results on both test sets to enable fair comparison to both approaches. We find that our performance is much higher using the official Wiki test set, rather than the more conventional open-domain test set, which we attribute to the official Wiki test set questions being simpler to answer from Wikipedia.

# E Further Details on FEVER

For FEVER classification, we follow the practice from [32], and first re-generate the claim, and then classify using the representation of the final hidden state, before finally marginalizing across documents to obtain the class probabilities. The FEVER task traditionally has two sub-tasks. The first is to classify the claim as either "Supported", "Refuted" or "Not Enough Info", which is the task we explore in the main paper. FEVER's other sub-task involves extracting sentences from Wikipedia as evidence supporting the classification prediction. As FEVER uses a different Wikipedia dump to us, directly tackling this task is not straightforward. We hope to address this in future work.

# F Null Document Probabilities

We experimented with adding "Null document" mechanism to RAG, similar to REALM [20] in order to model cases where no useful information could be retrieved for a given input. Here, if $k$ documents were retrieved, we would additionally "retrieve" an empty document and predict a logit for the null document, before marginalizing over $k + 1$ predictions. We explored modelling this null document logit by learning (i) a document embedding for the null document, (ii) a static learnt bias term, or (ii) a neural network to predict the logit. We did not find that these improved performance, so in the interests of simplicity, we omit them. For Open MS-MARCO, where useful retrieved documents cannot always be retrieved, we observe that the model learns to always retrieve a particular set of documents for questions that are less likely to benefit from retrieval, suggesting that null document mechanisms may not be necessary for RAG.

# G Parameters

Our RAG models contain the trainable parameters for the BERT-base query and document encoder of DPR, with 110M parameters each (although we do not train the document encoder ourselves) and 406M trainable parameters from BART-large, 406M parameters, making a total of 626M trainable parameters. The best performing "closed-book" (parametric only) open-domain QA model is T5-11B with 11 Billion trainable parameters. The T5 model with the closest number of parameters to our models is T5-large (770M parameters), which achieves a score of $2 8 . 9 \mathrm { E M }$ on Natural Questions [52], substantially below the 44.5 that RAG-Sequence achieves, indicating that hybrid parametric/nonparametric models require far fewer trainable parameters for strong open-domain QA performance. The non-parametric memory index does not consist of trainable parameters, but does consists of 21M 728 dimensional vectors, consisting of 15.3B values. These can be easily be stored at 8-bit floating point precision to manage memory and disk footprints.

Table 7: Number of instances in the datasets used. $^ { * } \mathrm { A }$ hidden subset of this data is used for evaluation   

<table><tr><td>Task</td><td>Train</td><td>Development</td><td>Test</td></tr><tr><td>Natural Questions</td><td>79169</td><td>8758</td><td>3611</td></tr><tr><td>TriviaQA</td><td>78786</td><td>8838</td><td>11314</td></tr><tr><td>WebQuestions</td><td>3418</td><td>362</td><td>2033</td></tr><tr><td>CuratedTrec</td><td>635</td><td>134</td><td>635</td></tr><tr><td>Jeopardy Question Generation</td><td>97392</td><td>13714</td><td>26849</td></tr><tr><td>MS-MARCO</td><td>153726</td><td>12468</td><td>101093*</td></tr><tr><td>FEVER-3-way</td><td>145450</td><td>10000</td><td>10000</td></tr><tr><td>FEVER-2-way</td><td>96966</td><td>6666</td><td>6666</td></tr></table>

# H Retrieval Collapse

In preliminary experiments, we observed that for some tasks such as story generation [11], the retrieval component would "collapse" and learn to retrieve the same documents regardless of the input. In these cases, once retrieval had collapsed, the generator would learn to ignore the documents, and the RAG model would perform equivalently to BART. The collapse could be due to a less-explicit requirement for factual knowledge in some tasks, or the longer target sequences, which could result in less informative gradients for the retriever. Perez et al. [46] also found spurious retrieval results when optimizing a retrieval component in order to improve performance on downstream tasks.

# I Number of instances per dataset

The number of training, development and test datapoints in each of our datasets is shown in Table 7.