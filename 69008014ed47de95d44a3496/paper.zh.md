# DeepSeekMath：推动开放语言模型中的数学推理极限

邵志鸿 $^ { 1 , 2 \ast }$ , 王培怡1,3\*\*, 祝岐浩 $Z \mathrm { { h u } } ^ { 1 , 3 * \dagger }$ , 徐润欣 ${ \tt X } { \tt u } ^ { 1 }$ , 宋俊霄1 , 曹小比 $ \bar { \mathrm { B i } } ^ { 1 }$ , 张浩伟1 , 张明川1 , 李燕凯 $ \mathrm { L i } ^ { 1 }$ , 吴亚1 , 郭大雅1\*

1DeepSeek-AI，清华大学，3北京大学

{zhihongshao,wangpeiyi,zhuqh,guoday}@deepseek.com https://github.com/deepseek-ai/DeepSeek-Math

# 摘要

数学推理对语言模型提出了重大挑战，因为其复杂且结构化的特性。在本文中，我们介绍了DeepSeekMath 7B，它在DeepSeek-Coder-Base-v1.5 7B的基础上进行了预训练，使用了来自Common Crawl的120B与数学相关的标记，以及自然语言和代码数据。DeepSeekMath 7B在竞争级MATH基准测试中取得了令人印象深刻的$5 1 . 7 \%$的成绩，没有依赖外部工具包和投票技术，接近Gemini-Ultra和GPT-4的性能水平。DeepSeekMath 7B在64个样本上的自一致性达到了$6 0 . 9 \%$。DeepSeekMath的数学推理能力归因于两个关键因素：首先，我们通过精心设计的数据选择管道利用了公开可用的网络数据的巨大潜力。其次，我们引入了相对组策略优化（GRPO），这是近端策略优化（PPO）的一个变体，提升了数学推理能力，同时优化了PPO的内存使用。

![](images/1.jpg)  
Figure 1 | Top1 accuracy of open-source models on the competition-level MATH benchmarl (Hendrycks et al., 2021) without the use of exteral toolkits and voting techniques.

# 1. 引言

大型语言模型（LLM）在人工智能的数学推理方法上引发了革命性的变化，推动了定量推理基准（Hendrycks等，2021）和几何推理基准（Trinh等，2024）的显著进展。此外，这些模型在帮助人类解决复杂数学问题方面也发挥了重要作用（Tao，2023）。然而，像GPT-4（OpenAI，2023）和Gemini-Ultra（Anil等，2023）这样的尖端模型并不公开可用，而目前可获取的开源模型在性能上大大滞后。

在本研究中，我们介绍了DeepSeekMath，一种特定领域的语言模型，其数学能力显著优于开源模型，并接近GPT-4在学术基准上的性能水平。为此，我们创建了DeepSeekMath语料库，这是一个包含1200亿数学标记的大规模高质量预训练语料库。该数据集是使用基于fastText的分类器（Joulin等，2016）从Common Crawl (CC)中提取的。在初始迭代中，分类器使用OpenWebMath的实例（Paster等，2023）作为正例进行训练，同时结合多样的其他网页作为负例。随后，我们利用分类器从CC中挖掘更多正例，并通过人工标注进一步精细化。然后，用这个增强的数据集更新分类器以提高其性能。评估结果表明，这个大规模语料库的质量很高，因为我们的基础模型DeepSeekMath-Base 7B在GSM8K（Cobbe等，2021）上达到了$6 4 . 2 \%$，在竞争级MATH数据集（Hendrycks等，2021）上达到了$3 6 . 2 \%$，超越了Minerva 540B（Lewkowycz等，2022a）。此外，DeepSeekMath语料库是多语言的，因此我们注意到在中文数学基准测试（Wei等，2023；Zhong等，2023）中有所改善。我们相信，我们在数学数据处理方面的经验为研究社区提供了一个起点，并在未来还有很大改进空间。

DeepSeekMath-Base 初始化于 DeepSeek-Coder-Base-v1.5 7B（郭等，2024），我们注意到从代码训练模型开始，比起通用的大型语言模型是更好的选择。此外，我们观察到数学训练也提高了模型在 MMLU（Hendrycks 等，2020）和 BBH 基准（Suzgun 等，2022）上的能力，这表明它不仅增强了模型的数学能力，还扩大了整体推理能力。

在预训练之后，我们对DeepSeekMath-Base应用数学指令调优，使用了思维链（Wei et al., 2022）、思维程序（Chen et al., 2022；Ga0 et al., 2023）以及工具集成推理（Gou et al., 2023）数据。最终得到的模型DeepSeekMath-Instruct 7B超越了所有7B同类模型，并且与70B的开源指令调优模型相媲美。

此外，我们引入了群体相对策略优化（GRPO），这是近端策略优化（PPO）的一种变体强化学习（RL）算法（Schulman等，2017）。GRPO放弃了评论员模型，而是从群体分数中估计基线，显著减少了训练资源。通过仅使用一部分英语指令调优数据，GRPO在强大的DeepSeekMath-Instruct上取得了显著提升，包括领域内（GSM8K: $8 2 . 9 \%  8 8 . 2 \%$ 数学: $4 6 . 8 \%  5 1 . 7 \%$）和领域外数学任务（例如，CMATH: $8 4 . 6 \%  8 8 . 8 \%$ 在强化学习阶段）。我们还提供了一个统一的范式来理解不同的方法，例如拒绝采样微调（RFT）（Yuan等，2023a），直接偏好优化（DPO）（Rafailov等，2023），PPO和GRPO。基于这样的统一范式，我们发现所有这些方法都被概念化为直接或简化的RL技术。我们还进行广泛的实验，例如，在线与离线训练，结果与过程监督，单轮与迭代RL等等，以深入研究该范式的基本要素。最后，我们解释了为什么我们的RL提升了指令调优模型的性能，并进一步总结了基于这一统一范式实现更有效RL的潜在方向。

# 1.1. 贡献

我们的贡献包括可扩展的数学预训练，以及对强化学习的探索和分析。

# 大规模数学预训练

我们的研究提供了有力的证据表明，公开可获取的 Common Crawl 数据包含对数学目的有价值的信息。通过实施精心设计的数据选择流程，我们成功构建了 DeepSeekMath 语料库，这是一个高质量的数据集，包含120亿个来自经过筛选的数学内容网页的词元，几乎是 Minerva（Lewkowycz 等，2022a）使用的数学网页大小的7倍，以及最近发布的 OpenWebMath（Paster 等，2023）大小的9倍。

我们的预训练基础模型DeepSeekMath-Base 7B的性能与Minerva 540B（Lewkowycz et al.，2022a）相当，这表明参数数量并不是数学推理能力的唯一关键因素。一个在高质量数据上预训练的较小模型也能够取得强劲的表现。

我们分享了数学训练实验的发现。数学训练前进行编码训练能够提高模型解决数学问题的能力，无论是否使用工具。这为一个长期存在的问题提供了部分答案：编码训练是否能改善推理能力？我们相信，它确实能够，至少对于数学推理是如此。

虽然在arXiv论文上进行训练是很常见的，尤其是在许多与数学相关的论文中，但在本论文采用的所有数学基准测试中并没有带来显著的改善。

# 强化学习的探索与分析

我们引入了一种高效且有效的强化学习算法——群体相对政策优化（GRPO）。GRPO放弃了评估模型，而是通过群体得分来估计基线，相较于近端政策优化（PPO），显著减少了训练资源。

我们证明了GRPO显著提高了我们指令调优模型DeepSeekMath-Instruct的性能，这完全是通过使用指令调优数据。此外，我们还观察到在强化学习过程中，领域外性能得到了改善。

•我们提供一个统一的范式来理解不同的方法，如RFT、DPO、PPO和GRPO。我们还进行广泛的实验，例如在线与离线训练、结果与过程监督、单轮与迭代强化学习等，以深入研究该范式的基本要素。

基于我们的统一范式，我们探讨强化学习有效性的原因，并总结了几个潜在方向，以实现更有效的LLM强化学习。

# 1.2. 评估和指标总结

• 英语和中文数学推理：我们对模型在英语和中文基准测试上的表现进行全面评估，涵盖从小学到大学级别的数学问题。英语基准测试包括GsM8K (Cobbe等，2021)、MATH (Hendrycks等，2021)、SAT (Azerbayev等，2023)、OCW课程 (Lewkowycz等，2022a)、MMLU-STEM (Hendrycks等，2020)。中文基准测试包括MGSM-zh (Shi等，2023)、CMATH (Wei等，2023)、高考数学填空 (Zhong等，2023) 和高考数学问答 (Zhong等，2023)。我们评估模型在不使用工具的情况下生成自包含文本解决方案的能力，以及使用Python解决问题的能力。

在英语基准测试中，DeepSeekMath-Base 与闭源的 Minerva 540B (Lewkowycz et al., 2022a) 具有竞争力，并且超越了所有开源基础模型（例如，Mistral 7B (iang et al., 2023) 和 Llemma-34B (Azerbayev et al., 2023)），无论它们是否经过数学预训练，通常都有显著的优势。值得注意的是，DeepSeekMath-Base 在中文基准测试中表现更佳，这可能是因为我们没有遵循之前的研究 (Azerbayev et al., 2023; Lewkowycz et al., 2022a) 来收集仅限英语的数学预训练数据，而是纳入了高质量的非英语数据。通过数学指导调优和强化学习，生成的 DeepSeekMath-Instruct 和 DeepSeekMath-RL 展现了强大的性能，首次在开源社区中在竞争级 MATH 数据集上的准确率超过了 $50 \%$。

•形式数学：我们使用 (iang et al., 2022) 中的非正式到正式定理证明任务评估 DeepSeekMath-Base，任务在 miniF2F (Zheng et al., 2021) 上进行，选用 Isabelle (Wenzel et al., 2008) 作为证明助手。DeepSeekMath-Base 展现了强大的少量样本自动形式化表现。

•自然语言理解、推理和编码：为了全面评估模型的一般理解、推理和编码能力，我们在大规模多任务语言理解（MMLU）基准上评估DeepSeekMath-Base（Hendrycks等人，2020），该基准涵盖57个多项选择任务，涉及多种主题，还包括BIG-Bench Hard（BBH）（Suzgun等人，2022），其中包含23个大多数需要多步推理才能解决的挑战性任务，以及HumanEval（Chen等人，2021）和MBPP（Austin等人，2021），这些都是广泛用于评估代码语言模型的基准。数学预训练有利于语言理解和推理性能的提升。

# 数学预训练

# 2.1. 数据收集与去污处理

在本节中，我们将概述从Common Crawl构建DeepSeekMath语料库的过程。如图2所示，我们呈现了一种迭代管道，展示了如何从Common Crawl系统地收集大规模数学语料库，首先从种子语料库（例如，一个小但高质量的数学相关数据集集合）开始。值得注意的是，这种方法也适用于其他领域，例如编码。

首先，我们选择OpenWebMath（Paster等，2023），一个高质量数学网络文本的集合，作为我们的初始种子语料库。利用这个语料库，我们训练一个fastText模型（Joulin等，2016），以回忆更多类似OpenWebMath的数学网页。具体而言，我们随机从种子语料库中选择500,000个数据点作为正面训练示例，并从Common Crawl中选择另外500,000个网页作为负面示例。我们使用一个开源库进行训练，将向量维度配置为256，学习率设置为0.1，词n-gram的最大长度设置为3，词出现的最小次数设置为3，训练周期数设置为3。为了减少原始Common Crawl的大小，我们采用基于URL的去重和近重复技术，最终得到400亿个HTML网页。然后，我们利用fastText模型从去重后的Common Crawl中召回数学网页。为了过滤低质量的数学内容，我们根据fastText模型预测的分数对收集到的网页进行排名，仅保留排名最高的网页。保留的数据量通过对前40B、80B、120B和160B个标记进行预训练实验来评估。在第一次迭代中，我们选择保留前40B个标记。

![](images/2.jpg)  
Figure 2 | An iterative pipeline that collects mathematical web pages from Common Crawl.

在第一次数据收集迭代后，仍有大量数学网页未被收录，主要因为fastText模型是基于一组缺乏足够多样性的正面示例进行训练的。因此，我们确定了额外的数学网页源以丰富种子语料库，从而优化fastText模型。具体来说，我们首先将整个Common Crawl组织成不相交的领域；领域被定义为共享相同基础URL的网页。对于每个领域，我们计算在第一次迭代中被收集的网页百分比。收集超过10%网页的领域被分类为与数学相关（例如，mathoverflow.net）。随后，我们手动标注这些被识别的领域中与数学内容相关的URL（例如，mathoverflow.net/questions）。与这些URL链接的仍未收集的网页将被添加到种子语料库中。这种方法使我们能够收集更多的正面示例，从而训练一个改进后的fastText模型，能够在后续迭代中调用更多的数学数据。在经过四次数据收集迭代后，我们最终得到了35.5M的数学网页，总计120B个标记。在第四次迭代中，我们注意到近98%的数据已在第三次迭代中被收集，因此我们决定停止数据收集。

为了避免基准污染，我们遵循Guo等人（2024）的做法，筛选掉包含英语数学基准（例如GSM8K（Cobbe等人，2021）和MATH（Hendrycks等人，2021））和中文基准（例如CMATH（Wei等人，2023）和AGIEval（Zhong等人，2023））中的问题或答案的网页。筛选标准如下：任何包含与评估基准中的任一子字符串完全匹配的10-gram字符串的文本段都将从我们的数学训练语料库中删除。对于短于10个gram但至少有3个gram的基准文本，我们采用精确匹配来筛选受污染的网页。

# 2.2. 验证DeepSeekMath语料库的质量

我们进行预训练实验，以调查DeepSeekMath语料库与最近发布的数学训练语料库的比较：

•MathPile（Wang et al., 2023c）：一个来自教科书、维基百科、ProofWiki、CommonCrawl、StackExchange 和 arXiv 的多源语料库（89亿标记），其中大部分（超过85%）来自 arXiv；

•OpenWebMath（Paster等，2023）：过滤后的CommonCrawl数据，针对数学内容，总计136亿个标记；

•Proof-Pile-2（Azerbayev等，2023）：一个数学语料库，由OpenWebMath、AlgebraicStack（10.3B数学代码令牌）和arXiv论文（28.0B令牌）组成。在对Proof-Pile-2进行实验时，我们遵循Azerbayev等（2023）的建议，使用arXiv:Web:Code的比例为2:4:1。

# 2.2.1. 训练设置

我们将数学训练应用于一个具有13亿参数的通用预训练语言模型，该模型与DeepSeek LLMs（DeepSeek-AI，2024）具有相同的框架，记作DeepSeekLLM 1.3B。我们分别在每个数学语料库上训练一个模型，共计150B个标记。所有实验均使用高效轻量的HAI-LLM（High-flyer，2023）训练框架进行。按照DeepSeek LLMs的训练实践，我们使用AdamW优化器（Loshchilov和Hutter，2017），其中$\beta _ { 1 } = 0 . 9$，$\beta _ { 2 } = 0 . 9 5$，权重衰减$= 0 . 1$，并采用多步学习率调度，其中学习率在2000个预热步骤后达到峰值，在训练过程的80%后降至峰值的31.6%，并在训练过程的90%后进一步降至峰值的10.0%。我们将学习率的最大值设置为$5 . 3 \mathrm { e } { \cdot } 4$，并使用4M标记的批量大小和4K的上下文长度。

<table><tr><td rowspan="2">数学语料库</td><td rowspan="2">大小</td><td colspan="5">英语基准</td><td colspan="3">中文基准</td></tr><tr><td>GSM8K数学</td><td></td><td>OCW</td><td>SAT</td><td>MMLU STEM</td><td>CMATH</td><td>高考数学填空</td><td>高考数学问答</td></tr><tr><td>无数学训练</td><td>N/A</td><td>2.9%</td><td>3.0%</td><td>2.9%</td><td>15.6%</td><td>19.5%</td><td>12.3%</td><td>0.8%</td><td>17.9%</td></tr><tr><td>MathPile</td><td>8.9B</td><td>2.7%</td><td>3.3%</td><td>2.2%</td><td>12.5%</td><td>15.7%</td><td>1.2%</td><td>0.0%</td><td>2.8%</td></tr><tr><td>OpenWebMath</td><td>13.6B</td><td>11.5%</td><td>8.9%</td><td>3.7%</td><td>31.3%</td><td>29.6%</td><td>16.8%</td><td>0.0%</td><td>14.2%</td></tr><tr><td>Proof-Pile-2</td><td>51.9B</td><td>14.3%</td><td>11.2%</td><td>3.7%</td><td>43.8%</td><td>29.2%</td><td>19.9%</td><td>5.1%</td><td>11.7%</td></tr><tr><td>DeepSeekMath语料库</td><td>120.2B</td><td>23.8%</td><td>13.6%</td><td>4.8%</td><td>56.3%</td><td>33.1%</td><td>41.5%</td><td>5.9%</td><td>23.6%</td></tr></table>

表1 | DeepSeek-LLM 1.3B 在不同数学语料库上的表现，使用少量示例链式思维提示进行评估。语料库大小是使用我们词汇表大小为100K的分词器计算的。

# 2.2.2. 评估结果

DeepSeekMath语料库的质量很高，涵盖了多语言的数学内容，并且是规模最大的。

•高质量：我们使用少样本思维链提示 Wei 等人（2022）在8个数学基准上评估下游性能。如表1所示，训练于 DeepSeekMath 语料库的模型明显表现更优。图3显示，训练于 DeepSeekMath 语料库的模型表现优于

![](images/3.jpg)  
Figure 3 | Benchmark curves of DeepSeek-LLM 1.3B trained on different mathematical corpora.

Proof-Pile-2在50B标记（Proof-Pile-2的1个完整周期）时，表明DeepSeekMath语料库的平均质量更高。

•多语言：DeepSeekMath语料库包含多种语言的数据，主要以英语和中文为主。正如表1所示，在DeepSeekMath语料库上训练提高了英语和中文的数学推理表现。相比之下，现有的数学语料库主要以英语为中心，显示出有限的改进，甚至可能阻碍中文数学推理的表现。

•大规模：DeepSeekMath语料库的规模是现有数学语料库的几倍。如图3所示，当DeepSeek-LLM 1.3B在DeepSeekMath语料库上训练时，学习曲线更陡峭，且改进更持久。相比之下，基准语料库要小得多，且在训练过程中已经经历了多轮重复，导致模型性能迅速达到平台期。

# 2.3. 训练和评估 DeepSeekMath-Base 7B

在本节中，我们介绍 DeepSeekMath-Base 7B，这是一个具有强大推理能力的基础模型，尤其是在数学方面。我们的模型以 DeepSeek-Coder-Base-v1.5 7B（Guo 等，2024）为初始化，并进行了 500B 令牌的训练。数据的分布如下：$56\%$ 来自 DeepSeekMath 语料库，$4\%$ 来自 AlgebraicStack，$10\%$ 来自 arXiv，$20\%$ 是 Github 代码，其余的 $10\%$ 是来自 Common Crawl 的自然语言数据，包括英语和中文。我们主要采用第 2.2.1 节中指定的训练设置，除了将学习率的最大值设置为 $4.2 \mathrm{e}\cdot 4$，并使用 10M 令牌的批量大小。

我们对DeepSeekMathBase7的数学能力进行全面评估，重点关注它在不依赖外部工具的情况下生成自包含的数学解答、使用工具解决数学问题以及进行正式定理证明的能力。除了数学之外，我们还提供基础模型的更一般化的概况，包括其自然语言理解、推理和编程技能的表现。

数学问题解决与逐步推理 我们评估了DeepSeekMathBase在数学问题上解决的表现，采用少量示例的连锁思维提示（Wei et al., 2022），在八个基准测试中进行评估，涵盖英语和中文。这些基准测试包括定量推理（例如，GSM8K（Cobbe et al., 2021）、MATH（Hendrycks et al., 2021）和CMATH（Wei et al., 2023））以及选择题（例如，MMLU-STEM（Hendrycks et al., 2020）和高考数学问答（Zhong et al., 2023）），涵盖从初级到大学水平复杂度的多种数学领域。

如表2所示，在所有八个基准测试中，DeepSeekMath-Base 7B在开源基础模型中表现领先（包括广泛使用的通用模型Mistral 7B（Jiang等，2023）和最近发布的Llemma 34B（Azerbayev等，2023），后者在Proof-Pile-2上进行了数学训练（Azerbayev等，2023））。值得注意的是，在竞争级别的MATH数据集上，DeepSeekMath-Base的表现超过现有开源基础模型超过$10\%$，并且优于Minerva 540B（Lewkowycz等，2022a），该闭源基础模型大77倍，基于PaLM（Lewkowycz等，2022b）并在数学文本上进行了进一步训练。

表2 | DeepSeekMath-Base 7B与强基础模型在英语和中文上的比较。模型评估来自Minerva的结果引用自Lewkowycz等（2022a）。

<table><tr><td rowspan="2">模型</td><td rowspan="2">规模</td><td colspan="4">英语基准</td><td colspan="3">中文基准</td></tr><tr><td>GSM8K数学</td><td>OCW</td><td>SAT</td><td>MMLU STEM</td><td>CMATH</td><td>高考数学填空</td><td>高考数学问答</td></tr><tr><td colspan="10">封闭源基础模型</td></tr><tr><td>Minerva</td><td>7B</td><td>16.2%</td><td>14.1%</td><td>7.7%</td><td>35.6%</td><td></td><td></td><td></td><td></td></tr><tr><td>Minerva Minerva</td><td>62B</td><td>52.4%</td><td>27.6%</td><td>12.0%</td><td>-</td><td>53.9%</td><td>-</td><td></td><td></td></tr><tr><td></td><td>540B</td><td>58.8%</td><td>33.6%</td><td>17.6%</td><td>-</td><td>63.9%</td><td></td><td></td><td></td></tr><tr><td colspan="10">开放源基础模型</td></tr><tr><td>Mistral</td><td>7B</td><td>40.3%</td><td>14.3%</td><td>9.2%</td><td>71.9%</td><td>51.1%</td><td>44.9%</td><td>5.1%</td><td>23.4%</td></tr><tr><td>Llemma</td><td>7B</td><td>37.4%</td><td>18.1%</td><td>6.3%</td><td>59.4%</td><td>43.1%</td><td>43.4%</td><td>11.9%</td><td>23.6%</td></tr><tr><td>Llemma</td><td>34B</td><td>54.0%</td><td>25.3%</td><td>10.3%</td><td>71.9%</td><td>52.9%</td><td>56.1%</td><td>11.9%</td><td>26.2%</td></tr><tr><td>DeepSeekMath-Base 7B</td><td></td><td>64.2%</td><td>36.2%</td><td>15.4%</td><td>84.4%</td><td>56.5%</td><td>71.7%</td><td>20.3%</td><td>35.3%</td></tr></table>

使用工具的数学问题解决 我们在GSM8K和MATH上评估基于程序辅助的数学推理，使用少量示例的思维程序提示（Chen等，2022；Gao等，2023）。模型被提示通过编写Python程序来解决每个问题，可以利用math和sympy等库进行复杂计算。程序的执行结果被评估为答案。如表3所示，DeepSeekMath-Base 7B的表现优于之前的最新技术Llemma 34B。

表3 | 基本模型在使用工具解决数学问题方面的少量样本评估，以及在Isabelle中进行非正式到正式定理证明的能力。

<table><tr><td rowspan="2">模型</td><td rowspan="2">大小</td><td colspan="2">使用工具解决问题</td><td colspan="2">非正式到正式证明</td></tr><tr><td>GSM8K+Python MATH+Python miniF2F-valid miniF2F-test</td><td></td><td></td><td></td></tr><tr><td>Mistral</td><td>7B</td><td>48.5%</td><td>18.2%</td><td>18.9%</td><td>18.0%</td></tr><tr><td>CodeLlama</td><td>7B</td><td>27.1%</td><td>17.2%</td><td>16.3%</td><td>17.6%</td></tr><tr><td>CodeLlama</td><td>34B</td><td>52.7%</td><td>23.5%</td><td>18.5%</td><td>18.0%</td></tr><tr><td>Llemma</td><td>7B</td><td>41.0%</td><td>18.6%</td><td>20.6%</td><td>22.1%</td></tr><tr><td>Llemma</td><td>34B</td><td>64.6%</td><td>26.3%</td><td>21.0%</td><td>21.3%</td></tr><tr><td>DeepSeekMath-Base 7B</td><td></td><td>66.9%</td><td>31.4%</td><td>25.8%</td><td>24.6%</td></tr></table>

形式数学 形式证明自动化有助于确保数学证明的准确性和可靠性，并提高效率，近年来受到越来越多的关注。我们在非正式到正式证明的任务上评估了DeepSeekMath-Base 7B（Jiang et al., 2022），该任务旨在根据非正式声明、声明的正式对应和非正式证明生成正式证明。我们在miniF2F（Zheng et al., 2021）上进行评估，这是一个正式的奥林匹克级数学基准，并为每个问题生成了一个在Isabelle中的正式证明，使用少量提示。我们遵循了Jiang et al.（2022）的做法，利用模型生成证明草图，并执行现成的自动证明工具Sledgehammer（Paulson, 2010）来填补遗漏的细节。如表3所示，DeepSeekMath-Base 7B在证明自动形式化方面表现出色。

<table><tr><td>模型</td><td>大小</td><td>MMLU</td><td>BBH</td><td>HumanEval (通过率@1) MBPP (通过率@1)</td><td></td></tr><tr><td>Mistral</td><td>7B</td><td>62.4%</td><td>55.7%</td><td>28.0%</td><td>41.4%</td></tr><tr><td>DeepSeek-Coder-Base-v1.5†</td><td>7B</td><td>42.9%</td><td>42.9%</td><td>40.2%</td><td>52.6%</td></tr><tr><td>DeepSeek-Coder-Base-v1.5</td><td>7B</td><td>49.1%</td><td>55.2%</td><td>43.2%</td><td>60.4%</td></tr><tr><td>DeepSeekMath-Base</td><td>7B</td><td>54.9%</td><td>59.5%</td><td>40.9%</td><td>52.6%</td></tr></table>

表4 | 在自然语言理解、推理和代码基准上的评估。DeepSeek-Coder-Base $\mathbf { \cdot v } 1 . 5 ^ { \dagger }$ 是在学习率衰减之前的检查点，用于训练DeepSeekMath-Base。在MMLU和BBH上，我们使用少量示例的思维链提示。在HumanEval和MBPP上，我们分别在零-shot设置和少量示例设置下评估模型性能。

自然语言理解、推理和编码 我们在 MMLU（Hendrycks et al.，2020）、BBH（Suzgun et al.，2022）上评估模型在自然语言理解和推理方面的表现，以及在 HumanEval（Chen et al.，2021）和 MBPP（Austin et al.）上的编码能力。

2021年）。如表4所示，DeepSeekMath-Base 7B在MMLU和BBH上的性能相较于其前身DeepSeek-Coder-Base-v1.5（Guo等人，2024）有显著改善，展示了数学训练对语言理解和推理的积极影响。此外，通过将代码标记纳入持续训练，DeepSeekMath-Base 7B有效维持了DeepSeek-Coder-Base-v1.5在这两个编码基准上的表现。总体而言，DeepSeekMath-Base 7B在这三个推理和编码基准上显著超越了通用模型Mistral 7B（Jiang等人，2023）。

# 有监督的细调

# 3.1. SFT 数据策划

我们构建了一个数学指令调优数据集，涵盖来自不同数学领域和不同复杂度水平的英文和中文问题：问题与链式思维（CoT）（Wei et al., 2022）、程序思维（PoT）（Chen et al., 2022；Gao et al., 2023）和工具集成推理格式（Gou et al., 2023）的解决方案配对。训练示例的总数为776K。

• 英文数学数据集：我们对GSM8K和MATH问题进行了工具集成解决方案的注释，并采用了MathInstruct（Yue等，2023）的一个子集，以及Lila-OOD的训练集（Mishra等，2022），这些问题是通过链式思维（CoT）或过程思维（PoT）解决的。我们的英文集合涵盖了数学的多个领域，例如代数、概率、数论、微积分和几何。

中文数学数据集：我们收集了涵盖76个子主题的中国K-12数学问题，例如线性方程，并以CoT和工具集成推理格式注释了解题方案。

# 3.2. 训练和评估 DeepSeekMath-Instruct 7B

在本节中，我们介绍DeepSeekMath-Instruct 7B，该模型基于DeepSeekMath-Base进行了数学指令微调。训练示例被随机连接，直到达到最大上下文长度4K个标记。我们以256的批量大小和0.00005的恒定学习率对模型进行500步训练。

我们评估模型的数学表现，包括在没有使用工具和使用工具的情况下，针对4个定量推理基准测试（英语和中文）。我们将我们的模型与当时的领先模型进行基准测试：

•封闭源模型包括：(1) GPT系列，其中GPT-4（OpenAI，2023）和GPT-4代码解释器2是最强大的，(2) Gemini Ultra和Pro（Anil等，2023），(3) Inflection-2（Inflection AI，2023），(4) Grok- $\cdot 1 ^ { 3 }$ ，以及中国公司最近推出的模型，包括(5) Baichuan- $3 ^ { 4 }$ ，(6) GLM家族中最新的GLM- $. 4 ^ { 5 }$ （杜等，2022）。这些模型是通用的，大多数经过了一系列的对齐程序。

开源模型包括：通用模型如（1）DeepSeek-LLM-Chat 67B（DeepSeekAI，2024），（2）Qwen 72B（Bai等，2023），（3）SeaLLM-v2 7B（Nguyen等，2023），和（4）

ChatGLM3 6B（ChatGLM3团队，2023），以及在数学方面有增强的模型，包括（5）InternLM2-Math $2 0 \mathrm { B } ^ { 6 }$，该模型基于InternLM2，并经过数学训练后进行了指令调优，（6）Math-Shepherd-Mistral 7B，该模型对Mistral 7B（Jiang et al., 2023）应用PPO训练（Schulman et al., 2017），并使用过程监督的奖励模型，（7）WizardMath系列（Luo et al., 2023），该系列改善了Mistral 7B和Llama-2 70B（Touvron et al., 2023）中的数学推理，使用渐进指令（即一种使用AI演变指令的指令调优版本）和PPO训练，主要训练问题来自GSM8K和MATH，（8）MetaMath 70B（Yu et al., 2023），该模型是在增强版本的GSM8K和MATH上对Llama-2 70B进行微调，（9）ToRA 34B（Gou et al., 2023），该模型是CodeLlama 34B微调用于工具集成的数学推理，（10）MAmmoTH 70B（Yue et al., 2023），该模型是在MathInstruct上进行指令调优的Llama-2 70B。

如表5所示，在禁止使用工具的评估设置中，DeepSeekMath.Instruct 7B 展现了强大的逐步推理能力。值得注意的是，在竞争级的 MATH 数据集上，我们的模型超过了所有开源模型和大多数专有模型（例如，Inflection-2 和 Gemini Pro）至少 $9 \%$ 的绝对值。即使对于更大规模的模型（例如，Qwen 72B）或经过数学强化学习专门增强的模型（例如，WizardMath-v1.1 7B）而言，这一点也是成立的。虽然 DeepSeekMath-Instruct 在 MATH 上与中国的专有模型 GLM-4 和 Baichuan-3 进行了竞争，但仍然不及 GPT-4 和 Gemini Ultra 的表现。

在评估设置中，模型可以结合自然语言推理和基于程序的工具使用来解决问题，DeepSeekMath-Instruct 7B在MATH上的准确率接近60%，超越了所有现有的开源模型。在其他基准测试中，我们的模型与之前的最先进的DeepSeek-LLM-Chat 67B竞争，该模型大约大10倍。

# 4强化学习

# 4.1. 群体相对政策优化

强化学习（RL）已被证明能够在监督微调（SFT）阶段之后进一步提高大型语言模型（LLMs）的数学推理能力（Luo et al., 2023；iR相对策略优化（GRPO）。

# 4.1.1. 从 PPO 到 GRPO

近端策略优化（PPO）（Schulman等，2017）是一种广泛用于大型语言模型（LLM）强化学习微调阶段的演员-评论家强化学习算法（Ouyang等，2022）。具体而言，它通过最大化以下替代目标来优化LLM：

$$
\mathcal { T } p r o  ( \theta ) = \mathbb { E } [ q \sim P ( Q ) , o \sim \pi \theta _ { o d d } ( O | q ) ] \frac { 1 } { | o | } \sum _ { t = 1 } ^ { | o | } \operatorname* { m i n } \left[ \frac { \pi _ { \theta } ( o _ { t } | q , o _ { < t } ) } { \pi _ { \theta _ { o d d } } ( o _ { t } | q , o _ { < t } ) } A _ { t } , \mathrm { c l i p } \left( \frac { \pi _ { \theta } ( o _ { t } | q , o _ { < t } ) } { \pi _ { \theta _ { o d d } } ( o _ { t } | q , o _ { < t } ) } , 1 - \varepsilon , 1 + \varepsilon \right) A _ { t } \right] ,
$$

其中 $\pi _ { \theta }$ 和 $\pi _ { \theta _ { o l d } }$ 是当前和旧的策略模型，$q, o$ 分别是从问题数据集中抽样的问答和旧政策 $\pi _ { \theta _ { o l d } } $ 的输出。$\varepsilon$ 是在 PPO 中引入的与裁剪相关的超参数，用于稳定训练。$A _ { t }$ 是优势，通过应用广义优势估计（GAE）(Schulman et al., 2015) 计算，基于奖励 $\left\{ r _ { \geq t } \right\}$ 和学习的值函数 $V _ { \psi }$。因此，在 PPO 中，需要训练一个值函数与策略模型一起进行，并且为了减轻奖励模型的过度优化，标准方法是在每个标记的奖励中添加来自参考模型的每个标记的 $\mathrm { K L }$ 惩罚 (Ouyang et al., 2022)，即：

表5 | 开源和闭源模型在英文和中文基准上的表现，采用链式思维和工具集成推理。灰色的分数表示32个候选者的多数投票；其他为Top1分数。DeepSeekMath-RL 7B超越了所有7B到70B的开源模型，以及大多数闭源模型。尽管DeepSeekMath-RL 7B仅在GSM8K和MATH的链式思维格式指令调优数据上进行了进一步训练，但在所有基准上均优于DeepSeekMath-Instruct 7B。

<table><tr><td rowspan="2">模型</td><td rowspan="2">大小</td><td colspan="2">英语基准 中文基准</td><td colspan="2"></td></tr><tr><td>GSM8K</td><td>MATH</td><td>MGSM-zh</td><td>CMATH</td></tr><tr><td colspan="6">思维链推理</td></tr><tr><td></td><td></td><td>封闭源码模型</td><td></td><td></td><td></td></tr><tr><td>Gemini Ultra</td><td>-</td><td>94.4%</td><td>53.2%</td><td></td><td>-</td></tr><tr><td>GPT-4</td><td></td><td>92.0%</td><td>52.9%</td><td></td><td>86.0%</td></tr><tr><td>Inflection-2</td><td>-</td><td>81.4%</td><td>34.8%</td><td></td><td>-</td></tr><tr><td>GPT-3.5</td><td></td><td>80.8%</td><td>34.1%</td><td></td><td>73.8%</td></tr><tr><td>Gemini Pro</td><td></td><td>86.5%</td><td>32.6%</td><td></td><td></td></tr><tr><td>Grok-1</td><td>-</td><td>62.9%</td><td>23.9%</td><td></td><td>-</td></tr><tr><td>Baichuan-3</td><td>-</td><td>88.2%</td><td>49.2%</td><td></td><td></td></tr><tr><td>GLM-4</td><td>-</td><td>87.6%</td><td>47.9%</td><td></td><td></td></tr><tr><td colspan="6">开源模型</td></tr><tr><td>InternLM2-Math</td><td>20B</td><td>82.6%</td><td>37.7%</td><td></td><td></td></tr><tr><td>Qwen</td><td>72B</td><td>78.9%</td><td>35.2%</td><td></td><td></td></tr><tr><td>Math-Shepherd-Mistral</td><td>7B</td><td>84.1%</td><td>33.0%</td><td></td><td></td></tr><tr><td>WizardMath-v1.1</td><td>7B</td><td>83.2%</td><td>33.0%</td><td>-</td><td>-</td></tr><tr><td>DeepSeek-LLM-Chat</td><td>67B</td><td>84.1%</td><td>32.6%</td><td>74.0%</td><td>80.3%</td></tr><tr><td>MetaMath</td><td>70B</td><td>82.3%</td><td>26.6%</td><td>66.4%</td><td>70.9%</td></tr><tr><td>SeaLLM-v2</td><td>7B</td><td>78.2%</td><td>27.5%</td><td>64.8%</td><td>-</td></tr><tr><td>ChatGLM3</td><td>6B</td><td>72.3%</td><td>25.7%</td><td>-</td><td>-</td></tr><tr><td>WizardMath-v1.0</td><td>70B</td><td>81.6%</td><td>22.7%</td><td>64.8%</td><td>65.4%</td></tr><tr><td>DeepSeekMath-Instruct</td><td>7B</td><td>82.9%</td><td>46.8%</td><td>73.2%</td><td>84.6%</td></tr><tr><td>DeepSeekMath-RL</td><td>7B</td><td>88.2%</td><td>51.7%</td><td>79.6%</td><td>88.8%</td></tr><tr><td colspan="6">工具集成推理</td></tr><tr><td></td><td></td><td></td><td>封闭源码模型</td><td></td><td></td></tr><tr><td>GPT-4 代码解释器</td><td></td><td>97.0%</td><td>69.7%</td><td></td><td></td></tr><tr><td colspan="6">开源模型</td></tr><tr><td>InternLM2-Math</td><td>20B</td><td>80.7%</td><td>54.3%</td><td>-</td><td></td></tr><tr><td>DeepSeek-LLM-Chat</td><td>67B</td><td>86.7%</td><td>51.1%</td><td>76.4%</td><td>85.4%</td></tr><tr><td>ToRA</td><td>34B</td><td>80.7%</td><td>50.8%</td><td>41.2%</td><td>53.4%</td></tr><tr><td>MAmmoTH</td><td>70B</td><td>76.9%</td><td>41.8%</td><td>-</td><td></td></tr><tr><td>DeepSeekMath-Instruct 7B</td><td></td><td>83.7%</td><td>57.4%</td><td>72.0%</td><td>84.3%</td></tr><tr><td>DeepSeekMath-RL</td><td>7B</td><td>86.7%</td><td>58.8%</td><td>78.4%</td><td>87.6%</td></tr></table>

![](images/4.jpg)  
Figure 4 | Demonstration of PPO and our GRPO. GRPO foregoes the value model, instead estimating the baseline from group scores, significantly reducing training resources.

$$
r _ { t } = r _ { \varphi } ( q , o _ { \le t } ) - \beta \log \frac { \pi _ { \theta } ( o _ { t } | q , o _ { < t } ) } { \pi _ { r e f } ( o _ { t } | q , o _ { < t } ) } ,
$$

其中 $r _ { \varphi }$ 是奖励模型，$\pi _ { r e f }$ 是参考模型，通常是最初的 SFT 模型，$\beta$ 是 KL 惩罚的系数。

在PPO中使用的价值函数通常是与策略模型规模相当的另一个模型，因此带来了可观的内存和计算负担。此外，在强化学习训练中，价值函数被视为计算优势时的基线，以降低方差。而在大语言模型的上下文中，通常只有最后一个标记通过奖励模型被分配奖励分数，这可能会 complicate 价值函数在每个标记上的准确训练。为了解决这个问题，如图4所示，我们提出了组相对策略优化（GRPO），它不再像PPO一样需要额外的价值函数近似，而是将对同一问题的多个采样输出的平均奖励作为基线。更具体地说，对于每个问题 $q $，GRPO 从旧策略 $\pi _ { \theta _ { o l d } }$ 中采样一组输出 $\{ o _ { 1 } , o _ { 2 } , \cdots , o _ { G } \}$，然后通过最大化以下目标来优化策略模型：

$$
\begin{array} { l } { \displaystyle \mathcal { J } _ { G R P O } ( \theta ) = \mathbb { E } [ q \sim P ( Q ) , \{ \alpha _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { o d } } ( O | q ) ] } \\ { \displaystyle \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { | \alpha _ { i } | } \sum _ { t = 1 } ^ { | \alpha _ { i } | } \left\{ \operatorname* { m i n } \left[ \frac { \pi _ { \theta } ( o _ { i , t } | q , o _ { i , c , t } ) } { \pi _ { \theta _ { o d } } ( o _ { i , t } | q , o _ { i , c , t } ) } \hat { A } _ { i , t } , \mathrm { c i l p } \left( \frac { \pi _ { \theta } ( o _ { i , t } | q , o _ { i , c , t } ) } { \pi _ { \theta _ { o d } } ( o _ { i , t } | q , o _ { i , c , t } ) } , 1 - \varepsilon , 1 + \varepsilon \right) \hat { A } _ { i , t } \right] - \beta \mathbb { D } _ { K L } \left[ \pi _ { \theta } | | \pi _ { r e f } | \right] \right\} , } \end{array}
$$

其中 $\varepsilon$ 和 $\beta$ 是超参数，$\hat { A } _ { i , t }$ 是基于每组内部输出的相对奖励计算的优势，具体将在以下子章节中详细说明。GRPO 用于计算优势的组相对方式，与奖励模型的比较性质非常吻合，因为奖励模型通常是在同一问题上对输出进行比较的数据集上训练的。此外，请注意，GRPO 不通过在奖励中添加 KL 惩罚来进行正则化，而是通过直接将训练策略与参考策略之间的 KL 散度添加到损失中进行正则化，从而避免了复杂化 $\hat { A } _ { i , t }$ 的计算。

输入初始策略模型 $\pi _ { \theta _ { \mathrm { i n i t } } }$；奖励模型 $r _ { \varphi }$；任务提示 $\mathcal { D }$；超参数 ε, β, µ

1: 策略模型 πθ ← πθinit   
2: 对于迭代 $= 1 , \hdots , \mathrm { I }$ do   
3: 参考模型 $\pi _ { r e f }  \pi _ { \theta }$ EPY   
4: 对于步骤 $\mathbf { \Omega } = 1 , \dots , \mathbf { M }$ do   
5: 从 $\mathcal { D }$ 中采样一个批次 $\mathcal { D } _ { b }$   
6: 更新旧的策略模型 $\pi _ { \theta _ { o l d } }  \pi _ { \theta }$ EMP   
7: 为每个问题 $q \in \mathcal { D } _ { b }$ 采样 $G$ 输出 $\{ o _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { o l d } } ( \cdot \mid q )$   
8: 通过运行 $r _ { \varphi }$ 计算每个采样输出 $o _ { i }$ 的奖励 $\{ r _ { i } \} _ { i = 1 } ^ { G }$   
9: 通过组相对优势估计计算 $o _ { i }$ 的第 $t$ 个标记的 $\hat { A } _ { i , t }$。   
10: 对于 GRPO 迭代 $= 1 , \ldots , \mu$ do   
11: 通过最大化 GRPO 目标（方程 21）更新策略模型 $\pi _ { \theta }$   
12: 通过使用重放机制的连续训练更新 $r _ { \varphi }$。

# 输出 $\pi _ { \theta }$

与（2）中使用的KL惩罚项不同，我们使用以下无偏估计量来估计KL散度（Schulman，2020）：

$$
\mathbb { D } _ { K L } \left[ \pi _ { \theta } | | \pi _ { r e f } \right] = \frac { \pi _ { r e f } ( o _ { i , t } | q , o _ { i , < t } ) } { \pi _ { \theta } ( o _ { i , t } | q , o _ { i , < t } ) } - \log \frac { \pi _ { r e f } ( o _ { i , t } | q , o _ { i , < t } ) } { \pi _ { \theta } ( o _ { i , t } | q , o _ { i , < t } ) } - 1 ,
$$

保证是正的。

# 4.1.2. 使用 GRPO 的结果监督强化学习

形式上，对于每个问题 $q $，从旧策略模型 $\pi _ { \theta _ { o l d } }$ 中抽样出一组输出 $\left\{ o _ { 1 } , o _ { 2 } , \cdots , o _ { G } \right\}$ 。然后，使用奖励模型对输出进行评分，产生相应的 $G$ 个奖励 $\mathbf { r } = \{ r _ { 1 } , r _ { 2 } , \cdots , r _ { G } \}$ 。随后，这些奖励通过减去组平均值并除以组标准差进行归一化。结果监督在每个输出 $o _ { i }$ 的末尾提供归一化奖励，并设置所有标记的优势 $\hat { A } _ { i , t }$ ，然后通过最大化方程（3）中定义的目标来优化策略。

# 4.1.3. 使用GRPO的过程监督强化学习

结果监督仅在每个输出结束时提供奖励，这可能不足以且效率不高地监督复杂数学任务中的策略。根据Wang等人（2023b）的研究，我们还探索了过程监督，该方法在每个推理步骤结束时提供奖励。形式上，给定问题$q$和$G$个采样输出$\{ o _ { 1 } , o _ { 2 } , \cdots , o _ { G } \} ,$一个$\mathbf { \dot { R } } = \{ \{ r _ { 1 } ^ { i n d e x ( 1 ) } , \cdots , r _ { 1 } ^ { i n d e x ( K _ { 1 } ) } \} , \cdots , \{ r _ { G } ^ { i n d e x ( 1 ) } , \cdots , r _ { G } ^ { i n d e x ( K _ { G } ) } \} \} ,$ us ieldin $( j )$ 对应的边际：$j \cdot$ $K _ { i }$ $i$ $\begin{array} { r } { \widetilde { r } _ { i } ^ { i n d e x ( j ) } = \frac { r _ { i } ^ { i n d e x ( j ) } - \mathrm { m e a n } ( \mathbf { R } ) } { \mathsf { s t d } ( \mathbf { R } ) } } \end{array}$ 随后，过程监督计算每个标记的优势作为归一化的$\begin{array} { r } { \hat { A } _ { i , t } = \sum _ { i n d e x ( j ) \geq t } \widetilde { r } _ { i } ^ { i n d e x ( j ) } } \end{array}$之和，最大化在方程（3）中定义的目标。

# 4.1.4. 使用GRPO的迭代强化学习

随着强化学习训练过程的进展，旧的奖励模型可能不足以监督当前的策略模型。因此，我们还探索了与GRPO结合的迭代RL。如算法1所示，在迭代GRPO中，我们基于策略模型的采样结果生成新的奖励模型训练集，并使用包含$10\%$历史数据的重放机制不断训练旧的奖励模型。然后，我们将参考模型设为策略模型，并使用新的奖励模型不断训练策略模型。

# 4.2. 训练和评估 DeepSeekMath-RL

我们基于DeepSeekMath-Instruct 7B进行强化学习（RL）。RL的训练数据是与GSM8K和MATH相关的链式思考格式的问题，这些问题来自于SFT数据，约有144K个问题。我们排除了其他SFT问题，以调查RL对缺乏数据的基准测试的影响。我们按照(Wang et al., 2023b)构建奖励模型的训练集。我们在DeepSeekMath-Base 7B上以2e-5的学习率训练初始奖励模型。对于GRPO，我们将策略模型的学习率设为1e-6。KL系数为0.04。对于每个问题，我们采样64个输出。最大长度设为1024，训练批量大小为1024。策略模型在每个探索阶段后仅进行一次更新。我们在基准测试上评估DeepSeekMath-RL 7B，基于DeepSeekMath-Instruct 7B。对于DeepSeekMath-RL 7B，具有链式思考推理的GSM8K和MATH可视为领域内任务，其他基准可视为领域外任务。

表5展示了开放源模型和封闭源模型在链式思维和工具集成推理下的表现，针对英语和中文基准进行评估。我们发现：1) DeepSeekMath-RL 7B在GSM8K和MATH上的准确率分别达到$8 8 . 2 \%$和$5 1 . 7 \%$，利用链式思维推理。这一表现超越了所有7B到70B范围内的开放源模型，以及大多数封闭源模型。2) 重要的是，DeepSeekMath-RL 7B仅在GSM8K和MATH的链式思维格式指导调优数据上进行训练，起始于DeepSeekMath-Instruct 7B。尽管其训练数据的范围有限，但在所有评估指标上均优于DeepSeekMath-Instruct 7B，展示了强化学习的有效性。

# 5. 讨论

在本节中，我们将分享我们在预训练和强化学习实验中的发现。

# 5.1. 预训练中学到的经验教训

我们首先分享在预训练中的经验。除非另有说明，我们将遵循第2.2.1节中概述的训练设置。值得注意的是，在本节提到DeepSeekMath语料库时，我们使用的是第二轮数据收集过程中的一个89B-token数据集。

# 5.1.1. 代码训练的好处 数学推理

一个流行但未经验证的假设认为，代码训练可以提高推理能力。我们尝试对此提供部分回应，特别是在数学领域：代码训练

<table><tr><td rowspan="2">训练设置</td><td colspan="3">训练标记</td><td colspan="3">不使用工具</td><td colspan="2">使用工具</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>通用代码 数学 GSM8K MATH CMATH GSM8K+Python MATH+Python</td></tr><tr><td>无持续训练</td><td>−</td><td></td><td>−</td><td>2.9%</td><td>3.0%</td><td>12.3%</td><td>2.7%</td><td>2.3%</td></tr><tr><td colspan="9">两阶段训练</td></tr><tr><td>阶段 1：通用训练</td><td>400B</td><td>−</td><td>−</td><td>2.9%</td><td>3.2%</td><td>14.8%</td><td>3.3%</td><td>2.3%</td></tr><tr><td>阶段 2：数学训练</td><td></td><td>−</td><td>150B</td><td>19.1%</td><td>14.4%</td><td>37.2%</td><td>14.3%</td><td>6.7%</td></tr><tr><td>阶段 1：代码训练</td><td></td><td>400B</td><td></td><td>5.9%</td><td>3.6%</td><td>19.9%</td><td>12.4%</td><td>10.0%</td></tr><tr><td>阶段 2：数学训练</td><td></td><td>−</td><td>150B</td><td>21.9%</td><td>15.3%</td><td>39.7%</td><td>17.4%</td><td>9.4%</td></tr><tr><td colspan="9">单阶段训练</td></tr><tr><td>数学训练</td><td></td><td>−</td><td>150B</td><td>20.5%</td><td>13.1%</td><td>37.6%</td><td>11.4%</td><td>6.5%</td></tr><tr><td>代码与数学混合训练</td><td></td><td>400B</td><td>150B</td><td>17.6%</td><td>12.1%</td><td>36.3%</td><td>19.7%</td><td>13.5%</td></tr></table>

表6 | 探讨在不同训练设置下代码如何影响数学推理。我们使用DeepSeek-LLM 1.3B进行实验，并在没有工具使用和使用工具的情况下，通过少量示例的连锁思维提示和少量示例的程序思维提示，评估其数学推理性能。

提高模型进行数学推理的能力，无论是否使用工具。

为了研究代码训练如何影响数学推理，我们进行了以下两阶段训练和单阶段训练设置的实验：

两阶段训练

代码训练400亿个标记，数学训练150亿个标记：我们对DeepSeekLLM 1.3B进行400亿个代码标记的训练，随后进行150亿个数学标记的训练；

•通用训练400B tokens，数学训练150B tokens：作为对照实验，我们还在训练的第一阶段用通用tokens（从DeepSeek-AI创建的大规模通用语料库中抽样）进行实验，而不是代码tokens，以探讨代码tokens在提高数学推理方面相对于通用tokens的优势。

# 单阶段训练

• 数学训练150B标记：我们为DeepSeek-LLM 1.3B训练150B数学标记； 
• 在400B代码标记和150B数学标记的混合训练：在代码训练后进行数学训练会降低编码性能。我们研究在一次阶段训练中，代码标记与数学标记混合时是否仍能提高数学推理，并减轻灾难性遗忘的问题。

Rlt 表6和表7展示了在不同设置下的下游表现。

代码训练优势计划帮助数学推理，无论在两阶段训练还是单阶段训练设置下。如表6所示，在两阶段训练设置下，仅凭代码训练就显著增强了使用Python解决GsM8K和MATH问题的能力。在第二阶段的数学训练中进一步取得了改善。有趣的是，在单阶段训练设置下，混合代码令牌和数学令牌有效缓解了两阶段训练带来的灾难性遗忘问题，同时也增强了编码（表7）和程序辅助的数学推理（表6）。

<table><tr><td rowspan="2">训练设置</td><td colspan="2">训练令牌</td><td rowspan="2">MMLU</td><td rowspan="2"></td><td rowspan="2">BBH HumanEval (Pass@1)MBPP (Pass@1)</td><td rowspan="2"></td></tr><tr><td>一般代码数学</td><td></td></tr><tr><td>无持续训练</td><td></td><td></td><td></td><td>24.5%</td><td>28.1%</td><td>12.2%</td><td>13.0%</td></tr><tr><td></td><td>两阶段训练</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>阶段 1: 一般训练</td><td>400B</td><td></td><td></td><td>25.9%</td><td>27.7%</td><td>15.2%</td><td>13.6%</td></tr><tr><td>阶段 2: 数学训练</td><td></td><td></td><td>150B</td><td>33.1%</td><td>32.7%</td><td>12.8%</td><td>13.2%</td></tr><tr><td>阶段 1: 代码训练</td><td></td><td>400B</td><td>−</td><td>25.0%</td><td>31.5%</td><td>25.0%</td><td>40.0%</td></tr><tr><td>阶段 2: 数学训练</td><td></td><td>—</td><td>150B</td><td>36.2%</td><td>35.3%</td><td>12.2%</td><td>17.0%</td></tr><tr><td></td><td></td><td></td><td></td><td>单阶段训练</td><td></td><td></td><td></td></tr><tr><td>数学训练</td><td></td><td></td><td>150B</td><td>32.3%</td><td>32.5%</td><td>11.6%</td><td>13.2%</td></tr><tr><td>代码与数学混合训练</td><td></td><td>400B</td><td>150B</td><td>33.5%</td><td>35.6%</td><td>29.3%</td><td>39.4%</td></tr></table>

表7 | 调查代码和数学训练的不同设置如何影响语言理解、推理和编码的模型性能。我们对DeepSeek-LLM 1.3B进行实验。我们使用少样本思维链提示在MMLU和BBH上对模型进行评估。在HumanEval和MBPP上，我们分别进行零样本和少样本评估。  
表8 | 数学训练对不同arXiv数据集的影响。模型性能通过少样本思维链提示进行评估。

<table><tr><td rowspan="2">模型</td><td rowspan="2"></td><td rowspan="2">ArXiv语料库大小</td><td colspan="4">英语基准</td><td rowspan="2"></td><td colspan="3">中文基准</td></tr><tr><td>GSM8K 数学 OCW</td><td></td><td>SAT</td><td>MMLU STEM</td><td>CMATH</td><td>高考 MathCloze MathQA</td><td>高考</td></tr><tr><td rowspan="3">DeepSeek-LLM</td><td rowspan="3">1.3B</td><td>无数学训练</td><td>2.9%</td><td>3.0%</td><td>2.9%</td><td>15.6%</td><td>19.5%</td><td>12.3%</td><td>0.8%</td><td>17.9%</td></tr><tr><td>MathPile</td><td>2.7%</td><td>3.3%</td><td>2.2%</td><td>12.5%</td><td>15.7%</td><td>1.2%</td><td>0.0%</td><td>2.8%</td></tr><tr><td>ArXiv-RedPajama</td><td>3.3%</td><td>3.4%</td><td>4.0%</td><td>9.4%</td><td>9.0%</td><td>7.4%</td><td>0.8%</td><td>2.3%</td></tr><tr><td rowspan="3">DeepSeek-Coder-Base-v1.5 7B</td><td rowspan="3"></td><td>无数学训练</td><td>29.0%</td><td>12.5%</td><td>6.6%</td><td>40.6%</td><td>38.1%</td><td>45.9%</td><td>5.9%</td><td>21.1%</td></tr><tr><td>MathPile</td><td>23.6%</td><td>11.5%</td><td>7.0%</td><td>46.9%</td><td>35.8%</td><td>37.9%</td><td>4.2%</td><td>25.6%</td></tr><tr><td>ArXiv-RedPajama</td><td>28.1%</td><td>11.1%</td><td>7.7%</td><td>50.0%</td><td>35.2%</td><td>42.6%</td><td>7.6%</td><td>24.8%</td></tr></table>

<table><tr><td>ArXiv语料库</td><td>miniF2F-valid</td><td>miniF2F-test</td></tr><tr><td>无数学训练</td><td>20.1%</td><td>21.7%</td></tr><tr><td>MathPile</td><td>16.8%</td><td>16.4%</td></tr><tr><td>ArXiv-RedPajama</td><td>14.8%</td><td>11.9%</td></tr></table>

表9 | 数学训练对不同arXiv语料库的影响，基础模型为DeepSeekCoder-Base-v1.5 7B。我们在Isabelle中评估非正式到正式的证明。

代码训练还在不使用工具的情况下改善数学推理。在两阶段训练设置下，代码训练的初始阶段已经带来了适度的提升。它还提高了后续数学训练的效率，最终导致最佳表现。然而，将代码标记和数学标记结合进行一阶段训练会妨碍不使用工具的数学推理。有一个猜测是，由于DeepSeek-LLM 1.3B的规模有限，它缺乏同时充分吸收代码和数学数据的能力。

# 5.1.2. ArXiv 论文似乎对提高数学推理效果有限

ArXiv 论文通常作为数学预训练数据的一部分（Azerbayev 等，2023；Lewkowycz 等，2022a；Polu 和 Sutskever，2020；Wang 等，2023c）。然而，关于它们对数学推理影响的详细分析尚未进行广泛研究。或许与直觉相反，根据我们的实验，arXiv 论文似乎对提高数学推理效果不佳。我们对不同规模的模型进行了实验，包括 DeepSeek-LLM 1.3B 和 DeepSeek-Coder-Base-v1.5 7B（Guo 等，2024），使用了经过多种处理流程的 arXiv 语料库：

•MathPile（王等，2023c）：一个8.9B令牌的语料库，采用清理和过滤启发式规则开发，其中超过85%为科学arXiv论文；• ArXiv-RedPajama（计算机，2023）：去除了前言、评论、宏和参考文献的完整arXiv LaTeX文件，总计28.0B令牌。

在我们的实验中，我们分别对DeepSeek-LLM 1.3B进行了150B tokens的训练，对DeepSeekCoder-Base-v1.5 7B进行了40B tokens的训练，使用每个arXiv语料库。似乎arXiv论文在提升数学推理方面无效。当仅在arXiv语料库上进行训练时，这两个模型在本研究中使用的不同复杂度的各种数学基准测试中均未显示出显著的改进，甚至有所下降。这些基准测试包括像GSM8K和MATH（表8）这样的定量推理数据集、像MMLU-STEM（表8）这样的多项选择挑战，以及像miniF2F（表9）这样的正式数学。

然而，这一结论有其局限性，应持谨慎态度。我们尚未研究：

arXiv 令牌对本研究未包含的特定数学相关任务的影响，例如定理的非正式化，即将形式陈述或证明转换为非正式版本；arXiv 令牌与其他类型数据结合时的效果；arXiv 论文的收益是否会在更大模型规模上显现。

因此，需要进一步的探索，我们将其留待未来的研究。

# 5.2. 强化学习的见解

# 5.2.1. 朝着统一范式迈进

在这一部分，我们提供一种统一的范式来分析不同的训练方法，例如SFT、RFT、DPO、PPO、GRPO，并进一步进行实验以探索统一范式的因素。一般来说，训练方法的参数$\theta$的梯度可以写作：

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { T } _ { \mathcal { R } } ( \boldsymbol { \theta } ) = \mathbb { E } [ \underbrace { ( \boldsymbol { q } , \boldsymbol { o } ) \sim \mathcal { D } } _ { D a t a \ S o u r c e } ] \left( \frac { 1 } { | \ o | } \sum _ { t = 1 } ^ { | o | } \underbrace { G C _ { \mathcal { R } } ( \boldsymbol { q } , \boldsymbol { o } , t , \pi _ { r f } ) } _ { G r a d i e n t \ C o e f f i c i e n t } \nabla _ { \boldsymbol { \theta } } \log \pi _ { \boldsymbol { \theta } } ( o _ { t } | \boldsymbol { q } , \boldsymbol { o } _ { < t } ) \right) .
$$

存在三个关键组件：1) 数据源 $\mathcal { D }$，决定训练数据；2) 奖励函数 $\pi _ { r f }$，是训练奖励信号的来源；3) 算法 $\mathcal { A }$，处理训练数据和奖励信号以生成梯度系数 GC，决定数据的惩罚或强化的大小。我们分析基于这种统一范式的几种代表性方法：

•监督微调（SFT）：SFT在人工选择的SFT数据上对预训练模型进行微调。

<table><tr><td>方法</td><td>数据来源</td><td>奖励函数</td><td>梯度系数</td></tr><tr><td>SFT</td><td>q, 0 ∼ Pt(Q, 0)</td><td></td><td>1</td></tr><tr><td>RFT</td><td>q ∼ P(Q), 0 ∼ π(0|q)</td><td>规则</td><td>公式10</td></tr><tr><td>DPO</td><td>q ∼ (Q), +, 0− ∼ (|q)</td><td>规则</td><td>公式14</td></tr><tr><td>在线RFT</td><td>q ∼ P(Q), 0 ∼ πθ(0|q)</td><td>规则</td><td>公式10</td></tr><tr><td>PPO</td><td>q ∼ Pst(Q), 0 ∼ πθ(0|q)</td><td>模型</td><td>公式18</td></tr><tr><td>GRPO</td><td>q ~ Psft(Q), {0iG∼~ πθ(0|q)</td><td>模型</td><td>公式21</td></tr></table>

表10 | 不同方法的数据来源和梯度系数。$P _ { s f t }$表示监督微调数据集的数据分布。$\pi _ { \theta _ { s f t } }$和$\pi _ { \theta }$分别表示在线训练过程中的监督微调模型和实时策略模型。

![](images/5.jpg)  
Figure 5 | Performance of the DeepSeekMath-Instruct 1.3B model, which was further trained using various methods, on two benchmarks.

• 拒绝采样微调（RFT）：RFT在基于SFT问题的SFT模型采样的过滤输出上进一步微调SFT模型。RFT根据答案的正确性过滤输出。

直接偏好优化（DPO）：DPO通过在从SFT模型采样的增强输出上微调T模型，进一步优化T模型，使用成对DPO损失。

•在线拒绝采样微调（Online RFT）：与RFT不同，在线RFT使用SFT模型初始化策略模型，并通过与实时策略模型采样的增强输出进行微调来进行精炼。

PPO/GRPO：PPO/GRPO使用SFT模型初始化策略模型，并通过从实时策略模型中采样的输出进行强化。

我们在表10中总结了这些方法的组成部分。有关更详细的推导过程，请参阅附录A.1。

关于数据源的观察，我们将数据源分为两类：在线采样和离线采样。在线采样指的是训练数据来自实时训练策略模型的探索结果，而离线采样则指训练数据来自初始SFT模型的采样结果。RFT和DPO采用离线风格，而在线RFT和GRPO则采用在线风格。

![](images/6.jpg)  
Figure 6 | Performance of iterative reinforcement learning with DeepSeekMath-Instruct 7B or two benchmarks.

如图5所示，我们发现在线RFT在两个基准上显著优于RFT。具体而言，在线RFT在训练的早期阶段与RFT相当，但在后期阶段获得了绝对优势，展示了在线训练的优越性。这是直观的，因为在初始阶段，执行者和SFT模型表现得很相似，采样的数据只显示出细微的差异。然而，在后期阶段，从执行者中采样的数据将显示出更显著的差异，实时数据采样将带来更大的优势。

关于梯度系数的观察 该算法处理奖励信号以获取梯度系数，从而更新模型参数。我们在实验中将奖励函数分为“规则”和“模型”。规则指的是根据答案的正确性来判断响应的质量，而模型表示我们训练一个奖励模型来对每个响应进行评分。奖励模型的训练数据基于规则判断。方程10和21突出显示了GRPO与在线RFT之间的一个关键区别：GRPO根据奖励模型提供的奖励值独特地调整其梯度系数。这允许根据响应的不同大小进行差异化的强化和惩罚。相比之下，在线RFT缺乏这一特性；它不惩罚错误响应，并对所有正确答案的响应以相同的强度进行统一的强化。

如图5所示，GRPO超越了在线RFT，突显了改变正负梯度系数的效率。此外，$\mathrm{G R P O + P S}$的表现优于${\mathrm{G R P O + O S}}$，这表明使用细粒度、步意识梯度系数的好处。此外，我们在实验中探讨了迭代强化学习，进行两轮迭代。如图6所示，我们注意到迭代强化学习显著提高了性能，尤其是在第一次迭代时。

![](images/7.jpg)  
Figure 7 | The Maj $@ \mathrm { K }$ and Pass $@ \mathrm { K }$ of SFT and RL DeepSeekMath 7B on GSM8K and MATH (temperature 0.7). It was noted that RL enhances Maj $@ \mathrm { K }$ but not Pass@K.

# 5.2.2. 为什么强化学习有效？

在本文中，我们基于一部分指令调优数据进行强化学习，取得了显著的性能提升，相较于指令调优模型。为了进一步解释为什么强化学习有效，我们评估了Instruct和RL模型在两个基准测试中的Pass@K和Maj@K准确率。如图7所示，RL增强了Maj@K的性能，但未能提升Pass@K。这些发现表明，RL通过使输出分布更加稳健来增强模型的整体性能，换句话说，似乎这种提升源于提高TopK中正确响应的比率，而不是基本能力的增强。同样，(Wang et al., 2023a)在SFT模型的推理任务中发现了不一致问题，表明可以通过一系列的偏好对齐策略（Song et al., 2023；Wang et al., 2023a；Yuan et al., 2023b）来改善SFT模型的推理性能。

# 5.2.3. 如何实现更有效的强化学习？

我们证明了强化学习在数学推理任务中效果很好。我们还提供了一个统一的范式来理解不同的代表性训练方法。在这个范式中，所有方法被概念化为直接或简化的强化学习技术。正如公式5所总结的，存在三个关键组件：数据源、算法和奖励函数。我们提供了一些关于这三个组件的潜在未来方向。

数据源 数据源是所有训练方法的原材料。在${ \mathrm { R L } } $的背景下，我们特别将数据源称为从策略模型中采样的未标记问题及其输出。本文中，我们仅使用来自指令微调阶段的问题，并采用简单的核采样来采样输出。我们认为这是我们RL流程仅提高Maj $@ \mathrm { K }$性能的一个潜在原因。在未来，我们将结合先进的采样（解码）策略，探索我们的RL流程在分布外问题提示上的应用，如基于树搜索的方法（Yao等，2023）。此外，效率推理技术（Kwon等，2023；Leviathan等，2023；Xia等，2023, 2024）决定了策略模型的探索效率，也发挥着极其重要的作用。

算法 算法处理数据和奖励信号以更新模型参数的梯度系数。根据公式5，在某种程度上，所有方法现在都完全信任奖励函数的信号，以增加或减少某个令牌的条件概率。然而，无法确保奖励信号始终可靠，尤其是在极其复杂的任务中。例如，即使是经过训练有素的注释者仔细标注的PRM800K数据集（Lightman等，2023），仍然包含约20%的错误注释。为此，我们将探索对噪声奖励信号稳健的强化学习算法。我们相信，这种弱到强的（Burns等，2023）对齐方法将对学习算法带来根本性的变化。

奖励函数 奖励函数是训练信号的来源。在强化学习中，奖励函数通常是神经奖励模型。我们认为奖励模型存在三个重要方向：1）如何增强奖励模型的泛化能力。奖励模型必须有效泛化，以处理分布外的问题和高级解码输出；否则，强化学习可能仅仅稳定大型语言模型的分布，而不是提升其基本能力；2）如何反映奖励模型的不确定性。不确定性可能作为弱奖励模型和弱到强学习算法之间的联系桥梁；3）如何高效构建高质量的过程奖励模型，以为推理过程提供细粒度的训练信号（Lightman等，2023；Wang等，2023b）。

# 6. 结论、局限性与未来工作

我们提出了DeepSeekMath，它在竞争级MATH基准上超过了所有开源模型，并接近闭源模型的表现。DeepSeekMath以DeepSeek-Coder-v1.5 7B为基础初始化，并进行了500B标记的持续训练，其中训练数据的一个重要组成部分是来源于Common Crawl的120B数学标记。我们的广泛消融研究表明，网页提供了高质量数学数据的显著潜力，而arXiv可能没有我们预期的那么有益。我们引入了组相对策略优化（GRPO），这是近端策略优化（PPO）的一个变体，它可以显著提高数学推理能力，同时减少内存消耗。实验结果表明，即使DeepSeekMath-Instruct 7B在基准测试中已达到高分，GRPO仍然有效。我们还提供了一个统一的范式，以理解一系列方法，并总结出几条可能的方向，以实现更有效的强化学习。

尽管DeepSeekMath在定量推理基准测试中取得了令人印象深刻的分数，但它在几何和定理证明方面的能力相对于闭合模型更弱。例如，在我们的干运行中，该模型无法处理与三角形和椭圆相关的问题，这可能表明在预训练和微调中存在数据选择偏差。此外，由于模型规模的限制，DeepSeekMath在少量样本能力上不如GPT-4。GPT-4能够通过少量样本输入提高其性能，而DeepSeekMath在零样本和少量样本评估中表现相似。未来，我们将进一步改进我们的工程数据选择流程，以构建更高质量的预训练语料库。此外，我们将探索更有效的LLM强化学习的潜在方向（第5.2.3节）。

参考文献  
R. Anil, S. Borgeaud, Y. Wu, J. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, D. Silver, S. Petrov, M. Johnson, I. Antonoglou, J. Schrittwieser, A. Glaese, J. Chen, EPtr, T.P. Lilicrp, A. Lazarou, O. Firat, J. Molloy, M. Isard, P.R. Barham, T. Henan, B. Lee, F. Viola, M. Reynolds, Y. Xu, R. Doherty, E. Collins, C. Meyer, E. Rutherford, E. Moreira, K. Ayoub, M. Goel, G. Tucker, E. Piqueras, M. Krikun, I. Barr, N. Savinov, I. Danihelka, B. Roelofs, A. White, A. Andreassen, T. von Glehn, L. Yagati, M. Kazemi, L. Gonzalez, M. Khalman, J. Sygnowski, 等。Gemini：一个高能力的多模态模型家族。CoRR, abs/2312.11805, 2023. doi: 10.48550/ARXIV.2312.11805. URL https://doi.org/10.48550/arXiv.2312.11805。  
J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. Cai, M. Terry, Q. Le, 等。使用大型语言模型进行程序合成。arXiv预印本 arXiv:2108.07732, 2021。  
Z. Azerbayev, H. Schoelkopf, K. Paster, M. D. Santos, S. McAleer, A. Q. Jiang, J. Deng, S. Biderman, 和 S. Welleck。Llemma：一个开放的数学语言模型。arXiv预印本 arXiv:2310.10631, 2023。  
J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang, 等。Qwen技术报告。arXiv预印本 arXiv:2309.16609, 2023。  
C. Burns, P. Izmailov, J. H. Kirchner, B. Baker, L. Gao, L. Aschenbrenner, Y. Chen, A. Ecoffet, M. Joglekar, J. Leike, 等。弱到强的泛化：通过弱监督引发强能力。arXiv预印本 arXiv:2312.09390, 2023。  
ChatGLM3团队。Chatglm3系列：开放的双语聊天语言模型，2023。URL https://github.com/THUDM/ChatGLM3。  
M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tag, I. Babuschkin, S. Balaji S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, 和 W. Zaremba。评估在代码上训练的大型语言模型。CoRR, abs/2107.03374, 2021。URL https://arxiv.org/abs/2107.03374。  
e, X. M, X. ... th D计算来自推理以解决数值推理任务。CoRR, abs/2211.12588, 2022。doi: 10.48550/ARXIV.2211.12588。URL https://doi.org/10.48550/arXiv.2211.12588。  
K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, 等。训练验证器以解决数学文字问题。arXiv预印本 arXiv:2110.14168, 2021。  
T. Computer。Redpajama：一个用于训练大型语言模型的开放数据集，2023年10月。URL https://github.com/togethercomputer/RedPajama-Data。  
DeepSeek-AI。Deepseek LLM：利用长期主义扩大开源语言模型。CoRR, abs/2401.02954, 2024。doi: 10.48550/ARXIV.2401.02954。URL https://doi.org/10.48550/arXiv.2401.02954。

Z. Du, Y. Qian, X. Liu, M. Ding, J. Qiu, Z. Yang, 和 J. Tang. Glm: 采用自回归空白填充的通用语言模型预训练. 在第60届计算语言学协会年会会议论文集（卷1：长篇论文）中, 第320335页, 2022.

L. Gao, A. Madaan, S. Zhou, U. Alon, P. Liu, Y. Yang, J. Callan, 和 G. Neubig。PAL：程序辅助语言模型。在 A. Krause、E. Brunskill、K. Cho、B. Engelhardt、S. Sabato 和 J. Scarlett 编者的《国际机器学习大会，ICML 2023》，2023年7月23日至29日，美国夏威夷檀香山，机器学习研究会议集，第202卷，页面10764-10799。PMLR，2023。网址 https://proceedings.mlr.press/v202/gao23f.html。

Z. Gou, Z. Shao, Y. Gong, Y. Shen, Y. Yang, M. Huang, N. Duan, 和 W. Chen. Tora：一个用于数学问题解决的工具集成推理代理. CoRR, abs/2309.17452, 2023. doi: 10.48550/ARXIV.2309.17452. URL https://doi.org/10.48550/arXiv.2309.17452.

D. Guo, Q. Zhu, D. Yang, Z. Xie, K. Dong, W. Zhang, G. Chen, X. Bi, Y. Wu, Y. K. Li, F. Luo, Y. Xiong, 和 W. Liang. Deepseek-coder：当大型语言模型遇上编程，代码智能的崛起，2024年。

D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, 和 J. Steinhardt. 测量大规模多任务语言理解. arXiv 预印本 arXiv:2009.03300, 2020.

D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, 和 J. Steinhardt. 使用数学数据集测量数学问题解决能力. arXiv 预印本 arXiv:2103.03874, 2021.

高飞者。Hai-llm: , 2023. URL https://www.high-flyer.n/en/blog/hai-llm。

Inflection AI. Inflection-2, 2023. URL https://inflection.ai/inflection-2.

A. Q. Jiang, S. Welleck, J. P. Zhou, W. Li, J. Liu, M. Jamni, T. Lacroix, Y. Wu, 和 G. Lample. 草拟、草图和证明：用非正式证明指导形式化定理证明器. arXiv 预印本 arXiv:2210.12283, 2022.

A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. d. 1. Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier，等。Mistral 7b。arXiv预印本arXiv:2310.06825，2023年。

A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou 和 T. Mikolov. Fasttext.zip：压缩文本分类模型. arXiv预印本 arXiv:1612.03651, 2016.

W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J.E. Gonzalez, H. Zhang, 和 I. Stoica. 使用分页注意力的用于大型语言模型服务的高效内存管理. 见于2023年ACM SIGOPS第29届操作系统原理研讨会论文集.

Y. Leviathan, M. Kalman, 和 Y. Matias. 通过推测解码从变压器中快速推理. 在国际机器学习会议, 第19274-19286页. PMLR, 2023.

A. Lewkowycz, A. Andreassen, D. Dohan, E. Dyer, H. Michalewski, V. Ramasesh, A. Slone, C. Anil, I. Schlag, T. Gutman-Solo 等. 使用语言模型解决定量推理问题. 神经信息处理系统进展, 35:38433857, 2022a.

A. Lewkowycz, A. Andreassen, D. Dohan, E. Dyer, H. Michalewski, V. V. Ramasesh, A. Slone, C. Anil, I. Schlag, T. Gutman-Solo, Y. Wu, B. Neyshabur, G. Gur-Ari, 和 V. Misra. 用语言模型解决定量推理问题. 在 S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, 和 A. Oh 编辑的《神经信息处理系统进展 35：2022 年神经信息处理系统年度会议，NeurIPS 2022，美国路易斯安那州新奥尔良，2022 年 11 月 28 日 - 12 月 9 日》，2022b. URL http://papers.nips.cc/paper_files/paper/2022/hash/18abbeef8cfe9203fdf9053c9c4fe191-Abstract-Conference.html.

H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, 和 K. Cobbe. 让我们逐步验证。arXiv 预印本 arXiv:2305.20050, 2023。

I. Loshchilov 和 F. Hutter. 解耦权重衰减正则化. arXiv 预印本 arXiv:1711.05101, 2017.

H. Luo, Q. Sun, C. Xu, P. Zhao, J. Lou, C. Tao, X. Geng, Q. Lin, S. Chen, 和 D. Zhang. Wizardmath: 通过增强进化指令为大型语言模型赋能数学推理. arXiv 预印本 arXiv:2308.09583, 2023.

S. Mishra, M. Finlayson, P. Lu, L. Tang, S. Welleck, C. Baral, T. Rajpurohit, O. Tafjord, A. Sabharwal, P. Clark, 和 A. Kalyan。LILA：一个统一的数学推理基准。在 Y. Goldberg, Z. Kozareva, 和 Y. Zhang 主编的《2022年自然语言处理实证方法会议论文集》，EMNLP 2022，阿布扎比，阿拉伯联合酋长国，2022年12月7日至11日，第5807-5832页。计算语言学协会，2022。doi: 10.18653/V1/2022.EMNLP-MAIN.392。网址 https://doi.org/10.18653/v1/2022.emnlp-main.392。

X. Nguyen，W. Zhang，X. Li，M. M. Aljunid，Q. Tan，L. Cheng，G. Chen，Y. Deng，S. Yang，C. Liu，H. Zhang 和 L. Bing. Seallms - 东南亚的大规模语言模型. CoRR, abs/2312.00738, 2023. doi: 10.48550/ARXIV.2312.00738. URL https : //doi . org/10 . 485 50/arXiv.2312.00738.

OpenAI. GPT4技术报告。arXiv预印本 arXiv:2303.08774，2023。

L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray 等. 通过人类反馈训练语言模型以遵循指令. 神经信息处理系统进展, 35:2773027744, 2022.

K. Paster, M. D. Santos, Z. Azerbayev, 和 J. Ba. Openwebmath: 一个高质量数学网页文本的开放数据集. CoRR, abs/2310.06786, 2023. doi: 10.48550/ ARXIV.2310.06786. URL https://doi.org/10.48550/arXiv.2310.06786.

L. C. Paulson. 三年的大锤经验：自动和交互定理证明器之间的实用链接。在R. A. Schmidt、S. Schulz和B. Konev编辑的《第二届自动推理实践方面研讨会论文集》，PAAR-2010，2010年7月14日，苏格兰爱丁堡，EPiC计算系列第9卷，第110页。EasyChair, 2010。doi: 10.29007/TNFD。URL https://doi.org/10.29007/tnfd。

S. Polu 和 I. Sutskever。用于自动定理证明的生成语言建模。CoRR，abs/2009.03393，2020。网址 https://arxiv.org/abs/2009.03393。

R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning 和 C. Finn。《直接偏好优化：你的语言模型秘密是一个奖励模型》。2023。

J. Schulman. 近似KL散度，2020年。网址 http://joschu.net/blog/k1-approx.html。  
J. Schulman, P. Moritz, S. Levine, M. Jordan, 与 P. Abbeel. 使用广义优势估计的高维连续控制。arXiv预印本 arXiv:1506.02438，2015年。  
J. Schulman, F. Wolski, P. Dhariwal, A. Radford, 与 O. Klimov. 近端策略优化算法。arXiv预印本 arXiv:1707.06347，2017年。  
F. Shi, M. Suzgun, M. Freitag, X. Wang, S. Srivats, S. Vosoughi, H. W. Chung, Y. Tay, S. Ruder, D. Zhou, D. Das, 与 J. Wei. 语言模型是多语言的链式思维推理者。在第十一届国际学习表征会议，ICLR 2023, Kigali, Rwanda, 2023年5月1-5日。OpenReview.net, 2023年。网址 https://openreview.net/pdf?id=fR3wGCk-IXp。  
F. Song, B. Yu, M. Li, H. Yu, F. Huang, Y. Li, 与 H. Wang. 人类对齐的偏好排名优化。arXiv预印本 arXiv:2306.17492，2023年。  
M. Suzun, N. Scales, N. Schärli, S. Gehrmann, Y. Tay, H. W. Chun, A. Chowdhery, Q. V. Le, E. H. Chi, D. Zhou, 等等. 挑战大基准任务及链式思维是否能够解决这些任务。arXiv预印本 arXiv:2210.09261，2022年。  
T. Tao. 拥抱变化并重设期望，2023年。网址 https://unlocked.microsoft.com/ai-anthology/terence-tao/。  
H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. Canton-Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. S. Koura, M. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M. Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan, I Zrov, Y. Zhan, A. Fan, M. Kambar, S. Nara, A. Roru, R. Stojic, S. Eu 和 T. Scialom. Llama 2: 开放的基础和微调的聊天模型。CoRR，abs/2307.09288，2023年。doi: 10.48550/arXiv.2307.09288。网址 https://doi.org/10.48550/arXiv.2307.09288。  
T. H. Trin, Y. Wu, Q. V. Le, H. He, 与 T. Luong. 在没有人类演示的情况下解决奥林匹克几何。自然，625(7995):476-482, 2024年。  
P Wn, L. Li, L. Chen, F. Son B. Lin Y. o T. Li, 和 Z. ui. 通过对齐提升更好的推理能力。arXiv预印本 arXiv:2309.02144，2023年。  
P. Wang, L. Li, Z. Shao, R. Xu, D. Dai, Y. Li, D. Chen, Y. Wu, 和 Z. Sui. 数学牧羊人：逐步验证和强化LLMs，无需人类注释。CoRR，abs/2312.08935，2023年。  
Z. Wang, R. Xia, 和 P. Liu. 数学生成AI：第一部分 - mathpile：用于数学的十亿标记规模预训练语料库。CoRR，abs/2312.17120，2023年。doi: 10.48550/ARXIV.2312.17120。网址 https://doi.org/10.48550/arXiv.2312.17120。  
J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. H. Chi, Q. V. Le, 和 D. Zhou. 链式思维提示在大型语言模型中引发推理。在NeurIPS, 2022。网址 http://papers.nips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html。

T. Wei，J. Luan，W. Liu，S. Dong，和B. Wang。Cmath：你的语言模型能通过中国小学数学测试吗？，2023。

M. Wenzel, L. C. Paulson 和 T. Nipkow. isabelle 框架. 在 O. A. Mohamed, C. A. Muñoz 和 S. Tahar 编辑的《高阶逻辑中的定理证明，第21届国际会议，TPHOLs 2008，加拿大蒙特利尔，2008年8月18-21日。会议录，计算机科学讲义笔记第5170卷，页码3338. Springer, 2008. doi: 10.1007/978-3-540-71067-7\_7. URL https://doi.org/10.1007/978-3-540-71067-7_7.

H. Xia, T. Ge, P. Wang, S.-Q. Chen, F. Wei, 和 Z. Sui。推测解码：利用推测执行加速 seq2seq 生成。在 H. Bouamor, J. Pino, 和 K. Bali 编辑的《计算语言学协会发现：EMNLP 2023》中，第3909-3925页，新加坡，2023年12月。计算语言学协会。doi: 10.18653/v1/2023.findings-emnlp.257。网址 https://aclanthology.org/2023.findings-emnlp.257。

H. Xia, Z. Yang, Q. Dong, P. Wang, Y. Li, T. Ge, T. Liu, W. Li, 和 Z. Sui. 在大型语言模型推理中的解锁效率：关于投机解码的综合调查。arXiv 预印本 arXiv:2401.07851, 2024。

S. Yao, D. Yu, J. Zhao, I. Shafran, T. L. Griffiths, Y. Cao, 和 K. Narasimhan. 思维树：使用大型语言模型进行深思熟虑的问题解决. arXiv 预印本 arXiv:2305.10601, 2023.

L. Yu, W. Jiang, H. Shi, J. Yu, Z. Liu, Y. Zhang, J. T. Kwok, Z. Li, A. Weller, 和 W. Liu. Metamath: 为大型语言模型自助生成数学问题. CoRR, abs/2309.12284, 2023. doi: 10.48550/ARXIV.2309.12284. URL https://doi.org/10.48550/arXiv.2309.12284.

Z. Yuan, H. Yuan, C. Li, G. Dong, C. Tan, 和 C. Zhou. 使用大型语言模型进行数学推理学习的规模关系. arXiv 预印本 arXiv:2308.01825, 2023a

Z. Yuan, H. Yuan, C. Tan, W. Wang, S. Huang, 和 F. Huang. Rrhf: 排名响应以无痛方式将语言模型与人类反馈对齐. arXiv 预印本 arXiv:2304.05302, 2023b.

X. 岳, X. 曲, G. 张, Y. 付, W. 黄, H. 孙, Y. 苏, 和 W. 陈. Mammoth: 通过混合指令调优构建数学通用模型. CoRR, abs/2309.05653, 2023. doi: 10.48550/ARXIV.2309.05653. URL https://doi.org/10.48550/arXiv.2309.05653.

K. Zheng, J. M. Han，和 S. Polu. Minif2f: 一个用于正式奥林匹克级数学的跨系统基准测试. arXiv 预印本 arXiv:2109.00110, 2021.

W. Zhong, R. Cui, Y. Guo, Y. Liang, S. Lu, Y. Wang, A. Saied, W. Chen, 和 N. Duan. AGIEval: 一个以人为中心的基准，用于评估基础模型. CoRR, abs/2304.06364, 2023. doi: 10.48550/arXiv.2304.06364. URL https ://doi.org/10 .48550/arXiv.2304.06364.

# A. 附录

# A.1. 强化学习分析

我们提供各种方法的数据来源和梯度系数的详细推导，包括SFT、RFT、在线RFT、DPO、PPO和GRPO。

# A.1.1. 监督微调

监督微调的目标是最大化以下目标：

$$
\mathcal { T } _ { S F T } ( \theta ) = \mathbb { E } \left[ q , o \sim P _ { s f t } ( Q , O ) \right] \left( \frac { 1 } { | o | } \sum _ { t = 1 } ^ { | o | } \log \pi _ { \theta } ( o _ { t } | q , o _ { < t } ) \right) .
$$

$\mathcal { T } _ { S F T } ( \theta )$的梯度是：

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { T } _ { S F T } = \mathbb { E } [ q , o \sim P _ { s f t } ( Q , O ) ] \left( \frac { 1 } { | o | } \sum _ { t = 1 } ^ { | o | } \nabla _ { \boldsymbol { \theta } } \log \pi _ { \boldsymbol { \theta } } ( o _ { t } | q , o _ { < t } ) \right) .
$$

数据来源：用于SFT的数据集。奖励函数：这可以视为人类选择。梯度系数：始终设为1。

# A.1.2. 拒绝采样微调

拒绝采样微调首先为每个问题从监督微调的LLM中采样多个输出，然后在带有正确答案的采样输出上训练LLM。正式地，RFT的目标是最大化以下目标：

$$
\mathcal { T } _ { R F T } ( \theta ) = \mathbb { E } [ q \sim P _ { s f t } ( Q ) , o \sim \pi _ { s f t } ( O | q ) ] \left( \frac { 1 } { | o | } \sum _ { t = 1 } ^ { | o | } \mathbb { I } ( o ) \log \pi _ { \theta } ( o _ { t } | q , o _ { < t } ) \right) .
$$

$\mathcal { T } _ { R F T } ( \theta )$的梯度是：

$$
\nabla _ { \theta } \mathcal { T } _ { R F T } ( \theta ) = \mathbb { E } [ q \sim P _ { s f t } ( Q ) , o \sim \pi _ { s f t } ( O | q ) ] \left( \frac { 1 } { | o | } \sum _ { t = 1 } ^ { | o | } \mathbb { I } ( o ) \nabla _ { \theta } \log \pi _ { \theta } ( o _ { t } | q , o _ { < t } ) \right) .
$$

数据来源：从SFT模型中抽取输出的SFT数据集中的问题。奖励函数：规则（答案是否正确）。梯度系数：

$$
G C _ { R F T } ( q , o , t ) = \mathbb { I } ( o ) = \left\{ { 1 \atop { 0 } } \right. \quad \mathrm { t h e ~ a n s w e r ~ o f ~ o ~ i s ~ c o r r e c t }
$$

# A.1.3. 在线拒绝采样微调

RFT和在线RFT之间的唯一区别在于，在线RFT的输出是从实时策略模型$\pi _ { \theta }$中抽样，而不是从SFT模型$\pi _ { \theta _ { s f t } }$中抽样。因此，在线RFT的梯度是：

$$
\nabla _ { \theta } \mathcal { T } _ { o n R F T } ( \theta ) = \mathbb { E } [ q \sim P _ { s f t } ( Q ) , o \sim \pi _ { \theta } ( O | q ) ] \left( \frac { 1 } { | o | } \sum _ { t = 1 } ^ { | o | } \mathbb { I } ( o ) \nabla _ { \theta } \log \pi _ { \theta } ( o _ { t } | q , o _ { < t } ) \right) .
$$

# A.1.4. 直接偏好优化 (DPO)

DPO的目标是：

$$
\mathcal { T } _ { D P O } ( \theta ) = \mathbb { E } [ q \sim P _ { s f t } ( Q ) , o ^ { + } , o ^ { - } \sim \pi _ { s f t } ( O | q ) ] \log \sigma \left( \beta \frac { 1 } { \lvert o ^ { + } \rvert } \sum _ { t = 1 } ^ { \lvert o ^ { + } \rvert } \log \frac { \pi _ { \theta } ( o _ { t } ^ { + } \lvert q , o _ { \ \cdot \epsilon } ^ { + } ) } { \pi _ { \mathrm { r e f } } ( o _ { t } ^ { + } \lvert q , o _ { \ \cdot \epsilon } ^ { + } ) } - \beta \frac { 1 } { \lvert o ^ { - } \rvert } \sum _ { t = 1 } ^ { \lvert o ^ { - } \rvert } \log \frac { \pi _ { \theta } ( o _ { \ \cdot \epsilon } ^ { - } \lvert q , o _ { \ \cdot \epsilon } ^ { - } ) } { \pi _ { \mathrm { r e f } } ( o _ { \ \ \epsilon } ^ { - } \lvert q , o _ { \ \cdot \epsilon } ^ { - } ) } \right)
$$

$\mathcal { T } _ { D P O } ( \theta )$ 的梯度是：

$$
\begin{array} { r l } & { \nabla _ { \theta } \mathcal { T } _ { D P O } ( \theta ) = \mathbb { E } [ q \sim P _ { s f t } ( Q ) , o ^ { + } , o ^ { - } \sim \pi _ { s f t } ( O | q ) ] \left( \frac { 1 } { | o ^ { + } | } \displaystyle \sum _ { t = 1 } ^ { | o ^ { + } | } G C _ { D P O } ( q , o , t ) \nabla _ { \theta } \log \pi _ { \theta } ( o _ { t } ^ { + } | q , o _ { < t } ^ { + } ) \right. } \\ & { ~ \left. - \frac { 1 } { | o ^ { - } | } \displaystyle \sum _ { t = 1 } ^ { | o ^ { - } | } G C _ { D P O } ( q , o , t ) \nabla _ { \theta } \log \pi _ { \theta } ( o _ { t } ^ { - } | q , o _ { < t } ^ { - } ) \right) } \end{array}
$$

数据来源：SFT数据集中问题，输出来自SFT模型。奖励函数：一般领域中的人类偏好（在数学任务中可以是“规则”）。梯度系数：

$$
G C _ { D P O } ( q , o , t ) = \sigma \left( \beta \log \frac { \pi _ { \theta } ( o _ { t } ^ { - } | q , o _ { < t } ^ { - } ) } { \pi _ { \mathrm { r e f } } ( o _ { t } ^ { - } | q , o _ { < t } ^ { - } ) } - \beta \log \frac { \pi _ { \theta } ( o _ { t } ^ { + } | q , o _ { < t } ^ { + } ) } { \pi _ { \mathrm { r e f } } ( o _ { t } ^ { + } | q , o _ { < t } ^ { + } ) } \right)
$$

# A.1.5. 近端策略优化 (PPO)

PPO的目标是：

$$
\mathcal { T } _ { P P O } ( \theta ) = \mathbb { E } [ q \sim P _ { s f i } ( Q ) , o \sim \pi _ { \theta \sim d } ( O | q ) ] \frac { 1 } { | \sigma | } \sum _ { t = 1 } ^ { | \mathfrak { c } | } \operatorname* { m i n } \left[ \frac { \pi _ { \theta } ( o _ { t } | q , o _ { < t } ) } { \pi _ { \theta _ { o d } } ( o _ { t } | q , o _ { < t } ) } A _ { t } , \mathrm { c l i p } \left( \frac { \pi _ { \theta } ( o _ { t } | q , o _ { < t } ) } { \pi _ { \theta _ { o d } } ( o _ { t } | q , o _ { < t } ) } , 1 - \varepsilon , 1 + \varepsilon \right) A _ { t } \right] .
$$

为了简化分析，假设模型在每个探索阶段之后只进行一次更新，从而确保 $\pi _ { \theta _ { o l d } } = \pi _ { \theta }$ 。在这种情况下，我们可以去掉最小值和裁剪操作：

$$
\mathcal { T } _ { P P O } ( \theta ) = \mathbb { E } [ q \sim P _ { s f t } ( Q ) , o \sim \pi _ { \theta _ { o l d } } ( O | q ) ] \frac { 1 } { | o | } \sum _ { t = 1 } ^ { | o | } \frac { \pi _ { \theta } ( o _ { t } | q , o _ { < t } ) } { \pi _ { \theta _ { o l d } } ( o _ { t } | q , o _ { < t } ) } A _ { t } .
$$

$\mathcal { T } _ { P P O } ( \theta )$ 的梯度是：

$$
\nabla _ { \theta } \mathcal { T } _ { P P O } ( \theta ) = \mathbb { E } \big [ q \sim P _ { s f t } ( Q ) , o \sim \pi _ { \theta _ { o l d } } ( O | q ) \big ] \frac { 1 } { | o | } \sum _ { t = 1 } ^ { | o | } A _ { t } \nabla _ { \theta } \log \pi _ { \theta } ( o _ { t } | q , o _ { < t } )
$$

数据来源：SFT数据集中问题的输出从策略模型中采样。奖励函数：奖励模型。梯度系数：

$$
G C _ { P P O } ( q , o , t , \pi _ { \theta _ { r m } } ) = A _ { t } ,
$$

其中 $A _ { t }$ 是优势，通过应用广义优势估计（GAE）(Schulman 等人, 2015) 计算得出，基于奖励 $\left\{ r _ { \geq t } \right\}$ 和学习到的价值函数 $V _ { \psi }$。

# A.1.6. 群组相对策略优化 (GRPO)

GRPO的目标是（假设 $\pi _ { \theta _ { o l d } } = \pi _ { \theta }$ 以简化分析）：

$$
\begin{array} { l } { \displaystyle \mathcal { J } _ { G R P O } ( \theta ) = \mathbb { E } [ q \sim P _ { s f t } ( Q ) , \{ o _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { o d } } ( O | q ) ] } \\ { \displaystyle \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { | o _ { i } | } \sum _ { t = 1 } ^ { | o _ { i } | } \left[ \frac { \pi _ { \theta } ( o _ { i , t } | q , o _ { i , < t } ) } { \pi _ { \theta _ { o d d } } ( o _ { i , t } | q , o _ { i , < t } ) } \hat { A } _ { i , t } - \beta ( \frac { \pi _ { r e f } ( o _ { i , t } | q , o _ { i , < t } ) } { \pi _ { \theta } ( o _ { i , t } | q , o _ { i , < t } ) } - \log \frac { \pi _ { r e f } ( o _ { i , t } | q , o _ { i , < t } ) } { \pi _ { \theta } ( o _ { i , t } | q , o _ { i , < t } ) } - 1 ) \right] . } \end{array}
$$

$\mathcal { T } _ { G R P O } ( \theta )$ 的梯度是：

$$
\begin{array} { l } { \nabla _ { \theta } \mathcal { T } _ { G R P O } ( \theta ) = \mathbb { E } [ q \sim P _ { s f t } ( Q ) , \{ o _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { o l d } } ( O | q ) ] } \\ { \displaystyle \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { | o _ { i } | } \sum _ { t = 1 } ^ { | o _ { i } | } \left[ \hat { A } _ { i , t } + \beta \left( \frac { \pi _ { r e f } ( o _ { i , t } | o _ { i , < t } ) } { \pi _ { \theta } ( o _ { i , t } | o _ { i , < t } ) } - 1 \right) \right] \nabla _ { \theta } \log \pi _ { \theta } ( o _ { i , t } | q , o _ { i , < t } ) . } \end{array}
$$

数据来源：SFT数据集中问题的输出从策略模型中采样。奖励函数：奖励模型。梯度系数：

$$
G C _ { G R P O } ( q , o , t , \pi _ { \theta _ { r m } } ) = \hat { A } _ { i , t } + \beta \left( \frac { \pi _ { r e f } ( o _ { i , t } | o _ { i , < t } ) } { \pi _ { \theta } ( o _ { i , t } | o _ { i , < t } ) } - 1 \right) ,
$$

其中 $\hat { A } _ { i , t }$ 是基于组奖励分数计算得出的。