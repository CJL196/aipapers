# 1. 论文基本信息

## 1.1. 标题
**H`_2`O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models**  
（H`_2`O：用于大语言模型高效生成式推理的重击者预言机）

## 1.2. 作者
**Zhenyu Zhang** (德克萨斯大学奥斯汀分校), **Ying Sheng** (斯坦福大学), **Tianyi Zhou** (加州大学圣地亚哥分校), **Christopher Ré** (斯坦福大学), **Zhangyang Wang** (德克萨斯大学奥斯汀分校), **Beidi Chen** (Meta AI / 卡内基梅隆大学) 等。

## 1.3. 发表期刊/会议
发表于 **NeurIPS 2023** (根据 arXiv 发布时间和社区记录，这是一篇在 NeurIPS 2023 极具影响力的论文)。

## 1.4. 发表年份
2023 年

## 1.5. 摘要
大语言模型（LLMs）虽然在各种任务中表现出色，但其部署成本极高，主要瓶颈在于 <strong>KV Cache（键值缓存）</strong> 的显存占用随序列长度线性增长。本文观察到在注意力机制中，只有极少部分的 <strong>词元 (tokens)</strong> 对注意力分数的贡献最大，作者将这些关键词元称为 <strong>“重击者”</strong> (Heavy Hitters, $H_2$)。基于此发现，论文提出了 **H`_2`O**，一种新颖的 KV Cache 驱逐策略，通过动态保留最近的词元和历史上的 $H_2$ 词元，在保证模型精度几乎不下降的前提下，将 KV Cache 的显存占用减少了 5-10 倍，并将推理吞吐量提升了高达 29 倍。

## 1.6. 原文链接
*   **ArXiv:** https://arxiv.org/abs/2306.14048
*   **PDF:** https://arxiv.org/pdf/2306.14048v3.pdf

    ---

# 2. 整体概括

## 2.1. 研究背景与动机
*   **核心问题:** 大语言模型在生成长文本（如对话系统、故事写作）时，需要存储大量的 **KV Cache**（Key-Value Cache）。KV Cache 存储了之前所有词元的键（Key）和值（Value）向量，以避免重复计算。然而，KV Cache 的大小随着序列长度和批次大小（Batch Size）线性增长，导致显存（GPU Memory）迅速耗尽，严重限制了模型的推理效率和最大支持的上下文长度。
*   **现有挑战:** 现有的方法（如 Sparse Transformer）通常是在训练阶段设计的稀疏注意力机制，直接应用于预训练模型的推理阶段会导致严重的精度下降。简单的缓存策略（如只保留最近的词元）在长文本任务中会丢失关键信息。
*   **创新思路:** 作者通过实证研究发现，LLM 的注意力矩阵在推理时天然具有高度稀疏性（>95%），且存在明显的“二八定律”：少数特定的词元（重击者）承载了绝大部分的注意力权重。因此，可以通过识别并保留这些“重击者”来压缩 KV Cache。

## 2.2. 核心贡献与主要发现
*   **实证发现:** 发现了 **$H_2$ (Heavy Hitters)** 现象，即一小部分词元在生成过程中反复获得高注意力分数。这些词元通常与文本中频繁共现的词高度相关。
*   **方法创新:** 提出了 **H`_2`O (Heavy-Hitter Oracle)**，这是一种贪心驱逐策略。它不需要重新训练模型，而是在推理过程中动态地计算词元的累积注意力分数，仅保留最重要的 $H_2$ 词元和最近的局部词元。
*   **理论保证:** 将 KV Cache 的驱逐问题建模为 <strong>动态次模最大化 (Dynamic Submodular Maximization)</strong> 问题，并证明了该贪心算法具有理论上的性能下界保证。
*   **系统性能:** 在 OPT、LLaMA 和 GPT-NeoX 等模型上验证了 H`_2`O。在保持精度相当的情况下，显存占用减少可达 5 倍以上，吞吐量相比 DeepSpeed Zero-Inference 和 Hugging Face Accelerate 提升高达 29 倍。

    下图（原文 Figure 1）展示了 H`_2`O 框架的核心概览以及不同策略对注意力图（Attention Map）的保留情况。可以看出 H`_2`O 策略（右上角）有效地保留了注意力矩阵中的关键高亮区域。

    ![Figure:Upper plots illustrate symbolic plots o anattention map deploying different KV cache policies in LLM generation. Lower right: contrasts their accuracy-memory trade-off. Left: the overview of ${ \\sf H } _ { 2 } \\sf { O }$ framework.](images/1.jpg)
    *该图像是展示 H`_2`O 框架的示意图，包含四种稀疏性策略的对比，分别为动态稀疏性、静态稀疏性（分隔和局部）以及基于 H`_2`O 的静态稀疏性。下方公式 $ ext{Value} = ext{Key} imes ext{Query}$ 对 H`_2`O 的操作进行了说明，右侧图表展示了不同稀疏性策略的准确率与内存减少之间的关系。*

---

# 3. 预备知识与相关工作

## 3.1. 基础概念
为了理解本文，初学者需要掌握以下核心概念：

*   <strong>KV Cache (键值缓存):</strong> 在 Transformer 模型的自回归生成过程中，每生成一个新的词元，都需要计算它与之前所有词元的注意力。为了避免重复计算之前词元的 Key 和 Value 向量，系统会将它们缓存在 GPU 显存中，这就是 KV Cache。它是长文本推理的“显存杀手”。
*   <strong>Attention Mechanism (注意力机制):</strong> Transformer 的核心。对于查询 $Q$、键 $K$ 和值 $V$，注意力输出计算如下：
    $$
    \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    $$
    其中 $QK^T$ 产生的矩阵衡量了不同词元之间的相关性（注意力分数）。
*   <strong>Sparsity (稀疏性):</strong> 指矩阵中大部分元素为 0 或接近 0 的特性。在注意力机制中，稀疏性意味着当前词元只关注很少一部分的历史词元，而忽略大部分其他词元。
*   <strong>Eviction Policy (驱逐策略):</strong> 类似于操作系统中的缓存置换算法（如 LRU）。在显存有限的情况下，当 KV Cache 满了，我们需要决定“踢出”哪些旧的词元，以腾出空间给新词元，同时尽可能少地影响模型性能。

## 3.2. 前人工作与差异化
*   <strong>Sparse Attention (稀疏注意力):</strong> 如 Sparse Transformer、Reformer 等。这些工作通常侧重于**训练阶段**或需要特定的架构修改，旨在降低计算复杂度（从 $O(N^2)$ 降低）。
    *   **差异:** H`_2`O 专注于**推理阶段**，不需要修改模型架构或重新训练，直接应用于现有的预训练 LLM。
*   <strong>Quantization &amp; Pruning (量化与剪枝):</strong> 通过降低数值精度（如 INT8）或剪除模型权重来压缩模型。
    *   **差异:** H`_2`O 解决的是 **KV Cache** 的显存瓶颈，与模型权重的压缩是正交的，可以结合使用（如实验中结合了 4-bit 量化）。
*   <strong>Context Compression (上下文压缩):</strong> 如 Gisting tokens，尝试压缩 Prompt。
    *   **差异:** H`_2`O 是动态的逐词元（token-wise）驱逐，更加灵活且适用于生成过程。

        ---

# 4. 方法论

## 4.1. 方法原理
H`_2`O 的核心思想基于一个关键观察：**并非所有历史词元都同等重要**。在生成过程中，只有一小部分历史词元（即 $H_2$ 词元）持续获得较高的注意力分数。

因此，H`_2`O 维护一个固定大小的 KV Cache 预算（例如总长度的 20%）。在这个预算内，它保留两类词元：
1.  <strong>最近的词元 (Local Tokens):</strong> 也就是滑动窗口中的最新词元，用于保持局部语法的连贯性。
2.  <strong>重击者词元 (Heavy Hitters):</strong> 历史上累积注意力分数最高的词元，用于保持长距离的关键上下文信息。

## 4.2. 核心方法详解 (逐层深入)

### 4.2.1. 定义生成过程与注意力分数
首先，我们需要定义在缓存受限情况下的生成过程。

假设我们当前的查询向量是 $Q_{i,*}$（第 $i$ 个词元的 Query），我们维护的缓存集合是 $S_i$（即保留在显存中的历史词元索引集合）。

在标准 Attention 中，我们需要所有历史 $K$，但在 H`_2`O 中，我们只使用 $S_i$ 中的 $K$。第 $i$ 步的注意力输出向量 $o_i$ 和归一化因子 $D_i$ 计算如下（对应原文 Definition 2.2）：

$$
o_i := D_i^{-1} \cdot \exp(Q_{i,*} (K_{S_i, *})^\top)
$$

$$
D_i := (\exp(Q_{i,*} (K_{S_i, *})^\top) - \mathbf{1}_{[i]\setminus S_i}) \cdot \mathbf{1}_i
$$

*   **符号解释:**
    *   $Q_{i,*}$: 当前第 $i$ 步的 Query 向量。
    *   $K_{S_i, *}$: 仅包含集合 $S_i$ 中词元对应的 Key 矩阵子集。
    *   $\exp(\cdot)$: 对矩阵进行逐元素的指数运算（Softmax 的分子部分）。
    *   $D_i$: 分母，即 Softmax 的归一化常数。
    *   **核心逻辑:** 该公式表明，我们仅利用保留在集合 $S_i$ 中的 KV 对来计算当前的注意力分布，被驱逐的词元被视为注意力为 0。

### 4.2.2. 重击者识别与累积注意力
如何决定谁是“重击者”？作者发现，一个词元在历史上被关注得越多，它在未来被关注的概率也越大。因此，算法的核心指标是 <strong>累积注意力分数 (Accumulated Attention Score)</strong>。

在每一步生成时，算法会更新所有在缓存中的词元的累积得分。假设 $s$ 是缓存中的某个词元，其累积得分计算为历史所有 Softmax 归一化后的注意力权重之和。

下图（原文 Figure 2）展示了预训练 LLM 的稀疏性（图 a）以及累积注意力分数的分布（图 b）。图 b 显示注意力分数遵循幂律分布（Power-law distribution），验证了少数“重击者”的存在。

![该图像是图表，包含四个部分：(a) 展示了不同模型在不同层上的注意力稀疏性；(b) 显示了单词索引与共现次数之间的关系，整体呈现出平滑的曲线特性；(c) 比较了基线和去掉重击词后的准确率表现；(d) 以雷达图形式展示了多种任务下的不同统计方法的准确性。这些图表有助于理解论文中提出的 H`_2`O 方法的有效性。](images/2.jpg)
*该图像是图表，包含四个部分：(a) 展示了不同模型在不同层上的注意力稀疏性；(b) 显示了单词索引与共现次数之间的关系，整体呈现出平滑的曲线特性；(c) 比较了基线和去掉重击词后的准确率表现；(d) 以雷达图形式展示了多种任务下的不同统计方法的准确性。这些图表有助于理解论文中提出的 H`_2`O 方法的有效性。*

### 4.2.3. H`_2`O 驱逐算法 (Algorithm 1)
这是 H`_2`O 的实际操作流程，它是一个贪心算法（Greedy Algorithm）。

**算法流程:**
1.  **输入:** 当前 Query $Q$，当前 Key $K$，缓存预算大小 $k$。
2.  **初始化:** 缓存集合 $S_0$ 为空。
3.  <strong>循环 (对于每个新生成的词元 $i$):</strong>
    *   **缓存未满时:** 如果当前词元数 $i \le k$，直接将 $i$ 加入缓存集合 $S_i$。
    *   <strong>缓存已满时 (核心步骤):</strong>
        1.  **计算注意力:** 使用当前缓存中的 Key 计算注意力分数。
            $$
            \text{CurrentScores} = \text{Softmax}(Q_{i,*} \cdot K_{S_{i-1}, *}^\top)
            $$
        2.  **更新累积得分:** 将当前的注意力分数加到历史累积得分上。这一步捕捉了词元的重要性。
            $$
            \text{AccumulatedScore}[v] \leftarrow \text{AccumulatedScore}[v] + \text{CurrentScores}[v], \quad \forall v \in S_{i-1}
            $$
        3.  **贪心驱逐:** 找到当前缓存 $S_{i-1} \cup \{i\}$ 中，**累积注意力得分最低**的那个词元 $u$。
            $$
            u \leftarrow \arg\min_{v \in S_{i-1} \cup \{i\}} \text{AccumulatedScore}[v]
            $$
            *(注：原文公式写为 $\arg\max F_{\text{score}}(S \setminus \{v\})$，即移除 $v$ 后剩余分数最大，等价于移除分数最小的 $v$)*。
        4.  **更新缓存:** 从集合中移除 $u$，形成新的缓存集合 $S_i$。
            $$
            S_i \leftarrow (S_{i-1} \cup \{i\}) \setminus \{u\}
            $$

下图（原文 Figure 3）生动演示了这一过程。在 Step 4，词元被计算分数；在 Step 5，由于缓存限制，累积得分最低的词元（图中颜色最淡的）被驱逐。

![Figure 3: Illustration of Algorithm 1 during two consecutive decoding steps.](images/3.jpg)
*该图像是示意图，展示了算法1在两个连续解码步骤中的过程。在解码步骤4中，关键字为"Children laughed and played in the sunny park"的值被计算并加权。在解码步骤5中，对于新关键字的处理及全局统计进行说明，右侧显示了相应的计算值。同时，图中也提到了一种驱逐策略的不可行性。*

### 4.2.4. 理论基础：动态次模最大化
作者不仅仅提出了启发式算法，还将该问题形式化为 <strong>动态次模最大化 (Dynamic Submodular Maximization)</strong> 问题。
*   <strong>次模性 (Submodularity):</strong> 类似于经济学中的“边际效用递减”。增加一个词元带来的信息增益随着已有词元的增加而减少。
*   **理论保证:** 作者证明了在一定假设下，这种贪心策略（保留累积得分最高的词元）能够提供接近最优解的理论保证（具体证明在附录 D 中）。这意味着 H`_2`O 不是随意丢弃数据，而是数学上可证明的近似最优策略。

    ---

# 5. 实验设置

## 5.1. 数据集与任务
实验涵盖了多种模型和广泛的任务，以验证 H`_2`O 的通用性。
*   **模型:** OPT (6.7B, 13B, 30B, 66B), LLaMA-7B/13B/30B/65B, GPT-NeoX-20B。
*   **数据集:**
    *   **语言建模:** WikiText-103 (用于验证稀疏性)。
    *   <strong>下游任务 (lm-eval-harness &amp; HELM):</strong> COPA, MathQA, OpenBookQA, PiQA, RTE, Winogrande, XSUM, CNN/Daily Mail。
    *   **长文本/流式任务:** 验证模型处理无限长度输入的能力。

## 5.2. 评估指标
*   <strong>困惑度 (Perplexity, PPL):</strong>
    *   *概念定义:* 衡量概率模型预测样本的好坏程度。PPL 越低，模型生成的文本越自然、越准确。
    *   *数学公式:* `PPL(X) = \exp\left(-\frac{1}{t}\sum_{i=1}^t \log p_\theta(x_i|x_{<i})\right)`
    *   *符号解释:* $X$ 是文本序列，$t$ 是序列长度，$p_\theta(x_i|x_{<i})$ 是模型预测第 $i$ 个词的概率。
*   <strong>吞吐量 (Throughput):</strong>
    *   *概念定义:* 系统每秒能够生成多少个词元 (tokens/second)。这是衡量推理效率的关键指标。
*   <strong>延迟 (Latency):</strong>
    *   *概念定义:* 生成一个词元或完成一个请求所需的平均时间。
*   <strong>准确率 (Accuracy) / ROUGE:</strong>
    *   针对问答和摘要任务的具体性能指标。

## 5.3. 对比基线
*   **Full Cache:** 保留所有 KV Cache，作为性能上限（Oracle）。
*   **Local:** 仅保留最近的 $k$ 个词元（滑动窗口策略）。
*   **Sparse Transformer:** 基于固定模式（Fixed）或步幅模式（Strided）的稀疏注意力机制。
*   **现有系统:** DeepSpeed Zero-Inference, Hugging Face Accelerate, FlexGen。

    ---

# 6. 实验结果与分析

## 6.1. 核心结果分析：精度保持
实验表明，H`_2`O 在大幅压缩 KV Cache 的情况下，依然能保持与 Full Cache 相当的精度，并且显著优于 Local 策略和其他稀疏注意力方法。

下图（原文 Figure 4）展示了在不同 KV Cache 预算（例如 20%）下，H`_2`O（蓝色实线）与其他方法的对比。可以看到 H`_2`O 的曲线几乎与 Full Cache 重合，而 Local 策略（绿色虚线）在低预算下性能急剧下降。

![该图像是多个实验结果的示意图，展示了不同KV缓存预算下，Heavy-Hitter Oracle、Local和Full方法的表现对比，包括ROUGE-2、Coverage和Accuracy等指标。各图中，随着KV缓存预算的减少，Heavy-Hitter Oracle的表现相对较优，显示出其在长文本生成中的效率优势。](images/4.jpg)
*该图像是多个实验结果的示意图，展示了不同KV缓存预算下，Heavy-Hitter Oracle、Local和Full方法的表现对比，包括ROUGE-2、Coverage和Accuracy等指标。各图中，随着KV缓存预算的减少，Heavy-Hitter Oracle的表现相对较优，显示出其在长文本生成中的效率优势。*

以下是原文 Table 1 的数据，展示了在不同任务上 H`_2`O 与 Full Cache 和 Local 策略的具体对比。

**表格引用:** 以下是原文 Table 1 的结果：

<table>
<thead>
<tr>
<th>Methods</th>
<th>PiQA</th>
<th>COPA</th>
<th>OpenbookQA</th>
<th>Winogrande</th>
</tr>
</thead>
<tbody>
<tr>
<td>Full</td>
<td>80.09</td>
<td>81.00</td>
<td>44.80</td>
<td>71.51</td>
</tr>
<tr>
<td>0-shot Full</td>
<td>78.89</td>
<td>76.00</td>
<td>41.40</td>
<td>70.00</td>
</tr>
<tr>
<td>1-shot Full</td>
<td>79.11</td>
<td>76.00</td>
<td>43.60</td>
<td>70.24</td>
</tr>
<tr>
<td>Local</td>
<td>57.94</td>
<td>56.00</td>
<td>28.40</td>
<td>51.30</td>
</tr>
<tr>
<td>H2O</td>
<td>79.22</td>
<td>85.00</td>
<td>43.80</td>
<td>71.67</td>
</tr>
</tbody>
</table>

*   **分析:** 在 PiQA、COPA 等任务中，使用 20% 缓存预算的 H`_2`O 性能非常接近甚至在某些情况下略微超过 Full Cache（作者认为这可能起到了一定的正则化作用）。相比之下，Local 策略性能崩塌严重（例如 OpenbookQA 从 44.8 降至 28.4）。

## 6.2. 系统性能：吞吐量与延迟
这是 H`_2`O 最引人注目的成果。通过减少显存占用，H`_2`O 允许使用更大的 Batch Size，从而极大提升了吞吐量。

**表格引用:** 以下是原文 Table 3 的结果，展示了在 T4 GPU 上不同系统的生成吞吐量 (token/s)：

<table>
<thead>
<tr>
<th rowspan="2">Seq. length</th>
<th colspan="2">512+32</th>
<th colspan="2">512+512</th>
<th colspan="2">512+1024</th>
</tr>
<tr>
<th>6.7B</th>
<th>30B</th>
<th>6.7B</th>
<th>30B</th>
<th>6.7B</th>
<th>30B</th>
</tr>
</thead>
<tbody>
<tr>
<td>Accelerate</td>
<td>20.4 (2, G)</td>
<td>0.6 (8, C)</td>
<td>15.5 (1, G)</td>
<td>0.6 (8, C)</td>
<td>5.6 (16, C)</td>
<td>0.6 (8, C)</td>
</tr>
<tr>
<td>DeepSpeed</td>
<td>10.2 (16, C)</td>
<td>0.6 (4, C)</td>
<td>9.6 (16, C)</td>
<td>0.6 (4, C)</td>
<td>10.1 (16, C)</td>
<td>0.6 (4, C)</td>
</tr>
<tr>
<td>FlexGen</td>
<td>20.2 (2, G)</td>
<td>8.1 (144, C)</td>
<td>16.8 (1, G)</td>
<td>8.5 (80, C)</td>
<td>16.9 (1, G)</td>
<td>7.1 (48, C)</td>
</tr>
<tr>
<td>H2O (20%)</td>
<td>**35.1** (4, G)</td>
<td>**12.7** (728, C)</td>
<td>**51.7** (4, G)</td>
<td>**18.83** (416, C)</td>
<td>**52.1** (4, G)</td>
<td>**13.82** (264, C)</td>
</tr>
</tbody>
</table>

*   **符号说明:** 括号内数字为 Batch Size，"G" 代表 GPU，"C" 代表 CPU Offloading。
*   **分析:** H`_2`O 在所有设置下都取得了显著的吞吐量提升。特别是在 OPT-30B 模型上，相比 DeepSpeed 和 Accelerate 提升了约 29 倍（从 0.6 到 ~18），相比 FlexGen 也有 2-3 倍的提升。这主要是因为 H`_2`O 极大地降低了显存需求，使得可以在 GPU 上运行更大的 Batch Size，或者减少了向 CPU 卸载数据的频率。

## 6.3. 长文本与消融实验
*   **无限长度生成:** 如下图（原文 Figure 5）所示，H`_2`O 可以支持长达 400 万词元的流式生成，且困惑度（Perplexity）保持稳定，优于 StreamLLM。
*   **生成多样性:** 附录实验表明，H`_2`O 生成的文本不仅准确，而且比 Local 策略更具多样性，减少了重复现象。

    ![Figure 5: (Upper) streaming with ${ \\sf H } _ { 2 } { \\sf O }$ to handle inputs with sequence lengths of four million tokens. (Bottom) Perplexity comparison between the original StreamLLM method and our ${ \\sf H } _ { 2 } { \\sf O }$ , results are collected on the first text sample of PG-19 \[54\].](images/5.jpg)
    *该图像是图表，展示了使用 H`_2`O 处理四百万个令牌的流式读取性能。图的上半部分显示了在不同输入长度下的 NLL 值；下半部分则比较了 StreamLLM 方法与 H`_2`O 方法在不同缓存大小下的困惑度，结果显示 H`_2`O 提升了性能。*

---

# 7. 总结与思考

## 7.1. 结论总结
H`_2`O 是一项具有重要实践意义的研究。它揭示了 LLM 推理过程中的 <strong>$H_2$（重击者）</strong> 现象，并利用这一特性设计了简单高效的 KV Cache 驱逐策略。
1.  **高效性:** 将 KV Cache 显存占用降低 5-10 倍，吞吐量提升高达 29 倍。
2.  **易用性:** 不需要重新训练模型，作为一个即插即用的策略，可以轻松集成到现有的推理系统（如 FlexGen）中。
3.  **理论支撑:** 提供了基于次模最大化的理论保证，证明了贪心策略的有效性。

## 7.2. 局限性与未来工作
*   **局限性:**
    *   **位置偏差:** 累积注意力分数可能会偏向于早期的词元（因为它们存在的时间更长，累积的分数更多）。虽然作者提到 H`_2`O 结合了最近的词元来缓解这一问题，但这仍是一个潜在的偏差来源。
    *   **实现开销:** 虽然算法复杂度低，但在极高并发下，频繁的 Cache 整理和分数更新仍会带来一定的计算开销。
*   **未来工作:**
    *   **MLP 层稀疏性:** 论文附录中提到 MLP 层也存在类似的重击者现象，未来可以探索将类似的策略应用于模型权重的动态加载或计算剪枝。
    *   **硬件优化:** 结合专门的硬件内核（Kernel）来进一步加速稀疏注意力的计算。

## 7.3. 个人启发与批判
*   **启发:** 这篇论文最精彩的地方在于它**打破了“过去所有信息都同等重要”的假设**。在长文本处理中，这为我们提供了一个非常直观的视角：人类阅读长文时也是会遗忘大部分细节，只记住关键点（Heavy Hitters）和当前读到的内容（Local）。H`_2`O 完美地模拟了这一认知过程。
*   **批判:** 虽然 H`_2`O 在标准基准测试中表现优异，但在某些极其依赖微小细节（Needle in a Haystack，大海捞针）的任务中，如果关键信息不是“重击者”（即没有被反复关注），可能会被误删。这需要更精细的评价指标来验证极端情况下的鲁棒性。此外，将该方法与 PagedAttention (vLLM) 等显存管理技术结合，可能会是工业界落地的终极形态。