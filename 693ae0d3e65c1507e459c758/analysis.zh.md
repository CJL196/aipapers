# 1. 论文基本信息

## 1.1. 标题
RT-1: Robotics Transformer for Real-World Control at Scale
(RT-1：面向规模化真实世界控制的机器人 Transformer)

## 1.2. 作者
论文作者团队来自 **Robotics at Google**, **Everyday Robots**, 以及 **Google Research, Brain Team**。作者列表非常长，包括 Anthony Brohan, Chelsea Finn, Sergey Levine 等在机器人学习领域的知名学者。这表明该研究是一项由大型工业研究实验室支持的、大规模的、跨团队合作的工程与研究项目。

## 1.3. 发表期刊/会议
该论文以预印本 (Pre-print) 的形式发布在 **arXiv** 上。arXiv 是一个开放获取的学术论文发布平台，广泛用于物理学、数学、计算机科学等领域。虽然未经同行评审，但它是研究者快速分享最新成果的重要渠道。许多顶级会议（如 NeurIPS, ICML, CoRL）的论文都会先在 arXiv 上发布。

## 1.4. 发表年份
2022年

## 1.5. 摘要
现代机器学习模型通过从大规模、多样化的、任务无关的数据集中迁移知识，能够以零样本 (zero-shot) 或仅需少量任务特定数据的方式，高效解决下游任务。这一能力已在计算机视觉、自然语言处理等领域得到验证，但在机器人学领域尚待证明。在机器人学中，模型的泛化能力尤为关键，因为收集真实世界的机器人数据非常困难。本文作者认为，成功的通用机器人模型的关键之一在于**开放式的、任务无关的训练**，并结合能够吸收所有多样化机器人数据的**高容量架构**。

为此，本文提出了一类名为 <strong>机器人 Transformer (Robotics Transformer, RT-1)</strong> 的模型，它展现了优秀的可扩展性。研究团队通过在真实机器人上进行大规模数据收集（覆盖真实世界任务），系统地研究了不同模型类别在数据量、模型大小和数据多样性变化时的泛化能力，并验证了他们的结论。

## 1.6. 原文链接
- **原文链接:** https://arxiv.org/abs/2212.06817
- **PDF 链接:** https://arxiv.org/pdf/2212.06817v2.pdf
- **发布状态:** 预印本 (Pre-print)

  ---

# 2. 整体概括

## 2.1. 研究背景与动机
- **核心问题：** 如何构建一个**单一的、通用的机器人控制模型**，使其能够理解并执行大量不同的任务，同时还能泛化到从未见过的新任务、新物体和新环境中？

- <strong>重要性与挑战 (Gap):</strong> 在机器人领域，传统的学习方法通常是“一个任务，一个模型”。即针对每个特定任务（如“拿起杯子”）收集专门的数据集并训练一个专门的模型。这种模式的扩展性极差，因为：
    1.  **数据收集成本高昂：** 在真实世界中让机器人收集一次成功任务的数据，无论是通过人类遥操作演示还是强化学习试错，都非常耗时、昂贵且需要大量工程投入。
    2.  **泛化能力有限：** 专门训练的模型很难将知识迁移到新任务上。例如，一个学会了“拿起杯子”的模型，并不知道如何“打开抽屉”。

        近年来，在自然语言处理 (NLP) 和计算机视觉 (CV) 领域，**大规模预训练模型**（如 GPT-3, CLIP）已经证明，通过在海量、多样化的数据上进行训练，单一模型可以获得强大的泛化能力，无需为每个新任务从头开始。机器人领域迫切需要这样的“基础模型”，但面临着比 NLP 和 CV 更严峻的挑战：**数据的稀缺性和异构性**。

- **切入点与创新思路：** 本文的作者们认为，解决这一问题的关键在于两点：
    1.  **大规模、多样化的真实世界数据集：** 必须有一个足够大、任务种类足够丰富的机器人操作数据集作为模型学习的基础。
    2.  **高容量、可扩展的模型架构：** 需要一个像 Transformer 一样强大的模型，能够“吸收”和“消化”这些海量数据中蕴含的知识。

        因此，本文的创新思路是：**将 Transformer 架构适配于机器人控制任务，并结合一个前所未有规模的真实世界机器人演示数据集，来验证“规模化”是否也能为机器人学带来类似 NLP 和 CV 的突破。**

## 2.2. 核心贡献/主要发现
本文的核心贡献可以总结为以下四点：

1.  **提出了 RT-1 模型架构：** 设计了一种名为 <strong>机器人 Transformer 1 (RT-1)</strong> 的高效模型。它巧妙地结合了多种技术，既拥有 Transformer 的高容量，又能满足机器人**实时控制**所需的**快速推理**要求（约 3Hz）。这是其与许多其他大型 Transformer 模型在应用上的关键区别。

2.  **构建了大规模真实世界数据集：** 团队投入了巨大的工程努力，在 17 个月内使用 13 台机器人收集了约 **13 万次**成功任务演示，涵盖了 **700 多种**不同的自然语言指令。这个数据集本身就是一项重要的贡献，为训练和评估通用机器人模型提供了宝贵的资源。

3.  **验证了模型的卓越性能与泛化能力：** 实验证明，RT-1 在其训练过的 700 多项任务上达到了 **97% 的成功率**。更重要的是，它表现出强大的<strong>零样本泛化 (zero-shot generalization)</strong>能力，在**新任务、新背景和有干扰物**的场景下，其性能显著优于先前的最先进模型（如 Gato, BC-Z）。

4.  **展示了强大的数据吸收能力：** RT-1 不仅能从自身的训练数据中学习，还能有效地**吸收来自异构数据源**的知识。实验表明，将**仿真数据**或<strong>来自不同形态机器人（Kuka 机械臂）的数据</strong>混合训练后，RT-1 不仅在新场景下的泛化能力得到提升，而且在原始任务上的性能几乎不受影响。这证明了其作为“数据海绵”的潜力，为构建更通用的机器人基础模型指明了方向。

    ---

# 3. 预备知识与相关工作

## 3.1. 基础概念

### 3.1.1. 模仿学习 (Imitation Learning)
模仿学习是一种让机器（如机器人）通过观察和模仿“专家”的演示来学习技能的方法。它不同于需要自己探索和试错的强化学习 (Reinforcement Learning)。
*   **核心思想：** 假设我们有一组专家演示数据，其中包含了在各种状态下专家所采取的“正确”动作。模仿学习的目标就是训练一个<strong>策略 (policy)</strong> 模型，这个模型能够输入当前状态，并输出一个与专家行为尽可能相似的动作。
*   <strong>行为克隆 (Behavioral Cloning, BC):</strong> 这是最简单直接的一种模仿学习方法，也是 RT-1 所采用的方法。它将模仿学习问题转化为一个标准的<strong>监督学习 (supervised learning)</strong> 问题。
    *   **数据：** 状态-动作对 `(s, a)` 的集合，其中 $s$ 是机器人观察到的状态（如摄像头图像），$a$ 是专家在该状态下执行的动作（如遥操作指令）。
    *   **目标：** 训练一个策略网络 $\pi_\theta(s)$，使其预测的动作 $\hat{a}$ 与专家的真实动作 $a$ 之间的误差最小。对于分类任务（如 RT-1 的离散化动作），通常使用<strong>交叉熵损失 (cross-entropy loss)</strong>；对于连续动作，通常使用<strong>均方误差损失 (mean squared error loss)</strong>。

### 3.1.2. Transformer
Transformer 是于 2017 年在论文 *Attention Is All You Need* 中提出的深度学习模型，最初用于自然语言翻译，现已成为 NLP、CV 等多个领域的主流架构。
*   <strong>核心机制：自注意力 (Self-Attention):</strong> Transformer 的强大能力源于其自注意力机制。对于一个序列中的每个元素（例如，一句话中的每个单词），自注意力机制能够计算它与序列中所有其他元素之间的“相关性”或“重要性”得分，然后根据这些得分对所有元素的信息进行加权求和，从而得到该元素新的、包含上下文信息的表示。这使得模型能够捕捉长距离依赖关系。
*   <strong>Q, K, V (查询, 键, 值):</strong> 自注意力的计算可以被形象地理解为一次数据库查询。对于序列中的每个元素，我们都生成三个向量：
    *   <strong>查询 (Query, Q):</strong> 代表当前元素，用于“发起查询”，询问与其他元素的关联。
    *   <strong>键 (Key, K):</strong> 代表序列中的其他元素，用于被 Q“查询”，以计算匹配度。
    *   <strong>值 (Value, V):</strong> 代表序列中其他元素所包含的信息。
*   **注意力计算公式:**
    自注意力的计算过程如下。首先，通过计算 Q 和 K 的点积来得到注意力分数，然后进行缩放（除以 $\sqrt{d_k}$ 防止梯度过小），接着通过 `softmax` 函数将分数归一化为权重，最后将这些权重与 V 相乘并求和。
    $$
    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    $$
    **符号解释:**
    *   $Q$: 查询矩阵。
    *   $K$: 键矩阵。
    *   $V$: 值矩阵。
    *   $K^T$: 键矩阵 $K$ 的转置。
    *   $d_k$: 键向量的维度。
    *   $\mathrm{softmax}$: 将输入向量转换为概率分布的函数。

        在 RT-1 中，Transformer 被用来处理由图像和语言指令编码而来的<strong>词元 (token)</strong> 序列，并输出代表机器人动作的词元序列。

### 3.1.3. FiLM (Feature-wise Linear Modulation)
FiLM 是一种<strong>条件网络 (conditioning network)</strong> 技术，它允许一个神经网络的输出（如语言指令）来动态地调整另一个神经网络（如图像处理网络）的行为。
*   **核心思想：** FiLM 层根据一个条件输入（例如，RT-1 中的语言指令嵌入）生成一对参数：一个缩放因子 $\gamma$ 和一个偏置因子 $\beta$。然后，它将这两个参数以<strong>仿射变换 (affine transformation)</strong> 的方式，逐元素地应用于目标网络的特征图 (feature map) 上。
    $$
    \mathrm{FiLM}(x | c) = \gamma(c) \odot x + \beta(c)
    $$
    **符号解释:**
    *   $x$: 目标网络的特征图。
    *   $c$: 条件输入（如语言指令嵌入）。
    *   $\gamma(c), \beta(c)$: 由条件 $c$ 生成的缩放和偏置参数。
    *   $\odot$: 逐元素相乘。

        在 RT-1 中，FiLM 层被插入到 `EfficientNet` 视觉主干网络中，使得**语言指令能够早期介入视觉特征的提取过程**，引导模型关注与任务相关的视觉区域（例如，指令是“拿起苹果”，FiLM 会帮助网络更关注图像中的苹果）。

## 3.2. 前人工作
作者将 RT-1 的工作置于机器人学习的几个关键研究脉络中：

1.  **基于 Transformer 的机器人策略：** 近年来，许多工作尝试将 Transformer 用于机器人控制。
    *   **Gato (Reed et al., 2022):** 一个通用的、多模态的序列模型，可以处理语言、视觉和控制任务。但 Gato 在真实机器人上的任务相对简单（堆叠色块），且未评估其在新任务上的泛化能力。
    *   **Decision Transformer (Chen et al., 2021):** 将强化学习问题重新构建为序列建模问题，在模拟环境中取得了成功。
    *   <strong>其他工作 (Jang et al., 2021; Shridhar et al., 2022):</strong> 这些工作使用 Transformer 处理语言指令或实现多任务学习，但要么真实世界任务的广度有限，要么更侧重于训练任务的性能而非对新任务的泛化。

2.  **多任务与语言条件学习：** 机器人领域有很长的历史在研究如何让机器人同时学习多个任务或根据语言指令行动。
    *   <strong>BC-Z (Jang et al., 2021) 和 SayCan (Ahn et al., 2022):</strong> 这些是 Google 内部的先前工作，BC-Z 是一种基于 ResNet 的多任务模仿学习模型，SayCan 则通过结合大型语言模型和机器人技能的可行性（affordances）来实现长时程规划。RT-1 在模型架构和数据规模上都超越了它们。
    *   <strong>其他工作 (Lynch &amp; Sermanet, 2020; Stepputtis et al., 2020):</strong> 这些工作也探索了端到端的语言条件模仿学习，但规模和任务多样性不及 RT-1。

3.  **大规模机器人数据集：** 社区已经认识到数据的重要性，并有工作致力于收集多样化的机器人数据集。
    *   **RoboNet (Dasari et al., 2019):** 收集了来自不同机器人平台的数据，强调跨机器人形态的泛化。
    *   **Bridge Data (Ebert et al., 2021):** 探索了如何利用来自不同实验室、不同环境的数据集来提升机器人技能的泛化能力。
    *   RT-1 的数据集在**单一机器人平台、任务多样性和数据总量**上达到了新的高度，专注于深度而非广度地探索规模化的力量。

## 3.3. 技术演进
机器人学习的技术路线大致经历了从**单一任务学习**到**多任务学习**，再到如今追求**通用基础模型**的演变。
1.  **早期：** 专注于解决单一、孤立的任务，如抓取、开门等。模型和数据都是任务特定的。
2.  **中期：** 开始探索多任务学习，让一个模型同时掌握多种技能，并通过共享表示来提高数据效率。语言指令开始被用作一种灵活的任务指定方式。
3.  **当前：** 受 NLP 和 CV 领域成功的启发，研究重心转向构建**大规模、通用的“基础模型”**。研究者希望通过在海量、多样化的数据上预训练一个高容量模型，使其能够零样本或少样本泛化到无数新任务，RT-1 正是这一趋势下的代表性工作。

## 3.4. 差异化分析
RT-1 与其最相关的基线模型相比，核心区别在于：

*   **相较于 Gato:**
    *   **语言融合时机:** RT-1 通过 FiLM 实现**早期融合**，在视觉特征提取阶段就融入了语言信息。而 Gato 是**晚期融合**，将语言和视觉词元拼接后才送入 Transformer。RT-1 的方式理论上能更早地提取任务相关特征。
    *   **推理效率:** RT-1 专为实时控制设计，通过 `TokenLearner` 等技术大幅压缩视觉词元数量，实现了快速推理。Gato 的原始设计未重点考虑实时性。
    *   **语言编码:** RT-1 使用了预训练的<strong>通用语句编码器 (Universal Sentence Encoder, USE)</strong>，直接利用了大型语言模型的语义理解能力。

*   **相较于 BC-Z:**
    *   **模型架构:** RT-1 是基于 **Transformer** 的序列模型，能够处理历史观测信息。BC-Z 是基于 **ResNet** 的前馈模型，只看当前时刻的图像。
    *   **动作表示:** RT-1 使用<strong>离散化的动作词元 (action tokens)</strong>，将控制问题转化为一个分类问题，可以更好地表示多模态的动作分布。BC-Z 输出**连续的动作值**。

        ---

# 4. 方法论
本节将详细拆解 RT-1 的系统架构和工作流程，严格遵循论文中的描述，并融合公式进行解释。

## 4.1. 方法原理
RT-1 的核心思想是将机器人控制问题视为一个**序列到序列的建模问题**。它接收一个包含历史图像和当前任务指令的输入序列，并生成一个代表机器人下一步动作的输出序列。为了在保持高模型容量的同时实现实时控制，RT-1 的设计在**表达能力**和**计算效率**之间做了精心的权衡。其架构如下图（原文 Figure 3）所示，我们将按照数据流动的顺序逐步解析。

下图（原文 Figure 3）展示了 RT-1 的完整架构图：

![Figure 3: The architecture diagram of RT-1. The instruction is transformed into a USE embedding and used to condition a pre-trained EffcientNet via FiLM layers. The resulting vision-language tokens are reduced by the TokenLearner and fed into a decoder-only Transformer, which outputs tokenized actions.](images/4.jpg)
*该图像是RT-1的架构示意图。它展示了如何将指令转换为USE嵌入并用于条件预训练的EfficientNet，继而通过FiLM层进行处理，生成视觉语言令牌，最后通过解码器变换器输出动作。该模型具有高度的可扩展性，并结合了自注意力机制与多种参数的特性。*

## 4.2. 核心方法详解 (逐层深入)

### 4.2.1. 输入
在每个控制时间步 $t$，RT-1 的输入包括两部分：
1.  <strong>图像历史 (Image History):</strong> 最近的 6 帧摄像头图像 $\{x_{t-5}, ..., x_t\}$。每张图像的分辨率为 $300 \times 300$ 像素。使用历史信息有助于模型理解动态过程（如物体移动速度）和处理遮挡。
2.  <strong>语言指令 (Language Instruction):</strong> 一个描述任务目标的自然语言字符串 $i$，例如 "pick up the apple"。这个指令在整个任务执行期间保持不变。

### 4.2.2. 模块一：指令与图像词元化 (Instruction and Image Tokenization)
这个阶段的目标是将高维的图像和文本输入，转换成适合 Transformer 处理的、紧凑的<strong>词元 (token)</strong> 序列。

**步骤 1: 语言指令编码**
首先，自然语言指令 $i$ 被送入一个预训练好的<strong>通用语句编码器 (Universal Sentence Encoder, USE)</strong>。
*   **USE (Cer et al., 2018):** 这是一个强大的语言模型，能够将任意长度的句子编码成一个固定维度的、能捕捉其语义信息的向量。
*   **输出:** 一个代表整个指令含义的嵌入向量，我们称之为 $e_{lang}$。

**步骤 2: FiLM 条件化的图像编码**
接下来，6 帧历史图像中的每一帧都会经过一个特殊的视觉编码器。这个编码器是基于一个在 ImageNet 上预训练的 `EfficientNet-B3` 模型，但通过 **FiLM 层** 进行了改造，以接收语言指令的条件。

*   <strong>主干网络 (Backbone):</strong> `EfficientNet-B3` 是一个高效的卷积神经网络 (CNN)，用于提取图像的视觉特征。
*   **FiLM 注入:** 在 `EfficientNet-B3` 的多个卷积模块 (MBConv blocks) 之后，都插入了 FiLM 层。
    *   对于每个 FiLM 层，语言嵌入 $e_{lang}$ 会通过两个小型的全连接层，分别生成一个**缩放向量 $\gamma$** 和一个**偏置向量 $\beta$**。
    *   然后，这两个向量被用来对 `EfficientNet` 中间层的特征图进行仿射变换：$y = \gamma \odot x + \beta$，其中 $x$ 是原始特征图，$y$ 是调制后的特征图。
*   <strong>身份初始化 (Identity Initialization):</strong> 为了在插入 FiLM 层时不破坏 `EfficientNet` 预训练权重的有效性，作者采用了一个巧妙的技巧：将生成 $\gamma$ 和 $\beta$ 的全连接层权重初始化为零。这样，在训练初期，$\gamma$ 接近于1，$\beta$ 接近于0，FiLM 层近似于一个恒等变换，从而保留了预训练模型的原始功能。随着训练的进行，模型会逐渐学会如何利用语言信息来有效调制视觉特征。
*   **输出:** 对于单张输入图像，经过 FiLM-EfficientNet 编码后，会得到一个 $9 \times 9 \times 512$ 的空间特征图。作者将其<strong>展平 (flatten)</strong>，视为 `81` 个视觉词元，每个词元的维度是 512。这实现了从原始像素到“视觉-语言”融合词元的转换。

### 4.2.3. 模块二：TokenLearner 压缩
此时，每张图像产生了 81 个词元，6 张图像总共就是 $6 \times 81 = 486$ 个词元。这个数量对于 Transformer 来说计算开销依然很大，会影响实时性。因此，作者引入了 `TokenLearner` 来进一步压缩词元数量。

*   **TokenLearner (Ryoo et al., 2021):** 这是一个可学习的模块，其本质是一个**空间注意力机制**。它会为输入的词元序列（此例中是 81 个词元）生成多个空间注意力图。
*   **工作原理:**
    1.  `TokenLearner` 学习生成 $S$ 个不同的空间注意力权重图（在 RT-1 中，$S=8$）。
    2.  每个注意力图都会与输入的 81 个词元进行加权求和，从而“学习”出一个新的、聚合了空间信息的词元。
    3.  这个过程重复 $S$ 次，最终将 81 个输入词元压缩成 $S=8$ 个输出词元。
*   **作用:** `TokenLearner` 能够**自适应地选择和聚合最重要的视觉信息**，丢弃冗余或无关的背景信息，从而在不损失太多性能的情况下，极大地减少了送入 Transformer 的序列长度。

### 4.2.4. 模块三：Transformer 序列处理
经过 `TokenLearner` 压缩后，每张图像只剩下 8 个词元。

*   **序列构建:** 将 6 帧图像的词元（每帧 8 个）拼接起来，形成一个总长度为 $6 \times 8 = 48$ 的词元序列。
*   <strong>位置编码 (Positional Encoding):</strong> 为这个 48 词元的序列添加位置编码，让 Transformer 能够区分不同时间步的图像信息。
*   **Transformer 模型:** 这个序列被送入一个 8 层的<strong>解码器式 Transformer (decoder-only Transformer)</strong>。
    *   它包含标准的<strong>多头自注意力 (Multi-Head Self-Attention)</strong> 层和<strong>前馈网络 (Feed-Forward Network)</strong> 层。
    *   由于是解码器式结构，它使用<strong>因果掩码 (causal masking)</strong>，确保在预测当前动作时，只能关注到输入图像序列的信息，而不能“看到未来”。

### 4.2.5. 模块四：动作词元化与输出
RT-1 的动作空间包含 11 个维度，涵盖了机械臂末端执行器的 6-DoF 位姿、1-DoF 夹爪开合度、移动底盘的 3-DoF 运动，以及一个用于切换控制模式（臂/底盘/终止）的离散变量。

*   <strong>动作离散化 (Action Discretization):</strong> RT-1 的一个关键设计是**不直接预测连续的动作值**。相反，它将每个连续的动作维度<strong>离散化 (discretize)</strong> 成 256 个<strong>桶 (bins)</strong>。例如，末端执行器的 X 坐标范围被均匀地切分成 256 段，每一段对应一个整数 ID（0-255）。
*   **输出预测:** Transformer 的任务被设定为：对于 11 个动作维度中的每一个，都预测出其对应的正确 bin ID。这本质上是 11 个并行的 256-分类问题。
*   <strong>非自回归生成 (Non-Autoregressive Generation):</strong> 与许多生成模型不同，RT-1 **并行地**预测所有 11 个动作维度的词元，而不是一个接一个地生成。这一设计极大地加快了推理速度，是实现实时控制的关键，尽管它牺牲了对动作维度之间依赖关系的建模。

### 4.2.6. 损失函数
模型训练采用行为克隆，其损失函数是标准的<strong>分类交叉熵损失 (Categorical Cross-Entropy Loss)</strong>，对所有动作维度的损失求和。
对于单个动作维度 $d$，损失函数为：
$$
\mathcal{L}_d = - \sum_{k=1}^{256} a_{d,k} \log(\hat{a}_{d,k})
$$
**符号解释:**
*   $a_{d,k}$: 一个独热编码 (one-hot) 向量，如果专家演示的动作在维度 $d$ 上属于第 $k$ 个 bin，则该值为 1，否则为 0。
*   $\hat{a}_{d,k}$: 模型预测的在维度 $d$ 上动作属于第 $k$ 个 bin 的概率。

    总损失是所有 11 个动作维度损失的平均值或总和。

### 4.2.7. 推理速度优化总结
为了在真实机器人上达到 3Hz 的控制频率（约 100ms 推理时间预算），RT-1 采用了两个关键的加速策略：
1.  **TokenLearner:** 将 Transformer 需要处理的词元数量从 486 个减少到 48 个，显著降低了自注意力机制 $O(N^2)$ 的计算复杂度。
2.  <strong>计算重用 (Computation Reuse):</strong> 在连续的控制步中，图像历史窗口是重叠的。例如，在 $t+1$ 时刻，输入的 6 帧图像中有 5 帧与 $t$ 时刻是相同的。RT-1 会缓存并重用这些已计算过的图像词元，避免重复计算，进一步提升了效率。

    ---

# 5. 实验设置

## 5.1. 数据集
实验的数据基础是其成功的关键。

*   **数据来源:** 数据由一个包含 13 台 **Everyday Robots 移动操作平台**（7自由度臂 + 两指夹爪 + 移动底座）的机器人队，在 17 个月的时间里，通过**人类遥操作演示**收集而成。
*   **规模与多样性:**
    *   总共约 **13 万个** 成功任务的演示片段（episodes）。
    *   涵盖超过 **700 种** 不同的自然语言指令。
    *   任务包括：拾取、放置、开关抽屉、将物品直立、推倒物品等。
    *   使用了大量不同的物体，以增强模型的泛化能力。
*   **收集环境:** 主要在一个模拟办公厨房环境的“机器人教室”中进行大规模收集。评估则在教室以及两个真实的、环境（光照、背景、布局）不同的办公厨房（Kitchen1, Kitchen2）中进行。

    下图（原文 Figure 2）展示了数据收集和评估的环境及物体：

    ![Figure 2: (a) Robot classroom where we collect data at scale; (b) a real office kitchen, one of the two realistic environments used for evaluation (named Kitchen1 in the rest of the paper); (c) a different office kitchen used for evaluation (named Kitchen2 in the rest of the paper); (d) mobile manipulator used throughout the paper; (e) a set of objects used for most of the skills to expand skill diversity; (f) a more diverse set of objects used mostly to expand object diversity of the picking skill.](images/3.jpg)
    *(a) 用于大规模数据收集的机器人教室；(b) 用于评估的真实办公室厨房 Kitchen1；(c) 用于评估的另一个办公室厨房 Kitchen2；(d) 实验所用的移动操作机器人；(e, f) 用于增加任务和物体多样性的物体集合。*

*   <strong>任务指令示例 (来自原文 Table 1):</strong>

    | 技能 (Skill) | 数量 (Count) | 描述 (Description) | 示例指令 (Example Instruction) |
    | :--- | :--- | :--- | :--- |
    | Pick Object | 130 | Lift the object off the surface | pick iced tea can |
    | Move Object Near Object | 337 | Move the first object near the second | move pepsi can near rxbar blueberry |
    | Place Object Upright | 8 | Place an elongated object upright | place water bottle upright |
    | Knock Object Over | 8 | Knock an elongated object over | knock redbull can over |
    | Open/Close Drawer | 6 | Open or close any of the cabinet drawers | open the top drawer / close the middle drawer |
    | Place Object into Receptacle | 84 | Place an object into a receptacle | place brown chip bag into white bowl |
    | Pick Object from Receptacle | 162 | Pick an object up from a location and then place it on the counter | pick green jalapeno chip bag from paper bowl and place on counter |
    | ... | ... | ... | ... |
    | **Total** | **744** | | |

## 5.2. 评估指标
论文中使用的唯一评估指标是<strong>成功率 (Success Rate)</strong>。

1.  <strong>概念定义 (Conceptual Definition):</strong> 成功率是指机器人在给定任务指令后，自主完成任务的试验次数占总试验次数的百分比。这是一个直接、明确且在机器人任务评估中非常标准的指标。例如，如果让机器人执行“拿起苹果”100 次，它成功了 90 次，那么成功率就是 90%。
2.  <strong>数学公式 (Mathematical Formula):</strong>
    $$
    \text{Success Rate} = \frac{\text{Number of Successful Trials}}{\text{Total Number of Trials}} \times 100\%
    $$
3.  **符号解释:**
    *   `Number of Successful Trials`: 机器人完全并正确地完成了指令所描述任务的次数。
    *   `Total Number of Trials`: 针对该指令进行的总测试次数。

## 5.3. 对比基线
为了证明 RT-1 的优越性，作者将其与两个具有代表性的最新模型进行了比较。**重要的是，所有基线模型都在与 RT-1 完全相同的 130k 数据集上进行了重新训练**，以确保这是一个公平的**架构对比**，而非数据对比。

1.  **Gato (Reed et al., 2022):** 一个通用的、基于 Transformer 的多模态模型。作者使用了一个参数量与 RT-1 相近（37M）的 Gato 版本进行公平比较，而不是原论文中的 1.2B 版本（因其推理太慢，无法用于真实机器人）。
2.  **BC-Z (Jang et al., 2021):** 一个基于 ResNet 的前馈模型，也是 Google 先前工作 SayCan 中使用的策略模型。它代表了非 Transformer、非序列模型的先进水平。
3.  **BC-Z XL:** 作者还训练了一个更大版本的 BC-Z，使其参数量与 RT-1 相当，以消除模型大小可能带来的影响。

    ---

# 6. 实验结果与分析

## 6.1. 核心结果分析
实验的核心问题是：RT-1 能否学习大量指令，并泛化到新任务、新物体和新环境中？

**原文 Table 2** 提供了核心性能对比，展示了 RT-1 在四个评估维度上的成功率，并与基线模型进行了比较。

以下是原文 Table 2 的结果：

<table>
<thead>
<tr>
<th>Model</th>
<th>Seen Tasks (已见任务)</th>
<th>Unseen Tasks (未见任务)</th>
<th>Distractors (干扰物)</th>
<th>Backgrounds (新背景)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Gato (Reed et al., 2022)</td>
<td>65%</td>
<td>52%</td>
<td>43%</td>
<td>35%</td>
</tr>
<tr>
<td>BC-Z (Jang et al., 2021)</td>
<td>72%</td>
<td>19%</td>
<td>47%</td>
<td>41%</td>
</tr>
<tr>
<td>BC-Z XL</td>
<td>56%</td>
<td>43%</td>
<td>23%</td>
<td>35%</td>
</tr>
<tr>
<td><b>RT-1 (ours)</b></td>
<td><b>97%</b></td>
<td><b>76%</b></td>
<td><b>83%</b></td>
<td><b>59%</b></td>
</tr>
</tbody>
</table>

**分析：**
*   <strong>已见任务 (Seen Tasks):</strong> RT-1 达到了惊人的 97% 成功率，远超所有基线，表明它具有强大的能力来学习和记忆训练数据集中包含的 700 多种技能。
*   <strong>未见任务 (Unseen Tasks):</strong> 这是衡量**组合泛化**能力的关键。RT-1 达到了 76% 的成功率，比次优的 Gato (52%) 高出 24%。这说明 RT-1 能很好地理解语言指令中对已知概念（技能、物体）的新组合，例如，如果它学过“拿起A”和“移动B”，它就能更好地泛化到“拿起B”。BC-Z 在这一项上表现很差（19%），说明其架构难以实现这种组合泛化。
*   <strong>干扰物鲁棒性 (Distractors):</strong> 在桌面上有许多无关物体时，RT-1 表现出极强的鲁棒性（83%），比次优的 BC-Z (47%) 高出 36%。这很可能得益于其<strong>早期语言融合 (FiLM)</strong>机制，该机制帮助视觉系统在早期就“过滤”掉与指令无关的物体。
*   <strong>背景鲁棒性 (Backgrounds):</strong> 在全新的厨房环境或不同的桌布上，RT-1 同样表现最佳（59%），比次优的 BC-Z (41%) 高出 18%。这表明其从大规模多样化数据中学到的视觉表示更加鲁棒。

### 6.1.1. 异构数据吸收能力
实验进一步探索了 RT-1 是否能从不同来源的数据中获益。

<strong>1. 吸收仿真数据 (Absorbing Simulation Data)</strong>
模型在混合了真实世界数据和仿真数据（包含真实世界未见过的新物体）后进行测试。

以下是原文 Table 4 的结果：

<table>
<thead>
<tr>
<th rowspan="2">Models</th>
<th rowspan="2">Training Data</th>
<th colspan="1">Real Objects</th>
<th colspan="2">Sim Objects (not seen in real)</th>
</tr>
<tr>
<th>Seen Skill w/ Objects</th>
<th>Seen Skill w/ Objects</th>
<th>Unseen Skill w/ Objects</th>
</tr>
</thead>
<tbody>
<tr>
<td>RT-1</td>
<td>Real Only</td>
<td>92%</td>
<td>23%</td>
<td>7%</td>
</tr>
<tr>
<td>RT-1</td>
<td><b>Real + Sim</b></td>
<td>90% (-2%)</td>
<td><b>87% (+64%)</b></td>
<td><b>33% (+26%)</b></td>
</tr>
</tbody>
</table>

**分析：**
*   混合仿真数据后，RT-1 在原始真实物体任务上的性能几乎没有下降（-2%）。
*   对于**只在仿真中见过的物体**，RT-1 在真实世界中的拾取成功率从 23% **飙升至 87%**，实现了出色的<strong>模拟到真实 (sim-to-real)</strong> 的迁移。
*   更令人印象深刻的是，对于**新技能和仿真物体的组合**（例如，仿真中只学过“拿起X”，测试时要求“移动X到Y”），成功率也从 7% **提升至 33%**。
*   **结论：** RT-1 是一个高效的“数据海绵”，可以无缝地吸收仿真数据来扩展其对新物体的知识，而不会忘记已有的技能。

<strong>2. 吸收来自不同机器人的数据 (Absorbing Data from Different Robots)</strong>
模型在混合了自身数据和来自完全不同机器人（Kuka IIWA）的抓取数据（QT-Opt 数据集）后进行测试。

以下是原文 Table 5 的结果：

<table>
<thead>
<tr>
<th>Models</th>
<th>Training Data</th>
<th>Classroom eval</th>
<th>Bin-picking eval</th>
</tr>
</thead>
<tbody>
<tr>
<td>RT-1</td>
<td><b>Kuka + EDR data</b></td>
<td>90% (-2%)</td>
<td><b>39% (+17%)</b></td>
</tr>
<tr>
<td>RT-1</td>
<td>EDR only data</td>
<td>92%</td>
<td>22%</td>
</tr>
<tr>
<td>RT-1</td>
<td>Kuka only data</td>
<td>0%</td>
<td>0%</td>
</tr>
</tbody>
</table>

**分析：**
*   混合 Kuka 数据后，RT-1 在其标准任务上的性能仅下降了 2%，再次证明其稳定性。
*   在一个模仿 Kuka 数据收集设置的<strong>新任务（箱内拣选, Bin-picking）</strong>上，混合数据训练的模型的性能从 22% **提升至 39%**，几乎翻倍。
*   只用 Kuka 数据训练的模型在 Everyday Robot (EDR) 上的成功率为 0%，说明直接的跨形态迁移非常困难。
*   **结论：** RT-1 能够从另一个机器人的经验中提取出**通用的技能知识**（如“如何接近并抓取一个物体”），并将其应用于自身的机器人平台上，实现了<strong>跨机器人形态 (cross-morphology)</strong> 的知识迁移。

### 6.1.2. 长时程任务 (Long-Horizon Scenarios)
RT-1 被整合到 **SayCan** 框架中，用于执行需要多个步骤的复杂指令。

以下是原文 Table 6 (即 Table 11) 的结果：

<table>
<thead>
<tr>
<th rowspan="2"></th>
<th colspan="2">SayCan tasks in Kitchen1</th>
<th colspan="2">SayCan tasks in Kitchen2</th>
</tr>
<tr>
<th>Planning</th>
<th>Execution</th>
<th>Planning</th>
<th>Execution</th>
</tr>
</thead>
<tbody>
<tr>
<td>Original SayCan (Ahn et al., 2022)*</td>
<td>73%</td>
<td>47%</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>SayCan w/ Gato (Reed et al., 2022)</td>
<td>87%</td>
<td>33%</td>
<td>87%</td>
<td>0%</td>
</tr>
<tr>
<td>SayCan w/ BC-Z (Jang et al., 2021)</td>
<td>87%</td>
<td>53%</td>
<td>87%</td>
<td>13%</td>
</tr>
<tr>
<td><b>SayCan w/ RT-1 (ours)</b></td>
<td><b>87%</b></td>
<td><b>67%</b></td>
<td><b>87%</b></td>
<td><b>67%</b></td>
</tr>
</tbody>
</table>

**分析：**
*   在熟悉的 Kitchen1 中，`SayCan+RT-1` 的执行成功率（67%）最高，远超其他基线。
*   **最关键的结果**在 Kitchen2，这是一个对泛化能力要求极高的全新环境。Gato 的成功率降至 0%，BC-Z 降至 13%，而 **RT-1 的性能毫无衰减，依然保持 67%**。
*   **结论：** RT-1 卓越的单步任务泛化能力，使其在组合成需要数十个步骤的长时程任务时，依然能保持高成功率，展现了在真实、复杂场景中部署的巨大潜力。

## 6.2. 消融实验/参数分析
作者进行了一系列消融实验来验证模型设计的合理性。

### 6.2.1. 数据量 vs. 数据多样性
以下是原文 Table 7 的结果，探究了数据量和任务多样性对性能的影响。

<table>
<thead>
<tr>
<th rowspan="2">Models</th>
<th rowspan="2">% Tasks</th>
<th rowspan="2">% Data</th>
<th rowspan="2">Seen Tasks</th>
<th colspan="4">Generalization</th>
</tr>
<tr>
<th>All</th>
<th>Unseen Tasks</th>
<th>Distractors</th>
<th>Backgrounds</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="8"><b>Smaller Data (数据量减少)</b></td>
</tr>
<tr>
<td><b>RT-1 (ours)</b></td>
<td>100</td>
<td>100</td>
<td><b>97</b></td>
<td><b>73</b></td>
<td><b>76</b></td>
<td><b>83</b></td>
<td><b>59</b></td>
</tr>
<tr>
<td>RT-1</td>
<td>100</td>
<td>51</td>
<td>71</td>
<td>50</td>
<td>52</td>
<td>39</td>
<td>59</td>
</tr>
<tr>
<td>RT-1</td>
<td>100</td>
<td>37</td>
<td>55</td>
<td>46</td>
<td>57</td>
<td>35</td>
<td>47</td>
</tr>
<tr>
<td>RT-1</td>
<td>100</td>
<td>22</td>
<td>59</td>
<td>29</td>
<td>14</td>
<td>31</td>
<td>41</td>
</tr>
<tr>
<td colspan="8"><b>Narrower Data (数据多样性减少)</b></td>
</tr>
<tr>
<td><b>RT-1 (ours)</b></td>
<td>100</td>
<td>100</td>
<td><b>97</b></td>
<td><b>73</b></td>
<td><b>76</b></td>
<td><b>83</b></td>
<td><b>59</b></td>
</tr>
<tr>
<td>RT-1</td>
<td><b>75</b></td>
<td>97</td>
<td>86</td>
<td>54</td>
<td>67</td>
<td>42</td>
<td>53</td>
</tr>
</tbody>
</table>

**分析：**
*   随着数据量减少，所有指标都下降，这符合预期。
*   **关键发现：** 观察 `Narrower Data` 行，<strong>仅仅移除 25% 的任务（只减少了 3% 的总数据量），泛化性能 (All) 就从 73% 掉到了 54%</strong>。这个掉幅与将数据量减少近一半（从 100% 到 51%）带来的掉幅（从 73% 到 50%）相当。
*   **结论：** 对于提升模型的泛化能力，<strong>数据多样性（任务种类的丰富程度）比单纯的数据量更重要</strong>。

### 6.2.2. 模型组件消融
以下是原文附录 Table 13 中对 RT-1 模型不同组件进行消融的结果。

<table>
<thead>
<tr>
<th>Model</th>
<th>Seen Tasks</th>
<th>Unseen Tasks</th>
<th>Distractors</th>
<th>Backgrounds</th>
<th>Inference Time (ms)</th>
</tr>
</thead>
<tbody>
<tr>
<td><b>RT-1 (ours)</b></td>
<td><b>97</b></td>
<td><b>76</b></td>
<td><b>83</b></td>
<td><b>59</b></td>
<td><b>15</b></td>
</tr>
<tr>
<td>RT-1 w/o pre-training</td>
<td>84 (-13)</td>
<td>43 (<b>-33</b>)</td>
<td>60 (-23)</td>
<td>41 (-18)</td>
<td>15</td>
</tr>
<tr>
<td>RT-1 w/ continuous actions</td>
<td>68 (<b>-29</b>)</td>
<td>43 (<b>-33</b>)</td>
<td>37 (<b>-46</b>)</td>
<td>35 (-24)</td>
<td>16</td>
</tr>
<tr>
<td>RT-1 w/o history</td>
<td>82 (-15)</td>
<td>62 (-14)</td>
<td>50 (<b>-33</b>)</td>
<td>59 (+0)</td>
<td>15</td>
</tr>
<tr>
<td>RT-1 w/o Transformer</td>
<td>86 (-13)</td>
<td>62 (-14)</td>
<td>67 (-16)</td>
<td>59 (+0)</td>
<td>26</td>
</tr>
<tr>
<td>RT-1 w/ auto-regressive actions</td>
<td>85 (-12)</td>
<td>71 (-5)</td>
<td>67 (-16)</td>
<td>65 (+6)</td>
<td><b>36 (>2x)</b></td>
</tr>
</tbody>
</table>

**分析：**
*   <strong>ImageNet 预训练 (`w/o pre-training`)</strong> 至关重要，移除后泛化能力（未见任务）暴跌 33%。
*   <strong>离散化动作 (`w/ continuous actions`)</strong> 是 RT-1 成功的另一个关键。换回连续动作预测后，所有指标全面大幅下降，尤其是在有干扰物的场景下（-46%），这表明离散化动作能更好地表示复杂和多模态的专家行为。
*   <strong>历史信息 (`w/o history`)</strong> 对处理动态和有遮挡的场景（如有干扰物）至关重要，移除后干扰物鲁棒性下降 33%。
*   <strong>Transformer 主干 (`w/o Transformer`)</strong> 提供了全面的性能提升，移除后各项指标均有下降。
*   <strong>动作自回归生成 (`w/ auto-regressive actions`)</strong> 对性能提升不大，但导致推理时间翻倍，验证了 RT-1 采用并行非自回归设计的正确性。

    ---

# 7. 总结与思考

## 7.1. 结论总结
本文成功地展示了将大规模、高容量模型与大规模、多样化的真实世界数据相结合，是推动机器人学习向前发展的有效路径。
1.  **RT-1 模型**被证明是一个强大且高效的机器人控制架构，它通过 `FiLM`、`TokenLearner` 等设计，在 Transformer 的强大表达能力与机器人实时控制的效率需求之间取得了出色的平衡。
2.  实验无可辩驳地证明了<strong>规模化 (scaling)</strong> 的力量：更大的模型在更多样化的数据上训练，能够获得前所未有的**泛化能力**和**鲁棒性**，在超过 700 种任务上达到了 97% 的高成功率，并能零样本泛化到新任务、新环境。
3.  RT-1 的<strong>数据吸收 (data absorption)</strong> 能力是一项突破性发现。它能够有效地利用来自仿真、甚至来自不同形态机器人的异构数据来提升自身能力，为未来构建“机器人基础模型”提供了一条可行的技术路线——即汇集来自全球不同机器人平台的数据进行联合训练。
4.  研究还给出了一个重要的实践指导：对于机器人学习，**数据多样性比单纯的数据量更关键**。

## 7.2. 局限性与未来工作
作者坦诚地指出了当前工作的局限性：
*   **模仿学习的上限：** 作为一个基于模仿学习的方法，RT-1 的性能理论上无法超越提供演示的人类专家。
*   **泛化能力的边界：** 模型的泛化主要体现在对已知概念的**新组合**上，它还无法创造出训练数据中完全没有见过的**全新动作或技能**。
*   **任务复杂性：** 尽管任务数量多，但主要集中在桌面操作（table-top manipulation），尚未涉及更灵巧、更动态的操作（如叠衣服、使用工具）。

    未来的工作将围绕以下方向展开：
*   **加速技能扩展：** 开发更高效的数据收集方法，例如让非专家用户通过模型提示 (prompting) 来指导机器人学习。
*   **提升环境多样性：** 在更多、更丰富的环境中收集数据，以进一步增强模型的背景鲁棒性。
*   **改进模型架构：** 探索可扩展的注意力和记忆机制，以提升模型的反应速度和长时程记忆能力。

## 7.3. 个人启发与批判
*   **启发：**
    1.  <strong>“大力出奇迹”</strong>在机器人领域的可能性： 这篇论文强有力地证明了，困扰机器人领域已久的泛化问题，确实可以通过“堆数据”和“增大模型”来大力缓解。它为机器人领域注入了与 NLP/CV 领域相似的乐观主义，即构建通用的“机器人基础模型”是可能的。
    2.  **架构设计的重要性：** RT-1 并非简单地将一个标准 Transformer 应用于机器人，而是进行了一系列针对性的设计（FiLM, TokenLearner, 动作离散化, 非自回归输出）。这些“魔改”对于在满足实时性约束的同时最大化模型性能至关重要，为后续研究提供了宝贵的工程经验。
    3.  **异构数据融合的未来：** RT-1 能够融合仿真和其他机器人的数据，这描绘了一个激动人心的未来。或许有一天，我们可以创建一个全球性的机器人数据联盟，将来自不同公司、不同实验室、不同机器人平台的数据汇集起来，共同训练一个无所不能的机器人大脑。

*   **批判与思考：**
    1.  **成本与可复现性：** 这项研究的成功建立在谷歌巨大的资源投入之上（13台机器人，17个月的数据收集）。这使得学术界的小型实验室几乎无法复现或跟进类似规模的研究，可能会加剧产学界的资源鸿沟。
    2.  **动作离散化的双刃剑：** 离散化动作是 RT-1 成功的关键之一，但它也引入了新的问题。例如，256 个 bin 的粒度是否足够精细？对于需要高精度控制的任务，这可能会成为瓶颈。此外，bin 的边界是如何确定的？这个超参数的选择是否会对性能产生很大影响？
    3.  **对物理理解的缺失：** 尽管 RT-1 表现出色，但它仍然是一个“黑箱”的模式匹配系统。它并不真正“理解”物理世界，比如重力、摩擦或物体材质。因此，它在面对训练分布之外的、需要物理推理的全新情境时，可能仍然会失效。未来的研究需要将这种数据驱动的方法与基于模型的、包含物理先验知识的方法更好地结合起来。
    4.  **安全与可靠性：** 97% 的成功率在研究中已经非常高，但在现实世界的家庭或工业应用中，3% 的失败率可能意味着灾难。如何保证这类大型模型在部署时的安全性和可靠性，是一个悬而未决的重大挑战。