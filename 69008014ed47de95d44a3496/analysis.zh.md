# 1. 论文基本信息

## 1.1. 标题
DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (DeepSeekMath: 推动开放语言模型在数学推理能力上的极限)

## 1.2. 作者
论文作者团队来自 DeepSeek-AI、清华大学和北京大学。主要作者包括 Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, 和 Daya Guo。他们的隶属机构表明，这项研究是产学研紧密合作的成果，结合了工业界强大的工程能力和学术界的科研实力。

## 1.3. 发表期刊/会议
该论文以预印本 (Pre-print) 的形式发布在 arXiv 上。arXiv 是一个开放获取的学术论文发布平台，广泛用于物理学、数学、计算机科学等领域的研究者在同行评审前快速分享其最新研究成果。这篇论文的发布状态意味着它代表了作者团队的最新进展，但尚未经过正式的同行评审流程。

## 1.4. 发表年份
2024年

## 1.5. 摘要
数学推理因其复杂和结构化的特性，对语言模型构成了重大挑战。本文介绍了 `DeepSeekMath 7B` 模型，该模型在 `DeepSeek-Coder-Base-v1.5 7B` 的基础上，使用了从 Common Crawl 中筛选出的 1200 亿（120B）数学相关词元 (token)，并结合了自然语言和代码数据进行持续预训练。`DeepSeekMath 7B` 在竞赛级的 `MATH` 基准测试上取得了 **51.7%** 的惊人分数，且未使用外部工具包或投票技术，其性能已接近 `Gemini-Ultra` 和 `GPT-4` 的水平。通过对 64 个样本进行自洽性 (Self-consistency) 验证，其在 `MATH` 上的得分可达 60.9%。`DeepSeekMath` 的卓越数学推理能力归功于两大关键因素：首先，通过精心设计的数据筛选流程，充分利用了公开网络数据中的巨大潜力；其次，本文提出了一种名为<strong>组相对策略优化 (Group Relative Policy Optimization, GRPO)</strong> 的算法，它是近端策略优化 (Proximal Policy Optimization, PPO) 的一个变体，该算法在提升数学推理能力的同时，还优化了 PPO 的内存使用。

## 1.6. 原文链接
*   **原文链接:** [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
*   **PDF 链接:** [https://arxiv.org/pdf/2402.03300v3.pdf](https://arxiv.org/pdf/2402.03300v3.pdf)
*   **发布状态:** 预印本 (Pre-print)

    ---

# 2. 整体概括

## 2.1. 研究背景与动机
*   **核心问题:** 当前，大语言模型 (LLMs) 在处理需要严格逻辑和多步推理的数学问题时仍然面临巨大挑战。数学推理能力是衡量模型高级认知能力的关键指标。
*   <strong>领域空白 (Gap):</strong> 尽管如 `GPT-4` 和 `Gemini-Ultra` 等顶尖模型在数学推理上表现出色，但它们是闭源的，其技术细节和训练数据不为公众所知，这限制了学术界对它们的深入研究。与此同时，现有的开源模型在数学能力上远远落后于这些闭源巨头，存在巨大的性能鸿沟。
*   **创新切入点:** 本文旨在缩小这一差距，其核心思路是：
    1.  **数据驱动:** 不依赖于昂贵的专有数据，而是探索如何从公开、海量的网络数据（如 `Common Crawl`）中“淘金”，构建一个规模空前且高质量的数学预训练语料库。
    2.  **算法优化:** 提出一种更高效的强化学习算法 `GRPO`，以更低的资源成本（特别是内存）来对模型进行对齐，使其更擅长生成正确的数学解题步骤。
    3.  **模型基础:** 验证了一个重要假设——从一个强大的代码模型 (`DeepSeek-Coder`) 出发，比从通用语言模型出发，更能有效地学习数学推理。

## 2.2. 核心贡献/主要发现
这篇论文的贡献是多方面的，涵盖了数据、模型、算法和实验洞见。

*   **大规模数学语料库的构建:**
    *   提出了一个精心设计的**迭代式数据筛选流程**，并成功从 `Common Crawl` 中构建了 `DeepSeekMath Corpus`，一个包含 1200 亿词元的高质量数学网络文本数据集。其规模远超之前的同类工作（如 `Minerva` 和 `OpenWebMath`）。
    *   证明了公开网络数据是训练强大数学模型的宝贵资源，为社区提供了可扩展的数据获取方案。

*   **强大的开源数学模型:**
    *   发布了 `DeepSeekMath` 系列模型（包括 `Base`, `Instruct`, 和 `RL` 版本）。其中，`DeepSeekMath-RL 7B` 成为首个在 `MATH` 基准上准确率突破 50% 的开源模型，性能达到了与顶级闭源模型相近的水平。
    *   证明了模型参数量并非唯一决定因素，一个 7B 的小模型在高质量、大规模的数据上训练后，其性能可以超越参数量大几十倍的模型（如 `Minerva 540B`）。

*   **创新的强化学习算法:**
    *   提出了 <strong>组相对策略优化 (Group Relative Policy Optimization, GRPO)</strong> 算法。该算法通过在采样组内计算相对奖励，<strong>摒弃了传统 PPO 中消耗大量内存的价值模型（critic model）</strong>，显著降低了训练资源需求，使得在有限资源下进行高效强化学习成为可能。

*   **深刻的实验洞见:**
    *   **代码训练的价值:** 实验证明，在进行数学训练之前先进行代码训练，能够显著提升模型在有工具和无工具场景下的数学解题能力，为“代码训练能否提升推理能力”这一长期问题提供了肯定的经验证据。
    *   **arXiv 数据的反思:** 令人意外地发现，在本文所采用的基准测试中，使用 `arXiv` 论文数据进行训练并未带来显著的性能提升，甚至在某些情况下出现性能下降，这对以往的研究范式提出了挑战。
    *   **强化学习机制的剖析:** 提出了一个统一的范式来理解 SFT、RFT、DPO、PPO 等多种对齐方法，并发现强化学习的有效性更多体现在**提升模型生成正确答案的概率（提高 `Maj@K`）**，而非从根本上提升其解决新问题的能力（`Pass@K` 提升不明显）。

        ---

# 3. 预备知识与相关工作

## 3.1. 基础概念
*   <strong>大语言模型 (Large Language Models, LLMs):</strong> 指的是参数量巨大（通常在十亿以上）的深度学习模型，通常基于 `Transformer` 架构。它们通过在海量文本数据上进行“预训练”(pre-training) 来学习语言的通用规律，然后在特定任务的数据上进行“微调”(fine-tuning) 来适应具体应用。
*   <strong>数学推理基准 (Mathematical Reasoning Benchmarks):</strong>
    *   `GSM8K`: 包含约 8500 个高质量的小学水平数学应用题。题目需要模型进行 2-8 步的推理才能解决，答案是最终的数字。
    *   `MATH`: 一个更具挑战性的数据集，包含 12500 个来自高中竞赛级别的数学问题，涵盖代数、几何、数论、微积分等 7 个领域。
    *   `MMLU (Massive Multitask Language Understanding)`: 一个综合性基准，包含 57 个科目（从初级数学到美国历史）的多项选择题，用于评估模型的广泛知识和问题解决能力。`MMLU-STEM` 是其中与科学、技术、工程和数学相关的子集。
*   <strong>思维链 (Chain-of-Thought, CoT):</strong> 一种提示 (prompting) 技术。它不要求模型直接给出答案，而是引导模型输出详细的、一步一步的推理过程，最后再给出答案。这种方式模仿了人类的思考过程，能显著提高模型在复杂推理任务上的准确性。
*   <strong>思维程序 (Program-of-Thought, PoT):</strong> `CoT` 的一种变体。它引导模型生成一段可执行的代码（如 Python 程序）来解决问题。模型将问题分解为代码逻辑，利用代码的精确计算能力来得出最终答案，特别适用于需要复杂数值计算的问题。
*   <strong>强化学习 (Reinforcement Learning, RL):</strong> 机器学习的一个分支，智能体 (agent) 通过与环境互动来学习。智能体执行一个动作 (action)，环境会给予一个奖励 (reward) 或惩罚 (penalty)，智能体的目标是学习一个策略 (policy) 来最大化长期累积奖励。在 LLM 中，RL 常用于“对齐”，即让模型的输出更符合人类的偏好或特定标准（如正确性）。

## 3.2. 前人工作
*   **闭源模型:**
    *   `GPT-4` (OpenAI, 2023) 和 `Gemini-Ultra` (Google, 2023): 代表了当前最先进水平的闭源 LLM，在各项任务（包括数学）上都表现出极强的能力，但其技术细节保密。
    *   `Minerva` (Google, 2022): 一个专门为数学和科学问题设计的 LLM，它在 `PaLM` 模型的基础上，用 118GB 的科学论文和包含数学表达式的网页数据进行微调，证明了领域数据对提升数学能力的重要性。
*   **开源模型:**
    *   `Llemma` (Azerbayev et al., 2023): 一个专注于数学的开源模型，它在 `Code Llama` 的基础上，使用一个名为 `Proof-Pile-2` 的数据集（包含网页、代码和 `arXiv` 论文）进行持续训练。
    *   `WizardMath` (Luo et al., 2023): 通过一种名为 `Evol-Instruct` 的方法自动生成复杂的数学指令，并使用强化学习（PPO）对 `Llama-2` 和 `Mistral` 等模型进行微调，以提升数学推理能力。
*   <strong>对齐算法 (Alignment Algorithms):</strong>
    *   <strong>近端策略优化 (Proximal Policy Optimization, PPO):</strong> 一种广泛用于 LLM 对齐的强化学习算法。它采用<strong>演员-评论家 (Actor-Critic)</strong> 架构。<strong>演员 (Actor)</strong> 是需要优化的 LLM 本身，它生成回答。<strong>评论家 (Critic)</strong> 是一个额外的模型（通常称为价值模型），用于评估演员生成的每一步回答的“好坏”（即价值），从而指导演员的更新。PPO 的目标是最大化奖励，同时通过一个“信任区域”约束，防止策略更新过快导致训练不稳定。其标准目标函数如下：
        $$
        L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
        $$
        其中，$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是新旧策略的概率比，$\hat{A}_t$ 是优势函数（Advantage Function），表示在状态 $s_t$ 下采取动作 $a_t$ 比平均水平好多少，$\epsilon$ 是裁剪系数。
    *   <strong>拒绝采样微调 (Rejection Sampling Fine-Tuning, RFT):</strong> 一种简单的对齐方法。让模型对一个问题生成多个答案，然后用一个外部标准（如检查答案是否正确）过滤出“好”的答案，最后用这些“好”的答案对模型进行微调。
    *   <strong>直接偏好优化 (Direct Preference Optimization, DPO):</strong> 一种更新颖的对齐方法，它不需要训练一个独立的奖励模型。DPO 直接利用偏好数据（即哪个回答更好）来优化 LLM，其目标是让模型对于“更好”的回答给出更高的概率，而对于“更差”的回答给出更低的概率。

## 3.3. 技术演进
该领域的技术演进脉络清晰：
1.  **通用 LLM 阶段:** 早期的 LLM（如 GPT-2/3）在通用文本上预训练，具备一定的零样本/少样本推理能力，但解决复杂数学问题能力有限。
2.  **领域数据微调阶段:** `Minerva` 等工作证明，使用特定领域（如数学、科学）的数据进行持续预训练或微调，可以显著提升模型在该领域的能力。
3.  <strong>指令微调 (Instruction Tuning) 与思维链 (CoT) 阶段:</strong> 通过在“问题-解题步骤-答案”格式的数据上进行微调，模型学会了 `CoT` 推理，这极大地增强了其解决多步问题的能力。
4.  **强化学习对齐阶段:** `InstructGPT` 和后续工作表明，在指令微调之后，使用基于人类反馈的强化学习 (RLHF) 可以进一步“对齐”模型，使其输出更符合人类偏好。`WizardMath` 将此思想应用于数学领域，使用奖励模型来引导模型生成更正确的解题步骤。
5.  **本文工作:** `DeepSeekMath` 处在这一脉络的前沿。它不仅将数据驱动的思路推向了新的高度（120B 网页数学数据），还对强化学习算法本身进行了创新（提出 `GRPO`），并结合了代码预训练的优势，形成了一套系统性的解决方案。

## 3.4. 差异化分析
*   **与 `Minerva` 和 `Llemma` 的对比:**
    *   **数据来源与规模:** `DeepSeekMath` 的核心数据来自通用网络爬虫 `Common Crawl`，并通过迭代式筛选获得了 120B 词元的超大规模语料。而 `Minerva` 和 `Llemma` 的数学数据更多依赖于 `arXiv` 和特定数学网站，规模相对较小。`DeepSeekMath` 证明了通用网络中蕴含着巨大的、可被挖掘的数学知识。
    *   **模型起点:** `DeepSeekMath` 明确地选择从一个强大的代码模型 `DeepSeek-Coder` 开始训练，而 `Minerva` 从通用模型 `PaLM` 开始，`Llemma` 从 `Code Llama` 开始。本文通过实验验证了代码预训练对数学推理的积极作用。
*   **与 `WizardMath` 的对比:**
    *   **强化学习算法:** `WizardMath` 使用标准的 PPO 算法，需要一个独立的、与策略模型大小相当的价值模型，内存开销大。而 `DeepSeekMath` 提出的 `GRPO` **无需价值模型**，通过组内奖励的相对比较来计算优势，大大降低了内存消耗，更具效率。
    *   **数据基础:** `DeepSeekMath` 的成功首先建立在极其强大的预训练基础上，而 `WizardMath` 更多地侧重于通过指令演进和 RL 来提升现有基础模型的能力。这体现了“强大的基础模型 + 高效的对齐”双轮驱动的思路。

        ---

# 4. 方法论

本文的方法论主要包含三个阶段：**1) 大规模数学预训练**，**2) 监督微调**，以及 **3) 基于 GRPO 的强化学习**。

## 4.1. 方法原理
方法的核心思想是“**数据、模型、算法**”三位一体：
*   <strong>数据 (Data):</strong> 相信公开网络数据中蕴藏着足够的高质量数学知识，关键在于如何有效地将其筛选和提炼出来。
*   <strong>模型 (Model):</strong> 相信代码与数学在底层逻辑结构上的共通性，因此从一个强大的代码模型开始训练，可以为学习数学推理提供更好的起点。
*   <strong>算法 (Algorithm):</strong> 相信强化学习是提升模型推理能力的关键步骤，但标准的 PPO 算法资源消耗过大。因此，需要设计一种更轻量、高效的 RL 算法，使其能够在有限的资源下发挥最大作用。

## 4.2. 核心方法详解 (逐层深入)

### 4.2.1. 第一阶段：大规模数学预训练

#### 4.2.1.1. DeepSeekMath 语料库的构建
构建高质量、大规模的数学语料库是本文成功的基石。作者设计了一个迭代式的流程来从 `Common Crawl` 的海量网页中筛选数学内容，如下图（原文 Figure 2）所示：

![Figure 2 | An iterative pipeline that collects mathematical web pages from Common Crawl.](images/2.jpg)

**步骤 1: 种子语料库与初始分类器训练**
*   **种子数据:** 以一个已知的、高质量的数学网络文本数据集 `OpenWebMath` 作为初始的“正样本”种子。
*   **训练分类器:** 从种子语料库中随机抽取 50万 条数据作为正样本，再从 `Common Crawl` 中随机抽取 50万 条网页作为负样本。使用这些数据训练一个 `fastText` 文本分类器。`fastText` 是一种快速、高效的文本分类模型，非常适合处理海量数据。

**步骤 2: 挖掘、排序与筛选**
*   **挖掘:** 将训练好的 `fastText` 分类器应用于经过 URL 去重的 400 亿（40B）个 `Common Crawl` 网页，预测每个网页包含数学内容的可能性得分。
*   **排序与筛选:** 根据分类器给出的分数对所有网页进行排序，只保留分数最高的网页作为候选的数学内容。在第一轮迭代中，作者保留了处理后约 400 亿词元的数据。

**步骤 3: 迭代式优化分类器**
*   **发现新数学源:** 仅靠初始种子，分类器可能无法覆盖所有类型的数学网页。为了扩大覆盖范围，作者首先按域名（如 `mathoverflow.net`）对 `Common Crawl` 进行分组。然后，计算在上一轮被筛选出的网页在每个域名中的占比。如果一个域名的被采纳比例超过 10%，则该域名被标记为“数学相关”。
*   **人工标注与扩充种子:** 在这些数学相关域名下，人工标注出包含数学内容的具体 URL 路径（如 $mathoverflow.net/questions$）。将这些路径下未被分类器召回的网页补充到种子语料库中，作为新的正样本。
*   **重新训练:** 使用扩充后的种子语料库重新训练 `fastText` 分类器，使其能够识别更多样化的数学内容。

**步骤 4: 终止迭代**
*   重复步骤 2 和 3。作者发现，在第四轮迭代中，有 98% 的数据已经在第三轮中被收集，表明数据收集过程已接近收敛。此时，停止迭代。最终，该流程收集了 3550 万个数学网页，总计 1200 亿词元，构成了 `DeepSeekMath Corpus`。

<strong>数据去污染 (Decontamination):</strong>
为了避免在训练集中泄露评测数据，作者采用了严格的 n-gram 匹配方法。任何包含与评测集（如 `GSM8K`, `MATH` 等）中 10-gram（10个连续词元）完全匹配的文本段落都会被从训练语料中移除。

#### 4.2.1.2. DeepSeekMath-Base 模型训练
*   **初始化:** 模型并非从零开始，而是以一个强大的代码预训练模型 `DeepSeek-Coder-Base-v1.5 7B` 作为起点。
*   **数据混合:** 在 5000 亿词元的持续预训练阶段，数据配比为：
    *   `DeepSeekMath Corpus`: 56%
    *   `AlgebraicStack` (数学代码): 4%
    *   `arXiv`: 10%
    *   `Github` 代码: 20%
    *   自然语言 (中英文): 10%
        这种混合策略旨在增强数学能力的同时，保持并利用其强大的代码和通用推理能力。

### 4.2.2. 第二阶段：监督微调 (Supervised Fine-Tuning, SFT)
在预训练获得强大的基础能力后，模型需要通过 SFT 来学习遵循指令和特定的输出格式（如 CoT）。
*   **SFT 数据:** 作者构建了一个包含 77.6 万个样本的数学指令微调数据集，覆盖中英文、不同难度和领域。数据格式多样，包括：
    *   <strong>思维链 (CoT):</strong> 自然语言的解题步骤。
    *   <strong>思维程序 (PoT):</strong> Python 代码的解题过程。
    *   **工具集成推理:** 结合自然语言和代码工具的解题方式。
*   **SFT 训练:** 基于 `DeepSeekMath-Base` 模型，使用上述数据进行微调，得到 `DeepSeekMath-Instruct 7B` 模型。

### 4.2.3. 第三阶段：强化学习 (Reinforcement Learning, RL)

为了进一步提升模型的数学推理能力，作者在 SFT 模型的基础上进行了强化学习。核心创新是提出了 `GRPO` 算法。下图（原文 Figure 4）直观对比了 PPO 和 GRPO 的区别。

![Figure 4 | Demonstration of PPO and our GRPO. GRPO foregoes the value model, instead estimating the baseline from group scores, significantly reducing training resources.](images/4.jpg)
*该图像是论文中图4的示意图，展示了传统PPO与本文提出的GRPO两种训练策略的流程对比。GRPO摈弃了价值模型，采用基于组得分的基线估计，显著降低了训练资源消耗。*

#### 4.2.3.1. 从 PPO 到 GRPO 的演进
首先，我们回顾 PPO。在 LLM 的 RL 训练中，PPO 的优化目标可以写为：
$$
\mathcal { T } _ { P P O } ( \theta ) = \mathbb { E } [ q \sim P ( Q ) , o \sim \pi_ { \theta _ { o l d } } ( O | q ) ] \frac { 1 } { | o | } \sum _ { t = 1 } ^ { | o | } \operatorname* { m i n } \left[ \frac { \pi _ { \theta } ( o _ { t } | q , o _ { < t } ) } { \pi _ { \theta _ { o l d } } ( o _ { t } | q , o _ { < t } ) } A _ { t } , \mathrm { c l i p } \left( \frac { \pi _ { \theta } ( o _ { t } | q , o _ { < t } ) } { \pi _ { \theta _ { o l d } } ( o _ { t } | q , o _ { < t } ) } , 1 - \varepsilon , 1 + \varepsilon \right) A _ { t } \right]
$$
**符号解释:**
*   $\pi_\theta$: 当前正在优化的策略模型（即 LLM）。
*   $\pi_{\theta_{old}}$: 上一轮的策略模型，用于采样数据。
*   `q, o`: 分别是问题 (question) 和模型生成的输出 (output)。
*   $\varepsilon$: 裁剪系数，用于稳定训练。
*   $A_t$: <strong>优势 (Advantage)</strong>，表示在时间步 $t$ 生成词元 $o_t$ 比平均水平好多少。它的计算依赖于一个<strong>价值模型 (Value Model) / 评论家 (Critic)</strong> $V_\psi$，这个价值模型需要额外训练，且通常与策略模型大小相当，**消耗大量内存**。

**GRPO 的核心思想：**
GRPO 通过一种巧妙的方式**完全移除了价值模型**。它的直觉是：对于同一个问题，我们可以让模型生成一组（比如 64 个）不同的答案。通过比较这一组答案的最终奖励（比如答案是否正确），我们可以知道哪些答案是“好”的，哪些是“坏”的。那么，<strong>这组答案的平均奖励就可以作为一个基线 (baseline)</strong>。一个答案的奖励如果高于这个平均值，就应该被鼓励；如果低于平均值，就应该被抑制。这样，我们就不再需要一个独立的价值模型来估计“平均水平”了。

#### 4.2.3.2. GRPO 目标函数
GRPO 的优化目标如下（原文公式 3 所在的部分，这里为了清晰呈现其核心，我们使用 $o_i$ 代表第 $i$ 个输出）：
$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)} \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min[\dots] - \beta \mathbb{D}_{KL}[\pi_\theta || \pi_{ref}] \right\}
$$
其中，`min[...]` 部分与 PPO 类似，但关键区别在于优势 $\hat{A}_{i,t}$ 的计算以及 KL 散度惩罚项。
*   **符号解释:**
    *   $G$: 每个问题采样的输出数量（组的大小，如 64）。
    *   $\{o_i\}_{i=1}^G$: 对问题 $q$ 采样出的 $G$ 个输出组成的组。
    *   $\hat{A}_{i,t}$: <strong>组相对优势 (Group Relative Advantage)</strong>。它不再由价值模型计算，而是直接由组内的奖励得分计算得出。
    *   $\mathbb{D}_{KL}[\pi_\theta || \pi_{ref}]$: 当前策略 $\pi_\theta$ 与参考策略 $\pi_{ref}$ (通常是 SFT 模型) 之间的 KL 散度，用于防止模型偏离原始分布太远。

#### 4.2.3.3. 两种监督方式
GRPO 支持两种不同的奖励计算方式，以提供不同粒度的监督信号：
*   <strong>结果监督 (Outcome Supervision, OS):</strong>
    1.  对一个问题 $q$，采样 $G$ 个完整输出 $\{o_1, \dots, o_G\}$。
    2.  用一个奖励模型（或规则）为每个完整输出打分，得到奖励 $\{r_1, \dots, r_G\}$。
    3.  对这组奖励进行标准化（减去均值，除以标准差）。
    4.  对于输出 $o_i$ 中的**所有词元**，它们的优势 $\hat{A}_{i,t}$ 都被设为这个标准化的最终奖励。这是一种粗粒度的监督。
*   <strong>过程监督 (Process Supervision, PS):</strong>
    1.  对每个输出 $o_i$ 的**每一个推理步骤**（例如，CoT 中的一句话）都进行打分，得到一系列步骤奖励 $\{r_i^{\text{step1}}, r_i^{\text{step2}}, \dots\}$。
    2.  将**所有输出的所有步骤奖励**放在一起进行标准化。
    3.  对于输出 $o_i$ 中的第 $t$ 个词元，其优势 $\hat{A}_{i,t}$ 被计算为它**之后所有步骤**的标准化奖励之和。即：
        $$\hat{A}_{i,t} = \sum_{\text{step\_j 的起始位置} \ge t} \tilde{r}_i^{\text{step\_j}}$$
        其中 $\tilde{r}$ 是标准化的步骤奖励。这种方式提供了更细粒度的监督，能更精确地定位错误发生在哪一步。

#### 4.2.3.4. 迭代式强化学习
随着策略模型的不断优化，初始的奖励模型可能不再能准确评估新模型的输出。因此，作者采用了迭代式 RL 流程 (原文 Algorithm 1)：
1.  **训练循环:** 使用当前策略模型 $\pi_\theta$ 和奖励模型 $r_\phi$ 进行 GRPO 训练。
2.  **数据生成:** 训练一段时间后，使用更新后的策略模型 $\pi_\theta$ 生成新的输出样本。
3.  **奖励模型更新:** 用这些新样本来继续训练（或重新训练）奖励模型 $r_\phi$，使其能跟上策略模型的进步。
4.  **重置参考模型:** 将当前策略模型 $\pi_\theta$ 设为新的参考模型 $\pi_{ref}$。
5.  **返回步骤 1:** 用更新后的奖励模型和新的参考模型开始下一轮 GRPO 训练。

    ---

# 5. 实验设置

## 5.1. 数据集
论文在多个维度上对模型进行了全面的评估，使用了以下数据集：

*   **英文数学推理:**
    *   `GSM8K`: 小学水平数学应用题。
    *   `MATH`: 竞赛级数学问题，难度很高。
    *   `SAT`: 美国大学入学考试中的数学部分。
    *   `OCW Courses`: 从麻省理工学院开放课程 (MIT OpenCourseWare) 中提取的大学水平数学问题。
    *   `MMLU-STEM`: `MMLU` 基准中与科学、技术、工程、数学相关的部分。
*   **中文数学推理:**
    *   `MGSM-zh`: `GSM8K` 的中文翻译版本。
    *   `CMATH`: 中国小学数学问题集。
    *   `Gaokao-MathCloze` / `Gaokao-MathQA`: 中国高考数学的完形填空和问答题。
*   <strong>形式化数学 (Formal Mathematics):</strong>
    *   `miniF2F`: 一个用于奥林匹克级别数学问题形式化证明的基准。实验任务是将非形式化的证明转换为 `Isabelle` 证明助手可识别的形式化证明。
*   **通用能力评估:**
    *   `MMLU`: 评估模型的广泛知识和多任务理解能力。
    *   `BBH (BIG-Bench Hard)`: 包含 23 个挑战性任务，大多需要多步推理。
    *   `HumanEval` / `MBPP`: 评估模型代码生成能力的常用基准。

        选择这些数据集是为了全面评估模型在不同语言、不同难度（从小学到大学竞赛）、不同题型（应用题、选择题、形式化证明）以及通用推理和编码能力上的表现。

## 5.2. 评估指标
论文中主要使用了以下评估指标：

*   <strong>准确率 (Accuracy / Top-1 Accuracy):</strong>
    1.  **概念定义:** 这是最直观的指标，衡量模型直接生成的第一个答案的正确率。即模型对于一个问题，只生成一次回答，这个回答被判定为正确的题目数占总题目数的比例。
    2.  **数学公式:**
        $$
        \text{Accuracy} = \frac{\text{Number of Correct Answers}}{\text{Total Number of Problems}}
        $$
    3.  **符号解释:**
        *   `Number of Correct Answers`: 模型回答正确的题目数量。
        *   `Total Number of Problems`: 评测集中的总题目数量。

*   **Pass@k:**
    1.  **概念定义:** 这个指标主要用于评估代码生成任务，但也适用于可以自动验证答案的推理任务。它衡量的是：从模型生成的 $k$ 个独立样本中，**至少有一个**是正确的概率。`Pass@1` 就等同于 Top-1 准确率。该指标能更好地评估模型生成正确解的“潜力”，而不是单次尝试的成功率。
    2.  **数学公式:** 由于直接计算概率很困难，通常使用以下无偏估计量：
        $$
        \text{Pass@k} \approx 1 - \frac{\mathbb{E}[\text{number of incorrect samples among } n \text{ samples}]}{\mathbb{E}[\text{number of samples}]} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}
        $$
    3.  **符号解释:**
        *   $n$: 为每个问题生成的总样本数（通常 $n > k$）。
        *   $c$: 在 $n$ 个样本中，正确样本的数量。
        *   $k$: 我们考虑的样本窗口大小。
        *   $\binom{n}{k}$: 组合数，表示从 $n$ 个元素中取 $k$ 个的组合方式数。

*   **Maj@k (Majority Voting @ k):**
    1.  **概念定义:** 也称为多数投票。该方法让模型对一个问题生成 $k$ 个独立的答案。然后，统计这 $k$ 个答案中出现频率最高的那个作为最终答案。如果这个最终答案是正确的，则认为模型回答正确。这个指标评估的是模型输出的“鲁棒性”或“一致性”。
    2.  **数学公式:** 没有统一的数学公式，其过程为算法描述：
        1.  For each problem, generate $k$ solutions $\{s_1, s_2, \dots, s_k\}$.
        2.  Extract final answers $\{a_1, a_2, \dots, a_k\}$ from each solution.
        3.  $a_{final} = \text{argmax}_{a} \text{Count}(a \in \{a_1, \dots, a_k\})$.
        4.  Score = 1 if $a_{final}$ is correct, 0 otherwise.
        5.  $\text{Maj@k Accuracy} = \frac{\sum \text{Score}}{\text{Total Number of Problems}}$.
    3.  **符号解释:**
        *   $k$: 生成的样本数量。
        *   $a_{final}$: 通过多数投票选出的最终答案。

## 5.3. 对比基线
论文将 `DeepSeekMath` 与一系列顶尖的开源和闭源模型进行了比较，这些基线具有很强的代表性。
*   **闭源模型:** `GPT-4`, `GPT-4 Code Interpreter`, `Gemini Ultra`, `Gemini Pro`, `Inflection-2`, `GLM-4` 等。这些是业界公认的 SOTA 模型，代表了当前能力的上限。
*   **开源模型:**
    *   **通用模型:** `DeepSeek-LLM-Chat 67B`, `Qwen 72B`, `Mistral 7B` 等。这些是当前最流行的通用开源模型。
    *   **数学增强模型:** `InternLM2-Math 20B`, `Math-Shepherd-Mistral 7B`, `WizardMath` 系列, `MetaMath 70B`, `ToRA 34B`, `MAmmoTH 70B`。这些模型都经过了针对数学能力的专门优化，是 `DeepSeekMath` 最直接的竞争对手。

        ---

# 6. 实验结果与分析

## 6.1. 核心结果分析
论文通过一系列详尽的实验，系统地展示了其方法在各个阶段的有效性。

### 6.1.1. 预训练数据质量验证
为了验证 `DeepSeekMath Corpus` 的高质量，作者在 1.3B 参数量的模型上进行了对比实验。结果如原文 Table 1 所示：

<table>
<thead>
<tr>
<th rowspan="2">Math Corpus</th>
<th rowspan="2">Size</th>
<th colspan="5">English Benchmarks</th>
<th colspan="3">Chinese Benchmarks</th>
</tr>
<tr>
<th>GSM8K</th>
<th>MATH</th>
<th>OCW</th>
<th>SAT</th>
<th>MMLU STEM</th>
<th>CMATH</th>
<th>Gaokao MathCloze</th>
<th>Gaokao MathQA</th>
</tr>
</thead>
<tbody>
<tr>
<td>No Math Training</td>
<td>N/A</td>
<td>2.9%</td>
<td>3.0%</td>
<td>2.9%</td>
<td>15.6%</td>
<td>19.5%</td>
<td>12.3%</td>
<td>0.8%</td>
<td>17.9%</td>
</tr>
<tr>
<td>MathPile</td>
<td>8.9B</td>
<td>2.7%</td>
<td>3.3%</td>
<td>2.2%</td>
<td>12.5%</td>
<td>15.7%</td>
<td>1.2%</td>
<td>0.0%</td>
<td>2.8%</td>
</tr>
<tr>
<td>OpenWebMath</td>
<td>13.6B</td>
<td>11.5%</td>
<td>8.9%</td>
<td>3.7%</td>
<td>31.3%</td>
<td>29.6%</td>
<td>16.8%</td>
<td>0.0%</td>
<td>14.2%</td>
</tr>
<tr>
<td>Proof-Pile-2</td>
<td>51.9B</td>
<td>14.3%</td>
<td>11.2%</td>
<td>3.7%</td>
<td>43.8%</td>
<td>29.2%</td>
<td>19.9%</td>
<td>5.1%</td>
<td>11.7%</td>
</tr>
<tr>
<td><strong>DeepSeekMath Corpus</strong></td>
<td><strong>120.2B</strong></td>
<td><strong>23.8%</strong></td>
<td><strong>13.6%</strong></td>
<td><strong>4.8%</strong></td>
<td><strong>56.3%</strong></td>
<td><strong>33.1%</strong></td>
<td><strong>41.5%</strong></td>
<td><strong>5.9%</strong></td>
<td><strong>23.6%</strong></td>
</tr>
</tbody>
</table>

**分析:**
*   **高质量:** 在所有英文和中文基准上，使用 `DeepSeekMath Corpus` 训练的模型性能都遥遥领先。例如，在 `GSM8K` 上达到 23.8%，远超使用 `Proof-Pile-2` 的 14.3%。
*   **多语言性:** 在中文基准（`CMATH`, `Gaokao`）上，`DeepSeekMath Corpus` 带来了巨大提升，而其他主要为英文的语料库（如 `MathPile`）甚至导致了性能下降。这证明了其数据筛选流程成功地保留了高质量的多语言内容。
*   **大规模优势:** 下图（原文 Figure 3）显示，随着训练的进行，使用 `DeepSeekMath Corpus` 的模型性能持续稳定提升，而其他规模较小的数据集很快就达到了性能瓶颈。

    ![Figure 3 | Benchmark curves of DeepSeek-LLM 1.3B trained on different mathematical corpora.](images/3.jpg)
    *该图像是图表，展示了DeepSeek-LLM 1.3B在不同数学语料库上的基准曲线，比较了MathPile、OpenWebMath、Proof-Pile-2和DeepSeekMath Corpus在GSM8K、MATH、CMATH和BBH四个数据集上的准确率随着训练Tokens数量变化的趋势。*

### 6.1.2. 最终模型性能对比
最终的 `DeepSeekMath-RL 7B` 模型在与所有开源和闭源模型的对比中表现出色。下表是原文 Table 5 的核心结果：

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Size</th>
<th colspan="2">English Benchmarks</th>
<th colspan="2">Chinese Benchmarks</th>
</tr>
<tr>
<th>GSM8K</th>
<th>MATH</th>
<th>MGSM-zh</th>
<th>CMATH</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="6"><strong>Chain-of-Thought Reasoning</strong></td>
</tr>
<tr>
<td colspan="6"><em>Closed-Source Model</em></td>
</tr>
<tr>
<td>Gemini Ultra</td>
<td>-</td>
<td>94.4%</td>
<td>53.2%</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>GPT-4</td>
<td></td>
<td>92.0%</td>
<td>52.9%</td>
<td>-</td>
<td>86.0%</td>
</tr>
<tr>
<td>Baichuan-3</td>
<td>-</td>
<td>88.2%</td>
<td>49.2%</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td colspan="6"><em>Open-Source Model</em></td>
</tr>
<tr>
<td>InternLM2-Math</td>
<td>20B</td>
<td>82.6%</td>
<td>37.7%</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>Qwen</td>
<td>72B</td>
<td>78.9%</td>
<td>35.2%</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>DeepSeek-LLM-Chat</td>
<td>67B</td>
<td>84.1%</td>
<td>32.6%</td>
<td>74.0%</td>
<td>80.3%</td>
</tr>
<tr>
<td><strong>DeepSeekMath-Instruct</strong></td>
<td><strong>7B</strong></td>
<td><strong>82.9%</strong></td>
<td><strong>46.8%</strong></td>
<td><strong>73.2%</strong></td>
<td><strong>84.6%</strong></td>
</tr>
<tr>
<td><strong>DeepSeekMath-RL</strong></td>
<td><strong>7B</strong></td>
<td><strong>88.2%</strong></td>
<td><strong>51.7%</strong></td>
<td><strong>79.6%</strong></td>
<td><strong>88.8%</strong></td>
</tr>
<tr>
<td colspan="6"><strong>Tool-Integrated Reasoning</strong></td>
</tr>
<tr>
<td colspan="6"><em>Closed-Source Model</em></td>
</tr>
<tr>
<td>GPT-4 Code Interpreter</td>
<td></td>
<td>97.0%</td>
<td>69.7%</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td colspan="6"><em>Open-Source Model</em></td>
</tr>
<tr>
<td>InternLM2-Math</td>
<td>20B</td>
<td>80.7%</td>
<td>54.3%</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>DeepSeek-LLM-Chat</td>
<td>67B</td>
<td>86.7%</td>
<td>51.1%</td>
<td>76.4%</td>
<td>85.4%</td>
</tr>
<tr>
<td><strong>DeepSeekMath-Instruct 7B</strong></td>
<td></td>
<td><strong>83.7%</strong></td>
<td><strong>57.4%</strong></td>
<td><strong>72.0%</strong></td>
<td><strong>84.3%</strong></td>
</tr>
<tr>
<td><strong>DeepSeekMath-RL</strong></td>
<td><strong>7B</strong></td>
<td><strong>86.7%</strong></td>
<td><strong>58.8%</strong></td>
<td><strong>78.4%</strong></td>
<td><strong>87.6%</strong></td>
</tr>
</tbody>
</table>

**分析:**
*   **SOTA 性能:** 在无工具的 `CoT` 推理中，`DeepSeekMath-RL 7B` 在 `MATH` 基准上取得了 **51.7%** 的准确率，首次在开源模型中突破 50% 大关。这一成绩不仅远超所有其他开源模型（无论大小），甚至超过了 `Baichuan-3` 等强大的闭源模型，非常接近 `GPT-4` 和 `Gemini Ultra`。
*   **RL 的巨大提升:** 从 `DeepSeekMath-Instruct` 到 `DeepSeekMath-RL`，仅通过在 `GSM8K` 和 `MATH` 的部分数据上进行 GRPO 训练，模型在 `MATH` 上的准确率就从 46.8% 提升到 51.7%（绝对提升 4.9%），在 `GSM8K` 上从 82.9% 提升到 88.2%（绝对提升 5.3%）。这强有力地证明了 `GRPO` 算法的有效性。
*   **泛化能力:** 值得注意的是，RL 训练只使用了英文的 `GSM8K` 和 `MATH` 数据，但在中文基准 `MGSM-zh` 和 `CMATH` 上也获得了显著提升。这表明 RL 不仅仅是记住了特定领域的知识，而是提升了模型底层的数学推理和泛化能力。
*   **工具使用:** 在允许使用代码工具的场景下，`DeepSeekMath-RL 7B` 同样表现出色，在 `MATH` 上达到 58.8%，超越了所有其他开源模型。

## 6.2. 消融实验/参数分析
论文进行了深入的消融研究，以探究其成功的关键因素。

### 6.2.1. 代码训练的影响
以下是原文 Table 6 的结果，探究了不同训练策略对数学能力的影响：

<table>
<thead>
<tr>
<th rowspan="2">Training Setting</th>
<th colspan="2">Training Tokens</th>
<th colspan="3">w/o Tool Use</th>
<th colspan="2">w / Tool Use</th>
</tr>
<tr>
<th>General/Code</th>
<th>Math</th>
<th>GSM8K</th>
<th>MATH</th>
<th>CMATH</th>
<th>GSM8K+Python</th>
<th>MATH+Python</th>
</tr>
</thead>
<tbody>
<tr>
<td>No Continual Training</td>
<td>−</td>
<td>−</td>
<td>2.9%</td>
<td>3.0%</td>
<td>12.3%</td>
<td>2.7%</td>
<td>2.3%</td>
</tr>
<tr>
<td colspan="8"><strong>Two-Stage Training</strong></td>
</tr>
<tr>
<td>Stage 1: General Training</td>
<td>400B</td>
<td rowspan="2">150B</td>
<td rowspan="2">19.1%</td>
<td rowspan="2">14.4%</td>
<td rowspan="2">37.2%</td>
<td rowspan="2">14.3%</td>
<td rowspan="2">6.7%</td>
</tr>
<tr>
<td>Stage 2: Math Training</td>
<td>−</td>
</tr>
<tr>
<td><strong>Stage 1: Code Training</strong></td>
<td><strong>400B</strong></td>
<td rowspan="2"><strong>150B</strong></td>
<td rowspan="2"><strong>21.9%</strong></td>
<td rowspan="2"><strong>15.3%</strong></td>
<td rowspan="2"><strong>39.7%</strong></td>
<td rowspan="2"><strong>17.4%</strong></td>
<td rowspan="2"><strong>9.4%</strong></td>
</tr>
<tr>
<td><strong>Stage 2: Math Training</strong></td>
<td><strong>−</strong></td>
</tr>
<tr>
<td colspan="8"><strong>One-Stage Training</strong></td>
</tr>
<tr>
<td>Math Training</td>
<td>−</td>
<td>150B</td>
<td>20.5%</td>
<td>13.1%</td>
<td>37.6%</td>
<td>11.4%</td>
<td>6.5%</td>
</tr>
<tr>
<td>Code &amp; Math Mixed Training</td>
<td>400B</td>
<td>150B</td>
<td>17.6%</td>
<td>12.1%</td>
<td>36.3%</td>
<td>19.7%</td>
<td>13.5%</td>
</tr>
</tbody>
</table>

**分析:** <strong>“先代码，后数学”</strong>的两阶段训练策略效果最佳。与先进行通用文本训练相比，先进行代码训练的模型在第二阶段数学训练后，无论是在无工具（CoT）还是有工具（PoT）的场景下，都取得了更高的分数。这有力地支持了“代码训练可以提升数学推理能力”的假设，可能是因为代码和数学共享相似的逻辑、结构和符号化表示。

### 6.2.2. arXiv 数据的无效性
与普遍认知相反，实验发现 `arXiv` 数据在本文所用的基准上效果不佳。如原文 Table 8 所示，无论是对于 1.3B 还是 7B 模型，在 `MathPile`（主要为 arXiv）或纯 `arXiv` 数据上训练后，模型在 `GSM8K`, `MATH` 等基准上的性能几乎没有提升，甚至出现下降。作者推测这可能是因为 `arXiv` 的文本风格（高度形式化、充满术语）与评测基准中的问题风格（通常是应用题或更直白的问题）差异较大。

### 6.2.3. 强化学习机制的探索
*   **在线 vs 离线:** 如下图（原文 Figure 5）所示，`Online RFT`（在训练过程中使用实时模型采样）的性能最终显著优于 `RFT`（使用初始 SFT 模型一次性采样）。这表明随着模型能力的提升，使用其自身的探索结果进行训练至关重要。
*   **GRPO 的优势:** $GRPO+OS$ 优于 `Online RFT`，说明了 GRPO 能够根据奖励大小进行差异化更新（而非简单的接受/拒绝）的优势。而 $GRPO+PS$（过程监督）又优于 $GRPO+OS$（结果监督），证明了细粒度的步骤级奖励信号对于指导复杂推理过程更加有效。
*   **迭代 RL 的效果:** 如下图（原文 Figure 6），迭代式 RL（更新奖励模型并进行多轮训练）能带来持续的性能提升，尤其在第一轮迭代中效果显著。

    | ![Figure 5 | Performance of the DeepSeekMath-Instruct 1.3B model, which was further trained using various methods, on two benchmarks.](images/5.jpg) | ![](images/6.jpg) |

*该图像是图表，展示了DeepSeekMath-Instruct 1.3B模型通过不同训练方法在GSM8K和MATH两个基准测试上的准确率随训练步骤变化的曲线。图中对比了RFT、Online RFT、GRPO+OS和GRPO+PS四种方法的表现，GRPO+PS在两个基准上均表现最佳。*

| :---: | :---: |
| 原文 Figure 5: 不同 RL 方法对比 | 原文 Figure 6: 迭代式 RL 性能 |

*   **RL 为何有效？** 这是一个非常深刻的洞见。如下图（原文 Figure 7），作者发现 RL 训练显著提升了 `Maj@K`（多数投票）的准确率，但对 `Pass@K` 的提升很小。
    *   **解读:** 这意味着 RL 并没有教会模型从“完全不会”到“会”解决一个新问题。相反，它做的是：对于那些模型已经“有能力”解决（即在多次尝试中至少有一次能做对）的问题，RL 增强了模型生成正确答案的**概率和稳定性**。换句话说，RL 更多地是在做**对齐和分布优化**，将模型潜在的能力更可靠地激发出来，而不是在传授全新的知识。

        ![Figure 7 | The Maj $@ \\mathrm { K }$ and Pass $@ \\mathrm { K }$ of SFT and RL DeepSeekMath 7B on GSM8K and MATH (temperature 0.7). It was noted that RL enhances Maj $@ \\mathrm { K }$ but not Pass@K.](images/7.jpg)
        *该图像是图表，展示了DeepSeekMath 7B模型在GSM8K和MATH数据集上，使用SFT和RL训练方法时，Maj@K和Pass@K准确率随候选数量K变化的趋势。图中说明RL提升了Maj@K但对Pass@K作用不明显。*

---

# 7. 总结与思考

## 7.1. 结论总结
`DeepSeekMath` 是一项里程碑式的工作，它系统性地解决了提升开源模型数学推理能力的核心挑战，其主要贡献和结论可以总结如下：
1.  **数据是关键:** 通过精心设计的迭代式数据筛选流程，从公开网络数据中构建了迄今为止最大规模的数学预训练语料库 `DeepSeekMath Corpus` (120B tokens)，并证明了其卓越的质量和有效性。
2.  **模型与算法双轮驱动:** 成功训练并发布了 `DeepSeekMath` 系列模型。`DeepSeekMath-RL 7B` 在 `MATH` 基准上达到 51.7% 的准确率，成为最强的开源数学模型，性能逼近顶级闭源模型。
3.  **GRPO 算法的提出:** 引入了 `GRPO`，一种内存高效的 PPO 变体，它通过移除价值模型，使得在更低资源下进行有效的强化学习成为可能。
4.  **提供了宝贵的经验洞见:**
    *   证实了**代码预训练**对数学推理能力的积极促进作用。
    *   挑战了**`arXiv` 数据**在通用数学能力提升上的传统认知。
    *   深入剖析了强化学习在数学推理中的作用机制，即**优化输出分布而非传授新知识**。

## 7.2. 局限性与未来工作
作者在论文中也坦诚地指出了当前工作的局限性，并展望了未来的研究方向。
*   **局限性:**
    *   **领域覆盖不均:** 模型在几何和定理证明等领域的表现相对较弱，可能源于预训练和微调数据中的选择性偏差。
    *   **少样本能力不足:** 与 `GPT-4` 相比，`DeepSeekMath` 的少样本 (few-shot) 学习能力提升不明显，其零样本 (zero-shot) 和少样本性能相近。这可能与模型规模有关。
*   **未来工作:**
    *   **数据层面:** 继续改进数据筛选流程，构建更全面、更高质量的预训练语料库，特别是补充几何等薄弱领域的数据。
    *   **算法层面:** 基于本文提出的统一范式，从**数据源**（如探索更先进的采样策略）、**算法**（如设计对噪声奖励更鲁棒的算法）和**奖励模型**（如提升泛化能力、引入不确定性建模）三个方面，探索更有效的强化学习方法。

## 7.3. 个人启发与批判
这篇论文不仅提供了一个强大的模型，更重要的是，它提供了一套可复现、可扩展的系统性方法论和深刻的洞见。
*   **启发:**
    1.  <strong>“数据炼金术”</strong>的力量: 本文最大的亮点在于展示了如何从看似杂乱的公开网络数据中提炼出高价值的宝藏。这种“数据工程”的思路对于资源有限的研究者和机构极具启发意义，即强大的模型并非只能靠昂贵的私有数据堆砌。
    2.  **算法的“优雅简化”:** `GRPO` 是一个非常漂亮的创新。它没有追求更复杂的机制，而是通过一个简单的思想（组内相对奖励）解决了 PPO 的一个核心痛点（内存消耗）。这种追求效率和简约的工程美学值得学习。
    3.  **批判性思维的重要性:** 论文对 `arXiv` 数据的反思提醒我们，在 AI 研究中不应盲从过去的经验。每种数据源的价值都是相对的，需要通过严格的实验进行检验。
*   **批判与思考:**
    1.  **RL 的能力边界:** 论文关于 `Maj@K` 和 `Pass@K` 的发现引人深思。这是否暗示了当前基于“偏好对齐”的 RL 范式存在一个内在的能力天花板？如果 RL 主要作用是“扶正”模型已有的能力，那么真正实现能力“突破”的源泉是否仍必须依赖于更大规模、更高质量的预训练？
    2.  **统一范式的价值:** 作者将 SFT、RFT、DPO、PPO 等方法统一到 `(数据源, 算法, 奖励函数)` 的框架下进行分析，这是一个非常有价值的理论贡献。它帮助我们从更高维度理解了不同对齐技术之间的内在联系，为未来设计新的对齐算法提供了清晰的思路。
    3.  **通用性与专业性的权衡:** 从代码模型出发训练数学模型取得了成功，这引发了一个更广泛的问题：对于不同的专业领域（如法律、医学），最佳的“起点”是什么？是否存在一个最优的“预训练课程”顺序来构建一个全能的专家模型？这项工作为此提供了有力的案例和探索方向。