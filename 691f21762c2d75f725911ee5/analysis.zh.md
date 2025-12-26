# 1. 论文基本信息

## 1.1. 标题
Self-Chained Image-Language Model for Video Localization and Question Answering (自链式图像-语言模型用于视频定位与问答)

## 1.2. 作者
Shoubin Yu, Jaemin Cho, Prateek Yadav, Mohit Bansal
隶属于 UNC Chapel Hill (北卡罗来纳大学教堂山分校)

## 1.3. 发表信息
*   <strong>发布日期 (UTC):</strong> 2023-05-11T17:23:00.000Z
*   **发布平台:** arXiv (预印本平台)

## 1.4. 摘要
最近的研究表明，利用大规模预训练图像-语言模型 (Image-Language Models, Image-LMs) 进行视频问答 (Video Question Answering, Video QA) 已取得可喜的成果。然而，这些图像-语言模型通常将均匀采样的视频帧简单拼接作为视觉输入，缺乏显式的语言感知时序建模 (temporal modeling)。当视频输入中只有一小部分与语言查询相关时，这种均匀帧采样往往会导致遗漏重要的视觉线索。尽管人类在回答问题时通常会找到视频的某个时刻并进行回放以进行关注，但训练一个查询感知 (query-aware) 的视频时刻定位器 (moment localizer) 通常需要昂贵的标注和高昂的计算成本。

为解决这一问题，本文提出了自链式视频定位-问答框架 (Self-Chained Video Localization-Answering, SeViLA)。这是一个新颖的框架，它利用单一的图像-语言模型 (BLIP-2) 来同时处理视频中的时间关键帧定位和问答任务。SeViLA 框架包含两个模块：定位器 (Localizer) 和回答器 (Answerer)，两者都通过对 BLIP-2 进行参数高效微调 (parameter-efficiently fine-tuning) 得到。

作者提出了两种链式连接这些模块的方式，用于级联推理 (cascaded inference) 和自优化 (self-refinement)。首先，在前向链 (forward chain) 中，`Localizer` 在视频中找到多个语言感知 (language-aware) 的关键帧，然后 `Answerer` 利用这些关键帧来预测答案。其次，在反向链 (reverse chain) 中，`Answerer` 生成关键帧伪标签 (pseudo-labels) 以精炼 `Localizer`，从而减轻了对昂贵视频时刻定位标注的需求。

SeViLA 框架在五个具有挑战性的视频问答和事件预测基准测试中优于多个强大的基线方法，并在微调 (NExT-QA, STAR) 和零样本 (zero-shot) (NExT-QA, STAR, How2QA, VLEP) 设置下都达到了最先进 (state-of-the-art) 的性能。论文还分析了 `Localizer` 的影响、`Localizer` 与其他时序定位模型的比较、`Localizer` 的预训练/自优化以及关键帧数量变化的影响。

## 1.5. 论文链接
*   原文链接: [https://arxiv.org/abs/2305.06988](https://arxiv.org/abs/2305.06988)
*   PDF 链接: [https://arxiv.org/pdf/2305.06988v2.pdf](https://arxiv.org/pdf/2305.06988v2.pdf)

# 2. 整体概括

## 2.1. 研究背景与动机
当前人工智能领域在图像-语言模型 (Image-Language Models, Image-LMs) 方面取得了显著进展，但在视频-语言模型 (Video-Language Models, Video-LMs) 方面由于其更高的计算和标注成本，发展相对滞后。许多现有方法试图通过将预训练的图像-语言模型迁移到视频任务来提高效率。然而，这些方法通常采用对视频帧进行均匀或随机采样，然后简单地拼接起来作为视觉输入。这种处理方式存在以下核心问题：
1.  <strong>缺乏语言感知时序建模 (Lack of Language-aware Temporal Modeling):</strong> 视频内容的重点往往只集中在与语言查询相关的特定时刻。均匀采样的方法未能显式地建模这种语言相关的时序信息，导致模型可能关注无关紧要的帧，从而丢失关键的视觉线索。例如，一个关于“人物在做什么”的问题，可能只需要视频中某几个动作发生的瞬间，而不是整个视频的每一帧。
2.  <strong>昂贵的标注成本 (Expensive Annotation Costs):</strong> 尽管人类在理解视频时会自然地找出关键时刻，但训练一个能自动定位这些查询感知 (query-aware) 视频时刻的模型，需要大量的、细粒度的时序标注数据。这种帧级别或时刻级别的标注工作非常耗时且昂贵，极大地限制了相关模型的发展。

    因此，论文的核心动机是解决如何在不依赖昂贵标注的情况下，让强大的图像-语言模型更好地理解视频中的语言相关时序信息，并将其应用于视频问答任务。作者希望找到一种方法，既能利用现有图像-语言模型的强大能力，又能克服视频特有的时序复杂性和标注稀缺性问题。

## 2.2. 核心贡献/主要发现
本文通过提出 `SeViLA` 框架，在视频理解领域做出了以下关键贡献：

1.  **提出 `SeViLA` 框架：** 引入了一个新颖的视频-语言框架 `SeViLA`，它巧妙地利用单一的预训练图像-语言模型 `BLIP-2` 来同时执行两个核心任务：语言感知的时间关键帧定位 (language-aware temporal keyframe localization) 和视频问答 (video question answering)。
2.  **模块化设计与参数高效微调：** `SeViLA` 框架由两个模块组成：`Localizer` (定位器) 和 `Answerer` (回答器)。这两个模块都是通过对 `BLIP-2` 进行参数高效微调而获得的，这意味着它们能继承 `BLIP-2` 强大的视觉和语言理解能力，同时避免了从头训练大型视频模型的巨大成本。
3.  **创新的自链机制：**
    *   <strong>前向链 (Forward Chain)：</strong> `Localizer` 首先从视频中识别并选择与语言查询相关的多个关键帧，然后将这些关键帧传递给 `Answerer` 以预测最终答案。这种级联推理方式使模型能够聚焦于视频中最相关的信息。
    *   <strong>反向链 (Reverse Chain) 用于自优化：</strong> `Answerer` 通过生成关键帧伪标签 (pseudo-labels) 来精炼 `Localizer`。具体来说，如果 `Answerer` 能够利用某帧正确回答问题，该帧就会被标记为伪关键帧。这种自优化机制极大地减轻了对昂贵的视频时刻定位标注的需求。
4.  **卓越的实验性能：** `SeViLA` 框架在五个具有挑战性的视频问答和事件预测基准测试中（包括 `NExT-QA`, `STAR`, `How2QA`, `TVQA`, `VLEP`），显著超越了多个强大的基线方法。它在微调 (fine-tuning) 设置下（`NExT-QA`, `STAR`）和零样本 (zero-shot) 设置下（`NExT-QA`, `STAR`, `How2QA`, `VLEP`）均达到了最先进的性能。
5.  **全面的框架分析：** 论文对 `SeViLA` 框架进行了深入的消融研究 (ablation studies) 和分析，包括：`Localizer` 对性能的影响、`Localizer` 与其他时序定位模型的比较、`Localizer` 的预训练和自优化过程的有效性，以及关键帧数量对模型性能的影响。这些分析为框架的设计选择提供了有力的证据。

# 3. 预备知识与相关工作

## 3.1. 基础概念
为了更好地理解 `SeViLA` 框架，我们需要回顾一些关键的基础概念。

*   <strong>图像-语言模型 (Image-Language Models, Image-LMs):</strong> 这是一类能够同时处理图像和文本数据的模型。它们通过大规模预训练学习图像与文本之间的对应关系，从而能够执行图像描述生成、视觉问答、图像检索等跨模态任务。例如，给定一张图片，`Image-LM` 可以回答关于图片内容的问题。
*   <strong>视频-语言模型 (Video-Language Models, Video-LMs):</strong> 类似于 `Image-LM`，但专注于处理视频和文本数据。由于视频数据包含连续的时序信息，`Video-LM` 不仅需要理解单帧内容，还需要捕捉帧之间的动态变化和事件发展。然而，视频数据的获取、处理和标注成本远高于图像数据，因此 `Video-LM` 的发展通常比 `Image-LM` 慢。
*   <strong>零样本学习 (Zero-shot Learning):</strong> 是一种机器学习范式，允许模型在训练时从未见过特定类别或任务的情况下，在推理时处理这些类别或任务。在本文中，零样本 `Video QA` 意味着模型在某个数据集上训练后，可以直接在另一个未见过标注的视频问答数据集上进行推理，而无需额外的微调。
*   <strong>微调 (Fine-tuning):</strong> 指的是在一个大型预训练模型（通常在海量通用数据上训练）的基础上，使用特定任务的小规模数据集对其参数进行进一步调整的过程。这使得模型能够适应新的任务，并通常能取得比从头训练更好的性能。
*   **BLIP-2:** 是本文 `SeViLA` 框架的核心骨干。它是一个由 Salesforce Research 推出的先进预训练图像-语言模型，旨在高效地将预训练的视觉编码器和大型语言模型 (Large Language Models, LLMs) 连接起来。
    *   **架构组成:** `BLIP-2` 主要包含三个关键组件：
        1.  <strong>冻结的图像编码器 (Frozen Image Encoder):</strong> 通常是一个像 `ViT` (Vision Transformer) 这样的模型，它在大量图像数据上预训练，负责从输入图像中提取高质量的视觉特征。在 `BLIP-2` 训练时，这个组件的参数是**冻结**的，不会被更新。
        2.  <strong>冻结的大语言模型 (Frozen Large Language Model, LLM):</strong> 例如 `Flan-T5`。它在海量文本数据上预训练，拥有强大的文本理解和生成能力。在 `BLIP-2` 训练时，这个组件的参数也是**冻结**的。
        3.  **Q-former (Querying Transformer):** 这是一个可训练的 Transformer 模块，充当图像编码器和 `LLM` 之间的**桥梁**。它通过学习一组可学习的查询嵌入 (query embeddings) 来提取图像编码器输出中最具信息量的视觉特征，并将这些特征格式化为 `LLM` 可以理解的“软视觉提示 (soft visual prompts)”。
    *   **两阶段预训练:** `BLIP-2` 的 `Q-former` 经历两个阶段的预训练：
        1.  <strong>图像-文本对齐 (Image-to-text Alignment):</strong> `Q-former` 与图像编码器连接，学习从图像中提取与文本描述最相关的视觉信息。
        2.  <strong>视觉-语言生成 (Vision-to-language Generation):</strong> `Q-former` 与 `LLM` 连接，将提取的视觉特征作为 `LLM` 的输入，训练 `LLM` 生成描述图像的文本。
    *   **在 `SeViLA` 中的作用:** `BLIP-2` 的这种设计使得 `SeViLA` 可以在保持强大视觉和语言能力的同时，仅微调 `Q-former` 和一个线性层，从而实现<strong>参数高效微调 (Parameter-efficient fine-tuning)</strong>。
*   **Q-former (Querying Transformer):** 如上所述，它是 `BLIP-2` 中的一个关键组件，用于连接视觉和语言模态。它通过少量的可学习查询向量与图像特征进行交互，从而高效地从图像中抽取出 `LLM` 所需的上下文信息。其作用类似于一个<strong>适配器 (adapter)</strong>，使得冻结的 `Image-LM` 和 `LLM` 能够协同工作。
*   <strong>参数高效微调 (Parameter-efficient fine-tuning):</strong> 一种优化大型预训练模型的方法。与微调整个模型的所有参数不同，它只训练模型中的一小部分参数（例如，只训练 `Q-former` 或引入少量新的适配器层），而冻结大部分预训练参数。这样可以大幅减少训练所需的计算资源和存储空间，同时仍能有效适应新任务。
*   <strong>伪标签 (Pseudo-labeling):</strong> 是一种半监督学习技术。在没有真实标注数据的情况下，模型首先在少量有标注的数据上训练，然后用这个训练好的模型去预测大量无标注数据的标签。这些模型预测出来的标签就被称为伪标签，它们被视为真实标签，用于进一步训练模型。这有助于利用大量无标注数据来提升模型性能。
*   <strong>时序建模 (Temporal Modeling):</strong> 在处理视频等序列数据时，理解和利用数据点之间的时间依赖关系的能力。对于视频而言，时序建模意味着不仅要理解单个帧的内容，还要理解帧之间如何演变、事件如何发生、以及动作的连续性。
*   <strong>关键帧定位 (Keyframe Localization):</strong> 指的是从一个视频中识别和选择出那些最能代表视频内容、最能回答特定问题、或者包含最重要事件信息的帧。这与均匀采样所有帧不同，它旨在精炼视觉输入，聚焦于最有价值的信息。

## 3.2. 前人工作
本文的工作是建立在图像-语言模型和视频-语言模型领域的大量前人研究基础上的。

*   <strong>图像-语言预训练模型 (Image-Language Pre-trained Models):</strong>
    *   过去几年，随着大型预训练语言模型（如 `GPT-3`, `Flan-T5`）的成功，视觉与语言交叉模态模型也蓬勃发展。`Image-LM`，如 `CLIP` [55], `BLIP` [34], `Flamingo` [1], `BLIP-2` [35]，在模型规模和预训练数据规模上都取得了显著进步（如附录图 5 所示）。
    *   这些模型通常在数亿甚至数十亿图像-文本对上进行预训练，学习到强大的跨模态对齐能力。
    *   相比之下，`Video-LM` 的发展由于视频数据标注的复杂性和高昂成本（如 `InternVideo` [71], `VIOLET` [37] 等），在规模上相对较小。

*   <strong>图像到视频的迁移学习 (Image-to-Video Transfer Learning):</strong>
    *   为了弥补 `Image-LM` 与 `Video-LM` 之间的差距，许多研究致力于将 `Image-LM` 的能力迁移到视频任务中。
    *   例如，`Luo et al. [44]` 尝试将预训练的 `CLIP` 骨干网络应用于视频片段检索。
    *   `Yang et al. [85]` 扩展了冻结的双向语言模型，以结合多个图像，并通过额外的视频级预训练来适应视频任务。
    *   `Wang et al. [72]` 提出将多个图像转换为分层描述，并结合时序顺序提示，以帮助语言模型理解视频级事件。
    *   **局限性：** 多数现有 `Image-to-Video` 迁移方法仍采用**均匀采样**策略。这种策略在视频问答等任务中存在弊端，因为它缺乏语言感知 (language-aware) 的机制，可能包含大量与查询无关的冗余信息，甚至遗漏关键视觉线索。

*   <strong>语言感知的关键帧定位 (Language-aware Keyframe Localization):</strong>
    *   为了解决均匀采样的问题，一些方法开始关注如何根据语言查询来选择视频中的重要帧或时刻。
    *   `Buch et al. [3]` 提出了一个端到端 (end-to-end) 的流水线，利用答案标签来选择单个关键帧。
    *   `Lu et al. [42]` 使用独立的图像和语言模型来选择帧，然后由一个问答模型基于这些帧回答问题。
    *   `Qian et al. [54]` 设计了一个视频片段提议模型，并与问答模型迭代训练。
    *   `Kim et al. [24]` 则利用半参数检索器 (semi-parametric retriever) 基于帧与语言特征相似性来获取关键帧。
    *   **局限性：** 这些方法或需要昂贵的时序接地 (temporal grounding) 标注，或未充分利用大型 `Image-LM` 的强大通用能力。

## 3.3. 技术演进与差异化分析
现有技术在将 `Image-LM` 应用于视频任务时，主要挑战在于如何有效地处理视频的时序信息和克服标注数据稀缺的问题。

*   **技术演进：**
    *   从最初的仅处理图像-文本对，到尝试将多帧图像拼接、或者引入简单的时序编码器，再到尝试通过外部标注进行关键时刻定位。
    *   核心瓶颈始终在于如何让模型在海量视频数据中“找到”与特定语言查询相关的“那一刻”，而不是被无关信息淹没。

*   **与相关工作的差异化分析：**
    本文 `SeViLA` 框架与上述前人工作的主要区别和创新点在于：
    1.  **统一模型与参数高效：** `SeViLA` 利用**单一**的 `BLIP-2` 模型作为骨干，通过参数高效微调，使其同时具备 `Localizer` 和 `Answerer` 的功能。这与那些使用独立的图像模型、语言模型和问答模型串联的方法不同，`SeViLA` 更具统一性和效率。
    2.  <strong>自链机制 (Self-Chaining)：</strong> 核心创新在于引入了前向链和反向链的自链机制。
        *   **前向链**：实现了语言感知的关键帧选择，确保 `Answerer` 只关注与问题最相关的视觉信息，解决了均匀采样带来的冗余和信息丢失问题。
        *   **反向链**：通过 `Answerer` 生成关键帧伪标签来精炼 `Localizer`，巧妙地解决了昂贵时序定位标注的难题。这使得 `Localizer` 可以在没有直接标注的情况下，学习到语言感知的帧重要性。
    3.  **克服标注瓶颈：** 现有语言感知关键帧定位方法往往需要额外的、昂贵的帧级或时刻级标注。`SeViLA` 的反向链自优化机制有效地缓解了这一标注瓶颈，使其在实践中更具可行性。
    4.  **强大的骨干模型：** 充分利用了 `BLIP-2` 这一最先进 `Image-LM` 的强大视觉和语言理解能力，通过 `Q-former` 机制高效地将视觉信息转化为 `LLM` 可处理的格式，从而在视频任务中发挥出 `Image-LM` 的潜力。

        通过这些创新，`SeViLA` 不仅在性能上超越了许多现有方法，更提供了一种在数据稀缺背景下，将现有 `Image-LM` 强大能力高效迁移到视频理解任务的通用范式。

# 4. 方法论

## 4.1. 方法原理
`SeViLA` (Self-Chained Video Localization-Answering) 框架的核心思想是模拟人类理解视频和回答问题的方式：首先找出视频中与问题相关的关键时刻，然后基于这些关键时刻来形成答案。为了实现这一目标，`SeViLA` 将一个强大的预训练图像-语言模型 `BLIP-2` 巧妙地拆分为两个协同工作的模块：一个负责**语言感知的时间关键帧定位**的 `Localizer`，以及一个负责**问答**的 `Answerer`。

这两个模块并非独立存在，而是通过一种创新的<strong>自链 (self-chaining)</strong> 机制相互连接和优化：
1.  <strong>前向链 (Forward Chain):</strong> 在推理过程中，`Localizer` 会根据给定的语言查询（问题和选项），从视频中智能地筛选出少数几个最相关的关键帧。随后，`Answerer` 接收这些精选的关键帧作为视觉输入，结合语言查询，来生成最终的答案。这种设计确保了 `Answerer` 能够聚焦于视频中最有价值的信息，避免被大量无关帧干扰。
2.  <strong>反向链 (Reverse Chain):</strong> 为了解决昂贵关键帧标注的问题，`SeViLA` 引入了一个自优化的反向链。在这里，`Answerer` 的预测能力被用来为 `Localizer` 生成<strong>伪标签 (pseudo-labels)</strong>。如果 `Answerer` 仅凭视频中的某一帧就能正确回答问题，那么这一帧就被视为一个“伪关键帧”，用于指导 `Localizer` 的学习，使其能够更准确地识别重要的语言相关时刻。

    通过这种双向自链的机制，`SeViLA` 旨在以参数高效的方式，将 `BLIP-2` 强大的图像-语言理解能力扩展到视频领域，同时克服了视频任务中时序建模和标注数据稀缺的挑战。

## 4.2. 核心方法详解
### 4.2.1. BLIP-2 概述
`SeViLA` 框架以 `BLIP-2` 作为其基础骨干。`BLIP-2` 是一个先进的预训练图像-语言模型，其设计旨在高效地连接冻结的图像编码器和冻结的大语言模型 (LLM)。

`BLIP-2` 的主要组成部分包括：
1.  <strong>冻结的图像编码器 (Frozen Image Encoder):</strong> 通常是 `ViT` (Vision Transformer) [16] 等模型，负责从输入图像中提取视觉特征。在 `SeViLA` 中，这个组件是**冻结**的，其参数在训练过程中不会被更新。
2.  <strong>冻结的大语言模型 (Frozen Large Language Model, LLM):</strong> 例如 `Flan-T5` [7]。它拥有强大的文本理解和生成能力。同样，在 `SeViLA` 中，`LLM` 的参数也是**冻结**的。
3.  **Q-former (Querying Transformer):** 这是一个可训练的 Transformer 模块。它作为图像编码器和 `LLM` 之间的<strong>适配器 (adapter)</strong> [62, 20]。`Q-former` 接收图像编码器输出的视觉特征 $h$ 和一组可学习的查询嵌入 $q$ 作为输入，然后输出固定长度的视觉特征 $v$。这些视觉特征 $v$ 随后被用作 `LLM` 的软视觉提示 (soft visual prompts) [22]。`BLIP-2` 的 `Q-former` 经过两阶段预训练，使其能够有效地从图像中提取与文本相关的信息，并利用 `LLM` 的生成能力。

    在 `SeViLA` 框架中，`BLIP-2` 的视觉编码器和 `LLM` 保持冻结状态，只有 `Q-former` 以及后续的一个线性层在训练过程中进行更新。这种策略实现了**参数高效微调**，仅需训练 `BLIP-2` 总参数的 `2.5%` (106M 参数)。

### 4.2.2. 自链式视频定位-问答框架 (SeViLA)
`SeViLA` 框架通过赋予 `BLIP-2` 两种不同的角色——`Localizer` 和 `Answerer` 来处理视频中的时序定位和问答任务。这两个模块共享 `BLIP-2` 的冻结骨干，但拥有各自独立的 `Q-former`。

#### Localizer 模块
`Localizer` 的主要目标是根据语言查询，从视频中选择 $k$ 个语言感知 (language-aware) 的关键帧。这里的 $k$ 通常远小于视频的总帧数 $n$。

**工作流程:**
1.  **帧特征提取:** 首先，通过冻结的图像编码器 $E_v$ 从视频中均匀采样 $n$ 帧 $\{ f_1, ..., f_n \}$。对于第 $i$ 帧，提取其特征 $h_i = E_v(f_i)$。所有帧特征组成视频的特征集合 $V = \{ h_1, ..., h_n \}$。这些特征只提取一次并缓存，供 `Localizer` 和 `Answerer` 重复使用。
2.  **视觉查询特征提取:** 对于集合 $V$ 中的每个原始帧特征 $h_i$，它独立地通过 `Localizer` 专用的 `Q-former` ($Q_{loc}$) 来提取视觉查询特征 $v_i$。
3.  **得分计算:** 将视觉查询特征 $v_i$ 与语言上下文 $L$ 进行拼接。语言上下文 $L$ 由问题、选项以及一个特定的<strong>定位提示 (localization prompt)</strong> 组成。该定位提示为："Does the information within the frame provide the necessary details to accurately answer the given question?"。拼接后的输入被送入 `LLM` (`Flan-T5`)。`LLM` 输出生成单词 'yes' 的概率 $s_i$，这个概率 $s_i$ 被视为帧 $i$ 的重要性得分。
4.  **关键帧选择:** 根据这些得分 $s_i$，选择得分最高的 $k$ 帧作为语言感知的关键帧。这些关键帧的视觉查询特征构成关键帧集合 $K = \{ v_1^k, ..., v_K^k \}$。

**公式表达:**
`Localizer` 的功能可以形式化为：
$$
K = \operatorname { L o c a L I Z E R } ( V , L ) , \quad | K | = k \ll n
$$
*   $K$: 选定的 $k$ 个语言感知关键帧的集合。
*   $\operatorname { L o c a L I Z E R }$: 定位器模块。
*   $V$: 视频的帧特征集合 $\{ h_1, ..., h_n \}$，其中 $h_i = E_v(f_i)$。
*   $L$: 语言上下文，包含问题、选项和定位提示。
*   $|K|$: 关键帧集合 $K$ 的大小，即选择的关键帧数量。
*   $k$: 用户指定或预设的选择关键帧的数量。
*   $n$: 均匀采样的视频帧总数。
*   $\ll$: 表示 $k$ 远小于 $n$。

#### Answerer 模块
在 `Localizer` 提供了关键帧集合 $K$ 之后，`Answerer` 模块负责利用这些关键帧来预测视频级别的答案。

**工作流程:**
1.  **关键帧查询特征处理:** 关键帧集合 $K$ 中的每个视觉查询特征 $v_i^k$ 会通过 `Answerer` 专用的 `Q-former` ($Q_{ans}$) 进行处理。
2.  **拼接与问答:** 所有处理后的关键帧查询特征（$v_1^k, ..., v_K^k$）与语言上下文 $L$ (包含问题和选项，但通常不含 `Localizer` 使用的定位提示) 进行拼接。这个拼接后的输入被送入 `LLM` (`Flan-T5`)。
3.  **答案预测:** `LLM` 根据输入预测视频级别的答案 $a$。

**公式表达:**
`Answerer` 的功能可以形式化为：
$$
a = \operatorname { A N S W E R E R } ( K , L )
$$
*   $a$: `Answerer` 模块预测的视频级答案。
*   $\operatorname { A N S W E R E R }$: 回答器模块。
*   $K$: 从 `Localizer` 获取的语言感知关键帧集合。
*   $L$: 语言上下文，包含问题和选项。

### 4.2.3. 自链训练 (Training AnswERER and LoCALIZER via Self-Chaining)
`SeViLA` 框架通过两种自链机制来训练和优化 `Answerer` 和 `Localizer` 模块。

#### 1. 前向链：微调 Answerer
如图 3（上）所示，`Answerer` 在下游任务上进行微调。其视觉输入是**由 `Localizer` 选出的关键帧**。
*   **训练目标:** 使 `Answerer` 在给定关键帧和语言上下文的情况下，能够准确预测正确的答案。
*   **与其他设置的比较:** 论文在附录中比较了默认设置（使用 `Localizer` 选出的关键帧）与其他设置（例如，输入帧是均匀选择的）的性能。

#### 2. 反向链：精炼 LocALizer
为了解决昂贵的帧级定位标注问题，本文在反向链中采用了<strong>伪标签 (pseudo-labeling)</strong> [26] 技术来精炼 `Localizer`。如图 3（下）所示，`Answerer` 的能力被用于生成关键帧的伪标签。
*   **伪标签生成:**
    1.  <strong>冻结 <code>Answerer</code>:</strong> 首先，使用一个已经训练好的（或零样本的）`Answerer` 模块。
    2.  **帧级预测:** 对于视频中的每一帧，将其单独作为 `Answerer` 的视觉输入，并结合问答任务提示，让 `Answerer` 预测一个帧级答案。
    3.  **伪标签赋值:** 将 `Answerer` 的预测结果与真实标注 (ground-truth) 答案进行比较。如果 `Answerer` 使用该帧能够输出正确答案，那么这帧就被标记为一个**关键帧**（赋予二进制伪标签 '1'）；否则，标记为非关键帧（'0'）。
*   **`Localizer` 训练:** `Localizer` 模块随后被训练，使其能够学习识别并定位这些由 `Answerer` 生成的语言感知伪标签关键帧。这个过程显著提高了 `Localizer` 的语言感知时序定位准确性，同时无需人工进行昂贵的帧级标注。

#### 3. 预训练 LocALizer (Pre-training LocALIZER with moment retrieval label)
为了进一步增强 `Localizer` 的能力，在其进行自优化之前，还会进行一项预训练任务。
*   **任务类型:** 在视频时刻检索/定位任务（如 `QVHighlights` [30]）上进行迁移学习。
*   **数据利用:** 使用 `QVHighlights` 数据集中的视频、查询以及视频级别的时序跨度标签。
*   **标签转换:** 将视频级别的时序跨度标注转换为帧级的二元定位标签。即，如果某一帧的时间戳落在标注的时序跨度内，则将其标记为关键帧。
*   **目标:** 通过在专门的时序定位任务上进行预训练，`Localizer` 能够更好地理解语言查询与视频时刻之间的对应关系，从而为后续的自优化和问答任务奠定基础。

### 4.2.4. 框架图示
以下是原文 Figure 2 的图示，展示了 `SeViLA` 框架中 `Localizer` 和 `Answerer` 的交互，以及它们如何从 `BLIP-2` 初始化：

![Figure 2: In SEVILA framework, LocALIzer (top) selects top-K video frames, which guides AnswErer (bottom) to focus on important language-aware video moments and predict answers. Both LocALIzER and AnswERER are initialized from a single pre-trained BLIP-2 model, where only Q-formers and a linear layer $2 . 5 \\%$ of total parameters) are tuned for each module. We omit the linear layer after the Q-former for simplicity.](images/2.jpg)
*该图像是示意图，展示了SEVILA框架中的LocALIzer（顶部）和AnswErer（底部）模块的结构及功能。LocALIzer通过选择多个语言相关的关键帧，指导AnswErer聚焦于重要的视觉时刻以预测答案。两者均从单一的预训练模型BLIP-2初始化，仅微调Q-former及线性层（`2.5 ext{%}`的总参数）。*

图示中，`Localizer` (顶部) 从视频中选择 $K$ 个关键帧，然后 `Answerer` (底部) 利用这些关键帧来预测答案。两个模块都基于 `BLIP-2`，仅微调 `Q-former` 和一个线性层。

以下是原文 Figure 3 的图示，展示了 `SeViLA` 框架的自链过程：

![Figure 3: Top: In the forward chain, the LocALizer finds multiple language-aware keyframes, then the Answerer utilizes these keyframes to predict answers. We use the forward chain for both inference and AnswErer fine-tuning. Bottom: In the reverse chain, we generate keyframe pseudo-labels by using the AnSWERer to refine the LocALIZER.](images/3.jpg)
*该图像是示意图，展示了自链式视频定位与问答框架（SeViLA）的双向推理过程。上方为正向链，定位器找到多个语言感知关键帧，回答者基于这些关键帧预测答案；下方为反向链，通过回答者生成伪标签来精炼定位器。*

图示中，上方为前向链：`Localizer` 找到关键帧，`Answerer` 预测答案。下方为反向链：`Answerer` 生成关键帧伪标签，用于精炼 `Localizer`。

# 5. 实验设置

## 5.1. 数据集
实验评估了 `SeViLA` 框架在视频问答 (Video QA)、视频事件预测 (Video Event Prediction, Video EP) 和视频时刻检索 (Video Moment Retrieval) 三种视频-语言任务上的性能。

*   **NExT-QA [77]:**
    *   **类型:** 多项选择视频问答。
    *   **特点:** 关注因果和时序推理。
    *   **规模:** 包含 5440 个视频，平均长度为 44 秒，约 5.2 万个问题。
    *   **问题类型:** 分为三类：时间 (Temporal, `Tem.`)、因果 (Causal, `Cau.`) 和描述 (Descriptive, `Des.`)。
*   **STAR [75]:**
    *   **类型:** 多项选择视频问答。
    *   **特点:** 关注情境推理。
    *   **规模:** 包含 2.2 万个视频片段，平均长度为 12 秒，约 6 万个问题。
    *   **问题类型:** 分为四类：交互 (Interaction, `Int.`)、序列 (Sequence, `Seq.`)、预测 (Prediction, `Pre.`) 和可行性 (Feasibility, `Fea.`)。
*   **How2QA [36]:**
    *   **类型:** 多项选择视频问答。
    *   **特点:** 视频内容主要来自教学类视频。
    *   **规模:** 包含 4.4 万个问题，来自 9035 个视频中选取的 2.2 万个 60 秒的片段。
*   **TVQA [27]:**
    *   **类型:** 多项选择视频问答。
    *   **特点:** 视频内容来自电视节目，通常包含多角色对话和复杂剧情。
    *   **规模:** 包含 15.2 万个问题，来自 2.1 万个视频片段（平均 76 秒）。
*   **VLEP [28]:**
    *   **类型:** 视频事件预测 (Video Event Prediction)。
    *   **特点:** 模型需要根据视频前提预测两个未来事件，通常被表述为多项选择问答任务。
    *   **规模:** 包含 28,726 个未来事件预测案例，来自 10,234 个多样的电视节目和 YouTube 生活 Vlog 视频片段。
*   **QVHighlights [30]:**
    *   **类型:** 视频时刻检索 (Video Moment Retrieval)。
    *   **特点:** 模型需要根据自然语言查询预测视频中对应的时序跨度。
    *   **规模:** 包含 10,148 个视频，平均时长 150 秒，18,367 个时刻和 10,310 个查询。
    *   **在本文中的用途:** 主要用于 `Localizer` 的预训练，并且 `Localizer` 的独立性能也在此数据集上进行评估。

## 5.2. 评估指标
### 5.2.1. 视频问答与事件预测
对于视频问答 (`NExT-QA`, `STAR`, `How2QA`, `TVQA`) 和视频事件预测 (`VLEP`) 任务，采用标准的<strong>答案准确率 (Answer Accuracy)</strong> 作为评估指标。

*   <strong>概念定义 (Conceptual Definition):</strong> 答案准确率衡量的是模型在多项选择问答或事件预测任务中，其预测的答案与真实标注答案相符的比例。它直接反映了模型回答问题的正确性。
*   <strong>数学公式 (Mathematical Formula):</strong>
    $$
    Accuracy = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
    $$
*   <strong>符号解释 (Symbol Explanation):</strong>
    *   `Accuracy`: 准确率。
    *   $\text{Number of Correct Predictions}$: 模型正确预测的答案数量。
    *   $\text{Total Number of Predictions}$: 总预测数量。

### 5.2.2. 视频时刻检索 (QVHighlights)
对于视频时刻检索任务，遵循 `Lei et al. [30]` 的标准，报告 <strong>平均精度均值 (mean Average Precision, mAP)</strong> 和 **Recall@1 (R@1)**。

*   <strong>平均精度均值 (mean Average Precision, mAP):</strong>
    *   <strong>概念定义 (Conceptual Definition):</strong> `mAP` 是一种衡量信息检索系统性能的综合指标，它考虑了检索结果的查准率 (Precision) 和查全率 (Recall)。对于 `QVHighlights`，`mAP` 是在多个 IoU 阈值（从 0.5 到 0.95，步长为 0.05）下计算的平均精度。它反映了模型检索相关视频时刻的准确性和排序质量。
    *   <strong>数学公式 (Mathematical Formula):</strong>
        $$
        mAP = \frac{1}{|Q|} \sum_{q \in Q} AP(q)
        $$
        其中，对于单个查询的平均精度 `AP(q)` 计算为：
        $$
        AP(q) = \sum_{k=1}^{N_q} P_q(k) \Delta r_q(k)
        $$
    *   <strong>符号解释 (Symbol Explanation):</strong>
        *   `mAP`: 平均精度均值。
        *   $Q$: 所有查询的集合。
        *   $|Q|$: 查询集合 $Q$ 的大小。
        *   `AP(q)`: 对于查询 $q$ 的平均精度。
        *   $N_q$: 对于查询 $q$ 检索到的时刻总数。
        *   $P_q(k)$: 在检索到第 $k$ 个时刻时，对于查询 $q$ 的查准率 (Precision)。
        *   $\Delta r_q(k)$: 对于查询 $q$，当检索到第 $k$ 个时刻时，查全率 (Recall) 的变化量。如果第 $k$ 个时刻是相关的，$\Delta r_q(k)$ 通常是 $\frac{1}{\text{Total relevant items for query } q}$。
*   **Recall@1 (R@1):**
    *   <strong>概念定义 (Conceptual Definition):</strong> `Recall@1` 衡量的是对于每个查询，模型检索到的**第一个**时刻是否与真实标注的时刻高度重叠。如果预测的时刻与真实标注时刻的交并比 (Intersection over Union, IoU) 大于或等于某个预设阈值（如 0.5 或 0.7），则认为这是一个正确的预测。
    *   <strong>数学公式 (Mathematical Formula):</strong>
        $$
        R@1 = \frac{\text{Number of queries with at least one correct prediction in top-1}}{\text{Total number of queries}}
        $$
        其中，对于预测时刻 $p$ 和真实标注时刻 $g$，判断其是否正确的条件是 $IoU(p, g) \ge \tau$，而 `IoU` 定义为：
        $$
        IoU(p, g) = \frac{\text{length}(p \cap g)}{\text{length}(p \cup g)}
        $$
    *   <strong>符号解释 (Symbol Explanation):</strong>
        *   `R@1`: 召回率@1。
        *   $\tau$: IoU 阈值，通常为 0.5 或 0.7。
        *   $\text{Number of queries with at least one correct prediction in top-1}$: 在所有查询中，其排名前一的预测至少有一个是正确的查询数量。
        *   $\text{Total number of queries}$: 总查询数量。
        *   `IoU(p, g)`: 预测时刻 $p$ 与真实标注时刻 $g$ 之间的交并比。
        *   $\text{length}(p \cap g)$: 预测时刻 $p$ 与真实标注时刻 $g$ 交集的长度。
        *   $\text{length}(p \cup g)$: 预测时刻 $p$ 与真实标注时刻 $g$ 并集的长度。

## 5.3. 对比基线
论文将 `SeViLA` 框架与以下几类具有代表性的模型进行了比较：

*   <strong>最先进的视频-语言预训练模型 (State-of-the-art Video-Language Pre-trained Models):</strong>
    *   **InternVideo [71]:** 作为目前最先进的 `Video-LM` 之一，论文使用了其最大的 `MM-L-14` 变体（包含 1B 参数），该变体从 `CLIP-L/14` [55] 初始化。论文作者自行对 `InternVideo` 在下游任务上进行了微调，并遵循其默认的 8 帧设置。
*   <strong>基于 BLIP-2 的视频适配基线 (BLIP-2 based Video Adaptation Baselines):</strong>
    为了展示 `SeViLA` 中关键帧选择和时序建模的有效性，论文构建了两种基于 `BLIP-2` 的基线：
    *   **BLIP-2voting:** 这种方法通过独立处理每个均匀采样的帧，并让 `BLIP-2` 对每个帧进行问答。最终答案通过对所有帧级答案进行<strong>多数投票 (majority voting)</strong> 来获得。这种方法缺乏帧间的时序建模。
    *   **BLIP-2concat (AnswERER):** 在这种设置下，`Q-former` 处理每个均匀采样的帧，然后 `Flan-T5 LLM` 将所有帧的视觉特征拼接 (concatenate) 起来作为前缀输入，从而进行问答。这代表了一种简单的帧间信息聚合方式。
    *   这两种基线都使用了与 `SeViLA` 相同的 `BLIP-2 ViT-G Flan-T5 XL` 骨干。
*   <strong>其他关键帧选择方法 (Other Keyframe Selection Methods):</strong>
    论文还比较了 `SeViLA` 的 `Localizer` 与其他关键帧选择方法，这些方法被集成到 `SeViLA` 的 `AnswERER` 中：
    *   **零样本设置:**
        *   **CLIP [55]:** 使用预训练的 `CLIP-ViT-B/32` 计算每个帧的视觉特征与问题、选项特征之间的图像-语言相似度，选择相似度最高的 4 个关键帧。
        *   **Moment-DETR [30]:** 一个预训练在 `QVHighlights` 上的时刻检测模型。它首先根据问题和选项语句检测一个时序跨度，然后从这个跨度内均匀采样 4 帧。
    *   **微调设置:**
        *   **ATP [3]:** 一个端到端 (end-to-end) 的方法，利用答案标签来选择单个关键帧。
        *   **Differentiable Top-K [8]:** 一个可微分的 `Top-K` 选择模块，通常插入到模型中以学习选择最相关的特征。
            这些方法的 `Q-former` 大小与 `SeViLA` 的 `Localizer` 相似，并且通常被插入在 `Q-former` 之后，以学习显著的帧特征选择。

# 6. 实验结果与分析

## 6.1. 核心结果分析
本节将深入分析 `SeViLA` 框架在各种实验设置下的性能表现，并与基线模型进行对比。

### 6.1.1. 微调设置下的 SOTA 比较 (表 1)
以下是原文 Table 1 的结果：

<table>
<thead>
<tr>
<td rowspan="3">Model (# Frames)</td>
<td colspan="4">NExT-QA</td>
<td colspan="4">STAR</td>
<td rowspan="3">How2QA</td>
<td rowspan="3">TVQA</td>
<td rowspan="3">VLEP</td>
</tr>
<tr>
<td colspan="4"></td>
<td colspan="4"></td>
</tr>
<tr>
<td>Tem.</td>
<td>Cau.</td>
<td>Des.</td>
<td>Avg.</td>
<td>Int.</td>
<td>Seq.</td>
<td>Pre.</td>
<td>Fea.</td>
<td>Avg.</td>
</tr>
</thead>
<tbody>
<tr>
<td colspan="12">(w/ speech input or use dense frames)</td>
</tr>
<tr>
<td>HERO (dense/1fps) [36]</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>-</td>
<td>73.8</td>
<td>73.6</td>
<td>-</td>
</tr>
<tr>
<td>JustAsk (20) [84]</td>
<td>51.4</td>
<td>49.6</td>
<td>63.1</td>
<td>52.3</td>
<td></td>
<td></td>
<td>-</td>
<td></td>
<td>84.4</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>FrozenBiLM (10) [85]</td>
<td>-</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>-</td>
<td>-</td>
<td>86.7</td>
<td>82.0</td>
<td>-</td>
</tr>
<tr>
<td>VidIL 4-shot (12) [72]</td>
<td>-</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>72.0</td>
</tr>
<tr>
<td>T+T (dense/1fps) [40]</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>-</td>
<td>-</td>
<td>92.4</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>T+T (+ASR, dense/1fps) [40]</td>
<td></td>
<td></td>
<td></td>
<td>-</td>
<td></td>
<td></td>
<td>-</td>
<td>-</td>
<td>93.2</td>
<td></td>
<td>-</td>
</tr>
<tr>
<td>Flamingo-80B 32-shot (30) [1]</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>FrozenBiLM (10) [85]</td>
<td>-</td>
<td></td>
<td>-</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>42.2</td>
<td>-</td>
<td>- 81.5</td>
<td>57.5</td>
<td></td>
</tr>
<tr>
<td>All-in-One (32) [67]</td>
<td>48.6</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>48.0</td>
<td>63.2</td>
<td>50.6</td>
<td>47.5</td>
<td>50.8</td>
<td>47.7</td>
<td>44.0</td>
<td>47.5</td>
<td>-</td>
<td>-</td>
<td></td>
</tr>
<tr>
<td>Temp[ATP] (32) [3]</td>
<td>49.3</td>
<td>48.6</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>65.0</td>
<td>51.5</td>
<td>50.6</td>
<td>52.8</td>
<td>49.3</td>
<td>40.6</td>
<td>48.3</td>
<td>-</td>
<td>-</td>
<td></td>
</tr>
<tr>
<td>VGT (32) [78]</td>
<td>55.0</td>
<td>52.2</td>
<td>64.0</td>
<td>55.0</td>
<td></td>
<td></td>
<td></td>
<td>44.2</td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>MIST (32) [18]</td>
<td>56.6</td>
<td>54.6</td>
<td>66.9</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>57.1</td>
<td>55.5</td>
<td>54.2</td>
<td>54.2</td>
<td>44.4</td>
<td>51.1</td>
<td>-</td>
<td></td>
<td></td>
</tr>
<tr>
<td>VFC (32) [50]</td>
<td>53.3</td>
<td>57.6</td>
<td>72.8</td>
<td>58.6</td>
<td>-</td>
<td>-</td>
<td></td>
<td>-</td>
<td>-</td>
<td></td>
<td></td>
</tr>
<tr>
<td>CoVGT (32) [79]</td>
<td>57.4</td>
<td>58.8</td>
<td>69.3</td>
<td>60.0</td>
<td></td>
<td>-</td>
<td></td>
<td>45.9</td>
<td>-</td>
<td></td>
<td></td>
</tr>
<tr>
<td>SeViTFiD (10) [24]</td>
<td>-</td>
<td></td>
<td></td>
<td>60.6</td>
<td></td>
<td></td>
<td></td>
<td>-</td>
<td></td>
<td>-</td>
<td></td>
</tr>
<tr>
<td>HiTeA (16) [87]</td>
<td>58.3</td>
<td>62.4</td>
<td>75.6</td>
<td>63.1</td>
<td>-</td>
<td>-</td>
<td></td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>InternVideo* (8) [71]</td>
<td>58.5</td>
<td>62.5</td>
<td>75.8</td>
<td>63.2</td>
<td></td>
<td></td>
<td></td>
<td>62.7</td>
<td>65.6</td>
<td>54.9</td>
<td>51.9</td>
<td>58.7</td>
<td>79.0</td>
<td>57.2</td>
<td>63.9</td>
</tr>
<tr>
<td>BLIP-2voting (4)</td>
<td>65.2</td>
<td>70.1</td>
<td>80.1</td>
<td>70.1</td>
<td></td>
<td></td>
<td></td>
<td>52.3</td>
<td>54.8</td>
<td>49.0</td>
<td>51.2</td>
<td>51.8</td>
<td>79.6</td>
<td>54.5</td>
<td>67.0</td>
</tr>
<tr>
<td>BLIP-2concat (ANSWERER) (4)</td>
<td>68.1</td>
<td>72.9</td>
<td>81.2</td>
<td>72.6</td>
<td></td>
<td></td>
<td></td>
<td>65.4</td>
<td>69.0</td>
<td>59.7</td>
<td>54.2</td>
<td>62.0</td>
<td>82.2</td>
<td>59.8</td>
<td>68.6</td>
</tr>
<tr>
<td>SEVILA† (32 → 4)</td>
<td>68.8</td>
<td>73.4</td>
<td>83.5</td>
<td>73.4</td>
<td></td>
<td>63.2</td>
<td>66.6</td>
<td>61.3</td>
<td>60.0</td>
<td>62.7</td>
<td></td>
<td></td>
<td>83.7</td>
<td>59.7</td>
<td>69.0</td>
</tr>
<tr>
<td>SeViLA (32 → 4)</td>
<td><b>69.4</b></td>
<td><b>74.2</b></td>
<td><u>81.3</u></td>
<td><b>73.8</b></td>
<td></td>
<td><b>63.7</b></td>
<td><b>70.4</b></td>
<td><b>63.1</b></td>
<td><b>62.4</b></td>
<td><b>64.9</b></td>
<td></td>
<td></td>
<td><b>83.6</b></td>
<td><b>61.6</b></td>
<td><b>68.9</b></td>
</tr>
</tbody>
</table>

**关键发现:**
*   **时序建模的重要性:** `BLIP-2voting`（独立处理帧，缺乏时序建模）在 `STAR`, `How2QA`, `TVQA`, `VLEP` 上的性能显著低于 `BLIP-2concat (AnswERER)`。特别是在需要强时序理解的 `STAR-Sequence` 任务上，`BLIP-2concat (AnswERER)` 比 `BLIP-2voting` 高出 `13.1%`（69.0% vs. 54.8%），这表明时序建模对于视频-语言任务至关重要。
*   **关键帧选择的优势:**
    *   `SEVILA†`（使用零样本 `Localizer`）在所有任务上都超越了最先进的 `Video-LM` (`InternVideo`)，平均优势为 `5.3%`。
    *   `SEVILA†` 也优于采用均匀帧采样的 `BLIP-2concat (AnswERER)`，在 `NExT-QA (+1.2%)`, `STAR (+0.7%)`, $How2QA (+1.5%)$, `VLEP (+0.4%)` 上均有提升。这突显了关键帧选择在视频-语言任务中的重要性。
*   **自优化的效果显著:** `SEVILA`（经过伪标签自优化）在 `NExT-QA (+0.4%)`, `STAR (+2.2%)`, `TVQA (+1.9%)` 上进一步提升了性能。
*   **SOTA 性能:** `SeViLA` 框架在 `NExT-QA`, `STAR`, `TVQA`, `VLEP` 上取得了新的最先进微调性能，并且仅使用视觉和语言模态。

### 6.1.2. 零样本设置下的 SOTA 比较 (表 2)
以下是原文 Table 2 的结果：

<table>
<thead>
<tr>
<td rowspan="3">Model (# Frames)</td>
<td colspan="4">NExT-QA</td>
<td colspan="5">STAR</td>
<td rowspan="3">How2QA</td>
<td rowspan="3">TVQA</td>
<td rowspan="3">VLEP</td>
</tr>
<tr>
<td colspan="4"></td>
<td colspan="5"></td>
</tr>
<tr>
<td>Tem.</td>
<td>Cau.</td>
<td>Des.</td>
<td>Avg.</td>
<td>Int.</td>
<td>Seq.</td>
<td>Pre.</td>
<td>Fea.</td>
<td>Avg.</td>
</tr>
</thead>
<tbody>
<tr>
<td colspan="13">(w/ speech input or use dense frames)</td>
</tr>
<tr>
<td>JustAsk (20) [84]</td>
<td>-</td>
<td>-</td>
<td></td>
<td></td>
<td></td>
<td>-</td>
<td></td>
<td></td>
<td>-</td>
<td>51.1</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>FrozenBiLM (10) [85]</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>-</td>
<td>58.4</td>
<td>59.2</td>
<td>-</td>
</tr>
<tr>
<td>ViperGPT (dense/1fps) [63]</td>
<td></td>
<td>-</td>
<td>-</td>
<td>60.0</td>
<td>-</td>
<td></td>
<td></td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>Flamingo-80B (30) [1]</td>
<td></td>
<td></td>
<td>-</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>FrozenBiLM (10) [85]</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td></td>
<td></td>
<td></td>
<td>-</td>
<td>39.7</td>
<td>- 41.9</td>
<td>- 29.7</td>
<td>-</td>
</tr>
<tr>
<td>VFC (32) [50]</td>
<td>- 45.4</td>
<td></td>
<td>51.6</td>
<td>64.1</td>
<td>51.5</td>
<td></td>
<td></td>
<td>-</td>
<td>-</td>
<td></td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>InternVideo* (8) [71]</td>
<td>43.4</td>
<td>48.0</td>
<td>65.1</td>
<td>49.1</td>
<td></td>
<td></td>
<td>43.8</td>
<td>43.2</td>
<td>42.3</td>
<td>37.4</td>
<td>41.6</td>
<td></td>
<td></td>
<td>62.2</td>
<td>35.9</td>
<td>58.7</td>
</tr>
<tr>
<td>BLIP-2voting (4)</td>
<td>59.1</td>
<td>61.3</td>
<td>74.9</td>
<td>62.7</td>
<td>41.8</td>
<td></td>
<td>39.7</td>
<td>40.2</td>
<td>39.5</td>
<td>40.3</td>
<td></td>
<td></td>
<td>69.8</td>
<td>35.7</td>
<td>63.8</td>
</tr>
<tr>
<td>BLIP-2concat (AnswereR) (4)</td>
<td>59.7</td>
<td>60.8</td>
<td>73.8</td>
<td>62.4</td>
<td></td>
<td></td>
<td>45.5</td>
<td>41.8</td>
<td>41.8</td>
<td>40.0</td>
<td>42.2</td>
<td></td>
<td></td>
<td>70.8</td>
<td>36.6</td>
<td>64.0</td>
</tr>
<tr>
<td>SEVILA† (32 → 4)</td>
<td><b>61.3</b></td>
<td></td>
<td><b>61.5</b></td>
<td><b>75.6</b></td>
<td><b>63.6</b></td>
<td><b>48.3</b></td>
<td><b>45.0</b></td>
<td><b>44.4</b></td>
<td><b>40.8</b></td>
<td><b>44.6</b></td>
<td></td>
<td></td>
<td><b>72.3</b></td>
<td><b>38.2</b></td>
<td><b>64.4</b></td>
</tr>
</tbody>
</table>

**关键发现:**
*   <strong>Image-LM 优于 Video-LM (无视频预训练):</strong> `BLIP-2voting`（不包含帧间时序建模）在零样本设置下，竟然超越了最先进的 `Video-LM` (`InternVideo`)。例如，在 `NExT-QA (+13.6%)`, $How2QA (+7.6%)$, `VLEP (+5.1%)` 上均有显著优势。在 `How2QA` 上，`BLIP-2voting` 甚至超过了 `FrozenBiLM`（该模型进行了额外的语音和视频预训练）`11.4%`。这表明 `Image-LM` 在模型规模和预训练数据方面的优势，使其在视频-语言任务中拥有巨大潜力。
*   **关键帧选择比均匀采样更有效:** `SEVILA†`（结合零样本 `Localizer` 和零样本 `AnswERER`）在 `NExT-QA (+1.2%)`, `STAR (+2.4%)`, $How2QA (+1.5%)$, `TVQA (+1.6%)`, `VLEP (+0.4%)` 上均优于采用均匀采样帧的 `BLIP-2concat (AnswERER)`。
*   **新的零样本 SOTA:** `SEVILA†` 在 `NExT-QA`, `STAR`, `How2QA`, `VLEP` 上达到了新的最先进零样本性能，并在 `TVQA` 上也取得了仅使用视觉和语言模态下的最先进性能。在 `STAR` 上，`SEVILA†` 甚至比零样本 `Flamingo-80B` (80B 参数) 高出 `4.9%`。这些结果再次证明了 `SeViLA` 框架在适应视频-语言任务方面的有效性，以及语言感知关键帧选择的重要性。

### 6.1.3. `SEVILA` 框架的消融研究 (表 3)
以下是原文 Table 3 的结果：

<table>
<thead>
<tr>
<td rowspan="3"></td>
<td colspan="2">AnSwERER</td>
<td rowspan="3">Keyframe</td>
<td colspan="4">NExT-QA</td>
<td colspan="5">STAR</td>
<td rowspan="3">How2QA</td>
<td rowspan="3">TVQA</td>
<td rowspan="3">VLEP</td>
</tr>
<tr>
<td rowspan="2"># frame</td>
<td rowspan="2">finetuned?</td>
<td colspan="4"></td>
<td colspan="5"></td>
</tr>
<tr>
<td>Tem.</td>
<td>Cau.</td>
<td>Des.</td>
<td>Avg.</td>
<td>Int.</td>
<td>Seq.</td>
<td>Pre.</td>
<td>Fea.</td>
<td>Avg.</td>
</tr>
</thead>
<tbody>
<tr>
<td>A.</td>
<td>32</td>
<td>×</td>
<td>uniform</td>
<td>54.7</td>
<td>56.7</td>
<td>67.8</td>
<td>57.7</td>
<td>46.2</td>
<td>43.6</td>
<td>40.7</td>
<td>41.0</td>
<td>42.8</td>
<td>67.0</td>
<td>33.2</td>
<td>54.0</td>
</tr>
<tr>
<td>B.</td>
<td>4</td>
<td>×</td>
<td>uniform</td>
<td>59.7</td>
<td>60.8</td>
<td>73.8</td>
<td>62.4</td>
<td>45.5</td>
<td>41.8</td>
<td>41.8</td>
<td>40.0</td>
<td>42.2</td>
<td>70.8</td>
<td>36.6</td>
<td>64.0</td>
</tr>
<tr>
<td>C.</td>
<td>4</td>
<td>×</td>
<td>LocaLizeR†</td>
<td>61.3</td>
<td>61.5</td>
<td>75.6</td>
<td>63.6</td>
<td>48.3</td>
<td>45.0</td>
<td>44.4</td>
<td>40.8</td>
<td>44.6</td>
<td>72.3</td>
<td>38.2</td>
<td>64.4</td>
</tr>
<tr>
<td>D.</td>
<td>4</td>
<td>×</td>
<td>LocaLizer</td>
<td>62.3</td>
<td>63.1</td>
<td>74.9</td>
<td>64.6</td>
<td>49.0</td>
<td>46.4</td>
<td>45.2</td>
<td>41.6</td>
<td>45.5</td>
<td>72.9</td>
<td>39.1</td>
<td>64.6</td>
</tr>
<tr>
<td>E.</td>
<td>4</td>
<td>√</td>
<td>uniform</td>
<td>68.1</td>
<td>72.9</td>
<td>81.2</td>
<td>72.6</td>
<td>65.4</td>
<td>69.0</td>
<td>59.7</td>
<td>54.2</td>
<td>62.0</td>
<td>82.2</td>
<td>59.8</td>
<td>68.6</td>
</tr>
<tr>
<td>F.</td>
<td>4</td>
<td>√</td>
<td>LOCAlizeR†</td>
<td>68.8</td>
<td>73.4</td>
<td>83.5</td>
<td>73.4</td>
<td>63.2</td>
<td>66.6</td>
<td>61.3</td>
<td>60.0</td>
<td>62.7</td>
<td>83.7</td>
<td>59.7</td>
<td>69.0</td>
</tr>
<tr>
<td>G.</td>
<td>4</td>
<td>√</td>
<td>LOcaLIzER</td>
<td><b>69.4</b></td>
<td><b>74.2</b></td>
<td>81.3</td>
<td><b>73.8</b></td>
<td><b>63.7</b></td>
<td><b>70.4</b></td>
<td><b>63.1</b></td>
<td><b>62.4</b></td>
<td><b>64.9</b></td>
<td><b>83.6</b></td>
<td><b>61.6</b></td>
<td><b>68.9</b></td>
</tr>
</tbody>
</table>

**关键发现:**
*   <strong>稀疏帧优于密集帧 (A vs. B):</strong> 当 `AnswERER` 的输入帧从 32 帧（A）减少到 4 帧（B）时，即使都是均匀采样，性能也有所提升（例如，NExT-QA Avg. 从 57.7% 提升到 62.4%）。这验证了作者的观点：原生的 `Image-LM` 时序建模能力有限，过多的密集帧反而可能干扰模型。
*   <strong>关键帧优于均匀采样帧 (B vs. C / E vs. F):</strong>
    *   在零样本 `AnswERER` 设置下（B vs. C），使用零样本 `LocALIzeR†` 提供的关键帧比均匀采样帧带来了显著性能提升，`NExT-QA (+1.0%)`, `STAR (+2.4%)`, $How2QA (+1.5%)$, `TVQA (+1.6%)`, `VLEP (+0.4%)`。
    *   在微调 `AnswERER` 设置下（E vs. F），`LocALIzeR†` 同样带来了提升，平均提升 `0.7%`。这再次强调了关键帧选择的重要性。
*   <strong>伪标签精炼的有效性 (C vs. D / F vs. G):</strong>
    *   在零样本 `AnswERER` 设置下（C vs. D），`LocALizer` 经过伪标签精炼后，平均性能提升 `2.1%`。
    *   在微调 `AnswERER` 设置下（F vs. G），伪标签精炼也带来了平均 `1.5%` 的提升。这证明了反向链中自优化机制的强大效果。
        这些消融实验有力地证明了 `SeViLA` 框架中 `Localizer` 的有效性，以及自优化机制在零样本和微调设置下都对视频-语言任务带来了非平凡的提升。

### 6.1.4. 视频时刻检索 SOTA 比较 (表 4)
以下是原文 Table 4 的结果：

<table>
<thead>
<tr>
<td>Model</td>
<td>R1@0.5</td>
<td>R1@0.7</td>
<td>mAP</td>
</tr>
</thead>
<tbody>
<tr>
<td>CAL [13]</td>
<td>25.4</td>
<td>11.5</td>
<td>9.8</td>
</tr>
<tr>
<td>XML [29]</td>
<td>41.8</td>
<td>30.3</td>
<td>32.1</td>
</tr>
<tr>
<td>Moment-DETR [30]</td>
<td>52.8</td>
<td>33.0</td>
<td>30.7</td>
</tr>
<tr>
<td>QD-DETR [51]</td>
<td><b>62.4</b></td>
<td><b>44.9</b></td>
<td><b>39.8</b></td>
</tr>
<tr>
<td>LocALizeR (Ours)</td>
<td>54.5</td>
<td>36.5</td>
<td>32.3</td>
</tr>
</tbody>
</table>

**关键发现:**
*   **`Localizer` 作为独立模型表现出色:** `SeViLA` 的 `Localizer` 模块（仅进行帧级操作，没有显式时序建模/训练）在 `QVHighlights` 数据集上取得了令人印象深刻的性能。
*   **超越传统时序模型:** `Localizer` 优于许多具有复杂时序建模和视频数据训练的早期方法，如 `CAL` [13]、`XML` [29]、`Moment-DETR` [30]。这表明，即使缺乏复杂的时序设计，大型 `Image-LM` 的强大能力在帧级定位任务中也具有竞争力。
*   **潜在方向:** 这结果暗示了将大型 `Image-LM` 与更先进的时序设计结合，可能会在视频时刻检索任务中取得更好的效果。

### 6.1.5. `Localizer` 预训练和自优化的影响 (表 5)
以下是原文 Table 5 的结果：

<table>
<thead>
<tr>
<td rowspan="2">PT</td>
<td rowspan="2">SR</td>
<td colspan="4">NExT-QA</td>
<td rowspan="2">How2QA</td>
</tr>
<tr>
<td>Tem.</td>
<td>Cau.</td>
<td>Des.</td>
<td>Avg.</td>
</tr>
</thead>
<tbody>
<tr>
<td>-</td>
<td>-</td>
<td>60.4</td>
<td>61.0</td>
<td>74.6</td>
<td>62.9</td>
<td>70.7</td>
</tr>
<tr>
<td>✓</td>
<td>-</td>
<td>61.3</td>
<td>61.5</td>
<td>75.6</td>
<td>63.6</td>
<td>72.3</td>
</tr>
<tr>
<td>-</td>
<td>✓</td>
<td>62.1</td>
<td>62.6</td>
<td>75.1</td>
<td>64.3</td>
<td>72.8</td>
</tr>
<tr>
<td>✓</td>
<td>✓</td>
<td><b>62.3</b></td>
<td><b>63.1</b></td>
<td><b>74.9</b></td>
<td><b>64.6</b></td>
<td><b>72.9</b></td>
</tr>
</tbody>
</table>

**关键发现:**
*   **预训练和自优化的独立贡献:** 未经训练的 `BLIP-2 Localizer` 仅带来了微小的性能提升。然而，`QVHighlights` 预训练和反向链自优化 (Self-Refinement, SR) 均能独立地为 `AnswERER` 带来显著性能提升。
*   **结合效果最佳:** 当同时应用预训练和自优化时，`Localizer` 达到最优性能，进一步证明了该方法在实现关键帧时序定位方面的标签高效性 (label-efficient)。

### 6.1.6. `Localizer` 与其他关键帧选择方法的比较 (表 6)
以下是原文 Table 6 的结果：

<table>
<thead>
<tr>
<td>Method</td>
<td colspan="4">NExT-QA</td>
</tr>
<tr>
<td></td>
<td>Tem.</td>
<td>Cau.</td>
<td>Des.</td>
<td>Avg.</td>
</tr>
</thead>
<tbody>
<tr>
<td>AnswERER</td>
<td>59.7</td>
<td>60.8</td>
<td>73.7</td>
<td>62.4</td>
</tr>
<tr>
<td>(zero-shot)</td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>+ CLIP [55]</td>
<td>59.2</td>
<td>59.5</td>
<td>72.1</td>
<td>60.6</td>
</tr>
<tr>
<td>+ Moment-DETR [30]</td>
<td>60.0</td>
<td>60.6</td>
<td>72.5</td>
<td>61.8</td>
</tr>
<tr>
<td>+ Localizer†</td>
<td><b>61.3</b></td>
<td><b>61.5</b></td>
<td><b>75.6</b></td>
<td><b>63.6</b></td>
</tr>
<tr>
<td>(fine-tuning)</td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>+ ATP [3]</td>
<td>60.4</td>
<td>61.3</td>
<td>73.4</td>
<td>62.8</td>
</tr>
<tr>
<td>+ Differentiable Top-K [8]</td>
<td>59.5</td>
<td>59.7</td>
<td>72.7</td>
<td>61.6</td>
</tr>
<tr>
<td>+ LocaliZeR</td>
<td><b>62.3</b></td>
<td><b>63.1</b></td>
<td><b>74.9</b></td>
<td><b>64.6</b></td>
</tr>
</tbody>
</table>

**关键发现:**
*   **零样本 `Localizer` 优于其他零样本方法:** `CLIP` 和 `Moment-DETR` 提取的关键帧并未帮助 `AnswERER` 提升性能，甚至可能因为其预训练目标与问答任务不完全一致，引入无关特征而产生干扰。相比之下，`SeViLA` 的零样本 `Localizer†` 平均提升 `1.2%`。
*   **精炼后的 `Localizer` 优于微调方法:** 经过伪标签精炼的 `SeViLA` `Localizer`，平均提升 `2.2%`，优于微调的 `ATP` 和 `Differentiable Top-K` 等方法。这表明 `SeViLA` 的 `Localizer` 在两种设置下都具有优越性。

### 6.1.7. 关键帧选择范围和数量的影响 (表 7)
以下是原文 Table 7 的结果：

<table>
<thead>
<tr>
<td rowspan="2">Settings</td>
<td colspan="4">NExT-QA</td>
<td rowspan="2">How2QA</td>
</tr>
<tr>
<td>Tem.</td>
<td>Cau.</td>
<td>Des.</td>
<td>Avg.</td>
</tr>
</thead>
<tbody>
<tr>
<td>BLIP-2voting (8)</td>
<td>59.9</td>
<td>60.2</td>
<td>72.4</td>
<td>62.0</td>
<td>69.8</td>
</tr>
<tr>
<td>8→1</td>
<td>59.8</td>
<td>61.1</td>
<td>76.0</td>
<td>62.9</td>
<td>72.4</td>
</tr>
<tr>
<td>16→1</td>
<td>59.2</td>
<td>62.6</td>
<td>74.9</td>
<td>63.4</td>
<td>73.2</td>
</tr>
<tr>
<td>16→4</td>
<td>60.7</td>
<td>61.5</td>
<td>75.8</td>
<td>63.4</td>
<td>72.4</td>
</tr>
<tr>
<td>32→4</td>
<td><b>61.3</b></td>
<td><b>61.5</b></td>
<td><b>75.6</b></td>
<td><b>63.6</b></td>
<td><b>72.3</b></td>
</tr>
<tr>
<td>32→8</td>
<td>59.4</td>
<td>60.9</td>
<td>74.7</td>
<td>62.5</td>
<td>71.3</td>
</tr>
<tr>
<td>64→8</td>
<td>58.9</td>
<td>60.9</td>
<td>74.0</td>
<td>62.2</td>
<td>71.8</td>
</tr>
</tbody>
</table>

**关键发现:**
*   **单个关键帧的有效性:** 即使只选择一个关键帧，`Localizer` 也能带来显著提升（例如，`8→1` 比 `BLIP-2voting (8)` 在 `NExT-QA-Causal (+0.9%)`, `NExT-QA-Description (+3.6%)`, $How2QA (+2.6%)$ 上表现更好）。这表明 `Localizer` 在定位选择性关键帧方面的有效性。
*   **多关键帧对时序问题有益:** 多关键帧（如 `16→4` 或 `32→4`）对 `NExT-QA-Temporal` 类问题更有益。
*   **密集帧的负面影响:** 随着输入帧数量的增加（如 `32→8` 和 `64→8`），性能反而下降。这与之前在消融研究中的发现一致：过多的密集帧会干扰 `Image-LM` 的处理。

### 6.1.8. `Answerer` 微调期间不同帧采样策略的影响 (表 8)
以下是原文 Table 8 的结果：

<table>
<thead>
<tr>
<td colspan="2">Frame Sampling</td>
<td colspan="4">NExT-QA</td>
</tr>
<tr>
<td>Training</td>
<td>Inference</td>
<td>Temp.</td>
<td>Cau.</td>
<td>Des.</td>
<td>Avg.</td>
</tr>
</thead>
<tbody>
<tr>
<td>Random</td>
<td>Uniform</td>
<td>68.1</td>
<td>72.9</td>
<td>81.2</td>
<td>72.6</td>
</tr>
<tr>
<td>Random</td>
<td>LoCAlizeR†</td>
<td>67.6</td>
<td>73.4</td>
<td>84.0</td>
<td>73.1</td>
</tr>
<tr>
<td>LOCALiZeR</td>
<td>Uniform</td>
<td>68.2</td>
<td>72.7</td>
<td>80.0</td>
<td>72.3</td>
</tr>
<tr>
<td>LocalizeR†</td>
<td>LOCalizeR</td>
<td><b>68.8</b></td>
<td><b>73.4</b></td>
<td><b>83.5</b></td>
<td><b>73.4</b></td>
</tr>
</tbody>
</table>

**关键发现:**
*   `SeViLA` 框架在训练和推理时都使用 `Localizer` 选出的关键帧时，性能最优。这可能是因为提供了更具信息量的关键帧，并且训练和评估之间的域偏移 (domain shifts) 更小。

### 6.1.9. Oracle 关键帧的性能上限分析 (表 9)
以下是原文 Table 9 的结果：

<table>
<thead>
<tr>
<td rowspan="2">Datasets</td>
<td colspan="2">BLIP-2voting (Oracle)</td>
</tr>
<tr>
<td>Zero-Shot</td>
<td>Fine-tuned</td>
</tr>
</thead>
<tbody>
<tr>
<td>NExT-QA (Avg.)</td>
<td>62.7 (70.1)</td>
<td>70.1 (79.7)</td>
</tr>
<tr>
<td>STAR (Avg.)</td>
<td>40.3 (52.9)</td>
<td>51.8 (72.2)</td>
</tr>
<tr>
<td>How2QA</td>
<td>69.8 (77.8)</td>
<td>79.6 (86.4)</td>
</tr>
<tr>
<td>TVQA</td>
<td>35.7 (45.4)</td>
<td>54.5 (69.0)</td>
</tr>
<tr>
<td>VLEP</td>
<td>63.8 (70.5)</td>
<td>67.0 (79.1)</td>
</tr>
</tbody>
</table>

**关键发现:**
*   论文通过假设存在一个“完美”的定位器（`Oracle`），能够始终为 `Answerer` 提供正确关键帧，来探索性能上限。
*   结果显示，`BLIP-2voting` 的性能与 `Oracle` 性能之间存在显著差距。例如，在零样本 `NExT-QA` 上，`BLIP-2voting` 为 62.7%，而 `Oracle` 为 70.1%。
*   这些差距表明，在时序定位方面仍有很大的改进空间，以便更有效地利用 `Image-LM` 处理视频-语言任务。

### 6.1.10. `Localizer` 的定性分析 (图 4, 7)
以下是原文 Figure 4 的图示：

![Figure 4: Visualization of our LocALIzER. We use zero-shot AnswERER with different frame sampling (uniform v.s. LocALizeR) to answer the question. Red options are answered wrongly with uniformly sampled frames. Green options are answered correctly with our LocALizeR. Best viewed in color.](images/4.jpg)
*该图像是示意图，展示了在视频问答任务中利用不同帧采样（均匀采样 vs. 我们的定位器）进行回答的效果。红色选项表示使用均匀采样错误回答，绿色选项表示使用我们的定位器正确回答。最佳查看效果为彩色。*

以下是原文 Figure 7 的图示：

![Figure 7: Visualization of our LocALizER. We show various keyframe amounts in those examples. We use zero-shot AnswEReR with different frame sampling (uniform v.s. LocALIzeR) to answer the question. Red options are answered wrongly with uniformly sampled frames. Green options are answered correctly with our LocALizER. Best viewed in color.](images/7.jpg)

**关键发现:**
*   通过可视化案例（图 4, 7），`SeViLA` 的 `Localizer` 能够比均匀采样更准确地定位与问题相关的关键帧，这些关键帧与人类标注高度一致。
*   这种准确的定位使得 `Answerer` 能够正确回答问题，而均匀采样则可能导致错误答案。这证实了 `Localizer` 能够有效定位任务相关的关键帧，从而提升下游任务的性能。

### 6.1.11. `Localizer` 的单帧与多帧比较 (表 12)
以下是原文 Table 12 的结果：

<table>
<thead>
<tr>
<td>AnSWERER</td>
<td># frames of LoCALIZER</td>
<td>NExT-QA (Average)</td>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">zero-shot</td>
<td>1</td>
<td><b>64.6</b></td>
</tr>
<tr>
<td>4</td>
<td>63.6</td>
</tr>
<tr>
<td rowspan="2">fine-tuned</td>
<td>1</td>
<td><b>73.4</b></td>
</tr>
<tr>
<td>4</td>
<td>71.3</td>
</tr>
</tbody>
</table>

**关键发现:**
*   **单帧 `Localizer` 表现更好:** 在零样本和微调设置下，单帧 `Localizer` 的表现都优于 4 帧的 `Localizer`。
*   **骨干模型限制:** 作者推测这可能是因为 `BLIP-2` 骨干模型在预训练时没有见过视频数据，因此其多帧连接（`long image`）可能无法有效地进行时序建模。这暗示了未来多帧 `Localizer` 仍有改进空间，特别是结合视频预训练。

### 6.1.12. `Localizer` 预训练设置的比较 (表 13)
以下是原文 Table 13 的结果：

<table>
<thead>
<tr>
<td>LOcaLIZER</td>
<td>NeXT-QA (Average)</td>
</tr>
</thead>
<tbody>
<tr>
<td>w/o Localizer</td>
<td>62.4</td>
</tr>
<tr>
<td>+ Moment-DETR</td>
<td>62.0</td>
</tr>
<tr>
<td>+ Our Localizer (without pre-training)</td>
<td>62.9</td>
</tr>
<tr>
<td>+ Our Localizer (weakly pre-trained with QVH ASR)</td>
<td>63.2</td>
</tr>
<tr>
<td>+ Our Localizer (pre-trained with QVH)</td>
<td><b>63.6</b></td>
</tr>
</tbody>
</table>

**关键发现:**
*   **预训练的有效性:** `Localizer` 的性能随着预训练的增强而提升。
*   **弱监督预训练的潜力:** 使用 `QVHighlights` (QVH) 的 ASR（自动语音识别）进行弱监督预训练，能够进一步缩小与使用完整 QVH 标注进行预训练的差距。这表明即使是弱监督信号，也能有效地提升 `Localizer` 的性能。

### 6.1.13. 迭代自优化的结果 (表 14)
以下是原文 Table 14 的结果：

<table>
<thead>
<tr>
<td>Iteration</td>
<td>NeXT-QA (Average)</td>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>73.8</td>
</tr>
<tr>
<td>2</td>
<td><b>74.2</b></td>
</tr>
<tr>
<td>3</td>
<td>73.7</td>
</tr>
</tbody>
</table>

**关键发现:**
*   **有限的迭代收益:** 两次迭代自优化相比一次迭代，性能略有提升。然而，从第三次迭代开始，性能趋于饱和甚至略有下降。这表明迭代自优化的收益可能存在上限，过多的迭代可能无法带来持续的提升。

### 6.1.14. `SeViLA` 框架的计算成本 (表 15)
以下是原文 Table 15 的结果：

<table>
<thead>
<tr>
<td>Model</td>
<td>Memory (GB)</td>
<td>Running Time (sec./sample)</td>
<td>Parameter (B)</td>
</tr>
</thead>
<tbody>
<tr>
<td>Answerer (4)</td>
<td>7.56</td>
<td>1.79</td>
<td>4.1</td>
</tr>
<tr>
<td>SeViLA (32 → 4)</td>
<td>7.98</td>
<td>3.28</td>
<td>4.2</td>
</tr>
</tbody>
</table>

**关键发现:**
*   **参数高效性:** `Localizer` 和 `Answerer` 大部分参数共享，因此添加 `Localizer` 只会带来非常小的额外内存开销 (`7.56GB` 增加到 `7.98GB`) 和参数量 (`4.1B` 增加到 `4.2B`)。
*   **运行时间增加:** 虽然参数量增加不大，但由于 `Localizer` 需要对多个帧进行独立处理以选择关键帧，导致每个样本的运行时间有所增加 (`1.79 sec./sample` 增加到 `3.28 sec./sample`)。

### 6.1.15. 提示词设计的影响 (表 16)
以下是原文 Table 16 的结果：

<table>
<thead>
<tr>
<td rowspan="2">Localization Prompt</td>
<td colspan="4">NExT-QA</td>
</tr>
<tr>
<td>Temporal</td>
<td>Casual</td>
<td>Descriptive</td>
<td>Average</td>
</tr>
</thead>
<tbody>
<tr>
<td>Does the frame have the information needed to answer the question correctly?</td>
<td>59.9</td>
<td>61.1</td>
<td>74.2</td>
<td>62.7</td>
</tr>
<tr>
<td>Does the provided frame contain the necessary information to accurately answer the given question?</td>
<td>59.9</td>
<td>60.8</td>
<td>75.0</td>
<td>62.7</td>
</tr>
<tr>
<td>Does the information within the frame provide the necessary details to accurately answer the given question?</td>
<td><b>60.4</b></td>
<td><b>61.0</b></td>
<td><b>74.6</b></td>
<td><b>62.9</b></td>
</tr>
</tbody>
</table>

**关键发现:**
*   **对提示词不敏感:** 实验结果表明，`Localizer` 对不同的定位提示词变化不敏感，性能差异较小。这表明 `BLIP-2` 强大的语言理解能力使其能够从各种语义相似的提示中捕获关键信息。

## 6.2. 数据呈现 (表格)

*   以下是原文 Table 1 的结果：

    <table>
    <thead>
    <tr>
    <td rowspan="3">Model (# Frames)</td>
    <td colspan="4">NExT-QA</td>
    <td colspan="4">STAR</td>
    <td rowspan="3">How2QA</td>
    <td rowspan="3">TVQA</td>
    <td rowspan="3">VLEP</td>
    </tr>
    <tr>
    <td colspan="4"></td>
    <td colspan="4"></td>
    </tr>
    <tr>
    <td>Tem.</td>
    <td>Cau.</td>
    <td>Des.</td>
    <td>Avg.</td>
    <td>Int.</td>
    <td>Seq.</td>
    <td>Pre.</td>
    <td>Fea.</td>
    <td>Avg.</td>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td colspan="12">(w/ speech input or use dense frames)</td>
    </tr>
    <tr>
    <td>HERO (dense/1fps) [36]</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>-</td>
    <td>73.8</td>
    <td>73.6</td>
    <td>-</td>
    </tr>
    <tr>
    <td>JustAsk (20) [84]</td>
    <td>51.4</td>
    <td>49.6</td>
    <td>63.1</td>
    <td>52.3</td>
    <td></td>
    <td></td>
    <td>-</td>
    <td></td>
    <td>84.4</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>FrozenBiLM (10) [85]</td>
    <td>-</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>-</td>
    <td>-</td>
    <td>86.7</td>
    <td>82.0</td>
    <td>-</td>
    </tr>
    <tr>
    <td>VidIL 4-shot (12) [72]</td>
    <td>-</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>72.0</td>
    </tr>
    <tr>
    <td>T+T (dense/1fps) [40]</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>-</td>
    <td>-</td>
    <td>92.4</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>T+T (+ASR, dense/1fps) [40]</td>
    <td></td>
    <td></td>
    <td></td>
    <td>-</td>
    <td></td>
    <td></td>
    <td>-</td>
    <td>-</td>
    <td>93.2</td>
    <td></td>
    <td>-</td>
    </tr>
    <tr>
    <td>Flamingo-80B 32-shot (30) [1]</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    </tr>
    <tr>
    <td>FrozenBiLM (10) [85]</td>
    <td>-</td>
    <td></td>
    <td>-</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>42.2</td>
    <td>-</td>
    <td>- 81.5</td>
    <td>57.5</td>
    <td></td>
    </tr>
    <tr>
    <td>All-in-One (32) [67]</td>
    <td>48.6</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>48.0</td>
    <td>63.2</td>
    <td>50.6</td>
    <td>47.5</td>
    <td>50.8</td>
    <td>47.7</td>
    <td>44.0</td>
    <td>47.5</td>
    <td>-</td>
    <td>-</td>
    <td></td>
    </tr>
    <tr>
    <td>Temp[ATP] (32) [3]</td>
    <td>49.3</td>
    <td>48.6</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>65.0</td>
    <td>51.5</td>
    <td>50.6</td>
    <td>52.8</td>
    <td>49.3</td>
    <td>40.6</td>
    <td>48.3</td>
    <td>-</td>
    <td>-</td>
    <td></td>
    </tr>
    <tr>
    <td>VGT (32) [78]</td>
    <td>55.0</td>
    <td>52.2</td>
    <td>64.0</td>
    <td>55.0</td>
    <td></td>
    <td></td>
    <td></td>
    <td>44.2</td>
    <td></td>
    <td></td>
    <td></td>
    </tr>
    <tr>
    <td>MIST (32) [18]</td>
    <td>56.6</td>
    <td>54.6</td>
    <td>66.9</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>57.1</td>
    <td>55.5</td>
    <td>54.2</td>
    <td>54.2</td>
    <td>44.4</td>
    <td>51.1</td>
    <td>-</td>
    <td></td>
    <td></td>
    </tr>
    <tr>
    <td>VFC (32) [50]</td>
    <td>53.3</td>
    <td>57.6</td>
    <td>72.8</td>
    <td>58.6</td>
    <td>-</td>
    <td>-</td>
    <td></td>
    <td>-</td>
    <td>-</td>
    <td></td>
    <td></td>
    </tr>
    <tr>
    <td>CoVGT (32) [79]</td>
    <td>57.4</td>
    <td>58.8</td>
    <td>69.3</td>
    <td>60.0</td>
    <td></td>
    <td>-</td>
    <td></td>
    <td>45.9</td>
    <td>-</td>
    <td></td>
    <td></td>
    </tr>
    <tr>
    <td>SeViTFiD (10) [24]</td>
    <td>-</td>
    <td></td>
    <td></td>
    <td>60.6</td>
    <td></td>
    <td></td>
    <td></td>
    <td>-</td>
    <td></td>
    <td>-</td>
    <td></td>
    </tr>
    <tr>
    <td>HiTeA (16) [87]</td>
    <td>58.3</td>
    <td>62.4</td>
    <td>75.6</td>
    <td>63.1</td>
    <td>-</td>
    <td>-</td>
    <td></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>InternVideo* (8) [71]</td>
    <td>58.5</td>
    <td>62.5</td>
    <td>75.8</td>
    <td>63.2</td>
    <td></td>
    <td></td>
    <td></td>
    <td>62.7</td>
    <td>65.6</td>
    <td>54.9</td>
    <td>51.9</td>
    <td>58.7</td>
    <td>79.0</td>
    <td>57.2</td>
    <td>63.9</td>
    </tr>
    <tr>
    <td>BLIP-2voting (4)</td>
    <td>65.2</td>
    <td>70.1</td>
    <td>80.1</td>
    <td>70.1</td>
    <td></td>
    <td></td>
    <td></td>
    <td>52.3</td>
    <td>54.8</td>
    <td>49.0</td>
    <td>51.2</td>
    <td>51.8</td>
    <td>79.6</td>
    <td>54.5</td>
    <td>67.0</td>
    </tr>
    <tr>
    <td>BLIP-2concat (ANSWERER) (4)</td>
    <td>68.1</td>
    <td>72.9</td>
    <td>81.2</td>
    <td>72.6</td>
    <td></td>
    <td></td>
    <td></td>
    <td>65.4</td>
    <td>69.0</td>
    <td>59.7</td>
    <td>54.2</td>
    <td>62.0</td>
    <td>82.2</td>
    <td>59.8</td>
    <td>68.6</td>
    </tr>
    <tr>
    <td>SEVILA† (32 → 4)</td>
    <td>68.8</td>
    <td>73.4</td>
    <td>83.5</td>
    <td>73.4</td>
    <td></td>
    <td>63.2</td>
    <td>66.6</td>
    <td>61.3</td>
    <td>60.0</td>
    <td>62.7</td>
    <td></td>
    <td></td>
    <td>83.7</td>
    <td>59.7</td>
    <td>69.0</td>
    </tr>
    <tr>
    <td>SeViLA (32 → 4)</td>
    <td><b>69.4</b></td>
    <td><b>74.2</b></td>
    <td><u>81.3</u></td>
    <td><b>73.8</b></td>
    <td></td>
    <td><b>63.7</b></td>
    <td><b>70.4</b></td>
    <td><b>63.1</b></td>
    <td><b>62.4</b></td>
    <td><b>64.9</b></td>
    <td></td>
    <td></td>
    <td><b>83.6</b></td>
    <td><b>61.6</b></td>
    <td><b>68.9</b></td>
    </tr>
    </tbody>
    </table>

*   以下是原文 Table 2 的结果：

    <table>
    <thead>
    <tr>
    <td rowspan="3">Model (# Frames)</td>
    <td colspan="4">NExT-QA</td>
    <td colspan="5">STAR</td>
    <td rowspan="3">How2QA</td>
    <td rowspan="3">TVQA</td>
    <td rowspan="3">VLEP</td>
    </tr>
    <tr>
    <td colspan="4"></td>
    <td colspan="5"></td>
    </tr>
    <tr>
    <td>Tem.</td>
    <td>Cau.</td>
    <td>Des.</td>
    <td>Avg.</td>
    <td>Int.</td>
    <td>Seq.</td>
    <td>Pre.</td>
    <td>Fea.</td>
    <td>Avg.</td>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td colspan="13">(w/ speech input or use dense frames)</td>
    </tr>
    <tr>
    <td>JustAsk (20) [84]</td>
    <td>-</td>
    <td>-</td>
    <td></td>
    <td></td>
    <td></td>
    <td>-</td>
    <td></td>
    <td></td>
    <td>-</td>
    <td>51.1</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>FrozenBiLM (10) [85]</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>-</td>
    <td>58.4</td>
    <td>59.2</td>
    <td>-</td>
    </tr>
    <tr>
    <td>ViperGPT (dense/1fps) [63]</td>
    <td></td>
    <td>-</td>
    <td>-</td>
    <td>60.0</td>
    <td>-</td>
    <td></td>
    <td></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>Flamingo-80B (30) [1]</td>
    <td></td>
    <td></td>
    <td>-</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    </tr>
    <tr>
    <td>FrozenBiLM (10) [85]</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td></td>
    <td></td>
    <td></td>
    <td>-</td>
    <td>39.7</td>
    <td>- 41.9</td>
    <td>- 29.7</td>
    <td>-</td>
    </tr>
    <tr>
    <td>VFC (32) [50]</td>
    <td>- 45.4</td>
    <td></td>
    <td>51.6</td>
    <td>64.1</td>
    <td>51.5</td>
    <td></td>
    <td></td>
    <td>-</td>
    <td>-</td>
    <td></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>InternVideo* (8) [71]</td>
    <td>43.4</td>
    <td>48.0</td>
    <td>65.1</td>
    <td>49.1</td>
    <td></td>
    <td></td>
    <td>43.8</td>
    <td>43.2</td>
    <td>42.3</td>
    <td>37.4</td>
    <td>41.6</td>
    <td></td>
    <td></td>
    <td>62.2</td>
    <td>35.9</td>
    <td>58.7</td>
    </tr>
    <tr>
    <td>BLIP-2voting (4)</td>
    <td>59.1</td>
    <td>61.3</td>
    <td>74.9</td>
    <td>62.7</td>
    <td>41.8</td>
    <td></td>
    <td>39.7</td>
    <td>40.2</td>
    <td>39.5</td>
    <td>40.3</td>
    <td></td>
    <td></td>
    <td>69.8</td>
    <td>35.7</td>
    <td>63.8</td>
    </tr>
    <tr>
    <td>BLIP-2concat (AnswereR) (4)</td>
    <td>59.7</td>
    <td>60.8</td>
    <td>73.8</td>
    <td>62.4</td>
    <td></td>
    <td></td>
    <td>45.5</td>
    <td>41.8</td>
    <td>41.8</td>
    <td>40.0</td>
    <td>42.2</td>
    <td></td>
    <td></td>
    <td>70.8</td>
    <td>36.6</td>
    <td>64.0</td>
    </tr>
    <tr>
    <td>SEVILA† (32 → 4)</td>
    <td><b>61.3</b></td>
    <td></td>
    <td><b>61.5</b></td>
    <td><b>75.6</b></td>
    <td><b>63.6</b></td>
    <td><b>48.3</b></td>
    <td><b>45.0</b></td>
    <td><b>44.4</b></td>
    <td><b>40.8</b></td>
    <td><b>44.6</b></td>
    <td></td>
    <td></td>
    <td><b>72.3</b></td>
    <td><b>38.2</b></td>
    <td><b>64.4</b></td>
    </tr>
    </tbody>
    </table>

*   以下是原文 Table 3 的结果：

    <table>
    <thead>
    <tr>
    <td rowspan="3"></td>
    <td colspan="2">AnSwERER</td>
    <td rowspan="3">Keyframe</td>
    <td colspan="4">NExT-QA</td>
    <td colspan="5">STAR</td>
    <td rowspan="3">How2QA</td>
    <td rowspan="3">TVQA</td>
    <td rowspan="3">VLEP</td>
    </tr>
    <tr>
    <td rowspan="2"># frame</td>
    <td rowspan="2">finetuned?</td>
    <td colspan="4"></td>
    <td colspan="5"></td>
    </tr>
    <tr>
    <td>Tem.</td>
    <td>Cau.</td>
    <td>Des.</td>
    <td>Avg.</td>
    <td>Int.</td>
    <td>Seq.</td>
    <td>Pre.</td>
    <td>Fea.</td>
    <td>Avg.</td>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>A.</td>
    <td>32</td>
    <td>×</td>
    <td>uniform</td>
    <td>54.7</td>
    <td>56.7</td>
    <td>67.8</td>
    <td>57.7</td>
    <td>46.2</td>
    <td>43.6</td>
    <td>40.7</td>
    <td>41.0</td>
    <td>42.8</td>
    <td>67.0</td>
    <td>33.2</td>
    <td>54.0</td>
    </tr>
    <tr>
    <td>B.</td>
    <td>4</td>
    <td>×</td>
    <td>uniform</td>
    <td>59.7</td>
    <td>60.8</td>
    <td>73.8</td>
    <td>62.4</td>
    <td>45.5</td>
    <td>41.8</td>
    <td>41.8</td>
    <td>40.0</td>
    <td>42.2</td>
    <td>70.8</td>
    <td>36.6</td>
    <td>64.0</td>
    </tr>
    <tr>
    <td>C.</td>
    <td>4</td>
    <td>×</td>
    <td>LocaLizeR†</td>
    <td>61.3</td>
    <td>61.5</td>
    <td>75.6</td>
    <td>63.6</td>
    <td>48.3</td>
    <td>45.0</td>
    <td>44.4</td>
    <td>40.8</td>
    <td>44.6</td>
    <td>72.3</td>
    <td>38.2</td>
    <td>64.4</td>
    </tr>
    <tr>
    <td>D.</td>
    <td>4</td>
    <td>×</td>
    <td>LocaLizer</td>
    <td>62.3</td>
    <td>63.1</td>
    <td>74.9</td>
    <td>64.6</td>
    <td>49.0</td>
    <td>46.4</td>
    <td>45.2</td>
    <td>41.6</td>
    <td>45.5</td>
    <td>72.9</td>
    <td>39.1</td>
    <td>64.6</td>
    </tr>
    <tr>
    <td>E.</td>
    <td>4</td>
    <td>√</td>
    <td>uniform</td>
    <td>68.1</td>
    <td>72.9</td>
    <td>81.2</td>
    <td>72.6</td>
    <td>65.4</td>
    <td>69.0</td>
    <td>59.7</td>
    <td>54.2</td>
    <td>62.0</td>
    <td>82.2</td>
    <td>59.8</td>
    <td>68.6</td>
    </tr>
    <tr>
    <td>F.</td>
    <td>4</td>
    <td>√</td>
    <td>LOCAlizeR†</td>
    <td>68.8</td>
    <td>73.4</td>
    <td>83.5</td>
    <td>73.4</td>
    <td>63.2</td>
    <td>66.6</td>
    <td>61.3</td>
    <td>60.0</td>
    <td>62.7</td>
    <td>83.7</td>
    <td>59.7</td>
    <td>69.0</td>
    </tr>
    <tr>
    <td>G.</td>
    <td>4</td>
    <td>√</td>
    <td>LOcaLIzER</td>
    <td><b>69.4</b></td>
    <td><b>74.2</b></td>
    <td>81.3</td>
    <td><b>73.8</b></td>
    <td><b>63.7</b></td>
    <td><b>70.4</b></td>
    <td><b>63.1</b></td>
    <td><b>62.4</b></td>
    <td><b>64.9</b></td>
    <td><b>83.6</b></td>
    <td><b>61.6</b></td>
    <td><b>68.9</b></td>
    </tr>
    </tbody>
    </table>

*   以下是原文 Table 4 的结果：

    <table>
    <thead>
    <tr>
    <td>Model</td>
    <td>R1@0.5</td>
    <td>R1@0.7</td>
    <td>mAP</td>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>CAL [13]</td>
    <td>25.4</td>
    <td>11.5</td>
    <td>9.8</td>
    </tr>
    <tr>
    <td>XML [29]</td>
    <td>41.8</td>
    <td>30.3</td>
    <td>32.1</td>
    </tr>
    <tr>
    <td>Moment-DETR [30]</td>
    <td>52.8</td>
    <td>33.0</td>
    <td>30.7</td>
    </tr>
    <tr>
    <td>QD-DETR [51]</td>
    <td><b>62.4</b></td>
    <td><b>44.9</b></td>
    <td><b>39.8</b></td>
    </tr>
    <tr>
    <td>LocALizeR (Ours)</td>
    <td>54.5</td>
    <td>36.5</td>
    <td>32.3</td>
    </tr>
    </tbody>
    </table>

*   以下是原文 Table 5 的结果：

    <table>
    <thead>
    <tr>
    <td rowspan="2">PT</td>
    <td rowspan="2">SR</td>
    <td colspan="4">NExT-QA</td>
    <td rowspan="2">How2QA</td>
    </tr>
    <tr>
    <td>Tem.</td>
    <td>Cau.</td>
    <td>Des.</td>
    <td>Avg.</td>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>-</td>
    <td>-</td>
    <td>60.4</td>
    <td>61.0</td>
    <td>74.6</td>
    <td>62.9</td>
    <td>70.7</td>
    </tr>
    <tr>
    <td>✓</td>
    <td>-</td>
    <td>61.3</td>
    <td>61.5</td>
    <td>75.6</td>
    <td>63.6</td>
    <td>72.3</td>
    </tr>
    <tr>
    <td>-</td>
    <td>✓</td>
    <td>62.1</td>
    <td>62.6</td>
    <td>75.1</td>
    <td>64.3</td>
    <td>72.8</td>
    </tr>
    <tr>
    <td>✓</td>
    <td>✓</td>
    <td><b>62.3</b></td>
    <td><b>63.1</b></td>
    <td><b>74.9</b></td>
    <td><b>64.6</b></td>
    <td><b>72.9</b></td>
    </tr>
    </tbody>
    </table>

*   以下是原文 Table 6 的结果：

    <table>
    <thead>
    <tr>
    <td>Method</td>
    <td colspan="4">NExT-QA</td>
    </tr>
    <tr>
    <td></td>
    <td>Tem.</td>
    <td>Cau.</td>
    <td>Des.</td>
    <td>Avg.</td>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>AnswERER</td>
    <td>59.7</td>
    <td>60.8</td>
    <td>73.7</td>
    <td>62.4</td>
    </tr>
    <tr>
    <td>(zero-shot)</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    </tr>
    <tr>
    <td>+ CLIP [55]</td>
    <td>59.2</td>
    <td>59.5</td>
    <td>72.1</td>
    <td>60.6</td>
    </tr>
    <tr>
    <td>+ Moment-DETR [30]</td>
    <td>60.0</td>
    <td>60.6</td>
    <td>72.5</td>
    <td>61.8</td>
    </tr>
    <tr>
    <td>+ Localizer†</td>
    <td><b>61.3</b></td>
    <td><b>61.5</b></td>
    <td><b>75.6</b></td>
    <td><b>63.6</b></td>
    </tr>
    <tr>
    <td>(fine-tuning)</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    </tr>
    <tr>
    <td>+ ATP [3]</td>
    <td>60.4</td>
    <td>61.3</td>
    <td>73.4</td>
    <td>62.8</td>
    </tr>
    <tr>
    <td>+ Differentiable Top-K [8]</td>
    <td>59.5</td>
    <td>59.7</td>
    <td>72.7</td>
    <td>61.6</td>
    </tr>
    <tr>
    <td>+ LocaliZeR</td>
    <td><b>62.3</b></td>
    <td><b>63.1</b></td>
    <td><b>74.9</b></td>
    <td><b>64.6</b></td>
    </tr>
    </tbody>
    </table>

*   以下是原文 Table 7 的结果：

    <table>
    <thead>
    <tr>
    <td rowspan="2">Settings</td>
    <td colspan="4">NExT-QA</td>
    <td rowspan="2">How2QA</td>
    </tr>
    <tr>
    <td>Tem.</td>
    <td>Cau.</td>
    <td>Des.</td>
    <td>Avg.</td>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>BLIP-2voting (8)</td>
    <td>59.9</td>
    <td>60.2</td>
    <td>72.4</td>
    <td>62.0</td>
    <td>69.8</td>
    </tr>
    <tr>
    <td>8→1</td>
    <td>59.8</td>
    <td>61.1</td>
    <td>76.0</td>
    <td>62.9</td>
    <td>72.4</td>
    </tr>
    <tr>
    <td>16→1</td>
    <td>59.2</td>
    <td>62.6</td>
    <td>74.9</td>
    <td>63.4</td>
    <td>73.2</td>
    </tr>
    <tr>
    <td>16→4</td>
    <td>60.7</td>
    <td>61.5</td>
    <td>75.8</td>
    <td>63.4</td>
    <td>72.4</td>
    </tr>
    <tr>
    <td>32→4</td>
    <td><b>61.3</b></td>
    <td><b>61.5</b></td>
    <td><b>75.6</b></td>
    <td><b>63.6</b></td>
    <td><b>72.3</b></td>
    </tr>
    <tr>
    <td>32→8</td>
    <td>59.4</td>
    <td>60.9</td>
    <td>74.7</td>
    <td>62.5</td>
    <td>71.3</td>
    </tr>
    <tr>
    <td>64→8</td>
    <td>58.9</td>
    <td>60.9</td>
    <td>74.0</td>
    <td>62.2</td>
    <td>71.8</td>
    </tr>
    </tbody>
    </table>

*   以下是原文 Table 8 的结果：

    <table>
    <thead>
    <tr>
    <td colspan="2">Frame Sampling</td>
    <td colspan="4">NExT-QA</td>
    </tr>
    <tr>
    <td>Training</td>
    <td>Inference</td>
    <td>Temp.</td>
    <td>Cau.</td>
    <td>Des.</td>
    <td>Avg.</td>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>Random</td>
    <td>Uniform</td>
    <td>68.1</td>
    <td>72.9</td>
    <td>81.2</td>
    <td>72.6</td>
    </tr>
    <tr>
    <td>Random</td>
    <td>LoCAlizeR†</td>
    <td>67.6</td>
    <td>73.4</td>
    <td>84.0</td>
    <td>73.1</td>
    </tr>
    <tr>
    <td>LOCALiZeR</td>
    <td>Uniform</td>
    <td>68.2</td>
    <td>72.7</td>
    <td>80.0</td>
    <td>72.3</td>
    </tr>
    <tr>
    <td>LocalizeR†</td>
    <td>LOCalizeR</td>
    <td><b>68.8</b></td>
    <td><b>73.4</b></td>
    <td><b>83.5</b></td>
    <td><b>73.4</b></td>
    </tr>
    </tbody>
    </table>

*   以下是原文 Table 9 的结果：

    <table>
    <thead>
    <tr>
    <td rowspan="2">Datasets</td>
    <td colspan="2">BLIP-2voting (Oracle)</td>
    </tr>
    <tr>
    <td>Zero-Shot</td>
    <td>Fine-tuned</td>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>NExT-QA (Avg.)</td>
    <td>62.7 (70.1)</td>
    <td>70.1 (79.7)</td>
    </tr>
    <tr>
    <td>STAR (Avg.)</td>
    <td>40.3 (52.9)</td>
    <td>51.8 (72.2)</td>
    </tr>
    <tr>
    <td>How2QA</td>
    <td>69.8 (77.8)</td>
    <td>79.6 (86.4)</td>
    </tr>
    <tr>
    <td>TVQA</td>
    <td>35.7 (45.4)</td>
    <td>54.5 (69.0)</td>
    </tr>
    <tr>
    <td>VLEP</td>
    <td>63.8 (70.5)</td>
    <td>67.0 (79.1)</td>
    </tr>
    </tbody>
    </table>

## 6.3. 消融实验/参数分析
论文通过一系列消融研究和参数分析，深入探讨了 `SeViLA` 框架各组件的有效性及其对性能的影响。

*   <strong>`Localizer` 预训练和自优化的影响 (表 5):</strong>
    *   **分析:** 结果表明，无论是 `QVHighlights` 数据集上的预训练 (`PT`) 还是反向链中的自优化 (`SR`)，都对 `Localizer` 的性能有显著提升。当两者结合时，性能达到最佳。这证明了 `Localizer` 的训练策略是有效的，尤其是在减少对昂贵标注依赖方面。
    *   **结论:** 预训练提供了基础的时序定位能力，而自优化则使其更加适应具体的问答任务。

*   <strong>关键帧选择范围和数量的影响 (表 7):</strong>
    *   **分析:** 即使只选择一个关键帧，`Localizer` 也能带来性能提升，这强调了关键帧选择的有效性。对于 `NExT-QA-Temporal` 等需要时序理解的任务，多关键帧通常表现更好。然而，过多的输入帧（如从 32 帧到 64 帧）反而导致性能下降。
    *   **结论:** 适量的关键帧（例如 4 帧）在大多数情况下表现最优，过多的帧可能会引入冗余信息，干扰 `Image-LM` 的处理。

*   <strong>`Answerer` 微调期间帧采样策略的影响 (表 8):</strong>
    *   **分析:** `Answerer` 在训练和推理时都使用 `Localizer` 提供的关键帧时，性能最佳。如果训练时使用随机帧，而推理时使用 `Localizer` 帧，或者反之，性能都会有所下降。
    *   **结论:** 训练和推理阶段保持帧采样策略的一致性（即都使用 `Localizer` 选择的关键帧）非常重要，这有助于减少域偏移，使模型更好地适应任务。

*   <strong>提示词设计的影响 (表 16):</strong>
    *   **分析:** 实验结果显示，`Localizer` 对不同的定位提示词（即使语义上略有差异）表现出较低的敏感性，性能波动很小。
    *   **结论:** `BLIP-2` 强大的语言理解能力使其能够从多种表达方式中捕获相似的意图，因此在提示词选择上具有一定的鲁棒性。

*   <strong>单帧与多帧 `Localizer` 的比较 (表 12):</strong>
    *   **分析:** 令人惊讶的是，单帧 `Localizer` 的性能优于多帧 `Localizer`（将多帧拼接成“长图像”输入）。作者推测这可能是因为 `BLIP-2` 在预训练时没有见过视频数据，因此其多帧处理能力可能不佳。
    *   **结论:** 在当前 `BLIP-2` 骨干下，单帧的精准定位比强行进行多帧时序建模更有效。这为未来的研究指明了方向，即如何更好地将 `Image-LM` 适配到视频的多帧时序建模。

*   <strong>迭代自优化的效果 (表 14):</strong>
    *   **分析:** 两次迭代自优化略微提升了性能，但之后性能趋于饱和或略有下降。
    *   **结论:** 自优化是有效的，但可能存在收益递减的效应，并不是越多迭代越好。

*   <strong>计算成本分析 (表 15):</strong>
    *   **分析:** `SeViLA` 框架在引入 `Localizer` 后，只增加了少量内存和参数。然而，每个样本的运行时间有所增加。
    *   **结论:** `SeViLA` 保持了良好的参数效率，但时序定位过程会带来一定的推理时间开销，这是性能提升的代价。

# 7. 总结与思考

## 7.1. 结论总结
本文提出了 `SeViLA` (Self-Chained Video Localization-Answering) 框架，成功地将一个单一的预训练图像-语言模型 `BLIP-2` 适配到视频理解任务中，以同时处理语言感知的时间关键帧定位和视频问答。

`SeViLA` 框架的核心创新在于其<strong>自链 (self-chaining)</strong> 机制：
1.  <strong>前向链 (Forward Chain):</strong> `Localizer` 模块负责从视频中智能地选择与语言查询相关的关键帧，然后 `Answerer` 模块利用这些精选的关键帧来生成准确的答案。这种级联推理方式确保了模型能够高效地聚焦于视频中最相关的信息。
2.  <strong>反向链 (Reverse Chain):</strong> `Answerer` 模块通过生成关键帧伪标签 (pseudo-labels) 的方式，对 `Localizer` 进行自优化。这种机制巧妙地解决了传统视频时序定位任务中昂贵人工标注数据的难题，提高了 `Localizer` 的语言感知时序定位准确性。

    实验结果表明，`SeViLA` 框架在五个具有挑战性的视频问答和事件预测基准测试中，无论是微调 (fine-tuning) 还是零样本 (zero-shot) 设置下，都取得了超越现有最先进 (state-of-the-art) 基线的性能。一系列全面的消融研究和分析也验证了 `Localizer` 的有效性、自优化机制的价值以及关键帧数量选择的重要性。

总而言之，`SeViLA` 提供了一种高效且有效的方法，将强大的图像-语言模型能力扩展到视频领域，同时克服了视频数据时序复杂性和标注稀缺性的挑战。这项工作鼓励未来继续深入研究视频理解中的时序定位问题。

## 7.2. 局限性与未来工作
论文作者指出了 `SeViLA` 框架的一些局限性，并提出了未来可能的研究方向：

*   <strong>局限性：帧级定位的细粒度问题 (Frame-level Localization Limitations):</strong>
    *   尽管 `SeViLA` 的 `Localizer` 能够有效定位语言感知的关键帧，但它目前仍是**帧级定位**。这意味着它将视频理解为一系列独立的帧，然后选择其中“最重要”的几帧。
    *   这种方法可能无法很好地处理一些<strong>复杂、细粒度 (fine-grained) 的时序事件</strong>。例如，区分“打开门”和“关闭门”需要理解动作的时序进程和状态变化，而不仅仅是识别某个关键帧。仅仅依靠几帧可能无法捕捉到动作的完整语境或精确的时序关系。
*   <strong>未来工作：结构化时序预测 (Structured Prediction for Temporal Localization):</strong>
    *   为了克服帧级定位的局限性，未来的研究可以探索超越帧级的<strong>结构化时序预测 (structured prediction for temporal localization)</strong> 方法。这意味着模型不仅要识别关键帧，还要能够预测事件的开始、结束时间，或者识别更复杂的时序结构（如事件序列、动作持续时间等）。
*   <strong>更广泛的影响：大型模型偏见 (Broader Impacts - Large Model Biases):</strong>
    *   `SeViLA` 框架依赖于在海量互联网规模数据上训练的大型图像-语言模型 `BLIP-2`。与大多数此类大型模型一样，它可能会偶尔产生意外或不当的响应，并可能反映出与性别、种族或性取向相关的社会偏见。
    *   **未来工作：** 需要对大型图像-语言模型进行更多研究，以评估和减轻这些负面偏见和有害输出。

## 7.3. 个人启发与批判
### 7.3.1. 个人启发
*   **Image-LM 潜力巨大：** 论文最令人振奋的发现之一是，像 `BLIP-2` 这样强大的 `Image-LM`，在经过适当的适配后，即使在零样本设置下，也能在某些视频任务上超越专门训练的 `Video-LM`。这表明了 `Image-LM` 在模型规模和预训练数据方面的优势是巨大的，未来将 `Image-LM` 高效迁移到其他多模态领域（如音频-语言、3D-语言）将是一个富有前景的方向。
*   **伪标签与自优化是关键：** 视频数据标注的高成本是限制 `Video-LM` 发展的主要瓶颈。`SeViLA` 提出的反向链伪标签自优化机制，提供了一个优雅且有效的方式来缓解这一问题。这种“模型教模型”的范式，对于数据稀缺领域的模型开发具有普遍的借鉴意义。
*   <strong>“聚焦”</strong>的重要性： 无论是人类还是AI，在处理复杂信息时，“聚焦”都是核心能力。`Localizer` 的作用就是为 `Answerer` 提供“最相关的部分”，这不仅提高了效率，也提升了准确性。这种分而治之、协同合作的模块化思想，可以在其他复杂的跨模态任务中得到应用。
*   **Prompt Engineering 的灵活性：** 尽管论文发现 `Localizer` 对定位提示词不敏感，这恰恰说明了 `BLIP-2` 骨干强大的语言理解能力，使得模型在 `prompt` 设计上具有一定的鲁棒性和灵活性，降低了 `prompt` 微调的难度。

### 7.3.2. 批判
*   **时序建模的深层挑战：**
    *   尽管 `SeViLA` 在帧级定位上表现出色，但其 `Localizer` 仍是基于单帧得分进行选择。论文也指出，尝试多帧 `Localizer` 效果不佳，并归因于 `BLIP-2` 未见过视频数据。这揭示了一个深层问题：`Image-LM` 即使规模再大，在未经视频数据预训练的情况下，其内在的视觉编码器可能未能学习到捕捉复杂时序变化的本质特征。仅仅拼接帧或选择关键帧，难以真正理解视频中事件的“发生”、“发展”和“结束”。
    *   对于那些需要理解“为什么发生”、“如何发生”或“下一步会怎样”的复杂时序或因果推理任务，这种帧级选择可能仍有其固有的天花板。例如，区分“一个人拿起杯子”和“一个人放下杯子”，可能需要更精细的时序特征提取和事件因果链的推理，而非仅仅关注几个关键帧。
*   **伪标签的质量依赖与潜在误差积累：**
    *   反向链的有效性高度依赖于 `Answerer` 的初始性能。如果 `Answerer` 本身存在缺陷或在某些场景下预测不准确，那么它生成的伪标签就会带有噪声。这种噪声可能会被 `Localizer` 学到，甚至在迭代过程中放大，导致模型陷入局部最优或积累误差。
    *   论文虽然进行了迭代自优化实验，但结果显示性能饱和，也可能暗示了伪标签质量的限制。
*   **计算效率的平衡：**
    *   尽管参数高效，但 `Localizer` 需要对所有均匀采样的帧进行独立处理以计算得分，这会增加推理时间。对于长视频或需要实时响应的应用，这种开销仍需进一步优化。
*   **对 `LLM` 的黑箱依赖：**
    *   `SeViLA` 严重依赖 `BLIP-2` 中冻结的 `LLM` 来进行问答和生成帧得分。`LLM` 的决策过程通常是黑箱的，这使得我们难以完全理解模型为何选择这些关键帧或给出某个答案，降低了模型的可解释性。
*   **泛化能力与领域适应：**
    *   虽然在多个基准上表现出色，但这些基准主要集中在问答和事件预测。对于更广泛的视频理解任务（如行为识别、异常检测、视频摘要等），`SeViLA` 的泛化能力如何，以及如何进行有效适配，仍需进一步探索。
    *   `Localizer` 的预训练是在 `QVHighlights`（一个时刻检索数据集）上进行的，其定位目标是“高亮时刻”，这可能与某些问答任务中的“关键时刻”存在语义偏差，尤其是在更抽象或因果关系的问题中。