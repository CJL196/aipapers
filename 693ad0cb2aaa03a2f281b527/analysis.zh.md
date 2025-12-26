# 1. 论文基本信息

## 1.1. 标题
OpenVLA: 一个开源的视觉-语言-行为模型 (OpenVLA: An Open-Source Vision-Language-Action Model)

## 1.2. 作者
Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, Chelsea Finn.

这些作者来自多个顶尖学术和研究机构，包括：
*   <strong>斯坦福大学 (Stanford University)</strong> (1)
*   <strong>加州大学伯克利分校 (UC Berkeley)</strong> (2)
*   <strong>谷歌 DeepMind (Google DeepMind)</strong> (4)
*   <strong>丰田研究院 (Toyota Research Institute, TRI)</strong> (3)
*   <strong>麻省理工学院 (MIT)</strong> (6)
*   **Physical Intelligence** (公司) (5)

    作者团队汇集了机器人学、机器学习和自然语言处理领域的众多知名学者和研究员，如 Sergey Levine, Chelsea Finn, Dorsa Sadigh, Percy Liang 等，显示了该研究的强大背景和权威性。

## 1.3. 发表期刊/会议
该论文目前作为预印本 (preprint) 发布在 **arXiv** 上。arXiv 是一个开放获取的学术论文发布平台，允许研究者在正式同行评审前分享他们的研究成果。虽然不是正式发表，但该平台是计算机科学等领域快速传播最新研究的重要渠道。

## 1.4. 发表年份
2024年6月13日

## 1.5. 摘要
大型策略模型通过在互联网规模的视觉-语言数据和多样化的机器人演示数据上进行预训练，有潜力改变我们教授机器人新技能的方式：我们可以微调 (fine-tune) 这种<strong>视觉-语言-行为 (Vision-Language-Action, VLA)</strong> 模型，以获得用于视觉运动控制的、鲁棒且可泛化的策略，而无需从头开始训练新行为。然而，VLA在机器人领域的广泛应用面临挑战，因为：1) 现有的VLA大多是闭源且公众无法访问的；2) 先前的工作未能探索有效微调VLA以适应新任务的方法，而这是推广应用的关键。

为了应对这些挑战，我们推出了 **OpenVLA**，一个70亿参数的开源VLA模型，它在包含97万个真实世界机器人演示的多样化数据集上进行了训练。OpenVLA建立在一个 **Llama 2** 语言模型之上，并结合了一个融合了来自 **DINOv2** 和 **SigLIP** 预训练特征的视觉编码器。

得益于增加的数据多样性和新的模型组件，OpenVLA在通用操作任务上展示了强大的效果，在横跨29个任务和多种机器人平台（具身形态）的测试中，其绝对任务成功率比闭源模型如 **RT-2-X (55B)** 高出16.5%，而参数量仅为其1/7。我们进一步证明，可以有效地将OpenVLA微调到新场景中，在涉及多个对象和强大语言关联能力的多任务环境中表现出特别强的泛化能力，并以20.4%的优势超越了如 **Diffusion Policy** 这样表现力强的从头模仿学习方法。

我们还探索了计算效率。作为一项独立贡献，我们展示了OpenVLA可以通过现代<strong>低秩自适应 (low-rank adaptation, LoRA)</strong> 方法在消费级GPU上进行微调，并通过<strong>量化 (quantization)</strong> 技术高效地提供服务，而不会降低下游任务的成功率。

最后，我们发布了模型检查点、微调笔记本和我们的PyTorch代码库，该代码库内置了对在 Open X-Embodiment 数据集上大规模训练VLA的支持。

## 1.6. 原文链接
*   **arXiv 页面:** [https://arxiv.org/abs/2406.09246](https://arxiv.org/abs/2406.09246)
*   **PDF 链接:** [https://arxiv.org/pdf/2406.09246v3.pdf](https://arxiv.org/pdf/2406.09246v3.pdf)
*   **发布状态:** 预印本 (Preprint)。

# 2. 整体概括

## 2.1. 研究背景与动机
当前机器人学习领域的一个核心难题是<strong>泛化性 (generalization)</strong>。通过模仿学习训练出的机器人策略，虽然在特定任务上表现良好（如在不同位置拿起同一个杯子），但当环境稍作改变（如出现新的干扰物、使用未见过的物体、或执行全新的指令）时，其性能会急剧下降。

与此同时，视觉和语言领域的<strong>基础模型 (Foundation Models)</strong>，如 `CLIP`、`Llama 2` 等，由于在海量的互联网数据上进行了预训练，展现出了惊人的泛化能力。它们能够理解新概念、识别新物体，并进行复杂的推理。

这就带来了一个明显的机遇与挑战：
*   **机遇：** 能否将这些强大的视觉和语言基础模型的能力“迁移”到机器人控制领域，从而训练出能够泛化到新物体、新场景和新任务的“通用机器人策略”？
*   <strong>挑战与空白 (Gap)：</strong>
    1.  **闭源与不可及性：** 谷歌等公司提出的 `RT-2` 等<strong>视觉-语言-行为 (VLA)</strong> 模型虽然验证了这条路线的可行性，但它们是闭源的。学术界和更广泛的开发者社区无法访问模型、数据和训练代码，这极大地阻碍了该领域的研究进展。
    2.  **缺乏高效适应方法：** 即使有了通用的预训练模型，如何将其高效地“适配”或“微调”到一个新的、特定的机器人任务上（尤其是在计算资源有限的情况下），也是一个未被充分探索的关键问题。

        这篇论文的切入点正是为了解决上述两个核心问题：**创建一个强大的、完全开源的VLA模型，并提供一套完整的、高效的微调和部署方案，从而推动整个机器人学习社区的发展。**

## 2.2. 核心贡献/主要发现
本文的核心贡献可以概括为以下四点：

1.  **发布了 OpenVLA 模型：** 提出了一个70亿参数的、**完全开源**的VLA模型。该模型在架构上创新地融合了 `DINOv2`（提供精细的空间特征）和 `SigLIP`（提供高级的语义特征）作为视觉主干，并以 `Llama 2` 作为语言模型基础。
2.  **实现了最先进的性能：** OpenVLA 在大规模机器人操作任务上取得了<strong>最先进的 (state-of-the-art)</strong> 性能。特别是在与之前最强的闭源模型 `RT-2-X` (55B参数) 的对比中，OpenVLA 以 **7倍少的参数量**，在29个任务上的平均成功率**高出16.5%**。这证明了其模型架构和数据策略的优越性。
3.  **验证了高效微调的可行性：** 论文首次系统地研究了VLA模型的微调问题，证明了OpenVLA可以通过少量数据（10-150个演示）高效适应新任务，并且性能优于<strong>从头训练 (from scratch)</strong> 的模仿学习方法（如 `Diffusion Policy`）。更重要的是，论文验证了使用 **LoRA** 等参数高效微调技术，可以在**消费级GPU**上完成微调，极大地降低了使用门槛。
4.  **提供了完整的开源生态：** 作者不仅发布了模型权重，还提供了一整套工具链，包括可复现的**训练代码**、<strong>微调教程 (Jupyter Notebooks)</strong> 和**部署方案**。这为社区研究和应用VLA模型奠定了坚实的基础。

# 3. 预备知识与相关工作

## 3.1. 基础概念
为了理解 OpenVLA，我们需要先了解构成它的几个关键技术概念。

### 3.1.1. 视觉-语言模型 (Vision-Language Models, VLMs)
VLM 是一种能够同时理解图像和文本的多模态模型。你可以把它想象成一个既能“看”又能“读”的AI。其典型架构如下（这也正是 OpenVLA 所采用的结构）：

1.  <strong>视觉编码器 (Vision Encoder):</strong> 负责“看”。它接收一张图片，并将其转换成一系列数字向量（称为<strong>图像嵌入 (image embeddings)</strong>）。这些向量捕捉了图像中的内容、物体和空间关系。常用的视觉编码器有 `CLIP` 的视觉部分或 `DINOv2`。
2.  <strong>语言模型 (Language Model):</strong> 负责“读”和“说”。它是一个大规模的语言模型（如 `GPT` 或 `Llama`），擅长处理和生成文本。
3.  <strong>投影器 (Projector):</strong> 充当“翻译官”。由于视觉编码器和语言模型最初是独立训练的，它们的“语言”（即向量空间）不通。投影器是一个小型神经网络（通常是多层感知机 `MLP`），它的作用是将图像嵌入“翻译”成语言模型能够理解的格式。

    通过这种方式，VLM 就可以执行“看图说话”（图像描述）、“视觉问答”（VQA）等任务。

### 3.1.2. 视觉-语言-行为模型 (Vision-Language-Action Models, VLAs)
VLA 是 VLM 在机器人领域的直接应用和扩展。其核心思想非常巧妙：**将机器人的“动作”也视为一种特殊的“语言”**。

具体做法是：
1.  将机器人连续的物理动作（如手臂末端的坐标 `(x, y, z)` 和旋转）进行<strong>离散化 (discretization)</strong>，变成一系列整数。例如，将 $x$ 坐标的范围 $[-1, 1]$ 划分为256个“桶”，每个“桶”对应一个从0到255的整数。
2.  将这些代表动作的整数，像处理普通文字一样，映射到语言模型的<strong>词元 (token)</strong> 词汇表中。
3.  训练模型，使其在接收到图像（机器人摄像头看到的画面）和文本指令（如“拿起那个苹果”）后，**像生成一句话一样，自回归地预测出一系列代表动作的词元**。

    这样，VLA 就把机器人控制问题转化为了一个标准的“下一词元预测”问题，从而可以直接利用强大的 VLM 架构和预训练知识。

### 3.1.3. 关键模型组件
*   **Llama 2:** Meta 公司发布的一个强大的开源大语言模型系列。OpenVLA 使用的是其 70亿（7B）参数的版本作为其语言处理和决策的核心。
*   **SigLIP (Sigmoid Loss for Language Image Pre-training):** 谷歌提出的一个强大的视觉-语言预训练模型。它擅长学习图像和文本之间的<strong>语义对齐 (semantic alignment)</strong>，即理解“苹果”这个词和苹果的图片是相关的。在 OpenVLA 中，它主要负责提取图像中的**高级语义特征**。
*   **DINOv2:** Meta AI 研究院提出的一个自监督学习模型，它在没有文本标注的情况下学习视觉特征。DINOv2 的一个显著优点是能学到非常精细的**像素级对应关系和空间结构**。在 OpenVLA 中，它主要负责提取图像中的**底层空间特征**，这对于机器人需要精确定位的任务至关重要。
*   **LoRA (Low-Rank Adaptation):** 一种<strong>参数高效微调 (Parameter-Efficient Fine-Tuning, PEFT)</strong> 技术。在微调大型预训练模型时，如果更新所有参数，计算开销会非常大。LoRA 的思想是：冻结原始模型的绝大部分参数，只在模型的某些层（如 `Attention` 层的权重矩阵）旁边增加两个小型的、“低秩”的矩阵（A和B）。在微调时，只训练这两个小矩阵。这样做可以将在需要训练的参数数量减少99%以上，同时还能达到与完全微调相近的性能。

## 3.2. 前人工作
*   <strong>通用机器人策略 (Generalist Robot Policies):</strong> 此前的研究，如 `RT-1` 和 `Octo`，致力于将在多个任务和机器人平台上收集的大规模数据进行汇集，训练一个“通用”模型。这类模型通常是将预训练的视觉编码器（如 `CLIP`）和语言编码器（如 `T5`）与一个从头开始训练的策略网络（如 `Transformer`）“拼接”起来。
*   <strong>闭源 VLA 模型 (Closed-Source VLAs):</strong> 以谷歌的 `RT-2` 和 `RT-2-X` 为代表，它们是首批成功将大型VLM直接微调用于机器人控制的模型。它们证明了通过在互联网数据上预训练，模型可以获得“涌现”的泛化能力，比如理解“把可乐罐移到泰勒·斯威夫特的照片旁边”这类包含网络概念的指令。然而，这些模型是闭源的，限制了学术研究。
*   **Diffusion Policy:** 一种先进的模仿学习方法，它不使用自回归的词元预测，而是将动作序列建模为一个<strong>扩散过程 (diffusion process)</strong>。它在数据量较少的单任务场景中表现出色，轨迹平滑且精确。

## 3.3. 技术演进
机器人策略学习的技术演进可以大致看作一个不断**提升数据规模和模型通用性**的过程：
1.  **单任务学习：** 为每个特定任务（如开门）单独收集数据并训练一个模型。泛化能力差。
2.  **多任务学习：** 在一个机器人上收集多个任务的数据，训练一个能完成这些任务的模型。
3.  <strong>多机器人/多任务学习 (如 `Octo`):</strong> 汇集来自不同机器人、不同环境的大规模数据集（如 `Open X-Embodiment`），训练一个通用的、可以控制多种机器人的策略模型。
4.  <strong>基于基础模型的 VLA (如 `RT-2`, `OpenVLA`):</strong> 不再仅仅使用机器人数据，而是站在视觉-语言基础模型的“肩膀”上。通过在海量互联网数据上预训练，再在机器人数据上微调，从而获得前所未有的语义理解和泛化能力。

## 3.4. 差异化分析
OpenVLA 与之前工作的核心区别在于：
*   **与 `Octo` 等模型的区别：** `Octo` 采用的是“拼接”式架构，其策略部分是从头学习的。而 OpenVLA 采用<strong>端到端微调 (end-to-end fine-tuning)</strong> 的 VLA 范式，直接调整一个预训练好的 VLM 来输出动作。这种方式能更好地保留和利用基础模型的内部知识。
*   **与 `RT-2-X` 的区别：**
    *   **开源 vs. 闭源：** 这是最本质的区别。OpenVLA 是第一个高性能的开源 VLA。
    *   **模型架构：** OpenVLA 使用了 `SigLIP` + `DINOv2` 的**双视觉编码器**融合方案，而 `RT-2-X` 基于 `PaLI-X`，其视觉编码器细节未知，但很可能是单一的。OpenVLA 的融合设计旨在同时捕获语义和空间信息。
    *   **效率和性能：** OpenVLA 以小得多的模型尺寸（7B vs 55B）取得了更优的性能。
    *   **研究重点：** `RT-2-X` 的论文主要关注“开箱即用”的泛化能力，而 OpenVLA 不仅关注这一点，还系统地研究了**高效微调**这一实际应用中的关键问题。
*   **与 `Diffusion Policy` 的区别：** `Diffusion Policy` 是一个强大的**从头学习**方法，适用于数据较少的场景。而 OpenVLA 是一个基于**预训练+微调**范式的大模型，它在需要复杂语言理解和多任务泛化的场景中更具优势。

# 4. 方法论

本部分将详细拆解 OpenVLA 的模型架构、训练流程和关键设计决策。

## 4.1. 方法原理
OpenVLA 的核心思想是**将机器人控制问题转化为一个视觉条件下的序列生成问题**。它继承了现代<strong>视觉-语言模型 (VLM)</strong> 的架构，并通过微调使其能够输出代表机器人动作的离散词元。其直觉是，一个在海量图文数据上预训练过的模型已经具备了强大的视觉理解和推理能力，我们只需要通过相对少量的机器人数据，教会它如何将这些理解“映射”到物理动作上。

## 4.2. 核心方法详解 (逐层深入)

### 4.2.1. 模型架构
OpenVLA 的架构由三个主要部分组成，如下图（原文 Figure 2）所示：

![Figure 2: OpenVLA model architecture. Given an image observation and a language instruction, the model predicts 7-dimensional robot control actions. The architecture consists of three key components:(1 a vision ener that concatenates Dino V2 \[25\] and SigLIP \[79\] features, () a projector that maps visual featuresto the language embeding space, and (3) the LLM backbone, a Llama 2 7B-parameter large language model \[10\].](images/2.jpg)
*该图像是OpenVLA模型架构示意图。该架构根据输入的图像和语言指令，预测7维机器人控制动作。主要由三个关键组件组成：视觉编码器DinoV2与SigLIP特征的连接、映射视觉特征至语言嵌入空间的多层感知器项目器，以及7B参数的Llama 2大语言模型。*

1.  <strong>视觉编码器 (Vision Encoder):</strong>
    *   **输入:** 单张 `224x224` 像素的图像（机器人视角）。
    *   **结构:** 这是一个创新的**双路径融合**结构。输入图像的图像块 (patches) 会被**同时**送入两个不同的、预训练好且在微调过程中参数**可更新**的视觉编码器：
        *   **SigLIP-ViT:** 一个强大的视觉 Transformer 模型，负责提取**高级语义特征**。例如，它能识别出图像中有一个“苹果”和一个“碗”。
        *   **DINOv2-ViT:** 另一个视觉 Transformer，它通过自监督学习，擅长捕捉**底层空间细节和物体结构**。例如，它能精确地定位苹果的轮廓和碗的边缘。
    *   **输出:** 两个编码器输出的特征向量在通道维度上被<strong>拼接 (concatenate)</strong> 起来，形成一个更丰富、结合了语义和空间信息的视觉表征。

2.  <strong>投影器 (Projector):</strong>
    *   **结构:** 一个简单的2层 MLP（多层感知机）网络。
    *   **功能:** 它的作用是将视觉编码器输出的融合特征向量，**投影**到 Llama 2 语言模型的词嵌入空间中。这一步是连接“视觉”和“语言”两个模态的关键桥梁。

3.  <strong>大语言模型主干 (LLM Backbone):</strong>
    *   **结构:** **Llama 2 7B**，一个拥有70亿参数的自回归 Transformer 模型。
    *   **输入:** 模型的输入序列由两部分组成：
        1.  经过投影器处理后的<strong>图像词元 (image tokens)</strong>。
        2.  描述任务的<strong>自然语言指令词元 (language instruction tokens)</strong>，例如 "put the apple in the bowl"。
    *   **功能:** LLM 接收这个混合序列，并基于这些信息，自回归地（一个接一个地）预测后续的词元。在 OpenVLA 中，这些被预测的词元就是代表机器人动作的<strong>动作词元 (action tokens)</strong>。

### 4.2.2. 训练流程：从动作到词元
将物理世界的连续动作转化为 LLM 可以处理的离散词元是 VLA 模型的关键步骤。

<strong>Step 1: 动作离散化 (Action Discretization)</strong>
机器人的动作通常是一个多维的连续向量，例如一个7维向量代表手臂末端的三维平移、三维旋转（用轴角表示）和一个夹爪的开合状态。OpenVLA 的处理方式如下：
*   对于一个 $D$ 维的连续动作向量 $a_t \in \mathbb{R}^D$。
*   首先，针对训练数据中每一维度的动作，计算其<strong>第1个百分位数 ($1^{st}$ quantile)</strong> 和<strong>第99个百分位数 ($99^{th}$ quantile)</strong>。这样做是为了忽略极端异常值，使离散化区间更稳定。
*   然后，将这个 `[1%, 99%]` 的区间**均匀地**划分为 256 个<strong>桶 (bins)</strong>。
*   每个连续的动作值根据其落入的桶，被映射到一个 `[0, 255]` 范围内的整数。
*   最终，一个 $D$ 维的连续动作 $a_t$ 就被转换成了一个由 $D$ 个整数组成的序列 $a_t^{tok} = (a_{t,1}^{tok}, a_{t,2}^{tok}, \dots, a_{t,D}^{tok})$。

<strong>Step 2: 动作词元化 (Action Tokenization)</strong>
Llama 2 的分词器有自己固定的词汇表，并且只预留了少量（100个）特殊词元用于微调。这不足以容纳256个新的动作词元。OpenVLA 采取了一个简单而有效的方法：
*   直接<strong>覆盖 (overwrite)</strong> Llama 2 词汇表中**最不常用**的256个词元。在 Llama 的词汇表中，这些通常是最后256个词元。
*   将整数 `0` 到 `255` 分别与这256个被覆盖的词元一一对应。

<strong>Step 3: 训练目标 (Training Objective)</strong>
经过上述处理后，机器人模仿学习问题就完全转化成了一个标准的语言模型训练问题。
*   **输入:** 图像 $I_t$ 和语言指令 $L$。
*   **目标输出:** 动作词元序列 $a_t^{tok} = (a_{t,1}^{tok}, \dots, a_{t,D}^{tok})$。
*   **损失函数:** 模型采用标准的<strong>下一词元预测 (next-token prediction)</strong> 目标，并使用<strong>交叉熵损失 (cross-entropy loss)</strong>。关键是，**损失只在预测动作词元时计算**，模型在预测其他文本词元（如指令复述）时的损失被忽略。

    对于一个在时间步 $t$ 的 $D$ 维动作，其损失函数可以表示为：
$$
\mathcal{L}_t = - \sum_{d=1}^{D} \log P(a_{t,d}^{tok} | I_t, L, a_{t,1}^{tok}, \dots, a_{t,d-1}^{tok}; \theta)
$$
**符号解释:**
*   $\mathcal{L}_t$: 在时间步 $t$ 的损失。
*   $a_{t,d}^{tok}$: 动作向量第 $d$ 维对应的离散整数词元。
*   $I_t$: 当前的图像观测。
*   $L$: 任务的自然语言指令。
*   $P(\cdot | \cdot; \theta)$: 由模型（参数为 $\theta$）给出的条件概率，即在给定图像、指令和已经预测出的前 `d-1` 个动作维度词元的条件下，预测第 $d$ 个动作维度词元的概率。
*   $\log$: 自然对数。

    总损失是整个训练数据集中所有轨迹、所有时间步损失的总和。模型通过梯度下降来最小化这个总损失。

### 4.2.3. 训练数据
*   **来源:** 论文使用了大规模的 **Open X-Embodiment (OpenX)** 数据集。这是一个社区共同努力汇集而成的数据集，包含了来自超过70个不同机器人学习数据集、超过200万条机器人轨迹。
*   <strong>数据筛选和混合 (Curation and Mixing):</strong> 为了保证训练质量和效率，作者进行了精心的数据处理：
    1.  **筛选:** 只保留了包含至少一个第三方视角摄像头、并且是单臂末端控制的操作任务。
    2.  **混合权重:** 借鉴了 `Octo` 模型的数据混合策略，对不同数据集分配不同的采样权重。多样性高、任务丰富的数据集（如 `BridgeData V2`）被赋予更高权重，而数据质量较低或任务单一的数据集则被降权或移除。
    3.  **增量数据:** 实验性地加入了 `Octo` 发布后新增到 OpenX 的数据集，如 `DROID`，但发现模型拟合 `DROID` 的多样性有困难，因此在训练后期将其移除，以保证最终模型的质量。

        最终，OpenVLA 在一个包含 **97万条轨迹** 的混合数据集上进行了训练。

### 4.2.4. 关键设计决策
在最终训练 OpenVLA 之前，研究团队在较小规模的数据集 (`BridgeData V2`) 上进行了一系列探索性实验，得出了几个关键结论：
*   **VLM 主干网络选择:** 对比了 `IDEFICS-1`, `LLaVA`, 和 `Prismatic` 三种VLM。发现 `Prismatic` 因其融合了 `SigLIP` 和 `DINOv2` 的视觉主干，在需要精确空间推理和语言关联的多物体场景中表现最好，因此被选为最终主干。
*   **图像分辨率:** 对比了 `224x224` 和 `384x384` 两种分辨率。发现更高分辨率并未带来机器人控制性能的提升，但训练时间却增加了3倍。因此选择了计算效率更高的 `224x224`。
*   **微调视觉编码器:** 与VLM训练中通常冻结视觉编码器的做法相反，论文发现**在VLA训练中微调视觉编码器至关重要**。他们推测，预训练的视觉特征虽然强大，但可能缺乏机器人精确操作所需的细粒度空间信息，需要通过微调来适应。
*   <strong>训练周期 (Epochs):</strong> 与LLM训练通常只过一遍数据不同，VLA训练需要**多次迭代**数据集。最终模型在训练集上迭代了 **27个周期**，直到训练集上的动作词元预测准确率超过95%。
*   **学习率:** 实验发现，使用与VLM预训练相同的固定学习率 `2e-5` 效果最好，并且学习率预热 (warmup) 并无益处。

# 5. 实验设置

## 5.1. 数据集
实验分为两大部分：<strong>直接评估 (out-of-the-box evaluation)</strong> 和 <strong>微调适应 (fine-tuning adaptation)</strong>。

### 5.1.1. 直接评估数据集
在不进行任何额外微调的情况下，直接评估 OpenVLA 在两个机器人平台上的性能。
*   **WidowX (BridgeData V2):** 一个桌面级的 6-DoF 机械臂，在一个厨房水槽环境中执行任务。评估任务被精心设计为包含各种<strong>分布外 (Out-of-Distribution, OOD)</strong> 挑战。下图（原文 Figure 7）展示了部分评估任务。

    ![Figure 7: BridgeData V2 WidowX robot evaluation tasks. We evaluate every generalist robot policy on 4 types out-of-distribution (OoD) generalization tasks:visual, motion, physical, and semantic (as defined in Secn 5.1. Every pair mages hows the start state and a example ed state ater the robot completes the task. We also rigorously assess language rounding in the 3 tasks shown in the bottom 3 rows, by changing the prompt while fxing the initial state and testing whether the policy can approach the correct target object.](images/7.jpg)
    *该图像是插图，展示了在不同类型的超出分布（OoD）泛化任务中评估机器人政策的示例，包括视觉生成、运动生成、物理生成、语义生成和语言嵌入。每组图像展示了任务的起始状态和机器人完成后的状态，重点评估语言理解能力并考虑提示变化。*

*   **Google Robot:** 一个带轮子的移动操作平台，曾在 `RT-1` 和 `RT-2` 的研究中使用。任务同样分为分布内和分布外。下图（原文 Figure 9）展示了其评估任务。

    ![Figure 9: Google robot evaluation tasks. We evaluate every generalist robot policy on in-distribution tasks and out-of-distribution (OOD) generalization tasks.OOD tasks involve unseen backgrounds, target objects, intutions/objec relations, n semantconcepts . photos rom he Interet that do ot appe ro action data).](images/9.jpg)
    *该图像是图表，展示了Google机器人在分布内和分布外任务的评估情况。上半部分为分布内任务，包括拾取和移动目标物体的场景；下半部分为分布外任务，涉及未见背景和未见目标物体的任务。图中展示了多个机器人操作示例。*

### 5.1.2. 微调适应数据集
评估 OpenVLA 在少量新数据上的学习能力。
*   **Franka-Tabletop:** 一个固定的 Franka Emika Panda 7-DoF 机械臂，在桌面上执行任务。数据集包含10-150个演示，任务从简单的单指令（如“把胡萝卜放进碗里”）到复杂的多指令（如“用毛巾盖住<指定物体>”）。下图（原文 Figure 10）展示了这些任务。

    ![Figure 10: Franka-Tabletop fine-tuning tasks.Franka-Tabletop tasks used in the data-effcient adaptation eperiments in Section 5. and described in detail in Fig.10 are depicted above.The first three of six tasks, so the to tree ows,yvolvea glesttn, whi the asthree tasks theott e rows involve multipleobjects and instructions (the instructions speciy the target object or targe location). The rst colu shows sample initl states matching the trainig data distribution, while the second clum showut-distribui ()initl states . nsebackroun, targe bject, distractors anje positns/orientations). Every polic Section .s evaluated with 1012 rollouts on -distribution tasks and 56 rollouts on OOD tasks.](images/10.jpg)
    *该图像是Franka-Tabletop任务的示意图，展示了八个任务的训练和测试状态。其中，左侧的任务为训练数据分布中的任务，右侧为超出分布的任务。每个任务相关的指令也在图中列出，显示了不同的操作对象和目标.*

*   **Franka-DROID:** 另一个 Franka 臂，安装在可移动的桌子上，来自 `DROID` 数据集。评估了“擦桌子”任务。
*   **LIBERO (Simulation):** 一个在**仿真**环境中进行的基准测试，用于评估模型在空间关系、不同物体、不同目标和长时程任务上的学习能力。

## 5.2. 评估指标
论文主要使用以下指标来衡量模型性能：

### 5.2.1. 成功率 (Success Rate, SR)
1.  <strong>概念定义 (Conceptual Definition):</strong> 成功率是评估机器人策略性能最直接、最常用的指标。它衡量了在给定任务中，机器人成功完成任务的试验次数占总试验次数的百分比。在某些复杂的任务中，可能会定义“部分成功”（例如，正确抓取了物体但未能放置到目标位置），并给予部分分数（如0.5分）。
2.  <strong>数学公式 (Mathematical Formula):</strong>
    $$
    \text{SR} = \frac{\sum_{i=1}^{N} \text{score}_i}{N} \times 100\%
    $$
3.  <strong>符号解释 (Symbol Explanation):</strong>
    *   $N$: 总的<strong>推演轨迹 (rollouts)</strong> 或试验次数。
    *   $\text{score}_i$: 第 $i$ 次试验的得分。对于只有成功/失败二元结果的任务，成功为1，失败为0。对于有部分成功的任务，得分可以是 `[0, 1]` 之间的值（如0, 0.5, 1）。

### 5.2.2. 标准误差 (Standard Error, StdErr)
1.  <strong>概念定义 (Conceptual Definition):</strong> 标准误差衡量的是样本均值（这里是计算出的成功率）的精确度。一个较小的标准误差意味着如果我们多次重复整个实验（例如，再进行100次试验），每次得到的成功率都会非常接近。它反映了实验结果的统计稳定性。
2.  <strong>数学公式 (Mathematical Formula):</strong>
    $$
    \text{StdErr} = \frac{s}{\sqrt{N}}
    $$
3.  <strong>符号解释 (Symbol Explanation):</strong>
    *   $N$: 总试验次数。
    *   $s$: 试验得分的**样本标准差**。对于得分只能是0或1的伯努利试验，样本标准差可以由成功率 $p = \text{SR}/100$ 估算：`s = \sqrt{p(1-p)}`。

## 5.3. 对比基线
论文将 OpenVLA 与以下代表性的基线模型进行了比较：
*   **RT-1-X (35M):** 一个在 OpenX 数据集上训练的 `Transformer` 策略模型，规模较小。代表了从头训练的通用策略。
*   **Octo (93M):** 目前**开源**的最先进的通用机器人策略模型。它采用“拼接”式架构，同样在 OpenX 数据集上训练。
*   **RT-2-X (55B):** 谷歌的**闭源** VLA 模型，是之前性能最强的模型。它代表了基于超大规模基础模型的 VLA 方法。
*   **Diffusion Policy:** 一个强大的**从头模仿学习**方法，不依赖大规模预训练。它代表了在数据量较少时，专门为单任务设计的先进方法。

# 6. 实验结果与分析

## 6.1. 核心结果分析

### 6.1.1. 直接评估：通用操作能力
实验旨在回答：OpenVLA “开箱即用”的能力如何？

<strong>BridgeData V2 (WidowX) 结果:</strong>
下图（原文 Figure 3）和下表（原文 Table 4）展示了在 WidowX 机器人上的详细结果。

![Figure 3: BridgeData V2 WidowX robot evaluation tasks and results. We evaluate OpenVLA and prior stateothernelis obot polic cpensivsuitaskcoverievealxeneln as well as tasks that specifically assess language conditioning ability. OpenVLA achieves highest overall perormance and even outperforms closed-source model RT-2-X in a categories except for semantic generalization. Average success rates $\\pm$ StdErr are computed across 170 total rollouts per approach. See Table 4 for detailed results.](images/3.jpg)
*该图像是图表，展示了OpenVLA与其他模型在各种评估任务中的成功率对比。图中包含多个模型（RT-1-X、Octo、RT-2-X以及OpenVLA）的平均成功率及其在视觉、运动、物理、语义泛化和语言理解等方面的表现。OpenVLA在所有类别中表现最佳，特别是在语言理解任务中取得了90.0%的成功率，而RT-2-X在相同任务中为85.0%。数据点呈现成功率的平均值及标准误差。*

<table>
<thead>
<tr>
<th rowspan="2">Category</th>
<th rowspan="2">Task</th>
<th rowspan="2"># Trials</th>
<th rowspan="2">RT-1-X # Successes</th>
<th rowspan="2">Octo # Successes</th>
<th rowspan="2">RT-2-X # Successes</th>
<th rowspan="2">OpenVLA (ours) # Successes</th>
</tr>
<tr></tr>
</thead>
<tbody>
<tr>
<td>Visual gen</td>
<td>Put Eggplant into Pot (Easy Version)</td>
<td>10</td>
<td>1</td>
<td>5</td>
<td>7</td>
<td>10</td>
</tr>
<tr>
<td>Visual gen</td>
<td>Put Eggplant into Pot</td>
<td>10</td>
<td>0</td>
<td>1</td>
<td>5</td>
<td>10</td>
</tr>
<tr>
<td>Visual gen</td>
<td>Put Cup from Counter into Sink</td>
<td>10</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>7</td>
</tr>
<tr>
<td>Visual gen</td>
<td>Put Eggplant into Pot (w/ Clutter)</td>
<td>10</td>
<td>1</td>
<td>3.5</td>
<td>6</td>
<td>7.5</td>
</tr>
<tr>
<td>Visual gen</td>
<td>Put Yellow Corn on Pink Plate</td>
<td>10</td>
<td>1</td>
<td>4</td>
<td>8</td>
<td>9</td>
</tr>
<tr>
<td>Motion gen</td>
<td>Lift Eggplant</td>
<td>10</td>
<td>3</td>
<td>0.5</td>
<td>6.5</td>
<td>7.5</td>
</tr>
<tr>
<td>Motion gen</td>
<td>Put Carrot on Plate (w/ Height Change)</td>
<td>10</td>
<td>2</td>
<td>1</td>
<td>4.5</td>
<td>4.5</td>
</tr>
<tr>
<td>Physical gen</td>
<td>Put Carrot on Plate</td>
<td>10</td>
<td>1</td>
<td>0</td>
<td>1</td>
<td>8</td>
</tr>
<tr>
<td>Physical gen</td>
<td>Flip Pot Upright</td>
<td>10</td>
<td>2</td>
<td>6</td>
<td>5</td>
<td>8</td>
</tr>
<tr>
<td>Physical gen</td>
<td>Lift AAA Battery</td>
<td>10</td>
<td>0</td>
<td>0</td>
<td>2</td>
<td>7</td>
</tr>
<tr>
<td>Semantic gen</td>
<td>Move Skull into Drying Rack</td>
<td>10</td>
<td>1</td>
<td>0</td>
<td>5</td>
<td>5</td>
</tr>
<tr>
<td>Semantic gen</td>
<td>Lift White Tape</td>
<td>10</td>
<td>3</td>
<td>0</td>
<td>0</td>
<td>1</td>
</tr>
<tr>
<td>Semantic gen</td>
<td>Take Purple Grapes out of Pot</td>
<td>10</td>
<td>0.5</td>
<td>0</td>
<td>5</td>
<td>4</td>
</tr>
<tr>
<td>Semantic gen</td>
<td>Stack Blue Cup on Pink Cup</td>
<td>10</td>
<td>2.5</td>
<td>4</td>
<td>5.5</td>
<td>4.5</td>
</tr>
<tr>
<td rowspan="2">Language grounding</td>
<td>Put {Eggplant, Red Bottle} into Pot</td>
<td>10</td>
<td>1.5</td>
<td>2.5</td>
<td>8.5</td>
<td>8.5</td>
</tr>
<tr>
<td>Lift {Cheese, Red Chili Pepper}</td>
<td>10</td>
<td>5</td>
<td>5.5</td>
<td>8.5</td>
<td>7.5</td>
</tr>
<tr>
<td>Language grounding</td>
<td>Put {Blue Cup, Pink Cup} on Plate</td>
<td>10</td>
<td>5</td>
<td>2.5</td>
<td>8.5</td>
<td>9.5</td>
</tr>
<tr>
<td colspan="2"></td>
<td>Mean Success Rate</td>
<td>18.5±2.7%</td>
<td>20.0±2.6%</td>
<td>50.6±3.5%</td>
<td>**70.6±3.2%**</td>
</tr>
</tbody>
</table>

**分析:**
*   **OpenVLA 表现最佳：** OpenVLA 的平均成功率达到了 **70.6%**，显著优于所有其他模型。
*   **超越闭源SOTA：** OpenVLA 比之前的最先进模型 `RT-2-X` (50.6%) 的绝对成功率高出 **20%**。考虑到 OpenVLA 的参数量（7B）远小于 `RT-2-X`（55B），这一结果尤其令人印象深刻。
*   **VLA 范式的优越性：** 两个 VLA 模型（OpenVLA 和 RT-2-X）的性能远超非 VLA 的通用策略模型 `RT-1-X` (18.5%) 和 `Octo` (20.0%)，证明了利用大型 VLM 预训练知识的巨大优势。
*   **强泛化能力：** OpenVLA 在各类泛化任务中（视觉、运动、物理）均表现出色，尤其是在需要精确操作的小物体任务（如 `Lift AAA Battery`）和物理属性变化的任务（如 `Put Carrot on Plate`）上，优势明显。

**Google Robot 结果:**
下图（原文 Figure 4）和下表（原文 Table 6）展示了在 Google 移动机器人上的结果。

![Figure 4: Google robot evaluation results. We evaluate generalist robot policies on in-distribution and out-ofdistribution (OOD) tasks on the mobile manipulator used in RT-1 and RT-2 evaluations \[2, 7\]. We find that OpenVLA and RT-2-X attain comparable performance and significantly outperform RT-1-X and Octo overall. Average success rates $\\pm$ StdErr are computed across 60 total rollouts per approach. See Table 6 for detailed results.](images/4.jpg)
*该图像是一个柱状图，展示了不同机器人政策在不同任务下的成功率。图中比较了 OpenVLA 与 RT-1-X、RT-2-X 和 Octo 模型的表现，显示 OpenVLA 在平均、训练数据内 (In-Distribution) 和超出分布 (OOD) 任务上的显著优势。成功率以百分比表示，误差条表示标准误差。*

<table>
<thead>
<tr>
<th>Category</th>
<th>Task</th>
<th># Trials</th>
<th>RT-1-X # Successes</th>
<th>Octo # Successes</th>
<th>RT-2-X # Successes</th>
<th>OpenVLA (ours) # Successes</th>
</tr>
</thead>
<tbody>
<tr>
<td>In-distribution</td>
<td>Pick Coke Can</td>
<td>5</td>
<td>5</td>
<td>1</td>
<td>5</td>
<td>5</td>
</tr>
<tr>
<td>In-distribution</td>
<td>Move Apple near Green Can</td>
<td>5</td>
<td>3</td>
<td>3</td>
<td>3</td>
<td>5</td>
</tr>
<tr>
<td>In-distribution</td>
<td>Move Blue Chip Bag near Apple</td>
<td>5</td>
<td>0</td>
<td>3</td>
<td>4</td>
<td>5</td>
</tr>
<tr>
<td>In-distribution</td>
<td>Place Coke Can Upright</td>
<td>5</td>
<td>0</td>
<td>0</td>
<td>4</td>
<td>4</td>
</tr>
<tr>
<td>In-distribution</td>
<td>Open Middle Drawer</td>
<td>5</td>
<td>0</td>
<td>4</td>
<td>2</td>
<td>3</td>
</tr>
<tr>
<td>OOD</td>
<td>Move Orange near Brown Chip Bag</td>
<td>5</td>
<td>1</td>
<td>2</td>
<td>5</td>
<td>5</td>
</tr>
<tr>
<td>OOD</td>
<td>Pick Pepsi Can</td>
<td>5</td>
<td>3</td>
<td>0</td>
<td>5</td>
<td>4</td>
</tr>
<tr>
<td>OOD</td>
<td>Pick Banana</td>
<td>5</td>
<td>5</td>
<td>3</td>
<td>5</td>
<td>5</td>
</tr>
<tr>
<td>OOD</td>
<td>Pick Green Cup</td>
<td>5</td>
<td>1</td>
<td>0</td>
<td>5</td>
<td>5</td>
</tr>
<tr>
<td>OOD</td>
<td>Place Apple on Plate</td>
<td>5</td>
<td>0</td>
<td>0</td>
<td>4</td>
<td>4</td>
</tr>
<tr>
<td>OOD</td>
<td>Place Banana in Pan</td>
<td>5</td>
<td>0</td>
<td>0</td>
<td>2</td>
<td>4</td>
</tr>
<tr>
<td>OOD</td>
<td>Move Coke Can near Taylor Swift</td>
<td>5</td>
<td>2</td>
<td>0</td>
<td>3</td>
<td>2</td>
</tr>
<tr>
<td colspan="2"></td>
<td>Mean Success Rate</td>
<td>33.3±6.1%</td>
<td>26.7±5.8%</td>
<td>**78.3±5.4%**</td>
<td>**85.0±4.6%**</td>
</tr>
</tbody>
</table>

**分析:**
*   在此平台上，OpenVLA (85.0%) 和 RT-2-X (78.3%) 的性能表现相当，并且两者都再次大幅领先于 RT-1-X 和 Octo。这进一步巩固了 VLA 范式的优势。

### 6.1.2. 微调适应：在新任务上的学习效率
实验旨在回答：OpenVLA 能否用少量数据高效学习新任务？

下图（原文 Figure 5）和下表（原文 Table 7）展示了在 Franka 机器人上的微调结果，比较了从头训练的 `Diffusion Policy`、微调的 `Octo` 和微调的 `OpenVLA`。

![Figure 5: Adapting to new robot setups. We evaluate the state-of-the-art Diffusion Policy trained from scratch on seven Franka Emika Panda tasks (10150 demonstrations each), as well as generalist robot policies Octo and OpenVLA fine-tuned on the same data. Diffusion Policy exhibits strong performance on narrow singe-instruction tasks, while Octo and OpenVLA perform betteron diverse ne-tuning tasks involving multiple instructions and distractor objects. Overall, OpenVLA achieves highest aggregate performance across both usthaiul o skve $\\pm$ StdErr are computed across 129 rollouts per approach (99 for Franka-Tabletop tasks and 30 for Franka-DROID tasks). See Table 7 for detailed results.](images/5.jpg)
*该图像是一个柱状图，展示了 OpenVLA 与其他机器人策略在不同任务中的成功率。横轴为任务类型，纵轴表示成功率（%）。OpenVLA 在多种指令任务上表现优异，尤其在视觉鲁棒性方面胜出。数据来源于 129 次实验，具体结果可见于表 7。*

<table>
<thead>
<tr>
<th colspan="2"></th>
<th># trials</th>
<th>Diffusion Policy</th>
<th>Diffusion Policy (matched)</th>
<th>Octo</th>
<th>OpenVLA (scratch)</th>
<th>OpenVLA (ours)</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="13">Franka-Tabletop (5Hz)</td>
<td>"Put Carrot in Bowl" (in-distribution)</td>
<td>10</td>
<td>90.0%</td>
<td>80.0%</td>
<td>40.0%</td>
<td>70.0%</td>
<td>70.0%</td>
</tr>
<tr>
<td>"Put Carrot in Bowl" (OOD)</td>
<td>5</td>
<td>20.0%</td>
<td>0.0%</td>
<td>20.0%</td>
<td>0.0%</td>
<td>40.0%</td>
</tr>
<tr>
<td>"Pour Corn into Pot" (in-distribution)</td>
<td>10</td>
<td>100.0%</td>
<td>90.0%</td>
<td>0.0%</td>
<td>10.0%</td>
<td>50.0%</td>
</tr>
<tr>
<td>"Pour Corn into Pot" (OOD)</td>
<td>5</td>
<td>80.0%</td>
<td>60.0%</td>
<td>0.0%</td>
<td>20.0%</td>
<td>60.0%</td>
</tr>
<tr>
<td>"Flip Pot Upright" (in-distribution)</td>
<td>10</td>
<td>100.0%</td>
<td>85.0%</td>
<td>40.0%</td>
<td>85.0%</td>
<td>100.0%</td>
</tr>
<tr>
<td>"Flip Pot Upright" (OOD)</td>
<td>5</td>
<td>50.0%</td>
<td>20.0%</td>
<td>0.0%</td>
<td>40.0%</td>
<td>80.0%</td>
</tr>
<tr>
<td>"Move &lt;object&gt; onto Plate" (in-distribution)</td>
<td>12</td>
<td>25.0%</td>
<td>25.0%</td>
<td>41.7%</td>
<td>8.3%</td>
<td>75.0%</td>
</tr>
<tr>
<td>"Move &lt;object&gt; onto Plate" (OOD)</td>
<td>6</td>
<td>8.3%</td>
<td>33.3%</td>
<td>8.3%</td>
<td>33.3%</td>
<td>58.3%</td>
</tr>
<tr>
<td>"Knock &lt;object&gt; Over" (in-distribution)</td>
<td>12</td>
<td>33.3%</td>
<td>25.0%</td>
<td>83.3%</td>
<td>75.0%</td>
<td>75.0%</td>
</tr>
<tr>
<td>"Knock &lt;object&gt; Over" (OOD)</td>
<td>6</td>
<td>16.7%</td>
<td>16.7%</td>
<td>33.3%</td>
<td>58.3%</td>
<td>50.0%</td>
</tr>
<tr>
<td>"Cover &lt;object&gt; with Towel" (in-distribution)</td>
<td>12</td>
<td>16.7%</td>
<td>20.8%</td>
<td>91.7%</td>
<td>41.7%</td>
<td>83.3%</td>
</tr>
<tr>
<td>"Cover &lt;object&gt; with Towel" (OOD)</td>
<td>6</td>
<td>16.7%</td>
<td>33.3%</td>
<td>91.7%</td>
<td>50.0%</td>
<td>50.0%</td>
</tr>
<tr>
<td>Average</td>
<td></td>
<td>48.5±4.9%</td>
<td>43.4±4.7%</td>
<td>43.4±4.4%</td>
<td>43.4±4.6%</td>
<td>**67.2±4.0%**</td>
</tr>
<tr>
<td rowspan="3">Franka-DROID (15Hz)</td>
<td>"Wipe Table" (in-distribution)</td>
<td>18</td>
<td>50.0%</td>
<td>27.8%</td>
<td>52.8%</td>
<td>25.0%</td>
<td>55.6%</td>
</tr>
<tr>
<td>"Wipe Table" + Distractors (OOD)</td>
<td>12</td>
<td>12.5%</td>
<td>25.0%</td>
<td>16.7%</td>
<td>16.7%</td>
<td>62.5%</td>
</tr>
<tr>
<td>Average</td>
<td></td>
<td>35.0±8.0%</td>
<td>26.7±7.5%</td>
<td>38.3±8.5%</td>
<td>21.7±6.6%</td>
<td>**58.3±7.2%**</td>
</tr>
</tbody>
</table>

**分析:**
*   **OpenVLA 综合表现最佳：** 在 Franka-Tabletop 和 Franka-DROID 两个环境的平均成功率上，微调后的 OpenVLA (67.2% 和 58.3%) 均排名第一。
*   **预训练的价值：**
    *   在**简单的单指令任务**（如 "Pour Corn into Pot"）中，从头训练的 `Diffusion Policy` 表现非常出色，轨迹更平滑。
    *   然而，在**复杂的、需要语言理解的多指令任务**（如 "Move <object> onto Plate"）中，经过大规模机器人数据预训练的 `OpenVLA` 和 `Octo` 表现明显更好。
    *   `OpenVLA (scratch)` 是一个消融实验，它跳过了 OpenX 预训练，直接在目标任务上微调 VLM。其性能远低于经过 OpenX 预训练的 OpenVLA，这强有力地证明了**大规模、多样化的机器人预训练数据**对于提升下游任务微调性能至关重要。
*   **结论：** OpenVLA 不仅是一个强大的通用模型，也是一个<strong>出色的预训练起点 (initialization)</strong>，能高效适应新任务，尤其在任务具有多样性和语言复杂性时优势更为明显。

## 6.2. 消融实验/参数分析

### 6.2.1. 参数高效微调 (LoRA)
实验旨在回答：能否用更少的计算资源微调 OpenVLA？

以下是原文 Table 1 的结果：

<table>
<thead>
<tr>
<th>Strategy</th>
<th>Success Rate</th>
<th>Train Params (× 10<sup>6</sup>)</th>
<th>VRAM (batch 16)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Full FT</td>
<td>69.7 ± 7.2 %</td>
<td>7,188.1</td>
<td>163.3 GB*</td>
</tr>
<tr>
<td>Last layer only</td>
<td>30.3 ± 6.1 %</td>
<td>465.1</td>
<td>51.4 GB</td>
</tr>
<tr>
<td>Frozen vision</td>
<td>47.0 ± 6.9 %</td>
<td>6,760.4</td>
<td>156.2 GB*</td>
</tr>
<tr>
<td>Sandwich</td>
<td>62.1 ± 7.9 %</td>
<td>914.2</td>
<td>64.0 GB</td>
</tr>
<tr>
<td rowspan="2">LoRA, rank=32</td>
<td>68.2 ± 7.5%</td>
<td>97.6</td>
<td>59.7 GB</td>
</tr>
<tr>
<td>rank=64</td>
<td>68.2 ± 7.8%</td>
<td>195.2</td>
<td>60.5 GB</td>
</tr>
</tbody>
</table>

**分析:**
*   **LoRA 效果显著：** `LoRA` (rank=32) 微调的成功率 (68.2%) 与<strong>完全微调 (Full FT)</strong> (69.7%) **几乎没有差别**。
*   **资源消耗大幅降低：** LoRA 只需训练 **1.4%** 的参数 (97.6M vs 7188.1M)，所需的显存 (VRAM) 也从 163.3 GB 降至 59.7 GB，使得在单张 A100 或 RTX 4090 等消费级/专业级 GPU 上进行微调成为可能。
*   **其他策略的局限性：** 只微调最后一层 (`Last layer only`) 或冻结视觉编码器 (`Frozen vision`) 都会导致性能严重下降，再次印证了**调整视觉特征**对于适应新场景的重要性。

### 6.2.2. 高效推理 (Quantization)
实验旨在回答：能否用更少的显存部署 OpenVLA，而不牺牲性能？

以下是原文 Table 2 的结果：

| Precision | Bridge Success | VRAM |
| :--- | :--- | :--- |
| bfloat16 | 71.3 ± 4.8% | 16.8 GB |
| int8 | 58.1 ± 5.1% | 10.2 GB |
| int4 | 71.9 ± 4.7% | 7.0 GB |

**分析:**
*   **4-bit 量化效果出色：** 使用 `int4` (4-bit) 量化，模型的性能 (71.9%) 与默认的 `bfloat16` 半精度 (71.3%) **完全相当**。
*   **显存占用减半：** `int4` 量化将推理所需的显存从 16.8 GB 大幅降低到 **7.0 GB**，这意味着 OpenVLA 可以在更广泛的消费级 GPU（如显存为 8GB 或 12GB 的型号）上运行。
*   **8-bit 量化问题：** `int8` 量化性能下降，论文分析这主要是由于 `int8` 操作引入的额外开销导致**推理速度变慢**，从而影响了需要较快控制频率的机器人任务。而 `int4` 因为内存传输的减少，速度反而更快，避免了此问题。

# 7. 总结与思考

## 7.1. 结论总结
这篇论文成功地推出了 **OpenVLA**，一个高性能、高效率且完全开源的70亿参数视觉-语言-行为模型。它在机器人通用操作任务上树立了新的标杆，以小得多的模型尺寸超越了之前的闭源 SOTA 模型。

论文的主要贡献和发现可以总结为：
1.  **架构创新有效：** 融合 `DINOv2` 和 `SigLIP` 的双视觉编码器架构，结合 `Llama 2` 主干，被证明是构建高性能 VLA 的有效途径。
2.  **开源打破壁垒：** OpenVLA 作为第一个高性能的开源 VLA，连同其代码库和教程，极大地降低了社区研究和应用 VLA 的门槛，对推动领域发展具有里程碑意义。
3.  **高效适应是可行的：** 论文系统地验证了 VLA 模型可以通过少量数据进行高效微调，并证明了 `LoRA` 和 `4-bit` 量化等技术是实现低成本微调和部署的关键，使得在消费级硬件上运行大型机器人模型成为现实。
4.  **预训练价值巨大：** 实验清晰地表明，无论是互联网规模的视觉-语言预训练，还是大规模机器人数据的预训练，对于提升模型的泛化能力和下游任务的适应性都至关重要。

## 7.2. 局限性与未来工作
论文作者坦诚地指出了当前 OpenVLA 存在的几个局限性以及未来的研究方向：
*   **单图像输入：** 目前模型每次决策只看当前的单帧图像，而没有利用历史观测信息（如视频序列）或机器人自身的本体感受信息（如关节角度）。扩展模型以支持多模态、时序输入是重要的下一步。
*   **推理速度：** 虽然模型在某些GPU上能达到约 6Hz，但这对于需要更高频率控制的灵巧操作任务（如 ALOHA 项目的 50Hz）来说仍然不足。探索动作分块 (action chunking) 或推测解码 (speculative decoding) 等技术来提升推理吞吐量是未来的一个方向。
*   **可靠性有待提升：** 尽管性能领先，但 OpenVLA 在大多数任务上的成功率仍低于90%，距离在真实世界中实现高度可靠的应用还有差距。
*   **更多待探索的设计问题：** 由于计算资源限制，许多问题仍未得到解答，例如：基础VLM的规模对最终VLA性能有多大影响？将机器人数据和互联网数据混合进行<strong>联合训练 (co-training)</strong> 是否会带来显著提升？最佳的视觉特征组合是什么？

## 7.3. 个人启发与批判
这篇论文给我带来了深刻的启发，也引发了一些思考：

*   **开源的力量：** 本文最重要的贡献或许并非模型本身，而是其**完全开源**的决心。它为整个机器人学习社区提供了一个强大且可及的基线平台，使得无数研究者和开发者可以在此基础上进行创新，从而极大地加速整个领域的发展。这与 `Llama` 系列模型在自然语言处理领域带来的影响如出一辙。
*   <strong>“预训练+微调”</strong>范式的胜利： OpenVLA 的成功再次印证了“预训练+微调”范式在机器人领域的巨大潜力。它表明，与其从零开始构建复杂的机器人专用模型，不如站在通用基础模型的“肩膀”上，将机器人的特定问题巧妙地“翻译”成基础模型擅长解决的形式。
*   **对数据质量和多样性的思考：** 论文的结果反复强调了数据的重要性。OpenVLA 的成功不仅源于其模型架构，也得益于对 `OpenX` 数据集进行的精心筛选和混合。这提醒我们，在追求更大模型的同时，如何构建和管理高质量、多样化的大规模机器人数据集，将是未来机器人学习的核心挑战之一。
*   **潜在问题与改进方向：**
    *   **动作离散化的影响：** 论文采用的均匀离散化是一种相对简单的方法。这种方式是否是最优的？对于需要极高精度的任务，256个桶的粒度是否足够？探索更先进的离散化方法（如 K-means 聚类）或者直接预测连续动作的混合模型，可能是一个值得研究的方向。
    *   **对失败的鲁棒性：** 模仿学习天然地假设演示数据都是成功的。模型在遇到训练数据中未见的失败状态时，如何进行恢复 (recovery)？虽然论文提到模型能从不稳的抓取中恢复，但这方面的能力仍需更系统的评估和增强。结合强化学习或从失败中学习的方法可能会是未来的一个补充。
    *   **计算门槛依然存在：** 尽管 `LoRA` 和量化技术降低了门槛，但7B模型的微调和推理对于许多个人开发者或资源有限的实验室来说，仍然是一笔不小的开销。未来继续探索更小、更高效的模型架构（如 `MoE` 结构）依然具有重要价值。