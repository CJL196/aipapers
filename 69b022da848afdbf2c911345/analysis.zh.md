# 1. 论文基本信息

## 1.1. 标题
**论文标题：** PaLM-E: An Embodied Multimodal Language Model（PaLM-E：一种具身多模态语言模型）

该标题揭示了论文的核心研究对象。其中，"PaLM-E"是模型的名称，"Embodied"意为“具身的”，指模型具备感知物理世界并与之互动的能力；"Multimodal Language Model"则表明这是一个能够处理多种模态（如文本、图像、传感器数据）的大语言模型。

## 1.2. 作者及隶属机构
**主要作者：** Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Vu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, Pete Florence.

**所属机构：**
*   **Robotics at Google:** 谷歌机器人研究部门，专注于机器人技术的前沿探索。
*   **TU Berlin:** 柏林工业大学，德国著名理工科大学，在机器人和计算机视觉领域享有盛誉。
*   **Google Research:** 谷歌研究院，负责基础人工智能研究和算法开发。

    这些作者主要来自顶级科技公司和顶尖高校，体现了该研究的工业界应用背景与学术界的深度结合。

## 1.3. 发表期刊/会议与年份
*   **发布状态：** 预印本 (ArXiv Preprint)
*   **发布时间：** 2023年3月6日 (UTC)
*   **原文链接:** https://arxiv.org/abs/2303.03378
*   **PDF 链接:** https://arxiv.org/pdf/2303.03378v1

    虽然截至当前时间（2026年），该论文可能已被顶级会议接收（如 CVPR, ICRA, RSS 等），但根据提供的信息，其核心发布日期为 2023 年 3 月。作为 arXiv 预印本，它已迅速成为具身智能与大语言模型交叉领域的奠基性工作之一。

## 1.4. 摘要
本文提出了一种名为 **PaLM-E** 的具身多模态语言模型。针对传统大语言模型（LLM）缺乏“接地性 (Grounding)”——即无法将语言符号与真实世界的物理感知相连接——的问题，PaLM-E 通过将连续的现实世界传感器模态（如图像、状态估计）直接整合到语言模型的嵌入空间中，建立了语言词汇与感知之间的直接联系。

**核心方法：** 输入为交错排列了视觉、连续状态估计和文本编码的多模态句子。这些编码与预训练的大语言模型一起进行端到端训练，用于执行序列机器人操作规划、视觉问答 (VQA) 和图像描述等多种具身任务。

**主要成果：**
1.  **单一模型多功能：** 评估表明，单个大规模具身边际模型可以解决各种观察模态下的不同具身推理任务，并表现出积极的知识迁移能力。
2.  **规模效应：** 最大的模型 **PaLM-E-562B**（5620 亿参数）不仅在机器人任务上表现出色，还在通用视觉语言基准（如 OK-VQA）上达到了最先进水平 (state-of-the-art)，且在增加规模的同时保持了通用的语言能力。
3.  **数据效率：** 通过跨任务和跨模态的共同训练，模型能够从极少量的机器人数据中高效学习，实现少样本甚至零样本泛化。

## 1.5. 原文链接与访问权限
论文的官方来源为 **arXiv** 开源库，文章类型为计算机科学类（cs.RO - Robotics, cs.CV - Computer Vision）。由于是预印本，公众可直接免费下载和阅读 PDF 版本，无需支付订阅费用。这有助于促进学术界和工业界对具身智能技术的快速理解和应用。

# 2. 整体概括

## 2.1. 研究背景与动机

### 2.1.1. 核心问题：具身智能中的“接地性”挑战
<strong>具身智能 (Embodied AI)</strong> 是指智能体（如机器人）通过身体与环境互动来获取信息和做出决策。当前的<strong>大语言模型 (Large Language Models, LLMs)</strong> 展现了强大的推理和对话能力，但它们本质上是基于文本训练的。这导致了一个核心问题：<strong>接地性 (Grounding)</strong>。
*   **问题描述：** LLMs 在庞大的文本数据上训练，虽然掌握了语言知识，但这些语言符号往往缺乏对物理世界的真实理解。例如，LLM 知道“杯子”这个词，但它不知道如何抓取一个真实的杯子，或者无法理解摄像头看到的画面中哪些是杯子。
*   **现有局限：** 现有的视觉 - 语言模型 (Visual-Language Models, VLMs) 大多针对静态任务（如看图说话），难以直接处理需要长期规划和实时控制的机器人任务。它们通常无法直接将视觉输入转化为可执行的物理动作序列。

### 2.1.2. 为什么这个问题很重要？
在现实世界中，要实现真正的通用机器人，模型必须具备以下能力：
1.  **多模态感知：** 能同时理解图像、传感器读数（如关节角度）和自然语言指令。
2.  **物理交互：** 能根据感知输出具体的控制计划（如“拿起红色方块”），并能在动态环境中调整计划。
3.  **泛化能力：** 面对从未见过的物体或环境组合时，仍能推理出合理的行动方案。

    如果不解决“接地性”问题，机器人只能执行预设的代码，而无法像人类一样通过自然语言理解新任务。因此，将 LLM 的强大推理能力与机器人的物理感知能力结合起来，是实现下一代智能系统的关键。

### 2.1.3. 切入思路
本文的创新切入点在于**架构层面的融合**。作者没有简单地将视觉编码器作为 LLM 的前置模块，而是提出了一种全新的输入格式："<strong>多模态句子 (Multi-modal Sentences)</strong>"。在这种格式下，连续的观测数据（如图像特征）被转换为与语言词元（Token）具有相同维度的向量，直接插入到 LLM 的输入序列中，与文本词元交织在一起，让 LLM 能够以处理文本的方式直接处理感知数据。

## 2.2. 核心贡献/主要发现

本文的主要贡献可以归纳为以下五点：
1.  **提出了通用具身决策代理的训练范式：** 证明了通过将具身数据混入多模态大语言模型的训练中，可以训练出一个通用的、具备跨实体能力的决策代理。
2.  **验证了通用视觉语言模型的局限性及改进方案：** 指出当前最先进的通用视觉语言模型在没有针对具身任务微调的情况下表现不佳，而本文提出的 PaLM-E 证明了可以通过训练使其成为高效的具身推理者。
3.  **引入了新颖的架构思想：** 提出了神经场景表示（Neural Scene Representations，具体为 OSRT）和实体标签多模态词元（Entity-labeling multimodal tokens），使模型能更好地引用和操作场景中的具体物体。
4.  **展示了模型的双重能力：** PaLM-E 不仅是一个具身推理器，也是一个数量上合格的视觉和语言通用专家（Generalist）。其在 OK-VQA 等基准测试上的表现优于许多专门微调的模型。
5.  **揭示了规模扩展的益处：** 证明了增大语言模型的规模可以减少在模态微调过程中的“灾难性遗忘 (Catastrophic Forgetting)"现象。即模型越大，在学习新的视觉/机器人任务时，保留原有语言能力的损失越小。

# 3. 预备知识与相关工作

## 3.1. 基础概念

为了深入理解本文，读者需要掌握以下关键技术术语：

1.  <strong>大语言模型 (Large Language Model, LLM):</strong>
    *   **定义：** 一种基于海量文本数据训练的深度学习模型，旨在理解、生成和处理人类语言。典型的架构是基于 Transformer 的解码器（Decoder-only）。
    *   **作用：** 在本文中，LLM 充当大脑，负责逻辑推理、任务规划和语言理解。
2.  <strong>词元 (Token):</strong>
    *   **定义：** 文本处理的基本单位。对于中文可能是字或词，对于英文通常是子词。在嵌入空间中，每个 Token 对应一个向量。
    *   **本文用法：** PaLM-E 将非文本数据（如图片）也转换成类似 Token 的向量，统称为多模态词元。
3.  <strong>视觉 Transformer (Vision Transformer, ViT):</strong>
    *   **定义：** 一种将图像分割成固定大小的图像块（Patch），并将其视为序列输入 Transformer 架构的视觉骨干网络。
    *   **作用：** 用于从原始图像中提取特征表示。
4.  <strong>具身 (Embodied):</strong>
    *   **定义：** 指系统拥有物理形态（身体），能够通过传感器感知环境，并通过执行器（如机械臂）对环境施加影响。
5.  <strong>微调 (Fine-tuning):</strong>
    *   **定义：** 将一个在大规模数据集上预训练好的模型，利用特定领域的小规模数据进行进一步训练，以适应特定任务的过程。
6.  <strong>灾难性遗忘 (Catastrophic Forgetting):</strong>
    *   **定义：** 神经网络在学习新任务时，往往会忘记之前学到的旧任务知识的现象。
7.  <strong>强化学习策略 (Reinforcement Learning Policy):</strong>
    *   **定义：** 在强化学习中，策略（Policy）是智能体根据当前状态选择动作的映射函数。本文中的低层策略负责执行具体的电机控制指令。
8.  <strong>真实标注数据 (Ground Truth):</strong>
    *   **定义：** 在机器学习和数据科学中，指数据的实际正确答案或标准状态，常用于评估模型性能。
    *   **注意：** 在机器人语境下，有时指物体的真实位姿或物理属性。

## 3.2. 前人工作

作者在文中回顾了与该研究相关的几个关键方向：

1.  <strong>通用视觉语言建模 (General vision-language modeling):</strong>
    *   **代表工作：** Flamingo (Alayrac et al., 2022), PaLI (Chen et al., 2022)。
    *   **特点：** 这些模型能够理解图像和文本，执行 VQA 和图像描述任务。
    *   **区别：** 大多数 VLM 侧重于感知任务，而非生成具体的动作序列来操控物理世界。
2.  <strong>动作输出模型 (Actions-output models):</strong>
    *   **代表工作：** VIMA (Jiang et al., 2022), Gato (Reed et al., 2022)。
    *   **特点：** 尝试直接预测动作或使用多模态提示来控制机器人。
    *   **区别：** 本文强调 PaLM-E 生成的是高阶文本指令，这些指令可以被低层策略解释，且模型利用了预训练 LLM 的内部世界知识，而不仅仅是模仿轨迹。
3.  **LLMs 在具身任务规划中的应用:**
    *   **代表工作：** SayCan (Ahn et al., 2022)。
    *   **特点：** 使用 LLM 结合可及性函数（Affordance functions）来决定动作。
    *   **局限：** 依赖额外的辅助模型来进行“接地”，且 LLM 本身未针对视觉输入进行联合训练。

## 3.3. 技术演进
该技术路线经历了从“纯文本 LLM"到“多模态 VLM"再到“具身 LLM"的演变：
1.  **阶段一：** 仅处理文本，擅长逻辑但无法感知物理。
2.  **阶段二：** 引入视觉，能“看懂”图片，但难以“动手”。
3.  <strong>阶段三 (本文)：</strong> 将感知直接嵌入语言空间，实现了“脑眼手”一体化，支持闭环控制和长程规划。

## 3.4. 差异化分析

| 特性 | 传统 LLM (如 PaLM) | 通用 VLM (如 PaLI) | 本文 PaLM-E |
| :--- | :--- | :--- | :--- |
| **输入模态** | 纯文本 | 文本 + 图像 | 文本 + 图像 + 状态向量 + 物体中心表示 |
| **核心能力** | 生成文本 | 问答、描述 | 机器人规划、具身推理、VQA |
| **接地方式** | 无物理接地 | 弱物理接地 | **强物理接地** (直接嵌入连续信号) |
| **输出形式** | 文本 | 文本 | 文本 (可被低层策略解析为动作) |
| **训练目标** | 语言预测 | 语言 - 图像对齐 | 多任务联合训练 (机器人+视觉+语言) |

本文的独特之处在于打破了语言和感知的界限，将两者置于同一个数学空间（嵌入空间）中，使得模型能够自然地用处理语言的方式来处理物理世界的约束。

# 4. 方法论

## 4.1. 方法原理
PaLM-E 的核心思想是<strong>注入连续观测 (Injection of Continuous Observations)</strong>。传统的多模态模型通常是将图像特征作为一个独立的 token 块放在文本前后。而 PaLM-E 主张将图像、状态估计等连续传感器模态，通过编码器映射到与语言 Token 相同的维度空间 $\mathcal{X}$，然后与文本 Token 交织在一起，形成“多模态句子”。这样做的好处是可以复用语言模型原有的位置编码 (Positional Encodings) 和自注意力机制 (Self-attention)，让模型在注意力计算中天然地关联文本语义与视觉内容。

## 4.2. 核心方法详解

### 4.2.1. 仅解码器大语言模型基础
PaLM-E 的基础是一个仅解码器（Decoder-only）的大语言模型。这种模型通过预测下一个词元的概率来生成文本。对于一段文本 $w_{1:L} = (w_1, \dots, w_L)$，其概率分布因子化为：

$$
p ( w _ { 1 : L } ) = \prod _ { l = 1 } ^ { L } p _ { \mathrm { L M } } ( w _ { l } | w _ { 1 : l - 1 } ) ,
$$

其中 $p_{\mathrm{LM}}$ 是大型 Transformer 网络。这个公式表达了自回归生成的基本逻辑：预测第 $l$ 个词元依赖于前 `l-1` 个词元的历史。

### 4.2.2. 前缀解码器 (Prefix-decoder) 模式
由于 LLM 是自回归的，我们可以利用其前缀条件预测的能力。给定前缀 $w_{1:n}$，模型预测后续 token 的概率为：

$$
p ( w _ { n + 1 : L } | w _ { 1 : n } ) = \prod _ { l = n + 1 } ^ { L } p _ { \mathrm { L M } } ( w _ { l } | w _ { 1 : l - 1 } ) .
$$

这里的前缀 $w_{1:n}$ 包含了任务描述或上下文，它不仅包含文本，还包含了嵌入后的多模态观测。这允许我们在不改变 LLM 架构的情况下，为其提供丰富的感知上下文。

### 4.2.3. 词元嵌入空间与多模态句子注入
在标准的 LLM 中，词元 $w_i$ 来自有限词汇表 $\mathcal{W}$，通过嵌入矩阵映射到向量空间 $\mathcal{X} \subset \mathbb{R}^k$，即 `x_i = \gamma(w_i)`。

PaLM-E 的创新在于直接将连续观测 $O$ 映射到同一个空间 $\mathcal{X}$。我们训练一个编码器 $\phi: \mathcal{O} \to \mathcal{X}^q$，它将观测空间 $\mathcal{O}$（包括图像、状态等）映射为 $q$ 个向量。这些向量与文本嵌入向量交织，构成前缀 $x_{1:i}$。

对于前缀中的第 $i$ 个向量 $x_i$，其来源由以下规则决定：

$$
x _ { i } = \left\{ \begin{array} { l l } { \gamma ( w _ { i } ) } & { \mathrm { i f ~ } i \mathrm { ~ a ~ i s ~ t e x t ~ t o k e n , ~ o r ~ } } \\ { \phi _ { j } ( O _ { j } ) _ { i } } & { \mathrm { i f ~ } i \mathrm { ~ c o r r e s p o n d s ~ t o ~ o b s e r v a t i o n ~ } O _ { j } . } \end{array} \right.
$$

**公式解析：**
*   如果 $i$ 是文本词元的位置，则使用标准的文本嵌入器 $\gamma$ 获取向量。
*   如果 $i$ 对应于观测 $O_j$ 的位置，则使用编码器 $\phi_j$ 获取对应的向量片段。
*   这意味着一个观测 $O_j$ 通常会被编码成多个嵌入向量，这些向量可以穿插在文本序列的任何位置。这种动态插入方式不同于将图像特征固定在特殊位置的旧方法。

### 4.2.4. 不同的输入模态编码器
为了实现上述映射，作者设计了针对不同传感器的编码器 $\phi$：
1.  <strong>状态估计向量 (State estimation vectors):</strong> 最简单的情况，机器人姿态或物体位姿向量 $\boldsymbol{s} \in \mathbb{R}^S$ 直接通过多层感知机 (MLP) $\phi_{\mathrm{state}}$ 映射到嵌入空间。
2.  <strong>视觉 Transformer (ViT):</strong> 将图像 $I$ 映射为一系列 Token 嵌入 $\tilde{x}_{1:m}$。由于 ViT 输出的维度可能与语言模型不一致，需要投影头 $\psi$ 进行变换：$x_i = \phi_{\mathrm{ViT}}(I)_i = \psi(\tilde{\phi}_{\mathrm{ViT}}(I)_i)$。
3.  <strong>对象中心表示 (Object-centric representations):</strong> 为了更好地理清场景中不同物体，使用了 <strong>对象场景表示 Transformer (OSRT)</strong>。OSRT 通过无监督学习发现物体槽（Object slots）$o_j$，将其映射为多模态 Token。这种方法不需要 Ground Truth 分割掩码即可自动分离物体。
4.  <strong>实体指代 (Entity referrals):</strong> 对于需要明确引用物体的任务，OSRT 输出的对象槽会被标记为特殊 Token（如 $<obj-1>$）。这使得模型可以在输出中使用 `obj-1` 这样的占位符来指代输入图像中的特定物体，便于底层策略执行。

### 4.2.5. 具身化输出与控制回路
PaLM-E 输出的是文本。为了连接物理世界，区分两种情况：
1.  **纯文本任务：** 如具身问答，输出直接作为答案。
2.  **具身规划任务：** 输出包含技能序列的自然语言文本（例如 "Grasp the red cup"）。假设存在底层策略（Low-level policy）可以将这些技能翻译为具体的电机控制指令。
    *   **控制循环：** PaLM-E 生成的决策被执行后，机器人获得新观测，PaLM-E 可以根据反馈重新规划（Replan）。这使得 PaLM-E 成为一个高层策略（High-level policy），负责调度和控制底层策略。

## 4.3. 训练配方
训练数据 $D$ 的形式为 $\{(I_{1:u_i}^i, w_{1:L_i}^i, n_i)\}$，包含观测、文本和前缀索引。损失函数是对非前缀部分的交叉熵损失：
$$
\mathcal{L} = -\sum_{l=n_i+1}^{L_i} \log p_{\mathrm{LM}}(w_l | w_{1:l-1})
$$
（注：原文公式虽未显式写出求和号，但在交叉熵描述中隐含了对非前缀词的损失平均）。

作者对比了两种训练策略：
1.  **全参数微调：** 更新编码器 $\tilde{\phi}$、投影器 $\psi$ 和 LLM 的所有参数。
2.  **冻结 LLM 策略：** 保持 LLM 参数不变，只训练输入编码器。这类似于“软提示 (Soft Prompts)"技术，试图让编码器输出能让冻结 LLM 产生正确反应的向量。实验表明，随着模型规模扩大，全参数微调能有效减少灾难性遗忘。

# 5. 实验设置

## 5.1. 数据集

### 5.1.1. 机器人数据集
实验涵盖了三种机器人环境，数据量相对较少，强调了数据效率：
1.  <strong>任务与运动规划 (TAMP):</strong> 涉及物体抓取和堆叠，包含复杂的几何关系推理。
2.  <strong>桌面推物 (Language-Table):</strong> 基于公开数据集，涉及多物体推动、排序任务，语言指令复杂度高。
3.  <strong>移动操作 (Mobile Manipulation):</strong> 模拟厨房场景，机器人需导航并寻找物体（如抹布、海绵），类似 SayCan 的任务设定。

### 5.1.2. 通用视觉 - 语言数据集
为了验证通用能力并促进迁移学习，使用了互联网规模的混合数据集（Full mixture），主要包括：
*   **Webli:** 网页图文对数据。
*   **VQAv2, OKVQA:** 视觉问答数据集。
*   **COCO:** 图像描述数据集。
*   **维基百科文本:** 纯文本数据。
*   **机器人数据占比:** 在完整混合数据集中，机器人相关数据仅占约 **8.9%**。其余大部分为通用图文数据。

**数据来源示例：**
*   表格 B.6 (Appendix) 显示了采样频率。例如 Webli 采样频率为 100，占比 52.4%；TAMP 机器人数据采样频率为 3，占比 1.6%。

## 5.2. 评估指标

为了全面评估模型性能，论文使用了多种指标：

1.  <strong>成功率 (Success Rate):</strong>
    *   **概念定义：** 衡量模型在执行机器人规划任务时，完成目标的次数占总尝试次数的比例。这是评价具身智能系统鲁棒性的核心指标。
    *   **计算公式：**
        $$ \text{Success Rate} = \frac{N_{\text{successful}}}{N_{\text{total}}} \times 100\% $$
    *   **符号解释：** $N_{\text{successful}}$ 为成功完成任务的次数，$N_{\text{total}}$ 为总测试样本数。

2.  <strong>F1 分数 (F1 Score):</strong>
    *   **概念定义：** 用于评估分类任务的精确度 (Precision) 和召回率 (Recall) 的综合指标，特别适用于不平衡数据或二分类任务（如故障检测）。
    *   **计算公式：**
        $$ F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$
    *   **符号解释：** Precision 为查准率（预测为正例中有多少是真的正例），Recall 为查全率（真实正例中有多少被找到了）。

3.  **OK-VQA Accuracy:**
    *   **概念定义：** 在 OK-VQA 基准上的准确率。这是一个需要外部知识的视觉问答数据集，考验模型结合视觉与常识推理的能力。
    *   **符号解释：** 准确回答问题的百分比。

4.  **VQA v2 Accuracy:**
    *   **概念定义：** 在 VQA v2 基准上的准确率，衡量基础的图像理解和问答能力。

## 5.3. 对比基线
作者选择了具有代表性的模型作为竞争对手：
1.  **SayCan:** 结合了 LLM 和可及性函数（Affordance functions）的经典方法。它展示了如何将 LLM 的输出连接到机器人策略，但其 LLM 本身是冻结的，且依赖外部可及性模型。
2.  **PaLI:** 当时最先进的通用视觉语言模型，但未在机器人数据上训练，作为零样本（Zero-shot）基线。
3.  **Flamingo:** 另一个著名的多模态 Few-shot 模型。

# 6. 实验结果与分析

## 6.1. 核心结果分析

### 6.1.1. 迁移学习的有效性 (Transfer Learning)
图 3 和图 4 直观展示了跨任务、跨实体共同训练带来的显著性能提升。
*   **现象：** 即使在机器人数据极少（如 TAMP 环境仅使用 1% 的数据，即 320 个样本）的情况下，经过“完整混合训练 (Full mixture)"的模型性能远高于仅在特定领域数据上训练的模型。
*   **原因：** 通用视觉 - 语言数据（如图文对、常识问答）帮助模型建立了对物理世界更广泛的理解（例如知道“杯子”是什么样，“倒水”是什么意思），这些知识迁移到了机器人规划任务中，弥补了机器人数据的不足。

    以下是原文 **Table 1** 的结果，展示了不同输入表示在 TAMP 环境下的表现：

    <table>
    <thead>
    <tr>
    <th rowspan="2">Input Rep.</th>
    <th rowspan="2">Object-centric</th>
    <th rowspan="2">LLM pre-train</th>
    <th colspan="4">Embodied VQA</th>
    <th colspan="2">Planning</th>
    </tr>
    <tr>
    <th>q1</th>
    <th>q2</th>
    <th>q3</th>
    <th>q4</th>
    <th>p1</th>
    <th>p2</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>SayCan (oracle afford.)</td>
    <td>-</td>
    <td>✓</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>38.7</td>
    <td>33.3</td>
    </tr>
    <tr>
    <td>PaLI (zero-shot)</td>
    <td>-</td>
    <td>✓</td>
    <td>-</td>
    <td>0.0</td>
    <td>0.0</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>PaLM-E (ours) w/ input enc:</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>State</td>
    <td>(GT)</td>
    <td>X</td>
    <td>-</td>
    <td>99.4</td>
    <td>89.8</td>
    <td>90.3</td>
    <td>88.3</td>
    <td>45.0</td>
    </tr>
    <tr>
    <td>State</td>
    <td>(GT)</td>
    <td>✓</td>
    <td>-</td>
    <td>100.0</td>
    <td>96.3</td>
    <td>95.1</td>
    <td>93.1</td>
    <td>49.7</td>
    </tr>
    <tr>
    <td>ViT + TL</td>
    <td>(GT)</td>
    <td>✓</td>
    <td>34.7</td>
    <td>54.6</td>
    <td>74.6</td>
    <td>-</td>
    <td>91.6</td>
    <td>24.0</td>
    </tr>
    <tr>
    <td>ViT-4B single robot</td>
    <td>X</td>
    <td>✓</td>
    <td>-</td>
    <td>45.9</td>
    <td>78.4</td>
    <td>-</td>
    <td>92.2</td>
    <td>32.9</td>
    </tr>
    <tr>
    <td>ViT-4B full mixture</td>
    <td>X</td>
    <td>✓</td>
    <td>-</td>
    <td>70.7</td>
    <td>93.4</td>
    <td>-</td>
    <td>92.1</td>
    <td>74.6</td>
    </tr>
    <tr>
    <td>OSRT (no VQA)</td>
    <td>✓</td>
    <td>X</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>71.9</td>
    <td>75.1</td>
    </tr>
    <tr>
    <td>OSRT</td>
    <td>✓</td>
    <td>✓</td>
    <td>99.7</td>
    <td>98.2</td>
    <td>100.0</td>
    <td>82.5</td>
    <td>76.2</td>
    <td>-</td>
    </tr>
    </tbody>
    </table>

**分析：**
*   **OSRT 的优势：** 即使没有大规模预训练，使用对象中心表示（OSRT）的方法（最后一行）在规划任务（p1, p2）上也取得了 71.9% 和 75.1% 的高分，证明了结构化表示的重要性。
*   **迁移效果：** 比较 "ViT-4B single robot" 和 "ViT-4B full mixture" 可知，混合训练使得规划成功率从 32.9% 大幅提升至 74.6%，几乎翻倍。

### 6.1.2. 不同环境下的表现
**Table 2** 展示了在 Language-Table 环境（仿真）下的规划任务成功率：

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">LLM Pretrain</th>
<th colspan="2">Training</th>
<th colspan="2">Testing</th>
<th colspan="3">Task 1 (# Demos)</th>
<th colspan="3">Task 2 (# Demos)</th>
<th colspan="3">Task 3 (# Demos)</th>
</tr>
<tr>
<th>Single robot</th>
<th>Full mixture</th>
<th>10</th>
<th>20</th>
<th>40</th>
<th>10</th>
<th>20</th>
<th>40</th>
<th>80</th>
</tr>
</thead>
<tbody>
<tr>
<td>SayCan (zero-shot)</td>
<td>✓</td>
<td>×</td>
<td>—</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td>PaLI (zero-shot)</td>
<td>✓</td>
<td>×</td>
<td>—</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td>PaLM-E-12B</td>
<td>×</td>
<td>✓</td>
<td>✓</td>
<td>—</td>
<td>—</td>
<td>50.0</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>28.3</td>
</tr>
<tr>
<td>PaLM-E-12B</td>
<td>×</td>
<td>✓</td>
<td>✓</td>
<td>—</td>
<td>—</td>
<td>80.0</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>50.0</td>
</tr>
<tr>
<td>PaLM-E-12B</td>
<td>×</td>
<td>✓</td>
<td>✓</td>
<td>70.0</td>
<td>80.0</td>
<td>80.0</td>
<td>31.3</td>
<td>58.8</td>
<td>58.8</td>
<td>57.5</td>
<td>56.3</td>
<td>56.3</td>
<td>56.3</td>
<td>56.3</td>
</tr>
<tr>
<td>PaLM-E-84B</td>
<td>×</td>
<td>✓</td>
<td>✓</td>
<td>—</td>
<td>—</td>
<td>90.0</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>53.8</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>64.4</td>
</tr>
</tbody>
</table>

*   **数据效率：** 仅需 10-80 个演示 (Few-shot)，PaLM-E 就能达到较高的成功率，特别是在 Full mixture 训练下。
*   **规模效应：** 84B 模型相比 12B 模型在 Task 1 上从 80.0% 提升到 90.0%。

### 6.1.3. 通用视觉语言任务
尽管重点是机器人，PaLM-E 在通用任务上同样出色。**Table 5** 显示其在 OK-VQA 上达到 **66.1%** (PaLM-E-562B)，超越了专门微调的 PaLI (64.5%)。这证明了模型并未因机器人任务而丧失通用视觉能力。

<details>
<summary>点击查看原文 Table 5 结构详情</summary>
<div style="padding: 10px; border-left: 2px solid #ccc; background-color: #f9f9f9;">
下表展示了不同模型在 VQA 和图像描述任务上的表现。<br><br>

<table>
<thead>
<tr>
<th>Model</th>
<th>VQAv2 test-dev</th>
<th>VQAv2 test-std</th>
<th>OK-VQA val</th>
<th>COCO Karpathy test</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="5"><b>Generalist (one model)</b></td>
</tr>
<tr>
<td>PaLM-E-12B</td>
<td>76.2</td>
<td>-</td>
<td>55.5</td>
<td>135.0</td>
</tr>
<tr>
<td>PaLM-E-562B</td>
<td>80.0</td>
<td>-</td>
<td>66.1</td>
<td>138.7</td>
</tr>
<tr>
<td colspan="5"><b>Task-specific finetuned models</b></td>
</tr>
<tr>
<td>Flamingo (Alayrac et al., 2022)</td>
<td>82.0</td>
<td>82.1</td>
<td>57.8†</td>
<td>138.1</td>
</tr>
<tr>
<td>PaLI (Chen et al., 2022)</td>
<td>84.3</td>
<td>84.3</td>
<td>64.5</td>
<td>149.1</td>
</tr>
<tr>
<td>PaLM-E-12B</td>
<td>77.7</td>
<td>77.9</td>
<td>60.1</td>
<td>136.0</td>
</tr>
<tr>
<td>PaLM-E-66B</td>
<td>-</td>
<td>-</td>
<td>62.9</td>
<td>-</td>
</tr>
<tr>
<td>PaLM-E-84B</td>
<td>80.5</td>
<td>-</td>
<td>63.3</td>
<td>138.0</td>
</tr>
<tr>
<td colspan="5"><b>Generalist (one model), with frozen LLM</b></td>
</tr>
<tr>
<td>(Tsimpoukelli et al., 2021)</td>
<td>48.4</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>PaLM-E-12B frozen</td>
<td>70.3</td>
<td>-</td>
<td>51.5</td>
<td>128.0</td>
</tr>
</tbody>
</table>

</div>
</details>

## 6.2. 消融实验与参数分析

### 6.2.1. 冻结 LLM 的效果
作者探讨了是否只需训练编码器而冻结 LLM（类似 Prompt Tuning）。
*   **结果：** 在小模型（如 12B）上，冻结 LLM 在机器人任务上表现稍差（Tab 2），因为小模型缺乏足够的内部知识来泛化。
*   **大图优势：** 在大模型（如 562B）上，全量训练反而更有效，因为它避免了灾难性遗忘，同时保留了更强的推理能力。

### 6.2.2. 数据混合的影响
图 4 展示了 TAMP 环境下不同训练策略对规划成功率的影响。
下图（原文 **Figure 4**）展示了使用 PaLM-E-12B 模型在 TAMP 环境中的规划成功率对比：

![Figure 4: Planning success results in the TAMP environment $1 \\%$ data) for PaLM-E-12B, comparing of the effects of PaLM-E models (i) using the full training mixture, (ii) pre-training (ViT and PaLM), and (iii) freezing or finetuning the language model. Transfer from full mixture is particularly effective. Note that full mixture contains only $1 \\%$ of the training data (320 examples each) for the tasks evaluated here. Shown is the mean of tasks $\\mathsf { p } _ { 1 } , \\mathsf { p } _ { 2 }$ .](images/12.jpg)
*该图像是一个条形图，展示了在 TAMP 环境中使用 PaLM-E-12B 模型的不同训练策略对规划成功率的影响。条形图显示了五种方法的成功率，分别为 LLM finetune（full mixture）、single robot 及 without pretraining，LLM frozen（full mixture）、single robot，成功率从 31.8\% 到 94.9\% 不等。*

*   **解读：** 图中横轴表示不同的训练配置，纵轴为成功率。可以看到，"full mixture"（全混合训练）配合"finetune"（微调）策略获得了最高的成功率（超过 90%），而仅使用单机器人数据或冻结 LLM 的策略表现明显下降。这证实了跨域数据混合是提升性能的关键。

### 6.2.3. 灾难性遗忘的缓解
图 6 展示了随着模型规模增加，PaLM-E 在通用语言任务上的性能保持情况。
下图（原文 **Figure 6**）显示了随着模型规模增大，语言能力的灾难性遗忘显著减少：

![Figure 6: Results on general language tasks ( $\\mathbf { N L G } =$ natural language generation): increasing scale leads to less catastrophic forgetting between a corresponding PaLM-E model and its inherited PaLM model. See full suite of tasks and results in Tab. 8.](images/14.jpg)
*该图像是图表，展示了PaLM和PaLM-E在自然语言生成任务（NLG）上的平均表现。随着模型规模的增加，PaLM-E模型在84B参数时相较于PaLM模型的下降幅度为61.6%，而在562B时下降仅为3.9%。*

*   **数据：** 最小的 12B 模型在 NLG 性能上下降了 87.3%，而最大的 562B 模型仅下降了 3.9%。这证明大参数规模对于多模态微调至关重要。

# 7. 总结与思考

## 7.1. 结论总结
本文提出的 **PaLM-E** 成功构建了一个统一的框架，将大语言模型的强大推理能力与机器人的物理感知及控制能力结合。
1.  **架构创新：** 通过注入连续观测，打破了语言与感知的壁垒。
2.  **性能突破：** 在机器人规划和通用视觉语言任务上均取得了 SOTA 性能。
3.  **迁移与缩放：** 证明了跨任务训练和模型规模扩大能显著提升数据效率和稳定性。

## 7.2. 局限性与未来工作

### 7.2.1. 局限性
1.  **对底层策略的依赖：** PaLM-E 生成的是高层指令，必须依赖预定义的底层策略（Low-level policies）才能执行物理动作。如果底层策略不支持某种技能，LLM 也无法凭空创造出来。
2.  **训练成本极高：** 562B 参数的模型训练和推理需要巨大的算力资源，限制了中小研究机构的复现和应用。
3.  **安全与不确定性：** 虽然是端到端模型，但 LLM 可能会产生幻觉（Hallucination），在物理世界中可能导致危险操作（例如认为某个不存在的物体是可抓取的）。文中提到会输出"uncertainty"（不确定）但这仍需更严谨的安全机制。

### 7.2.2. 未来方向
1.  **更细粒度的控制：** 未来的工作可能希望模型直接输出底层的控制信号（如力矩、速度），而不仅仅是抽象的技能指令。
2.  <strong>仿真与现实的差距 (Sim-to-Real)：</strong> 目前部分实验在仿真中进行，如何进一步提高在真实复杂环境中的鲁棒性仍是挑战。
3.  **视频与时序建模：** 目前的输入主要是静态图像，引入视频流作为连续的时间感知输入将是重要趋势。

## 7.3. 个人启发与批判
**启发：**
*   **通用性是具身智能的必经之路：** 过去的机器人研究倾向于“专才”模型（专门学摆咖啡，专门学开瓶）。PaLM-E 证明了通过统一的多模态空间，可以培养“通才”模型，这在降低系统复杂性方面极具价值。
*   **数据混合的力量：** 机器人数据极其昂贵且稀缺。通过加入大量廉价的互联网图文数据（Web data）来增强机器人的常识理解，是一种性价比极高的策略。

**批判性思考：**
*   **黑箱风险：** 虽然模型表现优异，但由于融合了多个复杂的预训练模块（ViT, PaLM, OSRT），其内部的决策逻辑依然是黑箱。在需要高可靠性的工业场景（如医疗手术机器人）中，这可能是一个隐患。
*   **评估指标的单一性：** 论文主要关注成功率（Success Rate），但在真实世界中，操作的平滑度、能耗、安全性等非结构化指标同样重要。目前的评测体系仍偏向于完成既定任务，而非任务质量。

    总的来说，PaLM-E 是具身智能发展史上的一个重要里程碑，它为未来的通用机器人提供了强有力的“大脑”参考架构。