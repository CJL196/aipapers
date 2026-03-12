# 1. 论文基本信息

## 1.1. 标题
**标题：** Vision-Language-Action (VLA) Models: Concepts, Progress, Applications and Challenges
**中文译名：** 视觉 - 语言 - 动作（VLA）模型：概念、进展、应用与挑战
**说明：** 本文是一篇综述性学术论文（Review Paper），旨在系统性地梳理视觉 - 语言 - 动作模型的当前状态、技术细节及未来方向。

## 1.2. 作者
**作者团队：** Ranjan Sapkota, Yang Cao, Konstantinos I. Roumeliotis, Manoj Karkee
**隶属机构：**
*   Cornell University, Biological & Environmental Engineering, Ithaca, New York, USA (康奈尔大学，生物与环境工程学院，美国纽约伊萨卡)
*   The Hong Kong University of Science and Technology, Department f Computer Science and Engineering, Hong Kong (香港科技大学，计算机科学与工程系，中国香港)
*   University of the Peloponnese, Department of Informatics and Telecommunications, Greece (伯罗奔尼撒大学，信息技术与通信系，希腊)
    **背景分析：** 作者团队来自顶尖的学术研究机构，涵盖美国常春藤盟校、亚洲顶尖理工院校以及欧洲研究重镇。这种跨地域的合作背景表明该研究具有较高的国际视野和跨学科整合能力，特别是在机器人学、计算机科学与农业工程领域的交叉点上。

## 1.3. 发表期刊/会议
**发布平台：** arXiv
**标识符：** https://arxiv.org/abs/2505.04769
**发布状态：** 预印本 (Preprint)
**发布时间：** 2025-05-07T19:46:43.000Z (UTC)
**说明：** 该文章发布于 arXiv 数据库，属于计算机科学（Computer Science）领域，具体分类可能涉及人工智能（AI）、机器人学（Robotics）和计算机视觉（CV）。虽然尚未正式收录于某个特定期刊（如 IEEE TPAMI 或 CVPR），但 arXiv 是学术界公认的高影响力预印本平台，便于快速传播前沿成果。

## 1.4. 摘要
本文是对视觉 - 语言 - 动作（VLA）模型的一次基础性综述。它综合了最近三年发布的 80 多个 VLA 模型的研究成果。
*   **研究目的：** 统一感知（Perception）、自然语言理解（Natural Language Understanding）和具身动作（Embodied Action）。
*   **核心内容：** 涵盖了概念基础、架构创新、训练效率策略、实时推理加速、应用领域（自动驾驶、医疗、农业等）以及挑战与解决方案。
*   **关键结论：** VLA 模型正在推动具身智能向通用化、社会对齐方向发展，但仍面临实时推理、安全、泛化等挑战。
*   **资源链接：** GitHub 项目仓库：https://github.com/Applied-AI-Research-Lab/Vision-Language-Action-Models-Concepts-Progress-Applications-and-Challenges

## 1.5. 原文链接
*   **ArXiv 原文链接：** https://arxiv.org/abs/2505.04769
*   **PDF 下载链接：** https://arxiv.org/pdf/2505.04769v2

    ---

# 2. 整体概括

## 2.1. 研究背景与动机
*   **核心问题：** 传统机器人系统和人工智能往往将**视觉系统**、**语言系统**和**行动系统**视为独立模块。例如，视觉模型只能识别物体但无法理解指令，语言模型能处理文本但不能控制物理世界。这导致系统难以在复杂多变的现实环境中实现自适应行为（Adaptive Behavior）。
*   **重要性：** 随着大语言模型（LLM）和多模态视觉语言模型（VLM）的爆发，将“看”、“想”、“动”结合起来是实现真正通用人工智能（AGI）和具身智能（Embodied AI）的关键瓶颈。缺乏统一的 VLA 框架会导致机器人在面对新任务时泛化能力差，且需要大量的人工重新编程。
*   **切入点：** 本文通过综述现有文献，系统性地提出了一个五维度的结构框架（概念、进展、应用、挑战、路线图），旨在为研究人员提供一个清晰的导航图，以理解如何构建能够感知环境、理解指令并执行动作的智能体。

## 2.2. 核心贡献/主要发现
1.  **系统性分类：** 首次将 VLA 模型的研究现状进行了结构化梳理，包括从早期融合架构到通用智能体的演变。
2.  **技术深度解析：** 详细解释了 VLA 的核心机制，特别是<strong>标记化（Tokenization）</strong>如何将视觉、语言和动作统一在同一空间，以及<strong>多模态融合（Multimodal Integration）</strong>的具体实现方式。
3.  **全面的应用地图：** 列举了 6 个主要应用领域（类人机器人、自动驾驶、工业、医疗、农业、AR 导航），展示了 VLA 在不同场景下的具体落地潜力。
4.  **前瞻性路线图：** 针对现有的局限性（如实时性、安全性、数据集偏差），提出了未来的技术发展路径，强调了智能体自我学习和伦理治理的重要性。

    ---

# 3. 预备知识与相关工作

## 3.1. 基础概念
为了理解这篇论文，读者需要掌握以下基础概念：
*   <strong>视觉 - 语言模型 (Vision-Language Model, VLM):</strong> 一种能够同时理解和生成图像和文本的多模态模型。它能回答关于图片的问题或根据文本描述生成图片。
*   <strong>大语言模型 (Large Language Model, LLM):</strong> 基于Transformer架构训练的超大规模参数模型，擅长自然语言理解与生成。
*   <strong>具身智能 (Embodied Intelligence):</strong> 指智能体通过与物理环境的互动（感知、行动）来学习和适应的能力，强调“身体”对认知的作用。
*   **Transformer:** 一种深度学习架构，利用自注意力机制（Self-Attention）处理序列数据，是目前 VLM 和 LLM 的主流骨干。
*   <strong>扩散模型 (Diffusion Model):</strong> 一类生成式模型，通过逐步去噪的过程生成数据。在 VLA 中常用于生成平滑的动作轨迹。
*   <strong>模仿学习 (Imitation Learning):</strong> 智能体通过观察专家（人类或其他高级系统）的行为数据来学习策略的方法。

## 3.2. 前人工作与技术演进
VLA 的发展经历了明显的三个阶段，反映了从简单耦合到深度集成的过程：
1.  <strong>基础融合阶段 (2022-2023):</strong> 代表性工作如 CLIPort [202] 和 RT-1 [19]。这一阶段主要尝试将预训练的视觉特征与简单的动作控制结合。
    *   *局限：* 缺乏复杂的推理能力，难以处理长序列任务。
2.  <strong>具身推理专业化阶段 (2024):</strong> 引入了领域特定的归纳偏置（Inductive Bias），如 VoxPoser [100] 和 Octo [218]。开始关注少样本适应（Few-shot Adaptation）和 3D 场景图。
    *   *进展：* 能够处理部分可观测环境和更复杂的任务规划。
3.  <strong>泛化与安全部署阶段 (2025):</strong> 最新模型如 GR00T N1 [14] 和 SafeVLA [274]。重点在于鲁棒性、形式化验证和安全对齐。
    *   *趋势：* 双系统架构（System 1 快速反应 + System 2 慢速推理）成为主流。

## 3.3. 差异化分析
与传统模块化流水线相比，VLA 的核心区别在于<strong>端到端的联合训练（End-to-End Joint Training）</strong>和<strong>统一的空间表示（Unified Representation Space）</strong>。
*   **传统方法：** 视觉输出标签 -> 语言解析器 -> 规划器 -> 控制器。中间步骤可能导致信息丢失，且模块间难以协同优化。
*   **VLA 方法：** 所有输入（图像、文本、状态）被转换为 Token（词元），输入同一个 Transformer 网络，直接输出动作 Token。这种方式使得模型能够捕捉模态间的深层语义关联，实现零样本泛化（Zero-shot Generalization）。

    ---

# 4. 方法论

本章将深入解析论文所探讨的 VLA 核心技术方案。由于这是一篇综述，本节重点拆解文中描述的 VLA **标准技术范式**，即当前主流模型是如何构建的。

## 4.1. 方法原理：统一的多模态表征
VLA 模型的核心直觉是将“看”、“说”、“做”这三个原本分离的过程，统一在一个数学空间中。
*   **直观解释：** 想象机器人在思考时，不再分别处理图片像素、文字字符和电机信号，而是把它们都变成一串数字序列（Tokens）。就像写文章一样，机器人把动作也当作“句子”的一部分来预测。
*   **理论基础：** 基于 Transformer 的自回归（Autoregressive）生成机制。传统的 LLM 预测下一个词，而 VLA 预测下一个动作 Token。

## 4.2. 核心方法详解：VLA 令牌化与编码流程
这是 VLA 最基础的技术模块，决定了模型如何理解世界。论文中明确给出了算法流程（Algorithm 1）。

### 4.2.1. 前缀令牌 (Prefix Tokens): 上下文与指令编码
这部分负责编码外部环境和用户指令。
*   **视觉编码器：** 使用 Vision Transformer (ViT) 或 ConvNeXt 处理 RGB-D 帧 $I$。
    $V \leftarrow \text{ViT}(I)$
    *   **结果：** 产生一组约 400 个视觉词元。
*   **语言编码器：** 使用 BERT 或 T5 处理文本指令 $T$。
    $L \leftarrow \text{BERT}(T)$
    *   **结果：** 产生约 12 个语言词元。
*   **融合：** 这两个部分的 Embedding 作为 Prefix（前缀），告诉模型“现在在哪里”以及“要做什么”。

### 4.2.2. 状态令牌 (State Tokens): 机器人的内部配置
这部分负责编码机器人自身的物理状态。
*   **状态输入：** 包括关节角度 $\theta$、末端执行器姿态、力传感器读数等。
*   **编码方式：** 通常通过多层感知机（MLP）进行压缩编码。
    $$ S \leftarrow \text{MLP}(\theta) $$
    *   **维度：** 压缩为 64 维的状态向量。
    *   **作用：** 确保机器人在做动作时知道手伸到了哪里，有没有碰到障碍物。

### 4.2.3. 动作令牌 (Action Tokens): 自回归控制生成
这是最终输出的部分，代表机器人的运动。
*   **解码：** 模型接收上述所有 Token 的融合表示 $F$，通过策略解码器（Policy Decoder）预测动作序列。
    $$ F \leftarrow \text{CrossAttention}(V, L, S) $$
    $A \leftarrow \text{FAST}(F)$
*   **输出：** 产生约 50 个离散的动作词元，这些词元随后会被解码为具体的电机控制命令 $\tau_{1:N}$。
    $\tau_{1:N} = \text{Decode}(A)$
*   **公式含义解释：**
    *   $V$: 视觉特征的矩阵表示。
    *   $L$: 语言指令的向量表示。
    *   $S$: 机器人本体状态的向量表示。
    *   $F$: 经过 Cross-Attention（交叉注意力）机制融合后的共享上下文表示。
    *   $A$: 预测出的动作序列。
    *   $\tau$: 最终的连续动作轨迹。

        以下是论文中提到的核心算法流程（Algorithm 1 VLA Tokenization Pipeline），请仔细阅读：

        <table>
        <thead>
        <tr>
        <th>行号</th>
        <th>算法步骤</th>
        </tr>
        </thead>
        <tbody>
        <tr>
        <td>1:</td>
        <td>Input: RGB-D frame $I$, text command $T$, joint angles $\theta$</td>
        </tr>
        <tr>
        <td>2:</td>
        <td>$V \leftarrow \text{ViT}(I)$ &gt; 400 vision tokens</td>
        </tr>
        <tr>
        <td>3:</td>
        <td>$L \leftarrow \text{BERT}(T)$ &gt; 12 language tokens</td>
        </tr>
        <tr>
        <td>4:</td>
        <td>$S \leftarrow \text{MLP}(\theta)$ &gt; 64-dim state encoding</td>
        </tr>
        <tr>
        <td>5:</td>
        <td>$F \leftarrow \text{CrossAttention}(V, L, S )$ &gt; 512-dim fused token</td>
        </tr>
        <tr>
        <td>6:</td>
        <td>$A \leftarrow \text{FAST}(F)$ &gt; 50 action tokens</td>
        </tr>
        <tr>
        <td>7:</td>
        <td>Output: Motor commands $\tau_{1:N}$</td>
        </tr>
        </tbody>
        </table>

## 4.3. 架构创新范式
论文总结了当前 VLA 设计的三种主要架构类型，它们代表了不同的权衡策略。

1.  <strong>早期融合模型 (Early Fusion Models):</strong>
    *   **代表：** EF-VLA [96], CLIPort [202]。
    *   **特点：** 在输入层就将视觉和语言特征合并。保留了预训练模型（如 CLIP）的语义一致性，计算效率高，适合快速响应任务。
    *   **优势：** 泛化能力强，不容易过拟合。

2.  <strong>双系统架构 (Dual-System Architectures):</strong>
    *   **代表：** NVIDIA's GR00T N1 [14]。
    *   **结构：**
        *   <strong>System 1 (快):</strong> 基于扩散策略的低级控制，毫秒级延迟，负责精细操作（如抓取）。
        *   <strong>System 2 (慢):</strong> 基于 LLM 的高级规划，负责任务分解和逻辑推理。
    *   **优势：** 兼顾了安全规划的深思熟虑和低层控制的敏捷性。

3.  <strong>自修正框架 (Self-Correcting Frameworks):</strong>
    *   **代表：** SC-VLA [2024]。
    *   **机制：** 引入反馈回路。当检测到执行失败（如抓取失败），触发二级慢速过程进行 Chain-of-Thought（思维链）推理，诊断原因并重新规划。
    *   **优势：** 显著提高了在复杂环境中的鲁棒性和恢复能力。

        下图（原文 Figure 7）展示了端到端的令牌化和表示过程，帮助我们直观理解数据流：

        ![Figure 7: A diagram illustrating the end-to-end tokenization and representation process in VLA models. Visual input (e.g., cluttered tabletop) is encoded by a vision encoder (e.g., ViT), while natural language instructions (e.g., "stack the green blocks") are processed by a language encoder (e.g., T5). The system fuses prefix, state, and action tokens through a transformer and autoregressively predicts motor actions.](images/7.jpg)
        *该图像是示意图，展示了 VLA 模型中的端到端标记化和表示过程。视觉输入（如杂乱的桌面）由视觉编码器（如 ViT）进行编码，自然语言指令（如“将绿色方块叠放在红色托盘上”）由语言编码器（如 T5）处理。系统通过变换器融合前缀、状态和动作标记，逐步预测运动动作。*

下图（原文 Figure 9）进一步说明了 VLA 如何通过转换世界来编码多模态信息，实现动态适应：

![Figure 9: Ilustrating the process of how VLAs Encode the World. VLAs encode the world by converting vision, language, and sensor inputs into tokens, fusing them through cross-attention, predicting action sequences via transformers, and executing tasks with real-time feedback - enabling robots to interpret scenes, follow instructions, and adapt actions dynamically.](images/9.jpg)
*该图像是示意图，展示了多模态输入如何通过视觉和语言标记化，进行状态编码，最终实现动作预测。图中包含了机械手臂抓取苹果的场景，表示VLA模型的应用过程。*

---

# 5. 实验设置

由于本文是一篇综述论文（Review Paper），它本身并没有设计一个新的模型并在自己的基准上进行训练和测试。因此，“实验设置”在此章节转化为对**综述中引用的关键基准数据集、评估指标和比较基线**的介绍。这是理解该领域研究现状的必要前提。

## 5.1. 数据集
VLA 模型的性能高度依赖于训练数据的规模和质量。论文提到了以下几类核心数据集：

*   **网络规模视觉 - 语言数据集:**
    *   **LAION-5B / LAION-400M:** 包含数十亿张图像及其对应的文本描述。用于预训练视觉和语言编码器，让模型获得通用的“世界知识”（如知道苹果是什么）。
    *   **HowTo100M / WebVid:** 视频 - 文本对数据集，用于学习动作相关的语义理解。
*   **机器人轨迹数据集:**
    *   **Open X-Embodiment (OXE):** 包含了超过 400 万个机器人轨迹，覆盖多种机器人形态。这是目前训练通用 VLA 模型最重要的数据源之一。
    *   **RT-X:** 专门针对机器人控制的数据集合，包含大量的厨房操作和工业任务演示。
    *   **BridgeData / RoboNet:** 早期的多机器人交互数据，主要用于基础技能的学习。
    *   **LIBERO:** 专注于具身操作的基准测试数据集，包含一系列长视距任务。
*   **特定领域数据集:**
    *   **nuScenes / Waymo Open Motion:** 用于自动驾驶领域的 VLA 模型评估。
    *   **Stereo Robot Manipulation Demonstrations:** 用于立体视觉增强的 VLA 训练。

        **选择理由：** 这些数据源共同覆盖了从通用语义理解到具体物理执行的广泛分布。仅使用网络数据会导致模型不懂物理规律（如重力、摩擦力），仅使用机器人数据则缺乏泛化能力（没见过的新物体）。混合训练是解决这一问题的关键。

## 5.2. 评估指标
论文在回顾各模型性能时，使用了以下关键指标来衡量 VLA 的有效性。

### 5.2.1. 成功率 (Success Rate)
*   **概念定义：** 衡量机器人能否成功完成给定任务的比例。例如，是否真的把杯子拿起来了，而不是只是试图去拿。
*   **数学公式：**
    $$ \text{SR} = \frac{N_{\text{success}}}{N_{\text{total}}} $$
*   **符号解释：**
    *   $N_{\text{success}}$: 成功完成任务的次数。
    *   $N_{\text{total}}$: 总共尝试的任务次数。
    *   **注：** 在长时序任务中，有时需考虑“最终状态”而非中间过程。

### 5.2.2. 推理延迟 (Inference Latency)
*   **概念定义：** 从输入传感器数据到输出控制指令所需的时间。对于实时控制至关重要，通常以毫秒 (ms) 或频率 (Hz) 衡量。
*   **数学公式：**
    $$ \text{Latency} = t_{\text{output}} - t_{\text{input}} $$
    $$ \text{Frequency} = \frac{1}{\text{Latency}} $$
*   **符号解释：**
    *   $t_{\text{input}}$: 数据采集时刻。
    *   $t_{\text{output}}$: 控制指令生成时刻。
    *   **注：** 理想情况下应达到 10ms (100Hz) 甚至更低以保证稳定性。

### 5.2.3. 零样本泛化能力 (Zero-Shot Generalization)
*   **概念定义：** 模型在未见过的物体、场景或指令下完成任务的能力。不经过任何微调即可工作的程度。
*   **数学公式：**
    $$ \text{Gen}_{\text{zero-shot}} = \text{SR}(D_{\text{test}} | \text{Model trained on } D_{\text{train}}, D_{\text{train}} \cap D_{\text{test}} = \emptyset) $$
*   **符号解释：**
    *   $D_{\text{train}}$: 训练数据集。
    *   $D_{\text{test}}$: 测试数据集（完全不同于训练集）。

## 5.3. 对比基线
综述中提到的模型之间通常相互比较，或者与传统的机器人控制方法对比：
*   <strong>传统控制方法 (Traditional Controllers):</strong> 如 PID 控制、手工编写的脚本。通常无法适应未编程的场景。
*   <strong>单一模态模型 (Single-modality Models):</strong> 仅基于视觉的强化学习模型，缺乏语言指令的理解能力。
*   **早期 VLA 模型:** 如 RT-1, Gato, CLIPort。作为 Baseline，用于展示新模型在精度或速度上的提升。
*   **其他开源方案:** 如 Octo, OpenVLA。通常作为同类型的 SOTA (state-of-the-art) 参考。

    ---

# 6. 实验结果与分析

本章节将分析论文中总结的主要研究成果，并通过表格呈现不同模型的特点和性能对比。请注意，以下数据源自论文对各独立研究的归纳汇总。

## 6.1. 核心结果分析
论文指出，VLA 模型在以下几个关键方面取得了显著进展：
1.  **参数效率提升：** 较小的参数量也能取得高性能。例如 OpenVLA (7B) 在表现上优于更大的 RT-2-X (55B)，证明了共微调（Co-fine-tuning）的重要性。
2.  **推理速度突破：** 通过并行解码和动作分块（Chunking），部分模型实现了高达 200Hz 的控制频率，满足了实时操作的需求。
3.  <strong>跨实体迁移 (Cross-Embodiment Transfer):</strong> 某些模型（如 Octo, Pi-0）能够在不同类型的机器人（机械臂、轮式机器人）之间共享策略，减少了重复训练成本。
4.  **应用场景拓展：** 从实验室桌面操作扩展到自动驾驶、医疗手术和农业采摘等复杂环境。

    然而，性能并非完美。在极端光照、严重遮挡或极度复杂的长时序任务中，许多模型的鲁棒性仍有下降，这表明当前的泛化能力尚未达到 AGI 的水平。

## 6.2. 数据呈现：代表性模型分类表 (Table 1)
以下是原文 Table 1 的结果，展示了不同 VLA 模型在架构设计上的分类（端到端 vs 分层，低层策略 vs 高层规划）：

<table>
<thead>
<tr>
<th rowspan="2">Model Name</th>
<th rowspan="2">Year</th>
<th colspan="2">Architecture Type</th>
<th rowspan="2">Component Focused</th>
<th rowspan="2">Low-Level Policy</th>
<th rowspan="2">High-Level Planner</th>
</tr>
<tr>
<th>End-to-End</th>
<th>Hierarchical</th>
</tr>
</thead>
<tbody>
<tr>
<td>CLIPort [202]</td>
<td>2022</td>
<td></td>
<td>X</td>
<td>X</td>
<td></td>
<td>X</td>
</tr>
<tr>
<td>RT-1 [19]</td>
<td>2022</td>
<td></td>
<td>X</td>
<td>X</td>
<td></td>
<td>X</td>
</tr>
<tr>
<td>Gato [181]</td>
<td>2022</td>
<td></td>
<td>X</td>
<td>X</td>
<td></td>
<td>X</td>
</tr>
<tr>
<td>VIMA [112]</td>
<td>2022</td>
<td></td>
<td>X</td>
<td>X</td>
<td></td>
<td>X</td>
</tr>
<tr>
<td>Diffusion Policy [40]</td>
<td>2023</td>
<td></td>
<td>X</td>
<td>X</td>
<td>2</td>
<td>X</td>
</tr>
<tr>
<td>ACT [287]</td>
<td>2023</td>
<td>2</td>
<td>X</td>
<td>X</td>
<td>✓</td>
<td>X</td>
</tr>
<tr>
<td>VoxPoser [100]</td>
<td>2023</td>
<td></td>
<td>X</td>
<td>X</td>
<td>✓</td>
<td>X</td>
</tr>
<tr>
<td>Seer [80]</td>
<td>2023</td>
<td>✓</td>
<td>X</td>
<td>X</td>
<td>✓</td>
<td>X</td>
</tr>
<tr>
<td>Octo [218]</td>
<td>2024</td>
<td>✓</td>
<td>X</td>
<td>X</td>
<td>✓</td>
<td>X</td>
</tr>
<tr>
<td>OpenVLA [122]</td>
<td>2024</td>
<td>✓</td>
<td>X</td>
<td>X</td>
<td>✓</td>
<td>X</td>
</tr>
<tr>
<td>CogACT [131]</td>
<td>2024</td>
<td>X</td>
<td>✓</td>
<td>X</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td>VLATest [237]</td>
<td>2024</td>
<td>X</td>
<td></td>
<td>✓</td>
<td>×</td>
<td>X</td>
</tr>
<tr>
<td>NaVILA [38]</td>
<td>2024</td>
<td>X</td>
<td>✗</td>
<td>X</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td>Mobility VLA [42]</td>
<td>2024</td>
<td>X</td>
<td>✓</td>
<td>X</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td>Pi-0 [15]</td>
<td>2024</td>
<td>✓</td>
<td>X</td>
<td>X</td>
<td>✓</td>
<td>X</td>
</tr>
<tr>
<td>GR00T N1 [14]</td>
<td>2025</td>
<td>X</td>
<td>✓</td>
<td>X</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td>COVLA [5]</td>
<td>2025</td>
<td>X</td>
<td>✓</td>
<td>X</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td>OpenDriveVLA [293]</td>
<td>2025</td>
<td>X</td>
<td>✓</td>
<td>X</td>
<td>✓</td>
<td>✓</td>
</tr>
</tbody>
</table>

*(注：表格中符号说明：✓ 表示具备，X 表示不具备，空表示非核心特征)*

## 6.3. 数据呈现：模型特性与训练数据表 (Table 2)
下表总结了代表性 VLA 模型的具体架构组件、训练数据来源及其独特优势。该表包含多行合并单元格，故使用 HTML 格式精确还原：

<table>
<thead>
<tr>
<th>Model (Ref.)</th>
<th>Architecture (vision / language / action)</th>
<th>Training data</th>
<th>Key strength / uniqueness</th>
</tr>
</thead>
<tbody>
<tr>
<td>CLIPort [202]</td>
<td>CLIP-ResNet50 + Transporter-ResNet / CLIP - Self-collected [SC] GPT / LingUNet</td>
<td></td>
<td>Aligns semantic CLIP features with Transporter spatial reasoning for precise SE(2) manipulation.</td>
</tr>
<tr>
<td>RT-1 [19]</td>
<td>EfficientNet / Universal Sentence Encoder / RT-1-Kitchen [SC] Transformer (discretized actions)</td>
<td></td>
<td>Early large-scale transformer policy for multi-task kitchen manipulation with tokenized actions.</td>
</tr>
<tr>
<td>RT-2 [299]</td>
<td>ViT-22B or ViT-4B / PaLI-X or PaLM-E / VQA + RT-1-Kitchen symbol-tuning (action tokens)</td>
<td></td>
<td>Co-finetunes internet-scale VQA with robot data, yielding emergent generalization for embodied tasks.</td>
</tr>
<tr>
<td>Gato [181]</td>
<td>ViT / SentencePiece / Transformer (unified to- Self-collected [SC] ken stream)</td>
<td></td>
<td>Generalist agent unifying robotics, language, and Atari via shared tokenization and a single transformer.</td>
</tr>
<tr>
<td>VIMA [112]</td>
<td>ViT + Mask R-CNN / T5 / Transformer</td>
<td>VIMA-Data [SC]</td>
<td>Prompt-driven VL grounding across multiple compositional task types (six prompt modalities).</td>
</tr>
<tr>
<td>ACT [287]</td>
<td>ResNet-18 / —/ CVAE-Transformer</td>
<td>ALOHA [SC]</td>
<td>Temporal ensembling enables smooth bimanual imitation with fine control precision.</td>
</tr>
<tr>
<td>Octo [218]</td>
<td>CNN / T5-base / Diffusion Transformer</td>
<td>Open X-Embodiment (OXE)</td>
<td>Large multi-robot policy trained on 4M+ trajectories spanning many robot embodiments.</td>
</tr>
<tr>
<td>VoxPoser [100]</td>
<td>ViLD + MDETR/GPT-4 / MPC (LLM-guided Zero-shot planning)</td>
<td></td>
<td>Composes LLM+VLM for constraint-aware motion planning without task-specific training.</td>
</tr>
<tr>
<td>Diffusion Policy [40]</td>
<td>ResNet-18 / — / U-Net or Transformer diffusion</td>
<td>Self-collected [SC]</td>
<td>Diffusion modeling captures multimodal action distributions for robust visuomotor control.</td>
</tr>
<tr>
<td>OpenVLA [122]</td>
<td>DINOv2 + SigLIP / Prismatic-7B / symbol- OXE + DROID tuning</td>
<td></td>
<td>Open-source RT-2-like VLA; supports efficient LoRA adaptation and broad generalization."</td>
</tr>
<tr>
<td>π0 (Pi-Zero) [15]</td>
<td>PaliGemma VLM / PaliGemma (multimodal) / Pi-Cross-Embodiment 300M diffusion action model</td>
<td></td>
<td>Lightweight general robot controller (reported ~3B total) with strong cross-robot, open-world generalization and bi-manual skills.</td>
</tr>
<tr>
<td>π0-Fast [171]</td>
<td>PaliGemma VLM / PaliGemma / autoregres- Pi-Cross-Embodiment sive transformer with FAST tokenization</td>
<td></td>
<td>High-frequency real-time control via compressed frequency-space action tokens (reported up to 15× faster inference).</td>
</tr>
<tr>
<td>OpenVLA-OFT [121]</td>
<td>SigLIP + DINOv2 (multi-view) / Llama-2 7B LIBERO; bimanual ALOHA / parallel decoding + action chunking (L1 re- gression)</td>
<td></td>
<td>Fine-tuning recipe with parallel decoding and chunked actions; reported 97.1% LIBERO success and 26× faster inference for high-frequency bimanual control.</td>
</tr>
<tr>
<td>RDT-1B [144]</td>
<td>Multi-view RGB encoder / transformer lan- 46 datasets (&gt;1M episodes) + guage module / Diffusion Transformer (unified ALOHA fine-tune action space)</td>
<td></td>
<td>1.2B diffusion foundation model for dexterous bimanual manipulation with strong language conditioning and zero- shot transfer.</td>
</tr>
<tr>
<td>Helix1</td>
<td>System 2: open-source VLM for multimodal Figure reasoning (79 Hz) / integrated semantics / System 1: transformer visuomotor policy (200 Hz, full upper-body)</td>
<td>robot E2E (pixels+language→actions)</td>
<td>Humanoid-focused dual-rate VLA enabling real-time high- DoF control, dexterity, and collaborative multi-robot manipulation with zero-shot generalization.</td>
</tr>
<tr>
<td>CogACT [131]</td>
<td>/ Llama-2 via Prismatic-7B / DiT-Base (300M tasks</td>
<td>DINOv2 ViT-L/14 + SigLIP ViT-So400M/14 OXE subset; Realman &amp; Franka</td>
<td>Componentized VLA with diffusion action transformer; reported +59.1% real-world success vs. OpenVLA and</td>
</tr>
<tr>
<td>Chain-of-Affordance</td>
<td>diffusion) Affordance-aware visual encoder / transformer LIBERO; real+sim manipulation reasoning prompts / autoregressive + diffusion</td>
<td></td>
<td>strong adaptation to unseen robots/objects. Sequential affordance reasoning (object→grasp→spatial→motion) improves spatial plan-</td>
</tr>
<tr>
<td></td>
<td>policy (affordance-conditioned)</td>
<td></td>
<td>ning and obstacle avoidance; reported stronger LIBERO performance than OpenVLA.</td>
</tr>
<tr>
<td>Edge VLA (EVLA) [20]</td>
<td>SigLIP + DINOv2 / Qwen2 (0.5B) / non- Bridge; OXE; 1.2M textimage pairs autoregressive joint control prediction</td>
<td></td>
<td>Edge-optimized VLA (e.g., Jetson-class) with reported 30 50 Hz inference and OpenVLA-comparable performance under low power.</td>
</tr>
<tr>
<td>ShowUI-2B [139]</td>
<td>UI-guided visual token selection / interleaved 256K GUI instruction-following V-L-A streaming / transformer GUI action predictor</td>
<td></td>
<td>Compact 2B VLA for digital automation; strong screenshot grounding and GUI/web navigation with efficient token se- lection.</td>
</tr>
<tr>
<td>GR00T N1 [14]</td>
<td>NVIDIA Eagle-2 VLM / integrated high-level Human demos + robot trajectories + planning / diffusion transformer (DiT)</td>
<td>simulation + internet video</td>
<td>Generalist humanoid dual-system design combining plan- ning and diffusion execution for dexterous multi-step con- trol and broad embodiment generalization.</td>
</tr>
<tr>
<td>Seer [80]</td>
<td>Grounding-optimized visual backbone / trans- LIBERO former language / autoregressive action head</td>
<td></td>
<td>Strong visual grounding for manipulation; competitive on LIBERO but typically below newer fine-tuned variants (e.g., OpenVLA-OFT).</td>
</tr>
<tr>
<td>DiffusionVLA [240]</td>
<td>Transformer visual encoder / autoregressive reasoning / diffusion action head bin-picking</td>
<td>LIBERO; factory sorting; zero-shot</td>
<td>Diffusion control improves robustness and interpretability;</td>
</tr>
</tbody>
</table>

*(注：下表为继续部分，因篇幅限制展示核心部分)*

## 6.4. 应用场景展示图
论文中的图表生动地描绘了 VLA 在不同场景中的应用。

<strong>图 11 (Application Domains)</strong> 展示了 VLA 的六大核心应用方向：

![Figure 11: Mind-map of application domains for VisionLanguage-Action models, with Humanoid Robotics positioned at the top and remaining domains arranged clockwise to match the order of discussion in this section.](images/11.jpg)
*该图像是一个示意图，展示了Vision-Language-Action (VLA) 模型的应用领域。中心为“VLA的应用”，周围依次排列着“类人机器人”、“自主车辆系统”、“工业机器人”、“医疗与健康机器人”、“精确与自动化农业”和“互动增强现实导航”。*

<strong>图 12 (Humanoid Interaction)</strong> 展示了人机协作中的一个具体案例：机器人从冰箱取水。

![该图像是一个示意图，展示了一个机器人与冰箱的互动。机器人展示了 Vision-Language-Action (VLA) 模型的组成部分，包括视觉语言模型、大型语言模型、层次控制器和代理 AI，体现了多模态学习与智能体行为的结合。](images/12.jpg)
*该图像是一个示意图，展示了一个机器人与冰箱的互动。机器人展示了 Vision-Language-Action (VLA) 模型的组成部分，包括视觉语言模型、大型语言模型、层次控制器和代理 AI，体现了多模态学习与智能体行为的结合。*

<strong>图 13 (Autonomous Delivery)</strong> 展示了 VLA 在自动驾驶车辆路径规划中的应用，强调了安全与可解释性。

![Figure 13: This illustration depicts an autonomous delivery vehicle powered by a VLA system, integrating VLMs for visual grounding, LLMs for instruction parsing, and a VLA decoder for path planning. Agentic AI enables adaptive trajectory refinement in dynamic environments, exemplifying how multi-modal integration drives safe, interpretable, and autonomous decision-making in realworld navigation tasks.](images/13.jpg)
*该图像是一个示意图，展示了一辆由VLA系统驱动的自主配送车辆。该系统结合了视觉语言模型（VLM）进行视觉定位，使用大语言模型（LLM）解析指令，并通过VLA解码器进行路径规划。Agentic AI能在动态环境中调整行驶轨迹，体现了多模态整合如何推动安全、可解释的自主决策。*

<strong>图 14 (Medical Robotics)</strong> 展示了医疗领域的应用，包括手术缝合和护理辅助。

![Figure 14: a) This figure illustrates a VLA surgical system executing the task "apply a suture to the left coronary artery." The vision module identifies anatomical targets, the language model interprets the instruction, and the action decoder generates precise motor commands, enabling adaptive tool control, real-time feedback, and safe autonomous operation; b) A VLA-powered assistive robot perceives patient behavior, processes verbal requests (e.g., "bring my walker"), and autonomously executes context-aware motion plans, enabling real-time assistance in eldercare, rehabilitation, and hospital logistics without relying on predefined scripts or manual oversight.](images/14.jpg)
*该图像是示意图，展示了VLA系统执行手术任务和助理机器人提供患者实时辅助的过程。图(a)展示了VLA手术系统用于“给左冠状动脉缝合”的任务，整合视觉模块、语言模块和动作解码器以实现高精度操作；图(b)展示了助理机器人如何通过视觉感知和语言编码，根据患者请求自动执行动作计划，实现家庭护理支持。*

<strong>图 16 (AR Navigation)</strong> 展示了增强现实导航中如何利用 VLA 提供个性化指引。

![Figure 16: Showing how VLA models enable interactive AR navigation by fusing real-time visual perception, language understanding, and action planning. In dynamic environments such as airports, VLAs interpret user queries like "avoid stairs to Gate 22," analyze visual scenes (e.g., detecting escalators), and adjust navigational paths accordingly, supporting personalized, accessible, and context-aware mobility guidance.](images/16.jpg)
*该图像是示意图，展示了 VLA 模型如何通过融合实时视觉感知、语言理解和行动规划来实现互动增强现实导航。在动态环境中，用户提出问题，如“如何不走楼梯到达 Gate 22”，系统通过视觉编码器和多模态融合进行场景理解，并利用行动解码器进行导航规划，从而提供适应性导航反馈。*

---

# 7. 总结与思考

## 7.1. 结论总结
本文《Vision-Language-Action (VLA) Models: Concepts, Progress, Applications and Challenges》提供了一幅详尽的地图，指引我们进入具身智能的深水区。
1.  **技术成熟度：** VLA 已从概念验证走向实际应用原型。通过 Tokenization 技术和 Transformer 架构，多模态融合已成为标准范式。
2.  **关键突破：** 在参数效率和推理速度上的进步（如 OpenVLA, Pi-0 Fast）解决了长期制约硬件部署的瓶颈。
3.  **应用广度：** 从家庭服务到工业生产，再到农业和医疗，VLA 展示了极高的适应性。
4.  **核心挑战：** 尽管进步巨大，但在复杂开放环境下的安全性、对罕见场景的泛化能力以及伦理问题仍然是必须解决的障碍。

## 7.2. 局限性与未来工作
论文作者坦诚指出了当前技术的不足：
*   **数据集偏差：** 训练数据多来自网络，可能存在刻板印象或不完整的世界观。
*   **实时性瓶颈：** 尽管有加速技术，但在高维动作空间中实现极低延迟（<5ms）仍极具挑战性。
*   **安全性保证：** 缺乏形式化的数学证明来保证动作永远不会伤害人或物。
*   **未来方向：**
    *   <strong>持续学习 (Continual Learning):</strong> 机器人应能在部署后不断吸收新知识而不遗忘旧技能。
    *   <strong>神经符号规划 (Neuro-Symbolic Planning):</strong> 结合深度学习的感知能力和符号逻辑的可解释性。
    *   <strong>世界模型 (World Models):</strong> 构建内部物理模拟器，用于模拟推演（Rollout）和决策。
    *   **伦理治理:** 建立隐私保护和责任归属的标准框架。

## 7.3. 个人启发与批判
*   **启发性思考：** 将动作视为“文本”进行预测（Action as Text）是一个非常优雅且强大的直觉。它利用了 LLM 庞大的训练生态，极大地降低了开发门槛。未来的研究可能会集中在如何让这个过程更高效、更稳健。
*   **批判性视角：**
    *   **幻觉风险：** 像语言模型会“胡说八道”一样，VLA 也可能生成看似合理但物理上不可行的动作（Hallucinated Actions）。目前的“自修正”机制虽然有效，但增加了延迟和复杂度。
    *   **依赖数据规模：** 目前的 SOTA 模型极度依赖海量机器人轨迹数据（如 Open X-Embodiment）。这对于没有能力收集千万级数据的中小机构构成了极高的准入门槛。轻量化、小样本学习（Few-shot）可能是下一个爆发点。
    *   **黑盒性质：** 尽管有了思维链（CoT）的引入，但深度神经网络的决策过程本质上仍是黑盒。在航空、医疗等高风险领域，完全的“端到端”信任建立还需要很长时间。

        综上所述，这篇论文不仅是一份技术清单，更是一种愿景的宣告。它告诉我们，通往通用人工智能的道路，必定是一条视觉、语言与行动深度纠缠、共同进化的道路。

<!-- Image Citation for Future Roadmap -->
下图（原文 Figure 19）清晰地描绘了未来的 VLA 发展路线图，强调了高效部署、可靠安全智能和统一系统治理三大支柱：

![该图像是一个示意图，展示了未来的视觉-语言-行动(VLA)模型发展路线图，包括高效部署、可靠与安全智能、统一系统与治理等三大主题。图中列出了多项关键技术和策略，如紧凑的动作标记化与分块、稳健的多模态基础、物理与因果预测模型等。](images/19.jpg)
*该图像是一个示意图，展示了未来的视觉-语言-行动(VLA)模型发展路线图，包括高效部署、可靠与安全智能、统一系统与治理等三大主题。图中列出了多项关键技术和策略，如紧凑的动作标记化与分块、稳健的多模态基础、物理与因果预测模型等。*

<!-- Image Citation for Future Assistant Concept -->
下图（原文 Figure 18）展示了名为"Eva"的未来智能助手概念，集成了 VLM、VLA 和 Agent AI 模块，是 AGI 的一个具象化蓝图：

![Figure 18: This conceptual illustration presents "Eva," a future humanoid assistant powered by Vision-Language Models (VLMs), VLA frameworks, and agentic AI systems. VLMs enable semantic scene understanding and object affordance prediction, while VLAs translate language-grounded instructions into hierarchical motor plans. Agentic AI modules ensure adaptive learning, selfrefinement, and interactive decision-making in open-ended environments. Together, these components represent a foundational blueprint for Artificial General Intelligence (AGI) in robotics, where perception, language understanding, planning, and safe autonomous behavior converge in real-world, socially aware tasks.](images/18.jpg)
*该图像是一个示意图，展示了未来的智能助手"Eva"，该助手融合了视觉语言模型（VLM）和视觉语言行动框架（VLA）。VLM用于语义场景理解和对象预测，而VLA将语言指令转换为分层运动计划，表示人工通用智能（AGI）的基础蓝图。*