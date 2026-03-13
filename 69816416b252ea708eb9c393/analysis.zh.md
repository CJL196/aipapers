# 1. 论文基本信息

## 1.1. 标题
DynamicVLA: 一个用于动态物体操作的视觉-语言-动作模型 (DynamicVLA: A Vision-Language-Action Model for Dynamic Object Manipulation)。

## 1.2. 作者
本文由来自新加坡南洋理工大学 S-Lab 的 Haozhe Xie\*、Beichen Wen\*、Jiarui Zheng、Zhaoxi Chen、Fangzhong Hong、Haiwen Diao 和 Ziwei Liu 共同完成。两位带星号的作者贡献相同。

## 1.3. 发表期刊/会议
本文作为预印本发表于 arXiv 平台。

## 1.4. 发表年份
2026年1月29日 (UTC)。

## 1.5. 摘要
尽管在静态操作中表现出强大的泛化能力，但视觉-语言-动作 (Vision-Language-Action, VLA) 模型在需要快速感知、时间预测和连续控制的动态场景中仍面临挑战。本文提出了 `DynamicVLA`，一个用于动态物体操作的框架，它通过三项关键设计整合了时间推理和闭环适应：1) 一个紧凑的 0.4B VLA 模型，使用卷积视觉编码器进行空间高效、结构忠实的编码，从而实现快速的多模态推理；2) `Continuous Inference`（连续推理），通过重叠推理和执行来降低延迟并及时适应物体运动；3) `Latent-aware Action Streaming`（潜在感知动作流），通过强制执行时间对齐的动作来弥合感知-执行差距。为了弥补动态操作数据基础的缺失，我们从头构建了 `Dynamic Object Manipulation (DOM)` 基准，其自动化数据收集管道高效地收集了跨 2.8K 场景和 206 个物体的 200K 合成回合，并实现了无需遥操作即可快速收集 2K 真实世界回合。广泛的评估表明，在响应速度、感知和泛化方面取得了显著改进，使 `DynamicVLA` 成为一个适用于跨不同机器人形态的通用动态物体操作的统一框架。

## 1.6. 原文链接
原文链接：https://arxiv.org/abs/2601.22153
PDF 链接：https://arxiv.org/pdf/2601.22153v1
发布状态：预印本 (arXiv)。

# 2. 整体概括

## 2.1. 研究背景与动机
### 2.1.1. 研究背景
机器人操作的传统研究主要集中在<strong>静态场景 (static settings)</strong>，即物体在操作过程中保持固定状态。然而，<strong>真实世界交互 (real-world interaction)</strong> 中常常涉及处于运动状态的物体，例如递送物品、重新定位或稳定物品。这要求机器人不仅能感知，还需要<strong>预测 (predict)</strong> 物体的未来运动并<strong>在快速变化的条件下采取行动 (act under rapidly changing conditions)</strong>。

### 2.1.2. 核心问题与挑战
当前的<strong>视觉-语言-动作 (Vision-Language-Action, VLA) 模型</strong>虽然在静态操作中表现出强大的<strong>泛化能力 (generalization)</strong>，但在处理<strong>动态物体操作 (dynamic object manipulation)</strong> 时却面临严峻挑战。主要问题体现在：
*   <strong>推理延迟 (Inference Latency):</strong> 现有 VLA 模型（如早期的 `3B-7B` 大型模型）的推理速度通常较慢，这在动态场景中是致命的。即使是轻微的延迟也可能导致<strong>任务失败 (task failure)</strong>，因为物体在模型推理期间会持续移动，使得感知与执行不同步。
*   <strong>缺乏时间推理 (Lack of Temporal Reasoning):</strong> 传统 VLA 模型通常假设物体状态在推理期间是固定的，因此不具备<strong>时间预测 (temporal anticipation)</strong> 的能力，难以预测物体的未来运动。
*   <strong>缺乏连续控制 (Lack of Continuous Control):</strong> 动态操作需要机器人进行<strong>连续的、闭环的控制 (continuous, closed-loop control)</strong>，以实时适应物体运动。
*   <strong>数据集稀缺 (Data Scarcity):</strong> 现有的机器人数据集绝大多数都捕获静态场景，缺乏大规模的动态物体操作数据，这阻碍了专门针对动态场景的 VLA 策略的训练和评估。
*   <strong>遥操作的局限性 (Limitations of Teleoperation):</strong> 对于快速移动的物体，人类的反应速度往往无法跟上，使得传统的遥操作数据收集方式对于动态操作任务是无效的。

### 2.1.3. 创新思路与切入点
为了解决上述挑战，本文提出了 `DynamicVLA` 框架，其核心创新思路是：
1.  **低延迟架构设计:** 构建一个紧凑且高效的 VLA 模型，以显著减少推理延迟。
2.  **时间感知与适应机制:** 引入新的执行机制，使推理和动作执行能够重叠，并确保在存在延迟的情况下动作与环境保持时间对齐。
3.  **大规模动态数据集:** 从头构建一个专门用于动态物体操作的基准，并通过自动化管道高效收集模拟和真实世界数据。

## 2.2. 核心贡献/主要发现
本文的主要贡献总结如下：
1.  **提出 `DynamicVLA` 框架:** `DynamicVLA` 是一个专门为动态物体操作设计的统一框架，它集成了<strong>时间推理 (temporal reasoning)</strong> 和<strong>闭环适应 (closed-loop adaptation)</strong> 能力。
2.  **紧凑型 0.4B VLA 模型:** 设计了一个参数量仅为 0.4B 的紧凑型 VLA 模型。该模型采用<strong>卷积视觉编码器 (convolutional vision encoder)</strong> `FastViT`，实现了高效的<strong>空间压缩 (spatial compression)</strong> 和更强的<strong>结构保留 (structural preservation)</strong>，从而支持快速的<strong>多模态推理 (multimodal inference)</strong>。
3.  <strong>`Continuous Inference`（连续推理）机制:</strong> 引入了一种流水线式执行方案，允许<strong>推理 (prediction)</strong> 和<strong>动作执行 (action execution)</strong> 重叠进行。这消除了<strong>块间等待 (inter-chunk waiting)</strong>，显著降低了<strong>延迟 (latency)</strong>，并能在动态物体运动下维持<strong>连续的动作流 (continuous action stream)</strong>。
4.  <strong>`Latent-aware Action Streaming`（潜在感知动作流）机制:</strong> 提出了一种<strong>延迟感知 (latency-aware)</strong> 的执行机制，通过丢弃过时动作并优先处理每个时间步最新预测的动作来恢复<strong>时间对齐 (temporal alignment)</strong>，从而弥合了感知-执行之间的差距。
5.  <strong>构建 `Dynamic Object Manipulation (DOM)` 基准:</strong> 从零开始构建了第一个大规模的动态物体操作基准。该基准通过一个<strong>自动化数据收集管道 (auto data collection pipeline)</strong>，高效地收集了跨 2.8K 场景和 206 个物体的 200K 合成回合，并实现了无需遥操作即可快速收集 2K 真实世界回合。
6.  **显著的实验结果:** 在广泛的评估中，`DynamicVLA` 在<strong>响应速度 (response speed)</strong>、<strong>感知 (perception)</strong> 和<strong>泛化 (generalization)</strong> 方面表现出显著改进，验证了其作为通用动态物体操作统一框架的有效性。

# 3. 预备知识与相关工作

## 3.1. 基础概念
为了充分理解 `DynamicVLA` 的工作原理及其在动态物体操作中的优势，首先需要了解以下核心概念：

### 3.1.1. 视觉-语言-动作模型 (Vision-Language-Action, VLA Model)
**概念定义:** `VLA` 模型是结合了<strong>计算机视觉 (Computer Vision)</strong>、<strong>自然语言处理 (Natural Language Processing)</strong> 和<strong>机器人控制 (Robotics Control)</strong> 的多模态人工智能模型。它们旨在让机器人能够理解人类的自然语言指令，通过视觉感知周围环境，并根据理解和感知生成相应的物理动作来完成任务。
**在本文中的意义:** `VLA` 模型在静态操作中已取得显著进展，但其核心挑战是在动态环境中如何保持低延迟和实时适应性。

### 3.1.2. 动态物体操作 (Dynamic Object Manipulation)
**概念定义:** 指机器人对处于<strong>运动状态 (in motion)</strong> 的物体进行感知、预测和执行抓取、放置、推拉等操作的任务。与传统的静态物体操作不同，动态操作要求机器人能够处理<strong>不确定运动 (uncertain motion)</strong>、<strong>精确接触 (precise contact)</strong> 和<strong>紧密的感知-动作对齐 (tight perception-action alignment)</strong>。
**在本文中的意义:** 这是本文的核心研究问题，也是现有 `VLA` 模型面临的主要挑战，因为其对时间敏感性要求极高。

### 3.1.3. 推理延迟 (Inference Latency)
**概念定义:** 指模型从接收输入（如图像帧、语言指令）到生成输出（如机器人动作指令）所需的时间间隔。
**在本文中的意义:** 在动态物体操作中，推理延迟是导致任务失败的关键因素，因为它会造成机器人的感知与环境的真实状态之间的时间错位。

### 3.1.4. 闭环适应 (Closed-loop Adaptation)
**概念定义:** 指机器人系统能够根据实时从环境中获取的反馈信息（例如物体位置、速度、形状等的变化）动态调整其行为和策略的能力。这种反馈回路使得机器人能够对环境的变化做出及时响应，而不是简单地执行预设的动作序列。
**在本文中的意义:** 动态操作场景是不断变化的，`DynamicVLA` 需要通过闭环适应来实时调整策略，以成功完成任务。

### 3.1.5. 扩散模型 (Diffusion Model)
**概念定义:** 扩散模型是一类<strong>生成模型 (Generative Models)</strong>，通过学习从数据中逐步去除噪声来生成新的、高质量的数据样本。其核心思想是，首先向真实数据中逐步添加高斯噪声，直到数据完全变为随机噪声；然后训练一个神经网络来学习逆过程，即从噪声中逐步恢复出原始数据。在生成任务时，模型从纯噪声开始，通过学习到的去噪步骤生成新的数据。
**在本文中的意义:** `DynamicVLA` 的<strong>动作专家 (Action Expert)</strong> 部分采用了<strong>基于扩散的动作建模 (diffusion-style action modeling)</strong>，将其实例化为<strong>条件流匹配 Transformer (conditional Flow Matching Transformer)</strong>，用于预测机器人动作序列。

### 3.1.6. 卷积视觉编码器 (Convolutional Vision Encoder)
**概念定义:** 一种使用<strong>卷积神经网络 (Convolutional Neural Networks, CNNs)</strong> 作为其主要构建模块的图像特征提取器。与基于 `Transformer` 的视觉编码器（如 `ViT`）不同，卷积网络通过<strong>局部感受野 (local receptive fields)</strong> 和<strong>权值共享 (weight sharing)</strong> 来提取图像特征，通常在处理空间信息和局部纹理方面具有优势，并能有效减少<strong>词元数量 (token count)</strong>。
**在本文中的意义:** `DynamicVLA` 采用卷积视觉编码器 `FastViT` 来实现高效的<strong>空间压缩 (spatial compression)</strong>，避免 `Transformer` 编码器中可能出现的<strong>词元数量二次增长 (quadratic token growth)</strong> 问题，从而降低计算成本和推理延迟。

## 3.2. 前人工作与技术演进
### 3.2.1. 视觉-语言-动作模型 (Vision-Language-Action Models) 的演进
*   **早期 `VLA` 模型:** 受<strong>大型语言模型 (Large Language Models, LLMs)</strong> 和<strong>视觉-语言模型 (Vision-Language Models, VLMs)</strong> 成功的启发，`VLA` 模型将 `VLM` 扩展到动作生成。
    *   **基于 Transformer 的方法:** 如 `RT-1` [7]，使用 `Transformer` 建模状态-动作-奖励序列。
    *   **基于 `LLM/VLM` 的方法:** 如 `OpenVLA` [17] 和 $\pi_0$ [6]，将 `VLA` 任务视为序列到序列的问题进行动作生成。
    *   **基于扩散模型的方法:** 如 `Diffusion Policy` [9] 和 $\pi_{0.5}$ [15]，将策略建模为去噪扩散模型。
    *   **`LLM` 与扩散模型结合:** 如 `Octo` [14]，结合 `LLM` 用于表示和扩散模型用于动作生成。
    *   **视频生成与逆运动学:** 如 `RoboGen` [47] 和 `VideoWorld` [49]，生成运动序列并转换为动作。
*   **挑战:** 这些现有 `VLA` 模型普遍存在<strong>推理速度慢 (slow inference speeds)</strong> 的问题，这严重限制了它们在需要精确或快速执行场景中的应用。
*   **轻量化和高效性改进:** 针对效率问题，一些近期工作如 `SmolVLA` [38] 和 `VLA-Adapter-Pro` [46] 尝试通过减小模型尺寸和提高吞吐量来提升效率。然而，它们主要关注<strong>静态操作 (static manipulation)</strong>，延迟在这些场景中影响较小。

### 3.2.2. 机器人学习数据集 (Robot Learning Datasets)
*   **现有数据集特点:**
    *   **真实世界数据集:** 如 `ROBOTURK` [28] 和 `Open X-Embodiment` [35]，提供高保真交互数据，但收集成本高且难以扩展。
    *   **模拟数据集:** 如 `CALVIN` [30] 和 `BEHAVIOR-1K` [20]，具有可扩展性，但存在<strong>模拟到真实 (sim-to-real) 差距</strong>。
    *   **任务范围:** 大多数基准关注简单的桌面操作（如抓取放置、推），任务多样性有限，尽管有工作探索了<strong>长时程 (long-horizon)</strong> [24]、<strong>语言条件 (language-conditioned)</strong> [58] 和<strong>触觉丰富 (tactile-rich)</strong> [52] 的设置。
*   **挑战:** 现有数据集普遍缺乏<strong>动态对象 (dynamic objects)</strong>，这限制了它们在涉及独立运动环境中的适用性。

### 3.2.3. 机器人动态操作 (Robot Dynamic Manipulation)
*   **早期方法:** 大部分机器人操作研究集中于静态设置。针对移动对象的方法通常是<strong>任务特定的 (task-specific)</strong> 或依赖于<strong>可预测的运动 (predictable motion)</strong>。
    *   例如，`DBC-TFP` [56] 和 `GEM` [22] 主要在结构化的、类似传送带的场景中运行。
*   **并发 `VLA` 方法:** `RDT-2` [43]、`RTVLA` [27] 和 `VLASH` [39] 展示了与快速移动目标实时交互的能力。
*   **挑战:** 这些方法通常允许较大的<strong>接触裕度 (contact margins)</strong>，不涉及精确的 <strong>6 自由度 (6DoF) 操作</strong>。因此，在不确定运动和精细接触约束下的<strong>通用动态操作 (general dynamic manipulation)</strong> 仍未得到充分探索。

## 3.3. 差异化分析
`DynamicVLA` 与现有工作的主要区别和创新点在于：
1.  **专注于动态操作中的低延迟:** `DynamicVLA` 明确将推理延迟视为动态操作的<strong>主要失败模式 (dominant failure mode)</strong>。它通过设计一个**紧凑的 0.4B VLA 模型**并采用**卷积视觉编码器**来提高推理速度，这与许多依赖大型 `Transformer` 视觉编码器和语言主干网络的 `VLA` 模型形成对比。
2.  **创新的执行机制:** 引入了 `Continuous Inference`（连续推理）和 `Latent-aware Action Streaming`（潜在感知动作流）这两个独特的执行机制。
    *   `Continuous Inference` 通过**重叠推理与执行**来消除**块间等待**，确保动作流的连续性。
    *   `Latent-aware Action Streaming` 通过**丢弃过时动作**和**优先处理最新预测**来解决推理延迟引起的时间错位问题。
        这些机制是专门为动态场景中的**闭环适应**和<strong>实时响应性 (real-time responsiveness)</strong> 设计的，是现有 `VLA` 模型中未被充分解决的关键问题。
3.  **大规模动态操作数据集:** 从头构建了 `Dynamic Object Manipulation (DOM)` 基准。这是第一个专门针对动态物体操作的大规模数据集，解决了该领域数据稀缺的问题，并且其自动化数据收集管道（包括真实世界数据的无遥操作收集）本身就是一项重要的贡献。
4.  **统一框架和泛化能力:** `DynamicVLA` 被定位为一个统一框架，旨在实现跨<strong>机器人形态 (embodiments)</strong> 的通用动态物体操作，并在响应速度、感知和泛化方面取得了显著的改进，特别是在处理不确定运动和精细接触约束方面。

# 4. 方法论

## 4.1. 方法原理
`DynamicVLA` 的核心思想是，在动态物体操作中，<strong>推理延迟 (inference latency)</strong> 是导致任务失败的主要原因，因为它会导致机器人<strong>感知 (perception)</strong> 和<strong>动作执行 (action execution)</strong> 之间的时间错位。为了解决这个问题，`DynamicVLA` 从三个层面进行创新设计：
1.  **架构层面:** 设计一个**紧凑、高效**的 `VLA` 模型，以最大限度地减少单次推理的耗时，从而支持<strong>高频率推理 (high-frequency reasoning)</strong>。
2.  **推理执行层面:** 引入<strong>流水线化 (pipelined)</strong> 的执行方案，使得模型的推理过程与机器人的动作执行过程能够<strong>重叠 (overlap)</strong>，从而消除传统串行执行中的等待时间，保持动作的连续性。
3.  **时间对齐层面:** 提出一种<strong>延迟感知 (latency-aware)</strong> 的策略，主动管理和调整生成的动作序列，确保即使存在推理延迟，机器人执行的动作也能与环境的<strong>最新状态 (most recent environment state)</strong> 保持<strong>时间对齐 (temporally aligned)</strong>。
    同时，为了克服现有数据集中缺乏动态操作数据的限制，`DynamicVLA` 还构建了一个**大规模、自动化**的动态物体操作基准 `DOM`，为模型的训练和评估提供了坚实的基础。

## 4.2. 核心方法详解 (逐层深入)

### 4.2.1. A. 问题表述 (Problem Formulation)
在动态物体操作中，机器人必须操纵在感知、推理和执行过程中状态持续变化的物体。
在时间步 $t$，`VLA` 模型 $\mathcal{M}$ 接收：
*   <strong>视觉观测窗口 (temporal window of visual observations):</strong> $\mathbf{O}_t = \{ \mathbf{o}_{t-k}, \dots, \mathbf{o}_t \}$，其中 $\mathbf{o}_t$ 代表当前时间步的视觉观测。
*   <strong>语言指令 (language instruction):</strong> $\mathbf{L}_t$。
*   <strong>本体感知状态 (proprioceptive state):</strong> $\mathbf{P}_t$，即机器人自身的关节位置、速度等信息。

    模型 $\mathcal{M}$ 的目标是预测一个动作序列 $\mathbf{A}_t = \{ \mathbf{a}_t, \dots, \mathbf{a}_{t+n} \}$，表示为：
$$
\mathbf{A}_t = \mathcal{M}(\mathbf{O}_t, \mathbf{L}_t, \mathbf{P}_t)
$$
**符号解释:**
*   $\mathbf{A}_t$: 在时间步 $t$ 预测的动作序列。
*   $\mathcal{M}$: `VLA` 模型。
*   $\mathbf{O}_t$: 在时间步 $t$ 观察到的视觉观测序列。
*   $\mathbf{L}_t$: 在时间步 $t$ 给定的语言指令。
*   $\mathbf{P}_t$: 在时间步 $t$ 机器人的本体感知状态。
*   $\mathbf{o}_{t-k}, \dots, \mathbf{o}_t$: 视觉观测序列中的各个帧，涵盖从 `t-k` 到 $t$ 的时间范围。
*   $\mathbf{a}_t, \dots, \mathbf{a}_{t+n}$: 预测的动作序列中的各个动作，涵盖从 $t$ 到 $t+n$ 的时间范围。

    **关键挑战:** 物理环境包括一个<strong>潜在物体状态 (latent object state)</strong> $\mathbf{s}_t$，描述物体的 6D 姿态和运动。关键在于，物体运动在推理过程中不会暂停。当模型对 $\mathbf{O}_t$ 进行推理时，物体会从 $\mathbf{s}_t$ 运动到 $\mathbf{s}_{t+m}$，其中 $m$ 表示<strong>推理延迟 (inference latency)</strong>。这种延迟导致了感知与执行之间潜在的<strong>错位 (misalignment)</strong>。

### 4.2.2. B. DynamicVLA 架构 (The DynamicVLA Architecture)
由于推理延迟直接限制了动态操作中物体运动的范围，`DynamicVLA` 设计了一个紧凑的 0.4B 参数 `VLA` 模型，用于快速且空间高效的多模态推理。其架构如以下 Figure 2a 所示：

![该图像是示意图，展示了DynamicVLA框架的架构与关键设计。图中包括三个主要部分：第一部分(a)显示了DynamicVLA架构，其中包含SmolLM2-360M模型和Action Expert；第二部分(b)介绍了连续推理的流程，强调推理循环与执行循环的关系；第三部分(c)展示了潜在感知动作流，描述了输入流与动作流的关系。这些设计旨在实现动态物体的高效操控和适应。](images/2.jpg)  
*该图像是示意图，展示了DynamicVLA框架的架构与关键设计。图中包括三个主要部分：第一部分(a)显示了DynamicVLA架构，其中包含SmolLM2-360M模型和Action Expert；第二部分(b)介绍了连续推理的流程，强调推理循环与执行循环的关系；第三部分(c)展示了潜在感知动作流，描述了输入流与动作流的关系。这些设计旨在实现动态物体的高效操控和适应。*

图示 2a：DynamicVLA 架构，包含 SmolLM2-360M 模型和 Action Expert。

#### 4.2.2.1. 视觉-语言主干网络 (Vision-Language Backbone)
*   <strong>语言主干网络 (Language Backbone):</strong> 采用 `SmolLM2-360M` [3] 作为语言主干网络，这使得整体模型尺寸非常小巧。为了显著降低推理延迟，同时对多模态推理影响最小，语言主干网络被截断到其前 16 个 `Transformer` 层，遵循 `SmolVLA` [38] 的实践。
*   <strong>视觉编码器 (Vision Encoder):</strong> 与现有 `VLM` 中依赖基于 `Transformer` 的视觉编码器不同，`DynamicVLA` 采用<strong>卷积视觉编码器 (convolutional vision encoder)</strong> `FastViT` [44]。
    *   `FastViT` 能够执行<strong>高效的空间压缩 (efficient spatial compression)</strong>。
    *   它能更好地<strong>保留结构 (structurally faithful encoding)</strong>。
    *   避免了处理多帧视觉输入时<strong>词元数量的二次增长 (quadratic token growth)</strong> 问题，从而实现更快的推理。

#### 4.2.2.2. 基于扩散的动作专家 (Diffusion-Based Action Expert)
*   **功能:** 动作专家 $\mathcal{E}_{\theta}$ 负责预测一个动作块 $\mathbf{A}_t$，该动作块以 `VLM` 主干网络产生的多模态特征为条件。
*   **实现:** 遵循<strong>扩散风格的动作建模 (diffusion-style action modeling)</strong> 方法 [23, 12]，$\mathcal{E}_{\theta}$ 被实例化为一个<strong>条件流匹配 Transformer (conditional Flow Matching Transformer)</strong> [6]。
*   **训练目标:** 动作专家使用以下目标函数进行训练：
    $$
    \ell^{\tau}(\theta) = \mathbb{E}_{p(\mathbf{A}_t \mid \mathbf{f}_t), q(\mathbf{A}_t^{\tau} \mid \mathbf{A}_t)} \left[ \left\| \mathcal{E}_{\theta}(\mathbf{A}_t^{\tau}, \mathbf{O}_t) - \mathbf{u}(\mathbf{A}_t^{\tau} \mid \mathbf{A}_t) \right\| \right]
    $$
    **符号解释:**
    *   $\ell^{\tau}(\theta)$: 模型参数 $\theta$ 的损失函数，上标 $\tau$ 表示流匹配的时间步。
    *   $\mathbb{E}[\cdot]$: 期望运算符。
    *   $p(\mathbf{A}_t \mid \mathbf{f}_t)$: 在给定多模态特征 $\mathbf{f}_t$ 的条件下，真实动作序列 $\mathbf{A}_t$ 的概率分布。
    *   $q(\mathbf{A}_t^{\tau} \mid \mathbf{A}_t)$: 从真实动作序列 $\mathbf{A}_t$ 生成带噪声的动作 $\mathbf{A}_t^{\tau}$ 的概率分布。
    *   $\mathbf{A}_t^{\tau} = \tau \mathbf{A}_t + (1 - \tau)\epsilon$: 在时间步 $\tau$ 处的带噪声动作，其中 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ 是一个标准正态分布的随机噪声向量。
    *   $\mathbf{f}_t$: `VLM` 主干网络从视觉观测 $\mathbf{O}_t$ 中提取的多模态特征。
    *   $\mathcal{E}_{\theta}(\mathbf{A}_t^{\tau}, \mathbf{O}_t)$: 动作专家模型，在给定带噪声动作 $\mathbf{A}_t^{\tau}$ 和观测 $\mathbf{O}_t$ 的情况下，预测去噪向量。
    *   $\mathbf{u}(\mathbf{A}_t^{\tau} \mid \mathbf{A}_t) = \epsilon - \mathbf{A}_t$: 目标去噪向量场。
    *   $\|\cdot\|$: 范数（通常是 $L_2$ 范数），衡量预测值与目标值之间的差异。

        **目标:** 在这个目标函数下，$\mathcal{E}_{\theta}(\mathbf{A}_t^{\tau}, \mathbf{O}_t)$ 学习匹配去噪向量场 $\mathbf{u}(\mathbf{A}_t^{\tau} \mid \mathbf{A}_t)$，从而能够从噪声中恢复出真实的动作序列。

#### 4.2.2.3. 多模态融合与投影 (Multi-modal Fusion and Projection)
`DynamicVLA` 使用轻量级的<strong>线性投影 (linear projections)</strong> 来对齐不同模块之间的表示。这包括：
1.  将机器人<strong>本体感知状态 (robot states)</strong> 嵌入到多模态特征空间。
2.  将动作表示<strong>适配 (adapting)</strong> 到基于扩散的动作专家。
3.  匹配 `VLM` 主干网络和动作专家之间的输出维度。

### 4.2.3. C. 连续推理 (Continuous Inference)
在现有的 `VLA` 模型中 [18, 6, 15]，新的推理周期只有在之前预测的动作序列 $\mathbf{A}_t$ 完全执行完毕后才会触发。这种<strong>串行化 (serializes)</strong> 的推理和执行方式引入了<strong>块间等待 (inter-chunk waiting)</strong>，导致控制暂停，直到下一个动作序列可用为止，从而在动态物体运动下降低了响应性。

`Continuous Inference` 机制的运作方式如下，如以下 Figure 2b 所示：

![该图像是示意图，展示了DynamicVLA框架的架构与关键设计。图中包括三个主要部分：第一部分(a)显示了DynamicVLA架构，其中包含SmolLM2-360M模型和Action Expert；第二部分(b)介绍了连续推理的流程，强调推理循环与执行循环的关系；第三部分(c)展示了潜在感知动作流，描述了输入流与动作流的关系。这些设计旨在实现动态物体的高效操控和适应。](images/2.jpg)  
*该图像是示意图，展示了DynamicVLA框架的架构与关键设计。图中包括三个主要部分：第一部分(a)显示了DynamicVLA架构，其中包含SmolLM2-360M模型和Action Expert；第二部分(b)介绍了连续推理的流程，强调推理循环与执行循环的关系；第三部分(c)展示了潜在感知动作流，描述了输入流与动作流的关系。这些设计旨在实现动态物体的高效操控和适应。*

图示 2b：连续推理流程，强调推理循环与执行循环的关系。

*   **并行化:** 推理周期在上次推理完成后立即触发，独立于之前预测的动作序列是否已执行完毕。
*   **推理延迟 $m$:** 令 $m$ 表示推理延迟，即从推理开始到完成的时间步数。推理在时间步 $t, t+m, t+2m, \ldots$ 完成（为简化起见，假设 $m$ 为常数）。
*   **重叠执行:** 在执行过程中，$\mathbf{A}_t$ 中的动作会持续执行，同时下一个动作序列 $\mathbf{A}_{t+m}$ 正在被推理。
*   **前提条件:** 假设 $n > m$，即预测的动作序列长度 $n$ 大于推理延迟 $m$。这确保了一个新的动作序列在当前序列执行完成之前即可用。
*   **效果:** 这样，动作执行就不会因为等待推理完成而阻塞，从而消除了<strong>块间等待 (inter-chunk waiting)</strong>。

### 4.2.4. D. 潜在感知动作流 (Latent-aware Action Streaming)
如以下 Figure 2c 所示，推理延迟 $m$ 会导致预测动作与不断演变的环境之间出现<strong>时间错位 (temporal misalignment)</strong>。这种错位主要体现在两个方面：

![该图像是示意图，展示了DynamicVLA框架的架构与关键设计。图中包括三个主要部分：第一部分(a)显示了DynamicVLA架构，其中包含SmolLM2-360M模型和Action Expert；第二部分(b)介绍了连续推理的流程，强调推理循环与执行循环的关系；第三部分(c)展示了潜在感知动作流，描述了输入流与动作流的关系。这些设计旨在实现动态物体的高效操控和适应。](images/2.jpg)  
*该图像是示意图，展示了DynamicVLA框架的架构与关键设计。图中包括三个主要部分：第一部分(a)显示了DynamicVLA架构，其中包含SmolLM2-360M模型和Action Expert；第二部分(b)介绍了连续推理的流程，强调推理循环与执行循环的关系；第三部分(c)展示了潜在感知动作流，描述了输入流与动作流的关系。这些设计旨在实现动态物体的高效操控和适应。*

图示 2c：潜在感知动作流，描述了输入流与动作流的关系。

#### 4.2.4.1. 1) 感知-执行差距 (Perception-Execute Gap)
当在时间 $t$ 启动推理以预测 $\mathbf{A}_t$ 时，这些预测的动作只有在 $t+m$ 时刻才可用。到那时，观测已经演变为 $\mathbf{O}_{t+m}$。因此，动作序列 $\left\{ \mathbf{a}_t, \ldots, \mathbf{a}_{t+m-1} \right\}$ 已不再与当前的观测对齐。它们是基于旧的、过时的环境状态生成的。

#### 4.2.4.2. 2) 重叠动作块冲突 (Conflicts Between Overlapping Action Chunks)
连续推理允许在 $\mathbf{A}_t$ 的执行完成之前生成新的动作序列 $\mathbf{A}_{t+m}$。这可能导致对于相同的执行时间步，存在多个候选动作（例如，来自 $\mathbf{A}_t$ 和 $\mathbf{A}_{t+m}$）。

`Latent-aware Action Streaming` 通过一个明确的执行策略解决了这两个问题：
*   **丢弃过时动作:** 动作序列 $\mathbf{A}_t$ 中对应于早于 $t+m$ 时间步的动作被<strong>丢弃 (discarded)</strong>，因为它们已经过时。执行将从子序列 $\{ \mathbf{a}_{t+m}, \ldots, \mathbf{a}_{t+n} \}$ 开始。
*   **优先最新预测:** 对于 $\mathbf{A}_t$ 和 $\mathbf{A}_{t+m}$ 之间存在重叠的时间步，来自<strong>较新序列 $\mathbf{A}_{t+m}$ 的动作会被优先执行 (prioritized)</strong>，覆盖掉来自 $\mathbf{A}_t$ 的动作。

    **效果:** 这种策略使得执行能够迅速适应最新的环境状态，特别是在动态物体运动的情况下，从而确保了<strong>时间一致的控制 (temporally consistent control)</strong>，尽管存在推理延迟。

## 4.3. 训练方案 (The Training Scheme)
`DynamicVLA` 的训练分为三个阶段：

### 4.3.1. 预训练阶段 (Pre-training Stage)
*   **目标:** 对齐视觉和语言表示。
*   **组件:** 视觉-语言主干网络（卷积视觉编码器 `FastViT` 和紧凑语言模型 `SmolLM2-360M`），均从各自的预训练权重初始化。
*   **数据:** 使用从 `COYO-700M` [8] 中采样的 1.5 亿个英语图像-文本对进行大规模视觉-语言预训练。

### 4.3.2. 中期训练阶段 (Mid-training Stage)
*   **目标:** 训练完整的 `VLA` 模型以进行动态物体操作。
*   **数据:** 在合成的 `Dynamic Object Manipulation (DOM)` 数据集（见第 IV 节）上进行训练。
*   **输入:** 每个回合提供<strong>时间演变的多视图视觉观测 (temporally evolving multi-view visual observations)</strong>。模型使用腕部摄像头和固定第三人称摄像头捕捉的图像。
*   **观测窗口:** `稀疏时间窗口` $\mathbf{O}_t = \{ \mathbf{o}_{t-2}, \mathbf{o}_t \}$，每个时间步使用两个视图，共四个图像输入。这些图像按通道维度拼接并由视觉编码器共同处理，旨在促进<strong>隐式物体速度感知 (implicit object velocity perception)</strong>。
*   **优化:** 使用随机采样的回合时间步形成的<strong>小批量 (minibatches)</strong> 进行优化。
*   **训练目标:** 模型在元组 $( \mathbf{O}_t, \mathbf{L}_t, \mathbf{P}_t )$ 上训练，同时动作专家根据公式 (1) 对带噪声的动作块 $\mathbf{A}_t^{\tau}$ 进行去噪。

### 4.3.3. 后训练阶段 (Post-training Stage)
*   **目标:** 将模型微调到特定机器人平台和感知配置。
*   **数据:** 使用机器人特定的真实世界演示数据。
*   **训练目标:** 沿用中期训练阶段的相同目标函数。

## 4.4. 模型架构细节 (Model Architecture Details)

### 4.4.1. VLM 主干网络 (VLM Backbone)
*   **视觉输入处理:** 时间观测窗口 $\mathbf{O}_t$ 中的 `RGB` 图像被拼接并通过 `FastViT` [44] 编码。
    *   图像大小: $384 \times 384$。
    *   编码器阶段: 逐步增加通道宽度 (96, 192, 384, 768, 1536)，对应模块深度 (2, 12, 24, 4, 2)。
    *   空间压缩: `FastViT` 使用初始大补丁大小 64 和步长下采样进行激进的空间压缩。
    *   词元混合: 在早期阶段使用 `RepMixer` 风格的词元混合，后期阶段使用 `Attention`。
    *   输出: 36 个固定维度为 960 的视觉词元，与语言嵌入空间对齐，显著减少了词元数量，同时保留了操作相关的空间结构。
*   **本体感知状态输入:** 机器人本体感知状态 $\mathbf{P}_t$ 作为显式条件信号。
    *   状态向量: 32 维，包含笛卡尔位置和方向（未使用的条目用零填充）。
    *   嵌入: 线性投影到语言嵌入空间，表示为单个 960 维状态词元。
*   **语言指令输入:** 语言指令 $\mathbf{L}_t$ 根据提示长度词元化为可变数量的语言词元。
*   **多模态处理:** 所有视觉、语言和状态词元被拼接，并由语言主干网络 (`SmolLM2-360M` 的前 16 个 `Transformer` 层) 共同处理。
*   **输出缓存:** 主干网络输出所有已处理词元的<strong>键-值表示 (key-value representations)</strong>，这些表示被缓存并在后续推理周期中重用。

### 4.4.2. 动作专家 (Action Expert)
*   **动作生成:** 由专门的<strong>基于扩散的动作专家 (diffusion-based action expert)</strong> 负责。
*   **实现:** 实例化为一个轻量级 `Transformer`，它复制自语言主干网络并截断到前 16 层。
*   **预测动作块:** 专家预测一个长度为 $n = 20$ 的动作块，这在 `Continuous Inference` 下足以保持低推理延迟。
*   **动作表示:** 每个动作是一个 32 维向量，表示末端执行器姿态和抓手状态（未使用的条目用零填充）。
*   **输入:** 训练期间，带噪声的动作输入形状为 $(n, 32)$；推理期间，输入为纯噪声。
*   **计算效率:** 动作专家使用 720 的缩减隐藏维度 (0.75 倍语言嵌入大小) 来降低计算量。
*   **去噪过程:** 带噪声的动作词元被投影到这个空间，并与扩散时间步嵌入结合。去噪更新通过查询缓存的键-值表示生成，无需重新编码感知输入。

# 5. 实验设置

## 5.1. 数据集
`DynamicVLA` 在<strong>动态物体操作 (Dynamic Object Manipulation, DOM)</strong> 基准上进行评估。`DOM` 是第一个大规模专门用于动态物体操作的基准，旨在解决现有数据集中缺乏动态物体操作数据的问题。

### 5.1.1. 模拟数据收集 (Simulation Data Collection)
*   **框架:** 使用 `Isaac Sim` [31] 构建高吞吐量管道，统一场景和物体采样、多视图感知、实时物体状态获取和闭环控制。
*   **物体和动态:**
    *   **物体:** 包含 206 种日常物体，来自 `Objeverse` [11]，涵盖水果、蔬菜、容器等家居用品，并进行纹理增强以增加视觉多样性。
    *   **运动参数:** 物体速度从 $0 \sim 0.75 \ \text{m/s}$（部分保持静止），摩擦系数从 $0.5 \sim 1.5$ 采样。
    *   **交互:** 工作空间中放置多个物体，允许运动期间的自然交互。
*   **场景和传感器:**
    *   **场景:** 从 `3D-FRONT` [13] 派生 2.8K 个多样化 3D 场景，确保干净、平坦的桌面，并移除自遮挡或不真实的物体放置。
    *   **摄像头:** 每个场景配备三台摄像头：两台第三人称视角摄像头（一台在机器人前方 1m、0.6m 高，一台在左侧 1m、0.35m 高）和一台腕部安装摄像头。
    *   **图像参数:** 所有摄像头以 25 FPS 捕获 RGB 帧，分辨率为 $480 \times 360$，焦距 $2.3 \ \text{mm}$（与 `Azure Kinect` 内参对齐）。
    *   **照明:** 随机化场景照明，色温从 $4000 \sim 8000 \ \text{K}$，光强度从 $150 \sim 750 \ \text{lm}$，光源位置在 $x \in [-50, 50] \ \text{m}$、$y \in [-50, 50] \ \text{m}$、$z \in [10, 20] \ \text{m}$ 范围内采样。
*   <strong>物体状态获取 (Object State Acquisition):</strong>
    *   模拟器在每个回合中维护 6D 物体状态的<strong>真值 (ground-truth)</strong>。
    *   `Isaac Sim` 通过物理引擎随机化物理参数并传播物体运动，从中以 25 Hz 提取每个物体的<strong>位置 (position)</strong>、<strong>旋转 (rotation)</strong> 以及<strong>线速度/角速度 (linear/angular velocity)</strong>。
    *   这些无噪声的轨迹为控制器提供实时运动线索，用于短时预测和状态转换。
*   <strong>状态机控制器 (State Machine Controller):</strong>
    *   消耗实时 6D 物体姿态、速度和静态目标物体的 6D 姿态。
    *   执行一个四阶段闭环例程：
        1.  <strong>接近对象 (Approach Object):</strong> 预测物体近期运动（约 $0.2 \sim 0.3 \ \text{s}$），并将末端执行器定位在预测位置上方 10cm 处，持续更新。
        2.  <strong>抓取并提升 (Grasp &amp; Lift):</strong> 下降、稳定残余运动，然后抓紧并提升。
        3.  <strong>接近目标并放置 (Approach Target &amp; Place):</strong> 朝向从目标物体 6D 几何体导出的放置姿态移动，并精确放置物体。
        4.  <strong>重置 (Reset):</strong> 返回起始姿态以开始下一个回合。
    *   这种设计产生<strong>反应性、预测驱动的轨迹 (reactive, prediction-informed trajectories)</strong>，能够生成可扩展的真实动态操作回合。

### 5.1.2. 真实世界数据收集 (Real-World Data Collection)
*   **挑战:** 遥操作 (teleoperation) 对动态操作无效，因为人类反应速度无法跟上快速移动的物体。真实世界缺乏 6D 物体状态的<strong>真值 (ground-truth)</strong>。
*   **解决方案:** 构建一个<strong>真实世界“模拟器” (real-world "simulator")</strong> 管道，使用商用 `RGB-D` 传感器近似模拟器风格的物体状态，并实现无遥操作的大规模动态操作数据收集。
*   **环境设置:**
    *   **物体:** 使用 25 个物理家用物体，包括容器、食物、瓶子和工具，每个回合包含多个物体，包括抓取/放置目标和自然干扰物。
    *   **传感器:** 场景由两台同步的第三人称 `RGB` 摄像头 (`Azure Kinect DK`)（分别位于前方和侧方）和一台腕部安装的 `RealSense D435i` 捕获。这些传感器与模拟器几何结构匹配，并提供同步、校准的 `RGB` 流用于状态估计。
*   <strong>物体状态获取 (Object State Acquisition):</strong>
    *   为了复制模拟器的状态接口，构建了一个输出 6D 物体姿态和速度的“实时”模拟器。
    *   `EfficientTAM` [51] 提供来自同步第三人称摄像头的每个视角的物体<strong>掩码 (object masks)</strong>。
    *   通过<strong>几何三角测量 (geometric triangulation)</strong> 步骤恢复 3D 中心点。
    *   通过在短时间窗口内拟合运动来获取<strong>线速度 (linear) 和角速度 (angular velocity)</strong>，生成平滑、低延迟的 6D 状态流，与控制器的要求兼容。
*   <strong>状态机控制器 (State-machine Controller):</strong>
    *   与模拟中使用的四阶段控制器相同，在真实世界中无需更改即可运行。
    *   消耗估计的 6D 物体状态和目标姿态。
    *   确保在 `Franka` 和 `PiPER` 机器人上实现一致的多机器人形态覆盖。

## 5.2. 评估指标
所有方法都使用以下三个指标进行评估：

### 5.2.1. 成功率 (Success Rate, SR)
*   **概念定义:** 成功完成指令操作的试验所占的比例，即在没有物体掉落或超时的情况下完成任务的试验百分比。它衡量了策略在动态操作任务中的可靠性和有效性。
*   **数学公式:**
    $$
    \text{SR} = \frac{\text{成功完成任务的试验次数}}{\text{总试验次数}} \times 100\%
    $$
*   **符号解释:**
    *   $\text{成功完成任务的试验次数}$: 机器人成功执行指令操作，且未发生物体掉落或超时的试验总数。
    *   $\text{总试验次数}$: 进行的全部试验总数。

### 5.2.2. 路径长度 (Path Length, PL)
*   **概念定义:** 在任务执行期间，机器人末端执行器（例如，机械臂的抓手部分）移动的总轨迹长度。这个指标可以衡量机器人动作的效率，路径越短通常意味着动作越直接和高效。
*   **数学公式:**
    对于在时间步 $i$ 具有位置 $(x_i, y_i, z_i)$ 的末端执行器轨迹，路径长度计算为相邻时间步之间距离的总和：
    $$
    \text{PL} = \sum_{i=1}^{N-1} \sqrt{(x_{i+1}-x_i)^2 + (y_{i+1}-y_i)^2 + (z_{i+1}-z_i)^2}
    $$
*   **符号解释:**
    *   $\text{PL}$: 末端执行器的总路径长度。
    *   $N$: 轨迹中离散时间步的总数。
    *   $(x_i, y_i, z_i)$: 末端执行器在第 $i$ 个时间步的笛卡尔坐标位置。
    *   $\sqrt{(x_{i+1}-x_i)^2 + (y_{i+1}-y_i)^2 + (z_{i+1}-z_i)^2}$: 第 $i$ 个时间步到第 $i+1$ 个时间步之间的欧几里得距离。

### 5.2.3. 任务完成时间 (Task Completion Time, T.Time)
*   **概念定义:** 从物体开始运动的时刻到任务终止（无论是成功完成、超时还是物体掉落）所经过的总时间。该指标衡量了机器人完成任务的速度和响应性，对于动态操作尤为重要。
*   **数学公式:**
    $$
    \text{T.Time} = t_{\text{end}} - t_{\text{start}}
    $$
*   **符号解释:**
    *   $\text{T.Time}$: 任务的总完成时间。
    *   $t_{\text{end}}$: 任务终止（成功、超时或物体掉落）的时刻。
    *   $t_{\text{start}}$: 动态物体开始运动的时刻。

## 5.3. 对比基线
### 5.3.1. 模拟实验对比基线
在模拟环境中，`DynamicVLA` 与以下代表性 `VLA` 基线模型进行了比较：
*   `Diffusion Policy` [9]
*   `OpenVLA-OFT` [18]
*   $\pi_0$ [6]
*   $\pi_{0.5}$ [15]
*   `SmolVLA` [38]
*   `GR00T-N1.5` [5]
*   `VLA-Adapter-Pro` [46]
*   `VLASH` [39]

    这些基线涵盖了通用 `VLA` 模型、基于轻量级适应的模型和延迟感知设计。所有基线都从公开可用的预训练权重初始化，并使用一致的微调协议适应到 `DOM` 基准。

### 5.3.2. 真实世界实验对比基线
在真实世界实验中，由于物理设置的限制，选取了部分模拟中表现较好的基线进行评估：
*   $\pi_{0.5}$
*   `SmolVLA`
*   `VLASH`

    这些模型在相同的物理设置下进行评估。

## 5.4. 执行约束
为了确保真实世界操作的安全，机器人工作空间被限制在预定义的范围内。如果预测的末端执行器位置超出了预设的安全阈值，机器人将中止当前尝试并返回到安全起始姿态，该试验被标记为失败。

## 5.5. 评估协议
*   **实验设置:** 实验在 `Isaac Sim` 模拟环境（带 `Franka Emika Panda` 机械臂）、真实世界 `Franka` 机械臂和真实世界 `AgileX PiPER` 机械臂三种环境下进行，涵盖了模拟和物理机器人形态。
*   **标准化物体运动:** 为了公平比较，使用一个辅助机械臂按照固定发射轨迹来标准化物体运动。尽管初始速度因物理噪声而异，但运动模式在试验之间保持可比性。
*   **重复次数:** 每个真实世界实验重复 20 次，结果取平均值。
*   **统一条件:** 所有方法在每个环境内都以相同的条件进行评估。

# 6. 实验结果与分析

## 6.1. 核心结果分析
本节分析 `DynamicVLA` 在<strong>动态物体操作 (dynamic object manipulation)</strong> 方面的性能，并将其与代表性 `VLA` 基线模型进行比较，涵盖<strong>交互 (interaction)</strong>、<strong>感知 (perception)</strong> 和<strong>泛化 (generalization)</strong> 三个维度。

### 6.1.1. 动态交互与响应性 (Dynamic Interaction and Reactivity)
该维度评估策略对不断变化的物体运动的响应能力，包括<strong>闭环响应性 (closed-loop reactivity)</strong>、<strong>动态适应 (dynamic adaptation)</strong> 和<strong>长时程序列 (long-horizon sequencing)</strong>。难度递增，从响应速度变化的运动，到从突然的事件驱动变化中恢复，再到在长时间交互中维持多个移动物体的协调。

以下是原文 Table I 的结果：

<table>
<thead>
<tr>
<td rowspan="2">Methods</td>
<th colspan="3">Interaction</th>
<th colspan="3">Perception</th>
<th colspan="3">Generalization</th>
<th colspan="3">Average</th>
</tr>
<tr>
<th>CR</th>
<th>DA</th>
<th>LS</th>
<th>VU</th>
<th>SR</th>
<th>MP</th>
<th>VG</th>
<th>MG</th>
<th>DR</th>
<th>SR ↑</th>
<th>Path Len ↓</th>
<th>Time ↓</th>
</tr>
</thead>
<tbody>
<tr>
<td>Diffusion Policy [9]</td>
<td>0.50</td>
<td>0.50</td>
<td>0.00</td>
<td>1.00</td>
<td>0.00</td>
<td>0.00</td>
<td>1.00</td>
<td>0.50</td>
<td>0.00</td>
<td>0.38</td>
<td>1.34</td>
<td>10.89</td>
</tr>
<tr>
<td>OpenVLA-OFT [18]</td>
<td>3.50</td>
<td>0.50</td>
<td>0.50</td>
<td>0.00</td>
<td>1.50</td>
<td>0.50</td>
<td>3.50</td>
<td>2.00</td>
<td>0.00</td>
<td>1.33</td>
<td>1.08</td>
<td>10.83</td>
</tr>
<tr>
<td>$\pi_0$ [6]</td>
<td>7.50</td>
<td>12.00</td>
<td>3.00</td>
<td>5.50</td>
<td>10.50</td>
<td>7.50</td>
<td>5.50</td>
<td>12.50</td>
<td>9.00</td>
<td>8.11</td>
<td>1.19</td>
<td>10.55</td>
</tr>
<tr>
<td>$\pi_{0.5}$ [15]</td>
<td>9.50</td>
<td>17.50</td>
<td>3.50</td>
<td>5.00</td>
<td>12.50</td>
<td>9.00</td>
<td>5.00</td>
<td>19.50</td>
<td>18.00</td>
<td>11.06</td>
<td>1.28</td>
<td>10.62</td>
</tr>
<tr>
<td>SmolVLA [38]</td>
<td>18.50</td>
<td>17.50</td>
<td>5.50</td>
<td>1.50</td>
<td>14.50</td>
<td>11.50</td>
<td>14.50</td>
<td>13.50</td>
<td>17.00</td>
<td>12.67</td>
<td>1.30</td>
<td>10.65</td>
</tr>
<tr>
<td>GROOT-N1.5 [5]</td>
<td>10.50</td>
<td>12.00</td>
<td>4.00</td>
<td>9.50</td>
<td>13.50</td>
<td>14.00</td>
<td>14.50</td>
<td>19.50</td>
<td>20.00</td>
<td>13.05</td>
<td>1.29</td>
<td>10.56</td>
</tr>
<tr>
<td>VLA-Adapter-Pro [46]</td>
<td>21.00</td>
<td>15.50</td>
<td>6.00</td>
<td>6.50</td>
<td>16.50</td>
<td>10.50</td>
<td>15.00</td>
<td>18.50</td>
<td>13.00</td>
<td>13.61</td>
<td>1.51</td>
<td>9.98</td>
</tr>
<tr>
<td>VLASH [39]</td>
<td>9.00</td>
<td>20.50</td>
<td>7.50</td>
<td>6.50</td>
<td>7.50</td>
<td>12.00</td>
<td>7.00</td>
<td>21.00</td>
<td>20.00</td>
<td>12.33</td>
<td>1.27</td>
<td>10.60</td>
</tr>
<tr>
<td>DynamicVLA</td>
<td><strong>60.50</strong></td>
<td><strong>38.50</strong></td>
<td><strong>40.50</strong></td>
<td><strong>51.50</strong></td>
<td><strong>48.00</strong></td>
<td><strong>33.50</strong></td>
<td><strong>59.50</strong></td>
<td><strong>65.00</strong></td>
<td><strong>26.50</strong></td>
<td><strong>47.06</strong></td>
<td><strong>2.50</strong></td>
<td><strong>8.53</strong></td>
</tr>
</tbody>
</table>

**CR**: 闭环响应性 (Closed-loop Reactivity)；**DA**: 动态适应 (Dynamic Adaptation)；**LS**: 长时程序列 (Long-horizon Sequencing)；**VU**: 视觉理解 (Visual Understanding)；**SR**: 空间推理 (Spatial Reasoning)；**MP**: 运动感知 (Motion Perception)；**VG**: 视觉泛化 (Visual Generalization)；**MG**: 运动泛化 (Motion Generalization)；**DR**: 抗干扰鲁棒性 (Disturbance Robustness)。

从 Table I 中<strong>交互 (Interaction)</strong> 维度的结果（`CR`/`DA`/`LS`）可以看出：
*   所有基线 `VLA` 模型在动态运动下的成功率都**持续较低**，这表明它们难以应对动态操作的挑战。
*   `DynamicVLA` 保持了**稳健的性能**。例如，在闭环响应性 (`CR`) 任务中，`DynamicVLA` 实现了 60.5% 的成功率，远超最强的基线 `VLA-Adapter-Pro` 的 21.0%。在动态适应 (`DA`) 和长时程序列 (`LS`) 任务中，`DynamicVLA` 的成功率分别为 38.5% 和 40.5%，同样显著优于基线。
*   具体而言，`DynamicVLA` 在所有交互设置中的成功率比最强的基线分别提高了 $+188.1%$ (`CR`)、$+87.8%$ (`DA`) 和 $+440.0%$ (`LS`)。
*   这种趋势在真实世界实验中也**保持一致**，如以下 Figure 4 所示。基线方法经常因反应延迟、过时动作执行或协调性丧失而失败，而 `DynamicVLA` 在严格的时间约束下更能可靠地重新对齐感知与动作。

    ![该图像是图表，展示了不同模型在动态对象操控任务中的成功率。图中比较了DynamicVLA、VLASH、SmolVLA和π0.5四种模型在六个不同任务中的表现，突显了DynamicVLA在各任务中的优越性，成功率达71.6%。](images/4.jpg)
    *该图像是图表，展示了不同模型在动态对象操控任务中的成功率。图中比较了DynamicVLA、VLASH、SmolVLA和π0.5四种模型在六个不同任务中的表现，突显了DynamicVLA在各任务中的优越性，成功率达71.6%。*

图示 4：真实世界交互评估。图中比较了 DynamicVLA、VLASH、SmolVLA 和 $\pi_{0.5}$ 四种模型在六个不同任务中的表现，突出显示了 DynamicVLA 在各任务中的优越性，平均成功率达 71.6%。

### 6.1.2. 多模态时空推理 (Multimodal Spatial-Temporal Reasoning)
该维度评估 `VLA` 策略在动态环境中感知和<strong>理解视觉与语言线索 (grounding visual and linguistic cues)</strong> 的能力。难度从<strong>视觉识别 (visual recognition)</strong> 逐步增加到<strong>空间推理 (spatial reasoning)</strong>，最后到<strong>运动感知 (motion perception)</strong>，每一项都对底层 `VLM` 提出了更高的要求。

从 Table I 中<strong>感知 (Perception)</strong> 维度的结果（`VU`/`SR`/`MP`）可以看出：
*   随着任务难度从视觉理解到空间推理再到运动感知，所有模型的性能都**持续下降**。
*   尽管许多 `VLA` 在静态操作中表现良好，但在动态场景下性能显著下降，尤其在需要及时准确解释不断演变的时空关系的<strong>空间和运动推理 (spatial and motion reasoning)</strong> 方面，性能下降更为剧烈。
*   为了满足交互延迟要求，轻量级 `VLA` 模型必须在 `VLM` 容量上做出妥协，这使得感知密集型动态任务更具挑战性。
*   `DynamicVLA` 在视觉理解 (`VU`)、空间推理 (`SR`) 和运动感知 (`MP`) 方面的成功率分别为 51.5%、48.0% 和 33.5%，远超所有基线模型。例如，在空间推理任务中，`DynamicVLA` 达到 48.0%，而最强的基线 `VLA-Adapter-Pro` 仅为 16.5%。
*   这种趋势在真实世界实验中也得到一致反映，如以下 Figure 5 所示。表现最好的基线模型由于频繁的<strong>时空错位 (spatial-temporal misalignment)</strong> 导致成功率较低 (11.7%)，而 `DynamicVLA` 达到了 51.9% 的成功率。

    ![该图像是图表，展示了不同模型在动态物体操作任务中的成功率。图表包含四种模型的性能比较：π0.5、SmolVLA、VLASH和DynamicVLA，涉及视觉理解、空间推理和运动感知三项任务。各模型的成功率以条形图形式呈现，DynamicVLA在多项任务中表现相对较好。](images/5.jpg)
    *该图像是图表，展示了不同模型在动态物体操作任务中的成功率。图表包含四种模型的性能比较：π0.5、SmolVLA、VLASH和DynamicVLA，涉及视觉理解、空间推理和运动感知三项任务。各模型的成功率以条形图形式呈现，DynamicVLA在多项任务中表现相对较好。*

图示 5：真实世界感知评估。图表包含四种模型的性能比较：$\pi_{0.5}$、SmolVLA、VLASH 和 DynamicVLA，涉及视觉理解、空间推理和运动感知三项任务。

### 6.1.3. 对未知场景的泛化能力 (Generalization to Unseen Frontiers)
该维度评估策略对训练条件之外的<strong>分布偏移 (distribution shifts)</strong> 的鲁棒性，包括<strong>外观变化 (appearance variation)</strong>、<strong>未知运动模式 (unseen motion patterns)</strong> 和<strong>环境扰动 (environmental perturbations)</strong>。

从 Table I 中<strong>泛化 (Generalization)</strong> 维度的结果（`VG`/`MG`/`DR`）可以看出：
*   先前的 `VLA` 模型在外观、运动和环境扰动下的分布偏移中成功率较低。
*   `DynamicVLA` 实现了更高的整体性能。例如，在视觉泛化 (`VG`) 和运动泛化 (`MG`) 任务中，`DynamicVLA` 的成功率分别达到 59.5% 和 65.0%，而最强的基线 `VLASH` 在 `MG` 中为 21.0%，`SmolVLA` 在 `VG` 中为 14.5%。
*   类似趋势在真实世界实验中也观察到，如以下 Figure 6 所示，适用于外观和运动偏移。然而，即使对于 `DynamicVLA` 而言，对环境扰动 (`DR`) 的鲁棒性仍然具有挑战性，成功率为 26.5%。
*   值得注意的是，`DR` 任务涉及模拟中更强的扰动，超出了理想化的物理假设。因此，原文省略了真实世界的结果，因为此类扰动难以可靠复现，且它们在物理环境中（例如，表面不规则性）的普遍性难以控制。

    ![Fig. 6: Real-world Generation Evaluation. We compare representative VLA models on four real-world dynamic manipulation tasks across Franka and PiPER, averaging success rates over 20 trials for each of three paired motionposition configurations, with object motion generated by a secondary robot arm.](images/6.jpg)
    *该图像是一个图表，展示了四种代表性的VLA模型在实际动态操作任务中的成功率比较。模型包括π0.5、SmolVLA、VLASH和DynamicVLA，数据展示了在视觉泛化与运动泛化任务中的表现。成功率以百分比形式显示，任务说明包括将瓶子、球以及其他物品放置于指定位置。*

图示 6：真实世界泛化评估。该图表展示了四种代表性的 VLA 模型在 Franka 和 PiPER 机器人上进行的四项真实世界动态操作任务中的成功率比较，包括视觉泛化与运动泛化任务。

## 6.2. 消融实验 (Ablation Studies)
为了评估 `DynamicVLA` 设计选择的影响，论文进行了消融研究，隔离了<strong>模型容量 (model capacity)</strong>、<strong>视觉编码 (visual encoding)</strong> 和<strong>执行机制 (execution mechanisms)</strong> 的影响。所有变体都在 `DOM` 基准上使用相同的训练协议和指标进行评估。

以下是原文 Table II 的结果：

<table>
<thead>
<tr>
<td></td>
<td>Size</td>
<td>FViT</td>
<td>CI</td>
<td>LAAS</td>
<td>SR (%) ↑</td>
<td>PL (m) ↓</td>
<td>Time (s) ↓</td>
</tr>
</thead>
<tbody>
<tr>
<td>[1]</td>
<td>360M</td>
<td>✓</td>
<td>×</td>
<td>×</td>
<td>30.27</td>
<td>2.77</td>
<td>9.86</td>
</tr>
<tr>
<td>[2]</td>
<td>360M</td>
<td>✓</td>
<td>✓</td>
<td>×</td>
<td>36.11</td>
<td>1.77</td>
<td>9.51</td>
</tr>
<tr>
<td>[3]</td>
<td>360M</td>
<td>×</td>
<td>✓</td>
<td>✓</td>
<td>39.72</td>
<td>2.61</td>
<td>8.84</td>
</tr>
<tr>
<td>[4]</td>
<td>135M</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>26.67</td>
<td>1.82</td>
<td>9.95</td>
</tr>
<tr>
<td>[5]</td>
<td>1.7B</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>24.33</td>
<td>1.77</td>
<td>9.91</td>
</tr>
<tr>
<td>[6]</td>
<td>360M</td>
<td>×</td>
<td>✓</td>
<td>✓</td>
<td>28.89</td>
<td>1.86</td>
<td>9.89</td>
</tr>
<tr>
<td>[7]</td>
<td>360M</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td><strong>47.06</strong></td>
<td><strong>2.50</strong></td>
<td><strong>8.53</strong></td>
</tr>
</tbody>
</table>

**FViT**: 使用 FastViT 作为视觉编码器；**CI**: 连续推理 (Continuous Inference)；**LAAS**: 潜在感知动作流 (Latent-aware Action Streaming)。

### 6.2.1. 主干网络容量 (Backbone Capacity)
*   为了评估语言模型容量的影响，比较了不同大小的 `SmolLM2` 主干网络（135M、360M 和 1.7B），在相同的架构和执行设置下。
*   <strong>结果 (Table II 中的 [4], [5], [7] ):</strong> 增加模型大小会提高表示能力，但也会导致更高的推理延迟，这在动态场景中会降低闭环响应性，从而导致成功率下降。相反，减小模型大小可以提高推理速度，但会限制推理能力，导致次优的动作预测。
*   **结论:** 360M 模型在推理效率和模型容量之间取得了最佳平衡，在动态物体操作中获得了最高的整体性能。

### 6.2.2. 视觉编码器 (Vision Encoder)
*   通过将卷积 `FastViT` 编码器替换为基于 `Transformer` 的视觉编码器（采用 `SmolVLM` [29] 中的相同配置），同时保持所有其他组件不变，来消融视觉编码器的选择。
*   <strong>结果 (Table II 中的 [6] 和 [7] ):</strong> `FastViT` 通过减少词元化降低了编码延迟，同时保持了结构忠实的视觉表示，因此优于基于 `Transformer` 的编码器。

### 6.2.3. 连续推理 (Continuous Inference, CI)
*   为了证明 `Continuous Inference (CI)` 的有效性，在禁用 `CI` 的情况下（而所有其他组件保持不变）进行了评估。
*   <strong>结果 (Table II 中的 [2] 和 [7] ):</strong> 没有 `CI`，推理只有在之前的动作块完全执行后才触发，引入了<strong>块间等待 (inter-chunk waiting)</strong>，这会降低响应性，导致动态操作任务的成功率降低和完成时间更长。

### 6.2.4. 潜在感知动作流 (Latent-aware Action Streaming, LAAS)
*   进一步分析 `Latent-aware Action Streaming (LAAS)` 在 `Continuous Inference` 下的贡献，通过在保留 `CI` 的情况下禁用 `LAAS` 进行评估。
*   <strong>结果 (Table II 中的 [3] 和 [7] ):</strong> 尽管 `CI` 实现了连续动作生成，但在推理延迟下，它单独仍然不足，因为预测动作与不断演变的环境之间的时间错位会降低性能。`LAAS` 通过丢弃过时动作并优先处理最新预测来解决这个问题，从而在动态场景中强制执行时间对齐的执行并提高了稳定性。
*   **综合分析:** 比较 [1] 和 [7] 可以发现，当 `CI` 和 `LAAS` 都被禁用时，性能下降更为严重，这表明它们在动态操作中扮演着互补的角色。

### 6.2.5. 时间视觉上下文 (Temporal Visual Context)
*   进行消融研究以分析<strong>时间视觉上下文 (temporal visual context)</strong> 的影响，通过在相同的 `DynamicVLA` 架构中改变观测窗口 $\mathbf{O}_t$ 的组成。
*   默认设置是 `稀疏时间窗口` $\mathbf{O}_t = \{ \mathbf{o}_{t-2}, \mathbf{o}_t \}$，旨在促进隐式物体速度感知。
*   以下是原文 Table III 的结果：

    <table>
    <thead>
    <tr>
    <th>t-3</th>
    <th>t-2</th>
    <th>t-1</th>
    <th>t</th>
    <th>SR ↑</th>
    <th>PL ↓</th>
    <th>T.Time ↓</th>
    <th>I.Time ↓</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>✓</td>
    <td>38.22</td>
    <td>2.27</td>
    <td>9.52</td>
    <td>0.225</td>
    </tr>
    <tr>
    <td>×</td>
    <td>✓</td>
    <td>×</td>
    <td>×</td>
    <td>43.39</td>
    <td>2.34</td>
    <td>8.77</td>
    <td>0.226</td>
    </tr>
    <tr>
    <td>×</td>
    <td>✓</td>
    <td>×</td>
    <td>✓</td>
    <td><strong>47.06</strong></td>
    <td><strong>2.50</strong></td>
    <td><strong>8.53</strong></td>
    <td>0.226</td>
    </tr>
    <tr>
    <td>×</td>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>46.89</td>
    <td>2.49</td>
    <td>8.51</td>
    <td>0.226</td>
    </tr>
    <tr>
    <td>×</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>47.11</td>
    <td>2.49</td>
    <td>8.46</td>
    <td>0.228</td>
    </tr>
    <tr>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>47.06</td>
    <td>2.47</td>
    <td>8.53</td>
    <td>0.229</td>
    </tr>
    </tbody>
    </table>

**SR**: 成功率；**PL**: 路径长度；**T.Time**: 任务完成时间；**I.Time**: 推理时间。

*   <strong>结果 (Table III):</strong> 不同的时间配置对推理延迟和参数数量的影响微乎其微。
    *   使用**单帧输入** $\{ \mathbf{o}_t \}$ 导致任务成功率明显下降，因为单一观测缺乏估计物体运动和动态所需的<strong>时间线索 (temporal cues)</strong>。
    *   将时间窗口扩展到两帧以上并没有带来进一步的显著收益，表明额外的视觉冗余存在<strong>收益递减 (diminishing returns)</strong>。
    *   与 $\{ \mathbf{o}_{t-2}, \mathbf{o}_t \}$ 相比，设置 $\{ \mathbf{o}_{t-1}, \mathbf{o}_t \}$ 的成功率较低，这表明<strong>较大的时间间隔 (larger temporal interval)</strong> 为速度估计提供了更多信息丰富的运动线索。
*   **结论:** 稀疏但间隔足够的时间上下文对于有效的动态操作至关重要，即使不增加推理频率。

### 6.2.6. LLM 主干网络深度 (Depth of LLM Backbone)
*   遵循主干网络截断策略 [38]，通过在推理期间仅保留 `LLM` 的前 $l$ 个 `Transformer` 层来减少推理延迟。
*   评估了多个主干网络深度 ($l = 8, 16, 24$)，并与完整模型 ($l = 32$) 进行了比较。
*   以下是原文 Table IV 的结果：

    <table>
    <thead>
    <tr>
    <th>#Layers</th>
    <th>SR ↑</th>
    <th>PL ↓</th>
    <th>T.Time ↓</th>
    <th>I.Time ↓</th>
    <th>#Param ↓</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>8</td>
    <td>44.17</td>
    <td>2.33</td>
    <td>8.92</td>
    <td>0.127</td>
    <td>303</td>
    </tr>
    <tr>
    <td>16</td>
    <td><strong>47.06</strong></td>
    <td><strong>2.50</strong></td>
    <td><strong>8.53</strong></td>
    <td><strong>0.226</strong></td>
    <td><strong>430</strong></td>
    </tr>
    <tr>
    <td>24</td>
    <td>48.44</td>
    <td>2.63</td>
    <td>8.43</td>
    <td>0.317</td>
    <td>558</td>
    </tr>
    <tr>
    <td>32</td>
    <td>42.11</td>
    <td>2.69</td>
    <td>8.39</td>
    <td>0.373</td>
    <td>685</td>
    </tr>
    </tbody>
    </table>

**#Layers**: LLM 主干网络层数；**#Param**: 参数数量（百万）。

*   <strong>结果 (Table IV):</strong> 增加主干网络深度会导致推理延迟适度增加。然而，这种额外的延迟可以通过 `Contiguous Inference` 和 `Latent-aware Action Streaming` 在很大程度上得到摊销，并且并未转化为任务成功率的显著提高。
*   相反，激进地截断主干网络会显著提高推理速度，但以降低模型容量为代价，导致成功率大幅下降。
*   **结论:** 16 层主干网络在效率和鲁棒性之间取得了最佳平衡。

### 6.2.7. CI 和 LAAS 的跨模型分析 (Cross-Model Analysis of CI and LAAS)
*   为了评估所提出的执行机制的通用性，`Continuous Inference (CI)` 和 `Latent-aware Action Streaming (LAAS)` 被集成到现有 `VLA` 模型中，包括 `SmolVLA` 和 $\pi_{0.5}$，而无需修改它们的主干网络架构。
*   以下是原文 Table V 的结果：

    <table>
    <thead>
    <tr>
    <th>Method</th>
    <th>SR (%) ↑</th>
    <th>PL (m) ↓</th>
    <th>Time (s) ↓</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>$\pi_{0.5}$† [15]</td>
    <td>15.89</td>
    <td>1.57</td>
    <td>9.95</td>
    </tr>
    <tr>
    <td>SmolVLA† [38]</td>
    <td>25.56</td>
    <td>1.65</td>
    <td>9.77</td>
    </tr>
    <tr>
    <td>DynamicVLA</td>
    <td><strong>47.06</strong></td>
    <td><strong>2.50</strong></td>
    <td><strong>8.53</strong></td>
    </tr>
    </tbody>
    </table>

† 表示 `CI` 和 `LAAS` 在推理时集成。

*   <strong>结果 (Table V):</strong>
    *   在 `SmolVLA` 上观察到**一致的性能改进**，表明 `CI` 和 `LAAS` 在适度推理延迟下有效增强了闭环响应性。
    *   相比之下，$\pi_{0.5}$ 仅表现出<strong>边际收益 (marginal gains)</strong>，因为其显著更大的主干网络会导致高推理延迟，这限制了重叠推理和时间对齐执行的有效性。
*   **结论:** 这些结果表明 `CI` 和 `LAAS` 是广泛适用的执行机制，但它们的实际效益受到模型底层推理延迟的限制。

# 7. 总结与思考

## 7.1. 结论总结
本文提出了 `DynamicVLA`，一个用于<strong>动态物体操作 (dynamic object manipulation)</strong> 的创新框架，它解决了现有<strong>视觉-语言-动作 (VLA) 模型</strong>在处理动态场景时面临的核心挑战，即<strong>感知与动作执行之间的时序错位 (temporal misalignment)</strong>。

`DynamicVLA` 的主要贡献在于其三项关键设计：
1.  **紧凑型 0.4B 参数主干网络:** 采用卷积视觉编码器 `FastViT` 和轻量级语言模型 `SmolLM2-360M`，实现了空间高效和结构忠实的编码，从而支持<strong>高频率推理 (high-frequency reasoning)</strong>。
2.  <strong>`Continuous Inference`（连续推理）:</strong> 引入了流水线式的执行方案，通过**重叠推理和执行**来消除<strong>块间等待 (inter-chunk waiting)</strong>，确保动作流的连续性并及时适应物体运动。
3.  <strong>`Latent-aware Action Streaming`（潜在感知动作流）:</strong> 提出了一种<strong>延迟感知 (latency-aware)</strong> 的执行机制，通过丢弃过时动作并优先处理最新预测，有效弥合了<strong>感知-执行差距 (perception-execution gap)</strong>，确保了动作与环境的实时对齐。

    此外，为了克服动态操作数据稀缺的问题，本文构建了 `Dynamic Object Manipulation (DOM)` 基准，其自动化数据收集管道高效地生成了 200K 合成回合和 2K 无遥操作的真实世界回合。

在广泛的评估中，`DynamicVLA` 在<strong>响应速度 (response speed)</strong>、<strong>感知 (perception)</strong> 和<strong>泛化 (generalization)</strong> 方面表现出显著优于现有 `VLA` 基线模型的性能。这些结果验证了 `DynamicVLA` 作为通用动态物体操作统一框架的有效性。

## 7.2. 局限性与未来工作
论文作者指出了当前研究的几个局限性，并提出了未来有希望的研究方向：

### 7.2.1. 更高效的 VLA 架构 (More Efficient VLA Architectures)
*   **局限性:** `DynamicVLA` 强调了<strong>延迟感知设计 (latency-aware design)</strong> 在动态操作中的重要性，但实时约束本质上是在<strong>多模态理解 (multimodal understanding)</strong> 和<strong>响应性 (responsiveness)</strong> 之间进行权衡。
*   **未来工作:** 动态任务紧密耦合了感知、推理和执行，需要新的架构和推理方案，以在严格的延迟预算下<strong>保持理解能力 (preserve understanding)</strong>。这意味着需要进一步探索如何在模型小型化和信息处理能力之间找到更好的平衡点。

### 7.2.2. 超越短时动态 (Beyond Short-horizon Dynamics)
*   **局限性:** 当前的公式主要强调<strong>短到中时程的反应性交互 (short- to medium-horizon reactive interaction)</strong>，这暴露了延迟导致的失败，但未能捕捉<strong>长时程动态行为 (longer-horizon dynamic behaviors)</strong>。
*   **未来工作:** 未来的研究应将动态操作扩展到具有持续物体运动的<strong>多阶段任务 (multi-stage tasks)</strong>，并整合<strong>规划 (planning)</strong>、<strong>记忆 (memory)</strong> 和<strong>任务分解 (task decomposition)</strong>，同时保持与语言条件和实时执行约束的兼容性。

### 7.2.3. 超越刚体动态 (Beyond Rigid-Body Dynamics)
*   **局限性:** 当前的数据管道假设<strong>刚体状态估计 (rigid-body state estimation)</strong>，而许多动态任务涉及<strong>非刚体 (non-rigid)</strong> 或<strong>流体动态 (fluid dynamics)</strong>，其持续演变的状态在模拟和真实世界中都难以表示。
*   **未来工作:** 将 `VLA` 模型和数据管道扩展到此类设置仍然是一个开放的挑战。这可能需要开发新的物体表示方法、物理模拟技术和感知算法。

## 7.3. 个人启发与批判
### 7.3.1. 个人启发
1.  **实时性的核心地位:** 这篇论文让我深刻认识到在机器人领域，尤其是在<strong>动态交互任务 (dynamic interaction tasks)</strong> 中，<strong>实时性 (real-time capability)</strong> 和<strong>低延迟 (low latency)</strong> 才是最核心的瓶颈和制胜关键。传统的以高准确率为单一目标的模型设计思路，在实际动态应用中可能寸步难行。将推理和执行分离并进行流水线优化，以及主动处理时间错位，是解决这一问题的有效策略。
2.  **数据生成的重要性:** `DOM` 基准的构建方式令人印象深刻。通过<strong>自动化数据收集管道 (automated data collection pipeline)</strong>，尤其是在真实世界中<strong>无需遥操作 (without teleoperation)</strong> 即可收集大规模动态数据，这为解决机器人学习中的<strong>数据稀缺问题 (data scarcity problem)</strong> 提供了一条高效且可扩展的路径。这对于推动机器人技术从实验室走向实际应用具有重要意义。
3.  **轻量化与高效架构的趋势:** 紧凑型 `VLA` 模型 (`0.4B`) 的成功，结合卷积视觉编码器的选择，预示着未来机器人 `AI` 发展的一个重要趋势：<strong>轻量化 (lightweighting)</strong> 和<strong>高效架构 (efficient architectures)</strong>。这不仅有助于降低部署成本，也为边缘计算和资源受限环境下的机器人部署提供了可能性。

### 7.3.2. 批判与潜在改进
1.  **成功率仍有提升空间:** 尽管 `DynamicVLA` 相比基线取得了显著进步，但其在模拟环境中的平均成功率 (`47.06%`) 以及在某些真实世界任务中（例如 Figure 4 中的 "Gather all ping pong balls into the paper box" 仅为 `50%`）仍有很大的提升空间。这表明在复杂的动态场景下，机器人仍面临相当大的挑战，特别是在高精度和高鲁棒性要求下。未来的工作可能需要探索更强大的策略学习算法或更精细的动作生成机制。
2.  **抗干扰鲁棒性的挑战:** 论文指出，即使对于 `DynamicVLA` 而言，对<strong>环境扰动 (environmental perturbations)</strong> 的鲁棒性 (`26.5%`) 仍然具有挑战性。这可能意味着模型在处理未预期的外部干扰或非结构化环境噪声方面仍存在不足。未来的研究可以探索更强大的<strong>扰动感知 (perturbation-aware)</strong> 学习方法，或引入额外的<strong>传感器模态 (sensor modalities)</strong>（如触觉、力觉）来增强对环境变化的感知。
3.  **Sim-to-Real 的进一步弥合:** 尽管论文努力通过“真实世界模拟器”和共享状态机控制器来弥合<strong>模拟到真实 (sim-to-real) 差距</strong>，但真实世界实验的成功率通常低于模拟环境（例如，模拟平均成功率 `47.06%`，真实世界平均 `~50%` 但并非所有任务）。这提示 `Sim-to-Real` 领域仍有改进空间，可能需要更精确的物理模拟、更全面的领域随机化或更强大的真实世界数据增强技术。
4.  **长时程和复杂任务的挑战:** 论文明确指出当前模型主要侧重于<strong>短到中时程的反应性交互 (short- to medium-horizon reactive interaction)</strong>。对于涉及更长时间规划、多阶段决策或复杂语义理解的长时程任务，如何有效整合 `DynamicVLA` 的低延迟能力与高层规划和记忆机制，将是一个巨大的挑战。这可能需要结合符号规划、深度强化学习或更先进的记忆网络。
5.  **非刚体动态的探索:** 论文的局限性也提到了<strong>非刚体 (non-rigid)</strong> 和<strong>流体动态 (fluid dynamics)</strong>，这是机器人操作中一个极其复杂但富有前景的领域。当前基于 6D 姿态和速度的刚体状态估计无法直接应用于此类场景。未来的研究可能需要探索<strong>基于点云 (point cloud-based)</strong>、<strong>网格 (mesh-based)</strong> 或<strong>神经场 (neural fields)</strong> 的表示方法，以及专门针对非刚体物理的建模和控制策略。