# 1. 论文基本信息

## 1.1. 标题
**Motus: A Unified Latent Action World Model** (Motus: 一个统一的潜在动作世界模型)

论文标题直接点明了其核心内容：提出了一个名为 `Motus` 的模型，其关键特性是“统一” (Unified) 和“潜在动作世界模型” (Latent Action World Model)。这意味着该模型旨在将多种功能整合到一个框架内，并且其核心机制涉及从数据中学习一种抽象的、非直接控制信号的“潜在动作”，并利用此动作来构建一个能够预测世界动态的模型。

## 1.2. 作者
Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, Hongyan Zhao, Hanyu Liu, Zhizhong Su, Lei Ma, Hang Su, Jun Zhu.

作者团队主要来自清华大学，并有来自北京大学和地平线机器人 (Horizon Robotics) 的研究人员。这是一个规模庞大的合作团队，其中多位被标记为共同第一作者 (Joint first authors) 和共同项目负责人 (Joint project lead)，表明这是一个大型、复杂的工程项目。主要通讯作者朱军 (Jun Zhu) 和苏航 (Hang Su) 是清华大学人工智能领域的知名教授，他们的研究背景保证了工作的学术深度和影响力。

## 1.3. 发表期刊/会议
论文目前发布于 `arXiv`，这是一个预印本服务器。`arXiv` 上的论文未经同行评审，但通常是研究成果的最早发布形式。从论文的质量和主题来看，其目标投递会议很可能是机器人或机器学习领域的顶级会议，如 CoRL (Conference on Robot Learning)、ICLR (International Conference on Learning Representations)、NeurIPS (Conference on Neural Information Processing Systems) 或 ICRA (International Conference on Robotics and Automation)。

## 1.4. 发表年份
预印本提交日期 (Published at UTC): 2025-12-15T06:58:40.000Z。这个日期是未来的，这在 `arXiv` 上通常是作者在提交时设置的一个占位符或系统自动生成的未来日期，实际提交时间应远早于此。从参考文献的年份看，这项工作反映了2024-2025年期间的技术发展趋势。

## 1.5. 摘要
一个通用的具身智能体 (embodied agent) 必须作为一个统一的系统来运作，但当前的方法通常为理解、世界建模和控制等功能构建孤立的模型。这种**碎片化**阻碍了多模态生成能力的统一，也妨碍了从大规模、异构数据中学习。

为此，论文提出了 `Motus`，一个统一的潜在动作世界模型，它利用了现有的通用预训练模型和丰富的、可共享的运动信息。`Motus` 的核心创新包括：
1.  **统一架构**: 引入一种<strong>混合变换器 (Mixture-of-Transformer, MoT)</strong> 架构，集成了三个专家：<strong>理解 (understanding)</strong>、<strong>视频生成 (video generation)</strong> 和<strong>动作 (action)</strong>。
2.  **灵活模式**: 采用类似 `UniDiffuser` 风格的调度器 (scheduler)，使其能够灵活地在五种不同的建模模式之间切换：世界模型 (WM)、视觉-语言-动作模型 (VLA)、逆动力学模型 (IDM)、视频生成模型 (VGM) 和视频-动作联合预测模型。
3.  **潜在动作**: 利用<strong>光流 (optical flow)</strong> 来学习<strong>潜在动作 (latent actions)</strong>，将其作为像素级的“增量动作” (delta action)，从而实现了大规模的动作预训练。
4.  **训练配方**: 提出了一套包含**三阶段训练流程**和**六层数据金字塔**的训练方法。

    实验结果表明，`Motus` 在模拟环境（比 X-VLA 提升15%，比 $\pi_{0.5}$ 提升45%）和真实世界场景（提升11%~48%）中均优于当前最先进的方法。这证明了将所有功能和先验知识进行统一建模，能够显著提升下游机器人任务的性能。

## 1.6. 原文链接
*   **原文链接:** [https://arxiv.org/abs/2512.13030](https://arxiv.org/abs/2512.13030)
*   **PDF 链接:** [https://arxiv.org/pdf/2512.13030v2](https://arxiv.org/pdf/2512.13030v2)
*   **发布状态:** 预印本 (Preprint)。

# 2. 整体概括

## 2.1. 研究背景与动机
当前，构建通用具身智能体（如能完成多种任务的机器人）面临一个核心困境：**模型能力的碎片化**。一个理想的智能体应该像人一样，能够无缝地整合多种认知能力：理解指令、观察世界、想象未来可能发生的事、预测自己动作的后果，并最终做出决策。然而，现有的研究范式将这些能力割裂开来：
*   <strong>视觉-语言-动作模型 (VLA)</strong>：直接学习从视觉和语言到动作的映射，但缺乏对世界动态的深入理解和规划能力。
*   <strong>世界模型 (WM)</strong>：可以预测未来，但在没有强大先验知识的情况下，从零开始学习一个准确的世界模型非常困难。
*   <strong>视频生成模型 (VGM)</strong>：能生成逼真的未来视频，但通常不直接与机器人的动作空间相连。
*   <strong>逆动力学模型 (IDM)</strong>：能从观测到的状态变化中推断动作，但本身不具备预测或规划能力。

    这种碎片化导致了两个关键问题（也是本文明确指出的**两大挑战**）：

1.  **挑战一：统一多模态生成能力的困难**。如何将上述 VLA、WM、IDM、VGM 等五种核心能力整合到一个统一的框架中？虽然已有工作（如 `UWM`）尝试统一，但它们通常是从零开始训练，或者依赖于较小的基础模型，缺乏利用现有强大的<strong>视觉语言模型 (VLM)</strong> 和<strong>视频生成模型 (VGM)</strong> 所携带的丰富先验知识。这使得它们的“世界知识”不够全面。

2.  **挑战二：异构数据的利用难题**。具身智能需要从各种来源学习，包括互联网视频、人类第一视角演示视频、不同机器人的操作轨迹等。这些数据是<strong>异构 (heterogeneous)</strong> 的：
    *   **动作空间不同**：不同机器人（甚至人和机器人）的动作指令（维度、范围、语义）千差万别。
    *   **标签缺失**：海量的互联网视频和人类演示视频包含丰富的物理交互知识，但**没有动作标签**。
        这使得直接预训练一个通用的“动作专家”变得极其困难，限制了模型从大规模数据中学习通用运动先验的能力。

`Motus` 的**切入点**正是为了解决这两个核心挑战。它提出了一种“集大成”的思路：<strong>不从零开始，而是通过一个巧妙的架构将已有的强大预训练模型（专家）融合起来，并设计一种通用的动作表示（潜在动作）来打通异构数据之间的壁垒。</strong>

## 2.2. 核心贡献/主要发现
`Motus` 的核心贡献可以概括为**一个统一模型**和**一套可扩展的训练配方**。

1.  <strong>提出了一个统一的具身智能基础模型 <code>Motus</code></strong>：
    *   **架构创新**：通过 <strong>混合变换器 (MoT)</strong> 和 <strong>三模型联合注意力 (Tri-model Joint Attention)</strong>，首次将强大的预训练 VLM（负责理解）、VGM（负责生成/想象）和一个可训练的动作专家整合进一个统一的生成模型中，同时保留了各自的专业能力。
    *   **功能统一**：利用类似 `UniDiffuser` 的调度机制，使单个模型能够在推理时灵活扮演五种不同的角色（VLA、WM、IDM、VGM、联合预测），实现了功能的完全统一，解决了**挑战一**。

2.  **提出了一套可扩展的机器人学习配方**：
    *   **潜在动作表示**：创造性地使用<strong>光流 (optical flow)</strong> 作为桥梁，学习一种与具体机器人形态无关的<strong>潜在动作 (latent actions)</strong>。这种表示捕捉了视觉层面的运动模式，使得模型可以从**无动作标签**的视频数据中学习通用的物理交互知识。
    *   **数据和训练流程**：设计了**六层数据金字塔**来组织从网络视频到特定机器人数据的异构数据源，并配合**三阶段训练流程**（视频预训练、潜在动作预训练、特定机器人微调），为动作专家提供了像 VLM 和 VGM 一样的“大规模预训练”可能性，有效解决了**挑战二**。

        **主要发现**：实验结果有力地证明，这种**统一建模**和**利用大规模异构数据进行预训练**的策略是极其有效的。通过融合通用多模态先验（来自 VLM/VGM）和领域特定先验（来自机器人数据），`Motus` 在模拟和真实机器人任务上的性能远超那些模型功能单一或无法利用无标签视频的先前方法。这表明，构建更通用、更强大的具身智能体，<strong>“统一”</strong>和“预训练”是关键路径。

# 3. 预备知识与相关工作

## 3.1. 基础概念
为了理解 `Motus`，我们需要了解以下几个核心概念：

*   <strong>具身智能 (Embodied Intelligence)</strong>: 指的是能够与物理世界或虚拟环境进行交互的智能系统（如机器人）。它不仅需要具备“思考”能力（如语言理解、推理），还需要具备“行动”能力（感知环境、执行动作）。

*   **五种核心建模范式**: 论文的核心是将以下五种概率分布的建模统一起来。
    *   **VLA (Vision-Language-Action Model)**: 视觉-语言-动作模型。给定当前观测 $\pmb{o}_t$ 和语言指令 $\ell$，预测未来一系列动作 $\pmb{a}_{t+1:t+k}$。其目标是建模 $p(\pmb{a}_{t+1:t+k} | \pmb{o}_t, \ell)$。这是策略学习的直接形式。
    *   **WM (World Model)**: 世界模型。给定当前观测 $\pmb{o}_t$ 和未来一系列动作 $\pmb{a}_{t+1:t+k}$，预测未来的观测 $\pmb{o}_{t+1:t+k}$。其目标是建模 $p(\pmb{o}_{t+1:t+k} | \pmb{o}_t, \pmb{a}_{t+1:t+k})$。它回答了“如果我这么做，世界会变成什么样？”。
    *   **IDM (Inverse Dynamics Model)**: 逆动力学模型。给定当前观测 $\pmb{o}_t$ 和未来的观测 $\pmb{o}_{t+1:t+k}$，推断出导致这一变化的动作 $\pmb{a}_{t+1:t+k}$。其目标是建模 $p(\pmb{a}_{t+1:t+k} | \pmb{o}_{t:t+k})$。它回答了“要达到那个状态，我需要做什么动作？”。
    *   **VGM (Video Generation Model)**: 视频生成模型。给定当前观测 $\pmb{o}_t$ 和语言指令 $\ell$，生成（或“想象”）未来可能的视频片段 $\pmb{o}_{t+1:t+k}$。其目标是建模 $p(\pmb{o}_{t+1:t+k} | \pmb{o}_t, \ell)$。它不依赖于动作，而是纯粹基于指令进行想象。
    *   **视频-动作联合预测模型**: 同时预测未来的视频和动作。其目标是建模联合概率分布 $p(\pmb{o}_{t+1:t+k}, \pmb{a}_{t+1:t+k} | \pmb{o}_t, \ell)$。

*   <strong>扩散模型 (Diffusion Models) 与 校正流 (Rectified Flow)</strong>:
    *   **扩散模型**是一类强大的生成模型。其基本思想是：首先，通过一个“前向过程”逐步向真实数据（如图像）中添加噪声，直到数据变成纯粹的高斯噪声；然后，训练一个神经网络来学习“反向过程”，即从纯噪声开始，逐步去除噪声，最终生成一个逼真的数据样本。
    *   <strong>校正流 (Rectified Flow)</strong> 是对扩散模型的一种改进。它将数据点和噪声点视为两个分布，并学习它们之间的直线路径。相比传统扩散模型弯曲的路径，直线路径使得训练更稳定、生成速度更快，因此被 `Motus` 采用作为其生成框架的基础。

*   <strong>混合变换器 (Mixture-of-Transformers, MoT)</strong>: 一种模型架构，它将多个专门的 Transformer 模型（称为“专家”）组合在一起。每个专家负责处理特定的任务或数据模态。在 `Motus` 中，这三个专家分别是处理理解、视频生成和动作的 Transformer。它们通过共享某些层（如注意力层）来协作，实现知识的融合。

*   <strong>光流 (Optical Flow)</strong>: 是计算机视觉中的一个概念，用于描述连续两帧图像之间每个像素的运动。它表现为一个二维向量场，其中每个向量表示一个像素从第一帧到第二帧的位移。光流提供了一种密集、纯粹的运动表示，与物体的具体外观（颜色、纹理）无关。

## 3.2. 前人工作
`Motus` 的研究建立在以下几个领域的工作之上：

*   <strong>统一多模态模型 (Unified Multimodal Models)</strong>:
    *   `Bagel` [18] 等工作尝试在一个生成框架内统一处理多种模态（如文本、图像）。`Bagel` 使用 `MoT` 架构，通过共享多头自注意力层来融合理解专家和生成专家。`Motus` 的架构设计受到了这类工作的启发，但将其应用到了更复杂的机器人领域，并集成了动作模态。
    *   `UWM` (Unified World Models) [64] 是一个重要的先行者，它首次尝试在一个基于扩散模型的骨干网络中统一上述五种机器人建模范式。然而，`UWM` 的主要局限在于它是从零开始训练的，未能有效利用现有大型预训练模型的强大先验知识。

*   <strong>具身基础模型 (Embodied Foundation Models)</strong>:
    *   `VLA` 类模型，如 `RT-X`、`OpenVLA` [27]、`X-VLA` [60] 等，主要利用强大的 VLM 作为主干网络来学习从视觉和语言到动作的映射。它们在模仿学习方面取得了巨大成功，但缺乏显式的世界建模和规划能力。
    *   `VGM` 类模型，通过生成未来视频来辅助决策。
    *   $\mathcal{F}_1$ [32] 是一个较新的进展，它结合了 `VLA` 和 `IDM`，通过显式地想象未来的视觉状态来生成动作。但它没有统一世界模型或视频生成模型，因此其统一性仍不完整。

*   <strong>潜在动作模型 (Latent Action Models)</strong>:
    *   早期的潜在动作模型通过一个自编码器结构，在给定当前帧和潜在动作的条件下，尝试重建下一帧图像。其思想是，如果潜在动作能帮助模型准确预测未来，那么它一定捕捉到了与任务相关的动态信息。
    *   **挑战**: 直接重建 RGB 图像会引入大量与任务无关的外观信息（如光照、背景），干扰动作的学习。
    *   **改进**: 为了解决这个问题，后续工作尝试了不同的重建目标，如 `DINOv2` 特征 [11, 15]、物体关键点 [17] 等，这些表示比原始像素更抽象、更关注语义。`LAOM` [34] 则引入少量真实动作标签来引导模型关注与机器人活动相关的模式。
    *   `Motus` 的创新在于使用**光流**作为重建目标。光流是一种天然的、通用的运动表示，这使得模型可以跨越不同的机器人形态（embodiment）学习通用的运动知识，为大规模预训练提供了可能。

## 3.3. 技术演进
具身智能领域的技术演进路线大致如下：
1.  **早期模仿学习**: 从专家演示中学习简单的策略，通常针对特定任务和机器人。
2.  **VLM 驱动的 VLA**: 随着大型 VLM 的兴起，研究人员开始利用其强大的视觉和语言理解能力作为策略模型的骨干网络，使得模型能够理解更复杂的指令并泛化到新场景，催生了如 `RT-2`、$\pi_{0.5}$、`X-VLA` 等一系列模型。
3.  **世界模型的引入**: 为了让机器人具备规划和推理能力，研究开始重新关注世界模型，即学习环境的动态变化规律。
4.  **走向统一**: 近期，研究趋势开始向“统一模型”发展。研究者意识到单一范式（如纯 VLA 或纯 WM）的局限性，开始探索如何将多种能力整合到一个模型中。$\mathcal{F}_1$ 和 `UWM` 是这一趋势的代表。

    `Motus` 正是处在**走向统一**这一技术脉络的前沿。它不仅追求功能的统一，还进一步解决了如何让统一模型有效利用现有生态（强大的预训练模型）和大规模异构数据的问题，这是之前工作未能充分解决的关键环节。

## 3.4. 差异化分析
`Motus` 与相关工作的主要区别和创新点如下：

| 特性 | Motus | UWM [64] | $\mathcal{F}_1$ [32] | X-VLA [60] / $\pi_{0.5}$ [8] |
| :--- | :--- | :--- | :--- | :--- |
| **功能统一性** | **5种** (VLA, WM, IDM, VGM, 联合) | **5种** (VLA, WM, IDM, VGM, 联合) | 2种 (VLA, IDM) | 1种 (VLA) |
| **利用预训练模型** | **是** (通过 MoT 融合 VLM, VGM) | 否 (从零训练或依赖小模型) | 是 (基于 VLM) | 是 (基于 VLM) |
| **架构** | <strong>混合专家 (MoT)</strong> | 单一 Transformer 骨干 | 想象-推断两阶段 | 单一 Transformer 骨干 |
| **利用无标签数据** | **是** (通过光流学习潜在动作) | 否 | 否 | 否 |
| **动作预训练** | **是** (大规模) | 否 | 否 | 否 |
| **核心创新** | **融合预训练专家 + 潜在动作** | 首次提出统一5种功能的框架 | 结合VLA和IDM，显式想象未来 | 将VLM扩展为强大的VLA |

总结来说，`Motus` 最大的差异化优势在于其**务实且可扩展的“集大成”方法**。它不像 `UWM` 那样试图从零构建一切，而是聪明地站在巨人的肩膀上（VLM, VGM）；同时，它通过创新的潜在动作表示，解决了 `X-VLA` 等模型无法利用海量无标签视频数据的核心痛点。

# 4. 方法论

## 4.1. 方法原理
`Motus` 的核心思想是**通过融合与解耦，构建一个统一且可扩展的具身智能模型**。

*   <strong>融合 (Fusion)</strong>: 它认为理解、生成和行动是具身智能不可分割的三个方面。因此，它没有设计独立的模块，而是通过一个<strong>混合变换器 (MoT)</strong> 架构，将分别代表这三种能力的三个“专家”模型紧密地融合在一起。融合的核心是<strong>三模型联合注意力 (Tri-model Joint Attention)</strong>，它让三个专家在处理信息时能够“看到”彼此的中间状态，从而实现跨模态的知识交流和互补。

*   <strong>解耦 (Decoupling)</strong>: 尽管功能上是统一的，但 `Motus` 在生成过程中对不同的模态（视频和动作）进行了巧妙的解耦。它借鉴了 `UniDiffuser` 的思想，为视频和动作分配了独立的噪声尺度和去噪时间步。这就像一个调度中心，可以灵活地控制生成过程：可以只生成动作（VLA模式），也可以同时生成视频和动作（联合预测模式），或者根据已知的未来视频反推动作（IDM模式）。这种解耦是实现五种功能灵活切换的关键。

*   <strong>桥梁 (Bridge)</strong>: 为了解决异构数据的利用难题，`Motus` 引入了<strong>潜在动作 (Latent Actions)</strong> 作为桥梁。这个桥梁的核心是<strong>光流 (optical flow)</strong>，一种通用的、与机器人形态无关的运动表示。通过训练一个模型来从光流中提取潜在动作，`Motus` 就能从任何视频（无论是否有动作标签，来自人还是机器人）中学习通用的“运动知识”。这极大地扩展了可用于预训练动作专家的数据规模。

    整体而言，`Motus` 的方法论是一个系统性工程：用 MoT 架构实现**模型层面的统一**，用 UniDiffuser 调度器实现**功能层面的统一**，用潜在动作实现**数据层面的统一**。

## 4.2. 核心方法详解 (逐层深入)

### 4.2.1. Motus 统一模型架构

`Motus` 的整体架构如原文 Figure 1 所示，它是一个基于<strong>校正流 (Rectified Flow)</strong> 的生成模型，其目标是联合预测未来的视频帧序列 $\pmb{o}_{t+1:t+k}$ 和动作序列 $\pmb{a}_{t+1:t+k}$。

![Figure 1. Motus Architecture. Here, $a _ { t } \\ldots a _ { t + k }$ are actions, $z _ { t } \\ldots z _ { t + k }$ are latent actions, and $\\tau _ { v }$ and $\\tau _ { a }$ are the rectified flow timesteps for the video generation model and the action expert, respectively.](images/1.jpg)

<strong>1. 混合变换器 (MoT) 架构</strong>

`Motus` 的骨干网络是一个混合变换器，由三个专家组成：
*   <strong>理解专家 (Understanding Expert)</strong>: 基于一个预训练的<strong>视觉语言模型 (VLM)</strong>，论文中选用了 `Qwen3-VL-2B`。它负责处理当前的视觉观测 $\pmb{o}_t$ 和语言指令 $\ell$，提供对场景和任务的深刻理解。其输出的特征作为后续生成的关键条件。
*   <strong>生成专家 (Generative Expert)</strong>: 基于一个预训练的<strong>视频生成模型 (VGM)</strong>，论文中选用了 `Wan 2.2 5B`。它负责处理和生成视频相关的特征。
*   <strong>动作专家 (Action Expert)</strong>: 这是一个专门为动作建模而设计的 Transformer 模块，其结构与生成专家类似。它负责处理和生成动作序列。

    这三个专家并非完全独立。它们的核心连接机制是<strong>三模型联合注意力 (Tri-model Joint Attention)</strong>。在每个 Transformer 块中，三个专家的多头自注意力 (Multi-Head Self-Attention) 层的键 (Key) 和值 (Value) 矩阵被拼接在一起，然后共享给所有专家。这意味着在计算注意力时，每个专家的查询 (Query) 都可以关注到其他两个专家的信息，从而实现了深度、跨模态的特征融合。

**2. UniDiffuser 风格的调度器与训练目标**

`Motus` 使用校正流进行训练。校正流的基本思想是在真实数据 $x_1$ 和纯噪声 $x_0$ 之间构建一个直线路径。对于任意时间步 $\tau \in [0, T_\tau]$，插值点 $x_\tau$ 可以表示为：
$$
x_\tau = (1 - \tau) x_1 + \tau x_0
$$
其中 $x_0 \sim \mathcal{N}(\mathbf{0}, I)$。模型需要预测这条路径的速度场 (velocity field) $v(x_\tau, \tau)$，其真实值为 $v = x_1 - x_0$。

`Motus` 将此思想应用于视频和动作的联合生成。它为视频观测和动作分配了不同的时间步 $\tau_o$ 和 $\tau_a$ 以及噪声 $\epsilon_o$ 和 $\epsilon_a$。训练的目标是最小化预测速度场与真实速度场之间的均方误差。

模型的总损失函数 $l^\theta$ 由动作损失 $l_{\mathrm{action}}^\theta$ 和观测损失 $l_{\mathrm{obs}}^\theta$ 组成：
$$
l^\theta = l_{\mathrm{action}}^\theta + l_{\mathrm{obs}}^\theta
$$

其中，动作损失定义为：
$$
l_{\mathrm{action}}^\theta = \mathbb{E}_{(\pmb{o}_{t:t+k}, \pmb{a}_{t+1:t+k}, \ell) \sim \mathcal{D}} \left\| v_a^\theta - (\epsilon_a - \pmb{a}_{t+1:t+k}) \right\|_2^2
$$
**符号解释:**
*   $\pmb{o}_{t:t+k}, \pmb{a}_{t+1:t+k}, \ell$ 是从数据集中采样的一段轨迹。
*   $\epsilon_a \sim \mathcal{N}(\mathbf{0}, I)$ 是采样的标准高斯噪声。
*   $\pmb{a}_{t+1:t+k}$ 是真实的动作序列（相当于校正流中的 $x_1$）。
*   $v_a^\theta$ 是模型预测的动作速度场。
*   $(\epsilon_a - \pmb{a}_{t+1:t+k})$ 是真实动作路径的速度场（相当于 $x_0 - x_1$, 论文中符号为 $v=x_1-x_0$ 的负值，但优化目标是L2损失，符号不影响结果）。

    同样地，观测损失定义为：
$$
l_{\mathrm{obs}}^\theta = \mathbb{E}_{(\pmb{o}_{t:t+k}, \pmb{a}_{t+1:t+k}, \ell) \sim \mathcal{D}} \left\| v_o^\theta - (\epsilon_o - \pmb{o}_{t+1:t+k}) \right\|_2^2
$$
**符号解释:**
*   $\epsilon_o \sim \mathcal{N}(\mathbf{0}, I)$ 是采样的标准高斯噪声。
*   $\pmb{o}_{t+1:t+k}$ 是真实的未来视频帧序列。
*   $v_o^\theta$ 是模型预测的视频速度场。

    通过为视频和动作分配不同的时间步和噪声，`Motus` 可以在推理时灵活地控制生成过程，从而实现五种不同的建模模式（具体算法见附录 Algorithm 2-6）。例如，在执行 VLA 任务时，模型会将观测的时间步 $\tau_o$ 保持在最大值（纯噪声），而逐步减小动作的时间步 $\tau_a$ 从最大值到0，从而只生成动作。

<strong>3. 动-密-视-疏预测 (Action-Dense Video-Sparse Prediction)</strong>

为了提高效率并平衡不同模态的重要性，`Motus` 采用了一种非对称的采样策略。在训练和推理时，动作序列的采样频率远高于视频帧。例如，视频帧率为 5Hz，而动作帧率为 30Hz。这确保了动作和视频的词元 (token) 数量大致平衡，避免模型过度关注视频生成而忽略了动作预测的精度。

![Figure 2. Action-Dense Video-Sparse Prediction. The sampling rates for video frames and actions differ.](images/2.jpg)
*该图像是示意图，展示了采样帧与采样动作在时间轴上的不同采样频率。上方为采样帧，底部为采样动作，二者在时间线上呈现出不一致的间隔，表明动作的稠密性与视频帧的稀疏性。*

### 4.2.2. 潜在动作学习

为了利用海量的无标签视频数据，`Motus` 设计了一种从光流中学习潜在动作的方法。

![Figure 3. The Latent Action VAE.](images/3.jpg)

**1. 基于光流的表示**

整个流程如原文 Figure 3 所示：
*   **计算光流**: 使用现成的光流估计算法 `DPFlow` [33] 计算连续视频帧之间的光流。
*   **压缩编码**: 使用一个<strong>深度卷积变分自编码器 (DC-AE)</strong> 对光流进行压缩。编码器 (Encoder) 将高维的光流图像压缩成一个低维的潜在表示。
*   **生成潜在动作**: 编码器输出的特征被一个轻量级网络进一步投影到一个14维的向量，这个向量就是<strong>潜在动作 (latent action)</strong> $z_t$。这个维度与典型机器人动作空间的维度大致匹配，有助于后续的对齐。

**2. 训练与分布对齐**

潜在动作VAE的训练是一个多目标任务，其总损失函数为：
$$
\mathcal{L} = \mathcal{L}_{\mathrm{recon}} + \lambda_a \| a_{\mathrm{real}} - a_{\mathrm{pred}} \|^2 + \beta \mathcal{L}_{\mathrm{KL}}
$$
**符号解释:**
*   $\mathcal{L}_{\mathrm{recon}}$: **重建损失**。这是主要的自监督信号，要求解码器 (Decoder) 能够从潜在动作中准确地重建出原始的光流图像。它驱使潜在动作捕捉到视觉上的运动信息。
*   $\| a_{\mathrm{real}} - a_{\mathrm{pred}} \|^2$: **对齐损失**。这是弱监督信号。对于一小部分（10%）带有真实动作标签 $a_{\mathrm{real}}$ 的数据，模型需要预测动作 $a_{\mathrm{pred}}$ 并最小化与真实动作的差距。这会“锚定”潜在空间，使其学习到的运动模式与真实可执行的机器人动作相关联。
*   $\mathcal{L}_{\mathrm{KL}}$: **KL散度正则化项**。这是 VAE 的标准部分，用于约束潜在空间的分布，使其接近标准正态分布，从而使得潜在空间更加规整和平滑。
*   $\lambda_a, \beta$: 控制各项损失权重的超参数。

    通过这种混合监督的方式，`Motus` 的潜在动作既能从大量无标签视频中学到通用的运动先验，又能通过少量有标签数据的引导，确保这些先验与机器人控制相关，最终成为连接视觉动态和物理动作的有效桥梁。

### 4.2.3. 三阶段训练流程与数据金字塔

`Motus` 的训练过程被精心设计为三个阶段，并使用一个六层数据金字塔来组织数据。

<strong>数据金字塔 (Embodied Data Pyramid)</strong>

![Figure 4. The Embodied Data Pyramid categorizes data into six levels, from Level 1 at the base to Level 6 at the top. Data quantity decreases from bottom to top, while data quality increases. The order of Levels 3 and 4 may sometimes vary.](images/4.jpg)
*该图像是一个示意图，展示了‘体现数据金字塔’的六个层级，从底部的‘用户中心视频’到顶部的‘目标机器人任务轨迹数据’，数据数量逐层减少，数据质量逐层增加。层级3和4的顺序有时可能会变化。*

数据被分为六个层级，从底层到顶层，数据量递减，但与目标任务的相关性（质量）递增：
*   **Level 1: Web Data**: 网页规模的图文对和视频文本对，用于预训练 VLM 和 VGM 的基础模型。
*   **Level 2: Egocentric Human Videos**: 人类第一视角视频，提供丰富的交互知识。
*   **Level 3: Synthetic Data**: 模拟器中生成的机器人数据。
*   **Level 4: Task-agnostic Data**: 任务无关的机器人探索数据，覆盖了机器人的整个动作空间。
*   **Level 5: Multi-Robot Task Trajectory Data**: 来自多种不同机器人的任务导向数据。
*   **Level 6: Target-Robot Task Trajectory Data**: 目标机器人上采集的少量高质量任务数据。

**三阶段训练流程**

以下是原文 Table 1 的内容，详细说明了每个阶段的训练细节：

<table>
<thead>
<tr>
<th>阶段</th>
<th>数据</th>
<th>训练内容</th>
</tr>
</thead>
<tbody>
<tr>
<td>预训练基础模型 (现成)</td>
<td>Level 1: Web Data</td>
<td>VGM and VLM</td>
</tr>
<tr>
<td><strong>Stage 1 (视频生成)</strong></td>
<td>Level 2, 3, 5: Egocentric Human Videos, Synthetic Data, Multi-Robot Data</td>
<td>只训练 VGM</td>
</tr>
<tr>
<td><strong>Stage 2 (使用潜在动作统一训练)</strong></td>
<td>Level 2, 3, 4, 5: Egocentric Human Videos, Synthetic Data, Task-agnostic Data, Multi-Robot Data</td>
<td>训练 Motus (全部3个专家，使用潜在动作)</td>
</tr>
<tr>
<td><strong>Stage 3 (SFT)</strong></td>
<td>Level 6: Target-Robot Task Trajectory Data</td>
<td>微调 Motus (全部3个专家，使用真实动作)</td>
</tr>
</tbody>
</table>

*   **Stage 1: 学习视觉动态**: 在多机器人和人类视频数据上微调预训练的 VGM。目的是让生成专家适应机器人交互场景，能够生成逼真的未来视频。
*   **Stage 2: 学习动作表示**: 这是核心的预训练阶段。在包含潜在动作的大规模异构数据上训练整个 `Motus` 模型（VLM 专家被冻结）。这一阶段将从视频中学到的通用运动知识注入到动作专家的参数中。
*   **Stage 3: 专精于目标机器人**: 在目标机器人的特定数据上对整个模型进行微调 (Supervised Fine-Tuning, SFT)。这一步将预训练阶段学到的通用知识适配到目标机器人的具体动力学和运动学特性上。

# 5. 实验设置

## 5.1. 数据集
`Motus` 的训练和评估使用了跨越模拟与现实世界的多种数据集。

*   <strong>预训练数据集 (Pre-training Datasets)</strong>:
    *   `Egodex` [24]: 大规模第一视角人类视频，用于学习通用的人类操作和交互模式。
    *   `Agibot` [1], `RDT` [31], `RoboMind` [48]: 包含了多种机器人（Genie-1, Aloha, Franka）执行任务的轨迹数据，用于学习跨机器人形态的运动知识。
    *   `RoboTwin` [14]: 在模拟环境中生成的机器人数据，提供了大量的可控场景。
    *   `Task-Agnostic Data` [39]: 通过在模拟器中随机采样动作空间生成的机器人数据，用于帮助对齐潜在动作和真实动作空间。
    *   `In-house Data`: 作者团队自己收集的目标机器人数据，用于最后的微调。

*   <strong>评估数据集/环境 (Evaluation Datasets/Environments)</strong>:
    *   **RoboTwin 2.0 (Simulation)**: 一个具挑战性的模拟基准，包含50个操作任务。评估在两种设置下进行：`Clean`（干净场景）和 `Randomized`（随机化场景，包括随机背景、桌面杂物、光照变化等）。随机化设置能更好地测试模型的泛化能力。
    *   **Real-World Robots**:
        *   `AC-One`: 一个双臂机器人平台。
        *   `Agilex-Aloha-2`: 另一个双臂机器人平台。
            在这两个平台上评估了一系列复杂的长时程任务，如“叠毛巾”、“用滴漏咖啡机煮咖啡”、“用研磨机磨咖啡豆”等。下图（原文 Figure 5）展示了部分真实世界任务的定义。

            ![Figure 5. Task Definitions and Visualizations. For each task, we describe its language instruction and definitions of each sub-task.](images/5.jpg)
            *该图像是任务定义的示意图，展示了机器人执行三个不同任务的过程：使用咖啡机煮咖啡、触摸指定的键盘以及将面包放入烤箱。每个任务都有相应的步骤说明，机器人需按照指令完成操作。*

    *   **LIBERO-Long**: 一个专注于长时程操作任务的基准，要求模型具备多阶段决策和技能迁移能力。
    *   **VLABench**: 一个评估通用语言条件操作任务的基准，考察模型在操作技能、视觉理解、常识推理等多方面的能力。

## 5.2. 评估指标
论文使用了多种指标来评估模型的不同方面能力。

*   <strong>成功率 (Success Rate, %)</strong>:
    *   **概念定义**: 这是评估机器人策略模型最直接的指标，衡量在多次尝试中，机器人成功完成指定任务的比例。
    *   **数学公式**:
        $$
        \text{Success Rate} = \frac{\text{Number of Successful Trials}}{\text{Total Number of Trials}} \times 100\%
        $$
    *   **符号解释**:
        *   `Number of Successful Trials`: 成功完成任务的试验次数。
        *   `Total Number of Trials`: 总的试验次数。

*   <strong>部分成功率 (Partial Success Rate, %)</strong>:
    *   **概念定义**: 对于复杂、长时程的任务，完全成功的难度很大。部分成功率将一个大任务分解为多个子任务或子目标。模型每完成一个子任务，就能获得相应的分数。最终的得分是所有试验中获得分数的平均值。这能更细致地衡量模型在复杂任务上的能力。
    *   **数学公式**:
        $$
        \text{Partial Success Rate} = \left( \frac{1}{N} \sum_{i=1}^{N} \text{Score}_i \right) \times 100\%
        $$
    *   **符号解释**:
        *   $N$: 总的试验次数。
        *   $\text{Score}_i$: 第 $i$ 次试验获得的分数（通常在 [0, 1] 区间，0代表完全失败，1代表完全成功）。

*   <strong>视频生成质量指标 (用于评估世界模型模式)</strong>:
    *   **Fréchet Inception Distance (FID)**: 衡量生成图像与真实图像在特征空间分布上的相似度。**值越低越好**。
    *   **Fréchet Video Distance (FVD)**: FID 的视频版本，衡量生成视频与真实视频在时空特征上的分布相似度。**值越低越好**。
    *   **Structural Similarity Index (SSIM)**: 衡量两张图像在结构、亮度和对比度上的相似性。**值越高越好**（范围-1到1）。
    *   **Learned Perceptual Image Patch Similarity (LPIPS)**: 利用深度网络提取特征，计算两张图像在感知上的相似度，更符合人类视觉。**值越低越好**。
    *   **Peak Signal-to-Noise Ratio (PSNR)**: 衡量图像质量的经典指标，基于像素级的均方误差。**值越高越好**。

*   <strong>均方误差 (Mean Squared Error, MSE) (用于评估IDM模式)</strong>:
    *   **概念定义**: 用于衡量模型预测的动作与真实动作之间的差异。
    *   **数学公式**:
        $$
        \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
        $$
    *   **符号解释**:
        *   $n$: 动作向量的维度或序列的长度。
        *   $y_i$: 真实的动作值。
        *   $\hat{y}_i$: 模型预测的动作值。

## 5.3. 对比基线
论文将 `Motus` 与以下几种代表性的模型进行了比较：
*   **$\pi_{0.5}$ [8]**: 一个强大的 VLA 模型，代表了基于 VLM 的模仿学习方法的先进水平。
*   **X-VLA [60]**: 另一个先进的 VLA 模型，特别擅长跨机器人形态的泛化。
*   <strong>w/o Pretrain (从零训练)</strong>: `Motus` 模型的一个变体，移除了所有的预训练阶段，直接在目标任务数据上进行训练。这是一个重要的消融实验基线，用于验证预训练的价值。
*   <strong>Stage1 (仅第一阶段预训练)</strong>: `Motus` 的另一个变体，只完成了第一阶段的 VGM 微调，但没有进行第二阶段的潜在动作预训练。用于验证潜在动作预训练的贡献。
*   **ResNet18+MLP / DINOv2+MLP**: 在评估 IDM 性能时，与两个专门训练的逆动力学模型基线进行比较。

# 6. 实验结果与分析

## 6.1. 核心结果分析

### 6.1.1. 模拟环境评估

在 `RoboTwin 2.0` 模拟基准上的评估结果充分展示了 `Motus` 的优越性。以下是原文 Table 2 的完整数据，展示了部分任务的成功率：

<table>
<thead>
<tr>
<th rowspan="2">Simulation Task</th>
<th colspan="2">π0.5</th>
<th colspan="2">X-VLA</th>
<th colspan="2">w/o Pretrain</th>
<th colspan="2">Stage1</th>
<th colspan="2">Motus</th>
</tr>
<tr>
<th>Clean</th>
<th>Rand.</th>
<th>Clean</th>
<th>Rand.</th>
<th>Clean</th>
<th>Rand.</th>
<th>Clean</th>
<th>Rand.</th>
<th>Clean</th>
<th>Rand.</th>
</tr>
</thead>
<tbody>
<tr>
<td>Place Dual Shoes</td>
<td>12%</td>
<td>7%</td>
<td>79%</td>
<td>88%</td>
<td>78%</td>
<td>80%</td>
<td>94%</td>
<td>94%</td>
<td>93%</td>
<td>87%</td>
</tr>
<tr>
<td>Move Stapler Pad</td>
<td>16%</td>
<td>18%</td>
<td>78%</td>
<td>73%</td>
<td>49%</td>
<td>37%</td>
<td>75%</td>
<td>68%</td>
<td>83%</td>
<td>85%</td>
</tr>
<tr>
<td>Stack Blocks Two</td>
<td>48%</td>
<td>56%</td>
<td>92%</td>
<td>87%</td>
<td>96%</td>
<td>94%</td>
<td>99%</td>
<td>99%</td>
<td>100%</td>
<td>98%</td>
</tr>
<tr>
<td>Scan Object</td>
<td>42%</td>
<td>38%</td>
<td>14%</td>
<td>36%</td>
<td>42%</td>
<td>50%</td>
<td>56%</td>
<td>69%</td>
<td>67%</td>
<td>66%</td>
</tr>
<tr>
<td>Place Object Stand</td>
<td>74%</td>
<td>65%</td>
<td>86%</td>
<td>88%</td>
<td>91%</td>
<td>93%</td>
<td>93%</td>
<td>96%</td>
<td>98%</td>
<td>97%</td>
</tr>
<tr>
<td>Place Fan</td>
<td>25%</td>
<td>36%</td>
<td>80%</td>
<td>75%</td>
<td>77%</td>
<td>85%</td>
<td>77%</td>
<td>85%</td>
<td>91%</td>
<td>87%</td>
</tr>
<tr>
<td>Move Pillbottle Pad</td>
<td>33%</td>
<td>29%</td>
<td>73%</td>
<td>71%</td>
<td>83%</td>
<td>83%</td>
<td>96%</td>
<td>90%</td>
<td>93%</td>
<td>96%</td>
</tr>
<tr>
<td>Pick Dual Bottles</td>
<td>10%</td>
<td>6%</td>
<td>47%</td>
<td>36%</td>
<td>58%</td>
<td>68%</td>
<td>7%</td>
<td>17%</td>
<td>96%</td>
<td>90%</td>
</tr>
<tr>
<td>Blocks Ranking Rgb ...50 tasks)</td>
<td>43%</td>
<td>35%</td>
<td>83%</td>
<td>83%</td>
<td>92%</td>
<td>88%</td>
<td>97%</td>
<td>98%</td>
<td>99%</td>
<td>97%</td>
</tr>
<tr>
<td>Turn Switch</td>
<td>5%</td>
<td>6%</td>
<td>40%</td>
<td>61%</td>
<td>69%</td>
<td>60%</td>
<td>59%</td>
<td>64%</td>
<td>84%</td>
<td>78%</td>
</tr>
<tr>
<td>Pick Diverse Bottles</td>
<td>5%</td>
<td>3%</td>
<td>58%</td>
<td>36%</td>
<td>53%</td>
<td>62%</td>
<td>18%</td>
<td>18%</td>
<td>90%</td>
<td>91%</td>
</tr>
<tr>
<td>Place Bread Basket</td>
<td>48%</td>
<td>56%</td>
<td>81%</td>
<td>71%</td>
<td>73%</td>
<td>83%</td>
<td>89%</td>
<td>87%</td>
<td>91%</td>
<td>94%</td>
</tr>
<tr>
<td>Stack Blocks Three</td>
<td>15%</td>
<td>16%</td>
<td>6%</td>
<td>10%</td>
<td>71%</td>
<td>76%</td>
<td>99%</td>
<td>95%</td>
<td>91%</td>
<td>95%</td>
</tr>
<tr>
<td>Put Bottles Dustbin</td>
<td>12%</td>
<td>9%</td>
<td>74%</td>
<td>77%</td>
<td>36%</td>
<td>33%</td>
<td>34%</td>
<td>24%</td>
<td>81%</td>
<td>79%</td>
</tr>
<tr>
<td>Place Can Basket</td>
<td>19%</td>
<td>25%</td>
<td>49%</td>
<td>52%</td>
<td>46%</td>
<td>62%</td>
<td>66%</td>
<td>55%</td>
<td>81%</td>
<td>76%</td>
</tr>
<tr>
<td>Stamp Seal</td>
<td>36%</td>
<td>23%</td>
<td>76%</td>
<td>82%</td>
<td>80%</td>
<td>88%</td>
<td>93%</td>
<td>95%</td>
<td>93%</td>
<td>92%</td>
</tr>
<tr>
<td>Hanging Mug</td>
<td>3%</td>
<td>3%</td>
<td>23%</td>
<td>27%</td>
<td>14%</td>
<td>10%</td>
<td>37%</td>
<td>25%</td>
<td>38%</td>
<td>38%</td>
</tr>
<tr>
<td>Handover Block</td>
<td>18%</td>
<td>19%</td>
<td>73%</td>
<td>37%</td>
<td>34%</td>
<td>15%</td>
<td>55%</td>
<td>55%</td>
<td>86%</td>
<td>73%</td>
</tr>
<tr>
<td>Stack Bowls Three</td>
<td>33%</td>
<td>35%</td>
<td>76%</td>
<td>86%</td>
<td>90%</td>
<td>74%</td>
<td>86%</td>
<td>83%</td>
<td>79%</td>
<td>87%</td>
</tr>
<tr>
<td>Place Object Basket Open Microwave</td>
<td>35%</td>
<td>37%</td>
<td>79%</td>
<td>71%</td>
<td>83%</td>
<td>82%</td>
<td>82%</td>
<td>84%</td>
<td>95%</td>
<td>91%</td>
</tr>
<tr>
<td><strong>Average (%)</strong></td>
<td><strong>42.98</strong></td>
<td><strong>43.84</strong></td>
<td><strong>72.80</strong></td>
<td><strong>72.84</strong></td>
<td><strong>77.56</strong></td>
<td><strong>77.00</strong></td>
<td><strong>82.26</strong></td>
<td><strong>81.86</strong></td>
<td><strong>88.66</strong></td>
<td><strong>87.02</strong></td>
</tr>
</tbody>
</table>

**分析**:
*   **显著优于基线**: 在最具挑战性的<strong>随机化 (Randomized)</strong> 场景中，`Motus` 的平均成功率达到了 **87.02%**。这相比 `X-VLA` (72.84%) 提升了约 **14.2%** (相对提升约19.5%)，相比 $\pi_{0.5}$ (43.84%) 更是取得了超过 **43%** 的绝对提升。这证明了 `Motus` 的统一架构和预训练策略带来了巨大的性能增益。
*   **泛化能力强**: `Motus` 在干净场景 (88.66%) 和随机化场景 (87.02%) 的性能差距非常小，表明模型具有很强的泛化能力，能够应对环境变化，这得益于其从大规模异构数据中学到的鲁棒先验知识。
*   **预训练的必要性**: `w/o Pretrain` (77.00%) 和 `Stage1` (81.86%) 的性能均低于完整的 `Motus` (87.02%)，这清晰地表明了三阶段训练流程中每一阶段的贡献，特别是第二阶段的潜在动作预训练是提升性能的关键。

### 6.1.2. 真实世界评估

`Motus` 在两个不同的真实机器人平台上的表现进一步验证了其有效性。以下是原文 Table 3 的结果：

<table>
<thead>
<tr>
<th>Task Description</th>
<th>π0.5</th>
<th>w/o Pretrain</th>
<th>Motus</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="4"><strong>AC-One</strong></td>
</tr>
<tr>
<td>Fold Towel</td>
<td>4</td>
<td>1</td>
<td>14.5</td>
</tr>
<tr>
<td>Brew Coffee using Coffee Maker</td>
<td>0</td>
<td>0</td>
<td>62</td>
</tr>
<tr>
<td>Get Water from Water Dispenser</td>
<td>30</td>
<td>8</td>
<td>36</td>
</tr>
<tr>
<td>Place Cube into Plate</td>
<td>46</td>
<td>60</td>
<td>100</td>
</tr>
<tr>
<td>Place Cube into Plate(OOD)</td>
<td>28.125</td>
<td>18.75</td>
<td>75</td>
</tr>
<tr>
<td>Grind Coffee Beans with Grinder</td>
<td>8</td>
<td>0</td>
<td>92</td>
</tr>
<tr>
<td>Pour Water from Kettle to Flowers</td>
<td>5</td>
<td>5</td>
<td>65</td>
</tr>
<tr>
<td>Touch Instructed Keyboard</td>
<td>0</td>
<td>100</td>
<td>82.5</td>
</tr>
<tr>
<td>Put Bread into Oven</td>
<td>12</td>
<td>40</td>
<td>42</td>
</tr>
<tr>
<td><strong>Average</strong></td>
<td><strong>14.79</strong></td>
<td><strong>25.86</strong></td>
<td><strong>63.22</strong></td>
</tr>
<tr>
<td colspan="4"><strong>Agilex-Aloha-2</strong></td>
</tr>
<tr>
<td>Fold Towel</td>
<td>27.5</td>
<td>0</td>
<td>39</td>
</tr>
<tr>
<td>Get Water from Water Dispenser</td>
<td>62</td>
<td>8</td>
<td>96</td>
</tr>
<tr>
<td>Pour Water from Kettle to Flowers</td>
<td>45</td>
<td>40</td>
<td>47.5</td>
</tr>
<tr>
<td>Touch Instructed Keyboard</td>
<td>72.5</td>
<td>85</td>
<td>80</td>
</tr>
<tr>
<td>Put Bread into Oven</td>
<td>36</td>
<td>0</td>
<td>34</td>
</tr>
<tr>
<td><strong>Average</strong></td>
<td><strong>48.60</strong></td>
<td><strong>26.60</strong></td>
<td><strong>59.30</strong></td>
</tr>
</tbody>
</table>

**分析**:
*   **真实世界性能优越**: 在 `AC-One` 平台上，`Motus` 的平均部分成功率 (63.22%) 远超 $\pi_{0.5}$ (14.79%) 和从零训练的版本 (25.86%)。在 `Agilex-Aloha-2` 平台上，`Motus` (59.30%) 也显著优于 $\pi_{0.5}$ (48.60%)。
*   **处理复杂任务的能力**: 在一些极具挑战性的任务上，如“煮咖啡”(Brew Coffee) 和“磨咖啡豆”(Grind Coffee Beans)，$\pi_{0.5}$ 和 `w/o Pretrain` 的成功率几乎为零，而 `Motus` 却能取得非常高的分数 (62% 和 92%)。这表明 `Motus` 学到的先验知识对于解决长时程、需要精确操作的任务至关重要。
*   **OOD 泛化**: 在 `Place Cube into Plate(OOD)` 任务中（将物体放置在训练中未见过的位置），`Motus` 的性能 (75%) 远超基线 (28.125%)，显示了其强大的泛化能力。

### 6.1.3. 统一模型能力验证

附录中的实验验证了 `Motus` 在五种模式下的能力。
*   **VGM & World Model**: 从 Figure 7, 9, 10, 11 和 Table 6 的结果来看，`Motus` 能够生成高质量、符合物理规律的未来视频，其 FID、FVD 等指标表现良好。
*   **IDM**: Table 7 显示，`Motus` 在作为逆动力学模型时，其动作预测的 MSE (0.014) 甚至低于专门为此任务训练的基线模型 (0.044)，证明了其统一框架内的 IDM 能力非常强大。
*   **VLA & 联合预测**: Table 8 显示，`Motus` 作为 VLA 模型时性能已经很强 (83.90%)，而使用视频-动作联合预测模式时性能更高 (87.02%)。这暗示了在决策时“想象”未来视觉场景对动作预测是有益的。

## 6.2. 消融实验/参数分析
消融实验的结果贯穿于主要的对比表格中，下图（原文 Figure 6）直观地总结了预训练阶段的贡献。

![Figure 6. Ablation in RoboTwin 2.0 Randomized Multi-task Setting. The figure presents the total success rates $( \\% )$ of the original Motus (Stage 2 Pretrain) and its two variants: Without Pretrain and Stage 1 Pretrain.](images/6.jpg)
*该图像是图表，展示了在RoboTwin 2.0随机化和清晰环境下，原始Motus（Stage 2 Pretrain）与其两种变体：无预训练和Stage 1 Pretrain的成功率（%）。成功率数据分别为77.00%、81.86%和87.02%（随机化）；77.56%、82.26%和88.66%（清晰），并且显示了相应的提升幅度+10.02%和+11.10%。*

**分析**:
*   **Stage 1 的贡献**: `Stage 1` 模型（仅微调VGM）相比 `w/o Pretrain` 模型在随机化环境中提升了约 5% (从 77.00% 到 81.86%)。这说明让模型适应机器人场景的视觉动态是有益的。
*   **Stage 2 的巨大贡献**: 完整的 `Motus` 模型（包含 Stage 2 的潜在动作预训练）相比 `Stage 1` 模型又提升了约 5% (从 81.86% 到 87.02%)。这证明了从大规模异构视频中学习潜在动作是提升泛化能力和最终性能的**核心驱动力**。
*   **总结**: 完整的**三阶段训练流程缺一不可**。直接在目标数据上训练（`w/o Pretrain`）效果最差，而逐步引入通用视觉动态知识（Stage 1）和通用运动知识（Stage 2）能显著、稳定地提升模型性能。

# 7. 总结与思考

## 7.1. 结论总结
`Motus` 是一项在具身智能领域具有里程碑意义的工作。它成功地应对了当前领域存在的两大核心挑战：模型能力的碎片化和异构数据的利用难题。

**主要贡献与发现总结如下**:
1.  **实现了前所未有的统一性**: `Motus` 通过创新的 `MoT` 架构和 `UniDiffuser` 风格的调度器，将五种主流的具身智能建模范式（VLA, WM, IDM, VGM, 联合预测）无缝地集成到一个单一的生成模型中。
2.  **有效利用现有生态系统**: 该模型并非从零开始，而是巧妙地融合了强大的预训练 VLM 和 VGM，继承了它们丰富的世界知识和生成能力。
3.  **解决了动作预训练的难题**: 通过引入基于光流的**潜在动作**，`Motus` 成功地搭建了一座连接无标签视频和机器人控制的桥梁，使其能够从海量的互联网视频和人类演示中学习通用的运动先验，实现了真正意义上的大规模动作预训练。
4.  **性能达到新高度**: 无论是在模拟环境还是复杂的真实世界任务中，`Motus` 的性能都大幅超越了现有的最先进方法，证明了其统一建模和大规模预训练策略的巨大优越性。

    `Motus` 的成功启发我们，未来的通用具身智能体不应是单一功能的“专家”的简单堆砌，而应是一个功能完备、知识融合的统一系统。

## 7.2. 局限性与未来工作
论文作者在结论部分指出了未来的研究方向，也暗示了当前工作的一些可拓展之处：
*   **更先进的统一模型架构**: `MoT` 是一种有效的融合方式，但未来可以探索更高效、更深度的融合机制，以更好地协调不同专家之间的合作。
*   **更通用的运动先验**: 当前的潜在动作基于光流，这是一种低级的运动表示。未来可以探索更高级、更抽象的运动原语 (motion primitives)，例如基于物体状态变化或因果关系的表示。
*   **更大规模的预训练**: 论文验证了从多机器人和人类视频中学习的有效性。未来的终极目标是从整个互联网的视频数据中学习潜在动作，这将需要更强的计算资源和更鲁棒的算法来处理噪声和多样性。

## 7.3. 个人启发与批判
这篇论文给我带来了深刻的启发，也引发了一些思考。

**启发**:
1.  <strong>“集大成”</strong>而非“另起炉灶”: 在基础模型时代，从零开始训练一个超大规模模型对于多数研究团队而言是不现实的。`Motus` 展示了一条非常务实且高效的技术路线：通过巧妙的架构设计，将社区已有的、最强大的开源模型作为“积木”，搭建出能力更全面的新系统。这种“站在巨人肩膀上”的思路极具借鉴意义。
2.  **寻找“通用货币”**: 异构性是机器人领域长期存在的难题。`Motus` 使用光流作为不同机器人、不同数据源之间的“通用货币”，成功统一了运动信息的表示。这启发我们在其他异构问题上（如不同的传感器、不同的环境）也可以尝试寻找类似的通用中间表示。
3.  **统一的价值不仅在于性能**: `Motus` 不仅在 VLA 任务上取得了更高的成功率，它还同时具备了世界建模、想象未来、反思动作等多种能力。这种多功能性为实现更高级的智能（如长期规划、在线适应、错误恢复）提供了基础，其长远价值可能远超当前任务性能的提升。

    **潜在问题与批判**:
1.  **对光流质量的依赖**: 整个潜在动作学习流程的起点是光流估计。如果光流估计算法在某些场景下（如快速运动、光照剧变、透明或反光物体）表现不佳，可能会严重影响潜在动作的质量，进而影响下游任务的性能。模型的鲁棒性可能受限于上游的光流算法。
2.  **计算成本高昂**: 三阶段的训练流程，尤其是在大规模异构数据上进行的预训练，需要巨大的计算资源。这可能成为限制该方法被广泛复现和应用的一个门槛。
3.  **潜在动作的可解释性**: 尽管潜在动作在任务中表现出色，但它仍然是一个“黑箱”的低维向量。其每一维是否对应着某种可解释的物理含义尚不清楚。未来如果能增强其可解释性，或许能让机器人行为的分析和调试变得更加容易。
4.  **真实世界与模拟的差距**: 尽管 `Motus` 在真实世界中表现优异，但其预训练数据中仍包含大量模拟数据。如何进一步缩小模拟与现实的差距（Sim-to-Real Gap），或者完全依赖真实世界数据进行学习，仍然是一个开放性问题。

    总而言之，`Motus` 是一项兼具理论深度和工程价值的杰出工作。它为构建下一代通用具身智能体提供了一个清晰、可行的蓝图，无疑将对该领域未来的研究产生深远影响。