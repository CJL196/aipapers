# 1. 论文基本信息

## 1.1. 标题
<strong>上下文即记忆：基于记忆检索的场景一致性交互式长视频生成 (Context as Memory: Scene-Consistent Interactive Long Video Generation with Memory Retrieval)</strong>

该标题直接点明了论文的核心思想：将历史上下文 (`Context`) 视为一种记忆 (`Memory`) 机制，并通过一个`记忆检索 (Memory Retrieval)`模块，来实现场景一致的 (`Scene-Consistent`)、可交互的 (`Interactive`) 长视频 (`Long Video`) 生成。

## 1.2. 作者
*   **Jiwen Yu, Yiran Qin, Xihui Lu:** 来自香港大学 (The University of Hong Kong)。
*   **Jianhong Bai:** 来自浙江大学 (Zhejiang University)。
*   **Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang:** 来自快手科技的“可灵”团队 (Kling Team, Kuaishou Technology)。

    作者团队结合了学术界（港大、浙大）和工业界（快手）的研究力量，特别是快手的可灵团队，是国内领先的视频生成模型研发团队，这表明该研究具有坚实的工程背景和应用前景。

## 1.3. 发表期刊/会议
论文以预印本 (preprint) 形式发布在 `arXiv` 上。`arXiv` 是一个广泛用于物理学、数学、计算机科学等领域学者发布最新研究成果的平台。虽然未经同行评审，但它是了解前沿技术动态的重要渠道。

## 1.4. 发表年份
2025年 (预印本首次提交于2024年6月，根据论文元数据，目标发表时间为2025年)。

## 1.5. 摘要
该论文旨在解决现有交互式视频生成方法在生成长视频时难以保持场景一致性的问题。作者认为，这个问题的根源在于模型对历史上下文的利用有限。为此，他们提出了一个名为 **`Context-as-Memory`** 的框架。该框架的核心设计非常简洁高效：
1.  **存储方式：** 直接将历史视频帧作为记忆存储，无需任何额外的后处理。
2.  **使用方式：** 在生成新帧时，将检索到的历史上下文帧与待预测的帧在“帧维度”上拼接起来，直接作为模型输入，无需引入额外的控制模块。

    考虑到将所有历史帧都作为输入会带来巨大的计算开销，论文进一步提出了一个 <strong>`记忆检索 (Memory Retrieval)`</strong> 模块。该模块通过计算相机位姿之间的<strong>视场 (Field of View, FOV)</strong> 重叠度，来筛选出与当前待生成帧最相关的历史帧，从而在不损失关键信息的前提下，显著减少了计算量。

实验结果表明，`Context-as-Memory` 在交互式长视频生成任务中，其“记忆能力”显著优于当前最先进的 (state-of-the-art) 方法，并且能够有效泛化到训练数据中未见过的开放领域场景。

## 1.6. 原文链接
*   **arXiv 链接:** [https://arxiv.org/abs/2506.03141](https://arxiv.org/abs/2506.03141)
*   **PDF 链接:** [https://arxiv.org/pdf/2506.03141v2](https://arxiv.org/pdf/2506.03141v2)
*   **发布状态:** 预印本 (Preprint)。

    ---

# 2. 整体概括

## 2.1. 研究背景与动机
### 2.1.1. 核心问题
当前视频生成模型（如 Sora、Kling）虽然取得了巨大进展，但在**交互式长视频生成**任务中普遍存在一个严重缺陷：**缺乏长期记忆能力**。具体表现为，当用户通过控制相机视角（例如，向左转90度再转回来）探索一个场景时，模型无法“记住”之前看过的景象，导致返回原位时看到的场景与之前完全不同，破坏了场景的一致性和沉浸感。

### 2.1.2. 问题重要性与现有挑战
*   **重要性:** 许多前沿应用，如游戏（可生成动态世界）、模拟器（用于自动驾驶或机器人训练）等，都极度依赖于能够生成连贯、一致且可交互的长视频。场景不一致会使得这些应用无法落地。
*   <strong>现有挑战 (Gap):</strong>
    1.  **有限的上下文窗口:** 现有方法（如 `Diffusion Forcing`）在生成新帧时，通常只参考最近的几十帧作为上下文。这种“滑动窗口”式的机制只能保证短期连续性，一旦场景元素移出窗口，相关信息就会丢失。
    2.  **计算资源限制:** 一个看似简单的解决方案是“让模型看到所有历史帧”。但这是不切实际的，因为随着视频变长，将所有历史帧都纳入计算会导致显存和计算量爆炸式增长。
    3.  **信息无关性与噪声:** 即便计算资源无限，将所有历史帧都输入模型也未必是好事。大部分历史帧与当前待生成的帧在内容上是无关的，强行输入会引入噪声，干扰生成过程。

### 2.1.3. 本文的切入点
作者提出了一个非常直观且优雅的切入点：**将所有历史上下文帧视为一个外部“记忆库”，在生成新内容时，只从这个库中“检索”出最相关的几帧来辅助生成**。这个想法将问题从“如何处理无限增长的上下文”转变为“如何高效地从历史中检索有用信息”。

## 2.2. 核心贡献/主要发现
本文的主要贡献可以总结为以下几点：

1.  **提出了 `Context-as-Memory` 框架:**
    *   **核心思想:** 明确提出将历史上下文直接作为视频生成的记忆库，为解决场景一致性问题提供了新的范式。
    *   **简洁实现:** 采用了两种简单但有效的设计：
        *   **直接存储:** 视频帧本身就是记忆，无需编码成特征向量或重建三维场景，避免了信息损失和额外计算。
        *   **直接拼接:** 通过在帧维度上拼接上下文与待预测内容，将其作为统一输入送入模型，无需复杂的适配器 (`Adapter`) 或交叉注意力 (`cross-attention`) 模块，易于实现。

2.  **设计了 `Memory Retrieval` 模块:**
    *   **解决了计算瓶颈:** 针对“所有历史帧作为输入”带来的计算难题，提出了一个基于<strong>相机轨迹视场 (FOV) 重叠</strong>的规则化检索方法。
    *   **高效筛选:** 该方法能够高效地从海量历史帧中筛选出与当前视角有内容重叠的帧，极大地减少了送入模型的上下文数量，同时保留了最关键的场景信息。

3.  **构建了新的专用数据集:**
    *   为了训练和验证模型，作者使用<strong>虚幻引擎5 (Unreal Engine 5)</strong> 创建了一个新的长视频数据集。
    *   **特点:** 该数据集包含多样化的场景、长时程的视频、以及**精确的相机位姿标注**。这对于训练一个依赖相机信息进行记忆检索的模型至关重要。

4.  **优异的实验结果:**
    *   实验证明，该方法在保持场景一致性方面**显著超越**了现有的 SOTA 方法。
    *   模型展现了良好的**泛化能力**，即使在训练时未见过的开放领域场景中，也能保持有效的记忆。

        ---

# 3. 预备知识与相关工作

## 3.1. 基础概念
### 3.1.1. 扩散模型 (Diffusion Models)
扩散模型是一类强大的生成模型，近年来在图像和视频生成领域取得了巨大成功。其核心思想分为两个过程：
*   <strong>前向过程（加噪）:</strong> 从一张干净的图像（或视频帧）开始，逐步地、多次地向其添加少量高斯噪声，直到图像完全变成纯粹的随机噪声。这个过程是固定的、不可学习的。
*   <strong>反向过程（去噪）:</strong> 训练一个神经网络（通常是 U-Net 或 Transformer 结构），让它学会如何从一张充满噪声的图像中，一步步地预测并去除噪声，最终还原出干净的图像。生成新图像的过程，就是从一个随机噪声开始，利用这个训练好的去噪网络，逐步将其“去噪”成一张清晰的图像。

    本文使用的基础模型就是一个基于 Transformer 架构的<strong>潜在扩散模型 (Latent Diffusion Model)</strong>。这意味着加噪和去噪过程不是在像素空间直接进行的，而是在一个由 **VAE (Variational Autoencoder)** 压缩得到的、更低维度的<strong>潜在空间 (latent space)</strong> 中进行，这样可以大大降低计算成本。

### 3.1.2. 扩散变换器 (Diffusion Transformer, DiT)
`DiT` 是指将 Transformer 架构用作扩散模型中去噪网络的一种模型。与传统的 U-Net 结构相比，Transformer 在处理长距离依赖关系方面更具优势，并且被证明具有更好的可扩展性。本文的基座模型就是一个 `DiT`。

### 3.1.3. 视场 (Field of View, FOV)
`FOV` 指的是摄像机在特定时刻能够“看到”的范围，通常以角度来衡量。在三维空间中，它形成一个锥体或扇形区域。两个不同位置和朝向的相机，如果它们的 `FOV` 区域有重叠，就意味着它们可以看到部分相同的场景内容。这是本文 `Memory Retrieval` 模块的核心物理基础。

## 3.2. 前人工作
### 3.2.1. 流式视频生成 (Streaming Video Generation)
这是指模型以自回归 (auto-regressive) 的方式，一小段一小段地连续生成视频。其数学形式可以表示为：
$$
p(x^0, x^1, ..., x^n) = \prod_{i=0}^{n} p(x^i | x^0, x^1, ..., x^{i-1})
$$
其中 $x^i$ 表示第 $i$ 帧（或视频片段）。现有方法主要分为两类：
*   **基于扩散模型的方法:** 如 `Diffusion Forcing` (DFoT)，它在一个固定的上下文窗口内进行去噪预测，实现视频续写。这类方法生成质量高，但如前文所述，记忆能力受限于窗口大小。
*   **类GPT的下一词元预测方法:** 这类方法将视频帧编码为离散的词元 (token)，然后像训练语言模型一样预测下一个词元。这类方法理论上可以处理更长的上下文，但通常生成质量和效率不如扩散模型。

### 3.2.2. 基于3D重建的记忆方法
一些工作尝试通过从已生成的视频中重建出**显式的三维场景表示**（如点云、网格或神经辐射场 NeRF），来作为记忆。当需要生成新视角时，先从这个三维模型中渲染出图像作为参考。
*   **缺点:** 这种方法的瓶颈在于3D重建的**速度和精度**。对于不断扩大的大规模场景，实时重建非常困难，且累积误差会越来越大，最终导致场景失真。

### 3.2.3. 上下文学习 (Context Learning)
近期一些工作开始探索长上下文在视频生成中的作用。
*   `FAR`: 提出了长短期上下文窗口来指导生成。
*   `FramePack`: 提出一种层级压缩方法，将历史帧压缩成固定数量的几帧作为条件。但其指数衰减的压缩方式会严重丢失早期历史信息。

## 3.3. 技术演进
视频生成技术从早期的 GAN、VAE，发展到如今以扩散模型为主流的时代。生成时长也从几秒的短片，向着分钟级的长视频发展。随着时长的增加，<strong>“一致性”</strong>成为了新的核心挑战。技术演进的脉络可以看作是：
1.  **提升单次生成质量和时长：** 更大的模型、更多的数据。
2.  **引入流式生成：** 实现无限时长的可能性，但牺牲了全局一致性。
3.  **探索长上下文/记忆机制：** <strong>（本文所处阶段）</strong> 试图在流式生成的基础上，通过引入记忆机制来恢复全局一致性。

## 3.4. 差异化分析
本文方法与相关工作的主要区别和创新点在于：

| 特征 | 本文 (Context-as-Memory) | DFoT / 滑动窗口 | 3D重建方法 | FramePack |
| :--- | :--- | :--- | :--- | :--- |
| **记忆形式** | **原始视频帧** | 最近的视频帧 | 显式三维模型 | 压缩后的视频帧 |
| **记忆容量** | 理论上**无限**（存储所有历史） | **固定**窗口大小 | 受限于重建模型能力 | 压缩后**固定**大小 |
| **信息损失** | **无**（直接存储） | 丢失窗口外信息 | 重建过程有损失 | 压缩过程有损失 |
| **检索机制** | **基于FOV的动态检索** | 无（固定窗口） | 渲染新视角 | 无（固定压缩） |
| **计算开销** | **可控**（只计算检索出的少量帧） | 低，但记忆有限 | 高（实时重建困难） | 低，但信息损失大 |
| **实现复杂度** | **低**（直接拼接输入） | 低 | 高 | 中等 |

**核心创新：** 本文没有设计复杂的记忆编码或模型结构，而是回归到一个简单本质的思路——<strong>“好记性不如烂笔头”</strong>。它把所有历史帧都“记下来”（存储），在需要的时候通过一个高效的索引（FOV检索）去“翻阅”查找最相关的内容。这种“外部记忆+高效检索”的范式，是其与之前方法最本质的区别。

---

# 4. 方法论

## 4.1. 方法原理
`Context-as-Memory` 的核心思想是将历史上下文作为可供检索的记忆库，以解决长视频生成中的场景一致性问题。其实现包含三个关键部分：
1.  一个支持**相机控制**的视频生成基座模型。
2.  一种将**上下文帧**作为条件注入到模型中的机制。
3.  一个高效的**记忆检索**模块，用于从所有历史帧中筛选出相关的上下文。

## 4.2. 核心方法详解 (逐层深入)
### 4.2.1. 基座模型与相机控制
论文的基座是一个标准的<strong>潜在视频扩散模型 (latent video diffusion model)</strong>，其核心组件是 `DiT`。

*   **步骤 1: VAE 编码**
    一个视频帧序列 $\mathbf{x} = \{x^0, x^1, ..., x^{n\bar{r}}\}$ 首先通过一个 3D VAE 的编码器 `Encoder` 被压缩到一个更低维的潜在空间，得到潜在表示 $\mathbf{z} = \mathrm{Encoder}(\mathbf{x})$。其中 $\mathbf{z} = \{z^0, z^1, ..., z^n\}$。

*   **步骤 2: 扩散过程与训练**
    在训练时，对干净的潜在表示 $\mathbf{z}_0$ 添加高斯噪声 $\epsilon$ 得到带噪的 $\mathbf{z}_t$。然后训练一个去噪网络 $\epsilon_{\phi}(\cdot)$ 来预测所添加的噪声。其损失函数为：
    $$
    \mathcal{L}(\phi) = \mathbb{E}[||\epsilon_{\phi}(\mathbf{z}_t, \mathbf{p}, t) - \epsilon||]
    $$
    *   $\phi$: 去噪网络的参数。
    *   $\mathbf{z}_t$: 在时间步 $t$ 的带噪潜在表示。
    *   $\mathbf{p}$: 输入的文本提示 (text prompt)。
    *   $t$: 扩散过程的时间步。
    *   $\epsilon$: 添加的真实高斯噪声。

*   **步骤 3: 相机控制注入**
    为了实现交互性，模型需要能根据用户提供的相机轨迹 $\mathbf{cam}$ 来生成视频。相机位姿信息通过一个相机编码器 $\mathcal{E}_c(\cdot)$（一个简单的 MLP 网络）进行编码，然后以类似 `AdaLN` 的方式注入到 `DiT` 的每个模块中。具体来说，它被加到空间注意力模块的输出上：
    $$
    \mathbf{F}_i = \mathbf{F}_o + \mathcal{E}_c(\mathbf{cam})
    $$
    *   $\mathbf{F}_o$: 空间注意力模块的输出特征。
    *   $\mathcal{E}_c(\mathbf{cam})$: 编码后的相机位姿特征。
    *   $\mathbf{F}_i$: 注入相机信息后，送入 3D 注意力模块的输入特征。
        训练带有相机控制的模型时，损失函数更新为：
    $$
    \mathcal{L}_{\mathbf{cam}}(\phi, \phi_{MLP}) = \mathbb{E}[||\epsilon_{\phi, \phi_{MLP}}(\mathbf{z}_t, \mathbf{p}, \mathbf{cam}, t) - \epsilon||]
    $$

### 4.2.2. 上下文帧学习机制
这是本文的第一个核心设计：**如何将检索到的历史上下文帧作为条件送入模型**。

*   **设计思想:** 简单、直接、有效。不引入任何额外的模块（如 `Adapter` 或 `cross-attention`）。
*   **具体实现:** 将干净的上下文潜在表示 $\mathbf{z}^c$ 和带噪的待预测潜在表示 $\mathbf{z}_t$ 在<strong>帧维度 (frame dimension)</strong> 上进行<strong>拼接 (concatenation)</strong>。
    
    如下图（原文 Figure 2）所示，拼接后的序列 `[context, output]` 作为一个整体送入 `DiT` 模块进行计算。在 `DiT` 内部的自注意力机制中，上下文帧的 `token` 和待预测帧的 `token` 可以相互关注，从而让模型“看到”历史信息。

    ![Fig. 2. Model Architecture. We concatenate the context to be conditioned and the predicted frames along the frame dimension. This method of injecting context is simple and effective, requiring no additional modules.](images/2.jpg)
    *该图像是示意图，展示了基于历史上下文进行视频生成的模型架构。通过将历史上下文与当前输出进行拼接，模型利用多种注意力机制（包括3D 和 2D 注意力）对帧进行处理，以提高生成视频的内容一致性。*

*   **关键细节:**
    1.  **输出处理:** 在 `DiT` 的一次前向传播后，只用其输出的噪声预测来更新带噪的 $\mathbf{z}_t$ 部分，而上下文部分 $\mathbf{z}^c$ 保持不变（因为它已经是干净的，无需去噪）。
    2.  **位置编码:** 为了适应可变长度的上下文，并保持预训练模型的能力，作者对位置编码进行了特殊处理。待预测帧 $\mathbf{z}_t$ 沿用预训练时的位置编码，而新加入的上下文帧 $\mathbf{z}^c$ 则被赋予新的位置编码。由于基座模型使用了 `RoPE (Rotary Position Embedding)`，它可以很自然地扩展到变长的序列。

### 4.2.3. 记忆检索 (Memory Retrieval)
这是本文的第二个核心设计：**如何从所有历史帧中高效地筛选出有用的上下文**。

*   **目标:** 从海量历史帧中，找出那些与**待生成帧**的**可见区域有重叠**的帧。

*   **本文方法：基于相机轨迹的搜索**
    该方法利用了模型是相机可控的这一特性，因为所有历史帧的相机位姿都是已知的。检索过程分为两步：**筛选**和**再筛选**。

    1.  **步骤一：基于 FOV 共视性的粗筛选**
        *   **如何判断共视性？** 作者提出一个简化的几何方法来判断两个相机位姿的 `FOV` 是否重叠。由于相机运动被限制在 XY 平面上，问题简化为二维。如下图（原文 Figure 4）所示，每个相机的 `FOV` 可以由其原点发出的左右两条射线表示。通过计算两台相机（一台是历史帧的，一台是待生成帧的）的四条射线之间的交点，可以快速判断 `FOV` 是否重叠。
        *   **过滤规则:**
            *   基本规则：历史帧的左射线与待生成帧的右射线相交，**并且** 历史帧的右射线与待生成帧的左射线相交（如图 a, b）。
            *   距离过滤：为了排除那些 `FOV` 虽然在远处相交但实际场景重叠很小的情况（如图 c），或者相机离得太近导致相交点在相机后方的情况（如图 d），会计算交点到相机的距离，并过滤掉过远或过近的情况。
        
                ![Fig. 4. Examples of FOV Overlap. We simplify FOV overlap detection to checking intersections between four rays from two camera origins. A practical rule that works for most cases requires: both left and right ray pairs intersect (a, b). However, we must filter out cases where intersection points are either too near (d) or too distant (c) from cameras. While this rule may not cover all scenarios and some corner cases exist (e, f), occasional missed or incorrect candidates don't substantially affect overall performance.](images/4.jpg)
                *该图像是示意图，展示了不同场景下的视域重叠示例。在图中，(a)、(b)和(c)展示了有效的视域重叠，而(d)则显示了视域交点过近或过远的情况。图(e)和(f)展示了一些边界案例，提醒在实际应用中可能会遗漏或错误识别候选帧。*

        通过这个方法，可以快速过滤掉大量与当前视角完全不相关的历史帧。

    2.  **步骤二：对候选帧的精细筛选**
        经过 FOV 筛选后，可能仍然有超过模型上下文限制（例如20帧）的候选帧。此时需要进一步筛选：
        *   <strong>策略1 (核心): `Non-adj` (非相邻选择)</strong>
            考虑到视频中连续的几帧内容高度冗余，从一组连续的候选帧中只**随机选择一帧**。这极大地减少了冗余信息，保留了多样性。
        *   <strong>策略2 (可选): `Far-space-time` (时空最远选择)</strong>
            在策略1的基础上，额外选择几帧在时间上或空间上与当前帧距离最远的上下文。这有助于补充可能被遗漏的长期记忆。

下图（原文 Figure 3(a)）直观地展示了整个 `Memory Retrieval` 的流程。

![该图像是示意图，展示了利用历史上下文进行视频生成的框架。左侧展示了上下文学习过程，其中包含无限长度历史上下文以及最新帧的输出。右侧说明了内存检索模块的工作原理，选择高重叠上下文以指导预测帧的生成。](images/3.jpg)
*该图像是示意图，展示了利用历史上下文进行视频生成的框架。左侧展示了上下文学习过程，其中包含无限长度历史上下文以及最新帧的输出。右侧说明了内存检索模块的工作原理，选择高重叠上下文以指导预测帧的生成。*

### 4.2.4. 训练与推理流程
论文给出了清晰的算法伪代码。

*   <strong>训练过程 (Algorithm 1):</strong>
    1.  从长视频数据中随机采样一小段作为**待预测序列** $x_0$。
    2.  从该视频的其余部分，使用 `Memory Retrieval` 方法选择 `k-1` 帧作为**历史上下文** $x_c$。同时，将 $x_0$ 的第一帧也加入上下文，以保证视频的连续性。
    3.  将 $x_0$ 和 $x_c$ 编码为潜在表示 $z_0$ 和 $z_c$。
    4.  对 $z_0$ 进行加噪得到 $z_t$。
    5.  将 $z_c$ 和 $z_t$ 拼接后送入 `DiT`，训练模型预测噪声，计算扩散损失。
    *   **特殊处理:** 训练中有 10% 的概率不使用检索到的历史上下文，只用最近的一帧。这是为了模拟视频刚开始生成时没有历史记忆库的情况。

        <table>
        <tr><td colspan="2"><b>ALGORITHM 1:</b> Training Process of Context-as-Memory</td></tr>
        <tr><td>1</td><td><b>Input:</b> Video sequence X and camera annotations C in training dataset, context size k</td></tr>
        <tr><td>2</td><td><b>while</b> not converged <b>do</b><br/>  Randomly select predicted video sequence x<sub>0</sub> from X;<br/></td></tr>
        <tr><td>3</td><td>  Retrieve k frames as context x<sub>c</sub>;</td></tr>
        <tr><td>4</td><td>  Obtain camera poses {cam<sub>0</sub>, cam<sub>c</sub>} for {x<sub>0</sub>, x<sub>c</sub>} from C;</td></tr>
        <tr><td>5</td><td>  Obtain latent embeddings {z<sub>0</sub>, z<sub>c</sub>} ← Encoder({x<sub>0</sub>, x<sub>c</sub>});</td></tr>
        <tr><td>6</td><td>  Sample t ~ U(1, T) and ε ~ N(0, I), then corrupt z<sub>0</sub> to z<sub>t</sub>;</td></tr>
        <tr><td>7</td><td>  Train ε<sub>φ</sub>(z<sub>t-1</sub> | z<sub>t</sub>, z<sub>c</sub>, cam<sub>0</sub>, cam<sub>c</sub>, t) using diffusion loss;<br/><b>end while</b></td></tr>
        </table>

*   <strong>推理过程 (Algorithm 2):</strong>
    1.  从一个初始帧（或视频片段）开始，维护一个已生成的视频序列 $X$ 和对应的相机位姿 $C$ 作为记忆库。
    2.  用户提供下一个目标相机位姿 `cam_next`。
    3.  使用 `Memory Retrieval` 方法，基于 `cam_next` 从记忆库 $X$ 中检索出 `k-1` 帧上下文 $x_c$。将最新生成的一帧也加入上下文。
    4.  将上下文 $x_c$ 编码为 $z_c$。
    5.  从随机噪声开始，以 $z_c$ 和 `cam_next` 为条件，通过反向扩散过程生成新的视频片段的潜在表示 $z_{new}$。
    6.  将 $z_{new}$ 解码为视频帧 $x_{new}$。
    7.  将 $x_{new}$ 和 `cam_next` 追加到记忆库 $X$ 和 $C$ 中。
    8.  重复步骤 2-7，实现交互式长视频生成。

        <table>
        <tr><td colspan="2"><b>ALGORITHM 2:</b> Inference Process of Context-as-Memory</td></tr>
        <tr><td>1</td><td><b>Input:</b> Initial frame set X = {x<sub>init</sub>} and camera poses C = {cam<sub>init</sub>}<br/><b>Output:</b> Generated video sequence X</td></tr>
        <tr><td>2</td><td><b>while</b> generation not finished <b>do</b></td></tr>
        <tr><td>3</td><td>  User provides next target camera pose cam<sub>next</sub>;</td></tr>
        <tr><td>4</td><td>  Retrieve context frames x<sub>c</sub> ⊂ X and cam<sub>c</sub> ⊂ C by checking FOV overlap with cam<sub>next</sub>;</td></tr>
        <tr><td>5</td><td>  Compute context latent z<sub>c</sub> ← Encoder(x<sub>c</sub>);</td></tr>
        <tr><td>6</td><td>  Sample noise e ~ N(0, I) and infer latent z<sub>new</sub> ∼ p(z | z<sub>c</sub>, p, cam<sub>next</sub>, cam<sub>c</sub>);</td></tr>
        <tr><td>7</td><td>  Decode generated frames x<sub>new</sub> ← Decoder(z<sub>new</sub>);</td></tr>
        <tr><td>8</td><td>  Append x<sub>new</sub> to X and cam<sub>next</sub> to C;<br/><b>end while</b></td></tr>
        </table>

---

# 5. 实验设置

## 5.1. 数据集
由于现有带相机位姿标注的数据集多为短视频，无法满足长视频场景一致性训练的需求，作者自行构建了一个数据集。
*   **来源:** 使用 <strong>虚幻引擎5 (Unreal Engine 5)</strong> 渲染生成。
*   **规模:** 100个长视频，每个视频包含 7,601 帧。
*   **内容:** 涵盖12种不同的场景风格（如城市街道、商场、乡村等），以保证多样性。
*   **相机轨迹:** 通过随机采样路径点并生成平滑的B样条曲线来创建相机轨迹。为了简化问题，相机运动被限制在 **2D平面** 上，旋转也仅限于 Z 轴（偏航角）。
*   **标注:** 每个视频都带有精确的相机内外参。此外，每隔77帧使用一个多模态大语言模型进行**文本描述标注**。

## 5.2. 评估指标
论文使用了两类指标：一类评估视频生成质量，另一类专门用于量化“记忆能力”。

### 5.2.1. 视频质量指标
#### FID (Fréchet Inception Distance)
1.  <strong>概念定义 (Conceptual Definition):</strong> `FID` 是一个广泛用于评估生成模型（尤其是图像生成）质量的指标。它通过比较生成样本和真实样本在某个预训练网络（通常是 InceptionV3）的特征空间中的统计分布来衡量二者之间的相似度。`FID` 分数越低，表示生成样本的分布与真实样本的分布越接近，即生成质量越高、多样性越好。
2.  <strong>数学公式 (Mathematical Formula):</strong>
    $$
    \text{FID}(x, g) = ||\mu_x - \mu_g||_2^2 + \text{Tr}(\Sigma_x + \Sigma_g - 2(\Sigma_x \Sigma_g)^{1/2})
    $$
3.  <strong>符号解释 (Symbol Explanation):</strong>
    *   $x$ 和 $g$ 分别代表真实图像和生成图像的集合。
    *   $\mu_x$ 和 $\mu_g$ 是真实图像和生成图像在 InceptionV3 网络某一激活层输出特征的均值向量。
    *   $\Sigma_x$ 和 $\Sigma_g$ 是这些特征的协方差矩阵。
    *   $||\cdot||_2^2$ 表示向量的L2范数的平方。
    *   $\text{Tr}(\cdot)$ 表示矩阵的迹（主对角线元素之和）。

#### FVD (Fréchet Video Distance)
1.  <strong>概念定义 (Conceptual Definition):</strong> `FVD` 是 `FID` 在视频领域的扩展。它专门用于评估生成视频的质量，不仅考虑了单帧画面的逼真度，还考虑了视频的**时间连贯性**。与 `FID` 类似，`FVD` 分数也是越低越好。
2.  <strong>数学公式 (Mathematical Formula):</strong> 计算方式与 `FID` 完全相同，区别在于所用的特征提取器。`FVD` 使用一个在大量视频数据上预训练的 3D 卷积网络 (I3D) 来提取时空特征。
    $$
    \text{FVD}(x, g) = ||\mu_x - \mu_g||_2^2 + \text{Tr}(\Sigma_x + \Sigma_g - 2(\Sigma_x \Sigma_g)^{1/2})
    $$
3.  <strong>符号解释 (Symbol Explanation):</strong>
    *   符号含义与 `FID` 相同，但这里的 $\mu$ 和 $\Sigma$ 是从 I3D 网络提取的视频特征计算得到的。

### 5.2.2. 记忆能力指标
为了量化记忆能力，作者设计了两种比较方式，并使用像素级的差异指标进行评估。

#### PSNR (Peak Signal-to-Noise Ratio)
1.  <strong>概念定义 (Conceptual Definition):</strong> 峰值信噪比是衡量图像质量的常用指标，它通过计算两张图像对应像素之间的均方误差 (MSE) 来衡量它们的差异。`PSNR` 值越高，表示两张图像越相似，失真越小。
2.  <strong>数学公式 (Mathematical Formula):</strong>
    $$
    \text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right)
    $$
    其中，均方误差 MSE 的计算公式为：
    $$
    \text{MSE} = \frac{1}{m \times n} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [I(i,j) - K(i,j)]^2
    $$
3.  <strong>符号解释 (Symbol Explanation):</strong>
    *   $\text{MAX}_I$: 图像像素值的最大可能值（例如，对于8位灰度图像，它是255）。
    *   $I$ 和 $K$ 分别代表原始图像和对比图像。
    *   `m, n`: 图像的高度和宽度。
    *   `I(i,j), K(i,j)`: 图像在坐标 `(i,j)` 处的像素值。

#### LPIPS (Learned Perceptual Image Patch Similarity)
1.  <strong>概念定义 (Conceptual Definition):</strong> `LPIPS` 是一种更符合人类感知习惯的图像相似度度量。它不像 `PSNR` 那样只计算像素值的绝对差异，而是比较两张图像在深度神经网络（如 AlexNet, VGG）不同层级提取出的特征图之间的距离。`LPIPS` 分数越低，表示两张图像在感知上越相似。
2.  <strong>数学公式 (Mathematical Formula):</strong>
    $$
    d(x, x_0) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} || w_l \odot (\hat{y}_{hw}^l - \hat{y}_{0hw}^l) ||_2^2
    $$
3.  <strong>符号解释 (Symbol Explanation):</strong>
    *   $d(x, x_0)$: 图像 $x$ 和 $x_0$ 之间的 `LPIPS` 距离。
    *   $l$: 网络的第 $l$ 层。
    *   $\hat{y}^l, \hat{y}_0^l$: 从第 $l$ 层提取的特征图，经过归一化处理。
    *   $H_l, W_l$: 第 $l$ 层特征图的高度和宽度。
    *   $w_l$: 一个可学习的权重，用于缩放不同通道的激活值，以更好地匹配人类感知。

## 5.3. 对比基线
为了验证方法的有效性，论文与以下几种代表性方法进行了比较：
*   **1st Frame as Context:** 只使用序列的第一帧作为上下文，这是一个最简单的基线。
*   **1st Frame + Random Context:** 使用第一帧和一些随机选择的历史帧作为上下文。
*   **DFoT (Diffusion Forcing Transformer):** 代表了使用**最近帧滑动窗口**作为上下文的 SOTA 方法。
*   **FramePack:** 代表了通过**层级压缩**来利用所有历史帧的 SOTA 方法。

    所有这些基线方法都在作者的基座模型和新数据集上进行了公平的重新训练，以保证比较的公正性。

---

# 6. 实验结果与分析

## 6.1. 核心结果分析
实验核心结果展示在原文的 Table 1 和 Figure 5 中。评估分为两种设置：
1.  <strong>Ground Truth Comparison (真值比较):</strong> 从真实视频中选取上下文，预测未来的帧，并与真实视频帧进行比较。这主要评估模型利用真实、干净的上下文进行预测的能力。
2.  <strong>History Context Comparison (历史上下文比较):</strong> 在长视频生成过程中，使用**模型自己先前生成的帧**作为上下文来生成新帧。这种自回归的评估方式更具挑战性，也更能真实地反映模型在实际应用中的记忆能力和误差累积情况。

    以下是原文 Table 1 的结果：

    <table>
    <thead>
    <tr>
    <th rowspan="2">Methods</th>
    <th colspan="3">Ground Truth Comparison</th>
    <th colspan="3">History Context Comparison</th>
    </tr>
    <tr>
    <th>PSNR↑ LPIPS↓</th>
    <th>FID↓</th>
    <th>FVD↓</th>
    <th>PSNR↑ LPIPS↓</th>
    <th>FID↓</th>
    <th>FVD↓</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>1st Frame as Context</td>
    <td>15.72 / 0.5282</td>
    <td>127.55</td>
    <td>937.51</td>
    <td>14.53 / 0.5456</td>
    <td>157.44</td>
    <td>1029.71</td>
    </tr>
    <tr>
    <td>1st Frame + Random Context</td>
    <td>17.70 / 0.4847</td>
    <td>115.94</td>
    <td>853.13</td>
    <td>17.07 / 0.3985</td>
    <td>119.31</td>
    <td>882.36</td>
    </tr>
    <tr>
    <td>DFoT [Song et al. 2025]</td>
    <td>17.63 / 0.4528</td>
    <td>112.96</td>
    <td>897.87</td>
    <td>15.70 / 0.5102</td>
    <td>121.18</td>
    <td>919.75</td>
    </tr>
    <tr>
    <td>FramePack [Zhang and Agrawala 2025]</td>
    <td>17.20 / 0.4757</td>
    <td>121.87</td>
    <td>901.58</td>
    <td>15.65 / 0.4947</td>
    <td>131.59</td>
    <td>974.52</td>
    </tr>
    <tr>
    <td><b>Context-as-Memory (Ours)</b></td>
    <td><b>20.22 / 0.3003</b></td>
    <td><b>107.18</b></td>
    <td><b>821.37</b></td>
    <td><b>18.11 / 0.3414</b></td>
    <td><b>113.22</b></td>
    <td><b>859.42</b></td>
    </tr>
    </tbody>
    </table>

**结果解读:**
*   <strong>记忆能力 (PSNR/LPIPS):</strong>
    *   在两项比较中，`Context-as-Memory` 的 `PSNR` **最高**，`LPIPS` **最低**，这表明其生成的帧与目标帧（无论是真值还是历史帧）在像素和感知层面都最为相似。这**强有力地证明了其卓越的记忆能力**。
    *   `DFoT` 和 `FramePack` 在更具挑战性的 `History Context Comparison` 中表现不佳，因为它们都无法有效利用**时间上遥远但内容上相关**的历史信息。一旦相机转回来，相关的历史帧早已超出了它们的上下文范围。
    *   有趣的是，`Random Context` 的表现甚至略好于 `DFoT` 和 `FramePack`，这说明即使是随机采样，只要有机会从整个历史中获取信息，也比局限于最近的帧或严重压缩的帧要好。

*   <strong>生成质量 (FID/FVD):</strong>
    *   `Context-as-Memory` 在 `FID` 和 `FVD` 指标上也取得了最佳或接近最佳的成绩。这说明提供充足且相关的上下文信息，不仅能提升记忆力，还能**提高整体的生成质量**。
    *   **原因分析:** 1) 强相关的上下文为生成过程提供了更强的条件约束，减少了生成的不确定性。2) 通过参考早期生成的、误差累积较少的帧，可以有效**抑制错误在长视频生成过程中的传播和累积**。

        下图（原文 Figure 5）提供了定性对比，直观展示了本文方法的优势。在“向前旋转再向后旋转”的测试中，只有 `Context-as-Memory` (C-a-M) 能够恢复出与之前几乎一致的场景，而其他方法生成的场景都发生了明显变化。

        ![该图像是图表，展示了与地面真实（GT）比较的结果，标注了不同方法（如C-a-M、Random、DFoT和FramePack）在场景一致性上的表现，包括前向和后向旋转的上下文比较。图中红框标示了不一致的区域，表明当前方法在历史上下文利用上表现优越。](images/5.jpg)
        *该图像是图表，展示了与地面真实（GT）比较的结果，标注了不同方法（如C-a-M、Random、DFoT和FramePack）在场景一致性上的表现，包括前向和后向旋转的上下文比较。图中红框标示了不一致的区域，表明当前方法在历史上下文利用上表现优越。*

## 6.2. 消融实验/参数分析
作者通过消融实验验证了模型设计的有效性。

### 6.2.1. 上下文大小的影响 (Ablation of Context Size)
以下是原文 Table 2 的结果，探究了用作条件的上下文帧数量（Context Size）对性能和速度的影响。

<table>
<thead>
<tr>
<th rowspan="2">Context Size</th>
<th colspan="2">GT Comp.</th>
<th colspan="2">HC Comp.</th>
<th rowspan="2">Speed (fps)↑</th>
</tr>
<tr>
<th>PSNR↑</th>
<th>LPIPS↓</th>
<th>PSNR↑</th>
<th>LPIPS↓</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>15.72</td>
<td>0.5282</td>
<td>14.53</td>
<td>0.5456</td>
<td>1.60</td>
</tr>
<tr>
<td>5</td>
<td>17.37</td>
<td>0.4825</td>
<td>15.97</td>
<td>0.5063</td>
<td>1.40</td>
</tr>
<tr>
<td>10</td>
<td>19.14</td>
<td>0.3554</td>
<td>17.75</td>
<td>0.3985</td>
<td>1.20</td>
</tr>
<tr>
<td>20</td>
<td>20.22</td>
<td>0.3003</td>
<td>18.11</td>
<td>0.3414</td>
<td>0.97</td>
</tr>
<tr>
<td>30</td>
<td>20.31</td>
<td>0.3137</td>
<td>18.19</td>
<td>0.3319</td>
<td>0.79</td>
</tr>
</tbody>
</table>

**分析:**
*   随着上下文大小从1增加到20，记忆能力指标 (`PSNR`/`LPIPS`) 持续显著提升。
*   当大小从20增加到30时，性能提升变得微乎其微，但生成速度 (`fps`) 却大幅下降。
*   这表明**上下文大小为20**是一个很好的<strong>性能与效率的权衡点 (trade-off)</strong>。

### 6.2.2. 记忆检索策略的影响 (Ablation of Memory Retrieval Strategy)
以下是原文 Table 3 的结果，比较了不同检索策略的效果。

<table>
<thead>
<tr>
<th rowspan="2">Strategy</th>
<th colspan="2">GT Comp.</th>
<th colspan="2">HC Comp.</th>
</tr>
<tr>
<th>PSNR↑</th>
<th>LPIPS↓</th>
<th>PSNR↑</th>
<th>LPIPS↓</th>
</tr>
</thead>
<tbody>
<tr>
<td>Random</td>
<td>17.70</td>
<td>0.4847</td>
<td>17.07</td>
<td>0.3985</td>
</tr>
<tr>
<td>FOV+Random</td>
<td>19.17</td>
<td>0.3825</td>
<td>17.47</td>
<td>0.3896</td>
</tr>
<tr>
<td>FOV+Non-adj</td>
<td>20.11</td>
<td>0.3075</td>
<td>18.19</td>
<td>0.3571</td>
</tr>
<tr>
<td><b>FOV+Non-adj+Far-space-time</b></td>
<td><b>20.22</b></td>
<td><b>0.3003</b></td>
<td><b>18.11</b></td>
<td><b>0.3414</b></td>
</tr>
</tbody>
</table>

**分析:**
*   **`FOV` 筛选的有效性:** 从 `Random` 到 $FOV+Random$，性能有显著提升。这证明了基于 `FOV` 重叠来筛选**相关帧**是至关重要的。
*   **`Non-adj` 筛选的有效性:** 从 $FOV+Random$ 到 `FOV+Non-adj`，性能再次大幅提升。这证明了从连续的候选帧中只选一帧来**去除冗余信息**是极其有效的。
*   `Far-space-time` 策略也带来了一些微小的提升，说明补充长时程信息有一定帮助。
*   **结论:** `FOV` 筛选和 `Non-adj` 筛选是记忆检索模块中两个最关键且有效的设计。

### 6.2.3. 开放领域泛化能力
作者还测试了模型在<strong>开放领域 (open-domain)</strong> 场景中的泛化能力。他们从网上找了一些风格各异的图片作为初始帧，然后使用“旋转离开再旋转回来”的相机轨迹进行长视频生成。如下图（原文 Figure 6）所示，即使面对训练时从未见过的场景（如水墨画风格的《黑神话：悟空》场景），模型依然能够展现出良好的记忆能力，在相机返回时恢复出一致的场景。这得益于其多样化的训练数据和预训练基座模型强大的先验知识。

![该图像是插图，展示了三组图像生成示例，每组对应不同的提示内容，如日本风景、黑神话悟空场景和幻想自然景观。这些图像展现了模型在结合历史上下文用于长视频生成中的应用，显示了场景一致性的能力。](images/6.jpg)
*该图像是插图，展示了三组图像生成示例，每组对应不同的提示内容，如日本风景、黑神话悟空场景和幻想自然景观。这些图像展现了模型在结合历史上下文用于长视频生成中的应用，显示了场景一致性的能力。*

---

# 7. 总结与思考

## 7.1. 结论总结
这篇论文针对交互式长视频生成中普遍存在的**场景不一致**问题，提出了一个名为 `Context-as-Memory` 的创新框架。其核心贡献在于：
1.  **提出新范式:** 将历史上下文直接作为可检索的外部记忆库，为解决长期依赖问题提供了一个简洁而强大的思路。
2.  **设计高效实现:** 通过“帧维度拼接”和“基于FOV的记忆检索”这两个简单而有效的设计，成功地在可控的计算成本下实现了强大的记忆能力。
3.  **提供有力证据:** 通过构建新数据集和全面的实验，证明了该方法在记忆能力和生成质量上均显著优于现有SOTA方法，并具备良好的开放领域泛化能力。

    总而言之，这项工作为实现真正可交互、可探索的生成式世界模型迈出了重要一步。

## 7.2. 局限性与未来工作
作者在论文中也坦诚地指出了当前方法的局限性，并展望了未来的研究方向：
*   **静态场景限制:** 当前的方法主要在**静态场景**中得到验证。对于包含复杂动态物体（如移动的行人、车辆）的场景，如何进行记忆检索是一个更大的挑战。因为场景内容本身在变化，简单的 `FOV` 重叠可能不足以判断相关性。
*   **复杂遮挡问题:** 在有复杂遮挡的场景（如室内多个相连的房间），仅靠 `FOV` 几何判断可能会失效，无法准确识别真正可见的上下文。
*   **误差累积问题:** 尽管有所缓解，但流式生成中固有的**误差累积**问题依然存在。生成视频越长，画质下降和内容漂移的风险就越大。这需要更强大的基座模型、更大规模的数据集和更长时间的训练来解决。
*   **未来方向:** 作者计划将该方法扩展到更大规模的基座模型上，支持更复杂的相机轨迹、更广阔的场景范围和更长的生成序列，最终目标是实现开放领域的自由、长时程导航。

## 7.3. 个人启发与批判
*   **启发:**
    1.  **奥卡姆剃刀原则:** 该研究最亮眼的地方在于其设计的简洁性。面对复杂的长期记忆问题，它没有堆砌复杂的网络模块，而是回归到“存储-检索”这一经典思路上，用一个巧妙的工程化方案（FOV检索）解决了核心瓶颈。这启示我们，在解决复杂问题时，简单、符合直觉的方案往往更有效。
    2.  **外部记忆的重要性:** 对于需要处理超长序列的任务，将所有信息都塞进模型有限的“内部工作记忆”（如Transformer的注意力上下文）中可能不是最优解。构建一个可扩展的外部记忆库，并设计高效的检索机制，可能是一个更具可扩展性的方向，这不仅适用于视频生成，也可能适用于长文本处理、具身智能等领域。

*   **批判性思考:**
    1.  **规则化检索的脆弱性:** 基于 `FOV` 的检索是一种硬编码的规则，虽然高效，但也可能很脆弱。例如，在有镜子反射的场景，或者透过窗户看到远处的情况下，几何上的 `FOV` 重叠与内容上的真正相关性可能会出现偏差。未来或许可以探索**可学习的检索模块**，让模型自己学会判断哪些历史帧是“值得回忆的”。
    2.  **对相机位姿的强依赖:** 该方法严重依赖精确的相机位姿信息。在无法获得精确位姿的真实世界视频中，需要先用 `SfM (Structure from Motion)` 等技术进行位姿估计，而估计误差可能会影响检索的准确性。
    3.  **记忆更新与遗忘机制:** 当前的记忆库是只增不减的，所有历史帧都被同等保存。对于一个真正高效的记忆系统，或许还需要引入**记忆更新**（例如用更高质量的重建来替代原始帧）和**遗忘**（丢弃不重要或冗余的记忆）机制，以管理不断增长的记忆库。