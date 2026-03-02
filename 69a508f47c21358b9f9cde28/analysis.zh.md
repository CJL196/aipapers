# 1. 论文基本信息

## 1.1. 标题
SkyReels-V4: Multi-modal Video-Audio Generation, Inpainting and Editing model (SkyReels-V4：多模态视频-音频生成、修复和编辑模型)

## 1.2. 作者
SkyReels Team, Skywork AI

## 1.3. 发表期刊/会议
预印本 (Preprint)，发布在 arXiv。arXiv 是一个收录物理学、数学、计算机科学、量化生物学、量化金融、统计学、电子工程及系统科学和经济学论文预印本的网站，在学术界具有重要的影响力，许多前沿研究成果都会首先在此发布。

## 1.4. 发表年份
2026年2月25日 (UTC)

## 1.5. 摘要
SkyReels V4 是一个统一的<strong>多模态 (multi-modal)</strong> 视频基础模型，用于联合视频-音频生成、修复（inpainting）和编辑。该模型采用<strong>双流多模态扩散变换器 (dual-stream Multimodal Diffusion Transformer, MMDiT)</strong> 架构，其中一个分支合成视频，另一个分支生成时间上对齐的音频，同时共享一个基于<strong>多模态大型语言模型 (Multimodal Large Language Models, MLLM)</strong> 的强大文本编码器。SkyReels V4 接受丰富的多模态指令，包括文本、图像、视频片段、掩码（masks）和音频参考。通过结合 MLLM 的多模态指令遵循能力与视频分支 MMDiT 中的<strong>上下文学习 (in-context learning)</strong>，该模型能够在复杂条件下注入细粒度的视觉引导，而音频分支 MMDiT 则同时利用音频参考来指导声音生成。在视频方面，该模型采用<strong>通道拼接 (channel concatenation)</strong> 的形式，统一了广泛的修复风格任务，例如图像到视频（image-to-video）、视频扩展（video extension）和视频编辑（video editing）到一个单一的接口下，并通过多模态提示（prompts）自然地扩展到视觉参考修复和编辑。SkyReels V4 支持高达 1080p 分辨率、32 FPS（每秒帧数）和15秒的时长，能够实现高保真度、多镜头、电影级别的视频生成和同步音频。为了使这种高分辨率、长时间的生成在计算上可行，该模型引入了一种效率策略：联合生成低分辨率的完整序列和高分辨率的关键帧，然后通过专门的超分辨率（super-resolution）和帧插值（frame interpolation）模型进行处理。据作者所知，SkyReels V4 是第一个同时支持多模态输入、联合视频-音频生成以及统一处理生成、修复和编辑的视频基础模型，同时在电影级分辨率和时长下保持强大的效率和质量。

## 1.6. 原文链接
原文链接: https://arxiv.org/abs/2602.21818
PDF 链接: https://arxiv.org/pdf/2602.21818v2
发布状态：预印本 (Preprint)

# 2. 整体概括

## 2.1. 研究背景与动机
**论文试图解决的核心问题:**
当前视频生成领域面临的主要挑战是缺乏一个能够**统一处理多模态输入、实现联合视频-音频生成、以及提供全面的修复和编辑功能**的单一框架。现有的最先进模型通常是碎片化的，例如：
*   <strong>音频驱动系统 (Audio-driven systems):</strong> 像 OmniHuman-1 和 Multitalk 等模型，通常采用浅层机制（如交叉注意力或权重提示），这些机制未能充分对齐音频-视觉表征，导致音视频不同步、唇语不匹配或单模态质量下降。
*   <strong>多模态参考模型 (Multi-modal-referenced models):</strong> 像 Kling-Omni 等模型，主要关注视觉条件的生成，缺乏原生的音频合成能力。
*   <strong>联合音视频生成模型 (Joint audio-video generation models):</strong> 即使是最近的 Kling 3.0、Seedance 2.0 和 Vidu-Q3 在整合多模态输入方面取得了进展，但它们仍然无法处理任意组合的文本、图像、视频、掩码和音频参考。
*   **计算效率:** 高分辨率、长时间的视频生成在计算上是昂贵的。

**为什么这个问题在当前领域是重要的？现有研究存在哪些具体的挑战或空白？**
随着多模态内容（尤其是视频）在现代媒体中的日益普及，对能够理解和生成复杂、高质量音视频内容的需求不断增长。电影制作、内容创作、虚拟现实等领域都需要高度逼真且可控的音视频生成工具。现有研究的碎片化和局限性导致：
*   **效率低下:** 需要多个模型或复杂的流水线来完成不同的任务，增加了工作流程的复杂性和成本。
*   **质量限制:** 各自为政的模态处理可能导致音视频内容缺乏整体连贯性和同步性，影响最终生成内容的质量和沉浸感。
*   **控制不足:** 难以在单一框架内实现对视觉和听觉元素的细粒度控制，限制了创作者的表达自由。
*   **计算门槛:** 高分辨率、长时长的音视频生成对计算资源要求极高，使得实际应用受限。

**这篇论文的切入点或创新思路是什么？**
SkyReels-V4 的创新切入点在于构建一个**统一的基础模型**，通过以下几点解决上述问题：
1.  **双流 MMDiT 架构:** 为视频和音频分别设计专门的分支，确保各自模态的生成质量，并通过共享的 MLLM 文本编码器和双向交叉注意力机制实现跨模态的语义对齐和时间同步。
2.  **丰富的多模态输入:** 接受文本、图像、视频片段、掩码和音频参考的任意组合，极大地增强了模型的灵活性和控制力。
3.  **统一的修复框架:** 引入**通道拼接**方法，将图像到视频、视频扩展、视频编辑等多种修复任务统一在一个接口下，简化了操作。
4.  **效率优化策略:** 采用“低分辨率完整序列 + 高分辨率关键帧”的联合生成策略，结合超分辨率和帧插值模型，实现了高分辨率、长时长的生成，同时保持计算可行性。
5.  <strong>上下文学习 (In-Context Learning) 和视觉引导:</strong> 利用 MLLM 的指令遵循能力和视频分支 MMDiT 中的上下文学习，注入细粒度的视觉引导。

## 2.2. 核心贡献/主要发现
论文最主要的贡献可以总结为以下几点：
*   **提出了 SkyReels-V4 模型:** 一个基于双流 MMDiT 架构的统一多模态视频基础模型，首次实现了联合视频-音频生成、修复和编辑的综合能力，并支持丰富的多模态输入。
*   **引入统一的视频修复框架:** 通过<strong>通道拼接 (channel concatenation)</strong> 实现了图像到视频、视频扩展、视频编辑和视觉参考修复等多种任务的统一处理。
*   **设计高效的生成策略:** 提出了**低分辨率完整序列与高分辨率关键帧联合生成**，并辅以超分辨率和帧插值模型的策略，使得 1080p、32 FPS、15秒、多镜头、带同步音频的视频生成在计算上变得可行。
*   **实现电影级质量和速度:** 模型能够在电影级别分辨率和时长下生成高质量的音视频内容，同时保持高效性。
*   **在评估中表现出色:** 在 `Artificial Analysis` 视频竞技场中排名第二，并在提出的 `SkyReels-VABench` 人工评估基准测试中，在指令遵循和运动质量方面表现尤为突出，总体平均得分最高，显著优于现有商业和开源基线模型。
*   **首创性声明:** 作者声称 SkyReels-V4 是第一个同时支持多模态输入、联合视频-音频生成以及统一处理生成、修复和编辑，同时在电影级分辨率和时长下保持强大效率和质量的视频基础模型。

# 3. 预备知识与相关工作

## 3.1. 基础概念

为了理解 SkyReels-V4 的工作原理，我们需要了解几个核心概念：

*   <strong>多模态 (Multi-modal):</strong> 指的是模型能够处理和整合来自多种不同类型数据的信息，例如文本、图像、视频和音频。在 SkyReels-V4 中，这意味着模型可以接受这些不同模态的输入，并生成跨模态（视频和音频）的输出。

*   <strong>扩散模型 (Diffusion Models):</strong> 一类生成模型，近年来在图像、视频和音频生成方面取得了巨大成功。它们的工作原理是逐步将噪声添加到数据中，然后在训练过程中学习如何逆转这个过程，即从噪声中恢复出清晰的数据。生成时，模型从纯噪声开始，通过多次迭代去噪，最终生成高质量的样本。
    *   **Latent Diffusion Models (LDMs):** 一种效率更高的扩散模型。它不直接在原始像素空间上操作，而是将高维数据（如图像）编码到低维的<strong>潜在空间 (latent space)</strong> 中，然后在这个潜在空间进行扩散和去噪操作。这样做可以显著降低计算成本，同时保持生成质量。SkyReels-V4 显然在潜在空间进行操作。

*   <strong>Transformer (变换器):</strong> 一种神经网络架构，最初为自然语言处理设计，以其<strong>注意力机制 (attention mechanism)</strong> 而闻名。注意力机制允许模型在处理序列数据时，动态地权衡输入序列中不同部分的相对重要性。这使得 Transformer 在处理长距离依赖关系方面表现出色，并已被广泛应用于视觉和多模态任务。
    *   **Diffusion Transformer (DiT):** 将 Transformer 架构应用于扩散模型中的去噪网络。它将潜在扩散模型的去噪 U-Net 替换为 Transformer，利用 Transformer 强大的表示能力和可扩展性来处理高维潜在表示。
    *   **Multimodal Diffusion Transformer (MMDiT):** SkyReels-V4 提出的核心架构。它扩展了 DiT，使其能够同时处理视频和音频两种模态，并通过共享组件和交叉注意力实现模态间的协同。

*   <strong>大型语言模型 (Large Language Models, LLMs) / 多模态大型语言模型 (Multimodal Large Language Models, MLLMs):</strong> LLMs 是在海量文本数据上训练的深度学习模型，能够理解、生成和处理人类语言。MLLMs 进一步扩展了 LLMs 的能力，使其能够处理和理解除了文本之外的其他模态数据，如图像、视频和音频。在 SkyReels-V4 中，MLLM 被用作一个强大的<strong>文本编码器 (text encoder)</strong>，负责理解和整合来自不同模态的指令。

*   <strong>上下文学习 (In-Context Learning):</strong> 大模型（特别是 LLMs）的一种能力，指的是模型通过分析输入中的示例（即“上下文”）来学习并执行新任务，而无需进行参数更新（即无需重新训练或微调）。在视频生成中，这意味着模型可以根据输入的参考图像或视频片段来理解生成的目标风格、内容或动作。

*   <strong>通道拼接 (Channel Concatenation):</strong> 一种常见的神经网络输入处理技术，指将不同特征图或张量沿通道维度进行堆叠。在 SkyReels-V4 中，它被用于统一不同的修复任务，通过将原始视频潜在表示、条件帧潜在表示和空间-时间掩码沿通道维度拼接起来，作为模型的输入。

*   <strong>超分辨率 (Super-Resolution, SR):</strong> 一种图像/视频处理技术，旨在从低分辨率（Low-Resolution, LR）输入生成高分辨率（High-Resolution, HR）输出，以提高图像或视频的视觉质量和细节。

*   <strong>帧插值 (Frame Interpolation, FI):</strong> 一种视频处理技术，用于在现有视频帧之间生成新的中间帧，从而提高视频的帧率（FPS）或使慢动作播放更流畅。

## 3.2. 前人工作
视频生成领域经历了显著的演进，从最初的单模态合成到现在的多模态联合生成。

*   **早期视频生成模型:**
    *   **2D+1D 架构:** `Video Diffusion Models` [21] 和 `AnimateDiff` [22] 等早期模型通过将 2D 图像处理与 1D 时间处理相结合来生成视频。
    *   **DiT-based 框架:** 随着 `Transformer` [23] 和 `Sora` [24] 的出现，基于 `Diffusion Transformer (DiT)` 的框架在处理大规模视频数据和时空注意力方面展现出巨大潜力。

*   **联合音视频生成模型:**
    *   **早期尝试:** `MM-Diffusion` [38] 等模型尝试使用耦合的 U-Net 架构进行音视频联合生成。
    *   **DiT-based 方法:** `AV-DiT` [39] 等将 `DiT` 应用于音视频领域。
    *   **专家组合与双流架构:** `MMDisCo` [40]、`UniVerse-1` [41] 采用专家组合或多模态判别器指导，`Ovi` [42]、`BridgeDiT` [43]、`JavisDiT` [44] 等则采用双流架构，通过交叉注意力或流匹配实现音视频交互。然而，这些方法往往计算成本较高。
    *   **效率优化:** `LTX-2` [36] 提出了非对称流以提高效率。
    *   **统一令牌:** `Apollo` [45] 尝试通过 `Omi-Full Attention` 联合处理音视频令牌，以实现多任务训练和更紧密的耦合。
    *   **挑战:** 尽管有这些进展，但同步的语音-视频合成和完整的音景生成仍未充分探索，精确的时空对齐仍然是一个开放的挑战。

*   **多模态参考视频生成:**
    *   **图像参考:** `Vidu` [7] 开创了从多张参考图像生成连贯视频的方法。
    *   **视频编辑:** `RunwayML Aleph` [8] 展示了基于上下文的视频编辑，能够添加、移除、转换对象，生成任意角度和修改风格/光照。
    *   **音频驱动:** `OmniHuman-1/1.5` [9, 10]、`SkyReels-A3` [11, 12] 等在驱动人脸动画和语音同步方面取得了显著成果。
    *   **图像与视频参考:** `Kling-Omni` [16] 是第一个支持图像和视频参考的视频生成模型，但仅限于视觉合成，缺乏音频输出。
    *   **桥接差距:** `Kling-3.0` [17]、`Seedance-2.0` [18]、`Vidu-Q3` [19] 正在努力将多模态输入与音视频联合生成结合起来，但仍未提供一个完全统一的解决方案。

## 3.3. 技术演进
从单模态的文本到视频（`Text-to-Video`, T2V）或视频到音频（`Video-to-Audio`, V2A）流水线，技术逐步演进到能够联合处理并生成音视频内容。早期模型往往独立处理各模态，导致音视频不同步、唇语不匹配等问题。随着 `Transformer` 架构和扩散模型的发展，模型开始能够更好地捕捉时空依赖和生成高质量内容。多模态大型语言模型（`MLLMs`）的兴起进一步推动了模型理解复杂多模态指令的能力。SkyReels-V4 正是站在这一技术浪潮的前沿，旨在通过统一的架构和高效的策略，解决现有模型的碎片化问题，实现更强大、更灵活、更高质量的音视频内容生成。

## 3.4. 差异化分析
与相关工作中的主要方法相比，SkyReels-V4 的核心区别和创新点在于：
*   **全面统一性:** 现有模型通常侧重于某个特定方面（例如，仅视觉生成、音频驱动、或多模态输入但缺乏音频输出）。SkyReels-V4 首次在一个单一的基础模型中<strong>同时统一了多模态输入（文本、图像、视频、掩码、音频参考）、联合视频-音频生成以及生成、修复和编辑这三大任务</strong>。
*   **双流 MMDiT 与 MLLM 结合:** 独特的双流 `MMDiT` 架构（一个视频分支、一个音频分支），通过共享的 `MLLM` 文本编码器，实现了深度的跨模态语义理解和时间同步，克服了浅层交互机制的局限性。
*   **创新的修复范式:** 引入<strong>通道拼接 (channel concatenation)</strong> 的统一修复框架，极大地简化和扩展了视频修复和编辑的能力，使其能够灵活处理多种修复风格的任务。
*   **高效的高分辨率生成:** 提出的“低分辨率完整序列 + 高分辨率关键帧”联合生成策略，结合专门的超分辨率和帧插值模型，是解决高分辨率、长时长音视频生成计算瓶颈的有效途径，使得电影级内容的生成在计算上变得可行。

# 4. 方法论

SkyReels-V4 的核心方法论围绕其<strong>双流多模态扩散变换器 (MMDiT)</strong> 架构展开，旨在统一多模态输入、联合视频-音频生成、以及视频的修复和编辑功能。

## 4.1. 方法原理
SkyReels-V4 的核心思想是构建一个统一的模型，能够从多种模态输入（文本、图像、视频、掩码、音频参考）生成同步的视频和音频。它通过以下几个关键设计实现这一目标：

1.  **双流架构:** 为视频和音频各设计一个独立的扩散模型分支，确保各自模态的生成质量。
2.  **共享 MLLM 文本编码器:** 利用一个强大的 `MLLM` 来编码多模态提示（prompt），提供统一的语义理解和指令遵循能力给两个分支。
3.  **跨模态交互:** 通过双向音频-视频交叉注意力以及对文本条件的强化，实现视频和音频之间的时间对齐和语义一致性。
4.  **统一修复框架:** 采用<strong>通道拼接 (channel concatenation)</strong> 的方式，将多种视频修复和编辑任务（如图像到视频、视频扩展、局部编辑）统一到同一个输入格式下。
5.  **效率策略:** 结合低分辨率全序列生成与高分辨率关键帧生成，并通过后处理的超分辨率和帧插值模型，实现高质量、长时长的生成。

    这种设计使得模型能够在一个框架内处理复杂的创作需求，同时保持计算效率和生成质量。

## 4.2. 核心方法详解

### 4.2.1. 双流 MMDiT 架构用于联合视频-音频生成
SkyReels-V4 的核心是一个**双流 MMDiT** 架构。这个架构包含两个独立的 `MMDiT` 分支：一个用于视频合成，另一个用于音频生成。虽然这两个分支是独立的，但它们通过共享的文本编码器和特殊的交叉注意力机制进行协同。

#### 4.2.1.1. 混合双流与单流 MMDiT 块 (Hybrid Dual-Stream and Single-Stream MMDiT Blocks)
每个 `Transformer` 块旨在通过混合架构平衡模态对齐和参数效率。
*   <strong>双流设计 (Dual-Stream Design):</strong> 在最初的 $M$ 层中，视频/音频和文本令牌（token）保持独立的参数（用于自适应层归一化 `LayerNormalization`、QKV 投影和多层感知机 `MLPs`）。然而，它们在<strong>联合自注意力 (joint self-attention)</strong> 过程中进行交互。
    其注意力计算公式如下：
    $$
    \begin{array} { r l } & { \mathbf { Q } _ { v } , \mathbf { K } _ { v } , \mathbf { V } _ { v } = \mathbf { Q } \mathbf { K } \mathbf { V } _ { v } ( \mathrm { LayerNorm } _ { v } ( \mathbf { x } _ { v } ) ) , } \\ & { \mathbf { Q } _ { t } , \mathbf { K } _ { t } , \mathbf { V } _ { t } = \mathbf { Q } \mathbf { K } \mathbf { V } _ { t } ( \mathrm { LayerNorm } _ { t } ( \mathbf { x } _ { t } ) ) , } \\ & { \qquad \mathbf { x } _ { v } ^ { \prime } , \mathbf { x } _ { t } ^ { \prime } = \mathrm { Attention } ( [ \mathbf { Q } _ { v } ; \mathbf { Q } _ { t } ] , [ \mathbf { K } _ { v } ; \mathbf { K } _ { t } ] , [ \mathbf { V } _ { v } ; \mathbf { V } _ { t } ] ) , } \end{array}
    $$
    其中：
    *   $\mathbf{x}_v$: 表示视频/音频令牌的嵌入 (embeddings)。
    *   $\mathbf{x}_t$: 表示文本令牌的嵌入。
    *   $\mathrm{LayerNorm}_v(\cdot)$ 和 $\mathrm{LayerNorm}_t(\cdot)$: 分别是视频/音频和文本模态的独立层归一化操作。
    *   $\mathbf{Q}\mathbf{K}\mathbf{V}_v(\cdot)$ 和 $\mathbf{Q}\mathbf{K}\mathbf{V}_t(\cdot)$: 分别是视频/音频和文本模态的独立查询（Query）、键（Key）、值（Value）投影。
    *   $[\cdot ; \cdot]$: 表示沿某个维度（通常是序列长度或令牌维度）进行拼接（concatenation）。在这里，它将视频/音频的 Q/K/V 与文本的 Q/K/V 拼接起来，形成一个统一的序列进行自注意力计算。
    *   $\mathrm{Attention}(\cdot)$: 标准的自注意力函数。
    *   $\mathbf{x}_v^\prime$ 和 $\mathbf{x}_t^\prime$: 经过自注意力计算后更新的视频/音频和文本令牌嵌入。
        这种设计在早期层中促进了强大的跨模态对齐。

*   <strong>单流设计 (Single-Stream Design):</strong> 随后的 $N$ 层则过渡到单流设计，所有令牌共享参数，以提高计算效率。这种混合策略比纯粹的双流或单流方法收敛更快。

#### 4.2.1.2. 通过交叉注意力强化文本条件 (Reinforced Text Conditioning via Cross-Attention)
为了解决文本特征可能出现的语义稀释问题，每个单流块还在自注意力之后额外应用了一个文本交叉注意力层。
其计算公式如下：
    $$
    \mathbf { x } _ { v } ^ { \prime \prime } = \mathbf { x } _ { v } ^ { \prime } + \mathrm { Attention } ( \mathbf { Q } = \mathbf { x } _ { v } ^ { \prime } , \mathbf { K } = \mathbf { x } _ { t } , \mathbf { V } = \mathbf { x } _ { t } ) ,
    $$
    其中：
    *   $\mathbf{x}_v^\prime$: 经过上一步联合自注意力后更新的视频/音频令牌嵌入。
    *   $\mathbf{x}_t$: 文本令牌嵌入。
    *   $\mathrm{Attention}(\mathbf{Q} = \mathbf{x}_v^\prime, \mathbf{K} = \mathbf{x}_t, \mathbf{V} = \mathbf{x}_t)$: 这是一个交叉注意力操作。视频/音频令牌 ($\mathbf{x}_v^\prime$) 作为查询（Query），文本令牌 ($\mathbf{x}_t$) 作为键（Key）和值（Value）。这允许视频/音频令牌从文本令牌中提取相关信息，从而在生成过程中强化文本的语义控制。
    *   $\mathbf{x}_v^{\prime\prime}$: 经过文本交叉注意力强化后，最终更新的视频/音频令牌嵌入。
        这个交叉注意力机制对于在模型后期阶段保持细粒度的语义控制至关重要。

#### 4.2.1.3. 双向音频-视频交叉注意力 (Bidirectional Audio-Video Cross-Attention)
为了实现模态之间的时间同步，每个 `Transformer` 块都包含一对交叉注意力层：音频特征关注视频特征，视频特征反过来关注音频特征。这种双向机制在整个网络深度中交换同步线索。
其计算公式如下：
    $$
    \begin{array} { r } { { \bf a } _ { i } ^ { \prime } = { \bf a } _ { i } + { \bf C } \mathrm { r o s s A t t n } ( { \bf Q } = { \bf a } _ { i } , { \bf K } = { \bf v } _ { i } , { \bf V } = { \bf v } _ { i } ) , } \\ { { \bf v } _ { i } ^ { \prime \prime } = { \bf v } _ { i } ^ { \prime } + { \bf C } \mathrm { r o s s A t t n } ( { \bf Q } = { \bf v } _ { i } ^ { \prime } , { \bf K } = { \bf a } _ { i } ^ { \prime } , { \bf V } = { \bf a } _ { i } ^ { \prime } ) , } \end{array}
    $$
    其中：
    *   $\mathbf{a}_i$: 表示在第 $i$ 层音频特征。
    *   $\mathbf{v}_i$: 表示在第 $i$ 层视频特征。
    *   $\mathrm{CrossAttn}(\cdot)$: 交叉注意力操作。
    *   $\mathbf{a}_i^\prime$: 音频特征通过关注视频特征得到更新。
    *   $\mathbf{v}_i^\prime$: 视频特征在第一步交叉注意力后（未在公式中直接显示，但从 $\mathbf{v}_i^{\prime\prime}$ 的输入可以看出）得到更新。
    *   $\mathbf{v}_i^{\prime\prime}$: 视频特征通过关注更新后的音频特征 ($\mathbf{a}_i^\prime$) 再次得到更新。
        这种架构对称性确保了两种模态共享相同的注意力模式，并从单模态预训练中相互受益。

#### 4.2.1.4. 旋转位置嵌入 (RoPE) 的时序对齐 (Temporal Alignment with RoPE)
尽管视频和音频的潜在表示在时序分辨率上可能不匹配（例如，视频跨越21帧，而音频包含218个令牌），模型通过对两种模态应用<strong>旋转位置嵌入 (Rotary Positional Embeddings, RoPE)</strong> 来对齐它们的时间尺度。音频 `RoPE` 的频率按 $21 / 218 \approx 0.09633$ 的比例进行缩放，以匹配视频的时序分辨率。这确保了音频和视频令牌在时序上进行一致的对应。

#### 4.2.1.5. 共享多模态文本编码器 (Shared Multi-Modal Text Encoder)
为了简化提示（prompt）条件设置，模型采用一个<strong>单一的冻结 MLLM 文本编码器 (frozen MLLM text encoder)</strong>，用于处理结合了视觉和听觉描述的组合提示。由此产生的多模态嵌入（multi-modal embeddings）由音频和视频分支独立地通过自注意力（`self-attention`）和交叉注意力（`cross-attention`）机制消费。这实现了语义连贯的条件化。

#### 4.2.1.6. 流匹配损失函数 (Flow Matching Loss Function)
SkyReels-V4 采用流匹配（Flow Matching）范式进行训练。对于给定的视频潜在表示 $\mathbf{z}_v^0$ 和音频潜在表示 $\mathbf{z}_a^0$，模型采样一个时间步 $t \sim \mathcal{U}(0, 1)$ 并构建噪声潜在表示 $\mathbf{z}_v^t = t \mathbf{z}_v^0 + (1-t) \epsilon_v$ 和 $\mathbf{z}_a^t = t \mathbf{z}_a^0 + (1-t) \epsilon_a$，其中 $\epsilon_v, \epsilon_a \sim \mathcal{N}(0, \mathbf{I})$ 是高斯噪声。模型预测一个速度场 $\mathbf{v}_\theta$，该速度场将噪声推向数据。
损失函数定义如下：
    $$
    \mathcal { L } _ { \mathrm { f l o w } } = \mathbb { E } _ { t , z _ { v } ^ { 0 } , z _ { a } ^ { 0 } , \epsilon _ { v } , \epsilon _ { a } } \left[ \left\| \mathbf { v } _ { \theta } ^ { v } ( t , \mathbf { z } _ { v } ^ { t } , \mathbf { z } _ { a } ^ { t } , \mathbf { c } ) - ( \mathbf { z } _ { v } ^ { 0 } - \epsilon _ { v } ) \right\| ^ { 2 } + \left\| \mathbf { v } _ { \theta } ^ { a } ( t , \mathbf { z } _ { a } ^ { t } , \mathbf { z } _ { v } ^ { t } , \mathbf { c } ) - ( \mathbf { z } _ { a } ^ { 0 } - \epsilon _ { a } ) \right\| ^ { 2 } \right] ,
    $$
    其中：
    *   $\mathcal{L}_{\mathrm{flow}}$: 流匹配损失。
    *   $\mathbb{E}[\cdot]$: 期望操作，表示对时间 $t$、原始潜在表示 ($\mathbf{z}_v^0, \mathbf{z}_a^0$) 和噪声 ($\epsilon_v, \epsilon_a$) 进行平均。
    *   $\mathbf{v}_\theta^v(\cdot)$ 和 $\mathbf{v}_\theta^a(\cdot)$: 模型预测的视频和音频模态的速度场。
    *   $\mathbf{c}$: 表示条件信息，包括多模态嵌入和可选的时空掩码。
    *   $\|\cdot\|^2$: L2 范数的平方，表示预测速度场与目标速度场之间的差异。
        这个联合目标函数在生成过程中同时训练两个分支，实现同步化，同时保留了每个模态的特定特征。

### 4.2.2. 通过通道拼接实现统一视频修复 (Unified Video Inpainting via Channel Concatenation)
SkyReels-V4 采用<strong>通道拼接 (channel concatenation)</strong> 的方式来统一各种视频修复风格任务。它将噪声视频潜在表示、VAE 编码的条件帧以及空间-时间掩码沿着通道维度拼接起来，作为模型的输入。
输入公式如下：
    $$
    { \bf Z } _ { \mathrm { i n p u t } } = \mathrm { Concat } ( { \bf V } , { \bf I } , { \bf M } ) ,
    $$
    其中：
    *   $\mathbf{Z}_{\mathrm{input}}$: 模型的最终输入潜在表示。
    *   $\mathrm{Concat}(\cdot)$: 拼接操作，将多个张量沿通道维度堆叠。
    *   $\mathbf{V} \in \mathbb{R}^{T \times H \times W \times C}$: 噪声视频潜在表示（noisy video latent）。$T$ 是时间维度，$H$ 是高度，$W$ 是宽度， $C$ 是通道数。
    *   $\mathbf{I} \in \mathbb{R}^{T \times H \times W \times C}$: 包含 VAE 编码的条件帧（VAE-encoded conditional frames）。这些帧提供模型在生成或修复时需要遵循的视觉信息。
    *   $\mathbf{M} \in \mathbb{R}^{T \times H \times W \times 1}$: 空间-时间掩码（spatiotemporal mask）。掩码的值为 1 表示条件区域（需要保留或作为条件），值为 0 表示需要生成或修复的区域。

        这种形式通过不同的掩码配置统一了多种生成任务：
*   <strong>文本到视频 (T2V):</strong> $\mathbf{M} = \mathbf{0}$（所有帧都由模型生成）。
*   <strong>图像到视频 (I2V):</strong> $\mathbf{M}_{t=0} = \mathbf{1}, \mathbf{M}_{t>0} = \mathbf{0}$（仅第一帧作为条件，后续帧生成）。
*   <strong>视频扩展 (Video Extension):</strong> $\mathbf{M}_{t<k} = \mathbf{1}, \mathbf{M}_{t \geq k} = \mathbf{0}$（前 $k$ 帧作为条件，扩展后续帧）。
*   <strong>起始-结束帧插值 (Start-End Frame Interpolation):</strong> $\mathbf{M}_{t=0} = \mathbf{M}_{t=T-1} = \mathbf{1}$，其他为 $\mathbf{0}$（仅起始帧和结束帧作为条件，插值中间帧）。
*   <strong>视频编辑 (Video Editing):</strong> $\mathbf{M}_{t,h,w} = \mathbf{1}$ 用于保留区域，$\mathbf{0}$ 用于编辑区域（任意空间-时间掩码）。

    这种统一的公式自然地适应了固定前景/背景掩码和动态逐帧编辑掩码，实现了对空间和时间修改的精确控制。音频分支通过双向交叉注意力机制保持与视频修改的时间同步。

### 4.2.3. 多模态上下文学习用于视觉参考生成和编辑 (Multi-Modal In-Context Learning for Vision-Referenced Generation and Editing)
除了文本和修复掩码，SkyReels-V4 还通过参考图像和视频片段支持多模态条件设置，从而实现复杂<strong>视觉参考生成任务 (vision-referenced generation tasks)</strong>，例如多身份视频生成和保持身份的视频编辑。

#### 4.2.3.1. MLLM 的多模态指令遵循 (Multi-Modal Instruction Following with MLLM)
模型将所有参考输入（图像、视频、音频）与文本提示一起通过 `MLLM` 文本编码器进行联合处理，以提取语义丰富的多模态嵌入。`MLLM` 的指令遵循能力使其能够理解复杂的请求，例如“以人物 B 的风格说‘你好吗？’（`hello, how are you`），并参考视频 $video_1$”。这些多模态嵌入被视频和音频分支共同消费。

#### 4.2.3.2. 通过自注意力进行上下文视觉条件设置 (In-Context Visual Conditioning via Self-Attention)
为了提供除语义 MLLM 嵌入之外的显式视觉参考信号，模型还直接将 VAE 编码的参考图像或视频帧作为<strong>条件潜在表示 (condition latents)</strong> 注入到视频分支。
这些条件潜在表示 $\mathbf{Z}_{\mathrm{cond}}$ 被预置（prepended）到噪声视频潜在表示 $\mathbf{Z}_{\mathrm{video}}$ 之前，再进行自注意力计算：
    $$
    \mathbf { Z } _ { \mathrm { attn } } = [ \mathbf { Z } _ { \mathrm { cond } } ; \mathbf { Z } _ { \mathrm { video } } ] ,
    $$
    其中：
    *   $\mathbf{Z}_{\mathrm{attn}}$: 用于自注意力计算的输入序列。
    *   $\mathbf{Z}_{\mathrm{cond}}$: VAE 编码的条件潜在表示，包含参考图像或视频帧的视觉信息。
    *   $\mathbf{Z}_{\mathrm{video}}$: 噪声视频潜在表示。
        这种方法允许模型在生成或编辑视频内容时直接利用视觉信息。

#### 4.2.3.3. 通过带偏移的 3D RoPE 进行时序位置消歧 (Temporal Positional Disambiguation via Offset 3D RoPE)
为了区分条件潜在表示和噪声视频潜在表示，并组织多个参考视觉信息，模型采用了带有时间索引偏移的 3D <strong>旋转位置嵌入 (Rotary Positional Embeddings, RoPE)</strong>。条件潜在表示接收负时间索引，而实际视频帧接收正时间索引。
其时序位置编码公式如下：
    $$
    \mathrm { RoPE } _ { \mathrm { temporal } } ( \mathbf { Z } _ { \mathrm { cond } , i } ) = \mathrm { RoPE } ( t = - N _ { \mathrm { cond } } + i ) , \quad \mathrm { RoPE } _ { \mathrm { temporal } } ( \mathbf { Z } _ { \mathrm { video } , j } ) = \mathrm { RoPE } ( t = j ) ,
    $$
    其中：
    *   $\mathrm{RoPE}_{\mathrm{temporal}}(\cdot)$: 时序旋转位置嵌入函数。
    *   $\mathbf{Z}_{\mathrm{cond},i}$: 第 $i$ 个条件令牌。
    *   $\mathbf{Z}_{\mathrm{video},j}$: 第 $j$ 个视频令牌。
    *   $N_{\mathrm{cond}}$: 条件令牌的总数。
    *   `i, j`: 分别是条件令牌和视频令牌的索引。
        这种基于偏移量的位置编码提供了一种有效且具归纳偏置的方式，用于区分条件信息和需要生成的数据，并处理不同类型（图像、短片段等）的参考视觉信息。

#### 4.2.3.4. 音频参考条件 (Audio Reference Conditioning)
类似地，音频参考（例如，语音样本、音乐主题、环境音景）可以通过 `MLLM` 进行处理，并直接注入到音频分支中。通过结合 `MLLM` 的多模态指导、视频分支的上下文视觉模式和音频参考的音频模式，模型实现了对视觉和听觉生成的细粒度控制。

### 4.2.4. 数据流水线 (Data Pipeline)
模型的训练依赖于一个大规模的数据流水线，该流水线处理三种模态：图像、视频和音频。

#### 4.2.4.1. 数据收集 (Data Collection)
*   <strong>真实世界数据 (Real-World Data):</strong> 收集了大量的公开和许可数据。
    *   **公开数据:** 图像数据集（如 LAION [46], Flickr [47]）、视频数据集（如 WebVid-10M [48], Koala-36M [49], OpenHumanVid [50]）和音频数据集（如 Emilia [51], AudioSet [52], VGGSound [53], SoundNet [54]）。
    *   **许可数据:** 授权电影、电视剧、短视频和网络剧。
*   <strong>合成数据 (Synthetic Data):</strong> 生成合成数据以弥补真实世界数据在稀疏场景和某些生成任务上的不足，包括多语言文本生成、多语言语音合成和多模态修复/编辑任务。
    *   **图像-文本数据:** 包含简单的文本渲染和上下文感知文本。
    *   **语音合成数据:** 应用多个 TTS 模型以覆盖多语言，包括韩语、德语、法语等。
    *   **修复数据:** 通过复杂的流水线构建，涉及视觉表示模型、图像/视频编辑模型和可控生成技术。

#### 4.2.4.2. 数据处理 (Data Processing)
数据处理流水线旨在保持高质量和多样性。

*   <strong>文本数据处理 (Text Data Processing):</strong>
    *   **质量筛选:** 基于文本质量、长度、主题和语法平衡数据集。
    *   **图像-文本平衡:** 使用 `Qwen3-Omni` [55] 等模型评估图像和文本之间的对齐，以进行细粒度平衡。
*   <strong>音频数据处理 (Audio Data Processing):</strong>
    *   **质量控制:** 计算语音活动率（`VAD`）、质量（`SNR`）、清晰度、自然度、情感和唱歌质量（使用 `Qwen3-Omni`）。
    *   **质量过滤:** 基于信噪比（`SNR`）、`MOS` 分数、裁剪比和音频带宽进行过滤。
    *   <strong>语音活动检测 (VAD):</strong> 筛选出语音活动率低于 0.1 的音频。
    *   **时间对齐:** 对于长音频，使用 `Whisper` 等模型进行语音转文本（`ASR`），并使用 `Qwen3-Omni` 统一标注所有音频。
*   <strong>视频数据处理 (Video Data Processing):</strong>
    *   **预处理:** 视频被分割成片段，并进行去重。传统方法（如 `PySceneDetect`, `TransNet-V2` [56]）生成的场景切割片段可能不适合 `VLM` 风格的上下文学习。因此，模型采用 `VideoCLIP` 嵌入 [57] 来识别场景变化并生成更具语义的片段。
    *   **过滤:** 基于帧质量（清晰度、伪影）、风格（动画、电影、真人）、运动质量（相机稳定性、运动幅度/速度、掉帧）进行过滤。
    *   **平衡:** 为了提高训练效率，数据在概念多样性（确保特定概念如“人说话”或“汽车在路上”）和风格多样性上进行平衡。
    *   <strong>音视频同步 (Audio-Video Synchronization):</strong> 使用 `SyncNet` [58] 模型，该模型利用 `ConvNet` 架构学习声音和嘴部图像之间的联合嵌入，识别出音频和视频中的关键帧，并生成标量置信度（confidence）和偏移量（offset）值。只保留满足 $|offset| \le 3$ 和 confidence $> 1.5$ 的片段。最后，所有视频片段都通过 `Qwen3-Omni` 进行标注。

#### 4.2.4.3. 标注 (Captioning)
模型训练的输入包括简洁的视频内容和音频信息描述。长标注（long captions）提供全面的主题、灯光、氛围等细节描述。结构化标注遵循分层描述顺序，并使用特殊令牌（tokens）来表示视频内文本（$<text></text>$）、音效（$<sfx></sfx>$）、语音内容（$<dialogue></dialogue>$）、歌唱内容（$<singing></singing>$）和背景音乐（$<bgm></bgm>$）。在实际训练中，使用一个提示增强器（`prompt enhancer`）将自由形式的输入转换为这种结构化表示。

### 4.2.5. 训练策略 (Training Strategy)
模型采用渐进式多阶段训练范式，系统地发展模型在视觉空间概念、时序动态、音频持续时间和多模态对齐方面的能力。每个阶段通常训练3个周期（epochs）。

#### 4.2.5.1. 视频预训练 (Video Pretrain)
*   <strong>阶段 1: 文本到图像基础 (Text-to-Image Foundation):</strong> 首先在 256px 分辨率下使用 30 亿张图像训练 `T2I` 任务 3 个周期，为空间构图和概念形成打下基础。
*   <strong>阶段 2: 初始视频学习 (Initial Video Learning):</strong> 引入 `T2V`（文本到视频）生成，同时保持 `T2I` 训练。在 256px 分辨率、16fps、2-10秒视频时长下，使用 10 亿张图像和 4 亿个视频训练 3 个周期。低分辨率训练允许模型更快地收敛于运动动态和时序连贯性。
*   <strong>阶段 3: 视频修复与编辑 (Video Inpainting and Editing):</strong> 引入图像修复、`I2V`、`V2V`（视频到视频）和视频编辑任务，每种任务占训练混合的 5%。此阶段训练 2 个周期，视频时长延长至 15 秒，使模型学习空间和时间修复能力。
*   <strong>阶段 4: 混合分辨率缩放 (Mixed Resolution Scaling):</strong> 在 256px 和 480px 的混合分辨率下进行训练，保持 16fps 和 2-15秒的视频时长。使用 1 亿张图像和 1 亿个视频训练 2 个周期，修复任务比例不变，使模型逐渐适应更高分辨率的生成。
*   <strong>阶段 5: 高分辨率训练 (High Resolution Training):</strong> 进一步扩展到 480px、720px 和 1080px 的混合分辨率，16fps，视频时长 3-15秒。使用 5000 万张图像和 5000 万个视频训练 2 个周期，显著提高模型的高分辨率生成质量。
*   <strong>阶段 6: 多模态条件预训练 (Multi-modal Condition Pretrain):</strong> 引入图像参考和视频参考条件，用于生成和修复任务，各占训练数据的 20%，剩余 60% 用于 `T2V`。此阶段在 2000 万张图像和 5000 万个视频上训练 2 个周期，使模型具备灵活的多模态条件能力。

    以下是原文表格 1 的内容：

    <table>
    <thead>
    <tr>
    <td rowspan="1">Task</td>
    <td rowspan="1">Stage</td>
    <td rowspan="1">Resolution</td>
    <td rowspan="1">Data Volume</td>
    <td rowspan="1">Epochs</td>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td rowspan="1" colspan="5">Video Pretrain</td>
    </tr>
    <tr>
    <td>T2I</td>
    <td>Stage 1</td>
    <td>256px</td>
    <td>3B images</td>
    <td>3</td>
    </tr>
    <tr>
    <td>T2I + T2V</td>
    <td>Stage 2</td>
    <td>256px, 16fps, 2-10s</td>
    <td>1B images / 400M videos</td>
    <td>3</td>
    </tr>
    <tr>
    <td>T2I + T2V + Inpaint(Image Inpaint, I2V, V2V, Edit)</td>
    <td>Stage 3</td>
    <td>256px, 16fps, 2-15s(Inpaint: 5% each)</td>
    <td>1B images / 400M videos</td>
    <td>2</td>
    </tr>
    <tr>
    <td>Mixed Tasks(T2I, T2V, Inpaint)</td>
    <td>Stage 4</td>
    <td>256/480px, 16fps, 2-15s(Inpaint ratio unchanged)</td>
    <td>100M images / 100M videos</td>
    <td>2</td>
    </tr>
    <tr>
    <td>Mixed Tasks(T2I, T2V, Inpaint)</td>
    <td>Stage 5</td>
    <td>480/720/1080px,16fps, 3-15s</td>
    <td>50M images / 50M videos</td>
    <td>2</td>
    </tr>
    <tr>
    <td>Multi-modal Condition(Image/Video Ref: 20% each)(T2V: 60%)</td>
    <td>Stage 6</td>
    <td>480/720/1080px,16fps, 3-15s</td>
    <td>20M images / 50M videos</td>
    <td>2</td>
    </tr>
    <tr>
    <td rowspan="1" colspan="5">Audio Pretrain</td>
    </tr>
    <tr>
    <td>Audio Backbone</td>
    <td colspan="2">Pretrain Variable length, up to 15s</td>
    <td>Hundreds of thousands of hours</td>
    <td>3</td>
    </tr>
    <tr>
    <td rowspan="1" colspan="5">Video-Audio Joint Training</td>
    </tr>
    <tr>
    <td>T2V + T2AV + T2A</td>
    <td>Joint Pretrain</td>
    <td>720/1080px, 16fps, 5-15s</td>
    <td>50% video data + T2A data</td>
    <td>2</td>
    </tr>
    <tr>
    <td rowspan="1" colspan="5">Video-Audio Supervised Fine-tuning</td>
    </tr>
    <tr>
    <td>T2AV + Multi-modal</td>
    <td>SFT Stage 1</td>
    <td>720/1080px, 16fps, 5-15s</td>
    <td>5M videos (Multi-modal: 20%)</td>
    <td>3</td>
    </tr>
    <tr>
    <td>T2AV + Multi-modal</td>
    <td>SFT Stage 2</td>
    <td>720/1080px, 16fps, 5-15s</td>
    <td>1M curated videos</td>
    <td>3</td>
    </tr>
    </tbody>
    </table>

#### 4.2.5.2. 音频预训练 (Audio Pretrain)
音频主干网络从头开始预训练，使用了数十万小时的语音数据，重点是保留音高的音色和自然度。音频预训练使模型能够生成与说话者特质（如音高和情感）一致的音频。

#### 4.2.5.3. 视频-音频联合训练 (Video-Audio Joint Training)
此阶段旨在训练复杂的任务，如 `T2V`、`T2AV`（文本到音视频）和 `T2A`（文本到音频）。它利用视频预训练数据进行 `T2V` 联合训练，同时整合 `T2A` 数据以实现同步的音视频生成。

#### 4.2.5.4. 视频-音频监督微调 (Video-Audio Supervised Fine-tuning, SFT)
*   **SFT 阶段 1:** 专注于 `T2AV` 和多模态条件支持（图像、视频和音频），其中多模态数据占 20%。
*   **SFT 阶段 2:** 最后在 100 万个手工筛选的高质量视频上进行微调，以进一步提升生成质量、运动连贯性和音视频对齐。

#### 4.2.5.5. 视频超分辨率和帧插值 (Refiner) (Video Super-Resolution and Frame Interpolation (Refiner))
为了进一步增强生成视频的视觉质量和时序平滑度，模型引入了一个专门的<strong>精炼器 (Refiner)</strong>，负责<strong>超分辨率 (super-resolution)</strong> 和<strong>帧插值 (frame interpolation)</strong>。

<strong>架构和设计 (Architecture and Design):</strong>
精炼器权重从预训练的视频生成模型初始化，以确保在后训练阶段的无缝切换。基础模型同时预测低分辨率的所有帧和高分辨率的关键帧潜在表示。最后，组合后的潜在表示与高分辨率潜在表示沿通道维度重新拼接，作为 `DiT` 模型的输入。
精炼器的设计允许它识别需要改进的区域和应保持不变的区域，从而能够处理无条件超分辨率和多模态条件修复。

<strong>计算效率 (Computational Efficiency):</strong>
为了解决长时序上下文和高分辨率输入带来的计算开销，模型采用了<strong>视频稀疏注意力 (Video Sparse Attention, VSA) [59]</strong>。`VSA` 是一种可训练的稀疏注意力机制，专门为视频扩散 `Transformer` 设计。`VSA` 采用分层两阶段方法：
1.  **粗略阶段:** 聚合稀疏注意力，同时通过与现代 `GPU` 内核兼容的块稀疏布局保持硬件效率。
2.  **细致阶段:** 在局部区域进行更密集的注意力计算。
    通过以可学习的方式利用时空冗余，`VSA` 能够将注意力计算成本降低约 3 倍，同时保持生成质量，使得在训练和推理过程中处理高分辨率视频序列变得可行。

<strong>训练数据配置 (Training Data Configuration):</strong>
精炼器的数据集构建涉及数百万个高质量视频片段，以确保模型能够处理各种分辨率和细节。精炼器也遵循流匹配范式进行训练。

## 4.3. 整体方法概览图

下图（原文 Figure 1）展示了 SkyReels-V4 的整体架构。它清晰地展示了双流 MMDiT 架构，视频和音频分支如何分别处理噪声，并如何通过共享的 MLLM text encoder、text cross-attention 和 bidirectional AV cross-attention 进行交互。同时，也展示了流匹配损失在训练中的应用。

![Figure 1: Overview of the proposed method.](images/1.jpg)
*该图像是一个示意图，展示了SkyReels V4中双流多模态扩散变换器(MMDiT)的结构，包括视频和音频的流匹配损失和块结构，采用多层嵌入与模型编码器。图中展示了视频和音频的单流与双流处理模块，以及文本的交叉注意力机制和临时拼接的方式。*

图 1: SkyReels-V4 提出的方法概览。

下图（原文 Figure 2）展示了 SkyReels-V4 的精炼器（Refiner）架构概览，包括低分辨率、高分辨率关键帧的生成，以及超分辨率和帧插值的流程。

![该图像是一个示意图，展示了多模态视频、音频生成与编辑的流程。图中包含有参考图像、视频和音频的信息，模型架构通过基本模型和VAE解码器处理噪声以生成最终内容。](images/2.jpg)
*该图像是一个示意图，展示了多模态视频、音频生成与编辑的流程。图中包含有参考图像、视频和音频的信息，模型架构通过基本模型和VAE解码器处理噪声以生成最终内容。*

图 2: SkyReels-V4 精炼器架构概览。

# 5. 实验设置

## 5.1. 数据集
论文并未明确指出使用了哪些特定的公开数据集，而是概括性地提到了以下几类数据源：
*   **真实世界数据:**
    *   **图像:** LAION [46], Flickr [47]
    *   **视频:** WebVid-10M [48], Koala-36M [49], OpenHumanVid [50]
    *   **音频:** Emilia [51], AudioSet [52], VGGSound [53], SoundNet [54]
    *   **许可数据:** 授权的电影、电视剧、短视频和网络剧。
*   **合成数据:** 用于解决稀疏场景和特定生成任务（如多语言文本、多语言语音合成、多模态修复/编辑）。

**数据集特点和选择理由:**
这些数据集涵盖了广泛的图像、视频和音频内容，具备：
*   **多样性:** 包含各种场景、主题、风格、语言和音效，以确保模型学习到丰富的表示能力。
*   **规模:** 大规模的数据量（数十亿图像、数亿视频、数十万小时音频）是训练强大基础模型的关键。
*   **多模态性:** 某些数据集（如 VGGSound, AudioSet）本身就包含音视频对，有利于联合训练。
*   **质量:** 经过严格的数据处理和过滤，确保数据的质量和同步性，这对于高保真生成至关重要。

    选择这些数据集是为了：
*   **支持多模态训练:** 模型的双流架构和多模态指令遵循能力需要来自各种模态的数据进行训练。
*   **覆盖广泛任务:** 从 `T2I`、`T2V` 到复杂的修复和编辑任务，都需要相应的数据来学习。
*   **提高生成质量:** 大规模、高质量的数据是实现电影级分辨率和高保真输出的基础。
*   **解决数据稀疏性:** 合成数据填补了真实世界数据在某些特定或复杂场景下的不足。

## 5.2. 评估指标
SkyReels-V4 的性能评估分为两个主要部分：`Artificial Analysis Arena` 的公共排行榜和 `SkyReels-VABench` 的人工评估。人工评估涵盖了五个关键维度。

### 5.2.1. 人工评估维度 (`SkyReels-VABench`)

该基准测试扩展了先前的 `SkyReels-Bench` [32]，增加了全面的音频维度和多镜头视频场景。它包含 2000 多个精心策划的提示，涵盖了各种内容复杂性、语言和模态。

**五个主要评估维度和其子维度及评估标准如下：**

1.  <strong>指令遵循 (Instruction Following)</strong>
    *   **概念定义:** 衡量模型生成的视频和音频内容在多大程度上准确地满足了用户提供的文本、图像、视频、掩码和音频参考指令。
    *   **子维度:**
        *   <strong>视频指令遵循 (Video Instruction Following):</strong>
            *   **主体描述:** 主体、属性和外观的准确表示。
            *   **主体交互:** 动作、交互和运动动态的正确执行。
            *   **摄像机运动:** 摄像机操作（平移、倾斜、缩放、推拉）的正确执行。
        *   <strong>风格和美学 (Style and aesthetics):</strong>
            *   **视觉风格、调色板和艺术方向的遵守。**
        *   <strong>多镜头一致性 (Multi-shot consistency):</strong>
            *   **镜头过渡、跨镜头连贯性和参考准确性。**
        *   <strong>音频指令遵循 (Audio Instruction Following):</strong>
            *   **语义一致性:** 与音频内容和特征的保真度。

2.  <strong>音视频同步 (Audio-Visual Synchronization)</strong>
    *   **概念定义:** 评估视频中的视觉事件与音频中的声音事件之间的时间对齐和一致性。
    *   **子维度:**
        *   <strong>唇语同步准确性 (Lip-sync accuracy):</strong> 精确的语音-嘴部同步和正确的说话人识别。
        *   <strong>音效对齐 (Sound effect alignment):</strong> 视觉事件与音效之间的时间对应。
        *   <strong>氛围匹配 (Atmospheric matching):</strong> 背景音乐、场景氛围和情感基调之间的一致性。

3.  <strong>视觉质量 (Visual Quality)</strong>
    *   **概念定义:** 衡量生成视频的视觉清晰度、色彩准确性、构图和是否存在视觉伪影。
    *   **子维度:**
        *   <strong>视觉清晰度 (Visual clarity):</strong> 清晰度、细节和分辨率。
        *   <strong>色彩准确性 (Color accuracy):</strong> 自然的色彩平衡和饱和度，无失真。
        *   <strong>构图质量 (Compositional quality):</strong> 审美构图、取景和视觉平衡。
        *   <strong>结构完整性 (Structural integrity):</strong> 无视觉伪影和损坏。

4.  <strong>运动质量 (Motion Quality)</strong>
    *   **概念定义:** 评估生成视频中运动的逼真度、流畅性、稳定性和时间一致性。
    *   **子维度:**
        *   <strong>物理合理性 (Physical plausibility):</strong> 遵守物理定律（重力、惯性、动量）。
        *   <strong>运动流畅性 (Motion fluidity):</strong> 平滑的过渡，无突然中断。
        *   <strong>运动稳定性 (Motion stability):</strong> 无抖动、变形和闪烁。
        *   <strong>时间一致性 (Temporal consistency):</strong> 跨帧动态元素的一致性。
        *   <strong>运动生动性 (Motion vividness):</strong> 动作、摄像机、氛围和情感表现力。

5.  <strong>音频质量 (Audio Quality)</strong>
    *   **概念定义:** 衡量生成音频的清晰度、音色真实性、空间感和动态范围，以及是否存在伪影。
    *   **子维度:**
        *   <strong>无伪影 (Absence of artifacts):</strong> 无削波、截断、失真或故障。
        *   <strong>空间声场 (Spatial soundstage):</strong> 适当的立体声成像和空间渲染。
        *   <strong>音色真实性 (Timbre realism):</strong> 自然逼真的音调质量。
        *   <strong>信号清晰度 (Signal clarity):</strong> 清晰的音频，具有适当的信噪比。
        *   <strong>动态范围 (Dynamic range):</strong> 适当的音量变化，无压缩伪影。

            以下是原文表格 2 的内容：

            <table>
            <thead>
            <tr>
            <td>Dimension</td>
            <td>Sub-dimension</td>
            <td>Evaluation Criteria</td>
            </tr>
            </thead>
            <tbody>
            <tr>
            <td rowspan="4">Instruction Follow- ing</td>
            <td>Video Instruction Following Subject description Subject interaction Camera movement</td>
            <td>Accurate representation of subjects, attributes, and appearances Correct execution of actions, interactions, and motion dynamics Proper execution of camera operations (pan, tilt, zoom, dolly)</td>
            </tr>
            <tr>
            <td>Style and aesthetics Multi-shot consistency Audio Instruction Following Semantic adherence</td>
            <td>Adherence to visual styles, color palettes, and artistic directions Correct shot transitions, cross-shot coherence, and reference accuracy Fidelity to audio content and characteristics</td>
            </tr>
            <tr>
            <td></td>
            <td></td>
            </tr>
            <tr>
            <td></td>
            <td></td>
            </tr>
            <tr>
            <td rowspan="3">Audio-Visual Syn-</td>
            <td>Lip-sync accuracy Sound effect alignment Atmospheric matching</td>
            <td>accuracy Precise speech-mouth synchronization and correct speaker identification Temporal correspondence between visual events and sound effects Coherence between BGM, scene atmosphere, and emotional tone</td>
            </tr>
            <tr>
            <td></td>
            <td></td>
            </tr>
            <tr>
            <td></td>
            <td></td>
            </tr>
            <tr>
            <td rowspan="4">Visual Quality</td>
            <td>Visual clarity Color accuracy Compositional quality Structural integrity</td>
            <td>Sharpness, definition, and resolution Natural color balance and saturation without distortion Aesthetic composition, framing, and visual balance Absence of visual artifacts and corruptions</td>
            </tr>
            <tr>
            <td></td>
            <td></td>
            </tr>
            <tr>
            <td></td>
            <td></td>
            </tr>
            <tr>
            <td></td>
            <td></td>
            </tr>
            <tr>
            <td rowspan="5">Motion Quality</td>
            <td>Physical plausibility Motion fluidity Motion stability Temporal consistency Motion vividness</td>
            <td>Adherence to physical laws (gravity, inertia, momentum) Smooth transitions without abrupt discontinuities Absence of jittering, deformation, and flickering Consistency of dynamic elements across frames Action, camera, atmospheric, and emotional expressiveness</td>
            </tr>
            <tr>
            <td></td>
            <td></td>
            </tr>
            <tr>
            <td></td>
            <td></td>
            </tr>
            <tr>
            <td></td>
            <td></td>
            </tr>
            <tr>
            <td></td>
            <td></td>
            </tr>
            <tr>
            <td rowspan="5">Audio Quality</td>
            <td>Absence of artifacts Spatial soundstage Timbre realism Signal clarity Dynamic range</td>
            <td>No clipping, truncation, distortion, or glitches Appropriate stereo imaging and spatial rendering Natural and realistic tonal qualities Clean audio with appropriate signal-to-noise ratio Appropriate audio level variation without compression artifacts</td>
            </tr>
            <tr>
            <td></td>
            <td></td>
            </tr>
            <tr>
            <td></td>
            <td></td>
            </tr>
            <tr>
            <td></td>
            <td></td>
            </tr>
            <tr>
            <td></td>
            <td></td>
            </tr>
            </tbody>
            </table>

### 5.2.2. 评估方法
*   <strong>绝对评分 (Absolute Scoring):</strong> 评估者使用 5 点 Likert 量表（1 = 极不满意，2 = 不满意，3 = 中立，4 = 满意，5 = 极满意）对每个维度进行评分。
*   <strong>好-相同-差 (Good-Same-Bad, GSB) 比较:</strong> 对模型输出进行成对比较，评估者将一个模型的结果相对于另一个模型标记为“好”（明显更好）、“相同”（质量相当）或“差”（明显更差）。

## 5.3. 对比基线
在 `Artificial Analysis Arena` 和 `SkyReels-VABench` 人工评估中，SkyReels-V4 与以下最先进的视频-音频生成系统进行了比较：
*   Veo 3.1 (Google) [1]
*   Kling 2.6 (Kuaishou) [3]
*   Seedance 1.5 Pro (ByteDance) [5]
*   Wan 2.6 (Alibaba) [6]
*   Grok-imagine-video (未在参考文献中明确列出，但提及在竞技场中进行比较)
*   Sora-2 (OpenAI) [2]
*   Vidu-Q3 (Vidu) [19]

    这些基线模型代表了视频-音频生成领域的顶尖技术，包括了来自 Google、OpenAI、Kuaishou、ByteDance 和 Alibaba 等知名公司的专有系统，以及一些前沿的开源模型。选择这些模型作为基线能够全面评估 SkyReels-V4 在行业内的竞争力。

# 6. 实验结果与分析

## 6.1. 核心结果分析

### 6.1.1. `Artificial Analysis Arena` 结果
`Artificial Analysis Arena` [20] 是一个评估生成模型的公共基准平台，通过用户对生成内容的成对比较计算 Elo 分数。SkyReels-V4 在`Text-to-Video with Audio Arena` 赛道中取得了优异的成绩。

*   **排名:** 截至 2026 年 2 月 25 日，SkyReels-V4 在排行榜上排名**第二**，仅次于 Veo 3.1。这表明其在公共用户偏好评估中展现出强大且具有竞争力的音视频生成质量。

    以下是原文 Figure 3 的内容：

    ![Figure 3:Artificial Analysis Text-to-Video with Audio Arena Leaderboard. Our model ranks second among all competing baselines including Veo 3.1, grok-imagine-vide, Sora-2, Vidu-Q3, Wan 2.6 and etc.](images/3.jpg)
    *该图像是一个图表，展示了SkyReels V4在文本到视频生成的音频领域中的排名。该模型在众多竞争基线中排名第二，包括Veo 3.1等，ELO得分为1,090。*

图 3: `Artificial Analysis Text-to-Video with Audio Arena` 排行榜。我们的模型在所有竞争基线（包括 Veo 3.1、grok-imagine-video、Sora-2、Vidu-Q3、Wan 2.6 等）中排名第二。

### 6.1.2. 人工评估 (`SkyReels-VABench`) 结果

#### 6.1.2.1. 绝对评分结果
通过 5 点 Likert 量表对 SkyReels-V4 及其基线模型进行评估。
*   **整体表现:** SkyReels-V4 在所有竞争模型中取得了**最高的整体平均分**，表明其在综合性能上的优势。
*   **维度优势:**
    *   在<strong>指令遵循 (Prompt Following)</strong> 和<strong>运动质量 (Motion Quality)</strong> 方面表现尤为突出，得分显著高于其他模型。这表明模型对多模态指令的理解和执行能力很强，并且能生成逼真流畅的运动。
    *   在<strong>视觉质量 (Visual Quality)</strong> 方面，SkyReels-V4 与最强的竞争模型表现相当，维持了高水平的视觉保真度。
    *   在<strong>音视频同步 (Audio-Visual Synchronization)</strong> 和<strong>音频质量 (Audio Quality)</strong> 方面，虽然优势相对温和，但仍保持了最先进的性能，强调了其在整个评估范围内的竞争力。

        以下是原文 Figure 4 的内容：

        ![Figure 4:Absolute scoring results (5-point Likert scale comparing SkyReels V4 against baselines. Higher score indicate better performance.](images/4.jpg)
        *该图像是一个图表，展示了SkyReels V4与多个基线模型的评分结果（5点Likert量表），涵盖整体质量、指令遵循、视听同步、视觉质量、动作质量和音频质量等指标。图中较高的评分表明了SkyReels V4在这些方面的优越性能。*

图 4: 绝对评分结果（5 点 Likert 量表），比较 SkyReels V4 与基线模型。分数越高表示性能越好。

#### 6.1.2.2. 好-相同-差 (GSB) 比较结果
GSB 比较提供了更细粒度的质量洞察，通过成对比较 SkyReels-V4 与每个基线模型。
*   **整体 GSB 比较:**
    *   在整体质量比较中，SkyReels-V4 相对于所有基线模型都持续获得更高比例的“好”评价，同时“差”的比例较低。这进一步验证了 SkyReels-V4 的优越性。

        以下是原文 Figure 5 的内容：

        ![该图像是一个条形图，展示了SkyReels V4与其他模型（Kling 2.6、Veo 3.1、Seedance 1.5 Pro、Wan2.6）在总体质量上的偏好比较。数据表明，SkyReels V4在所有比较中均表现较好。](images/5.jpg)
        *该图像是一个条形图，展示了SkyReels V4与其他模型（Kling 2.6、Veo 3.1、Seedance 1.5 Pro、Wan2.6）在总体质量上的偏好比较。数据表明，SkyReels V4在所有比较中均表现较好。*

图 5: GSB 整体质量比较：SkyReels V4 对比基线模型。每个条形显示“好”和“差”评级的比例。

*   **分维度 GSB 比较:**
    *   **SkyReels V4 vs. Kling 2.6:** SkyReels V4 在所有五个评估维度上都显著优于 Kling 2.6，尤其在指令遵循和运动质量方面优势明显。
    *   **SkyReels V4 vs. Seedance 1.5 Pro:** SkyReels V4 在指令遵循、视觉质量和运动质量方面表现更佳，而在音视频同步和音频质量方面略有优势或持平。
    *   **SkyReels V4 vs. Veo 3.1:** SkyReels V4 在指令遵循和运动质量方面优于 Veo 3.1，在视觉质量上相当，并在音视频同步和音频质量方面略有改善。
    *   **SkyReels V4 vs. Wan 2.6:** SkyReels V4 在所有五个维度上均显著优于 Wan 2.6。

        这些结果表明，SkyReels-V4 持续优于大多数竞争基线系统，尤其在多模态指令理解和生成高质量运动视频方面表现卓越。

以下是原文 Figure 6、7、8、9 的内容：

![该图像是条形图，展示了 SkyReels V4 与竞争对手 Kling 2.6 在多个质量维度上的偏好比较，包括指令遵循、音视同步、视觉质量、动作质量和音频质量。各项指标上，SkyReels V4 的偏好较高，显示了其在多模态生成中的优势。](images/6.jpg)
*该图像是条形图，展示了 SkyReels V4 与竞争对手 Kling 2.6 在多个质量维度上的偏好比较，包括指令遵循、音视同步、视觉质量、动作质量和音频质量。各项指标上，SkyReels V4 的偏好较高，显示了其在多模态生成中的优势。*

图 6: (a) SkyReels V4 vs. Kling 2.6

![该图像是一个比较条形图，展示了SkyReels V4 与 Sedance 1.5 Pro 在多个维度上的偏好评估，包括指令跟随、音视频同步、视觉质量、运动质量和音频质量。图中绿色条表示SkyReels V4更受偏爱，橙色条表示竞争对手更受偏爱，灰色条表示两者偏好相同。](images/7.jpg)
*该图像是一个比较条形图，展示了SkyReels V4 与 Sedance 1.5 Pro 在多个维度上的偏好评估，包括指令跟随、音视频同步、视觉质量、运动质量和音频质量。图中绿色条表示SkyReels V4更受偏爱，橙色条表示竞争对手更受偏爱，灰色条表示两者偏好相同。*

图 7: (b) SkyReels V4 vs. Seedance 1.5 Pro

![该图像是一个对比图，展示了SkyReels V4与其竞争对手Veo 3.1在多项指标上的偏好，包括指令遵循、视听同步、视觉质量、运动质量和音频质量。通过这些比较，可以观察到SkyReels V4在多个方面的优势。](images/8.jpg)
*该图像是一个对比图，展示了SkyReels V4与其竞争对手Veo 3.1在多项指标上的偏好，包括指令遵循、视听同步、视觉质量、运动质量和音频质量。通过这些比较，可以观察到SkyReels V4在多个方面的优势。*

图 8: GSB 比较结果。上图：SkyReels V4 与所有基线模型的整体质量比较。下图：跨指令遵循、音视频同步、视觉质量、运动质量和音频质量五个评估维度的分维度 GSB 比较。

![该图像是一个图表，展示了SkyReels V4与Wan2.6在多个质量指标上的偏好比较。图表列出了指令遵循、视听同步、视觉质量、运动质量和音频质量五个维度，使用不同颜色表示SkyReels V4偏好、相同和竞争对手偏好。SkyReels V4在多个方面表现出更高的偏好。](images/9.jpg)
*该图像是一个图表，展示了SkyReels V4与Wan2.6在多个质量指标上的偏好比较。图表列出了指令遵循、视听同步、视觉质量、运动质量和音频质量五个维度，使用不同颜色表示SkyReels V4偏好、相同和竞争对手偏好。SkyReels V4在多个方面表现出更高的偏好。*

图 9: SkyReels V4 vs. Wan 2.6

## 6.2. 应用程序示例 (Appendix A)

论文的附录 A 展示了 SkyReels-V4 在视频生成、修复和编辑任务中的应用示例，进一步证明了其多样化的能力。

### 6.2.1. 参考生成 (Reference-based Generation)

*   <strong>多图像和音频参考生成 (Multiple Image and Audio Reference Generation):</strong>
    模型能够根据多个参考图像和音频输入生成连贯且与音频匹配的视频，即使在复杂的多角色场景中也能保持一致性。
    以下是原文 Figure 10 的内容：

    ![Figure 6: Example of multiple images and audios reference.](images/10.jpg)
    *该图像是一个示意图，展示了参考图像与音频的关系，以及生成的视频输出。上方是三个参考图像，每个图像下方对应一个音频波形，下方则显示输出的视频场景，展示了多角色互动的结果。*

    图 10: 多图像和音频参考示例。

*   <strong>图像参考和运动参考生成 (Image Reference and Motion Reference Generation):</strong>
    模型可以利用图像和视频/运动参考（如姿态序列、轨迹）来控制生成视频的动态特性。例如，将参考图像中的人物动画化为参考视频中的动作。
    以下是原文 Figure 12 的内容：

    ![Figure 7: Examples of motion transfer in video reference.](images/12.jpg)
    *该图像是示意图，展示了多种动态传输的示例，包括与音乐和表演相关的场景。图中涉及不同的角色和背景，呈现了多元的视觉效果。*

    图 12: 视频参考中的运动迁移示例。

### 6.2.2. 视频修复 (Video Inpainting)

*   <strong>主体/属性/背景修复 (Subject/Attribute/Background Inpainting):</strong>
    模型能够智能地修复视频区域中的主体、属性或背景。例如，将视频中掩码区域的主体替换为指定的对象（如鹿），或改变物体（如领带）的颜色，甚至替换整个背景。
    以下是原文 Figure 15 的内容：

    ![Figure 8: Examples of subject/attribute/background inpainting.](images/15.jpg)
    *该图像是关于主题/属性/背景的修复示例，共包含六个不同的图像展示，展示了不同场景下的修复效果，如自然风光与背景虚化处理。*

    图 15: 主体/属性/背景修复示例。

*   <strong>图像参考修复 (Image Reference Inpainting):</strong>
    模型可以利用参考图像进行修复，以保持风格一致性。例如，将参考图像中的人物添加到视频的指定区域。
    以下是原文 Figure 17 的内容：

    ![Figure 9: Examples of image reference inpainting.](images/17.jpg)
    *该图像是插图，展示了猫粮包装和一系列情景对话。图中包括人类角色在办公室环境中的互动，以及一只猫参与对话的场景，展示多种图像引用的效果。*

    图 17: 图像参考修复示例。

### 6.2.3. 视频编辑 (Video Editing)

*   <strong>局部编辑 (Local Editing):</strong>
    模型支持细粒度的局部视频编辑，包括主体、属性和元素编辑。
    *   <strong>水印/字幕/Logo 移除 (Watermark/Subtitle/Logo Removal):</strong> 智能识别并移除视频中的水印、字幕、Logo 等元素，同时保持内容连贯性和自然度。
    *   <strong>主体操作 (Subject Manipulation):</strong> 在保持时间一致性的前提下，添加、删除或修改视频中的主体。
    *   <strong>局部属性编辑 (Local Attribute Editing):</strong> 改变视频中特定元素的局部属性，如颜色、材质。
    *   <strong>背景编辑 (Background Editing):</strong> 在保留前景主体的情况下，修改背景元素。
        以下是原文 Figure 10 (中的部分示例，此处未复现完整 Figure 10，但原文的 Figure 10 包含了多个水印/字幕/Logo 移除的子图，故只引用了 Figure 10 的整体概念), Figure 21 (subject manipulation), Figure 24 (local attribute editing) 的内容：

        ![Figure 11: Examples of subject manipulation.](images/21.jpg)
        *该图像是示意图，展示了视频输入和输出的例子。上方展示了去除视频中蜜蜂的操作，显示出原始视频和处理后的结果。下方则是花朵上蜜蜂的放大视图，输入与输出对比，显示了去除蜜蜂后的变化。*

    图 11: 主体操作示例。

    ![Figure 12: Examples of local attribute editing.](images/24.jpg)
    *该图像是示意图，展示了多张图像中的局部属性编辑示例。每对图像展示了不同的编辑效果，表明在视觉内容生成中的应用。整体场景为夜晚街道，带有商业建筑背景。*

    图 12: 局部属性编辑示例。

*   <strong>全局编辑 (Global Editing):</strong>
    模型能够进行影响整个视频的全局编辑，如风格迁移和相机控制。
    *   <strong>风格迁移 (Style Transfer):</strong> 将视频转换为不同的艺术风格。
    *   <strong>相机控制 (Camera Control):</strong> 修改相机属性，包括拍摄角度、类型和位置。
    *   <strong>场景属性 (Scene Attributes):</strong> 编辑天气、灯光、色调和时间。
        以下是原文 Figure 26 (style transfer), Figure 27 (camera control), Figure 28 (global scene attributes) 的内容：

        ![Figure 13: Examples of style transfer.](images/26.jpg)
        *该图像是一个示意图，展示了输入视频和输出视频之间的风格转换。上方为输入视频，下方为输出视频，分别呈现了相似的动作或场景，但输出视频以乐高人物的形式重现这些动作。此图示范了风格迁移技术的应用。*

    图 13: 风格迁移示例。

    ![Figure 14: Examples of camera control.](images/27.jpg)
    *该图像是示意图，展示了一个人在不同阶段抛接一个黄色物体的动态过程，表现了手势与运动的变化。*

    图 14: 相机控制示例。

    ![Figure 15: Examples of global scene attributes.](images/28.jpg)
    *该图像是示意图，展示了视频处理的输入和输出示例。第一行呈现了日间和夜间场景的转换，第二行展示了输入视频和输出后景雪景效果的变化。指令为将视频 @video_1 转变为雪景效果。*

    图 15: 全局场景属性示例。

*   <strong>基于参考的编辑 (Reference-Based Editing):</strong>
    模型支持基于参考进行视频编辑，包括主体参考、背景参考、表情参考或视觉效果指导。
    *   <strong>主体参考与运动参考 (Subject Reference with Motion Reference):</strong> 结合参考图像中的主体和参考视频中的运动模式来生成视频。
    *   <strong>主体参考与表情参考 (Subject Reference with Expression Reference):</strong> 将参考视频中的面部表情迁移到参考图像中的主体上。
    *   <strong>背景参考与视频参考 (Background Reference with Video Reference):</strong> 将参考图像中的背景与参考视频中的内容或运动相结合。
    *   <strong>首帧与效果参考 (First-Frame + Effect Ref):</strong> 从参考图像开始生成视频，并从参考视频中迁移效果。
        以下是原文 Figure 29 (subject reference with motion reference), Figure 30 (subject reference with expression reference), Figure 31 (background reference with video reference), Figure 32 (first-frame reference with effect reference) 的内容：

        ![Figure 16: Example of subject reference with motion reference.](images/29.jpg)
        *该图像是一个示意图，展示了多种手势和动作，对于模型在视频音频生成和编辑中的应用具有参考意义。*

    图 16: 主体参考与运动参考示例。

    ![Figure 17: Example of subject reference with expression reference](images/30.jpg)
    *该图像是示意图，展示了不同姿态和表情的示例，可能用于说明情感表达的变化。图中主体分别处于多种视角，背景环境简单，以突出主体的表现力。*

    图 17: 主体参考与表情参考示例。

    ![Figure 18: Example of background reference with video reference.](images/31.jpg)
    *该图像是多个画面组成的插图，展示了一只猎犬在雪地和泥土中寻找食物的过程。图像中犬只的细节清晰，体现了它探寻的专注神态及周边环境的变化。*

    图 18: 背景参考与视频参考示例。

    ![Figure 19: Example of first-frame reference with effect reference.](images/32.jpg)
    *该图像是一个插图，展示了多个不同效果的服装设计样本。插图展示了服装在不同角度和样式下的细节，展现了现代时尚的多样性和艺术表现力。*

    图 19: 首帧参考与效果参考示例。

## 6.3. 消融实验/参数分析
论文中没有明确给出详细的消融实验或超参数分析。然而，训练策略部分（4.2.5）描述了模型训练的渐进式多阶段方法，这本身就隐含了对不同组件和参数（如分辨率、帧率、视频时长、数据类型和比例）的逐步优化和验证。例如：
*   从 `T2I` 到 `T2V` 的逐步引入，以及随后修复和多模态条件的添加，可以看作是模型能力逐步构建和验证的过程。
*   混合分辨率训练（Stage 4 和 Stage 5）展示了模型如何逐渐适应更高分辨率的生成，这暗示了对分辨率参数的探索和优化。
*   多模态条件预训练（Stage 6）中不同参考类型（图像/视频）的比例设置（20%）也是一种参数选择。
*   精炼器（Refiner）的引入和 `VSA` 的应用，是为了解决高分辨率和长时长视频生成的计算效率问题，这反映了对效率策略的考量和优化。

    尽管没有提供标准意义上的消融实验表格，但训练过程本身就体现了对模型组件和训练策略的迭代优化。

# 7. 总结与思考

## 7.1. 结论总结
SkyReels-V4 是一款开创性的统一多模态视频基础模型，成功地将视频-音频生成、修复和编辑功能整合到一个单一的框架中。该模型采用双流多模态扩散变换器 (MMDiT) 架构，通过共享的 MLLM 文本编码器和多层跨模态注意力机制，实现了视频和音频的同步生成以及对多模态指令的精准遵循。其创新的通道拼接修复框架，使得图像到视频、视频扩展和视频编辑等多种任务得以统一处理。为了应对高分辨率和长时长的计算挑战，模型引入了低分辨率全序列与高分辨率关键帧联合生成的效率策略，并辅以专门的超分辨率和帧插值模型。

实验结果表明，SkyReels-V4 在公共 `Artificial Analysis Arena` 排行榜上取得了第二名的优异成绩，并在自建的 `SkyReels-VABench` 人工评估基准测试中获得了最高的整体平均分，尤其在指令遵循和运动质量方面表现突出，显著优于现有的商业和开源基线模型。SkyReels-V4 的出现为多模态视频生成领域树立了新的基准，展示了电影级质量和速度下处理复杂创作任务的巨大潜力。

## 7.2. 局限性与未来工作
论文中未明确指出自身的局限性或未来工作方向。然而，从模型的描述和现有的挑战中可以推断出一些潜在的局限性和未来的研究方向：

**潜在局限性：**
*   **计算资源需求:** 尽管引入了效率策略，但训练和部署如此大规模、高分辨率、多模态的模型仍然需要庞大的计算资源。对于个人研究者或小型团队来说，这可能是一个难以逾越的门槛。
*   **数据依赖:** 模型的成功高度依赖于大规模、高质量和多样化的多模态数据集，尤其是许可数据。数据的获取、清洗和标注本身就是一项艰巨的任务，可能存在偏差。
*   **生成内容的精确控制:** 尽管模型支持丰富的多模态指令，但在某些极其复杂的场景下，精确到像素或帧级别的控制可能仍有挑战。例如，在长时间视频中保持特定角色的连续性和稳定性。
*   **实时性:** 论文强调了效率，但对于 15 秒 1080p 32FPS 视频的生成，其推理速度是否能达到实时或接近实时的交互性，仍需进一步验证。
*   **泛化能力:** 虽然使用了大量数据进行训练，但模型在处理训练数据分布之外的极端或新颖场景时，其泛化能力可能仍有限制。

**未来研究方向：**
*   **进一步提升效率:** 探索更先进的稀疏注意力机制、模型压缩技术或更优化的推理策略，以降低计算成本，提高生成速度。
*   **更细粒度的控制接口:** 开发更直观、更强大的用户接口，允许创作者对生成过程中的特定元素进行更精细的控制，例如情感表达、物理交互的精确模拟等。
*   **长视频生成:** 进一步扩展模型处理更长时间视频的能力，同时保持高连贯性和一致性。
*   **交互式生成与编辑:** 探索模型的实时交互能力，实现用户在生成过程中进行实时修改和指导。
*   **道德和偏见研究:** 随着生成模型能力的增强，对生成内容中潜在的偏见、误导性或不当内容的检测和缓解将变得越来越重要。
*   **多语言和跨文化支持:** 进一步提升模型在不同语言和文化背景下的理解和生成能力。

## 7.3. 个人启发与批判

**个人启发：**
*   **统一性是趋势:** SkyReels-V4 再次印证了构建统一的基础模型是多模态生成领域的重要趋势。将生成、修复和编辑等多个任务整合到单一架构中，极大地提高了模型的实用性和泛用性，降低了复杂内容创作的门槛。
*   **多模态指令的重要性:** MLLM 作为共享文本编码器的作用至关重要，它使得模型能够理解和整合来自多种模态的复杂指令，这是实现高级内容创作的关键。
*   **效率与质量的平衡:** 在追求高分辨率和长时长视频生成的同时，通过创新的效率策略（如低分辨率序列+高分辨率关键帧生成、`VSA`）来平衡计算成本，这对于实际应用是至关重要的。
*   **分阶段训练的有效性:** 渐进式多阶段训练范式，从简单任务（`T2I`）到复杂任务（多模态音视频生成、修复和编辑），是一种非常有效的模型能力构建策略，值得借鉴。
*   <strong>精炼器（Refiner）的重要性:</strong> 针对高分辨率和长视频的质量提升，引入专门的后处理精炼器，是一个非常实用的工程化方案，可以有效弥补基础模型在直接生成高保真细节方面的不足。

**批判：**
*   **缺乏详细的消融实验:** 尽管训练策略中包含了逐步引入能力的描述，但如果能提供更正式的消融实验，量化每个关键组件（如双流 MMDiT、交叉注意力机制、通道拼接修复框架、效率策略）对最终性能的具体贡献，将使论文的说服力更强，也更有助于社区理解模型设计背后的权衡。
*   **对数据质量和多样性的依赖缺乏深入讨论:** 论文强调了高质量、多样化数据的收集和处理，但对于不同数据来源的贡献度、合成数据的具体生成方法细节、以及数据偏差可能带来的影响等方面的深入分析较少。在构建如此庞大的数据集时，这些都是关键的考虑因素。
*   <strong>“电影级”</strong>质量的客观评估: 尽管有人工评估和竞技场排名，但“电影级”质量是一个主观性较强的概念。除了目前的评估指标，是否可以引入更多的客观指标（如 `FID`、`IS` 等，尽管这些指标在视频领域也面临挑战）或更专业的电影行业专家评估，来进一步量化和验证“电影级”的说法。
*   **未来扩展性的明确性:** 尽管模型能力强大，但对于如何进一步扩展到更长的视频、更高分辨率、更复杂的物理交互或更精细的角色控制，论文可以给出更具体的展望和技术路线图。
*   **资源的透明度:** 作为一个基础模型，其所需的训练计算资源（GPU 类型、数量、训练时间）如果能更透明地披露，将有助于社区理解其规模和复现难度。