# 1. 论文基本信息

## 1.1. 标题
<strong>Phenaki: 从开放域文本描述生成可变长度视频 (Phenaki: Variable Length Video Generation From Open Domain Textual Description)</strong>

论文标题明确指出了研究的核心主题：一个名为 `Phenaki` 的模型，其主要功能是根据开放领域的文本描述生成视频，并且能够处理<strong>可变长度 (variable length)</strong> 的视频。这暗示了模型在时间和内容上的高度灵活性。

## 1.2. 作者
论文作者团队来自 **Google Brain**、<strong>密歇根大学 (University of Michigan)</strong> 和 <strong>伦敦大学学院 (University College London)</strong>。Google Brain 是谷歌内部专注于深度学习和人工智能的研究团队，在生成模型、Transformer 架构等领域有着深厚的技术积累，这为本研究的质量和创新性提供了信誉背书。

## 1.3. 发表期刊/会议
该论文以预印本 (pre-print) 的形式发布在 **arXiv** 上。arXiv 是一个开放获取的学术论文发布平台，允许研究者在同行评审前分享他们的研究成果。这使得最新的研究能够被快速传播，但需要注意的是，预印本论文尚未经过正式的同行评审流程。

## 1.4. 发表年份
2022年10月5日。

## 1.5. 摘要
我们提出了 Phenaki，一个能够根据一系列文本提示 (textual prompts) 合成逼真视频的模型。从文本生成视频尤其具有挑战性，原因在于计算成本高、高质量文本-视频数据量有限以及视频长度可变。为了解决这些问题，我们引入了一种新的学习视频表示的模型，该模型将视频压缩为离散词元 (discrete tokens) 的小型表示。这个词元化器 (tokenizer) 在时间维度上使用因果注意力 (causal attention)，使其能够处理可变长度的视频。为了从文本生成视频词元，我们使用一个双向掩码 Transformer (bidirectional masked transformer)，并以预先计算好的文本词元为条件。生成的视频词元随后被反词元化 (de-tokenized) 以创建实际的视频。为了解决数据问题，我们展示了如何在大量的图像-文本对语料库以及较小数量的视频-文本样本上进行联合训练，从而实现超越视频数据集中可用内容的泛化能力。与以往的视频生成方法相比，Phenaki 能够根据一系列提示（即随时间变化的文本或一个故事）在开放领域中生成任意长度的视频。据我们所知，这是首次有论文研究从随时间变化的提示中生成视频。此外，与逐帧 (per-frame) 的基线模型相比，我们提出的视频编码器-解码器为每个视频计算的词元更少，但能产生更好的时空一致性。

## 1.6. 原文链接
- <strong>官方来源 (arXiv):</strong> https://arxiv.org/abs/2210.02399
- **PDF 链接:** https://arxiv.org/pdf/2210.02399v1.pdf
- **发布状态:** 预印本 (Pre-print)

  ---

# 2. 整体概括

## 2.1. 研究背景与动机
### 2.1.1. 论文试图解决的核心问题是什么？
论文旨在解决从文本描述生成高质量、长时程、内容连贯的视频这一核心问题。尽管文本生成图像技术已取得巨大成功，但将其扩展到视频领域面临三大主要障碍：
1.  **高昂的计算成本：** 视频包含大量帧，直接处理原始像素数据的计算量远超图像生成。
2.  **数据稀缺与质量问题：** 与动辄数十亿级别的图像-文本数据集（如 LAION-5B）相比，高质量、大规模的视频-文本数据集非常有限，这限制了模型学习复杂动态和概念的能力。
3.  **视频长度的可变性：** 现实世界中的视频长度各不相同，而许多现有模型只能生成固定长度的视频片段，缺乏灵活性。

### 2.1.2. 为什么这个问题在当前领域是重要的？现有研究存在哪些具体的挑战或空白（Gap）？
这个问题的重要性在于，它代表了内容创作领域的下一个前沿。能够根据故事或脚本自动生成视频，将极大地赋能艺术、设计、娱乐和教育等行业。

现有研究主要存在以下空白：
*   **缺乏时间连贯性：** 许多方法将视频视为独立图像序列，逐帧生成。这虽然能处理可变长度，但忽略了帧间的时间依赖性，导致生成的视频缺乏动态连贯性，出现闪烁、抖动等问题。
*   **固定长度限制：** 一些模型通过联合编码时空信息来提升连贯性，但其架构设计通常要求输入和输出的视频长度是固定的，无法生成任意时长的视频。
*   **单一提示的局限性：** 现有工作大多基于单个、简短的文本提示生成视频。然而，一个复杂的视频故事无法用一句话描述完整。<strong>如何根据一个随时间变化的文本序列（即一个“故事”）来生成对应的长视频，是一个从未被探索过的领域空白</strong>。

### 2.1.3. 这篇论文的切入点或创新思路是什么？
Phenaki 的创新思路是“**分而治之，联合学习**”，并引入了<strong>故事驱动 (story-driven)</strong> 的生成范式。
1.  **分而治之：** 将复杂的视频生成任务分解为两个阶段。
    *   <strong>阶段一（视频压缩）：</strong> 设计一个新颖的视频编码器-解码器 `C-ViViT`，它能高效地将视频压缩成一小组离散的词元 (tokens)。关键在于，`C-ViViT` 在时间维度上是<strong>因果的 (causal)</strong> 和<strong>自回归的 (auto-regressive)</strong>，这使其天然支持可变长度的视频处理。
    *   <strong>阶段二（文本到视频词元生成）：</strong> 使用一个高效的<strong>双向掩码 Transformer (bidirectional masked transformer)</strong>，学习从文本描述到压缩视频词元的映射。
2.  **联合学习：** 为了克服视频数据稀缺的问题，Phenaki 在训练时巧妙地**混合了大量的图像-文本数据和少量的视频-文本数据**。这使得模型能从图像数据中学到丰富的视觉概念（如“铅笔画风格”），并将其泛化应用到视频的动态生成中。
3.  **故事驱动生成：** 利用 `C-ViViT` 的自回归特性，Phenaki 能够生成一段视频后，以前一段的结尾为基础，结合**新的文本提示**继续生成下一段视频，从而实现根据一个完整的故事线（一系列变化的提示）生成一个连贯的长视频。这是本文最突出的创新点。

## 2.2. 核心贡献/主要发现
1.  **提出了 Phenaki 模型：** 一个能够从开放域文本生成高质量、任意长度视频的系统。
2.  **设计了 C-ViViT 架构：** 一种新颖的因果视频编码器-解码器，它能有效压缩视频，同时其**时间上的因果结构**是实现可变长度视频生成的关键。与逐帧方法相比，它使用更少的词元，却实现了更好的时空一致性。
3.  **验证了图文与视频数据联合训练的有效性：** 证明了结合大规模图像-文本数据集进行训练，可以让视频生成模型学习到视频数据集中不存在的视觉概念和风格，显著提升了模型的泛化能力和创造力。
4.  **首次实现了故事驱动的视频生成：** 论文首次展示了根据<strong>随时间变化的文本提示序列（一个故事）</strong> 生成连贯长视频的能力。模型可以在保持主体一致性的同时，根据新提示平滑地过渡场景和动作，如下图所示，泰迪熊根据提示自然地“变形”为熊猫。

    下图（原文 Figure 1）展示了这种故事驱动的生成能力。视频从第一个提示开始生成，几帧后切换到下一个提示，模型能够保持视频的连贯性，同时适应新的描述。

    ![Figure 1. Time variable text (i.e. story) conditional video generation. The entire figure is one continuous video generated auto-regressively. We start by generating the video conditioned on the first prompt and then after a couple of frames we change the prompt to the next one. Each row contains a selected number of frames (from left to right in order) while the model was conditioned on that particular prompt. The model manages to preserve the temporal coherence of the video while adapting to the new prompt, usually taking the shortest path for the adaption (notice the morphing of the teddy bear to the panda). Please note that the generated video has complex visual features such as reflections, occlusions, interactions and scene transitions. Full video is available at phenaki.github.io.](images/1.jpg)
    *该图像是示意图，展示了一个变换的故事情节，其中玩具熊在水中游动，随着不同提示生成了不同的画面。第一行为提示"玩具熊在水下"，逐渐转变为"熊猫在水中游泳"，展现了平滑的视觉过渡和时序一致性。*

---

# 3. 预备知识与相关工作

## 3.1. 基础概念
### 3.1.1. Transformer 与自注意力机制 (Self-Attention)
Transformer 是一种最初用于自然语言处理的深度学习模型架构，现已广泛应用于计算机视觉等领域。其核心是<strong>自注意力机制 (Self-Attention)</strong>，它允许模型在处理一个序列（如一句话或一系列图像块）时，为序列中的每个元素计算一个加权表示，这个权重取决于该元素与序列中所有其他元素的关联程度。这使得模型能够捕捉长距离依赖关系。

自注意力机制的计算过程如下：对于输入序列中的每个元素，我们创建三个向量：查询 (Query, Q)、键 (Key, K) 和值 (Value, V)。一个元素对另一个元素的注意力分数是通过计算前者的 Q 向量与后者的 K 向量的点积得到的。这些分数经过缩放和 `softmax` 归一化后，用于对所有 V 向量进行加权求和，得到该元素的最终输出。

<strong>核心公式 (Scaled Dot-Product Attention):</strong>
$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
**符号解释:**
*   $Q$: 查询矩阵，代表当前元素，用于发起查询。
*   $K$: 键矩阵，代表序列中所有元素，用于被查询。
*   $V$: 值矩阵，代表序列中所有元素的实际信息。
*   $d_k$: 键向量的维度。除以 $\sqrt{d_k}$ 是一个缩放因子，用于防止点积结果过大导致梯度消失。
*   $\mathrm{softmax}$: 归一化函数，将注意力分数转换为总和为1的权重。

### 3.1.2. 向量量化 (Vector Quantization, VQ) 与 VQ-VAE/VQ-GAN
<strong>向量量化 (VQ)</strong> 是一种数据压缩技术。它将一个连续的高维向量空间映射到一个离散的、有限的码本 (codebook) 中。具体来说，对于任意一个输入向量，VQ 会在码本中找到与之最接近的码向量 (code vector) 来替代它。

**VQ-VAE (Vector Quantised-Variational Autoencoder)** 是一种结合了 VQ 和自编码器 (Autoencoder) 的生成模型。
*   <strong>编码器 (Encoder):</strong> 将输入数据（如图像）编码成一个连续的潜在向量。
*   **量化层:** 使用 VQ 技术，将这个连续的潜在向量替换为码本中最接近的离散码向量。
*   <strong>解码器 (Decoder):</strong> 将离散的码向量重建成原始数据。
    这样做的好处是，模型学习到了一个离散的、结构化的潜在空间，非常适合后续使用 Transformer 等序列模型进行建模。

**VQ-GAN (Vector Quantised-Generative Adversarial Network)** 进一步将 VQ-VAE 与生成对抗网络 (GAN) 结合。它使用一个判别器 (Discriminator) 来判断解码器生成的图像是否真实，通过对抗训练来提升生成图像的真实感和细节。Phenaki 中的 `C-ViViT` 就采用了类似 VQ-GAN 的思想来学习视频的离散表示。

### 3.1.3. 双向掩码 Transformer (Bidirectional Masked Transformer)
这是一种受到 BERT 和 MaskGIT 启发的 Transformer 模型。与传统的自回归 (auto-regressive) 模型（从左到右逐个生成词元）不同，双向掩码 Transformer 可以在一次前向传播中<strong>并行预测所有被掩盖 (masked) 的词元</strong>。
*   **训练:** 随机“掩盖”掉输入序列的一部分词元，然后让模型根据未被掩盖的上下文来预测这些被掩盖的词元。
*   **推理/生成:** 从一个完全被掩盖的序列开始，模型在多个步骤中迭代地生成所有词元。在每一步，模型都会预测所有当前被掩盖的词元，然后保留其中一部分置信度最高的预测，将其他的重新掩盖，进入下一步，直到所有词元都被生成。这种并行预测的方式大大加快了生成速度。

## 3.2. 前人工作
*   **自回归文本到图像/视频模型:**
    *   `DALL-E` 和 `Parti`: 这些模型将图像压缩为离散词元，然后使用一个大型自回归 Transformer 从文本生成这些图像词元。它们证明了 Transformer 在高质量生成任务上的强大能力。
    *   `GODIVA`, `NUWA`, `CogVideo`: 这些工作将自回归思想扩展到视频生成。它们通常将视频视为一系列独立的图像，使用逐帧的图像编码器（如 VQ-VAE）获取词元，然后用 Transformer 生成视频词元序列。
*   <strong>视频扩散模型 (Video Diffusion Models, VDM):</strong>
    *   `VDM` 及其变体：这类模型通过一个去噪过程直接在像素空间或潜在空间中生成视频。它们通常使用 3D U-Net 架构来同时处理时空信息，能够生成高质量的视频片段，但通常受限于固定长度，且采样速度较慢。

## 3.3. 技术演进
文本到内容生成技术的发展脉络如下：
1.  **文本到图像：** 早期的 GANs 和 VAEs 奠定了基础。随后，`DALL-E`, `Imagen`, `Stable Diffusion` 等大规模模型将生成图像的质量和可控性推向了新的高度，主要分为自回归、扩散和混合等技术路线。
2.  **文本到视频：** 这是更具挑战性的下一步。早期尝试通常是逐帧生成，时序一致性差。后续工作如 `NUWA` 和 `CogVideo` 采用了更强的自回归模型，但仍将视频视为图像序列。`VDM` 等工作则尝试在时空维度上联合建模，但灵活性不足。
3.  **Phenaki 的位置：** Phenaki 处在一个承前启后的关键位置。它吸取了自回归模型（如 `Parti`）的思想，但通过设计专用的视频词元化器 `C-ViViT` 解决了逐帧方法缺乏时间连贯性的问题。同时，它通过因果注意力和自回归生成机制，解决了固定长度模型（如一些 `VDM`）的灵活性问题，并开创性地提出了“故事驱动”的生成模式。

## 3.4. 差异化分析
*   <strong>与逐帧自回归模型 (`NUWA`, `CogVideo`) 的区别：</strong>
    *   **视频表示：** 逐帧模型将视频视为独立的图像序列，其视频词元在时间上是冗余的。Phenaki 的 `C-ViViT` 显式地建模了时空关系，通过时空块 (spatio-temporal patches) 和时间维度的 Transformer 来压缩冗余，生成更紧凑、更具时间信息的视频词元。
    *   **时间连贯性：** `C-ViViT` 的架构设计天然地促进了时间连贯性，而逐帧模型需要额外的机制（如 `NUWA-Infinity` 中解码时参考前一帧）来弥补这一缺陷。
*   <strong>与固定长度模型 (`VDM`) 的区别：</strong>
    *   **灵活性：** `VDM` 等模型的架构（如 3D U-Net）通常要求固定的输入/输出尺寸。Phenaki 的 `C-ViViT` 由于在时间维度上采用了**因果注意力**，使其可以处理任意数量的帧，从而能够生成可变长度的视频。
    *   **长视频生成：** Phenaki 可以通过自回归的方式不断“接续”已生成的视频，实现任意时长的生成。而 `VDM` 要生成长视频则需要复杂的扩展机制，计算成本极高。
*   **核心创新点总结：**
    1.  <strong>可变长度视频词元化器 (`C-ViViT`)</strong>: 解决了效率和时间连贯性的权衡问题。
    2.  **图文-视频联合训练**: 解决了视频数据稀缺和概念覆盖不足的问题。
    3.  <strong>时序可变提示 (故事生成)</strong>: 解决了单一提示无法描述复杂动态视频的问题，是全新的应用范式。

        ---

# 4. 方法论

Phenaki 的整体架构如下图（原文 Figure 2）所示，主要包含两个核心组件：
1.  **C-ViViT 视频编码器-解码器：** 负责将视频高效地压缩成离散的词元序列，以及将词元序列重建为视频。
2.  <strong>双向掩码 Transformer (MaskGIT)：</strong> 负责根据文本提示生成这些视频词元。

    ![Figure 2. The architecture of Phenaki. Left: C-ViViT encoder architecture. The embeddings of images and video patches from raw frames x are processed by a spatial and then a causal transformer (auto-regressive in time) to generate video tokens z. Center: MaskGiT is trained to reconstruct masked tokens $\\mathbf { z }$ predicted by a frozen C-ViViT encoder and conditioned on T5X tokens of a given prompt $\\mathbf { p } _ { 0 }$ . Right: How Phenaki can generate arbitrary long videos by freezing the past token and generating the future tokens. The prompt can change over time to enable time-variable prompt (i.e. story) conditional generation. The subscripts represent time (i.e. frame number).](images/2.jpg)
    *该图像是Phenaki架构的示意图，展示了C-ViViT编码器、训练变换器和视频生成模块的结构。其中，编码器通过空间和因果变换器生成视频令牌 $z$，训练变换器利用随机掩码重建掩码令牌，而视频生成模块则通过固定过去令牌生成未来令牌，支持基于时间变化的提示。图中包含了多个操作和令牌的状态。*

## 4.1. C-ViViT: 因果 ViViT 视频编码器-解码器
`C-ViViT` 的目标是创建一个既能高效压缩视频（利用时空冗余），又能处理可变长度视频的表示模型。

### 4.1.1. 编码器架构 (Encoder Architecture)
`C-ViViT` 的编码过程巧妙地分离了第一帧和后续帧的处理，并按“先空间，后时间”的顺序进行信息整合。
1.  <strong>输入与分块 (Input and Patching):</strong>
    *   输入是一个视频序列 $\mathbf { x } \in \mathbb { R } ^ { ( t _ { x } + 1 ) \times h _ { x } \times w _ { x } \times c _ { x } }$，包含 $t_x+1$ 帧。
    *   **第一帧**被特殊处理：它被切分成多个不重叠的二维图像块 (image patches)，大小为 $w_p \times h_p$。
    *   **后续 $t_x$ 帧**被切分成多个不重叠的三维时空块 (spatio-temporal patches)，大小为 $t_p \times w_p \times h_p$。
    *   这种设计使得第一帧可以独立编码，为后续的图文联合训练（将单张图片视为视频的第一帧）和以图生视频任务提供了便利。

2.  <strong>线性投影 (Linear Projection):</strong>
    *   每个块（无论是二维还是三维）都被展平 (flatten) 并线性投影到一个 $d_z$ 维的嵌入空间中，形成一个形状为 $(t_z+1) \times (w_z \cdot h_z) \times d_z$ 的张量，其中 $t_z = t_x/t_p$, $w_z = w_x/w_p$, $h_z = h_x/h_p$。

3.  <strong>空间 Transformer (Spatial Transformer):</strong>
    *   对每个时间步（即每一帧对应的块集合）独立应用多层 Transformer。注意力机制在**空间维度**上进行，即每个块会关注同一时间步内的所有其他块。这用于学习每帧内部的空间结构。

4.  <strong>因果时间 Transformer (Causal Temporal Transformer):</strong>
    *   在空间 Transformer 的输出上，再应用多层 Transformer。这次，注意力机制在**时间维度**上进行，但使用的是<strong>因果注意力 (causal attention)</strong>。
    *   **因果注意力**意味着，在处理第 $t$ 个时间步的块时，它只能关注来自时间步 `0, 1, ..., t` 的块，而不能看到未来的信息 ($t+1, ...$)。
    *   **这是实现可变长度处理和自回归生成的关键**。因为无论视频有多长，每个时间步的编码都只依赖于其历史信息，模型可以一帧一帧地向前处理，而无需预知总长度。

5.  <strong>向量量化 (Vector Quantization):</strong>
    *   时间 Transformer 的输出 $\mathbf{z}$ 最终通过向量量化层，被映射到码本 $\mathbf{E}$ 中最接近的离散码向量 $\mathbf{e}$。这样，整个视频就被压缩成了一个离散的词元序列。

### 4.1.2. 解码器架构 (Decoder Architecture)
解码器是编码器的逆过程，结构对称：
1.  将输入的离散视频词元转换为嵌入向量。
2.  通过时间 Transformer（同样可以是因果的，或者在生成时是全注意力的）。
3.  通过空间 Transformer。
4.  通过一个线性投影层将输出的嵌入向量映射回像素空间，重建视频帧。

### 4.1.3. 量化与损失函数
为了训练 `C-ViViT`，模型使用了多种损失函数的组合。
*   <strong>VQ 损失 (VQ Loss):</strong> 用于学习码本和确保编码器输出能够很好地被量化。
    $$
    L _ { V Q } = \lVert \mathbf { s g ( z ) } - \mathbf { e } \rVert _ { 2 } ^ { 2 } + \beta \lVert \mathbf { z } - s g ( \mathbf { e } ) \rVert _ { 2 } ^ { 2 }
    $$
    **符号解释:**
    *   $\mathbf{z}$: 编码器输出的连续向量。
    *   $\mathbf{e}$: 码本中与 $\mathbf{z}$ 最接近的码向量。
    *   $\mathrm{sg}(\cdot)$: 停止梯度 (stop-gradient) 操作，意味着在反向传播时该路径的梯度为0。
    *   第一项（**码本损失**）：更新码向量 $\mathbf{e}$，使其向编码器输出 $\mathbf{z}$ 靠拢。
    *   第二项（**承诺损失, commitment loss**）：更新编码器，使其输出 $\mathbf{z}$ 向选定的码向量 $\mathbf{e}$ 靠拢。
    *   $\beta$: 承诺损失的权重超参数。

*   <strong>总损失函数 (Total Loss Function):</strong>
    $$
    L = L _ { V Q } + 0 . 1 \times L _ { A d \nu } + 0 . 1 \times L _ { I P } + 1 . 0 \times L _ { V P } + 1 . 0 \times L _ { 2 }
    $$
    **符号解释:**
    *   $L_{VQ}$: 上述的向量量化损失。
    *   $L_{Ad\nu}$: <strong>对抗性损失 (Adversarial Loss)</strong>，使用一个判别器（如 StyleGAN 的判别器）来提高重建视频的真实感。
    *   $L_{IP}$: <strong>图像感知损失 (Image Perceptual Loss)</strong>，计算重建帧和原始帧在预训练网络（如 VGG）的特征空间中的差异，以保证视觉质量。
    *   $L_{VP}$: <strong>视频感知损失 (Video Perceptual Loss)</strong>，类似于 $L_{IP}$，但使用预训练的视频网络（如 I3D）作为特征提取器，以保证动态和运动的真实性。
    *   $L_2$: <strong>重建损失 (Reconstruction Loss)</strong>，即重建视频和原始视频之间的像素级 L2 距离。

## 4.2. 文本到视频生成 (Text-to-Video Generation)
在 `C-ViViT` 训练完成后，它作为一个固定的视频词元化器。第二阶段的任务是训练一个模型，根据文本生成对应的视频词元序列。

### 4.2.1. 双向掩码 Transformer (Masked Bidirectional Transformer)
Phenaki 采用了一种基于 MaskGIT 的高效生成策略，而非传统的逐个生成词元的自回归方法。
*   **训练过程:**
    1.  将一个视频通过预训练好的 `C-ViViT` 编码器，得到其真实的视频词元序列 $\mathbf{a}$。
    2.  随机选择一个掩码率 $\gamma_i$，并随机地将序列 $\mathbf{a}$ 中 $\lceil \gamma_i \cdot N \rceil$ 个词元替换为特殊的 `[MASK]` 词元，得到掩码后的序列 $\mathbf{a}_{\bar{M}}$。
    3.  模型的目标是根据文本嵌入 $\mathbf{p}$ 和未被掩码的视频词元，预测出所有被掩码位置的原始词元。
    4.  损失函数为掩码位置上的<strong>交叉熵损失 (cross-entropy loss)</strong>。
        $$
    L _ { \mathrm { mask } } = - \sum _ { \forall i \in [ 1 , N ] , m _ { i } = 1 } \log p ( a _ { i } | \mathbf { a } _ { \bar { M } } , \mathbf { p } )
    $$
    **符号解释:**
    *   $\mathbf{a}$: 原始视频词元序列。
    *   $\mathbf{a}_{\bar{M}}$: 掩码后的视频词元序列。
    *   $m_i=1$ 表示第 $i$ 个词元被掩码。
    *   $N$: 视频词元序列的总长度。
    *   $\mathbf{p}$: 来自预训练语言模型（如 T5X）的文本嵌入。
    *   $p(\cdot)$: 模型预测的概率分布。

*   **图文-视频联合训练策略:**
    *   为了同时利用图像和视频数据，训练时会动态调整损失计算的范围。
    *   如果输入的是**视频**，损失函数 $L_{mask}$ 会在所有视频词元上计算。
    *   如果输入的是**单张图片**（被视为只有一帧的视频），损失函数只在代表第一帧的词元上计算。这使得模型可以无缝地处理两种数据类型。

*   <strong>分类器无关指导 (Classifier-Free Guidance):</strong>
    *   训练时，有 $10\%$ 的概率会随机丢弃文本条件 $\mathbf{p}$，让模型学会在无条件的情况下生成视频词元。这使得在推理时，可以通过调整指导尺度 (guidance scale) $\lambda$ 来控制生成内容与文本的匹配程度。

### 4.2.2. 推理与长视频生成
*   **单视频生成:**
    1.  从一个所有词元都被 `[MASK]` 覆盖的序列开始。
    2.  在多个迭代步骤中生成视频。在第 $i$ 步：
        *   模型并行预测所有被掩码的词元。
        *   根据一个预设的采样策略，保留其中置信度最高的 $\beta_i$ 比例的预测结果。
        *   将其余未保留的词元重新设置为 `[MASK]`。
    3.  重复此过程，直到所有词元都被生成。通常只需要 12 到 48 个步骤，远快于自回归模型。

*   <strong>长视频与故事生成 (Auto-regressive Generation of Long Videos):</strong>
    1.  **第一段视频：** 根据第一个文本提示生成一段视频。
    2.  **衔接：** 取出第一段视频的最后 $K$ 帧，用 `C-ViViT` 编码器将其转换为视频词元。
    3.  **第二段视频：** 将这些词元作为“历史上下文”或“初始帧”，输入到 MaskGIT 中，并提供**第二个文本提示**。然后，MaskGIT 会接着这些词元继续生成新的视频词元。
    4.  重复此过程，即可根据一系列随时间变化的提示（一个故事）生成一个连贯的长视频。`C-ViViT` 的因果自回归特性确保了这种衔接的自然和可行性。

        ---

# 5. 实验设置

## 5.1. 数据集
Phenaki 的训练和评估使用了多个不同类型和规模的数据集：
*   **内部文本-视频数据集:** 约 1500 万个文本-视频对，以 8 FPS (每秒8帧) 的帧率进行处理。
*   **内部文本-图像数据集:** 约 5000 万个文本-图像对。
*   **LAION-400M:** 一个公开的大规模图像-文本数据集，包含约 4 亿个样本。用于增强模型对广泛视觉概念的理解。
*   **Kinetics-400 & Kinetics-600:** 广泛使用的动作识别视频数据集。Phenaki 在这些数据集上进行<strong>零样本 (zero-shot)</strong> 评估，即模型在训练时未见过这些数据，直接测试其生成能力。
*   **Moments-in-Time (MiT):** 一个高质量、类别均衡的视频数据集，包含约 100 万个短视频，用于评估 `C-ViViT` 的视频量化和重建性能。
*   **BAIR Robot Pushing:** 一个机器人推动物体的视频数据集，用于视频预测 (video prediction) 任务的基准测试。

## 5.2. 评估指标
### 5.2.1. Fréchet Inception Distance (FID)
1.  <strong>概念定义 (Conceptual Definition):</strong> FID 是一种用于评估生成模型（特别是图像生成模型）性能的指标。它通过比较生成样本的特征分布与真实样本的特征分布之间的距离来衡量生成图像的**质量**和**多样性**。FID 分数越低，表示生成图像的分布与真实图像的分布越接近，即生成质量越高。它对模式崩溃 (mode collapse) 和多样性不足等问题很敏感。
2.  <strong>数学公式 (Mathematical Formula):</strong>
    $$
    \mathrm{FID}(x, g) = \left\| \mu_x - \mu_g \right\|_2^2 + \mathrm{Tr}\left( \Sigma_x + \Sigma_g - 2(\Sigma_x \Sigma_g)^{1/2} \right)
    $$
3.  **符号解释:**
    *   $x$ 和 $g$ 分别代表真实图像分布和生成图像分布。
    *   $\mu_x$ 和 $\mu_g$ 是真实图像和生成图像在 Inception-v3 网络某一层的激活特征的均值向量。
    *   $\Sigma_x$ 和 $\Sigma_g$ 是这些激活特征的协方差矩阵。
    *   $\|\cdot\|_2^2$ 表示向量的 L2 范数的平方。
    *   $\mathrm{Tr}(\cdot)$ 表示矩阵的迹 (trace)，即对角线元素之和。

### 5.2.2. Fréchet Video Distance (FVD)
1.  <strong>概念定义 (Conceptual Definition):</strong> FVD 是 FID 在视频领域的扩展，专门用于评估生成视频的质量。与 FID 只关注单帧图像质量不同，FVD 评估的是视频的<strong>时空特征 (spatio-temporal features)</strong>。它同时考量了视频的**视觉保真度**（每帧的清晰度和真实性）和**时间连贯性**（运动的流畅性和逻辑性）。FVD 分数越低，表示生成视频在内容和动态上越接近真实视频。
2.  <strong>数学公式 (Mathematical Formula):</strong> FVD 的计算公式与 FID 形式上完全相同，但其特征提取器不同。
    $$
    \mathrm{FVD}(x, g) = \left\| \mu_x - \mu_g \right\|_2^2 + \mathrm{Tr}\left( \Sigma_x + \Sigma_g - 2(\Sigma_x \Sigma_g)^{1/2} \right)
    $$
3.  **符号解释:**
    *   $x$ 和 $g$ 分别代表真实视频分布和生成视频分布。
    *   特征提取器是一个在动作识别任务上预训练的 3D 卷积网络（如 I3D）。
    *   $\mu_x, \mu_g, \Sigma_x, \Sigma_g$ 的含义与 FID 类似，但它们是基于从视频中提取的**时空特征**计算得出的均值和协方差。

### 5.2.3. CLIP Score
1.  <strong>概念定义 (Conceptual Definition):</strong> CLIP Score 用于衡量生成的图像/视频与给定的文本描述之间的**语义一致性**。它利用了 OpenAI 的 CLIP (Contrastive Language-Image Pre-Training) 模型，该模型能够理解图像和文本之间的关联。CLIP Score 越高，表示生成内容与文本提示的匹配度越好。
2.  <strong>数学公式 (Mathematical Formula):</strong>
    $$
    \text{CLIP Score} = 100 \times \cos(\mathbf{f}_I, \mathbf{f}_T)
    $$
3.  **符号解释:**
    *   $\mathbf{f}_I$: 由 CLIP 的图像编码器从生成图像/视频帧中提取的特征向量。
    *   $\mathbf{f}_T$: 由 CLIP 的文本编码器从输入文本提示中提取的特征向量。
    *   $\cos(\cdot, \cdot)$: 两个向量之间的余弦相似度。乘以 100 是为了将得分缩放到一个更直观的范围。

## 5.3. 对比基线
论文将 Phenaki 与多个领域的先进模型进行了比较：
*   **文本到视频生成:** `T2V`, `TFGAN`, `NUWA`。这些模型代表了当时主流的文本到视频生成技术。
*   **视频量化/重建:** 逐帧的 `Conv VQ-GAN` 和 `ViT VQ-GAN`。用于证明 `C-ViViT` 在视频压缩和重建方面的优势。
*   **视频预测:** `Video Transformer`, `CogVideo`, `DVD-GAN`, `Transframer`, `Video Diffusion` 等。用于展示 Phenaki 学习到的视频表示在传统的视频动态建模任务上的竞争力。

    ---

# 6. 实验结果与分析

## 6.1. 文本条件视频生成
*   **定性结果分析:**
    *   从下图（原文 Figure 3）可以看出，Phenaki 能够根据开放域的文本提示生成多样化且连贯的视频。
    *   一个关键的发现是，**模型能够生成视频数据集中不存在的风格**。例如，训练用的视频数据没有“铅笔画”风格，但由于模型在训练时也看到了包含这种风格的图像-文本数据，它成功地将这种静态风格泛化到了动态的视频生成中（例如第一行的熊猫视频）。
    *   这强有力地证明了**图文-视频联合训练**策略的有效性。

        ![Figure 3. Text conditional video generation. Each row shows selected frames from a video generated given the prompt. The model is trained on a mix of images and videos. The video dataset does not include any stylized videos such as pencil drawings, however, the image dataset does. The model can generalize from still images to videos. This figure also demonstrate the capability of the model in generating new unseen compositions. Full videos are available at phenaki.github.io.](images/3.jpg)
        *该图像是插图，展示了基于文本生成的视频示例。每一行显示了根据给定提示生成的不同场景，内容包括可爱的熊猫、宇航员等，展示了模型在多种情境下的创造能力。*

*   **定量结果分析:**
    *   以下是原文 Table 1 的结果，在 Kinetics-400 数据集上进行零样本评估：

        <table>
        <thead>
        <tr>
        <th>Method</th>
        <th>FID Image →</th>
        <th>FID Video V</th>
        </tr>
        </thead>
        <tbody>
        <tr>
        <td>T2V [25]</td>
        <td>82.13</td>
        <td>14.65</td>
        </tr>
        <tr>
        <td>SC [5] TFGAN [5]</td>
        <td>33.51</td>
        <td>7.34</td>
        </tr>
        <tr>
        <td>NUWA</td>
        <td>31.76</td>
        <td>7.19</td>
        </tr>
        <tr>
        <td><strong>Phenaki [0-Shot]</strong></td>
        <td><strong>28.46</strong></td>
        <td><strong>7.05</strong></td>
        </tr>
        </tbody>
        </table>

    *   **分析:** Phenaki 在零样本设置下，即没有在 Kinetics-400 上进行任何训练或微调的情况下，其 FID 分数（此处 FID 指的是逐帧图像质量）优于所有经过该数据集训练或微调的先前方法。这表明 Phenaki 具有强大的泛化能力。*（注意：原文表格和文本描述存在微小出入，表格中Phenaki的FID是28.46，文本中是37.74，FVD是3.84。此处遵循表格数据，但需注意原文可能存在的笔误。）*

## 6.2. 图文-视频联合训练的消融实验
*   **核心结果分析:**
    *   以下是原文 Table 2 的结果，展示了不同比例的图像和视频数据对模型性能的影响：

        <table>
        <thead>
        <tr>
        <th colspan="2">Data Split</th>
        <th colspan="2">Text to Video</th>
        <th></th>
        <th colspan="2">Text to Image</th>
        </tr>
        <tr>
        <th>Vid% / Img%</th>
        <th>CLIP ↑</th>
        <th>FID ↓</th>
        <th>FVD ↓</th>
        <th>CLIP ↑</th>
        <th>FID↓</th>
        </tr>
        </thead>
        <tbody>
        <tr>
        <td>100% / 0%</td>
        <td>0.298</td>
        <td>19.2</td>
        <td><strong>168.9</strong></td>
        <td>0.240</td>
        <td>53.9</td>
        </tr>
        <tr>
        <td>80% / 20%</td>
        <td><strong>0.303</strong></td>
        <td>21.4</td>
        <td>198.4</td>
        <td><strong>0.289</strong></td>
        <td><strong>29.4</strong></td>
        </tr>
        <tr>
        <td>50% / 50%</td>
        <td>0.302</td>
        <td>21.4</td>
        <td>239.7</td>
        <td>0.287</td>
        <td>30.5</td>
        </tr>
        </tbody>
        </table>

    *   **分析:**
        *   <strong>存在一个权衡 (trade-off):</strong>
            *   <strong>纯视频数据 (100% Vid):</strong> 获得了最佳的视频动态质量（FVD 最低）。这符合直觉，因为模型只学习了真实的视频动态。但其文本对齐能力 (CLIP score) 和图像生成能力 (Image FID) 较差。
            *   <strong>混合数据 (80% Vid / 20% Img):</strong> 尽管 FVD 有所上升（视频动态质量略微下降），但在文本-视频对齐 (Text to Video CLIP)、文本-图像对齐 (Text to Image CLIP) 和图像生成质量 (Text to Image FID) 上都取得了显著提升。
        *   **结论:** 引入图像数据可以极大地丰富模型对视觉概念的理解，提升其泛化和遵循指令的能力，代价是轻微牺牲视频的动态真实性。这验证了联合训练策略的价值，并揭示了其中的内在权衡。

## 6.3. C-ViViT 视频编码性能分析
*   **核心结果分析:**
    *   以下是原文 Table 3 的结果，在 Moments-in-Time 数据集上比较不同视频编码器的重建性能：

        <table>
        <thead>
        <tr>
        <th>Method</th>
        <th>FID ↓</th>
        <th>FVD ↓</th>
        <th>Number of Tokens ↓</th>
        </tr>
        </thead>
        <tbody>
        <tr>
        <td>Conv VQ-GAN [12]</td>
        <td>7.5</td>
        <td>306.1</td>
        <td>2560</td>
        </tr>
        <tr>
        <td>Conv VQ-GAN + Video loss</td>
        <td>13.7</td>
        <td>346.5</td>
        <td>2560</td>
        </tr>
        <tr>
        <td>ViT VQ-GAN [58]</td>
        <td>3.4</td>
        <td>166.6</td>
        <td>2560</td>
        </tr>
        <tr>
        <td>ViT VQ-GAN + Video loss</td>
        <td>3.8</td>
        <td>173.1</td>
        <td>2560</td>
        </tr>
        <tr>
        <td><strong>C-ViViT VQ-GAN (Ours)</strong></td>
        <td>4.5</td>
        <td><strong>65.78</strong></td>
        <td><strong>1536</strong></td>
        </tr>
        </tbody>
        </table>

    *   **分析:**
        *   <strong>时空一致性 (FVD):</strong> `C-ViViT` 的 FVD 分数 (65.78) **显著优于**所有基于逐帧图像编码的基线模型（最低为 166.6）。这证明 `C-ViViT` 通过其时空联合建模，能够更好地捕捉和重建视频的动态信息。
        *   <strong>压缩效率 (Number of Tokens):</strong> `C-ViViT` 将视频压缩为 1536 个词元，比基线模型的 2560 个词元**减少了 40%**。这意味着 `C-ViViT` 的表示更加紧凑和高效，大大降低了后续 Transformer 模型的计算负担。
        *   <strong>单帧质量 (FID):</strong> `C-ViViT` 的 FID (4.5) 略高于最好的逐帧模型 ViT VQ-GAN (3.4)，但在视觉上仍然是高质量的。
        *   **结论:** `C-ViViT` 在牺牲极小的单帧图像质量的代价下，极大地提升了视频的时间连贯性，并显著提高了压缩效率，证明了其作为视频词元化器的优越性。

## 6.4. 视频预测任务分析
*   **核心结果分析:**
    *   以下是原文 Table 4 (Kinetics-600) 和 Table 5 (BAIR) 的结果：

        <table>

      <caption>Table 4. Video prediction on Kinetics-600</caption>
      <thead>
        <tr>
          <th>Method</th>
          <th>FVD ↓</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Video Transformer [51]</td>
          <td>170.0 ± 5.00</td>
        </tr>
        <tr>
          <td>CogVideo [18]</td>
          <td>109.2</td>
        </tr>
        <tr>
          <td>DVD-GAN-FP [9]</td>
          <td>69.1 ± 0.78</td>
        </tr>
        <tr>
          <td>Video VQ-VAE [49]</td>
          <td>64.3 ± 2.04</td>
        </tr>
        <tr>
          <td>CCVS [28]</td>
          <td>55.0 ± 1.00</td>
        </tr>
        <tr>
          <td>TrIVD-GAN-FP [27]</td>
          <td>25.7 ± 0.66</td>
        </tr>
        <tr>
          <td>Transframer [31]</td>
          <td>25.4</td>
        </tr>
        <tr>
          <td>RaMViD [19]</td>
          <td>16.5</td>
        </tr>
        <tr>
          <td>Video Diffusion [17]</td>
          <td>16.2 ± 0.34</td>
        </tr>
        <tr>
          <td><strong>Phenaki (Ours)</strong></td>
          <td><strong>36.4 ± 0.19</strong></td>
        </tr>
      </tbody>
    </table>

    <table>

      <caption>Table 5. Video prediction on BAIR</caption>
      <thead>
        <tr>
          <th>Method</th>
          <th>FVD↓</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>DVD-GAN [9]</td>
          <td>109.8</td>
        </tr>
        <tr>
          <td>VideoGPT [55]</td>
          <td>103.3</td>
        </tr>
        <tr>
          <td>TrIVD-GAN [27]</td>
          <td>103.3</td>
        </tr>
        <tr>
          <td>Transframer [31]</td>
          <td>100.0</td>
        </tr>
        <tr>
          <td>HARP [57]</td>
          <td>99.3</td>
        </tr>
        <tr>
          <td>CCVS [28]</td>
          <td>99.0</td>
        </tr>
        <tr>
          <td>Video Transformer [51]</td>
          <td>94.0</td>
        </tr>
        <tr>
          <td>FitVid [3]</td>
          <td>93.6</td>
        </tr>
        <tr>
          <td>MCVD [47]</td>
          <td>89.5</td>
        </tr>
        <tr>
          <td>NUWA [54]</td>
          <td>86.9</td>
        </tr>
        <tr>
          <td>RaMViD [19]</td>
          <td>84.2</td>
        </tr>
        <tr>
          <td><strong>Phenaki (Ours)</strong></td>
          <td><strong>97.0</strong></td>
        </tr>
      </tbody>
    </table>
    *   **分析:** 视频预测（根据给定的起始帧生成后续帧）并非 Phenaki 的主要设计目标。尽管如此，Phenaki 在 Kinetics-600 和 BAIR 这两个标准的视频预测基准上，仍然取得了与许多专门为此任务设计的 SOTA（最先进的）模型相竞争的结果。这进一步证明了 `C-ViViT` 学习到的视频表示能够有效地建模视频的动态性，是其能够生成连贯视频的坚实基础。

        ---

# 7. 总结与思考

## 7.1. 结论总结
Phenaki 是一项在文本到视频生成领域的开创性工作。其主要贡献和发现可以总结如下：
1.  **实现了高质量、可变长度的视频生成：** 通过新颖的 `C-ViViT` 架构，Phenaki 解决了传统方法在视频长度灵活性和时空连贯性上的矛盾，能够生成任意时长的视频。
2.  **开创了故事驱动的生成范式：** Phenaki 首次实现了根据一系列随时间变化的文本提示生成连贯视频的能力，这极大地扩展了文本到视频技术的应用场景，使其能用于真正的视觉叙事。
3.  **证明了联合训练的巨大潜力：** 论文通过实验证明，将大规模图像-文本数据与有限的视频-文本数据结合进行训练，能够显著提升模型对视觉概念的理解和泛化能力，生成更加多样化和富有创意的视频内容。
4.  **提供了高效的视频表示学习方案：** `C-ViViT` 不仅在重建视频时保持了优异的时空连贯性，还以更高的效率压缩视频，为后续的生成模型减小了计算压力。

## 7.2. 局限性与未来工作
论文在“伦理声明 (Ethics Statement)”部分坦诚地讨论了模型的局限性和潜在风险：
*   **潜在的恶意使用：** 像 Phenaki 这样易于使用的生成系统，可能被用于制造虚假内容（深度伪造, deepfakes），并加速其传播。尽管当前生成视频的质量还未达到与真实视频无法区分的程度，但这只是时间问题。
*   **数据偏见：** 模型使用了包含 LAION-400M 在内的大规模网络数据集进行训练，这些数据集已知包含暴力、色情、血腥等不良内容，以及社会偏见。这些偏见可能会在生成的视频中被复现或放大。
*   **未来工作方向：** 基于以上考量，作者决定**暂不发布模型、代码和数据**。未来的工作将集中在：
    *   更好地理解和过滤数据、提示和输出内容。
    *   更明确地度量模型输出中编码的偏见。
    *   在数据、模型或后处理阶段积极地缓解这些偏见。

## 7.3. 个人启发与批判
*   **启发：**
    1.  **架构设计的巧思：** `C-ViViT` 将第一帧与后续帧分开处理，并在时间维度上采用因果注意力的设计，是解决可变长度视频建模和图文联合训练问题的优雅方案。这种“非对称”处理和因果约束的思想值得在其他序列建模任务中借鉴。
    2.  **数据利用的智慧：** 在目标域数据稀缺时，巧妙地利用相关但更丰富的源域数据（图像-文本）进行联合训练，是一种非常有效且务实的策略。这对于许多数据受限的领域具有重要的参考价值。
    3.  **应用范式的创新：** “故事驱动”的视频生成不仅是一个技术突破，更是一种全新的应用范式。它将生成模型从“单点生成”提升到了“叙事生成”的层面，为创意产业提供了巨大的想象空间。

*   **批判与思考：**
    1.  **动态质量与概念丰富度的权衡：** 实验揭示了在视频动态质量 (FVD) 和概念丰富度 (CLIP, FID) 之间的权衡。未来的研究需要探索如何在不牺牲动态真实性的前提下，更有效地从图像数据中迁移知识。这可能需要更精细的联合训练策略或模型架构设计。
    2.  **可控性仍有提升空间：** 尽管 Phenaki 可以遵循故事线，但对于视频中物体的精细运动、交互和物理规律的遵循程度，仍有很大的提升空间。例如，如何精确控制一个物体从 A 点移动到 B 点的路径，或者确保物理交互的真实性，是未来需要解决的难题。
    3.  **对计算资源的高度依赖：** 训练 Phenaki 这样的模型需要巨大的计算资源（论文中提到在 512 的批量大小下训练了 100 万步），这使得学术界和小型研究团队难以复现和跟进。如何设计更轻量级、更高效的文本到视频模型，是推动该领域普及的关键。
    4.  **伦理问题的紧迫性：** 作者负责任地讨论了伦理问题并选择不发布模型。这反映了顶级研究机构对 AGI 安全和责任的日益重视。随着生成技术的飞速发展，如何建立有效的监管、检测和溯源机制，将成为与技术研发同等重要的议题。