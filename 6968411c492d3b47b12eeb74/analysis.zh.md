# 1. 论文基本信息

## 1.1. 标题
**MotionStream: Real-Time Video Generation with Interactive Motion Controls**

**中文标题：** MotionStream：具有交互式运动控制的实时视频生成

论文标题直接点明了其核心研究主题：实现一种能够进行<strong>实时 (Real-Time)</strong> 和<strong>交互式 (Interactive)</strong> 运动控制的视频生成模型。`MotionStream` 这个名字也形象地传达了其“流式”生成视频的特性。

## 1.2. 作者
Joonghyuk Shin, Zhengqi Li, Richard Zhang, Jun-Yan Zhu, Jaesik Park, Eli Shechtman, Xun Huang.

作者团队来自多个顶级研究机构，包括 **Adobe Research**、<strong>卡内基梅隆大学 (Carnegie Mellon University)</strong> 和 <strong>首尔国立大学 (Seoul National University)</strong>。这代表了工业界顶级研究实验室与学术界顶尖高校的强强联合，通常预示着研究工作兼具前沿性和实用性。

## 1.3. 发表期刊/会议
该论文目前作为预印本 (preprint) 发布在 arXiv 上。根据其提交时间和研究内容，该工作很可能投递或已被顶级计算机视觉或计算机图形学会议接收，例如 CVPR、ICCV、SIGGRAPH 或 NeurIPS。这些会议在人工智能领域享有极高的声誉。

## 1.4. 发表年份
论文于 2025 年 11 月 3 日提交至 arXiv。这是一个未来的占位日期，表明该论文在撰写本分析时是一项非常前沿的研究工作。

## 1.5. 摘要
当前的运动条件视频生成方法存在两个核心痛点：**延迟极高**（生成一段视频需要数分钟）和**非因果处理**（必须在生成前知道整个运动轨迹），这使得实时交互成为不可能。为了解决这些问题，论文提出了 `MotionStream`，一个能够在单个 GPU 上实现亚秒级延迟、高达 29 FPS 流式生成的系统。

该方法分为两步：
1.  首先，作者在一个文本到视频模型的基础上增加了运动控制能力，训练出一个高质量的<strong>双向教师模型 (bidirectional teacher)</strong>。该模型能很好地遵循文本和运动指令，但速度慢且非因果。
2.  然后，通过一种名为 <strong>“带分布匹配蒸馏的自强制 (Self Forcing with Distribution Matching Distillation)”</strong> 的技术，将这个强大的教师模型蒸馏成一个<strong>因果学生模型 (causal student)</strong>，从而实现实时流式推理。

    为了解决生成长视频时遇到的**领域差距**、**误差累积**和**计算成本增长**等挑战，论文引入了精心设计的<strong>滑动窗口因果注意力 (sliding-window causal attention)</strong>，并结合了<strong>注意力池 (attention sinks)</strong> 的概念。通过在训练中模拟推理时的行为（如自推演和 KV 缓存滚动），模型能够在保持恒定生成速度的同时，生成任意长度的高质量视频。

最终，`MotionStream` 在运动遵循和视频质量上达到了最先进的水平，同时速度快了两个数量级，实现了真正意义上的交互式视频创作体验。

## 1.6. 原文链接
*   **原文链接:** https://arxiv.org/abs/2511.01266
*   **PDF 链接:** https://arxiv.org/pdf/2511.01266v2
*   **发布状态:** 预印本 (Preprint)。

    ---

# 2. 整体概括

## 2.1. 研究背景与动机
**核心问题：** 当前的视频生成技术，特别是那些允许用户通过绘制轨迹来控制物体运动的模型，离真正的“交互式”创作体验相去甚远。

<strong>现有研究的挑战与空白 (Gap)：</strong>
1.  <strong>速度过慢 (Slow):</strong> 生成一段几秒钟的视频通常需要数分钟甚至更长时间。这使得用户陷入了“渲染-等待”的循环，无法即时看到修改效果，极大地影响了创作流程。
2.  <strong>非因果性 (Non-causal):</strong> 现有模型（大多基于扩散模型）采用<strong>双向注意力 (bidirectional attention)</strong>，这意味着模型在生成第 N 帧时，需要看到第 N+1 帧乃至最后一帧的运动信息。因此，用户必须先完整地规划好所有运动轨迹，然后才能开始生成，无法做到“一边画，一边看结果”。
3.  <strong>时长受限 (Short-duration):</strong> 大多数模型只能生成几秒钟的短视频，无法支持更长时间的创作需求。

    这些限制共同导致了当前技术无法提供流畅、即时的交互体验，阻碍了视频生成技术在创意领域的广泛应用。

**本文的切入点与创新思路：**
`MotionStream` 的核心思路是彻底抛弃“一次性生成完整视频”的模式，转而采用<strong>自回归 (autoregressive)</strong> 的流式生成范式。其创新之处在于系统性地解决流式生成中的各种挑战：
*   **速度问题：** 通过<strong>知识蒸馏 (knowledge distillation)</strong>，将一个高质量但缓慢的“教师”模型的能力迁移到一个轻量、快速的“学生”模型上。
*   **因果性问题：** 将学生模型设计为<strong>因果 (causal)</strong> 架构，使其只依赖过去的信息来生成当前帧，从而实现流式生成。
*   **长视频问题：** 巧妙地借鉴了 `StreamingLLM` 中的<strong>注意力池 (attention sinks)</strong> 思想，并结合<strong>滑动窗口注意力 (sliding-window attention)</strong>，解决了长视频生成中常见的质量下降（漂移）问题，同时保持了恒定的计算开销。

## 2.2. 核心贡献/主要发现
论文的主要贡献可以总结为以下四点：
1.  **首个实时流式运动控制视频生成系统：** 提出了第一个能够在单个 H100 GPU 上达到 29.5 FPS 的流式运动条件视频生成管线，实现了真正的实时交互。
2.  **高效的系统设计与协同优化：** 设计了一套协同工作的系统，包括轻量化的轨迹编码头、高效的条件注入方式、集成了联合指导的蒸馏过程，并通过一个微型 VAE (`Tiny VAE`) 进一步加速，实现了极致的性能。
3.  **创新的长视频蒸馏策略：** 首次将**注意力池**和<strong>具备外推意识的训练 (extrapolation-aware training)</strong> 系统性地应用于视频生成蒸馏，通过在训练中模拟推理过程，有效防止了模型在生成长视频时的质量漂移。
4.  **卓越的性能与应用泛化：** 在运动迁移和相机控制等任务上取得了最先进 (state-of-the-art) 的结果，并且速度远超现有方法，能够稳健地泛化到各种交互式应用场景。

    ---

# 3. 预备知识与相关工作

## 3.1. 基础概念
*   <strong>扩散模型 (Diffusion Models):</strong> 这是一类强大的生成模型。其基本思想分为两步：
    1.  <strong>前向过程（加噪）：</strong> 从一张清晰的图像（或视频）开始，逐步地、迭代地向其添加高斯噪声，直到它变成完全随机的噪声。
    2.  <strong>反向过程（去噪）：</strong> 训练一个深度神经网络（通常是 U-Net 或 Transformer 架构），让它学习如何从一张充满噪声的图像中，一步步地把噪声去除，最终恢复出原始的清晰图像。
        在生成新内容时，模型从纯粹的随机噪声开始，利用学到的去噪能力，逐步“雕琢”出全新的、高质量的图像或视频。扩散模型因其出色的生成质量而备受青睐，但其迭代去噪的特性也导致了生成速度较慢。

*   <strong>自回归模型 (Autoregressive Models, AR):</strong> 这类模型生成序列数据（如文本、音频或视频帧）的方式是“循序渐进”的。在生成序列中的第 $t$ 个元素时，模型会依赖它已经生成的前 `t-1` 个元素作为输入。例如，在生成一句话时，模型会先生成第一个词，然后基于第一个词生成第二个词，再基于前两个词生成第三个词，依此类推。这种因果依赖性使其天然适合流式 (streaming) 应用。

*   <strong>知识蒸馏 (Knowledge Distillation):</strong> 这是一种模型压缩技术，其核心思想是“大马拉小车”。我们有一个庞大、复杂、性能强大但运行缓慢的<strong>教师模型 (teacher model)</strong>，以及一个结构更简单、参数更少、运行速度快的<strong>学生模型 (student model)</strong>。知识蒸馏的目标是让学生模型学习并模仿教师模型的输出行为，从而在保持较高性能的同时，获得巨大的速度提升。

*   <strong>注意力机制 (Attention Mechanism):</strong> 这是 Transformer 架构的核心。它允许模型在处理序列数据时，动态地为不同位置的输入分配不同的“注意力权重”。
    *   <strong>双向注意力 (Bidirectional Attention):</strong> 在处理序列中的某个元素时，可以同时“看到”它前面和后面的所有元素。这对于理解上下文很有帮助，但破坏了因果性，不适用于实时生成。
    *   <strong>因果注意力 (Causal Attention):</strong> 在处理某个元素时，只能“看到”它自己以及它前面的元素，不能“看到”未来的信息。这是实现自回归模型的关键。

*   <strong>KV 缓存 (KV Cache):</strong> 在自回归生成中，为了计算当前词元 (token) 的注意力，需要用到所有先前词元的键 (Key, K) 和值 (Value, V) 向量。`KV 缓存` 是一种优化技术，它将这些计算好的 K 和 V 向量存储起来，在生成下一个词元时直接复用，避免了大量重复计算，从而极大地加速了生成过程。

## 3.2. 前人工作
*   <strong>可控视频生成 (Controllable Video Generation):</strong> 此前的研究已经探索了多种控制信号，如光流、2D/3D 轨迹、边界框等。例如，`ControlNet` 通过复制主干网络 (backbone) 的一部分来添加额外的控制条件，效果很好但计算成本加倍。这些方法虽然控制精准，但都受限于扩散模型的非因果和慢速特性，无法用于实时交互。

*   **自回归与扩散模型的结合:** 为了兼顾质量和速度，一些工作尝试将自回归与扩散模型结合。`MotionStream` 便是沿着这条路线，但专注于解决长视频生成中的稳定性问题。

*   **实时性能的蒸馏方法:** `MotionStream` 直接借鉴了 `Self Forcing` 和 `CausVid` 的思想。
    *   `Self Forcing`: 提出了一种关键的训练策略，即在蒸馏过程中，让学生模型基于**自己**之前生成的（可能不完美的）输出进行后续生成，即<strong>自推演 (self-rollout)</strong>。这有助于缩小训练时（通常使用真实数据作为上下文）和测试时（只能依赖自身生成）之间的差距，提高生成稳定性。
    *   `CausVid`: 探索了如何将一个双向的扩散模型适配成一个因果模型，为 `MotionStream` 的学生模型初始化提供了方案。

*   <strong>长序列生成的稳定性 (`StreamingLLM`):</strong> 在大语言模型领域，`StreamingLLM` 发现，当使用滑动窗口注意力生成长文本时，如果简单地丢弃旧的词元，模型性能会迅速崩溃。其关键发现是，<strong>最初的几个词元 (initial tokens)</strong> 对维持注意力结构的稳定性至关重要。将这些初始词元固定在注意力窗口中，就像一个“锚”一样，可以显著提高长序列生成的质量。`MotionStream` 将这一思想创造性地引入视频领域，称之为<strong>注意力池 (attention sinks)</strong>，用视频的初始帧作为“锚”，解决了长视频生成中的漂移问题。

## 3.3. 技术演进
视频生成领域的技术演进脉络大致如下：
1.  **早期探索:** 基于 GAN 的模型开始尝试视频生成，但质量和稳定性有限。
2.  **扩散模型时代:** 扩散模型带来了生成质量的飞跃，成为主流范式。
3.  **可控性增强:** 为了让模型更实用，`ControlNet` 等技术被提出，允许用户通过各种条件（如姿态、深度、轨迹）来控制生成内容。但这些模型依然是“离线”的。
4.  **追求实时性:** 为了解决速度瓶颈，知识蒸馏被引入，出现了如 `LCM`、`UFOGen` 等快速图像/视频生成模型。
5.  **挑战长视频与流式生成:** 当模型变快后，新的挑战出现了：如何生成任意长的视频而不出现质量衰减？`MotionStream` 正是处在这一技术前沿，它系统性地解决了实时、可控、长时程这三个核心挑战。

## 3.4. 差异化分析
与相关工作相比，`MotionStream` 的核心差异化和创新点在于：
*   **目标不同:** 之前的工作要么关注生成质量（但慢），要么关注速度（但控制能力弱或无法生成长视频）。`MotionStream` 的目标是**同时实现实时、可控和无限长**这三个特性，打造真正的交互式体验。
*   **系统性地解决长视频漂移:** 它是**首个**将 `StreamingLLM` 的**注意力池**思想与**自推演**蒸馏范式结合，并应用于**运动控制视频生成**的工作。通过在训练阶段就用带注意力池和滚动 KV 缓存的自推演来模拟推理过程，它完美地解决了训练与推理之间的鸿沟，这是其能够稳定生成长视频的关键。
*   **架构和训练效率:** 它没有采用 `ControlNet` 那样高成本的架构，而是通过轻量化的编码头和简单的通道拼接来实现运动控制，效率更高。同时，它通过在蒸馏目标中“烘焙”复杂的联合指导，使得学生模型在推理时无需任何额外计算开销就能享受到高质量的指导效果。

    ---

# 4. 方法论
`MotionStream` 的方法论可以清晰地划分为两个核心阶段：首先，构建一个高质量但缓慢的**双向教师模型**；然后，通过专门设计的因果蒸馏流程，将其能力迁移到一个快速且能进行流式生成的**因果学生模型**。

## 4.1. 阶段一：构建运动可控的教师模型 (Sec 3.1)
这一阶段的目标是打造一个能力上限，为后续的蒸馏提供高质量的指导信号。

### 4.1.1. 轨迹表示与编码
为了让模型理解运动，首先需要将 2D 运动轨迹转换成模型能处理的格式。
*   **轨迹表示:** 受到 `MotionPrompting` 的启发，系统为场景中的每一条运动轨迹分配一个唯一的 ID。这个 ID 通过<strong>正弦位置编码 (sinusoidal positional encoding)</strong> 转换成一个 $d$ 维的嵌入向量 $\phi_n$。
*   **条件构建:** 接着，系统创建一个与视频潜在空间尺寸相匹配的张量 $c_m$。在每一帧 $t$ 的特定空间位置 $(x_t^n, y_t^n)$，如果轨迹 $n$ 是可见的，就将对应的嵌入向量 $\phi_n$ 放置在该位置。这个过程可以用以下公式描述：
    $$
    c _ { m } \big [ t , \lfloor \frac { y _ { t } ^ { n } } { s } \rfloor , \lfloor \frac { x _ { t } ^ { n } } { s } \rfloor \big ] = v [ t , n ] \cdot \phi _ { n }
    $$
    **符号解释:**
    *   $c_m$: 最终输入给模型的运动条件张量。
    *   $t$: 时间帧索引。
    *   $(x_t^n, y_t^n)$: 轨迹 $n$ 在第 $t$ 帧的 2D 坐标。
    *   $s$: VAE（视频编码器）的空间下采样率，用于将坐标对齐到潜在空间。
    *   `v[t, n]`: 一个二进制值（0 或 1），表示轨迹 $n$ 在第 $t$ 帧是否可见。
    *   $\phi_n$: 轨迹 $n$ 的 $d$ 维嵌入向量。
*   **轻量化编码头:** 在将 $c_m$ 输入主干网络前，会先通过一个轻量级的<strong>轨迹头 (track head)</strong> 进行处理，该头包含 $4\times$ 的时间压缩和一次 $1\times1\times1$ 卷积。重要的是，处理后的轨迹嵌入是直接与视频的潜在表示在通道维度上<strong>拼接 (concatenate)</strong> 的，而不是像 `ControlNet` 那样需要一个庞大的并行网络。这种设计极大地降低了计算开销。

### 4.1.2. 训练与鲁棒性增强
*   **训练目标:** 教师模型使用<strong>修正流匹配 (rectified flow matching)</strong> 目标进行训练。这是一种先进的生成模型训练范式，模型被训练来预测从噪声到真实数据的“速度场”。
*   **随机掩码:** 为了解决一个实际问题——模型无法区分“物体被遮挡”和“用户停止拖拽”（两者在输入中都表现为轨迹消失），作者在训练中引入了<strong>随机中段帧掩码 (stochastic mid-frame masking)</strong>。即以一定概率（$p_{mask}=0.2$）将视频片段中间某几帧的运动条件 $c_m$ 设为零。这使得模型学会了在运动信号短暂中断时也能保持视频内容的连贯性。

### 4.1.3. 联合文本与运动指导
为了平衡运动的精确性和画面的自然生动性，论文提出了一种<strong>联合指导 (joint guidance)</strong> 策略。
*   **动机:** 单纯的**运动指导**能精确地控制轨迹，但可能导致画面僵硬、缺乏细节（如物体只会平移）。单纯的**文本指导**能生成生动的动态效果（如天气变化、背景互动），但可能无法严格遵守指定的轨迹。
*   **联合指导公式:** 结合两者之长，最终的预测速度 $\hat{v}$ 由三部分组成：一个基础项、一个文本指导项和一个运动指导项。
    $$
    \boldsymbol { \hat { v } } = v _ { \mathrm { base } } + \boldsymbol { w _ { t } } \cdot \big ( \boldsymbol { v } ( c _ { t } , c _ { m } ) - v ( \emptyset , c _ { m } ) \big ) + \boldsymbol { w _ { m } } \cdot \big ( \boldsymbol { v } ( c _ { t } , c _ { m } ) - v ( c _ { t } , \emptyset ) \big )
    $$
    **符号解释:**
    *   $\hat{v}$: 最终用于去噪的速度向量。
    *   $v(c_t, c_m)$: 同时使用文本条件 $c_t$ 和运动条件 $c_m$ 时的模型预测。
    *   $v(\emptyset, c_m)$: 只使用运动条件时的预测（无文本条件）。
    *   $v(c_t, \emptyset)$: 只使用文本条件时的预测（无运动条件）。
    *   $w_t, w_m$: 分别是文本和运动指导的权重（超参数）。
    *   $v_{base}$: 一个加权的基础预测项，进一步平衡不同条件下的预测。

        这种联合指导策略虽然效果好，但每次去噪步骤需要模型进行 3 次前向传播，计算成本很高。这正是下一阶段蒸馏要解决的问题。

## 4.2. 阶段二：因果蒸馏 (Sec 3.2)
这一阶段的核心是将教师模型的强大能力“蒸馏”到一个能够实时流式运行的学生模型中。

### 4.2.1. 核心思想：注意力池与滚动缓存
在进入蒸馏流程之前，一个关键的洞察来自于对注意力图的可视化分析。如下图（原文 Figure 3）所示，无论是双向模型还是因果模型，许多注意力头都会持续关注视频的**初始帧**对应的词元。

![Figure 3: Visualization of self attention probability map. We visualize attention probability maps for bidirectional, full causal, and causal sliding window attentions. Several attention heads focus on the tokens corresponding to the initial frame throughout denoising generation.](images/3.jpg)  
*Figure 3: 自注意力概率图可视化。图中展示了双向、全因果和因果滑动窗口注意力的概率图。可以观察到，一些注意力头在整个生成过程中始终关注着对应于初始帧的词元。*

这个现象启发了作者借鉴 `StreamingLLM` 的思想，设计了<strong>注意力池 (attention sinks)</strong> 机制：
*   在自回归生成时，不采用简单的滑动窗口，而是将上下文分为两部分：
    1.  **注意力池:** 视频最开始的一个或几个“块” (chunks) 的 `KV` 向量被永久保留在缓存中。
    2.  **本地窗口:** 最近生成的几个“块”的 `KV` 向量组成一个滑动的窗口。
*   当生成新的块时，本地窗口会“滚动”：丢弃最旧的块，加入最新的块，而注意力池始终保持不变。这种机制为长视频生成提供了一个稳定的“锚点”，有效防止了质量漂移，同时通过固定大小的 `KV` 缓存保持了恒定的计算速度。

### 4.2.2. 自强制风格的蒸馏流程
`MotionStream` 的蒸馏流程如下图（原文 Figure 2 下半部分）所示，它基于 `Self Forcing` 和 <strong>分布匹配蒸馏 (Distribution Matching Distillation, DMD)</strong>。

![Figure 2: Model architecture and training pipeline. To build a teacher motion-controlled video model, we extract and randomly sample 2D tracks from the input video and encode them using a lightweight track head. The resulting track embeddings are combined with the input image, noisy video latents, and text embeddings as input to the diffusion transormer with bidirectional attention, which is then trained with a flow matching loss (top). We then distill a few-step causal diffusion model from the teacher through Self Forcing-style DMD distillation, integrating joint text-motion guidance into the objective, where autoregressive rollout with rolling KV cache and attention sink is applied during both training and inference (bottom).](images/2.jpg)  
*Figure 2: 模型架构与训练流程。顶部展示了教师模型的构建，底部则描绘了学生模型的蒸馏过程，其中自回归推演、滚动 KV 缓存和注意力池在训练和推理中都得到了应用。*

1.  <strong>自回归推演 (Autoregressive Rollout):</strong> 在训练的每一步，学生模型 $G_\theta$ 都会生成一系列视频块 $\hat{z}_0 = \{z_0^1, ..., z_0^L\}$。关键在于，在生成第 $i$ 个块时，它所依赖的上下文 $\mathcal{C}_i$ 是由**注意力池**和**它自己之前生成的块**组成的，而不是真实的视频数据。其上下文定义如下：
    $$
    \mathcal { C } _ { i } = \{ z _ { t } ^ { i } \} \cup \{ z _ { 0 } ^ { j } \} _ { j \leq S } \cup \{ z _ { 0 } ^ { j } \} _ { \operatorname* { m a x } ( 1 , i - W ) \leq j < i }
    $$
    **符号解释:**
    *   $z_t^i$: 当前正在去噪的第 $i$ 个块。
    *   $\{z_0^j\}_{j \le S}$: 之前生成并作为注意力池的 $S$ 个干净块。
    *   $\{z_0^j\}_{max(1, i-W) \le j < i}$: 之前生成并处于本地窗口的 $W$ 个干净块。

2.  **DMD 目标函数:** 学生模型生成完整的序列 $\hat{z}_0$ 后，DMD 目标函数被用来更新学生模型的参数 $\theta$。其梯度近似为：
    $$
    \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { DMD } } \approx - \mathbb { E } _ { t , \hat { z } _ { 0 } } \left[ \left( s _ { \mathrm { real } } ( ... ) - s _ { \mathrm { fake } } ( ... ) \right) \cdot \frac { \partial \hat { z } _ { 0 } } { \partial \boldsymbol { \theta } } \right]
    $$
    **符号解释:**
    *   $\mathcal{L}_{\mathrm{DMD}}$: DMD 损失函数，旨在让学生生成分布匹配教师的分布。
    *   $\nabla_\theta$: 对学生模型参数 $\theta$ 的梯度。
    *   $\hat{z}_0$: 学生模型生成的完整视频序列。
    *   $s_{\mathrm{real}}$: “真实”分数，由**固定的教师模型**给出。它告诉学生模型“一个好的生成应该是什么样的”。
    *   $s_{\mathrm{fake}}$: “伪造”分数，由一个与学生模型共同训练的<strong>批评家网络 (critic)</strong> 给出。它告诉学生模型“你当前的生成是什么样的”。
    *   $\frac{\partial \hat{z}_0}{\partial \theta}$: 从学生模型参数到其输出的梯度，用于反向传播。
    
        直观上，这个梯度通过 `(真实分数 - 伪造分数)` 的差值来驱动学生模型的更新，促使其生成的样本越来越能“骗过”教师模型。

3.  <strong>“烘焙”</strong>联合指导: 这是该方法的一个精妙之处。为了让学生模型在无需多次前向传播的情况下也能获得联合指导的好处，作者将昂贵的联合指导过程融入了 $s_{\mathrm{real}}$ 的定义中：
    $$
    s _ { \mathrm { real } } = s _ { \mathrm { base } } + w _ { t } \cdot ( f _ { \phi } ( c _ { t } , c _ { m } ) - f _ { \phi } ( \emptyset , c _ { m } ) ) + w _ { m } \cdot ( f _ { \phi } ( c _ { t } , c _ { m } ) - f _ { \phi } ( c _ { t } , \emptyset ) )
    $$
    与此同时，`s_fake` 的定义非常简单，不包含任何指导：
    $$
    s _ { \mathrm { fake } } = f _ { \psi } ( \overline { { c _ { t } } } , \overline { { c _ { m } } } )
    $$
    通过这种设计，联合指导的计算开销被完全“吸收”到了训练过程的损失函数中。学生模型 $G_\theta$ 被迫学习去直接生成符合联合指导效果的输出，而在**推理时，它只需要进行一次前向传播**，极大地提升了效率。

4.  **推理过程:** 推理过程与训练中的自推演过程**完全一致**：使用相同的注意力池和滚动 KV 缓存机制。这种训练-推理的强一致性是 `MotionStream` 能够稳定生成高质量长视频的核心保障。

    ---

# 5. 实验设置

## 5.1. 数据集
*   **训练数据集:**
    *   **OpenVid-1M:** 一个大规模的真实世界视频数据集。作者筛选出约 60 万个时长足够且宽高比为 16:9 的视频用于第一阶段训练。
    *   **合成数据集:** 使用更强大的 `Wan` 系列文生视频大模型生成的高质量、内容干净的视频。其中 7 万个 480p 样本用于训练 1.3B 模型，3 万个 720p 样本用于训练 5B 模型。合成数据主要用于微调阶段，以提升模型对轨迹的遵循能力和画面质量。
*   **评估数据集:**
    *   **DAVIS 验证集:** 包含 30 个视频，是视频对象分割领域的经典基准。其特点是场景复杂，包含大量遮挡，对模型的鲁棒性是很好的考验。
    *   **Sora Demo 子集:** 从 OpenAI Sora 模型的展示页面精心挑选了 20 个视频。这些视频通常运镜平滑、内容清晰、物体可见性好，适合评估模型的最佳性能。
    *   **LLFF 数据集:** 用于零样本 (zero-shot) 的新视角合成任务，以评估模型的相机控制能力。

## 5.2. 评估指标
论文使用了多项指标来全面评估模型的性能，涵盖了视频质量、运动准确性和生成速度。

*   <strong>PSNR (Peak Signal-to-Noise Ratio, 峰值信噪比):</strong>
    1.  **概念定义:** PSNR 是衡量图像或视频重建质量的经典指标。它通过计算原始信号与生成信号之间的均方误差 (MSE) 来衡量失真程度。PSNR 值越高，表示生成结果与真实标注数据 (Ground Truth) 之间的差异越小，质量越高。
    2.  **数学公式:**
        $$
        \text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right)
        $$
    3.  **符号解释:**
        *   $\text{MAX}_I$: 图像像素值的最大可能值（例如，对于 8 位图像是 255）。
        *   $\text{MSE}$: 原始图像和生成图像之间像素差值的均方误差。

*   <strong>SSIM (Structural Similarity Index Measure, 结构相似性指数):</strong>
    1.  **概念定义:** SSIM 是一种衡量两张图像相似度的指标，它比 PSNR 更符合人类的视觉感知。它综合考量了图像的亮度、对比度和结构信息。SSIM 的取值范围为 -1 到 1，越接近 1 表示两张图像越相似。
    2.  **数学公式:**
        $$
        \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}
        $$
    3.  **符号解释:**
        *   $\mu_x, \mu_y$: 图像 $x$ 和 $y$ 的平均值。
        *   $\sigma_x^2, \sigma_y^2$: 图像 $x$ 和 $y$ 的方差。
        *   $\sigma_{xy}$: 图像 $x$ 和 $y$ 的协方差。
        *   $c_1, c_2$: 用于维持稳定性的两个小常数。

*   <strong>LPIPS (Learned Perceptual Image Patch Similarity, 学习感知图像块相似度):</strong>
    1.  **概念定义:** LPIPS 是一种更先进的图像相似度度量，它利用深度神经网络（如 VGG, AlexNet）的特征来模拟人类的感知判断。它计算两张图像在网络深层特征空间中的距离。LPIPS 分数越低，表示两张图像在人类看来“越像”。
    2.  **数学公式:**
        $$
        d(x, x_0) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} \| w_l \odot (\hat{y}_{hw}^l - \hat{y}_{0hw}^l) \|_2^2
        $$
    3.  **符号解释:**
        *   $d(x, x_0)$: 图像 $x$ 和 $x_0$ 之间的 LPIPS 距离。
        *   $l$: 神经网络的第 $l$ 层。
        *   $\hat{y}^l, \hat{y}_0^l$: 从图像 $x, x_0$ 中提取的第 $l$ 层特征。
        *   $w_l$: 用于缩放不同通道激活的权重向量。

*   <strong>EPE (End-Point Error, 端点误差):</strong>
    1.  **概念定义:** EPE 是衡量运动估计准确性的标准指标。在本文中，它被用来量化生成视频中的运动轨迹与输入控制轨迹之间的偏差。它计算的是在可见帧上，输入轨迹点和从生成视频中提取出的对应轨迹点之间的欧几里得距离（L2 距离）。EPE 值越低，表示模型对运动轨迹的遵循能力越强。
    2.  **数学公式:**
        $$
        \text{EPE} = \frac{1}{N_{vis}} \sum_{i=1}^{N_{vis}} \sqrt{(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2}
        $$
    3.  **符号解释:**
        *   $N_{vis}$: 可见轨迹点的总数。
        *   $(x_i, y_i)$: 第 $i$ 个输入轨迹点的坐标。
        *   $(\hat{x}_i, \hat{y}_i)$: 从生成视频中跟踪得到的对应点的坐标。

## 5.3. 对比基线
论文将 `MotionStream` 与当前领域内多个最先进的运动控制视频生成模型进行了比较：
*   **Image Conductor:** 一种基于 AnimateDiff 的方法。
*   **Go-With-The-Flow (GWTF):** 一种基于 CogVideoX 的模型，使用光流进行控制。
*   **Diffusion-As-Shader (DAS):** 同样基于 CogVideoX，使用 3D 轨迹控制。
*   **ATI (Any Trajectory Instruction):** 基于强大的 $Wan 2.1-14B$ 模型。

    这些基线模型代表了当时运动控制视频生成领域的最高水平，但它们共同的特点是速度慢、非因果，是 `MotionStream` 旨在超越的典型代表。

---

# 6. 实验结果与分析

## 6.1. 核心结果分析
`MotionStream` 在运动迁移和相机控制两大任务上都展示了其卓越的性能。

### 6.1.1. 运动迁移（重建）任务
该任务旨在评估模型根据给定的运动轨迹重建原始视频的能力。结果如原文 Table 1 所示。

<table>
<tr>
<td rowspan="2">Method</td>
<td rowspan="2">Backbone &amp; Resolution</td>
<td rowspan="2">FPS</td>
<td colspan="4">DAVIS Validation Set</td>
<td colspan="4">Sora Demo Subset</td>
</tr>
<tr>
<td>PSNR</td>
<td>SSIM</td>
<td>LPIPS</td>
<td>EPE</td>
<td>PSNR</td>
<td>SSIM</td>
<td>LPIPS</td>
<td>EPE</td>
</tr>
<tr>
<td>Image Conductor (Li et al., 2025d)</td>
<td>AnimateDiff (256P)</td>
<td>2.98</td>
<td>11.30</td>
<td>0.214</td>
<td>0.664</td>
<td>91.64</td>
<td>10.29</td>
<td>0.192</td>
<td>0.644</td>
<td>31.22</td>
</tr>
<tr>
<td>Go-With-The-Flow Burgert et al. (2025)</td>
<td>CogVideoX-5B (480P)</td>
<td>0.60</td>
<td>15.62</td>
<td>0.392</td>
<td>0.490</td>
<td>41.99</td>
<td>14.59</td>
<td>0.410</td>
<td>0.425</td>
<td>10.27</td>
</tr>
<tr>
<td>Diffusion-As-Shader (Gu et al., 2025b)</td>
<td>CogVideoX-5B (480P)</td>
<td>0.29</td>
<td>15.80</td>
<td>0.372</td>
<td>0.483</td>
<td>40.23</td>
<td>14.51</td>
<td>0.382</td>
<td>0.437</td>
<td>18.76</td>
</tr>
<tr>
<td>ATI (Wang et al., 2025b)</td>
<td>Wan 2.1-14B (480P)</td>
<td>0.23</td>
<td>15.33</td>
<td>0.374</td>
<td>0.473</td>
<td>17.41</td>
<td>16.04</td>
<td>0.502</td>
<td>0.366</td>
<td>6.12</td>
</tr>
<tr>
<td>Ours Teacher (Joint CFG)</td>
<td>Wan 2.1-1.3B (480P)</td>
<td>0.79</td>
<td><strong>16.61</strong></td>
<td><strong>0.477</strong></td>
<td><strong>0.427</strong></td>
<td><strong>5.35</strong></td>
<td><strong>17.82</strong></td>
<td><strong>0.586</strong></td>
<td><strong>0.333</strong></td>
<td><strong>2.71</strong></td>
</tr>
<tr>
<td>Ours Causal (Distilled)</td>
<td>Wan 2.1-1.3B (480P)</td>
<td><strong>16.7</strong></td>
<td>16.20</td>
<td>0.447</td>
<td>0.443</td>
<td>7.80</td>
<td>16.67</td>
<td>0.531</td>
<td>0.360</td>
<td>4.21</td>
</tr>
<tr>
<td>Ours Teacher (Joint CFG)</td>
<td>Wan 2.2-5B (720P)</td>
<td>0.74</td>
<td>16.10</td>
<td>0.466</td>
<td>0.427</td>
<td>7.86</td>
<td>17.18</td>
<td>0.571</td>
<td>0.331</td>
<td>3.16</td>
</tr>
<tr>
<td>Ours Causal (Distilled)</td>
<td>Wan 2.2-5B (720P)</td>
<td><strong>10.4</strong></td>
<td><strong>16.30</strong></td>
<td>0.456</td>
<td>0.438</td>
<td>11.18</td>
<td>16.62</td>
<td>0.545</td>
<td>0.343</td>
<td>4.30</td>
</tr>
</table>

**分析:**
*   **质量与运动准确性:** 无论是 1.3B 模型还是 5B 模型，`Ours Teacher`（教师模型）在几乎所有质量指标（PSNR, SSIM, LPIPS）和运动准确性指标（EPE）上都全面超越了所有基线方法。这证明了其教师模型本身的高质量。`Ours Causal`（蒸馏后的学生模型）的性能略有下降，但仍然与最强的基线（如 ATI）相当或更优。
*   **速度的压倒性优势:** 这是最引人注目的结果。`