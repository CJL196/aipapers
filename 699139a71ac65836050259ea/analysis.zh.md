# 1. 论文基本信息

## 1.1. 标题
ConLA: Contrastive Latent Action Learning from Human Videos for Robotic Manipulation

## 1.2. 作者
Weisheng Dai \(^{1\dagger}\), Kai Lan \(^{2,3\dagger}\), Jianyi Zhou \(^{1}\), Bo Zhao \(^{4}\), Xiu \(\mathrm{Su}^5\), Junwen Tong \(^{2,3}\), Weili Guan \(^{1}\), Shuo Yang \(^{1\boxed{\ast2}}\)
\(^{1}\) Harbin Institute of Technology, Shenzhen
\(^{2}\) State Key Laboratory of Mobile Network and Mobile Multimedia Technology, Shenzhen
\(^{3}\) ZTE Corporation
\(^{4}\) Shanghai Jiao Tong University
\(^{5}\) Central South University
shuoyang@hit.edu.cn

## 1.3. 发表期刊/会议
arXiv 预印本，尚未正式发表。

## 1.4. 发表年份
2026年1月31日 (UTC)

## 1.5. 摘要
大型视觉-语言-动作 (Vision-Language-Action, VLA) 模型通过在海量机器人遥操作 (teleoperation) 数据集上进行预训练，展现出初步的泛化 (generalization) 能力。然而，获取能够全面覆盖各种任务和环境的数据集极其昂贵且难以扩展。相比之下，人类演示视频提供了丰富且可扩展的场景和操作行为来源，但它们缺乏明确的动作监督 (action supervision)，阻碍了直接利用。现有工作利用基于 VQ-VAE 的框架以无监督 (unsupervised) 方式从人类视频中学习潜在动作 (latent actions)。然而，由于训练目标主要关注重建视觉外观 (visual appearances) 而非捕捉帧间动态 (inter-frame dynamics)，所学到的表示 (representations) 往往依赖虚假视觉线索 (spurious visual cues)，导致捷径学习 (shortcut learning) 和纠缠的潜在表示，从而降低了可迁移性 (transferability)。为了解决这个问题，本文提出了 ConLA，一个用于从人类视频中学习机器人策略 (robotic policies) 的无监督预训练框架。ConLA 引入了一种对比解耦机制 (contrastive disentanglement mechanism)，该机制利用动作类别先验 (action category priors) 和时间线索 (temporal cues) 来将运动动态 (motion dynamics) 从视觉内容中分离出来，从而有效缓解捷径学习。广泛的实验表明，ConLA 在各种基准测试中取得了强大的性能。值得注意的是，通过仅在人类视频上进行预训练，本文方法首次超越了通过真实机器人轨迹 (robot trajectory) 预训练所获得的性能，突显了其提取纯净且语义一致的潜在动作表示的能力，为可扩展的机器人学习 (scalable robot learning) 奠定了基础。

## 1.6. 原文链接
https://arxiv.org/abs/2602.00557
PDF 链接: https://arxiv.org/pdf/2602.00557v1

# 2. 整体概括

## 2.1. 研究背景与动机
### 2.1.1. 研究背景
近年来，大型语言模型 (Large Language Models, LLMs) 的成功揭示了可预测的扩展定律：随着模型规模、数据集规模和计算量的增加，性能会提高，泛化能力也会自然涌现。受此启发，视觉-语言-动作 (Vision-Language-Action, VLA) 模型在大型机器人遥操作数据集上进行预训练，在实现初步泛化方面取得了令人鼓舞的进展。这些模型能够将视觉观察和语言指令映射到机器人动作，从而执行各种操作任务。

### 2.1.2. 动机与现有挑战
然而，获取能够全面覆盖所有可能环境并包含各种任务的机器人遥操作数据集在实践中是不可行的，而且对于某些特定环境或任务，数据收集可能极其困难甚至不可能。这严重限制了 VLA 模型的可扩展性 (scalability) 和更广泛的适用性。

与此形成对比的是，大量人类演示视频提供了天然丰富且可扩展的数据源，具有增强 VLA 模型泛化能力的巨大潜力。然而，这些视频缺乏明确的机器人动作轨迹 (robotic action trajectories)，使得直接用于 VLA 训练变得困难。

现有解决这一挑战的工作（如 LAPA）通常采用基于 VQ-VAE (Vector Quantized-Variational AutoEncoder) 的框架，以无监督方式从视频中提取潜在动作 (latent actions)，从而将人类视频的运动先验 (motion prior) 迁移到 VLA 模型中。但这些方法存在一个根本性局限：基于 VQ-VAE 的潜在动作提取方法容易出现<strong>捷径学习 (shortcut learning)</strong>。由于其训练目标主要侧重于重建视觉外观 (visual appearances) 而非捕捉真正的帧间动态 (inter-frame dynamics)，模型往往无法捕获有意义的运动信息，而是记忆未来的视觉内容以最小化重建误差。这导致潜在动作空间 (latent action space) 与不相关的视觉特征 (irrelevant visual features) 混合，形成<strong>纠缠的潜在表示 (entangled latent representations)</strong>，从而限制了其在机器人学习中的可迁移性。这个问题在人类视频中尤为突出，因为其固有的复杂视觉变化使得潜在动作提取更加困难。

### 2.1.3. 创新思路
本文的切入点在于：如何缓解捷径学习的影响，并从人类视频中提取更纯净的潜在动作，从而充分释放人类视频预训练对 VLA 模型的潜力？作者观察到，人类操作视频包含大量重复的动作原语 (action primitives)（例如：抓取、放置、移动），这为潜在动作学习提供了自然的语义线索 (semantic cues)。同时，视频天然包含丰富的时间信息：运动特征对时间顺序高度敏感，而视觉外观相对稳定。基于这些洞察，本文提出了一种对比解耦机制，利用动作类别先验和时间线索来分离运动动态和视觉内容。

## 2.2. 核心贡献/主要发现
本文的主要贡献如下：
*   **识别并解决捷径学习问题：** 识别出现有基于 VQ-VAE 的潜在动作学习方法普遍存在捷径学习问题，即模型过度依赖视觉外观而非真实运动动态。为解决此问题，引入对比学习 (contrastive learning) 来解耦视觉和动作表示，使潜在动作能够更忠实地捕捉真实运动语义。
*   <strong>提出对比解耦架构 (Contrastive Disentanglement Architecture)：</strong> 提出了一种对比解耦架构，该架构利用动作类别先验 (action category priors) 和时间先验 (temporal priors) 来确保具有相同语义的潜在动作在不同环境和主体 (embodiments) 中紧密聚类 (cluster compactly)，从而改进了从人类视频中学习潜在动作的过程。
*   **实现最先进的性能：** ConLA 在仿真基准 (simulation benchmarks) 和真实机器人测试中均达到了最先进的 (state-of-the-art) 性能。在 SimplerEnv [27] 基准测试中，相较于 LAPA [57]，成功率提高了 12.5%，在真实世界测试中提高了 15.9%。
*   **首次超越真实机器人数据预训练：** 值得注意的是，仅通过在人类视频上进行预训练，ConLA 的策略 (policy) 性能首次超越了使用真实机器人轨迹数据预训练的策略，提高了 1.1%。这证明了使用大规模人类视频数据集扩展 VLA 的可行性。

# 3. 预备知识与相关工作

## 3.1. 基础概念
### 3.1.1. 视觉-语言-动作模型 (Vision-Language-Action Models, VLA)
**概念定义：** VLA 模型是一类结合了计算机视觉、自然语言处理和机器人控制的模型。它们旨在通过理解视觉观察和语言指令，生成或预测机器人执行任务所需的动作。VLA 模型的目标是使机器人能够像人类一样理解复杂的指令并与世界互动，从而实现通用型机器人操控。

### 3.1.2. 遥操作 (Teleoperation)
**概念定义：** 遥操作是指通过人远程控制机器人的操作方式。在机器人学习中，遥操作通常用于收集带有明确动作标注 (action annotations) 的数据集，即人类操作员在控制机器人时，其发出的指令（如关节角度、末端执行器姿态等）会被记录下来，作为机器人的“真值 (Ground Truth)”动作。

### 3.1.3. 潜在动作 (Latent Actions)
**概念定义：** 潜在动作是指一种抽象的、非直接可执行的动作表示。它不直接对应于机器人末端执行器的具体关节角度或速度，而是通过学习从视觉输入中提取出的、更高级别的、语义化的运动描述。这些潜在动作通常是离散的词元 (tokens) 或连续的嵌入 (embeddings)，旨在捕捉视频中的核心运动意图，而忽略不相关的视觉细节。通过学习潜在动作，模型可以从缺乏显式动作标签的视频（如人类视频）中提取运动先验，然后将其映射到机器人可执行的动作空间。

### 3.1.4. VQ-VAE (Vector Quantized-Variational AutoEncoder)
**概念定义：** VQ-VAE 是一种生成模型，它结合了变分自编码器 (Variational AutoEncoder, VAE) 和向量量化 (Vector Quantization, VQ) 的思想。其核心在于学习离散的潜在表示。
**工作原理：**
*   <strong>编码器 (Encoder):</strong> 将输入（例如图像或视频帧）编码成一个连续的潜在向量 (latent vector)。
*   <strong>量化模块 (Quantization Module):</strong> 这是 VQ-VAE 的关键。它将编码器输出的连续潜在向量“吸附”到最近的离散码本 (codebook) 向量上。码本是一个预定义的、有限数量的离散向量集合。这种“吸附”过程实现了信息的离散化，迫使模型学习具有语义意义的、可分类的潜在表示。
*   <strong>解码器 (Decoder):</strong> 根据量化后的离散向量重建原始输入。
    **目的：** 在本文中，VQ-VAE 用于从视频帧对中学习离散的潜在动作词元。这些词元代表了帧间的运动信息，从而可以将连续的运动空间离散化。

### 3.1.5. 对比学习 (Contrastive Learning)
**概念定义：** 对比学习是一种自监督 (self-supervised) 或弱监督 (weakly-supervised) 学习范式，其目标是通过最大化“正样本对”之间的一致性，同时最小化“负样本对”之间的一致性，来学习有意义的表示。
**工作原理：**
*   <strong>正样本对 (Positive Pairs):</strong> 指在语义上相似或相关的样本（例如，同一图像的不同增强版本，或具有相同动作标签的不同视频片段）。
*   <strong>负样本对 (Negative Pairs):</strong> 指在语义上不相似或不相关的样本。
*   <strong>对比损失 (Contrastive Loss):</strong> 旨在拉近正样本对的表示，推开负样本对的表示。常见的对比损失函数包括 InfoNCE loss 或 Supervised Contrastive Loss。
    **目的：** 在本文中，对比学习用于解耦潜在动作与视觉内容，通过强制相同动作类别的潜在动作聚类，同时区分不同动作类别的动作，并利用时间信息区分运动和静态视觉特征。

### 3.1.6. 捷径学习 (Shortcut Learning)
**概念定义：** 捷径学习是指机器学习模型在训练过程中，并非真正学习到任务的核心规律或因果关系，而是利用数据中一些表面上、统计上的相关性（即“捷径”），以最小化训练损失。这些捷径在训练数据中可能有效，但在泛化到新环境或略有不同的分布时会失效。
**本文语境：** 在从视频中学习潜在动作时，如果模型只关注视觉重建，它可能会学会记忆未来帧的视觉内容（例如物体的颜色、背景纹理），而不是真正理解帧间的运动动态（例如物体从A点移动到B点）。这种对视觉外观的过度依赖就是捷径学习，导致学习到的潜在动作无法泛化到不同视觉背景但运动模式相同的场景。

### 3.1.7. 逆动力学模型 (Inverse Dynamics Model, IDM) 和正动力学模型 (Forward Dynamics Model, FDM)
**概念定义：**
*   <strong>逆动力学模型 (IDM):</strong> 接收两个连续状态（例如，当前帧 $O_t$ 和未来帧 $O_{t+k}$）作为输入，并预测导致从第一个状态到第二个状态转变的动作。
*   <strong>正动力学模型 (FDM):</strong> 接收当前状态 $O_t$ 和一个动作（真实动作或潜在动作）作为输入，并预测未来的状态 $O_{t+k}$。
    **目的：** 在本文中，编码器被视为 IDM，它从帧对中提取潜在动作；解码器被视为 FDM，它利用当前帧和潜在动作来重建未来帧。

### 3.1.8. 自回归模型 (Autoregressive Models)
**概念定义：** 自回归模型是一种序列模型，它根据序列中前面已生成或观察到的元素来预测下一个元素。在语言模型中，这意味着模型会根据已经生成的词元来预测下一个词元。
**目的：** 在本文的潜在动作预训练阶段，VLA 策略被训练成一个自回归模型，根据当前视觉观察和语言指令，预测一系列离散的潜在动作词元。

## 3.2. 前人工作
### 3.2.1. 视觉-语言-动作模型 (VLA Models)
VLA 模型是近年来结合大型语言模型 (LLMs) 和视觉-语言模型 (VLMs) 成功发展起来的。它们将视觉观察和语言指令映射到机器人动作，以执行操作任务。
*   **OpenVLA [24]:** 通过在大型遥操作数据集上预训练，将动作建模为语言模型词汇表中的词元，从而实现通用操控能力。
*   **$\pi 0$ [6] 和 $\pi 0.5$ [18]:** 进一步利用跨主体 (cross-embodiment)、多源 (multi-source) 遥操作数据，并采用基于流匹配 (flow-matching) 的架构，增强了执行精细任务的能力并展现出更强的泛化性。
    **局限性：** 现有方法高度依赖于带有动作标注的大规模遥操作数据集，这限制了它们的可扩展性和广泛适用性。

### 3.2.2. 从人类视频中学习 (Learning from Human Videos)
由于收集大规模机器人遥操作数据非常困难，从视频演示中学习成为一个有前景的范式。
*   **显式信息提取方法：**
    *   **EgoMimic [20] 和 HAT [40]:** 从以自我为中心的 (egocentric) 人类视频中训练任务特定策略，但依赖于配对的人-机器人数据，限制了可扩展性和泛化性。
    *   **EgoVLA [56] 和 Being-HO [36]:** 使用以自我为中心的人类视频预训练策略，但仍无法利用大规模的免费互联网视频，需要精心收集的人类演示，并需处理人手到机器人手的重定向 (hand retargeting)。
        **局限性：** 这些方法尽管比遥操作数据更容易获取，但仍受限于数据收集所需的努力，限制了其可扩展性。

### 3.2.3. 从视频中学习潜在动作 (Learning Latent Actions from Videos)
另一类工作侧重于从视频中学习潜在动作，并将其用于策略建模。这些方法通常依赖于无监督的逆动力学模型 (Inverse Dynamics Models, IDMs) 从无标签视频中提取动作先验，然后用于训练 VLA 策略。
*   **LAPA [57]:** 率先利用 VQ-VAE [46] 从连续帧中提取运动先验，从而将人类视频中的知识迁移到机器人操控中。
*   **CLAM [28] 和 COMO [55]:** 强调离散潜在动作在表达能力上的局限性，主张在连续动作空间中建模潜在动作以提高表示容量。
    **局限性：** 这些基于 VQ-VAE 的潜在动作提取方法容易出现捷径学习。
*   **UniVLA [9]:** 通过重建未来帧的 DINOv2 [37] 特征来部分解决这个问题，并构建以任务为中心的潜在动作，以减少不相关的环境噪声。
    **局限性：** UniVLA 缺乏明确的归纳偏置 (inductive biases)，其表示仍未能完全捕捉人类视频中的运动语义。

## 3.3. 技术演进
从大型语言模型 (LLMs) 和视觉-语言模型 (VLMs) 的成功，研究人员受到了启发，将其扩展到机器人领域，形成了视觉-语言-动作模型 (VLAs)。早期 VLAs 依赖于昂贵且难以扩展的机器人遥操作数据。为了克服数据瓶颈，研究转向了利用更易获取的人类视频。这引发了两条主要的技术路线：一是通过显式地提取结构化信息（如手部姿态）将人类动作重定向到机器人，但这通常需要配对数据或特定传感器；二是更通用的方法，即从视频中无监督地学习抽象的“潜在动作”。LAPA 是这一方向的代表，它使用 VQ-VAE 将视频帧间的运动编码为离散的潜在动作词元。然而，LAPA 及类似方法面临着“捷径学习”的挑战，模型倾向于记忆视觉内容而非真正的运动动态，导致潜在动作表示的纠缠和泛化能力不足。ConLA 正是在这一背景下提出的，旨在通过引入对比解耦机制，利用动作类别先验和时间线索，更有效地从人类视频中提取纯净、语义一致的潜在动作，从而推动从人类视频中进行机器人学习的边界。

## 3.4. 差异化分析
ConLA 与相关工作的核心区别和创新点在于其引入的**对比解耦机制**，旨在直接解决现有方法（特别是基于 VQ-VAE 的 LAPA [57] 和部分 UniVLA [9]）中存在的**捷径学习**和**纠缠潜在表示**问题。

*   **与 LAPA [57] 的区别：**
    *   **LAPA：** 仅依赖无监督的 VQ-VAE 进行潜在动作提取。其训练目标主要通过重建视觉外观来优化，缺乏对运动语义的直接激励，因此容易出现捷径学习，导致潜在动作混杂视觉噪声，语义一致性差。
    *   **ConLA：** 在 VQ-VAE 框架的基础上，引入了<strong>动作中心对比学习 (Action-Centric Contrastive Learning)</strong> 和<strong>视觉中心对比学习 (Vision-Centric Contrastive Learning)</strong>。
        *   **动作中心对比学习**利用**动作类别标签**作为弱监督信号，强制相同动作类别的潜在动作在特征空间中紧密聚类，从而增强其语义一致性，防止模型通过记忆视觉内容来最小化重建损失。
        *   **视觉中心对比学习**利用<strong>时间反转增强 (inverse-order augmentation)</strong> 的时间先验，将正向和反向视频对的视觉内容部分拉近，但允许动作部分发生显著变化。这鼓励模型学习对运动变化不敏感的视觉表示，并帮助将运动动态与静态视觉线索分离。
    *   **结果：** ConLA 能够提取更纯净、语义更一致的潜在动作，从而显著提高了从人类视频中学习的性能，甚至超越了真实机器人轨迹预训练的基线。

*   **与 UniVLA [9] 的区别：**
    *   **UniVLA：** 通过重建未来帧的 DINOv2 [37] 特征来减少环境噪声，并构建以任务为中心的潜在动作。它在一定程度上缓解了不相关的环境噪声问题。
    *   **ConLA：** 尽管 UniVLA [9] 尝试减少噪声，但它缺乏<strong>显式归纳偏置 (explicit inductive biases)</strong> 来直接指导模型解耦运动和视觉内容。ConLA 则明确地利用了**动作类别先验**和**时间先索**作为归纳偏置，通过对比学习机制，更直接地将动作动态从视觉干扰中分离出来，从而学习到更紧凑、解耦更有效的潜在表示。

        总之，ConLA 的核心创新在于其**对比解耦模块**，它通过主动利用视频的内在结构（动作语义和时间属性）来指导潜在动作的生成，而不是仅仅依赖视觉重建或间接的特征重建，从而更有效地解决了捷径学习和表示纠缠的根本问题。

# 4. 方法论

本文提出的 ConLA 框架是一个无监督预训练框架，用于从人类视频中学习机器人策略。其核心在于通过对比学习从视频中解耦出潜在动作，并利用这些潜在动作进行策略预训练。整个框架包含三个关键阶段：1) 对比潜在动作学习，2) 潜在动作预训练，3) 动作微调。

## 4.1. 方法原理
ConLA 的核心思想是，从人类演示视频中提取一个紧凑且语义一致的潜在动作表示，以促进运动知识向机器人学习的迁移。为了解决现有方法中存在的捷径学习问题（即模型倾向于学习视觉外观而非真正的运动动态），ConLA 引入了一个**对比解耦机制**。该机制利用视频中固有的**动作类别先验**和**时间先验**作为归纳偏置 (inductive biases)，通过对比学习来主动将运动动态与不相关的视觉内容分离。这样可以确保学习到的潜在动作忠实地捕捉运动语义，并在不同视觉背景下保持一致性。

## 4.2. 核心方法详解 (逐层深入)

### 4.2.1. 阶段一：对比潜在动作学习 (Contrastive Latent Action Learning)
此阶段的目标是训练一个基础模型，为视频生成伪标签（即潜在动作词元）。具体来说，对比学习被用来引导潜在动作表示与视觉噪声的解耦，从而产生更具判别性的伪标签，为第二阶段的策略预训练提供可靠的基础。

#### 4.2.1.1. 潜在动作量化 (Latent Action Quantization)
首先，从视频中构建一个视频对 $[O_t, O_{t+k}]$，其中 $O_t$ 是当前帧，$O_{t+k}$ 是未来帧，帧间隔为 $k$。同时，获取其对应的动作类别标签 $y$。为了融入时间先验，本文应用逆序增强 (reverse-order augmentation) 来创建逆向对 $[O_{t+k}, O_t]$。

潜在动作模型包含一个<strong>逆动力学模型 (Inverse Dynamics Model)</strong> 作为编码器 $I$ 和一个<strong>正动力学模型 (Forward Dynamics Model)</strong> 作为解码器 $F$。编码器 $I$ 采用 C-ViViT 分词器 [47] 中的时空 Transformer [54] 实现，它接收当前帧 $O_t$ 和未来帧 $O_{t+k}$ 作为输入，并提取这两帧之间的运动信息，生成一个潜在动作嵌入 $Z \in \mathbb{R}^d$，其中 $d$ 是预定义的维度。

为了获得语义一致的潜在动作表示，$Z$ 会进一步由<strong>对比解耦模块 (Contrastive Disentanglement Module)</strong> 处理，得到更具判别性和结构化的嵌入 $Z_a$。然后对 $Z_a$ 应用潜在量化 (latent quantization) 得到 $Z_{aq}$，并使用 VQ-VAE [46] 目标函数和大小为 $|C|$ 的码本 (codebook) 进行优化。解码器 $F$ 采用空间 Transformer 实现，它接收当前帧 $O_t$ 和量化后的潜在动作词元 $Z_{aq}$ 作为输入，生成预测的未来帧 $\hat{O}_{t+k}$。此阶段的目标是最小化重建误差：$\| \hat{O}_{t+k} - O_{t+k}\|^2$。

#### 4.2.1.2. 对比解耦模块 (Contrastive Disentanglement Module)
如 Figure 2 所示，先前的 VQ-VAE [46] 范式存在严重的捷径学习问题：模型学习到的潜在动作通常编码的是未来帧视觉内容的离散副本，而非真正的帧间动态。为减轻视觉信息在潜在动作提取中的干扰，本文引入了一个对比解耦框架，该框架包含<strong>动作中心对比学习 (Action-Centric Contrastive Learning)</strong> 和<strong>视觉中心对比学习 (Vision-Centric Contrastive Learning)</strong>。这两个组件共同将动作与视觉内容解耦，使模型能够生成高质量且语义一致的潜在动作表示，用于下游策略学习。

下图（原文 Figure 3）展示了对比潜在动作学习的框架：

![fig 7](images/7.jpg)
*该图像是一个示意图，展示了ConLA框架的工作流程，包括对比数据准备、视觉-动作对比解耦和基于动作的图像重建。框架通过引入对比学习机制，旨在从人类视频中有效提取动作动态，促进机器人学习。*

Figure 3. 对比潜在动作学习。本文提出一个对比解耦框架，用于在涵盖当前帧和未来帧的视频片段中将动作从视觉干扰中分离出来，指导构建紧凑的潜在动作表示。具体来说，带有动作类别标签的样本及其反向增强的对应样本被编码成潜在动作嵌入，这些嵌入被均匀分割并输入到动作头（Action head）进行动作中心对比学习，以及输入到视觉头（Visual head）进行视觉中心对比学习，以实现解耦表示。来自动作头部的优化表示进一步量化，所得的量化潜在动作与当前帧 $O_t$ 一起用于重建未来帧 $O_{t+k}$。

1.  <strong>动作中心对比学习 (Action-Centric Contrastive Learning):</strong>
    在通过编码器获取潜在动作表示 $Z$ 后，我们将其均匀地分成两部分：$Z_{a'}$（与动作相关）和 $Z_{v'}$（与视觉相关）。这通过以下方式实现：
    $$
    Z = I\big([O_{t},O_{t + k}]\big),\quad Z\in \mathbb{R}^{d} \quad (1)
    $$
    $$
    Z = \left[Z_{a'};Z_{v'}\right],Z_{a'},Z_{v'}\in \mathbb{R}^{d / 2} \quad (2)
    $$
    其中，$I$ 是编码器，接收当前帧 $O_t$ 和未来帧 $O_{t+k}$ 作为输入，输出潜在动作嵌入 $Z$，其维度为 $d$。$Z$ 被分割成两个维度为 $d/2$ 的子向量 $Z_{a'}$ 和 $Z_{v'}$。

    对于 $Z_{a'}$，本文应用一个两层 MLP 作为动作头 (action head) 将表示投影到动作空间，得到 $Z_a$：
    $$
    \pmb{Z}_{a} = \operatorname {MLP}_{\mathrm{action}}(\pmb {Z}_{a^{\prime}}),\quad \pmb {Z}_{a}\in \mathbb{R}^{d} \quad (3)
    $$
    为了学习紧凑的潜在动作表示，本文采用动作中心对比学习，通过优化一个动作损失 $L_{\text{action}}$。该损失被实现为监督对比目标 (supervised contrastive objective) [22]：
    $$
    \mathcal{L}_{\mathrm{action}} = \sum_{i\in I}\frac{-1}{|P(i)|}\sum_{p\in P(i)}\log \frac{\exp{(Z_{a,i}\cdot Z_{a,p} / \tau)}}{\sum_{a\in A(i)}\exp{(Z_{a,i}\cdot Z_{a,a} / \tau)}} \quad (4)
    $$
    **符号解释：**
    *   $i \in I \equiv \{1, \dots, N\}$：表示批次中样本的索引，称为锚点 (anchor)。
    *   $\pmb{Z}_{a,i}$：表示第 $i$ 个样本的动作嵌入 (action embedding)。
    *   $\tau$：是一个标量温度参数 (scalar temperature parameter)。
    *   $A(i) \equiv I \backslash \{i\}$：表示批次中除 $i$ 之外的所有索引的集合。
    *   $P(i) \equiv \{p \in A(i) : \tilde{y}_{p} = \tilde{y}_{i}\}$：表示锚点 $i$ 的所有正样本 (positive samples) 的集合，即与 $i$ 共享相同动作标签 $\tilde{y}$ 的样本。
    *   $|P(i)|$：表示集合 `P(i)` 的基数（元素数量）。
        **目的：** 与 LAPA [57] 仅依赖无监督 VQ-VAE 不同，引入动作类别标签形式的弱监督 (weak supervision) 显著提高了潜在表示的判别性。在没有监督的情况下，潜在动作极易受到视觉干扰（如背景变化），这可能导致相似动作被编码为完全不同的潜在表示，从而形成纠缠的表示空间。动作损失通过将相同动作类别的表示拉近，同时推开不同类别的表示，从而在潜在空间中形成紧凑且语义连贯的聚类。这种机制有效缓解了捷径学习，并产生了更具判别性的潜在动作表示。

2.  <strong>视觉中心对比学习 (Vision-Centric Contrastive Learning):</strong>
    在逆动力学建模中，当前帧和未来帧之间的差异不仅包含运动信息，还包含不可避免的环境噪声，例如相机抖动、视角变化或光照波动。如果没有归纳偏置，使用无监督学习很难解耦这些组件。本文利用<strong>时间敏感性先验 (temporal sensitivity prior)</strong>：当帧顺序反转时，运动信息会发生显著变化，而内容信息和视觉干扰则相对稳定。基于这一先验，本文引入视觉中心对比学习目标，以保持内容一致性，同时减少运动变化的影响。

    具体来说，本文将反转的帧对 $[O_{t+k}, O_t]$ 输入编码器，以获得逆序序列的潜在动作表示，记为 $\mathbf{Z}^I$。然后将 $\mathbf{Z}^I$ 均匀地分成两部分，得到 $\mathbf{Z}_{\alpha'}^{\mathcal{I}}$ 和 $\mathbf{Z}_{\upsilon'}^{\mathcal{I}}$：
    $$
    \begin{array}{c}\mathbf{Z}^{I} = I\left([O_{t + k},O_{t}]\right),\quad \pmb{Z}^{I}\in \mathbb{R}^{d}~\\ \mathbf{Z}^{I} = [\mathbf{Z}_{\alpha^{\prime}}^{I};\mathbf{Z}_{\upsilon^{\prime}}^{1}],\quad \pmb{Z}_{\alpha^{\prime}}^{I},\mathbf{Z}_{\upsilon^{\prime}}^{1}\in \mathbb{R}^{d / 2} \end{array} \quad (5)
    $$
    我们通过视觉头 (visual head) 将 $\pmb{Z}_{\upsilon'}$ 和 $\mathbf{Z}_{\upsilon'}^{\mathcal I}$ 投影到视觉空间，得到 $\mathbf{Z}_v$ 和 $\mathbf{Z}_{\upsilon}^I$：
    $$
    \begin{array}{rl} & {\mathbf{Z}_{v} = \mathrm{MLP}_{\mathrm{visual}}(\mathbf{Z}_{v^{\prime}}),\quad \mathbf{Z}_{v}\in \mathbb{R}^{d}}\\ & {\mathbf{Z}_{v}^{I} = \mathrm{MLP}_{\mathrm{visual}}(\mathbf{Z}_{v^{\prime}}^{I}),\quad \mathbf{Z}_{v}^{I}\in \mathbb{R}^{d}} \end{array} \quad (7)
    $$
    本文将逆向视觉表示 $\mathbf{Z}_v^I$ 视为正样本，构建了一个视觉中心对比学习目标，其中优化由一个视觉损失 $L_{\mathrm{visual}}$ 引导，该损失被实现为 InfoNCE [11] 损失：
    $$
    \mathcal{L}_{\mathrm{visual}} = -\sum_{i\in I}\log \frac{\exp(\tilde{Z}_{v,i}\cdot \tilde{Z}_{v,j}(i) / \tau)}{\sum_{a\in A(i)}\exp(\tilde{Z}_{v,i}\cdot \tilde{Z}_{v,a} / \tau)}. \quad (8)
    $$
    **符号解释：**
    *   $i \in I \equiv \{1, \dotsc, 2N\}$：表示批次中样本的索引。
    *   `j(i)`：是锚点样本 $i$ 对应的正样本的索引。
    *   $\tilde{Z}_v = [\mathbf{Z}_v; \mathbf{Z}_{\upsilon}^I] \in \mathbb{R}^{2N \times d}$：表示批次中包含 `2N` 个样本的连接视觉嵌入，其中 $\mathbf{Z}_{\upsilon} \in \mathbb{R}^{N \times d}$ 和 $\mathbf{Z}_{\upsilon}^{I} \in \mathbb{R}^{N \times d}$ 互为正样本对。
    *   $\tilde{Z}_{v,i}$：表示批次中第 $i$ 个样本的视觉嵌入。
    *   $\tau$：是一个标量温度参数。
        **目的：** 视觉中心对比目标鼓励模型捕捉内容一致且运动不变的特征。通过在运动扰动下对比视觉表示，视觉损失驱动模型将外观信息从动态变化中分离出来，从而促进视觉和运动表示的解耦。

    **总损失：** 第一阶段的总损失为重建损失、动作中心对比损失和视觉中心对比损失之和：
    $$
    L_{\mathrm{total}} = L_{\mathrm{MSE}} + L_{\mathrm{action}} + L_{\mathrm{visual}}
    $$

**算法 1：对比潜在动作学习**

```
1: 输入: V_unlabeled （无标签视频 (O_t, I_t) 对，即 (观察, 指令)）, Y_cls (动作类别标签), 编码器 I_phi, 解码器 F_psi
2: N_w: 预热更新步数
3: N_C: ConLA 更新步数
4: for iter = 1 to N_C do
5:    从 V_unlabeled 中采样 (O_t, O_{t+k}) 和 (O_{t+k}, O_t)
6:    Z = I_phi(O_t, O_{t+k}); [Z_a'; Z_v'] = Split(Z)
7:    Z_a = MLP_action(Z_a'); Z_v = MLP_action(Z_v')
8:    if iter < N_w then
9:        O_{t+k}_hat = F_psi(O_t, Z_a)
10:       L_total = L_MSE(phi, psi) = ||O_{t+k}_hat - O_{t+k}||^2
11:   else
12:       Z^I = I_phi(O_{t+k}, O_t); [Z_alpha'^I; Z_upsilon'^I] = Split(Z^I)
13:       Z_a^I = MLP_action(Z_alpha'^I); Z_v^I = MLP_visual(Z_upsilon'^I)
14:       O_{t+k}_hat = F_psi(O_t, Z_a)
15:       L_MSE(phi, psi) = ||O_{t+k}_hat - O_{t+k}||^2
16:       L_action = L_supContrast(Z_a', Y_cls) (Eq. 4)
17:       L_visual = L_infoNCE(Z_v, Z_v^I) (Eq. 8)
18:       L_total = L_MSE + L_action + L_visual
19:   end if
20:   更新模型参数 phi, psi
21: end for
```
<strong>预热阶段 (Warmup Phase)：</strong> 如算法 1 所示，在执行对比潜在动作学习之前，本文首先进行 5,000 步的预热阶段。在此期间，模型仅使用重建损失 $L_{\mathrm{MSE}}$ 进行优化。这是因为在训练初期，模型尚未学习到稳定的表示，此时应用对比学习可能会导致模型崩溃。通过首先优化重建损失，模型可以获得初步的潜在表示，然后这些表示将用于指导对比潜在动作学习，从而增强运动表示。

### 4.2.2. 阶段二：潜在动作预训练 (Latent Action Pretraining)
在此阶段，本文利用第一阶段训练好的潜在动作量化编码器作为逆动力学模型，从视频中提取潜在动作，这些动作作为伪标签 (pseudo-labels)。具体来说，对于每一对当前帧 $O_t$ 和未来帧 $O_{t+k}$，本文通过从动作中心码本中检索最近的量化表示来生成相应的潜在动作 $Z_{aq}$，从而构建一个包含观察-指令-伪动作标签三元组的数据集。

然后，本文在这个数据集上执行潜在动作预训练，方法是使用一个预训练的视觉-语言模型 (Vision-Language Model, VLM) 来预测 $Z_{aq}$，条件是任务指令和当前帧 $O_t$。本文遵循 LAPA [57] 的方法，在 VLM 的语言模型头 (language model head) 之后附加一个额外的潜在动作头 (latent action head)，该头部实现为一个单层 MLP，其词汇表大小为 $|C|$ (即码本大小)。在训练过程中，视觉编码器 (vision encoder) 被冻结 (frozen)，而语言模型的所有参数都被解冻 (unfrozen) 进行优化。本文的通用策略基于 7B Large World Model [30]。

### 4.2.3. 阶段三：动作微调 (Action Finetuning)
在第二阶段潜在动作预训练之后，视频中的运动先验已成功转移到策略中。然而，由此产生的潜在动作不能直接在下游机器人任务上执行，因为它们不对应于实际的末端执行器运动。为了将潜在动作映射到真实机器人动作，本文使用少量包含真实机器人动作 (ground-truth robot actions) 的轨迹对预训练策略进行微调。在动作预测过程中，本文将每个机器人维度的连续动作空间离散化。在微调阶段，原始的潜在动作头被丢弃，并替换为新的动作头以生成真实动作。与潜在动作预训练一致，视觉编码器被冻结，而底层语言模型的所有参数都被解冻以进行优化。

**算法 2：潜在动作预训练与动作微调**

```
1: 输入: 编码器 I_phi, D_labeled (真实动作轨迹 (O_t, I_t, A_t) 对，用于微调), 潜在动作策略 P_theta
2: V_unlabeled: 无标签视频 (O_t, I_t) 对 (观察, 指令)
3: N_P: 策略预训练更新步数
4: N_F: 策略微调更新步数

5: 潜在动作预训练
6: for iter = 1 to N_P do
7:    从伪标签数据集 D_pseudo (由 V_unlabeled 经过 I_phi 生成 Z_a^I) 中采样 (O_t, I_t, Z_a^I)
8:    Z_a_hat = P_theta(O_t, I_t)
9:    L_MSE(theta) = ||Z_a_hat - Z_a^I||^2
10:   更新策略参数 theta
11: end for

12: 动作微调
13: for iter = 1 to N_F do
14:   从真实动作轨迹数据集 D_labeled 中采样 (O_t, I_t, A_t)
15:   A_t_hat = P_theta(O_t, I_t)
16:   L_MSE(theta) = ||A_t_hat - A_t||^2
17:   更新策略参数 theta
18: end for
```

# 5. 实验设置

## 5.1. 数据集
本文在机器人视频数据集和人类视频数据集上预训练 VLM 策略。
### 5.1.1. 预训练数据集
1.  **BridgeV2 [49]:** 这是一个大规模机器人操控数据集，包含 60,096 条轨迹，涵盖 24 种环境。该数据集包含了多种技能，如抓取、放置、推动、清扫、堆叠和折叠。所有轨迹都配有自然语言指令。
    *   **数据预处理：** 对于 BridgeV2，本文将自然语言指令分为 80 个动作类别，形成第一阶段潜在动作学习中使用的动作类别标签。具体的数据预处理流程在 Appendix A.2 中描述。
2.  **Something-SomethingV2 [16]:** 这是一个包含 220,847 个视频剪辑的集合，记录了人类对日常物体执行预定义的基本动作。尽管该数据集不包含真实动作标签，但它提供了每个视频剪辑的预定义动作类别标签，共覆盖 174 个动作类别。

### 5.1.2. 微调数据集
*   **SimplerEnv [27]:** SimplerEnv [27] 旨在忠实反映真实世界策略的性能，通过模拟物理动力学和视觉外观。本文关注“WindowX + Bridge”设置中的四个任务：(1) 将勺子放在桌布上，(2) 将胡萝卜放在盘子上，(3) 将绿色方块堆叠在黄色方块上，(4) 将茄子放入篮子。由于 SimplerEnv [27] 缺乏微调轨迹，本文遵循 LAPA [57] 的实验设置，基于在 BridgeV2 数据集 [49] 上训练的 VLA 模型的成功推演 (rollouts) 收集了 100 条多任务轨迹。抓取对象的姿态和位置使用不同的随机种子进行随机初始化。
*   <strong>真实世界桌面操控实验 (Real-World Tabletop Manipulation):</strong> 实验使用 7 自由度 (7-DoF) Franka Research 3 机器人臂在三种环境中进行，配备第三视角 Realsense D435i RGB-D 摄像头（仅使用 RGB 图像）。模型在三个多指令任务上进行微调：(1) 击倒 <物体>，(2) 用毛巾覆盖 <物体>，(3) 捡起 <物体> 放入盒子。对于每个任务，收集 150 条轨迹。

### 5.1.3. 数据预处理（用于动作类别标签生成）
本文利用自然语言指令作为提取结构化动作类别标签的桥梁，因为指令与可执行动作类别高度相关。自然语言传达丰富的运动和空间语义，可以提炼成明确的动作类别信号，为潜在动作学习提供清晰的监督。这使得能够从没有真实标注的视频中自动生成动作类别标签，从而支持下游对比潜在动作学习或策略学习。数据预处理流程包括以下阶段：
1.  <strong>指令标准化 (Instruction normalization)：</strong> 所有指令转换为小写，并移除非字母数字字符。过滤掉包含连词（例如“and”）的句子，因为此类句子通常描述多个动作，这使得原子动作分类复杂化。
2.  <strong>动作提取 (Action extraction)：</strong> 使用 SpaCy (en_core_web_lg) 进行分词 (tokenization) 和词性标注 (part-of-speech tagging)。SpaCy 是一个高效的自然语言处理库，支持分词、词性标注和依存句法分析 (dependency parsing)。本文使用它来识别每条指令中的主要动词作为核心动作信息。
3.  <strong>空间方向映射 (Spatial Direction Mapping)：</strong> 方向性关键词（例如“top”、“left”、“in front of”）使用手动构建的词典映射到标准化的方向类别集。
4.  <strong>标签组合 (Label composition)：</strong> 每条指令表示为一个（动词，方向）对，形成一个离散的动作标签。
5.  <strong>数据清洗和类别整合 (Data cleaning and category consolidation)：</strong> 丢弃缺乏有效动词、语义模糊或文本内容不足的指令。样本数量低于最小阈值的类别合并到“uncertain”类别中。

## 5.2. 评估指标
### 5.2.1. 成功率 (Success Rate)
**概念定义：** 成功率是指在一定数量的尝试中，任务被完全成功完成的次数所占的比例。它衡量了策略在实现任务目标方面的整体有效性。
**数学公式：**
$$
\text{Success Rate} = \frac{\text{Number of Successful Trials}}{\text{Total Number of Trials}} \times 100\%
$$
**符号解释：**
*   $\text{Number of Successful Trials}$：成功完成任务的试验次数。
*   $\text{Total Number of Trials}$：总共进行的试验次数。

### 5.2.2. 抓取率 (Grasping Rate)
**概念定义：** 抓取率是指在需要抓取操作的任务中，机器人成功抓取目标对象的次数所占的比例。它衡量了策略在执行抓取动作方面的能力。
**数学公式：**
$$
\text{Grasping Rate} = \frac{\text{Number of Successful Grasps}}{\text{Total Number of Trials}} \times 100\%
$$
**符号解释：**
*   $\text{Number of Successful Grasps}$：成功抓取对象的试验次数。
*   $\text{Total Number of Trials}$：总共进行的试验次数。

### 5.2.3. 移动率 (Moving Rate)
**概念定义：** 移动率是指在需要移动对象操作的任务中，机器人成功将目标对象从起始位置移动到正确区域的次数所占的比例。它衡量了策略在执行移动动作方面的能力。
**数学公式：**
$$
\text{Moving Rate} = \frac{\text{Number of Successful Moves}}{\text{Total Number of Trials}} \times 100\%
$$
**符号解释：**
*   $\text{Number of Successful Moves}$：成功移动对象的试验次数。
*   $\text{Total Number of Trials}$：总共进行的试验次数。

### 5.2.4. 部分成功标准 (Partial Success Criterion) (仅用于真实世界机器人实验)
为了更细粒度地评估，本文采用了与 OpenVLA [24] 相同的任务特定部分成功标准。
*   **击倒 \<物体> (Knock <object> Over):**
    *   0.5 分：机器人触及正确物体。
    *   1 分：机器人成功击倒物体。
*   **用毛巾覆盖 \<物体> (Cover <object> with Towel):**
    *   0.33 分：机器人成功拿起毛巾。
    *   0.66 分：机器人触及正确物体并部分覆盖。
    *   1 分：机器人完全覆盖目标物体。
*   **捡起 \<物体> 并放入盒子 (Pick <object> into Box):**
    *   0.25 分：机器人触及正确物体。
    *   0.5 分：机器人成功抓取物体。
    *   0.75 分：抓取并将其移向盒子但未能成功放置。
    *   1 分：正确将物体放入盒子。

### 5.2.5. 泛化设置 (Generalization Settings) (仅用于真实世界机器人实验)
为了评估泛化能力，实验设计了三种不同的设置：
1.  <strong>未见物体组合 (Unseen Object Combination):</strong> 微调过程中见过的物体，但以新的组合方式出现。
2.  <strong>未见物体 (Unseen Object):</strong> 微调过程中完全未见的物体，这些物体在预训练阶段可能见过，也可能未见过。
3.  <strong>未见指令 (Unseen Instruction):</strong> 需要语义推理的新指令，这些指令在微调过程中未出现过。

## 5.3. 对比基线
本文选择以下模型作为对比基线：
1.  **UNIPI [15]:** 采用视频扩散模型 (video diffusion model) 进行语言条件下的推演生成 (language-conditioned rollout generation) 预训练，并使用逆动力学模型进行真实动作微调。
2.  **VPT [4] (Video PreTraining):** 在带标签数据上训练逆动力学模型，从视频中提取伪动作 (pseudo actions)，然后用于预训练 VLM。
3.  **LAPA [57]:** 使用朴素的 VQ-VAE [46] 从视频中学习潜在动作，并利用提取的潜在动作预训练 VLM。这是本文的主要比较对象，因为它代表了当前从人类视频中学习潜在动作的基线方法。
4.  **SCRATCH:** 仅在微调数据集上从头开始训练相同的骨干 VLM。这作为一个下限基线，用于评估预训练带来的收益。
5.  **ACTIONVLA:** 使用真实动作标注的机器人数据预训练相同的骨干 VLM。这可以被视为一个上限，因为它依赖于获取真实的动作标签。
6.  **UniVLA [9]:** (在 Appendix B.1 中提及) 利用 DINOv2 [37] 特征重建未来帧，以减轻环境噪声并构建以任务为中心的表示，从而增强潜在动作学习。为公平比较，UniVLA 的基础模型与 ConLA 和 LAPA 一致，均使用 Large World Model-7B [30]。

# 6. 实验结果与分析

## 6.1. 核心结果分析

### 6.1.1. SimplerEnv 评估
本文在 SimplerEnv [27] 上对策略进行评估。预训练策略使用了机器人视频（BridgeV2 [49]）和人类视频（Something-SomethingV2 [16]）。机器人视频通常在受控环境中收集，噪声较少但数量有限且昂贵；人类视频则数量庞大且易于获取，但环境噪声高，对潜在动作学习构成挑战。这项实验旨在评估 ConLA 在两种视频类型上的通用性，并检验改进的潜在动作表示能否缓解人类视频中的挑战，提高其效用，并促进运动先验向机器人操作任务的迁移。

以下是原文 Table 1 的结果：

<table><tr><td>Pretraining Data</td><td>Data Type</td><td>Policy</td><td>stack green <br>to yellow block</td><td>put carrot <br>on plate</td><td>put spoon <br>on towel</td><td>put eggplant <br>in basket</td><td>Average</td></tr><tr><td>-</td><td>-</td><td>SCRATCH <br>ACTIONVLA</td><td>29.2</td><td>29.2</td><td>50.0</td><td>29.2</td><td>34.4</td></tr><tr><td>BridgeV2 [49]</td><td>Robot Trajectories</td><td>UNITII [15] <br>VPT [4] <br>LAPA [57] <br>ConLA (ours)</td><td>2.7 <br>45.8 <br>54.2 <br>62.5</td><td>2.7 <br>37.5 <br>45.8 <br>45.8</td><td>0.0 <br>70.8 <br>70.8</td><td>0.0 <br>50.0 <br>58.3</td><td>1.3 <br>51.0 <br>57.3</td></tr><tr><td>BridgeV2 [49]</td><td>Robot Videos</td><td>UNITII [15] <br>VPT [4] <br>LAPA [57] <br>ConLA (ours)</td><td>0.0 <br>50.0 <br>50.0 <br>62.5</td><td>1.3 <br>29.1 <br>50.0 <br>50.0</td><td>1.3 <br>37.5 <br>50.0 <br>79.2</td><td>0.0 <br>66.6 <br>50.0 <br>58.3</td><td>0.7 <br>45.8 <br>52.1</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>64.6 (+12.5)</td></tr></table>

**结果与分析：**
如 Table 1 所示，ConLA 在 SimplerEnv 上以显著优势超越了所有基线方法。
*   **人类视频预训练的突破：** 最值得注意的是，<strong>仅在人类视频上进行预训练的 ConLA 甚至超越了在真实机器人轨迹上预训练的模型 (ACTIONVLA) 1.1%</strong>。这一结果是里程碑式的，它表明 ConLA 有能力高效利用人类视频中丰富多样的运动信息。
*   **对比 LAPA：** ConLA 在人类视频预训练下，相比 LAPA [57] (同样使用人类视频预训练) 的平均成功率提高了 12.5% (64.6% vs 52.1%)，这验证了 ConLA 提出的对比解耦机制在应对人类视频复杂性方面的有效性。
*   **人类视频潜力：** 传统范式未能有效应对人类视频数据的挑战：尽管人类视频规模大于机器人数据集且包含更丰富多样的运动信息，但预训练性能却不如预期。ConLA 显著提高了人类视频的利用效率，为未来释放其全部潜力铺平了道路。

### 6.1.2. 真实世界结果
本文在真实世界机器人上评估了 ConLA 的性能。预训练策略同样使用了 BridgeV2 [49] 和 Something-SomethingV2 [16]，随后使用少量真实机器人轨迹进行微调。

下图（原文 Figure 4）展示了真实世界操控机器人结果：

![fig 5](images/5.jpg)
**结果与分析：**

*   Figure 4 展示了 ConLA 在三个任务和三种泛化设置（未见物体组合、未见物体、未见指令）下的真实机器人性能。
*   **视频预训练的价值：** LAPA [57] 和 ConLA 都优于 SCRATCH 基线，这验证了视频预训练的价值。
*   **人类视频的挑战与 ConLA 的优势：** LAPA [57] 在人类视频预训练下，相比机器人视频预训练几乎没有优势，这表明尽管人类视频多样且规模大，但其领域复杂性和分布偏移 (distribution shift) 阻碍了有效利用。相比之下，ConLA 不仅进一步提升了 BridgeV2 [49] 预训练的性能，更重要的是，在人类视频预训练下取得了显著的性能提升，**超越 LAPA [57] 达 15.9%**。这归因于 ConLA 能够提取语义一致的潜在动作，从而忠实地从人类视频中获取运动先验。
*   **泛化能力：** 如 Table 2 所示，LAPA [57] 和 ConLA 在人类视频预训练下，尤其在未见物体设置中展现出强大的泛化能力，这得益于大规模人类视频数据集中更广泛的物体多样性。ConLA 进一步显著提升了这种泛化能力。

    以下是原文 Table 2 的结果：

    <table><tr><td>Method</td><td>Absent Opt. <br>Unseen Combo</td><td>Unseen Obj.</td><td>Absent Obj. <br>Unseen Instruct.</td><td>AVG</td></tr><tr><td>SCRATCH</td><td>18.4</td><td>10.5</td><td>17.1</td><td>15.3</td></tr><tr><td>LAPA (Bridge)</td><td>36.0</td><td>22.1</td><td>35.6</td><td>31.2</td></tr><tr><td>ConLA (Bridge)</td><td>46.2</td><td>25.4</td><td>37.8</td><td>36.5</td></tr><tr><td>LAPA (Human Videos)</td><td>36.0</td><td>25.8</td><td>35.1</td><td>32.3</td></tr><tr><td>ConLA (Human Videos)</td><td>59.1</td><td>47.2</td><td>38.3</td><td>48.2</td></tr></table>

总体而言，这些结果突显了人类视频预训练的可扩展潜力，并证明 ConLA 显著增强了人类运动先验向机器人控制的迁移。

## 6.2. 潜在动作分析

### 6.2.1. 捷径学习分析
为了评估 ConLA 在缓解潜在动作提取过程中捷径学习的有效性，本文进行了定性可视化分析。具体来说，从一对视频片段中提取了四个代表性的潜在动作：向下、向上、向左和向右。然后，将这些提取的潜在动作应用于重建其他图像的帧，旨在评估学习到的潜在动作是否能控制不同视觉上下文下的运动生成。

下图（原文 Figure 5）展示了潜在动作分析：

![fig 6](images/6.jpg)
*该图像是一个示意图，展示了从人类视频中提取的潜在动作。每一列展示了不同的方法，分别为输入、LAPA、我们的方法及真实标签（GT），涵盖了向下、向上、向左和向右的操作示例。*

Figure 5. 潜在动作分析。潜在动作提取中捷径学习的可视化。以提取的潜在动作为条件重建图像，表明本文方法捕捉了与运动相关的动作，缓解了捷径学习。

**结果与分析：**
在分析中，本文使用当前帧（输入）作为条件，并根据提取的潜在动作生成预测的未来帧。本文将 ConLA 的重建结果与朴素 LAPA [57] 基线进行了比较。
*   **LAPA 的捷径学习：** 如 Figure 5 所示，在人类视频中，LAPA [57] 存在严重的捷径学习问题。提取的潜在动作由视觉内容主导（如第一列右侧图像所示），而不是运动语义。在机器人视频中，提取的潜在动作也表现出语义不一致性。这表明 LAPA 学习到的潜在动作编码了视觉外观信息，而非真正的运动动态。
*   **ConLA 的有效性：** 相比之下，ConLA 捕捉了具有运动意义的潜在动作，真正代表了底层的动态。这些结果表明，本文的方法有效缓解了捷径学习，并学习到语义一致的潜在动作表示。

### 6.2.2. 潜在动作表示分析
为了分析潜在动作表示空间的结构，本文从每个动作类别中随机采样 100 个视频片段，并提取它们的潜在动作嵌入，然后使用 t-SNE 进行可视化。

下图（原文 Figure 6）展示了潜在动作嵌入的 t-SNE 可视化：

![fig 8](images/8.jpg)
*该图像是一个对比图，展示了两种不同的潜在动作表示方法：左侧为 LAPA 方法的结果，右侧为 ConLA 方法的结果。左图的点分布较为分散，而右图中的点聚集更为明显，显示了 ConLA 在动作中心潜在表示方面的优势。*

Figure 6. 潜在动作嵌入的 t-SNE 可视化显示，本文方法产生了语义一致且紧凑的表示，相同类别的动作形成紧密聚类。

**结果与分析：**
*   **LAPA 的纠缠表示：** 如 Figure 6 左图所示，由朴素 VQ-VAE [46] 获得的潜在动作空间在不同动作类别之间是混乱且纠缠的。相似的运动可能由于视觉外观差异而被分离，导致表示空间缺乏结构。
*   **ConLA 的紧凑表示：** 相比之下，ConLA 生成了一个更紧凑和语义连贯的潜在动作空间（如 Figure 6 右图所示）。具有相同底层动态的相似运动不再因视觉外观差异而分离。这种表示能够更忠实地将人类运动先验迁移到机器人训练中，从而提高了利用人类视频数据进行机器人学习的效率。

## 6.3. 消融实验/参数分析

### 6.3.1. 对比解耦模块 (Contrastive Disentanglement Module)
为了评估对比解耦过程中每个组件的贡献，本文在第一阶段潜在动作学习中，使用 Something-SomethingV2 [16] 数据集进行了消融研究，并在 SimplerEnv [27] 基准上验证了性能，以平均任务成功率作为评估指标。

以下是原文 Table 3 的结果：

<table><tr><td>Method</td><td>Avg.</td></tr><tr><td>LAPA (base)</td><td>52.1</td></tr><tr><td>+ Action contrast</td><td>58.4</td></tr><tr><td>+ Action + Visual contrast (w/o inv. aug.)</td><td>57.3</td></tr><tr><td>Full ConLA</td><td>64.6</td></tr></table>

**结果与分析：**
*   **动作中心对比学习的贡献：** 以 LAPA [57] 作为基线（平均成功率 52.1%），引入**动作中心对比学习**（`+ Action contrast`）显著提升了潜在动作表示，平均成功率达到 58.4%。这表明利用动作类别标签的弱监督对于学习更具判别性的潜在表示至关重要。
*   **视觉中心对比学习和时间反转增强的重要性：**
    *   当进一步引入**视觉中心对比学习**但移除**时间反转增强**（`+ Action + Visual contrast (w/o inv. aug.)`）时，性能略有下降，平均成功率为 57.3%。这说明如果没有时间反转，动作和视觉嵌入变得更加相似，导致表示纠缠。
    *   在**完整 ConLA**（`Full ConLA`）中，重新引入时间反转增强，性能大幅提升至 64.6%。这证实了时间反转增强在保持动作和视觉特征之间清晰分离方面的关键作用，从而带来了额外的性能提升。

### 6.3.2. 数据可扩展性 (Data scalability)
为了评估 ConLA 在人类演示视频数据集上的扩展能力，本文在 Something-SomethingV2 [16] 数据集上进行了实验。具体来说，本文使用不同比例的数据集（从 10% 到 100%）预训练模型，以检查性能如何随数据量的增加而扩展。同时与 LAPA [57] 基线进行了比较。

以下是原文 Table 4 的结果：

<table><tr><td>Method</td><td>10% Data</td><td>50% Data</td><td>100% Data</td></tr><tr><td>LAPA</td><td>50.0</td><td>51.0</td><td>52.1</td></tr><tr><td>ConLA</td><td>58.3</td><td>60.4</td><td>64.6</td></tr></table>

**结果与分析：**
*   **性能随数据量正向扩展：** 结果表明，性能与数据量呈正相关，即随着预训练数据量的增加，ConLA 的性能持续提升。
*   **ConLA 的数据效率：** 相比基线 LAPA [57]，ConLA 在所有数据规模下都表现出更强的性能。特别是在 10% 数据量时，ConLA (58.3%) 已经显著优于 LAPA 使用 100% 数据量时的性能 (52.1%)，这表明 ConLA 能够更有效地利用数据，甚至在数据量较少的情况下也能提取更高质量的潜在动作。

## 6.4. 其他详细实验结果

### 6.4.1. SimplerEnv BridgeV2 预训练结果
以下是原文 Table 6 的结果：

<table><thead><tr><th>Success Rate</th><th>Scratch</th><th>UNIPI</th><th>VPT</th><th>LAPA</th><th>UniVLA*</th><th>ConLA</th><th>ActionVLA</th></tr></thead><tbody><tr><td>StackG2Y</td><td>29.2</td><td>2.7</td><td>45.8</td><td>54.2</td><td>41.7</td><td>62.5</td><td>75.0</td></tr><tr><td>Carrot2Plate</td><td>29.2</td><td>2.7</td><td>37.5</td><td>45.8</td><td>45.8</td><td>45.8</td><td>58.0</td></tr><tr><td>Spoon2Towel</td><td>50.0</td><td>0.0</td><td>70.8</td><td>70.8</td><td>75.0</td><td>75.0</td><td>70.8</td></tr><tr><td>Eggplant2Bask</td><td>29.2</td><td>0.0</td><td>50.0</td><td>58.3</td><td>62.5</td><td>58.3</td><td>50.0</td></tr><tr><td>AVG</td><td>34.4</td><td>1.3</td><td>51.0</td><td>57.3</td><td>56.2</td><td>60.4</td><td>63.5</td></tr><tr><th>Grasping Rate</th><th>Scratch</th><th>UNIPI</th><th>VPT</th><th>LAPA</th><th>UniVLA*</th><th>ConLA</th><th>ActionVLA</th></tr><tr><td>Grasp Green Block</td><td>66.6</td><td>20.8</td><td>62.5</td><td>62.5</td><td>58.3</td><td>62.5</td><td>87.5</td></tr><tr><td>Grasp Carrot</td><td>45.8</td><td>33.2</td><td>54.1</td><td>58.3</td><td>46.8</td><td>45.8</td><td>75.0</td></tr><tr><td>Grasp Spoon</td><td>70.8</td><td>22.2</td><td>79.2</td><td>83.3</td><td>75.0</td><td>75.0</td><td>83.3</td></tr><tr><td>Grasp Eggplant</td><td>62.5</td><td>16.0</td><td>70.8</td><td>83.3</td><td>79.2</td><td>75.0</td><td>75.0</td></tr><tr><td>AVG</td><td>61.4</td><td>23.1</td><td>66.7</td><td>71.9</td><td>64.8</td><td>64.6</td><td>80.2</td></tr><tr><th>Moving Rate</th><th>Scratch</th><th>UNIPI</th><th>VPT</th><th>LAPA</th><th>UniVLA*</th><th>ConLA</th><th>ActionVLA</th></tr><tr><td>Move Green Block</td><td>58.3</td><td>29.1</td><td>58.3</td><td>66.6</td><td>58.3</td><td>62.5</td><td>91.6</td></tr><tr><td>Move Carrot</td><td>45.8</td><td>48.6</td><td>66.6</td><td>75.0</td><td>50.0</td><td>54.2</td><td>91.6</td></tr><tr><td>Move Spoon</td><td>70.8</td><td>34.6</td><td>79.2</td><td>83.3</td><td>75.0</td><td>75.0</td><td>79.2</td></tr><tr><td>Move Eggplant</td><td>87.5</td><td>58.0</td><td>70.8</td><td>87.5</td><td>79.2</td><td>83.3</td><td>91.6</td></tr><tr><td>AVG</td><td>65.6</td><td>42.6</td><td>68.7</td><td>77.1</td><td>65.6</td><td>68.8</td><td>88.5</td></tr></tbody></table>

### 6.4.2. SimplerEnv 人类操控视频预训练结果
以下是原文 Table 7 的结果：

<table><thead><tr><th>Success Rate</th><th>Scratch</th><th>UNIPI</th><th>VPT</th><th>LAPA</th><th>UniVLA*</th><th>ConLA</th></tr></thead><tbody><tr><td>StackG2Y</td><td>29.2</td><td>0.0</td><td>50.0</td><td>50.0</td><td>62.5</td><td>62.5</td></tr><tr><td>Carrot2Plate</td><td>29.2</td><td>1.3</td><td>29.1</td><td>50.0</td><td>37.5</td><td>50.0</td></tr><tr><td>Spoon2Towel</td><td>50.0</td><td>1.3</td><td>37.5</td><td>50.0</td><td>70.8</td><td>79.2</td></tr><tr><td>Eggplant2Bask</td><td>29.2</td><td>0.0</td><td>66.6</td><td>58.3</td><td>50.0</td><td>66.6</td></tr><tr><td>AVG</td><td>34.4</td><td>0.7</td><td>45.8</td><td>52.1</td><td>55.2</td><td>64.6</td></tr><tr><th>Grasping Rate</th><th>Scratch</th><th>UNIPI</th><th>VPT</th><th>LAPA</th><th>-</th><th>ConLA</th></tr><tr><td>Grasp Green Block</td><td>66.6</td><td>2.7</td><td>66.6</td><td>58.3</td><td>66.7</td><td>62.5</td></tr><tr><td>Grasp Carrot</td><td>45.8</td><td>31.7</td><td>45.8</td><td>62.5</td><td>45.8</td><td>45.8</td></tr><tr><td>Grasp Spoon</td><td>70.8</td><td>21.7</td><td>70.8</td><td>75.0</td><td>75.0</td><td>87.5</td></tr><tr><td>Grasp Eggplant</td><td>62.5</td><td>6.8</td><td>91.6</td><td>70.8</td><td>62.5</td><td>75.0</td></tr><tr><td>AVG</td><td>61.4</td><td>15.7</td><td>68.7</td><td>66.7</td><td>62.5</td><td>67.7</td></tr><tr><th>Moving Rate</th><th>Scratch</th><th>UNIPI</th><th>VPT</th><th>LAPA</th><th>UniVLA*</th><th>ConLA</th></tr><tr><td>Move Green Block</td><td>58.3</td><td>2.7</td><td>62.5</td><td>62.5</td><td>62.5</td><td>62.5</td></tr><tr><td>Move Carrot</td><td>45.8</td><td>37.5</td><td>58.3</td><td>70.8</td><td>54.2</td><td>58.3</td></tr><tr><td>Move Spoon</td><td>70.8</td><td>18.1</td><td>54.1</td><td>75.0</td><td>83.3</td><td>87.5</td></tr><tr><td>Move Eggplant</td><td>87.5</td><td>50.3</td><td>91.6</td><td>93.3</td><td>75.0</td><td>79.2</td></tr><tr><td>AVG</td><td>65.6</td><td>27.1</td><td>66.6</td><td>72.9</td><td>68.8</td><td>71.9</td></tr></tbody></table>

**UniVLA* 结果分析：**
在 Appendix B.1 中，作者提供了 UniVLA [9] 的额外结果。UniVLA [9] 通过重建 DINOv2 [37] 特征来减轻环境噪声并构建以任务为中心的表示。
*   在 BridgeV2 [49] 预训练下 (Table 6)，UniVLA* 取得了与 LAPA [57] (56.2% vs 57.3%) 相当的性能。
*   在人类视频预训练下 (Table 7)，UniVLA* (55.2%) 相比 LAPA [57] (52.1%) 有了显著提升，表明 UniVLA 在复杂人类演示环境下提取高质量潜在动作的有效性。
*   然而，由于缺乏显式归纳偏置，UniVLA 仍然容易受到不相关视觉信息的干扰，这限制了其进一步提升性能的能力。ConLA (64.6%) 再次显著超越 UniVLA* (55.2%)，进一步验证了其对比解耦机制的优越性。

### 6.4.3. 真实世界任务详细结果
以下是原文 Table 8 的结果：

<table>
<thead>
<tr>
<td></td>
<td>Scratch</td>
<td>LAPA (Bridge)</td>
<td>ConLA (Bridge)</td>
<td>LAPA (Sthv2)</td>
<td>ConLA (Sthv2)</td>
</tr>
</thead>
<tbody>
<tr>
<td colspan="6">Seen Objects, Unseen Object Combinations</td>
</tr>
<tr>
<td>bottle</td>
<td>0.5</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>1</td>
</tr>
<tr>
<td>chocolate</td>
<td>0</td>
<td>0</td>
<td>1</td>
<td>0.5</td>
<td>1</td>
</tr>
<tr>
<td>crisp</td>
<td>0</td>
<td>0.5</td>
<td>0.5</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<td>cocacola</td>
<td>0.5</td>
<td>0</td>
<td>0.5</td>
<td>0.5</td>
<td>0</td>
</tr>
<tr>
<td>pie</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0.5</td>
<td>0.5</td>
</tr>
<tr>
<td>pocky</td>
<td>0.5</td>
<td>1</td>
<td>1</td>
<td>1</td>
<td>1</td>
</tr>
<tr>
<td>SUM</td>
<td>1.5</td>
<td>1.5</td>
<td>3</td>
<td>2.5</td>
<td>3.5</td>
</tr>
<tr>
<td colspan="6">Unseen Objects</td>
</tr>
<tr>
<td>pepsi</td>
<td>0</td>
<td>0</td>
<td>1</td>
<td>1</td>
<td>1</td>
</tr>
<tr>
<td>conditioner</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<td>CALPIS</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<td>grey-chocolate</td>
<td>0</td>
<td>1</td>
<td>0</td>
<td>0</td>
<td>0.5</td>
</tr>
<tr>
<td>milk-tea</td>
<td>0</td>
<td>0</td>
<td>0.5</td>
<td>0</td>
<td>1</td>
</tr>
<tr>
<td>shampoo</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<td>SUM</td>
<td>0</td>
<td>1</td>
<td>1.5</td>
<td>1</td>
<td>2.5</td>
</tr>
<tr>
<td colspan="6">Seen Objects, Unseen Instructions</td>
</tr>
<tr>
<td>pillared object</td>
<td>0</td>
<td>0</td>
<td>1</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<td>red-packed food</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0.5</td>
<td>1</td>
</tr>
<tr>
<td>white-bagged snacks</td>
<td>0</td>
<td>1</td>
<td>1</td>
<td>0.5</td>
<td>0</td>
</tr>
<tr>
<td>carbonated drinks</td>
<td>0.5</td>
<td>1</td>
<td>0.5</td>
<td>1</td>
<td>1</td>
</tr>
<tr>
<td>cookie box</td>
<td>0.5</td>
<td>1</td>
<td>0</td>
<td>1</td>
<td>1</td>
</tr>
<tr>
<td>rectangle object</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0.5</td>
</tr>
<tr>
<td>SUM</td>
<td>1</td>
<td>3</td>
<td>2.5</td>
<td>3</td>
<td>3.5</td>
</tr>
<tr>
<td>Success Rate (Strict)</td>
<td>0%</td>
<td>33.33%</td>
<td>27.78%</td>
<td>27.78 %</td>
<td>44.44%</td>
</tr>
<tr>
<td>Success Rate</td>
<td>13.89%</td>
<td>30.56%</td>
<td>38.89%</td>
<td>36.11%</td>
<td>52.78%</td>
</tr>
<tr>
<td>Reaching Success Rate</td>
<td>27.78%</td>
<td>33.33%</td>
<td>50%</td>
<td>50%</td>
<td>61.11%</td>
</tr>
</tbody>
</table>

以下是原文 Table 9 的结果：

<table>
<thead>
<tr>
<td></td>
<td>Scratch</td>
<td>LAPA (Bridge)</td>
<td>ConLA (Bridge)</td>
<td>LAPA (Sthv2)</td>
<td>ConLA (Sthv2)</td>
</tr>
</thead>
<tbody>
<tr>
<td colspan="6">Seen Objects, Unseen Object Combinations</td>
</tr>
<tr>
<td>banana</td>
<td>0.33</td>
<td>0.33</td>
<td>0.66</td>
<td>0.33</td>
<td>0.66</td>
</tr>
<tr>
<td>peanut</td>
<td>0</td>
<td>0.33</td>
<td>0.33</td>
<td>0.33</td>
<td>0.33</td>
</tr>
<tr>
<td>pepper</td>
<td>0.33</td>
<td>0.33</td>
<td>0.33</td>
<td>0.33</td>
<td>0.66</td>
</tr>
<tr>
<td>cabbage</td>
<td>0.33</td>
<td>0.33</td>
<td>0.66</td>
<td>0.66</td>
<td>1</td>
</tr>
<tr>
<td>purple-block</td>
<td>0</td>
<td>0.66</td>
<td>0.33</td>
<td>0.33</td>
<td>0.33</td>
</tr>
<tr>
<td>red-block</td>
<td>0.33</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>0.66</td>
</tr>
<tr>
<td>SUM</td>
<td>1.32</td>
<td>1.98</td>
<td>3.31</td>
<td>1.98</td>
<td>3.64</td>
</tr>
<tr>
<td colspan="6">Unseen Objects</td>
</tr>
<tr>
<td>strawberry</td>
<td>0.66</td>
<td>0.66</td>
<td>0.33</td>
<td>0.33</td>
<td>1</td>
</tr>
<tr>
<td>potato</td>
<td>0.33</td>
<td>0</td>
<td>0.33</td>
<td>0.33</td>
<td>0.33</td>
</tr>
<tr>
<td>heart-shaped block</td>
<td>0.33</td>
<td>0.33</td>
<td>0.33</td>
<td>0.66</td>
<td>0.33</td>
</tr>
<tr>
<td>oval block</td>
<td>0</td>
<td>0.33</td>
<td>0.66</td>
<td>1</td>
<td>1</td>
</tr>
<tr>
<td>knife</td>
<td>0.33</td>
<td>0.66</td>
<td>0</td>
<td>1</td>
<td>1</td>
</tr>
<tr>
<td>bowl</td>
<td>0</td>
<td>0</td>
<td>0.66</td>
<td>0.33</td>
<td>0.33</td>
</tr>
<tr>
<td>SUM</td>
<td>1.65</td>
<td>1.98</td>
<td>2.31</td>
<td>2.65</td>
<td>3.99</td>
</tr>
<tr>
<td colspan="6">Seen Objects, Unseen Instructions</td>
</tr>
<tr>
<td>yellow fruit</td>
<td>0.33</td>
<td>0</td>
<td>0.33</td>
<td>0.33</td>
<td>0.66</td>
</tr>
<tr>
<td>green vegetable</td>
<td>0.33</td>
<td>0.33</td>
<td>0.66</td>
<td>0.33</td>
<td>0.66</td>
</tr>
<tr>
<td>nut</td>
<td>0</td>
<td>0.33</td>
<td>0.33</td>
<td>0.66</td>
<td>0.33</td>
</tr>
<tr>
<td>spicy vegetable</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<td>rectangle object</td>
<td>0.33</td>
<td>0.66</td>
<td>0.33</td>
<td>0.33</td>
<td>0.33</td>
</tr>
<tr>
<td>polygonal block</td>
<td>0.33</td>
<td>0.33</td>
<td>0.66</td>
<td>0.66</td>
<td>0.66</td>
</tr>
<tr>
<td>SUM</td>
<td>1.32</td>
<td>1.65</td>
<td>2.31</td>
<td>2.31</td>
<td>2.64</td>
</tr>
<tr>
<td>Success Rate (Strict)</td>
<td>0%</td>
<td>5.5%</td>
<td>5.5%</td>
<td>11.11%</td>
<td>22.22%</td>
</tr>
<tr>
<td>Success Rate</td>
<td>23.83%</td>
<td>36.72%</td>
<td>44.06%</td>
<td>38.56%</td>
<td>57.06%</td>
</tr>
<tr>
<td>Reaching Success Rate</td>
<td>5.56%</td>
<td>27.78%</td>
<td>38.89%</td>
<td>33.33%</td>
<td>50%</td>
</tr>
</tbody>
</table>

以下是原文 Table 10 的结果：

<table>
<thead>
<tr>
<td></td>
<td>Scratch</td>
<td>LAPA (Bridge)</td>
<td>ConLA (Bridge)</td>
<td>LAPA (Sthv2)</td>
<td>ConLA (Sthv2)</td>
</tr>
</thead>
<tbody>
<tr>
<td colspan="6">Seen Objects, Unseen Object Combinations</td>
</tr>
<tr>
<td>apple</td>
<td>0.25</td>
<td>0.25</td>
<td>0.25</td>
<td>0.25</td>
<td>0.5</td>
</tr>
<tr>
<td>bean</td>
<td>0</td>
<td>1</td>
<td>0.75</td>
<td>0.75</td>
<td>1</td>
</tr>
<tr>
<td>cabbage</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0.75</td>
</tr>
<tr>
<td>carrot</td>
<td>0</td>
<td>0.75</td>
<td>1</td>
<td>1</td>
<td>1</td>
</tr>
<tr>
<td>mango</td>
<td>0.25</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0.25</td>
</tr>
<tr>
<td>peanut</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<td>SUM</td>
<td>0.5</td>
<td>2</td>
<td>2</td>
<td>2</td>
<td>3.5</td>
</tr>
<tr>
<td colspan="6">Unseen Objects</td>
</tr>
<tr>
<td>tomato</td>
<td>0</td>
<td>0.25</td>
<td>0.25</td>
<td>0.5</td>
<td>1</td>
</tr>
<tr>
<td>peach</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<td>avocado</td>
<td>0</td>
<td>0.25</td>
<td>0.25</td>
<td>0.25</td>
<td>0.25</td>
</tr>
<tr>
<td>banana</td>
<td>0.25</td>
<td>0</td>
<td>0</td>
<td>0.25</td>
<td>0.5</td>
</tr>
<tr>
<td>purple-block</td>
<td>0</td>
<td>0.25</td>
<td>0</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<td>red-block</td>
<td>0</td>
<td>0.25</td>
<td>0.25</td>
<td>0</td>
<td>0.25</td>
</tr>
<tr>
<td>SUM</td>
<td>0.25</td>
<td>1</td>
<td>0.75</td>
<td>1</td>
<td>2</td>
</tr>
<tr>
<td colspan="6">Seen Objects, Unseen Instructions</td>
</tr>
<tr>
<td>an object that is red</td>
<td>0.55</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<td>an object that is green</td>
<td>0</td>
<td>0.25</td>
<td>0.5</td>
<td>0</td>
<td>0.25</td>
</tr>
<tr>
<td>an object that is a vegetable</td>
<td>0</td>
<td>1</td>
<td>1</td>
<td>0.5</td>
<td>0.25</td>
</tr>
<tr>
<td>an object that is orange</td>
<td>0.25</td>
<td>0.5</td>
<td>0.25</td>
<td>0.5</td>
<td>0.25</td>
</tr>
<tr>
<td>an object that is yellow</td>
<td>0</td>
<td>0</td>
<td>0.25</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<td>nut</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<td>SUM</td>
<td>0.75</td>
<td>1.75</td>
<td>2</td>
<td>1</td>
<td>0.75</td>
</tr>
<tr>
<td>Success Rate (Strict)</td>
<td>0%</td>
<td>11.11%</td>
<td>11.11%</td>
<td>5.6%</td>
<td>16.67%</td>
</tr>
<tr>
<td>Success Rate</td>
<td>8.33%</td>
<td>26.39%</td>
<td>26.39%</td>
<td>22.22%</td>
<td>34.72%</td>
</tr>
<tr>
<td>Reaching Success Rate</td>
<td>27.78%</td>
<td>55.56%</td>
<td>55.56%</td>
<td>44.44%</td>
<td>66.67%</td>
</tr>
</tbody>
</table>

以下是原文 Table 11 的结果：

<table><thead><tr><td></td><td>Scratch</td><td>LAPA(Bridge)</td><td>ConLA (Bridge)</td><td>LAPA (Sthv2)</td><td>ConLA (Sthv2)</td></tr></thead><tbody><tr><td>Total Success Rate</td><td>15.35%</td><td>31.22%</td><td>36.45%</td><td>32.30%</td><td>48.18%</td></tr><tr><td>Total Success Rate (Strict)</td><td>0%</td><td>14.80%</td><td>14.80%</td><td>14.83%</td><td>27.78%</td></tr></tbody></table>

**真实世界结果总结分析：**
*   **预训练的有效性：** 所有预训练模型（LAPA 和 ConLA，无论是使用 BridgeV2 还是 Something-SomethingV2 数据）都显著优于 `Scratch` 基线，再次证明了视频预训练对于机器人学习的价值。
*   **人类视频的优势：** 在“未见物体”设置中，无论是 LAPA 还是 ConLA，使用人类视频 (Something-SomethingV2) 进行预训练都持续优于使用 BridgeV2 预训练。这支持了人类视频数据集中丰富的物体多样性有助于提高模型对新物体的泛化能力。例如，在 Table 2 中，“Unseen Obj.”列下，LAPA (Human Videos) 为 25.8%，ConLA (Human Videos) 为 47.2%，均高于 LAPA (Bridge) 的 22.1% 和 ConLA (Bridge) 的 25.4%。
*   **ConLA 的卓越性能：** ConLA 在所有任务和泛化设置下都表现出领先的性能，尤其是在人类视频预训练下。例如，在 Table 11 中，ConLA (Sthv2) 的总成功率为 48.18%，远高于 LAPA (Sthv2) 的 32.30%。在严格成功率方面，ConLA (Sthv2) (27.78%) 也几乎是 LAPA (Sthv2) (14.83%) 的两倍。这再次归因于 ConLA 能够从人类视频中提取更高质量的潜在动作，从而更有效地将运动先验迁移到下游策略中。

## 6.5. 更多可视化分析
下图（原文 Figure 8）展示了潜在动作一致性可视化分析：

![fig 3](images/3.jpg)
*该图像是一个示意图，展示了如何从输入视频中提取潜在动作。图中比较了四种方法的结果，包括输入视频、LAPA、我们的算法和真实样本（GT），分别对应不同的动作方向：下、上、左和右。*

Figure 8. 潜在动作一致性可视化分析。

**结果与分析：**
Figure 8 进一步展示了潜在动作一致性的可视化。
*   **LAPA 的不一致性：** 结果清晰地表明 LAPA [57] 存在显著的潜在动作不一致性，尤其是在人类视频数据上训练时，捷径学习更容易发生。例如，在人类视频部分，当提取“左”和“右”运动的潜在动作时，LAPA 的重建结果无意中复制了用于提取潜在动作的帧的视觉内容，这直接表明提取的表示编码的是视觉外观而非运动。
*   **ConLA 的一致性：** 相比之下，ConLA 成功提取了以运动为中心的潜在动作，并重建出预期的运动结果，而没有泄露外观信息。
*   **机器人视频场景：** 即使在视觉噪声较少的机器人视频场景中，LAPA [57] 仍然表现出潜在动作不一致性。例如，在第一行，当提取“水平向下”运动时，LAPA 错误地捕捉到“垂直向下”运动；而 ConLA 则正确捕捉到“水平向下”方向。在第二行，LAPA 重建出“左上”运动，而 ConLA 更准确地提取出“向上”运动，表明意图动作与提取动作之间更好的对齐。
    这些可视化进一步强化了 ConLA 在缓解捷径学习和学习语义一致、运动相关潜在动作方面的优势。

# 7. 总结与思考

## 7.1. 结论总结
本文提出了 ConLA，一个简单而有效的方法，用于从人类演示视频中为视觉-语言-动作 (VLA) 模型提取高质量的潜在动作。ConLA 通过引入对比潜在动作学习机制，结合动作类别先验和时间先验来构建以动作为中心的表示。这种方法成功缓解了捷径学习问题，并产生了鲁棒的潜在动作。大量的实验结果表明，ConLA 始终优于现有方法，即使仅在人类视频数据上进行预训练。值得注意的是，ConLA 首次在仅使用人类视频进行预训练的情况下，超越了通过真实机器人轨迹预训练所获得的性能。这些激动人心的结果充分证明了大规模人类视频预训练在 VLA 领域的强大潜力和可行性。

## 7.2. 局限性与未来工作
本文作者指出的主要局限性在于当前数据预处理流程对动作类别标签的生成依赖于一个相对简单且半自动化的流水线。虽然这种流水线生成的伪动作类别标签足够稳定和连贯以支持对比潜在动作学习，但它仍有改进空间。

基于此，作者提出了未来的研究方向：
*   **自动化和细粒度动作类别标签提取：** 在未来的工作中，将研究更自动化的方法，从视频和自然语言指令中提取更细粒度的动作类别标签。目标是进一步提高对比潜在动作学习的性能。

## 7.3. 个人启发与批判
### 7.3.1. 个人启发
*   **先验知识的重要性：** ConLA 的成功强调了在无监督或弱监督学习中，巧妙地利用领域特定先验知识的重要性。动作类别先验（通过弱监督的对比学习）和时间先验（通过逆序增强的视觉对比学习）是其解决捷径学习和表示纠缠的关键。这启发我们，在设计模型时，除了强大的神经网络架构，如何将人类对问题本质的理解融入到学习目标中，是提升模型性能和泛化能力的重要途径。
*   **人类视频的巨大潜力：** 能够仅通过人类视频预训练就超越真实机器人轨迹预训练的性能，这一结果是令人振奋的。它表明人类视频作为一种廉价、大规模且多样化的数据源，其潜力远未被完全挖掘。ConLA 为如何有效利用这些数据提供了一个强有力的范例，对于未来通用型机器人学习的数据规模化具有重要意义。
*   **解耦表示的价值：** 学习解耦的表示对于可迁移性和泛化能力至关重要。当动作表示能够独立于具体的视觉背景时，机器人才能将从人类演示中学习到的运动知识泛化到新的物体、环境甚至机器人本体。ConLA 在这一点上的突破，为机器人从人类示教中学习提供了更坚实的基础。

### 7.3.2. 批判与潜在改进
*   **动作类别先验的依赖性：** 尽管 ConLA 采用了一种自动化的方式从指令中提取动作类别标签，但其性能仍可能受限于这些预定义类别集的粒度和完整性。例如，如果人类视频中的某些动作无法很好地映射到现有的 80 或 174 个类别中，或者某些动作具有更细微的差异，那么“动作中心对比学习”可能无法充分发挥作用。
    *   **改进方向：** 可以探索更深层次的语义或层次化动作表示学习，例如，通过聚类视频中的原子动作片段来发现更自然的动作原语，或者引入生成式模型来合成新的动作类别。
*   **时间反转增强的启发式性质：** 视觉中心对比学习中利用时间反转来分离运动和外观是一个巧妙的启发式方法。其核心假设是“运动信息变化显著，而内容信息和视觉干扰相对稳定”。然而，在某些复杂的场景中，例如背景动态（如树叶摇曳、水波流动），或者摄像机剧烈运动时，这个假设可能不完全成立，从而可能引入新的噪声。
    *   **改进方向：** 可以探索更鲁棒的时间不变性/敏感性建模方式，例如，通过学习显式的运动场 (motion fields) 或光流 (optical flow) 信息来更精确地分离运动成分，或者使用更高级的时间不变性数据增强策略。
*   **重建损失与对比损失的平衡：** 论文中总损失是重建损失和两个对比损失的简单求和。这些损失项的权重或平衡可能会显著影响模型性能和解耦效果。目前未详细说明其权重选择和敏感性分析。
    *   **改进方向：** 进行更详尽的消融实验来研究不同损失项权重的选择对性能的影响。或探索自适应权重调整机制，使模型能够根据训练进展动态调整各项损失的贡献。
*   **泛化到更复杂技能：** 尽管在 SimplerEnv 和真实世界桌面操作任务中取得了成功，但这些任务相对简单。对于需要精细控制、多步规划或复杂物体交互的更高级技能，ConLA 的潜在动作表示是否仍能有效捕捉，以及如何与更复杂的规划和推理模块结合，是值得进一步探索的问题。
    *   **改进方向：** 在更复杂的基准（例如需要工具使用、多臂协作或长期规划）上进行评估，并研究如何将 ConLA 的潜在动作与高级规划算法（如任务和运动规划）相结合。