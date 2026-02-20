# 1. 论文基本信息

## 1.1. 标题
Olaf-World: Orienting Latent Actions for Video World Modeling

## 1.2. 作者
Yuxin Jiang$^{1,2}$, Yuchao Gu$^1$, Ivor W. Tsang$^2$, Mike Zheng Shou$^1$
$^1$Showlab, $^2$National University of Singapore

## 1.3. 发表期刊/会议
该论文作为预印本发表在 arXiv 上。 arXiv 是一个广泛使用的预印本服务器，在机器学习和人工智能领域具有重要影响力，允许研究者在正式同行评审和发表前分享其研究成果，以便快速交流和获取反馈。

## 1.4. 发表年份
2026年

## 1.5. 摘要
扩大可控 <strong>世界模型 (world models)</strong> 的规模受到动作标签稀缺的限制。虽然 <strong>潜在动作学习 (latent action learning)</strong> 有望从无标签视频中提取控制接口，但学习到的潜在动作往往无法跨上下文转移：它们纠缠着场景特有的线索，并且缺乏共享的坐标系。发生这种情况是因为标准目标函数仅在每个视频片段内操作，没有提供在不同上下文之间对齐动作语义的机制。我们的关键见解是，尽管动作是未被观测的，但它们的语义效果是可观测的，并且可以作为共享的参照。我们引入了 `SeqΔ-REPA`（Sequence-level Control-Effect Alignment Objective），这是一种序列级别的控制-效果对齐目标函数，它将整合的潜在动作锚定到从一个冻结的、自监督视频编码器中提取的时间特征差异。在此基础上，我们提出了 `Olaf-World`，这是一个从大规模被动视频中预训练动作条件视频世界模型的流程。广泛的实验表明，我们的方法学习到了一个更具结构化的潜在动作空间，从而实现了比最先进的基线更强的零样本动作转移和更数据高效的适应新控制接口的能力。

## 1.6. 原文链接
*   [https://arxiv.org/abs/2602.10104](https://arxiv.org/abs/2602.10104)
*   [https://arxiv.org/pdf/2602.10104v1](https://arxiv.org/pdf/2602.10104v1)

# 2. 整体概括

## 2.1. 研究背景与动机
<strong>世界模型 (World models)</strong> 是一种能够预测未来观测值（例如视频帧）在给定动作下的行为的模型，对于规划（planning）和交互式模拟至关重要。近年来，视频生成模型在互联网规模数据上学习到了丰富的视觉和物理动力学先验知识，这使得它们成为构建视频世界模型的有前景的 <strong>主干网络 (backbones)</strong>。然而，将这些模型转化为可由动作控制的模拟器，通常需要大量、与帧对齐的动作标签。这些标签的获取成本高昂，且往往与特定的领域或控制接口紧密绑定。

<strong>潜在动作学习 (latent action learning)</strong> 提供了一种可扩展的解决方案，它直接从无标签视频中发现一个动作空间。一个 <strong>逆动力学编码器 (inverse-dynamics encoder)</strong> 从观测到的状态转移 $(x_i, x_{i+1})$ 推断出潜在动作 $z_i$，然后一个 <strong>前向模型 (forward model)</strong> 根据过去帧和推断出的动作预测未来帧。

然而，<strong>学习可转移的潜在动作 (transferable latent actions)</strong> 仍然具有挑战性。动作被认为是可转移的，如果它们能够跨上下文保持控制语义：即使视觉上下文（外观、视角、布局、光照等）发生变化，对应相同底层动作的状态转移也应该产生相似的潜在动作 $z_i$。本文识别了两个主要的失败模式：
1.  <strong>快捷学习 (shortcut learning)</strong>：逆动力学编码器通常会受到快捷学习的影响，$z_i$ 可能依赖于与 $x_{i+1}$ 相关的上下文相关视觉线索，而不是底层的可控原因，从而导致学习到的动作与场景外观纠缠不清。
2.  <strong>跨上下文不可识别性 (cross-context non-identifiability)</strong>：更根本的是，局部重建目标在不同上下文之间是不可识别的。由于训练仅限于单个视频片段，模型没有被鼓励在不同上下文之间使用共享的潜在坐标系，因此相同的语义动作（例如，“向前移动”）在不同环境中可能对应不同的潜在方向。

    这些问题共同阻碍了共享控制接口的出现：相同的动作语义不需要映射到潜在空间中一致的区域，从而削弱了转移和下游可控性。

## 2.2. 核心贡献/主要发现
为了解决上述挑战，本文提出了 `SeqΔ-REPA` 和 `Olaf-World`，其主要贡献如下：
1.  <strong>特性化跨上下文不可识别性 (Characterize cross-context non-identifiability)</strong>：论文明确指出了潜在动作学习中跨上下文不可识别性的问题，并解释了为何基于逐步重建的方法无法学习可转移的控制。附录 A 提供了正式的分析。
2.  <strong>提出 `SeqΔ-REPA` 目标函数 (Propose SeqΔ-REPA objective)</strong>：引入了一种新颖的序列级别控制-效果对齐目标函数 `SeqΔ-REPA`。该方法通过将潜在动作轨迹锚定到从自监督视频表示中提取的语义变化，鼓励实现上下文不变的动作语义。其核心思想是：虽然显式动作标签不可用，但控制的语义效果在视频中是可观测的。
3.  <strong>介绍 `Olaf-World` 预训练流程 (Introduce Olaf-World pretraining pipeline)</strong>：基于 `SeqΔ-REPA` 学习到的潜在动作，本文提出了 `Olaf-World`，这是一个从被动视频中学习动作条件视频世界模型的预训练流程。该流程能够实现可靠的跨上下文动作转移，并以最少的标签数据进行高效适应。
4.  <strong>实验验证 (Experimental Validation)</strong>：通过广泛的实验证明，`Olaf-World` 学习到了更具结构化的潜在动作空间，从而在零样本动作转移和数据高效适应新控制接口方面优于最先进的基线。
    *   <strong>上下文不变的零样本动作转移 (Context-invariant zero-shot action transfer)</strong>：从一个上下文的演示中提取的潜在动作可以被重用于在新上下文中诱导类似的控制效果。
    *   <strong>高效适应 (Efficient adaptation)</strong>：当有真实标签可用时，可以学习一个轻量级的映射到预训练的动作空间，从而以最少的数据和参数更新进行适应。
    *   <strong>更好地泛化到新场景 (Better generalization to unseen context)</strong>：由于潜在动作预训练使模型接触到多样化的状态转移，`Olaf-World` 比从零开始在标记数据集上训练的模型更好地泛化到新颖场景。

# 3. 预备知识与相关工作

## 3.1. 基础概念

### 3.1.1. 世界模型 (World Models)
<strong>世界模型 (World Models)</strong> 是一种预测未来观测值（例如视频帧）在给定动作下的行为的模型。它们通过学习环境的动力学模型，使 <strong>智能体 (agent)</strong> 能够在内部模拟和规划，而无需直接与真实世界互动。这对于 <strong>规划 (planning)</strong> 和 <strong>交互式模拟 (interactive simulation)</strong> 至关重要，尤其是在游戏、机器人和自动驾驶等领域。著名的世界模型包括 `Dreamer` 系列和 `Planet` 等。

### 3.1.2. 潜在动作学习 (Latent Action Learning, LAL)
<strong>潜在动作学习 (Latent Action Learning, LAL)</strong> 旨在从无标签的视频数据中自动发现或提取抽象的、低维的动作表示（即“潜在动作”）。其核心思想是，即使没有显式的动作标签，视频中的视觉变化也蕴含了导致这些变化的动作信息。LAL 通常通过训练一个 <strong>逆动力学模型 (inverse dynamics model)</strong> 来实现，该模型从连续的帧 $(x_i, x_{i+1})$ 中推断出中间的潜在动作 $z_i$。然后，这些潜在动作可以用于条件化一个 <strong>前向模型 (forward model)</strong>，以预测未来的帧 $x_{i+1}$ 或 $x_{i+k}$。

### 3.1.3. $\beta$-VAE
**$\beta$-VAE (Beta-Variational Autoencoder)** 是 <strong>变分自编码器 (Variational Autoencoder, VAE)</strong> 的一种变体，用于学习数据的低维潜在表示。它在标准 VAE 的目标函数中引入了一个超参数 $\beta$，用于平衡重建损失和 <strong>KL 散度 (Kullback-Leibler Divergence)</strong> 项。
标准 VAE 目标函数是：
$$
\mathcal{L}_{VAE} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))
$$
其中：
*   $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$ 是 <strong>重建损失 (reconstruction loss)</strong>，衡量解码器从潜在表示 $z$ 重建原始数据 $x$ 的效果。
*   $D_{KL}(q_\phi(z|x) || p(z))$ 是 <strong>KL 散度 (KL divergence)</strong>，衡量编码器后验分布 $q_\phi(z|x)$ 与先验分布 `p(z)` 之间的差异。它鼓励潜在空间具有良好的结构。
    $\beta$-VAE 的目标函数为：
$$
\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta D_{KL}(q_\phi(z|x) || p(z))
$$
其中 $\beta > 1$ 会更强调 KL 散度项，强制模型学习更解耦的、更符合先验分布的潜在表示，从而改善潜在空间的结构化程度和可解释性。

### 3.1.4. 自监督视频编码器 (Self-Supervised Video Encoder)
<strong>自监督视频编码器 (Self-Supervised Video Encoder)</strong> 是一种通过设计 <strong>代理任务 (pretext tasks)</strong> 在大规模无标签视频数据上进行训练的神经网络。这些代理任务不需要人工标注，例如预测视频的未来帧、识别视频片段的播放顺序、或填充视频中的掩码区域等。通过这些任务，编码器能够学习到捕捉视频中语义和时空动力学特征的强大表示。本文中提到的 `V-JEPA` (Vision Joint-Embedding Predictive Architecture) 就是一种自监督学习模型，它通过预测不同视图（例如，同一个视频帧的不同裁剪区域或不同时间步）的特征来学习鲁棒的视觉表示。冻结的自监督视频编码器可以提供一个稳定的、语义丰富的特征空间，用于对齐其他模型的潜在表示。

### 3.1.5. 扩散模型 (Diffusion Models) 和 扩散 Transformer (Diffusion Transformer, DiT)
<strong>扩散模型 (Diffusion Models)</strong> 是一类生成模型，它们通过逐步去噪一个随机噪声来生成数据。在训练过程中，模型学习如何反转一个逐渐向数据添加噪声的扩散过程。在推理时，模型从纯噪声开始，通过学习到的去噪步骤迭代地生成新的数据样本。
<strong>扩散 Transformer (Diffusion Transformer, DiT)</strong> 是将 Transformer 架构应用于扩散模型的一种方法。它将扩散模型中的 U-Net 架构替换为 Transformer 架构，通过处理潜在空间中的特征图作为序列，利用 Transformer 的自注意力机制来建模不同空间位置和时间步之间的依赖关系。DiT 在图像和视频生成方面都取得了最先进的性能，因为它能够更好地捕捉长距离依赖和全局结构。

### 3.1.6. LoRA (Low-Rank Adaptation)
**LoRA (Low-Rank Adaptation)** 是一种高效的 <strong>微调 (fine-tuning)</strong> 方法，主要用于大型预训练模型。它通过在预训练模型的 Transformer 层的权重矩阵旁边注入小的、低秩的适应矩阵来工作。在微调过程中，预训练模型的原始权重保持冻结，只有这些小型的低秩矩阵被训练。这大大减少了需要训练的参数数量，从而显著降低了计算成本和存储需求，同时在许多下游任务上实现了与全参数微调相当的性能。

## 3.2. 前人工作

### 3.2.1. 学习视频中的潜在动作 (Learning Latent Action from Videos)
先前的 <strong>潜在动作模型 (Latent Action Models, LAMs)</strong> 旨在从无标签视频中推断潜在控制信号。它们被用于：
1.  <strong>统一控制接口 (unified control interfaces)</strong>：为交互式世界模型提供统一的控制接口。
2.  <strong>动作表示 (action representations)</strong>：作为策略学习的动作表示，尤其是在机器人领域弥合不同 <strong>具身 (embodiment)</strong> 之间的差距。
3.  <strong>纯观测离线强化学习 (observation-only offline RL)</strong>：实现仅依靠观测数据的离线强化学习。
    大多数 LAMs 学习一个 <strong>逆模型 (inverse model)</strong>，该模型从观测到的状态转移中推断每一步的潜在动作，并使用重建或预测目标训练一个 <strong>前向解码器 (forward decoder)</strong>。离散（基于 VQ）和连续潜在参数化都被探索过。

**挑战与局限性**：
*   <strong>快捷学习 (shortcut solutions)</strong>：先前的工作认识到，基于局部状态转移的目标函数对 <strong>干扰因素 (nuisance factors)</strong> 和与动作相关的 <strong>干扰项 (action-correlated distractors)</strong> 敏感，这可能导致快捷解决方案并降低下游使用的效果。
*   <strong>缺乏跨环境一致性 (lack of cross-environment consistency)</strong>：现有的方法虽然尝试通过施加潜在空间约束或设计强调运动而非像素外观的目标函数来缓解上述问题，但它们仍然在孤立的视频片段上操作，无法确保潜在动作语义在不同环境中的一致性。

### 3.2.2. 视频世界模型 (Video World Model)
<strong>视频世界模型 (Video World Model)</strong> 预测未来观测值，并支持在游戏、机器人和驾驶等领域进行规划或交互式模拟。
*   <strong>依赖显式控制信号 (reliance on explicit control signals)</strong>：大多数可控的视频世界模型依赖于从交互式游戏引擎（如 Unreal Engine, Minecraft）收集的显式控制信号，其中帧级别的键盘/鼠标输入和其他交互注释被记录为控制信号。这虽然能实现强大的可控性，但也使得学习到的模型与特定的动作模式和数据收集流程绑定。
*   <strong>潜在动作世界模型 (latent-action world models)</strong>：这些模型通过直接从视频中推断控制接口，实现无需真实动作的交互。然而，它们的可控性和可转移性最终取决于学习到的潜在动作空间在不同上下文之间是否一致——这正是本文工作旨在解决的瓶颈。

### 3.2.3. 表示对齐 (Representation Alignment)
<strong>表示对齐 (Representation Alignment)</strong> 方法旨在将生成模型的内部特征与大型自监督编码器进行匹配，以提高语义保真度和训练效率。
*   <strong>早期工作 (early work)</strong>：最初主要集中在图像生成中的空间特征对齐。
*   <strong>近期扩展 (recent extensions)</strong>：视频领域的扩展已经整合了时间结构，将视频生成器内部状态与预训练视频编码器的内部状态进行对齐。这些方法主要目的是改善生成器内部状态表示，以实现更高质量的合成，即 <strong>特征到特征的对齐 (feature-to-feature alignment)</strong>。

    **与本文的差异**：
本文的方法不同于传统的特征到特征对齐。我们使用冻结的 <strong>时空编码器 (spatiotemporal encoder)</strong> 作为参考，通过匹配语义效果（特征变化），监督潜在动作，实现 <strong>控制到效果的对齐 (control-to-effect alignment)</strong>。

## 3.3. 差异化分析
本文 `Olaf-World` 的核心创新点在于通过 `SeqΔ-REPA` 解决了现有潜在动作学习方法在 <strong>跨上下文一致性 (cross-context consistency)</strong> 方面的根本性挑战。

*   **现有潜在动作学习的局限**：大多数 `LAMs` 虽然能够从无标签视频中学习潜在动作，但它们依赖于局部状态转移的重建目标。这导致学习到的潜在空间容易受到上下文相关视觉线索的影响（快捷学习），并且在不同环境中无法保持动作语义的一致性（跨上下文不可识别性）。这意味着，同一个语义动作在不同场景下可能被映射到潜在空间中的不同区域，从而阻碍了动作的有效转移。
*   <strong>本文的创新点——控制到效果的对齐 (Control-to-Effect Alignment)</strong>：`SeqΔ-REPA` 引入了一个序列级别的对齐机制。它利用一个冻结的、预训练的自监督视频编码器（如 `V-JEPA 2`）来捕捉视频片段的 <strong>语义效果方向 (semantic effect direction)</strong>，即视频特征在时间上的净变化。然后，它将模型学习到的集成潜在动作（在同一时间窗口内）与这个语义效果方向进行对齐。
    *   <strong>上下文不变性 (Context-Invariance)</strong>：通过使用时间特征差异作为效果参考，模型能够抑制静态外观细节，从而使参考信号对上下文变化具有鲁棒性。这种方法提供了一个共享的全局参考，鼓励在不同上下文之间实现一致的动作含义，并阻止模型依赖上下文相关的视觉快捷方式。
    *   <strong>解决不可识别性 (Addressing Non-Identifiability)</strong>：通过将潜在动作与一个跨上下文稳定的“效果”信号挂钩，`SeqΔ-REPA` 克服了传统 `β-VAE` 目标函数中存在的潜在坐标对称性问题，使得潜在动作在不同环境中具有可比性，从而实现真正的可转移性。

        简而言之，现有方法主要关注如何在单个片段内更好地重建或表示动作，而 `Olaf-World` 关注如何通过外部的、上下文无关的语义效果信号，**强制**潜在动作在 **不同片段和不同上下文之间** 保持一致的语义。这使得学习到的潜在动作更具通用性和可转移性，从而显著提升下游世界模型在零样本转移和数据高效适应方面的能力。

# 4. 方法论

本文的目标是从无标签视频中学习一个动作可控的视频世界模型。`Olaf-World` 流程分为两个阶段：
1.  <strong>学习可转移的潜在动作空间 (Learning a transferable latent action space)</strong>：通过 `SeqΔ-REPA` 目标函数，学习一个将动力学与视觉上下文解耦的潜在动作空间。
2.  <strong>训练动作条件视频世界模型 (Training an action-conditioned video world model)</strong>：使用学习到的潜在动作作为统一的控制接口，预训练一个视频生成世界模型。

## 4.1. 学习潜在动作模型 (Latent Action Model, LAM)

### 4.1.1. $\beta$-VAE 目标函数
给定一个视频片段 $x_{0:K}$，我们为每个状态转移 $(x_i, x_{i+1})$ 建模一个潜在动作 $z_i \in \mathbb{R}^{d_z}$，其中 $i = 0, \ldots, K-1$。一个标准的潜在动作模型由一个 <strong>因果逆动力学编码器 (causal inverse-dynamics encoder)</strong> 组成，它产生 $q_\phi(z_i \mid x_{0:i+1})$，以及一个 <strong>前向解码器 (forward decoder)</strong>，它预测在给定当前帧 $x_i$ 和推断动作 $z_i$ 的情况下，下一帧 $p_\theta(x_{i+1} \mid x_i, z_i)$。这确保潜在动作捕捉解释像素变化所需的动力学。模型通过以下步进式 $\beta$-VAE 目标函数进行训练：

$$
\mathcal{L}_{\theta ,\phi}^{VAE} = \frac{1}{K}\sum_{i = 0}^{K - 1}\Big(-\mathbb{E}_{q_{\phi}(z_i\mid x_{0:i + 1})}\big[\log p_\theta (x_{i + 1}\mid x_i,z_i)\big] +\beta \operatorname {KL}\big(q_\phi (z_i\mid x_{0:i + 1})||p(z_i)\big)\big) \quad (1)
$$

其中：
*   $K$ 是视频片段的长度。
*   $x_{0:K}$ 表示从时间步 `0` 到 $K$ 的视频帧序列。
*   $z_i$ 是在时间步 $i$ 推断出的潜在动作，维度为 $d_z$。
*   $q_\phi(z_i \mid x_{0:i+1})$ 是逆动力学编码器，它根据当前帧 $x_i$ 和下一帧 $x_{i+1}$（以及可选的过去帧 $x_{0:i}$）推断潜在动作 $z_i$ 的后验分布。
*   $p_\theta(x_{i+1} \mid x_i, z_i)$ 是前向解码器，它根据当前帧 $x_i$ 和潜在动作 $z_i$ 预测下一帧 $x_{i+1}$。
*   $\mathbb{E}[\cdot]$ 表示期望。
*   $\log p_\theta(x_{i+1} \mid x_i, z_i)$ 是重建损失项的负对数似然，鼓励解码器准确预测下一帧。
*   $\operatorname{KL}(q_\phi(z_i \mid x_{0:i+1}) || p(z_i))$ 是 KL 散度项，衡量推断出的潜在动作分布与预设先验分布 $p(z_i)$（这里是标准正态分布 $\mathcal{N}(0, I)$）之间的差异。
*   $\beta$ 是一个超参数，用于平衡重建损失和 KL 散度项。

    **该目标函数的局限性**：尽管公式 (1) 可以实现较低的单步预测误差，但其本身并不能确保潜在动作空间在不同上下文之间具有语义一致性。论文指出两个主要失败模式：
1.  <strong>快捷学习（上下文泄露）</strong>：由于后验分布 $q_\phi(z_i \mid x_{0:i+1})$ 条件化于 $x_{i+1}$，一个富有表现力的解码器可以通过编码与 $x_{i+1}$ 相关的上下文依赖性线索，而非可转移的控制，来减少损失。
2.  **跨上下文不可识别性**：由于损失函数从不比较不同轨迹之间的潜在动作，潜在坐标系是不受约束的，并且可能在不同上下文之间漂移。这意味着，相同的语义运动在不同视频中可能映射到潜在空间的不同方向，从而破坏了动作的转移能力（详见附录 A）。

## 4.1.2. SeqΔ-REPA：序列级别控制-效果对齐目标

为了解决上述歧义，本文引入了 `SeqΔ-REPA`，这是一个序列级别的对齐约束，它将潜在动作锚定到在不同视频和上下文之间具有可比性的效果信号（参见 Figure 3a）。

**核心思想**：虽然显式动作标签不可用，但控制的语义效果在视频中是可观测的：由相似底层动作驱动的状态转移，尽管外观可能不同，也应在不同上下文之间引起相似的语义变化。

**实现步骤**：
1.  <strong>定义效果方向 ($\tau_*$)</strong>：
    首先，使用一个冻结的 <strong>自监督视频编码器 (self-supervised video encoder)</strong> $f$（例如 `V-JEPA 2 ViT` (Assran et al., 2025)）。给定视频片段 $x_{0:K}$，编码器 $f$ 输出时空视觉 <strong>词元 (visual tokens)</strong>。我们对这些词元进行空间池化，以获得每帧的描述符 $s_i \in \mathbb{R}^D$。
    效果方向 $\tau_*$ 被定义为特征变化的净方向：
    $$
    \tau_{*} = \frac{1}{K}\sum_{i = 0}^{K - 1}\left(s_{i + 1} - s_{i}\right)\in \mathbb{R}^{D} \quad (2)
    $$
    其中：
    *   $f$ 是一个冻结的自监督视频编码器。
    *   $s_i \in \mathbb{R}^D$ 是第 $i$ 帧经过空间池化后得到的特征描述符，维度为 $D$。
    *   $s_{i+1} - s_i$ 表示相邻帧之间的特征差异，捕捉了语义变化。
    *   $\frac{1}{K}\sum_{i=0}^{K-1}(\cdot)$ 表示在整个视频片段上取平均，得到一个代表该片段整体语义变化的方向。
    *   **关键点**：由于 $\tau_*$ 是从时间差异计算并随时间平均的，它强调特征空间中连贯的时间变化，并且 $\Delta s$ 对静态外观不那么敏感，使得效果参考在上下文变化下保持稳定。

2.  <strong>整合潜在动作并映射 (Aggregate latent actions and map)</strong>：
    在潜在动作方面，逆模型推断出一系列潜在动作 $z_{0:K-1}$。我们将它们聚合起来，并映射到编码器的特征空间：
    $$
    \bar{z} = \frac{1}{K}\sum_{i = 0}^{K - 1}z_{i}\in \mathbb{R}^{d_{z}},\qquad u = h_{\psi}(\bar{z})\in \mathbb{R}^{D} \quad (3)
    $$
    其中：
    *   $z_i$ 是每帧推断出的潜在动作。
    *   $\bar{z}$ 是在整个视频片段上平均得到的集成潜在动作，维度为 $d_z$。
    *   $h_\psi: \mathbb{R}^{d_z} \to \mathbb{R}^D$ 是一个可训练的 MLP <strong>投影头 (projection head)</strong>，它将集成潜在动作 $\bar{z}$ 映射到与效果方向 $\tau_*$ 相同的特征空间 $\mathbb{R}^D$。
    *   $u$ 是映射后的集成控制方向。

3.  <strong>对齐集成控制方向与效果方向 (Align integrated control direction with effect direction)</strong>：
    然后，使用 <strong>余弦相似度 (cosine similarity)</strong> 对齐集成控制方向 $u$ 和效果方向 $\tau_*$：
    $$
    \mathcal{L}_{\psi}^{\mathrm{S e q}\Delta \cdot \mathrm{R E P A}} = 1 - \langle \mathrm{norm}(u),\mathrm{norm}(\tau_{*})\rangle \quad (4)
    $$
    其中：
    *   $\mathrm{norm}(\cdot)$ 表示 L2 范数归一化。
    *   $\langle \cdot, \cdot \rangle$ 表示点积。
    *   余弦相似度的范围是 $[-1, 1]$。当两个向量完全对齐时，余弦相似度为 1；当它们完全相反时，为 -1。因此，$1 - \langle \mathrm{norm}(u),\mathrm{norm}(\tau_{*})\rangle$ 作为损失函数，当 $u$ 和 $\tau_*$ 方向一致时，损失最小（接近 0）。
    *   **关键点**：与 <strong>特征到特征的对齐 (feature-to-feature alignment)</strong> 不同，公式 (4) 施加了一个 <strong>控制到效果的对齐 (control-to-effect alignment)</strong> 约束：它将集成的潜在控制与共享的语义变化概念对齐，从而鼓励在不同上下文之间实现一致的动作含义。

        **最终训练目标**：
模型 $(\theta, \phi, \psi)$ 通过以下损失函数进行训练：
$$
\mathcal{L}_{\mathrm{LAM}} = \mathcal{L}_{\theta ,\phi}^{\mathrm{VAE}} + \lambda \mathcal{L}_{\psi}^{\mathrm{Seq}\Delta}\mathrm{-REPA} \quad (5)
$$
其中：
*   $\mathcal{L}_{\theta, \phi}^{\mathrm{VAE}}$ 是 $\beta$-VAE 损失（公式 1）。
*   $\mathcal{L}_{\psi}^{\mathrm{Seq}\Delta\mathrm{-REPA}}$ 是 `SeqΔ-REPA` 对齐损失（公式 4）。
*   $\lambda > 0$ 是损失权重，用于平衡两个损失项。
    在训练过程中，参考编码器 $f$ 保持冻结。

下图（Figure 3a）展示了 `SeqΔ-REPA` 在潜在动作模型 (LAM) 训练中的作用：

![fig 13](images/13.jpg)
*Figure 3a. 我们训练一个潜在动作模型 (LAM)，并通过在冻结视频特征空间中对齐动作效果（使用 SeqΔ-REPA）来鼓励跨上下文的一致性。*

## 4.2. Olaf-World：动作感知预训练世界模型

### 4.2.1. 动作感知预训练 (Action-aware Pretraining)
给定一个视频 $x_{0:T}$，冻结的 `LAM` 生成每帧的潜在动作 $z_{0:T-1} \in \mathbb{R}^{d_z}$。
`Olaf-World` 基于一个预训练的 <strong>潜在图像到视频扩散 Transformer (latent image-to-video diffusion Transformer, DiT)</strong>（例如 `SkyReels-V2`）进行构建，并使用标准的 <strong>流匹配 (flow-matching)</strong> 目标函数在帧序列与潜在动作配对的数据上进行训练（参见 Figure 3b）。

**动作条件化机制**：
*   每帧的 $z_t$ 会被线性投影（从 $d_z=32$ 映射到 `1536` 维），然后添加到扩散时间步嵌入中。
*   融合后的嵌入被映射到每块的 `AdaLAN-Zero` 调制参数，这些参数用于条件化每个 `DiT` 块。
*   **时间压缩**：由于 <strong>主干网络 (backbone)</strong> 在由 3D 视频 VAE 编码的潜在空间上操作，输入视频会经历 $r=4$ 的时间压缩。因此，每 $r$ 个连续的步进动作被组合成一个潜在时间条件向量。
*   **效果**：世界模型被 `LAM` 潜在动作条件化，提供了一个统一的控制接口，该接口可以在具有不同原始动作约定（raw action conventions）的环境中转移。

    下图（Figure 3b）展示了 `Olaf-World` 动作感知预训练的流程：

    ![fig 13](images/13.jpg)
    *Figure 3b. 然后，我们将冻结的 LAM 应用于无标签视频以提取潜在动作序列，并将其用作统一控制接口来预训练动作条件视频世界模型。*

### 4.2.2. 特定世界适应 (Specific-world Adaptation)
在目标交互环境中，我们观测到来自特定环境动作空间的显式真实动作 $a_t$。
`Olaf-World` 学习一个小的 <strong>动作适配器 (action adapter)</strong> $A_\eta$，它将环境动作映射到潜在动作：`\hat{z}_t = A_\eta(a_t)`，并使用 $\hat{z}_t$ 控制预训练的世界模型。

**适应机制**：
*   **离散动作空间**：对于离散动作集 $\mathcal{A}$，$A_\eta$ 可以实现为一个 <strong>嵌入表 (embedding table)</strong> $E \in \mathbb{R}^{|\mathcal{A}| \times d_z}$，其中 $\hat{z}_t = E[a_t]$。
*   **初始化**：$E$ 使用从目标数据计算出的 <strong>类别原型 (class-wise prototypes)</strong> 进行初始化：对于每个动作 $a \in \mathcal{A}$，在标记为 $a$ 的片段上运行冻结的 `LAM`，并将 `E[a]` 设置为平均推断潜在动作。
*   <strong>微调 (Finetuning)</strong>：然后，使用相同的流匹配目标函数对 (i) 动作适配器 $A_\eta$ 和 (ii) `主干网络 (backbone)` 上的少量 `LoRA` 参数进行 <strong>微调 (finetuning)</strong>。
*   **效果**：这可以快速将模型专门化到新的动作空间，同时保留从被动视频中学习到的全局对齐的潜在控制语义。

# 5. 实验设置

本节旨在验证 `SeqΔ-REPA` 的性能以及改进后的潜在动作对下游应用程序的影响。实验主要回答以下问题：
*   <strong>RQ1 (结构)</strong>：学习到的潜在动作是否编码了可线性解码且跨域一致的动作语义？(Section 4.2)
*   <strong>RQ2 (转移)</strong>：这种对齐是否能够实现零样本控制转移到新上下文？(Section 4.3)
*   <strong>RQ3 (适应)</strong>：对齐的潜在空间是否能够实现对特定控制接口的数据高效适应？(Section 4.4)

## 5.1. 数据集

*   **LAM 和 WM 预训练**：
    *   **MiraData (Ju et al., 2024)**：用于训练潜在动作模型 (LAM) 和动作条件世界模型。具体使用了 `MiraData` 的 `3D Rendering` 和 `City Walking` 类别。`MiraData` 是一个大规模视频数据集，具有长时长和结构化字幕，提供了丰富的视觉和动态数据。
*   **特定世界适应和受控评估**：
    *   **MIND (Ye et al., 2026)**：一个开放域数据集，包含在 `Unreal Engine 5` 中收集的帧对齐动作标签。
    *   **MIND 子集**：`MIND` 包含两个不相交的子集，具有不同的场景和相机设置：
        *   <strong>第一人称 (1ST-P, First-Person)</strong>
        *   <strong>第三人称 (3RD-P, Third-Person)</strong>
    *   **动作空间**：两者共享相同的 8 种动作标签空间：导航 (w/S/A/D: forward/back/left/right) 和相机控制 ((+/<-/->: look up/down/left/right)。这种划分允许在外观和视角变化下严格测试跨上下文转移。

## 5.2. 评估指标
对论文中出现的每一个评估指标，必须按照以下三段结构提供完整说明：概念定义、数学公式和符号解释。

### 5.2.1. 宏观 F1 (Macro-F1)
**概念定义**：<strong>F1 分数 (F1 Score)</strong> 是精确率 (Precision) 和召回率 (Recall) 的调和平均值，用于评估分类模型的性能，特别是在类别不平衡的数据集上。<strong>宏观 F1 (Macro-F1)</strong> 是对每个类别计算 F1 分数，然后取所有类别的平均值。这给予所有类别相同的权重，无论其样本数量多少，因此它更能反映模型在所有类别上的平均性能，而不是偏向于样本量大的类别。在本文中，`Macro-F1` 用于评估潜在动作空间的线性可分离性和跨域不变性，即潜在动作能否准确地线性解码为预定义的 8 种动作类别。

**数学公式**：
<strong>精确率 (Precision)</strong>:
$$
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
$$
<strong>召回率 (Recall)</strong>:
$$
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
$$
**F1 分数**:
$$
\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$
<strong>宏观 F1 (Macro-F1)</strong>：
$$
\text{Macro-F1} = \frac{1}{N_{\text{classes}}} \sum_{c=1}^{N_{\text{classes}}} \text{F1}_c
$$
**符号解释**：
*   $\text{True Positives}$：正确预测为正例的样本数。
*   $\text{False Positives}$：错误预测为正例的样本数（实际为负例）。
*   $\text{False Negatives}$：错误预测为负例的样本数（实际为正例）。
*   $\text{Precision}_c$：类别 $c$ 的精确率。
*   $\text{Recall}_c$：类别 $c$ 的召回率。
*   $\text{F1}_c$：类别 $c$ 的 F1 分数。
*   $N_{\text{classes}}$：总类别数。

### 5.2.2. VBench 视觉指标 (VBench Visual Metrics)
**概念定义**：**VBench** 是一个全面的基准套件，用于评估视频生成模型的性能，涵盖多个维度。本文中，`VBench` 被用于评估生成视频的视觉质量。具体关注的指标包括：
1.  <strong>图像质量 (Image Quality, Image Qual. ↑)</strong>：衡量生成视频中单帧图像的视觉保真度、清晰度和真实感。通常通过 FID (Fréchet Inception Distance) 或 Inception Score 等指标的变体来评估，但在这里表示为“越高越好”的综合分数。
2.  <strong>时间一致性 (Temporal Consistency, Temp. Cons. ↑)</strong>：衡量生成视频中帧与帧之间运动和内容变化的流畅性、连贯性和合理性。高时间一致性意味着视频内容随时间自然演变，没有突兀的跳动或不连贯的视觉元素。

    **数学公式**：
`VBench` 是一套综合指标，其内部可能包含多种子指标及其加权平均。论文中未给出具体的计算公式，但通常这些指标基于深度学习特征提取器（如 Inception 网络）对生成视频和真实视频进行比较，或基于光流（optical flow）分析视频帧间变化。由于原文未提供具体公式，这里不做详细数学公式展开。

**符号解释**：
*   `Image Qual. ↑`：一个综合性的图像质量得分，箭头向上表示数值越高代表图像质量越好。
*   `Temp. Cons. ↑`：一个综合性的时间一致性得分，箭头向上表示数值越高代表时间一致性越好。

### 5.2.3. 相对姿态误差 (Relative Pose Error, RPE)
**概念定义**：<strong>相对姿态误差 (Relative Pose Error, RPE)</strong> 是一种常用的评估 <strong>视觉里程计 (Visual Odometry)</strong> 和 <strong>同步定位与地图构建 (Simultaneous Localization and Mapping, SLAM)</strong> 算法精度的指标，也可以用于评估世界模型在动作控制下生成轨迹的准确性。它衡量的是在给定动作序列下，模型生成的相机或智能体轨迹与真实轨迹之间的吻合程度。`RPE` 通过比较两个轨迹中连续姿态变换的误差来评估局部一致性。本文采用 `RPE-trans`（平移误差）和 `RPE-rot`（旋转误差）来量化动作遵循的忠实度。较低的 `RPE` 值表示生成轨迹与预期轨迹之间的吻合度更高，即动作遵循更准确。

**数学公式**：
给定两个相机姿态序列 $P_{GT} = \{T_{GT,0}, T_{GT,1}, \ldots, T_{GT,N}\}$ 和 $P_{Gen} = \{T_{Gen,0}, T_{Gen,1}, \ldots, T_{Gen,N}\}$，其中 $T_i \in SE(3)$ 表示在时间步 $i$ 的相机姿态（包含旋转和平移）。
对于每个时间步 $i$，我们计算相对姿态变换：
*   <strong>地面真值 (Ground Truth)</strong> 相对姿态变换：$\Delta T_{GT,i} = T_{GT,i}^{-1} T_{GT,i+1}$
*   <strong>生成 (Generated)</strong> 相对姿态变换：$\Delta T_{Gen,i} = T_{Gen,i}^{-1} T_{Gen,i+1}$

    RPE 衡量的是这两个相对姿态变换之间的差异。一个常用的定义是使用李代数（Lie Algebra）表示姿态误差。
假设 $\Delta T = \begin{pmatrix} R & t \\ 0 & 1 \end{pmatrix}$，其中 $R$ 是旋转矩阵，$t$ 是平移向量。
<strong>RPE 平移误差 (RPE-trans)</strong>：
$$
\text{RPE-trans} = \sqrt{(t_{GT,x} - t_{Gen,x})^2 + (t_{GT,y} - t_{Gen,y})^2 + (t_{GT,z} - t_{Gen,z})^2}
$$
或者更一般地，是两个平移向量之间的欧氏距离。
<strong>RPE 旋转误差 (RPE-rot)</strong>：
旋转误差通常通过计算两个旋转矩阵 $R_{GT}$ 和 $R_{Gen}$ 之间的角度差来衡量。例如，可以使用 $R_{Err} = R_{GT}^{-1} R_{Gen}$，然后将 $R_{Err}$ 转换为轴角表示，取其角度。
$$
\text{RPE-rot} = \arccos\left(\frac{\text{trace}(R_{GT}^{-1}R_{Gen}) - 1}{2}\right)
$$
在本文中，`RPE-trans` 和 `RPE-rot` 具体指的是平移误差幅度和旋转误差角度。

**符号解释**：
*   $P_{GT}$：真实（地面真值）相机姿态序列。
*   $P_{Gen}$：生成视频中重建的相机姿态序列。
*   $T_{GT,i}, T_{Gen,i}$：在时间步 $i$ 的真实和生成相机姿态（一个 $4 \times 4$ 齐次变换矩阵）。
*   $T_{GT,i}^{-1}$：真实姿态 $T_{GT,i}$ 的逆变换。
*   $\Delta T_{GT,i}$：真实轨迹中从 $i$ 到 $i+1$ 的相对姿态变换。
*   $\Delta T_{Gen,i}$：生成轨迹中从 $i$ 到 $i+1$ 的相对姿态变换。
*   `R, t`：旋转矩阵和平移向量，构成姿态变换矩阵。
*   $\text{trace}(\cdot)$：矩阵的迹。
*   $\text{RPE-trans}$：平移误差，单位通常是米（m）。
*   $\text{RPE-rot}$：旋转误差，单位通常是弧度（rad）或度（deg）。
*   $\downarrow$：箭头向下表示数值越低代表性能越好。

## 5.3. 对比基线 (Baselines)
本文将提出的方法 `Olaf-World` 与以下基线模型进行比较：

1.  **AdaWorld (Gao et al., 2025a)**：
    *   **概念**：`AdaWorld` 是一种最先进的潜在动作世界模型，它也旨在通过学习潜在动作来构建适应性强的世界模型。
    *   **比较方式**：为了进行受控比较，本文在与 `Olaf-World` 相同的视频模型主干网络、数据和训练/适应预算下运行 `AdaWorld`，同时保持其官方潜在动作训练流程和配置不变。
    *   **目的**：这种比较旨在隔离潜在动作学习目标函数本身的效果，即 `SeqΔ-REPA` 带来的改进。

## 5.4. 实现细节 (Implementation Details)

### 5.4.1. 潜在动作模型 (LAM)
*   **架构**：`LAM` 实现为一个基于 VAE 的视频预测框架，包含一个 <strong>因果时空编码器 (causal spatio-temporal encoder)</strong> 和一个 <strong>仅空间解码器 (spatial-only decoder)</strong>。
    *   编码器和解码器都采用 `Transformer` 架构，具有 16 个块、1024 嵌入维度和 16 个注意力头。
    *   编码器对时间注意力层应用 <strong>因果掩码 (causal masking)</strong>，以防止未来帧信息泄露。
    *   潜在动作维度 $d_z = 32$。
    *   训练时使用窗口大小 $K=16$，分辨率为 $272 \times 480$。
*   **对齐**：使用一个投影头进行对齐，该投影头包含 `LayerNorm` 紧随一个 3 层 MLP（Linear $\to$ SiLU $\to$ Linear $\to$ SiLU $\to$ Linear），将池化后的潜在动作投影到效果方向的维度 $D=1408$。
*   **效果教师**：作为冻结的效果教师，使用 `V-JEPA 2 ViT-Giant/16 (384)` (Assran et al., 2025)，其在视频数据上进行了预训练。
*   **训练**：
    *   优化器：`AdamW`，学习率 $2.5 \times 10^{-5}$，权重衰减 $10^{-2}$。
    *   总批次大小：32，在 $8 \times \text{H}200$ GPU 上。
    *   超参数：$\beta = 2 \times 10^{-4}$ (KL 项)，$\lambda = 0.02$ (对齐损失)。
    *   数据增强：为保留 `V-JEPA 2` 提取的效果轨迹的保真度，训练期间禁用颜色抖动 (color jitter)。
    *   训练时长：100 个 epoch（约 146k 步），耗时约 4.5 天。

### 5.4.2. Olaf-World (世界模型)
*   **架构**：`Olaf-World` 基于 `SkyReels 12V 1.3B DiT 主干网络` (Chen et al., 2025a) 构建。
    *   潜在动作注入：通过线性投影（从 `32` 维到 `1536` 维）将潜在动作注入到时间步嵌入流中，并使用一个学习到的增益 $\gamma$，初始化为 2.0。
    *   适应：对于适应阶段，使用 `LoRA`，秩为 16，应用于每个 `DiT` 块中的注意力层（`attn. {q, k, v, o}`）和前馈网络层（`ffn. {0, 2}`）的线性层。
*   **训练**：
    *   预训练：潜在动作条件视频生成器训练 10k 步，使用 `AdamW`，学习率 $5 \times 10^{-5}$，权重衰减 $10^{-3}$。
    *   硬件：在 $4 \times \text{NVIDIA H200}$ GPU 上分布式训练，每设备批次大小为 4。
    *   下游适应：仅微调 `LoRA` 参数（秩 $r=16$），学习率 $1 \times 10^{-4}$，权重衰减为 0。

# 6. 实验结果与分析

## 6.1. 潜在空间诊断 (Latent Space Diagnostics)

### 6.1.1. 跨上下文线性探测 (Cross-Context Linear Probing)
**设置**：评估学习到的潜在动作空间 $z_t$ 是否是线性可分离的，并且对域偏移（domain shifts）具有不变性。
*   **方法**：在冻结的潜在动作 $z_t$ 之上训练一个线性探测器，以预测 8 种原子动作。
*   **上下文不变性测试**：采用跨域探测协议。在一个域（例如 1ST-P）上训练和验证探测器，选择在域内验证 `Macro-F1` 分数最高的检查点，然后将该探测器零样本（zero-shot）应用于另一个域（例如 3RD-P）。反向重复此过程（3RD-P $\to$ 1ST-P）。
*   **指标**：报告 `Macro-F1` 进行类别平衡比较。

    **结果**：
下图（Figure 6）展示了在训练过程中，域内/跨域线性探测的宏观 F1 分数变化。

![fig 6](images/6.jpg)
*Figure 6. 域内/跨域线性探测在训练过程中的表现。 (a) 探测器在 1ST-P 训练，在 {1ST-P, 3RD-P} 上评估。 (b) 探测器在 3RD-P 训练，在 {3RD-P, 1ST-P} 上评估。*

以下是原文 Table I 的结果：

<table><tr><td>Method</td><td>1st→1st</td><td>1st→3rd</td><td>3rd→3rd</td><td>3rd→1st</td></tr><tr><td>AdaWorld</td><td>0.6004</td><td>0.4820</td><td>0.4827</td><td>0.4999</td></tr><tr><td>Ours</td><td>0.8138</td><td>0.6250</td><td>0.8256</td><td>0.5904</td></tr></table>

*Table I. 域内/跨域线性探测 (Macro-F1, ↑)。灰色列表示跨域探测 (源域→目标域)。*

**分析**：
*   `SeqΔ-REPA` 学习到的潜在动作不仅更具线性可解码性，而且在跨上下文方面表现出更高的不变性。
*   在所有检查点上，`Olaf-World` 在域内 `Macro-F1` 上均高于 `AdaWorld`，并且在双向的跨域评估（1ST-P $\leftrightarrow$ 3RD-P）中也始终优于 `AdaWorld`。这表明动作语义在视角和外观变化下的对齐得到了显著改善。
*   尤其是在更具挑战性的 3RD-P 子集上进行探测时，`Olaf-World` 的增益最为显著，`AdaWorld` 在低 `Macro-F1` 处饱和，而我们的方法则显著更高（Table I）。
*   通过对齐效果方向，本文方法能够促进早期学习，减少训练过程中的波动，从而产生更稳定和可转移的潜在结构。

### 6.1.2. 跨上下文动作一致性 (Cross-Context Action Consistency)
**设置**：为了测试动作语义在不同上下文之间是否一致，本文分别在每个域内计算动作原型（类别中心点），并可视化这些原型之间的跨域余弦相似度。
*   **期望结果**：一个良好对齐的潜在空间应该具有对角占优（diagonal-dominant）的特性，即每个 1ST-P 动作应该与它的 3RD-P 对应动作最相似。

    **结果**：
下图（Figure 7）比较了 1ST-P 和 3RD-P 之间跨上下文原型相似度。

![fig 7](images/7.jpg)
*Figure 7. 跨域动作相似度。1ST-P（行）和 3RD-P（列）中每个动作原型之间的余弦相似度。SeqΔ-REPA 产生了更对角占优的矩阵（跨上下文更强的一一匹配）。*

**分析**：
*   `AdaWorld` 基线（Figure 7a）在所有地方都显示出高相似度，这意味着 1ST-P 中的不同动作在 3RD-P 中通常看起来“相似”于多个动作。这表明潜在空间在上下文变化下不是唯一的动作特定（即跨上下文可识别性弱）。
*   相比之下，`Olaf-World`（Figure 7b）明显更具对比性：匹配的动作保持高相似度，而非匹配的对则被推向接近零或负值。这表明 `SeqΔ-REPA` 在视角和外观变化下学习到了更一致、可转移的动作语义。
*   剩余的混淆出现在偏航（yaw）“look”动作（$\leftarrow$/$\rightarrow$）中，这是可以预期的，因为在以自我为中心的相机（egocentric camera）和第三人称相机（third-person camera）设置下，相同的控制实际上会引起不同的可观测运动，所以这些动作的对齐程度较低。

## 6.2. 零样本动作转移 (Zero-shot Action Transfer)
**设置**：定性评估模型是否能够独立于视觉上下文遵循控制信号 $z_t$。
*   **方法**：从一个参考视频片段中提取潜在动作序列 $z_{0:T-1}$，并用它来驱动从不同目标初始帧开始的生成。
*   **成功标准**：成功的转移需要重现参考运动，同时保留目标外观。

    **结果**：
下图（Figure 3）展示了零样本动作序列转移的示例。

![fig 3](images/3.jpg)
*Figure 3. 零样本动作序列转移。我们从参考视频片段（顶部）中提取动作序列，并将其零样本应用于不同的目标上下文。AdaWorld 经常表现出时间模糊、智能体消失和运动漂移，而 Olaf-World 在目标外观保留和运动忠实度方面表现更好。数字表示帧索引。*

**分析**：
*   `Olaf-World` 更可靠地转移动作序列，同时更好地保留了目标上下文。
*   在所有四个案例中，`AdaWorld` 在转移时经常退化：它表现出 (i) 时间模糊和不稳定 (A)，(ii) 控制角色丢失或尺度漂移 (B, D)，以及 (iii) 轨迹漂移，趋向于偏离参考的通用运动 (C)。
*   相比之下，`Olaf-World` 保持了场景和主体的持久性，同时忠实地执行了预期运动。
*   **结论**：这些结果表明，`SeqΔ-REPA` 产生了一个控制信号，其语义在大的上下文变化下仍然保持动作特定性。

## 6.3. 世界模型适应 (World Model Adaptation)

### 6.3.1. 数据高效适应 (Data-Efficient Adaptation)
**设置**：研究预训练的视频世界模型在有限的标记交互下，如何高效地适应目标控制接口。
*   **比较方法**：
    1.  **DirectAct**：直接条件化于地面真值动作。
    2.  **AdaWorld**：使用香草 $\beta$-VAE 进行潜在动作预训练。
    3.  **Olaf-World (Ours)**：使用 $\beta$-VAE + `SeqΔ-REPA` 进行潜在动作预训练。
*   **统一设置**：所有方法使用相同的视频主干网络和适应能力 (`LoRA` 秩 16，匹配的步数和优化器)。
*   **变量**：改变标记适应数据集的大小（`#Adapt Videos` $\in \{0, 1, 50\}$；分别对应约 0、1 分钟和 2 小时的数据）。
*   **指标**：
    *   视频质量：`VBench` 视觉指标 (`Image Qual. ↑`, `Temp. Cons. ↑`)。
    *   可控性：平移和旋转相对姿态误差 (`RPE-trans ↓`, `RPE-rot ↓`)。

        **结果**：
以下是原文 Table 2 的结果：

<table><tr><td rowspan="3">Method</td><td colspan="2" rowspan="2"># Adapt Videos</td><td colspan="3">1ST-P</td><td colspan="3">3RD-P</td></tr><tr><td colspan="2">Visual Quality</td><td>Action Accuracy (RPE)</td><td colspan="2">Visual Quality</td><td>Action Accuracy (RPE)</td></tr><tr><td colspan="3">Image Qual. ↑</td><td>Temp. Cons. ↑</td><td>Trans ↓</td><td>Rot. ↓</td><td>Image Qual. ↑</td><td>Temp. Cons. ↑</td></tr><tr><td>DirectAct</td><td>0</td><td>0.7213</td><td>0.8993</td><td>0.0703</td><td>1.4311</td><td>0.6970</td><td>0.9086</td><td>0.0897</td><td>0.7968</td></tr><tr><td>AdaWorld</td><td>0</td><td>0.5600</td><td>0.9226</td><td>0.0470</td><td>1.0844</td><td>0.6102</td><td>0.9344</td><td>0.0723</td><td></td></tr><tr><td>Ours</td><td>0</td><td>0.5400</td><td>0.9123</td><td>0.0387</td><td>0.8773</td><td>0.5909</td><td>0.9203</td><td>0.0461</td><td>0.4873</td></tr><tr><td>DirectAct</td><td>1</td><td>0.5269</td><td>0.8828</td><td>0.0672</td><td>1.2822</td><td>0.6019</td><td>0.8851</td><td>0.0708</td><td>0.8543</td></tr><tr><td>AdaWorld</td><td>1</td><td>0.5623</td><td>0.8955</td><td>0.0318</td><td>0.6420</td><td>0.6033</td><td>0.8989</td><td>0.0525</td><td></td></tr><tr><td>Ours</td><td>1</td><td>0.5726</td><td>0.9015</td><td>0.0284</td><td>0.4680</td><td>0.5844</td><td>0.8974</td><td>0.0348</td><td>0.3861</td></tr><tr><td>DirectAct</td><td>50</td><td>0.5936</td><td>0.9345</td><td>0.0351</td><td>0.4527</td><td>0.6265</td><td>0.9286</td><td>0.0402</td><td>0.3846</td></tr><tr><td>AdaWorld</td><td>50</td><td>0.6177</td><td>0.9239</td><td>0.0263</td><td>0.3834</td><td>0.6459</td><td>0.9306</td><td>0.0393</td><td></td></tr><tr><td>Ours</td><td>50</td><td>0.6312</td><td>0.9263</td><td>0.0230</td><td>0.3785</td><td>0.6486</td><td>0.9287</td><td>0.0222</td><td>0.2082</td></tr></table>

*Table 2. 将世界模型适应到具有不同标记数据量（#Adapt Videos）的目标控制域。我们报告 VBench 视觉指标（↑）和通过 RPE 衡量的动作准确性（↓）。Olaf-World 在所有设置下都实现了最低的 RPE，表明动作遵循最忠实。每个域和预算的最佳结果以粗体显示。*
**注意：** 原始表格中 `AdaWorld` 在 `3RD-P` 域的 `Action Accuracy (RPE) Rot. ↓` 列下存在数据缺失，这里保持原文。

**定量结果分析**：
*   `Olaf-World` 在所有适应预算下，在 1ST-P 和 3RD-P 上都实现了最低的 `RPE-trans` 和 `RPE-rot`，这表明其动作遵循最为忠实。
*   与 `AdaWorld` 相比，`Olaf-World` 持续提高了可控性，同时保持了可比的视频质量，这表明 `SeqΔ-REPA` 学习到的潜在控制表示更容易适应。
*   `DirectAct` 在 0 视频（即无动作监督）时退化为标准的图像到视频生成，解释了其强大的视觉分数但无信息的可控性。随着动作监督的增加，`DirectAct` 有所改进，但在相同的 `LoRA` 秩 16 设置下，其可控性仍不如潜在动作预训练方法。
*   论文预测，如果使用更大的适应能力（例如，更高的 `LoRA` 秩或完全微调），`DirectAct` 与潜在动作方法的差距可能会缩小。

    **定性结果分析**：
下图（Figure 8）提供了动作条件生成在适应后的定性比较。

![fig 8](images/8.jpg)
*该图像是示意图，展示了在不同场景下（1ST-P 和 3RD-P）使用 DirectAct、AdaWorld 以及我们的方法的效果对比，每个方法的表现用不同颜色标识。图中左侧为 1ST-P，右侧为 3RD-P，实验结果提供了对各种方法在控制效果上的直观比较。*
*Figure 8. 动作条件生成在适应后的定性比较。给定相同的初始帧和动作序列，Olaf-World 更忠实地遵循控制指令，并随着新区域的显露保留外观一致性。动作是按状态转移对齐的：$a_t$ 对应于从 $x_t$ 到 $x_{t+1}$ 的变化。*

*   Figure 8 的定性结果与定量趋势一致。在完全适应（50 个视频）后，`Olaf-World` 更可靠地遵循预期控制，并保持生成的场景在视觉上的一致性：当相机旋转或智能体侧向移动时，新可见区域的合成细节稳定且与初始帧匹配。
*   相比之下，`AdaWorld` 在多键控制（例如，3RD-P 转弯-向左移动）下可靠性较低：其生成往往只旋转而没有预期的左移运动，导致动作条件生成忠实度较低。

### 6.3.2. 泛化到未见上下文 (Generalization to Unseen Contexts)
**设置**：评估适应后的模拟器在测试时探索多样视觉世界时的可靠性。
*   **方法**：使用 Section 4.4.1 中的完全适应模型（1ST-P 动作空间），构建一个包含 50 个初始帧的 **OOD (Out-Of-Distribution)** 测试集，这些帧涵盖了不同的风格和场景。
*   **指标**：报告相同的 `VBench` 视觉指标和 `RPE`。

    **结果**：
以下是原文 Table 3 的结果：

<table><tr><td rowspan="2">Model</td><td colspan="2">Visual &amp; Temporal Quality</td><td colspan="2">Action Accuracy (RPE)</td></tr><tr><td>Image Qual. ↑</td><td>Temp. Cons. ↑</td><td>Trans ↓</td><td>Rot. ↓</td></tr><tr><td>DirectAct</td><td>0.6322</td><td>0.8585</td><td>0.0547</td><td>1.2343</td></tr><tr><td>AdaWorld</td><td>0.6181</td><td>0.8719</td><td>0.0482</td><td>1.7063</td></tr><tr><td>Ours</td><td>0.6274</td><td>0.8743</td><td>0.0478</td><td>1.2221</td></tr></table>

*Table 3. 适应后对未见视觉上下文的泛化能力。Olaf-World 实现了最低的 RPE，表明在外观变化下动作遵循最忠实。*

**定量结果分析**：
*   `Olaf-World` 在未见视觉上下文下保持了最佳的可控性，实现了最低的 `RPE`。这表明学习到的潜在控制在外观发生变化时仍然可用，而不会过度拟合适应时的视觉效果。

    **定性结果分析**：
下图（Figure 4）展示了在未见上下文下的定性泛化结果。

![fig 4](images/4.jpg)
*Figure 4. 未见上下文下的定性泛化。左图：基线方法在完成新揭示区域时，通常会破坏风格一致性；右图：基线方法显示主体漂移，而 Olaf-World 在相同的动作序列下更好地保持了稳定的外观。*

*   Figure 4 突出显示了两个具有代表性的案例。
    *   **未见风格**：在未见风格的场景中，当相机移动时，基线方法在 <strong>幻觉 (hallucinating)</strong> 新揭示的区域时，通常会破坏风格一致性。
    *   **未见物体**：在未见物体的场景中，基线方法难以保持物体身份，同时使其姿态/尺度/视角以与指令动作匹配的方式演变。
*   `Olaf-World` 在相同的动作序列下，更好地保持了外观一致性，同时产生了与动作一致的变化。
*   **结论**：这些结果表明，潜在动作预训练提高了动作条件动力学对 `OOD (Out-Of-Distribution)` 的鲁棒性。

## 6.4. 消融研究 (Ablation Studies)

### 6.4.1. SeqΔ-REPA 设计
**设置**：消融 `SeqΔ-REPA` 中的关键设计选择。
1.  **`w/o Δ`**：对齐静态特征而非效果方向。这意味着 `SeqΔ-REPA` 损失不再使用帧间特征差异 $\Delta s = s_{i+1} - s_i$，而是直接对齐平均静态特征 $s_i$。
2.  **`w/o norm`**：移除 $L_2$ 范数归一化，并将余弦对齐替换为对尺度敏感的 MSE 损失。这意味着对齐不再是方向性的，而是考虑了特征向量的绝对大小。

    **结果**：
下图（Figure 9）展示了 `SeqΔ-REPA` 在域内/跨上下文线性探测上的消融结果。

![fig 5](images/5.jpg)
*Figure 9. SeqΔ-REPA 在域内/跨上下文线性探测上的消融结果。实线：域内评估；虚线：跨域评估。*

以下是原文 Table 4 的结果：

<table><tr><td>Method</td><td>1st→1st</td><td>1st→3rd</td><td>3rd→3rd</td><td>3rd→1st</td></tr><tr><td rowspan="2">w/o Δ <br>w/o norm <br>Full</td><td>0.6805</td><td>0.5287</td><td>0.7137</td><td>0.4823</td></tr><tr><td>0.8064</td><td>0.5311</td><td>0.7096</td><td>0.5934</td></tr><tr><td></td><td>0.8138</td><td>0.6250</td><td>0.8256</td><td>0.5904</td></tr></table>

*Table 4. SeqΔ-REPA 消融实验 (Macro-F1, ↑)。灰色列表示跨域探测 (源域→目标域)。*

**分析**：
*   <strong>移除 $Δ$ (`w/o Δ`)</strong>：导致 `Macro-F1` 显著下降。这表明对齐静态特征允许上下文依赖的空间线索泄露到动作表示中，使得探测器的可分离性降低，并且在不同域之间的一致性也大大降低。
*   <strong>移除归一化 (`w/o norm`)</strong>：对齐变得对特征幅度敏感，而特征幅度可能在不同域之间变化。这使得学习到的潜在动作不稳定，导致在两个域中都没有可靠的良好表现。
*   <strong>完整目标函数 (Full)</strong>：整体目标函数表现最佳，并且在不同域之间最一致。这支持了 `SeqΔ-REPA` 通过对齐动作效果与一个稳定、尺度不变的相似性来提高转移能力。

### 6.4.2. 数据预算消融 (Data Budget Ablation)
**设置**：研究适应性能如何随标记目标域监督数据量的增加而变化，超出了主要论文中的 $\{0, 1, 50\}$ 预算。
*   **方法**：改变来自目标域的标记适应视频数量，分别为 $\{0, 1, 3, 5, 10, 25, 50\}$，对应约 $\{0, 1, 6, 13, 26, 60, 120\}$ 分钟的监督数据。
*   **指标**：观察动作准确性 (`RPE`) 和视频质量的变化。

    **结果**：
以下是原文 Table 10a 的结果：

<table><tr><td rowspan="2">#Vids</td><td colspan="3">Image Data (These Tenses)</td><td>Accepted Accuracy (RPE)</td></tr><tr><td>Visual</td><td>&amp; Temporal Quality</td><td></td><td></td></tr><tr><td></td><td>Image Qual. ↑</td><td>Temp. Cons. ↑</td><td>Trans ↓</td><td>Rot ↓</td></tr><tr><td>0</td><td>0.5400</td><td>0.9123</td><td>0.0387</td><td>0.8773</td></tr><tr><td>1</td><td>0.5726</td><td>0.9015</td><td>0.0284</td><td>0.4680</td></tr><tr><td>3</td><td>0.6542</td><td>0.9274</td><td>0.0304</td><td>0.4187</td></tr><tr><td>5</td><td>0.6171</td><td>0.9139</td><td>0.0284</td><td>0.4893</td></tr><tr><td>10</td><td>0.6311</td><td>0.9218</td><td>0.0271</td><td>0.4416</td></tr><tr><td>25</td><td>0.6321</td><td>0.9239</td><td>0.0250</td><td>0.3989</td></tr><tr><td>50</td><td>0.6312</td><td>0.9263</td><td>0.0230</td><td>0.3785</td></tr></table>

*Table 10a. 数据预算扫描 (固定适应容量)。*
下图（Figure 10c 和 Figure 10d）展示了 RPE-Trans 和 RPE-Rot 随视频数量变化的曲线。

![fig 9](images/9.jpg)
*Figure 10c. RPE-Trans 与 #Videos。*

![fig 10](images/10.jpg)
*Figure 10d. RPE-Rot 与 #Videos。*

**分析**：
*   随着监督数据量的增加，动作准确性 (`RPE`) 显著提高，尤其是在低数据量阶段（例如从 0 到 1 个视频），这与本文关注数据高效适应的目标一致。
*   视频质量在不同预算下保持可比性，表明额外的标签主要改善了控制对齐，而非视觉保真度。

### 6.4.3. LoRA 秩消融 (LoRA Rank Ablation)
**设置**：研究适应能力如何随 `LoRA` 秩的变化而变化，同时保持数据预算固定。
*   **方法**：在 50 个视频的设置下，使用 $\{16, 32, 64, 128, 256\}$ 的 `LoRA` 秩进行适应，并包含一个全参数更新作为上限参考。
*   **指标**：观察动作准确性 (`RPE`) 和视频质量的变化。

    **结果**：
以下是原文 Table 10b 的结果：

<table><tr><td rowspan="2">Rank</td><td colspan="3">Visual &amp; Temporal Quality</td><td colspan="3">Action Accuracy (RPE)</td></tr><tr><td>Image Qual. ↑</td><td>Temp. Cons. ↑</td><td>Trans ↓</td><td>Rot ↓</td><td></td><td></td></tr><tr><td>16</td><td>0.6312</td><td>0.9263</td><td>0.0230</td><td>0.3785</td><td></td><td></td></tr><tr><td>32</td><td>0.6265</td><td>0.9249</td><td>0.0230</td><td>0.3915</td><td></td><td></td></tr><tr><td>64</td><td>0.6394</td><td>0.9257</td><td>0.0251</td><td>0.3633</td><td></td><td></td></tr><tr><td>128</td><td>0.6309</td><td>0.9304</td><td>0.0213</td><td>0.3202</td><td></td><td></td></tr><tr><td>256</td><td>0.6372</td><td>0.9265</td><td>0.0220</td><td>0.2928</td><td></td><td></td></tr><tr><td>Full</td><td>0.6267</td><td>0.9210</td><td>0.0185</td><td>0.2980</td><td></td><td></td></tr></table>

*Table 10b. LoRA 秩扫描 (固定数据预算)。*
下图（Figure 10e）展示了 RPE-Trans 随 LoRA 秩变化的曲线。

![fig 11](images/11.jpg)
*Figure 10e. RPE-Trans 与 Rank。*

![fig 12](images/12.jpg)
*Figure 10f. RPE-Rot 与 Rank。*

**分析**：
*   较高的 `LoRA` 秩通常会改善动作准确性（较低的 `RPE`），这表明除了默认设置之外还有额外的提升空间。
*   视频质量在不同秩下保持相对稳定。
*   本文在主要实验中使用了秩 16 作为高效的默认设置，此消融实验证实了结论并不依赖于特定的秩选择。

# 7. 总结与思考

## 7.1. 结论总结
本文深入探讨了无监督潜在动作学习中的一个关键局限性：<strong>跨上下文不可识别性 (cross-context non-identifiability)</strong>。传统的逆动力学目标函数无法识别一个全局动作基础，导致学习到的潜在动作与上下文纠缠不清，并且难以有效地转移。

为了解决这一问题，本文提出了 `SeqΔ-REPA`，这是一个新颖的序列级别目标函数。`SeqΔ-REPA` 通过将潜在动作锚定到从自监督视频编码器中提取的特征差异所衡量的动作效果，从而鼓励生成上下文不变的动作语义。

在此基础上，本文引入了 `Olaf-World`，一个可扩展的潜在动作世界建模框架。`Olaf-World` 利用 `SeqΔ-REPA` 学习到的可转移潜在动作，能够从被动视频中预训练动作条件视频世界模型。通过广泛的实验，`Olaf-World` 被证明能够实现更强大的零样本动作转移，并以更数据高效的方式适应新的控制空间，优于现有的最先进基线方法。

## 7.2. 局限性与未来工作

本文指出了 `Olaf-World` 及其 `SeqΔ-REPA` 方法的一些局限性，并提出了几个未来研究方向：

### 7.2.1. 效果对齐的潜在动作 (Effect-aligned latent actions)
1.  **目标函数和效果目标**：目前使用简单有效的余弦对齐，将潜在动作与由冻结视频编码器特征差异定义的效果方向对齐。未来可以探索替代的效果目标和对齐公式，以进一步提高在不同上下文中的鲁棒性，并改善学习到的潜在动作空间的结构。
2.  <strong>分层潜在动作（技能）</strong>：当前的潜在动作是步进级别的（每秒 16 帧一个潜在动作）。学习一个分层的潜在动作，其中短时域控制组成更长时域的“技能”，可能会改善长时间 <strong>推演 (rollouts)</strong>、实现多速率控制，并为下游决策提供更清晰的接口。
3.  **物理规则转移**：未来的一个自然步骤是使用基于物理的约束增强效果对齐的潜在动作，以确保转移的轨迹在视觉上忠实且符合物理定律。例如，可以整合可验证的运动学或碰撞一致性奖励。更进一步，可以将动作条件转移扩展到包含复杂操作的多对象接触丰富的交互。
4.  **多实体动力学和分解控制**：`SeqΔ-REPA` 目前通过单个效果信号总结观测到的变化，这可能混合了不同来源的控制，如相机/自我运动、可控智能体运动、其他智能体行为和环境驱动事件。分解效果（自我 vs. 其他 vs. 环境）并学习实体特定的潜在控制可以提高可解释性，并实现更丰富的多实体可控世界建模。

### 7.2.2. 用于规划和推理的潜在动作 (Latent actions for planning and reasoning)
1.  **潜在动作空间中的规划和采样**：目前潜在动作主要用于转移和作为控制接口。下一步是直接在潜在动作序列上使用世界模型进行规划，例如基于想象的搜索或轨迹优化。
2.  **从帧级“视觉思维链”到潜在动作轨迹**：最近的工作表明大型视频模型可以展现出零样本能力，而视频生成工作也开始使用视觉思维链（例如稀疏关键帧、中间“思维”提示或故事板计划）作为指导，以改善长时域连贯性和可控性。一个有趣的方向是将潜在动作序列视为比密集帧级视觉思维链更紧凑、冗余更少的动力学轨迹，并研究这些轨迹如何支持动作和事件的评估、编辑和更高层次的推理。

### 7.2.3. 失败案例 (Failure cases)
论文在附录 E 中提供了一些 `Olaf-World` 的代表性失败案例，例如：
1.  <strong>控制-物理不匹配 (Control-physics mismatch)</strong>：当转移的动作在目标场景中导致碰撞时（例如，向前行驶然后左转），模型可能会 <strong>幻觉 (hallucinate)</strong> 场景变化以移除或改变障碍物，从而避免碰撞，但同时保留了预期运动。
2.  <strong>大范围揭示下的降级完成 (Degraded completion under large reveal)</strong>：像缩小镜头这样的动作需要合成大量新可见的内容。在这种情况下，视频的扩展部分（例如，玩家的腿）可能显得模糊或不一致。
3.  <strong>事件驱动动作的模糊实现 (Ambiguous realization for event-driven actions)</strong>：对于暗示事件的动作（例如，新角色进入），在跨上下文转移下，进入实体的身份未被指定。模型可能会将控制实现为背景/相机漂移，同时保持现有主体一致，这是一种合理的相对运动解释，但不是相同的事件语义。

## 7.3. 个人启发与批判
`Olaf-World` 提出的 `SeqΔ-REPA` 机制具有很强的直觉性和创新性，它从“动作的效果是可观测的”这一核心洞察出发，解决了长期困扰潜在动作学习的跨上下文不可识别性问题。这种“控制到效果”的对齐思想，通过利用预训练的自监督视频编码器提供的稳定语义特征作为锚点，巧妙地绕开了显式动作标签的依赖，为无监督视频世界模型的可转移性开辟了新路径。

**启发**：
1.  **效果驱动学习范式**：本文强调了“效果”在学习动作语义中的核心作用。这提示我们，在很多无标签或弱标签的任务中，与其直接模拟复杂的内部机制，不如关注其外部、可观测的“效果”来作为学习信号。这种范式可能适用于更广泛的领域，例如在机器人学中，我们可以关注机器人手臂末端执行器的轨迹变化，而非复杂的关节力矩。
2.  **通用特征表示的价值**：冻结的自监督视频编码器在 `SeqΔ-REPA` 中扮演了“通用语义参照系”的关键角色。这再次凸显了在大规模无标签数据上预训练的通用表示学习的重要性，它们可以作为下游任务的强大先验知识，甚至用于解决更深层次的语义对齐问题。
3.  **模型解耦与组合**：`Olaf-World` 的两阶段流程——先学习可转移的潜在动作，再用其条件化预训练的视频生成器，最后通过 `LoRA` 进行轻量级适应——展示了一种高效且模块化的模型开发策略。这种解耦可以使每个组件专注于其核心任务，并易于替换或升级。

    **批判与可以改进的地方**：
1.  **效果定义的局限性**：尽管 `SeqΔ-REPA` 的效果定义（帧间特征差异的平均）在本文场景下表现良好，但它可能过于简化。视频中的“效果”可以是多层次的：局部的、全局的、即时的、延迟的、物理性的、语义性的。例如，一个“按下开关”的动作，其即时视觉效果可能很小，但其语义效果（“灯亮了”）是延迟且抽象的。未来的工作可以探索更复杂、更具层次感的效果定义，甚至可以引入因果推理来区分真正的动作效果和伴随的视觉变化。
2.  **对冻结编码器的依赖**：`SeqΔ-REPA` 的性能在很大程度上依赖于冻结的自监督视频编码器（如 `V-JEPA 2`）所学习到的特征表示质量。如果该编码器在某些特定类型的动作或场景中表现不佳，可能会直接影响潜在动作的对齐效果。未来的工作可以研究如何使其对编码器的选择更鲁棒，或者探索共同优化编码器和 `LAM` 的可能性，尽管这会增加训练复杂性。
3.  **未能完全解决所有语义模糊**：论文提到在偏航“look”动作上仍存在混淆，因为自我中心和第三人称相机下相同的控制会引起不同的可观测运动。这表明，仅仅基于视觉效果差异可能无法完全解决所有语义层面的模糊性。对于某些动作，可能需要更深入的语义理解，甚至结合多模态信息（如文本描述）才能实现完美对齐。
4.  **规划和推理的进一步探索**：论文在未来工作中提到了潜在动作在规划和推理中的应用，但当前工作主要聚焦于动作表示学习和生成。将 `Olaf-World` 的潜在动作真正应用于复杂的长时域规划和决策，是其从“可控生成”迈向“智能决策”的关键一步，这将涉及如何有效地在潜在动作空间中进行搜索、评估和优化。
5.  **失败案例的深层原因**：论文提到的失败案例（控制-物理不匹配、大范围揭示下的降级完成、事件驱动动作的模糊实现）揭示了当前模型在物理常识、长时域连贯性以及高层语义理解方面的不足。未来的工作需要进一步增强模型对物理世界规律的建模能力，以及对抽象事件和实体身份的理解，而不仅仅是像素级的动态。

    总体而言，`Olaf-World` 在解决视频世界模型的可转移性瓶颈方面迈出了重要一步，其提出的 `SeqΔ-REPA` 为无监督动作表示学习提供了一个强大的新工具。