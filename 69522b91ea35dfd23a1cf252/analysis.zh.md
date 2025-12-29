# 1. 论文基本信息

## 1.1. 标题
**StoryMem: Multi-shot Long Video Storytelling with Memory**
（StoryMem：带有记忆机制的多镜头长视频故事生成）

## 1.2. 作者
**Kaiwen Zhang $^{1,2,*}$, Liming Jiang $^{2,\dagger}$, Angtian Wang $^{2}$, Jacob Zhiyuan Fang $^{2}$, Tiancheng Zhi $^{2}$, Qing Yan $^{2}$, Hao Kang $^{2}$, Xin Lu $^{2}$, Xingang Pan $^{1,3}$**
*   **隶属机构：** 1. 南洋理工大学 S-Lab；2. 字节跳动智能创作团队；3. 上海人工智能实验室。
*   **背景说明：** 该研究由字节跳动主导，第一作者张凯文在字节跳动实习期间完成了主要工作。

## 1.3. 发表期刊/会议
该论文目前发布于 **arXiv** 预印本平台（发布日期：2025年12月23日）。其技术深度和实验规模显示其目标可能是 CVPR 或 ICCV 等计算机视觉顶会。

## 1.4. 发表年份
**2025年**（注：原文日期显示为 2025 年 12 月 23 日，代表该研究处于该领域的最新前沿）。

## 1.5. 摘要
视觉故事讲述要求生成的视频不仅具有电影级的画质，还要具备长期的跨镜头一致性。受人类记忆机制启发，作者提出了 **StoryMem**，这是一种将长视频生成重新定义为“基于显式视觉记忆的迭代镜头合成”的新范式。通过一种创新的 <strong>从记忆到视频 (Memory-to-Video, M2V)</strong> 设计，StoryMem 维护一个由历史关键帧组成的动态记忆库，并通过 <strong>潜空间拼接 (Latent Concatenation)</strong> 和 <strong>负向旋转位置编码偏移 (Negative RoPE Shifts)</strong> 将记忆注入预训练的单镜头视频扩散模型中。实验表明，StoryMem 在保持高画质的同时，实现了卓越的跨镜头一致性，是迈向分钟级连贯视频故事生成的重要一步。

## 1.6. 原文链接
*   **arXiv 链接:** [https://arxiv.org/abs/2512.19539](https://arxiv.org/abs/2512.19539)
*   **PDF 链接:** [https://arxiv.org/pdf/2512.19539v1.pdf](https://arxiv.org/pdf/2512.19539v1.pdf)
*   **项目主页:** [https://kevin-thu.github.io/StoryMem](https://kevin-thu.github.io/StoryMem)

    ---

# 2. 整体概括

## 2.1. 研究背景与动机
目前的视频生成模型（如 Sora, Kling, Wan2.1）在生成单个 5-10 秒的短片时表现出色，但在生成包含多个镜头的长视频故事时面临巨大挑战：
1.  <strong>连贯性困境 (Consistency Dilemma):</strong> 故事需要角色、场景和艺术风格在不同镜头间保持一致。
2.  **现有方法的局限:**
    *   <strong>联合训练法 (Joint Training):</strong> 尝试一次性训练生成长视频。缺点是计算成本随长度呈平方级增长，且缺乏高质量的长视频训练数据，容易导致画质下降。
    *   <strong>解耦生成法 (Decoupled Generation):</strong> 先生成关键帧图像，再用图生视频 (I2V) 模型扩展。缺点是每个镜头独立生成，缺乏上下文信息，导致镜头切换生硬，角色细节在后期发生漂移。

        **StoryMem 的切入点：** 既然人类通过记忆来维持对故事的认知，为什么不给视频模型加一个“记忆库”？

## 2.2. 核心贡献/主要发现
1.  **StoryMem 范式:** 提出将长视频生成视为“迭代镜头合成”，通过显式视觉记忆库实现镜头间的通信。
2.  **M2V 设计:** 引入 <strong>负向旋转位置编码偏移 (Negative RoPE Shifts)</strong>，让模型在生成当前镜头时能“回看”过去的记忆帧，且无需对模型架构做大规模改动。
3.  **记忆管理策略:** 结合 <strong>语义关键帧选择 (Semantic Keyframe Selection)</strong> 和 <strong>美学偏好过滤 (Aesthetic Preference Filtering)</strong>，确保记忆库既精简又高质量。
4.  **ST-Bench 评估基准:** 贡献了一个包含 30 个复杂故事剧本、300 个详细提示词的多镜头视频故事生成评估基准。

    下图（原文 Figure 1）展示了 StoryMem 生成的具有高度一致性的多镜头视频示例：

    ![Figur1 Given a story sript with per-shot text descriptions, StoryMem generates appealing minute-ong, multi-shot iv uayTro o generation using a memory-conditioned single-shot video diffusion model.](images/1.jpg)
    *该图像是一个插图，展示了不同故事情节的多镜头视频生成示例，包含了《王子与公主》、《小机器人》、《小美人鱼》和《鲁滨逊漂流记》的情节。每个故事都通过不同的镜头呈现，反映出故事发展的关键时刻。*

---

# 3. 预备知识与相关工作

## 3.1. 基础概念
*   <strong>视频扩散模型 (Video Diffusion Model):</strong> 一种通过模拟扩散过程（加噪）和逆扩散过程（去噪）来生成视频的 AI 模型。
*   <strong>扩散转换器 (Diffusion Transformer, DiT):</strong> 结合了 Transformer 架构和扩散模型。模型不再使用传统的卷积网络（如 U-Net），而是使用 Transformer 处理视频数据的 <strong>词元 (token)</strong>。
*   <strong>旋转位置编码 (Rotary Position Embedding, RoPE):</strong> 一种在 Transformer 中为每个词元标记位置的方法。它通过旋转向量来表示相对位置关系。
*   <strong>潜空间 (Latent Space):</strong> 视频数据通过 <strong>变分自编码器 (VAE)</strong> 压缩后的低维表示，扩散过程在这里进行以节省计算力。

## 3.2. 前人工作
*   <strong>Wan2.2 (主干网络):</strong> 本文基于 Wan2.2 图生视频 (I2V) 模型构建。该模型使用了 **3D-RoPE** 来处理时空位置。
*   <strong>整流流 (Rectified Flow, RF):</strong> 一种改进的扩散训练目标，它试图让生成路径变成一条直线，从而加快采样速度。其损失函数为：
    $$ \mathcal{L}_{RF} = \mathbb{E}_{z_0, \epsilon, t} \left[ \| v_{\Theta}(z_t, t) - (z_0 - \epsilon) \|^2 \right] $$
    其中 $z_0$ 是真实视频，$z_t$ 是加噪后的视频，$v_{\Theta}$ 是模型预测的速度场。

## 3.3. 差异化分析
相比于 **HoloCine**（联合训练长视频）和 **StoryDiffusion**（图像连贯但视频独立），**StoryMem** 的核心区别在于：它不需要超长视频数据训练，而是通过轻量级的 <strong>微调 (fine-tuning)</strong> 和推理时的 **记忆更新** 来实现长程连贯性。

---

# 4. 方法论

## 4.1. 方法原理
StoryMem 的核心思想是将长视频生成分解为一系列受记忆约束的条件生成任务。

## 4.2. 核心方法详解 (逐层深入)

### 4.2.1. 问题形式化
我们将生成 $N$ 个镜头的视频 $\mathcal{V} = \{v_1, \dots, v_N\}$ 的过程，根据概率论的链式法则分解为自回归形式：
$$ p_{\Theta}(\mathcal{V} \mid \mathcal{T}) = \prod_{i=1}^{N} p_{\Theta}(v_i \mid v_{1:i-1}, \mathcal{T}) $$
其中 $\mathcal{T}$ 是文本描述。由于视频冗余度高，直接以过去所有视频为条件计算量太大。StoryMem 引入了 <strong>记忆库 (Memory Bank)</strong> $m_{i-1}$ 来摘要过去的视觉信息：
$$ p_{\Theta}(\mathcal{V} \mid \mathcal{T}) \approx \prod_{i=1}^{N} p_{\Theta}(v_i \mid t_i, m_{i-1}) $$

### 4.2.2. 从记忆到视频 (M2V) 架构设计
为了让模型感知记忆，作者在 Wan2.2-I2V 的基础上进行了改进。
1.  **记忆注入:** 将记忆库中的关键帧通过 3D VAE 编码为 <strong>记忆潜变量 (Memory Latents)</strong> $z_m$。
2.  **特征拼接:** 将 $z_m$ 与当前镜头的带噪潜变量 $z_t$ 在时间维度上进行拼接。为了区分哪些是已知记忆，哪些是待生成内容，还引入了一个二进制 <strong>掩码 (Mask)</strong> $M$。
3.  <strong>负向旋转位置编码偏移 (Negative RoPE Shift):</strong> 这是本文最巧妙的设计。如果直接给记忆帧分配 0, 1, 2... 的索引，会破坏当前镜头的时序感。作者给记忆帧分配了 **负索引**。
    假设当前镜头有 $f$ 帧，记忆有 $f_m$ 帧，时间索引序列设为：
    $$ \{-f_m S, -(f_m-1) S, \dots, -S, 0, 1, \dots, f-1\} $$
    其中 $S$ 是一个固定的偏移量（Offset）。这样，当前镜头依然从 0 开始，而记忆被模型视为“发生在过去的事情”。

### 4.2.3. 记忆库的动态维护
为了防止记忆库无限增长并保证其质量，StoryMem 采用了以下策略：
*   <strong>语义关键帧选择 (Semantic Keyframe Selection):</strong> 使用 **CLIP** 计算帧间的余弦相似度。只有当新帧与现有记忆的相似度低于阈值时，才将其存入记忆库。
*   **美学过滤:** 使用 **HPSv3 (Human Preference Score v3)** 对候选关键帧打分，剔除模糊或低美感的帧。
*   **混合管理机制:** 采用 <strong>记忆汇点 (Memory Sink)</strong> + <strong>滑动窗口 (Sliding Window)</strong> 策略。最早生成的几帧作为“长期锚点”永久保留（Memory Sink），而最近的几帧在滑动窗口中动态更新。

    下图（原文 Figure 2）展示了 StoryMem 的整体架构和数据流：

    ![Figure 2 Overview of StoryMem. StoryMem generates each shot conditioned on a memory bank that stores keams fom previusy nrate shots. Duri eneration, the seecmemory ame are ncoded b 3D VAE, fused with noisy video latents and binary masks, and fed into a LoRA-finetuned memory-conditioned Video DiT to apivo narrative progression. Byerativey enerating shots wihmemory updatesStoryMem produce coherent minue-n, multi-shot story videos.](images/2.jpg)
    *该图像是示意图，展示了StoryMem的工作流程。它描述了如何通过内存银行生成视频镜头，其中包括3D VAE编码器、LoRA微调以及视觉记忆的更新过程。此外，还涉及到语义关键帧选择和美学偏好过滤，以提升生成的视频质量。*

---

# 5. 实验设置

## 5.1. 数据集
*   **训练数据:** 使用了 40 万条 5 秒长的单镜头视频。为了模拟记忆，从同一视频组中随机抽取 1-10 帧作为记忆帧，训练模型根据这些帧和提示词重建目标视频。
*   **评估基准 ST-Bench:** 作者利用 GPT-5 生成了 30 个完整剧本，涵盖了写实、童话、古代、现代等多种风格。

## 5.2. 评估指标
1.  <strong>美学质量 (Aesthetic Quality):</strong>
    *   **概念定义:** 量化视频的视觉吸引力，包括色彩平衡、清晰度和真实感。
    *   **计算方式:** 使用在 LAION 数据集上预训练的美学预测器打分。
2.  <strong>提示词遵循 (Prompt Following):</strong>
    *   **概念定义:** 衡量生成的视频与文本描述的一致程度。
    *   **计算公式:** 计算视频特征向量 $v_{feat}$ 与文本特征向量 $t_{feat}$ 之间的余弦相似度：
        $$ S_{Prompt} = \frac{v_{feat} \cdot t_{feat}}{\|v_{feat}\| \|t_{feat}\|} $$
    *   **说明:** 使用 ViCLIP 模型提取特征。
3.  <strong>跨镜头一致性 (Cross-shot Consistency):</strong>
    *   **概念定义:** 衡量不同镜头之间在视觉特征上的稳定性。
    *   **计算公式:** 计算所有镜头对 $(v_i, v_j)$ 之间的平均特征相似度。

## 5.3. 对比基线
*   **Wan2.2-T2V:** 原始单镜头模型（完全不考虑一致性）。
*   **StoryDiffusion / IC-LoRA:** 先生成连贯图像再扩展为视频的二阶段方法。
*   **HoloCine:** 最先进的 (state-of-the-art) 联合训练长视频生成模型。

    ---

# 6. 实验结果与分析

## 6.1. 核心结果分析
下表（原文 Table 1）展示了在 ST-Bench 上的量化对比结果：

<table>
<thead>
<tr>
<th rowspan="2">方法 (Method)</th>
<th rowspan="2">美学质量 (Aesthetic) ↑</th>
<th colspan="2">提示词遵循 (Prompt Following) ↑</th>
<th colspan="2">跨镜头一致性 (Consistency) ↑</th>
</tr>
<tr>
<th>全局 (Global)</th>
<th>单镜头 (Single-shot)</th>
<th>总体 (Overall)</th>
<th>Top-10 对</th>
</tr>
</thead>
<tbody>
<tr>
<td>Wan2.2 (独立生成)</td>
<td>0.5962</td>
<td>0.2115</td>
<td>**0.2372**</td>
<td>0.3934</td>
<td>0.4578</td>
</tr>
<tr>
<td>StoryDiffusion + Wan2.2</td>
<td>0.6085</td>
<td>0.2234</td>
<td>0.2305</td>
<td>0.4357</td>
<td>0.4852</td>
</tr>
<tr>
<td>IC-LoRA + Wan2.2</td>
<td>0.6053</td>
<td>0.2241</td>
<td>0.2307</td>
<td>0.4468</td>
<td>0.4933</td>
</tr>
<tr>
<td>HoloCine</td>
<td>0.5734</td>
<td>0.2217</td>
<td>0.2211</td>
<td>0.4629</td>
<td>0.5117</td>
</tr>
<tr>
<td>**StoryMem (Ours)**</td>
<td>**0.6133**</td>
<td>**0.2289**</td>
<td>0.2313</td>
<td>**0.5065**</td>
<td>**0.5337**</td>
</tr>
</tbody>
</table>

**分析点：**
1.  **一致性飞跃:** StoryMem 在“总体一致性”上超过了 HoloCine 约 **9.4%**，相比独立生成的 Wan2.2 提升了 **28.7%**。这证明了显式记忆比隐式联合训练更有效。
2.  **画质保持:** 相比 HoloCine 因为联合训练导致画质下降（0.5734），StoryMem（0.6133）甚至超越了原始模型，这得益于 LoRA 微调和美学过滤记忆的引导。

## 6.2. 消融实验
作者验证了各个组件的必要性（见原文 Table 2）：
*   **去语义选择:** 角色一致性变差，因为记忆中可能漏掉了关键的角色特写。
*   **去美学过滤:** 画质下降，因为模型可能会被记忆库中模糊的帧误导。
*   <strong>去记忆汇点 (Memory Sink):</strong> 长视频后期的一致性会发生崩溃。

    下图（原文 Figure 3）对比了不同方法在维持角色一致性方面的差异：

    ![Figure 3 Qualitative comparison. Our StoryMem generates coherent multi-scene, multi-shot story videos aligned w per-ho eptions.n ontrast,he reraimode ankeambasbaselnes preeveo character and scene consistency, while HoloCine \[31\] exhibits noticeable degradation in visual quality.](images/3.jpg)
    *该图像是一个示意图，展示了使用StoryMem生成的多场景多镜头故事视频。这些镜头展现了不同的场景，分别采用了各种拍摄技巧和镜头长度，以突显视频的连贯性和美学品质。图中比较了不同方法的输出效果，包括HoloCine的表现。*

---

# 7. 总结与思考

## 7.1. 结论总结
StoryMem 成功地证明了：**不需要昂贵的长视频联合训练，通过给单镜头模型增加“视觉记忆”和轻量级微调，就能生成连贯且高质量的多镜头长视频故事。** 这种基于记忆的迭代生成范式，极大地降低了生成连贯长视频的门槛。

## 7.2. 局限性与未来工作
*   **复杂多角色挑战:** 在多个角色频繁互动的场景下，仅靠视觉记忆有时会产生歧义。作者建议未来加入“实体感知 (Entity-aware)”的记忆表示（见 Figure 7）。
*   **动作连贯性:** 虽然视觉一致性很好，但在某些镜头切换处，如果前后的动作幅度非常大，物理规律的衔接（如运动速度）可能不够完美。

## 7.3. 个人启发与批判
*   **启发:** 负向 RoPE 偏移的设计非常具有启发性。它通过坐标空间的巧妙映射，解决了 Transformer 无法区分“历史”和“现状”的难题。
*   **批判:** 目前的记忆是完全自动选择的。虽然文中提到了可以引入人类干预，但如果能进一步结合 <strong>大语言模型 (LLM)</strong> 自动分析剧情重要性来筛选记忆帧，系统可能会更加鲁棒。
*   **迁移性:** 这种 M2V 思想不仅可以用于生成视频，也可以用于视频编辑（保持编辑前后的一致性）或机器人视觉导航。