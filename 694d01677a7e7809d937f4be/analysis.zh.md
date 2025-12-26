# 1. 论文基本信息

## 1.1. 标题
**Memorize-and-Generate: Towards Long-Term Consistency in Real-Time Video Generation**
（记忆与生成：迈向实时视频生成中的长期一致性）

## 1.2. 作者
Tianrui Zhu\*, Shiyi Zhang\*, Zhirui Sun, Jingqi Tian, Yansong Tang（清华大学深圳国际研究生院）。作者在计算机视觉和多媒体处理领域具有深厚背景。

## 1.3. 发表期刊/会议
该论文发布于 2024 年 12 月（根据提供的日期 2025-12-21，推测为预印本或即将发表于 2025 年顶级会议，如 CVPR/ICCV）。

## 1.4. 发表年份
2025年（预印本发布时间为 2024 年底至 2025 年初）。

## 1.5. 摘要
帧级自回归 (Frame-level autoregressive, frame-AR) 模型在实时视频生成中取得了显著进展，但长视频生成面临着“内存消耗”与“历史一致性”的权衡。现有方法多采用窗口注意力 (Window attention)，容易导致“灾难性遗忘”。本文提出了 **MAG (Memorize-and-Generate)** 框架，将内存压缩和帧生成解耦。通过训练专门的<strong>内存模型 (Memory model)</strong> 将历史信息压缩为紧凑的 <strong>键值缓存 (KV cache)</strong>，并由<strong>生成器模型 (Generator model)</strong> 利用该缓存合成后续帧。此外，本文引入了 **MAG-Bench** 基准测试，专门评估历史记忆保持能力。实验证明，MAG 在保持实时性能的同时，显著提升了场景的一致性。

## 1.6. 原文链接
- **原文链接:** [https://arxiv.org/abs/2512.18741](https://arxiv.org/abs/2512.18741)
- **PDF 链接:** [https://arxiv.org/pdf/2512.18741v2.pdf](https://arxiv.org/pdf/2512.18741v2.pdf)
- **发布状态:** 预印本 (v2 版本)。

  ---

# 2. 整体概括

## 2.1. 研究背景与动机
*   **核心问题:** 视频生成模型正从双向注意力（计算慢，难以生成长视频）转向<strong>自回归 (Autoregressive)</strong> 模式（逐帧生成，速度快）。然而，自回归模型在生成长视频时，为了节省显存（GPU Memory），不得不丢弃较远的历史帧（即“滑动窗口”策略）。这会导致模型“转头就忘”——例如相机向左转后再转回右边，原来的场景可能已经完全变样了。
*   **挑战:** 
    1.  **内存爆炸:** 保留所有历史帧的键值缓存（KV cache）会迅速占满显存。
    2.  <strong>退化解 (Degenerate Solution):</strong> 在训练过程中，模型往往过度依赖文本提示词（Text Prompt），而忽视了历史图像信息，导致物理逻辑不连贯。
*   **创新思路:** 本文提出将“记住过去”和“生成未来”分工。先通过一个模型把过去的画面压缩成极小的“记忆精华”，再让另一个模型根据这些“精华”来续写视频。

## 2.2. 核心贡献/主要发现
*   **MAG 框架:** 提出了“解耦记忆与生成”的新范式，通过内存模型实现 3 倍的缓存压缩，且几乎无损。
*   **历史一致性强化:** 引入了<strong>无文本条件 (Text-free condition)</strong> 损失函数，强迫模型在没有文字指导的情况下也能根据历史画面推断后续内容。
*   **MAG-Bench 基准:** 填补了评价“场景回溯（相机离开又回来）”一致性的空白，是目前评估长视频记忆能力的有效工具。
*   **卓越性能:** 在单张 H100 GPU 上实现了 21.7 FPS 的生成速度，达到了真正的实时性。

    ---

# 3. 预备知识与相关工作

## 3.1. 基础概念
*   <strong>自回归生成 (Autoregressive Generation):</strong> 一种生成方式，模型根据已经生成的 $n$ 帧来预测第 $n+1$ 帧。类似于人类写文章，写完一个词再写下一个。
*   <strong>键值缓存 (KV cache):</strong> 在 Transformer 架构中，为了避免重复计算已经处理过的词（或图像块）的中间特征，将其存储在显存中。长视频生成中，KV 缓存的体积随视频长度线性增加。
*   <strong>分发匹配蒸馏 (Distribution Matching Distillation, DMD):</strong> 一种模型压缩/加速技术。让一个简单的“学生”模型去模仿一个复杂的“老师”模型（如传统的扩散模型）的输出分布，从而将几十步的降噪过程缩减到 1-4 步。
*   <strong>自强迫 (Self Forcing):</strong> 一种训练策略，确保模型在训练时看到的历史数据和推理时产生的历史数据分布一致，从而减少误差累积。

## 3.2. 前人工作
*   **双向注意力模型:** 如 `Wan2.1`。虽然质量高，但计算量大（复杂度随长度平方增长），生成几秒视频需要数分钟，无法实时。
*   **自回归扩散模型:** 如 `Self Forcing` 和 `LongLive`。通过 DMD 蒸馏实现了实时生成，但它们大多使用滑动窗口（只看最近几帧），导致长期记忆缺失。
*   **显式 3D 记忆:** 将视频投影成 3D 点云。优点是一致性好，缺点是渲染算法复杂且在动态场景下容易崩溃。

## 3.3. 差异化分析
相比于 `LongLive` 等方法通过简单丢弃历史来实现实时，**MAG** 通过**可学习的压缩**保留了所有历史。相比于 `TTT-video` 通过在推理时更新模型权重来记忆（速度慢），**MAG** 保持了恒定的推理速度，真正兼顾了“记得久”和“跑得快”。

---

# 4. 方法论

## 4.1. 方法原理
MAG 的核心思想是：**信息是有冗余的**。视频中的连续几帧其实包含大量重复信息，不需要全部原始存储。通过一个专门训练的“内存模型”，我们可以将多帧的特征压缩到一个极小的特征空间中，同时保留重建这些帧的能力。

## 4.2. 核心方法详解 (逐层深入)

### 4.2.1. 重新思考长视频生成中的 DMD 优化
作者发现，现有的长视频蒸馏方法存在一个**退化解**问题。
在标准 DMD 蒸馏中，优化目标是最小化学生模型分布 $p_{\boldsymbol{\theta}}^{\mathcal{S}}$ 与老师模型分布 $p^{\mathcal{T}}$ 之间的 KL 散度。其梯度近似公式为：
$$
\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\mathrm{DMD}} \approx \mathbb{E}_{i \sim U\{1, k\}} \mathbb{E}_{z \sim \mathcal{N}(0, I)} \left[ s^{\mathcal{T}}(\boldsymbol{x}_{i}) - s_{\boldsymbol{\theta}}^{\mathcal{S}}(\boldsymbol{x}_{i}) \frac{d G_{\boldsymbol{\theta}}(\boldsymbol{z}_{i})}{d \boldsymbol{\theta}} \right]
$$
*   **公式解释:**
    *   $\boldsymbol{x}_i$: 从长视频中随机采样的短片段。
    *   $s^{\mathcal{T}}$ 和 $s_{\boldsymbol{\theta}}^{\mathcal{S}}$: 分别代表老师和学生模型的评分函数 (Score functions)，用于指导降噪方向。
    *   $G_{\boldsymbol{\theta}}(\boldsymbol{z}_i)$: 生成器的输出。

        **问题在于:** 老师模型通常是基于文本生成的短视频模型，它不具备“理解长历史”的能力。学生模型在学习时会发现，只要根据文本就能生成好看的画面，何必去费力理解复杂的历史 KV 缓存呢？于是模型学会了偷懒（只看文本，不看历史）。

为了解决这个问题，作者引入了<strong>历史一致性强化损失 (History consistency reinforcement loss)</strong>：
$$
\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\mathrm{history}} = \mathbb{E}_{\boldsymbol{x} \sim p_{\boldsymbol{\theta}}^{G}(\boldsymbol{x} | h, \emptyset)} [\nabla_{\boldsymbol{\theta}} D_{KL}(p_{\boldsymbol{\theta}}^{S}(\boldsymbol{x}) \| p^{\mathcal{T}}(\boldsymbol{x}))]
$$
*   **公式解释:**
    *   $\emptyset$: 代表**空文本条件**。
    *   $h$: 历史帧。
    *   **核心逻辑:** 强迫生成器在没有文本指导（即文本为空）的情况下，仅根据历史信息 $h$ 来生成下一帧。如果生成得好，说明模型真正掌握了历史一致性。

        最终的综合损失函数为：
$$
\nabla_{\boldsymbol{\theta}} \mathcal{L} = (1 - \lambda) \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\mathrm{DMD}} + \lambda \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\mathrm{history}}
$$
其中 $\lambda$ 是平衡因子，通过随机采样实现。

### 4.2.2. 第一阶段：内存模型训练 (Memory Model)
目标是将一个块 (Block) 内的多帧（例如 3 帧）压缩。
*   **编码过程:** 利用块内全注意力 (Intra-block full attention)，将所有帧的信息汇聚到最后一帧的 KV 缓存中。
*   **解码过程:** 要求模型根据这个压缩后的 KV 缓存，通过降噪重建出块内所有原始帧的像素。
*   <strong>注意力掩码 (Attention Mask):</strong> 如下图（原文 Figure 3）所示，作者设计了特殊的掩码，确保模型只能通过目标缓存来重建信息，从而实现强制压缩。

    ![Fig. 3: The attention mask of memory model training. We achieve efficient parallel training of the encode-decode process by concatenating noise and clean frame sequences. By masking out the KV cache of other frames within the block, the model is forced to compress information into the target cache.](images/3.jpg)
    *该图像是示意图，展示了记忆模型训练的注意力掩码。通过将噪声和干净帧序列连接，实现了编码-解码过程的高效并行训练。图中通过屏蔽出块内其他帧的KV缓存，迫使模型将信息压缩到目标缓存中。*

### 4.2.3. 第二阶段：生成模型训练 (Generator Model)
在内存模型冻结（参数固定）后，训练生成模型。生成模型学会如何读取这些压缩后的缓存来合成全新的未来帧。这一阶段遵循长视频生成的自回归流程，并应用了 4.2.1 节中提到的历史强化策略。

下图（原文 Figure 2）展示了这两个阶段的完整训练流水线：

![Fig. 2: The training pipeline. The training process of MAG comprises two stages. In the first stage, we train the memory model for the triple compressed KV cache, retaining only one frame within a full attention block. The loss function requires the model to reconstruct the pixels of all frames in the block from the compressed cache. The process utilizes a customized attention mask to achieve efficient parallel training. In the second stage, we train the generator model within the long video DMD training framework to adapt to the compressed cache provided by the frozen memory model.](images/2.jpg)
*该图像是示意图，展示了MAG训练过程的两个阶段。第一阶段中，训练内存模型以生成三重压缩的KV缓存，并重建像素，使用自定义的注意力掩码实现并行训练。第二阶段训练生成模型，以适应由固定的内存模型提供的压缩缓存。*

---

# 5. 实验设置

## 5.1. 数据集
1.  **VPData:** 包含 39 万条高质量真实世界视频，用于训练内存模型的重建能力。
2.  **VidProM:** 百万级真实提示词数据集，用于生成模型的文本对齐训练。
3.  <strong>MAG-Bench (自建):</strong> 包含 176 个视频，涵盖室内、室外、物体和游戏场景。其特点是相机路径包含“离开后再返回”的轨迹，专门测试历史一致性。

## 5.2. 评估指标
*   <strong>PSNR (Peak Signal-to-Noise Ratio, 峰值信噪比):</strong>
    *   **定义:** 衡量图像重建质量的经典指标，数值越高代表失真越小。
    *   **公式:** $\mathrm{PSNR} = 10 \cdot \log_{10}\left(\frac{MAX^2}{MSE}\right)$
    *   **符号:** `MAX` 是像素最大值，`MSE` 是均方误差。
*   <strong>SSIM (Structural Similarity Index, 结构相似性):</strong>
    *   **定义:** 从亮度、对比度和结构三个维度衡量两幅图像的相似性，更符合人类视觉感知。
    *   **公式:** $\mathrm{SSIM}(x,y) = \frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}{(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}$
*   **LPIPS (Learned Perceptual Image Patch Similarity):**
    *   **定义:** 利用深度学习特征计算感官相似度，越低代表两张图看起来越像。
*   **VBench:** 综合评估视频质量、语义对齐、背景和主体一致性的标准工具。

## 5.3. 对比基线
*   **Wan2.1:** 基础模型，非实时。
*   **Self Forcing:** 实时生成的开山之作，但长视频一致性较差。
*   **LongLive:** 最新的实时长视频生成模型，使用滑动窗口。

    ---

# 6. 实验结果与分析

## 6.1. 核心结果分析
*   **实时性:** MAG 在 H100 上达到 **21.7 FPS**，不仅超过了基础模型，甚至比同样是实时的 `LongLive` 还要快。
*   **一致性:** 在 `VBench-Long` 测试中，MAG 在背景（97.99）和主体一致性（99.18）上全面领先。
*   **场景回溯:** 在 MAG-Bench 上，由于 MAG 保留了压缩后的全量历史，其重现先前场景的能力远超窗口注意力模型（见下图 Figure 7）。

    以下是原文 **Table 1**（5秒短视频性能对比）的完整转录：

    <table>
    <thead>
    <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">Throughput FPS↑</th>
    <th colspan="5">Vbench scores on 5s ↑</th>
    </tr>
    <tr>
    <th>Total</th>
    <th>Quality</th>
    <th>Semantic</th>
    <th>Background</th>
    <th>Subject</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td colspan="7" style="background-color: #f2f2f2;"><b>Multi-step model (非蒸馏模型)</b></td>
    </tr>
    <tr>
    <td>SkyReels-V2</td>
    <td>0.49</td>
    <td>82.67</td>
    <td>84.70</td>
    <td>74.53</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>Wan2.1 (Base)</td>
    <td>0.78</td>
    <td>84.26</td>
    <td>85.30</td>
    <td>80.09</td>
    <td>97.29</td>
    <td>96.34</td>
    </tr>
    <tr>
    <td colspan="7" style="background-color: #f2f2f2;"><b>Few-step distillation model (蒸馏加速模型)</b></td>
    </tr>
    <tr>
    <td>CausVid</td>
    <td>17.0</td>
    <td>82.46</td>
    <td>83.61</td>
    <td>77.84</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>Self Forcing</td>
    <td>17.0</td>
    <td>83.98</td>
    <td>84.75</td>
    <td>80.86</td>
    <td>96.21</td>
    <td>96.80</td>
    </tr>
    <tr>
    <td>Longlive</td>
    <td>20.7</td>
    <td>83.32</td>
    <td>83.99</td>
    <td>80.68</td>
    <td>96.41</td>
    <td>96.54</td>
    </tr>
    <tr>
    <td><b>MAG (Ours)</b></td>
    <td><b>21.7</b></td>
    <td>83.52</td>
    <td>84.11</td>
    <td>81.14</td>
    <td><b>97.44</b></td>
    <td><b>97.02</b></td>
    </tr>
    </tbody>
    </table>

以下是原文 **Table 3**（MAG-Bench 历史一致性定量测试）的结果：
该实验对比了模型在有无压缩训练情况下的表现。

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="3">History Context Comparison</th>
<th colspan="3">Ground Truth Comparison</th>
</tr>
<tr>
<th>PSNR↑</th>
<th>SSIM↑</th>
<th>LPIPS↓</th>
<th>PSNR↑</th>
<th>SSIM↑</th>
<th>LPIPS↓</th>
</tr>
</thead>
<tbody>
<tr>
<td>Self Forcing</td>
<td>14.46</td>
<td>0.48</td>
<td>0.49</td>
<td>15.65</td>
<td>0.51</td>
<td>0.42</td>
</tr>
<tr>
<td>Longlive</td>
<td>16.42</td>
<td>0.53</td>
<td>0.32</td>
<td>18.92</td>
<td>0.62</td>
<td>0.22</td>
</tr>
<tr>
<td><b>MAG (Ours)</b></td>
<td><b>18.99</b></td>
<td><b>0.60</b></td>
<td><b>0.23</b></td>
<td><b>20.77</b></td>
<td><b>0.66</b></td>
<td><b>0.17</b></td>
</tr>
</tbody>
</table>

## 6.2. 消融实验
作者测试了不同的压缩倍率。如 **Table 4** 所示，当压缩率为 3（即 3 帧压成 1 帧的缓存）时，重建质量 PSNR 为 31.73，视觉上几乎无损。随着倍率增加到 5，质量开始明显下降。这证明了 3 倍压缩是目前的性能“甜点位”。

---

# 7. 总结与思考

## 7.1. 结论总结
MAG 成功解决了一个困扰长视频自回归模型已久的问题：**如何用有限的显存记住无限的过去**。
*   通过**内存模型**，将显存压力降低了 3 倍。
*   通过**历史一致性强化损失**，解决了模型“依赖文本、忽视图像”的训练退化问题。
*   在实时性、图像质量和长期一致性三个维度上都取得了当前最先进的 (state-of-the-art) 表现。

## 7.2. 局限性与未来工作
1.  **数据依赖:** 目前的生成器在选择历史信息时仍显被动，未来需要更多包含复杂交互和逻辑因果的数据来训练。
2.  **交互式世界模型:** 目前的 DMD 框架还是“无数据”的纯模仿，难以直接扩展到带有动作输入（Action-based）的游戏引擎或世界模型中，这需要更强大的老师模型和真实的动作序列数据。

## 7.3. 个人启发与批判
*   **启发:** “解耦”思想在深度学习中屡试不爽。将记忆存储（存储效率）与记忆应用（生成逻辑）分开训练，比强行让一个模型完成所有任务更有效。
*   **批判:** 虽然论文宣称 3 倍压缩，但对于超长视频（如数小时），KV 缓存依然会线性增长。未来或许需要引入类似人类大脑的“分级记忆”——最近的精细记，远处的模糊记。此外，MAG-Bench 虽然巧妙，但 176 条视频的规模依然较小，其泛化性仍有待更大规模的数据集验证。