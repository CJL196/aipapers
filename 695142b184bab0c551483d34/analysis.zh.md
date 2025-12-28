# 1. 论文基本信息

## 1.1. 标题
**DanceGRPO: Unleashing GRPO on Visual Generation**
（DanceGRPO：在视觉生成领域释放 GRPO 的潜力）

## 1.2. 作者
Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, Ping Luo。
作者隶属于 <strong>字节跳动 Seed (ByteDance Seed)</strong> 和 <strong>香港大学 (The University of Hong Kong)</strong>。

## 1.3. 发表期刊/会议
本项目发布于 **arXiv** 预印本平台（截至目前为 v4 版本）。考虑到其背后的字节跳动团队及其在视觉生成（如 HunyuanVideo）领域的深厚背景，该研究在生成式 AI 社区具有极高的关注度和影响力。

## 1.4. 发表年份
**2025年**（初稿发布于 2025 年 5 月 1 日，更新于 5 月 12 日）。

## 1.5. 摘要
尽管生成式 AI 在视觉内容创作上取得了巨大进步，但使模型输出与人类偏好对齐仍是一个关键挑战。现有的基于强化学习（RL）的微调方法（如 `DDPO` 和 `DPOK`）在扩展到大规模、多样化提示词集时面临稳定性差的瓶颈。本文提出了 **DanceGRPO** 框架，首次将 <strong>组相对策略优化 (Group Relative Policy Optimization, GRPO)</strong> 引入视觉生成任务。通过将扩散模型和整流流（Rectified Flows）的采样过程重新构建为随机微分方程 (SDE)，DanceGRPO 在多种生成范式（如 SD、FLUX、HunyuanVideo）上实现了稳定且高效的策略优化。实验表明，DanceGRPO 在美学、图文对齐和运动质量等指标上显著优于现有基线，最高提升达 181%。

## 1.6. 原文链接
- **原文链接:** [https://arxiv.org/abs/2505.07818](https://arxiv.org/abs/2505.07818)
- **PDF 链接:** [https://arxiv.org/pdf/2505.07818v4.pdf](https://arxiv.org/pdf/2505.07818v4.pdf)
- **项目主页:** [https://dancegrpo.github.io/](https://dancegrpo.github.io/)

  ---

# 2. 整体概括

## 2.1. 研究背景与动机
*   **核心问题:** 如何让视觉生成模型（如生成图片和视频的模型）生成的产物更符合人类的胃口（即“对齐”）？
*   **重要性:** 虽然预训练模型可以生成高质量图像，但它们往往无法精准遵循复杂的指令，或者生成的内容在美学上不够极致。
*   **现有挑战:** 
    1.  **现有 RL 方法不稳定:** 传统的策略梯度方法（如 `DDPO`）在处理超过 100 条提示词的大规模数据时，训练非常容易崩溃（不收敛）。
    2.  **计算资源消耗大:** 某些方法（如 `ReFL`）需要可微分的奖励模型，这在生成视频时会消耗惊人的显存。
    3.  **模型兼容性差:** 很多方法只针对早期的扩散模型，不直接适用于现在主流的整流流 (Rectified Flows) 模型（如 FLUX）。
*   **创新思路:** 借鉴大语言模型（LLM）领域 DeepSeek-R1 成功的经验，将 <strong>组相对策略优化 (GRPO)</strong> 引入视觉生成。GRPO 的核心是不需要额外的评论者模型 (Critic Model)，而是通过同一组样本之间的相对表现来计算优势，从而极大提高了训练稳定性。

## 2.2. 核心贡献/主要发现
*   **稳定性先锋:** 首次发现并验证了 GRPO 的稳定性机制能有效解决视觉生成 RL 训练中的崩溃问题。
*   **统一框架:** 提出了一个通用的框架，能够无缝适配扩散模型和整流流模型，并支持文生图 (T2I)、文生视频 (T2V) 和图生视频 (I2V) 三大任务。
*   **性能飞跃:** 在多个基准测试（如 HPS-v2.1, CLIP, VideoAlign）中大幅超越基线模型，尤其在视频运动质量上取得了 181% 的提升。
*   **新见解:** 揭示了共享初始化噪声、时间步选择（Timestep Selection）和多奖励模型融合对视觉 RLHF 的重要性。

    ---

# 3. 预备知识与相关工作

## 3.1. 基础概念
*   <strong>扩散模型 (Diffusion Model):</strong> 一种通过逐步向数据添加噪声再学习逆向去噪过程来生成图像的模型。
*   <strong>整流流 (Rectified Flow):</strong> 现代生成模型（如 FLUX）常用的一种数学框架，它将生成过程视为从噪声到图像的直线运动，通常比传统扩散模型更高效。
*   <strong>基于人类反馈的强化学习 (Reinforcement Learning from Human Feedback, RLHF):</strong> 利用人类的偏好数据训练一个奖励模型，再用强化学习算法优化生成模型，使其产出得分更高的内容。
*   <strong>策略 (Policy):</strong> 在这里指生成模型本身，即它决定了在每一个去噪步骤中如何从当前状态推演 (rollout) 到下一个状态。

## 3.2. 前人工作与技术演进
视觉生成的对齐技术经历了以下演进：
1.  <strong>DPO (Direct Preference Optimization) 类:</strong> 通过对比“好”和“坏”的样本直接微调，无需强化学习，但提升幅度有限。
2.  <strong>可微奖励方法 (如 ReFL):</strong> 直接对奖励模型的得分求梯度。缺点是必须要求奖励模型是可微的，且显存压力极大。
3.  <strong>策略梯度方法 (如 DDPO, DPOK):</strong> 经典的强化学习路径，将奖励视为“黑盒”。虽然理论完美，但实践中极不稳定，难以扩展到大规模数据集。

## 3.3. GRPO 的核心差异
<strong>组相对策略优化 (Group Relative Policy Optimization, GRPO)</strong> 最初由 DeepSeek 提出。与传统的 PPO 算法相比，它最大的区别在于：
*   **取消 Critic 模型:** PPO 需要一个额外的神经网络（Critic）来估算状态价值。在处理图像/视频时，这个 Critic 模型本身就很难训练且占空间。
*   **相对评估:** GRPO 针对同一个提示词生成一组样本（例如 16 张图），计算这组样本奖励的平均值和标准差，通过样本得分与组内平均分的相对差距来计算优势。这种方式天然具有归一化效果，使得训练异常稳定。

    ---

# 4. 方法论

DanceGRPO 的核心是将视觉生成过程形式化为一个强化学习问题，并利用 GRPO 算法进行优化。

## 4.1. 任务形式化：去噪即马尔可夫决策过程 (MDP)
作者将扩散模型或整流流模型的迭代去噪过程建模为一个马尔可夫决策过程 (MDP)：
*   <strong>状态 (State) $s_t$:</strong> 定义为 $(\mathbf{c}, t, \mathbf{z}_t)$，其中 $\mathbf{c}$ 是提示词，$t$ 是当前时间步，$\mathbf{z}_t$ 是当前的潜空间特征（带有噪声的图像）。
*   <strong>动作 (Action) $a_t$:</strong> 即生成的下一步特征 $\mathbf{z}_{t-1}$。
*   <strong>策略 (Policy) $\pi$:</strong> 模型根据当前状态生成下一步的概率分布。
*   <strong>奖励 (Reward) $R$:</strong> 仅在去噪完成后的最终图像 $\mathbf{z}_0$ 上计算，公式如下：
    $$R(\mathbf{s}_t, \mathbf{a}_t) = \begin{cases} r(\mathbf{z}_0, \mathbf{c}), & \text{if } t = 0 \\ 0, & \text{otherwise} \end{cases}$$
    这里 $r(\mathbf{z}_0, \mathbf{c})$ 是奖励模型（如美学评分模型）给出的得分。

## 4.2. 采样随机化：从 ODE 到 SDE
强化学习需要“探索”（即尝试不同的路径）。整流流模型通常使用确定性的常微分方程 (ODE) 采样，这不符合强化学习的要求。因此，作者引入了<strong>随机微分方程 (Stochastic Differential Equations, SDE)</strong> 来增加采样的随机性。

对于整流流模型，逆向生成过程的 SDE 公式为：
$$\mathrm{d} \mathbf{z}_t = \left( \mathbf{u}_t - \frac{1}{2} \varepsilon_t^2 \nabla \log p_t (\mathbf{z}_t) \right) \mathrm{d} t + \varepsilon_t \mathrm{d} \mathbf{w}$$
*   $\mathbf{u}_t$: 模型预测的“速度”或向量场。
*   $\varepsilon_t$: 引入的随机噪声强度，决定了探索的程度。
*   $\mathrm{d} \mathbf{w}$: 布朗运动噪声。
*   $\nabla \log p_t (\mathbf{z}_t)$: 分数函数 (Score function)，引导模型回到高概率密度区域。

## 4.3. GRPO 核心算法流程
DanceGRPO 的训练分为以下几个关键步骤：

1.  **分组采样:** 针对一个提示词 $\mathbf{c}$，使用当前的旧策略 $\pi_{\theta_{old}}$ 推演 (rollout) 出一组样本 $\{ \mathbf{o}_1, \mathbf{o}_2, \dots, \mathbf{o}_G \}$。
2.  **计算奖励:** 使用一个或多个奖励模型为这些样本打分，得到 $\{ r_1, r_2, \dots, r_G \}$。
3.  <strong>计算相对优势 (Advantage):</strong> 
    $$A_i = \frac{r_i - \operatorname{mean}(\{r_1, \dots, r_G\})}{\operatorname{std}(\{r_1, \dots, r_G\})}$$
    这一步非常关键，它通过组内对比消除了不同提示词难度不同的影响。
4.  **目标函数优化:** 最大化以下 GRPO 目标函数 $\mathcal{I}(\theta)$：
    $$\mathcal{I}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{T} \sum_{t=1}^T \min \left( \rho_{t,i} A_i, \mathrm{clip}(\rho_{t,i}, 1-\epsilon, 1+\epsilon) A_i \right) \right]$$
    *   $\rho_{t,i} = \frac{\pi_\theta(a_{t,i}|s_{t,i})}{\pi_{\theta_{old}}(a_{t,i}|s_{t,i})}$: 这是新旧策略的概率比率。
    *   $\mathrm{clip}$: 剪切函数，防止步子迈得太大导致策略剧烈波动。
    *   $T$: 时间步总数。

## 4.4. 核心训练技巧
*   <strong>共享初始化噪声 (Shared Initialization Noise):</strong> 作者发现，如果同一组样本使用完全不同的初始噪声，会导致训练不稳定（即所谓的“奖励劫持 Reward Hacking”）。因此，DanceGRPO 让同一组样本在训练时共享相同的初始噪声。
*   <strong>时间步选择 (Timestep Selection):</strong> 并不需要更新所有的去噪步。实验发现，更新前 30% 或 60% 的时间步对性能贡献最大。
*   **多奖励聚合:** 通过聚合多个优势函数（如美学 + 图文对齐）来平衡视觉质量和指令遵循能力。

    ---

# 5. 实验设置

## 5.1. 数据集
*   <strong>文生图 (T2I):</strong> 使用超过 10,000 条精心挑选的提示词进行优化。
*   <strong>文生视频 (T2V):</strong> 使用 `VidProM` 数据集中的提示词，分辨率设定为 480x480。
*   <strong>图生视频 (I2V):</strong> 使用 `ConsisID` 构造的提示词数据集，并配合 FLUX 生成的参考图。

## 5.2. 评估指标
论文使用了五个核心指标来衡量模型表现：
1.  **HPS-v2.1 (Human Preference Score):** 衡量图像是否符合人类美学偏好。
2.  **CLIP Score:** 衡量生成的图像/视频与文本提示词的匹配程度。
    $$\text{CLIP Score} = \cos(\mathbf{v}_{img}, \mathbf{v}_{text})$$
    其中 $\mathbf{v}$ 是经过 CLIP 编码器提取的特征向量。
3.  **VideoAlign (VQ & MQ):** 专门针对视频的指标，包括视觉质量 (VQ) 和运动质量 (MQ)。
4.  **Pick-a-Pic:** 模拟人类在两张图中选一张更好看的概率。
5.  **GenEval:** 一个综合评估模型对复杂物体组合、属性和空间关系理解能力的基准。

## 5.3. 对比基线
*   **DDPO / DPOK:** 传统的强化学习策略梯度方法。
*   **ReFL:** 基于可微奖励的直接反向传播方法。
*   **DPO / OnlineVPO:** 基于偏好对比的学习方法。

    ---

# 6. 实验结果与分析

## 6.1. 核心结果分析
DanceGRPO 在所有任务上都表现出了碾压级的性能。

以下是原文 **Table 2**（Stable Diffusion 1.4）的结果：

<table>
<thead>
<tr>
<th>模型 (Models)</th>
<th>HPS-v2.1 ↑</th>
<th>CLIP Score ↑</th>
<th>Pick-a-Pic ↑</th>
<th>GenEval ↑</th>
</tr>
</thead>
<tbody>
<tr>
<td>Stable Diffusion (原始版)</td>
<td>0.239</td>
<td>0.363</td>
<td>0.202</td>
<td>0.421</td>
</tr>
<tr>
<td>Stable Diffusion + DanceGRPO (HPS)</td>
<td><strong>0.365</strong></td>
<td>0.380</td>
<td><strong>0.217</strong></td>
<td>0.521</td>
</tr>
<tr>
<td>Stable Diffusion + DanceGRPO (HPS & CLIP)</td>
<td>0.335</td>
<td><strong>0.395</strong></td>
<td>0.215</td>
<td><strong>0.522</strong></td>
</tr>
</tbody>
</table>

**分析:** 当只用 HPS（美学）优化时，美学分最高；当同时用 HPS 和 CLIP 优化时，模型在保持高美学的同时，指令遵循能力（GenEval）进一步提升。

以下是原文 **Table 5**（HunyuanVideo 视频生成）的结果：

<table>
<thead>
<tr>
<th>基准 (Benchmarks)</th>
<th>视觉质量 (VQ) ↑</th>
<th>运动质量 (MQ) ↑</th>
<th>文本对齐 (TA)</th>
</tr>
</thead>
<tbody>
<tr>
<td>基线 (Baseline)</td>
<td>4.51</td>
<td>1.37</td>
<td>1.75</td>
</tr>
<tr>
<td>DanceGRPO (Ours)</td>
<td>7.03 (<strong>+56%</strong>)</td>
<td>3.85 (<strong>+181%</strong>)</td>
<td>1.59</td>
</tr>
</tbody>
</table>

**分析:** 运动质量 (MQ) 提升了惊人的 181%，这解决了视频生成中常见的“物体不动”或“动作僵硬”的问题。

## 6.2. 稳定性对比
下图（原文 Figure 5）展示了 DanceGRPO 与 DDPO 的稳定性对比：

![Figure 5 We visualize the results of DDPO and Ours. DDPO always diverges when applied to rectified fow SDEs](images/5.jpg)
*该图像是图表，展示了DDPO和我们的方法在训练迭代过程中的奖励变化。可以看到，我们的方法在多个迭代中保持了更高的奖励，而DDPO则表现出明显的波动和收敛困难。*

从中可以看出，在整流流模型上，DDPO 的奖励曲线在经过一段时间后会发生剧烈震荡甚至发散，而 DanceGRPO 则稳步上升并保持高位。

## 6.3. 消融实验
*   **时间步的影响:** 如下图（原文 Figure 4b）所示，仅仅训练前 30% 的时间步就能获得大部分收益，这极大地节省了训练开销。
*   **噪声水平 $\varepsilon_t$:** 噪声太小会导致探索不足，性能受限；噪声太大（如 > 0.3）会导致生成的图像背景带有噪点。

    ![该图像是一个示意图，展示了三组不同实验条件下的奖励变化，包含(a) Best-of-N推理缩放的奖励，(b) 时间步长分数的消融研究，(c) 不同噪声水平的消融研究。每组数据在训练迭代中的奖励表现不同，展示了模型优化过程中的稳定性与变化趋势。](images/4.jpg)
    *该图像是一个示意图，展示了三组不同实验条件下的奖励变化，包含(a) Best-of-N推理缩放的奖励，(b) 时间步长分数的消融研究，(c) 不同噪声水平的消融研究。每组数据在训练迭代中的奖励表现不同，展示了模型优化过程中的稳定性与变化趋势。*

---

# 7. 总结与思考

## 7.1. 结论总结
DanceGRPO 成功地将大语言模型中的 GRPO 算法迁移到了视觉生成领域。它通过重新构建采样 SDE、采用组相对评估机制，解决了视觉 RLHF 训练中长期存在的**不稳定性**和**难以扩展**的痛点。该框架不仅在图像生成上效果显著，更是在极具挑战性的视频生成任务中证明了其强大的对齐能力。

## 7.2. 局限性与未来工作
*   **奖励模型依赖:** 强化学习的效果高度依赖于奖励模型。如果奖励模型本身有偏差（例如偏好过于饱和的颜色），生成模型也会产生“审美疲劳”或生成“油腻”的图片（见 Figure 9）。
*   **未来方向:** 探索更强大的多模态大模型（MLLM）作为奖励器；将 GRPO 扩展到更通用的多模态统一模型优化中。

## 7.3. 个人启发与批判
*   **启发:** 这篇论文再次证明了“算法架构的稳定性大于一切”。GRPO 这种“取消 Critic、组内相对打分”的思想在视觉领域同样是降维打击。
*   **批判:** 论文中提到的“共享初始化噪声”虽然增加了稳定性，但也可能限制了样本的多样性。此外，对于视频生成这种极高维度的空间，如何更精细地定义“运动质量”的奖励，仍是未来需要突破的关键。总的来说，这是视觉生成对齐领域里程碑式的工作。