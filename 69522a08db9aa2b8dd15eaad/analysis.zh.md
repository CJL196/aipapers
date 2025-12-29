# 1. 论文基本信息

## 1.1. 标题
**VideoSSM: Autoregressive Long Video Generation with Hybrid State-Space Memory**
（VideoSSM：基于混合状态空间记忆的自回归长视频生成）

## 1.2. 作者
Yifei Vu, Xiaoshan Wu, Xinting Hu, Tao Hu, Yang-Tian Sun, Xiaoyang Lyu, Bo Wang, Lin Ma, Yuewen Ma, Zhongrui Wang, Xiaojuan Qi。
作者隶属于香港大学（HKU）、字节跳动（ByteDance PICO）和南方科技大学（SUSTech）。

## 1.3. 发表期刊/会议
该论文目前发布于预印本平台 **arXiv**，发布日期为 2025 年 12 月 4 日。

## 1.4. 摘要
自回归 (Autoregressive, AR) 扩散模型通过因果式地生成视频帧，实现了流式、交互式的长视频生成。然而，在分钟级的时间跨度内保持连贯性仍具挑战，主要受限于误差累积、运动漂移和内容重复。本文提出了 **VideoSSM**，一种将自回归扩散与混合状态空间记忆相结合的长视频模型。该模型包含两个核心记忆组件：<strong>状态空间模型 (State-Space Model, SSM)</strong> 作为演进的全局记忆，捕捉整个序列的场景动态；<strong>上下文窗口 (Context Window)</strong> 作为局部记忆，捕捉运动线索和微观细节。这种设计在不产生僵化、重复模式的前提下保持了全局一致性，支持提示词自适应交互，并随序列长度呈线性时间扩展。实验表明，VideoSSM 在分钟级长视频生成中达到了最先进的 (state-of-the-art) 时间一致性和运动稳定性。

## 1.5. 原文链接
*   **PDF 链接:** [https://arxiv.org/pdf/2512.04519v1.pdf](https://arxiv.org/pdf/2512.04519v1.pdf)
*   **arXiv 页面:** [https://arxiv.org/abs/2512.04519](https://arxiv.org/abs/2512.04519)

    ---

# 2. 整体概括

## 2.1. 研究背景与动机
**研究背景：**
长视频生成是生成式视觉智能的长期目标，旨在模拟具有持久身份和时间连贯性的视觉世界。目前的扩散转换器 (Diffusion Transformer, DiT) 架构主要受限于短时间上下文和全注意力机制 (Full Attention) 的二次方计算成本，难以扩展到超长序列。

**解决的核心问题：**
自回归 (AR) 扩散模型虽然支持流式生成，但在处理分钟级视频时面临三大痛点：
1.  <strong>误差累积 (Error Accumulation):</strong> 每一帧的微小误差会随时间放大。
2.  <strong>运动漂移 (Motion Drift):</strong> 场景内容逐渐偏离初始设定。
3.  <strong>内容重复 (Content Repetition):</strong> 为了稳定生成，现有方法往往固定早期帧作为锚点（注意力沉没），但这会导致场景陷入死循环。

**创新思路：**
受人类记忆启发，作者认为长视频生成需要**动态更新的记忆**。作者提出了一种混合记忆架构，将精确的局部细节（通过滑动窗口）与压缩的全局概括（通过 SSM）相结合，既避免了内容的僵化，又保证了长期的稳定性。

## 2.2. 核心贡献/主要发现
1.  **提出 VideoSSM 架构:** 首次将 SSM 作为动态全局记忆集成到自回归扩散模型中，解决了长视频生成中的“内容冻结”和“记忆丢失”矛盾。
2.  <strong>混合记忆机制 (Hybrid Memory):</strong> 设计了局部滑动窗口与全局状态空间的并行路径，并通过位置感知的路由机制进行融合。
3.  **高效的蒸馏策略:** 提出了一种两阶段训练方案，将高质量的双向 (Bidirectional) 模型知识迁移到因果 (Causal) 模型中，并在长程推演中增强其自我纠错能力。
4.  **卓越的性能:** 在长视频基准测试中，VideoSSM 在保持高一致性的同时，展现了远高于竞品的“动态程度 (Dynamic Degree)”，有效缓解了画面静止或重复的问题。

    ---

# 3. 预备知识与相关工作

## 3.1. 基础概念
*   <strong>自回归生成 (Autoregressive Generation, AR):</strong> 一种序列生成方式，模型根据已生成的历史内容（如前 $N$ 帧）来预测下一部分内容（如下一帧）。
*   <strong>扩散模型 (Diffusion Models):</strong> 通过学习将噪声逐渐转化为清晰图像/视频的过程来生成数据。
*   <strong>转换器 (Transformer):</strong> 一种基于注意力机制的神经网络架构。在视频领域，它将视频划分为小的块（词元 `token`），并计算它们之间的关联。
*   <strong>状态空间模型 (State-Space Model, SSM):</strong> 一种处理序列数据的数学模型（如 Mamba 架构）。它能以循环的方式处理信息，其内存开销不随序列长度增加而爆炸，适合建模超长依赖。
*   <strong>键值缓存 (KV Cache):</strong> 在 Transformer 推理中，为了避免重复计算已处理词元的注意力，将其计算结果存储起来的技术。

## 3.2. 前人工作与技术演进
视频生成技术从早期的循环网络发展到现在的扩散模型。
*   **DiT 架构:** 使用全注意力机制，计算量随帧数 $T$ 呈 $O(T^2)$ 增长。
*   <strong>自回归扩散 (AR DiT):</strong> 为了处理无限长度，模型转为因果模式（Causal），只关注过去。
*   <strong>注意力沉没 (Attention Sink):</strong> 发现保留最开始的几帧 `token` 可以稳定长序列生成。
*   **局限性:** 现有长视频方法（如 `LongLive`）往往使用静态的沉没词元（Fixed Sinks），这虽然稳定，但让模型倾向于反复生成与开头相似的内容，导致运动停滞。

## 3.3. 差异化分析
相比于 `Self-Forcing` 或 `LongLive`：
*   `LongLive` 使用固定的早期帧作为“全局记忆”，这是一种**静态记忆**。
*   **VideoSSM** 使用 SSM 实时压缩并更新所有被移出窗口的历史信息，这是一种**动态演进的记忆**。

    ---

# 4. 方法论

## 4.1. 方法原理
VideoSSM 将视频生成视为一个循环动力学过程。它使用<strong>局部记忆 (Local Memory)</strong> 记录最近几帧的精确像素级细节，使用<strong>全局记忆 (Global Memory)</strong> 记录场景的抽象演化趋势。

下图（原文 Figure 2）展示了从标准 DiT 到 VideoSSM 的演进：

![该图像是示意图，展示了三种不同的 DiT 块结构，包括标准 DiT 块、因果 DiT 块和带有记忆的因果 DiT 块。每种结构针对注意力机制的优化进行了说明，特别是其在因果性、流式处理和长序列效率方面的特点。](images/2.jpg)
*该图像是示意图，展示了三种不同的 DiT 块结构，包括标准 DiT 块、因果 DiT 块和带有记忆的因果 DiT 块。每种结构针对注意力机制的优化进行了说明，特别是其在因果性、流式处理和长序列效率方面的特点。*

## 4.2. 核心方法详解

### 4.2.1. 局部记忆：滑动窗口自注意力
为了保留精确的运动线索，模型维持一个长度为 $L$ 的滑动窗口。
对于当前帧 $t$ 的输入隐藏状态 $H_t^{in}$，通过线性映射计算查询 (Query, $Q$)、键 (Key, $K$) 和值 (Value, $V$)：
$$
\{ Q_t, K_t, V_t \} = \{ H_t^{in} W_Q, H_t^{in} W_K, H_t^{in} W_V \}
$$
其中 $W_Q, W_K, W_V$ 是可学习的权重矩阵。局部路径只计算窗口内的自注意力：
$$
H_t^{local} = \mathrm{SelfAttention}(Q_t, K_t^{local}, V_t^{local})
$$
这里 $K_t^{local}$ 包含沉没词元和最近的 $L$ 个词元。

### 4.2.2. 全局记忆：动态状态计算
当词元被移出局部窗口时，它们不再参与直接的注意力计算，而是被送入全局记忆模块进行压缩。该模块使用四部分协同工作：

<strong>1. 同步门控缓存 (Synchronized Gate Caching):</strong>
在词元退出窗口前，计算注入门 $\beta_t$（控制新信息进入量）和衰减门 $\alpha_t$（控制旧记忆遗忘率）：
$$
\begin{array} { r l } & { \beta _ { t } = \sigma ( \mathbf { W } _ { \beta } \mathbf { H } _ { t } ^ { \mathrm { i n } } ) } \\ & { \alpha _ { t } = - \exp ( \mathbf { A } ) \cdot \mathrm { SoftPlus } ( \mathbf { W } _ { \alpha } \mathbf { H } _ { t } ^ { \mathrm { i n } } + \mathbf { B } ) } \end{array}
$$
其中 $\sigma$ 是 `Sigmoid` 函数，$\mathbf{A}, \mathbf{W}_\alpha, \mathbf{W}_\beta$ 是可学习参数。

<strong>2. 状态更新 (State Update):</strong>
使用 **Gated $\Delta$-rule** 更新全局状态 $M_t$。该规则的核心是只存储“无法预测”的新信息：
$$
\begin{array} { r l r } & { } & { \mathbf { V } _ { \mathrm { n e w } , t } ^ { \mathrm { e v t } } = \mathbf { V } _ { t } ^ { \mathrm { e v t } } - \mathrm { Predict } ( \mathbf { M } _ { t - 1 } , \mathbf { K } _ { t } ^ { \mathrm { e v t } } , \beta _ { t } ^ { \mathrm { e v t } } ) , } \\ & { } & { \mathbf { M } _ { t } = \exp ( \bar { \mathbf { g } } _ { t } ) \cdot \mathbf { M } _ { t - 1 } + \mathbf { K } _ { t } ^ { \mathrm { e v t } } \cdot ( \mathbf { V } _ { \mathrm { n e w } , t } ^ { \mathrm { e v t } } ) ^ { T } , } \end{array}
$$
其中 $\bar{\mathbf{g}}_t = \sum \alpha_s^{evt}$ 是累积负门控，负责控制状态的稳定性。

<strong>3. 记忆检索 (Retrieval):</strong>
通过当前查询 $Q_t$ 从压缩状态 $M_t$ 中读取相关信息，并由输出门 $g_t^{out}$ 控制流量：
$$
\begin{array} { r } { \mathbf { g } _ { t } ^ { \mathrm { o u t } } = \mathrm { Linear } ( \mathbf { H } _ { t } ^ { \mathrm { i n } } ) , \quad \quad \quad } \\ { \mathbf { H } _ { t } ^ { \mathrm { g l o b a l } } = \mathrm { Swish } ( \mathbf { g } _ { t } ^ { \mathrm { o u t } } \odot \mathrm { RMSNorm } ( \mathbf { Q } _ { t } \mathbf { M } _ { t } ) ) , } \end{array}
$$

下图（原文 Figure 5）展示了混合记忆的具体架构：

![Figure 5. Architecture of the proposed hybrid memory module. The input $H _ { t } ^ { \\mathrm { i n } }$ is processed in two streams. The local path (top) uses windowed attention with a sliding KV cache to compute $H _ { t } ^ { \\mathrm { l o c a l } }$ To pSaMoSM) to recurrently compress historical information into a memory state $M$ , which is retrieved to produce $H _ { t } ^ { \\mathrm { g l o b a l } }$ . A router then dynamically fuses the local and global outputs.](images/5.jpg)
*该图像是示意图，展示了提出的混合记忆模块的架构。输入的隐藏状态 $H_t^{\mathrm{in}}$ 通过两个路径进行处理。局部路径使用窗口注意力和滑动KV缓存计算 $H_t^{\mathrm{local}}$，并通过 pSaMoSM 将历史信息压缩到记忆状态 $M_t$ 中，随后通过回忆产生 $H_t^{\mathrm{global}}$。路由器动态融合局部和全局输出。*

### 4.2.3. 位置感知门控融合
为了平衡局部和全局信息，作者设计了一个路由机制。在视频开始时，全局记忆尚未建立，应主要依赖局部窗口；随着视频变长，全局记忆的重要性逐渐提升。
定义相对位置比例 $\rho_t = (t+1)/T$，融合权重 $\gamma_t$ 计算如下：
$$
\gamma_t = \sigma(w_{router} \log(\rho_t) + b_{router})
$$
最终输出为：
$$
H_t^{fused} = H_t^{local} + \gamma_t \cdot H_t^{global}
$$

## 4.3. 训练策略
1.  **第一阶段：因果模型蒸馏。** 从预训练的双向模型（Wan 2.1）初始化，训练学生模型在 5 秒短片段上匹配老师的生成轨迹。
2.  **第二阶段：长视频训练。** 使用 <strong>长程自我推演 (Long Self-Rollout)</strong> 训练。模型在 60 秒的长序列上运行，仅使用自己生成的历史作为参考，并使用分布匹配蒸馏 (DMD) 损失函数进行纠错，使模型适应自回归过程中的误差累积。

    ---

# 5. 实验设置

## 5.1. 数据集
*   **训练数据:** 使用从 `VidProM` 数据集采样并经 LLM 扩充的提示词。
*   **评估基准:** **VBench**。这是一个综合性的视频生成评估框架，涵盖了从 5 秒短视频到分钟级长视频的多个维度。

## 5.2. 评估指标
1.  <strong>主体/背景一致性 (Subject/Background Consistency):</strong> 衡量视频中物体和环境在长时间跨度内是否保持不变。
    *   *注：通常使用特征余弦相似度计算。*
2.  <strong>动态程度 (Dynamic Degree):</strong> 衡量视频画面的运动丰富程度，防止模型生成“幻灯片”。
    $$
    \mathrm{DD} = \frac{1}{T-1} \sum_{t=1}^{T-1} \| \text{frame}_t - \text{frame}_{t-1} \|
    $$
3.  <strong>时间闪烁 (Temporal Flickering):</strong> 衡量帧间亮度或内容突变的程度。
4.  <strong>审美/成像质量 (Aesthetic/Imaging Quality):</strong> 衡量画面的艺术性和清晰度。

## 5.3. 对比基线
*   **基础 AR 模型:** `SkyReels-V2`, `MAGI-1`, `CausVid`。
*   **改进的 AR 模型:** `Self-Forcing`, `LongLive` (目前最先进的基线之一)。
*   <strong>双向模型 (参考):</strong> `Wan 2.1` (原始教师模型)。

    ---

# 6. 实验结果与分析

## 6.1. 核心结果分析
<strong>1. 短视频性能 (Table 1):</strong>
VideoSSM 在 5 秒片段的测试中，总分（83.95）和质量分（84.88）均位列 AR 模型第一，甚至逼近了参数量更大的非因果模型。

<strong>2. 长视频稳定性 (Table 2):</strong>
在 60 秒视频测试中，VideoSSM 的优势尤为明显：
*   **一致性:** 它的主体一致性（92.51）和背景一致性（93.95）均最高。
*   **动态性:** 关键点在于，VideoSSM 的 `Dynamic Degree` 达到了 **50.50**，远超 `LongLive` (37.50) 和 `Self-Forcing` (35.00)。这说明 VideoSSM 在保证不崩坏的同时，画面依然生动，没有陷入静态循环。

    以下是原文 Table 1 的完整对比数据：

    <table>
    <thead>
    <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">#Params</th>
    <th colspan="3">Evaluation scores ↑</th>
    </tr>
    <tr>
    <th>Total</th>
    <th>Quality</th>
    <th>Semantic</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td colspan="5" style="background-color: #f0f0f0;"><b>Bidirectional Diffusion models</b></td>
    </tr>
    <tr>
    <td>LTX-Video [20]</td>
    <td>1.9B</td>
    <td>80.00</td>
    <td>82.30</td>
    <td>70.79</td>
    </tr>
    <tr>
    <td>Wan2.1 [45]</td>
    <td>1.3B</td>
    <td>84.26</td>
    <td>85.30</td>
    <td>80.09</td>
    </tr>
    <tr>
    <td colspan="5" style="background-color: #f0f0f0;"><b>Autoregressive models</b></td>
    </tr>
    <tr>
    <td>SkyReels-V2 [7]</td>
    <td>1.3B</td>
    <td>82.67</td>
    <td>84.70</td>
    <td>74.53</td>
    </tr>
    <tr>
    <td>MAGI-1 [42]</td>
    <td>4.5B</td>
    <td>79.18</td>
    <td>82.04</td>
    <td>67.74</td>
    </tr>
    <tr>
    <td>CausVid [60]</td>
    <td>1.3B</td>
    <td>81.20</td>
    <td>84.05</td>
    <td>69.80</td>
    </tr>
    <tr>
    <td>Self Forcing [25]</td>
    <td>1.3B</td>
    <td>83.00</td>
    <td>83.71</td>
    <td>80.14</td>
    </tr>
    <tr>
    <td>LongLive [57]</td>
    <td>1.3B</td>
    <td>83.52</td>
    <td>84.26</td>
    <td>80.53</td>
    </tr>
    <tr>
    <td><b>VideoSSM (Ours)</b></td>
    <td><b>1.4B</b></td>
    <td><b>83.95</b></td>
    <td><b>84.88</b></td>
    <td><b>80.22</b></td>
    </tr>
    </tbody>
    </table>

## 6.2. 定性分析与用户研究
下图（原文 Figure 6）展示了在 60 秒生成中，VideoSSM（最右侧）能保持汉堡和潜水员的结构稳定，而 `SkyReels` 出现了内容崩溃，`Self-Forcing` 出现了漂移。

![该图像是示意图，展示了不同时间段（0-60秒）内，基于VideoSSM模型生成的视频帧对比，包含汉堡和水下场景的合成效果。每个模型的生成效果在各时间段中呈现差异，显示出VideoSSM在长视频生成中的优势。](images/6.jpg)

<strong>用户研究结果 (Table 3):</strong>
在 40 名参与者的排名中，VideoSSM 获得的“排名第一”选票比例最高 (41.07%)，平均排名也最靠前 (1.85)。

以下是原文 Table 3 的结果：

<table>
<thead>
<tr>
<th>Model</th>
<th>Rank 1 (%)</th>
<th>Rank 2 (%)</th>
<th>Rank 3 (%)</th>
<th>Rank 4 (%)</th>
<th>Avg Rank</th>
</tr>
</thead>
<tbody>
<tr>
<td>Self Forcing</td>
<td>11.79</td>
<td>13.21</td>
<td>23.21</td>
<td>51.79</td>
<td>3.18</td>
</tr>
<tr>
<td>CausVid</td>
<td>7.50</td>
<td>16.07</td>
<td>42.14</td>
<td>34.29</td>
<td>3.03</td>
</tr>
<tr>
<td>LongLive</td>
<td>39.64</td>
<td>36.43</td>
<td>15.00</td>
<td>8.93</td>
<td>1.92</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>41.07</b></td>
<td><b>34.29</b></td>
<td><b>19.64</b></td>
<td><b>5.00</b></td>
<td><b>1.85</b></td>
</tr>
</tbody>
</table>

---

# 7. 总结与思考

## 7.1. 结论总结
VideoSSM 通过创新的混合记忆架构，成功解决了长视频生成中的一致性与动态性之间的权衡难题。它利用 SSM 的线性扩展优势，实现了高效的全局信息沉淀，使得自回归扩散模型能够稳定、生动地生成分钟级乃至更长的视频内容，并支持实时的交互式提示词切换。

## 7.2. 局限性与未来工作
*   **局限性:** 虽然记忆是动态的，但在处理极其复杂的场景切换时，SSM 的压缩表示仍可能丢失某些极细微的空间细节。
*   **未来方向:**
    1.  集成显式的多模态调节（如音频同步）。
    2.  引入相机感知和几何先验知识，以支持更精确的 3D 一致性。
    3.  扩展到长格式的视频编辑任务。

## 7.3. 个人启发与批判
**启发:**
这篇论文非常巧妙地借鉴了认知心理学中的“工作记忆（短时）”和“长期记忆”概念。在深度学习中，全注意力通常被视为工作记忆，但它太贵了；固定的 Sink 是死板的索引。VideoSSM 证明了 <strong>“被遗忘的内容应该被有选择地压缩，而不是直接丢弃或永久冻结”</strong>。

**批判性思考:**
尽管 SSM 提供了线性复杂度，但 $Gated Δ-rule$ 的引入增加了计算步骤。在实际部署中，这种混合架构对推理算力的具体增加量（相比单纯的滑动窗口）还需要更详尽的延迟 (Latency) 测试报告。此外，模型对初始 5 秒质量的依赖极强，如果教师模型本身存在偏差，这种混合记忆可能会加速偏见的累积。