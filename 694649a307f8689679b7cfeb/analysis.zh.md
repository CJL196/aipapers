# 1. 论文基本信息

## 1.1. 标题
**Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation** (Perceiver-Actor: 一种用于机器人操控的多任务 Transformer)

## 1.2. 作者
<strong>Mohit Shridhar (华盛顿大学), Lucas Manuelli (NVIDIA), Dieter Fox (华盛顿大学 &amp; NVIDIA)</strong>。这些作者在计算机视觉、语言接地以及机器人操控领域拥有深厚的研究背景。

## 1.3. 发表期刊/会议
发表于 **CoRL 2022 (Conference on Robot Learning)**。CoRL 是机器人学习领域的顶级学术会议，以其严格的评审和对算法创新与实际机器人应用结合的重视而闻名。

## 1.4. 发表年份
**2022年** (2022-09-12 提交至 arXiv)。

## 1.5. 摘要
Transformer 在视觉和自然语言处理领域取得了巨大成功，但在机器人操控中，数据往往稀缺且昂贵。本文提出了 **PerAct (Perceiver-Actor)**，这是一种语言条件的<strong>行为克隆 (Behavior Cloning)</strong> 智能体，用于多任务的 <strong>六自由度 (6-DoF)</strong> 操控。PerAct 的核心创新在于将 RGB-D 观测转化为<strong>体素 (Voxel)</strong> 观测，并使用 **Perceiver Transformer** 来处理高维的体素数据。通过将操控任务建模为“检测下一个最佳体素动作”，PerAct 在仅需少量演示的情况下，就能在 18 个模拟任务和 7 个现实任务中展现出极强的性能，显著优于传统的 2D 图像到动作模型和 3D 卷积神经网络。

## 1.6. 原文链接
- **arXiv 链接:** [https://arxiv.org/abs/2209.05451](https://arxiv.org/abs/2209.05451)
- **PDF 链接:** [https://arxiv.org/pdf/2209.05451v2.pdf](https://arxiv.org/pdf/2209.05451v2.pdf)
- **发布状态:** 已正式发表于 CoRL 2022。

  ---

# 2. 整体概括

## 2.1. 研究背景与动机
*   **核心问题:** 如何让 Transformer 模型在机器人操控这种“数据极度匮乏”的领域发挥作用？
*   **挑战:** Transformer 通常需要海量数据进行训练（如 GPT 系列）。在机器人领域，获取人类演示（Demonstrations）成本极高。现有的端到端方法要么依赖 2D 图像（缺乏 3D 结构先验），要么依赖 3D 卷积（感受野受限且难以扩展到多任务）。
*   <strong>Gap (研究空白):</strong> 缺乏一种既能利用 3D 结构空间先验（3D Structural Prior），又能利用 Transformer 全球感受野（Global Receptive Field）和通用建模能力的端到端多任务框架。
*   **创新思路:** 将 3D 观测体素化，并借鉴 **ViT (Vision Transformer)** 的思想，将体素网格切分为 3D 补丁（Patches）。通过 **Perceiver** 架构解决高维输入带来的计算瓶颈，将机器人动作预测转化为一个“体素分类”任务。

## 2.2. 核心贡献/主要发现
*   **提出 PerAct 架构:** 一个统一的、语言条件的 Transformer 智能体，能够同时学习多种不同的机器人技能。
*   **3D 体素动作空间:** 证明了将 3D 环境表示为体素补丁，比直接处理 2D 像素或非结构化点云更能有效地学习六自由度动作。
*   **高效的数据利用率:** 仅通过每项任务几组到几十组的演示，模型就能学习到复杂的长程任务（如叠方块、清理台面）。
*   **实验结果:** 在 18 个 RLBench 任务（包含 249 种变体）中，PerAct 的表现是 2D 图像基准的 34 倍，是 3D 卷积基准的 2.8 倍。

    ---

# 3. 预备知识与相关工作

## 3.1. 基础概念
*   <strong>六自由度 (6-DoF):</strong> 指物体在三维空间中可以进行的六种独立运动，包括三个位置坐标（x, y, z）和三个旋转角度（俯仰 Pitch, 偏航 Yaw, 翻滚 Roll）。
*   <strong>体素 (Voxel):</strong> “体积元素”的简称，是三维空间中的“像素”。本文将机器人的工作空间划分为 $100 \times 100 \times 100$ 的立方体网格。
*   <strong>行为克隆 (Behavior Cloning, BC):</strong> 模仿学习的一种，智能体通过观察专家的演示（输入观测，输出动作），学习一个从观测到动作的直接映射函数。
*   <strong>语言接地 (Language Grounding):</strong> 将自然语言指令（如“打开中间的抽屉”）映射到具体的视觉特征和机器人动作上的过程。

## 3.2. 前人工作
*   **Transformer 与 Attention:** Transformer 依赖于<strong>自注意力 (Self-Attention)</strong> 机制。其计算公式如下（为了帮助初学者，我们复述这一核心背景）：
    $$ \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
    其中 $Q$ (Query), $K$ (Key), $V$ (Value) 分别代表查询、键和值。$d_k$ 是缩放因子。Transformer 的优势在于能捕捉序列中任意两个位置之间的全局依赖关系。
*   **C2FARM:** 之前最先进的 3D 操控方法，使用 3D U-Net 和粗到细（Coarse-to-fine）的策略。其局限性在于卷积核的局部性（Locality），难以理解场景中相距较远的物体之间的逻辑关系。

## 3.3. 技术演进
机器人操控经历了从“手工特征 + 运动学逆解”到“深度学习 + 2D 图像”，再到现在的“Transformer + 3D 结构化表示”的演进。PerAct 处在这一脉络的最前沿，即通过统一的架构处理多模态（语言 + 视觉）和多任务。

## 3.4. 差异化分析
相比于 **Gato (DeepMind)** 等模型，PerAct 不仅仅是将动作离散化为词元（Tokens），而是利用了 **3D 体素补丁** 这一结构化先验，使得模型对物体的空间位置极其敏感，从而极大提升了数据效率。

---

# 4. 方法论

## 4.1. 方法原理
PerAct 的核心直觉是：如果机器人能准确地“看”出空间中哪一个体素是当前动作的最佳操作点（例如抓取物体的中心），那么六自由度操控问题就变成了一个视觉识别问题。

## 4.2. 核心方法详解 (逐层深入)

下图（原文 Figure 2）展示了 PerAct 的整体工作流程：

![该图像是一个示意图，展示了Perceiver-Actor的结构，其中包含了语言编码器、体素编码器和体素解码器的关系。图中描述了如何通过$Q_{trans}$获得下一个最佳体素动作，体现了六自由度操作的多任务处理能力。](images/2.jpg)
*该图像是一个示意图，展示了Perceiver-Actor的结构，其中包含了语言编码器、体素编码器和体素解码器的关系。图中描述了如何通过$Q_{trans}$获得下一个最佳体素动作，体现了六自由度操作的多任务处理能力。*

### 4.2.1. 观测空间：体素化处理
首先，智能体从四个相机获取 RGB-D（彩色+深度）图像。利用相机的内参和外参，将像素投射到三维空间，构建出一个 $100 \times 100 \times 100$ 的体素网格 $\mathbf{v}$。
每个体素包含 10 个通道的信息：RGB（彩色）、3D 坐标、占用状态（Occupancy）以及位置索引。

### 4.2.2. 语言目标编码
自然语言指令 $\mathbf{l}$ 通过预训练的 **CLIP 语言编码器** 转化为向量序列。
$$ \mathbf{f}_{lang} = \mathrm{CLIPEncoder}(\mathbf{l}) $$
这使得模型能够理解“红色”、“中间”、“方块”等语义概念。

### 4.2.3. 体素补丁与 Perceiver Transformer
为了将巨大的体素网格输入 Transformer，PerAct 将其切分为 $5 \times 5 \times 5$ 的补丁（Patches），类似于 ViT 处理图像的方式。这样 $100^3$ 的体素就变成了 $20^3 = 8000$ 个词元。

由于 8000 个词元的自注意力计算量巨大（$O(N^2)$），PerAct 采用了 **Perceiver Transformer**（原文 Figure 6）：

![Figure 6. Perceiver Transformer Architecture. Perceiver is a latent-space transformer. Q, K, V represent queries, keys, and values, respectively. We use 6 selfattention layers in our implementation.](images/6.jpg)
*该图像是一个示意图，展示了PerceiverIO Transformer的架构。输入部分包含了K和V，Latents部分则含有Q。图中分别展示了cross attention和self attention的模块，说明了其在数据处理中的作用。*

1.  <strong>交叉注意力 (Cross-Attention):</strong> 模型首先初始化一小组学习到的“潜变量 (Latent vectors)”（例如 2048 个）。让这些潜变量去“查询”输入的 8000 个体素词元。
2.  <strong>潜空间自注意力 (Latent Self-Attention):</strong> 在较小的潜空间内进行 6 层自注意力计算，提取全局特征。
3.  **上采样:** 最后再通过交叉注意力将特征映射回原始的体素维度。

### 4.2.4. 动作预测与目标函数
PerAct 将动作解码为四个分量：平移（Translation）、旋转（Rotation）、夹具状态（Gripper open/close）和碰撞避免（Collide）。

模型为每个体素预测一个得分 $Q$，选择得分最高的体素作为动作执行点：
$$ \mathcal{T}_{\mathrm{trans}} = \underset{(x,y,z)}{\mathrm{argmax}} \ Q_{\mathrm{trans}}((x,y,z) \mid \mathbf{v}, \mathbf{l}) $$
$$ \mathcal{T}_{\mathrm{rot}} = \underset{(\psi, \theta, \phi)}{\mathrm{argmax}} \ Q_{\mathrm{rot}}((\psi, \theta, \phi) \mid \mathbf{v}, \mathbf{l}) $$

训练时，使用<strong>交叉熵损失 (Cross-Entropy Loss)</strong>，将该问题视为分类任务：
$$ \mathcal{L}_{\mathrm{total}} = - \mathbb{E}_{Y_{\mathrm{trans}}} [\log \mathcal{V}_{\mathrm{trans}}] - \mathbb{E}_{Y_{\mathrm{rot}}} [\log \mathcal{V}_{\mathrm{rot}}] - \mathbb{E}_{Y_{\mathrm{open}}} [\log \mathcal{V}_{\mathrm{open}}] - \mathbb{E}_{Y_{\mathrm{collide}}} [\log \mathcal{V}_{\mathrm{collide}}] $$
这里的 $\mathcal{V}$ 是对预测得分 $Q$ 进行 `softmax` 后的概率分布。这种方法通过“拉高”专家演示中动作点的概率，同时“压低”其他位置的概率来学习。

---

# 5. 实验设置

## 5.1. 数据集
*   **RLBench:** 一个大规模机器人学习基准环境。
*   **任务规模:** 18 个任务，如 `open drawer` (开抽屉), `stack blocks` (叠方块), `sweep to dustpan` (扫入簸箕) 等。
*   **变体:** 共有 249 种变体（涉及不同颜色、大小、数量和位置）。
*   **样本示例:** "stack 2 red blocks"（堆叠两个红色方块）。

## 5.2. 评估指标
1.  <strong>成功率 (Success Rate):</strong>
    *   **概念定义:** 该指标衡量智能体在给定任务中完成预定目标的比例。任务必须完全成功才记为 100，否则为 0。
    *   **数学公式:**
        $$ \text{Success Rate} = \frac{N_{success}}{N_{total}} \times 100\% $$
    *   **符号解释:** $N_{success}$ 是成功推演的次数，$N_{total}$ 是总评估次数。

## 5.3. 对比基线
*   **Image-BC:** 类似于 BC-Z 的 2D 图像到动作模型。
*   **C2FARM-BC:** 使用 3D 卷积网络的 SOTA 方法，但缺乏 Transformer 的全局建模能力。

    ---

# 6. 实验结果与分析

## 6.1. 核心结果分析
PerAct 在多任务学习上表现出了压倒性的优势。特别是在需要理解全局场景的任务（如从三个抽屉中选出指定的一个）中，PerAct 远超局部感受野的 C2FARM。

以下是原文 **Table 1** 的完整实验结果：

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="2">Open Drawer</th>
<th colspan="2">Slide Block</th>
<th colspan="2">Sweep to Dustpan</th>
<th colspan="2">Meat off Grill</th>
<th colspan="2">Turn Tap</th>
<th colspan="2">Stack Blocks</th>
</tr>
<tr>
<th>10 demos</th>
<th>100 demos</th>
<th>10 demos</th>
<th>100 demos</th>
<th>10 demos</th>
<th>100 demos</th>
<th>10 demos</th>
<th>100 demos</th>
<th>10 demos</th>
<th>100 demos</th>
<th>10 demos</th>
<th>100 demos</th>
</tr>
</thead>
<tbody>
<tr>
<td>Image-BC (CNN)</td>
<td>4</td>
<td>4</td>
<td>4</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>20</td>
<td>8</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<td>C2FARM-BC</td>
<td>28</td>
<td>20</td>
<td>12</td>
<td>16</td>
<td>0</td>
<td>0</td>
<td>4</td>
<td>12</td>
<td>68</td>
<td>12</td>
<td>4</td>
<td>0</td>
</tr>
<tr>
<td><strong>PerAct (Ours)</strong></td>
<td><strong>68</strong></td>
<td><strong>80</strong></td>
<td><strong>32</strong></td>
<td><strong>72</strong></td>
<td><strong>72</strong></td>
<td><strong>56</strong></td>
<td><strong>68</strong></td>
<td><strong>84</strong></td>
<td><strong>72</strong></td>
<td><strong>80</strong></td>
<td><strong>12</strong></td>
<td><strong>36</strong></td>
</tr>
</tbody>
</table>

**分析:** 我们可以看到，在 `100 demos` 的情况下，PerAct 在几乎所有任务上都大幅领先。

## 6.2. 全局 vs. 局部感受野
下图（原文 Figure 4）通过“开抽屉”任务验证了全局感受野的重要性。

![Figure 4. Global vs. Local Receptive Field Experiments. Success rates of PERACT against various C2FARM-BC \[14\] baselines](images/4.jpg)
*该图像是图表，展示了PerAct与多种C2FARM-BC基准在不同训练步骤下的成功率对比。随着训练步骤的增加，PerAct的成功率显著高于其他基准，显示出其在多任务操作中更优的学习效果。*

当任务指令是“打开中间抽屉”时，局部感受野模型（C2FARM）往往会混淆长相相似的三个抽屉把手，而 PerAct 能通过 Transformer 观察整个柜子，从而精准定位。

## 6.3. 现实世界机器人实验
作者在真实的 Franka Panda 机器人上进行了验证。仅用 53 个演示就训练出了一个能完成 7 种任务的多任务智能体，成功率在简单任务上高达 90%。

![Figure 8. Real-Robot Setup with Kinect-2 and Franka Panda.](images/8.jpg)
*该图像是机器人操作的实际设置，展示了Franka Emika Panda机械臂、Kinect 2 RGB-D相机与用于手眼协调的AR标记。桌面上有多个不同颜色的立方体，作为操作对象，使观察者能直观了解机器人如何进行操控任务。*

---

# 7. 总结与思考

## 7.1. 结论总结
PerAct 成功地证明了：<strong>只要有正确的表示方法（3D 体素）和合适的架构（Perceiver Transformer），Transformer 可以在数据稀缺的机器人领域大放异彩</strong>。它将多任务操控统一在了一个简单的分类框架下。

## 7.2. 局限性与未来工作
*   **运动规划的依赖:** PerAct 预测的是关键位置，然后依赖运动规划器（Motion Planner）去执行。这使得它难以处理需要实时连续调整的动态任务（如接球）。
*   **高精度任务:** 在 `insert peg` (插木栓) 等极高精度的任务中表现欠佳，这受限于体素的分辨率。
*   **未来方向:** 结合预训练的视觉大模型特征（如 R3M）来进一步提升泛化能力；探索不依赖运动规划器的端到端连续速度控制。

## 7.3. 个人启发与批判
*   **启发:** 3D 结构化先验是机器人学习的“捷径”。直接从像素学习 3D 空间关系太难，而体素化为模型提供了一个非常好的起点。
*   **批判:** 尽管 Perceiver 减少了计算量，但 $100^3$ 的体素化过程和数据增强依然非常耗时（论文提到在 8 张 V100 上训练了 16 天）。这种高昂的计算成本对于普通研究者来说是一个门槛。此外，该模型目前是离线（Offline）训练的，如何进行在线纠错（Online correction）也是一个值得探讨的问题。