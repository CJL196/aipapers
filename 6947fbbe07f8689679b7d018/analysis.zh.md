# 1. 论文基本信息

## 1.1. 标题
<strong>V-JEPA 2: 自监督视频模型实现理解、预测与规划 (V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning)</strong>

## 1.2. 作者
**Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido 等人**。
作者团队主要来自 <strong>Meta 的 FAIR 实验室 (FAIR at Meta)</strong>，部分作者隶属于 <strong>Mila (魁北克人工智能研究所)</strong>。领衔作者团队包括 **Yann LeCun**（图灵奖得主，世界模型的倡导者）。

## 1.3. 发表期刊/会议
该论文发布于 **arXiv** 预印本平台，日期为 2025 年 6 月。鉴于其作者阵容和实验规模，该工作属于计算机视觉与机器人学习领域的顶尖研究，通常会提交至 CVPR、ICCV 或 NeurIPS 等顶级会议。

## 1.4. 发表年份
**2025 年**。

## 1.5. 摘要
本文探索了一种自监督学习方法，将互联网规模的视频数据与少量机器人交互数据（轨迹）相结合，开发出能够理解、预测和规划物理世界的模型。研究首先在超过 100 万小时的视频数据集上预训练了一个无动作的 <strong>联合嵌入预测架构 (Joint-Embedding-Predictive Architecture, JEPA)</strong>，即 **V-JEPA 2**。该模型在运动理解和人类动作预测（Action Anticipation）方面达到了最先进的性能。随后，通过将 V-JEPA 2 与大语言模型对齐，在视频问答（VidQA）任务中刷新了纪录。最后，通过在不足 62 小时的机器人数据上进行后训练，构建了动作条件化世界模型 **V-JEPA 2-AC**，实现了在未见过的实验室环境中对机器人手臂进行零起点（Zero-shot）的物体抓取与放置规划。

## 1.6. 原文链接
*   **原文链接:** [https://arxiv.org/abs/2506.09985](https://arxiv.org/abs/2506.09985)
*   **PDF 链接:** [https://arxiv.org/pdf/2506.09985v1.pdf](https://arxiv.org/pdf/2506.09985v1.pdf)
*   **发布状态:** 预印本 (Preprint)。

    ---

# 2. 整体概括

## 2.1. 研究背景与动机
现代人工智能的一个重大挑战是**像人类一样通过观察来学习理解世界并学会行动**。
*   **核心问题:** 现有的机器人控制模型通常依赖于大量的交互数据和明确的奖励反馈，这在现实世界中难以大规模获取。
*   **现有挑战:** 视频生成模型（Generative Models）虽然能生成逼真的视频，但在表示物理规律和进行计算高效的规划方面存在不足，因为它们花费了过多算力去预测无关紧要的像素细节。
*   **创新思路:** 本文遵循 Yann LeCun 提出的“世界模型”愿景，采用 <strong>联合嵌入预测架构 (JEPA)</strong>。JEPA 不在像素空间预测，而是在抽象的**特征表示空间**进行预测。这意味着模型可以忽略随风摆动的叶子等细节，而专注于物体的运动轨迹等关键物理特征。

## 2.2. 核心贡献/主要发现
1.  <strong>大规模预训练 (Scaling):</strong> 构建了包含 10 亿参数的 <strong>主干网络 (backbone)</strong>，并在 100 万小时视频上预训练，证明了特征空间掩码预测任务在视频领域具有极强的扩展性。
2.  <strong>V-JEPA 2-AC (动作条件化):</strong> 提出了一种两阶段训练方案，通过极少量的机器人数据（62 小时），使预训练的视频模型具备了根据机器人动作预测未来状态的能力。
3.  <strong>零起点规划 (Zero-shot Planning):</strong> 首次展示了基于 JEPA 的模型在完全陌生的环境中，无需任何特定任务训练或奖励函数，仅通过“想象”未来状态即可完成复杂的机械臂操作。
4.  **性能突破:** 在运动理解、视频问答和动作预测等多个基准测试中均达到 <strong>最先进的 (state-of-the-art)</strong> 水平。

    ---

# 3. 预备知识与相关工作

## 3.1. 基础概念
*   <strong>自监督学习 (Self-Supervised Learning, SSL):</strong> 一种机器学习方法，模型通过数据自身（如视频的下一帧、被遮挡的部分）提取标签进行学习，无需人工标注。
*   <strong>联合嵌入预测架构 (Joint-Embedding-Predictive Architecture, JEPA):</strong> 与传统的生成模型预测像素不同，JEPA 包含两个路径：一个处理完整数据，另一个处理缺失数据。模型的目标是在隐藏的特征空间里使两者的输出匹配。
*   <strong>掩码建模 (Masked Modeling):</strong> 将输入数据（如图像块或视频段）随机遮盖，要求模型预测这些被遮盖部分的内容。
*   <strong>词元 (token):</strong> 将视频帧分割成的小块，作为 Transformer 模型的输入单位。
*   <strong>模型预测控制 (Model Predictive Control, MPC):</strong> 一种规划算法。模型在脑中模拟多种可能的动作序列（推演），评估哪个序列最能达到目标，然后执行第一步动作。

## 3.2. 前人工作与技术演进
本文建立在以下核心技术基础之上：
*   **V-JEPA (Bardes et al., 2024):** 提出了视频领域的联合嵌入预测架构，但在模型规模和下游规划应用上有限。
*   **Vision Transformer (ViT):** 将 Transformer 架构应用于视觉，本文使用的 `ViT-g` 是其超大规模版本。
*   <strong>旋转位置嵌入 (Rotary Position Embedding, RoPE):</strong> 原本用于处理文本序列的位置信息。本文将其扩展为 **3D-RoPE**，用于同时处理视频的时间、高度和宽度维度，这对于稳定超大规模模型的预训练至关重要。

## 3.3. 差异化分析
相比于 **Cosmos** 等基于视频生成的模型，V-JEPA 2 的核心区别在于：
1.  **非生成式:** V-JEPA 2 不生成像素，只生成表示。这使其在规划时速度更快（相比 Cosmos 快 15 倍以上）。
2.  **低数据需求:** 仅需 62 小时机器人交互数据，而生成式模型通常需要成千上万小时。
3.  **抗噪性:** 能够忽略视频中不可预测的背景噪声，更专注于物理实体。

    ---

# 4. 方法论

V-JEPA 2 的训练分为两个关键阶段：第一阶段是无动作的视频特征学习，第二阶段是动作条件化的世界模型构建。

## 4.1. 第一阶段：V-JEPA 2 预训练
该阶段的目标是让模型学会“看”视频。下图（原文 Figure 2 左侧）展示了其预训练流程：

![Figure2 Multistage training. (Left) We first pretrain the V-JEPA 2 vido encoder on internet-scale image and atiea ioe nask s pl yopi se theoesTheerh pr mask viduc nutputs bedivectorornu token. Next, theutput thenc cntenate wise learnable mask tokenshat spehe posiithe aske pat, n subsey processed by the predictor. The outputs of the predictor are then regressed t the prediction targets using an L1 loss. The prediction targets are computed by an ema-encoder, the weights of which are defined as an exponential moving averageof the encoder weights. (Right) After pretraining, we freeze the video encoder and learn a new action-conditioned predictor, V-JEPA 2-AC, on top of the learned representation. We leverage an autoregressive faprcijecivavol riciherationsuimntine p vi cnsann-fat.Our-cpeorublc-cualtnpt a p ne a from current and previous time steps.](images/2.jpg)
*该图像是一个示意图，展示了 V-JEPA 2 和 V-JEPA 2-AC 的多阶段训练过程。左侧描述了 V-JEPA 2 的预训练步骤，包括编码器、预测器和 L1 损失的连接；右侧则展示了 V-JEPA 2-AC 的新特性，突出机器人动作与姿态的关系。*

### 4.1.1. 掩码去噪目标函数
模型通过预测隐藏的视频部分来学习。给定一段视频 $y$，我们随机遮盖其中的一些块，得到一个受损的视图 $x$。
*   <strong>编码器 $E_\theta(\cdot)$:</strong> 将视频块映射到特征空间。
*   <strong>预测器 $P_\phi(\cdot)$:</strong> 根据未遮盖部分的特征和遮盖位置的占位符（掩码词元），预测被遮盖部分的特征。

    其核心目标函数（公式 1）如下：
$$
\min_{\theta, \phi, \Delta_y} \| P_\phi(\Delta_y, E_\theta(x)) - \mathrm{sg}(E_{\bar{\theta}}(y)) \|_1
$$
**符号解释：**
*   $\theta, \phi$: 分别是编码器和预测器的参数。
*   $x$: 被遮盖的视频视图。
*   $y$: 完整的原始视频（作为目标）。
*   $\Delta_y$: 可学习的掩码词元，指示需要预测的视频位置。
*   $E_{\bar{\theta}}$: 教师编码器，其参数 $\bar{\theta}$ 是学生编码器参数 $\theta$ 的 <strong>指数移动平均 (Exponential Moving Average, EMA)</strong>。
*   $\mathrm{sg}(\cdot)$: <strong>停止梯度 (Stop-gradient)</strong> 操作，防止模型产生平凡解（崩塌）。
*   $\|\cdot\|_1$: $L_1$ 损失函数，衡量预测特征与真实特征之间的平均绝对误差。

## 4.2. 第二阶段：动作条件化世界模型 (V-JEPA 2-AC)
在预训练好的视觉特征基础上，我们加入机器人的动作指令，训练模型预测“如果我执行动作 $a$，未来会变成什么样”。

### 4.2.1. 输入与损失函数
我们将视频帧序列映射为特征序列 $(z_k)$，并将机器人末端执行器的状态 $s_k$ 和动作 $a_k$ 交织输入到预测器中。预测器采用<strong>块因果注意力机制 (Block-causal attention)</strong>，确保预测第 $k+1$ 帧时只能看到当前及以前的信息。

训练涉及两个损失函数（见原文 Figure 6）：
1.  <strong>教师强制损失 (Teacher-forcing Loss):</strong> 每一帧都给模型真实的当前帧，让它预测下一帧。
    $$
    \mathcal{L}_{\mathrm{teacher-forcing}}(\phi) := \frac{1}{T} \sum_{k=1}^{T} \| P_\phi((a_t, s_t, E(x_t))_{t \leq k}) - E(x_{k+1}) \|_1
    $$
2.  <strong>推演损失 (Rollout Loss):</strong> 让模型基于自己的预测进行多步“脑补”，以减少长程误差积累。
    $$
    \mathcal{L}_{\mathrm{rollout}}(\phi) := \| P_\phi(a_{1:T}, s_1, z_1) - z_{T+1} \|_1
    $$
    通过结合这两个损失，V-JEPA 2-AC 学会了在长时间内保持预测的准确性。

## 4.3. 动作规划与推理
在机器人部署阶段，我们使用 <strong>模型预测控制 (MPC)</strong>。

下图（原文 Figure 7）展示了规划原理：

![Figure 7 Planning. We plan an action sequence for a fixed time horizon $T$ by minimizing the L1 distance between the world model's imagined state representation $T$ steps into the future and its goal representation. The L1 loss is optimized with respect to the actions $( a _ { k } ) _ { k \\in \[ T \] }$ using the cross-entropy method (Rubinstein, 1997). Specifically, in each pla epe samle het corat pon he plahoi omq Gu isulzzan The atiaheo-r T pv before finally returning the mean of the sequence of Gaussians as the selected action trajectory.](images/7.jpg)
*该图像是示意图，展示了如何通过最小化世界模型的想象状态表示与目标表示之间的 L1 距离来规划一个固定时间范围 $T$ 的动作序列。具体地，图中分别展示了从当前观察到目标图像的处理步骤。*

给定一个目标图像的特征 $z_g$，模型通过最小化以下<strong>能量函数 (Energy Function)</strong> 来搜索最优动作序列 $\hat{a}_{1:T}$：
$$
\mathcal{E}(\hat{a}_{1:T}; z_k, s_k, z_g) := \| P(\hat{a}_{1:T}; s_k, z_k) - z_g \|_1
$$
**符号解释：**
*   $P(\hat{a}_{1:T}; \dots)$: 模型根据待选动作序列推演出的未来状态特征。
*   $z_g$: 目标图像的特征表示。
*   $\mathcal{E}$: 衡量当前“想象”的终点与目标的差距。

    模型使用 <strong>交叉熵方法 (Cross-Entropy Method, CEM)</strong> 不断采样并优化动作序列，最终执行能使特征距离最小化的动作。

---

# 5. 实验设置

## 5.1. 数据集
1.  **VideoMix22M (VM22M):** 本文构建的大规模视觉数据集。
    *   **来源:** 包含 Something-Something v2 (SSv2)、Kinetics、HowTo100M、YouTube-Temporal-1B (YT1B) 以及 ImageNet 图像。
    *   **规模:** 2200 万个样本，总时长超过 100 万小时。
2.  **Droid 数据集:** 用于后训练机器人动作。
    *   **规模:** 选取了约 62 小时的 Franka 机械臂操作视频。
3.  **Epic-Kitchens-100 (EK100):** 用于人类动作预测任务，包含第一人称视角的厨房活动。

## 5.2. 评估指标
1.  <strong>Top-1 准确率 (Top-1 Accuracy):</strong>
    *   **概念:** 模型预测概率最高的类别是否与真实标签一致。
    *   **公式:** $\text{Acc} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\hat{y}_i = y_i)$，其中 $\mathbb{I}$ 为指示函数。
2.  **Recall-at-5 (R@5):**
    *   **概念:** 在预测概率排名前 5 的类别中，只要包含真实标签即视为成功。常用于具有不确定性的动作预测任务。
    *   **公式:** $\text{R@5} = \frac{\text{真值出现在前5名的样本数}}{\text{总样本数}}$。
3.  <strong>成功率 (Success Rate):</strong> 机器人任务中，独立观察者判断任务是否完成。

## 5.3. 对比基线
*   **图像编码器:** DINOv2、SigLIP2、Perception Encoder。
*   **视频编码器:** VideoMAEv2、InternVideo2。
*   **机器人模型:** **Octo**（基于行为克隆的 VLA 模型）、**Cosmos**（基于视频生成的扩散模型）。

    ---

# 6. 实验结果与分析

## 6.1. 视频理解与动作分类
V-JEPA 2 在需要深层运动理解的任务（如 SSv2）上表现极其出色。

以下是原文 **Table 4** 的部分转录：

<table>
<thead>
<tr>
<th rowspan="2">方法</th>
<th rowspan="2">参数量</th>
<th colspan="3">运动理解 (Motion)</th>
<th colspan="3">外观理解 (Appearance)</th>
</tr>
<tr>
<th>SSv2</th>
<th>Diving-48</th>
<th>Jester</th>
<th>K400</th>
<th>COIN</th>
<th>IN1K</th>
</tr>
</thead>
<tbody>
<tr>
<td>DINOv2 (图像模型)</td>
<td>1.1B</td>
<td>50.7</td>
<td>82.5</td>
<td>93.4</td>
<td>83.6</td>
<td>90.7</td>
<td>86.1</td>
</tr>
<tr>
<td>InternVideo2</td>
<td>1B</td>
<td>69.7</td>
<td>86.4</td>
<td>-</td>
<td>89.4</td>
<td>93.8</td>
<td>85.8</td>
</tr>
<tr>
<td><b>V-JEPA 2 ViT-g (384)</b></td>
<td>1B</td>
<td><b>77.3</b></td>
<td><b>90.2</b></td>
<td><b>97.8</b></td>
<td>87.3</td>
<td>91.1</td>
<td>85.1</td>
</tr>
</tbody>
</table>

**分析:** V-JEPA 2 在 SSv2 上的 77.3 分显著超过了其他模型，这说明其在特征空间预测运动轨迹的能力远强于仅看单帧或对比学习的模型。

## 6.2. 机器人操作结果 (零起点规划)
这是本文最令人印象深刻的结果。模型在 Droid 数据集上训练后，直接部署在两个完全不同的实验室环境中。

以下是原文 **Table 2** 的转录：

<table>
<thead>
<tr>
<th rowspan="2">方法</th>
<th rowspan="2">到达 (Reach)</th>
<th colspan="2">抓取 (Grasp)</th>
<th colspan="2">搬运 (Reach w/ Obj)</th>
<th colspan="2">拾取与放置 (Pick-&-Place)</th>
</tr>
<tr>
<th>杯子</th>
<th>盒子</th>
<th>杯子</th>
<th>盒子</th>
<th>杯子</th>
<th>盒子</th>
</tr>
</thead>
<tbody>
<tr>
<td>Octo (行为克隆)</td>
<td>100%</td>
<td>15%</td>
<td>0%</td>
<td>15%</td>
<td>70%</td>
<td>15%</td>
<td>10%</td>
</tr>
<tr>
<td><b>V-JEPA 2-AC (本方法)</b></td>
<td>100%</td>
<td><b>65%</b></td>
<td><b>25%</b></td>
<td><b>75%</b></td>
<td><b>75%</b></td>
<td><b>80%</b></td>
<td><b>65%</b></td>
</tr>
</tbody>
</table>

**分析:** 相比于模仿学习模型 `Octo`，V-JEPA 2-AC 在复杂任务（如拾取与放置）上的成功率高出数倍。这证明了**世界模型+规划**在应对新环境和新物体时比简单的端到端映射（动作克隆）更具鲁棒性。

## 6.3. 规划效率对比
与生成式模型 **Cosmos** 相比，V-JEPA 2-AC 在单块 GPU 上的规划速度显著更快（原文 Table 3）：
*   **Cosmos:** 计算一次动作需 **4 分钟**，完成一次任务需 1 小时。
*   **V-JEPA 2-AC:** 计算一次动作仅需 **16 秒**。

    ---

# 7. 总结与思考

## 7.1. 结论总结
V-JEPA 2 证明了 Yann LeCun 的**世界模型**愿景在大规模视频数据上的可行性。通过在隐藏的表示空间进行预测，模型不仅获得了顶级的视觉理解能力，更重要的是，它能够作为一个“大脑”在物理世界中进行零起点的行动规划。

## 7.2. 局限性与未来工作
*   **相机灵敏度:** 模型对相机位置较敏感。由于是在单目 RGB 下训练，如果相机视角发生大幅旋转，模型推断的坐标轴可能会出现偏差（见附录 Figure 17）。
*   **长程规划:** 目前模型主要能规划几秒钟内的动作序列。更长任务（如做饭的全过程）需要分层规划，即在更抽象的时间尺度上进行预测。
*   **目标规范方式:** 目前依赖图像作为目标（Image Goal），未来应支持更自然的自然语言指令（Language Goal）。

## 7.3. 个人启发与批判
**启发:** 该工作展示了“预测”这一简单的自监督任务是如何演化出“规划”这一高级智能行为的。它避开了视频生成的昂贵计算开销，为轻量化、高性能的具身智能（Embodied AI）指明了方向。
**批判:** 虽然 62 小时的机器人数据很少，但 Droid 数据集本身质量极高且包含多视角。如果是在完全杂乱、无结构的野外环境下，仅靠单目相机，JEPA 的特征空间预测是否还能保持物理一致性，仍需进一步验证。此外，16 秒的规划延迟虽然比 Cosmos 快，但距离实时避障（毫秒级）仍有距离。