# 1. 论文基本信息

## 1.1. 标题
**Generalizable Humanoid Manipulation with 3D Diffusion Policies**
（基于 3D 扩散策略的可泛化人形机器人操作）

## 1.2. 作者
Yanjie Ze, Zixuan Chen, Wenhao Wang, Tianyi Chen, Xialin He, Ying Yuan, Xue Bin Peng, Jiajun Wu。
作者分别来自斯坦福大学 (Stanford University)、西蒙菲莎大学 (Simon Fraser University)、宾夕法尼亚大学 (UPenn)、伊利诺伊大学厄巴纳-香槟分校 (UIUC) 以及卡内基梅隆大学 (CMU)。

## 1.3. 发表期刊/会议
该论文发表于 **arXiv**（预印本平台），并被机器人领域顶级会议 **CoRL 2024**（机器人学习大会）录用。CoRL 是机器人学与机器学习交叉领域的顶级学术会议，以严谨的审稿和对实际物理效果的重视著称。

## 1.4. 发表年份
2024年（最新修订版于2024年10月发布）。

## 1.5. 摘要
人形机器人在多样化环境中自主运行一直是机器人专家的目标。然而，目前的自主操作大多局限于特定场景，主要原因是难以获得可泛化的技能以及野外机器人数据成本高昂。本研究构建了一个真实世界的机器人系统来解决这一挑战。该系统集成了：1) 全上半身机器人远程操作控制系统，用于获取类人机器人数据；2) 一个具有 25 个自由度 (DoF) 的人形机器人平台，配备可调高度的小车和 3D 激光雷达 (LiDAR) 传感器；3) 一种改进的 3D 扩散策略 (Improved 3D Diffusion Policy, iDP3) 学习算法，使机器人能从嘈杂的人类数据中学习。通过在真实机器人上进行 2000 多次推理轨迹 (rollout) 评估，证明了仅使用单一场景采集的数据，人形机器人即可在厨房、会议室和办公室等多种未见场景中自主执行任务。

## 1.6. 原文链接
- **arXiv 链接:** [https://arxiv.org/abs/2410.10803](https://arxiv.org/abs/2410.10803)
- **PDF 链接:** [https://arxiv.org/pdf/2410.10803v3.pdf](https://arxiv.org/pdf/2410.10803v3.pdf)
- **项目主页:** [https://humanoid-manipulation.github.io](https://humanoid-manipulation.github.io)

  ---

# 2. 整体概括

## 2.1. 研究背景与动机
*   **核心问题:** 尽管人形机器人硬件（如特斯拉 Optimus, Figure 等）取得了巨大进步，但它们的自主操作技能通常“见光死”——即在训练环境之外的新场景中表现极差。
*   **重要性:** 如果人形机器人不能走进千家万户或不同的工厂，其商业价值将大打折扣。
*   <strong>现有挑战 (Gap):</strong>
    1.  **数据成本:** 在野外（多样化场景）收集人形机器人数据非常昂贵且危险。
    2.  **泛化能力弱:** 现有的模仿学习算法多依赖 2D 图像，容易受背景、光照和物体颜色的干扰。
    3.  **系统复杂性:** 人形机器人自由度高，协同控制全身（头、手、腰）进行操作非常困难。
*   **创新思路:** 利用 <strong>3D 点云 (Point Cloud)</strong> 信息的几何不变性，结合 <strong>扩散策略 (Diffusion Policy)</strong> 的强大分布建模能力，开发一套只需在单一实验室场景训练，就能在现实世界中“零样本 (Zero-shot)”泛化的系统。

## 2.2. 核心贡献/主要发现
1.  **硬件系统集成:** 搭建了基于 Fourier GR1 的人形机器人平台，通过移动小车解决移动问题，通过 3D 激光雷达获取高质量空间数据。
2.  **数据采集系统:** 设计了基于 Apple Vision Pro 的全上半身远程操作 (Teleoperation) 系统，能够精准捕捉人类的头、手轨迹并映射给机器人。
3.  <strong>算法改进 (iDP3):</strong> 针对人形机器人的实时性和鲁棒性需求，改进了原有的 3D 扩散策略，使其不再依赖相机校准，且在噪声数据下更稳定。
4.  **实证泛化:** 首次证明了人形机器人可以凭借单一场景的 3D 训练数据，在完全陌生的环境中完成抓取、倒水等复杂任务。

    ---

# 3. 预备知识与相关工作

## 3.1. 基础概念
*   <strong>自由度 (Degrees of Freedom, DoF):</strong> 指机器人关节可以独立运动的方向数量。本文的人形机器人上半身拥有 25 个自由度。
*   <strong>3D 点云 (Point Cloud):</strong> 由 3D 传感器（如 LiDAR）扫描得到的空间点集合，每个点包含 `(x, y, z)` 坐标。相比 2D 图像，点云能直接表达物体的形状和距离，不随颜色或光照改变。
*   <strong>扩散策略 (Diffusion Policy):</strong> 一种基于扩散生成模型的机器人动作学习方法。它不像传统模型只预测一个动作，而是学习动作序列的概率分布，能够处理人类演示中的多峰性（即同一个任务有多种解法）。
*   <strong>远程操作 (Teleoperation):</strong> 由人类穿戴设备（如 VR 头显）实时控制机器人，以此采集高质量的专家演示数据。

## 3.2. 前人工作
作者对比了当前主流的人形/灵巧操作研究，如下表（原文 Table I）所示：

以下是原文 Table I 的结果：

<table>
<thead>
<tr>
<th rowspan="2">方法 (Method)</th>
<th colspan="3">远程操作 (Teleoperation)</th>
<th colspan="3">泛化能力 (Generalization Abilities)</th>
<th>策略评估</th>
</tr>
<tr>
<th>臂与手</th>
<th>头部</th>
<th>腰部</th>
<th>物体</th>
<th>视角</th>
<th>场景</th>
<th>真实世界回合数</th>
</tr>
</thead>
<tbody>
<tr>
<td>AnyTeleop [1]</td>
<td>✓</td>
<td>X</td>
<td>X</td>
<td>✓</td>
<td>X</td>
<td>X</td>
<td>0</td>
</tr>
<tr>
<td>DP3 [2]</td>
<td>-</td>
<td>X</td>
<td>X</td>
<td>√</td>
<td>√</td>
<td>X</td>
<td>186</td>
</tr>
<tr>
<td>HumanPlus [7]</td>
<td>✓</td>
<td>X</td>
<td>X</td>
<td>X</td>
<td>X</td>
<td>X</td>
<td>160</td>
</tr>
<tr>
<td>OpenTeleVision [10]</td>
<td>✓</td>
<td>✓</td>
<td>X</td>
<td>√</td>
<td>√</td>
<td>X</td>
<td>75</td>
</tr>
<tr>
<td><b>本工作 (This Work)</b></td>
<td><b>✓</b></td>
<td><b>✓</b></td>
<td><b>✓</b></td>
<td><b>√</b></td>
<td><b>√</b></td>
<td><b>√</b></td>
<td><b>2253</b></td>
</tr>
</tbody>
</table>

## 3.3. 差异化分析
*   **与 2D 策略相比:** 传统的 `Diffusion Policy` 使用图像作为输入，换个房间或换个桌子颜色通常就会失败。本文使用 3D 点云，关注的是“杯子是一个圆柱形空间物体”这一几何属性，因此换场景也能工作。
*   **与原生 DP3 相比:** 原生 `DP3` 针对机械臂设计，需要精确的相机校准（即知道相机相对于桌子的确切位置）。而在人形机器人上，头部是经常转动的，`iDP3` 引入了 <strong>第一人称视角 (Egocentric)</strong> 表示，消除了校准需求。

    ---

# 4. 方法论

本研究提出的系统通过集成硬件平台、远程操作和改进的算法，实现了高泛化性的操作。

## 4.1. 硬件与数据采集流
下图（原文 Figure 2）展示了系统的整体流程：从硬件平台到数据采集，再到算法学习与部署。

![该图像是一个示意图，展示了 humanoid 机器人的操作平台、数据采集、学习算法以及部署阶段。左侧介绍了包括 LiDAR 相机和人形机器人 GR1 的平台，中间部分展示了全身机电操作系统获取人类样本数据，右侧则描述了训练于单一场景并能推广到多样化场景的能力。](images/2.jpg)
*该图像是一个示意图，展示了 humanoid 机器人的操作平台、数据采集、学习算法以及部署阶段。左侧介绍了包括 LiDAR 相机和人形机器人 GR1 的平台，中间部分展示了全身机电操作系统获取人类样本数据，右侧则描述了训练于单一场景并能推广到多样化场景的能力。*

1.  **人形机器人平台:** 使用 Fourier GR1 机器人，配备灵巧手。为保证稳定性，将下半身固定在可调高度的移动小车上。
2.  **数据采集:** 使用 Apple Vision Pro 捕捉人类的姿态。
    *   **手部/手臂控制:** 利用 `Relaxed IK`（一种逆运动学算法）将人类手腕的 3D 位置映射为机器人的关节角。
    *   **头/腰协同:** 人类头部的旋转同步带动机器人的头部和腰部，从而扩大操作范围。
3.  **视觉反馈:** 将激光雷达采集的实时画面传回 Vision Pro，实现沉浸式的第一人称控制。

## 4.2. 核心算法：改进的 3D 扩散策略 (iDP3)

`iDP3` 的核心思想是学习一个条件生成模型，根据当前的 3D 观测预测未来的动作序列。

### 4.2.1. 第一人称视角 3D 表示 (Egocentric 3D Representations)
原有的 `DP3` 在世界坐标系（World Frame）下工作，这需要复杂的相机标定。`iDP3` 直接在 <strong>相机坐标系 (Camera Frame)</strong> 下处理点云。

如下图（原文 Figure 3）所示：

![Fig. 3: iDP3 utilizes 3D representations in the camera frame, while the 3D representations of other recent 3D policies including DP3 \[2\] are in the world frame, which relies on accurate camera calibration and can not be extended to mobile robots.](images/3.jpg)
*该图像是示意图，展示了相机坐标系（iDP3）和世界坐标系（DP3及其他3D策略）之间的关系。相机坐标系中的3D表示依赖于相机框架，而世界坐标系则需要准确的相机校准，无法扩展到移动机器人。*

这意味着无论机器人如何走动或转头，点云数据始终相对于相机中心定义。这种方式天然适合移动机器人和人形机器人，因为它们没有固定的“世界参考系”。

### 4.2.2. 视觉编码器与特征提取
为了从原始点云中提取有用信息，`iDP3` 采用了一个 <strong>金字塔卷积编码器 (Pyramid Convolutional Encoder)</strong>。

1.  **点云采样:** 系统首先对原始点云进行处理。为了兼顾速度和覆盖面，采用 <strong>体素采样 (Voxel Sampling)</strong> 结合 <strong>均匀采样 (Uniform Sampling)</strong>。
2.  **规模化输入:** 原生 `DP3` 只使用 1024 个点，这对于复杂场景不够。`iDP3` 将采样点数增加到 $N=4096$，以捕捉更多的背景和环境信息。
3.  **多层特征融合:** 编码器不仅使用最后一层的全局特征，还融合了中间层的局部特征。通过一系列卷积层提取特征向量 $f_{vis}$。

### 4.2.3. 扩散动作生成
动作预测被建模为一个去噪过程。假设我们要预测未来 $T_p$ 步的动作序列 $A = \{a_1, a_2, ..., a_{T_p}\}$：

1.  **状态输入:** 将视觉特征 $f_{vis}$ 与机器人的本体感受数据 $s_{prop}$（如关节位置）拼接。
2.  **预测步长:** 论文发现，人类演示的数据存在抖动（噪声）。为了平滑这些噪声，`iDP3` 将 <strong>预测步长 (Prediction Horizon)</strong> 从原本的 4 步延长到 $T_p=16$ 步。
3.  **去噪迭代:** 算法从一个纯高斯噪声序列 $A^K$ 开始，通过 $K$ 次去噪迭代得到最终动作 $A^0$。在第 $k$ 步去噪中：
    $$A^{k-1} = \alpha \cdot \epsilon_{\theta}(A^k, k, f_{vis}, s_{prop}) + \gamma \cdot A^k$$
    （注：此处公式为扩散模型去噪的简化示意，$\epsilon_{\theta}$ 为待学习的神经网络，负责预测噪声；$\alpha, \gamma$ 为调度参数。）

这种长程预测机制使得机器人的动作比直接跟随人类的即时指令更加丝滑和准确。

---

# 5. 实验设置

## 5.1. 实验任务与数据
实验选择了三个具有代表性的任务：
*   <strong>抓取与放置 (Pick &amp; Place):</strong> 抓取随机位置的杯子并移动。
*   <strong>倒水 (Pour):</strong> 拿起水壶将水倒入杯中。
*   <strong>擦拭 (Wipe):</strong> 使用抹布清洁桌面。

    **训练集规模:** 极其精简。每个任务仅在 **一个固定实验室场景** 中收集约 200 个成功回合。

## 5.2. 评估指标
对每一个策略，论文运行了超过 100 次真实世界测试，并记录以下指标：
1.  <strong>成功率 (Success Rate):</strong> 
    $$SR = \frac{\text{成功完成任务的次数}}{\text{总尝试次数}}$$
    用于衡量策略的准确性。
2.  <strong>尝试总数 (Total Attempts):</strong> 在 1000 步动作内，机器人尝试进行抓取等核心动作的总次数。如果该数值过低，说明策略在犹豫或“卡死”；如果过高且不成功，说明动作不平稳。

## 5.3. 对比基线 (Baselines)
*   **DP (Diffusion Policy):** 基于图像的标准扩散策略。
*   **DP (*R3M):** 使用预训练视觉模型 R3M 提取图像特征的扩散策略。
*   **DP3:** 未经改进的原始 3D 扩散策略。

    ---

# 6. 实验结果 with 分析

## 6.1. 核心结果：3D 信息的优越性
以下是原文 Table II 展示的在不同训练规模（如 1st-1 代表第一种设置下的少量数据）下的表现对比：

以下是原文 Table II 的结果：

<table>
<thead>
<tr>
<th>基线模型 (Baselines)</th>
<th>DP</th>
<th>DP3</th>
<th>DP (*R3M 冻结)</th>
<th>DP (*R3M 微调)</th>
<th>iDP3 (DP3 编码器)</th>
<th><b>iDP3 (本方法)</b></th>
</tr>
</thead>
<tbody>
<tr>
<td><b>成功/总尝试 (总计)</b></td>
<td>24/106</td>
<td>0/0</td>
<td>62/138</td>
<td>99/147</td>
<td>58/127</td>
<td><b>75/139</b></td>
</tr>
</tbody>
</table>

**分析:** 
*   **原版 DP3 失效:** 在人形机器人这种动态视角下，不经改进的 `DP3` 完全无法工作（0 成功），证明了第一人称视角改进的必要性。
*   **iDP3 vs 图像模型:** 虽然微调过的图像模型 (DP *R3M) 在训练场景中表现极佳，但在随后的泛化测试中表现极差。

## 6.2. 零样本泛化能力 (Zero-shot Generalization)
这是本论文最惊人的结果。在训练场景之外的测试中（见原文 Table IV）：

<table>
<thead>
<tr>
<th>任务场景</th>
<th>图像模型 (DP) 成功率</th>
<th><b>iDP3 成功率</b></th>
</tr>
</thead>
<tbody>
<tr>
<td><b>训练场景</b></td>
<td>9/10</td>
<td>9/10</td>
</tr>
<tr>
<td><b>新物体 (New Object)</b></td>
<td>3/10</td>
<td><b>9/10</b></td>
</tr>
<tr>
<td><b>新视角 (New View)</b></td>
<td>2/10</td>
<td><b>9/10</b></td>
</tr>
<tr>
<td><b>新场景 (New Scene)</b></td>
<td>2/10</td>
<td><b>9/10</b></td>
</tr>
</tbody>
</table>

**深入分析:**
*   **视角不变性:** 如图 Figure 8 所示，即使人为移动相机位置，`iDP3` 依然能准确抓取。这是因为 3D 点云提供的空间结构信息在坐标变换下是稳定的，而图像模型会因为像素分布的巨大变化而崩溃。
*   **物体泛化:** 即使训练时只用过红杯子，面对 Figure 9 中的各种瓶子，`iDP3` 也能凭借“圆柱状物体”的 3D 特征成功抓取。

## 6.3. 训练效率
如下图（原文 Figure 7）所示，`iDP3` 的训练时间显著低于基于图像的模型。

![Fig. 7: Training time. Due to using 3D representations, iDP3 saves training time compared to Diffusion Policy (DP), even after we scale up the 3D vision input. This advantage becomes more evident when the number of demonstrations gets large.](images/7.jpg)
*该图像是图表，展示了两种算法（DP与iDP3）在训练时间上的比较。图表中，DP算法耗时80分钟，而iDP3算法仅需30分钟，显示出iDP3在训练效率上的显著优势。*

这是因为点云数据的维度虽然在空间上是 3D 的，但经过稀疏采样后，其有效输入量远小于高分辨率图像像素，极大地加速了模型的收敛。

---

# 7. 总结与思考

## 7.1. 结论总结
本文成功构建了一个端到端的人形机器人学习系统。其核心结论是：**3D 视觉表示是机器人实现跨场景泛化的关键。** 通过将 3D 扩散策略改进为适用于移动头部的第一人称模式，并配合长程动作预测，人形机器人展现出了极强的适应性，打破了“一机一景”的实验局限。

## 7.2. 局限性与未来工作
*   **远程操作疲劳:** 使用 Apple Vision Pro 进行长时间数据采集对人类来说非常劳累，限制了数据规模的进一步扩大。
*   **传感器噪声:** 激光雷达在面对透明物体或强光时会产生空洞或噪点，这会直接影响 `iDP3` 的判断。
*   **下半身缺失:** 目前为了安全禁用了腿部行走，未来需要探索如何将该策略与真正的全身平衡控制（Whole-body Control）相结合。
*   **精细操作:** 对于拧螺丝等极高精度要求的任务，目前的 3D 采样密度可能还不够。

## 7.3. 个人启发与批判
*   **启发:** 该工作证明了“几何胜过语义”。在机器人操作中，理解物体的 3D 结构（它是多大、在哪）比识别它的语义标签（它是可口可乐还是百事可乐）对于泛化更重要。
*   **批判:** 论文中虽然强调了“单一场景训练”，但实际上为了保证泛化，其 3D 采样的点数高达 4096，这在车载芯片或嵌入式端进行 15Hz 实时推理时压力很大。此外，系统对高度可调小车的依赖，掩盖了人形机器人最核心的平衡难题。真正的泛化应该包括对地面崎岖程度和机器人晃动的适应。