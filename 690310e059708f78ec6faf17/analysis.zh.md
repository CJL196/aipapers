# 1. 论文基本信息

## 1.1. 标题
论文标题为 **$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control**（$\pi_0$：一种用于通用机器人控制的视觉 - 语言 - 动作流模型）。该标题直接揭示了论文的核心研究对象是一个名为 $\pi_0$ 的模型，其关键特征在于结合了视觉（Vision）、语言（Language）和动作（Action），并采用了流匹配（Flow Model）技术，旨在解决通用机器人控制问题。

## 1.2. 作者
论文作者来自 **Physical Intelligence** 公司以及部分学术机构合作者。主要作者包括 Kevin Black, Noah Brown, Danny Driess, Chelsea Finn, Sergey Levine 等。这些作者在机器人学习（Robot Learning）、强化学习（Reinforcement Learning）以及多模态大模型（Multimodal Large Models）领域具有深厚的研究背景。Physical Intelligence 是一家专注于开发通用机器人基础模型的公司。

## 1.3. 发表期刊/会议
该论文发布于 **arXiv** 预印本平台，发布时间为 **2024 年 10 月 31 日**。截至当前分析时间，该论文尚未正式发表于特定的学术期刊或会议，但 arXiv 是计算机科学和人工智能领域最权威的预印本发布渠道，具有重要的学术影响力。

## 1.4. 发表年份
论文发表年份为 **2024 年**。

## 1.5. 摘要
论文摘要指出，机器人学习有望解锁灵活、通用和灵巧机器人系统的全部潜力，但在数据、泛化性和鲁棒性方面面临重大障碍。本文探讨了通用机器人策略（即机器人基础模型）如何解决这些挑战。作者提出了一种建立在预训练视觉 - 语言模型（VLM）之上的新型<strong>流匹配（Flow Matching）</strong>架构，以继承互联网规模的语义知识。该模型在来自多个灵巧机器人平台（包括单臂、双臂和移动操纵器）的大型多样化数据集上进行训练。评估结果显示，该模型在预训练后具备零样本（Zero-shot）任务执行能力，能遵循人类语言指令，并能通过微调（Fine-tuning）获取新技能。实验涵盖了折叠衣物、清理桌子等多种复杂任务。

## 1.6. 原文链接
*   **ArXiv 摘要页:** https://arxiv.org/abs/2410.24164
*   **PDF 下载链接:** https://arxiv.org/pdf/2410.24164v3.pdf
*   **发布状态:** 预印本（Preprint）

# 2. 整体概括

## 2.1. 研究背景与动机
当前机器人学习领域面临的核心问题是<strong>泛化性（Generality）</strong>不足。现有的机器人系统通常是专用的，针对特定任务训练，难以适应新环境或新任务。虽然大型语言模型（LLM）和视觉 - 语言模型（VLM）在语义理解和通用推理方面表现出色，但它们缺乏与物理世界的实际交互能力（即“具身性”，Embodiment）。

具体挑战包括：
1.  **数据稀缺：** 高质量的机器人操作数据难以收集，限制了模型的训练规模。
2.  **泛化困难：** 专用模型难以将技能迁移到未见过的物体或环境中。
3.  **鲁棒性不足：** 现有模型在面对干扰或错误时缺乏恢复能力。

    论文的切入点是利用<strong>机器人基础模型（Robot Foundation Models）</strong>的概念，借鉴自然语言处理中“预训练 + 微调”的成功范式。通过在一个巨大的、多样化的机器人数据集上预训练模型，使其获得广泛的物理世界知识，然后通过微调适应特定任务。

## 2.2. 核心贡献/主要发现
论文的主要贡献包括：
1.  **$\pi_0$ 模型架构：** 提出了一种新的视觉 - 语言 - 动作（VLA）模型架构。它基于预训练的 VLM（PaliGemma），并引入了<strong>流匹配（Flow Matching）</strong>技术来生成连续的动作分布，替代了传统的自回归（Autoregressive）离散动作生成方法。这使得模型能够以高频（高达 50 Hz）控制机器人执行灵巧任务。
2.  <strong>训练策略（Recipe）：</strong> 设计了一套包含<strong>预训练（Pre-training）</strong>和<strong>后训练（Post-training）</strong>的多阶段训练流程。预训练使用多样化但质量参差不齐的数据以获取通用能力，后训练使用高质量 curated 数据以提升特定任务的性能和鲁棒性。
3.  **大规模实验验证：** 在超过 10,000 小时的机器人数据上进行了预训练，并在 7 种不同的机器人配置上进行了评估。实验表明，$\pi_0$ 在零样本任务、语言指令遵循以及复杂多阶段任务（如折叠衣物、组装盒子）上均显著优于现有的基线模型（如 OpenVLA, Octo）。

# 3. 预备知识与相关工作

## 3.1. 基础概念
为了理解本文，读者需要掌握以下核心概念：

*   <strong>视觉 - 语言模型 (Vision-Language Model, VLM):</strong> 一种能够同时处理图像和文本输入的人工智能模型。它们通常在互联网规模的图像 - 文本对上进行预训练，能够理解图像内容并用自然语言描述，或根据文本指令识别图像中的对象。本文利用 VLM 作为主干网络，以继承其强大的语义理解能力。
*   <strong>流匹配 (Flow Matching):</strong> 一种生成模型技术，类似于扩散模型（Diffusion Models）。它通过学习一个向量场（Vector Field），将简单的噪声分布（如高斯分布）逐渐“流动”变换为复杂的数据分布（如机器人动作分布）。相比传统的扩散模型，流匹配通常具有更简单的训练目标和更高效的采样过程。
*   <strong>动作分块 (Action Chunking):</strong> 机器人控制策略不是一次预测一个动作，而是预测未来一段时间内的一系列动作（即一个“块”或 Chunk）。这有助于提高控制的平滑性和频率，减少累积误差。
*   <strong>跨实体训练 (Cross-Embodiment Training):</strong> 指将来自不同形态机器人（如单臂、双臂、移动底盘）的数据混合在一起训练同一个模型。这要求模型能够处理不同维度的动作空间和观测空间。
*   <strong>微调 (Fine-tuning):</strong> 在预训练模型的基础上，使用特定任务的小规模数据集进行进一步训练，以使模型适应特定任务的过程。

## 3.2. 前人工作
本文建立在以下几个关键研究方向之上：

*   <strong>视觉 - 语言 - 动作模型 (VLA Models):</strong> 如 RT-2 和 OpenVLA。这些模型将 VLM 与机器人动作输出相结合。通常，它们使用自回归方式将动作离散化为 token 进行预测。本文指出这种方法难以支持高频动作分块。
*   <strong>扩散策略 (Diffusion Policy):</strong> 如 Diffusion Policy 和 Octo。这些模型使用扩散过程生成动作，能够处理连续动作空间。本文的流匹配方法是对这一思路的改进，结合了 VLM 的语义能力。
*   **大规模机器人学习:** 如 Open X-Embodiment (OXE) 数据集。 prior work 收集了大量机器人数据，但通常用于训练较小的模型或特定任务。本文使用了规模更大（10,000+ 小时）且更侧重灵巧操作的数据集。

## 3.3. 技术演进
机器人控制策略经历了从经典控制理论到深度学习，再到基础模型的演变。
1.  **早期：** 基于规则的控制器或简单的强化学习，泛化性差。
2.  **中期：** 模仿学习（Imitation Learning）和行为克隆（Behavior Cloning），依赖特定任务数据。
3.  **近期：** 引入 Transformer 架构和大规模预训练。RT-2 展示了 VLM 知识迁移到机器人的可能性，但动作表示受限。Octo 展示了扩散模型在机器人上的应用，但缺乏 VLM 的语义 grounding。
4.  **本文工作：** 处于技术前沿，结合了 VLM 的语义能力、流匹配的连续动作生成能力以及大规模跨实体预训练策略。

## 3.4. 差异化分析
与相关工作相比，$\pi_0$ 的核心区别在于：
1.  **动作生成机制：** 不同于 OpenVLA 的自回归离散化，$\pi_0$ 使用流匹配生成连续动作，支持高频控制（50 Hz）。
2.  **架构设计：** 引入了<strong>动作专家（Action Expert）</strong>模块，专门处理机器人特有的状态和动作输入，而 VLM 主干处理视觉和语言，两者通过注意力机制交互。
3.  **训练数据规模与质量：** 使用了远超以往研究的灵巧操作数据量，并明确区分了预训练（多样化）和后训练（高质量）的数据策略。

# 4. 方法论

## 4.1. 方法原理
$\pi_0$ 的核心思想是将机器人控制建模为一个条件生成问题。给定当前的观测（图像、语言指令、本体状态），模型需要生成未来的动作序列。为了实现这一点，作者采用了<strong>条件流匹配（Conditional Flow Matching）</strong>。

直觉上，流匹配试图学习一个过程，将随机噪声逐渐转化为符合真实数据分布的动作。与扩散模型类似，但流匹配通过优化一个更直接的向量场目标，通常能提供更稳定的训练和更快的推理。通过基于预训练的 VLM，模型能够理解复杂的语言指令和视觉场景，从而生成语义正确的动作。

## 4.2. 核心方法详解

### 4.2.1. 模型架构
$\pi_0$ 模型主要基于 **PaliGemma** 视觉 - 语言模型。PaliGemma 是一个开源的 30 亿参数 VLM。为了适应机器人控制，作者对其进行了扩展：

1.  **输入编码：** 机器人的观测 $\mathbf{o}_t$ 包括多个 RGB 图像 $\mathbf{I}_t^i$、语言指令 $\ell_t$ 和本体状态 $\mathbf{q}_t$（如关节角度）。图像和状态通过编码器映射到与语言词元（Token）相同的嵌入空间。
2.  <strong>动作专家 (Action Expert)：</strong> 这是一个关键的架构创新。模型包含两组权重：一组用于处理图像和文本（VLM 主干），另一组专门用于处理机器人特定的输入（状态和动作）。这类似于混合专家（Mixture of Experts）设计。动作专家使用双向注意力掩码，允许所有动作词元相互关注。
3.  **输出：** 模型输出不是离散的动作 token，而是用于流匹配过程的向量场。

    下图（原文 Figure 3）展示了模型的整体框架概述：

    ![该图像是示意图，展示了一个基于预训练的视觉语言模型 (VLM) 的机器人控制架构。图中包含了多种机器人操作平台及其相关任务，如折衬衫与清理桌子，强调了机器人学习在复杂任务中的应用。](images/3.jpg)
    *该图像是示意图，展示了一个基于预训练的视觉语言模型 (VLM) 的机器人控制架构。图中包含了多种机器人操作平台及其相关任务，如折衬衫与清理桌子，强调了机器人学习在复杂任务中的应用。*

### 4.2.2. 流匹配动作生成
模型使用条件流匹配来建模动作的条件分布 $p(\mathbf{A}_t | \mathbf{o}_t)$，其中 $\mathbf{A}_t$ 是未来动作块（Action Chunk），$\mathbf{o}_t$ 是观测。

在训练过程中，模型学习预测一个向量场，该场将噪声动作引导回真实动作。具体的损失函数如下：

$$
L^{\tau}(\boldsymbol{\theta}) = \mathbb{E}_{p(\mathbf{A}_{t}|\mathbf{o}_{t}),q(\mathbf{A}_{t}^{\tau}|\mathbf{A}_{t})}||\mathbf{v}_{\boldsymbol{\theta}}(\mathbf{A}_{t}^{\tau},\mathbf{o}_{t})-\mathbf{u}(\mathbf{A}_{t}^{\tau}|\mathbf{A}_{t})||^{2},
$$

其中：
*   $L^{\tau}(\boldsymbol{\theta})$ 是流匹配损失函数。
*   $\boldsymbol{\theta}$ 是模型参数。
*   $\mathbb{E}$ 表示期望值。
*   $p(\mathbf{A}_{t}|\mathbf{o}_{t})$ 是真实动作数据的条件分布。
*   $q(\mathbf{A}_{t}^{\tau}|\mathbf{A}_{t})$ 是噪声过程，定义为 $q(\mathbf{A}_{t}^{\tau}|\mathbf{A}_{t}) = \mathcal{N}(\tau \mathbf{A}_{t}, (1 - \tau) \mathbf{I})$。这里 $\tau \in [0, 1]$ 是流匹配的时间步。
*   $\mathbf{A}_{t}^{\tau}$ 是加噪后的动作块，计算方式为 $\mathbf{A}_{t}^{\tau} = \tau \mathbf{A}_{t} + (1 - \tau) \epsilon$，其中 $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 是随机噪声。
*   $\mathbf{v}_{\boldsymbol{\theta}}(\mathbf{A}_{t}^{\tau},\mathbf{o}_{t})$ 是模型预测的向量场。
*   $\mathbf{u}(\mathbf{A}_{t}^{\tau}|\mathbf{A}_{t})$ 是目标去噪向量场，定义为 $\mathbf{u}(\mathbf{A}_{t}^{\tau}|\mathbf{A}_{t}) = \epsilon - \mathbf{A}_{t}$。

    这个公式的本质是训练模型预测噪声与真实动作之间的差异方向。通过最小化预测向量场与目标向量场之间的均方误差，模型学会了如何从噪声中恢复出正确的动作。

为了优化训练，作者采样流匹配时间步 $\tau$ 时使用了 Beta 分布，强调较低的时间步（即噪声较大的状态），因为此时预测任务更难且更重要。下图（原文 Figure 14）展示了时间步采样分布：

![Fig. 14: Flow matching timestep sampling distribution. We sample $\\tau$ from a shifted beta distribution that emphasizes lower timesteps (corresponding to noisier actions), and does not sample timesteps at all above a cutoff value $s$ We use $s = 0 . 9 9 9$ in our experiments.](images/14.jpg)
*该图像是图表，展示了流匹配时间步采样分布 `p( au)`。我们从一个偏移的贝塔分布中采样 `au`，该分布强调较低时间步（对应于噪声较大的动作），并且在截止值 $s$ 以上不进行采样。实验中，我们使用 $s = 0.999$ 作为截止值。*

### 4.2.3. 推理过程
在推理（Inference）阶段，模型通过积分学习到的向量场来生成动作。从随机噪声 $\mathbf{A}_{t}^{0} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 开始，逐步更新动作直到 $\tau=1$。使用的前向欧拉积分规则（Forward Euler Integration Rule）如下：

$$
\mathbf{A}_{t}^{\tau+\delta}=\mathbf{A}_{t}^{\tau}+\delta\mathbf{v}_{\theta}(\mathbf{A}_{t}^{\tau},\mathbf{o}_{t}),
$$

其中：
*   $\mathbf{A}_{t}^{\tau+\delta}$ 是下一步的动作估计。
*   $\delta$ 是积分步长（实验中 $\delta = 0.1$，即 10 步积分）。
*   $\mathbf{v}_{\theta}(\mathbf{A}_{t}^{\tau},\mathbf{o}_{t})$ 是模型在当前噪声水平下预测的向量场。

    为了提高效率，推理时可以缓存观测 $\mathbf{o}_t$ 的注意力键值（Keys and Values），仅重新计算动作词元部分。

### 4.2.4. 训练策略：预训练与后训练
论文强调训练策略（Recipe）与架构同样重要。采用了两阶段训练法：

1.  <strong>预训练 (Pre-training):</strong> 目标是获得广泛的通用能力。使用多样化但质量不一的数据集（包括开源 OXE 数据和作者自有的灵巧操作数据）。这使得模型能够学会从错误中恢复，并适应多种场景。
2.  <strong>后训练 (Post-training):</strong> 目标是获得特定任务的高性能。使用高质量、精心策划的数据集对预训练模型进行微调。这教会模型以流畅、高效的策略执行任务。

    下图（原文 Figure 4）展示了预训练数据混合的概述：

    ![该图像是饼图，展示了不同机器人平台在某一任务中的使用比例。左侧饼图显示各平台的占比情况，其中“Bimanual ARX”占比最高，为51%。右侧饼图则细分了其他平台的具体占比情况，“Bimanual AgileX”和“UR5e”等平台的比例依次为34.2%、13.7%和16.3%。图中颜色和标签清晰区分了各个机器人平台的名称。](images/4.jpg)
    *该图像是饼图，展示了不同机器人平台在某一任务中的使用比例。左侧饼图显示各平台的占比情况，其中“Bimanual ARX”占比最高，为51%。右侧饼图则细分了其他平台的具体占比情况，“Bimanual AgileX”和“UR5e”等平台的比例依次为34.2%、13.7%和16.3%。图中颜色和标签清晰区分了各个机器人平台的名称。*

# 5. 实验设置

## 5.1. 数据集
实验使用了大规模且多样化的机器人数据集，总计超过 10,000 小时的机器人操作数据。

*   **开源数据：** 包含 Open X-Embodiment (OXE)、Bridge v2 和 DROID 数据集。这些数据覆盖了广泛的物体和环境，但通常控制频率较低（2-10 Hz）。
*   **自有数据：** 作者收集了 903M 个时间步的数据，其中 106M 来自单臂机器人，797M 来自双臂机器人。涵盖 68 种任务，包括复杂的灵巧操作（如清理桌子、折叠衣物）。
*   **机器人平台：** 数据来自 7 种不同的机器人配置，包括 UR5e、Franka、Trossen、ARX、AgileX 以及移动操纵器（Mobile Manipulators）。

    下图（原文 Figure 5）展示了实验中使用的多种机器人平台：

    ![Fig. 5: The robots used in our experiments. These include single and dual-arm manipulators with 6-DoF and 7-DoF arms, as well as holonomic and nonholonomic mobile manipulators. $\\pi _ { 0 }$ is trained jointly on all of these platforms.](images/5.jpg)
    *该图像是图示，展示了用于实验的多种机器人，包括双臂和单臂操纵器，以及移动操纵器。图中的机器人包括双臂UR5e、双臂Trossen、双臂ARX、UR5e、Franka、移动Trossen和移动Fibocom，体现了多样化的机器人平台。*

数据混合时，为了平衡不同任务 - 机器人组合的样本量，作者对每个组合的权重进行了调整（权重与样本数的 0.43 次方成正比），以防止过代表的数据主导训练。

## 5.2. 评估指标
论文使用了多种评估指标来量化模型性能，主要包括任务成功率及归一化分数。

*   **概念定义：** 对于每个任务，设计了一个评分标准（Rubric），衡量任务完成的进度。完全成功得 1.0 分（或该任务的最大分），部分成功得分数比例。
*   **数学公式：** 归一化分数通常计算为：
    $$
    \text{Score} = \frac{\text{Actual Points Earned}}{\text{Maximum Possible Points}}
    $$
    其中 $\text{Actual Points Earned}$ 是机器人实际获得的分数，$\text{Maximum Possible Points}$ 是任务满分。
*   **符号解释：** 例如，在“清理桌子（Bussing）”任务中，如果桌上有 12 个物体，每个正确放置得 1 分，满分 12 分。若机器人正确放置了 6 个，则得分为 $6/12 = 0.5$。

    具体的评分标准详见附录 E，涵盖了折叠、抓取、放置等多个维度的成功判定。

## 5.3. 对比基线
为了验证 $\pi_0$ 的有效性，论文选择了多个具有代表性的基线模型进行比较：

1.  **OpenVLA:** 一个 70 亿参数的 VLA 模型，使用自回归方式生成动作。代表了当前基于 VLM 的机器人策略的先进水平。
2.  **Octo:** 一个 9300 万参数的模型，使用扩散过程生成动作。代表了非 VLM 初始化的通用机器人策略。
3.  **$\pi_0$-small:** $\pi_0$ 的缩小版，不使用 VLM 初始化，参数量约 4.7 亿。用于评估 VLM 预训练带来的增益。
4.  **ACT & Diffusion Policy:** 专门用于灵巧操作的传统模仿学习算法，通常在小数据集上训练。

## 5.4. 实现细节与推理效率
模型在 NVIDIA GeForce RTX 4090 GPU 上进行推理测试。由于模型生成整个动作块，推理不需要每步都进行。对于 20 Hz 的机器人，每 0.8 秒推理一次；对于 50 Hz 的机器人，每 0.5 秒推理一次。

以下是原文 Table I 的推理时间结果：

| model part | inference time |
| :--- | :--- |
| image encoders | 14 ms |
| observation forward pass | 32 ms |
| x10 action forward pass (flow) | 27 ms |
| network latency (if off-board) | 13 ms |
| **total on-board inference** | **73 ms** |
| **total off-board inference** | **86 ms** |

这表明模型能够满足实时控制的需求（总延迟远低于控制周期）。

# 6. 实验结果与分析

## 6.1. 核心结果分析

### 6.1.1. 开箱即用评估 (Out-of-Box Evaluation)
在预训练后，不进行任何微调，直接评估模型在五个任务上的表现：衬衫折叠、简单清理、困难清理、杂货装袋、吐司机取吐司。

下图（原文 Figure 6）展示了这些开箱即用评估任务：

![Fig. 6: Out-of-box evaluation tasks: To evaluate our base model, we run it after pre-training on five tasks: shirt folding, bussing easy, bussing hard, grocery bagging, and toast out of toaster. The tasks require a combination of dexterous manipulation, multi-stage behaviors, and semantic recognition.](images/6.jpg)
*该图像是展示针对五个任务的评估过程示意图，包括衬衫折叠、简单和复杂的餐具清理、装袋杂货和吐司机里拿吐司等。每个任务结合了灵巧的操作、多阶段的行为和语义识别。*

实验结果显示，$\pi_0$ 在所有任务上均显著优于基线模型。即使是训练步数较少（160k 步，与基线持平）的 $\pi_0$ 版本，也超过了所有基线。完整的 $\pi_0$ 模型（700k 步）取得了最佳结果。

下图（原文 Figure 7）展示了开箱即用评估的详细结果对比：

![Fig. 7: Out-of-box evaluation results: We evaluate $\\pi _ { 0 }$ trained for the full 700k steps, a version trained for $1 6 0 \\mathrm { k }$ steps that matches the number of updates for baseline models, $\\pi _ { 0 }$ -small, and three baselines: OpenVLA and Octo trained on all of our data, and OpenVLA trained only on the UR5e tasks (which we found to work better on UR5e tasks). Across all tasks and all comparisons, even the "parity" version of our model outperforms all baselines, and the full version of our model achieves the best results by a large margin.](images/7.jpg)
*该图像是图表，展示了不同模型在多个任务中的直接提示性能。各模型的平均任务进展通过柱状图表示，包含了模型 $oldsymbol{ ext{π}_0}$、$oldsymbol{ ext{π}_0}$ (parity)、$oldsymbol{ ext{π}_0}$-small、OpenVLA 以及 Octo。结果显示，$oldsymbol{ ext{π}_0}$ 模型在大多数任务中表现最佳。*

**分析：** OpenVLA 表现较差，因为其自回归架构不支持动作分块，难以处理高频控制。Octo 支持动作分块但表示能力有限。$\pi_0$-small 优于 OpenVLA 但不如完整 $\pi_0$，证明了 VLM 预训练和模型规模的重要性。

### 6.1.2. 语言指令遵循能力
评估模型遵循语言指令的能力，包括直接指令和中间步骤指令。任务包括清理桌子、设置桌子、杂货装袋。

下图（原文 Figure 8）展示了语言评估中的任务示意图：

![Fig. 8: The tasks in our language evaluation. We evaluate our model on 3 different language-conditioned tasks, each of which requires following a sequence of intermediate language commands. The tasks involve bussing a table (top) to put dishes in a bin and garbage in a trash bin, setting a table (middle) by taking items out of a bin, and packing a shopping bag (bottom).](images/8.jpg)
*该图像是一个示意图，展示了在语言评估中评估模型的三种不同语言条件任务。任务包括对桌子进行清理（顶部），将餐具放入箱子和垃圾放入垃圾桶；设置桌子（中间），从箱子中取出物品；以及打包购物袋（底部）。*

实验比较了三种条件：
1.  **Flat:** 仅接收总体任务命令。
2.  **Human:** 接收来自人类专家的中间步骤命令。
3.  **HL (High-Level):** 接收来自高级 VLM 策略的中间命令。

    下图（原文 Figure 9）展示了语言评估的结果：

    ![Fig. 9: Language evaluation. We compare "flat" versions of our policies, —flat, which receive only the overall task command (e.g., "bag the groceries") with a method that receives intermediate commands from a human expert, —human, or a high-level VLM policy, $- \\mathrm { H L }$ . We also compare our model to a small non-VLM variant under the "expert" condition, $\\pi _ { 0 }$ and $\\pi _ { 0 }$ -small, in terms of language following accuracy. The results show a significant improvement with $\\pi _ { 0 }$ from intermediate language commands provided by a human expert and to a lesser degree by an autonomous high-level policy. Notably, due to $\\pi _ { 0 }$ -small's limited language following ability, overall it does not gain with the addition of a high-level expert.](images/9.jpg)
    *该图像是一个条形图，展示了在语言跟随率和任务表现方面的比较。左侧显示了不同策略（$ ho_0$-small 和 $ ho_0$）在不同任务（如 Grocery Bagging 和 Table Setting）中的语言跟随率；右侧展示了这些策略的任务表现，提高了中间语言指令的跟随能力。*

    **分析：** $\pi_0$ 的语言遵循准确率显著高于 $\pi_0$-small。这表明 VLM 预训练极大地提升了模型理解和对齐语言指令的能力。此外，$\pi_0$ 能够从高阶策略（人类或 VLM）提供的中间指令中获益，显著提升任务表现，而 $\pi_0$-small 由于语言能力有限，无法有效利用这些中间指令。

### 6.1.3. 学习新灵巧任务 (微调)
评估模型在未见过的任务上的微调能力。任务分为“简单”（与预训练相似）和“困难”（新物体或新动作）。

下图（原文 Figure 10）展示了微调评估的任务：

![Fig. 10: Fine-tuning evaluation tasks: We fine-tune our model to a variety of downstream tasks that are distinct from the tasks seen in pre-training. Our tasks represent a range of similarity from the pre-training tasks, with tasks that are most similar to pre-training (stack bowls and towel folding), a task that introduces an unseen new element (a microwave), and tasks that require new motions and new object types (Franka items in drawer and paper towel replacement).](images/10.jpg)
*该图像是一个示意图，展示了模型在多个下游任务中的微调评估。任务包括叠放碗、折叠毛巾等与预训练任务相似的活动，以及引入新元素的微波炉使用和需要新动作的新物体类型（如 Fraka 项目）。*

实验测试了不同数据量（1 小时 vs 5 小时）下的微调效果。

下图（原文 Figure 11）展示了不同数据量下的微调结果：

![Fig. 11: Fine-tuning with varying amounts of data. $\\pi _ { 0 }$ can learn some easier tasks even with smaller amounts of data, and the pre-trained model often attains a larger improvement over the model trained from scratch.](images/11.jpg)
*该图像是图表，展示了不同算法在多项任务上的微调效果。`heta _0`模型在处理较简单任务时，即使数据较少，也能显著提升表现，且预训练模型普遍优于从头开始训练的模型。*

**分析：** $\pi_0$ 在少量数据（1 小时）下即可学习简单任务，且预训练模型通常比从头训练（Scratch）的模型表现更好，有时提升达 2 倍。对于困难任务，预训练带来的优势更加明显。这验证了预训练 + 微调范式在机器人领域的有效性。

### 6.1.4. 掌握复杂多阶段任务
最后，评估模型在极其复杂、长时间跨度任务上的表现，如折叠多件衣物、组装盒子、打包鸡蛋等。这些任务通常需要 5-20 分钟完成。

下图（原文 Figure 12）展示了这些复杂且时间延续的任务：

![Fig. 12: We evaluate a range of complex and temporally extended tasks. This includes: folding laundry from a bin with a stationary (a) or mobile (b) robot, bussing a real lunch table (c), assembling a box (d), packing eggs into a carton (e), and packing food into a to-go box (f). These tasks require combining dozens of individual behaviors, such as grasping, stacking, folding, and flattening, generalization to a huge variety of object configurations, and complex physical properties, such as deformable objects or flexible cardboard.](images/12.jpg)
*该图像是展示复杂且时间延续的任务的系列图像，包括折叠衣物、清理餐桌、组装盒子等。这些任务需要结合多种单独行为，如抓取、堆叠和折叠，展示了机器人在执行复杂操作时的灵活性和精确性。*

下图（原文 Figure 13）展示了复杂任务的后训练结果：

![Fig. 13: Post-training results on complex tasks in terms of average scores over 10 trials. The full pre-trained $\\pi _ { 0 }$ model attains more than $50 \\%$ of the maximum score across all of the tasks, and typically outperforms the ablations, with especially significant improvements on the hardest tasks.](images/13.jpg)
*该图像是图表，展示了模型在不同任务上的微调效果。上半部分显示了在预训练任务（如洗衣折叠、桌子清理等）上的平均任务进展，下半部分展示了未在预训练中出现的任务（如建箱、打包鸡蛋）的平均任务进展。不同颜色的条形代表了不同的微调策略。*

**分析：** 完整的预训练 + 后训练 $\pi_0$ 模型在所有任务上均取得了超过 50% 的最高分，且在 hardest 任务上优势显著。相比之下，仅预训练（开箱即用）或仅后训练（从头训练）的模型表现较差。这表明多样化预训练提供了恢复能力和通用性，而高质量后训练提供了执行特定复杂策略的能力，两者缺一不可。

# 7. 总结与思考

## 7.1. 结论总结
本文提出了 $\pi_0$，一个基于 VLM 和流匹配技术的通用机器人基础模型。通过在超过 10,000 小时的多样化机器人数据上进行预训练，并结合高质量数据的后训练，$\pi_0$ 在零样本任务、语言指令遵循以及复杂灵巧操作任务上均取得了 state-of-the-art 的性能。论文证明了将互联网规模的语义知识与大规模物理交互数据相结合，是通往通用机器人系统的有效路径。

## 7.2. 局限性与未来工作
作者诚实地指出了当前的局限性：
1.  **数据 composition 理解不足：** 目前尚不清楚何种类型的数据对预训练最有益，以及应如何加权。
2.  **任务可靠性预测：** 并非所有任务都能可靠完成，难以预测需要多少数据才能达到完美性能。
3.  **领域迁移：** 目前主要集中在操纵任务，这种通用性是否能扩展到自动驾驶、导航或足式 locomotion 等领域仍有待验证。

## 7.3. 个人启发与批判
**启发：**
*   **预训练范式的迁移：** 论文成功将 NLP 领域的“预训练 + 对齐/微调”范式迁移到机器人学，证明了大规模多样化数据对于构建鲁棒策略的重要性。
*   **连续动作生成的优势：** 流匹配技术相比自回归离散化，更适合高频、平滑的机器人控制，这为未来的 VLA 模型架构设计提供了新方向。
*   **跨实体学习的潜力：** 单一模型控制多种形态机器人（单臂、双臂、移动）展示了通用策略的巨大潜力，降低了部署成本。

**批判与思考：**
*   **计算成本：** 训练 33 亿参数模型且需要 10,000 小时数据，计算资源门槛极高，这可能限制中小型研究机构的复现和跟进。
*   **安全性问题：** 论文主要关注任务成功率，但对于机器人在失败时的安全性（如不损坏物体、不伤害人类）讨论较少。在现实部署中，安全性比成功率更为关键。
*   **仿真与现实的差距：** 虽然使用了真实机器人数据，但大部分数据是在受控环境中收集的。在完全非结构化、动态变化的真实家庭环境中，模型的泛化能力仍需进一步验证。
*   **推理延迟：** 尽管论文报告了推理时间，但流匹配需要多步积分（10 步），相比单步前馈网络仍有延迟。在极高动态任务中，这可能成为瓶颈。

    总体而言，$\pi_0$ 是机器人基础模型领域的一个重要里程碑，为构建真正通用的具身智能系统奠定了坚实基础。