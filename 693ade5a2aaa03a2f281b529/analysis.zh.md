# 1. 论文基本信息

## 1.1. 标题
**Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**
(中文：扩散策略：通过动作扩散进行视觉-运动策略学习)

论文的核心主题是提出一种名为 <strong>扩散策略 (Diffusion Policy)</strong> 的新方法，用于机器人模仿学习。该方法将机器人的视觉-运动策略（即从视觉输入到电机动作的映射）建模为一个条件扩散过程，从而生成机器人的行为。

## 1.2. 作者
Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song.

作者团队来自多个顶尖学术和研究机构，包括：
*   <strong>哥伦比亚大学 (Columbia University)</strong> (1, 4)
*   <strong>丰田研究院 (Toyota Research Institute)</strong> (2)
*   <strong>麻省理工学院 (MIT)</strong> (3)

    这些作者在机器人学、计算机视觉和机器学习领域，特别是在模仿学习、强化学习和生成模型方面，有着深厚的研究背景。例如，Russ Tedrake 是机器人控制领域的知名学者，Shuran Song 在 3D 视觉和机器人操纵方面有杰出贡献。

## 1.3. 发表期刊/会议
论文发表于 **Robotics: Science and Systems (RSS) 2023**。

RSS 是机器人学领域的顶级国际会议之一，与 ICRA 和 IROS 齐名。该会议以其对论文质量的严格要求和对机器人科学与系统理论基础的重视而闻名。在 RSS 上发表意味着该研究成果具有很高的学术价值和创新性。本文是会议论文 Chi et al. (2023) 的扩展版本。

## 1.4. 发表年份
2023年。预印本 (Preprint) 首次发布于 2023 年 3 月 7 日。

## 1.5. 摘要
本文介绍了一种名为 <strong>扩散策略 (Diffusion Policy)</strong> 的新方法，通过将机器人的视觉-运动策略表示为一个<strong>条件去噪扩散过程 (conditional denoising diffusion process)</strong> 来生成机器人行为。我们在4个不同的机器人操纵基准测试中的12个不同任务上对扩散策略进行了评估，发现它始终优于现有的最先进机器人学习方法，平均性能提升了 **46.9%**。扩散策略学习动作分布的<strong>分数函数梯度 (gradient of the action-distribution score function)</strong>，并在推理时通过一系列<strong>随机朗之万动力学 (stochastic Langevin dynamics)</strong> 步骤，依据该梯度场进行迭代优化。我们发现，扩散公式为机器人策略带来了强大的优势，包括：优雅地处理<strong>多模态动作分布 (multimodal action distributions)</strong>，适用于**高维动作空间**，以及表现出令人印象深刻的**训练稳定性**。为了在实体机器人上完全释放扩散模型在视觉-运动策略学习中的潜力，本文提出了一系列关键技术贡献，包括引入<strong>滚动时域控制 (receding horizon control)</strong>、<strong>视觉条件化 (visual conditioning)</strong> 以及 <strong>时序扩散 Transformer (time-series diffusion transformer)</strong>。我们希望这项工作能激励新一代的策略学习技术，以利用扩散模型强大的生成建模能力。

## 1.6. 原文链接
*   **arXiv 链接:** https://arxiv.org/abs/2303.04137
*   **PDF 链接:** https://arxiv.org/pdf/2303.04137v5.pdf
*   **项目主页:** diffusion-policy.cs.columbia.edu

    本文分析基于 arXiv 上的 v5 版本，这是一个已在 RSS 2023 发表的会议论文的扩展版本。

# 2. 整体概括

## 2.1. 研究背景与动机
**核心问题：** 在机器人模仿学习中，如何从人类演示数据中学习一个能够有效处理现实世界复杂性的策略？

**重要性与挑战：** 模仿学习，即将策略学习看作一个从观测到动作的监督学习问题，是让机器人获得复杂技能的有效途径。然而，这个看似简单的回归任务在实践中充满挑战，主要因为机器人动作预测具有独特性：
1.  <strong>多模态性 (Multimodality):</strong> 对于同一个状态，可能存在多个同样有效的动作或动作序列。例如，绕过一个障碍物可以从左边，也可以从右边。传统方法（如简单的回归）倾向于输出所有可能动作的平均值，导致行为无效或抖动。
2.  <strong>时序相关性 (Sequential Correlation):</strong> 机器人的动作通常是一个序列，需要保持时间上的一致性和平滑性，以避免“近视”规划和不稳定的行为。
3.  <strong>高精度要求 (High Precision):</strong> 许多机器人任务，如工具操作或装配，要求动作具有极高的精度。

<strong>现有研究的空白 (Gap):</strong>
*   <strong>显式策略 (Explicit Policies):</strong> 如高斯混合模型（GMM）或动作离散化，虽然能一定程度上处理多模态，但模型表达能力有限，且对超参数敏感。
*   <strong>隐式策略 (Implicit Policies):</strong> 如基于能量的模型（EBMs），虽然理论上能表示任意复杂的分布，但训练过程依赖于<strong>负采样 (negative sampling)</strong> 来估计一个难以计算的归一化常数，导致训练非常不稳定，难以复现和调优。

**本文的切入点/创新思路：**
本文提出，**不要直接预测动作，也不要学习一个需要负采样的能量函数，而是学习动作分布的“梯度场”**。具体来说，作者将策略建模为一个在动作空间上的**条件去噪扩散过程**。在推理时，策略从一个随机噪声开始，沿着学习到的梯度方向，通过多步迭代“去噪”，最终生成一个高质量的动作（序列）。这个思路巧妙地借鉴了近年来在图像生成领域大获成功的<strong>扩散模型 (Diffusion Models)</strong>，并将其优势引入机器人策略学习。

## 2.2. 核心贡献/主要发现
本文的核心贡献是将扩散模型成功应用于机器人视觉-运动策略学习，并系统性地解决了其中的关键技术挑战，从而实现了SOTA性能。

1.  **提出了 Diffusion Policy：** 首次将机器人策略建模为条件去噪扩散过程，继承了扩散模型的三大优势：
    *   **强大的表达能力：** 能自然地表示任意复杂的多模态动作分布。
    *   **高维输出能力：** 能直接预测高维的动作序列，保证了动作的时序连贯性。
    *   **稳定的训练过程：** 通过学习分数函数（梯度的近似），绕过了隐式模型中不稳定的负采样环节。

2.  **关键技术贡献以适配机器人场景：**
    *   <strong>滚动时域控制 (Receding Horizon Control):</strong> 结合了动作序列预测和闭环控制，让策略既有远见（预测未来动作序列）又具响应性（根据新观测持续重规划）。
    *   <strong>高效的视觉条件化 (Visual Conditioning):</strong> 将视觉观测作为条件输入，而非生成目标的一部分。这使得视觉特征只需提取一次，大大降低了推理延迟，满足了实时控制的需求。
    *   <strong>时序扩散 Transformer (Time-series Diffusion Transformer):</strong> 提出了一种新的基于 Transformer 的扩散网络架构，有效缓解了传统 CNN 架构的过度平滑问题，更适合需要高频动作变化的任务。

3.  **全面的实验验证：** 在 4 个基准（共 15 个任务）上进行了广泛的模拟和真实世界实验，涵盖了不同自由度、不同物体类型（刚性/流体）、单/双手臂等复杂场景。

**主要发现：**
*   **性能全面超越：** Diffusion Policy 在所有测试基准上均显著优于先前的最先进方法（如 IBC, BET, LSTM-GMM），平均成功率提升了 **46.9%**。
*   **位置控制更优：** 与多数工作倾向于使用速度控制不同，本文发现 Diffusion Policy 在<strong>位置控制 (position control)</strong> 模式下表现更佳，因为它能更好地利用位置控制的优势（误差不累积）并克服其劣势（多模态性更强）。
*   **鲁棒性与稳定性：** Diffusion Policy 训练过程稳定，对超参数不敏感，并且在真实机器人上对物理和视觉扰动表现出很强的鲁棒性。

# 3. 预备知识与相关工作

## 3.1. 基础概念

### 3.1.1. 模仿学习 (Imitation Learning, IL)
模仿学习是一种让机器人通过观察和模仿专家（通常是人类）的演示来学习技能的方法。最简单的形式是<strong>行为克隆 (Behavior Cloning, BC)</strong>，它将问题建模为一个监督学习任务：学习一个策略函数 $\pi_\theta(\mathbf{a}_t | \mathbf{o}_t)$，该函数将当前观测 $\mathbf{o}_t$ 映射到专家会采取的动作 $\mathbf{a}_t$。本文的所有实验都基于行为克隆的框架。

### 3.1.2. 能量基模型 (Energy-Based Models, EBMs)
EBMs 是一种生成模型，它不直接定义一个概率密度函数，而是通过一个<strong>能量函数 (Energy Function)</strong> $E_\theta(\mathbf{x})$ 来为每个数据点 $\mathbf{x}$ 分配一个标量能量值。能量越低，表示该数据点出现的概率越高。其概率密度可以表示为：
$$
p_\theta(\mathbf{x}) = \frac{e^{-E_\theta(\mathbf{x})}}{Z(\theta)}
$$
其中 $Z(\theta) = \int e^{-E_\theta(\mathbf{x})} d\mathbf{x}$ 是一个通常难以计算的归一化常数，称为<strong>配分函数 (partition function)</strong>。在策略学习中，EBMs 可以用来表示动作的条件分布 $p_\theta(\mathbf{a} | \mathbf{o})$，其中能量函数为 $E_\theta(\mathbf{o}, \mathbf{a})$。

### 3.1.3. 去噪扩散概率模型 (Denoising Diffusion Probabilistic Models, DDPMs)
DDPMs 是一类强大的深度生成模型。其核心思想包含两个过程：
1.  <strong>前向过程（扩散过程）：</strong> 从一个真实的、干净的数据样本 $\mathbf{x}^0$ 开始，逐步、多次地向其添加少量高斯噪声，直到经过 $K$ 步后，数据完全变成一个纯高斯噪声 $\mathbf{x}^K$。这个过程是固定的，不需要学习。
2.  <strong>反向过程（去噪过程）：</strong> 学习一个神经网络 $\varepsilon_\theta(\mathbf{x}^k, k)$，该网络的目标是预测在第 $k$ 步时添加到干净数据上的噪声。在生成新样本时，从一个随机高斯噪声 $\mathbf{x}^K$ 开始，利用学习到的网络 $\varepsilon_\theta$ 进行 $K$ 次迭代，逐步去除噪声，最终恢复出一个干净的数据样本 $\mathbf{x}^0$。

    DDPMs 的一个重要理论洞见是，训练这个噪声预测网络等价于优化数据分布的<strong>分数函数 (score function)</strong> $\nabla_{\mathbf{x}} \log p(\mathbf{x})$。这使得模型可以学习到数据分布的梯度场，并通过<strong>朗之万动力学 (Langevin dynamics)</strong>（一种基于梯度的随机采样方法）来生成样本。

## 3.2. 前人工作
本文主要与以下几类机器人模仿学习方法进行对比：

1.  <strong>显式策略 (Explicit Policies):</strong>
    *   <strong>高斯混合模型 (Mixture Density Networks, MDN):</strong> 如 `LSTM-GMM` (Mandlekar et al., 2021)，它将策略的输出建模为多个高斯分布的混合，从而显式地表示多模态。但其表达能力有限，且混合成分的数量需要预先指定。
    *   <strong>动作离散化 (Action Discretization):</strong> 如 `Gato`, `RT-1` 等，将连续的动作空间划分为有限个“桶”，把回归问题转化为分类问题。虽然能处理多模态，但在高维动作空间中，所需“桶”的数量会指数级增长，即“维度灾难”。`BET` (Shafiullah et al., 2022) 是一种结合了k-means聚类和Transformer的方法，也属于这一范畴。

2.  <strong>隐式策略 (Implicit Policies):</strong>
    *   <strong>隐式行为克隆 (Implicit Behavioral Cloning, IBC):</strong> (Florence et al., 2021) 是该方向的代表工作。它使用 EBM 来定义动作分布，在推理时通过优化能量函数来寻找最佳动作。理论上，IBC 可以表示任意复杂的多模态分布。然而，其训练依赖于一种名为 **InfoNCE** 的对比损失函数，该函数需要进行**负采样**来近似配分函数。这种近似是不精确的，导致训练过程非常不稳定，性能对超参数和随机种子极为敏感，使得选择最佳模型变得困难。

3.  **基于扩散模型的规划与控制:**
    *   **Diffuser (Janner et al., 2022a):** 将扩散模型用于离线强化学习中的<strong>规划 (planning)</strong>，通过扩散生成一个包含状态和动作的完整轨迹。与 `Diffusion Policy` 不同，`Diffuser` 生成的是开环的轨迹，并且需要一个动力学模型。而 `Diffusion Policy` 是一个闭环的<strong>策略 (policy)</strong>，直接从当前观测生成动作，实时性更强。
    *   **Diffusion-QL (Wang et al., 2022):** 在强化学习背景下，使用扩散模型来表示 Actor-Critic 框架中的策略（Actor），主要在状态空间上进行探索。
    *   **同期工作:** 论文也提到了几篇同期的工作（Pearce et al., 2023; Reuss et al., 2023），它们同样探索了在模仿学习或强化学习中使用扩散模型，但侧重点不同（如采样策略、目标条件化等），且 `Diffusion Policy` 在真实世界机器人的广泛验证和关键技术设计（如滚动时域控制、位置/速度控制对比）上更为深入。

## 3.3. 技术演进
机器人模仿学习的技术演进可以看作是对**动作分布复杂性**的建模能力不断增强的过程：
1.  **单模态时代：** 最初的行为克隆采用简单的神经网络，直接从观测回归到单一动作，假设“一个观测只对应一个最佳动作”。这在简单任务中可行，但在多模态场景下会失效。
2.  **显式多模态时代：** 为了解决多模态问题，研究者引入了 GMM 或动作离散化等方法，试图显式地建模几个不同的动作模式。这些方法比单模态有所改进，但表达能力受限。
3.  **隐式多模态时代：** 以 IBC 为代表，使用 EBM 隐式地定义能量景观，理论上可以表示任意复杂的分布。这是一个巨大的进步，但代价是训练不稳定。
4.  <strong>扩散模型时代（本文）：</strong> `Diffusion Policy` 提出，我们可以保留 EBM 的强大表达能力，同时通过学习分数函数来规避其训练不稳定的问题。这代表了在建模复杂动作分布方面的一个更稳定、更有效的新范式。

## 3.4. 差异化分析
与最相关的 `IBC` 相比，`Diffusion Policy` 的核心区别在于**训练目标和过程**：
*   **IBC:** 学习能量函数 $E_\theta(\mathbf{o}, \mathbf{a})$。训练时需要将正样本（演示动作）与大量随机采样的负样本进行对比，目标是拉低正样本的能量，推高负样本的能量。这个过程依赖负样本的质量和数量，导致训练不稳定。
*   **Diffusion Policy:** 学习噪声预测函数 $\varepsilon_\theta(\mathbf{o}, \mathbf{A}_t^k, k)$，它近似于条件动作分布的对数梯度的负值（即分数函数）。训练时，只需在一个加噪的演示动作上预测所加的噪声，这是一个简单的均方误差回归任务，**完全不需要负采样**。

    因此，`Diffusion Policy` 在保持与 `IBC` 相当（甚至更强）的表达能力的同时，实现了前所未有的训练稳定性。

# 4. 方法论

本节将深入拆解 `Diffusion Policy` 的技术细节，严格遵循原文的公式和逻辑。

## 4.1. 方法原理
`Diffusion Policy` 的核心思想是将生成机器人动作的过程，视为一个从纯噪声中逐步“去噪”以恢复出专家动作的过程。这个去噪过程由一个深度神经网络引导，该网络学习了在给定视觉观测条件下，专家动作分布的梯度场。

在**训练阶段**，模型学习如何从一个被噪声污染的专家动作中预测出原始添加的噪声。
在**推理阶段**，模型从一个完全随机的噪声动作序列开始，利用训练好的噪声预测器，通过迭代式的“减去预测噪声”操作，最终生成一个连贯、有效且符合专家行为模式的动作序列。

## 4.2. 核心方法详解 (逐层深入)

### 4.2.1. 基础：去噪扩散概率模型 (DDPM)
首先，我们回顾标准的 DDPM。它是一个生成模型，其生成过程被建模为一个去噪过程，也称为<strong>随机朗之万动力学 (Stochastic Langevin Dynamics)</strong>。

从一个从高斯噪声中采样得到的 $\mathbf{x}^K$ 开始，DDPM 执行 $K$ 次迭代来生成一系列噪声水平递减的中间结果 $\mathbf{x}^k, \mathbf{x}^{k-1}, \dots, \mathbf{x}^0$，直到最终形成一个无噪声的期望输出 $\mathbf{x}^0$。这个过程遵循以下方程：
$$
\mathbf { x } ^ { k - 1 } = \alpha ( \mathbf { x } ^ { k } - \gamma \varepsilon _ { \theta } ( \mathbf { x } ^ { k } , k ) + \mathcal { N } \big ( 0 , \sigma ^ { 2 } I \big ) )
$$
<strong>公式 (1) 详解:</strong>
*   $\mathbf{x}^k$: 第 $k$ 步的带噪数据。
*   $\mathbf{x}^{k-1}$: 经过一步去噪后，噪声更少的第 `k-1` 步数据。
*   $\varepsilon_\theta(\mathbf{x}^k, k)$: 核心的**噪声预测网络**，其参数为 $\theta$。它的输入是当前的带噪数据 $\mathbf{x}^k$ 和当前的迭代步数 $k$，输出是对添加到原始干净数据上的噪声的预测。
*   $\gamma$: 一个控制去噪步骤大小的系数。
*   $\alpha, \sigma$: <strong>噪声调度 (noise schedule)</strong> 的一部分，它们是随步数 $k$ 变化的函数，用于控制每一步去噪的幅度和添加随机性的强度。
*   $\mathcal{N}(0, \sigma^2 I)$: 在每一步迭代中添加的少量高斯噪声，这使得该过程是随机的，有助于探索和采样。

    这个方程可以被直观地理解为一个**带噪声的梯度下降步骤**。如果我们将能量函数定义为 $E(\mathbf{x})$，那么其梯度下降步骤为：
$$
\mathbf { x } ^ { \prime } = \mathbf { x } - \gamma \nabla E ( \mathbf { x } )
$$
<strong>公式 (2) 详解:</strong>
通过对比公式 (1) 和 (2)，可以发现噪声预测网络 $\varepsilon_\theta(\mathbf{x}, k)$ 实际上是在有效地预测数据分布的**能量梯度场** $\nabla E(\mathbf{x})$。因此，扩散模型的生成过程本质上是在学习到的能量函数的梯度场中进行优化。

DDPM 的训练过程非常简单。我们从数据集中随机抽取一个干净的样本 $\mathbf{x}^0$，随机选择一个去噪步骤 $k$，然后从一个适当方差的高斯分布中采样一个随机噪声 $\varepsilon^k$。我们将这个噪声添加到干净样本上得到 $\mathbf{x}^0 + \varepsilon^k$，然后要求噪声预测网络从这个加噪样本中预测出原始添加的噪声 $\varepsilon^k$。损失函数是一个简单的<strong>均方误差 (Mean Squared Error, MSE)</strong>：
$$
\mathcal { L } = MSE ( \varepsilon ^ { k } , \varepsilon _ { \theta } ( \mathbf { x } ^ { 0 } + \varepsilon ^ { k } , k ) )
$$
<strong>公式 (3) 详解:</strong>
这个损失函数的目标是让网络预测的噪声 $\varepsilon_\theta(\dots)$ 与真实添加的噪声 $\varepsilon^k$ 尽可能接近。Ho et al. (2020) 证明了最小化这个简单的 MSE 损失等价于最小化真实数据分布与模型生成分布之间KL散度的变分下界，从而为这个直观的训练目标提供了坚实的理论基础。

### 4.2.2. 适配机器人策略：Diffusion Policy 的改进
为了将 DDPM 应用于机器人视觉-运动策略学习，论文进行了两个关键的修改，如下图所示。

![Figure 2. Diffusion Policy Overview a) General formulation. At time step $t$ , the policy takes the latest `T _ { o }` steps of observation data `O _ { t }` as input and outputs `T _ { a }` steps of actions `A _ { t }` . b) In the CNN-based Diffusion Policy, FiLM (Feature-wise Linear Modulation) Per l.8conditioning f he servation fe `O _ { t }` isapplied to every convolution layer, channel-wise. Starting from $\\mathbf { A } _ { t } ^ { K }$ drawn from Gaussian noise, the output of noise-prediction network $\\varepsilon _ { \\theta }$ is subtracted, repeating $K$ times to get ${ \\bf A } _ { t } ^ { 0 }$ , the denoised a $\\mathbf { O } _ { t }$ is passed itself and previous action embeddings (causal attention) using the attention mask illustrated.](images/2.jpg)
*该图像是一个示意图，展示了扩散策略的框架。上方部分说明了输入的图像观测序列和预测的动作序列之间的关系，同时突出了扩散政策的核心过程，使用 $ abla E(A_t) $ 来优化动作输出。下方部分则分别介绍了基于CNN和基于Transformer的扩散政策架构，显示了FiLM条件调节和交叉注意力机制的应用。*

1.  <strong>闭环动作序列预测 (Closed-loop action-sequence prediction):</strong>
    *   **问题：** 机器人动作需要时序连贯性，并且能对环境变化做出快速反应。
    *   **方案：** 策略的输出不再是单个动作，而是一个**未来的动作序列**。具体来说，在时间步 $t$，策略接收过去 $T_o$ 步的观测数据 $\mathbf{O}_t$，并预测未来 $T_p$ 步的动作序列。然后，机器人只执行这个序列中的前 $T_a$ 步 ($T_a \le T_p$)，之后再次接收新的观测并重新规划。这种<strong>滚动时域控制 (receding horizon control)</strong> 的方式，既通过预测长序列 ($T_p$) 保证了动作的连贯性和远见，又通过频繁重规划（每 $T_a$ 步）保证了对环境变化的响应性。

2.  <strong>视觉观测条件化 (Visual observation conditioning):</strong>
    *   **问题：** 策略需要根据当前的视觉观测 $\mathbf{O}_t$ 来生成动作 $\mathbf{A}_t$。如何将观测信息融入扩散过程？
    *   **方案：** 将扩散模型修改为**条件生成模型**，使其近似条件分布 $p(\mathbf{A}_t | \mathbf{O}_t)$。这与一些将观测和动作一起建模为联合分布 $p(\mathbf{A}_t, \mathbf{O}_t)$ 的工作不同。只对动作进行扩散和去噪，而将观测作为固定的条件输入。
    *   **优势：** 这样做极大地提高了推理效率。因为在整个去噪迭代过程中，视觉特征只需要计算一次，然后重复使用。如果对观测和动作联合建模，则每一步去噪都需要重新推理视觉部分，计算成本会高得多。

        条件化的去噪过程由以下公式描述：
    $$
    \mathbf { A } _ { t } ^ { k - 1 } = \alpha ( \mathbf { A } _ { t } ^ { k } - \gamma \varepsilon _ { \theta } ( \mathbf { O } _ { t } , \mathbf { A } _ { t } ^ { k } , k ) + \mathcal { N } \big ( 0 , \sigma ^ { 2 } I \big ) )
    $$
    <strong>公式 (4) 详解:</strong>
    *   $\mathbf{A}_t^k$: 在时间步 $t$、去噪迭代第 $k$ 步的**动作序列**。
    *   $\varepsilon_\theta(\mathbf{O}_t, \mathbf{A}_t^k, k)$: **条件噪声预测网络**。与公式 (1) 不同，它额外接收当前的**观测数据** $\mathbf{O}_t$ 作为条件输入。

        相应的，训练损失函数也变为条件化的形式：
    $$
    \mathcal { L } = MSE ( \boldsymbol { \varepsilon } ^ { k } , \boldsymbol { \varepsilon } _ { \theta } ( \mathbf { O } _ { t } , \mathbf { A } _ { t } ^ { 0 } + \boldsymbol { \varepsilon } ^ { k } , k ) )
    $$
    <strong>公式 (5) 详解:</strong>
    *   $\mathbf{A}_t^0$: 从数据集中采样的专家演示中的干净动作序列。
    *   $\mathbf{O}_t$: 与该动作序列对应的观测数据。
    *   训练目标是让网络在给定观测 $\mathbf{O}_t$ 的条件下，从加噪的动作序列 $\mathbf{A}_t^0 + \varepsilon^k$ 中预测出噪声 $\varepsilon^k$。

### 4.2.3. 关键设计决策 (Key Design Decisions)

1.  <strong>网络架构选项 (Network Architecture Options):</strong>
    *   <strong>基于CNN的扩散策略 (CNN-based Diffusion Policy):</strong> 采用一维时序卷积网络。观测特征 $\mathbf{O}_t$ 通过 **FiLM (Feature-wise Linear Modulation)** 层注入到每个卷积层中，实现条件化。这种架构是大多数任务的稳健基准。
    *   <strong>时序扩散 Transformer (Time-series Diffusion Transformer):</strong> 针对 CNN 在处理高频变化动作（如速度控制）时的过度平滑问题，作者提出了一种基于 Transformer 的新架构。它将加噪的动作序列作为 Transformer 解码器的输入词元 (token)，并将观测特征作为交叉注意力的 `key` 和 `value` 输入，从而预测去噪结果。

2.  <strong>视觉编码器 (Visual Encoder):</strong>
    使用一个标准的 `ResNet-18` 作为图像编码器，但做了两个关键修改：
    *   用 <strong>空间 Softmax 池化 (spatial softmax pooling)</strong> 替换全局平均池化，以保留图像中的空间信息。
    *   用 <strong>组归一化 (GroupNorm)</strong> 替换批归一化 (BatchNorm)，以确保与扩散模型中常用的指数移动平均（EMA）一起使用时训练的稳定性。

3.  <strong>推理加速 (Accelerating Inference):</strong>
    为了满足机器人实时控制（约10Hz）的需求，论文采用了 **DDIM (Denoising Diffusion Implicit Models)** 加速技术。DDIM 允许在推理时使用比训练时少得多的去噪步骤（例如，训练用100步，推理用10步），从而在牺牲极少性能的情况下，将推理延迟降低到可接受的范围内（如0.1秒）。

### 4.2.4. 与控制理论的联系
论文在第 4.5 节中探讨了 `Diffusion Policy` 在一个简单的线性系统中的行为，以建立与传统控制理论的联系。

考虑一个线性动态系统：
$$
\mathbf { s } _ { t + 1 } = \mathbf { A } \mathbf { s } _ { t } + \mathbf { B } \mathbf { a } _ { t } + \mathbf { w } _ { t } , \qquad \mathbf { w } _ { t } \sim \mathcal { N } ( 0 , \Sigma _ { w } )
$$
假设专家演示来自于一个线性反馈控制器 $\mathbf{a}_t = -\mathbf{K}\mathbf{s}_t$。如果 `Diffusion Policy` 的预测时域 $T_p=1$，其训练目标是最小化：
$$
\mathcal { L } = MSE ( \boldsymbol { \varepsilon } ^ { k } , \boldsymbol { \varepsilon } _ { \theta } ( \mathbf { s } _ { t } , - \mathbf { K } \mathbf { s } _ { t } + \boldsymbol { \varepsilon } ^ { k } , k ) )
$$
可以证明，最优的噪声预测器 $\varepsilon_\theta$ 是：
$$
\varepsilon _ { \theta } ( \mathbf { s } , \mathbf { a } , k ) = \frac { 1 } { \sigma _ { k } } [ \mathbf { a } + \mathbf { K } \mathbf { s } ]
$$
在推理时，DDIM 采样过程将收敛到全局最小值，即 $\mathbf{a} = -\mathbf{K}\mathbf{s}$。这表明，`Diffusion Policy` 能够完美地克隆线性控制器。

更有趣的是，当预测时域 $T_p > 1$ 时，为了预测未来的动作 $\mathbf{a}_{t+t'}$，最优的去噪器会学习到 $\mathbf{a}_{t+t'} = -\mathbf{K}(\mathbf{A} - \mathbf{B}\mathbf{K})^{t'}\mathbf{s}_t$。这揭示了一个深刻的洞见：<strong>为了预测未来的动作，模仿学习模型必须隐式地学习到一个（与任务相关的）系统动力学模型</strong>。

# 5. 实验设置

## 5.1. 数据集
论文在4个标准机器人操纵基准测试的15个任务上进行了系统评估，涵盖了模拟和真实世界。

以下是原文 Table 3 的内容，总结了各个任务的属性：

<table>
<thead>
<tr>
<th>Task</th>
<th># Rob</th>
<th># Obj</th>
<th>ActD</th>
<th>#PH</th>
<th>#MH</th>
<th>Steps</th>
<th>Img?</th>
<th>HiPrec</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="9" style="text-align:center; font-weight:bold;">Simulation Benchmark</td>
</tr>
<tr>
<td>Lift</td>
<td>1</td>
<td>1</td>
<td>7</td>
<td>200</td>
<td>300</td>
<td>400</td>
<td>Yes</td>
<td>No</td>
</tr>
<tr>
<td>Can</td>
<td>1</td>
<td>1</td>
<td>7</td>
<td>200</td>
<td>300</td>
<td>400</td>
<td>Yes</td>
<td>No</td>
</tr>
<tr>
<td>Square</td>
<td>1</td>
<td>1</td>
<td>7</td>
<td>200</td>
<td>300</td>
<td>400</td>
<td>Yes</td>
<td>Yes</td>
</tr>
<tr>
<td>Transport</td>
<td>2</td>
<td>3</td>
<td>14</td>
<td>200</td>
<td>300</td>
<td>700</td>
<td>Yes</td>
<td>No</td>
</tr>
<tr>
<td>ToolHang</td>
<td>1</td>
<td>2</td>
<td>7</td>
<td>200</td>
<td>0</td>
<td>700</td>
<td>Yes</td>
<td>Yes</td>
</tr>
<tr>
<td>Push-T</td>
<td>1</td>
<td>1</td>
<td>2</td>
<td>200</td>
<td>0</td>
<td>300</td>
<td>Yes</td>
<td>Yes</td>
</tr>
<tr>
<td>BlockPush</td>
<td>1</td>
<td>2</td>
<td>2</td>
<td>1000</td>
<td>0</td>
<td>350</td>
<td>No</td>
<td>No</td>
</tr>
<tr>
<td>Kitchen</td>
<td>1</td>
<td>7</td>
<td>9</td>
<td>656</td>
<td>0</td>
<td>280</td>
<td>No</td>
<td>No</td>
</tr>
<tr>
<td colspan="9" style="text-align:center; font-weight:bold;">Realworld Benchmark</td>
</tr>
<tr>
<td>Push-T</td>
<td>1</td>
<td>1</td>
<td>2</td>
<td>136</td>
<td>0</td>
<td>600</td>
<td>Yes</td>
<td>Yes</td>
</tr>
<tr>
<td>6DoF Pour</td>
<td>1</td>
<td>liquid</td>
<td>6</td>
<td>90</td>
<td>0</td>
<td>600</td>
<td>Yes</td>
<td>No</td>
</tr>
<tr>
<td>Peri Spread</td>
<td>1</td>
<td>liquid</td>
<td>6</td>
<td>90</td>
<td>0</td>
<td>600</td>
<td>Yes</td>
<td>No</td>
</tr>
<tr>
<td>Mug Flip</td>
<td>1</td>
<td>1</td>
<td>7</td>
<td>250</td>
<td>0</td>
<td>600</td>
<td>Yes</td>
<td>No</td>
</tr>
</tbody>
</table>

*   **Robomimic (Mandlekar et al., 2021):** 一个大规模的模仿学习基准，包含 `Lift`、`Can`、`Square`、`Transport` 和 `ToolHang` 等任务。它提供了两种数据集：`proficient human (PH)` (熟练人类演示) 和 `multi-human (MH)` (混合水平人类演示)。
*   **Push-T (Florence et al., 2021):** 一个需要精确接触和复杂物理交互的平面推动任务，用于测试策略的精度和对多模态的建模能力。
*   **Multimodal Block Pushing (Shafiullah et al., 2022):** 一个专门设计用于测试长时程多模态的任务，要求机器人以任意顺序将两个积木推到两个目标区域。
*   **Franka Kitchen (Gupta et al., 2019):** 一个复杂的厨房环境，包含多个可交互对象，用于评估策略在多任务和长时程场景下的能力。
*   <strong>真实世界任务 (Real-world Tasks):</strong> 包括 `Push-T` 的真实版本、`Sauce Pouring` (倒酱汁)、`Sauce Spreading` (抹酱汁)、`Mug Flipping` (翻杯子) 以及一系列双臂协作任务，如 `Egg Beater` (打蛋)、`Mat Unrolling` (展开垫子) 和 `Shirt Folding` (叠衣服)。

## 5.2. 评估指标

1.  <strong>成功率 (Success Rate):</strong>
    *   **概念定义:** 这是最主要的评估指标，衡量策略在多次尝试中成功完成任务的百分比。一个任务是否成功通常由一个预定义的二进制条件判断（例如，物体是否到达目标位置）。
    *   **数学公式:**
        $$
        \text{Success Rate} = \frac{\sum_{i=1}^{N} \mathbb{I}(\text{Trial}_i \text{ is successful})}{N}
        $$
    *   **符号解释:**
        *   $N$: 总的评估试验次数。
        *   $\text{Trial}_i$: 第 $i$ 次试验。
        *   $\mathbb{I}(\cdot)$: 指示函数，当条件为真时取值为1，否则为0。

2.  <strong>交并比 (Intersection over Union, IoU):</strong>
    *   **概念定义:** 主要用于评估物体最终位置或区域覆盖的准确性，如 `Push-T` 和 `Sauce Pouring` 任务。它计算的是模型预测的区域（如物体最终的掩码或酱汁覆盖的区域）与目标区域（真值）之间的重叠程度。值域为 [0, 1]，越接近1表示越准确。
    *   **数学公式:**
        $$
        \text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}
        $$
    *   **符号解释:**
        *   $A$: 模型预测的区域（例如，像素集合）。
        *   $B$: 目标真值区域。
        *   $|A \cap B|$: 预测区域与目标区域的交集面积。
        *   $|A \cup B|$: 预测区域与目标区域的并集面积。

3.  <strong>覆盖率 (Coverage):</strong>
    *   **概念定义:** 在 `Sauce Spreading` (抹酱汁) 任务中使用，衡量酱汁在披萨饼皮上覆盖的面积比例。
    *   **数学公式:**
        $$
        \text{Coverage} = \frac{\text{Area}_{\text{sauce}}}{\text{Area}_{\text{dough}}}
        $$
    *   **符号解释:**
        *   $\text{Area}_{\text{sauce}}$: 酱汁覆盖的像素面积。
        *   $\text{Area}_{\text{dough}}$: 整个披萨饼皮的像素面积。

## 5.3. 对比基线
论文将 `Diffusion Policy` 与以下三个具有代表性的最先进（SOTA）方法进行了比较：
*   **LSTM-GMM (Mandlekar et al., 2021):** 一种**显式多模态**策略，使用 LSTM 捕捉时序信息，并用高斯混合模型（GMM）输出多模态动作分布。
*   **IBC (Florence et al., 2021):** 一种**隐式多模态**策略，使用能量基模型（EBM）来表示动作分布，是 `Diffusion Policy` 最直接的对标方法。
*   **BET (Shafiullah et al., 2022):** 一种基于 Transformer 和动作离散化的方法，通过将动作聚类并将策略学习转化为分类问题来处理多模态。

    这些基线覆盖了当前模仿学习领域处理多模态问题的主要技术路线，使得比较非常全面。

# 6. 实验结果与分析

## 6.1. 核心结果分析

### 6.1.1. 模拟环境实验结果
`Diffusion Policy` 在所有模拟任务中均显著优于基线方法。

以下是原文 Table 1 的结果，展示了在 `Robomimic` 和 `Push-T` 基准上，使用<strong>状态 (state)</strong> 作为输入的性能对比。

<table>
<thead>
<tr>
<th rowspan="2"></th>
<th colspan="2">Lift</th>
<th colspan="2">Can</th>
<th colspan="2">Square</th>
<th colspan="2">Transport</th>
<th rowspan="2">ToolHang ph</th>
<th rowspan="2">Push-T ph</th>
</tr>
<tr>
<th>ph</th>
<th>mh</th>
<th>ph</th>
<th>mh</th>
<th>ph</th>
<th>mh</th>
<th>ph</th>
<th>mh</th>
</tr>
</thead>
<tbody>
<tr>
<td>LSTM-GMM</td>
<td>1.00/0.96</td>
<td>1.00/0.93</td>
<td>1.00/0.91</td>
<td>1.00/0.81</td>
<td>0.95/0.73</td>
<td>0.86/0.59</td>
<td>0.76/0.47</td>
<td>0.62/0.20</td>
<td>0.67/0.31</td>
<td>0.67/0.61</td>
</tr>
<tr>
<td>IBC</td>
<td>0.79/0.41</td>
<td>0.15/0.02</td>
<td>0.00/0.00</td>
<td>0.01/0.01</td>
<td>0.00/0.00</td>
<td>0.00/0.00</td>
<td>0.00/0.00</td>
<td>0.00/0.00</td>
<td>0.00/0.00</td>
<td>0.90/0.84</td>
</tr>
<tr>
<td>BET</td>
<td>1.00/0.96</td>
<td>1.00/0.99</td>
<td>1.00/0.89</td>
<td>1.00/0.90</td>
<td>0.76/0.52</td>
<td>0.68/0.43</td>
<td>0.38/0.14</td>
<td>0.21/0.06</td>
<td>0.58/0.20</td>
<td>0.79/0.70</td>
</tr>
<tr>
<td>DiffusionPolicy-C</td>
<td>1.00/0.98</td>
<td>1.00/0.97</td>
<td>1.00/0.96</td>
<td>1.00/0.96</td>
<td>1.00/0.93</td>
<td>0.97/0.82</td>
<td>0.94/0.82</td>
<td>0.68/0.46</td>
<td>0.50/0.30</td>
<td>0.95/0.91</td>
</tr>
<tr>
<td>DiffusionPolicy-T</td>
<td>1.00/1.00</td>
<td>1.00/1.00</td>
<td>1.00/1.00</td>
<td>1.00/0.94</td>
<td>1.00/0.89</td>
<td>0.95/0.81</td>
<td>1.00/0.84</td>
<td>0.62/0.35</td>
<td>1.00/0.87</td>
<td>0.95/0.79</td>
</tr>
</tbody>
</table>

以下是原文 Table 2 的结果，展示了使用<strong>图像 (image)</strong> 作为输入的性能对比。

<table>
<thead>
<tr>
<th rowspan="2"></th>
<th colspan="2">Lift</th>
<th colspan="2">Can</th>
<th colspan="2">Square</th>
<th colspan="2">Transport</th>
<th rowspan="2">ToolHang ph</th>
<th rowspan="2">Push-T ph</th>
</tr>
<tr>
<th>ph</th>
<th>mh</th>
<th>ph</th>
<th>mh</th>
<th>ph</th>
<th>mh</th>
<th>ph</th>
<th>mh</th>
</tr>
</thead>
<tbody>
<tr>
<td>LSTM-GMM</td>
<td>1.00/0.96</td>
<td>1.00/0.95</td>
<td>1.00/0.88</td>
<td>0.98/0.90</td>
<td>0.82/0.59</td>
<td>0.64/0.38</td>
<td>0.88/0.62</td>
<td>0.44/0.24</td>
<td>0.68/0.49</td>
<td>0.69/0.54</td>
</tr>
<tr>
<td>IBC</td>
<td>0.94/0.73</td>
<td>0.39/0.05</td>
<td>0.08/0.01</td>
<td>0.00/0.00</td>
<td>0.03/0.00</td>
<td>0.00/0.00</td>
<td>0.00/0.00</td>
<td>0.00/0.00</td>
<td>0.00/0.00</td>
<td>0.75/0.64</td>
</tr>
<tr>
<td>DiffusionPolicy-C</td>
<td>1.00/1.00</td>
<td>1.00/1.00</td>
<td>1.00/0.97</td>
<td>1.00/0.96</td>
<td>0.98/0.92</td>
<td>0.98/0.84</td>
<td>1.00/0.93</td>
<td>0.89/0.69</td>
<td>0.95/0.73</td>
<td>0.91/0.84</td>
</tr>
<tr>
<td>DiffusionPolicy-T</td>
<td>1.00/1.00</td>
<td>1.00/0.99</td>
<td>1.00/0.98</td>
<td>1.00/0.98</td>
<td>1.00/0.90</td>
<td>0.94/0.80</td>
<td>0.98/0.81</td>
<td>0.73/0.50</td>
<td>0.76/0.47</td>
<td>0.78/0.66</td>
</tr>
</tbody>
</table>

**关键发现：**
*   **一致的优势：** 无论是在简单任务（如 `Lift`）还是复杂任务（如 `Transport`, `ToolHang`），无论是使用状态还是图像输入，`Diffusion Policy`（包括 CNN 和 Transformer 版本）的性能都等于或显著优于所有基线。
*   **对付噪声数据：** 在 `mh` (混合水平) 数据集上，性能差距尤为明显。这表明 `Diffusion Policy` 对次优或有噪声的演示数据具有更强的鲁棒性。
*   **IBC 的不稳定性：** `IBC` 在很多任务上性能几乎为零，这印证了其训练不稳定的问题。而 `Diffusion Policy` 的训练过程非常稳定，如下图（原文 Figure 6）所示，其评估成功率平稳上升，而 IBC 则剧烈震荡。

    ![Figure 6. Training Stability. Left: IBC fails to infer training actions with increasing accuracy despite smoothly decreasing training loss for energy function. Right: IBC's evaluation success rate oscillates, making checkpoint selection difficult (evaluated using policy rollouts in simulation).](images/6.jpg)
    *该图像是图表，展示了Diffusion Policy与IBC在Real PushT Img和Sim PushT State任务上的训练表现。左侧显示了训练动作预测的均方误差（MSE）随训练轮次的变化，Diffusion Policy在训练损失上相比IBC表现更优；右侧则展示了在Sim PushT State任务中，成功率随训练轮次的变化，Diffusion Policy的成功率维持在较高水平。*

### 6.1.2. 多模态行为分析
`Diffusion Policy` 能够优雅地处理两种类型的多模态：

1.  <strong>短时程多模态 (Short-horizon multimodality):</strong> 如下图（原文 Figure 3）所示，在 `Push-T` 任务中，当机械臂需要推动 T 形块时，可以从左侧或右侧接近。`Diffusion Policy` 能够学习到这两种模式，并在单次执行中稳定地选择其中一种。相比之下，`LSTM-GMM` 和 `IBC` 倾向于偏向一种模式，而 `BET` 则在两种模式间抖动，无法做出决策。

    ![Figure 3. Multimodal behavior. At the given state, the end-effector (blue) can either go left or right to push the block. Diffusion Policy learns both modes and commits to only one mode within each rollout. In contrast, both LSTM-GMM Mandlekar et al. (2021) and IBC Florence et al. (2021) are biased toward one mode, while BET Shafiullah et al. (2022) fails to commit to a single mode due to its lack of temporal action consistency. Actions generated by rolling out 40 steps for the best-performing checkpoint.](images/3.jpg)
    *该图像是一个示意图，展示了Diffusion Policy与其他方法（LSTM-GMM、BET 和 IBC）在机器人行为生成中的轨迹对比。在相同状态下，Diffusion Policy展示了更高的多模态行为能力，而其他方法则对行为路径有明显偏倚或缺乏时间一致性。*

2.  <strong>长时程多模态 (Long-horizon multimodality):</strong> 在 `Block Push` （任意顺序推两个积木）和 `Kitchen` （任意顺序完成多个子任务）任务中，`Diffusion Policy` 的性能远超基线。这表明它能够处理由于子任务顺序不同而导致的全局性、长时程的多模态问题。

### 6.1.3. 真实世界实验结果
在真实机器人上的表现进一步证实了 `Diffusion Policy` 的有效性和鲁棒性。

*   **真实 Push-T 任务:** `Diffusion Policy` 取得了 **95%** 的成功率，接近人类水平，而 `IBC` 和 `LSTM-GMM` 的成功率分别为 0% 和 20%。基线方法的主要失败原因是在任务阶段转换时（例如，推完物块后移开），由于多模态性增强而无法做出正确决策。

*   <strong>酱汁处理任务 (Sauce Manipulation):</strong> 在 `Pouring` (倒酱汁) 和 `Spreading` (抹酱汁) 任务中，`Diffusion Policy` 同样取得了接近人类的性能。这些任务涉及与流体等非刚性物体的交互以及周期性动作，对策略的精度和鲁棒性要求极高。`LSTM-GMM` 在这些任务上完全失败。

*   **抗扰动能力:** 如下图（原文 Figure 8）所示，在真实 `Push-T` 任务中，即使在摄像头被遮挡、或 T 形块被中途移动的情况下，`Diffusion Policy` 也能迅速做出反应，重新规划路径并完成任务，甚至展现出演示数据中从未出现过的纠错行为。

    ![该图像是实验设置的示意图，展示了一台机器人系统的多个摄像头布局。图(a)显示了一个顶部向下的相机用于评估，以及前置和腕部相机的位置。图(b)展示了机器人在执行任务时的动作步骤，包括抓取和移动红色“T”形物体的关键阶段。图(c)则展示了“T”形物体被成功放置后的状态。这些视觉资源支持了研究论文中关于机器人操作和政策学习的讨论。](images/8.jpg)
    *该图像是实验设置的示意图，展示了一台机器人系统的多个摄像头布局。图(a)显示了一个顶部向下的相机用于评估，以及前置和腕部相机的位置。图(b)展示了机器人在执行任务时的动作步骤，包括抓取和移动红色“T”形物体的关键阶段。图(c)则展示了“T”形物体被成功放置后的状态。这些视觉资源支持了研究论文中关于机器人操作和政策学习的讨论。*

*   **双臂协作任务:** 在更具挑战性的双臂任务如 `Egg Beater` (打蛋)、`Mat Unrolling` (展开垫子) 和 `Shirt Folding` (叠衣服) 中，`Diffusion Policy` 无需特殊调参即可直接应用，并取得了 **55%-75%** 的高成功率，展示了其强大的泛化能力和可扩展性。

## 6.2. 消融实验/参数分析

### 6.2.1. 控制模式：位置 vs. 速度
一个令人惊讶但非常重要的发现是 `Diffusion Policy` 与<strong>位置控制 (position control)</strong> 的协同作用。大多数模仿学习工作偏爱速度控制，因为位置目标的多模态问题更严重。然而，如下图（原文 Figure 4）所示，当从速度控制切换到位置控制时，基线方法的性能普遍下降，而 `Diffusion Policy` 的性能却显著提升。

![Figure 4. Velocity v.s. Position Control. The performance difference when switching from velocity to position control. While both BCRNN and BET performance decrease, Diffusion Policy is able to leverage the advantage of position and improve its performance.](images/4.jpg)

**原因推测：**
1.  `Diffusion Policy` 强大的多模态建模能力使其不惧怕位置控制中更强的多模态性。
2.  位置控制的误差不会像速度控制那样随时间累积，这与 `Diffusion Policy` 预测动作**序列**的能力相得益彰。

### 6.2.2. 动作序列长度与延迟
如下图（原文 Figure 5）所示的消融研究揭示了：
*   <strong>动作执行时域 ($T_a$) 的权衡：</strong> 动作执行时域（图中称 action horizon）并非越长或越短越好。太短（如1）无法保证时序连贯性，太长则导致策略对环境变化反应迟钝。实验发现在大多数任务中，$T_a=8$ 是一个最佳的平衡点。
*   **对延迟的鲁棒性：** 由于采用了滚动时域控制和位置控制，`Diffusion Policy` 对系统延迟（从观测到执行的延迟）表现出很强的鲁棒性，在长达4个时间步的延迟下性能几乎不受影响。

    ![Figure 5. Diffusion Policy Ablation Study. Change (difference) in success rate relative to the maximum for each task is shown on the Y-axis. Left: trade-off between temporal consistency and responsiveness when selecting the action horizon. Right: Diffusion Policy with position control is robust against latency. Latency is defined as the number of steps between the last frame of observations to the first action that can be executed.](images/5.jpg)
    *该图像是图表，展示了Diffusion Policy的消融研究中，动作视野和延迟鲁棒性对成功率的相对影响。在左侧图中，随着动作视野步骤的增加，成功率的变化表现出不同的趋势；右侧图显示了在不同延迟步骤下的性能变化。*

### 6.2.3. 视觉编码器选择
以下是原文 Table 5 的结果，比较了不同视觉编码器架构和训练策略的性能。

<table>
<thead>
<tr>
<th>Architecture & Pretrain Dataset</th>
<th>From Scratch</th>
<th>Pretrained frozen</th>
<th>Pretrained finetuning</th>
</tr>
</thead>
<tbody>
<tr>
<td>Resnet18 (in21)</td>
<td>0.94</td>
<td>0.40</td>
<td>0.92</td>
</tr>
<tr>
<td>Resnet34 (in21)</td>
<td>0.92</td>
<td>0.58</td>
<td>0.94</td>
</tr>
<tr>
<td>ViT-base (clip)</td>
<td>0.22</td>
<td>0.70</td>
<td>0.98</td>
</tr>
</tbody>
</table>

**分析：**
*   **从零训练 vs. 预训练:** 从零开始训练 `ViT` 效果很差，表明在数据量有限的情况下，`ViT` 需要预训练。
*   **冻结 vs. 微调:** 使用冻结的预训练编码器性能不佳，说明通用的视觉特征（如 ImageNet 或 CLIP）与机器人操纵任务所需的特定特征存在偏差。
*   **最佳策略:** <strong>微调 (finetuning)</strong> 预训练的视觉编码器是最佳策略，特别是微调在 CLIP 数据集上预训练的 `ViT-B/16` 取得了最高的成功率 (98%)。这表明利用大规模预训练模型的知识，并针对下游任务进行微调，是提升视觉-运动策略性能的有效途径。

# 7. 总结与思考

## 7.1. 结论总结
这篇论文成功地将扩散模型引入机器人视觉-运动策略学习领域，提出了 `Diffusion Policy`，一个性能卓越且训练稳定的新范式。
*   **核心贡献：** `Diffusion Policy` 通过将策略建模为条件去噪扩散过程，继承了扩散模型处理高维多模态分布的强大能力，同时通过学习分数函数梯度避免了传统隐式模型不稳定的训练问题。
*   **关键设计：** 论文提出的一系列关键技术，包括滚动时域控制、高效的视觉条件化和时序 Transformer 架构，是成功将扩散模型应用于真实机器人系统的关键。
*   **实验结果：** 在涵盖模拟和真实世界、单臂和双臂的15个复杂操纵任务上的全面评估，有力地证明了 `Diffusion Policy` 相对于现有最先进方法的巨大优势（平均性能提升46.9%），并展现了其在真实世界中的鲁棒性和泛化能力。
*   **重要洞见：** 论文还揭示了几个重要的设计选择，例如**位置控制**优于速度控制的惊人发现，这挑战了该领域的普遍认知，并为未来的研究提供了新的方向。

    总而言之，这项工作不仅提出了一个强大的新算法，更重要的是，它揭示了策略表示的**结构本身**是行为克隆性能的一个重要瓶颈，并指明了一条利用现代生成模型强大能力的康庄大道。

## 7.2. 局限性与未来工作
作者在论文中指出了以下局限性和未来研究方向：
*   **数据依赖性：** 作为一种行为克隆方法，`Diffusion Policy` 的性能仍然受限于演示数据的质量和数量。当演示数据不足或质量低下时，其性能会下降。未来的一个方向是将其与<strong>强化学习 (Reinforcement Learning)</strong> 相结合，利用次优数据或通过试错来进一步提升策略。
*   **计算成本与延迟：** 尽管使用了 DDIM 加速，但扩散模型的迭代式推理过程仍然比单次前向传播的显式策略（如 LSTM-GMM）计算成本更高、延迟更大。这可能限制其在需要极高控制频率（> 10-20Hz）的任务中的应用。未来可以探索更先进的扩散模型加速技术，如一致性模型 (Consistency Models) 或新的采样器，来进一步降低推理延迟。

## 7.3. 个人启发与批判
这篇论文给我带来了深刻的启发，也引发了一些批判性思考：

**启发：**
1.  **范式迁移的力量：** 本文是将在一个领域（图像生成）取得巨大成功的技术范式（扩散模型）巧妙迁移到另一个领域（机器人控制）的典范。它告诉我们，很多看似不相关领域的突破，可能为我们自己领域内的“老大难”问题提供了全新的解决方案。
2.  <strong>“学习什么”</strong>比“如何学习”更重要： 传统的模仿学习方法纠结于如何用有限的模型（如GMM）去拟合复杂的多模态分布，而 `Diffusion Policy` 改变了游戏规则——它不去直接拟合分布，而是去学习分布的**梯度场**。这种对学习目标的重新定义，从根本上解决了训练不稳定的问题。这启发我们，在遇到瓶颈时，或许应该跳出优化算法的细节，重新思考我们到底应该让模型学习什么。
3.  **系统性思维的重要性：** 论文的成功不仅仅在于提出了一个算法，更在于系统性地解决了将其落地到真实机器人上的一系列工程和理论问题，如控制模式选择（位置 vs. 速度）、实时性（DDIM加速）、长时程规划（滚动时域控制）等。这体现了顶尖机器人研究中理论创新与系统工程紧密结合的特点。

**批判性思考：**
1.  **能量消耗与可持续性：** 扩散模型的训练和推理成本相对较高。虽然论文通过 DDIM 进行了优化，但与更轻量级的模型相比，其在边缘设备或能耗受限的机器人平台上的部署仍面临挑战。随着模型规模的扩大，计算资源的消耗和碳足迹问题值得关注。
2.  **可解释性与安全性：** 作为一个深度生成模型，`Diffusion Policy` 的决策过程是一个“黑箱”。虽然其行为在实验中表现鲁棒，但在安全攸关的场景下，我们很难解释它为什么会生成某个特定的动作序列。如果模型在未见过的场景下产生灾难性的失败，诊断和修复将非常困难。这提示了在追求性能的同时，对模型可解释性和安全验证的研究也至关重要。
3.  **对“多模态”的进一步思考：** 论文成功地解决了由多种可行路径或任务顺序引起的多模态问题。然而，在现实中，这些“模态”可能并非同等最优。有些演示可能只是次优的，或者是在特定情境下的无奈之举。`Diffusion Policy` 目前似乎平等地对待所有模式。未来的研究或许可以探索如何让模型在学习多模态的同时，还能隐式地学习到不同模式的“偏好”或“优劣”，从而在推理时能更倾向于选择最优或最高效的行为模式。