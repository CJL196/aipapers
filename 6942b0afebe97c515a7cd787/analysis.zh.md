# 1. 论文基本信息

## 1.1. 标题
Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion (扩散强制：当下一词元预测遇见全序列扩散)

这篇论文的核心主题是提出一种名为<strong>扩散强制 (Diffusion Forcing)</strong> 的新型训练范式，旨在融合<strong>下一词元预测 (Next-token Prediction)</strong> 模型和<strong>全序列扩散 (Full-Sequence Diffusion)</strong> 模型的优点。

## 1.2. 作者
*   **Boyuan Chen:** MIT CSAIL (麻省理工学院计算机科学与人工智能实验室)
*   **Diego Marti Monso:** Technical University of Munich (慕尼黑工业大学)
*   **Yilun Du:** MIT CSAIL
*   **Max Simchowitz:** MIT CSAIL
*   **Russ Tedrake:** MIT CSAIL
*   **Vincent Sitzmann:** MIT CSAIL

    作者团队主要来自全球顶尖的人工智能研究机构 MIT CSAIL，他们在机器人学、计算机视觉和生成模型领域有着深厚的研究背景。

## 1.3. 发表期刊/会议
论文目前作为预印本 (Pre-print) 发布于 **arXiv**。arXiv 是一个开放获取的学术论文存档网站，用于在正式同行评审和发表前快速分享研究成果。虽然未在顶级会议（如 NeurIPS, ICML, ICLR）上发表，但其作者背景和研究内容表明了其具有较高的学术水准。

## 1.4. 发表年份
2024年

## 1.5. 摘要
本文提出了一种名为 <strong>扩散强制 (Diffusion Forcing, DF)</strong> 的新训练范式。在该范式中，一个扩散模型被训练用于对一组具有**独立噪声水平**的词元 (token) 进行去噪。作者将 DF 应用于序列生成建模，通过训练一个<strong>因果 (causal)</strong> 的下一词元预测模型来生成一个或多个未来词元，而无需完全扩散过去的词元。

研究表明，该方法结合了下一词元预测模型的优点（如可变长度生成）和全序列扩散模型的优点（如引导采样到期望轨迹的能力）。此外，该方法还提供了一系列额外能力，包括：
1.  对于连续词元序列（如视频），能够生成远超训练长度的序列，而基线方法会发散。
2.  提出了新的采样和引导方案，这些方案独特地利用了 DF 的可变视野和因果架构，在决策和规划任务中取得了显著的性能提升。

    除了经验上的成功，该方法在理论上被证明可以优化一个从真实联合分布中提取的所有子序列的似然的变分下界。

## 1.6. 原文链接
*   **原文链接:** [https://arxiv.org/abs/2407.01392](https://arxiv.org/abs/2407.01392)
*   **PDF 链接:** [https://arxiv.org/pdf/2407.01392v4.pdf](https://arxiv.org/pdf/2407.01392v4.pdf)
*   **发布状态:** 预印本 (Pre-print)

# 2. 整体概括

## 2.1. 研究背景与动机
### 2.1.1. 核心问题
概率序列建模是机器学习中的一个基础性问题，广泛应用于自然语言处理、视频预测和决策制定等领域。当前主流的序列生成方法主要分为两大类，但各自存在明显的局限性：

1.  <strong>下一词元预测模型 (Next-token Prediction Models):</strong>
    *   **代表:** GPT系列等自回归模型。
    *   **训练方式:** 通常使用<strong>教师强制 (Teacher Forcing)</strong>，即基于真实的过去序列预测下一个词元。
    *   **优点:**
        *   可以生成可变长度的序列。
        *   可以基于不同长度的历史进行条件生成。
        *   支持高效的树搜索算法。
    *   <strong>缺点/挑战 (Gaps):</strong>
        *   **缺乏引导机制:** 无法在生成过程中引导整个序列以满足某个全局目标（例如，在规划任务中最大化总奖励）。
        *   **连续数据不稳定:** 在处理视频等连续数据时，自回归生成过程中微小的预测误差会逐帧累积，导致序列在超出训练长度后迅速发散和失真。

2.  <strong>全序列扩散模型 (Full-sequence Diffusion Models):</strong>
    *   **代表:** 视频扩散模型、Diffuser等。
    *   **训练方式:** 将整个序列视为一个大数据块，对所有词元施加**相同水平**的噪声，并训练模型一次性去噪整个序列。
    *   **优点:**
        *   支持<strong>扩散引导 (diffusion guidance)</strong>，可以在采样时将序列引向期望的属性（如高奖励）。
        *   在生成视频等连续信号方面表现出色，鲁棒性强。
    *   <strong>缺点/挑战 (Gaps):</strong>
        *   **固定长度生成:** 只能生成固定长度的序列。
        *   **非因果架构:** 通常使用无掩码的非因果架构（如U-Net），这限制了其在需要可变历史和未来的场景中的灵活性。
        *   **引导能力受限:** 非因果结构限制了更复杂的引导策略。

            下图（原文 Figure 1）直观展示了现有方法的优劣势以及 Diffusion Forcing 的定位。

            ![Figure 1: Diffusion Forcing capabilities. Today, different applications such as language modeling \[6\], planning \[37\], or video generation \[32, 70\] rely on either auto-regressive next-token prediction or full-sequence diffusion, according to their respective unique capabilities. The proposed Diffusion Forcing is a novel sequence generative model that enjoys key strengths of both model types.](images/1.jpg)
            *该图像是一个示意图，展示了Diffusion Forcing方法的关键能力，包括引导性、树搜索、组合性、因果不确定性和灵活的时间范围。图中显示了Diffusion Forcing与教师强迫和全序列扩散在不同特性上的对比，标记了各个方法的适用性。*

### 2.1.2. 创新切入点
论文的创新切入点在于提出一个统一的框架，旨在**融合上述两种方法的优点**。其核心思想是：将加噪过程视为一种<strong>部分掩码 (partial masking)</strong>。完全无噪的词元是可见的，而完全加噪的词元则被完全“掩盖”。

基于此，**Diffusion Forcing (DF)** 提出了一种全新的训练范式：在训练时，序列中的**每一个词元都关联一个随机且独立的噪声水平**。这迫使模型学会从任意噪声组合的序列中恢复原始数据，从而在“全可见”（类似教师强制）和“全掩盖”（类似扩散模型初始状态）之间架起了一座桥梁。这种设计既保留了下一词元预测的因果结构，又引入了扩散模型的去噪和引导能力。

## 2.2. 核心贡献/主要发现
1.  <strong>提出 Diffusion Forcing (DF) 范式:</strong> 提出了一种新的概率序列模型训练方法。该方法通过为序列中的每个词元独立采样噪声水平，结合了下一词元预测模型的灵活性和全序列扩散模型的长时程引导能力。

2.  <strong>提出 Causal Diffusion Forcing (CDF):</strong> 将 DF 范式具体实例化为一个用于序列生成的因果模型（CDF），该模型使用循环神经网络（RNN）或带掩码的 Transformer 等因果架构。

3.  **开发了新的采样和引导能力:**
    *   **稳定长时程生成:** 解决了自回归模型在连续数据（如视频）上生成长序列时误差累积和发散的问题。
    *   <strong>蒙特卡洛引导 (Monte Carlo Guidance, MCG):</strong> 提出了一种新的引导机制。利用 CDF 中未来词元的不确定性，通过对多个未来轨迹的期望奖励进行采样和平均，从而更鲁棒地指导当前词元的生成。这在决策和规划任务中带来了显著的性能提升。

4.  **提供了理论证明:** 从理论上证明了 DF 的训练目标是在优化一个关于所有子序列似然的变分下界 (ELBO)，为该方法的有效性提供了坚实的数学基础。

5.  **广泛的实验验证:** 在视频生成、模型规划、视觉模仿学习和时间序列预测等多个不同领域验证了 CDF 的有效性和独特能力。

# 3. 预备知识与相关工作

## 3.1. 基础概念
### 3.1.1. 贝叶斯滤波 (Bayesian Filtering)
贝叶斯滤波是一种在<strong>隐马尔可夫模型 (Hidden Markov Model, HMM)</strong> 中随时间递归地估计潜变量状态的概率方法。它主要包含两个核心模型：
*   <strong>先验模型 (Prior Model) / 转移模型 (Transition Model):</strong> $p(\mathbf{z}_{t+1} | \mathbf{z}_t)$，根据当前潜状态 $\mathbf{z}_t$ 预测下一个潜状态 $\mathbf{z}_{t+1}$ 的分布。
*   <strong>观测模型 (Observation Model):</strong> $p(\mathbf{x}_t | \mathbf{z}_t)$，根据当前潜状态 $\mathbf{z}_t$ 推断当前观测值 $\mathbf{x}_t$ 的分布。
    当获得一个新的观测值 $\mathbf{x}_{t+1}$ 时，通过贝叶斯定理更新潜状态的估计，得到<strong>后验模型 (Posterior Model)</strong>：$p(\mathbf{z}_{t+1} | \mathbf{z}_t, \mathbf{x}_{t+1})$。
在本文中，CDF 的因果结构借鉴了贝叶斯滤波的思想，其中 RNN 的隐藏状态 $\mathbf{z}_t$ 扮演了潜变量的角色，总结了过去的信息以预测未来。

### 3.1.2. 扩散模型 (Diffusion Models)
扩散模型是一类强大的生成模型，其核心思想是通过两个过程来学习数据分布：
1.  <strong>前向过程 (Forward Process):</strong> 从一个真实数据点 $\mathbf{x}^0$ 开始，通过 $K$ 步逐渐向其添加高斯噪声，直到数据最终变成纯粹的噪声 $\mathbf{x}^K$（通常是标准正态分布）。第 $k$ 步的加噪过程可以表示为：
    $$
    q(\mathbf{x}^k | \mathbf{x}^{k-1}) = \mathcal{N}(\mathbf{x}^k; \sqrt{1 - \beta_k} \mathbf{x}^{k-1}, \beta_k \mathbf{I})
    $$
    其中 $\beta_k$ 是预设的噪声方差。由于这个过程是马尔可夫的，我们可以直接从 $\mathbf{x}^0$ 得到任意步 $k$ 的带噪样本 $\mathbf{x}^k$：
    $$
    \mathbf{x}^k = \sqrt{\bar{\alpha}_k} \mathbf{x}^0 + \sqrt{1 - \bar{\alpha}_k} \boldsymbol{\epsilon}, \quad \text{其中 } \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})
    $$
    这里 $\alpha_k = 1 - \beta_k$，$ \bar{\alpha}_k = \prod_{i=1}^k \alpha_i $。

2.  <strong>反向过程 (Reverse Process):</strong> 训练一个神经网络 $\boldsymbol{\epsilon}_\theta(\mathbf{x}^k, k)$ 来预测在第 $k$ 步添加的噪声 $\boldsymbol{\epsilon}$。训练目标通常是最小化预测噪声与真实噪声之间的均方误差 (MSE)：
    $$
    \mathcal{L}(\boldsymbol{\theta}) = \mathbb{E}_{k, \mathbf{x}^0, \boldsymbol{\epsilon}} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}} (\mathbf{x}^k, k) \|^2 \right]
    $$
    在生成样本时，从一个纯噪声 $\mathbf{x}^K \sim \mathcal{N}(0, \mathbf{I})$ 开始，利用训练好的模型 $\boldsymbol{\epsilon}_\theta$ 逐步去噪，直到得到一个清晰的样本 $\mathbf{x}^0$。

### 3.1.3. 扩散模型的引导 (Guidance of Diffusion Models)
引导是一种在采样阶段控制扩散模型生成内容的技术。本文关注的是<strong>分类器引导 (classifier guidance)</strong>。其核心思想是利用一个分类器 $c(y|\mathbf{x}^k)$ 的梯度来“引导”去噪过程，使其生成的样本 $\mathbf{x}$ 更可能属于期望的类别 $y$。具体来说，修改后的噪声预测变为：
$$
\hat{\boldsymbol{\epsilon}}_{\theta}(\mathbf{x}^k, k) = \boldsymbol{\epsilon}_{\theta}(\mathbf{x}^k, k) - w \sqrt{1 - \bar{\alpha}_k} \nabla_{\mathbf{x}^k} \log c(y | \mathbf{x}^k)
$$
其中 $w$ 是引导强度。这个原理可以推广到任意可微的能量函数或目标函数，例如在规划任务中，可以用一个奖励模型来引导生成高回报的轨迹。

### 3.1.4. 下一词元预测模型 (Next-Token Prediction Models)
这类模型以自回归的方式生成序列。
*   **训练:** 使用<strong>教师强制 (Teacher Forcing)</strong>，即给定一个真实的序列前缀 $\mathbf{x}_{1:t}$，模型被训练来预测下一个真实的词元 $\mathbf{x}_{t+1}$。
*   **采样:** 从一个初始词元开始，模型预测出下一个词元 $\hat{\mathbf{x}}_{t+1}$，然后将这个预测的词元加入到输入序列中，再预测 $\hat{\mathbf{x}}_{t+2}$，如此循环，直到生成完整的序列。这种模式也称为<strong>自回归采样 (auto-regressive sampling)</strong>。

## 3.2. 前人工作
*   <strong>序列扩散模型 (Diffusion Sequence Models):</strong>
    *   <strong>全序列扩散 (Full-sequence diffusion):</strong> 如 `Diffuser` [37]，它将整个轨迹（状态、动作、奖励）视为一个整体进行扩散，并使用引导来规划高奖励的路径。但它生成的是固定长度的轨迹，且架构是非因果的。
    *   <strong>自回归扩散 (Autoregressive diffusion):</strong> 如 `TimeGrad` [50] 等模型，它们将扩散过程应用于单个词元的生成。在每一步，模型基于**无噪声的**历史信息，通过一个完整的扩散过程来生成下一个词元。这种方法本质上还是自回归的，容易累积误差。
*   **具有变化噪声水平的序列扩散模型:**
    *   **AR-Diffusion [66]:** 这是与本文最相似的工作。它也使用因果架构训练扩散模型，但其关键区别在于，噪声水平 $k_t$ 是随词元位置 $t$ **线性变化**的，而不是独立的。这种固定的、依赖性的噪声方案限制了其采样的灵活性，无法实现 DF 所带来的如稳定自回归生成、蒙特卡洛引导等高级功能。
    *   **Rolling Diffusion [52]:** 同样提出了一种依赖于词元位置的噪声方案，旨在使近期未来的不确定性低于远期未来。它也存在与 AR-Diffusion 类似的局限性，即噪声方案在训练时被固定，缺乏采样时的灵活性。

## 3.3. 差异化分析
Diffusion Forcing 的核心创新在于**训练时每个词元噪声水平的独立性**。下图（原文 Figure 2）清晰地展示了 DF 与传统方法的区别：

![Figure 2: Method Overview. Diffusion Forcing trains causal sequence neural networks (such as an RNN or a masked transformer) to denoise flexible-length sequences where each frame of the sequence can have a different noise level. In contrast, next-token prediction models, common in language modeling, are trained to predict a single next token from a ground-truth sequence (teacher forcing \[65\]), and full-sequence diffusion, common in video generation, train non-causal architectures to denoise all frames in a sequence at once with the same noise level. Diffusion Forcing thus interleaves the time axis of the sequence and the noise axis of diffusion, unifying strengths of both alternatives and enabling completely new capabilities (see Secs. 3.2,3.4).](images/2.jpg)
*该图像是示意图，展示了 Diffusion Forcing、Teacher Forcing 和 Full-Seq. Diffusion 三种不同的序列生成机制。上方部分为训练过程，显示了不同噪声水平下的序列生成；下方为采样过程，展示了各机制的生成流程与噪声添加方式。*

| 特性 | 下一词元预测 (教师强制) | 全序列扩散 | Diffusion Forcing (本文方法) |
| :--- | :--- | :--- | :--- |
| **训练噪声** | 只有最后一个词元是“完全加噪”（即被掩码），历史词元完全无噪。 | 所有词元具有**相同**的噪声水平。 | 每个词元具有**独立**、随机的噪声水平。 |
| **架构** | 因果 (Causal) | 非因果 (Non-causal) | 因果 (Causal) |
| **生成长度** | 可变 | 固定 | 可变 |
| **引导能力** | 无 | 有（但受非因果架构限制） | 有（更灵活，支持 MCG 等新方案） |
| **连续数据稳定性** | 差（误差累积） | 好 | 好（通过噪声注入稳定） |
| **核心思想** | 预测下一个 | 一次性生成全部 | 逐步去噪任意噪声组合的序列 |

# 4. 方法论

## 4.1. 方法原理
Diffusion Forcing (DF) 的核心思想是<strong>将加噪视为一种部分掩码 (Noising as partial masking)</strong>。传统的掩码方法（如 BERT）将词元分为“可见”或“不可见”两种状态。DF 将此概念推广到一个连续的谱系：
*   <strong>无噪声 ($k=0$)</strong>: 词元 $\mathbf{x}_t^0$ 完全可见，等同于未被掩码。
*   <strong>完全噪声 ($k=K$)</strong>: 词元 $\mathbf{x}_t^K$ 变为纯高斯噪声，不含任何原始信息，等同于被完全掩码。
*   **部分噪声 ($0 < k < K$)**: 词元 $\mathbf{x}_t^k$ 被部分掩码，保留了部分原始信息。

    DF 框架通过让模型学习去噪一个序列 $(\mathbf{x}_t^{k_t})_{1 \le t \le T}$，其中每个词元的噪声水平 $k_t$ 都是独立随机采样的。这迫使模型学会处理任意“掩码”组合的序列，从而获得了极大的灵活性。

## 4.2. 核心方法详解 (逐层深入)
本文将 DF 实例化为<strong>因果扩散强制 (Causal Diffusion Forcing, CDF)</strong>，并以一个循环神经网络 (RNN) 为例进行说明。

### 4.2.1. CDF 架构与动态
CDF 模型的核心是一个 RNN 单元，它维护一个随时间演变的潜状态 $\mathbf{z}_t$。这个潜状态总结了截至时间 $t$ 的所有历史信息。
*   <strong>潜状态更新 (Dynamics Model):</strong> 当在时间 $t$ 接收到一个带噪的观测值 $\mathbf{x}_t^{k_t}$ 时，潜状态 $\mathbf{z}_t$ 根据前一个潜状态 $\mathbf{z}_{t-1}$、当前带噪观测值 $\mathbf{x}_t^{k_t}$ 以及其噪声水平 $k_t$ 进行更新。这个过程可以表示为一个概率转移：
    $$
    \mathbf{z}_t \sim p_\theta(\mathbf{z}_t | \mathbf{z}_{t-1}, \mathbf{x}_t^{k_t}, k_t)
    $$
    这个更新过程可以看作贝叶斯滤波的推广：
    *   当 $k_t=0$ 时（观测无噪），这类似于贝叶斯滤波中的**后验更新**。
    *   当 $k_t=K$ 时（观测为纯噪声，无信息），这类似于贝叶斯滤波中的**先验预测** $p_\theta(\mathbf{z}_t | \mathbf{z}_{t-1})$。

*   <strong>噪声预测 (Observation Model):</strong> 基于潜状态 $\mathbf{z}_{t-1}$ 和当前带噪观测 $\mathbf{x}_t^{k_t}$，模型需要预测出添加到原始词元 $\mathbf{x}_t^0$ 上的噪声 $\boldsymbol{\epsilon}_t$。这个预测由一个神经网络 $\boldsymbol{\epsilon}_\theta$ 完成：
    $$
    \hat{\boldsymbol{\epsilon}}_t = \boldsymbol{\epsilon}_\theta(\mathbf{z}_{t-1}, \mathbf{x}_t^{k_t}, k_t)
    $$

### 4.2.2. 训练过程 (Algorithm 1)
CDF 的训练过程非常直观，其伪代码在原文 Algorithm 1 中给出。
**算法 1: Diffusion Forcing 训练**
1.  **循环执行:**
2.  从数据集中采样一个真实的观测序列 $(\mathbf{x}_1, \dots, \mathbf{x}_T)$。
3.  **对序列中的每个时间步 $t = 1, \dots, T$ 执行:**
4.  为当前词元 $\mathbf{x}_t$ 独立地从 $\{0, 1, \dots, K\}$ 中均匀采样一个噪声水平 $k_t$。
5.  使用前向扩散过程将 $\mathbf{x}_t$ 加噪到水平 $k_t$，得到 $\mathbf{x}_t^{k_t} = \sqrt{\bar{\alpha}_{k_t}}\mathbf{x}_t + \sqrt{1 - \bar{\alpha}_{k_t}}\boldsymbol{\epsilon}_t$。
6.  定义目标噪声 $\boldsymbol{\epsilon}_t$。
7.  更新潜状态 $\mathbf{z}_t \sim p_\theta(\mathbf{z}_t | \mathbf{z}_{t-1}, \mathbf{x}_t^{k_t}, k_t)$。
8.  使用模型预测噪声 `\hat{\boldsymbol{\epsilon}}_t = \boldsymbol{\epsilon}_\theta(\mathbf{z}_{t-1}, \mathbf{x}_t^{k_t}, k_t)`。
9.  **结束对时间步的循环。**
10. 计算所有时间步的预测噪声与真实噪声之间的均方误差损失：
    $$
    \mathcal{L} = \text{MSELoss}([\hat{\boldsymbol{\epsilon}}_1, \dots, \hat{\boldsymbol{\epsilon}}_T], [\boldsymbol{\epsilon}_1, \dots, \boldsymbol{\epsilon}_T])
    $$
11. 使用损失 $\mathcal{L}$ 进行反向传播并更新模型参数 $\theta$。
12. **结束循环。**

    这个训练过程的核心是最小化以下损失函数：
$$
\underset { \substack { k _ { t } , \mathbf { x } _ { t } , \epsilon _ { t } } } { \mathbb { E } } \sum _ { t=1 } ^ { T } \bigg [ \| \epsilon _ { t } - \epsilon _ { \theta } \big ( \mathbf { z } _ { t - 1 } , \mathbf { x } _ { t } ^ { k _ { t } } , k _ { t } \big ) \| ^ { 2 } \bigg ] \quad \text{, where } \mathbf { z } _ { t } \sim p _ { \theta } ( \mathbf { z } _ { t } | \mathbf { z } _ { t - 1 } , \mathbf { x } _ { t } ^ { k _ { t } } , k _ { t } )
$$
其中，期望 $\mathbb{E}$ 是对所有可能的噪声水平 $k_t$（从 `[K]` 中均匀采样）、数据 $\mathbf{x}_t$（从训练数据中采样）以及标准高斯噪声 $\boldsymbol{\epsilon}_t$ 取的。

### 4.2.3. 理论依据 (Theorem 3.1 & Appendix A)
论文在附录A中证明，上述训练目标实际上是在优化一个<strong>证据下界 (Evidence Lower Bound, ELBO)</strong>，该下界是所有可能子序列对数似然的加权和。

<strong>Theorem 3.1 (非正式)</strong>: Diffusion Forcing 训练过程（算法1）优化了在所有子序列（通过不同噪声水平组合定义）的期望对数似然上的一个变分下界。在适当条件下，优化这个目标函数等价于同时最大化所有噪声水平序列的似然下界。

这意味着模型不仅学习了生成完整的序列，还学习了生成任意子序列的条件分布（例如，给定 $\mathbf{x}_1, \mathbf{x}_3$，生成 $\mathbf{x}_2$）。

### 4.2.4. 采样过程 (Algorithm 2)
DF 的采样过程非常灵活，由一个二维的噪声调度矩阵 $\boldsymbol{\kappa} \in [K]^{M \times T}$ 控制。
*   $\boldsymbol{\kappa}$ 的行 $m$ 代表去噪步骤（共 $M$ 步）。
*   $\boldsymbol{\kappa}$ 的列 $t$ 代表序列的时间步（共 $T$ 步）。
*   $\boldsymbol{\kappa}_{m,t}$ 表示在第 $m$ 个去噪步骤时，第 $t$ 个词元的噪声水平。

**算法 2: 带引导的 DF 采样**
1.  **输入:** 模型 $\theta$, 调度矩阵 $\boldsymbol{\kappa}$, 初始潜状态 $\mathbf{z}_0$, 引导代价函数 $c(\cdot)$。
2.  **初始化:** 生成一个长度为 $T$ 的纯噪声序列 $\mathbf{x}_{1:T}$，对应噪声水平为 $K$。
3.  **对每个去噪步骤 $m = M-1, \dots, 0$ 执行:**
4.  **对序列中的每个时间步 $t = 1, \dots, T$ 执行:**
5.  更新潜状态：$\mathbf{z}_t \sim p_\theta(\mathbf{z}_t | \mathbf{z}_{t-1}, \mathbf{x}_t^{\boldsymbol{\kappa}_{m+1,t}}, \boldsymbol{\kappa}_{m+1,t})$。
6.  获取当前时间步的目标噪声水平 $k \leftarrow \boldsymbol{\kappa}_{m,t}$。
7.  执行一步去噪，得到新的词元 $\mathbf{x}_t^{\text{new}}$。
8.  **结束对时间步的循环。**
9.  （可选）**应用引导:**
10. 基于代价函数 $c(\cdot)$ 的梯度修改更新后的序列 $\mathbf{x}_{1:T}^{\text{new}}$。
11. **结束对去噪步骤的循环。**
12. **返回** 完全去噪的序列 $\mathbf{x}_{1:T}$。

    通过设计不同的调度矩阵 $\boldsymbol{\kappa}$，可以实现多种采样行为，而无需重新训练模型。

### 4.2.5. DF 带来的新能力
#### 稳定自回归生成
传统自回归模型在生成长视频时会发散。DF 通过在每一步生成后，不直接使用完全去噪的词元 $\mathbf{x}_t^0$ 来更新潜状态，而是使用一个**带轻微噪声的词元 $\mathbf{x}_t^{k_{\text{small}}}$**（其中 $k_{\text{small}}$ 是一个很小的正整数）。因为模型在训练时见过各种噪声水平的输入，所以它对这种带噪输入是鲁棒的，从而有效抑制了误差的累积。

#### 蒙特卡洛引导 (Monte Carlo Guidance, MCG)
这是 DF 在决策任务中的一个关键创新。在传统的引导中，我们基于**一个**采样的未来轨迹来计算引导梯度。但由于 CDF 的因果结构和未来词元的不确定性（即它们仍然是带噪的），我们可以：
1.  从当前的带噪状态 $(\mathbf{x}_t^k, \mathbf{z}_{t-1})$ 出发，<strong>采样多条（N条）可能的未来轨迹</strong>。
2.  为每一条未来轨迹计算引导梯度（例如，基于累计奖励）。
3.  将这 N 个梯度**平均**，用这个更稳定、方差更小的平均梯度来指导当前词元 $\mathbf{x}_t^k$ 的去噪。

    这相当于用**未来所有可能结果的期望回报**来指导当前决策，而不是仅仅依赖于某一个随机的未来。这在随机环境中尤其有效。全序列扩散模型无法实现这一点，因为它们在每一步去噪中，所有词元的状态是确定的，没有内在的随机性源来采样多个未来。

下图（原文 Figure 5）展示了如何通过控制噪声水平 k 来实现不同的预测和规划效果。

![Figure 5: Diffusion Forcing is trained on independent level of noises at different timesteps. As a result, we can control the noise level $k$ to achieve different effects on conditioning and prediction.](images/6.jpg)
*该图像是示意图，展示了Diffusion Forcing在不同噪声水平下的训练过程。通过设置不同的噪声级别 $k$，模型能够用于对历史、近未来和远未来的不同处理，从而实现有效的预测和规划。*

# 5. 实验设置

## 5.1. 数据集
论文在多个领域验证了 Diffusion Forcing 的性能：

*   <strong>视频预测 (Video Prediction):</strong>
    *   **Minecraft:** 包含在 Minecraft 游戏中随机行走的第一人称视角视频。
    *   **DMLab:** 包含在 3D 迷宫环境中随机行走的视频。
    *   这两个数据集均来自先前的研究 [69]，用于测试模型生成长时程、时序一致视频的能力。

*   <strong>扩散规划 (Diffusion Planning):</strong>
    *   **D4RL [18]:** 一个标准的离线强化学习基准。实验使用了其中的 2D 迷宫环境 (`maze2d-medium-v1`, `maze2d-large-v1`, `maze2d-umaze-v1`)，这些任务具有稀疏奖励和长时程特性，非常适合评估规划算法。

*   <strong>可控序列组合生成 (Controllable Sequential Compositional Generation):</strong>
    *   **2D 平面轨迹:** 一个自建数据集，包含从一个角落移动到对角角落的“十”字形轨迹。用于展示模型通过改变采样策略（全时程 vs. 无记忆 MPC）来组合子序列的能力。
    *   下图（原文 Figure 7）展示了此实验中的数据形态与生成结果。

        ![Figure 7: Given a dataset of trajectories (a), Diffusion Forcing models the joint distribution of all subsequences of arbitrary length. At sampling time, we can sample from the trajectory distribution by sampling Diffusion Forcing with full horizon (b) or recover Markovian dynamics by disregarding previous states (c).](images/8.jpg)
        *该图像是示意图，展示了数据集（a）、具有记忆的模型（b）以及不具有记忆的模型（c）在生成轨迹时的表现。每个子图表现了不同的轨迹生成方式，展示了Diffusion Forcing在处理时间序列数据时的效果。*

*   <strong>机器人学 (Robotics):</strong>
    *   **Franka 机器人水果交换:** 一个真实的机器人任务。机器人需要利用一个空槽来交换桌上两个水果（苹果和橙子）的位置。由于水果的初始位置是随机的，任务需要模型具备记忆能力才能正确执行。
    *   下图（原文 Figure 4）展示了该任务的复杂性（需要记忆）以及模型的视频生成能力。

        ![Figure 4: In our real robot task, a robot arm is asked to swap the slots of two fruits using a third slot. Since the fruits are input in random slots at the beginning, one cannot determine the next steps from a single observation without knowledge of the initial placement of the fruits. As illustrated in (a) and (b), the upper observation is the same but the desired outcome illustrated below can vary—the task thus requires remembering the initial configuration. In addition, as shown in (c), the same model that generates actions also synthesizes realistic video from just a single frame.](images/5.jpg)
        *该图像是图表，展示了一个机器人臂在两种状态下交换水果的任务。左侧展示了任务的目标状态与中间状态，右侧则显示了从输入帧生成的视频预测，展示了不同时间点的预测结果。生成的视频通过模型在单帧输入的基础上合成。*

*   <strong>时间序列预测 (Time Series Forecasting):</strong>
    *   **GluonTS [2] 数据集:** 包括 `Exchange`, `Solar`, `Electricity`, `Traffic`, `Taxi`, `Wikipedia` 等多个真实世界的高维时间序列数据集。

## 5.2. 评估指标
*   <strong>规划任务 (D4RL):</strong>
    *   <strong>Episode Reward (回合奖励):</strong> 智能体在一个回合（episode）内获得的总奖励。越高越好。

*   **时间序列预测:**
    *   **Summed Continuous Ranked Probability Score ($\text{CRPS}_{\text{sum}}$):** 一种衡量预测概率分布与真实观测值匹配程度的指标。
        1.  <strong>概念定义 (Conceptual Definition):</strong> CRPS 评估的是一个概率预测的整体表现，它将预测的累积分布函数 (CDF) 与观测值的阶跃函数进行比较。其值可以解释为预测值与真实值之间的绝对误差的一般化。$\text{CRPS}_{\text{sum}}$ 是将多维时间序列在特征维度上求和后，再计算 CRPS，并对预测窗口内的所有时间步取平均。值越低表示预测越准。
        2.  <strong>数学公式 (Mathematical Formula):</strong>
            单变量的 CRPS 定义为：
            $$
            \mathrm { CRPS } ( F , x ) = \int _ { \mathbb { R } } \left( F ( z ) - \mathbb { I } \left\{ x \leq z \right\} \right) ^ { 2 } \mathrm { d } z
            $$
            本文使用的 $\text{CRPS}_{\text{sum}}$ 为：
            $$
            \mathrm { CRPS } _ { \mathrm { sum } } = \mathbb { E } _ { t \sim \mathcal { U } ( t _ { 0 } , T ) } \left[ \mathrm { CRPS } \left( \hat { F } _ { \mathrm { sum } } ( t ) , \sum _ { i } x _ { i , t } ^ { 0 } \right) \right]
            $$
        3.  <strong>符号解释 (Symbol Explanation):</strong>
            *   `F(z)`: 预测值的累积分布函数 (CDF)。
            *   $x$: 真实的观测值。
            *   $\mathbb{I}\{x \le z\}$: 指示函数，当 $x \le z$ 时为 1，否则为 0。
            *   $\hat{F}_{\text{sum}}(t)$: 在时间步 $t$，沿特征维度求和后得到的预测值的经验 CDF。
            *   $\sum_i x_{i,t}^0$: 在时间步 $t$，沿特征维度求和后得到的真实观测值。
            *   $\mathbb{E}_{t \sim \mathcal{U}(t_0, T)}$: 对预测窗口 $[t_0, T]$ 内的时间步取期望（平均）。

## 5.3. 对比基线
*   **视频预测:**
    *   `Next-frame diffusion`: 基于教师强制的下一帧扩散模型。
    *   `Causal full-sequence diffusion`: 使用因果架构的全序列扩散模型。
*   **扩散规划:**
    *   `Diffuser`: 最先进的基于扩散的规划方法。
    *   `CQL`, `IQL`: 经典的离线强化学习算法。
    *   `MPPI`: 一种基于采样的模型预测控制算法。
*   **机器人学:**
    *   `Diffusion Policy`: 一种无记忆的、基于扩散的模仿学习算法。
*   **时间序列预测:**
    *   `TimeGrad`, `ScoreGrad`: 基于扩散和得分匹配的时间序列预测模型。
    *   `Transformer-MAF`: 基于 Transformer 和归一化流的模型。
    *   `DeepAR`, `GP-Copula` 等经典和现代时间序列模型。

# 6. 实验结果与分析

## 6.1. 核心结果分析
### 6.1.1. 视频预测：稳定无限推演
*   **结果:** 如图 3 所示，在 Minecraft 和 DMLab 数据集上，Diffusion Forcing (Ours) 能够生成时序上连贯且稳定的长视频序列（例如，生成 1000 帧，远超训练时的几十帧），而其他基线方法很快出现图像失真、内容跳变或完全发散。
*   **分析:** 这有力地证明了 DF 通过在潜状态更新时注入少量噪声来稳定自回归生成的能力。该方法有效打破了误差累积的恶性循环，这是传统自回归模型在连续高维数据上面临的核心难题。

    下图（原文 Figure 3）直观对比了不同方法的生成质量。

    ![Figure 3: Video Generation. Among tested methods, Diffusion Forcing generations are uniquely temporally consistent and do not diverge even when rolling out well past the training horizon. Please see the project website for video results.](images/4.jpg)
    *该图像是一个示意图，展示了不同方法生成的图像序列，包括 DMLab 和 Minecraft 的输入及各自生成的图像。图中显示了 Diffusion Forcing 与其他生成方法的结果对比，强调其在时间一致性上的表现。*

### 6.1.2. 扩散规划：MCG 与因果建模的优势
*   **结果:** 在 D4RL 迷宫任务中，DF 在所有 6 个环境中均显著优于包括 `Diffuser` 在内的所有基线方法。
*   **分析:**
    *   <strong>蒙特卡洛引导 (MCG) 的威力:</strong> 消融实验表明，去掉 MCG 后，DF 的性能有所下降，但仍然具有竞争力。这证明了通过对多个未来进行采样来平滑引导梯度的策略是其性能提升的关键因素。
    *   **因果建模的重要性:** `Diffuser` 是非因果的，其生成的动作和状态之间没有严格的动态一致性。因此，其实际部署时**必须**丢弃生成的动作，转而使用一个手工设计的 PD 控制器根据生成的状态来推算动作。相比之下，DF 的因果架构保证了生成的动作和状态是自洽的，可以直接执行生成的动作，并且性能远超 `Diffuser`+PD 控制器的组合。
    *   **灵活的规划视野:** DF 天然支持可变长度的规划，而 `Diffuser` 等全序列模型难以适应规划视野动态缩减的任务。

        以下是原文 Table 1 的结果，由于该表格包含合并行，故使用 HTML 格式呈现：

        <table>
        <thead>
        <tr>
        <th>Environment</th>
        <th>MPPI</th>
        <th>CQL</th>
        <th>IQL</th>
        <th>Diffuser*</th>
        <th>Diffuser w/ diffused action</th>
        <th>Ours wo/ MCG</th>
        <th>Ours</th>
        </tr>
        </thead>
        <tbody>
        <tr>
        <td colspan="8" align="center"><strong>Single-task</strong></td>
        </tr>
        <tr>
        <td>Maze2D U-Maze</td>
        <td>33.2</td>
        <td>5.7</td>
        <td>47.4</td>
        <td>113.9 ± 3.1</td>
        <td>6.3 ± 2.1</td>
        <td>110.1 ± 3.9</td>
        <td>116.7 ± 2.0</td>
        </tr>
        <tr>
        <td>Maze2D Medium</td>
        <td>10.2</td>
        <td>5.0</td>
        <td>34.9</td>
        <td>121.5 ± 2.7</td>
        <td>13.5 ± 2.3</td>
        <td>136.1 ± 10.2</td>
        <td>149.4 ± 7.5</td>
        </tr>
        <tr>
        <td>Maze2D Large</td>
        <td>5.1</td>
        <td>12.5</td>
        <td>58.6</td>
        <td>123.0 ± 6.4</td>
        <td>6.3 ± 2.1</td>
        <td>142.8 ± 5.6</td>
        <td>159.0 ± 2.7</td>
        </tr>
        <tr>
        <td><strong>Average</strong></td>
        <td>16.2</td>
        <td>7.7</td>
        <td>47.0</td>
        <td>119.5</td>
        <td>8.7</td>
        <td>129.67</td>
        <td>141.7</td>
        </tr>
        <tr>
        <td colspan="8" align="center"><strong>Multi-task</strong></td>
        </tr>
        <tr>
        <td>Multi2D U-Maze</td>
        <td>41.2</td>
        <td>-</td>
        <td>24.8</td>
        <td>128.9 ± 1.8</td>
        <td>32.8 ± 1.7</td>
        <td>107.7 ± 4.9</td>
        <td>119.1 ± 4.0</td>
        </tr>
        <tr>
        <td>Multi2D Medium</td>
        <td>15.4</td>
        <td>-</td>
        <td>12.1</td>
        <td>127.2 ± 3.4</td>
        <td>22.0 ± 2.7</td>
        <td>145.6 ± 6.5</td>
        <td>152.3 ± 9.9</td>
        </tr>
        <tr>
        <td>Multi2D Large</td>
        <td>8.0</td>
        <td>-</td>
        <td>13.9</td>
        <td>132.1 ± 5.8</td>
        <td>6.9 ± 1.7</td>
        <td>129.8 ± 1.5</td>
        <td>167.1 ± 2.7</td>
        </tr>
        <tr>
        <td><strong>Average</strong></td>
        <td>21.5</td>
        <td>-</td>
        <td>16.9</td>
        <td>129.4</td>
        <td>20.6</td>
        <td>127.7</td>
        <td>146.2</td>
        </tr>
        </tbody>
        </table>

### 6.1.3. 机器人模仿学习：记忆与鲁棒性
*   **结果:** 在需要记忆的水果交换任务中，DF 取得了 80% 的成功率，而无记忆的 SOTA 方法 `Diffusion Policy` 完全失败。当引入视觉干扰（如遮挡摄像头）时，DF 的成功率仅轻微下降至 76%，而基线方法降至 48%。
*   **分析:** 这表明 DF 的潜状态 $\mathbf{z}_t$ 成功地编码了任务所需的历史信息（记忆），这是完成长时程、部分可观测任务的关键。同时，由于 DF 训练时见过各种噪声水平的输入，它可以在观测被干扰时，通过将当前观测的噪声水平 $k_t$ 设为一个较大的值，来降低对错误观测的依赖，更多地依赖其内部的先验模型进行预测，从而表现出极强的鲁棒性。

### 6.1.4. 时间序列预测
*   **结果:** 在多个标准时间序列预测基准上，DF 的性能与最先进的方法（如 `ScoreGrad`）相当或更好。
*   **分析:** 这表明 DF 作为一个通用的序列生成模型，其新的训练目标并没有以牺牲在传统任务上的性能为代价。它在提供新能力的同时，保持了强大的基础建模能力。

    以下是原文 Table 2 的时间序列预测结果：

    <table>
    <thead>
    <tr>
    <th>Method</th>
    <th>Exchange</th>
    <th>Solar</th>
    <th>Electricity</th>
    <th>Traffic</th>
    <th>Taxi</th>
    <th>Wikipedia</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>VES [36]</td>
    <td>0.005 ± 0.000</td>
    <td>0.900 ± 0.003</td>
    <td>0.880 ± 0.004</td>
    <td>0.350 ± 0.002</td>
    <td></td>
    <td></td>
    </tr>
    <tr>
    <td>VAR [45]</td>
    <td>0.005 ± 0.000</td>
    <td>0.830 ± 0.006</td>
    <td>0.039 ± 0.001</td>
    <td>0.290 ± 0.001</td>
    <td></td>
    <td></td>
    </tr>
    <tr>
    <td>VAR-Lasso [45]</td>
    <td>0.012 ± 0.000</td>
    <td>0.510 ± 0.006</td>
    <td>0.025 ± 0.000</td>
    <td>0.150 ± 0.002</td>
    <td></td>
    <td>3.100 ± 0.004</td>
    </tr>
    <tr>
    <td>GARCH [62]</td>
    <td>0.023 ± 0.000</td>
    <td>0.880 ± 0.002</td>
    <td>0.190 ± 0.001</td>
    <td>0.370 ± 0.001</td>
    <td></td>
    <td></td>
    </tr>
    <tr>
    <td>DeepAR [55]</td>
    <td></td>
    <td>0.336 ± 0.014</td>
    <td>0.023 ± 0.001</td>
    <td>0.055 ± 0.003</td>
    <td></td>
    <td>0.127 ± 0.042</td>
    </tr>
    <tr>
    <td>LSTM-Copula [54]</td>
    <td>0.007 ± 0.000</td>
    <td>0.319 ± 0.011</td>
    <td>0.064 ± 0.008</td>
    <td>0.103 ± 0.006</td>
    <td>0.326 ± 0.007</td>
    <td>0.241 ± 0.033</td>
    </tr>
    <tr>
    <td>GP-Copula [54]</td>
    <td>0.007 ± 0.000</td>
    <td>0.337 ± 0.024</td>
    <td>0.025 ± 0.002</td>
    <td>0.078 ± 0.002</td>
    <td>0.208 ± 0.183</td>
    <td>0.086 ± 0.004</td>
    </tr>
    <tr>
    <td>KVAE [41]</td>
    <td>0.014 ± 0.002</td>
    <td>0.340 ± 0.025</td>
    <td>0.051 ± 0.019</td>
    <td>0.100 ± 0.005</td>
    <td></td>
    <td>0.095 ± 0.012</td>
    </tr>
    <tr>
    <td>NKF [14]</td>
    <td></td>
    <td>0.320 ± 0.020</td>
    <td>0.016 ± 0.001</td>
    <td>0.100 ± 0.002</td>
    <td></td>
    <td>0.071 ± 0.002</td>
    </tr>
    <tr>
    <td>Transformer-MAF [51]</td>
    <td>0.005 ± 0.003</td>
    <td>0.301 ± 0.014</td>
    <td>0.021 ± 0.000</td>
    <td>0.056 ± 0.001</td>
    <td>0.179 ± 0.002</td>
    <td>0.063 ± 0.003</td>
    </tr>
    <tr>
    <td>TimeGrad [50]</td>
    <td>0.006 ± 0.001</td>
    <td>0.287 ± 0.020</td>
    <td>0.021 ± 0.001</td>
    <td>0.044 ± 0.006</td>
    <td>0.114 ± 0.020</td>
    <td>0.049 ± 0.002</td>
    </tr>
    <tr>
    <td>ScoreGrad sub-VP SDE [68]</td>
    <td>0.006 ± 0.001</td>
    <td>0.256 ± 0.015</td>
    <td>0.019 ± 0.001</td>
    <td>0.041 ± 0.004</td>
    <td>0.101 ± 0.004</td>
    <td>0.043 ± 0.002</td>
    </tr>
    <tr>
    <td>Ours</td>
    <td>0.003 ± 0.001</td>
    <td>0.289 ± 0.002</td>
    <td>0.023 ± 0.001</td>
    <td>0.040 ± 0.004</td>
    <td>0.075 ± 0.002</td>
    <td>0.085 ± 0.007</td>
    </tr>
    </tbody>
    </table>

# 7. 总结与思考

## 7.1. 结论总结
本文成功地提出了 **Diffusion Forcing (DF)**，一个新颖且强大的序列生成模型训练范式。通过**为序列中每个词元独立采样噪声**，DF 巧妙地统一了下一词元预测模型和全序列扩散模型的优点，克服了它们各自的局限性。

主要贡献和发现包括：
1.  **统一框架:** DF 提供了一个统一的视角，将加噪视为部分掩码，从而在因果结构下实现了灵活的序列去噪。
2.  **新能力:** 基于 DF 框架，本文开发了多种新颖的采样和引导技术，特别是<strong>蒙特卡洛引导 (MCG)</strong>，在决策和规划任务中展现出巨大潜力。同时，它解决了自回归模型在生成长时程连续数据时的稳定性问题。
3.  **强大性能:** 在视频生成、机器人模仿学习、模型规划和时间序列预测等多样化任务中，DF 均表现出与最先进方法相当或更优的性能。
4.  **理论支撑:** 为 DF 的训练目标提供了优化 ELBO 的理论证明，增强了方法的可信度。

    总而言之，Diffusion Forcing 不仅仅是一个增量改进，更是一个具有高度可扩展性和应用前景的基础性框架，为未来的序列生成研究开辟了新的方向。

## 7.2. 局限性与未来工作
论文作者在讨论部分指出了当前工作的一些局限性，并展望了未来的研究方向：
*   **架构局限性:** 当前的因果实现主要基于 RNN。虽然 RNN 在在线决策中效率较高，但在处理更高分辨率视频或更复杂的分布时，可能需要采用更强大的**Transformer**架构作为主干网络 (backbone)。
*   **规模化验证:** 本文的实验尚未在互联网规模的超大数据集上进行验证。未来的工作需要探索 DF 在更大规模任务上的扩展性和性能表现。
*   **应用领域扩展:** 未来的工作可以研究将 DF 应用于时间序列生成建模之外的其他领域，例如自然语言处理中的可控文本生成、音乐生成等。

## 7.3. 个人启发与批判
这篇论文给我带来了深刻的启发，其方法设计优雅且富有洞察力。

*   **思想的统一性:** “加噪即掩码”的思想非常精妙。它将离散的掩码操作和连续的加噪过程统一起来，并进一步推广到“独立的部分掩码”，这是整个方法能够成功的基石。这种从更高维度统一不同技术范式的思路极具启发性。

*   **灵活性的价值:** DF 的核心优势在于其**灵活性**。通过在训练时引入“独立噪声”这一简单的改动，它在采样时解锁了巨大的设计空间（通过设计调度矩阵 $\boldsymbol{\kappa}$），从而衍生出稳定推演、MCG 等强大的新能力。这表明，在模型设计中，有时为训练过程引入更多的随机性和多样性，可以换来推理时更强的泛化能力和可控性。

*   **潜在问题与改进方向:**
    *   **训练开销:** 虽然作者提到计算开销不大，但在实践中，为序列中的每个词元独立采样和处理不同的噪声水平，可能会比全序列扩散（所有词元共享噪声水平和计算）带来更大的计算和内存负担，尤其是在使用 Transformer 架构和长序列时。未来可以研究更高效的实现方式，例如对噪声水平进行分组或采样。
    *   **调度矩阵 $\boldsymbol{\kappa}$ 的设计:** 本文展示了 $\boldsymbol{\kappa}$ 矩阵的强大作用，但如何为特定任务自动或自适应地设计最优的 $\boldsymbol{\kappa}$ 矩阵，仍然是一个开放问题。这可能成为一个有趣的研究方向，例如通过元学习或强化学习来搜索最佳采样策略。
    *   **在离散数据上的应用:** 论文的实验主要集中在连续数据（视频、状态、时间序列值）上。虽然理论上可以应用于文本等离散数据，但其在离散空间扩散模型上的具体表现和优势有待进一步验证。例如，MCG 在文本生成中是否能有效引导生成具有特定情感或主题的长文，将是一个值得探索的课题。