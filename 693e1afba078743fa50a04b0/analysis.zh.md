# 1. 论文基本信息

## 1.1. 标题
**One-step Diffusion with Distribution Matching Distillation**
(基于分布匹配蒸馏的单步扩散模型)

论文标题直接点明了研究的核心：通过一种名为 <strong>分布匹配蒸馏 (Distribution Matching Distillation, DMD)</strong> 的技术，将需要多次迭代采样的扩散模型（Diffusion Models）压缩成一个<strong>单步 (One-step)</strong> 即可生成高质量图像的模型。

## 1.2. 作者
*   **作者列表:** Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T. Freeman, Taesung Park
*   **隶属机构:**
    *   Massachusetts Institute of Technology (MIT) - 麻省理工学院
    *   Adobe Research - Adobe研究院
*   **背景分析:** 作者团队由学术界顶尖学府（MIT）和业界领先的研究机构（Adobe）的研究人员组成。这种产学研结合的背景通常意味着研究不仅具有理论深度，也高度关注实际应用和落地效果，例如图像生成速度和质量，这与Adobe在创意软件领域的地位相符。

## 1.3. 发表期刊/会议
*   **发表平台:** arXiv (预印本)
*   **发表状态:** 本文是一篇预印本论文，发表于 2023 年 11 月 30 日。预印本意味着它未经同行评审，但这是机器学习领域为了快速交流研究成果的常见做法。

## 1.4. 发表年份
2023

## 1.5. 摘要
扩散模型虽然能生成高质量图像，但其迭代采样过程非常耗时，通常需要几十次网络前向传播。为了解决此问题，本文提出了一种名为 <strong>分布匹配蒸馏 (Distribution Matching Distillation, DMD)</strong> 的新方法，旨在将一个多步扩散模型转化为一个单步图像生成器，同时尽可能不损失图像质量。该方法的核心思想是：强制单步生成器在**分布层面**与原始扩散模型保持一致。具体而言，通过最小化一个近似的 <strong>KL散度 (KL divergence)</strong> 来实现。该KL散度的梯度可以巧妙地表示为两个<strong>分数函数 (score functions)</strong> 之差：一个来自目标分布（即原始多步模型生成的图像分布），另一个来自单步生成器当前产生的合成图像分布。这两个分数函数分别由两个在各自数据分布上训练的扩散模型来参数化。此外，为了保证生成图像的大尺度结构与原始模型一致，DMD还引入了一个简单的<strong>回归损失 (regression loss)</strong>。实验结果表明，该方法在少步数（尤其是单步）扩散生成领域超越了所有已发表的方法，在 ImageNet 64x64 数据集上取得了 2.62 的 FID 分数，在零样本 COCO-30k 测试上取得了 11.49 的 FID，其性能与强大的 Stable Diffusion 相当，但速度却快了几个数量级。在FP16精度下，该模型能在现代硬件上达到每秒20帧 (20 FPS) 的生成速度。

## 1.6. 原文链接
*   **原文链接:** https://arxiv.org/abs/2311.18828
*   **PDF 链接:** https://arxiv.org/pdf/2311.18828v4.pdf

# 2. 整体概括

## 2.1. 研究背景与动机
*   **核心问题:** 扩散模型已成为图像生成领域的标杆，以其卓越的生成质量和多样的结果而闻名。然而，其最大的短板在于**生成速度**。标准的扩散模型采样过程是迭代式的，需要从纯噪声开始，通过几十甚至上百次调用神经网络进行逐步去噪，才能得到最终的清晰图像。这个过程计算成本高昂，严重限制了其在需要实时交互的场景中的应用，例如创意设计工具、实时预览等。

*   **重要性与挑战:** 加速扩散模型的推理过程是当前生成模型领域的一个关键研究方向。现有的加速方法主要分为两类：
    1.  <strong>快速采样器 (Fast Samplers):</strong> 设计更高效的数值求解器，在不改变模型的情况下减少采样步数。但当步数减少到极低（如少于10步）时，图像质量会急剧下降。
    2.  <strong>模型蒸馏 (Model Distillation):</strong> 训练一个“学生”模型，让它学会用更少的步数（甚至一步）来模拟原始“教师”模型（多步扩散模型）的生成结果。这种方法的挑战在于，让学生模型精确拟合从一个高维噪声到一张复杂图像的映射函数极其困难，且为学生模型计算一次损失就需要完整运行一次昂贵的教师模型，训练成本高昂。因此，现有蒸馏模型的性能通常仍落后于原始的多步模型。

*   **创新切入点:** 论文作者另辟蹊径，提出了一种全新的蒸馏思路。他们认为，没有必要强制学生模型复现教师模型<strong>“从特定噪声到特定图像”</strong>的一一对应关系。更重要的是，学生模型生成的<strong>图像集合 (分布)</strong> 应该与教师模型生成的<strong>图像集合 (分布)</strong> 在统计上无法区分。换言之，只要学生生成的图像看起来和教师生成的同样“真实”和“多样”，就达到了目的。这个“匹配分布”而非“匹配实例”的思想，是本文最核心的创新。

## 2.2. 核心贡献/主要发现
*   **提出了DMD方法:** 本文提出了 <strong>分布匹配蒸馏 (Distribution Matching Distillation, DMD)</strong>，一个全新的将多步扩散模型蒸馏为单步生成器的框架。

*   **创新的分布匹配损失:** DMD的核心是一个基于近似 <strong>KL散度 (KL divergence)</strong> 的分布匹配损失。其梯度被巧妙地表示为两个<strong>分数函数 (score functions)</strong> 的差值。这一形式将 GAN 中“对抗”的思想与分数匹配的理论相结合，用一个可微的、有方向的梯度信号（“更真实一点”和“更不假一点”）来指导生成器的训练，而非传统 GAN 中简单的二元判别信号。

*   **双扩散模型估计分数:** 本文提出使用两个扩散模型来分别估计真实分布（教师模型输出）和虚假分布（学生模型输出）的分数。其中，真实分数模型是固定的预训练教师模型，而虚假分数模型则在训练过程中动态学习，以追踪学生模型生成分布的变化。

*   **回归损失稳定训练:** 作者发现，单纯的分布匹配损失可能会导致模式坍塌（即生成器只学会了生成教师模型输出的一部分模式）。为了解决这个问题，他们引入了一个简单的**回归损失**作为正则化项。该损失在一个预先生成的小型配对数据集（噪声-图像对）上计算，强制学生模型的输出在宏观结构上与教师模型对齐，有效提升了训练的稳定性和生成结果的多样性。

*   **SOTA的性能:** 实验结果表明，DMD 在单步生成任务上取得了<strong>最先进的 (state-of-the-art)</strong> 性能，在多个标准测试集上显著优于之前的少步数生成方法（如一致性模型），并且其生成质量非常接近原始的多步 Stable Diffusion 模型，但速度快了几个数量级，实现了实时生成。

# 3. 预备知识与相关工作

## 3.1. 基础概念
### 3.1.1. 扩散模型 (Diffusion Models)
扩散模型是一类生成模型，其工作原理包含两个过程：
*   <strong>前向过程 (Forward Process):</strong> 这是一个固定的过程。从一张真实的、清晰的图像 $x_0$ 开始，在 $T$ 个时间步内逐步地向图像中添加高斯噪声。当时间步 $t$ 足够大时（如 $T=1000$），原始图像 $x_0$ 会变成一张纯粹的、无规律的高斯噪声图像 $x_T$。
*   <strong>反向过程 (Reverse Process):</strong> 这是模型需要学习的过程。模型的目标是逆转前向过程，即从一张纯噪声图像 $x_T$ 开始，通过一个神经网络（通常是 U-Net 架构）在每个时间步 $t$ 预测并去除噪声，逐步恢复出清晰的图像 $x_0$。这个去噪网络 $\mu(x_t, t)$ 就是扩散模型的核心。由于这个过程需要从 $T$ 到 `0` 迭代执行，所以生成一张图像非常耗时。

### 3.1.2. 知识蒸馏 (Knowledge Distillation)
这是一种模型压缩技术。其核心思想是，用一个已经训练好的、性能强大但结构复杂的大模型（称为“教师模型”）来指导一个小模型（称为“学生模型”）的训练。目标是让学生模型以更小的计算代价（如更少的层、更少的参数、更少的推理步数）来逼近教师模型的性能。

### 3.1.3. 分数函数 (Score Function)
在统计学中，一个概率密度函数 `p(x)` 的分数函数被定义为其对数概率对输入 $x$ 的梯度，即 $s(x) = \nabla_x \log p(x)$。这个向量直观地指向了数据概率密度增长最快的方向。也就是说，如果有一个数据点 $x$，沿着分数函数 `s(x)` 的方向移动它，它会变得“更像”来自 `p(x)` 分布的样本。扩散模型的一个重要理论解释是，其训练的去噪网络实质上是在估计加噪后数据分布的分数函数。

### 3.1.4. KL散度 (Kullback-Leibler Divergence)
KL散度，也称为相对熵，是衡量两个概率分布之间差异的一种方式。对于两个概率分布 $P$ 和 $Q$，从 $P$ 到 $Q$ 的KL散度记为 $D_{KL}(P \| Q)$。它衡量的是，如果我们用分布 $Q$ 来近似真实分布 $P$，会产生多少信息损失。$D_{KL}(P \| Q) = 0$ 当且仅当 $P$ 和 $Q$ 是相同的分布。在机器学习中，最小化两个分布间的KL散度是让一个模型分布（如生成器输出的分布）去拟合一个目标分布（如真实数据分布）的常用方法。

## 3.2. 前人工作
*   <strong>扩散模型加速 (Diffusion Acceleration):</strong>
    *   <strong>快速采样器 (Fast Samplers):</strong> 如 `DDIM` [72], `DPM-Solver` [45, 46], `UniPC` [91] 等。它们通过设计更高级的常微分方程（ODE）求解器来用更大的步长近似去噪轨迹，从而将采样步数从几百步减少到几十步。但步数过少时，近似误差增大，质量下降明显。
    *   <strong>模型蒸馏 (Model Distillation):</strong>
        *   <strong>渐进式蒸馏 (Progressive Distillation, PD)</strong> [65]: 采用一种迭代的蒸馏策略。每一次蒸馏都将采样步数减半。例如，先训练一个学生模型用50步模拟原始100步的效果，然后再训练一个新学生用25步模拟这个50步模型的效果，以此类推，最终得到一个少步数模型。
        *   <strong>一致性模型 (Consistency Models, CM)</strong> [75]: 强制模型在同一条去噪轨迹上的任意两个点的输出应该是一致的（都指向轨迹的终点，即清晰图像）。通过这种自洽性约束进行训练，模型可以直接从噪声一步映射到清晰图像。
        *   **Rectified Flow** [42]: 学习将噪声和图像之间的线性插值路径“拉直”，使得一步预测的误差更小。

*   <strong>分布匹配方法 (Distribution Matching):</strong>
    *   <strong>生成对抗网络 (Generative Adversarial Networks, GANs)</strong> [15]: GANs 是分布匹配的经典范例。它通过一个生成器 (Generator) 和一个判别器 (Discriminator) 的对抗游戏来学习数据分布。生成器试图生成以假乱真的图像，而判别器则努力区分真实图像和生成图像。这种对抗训练迫使生成器的输出分布接近真实数据分布。
    *   <strong>变分分数蒸馏 (Variational Score Distillation, VSD)</strong> [80]: 这项工作对本文启发很大。VSD 利用一个预训练的文本到图像扩散模型作为“判别器”，为3D场景的优化提供损失。它证明了可以用扩散模型的分数来衡量生成样本与目标分布的差异。但VSD主要用于优化一个静态的3D表示（NeRF），而不是训练一个能快速生成大量不同样本的神经网络生成器。

## 3.3. 技术演进
图像生成技术经历了从 VAEs、GANs 到扩散模型的演进。扩散模型在质量上达到了顶峰，但速度成为新的瓶颈。因此，研究焦点转向了如何**在保持高质量的同时提升生成速度**。
1.  <strong>早期扩散模型 (DDPM):</strong> 需要上千步采样，速度极慢。
2.  **快速采样器:** 将步数降至几十步，实现了可用性，但仍非实时。
3.  **模型蒸馏:** 目标是实现极少步数（1-4步）甚至单步生成。
    *   `PD`, `CM` 等方法通过复杂的训练策略（如迭代训练、轨迹一致性）来蒸馏模型。
    *   本文的 `DMD` 则提出了一种新的、更直接的分布匹配思路，将 `VSD` 的思想从优化单个场景扩展到训练整个生成网络，并结合了经典蒸馏中的回归损失来保证稳定性。

## 3.4. 差异化分析
*   **与传统蒸馏方法的区别:** 传统蒸馏方法（如 `Progressive Distillation` 和 `Consistency Models`）通常关注于<strong>实例级别 (instance-level)</strong> 的匹配，即让学生模型在特定的输入噪声下，输出尽可能接近教师模型的图像。DMD 的核心是<strong>分布级别 (distribution-level)</strong> 的匹配，它不强求一一对应，而是要求学生模型生成的图像整体分布与教师模型一致。这给了模型更大的自由度，可能更容易学习。

*   **与GANs的区别:** GANs 使用一个二元分类器（判别器）来判断真假，提供的梯度信号相对简单。DMD 使用两个强大的扩散模型来估计分数函数，提供的是一个更丰富的、有方向的梯度信号，即“如何修改这张图像让它更真实/更不假”。这利用了预训练扩散模型中蕴含的丰富先验知识，训练过程可能比GAN更稳定。

*   **与VSD的区别:** VSD 的目标是优化一个特定的3D场景参数，属于“测试时优化”。DMD 的目标是训练一个通用的**神经网络生成器** $G_\theta$，该网络训练完成后可以快速生成任意多的新样本。DMD 将 VSD 的核心损失函数应用到了一个全新的场景——训练生成模型。

# 4. 方法论

本部分将详细拆解论文提出的 <strong>分布匹配蒸馏 (Distribution Matching Distillation, DMD)</strong> 方法。其目标是给定一个预训练好的、多步采样的教师扩散模型 $\mu_{base}$，训练一个单步生成器 $G_{\theta}$，使其能够快速生成高质量图像。

## 4.1. 方法原理
DMD 的核心直觉是：我们不需要让单步生成器 $G_{\theta}$ 完美复刻教师模型从某个噪声 $z$ 到特定图像 $y$ 的复杂映射关系。我们只需要保证 $G_{\theta}$ 生成的图像分布 $p_{fake}$ 与教师模型生成的图像分布 $p_{real}$ 难以区分即可。

为了实现这一目标，DMD 框架主要由两部分组成：
1.  <strong>分布匹配损失 ($\mathcal{L}_{KL}$):</strong> 这是方法的核心。它通过最小化 $p_{fake}$ 和 $p_{real}$ 之间的 KL 散度，在分布层面上拉近两者。
2.  <strong>回归损失 ($\mathcal{L}_{reg}$):</strong> 这是一个辅助损失。它在一个小的配对数据集上进行实例级别的匹配，充当正则化器，防止模式坍塌并稳定训练。

    下图（原文 Figure 2）展示了DMD方法的整体流程。

    ![Figure 2. Method overview. We train one-step generator $G _ { \\theta }$ to map random noise $z$ into a realistic image. To match the multi-step distribution matching gradient $\\nabla _ { \\boldsymbol { \\theta } } \\overline { { \\boldsymbol { D } _ { K L } } }$ to the fake image to enhance realism. We inject a random amount of noise to the fake image and the one-step generator.](images/2.jpg)
    *该图像是示意图，展示了一步生成器 $G_{θ}$ 的结构及其与配对数据集之间的关系。图中包含分布匹配梯度计算过程，重点在于如何通过 $∇_{θ}D_{KL}$ 来增强生成图像的真实感，并引入随机噪声以优化假图像与真实图像之间的分布匹配。*

## 4.2. 核心方法详解 (逐层深入)

### 4.2.1. 预备模型设定
*   <strong>教师模型 ($\mu_{base}$):</strong> 一个预训练好的扩散模型去噪器，如 Stable Diffusion 或 EDM。它可以在给定的噪声图像 $x_t$ 和时间步 $t$ 下，预测出对应的清晰图像。
*   <strong>单步生成器 ($G_{\theta}$):</strong> 结构与教师模型 $\mu_{base}$ 的去噪网络相同，但不接受时间步 $t$ 作为输入。其参数 $\theta$ 由教师模型的权重初始化（在最后一个时间步 `T-1` 的表现）。

### 4.2.2. 分布匹配损失 (Distribution Matching Loss)

**1. 理论目标：最小化KL散度**

理想的优化目标是最小化由生成器 $G_{\theta}$ 产生的虚假分布 $p_{fake}$ 与教师模型产生的真实分布 $p_{real}$ 之间的 KL 散度：

$$
\begin{array} { r l } & { D _ { K L } \left( p _ { \mathrm { f a k e } } \parallel p _ { \mathrm { r e a l } } \right) = \underset { x \sim p _ { \mathrm { f a k e } } } { \mathbb { E } } \left( \log \left( \frac { p _ { \mathrm { f a k e } } ( x ) } { p _ { \mathrm { r e a l } } ( x ) } \right) \right) } \\ & { \qquad = \underset { z \sim \mathcal { N } ( 0 ; \mathbf { I } ) } { \mathbb { E } } - \left( \log \mathbf { \delta } p _ { \mathrm { r e a l } } ( x ) - \log \mathbf { \delta } p _ { \mathrm { f a k e } } ( x ) \right) . } \\ & { \qquad \quad x = G _ { \theta } ( z ) } \end{array}
$$

*   **公式解释:**
    *   $D_{KL}(p_{fake} \| p_{real})$: 从 $p_{fake}$ 到 $p_{real}$ 的KL散度。
    *   $\underset{x \sim p_{fake}}{\mathbb{E}}[\cdot]$: 对生成器产生的所有样本 $x$ 求期望。
    *   $x = G_{\theta}(z)$: 样本 $x$ 是由生成器 $G_{\theta}$ 从标准高斯噪声 $z$ 生成的。
    *   $\log p_{fake}(x)$ 和 $\log p_{real}(x)$: 分别是虚假分布和真实分布的对数概率密度。

        直接计算这个损失是不可行的，因为我们无法得到 $p_{fake}(x)$ 和 $p_{real}(x)$ 的解析形式。但我们可以通过梯度下降来优化生成器 $G_{\theta}$，因此我们只需要计算该损失对参数 $\theta$ 的梯度。

**2. 梯度与分数函数**

对上述 KL 散度求关于 $\theta$ 的梯度，可以得到：

$$
\nabla _ { \theta } D _ { K L } = \underset { z \sim \mathcal { N } ( 0 ; \mathbf { I } ) } {\mathbb{E}} \Big [ - \big ( s _ { \mathrm { r e a l } } ( x ) - s _ { \mathrm { f a k e } } ( x ) \big ) \frac { d G } { d \theta } \Big ] ,
$$

*   **公式解释:**
    *   $\nabla_{\theta} D_{KL}$: KL散度对生成器参数 $\theta$ 的梯度。
    *   $s_{real}(x) = \nabla_x \log p_{real}(x)$: 真实分布的分数函数。
    *   $s_{fake}(x) = \nabla_x \log p_{fake}(x)$: 虚假分布的分数函数。
    *   $\frac{dG}{d\theta}$: 生成器 $G_{\theta}(z)$ 对参数 $\theta$ 的雅可比矩阵。

*   **直观理解:** 这个梯度更新公式的含义是，对于一个生成的样本 $x$，计算它的真实分数 $s_{real}(x)$ 和虚假分数 $s_{fake}(x)$。梯度更新的方向是 $(s_{real}(x) - s_{fake}(x))$。
    *   $s_{real}(x)$ 会将 $x$ 推向真实数据密度更高的区域（使其“更真实”）。
    *   $-s_{fake}(x)$ 会将 $x$ 推离虚假数据密度更高的区域（使其“更不假”，有助于避免模式坍塌，增加多样性）。

**3. 利用扩散模型估计分数**

然而，直接计算 $s_{real}(x)$ 和 $s_{fake}(x)$ 仍然困难，原因有二：
1.  当生成样本 $x$ 质量很差时，它在真实分布 $p_{real}$ 中的概率趋近于零，导致分数函数 $s_{real}(x)$ 发散。
2.  我们没有 $p_{real}$ 和 $p_{fake}$ 的显式表达。

    论文巧妙地利用扩散模型的理论来解决这个问题。核心思想是：**我们不直接在清晰图像上计算分数，而是在加噪后的图像上计算。**
首先，对生成器输出的清晰图像 $x = G_{\theta}(z)$ 添加高斯噪声，得到加噪图像 $x_t$：

$$
q _ { t } ( x _ { t } | x ) \sim \mathcal { N } ( \alpha _ { t } x ; \sigma _ { t } ^ { 2 } \mathbf { I } ) ,
$$

*   **公式解释:**
    *   $x_t$: 在时间步 $t$ 的加噪图像。
    *   $\alpha_t, \sigma_t$: 与扩散过程时间步 $t$ 相关的噪声调度系数。

        加噪使得真实分布和虚假分布的支撑域扩展到整个空间并相互重叠（如下图原文 Figure 4 所示），解决了分数发散的问题。

        ![Figure 4. Without perturbation, the real/fake distributions may not overlap (a). Real samples only get a valid gradient from the real score, and fake samples from the fake score. After diffusion (b), our distribution matching objective is well-defined everywhere.](images/4.jpg)
        *该图像是示意图，展示了在未扰动分布下（a），真实与虚假分布的得分可能无法同时定义的情况；而在扩散后（b），这两个分布重叠，使我们的目标更加明确定义。*

根据分数匹配理论，一个在数据分布上训练的去噪扩散模型 $\mu(x_t, t)$ 可以用来估计该数据分布在加噪后的分数函数。

*   <strong>真实分数 ($s_{real}$):</strong> 使用固定的教师模型 $\mu_{base}$ 来估计。
    $$
    s _ { \mathrm { r e a l } } ( x _ { t } , t ) = - \frac { x _ { t } - \alpha _ { t } \mu _ { \mathrm { b a s e } } ( x _ { t } , t ) } { \sigma _ { t } ^ { 2 } } .
    $$

*   <strong>虚假分数 ($s_{fake}$):</strong> 这是DMD的另一个关键创新。作者引入了第二个扩散模型 $\mu_{fake}^{\phi}$，它的任务是在训练过程中**动态地学习**当前生成器 $G_{\theta}$ 所产生的虚假分布。$\mu_{fake}^{\phi}$ 的训练目标是一个标准的去噪损失，其训练数据是 $G_{\theta}$ 生成的样本：
    $$
    \mathcal { L } _ { \mathrm { d e n o i s e } } ^ { \phi } = | | \mu _ { \mathrm { f a k e } } ^ { \phi } ( x _ { t } , t ) - x _ { 0 } | | _ { 2 } ^ { 2 } ,
    $$
    其中 $x_0 = G_{\theta}(z)$ 是生成器产生的“干净”图像。这个虚假分数模型 $\mu_{fake}^{\phi}$ 因此可以估计出虚假分布的分数：
    $$
    s _ { \mathrm { f a k e } } ( x _ { t } , t ) = - \frac { x _ { t } - \alpha _ { t } \mu _ { \mathrm { f a k e } } ^ { \phi } ( x _ { t } , t ) } { \sigma _ { t } ^ { 2 } } .
    $$

**4. 最终的分布匹配梯度**

将用扩散模型估计出的分数代入梯度公式，并对时间步 $t$ 求期望，得到最终可计算的分布匹配梯度：

$$
\nabla _ { \theta } D _ { K L } \simeq \underset { z , t , x , x _ { t } } { \mathbb { E } } \left[ w _ { t } \alpha _ { t } \left( s _ { \mathrm { f a k e } } ( x _ { t } , t ) - s _ { \mathrm { r e a l } } ( x _ { t } , t ) \right) \frac { d G } { d \theta } \right] ,
$$

*   **公式解释:**
    *   $z \sim \mathcal{N}(0; \mathbf{I})$, $x = G_{\theta}(z)$, $t \sim \mathcal{U}(T_{min}, T_{max})$, $x_t \sim q_t(x_t|x)$: 梯度是在随机噪声、随机时间步和对应的加噪图像上计算的期望。
    *   $w_t$: 一个时间步相关的权重因子，用于平衡不同噪声水平下的梯度大小，以稳定训练。其具体形式为：
        $$
        w _ { t } = \frac { \sigma _ { t } ^ { 2 } } { \alpha _ { t } } \frac { C S } { \lvert \lvert \mu _ { \mathrm { b a s e } } ( x _ { t } , t ) - x \rvert \rvert _ { 1 } } ,
        $$
        其中 $C$ 是通道数，$S$ 是空间像素数。这个权重与教师模型在当前噪声水平下的去噪误差成反比，当误差大（去噪难）时，权重变小，反之亦然。

### 4.2.3. 回归损失 (Regression Loss)
作者发现，仅使用分布匹配损失有时会使训练不稳定或导致<strong>模式坍塌 (mode collapse)</strong>——即生成器只学会了生成教师模型输出的一部分高频模式，而忽略了其他模式。如下图（原文 Figure 3）所示，仅有分布匹配损失时（b），模型只恢复了双峰分布中的一个峰。

![Figure 3. Optimizing various objectives starting from the same configuration (left) leads to different outcomes. (a) Maximizing the real score only, the fake samples all collapse to the closest mode of the real distribution. (b) With our distribution matching objective but not regression loss, the generated fake data covers more of the real distribution, but only recovers the closest mode, missing the second mode entirely. (c) Our full objective, with the regression loss, recovers both modes of the target distribution.](images/3.jpg)
*该图像是一个示意图，展示了不同优化目标下生成的结果。左侧为初始状态，(a) 仅优化真实分数，假样本塌缩至真实分布的最近模式；(b) 结合真实与假样本分数，生成数据更广泛，但仅恢复最近的模式；(c) 完整目标结合回归损失，成功恢复目标分布的两个模式。*

为了解决这个问题，DMD 引入了一个简单的回归损失 $\mathcal{L}_{reg}$ 作为正则化项。
1.  **构建配对数据集 $\mathcal{D}$:** 首先，离线预先生成一个规模不大的数据集 $\mathcal{D} = \{ (z_i, y_i) \}_{i=1}^N$。其中，$z_i$ 是随机高斯噪声，$y_i$ 是用教师模型 $\mu_{base}$ 通过确定性 ODE 求解器从 $z_i$ 生成的对应图像。
2.  **计算回归损失:** 在训练时，从 $\mathcal{D}$ 中采样一批 `(z, y)`，计算生成器 $G_{\theta}(z)$ 的输出与预存的教师输出 $y$ 之间的距离。
    $$
    \mathcal { L } _ { \mathrm { r e g } } = \underset { ( z , y ) \sim \mathcal { D } } { \mathbb { E } } \ell ( G _ { \theta } ( z ) , y ) .
    $$
*   **公式解释:**
    *   $\ell(\cdot, \cdot)$: 距离函数。论文中使用了 **LPIPS (Learned Perceptual Image Patch Similarity)**，这是一种基于深度特征的感知损失，比 L2 损失更符合人类的视觉感知。

        这个损失强制生成器在某些固定的噪声输入上，其输出的宏观结构和内容要与教师模型保持一致，从而有效地保留了教师模型生成结果的全局多样性，起到了稳定训练和防止模式坍塌的作用。

### 4.2.4. 最终目标与训练流程
最终的训练目标是分布匹配损失和回归损失的加权和。生成器 $G_{\theta}$ 的总损失为 $\mathcal{L}_G = \mathcal{L}_{KL} + \lambda_{reg} \mathcal{L}_{reg}$。同时，虚假分数模型 $\mu_{fake}^{\phi}$ 也在同步更新。

<strong>DMD 训练算法流程 (基于原文 Algorithm 1):</strong>

1.  **初始化:**
    *   生成器 $G$: 用教师模型 $\mu_{real}$ 的权重初始化。
    *   虚假分数模型 $\mu_{fake}$: 同样用教师模型 $\mu_{real}$ 的权重初始化。
    *   准备好预计算的配对数据集 $\mathcal{D} = \{z_{ref}, y_{ref}\}$。

2.  **进入训练循环:**
    *   **生成图像:**
        *   采样一批用于分布匹配的随机噪声 $z$。
        *   从配对数据集 $\mathcal{D}$ 中采样一批参考对 $(z_{ref}, y_{ref})$。
        *   通过生成器得到虚假图像 `x = G(z)` 和参考图像的生成结果 $x_{ref} = G(z_{ref})$。

    *   **更新生成器 $G$:**
        *   **计算分布匹配损失 $\mathcal{L}_{KL}$:** 对 $x$ (以及 $x_{ref}$) 加噪，然后使用 $\mu_{real}$ 和 $\mu_{fake}$ 计算分数差，得到梯度并构造损失（如原文 Algorithm 2 所示）。
        *   **计算回归损失 $\mathcal{L}_{reg}$:** 计算 $x_{ref}$ 和 $y_{ref}$ 之间的 LPIPS 距离。
        *   **总损失:** $\mathcal{L}_G = \mathcal{L}_{KL} + \lambda_{reg} \mathcal{L}_{reg}$。
        *   使用 $\mathcal{L}_G$ 的梯度更新生成器 $G$ 的参数。

    *   **更新虚假分数模型 $\mu_{fake}$:**
        *   将生成器产生的图像 $x$（已停止梯度回传）作为“干净”图像。
        *   对其进行加噪，得到 $x_t$。
        *   使用标准去噪损失 $\mathcal{L}_{denoise}$（如原文 Algorithm 3 所示）更新 $\mu_{fake}$ 的参数，使其学会对 $x$ 进行去噪。

3.  **重复循环直至收敛。**

# 5. 实验设置

## 5.1. 数据集
*   **CIFAR-10:** 一个包含10个类别的 32x32 彩色小图像数据集，共6万张。是图像生成领域的经典入门级基准。
*   **ImageNet:** 一个大规模图像识别数据集。本文中使用的是 64x64 分辨率的子集，用于类别条件生成任务。
*   **MS COCO 2014:** 一个包含复杂场景和多种物体的大规模数据集。本文没有直接在该数据集上训练，而是在 LAION 数据集上训练模型，然后在 MS COCO 的 30,000 个标题（caption）上进行<strong>零样本 (zero-shot)</strong> 评估，即模型在训练时没见过这些标题。
*   **LAION-Aesthetics:** LAION 是一个巨大的图文对数据集。本文使用了其中的两个子集 `LAION-Aesthetics-6.25+` 和 `LAION-Aesthetics-6+`，这些子集根据美学评分进行了筛选，包含了更高质量的图像，适合用于训练文本到图像模型。

    选择这些数据集是为了在不同规模、不同分辨率和不同任务（无条件生成、类别条件生成、文本到图像生成）上全面地验证 DMD 方法的有效性和泛化能力。

## 5.2. 评估指标
### 5.2.1. FID (Fréchet Inception Distance)
*   **概念定义:** FID 是一种广泛用于评估生成模型图像质量和多样性的指标。它通过比较真实图像集和生成图像集在 InceptionV3 网络特征空间中的统计分布来衡量两者的相似度。具体来说，它计算两个分布的均值和协方差矩阵之间的 Fréchet 距离。**FID 分数越低，表示生成图像的分布与真实图像的分布越接近，即生成图像的质量和多样性越好。**

*   **数学公式:**
    $$
    \mathrm{FID}(x, g) = ||\mu_x - \mu_g||_2^2 + \mathrm{Tr}(\Sigma_x + \Sigma_g - 2(\Sigma_x \Sigma_g)^{1/2})
    $$

*   **符号解释:**
    *   $x$: 真实图像集。
    *   $g$: 生成图像集。
    *   $\mu_x, \mu_g$: 真实图像和生成图像在 InceptionV3 网络某一层激活的特征向量的均值。
    *   $\Sigma_x, \Sigma_g$: 真实图像和生成图像特征向量的协方差矩阵。
    *   $||\cdot||_2^2$: 向量的L2范数的平方，计算均值向量之间的距离。
    *   $\mathrm{Tr}(\cdot)$: 矩阵的迹（主对角线元素之和）。

### 5.2.2. CLIP Score
*   **概念定义:** CLIP Score 用于衡量生成的图像与给定的文本描述（prompt）的语义匹配程度。它利用了 OpenAI 的 CLIP (Contrastive Language-Image Pre-Training) 模型，该模型能将图像和文本嵌入到同一个多模态特征空间中。通过计算图像嵌入和文本嵌入之间的余弦相似度，可以量化图像内容与文本描述的关联性。**CLIP Score 越高，表示图像与文本的匹配度越好。**

*   **数学公式:**
    $$\text{CLIP Score} = \text{cosine\_similarity}(\text{Emb}_I(I), \text{Emb}_T(T))$$
    （通常会乘以100）

*   **符号解释:**
    *   $I$: 生成的图像。
    *   $T$: 输入的文本描述。
    *   $\text{Emb}_I(I)$: 使用 CLIP 图像编码器得到的图像特征嵌入。
    *   $\text{Emb}_T(T)$: 使用 CLIP 文本编码器得到的文本特征嵌入。
    *   $\text{cosine\_similarity}(\cdot, \cdot)$: 计算两个向量之间的余弦相似度。

## 5.3. 对比基线
论文将 DMD 与多个具有代表性的模型进行了比较，涵盖了不同类型的生成模型和加速技术：
*   **GANs:** 如 `BigGAN-deep` [4], `StyleGAN-T` [67], `GigaGAN` [26] 等。这些是最先进的GAN模型，以生成速度快著称。
*   <strong>原始多步扩散模型 (Teacher):</strong> 如 `EDM` [31] 和 `Stable Diffusion v1.5` [63]。这是 DMD 方法性能的理论上限（或目标）。
*   **快速扩散采样器:** 如 $DPM++$ [46] 和 `UniPC` [91]。这些方法通过改进求解器来减少步数，是在不改变模型的前提下进行的加速。
*   **其他扩散蒸馏方法:**
    *   `Progressive Distillation` [65]
    *   `Consistency Model` [75]
    *   `TRACT` [3]
    *   `InstaFlow` [43]
    *   `Latent Consistency Models (LCM)` [48, 49]
        这些是与 DMD 直接竞争的、旨在将扩散模型压缩到极少步数（1-4步）的SOTA方法。

# 6. 实验结果与分析

## 6.1. 核心结果分析

### 6.1.1. 类别条件图像生成 (ImageNet)
以下是原文 Table 1 的结果，比较了在 ImageNet 64x64 数据集上不同方法的 FID 分数。

<table>
<thead>
<tr>
<th>Method</th>
<th># Fwd Pass (↓)</th>
<th>FID (↓)</th>
</tr>
</thead>
<tbody>
<tr>
<td>BigGAN-deep [4]</td>
<td>1</td>
<td>4.06</td>
</tr>
<tr>
<td>ADM [9]</td>
<td>250</td>
<td>2.07</td>
</tr>
<tr>
<td colspan="3" style="border-bottom: 1px solid #ccc;"></td>
</tr>
<tr>
<td>Progressive Distillation [65]</td>
<td>1</td>
<td>15.39</td>
</tr>
<tr>
<td>DFNO [92]</td>
<td>1</td>
<td>7.83</td>
</tr>
<tr>
<td>BOOT [16]</td>
<td>1</td>
<td>16.30</td>
</tr>
<tr>
<td>TRACT [3]</td>
<td>1</td>
<td>7.43</td>
</tr>
<tr>
<td>Meng et al. [51]</td>
<td>1</td>
<td>7.54</td>
</tr>
<tr>
<td>Diff-Instruct [50]</td>
<td>1</td>
<td>5.57</td>
</tr>
<tr>
<td>Consistency Model [75]</td>
<td>1</td>
<td>6.20</td>
</tr>
<tr>
<td><strong>DMD (Ours)</strong></td>
<td><strong>1</strong></td>
<td><strong>2.62</strong></td>
</tr>
<tr>
<td colspan="3" style="border-bottom: 1px solid #ccc;"></td>
</tr>
<tr>
<td>EDM (Teacher) [31]</td>
<td>512</td>
<td>2.32</td>
</tr>
</tbody>
</table>

*   **分析:** 在单步生成 ($# Fwd Pass = 1$) 的设定下，<strong>DMD 的 FID 分数 (2.62) 显著优于所有其他已发表的蒸馏方法</strong>。例如，它比当时非常先进的一致性模型 (Consistency Model) 的 6.20 提升了超过一倍。更引人注目的是，DMD 的单步生成质量已经<strong>非常接近需要 512 步采样的教师模型 EDM (FID 2.32)</strong>，差距仅为 0.3。这表明 DMD 在几乎不损失质量的前提下，实现了超过500倍的速度提升。

    下图（原文 Figure 7）展示了DMD在ImageNet上生成的样本，可见其质量非常高。

    ![Figure 7. One-step samples from our class-conditional model on ImageNet $\\mathrm { F I D } { = } 2 . 6 2$](images/9.jpg)
    *该图像是从我们的类别条件模型生成的单步样本，展示了多种物体和场景，FID为2.62，来源于ImageNet数据集。*

### 6.1.2. 文本到图像生成 (MS COCO)
以下是原文 Table 3 的结果，比较了在 MS COCO 30k 零样本测试集上不同方法的性能。这里评估的是从 Stable Diffusion v1.5 (SDv1.5) 蒸馏得到的模型，指导系数 (guidance scale) 为 3。

<table>
<thead>
<tr>
<th>Family</th>
<th>Method</th>
<th>Resolution (↑)</th>
<th>Latency (↓)</th>
<th>FID (↓)</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="9">Original, unaccelerated</td>
<td>DALL·E [60]</td>
<td>256</td>
<td></td>
<td>27.5</td>
</tr>
<tr>
<td>DALL·E 2 [61]</td>
<td>256</td>
<td>-</td>
<td>10.39</td>
</tr>
<tr>
<td>Imagen [64]</td>
<td>256</td>
<td>9.1s</td>
<td>7.27</td>
</tr>
<tr>
<td colspan="5" style="border-bottom: 1px solid #ccc;"></td>
</tr>
<tr>
<td rowspan="3">GANs</td>
<td>StyleGAN-T [67]</td>
<td>512</td>
<td>0.10s</td>
<td>13.90</td>
</tr>
<tr>
<td>GigaGAN [26]</td>
<td>512</td>
<td>0.13s</td>
<td>9.09</td>
</tr>
<tr>
<td colspan="5" style="border-bottom: 1px solid #ccc;"></td>
</tr>
<tr>
<td rowspan="6">Accelerated diffusion</td>
<td>DPM++ (4 step) [46]†</td>
<td>512</td>
<td>0.26s</td>
<td>22.36</td>
</tr>
<tr>
<td>UniPC (4 step) [91]†</td>
<td>512</td>
<td>0.26s</td>
<td>19.57</td>
</tr>
<tr>
<td>LCM-LoRA (4 step)[49]†</td>
<td>512</td>
<td>0.19s</td>
<td>23.62</td>
</tr>
<tr>
<td>InstaFlow-0.9B [43]</td>
<td>512</td>
<td>0.09s</td>
<td>13.10</td>
</tr>
<tr>
<td>UFOGen [84]</td>
<td>512</td>
<td>0.09s</td>
<td>12.78</td>
</tr>
<tr>
<td><strong>DMD (Ours)</strong></td>
<td><strong>512</strong></td>
<td><strong>0.09s</strong></td>
<td><strong>11.49</strong></td>
</tr>
<tr>
<td colspan="5" style="border-bottom: 1px solid #ccc;"></td>
</tr>
<tr>
<td>Teacher</td>
<td>SDv1.5† [63]</td>
<td>512</td>
<td>2.59s</td>
<td>8.78</td>
</tr>
</tbody>
</table>

*   **分析:** 在更具挑战性的文本到图像生成任务上，DMD 再次展现了其优越性。
    *   **性能:** DMD 的 FID 为 11.49，优于所有其他加速方法，包括4步的快速采样器（FID 约 20）和同为单步的 InstaFlow (13.10)。它甚至超过了之前的SOTA GAN模型 GigaGAN (9.09)（注：此处GigaGAN仍有优势，但DMD已进入同一竞争梯队）。
    *   **与教师模型的差距:** DMD 的 FID (11.49) 与需要50步采样的教师模型 SDv1.5 (8.78) 差距不大，证明了其高质量的蒸馏效果。
    *   **速度:** DMD 的生成延迟为 0.09s，与最快的 GAN 和其他单步方法相当，但比教师模型 SDv1.5 (2.59s) 快了约 30 倍，实现了实时生成（约 11 FPS，论文提到用 FP16 可达 20 FPS）。

        下图（原文 Figure 6）直观地对比了 DMD 与其他方法及教师模型的生成效果。

        ![该图像是展示多种使用一阶扩散生成的图像对比的图表，包括不同模型生成的图像及其相应的生成时间，标注了DMD（1步）、InstaFlow（1步）、LCM（1步）、LCM（2步）、DPM++（4步）和SD（50步）等方法及其生成速度。](images/8.jpg)
        *该图像是展示多种使用一阶扩散生成的图像对比的图表，包括不同模型生成的图像及其相应的生成时间，标注了DMD（1步）、InstaFlow（1步）、LCM（1步）、LCM（2步）、DPM++（4步）和SD（50步）等方法及其生成速度。*

## 6.2. 消融实验/参数分析
为了验证 DMD 方法中各个组件的有效性，作者进行了一系列消融实验。

以下是原文 Table 2 的结果：

<table>
<thead>
<tr>
<th colspan="3">Training loss</th>
<th colspan="2">Sample weighting</th>
</tr>
<tr>
<th></th>
<th>CIFAR</th>
<th>ImageNet</th>
<th></th>
<th>CIFAR</th>
</tr>
</thead>
<tbody>
<tr>
<td>w/o Dist. Matching</td>
<td>3.82</td>
<td>9.21</td>
<td>σt/αt [58]</td>
<td>3.60</td>
</tr>
<tr>
<td>w/o Regress. Loss</td>
<td>5.58</td>
<td>5.61</td>
<td>σt³/αt [58, 80]</td>
<td>3.71</td>
</tr>
<tr>
<td><strong>DMD (Ours)</strong></td>
<td><strong>2.66</strong></td>
<td><strong>2.62</strong></td>
<td><strong>Eq. 8 (Ours)</strong></td>
<td><strong>2.66</strong></td>
</tr>
</tbody>
</table>

*   <strong>分析 (左半部分 - 损失函数消融):</strong>
    *   <strong>去掉分布匹配损失 (w/o Dist. Matching):</strong> 模型退化为一个仅使用回归损失的普通蒸馏模型。在 ImageNet 上，FID 从 2.62 飙升至 9.21。这证明**分布匹配损失是实现高质量生成的关键**。下图（原文 Figure 5a）显示，没有该损失，图像的真实感和结构完整性严重受损。

        ![该图像是插图，展示了通过Distribution Matching Distillation生成的多种图像样例，包括羊、汽车、狗和兔子。每个图像呈现了不同的视觉特征，表明该方法能生成多样化的高质量图像。](images/6.jpg)
        *该图像是插图，展示了通过Distribution Matching Distillation生成的多种图像样例，包括羊、汽车、狗和兔子。每个图像呈现了不同的视觉特征，表明该方法能生成多样化的高质量图像。*

    *   <strong>去掉回归损失 (w/o Regress. Loss):</strong> 在 ImageNet 上，FID 从 2.62 上升到 5.61。这表明回归损失虽然是辅助性的，但对于稳定训练和保证生成多样性至关重要。下图（原文 Figure 5b）显示，没有该损失，模型容易发生模式坍塌（例如，只生成灰色的车）。

        ![该图像是示意图，展示了使用 DMD 方法生成的汽车图像（左）与未使用回归损失生成的汽车图像（右）之间的对比。可以看到，通过 DMD 方法生成的图像在视觉质量上更为出色。](images/7.jpg)
        *该图像是示意图，展示了使用 DMD 方法生成的汽车图像（左）与未使用回归损失生成的汽车图像（右）之间的对比。可以看到，通过 DMD 方法生成的图像在视觉质量上更为出色。*

*   <strong>分析 (右半部分 - 权重策略消融):</strong>
    *   作者对比了他们提出的权重策略 (Eq. 8) 与之前工作 (DreamFusion, ProlificDreamer) 中使用的策略。结果显示，<strong>本文提出的权重策略取得了最低的 FID (2.66)</strong>，证明了其设计的有效性，它能更好地平衡不同噪声水平下的梯度，从而稳定优化过程。

# 7. 总结与思考

## 7.1. 结论总结
本文提出了一种名为 <strong>分布匹配蒸馏 (DMD)</strong> 的高效方法，成功地将计算密集型的多步扩散模型转化为一个高质量的单步图像生成器。其核心贡献在于：
1.  **创新的分布匹配目标:** 摒弃了传统的实例级映射，转而通过最小化KL散度在分布层面进行匹配，为模型蒸馏提供了新的思路。
2.  **优雅的梯度形式:** 将KL散度的梯度表示为真实分数和虚假分数的差值，巧妙地融合了GAN的对抗思想和分数匹配的理论，并利用强大的预训练扩散模型作为分数估计器。
3.  **实用的正则化策略:** 引入简单的回归损失，有效解决了分布匹配可能带来的模式坍塌问题，显著增强了训练的稳定性和生成结果的多样性。
4.  **卓越的性能:** DMD 在多个基准测试中均取得了当前最先进的单步生成性能，其图像质量与多步教师模型相当，但速度快了几个数量级，为扩散模型在实时交互应用中的普及铺平了道路。

## 7.2. 局限性与未来工作
论文作者坦诚地指出了当前方法的局限性：
*   **质量差距:** 尽管已经非常接近，但单步 DMD 模型的生成质量与迭代上百步的教师模型之间仍然存在微小的差距。
*   **训练内存开销:** DMD 的训练过程需要同时在内存中维护三个大型模型（生成器、真实分数模型、虚假分数模型），这导致训练期间的内存占用非常大。
*   **未来方向:** 作者提出，可以探索使用 **LoRA (Low-Rank Adaptation)** 等参数高效微调技术来减少训练时的内存和计算开销。

## 7.3. 个人启发与批判
*   **个人启发:**
    *   **思想的迁移与融合:** DMD 是一个将不同领域思想成功融合的典范。它将 GAN 的“对抗”思想、扩散模型的“分数匹配”理论以及知识蒸馏的框架巧妙地结合在一起，创造出一种全新的、更有效的方法。这启发我们，在解决问题时应积极寻求不同技术范式之间的共通之处。
    *   **理论与实践的平衡:** 该方法结合了理论上优美的分布匹配损失和实践中非常有效的回归损失。这表明，在追求理论完备性的同时，加入一些简单、实用的“技巧”（如正则化项）往往能取得更好的实际效果。
    *   <strong>“判别器”</strong>的新范式: DMD 使用预训练扩散模型作为“分数引擎”，提供比传统GAN判别器更丰富、更结构化的梯度信号。这为如何利用大型预训练模型中蕴含的丰富知识提供了新的思路，即不只是作为生成器，也可以作为一种强大的、可微的“评估器”或“批评家”。

*   **批判性思考:**
    *   **训练复杂性:** DMD 的训练流程相对复杂，需要同时训练生成器和虚假分数模型，并小心地处理两者之间的梯度流。这种“三体问题”（生成器、真实分数、虚假分数）的动态平衡可能对超参数和训练策略较为敏感。
    *   **回归损失的依赖性:** 实验表明，回归损失对于防止模式坍塌至关重要。这在一定程度上说明，分布匹配损失本身可能还不足以完全捕捉到目标分布的所有模式。这到底是该损失的固有缺陷，还是可以通过更好的权重策略或训练技巧来缓解？这是一个值得深入研究的问题。
    *   **数据集构建成本:** 回归损失需要一个预先构建的配对数据集。虽然论文声称这个数据集不大，但生成它仍然需要运行多次昂贵的教师模型。这个一次性成本在面对超大规模数据集时是否依然可以接受？性能对这个配对数据集的大小和质量有多敏感？论文对此未做深入探讨。
    *   **通用性探索:** DMD 的核心思想——用分数差来指导生成模型训练——似乎具有更广泛的适用性。除了用于蒸馏，它是否可以用于从头开始训练一个单步模型（即没有教师模型，只有一个在真实数据上预训练好的固定分数模型）？或者用于其他生成任务，如3D生成、音频合成等？这为未来的研究留下了广阔的空间。