# 分阶段 DMD：通过在子区间内的分数匹配进行少步分布匹配蒸馏

范向宇<sup>1</sup>、裘泽松<sup>1</sup>、吴竹贯<sup>1</sup>、王璠舟<sup>1</sup>、林志倩<sup>1</sup>、任天相<sup>1</sup>、林大华<sup>1</sup>、龚瑞豪<sup>1,2</sup>、杨磊<sup>1,2</sup> <sup>1</sup>商汤科技研究院，<sup>2</sup>北京航空航天大学

## 摘要

摘要分布匹配蒸馏（DMD）将基于评分的生成模型蒸馏为高效的一步生成器，而无需与其教师的采样轨迹建立一一对应关系。然而，有限的模型容量导致一步蒸馏模型在复杂生成任务中表现不佳，例如在文本到视频生成中合成复杂的物体运动。直接将DMD扩展为多步蒸馏增加了内存使用和计算深度，导致不稳定和效率降低。尽管之前的研究提出了随机梯度截断作为潜在解决方案，但我们观察到这显著降低了多步蒸馏模型的生成多样性，使其降至与一步模型相同的水平。为了解决这些局限性，我们提出了阶段DMD，这是一种将阶段性蒸馏的思想与专家混合（MoE）相结合的多步蒸馏框架，降低了学习难度同时增强了模型容量。阶段DMD基于两个关键思想：渐进式分布匹配和子区间内的评分匹配。首先，我们的模型将信噪比（SNR）范围划分为子区间，逐步将模型优化到更高的SNR水平，以更好地捕捉复杂分布。接下来，为确保每个子区间内的训练目标准确无误，我们进行了严谨的数学推导。我们通过蒸馏最先进的图像和视频生成模型来验证阶段DMD，包括Qwen-Image（20亿参数）和Wan2.2（28亿参数）。实验结果表明，阶段DMD在保留关键生成能力的同时，更好地保持了输出多样性。我们将发布我们的代码和模型。

## 1 引言

近年来，最先进的扩散模型在图像和视频生成方面取得了显著进展。在图像生成中，最先进的模型（Wu et al., 2025；OpenAI, 2025；Team, 2025b；Cao et al., 2025；GoogleAI, 2025a；Seedream et al., 2025）展现了精确的提示控制，使得复杂的文本到图像渲染和准确的布局规范成为可能。在视频生成中，这些模型（Wan et al., 2025；Kong et al., 2024；GoogleAI, 2025b；OpenAI, 2024）在动态场景生成方面表现出显著的改进，例如运动中的快速移动物体和复杂的摄像机运动，如自我中心视频。同时，基础模型参数规模和计算需求的增加突显了加速扩散模型采样的重要性。

为了加速扩散模型，已经提出了几种技术，包括无分类器引导（CFG）蒸馏（Meng et al., 2023）、步骤蒸馏（Song et al., 2023; Wang et al., 2024; Salimans & Ho, 2022; Yin et al., 2024a; Luo et al., 2023; Luo, 2024; Zhou et al., 2024; Huang et al., 2024a; Lin et al., 2024; 2025a; Frans et al., 2024; Geng et al., 2025）、SVDQuant（Li*, et al., 2025）、专家组合模型（MoE）（Balaji et al., 2022; Feng et al., 2023; Wantal et al., 2025）和并行计算（Fang et al., 2024）。在这些方法中，基于变分分数蒸馏（VSD）的步骤蒸馏方法，包括diff-instruct（Luo et al., 2023）、DMD（Yin et al., 2024a）、SID（Zhou et al., 2024），通过将模型蒸馏为单步生成器，实现了高质量的生成。然而，单步蒸馏模型的网络容量有限（Lin et al., 2024），限制了它们处理复杂任务的能力，如复杂文本渲染或动态场景生成，这对于这些基础模型的广泛应用至关重要。

![fig 1](images/1.jpg)

Figure 1: Schematic diagram of (a) Few-step DMD (Yin et al., 2024a), (b) Few-step DMD with stochastic gradient truncation strategy (SGTS) (Huang et al., 2025), (c) Phased DMD and (d) Phased DMD with SGTS.   

少步蒸馏在计算成本和生成质量之间取得了平衡（Luo et al., 2025）。然而，如图1a所示，直接将VSD应用于少步蒸馏（Yin et al., 2024a）会引入一些挑战，例如计算图深度增加和内存开销增大。此外，对中间生成步骤缺乏明确的约束会降低训练稳定性，并导致少步模型的性能不理想。为了解决这些问题，黄等人（2025）提出了一种随机梯度截断策略（SGTS），该策略允许多步采样在随机步骤终止，梯度反向传播仅限于最终去噪步骤（见图1b）。这种方法通过对所有中间步骤进行监督，提高了训练收敛性和稳定性，同时通过对非最终步骤的梯度脱离增强了内存效率。然而，SGTS在训练过程中可能仅在一步后终止采样，导致该迭代仅蒸馏出一个步骤生成器。因此，使用SGTS训练的少步生成器的生成多样性降低到了与一步生成器相似的水平。

扩散理论（Song et al., 2020）表明，在从零到无穷的信噪比（SNR）范围内，存在无穷多作为评分估计器的神经网络。在生成过程中，扩散模型表现出不同的时间动态特性（Balaji et al., 2022; Ouyang et al., 2024）。具体来说，低SNR阶段专注于建模图像结构和视频动态，而高SNR阶段则细化视觉细节。在实际应用中，通常在去噪过程中使用单一神经网络，这要求模型能够同时学习和执行多种去噪任务。近期研究（Balaji et al., 2022; Feng et al., 2023; Wan et al., 2025）在扩散模型中引入了MoE架构，通过将不同SNR水平分配给专门的专家，MoE在不增加推理成本的情况下增强了模型的能力和生成性能。在视频生成中，这种性能提升尤为显著（Wan et al., 2025），低SNR专家在捕捉动态内容方面表现突出。在本工作中，我们提出了Phased DMD，这是一种用于少步生成的新型蒸馏框架。我们的方法受到更广泛愿景的启发：通过将复杂任务分解为可学习的阶段，每个阶段自然形成一个专家，集体增强模型的能力，类似MoE。我们的方法基于两个关键组件：- 渐进分布匹配：在概念上类似于ProGAN（Karras et al., 2017），Phased DMD将SNR划分为子区间，并逐步蒸馏模型以应对更高的SNR水平。- SNR子区间内的评分匹配：由于每个阶段在一个子区间内训练，训练目标会发生变化。为了确保理论严谨性，我们推导了每个子区间内假评分估计器的训练目标。如图1c所示，Phased DMD提供了几个优点：首先，通过将SNR划分为子区间，模型逐步学习复杂数据分布，从而提高训练稳定性和生成性能。其次，每个阶段只涉及一步单梯度记录的采样步骤，避免了额外的计算和内存开销。第三，值得注意的是，Phased DMD自然生成一个少步MoE生成模型，无论教师模型是否采用MoE架构。最后，如图1d所示，Phased DMD可以与SGTS结合，实现2个阶段的4步推理，同时简化了训练和推理的复杂性。我们通过对SOTA图像和视频生成模型进行蒸馏来验证Phased DMD，包括20B参数的Qwen-Image（Wu et al., 2025）和14/28B参数的Wan2.1/Wan2.2（Wan et al., 2025）。实验结果表明，Phased DMD在保持基础模型关键能力的同时，能够更好地保持输出多样性，例如Qwen-Image中的真实文本渲染和Wan2.2中的真实动态运动。我们的贡献总结如下：：我们提出了Phased DMD，这是一种无数据的少步扩散模型蒸馏框架。该框架结合了DMD和MoE的思想，达到更高的性能上限，同时保持与单步蒸馏相似的内存使用。：我们推导了子区间扩散模型的理论训练目标，而无需依赖干净样本等外部信息。我们强调了这种正确性对DMD蒸馏的必要性。: Phased DMD在不需要GAN损失或回归损失的情况下，在文本到图像和文本到视频生成模型上取得了SOTA结果。据我们所知，这是报道的最大蒸馏验证。实验结果表明，我们的方法有效减少了多样性损失，同时保留了基础模型的关键能力，包括复杂文本渲染和高动态视频生成。

## 2 方法

为了阐明相位 DMD 的原理，我们首先介绍与扩散模型（Kingma 等，2023；Zhang 等，2024）、分数匹配（Song 等，2020；Karras 等，2022）以及分布匹配蒸馏（Yin 等，2024b；a）相关的理论背景和符号。我们明确强调 DMD 原理为何仅适用于基于分数的生成模型。在此基础上，我们提出 Phased DMD 的动机，并解释其如何本质上实现生成多样性的提高。接下来，我们详细介绍 Phased DMD 的两个关键组成部分：渐进式分布匹配和子网络内的分数匹配。

## 2.1 初步研究

#### 2.1.1 扩散模型与分数匹配

考虑在区间 \(0 \leq t \leq 1\) 上定义的连续时间高斯扩散过程。真实分布表示为 \(p(x_{0})\)。对于任何 \(0 \leq t \leq 1\)，前向扩散过程由以下条件分布描述：

\[p(\pmb {x}_t|\pmb {x}_0) = \mathcal{N}(\pmb {x}_t;\alpha_t\pmb {x}_0,\sigma_t^2\pmb {I}) \quad (1)\]  

其中 \(\sigma_{t}\) 和 \(\sigma_{t}^{2}\) 是与 \(t\) 相关的正标量值函数。信噪比（SNR）定义为 \(\mathrm{SNR}(t) = \alpha_{t}^{2} / \sigma_{t}^{2}\)。假设 \(\mathrm{SNR}(t)\) 随时间严格单调递减。对 \(\alpha_{t}\) 和 \(\sigma_{t}\) 之间的关系没有额外的约束，从而确保符号与不同类型的扩散模型（Ho et al., 2020; Karras et al., 2022; Song et al., 2022; Podell et al., 2023）和流模型（Liu et al., 2022; Esser et al., 2024）兼容。扩散过程是马尔可夫过程（Kingma et al., 2023），这意味着 \(p\big(\pmb {x}_t|\pmb {x}_s,\pmb {x}_0\big) = p\big(\pmb {x}_t|\pmb {x}_s\big)\)。此外，\(p(\pmb {x}_t|\pmb {x}_s)\) 也是高斯分布，可以表示为：

\[p(x_{t}|x_{s}) = \mathcal{N}(\pmb{x}_{t};\alpha_{t\mid s}x_{s},\sigma_{t\mid s}^{2}\pmb {I}) \quad (2)\]  

其中 \(\alpha_{t\mid s} = \alpha_{t} / \alpha_{s}\) 和 \(\sigma_{t\mid s}^{2} = \sigma_{t}^{2} - \alpha_{t\mid s}^{2}\sigma_{s}^{2}\)。对于任何 \(0\leq s< t\leq 1\)，\(x_{s}\) 和 \(x_{t}\) 的边际分布为 \(p(x) = p(x)px)px)dx0\) 和 \(p(xuality) = p(x)px)dx0\)。如果只观察到 \(p(x_{s})\) 而不是 \(p(x_{0})\)，则 \(\mathbf{\mathcal{x}}_{t}\) 的边际分布可以替换表达为：\(p(\pmb{x}_{t}) = \int p(\pmb {x}_{t}|\pmb{x}_{s})p(\pmb{x}_{s})d\pmb{x}_{s}\)。因此，我们得到了以下等价关系：

\[p(\boldsymbol{x}_t) = \int p(\boldsymbol{x}_t|\boldsymbol{x}_0)p(\boldsymbol{x}_0)d\boldsymbol{x}_0 = \int p(\boldsymbol{x}_t|\boldsymbol{x}_s)p(\boldsymbol{x}_s)d\boldsymbol{x}_s \quad (3)\]  

在训练过程中，\(\alpha_{t}\) 和 \(\sigma_{t}\) 是预定义的关于 \(t\) 的函数，而 \(\mathbf{x}_0\) 是从数据集分布中抽样得到的，\(\boldsymbol{x}_0\sim p(\boldsymbol{x}_0)\)。时间步 \(t\) 从定义的分布中抽样，区间为 [0, 1]，例如均匀分布或对数正态分布（Esser 等，2024），即 \(t\sim T(t;0,1)\)。然后样本 \(\boldsymbol{x}_t\) 由下式给出：\(\mathbf{x}_t = \alpha_t\boldsymbol{x}_0 + \sigma_t\mathbf{\epsilon}\)，其中 \(\pmb{\epsilon}\sim \mathcal{N}(\pmb{\epsilon};0,\mathbf{I})\)。在后续段落中，除非另有说明，我们用 \(t\sim T\) 和 \(\epsilon \sim \mathcal{N}\) 表示以简化表述。Song 等（2020）在基于分数的生成模型的理论框架下统一了扩散模型，并证明连续扩散过程基本上是由随机微分方程（SDE）主导的。在这里，我们以流速预测为例，演示其与分数匹配的关系。令 \(\psi_{\pmb{\theta}}\) 表示由 \(\pmb \theta\) 参数化的扩散模型。流匹配与分数匹配之间的关系如下所示。

\[\begin{array}{rl} & {J_{flow}(\pmb {\theta}) = \mathbb{E}_{\pmb {x}_0\sim p(\pmb {x}_0),\pmb {\epsilon}\sim \mathcal{N},t\sim \mathcal{T},\pmb {x}_t = \alpha_t\pmb{x}_0 + \sigma_t\pmb {\epsilon}}||\pmb {\psi}_{\pmb \theta}(\pmb {x}_t) - (\pmb {\epsilon} - \pmb {x}_0)||^2]}\\ & {= \mathbb{E}_{\pmb {x}_0\sim p(\pmb {x}_0),t\sim \mathcal{T},\pmb {x}_t\sim p(\pmb {x}_t|\pmb {x}_0)}||\pmb {\psi}_{\pmb \theta}(\pmb {x}_t) + \pmb {x}_t / \alpha_t + (\sigma_t + \sigma_t^2 /\alpha_t)\nabla \pmb {x}_t\log (p(\pmb {x}_t|\pmb {x}_0))||^2]}\\ & {\qquad = \mathbb{E}_{t\sim \mathcal{T},\pmb {x}_t\sim p(\pmb {x}_t)}||\pmb {\psi}_{\pmb \theta}(\pmb {x}_t) + \pmb {x}_t / \alpha_t + (\sigma_t + \sigma_t^2 /\alpha_t)\nabla \pmb {x}_t\log (p(\pmb {x}_t))||^2] \end{array} \quad (4)\]  

公式 5 的推导基于去噪得分匹配 (DSM) 与显式得分匹配 (ESM) 之间的等价性，这一结论最早由 Vincent（2011）证明。在附录 A 中，我们提供了公式 5 的详细推导。此外，我们在附录 A 中展示了样本预测（即 x 预测）与得分匹配之间的联系。

#### 2.1.2 分布匹配蒸馏

设 \(\boldsymbol{G_{\phi}}\) 表示由 \(\phi\) 参数化的生成器。DMD 的目标是最小化真实数据分布 \(p_{real}(\pmb {x}_0)\) 与由 \(\boldsymbol{G_{\phi}}\) 生成的数据分布 \(p_{fake}(\pmb {x}_0)\) 之间的反向 Kullback-Leibler（KL）散度。

\[D_{KL}(p_{fake}||p_{real}) = \mathbb{E}_{\epsilon \sim \mathcal{N},\pmb{x}_0 = \pmb{G}\phi (\pmb {\epsilon})}[\log p_{fake}(\pmb {x}_0) - \log p_{real}(\pmb {x}_0)] \quad (6)\]  

我们用 \(D_{KL}\) 来缩写 \(D_{KL}(p_{fake}||p_{real})\) 在后续段落中。为了利用预训练的扩散模型作为得分估计器，生成的样本被扩散，目标变为：

\[D_{KL} = \mathbb{E}_{\epsilon \sim \mathcal{N},\pmb {x}_0 = \pmb{G}_\phi (\pmb {\epsilon}),t\sim \mathcal{T},\pmb {x}_t\sim p(\pmb {x}_t|\pmb {x}_0)}[\log p_{fake}(\pmb {x}_t) - \log p_{real}(\pmb {x}_t)] \quad (7)\]  

通过结合公式 5 和公式 7，我们可以将目标近似为：

\[D_{KL}\approx \mathbb{E}_{\epsilon \sim \mathcal{A},\pmb {x}_0 = \mathcal{G}_{\pmb {\phi}}}(\pmb {\epsilon}),t\sim \mathcal{T},\pmb {x}_t\sim p(\pmb {x}_t|x_{03})[\lambda_t(\pmb {T}_{\pmb {\theta}}(\pmb {x}_t) - \pmb {F}_{\pmb {\theta}}(\pmb {x}_t))] \quad (8)\]  

其中 \(\lambda_{t} = 1 / (\sigma_{t} + \sigma_{t}^{2} / \alpha_{t})\)，\(\textbf{F}_{\pmb{\theta}}\) 表示伪扩散模型，而 \(\pmb{T}_{\pmb{\theta}}\) 表示教师扩散模型。 \(\pmb{\theta}\) 初始化自 \(\hat{\pmb{\theta}}\)，并且 \(\boldsymbol {F}_{\theta}\) 根据公式 4 在 \(p_{fake}\) \((\boldsymbol{x}_{0})\) 上进行更新。从公式 7 到公式 8 的推导在模型为基于评分的生成模型的条件下是有效的。正式地，当 \(\pmb{F}_{\pmb{\theta}}(\pmb{x}_{t})\approx a_{t}\nabla \pmb{x}_{t}\log (p_{fake}(\pmb{x}_{t})) + b_{t}\pmb{x}_{t}\) 和 \(\mathbf{\zeta}_{\pmb {T}_{\pmb{\theta}}} (\pmb {x}_{t})\approx a_{t}\nabla \boldsymbol{x}_{t}\log (p_{real}(\pmb{x}_{t})) + b_{t}\xrightarrow[]{} = \mathbf{\zeta}_{\pmb {\theta}}\) 时，该近似成立，其中 \(a_{t}\) 是 \(t\) 的任意非零函数，而 \(b_{t}\) 是 \(t\) 的任意函数。对公式 8 关于生成器参数取梯度，我们得到：

\[\nabla \phi D_{KL}\approx \mathbb{E}_{\epsilon \sim \mathcal{N},\pmb{x}_0 = \pmb{G}^\epsilon (\epsilon),t\sim \mathcal{T},\pmb{x}_t\sim p(\pmb{x}_t|\pmb{x}_0)}[w_t(\pmb{T}_\theta (\pmb {x}_t) - \pmb {F}_\theta (\pmb {x}_t))]d = \pmb {G} = e \quad (9)\]  

其中 \(w_{t} = \lambda_{t}\alpha_{t}\)。与生成对抗网络（GANs）相似（Goodfellow et al., 2014），DMD 在每次迭代中采用由两个阶段组成的对抗训练过程。在假扩散优化阶段，\(F_{\pmb{\theta}}\) 根据公式 4 在生成分布上进行优化，使其能够作为 \(p_{fake}(\pmb {x}_{t})\) 的评分估计器。在生成器优化阶段，\(G_{\phi}\) 根据公式 9 进行更新，鼓励生成分布更接近真实分布。为了保证训练稳定性，\(F_{\pmb{\theta}}\) 接收更频繁的更新，使其能够准确估计不断变化的生成分布的评分（Yin et al., 2024a）。

### 2.2 从单步蒸馏到少步蒸馏

在 \(N\cdot\) 步蒸馏中，我们有一个调度器 \(\pmb{S}\)，它包含 \(N + 1\) 个时间步，\(\mathbf{t} = \{t_0, t_1, t_2, \ldots , t_N\}\)，其中 \(0 = t_{N}< t_{i}< t_{i - 1}< t_{0} = 1\) 对于任何 \(i\in \{2,\dots,N - 1\}\)。采样过程开始于 \(x_{t_0} = \epsilon \sim \mathcal{N}(\epsilon ;0,T)\)。然后样本 \(x_{0}\) 将通过迭代生成：对于 \(i = 0,1,\dots,N - 1\)，我们计算 \(\begin{array}{r}x_{t_{i + 1}} = S(G_{\phi}(\pmb{x}_{t_i}),\pmb{x}_{t_i},t_i,t_{i + 1}) \end{array}\)。令管道 \(\mathcal{(G}_{\phi},t,\epsilon ,S)\) 表示这一迭代采样过程。因此，方程 9 被如下调整：

\[\nabla \phi D_{K L}\approx \mathbb{E}_{\epsilon \sim \mathcal{N},\pmb{x}_{0} = \mathrm{pipeline}(G_{\phi},\pmb {t},\epsilon ,\mathcal{S}),t\sim \mathcal{T},\pmb {x}\sim p(\pmb{x}_{t}|\pmb{x}_{0})[w_{t}(\pmb{T}_{\theta}(\pmb{x}_{t}) - \pmb {F}_{\theta}(\pmb {x}_{t}))]d G / d\phi \quad (10)\]  

如图1a所示，生成器优化过程中计算图的深度随\(N\)线性增加，这降低了训练的稳定性并增加了内存开销。为了解决这个问题，Huang等人（2025）提出了一种随机梯度截断策略（SGTS），如图1b所示。在该策略中，从\(\{1,2,\ldots,N\}\)中随机选择一个索引\(j\)，并将对应的时间步\(t_{j}\)设为0。然后仅对步骤\(i = 0,1,\ldots,j - 1\)执行采样流程。关键的是，当\(j = 1\)时，训练迭代简化为一步蒸馏。因此，虽然SGTS提高了内存效率和训练稳定性，但它降低了少步模型的生成多样性，因为生成的分布偏向于一步生成器的分布。

### 2.3 分阶段动态模态分解

与可以在某些迭代中退化为一步蒸馏的 SGTS 的 DMD 相比，分阶段 DMD 通过将蒸馏过程划分为不同阶段并在中间时间步骤实施监督，从而避免了这个问题。在最后一个阶段之前的每个阶段，生成器被优化以最小化中间时间步骤的反向 KL 散度，而假扩散模型则通过在扩散过程的一个子区间内进行评分匹配进行更新。

### 2.3.1 中间时间步的分布匹配

Phased DMD的动机可以通过重新审视公式10来理解。为了采样 \(x_{t}\)，之前的方法（Yin et al., 2024a; Huang et al., 2025）首先生成 \(x_{0}\)，然后根据公式1将其扩散到 \(x_{t}\)。在Phased DMD中，流程被修改为生成中间样本 \(x_{t_k}\)，其中 \(0< k\leq N\)，而不是生成 \(x_{0}\)。样本 \(x_{t_k}\) 然后根据公式2进行扩散，取 \(s = t_k\) 并且 \(t\) 从子区间 \((t_{k},1)\) 中采样，即 \(t\sim \mathcal{T}(t;t_{k},1)\)。如图1c所示，Phased DMD逐步将生成器提炼到更高的信噪比（SNR）水平。在每个阶段 \(k\) 中，仅训练一个专家 \(G_{\phi_{k}}\)。该专家将分布 \(p(\boldsymbol{x}_{t_{k - 1}})\) 映射到 \(p(\boldsymbol{x}_{t_{k}})\)。第 \(k\) 阶段的生成器优化目标为：

\[\begin{array}{rl} & {\nabla \phi_{k}D_{K L}\approx \mathbb{E}_{\epsilon \sim \mathcal{N},\pmb{x}_{t_{k}} = \mathrm{pipeline}(G_{\phi_{1}}, G_{\phi_{2}},\dots G_{\phi_{k}},\{t_{1},t_{2},\dots, t_{k}\} ,\epsilon ,S),t\sim \mathcal{T}(t;t_{k},1),\pmb{x}_{t}\sim p(\pmb{x}_{t}|\pmb{x}_{t_{k}})}\\ & {\qquad [w_{t|s}(T_{\hat{\pmb{\theta}}}(\pmb{x}_{t}) - F_{\hat{\pmb{\theta}}_{k}}(\pmb{x}_{t}))] d G / d\phi_{k}} \end{array} \quad (11)\]  

其中 \(w_{t|s} = \lambda_{t}\alpha_{t|s}\)。经验上，我们发现采样 \(t \sim \mathcal{T}(t;t_{k}, 1)\) 而不是 \(t \sim \mathcal{T}(t;t_{k}, t_{k - 1})\) 更加符合阶段性 DMD 的渐进设计，并且表现更优（见附录 E.2）。在每个阶段开始时，假扩散模型 \(F_{\phi_{k}}\) 将从预训练的教师模型 \(T_{\theta}\) 重新初始化，并独立于前一阶段的模型进行训练。尽管最终得到的 MoE 生成器比单一网络生成器需要更多的 GPU 内存，但由于三个原因，这种开销是可以管理的。首先，仅需为第 \(k\) 个可训练专家提供优化器。其次，使用低秩适配（LoRA）（Hu 等，2021）可以显著减少这部分开销。具体而言，所有专家可以共享一个共同的主干网络，个别专家通过切换各自的 LoRA 权重来激活。最后，阶段性 DMD 可以与 SGTS 结合（如图 1d 所示），且蒸馏阶段的数量可以少于采样步骤的数量。

#### 2.3.2 子区间内的评分匹配

在阶段性动态模式分解（Phased DMD）中，一个关键挑战是干净的数据样本 \(x_{0}\) 仅在最后一个阶段可获得。因此，公式4中假设的伪扩散模型 \(F_{\theta_{k}}\) 的训练目标不再适用。为了解决这一问题，我们推导出基于子区间的评分匹配的训练目标。假设我们在中间时间步 \(s\) 处拥有观测值 \(x_{s}\sim p(x_{s})\)，其中 \(0< s< 1\)。

![fig 1](images/1.jpg)

Figure 2: Sampling trajectories for 200 samples in a 1D toy experiment. (a) Training with the full-interval objective (Eq. 4). (b) Training on \(0.5 < t < 1\) with the correct subinterval objective (Eq. 13). (c) Training on \(0.5 < t < 1\) with an incorrect target: \(\| (\psi_{\boldsymbol \theta}(\boldsymbol{x}_t) - (\pmb {\epsilon} - \pmb{x}_s)\|^2\)   

扩散模型 \(\psi_\theta\) 可以在子区间 \((s, 1)\) 内使用以下目标进行优化，该目标由公式 5 推导而来：

\[\begin{array}{rl} & {J_{flow}(\pmb {\theta}) = \mathbb{E}_{t\sim \mathcal{T}(t; s,1),x_t\sim p(x_t)}[\| \psi_\theta (\pmb {x}_t) + \pmb {x}_t / \alpha_t + (\sigma_t + \sigma_t^2 /\alpha_t)\nabla \pmb {x}_t\log (p(\pmb {x}_t))\| ^2 ]}\\ & {= \mathbb{E}_{x_s\sim p(\pmb {x}_s),t\sim \mathcal{T}(t; s,1),x_t\sim p(x_t|x_s)}[\| \psi_\theta (\pmb {x}_t) + \pmb {x}_t / \alpha_t + (\sigma_t + \sigma_t^2 /\alpha_t)\nabla \pmb {x}_t\log (p(\pmb {x}_t|x_s))\| ^2 ]}\\ & {= \mathbb{E}_{x_s\sim p(x_s),\epsilon \sim \mathcal{N},t\sim \mathcal{T}(t; s,1),x_t = \alpha_{t|s}x_s + \sigma_t|x_s}\epsilon [\| \psi_\theta (\pmb {x}_t) - ((\alpha_s^2\sigma_t + \alpha_t\sigma_s^2) / (\alpha_s^2\sigma_{t|s})\epsilon -(1 / \alpha_s)x_s)\| ^2 ]} \end{array} \quad (12)\]  

在相分解 DMD 的第 \(k\) 阶段，分布 \(p(\mathbf{x}_s)\) 使用 MoE 生成器流水线 \(\pmb {G}_{\phi_1},\pmb {G}_{\phi_2},\dots,\pmb {G}_{\phi_k}\) 的输出进行近似。由于当 \(t\rightarrow s\) 时 \(\sigma_{t|s}\to 0\)，公式中第 12 式会遇到奇异性和数值不稳定性。为了解决这个问题，我们应用了一个钳制函数，从而得到最终目标：

\[\begin{array}{rl} & {J_{flow}(\pmb {\theta}) = \mathbb{E}_{\pmb {x}_s\sim \beta (\pmb {x}_s),\pmb {\epsilon}\sim \mathcal{N},t\sim \mathcal{T}(t;s,1),\pmb {x}_t = \alpha_{t|s}\pmb {x}_s + \sigma_{t|s}\epsilon}}\\ & {\left[\operatorname {clamp}(1 / (\sigma_{t|s})^2)\| \sigma_{t|s}\psi_\pm (\pmb {x}_t) - (({\alpha}_s^2\sigma_t + \alpha_t\sigma_s^2) / {\alpha}_s^2)\pmb {\epsilon} - (\sigma_{t|s} / \alpha_s)x_s)\| ^2 \right]} \end{array} \quad (13)\]  

在这里，\(\operatorname {clamp}(1 / (\sigma_{t|s})^2)\) 将值限制在预定义的范围内，以防止溢出。我们设计了一个一维玩具实验来验证这个训练目标的效果，如图2所示。图2b中采样轨迹的紧密重叠表明，在定义的子区间内，使用方程13训练的流模型等同于使用方程4的标准目标训练的模型。相反，图2c说明了目标的不正确公式如何导致偏差估计。有关玩具实例的详细设置，请参见附录D。

## 3 实验与结果

我们将分阶段动态模式分解（Phased DMD）应用于最先进的图像和视频生成模型。所有实验均采用4步、2阶段的配置，如图1d所示。因此，每个基础模型被提炼为两个专家网络。为了证明性能的提升主要源于我们新颖的提炼范式，而不仅仅是可训练参数的增加，我们在实验中包含了Wan.2.2-T2V-A14B模型（Wan等，2025）。该模型已经具备混合专家（MoE）结构，标准DMD和我们的分阶段DMD都将其提炼为两个专家。这允许我们在等效参数预算下进行直接比较。由于计算需求，基础DMD（Yin等，2024a）方法仅应用于最小模型配置，即Wan.2.1-T2V-14B。实验配置的概述见表1，详细描述可在附录C中找到。

### 3.1 生成多样性的保护

为了评估生成多样性，我们构建了一个包含21个提示的文本到图像测试集。每个提示提供了图像内容的简短描述，而没有详细的规范。对于每个提示，我们使用种子0到7生成了8张图像。基础模型生成图像时采用40步采样，CFG比例为4。所有蒸馏模型采用4步和CFG比例为1进行采样。如图3b所示，4步DMD模型生成的图像细节丢失。尽管带有SGTS的4步DMD模型改善了图像质量，但这以减少多样性为代价。图3c显示，生成的图像通常采用类似的特写视图，并在不同随机种子之间表现出有限的构图变化。相比之下，Phased DMD更好地保留了多样性，生成了具有更广泛自然构图的图像，如图3d所示。生成多样性使用两个互补的指标进行评估：（1）DINOv3特征的平均成对余弦相似度（Simoni等，2023），值越低表示多样性越高；（2）平均成对LPIPS距离（Zhang等，2018），值越高表示多样性越大。这两个指标都是基于使用不同种子生成的相同提示的图像计算的。定量结果见表2。正如预期，基础模型实现了最高的多样性。值得注意的是，带SGTS的DMD的多样性略低于普通DMD。我们的Phased DMD在保留原始模型生成多样性方面优于两个蒸馏基线。Qwen-Image上的多样性改善幅度有限。我们认为这源于基础模型自身输出多样性有限。

Table 1: Overview of Experimental Setup.   

<table><tr><td>Base Model</td><td>Task</td><td>DMD</td><td>DMD with SGTS</td><td>Phased DMD (Ours)</td></tr><tr><td>Wan2.1-T2V-14B</td><td>T21</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Wan2.2-T2V-A14B</td><td>T21, T2V, I2V</td><td>×</td><td>√</td><td>√</td></tr><tr><td>Qwen-Image-20B</td><td>T21</td><td>×</td><td>√</td><td>√</td></tr></table>  

Table 2: Two metrics for quantitative diversity evaluation: average pairwise DINOv3 cosine similarity (lower is better) and LPIPS distance (higher is better). Phased DMD outperforms the vanilla DMD and DMD with SGTS in preserving generative diversity of the base models.

Table 2.2: Two metrics for quantitative diversity evaluation: average pairwise DINOv3 cosine sim- larity (lower is better) and LPIPS distance (higher is better). Phased DMD outperforms the vanilla DMD and DMD with SGTS in preserving generative diversity of the base models.   

<table><tr><td>Method</td><td>Wan2.1-T2V-14B DINOv3↓ LPIPS↑</td><td>Wan2.2-T2V-A14B DINOv3↓ LPIPS↑</td><td>Qwen-Image DINOv3↓ LPIPS↑</td></tr><tr><td>Base model</td><td>0.708</td><td>0.607</td><td>0.732</td></tr><tr><td>DMD</td><td>0.825</td><td>0.522</td><td>-</td></tr><tr><td>DMD with SGTS</td><td>0.826</td><td>0.521</td><td>0.427</td></tr><tr><td>Phased DMD (Ours)</td><td>0.782</td><td>0.544</td><td>0.681</td></tr></table>  

### 3.2 保留基础模型的关键能力

Wan2.2 视频生成模型在运动动态和相机控制方面展现了非凡的能力。然而，我们观察到带有 SGTS 的 DMD 会削弱这些属性，因为它们并没有专门针对低信噪比基专家。相位 DMD 从根本上解决了这个问题，通过将蒸馏过程划分为多个阶段，并在最后阶段明确消除了对 \(x_0\) 的依赖。

在第一阶段，仅低信噪比专家参与，并根据公式 11 和公式 13 进行蒸馏。由于预训练的低信噪比专家也是在低信噪比子区间上训练的，这种对齐更好地保留了其能力。如图 6 所示，使用 SGTS 的动态模式分解（DMD）生成的运动动态相比基线模型和分阶段 DMD 更为缓慢。类似地，图 7 显示使用 SGTS 的 DMD 倾向于产生特写视图，而分阶段 DMD 和基线模型更好地遵循提示的相机指令。我们使用一组 220 个文本提示（用于文本到视频，T2V）和 220 对图像提示（用于图像到视频，I2V）评估运动质量，每个提示生成一个视频，固定种子为 42。运动强度通过使用 Unimatch 计算的均值绝对光流（Xu 等，2023）和来自 VBench 的动态度量（Huang 等，2024b）进行量化。如表 3 所示，分阶段 DMD 产生的运动动态明显强于使用 SGTS 的 DMD，确认了其更优越的保留基线模型运动能力的能力。附加的比较视频在补充材料中提供。

![fig 1](images/1.jpg)

Figure 3: Samples (seeds 0-3) from the Wan2.1-T2V-14B base model (40 steps, CFG=4) and its distilled variants (4 steps, CFG=1): (a) Base, (b) DMD, (c) DMD with SGTS, (d) Phased DMD.   

![fig 2](images/2.jpg)

Figure 4: Examples generated by the Owen-Image distilled with Phased DMD.

Qwen-Image因其对提示的忠实遵循和高质量文本渲染而受到认可。为了评估经过蒸馏后这些能力的保留，我们对Qwen-Image应用了分阶段动态模态分解（Phased DMD），并使用其官方网站（Team, 2025a）上的提示生成图像。如图4所示，经过分阶段动态模态分解蒸馏的模型展现了良好的能力保留，生成了高质量的图像和准确的文本渲染。

### 3.3 MOE的优点

我们的实证研究发现，在蒸馏过程中，DMD最初捕捉结构信息，然后再学习更精细的纹理细节。在完全获取纹理细节之前，生成的图像和视频往往表现出过于平滑的特征，例如模糊的头发和塑料般的皮肤纹理。另一方面，逆KL散度的模式寻求特性导致随着训练迭代的增加，生成多样性下降。分阶段DMD通过将DMD划分为不同的训练阶段来解决质量与多样性之间的权衡。在低信噪比阶段，图像和视频的构成得到有效建立。在随后的高信噪比阶段，低信噪比专家被冻结，从而允许进行延长训练以提高生成质量，而不降低输出的结构构成。如图5所示，延长高信噪比专家的训练主要影响光照和纹理细节，同时保持图像的整体结构构成不变。

![fig 1](images/1.jpg)

Figure 5: Samples generated with high-SNR experts from different training stages (top: 100 iterations; bottom: 400 iterations) and a shared low-SNR expert. Each column uses identical prompts and seeds.   

## 4 相关工作

我们的工作建立在变分评分蒸馏（VSD）基础之上，包含一个可训练的生成器、一个伪评分估计器和一个预训练的教师评分估计器。与之最相关的工作是TDM（Luo等，2025），该工作同样将DMD扩展到少步蒸馏。然而，分层DMD在三个关键方面有所不同：（a）TDM缺乏理论基础，导致伪流训练不正确；（b）我们的框架固有地产生混合专家（MoE）模型；（c）我们使用反向嵌套SNR区间，而不是TDM的离散区间。关于相关工作的完整讨论见附录B。

## 5 结论与讨论

分阶段动态模式分解主要增强了生成的结构方面，如图像构图多样性、运动动态和相机控制。然而，对于像Qwen-Image这样的基础模型，其输出本质上缺乏多样性，因此改进效果不明显。尽管这项工作在动态模式分解框架内展示了分阶段蒸馏，但该方法可以推广到其他目标，如SiD中的Fisher散度（Zhou et al., 2024），我们将其留待未来探索。有可能将其他增强多样性和动态的方法整合进来，例如利用基础模型预生成的轨迹数据。然而，这将妨碍动态模式分解的无数据优势。虽然我们未来可能会探索这样的方向，但这项工作优先考虑无数据范式。

## 6 伦理声明

6 伦理声明 本文遵循ICLR伦理规范。所提方法基于动态模式分解（DMD），为无数据蒸馏框架。然而，用于蒸馏的基础模型可能由于训练集中的人类数据而生成人体图像，可能引发隐私和同意方面的担忧。为了解决这一问题，我们仅专注于人类运动动态，完全不使用个人可识别信息。关于视频生成模型，尽管它在内容创作中具有积极的应用，但也存在被滥用用于欺骗内容或监控的风险。我们承认这些风险，并强调我们的模型仅用于科学研究和积极的使用案例。

## 7 重现性声明

7 reproducibility 声明 我们采取了广泛的措施以确保可重现性。要重现相位动态模式分解（Phased DMD），核心方程在正文第2节中提供，详细推导见附录A。有关通过子区间验证评分匹配有效性的示例的相关细节，可以在附录D中找到。要复制我们的实验，实验设置、超参数、评估指标和实现选择的详细信息可在正文第3节和附录3中获取。代码和模型也将发布。

## REFERENCES  

REFERENCESYogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittila, Timo Aila, Samuli Laine, et al. edifici tax- to- image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022. Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kip- per Gong, Tianpeng Gu, Xiusen Gu, et al. Hunyuaimage 3.0 technical report. arXiv preprint arXiv:2509.23951, 2025. Patrick Esser, Sumith Sulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high- resolution image synthesis, 2024. URL https://arxiv.org/abs/ 2403.03206. Jiarui Fang, Jinzhe Pan, Xibo Sun, Aoyu Li, and Jiannan Wang. xdit: an inference engine for diffusion transformers (dits) with massive parallelism. arXiv preprint arXiv:2411.01738, 2024. Zhida Feng, Zhenyu Zhang, Xintong Yu, Yewei Fang, Lanxin Li, Xuyi Chen, Yuxiang Lu, Jiaxiang Liu, Weichong Yin, Shikun Feng, et al. Ernie- vilt Gray 2.0: Improving text- to- image diffusion model with knowledge- enhanced mixture- of- denoting- experts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10135- 10145, 2023. Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step diffusion via shortcut models. arXiv preprint arXiv:2410.12557, 2024. Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one- step generative modeling. arXiv preprint arXiv:2505.13447, 2025. Ian J. Goodfellow, Jean Pouget- Abadie, Mehdi Mirza, Bing Xu, David Warde- Farley, Sherjill Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks, 2014. URL https://arxiv.org/abs/1406.2661. GoogleAI. Image generation with gemini (aka nano banana), 2025a. URL https://ai.google.dev/gemini- api/docs/image- generation.GoogleAI. Generate videos with veo 3 in gemini api, 2025b. URL https://ai.google.dev/gemini- api/docs/video?example=dialogue.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

OpenAI. Video generation models as world simulators, 2024. URL https://openai.com/index/video-generation-models-as-world-simulators/.  

OpenAI. Introducing 4o image generation, 2025. URL https://openai.com/index/introducing- 4o-image-generation/.  

Wenqi Ouyang, Yi Dong, Lei Yang, Jianlou Si, and Xingang Pan. I2vedit: First- frame- guided video editing via image- to- video diffusion models. In SIGGRAPH Asia 2024 Conference Papers, pp. 1- 11, 2024.  

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxt: Improving latent diffusion models for high- resolution image synthesis, 2023. URL https://arxiv.org/abs/2307.01952.  

Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.  

Team Seedream, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, et al. Seedream 4.0: Toward next- generation multimodal image generation. arXiv preprint arXiv:2509.20427, 2025.  

Oriane Simeoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonijosea, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Theo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jegou, Patrick Labatut, and Piotr Bojanowski. DINOv3, 2025. URL https://arxiv.org/abs/2508.10104.  

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models, 2022. URL https://arxiv.org/abs/2010.02502.  

Yang Song, Jascha Sohl- Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score- based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.  

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models, 2023. URL https://arxiv.org/abs/2303.01469.  

Qwen Team. Qwen- image: Crafting with native text rendering, 2025a. URL https://qwenlm.github.io/blog/qwen- image/.  

Tencenet Hongyuan Team. Hyunyuanimage 2.1: An efficient diffusion model for high- resolution (2k) text- to- image generation. https://github.com/Tencenet- Hunyuan/HunyuanImage- 2.1, 2025b.  

Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation, 23(7):1661- 1674, 2011.  

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen- Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi- Fan Wu, and Ziyu Liu. Wan: Open and advanced large- scale video generative models. arXiv preprint arXiv:2503.20314, 2025.  

Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xinzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi- Fan Wu, and ZijIa Liu. Wan: Open and advanced large- scale video generative models. arXiv preprint arXiv:2503.20314, 2025.  

Fu- Yun Wang, Zhaoyang Huang, Alexander Bergman, Dazhong Shen, Peng Gao, Michael Lingle- bach, Keqiang Sun, Weikang Bian, Guanglu Song, Yu Liu, et al. Phased consistency models. Advances in neural information processing systems, 37:83951- 84009, 2024.

Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High- fidelity and diverse text- to- 3d generation with variational score distillation. Advances in neural information processing systems, 36:8406- 8441, 2023.  

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen- image technical report, 2025. URL https://arxiv.org/abs/2508.02324.  

Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, Fisher Yu, Dacheng Tao, and Andreas Geiger. Unifying flow, stereo and depth estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.  

Tianwei Yin, Michael Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and Bill Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455- 47487, 2024a.  

Tianwei Yin, Michael Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T. Freeman, and Taesung Park. One- step diffusion with distribution matching distillation, 2024b. URL https://arxiv.org/abs/2311.18828.  

Pengze Zhang, Hubery Yin, Chen Li, and Xiaohua Xie. Tackling the singularities at the endpoints of time intervals in diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6945- 6954, 2024.  

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.  

Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one- step generation. In Forty- first International Conference on Machine Learning, 2024.

## A DETAILED DERIVATION OF METHOD  

We show the detailed derivation of Eq. 5 as follows:  

\[\begin{array}{r l} & {J_{f l o w}(\pmb {\rho})\coloneqq \mathbb{E}_{x_{0}\sim p(x_{0}),\epsilon \sim \mathcal{N},t\sim \mathcal{T},x_{t} = \alpha_{t}x_{0} + \sigma_{t}\epsilon}\big\{\| \psi_{\pmb \theta}(x_{t}) - (\epsilon -x_{0})\|^{2}\big\}}\\ & {\qquad = \mathbb{E}_{x_{0}\sim p(x_{0}),\epsilon \sim \mathcal{N},t\sim \mathcal{T},x_{t} = \alpha_{t}x_{0} + \sigma_{t}\epsilon}\big\{\| \psi_{\pmb \theta}(x_{t}) - (\epsilon -(x_{t} - \sigma_{t}\epsilon) / \alpha_{t})\|^{2}\big\}}\\ & {\qquad = \mathbb{E}_{x_{0}\sim p(x_{0}),\epsilon \sim \mathcal{N},t\sim \mathcal{T},x_{t} = \alpha_{t}x_{0} + \sigma_{t}\epsilon}\big\{\| \psi_{\pmb \theta}(x_{t}) + x_{t} / \alpha_{t} - (1 + \sigma_{t} / \alpha_{t})\epsilon \|^{2}\big\}}\\ & {\qquad = \mathbb{E}_{x_{0}\sim p(x_{0}),t\sim \mathcal{T},\pmb {x}_{t}\sim p(\pmb{x}_{t}\| x_{0})\big\{\| \psi_{\pmb \theta}(x_{t}) + x_{t} / \alpha_{t} + (\sigma_{t} + \sigma_{t}^{2} / \alpha_{t})\nabla x_{t}\log (p(x_{t}\| x_{0}))\| ^{2}\big\}}\\ & {\qquad = \mathbb{E}_{t\sim \mathcal{T},\pmb {x}_{t}\sim p(\pmb{x}_{t})\big\{\| \psi_{\pmb \theta}\big(\pmb{x}_{t}\big) + x_{t} / \alpha_{t} + (\sigma_{t} + \sigma_{t}^{2} / \alpha_{t})\nabla x_{t}\log (p(\pmb {x}_{t}))\|^{2}\big\}} \end{array} \quad (13)\]  

In the derivation, we use the the score of \(p(x_{t}|x_{0})\) i.e., \(\nabla x_{t}\log (p(x_{t}|x_{0})) = -(1 / \sigma_{t})\epsilon\) , and the equivalence between DSM and ESM (Vincent, 2011).  

We show the detailed derivation of Eq. 12 as follows:  

\[\begin{array}{r l} & {J_{f l o w}(\pmb \theta) = \mathbb{E}_{t\sim \mathcal{T}(t;s,1),\pmb{x}_{t}\sim p(\pmb{x}_{t})}\big[\| \psi_{\pmb \theta}(x_{t}) + \pmb{x}_{t} / \alpha_{t} + (\sigma_{t} + \sigma_{t}^{2} / \alpha_{t})\nabla \pmb{x}_{t}\log (p(x_{t}))\|^{2}\big]}\\ & {= \mathbb{E}_{\pmb{x}_{s}\sim p(\pmb{x}_{s}),t\sim \mathcal{T}(t;s,1),\pmb{x}_{t}\sim p(\pmb{x}_{t}|x_{s})}\left[\| \psi_{\pmb \theta}(\pmb{x}_{t}) + \pmb{x}_{t} / \alpha_{t} + (\sigma_{t} + \sigma_{t}^{2} / \alpha_{t})\nabla \pmb{x}_{t}\log (p(x_{t}|x_{s}))\|^{2}\right]}\\ & {= \mathbb{E}_{\pmb{x}_{s}\sim p(\pmb x_{s}),\pmb{\epsilon}\sim \mathcal{N},t\sim \mathcal{T}(t;s,1),\pmb{x}_{t} = \alpha_{t}|_{s}\pmb{x}_{s} + \sigma_{t}|_{s}\epsilon \big[\| \psi_{\pmb \theta}(\pmb {x}_{t}) + \pmb{x}_{t} / \alpha_{t} - ((\sigma_{t} + \sigma_{t}^{2} / \alpha_{t}) / \sigma_{t}|_{s})\pmb {\epsilon}\|^{2}\big]}\\ & {= \mathbb{E}_{\pmb{x}_{s},\pmb {\epsilon}\sim p(\pmb{x}_{s}),\pmb{\epsilon}\sim N,t\sim \mathcal{T}(t;s,1),\pmb{x}_{t} = \alpha_{t}|_{s}\pmb{x}_{s} + \sigma_{-t}|_{s}\epsilon \big[\| \psi_{\pmb \theta}(\pmb {x}_{t}) + (\alpha_{t}|_{s}\pmb{x}_{s} + \sigma_{t}|_{s}\pmb {\epsilon}) / \alpha_{t} - (((\sigma_{t} + \sigma_{t}^{2} / \alpha_{t}) / \sigma_{t}|_{s})\epsilon \|^{2}\bigg]}\\ & {= \mathbb{E}_{\pmb{x}_{s}\sim p(\pmb {x}_{s}),\pmb{\epsilon}\sim N,t\sim \mathcal{T}(t;s,1),\pmb{x}_{t} = \alpha_{t}|_{s}\pmb {x}_{s} + \sigma_{t}|_{s}\pmb {\epsilon}[\| \psi_{\pmb \theta}(\pmb{x}_{t})-((\alpha_{s}^{2}\sigma_{t}+\alpha_{t}\sigma_{s}^{2})/(\alpha_{s}^{2}\sigma_{t}|_{s})\epsilon-(1/\sigma_{s})\pmb{x}_{s}))\big]\widetilde{\pmb{x}_{s}}]} \end{array} \quad (14)\]  

The relationship between sample prediction (x- prediction) and score matching is derived as follows:  

\[\begin{array}{r l} & {J_{s a m p l e}(\pmb {\theta}) = \mathbb{E}_{x_{0}\sim p(x_{0}),\epsilon \sim \mathcal{N},t\sim \mathcal{T},x_{t} = \alpha_{t}x_{0} + \sigma_{t}\epsilon}\big[\| \pmb {\mu}_{\theta}(x_{t}) - x_{0}[2\big]\bigg]}\\ & {\qquad = \mathbb{E}_{x_{0}\sim p(x_{0}),\epsilon \sim \mathcal{N},t\sim \mathcal{T},x_{t} = \alpha_{t}x_{0} + \sigma_{t}\epsilon}\big[\| \pmb{\mu}_{\theta}(x_{t}) - (x_{t} - \sigma_{t}\epsilon) / \alpha_{t}\big]^{2}\bigg]}\\ & {\qquad = \mathbb{E}_{x_{0}\sim p(x_{0}),\epsilon \sim \mathcal{N},t\sim \mathcal{T},x_{t} = \alpha_{t}\pmb{x}_{0} + \sigma_{t}\epsilon}\big[\| \pmb{\mu}_{\theta}(x_{t}) - x_{t} / \alpha_{t} + (\sigma_{t} / \alpha_{t})\pmb{\epsilon}\}^{2}\bigg]} \end{array} \quad (14)\]  

The training objective for x- prediction diffusion models within a subinterval is as follows:  

\[\begin{array}{r l} & {J_{\ast m p l e}(\pmb \theta) = \mathbb{E}_{t\sim \mathcal T,x_{t}\sim p(\pmb x_{t})}\big[\| \pmb{\mu}_{\pmb \theta}(\pmb x_{t}) - \pmb x_{t} / \alpha_{t} - (\sigma_{t}^{2} / \alpha_{t})\nabla \pmb{x}_{t}\log (p(\pmb {x}_{t}))\|^{2}\big]}\\ & {\quad = \mathbb{E}_{\pmb{x}_{s}\sim p(\pmb {x}_{s}),t\sim \mathcal T(t;s,1),\pmb x_{t}\sim p(\pmb{x}_{t}|x_{s})}\big[\| \pmb {\mu}_{\pmb \theta}(\pmb x_{t}) - \pmb{x}_{t} / \alpha_{t} - (\sigma_{t}^{2} / \alpha_{t})\nabla \pmb{x}_{t}\log (p(\pmb{x}_{s}|x_{s}))\|^{2}\big]}\\ & {\quad = \mathbb{E}_{\pmb{x}_{s}\sim p(\pmb {x}_{s}),\epsilon \sim \mathcal N,t\sim \mathcal T(t;s,1),\pmb{x}_{t} = \alpha_{t}|_{s}\pmb{x}_{s} + \sigma_{t}|_{s}\epsilon}\big[\| \pmb {\mu}_{\pmb \theta}(\pmb x_{t}) - \pmb {x}_{t} / \alpha_{t} + ((\sigma_{t}^{2} / \alpha_{t}) / \sigma_{t}|_{s})\pmb {\epsilon}\|^{2}\big]}\\ & {\quad = \mathbb{E}_{\pmb{x}_{s}\sim p(\pmb{x}_{s}),\epsilon \sim \mathcal N,t\sim \mathcal T(t;s,1),\pmb{x}_{t} = \alpha_{t}|_{s}\pmb{x}_{s} + \sigma_{t}|_{s}\epsilon}\big[\| \pmb {\mu}_{\pmb \theta}(\pmb{x}_{t}) - (\alpha_{t}|_{s}\pmb{x}_{s} + \sigma_{t}|_{s}\epsilon) / \alpha_{t} + ((\sigma_{t}^{2} / (\alpha_{t}\sigma_{t}|_{s})\epsilon)\|^{2}\big]}\\ & {\quad = \mathbb{E}_{\pmb{x}_{s}\sim p(\pmb {x}_{s}),\epsilon \sim \mathcal N,t\sim \mathcal T(t;s,1),\pmb{x}_{t} = \alpha_{t}|_{s}\pmb{x}_{s} + \sigma_{t}|_{s}\epsilon}\big[\| \pmb {\mu}_{\pmb \theta}(x_{t}) - (1/_{s})_{s}\pmb {x}_{s} - (\alpha_{t}\sigma_{s}^{2} / \alpha_{s}\sigma_{s}|_{s})\epsilon \|^{2}\big]} \end{array} \quad (15)\]  

Optimizing within the subinterval according to Eq. 15 gives an unbiased estimation of x- prediction. In contrast, the objective \(\big[\| \mu_{\theta}(x_t) - x_s\| ^2 \big]\) yields a biased estimation.  

## B RELATED WORKS  

Our work is situated within the framework of Variational Score Distillation (VSD) (Wang et al., 2023). VSD involves three components: a trainable generator, a fake score estimator, and a pretrained teacher score estimator. The generator is optimized to produce a distribution that approximates the real data distribution. Concurrently, the fake score estimator learns to estimate the score of the generator's output distribution. The update direction for the generator is then determined by the discrepancy between the teacher's score (for the real distribution) and the fake score estimator's score.  

Similar to GANs, the VSD framework is adversarial. The fake score estimator must be precisely optimized to learn the score of the current generated distribution. This accurate estimation is crucial,

as it combines with the fixed teacher model (which provides the score for the real data) to produce a correct guidance signal for the generator. This principle explains why DMD2 (Yin et al., 2024a) operates successfully without external real data, in contrast to its predecessor DMD (Yin et al., 2024b).  

A key advantage of VSD over GANs for distilling pre- trained diffusion models is initialization. The pre- trained model serves a dual role: it is a powerful multi- step generator and an accurate estimator of the real data distribution's score. This allows it to effectively initialize all three components in the VSD framework, leading to significantly enhanced training stability.  

Several methods are built upon the VSD framework, including Diff- Instruct (Luo et al., 2023), DMD (Yin et al., 2024a), SID (Zhou et al., 2024), and FGM (Huang et al., 2024a). The fundamental distinction between these approaches lies in the specific divergence they minimize. DMD, for instance, optimizes the reverse KL divergence between the real and generated distributions. A key advantage of this choice is its computational efficiency compared to alternatives like the Fisher divergence used in SID (Zhou et al., 2024). Specifically, during generator optimization, DMD does not require gradients to be backpropagated through the fake and teacher score estimators, whereas SID does. This does not imply the two estimators are trainable in this stage for SID, but rather reflects a difference in the computational graph. This property makes DMD more amenable to engineering implementation and scalable to large base models.  

Similar to our work, TDM (Luo et al., 2025) also aimed to extend DMD to few- step distillation. However, our approach differs from TDM in three key aspects: (a) The lack of proper theoretical grounding in TDM renders its fake flow training formulation incorrect, undermining the foundations of DMD. (b) Our framework inherently produces MoE models for few- step generation. (c) While TDM uses disjoint SNR intervals, our method employs reverse nested intervals, where each interval is a subset of the subsequent one.  

## C EXPERIMENTAL DETAILS  

We conduct experiments on three tasks: text- to- image, text- to- video and image- to- video generation. The following global settings are applied across all experiments: a batch size of 64; a fake diffusion model learning rate of 4e- 7 with full- parameter training; a generator learning rate of 5e- 5 using LoRA with a rank of 64 and an alpha value of 8. The AdamW optimizer is used for both the fake diffusion model and the generator, with hyperparameter \(\beta_{1} = 0\) , \(\beta_{2} = 0.999\) . The fake diffusion model is updated five times for every generator update.  

For the Wan2. x base models, distillation for the text- to- image task is performed at a data resolution of frame \(= 1\) , width \(= 1280\) , height \(= 720\) .  

For the Wan2.2- T2V- A14B model, distillation for the text- to- video and image- to- video task uses a mixture of data resolutions: (81, 720, 1280), (81, 1280, 720), (81, 480, 832), (81, 832, 480).  

For the Qwen- Image model, distillation for the text- to- image task uses a mixture of data resolutions: (1, 1382, 1382), (1, 1664, 928), (1, 928, 1664), (1, 1472, 1104), (1, 1104, 1472), (1, 1584, 1056), (1, 1056, 1584).  

## D TOY EXAMPLE DETAILS  

We construct a toy example where \(x_{0}\) takes only four values: {- 1, 0, 1, 2}. A minimal model is designed, consisting of four MLPs with \(\text{dim} = 512\) , conditioned solely on \(t\) . Fig. 1a shows training on the full interval using Eq. 4, Fig. 1b shows training on subintervals using Eq. 13, and Fig. 1c shows training on subintervals using Eq. 4, simply replacing \(x_{0}\) with \(x_{s}\) . As shown in Fig. 4b, when the correct objective is used, the trajectories on subintervals perfectly align with those on the full interval. In contrast, using an incorrect objective introduces trajectory deviations, as illustrated in Fig. 4c. Such a trajectory deviation signifies that the trained model no longer satisfies the score- matching objective (i.e., Eq. 5 is violated), thus contravening a core principle of DMD.

## E MORE RESULTS  

## E.1 MOTION DYNAMICS AND CAMERA CONTROL  

As shown in Fig. 6, DMD with SGTS generates slower motion dynamics compared to the base model and Phased DMD. Similarly, Fig. 7 show that DMD with SGTS tends to produce close- up views, while Phased DMD and the base model better adhere to the prompt's camera instructions.  

## E.2 ABLATION ON DIFFUSION TIMESTEP SUBINTERVALS  

Empiritically, we observe that sampling \(t\sim \mathcal{T}\big(t;t_{k},1\big)\) outperforms sampling \(t\sim \mathcal{T}\big(t;t_{k},t_{k - 1}\big)\) in terms of generation quality. Fig. 8 illustrates the results of these two methods in the Wan2.2 T2V distillation task. Specifically, sampling \(t\sim \mathcal{T}\big(t;t_{k},1\big)\) yields normal color tones and accurate structures, whereas sampling \(t\sim \mathcal{T}\big(t;t_{k},t_{k - 1}\big)\) results in low- contrast tones and degraded facial structures.  

At the beginning of each phase in Phased DMD, there is a substantial gap between the distribution of samples generated by the few- step generator and the distribution of real samples. The generated samples fall outside the domain of the teacher model, leading to inaccurate score estimations. This discrepancy is particularly pronounced in the high- SNR (low noise level) range, where samples are less corrupted by noise. In contrast, in the low- SNR (high noise level) range, the diffused generated distribution overlaps more significantly with the diffused real distribution, enabling the teacher model to provide more accurate score estimations. Consequently, noise injection at high noise levels plays a crucial role in DMD training.  

To validate this analysis, we perform ablation studies on vanilla DMD for the Wan2.1 T21 task. Specifically, the diffusion timestep \(t\) is fixed at 0.357 for one experiment and at 0.882 for another. Wang et al. (2023) has proven that \(D_{KL}(p_{fake}(x_t)\| p_{real}(x_t) = 0\Leftrightarrow D_{KL}(p_{fake}(x_0)\| p_{real}(x_0) = 0\) for any \(0< t< 1\) . Thus, both experiments are theoretically valid. However, the experiment with a diffusion timestep \(t = 0.357\) fails to converge, as illustrated in Fig. 9, while the experiment with \(t = 0.882\) demonstrates correct results. This controlled experiment highlights that incorporating high noise levels is essential for effective DMD training.

![fig 5](images/5.jpg)

Figure 6: Comparison of video frames generated by the Wan2.2-T2V-A14B base model and its distilled versions using DMD with SGTS and Phased DMD. Each video consists of 81 frames and frames with indices \(\{0,10,\dots,80\}\) are combined as a preview. The base model was sampled with 40 steps and CFG of 4, while the distilled models used 4 steps and CFG of 1 (see fixed at 42). The prompt is “A parkour athlete swiftly runs horizontally along a brick wall in an urban setting. Pushing off powerfully with one foot, they launch themselves explosively into a twisting front flip. The camera tenaciously stays with them in mid-air as they tuck their legs tightly to their chest to rapidly accelerate the rotation, then extend them forcefully outwards again, precisely spotting their landing on the concrete below. The dynamic movement is vividly captured against a backdrop of city lights and shadows.”

![fig 5](images/5.jpg)

Figure 7: Comparison of video frames generated by the WAN2.2-T2V-A14B base model and its distilled versions using DMD with SGTS and Phased DMD. Each video consists of 81 frames and frames with indices \(\{0,10,\dots,80\}\) are combined as a preview. The base model was sampled with 40 steps and CFG of 4, while the distilled models used 4 steps and CFG of 1 (seed fixed at 42). The prompt is "Day time, sunny lighting, low angle shot, warm colors. A dynamic individual in a vibrant, multi-colored outfit and a red helmet executes a fast-paced slalom on roller skates through a bustling urban park. The camera starts focused on the skates carving sharp turns on the pavement and tilts up to reveal their entire body leaning into the motion. Their face shows a mix of joy and deep concentration. The warm afternoon sun filters through the lush greenery, with the azure sky visible above, creating a scene bursting with energy."

![fig 4](images/4.jpg)

Figure 8: The effect of noise injection intervals. Luo et al. (2025) employs disjoint noise injection timestep intervals for different generation steps, where the intervals do not overlap. In contrast, we adopt reverse nested intervals, where the diffusion timestep interval in each phase terminates at 1.0. Integrating disjoint intervals into Phased DMD leads to unnatural colors and deteriorated facial structures, as illustrated on the left. Conversely, adopting reverse nested intervals yields correct results.

![fig 3](images/3.jpg)

Figure 9: The effect of noise injection timestep in DMD training. In DMD training, noise is injected into the generated samples at a low noise level (left) and a high noise level (right). The training fails to converge correctly when noise is injected exclusively at a low noise level.