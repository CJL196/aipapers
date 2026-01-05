# 改进的分布匹配蒸馏用于快速图像合成

陈伟尹1 米哈伊尔·加比1 泰松·朴2 理查德·张2 埃利·谢赫特曼2 弗雷多·杜兰1 威廉·T·弗里曼1 1麻省理工学院 2Adobe研究 https://tianweiy.github.io/dmd2/

# 摘要

最近的方法展示了将昂贵的扩散模型提炼为高效的一步生成器的潜力。其中，分布匹配蒸馏（DMD）生成的“一步生成器”在分布上与其教师模型匹配，即蒸馏过程并不强制与教师模型的采样轨迹一一对应。然而，为了确保训练的稳定性，DMD需要额外的回归损失，这是通过教师模型使用确定性采样器生成的大量噪声图像对计算得出的。这不仅对大规模文本到图像合成计算开销大，而且还限制了学生模型的质量，使其过于依赖于教师模型的原始采样路径。我们引入了DMD2，一组技术来消除这一限制并改善DMD训练。首先，我们消除了回归损失和昂贵数据集构建的需求。我们证明，导致不稳定性的原因在于“假”评估者未能充分准确地估计生成样本的分布，并提出了一种双时间尺度更新规则作为补救措施。其次，我们将GAN损失整合到蒸馏过程中，以区分生成样本和真实图像。这使我们能够在真实数据上训练学生模型，从而减轻教师模型对“真实”分数估计的不准确性，从而提升质量。第三，我们引入了一种新的训练程序，使学生模型能够进行多步采样，并通过在训练期间模拟推理时生成器的样本来解决先前工作的训练-推理输入不匹配问题。综合来看，我们的改进在一步图像生成上设立了新的基准，在ImageNet-$64 \times 64$上FID分数达到了1.28，在零-shot COCO 2014上FID分数为8.35，超过了原始教师模型，尽管推理成本减少了$500\times$。此外，我们展示了我们的方法能够通过提炼SDXL生成百万像素图像，在少步方法中展现出卓越的视觉质量，并超过了教师模型。我们发布了我们的代码和预训练模型。

# 1 引言

扩散模型在视觉生成任务中达到了前所未有的质量[18]。但它们的采样过程通常需要几十个迭代去噪步骤，每一步都是神经网络的前向传播。这使得高分辨率的文本到图像合成速度缓慢且成本高昂。为了解决这个问题，已开发出许多蒸馏方法，将教师扩散模型转换为高效的少步学生生成器[920]。然而，这些方法常常导致质量下降，因为学生模型通常是通过损失函数来学习教师的成对噪声到图像的映射，但难以完美地模仿其行为。

![](images/1.jpg)  

Figure 1: $1 0 2 4 \times 1 0 2 4$ samples produced by our 4-step generator distilled from SDXL. Please zoom in for details.

然而，需要注意的是，旨在匹配分布的损失函数，如GAN或DMD损失，并不需要承担精确学习从噪声到图像特定路径的复杂性，因为它们的目标是通过最小化学生和教师输出分布之间的Jensen-Shannon（JS）或近似Kullback-Leibler（KL）散度来与教师模型在分布上对齐。特别是，DMD在蒸馏Stable Diffusion 1.5方面展示了最先进的结果，但其研究程度仍低于基于GAN的方法。一个可能的原因是DMD仍然需要额外的回归损失以确保训练稳定。这反过来又需要通过运行教师模型的完整采样步骤来创建数百万对噪声-图像对，这对文本到图像合成来说特别昂贵。回归损失还抵消了DMD无配对分布匹配目标的关键优势，因为它导致学生的质量受到教师的上限限制。在本文中，我们展示了如何消除DMD的回归损失，而不影响训练稳定性。然后，我们通过将GAN框架整合到DMD中，推动分布匹配的极限，并通过一种我们称之为“反向模拟”的新训练程序实现少步骤采样。总体而言，我们的贡献带来了先进的快速生成模型，能够在仅使用4个采样步骤的情况下超越其教师。我们称之为DMD2的方法在一步图像生成中实现了最先进的结果，在ImageNet $6 4 { \times } 6 4$上设定了1.28的FID分数，在零-shot COCO 2014上为8.35。我们通过蒸馏SDXL以生成高质量的百万像素图像，展示了我们方法的可扩展性，确立了少步骤方法中的新标准。简而言之，我们的贡献如下： • 我们提出了一种新的分布匹配蒸馏技术，不需要回归损失以实现稳定训练，从而消除了成本高昂的数据收集需求，并允许更灵活和可扩展的训练。 • 我们展示了DMD在没有回归损失的情况下的训练不稳定源于假扩散评论器训练不足，并实施了一种双时间尺度更新规则来解决该问题。 我们将GAN目标整合到DMD框架中，其中判别器被训练区分学生生成器生成的样本与真实图像。这种额外的监督在分布级别上操作，比原始回归损失更好地与DMD的分布匹配理念对齐。它减轻了教师扩散模型中的近似误差并提升了图像质量。 • 虽然原始DMD仅支持一步学生，我们引入了一种技术以支持多步生成器。与之前的多步蒸馏方法不同，我们通过在训练期间模拟推理时生成器输入，避免了训练与推理之间的领域不匹配，从而提高了整体性能。

# 2 相关工作

扩散蒸馏。近期的扩散加速技术集中于通过蒸馏加速生成过程。它们通常训练一个生成器，在更少的采样步骤中逼近教师模型的常微分方程（ODE）采样轨迹。值得注意的是，Luhman等人预计算了一组噪声与图像对的数据集，该数据集由教师使用ODE采样器生成，并利用该数据集训练学生模型在单次网络评估中回归映射。后续工作如渐进蒸馏消除了离线预计算这一配对数据集的需求。它们迭代地训练一系列学生模型，每个模型的采样步骤数量减少为其前身的一半。一种互补技术，Instaflow，使ODE轨迹变得平直，从而更易于通过一步学生进行逼近。一致性蒸馏和TRACT训练学生模型，使其在ODE轨迹的任何时间步输出自一致，从而与教师模型保持一致。

GANs。另一条研究方向采用对抗训练，以在更广泛的分布层面上将学生与教师对齐。在ADD [23]中，生成器使用来自扩散模型的权重进行初始化，并使用基于图像空间分类器的投影GAN目标进行训练[34]。在此基础上，LADD [24]利用预训练的扩散模型作为判别器，并在潜在空间中操作，从而提高可扩展性并实现更高分辨率的合成。受到DiffusionGAN [28, 29]的启发，UFOGen [25]在判别器中的真伪分类之前引入噪声注入，以平滑分布，进而稳定训练动态。一些最近的方法将对抗目标与保持原始采样轨迹的蒸馏损失相结合。例如，SDXL-Lightning [27]将DiffusionGAN损失[25]与渐进蒸馏目标[10, 13]结合，而一致性轨迹模型[26]则将GAN [35]与改进的一致性蒸馏[9]相结合。分数蒸馏最初是在文本到3D合成的背景下引入的[3639]，利用预训练的文本到图像扩散模型作为分布匹配损失。这些方法通过将渲染视图与文本条件下的图像分布对齐，利用预训练扩散模型预测的分数来优化3D对象。最近的工作将分数蒸馏[36, 37, 4042]扩展到扩散蒸馏[22, 4345]。值得注意的是，DMD [22]最小化近似KL散度，其梯度表示为两个分数函数的差：一个是固定的、预训练的，针对目标分布，另一个是动态训练，针对生成器的输出分布。

![](images/2.jpg)  

Figure 2: $1 0 2 4 \times 1 0 2 4$ samples produced by our 4-step generator distilled from SDXL. Please zoom in for details.

DMD使用扩散模型对两种评分函数进行参数化。这个训练目标比基于GAN的方法更稳定，并在一步图像合成中展现出优越的性能。一个重要的注意事项是，DMD要求使用预计算的噪声-图像配对计算回归损失以保持稳定，类似于Luhman等人的研究。我们的工作消除了这一要求。我们引入了稳定DMD训练过程的技术，不需要回归正则项，从而显著降低了由于配对数据预计算所带来的计算成本。此外，我们扩展DMD以支持多步骤生成，并整合了GAN和分布匹配方法的优势，进而在文本到图像合成中取得了最先进的成果。

# 3 背景：扩散与分布匹配蒸馏

本节简要概述了扩散模型和分布匹配蒸馏（DMD）。

扩散模型通过迭代去噪生成图像。在正向扩散过程中，噪声逐步添加到来自数据分布的样本 $x \sim p _ { \mathrm { r e a l } }$ 中，使其变成纯高斯噪声，经历预设的步数 $T$，因此在每个时间步 $t$，扩散后的样本遵循分布 $\begin{array} { r } { p _ { \mathrm { r e a l } , t } ( x _ { t } ) = \int p _ { \mathrm { r e a l } } ( x ) q ( \dot { x } _ { t } | x ) d x } \end{array}$ ，其中 $q _ { t } ( x _ { t } | x ) \sim \mathcal { N } ( \alpha _ { t } x , \sigma _ { t } ^ { 2 } \mathbf { I } )$，并且 $\alpha _ { t } , \sigma _ { t } > 0$ 是由噪声调度确定的标量 [46, 47]。扩散模型通过预测去噪估计 $\mu ( x _ { t } , t )$ ，依赖于当前的噪声样本 $x _ { t }$ 和时间步 $t$，以迭代逆转损坏过程，最终从数据分布 $p _ { \mathrm { r e a l } }$ 中生成图像。在训练后，去噪估计与数据似然函数的梯度或扩散分布的得分函数 [47] 相关：

$$
s _ { \mathrm { r e a l } } ( x _ { t } , t ) = \nabla _ { x _ { t } } \log p _ { \mathrm { r e a l } , t } ( x _ { t } ) = - \frac { x _ { t } - \alpha _ { t } \mu _ { \mathrm { r e a l } } ( x _ { t } , t ) } { \sigma _ { t } ^ { 2 } } .
$$

对图像进行采样通常需要数十到数百步去噪 [4851]。分布匹配蒸馏（DMD）通过最小化关于 $t$ 的真实目标分布 $p _ { \mathrm { r e a l } , t }$ 与生成器输出分布 $p _ { \mathrm { f a k e } , t }$ 之间的近似 Kullback-Liebler (KL) 散度的期望，将多步扩散模型蒸馏为一个单步生成器 $G$ [22]。由于 DMD 是通过梯度下降来训练 $G$ 的，因此只需要该损失的梯度，该梯度可以通过两个评分函数的差来计算：

$$
7 . \mathrm { { Z } _ { \mathrm { { D M D } } } } = \mathbb { E } _ { t } ( \nabla _ { \theta } \mathrm { K L } ( p _ { \mathrm { f a c } , t } | | p _ { \mathrm { r e a l } , t }  ) ) = - \mathbb { E } _ { t } ( \int ( s _ { \mathrm { r e a l } } ( F ( G _ { \theta } ( z ) , t ) , t ) - s _ { \mathrm { f a c } } ( F ( G _ { \theta } ( z ) , t ) , t ) ) \frac { d G _ { \theta } ( z ) } { d \theta } d z ) ,
$$

其中 $z \sim \mathcal { N } ( 0 , \mathbf { I } )$ 是随机高斯噪声输入，$\theta$ 是生成器参数，$F$ 是前向扩散过程（即噪声注入），对应时间步 $t$ 的噪声水平，而 $s _ { \mathrm { r e a l } }$ 和 $s _ { \mathrm { f a k e } }$ 是使用扩散模型 $\mu _ { \mathrm { r e a l } }$ 和 $\mu _ { \mathrm { f a k e } }$ 在各自分布上训练后近似得到的分数（公式 (1)）。DMD 使用一个冻结的预训练扩散模型作为 $\mu _ { \mathrm { r e a l } }$（教师），并在训练 $G$ 的过程中动态更新 $\mu _ { \mathrm { f a k e } }$，使用来自一步生成器的样本上的去噪分数匹配损失，即假数据 [22, 46]。Yin 等 [22] 发现需要一个额外的回归项 [16] 来正则化分布匹配梯度（公式 (2)），以实现高质量的一步模型。为此，他们收集了一组噪声-图像对 $( z , y )$，其中图像 $y$ 是基于教师扩散模型生成的，使用确定性采样器 [48, 49, 52]，以噪声图 $z$ 开始。在给定相同输入噪声 $z$ 的情况下，回归损失将生成器输出与教师的预测进行比较：

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r e g } } = \mathbb { E } _ { ( z , y ) } d ( G _ { \theta } ( z ) , y ) , } \end{array}
$$

其中 $d$ 是一个距离函数，例如在其实现中使用的 LPIPS [53]。对于像 CIFAR-10 这样的小数据集，收集这些数据的成本可以忽略不计，但在大规模文本到图像合成任务或具有复杂条件的模型中，这就成为一个显著的瓶颈 [5456]。例如，为 SDXL [57] 生成一个噪声图像对大约需要 5 秒，覆盖 LAION 6.0 数据集中的 1200 万个提示则需要大约 700 A100 天 [58]，正如 Yin 等人 [22] 所利用的那样。仅该数据集的构建成本已超过我们总训练计算的 $4 \times$（详细信息见附录 F）。这个正则化目标与 DMD 匹配学生和教师分布的目标相悖，因为它鼓励遵循教师的采样路径。

# 4 改进的分布匹配蒸馏

我们重新审视了 DMD 算法中的多个设计选择，并识别出显著的改进。

![](images/3.jpg)  

Figure 3: Our method distills a costly diffusion model (gray, right) into a one- or multi-step generator (red, left). Our training alternates between 2 steps: 1. optimizing the generator using the gradient of an implicit distribution matching objective (red arrow) and a GAN loss (green), and 2. training a score function (blue) to model the distribution of "fake" samples produced by the generator, as well as a GAN discriminator (green) to discriminate between fake samples and real images. The student generator can be a one-step or a multi-step model, as shown here, with an intermediate step input.

# 4.1 移除回归损失：真实分布匹配与更简便的大规模训练

DMD中使用的回归损失确保了模式覆盖和训练稳定性，但正如我们在第3节中讨论的，它使大规模蒸馏变得繁琐，并且与分布匹配的理念相悖，从而固有地限制了蒸馏生成器的性能与教师模型一致。我们的第一个改进是去除这个损失。

# 4.2 利用双时间尺度更新规则稳定纯分布匹配

简单省略回归目标（如公式（3）所示）会导致 DMD 训练不稳定，并显著降低质量（见表 3）。例如，我们观察到生成样本的平均亮度及其他统计数据有显著波动，未能收敛至稳定点（见附录 C）。我们将这种不稳定性归因于虚假扩散模型 $\mu _ { \mathrm { f a k e } }$ 中的近似误差，因为它并未准确追踪虚假分数，而是动态优化在生成器的非平稳输出分布上。这导致了近似误差和偏见生成器梯度（如 [30] 中讨论的）。我们借鉴 Heusel 等人 [59] 的方法，采用两时间尺度更新规则来解决这一问题。具体而言，我们以不同频率训练 $\mu _ { \mathrm { f a k e } }$ 和生成器 $G$，以确保 $\mu _ { \mathrm { f a k e } }$ 准确追踪生成器的输出分布。我们发现，在未使用回归损失的情况下，每次生成器更新时进行 5 次虚假分数更新提供了良好的稳定性，并且在 ImageNet 上的质量与原始 DMD 相匹配（见表 3），同时实现了更快的收敛。进一步的分析包含在附录 C 中。

# 4.3 使用 GAN 损失和真实数据超越教师模型

到目前为止，我们的模型在训练稳定性和性能上与 DMD [22] 可媲美，而无需昂贵的数据集构建（见表 3）。然而，蒸馏生成器与教师扩散模型之间仍存在性能差距。我们假设这一差距可能归因于 DMD 中使用的真实评分函数 $\mu _ { \mathrm { r e a l } }$ 的近似误差，这些误差会传播到生成器并导致次优结果。由于 DMD 的蒸馏模型从未使用真实数据进行训练，它无法从这些误差中恢复。我们通过在我们的流程中加入额外的 GAN 目标来解决这个问题，其中判别器被训练以区分真实图像和由我们的生成器产生的图像。使用真实数据训练的 GAN 分类器不会受到教师的限制，有可能使我们的学生生成器在样本质量上超越它。我们将 GAN 分类器集成到 DMD 中，遵循简约设计：我们在虚假扩散去噪器的瓶颈上方添加了一个分类分支（见图 3）。分类分支和 UNet 中的上游编码器特征通过最大化标准的非饱和 GAN 目标进行训练：

$$
\mathcal { L } _ { \mathrm { G A N } } = \mathbb { E } _ { x \sim p _ { \mathrm { r e a l } } , t \sim [ 0 , T ] } [ \log D ( F ( x , t ) ) ] + \mathbb { E } _ { z \sim p _ { \mathrm { n o i s e } } , t \sim [ 0 , T ] } [ - \log ( D ( F ( G _ { \theta } ( z ) , t ) ) ) ] ,
$$

其中 $D$ 是判别器，$F$ 是前向扩散过程（即，噪声注入），其定义见第三节，噪声级别对应于时间步 $t$。生成器 $G$ 旨在最小化该目标。我们的设计受到之前将扩散模型用作判别器的工作的启发[24,25,27]。我们注意到，这个GAN目标与分布匹配哲学更加一致，因为它不需要配对数据，且独立于教师的采样轨迹。

# 4.4 多步生成器

通过所提出的改进，我们能够在 ImageNet 和 COCO 上匹配教师扩散模型的性能（见表 1 和表 5）。然而，我们发现像 SDXL 这样的较大规模模型在蒸馏成一阶生成器时仍然具有挑战性，这主要是由于模型容量的限制和从噪声到高度多样化和细致图像的直接映射学习的复杂优化景观。这促使我们扩展 DMD，以支持多步采样。

我们固定一个预先设定的时间表，包含 $N$ 个时间步 $\left\{ t_{1}, t_{2}, \dots, t_{N} \right\}$，在训练和推断过程中保持一致。在推断过程中，在每一步中，我们根据一致性模型 [9] 在去噪和噪声注入步骤之间交替进行，以提高样本质量。具体而言，从高斯噪声 $\begin{array}{r} z_{0} \sim \mathcal{N}(0, \mathbf{I}) \end{array}$ 开始，我们在去噪更新 $\hat{x}_{t_{i}}^{\top} = \top G_{\theta}(x_{t_{i}}, \dot{t}_{i})$ 和前向扩散步骤 $x_{t_{i+1}} = \alpha_{t_{i+1}} \hat{x}_{t_{i}} + \sigma_{t_{i+1}} \epsilon$（其中 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$）之间进行交替，直到获得我们的最终图像 $\hat{x}_{t_{N}}$。我们的四步模型使用以下时间表：999, 749, 499, 249，对于一个经过1000步训练的教师模型。

# 4.5 多步生成器仿真以避免训练/推理不匹配

以往的多步骤生成器通常是训练来去噪带噪声的真实图像 [23,24,27]。然而，在推理过程中，除了第一个步骤是从纯噪声开始之外，生成器的输入来自于先前生成器的采样步骤 $\hat { x } _ { t _ { i } }$。这造成了训练与推理之间的不匹配，严重影响了图像质量（图 4）。我们通过在训练期间用当前学生生成器运行多个步骤生成的带噪声合成图像 $\boldsymbol { x } _ { t _ { i } }$ 替换带噪声的真实图像来解决这个问题，这与我们的推理流程 $( \ S 4 . 4 )$ 类似。之所以可行，是因为与教师扩散模型不同，我们的生成器只需运行几个步骤。我们的生成器随后对这些模拟图像进行去噪，输出结果则根据所提出的损失函数进行监督。使用带噪声的合成图像避免了不匹配，提升了整体性能（参见第5.3节）。

![](images/4.jpg)  

Figure 4: Most multi-step distillation methods simulate intermediate steps using forward diffusion during training (left). This creates a mismatch with the inputs the model sees during inference. Our proposed solution (right) remedies the problem by simulating the inference-time backward process during training.

一项并行工作，Imagine Flash [60]，提出了一种类似的技术。他们的反向蒸馏算法与我们的动机相同，旨在通过在训练时使用学生生成的图像作为后续采样步骤的输入来缩小训练与测试之间的差距。然而，他们并没有完全解决不匹配问题，因为回归损失的教师模型现在受到训练与测试差距的影响：它从未使用合成图像进行训练。这种误差在采样路径中累积。相比之下，我们的分布匹配损失不依赖于学生模型的输入，从而缓解了这一问题。

# 4.6 整合所有内容

总之，我们的蒸馏方法消除了 DMD [22] 对预先计算的噪声图像对的严格要求。它进一步整合了 GAN 的优势并支持多步生成器。如图 3 所示，从一个预训练的扩散模型开始，我们交替优化生成器 $G _ { \theta }$，以最小化原始分布匹配目标和 GAN 目标，同时使用针对假数据的去噪分数匹配目标和 GAN 分类损失来优化假分数估计器 $\mu _ { \mathrm { f a k e } }$。为了确保假分数估计准确且稳定，尽管是在在线优化中，我们以比生成器更高的频率更新它（5 步与 1 步）。

# 5 实验

我们使用多个基准来评估我们的方法 DMD2，包括在 ImageNet-$64 \times 64$ 上的类别条件图像生成 [61]，以及在 C0CO 2014 [62] 上的文本到图像合成，采用了各种教师模型 [1, 57]。我们使用 Fréchet Inception Distance (FID) [59] 来衡量图像质量和多样性，同时使用 CLIP Score [63] 评估文本与图像的一致性。对于 SDXL 模型，我们还报告了 patch FID [27, 64]，它在每张图像的 $299 \mathrm{x}$ 中心裁剪块上测量 FID，以评估高分辨率细节。最后，我们进行人工评估，以将我们的方法与其他最先进的方法进行比较。全面的评估确认，使用我们的方法训练的蒸馏模型优于之前的工作，甚至与教师模型的性能相抗衡。详细的训练和评估程序在附录中提供。

# 5.1 类条件图像生成

表1比较了我们的模型与近期基准在ImageNet-$64 \times 64$上的表现。通过一次前向传播，我们的方法在性能上显著超过现有的蒸馏技术，甚至超过了使用ODE采样器的教师模型[52]。我们将这一显著性能归因于去除DMD的回归损失（第4.1节和4.2节），这消除了ODE采样器施加的性能上限，以及我们额外的GAN项（第4.3节），这减轻了教师扩散模型得分近似错误的负面影响。

Table 1: Image quality comparison on ImageNet $6 4 \times 6 4$ .   

<table><tr><td>Method</td><td># Fwd Pass (↓)</td><td>FID ()</td></tr><tr><td>BigGAN-deep [65]</td><td>1</td><td>4.06</td></tr><tr><td>ADM [66]</td><td>250</td><td>2.07</td></tr><tr><td>RIN [67]</td><td>1000</td><td>1.23</td></tr><tr><td>StyleGAN-XL [35]</td><td>1</td><td>1.52</td></tr><tr><td>Progress. Distill. [10]</td><td>1</td><td>15.39</td></tr><tr><td>DFNO [68]</td><td>1</td><td>7.83</td></tr><tr><td>BOOT [20]</td><td>1</td><td>16.30</td></tr><tr><td>TRACT [33]</td><td>1</td><td>7.43</td></tr><tr><td>Meng et al. [13]</td><td>1</td><td>7.54</td></tr><tr><td>Diff-Instruct [44]</td><td>1</td><td>5.57</td></tr><tr><td>Consistency Model [9]</td><td>1</td><td>6.20</td></tr><tr><td>iCT-deep [12]</td><td>1</td><td>3.25</td></tr><tr><td>CTM [26]</td><td>1</td><td>1.92</td></tr><tr><td>DMD [22]</td><td>1</td><td>2.62</td></tr><tr><td>DMD2 (Ours)</td><td>1</td><td>1.51</td></tr><tr><td>+longer training (Ours)</td><td>1</td><td>1.28</td></tr><tr><td>EDM (Teacher, ODE) [52]</td><td>511</td><td>2.32</td></tr><tr><td>EDM (Teacher, SDE) [52]</td><td>511</td><td>1.36</td></tr></table>

Table 2: Image quality comparison with SDXL backbone on 10K prompts from COCO 2014.   

<table><tr><td>Method</td><td># Fwd Pass (↓)</td><td>FID (↓)</td><td>Patch FID (↓)</td><td>CLIP ()</td></tr><tr><td>LCM-SDXL [32]</td><td>1 4</td><td>81.62 22.16</td><td>154.40 33.92</td><td>0.275 0.317</td></tr><tr><td>SDXL-Turbo [23]</td><td>1 4</td><td>24.57 23.19</td><td>23.94 23.27</td><td>0.337 0.334</td></tr><tr><td>SDXL</td><td>1</td><td>23.92</td><td>31.65</td><td>0.316</td></tr><tr><td>Lightning [27]</td><td>4</td><td>24.46</td><td>24.56</td><td>0.323</td></tr><tr><td>DMD2 (Ours)</td><td>1 4</td><td>19.01 19.32</td><td>26.98 20.86</td><td>0.336 0.332</td></tr><tr><td>SDXL</td><td></td><td></td><td></td><td></td></tr><tr><td>Teacher, cfg=6 [57]</td><td>100</td><td>19.36</td><td>21.38</td><td>0.332</td></tr><tr><td>SDXL Teacher, cfg=8 [57]</td><td>100</td><td>20.39</td><td>23.21</td><td>0.335</td></tr></table>

# 5.2 文本生成图像

我们在零-shot COCO 2014 数据集上评估了 DMD2 的文本到图像生成性能。我们的生成器分别通过蒸馏 SDXL 和 SD v1.5 进行训练，使用来自 LAION-Aesthetics 的 300 万个提示的子集。此外，我们从 LAIONAesthetic 收集了 $500 \mathrm{k}$ 张图像作为 GAN 判别器的训练数据。表 2 总结了 SDXL 模型的蒸馏结果。我们的四步生成器生成了高质量和多样化的样本，达到了 19.32 的 FID 分数和 0.332 的 CLIP 分数，在图像质量和提示一致性方面与教师扩散模型相媲美。为了进一步验证我们方法的有效性，我们进行了一项广泛的用户研究，将我们模型的输出与教师模型和现有蒸馏方法的输出进行比较。我们采用了来自 PartiPrompts 的 128 个提示的子集，遵循 LADD 的方法。在每次比较中，我们随机选取五位评估者，选择更具视觉吸引力的图像，以及更好地代表文本提示的图像。关于人类评估的详细信息包含在附录 H 中。如图 5 所示，我们的模型在用户偏好方面远高于基线方法。值得注意的是，我们的模型在 $24\%$ 的样本中超越了其教师在图像质量上的表现，并且获得了可比的提示一致性，同时所需的前向通过次数减少了 $25$ 倍（4 vs 100）。定性比较见于图 6。在附录 A 的表 5 中提供了 SDv1.5 的结果。同样，使用 DMD2 训练的一步模型超越了所有以前的扩散加速方法，获得了 8.35 的 FID 分数，比原始 DMD 方法提高了显著的 3.14 点。我们的结果也超过了使用 50 步 PNDM 采样器的教师模型的表现。

![](images/5.jpg)  

Figure 5: User study comparing our distilled model with its teacher and competing distillation baselines [23, 27, 31]. All distilled models use 4 sampling steps, the teacher uses 50. Our model achieves the best performance for both image quality and prompt alignment.

# 5.3 消融研究

Table 3: Ablation studies on ImageNet. TTUR stands for two-timescale update rule.   

<table><tr><td colspan="4">DMD No Regress. TTUR GAN FID (↓)</td></tr><tr><td></td><td></td><td></td><td>2.62</td></tr><tr><td></td><td></td><td></td><td>3.48</td></tr><tr><td></td><td>Y ✓</td><td></td><td>2.61</td></tr><tr><td>&gt;&gt;&gt;&gt;</td><td></td><td></td><td>1.51</td></tr><tr><td></td><td></td><td></td><td>2.56</td></tr><tr><td></td><td>✓</td><td>Y</td><td>2.52</td></tr></table>

Table 4: Ablation studies with SDXL backbone on 10K prompts from COCO 2014.   

<table><tr><td>Method</td><td colspan="3">FID (↓) Patch FID (↓) CLIP (↑)</td></tr><tr><td>w/o GAN</td><td>26.90</td><td>27.66</td><td>0.328</td></tr><tr><td>w/o Distribution Matching</td><td>13.77</td><td>27.96</td><td>0.307</td></tr><tr><td>w/o Backward Simulation</td><td>20.66</td><td>24.21</td><td>0.332</td></tr><tr><td>DMD2 (Ours)</td><td>19.32</td><td>20.86</td><td>0.332</td></tr></table>

表3展示了我们提出的方法在ImageNet上不同组件的消融实验。仅仅去掉原始DMD中的ODE回归损失就导致了FID下降到3.48，因为训练不稳定（具体分析见附录C）。然而，采用我们的双时间尺度更新规则（TTUR，4.2节）缓解了这一性能下降，使其达到了与DMD基线相当的性能，而不需要额外的数据集构建。添加我们的GAN损失进一步提高了FID，增幅为1.1点。我们的综合方法超越了单独使用GAN（没有分布匹配目标）的性能，并且将双时间尺度更新规则应用于单独的GAN并没有提升其性能，突显了在统一框架中将分布匹配与GAN结合的有效性。

在表4中，我们消融了GAN项的影响（第4.3节）、分布匹配目标（公式2）和反向模拟（第4.4节），用于将SDXL模型蒸馏为一个四步生成器。定性结果如图7所示。在没有GAN损失的情况下，我们的基线模型生成的图像出现了过饱和和过平滑的现象（图7第三列）。同样，消除分布匹配目标（公式2）将我们的方法简化为一种纯GAN方法，这在训练稳定性方面面临困难[70, 71]。此外，纯GAN方法也缺乏自然的方式来融入无分类器指导[72]，而这是高质量文本生成图像合成所必需的[1, 2]。因此，尽管基于GAN的方法通过紧密匹配真实分布达到了最低的FID，但在文本对齐和美学质量上却明显表现不佳（图7第二列）。同样，省略反向模拟会导致图像质量下降，正如下降的补丁FID分数所指示的那样。

![](images/6.jpg)  
A photo of llama wearing sunglasses standing on the deck of a spaceship with the Earth in the background   

Figure 6: Visual comparison between our model, the SDXL teacher, and selected competing methods [23, 27, 31]. All distilled models use 4 sampling steps while the teacher model uses 50 sampling steps with classifier-free guidance. All images are generated using identical noise and text prompts. Our model produces images with superior realism and text alignment. (Zoom in for details.) More comparisons are available in Appendix Figure 10.

# 6 限制因素

在实现卓越的图像质量和文本对齐的同时，我们的蒸馏生成器在图像多样性上相比于教师模型略有下降（见附录B）。此外，我们的生成器仍需经过四个步骤以匹配最大的SDXL模型的质量。这些局限性虽然并非我们模型所独有，但强调了进一步改进的领域。与大多数以前的蒸馏方法一样，我们在训练过程中使用固定的引导比例，限制了用户的灵活性。引入可变引导比例可能是未来研究的一个有希望的方向。此外，我们的方法针对分布匹配进行了优化；结合人类反馈或其他奖励函数可能进一步提升性能。最后，训练大规模生成模型计算密集，使得大多数研究人员无法接触。我们希望

![](images/7.jpg)

![](images/8.jpg)  
A soft beam of light shines down on an armored granite wombat warrior statue holding a broad sword. The statue stands an ornate pedestal in the cella of a temple. wide-angle lens. anime oil painting.

一位女性面部特写，映衬在昏暗复古餐馆中霓虹灯的柔和光辉下，暗示着渴望和怀旧的故事。电影感照片中，一位美丽女孩在丛林中骑着恐龙，周围泥泞，阳光明媚，天空清澈。35mm 摄影，胶卷，专业，4K，细节丰富。

![](images/9.jpg)  

Figure 7: SDXL Qualitative Ablations. All images are generated using identical noise and text prompts. Removing the distribution matching objective significantly degrades aesthetic quality and text alignment. Omitting the GAN loss results in oversaturated and overly smoothed images. The baseline without backward simulation produces images of lower quality.

我们高效的方法和优化的用户友好型代码库将帮助实现该领域未来研究的民主化。

# 7 更广泛的影响

我们在提高扩散模型效率和质量方面的工作可能带来多种社会影响，包括正面和负面影响。在正面方面，快速图像合成的进展可以显著惠及各类创意产业。这些模型能够通过为艺术家提供强大的工具，来高效生成高质量视觉内容，从而提升图形设计、动画和数字艺术。此外，改进的文本图像合成能力可用于教育和娱乐，帮助创建个性化学习材料和沉浸式体验。然而，潜在的负面社会影响也必须考虑。滥用风险包括生成虚假信息和创建虚假个人资料，这可能传播错误信息并操纵公众舆论。部署这些技术可能导致对特定群体的不公平偏见，尤其是如果模型在有偏数据集上进行训练，这可能会延续或加剧现有社会偏见。为了减轻这些风险，我们有兴趣开发监测机制以检测和防止滥用，并探索提高输出多样性和公平性的方法。

# 8 致谢

我们感谢姜敏国和金承旭在进行人工评估方面的帮助。我们还感谢来泽强提出在我们的一步生成器中使用的时间步移技术。此外，我们感谢我们的朋友和同事们的深刻讨论和宝贵意见。本工作得到了国家科学基金会的支持，合作协议号为PHY-2019786（国家科学基金会人工智能与基础相互作用研究所，http://iaifi.org/），以及NSF资助2105819、NSF CISE奖1955864，以及来自谷歌、光州科技学院、亚马逊和广达电脑的资助。

# References

[1] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.   
[2] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.   
[3] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.   
[4] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-toimage diffusion models with deep language understanding. In NeurIPS, 2022.   
[5] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, et al. ediff: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022.   
[6] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.   
[7] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In CVPR, 2022.   
[8] Shaoan Xie, Zhifei Zhang, Zhe Lin, Tobias Hinz, and Kun Zhang. Smartbrush: Text and shape guided object inpainting with diffusion model. In CVPR, 2023.   
[9] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In ICML, 2023.   
10 Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In ICLR, 2022.   
1 Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, and Qiang Liu. Instafow: One step is enough for high-quality diffusion-based text-to-image generation. arXiv preprint arXiv:2309.06380, 2023.   
[12] Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. In ICLR, 2024.   
[13] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In CVPR, 2023.   
[14] Jonathan Heek, Emiel Hoogeboom, and Tim Salimans. Multistep consistency models. arXiv preprint arXiv:2403.06807, 2024.   
[15] Hanshu Yan, Xingchao Liu, Jiachun Pan, Jun Hao Liew, Qiang Liu, and Jiashi Feng. Perlow: Piecewise rectified flow as universal plug-and-play accelerator. arXiv preprint arXiv:2405.07510, 2024.   
[16] Eric Luhman and Troy Luhman. Knowledge distillation in iterative generative models for improved sampling speed. arXiv preprint arXiv:2101.02388, 2021.   
[17] Yuxi Ren, Xin Xia, Yanzuo Lu, Jiacheng Zhang, Jie Wu, Pan Xie, Xing Wang, and Xuefeng Xiao. Hyper-sd: Trajectory segmented consistency model for efficient image synthesis. arXiv preprint arXiv:2404.13686, 2024.   
u    ub  , oZ image generation with sub-path linear approximation model. arXiv preprint arXiv:2404.13903, 2024.   
[19] Jianbin Zheng, Minghui Hu, Zhongyi Fan, Chaoyue Wang, Changxing Ding, Dacheng Tao, and Tat-Jen Cham. Trajectory consistency distillation. arXiv preprint arXiv:2402.19159, 2024.   
[0 JiataoGu, Shuanei Zhai, Yizhe Zhang, Lingj Liu, and JoshuaM Susskind Boot: Data-ree istillation of denoising diffusion models with bootstrapping. In ICML 2023 Workshop on Structured Probabilistic Inference & Generative Modeling, 2023.   
[21] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative Adversarial Nets. In NIPS, 2014.   
[22] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In CVPR, 2024.   
[23] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. arXiv preprint arXiv:2311.17042, 2023.   
[24] Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombac.Fast high-resolution image synthesis with latent adversarial diffusion distillation.arXiv preprint arXiv:2403.12015, 2024.   
[25] Yanwu Xu, Yang Zhao, Zhisheng Xiao, and Tingbo Hou. Ufogen: You forward once large scale text-toimage generation via diffusion gans. In CVPR, 2024.   
[26] Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ode trajectory of diffusion. In ICLR, 2024.   
[27]Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxl-ightning: Progressive adversarial diffusion distillation. arXiv, 2024.   
[28] Zhendong Wang, Huangjie Zheng, Pengcheng He, Weizhu Chen, and Mingyuan Zhou. Diffusion-gan: Training gans with diffusion. In ICLR, 2023.   
T diffusion gans. In ICLR, 2022.   
[30] Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity distilation: Exponentially fast distillation f pretrained diffusion models or one-step generation. In IC, 2024.   
[31] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.   
[32] Simian Luo, Yiqin Tan, Suraj Patil, Daniel Gu, Patrick von Platen, Apolinário Passos, Longbo Huang, Jian Li, and Hang Zhao. Lcm-lora:Auniversal stable-diffusion acceleration module.arXiv preprint arXiv:2310.04378, 2023.   
[33] David Berthelot, Arnaud Autef, Jierui Lin, Dian Ang Yap, Shuangfei Zhai, Siyuan Hu, Daniel Zheng, Walter Talbot, and Eric Gu. Tract: Denoising diffusion models with transitive closure time-distillation. arXiv preprint arXiv:2303.04248, 2023.   
[34] Axel Sauer, Kashyap Chitta, Jens Müller, and Andreas Geiger. Projected gans converge faster. In NeurIPS, 2021.   
[5Axel Sauer, Katj Schwarz, and Andreas Geiger. Stylegan-x: Scaling stylegan to large diverse datasets. In SIGGRAPH, 2022.   
[36] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In ICLR, 2023.   
[37] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. arXiv preprint arXiv:2305.16213, 2023.   
[38] Amir Hertz, Kfir Aberman, and Daniel Cohen-Or. Delta denoising score. In ICCV, 2023.   
[39] Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A Yeh, and Greg Shakhnarovich. Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In CVPR, 2023.   
[40] Mingxuan Yi, Zhanxing Zhu, and Song Liu. Monoflow: Rethinking divergence gans via the perspective of wasserstein gradient flows. In ICML, 2023.   
[41] Siddarth Asokan, Nishanth Shetty, Aadithya Srikanth, and Chandra Sekhar Seelamantula. Gans settle scores! arXiv preprint arXiv:2306.01654, 2023.   
[42] Romann M Weber. The score-difference fow for implicit generative modeling. In TMLR, 2023.   
[43] Jean-Yves Franceschi, Mike Gartrell, Ludovic Dos Santos, Thibaut Issenhuth, Emmanuel de Bézenac, Mickaël Chen, and Alain Rakotomamonjy. Unifying gans and score-based diffusion as generative particle models. In NeurIPS, 2023.   
[44] Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, and Zhihua Zhang. Diff-instruct: A universal approach for transferring knowledge from pre-trained diffusion models. In NeurIPS, 2023.   
[45] Thuan Hoang Nguyen and Anh Tran. Swiftbrush: One-step text-to-image diffusion model with variational score distillation. In CVPR, 2024.   
[46] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020.   
[47] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021.   
[48] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021.   
[49] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. In ICLR, 2022.   
[50] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. In arXiv preprint arXiv:2211.01095, 2022.   
[51] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. In NeurIPS, 2022.   
[52] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In NeurIPS, 2022.   
[53] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.   
[54] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023.   
[T Broks,Aleka Holynsk an lex  . Insrx2ix: Lea oll instructions. In CVPR, 2023.   
[56] Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yv Taigman. Emu edt: Prei mage editng vi recogntion nd generation tasks.arXiv preint arXiv:2311.10089, 2023.   
[57] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.   
[58] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. In NeurIPS, 2022.   
[59] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017.   
[60] Jonas Kohler, Albert Pumarola, Edgar Schönfeld, Artsiom Sanakoyeu, Roshan Sumbaly, Peter Vajda, and AiThabet. Imagine fash:Accelerating emu diffusion models with backward distillationarXiv prerint arXiv:2405.05224, 2024.   
[Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, andLiFei-Fei. Imagenet:A large-scale hierarchial image database. In CVPR, 2009.   
[62] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014.   
[63] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021.   
[64] Lucy Chai, Michael Gharbi, Eli Shechtman, Phlip Isola, and Richard Zhang.Any-resolution training for high-resolution image synthesis. In ECCV, 2022.   
[65] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. In ICLR, 2019.   
[66] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In NeurIPS, 2021.   
[7Allan Jabri, David Fleet, and Ting Chen. Scalable adaptive computation for iterative generation. In ICML, 2023.   
[68] Hongkai Zheng, Weili Nie, Arash Vahdat, Kamyar Azizzadenesheli, and Anima Anandkumar. Fast sampling of diffusion models via operator learning. In ICML, 2023.   
[69] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2022.   
[70] Lars Mescheder, Andreas Geiger, and Sebastian Nowozin. Which training methods for gans do actually converge? In ICML, 2018.   
[71] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In CVPR, 2023.   
[72] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In arXiv preprint arXiv:2207.12598, 2022.   
[73] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Eron, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. arXiv preprint arXiv:2311.12908, 2023.   
[74] Zhendong Wang, Jianmin Bao, Wengang Zhou, Weilun Wang, Hezhen Hu, Hong Chen, and Houqiang Li. Dire for diffusion-generated image detection. In ICCV, 2023.   
[75] Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew Owens, and Alexei A Efros. Cnn-generated images are surprisingly easy to spot.. for now. In CVPR, 2020.   
[76] Xug Shen, Chao Du, Tianyu Pang, Min Lin, Yongkang Wong, and Mohan Kankanhal. Finug text-to-image diffusion models for fairness. In ICLR, 2024.   
[77] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, 2021.   
[78] Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Make-a-scene: Scene-based text-to-image generation with human priors. In ECCV, 2022.   
[79] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In ICML, 2022.   
[80] Yufan Zhou, Ruiyi Zhang, Changyou Chen, Chunyuan Li, Chris Tensmeyer, Tong Yu, Jiuxiang Gu, Jinhui Xu, and Tong Sun. Towards language-free training for text-to-image generation. In CVPR, 2022.   
[81] Axel Sauer, Tero Karras, Samuli Laine, Andreas Geiger, and Timo Aila. Stylegan-t: Unlocking the power of gans for fast large-scale text-to-image synthesis. ICML, 2023.   
[82] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. Unipc: A unifed predictor-corrector framework for fast sampling of diffusion models. arXiv preprint arXiv:2302.04867, 2023.   
[83] Yifan Zhang and Bryan Hooi. Hip: Enabling one-step text-to-image diffusion models via high-frequencypromoting adaptation. arXiv preprint arXiv:2311.18158, 2023.   
[84] Xun Huang, Ming-Yu Liu, Serge Belongie, and Jan Kautz. Multimodal unsupervised image-to-image translation. In ECCV, 2018.   
[85] Jun-Yan Zhu, Richard Zhang, Deepak Pathak, Trevor Darrell, Alexei A Efros, Oliver Wang, and Eli Shechtman. Toward multimodal image-to-image translation. In NeurIPS, 2017.   
[86] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019.   
[87] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart- \sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. arXiv preprint arXiv:2403.04692, 2024.   
[88] Zeqiang Lai. Opendmd: Open source implementation and models of one-step diffusion with distribution matching distillation. https://github. com/Zeqiang-Lai/OpenDMD, 2024. Accessed: 2024-05-21.   
[89] Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu. On aliased resizing and surprising subtleties in gan evaluation. In CVPR, 2022.

# Table of Contents

1 Introduction 1   
2 Related Work 3   
3 Background: Diffusion and Distribution Matching Distillation 5   
4 Improved Distribution Matching Distillation 6   
4.1 Removing the regression loss: true distribution matching and easier large-scale training 6   
4.2 Stabilizing pure distribution matching with a Two Time-scale Update Rule . 6   
4.3 Surpassing the teacher model using a GAN loss and real data 6   
4.4 Multi-step generator . 7   
4.5 Multi-step generator simulation to avoid training/inference mismatch 7   
4.6 Putting everything together 7

# ; Experiments

# 8

5.1 Class-conditional Image Generation 8   
5.2 Text-to-Image Synthesis 8   
5.3 Ablation Studies . 9   
6 Limitations 10   
7 Broader Impact 12   
8 Acknowledgements 12   
A SD v1.5 Results 17   
B Text-to-Image Synthesis Further Analysis 17   
C Two Time-scale Update Rule Further Analysis 18   
D Additional Text-to-Image Synthesis Results 20   
E ImageNet Visual Results 22   
F Implementation Details 23   
F.1 GAN Classifier Design 23   
F.2 ImageNet 23   
F.3 SD v1.5 23   
F.4 SDXL 23   
G Evaluation Details 24   
H User Study Details 24   
I Prompts for Figure 1, Figure 2, and Figure 11 25

# A SD v1.5 Results

Table 5 presents detailed comparisons between our one-step generator distilled from SD v1.5 and competing approaches.

# B Text-to-Image Synthesis Further Analysis

Qualitative ablation results using SDXL backbone are shown in Figure 7. Additionally, we compare the image diversity of our 4-step generator with other competing approaches distilled from SDXL [23, 27,31]. We employ an LPIPS-based diversity score, similar to that used in multi-modal image-toimage translation [84, 85]. Specifically, we generate four images per prompt and calculate the average pairwise LPIPS distance [53]. For this evaluation, we use the LADD [24] subset of PartiPrompts [69]. We also report the FID and CLIP score measured on 10K prompts from COCO 2014 on the side. Table 6 summarizes the results. Our model achieves the best image quality, indicated by the lowest FID and Patch FID scores. We also achieve text alignment comparable to SDXL-Turbo while attaining a better diversity score. While SDXL-Lightning [27] exhibits a higher diversity score than our approach, it suffers from considerably worse text alignment, as reflected by the lower CLIP score and human evaluation (Fig. 5). This suggests that the improved diversity is partially due to random outputs lacking prompt coherence. We note that it is possible to increase the diversity of our model by raising the weights for the GAN objective, which aligns with the more diverse unguided distribution. Further investigation into finding the optimal balance between distribution matching and the GAN objective is left for future work.

Table 5: Sample quality comparison on 30K prompts from COCO 2014.   

<table><tr><td>Family</td><td>Method</td><td colspan="3">Resolution (↑) Latency (↓) FID (↓)</td></tr><tr><td rowspan="9">Original, unaccelerated</td><td>DALL·E [77]</td><td>256</td><td></td><td>27.5</td></tr><tr><td>DALL·E 2 [3]</td><td>256</td><td></td><td>10.39</td></tr><tr><td>Parti-750M [69]</td><td>256</td><td>-</td><td>10.71</td></tr><tr><td>Parti-3B [69]</td><td>256</td><td>6.4s</td><td>8.10</td></tr><tr><td>Make-A-Scene [78]</td><td>256</td><td>25.0s</td><td>11.84</td></tr><tr><td>GLIDE [79]</td><td>256</td><td>15.0s</td><td>12.24</td></tr><tr><td>LDM [1]</td><td>256</td><td>3.7s</td><td>12.63</td></tr><tr><td>Imagen [4]</td><td>256</td><td>9.1s</td><td>7.27</td></tr><tr><td>eDiff-I [5]</td><td>256</td><td>32.0s</td><td>6.95</td></tr><tr><td rowspan="3">GANs</td><td>LAFITE [80]</td><td>256</td><td>0.02s</td><td>26.94</td></tr><tr><td>StyleGAN-T [81]</td><td>512</td><td>0.10s</td><td>13.90</td></tr><tr><td>GigaGAN [71]</td><td>512</td><td>0.13s</td><td>9.09</td></tr><tr><td rowspan="9">Accelerated diffusion</td><td>DPM++ (4 step) [50]</td><td>512</td><td>0.26s</td><td>22.36</td></tr><tr><td>UniPC (4 step) [82]</td><td>512</td><td>0.26s</td><td>19.57</td></tr><tr><td>LCM-LoRA (4 step) [32]</td><td>512</td><td>0.19s</td><td>23.62</td></tr><tr><td>InstaFlow-0.9B [11]</td><td>512</td><td>0.09s</td><td>13.10</td></tr><tr><td>SwiftBrush [45]</td><td>512</td><td>0.09s</td><td>16.67</td></tr><tr><td>HiPA [83]</td><td>512</td><td>0.09s</td><td>13.91</td></tr><tr><td>UFOGen [25]</td><td>512</td><td>0.09s</td><td>12.78</td></tr><tr><td>SLAM (4 step) [18]</td><td>512</td><td>0.19s</td><td>10.06</td></tr><tr><td>DMD [22]</td><td>512</td><td>0.09s</td><td>11.49</td></tr><tr><td rowspan="2">Teacher</td><td>DMD2 (Ours)</td><td>512</td><td>0.09s</td><td>8.35</td></tr><tr><td>SDv1.5 (50 step, cfg=3, ODE) [1, 49]</td><td>512</td><td>2.59s</td><td>8.59</td></tr><tr><td></td><td>SDv1.5 (200 step, cfg=2, SDE) [1, 46]</td><td>512</td><td>10.25s</td><td>7.21</td></tr></table>

Table 6: Image quality and diversity comparison with SDXL backbone.   

<table><tr><td>Method</td><td># Fwd Pass (↓)</td><td>FID (↓)</td><td>Patch FID (↓)</td><td>CLIP (↑)</td><td>Diversity Score (↑)</td></tr><tr><td>LCM-SDXL [32]</td><td>4</td><td>22.16</td><td>33.92</td><td>0.317</td><td>0.61</td></tr><tr><td>SDXL-Turbo [23]</td><td>4</td><td>23.19</td><td>23.27</td><td>0.334</td><td>0.58</td></tr><tr><td>SDXL-Lightning [27]</td><td>4</td><td>24.46</td><td>24.56</td><td>0.323</td><td>0.63</td></tr><tr><td>DMD2 (Ours)</td><td>4</td><td>19.32</td><td>20.86</td><td>0.332</td><td>0.61</td></tr><tr><td>SDXL-Teacher, cfg=6 [57]</td><td>100</td><td>19.36</td><td>21.38</td><td>0.332</td><td>0.64</td></tr><tr><td>SDXL-Teacher, cfg=8 [57]</td><td>100</td><td>20.39</td><td>23.21</td><td>0.335</td><td>0.64</td></tr></table>

# C Two Time-scale Update Rule Further Analysis

In Section 4.2, we discuss that updating the fake score multiple times (5 updates) per generator update leads to better stability. Here, we provide further analysis. Figure 8 visualizes pixel brightness variations throughout training. The baseline approach, which omits the regression objective from DMD and uses just 1 fake score update, results in significant training instability, as evidenced by periodic fluctuations in pixel brightness. In contrast, our two time-scale update rule with 5 fake score updates per generator update stabilizes the training and leads to better sample quality, as shown in Tab. 3.

We further examine the influence of the update frequency for the fake diffusion model $\mu _ { \mathrm { f a k e } }$ in Figure 9. An update frequency of 1 fake diffusion update per generator update corresponds to the naive baseline (red line) and suffers from training instability. Although a frequency of 10 updates (magenta line) provides excellent stability, it significantly slows down the training process. We found that a moderate frequency of 5 updates (green line) achieves the best balance between stability and convergence speed on ImageNet. Our approach proves more effective than using asynchronous learning rates [59] (cyan line) and converges significantly faster than the original DMD method that employs a regression loss [22] (dark blue line). For new models and datasets, we recommend adjusting the iteration number to the smallest value that ensures the stability of general image statistics, such as pixel brightness.

![](images/10.jpg)  
Figure 8: Visualization of pixel brightness variations throughout training. The baseline approach, which naively removes the regression loss from the original DMD [22], suffers from significant training instability, leading to fluctuating general image statistics like the overall pixel brightness. In contrast, our two time-scale update rule, which optimizes the fake diffusion model five times per generator update, significantly stabilizes training and enhances sample quality.

![](images/11.jpg)  
Figure 9: Visualization of FID score progression during training. Naively removing the regression loss leads to training instability (red line). A two time-scale update rule with five fake diffusion critic updates per generator update stabilizes training and is more effective than using a larger number of fake diffusion updates or an asynchronous learning rate where the fake diffusion model uses a learning rate 5 times larger than the generator. The model trained with our two time-scale update rule (green) also converges significantly faster than the original DMD method with a regression loss (dark blue), even though TTUR performs less number of the generator weight updates.

# D Additional Text-to-Image Synthesis Results

Additional visual comparisons for the 4-step distilled models are shown in Figure 10. Sample outputs from our one-step generator are presented in Figure 11.

![](images/12.jpg)  
Figure 10: Additional visual comparison between our model, the SDXL teacher, and selected competing methods [23, 27,31]. All distilled models use 4 sampling steps while the teacher model uses 50 sampling steps with classifier-free guidance. All images are generated using identical noise and text prompts. Our model produces images with superior realism and text alignment. Please zoom in for details.

![](images/13.jpg)  
Figure 11: Additional $1 0 2 4 \times 1 0 2 4$ samples produced by our 1-step generator distilled from SDXL. Please zoom in for details.

# E ImageNet Visual Results

In Figure 12, we present qualitative results obtained from our one-step distilled model trained on the ImageNet dataset.

![](images/14.jpg)  
Figure 12: One-step samples from our generator trained on ImageNet $\mathrm { { F I D } = 1 } . 2 8$ . Please zoom in for details.

# F Implementation Details

This section provides a brief overview of the implementation details. All results presented can be easily reproduced using our open-source training and evaluation code.

# F.1 GAN Classifier Design

Our GAN classifier design is inspired by SDXL-Lightning [27]. Specifically, we attach a prediction head to the middle block output of the fake diffusion model. The prediction head consists of a stack of $4 \times 4$ convolutions with a stride of 2, group normalization, and SiLU activations. All feature maps are downsampled to $4 \times 4$ resolution, followed by a single convolutional layer with a kernel size and stride of 4. This layer pools the feature maps into a single vector, which is then passed to a linear projection layer to predict the classification result.

# F.2 ImageNet

Our ImageNet implementation closely follows the DMD paper [22]. Specifically, we distill a one-step generator from the EDM pretrained model [52], released under the CC BY-NC-SA 4.0 License. For the standard training setup, we use the AdamW optimizer [86] with a learning rate of $2 \times 1 0 ^ { - 6 }$ ,a weight decay of 0.01, and beta parameters (0.9, 0.999). We use a batch size of 280 and train the model on 7 A100 GPUs for 200K iterations, which takes approximately 2 days. The number of fake diffusion model update per generator update is set to 5. The weight for the GAN loss is set to $3 \times 1 0 ^ { - 3 }$ . For the extended training setup shown in Table 1, we first pretrain the model without GAN loss for 400K iterations. We then resume from the best checkpoint (as measured by FID), enable the GAN loss with a weight of $3 \times 1 0 ^ { - 3 }$ , reduce the learning rate to $5 \times 1 0 ^ { - 7 }$ , and continue training for an additional 150K iterations. The total training time for this run is approximately 5 days.

# F.3 SD v1.5

We distill a one-step generator from the SD v1.5 model [1], released under the CreativeML Open RAIL-M license, using prompts from the LAION-Aesthetic $6 . 2 5 +$ dataset [58]. Additionally, we collect 500K images from LAION-Aesthetic $5 . 5 +$ as training data for the GAN discriminator, filtering out images smaller than $1 0 2 4 \times 1 0 2 4$ and those containing unsafe content. Our training process involves two stages. In the first stage, we disable the GAN loss and use the AdamW optimizer with a learning rate of $1 \times 1 0 ^ { - 5 }$ , a weight decay of 0.01, and beta parameters of (0.9, 0.999). The fake diffusion model is updated 10 times per generator update. We set the guidance scale for the real diffusion model to be 1.75. We use a batch size of 2048 and train the model on 64 A100 GPUs for 40K iterations. In the second stage, we enable the GAN loss with a weight of $1 0 ^ { - 3 }$ , reduce the learning rate to $5 \times 1 0 ^ { - 7 }$ , and continue training for an additional 5K iterations. The total training time is approximately 26 hours.

# F.4 SDXL

We train both one-step and four-step generators by distilling from the SDXL model [57], released under the CreativeML Open RAII $\mathbf { \Gamma } _ { + + - \mathbf { M } }$ License. For the one-step generator, we observed similar block noise artifacts as reported in SDXL-Lightning [27] and Pixart-Sigma [87]. We addressed this by adopting the timestep shift technique from OpenDMD [88] and Pixart-Sigma [87], setting the conditioning timestep to 399. Additionally, we initialized the one-step generator by pretraining it with a regression loss using a small set of 10K pairs for a short period. These adjustments are not necessary for the multi-step model or other backbones, suggesting this issue might be specific to SDXL. Similar to SD v1.5, we use prompts from the LAION-Aesthetic $6 . 2 5 +$ dataset [58] and collect 500K images from LAION-Aesthetic $5 . 5 +$ as training data for the GAN discriminator, filtering out images smaller than $1 0 2 4 \times 1 0 2 4$ and those containing unsafe content. The generator is trained using the AdamW optimizer with a learning rate of $5 \times 1 0 ^ { - 7 }$ , a weight decay of 0.01, and beta parameters of (0.9, 0.999). The fake diffusion model is updated 5 times per generator update. We set the guidance scale for the real diffusion model to be 8. We use a batch size of 128 and train the model on 64 A100 GPUs for 20K iterations for the 4-step generator and 25K iterations for the 1-step generator, taking approximately 60 hours.

# G Evaluation Details

For the COCO experiments, we follow the exact evaluation setup as GigaGAN [71] and DMD [22]. For the results presented in Table 5, we use 30K prompts from the C0CO 2014 validation set and generate the corresponding images. The outputs are downsampled to $2 5 6 \times 2 5 6$ and compared with 40,504 real images from the same validation set using clean-FID [89]. For the results presented in Table 2, we use a random set of 10K prompts from the COCO 2014 validation set and generate the corresponding images. The outputs are downsampled to $5 1 2 \times 5 1 2$ and compared with the corresponding 10K real images from the validation set with the same prompts. We compute the CLIP score using the OpenCLIP-G backbone. For the ImageNet results, we generate 50,000 images and calculate the FID statistics using EDM's evaluation code [52].

# H User Study Details

To conduct the human preference study, we use the Prolific platform (https://www.prolific.com). We use 128 prompts from the LADD [24] subset of PartiPrompts [69]. All approaches generate corresponding images, which are presented in pairs to human evaluators to measure aesthetic and prompt alignment preference. The specific questions and interface are shown in Figure 13. Consent is obtained from the voluntary participants, who are compensated at a flat rate of 12 dollars per hour. We manually verify that all generated images contain standard visual content that poses no risks to the study participants.

![](images/15.jpg)  
Figure 13: A sample interface for our user preference study, where images are presented in a random left/right order.

# I Prompts for Figure 1, Figure 2, and Figure 11

We use the following prompts for Figure 1. From left to right, top to bottom:

•a girl examining an ammonite fossil   
A photo of an astronaut riding a horse in the forest.   
a giant gorilla at the top of the Empire State Building   
•A close-up photo of a wombat wearing a red backpack and raising both arms in the air. Mount Rushmore is in the background.   
An oil painting of two rabbits in the style of American Gothic, wearing the same clothes as in the original.   
a portrait of an old man   
a watermelon chair   
A sloth in a go kart on a race track. The sloth is holding a banana in one hand. There is a banana peel on the track in the background.   
•a penguin standing on a sidewalk   
•a teddy bear on a skateboard in times square

We use the following prompts for Figure 2. From left to right, top to bottom:

•a chimpanzee sitting on a wooden bench   
a cat reading a newspaper   
•A television made of water that displays an image of a cityscape at night.   
•a portrait of a statue of the Egyptian god Anubis wearing aviator goggles, white t-shirt and leather jacket. The city of Los Angeles is in the background.   
•a squirrell driving a toy car   
an elephant walking on the Great Wall   
a capybara made of voxels sitting in a field   
Cinematic photo of a beautiful girl riding a dinosaur in a jungle with mud, sunny day shiny clear sky. $3 5 \mathrm { m m }$ photograph, film, professional, 4k, highly detailed.   
A still image of a humanoid cat posing with a hat and jacket in a bar.   
A soft beam of light shines down on an armored granite wombat warrior statue holding a broad sword. The statue stands an ornate pedestal in the cella of a temple. wide-angle lens. anime oil painting.   
children   
A photograph of the inside of a subway train. There are red pandas sitting on the seats. One of them is reading a newspaper. The window shows the jungle in the background.   
•a goat wearing headphones   
motion   
A close-up of a woman's face, lit by the soft glow of a neon sign in a dimly lit, retro diner, hinting at a narrative of longing and nostalgia.

We use the following prompts for Figure 11. From left to right, top to bottom:

A close-up of a woman's face, lit by the soft glow of a neon sign in a dimly lit, retro diner, hinting at a narrative of longing and nostalgia.   
a cat reading a newspaper   
•A television made of water that displays an image of a cityscape at night.   
a portrait of a statue of the Egyptian god Anubis wearing aviator goggles, white t-shirt and leather jacket. The city of Los Angeles is in the background.   
•a squirrell driving a toy car   
•an elephant walking on the Great Wall   
•a capybara made of voxels sitting in a field   
•A soft beam of light shines down on an armored granite wombat warrior statue holding a broad sword. The statue stands an ornate pedestal in the cella of a temple. wide-angle lens. anime oil painting.   
a goat wearing headphones   
An oil painting of two rabbits in the style of American Gothic, wearing the same clothes as in the original.   
•a girl examining an ammonite fossil   
•a chimpanzee sitting on a wooden bench children   
A still image of a humanoid cat posing with a hat and jacket in a bar. motion