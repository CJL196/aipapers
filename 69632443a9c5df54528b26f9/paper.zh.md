# AdaWorld：通过潜在动作学习可适应的世界模型

高申元 1 周思源 1 杜怡伦 2 张俊 1 甘创 3 4 adaptable-world-model.github.io

# 摘要

# 无关动作用预训练

世界模型旨在学习基于动作的未来预测，对于智能体的发展至关重要。然而，现有的大多数世界模型在很大程度上依赖大量的动作标注数据和高昂的训练成本，这使得它们在有限的交互下难以适应具有异构动作的新环境。这一限制可能会阻碍它们在更广泛领域中的应用。为了解决这个问题，我们提出了AdaWorld，一种创新的世界模型学习方法，能够实现高效适应。其关键思想是在世界模型的预训练过程中引入动作信息。这是通过自监督方式从视频中提取潜在动作来实现的，从而捕捉帧之间最关键的转变。随后，我们开发了一种条件基于这些潜在动作的自回归世界模型。这种学习模式使得高度适应性的世界模型成为可能，能够在有限的交互和微调情况下高效地转移和学习新动作。我们在多个环境上的全面实验表明，AdaWorld在模拟质量和视觉规划方面都达到了卓越的性能。

# 1. 引言

智能体应在各种任务中有效执行（Reed等，2022；Lee等，2022；Durante等，2024；Raad等，2024）。实现这一目标的一个有前景的解决方案是开发能够模拟不同环境的世界模型（Wu等，2023；2024；Yang等，2024c；Hansen等，2024）。近期的世界模型通常是从预训练的视频模型初始化的（Xiang等，2024；Agarwal等，2025；He等，2025）。尽管泛化能力有所提高，这些模型仍然需要大量的动作标签和高昂的训练成本，以获得精确的动作可控性。尽管可以为视频注释伪标签（Baker等，2022；Zhang等，2022），但为一般环境定义统一的动作格式仍然具有挑战性。因此，现有的方法在适应具有不同动作规范的新环境时，通常需要昂贵的训练成本（Gao等，2024；Chi等，2024；Che等，2025）。这些限制给基于有限交互和微调转移和学习新动作带来了重大挑战。

![](images/1.jpg)  
Figure 1. Different world model learning paradigms. Prior methods often require expensive labeling and training to achieve action controllability in new environments. To overcome this, we introduce latent actions as a unified condition for action-aware pretraining from videos, enabling highly adaptable world modeling. Our world model, dubbed AdaWorld, can readily transfer actions across contexts without training. By initializing the control interface with the corresponding latent actions, AdaWorld can also be adapted into specialized world models efficiently and achieve significantly better planning results than the action-agnostic baseline.

作为人类，我们可以通过有限的经验估计不同动作的效果（Ha & Schmidhuber, 2018）。这种能力可能源于我们从广泛观察中学习到的对动作的内部表征（Rizzolatti et al., 1996；Romo et al., 2004；Dominici et al., 2011）。这些共通知识可以在不同上下文中重用，并与特定的动作空间高效关联（Rybkin et al., 2019；Schmeckpeper et al., 2020；Sun et al., 2024）。因此，人类可以轻松地将观察到的动作转移到各种上下文中，并通过少量交互想象新环境的变化（Poggio & Bizzi, 2004）。这些见解促使我们思考：我们是否可以通过从观察中学习可转移的动作表征来实现类似人类的世界建模适应性？在本文中，我们提出了AdaWorld，这是一种创新的预训练方法，旨在构建高度适应的世界模型。与之前仅在无动作信息视频上进行预训练的方法不同，我们认为在预训练过程中融入动作信息将显著增强世界模型的适应性。如图1所示，AdaWorld的适应性主要体现在两个方面：（1）在给定一个动作示例的情况下，AdaWorld可以轻松将该动作转移到各种上下文中，而无需进一步训练。（2）它还可以通过少量的交互和微调，将原始动作输入高效地适应为专门化的世界模型，从而在各种环境中实现更有效的规划。

AdaWorld 由两个关键组成部分构成：一个从未标记视频中提取动作的潜在动作自编码器，以及一个以提取的动作作为条件的自回归世界模型。我们面临的主要挑战是，在真实环境中的视频常常涉及复杂的上下文（例如颜色和纹理），这妨碍了有效的动作识别。为了解决这个挑战，我们在潜在动作自编码器中引入了信息瓶颈设计。具体而言，潜在动作编码器从两个连续帧中提取出紧凑的编码。我们将此编码称为潜在动作，因为它用于表示这两个帧之间的过渡。基于潜在动作和前一帧，潜在动作解码器尽力预测后续帧。通过最小化使用潜在动作中编码的最小信息的预测损失，我们的自编码器被激励去分离出最关键的动作与其上下文。与之前的方法（Bruce等，2024；Chen等，2024b；Ye等，2025）专注于可玩性和行为克隆不同，我们将潜在动作压缩到一个连续的潜在空间，以最大化表达力并实现灵活组合。我们发现，潜在动作是上下文不变的，能够有效地在不同上下文之间迁移。随后，我们预训练了一个以潜在动作为条件的自回归世界模型。由于我们的潜在动作具有强大的迁移能力，生成的世界模型可以轻松适应各种环境。特别是，由于我们的世界模型已经学习模拟由任何潜在动作表示的不同动作，适应新环境就如同寻找其动作空间中对应潜在动作的映射。给定一个动作的演示，我们的模型可以通过潜在动作编码器提取潜在动作，并在不同上下文中重复使用它，从而轻松转移演示的动作。当提供动作标签时，我们可以类似地获取它们的潜在动作，并高效地初始化控制接口。这使得我们能够以最小的微调将模型适应于专业的世界模型。当仅有有限数量的交互（例如，50次交互）可用时，我们的方法显著高于从不考虑动作的视频中预训练的效率。请注意，我们的方法与现有的视频预训练方法一样具备可扩展性（Seo等，2022；Mendonca等，2023；Wu等，2023；Agarwal等，2025；He等，2025；Yu等，2025）。为了增强 AdaWorld 的泛化能力，我们通过自动生成从数千个环境中收集了大量视频语料库。生成的数据集涵盖了广泛的交互场景，跨越自我视角、第三人称视角、虚拟游戏和现实世界活动。经过大规模动作感知预训练后，我们展示了 AdaWorld 的适应性可以无缝地推广到多种领域。总之，我们做出了以下贡献：• 我们提出了 AdaWorld，这是一种在各种环境中高度适应的自回归世界模型。它可以轻松将动作转移到不同的上下文中，并允许在有限交互下进行高效适配。• 我们在来自极为多样化的环境的大规模数据集上建立了 AdaWorld。经过广泛预训练，AdaWorld 在各种领域中展示了强大的泛化能力。• 我们在多个环境中进行了全面实验，以验证 AdaWorld 的有效性。我们的模型在动作转移、世界模型适应和视觉规划方面取得了令人满意的结果。

# 2. 方法

在本节中，我们首先介绍我们的潜在动作自编码器的架构设计（第2.1节）。通过利用潜在动作作为条件，我们接着构建了一个通过动作感知预训练的自回归世界模型（第2.2节）。最后，我们演示了我们的模型如何促进高度适应性的世界建模（第2.3节）。

![](images/2.jpg)  
Figure 2. Latent action autoencoder. With an information bottleneck design, our latent action autoencoder is able to extract the most critical action information from videos and compresses it into a continuous latent action.

# 2.1. 潜在动作自编码器

我们的核心创新是，在世界模型预训练阶段中加入动作信息，而不仅仅是使用无动作视频进行预训练。通过利用预训练的动作可控性知识，得到的世界模型能够高效地使用有限的真实标注动作进行适应。然而，自然环境中的视频几乎没有动作标签。虽然通过互动收集这些标签是一种常见做法，但在众多环境中收集标签会带来大量的人力成本。此外，在不同环境中定义统一的动作格式往往不可行，而当前的世界模型需要昂贵的训练才能适应新的动作格式。为了解决这些挑战，我们提出从视频中提取潜在动作，作为世界模型预训练的统一条件，而不是依赖显式的动作注释。然而，在一般视频中，动作信息往往与上下文纠缠在一起，这给有效的动作识别带来了重大困难。受到观察启发，即在大多数互动场景中，智能体的动作通常驱动着主导变化（Rybkin et al., 2019；Menapace et al., 2021；2022；Bruce et al., 2024），我们引入了信息瓶颈，自动区分观察中的动作。

具体来说，我们基于变压器架构（Vaswani et al., 2017）实例化了一个潜在动作自编码器，其中编码器从两个连续帧 $f _ { t : t + 1 }$ 中提取潜在动作 $\tilde { a }$，解码器则根据潜在动作 $\tilde { a }$ 和前一帧 $f _ { t }$ 预测后续帧 $f _ { t + 1 }$。潜在动作编码器将两个帧 $f _ { t : t + 1 }$ 划分为 $16 \times 16$ 的图像块。这些图像块随后被投影为图像块嵌入，并沿空间维度进行展平。之后，它们与两个可学习的标记 $a _ { t : t + 1 }$ 进行拼接。此外，还对每个帧应用了正弦位置嵌入（Dosovitskiy et al., 2021），以指示空间信息。为了高效编码这两个帧中的标记，我们采用了一个具有 $L$ 个堆叠块的时空变压器（Bruce et al., 2024）。每个块包含交错的空间和时间注意力模块，后跟一个前馈网络。

![](images/3.jpg)  
Figure 3. Action-aware pretraining. We extract latent actions from unlabeled videos using the latent action encoder. By leveraging the extracted actions as a unified condition, we pretrain a world model that can perform autoregressive rollouts at inference.

空间注意力可以关注每一帧中的所有词元，而时间注意力可以访问两个帧中相同空间位置的两个词元。我们还在时间注意力中引入了旋转嵌入（Su et al., 2024），以指示因果关系。在足够的注意力相关性之后，学习型词元 $a _ { t : t + 1 }$ 可以自适应地聚合两个输入帧之间的时间动态。随后，我们丢弃所有词元，仅将 $a _ { t + 1 }$ 投影以估计潜在动作的后验 $( \mu _ { \tilde { a } } , \sigma _ { \tilde { a } } )$，遵循标准变分自编码器（VAE）（Kingma & Welling, 2014）。接着，我们从近似后验中采样 $\tilde { a }$ 并将其附加到 $f _ { t }$，然后发送到潜在动作解码器。潜在动作解码器是一个空间变换器，它在像素空间中预测后续帧 $f _ { t + 1 }$。整个潜在动作自编码器是通过VAE目标进行优化的：

$$
\begin{array} { r l } & { \mathcal { L } _ { \theta , \phi } ^ { p r e d } ( f _ { t + 1 } ) = \mathbb { E } _ { q _ { \phi } ( \tilde { a } \mid f _ { t : t + 1 } ) } \log p _ { \theta } ( f _ { t + 1 } | \tilde { a } , f _ { t } ) } \\ & { \quad \quad \quad - D _ { K L } ( q _ { \phi } ( \tilde { a } | f _ { t : t + 1 } ) | | p ( \tilde { a } ) ) . } \end{array}
$$

与原始像素空间相比，我们的潜在动作的维度极为紧凑。因此，通过潜在动作将整个后续帧传递给解码器具有挑战性。为了最小化后续帧的预测误差，潜在动作 $\tilde { a }$ 必须封装相对于前一帧的最关键变化。这导致上下文不变的动作表示，与智能体采取的真实动作紧密对应。然而，我们通过实验证明，使用上述公式训练的潜在动作自编码器在表达帧之间的多样化过渡方面存在困难。这个问题的出现是因为标准变分自编码器（VAE）对后验分布施加了强烈约束。相反，去除这一约束可能会妨碍VAE的解缠能力（Burgess等，2017）。为了解决这个问题，我们采用了 $\beta$ -VAE 公式（Higgins等，2017；Alemi等，2017），引入了一个可调的超参数 $\beta$ :

![](images/4.jpg)  
.

$$
\begin{array} { r } { \mathcal { L } _ { \theta , \phi } ^ { p r e d } ( f _ { t + 1 } ) = \mathbb { E } _ { q _ { \phi } ( \tilde { a } | f _ { t : t + 1 } ) } \log p _ { \theta } ( f _ { t + 1 } | \tilde { a } , f _ { t } ) } \\ { - \beta D _ { K L } \big ( q _ { \phi } ( \tilde { a } | f _ { t : t + 1 } ) | | p ( \tilde { a } ) \big ) . } \end{array}
$$

额外的超参数使我们能够灵活控制潜在动作所包含的信息。在实践中，我们通过经验调整这个超参数，以实现潜在动作的表达能力与上下文解耦能力之间的良好权衡。如图4所示，我们的潜在动作自编码器能够提取在不同上下文中可迁移的上下文不变动作。

# 2.2. 动作感知预训练

在训练潜在动作自编码器后，我们可以使用其编码器从视频中自动提取动作信息。这使我们能够将动作信息融入世界模型的预训练中，我们称之为动作感知预训练。为了实现这一点，我们预训练一个世界模型，该模型根据当前的潜在动作预测下一个帧。如图3所示，我们利用潜在动作编码器提取帧之间的潜在动作，并将其作为我们世界模型的输入。与之前通常预测视频片段的方法不同（Yang et al., 2024c；

![](images/5.jpg)

Xiang 等人（2024）；Agarwal 等人（2025），我们的模型支持帧级控制，为交互提供更细粒度的控制。为了确保平滑过渡，我们维护一个包含 $K$ 个历史帧的短期记忆。在推理过程中，我们的模型可以通过自回归地重复下一帧的预测过程，预测一系列未来的帧，并将预测的帧附加到记忆中。

尽管将潜在动作解码器重新用于世界模型的功能似乎很简单，但它仅通过一次前向传播进行粗略预测，导致在经过多次交互后质量显著下降。为了实现真正的预测，我们基于扩散模型建立一个独立的世界模型。具体而言，我们使用稳定视频扩散（Stable Video Diffusion, SVD）（Blattmann等，2023）初始化世界模型，这是一个使用EDM框架（Karras等，2022）训练的潜在扩散模型。与原始SVD不同，我们每次仅去噪一帧噪声图像。为了与动作信息进行深度聚合，潜在动作与时间步嵌入和来自原始SVD的CLIP图像嵌入连接在一起。记忆中的最后一帧用作SVD的条件图像。为了继承预训练的时间建模能力，我们使用SVD图像编码器对历史帧进行编码，并将其与帧的噪声潜在图结合进行预测。由于实际可用的历史帧数量可能有所不同，我们在训练期间随机抽样历史帧，最长为6帧，并将记忆长度条件发送给世界模型。遵循之前的做法（He等，2022；Valevski等，2025），在训练过程中还应用噪声增强来破坏历史帧。这种增强可以有效缓解长期漂移问题，即便在推理过程中没有应用噪声。我们通过最小化以下扩散损失，在我们的大规模数据集上预训练世界模型：

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p r e t r a i n } } = \mathbb { E } _ { \pmb { x } _ { 0 } , \epsilon , t } \Big [ \| \pmb { x } _ { 0 } - \hat { \pmb { x } } _ { 0 } ( \pmb { x } _ { t } , t , \pmb { c } ) \| ^ { 2 } \Big ] , } \end{array}
$$

其中 $\scriptstyle { \hat { \mathbf { x } } } _ { 0 }$ 是我们世界模型的预测，$^ c$ 是包含历史帧和潜在动作 $\tilde { a }$ 的条件信息。

# 2.3. 高度适应性世界模型

经过在各种环境中进行的动作感知预训练，世界模型可以通过不同的潜在动作进行控制，使其在多个应用中高度适应，包括高效的动作转移、世界模型适应，甚至动作创作。高效的动作转移。当展示一个演示视频时，我们使用潜在动作编码器提取一系列潜在动作。这使我们能够将动作与其上下文拆解，并在不同上下文中复制该动作。具体而言，在新的上下文中给定初始帧时，我们可以重复使用提取的潜在动作序列作为生成新视频的条件，采用自回归方式。如图4所示，AdaWorld 自然地将动作从源视频转移到各种上下文。高效的世界模型适应。AdaWorld 还允许在有限的动作标签和训练步骤下高效地进行世界模型适应。具体而言，在通过交互收集到几个动作视频对后，我们使用潜在动作编码器推断它们的潜在动作。由于我们的潜在动作空间是连续的，相同标签的潜在动作可以直接平均。我们经验证明，平均后的嵌入可以始终代表预期的动作。因此，对于具有 $N$ 个离散动作的新环境，我们使用 $N$ 个平均潜在动作初始化一个专门的世界模型，并对整个模型进行几步微调。对于具有连续动作空间的环境，由于选项是无限的，我们添加了一个轻量级的 MLP，将原始动作输入映射到潜在动作接口。该接口也可以通过用最小的动作-潜在动作对微调 MLP 进行高效初始化。图6显示，以上述方式初始化的模型可以通过最少的微调有效适应控制输入。动作组合与创作。同样值得注意的是，AdaWorld 相较于现有的世界模型启用了若干独特的应用。例如，它允许通过在潜在空间中插值观察到的动作来组合新动作，如图5所示。此外，通过收集和聚类潜在动作，我们可以轻松创建具有不同功能和强控制能力的灵活控制选项。这表明，AdaWorld 可能成为生成互动环境的替代方案（Bruce et al., 2024）。有关动作创作的实验细节，请参见附录C。

# 3. 实验

在这一部分，我们首先展示了AdaWorld在动作转移中的优势（第3.1节）。接着，我们研究了高效的世界模型适应如何促进更好的模拟与规划（第3.2节）。最后，我们通过消融研究分析了我们设计的有效性（第3.3节）。为了全面理解我们方法的适应性，我们将AdaWorld与三个代表性基线进行比较： • 不依赖动作的预训练。在这一设置中，我们训练了一个世界模型，其架构与AdaWorld相同，但在预训练期间始终将零作为动作条件。该基线用于展示主要依赖仅包含动作无关视频的预训练范式的效果（Mendonca et al., 2023；Wu et al., 2023；Ga0 et al., 2024；Che et al., 2025；Agarwal et al., 2025；He et al., 2025）。 • 将光流作为动作感知条件。我们使用UniMatch（Xu et al., 2023a）从视频中自动预测光流。流场被下采样为$1 6 \times 1 6$并展平作为条件编码，以替代预训练期间的潜在动作。该基线作为从未标记视频中提取动作信息的替代解决方案。 • 将离散潜在动作作为动作感知条件。我们还实现了一种基于标准VQ-VAE（Van Den Oord et al., 2017）的潜在动作自编码器变体。该变体采用具有8个离散编码的VQ代码本，而不是使用连续的潜在动作空间，遵循Genie的设定（Bruce et al., 2024）。除了上述修改外，我们为基线和我们的方法对齐了其他训练设置。所有比较方法的世界模型均训练了50K次迭代，以确保公平比较。我们的训练数据集包括四个公开可用的数据集（Goyal et al., 2017；Grauman et al., 2022；O'Neill et al., 2024；Ju et al., 2024）以及从Gym Retro（Nichol et al., 2018）和Procgen Benchmark（Cobbe et al., 2020）中的1016个环境自动收集的视频。这总共产生了约20亿帧的交互场景。有关我们的数据集和实现的更多细节，请参见附录A和B。

Table 1. Action transfer comparison. In both datasets, AdaWorld excels at transferring the demonstrated actions to different contexts.   

<table><tr><td rowspan="2">Method</td><td colspan="3">LIBERO</td><td colspan="3">SSv2</td></tr><tr><td>FVD↓</td><td>ECS↑</td><td>Human↑</td><td>FVD↓</td><td>ECS↑</td><td>Human↑</td></tr><tr><td>Act-agnostic</td><td>1545.2</td><td>0.702</td><td>0%</td><td>847.2</td><td>0.592</td><td>1%</td></tr><tr><td>Flow cond.</td><td>1409.5</td><td>0.724</td><td>2%</td><td>702.8</td><td>0.611</td><td>10.5%</td></tr><tr><td>Discrete cond.</td><td>1504.5</td><td>0.700</td><td>3.5%</td><td>726.8</td><td>0.596</td><td>21.5%</td></tr><tr><td>AdaWorld</td><td>767.0</td><td>0.804</td><td>70.5%</td><td>473.4</td><td>0.639</td><td>61.5%</td></tr></table>

# 3.1. 行动转移

AdaWorld能够将已展示的动作轻松转移到各种上下文中，而无需进一步训练。以下，我们提供定性和定量评估，以展示AdaWorld在动作转移方面的有效性。定性结果。我们通过自回归生成在图4中转移长度为20的动作序列。这表明AdaWorld能够有效地解耦已展示的动作并在不同上下文中模拟它们。与其他基线的定性比较可在附录C中找到。

定量结果。为了与其他基线进行定量比较，我们构建了一个评估集，该评估集来源于未见过的LIBERO（Liu et al., 2023）和Something-Something v2 (SSv2)（Goyal et al., 2017）数据集。具体而言，我们从LIBERO中选择并配对相同任务的视频，以及在SSv2中属于前10个最频繁标签的相同标签，这样生成了1300对用于评估（更多细节见附录D）。尽管所选的视频对包含相似的动作，但我们发现LIBERO中的视频对在物体的排列上往往有所不同，而SSv2中的视频则在上下文中有显著的差异。对于每对视频，我们将第一个视频作为演示视频，并使用第二个视频的第一帧作为初始帧。随后，我们通过从演示视频中提取动作条件，并利用不同模型自回归预测从初始帧开始的下一20帧，生成视频。评估通过使用Fréchet视频距离(FVD)（Unterthiner et al., 2018）来测量生成的视频与原始视频之间的差异。为了补充反映整体分布相似性的FVD评估，我们还采用了嵌入余弦相似度（ECS）（Sun et al., 2024），通过I3D（Carreira & Zisserman, 2017）进行帧级测量。此外，我们还对来自LIBERO和SSv2的50对视频进行了人工评估。我们邀请了四位志愿者判断动作是否成功传递。表1中的自动评估和人工评估均显示我们的连续潜在动作在动作传递性能上表现最佳，强调了它在不失去通用性的情况下表达更细腻动作的能力。

# 3.2. 世界模型适应

我们还研究了所提出的方法在仿真质量方面如何促进高效的世界模型适应性。

<table><tr><td rowspan="2">Method</td><td colspan="2">Habitat (discrete action)</td><td colspan="2">Minecraft (discrete action)</td><td colspan="2">DMLab (discrete action)</td><td colspan="2">nuScenes (continuous action)</td></tr><tr><td>PSNR↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td>Act-agnostic</td><td>20.34</td><td>0.450</td><td>19.44</td><td>0.532</td><td>20.96</td><td>0.386</td><td>20.86</td><td>0.475</td></tr><tr><td>Flow cond.</td><td>22.49</td><td>0.373</td><td>20.71</td><td>0.492</td><td>22.22</td><td>0.357</td><td>20.94</td><td>0.462</td></tr><tr><td>Discrete cond.</td><td>23.31</td><td>0.342</td><td>21.33</td><td>0.465</td><td>22.36</td><td>0.349</td><td>21.28</td><td>0.450</td></tr><tr><td>AdaWorld</td><td>23.58</td><td>0.327</td><td>21.59</td><td>0.457</td><td>22.92</td><td>0.335</td><td>21.60</td><td>0.436</td></tr></table>

![](images/6.jpg)  
new environments more rapidly than conventional pretraining methods.

视觉规划性能。

# 3.2.1. 仿真质量

设置。为了评估适应后的仿真质量，我们选择了三个具有离散动作空间的环境（Habitat (Savva et al., 2019)、Minecraft、DMLab (Beattie et al., 2016)）和一个具有连续动作空间的环境（nuScenes (Caesar et al., 2020)，该环境未包含在我们的训练数据集中）。每个环境都有一个包含300个样本的验证集，用于根据PSNR (Hore & Ziou, 2010) 和LPIPS (Zhang et al., 2018)评估适应质量。为展示在受限标签下的适应能力，我们在每个离散环境中仅收集每个动作的100个样本，并为nuScenes收集100个轨迹。然后，我们利用有限的交互数据，对所有比较的世界模型进行800步的微调，批量大小为32，学习率为$5 \times 10^{-5}$。预训练权重的学习率折扣因子为0.1。对于无动作感知基线，我们以随机参数初始化动作嵌入。对于其他三个模型，我们使用从100个样本中提取的平均动作条件来初始化动作嵌入，如第2.3节所述。需要注意的是，对于nuScenes，我们添加了一个两层的多层感知机（MLP），用于将连续位移映射到潜在的动作接口。该MLP通过有限的动作-潜在动作对进行微调，持续了3000步，这在单个GPU上耗时不到30秒。结果。如表2所示，AdaWorld在有限交互和计算后实现了最佳保真度。比较结果表明，所提出的方法使世界模型能够高效地在未见环境中模拟新的动作控制。请注意，所有动作感知变体显著优于无动作感知基线，强调了我们关键创新的重要性，即在预训练期间结合动作信息。为了进一步展示我们的样本效率和微调效率，我们在Minecraft和nuScenes上进行了更多比较实验，使用不同的样本数量和微调步骤。我们在图6中展示了PSNR的演变曲线。在所有情况下，AdaWorld在开始时表现得更好，并在经过几步微调后显著加速提升。这表明我们的方法相比传统的预训练方法，为高效的世界模型适应提供了更优的初始化。

# 3.2.2. 游戏中的视觉规划

设置。在学习了动作控制后，世界模型可以用于规划。为了展示AdaWorld在规划性能上的优越性，我们首先使用基于采样的模型预测控制（MPC）与无动作基线在视频游戏环境中进行比较，该MPC经过交叉熵方法优化（De Boer等，2005；Chua等，2018）。MPC的规划和优化过程详见附录B.4。我们基于Procgen基准（Cobbe等，2020）定义了一个目标达成任务，并从四个环境（Heist、Jumper、Maze、CaveFlyer）中选择了30个场景。这确保了指定的目标可以在可接受的步数内到达（具体细节见附录E）。对于每个场景，我们随机收集100个默认动作空间（LEFT、DOWN、UP、RIGHT）中每个动作的样本。根据收集到的样本，对预训练的世界模型进行了500步的微调，批大小为32，学习率为$5 \times 10^{-5}$。然后，我们使用微调后的世界模型在所选场景中执行MPC规划。奖励被定义为预测观察与最终状态图像之间的余弦相似度。如果智能体在20步内到达最终状态，则规划被视为成功，无需微调。Oracle：我们使用真实模拟器进行MPC，这表明该规划策略的上限。

<table><tr><td rowspan="2">Method</td><td colspan="5">Success Rate↑</td></tr><tr><td>Heist</td><td>Jumper</td><td>Maze</td><td>CaveFlyer</td><td>Average</td></tr><tr><td>Random</td><td>19.33±4.41%</td><td>22.00±2.50%</td><td>41.33±5.44%</td><td>22.00±2.50%</td><td>26.17±2.55%</td></tr><tr><td>Act-agnostic AdaWorld</td><td>20.67±3.55%</td><td>20.67±2.45%</td><td>39.33±2.87%</td><td>23.33±1.84%</td><td>26.00±0.98%</td></tr><tr><td>w/o finetune</td><td>38.67±2.01%</td><td>68.00±2.25%</td><td>41.33±2.72%</td><td>31.33±2.50%</td><td>44.83±1.37%</td></tr><tr><td>w/ finetune</td><td>66.67±4.09%</td><td>58.67±2.50%</td><td>68.00±1.69%</td><td>33.33±3.80%</td><td>56.67±2.16%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Q-learning</td><td>22.67±3.87%</td><td>47.33±6.71%</td><td>4.67±0.81%</td><td>34.00±6.17%</td><td>27.17±1.27%</td></tr><tr><td>Oracle (GT env.)</td><td>86.67±3.16%</td><td>77.33±2.67%</td><td>84.67±2.91%</td><td>74.00±3.99%</td><td>80.67±2.11%</td></tr></table>

$\mathrm { { V P ^ { 2 } } }$ 我们还报告了归一化的总体成功率，按照真实标注数据模拟器的分数进行规范化。

<table><tr><td rowspan="2">Method</td><td colspan="6"></td></tr><tr><td>Robosuite push</td><td>Open slide</td><td>Blue button</td><td>Green button</td><td>Red button</td><td>Upright block</td><td>Aggregate</td></tr><tr><td>Act-agnostic</td><td>17.50±0.50%</td><td>1.67±1.67%</td><td>5.00±1.67%</td><td>3.33±0.00%</td><td>0.00±0.00%</td><td>1.67±1.67%</td><td>5.03</td></tr><tr><td>AdaWorld</td><td>63.50±1.71%</td><td>5.83±2.85%</td><td>29.17±2.50%</td><td>10.83±2.50%</td><td>10.00±2.36%</td><td>5.00±0.96%</td><td>21.54</td></tr></table>

结果。表3展示了基于5个随机种子的成功率平均值。尽管无动作基线与随机规划表现相似，但AdaWorld在所有环境中显著提高了成功率。这表明我们的方法不仅适应更高效，还支持更有效的规划。访问我们的项目页面，查看在游戏中智能体的规划演示。此外，我们评估了在不微调模型的情况下，利用收集样本的视觉规划性能。具体而言，我们仅利用这些样本得出的平均潜在动作作为对应场景的动作嵌入。表3中的结果表明，即使不更新模型权重，我们的变体仍然优于微调的无动作预训练基线。为了进一步展示我们方法的有效性，我们还将规划结果与Q学习（Sutton & Barto, 2018）进行比较，这是一种经典的无模型强化学习方法。对于每个场景，我们使用与MPC规划收集的相同样本构建Q表。Q表的状态由量化图像表示，奖励通过计算与目标图像的余弦相似度获得。正如表3所示，AdaWorld明显优于Q学习方法，这表明我们的方法更有效地利用了有限的交互。

# 3.2.3. 机器人任务中的视觉规划

设置。为验证我们在机器人控制任务中的有效性，我们预训练了AdaWorld的低分辨率变体，并在适应后评估其在$\mathrm { { V P ^ { 2 } } }$基准上的规划性能（Tian等，2023）。规划也是基于采样的，并使用模型预测路径积分（MPPI）进行（Williams等，2016；Nagabandi等，2020）。我们专注于类似的计算高效设置，对预训练变体和一个无动作基线进行了1K步的微调。评估是在100个桌面Robosuite任务（Zhu等，2020）和7个RoboDesk任务（Kannan等，2021）上进行的。更多细节见附录B.5。结果。表4报告了在$\mathrm { { V P ^ { 2 } } }$上的成功率。我们省略了RoboDesk中的平面块和打开抽屉任务，因为在我们的受限适应设置下，这些任务没有产生有意义的得分。结果表明，AdaWorld在有限的微调步骤下能够更有效地适应，并显著提高规划性能。

# 3.3. 消融实验与分析

接口初始化。我们在第2.3节中去掉潜在动作初始化方法，用随机初始化并将得到的模型调整到未见过的Minecraft和nuScenes。图6显示，随机初始化AdaWorld的控制接口在起初导致质量下降。然而值得注意的是，尽管在微调开始时，我们的变体略逊于无动作预训练基线，但在仅仅200步之后，它迅速超过了无动作基线。这是因为AdaWorld通过动作感知预训练学习了高度适应的控制接口，使其能够通过简单调整新的动作空间的动作嵌入来高效适应未见环境。

Table 5. Impacts of training data diversity. Increasing data diversity enhances the generalization of latent actions to new domains.   

<table><tr><td rowspan="2">Training Data</td><td colspan="2">Procgen</td></tr><tr><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td>OpenX</td><td>25.51</td><td>0.318</td></tr><tr><td>Retro</td><td>26.43</td><td>0.250</td></tr><tr><td>Retro+OpenX</td><td>26.62</td><td>0.234</td></tr></table>

Table 6. Generality of AdaWorld. Applying action-aware pretraining to iVideoGPT also significantly improves its adaptability.   

<table><tr><td rowspan="2">Model</td><td colspan="2">BAIR</td></tr><tr><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td>iVideoGPT</td><td>16.59</td><td>0.220</td></tr><tr><td>iVideoGPT+AdaWorld</td><td>17.40</td><td>0.204</td></tr></table>

数据多样性。我们还研究了数据混合对潜在动作的泛化能力的影响。为此，我们实现了三种潜在动作自编码器，使用不同组合的Open X-Embodiment (OpenX) (O'Neill et al., 2024) 和 Gym Retro (Nichol et al., 2018) 数据集进行40K步的训练。然后，我们在Procgen基准测试(Cobbe et al., 2020)中评估这三种变体的潜在动作解码器预测，如表5所示。令人惊讶的是，我们发现尽管OpenX主要由现实世界的机器人视频组成，但结合OpenX有助于潜在动作自编码器在Procgen中泛化到未见过的2D虚拟游戏。这表明，进一步增加数据多样性可能对我们潜在动作的泛化产生积极影响。方法通用性。为了证明我们方法的通用性，我们使用iVideoGPT (Wu et al., 2024) 作为最先进的基线。iVideoGPT是一个具有行动控制的世界模型，采用自回归Transformer架构。它通过与动作无关的视频预测进行预训练，并在微调期间添加线性投影以学习动作控制。为了公平比较，我们实现了一个变体，在预训练期间以我们的潜在动作为条件对iVideoGPT进行配置。培训细节见附录B.6。微调后，我们在BAIR机器人推送数据集(Ebert et al., 2017)中比较动作控制仿真的质量，如表6所示。所提出的动作感知预训练显著增强了iVideoGPT的适应性，表明我们的方法普遍适用于不同的世界模型。

![](images/7.jpg)  
Figure 7. UMAP of latent actions. Reducing the value of $\beta$ increases expressiveness but sacrifices disentanglement from context.

超参数选择。在公式 (2) 中，超参数 $\beta$ 被调整以实现潜在动作的表现力和上下文解耦能力之间的良好平衡。为了提供更直观的说明，我们从 Habitat、Minecraft 和 DMLab 随机收集了每个动作的 1000 个样本，并使用 UMAP（McInnes 等人，2018）进行可视化。图 7 显示了相同的动作，即使来自不同的环境，也聚集在一起，这验证了我们潜在动作的上下文不变性属性。需要注意的是，由于在某些状态下无法执行动作输入（例如，前方有障碍物时无法前进），因此存在噪声。我们还比较了使用较低 $\beta$ 训练的模型推断的样本。尽管这导致了更多可微分的潜在动作，但也减少了不同环境之间的动作重叠，从而牺牲了解耦能力。因此，我们默认将 $\beta$ 设置为 $2 \times 1 0 ^ { - 4 }$。

# 4. 结论

在本文中，我们介绍了AdaWorld，一种新的世界模型学习方法，旨在促进跨多种环境的高效适应。它在有限交互和微调的情况下，能够高效地传递和学习新动作。大量实验和分析表明，AdaWorld具备卓越的适应性，凸显了其作为世界模型预训练新范式的潜力。局限性。尽管AdaWorld促进了可适应的世界建模，但仍然存在若干挑战。首先，它的操作频率并非实时。未来的工作可以结合蒸馏和采样技术（Feng等，2024；Yin等，2025）来加速推理速度。与之前的研究（Yang等，2024e）类似，当推演超过初始场景时，AdaWorld在创作新内容方面存在困难。通过扩大模型和训练数据规模（Bruce等，2024；Bar等，2025），该问题可能会得到解决。此外，我们的模型在实现极长时间的推演时表现不佳，未来的工作将探索潜在的解决方案（Chen等，2024a；Feng等，2024；Ruhe等，2024）。我们还在附录C中附上了一些主要的失败案例。

# 致谢

高申苑和张军得到了香港研究资助局的支持，资助项目为国家自然科学基金/香港研究资助局合作研究计划，拨款编号为 CRS_HKUST603/22。

# 影响声明

本文展示的工作旨在推动机器智能领域的发展。我们的研究可能会产生许多社会影响，但我们认为没有必要在此特别强调。

# References

Agarwal, N., Ali, A., Bala, M., Balaji, Y., Barker, E., Cai, T., Chattopadhyay, P., Chen, Y., Cui, Y., Ding, Y., et al. Cosmos World Foundation Model Platform for Physical AI. arXiv preprint arXiv:2501.03575, 2025.

Alemi, A. A., Fischer, I. Dillon, J.V., and Murphy, K.Deep Variational Information Bottleneck. In ICLR, 2017.

Alonso, E., Jelley, A., Micheli, V., Kanervisto, A., Storkey, A., Pearce, T., and Fleuret, F. Diffusion for World Modeling: Visual Details Matter in Atari. In NeurIPS, 2024.

Baker, B., Akkaya, I., Zhokov, P., Huizinga, J., Tang, J. Ecoffet, A., Houghton, B., Sampedro, R., and Clune, J. Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos. In NeurIPS, 2022.

Bar, A., Zhou, G., Tran, D., Darrell, T., and LeCun, Y. Navigation World Models. In CVPR, 2025.

Beattie, C., Leibo, J. Z., Teplyashin, D., Ward, T., Wainwright, M., Küttler, H., Lefrancq, A., Green, S., Valdés, V., Sadik, A., et al. DeepMind Lab. arXiv preprint arXiv:1612.03801, 2016.

Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M., Lorenz, D., Levi, Y., English, Z., Voleti, V. Letts, A., et al. Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets. arXiv preprint arXiv:2311.15127, 2023.

Bruce, J., Dennis, M. D., Edwards, A., Parker-Holder, J., Shi, Y., Hughes, E., Lai, M., Mavalankar, A., Steigerwald, R., Apps, C., et al. Genie: Generative Interactive Environments. In ICML, 2024.

Bu, Q., Zeng, J., Chen, L., Yang, Y., Zhou, G., Yan, J., Luo, P., Cui, H., Ma, Y., and Li, H. Closed-Loop Visuomotor Control with Generative Expectation for Robotic Manipulation. In NeurIPS, 2024.

Bu, Q., Cai, J., Chen, L., Cui, X., Ding, Y., Feng, S., Gao, S., He, X., Huang, X., Jiang, S., et al. AgiBot World

Colosseo: A Large-Scale Manipulation Platform for Scalable and Intelligent Embodied Systems. arXiv preprint arXiv:2503.06669, 2025.

Burgess, C. P., Higgins, I., Pal, A., Matthey, L., Watters, N., Desjardins, G., and Lerchner, A. Understanding Disentangling in $\beta$ VAE. In NeurIPS Workshops, 2017.

Caesar, H., Bankiti, V., Lang, A. H., Vora, S., Liong, V. E., Xu, Q., Krishnan, A., Pan, Y., Baldan, G., and Beijbom, O. nuScenes: A Multimodal Dataset for Autonomous Driving. In CVPR, 2020.

Carreira, J. and Zisserman, A. Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset. In CVPR, 2017.

Che, H., He, X., Liu, Q., Jin, C., and Chen, H. GameGenX: Interactive Open-World Game Video Generation. In ICLR, 2025.

Chen, B., Monso, D. M., Du, Y., Simchowitz, M., Tedrake, R., and Sitzmann, V. Diffusion Forcing: Next-Token Prediction Meets Full-Sequence Diffusion. In NeurIPS, 2024a.

Chen, X., Guo, J., He, T., Zhang, C., Zhang, P., Yang, D. C., Zhao, L., and Bian, J. IGOR: Image-GOal Representations are the Atomic Control Units for Foundation Models in Embodied AI. arXiv preprint arXiv:2411.00785, 2024b.

Chen, Y., Ge, Y., Li, Y., Ge, Y., Ding, M., Shan, Y., and Liu, X. Moto: Latent Motion Token as the Bridging Language for Robot Manipulation. arXiv preprint arXiv:2412.04445, 2024c.

Chi, X., Zhang, H., Fan, C.-K., Qi, X., Zhang, R., Chen, A., Chan, C.-m., Xue, W., Luo, W., Zhang, S., et al. EVA: An Embodied World Model for Future Video Anticipation. arXiv preprint arXiv:2410.15461, 2024.

Chua, K., Calandra, R., McAllister, R., and Levine, S. Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models. In NeurIPS, 2018.

Cobbe, K., Hesse, C., Hilton, J., and Schulman, J. Leveraging Procedural Generation to Benchmark Reinforcement Learning. In ICML, 2020.

Cui, Z. J., Pan, H., Iyer, A., Haldar, S., and Pinto, L. DynaMo: In-Domain Dynamics Pretraining for Visuo-Motor Control. In NeurIPS, 2024.

De Boer, P.-T., Kroese, D. P., Mannor, S., and Rubinstein, R. Y. A Tutorial on the Cross-Entropy Method. Annals of Operations Research, 2005.

Dominici, N., Ivanenko, Y. P., Cappellini, G., d' Avella, A., Mondi, V., Cicchese, M., Fabiano, A., Silei, T., Di Paolo, A., Giannini, C., et al. Locomotor Primitives in Newborn Babies and Their Development. Science, 2011.   
Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In ICLR, 2021.   
Du, Y., Yang, S., Dai, B., Dai, H., Nachum, O., Tenenbaum, J., Schuurmans, D., and Abbeel, P. Learning Universal Policies via Text-Guided Video Generation. In NeurIPS, 2023.   
Du, Y., Yang, M., Florence, P., Xia, F., Wahid, A., Ichter, B., Sermanet, P., Yu, T., Abbeel, P., Tenenbaum, J. B., et al. Video Language Planning. In ICLR, 2024.   
Durante, Z., Sarkar, B., Gong, R., Taori, R., Noda, Y., Tang, P., Adeli, E., Lakshmikanth, S. K., Schulman, K., Milstein, A., et al. An Interactive Agent Foundation Model. arXiv preprint arXiv:2402.05929, 2024.   
Ebert, F., Finn, C., Lee, A. X., and Levine, S. SelfSupervised Visual Planning with Temporal Skip Connections. In CoRL, 2017.   
Edwards, A., Sahni, H., Schroecker, Y., and Isbell, C. Imitating Latent Policies from Observation. In ICML, 2019.   
Feng, R., Zhang, H., Yang, Z., Xiao, J., Shu, Z., Liu, Z., Zheng, A., Huang, Y., Liu, Y., and Zhang, H. The Matrix: Infinite-Horizon World Generation with Real-Time Moving Control. arXiv preprint arXiv:2412.03568, 2024.   
Gao, S., Yang, J., Chen, L., Chitta, K., Qiu, Y., Geier, A., Zhang, J., and Li, H.Vista:A Generalizable Drivin World Model with High Fidelity and Versatile Controllability. In NeurIPS, 2024.   
Goyal, R., Ebrahimi Kahou, S., Michalski, V., Materzynska, J., Westphal, S., Kim, H., Haenel, V., Fruend, I., Yianilos, P., Mueller-Freitag, M., et al. The "Something Something" Video Database for Learning and Evaluating Visual Common Sense. In ICCV, 2017.   
Grauman, K., Westbury, A., Byrne, E., Chavis, Z., Furnari, A., Girdhar, R., Hamburger, J., Jiang, H., Liu, M., Liu, X., et al. Ego4D: Around the World in 3,000 Hours of Egocentric Video. In CVPR, 2022.   
Ha, D. and Schmidhuber, J. Recurrent World Models Facilitate Policy Evolution. In NeurIPS, 2018.   
Her .s J Ba J Diverse Domains through World Models. arXiv preprint arXiv:2301.04104, 2023.   
Hansen, N., Su, H., and Wang, X. TD-MPC2: Scalable, Robust World Models for Continuous Control. In ICLR, 2024.   
Hassan, M., Stapf, S., Rahimi, A., Rezende, P., Haghighi, Y., Brüggemann, D., Katircioglu, I., Zhang, L., Chen, X., Saha, S., et al. GEM: A Generalizable Ego-Vision Multimodal World Model for Fine-Grained Ego-Motion, Object Dynamics, and Scene Composition Control. In CVPR, 2025.   
He, H., Zhang, Y., Lin, L., Xu, Z., and Pan, L. Pre-Trained Video Generative Models as World Simulators. arXiv preprint arXiv:2502.07825, 2025.   
He, Y., Yang, T., Zhang, Y., Shan, Y., and Chen, Q. Latent Video Diffusion Models for High-Fidelity Long Video Generation. arXiv preprint arXiv:2211.13221, 2022.   
Higgins, I., Matthey, L., Pal, A., Burgess, C. P., Glorot, X., Botvinick, M. M., Mohamed, S., and Lerchner, A. betaVAE: Learning Basic Visual Concepts with a Constrained Variational Framework. In ICLR, 2017.   
Hong, Y., Liu, B., Wu, M., Zhai, Y., Chang, K.-W., Li, L., Lin, K., Lin, C.-C., Wang, J., Yang, Z., et al. SlowFastVGen: Slow-Fast Learning for Action-Driven Long Video Generation. In ICLR, 2025.   
Hore, A. and Ziou, D. Image Quality Metrics: PSNR vs. SSIM. In ICPR, 2010.   
Hu, A., Russell, L., Yeo, H., Murez, Z., Fedoseev, G., Kendall, A., Shotton, J., and Corrado, G. GAIA-1: A Generative World Model for Autonomous Driving. arXiv preprint arXiv:2309.17080, 2023.   
Ju, X., Gao, Y., Zhang, Z., Yuan, Z., Wang, X., Zeng, A., Xiong, Y., Xu, Q., and Shan, Y. MiraData: A LargeScale Video Dataset with Long Durations and Structured Captions. In NeurIPS Datasets and Benchmarks, 2024.   
Kaiser, L., Babizadeh, M., Milos, P., Osinski, B., Campbell, R. H., Czechowski, K., Erhan, D., Finn, C., Kozakowski, P., Levine, S., et al. Model-Based Reinforcement Learning for Atari. In ICLR, 2020.   
Kannan, H., Hafner, D., Finn, C., and Erhan, D. RoboDesk: A Multi-Task Reinforcement Learning Benchmark. https://github.com/ google-research/robodesk,2021.   
Karras, T. Aittala, M., Aila, T., and Laine, S. Eluciag the Design Space of Diffusion-Based Generative Models. In NeurIPS, 2022.   
Kazemi, N., Savov, N., Paudel, D., and Van Gool, L. Learning Generative Interactive Environments by Trained Agent Exploration. In NeurIPS Workshops, 2024.

Kim, M. J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E., Lam, G., Sanketi, P., et al. OpenVLA: An Open-Source VisionLanguage-Action Model. In CoRL, 2024.

Kim, S. W., Zhou, Y., Philion, J., Torralba, A., and Fidler, S. Learning to Simulate Dynamic Environments with GameGAN. In CVPR, 2020.

Kim, S. W., Philion, J., Torralba, A., and Fidler, S. DriveGAN: Towards a Controllable High-Quality Neural Simulation. In CVPR, 2021.

Kingma, D. P. K. and Welling, M. Auto-Encoding Variational Bayes. In ICLR, 2014.

Ko, P.-C., Mao, J., Du, Y., Sun, S.-H., and Tenenbaum, J. B. Learning to Act from Actionless Videos through Dense Correspondences. In ICLR, 2024.

Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., et al. HunyuanVideo: A Systematic Framework For Large Video Generative Models. arXiv preprint arXiv:2412.03603, 2024.

Lee, K.-H., Nachum, O., Yang, M. S., Lee, L., Freeman, D., Guadarrama, S., Fischer, I., Xu, W., Jang, E., Michalewski, H., et al. Multi-Game Decision Transformers. In NeurIPS, 2022.

Ling, P., Bu, J., Zhang, P., Dong, X., Zang, Y., Wu, T., Chen, H., Wang, J., and Jin, Y. MotionClone: Training-Free Motion Cloning for Controllable Video Generation. In ICLR, 2025.

Liu, B., Zhu, Y., Gao, C., Feng, Y., Liu, Q., Zhu, Y., and Stone, P. LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. In NeurIPS Datasets and Benchmarks, 2023.

Loshchilov, I. and Hutter, F. Decoupled Weight Decay Regularization. In ICLR, 2019.

Lu, T., Shu, T., Xiao, J., Ye, L., Wang, J., Peng, C. Wei, C., Khashabi, D., Chellappa, R., Yuille, A., et al. GenEx: Generating an Explorable World. arXiv preprint arXiv:2412.09624, 2024.

Lu, T., Shu, T., Yuille, A., Khashabi, D., and Chen, J. Generative World Explorer. In ICLR, 2025.

Mazzaglia, P., Verbelen, T., Dhoedt, B., Courville, A., and Rajeswar, S. GenRL: Multimodal-Foundation World Models for Generalization in Embodied Agents. In NeurIPS, 2024.

McInnes, L., Healy, J., and Melville, J. UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv preprint arXiv:1802.03426, 2018.

Menapace, W., Lathuiliere, S., Tulyakov, S., Siarohin, A., and Ricci, E. Playable Video Generation. In CVPR, 2021.

Menapace, W., Lathuilière, S., Siarohin, A., Theobalt, C., Tulyakov, S., Golyanik, V., and Ricci, E. Playable Environments: Video Manipulation in Space and Time. In CVPR, 2022.

Mendonca, R., Bahl, S., and Pathak, D. Structured World Models from Human Videos. In RSS, 2023.

Micheli, V., Alonso, E., and Fleuret, F. Transformers are Sample-Efficient World Models. In ICLR, 2023.

Nagabandi, A., Konolige, K., Levine, S., and Kumar, V. Deep Dynamics Models for Learning Dexterous Manipulation. In CoRL, 2020.

Nichol, A., Pfau, V., Hesse, C., Klimov, O., and Schulman, J. Gotta Learn Fast: A New Benchmark for Generalization in RL. arXiv preprint arXiv:1804.03720, 2018.

Nikulin, A., Zisman, I., Tarasov, D., Lyubaykin, N., Polubarov, A., Kiselev, I., and Kurenkov, V. Latent Action Learning Requires Supervision in the Presence of Distractors. arXiv preprint arXiv:2502.00379, 2025.

O'Neill, A., Rehman, A., Gupta, A., Maddukuri, A., Gupta, A., Padalkar, A., Lee, A., Pooley, A., Gupta, A., Mandlekar, A., et al. Open X-Embodiment: Robotic Learning Datasets and RT-X Models. In ICRA, 2024.

Pearce, T., Rashid, T., Bignell, D., Georgescu, R., Devlin, S., and Hofmann, K. Scaling Laws for Pre-Training Agents and World Models. arXiv preprint arXiv:2411.04434, 2024.

Peebles, W. and Xie, S. Scalable Diffusion Models with Transformers. In ICCV, 2023.

Poggio, T. and Bizzi, E. Generalization in Vision and Motor Control. Nature, 2004.

Qi, H., Yin, H., Du, Y., and Yang, H. Strengthening Generative Robot Policies through Predictive World Modeling. arXiv preprint arXiv:2502.00622, 2025.

Raad, M. A., Ahuja, A., Barros, C., Besse, F., Bolt, A., Bolton, A., Brownfield, B., Buttimore, G., Cant, M., Chakera, S., et al. Scaling Instructable Agents Across Many Simulated Worlds. arXiv preprint arXiv:2404.10179, 2024.

Reed, S., Zolna, K., Parisotto, E., Colmenarejo, S. G., Novikov, A., Barth-Maron, G., Gimenez, M., Sulsky, Y., Kay, J., Springenberg, J. T., et al. A Generalist Agent. In TMLR, 2022.

Ren, Z., Wei, Y., Guo, X., Zhao, Y., Kang, B., Feng, J. and Jin, X. VideoWorld: Exploring Knowledge Learning from Unlabeled Videos. In CVPR, 2025.

Rigter, M., Gupta, T., Hilmkil, A., and Ma, C. AVID: Adapting Video Diffusion Models to World Models. arXiv preprint arXiv:2410.12822, 2024.

Rizzolatti, G., Fadiga, L., Gallese, V., and Fogassi, L. Premotor Cortex and the Recognition of Motor Actions. Cognitive Brain Research, 1996.

Romo, R., Hernández, A., and Zainos, A. Neuronal Correlates of a Perceptual Decision in Ventral Premotor Cortex. Neuron, 2004.

Ruhe, D., Heek, J., Salimans, T., and Hoogeboom, E. Rolling Diffusion Models. In ICML, 2024.

Rybkin, O., Pertsch, K., Derpanis, K. G., Daniilidis, K., and Jaegle, A. Learning What You Can Do before Doing Anything. In ICLR, 2019.

Savva, M., Kadian, A., Maksymets, O., Zhao, Y., Wijmans, E., Jain, B., Straub, J., Liu, J., Koltun, V., Malik, J., et al. Habitat: A Platform for Embodied AI Research. In ICCV, 2019.

Schmeckpeper, K., Xie, A., Rybkin, O., Tian, S., Daniilidis, K., Levine, S., and Finn, C. Learning Predictive Models From Observation and Interaction. In ECCV, 2020.

Schmidt, D. and Jiang, M. Learning to Act without Actions. In ICLR, 2024.

Seo, Y., Lee, K., James, S. L., and Abbeel, P. Reinforcement Learning with Action-Free Pre-Training from Videos. In ICML, 2022.

Seo, Y., Hafner, D., Liu, H., Liu, F., James, S., Lee, K., and Abbeel, P. Masked World Models for Visual Control. In CoRL, 2023.

Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. RoFormer: Enhanced Transformer with Rotary Position Embedding. Neurocomputing, 2024.

Sun, Y., Zhou, H., Yuan, L., Sun, J. J., Li, Y., Jia, X., Adam, H., Hariharan, B., Zhao, L., and Liu, T. Video Creation by Demonstration. arXiv preprint arXiv:2412.09551, 2024.

Sutton, R. S. Dyna, an Integrated Architecture for Learning, Planning, and Reacting. ACM Sigart Bulletin, 1991.

Sutton, R. S. and Barto, A. G. Reinforcement Learning: An Introduction. MIT Press, 2018.

Tian, S., Finn, C., and Wu, J. A Control-Centric Benchmark for Video Prediction. In ICLR, 2023.

Unterthiner, T., Van Steenkiste, S., Kurach, K., Marinier, R., Michalski, M., and Gelly, S. Towards Accurate Generative Models of Video: A New Metric & Challenges. arXiv preprint arXiv:1812.01717, 2018.

Valevski, D., Leviathan, Y., Arar, M., and Fruchter, S. Diffusion Models are Real-Time Game Engines. In ICLR, 2025.

Van Den Oord, A., Vinyals, O., et al. Neural Discrete Representation Learning. In NeurIPS, 2017.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is All You Need. In NeurIPS, 2017.

Villar-Corrales, A. and Behnke, S. PlaySlot: Learning Inverse Latent Dynamics for Controllable ObjectCentric Video Prediction and Planning. arXiv preprint arXiv:2502.07600, 2025.

Wang, L., Zhao, K., Liu, C., and Chen, X. Learning Real-World Action-Video Dynamics with Heterogeneous Masked Autoregression. arXiv preprint arXiv:2502.04296, 2025.

Wang, X., Zhu, Z., Huang, G., Chen, X., Zhu, J., and Lu, J. DriveDreamer: Towards Real-World-Driven World Models for Autonomous Driving. In ECCV, 2024.

Watter, M., Springenberg, J., Boedecker, J., and Riedmiller, M. Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images. In NeurIPS, 2015.

Willi, T., Jackson, M. T., and Foerster, J. N. Jafar: An Open-Source Genie Reimplemention in JAX. In ICML Workshops, 2024.

Williams, G., Drews, P., Goldfain, B., Rehg, J. M., and Theodorou, E. A. Aggressive Driving with Model Predictive Path Integral Control. In ICRA, 2016.

Wu, J., Ma, H., Deng, C., and Long, M. Pre-Training Contextualized World Models with In-the-Wild Videos for Reinforcement Learning. In NeurIPS, 2023.

Wu, J., Yin, S., Feng, N., He, X., Li, D., Hao, J., and Long, M. iVideoGPT: Interactive VideoGPTs are Scalable World Models. In NeurIPS, 2024.

Wu, P., Escontrela, A., Hafner, D., Abbeel, P., and Goldberg, K. DayDreamer: World Models for Physical Robot Learning. In CoRL, 2022.

Wu, Y., Tian, R., Swamy, G., and Bajcsy, A. From Foresight to Forethought: VLM-In-the-Loop Policy Steering via Latent Alignment. arXiv preprint arXiv:2502.01828, 2025.

X J Y. o Q. Y. aY, Z., Tao, T., Hao, S., Shi, Y., et al. Pandora: Towards General World Model with Natural Language Actions and Video States. arXiv preprint arXiv:2406.09455, 2024.   
Xu, H., Zhang, J., Cai, J., Rezatofighi, H., Yu, F., Tao, D., and Geiger, A. Unifying Flow, Stereo and Depth Estimation. IEEE TPAMI, 2023a.   
Xu, M., Xu, Z., Chi, C., Veloso, M., and Song, S. XSkill: Cross Embodiment Skill Discovery. In CoRL, 2023b.   
Yang, J., Gao, S., Qiu, Y., Chen, L., Li, T., Dai, B., Chitta, K., Wu, P., Zeng, J., Luo, P., et al. Generalized Predictive Model for Autonomous Driving. In CVPR, 2024a.   
Yang, M., Du, Y., Dai, B., Schuurmans, D., Tenenbaum, J. B., and Abbeel, P. Probabilistic Adaptation of Text-toVideo Models. In ICLR, 2024b.   
Yang, M., Du, Y., Ghasemipour, K., Tompson, J., Schuurmans, D., and Abbeel, P. Learning Interactive Real-World Simulators. In ICLR, 2024c.   
Y ., Li J .he S.YuY.Fu Q., W., and Ye, D. Playable Game Generation. arXiv preprint arXiv:2412.00887, 2024d.   
Yang, S., Walker, J., Parker-Holder, J., Du, Y., Bruce, J., Barreto, A., Abbeel, P., and Schuurmans, D. Video as the New Language for Real-World Decision Making. In ICML, 2024e.   
Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., et al. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. In ICLR, 2025.   
Yatim, D., Fridman, R., Bar-Tal, O., Kasten, Y., and Dekel, T. Space-Time Diffusion Features for Zero-Shot TextDriven Motion Transfer. In CVPR, 2024.   
Ye, S., Jang, J., Jeon, B., Joo, S., Yang, J., Peng, B., Mandlekar, A., Tan, R., Chao, Y.-W., Lin, B. Y., et al. Latent Action Pretraining from Videos. In ICLR, 2025.   
Ye, W., Zhang, Y., Abbeel, P., and Gao, Y. Become a   
Pcnt ayer ith Limid Da through Wathing Pure Videos. In ICLR, 2023.   
Yin, T., Zhang, Q., Zhang, R., Freeman, W. T., Durand, F., Shechtman, E., and Huang, X. From Slow Bidirectional to Fast Causal Video Generators. In CVPR, 2025.   
Yu, J., Qin, Y., Wang, X., Wan, P., Zhang, D., and Liu, X. GameFactory: Creating New Games with Generative Interactive Videos. arXiv preprint arXiv:2501.08325, 2025.   
Zhang, J., Zhu, R., and Ohn-Bar, E. SelfD: Self-Learning Large-Scale Driving Policies From the Web. In CVPR, 2022.   
Zhang, L., Kan, M., Shan, S., and Chen, X. PreLAR: World Model Pre-Training with Learnable Action Representation. In ECCV, 2024.   
Zhang, R., Isola, P., Efros, A. A., Shechtman, E., and Wang, O. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In CVPR, 2018.   
Zhen, H., Qiu, X., Chen, P., Yang, J., Yan, X., Du, Y., Hong, Y., and Gan, C. 3D-VLA: A 3D Vision-Language-Action Generative World Model. In ICML, 2024.   
Zhou, G., Pan, H., LeCun, Y., and Pinto, L. DINO-WM: World Models on Pre-Trained Visual Features enable Zero-Shot Planning. arXiv preprint arXiv:2411.04983, 2024a.   
Zhou, S., Du, Y., Chen, J., Li, Y., Yeung, D.-Y., and Gan, C. RoboDreamer: Learning Compositional World Models for Robot Imagination. In ICML, 2024b.   
Zhu, F., Wu, H., Guo, S., Liu, Y., Cheang, C., and Kong, T. IRASim: Learning Interactive Real-Robot Action Simulators. arXiv preprint arXiv:2406.14540, 2024.   
Zhu, Y., Wong, J., Mandlekar, A., Martín-Martín, R., Joshi, A., Nasiriany, S., and Zhu, Y. robosuite: A Modular Simulation Framework and Benchmark for Robot Learning. arXiv preprint arXiv:2009.12293, 2020.

# A. Datasets

# A.1. Data Collection and Generation

Oa  ubl ; X- e  e po low-frequency and low-resolution frames.

eve  o ROMs weun int GReto (Nicho  l. 18), sultn total 0teiv vt ee f     olar v   sar e tme p t collect  ransitions or ah GymRet evioment and 10Mor eac Procen evirmet.Unke s v s v visualize some representative environments in our final dataset.

# A.2. Data Mixture

usheet

uniform action sampling strategy (1000 steps)   

<table><tr><td>Category</td><td>Data Source</td><td>Automated</td><td># Frames</td><td>Ratios</td></tr><tr><td rowspan="2">2D Video Game</td><td>Gym Retro (Nichol et al., 2018)</td><td>✓</td><td>1000M</td><td>49%</td></tr><tr><td>Procgen Benchmark (Cobbe et al., 2020)</td><td>✓</td><td>144M</td><td>2%</td></tr><tr><td>Robot Data</td><td>Open X-Embodiment (O&#x27;Neillet al., 2024)</td><td>×</td><td>170M</td><td>30%</td></tr><tr><td rowspan="2">Human Activity</td><td>Ego4D (Grauman et al., 2022)</td><td>X</td><td>330M</td><td>1%</td></tr><tr><td>Something-Something V2 (Goyal et al., 2017)</td><td>X</td><td>7M</td><td>3%</td></tr><tr><td>3D Rendering</td><td>MiraData (Ju et al., 2024)</td><td>X</td><td>200M</td><td>14%</td></tr><tr><td>City Walking</td><td>MiraData (Ju et al., 2024)</td><td>×</td><td>120M</td><td>1%</td></tr></table>

![](images/8.jpg)  
agents to explore longer horizons.

Flt o   

<table><tr><td>ActRaiser2-Stes 1942-Nes AddamsFamilyPugsleyScaver ngerHun-Nes AdventuesOfamanAndRobis-Genesis</td><td>1943-Nes ActionPachio-Snes AdventuresOfBayouBilly-Nes</td><td>NinjasKickBack-Genesis AdancedBusterwkGleylancer-Ge AddamsFamily-GameBoy AdventuresOfDinoRiki-Nes</td><td>SEyes-Nes AddamsFamily-Genesis Adventure-Auari2600 AdventuresOD:Frankes-Snes</td><td>AaahhRealMonsters-Genesis Adventarelsland-GameBoy AddamsFamily-Nes AdventuresOfKidKleets-Sncs</td><td>AbadoxTheDeadlyInnerWar-Nes AddamsFamily-Sms arelsland3-Nes AdventuresOMightyMas-Genesis</td><td>AcceleBrid-Snes Adventu AddamsFamily-Snes AdventuresOMightyMas-Snes</td></tr><tr><td>AeroTheAcoBat-Snes AirDiver-Genesis Alies-Atari2600 AlphaMossion-Nes ArcherMacleansSuperDropuone-Snes</td><td>OfRockyAndBullwinkleAndFriends-Nes AcoTheAcroBa2-Genesis AirRaid-Atari2600 Alien3-Nes AleredBeast-Genesis AndyLightfot-Stes</td><td>OfRockyAndBullwinkleAndFriends-Snes AcroTheAcroBat2-Snes Airstriker-Genesis Alien3-Sms Amagos-Nes Argus-Nes</td><td>dventuresOfStarSaver-GameBoy AferBurerll-Geesis Airwoll-Nes ArielThelitleMermaid-Genesis cingPenguin-GameBoy</td><td>AlfredChickes-GameBoy AlienSyndrome-Sms AmericanGladiators-Genesis</td><td>Angh S AlienVsPredator-Snes AlfredChickes-Nes</td><td>Airy AlfredChicken-Snes Alleyway-GameBoy</td></tr><tr><td>AsterisAndObelix-GameBoy ArrowFlash-Genesis</td><td>AntOfFighting-Genesis AsterisAndTheGreaRescue-Genesis</td><td>AnOfFighting-Snes AsterisAndTheGratRescue-Sms</td><td>Assaul-Atari2500</td><td>Arkanoid-Nes Asteris-Atari2600</td><td>Amidar-Atari2600 ArkistasRing-Nes</td><td>AchRivalsTheArcadeGame-Genesi Armadillo-Nes</td></tr><tr><td></td><td></td><td></td><td>AsterixAndThePowerOfTheGods-Genesis</td><td>AsterixAndTheSecretMission-Sms</td><td>Asterix-Sms</td><td></td></tr><tr><td>AstroRabby-GameBoy</td><td>AstroRobcSass-Nes</td><td>AstroWarrior-Sms</td><td>Astyatax-Nes</td><td></td><td></td><td>Asteroids-GameBoy Asterix-Snes</td></tr><tr><td>AtomicRoboKid-Genesis</td><td>AlomicRunner-Geneis</td><td>AttackAnimalGakuen-Nes</td><td>AltackOfTheKilerTomatoes-GameBoy</td><td>Athesa-Nes</td><td></td><td></td></tr><tr><td></td><td>BOB-Genesis</td><td>BOB-Saes</td><td>BWings-Nes</td><td>AltackOfTheKillerTomatees-Nes BackToTheFuturePartl-Genesis</td><td>BadDudes-Nes</td><td>Axelay-Snes</td></tr><tr><td>BakaretuSenshiWarrier-GameBoy</td><td>BarkleyShurUpAndlam-Genesis BalloonFight-Nes</td><td>BallonKid-GameBoy</td><td>Baltron-Nes</td><td>BananaPrince-Nes</td><td>BanishingRace-GameBoy</td><td>BadStreetBrawle-Nes</td></tr><tr><td>Barbie-Nes</td><td>BumanRetuns-Snes</td><td>BattleArenaToshinden-GameBoy</td><td></td><td></td><td>Batman-Genesis</td><td>BatmanReturns-Genesis BankHeist-Atari2600</td></tr><tr><td>BatmasReturns-Nes</td><td>BattletcadsInBattlen</td><td></td><td>B BatteBull-GameBoy</td><td>Battletcads-Genesis BattleCity-Nes</td><td>BattleMasterKyukyokuNeSenshiTachi-Snes</td><td>BatteSquadron-Genesis</td></tr><tr><td>BanleTechAGameOAmmoredCombat-Genesis BattletoadsDoubleDeagon-Snes</td><td>BillAndTedsExcellentGameBoyAdventure-GaneBoy</td><td>BiminiRun-Genesis</td><td>BinayLand-Nes</td><td>BeatyAndTheBeastBellesQuest-Gc</td><td></td><td></td></tr><tr><td>Berzerk-Atari2600 BicSenshiDanlncreaserTonoTatakai-Nes</td><td>BlockKazshiGB-GameBoy BirdWeek-Nes</td><td>BishoujeSenshiSailorMoon-Genesis Blockout-Genesis</td><td>BishoujoSenshiSailorMoonR-Snes</td><td>BioHazardBatte-Genesis BlaZeonTheBioCyborgChallenge-Snes</td><td>BioMeal-Ses</td><td>BioMiracleBokutteUpa-Nes BladesOfVengeance-Geneis</td></tr><tr><td>BlockKuzshi-Snes BoobyBoys-CGameBoy</td><td></td><td>BoogemanAPckAndFlickAdventure-Geneis</td><td>BodyCount-Genesis BoogermanAPckAndFlickAdventure-Snes</td><td>Bomblack CameBoy BoogicWoogicBowling-Genesis</td><td>Biosk Ss</td><td>BonkersWaxUp-Sms</td></tr><tr><td>Bowling-Atari2600</td><td>eeCcoy</td><td></td><td>BabbaNStis-Genesis</td><td></td><td>BoulderdDasb-GameBoy BramStokersDracula-Snes</td><td>BoulderDash-Nes</td></tr><tr><td>BreakThrs-Nes BubbleBobblePart2-Nes</td><td>BullsVersusBlazrsAndTheNBAPtyoffs-Genesi BubbleGhost-GameBoy</td><td>BallsVLakersAndTheNBAPlayoffs-Gene Bubsyll-Genesis</td><td>Bubsyll-Ssnes</td><td>BubsylnClassEncoutesOfTheFurdKind-Genesis</td><td>BubbleBobble-Nes BubsylnClawsEncountersOfTheFuredKind-Snes</td><td>Brilh s</td></tr><tr><td>CalRipkenlBaseball-Genesis</td><td>Caliber50-Genesis</td><td>CalifomiaCames-Genesis</td><td>BuraiFighter-Nes Cameliry-Snes</td><td>BurningForce-Genesis CannonFodder-Genesis</td><td>CannonFodder-Snes</td><td>BuckyOHare-Nes Cadash-Genesis</td></tr><tr><td>CaptainAmericaAndTheAvengers-Nes</td><td>CaptainAmericaAndTheAvengers-Snes CastleOfllusion-Genesis</td><td>CaptainCommando-Snes Castleratia-Nes</td><td>CaptainPanetAndThePancteers-Genesis</td><td>CaptainPlanetAndThePlaneteers-Nes CastlevaniaDraculaX-Snes</td><td>CaptainSilver-Nes</td><td>CaptainAmericaAndTheAven</td></tr><tr><td>CaNindenTeyandee-Nes</td><td>CheeseCaAstropheStarringSpeedyGonzales-Genesis Centipede-Atari2600</td><td>CheeseCaAstropheStarringSpeedlyGonzales-Sms ChacknPop-Nes</td><td>ChesterCheetahTooCoolToFoo-Genesis Challenger-Nes</td><td>ChampionsWcrldClasSoccer-Genesis</td><td>ChampionshipProAm-Genesis</td><td>ChaosEingine-Genesis</td></tr><tr><td>ChaseHQil-Genesis ChiChisPoChallengeGoll-Genesis</td><td>ChiliChilkiBoys-Genesis ChubbyCherub-Nes</td><td>Choplifer-Nes ChuckRock-Genesis</td><td></td><td>ChesterCheetahTooCoefToFool-Snes</td><td>ChesterCheetahWiWiaQuest-Genesis</td><td>ChesterCheetahWildWildQuest-Snes</td></tr><tr><td>ChoujikuuYousaiMacrosScrambleValkyrie-Snes CincasCaper-Nes</td><td>CircusCharlie-Nes CloudMaster-Sms</td><td>CityConnection-Nes</td><td>ClayFighter-Genesis ChuckRock-Sms</td><td>Claymates-Snes</td><td>ChuckRockllSonOFChuck-Genesis</td><td>ChuckRockISonOfChuck-Sms</td></tr><tr><td>Cliffhunger-Snes ColumnsIII-Genesis</td><td>CombatCars-Genesis</td><td>ComicalMachineGunloe-Sms CluCluLand-Nes</td><td>CobraTriangle-Nes ComixZone-Genesis</td><td>CodeNameViper-Nes Conan-Nes</td><td>Cliffhanger-Genesis CollegeSlan-Genesis</td><td>Cliffhanger-Nes Columns-Genesis</td></tr><tr><td>ContraForce-Nes CrazyClimber-Atari2600</td><td>CoolSpot-Genesis CrossFire-Nes</td><td>CoolSpot-Sms CrueBallHeavyMetalPinball-Ger</td><td>CoolSpot-Snes</td><td>CosmicEpsion-Nes</td><td>CongosCaper-Saes CosmoGiung TheVideo-Snes</td><td>ConquestOfTheCryotalPalace-Nes</td></tr><tr><td>Cyherbull-Genesis</td><td>Cyhenator-Snes DangerouSeed-Genesis</td><td>Cyborglustice-Genesis</td><td>DJBoy-Genesis Curse-Genesis</td><td>CutieSuzukiNoRingsideAngel-Genesis DaffyDacklnHollywood-Sms</td><td>DaffyDuckTheMarvinMission-Snes CutthroatIsland-Genesis</td><td>CrackDown-Genesis CyberShinobi-Sms</td></tr><tr><td>Dhi yhedaorghis me</td><td>DeepDuckTroubleSurringDonaldDuck-Sms DashGalaxyInTheAlienAsylum-Nes</td><td>DariusForce-Snes</td><td>DuvidRobins DariusIl-Genesis nsSupremeCourt-Genesis</td><td>DariusTwin-Snes DuzeBeforeChristmas-Genesis</td><td></td><td>DaisenpuTwinHawk-Genesis Darkmas-Nes</td></tr><tr><td>DeathDuel-Genesis DevilCrashMD-Genesis</td><td>DevilshTheNexPossession-Genesis</td><td>Defender-Adari2600 DickTracy-Genesis</td><td>Defenderll-Nes Dick Tracy-Sms</td><td>DemonAttack-Auari2600</td><td>DuzeBeforeChristmas-Snes DesnisTheMenace-Snes</td><td>DesertStrikeRetusToTheGll-Genesis DeadlyMoves-Gene</td></tr><tr><td>DonkeyKongCountry-Snes orce-Snes</td><td>DonkeyKongCountry2-Snes DinoCity-Snes</td><td>DonkeyKongCountry3DisieKongsDoubleTrouble-Snes DinoLand-Genesis</td><td>Dity Harry-Nes DonkeyKongle-Nes</td><td>DickVialesAwresomeBabyCollegeHoop-Genesis DonDokoDon-Nes</td><td>DigDugITroublelsParadise-Nes</td><td>DiggerTheLegendOfTheLoatCity-Nes</td></tr><tr><td>Daues</td><td>DoubleDragonVTheShadowFall-Genesis</td><td>DoableDeibbleThePayoffEditicn-Genesis</td><td>DoubleDunk-Atari2600</td><td>DoubleDeragon-Genesis DrRobotniksMeanBeanMachine-Genesis</td><td>Da </td><td></td></tr><tr><td>DynamiteHeaddy-Genesis ElevatorAction-Nes</td><td>ESPNBaschallTonight-Genesis EliminateDown-Genesis</td><td>EamnestEvans-Genesis DragonsLair-Snes</td><td>EarthDefenseForce-Snes</td><td>DreamTeamUSA-Geneis EViento-Genesis</td><td>ElementalMaster-Genesis</td><td>DragonSpiritTheNewLegend-Nes DynamiteDuke-Sms</td></tr><tr><td>FZSenkiAxisFinalZone-Genesi</td><td>FaeryTaleAdventure-Genesi</td><td>Enduro-Atari2600 FamilyDog-Snes</td><td>anClubSoccer-Genesis</td><td>ExMutants-Genesis</td><td>Excrion-Nes</td><td>ElevatorAction-Atari2600 F1-Genesis</td></tr><tr><td>FantasyZonelTheTeansOOpaOpu-Sms</td><td>FantasyZoneTheMaze-Sms</td><td>FatalFury-Genesis</td><td>Feat oein</td><td>FEalla is</td><td>FantayZone-Sma FatalRewind-Genesis</td><td>FelixTheCat-Nes</td></tr><tr><td></td><td>FlyingDragenTheSecretSceroll-Nes</td><td>Framay ae Ss</td><td></td><td>FinalFight2-Snes</td><td>FinalFight3-Snes</td><td></td></tr><tr><td>FessPeterPanAndThePratesTheRevengeOfCaptainHook-Nes FintosesTheRescueOfDinoAndHoppy-Nes</td><td>FrankThomasBigHurtBasehall-Genesis GlloeTheAdantisFactor-Nes</td><td>Freeway-Atari2600 FlyingHero-Nes</td><td>FlyingHeroBugyurNoDaiboaken-Snes Frogger-Genesis</td><td>FrontLine-Nes</td><td>Forgotten Worlds-Genesis Frostbite-Atari2600</td><td>FormationZ-Nes</td></tr><tr><td>GilloeARealAmericanHero-Nes GalaxyFocell-Genesis</td><td></td><td>GadgetTwins-Genesis Gauntlet-Sms</td><td>Gaiares-Genesis Geimos-Nes</td><td>GainGround-Genesi GeneralChaos-Genesis</td><td></td><td>FushigiNoOshiroPtPot-Sms GalaxyForce-Sms</td></tr><tr><td>GoldenAse-Genesis</td><td>Gimnick-Nes GoldenAxellI-Genesis</td><td>GlobalDefemie-Sms Gopher-Atari2600</td><td>GlobalGiladitors-Genesis</td><td>Gods-Genesis</td><td>Gods-Snes</td><td>GhoulSchool-Nes</td></tr><tr><td>Cremin2TheNewBachb-Nes Granada-Genesis</td><td>Geavitar-Alari2600 GrindStommer-Genesis</td><td></td><td>GreaCircusMysteryStarringMickeyAndMinnie-Snes Gradius-Nes</td><td>Gradiusl-Nes</td><td>GreatWaldoSeanch-Genesis GradiusII-Snes</td><td>GradiusThelntestellarAsaul-GaneBoy</td></tr><tr><td>Gynoug-Genesis</td><td>Gyrodine-Nes</td><td>Gynuss-Nes</td><td>GuardianLegend-Nes HammerinHary-Nes</td><td>Cirala Ne HangOn-Sms</td><td>GunNac-Nes</td><td>GreendogTheBeachedSurferDude-Genesis Gunship-Genesis</td></tr><tr><td>Havec-Cesesis</td><td>Heavy Barrel-Nes HomeAlone-Genesis</td><td>HearyNova-Genesis HomeAlone2LostlnNewYork-Genesis</td><td></td><td>Hellire-Gienesis</td><td>HardDrivin-Genesis HelloKittyWorld-Nes</td><td>HarleysHumongousAdventure-Snes Hero-Atari2600</td></tr><tr><td>HundForRedOctober-Snes</td><td>Hurricanes-Genesis</td><td>Hurricanes-Snes</td><td>IMGlsternationalTourTennis-Genesis</td><td>IceClimber-Nes Hook-Genesis</td><td>Hok-Sas IceHeckey-Atari2500</td><td></td></tr><tr><td>Hn</td><td>IsolatedWarrior-Nes</td><td>IachyAndScratchyGame-Snes</td><td>nct</td><td>InsctorX-Nes</td><td>ImpectorGadges-Snes</td><td>IncredibleHalk-Sms lkari-Nes</td></tr><tr><td>IshidoTheWayOfStones-Geneis</td><td>JamesBond007TheDuel-Genesis</td><td>JamesBond007TheDuel-Sms</td><td>lzzysQuesForTheOlympicRings-Genesis</td><td>1zzysQuestFerTheOlympicRings-Snes JamesPond2CodenameRoboCod-Smm</td><td>Jackal-Nes</td><td>JackieChansActionKungFe-Nes</td></tr><tr><td>JoeAndMac-Genesis watetAgeni-Genesis</td><td>JoeAndMac-Nes</td><td>Jaws-Nes JoeAndMac-Stes</td><td>JoeAndMacHLostlsTheTropics-Snes</td><td>JetsonsCogwellsCaper-Nes JouneyEscape-Aari2600</td><td>JamesPond3-Genesis</td><td>JamesPondICodenameRobocod-Genesis</td></tr><tr><td>JuJuDessetsTokiGoingApeSpi-Geneis</td><td>Jualp Coin</td><td>JudgeDeead-Snes KabakiQuantumFighter-Nes</td><td>JungleBook-Genesis</td><td></td><td>JourneyToSiius-Nes JungleBook-Snes</td><td>Joust-Nes</td></tr><tr><td>Kangaroo-Alari2600</td><td>KanshakudamaNageKantarouNoToukaidouGojusanTsugi-Nes</td><td>KeroppiToKesrinuNoSplashBomb-Nes</td><td>KidChaneleon-Genesis</td><td>Killcans-Nes</td><td>KidKlownlsCrazyChase-Snes</td><td>JusticeLeagueTaskForce-Genesis</td></tr><tr><td>KidNiiRadicalNinja-Nes Kange-Nes</td><td>Ki ss</td><td>KingOTheMonsend2-Genesis KungFuKid-Sms</td><td>Kia Chmeds oin</td><td></td><td>KnighOrTheRound-Snes</td><td>KidKlownlsNighMayorWorld-Nes Krull-Auri2600</td></tr><tr><td>LastActonHero-Snes</td><td>LegendaryWings-Nes LastBatte-Genesis</td><td>LethalEnforcers-Genesis LastSarfighter-Nes</td><td>LethalEnforcensGunFighters-Genesins LawnmowerMan-Genesis</td><td>Legend-Snes</td><td>LegendO6Galahad-Genesis</td><td>LastActionHero-Nes LegendOfKage-Nes</td></tr><tr><td>LegendOPinceValiant-Nes LitfleMermaid-Nes</td><td>LowGMasTheLowGravityMas-Nes MagicalTaruranoKun-Genesis</td><td>LackyDineCaperStarringDonaldDuck-Sms</td><td>MCKids-Nes</td><td>LethalWeapon-Saes MUSHA-Genesis</td><td>LifeForce-Nes MagicBoy-Snes</td><td>LineOfFire-Sms</td></tr><tr><td>MagicalQuestSurringMickeyMouse-Sncs MarioBros-Nes</td><td>Marko-Genesis</td><td>Marsupilami-Genesis Magnas-Nes</td><td>MacuRenjishi-Genesis MarvelLand-Genesis</td><td>MappyLand-Nes Mask-Snes</td><td>MarbleMadness-Geneis MasterOfDarkness-Sms</td><td>MagicSwoed-Snes MarbieMadness-Sms</td></tr><tr><td>MechanizedAnack-Nes</td><td>MegaMan-Nes MichaeUacksonsMoonwalker-Genesis</td><td>MegaMan2-Nes MichaellacksonsMoowalker-Sms</td><td>MegaMasTheWilyWa-Genesis MickeyMossecapade-Nes</td><td>MidnighResistance-Gesesis MegaSWTV-Genesis</td><td>MegaTurrican-Genesis</td><td>McDonaldsTreasureLandAdventure-Genesis MendelPalace-Nes</td></tr><tr><td>MetalStorm-Nes MighyMorphinPowerRangers-Genesis</td><td>MightyMerphinPowerRangersTheMovie-Geneis MonserParty-Nes</td><td>MightyMorphinPowerRangersTheMovie-Snes</td><td>Mellipede-Nes MoonPatrol-Atari2600</td><td>MitsumeGaTooru-Nes</td><td>MightyBomblack-Nes MoeroTwinBeeCinnamonHakaseOSukue-Nes</td><td>MightyFinalFight-Nes MonsterlnMyPocket-Nes</td></tr><tr><td>MonsterLaiz-Genesis MrNutz-Genesis</td><td>MrNutz-Snes MysteryQuest-Nes</td><td>MsPacMan-Genesis NARC-Nes</td><td>MsPacMan-Nes</td><td>MortalKembut-Genesis MsPacMan-Sms</td><td>MortalKombat3-Genesis MsPacman-Atari2600</td><td>MoralKombatl-Genesis</td></tr><tr><td>MyHero-Sms NewZealandStory-Sms</td><td>Ninja-Sms</td><td>NinjaCrusaders-Nes</td><td>NHIL.94-Genesis NinjaGaiden-Nes</td><td>NHL94lonl-Genesis NinjaGaiden-Sms</td><td>NinjaGaidenllITheAncientShipOfDoom-Nes NameThisGame-Atari2600</td><td>MutantVirsCrisislnACompaterWorld-Nes NewZealandStury-Genesis</td></tr><tr><td>NinjaKid-Nes OverHorizon-Nes</td><td>NoahsAk-Nes PoWPisnnOfWar-Nes</td><td>NormysBeachBabeORama-Genesis PaclnTime-Snes</td><td>OperationWelf-Nes PacManNamco-Nes</td><td>Onifants-Genesis PacMania-Genesis</td><td>Er  </td><td>NinjaGaidelITheDarkSwordOfChacs-Nes OurToLunch-Snes</td></tr><tr><td>Paperboy-Genesis</td><td>Paperboy-Nes Phalanx-Snes</td><td>Paperboy-Sms Phelios-Genesis</td><td>Paperboy2-Genesis Phoenix-Atari2600</td><td>Parodius-Nes</td><td>Parodius-Snes</td><td>PanicRestaurant-Nes PeaceKeepers-Snes</td></tr><tr><td>PenguinKunWars-Nes PaFighter-Sas</td><td>Pitfall-Atari2600 Pooyan-Nes</td><td>PitalITheMayanAdventuro Popeye-Nes</td><td></td><td>PinkGeesToHolywood-Genesis</td><td>PiratesOfDarkWater-Snes</td><td>PiFighte-Genesis Pong-Atari2600</td></tr><tr><td>Pooyan-Atari2600 PowerAthlete-Gesesis</td><td>PowerPiggsOfTheDarkAge-Snes PsychicWorld-Sms</td><td>PowerStrike-Sms Qbert-Atari2600</td><td>PowerStrikell-Sms PanchOus-Nes</td><td>Prate min</td><td>Pra Ss</td><td>PoseidonWars3D-Sms PehistorikMan-Snes</td></tr><tr><td>PrivateEye-Atari2600 PuttySquad-Snes RTypelll-Sness</td><td>RadicalRes-Genesis QBert-Nes RambolII-Genesis</td></table>

# B. Implementation Details

# B.1. Architecture

Th I or me dots 3Net VD (Blat a 03 w5rae pe memory length of 6. The default input resolution for both models is $2 5 6 \times 2 5 6$ .

# B.2. Training

The latent actucoer  taior0Kses o scratc wi  batc z We plyhemW optimizer (Loshchilov & Hutter, 2019) with a learning rate of $2 . 5 \times 1 0 ^ { - 5 }$ and a weight decay of 0.01. The hyperparameter $\beta$ is set to $2 \times 1 0 ^ { - 4 }$ to achieve a good balance between representation capacity and context disentangling ability.

The autoregressive world model is trained for 80K steps with a batch size of 64 and a learning rate of $5 \times 1 0 ^ { - 5 }$ on 16 NVIDIA 100 GPUs Wedopt  csie arin ratheduler wi 10K waru stes. T a predi l, oep T ange of 0.0 to 0.7, with an interval of 0.1.

![](images/9.jpg)  
Fiure Diversiy urrainig datast. Our uratedatase agregates  extremely widerange  c.

F zo of $1 0 \ : \mathrm { H z }$ for training.

# B.3. Sampling

Byu  a al results. A timestep shifting strategy (Kong et al., 2024) is also applied to enhance generation quality.

# B.4. Visual Planning on the Procgen Benchmark

We summarize our model predictive control process for visual planning as below:

defined as the maximum reward obtained along the planned trajectory.   
2.At each iteration, we sample a population of $N$ action sequences, each with a length of $L$ , from a distribution. The initial distribution is set to a uniform distribution over four actions (LEFT, DOWN, UP, RIGHT).   
For ea sample actin sequence, the worlmode is used to predict the resultng trajcry and the rear is calculated for each trajectory.   
4. The top $K$ action sequences with the highest rewards are selected, and we update the distribution by increasing the sampling probabilities of these selected actions.   
5. A new set of $N$ $i$ Cross-Entropy Method iterations.   
6. After $i$ optimization iterations, the first $T$ action in the action sequence with the highest probability is executed in the emTh pnia hewhenheal sta s hive whehearc m s x.

In practice, we use $i = 2$ Cross-Entropy Method iterations. For each iteration, $N = 1 0 0$ action sequences with a length of $L = 1 5$ are sampled, and the best $K = 1 0$ samples are selected to update the action sampling distribution. After the

optimization procedure is done, the first $T = 5$ actions are executed in the environment. We set the search limit to 20 steps.   
For efficiency, we use only 3 denoising steps and disable classifier-free guidance during planning.

# B.5. Visual Planning on the $\mathbf { V P } ^ { 2 }$ Benchmark

We train low-resolution variants at a resolution of $6 4 \times 6 4$ for control-centric evaluation on the $\mathrm { { V P ^ { 2 } } }$ benchmark. We follow the official protocol of $\mathrm { { V P ^ { 2 } } }$ to evaluate all models. During adaptation, we use 5K given trajectories for Robosuite and 35K s  ot size of 32 and a learning rate of $5 \times 1 0 ^ { - 4 }$ .A cost below 0.05 is considered as success in Robosuite tabletop pushing tasks.

# B.6. iVideoGPT Training Details

Wo eeoino tee p ex ese  oeiboctsr teAo pshata.heult are tested on 256 test videos.

# C. Additional Results

# C.1. Action Transfer

ualitative oparison.e ualtativey paraWorl trane o 0 teps wiherbaseline Fure The u ob

VliztsW mon ul t s generated by transferring a latent action sequence with a length of 20.

uW o view shifts.

# C.2. World Model Adaptation

T ue ue.  uy Thidoeh cul beizameho moeptat conventional methods.

# C.3. Action Creation through Clustering

As mtS.AdaWor nls eas catxienbecnto ts hrou late R derived with AdaWorld, we adopt the $\Delta$ PSNR metric following Genie (Bruce et al., 2024). Table 9 shows the $\Delta$ PSNR of the latent action decoder predictions. The larger the $\Delta$ PSNR, the more the predictions are affected by the action conditions AWorab does not support a customizable number of actions, as it is fixed once trained.

<table><tr><td rowspan="2">Method</td><td colspan="9">ΔPSNR</td></tr><tr><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td></tr><tr><td>Discrete cond.</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>6.47</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>AdaWorld</td><td>5.67</td><td>5.15</td><td>7.28</td><td>8.23</td><td>6.26</td><td>7.32</td><td>6.07</td><td>6.68</td><td>6.53</td></tr></table>

# D. Something-Something v2 Categories for Action Transfer

, Co  i h", "sh hogh  U h  hb,  " u

# E. Selected Scenes for Visual Planning

Tuit  ioo

# F. Related Work

# F.1. World Models

Dv X i ehepliathenheorr a u H &, Y  0  y ve  iHa Th lt 0He 0H a H u Zl0Zu 0Zhu Z Bu et al., 2024; Wang et al., 2025; Qi et al., 2025; Wu et al., 2025).

preai assiideWatte  l0e l 0Wu l. H tmssoa H by assuming that the parameters of the pretrained video models are frozen or inaccessible.

W  rul haplrivt o unique applications for adaptable world modeling.

# F.2. Latent Action from Videos

H zat T a sacdeveuc potoype u  3 Howevehe eqoly paired videos, which limits their scalability for most real-world tasks.

T pl h Crale&Bke . However, hyiu et i cexiy Whio e  u et al., 2025). Thus, the potential of latent actions for adaptable world modeling remains underexplored.

![](images/10.jpg)  
transfer it to another context, while the other baselines fall short in doing so.

![](images/11.jpg)  
perform the correct actions.

![](images/12.jpg)  
Figure 12. Additional action transfer results by AdaWorld. Best viewed with zoom-in.

![](images/13.jpg)  
Figure 13. Additional action transfer results by AdaWorld. Best viewed with zoom-in.

![](images/14.jpg)  
Figure 14. Additional action transfer results by AdaWorld. Best viewed with zoom-in.

![](images/15.jpg)  
Figure 15. Additional action transfer results by AdaWorld. Best viewed with zoom-in.

![](images/16.jpg)  
Figure 16. Additional action transfer results by AdaWorld. Best viewed with zoom-in.

![](images/17.jpg)  
Fgur  Heist scenes for planning evaluation.Thegure illustrates the initial states all selectese.

![](images/18.jpg)  
uJ  or p valaiTegulua all.

![](images/19.jpg)  
Figur Maze cenes or plannng valuation. Thegureustrates henitialstate  all select s.

![](images/20.jpg)  
avFly   vaaTeul lat .

![](images/21.jpg)  
ollouts, and significant view changes.