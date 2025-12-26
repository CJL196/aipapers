# 从经验中学习的 VLA

物理智能 H us k TB J # https://pi.website/blog/pistar06

![](images/1.jpg)  
F and conditioning the VLA on these updated advantage estimates in turn improves policy behavior.

摘要—我们研究了视觉-语言-动作（VLA）模型如何通过实际部署和强化学习（RL）进行改进。我们提出了一种通用方法，称为基于优势条件策略的经验与修正的强化学习（RECAP），该方法通过优势条件实现VLA的RL训练。我们的方法将异构数据融入自我改进过程中，包括演示数据、来自策略执行的数据以及在自主执行期间提供的专家远程操作干预。ReCAP首先通过离线RL预训练一个通用型VLA，我们称之为$\pi _ { 0 . 6 } ^ { * }$，然后可以通过机器人上的数据收集进行专门化，以在下游任务上达到高性能。我们展示了使用完整RECAP方法训练的$\pi _ { 0 . 6 } ^ { * }$模型能够在真实家庭中叠洗衣物、可靠地组装箱子，并使用专业级意式咖啡机制作浓缩咖啡。在一些最困难的任务中，RECAP的任务吞吐量提高了超过一倍，任务失败率大约降低了一半。

# 一、引言

如果你不怕尝试，你会发现可以学到许多惊人的东西。——罗伯特·A·海因莱因《有太空服，将旅行》

实践造就完美：尽管人类在获取新技能方面展现出显著的灵活性，但掌握一项技能必然需要通过反复尝试来学习。借助于通用型机器人基础模型，如视觉-语言-动作（VLA）模型，我们可以通过提示灵活地为通用机器人指定任务。但与人类一样，这些模型也需要通过练习来掌握技能。这不仅意味着利用演示数据，还包括自我收集的经验数据，使得策略能够纠正其在实际部署中所犯的错误，提升速度和鲁棒性，超越人类远程操作的水平，并适应新的部署条件。通过自主练习进行学习的基础，如强化学习（RL）所形式化的，早已为人所知，但在通用和可扩展的机器人学习系统中实例化这些原则面临着重大的挑战：为大模型设计可扩展且稳定的RL方法，处理来自不同策略的异质数据，以及在现实世界中设置带有奖励反馈的RL训练，这里奖励信号可能模糊或具有随机性。

在本文中，我们介绍了RECAP，一种方法使得VLA模型能够在训练流程的所有阶段中纳入奖励反馈，从预训练到基于自主执行的数据训练。RECAP旨在通过一种通用的方案来解决这个问题，结合演示、自主经验和专家干预。从针对通用VLA的训练方案出发，并在来自多种不同机器人平台的多样化数据上进行训练，RECAP首先通过离线强化学习对VLA进行预训练，随后在通过部署收集的数据上进行额外训练。在这些部署过程中，机器人根据每次试验的结果获得（稀疏的）奖励反馈，并可能得到额外的专家干预以纠正错误。训练过程遵循离线强化学习方案：我们训练一个价值函数来评估成功完成任务的进展，然后利用该价值函数来估计数据集中每个动作的优势。通过基于这一优势对策略进行条件化，我们可以获得改进后的策略。图1提供了RECAP的高层次概述。

![](images/2.jpg)  
Fig. 2: Some of the tasks learned by RECAP. $\pi _ { 0 . 6 } ^ { * }$ trained with ReCAP can make espresso drinks, assemble cardboard boxes, and fold diverse and realistic pouring liquids, and folding laundry requires generalization to a wide range of clothing items.

我们可以使用RecAP来训练复杂任务的策略，例如折叠各种衣物、组装箱子或制作意式浓缩咖啡。图2展示了这些任务中的一些。该方法首先通过离线强化学习在多样化的多任务和多机器人数据集上对$\pi _ { 0 . 6 } ^ { * }$模型进行预训练。$\pi _ { 0 . 6 } ^ { * }$是针对强化学习的$\pi _ { 0 . 6 }$模型的适应，而$\pi _ { 0 . 6 }$则是在$\pi _ { 0 . 5 }$ [5]的基础上进行了改进，增加了更大的主干网络和更多样化的条件设定 [6]。$\pi _ { 0 . 6 } ^ { * }$增加了对二值化优势值进行条件设定的能力，这使得可以整合价值函数以改善策略。在预训练后，$\pi _ { 0 . 6 } ^ { * }$对下游任务进行微调，并通过示范进行训练，接着执行一轮或多轮机器人上的数据收集，以通过强化学习改善模型。在某些最具挑战性的任务上，使用RECAP对$\pi _ { 0 . 6 } ^ { * }$进行训练的吞吐量超过了一倍，并且可以将失败率降低$2 \times$或更多。这使得$\pi _ { 0 . 6 } ^ { * }$能够连续运行13个小时制作意式浓缩咖啡，在新的家中不间断地折叠新型衣物超过两个小时，并组装用于工厂真实包装的箱子。尽管RECAP基于以前研究中探索的各个算法组件，但这些组件的特定组合是新颖的，结果首次表明，具有人工奖励反馈和干预的通用强化学习方案可以显著提高VLA模型在实际部署中收集的经验的鲁棒性和吞吐量。

# II. 相关工作

通过模仿学习训练的策略被认为会遭受复合错误，并且在最佳情况下，其性能只能与示范数据相当。本工作的目标是提高视觉-语言-动作策略的可靠性和速度，通过超越从离线示范中进行的模仿学习。先前的研究已经使用在线干预来改进机器人操作策略。我们采用了这种干预的一种形式，称为人类门控DAgger。与这些研究相比，我们的方法同时使用专家干预和完全自主的经验，形成一个基于强化学习的框架，整合多种数据来源。有大量工作探讨如何利用强化学习自主改进机器人操作策略，包括使用基于扩散的策略、在多任务设置中的方法，以及使用预训练的多任务策略。不同于这些研究，我们研究如何将真实世界的强化学习扩展到大型视觉-语言-动作策略，以实现长时间跨度、细粒度的操作任务。许多近期研究探讨了如何通过强化学习改进基础视觉-语言-动作模型。一些研究直接将近端策略优化（PPO）算法及其变体应用于视觉-语言-动作的微调，产生的方案在有效和可扩展的真实世界强化学习中难以推广。另一条研究线索探讨了在预训练视觉-语言-动作模型上进行强化学习微调，其中强化学习要么训练残差策略，要么微调动作头网络，选择或细化视觉-语言-动作提出的动作，或优化在基于扩散的视觉-语言-动作噪声空间中动作的策略。这些研究中的一些还探索了将学习到的行为回归到视觉-语言-动作中以实现端到端的迭代改进。先前的工作通常使用离散动作或简单的高斯连续动作分布。一个关键的区别是我们使用（迭代）离线强化学习端到端地训练整个视觉-语言-动作模型，配合表现力丰富的流匹配视觉-语言-动作模型。这得益于一种简单且可扩展的基于优势的策略提取方法，它消除了在大型视觉-语言-动作模型中使用策略梯度类型目标的复杂性。在我们的比较中，我们展示了这一方法显著优于更传统的基于政策梯度的提取方案。

与 RECAP 在方法论上的更紧密关系，若干先前的研究将价值函数和对真实机器人的端到端强化学习（RL）训练结合起来。例如，Huang 等 [43] 将校准的 Q 学习应用于抓取任务的离线演示数据集，未包含在线改进阶段。Zhang 等 [44] 使用直接偏好优化（DPO）从人类偏好中优化拾取和放置技能，利用来自 VLA 的在线推演。最后，Zhai 等 [45] 和 Ghasemipour 等 [46] 分别使用 PPO 和 REINFORCE 与完成时间价值函数训练 VLA，执行如移动碗、展开垫子和推对象等任务。与这些先前的研究相比，我们描述了一种迭代的离线 RL 框架，具有多个优势。首先，我们的方法支持高容量的扩散和基于流的 VLA，这与先前研究中的离散动作模型不同。其次，通过使用优势条件策略提取策略，我们避免了对在线 PPO 或 REINFORCE 的需求，该策略可以利用所有以往的（离政策或离线）数据。最后，我们的评估由复杂、灵巧和时间延续的任务组成，在处理可变形物体、液体和多阶段任务时，我们的方法提高了大约 $2 \times$ 的吞吐量。先前的研究探讨了根据奖励、价值和优势对策略进行条件化的想法 [4756]，包括使用无分类器引导的方法 [4]。我们将这种方法扩展到预训练和微调大型通用 VLA 策略 [5]，结合多种数据源（包括演示、干预和自主策略推演），以学习真实的机器人操作任务。最近的研究还探讨了如何有效地训练多任务、语言条件的奖励函数 [5763] 和价值函数 [45, 64, 65]。在这些工作的基础上，我们还训练了一个语言条件的分布式价值函数，使我们能够估计优势条件 VLA 训练框架的状态-动作优势。

# III. 准备工作

强化学习。我们考虑标准的强化学习设置，在这种设置中，智能体由策略 $\pi ( \mathbf { a } _ { t } | \mathbf { o } _ { t } )$ 给出，根据观测 $\begin{array} { r l r } { \mathbf { o } _ { t } } & { { } \in } & { \mathcal { O } } \end{array}$ 选择动作 $\mathbf { a } _ { t }$。我们将轨迹定义为 $\tau = \left( \mathbf { o } _ { 0 } , \mathbf { a } _ { 0 } , \cdots , \mathbf { o } _ { T } \right) \in { \mathcal { O } } \times { \mathcal { A } } \cdots { \mathcal { O } }$。策略 $\pi ( \mathbf { a } _ { t } | \mathbf { o } _ { t } )$ 和随机动态 $p ( \mathbf { o } _ { t + 1 } | \mathbf { o } _ { t } , \mathbf { a } _ { t } )$ 诱导了一种轨迹的分布 $\rho _ { \pi } ( \tau )$：

$\begin{array} { r } { \rho _ { \pi } ( \tau ) = p ( \mathbf { o } _ { 0 } ) \prod _ { t = 0 } ^ { T - 1 } \pi ( \mathbf { a } _ { t } | \mathbf { o } _ { t } ) p ( \mathbf { o } _ { t + 1 } | \mathbf { o } _ { t } , \mathbf { a } _ { t } ) } \end{array}$ 奖励函数由 $r ( \mathbf { o } _ { t } , \mathbf { a } _ { t } )$ 给出，我们将其简写为 $r _ { t }$ 以简化表示，其中 $r _ { T }$ 是终端奖励。我们可以定义折扣累积奖励，或称为回报，如下所示：$\begin{array} { r } { R ( \tau ) = \sum _ { t = 0 } ^ { T } r _ { t } } \end{array}$（我们并未使用折扣因子，尽管可以很容易加入一个）。强化学习的目标是最大化累积奖励（或回报），学习一个策略以最大化 $\begin{array} { r } { \mathcal { I } ( \pi ) = \mathbb { E } _ { \tau \sim \rho _ { \pi } } [ R ( \tau ) ] = \mathbb { E } _ { \tau \sim \rho _ { \pi } } [ \sum _ { t = 0 } ^ { T } r _ { t } ] } \end{array}$。然后，策略被定义为 $\begin{array} { r } { \overline { { V ^ { \pi } ( \mathbf { o } _ { t } ) } } = \mathbb { E } _ { \tau _ { t + 1 : T } } [ \sum _ { t = t } ^ { T } r _ { t } ] } \end{array}$。值函数 $\mathbf { a } _ { t }$ $\begin{array} { r } { A ^ { \pi } ( \mathbf { o } _ { t } , \mathbf { a } _ { t } ) \ = \ \mathbb { E } _ { \rho _ { \pi } ( \tau ) } [ \sum _ { t ^ { \prime } = t } ^ { t + N - 1 } \tilde { \mathbf { \Gamma } } _ { t ^ { \prime } } ^ { - } + V ^ { \pi } ( \mathbf { o } _ { t + N } ) ] \ - \ V ^ { \pi } ( \mathbf { o } _ { t } ) , } \end{array}$ 对应于 n 步估计。

正则化强化学习。与其最大化 $\mathcal { I } ( \pi )$，更常见的是在强化学习中使用正则化，优化一个政策，使其在最大化奖励的同时保持与某个参考政策 $\pi _ { \mathrm { r e f } }$ 的接近性 [6670]。这在我们希望对同一数据进行多次梯度更新时尤为重要，此时 $\pi _ { \mathrm { r e f } }$ 通常对应于收集训练数据的行为策略。可以通过目标 $\mathcal { I } ( \pi , \pi _ { \mathrm { r e f } } ) =$ $\begin{array} { r } { \mathbb { E } _ { \tau \sim \rho _ { \pi _ { \theta } } } [ \sum _ { t = 0 } ^ { T } \gamma ^ { t } r _ { t } ] \ - \ \beta \mathbb { E } _ { \mathbf { o } \sim \rho _ { \pi _ { \theta } } } [ D ( \pi ( \cdot | \mathbf { \tilde { o } } ) | | \pi _ { \mathrm { r e f } } ( \cdot | \mathbf { o } ) ) ] } \end{array}$ 来形式化，其中 $D$ 表示某种散度度量。当 $D$ 为 $\mathrm { K L }$ 散度时，我们得到了著名的结果：$\hat { \pi } ( \mathbf { a } | \mathbf { o } ) \propto \pi _ { \mathrm { r e f } } ( \mathbf { a } | \mathbf { o } ) \exp ( A ^ { \pi _ { \mathrm { r e f } } } ( \mathbf { o } , \mathbf { a } ) / \beta )$ 是 $\operatorname* { m a x } _ { \pi } J ( \pi , \pi _ { \mathrm { r e f } } )$ 的解，拉格朗日乘子为 $\beta$ [6770]。我们的优势条件政策提取方法基于一个密切相关但不太知名的结果：如果我们定义政策 $\hat { \pi } ( \mathbf { a } | \mathbf { o } ) \propto { \pi } _ { \mathrm { r e f } } ( \mathbf { a } | \mathbf { o } ) p ( I | A ^ { \pi _ { \mathrm { r e f } } } ( \mathbf { o } , \mathbf { a } ) ) ^ { \beta }$，其中 $\begin{array} { r } { p ( I \vert { \cal A } ^ { \pi _ { \mathrm { r e f } } } ( { \bf o } , { \bf a } ) ) = { \cal g } ( { \cal A } ^ { \pi _ { \mathrm { r e f } } } ( { \bf o } , { \bf a } ) ) / \int { g ( { \cal A } ^ { \pi _ { \mathrm { r e f } } } ( { \bf o } , { \bf a } ^ { \prime } ) ) \mathrm { d } { \bf a } ^ { \prime } } } \end{array}$ 是任何动作 a 在 $\pi _ { \mathrm { r e f } }$ 下的表现被单调递增函数 $g$ 衡量的概率，则可以保证 $\hat { \pi }$ 在性能上优于 $\pi _ { \mathrm { r e f } }$，即 $\mathcal { I } ( \hat { \pi } ) \geq \mathcal { I } ( \pi _ { \mathrm { r e f } } )$ [4, 71]。我们将在第四节B部分利用这一属性推导我们的政策提取方法。使用这个定义，我们可以通过解决以下最小化问题，从 $\hat { \pi }$ 的闭式定义中获得参数化策略：$\begin{array} { r } { \operatorname* { m i n } _ { \theta } \mathbb { E } _ { s \sim \rho _ { \pi _ { \mathrm { r e f } } } } [ K L ( \hat { \pi } , \pi _ { \theta } ) ] } \end{array}$。

# IV. 通过优势条件策略的经验与修正的强化学习（重述）

我们的方法包含以下步骤，可以重复一遍或多遍以提升基础 VLA 模型：1）数据收集。我们在任务上运行 VLA，用任务结果标签标记每个回合（这些标签决定奖励），并可选地提供人类干预，以提供早期迭代中错误的修正示例。2）值函数训练。我们使用至今收集的所有数据来训练一个大型的多任务值函数，称为 $V^{\pi_{\mathrm{ref}}}$，它可以检测失败并评估完成任务的预期时间。3）优势条件训练。为了利用该值函数改进 VLA 策略，我们在 VLA 前缀中加入基于该值函数推导的优势值的最优性指标。这个“优势条件”方法提供了一种简单有效的方式，从 suboptimal 数据中提取出更优的策略。图 1 说明了训练过程的总体结构，而图 3 则提供了值函数和策略架构的更具体细节。我们的预训练阶段包含对整个预训练数据集执行步骤（2）和（3），该数据集包含来自多个任务和各种不同机器人数万小时的演示。随后，我们执行步骤（1）、（2）和（3）一遍或多遍，以进一步改进 VLA，使用自主收集的数据。我们将在下面描述值函数训练和策略训练步骤，然后在第五节中介绍我们这一方法在训练 $\pi_{0.6}^{*}$ 时的具体实例化。

# A. 分布式价值函数训练

为了训练一个能够作为任何任务的可靠评论者的值函数，我们使用多任务分布式值函数 $p _ { \phi } ( V | \mathbf { o } _ { t } , \ell ) \in \Delta _ { B }$ 来表示 $V ^ { \pi _ { \mathrm { r e f } } }$ [72]，将观察值 $\mathbf { o } _ { t }$ 和语言指令 $\ell$ 映射到 $B$ 个离散值区间的分布。在我们的实现中，这个值函数使用与 VLA 策略相同的架构，但采用更小的 VLM 主干网络。使用 $\begin{array} { r } { R _ { t } ( \tau ) = \sum _ { t ^ { \prime } = t } ^ { \tilde { T } } \dot { r _ { t ^ { \prime } } } } \end{array}$ 来表示从时间步 $t$ 到结束的轨迹 $\tau$ 的经验回报，我们通过首先将经验回报值 $R _ { t } ( \tau )$ 离散化为 $B = 201$ 个区间（使用 $R _ { t } ^ { B }$ 表示离散化后的回报），然后在当前数据集 $\mathcal { D }$ 的轨迹上最小化交叉熵 $H$ 来训练 $p _ { \phi } ( V | \mathbf { o } _ { t } , \ell )$ ：

$$
\operatorname* { m i n } _ { \phi } \mathbb { E } _ { \tau \in \mathcal { D } } \left[ \sum _ { \mathbf { o } _ { t } \in \tau } H ( R _ { t } ^ { B } ( \tau ) , p _ { \phi } ( V | \mathbf { o } _ { t } , \ell ) ) \right] .
$$

这是一个用于政策价值函数的蒙特卡洛估计器，该政策由数据集 $\mathcal { D } $ 表示（即行为策略 $\pi _ { \mathrm { r e f } } $）。我们可以利用 $\begin{array} { r } { V ^ { \pi _ { \mathrm { r e f } } } ( o _ { t } , \ell ) = \sum _ { b \in [ 0 , B ] } p _ { \phi } ( V = b | \mathbf { o } _ { t } ) v ( b ) } \end{array}$ 从学习到的价值分布中提取连续的价值函数（因此也提取出优势），其中 $v ( b )$ 表示与桶 $b$ 对应的价值。在预训练阶段，数据集 $\mathcal { D } $ 对应于人类示范，价值函数捕捉我们所条件化的任务和元数据的预期回报，而在后续迭代中，它偏向于示范的回报和学习到的策略的加权组合。虽然这个在政策估计器不如更经典的离政策Q函数估计器最优，但我们发现它简单且高度可靠，同时仍然允许在模仿学习上有实质性的改进。我们的方法可以在未来的工作中扩展以适应离政策估计器。

# B. 通过优势条件提取策略

一旦我们获得了价值函数 $V ^ { \pi _ { \mathrm { r e f } } }$，我们需要一种方法来训练改进的策略，这被称为策略提取。在我们的设置中，一种有效的策略提取方法需要满足几个标准。首先，它需要有效地利用多样的离策略数据，包括初始示范、专家干预，以及来自最新策略和旧策略的自主轨迹。这与离线强化学习方法面临的挑战密切相关。其次，它需要具有可扩展性，能够轻松应用于大型变分自编码器 (VLA) 模型，包括使用流匹配或扩散生成动作的模型。第三，它需要有效利用好的（近于最优的）和差的（次优的）数据，这在我们想要利用自主经验改进策略时尤为重要。

![](images/3.jpg)  
Fig. 3: Interaction between the $\pi _ { 0 . 6 } ^ { * }$ VLA and value function during RECAP training. The $\pi _ { 0 . 6 } ^ { * }$ VLA uses a pre-trained VLM backbone. Training follows the KI recipe [73], with next-token prediction on many data sources in pre-training, and an flow-matching action-expert with stop gradient. The VLA is conditioned on a binarized advantage indicator, obtained from a separate value function initialized from a pre-trained but smaller VLM model.

在现有的策略提取方法中，策略梯度方法（包括正则化策略梯度和重参数化梯度）可能是使用最广泛的[66, 74]，但这些方法在流匹配模型中难以应用，因为它们并不容易提供可处理的对数似然性，这使得它们难以扩展到现代VLA架构（见第六节的比较）。一个替代方法是使用加权回归方法，如AWR[68, 75, 76]，这些方法隐式地对行为策略进行正则化，并使用简单的（重要性加权）监督学习目标。然而，这些方法会丢弃或显著降低大量数据的权重，有效地实现了一种过滤模仿技术。相反，我们采用了一种优势条件化的变体[48]，在所有数据上进行策略的监督学习训练，但额外输入指示基于优势的行动的最优程度。这与文献中多种方法密切相关，这些方法建议根据结果轨迹的某些函数对策略进行条件化[47, 50]。我们方法中的具体表述与CFGRL[4]最为相关。基于第三节中的表述，我们可以应用贝叶斯规则重写策略改进的概率为$p ( I \vert A ^ { \pi _ { \mathrm { r e f } } } ( \mathbf { o } , \mathbf { a } ) ) = \pi _ { \mathrm { r e f } } ( \mathbf { a } \vert I , \mathbf { o } ) / \pi _ { \mathrm { r e f } } ( \mathbf { a } \vert \mathbf { o } )$。将其应用于我们的设置，并包括语言条件化，我们可以获得第三节中描述的改进的正则化策略的另一种闭式解。

![](images/4.jpg)  
length to $( - 1 , 0 )$ well as the speed of progress.

$$
\hat { \pi } ( \mathbf { a } , | \mathbf { o } , \ell ) \propto \pi _ { \mathrm { r e f } } ( \mathbf { a } | \mathbf { o } , \ell ) \left( \frac { \pi _ { \mathrm { r e f } } ( \mathbf { a } | I , \mathbf { o } , \ell ) } { \pi _ { \mathrm { r e f } } ( \mathbf { a } | \mathbf { o } , \ell ) } \right) ^ { \beta } .
$$

对于特例 $\beta = 1$ ，${ \hat { \pi } } ( \mathbf { a } , | \mathbf { o } , \ell ) = \pi _ { \mathrm { r e f } } ( \mathbf { a } | I , \mathbf { o } , \ell )$ 。

因此，我们可以在不需要显式表示改进概率 $p ( I | A ^ { \pi _ { \mathrm { r e f } } } ( \mathbf { o } , \mathbf { a } ) )$ 的情况下表示 $\hat { \pi }$ ，只要我们训练策略，使其能够同时表示 $\pi _ { \mathrm { r e f } } ( \mathbf { a } | \mathbf { o } , \ell )$ 和 $\pi _ { \mathrm { r e f } } ( \mathbf { a } | I , \mathbf { o } , \ell )$ 。这个原理类似于无分类器引导的方法，其中扩散模型在有条件变量和无条件变量的情况下进行数据建模 [4]。我们假设改进指示符 $I$ 遵循一个具有任务依赖改进阈值 $\epsilon _ { \ell }$ 的δ分布。这个阈值使我们能够控制最优性指示符，并最小化在训练后寻找衰减因子 $\beta$ 的需要，以锐化改进条件分布。政策目标则对应于最小化以下负对数似然性：

$$
p ( I | A ^ { \pi _ { \mathrm { r e f } } } ( o , a , \ell ) ) = \delta ( A ^ { \pi _ { \mathrm { r e f } } } ( o , a , \ell ) > \epsilon _ { \ell } ) ,
$$

$$
\begin{array} { r l } & { \underset { \theta } { \mathrm { m i n } } \mathbb { E } _ { \mathcal { D } _ { \pi _ { \mathrm { r e f } } } } \Big [ - \log \pi _ { \theta } ( \mathbf { a } _ { t } | \mathbf { o } _ { t } , \boldsymbol { \ell } ) - \alpha \log \pi _ { \theta } ( \mathbf { a } _ { t } | I _ { t } , \mathbf { o } _ { t } , \boldsymbol { \ell } ) \Big ] , } \\ & { \quad \quad \mathrm { w h e r e ~ } I _ { t } = \mathbb { 1 } \big ( A ^ { \pi _ { \mathrm { r e f } } } ( \mathbf { o } _ { t } , \mathbf { a } _ { t } , \boldsymbol { \ell } ) > \epsilon _ { \ell } \big ) . } \end{array}
$$

优势值 $A ^ { \pi _ { \mathrm { r e f } } } ( \mathbf { o } _ { t } , \mathbf { a } _ { t } , \ell )$ 是从上一节的价值函数中获得的，$\alpha$ 是一个折中超参数。在实践中，数据集 $\mathcal { D } _ { \pi _ { \mathrm { r e f } } }$ 包含迄今为止收集的所有数据，包括所有示范和自主任务尝试，因此参考策略 $\pi _ { \mathrm { r e f } }$ 是人类行为与先前部署策略的混合。为了包含人类修正，我们发现强制将 $I _ { t } =$ True（即，正值）用于在自主推理过程中提供的作为人类修正的动作是有用的。如果我们假设人类专家总是能提供良好的修正动作，这一选择是合理的。正如我们将在第五节中讨论的，在实践中，我们的VLA模型同时产生离散和连续的输出，连续分布通过流匹配表示。因此，实际的训练目标结合了离散值的似然与连续值的流匹配目标。在实践中，我们预训练一个模型来表示 $\pi _ { \boldsymbol { \theta } } ( \mathbf { a } _ { t } | I _ { t } , \mathbf { o } _ { t } , \ell )$ 在我们整个预训练数据集上，然后对每个任务执行一次或多次使用在策略推理（并可选地，专家修正干预）的方法的迭代。

# C. 方法总结

我们在算法 1 中提供了完整方法的概述。如本节开头总结的，该方法可以通过三个子程序的应用全面定义：通过自主推演收集数据（可选地接受专家的纠正干预）、根据公式 1 训练价值函数，以及根据公式 3 训练策略。该方法不同步骤之间唯一变化的是提供给每个子程序的数据：预训练阶段使用所有先前的示范数据，而每个技能 $\ell ^ { ( i ) }$ 的专家训练过程使用额外的自主数据。在实践中，专家是从预训练模型微调而来，而最终的通用模型则是从头开始训练。关于该方法的更多细节请见附录 F。

# V. 实现、模型与系统细节

我们用一个称为 π0.6 的 VLA 实例化 RECAP。π0.6 基于 $\pi _ { 0 . 6 }$ VLA，这是一种对 $\pi _ { 0 . 5 }$ VLA [5] 的演变，包含了一些改进，我们在随附的模型卡 [6] 中详细说明。$\pi _ { 0 . 6 } ^ { * }$ 还增加了基于二值优势指标 $I _ { t }$ 进行条件化的能力，使其适合与 RECAP 的强化学习训练。模型架构如图 3 所示。我们在 VLA 的基础上训练一个值函数，遵循 IV-A 节中描述的方法。该值函数也从 VLM 初始化。使用 RECAP 训练该值函数和 VLA 结果生成我们的最终模型，我们称之为 $\pi _ { 0 . 6 } ^ { * }$。在本节中，我们首先详细阐述模型的设计及其如何扩展以使用来自值函数的优势值，然后描述奖励函数和值函数，最后详细介绍我们实现中的训练和数据收集过程。

<table><tr><td>Algorithm 1 RL with Experience and Corrections via Advantage-conditioned Policies (RECAP)</td></tr><tr><td>Require: multi-task demonstration dataset Ddemo</td></tr><tr><td>1: Train Vpre on Ddemo using Eq. 1 2: Train πpre on Ddemo using Eq. 3 and Vpre</td></tr><tr><td>3: Initialize D with demonstrations for </td></tr><tr><td>4: Train V0 from Vpre on D using Eq. 1</td></tr><tr><td>5: Train π0 from πpre on D using Eq. 3and V0 6: for k = 1 to K do</td></tr><tr><td>Collect daa with k1, a it o 7:</td></tr><tr><td>8: Train Vk from Vpre on D using Eq. 1</td></tr><tr><td>9: Train π from πpre on D using Eq. 3 and Vk 10: end for</td></tr></table>

# A. $\pi _ { 0 . 6 }$ 模型

$\pi _ { 0 . 6 }$ 模型 [6] 源自 $\pi _ { 0 . 5 }$ 模型，该模型通过流匹配灵活地表示分块的动作分布，并生成用于高层策略推理的中间文本。它使用知识隔离 (KI) 训练程序 [73]，在连续动作和离散化的词元（包括通过 FAST [77] 离散化的动作）上对整个模型进行端到端训练，同时使用停止梯度来防止流匹配动作专家对模型其他部分的影响。预训练使用来自网络的机器人数据和视觉语言共训练数据。$\pi _ { 0 . 6 }$ 在多个方面改进了 $\pi _ { 0 . 5 }$ ： (i) 预训练数据集通过来自多个机器人平台的额外数据进行增强。 (ii) 基础视觉语言模型 (VLM) 为 Gemma 3 [78] 4B 模型。 (iii) 动作专家的参数数量增加至 860M。

模型可以表示为 $\pi _ { \boldsymbol { \theta } } \big ( \mathbf { a } _ { t : t + H } , \widehat { \ell } | \mathbf { o } _ { t } , \overline { { \ell } } \big )$ ，其中 $\mathbf o _ { t } = [ \mathbf X _ { t } ^ { 1 } , . . . , \mathbf X _ { t } ^ { n } , \mathbf q _ { t } ]$ 包含相机图像 $\mathbf { X }$、机器人的状态配置 $q$，以及 $\boldsymbol { \ell } = \boldsymbol { \ell } _ { t } + \boldsymbol { s }$ 是由整体任务提示 $\ell _ { t }$（例如，“给我做一杯浓缩咖啡”）和附加语言输入 $s$ 组成的语言输入，这些附加输入提供了更多的元数据，从而进一步调节任务的执行方式。模型生成的动作序列 $\mathbf { a } _ { t : t + H }$ 包含在 $50 ~ \mathrm { H z }$ 的关节角度和夹具指令，使用一个独立的“动作专家”——一组专门为动作生成训练的权重（860M 参数），这些权重通过流匹配进行训练，但可以关注模型其余部分的激活。模型还会产生经过标记的离散输出 $\boldsymbol { \hat { \ell } } _ { + }$ ，其中包括用于高层决策的下一个预测子任务的文本表示（例如“拿起咖啡杯”）。由于动作是在 $\hat { \ell } $ 之后生成的，动作生成实际上是以此预测的子任务为条件的，提供了高层指导。在推理时，子任务预测的频率低于动作生成的频率。在训练过程中，模型还会预测动作序列 $\mathbf { a } _ { t : t + H }$ 的标记化表示，使用 FAST 标记器 [77]，作为 KI 方法 [73] 的一部分。我们将这些离散化的动作表示为 $a _ { t : t + H } ^ { \ell }$。动作专家不接收这些作为输入，因而离散和连续动作是独立预测的。这导致最终的训练对数似然 $\mathrm { l o g } ^ { \mathbf { \bar { \alpha } } } \pi _ { \theta } ( \mathbf { a } _ { t : t + H } , a _ { t : t + H } ^ { \ell } , \boldsymbol { \hat { \ell } } | \mathbf { o } _ { t } , \boldsymbol { \ell } )$ 。由于我们首先预测 $\hat { \ell }$ ，因此可以根据以下方式分解此对数似然：

$$
\begin{array} { r l } & { \log \pi _ { \boldsymbol { \theta } } \big ( \mathbf { a } _ { t : t + H } , a _ { t : t + H } ^ { \ell } , \widehat { \ell } | \mathbf { o } _ { t } , \ell \big ) = \log \pi _ { \boldsymbol { \theta } } \big ( \widehat { \ell } | \mathbf { o } _ { t } , \ell \big ) } \\ & { \qquad + \log \pi _ { \boldsymbol { \theta } } \big ( a _ { t : t + H } ^ { \ell } | \mathbf { o } _ { t } , \ell , \widehat { \ell } \big ) + \log \pi _ { \boldsymbol { \theta } } \big ( \mathbf { a } _ { t : t + H } | \mathbf { o } _ { t } , \ell , \widehat { \ell } \big ) . } \end{array}
$$

# B. 从 $\pi _ { 0 . 6 }$ 到 $\pi _ { 0 . 6 } ^ { * }$ 通过优势条件化

为了将优势信息纳入策略中，我们将模型输入扩展为包含额外的改进指示符作为额外的文本输入，当 $I _ { t } ~ =$ True 时输入 "Advantage: positive"，否则输入 "Advantage: negative"。VLA 模型在其他方面与第 V-A 节中描述的相同。优势指示符在训练序列中出现在 $\hat { \ell }$ 之后但在（离散和连续）动作之前，从而仅影响动作的对数似然。对数似然的连续部分无法精确评估，而是通过流匹配损失进行训练 [79]。在某些假设下，可以将流匹配与扩散进行紧密比较，而后者可以被解释为对数似然的下界 [80]，所以我们可以粗略地将离散动作的对数似然与连续动作的流匹配损失的和视为整体动作似然的下界：

$$
\begin{array} { r l } & { \log \pi _ { \boldsymbol { \theta } } ( \mathbf { a } _ { t : t + H } , a _ { t : t + H } ^ { \ell } \vert I _ { t } , \mathbf { o } _ { t } , \ell , \hat { \ell } ) \geq } \\ & { \mathbb { E } _ { \eta , \omega } \Big [ \log p _ { \boldsymbol { \theta } } \big ( a _ { t : t + H } ^ { \ell } \vert I _ { t } , \mathbf { o } _ { t } , \ell , \hat { \ell } \big ) - } \\ & { \qquad \alpha _ { \eta } \left. \omega - \mathbf { a } _ { t : t + H } - f _ { \boldsymbol { \theta } } ( \mathbf { a } _ { t : t + H } ^ { \eta , \omega } , I _ { t } , \mathbf { o } _ { t } , \ell , \hat { \ell } ) \right. ^ { 2 } \Big ] ^ { \mathrm { , } } } \end{array}
$$

wih h $\mathbf { a } _ { t : t + H } ^ { \eta , \omega } = \eta \mathbf { a } _ { t : t + H } + ( 1 - \eta ) \omega$ $\boldsymbol \omega \sim \mathcal { N } ( 0 , \bf { I } )$ $\eta \in [ 0 , 1 ]$ 索引和 $f _ { \theta }$ 表示扩散专家的连续输出。$\alpha _ { \eta }$ 是一个损失加权项（可以选择依赖于噪声）。损失的完整细节请参见附录 C。在训练过程中，我们随机省略指示符 $I _ { t }$，而不是调整损失倍增器 $\alpha$，以使我们能够直接从策略中采样，当 $I _ { t } =$ True 时（对应于在方程 (2) 中设置 $\beta = 1$），或者同时使用条件模型和无条件模型来实现无分类器指导（CFG），这使得推断可以在 $\beta > 1$ 的情况下进行。详见附录 E。

# C. 奖励定义与价值函数训练

由于我们的目标是开发一种通用且广泛适用的方法来基于经验训练视觉语言模型（VLA），我们使用了一种可以适用于几乎任何任务的一般稀疏奖励定义。对于每个回合，我们获得一个标签，指示该回合是否成功。我们根据这一回合级成功标签导出奖励，使得价值函数对应于成功完成回合之前的（负）步骤数。这等价于以下奖励函数，其中 $T$ 对应于回合中的最后一步，$C _ { \mathrm { f a i l } }$ 是一个选择为确保失败的回合具有低值的大常数：

$$
r _ { t } = \left\{ \begin{array} { l l } { 0 } & { \mathrm { i f ~ t = T ~ a n d ~ s u c c e s s } } \\ { - C _ { \mathrm { f a i l } } } & { \mathrm { i f ~ t = T ~ a n d ~ f a i l u r e } } \\ { - 1 } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

通过该奖励函数，我们训练价值函数来预测成功回合中直到成功的剩余步骤数的负值，对于失败回合则预测较大的负值。在实践中，我们将预测值归一化到 $( - 1 , 0 )$ 之间。由于我们在任务长度非常不同的多样化任务上进行训练，因此我们根据每个任务的最大回合长度来归一化值。价值函数输入与 $\pi _ { 0 . 6 } ^ { * }$ VLA 相同的语言输入，并使用相同的架构设计，具有较小的670M参数的VLM主干网络，该网络同样初始化自Gemma 3（见图3）。为了防止过拟合，我们还在少量的多模态网络数据混合上共同训练价值函数。图4展示了价值函数在一些成功和失败回合示例上的可视化，附录B的图13中还有其他可视化。

# D. 预训练、数据收集与经验学习

我们模型预训练阶段使用的数据混合主要遵循 $\pi _ { 0 . 5 }$ [5] 中的配方，包含来自网络的视觉语言数据、子任务的预测 $\hat { \ell }$，以及在各种任务下对低级动作的预测，与 $\pi _ { 0 . 6 } ^ { * }$ 相比，能执行更多的任务，而这些任务在第六节的评估中并未使用。在预训练过程中，我们首先在相同的数据集上训练价值函数，预测每个任务成功完成所需步骤的（负数）。然后，我们估计用于确定基于优势的改进指标 $I _ { t }$ 的每个任务的改进阈值 $\epsilon _ { \ell }$。我们将 $\epsilon _ { \ell }$ 设置为价值函数为任务 l 预测值的 $30\%$ 百分位数。接着，在 VLA 训练期间，我们实时运行价值函数，以估计每个示例的 $A ^ { \pi _ { \mathrm { r e f } } } ( \mathbf { o } _ { t } , \mathbf { a } _ { t } , \ell )$，并根据 $\boldsymbol { \epsilon } _ { \ell }$ 计算 $I _ { t }$，$I _ { t }$ 被作为输入包含到 $\pi _ { 0 . 6 } ^ { * }$ 中，如第 V-A 节所述。由于我们为价值函数使用了相对较小的 VLM 主干网络（670M），价值函数的实时推理在 VLA 训练期间的额外成本最小。

在预训练后，我们开始针对目标任务的策略改进循环。我们首先使用目标任务的演示数据 $\mathcal { D } _ { \ell }$ 对 $\pi _ { 0 . 6 } ^ { * }$ 进行微调。在这一阶段，我们将指示器 $I _ { t }$ 固定为 True，我们发现这样可以带来稍微更好的结果，因此这一阶段对应于监督微调（SFT）。这导致初始策略 $\pi _ { \ell } ^ { 0 }$，然后用于收集额外的数据，这些数据会被添加到 $\mathcal { D } _ { \ell }$。虽然一些剧集是完全自主收集的，但有些是由专家遥控操作员监控的，他们可以介入提供修正。这些修正可以向策略展示如何避免灾难性故障或如何从错误中恢复。然而，请注意，单靠这些修正不太可能解决所有问题：在自主执行期间的干预是一个破坏性事件，即使是专家人类操作员也不能保证干预的一致质量，也无法改善行为的微妙方面，例如整体速度。因此，这些修正更多是用来修正重大错误和克服探索中的挑战，不能单独提供最佳的监督，这与理论相悖[7]。回顾IV-B节，我们强制 $I _ { t } ~ =$ True 对于所有修正，但无论是否提供修正，整个剧集（自主部分和修正部分）都是可选地添加到数据集 $\mathcal { D } _ { \ell }$ 中。

![](images/5.jpg)  
Fig. 5: The robot setup used in our experiments. $\pi _ { 0 . 6 } ^ { * }$ is trained on data from many different robots in pre-training. For the iterative improvement experiments, we use a static bimanual system with two 6 DoF arms with parallel jaw grippers. The arms are controlled at $5 0 ~ \mathrm { H z }$ with joint positions. Observations consist of joint and gripper positions, as well as images from three cameras: a base camera mounted between the arms, and a wrist-mounted camera on each arm. The setup can be mounted flexibly, e.g. on a table.

在数据收集之后，我们在到目前为止为该任务收集的所有数据上微调价值函数，然后利用更新的指标 $I_{t}$ 使用与预训练相同的程序微调策略。价值函数和策略都是从预训练的检查点进行微调，而不是从上一次迭代的策略和价值函数开始。我们发现这样做对于避免多个迭代中的漂移非常有用，尽管也可能通过始终从最后的模型进行微调获得良好结果。我们可以根据需要重复这个过程进行多次迭代，尽管实际上我们发现即使只进行一次迭代通常也能显著改善结果。

# VI. 实验评估

在我们的实验评估中，我们使用 RECAP 来训练 $\pi _ { 0 . 6 }$ 模型，该模型针对一组现实任务：制作浓缩咖啡饮品、折叠各种衣物和组装盒子。每个任务需要多个步骤，持续时间从 5 到 15 分钟，涉及复杂的操作行为（受限的强迫操作、倒液体、操作布料和纸板等），并且需要快速执行以提供高吞吐量。我们在图 5 中展示了我们实验中使用的机器人平台。接下来，我们将详细介绍任务和基线，然后是定量实验。

![](images/6.jpg)  
an espresso machine.

# A. 评估任务

我们的定量评估和比较使用了三个广泛的任务类别，每个类别包含多个具体任务变体：洗衣折叠、咖啡制作和盒子组装。我们将这些任务总结如下，并在图 6 中提供插图： 洗衣（T恤和短裤）。这是 $\pi _ { 0 }$ 论文 [81] 中的标准洗衣折叠任务。该任务涉及在变化的初始条件下，从篮子中取出一件 T恤或短裤，进行平整和折叠。成功的标准是在 200 秒内将一件衣物正确折叠并堆放在桌子的右上角。 洗衣（多样物品）。多样洗衣任务要求折叠更大种类的物品，考虑到 11 种物品类型，包括毛巾、扣领衬衣、毛衣、牛仔裤、T恤、短裤、Polo 衫、裙子、长袖衬衫、袜子和内衣。为了在实验中获得低方差指标，我们在最具挑战性的物品——扣领衬衣上测量性能。然而，策略是在所有物品上进行训练的，并且附带的视频展示了多种衣物的结果。成功的标准是在 500 秒内将目标物品正确折叠并放置在桌上的堆叠中。 洗衣（目标失败移除）。洗衣折叠任务的最终版本考虑了更为结构化的设置，用于我们的消融实验，其中任务涉及从固定的平整初始状态折叠一件橙色 T恤。我们对成功的重视程度最高，成功标准要求在 200 秒内正确折叠，并确保领子始终朝上。我们发现这个任务对于评估 RECAP 是否能够通过强化学习去除特定的不良行为（在这种情况下，领子朝下而不是朝上）非常有用。 咖啡馆（双倍浓缩咖啡）。我们在制作咖啡的挑战性长时间任务上评估我们的策略，该任务使用商业浓缩咖啡机。虽然我们的咖啡馆策略可以制作多种饮品（如拿铁、冰美式、浓缩咖啡等），甚至可以用毛巾清洁咖啡机，但在定量实验中我们专注于双倍浓缩咖啡的任务。该任务包括拿起滤杯，将其放在磨豆机上，研磨咖啡豆，拍紧研磨咖啡豆，将滤杯锁入咖啡机，端上杯子，提取完整的浓缩咖啡，然后进行服务。成功的标准是完成所有步骤并在 200 秒内无关键错误（例如掉落滤杯或洒出咖啡）。 盒子组装。我们在现实工厂部署场景中评估我们的策略，解决包装盒装配的问题。盒子组装涉及从平整的纸板片开始折叠纸箱，贴上标签并将盒子放置在适当的位置。为了定量实验的目的，我们专注于任务的所有部分，并将整体成功标准定义为在 600 秒内从平整的纸箱到组装并堆叠好的纸箱。

# B. 比较与消融实验

我们将RECAP与多个基线进行比较：预训练的$\pi _ { 0 . 5 }$ [5]。该基线不使用强化学习，也不利用RECAP。预训练的$\pi _ { 0 . 6 }$ [6]。它不包含优势指标$I _ { t }$，并使用监督学习进行预训练。强化学习预训练的$\pi _ { 0 . 6 } ^ { * }$。它与其价值函数一起通过强化学习进行预训练，并包含在第V-D节中描述的优势指标$I _ { t }$。$\pi _ { 0 . 6 } ^ { * }$通过示范数据对基础$\pi _ { 0 . 6 } ^ { * }$预训练检查点进行微调，称为“ SFT”，因为优势值在所有示范中都固定为True。我们发现这种离线强化学习预训练的$\pi _ { 0 . 6 } ^ { * }$模型与高质量的SFT相结合，优于标准的SFT（没有离线强化学习预训练），并为使用机器人数据的强化学习提供了良好的起点。

![](images/7.jpg)  
Eo bars show standard error. This metric measures both scess and speed. In all cases, ReA apl to $\pi _ { 0 . 6 } ^ { * }$ (Ours) leads to substantial improvements

![](images/8.jpg)  
diverse laundry and espresso tasks seeing the largest gains success rate, corresponding to more than $2 \times$ reduction in failure rates. For the box assembly task .

$\pi _ { 0 . 6 } ^ { * }$ (urs)。这是在目标任务上使用 ReCAP 训练的最终模型，包括自主推演和专家修正。默认情况下，我们使用 $\beta \ : = \ : 1$ 进行评估。在某些实验中，我们还考虑使用 CFG 进行推理，这对应于 $\beta > 1$。我们还考虑文献中两种替代策略提取方法，作为我们条件优势方法的比较，两者都使用与 RECAP 相同的机器人数据，但采用不同的策略学习方法：AWR。从相同的预训练模型 $\pi _ { 0 . 6 }$（没有条件优势）开始，我们使用基于从我们的价值函数中提取的优势的优势加权回归 [68] 进行微调。PPO。我们实现了 DPPO/FPO [23, 82] 的一种变体，其中我们基于单步扩散目标计算似然，并采用根据 SPO [83] 的 PPO 约束的替代定义（详细信息见附录 D）。

# C. 定量结果

我们在评估中使用两个指标：吞吐量和成功率。吞吐量衡量每小时成功任务执行的数量，从而将速度和成功率合并为一个实际相关的量。成功率衡量成功的插曲所占的比例，并由人工提供的标注得出。评估员被要求根据多个质量指标对插曲进行判断，我们将这些质量指标汇总成一个成功标签。

1) RECAP对策略的改进有多大？为了回答这个问题，我们在图7和图8中展示了主要的定量结果。在所有任务中，最终的$\pi _ { 0 . 6 } ^ { * }$在基线（监督）$\pi _ { 0 . 6 }$模型、RL预训练的$\pi _ { 0 . 6 } ^ { * }$模型和离线$\mathbf { R L } + \mathbf { S F T }$的$\pi _ { 0 . 6 } ^ { * }$模型上显著改善。通过包括机器人上的数据，处理能力在多样化的洗衣折叠和浓缩咖啡任务中增加了两倍以上（从离线$\mathbf { R L } + \mathbf { S F T }$到最终的$\pi _ { 0 . 6 } ^ { * }$模型的改进），而失败率降低了约一半。在相对简单的洗衣任务（T恤和短裤）中，成功率在SFT阶段后已接近最大值，但最终模型的处理能力仍显著提升。在除多样化洗衣外的所有任务中，最终$\pi _ { 0 . 6 } ^ { * }$模型的成功率均在$9 0 \% +$范围内。这使其在实际环境中使用成为可行，例如在办公室制作浓缩咖啡饮料或在工厂组装箱子，如附带的视频所示。对于箱子组装任务，图8（右）包含了任务在四个阶段的成功率细分：拾取箱板、组装箱子、标记箱子和将其放置在运输箱中的可用位置。与其他模型相比，$\pi _ { 0 . 6 } ^ { * }$在所有阶段都达到了更高的成功率。这些阶段的大多数失败是因为策略耗时过多。附带的视频展示了每个任务运行多个小时的时间推移。

![](images/9.jpg)  
Fig. 9: Improvement in throughput over multiple iterations. Both tasks improve significantly in throughput as we take more iterations of ReCAP, with box assembling first dropping and then improving significantly.

![](images/10.jpg)  
Fig. 10: Improvement in success rate over multiple iterations. The laundry task quickly reaches the maximum success rate (but continues to improve in throughput as shown in Figure 9, while box assembly continues to improve.

RECAP 在多个迭代中如何提高 $\pi _ { 0 . 6 } ^ { * }$ 的表现：我们接下来阐明通过 RECAP 训练如何提升策略，涉及多次数据收集和训练。我们研究了 T 恤和短裤折叠任务以及箱子组装任务。对于 T 恤折叠任务，仅使用通过自主评估（没有人类修正）收集的数据在两个迭代中进行策略改进，以评估我们的方法通过强化学习（RL）单独改善策略的能力。每个迭代中我们在四台机器人上收集 300 条轨迹。箱子组装任务则使用自主试验以及专家遥控操作干预，两者在每个迭代中分别进行 600 次自主试验和 360 次干预试验。我们在图 9 中绘制了迭代中的吞吐量，比较了两个 RECAP 迭代，分别标记为 $i \ = \ 1$ 和 $i \ = \ 2$。最后的迭代（标记为“我们的”）对应于上一部分中呈现的这些任务的最佳结果。我们还比较了初始数据收集策略，该策略使用离线 RL 预训练的 $\pi _ { 0 . 6 } ^ { * }$ 模型经过 SFT 微调。对于这两个任务，$\pi _ { 0 . 6 } ^ { * }$ 在两个迭代中都有所提升。在洗衣任务中我们可以看到持续的改进，总体吞吐量提高了 $50\%$。对于长时间的箱子组装任务，需要更多数据才能显著改进，但在第二次迭代后，我们看到吞吐量有 $2 \times$ 的提升。我们也展示了图 10 中迭代的成功率。对于洗衣任务，第一次迭代的成功率已经提升到 $90\%$ 以上，而第二次迭代主要是改善吞吐量。对于箱子组装任务，我们在两个迭代中看到成功率明显提高。尽管仍有一些失败（尤其是在最后将箱子放在堆栈上时），最终策略在折叠箱子和在分配的 600 秒时间限制内进行标记时，成功率达到约 $90\%$。

![](images/11.jpg)  
Fig. 11: Comparison of different policy extraction methods. RECAP applied to $\pi _ { 0 . 6 } ^ { * }$ to AWR and PPO.

![](images/12.jpg)  
Fig. 12: Failure mode removal. Here we apply RECAP on a variant of the laundry task with one item but a very strict success criteria. RECAP is particularly effective at removing failure modes that would be considered non successful under the strict criteria. Therefore, our method can also be used to alter a policy's behavior with relatively little data effectively.

3) RECAP中的优势条件策略提取方法与其他方法的比较：我们将第四节B部分的优势条件策略提取方法与文献中的其他方法（AWR和PPO）进行比较。我们使用T恤和短裤任务进行此比较。为了确保比较的控制性，我们使用与训练最终模型相同的数据进行这些比较。这为基线模型提供了一定的优势，因为它们可以访问在运行RECAP时收集的更好数据。结果如图11所示。虽然AWR和PPO都能取得合理的结果，但它们远远不及我们的方法，并且在改善离线$\mathrm { R L } +$ SFT $\pi _ { 0 . 6 } ^ { * }$模型方面遇到困难。对于PPO，我们不得不使用较小的信任区约束$( \eta = 0 . 0 1 )$来稳定这种离线策略设置下的训练，尽管这使得训练稳定，但该方法未能获得良好的性能。AWR可以实现合理的成功率，但导致策略更慢且吞吐量较低。

4) RECAP 是否能够在相对较少的数据中显著改变政策行为并消除失败模式？：虽然之前的实验集中在政策表现的整体端到端评估上，但我们也可以专注于特定的失败模式，以检查通过 RECAP 进行强化学习训练是否能消除政策中的特定错误。为了回答这个问题，我们采用了一个有严格成功标准的洗衣任务版本，该标准要求政策将T恤折叠时领口居中且朝上。每个回合都以一种特定的对抗条件初始化，即衬衫平放在桌子上，导致基线的离线 $\mathrm { R L } + \mathrm { S F T }$ 政策常常无法正确折叠。正如图12所示，在这种情况下应用 RECAP 进行两轮迭代（每轮收集600条轨迹）后，政策成功率达到了 $9 7 \%$，并且速度很快。因此，我们得出结论，RECAP 在消除特定失败模式方面是有效的，即使在完全通过强化学习而不需要任何干预数据或额外演示的情况下。

# VII. 讨论与未来工作

在机器人学习中，训练能够在现实世界任务中实现与人类相同的鲁棒性、速度和流畅性的方法面临重大挑战。本文讨论了如何通过结合DAgger风格的指导和强化学习，从经验中学习，以开始应对这一挑战。我们描述了RECAP，这是一种通过自主试验、奖励反馈和人类干预训练虚拟学习助手（VLA）的方法，并针对使用RECAP训练的模型$\pi _ { 0 . 6 } ^ { * }$在一系列现实任务上的表现进行了结果展示：制作浓缩咖啡饮品、折叠多样化的衣物以及组装箱子。RECAP的核心是一个适合VLA策略可扩展训练的强化学习方法，它利用优势条件对策略进行提取和价值函数。该强化学习方法的数据通过自主推演和人类干预的结合收集，干预纠正错误，同时细化自主数据上的行为细节。我们的实验表明，RECAP能够提高VLA的成功率和吞吐量，在一些更具挑战性的任务中吞吐量提高了两倍以上，同时失效率减少了约$2 \times$。

有几个方面可以改进 RECAP。首先，我们的系统并不是完全自主的：它依赖于人工标注和努力来提供奖励反馈、干预和回合重置。一些先前的工作探索了自动化这些组成部分的方法，VLAs 提供了通过使用高级策略来更自动化数据收集的新方式，例如推理如何重置场景。其次，我们的系统在探索的方式上相对简单：探索主要是贪婪的，依赖于策略中的随机性和人类干预来探索新解决方案。当初始模仿学习策略已经采取合理行动时，这种方式是合理的，但在更复杂的探索方法上仍然有很大的改进空间。最后，RECAP 执行迭代的“离线”更新（即收集一批数据，重新训练模型，并重复该过程），而不是实时更新策略和值函数的完全在线RL循环，因为数据的收集正进行中。我们出于便利做出了这个决定，但将我们的方法扩展到完全并发的在线RL框架是未来工作的一个有前景的方向。从更广泛的角度来看，用 RL 训练 VLAs 可能是实现满足现实世界用例的性能水平的最直接路径。与 VLAs 结合的 RL 面临许多挑战，从高容量模型的大规模 RL 训练的难度到样本复杂性、自主性和延迟反馈。虽然现有的 RL 框架针对小规模系统或“虚拟”领域（如 LLMs）可以提供良好的起点，但需要更多研究使 RL 成为 VLA 训练的实用工具。我们希望我们的工作能够在这个方向上代表一个重要的进展。

# 致谢

我们感谢我们的机器人操作员在数据收集、评估、后勤和视频录制方面的贡献，以及感谢我们的技术人员在机器人维护和修理方面的支持。完整的贡献声明请见附录A。

# REFERENCES

[1] Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press, 2018. 1   
[2] Sascha Lange, Thomas Gabel, and Martin A. Riedmiller. Batch reinforcement learning. In Marco A. Wiering and Martijn van Otterlo, editors, Reinforcement Learning, volume 12 of Adaptation, Learning, and Optimization, pages 4573. Springer, 2012. doi: 10.1007/978-3-642-27645-3\_2. 2, 4   
[3] Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020. 2, 4   
[4] Kevin Frans, Seohong Park, Pieter Abbeel, and Sergey Levine. Diffusion guidance is a controllable policy improvement operator. arXiv preprint, arXiv:2505.23458, 2025. 2, 3, 4, 5, 17   
[5] Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Robert Equi, Chelsea Finn, Niccolo Fusai, Manuel Y Galliker, et al. $\pi _ { 0 . 5 }$ : a vision-language-action model with openworld generalization. In 9th Annual Conference on Robot Learning, 2025. 2, 3, 5, 7, 8   
[6] Physical Intelligence Team. $\pi _ { 0 . 6 }$ model card. 2025. 2, 5, 6, 8   
[7] Stéphane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In AISTATS, pages 627635, 2011. 2, 7   
[8] Michael Laskey, Jonathan Lee, Roy Fox, Anca Dragan, and Ken Goldberg. Shiv: Reducing supervisor burden in dagger using support vectors for efficient learning from demonstrations in high dimensional state spaces. In Proceedings of the 2016 IEEE International Conference on Robotics and Automation (ICRA), pages 462469, 2016. doi: 10.1109/ICRA.2016.7487175. 2 [9] Michael Laskey, Jonathan Lee, Roy Fox, Anca D. Dragan, and Ken Goldberg. Dart: Noise injection for robust imitation learning. In Proceedings of the 34th International Conference on Machine Learning (ICML), volume 70 of Proceedings of Machine Learning Research, pages 19891998. PMLR, 2017.   
[10] Eric Jang, Alex Irpan, Mohi Khansari, Daniel Kappler, Frederik Ebert, Corey Lynch, Sergey Levine, and Chelsea Finn. Bc-z: Zero-shot task generalization with robotic imitation learning. In Conference on Robot Learning, pages 9911002. PMLR, 2022. 2   
[11] Zheyuan Hu, Robyn Wu, Naveen Enock, Jasmine Li, Riya Kadakia, Zackory Erickson, and Aviral Kumar. Rac: Robot learning for long-horizon tasks by scaling recovery and correction. arXiv preprint, arXiv:2509.07953, 2025. 2   
[12] Michael Kelly, Chelsea Sidrane, Katherine DriggsCampbell, and Mykel J Kochenderfer. Hg-dagger: Interactive imitation learning with human experts. In ICRA, 2019. 2   
[13] Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel. End-to-end training of deep visuomotor policies. The Journal of Machine Learning Research, 17(1):1334 1373, 2016. 2   
[14] Dmitry Kalashnikov, Alex Irpan, Peter Pastor, Julian Ibarz, Alexander Herzog, Eric Jang, Deirdre Quillen, Ethan Holly, Mrinal Kalakrishnan, Vincent Vanhoucke, l Q-pt: Scalable deep reinorcement learnin for vision-based robotic manipulation. arXiv preprint arXiv:1806.10293, 2018.   
[15] Ajay Mandlekar, Fabio Ramos, Byron Boots, Li FeiFei, Animesh Garg, and Dieter Fox. Iris: Implicit reinforcement without interaction at scale for learning control from offline robot manipulation data. ICRA, 2020.   
[16] Archit Sharma, M. Ahmed Ahmed Rehaan Ahmad, and Chelsea Finn. Self-improving robots: End-to-end autonomous visuomotor reinforcement learning. In Proceedings of the 7th Conference on Robot Learning (CoRL), volume 229, pages 32923308. PMLR, 2023.   
[17] Russell Mendonca, Shikhar Bahl, and Deepak Pathak. Alan: Autonomously exploring robotic agents in the real world. In Proceedings of the 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 30443050, 2023. doi: 10.1109/ICRA48891.2023. 10013321.   
[18] Russell Mendonca, Emmanuel Panov, Bernadette Bucher, Jiuguang Wang, and Deepak Pathak. Continuously improving mobile manipulation with autonomous realworld rl. In Proceedings of the 8th Conference on Robot Learning (CoRL), pages 52045219, 2024.   
[19] Jianlan Luo, Zheyuan Hu, Charles Xu, You Liang Tan, Jacob Berg, Archit Sharma, Stefan Schaal, Chelsea Finn, Abhishek Gupta, and Sergey Levine. Serl: A software suite for sample-efficient robotic reinforcement learning, 2024.   
[20] Lars Ankile, Zhenyu Jiang, Rocky Duan, Guanya Shi, Pieter Abbeel, and Anusha Nagabandi. Residual offpolicy rl for finetuning behavior cloning policies. arXiv preprint arXiv:2509.19301, 2025.   
[21] Thomas Lampe, Abbas Abdolmaleki, Sarah Bechtle, Sandy H. Huang, Jost Tobias Springenberg, Michael Bloesch, Oliver Groth, Roland Hafner, Tim Hertweck, Michael Neunert, Markus Wulfmeier, Jingwei Zhang, Francesco Nori, Nicolas Heess, and Martin Riedmiller. Mastering stacking of diverse shapes with large-scale iterative reinforcement learning on real robots. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 77727779, 2024. doi: 10.1109/ ICRA57147.2024.10610297. 2   
[22] Perry Dong, Suvir Mirchandani, Dorsa Sadigh, and Chelsea Finn. What matters for batch online reinforcement learning in robotics? arXiv preprint, arXiv:2505.08078, 2025. 2   
[23] Allen Z. Ren, Justin Lidard, Lars Lien Ankile, Anthony Simeonov, Pulkit Agrawal, Anirudha Majumdar, Benjamin Burchfiel, Hongkai Dai, and Max Simchowitz. Diffusion Policy Policy Optimization. In Procedings of the 2025 International Conference on Learning Representations (ICLR), 2025. 9, 17   
[24] Kun Lei, Huanyu Li, Dongjie Yu, Zhenyu Wei, Lingxiao Guo, Zhennan Jiang, Ziyu Wang, Shiyu Liang, and Huazhe Xu. Rl-100: Performant robotic manipulation with real-world reinforcement learning. arXiv preprint, arXiv:2510.14830, 2025. 2   
[25] Dmitry Kalashnkov, Jake Varley, Yevgen Chebotar, Ben Swanson, Rico Jonschkowski, Chelsea Finn, Sergey Levine, and Karol Hausman. Mt-opt: Continuous multitask robotic reinforcement learning at scale. arXiv, 2021. 2   
[26] Abhishek Gupta, Justin Yu, Tony Z. Zhao, Vikash Kumar, Aaron Rovinsky, Kelvin Xu, Thomas Devlin, and Sergey Levine. Reset-free reinforcement learning via multitask learning: Learning dexterous manipulation behaviors without human intervention. In Proceedings of the 2021 IEEE International Conference on Robotics and Automation (ICRA), pages 66646671, 2021. 2   
[27] Konstantinos Bousmalis, Giulia Vezzani, Dushyant Rao, Coline Devin, Alex X Lee, Maria Bauza, Todor Davchev, Yuxiang Zhou, Agrim Gupta, Akhil Raju, et al. Robocat: A self-improving foundation agent for robotic manipulation. arXiv preprint arXiv:2306.11706, 2023. 2   
[28] Aviral Kumar, Anikait Singh, Frederik Ebert, Mitsuhiko Nakamoto, Yanlai Yang, Chelsea Finn, and Sergey Levine. Pre-training for robots: Offline reinforcement learning enables learning new tasks from a handful of trials. In Proceedings of Robotics: Science and Systems (RSS), 2023. doi: 10.15607/RSS.2023.XIX.019.   
[29] Jingyun Yang, Max Sobol Mark, Brandon Vu, Archit Sharma, Jeannette Bohg, and Chelsea Finn. Robot fine-tuning made easy: Pre-training rewards and policies for autonomous real-world reinforcement learning. In Proceedings of the 2024 IEEE International Conference on Robotics and Automation (ICRA), 2024. doi: 10.1109/ICRA57147.2024.10610421. 2   
[30] Shuhan Tan, Kairan Dou, Yue Zhao, and Philipp Krähenbühl. Interactive post-training for vision-language-action models. arXiv preprint, arXiv:2505.17016, 2025. 2   
[31] Guanxing Lu, Wenkai Guo, Chubin Zhang, Yuheng Zhou, Haonan Jiang, Zifeng Gao, Yansong Tang, and Ziwei Wang. Vla-rl: Towards masterful and general robotic manipulation with scalable reinforcement learning. arXiv preprint, arXiv:2505.18719, 2025.   
[32] Jijia Liu, Feng Gao, Bingwen Wei, Xinlei Chen, Qingmin L  Wu, Cu, and u W Wa to vla generalization? an empirical study. arXiv preprint, arXiv:2505.19789, 2025.   
[33] Kang Chen, Zhihao Liu, Tonghe Zhang, Zhen Guo, Si Xu, Hao Lin, Hongzhi Zang, Quanlu Zhang, Zhaofei Yu, Guoliang Fan, Tiejun Huang, Yu Wang, and Chao Yu. $\pi _ { \tt r l }$ : Online rl fine-tuning for flowbased vision-language-action models. arXiv preprint, arXiv:2510.25889, 2025.   
[34] Haozhan Li, Yuxin Zuo, Jiale Yu, Yuhao Zhang, Zhaohui Yang, Kaiyan Zhang, Xuekai Zhu, Yuchen Zhang, Tianxing Chen, Ganqu Cui, Dehui Wang, Dingxiang Luo, Yuchen Fan, Youbang Sun, Jia Zeng, Jiangmiao Pang, Shanghang Zhang, Yu Wang, Yao Mu, Bowen Zhou, and Ning Ding. Simplevla-rl: Scaling vla training via reinforcement learning. arXiv preprint, arXiv:2509.09674, 2025. 2   
[35] Yanjiang Guo, Jianke Zhang, Xiaoyu Chen, Xiang Ji, Yen-Jen Wang Yucheg Hu, and Jiu Chen. Ipi vision-language-action model with online reinforcement learning. arXiv preprint, arXiv:2501.16664, 2025. 2   
[36] Wenli Xiao, Haotian Lin, Andy Peng, Haoru Xue, Tairan He, Yuqi Xie, Fengyuan Hu, Jimmy Wu, Zhengyi Luo, Linxi "Jim" Fan, Guanya Shi, and Yuke Zhu. Selfimproving vision-language-action models with data generation via residual rl, 2025. 2   
[37] Yuhui Chen, Shuai Tian, Shugao Liu, Yingting Zhou, Haoran Li, and Dongbin Zhao. Conrft: A reinforced fine-tuning method for vla models via consistency policy. arXiv preprint arXiv:2502.05450, 2025. 2   
[38] Max Sobol Mark, Tian Gao, Georgia Gabriela Sampaio, Mohan Kumar Srirama, Archit Sharma, Chelsea Finn, and Aviral Kumar. Policy-agnostic rl: Offline rl and online rl fine-tuning of any class and backbone. arXiv preprint, arXiv:2412.06685, 2024. 2   
[39] Mitsuhiko Nakamoto, Oier Mees, Aviral Kumar, and Sergey Levine. Steering your generalists: Improving robotic foundation models via value guidance. In Conference on Robot Learning, pages 49965013. PMLR, 2025.   
[401 Yang Zhang. Chenwei Wang. Ouvang I.u. Yuan 7hao. Yunfei Ge, Zhenglong Sun, Xiu Li, Chi Zhang, Chenjia Bai, and Xuelong Li. Align-then-steer: Adapting the vision-language action models through unified latent guidance. arXiv preprint arXiv:2509.02055, 2025. 2   
[41] Andrew Wagenmaker, Mitsuhiko Nakamoto, Yunchu Zhang, Seohong Park, Waleed Yagoub, Anusha Nagabandi, Abhishek Gupta, and Sergey Levine. Steering your diffusion policy with latent space reinforcement learning. In Proceedings of the 9th Conference on Robot Learning (CoRL), 2025. 2   
[42] Charles Xu, Qiyang Li, Jianlan Luo, and Sergey Levine. Rldg: Robotic generalist policy distillation via reinforcement learning. arXiv preprint arXiv:2412.09858, 2024. 2   
[43] Dongchi Huang, Zhirui Fang, Tianle Zhang, Yihang Li, Lin Zhao, and Chunhe Xia. Co-rft: Efficient finetuning of vision-language-action models through chunked offline reinforcement learning. arXiv preprint, arXiv:2508.02219, 2025. 3   
[44] Zijian Zhang, Kaiyuan Zheng, Zhaorun Chen, Joel Jang, Yi Li, Siwei Han, Chaoqi Wang, Mingyu Ding, Dieter Fox, and Huaxiu Yao. Grape: Generalizing robot policy via preference alignment. arXiv preprint, arXiv:2411.19309, 2024. 3   
[45] Shaopeng Zhai, Qi Zhang, Tianyi Zhang, Fuxian Huang, Haoran Zhang, Ming Zhou, Shengzhe Zhang, Litao Liu, Sixu Lin, and Jiangmiao Pang. A vision-languageaction-critic model for robotic real-world reinforcement learning. arXiv preprint, arXiv:2509.15937, 2025. 3   
[46] Seyed Kamyar Ghasemipour, Ayzaan Wahid, Jonathan Tompson, Pannag Sanketi, and Igor Mordatch. Selfimproving embodied foundation models. arXiv preprint, arXiv:2509.15155, 2025. 3   
[47] Jürgen Schmidhuber. Reinforcement learning upside down: Don't predict rewards — just map them to actions. arXiv preprint, arXiv:1912.02875, 2019. 3, 4   
[48] Aviral Kumar, Xue Bin Peng, and Sergey Levine. Reward-conditioned policies. CoRR, abs/1912.13465, 2019. 4   
[49] Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. Decision transformer: Reinforcement learning via sequence modeling. In Advances in Neural Information Processing Systems (NeurIPS) 34, 2021.   
[50] David Brandfonbrener, Alberto Bietti, Jacob Buckman, Romain Laroche, and Joan Bruna. When does returnconditioned supervised learning work for offline reinforcement learning? In Advances in Neural Information Processing Systems (NeurIPS) 35, 2022. 4   
[51] Scott Emmons, Benjamin Eysenbach, Ilya Kostrikov, and Sergey Levine. Rvs: What is essential for offline rl via supervised learning? In Proceedings of the 10th International Conference on Learning Representations (ICLR), 2022.   
5011:-1: 101. 21. Generalized decision transformer for offine hindsight information matching. In Proceedings of the 10th International Conference on Learning Representations (ICLR), 2022.   
[53] Taku Yamagata, Ahmed Khalil, and Raúl SantosRodríguez. Q-learning decision transformer: Leveraging dynamic programming for conditional sequence modelling in offline rl. In Proceedings of the 40th International Conference on Machine Learning (ICML), volume 202 of Proceedings of Machine Learning Research, pages 3898939007. PMLR, 2023.   
[54] Qinqing Zheng, Amy Zhang, and Aditya Grover. Online decision transformer. In Proceedings of the 39th International Conference on Machine Learning (ICML), volume 162 of Proceedings of Machine Learning Research, pages 2704227059. PMLR, 2022.   
[55] Jakub Grudzien Kuba, Pieter Abbeel, and Sergey Levine. Advantage-conditioned diffusion: Offline rl via generalization. 2023.   
[56] Yueh-Hua Wu, Xiaolong Wang, and Masashi Hamaya. Elastic decision transformer. In Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS), 2023. doi: 10.5555/3666122.3666936. 3   
[57] Lin Shao, Toki Migimatsu, Qiang Zhang, Kaiyuan Yang, and Jeannette Bohg. Concept2robot: Learning manipulation concepts from instructions and human demonstrations. In Proceedings of Robotics: Science & Systems (RSS), 2020. doi: 10.15607/RSS.2020.XVI.082. 3   
[58] Annie S. Chen, Suraj Nair, and Chelsea Finn. Learning generalizable robotic reward functions from "in-thewild" human videos. In Proceedings of Robotics: Science & Systems (RSS) 2021, 2021.   
[59] Suraj Nair, Eric Mitchell, Kevin Chen, Brian Ichter, Silvio Savarese, and Chelsea Finn. Learning languageconditioned robot behavior from offline data and crowdsourced annotation. In Proceedings of the 5th Conference on Robot Learning (CoRL), volume 164 of Proceedings of Machine Learning Research, pages 13031315. PMLR, 2022.   
[60] Sumedh A. Sontakke, Jesse Zhang, Sébastien M.R. Arnold, Karl Pertsch, Erdem B1yk, Dorsa Sadigh, Chelsea Finn, and Laurent Iti. Roboclip: One demonstration is enough to learn robot policies. In Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS), 2023.   
[61] Wenhao Yu, Nimrod Gileadi, Chuyuan Fu, Sean Kirmani, Kuang-Huei Lee, Montse Gonzalez Arenas, HaoTien Lewis Chiang, Tom Erez, Leonard Hasenclever, Jan Humplik, Brian Ichter, Ted Xiao, Peng Xu, Andy Zeng, Tingnan Zhang, Nicolas Heess, Dorsa Sadigh, Jie Tan, Yuval Tassa, and Fei Xia. Language to rewards for robotic skill synthesis. In Proceedings of the 7th Conference on Robot Learning (CoRL), volume 229 of Proceedings of Machine Learning Research, pages 374 404. PMLR, 2023.   
[62] Jiahui Zhang, Yusen Luo, Abrar Anwar, Sumedh Anand Sontakke, Joseph J. Lim, Jesse Thomason, Erdem B1y1k, and Jesse Zhang. Rewind: Language-guided rewards teach robot policies without new demonstrations. In Proceedings of the h Conference on Robot Learning (CoRL), 2025.   
[63] Minttu Alakuijala, Reginald McLean, Isaac Woungang, Nariman Farsad, Samuel Kaski, Pekka Marttinen, and Kai Yuan. Video-language critic: Transferable reward functions for language-conditioned robotics. Transactions on Machine Learning Research, 2025:122, 2025. 3   
[64] Yecheng Jason Ma, William Liang, Vaidehi Som, Vikash Kumar, Amy Zhang, Osbert Bastani, and Dinesh Jayaraman. Liv: Language-image representations and rewards for robotic control. In Proceedings of the 4Oth International Conference on Machine Learning (ICML), 2023. 3   
[65] Yecheng Jason Ma, Joey Hejna, Chuyuan Fu, Dhruv Shah, Jacky Liang, Zhuo Xu, Sean Kirmani, Peng Xu, Danny Driess, Ted Xiao, Osbert Bastani, Dinesh Jayaraman, Wenhao Yu, Tingnan Zhang, Dorsa Sadigh, and Fei Xia. Vision language models are in-context value learners. In Proceedings of the 13th International Conference on Learning Representations (ICLR), 2025. 3   
[66] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 3, 4,17   
[67] Abbas Abdolmaleki, Jost Tobias Springenberg, Yuval Tassa, Remi Munos, Nicolas Heess, and Martin Riedmiller. Maximum a posteriori policy optimisation. In International Conference on Learning Representations, 2018. 3   
[68] Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019. 4, 9   
[69] Peter Dayan and Geoffrey E. Hinton. Using expectationmaximization for reinforcement learning. Neural Computation, 9(2):271278, 1997. doi: 10.1162/neco.1997.9. 2.271.   
[70] Jan Peters, Katharina Mülling, and Yasemin Altün. Relative entropy policy search. In Proceedings of the Twenty-Fourth AAAI Conference on Artificial Intelligence, AAAI'10, page 16071612. AAAI Press, 2010. 3   
[71] Qing Wang, Jiechao Xiong, Lei Han, peng sun, Han Liu, and Tong Zhang. Exponentially weighted imitation learning for batched historical data. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 31, 2018. 3   
[72] Marc G Bellemare, Will Dabney, and Rémi Munos. A distributional perspective on reinforcement learning. In International conference on machine learning. pages   
449458. PMLR, 2017. 4 [73] Danny Driess, Jost Tobias Springenberg, Brian Ichter, Lili Yu, Adrian Li-Bell, Karl Pertsch, Allen Z Ren, Homer Walke, Quan Vuong, Lucy Xiaoyang Shi, et al. Knowledge insulating vision-language-action models: Train fast, run fast, generalize better. In Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS), 2025. 4, 6 [74] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. ICML, 2018. 4 [75] Ziyu Wang, Alexander Novikov, Konrad Zolna, Josh S Merel, Jost Tobias Springenberg, Scott E Reed, Bobak Shahriari, Noah Siegel, Caglar Gulcehre, Nicolas Heess, and Nando de Freitas. Critic regularized regression. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 77687778, 2020.   
4 [76] Ilya Kostrikov, Ashvin Nair, and Sergey Levine. Offine reinforcement learning with implicit q-learning. In International Conference on Learning Representations, 2022.   
4 [77] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. FAST: Efficient action tokenization for vision-language-action models. Robotics: Science and Systems, 2025. 6 [78] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, Gaël Liu, Francesco Visin, Kathleen Kenealy, Lucas Beyer, Xiaohai Zhai, Anton Tsitsulin, Robert BusaFekete, Alex Feng, Noveen Sachdeva, Benjamin Coleman, Yi Gao, Basil Mustafa, Iain Barr, Emilio Parisotto, David Tian, Matan Eyal, Colin Cherry, Jan-Thorsten Peter, Danila Sinopalnikov, Surya Bhupatiraju, Rishabh Agarwal, Mehran Kazemi, Dan Malkin, Ravin Kumar, David Vilar, Idan Brusilovsky, Jiaming Luo, Andreas Steiner, Abe Friesen, Abhanshu Sharma, Abheesht Sharma, Adi Mayrav Gilady, Adrian Goedeckemeyer, Alaa Saade, Alex Feng, Alexander Kolesnikov, Alexei Bendebury, Alvin Abdagic, Amit Vadi, András György, André Susano Pinto, Anil Das, Ankur Bapna, Antoine Miech, Antoine Yang, Antonia Paterson, Ashish Shenoy, Ayan Chakrabarti, Bilal Piot, Bo Wu, Bobak Shahriari, Bryce Petrini, Charlie Chen, Charline Le Lan, Christopher A. Choquette-Choo, CJ Carey, Cormac Brick, Daniel Deutsch, Danielle Eisenbud, Dee Cattle, Derek Cheng, Dimitris Paparas, Divyashree Shivakumar Sreepathihalli, Doug Reid, Dustin Tran, Dustin Zelle, Eric Noland. Erwin Huizenga. Eugene Kharitonov. Frederick Liu, Gagik Amirkhanyan, Glenn Cameron, Hadi Hashemi, Hanna Klimczak-Pluciáska, Harman Singh, Harsh Mehta, Harshal Tushar Lehri, Hussein Hazimeh, Ian Ballantyne, Idan Szpektor, Ivan Nardini, Jean PougetAbadie, Jetha Chan, Joe Stanton, John Wieting, Jonathan Lai, Jordi Orbay, Joseph Fernandez, Josh Newlan, Ju yeong Ji, Jyotinder Singh, Kat Black, Kathy Yu, Kevin Hui, Kiran Vodrahalli, Klaus Greff, Linhai Qiu, Marcella Valentine, Marina Coelho, Marvin Ritter, Matt Hoffman, Matthew Watson, Mayank Chaturvedi, Michael Moynihan, Min Ma, Nabila Babar, Natasha Noy, Nathan Byrd, Nick Roy, Nikola Momchev, Nilay Chauhan, Noveen Sachdeva, Oskar Bunyan, Pankil Botarda, Paul Caron, Paul Kishan Rubenstein, Phil Culliton, Philipp Schmid, Pier Giuseppe Sessa, Pingmei Xu, Piotr Stanczyk, Pouya Tafti, Rakesh Shivanna, Renjie Wu, Renke Pan, Reza Rokni, Rob Willoughby, Rohith Vallu, Ryan Mullins, Sammy Jerome, Sara Smoot, Sertan Girgin, Shariq Iqbal, Shashir Reddy, Shruti Sheth, Siim Pder, Sijal Bhatnagar, Sindhu Raghuram Panyam, Sivan Eiger, Susan Zhang, Tianqi Liu, Trevor Yacovone, Tyler Liechty, Uday Kalra, Utku Evci, Vedant Misra, Vincent Roseberry, Vlad Feinberg, Vlad Kolesnikov, Woohyun Han, Woosuk Kwon, Xi Chen, Yinlam Chow, Yuvein Zhu, Zichuan Wei, Zoltan Egyed, Victor Cotruta, Minh Giang, Phoebe Kirk, Anand Rao, Kat Black, Nabila Babar, Jessica Lo, Erica Moreira, Luiz Gustavo Martins, Omar Sanseviero, Lucas Gonzalez, Zach Gleicher, Tris Warkentin, Vahab Mirrokni, Evan Senter, Eli Collins, Joelle Barral, Zoubin Ghahramani, Raia Hadsell, Yossi Matias, D. Sculley, Slav Petrov, Noah Fiedel, Noam Shazeer, Oriol Vinyals, Jeff Dean, Demis Hassabis, Koray Kavukcuoglu, Clement Farabet, Elena Buchatskaya, Jean-Baptiste Alayrac, Rohan Anil, Dmitry, Lepikhin, Sebastian Borgeaud, Olivier Bachem, Armand Joulin, Alek Andreev, Cassidy Hardin, Robert Dadashi, and Léonard Hussenot. Gemma 3 technical report, 2025. 6   
[79] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 6, 16   
[80] Diederik Kingma and Ruiqi Gao. Understanding diffusion objectives as the elbo with simple data augmentation. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 65484 65516, 2023. 6, 16   
[81] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. $\pi _ { 0 }$ : A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024. 8   
[821 David McAllister Songwei Ge Brent Yi Chung Min Kim, Ethan Weber, Hongsuk Choi, Haiwen Feng, and Angjoo Kanazawa. Flow matching policy gradients, 2025. 9, 16, 17   
[83] Zhengpeng Xie, Qiang Zhang, Fan Yang, Marco Hutter, and Renjing Xu. Simple policy optimization. In Fortysecond International Conference on Machine Learning (ICML), 2025. 9, 17   
[84] Henry Zhu, Justin Yu, Abhishek Gupta, Dhruv Shah, Kristian Hartikainen, Avi Singh, Vikash Kumar, and Sergey Levine. The ingredients of real-world robotic reinforcement learning. arXiv preprint arXiv:2004.12570, 2020. 11   
[85] Archit Sharma, Kelvin Xu, Nikhil Sardana, Abhishek Gupta, Karol Hausman, Sergey Levine, and Chelsea Finn. Autonomous reinforcement learning: Formalism and benchmarking. arXiv preprint arXiv:2112.09605, 2021. 11   
[86] Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, Liyiming Ke, Karl Pertsch, Quan Vuong, James Tanner, Anna Walling, Haohuan Wang, Niccolo Fusai, et al. Hi robot: Open-ended instruction following with hierarchical vision-language-action models. arXiv preprint arXiv:2502.19417, 2025. 11

# APPENDIX

# A. Contributions

Data collection and operations. Michael Equi, Chelsea Finn, Lachy Groom, Hunter Hancock, Karol Hausman, Rowan Jen, Liyiming Ke, Marinda Lamb, Vishnu Mano, Suraj Nair, Charvi Sharma, Laura Smith, Will Stoeckle, Anna Walling, Blake Williams.   
Annotation and supplemental data. Chelsea Finn, Catherine Glossop, Hunter Hancock, Brian Ichter, Rowan Jen, Liyiming Ke, Chandra Kuchi, Karl Pertsch, Laura Smith, Will Stoeckle, Quan Vuong, Anna Walling.   
Policy training and research. Ashwin Balakrishna, Kevin Black, Danny Driess, Michael Equi, Yunhao Fang, Chelsea Finn, Catherine Glossop, Karol Hausman, Gashon Hussein, Brian Ichter, Liyiming Ke, Sergey Levine, Yao Lu, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, Alex Swerdlow, Marcel Torne, Quan Vuong, Lili Yu, Zhiyuan Zhou.   
Policy infrastructure. Kevin Black, Karan Dhabalia, Danny Driess, Michael Equi, Liyiming Ke, Adrian Li-Bell, Suraj Nair, Allen Z. Ren, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, Alex Swerdlow, Haohuan Wang, Ury Zhilinsky, Zhiyuan Zhou.   
Robot hardware. Ali Amin, Raichelle Aniceto, Grace Connors, Adnan Esmail, Thomas Godden, Ivan Goryachev, Tim Jones, Ben Katz, Devin LeBlanc, Mohith Mothukuri, Sukwon Yoo.   
Robot infrastructure. Ken Conley, James Darpinian, Jared DiCarlo, Karol Hausman, Szymon Jakubczak, James Tanner. Writing and illustration. Kevin Black, Danny Driess, Michael Equi, Chelsea Finn, Hunter Hancock, Karol Hausman,

Brian Ichter, Liyiming Ke, Sergey Levine, Suraj Nair, Allen Z. Ren, Laura Smith, Jost Tobias Springenberg, Zhiyuan Zhou

# B. Additional Value Function Visualization

Figure 13 shows additional visualizations of our trained value function on five different tasks, including tasks on which we evaluate our policies (espresso making, box assembly) and also broader tasks (hang towel, attach hook). The parts with the most prominent changes are highlighted: red corresponds to where value function drops, green corresponds to where value function increases, and yellow corresponds to oscillating values. Images show the corresponding frames and description of the episode.

# C. Computing the log-likelihood for policy improvement

To derive the log-likelihood from Equation (4) we can first observe that we can decompose the full model likelihood into autoregressive and diffusion terms

$$
\begin{array} { r l } & { \pi _ { \boldsymbol { \theta } } ( \mathbf { a } _ { t : t + H } , \mathbf { a } _ { t : t + H } ^ { \ell } , \widehat { \ell } | I _ { t } , \mathbf { o } _ { t } , \ell ) = } \\ & { \pi _ { \boldsymbol { \theta } } ( \mathbf { a } _ { t : t + H } | I _ { t } , \mathbf { o } _ { t } , \ell , \widehat { \ell } ) \pi _ { \boldsymbol { \theta } } ( \mathbf { a } _ { t : t + H } ^ { \ell } | I _ { t } , \mathbf { o } _ { t } , \ell , \widehat { \ell } ) \pi _ { \boldsymbol { \theta } } ( \widehat { \ell } | I _ { t } , \mathbf { o } _ { t } , \ell ) , } \end{array}
$$

where the first term is modeled with flow matching, the second term is the autoregressive likelihood of the discretized actions $\mathbf { a } _ { t : t + H } ^ { \ell }$ mated in the usual way, using the cross-entropy loss evaluated on ground truth tokens. For the continuous likelihood over $\mathbf { a } _ { t : t + H }$ , a closed form likelihood is not available [79]. We can, however follow prior work [82], and consider the one-step diffusion process as a Gaussian distribution with likelihood

$$
\begin{array} { r l } & { \log \pi _ { \boldsymbol { \theta } } \big ( \mathbf { a } _ { t : t + H } \big \vert \mathbf { a } _ { 1 : H } ^ { \eta , \omega } , I _ { t } , \mathbf { o } _ { t } , \boldsymbol { \ell } , \hat { \ell } \big ) = } \\ & { \qquad \log \mathcal { N } \Big ( \omega - f _ { \boldsymbol { \theta } } \big ( \mathbf { a } _ { 1 : H } ^ { \eta , \omega } , I _ { t } , \mathbf { o } _ { t } , \boldsymbol { \ell } , \hat { \ell } \big ) , \mathbf { I } \Big ) , } \end{array}
$$

wi ith $\mathbf { a } _ { t : t + H } ^ { \eta , \omega } = \eta \mathbf { a } _ { t : t + H } + ( 1 - \eta ) \omega$ and $\boldsymbol \omega = \mathcal { N } ( 0 , { \bf I } )$ Fromr following [80, 82] (effectively marginalizing over $\eta$ and $\omega$ ) which yields

$$
\begin{array} { r l r } {  { \log \pi _ { \theta } \big ( \mathbf { a } _ { t : t + H } \big | I _ { t } , \mathbf { o } _ { t } , \ell , \widehat { \ell } \big ) \geq } } \\ & { } & { \frac { 1 } { 2 } \mathbb { E } _ { \eta , \omega } \Big [ - w ( \eta ) \| \omega - \mathbf { a } _ { 1 : H } - f _ { \theta } \big ( \mathbf { a } _ { 1 : H } ^ { \eta , \omega } , I _ { t } , \mathbf { o } _ { t } , \ell , \widehat { \ell } \big ) \| ^ { 2 } \Big ] + c , } \\ & { } & { \quad \mathrm { o s } } \end{array}
$$

where $w ( \eta ) = e ^ { - \eta / 2 }$ is a noise dependent weighting term, and c is a constant independent of $f _ { \theta }$ . For the derivation, see [80], which also derives the relationship between flow matching and diffusion in Appendix D.3 for this choice of weighting term. Finally putting the lower bound together with the autoregressive likelihood for the discretized action part of the text output $\hat { \ell }$ , and subsuming the weighting terms in $\alpha$ gives

$$
\begin{array} { r l } & { \log \pi _ { \boldsymbol { \theta } } \big ( \mathbf { a } _ { t : t + H } , \mathbf { a } _ { t : t + H } ^ { \boldsymbol { \ell } } \vert I _ { t } , \mathbf { o } _ { t } , \boldsymbol { \ell } , \hat { \ell } \big ) \geq } \\ & { \mathbb { E } _ { \eta , \omega } \Big [ \log p _ { \boldsymbol { \theta } } \big ( \mathbf { a } _ { t : t + H } ^ { \boldsymbol { \ell } } \vert I _ { t } , \mathbf { o } _ { t } , \boldsymbol { \ell } , \hat { \ell } \big ) } \\ & { \qquad - \alpha _ { \eta } \left. \omega - \mathbf { a } _ { 1 : H } - f _ { \boldsymbol { \theta } } \big ( \mathbf { a } _ { 1 : H } ^ { \eta , \omega } , I _ { t } , \mathbf { o } _ { t } , \boldsymbol { \ell } , \hat { \ell } \big ) \right. ^ { 2 } \Big ] , } \end{array}
$$

which is the bound given in the main part of the paper.

![](images/13.jpg)  
Fig. 13: Additional visualization of value function on five different tasks. Red parts highlight places where value drops, green parts highlight places where value increases, and yellow parts highlight oscillating value regions. Images show the corresponding frames and descriptions of the episode.

can be written as

$$
\begin{array} { r l } & { \log \pi _ { \boldsymbol { \theta } } \big ( \mathbf { a } _ { t : t + H } , \mathbf { a } _ { t : t + H } ^ { \ell } \big | \mathbf { o } _ { t } , \ell , \hat { \ell } \big ) \geq } \\ & { \quad \mathbb { E } _ { \eta , \omega } \bigg [ \log p _ { \boldsymbol { \theta } } \big ( \mathbf { a } _ { t : t + H } ^ { \ell } \big | \mathbf { o } _ { t } , \ell , \hat { \ell } \big ) } \\ & { \qquad - \alpha _ { \eta } \left\| \omega - \mathbf { a } _ { 1 : H } - f _ { \boldsymbol { \theta } } \big ( \mathbf { a } _ { 1 : H } ^ { \eta , \omega } , \mathbf { o } _ { t } , \ell , \hat { \ell } \big ) \right\| ^ { 2 } \bigg ] , } \end{array}
$$

which is analogous to the diffusion likelihood bound used in FPO [82]. And we combine it with a PPO style loss separated into diffusion and autoregressive terms. In preliminary experiments we found that for our setting it was difficult to enforce a trust region constraint on the action expert (which models actions with an unbounded diffusion head) when using the standard PPO clipping objective. Presumably, this is partially due to the "offine" nature of our algorithm setting, where we cannot afford to collect new data from real robots every few gradient steps. To stabilize training we found using an alternative definition of the PPO constraint following SPO [83] to be effective. The resulting loss is given as:

$$
\begin{array} { r l } & { \quad \mathcal { L } _ { S r f o } + _ { C o } v L A ( \theta ) = } \\ & { \quad \Bigg \{ \frac { \pi _ { \theta } ( a _ { \ell } \varepsilon \hat { \varepsilon } ( \hat { \varepsilon } | \mathbf { o } _ { t } , \ell ) ) } { \pi _ { \mathrm { e r f } } ( a _ { \ell } \varepsilon \hat { \varepsilon } / \mathbf { { l } _ { 0 } } , \ell ) } A ^ { \pi _ { \theta } } ( o _ { t } , a _ { t } , \ell ) } \\ & { \quad - \frac { \left. \ A ^ { \pi _ { \theta } } \left( o _ { t } , a _ { t } , \ell \right) \right. } { 2 \varepsilon _ { \mathrm { a r } } } \Bigg [ \frac { \pi _ { \theta } ( a _ { \ell } \varepsilon \hat { \varepsilon } ( \hat { \varepsilon } | \mathbf { o } _ { t } , \ell ) } { \pi _ { \mathrm { e r f } } ( a _ { \ell } \varepsilon \hat { \varepsilon } / \mathbf { { l } _ { 0 } } , \ell ) } - 1 \Bigg ] \Bigg \} } \\ & { \quad + \alpha \Bigg \{ \frac { \pi _ { \theta } ( \mathbf { a } _ { t + t + H } | \mathbf { o } _ { t } , \ell ) } { \pi _ { \mathrm { e r f } } ( \mathbf { a } _ { t ; t + H } | \mathbf { o } _ { t } , \ell ) } A ^ { \pi _ { \theta } } ( o _ { t } , a _ { t } , \ell ) } \\ & { \quad \quad - \frac { \left. \ A ^ { \pi _ { \theta } } \left( o _ { t } , a _ { t } , \ell \right) \right. } { 2 \varepsilon _ { \mathrm { t o w } } } \Bigg [ \frac { \pi _ { \theta } ( \mathbf { a } _ { t + t + H } | \mathbf { o } _ { t } , \ell ) } { \pi _ { \mathrm { e r f } } ( \mathbf { a } _ { t ; t + H } | \mathbf { o } _ { t } , \ell ) } - 1 \Bigg ] \Bigg \} , } \end{array}
$$

where $\alpha$ is a trade-off parameter and $\epsilon _ { \mathrm { a r } } , ~ \epsilon _ { \mathrm { f l o w } }$ are trust-region parameters for autoregressive and flow-matching model parts respectively. We use this variant to perform training on eval data starting from the $\pi _ { 0 . 6 }$ checkpoint.

# E. Using CFG for test-time policy improvement with $\beta > 1$

After training we can choose to further sharpen the policy used for evaluation by setting $\beta > 1$ in Eq. (2). As shown in prior work [4] we can recover this sharpened policy without additional training since it is implicitly defined by the learned policies $\pi _ { \boldsymbol { \theta } } ( \mathbf { a } _ { t : t + H } | I _ { t } , \mathbf { o } _ { t } , \ell )$ and $\pi _ { \boldsymbol { \theta } } \big ( \mathbf { a } _ { t : t + H } \big | \mathbf { o } _ { t } , \boldsymbol { \ell } \big )$ . Specifically, after training we can form the approximation

# D. PPO implementation

We implement a variant of PPO [66] related to DPPO and FPO [23, 82] and use it as an additional baseline. To allow for training both the autoregressive part of the model as well as the diffusion based action expert in a compute effective manner we calculate likelihoods based on the single step diffusion objective alone.

In particular, we use a likelihood bound analogous to Eq. (9) (previous section) but without the improvement indicator. Decomposing into autoregressive and flow-matching terms this

$$
\hat { \pi } ( \mathbf { a } _ { t : t + H } | \mathbf { o } _ { t } , \ell ) \propto \pi _ { \mathrm { r e f } } ( \mathbf { a } _ { t : t + H } | \mathbf { o } _ { t } , \ell ) \left( \frac { \pi _ { \mathrm { r e f } } ( \mathbf { a } _ { t : t + H } | I _ { t } , \mathbf { o } _ { t } , \ell ) } { \pi _ { \mathrm { r e f } } ( \mathbf { a } _ { t : t + H } | \mathbf { o } _ { t } , \ell ) } \right) ^ { \beta } .
$$

One can now realize that the diffusion model effectively learns the gradient of the likelihoods, i.e. it represents $\nabla _ { \mathbf { a } } \log \pi _ { \boldsymbol { \theta } } \big ( \mathbf { a } _ { t : t + H } \big | I _ { t } , \mathbf { o } _ { t } , \boldsymbol { \ell } \big )$ and $\nabla _ { \mathbf { a } } \log \pi _ { \boldsymbol { \theta } } ( \mathbf { a } _ { t : t + H } | \mathbf { o } _ { t } , \boldsymbol { \ell } )$ respectively. From this, following Frans et al. [4], we can see that if we run flow-matching inference following the gradient

$$
\begin{array} { r l } & { \nabla _ { \mathbf { a } } \log \pi _ { \boldsymbol { \theta } } ( \mathbf { a } _ { t : t + H } | \mathbf { o } _ { t } , \boldsymbol { \ell } ) + } \\ & { \quad \beta ( \nabla _ { \mathbf { a } } \log \pi _ { \boldsymbol { \theta } } ( \mathbf { a } _ { t : t + H } | I _ { t } , \mathbf { o } _ { t } , \boldsymbol { \ell } ) - \nabla _ { \mathbf { a } } \log \pi _ { \boldsymbol { \theta } } ( \mathbf { a } _ { t : t + H } | \mathbf { o } _ { t } , \boldsymbol { \ell } ) ) , } \end{array}
$$

we are effectively sampling from the desired attenuated distribution. We note that, as mentioned in the main paper, the parameter $\beta$ is loosely connected to the advantage threshold $\epsilon _ { \ell }$ that we introduce during training (in the sense that both sharpen the distribution, one at inference and one at training time). We find that sharpening the distribution after training with high settings for $\beta$ can lead to pushing the action distribution towards the boundaries of its learned support (which can lead to overly aggressive motions) and thus primarily rely on $\epsilon _ { \ell }$ for obtaining a good conditioned policy directly after training and combine it with moderate settings (e.g. $\beta \in [ 1 . 5 , 2 . 5 ] ,$ where useful.

# $F .$ Additional algorithm details

We describe details for setting the task specific parameters used in Algorithm 1.

Advantage Estimation: During post-training, we estimate the advantage function using ${ \cal A } ^ { \overline { { \pi } } } ( { \bf \bar { o } } _ { t } , { \bf a } _ { t } ) = \sum _ { t ^ { \prime } = t } ^ { t + N - 1 } r _ { t } ^ { \prime } +$ $V ^ { \pi } ( \mathbf { o } _ { t + N } ) - V ^ { \pi } ( \mathbf { o } _ { t } )$ ,where $\mathbf { o } _ { t + N }$ is an observation sampled from $N$ steps ahead from the same trajectory. We use $N = 5 0$ lookahead to calculate this advantage. During pre-training, we calculate the advantage estimate as $\begin{array} { r } { A ^ { \pi } ( \mathbf { o } _ { t } , \mathbf { \bar { a } } _ { t } ) = \sum _ { t ^ { \prime } = 0 } ^ { T } r _ { t } ^ { \prime } - } \end{array}$ $V ^ { \pi } ( \mathbf { o } _ { t } )$ , setting $N = T$ for each episode, which is a higher variance estimate of the advantage. We use this advantage calculation since it allows us to calculate the advantage values on-the-fly during pre-training using a single inference call to the value function. We find empirically that this advantage estimate works well when the policy is trained on large amounts of data from diverse tasks during pre-training.

Advantage conditioning dropout: During training, we randomly drop out the conditioning on the advantage indicator $30 \%$ of the time. We employ this dropout so that we can directly sample directly from either the conditional or unconditional policy during inference time and use CFG for test-time policy improvement (see Section E for details); and it effectively replaces the loss multiplier $\alpha$ .

Advantage threshold: The per task advantage threshold $\epsilon _ { \ell }$ is set as follows. During pre-training we select the threshold for each task such that approximately $3 0 \%$ of the demonstration data has positive advantage (as calculatedd on a random sample of 10k datapoints). During fine-tuning we generally set the threshold such that approximately $4 0 \%$ of the evaluation rollouts in each iteration have positive advantage. For the Tshirt and shorts laundry folding task (in which training on high-quality demonstration data yields slow policies but with high success rate) we increase the threshold such that only approximately $1 0 \%$ of the data has positive advantage.

Dataset composition: We use the dataset aggregation strategy described in Algorithm 1 for all tasks. However each of our task has distinct nature: the episode lengths vary, the performances of Iteration 0 model on each task are different, and one task (Assemble Box) is performed offsite in a deployment scenario. Therefore, we have different amount of demonstration data to begin with and collect different amounts of experience data for iterative improvement. For laundry (Tshirt and shorts), we use autonomous evaluation data only without expert corrections. As we push model performance to closely resemble the expert data collector in terms of speed, it becomes hard to provide corrections. For this task, We collect 300 episodes across 4 robot stations for reporting eval performance. For the diverse laundry folding task we collect 450 evaluation episodes and 287 correction episodes. For the failure mode removal ablation we collect both autonomous and policy correction data. In total we collect $\sim 1 0 0 0$ autonomous and $2 8 0 + 3 7 8$ correction episodes spread over 3 robots. For box assembly we collect data in the deployment scenario directly, collecting 600 demonstrations and 360 correction episodes in each iteration, using 3 robots in total. For cafe we perform a single iteration and collect 429 correction episodes as well as 414 autonomous episodes.