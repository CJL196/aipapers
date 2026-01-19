# 解缠结世界模型：从干扰视频中学习转移语义知识以增强学习

Qi Wang1,2,3\* Zhipeng Zhang4,5\* Baao Xie2,3\* Xin Jin2,3 Yunbo Wang1 Shiyu Wang5,6 Liaomo Zheng5,6 Xiaokang Yang1 Wenjun Zeng2,3 1 上海交通大学人工智能教育部重点实验室，人工智能研究所 2 中国东部技术学院数字双胞胎研究所，中国宁波 3 中国宁波空间智能与数字衍生重点实验室 4 中国科学院大学 5 中国科学院沈阳计算技术研究所 6 沈阳中国科学院新材料技术有限公司 https://qiwang067.github.io/diswm

# 摘要

在实际场景中训练视觉强化学习（RL）面临显著挑战，即RL智能体在存在变化的环境中样本效率较低。尽管各种方法尝试通过解耦表示学习来缓解这个问题，但这些方法通常都是从零开始学习，没有世界的先验知识。本文则试图通过离线到在线的潜在蒸馏和灵活的解耦约束，从干扰视频中学习和理解潜在的语义变化。为了实现有效的跨域语义知识转移，我们引入了一种可解释的基于模型的RL框架，称为解耦世界模型（DisWM）。具体而言，我们使用解耦正则化对无动作的视频预测模型进行离线预训练，以从干扰视频中提取语义知识。然后，通过潜在蒸馏将预训练模型的解耦能力转移到世界模型中。为了在在线环境中进行微调，我们利用预训练模型的知识，并为世界模型引入了解耦约束。在适应阶段，从在线环境交互中融入的动作和奖励丰富了数据的多样性，进而增强了解耦表示学习。实验结果验证了我们方法在各种基准测试中的优越性。

![](images/1.jpg)  
Figure 1. Overview of our proposed framework. The key idea is to leverage distracting videos for semantic knowledge transfer, enabling the downstream agent to improve sample efficiency on unseen tasks.

# 1. 引言

视觉强化学习（VRL）为在复杂环境中训练智能体提供了一种有前景的方法。然而，由于环境的复杂性、波动性和视觉干扰，VRL在实际场景中常常面临性能下降的问题。即使是微小的环境变化也可能导致显著的像素级位移，使得训练后的VRL策略失效或次优。例如，光照条件的轻微变化可能会影响物体的外观（例如颜色、阴影或其他视觉属性）。因此，增强模型的可解释性，使其能够感知、学习和理解语义环境变化，显得至关重要。

解耦表示学习（DRL）提供了一种有前景的方法，用以解决深度学习算法固有的“黑箱”特性所带来的可解释性挑战。从根本上讲，DRL 方法模仿了生物智能的认知过程，其中理解世界是通过将观察结果分解为不同且独立的因素来实现的。在这种形式下，当某个变化因素（例如，颜色）发生变化时，解耦表示中的只有一小部分特征会受到影响，使得智能体能够快速恢复性能。多项研究探索了 DRL 算法在 VRL 领域的集成。例如，Higgins 等人训练了一个 $\beta$ VAE 以离线方式获取用于强化学习的解耦表示。TED 采用了一种自监督辅助任务来学习用于强化学习的时间解耦表示。此外，Dunion 等人引入条件互信息以实现与相关数据的解耦表示。然而，现有方法通常从零开始学习表示，缺乏对世界的任何先验知识。这些方法通常需要与环境进行广泛的交互以获得期望的行为。

Table 1. MuJoCo (downstream domain) vs. DMC (accessible distracting videos).   

<table><tr><td></td><td>Video: DMC</td><td>Target: MuJoCo</td><td>Similarity / Difference</td></tr><tr><td>Task</td><td>Reacher Easy</td><td>Pusher</td><td>Relevant robotic control tasks</td></tr><tr><td>Dynamics</td><td>Two-link planar</td><td>Multi-jointed robot arm</td><td>Different</td></tr><tr><td>Action space</td><td>Box(-1, 1, (2,), float32)</td><td>Box(-2, 2, (7,), float32)</td><td>Different</td></tr><tr><td>Reward range</td><td>[0, 1]</td><td>[-4.49, 0]</td><td>Different</td></tr></table>

为应对这一挑战，我们提出了一种基于模型的可解释VRL框架，称为解缠世界模型（Disentangled World Models, DisWM），该框架利用从干扰视频中提取的先验知识，通过潜在蒸馏促进对未见下游任务的学习。需要特别指出的是，干扰视频指的是具有视觉干扰的视频，这对于学习解缠表示是有益的。具体而言，如图1所示，我们的框架由两个阶段组成：首先，我们对DRL编码器进行预训练，以从干扰视频中学习解缠的潜在表示。通过这样做，预训练的DRL编码器在表示解缠方面变得“知识渊博”。随后，我们微调一个正交设计的世界模型，该模型在解缠和蒸馏的双重约束下运作，利用通过离线到在线潜在蒸馏从预训练模型转移的语义知识。解缠世界模型适应的另一个好处是，从与环境的在线交互中引入动作和奖励，丰富了视觉观察的多样性，进而加强了解缠表示学习的过程。值得一提的是，作为一个跨域框架，DisWM不要求预训练视频来自与下游任务相同的领域。实验结果表明，我们提出的方法在提升我们修改后的DeepMind Control和MuJoCo Pusher中VRL智能体的样本效率方面是有效的。本工作的贡献可以总结如下： • 我们将学习可解释VRL智能体的问题框架定义为一个领域迁移学习问题。关键思想是从干扰视频中提取语义知识，并将这种解缠能力转移到下游控制任务。 • 我们提出了DisWM，这是一种遵循预训练-微调范式的方法，利用干扰视频，并结合了离线到在线潜在蒸馏和灵活的解缠约束等具体技术。

# 2. 问题设定

我们将视觉强化学习构建为部分可观察的马尔可夫决策过程（POMDP），并使用DMC和MuJoCo Pusher作为测试平台。具体来说，我们集中于可以访问没有动作和奖励的视频场景，从而实现世界知识的迁移。我们的目标是通过从视频中迁移共享的世界知识，最大化目标POMDP $\langle \mathcal { O } , \mathcal { A } , \mathcal { T } , \mathcal { R } , \gamma \rangle$ 的累积奖励。这些符号分别对应于视觉观察空间、动作空间、状态转移概率、奖励函数和折扣因子。例如，在其中一个跨域实验中，我们将MuJoCo作为下游领域，并使用从DMC收集的帧作为干扰视频。表1强调了两个领域在视觉外观、物理动态、动作空间和奖励函数方面的差异。

# 3. 方法

# 3.1. DisWM 概述

在本节中，我们介绍DisWM的细节，该方法涉及三个主要阶段（见图2）：a) 解耦表示预训练：从干扰视频中预训练一个基于深度强化学习的视频预测模型，以提取解耦特征。 b) 离线到在线潜在蒸馏：通过跨领域潜在蒸馏，将预训练模型中的语义知识转移到世界模型中。 c) 解耦世界模型适应：通过结合动作和奖励信息，为下游智能体进行微调，并加入解耦约束。

![](images/2.jpg)  
trained on distracting videos offine for the wel-disentangled latent varable $\mathbf { z } _ { \mathrm { d i s e n } }$ , which extracts semantic knowledge from the visual observations. The disentangled capability of $\mathbf { z } _ { \mathrm { d i s e n } }$ is then transferred to the world model through latent distillation. (b) The action-conditioned

# 3.2. 解缠表示预训练

为了提取可以迁移到下游世界模型的良好解耦表征，我们首先在干扰视频上训练一个视频预测模型，而不结合动作信息（算法1的第4-8行）。该模型由三个关键组件组成：(i) 后验学习器，使用$\beta$ -VAE编码器将观测$o _ { t }$编码为潜在状态$z _ { t }$，它作为典型的深度强化学习框架从观测中提取潜在特征$\mathbf { z } _ { t }$，(ii) 先验模块，根据历史状态预测未来潜在状态，而不直接依赖当前观测$o _ { t }$，以及(iii) 基于$\beta$ -VAE的解码器，从潜在状态$z _ { t }$重构$\hat { o } _ { t }$。具体而言，模型可以 formulized 如下：

$\beta$ VAE 编码器：$\begin{array} { r l } & { \mathbf { z } _ { t } = e _ { \phi ^ { \prime } } ( o _ { t } ) } \\ & { z _ { t } \sim q _ { \phi ^ { \prime } } ( z _ { t } \mid z _ { t - 1 } , \mathbf { z } _ { t } ) } \\ & { \hat { z } _ { t } \sim p _ { \phi ^ { \prime } } ( \hat { z } _ { t } \mid z _ { t - 1 } ) } \\ & { \hat { o } _ { t } \sim p _ { \phi ^ { \prime } } ( \hat { o } _ { t } \mid z _ { t } ) } \end{array}$ 后验状态： 先验状态： 重构：各向同性单位高斯分布：$p ( \mathbf { z } ) = \mathcal { N } ( \mathbf { 0 } , I )$。其中 $\phi ^ { \prime }$ 表示模型的参数。基于 $\beta$ VAE 的视频预测模型被训练以最小化以下损失函数：

$$
\begin{array} { r l } & { \mathcal { L } ( \phi ^ { \prime } ) = \mathbb { E } _ { q _ { \phi ^ { \prime } } } \Big [ \displaystyle \sum _ { t = 1 } ^ { T } \underbrace { - \ln p _ { \phi ^ { \prime } } ( o _ { t } \mid z _ { t } ) } _ { \mathrm { i m a g e r e c o n s t u c t i o n } } } \\ & { \qquad + \underbrace { \beta _ { 1 } \mathrm { K L } [ q _ { \phi ^ { \prime } } ( \boldsymbol { z } _ { t } \mid \boldsymbol { z } _ { t - 1 } , \boldsymbol { o } _ { t } ) \| p _ { \phi ^ { \prime } } ( \hat { \boldsymbol { z } } _ { t } \mid \boldsymbol { z } _ { t - 1 } ) ] } _ { \mathrm { a c t i o n . f r e K L l o s s } } } \\ & { \qquad + \beta _ { 2 } \mathrm { K L } [ q _ { \phi ^ { \prime } } ( \mathbf { z } _ { t } \mid \boldsymbol { o } _ { t } ) \| p ( \mathbf { z } _ { t } ) \| ] . } \end{array}
$$

变分后验分布 $q _ { \phi ^ { \prime } } ( \mathbf { z } _ { t } \mid \mathbf { \theta } _ { o _ { t } } )$ 被鼓励接近标准多元高斯分布 $\mathcal { N } ( \mathbf { 0 } , I )$，以增强潜在空间的正交性和解耦性。解耦损失项的重要性由 $\beta _ { 2 }$ 控制。

# 3.3. 离线到在线潜在蒸馏

在使用干扰视频进行离线预训练后，模型通过整合动作和奖励在线进行微调，以适应下游任务（算法1第13行）。转移解耦特征的一个简单方法是用从预训练视频预测模型获得的检查点初始化基于动作的世界模型。然而，这可能会遇到由于两个领域在视觉外观和物理动态上的差异造成的潜在不匹配问题。直接将预训练-微调范式应用于下游任务往往会覆盖预训练潜在特征中编码的解耦信息，从而在源领域与目标领域之间存在较大领域差异时导致性能下降。

通过对包含多样视觉变化的干扰视频进行全面的预训练，视频预测模型构建了一个可解释且正交的潜在空间。在这个空间中，潜在变量 $\mathbf { z } _ { \mathrm { d i s e n } }$ 实现了高度的解耦。为了利用预训练模型的先验语义知识并提高下游任务的样本效率，我们引入了一种离线到在线的潜在蒸馏方法。这种方法使得预训练模型的 $\mathbf { z } _ { \mathrm { d i s e n } }$ 的解耦能力能够有效转移到世界模型的潜在变量 $\mathbf { z } _ { \mathrm { t a s k } }$。具体来说，这是通过最小化两个领域的潜在分布之间的Kullback-Leibler (KL) 散度来实现的。相应的蒸馏损失 ${ \mathcal { L } } _ { \mathrm { d i s t i l l } }$ 可以表述如下：

$$
{ \mathcal { L } } _ { \mathrm { d i s t i l l } } = \mathrm { K L } \left( \mathbf { z } _ { \mathrm { d i s e n } } \| \mathbf { z } _ { \mathrm { t a s k } } \right) = \sum \mathbf { z } _ { \mathrm { d i s e n } } \cdot \log \left( { \frac { \mathbf { z } _ { \mathrm { d i s e n } } } { \mathbf { z } _ { \mathrm { t a s k } } } } \right)
$$

# 3.4. 解耦世界模型适应性

通过获得良好解缠的表示 $\mathbf { z } _ { \mathrm { d i s e n } }$ 并采用潜在蒸馏进行知识转移，我们提出了一种基于深度强化学习的世界模型 $\mathcal { M } _ { \phi }$，旨在利用这些特征提高对环境变化的互操作性和鲁棒性（算法 1 第 14-15 行）。$\mathcal { M } _ { \phi }$ 的组成部分可以详细描述如下：

递归转移：$\begin{array} { l } { h _ { t } = f _ { \phi } ( h _ { t - 1 } , z _ { t - 1 } , a _ { t - 1 } ) } \\ { \mathbf { z } _ { t } \sim e _ { \phi } ( o _ { t } ) } \\ { z _ { t } \sim q _ { \phi } ( z _ { t } \mid h _ { t } , \mathbf { z } _ { t } ) } \\ { \tilde { z } _ { t } \sim p _ { \phi } ( \tilde { z } _ { t } \mid h _ { t } ) } \\ { \tilde { \phi } _ { t } \sim p _ { \phi } ( \hat { o } _ { t } \mid h _ { t } , z _ { t } ) } \\ { \hat { r } _ { t } \sim r _ { \phi } ( \hat { r } _ { t } \mid h _ { t } , z _ { t } ) } \\ { \tilde { \gamma } _ { t } \sim p _ { \phi } ( \hat { r } _ { t } \mid h _ { t } , z _ { t } ) } \end{array}$ $\beta$ VAE 编码器： 后验状态： 先验状态： 重构： 奖励预测： 折扣因子： 各向同性单位高斯分布：$p ( \mathbf { z } ) = \mathcal { N } ( \mathbf { 0 } , I )$，其中 $\phi$ 表示世界模型的组合参数。我们根据从重放缓冲区 $\boldsymbol { B }$ 采样的数据训练 $\mathcal { M } _ { \phi }$，使用以下损失函数：

$$
\begin{array} { r l } & { \mathcal { L } ( \phi ) = \mathbb { E } _ { q _ { \phi } } \Big [ \displaystyle \sum _ { t = 1 } ^ { T } \underbrace { - \ln p _ { \phi } ( o _ { t } \mid h _ { t } , z _ { t } ) } _ { \mathrm { i m g e r e c o n s t u c i o n } } \underbrace { - \ln r _ { \phi } ( r _ { t } \mid h _ { t } , z _ { t } ) } _ { \mathrm { r e w a r d p r e d i c i o n } } } \\ & { \qquad \underbrace { - \ln p _ { \phi } ( \gamma _ { t } \mid h _ { t } , z _ { t } ) } _ { \mathrm { d i s c o u n p r e d i c i o n } } \underbrace { + \alpha \mathrm { K L } \left[ q _ { \phi } ( z _ { t } \mid h _ { t } , o _ { t } ) \right] \mid p _ { \phi } ( \hat { z } _ { t } \mid h _ { t } ) \Big ] } _ { \mathrm { K L d i v e r g e n c e } } } \\ & { \qquad \underbrace { + \beta \mathrm { K L } \left[ q _ { \phi } ( \mathbf { z } _ { t } \mid o _ { t } ) \mid \mid p ( \mathbf { z } _ { t } ) \right] } _ { \mathrm { . . . . . . ~ . ~ . } } + \underbrace { \eta \mathcal { L } _ { \mathrm { d i s s i n } } } _ { \mathrm { . . . . . . ~ . } } \Big ] . } \end{array}
$$

其中 $\beta$ 是用于平衡重建质量与解缠能力的超参数。在适应阶段，$\eta$ 作为一个超参数，从 0.1 逐渐降低到 0.01。直观上，$\eta$ 控制着通过从预训练的视频预测模型转移的共享世界知识逐步适应世界模型。在这个框架的全面训练过程中，我们赋予 DisWM 学习和理解潜在语义表示的能力。这一增强使得模型对环境变化的敏感性降低，例如物体颜色、位置和背景的变化。此外，通过在微调阶段引入动作和奖励，世界模型能够生成具有更丰富表现的数据，从而改善解缠表示的学习。对于行为学习，我们采用与 DreamerV2 [10] 一致的 actor-critic 方法（算法 1 的第 16-18 行）。有关行为学习的更多细节，请参阅补充材料 B。

# 4. 实验

# 4.1. 实验设置

基准测试 我们在DeepMind控制套件（DMC）[34]、MuJoCo推杆[35]和抽屉世界[37]上评估DisWM。DMC是一个广泛采用的基准，涵盖了全面且灵活的机器人控制任务。在DMC基准中，我们使用5个任务，即Walker Walk、Cheetah Run、Hopper Stand、Finger Spin和Cartpole Swingup。在MuJoCo推杆中，采用了一种多关节机器人手臂来操控一个目标圆柱体（物体）。目标是使用机器人末端执行器（指尖）将物体移动到指定的目标位置。智能体会收到负奖励，这个负奖励由三个部分组成：指尖与目标之间的距离、物体与目标位置之间的距离，以及对于大动作的惩罚。抽屉世界是一个修改版的Metaworld [43]基准，旨在评估操作任务中的纹理适应性。它包括来自真实照片和网格纹理的五种附加纹理。在训练期间，我们最初使用网格纹理，然后在中途更换为木纹纹理，同时在评估时专门采用金属纹理。相应的抽屉世界结果在补充材料C.2中报告。此外，比较基线的介绍见补充材料A。实现细节。在线微调阶段的视觉观察被调整为$64 \times 64$像素。受APV [29]的启发，我们使用DreamerV2 [10]构建了包含100万帧的干扰视频数据集，以使用视觉颜色干扰器与环境进行交互。该视频数据集由在整个过程中存储在回放缓冲区中的样本组成。

# 算法 1 DisWM 的训练流程。

超参数：$H$ : 潜在想象的时间范围。

需要：干扰视频数据集 $\mathcal { D }$ 。 初始化：模型参数 $\{ \phi , \psi , \xi \}$ 。 对于训练步骤 $t = 1 , 2 , \ldots , K _ { 1 }$ ，执行解耦表示预训练 抽样随机小批量 $\{ o _ { t } \} _ { t = 1 } ^ { T } \sim \mathcal { D }$ 。 从各向同性单位高斯 $\mathcal { N } ( \mathbf { 0 } , I )$ 获取高斯先验 $\mathbf { z } _ { t }$ 。 通过最小化公式 (2) 来预训练无动作的视频预测模型，并加入解耦正则化。 结束循环 训练随机智能体并收集回放缓冲区 $\boldsymbol { B }$ 。 当未收敛时，执行以下操作： 对于训练步骤 $t = 1 , 2 , \ldots , K _ { 2 }$ ，执行 抽样 $\{ ( o _ { t } , a _ { t } , r _ { t } ) \} _ { t = 1 } ^ { T } \sim \{$ 。 使用公式 (3) 将解耦特征蒸馏至世界模型。离线到在线潜在蒸馏 从各向同性单位高斯 $\mathcal { N } ( \mathbf { 0 } , I )$ 获取高斯先验 $\mathbf { z } _ { t }$ 。 $\triangleright$ 解耦世界模型适应 使用公式 (5) 在潜在蒸馏和解耦约束下训练世界模型 $\mathcal { M } _ { \phi }$ 。 使用 $\pi _ { \psi }$ 和 $\mathcal { M } _ { \phi }$ 生成 $\{ ( \hat { z } _ { i } , \hat { a } _ { i } ) \} _ { i = t } ^ { t + H }$ 行为学习 在 $\{ ( \hat { z } _ { i } , \hat { a } _ { i } ) \} _ { i = t } ^ { t + H }$ 上训练评论家 $v _ { \xi }$ 在 $\{ ( \hat { z } _ { i } , \hat { a } _ { i } ) \} _ { i = t } ^ { t + H }$ 上训练演员 $\pi _ { \psi }$ 。 结束循环 $o _ { 1 } \gets \mathrm { n v }$ .reset() 环境交互 对于时间步骤 $t = 1 , 2 , \dots , T$ ，执行 抽样 $\hat { a } _ { t } \sim \pi _ { \psi } \big ( \hat { a } _ { t } \mid \hat { z } _ { t } \big )$ 。 $r_t, O_{t+1} \leftarrow \text{env.step} (\hat { a } _ { t })$ 。 结束循环 将数据附加到回放缓冲区 $\boldsymbol { B }$ 。 结束当训练过程直至智能体达到最大分数。对于 DMC 基准中的任务，智能体的训练步骤限制为 $1 \times 10 ^ { 6 }$ 环境步骤。每次运行 DisWM 需要大约 5GB 的显存，并在单个 RTX 3090 GPU 上训练约 16 小时。我们的方法中，良好解耦的潜在 $\mathbf { z } _ { \mathrm { d i s e n } }$ 和下游任务潜在 $\mathbf { z } _ { \mathrm { t a s k } }$ 的维度均设置为 20。在图 3 中，我们展示了各种任务的示例观察及其颜色干扰物。我们使用固定的颜色集来训练智能体，其中 RGB 值在原始值周围的有限范围内变化。此外，在训练过程的中点，我们将切换为不同的颜色方案以适应不同的干扰物。

# 4.2. 主要比较

我们评估了所有方法的样本效率和任务性能，通过训练曲线分析每个回合的回报。图4展示了DisWM及所有基线方法的表现。值得注意的是，它的表现超过了TED [7]，后者在RAD [17]之上，是针对包含干扰因素的环境设计的。对于离线到在线微调模型，DV2微调通过从干扰视频中转移知识，获得了第二好的性能。然而，我们观察到在样本效率上显著下降，尤其是在源域和目标域之间存在较大的数据分布变化的场景中（例如，$\mathrm { D M C } \mathrm { M u J o C o ) }$。这些变化可能出现在视觉观察、物理动态、奖励定义或机器人的动作空间等多个方面。另一个重要的基线是APV [29]，它专注于从具有堆叠潜在预测模型的视频中转移知识。然而，缺乏针对视觉干扰因素的环境特定设计，直接训练可能最终导致下游任务性能下降。CURL模型在学习有效的行为策略方面面临困难，特别是在Hopper Stand任务中。关于具有挑战性的DMC人形机器人行走的更多结果请参见补充材料C.1。此外，我们在图5和图6中展示了定性结果。图5展示了在预训练阶段$\beta$-VAE的遍历情况。在每一行遍历中，一个独特的属性发生变化，而其他属性保持不变，表明预训练模型成功拆解和学习了该属性，从而提高了RL智能体的样本效率。图6展示了在微调阶段MuJoCo Pusher上的细粒度拆解结果，证明了世界模型能够有效地拆解变化。

# 4.3. 模型分析

消融研究。我们进行消融研究以验证潜在蒸馏和解耦约束的效果。图7（左）展示了DMC Walker Walk Cheetah Run中的对应结果。绿色曲线显示，移除DisWM的潜在蒸馏导致性能下降，这表明潜在蒸馏在早期训练阶段至关重要。对于蓝色曲线所代表的模型，我们在预训练和微调阶段都没有采用解耦约束。可以看出，引入基于深度强化学习的训练和解耦表示显著提高了智能体的学习效率。

![](images/3.jpg)  
Figure 3. Example image observations of our modified DMC and MuJoCo Pusher with color distractors.

![](images/4.jpg)  
Figure 4. Comparison of DisWM against visual RL baselines, including DreamerV2 [10], $A P V$ [29], DV2 Finetune, TED [7], CURL [18].

![](images/5.jpg)  
Figure 5. Visualization of traversals of $\beta$ VAE during the pretraining phase.

![](images/6.jpg)  
displays the traversal results on a specific attribute.

敏感性分析。我们对 DMC（Cheetah $R u n $ Walker Walk）进行敏感性分析。如图 7（中间）所示，当表示解耦的 $\beta$ 值过小时，模型学习到纠缠的潜在表示。当 $\beta$ 值过大时，会妨碍图像的重建，导致性能下降。潜在蒸馏权重 $\eta$ 控制跨域迁移的规模。直观上，将该超参数设置得过低可能导致下游智能体无法从预训练模型中获得足够的知识。相反，过高的 $\eta$ 可能导致模型过拟合于预训练模型，对下游任务的学习不利。关于潜在空间维度的额外敏感性分析见补充材料 C.3。视频域的影响。在图 8 中，我们评估了在 DMC Cartpole Swingup 上的 DisWM，通过在替代视频数据集上进行预训练，包括从 Finger Spin、Reacher Easy、Walker Walk 和 Hopper Stand 收集的帧。有趣的是，与没有预训练的 DreamerV2 智能体相比，DisWM 总是能够通过离线到在线的潜在蒸馏从预训练中受益。它从预训练模型中获取了语义知识，并在微调阶段增强了解耦能力。

![](images/7.jpg)  
FgureThese gures llustrate theablation studies and sensitivity analyses o DisWMon DMC Walker Wal Cheetah Run. Left: cross-domain latent distillation weight. Right: The performance of DisWM with different disentanglement scale.

![](images/8.jpg)  
Figure 8. Performance of DisWM on DMC Cartpole Swingup with different video datasets.

# 5. 相关工作

视觉模型基强化学习。视觉强化学习从原始像素中学习控制策略，在多种任务中取得了显著的性能 [3, 4, 31, 37]，而之前的强化学习研究则集中于从低维状态中学习策略。现有的方法主要可分为两大方向：无模型强化学习 [18, 19, 24, 32, 42, 44, 47] 和基于模型的强化学习 [1, 8, 10, 12, 20, 21, 23, 27, 30, 36, 46]。以下方法专门解决视觉模型基强化学习中的环境变化。Pan [28] 等人通过反向动力学的优化将视觉动态分解为可控状态和不可控状态。SeeX [16] 提出了一个双层优化框架，采用分离的世界模型并最大化与任务相关的不确定性。与这些研究正交，我们的方法采用基于深度强化学习的世界模型，以减轻视觉变化所带来的问题。迁移强化学习 为了促进对未见任务的学习，迁移强化学习 [22, 24, 26, 33, 36, 48] 利用从过去任务中学到的知识。一个具有前景的方法是将来自可获取视频的世界知识迁移，以改善下游控制。APV [29] 建立了一个使用堆叠潜在预测模型和基于视频的内在奖励的预训练-微调框架。IPV [39] 引入了上下文化的世界模型，这些模型在多样的野外视频上进行了预训练。它结合了一个上下文编码器，该编码器与潜在动态模型一起工作，融入图像编码器以捕获丰富的上下文信息。PreLAR [45] 通过使用反向动力学编码器从无动作视频中推导有意义的动作对世界模型进行了预训练。不同于这些方法，我们提出了一种新方案，通过离线到在线的潜在蒸馏，将来自分散视频的世界知识迁移，以提高下游任务的学习效率。

# 6. 结论与局限性

在本文中，我们提出了一种名为 DisWM 的迁移强化学习方法，旨在解决实际场景中环境变异的挑战。我们的关键见解是利用可获取的干扰视频来提高下游任务的样本效率，从而提供灵活的解耦约束。具体来说，我们引入了解耦表示预训练、离线到在线潜在蒸馏和解耦世界模型适应，以改善下游控制。DisWM 在各种基准测试中展示了优于现有视觉强化学习基线的性能。我们方法的一个局限性是，解耦表示学习在复杂环境中面临挑战。探索具有更复杂变异的非平稳环境，如时间变化的背景视频干扰，可能进一步凸显我们方法在实际场景中的潜力。

致谢。本研究得到了国家自然科学基金（NSFC）62302246与62250062、浙江省自然科学基金（ZJNSFC）LQ23F010008、宁波市2023Z237、2024Z284、2024Z289、2023CX050011与2025Z038、智能电网国家科技重大专项（2024ZD0801200）、上海市科技重大项目（2021SHZDZX0102）、中央高校基本科研业务费以及青年博士创新基金（IDT Foundation）(S203.2.01.32.002)的支持。此外，沈阳青年中年科技创新人才支持计划（项目编号RC210488）、省级博士研究启动基金（项目编号2023-BS-214）、宁波东极高性能计算中心以及宁波数字双胞胎研究所也提供了额外支持。

# References

[1] Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, and François Fleuret. Diffusion for world modeling: Visual details matter in atari. In NeurIPS, 2024. 8   
[2] Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation learning: A review and new perspectives. TPAMI, 35(8):17981828, 2013. 2   
[3] Hyesong Choi, Hunsang Lee, Seongwon Jeong, and Dongbo Min. Environment agnostic representation for visual reinforcement learning. In ICCV, pages 263273, 2023. 8   
[4] Hyesong Choi, Hunsang Lee, Wonil Song, Sangryul Jeon, Kwanghoon Sohn, and Dongbo Min. Local-guided global: Paired similarity representation for visual reinforcement learning. In CVPR, pages 1507215082, 2023. 8   
[5] Djork-Arné Clevert, Thomas Unterthiner, and Sepp Hochreiter. Fast and accurate deep network learning by exponential linear units (elus). arXiv preprint arXiv:1511.07289, 2015. 1   
[6] Mhairi Dunion, Trevor McInroe, Kevin Sebastian Luck, Josiah Hanna, and Stefano Albrecht. Conditional mutual information for disentangled representations in reinforcement learning. In NeurIPS, 2023. 1, 2   
[7] Mhairi Dunion, Trevor McInroe, Kevin Sebastian Luck, Josiah P Hanna, and Stefano V Albrecht. Temporal disentanglement of representations for improved generalisation in reinforcement learning. In ICLR, 2023. 1, 2, 5, 6   
[8] Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In ICML, 2019. 8   
[9] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. In ICLR, 2020.   
10] Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, and Jimmy Ba. Mastering atari with discrete world models. In ICLR, 2021. 4, 6, 8, 1   
[11] Danjar Harner, Kuang-Huel Lee, lan riscner, ana Peter Abbeel. Deep hierarchical planning from pixels. arXiv preprint arXiv:2206.04114, 2022. 1   
[12] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. Nature, 2025. 8, 1   
[13] Nicklas Hansen, Hao Su, and Xiaolong Wang. Td-mpc2: Scalable, robust world models for continuous control. In ICLR, 2024. 1   
[14] Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. beta-vae: Learning basic visual concepts with a constrained variational framework. In ICLR, 2017. 2, 3   
[15] Irina Higgins, Arka Pal, Andrei Rusu, Loic Matthey, Christopher Burgess, Alexander Pritzel, Matthew Botvinick, Charles Blundell, and Alexander Lerchner. Darla: Improving zero-shot transfer in reinforcement learning. In ICML, 2017. 2   
[16] Kaichen Huang, Shenghua Wan, Minghao Shao, Hai-Hang Sun, Le Gan, Shuai Feng, and De-Chuan Zhan. Leveraging separated world model for exploration in visually distracted environments. NeurIPS, 2024. 8   
[17] Misha Laskin, Kimin Lee, Adam Stooke, Lerrel Pinto, Pieter Abbeel, and Aravind Srinivas. Reinforcement learning with augmented data. In NeurIPS, 2020. 5   
[18] Michael Laskin, Aravind Srinivas, and Pieter Abbeel. Curl: Contrastive unsupervised representations for reinforcement learning. In ICML, pages 56395650, 2020. 1, 6, 8   
[19] Haoran Li, Zhennan Jiang, YUHUI CHEN, and Dongbin Zhao. Generalizing consistency policy to visual rl with prioritized proximal experience regularization. In NeurIPS, 2024. 8   
[20] Jiajian Li, Qi Wang, Yunbo Wang, Xin Jin, Yang Li, Wenjun Zeng, and Xiaokang Yang. Open-world reinforcement learning over long short-term imagination. In ICLR, 2025. 1,8   
[21] Jessy Lin, Yuqing Du, Olivia Watkins, Danijar Hafner, Pieter Abbeel, Dan Klein, and Anca Dragan. Learning to model the world with language. In ICML, 2024. 8   
[22] Chris Lu, Yannick Schroecker, Albert Gu, Emilio Parisotto, Jakob Foerster, Satinder Singh, and Feryal Behbahani. Structured state space models for in-context reinforcement learning. NeurIPS, 2023. 8   
[23] Haoyu Ma, Jialong Wu, Ningya Feng, Chenjun Xiao, Dong Li, Jianye Hao, Jianmin Wang, and Mingsheng Long. Harmonydream: Task harmonization inside world models. In ICML, 2024. 8   
[24] Yecheng Jason Ma, Shagun Sodhani, Dinesh Jayaraman, Osbert Bastani, Vikash Kumar, and Amy Zhang. Vip: Towards universal visual reward and representation via value-implicit pre-training. In ICLR, 2023. 8   
[25] Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. JMLR, 9(Nov):25792605, 2008. 2   
[26] Pietro Mazzaglia, Tim Verbelen, Bart Dhoedt, Alexandre Lacoste, and Sai Rajeswar. Choreographer: Learning and adapting skills in imagination. In ICLR, 2023. 8   
[27] Pietro Mazzaglia, Tim Verbelen, Bart Dhoedt, Aaron C Courville, and Sai Rajeswar Mudumba. Genrl: Multimodalfoundation world models for generalization in embodied agents. NeurIPS, 2024. 8   
[28] Minting Pan, Xiangming Zhu, Yunbo Wang, and Xiaokang Yang. Iso-dream: Isolating and leveraging noncontrollable visual dynamics in world models. In NeurIPS, 2022. 1, 8   
[29] Younggyo Seo, Kimin Lee, Stephen L James, and Pieter Abbeel. Reinforcement learning with action-free pretraining from videos. In ICML, 2022. 4, 5, 6, 8, 1   
[30] Younggyo Seo, Junsu Kim, Stephen James, Kimin Lee, Jinwoo Shin, and Pieter Abbeel. Multi-view masked world models for visual robotic manipulation. In ICML, 2023. 8   
[31] Wonil Song, Hyesong Choi, Kwanghoon Sohn, and Dongbo Min. A simple framework for generalization in visual rl under dynamic scene perturbations. NeurIPS, 37:121790 121826, 2024. 8   
[32] Adam Stooke, Kimin Lee, Pieter Abbeel, and Michael Laskin. Decoupling representation learning from reinforcement learning. In ICML, pages 98709879, 2021. 8   
[33] Yanchao Sun, Ruijie Zheng, Xiyao Wang, Andrew Cohen, and Furong Huang. Transfer rl across observation feature spaces via model-based regularization. In ICLR, 2022. 8   
[34] Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las Casas, David Budden, Abbas Abdolmaleki, Josh Merel, Andrew Lefrancq, et al. Deepmind control suite. arXiv preprint arXiv:1801.00690, 2018. 4   
[35] Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In IROS, 2012. 4   
[36] Qi Wang, Junming Yang, Yunbo Wang, Xin Jin, Wenjun Zeng, and Xiaokang Yang. Making offine rl online: Collaborative world models for offine visual reinforcement learning. In NeurIPS, 2024. 8   
[37] Xudong Wang, Long Lian, and Stella X Yu. Unsupervised visual attention and invariance for reinforcement learning. In CVPR, pages 66776687, 2021. 4, 8, 1   
[38] Xin Wang, Hong Chen, Zihao Wu, Wenwu Zhu, et al. Disentangled representation learning. TPAMI, 2024. 2   
[39] Jialong Wu, Haoyu Ma, Chaoyi Deng, and Mingsheng Long. Pre-training contextualized world models with in-the-wild videos for reinforcement learning. NeurIPS, 2023. 8, 1   
[40] Baao Xie, Bohan Li, Zequn Zhang, Junting Dong, Xin Jin, Jingyu Yang, and Wenjun Zeng. Navinerf: Nerf-based 3d representation disentanglement by latent semantic navigation. In ICCV, 2023. 2   
[41] Baao Xie, Qiuyu Chen, Yunnan Wang, Zequn Zhang, Xin Jin, and Wenjun Zeng. Graph-based unsupervised disentangled representation learning via multimodal large language models. In NeurIPS, 2024. 2   
[42] Denis Yarats, Rob Fergus, Alessandro Lazaric, and Lerrel Pinto. Mastering visual continuous control: Improved dataaugmented reinforcement learning. In ICLR, 2022. 8   
[43] Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Metaworld: A benchmark and evaluation for multi-task and meta reinforcement learning. In CoRL. 2019. 4   
[44] Amy Zhang, Rowan McAllister, Roberto Calandra, Yarin Gal, and Sergey Levine. Learning invariant representations for reinforcement learning without reconstruction. In ICLR, 2021. 8   
[45] Lixuan Zhang, Meina Kan, Shiguang Shan, and Xilin Chen. Prelar: World model pre-training with learnable action representation. In ECCV, 2024. 1, 8   
[46] Weipu Zhang, Gang Wang, Jian Sun, Yetian Yuan, and Gao Huang. Storm: Efficient stochastic transformer based world models for reinforcement learning. In NeurIPS, 2023. 8   
[47] Ruijie Zheng, Xiyao Wang, Yanchao Sun, Shuang Ma, Jieyu Zhao, Huazhe Xu, Hal Daumé III, and Furong Huang. Taco: Temporal latent action-driven contrastive loss for visual reinforcement learning. In NeurIPS, 2023. 8   
[48] Zhuangdi Zhu, Kaixiang Lin, Anil K Jain, and Jiayu Zhou. Transfer learning in deep reinforcement learning: A survey. TPAMI, 45(11):1334413362, 2023. 8

# Disentangled World Models: Learning to Transfer Semantic Knowledge from Distracting Videos for Reinforcement Learning

Supplementary Material

# A. Compared Baselines

We compare DisWM with strong visual RL agents, including

• DreamerV2 [10]: A model-based RL (MBRL) approach that trains world model and learns by imagining future latent states.

• APV [29]: It learns informational representations via action-free pretraining on videos and finetunes the agent with learned representations in the downstream tasks with action.

•DV2 Finetune: It pretrains a DreamerV2 agent [10] on distracting videos and then finetunes the trained model in the downstream tasks. Note that some tasks have different action spaces, which makes it difficult to finetune directly. Therefore, the action space of two tasks is set as the maximum action space of both environments.

•TED [7]: It adopts a classification task to learn temporally disentangled representations in visual RL.

• CURL [18]: A model-free RL method that employs contrastive learning to improve its sample efficiency.

# B. Behavior Learning

For the behavior learning of DisWM, we adopt the actorcritic method following DreamerV2 [10]. Concretely, the actor and critic are both implemented as MLPs with ELU activations [5]. Formally, the actor and critic are defined as below:

$$
\begin{array} { r l } & { \mathrm { A c t o r : ~ } \hat { a } _ { t } \sim \pi _ { \psi } ( \hat { a } _ { t } | \hat { z } _ { t } ) } \\ & { \mathrm { C r i t i c : ~ } v _ { \xi } ( \hat { z } _ { t } ) \approx \mathbb { E } _ { p _ { \phi } , p _ { \psi } } \Big [ \sum _ { \tau \geq t } \hat { \gamma } _ { \tau - t } \hat { r } _ { \tau } \Big ] . } \end{array}
$$

The actor $\pi _ { \psi }$ is optimized by maximizing

$$
\begin{array} { r l } & { \displaystyle \mathcal { L } ( \psi ) = \mathbb { E } _ { p _ { \phi } , p _ { \psi } } \Big [ \displaystyle \sum _ { t = 1 } ^ { H - 1 } ( \underbrace { \beta \mathrm { H } \left[ a _ { t } | \hat { z } _ { t } \right] } _ { \mathrm { e n t r o p y ~ r e g u l a r i z a t i o n } } + \underbrace { \rho V _ { t } } _ { \mathrm { d y n a m i c s ~ b a c k p r o p } } } \\ & { \displaystyle + \underbrace { ( 1 - \rho ) \ln \pi _ { \psi } ( \hat { a } _ { t } | \hat { z } _ { t } ) \mathrm { s g } ( V _ { t } - v _ { \xi } ( \hat { z } _ { t } ) ) } _ { \mathrm { R E I N F O R C E } } \Big ] . } \end{array}
$$

We train the critic $v _ { \xi }$ by minimizing

$$
\mathcal { L } ( \xi ) = \mathbb { E } _ { p _ { \phi } , p _ { \psi } } \Big [ \sum _ { t = 1 } ^ { H - 1 } \frac { 1 } { 2 } \left( v _ { \xi } \left( \hat { z } _ { t } \right) - \mathrm { s g } \left( V _ { t } \right) \right) ^ { 2 } \Big ] .
$$

where $\mathtt { S g }$ is a stop gradient operator.

The $\lambda$ -target $V _ { t }$ that involves a weighted average of reward information used in Eq. (7) and Eq. (8) is defined as:

$$
V _ { t } \doteq { \hat { r } } _ { t } + { \hat { \gamma } } _ { t } \left\{ { \begin{array} { l l } { ( 1 - \lambda ) v _ { \xi } \left( { \hat { z } } _ { t + 1 } \right) + \lambda V _ { t + 1 } } & { { \mathrm { i f ~ } } t < H } \\ { v _ { \xi } \left( { \hat { z } } _ { H } \right) } & { { \mathrm { i f ~ } } t = H } \end{array} } \right. .
$$

where $H$ is the imagination horizon. Notably, the disentangled world model is not optimized during behavior learning.

# C. Additional Results

# C.1. Results on DMC

We compare the performance of DreamerV3 [12], TDMPC2 [13], ContextWM [39], and our approach on DMC. As shown Table A, DisWM outperforms other strong baselines in terms of episode return.

# C.2. Results on DrawerWorld

We present results on DrawerWorld [37] in Table B. As reported in Table B, DisWM (source: Finger Spin) outperforms other baselines in terms of success rate $( \% )$ on all tasks.

# C.3. Sensitivity of the Latent Space Dimension

We visualize sensitivity analyses on the latent space dimension in Figure I. We observe that when $\mathbf { z } _ { \mathrm { d i m } }$ for the $\beta$ VAE is too small, it impedes the learning of disentangled representations, leading to a decline in performance.

![](images/9.jpg)  
Figure I. Sensitivity analyses on Cheetah Run Walker Walk

Table A. Comparison with strong baselines on DMC.   

<table><tr><td>Model</td><td></td><td>Reacher Easy → Cheetah Run Walker Walk → Humanoid Walk</td></tr><tr><td>DreamerV3</td><td>662 ± 9</td><td>12 ± 17</td></tr><tr><td>TD-MPC2</td><td>510 ± 15</td><td>1 ± 0</td></tr><tr><td>ContextWM</td><td>661 ± 49</td><td>1 ± 0</td></tr><tr><td>DisWM</td><td>817 ± 59</td><td>147 ± 85</td></tr></table>

<table><tr><td>Model</td><td>DrawerClose</td><td>DrawerOpen</td></tr><tr><td>TDMPC2</td><td>3 ± 6</td><td>43 ± 25</td></tr><tr><td>ContextWM</td><td>37 ± 12</td><td>23 ± 25</td></tr><tr><td>DisWM</td><td>77 ± 6</td><td>70 ± 10</td></tr></table>

Table B. Performance on DrawerWorld with texture variations.

# C.4. Runtime Comparisons

We provide the detailed runtime and parameter comparisons with baselines in Table C. Note that the inference time is computed for one episode.

Table C. Runtime and model size comparisons evaluated on DMC (Finger Spin Reacher Easy). DV2 FT is short for DreamerV2 finetune.   

<table><tr><td>Model</td><td>Training Steps</td><td>Training time</td><td>Inference time</td><td>Params (M)</td></tr><tr><td>CURL</td><td>100k</td><td>303 min</td><td>4.97 sec</td><td>10.7</td></tr><tr><td>DV2 FT</td><td>200k</td><td>1522 min</td><td>9.88 sec</td><td>12.1</td></tr><tr><td>APV</td><td>200k</td><td>1722 min</td><td>10.15 sec</td><td>13</td></tr><tr><td>TED</td><td>100k</td><td>1051 min</td><td>20.49 sec</td><td>11.5</td></tr><tr><td>DV2</td><td>100k</td><td>901 min</td><td>9.59 sec</td><td>12.1</td></tr><tr><td>DisWM</td><td>200k</td><td>1311 min</td><td>9.48 sec</td><td>5.8</td></tr></table>

# C.5. Sample Diversity Visualization

The adaptation stage enriches the sample diversity, as shown in Figure J, for Cheetah $R u n $ Walker Walk, we sample 200 video clips of length 50 and visualize the corresponding latent features using t-SNE [25]. We find that the latent features of the online interactions are more diverse than those of the offline dataset.

# D. Hyperparameters

The final hyperparameters of DisWM are reported in Table D.

![](images/10.jpg)  
Figure J. Sample diversity enhanced by adaptation.

Table D. Hyperparameters of DisWM.   

<table><tr><td>Name</td><td>Notation</td><td>Value</td></tr><tr><td>Video prediction model</td><td></td><td></td></tr><tr><td>Image size KL divergence scale</td><td>β1</td><td>64 × 64 1</td></tr><tr><td>Disentanglement scale Latent dimension</td><td>β2</td><td>0.015 20</td></tr><tr><td>Learning rate</td><td></td><td>3 ·10−4</td></tr><tr><td>Disentangled World Model</td><td></td><td></td></tr><tr><td>Latent distillation weight</td><td>η</td><td>0.1</td></tr><tr><td>Disentanglement scale</td><td>β</td><td>0.015</td></tr><tr><td>KL divergence scale</td><td>α</td><td>1</td></tr><tr><td>Latent dimension</td><td>−</td><td>20</td></tr><tr><td>Batch size</td><td>B</td><td>50</td></tr><tr><td>Batch length</td><td>L</td><td>50</td></tr><tr><td>Learning rate</td><td></td><td>3 · 10−4</td></tr><tr><td>Behavior Learning</td><td></td><td></td></tr><tr><td>Imagination horizon</td><td>H</td><td>15</td></tr><tr><td>Discount</td><td>γ</td><td>0.99</td></tr><tr><td>λ-target</td><td>λ</td><td>0.95</td></tr><tr><td>Actor learning rate</td><td></td><td>8·10-5</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Critic learning rate</td><td></td><td>8·10-5</td></tr></table>