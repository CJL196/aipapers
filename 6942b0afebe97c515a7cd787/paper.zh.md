# 扩散强迫：下一令牌预测与全序列扩散相结合

Boyuan Chen 麻省理工学院 CSAIL boyuanc@mit.edu Diego Marti Monso* 慕尼黑工业大学 diego.marti@tum.de Yilun Du 麻省理工学院 CSAIL yilundu@mit.edu Max Simchowitz 麻省理工学院 CSAIL msimchow@mit.edu Russ Tedrake 麻省理工学院 CSAIL russt@mit.edu Vincent Sitzmann 麻省理工学院 CSAIL sitzmann@mit.edu

# 摘要

本文提出了扩散强制，这是一种新的训练范式，通过训练扩散模型对具有独立每个令牌噪声水平的一组令牌进行去噪。我们将扩散强制应用于序列生成建模，训练一个因果下一个令牌预测模型，生成一个或多个未来令牌，而不完全扩散过去的令牌。我们的方法展示了将下一个令牌预测模型的优势（如可变长度生成）与全序列扩散模型的优势（如引导采样到理想轨迹的能力）相结合的能力。我们的方法提供了一系列额外的功能，如（1）生成超出训练范围长度的连续令牌序列，例如视频，在基线发散的情况下，以及（2）新的采样和引导方案，独特地受益于扩散强制的可变视野和因果架构，这在决策和规划任务中带来了显著的性能提升。除了实证成功外，我们的方法被证明能够优化从真实联合分布中抽取的所有令牌子序列的似然性的变分下界。项目网站： https : / /boyuan . space/ diffusion-forcing

# 1 引言

概率序列建模在自然语言处理、视频预测和决策制定等多种机器学习应用中发挥着至关重要的作用。特别是下一token预测模型具有许多理想特性。它们能够生成长度各异的序列（通过自回归采样生成单个token或无限多个token），可以基于不同历史信息进行条件化，支持高效的树搜索，并可用于在线反馈控制。当前的下一token预测模型通过教师强迫进行训练，其中模型基于之前token的真实历史预测紧接着的token。这导致了两个局限：第一，缺乏机制指导序列采样以最小化某一特定目标；第二，当前的下一token模型在处理连续数据时容易不稳定。例如，当试图自回归生成视频时，较小的帧间预测误差会积累，从而导致模型发散。

![](images/1.jpg)  

Figure 1: Diffusion Forcing capabilities. Today, different applications such as language modeling [6], planning [37], or video generation [32, 70] rely on either auto-regressive next-token prediction or full-sequence diffusion, according to their respective unique capabilities. The proposed Diffusion Forcing is a novel sequence generative model that enjoys key strengths of both model types.

全序列扩散似乎提供了一种解决方案。常用于视频生成和长远规划，它通过扩散固定数量词元的连接来直接建模它们的联合分布[32, 1]，所有词元的噪声水平相同。它们提供扩散指导[31, 16]，以引导采样生成理想的序列，这在决策（规划）应用中极为重要[37, 35]。此外，它们在生成视频等连续信号方面表现突出[32]。然而，全序列扩散普遍通过非因果、非掩蔽的体系结构进行参数化。除了将采样限制为全序列而非可变长度生成外，我们还展示了这限制了指导和子序列生成的可能性（图1）。进一步地，我们证明，简单地通过训练全序列扩散的下一个词元预测模型来结合两者的优点，会导致生成质量较差，直观的原因在于这并没有建模早期词元的小不确定性必然需要后期词元较高不确定性的事实。在本文中，我们引入扩散强迫（DF），这是一种训练和采样范式，其中每个词元与一个随机、独立的噪声水平相关联，并且词元可以根据任意独立的每词元调度进行去噪，通过一个共享的下一个或下几个词元预测模型来实现。我们的做法受到以下观察的启发：对词元进行加噪是一种部分掩蔽的形式——零噪声意味着一个词元是未掩蔽的，而完全噪声则完全掩蔽该词元。因此，DF迫使模型学习“去掩蔽”任何变噪词元的集合（图2）。与此同时，通过将预测参数化为下一词元预测模型的组合，我们的系统能够灵活生成可变长度的序列，并在组合上泛化到新的轨迹（图1）。

我们将序列生成的扩散框架实现为因果扩散强制（CDF），其中未来的词元依赖于过去的词元，通过因果架构进行建模。我们训练模型一次性去噪整个序列的所有词元，为每个词元设置独立的噪声水平。在采样过程中，CDF逐步将一系列高斯噪声帧去噪成干净样本，不同帧在每个去噪步骤中可能具有不同的噪声水平。与下一个词元预测模型一样，CDF可以生成可变长度的序列；不同于下一个词元预测，它能够稳定地从紧接着的词元生成到未来的数千个词元，甚至是连续的词元。此外，与全序列扩散相似，它同样接受指导，以实现高奖励生成。CDF通过协同利用因果性、灵活的时间范围和可变噪声调度，使得蒙特卡洛指导（MCG）成为可能，这显著提升了高奖励生成的采样效果，相较于非因果全序列扩散模型。图1概述了这些能力。

总之，我们的贡献包括：(1) 我们提出了Diffusion Forcing，这是一种新的概率序列模型，具有与下一个标记预测模型的灵活性，同时能够像全序列扩散模型一样执行长时间指导。(2) 利用Diffusion Forcing的独特能力，我们提出了一种新颖的决策框架，使我们能够将Diffusion Forcing同时用作策略和规划器。(3) 我们正式证明，在适当的条件下，优化我们提出的训练目标最大化了观察到的所有子序列的联合分布似然下界。(4) 我们在视频生成、基于模型的规划、视觉模仿学习和时间序列预测等多个领域对CDF进行了实证评估，并展示了CDF的独特能力，例如稳定长推理自回归视频生成、与用户确定的记忆范围组合训练时观察到的子序列、蒙特卡洛指导等。

![](images/2.jpg)  

Figure 2: Method Overview. Diffusion Forcing trains causal sequence neural networks (such as an RNN or a masked transformer) to denoise flexible-length sequences where each frame of the sequence can have a different noise level. In contrast, next-token prediction models, common in language modeling, are trained to predict a single next token from a ground-truth sequence (teacher forcing [65]), and full-sequence diffusion, common in video generation, train non-causal architectures to denoise all frames in a sequence at once with the same noise level. Diffusion Forcing thus interleaves the time axis of the sequence and the noise axis of diffusion, unifying strengths of both alternatives and enabling completely new capabilities (see Secs. 3.2,3.4).

# 2 相关工作与基础知识

我们讨论与我们核心应用——序列生成建模——相关的工作和基础知识；更多文献综述请参见附录C。我们的方法统一了序列建模的两个视角：沿时间轴的贝叶斯滤波，用下标$t$表示，以及沿“不确定性”（或噪声水平）轴的扩散，用上标$k$表示。接下来，我们将观测表示为$\mathbf{x} \in \mathcal{X}$，将潜态表示为$\mathbf{z} \in \mathcal{Z}$。

贝叶斯滤波。给定一个由潜在状态 $\mathbf { z } _ { t }$ 和观测 $\mathbf { x } _ { t }$ 定义的隐马尔可夫模型（HMM），贝叶斯滤波是一种从输入观测中递归估计潜在状态的概率方法。先验模型 $p ( \mathbf { z } _ { t + 1 } | \mathbf { z } _ { t } )$ 基于当前状态推断下一个状态的信念，而观测模型则基于当前潜在状态推断下一个观测的信念 $p ( \mathbf { x } _ { t } | \mathbf { z } _ { t } )$。当进行新的观测时，后验模型 $p ( \mathbf { z } _ { t + 1 } | \mathbf { z } _ { t } , \mathbf { x } _ { t + 1 } )$ 提供关于下一个潜在状态 $\mathbf { z } _ { t + 1 }$ 的更新估计。当与神经网络端到端训练时，潜在状态并不代表任何物理量的估计，而是足够表达的潜在变量，用于总结过去的观测以预测序列中未来的观测 $( \mathbf { x } _ { t ^ { \prime } } ) _ { t ^ { \prime } > t }$。扩散模型。扩散模型 [57, 29] 已被证明是一种高度表达性和可靠的生成模型。我们在此回顾其基本要素。令 $q ( \mathbf { x } )$ 表示感兴趣的数据分布，且令 $\mathbf { x } ^ { 0 } \equiv \mathbf { x } \sim q$。我们考虑一个前向扩散过程，该过程逐步向数据点添加高斯噪声，经过一系列时间步。该过程被建模为马尔可夫链，在每一步 $k$ 中，数据逐步被加噪：

$$
q ( \mathbf { x } ^ { k } | \mathbf { x } ^ { k - 1 } ) = \mathcal { N } ( \mathbf { x } ^ { k } ; \sqrt { 1 - \beta _ { k } } \mathbf { x } ^ { k - 1 } , \beta _ { k } \mathbf { I } )
$$

其中 $\mathcal { N }$ 是正态分布，$\beta _ { k }$ 是在每一步添加的噪声的方差，该噪声由调度 $\{ \beta _ { k } \in ( 0 , 1 ) \} _ { k = 1 } ^ { K }$ 控制。$\mathbf { x } ^ { K }$ 该过程也是马尔可夫链，并尝试利用参数化模型 $p _ { \theta }$ 从噪声中重建原始数据。

$$
p _ { \theta } ( \mathbf { x } ^ { k - 1 } | \mathbf { x } ^ { k } ) = \mathcal { N } ( \mathbf { x } ^ { k - 1 } ; \pmb { \mu } ( \mathbf { x } ^ { k } , k ) , \gamma _ { k } \mathbf { I } ) ,
$$

其中均值 $\pmb { \mu }$ 是一个由神经网络构成的模型，并且有文献 [30] 表明，可以将协方差设置为乘以一个固定常数 $\gamma _ { k }$ 的单位矩阵。采用标准表述，我们通过噪声预测 $\mathbf { \epsilon }$ 对均值 $\pmb { \mu }$ 进行重新参数化，得到 $\epsilon = ( \sqrt { 1 - \bar { \alpha } _ { t } } ) ^ { - 1 } \mathbf { x } _ { t } ^ { k _ { t } } - \sqrt { \bar { \alpha } _ { t } } \mu$。这导致了以下的最小二乘目标 [29]：

$$
\begin{array} { r } { \mathcal { L } ( \boldsymbol { \theta } ) = \mathbb { E } _ { k , \mathbf { x } ^ { 0 } , \epsilon } \left[ \| \boldsymbol { \epsilon } ^ { k } - \boldsymbol { \epsilon } _ { \boldsymbol { \theta } } ( \mathbf { x } ^ { k } , \boldsymbol { k } ) \| ^ { 2 } \right] , } \end{array}
$$

其中 $\mathbf{x}^{k} = \sqrt{\bar{\alpha_{t}}} \mathbf{x}^{0} + \sqrt{1 - \bar{\alpha_{t}}} \epsilon^{k}$，且 $\epsilon^{k} \sim \mathcal{N}(0, \mathbf{I})$。然后可以通过朗之万动力学从该模型中采样：$\mathbf{x}^{k-1} = \frac{1}{\sqrt{\alpha_{k}}} \left( \mathbf{x}_{t}^{k} - \frac{1 - \alpha_{k}}{\sqrt{1 - \bar{\alpha}_{k}}} \epsilon_{\theta}(\mathbf{x}_{t}^{k}, k) + \sigma_{k} \mathbf{w} \right)$。

扩散模型的引导。引导 [31, 16] 允许在采样时将扩散生成偏向于期望的预测。我们重点关注分类器引导 [16]：给定一个关于某个期望的 $y$（例如类别或成功指示器）的分类器 $c ( y | \mathbf { x } ^ { k } )$，可以将Langevin采样 [30] 的梯度 $\epsilon _ { \theta } ( \mathbf { x } ^ { k } , k )$ 修改为 $\epsilon _ { \theta } ( \mathbf { x } ^ { k } , k ) - \sqrt { 1 - \bar { \alpha } _ { k } } \nabla _ { x ^ { k } } \log c ( y | \mathbf { x } ^ { k } )$。这使得无需训练条件模型即可从 $\mathbf { x }$ 和类别标签 $y$ 的联合分布中采样。其他能量，例如最小二乘目标，通过比较模型输出与期望真值，也已在决策制定等应用中探索 [16, 37]。

下一个令牌预测模型。下一个令牌预测模型是序列模型，它根据过去的帧 $\mathbf { x } _ { 1 : t }$ 预测下一个帧 $\mathbf { x } _ { t + 1 }$。在训练时，将 $\mathbf { x } _ { 1 : t }$ 输入神经网络，并最小化连续数据的 $| | \hat { \mathbf { x } } - \mathbf { x } | | ^ { 2 }$ 或离散数据的交叉熵损失[65]。在采样时，按照 $p ( \mathbf { x } _ { t + 1 } | \mathbf { x } _ { 1 : t } )$ 采样下一个帧 $\hat { \mathbf { x } } _ { t + 1 }$。如果将 $\hat { \mathbf { x } } _ { t + 1 }$ 视为 $\mathbf { x } _ { t + 1 }$，则可以使用相同的模型来预测 $\mathbf { x } _ { t + 2 }$ 并重复此过程直到采样完整序列。与全序列扩散模型不同，下一个令牌模型不接受多步骤引导，因为先前的帧必须完全确定才能采样未来的帧。

扩散序列模型。扩散已广泛应用于序列建模。 [44] 使用全序列扩散模型通过引导实现可控的文本生成，例如生成符合特定词类的文本。 [32] 训练全序列扩散模型以合成短视频，并使用滑动窗口基于之前生成的帧推演更长的视频。 [37] 在离线强化学习中将全序列扩散模型用作规划器。这是通过在与环境的交互轨迹的数据集上训练，并在采样时使用分类器引导采样具有高奖励的轨迹，以实现所选目标。 [50] 修改自回归模型，以去噪基于先前词元的下一个词元。它使用教师强制进行训练，并对时间序列数据进行自回归采样下一个词元。与我们的工作最相似的是AR-Diffusion [66]，该方法以因果架构训练全序列文本扩散，其噪声级别在时间轴上呈线性相关。我们在附录C中详细比较了这一方法与我们的工作。

# 3 方法

# 3.1 噪声作为部分遮蔽

请记住，遮蔽是遮挡数据子集的做法，例如图像的某些区域[27]或序列中的时间步[15, 49]，并训练模型恢复未遮蔽的部分。在不失一般性的情况下，我们可以将任何一组词元，无论是顺序的还是非顺序的，视为由$t$索引的有序集合。使用教师强制进行下一个词元预测可以解释为在时间$t$遮蔽每个词元$\mathbf { x } _ { t }$，并基于过去的$\mathbf { x } _ { 1 : t - 1 }$进行预测。局限于序列时，我们将所有这些做法称为沿时间轴的遮蔽。我们还可以将完整序列前向扩散视为沿噪声轴的遮蔽，即$\mathbf { x } _ { 1 : T } ^ { 0 } \equiv \mathbf { x } _ { 1 : T }$。实际上，在经过$K$步加噪声之后，$\mathbf { x } _ { 1 : T } ^ { K }$大致上是纯白噪声，不包含原始数据的信息。

我们沿着掩蔽的两个轴建立统一视图（见图2）。我们用 $\mathbf { x } _ { 1 : T }$ 来表示一个词元序列，其中下标表示时间轴。如上所述，$\mathbf { x } _ { t } ^ { k _ { t } }$ 表示在前向扩散过程（2.1）下噪声等级为 $k _ { t }$ 的 $\mathbf { x } _ { t }$；$\mathbf { x } _ { t } ^ { 0 } = \mathbf { x }$ 是未加噪声的词元，而 $\mathbf { x } _ { t } ^ { K }$ 是白噪声 $\mathcal { N } ( 0 , \bf { I } )$。因此，$( \mathbf { x } _ { t } ^ { k _ { t } } ) _ { 1 \leq t \leq T }$ 表示一系列带噪观察，其中每个词元具有不同的噪声等级 $k _ { t }$，可以看作是通过添加噪声对每个词元应用的部分掩蔽程度。

# 3.2 扩散强迫：不同词元适用不同噪声水平

扩散强制（DF）是一个用于训练和采样任意序列长度的噪声词元$( \mathbf { x } _ { t } ^ { k _ { t } } ) _ { 1 \leq t \leq T }$的框架，其中每个词元的噪声水平$k _ { t }$在时间步上可以变化。本文集中于时间序列数据，因此我们使用因果架构实例化扩散强制（其中$\mathbf { x } _ { t } ^ { k _ { t } }$仅依赖于过去的噪声词元），我们称之为因果扩散强制（CDF）。为了简化，我们专注于使用普通的递归神经网络（RNN）[11]的最小实现。扩散强制还可以通过变换器实现，但我们将该讨论推迟到附录B.1。

<table><tr><td colspan="2">Algorithm 1 Diffusion Forcing Training</td><td>Algorithm 2 DF Sampling with Guidance</td></tr><tr><td colspan="2">1:loop</td><td>1: Input: Model θ, scheduling matrix K, initial latent</td></tr><tr><td>2:</td><td>Sample tajectory of observations (x1, ., XT ).</td><td>z0, guidance cost c(·).</td></tr><tr><td>3:</td><td>for t = 1, ..., T do</td><td>2: Initialize x1, . . . , XT ∼ N (0, σ2 I).</td></tr><tr><td>4:</td><td>Sample independent noise level k</td><td>3: for row m = M − 1, ..., 0 do</td></tr><tr><td>5:</td><td>{0, 1, .., K} xt = ForwardDiffuse(xt, kt)</td><td>for t = 1, . . . , T do ∼ pθ(Zt | zt−1, Xt, Km+1,t).</td></tr><tr><td>6:</td><td>−√¯αktxt Define  = $\fa</td><td>k ← Km,t, w ~ N (0, I). 1−αk (x ←</td></tr><tr><td>7:</td><td>√1− ¯αkt Update Zt ∼ pθ(zt|zt−1, xkt, kt).</td><td>θ (e, , x, k)) + √αk √1−¯αk σW</td></tr><tr><td>8:</td><td>Set t = θ(zt−1, xt, kt)</td><td>end for</td></tr><tr><td>9:</td><td>end for</td><td>9:</td></tr><tr><td>10:</td><td>L =MSELoss([1, .…, €n] , [1, .…, , n)</td><td>10: x1:H ←AddGuidance(xnew ,  log c(xnew )) 11:end for</td></tr><tr><td>11: 12:</td><td>Backprop with L and update θ</td><td>12: Return X1:T.</td></tr><tr><td></td><td>end loop</td><td></td></tr></table>

带权重 $\theta$ 的循环神经网络（RNN）维护潜变量 $\mathbf{z}_{t}$，以捕捉过去标记的影响，这些潜变量通过动态演变 $\mathbf{z}_{t} \sim p_{\theta}(\mathbf{z}_{t} | \mathbf{z}_{t-1}, \mathbf{x}_{t}^{k_{t}}, k_{t})$，并具有递归层。当接收到噪声观测 $\mathbf{x}_{t}^{k_{t}}$ 时，$\mathbf{z}_{t} \sim p_{\theta}(\mathbf{z}_{t} | \mathbf{z}_{t-1}, \mathbf{x}_{t_{.}}^{k_{t}}, k_{t})^2$。当 $k_{t} = 0$ 时，这是贝叶斯滤波中的后验更新；而当 $k_{t} = K$（且 $\mathbf{x}_{t}^{K}$ 是纯噪声，因此不提供信息）时，相当于在贝叶斯滤波中建模“先验分布” $p_{\theta}(\mathbf{z}_{t} \mid \mathbf{z}_{t-1})$。在给定潜变量 $\mathbf{z}_{t}$ 的情况下，观测模型 $p_{\theta}(\mathbf{x}_{t}^{\bar{0}} | \mathbf{z}_{t})$ 用于预测 $\mathbf{x}_{t}$。

训练。动态模型 $p _ { \theta } ( \mathbf { z } _ { t } | \mathbf { z } _ { t - 1 } , \mathbf { x } _ { t } ^ { k _ { t } } , k _ { t } )$ 和观测模型 $p _ { \theta } ( \mathbf { x } _ { t } ^ { 0 } | \mathbf { z } _ { t } )$ 一起构成一个 RNN 单元。该单元具有与标准条件扩散模型相同的输入输出行为，使用条件变量 $\mathbf { z } _ { t - 1 }$ 和嘈杂令牌 $\mathbf { x } _ { t } ^ { k _ { t } }$ 作为输入来预测无噪声的 $\mathbf { x } _ { t } = \mathbf { x } _ { t } ^ { 0 }$，并通过仿射重参数化间接预测噪声 $\epsilon ^ { k _ { t } }$ [30]。因此，我们可以直接使用常规扩散训练目标训练（因果）扩散强制。我们通过噪声预测 $\epsilon _ { \theta } ( \mathbf { z } _ { t - 1 } , \mathbf { x } _ { t } ^ { k _ { t } } , k _ { t } )$ 对上述单元进行参数化。然后，我们通过最小化损失来寻找参数 $\theta$，其中我们从 $[ K ] ^ { T }$ 中均匀采样 $k _ { 1 : T }$，从我们的训练数据中采样 $\mathbf { x } _ { 1 : T }$，且 $\epsilon _ { t } \sim \mathcal { N } ( 0 , \sigma _ { k _ { t } } ^ { 2 } I )$，符合正向扩散过程（见算法 1 的伪代码）。重要的是，损失 (3.1) 捕捉了贝叶斯滤波和条件扩散的基本元素。在附录 D.1 中，我们进一步重新推导了扩散模型训练中用于扩散强制的常见技术，这对视频预测实验极为有用。在附录 B.2 中，我们讨论了均匀采样 $k _ { 1 : T }$ 的必要性。最后，我们在附录 A 中证明了以下定理 3.1 中非正式陈述的这一目标的有效性。

$$
\underset { \substack { k _ { t } , \mathbf { x } _ { t } , \epsilon _ { t } } } { \mathbb { E } } \sum _ { \substack { k _ { t } \sim p _ { \theta } ( \mathbf { z } _ { t } | \mathbf { z } _ { t - 1 } , \mathbf { x } _ { t } ^ { k _ { t } } , k _ { t } ) } } ^ { T } \bigg [ \| \epsilon _ { t } - \epsilon _ { \theta } \big ( \mathbf { z } _ { t - 1 } , \mathbf { x } _ { t } ^ { k _ { t } } , k _ { t } \big ) \| ^ { 2 } \bigg ] ,
$$

定理 3.1（非正式）。扩散强制训练程序（算法 1）优化期望对数似然的证据下界（ELBO）的重加权，即 $\ln p _ { \pmb { \theta } } \big ( \big ( \mathbf { x } _ { t } ^ { k _ { t } } \big ) _ { 1 \leq t \leq T } \big )$，其中期望是对噪声水平 $k _ { 1 : T } \sim [ K ] ^ { T }$ 取平均，并且 $\mathbf { x } _ { t } ^ { k _ { t } }$ 是根据正向过程加噪声的。此外，在适当条件下，优化 (3.1) 也同时最大化所有噪声水平序列的似然下界。我们指出“所有噪声水平序列”的特例是 $k _ { t } = 0$ 或 $k _ { t } = K$ 的情况；因此，可以遮蔽任何先前的词元，扩散强制将学习从正确的条件分布中采样，并建模训练集所有可能子序列的分布。

采样。扩散强制采样在算法 2 中描述，并通过在 2D $M \times T$ 网格 $\check { \kappa } \in [ \check { K } ] ^ { M \times T }$ 上规定噪声调度来定义；列对应于时间步 $t$，而索引为 $m$ 的行决定噪声级别。$\mathcal { K } _ { m , t }$ 表示行 $m$ 中时间步 $t$ 词元的期望噪声级别。为了生成长度为 $T$ 的完整序列，初始化词元 $\mathbf { x } _ { 1 : T }$ 为白噪声，对应于噪声级别 $k = K$。我们逐行遍历网格，按列从左到右进行去噪，直至达到 $\kappa$ 中规定的噪声级别。在最后一行 $m = 0$，词元是干净的，即它们的噪声级别为 ${ \cal K } _ { 0 , t } \equiv 0$。附录 D.5 讨论了该方案的边缘情况；超参数 $\left( { \alpha _ { k } , \bar { \alpha } _ { k } , \sigma _ { k } } \right)$ 设置为其标准值 [30]。矩阵 $\kappa$ 指定了在序列扩散的每一步中每个词元去噪的速度。由于扩散强制是训练用来去噪所有噪声级别序列的词元，因此可以灵活设计 $\kappa$ 以实现不同的行为，而无需重新训练模型。

# 3.3 序列生成中的新能力

我们现在来解释这一灵活采样范式所提供的新功能。

![](images/3.jpg)

稳定自回归生成。对于高维连续序列，如视频，自回归架构已知在训练范围之外采样时容易发散。相比之下，扩散强迫能够稳定地生成超出训练序列长度的长序列，通过使用与某些小噪声水平 $0 < k \ll K$ 相关的稍微“噪声词元”更新潜变量。我们的实验（第4.1节）展示了在长时间生成能力上显著的改进；附录B.4提供了进一步的直观理解。

保持未来的不确定性。从一系列白噪声词元 $[ \mathbf { x } _ { 1 } ^ { K } , \mathbf { x } _ { 2 } ^ { K } , \mathbf { x } _ { 3 } ^ { K } ] ^ { \top }$ 开始，$[ \mathbf { x } _ { 1 } ^ { 0 } , \mathbf { x } _ { 2 } ^ { K / 2 } , \mathbf { x } _ { 3 } ^ { K } ] ^ { \top }$，然后是 $[ \mathbf { x } _ { 1 } ^ { 0 } , \mathbf { x } _ { 2 } ^ { 0 } , \mathbf { x } _ { 3 } ^ { K / 2 } ] ^ { \top }$ 和 $[ \mathbf { x } _ { 1 } ^ { 0 } , \mathbf { x } _ { 2 } ^ { 0 } , \mathbf { x } _ { 3 } ^ { 0 } ] ^ { \top }$ 不确定性，这种“之”字形采样方案直观上将近期的未来编码为比远期未来更为确定。第 3.4 节描述了这如何导致更有效的序列引导。长时间跨度的引导。在算法 2 的第 10 行，可以对部分扩散轨迹 $\mathbf { x } _ { 1 : T }$ 添加引导，正如第 2 节所述。由于未来词元对过去的依赖，来自未来词元的引导梯度可以在时间上向后传播。扩散强制的独特优势在于，因为我们可以在未完全扩散过去的情况下扩散未来词元，因此梯度引导过去词元的采样，从而实现长时间跨度的引导，同时遵循因果关系。我们在附录 B.3 中详细阐述了实施细节。如第 4.2 节所示，以这种方式进行规划显著优于引导的全序列扩散模型。

# 3.4 适应性序列决策的扩散强制方法

扩散强制所提供的能力激励我们提出了用于序列决策的创新框架，其关键应用于机器人技术和自主智能体。考虑一个由环境定义的马尔可夫决策过程，其中包含动态 $p ( \mathbf { s } _ { t + 1 } | \mathbf { s } _ { t } , \mathbf { a } _ { t } )$，观察 $p ( \mathbf { o } _ { t } | \mathbf { s } _ { t } )$ 和奖励 $p ( \mathbf { r } _ { t } | \mathbf { s } _ { t } , \mathbf { a } _ { t } )$。我们的目标是训练一个策略 $\pi ( \mathbf { a } _ { t } | \mathbf { o } _ { 1 : t } )$，使得轨迹的期望累积奖励 $\mathbb { E } [ \sum _ { t = 1 } ^ { T } \mathbf { r } _ { t } ]$ 最大化。我们将令牌定义为 $\mathbf x _ { t } = \left[ \mathbf a _ { t } , \mathbf r _ { t } , \mathbf o _ { t + 1 } \right]$。轨迹是一个序列 $\mathbf { x } _ { 1 : T }$，可能具有可变长度；训练过程按照算法1进行。在每个执行步骤$t$中，过去的（无噪声）令牌 $\mathbf { x } _ { 1 : t - 1 }$ 被一个潜在变量 $\mathbf { z } _ { t - 1 }$ 总结。在此潜在变量的条件下，我们通过算法2采样出一个计划 $\hat { \mathbf { x } } _ { t : t + H }$，其中 $\hat { \mathbf { x } } _ { t } = [ \hat { \mathbf { a } } _ { t } , \hat { \mathbf { r } } _ { t } , \hat { \mathbf { o } } _ { t + 1 } ] ^ { \top }$ 包含预测的动作、奖励和观察。$H$ 是一个前瞻窗口，类似于模型预测控制中的未来预测 [20]。在采取计划动作 $\hat { \mathbf { a } } _ { t }$ 后，环境生成奖励 $\mathbf { r } _ { t }$ 和下一个观察 $\mathbf { o } _ { t + 1 }$，形成下一个令牌 $\mathbf { x } _ { t } = \left[ \hat { \mathbf { a } } _ { t } , \mathbf { r } _ { t } , \mathbf { o } _ { t + 1 } \right] ^ { \top }$。潜在变量根据后验分布 $p _ { \theta } ( \mathbf { z } _ { t } | \mathbf { z } _ { t - 1 } , \mathbf { x } _ { t } , 0 )$ 更新。我们的框架使得其既可以作为策略也可以作为规划器：

![](images/4.jpg)  

Figure 3: Video Generation. Among tested methods, Diffusion Forcing generations are uniquely temporally consistent and do not diverge even when rolling out well past the training horizon. Please see the project website for video results.

灵活的计划视野。扩散引导（a）可以应用于变量视野的任务，因为每个新动作是顺序选择的，并且（b）其前瞻窗口 $H$ 可以缩短以降低延迟（将扩散引导作为策略使用），或者延长以进行长视野规划（通过下面描述的引导），而无需重新训练或修改架构。注意，对于全序列扩散模型，如Diffuser [37]，由于需要全轨迹生成，(a) 是不可能的，而扩散策略 [10] 需要固定的小前瞻大小，限制了 (b)。灵活的奖励引导。如附录B.3所述，扩散引导可以通过引导进行规划，使用 ${\sum_{t=1}^{T} \mathbf{r}_t}$ 替代 $\log c$ 的奖励，这些奖励指示目标完成 ${\bf \bar{\theta}} - \| \mathbf{o}_T - \mathbf{g} \|^2$。每个时间步的策略无法利用这种较长视野的引导。

蒙特卡罗引导（MCG），未来不确定性。因果扩散强制使我们能够通过对未来所有 $\mathbf{x}_{t+1:T}$ 分布的引导来影响一个词元 $\mathbf{x}_{t}^{k}$ 的生成。我们可以绘制多个未来样本并对它们的引导梯度进行平均，而不是只绘制一个轨迹样本来计算这个引导梯度。我们称之为蒙特卡罗引导。遵循所谓的射击方法，如MPPI [64]，$\mathbf{x}_{t}^{k}$ 的引导是基于所有未来结果的期望奖励，而不是某个特定结果。当结合采样计划（在去噪立即下一个词元时保持未来词元噪声水平较高，例如第3.3节中描述的锯齿形计划）时，MCG的效果得到了增强，考虑到未来更大的不确定性。附录B.5进一步论证了MCG的重要性，以及扩散强制如何独特地利用它。

# 4 实验

我们广泛评估了扩散强制作为生成序列模型在视频和时间序列预测、规划以及模仿学习等多种应用中的优点。数据集和可重复性详细信息请见附录，以及项目网站上的视频结果。

# 4.1 视频预测：一致稳定的序列生成与无限推演。

我们在Minecraft游戏视频和DMLab导航视频上训练了因果扩散强制的卷积递归神经网络实现，用于视频生成建模。在采样时，我们

Maze2d-medium-v1 Maze2d-large-v1 起始 结束 去噪步骤 去噪步骤 • 到 20 : o G 环境 MPPI CQL IQL Diffuser\* 有扩散动作的 Diffuser 我们的 不包含 MCG 的我们 Maze2D U-Maze 33.2 5.7 47.4 113.9 ± 3.1 6.3 ± 2.1 110.1 ± 3.9 116.7 ± 2.0 Maze2D 中等 10.2 5.0 34.9 121.5 ± 2.7 13.5±2.3 136.1 ± 10.2 149.4 ± 7.5 Maze2D 大 5.1 12.5 58.6 123.0 ± 6.4 6.3 ±2.1 142.8 ± 5.6 159.0 ± 2.7 单任务平均 16.2 7.7 47.0 119.5 8.7 129.67 141.7 Multi2D U-Maze 41.2 - 24.8 128.9 ± 1.8 32.8±1.7 107.7 ± 4.9 119.1 ± 4.0 Multi2D 中等 15.4 12.1 127.2 ± 3.4 22.0±2.7 145.6 ± 6.5 152.3 ± 9.9 Multi2D 大 8.0 13.9 132.1 ± 5.8 6.9 ±1.7 129.8 ± 1.5 167.1 ±2.7 多任务平均 21.5 - 16.9 129.4 20.6 127.7 146.2 进行自回归推演，使用第 3.3 节中提出的稳定化方法。我们考虑两个基线，均利用相同的 RNN 架构：一个是使用教师强迫训练的下一帧扩散基线 [65]，另一个是因果全序列扩散模型。图 3 展示了由扩散强迫和基线生成的推演的定性结果，这些推演开始于未见帧的两个数据集。虽然扩散强迫在其训练范围远超（例如 1000 帧）时仍能稳定推演，但教师强迫和全序列扩散基线则迅速发散。此外，在训练范围内，我们观察到全序列扩散经历帧间的不连续性，视频序列剧烈跳跃，而扩散强迫的推演则通过一致的 3D 环境展现自我运动。这凸显了扩散强迫在不累积误差的情况下稳定高维序列推演的能力。

# 4.2 扩散规划：多类生成，因果不确定性，灵活时域控制。

决策制定独特地受益于扩散引导的能力。我们在标准离线强化学习基准D4RL [18]上评估我们提出的决策制定框架。具体而言，我们在一组具有稀疏奖励的二维迷宫环境中对扩散引导进行基准测试。智能体的任务是从一个随机起始位置到达指定的目标位置。附录E.5中提供了环境的详细描述。该基准提供了在迷宫中随机行走的数据集（因此是随机的）。我们为每个迷宫训练一个模型。我们将所提出的决策制定框架与最先进的离线强化学习方法和最近推出的扩散规划框架Diffuser [37]进行基准测试。有关定性和定量结果，请参见图1：扩散引导在所有6个环境中均优于Diffuser和所有基准。蒙特卡洛引导的益处。强化学习问题的典型目标是找到能够最大化预期未来奖励的动作，我们通过MCG实现这一目标。全序列扩散模型如Diffuser不支持采样以最大化预期奖励，正如我们在附录B.5中正式推导的。为了理解MCG的重要性，我们在表1中进行了消融实验。去除MCG引导会降低我们的性能，尽管扩散引导在此情况下仍然保持竞争力。

![](images/5.jpg)  

Figure 4: In our real robot task, a robot arm is asked to swap the slots of two fruits using a third slot. Since the fruits are input in random slots at the beginning, one cannot determine the next steps from a single observation without knowledge of the initial placement of the fruits. As illustrated in (a) and (b), the upper observation is the same but the desired outcome illustrated below can vary—the task thus requires remembering the initial configuration. In addition, as shown in (c), the same model that generates actions also synthesizes realistic video from just a single frame.

因果建模的优势。与纯生成建模不同，顺序决策需要采取行动并接收反馈。由于不确定性的叠加，紧接着的行动比遥远未来的行动更为重要。尽管Diffuser及后续模型被训练来生成行动-奖励-状态元组$\left[ \mathbf { a } _ { t } , \mathbf { r } _ { t } , \mathbf { o } _ { t } \right]$的序列，但直接执行这些行动将导致轨迹与生成的状态显著偏离。换句话说，生成的状态和行动之间缺乏因果一致性。为了解决这一缺陷，Diffuser的实现忽略了生成的行动，而是依赖手工设计的PD控制器从生成的状态推断行动。在表1中，我们可以看到，Diffuser在直接执行生成行动时性能显著下降。相较之下，Diffusion Forcing的原始行动生成是自洽的，甚至超越了将Diffuser的状态预测与手工设计的PD控制器相结合所选出的行动。 灵活时间窗口的优势。许多强化学习任务具有固定的时间窗口，这要求在智能体在任务中取得进展时，规划时间窗口需逐渐缩小。Diffusion Forcing通过设计实现了这一点，而像Diffuser这样的完整序列模型，即使稍加调整也表现不佳，具体情况在附录B.6中进行了说明。

# 4.3 可控序列组合生成

我们展示了仅通过修改采样方案，可以灵活构建在训练时观察到的序列的子序列。我们考虑一个在二维方形平面上的轨迹数据集，其中所有轨迹从一个角落开始并结束于对角的角落，形成一个十字形。如图1所示，当不需要组合行为时，可以让DF保持完整的记忆，复制十字形分布。当需要组合性时，可以让模型在不使用记忆的情况下生成较短的计划，使用MPC，导致十字的子轨迹拼接，形成一个V形轨迹。由于篇幅有限，我们将结果推迟到附录E.2中。

# 4.4 机器人学：长时间域模仿学习与鲁棒视觉运动控制

最后，我们展示了扩散驱动（Diffusion Forcing, DF）在现实世界机器人视觉运动控制中开辟了新的机遇。模仿学习是一种流行的机器人操作技术，通过专家演示学习观察到的行动映射。然而，缺乏记忆往往使模仿学习无法完成长时间的任务。DF不仅缓解了这一缺陷，还提供了一种使模仿学习变得稳健的方法。带记忆的模仿学习。我们通过远程操作Franka机器人收集了一组视频和动作数据。在选择的任务中，需要使用一个第三位置来交换一个苹果和一个橙子的位置。请参见图4以获取说明。水果的初始位置是随机的，因此有两种可能的目标状态。如图4所示，当一个水果位于第三个位置时，无法从当前观察中推断出期望的结果——策略必须记住初始配置，以确定移动哪个水果。与常见的行为克隆方法相比，DF自然地将记忆纳入其潜在状态。我们发现DF的成功率达到了80%，而扩散政策（diffusion policy），一种没有记忆的最先进模仿学习算法，则失败了。对于缺失或噪声观察的鲁棒性。由于它结合了贝叶斯过滤的原理，扩散驱动能够在对噪声或缺失观察鲁棒的情况下执行模仿学习。我们通过在执行过程中添加视觉干扰，甚至完全遮挡摄像头来证明这一点。DF允许我们通过使用 $k > 0$ ，轻松地将这些观察标记为“噪声”，在这种情况下，DF主要依赖其先验模型来预测动作。因此，成功率仅下降了4%，降至76%。相比之下，下一帧扩散模型基线的成功率仅为48%：它必须将扰动观察视为真实数据，并遭受分布外的误差。视频预训练的潜力。最后，图4还表明，扩散驱动能够仅基于初始帧生成机器人的任务执行视频，这统一了扩散政策/模仿学习和视频生成建模，为在未标记视频上进行预训练铺平了道路。

# 4.5 时间序列预测：扩散强迫是一个良好的通用序列模型

在附录E中，我们展示了DF在多变量时间序列预测中，与之前的扩散模型[50]和基于变换器的模型[51]具有竞争性，遵循了实验设置[54]。

# 5 讨论

限制性。我们当前的因果实现基于递归神经网络（RNN）。应用于更高分辨率的视频或更复杂的分布，可能需要遵循附录B.1中的指令的大型变换模型。我们没有研究扩散强化在互联网规模的数据集和任务上的扩展行为。结论。在本文中，我们介绍了扩散强化，这是一种新的训练范式，其中模型被训练以去噪独立的、每个词元具有不同噪声水平的词元集合。应用于时间序列数据，我们展示了如何利用扩散强化训练的下一个词元预测模型结合了下一个词元模型和全序列扩散模型的优势。我们引入了新的采样和引导方案，当应用于序列决策任务时，能够显著提高性能。未来的工作可能会探索将扩散强化应用于时间序列生成建模以外的领域，并将扩散强化扩展到更大的数据集。致谢。本工作得到了国家科学基金会（Grant No. 2211259）、新加坡国防科技局（DST000ECI20300823，针对标签高效视觉的3D自监督学习）、情报高级研究项目活动（IARPA）通过内政部/内部业务中心（DOI/IBC）资助（140D0423C0075）以及亚马逊科学中心的支持。

# References

[ A. Ajay, Y. Du, A. Gupt, J. Tenbaum, T. Jaakola, and P. Arawal. Is conditial eneativemoe all you need for decision-making? arXiv preprint arXiv:2211.15657, 2022.   
[2] A. Alexandrov, K. Benidis, M. Bohlke-Schneider, V. Flunkert, J. Gasthaus, T. Januschowski, D. C. Maddix, S. Rangapuram, D. Salinas, J. Schulz, L. Stella, A. C. Türkmen, and Y. Wang. Gluonts: Probabilistic and neural time series modeling in python. Journal of Machine Learning Research, 21(116):16, 2020.   
[3C. Berer, G. Brockman, B.Chan, V. Cheung, P. Debiak, C. Dennison, D. Farhi, Q. Fischer, S. Hashme, C. Hesse, R. Józefowicz, S. Gray, C. Olsson, J. Pachocki, M. Petrov, H. P. de Oliveira Pinto, J. Raiman, T.Salimans, J.Schlatter, J.Schnider, S. Sidor, I. Sutskever, J. Tan, F. Wolski, andS.Zhang. Dota  wih large scale deep reinforcement learning. CoRR, abs/1912.06680, 2019.   
[4 A. Blattan, T. Dockhor, S. Kulal, D.Mendlevitch, M.Kilian, D. Lorenz, Y. Levi, Z. English, V.Vole, A. Letts, V. Jampani, and R. Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets, 2023.   
[5 A. Block, A. Jadbabaie, D. Pfrommer, M. Simchowitz, and R. Tedrake. Provable guarantees for generative behavior cloning: Bridging low-level stability and high-level behavior. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.   
[ T.B. Brown, B. Man, N. Ryer, M.Subbih, J. Kaplan, P. Dharial, A. Neelakantan, P. Shyam, G. Sasry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei. Language models are few-shot learners. CoRR, abs/2005.14165, 2020.   
[7] S. H. Chan. Tutorial on diffusion models for imaging and vision. arXiv preprint arXiv:2403.18103, 2024.   
[8] M. Chen, A. Radford, R. Child, J. Wu, H. Jun, D. Luan, and I. Sutskever. Generative pretraining from pixels. In International conference on machine learning, pages 16911703. PMLR, 2020.   
[9] T. Chen. On the importance of noise scheduling for diffusion models, 2023.   
[0] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion, 2024.   
[11] K. Cho, B. van Merrienboer, Ç. Gülçehre, F. Bougares, H. Schwenk, and Y. Bengio. Learning phrase representations using RNN encoder-decoder for statistical machine translation. CoRR, abs/1406.1078, 2014.   
[2 J.Chug, K. Kaster, L. Dinh, K. Goel, A.C. Courville, and Y. Bengi.A recet atent varableel for sequential data. Advances in neural information processing systems, 28, 2015.   
[13] J. Cohen, E. Rosenfeld, and Z. Kolter. Certified adversarial robustness via randomized smoothing. In international conference on machine learning, pages 13101320. PMLR, 2019.   
[14] E. de Bézenac, S. S. Rangapuram, K. Benidis, M. Bohlke-Schneider, R. Kurle, L. Stella, H. Hasson, P. Gallinari, and T. Januschowski. Normalizing kalman filters for multivariate time series analysis. In Advances in Neural Information Processing Systems, volume 33, 2020.   
[1 J. Devln, M. Chang, K. Lee, and K. Toutanova. BERT: pre-trainn o deep bidirectonal transrmes for language understanding. CoRR, abs/1810.04805, 2018.   
[16] P. Dhariwal and A. Nichol. Diffusion models beat gans on image synthesis. CoRR, abs/2105.05233, 2021.   
[ C. Feichtenhoer, Y.Li, K. He,  al. Maske autcoders as spatiotemporal earners. Advances in nural information processing systems, 35:3594635958, 2022.   
[18 J. Fu, A. Kumar, O. Nachum, G. Tucker, and S. Levine. DRL: datasets for deep data-driven reinforcement learning. CoRR, abs/2004.07219, 2020.   
[9] S. Go, P. Zhou, M.-M.Cheng, and S. Yan. Masked diffusion tranormer is a strong image synthesizer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2316423173, 2023.   
[20] C. E. Garcia, D. M. Prett, and M. Morari. Model predictive control: Theory and practice—a survey. Automatica, 25(3):335348, 1989.   
[1] F. Gers, J. Schmidhuber, and F. Cummins. Learning to forget: continual prediction with lstm. In 1999 Ninth International Conference on Artificial Neural Networks ICANN 99. (onf. Publ. No. 470), volume 2, pages 850855 vol.2, 1999.   
[2] D. Hafner, T. P. Lillicrap, J. Ba, and M. Norouzi. Dream to control: Learning behaviors by latent imagination. CoRR, abs/1912.01603, 2019.   
[D. H T. P.Lilicp, I.Fiser,R.Villeas, D. Ha, H. Lee, and J. Davi. Lear t for planning from pixels. CoRR, abs/1811.04551, 2018.   
[24] T. Hang, S. Gu, C. Li, J. Bao, D. Chen, H. Hu, X. Geng, and B. Guo. Efficient diffusion training via min-snr weighting strategy, 2024.   
[25] N. Hansen, X. Wang, and H. Su. Temporal difference learning for model predictive control, 2022.   
Hary, ..M  Wei,n W.Fe videos. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 2795327965. Curran Associates, Inc., 2022.   
[27] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick. Masked autoencoders are scalable vision laners. In Proceding of the IEEE/CV conferenceon computer vision and patten recognition, pages 1600016009, 2022.   
[28] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. CoRR, abs/1512.03385, 2015.   
[9]J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. Advances in Neural Informatin Processing Systems (NeurIPS), 33:68406851, 2020.   
[30] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. CoRR, abs/2006.11239, 2020.   
[31] J. Ho and T. Salimans. Classifier-free diffusion guidance, 2022.   
[32] J. Ho, T. Salimans, A. Gritsenko, W. Chan, M. Norouzi, and D. J. Fleet. Video diffusion models, 2022.   
[33] S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural Comput., 9(8):17351780, nov 1997.   
[34] A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, and G. Corrado. Gaia-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023. In  C and Pattern Recognition, pages 1675016761, 2023.   
[36] R. Hyndman, A. B. Koehler, J. K. Ord, and R. D. Snyder. Forecasting with Exponential Smoothing: The State Space Approach. Springer Science & Business Media, 2008.   
[7 M. Janer, Y. Du, J. B. Teebau, nd S. Levie. ang wih diffuin or fexile behavir neis. Proceedings of the International Conference on Machine Learning (ICML), 2022.   
[38] A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. CoRR, abs/2006.16236, 2020.   
[ L. e, J. Wag, T.Bhaare, B.Bots, an Srivasa.Graspg wit copsics Combat coe shift in model-free imitation learning for fine manipulation. In 2021 IEEE International Conference on Robotics and Automation (ICRA), pages 61856191. IEEE, 2021.   
[40] D. Kingma and R. Gao. Understanding diffusion objectives as the elbo with simple data augmentation. Advances in Neural Information Processing Systems, 36, 2024.   
[41] R. G. Krishnan, U. Shalit, and D. Sontag. Structured inference networks for nonlinear state space models. In AAAI, 2017.   
[42] G. Lai, W. Chang, Y. Yang, and H. Liu. Modeling long- and short-term temporal patterns with deep neural networks. CoRR, abs/1703.07015, 2017.   
. J. .x. .   . In Conference on robot learning, pages 143156. PMLR, 2017.   
[ X.L. Li, J.Ti, I.Gul .L n . B. Hashgeneration, 2022.   
[45] H. Lütkepohl. New Introduction to Multiple Time Series Analysis. Springer Science & Business Media, 2005.   
[ J. E. Mahend . L. W. So le ort pobably strius an Science, 22(10):10871096, 1976.   
[47] A. Nichol and P. Dhariwal. Improved denoising diffusion probabilistic models. CoRR, abs/2102.09672, 2021.   
[.P, .Ae, Q.yA.Abak, .r .Bi, H., X.C ., L. Derczynski, X. Du, M. Grella, K. Gv, X. He, H. Hou, P. Kazienko, J. Kocon, J. Kong, B. Koptyra, H.La J. Lin K.S.I. M FM A.Sai G. Sg, X. Ta J. W S.  Z. Q. Zhou, J. Zhu, and R.-J. Zhu. RWKV: Reinventing RNNs for the transformer era. In H. Bouamor, J. Pino, and K. Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 1404814077, Singapore, Dec. 2023. Association for Computational Linguistics.   
[ C. Rafe N.Shaz A. Rober, K.Le, S. Nara, M. Mat, . Zho,W. Li, ad P. J.Liu.Exp the limits of transfer learning with a unified text-to-text transformer. CoRR, abs/1910.10683, 2019.   
[50] K. Rasul, C. Seward, I. Schuster, and R. Vollgraf. Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, 2021.   
[. Rasul, A.S. Shei, I. Schuster, U.M.Bera, andR. Volgra. Multivariae probablisicime e forecasting via conditioned normalizing flows. In International Conference on Learning Representations, 2021.   
[52] D. Ruhe, J. Heek, T. Salimans, and E. Hoogeboom. Rolling diffusion models. arXiv preprint arXiv:2402.09470, 2024.   
[3] T. Salimans and J. Ho. Progressive distillation for fast sampling of diffusion models. CoRR, abs/2202.00512, 2022.   
[54] D. Salinas, M. Bohlke-Schneider, L. Callot, R. Medico, J. Gasthaus, and R. Medico. High-dimensional multivariate forecasting with low-rank gaussian copula processes. In NeurIPS, 2019.   
[5 D.Salias, V.Flunkert, J. Gasthaus, and T. Januhowski. Deepar: Probabilistc forecastig with auoegressive recurrent networks. International Journal of Forecasting, 36(3):11811191, 2020.   
[ .Salinas, V. Flunker, J. Gasthaus, and T. Januhowski. Deepar: robabilistic forecasting with auoregressive recurrent networks. International Journal of Forecasting, 36(3):11811191, 2020.   
[ J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, an . Ganu. Deeunsupervd learng u nonequilibrium thermodynamics. In Proceedings of the International Conference on Machine Learning (ICML), 2015.   
[58] J. Song, C. Meng, and S. Ermon. Denoising diffusion implicit models. CoRR, abs/2010.02502, 2020.   
[59] B. Tang and D. S. Matteson. Probabilistic transformer for time series analysis. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. W. Vaughan, editors, Advances in Neural Information Processing Systems, 2021.   
[0 H. Touvrn, P. Bojaowski, M.Caron, M.Cor A. E-Nouby, E.Grave, A. Joulin, G. Synve, J.Vrbek, and H. Jégou. Resmlp: Feedforward networks for image classification with data-efficient training. CoRR, abs/2105.03404, 2021.   
[A. Van den Oord, N. Kalchbrenner, L. Espeholt, O. Vinyals, A. Graves, et al. Conditional image generation with pixelcnn decoders. Advances in neural information processing systems, 29, 2016.   
[62] R. van der Weide. Go-garch: A multivariate generalized orthogonal garch model. Journal of Applied Econometrics, 17(5):549564, 2002.   
[3]C. Wei, K. Mangalam, P.-Y. Huan, Y. Li, H. Fan, H. Xu, H. Wang, C. Xie, A. Yuile, and C. Feichteor. Diffusion models as masked autoencoders. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1628416294, 2023.   
[64] G. Williams, A. Aldrich, and E. Theodorou. Model predictive path integral control using covariance variable importance sampling. arXiv preprint arXiv:1509.01149, 2015.   
[65] R. J. Williams and D. Zipser. A Learning Algorithm for Continually Running Fully Recurrent Neural Networks. Neural Computation, 1(2):270280, 06 1989.   
[ . Wu Z.Fan X.LiuY.Go .Shen, J. Jiao, H.T. Ze, J. Li, Z. Wei J.Guo, . nd W.. Ar-diffusion: Auto-regressive diffusion model for text generation, 2023.   
Wu Z.Fan X.Liu H.T. Ze . J. Jio, J.Li J.Guo .D W.he  .Ar-if: Auto-regressive diffusion model for text generation. Advances in Neural Information Processing Systems, 36:3995739974, 2023.   
[68] T. Yan, H. Zhang, T. Zhou, Y. Zhan, and Y. Xia. Scoregrad: Multivariate probabilistic time series forecasting with continuous energy-based generative models, 2021.   
[9] W. Yan, D. Hafner, S. James, and P. Abbeel. Temporally consistent transformers for video generation, 2023.   
[70] R. Yang, P. Srivastava, and S. Mandt. Diffusion probabilistic modeling for video generation. Entropy, 25(10):1469, 2023.   
.Yao D.Yu, JZao, Sa T.L.GriY.Cao an. ara.Tree o:Del problem solving with large language models, 2023.   
[72] H. Yu, N. Rao, and I. S. Dhillon. Temporal regularized matrix factorization. CoRR, abs/1509.08333, 2015.

# A Theoretical Justification

I ti c e roihel stathea i FrTei be summarized as follows:

•We show that our training methods optimize a reweighting of the Evidence Lower Bound (ELBO) on the average log-likelihood of our data. We first establish this in full generality (Theorem A.1), and then specialize to the form of Gaussian diffusion (Corollary A.2). We show that the resulting terms decouple in such a fashion that, in the limit of a fully expressive latent and model, makes the reweighting terms immaterial.   
•We show that the expected likelihood over any distribution over sequences of noise levels can be lower bounded by a sum over nonnegative terms which, when reweighted, correspond to the terms optimized in the Diffusion Forcing training objective maximizes. Thus, for a fully expressive network that can drive all terms to their minimal value, Diffusion Forcing optimizes a valid surrogate of the likelihood of all sequences of noise levels simultaneously.

We begin by stating an ELBO for general Markov forward processes $q ( \cdot )$ , and generative models $p _ { \theta } ( \cdot )$ , and then specialize to Gaussian diffusion, thereby recovering our loss. We denote our Markov forward process $q ( \cdot )$ as

$$
q ( \mathbf { x } ^ { 1 : K } \mid \mathbf { x } ^ { 0 } ) = \prod _ { k = 1 } ^ { K } q ( \mathbf { x } ^ { k } \mid \mathbf { x } ^ { k - 1 } ) ,
$$

and a parameterized probability model

$$
p _ { \pmb { \theta } } \big ( \big ( ( \mathbf { x } _ { t } ^ { k } ) _ { 1 \leq k \leq K } , \mathbf { z } _ { t } \big ) _ { t \geq 1 } \big )
$$

We assume that $p _ { \theta }$ satisfies the Markov property that

$$
p _ { \theta } ( \mathbf { z } _ { t } , \mathbf { x } _ { t } ^ { k _ { t } } \mid \mathbf { z } _ { 1 : t - 1 } , ( \mathbf { x } _ { s } ^ { k _ { s } } ) _ { 1 \leq s < t } ) = p _ { \theta } ( \mathbf { z } _ { t } , \mathbf { x } ^ { k _ { t } } \mid \mathbf { z } _ { t - 1 } )
$$

that is, the latent codes $\mathbf { z } _ { t - 1 }$ is a sufficient statistic for $\mathbf { x } ^ { k _ { t } }$ given the history. We say that $p _ { \theta }$ has deterministic latents if $p _ { \theta } ( \mathbf { z } _ { t } \mid \mathbf { z } _ { 1 : t - 1 } , ( \mathbf { x } _ { s } ^ { k _ { s } } ) _ { 1 \leq s < t } , \mathbf { x } _ { t } ^ { k _ { t } } )$ is a Dirac delta.

Remark 1. In order for $p _ { \theta }$ to have deterministic latents and correspond to a valid probability distribution, we need to view the latents $\mathbf { z } _ { t }$ not as individual variables, but as a collection of variables $\bar { \mathbf { z } } _ { t } ( k _ { 1 : t } )$ indexed by $t \in [ T ]$ a e $k _ { 1 : t } \in \{ 0 , 1 , \ldots , K \} ^ { t }$ In tis cae, siply setting $\mathbf { z } _ { t } ( k _ { 1 : t } ) = ( k _ { 1 : t } , ( \mathbf { x } _ { s } ^ { k _ { s } } ) _ { 1 \leq s \leq t }$ o therwise, tautologically produces deterministic latents. The reason for indexing $p _ { \pmb { \theta } } \big ( \mathbf { z } _ { t } \mid \big ( ( \mathbf { x } _ { s } ^ { k _ { s } } \big ) _ { 1 \leq s \leq t } , ( \mathbf { x } _ { s } ^ { k _ { s } ^ { \prime } } \big ) _ { 1 \leq s \leq t } \big )$ would be l lfined unless $\mathbf { z } _ { t } ( k _ { 1 : t } )$ $k _ { s } = k _ { s } ^ { \prime }$ with for  alt then arises because, $1 \leq s \leq t$ athus, $p _ { \pmb { \theta } }$ would not correspond to a joint probability measure. The exposition and theorem that follows allow $\mathbf { z } _ { t } ( k _ { 1 : t } )$ to be indexed on past noise levels $k _ { 1 : t }$ but suppresses dependence on $k _ { 1 : t }$ to avoid notational confusion.

# A.1 Main Results

We can now state our main theorem, which provides an evidence lower bound (ELBO) on the expected loglikelihood of partially-noised sequences $( \mathbf { x } _ { t } ^ { k _ { t } } ) _ { 1 \leq t \leq T }$ , under uniformly sampled levels $k _ { t }$ and $\mathbf { x } _ { t } ^ { k _ { t } }$ obtained by noising according to $q ( \cdot )$ $q ( \cdot )$ r $p _ { \theta }$ , but we will specialize to Gaussian diffusion in the following section.

Theorem A.1. $F i x { \bf x } _ { 1 : T } ^ { 0 }$ .Define the expectation over the forward process with random noise level $k _ { 1 : T }$ as

$$
\underline { { \mathbb { E } } } _ { \mathrm { r w a r d } } [ \cdot ] : = \mathbb { E } \underset { k _ { 1 } , \ldots , k _ { T } \underset { \sim } { \mathrm { u n i f } } [ K ] } { \mathbb { E } } \times \underset { s } { \mathbb { E } } { \mathbb { E } } _ { s } \sim q ( \mathbf { x } _ { s } ^ { k _ { s } } \mid \mathbf { x } _ { s } ^ { 0 } ) , 1 \leq s \leq T  ( \cdot ] ,
$$

and the expectation over the latents under $p _ { \theta } ( \cdot )$ conditioned on $k _ { 1 : T } , ( \mathbf { x } _ { s } ^ { k _ { t } } ) _ { 1 \leq t \leq T }$ as

$$
\underset { p , \mathbf { z } _ { 1 : T } } { \mathbb { E } } [ \cdot ] : = \underset { \mathbf { z } _ { s } \sim p ( \mathbf { z } _ { s } \mid \mathbf { z } _ { s - 1 } , \mathbf { x } _ { s } ^ { k _ { s } } ) , s \leq T } { \mathbb { E } } \Big [ \cdot \mid k _ { 1 : T } , ( \mathbf { x } _ { t } ^ { k _ { t } } ) _ { 1 \leq t \leq T } \Big ]
$$

Then, as long as pe satisfies the Markov property,

$$
\begin{array} { l } { { \displaystyle \mathop { \mathbb { E } } _ { \mathrm { o r s a r d } } ^ { \mathbb { E } } \| { \bf n } p _ { \theta } \big ( ( { \bf x } _ { t } ^ { k _ { t } } ) _ { 1 \le t \le T } \big ) \big \| \ge C ( { \bf x } _ { 1 : T } ^ { 0 } ) } } \\ { { \displaystyle + \mathop { \mathbb { E } } _ { \mathrm { f o r s a r d } } p _ { , { \bf z } _ { 1 : T } } \left[ \sum _ { t = 1 } ^ { T } \left( \frac { 1 } { K + 1 } \ln p _ { \theta } \big ( { \bf x } _ { t } ^ { 0 } \mid { \bf x } _ { t } ^ { 1 } , { \bf z } _ { t - 1 } \big ) + \sum _ { j = 2 } ^ { K } \frac { j } { K + 1 } \mathrm { D } _ { \mathbb { S } \mathbb { L } } \left( q \big ( { \bf x } _ { t } ^ { j - 1 } \mid { \bf x } _ { t } ^ { j } , { \bf x } _ { t } ^ { 0 } \big ) \big \| p _ { \theta } \big ( { \bf x } _ { t } ^ { j } \mid { \bf x } _ { t } ^ { j - 1 } , { \bf z } _ { t - 1 } \big ) \right) \right) \right] , } } \end{array}
$$

where $C ( \mathbf { x } _ { 1 : T } ^ { 0 } )$ is a constant depending only on $\mathbf { x } _ { 1 : T } ^ { 0 }$ (the unnoised data). Moreover, if the latents are deterministic (i.e. $p _ { \pmb { \theta } } ( \mathbf { z } _ { t } \mid \mathbf { z } _ { t - 1 } , \mathbf { x } _ { t } ^ { k _ { t } } )$ $q ( \mathbf { x } _ { t } ^ { k _ { t } + 1 : T } \mid \mathbf { x } _ { t } ^ { k _ { t } } ) \equiv p _ { \theta } ( \mathbf { x } _ { t } ^ { k _ { t } + 1 : T } \mid \mathbf { x } _ { t } ^ { k _ { t } } , \mathbf { z } _ { t - 1 } )$

The proof o the above theorem is given in Appendix A.. Remarkably, it involves only two inequalities! The frst hol wi eualyndeeatentanhe hol n nval is exact: $q ( \mathbf { x } _ { t } ^ { k _ { t } + 1 : T } \mid \mathbf { x } _ { t } ^ { k _ { t } } ) \equiv p _ { \theta } ( \mathbf { x } _ { t } ^ { k _ { t } + 1 : T } \mid \mathbf { x } _ { t } ^ { k _ { t } } , \mathbf { z } _ { t - 1 } )$ T   B h in Theorem A.1 is a relatively strong surrogate objective for optimizing the likelihoods.

# A.1.1 Specializing to Gaussian diffusion

We now special Theorem A.1 to Gaussian diffusion. For now, we focus on the " $\mathbf { x }$ -prediction" formulation of diffusion, which is the one used in our implementation. The " $\epsilon$ -prediction" formalism, used throughout the theorem follows directly by apply standard likelihood and KL-divergence computations for the DDPM [29, 7] to Theorem A.1.

# Corollary A.2. Let

$$
q ( \mathbf { x } ^ { k + 1 } \mid \mathbf { x } _ { t } ^ { k } ) = \mathcal { N } ( \mathbf { x } ^ { k } ; \sqrt { 1 - \beta _ { k } } \mathbf { x } ^ { k - 1 } , \beta _ { k } \mathbf { I } ) ,
$$

and define $\alpha _ { k } ~ = ~ ( 1 - \beta _ { k } )$ $\begin{array} { r } { \bar { \alpha } _ { k } \ = \ \prod _ { j = 1 } ^ { k } \alpha _ { j } } \end{array}$ Suppose that we parametrize $\begin{array} { r l } { p _ { \pmb { \theta } } ( \mathbf { x } _ { t } ^ { j } } & { { } | ~ \mathbf { x } _ { t } ^ { j + 1 } , \mathbf { z } _ { t - 1 } ) ~ = } \end{array}$ $\mathcal { N } ( \mu _ { \pmb { \theta } } ( \mathbf { x } _ { t } ^ { j + 1 } , \mathbf { z } _ { t - 1 } , j ) , \sigma _ { j } ^ { 2 } )$ , where further,

$$
\iota _ { \theta } \big ( \mathbf { x } _ { t } ^ { j } , \mathbf { z } _ { t - 1 } , j \big ) = \frac { \big ( 1 - \bar { \alpha } _ { j - 1 } \big ) \sqrt { \alpha _ { j } } } { 1 - \bar { \alpha } _ { j } } \mathbf { x } _ { t } ^ { j } + \frac { \big ( 1 - \alpha _ { j } \big ) \sqrt { \bar { \alpha } _ { j - 1 } } } { 1 - \bar { \alpha } _ { j } } \hat { \mathbf { x } } _ { \theta } \big ( \mathbf { x } _ { t } ^ { j } , \mathbf { z } _ { t - 1 } , j \big ) , \quad \sigma _ { j } ^ { 2 } : = \frac { \big ( 1 - \alpha _ { j } \big ) \big ( 1 - \sqrt { \bar { \alpha } _ { j - 1 } } \big ) } { 1 - \bar { \alpha } _ { j } } .
$$

Then, as long as pe satisfies the Markov property, we obtained

$$
\begin{array} { r } { \underset { \mathrm { r e w a r d } } { \mathbb { E } } \big [ \ln p _ { \theta } \big ( \big ( \mathbf { x } _ { t } ^ { k _ { t } } \big ) _ { 1 \le t \le T } \big ) \big ] + C ( \mathbf { x } _ { 1 : T } ^ { 0 } ) \ge \underset { \mathrm { f o r w a r d } } { \mathbb { E } } \underset { p , \mathbf { z } _ { 1 : T } } { \mathbb { E } } \left[ \displaystyle \sum _ { t = 1 } ^ { T } \frac { j } { K + 1 } \sum _ { j = 1 } ^ { K } c _ { j } \| \hat { \mathbf { x } } _ { \theta } ^ { 0 } ( \mathbf { x } _ { t } ^ { j } , \mathbf { z } _ { t - 1 } , j ) - \mathbf { x } _ { t } ^ { 0 } \| ^ { 2 } \right] } \\ { = \underset { \mathrm { f o r w a r d } } { \mathbb { E } } \underset { p , \mathbf { z } _ { 1 : T } } { \mathbb { E } } \left[ \displaystyle \sum _ { t = 1 } ^ { T } \mathbf { 1 } \big \{ k _ { t } \ge 1 \big \} \cdot k _ { t } c _ { k _ { t } } \| \big \hat { \mathbf { x } } _ { \theta } ^ { 0 } ( \mathbf { x } _ { t } ^ { k _ { t } } , \mathbf { z } _ { t - 1 } , k _ { t } ) - \mathbf { x } _ { t } ^ { 0 } \| ^ { 2 } \right] , } \end{array}
$$

where above, e define $\begin{array} { r } { c _ { j } = \frac { ( 1 - \alpha _ { j } ) ^ { 2 } \bar { \alpha } _ { j - 1 } } { 2 \sigma ^ { 2 } ( 1 - \bar { \alpha } _ { j } ) ^ { 2 } } } \end{array}$

Proof. The first inequality follows from the standard computations for the ' $\mathbf { \dot { x } }$ -prediction" formulation of Diffusion (see Section 2.7 of [7] and references therein). The second follows by replacing the sum over $j$ with an expectation over $k _ { t } \stackrel { \mathrm { u n i f } } { \sim } \{ 0 , 1 , \ldots , K \}$ . □

We make a couple of remarks:

•As noted above, Corollary A.2 can also be stated for $\epsilon$ -prediction, or the so-called "v-prediction" formalism, as all are affinely related.   
•Define an idealized latent $\tilde { \mathbf { z } } _ { t - 1 }$ consisting of all past tokens $\left( \mathbf { x } _ { t } ^ { k _ { t } } \right)$ as well as of their noise levels $k _ { t }$ .This is a sufficient statistic for $\mathbf { z } _ { t - 1 }$ , and thus we can always view $\hat { \mathbf { x } } _ { \pmb { \theta } } ^ { 0 } ( \mathbf { x } _ { t } ^ { k _ { t } } , \mathbf { z } _ { t - 1 } , k _ { t } ) =$ $\hat { \mathbf { x } } _ { \pmb { \theta } } ^ { 0 } \big ( \mathbf { x } _ { t } ^ { k _ { t } } , \bar { \mathbf { z } } _ { t - 1 } , k _ { t } \big )$ , where $\mathbf { z } _ { t - 1 }$ is just compressing $\bar { \mathbf { z } } _ { t - 1 }$ When applying the expectation of $\mathbf { x } _ { 1 : T } \sim q$ to both sides of the bound in Corollary A.2, and taking an infimum over possible function approximator $\hat { \mathbf { x } } _ { \theta } ^ { 0 }$ ,we obtain

$$
\begin{array} { r l } & { \underset { p _ { \theta } } { \operatorname* { i n f } } \mathbb { E } \underset { q \mathrm { ~ f o r w a r d } } { \mathbb { E } } \underset { p , \mathbf { z } _ { 1 : T } } { \mathbb { E } } \lVert \hat { \mathbf { x } } _ { \theta } ^ { 0 } ( \mathbf { x } _ { t } ^ { k _ { t } } , \mathbf { z } _ { t - 1 } , k _ { t } ) - \mathbf { x } _ { t } ^ { 0 } \rVert ^ { 2 } = \underset { p _ { \theta } } { \operatorname* { i n f } } \mathbb { E } \underset { q \mathrm { ~ f o r w a r d } } { \mathbb { E } } \underset { p , \mathbf { z } _ { 1 : T } } { \mathbb { E } } \lVert \hat { \mathbf { x } } _ { \theta } ^ { 0 } ( \mathbf { x } _ { t } ^ { k _ { t } } , \bar { \mathbf { z } } _ { t - 1 } ) - \mathbf { x } _ { t } ^ { 0 } \rVert ^ { 2 } } \\ & { \qquad = \mathbf { V a r } _ { q } [ \mathbf { x } _ { t } ^ { 0 } \mid ( \mathbf { x } _ { s } ^ { k _ { s } } ) _ { 1 \leq s \leq t } , k _ { 1 } , \dots , k _ { t } ] . } \end{array}
$$

This leads to a striking finding: with expressive enough latents and $p _ { \theta }$ , we can view the maximization of each term in Corollary A.2 separately across time steps. The absence of this coupling means that the weighting terms are immaterial to the optimization, and thus can be ignored.

•Given the above remarks, we can optimize the ELBO by taking gradients through the objective specified by Corollary A.2, and are free to drop any weighting terms (or rescale them) as desired. Backpropagation through $\mathbb { E } _ { p , \mathbf { z } _ { 1 : T } }$ is straightforward due to deterministic latents. This justifies the correctness of our training objective (3.1) and protocol Algorithm 1.

# A.1.2 Capturing all subsequences

Theorem A.1 stipulates that, up to reweighting, the Diffusion Forcing objective optimizes a valid ELBO on the expected log-likelihoods over uniformly sampled noise levels. The following theorem can be obtained by a straightforward modification of the proof of Theorem A.1 generalizes this to arbitrary (possibly temporally correlated) sequences of noise.

Theorem A.3. Let $\mathcal { D }$ be an arbitrarydistributionovr $[ K ] ^ { T }$ , and define $P t ( j \mid k _ { 1 : t - 1 } ) : = \operatorname* { P r } _ { \mathcal { D } } [ k _ { t } = j \mid k _ { 1 : t - 1 } ]$ . Fix $\mathbf { x } _ { 1 : T } ^ { 0 }$ . Define the expectation over the forward process with random noise level $k _ { 1 : T }$ as

$$
\underset { \mathrm { f o r w a r d } , \mathcal { D } } { \mathbb { E } } [ \cdot ] : = \underset { k _ { 1 } , \ldots , k _ { T } \sim \mathcal { D } } { \mathbb { E } } _ { \mathbf { x } _ { s } ^ { k _ { s } } \sim q ( \mathbf { x } _ { s } ^ { k _ { s } } \mid \mathbf { x } _ { s } ^ { 0 } ) , 1 \leq s \leq T } [ \cdot ] ,
$$

and the expectation over the latent under $p _ { \theta } ( \cdot )$ conditioned on $k _ { 1 : T } , ( \mathbf { x } _ { s } ^ { k _ { t } } ) _ { 1 \leq t \leq T }$ as

$$
\underset { p , \mathbf { z } _ { 1 : T } } { \mathbb { E } } [ \cdot ] : = \underset { \mathbf { z } _ { s } \sim p ( \mathbf { z } _ { s } \mid \mathbf { z } _ { s - 1 } , \mathbf { x } _ { s } ^ { k _ { s } } ) , s \leq T } { \mathbb { E } } \Big [ \cdot \mid k _ { 1 : T } , ( \mathbf { x } _ { t } ^ { k _ { t } } ) _ { 1 \leq t \leq T } \Big ]
$$

Then, as long as pe satisfies the Markov property,

b $\begin{array} { r l } & { \quad \underset { \mathrm { o r w a r d } , \mathscr { T } } { \mathbb { E } } [ \ln p \theta \left( \left( \mathbf { x } _ { t } ^ { k _ { t } } \right) _ { 1 \leq t \leq T } \right) ] \geq C ( \mathbf { x } _ { 1 : T } ^ { 0 } ) + \underset { \mathrm { f o r w a r d } , \mathscr { D } _ { p , \mathbf { z } _ { 1 : T } } } { \mathbb { E } } \left[ \underset { t = 1 } { \overset { T } { \sum } } \Xi _ { t } \right] , w h e r e } \\ & { \overset { \geq } \varepsilon _ { t } : = \left( P _ { t } ( 1 \mid k _ { 1 : t - 1 } ) \ln p \theta \left( \mathbf { x } _ { t } ^ { 0 } \mid \mathbf { x } _ { t } ^ { 1 } , \mathbf { z } _ { t - 1 } \right) + \underset { j = 2 } { \overset { K } { \sum } } j P _ { t } ( j \mid k _ { 1 : t - 1 } ) \mathrm { D } _ { \mathbb { K L } } \left( q ( \mathbf { x } _ { t } ^ { j - 1 } \mid \mathbf { x } _ { t } ^ { j } , \mathbf { x } _ { t } ^ { 0 } ) \parallel p \theta \left( \mathbf { x } _ { t } ^ { j } \mid \mathbf { x } _ { t } ^ { j - 1 } , \mathbf { z } _ { t - 1 } \right) \right) \right) } \end{array}$ . ) where $C ( \mathbf { x } _ { 1 : T } ^ { 0 } )$ is a constant depending only on $\mathbf { x } _ { 1 : T } ^ { 0 }$ (the noise-free data), and where the inequality is an equality under the conditions that (a) $p _ { \pmb { \theta } } ( \mathbf { z } _ { t } \mid \mathbf { z } _ { t - 1 } , \mathbf { x } _ { t } ^ { k _ { t } } )$ is a Dirac distribution (deterministic latents), and (b) $q ( \mathbf { x } _ { t } ^ { k _ { t } + 1 : T } \mid \mathbf { x } _ { t } ^ { k _ { t } } ) \equiv p _ { \theta } ( \mathbf { x } _ { t } ^ { k _ { t } + 1 : T } \mid \mathbf { x } _ { t } ^ { k _ { t } } , \mathbf { z } _ { t - 1 } )$

In particular, in the Gaussian case of Corollary A.2, we have

$$
\underset { \mathrm { r e w a r d } , \mathcal { D } } { \mathbb { E } } [ \ln p \theta \big ( ( \mathbf { x } _ { t } ^ { k _ { t } } ) _ { 1 \le t \le T } \big ) ] + C ( \mathbf { x } _ { 1 : T } ^ { 0 } ) \ge \underset { \mathrm { f o r w a r d } , \mathcal { D } } { \mathbb { E } } \underset { p , \mathbf { z } _ { 1 : T } } { \mathbb { E } } \left[ \sum _ { t = 1 } ^ { T } \mathbf { 1 } \{ k _ { t } \ge 1 \} k _ { t } c _ { k _ { t } } \| \hat { \mathbf { x } } _ { \theta } ^ { 0 } ( \mathbf { x } _ { t } ^ { k _ { t } } , \mathbf { z } _ { t - 1 } , k _ { t } ) - \mathbf { x } _ { t } ^ { 0 } \| ^ { 2 } \right] .
$$

The most salient case for us is the restriction of $\mathcal { D }$ to fixed sequences of noise $k _ { 1 } , \dots , k _ { T }$ (i.e. Dirac distributions on $[ K ] ^ { T } )$ . In this case, $P _ { t } ( j \mid k _ { 1 : t - 1 } ) = 0$ for all but $j = k _ { t }$ , and thus our training objective need not be a lower bound on $\mathbb { E } _ { \mathrm { f o r w a r d } , \mathcal { D } } \big [ \ln p _ { \theta } \big ( \big ( \mathbf { x } _ { t } ^ { k _ { t } } \big ) _ { 1 \leq t \leq T } \big ) \big ]$ .However, the terms in the lower bound are, up to reweighting, an thos r tiz th eciThu in ht  the ar olowCrolA., a fully expressive network can optimize all the terms in the loss simultaneously. We conclude that, for a fuy eresive neural etwork,tizing the trai jetiv 3.1is a val srat or axi the likelihood of all possible noise sequences.

# A.2 Proof of Theorem A.1

Defie $\mathbb { E } _ { < t } [ \cdot ]$ as shorthhand for $\mathbb { E } _ { k _ { 1 : s } \sim [ k ] } \mathbb { E } _ { \mathbf { x } _ { s } ^ { k _ { s } } \sim q ( \mathbf { x } _ { s } ^ { k _ { s } } \mid \mathbf { x } _ { s } ^ { 0 } ) , 1 \le s \le t - 1 } \mathbb { E } _ { \mathbf { z } _ { s } \sim p ( \mathbf { z } _ { s } \mid \mathbf { z } _ { s - 1 } , \mathbf { x } _ { s } ^ { k _ { s } } ) , s \le t } [ \cdot ]$ We begn with the following claim

Claim 1 (Expanding the latents). The following lower bound holds:

$$
\underset { \mathrm { f o r w a r d } } { \mathbb { E } } [ \ln p \theta \big ( \big ( \mathbf { x } _ { t } ^ { k _ { t } } \big ) _ { 1 \le t \le T } \big ) ] \ge \sum _ { t = 1 } ^ { T } \underset { < t } { \mathbb { E } } \underset { k _ { t } ^ { \mathrm { u n i f } } \ \{ 0 , 1 , \dots , K \} } { \mathbb { E } } \ \underset { \mathbf { x } _ { t } ^ { k _ { t } } \sim q ( \mathbf { x } _ { t } ^ { k _ { t } } \mid \mathbf { x } _ { t } ^ { 0 } ) } { \mathbb { E } } \Big [ \ln p \theta \big ( \mathbf { x } _ { t } ^ { k _ { t } } \mid \mathbf { z } _ { t - 1 } \big ) \Big ] ,
$$

Moreover, this lower bound holds with equality $i f \mathbf { z } _ { s } \sim p ( \mathbf { z } _ { s } \mid \mathbf { z } _ { s - 1 } , \mathbf { x } _ { s } ^ { k _ { s } } )$ buion .eistic latents).

Proof. Let's fix a sequence $k _ { 1 : T }$ . It holds that

$$
\begin{array} { l } { \displaystyle p \theta \big ( ( { \mathbf x } _ { t } ^ { k _ { t } } ) _ { 1 \leq t \leq T } \big ) = \int _ { \mathbf z _ { 1 } \times T } \displaystyle \prod _ { t = 1 } ^ { T } p ( { \mathbf x } _ { t } ^ { k _ { t } } , \mathbf z _ { t } \mid ( { \mathbf x } _ { s } ^ { k _ { s } } , \mathbf z _ { s } ) _ { s < t } ) } \\ { = \int _ { \mathbf z _ { 1 } \times T } \displaystyle \prod _ { t = 1 } ^ { T } p ( { \mathbf x } _ { t } ^ { k _ { t } } , \mathbf z _ { t } \mid \mathbf z _ { t - 1 } ) } \\ { = \int _ { \mathbf z _ { 1 } \times T } \displaystyle \prod _ { t = 1 } ^ { T } p ( { \mathbf z } _ { t } \mid \mathbf z _ { t - 1 } , \mathbf x _ { t } ^ { k _ { t } } ) p \theta ( { \mathbf x } _ { t } ^ { k _ { t } } \mid \mathbf z _ { t - 1 } ) } \\ { = \int _ { \mathbf z _ { 1 } \times T } \displaystyle \mathrm { K } \sum _ { t = 1 } ^ { T } p ( { \mathbf z } _ { t } \mid \mathbf z _ { t - 1 } , \mathbf x _ { t } ^ { k _ { t } } ) p \theta ( { \mathbf x } _ { t } ^ { k _ { t } } \mid \mathbf z _ { t - 1 } ) } \\ { = \underbrace { T } _ { \displaystyle \mathbf z _ { s } \sim p ( \mathbf z _ { s } \mid \mathbf z _ { s - 1 } , \mathbf x _ { s } ^ { k _ { s } } ) , s \in T } \displaystyle \prod _ { t = 1 } ^ { T } p _ { \theta } ( { \mathbf x } _ { t } ^ { k _ { t } } \mid \mathbf z _ { t - 1 } ) . } \end{array}
$$

(Importance Sampling)

Thus, by Jensen's inequaliy,

$$
\ln p _ { \theta } \big ( \big ( \mathbf { x } _ { t } ^ { k _ { t } } \big ) _ { 1 \le t \le T } \big ) \ge \underset { \mathbf { z } _ { s } \sim p ( \mathbf { z } _ { s } | \mathbf { z } _ { s - 1 } , \mathbf { x } _ { s } ^ { k _ { s } } ) , s \le T } { \mathbb { E } } \sum _ { t = 1 } ^ { T } \ln p _ { \theta } \big ( \mathbf { x } _ { t } ^ { k _ { t } } \mid \mathbf { z } _ { t - 1 } \big ) = \underset { p , \mathbf { z } _ { 1 : T } } { \mathbb { E } } \left[ \sum _ { t = 1 } ^ { T } \ln p _ { \theta } \big ( \mathbf { x } _ { t } ^ { k _ { t } } \mid \mathbf { z } _ { t - 1 } \big ) \right] ,
$$

wh he equalitys nqualiy whn $p _ { \pmb { \theta } } \big ( \mathbf { z } _ { s } \mid \mathbf { z } _ { s - 1 } , \mathbf { x } _ { s } ^ { k _ { s } } \big )$ s aisrtin. By pp $\mathbb { E } _ { \mathrm { f o r w a r d } }$ to both sides of the above display, and invoking the Markov property of the latents, we conclude that

$$
\begin{array} { r l } & { \quad \underset { \mathrm { f o r w a r d } } { \mathbb { E } } [ \ln p \theta \big ( \big ( \mathbf { x } _ { t } ^ { k _ { t } } \big ) _ { 1 \le t \le T } \big ) ] \ge \underset { \mathrm { f o r w a r d } } { \mathbb { E } } \underset { p , \mathbf { z } _ { 1 : T } } { \mathbb { E } } \left[ \overset { T } { \sum _ { t = 1 } ^ { T } } \ln p \theta \big ( \mathbf { x } _ { t } ^ { k _ { t } } \mid \mathbf { z } _ { t - 1 } \big ) \right] } \\ & { \quad \quad \quad \quad \quad = \underset { t = 1 } { \overset { T } { \sum } } \underset { < t _ { k _ { t } } \mathrm { w a r d } } { \mathbb { E } } \underset { \{ 0 , 1 , \dots , K \} } { \mathbb { E } } \underset { \mathbf { x } _ { t } ^ { k _ { t } } \sim q \big ( \mathbf { x } _ { t } ^ { k _ { t } } \mid \mathbf { x } _ { t } ^ { 0 } \big ) } { \mathbb { E } } \left[ \ln p \theta \big ( \mathbf { x } _ { t } ^ { k _ { t } } \mid \mathbf { z } _ { t - 1 } \big ) \right] . } \end{array}
$$

We now unpack the terms obtained from the preceding claim.

Claim 2 (ELBO w.r.t. $q$ ). It holds that

$$
\mathbb { E } _ { \mathbf { z } _ { t } ^ { k _ { t } } \sim q ( \mathbf { x } _ { t } ^ { k _ { t } } \mid \mathbf { x } _ { t } ^ { 0 } ) } \left[ \ln p _ { \theta } ( \mathbf { x } _ { t } ^ { k _ { t } } \mid \mathbf { z } _ { t - 1 } ) \right] \geq C _ { 1 } ( \mathbf { x } _ { 0 } , k _ { t } ) + \left[ \underset { \mathbf { x } _ { t } ^ { k _ { t } + K } \sim q ( \mathbf { x } _ { t } ^ { k _ { t } + K } \mid \mathbf { x } _ { t } ^ { 0 } ) } { \mathbb { E } } \ln \frac { \ln p _ { \theta } ( \mathbf { x } _ { t } ^ { k _ { t } + K } \mid \mathbf { z } _ { t - 1 } ) } { q ( \mathbf { x } _ { t } ^ { k _ { t } + 1 : K } \mid \mathbf { x } _ { t } ^ { 0 } ) } \right] .
$$

where $C _ { 1 } ( \mathbf { x } _ { 0 } , k _ { t } )$ is a constant depending only on $\mathbf { x } _ { \mathrm { 0 } }$ and $k _ { t }$ , and where the inequality holds with equality if and only i $^ { c } q ( \mathbf { x } _ { t } ^ { k _ { t } + 1 : T } \mid \mathbf { x } _ { t } ^ { k _ { t } } ) \equiv p _ { \theta } ( \mathbf { x } _ { t } ^ { k _ { t } + 1 : T } \mid \mathbf { x } _ { t } ^ { k _ { t } } , \mathbf { z } _ { t - 1 } )$

Proof. We have that

$$
\begin{array} { r l } & { \quad \times _ { i } ^ { \mathbb { E } _ { k } } , \quad \underset { s ^ { \prime } = 0 } { \overset { \mathbb { E } } { \sum } } \left[ \mathrm { i n } \rho ( X _ { \epsilon } ^ { k , \epsilon } | \mathcal { H } _ { \epsilon - 1 } ) \right] } \\ & { = \underset { s _ { i } ^ { \mathbb { E } _ { k } } , \quad 0 \leq i \leq n } { \overset { \mathbb { E } } { \sum } } \underset { s ^ { \prime } = 0 } { \overset { \mathbb { E } } { \sum } } \left[ \mathrm { i n } \int \mathrm { p } \rho ( X _ { \epsilon } ^ { k , \epsilon , K } | \mathcal { Z } _ { \epsilon - 1 } ) \mathrm { d } \Omega _ { \epsilon } ^ { k , \epsilon + 1 , K } \right] } \\ & { = \underset { s _ { i } ^ { \mathbb { E } _ { k } } , \quad 0 \leq i \leq n } { \overset { \mathbb { E } } { \sum } } \underset { s ^ { \prime } = 0 } { \overset { \mathbb { E } } { \sum } } \left[ \mathrm { i n } \left( \underset { s _ { i } ^ { \mathbb { E } _ { k } + 1 + 1 , K } \sim \underset { s \in \mathcal { E } _ { \epsilon + 1 } } { \sum } \times \underset { s \in \mathcal { E } _ { \epsilon + 1 } } { \sum } \times \underset { s \in \mathcal { E } _ { \epsilon + 1 } } { \sum } } \right) \right] } \\ & { \overset { \mathbb { E } } { \underset { s ^ { \prime } \sim 0 } { \overset { \mathbb { E } } { \sum } } } \left[ \mathrm { i n } \left( \underset { s ^ { \prime } \sim 0 } { \overset { \mathbb { E } } { \sum } } \right) \right] } \\ &  \leq \underset { s _ { i } ^ { \mathbb { E } _ { k } } , \quad 0 \leq i \leq n } { \overset { \mathbb { E } } { \sum } } \underset { s ^ { \prime } \sim 0 } { \overset { \mathbb { E } } { \sum } } \left[ \mathrm { i n } \underset { s ^ { \prime } \sim 0 } { \overset { \mathbb { E } } { \sum } } \mathrm { i n } \underset { s ^ { \prime } \sim 0 } { \overset { \mathbb { E } } { \sum } } \right] \underset { s ^ { \prime } \sim 0 } { \overset { \mathbb { E } } { \sum } } \left[ \mathrm { i n } \underset { s ^ { \prime } \sim \mathrm { i n } \times \underset { s ^ { \prime } } { \sum } \times \underset { s ^ { \prime } = 1 } { \overset { \mathbb { E } } { \sum } } \right] } \\ &  = \end{array}
$$

[uality))

of $q ( \cdot ) \ddot { }$ b

where the(, ) q(x+1ik q(xk}+1) depends only on $\mathbf { x } _ { \mathrm { 0 } }$ and $k _ { t }$ . To c $q ( \mathbf { x } _ { t } ^ { k _ { t } + 1 : T } \mid \mathbf { x } _ { t } ^ { k _ { t } } ) \equiv p _ { \theta } ( \mathbf { x } _ { t } ^ { k _ { t } + \mathrm { i } : T } \mid \mathbf { x } _ { t } ^ { k _ { t } } , \mathbf { z } _ { t - 1 } )$ , then

$\underset { t _ { t } + 1 : K \sim q ( \mathbf { x } _ { t } ^ { k _ { t } + 1 : K } \mid \mathbf { x } _ { t } ^ { k _ { t } } ) } { \mathbb { E } } \left[ \ln \frac { p \varrho \left( \mathbf { x } _ { t } ^ { k _ { t } : K } \mid \mathbf { z } _ { t - 1 } \right) } { q \left( \mathbf { x } _ { t } ^ { k _ { t } + 1 : K } \mid \mathbf { x } _ { t } ^ { k _ { t } } \right) } \right] = \ln p \varrho \left( \mathbf { x } _ { t } ^ { k _ { t } } \mid \mathbf { z } _ { t - 1 } \right) + \underset { \mathbf { x } _ { t } ^ { k _ { t } + 1 : K } \sim q ( \mathbf { x } _ { t } ^ { k _ { t } + 1 : K } \mid \mathbf { x } _ { t } ^ { k _ { t } } ) } { \mathbb { E } } \left[ \ln p \varrho \left( \mathbf { x } _ { t } ^ { k _ { t } + 1 : K } \mid \mathbf { x } _ { t } ^ { k _ { t } } \right) \right]$ |z−1,x+)] Since $\ln ( \cdot )$ is stricly concave, $\mathbb { E } _ { \mathbf { x } _ { t } ^ { k _ { t } + 1 : K } \sim q ( \mathbf { x } _ { t } ^ { k _ { t } + 1 : K } | \mathbf { x } _ { t } ^ { k _ { t } } ) } \left[ \ln p _ { \theta } ( \mathbf { x } _ { t } ^ { k _ { t } } \mid \mathbf { z } _ { t - 1 } ) \right] = 0$ and only if $p _ { \pmb { \theta } } ( \mathbf { x } _ { t } ^ { k _ { t } + 1 : K } \ )$ $\mathbf { z } _ { t - 1 } , \mathbf { x } _ { t } ^ { k _ { t } } ) = q ( \mathbf { x } _ { t } ^ { k _ { t } + 1 : K } \mid \mathbf { x } _ { t } ^ { k _ { t } } ) .$

Claim 3 (Computing the expected ELBO).

$$
\begin{array} { r l } & { \underset { \ell _ { t } ^ { k \ell } \sim \ell _ { \ell } ( { \mathbf x } _ { t } ^ { k } \ell ^ { k } ) } { \mathbb { E } } \underset { | { \mathbf x } _ { t } ^ { 0 } ( { \mathbf x } _ { t } ^ { k } + 1 : K _ { \ell } ^ { 0 } \mid { \mathbf x } _ { t } ^ { 0 } ) } { \mathbb { I } } \ln \frac { p \theta ( { \mathbf x } _ { t } ^ { k _ { t } ^ { \ell } \cdot K } \mid { \mathbf z } _ { t - 1 } ) } { q ( { \mathbf x } _ { t } ^ { k _ { t } ^ { \ell + 1 : K } } \mid { \mathbf x } _ { t } ^ { 0 } ) } } \\ & { = C _ { 3 } ( { \mathbf x } _ { 0 } , k _ { t } ) + 1 \{ k _ { t } = 0 \} \ln p \theta ( { \mathbf x } _ { t } ^ { 0 } \mid { \mathbf x } _ { t } ^ { 1 } , \mathbf z _ { t - 1 } ) + \underset { j = 1 } { \overset { K - 1 } { \sum } } { \mathbf 1 } \{ j \geq k _ { t } \} \mathrm { D } _ { \mathbb { E } \mathbb { L } } ( q ( { \mathbf x } _ { t } ^ { j } \mid { \mathbf x } _ { t } ^ { j + 1 } , { \mathbf x } _ { t } ^ { 0 } ) \mid \Vert p \theta ( { \mathbf x } _ { t } ^ { j } \mid { \mathbf x } _ { t } ^ { j + 1 } , \mathbf z _ { t - 1 } ) ) } \end{array}
$$

where $C _ { 2 } ( \mathbf { x } _ { 0 } , k _ { t } )$ is some other constant depending on $\mathbf { x } _ { \mathrm { 0 } }$ and $k _ { t }$

PrThe proof invokes similar manipulations t the standard ELBO derivation for diffusion, but with a few careful modifications to handle the fact that we only noise to level $k _ { t }$ . As is standard, we require the identity

$$
q ( \mathbf { x } _ { t } ^ { j } \mid \mathbf { x } _ { t } ^ { j - 1 } , \mathbf { x } _ { t } ^ { 0 } ) = q ( \mathbf { x } _ { t } ^ { j - 1 } \mid \mathbf { x } _ { t } ^ { j } , \mathbf { x } _ { t } ^ { 0 } ) \cdot { \frac { q ( \mathbf { x } _ { t } ^ { j } \mid \mathbf { x } _ { t } ^ { 0 } ) } { q ( \mathbf { x } _ { t } ^ { j - 1 } \mid \mathbf { x } _ { t } ^ { 0 } ) } } .
$$

Part 1: Expanding the likelihood ratios . Using the above identity, we obtain

$$
\begin{array} { r l } & { \begin{array} { r l } & { u _ { \widehat { \mu } } ^ { \beta } ( | \mathbf { x } _ { t } ^ { \mu } | ^ { s } , v | _ { \mathbf { z } _ { t - 1 } } ) } \\ & { u _ { \widehat { \mu } } ^ { \beta } ( | \mathbf { x } _ { t } ^ { \mu } | ^ { s + 1 / s } | v | ) } \end{array} } \\ & { = \operatorname* { l i m } _ { \widehat { \mu } } \gamma ( \mathbf { x } _ { t } ^ { \mu } | ^ { s } | \mathbf { z } _ { t - 1 }  ) + \operatorname* { l i m } \frac { p ( \mathbf { x } _ { t } ^ { \beta } | ^ { s } | \mathbf { x } _ { t } ^ { \kappa + 1 } , \mathbf { z } _ { t - 1 }  ) } { q ( \mathbf { x } _ { t } ^ { \kappa + 1 } ) ^ { 1 / s } q ^ { \widehat { \alpha } } } + \displaystyle \sum _ { t = k + 2 } ^ { K } \operatorname* { l i m } _ { \widehat { \mu } } \frac { p ( \mathbf { x } _ { t } ^ { \alpha - 1 } | \mathbf { x } _ { t } ^ { \kappa } , \mathbf { z } _ { t - 1 }  ) } { q ( \mathbf { x } _ { t } ^ { \kappa + 1 } ) ^ { 1 / s } q ^ { \widehat { \alpha } } } } \\ &  \overset { \mathrm { ( 1 ) } } { = } \operatorname* { l i m } _ { \widehat { \mu } } \gamma ( \mathbf { x } _ { t } ^ { \beta } | ^ { s } | \mathbf { z } _ { t - 1 } ) + \operatorname* { l i m } \frac { p ( \mathbf { x } _ { t } ^ { \beta } | ^ { s } | \mathbf { x } _ { t } ^ { \kappa + 1 } , \mathbf { z } _ { t - 1 }  ) } { q ( \mathbf { x } _ { t } ^ { \kappa + 1 } ) ^ { 1 / s } q ^ { \widehat { \alpha } } } + \displaystyle \sum _ { t = k + 2 } ^ { K } ( \operatorname* { l i m } _ { \widehat { \mu } } \frac { p ( \mathbf { x } _ { t } ^ { \beta - 1 } | \mathbf { x } _ { t } ^ { \kappa } , \mathbf { z } _ { t - 1 } ) } { q ( \mathbf { x } _ { t } ^ { \kappa - 1 } | \mathbf { x } _ { t } ^ { \kappa } ) q _ { \star } ^ { \kappa } } + \operatorname* { l i m } _ { \widehat { \mu } } \frac  q ( \mathbf { x } _ { t } ^ { \beta - 1 } | \mathbf { x } _ { t } ^ { \kappa } | \ \end{array}
$$

where $( i )$ uses A.10, $( i i )$ invokes a cancellation in the telescoping sum, and the final display follows from the computation

$$
\begin{array} { r } { q ( \mathbf { x } _ { t } ^ { k _ { t } } \mid \mathbf { x } _ { t } ^ { k _ { t } + 1 } ) ^ { \mathbf { 1 } \{ k _ { t } \geq 1 \} } = \left\{ \begin{array} { l l } { 1 } & { k _ { t } = 0 } \\ { q ( \mathbf { x } _ { t } ^ { k _ { t } } \mid \mathbf { x } _ { t } ^ { k _ { t } + 1 } ) } & { k _ { t } \geq 1 } \end{array} \right. . } \end{array}
$$

$\begin{array} { r } { p ( \mathbf { x } _ { t } ^ { K } \mid \mathbf { z } _ { t - 1 } ) , \ln \Big ( q ( \mathbf { x } _ { t } ^ { k _ { t } } \mid \mathbf { x } _ { t } ^ { k _ { t } + 1 } ) ^ { \mathbf { 1 } \{ k _ { t } \geq 1 \} } \Big ) + \frac { \ln p ( \mathbf { x } _ { t } ^ { K } \mid \mathbf { z } _ { t - 1 } ) } { q ( \mathbf { x } _ { t } ^ { K } \mid \mathbf { x } _ { t } ^ { k _ { t } } ) } } \end{array}$ can be regarded as some constant $C ^ { \prime } ( \mathbf { x } _ { t } ^ { k _ { t } } , \mathbf { x } _ { t } ^ { k _ { t } + 1 } , \mathbf { x } _ { t } ^ { K } )$ Thus,

$$
\ln \frac { p _ { \theta } ( \mathbf { x } _ { t } ^ { k _ { t } \cdot \kappa } \mid \mathbf { z } _ { t - 1 } ) } { q ( \mathbf { x } _ { t } ^ { k _ { t } + 1 \cdot K } \mid \mathbf { x } _ { t } ^ { 0 } ) } = C ^ { \prime } ( \mathbf { x } _ { t } ^ { k _ { t } } , \mathbf { x } _ { t } ^ { k _ { t } + 1 } , \mathbf { x } _ { t } ^ { K } ) + \ln \frac { p _ { \theta } ( \mathbf { x } _ { t } ^ { k _ { t } } \mid \mathbf { x } _ { t } ^ { k _ { t } + 1 } , \mathbf { z } _ { t - 1 } ) } { q ( \mathbf { x } _ { t } ^ { k _ { t } } \mid \mathbf { x } _ { t } ^ { k _ { t } + 1 } ) \mathbf { 1 } \{ k _ { t } \geq 1 \} } + \sum _ { j = k _ { t } + 1 } ^ { K - 1 } \ln \frac { p _ { \theta } ( \mathbf { x } _ { t } ^ { j } \mid \mathbf { x } _ { t } ^ { j + 1 } , \mathbf { z } _ { t - 1 } ) } { q ( \mathbf { x } _ { t } ^ { j } \mid \mathbf { x } _ { t } ^ { j + 1 } , \mathbf { x } _ { t } ^ { 0 } ) }
$$

Part 2: Taking expecations. We can now simplify to taking expectations. Observe that

$$
\underset { \mathbf { x } _ { t } ^ { k } : ^ { K } \sim q ( \mathbf { x } _ { t } ^ { k } : ^ { K } | \mathbf { x } _ { t } ^ { 0 } ) } { \mathbb { E } } \ln \frac { p _ { \theta } ( \mathbf { x } _ { t } ^ { j } \mid \mathbf { x } _ { t } ^ { j + 1 } , \mathbf { z } _ { t - 1 } ) } { q ( \mathbf { x } _ { t } ^ { j } \mid \mathbf { x } _ { t } ^ { j + 1 } , \mathbf { x } _ { t } ^ { 0 } ) } = \mathrm { D } _ { \mathbb { K L } } \left( q ( \mathbf { x } _ { t } ^ { j } \mid \mathbf { x } _ { t } ^ { j + 1 } , \mathbf { x } _ { t } ^ { 0 } ) \mid \mid p _ { \theta } ( \mathbf { x } _ { t } ^ { j } \mid \mathbf { x } _ { t } ^ { j + 1 } , \mathbf { z } _ { t - 1 } ) \right) ,
$$

and similarly,

$$
\underset { \substack { k _ { t } \colon K _ { \sim q } ( { \bf x } _ { t } ^ { k _ { t } } \colon K _ { i } ^ { \infty } ) } } { \mathbb { E } } \ln \frac { p _ { \theta } ( { \bf x } _ { t } ^ { k _ { t } } \mid { \bf x } _ { t } ^ { k _ { t } + 1 } , { \bf z } _ { t - 1 } ) } { q ( { \bf x } _ { t } ^ { k _ { t } } \mid { \bf x } _ { t } ^ { k + 1 } ) ^ { 1 } \{ k _ { t } \geq 1 \} } = \left\{ \begin{array} { l l } { \ln p _ { \theta } ( { \bf x } _ { t } ^ { 0 } \mid { \bf x } _ { t } ^ { 1 } , { \bf z } _ { t - 1 } ) } & { k _ { t } = 0 } \\ { { \operatorname* { D } _ { \mathbb { K } \bot } \left( q ( { \bf x } _ { t } ^ { k _ { t } } \mid { \bf x } _ { t } ^ { k _ { t } + 1 } , { \bf x } _ { t } ^ { 0 } ) \parallel p _ { \theta } ( { \bf x } _ { t } ^ { k _ { t } } \mid { \bf x } _ { t } ^ { j + 1 } , { \bf z } _ { t - 1 } ) \right) } } & { k _ { t } \geq 1 . } \end{array} \right.
$$

Finally, $\mathbb { E } _ { \mathbf { x } _ { t } ^ { k _ { t } : K } \sim q ( \mathbf { x } _ { t } ^ { k _ { t } : K } | \mathbf { x } _ { t } ^ { 0 } ) } C ^ { \prime } \big ( \mathbf { x } _ { t } ^ { k _ { t } } , \mathbf { x } _ { t } ^ { k _ { t } + 1 } , \mathbf { x } _ { t } ^ { K } \big )$ is a constant ${ C } _ { 2 } ( k _ { t } , { \bf x } _ { 0 } )$ depending only on $k _ { t } , \mathbf { x } _ { 0 }$ Thus, from (A.12)

$$
\begin{array} { r l } & { \quad \underset { \underset { t ^ { k _ { t } \cdots K } \sim q ( { \mathbf x } _ { t } ^ { k _ { t } \cdots K } \mid { \mathbf x } _ { t } ^ { 0 } ) } { \sum _ { t } } } { \mathbb { I } } _ { \mathbf { \Phi } _ { t } ^ { p } ( { \mathbf x } _ { t } ^ { \mathbf { k } _ { t } ^ { k _ { t } \cdots K } } \mid { \mathbf z } _ { t - 1 } ) } } \\ & { = C _ { 2 } ( k _ { t } , \mathbf { x } _ { 0 } ) + \mathbf { 1 } \{ k _ { t } = 0 \} \ln p \theta ( \mathbf { x } _ { t } ^ { 0 } \mid \mathbf { x } _ { t } ^ { 1 } , \mathbf { z } _ { t - 1 } ) + \underset { j = \operatorname* { m a x } \{ 1 , k _ { t } \} } { \overset { K - 1 } { \sum } } \operatorname { D g . } _ { \mathbb { E } \downarrow } \left( q ( \mathbf { x } _ { t } ^ { j } \mid \mathbf { x } _ { t } ^ { j + 1 } , \mathbf { x } _ { t } ^ { 0 } ) \parallel p \theta ( \mathbf { x } _ { t } ^ { j } \mid \mathbf { x } _ { t } ^ { j + 1 } , \mathbf { z } _ { t - 1 } ) \right) } \\ & { = C _ { 2 } ( k _ { t } , \mathbf { x } _ { 0 } ) + \mathbf { 1 } \{ k _ { t } = 0 \} \ln p \theta ( \mathbf { x } _ { t } ^ { 0 } \mid \mathbf { x } _ { t } ^ { 1 } , \mathbf { z } _ { t - 1 } ) + \underset { j = 1 } { \overset { K - 1 } { \sum } } \mathbf { 1 } \{ j \geq k _ { t } \} \operatorname { D g . } _ { \mathbb { E } \downarrow } \left( q ( \mathbf { x } _ { t } ^ { j } \mid \mathbf { x } _ { t } ^ { j + 1 } , \mathbf { x } _ { t } ^ { 0 } ) \parallel p \theta ( \mathbf { x } _ { t } ^ { j } \mid \mathbf { x } _ { t } ^ { j + 1 } , \mathbf { z } _ { t - 1 } ) \right) } \end{array}
$$

Completing the proof of the ELBO. We are now ready to complete the proof. By combining the previous two claims, we have

$$
\begin{array} { l } { { \displaystyle \mathbb { E } _ { t } ^ { k } \kappa _ { q ( { \bf x } _ { t } ^ { k } \mid { \bf x } _ { t } ^ { 0 } ) } \left[ \ln p \theta \left( { \bf x } _ { t } ^ { k _ { t } } \mid { \bf z } _ { t - 1 } \right) \right] } \ ~ } \\ { { \displaystyle ~ \sum _ { c ^ { k _ { t } } \sim q ( { \bf x } _ { t } ^ { k _ { t } } \mid { \bf x } _ { t } ^ { 0 } ) } \left[ \ln p \theta \left( { \bf x } _ { t } ^ { 0 } \mid { \bf x } _ { t } ^ { 1 } , { \bf z } _ { t - 1 } \right) + \sum _ { j = 1 } ^ { K - 1 } { \bf 1 } \{ j \geq k _ { t } \} \mathrm { D } _ { \mathbb { E } \perp } \left( q ( { \bf x } _ { t } ^ { j } \mid { \bf x } _ { t } ^ { j + 1 } , { \bf x } _ { t } ^ { 0 } ) \parallel p \theta \left( { \bf x } _ { t } ^ { j } \mid { \bf x } _ { t } ^ { j + 1 } , { \bf z } _ { t - 1 } \right) \right) \right] } \ ~ , }  \end{array}
$$

where $C _ { 3 } ( \mathbf { x } _ { 0 } , k _ { t } ) = C _ { 1 } ( \mathbf { x } _ { 0 } , k _ { t } ) + C _ { 2 } ( \mathbf { x } _ { 0 } , k _ { t } )$ and whe again, thebove is a equality when $q ( \mathbf { x } _ { t } ^ { k _ { t } + 1 : T } \ )$ $\mathbf { x } _ { t } ^ { k _ { t } } \mathbf { \Big ) } \equiv p _ { \pmb { \theta } } \big ( \mathbf { x } _ { t } ^ { k _ { t } + 1 : T } \ \big \vert \ \mathbf { x } _ { t } ^ { k _ { t } } , \mathbf { z } _ { t - 1 } \big )$ Taking an expectation over $k _ { t } \stackrel { \mathrm { u n i f } } { \sim } \{ 0 , 1 , \ldots , K \}$ ,we have

$$
\underset { k _ { t } \ " \sim \{ 0 , 1 , \ldots , K \} } { \mathbb { E } } [ \mathbf { 1 } \{ k _ { t } = 0 \} ] = \frac { 1 } { K + 1 } , \quad \underset { k _ { t } \ " \sim \{ 0 , 1 , \ldots , K \} } { \mathbb { E } } \mathbf { 1 } \{ j \geq k _ { t } \} = \frac { j + 1 } { K + 1 } .
$$

and consequently,

$$
\begin{array} { l } { \displaystyle \underset { k ^ { \mathrm { u n i } } \times \infty } { \mathbb { E } } \underset { \{ 0 , 1 , \ldots , K \} } { \mathbb { E } } \times _ { t } \sim q ( \mathbf { x } _ { t } ^ { k } \vert \mathbf { x } _ { t } ^ { 0 } ) , 1 \le t \le T } \end{array}
$$

Invoking Claim 1,

$$
\begin{array} { l } { \underset { \mathrm { o r s a r d } } { \mathbb { E } } [ \ln p _ { \theta } \big ( \big ( \mathbf { x } _ { t } ^ { k _ { t } } \big ) _ { 1 \le t \le T } \big ) ] } \\ { \underset { \mathrm { o r s a r d } } { \ge } \underset { \mathit { t } = 1 } { \overset { T } { \sum } } \underset { \mathit { k } _ { t } \underset { \sim 1 } { \mathbb { E } } ^ { k _ { t } } \underset { \sim 1 } { \mathrm { m i n } } \left\{ \mathbb { E } \right. } , \mathit { k } , \mathit { \xi } _ { t } ^ { k _ { t } } \sim q ( \mathbf { x } _ { t } ^ { k _ { t } } \vert \mathbf { x } _ { t } ^ { 0 } ) } { \mathbb { E } } \left[ \ln p _ { \theta } \big ( \mathbf { x } _ { t } ^ { k _ { t } } \mid \mathbf { z } _ { t - 1 } \big ) \right]  \\ { = \underset { t = 1 } { \overset { T } { \sum } } \underset { \mathit { t } = 1 } { \overset { \mathbb { E } } { \sum } } \left[ C _ { 4 } \big ( \mathbf { x } _ { t } ^ { 0 } \big ) + \frac { 1 } { K + 1 } \ln p _ { \theta } \big ( \mathbf { x } _ { t } ^ { 0 } \mid \mathbf { x } _ { t } ^ { 1 } , \mathbf { z } _ { t - 1 } \big ) + \underset { j = 1 } { \overset { K - 1 } { \sum } } \frac { j + 1 } { K + 1 } \mathrm { D } _ { \mathbb { K } \perp } \left( q \big ( \mathbf { x } _ { t } ^ { j } \mid \mathbf { x } _ { t } ^ { j + 1 } , \mathbf { x } _ { t } ^ { 0 } \big ) \mid \mid p _ { \theta } \big ( \mathbf { x } _ { t } ^ { j } \mid \mathbf { x } _ { t } ^ { j + 1 } , \mathbf { z } _ { t - 1 } \big ) \right) \right. } \end{array}
$$

We conclude by observing that $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \mathbb { E } _ { < t } \left[ C _ { 4 } ( { \mathbf x } _ { t } ^ { 0 } ) \right] } \end{array}$ is a constant $C ( \mathbf { x } _ { 1 : T } ^ { 0 } )$ , and that

$$
\begin{array} { r l } & { \underset { < t } { \mathbb { E } } \left[ \ln p _ { \theta } ( \mathbf { x } _ { t } ^ { 0 } \mid \mathbf { x } _ { t } ^ { 1 } , \mathbf { z } _ { t - 1 } ) \right] = \underset { \mathrm { f o r w a r d } } { \mathbb { E } } \underset { p , \mathbf { z } _ { 1 : T } } { \mathbb { E } } \left[ \ln p _ { \theta } ( \mathbf { x } _ { t } ^ { 0 } \mid \mathbf { x } _ { t } ^ { 1 } , \mathbf { z } _ { t - 1 } ) \right] } \\ & { \underset { < t } { \mathbb { E } } \left[ \mathrm { D } _ { \mathbb { E L } } \left( q ( \mathbf { x } _ { t } ^ { j } \mid \mathbf { x } _ { t } ^ { j + 1 } , \mathbf { x } _ { t } ^ { 0 } ) \mid \mid p _ { \theta } ( \mathbf { x } _ { t } ^ { j } \mid \mathbf { x } _ { t } ^ { j + 1 } , \mathbf { z } _ { t - 1 } ) \right) \right] } \\ & { = \underset { \mathrm { f o r w a r d } } { \mathbb { E } } \underset { p , \mathbf { z } _ { 1 : T } } { \mathbb { E } } \left[ \mathrm { D } _ { \mathbb { E L } } \left( q ( \mathbf { x } _ { t } ^ { j } \mid \mathbf { x } _ { t } ^ { j + 1 } , \mathbf { x } _ { t } ^ { 0 } ) \mid \mid p _ { \theta } ( \mathbf { x } _ { t } ^ { j } \mid \mathbf { x } _ { t } ^ { j + 1 } , \mathbf { z } _ { t - 1 } ) \right) \right] , } \end{array}
$$

since both terms only depend on $k _ { 1 : t - 1 } , ( \mathbf { x } _ { s } ^ { k _ { s } } ) _ { 1 \leq s \leq t - 1 }$ and $\mathbf { z } _ { 1 : t - 1 }$ .We conclude then that

$$
\begin{array} { l } { { \displaystyle \mathop { \mathbb { E } } _ { \mathrm { o r s a r d } } ^ { \mathbb { E } } \left\| \ln p _ { \theta } \big ( ( { \bf x } _ { t } ^ { k _ { t } } ) _ { 1 \le t \le T } \big ) \right\| \ge C ( { \bf x } _ { 1 : T } ^ { 0 } ) } } \\ { { \displaystyle + \mathop { \mathbb { E } } _ { \mathrm { f o r s a r d } } p _ { , z _ { 1 : T } } \left[ \displaystyle \sum _ { t = 1 } ^ { T } \left( \frac { 1 } { K + 1 } \ln p _ { \theta } \big ( { \bf x } _ { t } ^ { 0 } \mid { \bf x } _ { t } ^ { 1 } , { \bf z } _ { t - 1 } \big ) + \displaystyle \sum _ { j = 1 } ^ { K - 1 } \frac { j + 1 } { K + 1 } \mathrm { D } _ { \mathbb { E } \mathrm { L } } \left( q ( { \bf x } _ { t } ^ { j } \mid { \bf x } _ { t } ^ { j + 1 } , { \bf x } _ { t } ^ { 0 } ) \right) \Big \| \ p _ { \theta } \big ( { \bf x } _ { t } ^ { j } \mid { \bf x } _ { t } ^ { j + 1 } , { \bf z } _ { t - 1 } \big ) \right) \right] , } } \end{array}
$$

as needed. Lastly, we recall that the above is an equality under the conditions that

(a) $p _ { \pmb { \theta } } \big ( \mathbf { z } _ { t } \mid \mathbf { z } _ { t - 1 } , \mathbf { x } _ { t } ^ { k _ { t } } \big )$ is a Drac stiion, $q ( \mathbf { x } _ { t } ^ { k _ { t } + 1 : T } \mid \mathbf { x } _ { t } ^ { k _ { t } } ) \equiv p \pmb { \theta } ( \mathbf { x } _ { t } ^ { k _ { t } + 1 : T } \mid \mathbf { x } _ { t } ^ { k _ { t } } , \mathbf { z } _ { t - 1 } )$ and we reindex $j \gets j + 1$ to ensure consistency with indexing in standard expositions of the diffusion ELBO.

# B Additional Intuitions and Explainations

# B.1 Extension to transformer backbone

While this paper focuses on a causal implementation of Diffusion Forcing with RNNs, it's easy to adopt Diffusion Forcing with modern architectures like transformers. One can simply modify a transformer-based sequence usion model totrai witindependent noiseevlsacross tokens nd ollowhe tecniques ist Section D.1. A strict implementation of causal Diffusion Forcing would involve a causal attention mask o the transformer. However, Diffusion Forcing's fractional masking can do something more interesting: Consider the scenario that we use a transormer without a causal mask.We can stil implement causality by conrollin beeu ul whohe  kinte pa tke tkens s eo e ak emo cley -cusa By beng oke , asight amount o informatin about the future is provided or the predictiono past tokens This eecively states thatney neea n-causal rhitecture, u controlractnal nois thefuture t parl ple ausalyThe extesins  beyond hescothis pape, ut welred rh effectiveness and thus provide them as intuitions for future works.

# B.2 The need for independent noise levels

When training Diffusion Forcing, we choose tosample per-token noise evel following i.iuniform istrutin from [1, 2..K]. One may wonder about the necessity of this choice. Here we discuss the unique abilities of independent noise and the compute overhead added by it.

The use o independent noise confer anumberof special capabilities in our model, includingstabilization o autoregressive rollout 3.3, modeling causal uncertainty 3., and removing the need for expensiv reconstruction guidance when conditioning on context B.6. None of these capabilities can be achieved by full-sequence difson. AR-diffusion [67] and Rollng Diffusion [52] can only achieve the frst and third one. There are more sampling-time applications such as fexibleframe interpolation. Finally, we also saw the practical benets o using independent noise in hyperparameter tuning. One can simply try different sampling schemes to figure out the most efective one for their applications. All these capabilities only require training the mode once wi Diffsn Forg In cnrast, any tnig of the smplng hee woul eque -raig he mdel or AR-diffusion and Rolling Diffusion.

On the other hand, we didn't observe much computing overhead when comparing Diffusion Forcing to fullseence difusion, a son as one closly follows our traii tecniques ke D.1.The mpirical eviden is based on our experiments with an experimental transformer implementation of Diffusion Forcing and is thus not fully onsitent with he ain papr.Howeve we preent te i-evel desciptions belo o eadentsd in more insights The complexity added byindependent noise level is inthe temporal dimension. Therefore, we f dopttanr teniqu rviedusonoelage re-rainng, stawayheoey ohemage pixels themselvesThen he coplexiy  is temporal prediction nly.We then take the pre-rained e    a I  t     iF wn We peulat hat heett sl i u hemet-tan e dib r wrks [0T shows that the overhead added by independent noise is well-warranted when considering the overall training compute (including image pre-training).

# B.3 Guidance as planning

As stated in Section 2, one can use the gradient of the logarithmic of a classifier $\log c ( y | \mathbf { x } _ { t } ^ { k } )$ to guide the sampling process of diffusion model towards samples with a desired attribute $_ y$ For example, $y$ can refer to the indicator of a success event. However, we can consider the logarithmic of a more general energy function $c ( \mathbf { x } _ { t } ^ { k } )$ . This has the interpretation as $\mathrm { P r } \big ( y | \mathbf { x } _ { t } ^ { k } \big )$ , where $\operatorname* { P r } \big [ y = 1 \ \lvert \ \mathbf { x } _ { t } ^ { k } \big ] = e ^ { c ( \mathbf { x } _ { t } ^ { k } ) }$ Some popular candidate energies include

$$
c ( \mathbf { x } _ { t } ^ { k } ) = \mathbb { E } \left[ \sum _ { t ^ { \prime } > t } \mathbf { r } ^ { \prime } ( \mathbf { x } _ { t ^ { \prime } } ^ { k _ { t ^ { \prime } } } ) \mid \mathbf { x } _ { t } ^ { k } \right] ,
$$

corresponding to a cost-to-go; we can obtain unbiased estimates of this gradient by using cumulative reward $\begin{array} { r } { \tilde { c } ( \mathbf { x } _ { t } ^ { k } ) = \sum _ { t ^ { \prime } \geq t } \mathbf { r } ^ { \prime } ( \mathbf { x } _ { t ^ { \prime } } ^ { k _ { t ^ { \prime } } } ) } \end{array}$ $c = - \| \mathbf { x } _ { T } ^ { k _ { T } } - \mathbf { g } \| ^ { 2 }$ details about the guidance function deployed in the maze2d planning experiment in Appendix D.5.

# B.4 Noising and stabilizing long-horizon generations

Here, we explain in detail how we use noising to stabilize long-horizon generation. At each time $t$ ,during $\mathbf { z } _ { t - 1 } ^ { k _ { \mathrm { s m a l l } } }$ $0 < k _ { \mathrm { s m a l l } } \ll K$ coresponding to some small amount of noise. We then do next token diffusion to diffuse the token $\mathbf { x } _ { t }$ across noise levels $\mathbf { x } _ { t } ^ { K } , \mathbf { x } _ { t } ^ { K - 1 } , \ldots , \mathbf { x } _ { t } ^ { 0 }$ (corresponding to Algorithm 2 with horizon $T = 1$ , initial latent $\mathbf { z } _ { t - 1 } ^ { k }$ , and noise schedule ${ \ K } _ { m , 1 } = m ,$ $\mathbf { z } _ { t } ^ { K } , \mathbf { z } _ { t } ^ { K - 1 } , \ldots , \mathbf { z } _ { t } ^ { 0 }$ associated with each noise level. From these, we use the latennt $\bar { \mathbf { z } } _ { t } ^ { k _ { \mathrm { s m a l l } } }$   
conditioning on $\mathbf { x } _ { t } ^ { k _ { \mathrm { s m a l l } } }$   
Forcing as we discussed in Appendix B.1, one caninstead run a forwarddiffusion n a fully diffused tokento achieve stabilization.

It is widely appreciated that adding noise to data ameliorates long-term compounding error in behavior cloning applications [39, 43], and even induces robustness to non-sequential adversarial attacks [13]. In autoregressive video generation, the noised $\mathbf { x } _ { t } ^ { k _ { \mathrm { s m a l l } } }$ pas seration train jeciv Hene, ths method can beterpreteas  spel cas theDART algorithm for behavior cloning [43], where the imitiator (in our case, video generator) is given actions (in our case, next video frames) from noisy observations (in our case, noised previous frames). Somewhat more pelybeuewe useboth tokens  taiie ai Difuin Forandusn hyosken for autoregression at test time, our approach inherits the theoretical guarantee of the HINT algorithm [5].

![](images/6.jpg)  
Figure 5: Diffusion Forcing is trained on independent level of noises at different timesteps. As a result, we can control the noise level $k$ to achieve different effects on conditioning and prediction.

# B.5 Why Monte Carlo Guidance relies on Diffusion Forcing

Monte Carl Guidance provides substantial variancereductionu estimate cos-to-o guidance (B.1.This k tMontearlestatesor adientsThis is ot asble with ullequene iffusion, because this re densng all tokens in tandem; thus, for a given fixed noiselevel, there is noobvious source o randoness to uefor the Monte Carlo estiate. It may be possible to achive variable horizon via the trick propose n the following subsection to simulate future rollouts, but to our knowledge, this approach is nonstandard.

# B.6 Does the replacement technique lead to flexible horizons in full-sequence diffusion?

A naive way to obtain fexible horizon generation in full-sequence diffusion is via the "replacement trick" consider a full sequence model trained to diffuse $\mathbf { x } _ { 1 : T }$ , which we partition into $\mathbf { x } _ { 1 : t - 1 , \mathbf { x } _ { t : T } } ]$ . Having diffused tokens $\mathbf { x } _ { 1 : t - 1 }$ , we can attempt to denoise tokens of the form $[ \tilde { \mathbf { x } } _ { 1 : t - 1 } ^ { \bar { k } } , \mathbf { x } _ { t : T } ^ { k } ]$ , where we fix $\tilde { \mathbf { x } } _ { 1 : t - 1 } ^ { k } = \mathbf { x } _ { 1 : t - 1 }$ to be the previously generated token, and only have score gradients update the remaining $\mathbf { x } _ { t : T } ^ { k }$ .One clear disadvantage  this method eiency  e stil nees todiffuse the whol equn even when ther se step left at $t = T - 1$ W   p byreplacement", is both mathematially unprincipled and can lead to inconsistency in the generate seque. The best fix proposed by [32] incorporates an additional gradient term with respect to $\mathbf { x } _ { t : T }$ at every diffusion se;th  i coplete nsufe he putain cos an extr backwar propagationo v sampling step.

# B.7 Further connection to Bayesian filtering

The core ideaof Diffusion Forcing can be interpreted as usingdiffusion toconstruct aninterpolation betwee prior distribution and posterior distribution of a Bayes filter. Consider the hybrid distribution $p ( \mathbf { z } _ { t } | \mathbf { z } _ { t - 1 } , \mathbf { x } _ { t } ^ { k } )$ When $k = 0$ , this hybrid distribution becomes the posterior $p ( \mathbf { z } _ { t } | \mathbf { z } _ { t - 1 } , \mathbf { x } _ { t } )$ On the other hand, when $k = K$ the hybrid distribution becomes $p ( \mathbf { z } _ { t } | \mathbf { z } _ { t - 1 } , \mathbf { n } )$ for $\mathbf { n } \sim \mathcal { N } ( 0 , \mathbf { I } )$ .Since the independent Gaussian noise term n contains no information about $\mathbf { z }$ , this is exactly the prior distribution $p ( \mathbf { z } _ { t } | \mathbf { z } _ { t - 1 } )$ .By varying $k$ between $K$ and 0, the same neural network can parameterize everything between prior and posterior.

# B.8 Connection to other sequence training schemes

Nois s masking provides a unif viw of different sequence trainng chemes. The owig exposition uses a length 3 sequence as an example: We always start with fully masked sequence $[ \mathbf { x } _ { 1 } ^ { K } , \mathbf { x } _ { 2 } ^ { K } , \mathbf { x } _ { 3 } ^ { K } ]$ with the goal oi t en suze $[ \mathbf { x } _ { 1 } ^ { 0 } , \mathbf { x } _ { 2 } ^ { 0 } , \mathbf { x } _ { 3 } ^ { 0 } ]$ sln p  3 DDIM.

Autoregressive. In teacher forcing, one trains a model to predict the next token conditioned on prior observations. One can train next-token diffusion models with teacher forcing such as [50]: feed neural network with past observations as wel as a currentbservation and ask t to predi cen current bservation.A tyial trainig pair an have the ipu o $[ \mathbf { x } _ { 1 } ^ { 0 } , \mathbf { x } _ { 2 } ^ { 0 } , \mathbf { x } _ { 3 } ^ { K } ] ^ { \top }$ and target of $[ \mathbf { x } _ { 1 } ^ { 0 } , \mathbf { x } _ { 2 } ^ { 0 } , \mathbf { x } _ { 3 } ^ { 0 } ] ^ { \top }$ .

Atsampling time, onefullydiffues the next token beforeadding the diffuse observation thistoryto perorm an autoregressive rollout. The diffusion process would thus look like

$$
\begin{array} { r l } & { \mathbf { \tilde { \Gamma } } [ \mathbf { x } _ { 1 } ^ { K } , \mathbf { x } _ { 2 } ^ { K } , \mathbf { x } _ { 3 } ^ { K } ] ^ { \top } } \\ & { \mathbf { \tilde { \Gamma } } [ \mathbf { x } _ { 1 } ^ { K / / 2 } , \mathbf { x } _ { 2 } ^ { K } , \mathbf { x } _ { 3 } ^ { K } ] ^ { \top } , } \\ & { \mathbf { \tilde { \Gamma } } [ \mathbf { x } _ { 1 } ^ { K } , \mathbf { x } _ { 2 } ^ { K } , \mathbf { x } _ { 3 } ^ { K } ] ^ { \top } , } \\ & { \mathbf { \tilde { \Gamma } } [ \mathbf { x } _ { 1 } ^ { 0 } , \mathbf { x } _ { 2 } ^ { K / 2 } , \mathbf { x } _ { 3 } ^ { K } ] ^ { \top } } \\ & { \mathbf { \tilde { \Gamma } } [ \mathbf { x } _ { 1 } ^ { 0 } , \mathbf { x } _ { 2 } ^ { K / 2 } , \mathbf { x } _ { 3 } ^ { K } ] ^ { \top } } \\ & { \mathbf { \tilde { \Gamma } } [ \mathbf { x } _ { 1 } ^ { 0 } , \mathbf { x } _ { 2 } ^ { 0 } , \mathbf { x } _ { 3 } ^ { K } ] ^ { \top } , } \\ & { \mathbf { \tilde { \Gamma } } [ \mathbf { x } _ { 1 } ^ { 0 } , \mathbf { x } _ { 2 } ^ { 0 } , \mathbf { x } _ { 3 } ^ { K / 2 } ] ^ { \top } , } \\ & { \mathbf { \tilde { \Gamma } } [ \mathbf { x } _ { 1 } ^ { 0 } , \mathbf { x } _ { 2 } ^ { 0 } , \mathbf { x } _ { 3 } ^ { 0 } ] ^ { \top } . } \end{array}
$$

Notably, Difuion Foran so permhismpiheme slinmeoplcatons i learning, when one wants to diffuse the next action as fast as possible.

Full Sequence Diffusion.Full sequence diffusion models accept a noisy sequence and denoises level-by-level

$$
\begin{array} { r l } & { [ { \bf x } _ { 1 } ^ { K } , { \bf x } _ { 2 } ^ { K } , { \bf x } _ { 3 } ^ { K } ] ^ { \top } } \\ & { [ { \bf x } _ { 1 } ^ { K / / 2 } , { \bf x } _ { 2 } ^ { K / / 2 } , { \bf x } _ { 3 } ^ { K / / 2 } ] ^ { \top } , } \\ & { [ { \bf x } _ { 1 } ^ { 0 } , { \bf x } _ { 2 } ^ { 0 } , { \bf x } _ { 3 } ^ { 0 } ] ^ { \top } . } \end{array}
$$

Notably, Diffusion Forcing can also perform this sampling scheme at sampling time.

Diffusion Forcing with causal uncertainty As shown in Figure 2, to model causal uncertainty, Diffusion Forcing keeps the far future more uncertain than the near future by having a larger noise level $k$ , at any time of diffusion. An example pattern looks like this:

$$
\begin{array} { r l } & { \left[ { \mathbf { x } } _ { 1 } ^ { K } , { \mathbf { x } } _ { 2 } ^ { K } , { \mathbf { x } } _ { 3 } ^ { K } \right] ^ { \top } } \\ & { \left[ { \mathbf { x } } _ { 1 } ^ { K / / 2 } , { \mathbf { x } } _ { 2 } ^ { K } , { \mathbf { x } } _ { 3 } ^ { K } \right] ^ { \top } , } \\ & { \left[ { \mathbf { x } } _ { 1 } ^ { 0 } , { \mathbf { x } } _ { 2 } ^ { K / / 2 } , { \mathbf { x } } _ { 3 } ^ { K } \right] ^ { \top } , } \\ & { \left[ { \mathbf { x } } _ { 1 } ^ { 0 } , { \mathbf { x } } _ { 2 } ^ { 0 } , { \mathbf { x } } _ { 3 } ^ { K / / 2 } \right] ^ { \top } } \\ & { \left[ { \mathbf { x } } _ { 1 } ^ { 0 } , { \mathbf { x } } _ { 2 } ^ { 0 } , { \mathbf { x } } _ { 3 } ^ { 0 } \right] ^ { \top } } \\ & { \left[ { \mathbf { x } } _ { 1 } ^ { 0 } , { \mathbf { x } } _ { 2 } ^ { 0 } , { \mathbf { x } } _ { 3 } ^ { 0 } \right] ^ { \top } } \end{array}
$$

Notabe, [66] is the rst one to proose such a lnearuncertainty sampling heme o causal diffusion models, although Diffusion Forcing provides a generalization of such scheme in combination with other abilities.

Diffusion Forcing with stablization Previously we introduced the autoregressive sampling scheme that Diffusion Forcing can also do. However, such a scheme can accumulate single-step errors because it treats predicted $\mathbf { x }$ as ground truth observation. Diffusion Forcing addresses this problem by telling the model that generated images should be treated as noisy ground truth, as shown in 2.

It first fully diffuses the first token,

$$
\begin{array} { r l } & { [ { \bf x } _ { 1 } ^ { K } , { \bf x } _ { 2 } ^ { K } , { \bf x } _ { 3 } ^ { K } ] ^ { \top } } \\ & { [ { \bf x } _ { 1 } ^ { K / / 2 } , { \bf x } _ { 2 } ^ { K } , { \bf x } _ { 3 } ^ { K } ] ^ { \top } , } \\ & { [ { \bf x } _ { 1 } ^ { 0 } , { \bf x } _ { 2 } ^ { K } , { \bf x } _ { 3 } ^ { K } ] ^ { \top } } \end{array}
$$

Then, it feed the diffused ${ \bf x } _ { 1 } ^ { 0 }$ iothe ode but t  isghtly highe oise v ${ \bf x } _ { 1 } ^ { 1 }$ to diffuse $\mathbf { x } _ { 2 }$ .

$$
\begin{array} { r l } & { [ \mathbf { x } _ { 1 } ^ { 1 } , \mathbf { x } _ { 2 } ^ { K / / 2 } , \mathbf { x } _ { 3 } ^ { K } ] ^ { \top } } \\ & { [ \mathbf { x } _ { 1 } ^ { 1 } , \mathbf { x } _ { 2 } ^ { 0 } , \mathbf { x } _ { 3 } ^ { K } ] ^ { \top } } \end{array}
$$

Then, it feeds the diffused $\mathbf { x } _ { 2 } ^ { 0 }$ intothe model but tells   o ahigher noiseleve, a ${ \bf x } _ { 2 } ^ { 1 }$ .

$$
\begin{array} { r l } & { [ \mathbf { x } _ { 1 } ^ { 1 } , \mathbf { x } _ { 2 } ^ { 1 } , \mathbf { x } _ { 3 } ^ { K / / 2 } ] ^ { \top } , } \\ & { [ \mathbf { x } _ { 1 } ^ { 1 } , \mathbf { x } _ { 2 } ^ { 1 } , \mathbf { x } _ { 3 } ^ { 0 } ] ^ { \top } . } \end{array}
$$

# C Extended Related Work

Reconstructing masked tokens. Masked Autoencoders for images [27] and videos [17] are a popular method for representation learning in pixel space. They have been extended to perform diffusion to generate masked patches conditioned on unmasked ones [63, 19].

Casting Image Generation as Sequence Generation. [61, 8] show that even generative modeling of non-sequential data, such as images, can be fruitfully cast as sequence generative modeling.

Non-Diffusion Probabilistic Sequence Models. [12] parameterize token-to-token transitions via a variational auto-encoder. This makes them probabilistic, but does not directly maximize the joint probability of sequences, but rather, enables sampling from the distribution of single-step transitions.

Sequence Diffusion with Varying Noise Levels. Most similar to our work is AR-Diffusion [66] which similarly aims to train next-token prediction models for sequence diffusion. Key differences are that ARDiffusion proposes a noise level that is linearly dependent on the position of each word in the sequence, while our critical contribution is to have each noise level be independent, as this uniquely enables our proposed sampling schemes, such as stabilizing auto-regressive generation and conditioning on corrupted observations. Further, AR-Diffusion only explores language modeling and does not explore guidance, while we investigate Difson Forigasa broal pplicableequecegeerative model with particular aplications t seqnal decision-making. In particular, we introduce Monte-Carlo Guidance as a novel guidance mechanism. Another closely related work is Rolling Diffusion [52], which proposes to diffuse a sequence with near future more c n  tu mon bln e  a mp he  D LikeAR-Diffussion, Rolling Diffusion's training noise levels are linearly dependent on the positions of tokens and must use the exact same noise level scheme at sampling time. It, therefore, shares the aforementioned limitations of AR-Diffusion as well.

# D Additional Method Details

# D.1 Fused SNR reweighting

SNR reweighting [24] is a widely used technique to accelerate the convergence of image diffusion models. In short, it reweighs the diffusion loss proportional to the signal-to-noise ratio (SNR) of noisy $\mathbf { x } ^ { k }$ . In Diffusion For cppe,in diable $\mathbf { z } _ { t - 1 }$ ca hovian syis an or has its nofii fovel b $\mathbf { x } _ { t }$ o stdio st to $\mathbf { x } _ { t } ^ { k _ { t } }$ $\mathbf { x } _ { t - 1 } ^ { k _ { t - 1 } }$ $k _ { t - 1 } = 0$ $\mathbf { z } _ { t - 1 }$ conains e $\mathbf { x } _ { t } ^ { 0 }$ rardless e $\mathbf { x } _ { t } ^ { k _ { t } }$ .

Therefore we re-derive SNR reweighting to reflect this change in Diffusion Forcing. We call this technique Fused SNR reweighting. We follow the intuition of original SNR reweighting to loosely define SNR in a seqence withindependent levels of noises at different time steps. Denote $S _ { t }$ as the normalized SNR reweighting factor for $\mathbf { x } _ { t } ^ { k _ { t } }$ flwin its noral derivation i diffusion models. For exaple, i one uss min r ra [24], its reweighting factor will always fall between $[ 0 , C ]$ which we divide by $C$ to get $S _ { t } \in [ 0 , 1 ]$ . Define signal decay factor $0 < \gamma < 1$ $\mathbf { x } _ { t - 1 } ^ { k _ { t - 1 } }$ contribute to denoising $\mathbf { x } _ { t } ^ { k _ { t } }$ This is the simple exponential decay model of sequential information. Now, define cumulated SNR recursively as the running mean of $S _ { t }$ : $\bar { S } _ { t } = \gamma \bar { \bar { S } } _ { t - 1 } + ( 1 - \gamma ) \bar { S } _ { t }$ to account for signals contributed by the entire noisy history to the denoising at time step $t$ $S _ { t }$ of noisy observation $\mathbf { x } _ { t } ^ { k _ { t } }$ To combine them, we use a simplified model for independent events. Notice $S _ { t }$ and $\bar { S } _ { t }$ always falls in range $[ 0 , 1 ]$ , and therefore can be reinterpreted as probabilities of having all the signal one needs to perfect denoise $\mathbf { x } _ { t } ^ { k _ { t } }$ .Since the noise level at $t$ is independent of prior noise levels, we can viw $S _ { t }$ and $\bar { S } _ { t - 1 }$ as probabilities of independent events and thus can composed to define a joint probability $S _ { t } ^ { \prime } = 1 - ( 1 - S _ { t } ) ( 1 - \bar { S } _ { t - 1 } )$ , and we use this $S _ { t } ^ { \prime }$ as our fused SNR reweighting factor for diffusion training.

In our experiments, we choose to follow the min-SNR reweighting strategy [24] to derive the $S$ Our Fused SNR reweighting proves extremely useful to accelerate the convergence of video prediction, while we didn't observe a boost on non-image domains so we didn't use it there.

# D.2 Architecture

Video Diffusion We choose both the raw image $\mathbf { x }$ and latent state $\mathbf { z }$ to be 2D tensors with channel, width, and height. For simplicity, we use the same width and height for $\mathbf { x }$ and $\mathbf { z }$ . We then implement the transition model $p ( \mathbf { x } _ { t } ^ { k _ { t } } | \mathbf { z } _ { t - 1 } )$ w  ypi i U- 7. W e u  he -  hu recurrent unit (GRU) and use $\mathbf { z } _ { t - 1 }$ as the hidden state feed into a GRU. The output of GRU is treated as $\mathbf { z } _ { t }$ . For observation model $p ( \mathbf { x } _ { t } | \mathbf { z } _ { t } )$ , we use a 1-layer resnet [28] followed by a conv layer. We combine these two models to create an RNN layer, where the latent of a particular time step is $\mathbf { z } _ { t - 1 }$ , input is $\mathbf { x } _ { t } ^ { k _ { t } }$ and output is $\hat { \bf x }$ . One can potentiallyobtain better results by rainig Diffusion Forcig with a causal transformer architecture However, since RNN is more eficient for online decision-making, we also stick with it for video prediction and it already gives us satisfying results.

We choose the number of channels in $\mathbf { z }$ to be 16 for DMlab and 32 for Minecraft. In total, our Minecraft model consists of 36 million parameters and our DMlab model consists of 24 million parameters. We can potentially ha the training duration reasonable ( $\leq 1$ day). In maze planning, the number of total parameters is 4.33 million.

Non-Video Diffusion For non-spatial $\mathbf { x }$ that is not video nor images, we use residue MLPs [60] instead of Unet as the backbone for the dynamics model. Residue MLP is basically the ResNet [28] equivalent for MLP. Similar to video prediction, we feed the output of resMLP into a GRU along with $\mathbf { z } _ { t - 1 }$ to get $\mathbf { z } _ { t }$ Another ResMLP serves as the observation model.

# D.3 Diffusion parameterization

In diffusion models, there are three equivalent prediction objectives, $\mathbf { x } _ { 0 }$ , $\epsilon$ [29], and $v$ parameterization [53]. Difn iv   t i ove te For example, $\epsilon$ parameterization and $v$ parameterization are essential in generating pixel data that favors high-frequency details.

In our experiments, we use $v$ parameterization for video prediction and found it essential to both convergence speed and quality.

We observe that $\mathbf { x } _ { 0 }$ pt v n y don't favor an artificial emphasis on high-frequency details.We observe the benefits ofv-parameterizationin time-series prediction.

# D.4 Noise schedule

We use sigmoid noise schedule [9] for video prediction, linear noise schedule for maze planning, and cosine schedule for everything else.

# D.5 Implementation Details of Sampling with Guidance

Corner case of sampling noise In our sampling algorithm, due to the flexibility of the scheduling matrix $\kappa$ , there are corner cases when $\mathbf { x } _ { t } ^ { k _ { t } }$ is required to stay at its same noise level during a sampling step. The core q $\cdot _ { t } ^ { k _ { t } }$ O  is   ve valThet  yb toresample under the diffusion process. While we conclude this can be an open question, we prefer the later apprach, resamping, and us  in Mont Carlo Guidance to generate multiple samples.We note that even one takes the first approach, the guidance gradient can still flow back in the time steps before $t$ as the dynamics model $p ( \mathbf { z } _ { t } | \mathbf { x } _ { t } ^ { k _ { t } } , \mathbf { z } _ { t - 1 } )$ l $\mathbf { z } _ { t - 1 }$ .

Other than Monte Carlo Guidance, this corner case only happens when $k _ { t } = 0$ or $k _ { t } = K$ throughout our experiments. That is, we chose our $\kappa$ such that once any token gets diffused slightly, it will keep diffusing. In the case of $k _ { t } = K$ , keeping $\mathbf { x } _ { t } ^ { k _ { t } }$ at the same noise level implies it will stay as white noise, and we don't even need to sample another white noise. In case $k _ { t } = 0$ , the time step is already completely diffused either approach should give us the same result so we just opt for copying over for simplicity.

Guidance for maze planning In maze planning, our main baseline Diffuer [37] discards the reward from the dataset and directly plans with the goal position and velocity. We adopt the same convention for Diffusion Forcing. One can perform guidance on goal position using log-likelihood $| | \mathbf { p } _ { T } - \mathbf { g } | |$ , but a flexible horizon model should not require users to manually specify a $T$ to reach its goal, instead we want it to try to reach the goal for any possible horizon. Therefore we use the reward model $\textstyle \sum _ { t } | | \mathbf { p } _ { T } - \mathbf { g } | |$ so any time step can be the fl   ec he lhs jiv alnu he o-v au maz, ut Diffsion Forcing can still reliably fnd plans without bumping into wall. However, we also observe that the aent tend to leave the goal location due to the natureof the provided dataset - he goal location is just e possible waypoint for the robot to pass through, and there are no trajectoris that simpy stay at the goal We also tried this reward for guidance with Diffuser, but it didn't work even with a good amount of tuning.

# D.6 Performance Optimization

Accelerating the diffusion smpling DiffsinForci is similar to that of normal diffusion models.Wedopt DDIM [58] sampling for the diffusion of each token. While we use $K = 1 0 0 0$ steps of diffusion, we sample with only 100 DDIM for video prediction and 50 for non-video domains.

While Diffusion Forcing can be implemented with transformers, we use an RNN as the backbone for Diffusion Forcing experiments it's widely used in decision-making for its flexibility and efficiency in online decisionmaking systems. To further reduce training time and GPU memory usage, we use frame-stacking to stack multiple served mages as agle $\mathbf { x }$ redng he memoi t hie s n   thisWe ee tha it' aseul  e o ut he model multiple times to generate almost identical tokens. For video datasets, we manually examine how many tm sestakeequiil ve rect powstepyim vr.Ther reason why we use frame stacking - many diffusion model techniques such as different noise schedules are designed to model $\mathbf { x }$ with correlated elements or redundancy. Low-dimensional systems may need drastically dient hyperparameers when they lack the data redundancy these techniques aretested on. Frame sackingis thus also helpful for our non-image experiments so we can start with canonical hyperparameters of diffusion models. We use a frame stack of 4 for DMlab video prediction, 8 for Minecraft, and 10 for maze planning.

At sampling time, we also have a design choice toreduce compute usage, as reflected in line 8 o Algorith 2. In line 8, e irectly assign $\mathbf { z } _ { t } ^ { \mathrm { n e w } }$ $\mathbf { z } _ { t }$ , instead of recalculating $\mathbf { z } _ { t }$ with posterior model $p ( \mathbf { z } _ { t } | \mathbf { z } _ { t - 1 } , \mathbf { x } _ { t } ^ { \mathrm { n e w } } , k - 1 )$ Since the model is trained to condition on $\mathbf { z } _ { t }$ estimated from arbitrary noisy history, we recognize that both T computing posterior every step. Second, this happens to be what we want for stabilization- $\mathbf { z } _ { t } ^ { \mathrm { n e w } }$ already contains the iormatio $\mathbf { x } _ { t } ^ { \mathrm { { n e w } } }$ p $k = k _ { t }$ ,a n $\mathbf { x } _ { t } ^ { \mathrm { { n e w } } }$ This hapens plment e behav e want  stabizan.

# D.7 Sampling schedule for causal uncertainty

Inernc sdepicAlgorhm  n igure. In Eqain . weillustrat  secstantatio he $\kappa$ matrix we used for causal planning. For simplicity, we denote the case where a latent $\mathbf { z } _ { 0 }$ is given and aim to generate $\mathbf { x } _ { 1 : H + 1 }$ .

$$
\begin{array} { r } { K ^ { \mathrm { p y r a m i d } } = \left[ \begin{array} { c c c c c c } { K } & { K } & { K } & { K } & { \ldots } & { K } \\ { K - 1 } & { K } & { K } & { \ldots } & { K } \\ { K - 2 } & { K - 1 } & { K } & { \ldots } & { K } \\ { \vdots } & { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { 1 } & { 2 } & { 3 } & { \ldots } & { H } \\ { 0 } & { 1 } & { 2 } & { \ldots } & { H - 1 } \\ { \vdots } & { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { 0 } & { 0 } & { 0 } & { \ldots } & { 1 } \\ { 0 } & { 0 } & { 0 } & { \ldots } & { 0 } \end{array} \right] } \end{array}
$$

Diffusion Forcing begins by sampling our sequences as white noise with noise level $K$ It then denoises along each row $m = 1 , \ldots , M$ of $\kappa$ in decreasing order. It does so by proceeding sequentially through frames $t = 1 , \dots , T$ , updating the latent (Line 5 of Algorithm 2), and then partially applying the backward process to noise level $k = \mathcal { K } _ { m , t }$ dictated by the scheduling matrix $\kappa$ (Line 6-7 of Algorithm 2). We call a $\kappa$ like this pyramid scheduling, as the tokens in the far future are kept at higher noise level than near future.

# D.8 Metrics for Maze Planning

We report the episode reward o Diffusion Forcingfor different maze planning environments in Table . However, we found that the episode reward isn't necessarily a good metric Intuitively, maze planning should reward sma agents that canfind thefastest routeto the l, not a slow-walking agent that ges there at the end the episode. The dataset never contains data on the behavior of stayin at the goal,o agents are supposeto walk away after reaching the goal with sequence planning methods. Diffuser may had an unfair advantage of just generating slow plans, which happens to let the agent stay in the neighborhood of the goal for more steps and get a very high reward as a result. This metricsees to exploit faws in the environment design - a good desg woul involvea penaltyo longer time taken o reach the goal.Therefore, in future works based ur pa wec leaiveii he tim t ak  each he lor herst tme whic Dif Forcing excels at.

![](images/7.jpg)  
Figure 6: Prediction intervals of Diffusion Forcing for the first prediction window of the test set in the Electricity time series dataset. Only the first 16 features out of 370 are plotted.

# D.9 Implementation Details of Timeseries Regression

We follow heplementation  pytorch- where the validation  is arandomsubse the training wih the same number  sequences as the test et. We use arly stopping whenvalidation cps-sum has'inces for 6 epochs. We leverage the same architecture (1 mlp and 4 grus) as well as a batch size of 32.

# D.10 Compute Resources

All of our experiments use $f p 1 6$ mixed precision training. Time series, maze planning, compositionally, and visual imitation experiments can be trained with a single $2 0 8 0 T i$ with 11GB of memory. We tune the batch sizsuch that we ully use the memory of GPUs.This translates to a batch sie f 2048 or maze planni and compositional experiments, and 32 for visual imitation learning. While we use early stopping on the validation set for time series experiments, we did not carefully search for the minimal number of training steps required, though the model usually converges between 50k to $1 0 0 k$ steps. The above environments thus usually take $4 - 8$ hours to train although there is without doubt a significant potential for speed up.

Video prediction is GPU intensive. We use 8 A100 GPUs for both video prediction datasets. We train for $5 0 K$ steps with a batch size of $8 \times 1 6$ . It usually takes 12 hours to converge at $4 0 K$ steps of training (occasional validation time also included).

# E Additional Experiment Results

# E.1 Multivariate Probabilistic Time Series Forecasting

To illustrate Diffusion Forcing's new training objective does not degrade it as a generic sequence model, we evaluate Diffusion Forcing on high-dimensional and long-horizon sequence prediction tasks in time series prediction. We adopt multiple time series datasets with real-world applications from GluonTS [2] and evaluate Diion For with stonbaselines wistanarmetri thismai In this ecion,e aiy the results nd nalyisFor detaildesiptonataset and hemei eerhereaderppeF..

Problem Formulation Let ${ \cal X } ~ = ~ \{ { \bf x } _ { t } \} _ { t = 1 } ^ { T }$ iv $D$ dimensional observations $\mathbf { x } _ { t } \in \mathbb { R } ^ { D }$ of some underlying dynamical process, sampled in discrete time steps $t \in \{ 1 , \ldots , T \}$ , where $T \in \mathbb N$ . In the problem setting of probabilistic time series forecasting, the sequence $X = \{ X _ { c } , X _ { p } \}$ is $t _ { 0 } \in \mathbb { N }$ with $1 < t _ { 0 } \le T$ e ontext window $X _ { c } : = \{ \mathbf { x } _ { t } \} _ { t = 1 } ^ { t _ { 0 } - 1 }$ (aals call histry ech $t _ { 0 } - 1$ , and the prediction window $X _ { p } : = \left\{ \mathbf { x } _ { t } \right\} _ { t = t _ { 0 } } ^ { T }$ of length $T - t _ { 0 } + 1$ (als nown as the prediction horizon). Then, the task is to model the conditial joint probabiliy distrut

$$
q ( \mathbf { x } _ { t _ { 0 } : T } \mid \mathbf { x } _ { 1 : t _ { 0 } - 1 } ) : = \prod _ { t = t _ { 0 } } ^ { T } q ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 1 : t - 1 } )
$$

over the samples in the prediction window. If we know the distribution in (E.1), we can sample forecast prediction sequences given some initial context from the evidence sequence. However, most time-dependent data generation proceses i ature hav complex dynamic and no tractable formulatin $q \big ( \mathbf { x } _ { t _ { 0 } : T } \mid \mathbf { x } _ { 1 : t _ { 0 } - 1 } \big )$ . Insteonstruc  atistilmode ha piatesheerai proces  (E.1n eate via Monte Carlo sampling of simulated trajectories. In this way, confidence levels or uncertainty measures can be calculated, and point forecasts can be produced as the mean or median trajectory [36].

Table 2: Results for time series forecasting. We report the test set $\mathrm { C R P S _ { s u m } }$ (the lower, the better) of comparable methods on six time series datasets. We measure the mean and standard deviation of our method from five runs trained with different seeds.   

<table><tr><td>Method</td><td>Exchange</td><td>Solar</td><td>Electricity</td><td>Traffic</td><td>Taxi</td><td>Wikipedia</td></tr><tr><td>VES [36]</td><td>0.005 ± 0.000</td><td>0.900 ± 0.003</td><td>0.880 ± 0.004</td><td>0.350 ± 0.002</td><td></td><td></td></tr><tr><td>VAR [45]</td><td>0.005 ± 0.000</td><td>0.830 ± 0.006</td><td>0.039 ± 0.001</td><td>0.290 ± 0.001</td><td></td><td></td></tr><tr><td>VAR-Lasso [45]</td><td>0.012 ± 0.000</td><td>0.510 ± 0.006</td><td>0.025 ± 0.000</td><td>0.150 ± 0.002</td><td></td><td>3.100 ± 0.004</td></tr><tr><td>GARCH [62]</td><td>0.023 ± 0.000</td><td>0.880 ± 0.002</td><td>0.190 ± 0.001</td><td>0.370 ± 0.001</td><td></td><td></td></tr><tr><td>DeepAR [55]</td><td></td><td>0.336 ± 0.014</td><td>0.023 ± 0.001</td><td>0.055 ± 0.003</td><td></td><td>0.127 ± 0.042</td></tr><tr><td>LSTM-Copula [54]</td><td>0.007 ± 0.000</td><td>0.319 ± 0.011</td><td>0.064 ± 0.008</td><td>0.103 ± 0.006</td><td>0.326 ± 0.007</td><td>0.241 ± 0.033</td></tr><tr><td>GP-Copula [54]</td><td>0.007 ± 0.000</td><td>0.337 ± 0.024</td><td>0.025 ± 0.002</td><td>0.078 ± 0.002</td><td>0.208 ± 0.183</td><td>0.086 ± 0.004</td></tr><tr><td>KVAE [41]</td><td>0.014 ± 0.002</td><td>0.340 ± 0.025</td><td>0.051 ± 0.019</td><td>0.100 ± 0.005</td><td></td><td>0.095 ± 0.012</td></tr><tr><td>NKF [14]</td><td></td><td>0.320 ± 0.020</td><td>0.016 ± 0.001</td><td>0.100 ± 0.002</td><td></td><td>0.071 ± 0.002</td></tr><tr><td>Transformer-MAF [51]</td><td>0.005 ± 0.003</td><td>0.301 ± 0.014</td><td>0.021 ± 0.000</td><td>0.056 ± 0.001</td><td>0.179 ± 0.002</td><td>0.063 ± 0.003</td></tr><tr><td>TimeGrad [50]</td><td>0.006 ± 0.001</td><td>0.287 ± 0.020</td><td>0.021 ± 0.001</td><td>0.044 ± 0.006</td><td>0.114 ± 0.020</td><td>0.049 ± 0.002</td></tr><tr><td>ScoreGrad sub-VP SDE [68]</td><td>0.006 ± 0.001</td><td>0.256 ± 0.015</td><td>0.019 ± 0.001</td><td>0.041 ± 0.004</td><td>0.101 ± 0.004</td><td>0.043 ± 0.002</td></tr><tr><td>Ours</td><td>0.003 ± 0.001</td><td>0.289 ± 0.002</td><td>0.023 ± 0.001</td><td>0.040 ± 0.004</td><td>0.075 ± 0.002</td><td>0.085 ± 0.007</td></tr></table>

Results. We evaluate the effectiveness of Diffusion Forcing as a sequence model on the canonical task of multivariate time series forecasting by following the experiment setup of [54, 51, 50, 59, 68] Concretely, we benmar Diffusion Forcion the atasets Solar, Electricity, Trac, Taxi an WikipeiaThesatases ave diferent dimensionality, domains, and samplingfrequencis, nd captureesnal patterns  different enghs. The features of each dataset are detailed in Table 3.We access the datasets from GluonTS [2], and set the context and prediction windows to the same length or each dataset. Additionally, we employ the same covariates as [50]. We evaluate the performance of the model quantitatively by estimating the Summed Continuous Ranked Probability Score $\mathrm { C R P S _ { s u m } }$ via quantiles. As a metric, $\mathrm { C R P S _ { s u m } }$ measures how well a forecast distribution matches the ground truth distribution. We provide detailed descriptions of the metric in Appendix F.4. We benchmark with other diffusion-based methods in time series forecastings, such as TimeGrad [50] and the transorm-based Transormer-MAF [51]. In particular, the main baseline interest, TmeGrad [50], is a exttoken diffusion sequence model trained with teacher forcing. We track the $\mathrm { C R P S _ { s u m } }$ metric on the validation set and use early stopping when the metric has not improved for 6 consecutive epochs, while all epochs are fixed to 100 batches across datasets. We then measure the $\mathrm { C R P S _ { s u m } }$ on the test set at the end of the training, which we report in Table . We use the exact same architecture and hyperparameters or al time series datasets and experiments. Diffusion Forcing outperorms all prior methods except for [68] with which Diffusion Forcing is overall tied, except for the Wikipedia dataset, on which Diffusion Forcing takes fourth place. Note that time seri is nothe coreapplication  Diffuin Forci, and that w merel seek todemnstrate that h Diffsn Forig objective is applicable to diverse domains with o apparent trade-off in performance over baseline objectives.

# E.2 Additional results in compositional generation

Since Diffusion Forcing models the joint distribution of any subset o a sequence, we can leverage thisnique property  achievecomposiional behavior i Diffusio Forci can samplfromthe stribution  s of the trajectory and compose these sub-trajectories into new trajectories.

In particular, we show that we can also have flexible control over how compositional Diffusion Forcing is. As show cnsieratase rajecoriesn D suare plane wherltrajcori tartrome and end up in the opposite corner, forming a cross shape. When no compositional behavior is desired one can let the models replicate the cross-shaped distribution by allowing full memory of the HMM model. When one desire compositional suc as enerating aVshaped trajectory whic stitches twosub-rajectoris together,e can let the model generate shorter plans with no-memory context using MPC. (Add figures).

![](images/8.jpg)  
Figure 7: Given a dataset of trajectories (a), Diffusion Forcing models the joint distribution of all subsequences of arbitrary length. At sampling time, we can sample from the trajectory distribution by sampling Diffusion Forcing with full horizon (b) or recover Markovian dynamics by disregarding previous states (c).

# E.3 Additional results in video prediction (wo/ cherry picking)

Infinite Rollout without sliding window Diffusion Forcing can rollout longer than maximum training hozoihout id indow.Tha is, we Diffusion For'R cntuusy hout vereinlz $\mathbf { z } _ { 0 }$ .This is a surprising effect we observed from the rollout stabilization property of Diffusion Forcing. In Figure 8, 10, we use Diffusion Forcing to generate video sequences of length 180 and visualize subsampled sequences. Notably, Diffusion Forcing used in these visualizations is trained with a maximum length of 72 frames for Minecraft and 36 frames for DMLab, illustrating it can rollout $2 \mathbf { X } ^ { - 5 \mathbf { X } }$ times longer than it's trained on without sliding window. In ddition, we also tried rollig these modes out or 0 frames and without seei the model blowing up on both datasets. There are ccasional cases where the Minecraft agent gets stuck and the e turns around.

![](images/9.jpg)  
Figure 8: Visualization shows Diffusion Forcing trained on 72 frames is able to rollout 180 frames on Minecraft dataset without sliding window. The visualization shows a non-cherry-picked subsampling of these 180 frames, although Diffusion Forcing can roll out much longer (such as 2000 frames) on this dataset.

Consistency We also present additional results where we only generate within our maximum training length.   
As shown in figure 13 12, Diffusion Forcing can generate consistent videos. Results are not cherry-picked.

![](images/10.jpg)  
Figure 9: Diffusion Forcing trained on 72 frames is able to rollout 180 frames on Minecraft dataset without sliding window. The visualization shows a non-cherry-picked subsampling of these 180 frames, although Diffusion Forcing can roll out much longer (such as 2000 frames) on this dataset. The first fewrames marked in red are the round truthimages of the dataset usedor conditnin.

# E.4 Additional results in planning

We provide some additional visualizations of causal planning in1. Wealso present additional visualization Diffusion Forcing performing model predictive control in action. As shown in figure 14, Diffusion Forcing can generate plans of shorter horizons since it's flexible horizon.

![](images/11.jpg)  
Figure 10: Visualization shows Diffusion Forcing trained on 36 frames is able to rollout 180 frames on DMLab dataset without sliding window. The visualization shows a non-cherry-picked subsampling of these 180 frames, although Diffusion Forcing can roll out almost infinitely on this dataset. The first few frames marked in red are the ground truth images of the dataset used for conditioning.

# E.5 Real robot experiment setup

In Figure 16 we visualize ur robot experiment setup with corruption n observation. The dataset is collected ho eent w servati couptThe ypil iluemod is henherobot oon eacts heal clus of the randomized location o objects. We didn bserve the robot act wildly due to visual distractors.

# F Additional details about datasets

# F.1 Dataset for video diffusion

We adopt the video prediction dataset Minecraft and DMlab used by TECO[69].

![](images/12.jpg)  
Figure 11: Visualization shows Diffusion Forcing trained on 36 frames is able to rollout 180 frames on DMLab dataset without sliding window. The visualization shows a non-cherry-picked subsampling of these 180 frames, although Diffusion Forcing can roll out almost infinitely on this dataset. The first few frames marked in red are the ground truth images of the dataset used for conditioning.

Minecraft Navigation The Minecraft navigation dataset consists of first-person-view videos of random wak in the Minera wa'bioeThe agent walks vi  tenique calle prt jupwhic allws it to jup aross blocks without gettig stuck at 1 block obstaces. The agent walks straight most of the time, small chan o turnig l r iht.The heiht and width of thevideo s 8 pixels and we trim on videos o susequences of 72 rames. Thedataset comes with paire action data but we discard them o more stochasticity to the prediction task. Due to limited compute, we only train on about $1 0 \%$ of the total subsequences.

One problem we noticed about the dataset is when the agent runs into obstacles with a height of 2 blocks or In hisahen   sucn heii onyn pat brown dirty patterns. This leads to a huge amount of frames with these patterns, making video models predic meaningless frames. Yet, we deem this as a problem of this dataset itself.

![](images/13.jpg)  
Figure 12: Additional non-cherry-picked video prediction results on DMLab dataset, generated within maximum training length. The first few frames marked in red are the ground truth images of the dataset used for conditioning.

DMLab Navigation Deepmind Lab navigation dataset consists of random walks in a 3D maze environment For DMLab, the resolution is 64 pixels and we use subsequences of 48 frames. We also disregard the provided actions due to training.

We note that the VQ-VAE latent that stable video diffusion [4] diffuses is also only $1 2 8 \times 1 2 8 \times 3$ ,indicating Diffusion Forcinghas the potential to scale up to higher resolutionmages with pre-raine image encoer and r D   ee z tetas e y $1 0 \%$ o tol dat sq a due to limited computing, as we observe that doing so already allows us to make good generations from iniial frames from the test set.

# F.2 Dataset for planning

D4RL [18] is a standard offline RL benchmark featuring a wide range of reinforcement learning environments. Eac environments associated with a provideddatase of offnenteractions with the environment eturin state, action, and reward trajectories.

Like Diffuer [37], we choose the 3 maze environments as they arechallenging long-horizon, multi-modal, sparse reward problems uniquely suited for visualization and evaluating planning algorithms. The IDs for the 3 used environments are "maze2d-medium-v1, "maze2d-large-v1", "maze2d-umaze-v1" In each environment, one cols he ratin  obo o wal t owars  The sati spac  disinal, locaton and velocity The actin space is D acceleration. The agent always receives arandom sart laton and the oal is o reach afd gal positnor each maze.The agent reivesa reward   it is ha circle of radius 0.5 centered at the goal state, and 0 otherwise.

![](images/14.jpg)  
Figure 13: Additional non-cherry-picked video prediction results on the Minecraft dataset, generated within maximum training length. The first few frames marked in red are the ground truth images of the dataset used for conditioning.

The offline RL dataset for the maze environments consists of random walks in the maze. Specifically, the atten d tuez i n  en ava waypoints with some randomization. As a result, the random walks are generated in a way that the path is collision-free with the walls. The random walks introduce stochasticity to the dataset, as trajectories in the dataset are never towards a specific goal.

The  oi doptor a ase Df [7] werearheewar he at nd plan wih goals ly.Wealso evaluateamult-ol variant ach evirment labele as mul Ta, where the goal is randomized just like the starting position.

# F.3 Dataset for robot learning

Wechoose a long horizon robotic manipulation task as described in Section 4.4Consider a tabletop with three slots where we an place bjects.One places an apple at slot A or sot andomly, and then places the other sot between A andBA robot is hallenged to swaphe positino twofruits usng the third slot C. Ta  n y uit  ty o   to  we e ps t ot e is at sot  it may ove he appe o slot , leavi slot A mpty Then ove he range tosot A and fnllymove eapp rom sot tosot B. Inurwe illusrat e on-arkovin property  te sk: When the apple is at slot B and the orange is at slot C, one cannot tell what the immediate action is without knowing the initial positions of objects.

We pu ie   tab  r e  or   de beaout double the iameter fruit.To makesure the taskrequires visalfeedback, wealso randmiz the locaion  ruiinside the ot.Weclecte10 exper demstrations  Frankarobot perorminh task using VR teleoperation and impedancecontrol. Among them, each initial slot configuration makes up hal f the dataset. We record videos from two camera views, one from a hand camera and one in the front capturing all three slots. Each demonstration also comes with 6 dof actions of the robot hand. During the data collection, since one successful demonstration wil swap the position of two objects, its end configuration will naturally serve as the starting configuration of the other randomized location, which we leverage to save time.

Each demonstration comprises $5 0 0 - 6 0 0$ frames and actions. We train Diffusion Forcing on the entire sequence However, since adjacent frames are visually lose, we pad and downsample thevideos to 40 frames where each frame is bundled with 15 actions.

# F.4 Dataset for time series

We use a set of time series datasets accessible via GluonTS [2], which are adopted from prior works like [72, 42, 56]. These datasets capture real-world data of high-dimensional dynamics like monetary exchange rates or theelectricity grid. In Table , we provide  smary o the eature o thesedatasets, such s the naliy i, t pleq   heuliva hea, an the predicton lengh.Weaccess he dtasets inTabl vi GluonT and wrap hedata processig c implemented in GluonTS in our own dataloaders. Each dataset consists of one long multivariate sequence, which W sam cardinaliy  the el-out test e s arandomly ample subse usequencs rom the trainin t.All splits are normalized by the mean and the standard deviation of the features in the training split.

![](images/15.jpg)  
Figure 14: Example MPC planning for maze medium environment. Blue indicated trajectories actually executed already. Red is the plan.

![](images/16.jpg)  
Figure 15: Example plans generated for maze medium (above) and maze large (below) environments.

COe saslmols ha)be flly uue $C = \left\{ \mathbf { c } _ { t } \right\} _ { t = 1 } ^ { T }$ seasonal patterns and other temporal dependencies. We follow the implementation in [51] to construct the corateqec ctieeen  ataseTas su urovariat of lagge inputs, as wel as learnedembeddings and handrafted temporal features that encode information such asou y  eeeahe pr being modeled. Therefore, covariates are known for the entire interval $[ 1 , T ]$ , even at inference. We can easily incorporate covariates into the probabilistic framework as

![](images/17.jpg)  
Figure 16: We randomly throw a target bag on the table as a strong visual distractor. Diffusion Forcing can be prompted to treat observation as corrupted rather than ground truth.

Table 3: Characteristics of the GluonTS datasets used to benchmark Diffusion Forcing in the domain of time series forecasting.   

<table><tr><td>Dataset</td><td>Dimension</td><td>Domain</td><td>Frequency</td><td>Steps</td><td>Prediction length</td></tr><tr><td>Exchange</td><td>8</td><td>R+</td><td>BUSINESS DAY</td><td>6,071</td><td>30</td></tr><tr><td>Solar</td><td>137</td><td>R+</td><td>HOUR</td><td>7,009</td><td>24</td></tr><tr><td>Electricity</td><td>370</td><td>R+</td><td>HOUR</td><td>5,833</td><td>24</td></tr><tr><td>Traffic</td><td>963</td><td>(0,1)</td><td>HOUR</td><td>4,001</td><td>24</td></tr><tr><td>Taxi</td><td>1,214</td><td>N</td><td>30-MIN</td><td>1,488</td><td>24</td></tr><tr><td>Wikipedia</td><td>2,000</td><td>N</td><td>DAY</td><td>792</td><td>30</td></tr></table>

$$
q ( \mathbf { x } _ { t _ { 0 } : T } \mid \mathbf { x } _ { 1 : t _ { 0 } - 1 } , \mathbf { c } _ { 1 : T } ) : = \prod _ { t = t _ { 0 } } ^ { T } q ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 1 : t _ { 0 } - 1 } , \mathbf { c } _ { 1 : T } ) .
$$

The bene btaine fo covariates is higl dependent nhe haracteristi both thatase and thee used, as well as the feature engineering practices followed.

Metric The Continuous Ranked Probability Score (CRPS) [46] is a scoring function that measures how well the forecast distribution matches the ground truth distribution:

$$
\mathrm { C R P S } ( F , x ) = \int _ { \mathbb { R } } \left( F ( z ) - \mathbb { I } \left\{ x \leq z \right\} \right) ^ { 2 } \mathrm { d } z ,
$$

where $F ( z )$ is the univariate cumulative distribution function (CDF) over the predicted value, $x$ is a ground truth observation, and $\mathbb { I } \left\{ x \leq z \right\}$ is the indicator function that is one if $x \leq z$ and zero otherwise. By summing the $D$ -dimensional time series along the feature dimension for simulated samples (resulting in $\hat { F } _ { \mathrm { s u m } } ( t ) )$ and ground truth data (as $\textstyle \sum _ { i } x _ { i , t } ^ { 0 } )$ , we can report the $\mathrm { C R P S _ { s u m } }$

$$
\mathrm { C R P S } _ { \mathrm { s u m } } = \mathbb { E } _ { t \sim \mathcal { U } ( t _ { 0 } , T ) } \left[ \mathrm { C R P S } \left( \hat { F } _ { \mathrm { s u m } } ( t ) , \sum _ { i } x _ { i , t } ^ { 0 } \right) \right]
$$

as the average over the prediction window. The lower the $\mathrm { C R P S _ { s u m } }$ value, the better the predicted distribution match the data distribution.

First, we manually sum the time series along the feature dimension and estimate the CDF $\hat { F } _ { \mathrm { s u m } } ( t )$ via 19 quantile levels at each time step $t$ from 100 sampled trajectories. We then use the implementation in GluonTs [2] to compute the CRPS, which we report as $\mathrm { C R P S _ { s u m } }$ in Table 2. While we aggregate the data manually, we verify that the numerical ero elativ he GluonT plementatin emais rde magnitude belowhe prsn threshold of the reported metric.