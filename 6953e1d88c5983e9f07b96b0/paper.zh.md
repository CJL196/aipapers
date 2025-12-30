# 梦想回忆：基于想象引导的经验检索用于记忆持久的视觉-语言导航

Yunzhe Xu, Yiyuan Pan, Zhe Liu A—在所有场景中都取得了显著改进，在IR2R上相比最佳的记忆持久性基线提升了$5.4\%$的SPL，伴随有$8.3\times$的训练速度提升和$74\%$的推理内存减少。结果验证了对环境和行为记忆的预测检索能够实现更有效的导航，分析表明该基于想象的范式有显著的提升空间（$73.3\%$ vs $93.4\%$的上限）。代码可在 https://github.com/xyz9911/Memoir 获取。关键字—视觉与语言导航、具身智能、记忆机制、世界模型。

# 1 引言 这是一项在具身人工智能领域中至关重要的挑战，要求智能体能够解读自然语言指令并在环境中导航以达到指定目标。传统的视觉语言导航任务的基本情节特性，智能体在不同情节中独立操作而不保留经验知识，限制了它们在持续改进和环境适应方面的能力，妨碍了在持续运行至关重要的现实世界中的应用。这一限制促使开发了记忆持久的导航任务，评估智能体在多个导航情节中积累和利用经验的能力。这些任务更准确地反映了实际应用场景，其中机器人智能体必须通过对环境的熟悉和学习的行为模式不断提升其导航能力。

近期在持久内存的视觉导航（VLN）领域的进展主要聚焦于渐进性的场景知识积累的长期记忆机制。早期的方法使用了诸如情节历史堆叠的策略，但简单的历史扩展会因为冗余导致性能下降。后续的研究通过增强视觉表征，拓宽空间视野，而不是加入导航历史来解决这个问题，OVER-NAV利用开放词汇检测构建多模态拓扑图，以增强关键词与观测的对应关系。最近，GR-DUET通过保留拓扑观测记忆，增强了DUET架构，在VLN场景适应中取得了良好的表现。尽管有这些进展，现有方法仍然存在两个关键限制。首先，当前方法缺乏有效的记忆访问机制，而是依赖于完全的记忆整合（导致无关信息的融入和计算开销）或固定视野的空间查找（可能导致宝贵经验的丧失）。其次，导航行为历史包含了有关智能体如何解读指令和在不同场景中选择行动的宝贵决策模式。然而，现有的持久内存VLN方法要么完全忽视这些历史，要么在尝试利用这些信息时未能有效发挥作用。

如何让智能体有效确定应访问哪些记忆以利用导航经验？人类导航者自然会进行导航路线的心理想象 [10] 和未来旅行事件的想象 [11]，在决策时参考经验，基于心理模拟来最终确定决策 [12]，这突显了想象作为查询机制的作用——智能体可以预测他们可能的导航方向，并检索与这些预测状态匹配的相关过往经历。这种范式不同于传统的想象规划方法 [13]，后者是在孤立的情况下生成轨迹；相反，想象是通过查询明确的长期记忆来实现的，确保检索到的经历能够直接影响决策，同时避免幻觉。为此，我们提出了基于模型的混合视角级记忆经验检索（Memoir），这是一个利用预测性世界建模进行视角粒度记忆检索的智能体。我们的方法通过统一框架解决了上述局限性。首先，采用想象的未来状态作为查询来实现自适应检索，避免了完全记忆整合和固定时间范围查找。其次，通过将导航历史编码为捕捉决策策略的潜在状态并进行视角级锚定，从而实现行为模式的保留。图1展示了Memoir的工作流程。

![](images/1.jpg)  
. approach adaptively retrieves both observation and histories for navigation planning through imagination.

实现这种基于想象的范式需要解决三个关键挑战：如何生成预测查询、存储和检索什么、以及如何将检索到的知识整合用于导航。Memoir通过统一框架来应对这些挑战。1）一种语言条件下的世界模型学习根据指令想象未来的导航状态。这些想象的状态具有双重用途：将当前的导航经验编码为潜在表示以供存储，并生成查询以通过兼容性匹配检索相似的过去经验。2）混合视角级别的记忆（HVM）通过将环境观察和行为模式锚定到视角来维护这些积累的知识，使得不仅能够检索智能体所看到的内容，还能检索他们的导航方式。3）导航模型随后通过专门的编码器处理当前观察与检索到的经验，以做出明智的决策。这实现了适应性记忆访问，在各个阶段保留战略知识。我们在多种视觉导航（VLN）方法上实现了Memoir，验证了其在具有10种独特测试场景的记忆持久基准上的有效性。Memoir展现出持续的改进，在IR2R上实现了$5 . 4 \%$的SPL提升，同时伴随$8 . 3 \times$的训练加速和$7 4 \%$的推理内存减少。我们还揭示了这一范式存在 substantial headroom $( 7 3 . 3 \%$ vs $9 3 . 4 \%$ upper bound)，并指出未来的研究方向。我们的关键贡献包括：我们提出了一种新颖的范式，其中想象作为一种检索机制，建立在明确的记忆基础上，解决了记忆持久的VLN中的两个基本局限：通过完全整合或固定视野查找实现的无效记忆访问，以及在关注环境观察的同时忽略编码决策模式的行为历史。这使得能够根据导航意图对环境和行为经验进行自适应过滤。 我们开发了Memoir，一个统一框架，其中语言条件的世界模型用于存储导航历史并生成想象轨迹作为检索查询，混合视角级别的记忆（HVM）在视角粒度上维护观察和行为模式，而增强经验的导航模型整合检索到的信息以实现稳健的决策。在10种多样的记忆持久场景中进行了广泛评估，我们展示了一致的改进及显著的效率收益，同时，通过预言者分析揭示了 substantial headroom，指明了推进这一范式的前景积极方向。

# 2 相关工作

# 2.1 视觉与语言导航

视觉与语言导航（VLN）需要智能体在朝向目标目的地导航的同时遵循自然语言指令。单一回合的 VLN 研究通过从演讲模型到合成数据的数据增强方法不断发展，使用大型语言模型（LLMs）和从历史表示到结构化空间系统的记忆架构，特别是拓扑观察记忆，这已被广泛采用。然而，这些单一回合的方法无法在不同回合之间积累知识。持久性记忆 VLN 基准解决了智能体应连续操作并通过积累的经验提升的现实需求。TourHAMT 通过堆叠完整的导航序列扩展历史记忆，但因过度冗余而导致性能下降。ESceme 增强了环境观察，提供更广泛的空间上下文，而 OVER-NAV 构建了具有固定距离检索的全图。MAPCMA 构建了全球语义图，并增强了固定视野的自我中心感知。GR-DUET 保留了完整的拓扑记忆，实现了在计算成本上获得的性能提升。这些方法都有一些基本局限性：依赖于完整的记忆整合或固定视野检索，且专注于环境观察，而忽略了记录跨上下文决策策略的导航行为模式。我们的工作通过想象引导的记忆检索来解决这些局限性，选择性地访问环境和行为历史。

# 2.2 记忆机制

导航系统中的记忆机制主要包括两种类型，它们在空间推理中发挥互补作用。导航历史记忆通过序列表示[19]、[22]或自然语言表述[23]、[24]捕捉时间决策模式和行为上下文，保存智能体在不同场景下的决策过程。环境观察记忆通过结构化的表示方式，如占用图[25]、[26]、语义图[27]、[28]、鸟瞰图[29]、[30]和拓扑记忆[31]、[32]、[33]，保存空间信息，维护场景理解所需的空间布局和视觉特征。在记忆持久的场景中，智能体必须在多样化的经验中积累知识，目前的方法[5]、[6]将这些信息源分开处理。单独的环境观察无法编码导航决策背后的行为推理，而没有空间锚定的导航历史无法在不同环境中消歧相似模式。这种分离限制了知识在导航场景中的转移，推动我们提出统一的记忆机制，利用空间和时间历史信息。

# 2.3 世界模型

预测世界模型通过潜在动态建模的POMDP解决方案在基于模型的强化学习中表现出显著的影响。递归状态空间模型（RSSM）是主要架构，并扩展至语言条件和大规模预训练。对比世界模型提供了不需要观察重构的计算效率。在导航领域，世界模型用于多种目的：未来观察合成以进行数据增强，想象中的轨迹规划，以及辅助任务的制定。然而，将世界模型与记忆结合以实现扎实的想象仍然未被充分探索。MBEC在情节控制中开创了以想象引导的记忆检索，但其局限于训练期间的情节历史，而非推理期间的时空检索。我们的方法通过将世界建模与经验检索统一，扩展了这一范式，使得世界模型不仅编码历史以供存储，还生成虚拟状态以进行时空检索。

# 3 前言

# 3.1 VLN 公式化

视觉与语言导航（VLN）要求智能体遵循指令并朝向目标导航。环境被表示为一个连通图 $\mathcal{G} = (\nu, \mathcal{E})$ ，其中 $\nu$ 表示可导航的视点，而 $\mathcal{E}$ 表示连接相邻视点的可 traversable 边。

单回合视觉语言导航（VLN）表述。在传统的回合设置中，智能体接收自然语言指令 $\ell$，并在起始视点 $v _ { 1 } \in \mathcal { V }$ 处初始化。在每个时间步 $t$，智能体观察到一个全景观察 $o _ { t } = \{ o _ { t } ^ { ( i ) } \} _ { i = 1 } ^ { 3 6 }$，该观察提供从三个不同高度（向上、水平、向下）捕捉的水平视角。智能体在视点 $v _ { t }$ 的动作空间包括导航到任何相邻视点 $v _ { j } \in \mathcal { N } ( v _ { t } )$ 以及一个终止停止动作，其中 $\mathcal { N } ( v _ { t } ) = \{ v _ { j } \in \mathcal { V } : ( v _ { t } , v _ { j } ) \in \mathcal { E } \}$ 表示相邻视点的集合。当智能体执行停止动作或达到最大步数限制 $T _ { \mathrm { m a x } }$ 时，回合结束。

记忆持久化的 VLN 公式。传统的 VLN 在有效评估基础的指令遵循能力方面表现良好，但未能捕捉到在持久操作中逐步改进的需求。记忆持久化地址 $\mathcal { M } = \{ ( \ell ^ { ( k ) } , \mathcal { G } ^ { ( k ) } , \mathcal { O } ^ { ( k ) } , \mathcal { A } ^ { ( k ) } ) \} _ { k = 1 } ^ { \vee }$ 累积了多个回合中的经验知识，其中对于第 $k$ 次回合，$\ell ^ { ( k ) }$ 是指令，$\begin{array} { r c l } { \dot { \mathcal { G } } ^ { ( k ) } } & { = } & { ( \mathcal { V } ^ { ( k ) } , \mathcal { E } ^ { ( k ) } ) } \end{array}$ 是 $k$ 次回合后的观察子图，O(k) = {o(k)rte，$\mathcal { A } ^ { ( k ) } = \{ a _ { t } ^ { ( k ) } \} _ { t = 1 } ^ { \bullet }$ 分别记录观察和动作。银行 $\mathcal { M }$ 在每次回合中不断更新，作为决策的持久存储库，通过累积的环境熟悉度和学习到的行为模式，促进逐步的性能提升。

# 3.2 双尺度图变换器 (DUET)

DUET [9] 通过拓扑映射和全局行动规划实现拓扑导航。

拓扑映射。智能体维护一个逐步构建的拓扑表示 $\mathcal { G } _ { t } ~ = ~ ( \nu _ { t } , \mathcal { E } _ { t } )$，用于表示已探索环境的状态，其中 $\mathcal { G } _ { t } ~ \subseteq ~ \mathcal { G }$ 表示在 $t$ 次导航步骤后的观察子集。视点集合 $\nu _ { t }$ 被划分为三类：已访问视点、前沿视点（可观察但未访问的邻居）和当前视点。在每个时间步 $t$，拓扑图通过将当前视点 $v _ { t }$ 和其可导航邻居 $\check { \mathcal { N } } ( v _ { t } )$ 纳入表示中进行更新，同时相应的边更新 $r _ { t } ~ = ~ \{ r _ { t } ^ { ( i ) } \} _ { i = 1 } ^ { 3 6 }$ 通过应用于视觉观察编码器进行计算。当前视点的视觉表示 $x _ { t }$ 是通过对 $\boldsymbol { r } _ { t }$ 进行平均池化获得的，而每个未访问的邻接视点 $\bar { v } _ { j } \in \bar { N } ( v _ { t } )$ 则通过其对应的方向嵌入 $r _ { t } ^ { ( i _ { j } ) }$ 表示，其中 $i _ { j}$ 表示朝向 $v _ { j }$ 的视角索引。对于从多个位置观察到的视点，嵌入会进行平均以保持一致性。

![](images/2.jpg)  
F respectively to determine the final action.

全球行动规划。DUET结合了在拓扑图上的粗尺度规划与对邻近环境的细尺度规划。指令$\ell$经过变换器处理，以获得文本表示$\hat { \ell }$。对于粗尺度规划，视点$\boldsymbol { v } _ { j } ~ \in ~ \mathcal { V } _ { t }$的节点表示$x _ { j }$与一个特殊的停止标记$x _ { 0 }$进行了增强。粗尺度编码器通过跨模态注意力和图感知自注意力（GASA）处理指令嵌入$\hat { \ell }$和视点表示$\begin{array} { r l } { X } & { { } = } \end{array}$ $[ x _ { 0 } , x _ { 1 } , \ldots , x _ { | \mathcal { V } _ { t } | } ]$：状态。给定观测序列$( o _ { 1 } , o _ { 2 } , \ldots , o _ { T } ) _ { \scriptscriptstyle { \cdot \cdot \cdot } }$，世界模型在捕捉环境动态的潜在状态$z _ { t }$上运行。联合分布因式分解为：

$$
p ( o , z ) = \prod _ { t = 1 } ^ { T } p ( z _ { t } \mid z _ { t - 1 } ) p ( o _ { t } \mid z _ { t } ) .
$$

为了最大化观察似然 $p { \big ( } o _ { 1 : T } )$，模型引入了变分后验 $q { \big ( } z _ { 1 : T } | o _ { 1 : T } { \big ) }$ 并推导出证据下界（ELBO）[49]：

$$
\begin{array} { r l } & { \ln p ( o ) \geq \displaystyle \sum _ { t = 1 } ^ { T } \Big ( \mathbb { E } _ { q ( z _ { t } \mid o _ { \leq t } ) } \big [ \underbrace { \ln p ( o _ { t } \mid z _ { t } ) } _ { \mathcal { I } _ { \mathrm { R E C O V E R } } } \big ] } \\ & { \qquad - \mathbb { E } _ { q ( z _ { t - 1 } \mid o _ { \leq t } ) } \big [ \underline { { \mathrm { K L } \big [ q ( z _ { t } \mid o _ { \leq t } ) \big \| \ p ( z _ { t } \mid z _ { t - 1 } ) \big ] } } \big ] \Big ) . } \end{array}
$$

为了增强模型的区分能力，同时避免像素级重构，术语 $\mathcal { I } _ { \mathrm { R E C O V E R } }$ 被替换为对比目标 [35], [39]。根据信息论的推导，我们可以使用噪声对比估计 (NCE) [50] 来对 TRECovER 进行下界估计：

$$
\mathrm { G A S A } ( X ) = \mathrm { S o f t m a x } \left( \frac { X W _ { q } ( X W _ { k } ) ^ { T } } { \sqrt { d } } + M \right) X W _ { v } ,
$$

距离编码矩阵 $M = E W _ { e } + b _ { e }$ 包含成对距离矩阵 $E$。对于精细尺度规划，精细尺度编码器处理指令 $\hat { \ell }$ 和全景特征 $r _ { t }$，为即时邻居 $\mathcal { N } ( v _ { t } )$ 生成动作评分。最终的导航决策通过学习的动态加权结合两个尺度，为每个候选视点生成动作评分。

# 3.3 对比变分世界模型

世界模型提供了环境动态的潜在表征，使得对未来的有效推理成为可能，其中 $\mathcal { D }$ 代表一批负样本。这个对比目标训练模型区分正确的状态-观测对 $\left( \boldsymbol { z } _ { t } , \boldsymbol { o } _ { t } \right)$ 和不正确的对 $( z _ { t } , o ^ { \prime } )$ ，有效地学习捕捉环境细节的表征，而无需显式重建。

$$
\begin{array} { r l } & { \mathcal { T } _ { \mathrm { R E C O V E R } } \geq \mathbb { E } _ { q ( \boldsymbol { z } _ { t } | \cdot ) } \Bigg [ \ln p ( \boldsymbol { z } _ { t } \mid \boldsymbol { o } _ { t } ) - \ln \sum _ { \boldsymbol { o } ^ { \prime } \in \mathcal { D } } p ( \boldsymbol { z } _ { t } \mid \boldsymbol { o } ^ { \prime } ) \Bigg ] } \\ & { \qquad = \mathcal { I } _ { \mathrm { N C E } } , } \end{array}
$$

# 4 回忆录

本节介绍了Memoir，一种基于持久记忆的VLN智能体，利用世界模型想象进行自适应。 TABLE 1 关键符号汇总。

算法 1：环境观察检索 2 对于 $i \gets 1$ 到 $| \tau _ { t } |$ do 3 初始化 $\mathcal { R } _ { \mathrm { t m p } } \emptyset$ 4 从 $\mathcal { G } ^ { ( k ) }$ 获取第 $i$ 阶邻居 $\mathcal { N } _ { i } ( v _ { t } )$ 5 对于每个视点 $v _ { n } \in \mathcal { N } _ { i } ( v _ { t } )$ do 6 从 $\tau _ { t }$ 中提取想象状态 $\hat { z } _ { t + i }$ 7 通过方程 (10) 计算相容性 $c _ { i , n }$ 8 将 $( v _ { n } , c _ { i , n } )$ 添加到 ${ \mathcal { R } } _ { \mathrm { t m p } }$ 9 按得分 $c _ { i , n }$ 降序排序 ${ \mathcal { R } } _ { \mathrm { t m p } }$ 10 保留 ${ \mathcal { R } } _ { \mathrm { t m p } }$ 的前 $( 1 - \rho _ { o } \cdot \gamma _ { o } ^ { i - 1 } )$ 部分 11 保留 ${ \mathcal { R } } _ { \mathrm { t m p } }$ 中的前 $W$ 个节点 12 对于每个 $( v _ { n } , c _ { i , n } ) \in \mathcal { R } _ { t m p }$ do 13 在 $\mathcal { G } ^ { ( k ) }$ 中找到从 $v _ { t }$ 到 $v _ { n }$ 的最短路径 $P _ { t , n }$ 14 添加路径视点：$\mathcal { R } \mathcal { R } \cup P _ { t , n }$

<table><tr><td>Notation</td><td>Description</td></tr><tr><td>Gt = (Vt, εt)</td><td>Episodic graph at step t (viewpoints, edges)</td></tr><tr><td>G(k) = (V(k), ε(k))</td><td>Persistent graph accumulated over k episodes</td></tr><tr><td>l,l</td><td>Natural language instruction &amp; embedding</td></tr><tr><td>}i=1, rt = {r(i) ,(i)}36 }i=1</td><td>Panoramic observation &amp; features (36 views)</td></tr><tr><td>xt = AvgPool(rt)</td><td>Viewpoint feature via average pooling</td></tr><tr><td>γt, </td><td>Reward signal (distance to goal), stop threshold</td></tr><tr><td>zt, 2t</td><td>Inferred state &amp; imagined state</td></tr><tr><td>ψs, ψo</td><td>State &amp; observation embedding functions</td></tr><tr><td>τt = {zt+i}i=1 lHt</td><td>Imagined trajectory with horizon Ht</td></tr><tr><td>D</td><td>Overshooting dist. and max imagination horizon</td></tr><tr><td>Mo = (Vo, χ)</td><td>Observation bank (viewpoints, features)</td></tr><tr><td>Mh = (Vh, Zh, Th)</td><td>History bank (viewpoints, states, trajectories)</td></tr><tr><td>ci,j , </td><td>Compatibility scores for retrieval</td></tr><tr><td>W , P</td><td>Max width for obs. &amp; max patterns for history</td></tr><tr><td>ρo, γo</td><td>Filter rate &amp; decay factor for obs. retrieval</td></tr><tr><td>θh, Yh</td><td>Base threshold &amp; decay factor for history retrieval</td></tr><tr><td>σc, σf , σh</td><td>Fusion weights for navigation model encoders</td></tr><tr><td>e(c) (f) , sj (h) sj</td><td>Action scores (coarse, fine, history)</td></tr></table>

为了将基本的对比世界模型（仅专注于环境动态）[35] 适配于VLN任务，我们的方法明确地引入了指令条件，以利用VLN任务中固有的强先验知识。我们通过引入指令 $\ell$ 和奖励信号 $\gamma _ { t }$（表示到达目标的距离）扩展了标准的有效变分下界（ELBO）公式：

$$
\begin{array} { l } { { \displaystyle \ln p ( o , \gamma \mid \ell ) \geq \sum _ { t = 1 } ^ { T } \Big ( \mathbb { E } _ { q ( z _ { t } \mid o _ { \le t } , \ell ) } \Big [ \underline { { \ln p ( \gamma _ { t } \mid z _ { t } ) } } \Big ] } \ ~ } \\ { { \displaystyle ~ + \mathbb { E } _ { q ( z _ { t } \mid o _ { \le t } , \ell ) } \Big [ \underline { { \ln p ( z _ { t } \mid o _ { t } ) - \ln \sum _ { o ^ { \prime } \in \mathcal { D } } p ( z _ { t } \mid o ^ { \prime } ) } } \Big ] } \ ~ } \\ { { \displaystyle ~ - \mathbb { E } _ { q ( z _ { t - 1 } \mid o _ { \le t } , \ell ) } \Big [ \underline { { \mathrm { K L } \big [ q ( z _ { t } \mid o _ { \le t } , \ell ) \big ] \mid p \big ( z _ { t } \mid z _ { t - 1 } \big ) } } \Big ] \Big ] } \Big ) , }  \end{array}
$$

其中TrewARD鼓励对目标接近度的准确预测，以终止想象。负样本集$\mathcal{D}$由每个训练批次内来自不同时间步和情境的观察组成。对比项$\mathcal{T}_{\mathrm{NCE}}$通过一个可学习的兼容性函数实现，该函数测量潜在状态与视觉观察之间的语义相似性：每个视角$v_{n} \in \mathcal{R}$有15个。

16 从 $\mathcal { M } _ { o }$ 中检索特征 $x _ { n }$ ，对应于视角 $v _ { n }$ 17 从 $\mathcal { G } ^ { ( k ) }$ 中检索边 $E _ { n }$ ，对应于 $v _ { n }$ 18 更新情景图：$\mathcal { G } _ { t }$ .update $( v _ { n } , E _ { n } )$ 19 存储特征 $x _ { n }$ 以便于视角 $v _ { n }$ 的经验检索。如图2所示，我们的方法包含三个部分：一个基于语言条件的对比世界模型，用于编码历史并将未来状态想象为检索查询，一个混合视角级别记忆（HVM），用于存储环境观测和导航历史以便检索，以及一个集成检索知识的经验增强导航模型，用于导航规划。为便于阅读，我们在表1中列出了关键符号的定义。

# 4.1 语言条件的世界模型

1 初始化检索集合 $\mathcal { R } \emptyset$ 20 返回更新后的情节图 $\mathcal { G } _ { t }$，其中 $x _ { t }$ 表示通过 DUET 的观察编码器提取的视觉特征，采用平均池化，$\psi _ { s }$ 和 $\psi _ { o }$ 是将状态和观察映射到共享嵌入空间的学习嵌入函数，$\begin{array} { r } { \dot { \sin ( a , b ) } = \frac { a ^ { \top } b } { \| a \| \| b \| } } \end{array}$ 表示余弦相似度，$\zeta$ 表示温度参数。该公式能够对想象的状态与存储在长期记忆中的观察之间的兼容性进行原则性的评估，为基于相似性的记忆检索提供了基础。

$$
\begin{array} { l } { { f ( z _ { t } , o _ { t } ) = \displaystyle \frac { 1 } { \zeta } \sin ( \psi _ { s } ( z _ { t } ) , \psi _ { o } ( x _ { t } ) ) } } \\ { { p ( z _ { t } \mid o _ { t } ) \propto \exp ( f ( z _ { t } , o _ { t } ) ) , } } \end{array}
$$

为了提高模型的长时间预测能力和增强记忆检索质量，我们扩展了包含多步超出目标的ELBO公式。$d$步超出目标鼓励在较长时间范围内进行准确的预测：

$$
\begin{array} { r } { \mathcal { I } ^ { ( d ) } = \displaystyle \sum _ { t = 1 } ^ { T } \Big ( \mathbb { E } _ { p ( z _ { t } \mid z _ { t - d + 1 } ) q ( z _ { t - d + 1 } \mid \cdot ) } \big [ \underbrace { \ln p \big ( \gamma _ { t } \mid z _ { t } \big ) } _ { \mathcal { I } \mathrm { S T o p } } \big ] + } \\ { \mathbb { E } _ { p ( z _ { t } \mid z _ { t - d + 1 } ) q ( z _ { t - d + 1 } \mid \cdot ) } \big [ \underbrace { \ln p \big ( z _ { t } \mid o _ { t } \big ) - \ln \sum _ { o ^ { \prime } } p \big ( z _ { t } \mid o ^ { \prime } \big ) } _ { \mathcal { I } _ { \mathrm { N C E } } } - } \\ { \mathbb { E } _ { p ( z _ { t - 1 } \mid z _ { t - d } ) q ( z _ { t - d } \mid \cdot ) } \big [ \mathrm { K L } \big [ q ( z _ { t } \mid o _ { \le t } , \ell ) \mid \mid p \big ( z _ { t } \mid z _ { t - 1 } \big ) \big ] \big ] \Big ) . } \end{array}
$$

在最大超越距离 $D$ 的情况下，最终的优化目标变为：

$$
\mathcal { T } = \mathcal { T } ^ { ( 1 ) } + \frac { 1 } { D - 1 } \sum _ { d = 2 } ^ { D } \mathcal { T } ^ { ( d ) } .
$$

为了高效地优化方程（8）中的目标，我们采用了递归状态空间模型（RSSM）架构 [34]，其由四个组成部分构成：推理模型：$\begin{array} { r l } & { z _ { t } \sim q ( z _ { t } \mid z _ { t - 1 } , o _ { t } , \ell ) } \\ & { \hat { z } _ { t } \sim p ( z _ { t } \mid z _ { t - 1 } ) } \\ & { p ( z _ { t } \mid o _ { t } ) \propto \exp ( f ( z _ { t } , o _ { t } ) ) } \\ & { \hat { \gamma } _ { t } \sim p ( \gamma _ { t } \mid z _ { t } ) , } \end{array}$ 状态转移模型： 兼容性模型： 奖励模型：在实践中，推理模型将 $x _ { t }$ 作为观察的输入，$\hat { \ell }$ 作为指令的输入。推理模型将导航历史编码为可存储的表征，而状态转移模型生成想象中的未来状态，以促进基于相似性的信息检索。

# 4.2 混合视点级内存 (HVM)

在阐明我们的世界模型如何想象和推断状态之后，我们现在描述想象的状态如何查询长期记忆。我们引入了一种双库存储架构，它以视点粒度维护环境观察和导航行为历史。HVM由两个互补的存储库组成，围绕在 $k$ 轮次中累积的持久图 $\mathcal { G } ^ { ( k ) } = ( \mathcal { V } ^ { ( k ) } , \mathcal { \overline { { E } } } ^ { ( k ) } )$ 组织：

观察库：$\mathcal { M } _ { o } = ( \mathcal { V } _ { o } , \mathcal { X } _ { o } )$，其中 $\mathcal { V } _ { o } = \{ v _ { j } \}$ 表示状态，$\mathcal { X } _ { o } = \{ x _ { j } \} _ { j = 1 } ^ { | \nu _ { o } | }$。历史库：$M _ { h } = ( \mathcal { V } _ { h } , \mathcal { Z } _ { h } , \mathcal { T } _ { h } )$，其中 $\mathcal { V } _ { h } = \{ v _ { j } \}$ 表示具有记录导航历史的视角，$\mathcal { Z } _ { h } = \{ \{ z _ { j } ^ { ( k ) } \} _ { k = 1 } ^ { N _ { j } } \} _ { j = 1 } ^ { | \mathcal { V } _ { h } | }$ 存储从过去情节推断的智能体状态，$\mathcal { T } _ { h } = \{ \{ \tau _ { j } ^ { ( k ) } \} _ { k = 1 } ^ { N _ { j } } \} _ { j = 1 } ^ { | \mathcal { V } _ { h } | }$ 包含对应的想象轨迹序列，其中 $N _ { j }$ 表示对视角 $v _ { j }$ 的历史访问次数。在每个时间步 $t$，两个记忆库根据 $\boldsymbol { v } _ { t }$ 进行更新：$\boldsymbol { \mathcal { M } } _ { o }$ 接收从观察 $o _ { t }$ 中提取的视角特征 $x _ { t }$，而 $\mathcal { M } _ { h }$ 存储从推理模型中获得的推断状态 $z _ { t }$ 和想象的轨迹 $\dot { \tau _ { t } } = \{ \hat { z } _ { t + i } \} _ { i = 1 } ^ { H _ { t } }$。$H _ { t }$ 代表想象范围，当预测距离 $\hat { \gamma } _ { t + i }$ 低于阈值 $\epsilon$ 或达到最大范围 $D$ 时终止。环境观察检索。给定在视角 $v _ { t }$ 的想象轨迹 $\tau _ { t } = \{ \hat { z } _ { t + i } \} _ { i = 1 } ^ { H _ { t } }$，我们通过状态-观察兼容性进行拓扑引导搜索来检索观察。对于想象状态 $\hat { z } _ { t + i }$ 和来自 $\mathcal { M } _ { o }$ 的存储特征 $x _ { j }$，我们计算兼容性得分：

$$
c _ { i , j } = \frac { 1 } { 2 } ( \sin ( \psi _ { s } ( \hat { z } _ { t + i } ) , \psi _ { o } ( x _ { j } ) ) + 1 ) .
$$

该评分机制直接利用了公式 (6) 中的对比目标，确保训练与检索之间的一致性。对于每个想象步骤 $i$ 及其对应的邻域顺序，算法识别出所有在第 $i$ 阶邻域 $\mathcal { N } _ { i } ( v _ { t } ) = \{ v \in \mathcal { V } ^ { ( k ) } : d ( v _ { t } , v ) = i \}$ 中的视点，并使用公式 (10) 计算兼容性评分。基于百分位的过滤保留按评分排名的前 $( 1 - \rho _ { o } \cdot \gamma _ { o } ^ { i - 1 } )$ 部分视点，接着从保留的集合中选择前 $W$ 个视点，用于从 20 次返回的更新情节图 $\mathcal { G } _ { t }$ 中获取。最后，从 $v _ { t }$ 到所有选定视点的最短路径被添加到情节图 $\mathcal { G } _ { t }$ 中。完整过程在算法 1 中详细描述。

<table><tr><td colspan="3">Algorithm 2: Navigation History Retrieval</td></tr><tr><td colspan="3">Input: P Max Patterns Current Viewpoint θh Threshold Tt Imagined States γh Decay Factor Mh History Bank G(k) Persistent Graph Gt Episodic Graph</td></tr><tr><td></td><td>2 for each (z′, τ′)  Q do</td><td>1 Retrieve all patterns Q from Mh for viewpoint vt</td></tr><tr><td>3 4</td><td>Initialize L ← min(|τt|, |Tτ′|), scores C ← for i ← 1 to L do</td><td></td></tr><tr><td>5</td><td>Get imagined state zt+i, z from τt, τ′</td><td></td></tr><tr><td>6 7</td><td>if ci &lt; θh · γ i− then</td><td>Compute compatibility ci via Equation (11)</td></tr><tr><td>8 9</td><td>break Append score: C ← C U {ci}</td><td></td></tr><tr><td>10</td><td>Store pattern with score: (z&#x27;, τ′, C)  Q</td><td></td></tr><tr><td>11</td><td></td><td></td></tr><tr><td></td><td>Sort Q in descending order (by length and score)</td><td></td></tr><tr><td>12</td><td>Retain top P patterns as Q</td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td>13 for each (z&#x27;, T′, C)  Q do</td><td></td></tr><tr><td>14</td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>15</td><td></td><td></td></tr><tr><td></td><td>for i ← 1 to |C| do</td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>16</td><td></td><td>Retrieve feature xi from Mo for viewpoint vi</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>17</td><td></td><td>Retrieve edges Ei from G(k) for vi</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>18</td><td></td><td>Update episodic graph: Gt.update(vi, Ei)</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>19</td><td></td><td>Store state z, score ci and xi for viewpoint vi</td></tr></table>

导航历史检索。历史检索识别存储的历史导航模式，这些模式展现出与当前智能体的想象轨迹相似的特征。该过程利用了一个洞见：具有相似未来期望的智能体可能采用可比的策略，并应当从彼此的经验中受益。对于历史库 $\mathcal { M } _ { h }$ 中在视角 $v _ { t }$ 处存储的轨迹 $\tau ^ { \prime } = \{ \hat { z } _ { i } ^ { \prime } \} _ { i = 1 } ^ { H ^ { \prime } }$，我们基于想象轨迹 $\boldsymbol { \tau } _ { t } ^ { \star } = \{ \hat { z } _ { t + i } \} _ { i = 1 } ^ { H _ { t } }$ 进行顺序相似性匹配。第 $i$ 步的想象状态之间的兼容性计算为：

$$
c _ { i } = \frac { 1 } { 2 } ( \sin ( \psi _ { s } ( \hat { z } _ { t + i } ) , \psi _ { s } ( \hat { z } _ { i } ^ { \prime } ) ) + 1 ) .
$$

对于轨迹中的每个想象步骤 $i$，我们继续匹配，直到达到两个轨迹长度的最小值，或者遇到低于兼容性得分 ${ \bar { \theta } } _ { h } \cdot \gamma _ { h } ^ { i - 1 }$ 的情况。匹配终止之前的兼容性得分被存储为 $\mathcal { C }$。我们使用两个阶段的标准对存储的轨迹进行排名：匹配长度（优先选择较长的匹配）和匹配步骤中的最小兼容性得分。选择前 ${ \bf \nabla } \cdot { \cal P }$ 轨迹模式。对于每个选定的模式，我们检索 $\{ z _ { i } ^ { \prime } , v _ { i } \} _ { i = 1 } ^ { | { \mathcal { C } } | }$，$| { \mathcal { C } } |$ 个视角所对应的子图结构，融入到 $\mathcal { G } _ { t }$ 以及状态表示和兼容性得分中。完整的过程在算法 2 中概述。

# 4.3 导航模型

在每个时间步 $t$，智能体想象未来状态 $\tau _ { t }$，并根据 $v _ { t }$ 提取环境观察和导航历史。然后，将提取的信息集成到由拓扑映射维护的情景拓扑图 $\mathcal { G } _ { t }$ 中。现在，我们通过专门的处理编码器扩展 DUET [9]，将提取的经验知识整合到导航决策中。我们的模型通过专门的编码器处理这些提取的信息：全局观察、局部观察和导航行为模式。导航模型由三个分支组成：粗尺度编码器。粗尺度编码器通过扩展视点表征 $X$，将提取的观察纳入考量，增加了一种类型—提取的视点。包含提取观察的完整视点表征 $X = [ x _ { 0 } , x _ { 1 } , \ldots , x _ { | \mathcal { V } _ { t } | } ]$ 会通过粗尺度编码器处理以得到 $\hat { X }$。全局动作评分计算为 $s _ { j } ^ { ( c ) } = \mathrm { F F N } ( \hat { x } _ { j } )$，适用于视点 $v _ { j }$，提供高级导航偏好。细尺度编码器。细尺度编码器处理即时全景特征 $r _ { t }$ 得到 $\hat { r } _ { t }$。局部动作评分 $s _ { j } ^ { ( f ) } = \mathrm{FFN}(\hat{r}_t)$ 被计算用于每个邻居 $v _ { j } \in \mathcal { N } ( v _ { t } )$，并转换为全局动作空间：

$$
s _ { j } ^ { ( f ^ { \prime } ) } = \left\{ \begin{array} { l l } { s _ { \mathrm { b a c k } } , } & { \mathrm { i f } v _ { j } \in \mathcal { V } _ { t } \setminus \mathcal { N } ( v _ { t } ) } \\ { s _ { j } ^ { ( f ) } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.
$$

其中 $i _ { j }$ 表示面向 $v _ { j }$ 的视角索引，而 $s _ { \mathrm { b a c k } }$ 汇总了视点 $v _ { t }$ 所访问邻居的得分，以便在必要时鼓励回溯。

导航历史编码器。导航历史编码器通过将历史状态与当前视点表示融合，处理检索到的行为模式。节点集 $\nu _ { h }$ 包含该分支处理的所有视点，包括当前访问的位置和从历史库检索的节点。对于每个视点 $v _ { j } \in \mathcal V _ { h }$ 及其检索到的状态 $\bar { Z } _ { j } = [ z _ { j } ^ { \prime ( 1 ) } , z _ { j } ^ { \prime ( 2 ) } , \dots , z _ { j } ^ { \prime ( N _ { j } ) } ]$ 和兼容性评分 $C _ { j } = [ c _ { j } ^ { ( 1 ) } , c _ { j } ^ { ( \bar { 2 } ) } , \ldots \bar { } , c _ { j } ^ { ( N _ { j } ) } ]$，其中 $N _ { j }$ 表示在 $v _ { j }$ 处检索到的状态数量，我们计算：

$$
u _ { j } = \left( \operatorname { s o f t m a x } \left( \frac { C _ { j } } { \zeta } \right) \right) ^ { \top } Z _ { j } + x _ { j } .
$$

对于没有检索到历史状态的访问节点，我们简单地使用观测值 $u _ { j } = x _ { j }$。融合的状态表示 $U = [ u _ { 1 } , u _ { 2 } , \dots , u _ { | \mathcal { V } _ { h } | } ]$ 通过变换器处理，以生成历史信息驱动的动作得分 $\bar { s _ { i } ^ { ( h ) } }$，然后将其映射到全局动作空间：

$$
s _ { i } ^ { ( h ^ { \prime } ) } = \left\{ \begin{array} { l l } { s _ { 0 } , } & { \mathrm { i f } v _ { i } \in \mathscr { V } _ { t } \setminus \mathscr { V } _ { h } } \\ { s _ { i } ^ { ( h ) } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

动态融合。我们实现了一种学习的动态融合机制，该机制根据当前的状态因素自动平衡三条分支的贡献。融合权重通过以下公式计算：

$$
[ \sigma _ { f } , \sigma _ { c } , \sigma _ { h } ] = { \mathrm { S o f t m a x } } ( { \mathrm { F F N } } ( [ \hat { r } _ { 0 } ; \hat { x } _ { 0 } ; \hat { u } _ { 0 } ] ) ) ,
$$

算法 3：回忆导航循环，其中 ${ \hat { r } } _ { 0 } , { \hat { x } } _ { 0 } ,$ 和 $\hat { u } _ { 0 }$ 分别表示来自精细尺度、粗略尺度和导航历史编码器的编码停止符号表示，$[ ; ]$ 表示连接。最终的导航分数整合了所有三个分支：

<table><tr><td colspan="2">Input: Tmax Max Step limit M Observation Bank</td></tr><tr><td colspan="2">D Imagination Horizon Mh History Bank Stop threshold €</td></tr><tr><td colspan="2">G(k) Persistent Graph</td></tr><tr><td colspan="2">1 Initialize episodic graph G0 ← Ø 2 Receive initial observation 01, viewpoint v1</td></tr><tr><td colspan="2"></td></tr><tr><td></td><td>3 for step t = 1 to Tmax do</td></tr><tr><td>4</td><td>Update topological graphs Gt and G(k)</td></tr><tr><td>5 6</td><td>Infer current state zt ∼ q(zt | zt−1, 0t, )</td></tr><tr><td>7</td><td>Initialize imagined trajectory Tt ← Ø for i = 1 to D do</td></tr><tr><td>8</td><td>Imagine next state zt+i ~ p(zt+i |zt+i−1)</td></tr><tr><td>9</td><td>Predict reward γt+i ∼ p(γt+i | zt+i)</td></tr><tr><td>10</td><td>Update trajectory Tt ← τt U {zt+i}</td></tr><tr><td>11</td><td>if γt+i &lt;  then</td></tr><tr><td>12</td><td>break</td></tr><tr><td>13</td><td>Gt ← ObsRetrieval(Mo, vt, τt, Gt, G(k))</td></tr><tr><td>14</td><td>// Algorithm 1 Gt ← HistoryRetrieval(Mh, vt, Tt, Gt, G(k))</td></tr><tr><td>15</td><td>// Algorithm 2 Extract viewpoint feature xt from ot</td></tr><tr><td>16</td><td>Mo.add(vt, xt)</td></tr><tr><td>17</td><td>Mh.add(vt, zt, Tt)</td></tr><tr><td>18</td><td>Compute score sj for each candidate node vj</td></tr><tr><td>19</td><td>Select action at ← argmaxj Sj</td></tr><tr><td>20</td><td>if at = stop then</td></tr><tr><td></td><td>break</td></tr><tr><td>21</td><td></td></tr><tr><td>22</td><td>Receive ot+1, vt+1 ← env.step(at)</td></tr></table>

$$
s _ { j } = \sigma _ { f } s _ { j } ^ { ( f ^ { \prime } ) } + \sigma _ { c } s _ { j } ^ { ( c ) } + \sigma _ { h } s _ { j } ^ { ( h ^ { \prime } ) } .
$$

如算法 3 所示，Memoir 通过生成想象轨迹作为查询，实现了想象引导的记忆检索，以自适应地访问持久记忆中的相关观察和行为历史。导航模型整合了检索到的经验，使得决策基于历史证据进行信息化，同时持续更新记忆库，以实现跨情节的渐进式改进。

# 5 实验

# 5.1 实验设置

数据集。我们在两个已建立的、持久记忆的视觉语言导航基准上评估了Memoir，这些基准提供了互补的评估视角。迭代房间到房间（IR2R）[5] 扩展了基础的房间到房间（R2R）数据集 [1]，通过结构化导览实现多剧集场景，共包含183个训练导览，平均长度为76.6剧集。验证分割包括已见环境（159个导览，平均6.4剧集）和未见环境（33个导览，平均71.2剧集）。通用场景适应（GSA-R2R）[6] 融入了150个Habitat-Matterport3D（HM3D）场景 [51]，每个场景包含600条路径，提供90,000个剧集，总共涵盖10个评估场景，包括住宅和非住宅环境，指令类型多样，包括基本导航指令、场景特定描述和用户个性化指令。

实现细节。我们在三个基础模型上实施Memoir：DUET [9] 和ScaleVLN [16] 代表传统的VLN模型，以及GR-DUET [6] 代表记忆持久的方法。所有模型均使用各自预训练阶段的预训练权重，没有进行特定任务的微调。为了进行严格比较，我们在相同的超参数和实验条件下重新训练所有基线模型，包括在GSA-R2R中同步的情节排序。我们的世界模型实现采用两种架构变体：GRU和Transformer。两种变体都使用文本嵌入，并与导航模型共享观测编码器。世界模型的预训练在R2R和增强轨迹 [15] 上进行5,000次迭代，批量大小为32，学习率为5e-5，随后进行模仿学习，学习率为1e-5。结果在3次独立运行中报告。 评估指标。我们采用标准的VLN指标 [52] 来评估导航性能。为了量化长期记忆检索的有效性，我们引入四个互补指标，评估观测检索和历史检索质量。这些指标包括：轨迹长度 (TL)：以米为单位的预测路径长度。导航误差 (NE)：智能体最终位置与目标之间的距离，以米为单位。成功率 (SR)：最终位置距离目标位置少于3米的百分比。路径长度惩罚成功率 (SPL)：SR通过最短路径长度与预测路径长度的比率进行归一化。归一化动态时间规整 (nDTW)：预测路径与专家路径之间的动态时间规整归一化。旅游归一化动态时间规整 (T-nDTW)：完整旅行中的整体导航一致性。观测准确率 (OA)：在整个情节中从观测库$\mathcal { M } _ { o }$中检索到的观测的精度：

$$
\mathrm { O A } = \frac { | \bigcup _ { t = 1 } ^ { T } ( \mathcal { R } _ { t } \cap \mathcal { V } _ { \mathrm { g t } , t } ^ { o } ) | } { | \bigcup _ { t = 1 } ^ { T } \mathcal { R } _ { t } | } ,
$$

其中 $\mathcal { R } _ { t }$ 表示在时间步 $t$ 从 $\mathcal { M } _ { o }$ 中检索到的视角，$\gamma _ { \mathrm { g t } , t } ^ { o }$ 表示在观察库中存在的 $D$ 步视角，$\gamma _ { o }$ 表示存储在 $\mathcal { M } _ { o }$ 中的所有视角，而 $T$ 是回合长度。观察回忆 (OR)：回合中相关环境观察的覆盖率：

$$
\mathrm { O R } = \frac { \vert \bigcup _ { t = 1 } ^ { T } ( \mathcal { R } _ { t } \cap \mathcal { V } _ { \mathrm { g t } , t } ^ { o } ) \vert } { \vert \bigcup _ { t = 1 } ^ { T } \mathcal { V } _ { \mathrm { g t } , t } ^ { o } \vert } .
$$

历史准确率（HA）：在整个情节中，从历史库 $\mathcal { M } _ { h }$ 中检索到的导航模式的精度：

$$
\mathrm { H A } = \frac { \sum _ { t = 1 } ^ { T } \sum _ { j = 1 } ^ { | Q _ { t } | } | \mathcal { V } _ { \mathrm { t r a j } , t , j } ^ { h } \cap \mathcal { V } _ { \mathrm { g t } , t , j } ^ { h } | } { \sum _ { t = 1 } ^ { T } \sum _ { j = 1 } ^ { | Q _ { t } | } | \mathcal { V } _ { \mathrm { t r a j } , t , j } ^ { h } | } ,
$$

其中 $| Q _ { t } |$ 表示在时间步 $t$ 检索到的导航历史模式的数量（详见算法 2），${ \mathcal V } _ { \mathrm { t r a j } , t , j } ^ { h } \ = \ \{ v _ { 1 } ^ { ( j ) } , v _ { 2 } ^ { ( j ) } , \ldots , v _ { | { \mathcal C } _ { j } | } ^ { \dot { ( j ) } } \}$ 表示第 $j$ 个检索到的导航历史轨迹中的视点序列，而 Vh 表示存在于原始历史轨迹中的教师轨迹上的视点。历史召回率 (HR)：在整个回合中相关导航模式的覆盖率：

$$
\mathrm { H R } = \frac { \sum _ { t = 1 } ^ { T } \sum _ { j = 1 } ^ { \lvert Q _ { t } \rvert } \lvert \mathcal { V } _ { \mathrm { t r a j } , t , j } ^ { h } \cap \mathcal { V } _ { \mathrm { g t } , t , j } ^ { h } \rvert } { \sum _ { t = 1 } ^ { T } \sum _ { j = 1 } ^ { \lvert Q _ { t } \rvert } \lvert \mathcal { V } _ { \mathrm { g t } , t , j } ^ { h } \rvert } .
$$

# 5.2 定量分析

# 5.2.1 迭代房间到房间（IR2R）

表2展示了Memoir相较于传统方法和记忆持久方法在IR2R基准测试上的比较。在应用于传统的VLN模型时，Memoir显示出显著的性能提升：基于DUET的实现相比于未见场景的SPL提升$11.1\%$，而基于ScaleVLN的实现则提升$5.6\%$。这些结果证明，从长期记忆中检索信息的整合作为稳健导航决策的有效先验，甚至对于最初并未设计为具备记忆持久性的模型也是如此。与记忆持久方法相比，Memoir显著优于GR-DUET，在未见场景下实现了$5.4\%$的SPL改善（$73.3\%$对比$67.9\%$），在已见场景中实现了$11.6\%$的提升。这一卓越表现验证了我们的假设：整合完整记忆信息会引入过多噪声，降低导航决策的质量，并限制在经验有限场景中的灵活性。我们的自适应检索方法有效解决了这些局限性。讨论。虽然在未见场景中表现卓越，记忆持久变体在已见场景中的表现往往低于其传统对应物。例如，DUET在已见环境下的SPL为$74.5\%$，而GR-DUET则为$55.1\%$，我们的方案约有$2\%$的SPL下降。这一现象源于：(1) 验证分割间巡回长度的差异，已见巡回仅平均6.4集，而未见巡回平均71.2集，限制了积累经验；(2) 正则化效应，即长期记忆整合通过鼓励更广泛的上下文推理而非环境细节记忆来防止对训练环境的过拟合。与GR-DUET相比，Memoir显著缩小了这一性能差距，表现出更为均衡的记忆利用。

# 5.2.2 一般场景适应 (GSA-R2R)

表3、4和5展示了Memoir在不同场景适应场景下的表现。表3与基于适应方法和记忆持久方法在五种不同用户指令风格下进行了比较。表4和5评估了在不同环境特征和指令表达下的表现。Memoir在八种不同的测试场景中始终优于基于适应和基于记忆的方法，相比于GR-DUET实现了平均$2.38\%$的成功率(SR)提升和$1.59\%$的路径长度(SPL)改进，实验配置一致。提升证明了混合记忆提供了传统方法所缺乏的关键上下文：通过访问过去的经历，在这些经历中智能体成功处理了类似的表达并执行了相应的动作，Memoir从历史模式中学习，而GR-DUET的观察记忆无法捕捉到这些模式。表2展示了Memoir与各种VLN方法在IR2R基准上的导航性能比较。

<table><tr><td colspan="5"></td><td colspan="5">Val Seen</td><td colspan="5">Val Unseen</td></tr><tr><td>Methods</td><td>PH TH PHI IW</td><td></td><td></td><td>TL↓ NE↓</td><td>nDTW↑</td><td></td><td>SR↑</td><td>SPL ↑ t-nDTW↑</td><td>TL↓</td><td>NE↓</td><td>nDTW↑</td><td>SR↑</td><td>SPL↑</td><td>t-nDTW↑</td></tr><tr><td>HAMT [20]</td><td></td><td></td><td></td><td>10.1 ±0.1 4.2 ±0.1</td><td>71 ±1</td><td>63 ±1</td><td>61 ±1</td><td>58 ±1</td><td>9.4 ±0.1 4.7 ±0.0</td><td></td><td>66 ±0</td><td>56 ±0</td><td>54 ±0</td><td>50 ±0</td></tr><tr><td>TourHAMT [5]</td><td>✓</td><td>✓ ✓</td><td>✓</td><td>9.4 ±0.4 5.8 ±0.1</td><td>59 ±0</td><td>45 ±1</td><td></td><td>43 ±1 45 ±0</td><td></td><td>10.0 ±0.2 6.2 ±0.1</td><td>52 ±0</td><td>39 ±1</td><td>36 ±0</td><td>32 ±1</td></tr><tr><td></td><td>✓</td><td>✓</td><td>✓</td><td>10.5 ±0.3 6.0 ±0.2</td><td></td><td>58 ±1</td><td>45 ±2</td><td>43 ±2</td><td>42 ±1</td><td>10.9 ±0.2 6.8 ±0.2</td><td>51 ±1</td><td>38 ±1</td><td>34 ±1</td><td>31 ±1</td></tr><tr><td></td><td>✓</td><td>✓</td><td></td><td>10.6 ±0.3 6.0 ±0.1</td><td></td><td>58 ±1</td><td>45 ±1</td><td>42 ±1</td><td>42 ±1</td><td>10.3 ±0.3 6.7 ±0.2</td><td>50 ±1</td><td>38 ±1</td><td>34 ±1</td><td>29 ±1</td></tr><tr><td></td><td>✓</td><td></td><td></td><td>10.9 ±0.3 6.1 ±0.1</td><td>58 ±1</td><td>45 ±1</td><td></td><td>42 ±1 41 ±0</td><td></td><td>11.0 ±0.6 6.7 ±0.1</td><td>51 ±0</td><td>38 ±0</td><td>34 ±0</td><td>28 ±1</td></tr><tr><td>OVER-NAV [7]</td><td></td><td></td><td></td><td>9.9 ±0.1 3.7 ± 0.1</td><td>73 ±1</td><td>65 ±1</td><td>63 ±1</td><td>62 ±0</td><td></td><td>9.4 ±0.1 4.1 ±0.1</td><td>69 ±0</td><td>60 ±1</td><td>57 ±0</td><td>55 ±1</td></tr><tr><td colspan="14">Comparison with Traditional VLN Models:</td></tr><tr><td>VLN models pretrained with default protocol:</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DUET [9]</td><td></td><td></td><td></td><td>12.5 ±0.4 2.2 ±0.1 79.8 ±1.1 79.8 ±0.7 74.5 ±0.9 69.1 ±1.7</td><td></td><td></td><td></td><td></td><td></td><td>14.4 ±0.1 3.5 ±0.0 65.0 ±0.1 69.2 ±0.3 58.0 ±0.1 47.0 ±0.8</td><td></td><td></td><td></td><td></td></tr><tr><td>+Memoir (Ours)</td><td></td><td></td><td></td><td>11.5 ±0.1 2.6 ±0.2 78.9 ±0.9 77.1 ±0.5 72.8 ±0.5 68.0 ±0.8 11.0 ±0.0 2.8 ±0.1 75.2 ±0.0 75.4 ±0.2 69.1 ±0.3 58.8 ±0.4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VLN models pretrained with environmental augmentation:</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ScaleVN 16]</td><td></td><td></td><td></td><td>12.8 ±0.0 2.2 ±0.0 79.6 ±0.4 79.5 ±0.5 74.1 ±0.6 67.0 ±0.2</td><td></td><td></td><td></td><td></td><td></td><td> 13.5 ±0.0 2.7 ±0.0 71.6 ±0.1 76.2 ±0.1 66.5 ±0.2 53.4 ±0.2</td><td></td><td></td><td></td><td></td></tr><tr><td>+Memoir (Ours)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">11.6 ±0.2.5 ±0.178.7 ±0.176.1 ±0.57.3 ±0.17.1 ±0.0 10.9 ±0.2 .6 ±0.077.2 ±0.677.4 ±0.2 72.1 ±0.4 6. ±. Comparison with Memory-Persistent VLN Models: VLN models pretrained with full navigation graph:</td></tr></table>

表 3 GSA-R2R 基准测试中带用户指令的导航性能比较。

<table><tr><td></td><td colspan="2">Child</td><td colspan="2">Keith</td><td colspan="2">Moira</td><td colspan="2">Rachel</td><td colspan="2">Sheldon</td></tr><tr><td>Methods</td><td>SR ↑</td><td>SPL↑</td><td>SR ↑</td><td>SPL↑</td><td>SR ↑</td><td>SPL↑</td><td>SR ↑</td><td>SPL↑</td><td>SR ↑</td><td>SPL↑</td></tr><tr><td>TourHAMT [5]</td><td>14.6 ±0.2</td><td>12.0 ±0.2</td><td>15.1 ±0.2</td><td>12.3 ±0.1</td><td>13.9 ±0.1</td><td>11.3 ±0.1</td><td>15.3 ±0.1</td><td>12.5 ±0.1</td><td>14.4 ±0.1</td><td>11.8 ±0.1</td></tr><tr><td>OVER-NAV [7]</td><td>20.9 ±0.1</td><td>16.1 ±0.2</td><td>20.5 ±0.1</td><td>16.4 ±0.1</td><td>19.5 ±0.2</td><td>15.4 ±0.2</td><td>20.6 ±0.3</td><td>16.2 ±0.2</td><td>20.5 ±0.1</td><td>16.2 ±0.1</td></tr><tr><td>DUET [9]</td><td>54.3</td><td>44.1</td><td>56.0</td><td>46.3</td><td>52.3</td><td>43.3</td><td>56.3</td><td>46.4</td><td>54.0</td><td>44.4</td></tr><tr><td>+MLM [15]</td><td>54.5 ±0.2</td><td>44.7 ±0.2</td><td>56.4 ±0.3</td><td>46.8 ±0.3</td><td>53.8 ±0.3</td><td>43.6 ±0.4</td><td>56.8 ±0.5</td><td>46.6 ±0.6</td><td>54.5 ±0.4</td><td>44.2 ±0.3</td></tr><tr><td>+MRC [15]</td><td>54.4 ±0.2</td><td>44.2 ±0.1</td><td>56.0 ±0.1</td><td>46.3 ±0.1</td><td>52.3 ±0.2</td><td>43.3 ±0.1</td><td>56.0 ±0.1</td><td>46.2 ±0.2</td><td>53.7 ±0.2</td><td>44.2 ±0.4</td></tr><tr><td>+BT [53]</td><td>57.5 ±0.7</td><td>54.0 ±0.9</td><td>61.2 ±0.3</td><td>57.9 ±0.1</td><td>57.3 ±0.5</td><td>54.0 ±0.6</td><td>61.6 ±0.8</td><td>58.1 ±0.7</td><td>57.6 ±0.5</td><td>54.3 ±0.5</td></tr><tr><td>+TENT [54]</td><td>54.3 ±0.2</td><td>41.7 ±0.1</td><td>55.4 ±0.2</td><td>43.8 ±0.2</td><td>51.7 ±0.2</td><td>41.0 ±0.1</td><td>55.0 ±0.2</td><td>43.2 ±0.2</td><td>53.0 ±0.2</td><td>41.9 ±0.1</td></tr><tr><td>+SAR [55]</td><td>54.5 ±0.5</td><td>41.5 ±0.4</td><td>54.9 ±0.3</td><td>43.1 ±0.2</td><td>51.0 ±0.4</td><td>40.3 ±0.6</td><td>55.3 ±0.5</td><td>43.0 ±0.6</td><td>52.9 ±0.2</td><td>41.4 ±0.4</td></tr><tr><td colspan="9">VLN models pretrained with full navigation graph:</td><td></td><td></td></tr><tr><td>GR-DUET [6]</td><td>65.2 ±0.1</td><td>59.7 ±0.1</td><td>66.7 ±0.1</td><td>62.0 ±0.1</td><td>60.9 ±0.2</td><td>56.2 ±0.2</td><td>67.1 ±0.1</td><td>62.2 ±0.1</td><td>63.9 ±0.1</td><td>58.9 ±0.1</td></tr><tr><td>GR-DUET* [6]</td><td>64.9 ±0.5</td><td>60.5 ±0.4</td><td>65.1 ±0.3</td><td>61.4 ±0.4</td><td>60.5 ±0.3</td><td>56.6 ±0.2</td><td>65.7 ±0.5</td><td>61.7 ±0.4</td><td>63.0 ±0.4</td><td>59.0 ±0.4</td></tr><tr><td>+Memoir (Ours)</td><td>66.5 ±0.5</td><td>61.3 ±0.5</td><td>68.0 ±0.1</td><td>63.6 ±0.2</td><td>62.5 ±0.3</td><td>57.5 ±0.4</td><td>68.2 ±0.1</td><td>63.6 ±0.3</td><td>65.3 ±0.1</td><td>60.4 ±0.3</td></tr></table>

表4 基于场景指令的GSA-R2R基准导航性能比较。

<table><tr><td rowspan="3">Methods</td><td colspan="5">Test-N-Scene</td></tr><tr><td>TL↓</td><td>NE ↓</td><td>SR ↑</td><td>SPL↑</td><td>nDTW↑</td></tr><tr><td>TourHAMT [5]</td><td>7.3 ±0.1 8.1 ±0.1</td><td>9.7 ±0.1</td><td>8.0 ±0.1</td><td>32.3 ±0.1</td></tr><tr><td>OVER-NAV [7] DUET [9]</td><td>11.8 ±0.1 14.9</td><td>7.6 ±0.2 6.4</td><td>16.7 ±0.4 39.6</td><td>12.6 ±0.2 30.1</td><td>34.6 ±0.3 40.9</td></tr><tr><td>+MLM [15]</td><td>14.3 ±0.1</td><td>6.5 ±0.1</td><td>39.8 ±0.1</td><td>30.5 ±0.1</td><td>41.1 ±0.1</td></tr><tr><td>+MRC [15]</td><td>14.9 ±0.1</td><td>6.4 ±0.1</td><td>39.7 ±0.1</td><td>30.2 ±0.1</td><td>40.9 ±0.1</td></tr><tr><td>+BT [53]</td><td>8.4 ±0.0</td><td>6.3 ±0.2</td><td>41.2 ±1.5</td><td>38.2 ±1.2</td><td>51.3 ±1.2</td></tr><tr><td>+TENT [54]</td><td>16.4 ±0.1</td><td>6.3 ±0.1</td><td>40.6 ±0.2</td><td></td><td></td></tr><tr><td>+SAR [55]</td><td></td><td></td><td></td><td>28.9 ±0.2</td><td>38.9 ±0.2</td></tr><tr><td></td><td>16.3 ±0.5</td><td>6.0 ±0.2</td><td>41.4 ±0.6</td><td>29.1 ±0.3</td><td>39.0 ±0.3</td></tr><tr><td>VLN models pretrained with full navigation graph:</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GR-DUET [6]</td><td>10.1 ±0.0</td><td>5.5 ±0.0</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>48.1 ±0.1</td><td>42.8 ±0.1</td><td>53.7 ±0.1</td></tr><tr><td>GR-DUET* [6]</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>9.9 ±0.3</td><td>5.5 ±0.0</td><td>47.1 ±0.5</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>42.2 ±0.8</td><td>54.1 ±0.6</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>+Memoir (Ours)</td><td>10.3 ±0.4</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>5.1 ±0.0</td><td>50.2 ±0.3</td><td>44.8 ±0.4</td><td>56.2 ±0.6</td></tr></table>

*在对齐实验条件下重复的结果。讨论。Memoir 的性能始终优于 GR-DUET，尽管相较于 IR2R 的增益较小。这一改善的减少源于内存密度的差异，GSA-R2R 平均积累了 600 个回合，从而提高了 GR-DUET 的拓扑完整性。

# 5.3 定性分析。

图3展示了在DUET和GR-DUET失败的情况下，记忆检索的有效性。在找定位“按摩床”的任务中，走廊里仅能隐约看到两个潜在候选者。DUET智能体错误地接近了错误目标而没有观察实际目标，而GR-DUET智能体在众多候选位置中感到困惑，并作出了错误决定。我们的模型通过观察检索（识别通往相关位置的有希望路径，同时控制冗余）和历史检索（匹配类似的过去事件，针对“按摩室”目标，促进智能体到达目的地）的结合取得成功。表5比较了在GSA-R2R基准上基本指令下的导航性能。

<table><tr><td rowspan="2">Methods</td><td colspan="5">Test-R-Basic</td><td colspan="5">Test-N-Basic</td></tr><tr><td>TL↓</td><td>NE↓</td><td>SR↑</td><td>SPL ↑</td><td>nDTW↑</td><td>TL↓</td><td>NE↓</td><td>SR↑</td><td>SPL ↑</td><td>nDTW ↑</td></tr><tr><td>TourHAMT [5]</td><td>11.6 ±0.1</td><td>7.4 ±0.1</td><td>14.9 ±0.1</td><td>12.2 ±0.1</td><td>34.7 ±0.1</td><td>9.4 ±0.1</td><td>7.7 ±0.1</td><td>11.0 ±0.2</td><td>8.6 ±0.2</td><td>32.2 ±0.1</td></tr><tr><td>OVER-NAV [7]</td><td>14.1 ±0.1</td><td>6.7 ±0.0</td><td>22.3 ±0.3</td><td>16.8 ±0.2</td><td>37.1 ±0.1</td><td>11.4 ±0.1</td><td>7.1 ±0.1</td><td>16.6 ±0.2</td><td>13.0 ±0.1</td><td>35.0 ±0.2</td></tr><tr><td>DUET [9]</td><td>13.1</td><td>4.2</td><td>57.7</td><td>47.0</td><td>55.6</td><td>14.8</td><td>5.3</td><td>48.1</td><td>37.3</td><td>45.9</td></tr><tr><td>+MLM [15]</td><td>13.1 ±0.1</td><td>4.1 ±0.1</td><td>57.9 ±0.2</td><td>47.3 ±0.1</td><td>55.9 ±0.2</td><td>13.1 ±0.2</td><td>5.3 ±0.1</td><td>48.3 ±0.5</td><td>38.8 ±0.5</td><td>48.4 ±0.3</td></tr><tr><td>+MRC [15]</td><td>13.1 ±0.1</td><td>4.2 ±0.1</td><td>57.7 ±0.1</td><td>47.0 ±0.1</td><td>55.6 ±0.1</td><td>14.7 ±0.1</td><td>5.3 ±0.1</td><td>48.1 ±0.1</td><td>37.3 ±0.1</td><td>45.9 ±0.1</td></tr><tr><td>+BT [53]</td><td>8.0 ±0.1</td><td>3.8 ±0.1</td><td>61.3 ±0.6</td><td>57.7 ±0.3</td><td>70.1 ±0.5</td><td>7.9 ±0.0</td><td>5.2 ±0.1</td><td>49.5 ±0.8</td><td>46.0 ±0.8</td><td>59.4 ±0.9</td></tr><tr><td>+TENT [54]</td><td>14.6 ±0.0</td><td>4.2 ±0.0</td><td>57.2 ±0.4</td><td>44.2 ±0.4</td><td>52.9 ±0.1</td><td>16.2 ±0.1</td><td>5.4 ±0.1</td><td>46.5 ±0.4</td><td>33.7 ±0.2</td><td>42.6 ±0.3</td></tr><tr><td>+SAR [55]</td><td>13.8 ±0.8</td><td>4.0 ±0.1</td><td>57.6 ±0.2</td><td>44.6 ±0.2</td><td>53.0 ±0.2</td><td>16.5 ±0.0</td><td>5.4 ±0.0</td><td>44.6 ±1.5</td><td>31.5 ±1.6</td><td>40.6 ±1.3</td></tr><tr><td colspan="9">VLN models pretrained with full navigation graph:</td><td></td><td></td></tr><tr><td>GR-DUET [6]</td><td>9.4 ±0.0</td><td>3.1 ±0.0</td><td>69.3 ±0.2</td><td>64.3 ±0.1</td><td>71.4 ±0.1</td><td>8.9 ±0.0</td><td>4.4 ±0.0</td><td>56.6 ±0.1</td><td>51.5 ±0.1</td><td>61.0 ±0.1</td></tr><tr><td>GR-DUET* [6]</td><td>8.6 ±0.2</td><td>3.2 ±0.1</td><td>67.6 ±0.5</td><td>63.6 ±0.6</td><td>71.9 ±0.5</td><td>8.7 ±0.4</td><td>4.4 ±0.0</td><td>55.3 ±0.2</td><td>50.4 ±0.3</td><td>60.8 ±0.4</td></tr><tr><td>+Memoir (Ours)</td><td>9.3 ±0.0</td><td>3.0 ±0.0</td><td>69.8 ±0.2</td><td>64.9 ±0.4</td><td>73.3 ±0.2</td><td>9.3 ±0.2</td><td>4.2 ±0.0</td><td>57.7 ±0.1</td><td>52.0 ±0.1</td><td>61.9 ±0.4</td></tr></table>

![](images/3.jpg)  
Ixieea  waiexeal.

表6 计算效率比较（批处理大小 $= 4$）

<table><tr><td rowspan="2">Methods</td><td colspan="2">Training</td><td colspan="2">Inference</td></tr><tr><td>Memory↓</td><td>Latency↓</td><td>Memory↓</td><td>Latency↓</td></tr><tr><td>DUET [9]</td><td>7.2 GB</td><td>0.15s</td><td>2.2 GB</td><td>0.13s</td></tr><tr><td>GR-DUET [6]</td><td>29.4 GB</td><td>4.39s</td><td>9.9 GB</td><td>0.25s</td></tr><tr><td>Memoir (Ours)</td><td>13.1 GB (-55%)</td><td>0.53s (-88%)</td><td>2.6 GB (-74%)</td><td>0.31s (+28%)</td></tr></table>

# 5.4 消融研究与分析

计算效率。表6展示了Memoir在计算上的优势，相较于内存持久基线。虽然DUET在内存开销上几乎为零，但GR-DUET的完全内存保留策略显著增加了资源需求（训练时29.4GB，推理时9.9GB），因为它同时处理所有累积的观察数据。我们的检索机制实现了显著的效率提升：训练内存减少$5 5 \%$，训练延迟减少$8 8 \%$，实现了$8 . 3 \times$的加速。在推理阶段，内存使用减少了$7 4 \%$，接近DUET的效率，同时保持内存持久能力。轻微的推理延迟增加（0.31s对比0.25s）反映了想象引导检索的开销，暗示通过缓存或并行处理的未来优化机会。这些结果确立了Memoir作为首个实现实际效率与最先进性能的内存持久视觉语言导航方法，使得在资源受限的部署环境中连续学习成为可能，同时推动导航能力超越现有方法。

![](images/4.jpg)  
F   
Fig. 4. Performance scaling across tour progression on IR2R.

表 7 记忆检索组件的消融实验。

<table><tr><td colspan="3">Observation</td><td colspan="3">History</td><td colspan="9">IR2R Val Unseen</td></tr><tr><td></td><td>Retr Rand Full PERF</td><td></td><td></td><td></td><td>Retr Rand Full Perf</td><td></td><td></td><td>TL↓NE↓ SR↑SPL↑NDTW↑OR↑OA↑HR↑HA↑</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">The upper-bound of long-term memory retrieval</td><td colspan="7"></td></tr><tr><td></td><td></td><td>✓</td><td></td><td></td><td></td><td>9.77</td><td>0.51</td><td>95.44</td><td>93.40</td><td>93.68</td><td></td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td></td><td colspan="7"></td><td>12.24 2.81</td><td>72.33</td><td>63.97</td><td>70.35</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td></td><td colspan="7"></td><td>10.44</td><td>2.86 74.67</td><td>69.98</td><td>76.29</td><td>100</td><td>9.81</td><td>100</td><td>19.33</td></tr><tr><td></td><td>✓</td><td>✓</td><td></td><td></td><td>✓</td><td></td><td>10.97</td><td>2.76</td><td>75.82</td><td>70.34</td><td>76.03</td><td>59.05 21.31</td><td></td><td>36.65</td><td>22.11</td></tr><tr><td>✓</td><td></td><td></td><td></td><td>✓</td><td></td><td></td><td>10.77</td><td>2.58</td><td>76.63</td><td>71.70</td><td>78.08</td><td>97.05</td><td>23.33</td><td>36.24</td><td>21.94</td></tr><tr><td></td><td>✓</td><td></td><td></td><td></td><td></td><td></td><td>10.80</td><td>2.61</td><td>76.63</td><td>71.03</td><td>76.98</td><td>58.83</td><td>21.72</td><td>98.36</td><td>22.40</td></tr><tr><td>✓</td><td></td><td></td><td></td><td>✓ ✓</td><td></td><td></td><td>10.32</td><td>2.53</td><td>78.03</td><td>73.46</td><td>79.46</td><td>96.49</td><td>24.58</td><td>96.52</td><td>24.21</td></tr></table>

RETR检索重构；RAND随机采样；FULL完整合并；PERF竞赛对比。表8 世界模型变体的消融研究。

<table><tr><td></td><td></td><td colspan="5">IR2R Val Unseen</td></tr><tr><td>MODEL</td><td>Dist</td><td>SR ↑</td><td>SPL ↑</td><td>OR ↑</td><td>OA ↑</td><td>HR↑ HA↑</td></tr><tr><td>GRU</td><td>5</td><td>| 76.12</td><td>71.48</td><td>97.54</td><td>16.90</td><td>30.21 30.78</td></tr><tr><td>+overshoot</td><td>5</td><td>| 76.71</td><td>72.30</td><td>96.21</td><td>18.60</td><td>98.24 21.65</td></tr><tr><td>Transformer</td><td>1 3</td><td>73.44 75.18</td><td>67.24 70.21</td><td>31.25 76.71</td><td>40.33 26.05</td><td>26.26 30.74 29.11 32.29</td></tr><tr><td></td><td>5</td><td>77.14</td><td>72.11</td><td>96.44</td><td>17.86</td><td>28.58 32.32</td></tr><tr><td>+overshoot</td><td>1</td><td>73.09</td><td>67.26</td><td>31.15</td><td>43.22</td><td>59.43 29.24</td></tr><tr><td></td><td>3</td><td>76.63 78.03</td><td>71.99 73.46</td><td>77.04 96.49</td><td>28.48 24.58 96.52</td><td>93.40 25.06 24.21</td></tr></table>

性能扩展。图4展示了在IR2R巡回中，智能体完成轮次时的扩展性能。Memoir表现出最显著的提升，其高性能在巡回结束时仍能持续保持。相比之下，DUET的性能相对稳定但较低（$6 5 \mathrm { - } 7 2 \%$ SR，$5 5 – 6 3 \%$ SPL），并未从积累的经验中学习。GR-DUET的扩展表现不一致，其性能在$9 0 \%$进展时显著恶化，低于DUET的基准。这一比较验证了以想象驱动的记忆检索，相较于简单的记忆整合，能够更一致且显著地提升基于积累经验的性能收益。

内存组件。表 7 展示了验证内存组件有效性的结果。“完美”变体直接整合了通向目标位置的记忆，模拟了具有完美检索能力的理想世界模型行为，在未见环境中实现了 $9 3 . 4 0 \%$ 的 SPL，突显了检索的必要性，并作为性能上限。禁用观察和历史组件的变体表现最差，其次是完全整合长期记忆。随机内存选择提升了 $0 . 3 6 \%$ 的导航性能，而用我们的检索方法替代随机选择提高了两种内存类型的导航性能，证明了观察和历史都包含了有价值的信号——只要它们被有选择地访问。我们的检索变体实现了最佳导航性能和检索准确率 $( 2 4 . 5 8 \%$ OA, $2 4 . 2 1 \%$ HA)。相反，完全内存整合降低了准确性，随机选择显著减少了召回率（OR 和 HR），证明了预测性想象能够实现精准的内存过滤，而不依赖于信息不足和消耗性的方法。与理想世界模型的显著差距揭示了当前世界模型在准确捕捉环境动态方面的挑战。世界模型将受益于数据扩展和先进架构，以进一步提升未来性能。表 9 导航历史整合的消融实验。

<table><tr><td colspan="6">IR2R Val Unseen</td></tr><tr><td>Hist Encoder</td><td>Embed Type</td><td>SR ↑</td><td>SPL↑</td><td>nDTW↑</td><td>t-nDTW↑</td></tr><tr><td></td><td>VP + State</td><td>76.59</td><td>72.35</td><td>78.60</td><td>65.57</td></tr><tr><td>✓</td><td>VP</td><td>76.54</td><td>71.48</td><td>78.24</td><td>65.66</td></tr><tr><td>√</td><td>State</td><td>77.35</td><td>72.83</td><td>78.70</td><td>66.37</td></tr><tr><td>v</td><td>VP + State</td><td>78.03</td><td>73.46</td><td>79.46</td><td>66.44</td></tr></table>

表 10 专家策略的消融实验。

<table><tr><td></td><td colspan="3">IR2R Val Unseen</td><td colspan="3">GSA Test-R-Basic</td></tr><tr><td>Expert PoLicy</td><td>SR↑</td><td>SPL ↑</td><td>nDTW↑</td><td>SR ↑</td><td>SPL↑</td><td>nDTW↑</td></tr><tr><td>SPL</td><td>76.20</td><td>71.71</td><td>78.03</td><td>67.34</td><td>62.22</td><td>71.30</td></tr><tr><td>+random sample</td><td>78.03</td><td>73.46</td><td>79.46</td><td>69.64</td><td>64.91</td><td>73.29</td></tr></table>

世界模型。表8评估了世界模型的不同变体，比较了带和不带超越的GRU与Transformer体系结构。在5步超越距离下，Transformer变体的SPL比GRU变体提高了$1 . 1 6 \%$，观察检索准确率（OA）提高了$7 . 6 8 \%$，历史检索准确率（HA）提高了$2 . 5 6 \%$。随着想象步骤的增加，检索效果通常会提高，因为探索范围扩大，观察（OR）和历史（HR）的召回率提高，从而更好地支持导航决策。然而，由于在更大距离下拓扑图中候选项指数级增加，检索准确率下降。超越目标显著增强了OA和HR，促进了稳健的导航性能（$ + 1 . 3 6 \%$ SPL）。这些结果验证了有效的检索需要强大的预测模型（Transformer优于GRU）和扎实的训练（超越），以平衡探索广度与查询精度。表11 世界模型预训练的消融实验。

<table><tr><td></td><td colspan="3">IR2R Val Unseen</td><td colspan="3">GSA Test-R-Basic</td></tr><tr><td>PRETRAIN</td><td>SR↑</td><td>SPL↑</td><td>nDTW ↑</td><td>SR↑</td><td>SPL↑</td><td>nDTW ↑</td></tr><tr><td></td><td>74.93</td><td>71.55</td><td>78.12</td><td>64.62</td><td>61.52</td><td>71.30</td></tr><tr><td>✓</td><td>78.03</td><td>73.46</td><td>79.46</td><td>69.64</td><td>64.91</td><td>73.29</td></tr></table>

表12 观察补全的消融实验。

<table><tr><td rowspan="2"></td><td colspan="3">IR2R Val Unseen</td><td colspan="3">GSA Test-R-Basic</td></tr><tr><td>Neighbor obs SR ↑</td><td>SPL ↑</td><td>nDTW↑</td><td>SR↑</td><td>SPL↑</td><td>nDTW↑</td></tr><tr><td>Partial</td><td>77.05</td><td>72.15</td><td>78.22</td><td>68.35</td><td>63.90</td><td>72.89</td></tr><tr><td>Completion</td><td>78.03</td><td>73.46</td><td>79.46</td><td>69.64</td><td>64.91</td><td>73.29</td></tr></table>

历史集成。表9比较了整合检索历史的策略。在专用编码器中将视点特征与状态特征串联，能够实现最佳性能，较仅使用视点特征提高了$1.98\%$的SPL，相较于仅使用状态特征提高了$0.63\%$。这表明状态特征携带关键的模式信息，同时仍然受益于观察表示的增强。未整合历史编码器时，在历史表示与粗尺度编码器输入串联的情况下，性能下降了$1.11\%$的SPL，展示了在三个不同编码器之间分工的必要性。专家策略。表10比较了在内存检索动态扩展可用动作空间超出直接邻域时的训练策略。在训练期间随机抽样多条最佳路径（实现$73.46\%$的SPL）优于基于确定性SPL的专家选择（$71.71\%$的SPL），因为它提供了更好的策略正则化和对导航选择的鲁棒性。世界模型预训练。表11展示了正确初始化世界模型的重要性。在联合训练之前对导航轨迹上的世界模型组件进行预训练，相较于未预训练的情况下，IR2R的性能提高了$1.91\%$的SPL，GSA-R2R的性能提高了$3.39\%$的SPL，这表明随机初始化的世界模型提供较差的检索信号。观察补全。表12展示了使用来自$\mathcal{M}_{o}$的存储特征，在未检索的视点上完成部分观察显著增强了对环境的理解。当情节图中的视点缺乏完整的视觉信息时，检索其存储的全景特征可以实现更为明智的决策。邻域整合。表13研究了在观察检索期间整合检索节点相邻视点的检索策略。包括直接邻居提供了关于连通性和周围环境的更丰富的空间上下文，使粗尺度编码器能够做出更为明智的规划决策。这种方法在各自基准上提升了$2.55\%$和$1.89\%$的SPL。参数研究。图5分析了关键检索超参数的影响。在观察检索中，SR随着滤波率的降低和搜索宽度的增加而改善，在$\rho_{o} = 0.2$和$W = 12$时达到峰值，随后随着过度上下文引入噪声而下降。这表明整合更广泛的视点观察有助于做出更为明智的导航决策。对于导航历史检索，模型优先考虑精度而非召回率，在$\theta_{h} = 0.2$时实现最佳性能，$P = 10$。在阈值$\theta_{h} \approx 1.0$和最大模式$\bar{P} = 20$（衰减因子$\gamma_{h} = 0.8$）处出现次优，通过增加模式接收来补偿高度限制的相似度阈值。表13 邻域整合的消融研究。

<table><tr><td></td><td colspan="3">IR2R Val Unseen</td><td colspan="3">GSA Test-R-Basic</td></tr><tr><td>RETRIEVE</td><td>SR ↑</td><td>SPL↑</td><td>nDTW↑</td><td>SR↑</td><td>SPL↑</td><td>nDTW ↑</td></tr><tr><td>VP only</td><td>76.46</td><td>70.91</td><td>77.45</td><td>67.44</td><td>63.02</td><td>71.83</td></tr><tr><td>VP + neighbors</td><td>78.03</td><td>73.46</td><td>79.46</td><td>69.64</td><td>64.91</td><td>73.29</td></tr></table>

![](images/5.jpg)  
Fig. 5. Study of hyper-parameters of retrieval on IR2R val-unseen. Left: Environmental observation retrieval. Right: Navigation history retrieval.

# 5.5 失败分析

图6展示了一个场景，在该场景中，Memoir虽然按照设计功能正常，但仍然失败。在情节A中，智能体必须导航到一个在之前情节中不存在的卧室。观察检索识别出两个干扰性的卧室入口作为候选项，而历史检索则挖掘出情节B（失败）和情节C（成功），这两者的目标都是不同的卧室。检索限制。检索到的观察结果倾向于整合在图5(a)中发现的丰富的有前景的候选项，突出两个卧室入口在语义上相关，但未能区分关键的空间特征——“距离书桌最近”。检索到的历史同样无法区分情节B和C，尽管它们的目标不同。在情节B中，因在一步后过早终止想象而限制了检索，只能获取到最近的入口，阻止了正确目标的发现。这些失败揭示了在两种记忆类型的预测检索中世界模型的不足。探索-利用权衡。当两个检索类型汇聚到同一个错误位置时，情节A失败。智能体优先选择高相似度的历史，如图5(b)所示，在检索未能覆盖真实目标时，默认为利用而非探索。它也未能区分任务结果，将情节B和C视为相同，而不是从成功中学习。这突显了一个新的挑战：何时应信任积累的经验，以及何时新颖的替代方案应该被调查。

![](images/6.jpg)

未来工作。这些失败模式提示了两个潜在的研究方向，以推进想象引导的记忆检索。首先，通过大规模预训练和明确的空间关系建模增强世界建模能力，可以解决在区分空间上不同的目标时的检索不准确和影响检索范围的过早想象视野问题。其次，自信稀疏检索用于判断何时应信任检索到的经验，这需要动态估计检索信心，以平衡利用与探索之间的关系。我们的方法与预言检索上限之间的显著性能差距（$73.46\%$ SPL与表7中的$93.40\%$ SPL）表明在这些方向上仍有很大的改进空间。

# 6 结论

本研究提出了Memoir，这是一个基于记忆持久性的视觉语言导航（VLN）智能体，利用预测性世界建模进行自适应体验检索。与传统的孤立轨迹生成想象规划不同，我们通过语言条件的世界模型将想象进行具体化，该模型为混合视角级别记忆（HVM），用于存储观察数据和行为模式，并结合了增强体验的导航模型。大量实验表明，在IR2R上提升了$5.4\%$的成功路径长度（SPL），实现了$8.3\times$的训练速度提升和$74\%$的推理内存减少，验证了环境和行为记忆的预测性检索能够实现更有效的导航。oracle检索性能为$93.4\%$的SPL，展示了基于想象引导的检索潜力。未来的工作应探讨增强的世界建模和基于信心的探索机制，以缩小这一差距，建立一个将预测模拟与具体记忆连接起来的原则性框架，以支持具身人工智能的发展。

# REFERENCES

[1] P. Anderson et al., "Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments,' in CVPR, 2018, pp. 36743683.   
[2] Y. Qi et al., "Reverie: Remote embodied visual referring expression in real indoor environments," in CVPR, 2020, pp. 99829991.   
[3] A. Ku, P. Anderson, R. Patel, E. Ie, and J. Baldridge, "Room-acrossroom: Multilingual vision-and-language navigation with dense spatiotemporal grounding," in EMNLP, 2020, pp. 43924412.   
[4] S. Wani, S. Patel, U. Jain, A. Chang, and M. Savva, "Multion: Benchmarking semantic map memory using multi-object navigation," in NeurIPS, vol. 33, 2020, pp. 97009712.   
[5] J. Krantz et al., "Iterative vision-and-language navigation," in CVPR, 2023, pp. 14 92114 930.   
[6] H. Hong, Y. Qiao, S. Wang, J. Liu, and Q. Wu, "General scene adaptation for vision-and-language navigation," in ICLR, 2025.   
[7] G. Zhao, G. Li, W. Chen, and Y. Yu, "Over-nav: Elevating iterative vision-and-language navigation with open-vocabulary detection and structured representation," in CVPR, 2024, pp. 16 29616 306.   
[8] Q. Zheng, D. Liu, C. Wang, J. Zhang, D. Wang, and D. Tao, Esceme: Vision-and-language navigation with episodic scene memory," IJCV, pp. 121, 2024.   
[9] S. Chen, P.-L. Guhur, M. Tapaswi, C. Schmid, and I. Laptev, "Think global, act local: Dual-scale graph transformer for vision-andlanguage navigation," in CVPR, 2022, pp. 16 53716 547.   
[10] M. Seeber et al., "Human neural dynamics of real-world and imagined navigation," Nat. Hum. Behav., vol. 9, no. 4, pp. 781793, 2025.   
[11] M. Karl, F. Kock, B. W. Ritchie, and J. Gauss, "Affective forecasting and travel decision-making: An investigation in times of a pandemic," Ann. Tour. Res., vol. 87, p. 103139, 2021.   
[12] Y. W. Li and L. C. Wan, "Inspiring tourists' imagination: How and when human presence in photographs enhances travel mental simulation and destination attractiveness," Tour. Manag., vol. 106, p. 104969, 2025.   
[13] H. Wang, W. Liang, L. Van Gool, and W. Wang, "Dreamwalker: ICCV, 2023, pp. 1087310883. navigation," in NeurIPS, vol. 31, 2018.   
[15] W. Hao, C. Li, X. Li, L. Carin, and J. Gao, "Towards learning a generic agent for vision-and-language navigation via pretraining," in CVPR, 2020, pp. 13 13713 146. navigation," in ICC, 2023, pp. 1200120.   
[] J. Zhang et l., "Navid: Video-based vlm plans the next step for vision-and-language navigation," in RSS, 2024.   
[18] Y. Xu, Y. Pan, Z. Liu, and H. Wang, "Flame: Learning to navigate with multimodal llm in urban environments," in AAAI, vol. 39, 2025, pp. 90059013.   
[19] Y. Hong, Q. Wu, Y. Qi, C. Rodriguez-Opazo, and S. Gould, "Vln bert: A recurrent vision-and-language bert for navigation," in CVPR, 2021, pp. 16431653.   
[20] S. Chen, P.-L. Guhur, C. Schmid, and I. Laptev, "History aware multma onndan t," i NeurIPS, vol. 34, 2021, pp. 58345847.   
[21] H. Wang, W. Wang, W. Liang, C. Xiong, and J. Shen, "Structured .   
[22] E. Parisotto and R. Salakhutdinov, "Neural map: Structured memory for deep reinforcement learning," in ICLR, 2018.   
. eo framework based on multimodal large language models," arXiv preprint arXiv:2507.13152, 2025.   
[24] C. Wang, S. Wei, and J. Qi, "Matchnav: Llm-based enhanced description and instruction matching in vision-and-language navigation," Inf. Fusion, vol. 125, p. 103444, 2026.   
[25] J. F. Henriques and A. Vedaldi, "Mapnet: An allocentric spatial memory for mapping environments," in CVPR, 2018, pp. 8476 8484.   
[26] S. K. Ramakrishnan, Z. Al-Halah, and K. Grauman, "Occupancy anticipation for efficient exploration and navigation," in ECCV, 2020, pp. 400418.   
[27] V. Cartillier, Z. Ren, N. Jain, S. Lee, I. Essa, and D. Batra, "Semantic mapnet: Building allocentric semantic maps and representations fro egocentric views," in AAAI, vol. 35, 2021, pp. 964972. maps for robot navigation," in ICRA, 2023, pp. 10 60810 615.   
[9] Z. Te  6: bird's-eye view," in WACV, 2024, pp. 373382.   
[30] R. Liu, X. Wang, W. Wang, and Y. Yang, "Bird's-eye-view scene graph for vision-language navigation," in ICCV, 2023, pp. 10968 10 980.   
[31] D. S. Chaplot, R. Salakhutdinov, A. Gupta, and S. Gupta, "Neural icl samal ation, R, 00 . 12884.   
[32] N. Kim, O. Kwon, H. Yoo, Y. Choi, J. Park, and S. Oh, "Topological semantic graph memory for image-goal navigation," in CoRL, 2023, pp. 393402.   
grounded robot navigation," in RSS, 2024.   
Hae   Leag latn na  pan om pixels," in ICML, 2019, pp. 25552565.   
[35] D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi, "Dream to control: Learning behaviors by latent imagination," in ICLR, 2020.   
[36] J. Lin et al., "Learning to model the world with language," in ICML, vol. 235, 2024, pp. 2999230017.   
[ J. W, H. M C. De, and . Lo, "re-ra cetalzd in NeurIPS, vol. 36, 2024.   
[38] X. Ma, S. Chen, D. Hsu, and W. S. Lee, "Contrastive variational reinforcement learning for complex observations," in CoRL, 2021, pp. 959972.   
[39] M. Okada and T. Taniguchi, "Dreaming: Model-based reinforceICRA, 2021, pp. 42094215.   
[40] A. Bar, G. Zhou, D. Tran, T. Darrell, and Y. LeCun, "Navigation world models," in CVPR, 2025, pp. 15 79115 801.   
[41] J. Li and M. Bansal, "Panogen: Text-conditioned panoramic environment generation for vision-and-language navigation," in NeurIPS, vol. 36, 2023, pp. 21 87821 894.   
[42] X. Yao, J. Gao, and C. Xu, "Navmorph: A self-evolving world model for vision-and-language navigation in continuous environments," arXiv preprint arXiv:2506.23468, 2025.   
[43] J. Y. Koh, H. Lee, Y. Yang, J. Baldridge, and P. Anderson, "Pathdreamer: A world model for indoor navigation," in ICCV, 2021, pp. 1473814748.   
[44] G. Georgakis, K. Schmeckpeper, K. Wanchoo, S. Dan, E. Miltsakaki, D. Roth, and K. Daniilidis, "Cross-modal map learning for vision and language navigation," in CVPR, 2022, pp. 15 460 15 470.   
[45] Y. Pan, Y. Xu, Z. Liu, and H. Wang, "Planning from imagnation: Episodic simulation and episodic memory for vision-andlanguage navigation," in AAAI, vol. 39, 2025, pp. 63456353.   
[46] J. Li and M. Bansal, "Improving vision-and-language navigation by geneating uture-vie ma smanti," i , 202, pp. 10 80310 812.   
[47] S. Wang et al., "Monodream: Monocular vision-language navigation with panoramic dreaming," arXiv preprint arXiv:2508.02549, 2025.   
[48] H. Le, T. Karimpanal George, M. Abdolshah, T. Tran, and S. Venkatesh, "Model-based episodic memory induces dynamic hybrid controls," in NeurIPS, vol. 34, 2021, pp. 3031330 325.   
[49] D. P. Kingma and M. Welling, "Auto-encoding variational bayes," in ICLR, 2013.   
[50] A. v. d. Oord, Y. Li, and O. Vinyals, "Representation learning with contrastive predictive coding," arXiv preprint arXiv:1807.03748, 2018.   
[51] S. K. Ramakrishnan et al., "Habitat-matterport 3d dataset (hm3d): 1000 large-scale 3d environments for embodied ai," in NeurIPS D&B, 2021.   
[52] P. Anderson et l., "On evaluation of embodied navigation agents," arXiv preprint arXiv:1807.06757, 2018.   
[53] H. Wang, W. Wang, T. Shu, W. Liang, and J. Shen, "Active visual information gathering for vision-language navigation," in ECCV, 2020, pp. 307322.   
[54] D. Wang, E. Shelhamer, S. Liu, B. Olshausen, and T. Darrell, "Tent: Fully test-time adaptation by entropy minimization," in ICLR, 2021.   
[5] . Niu et al., "Towards stable test-time adaptation in dynamic wild world," in ICLR, 2023.