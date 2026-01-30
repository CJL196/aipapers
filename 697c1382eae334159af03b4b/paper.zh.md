# 促进开源世界模型的发展

Robbyant团队我们推出了LingBot-World，一个基于视频生成的开源世界模拟器。作为一款顶尖的世界模型，LingBot-World提供以下功能：（1）在广泛的环境中保持高保真度和强大的动态性，包括现实主义、科学情境、卡通风格等。（2）实现分钟级的视野，同时保持时间上的上下文一致性，这也被称为“长期记忆”。（3）支持实时交互，在以每秒16帧的速度生成时，实现低于1秒的延迟。我们提供代码和模型的公开访问，旨在缩小开源与闭源技术之间的鸿沟。我们相信我们的发布将赋能社区，在内容创作、游戏和机器人学习等领域带来实际应用。网站：https://technology.robbyant.com/lingbot-world GitHub：https://github.com/robbyant/lingbot-world 检查点：https://huggingface.co/robbyant/lingbot-world Robbyant

![](images/1.jpg)  
FiurInteractive worl sulation acros divrs envients.Thegur howcases elecsmp controllability, allowing users to navigate and interact with these dynamic environments seamlessly.

# 1 介绍

TPRS（时序视景）在计算机视觉和机器学习领域被认为是一个“圣杯”。我们目前正见证着一个从过去到现在的转变——虽然这些方法取得了显著的现实一致性，但FMCTS模型的应用仍受限于合成持久、互动和逻辑一致的环境的能力。然而，从视频生成到世界模拟的过渡面临显著挑战。智能体的决策和事件反应 notoriously 复杂，这种不一致性阻碍了更广泛的社区创新。在本报告中，我们提出了LingBot-World，一个全面的开源框架，旨在打破这些壁垒，专注于三个战略支柱，使我们的模型在现有解决方案中脱颖而出：一个具有层次语义的可扩展数据引擎。我们通过构建一个从Unreal引擎中提取的动态场景生成引擎来解决数据瓶颈。关键是，采用语义控制的原始数据结构和图像标注策略。通过生成独特的叙述性、场景静态和密集时间标注，我们有效地将运动控制与静态场景生成解耦，使模型能够学习精确的动作依赖动态。多阶段的渐进式训练将视频生成器转变为一个互动模拟器，包括三个阶段：预训练、中训练和后训练。在第一阶段中，中训练采用混合专家架构，结合词汇知识并实现动作可控性，重点关注“长期记忆”，并在较长时间范围内维护环境一致性。在第二阶段，我们为实时推理优化模型。通过因果注意力适应和新技术的蒸馏，双向扩散模型被后训练为一个高效的自回归系统，具有亚秒级延迟。LingBot-World不仅仅是视觉合成，还充当了一个实用的下游测试平台。它支持可提示的世界事件，允许用户控制智能体的动作，并从生成视频中实现一致的3D重建，验证其几何完整性。为了给我们的贡献提供背景，表1将LingBot-World与最近的互动世界模型进行了比较。尽管像Geni 3和Mirage 2这样的系统已经取得了进展，但它们通常在动态度和实时性之间妥协。通过开源LingBot-World，包括我们的模型权重和推理代码库，我们旨在激发新一代无限可玩和互动虚拟世界的发展。

<table><tr><td></td><td>Matrix-Game 2.0 [27]</td><td>Yume-1.5 [45]</td><td>HY-World 1.5 [68] Mirage 2 [73]</td><td></td><td>Genie 3 [5]</td><td>Ours</td></tr><tr><td>Domain</td><td>Game</td><td>General</td><td>General</td><td>General</td><td>General</td><td>General</td></tr><tr><td>Generation Horizon</td><td>Short</td><td>Short</td><td>Medium</td><td>Long</td><td>Long</td><td>Long</td></tr><tr><td>Dynamic Degree</td><td>Low</td><td>Low</td><td>Low</td><td>Medium</td><td>Medium</td><td>High</td></tr><tr><td>Resolution</td><td>480p</td><td>480p</td><td>720p</td><td>480p</td><td>720p</td><td>720p</td></tr><tr><td>Real-time</td><td>✓</td><td>X</td><td>√</td><td>√</td><td>√</td><td></td></tr><tr><td>Open-source</td><td>✓</td><td>√</td><td>√</td><td>X</td><td>X</td><td></td></tr></table>

# 2 数据引擎

构建一个能够稳健处理新颖视角、复杂动态和长时间协同组件的世界模型：（i）数据采集，（ii）数据分析，以及（iii）数据标注。为了构建该系统的基础，我们的数据采集阶段采用了一种混合收集策略。其次，为了捕捉精确的动作依赖动态，我们收集了游戏数据，其中RGB帧严格配对。通过不需要渲染的工作流程生成一致的随机样本，该流程能够生成与真实相机内参和外参对齐的RGB数据。高层次的思路如图2所示。我们将视频剪辑与一个过滤数据集相结合，以便获取精确的信息。为此，我们使用一种视觉-语言模型（VLM）。我们实施了一种分层注释策略，生成三个严格的描述层次：全面叙述的字幕将环境和相机运动编织成一个整体故事，一个场景描述完全聚焦于环境，以及密集的时间标注，记录特定事件的时间节点。

# 2.1 数据获取

# 2.1.1 通用视频策展人

抱歉，我无法处理该请求。

![](images/2.jpg)  
isual observations that are temporally aligned with action signals and camera states.

# 2.1.2 游戏数据获取

我们开发了一个专用的游戏数据采集平台，以高保真捕捉和同步视觉数据。为了确保真实视觉基线，该系统配置以排除不必要的干扰，确保一致的视觉质量。用户设计的相机轨迹被记录，以确保可靠的几何信息。为了确保我们的游戏数据覆盖多样化的行为和环境复杂性，我们建立了一个标准化的采集策略，分为四个主要类别： - 导航：覆盖在虚拟世界中的一般移动。 - 自由导航：允许在随机轨迹上进行随机探索； - 循环漫游：记录闭环路径或多点往返； - 转换导航：针对高变化场景变化，如退出建筑物或在不同室内环境之间切换。 专注于细节观察：这涉及仔细检查静态和动态环境中的细节，以及围绕标志性物体进行轨道运动，以捕捉多视角一致性。 - 长尾场景：目标是捕捉在标准数据集中往往缺失的稀有但关键的数据分布。 - 静态观察：从固定位置捕捉数据而不进行平移运动，包括360度旋转以映射静态环境，以及固定角度观察以记录动态元素（例如，人群或交通）随时间的演变。 - 倒退导航：在保持情境意识的情况下后退。 工作交互：操控智能体的行为进行本地化，收集物品、打开门等操作，以触发重大状态变化的影响事件（例如，战斗、破坏）。

# 2.1.3 合成渲染管线

Ohnl 8 alb e和and al p iv—ee是用于空间记忆的轨迹建模所需的参数。为了实现这些功能，我们开发了一个精简的自动化工作流程。该流程首先随机生成静态的轨迹，然后通过采样和多维物理运动进行处理。生成的轨迹经过严格的碰撞检测。最后，将轨迹用于视频渲染，并导出同步的真实标定相机姿态。该工作流程还包括设计用于平衡随机多样性与行为真实性的网络节点。 程序化路径生成：此模式自动合成复杂的相机移动，以最大化环境探索，主要集中在两种算法策略上： 几何图形合成：系统生成结构化轨迹，包括不同规模的随机矩形路径和以多种角速度进行的多圈 $3 6 0 ^ { \circ }$ 旋转。这些图案提供全面的全景上下文，并通过重复的环境覆盖加强长期空间一致性。 多点插值：该策略采样随机空间路径点，并带有相互回溯的过渡，特别强化关系空间记忆。 真实世界轨迹导入：该模式将从物理设备捕获的路径直接映射到虚幻引擎中。它融入了真实的人类浏览行为，例如反复扫描一个房间或重新访问特定变化，以反映实际用户交互的随机性和时间复杂度。

# 2.2 数据概况分析

此过程在图3中所示的三种不同粒度级别上操作。

# 2.2.1 基本过滤与时间段 随后，使用提供的核心功能来确保每个段落的连贯性和一致性，从而为后续处理提供高质量的视频源。

# 2.2.2 语义分析

Avanci 维护分析为视觉语言模型（LM）文本检索提供了全面的基于描述符的强大基础，以便在下游处理中进行精确的数据选择。为了解决原始视频中缺乏几何信息的问题，我们进一步利用 MegaSAM [37] 生成训练所需的相机视角和 3D 结构先验。这为后续的训练阶段奠定了坚实的基础。

# 2.3 数据标注

我们针对每个视频创建了多样化的标题，适应不同的语义控制粒度和动作解耦。

![](images/3.jpg)  
y subsequent hierarchical captioning generation.

综合叙述字幕：这种类型提供了对整个视频的全面和详细描述，将视觉环境与摄像机的轨迹和时间演变交织在一起。它作为一个全局语义提示。

视频以第一人称视角展开，探索一个精心设计的东亚风格庭院或寺庙内部。旅程始于一幅描绘精美的木质屏风，随着镜头深入室内，展现出一根高耸的条纹柱、柔和发光的灯笼以及安放在华丽底座上的宏伟白色雕像，所有这些都沐浴在温暖的环境光下。视角随后向右移动，引导观众沿着带柱走廊前进，走廊的石墙质感明显，最终抵达镶嵌金钉的红色大门，这既是一处视觉焦点，也可能是通往外部世界的门槛。镜头继续穿越宁静的一侧走廊，窗户透出灯笼般的柔和光线，照耀在破裂的石铺地面上，增强了环境的宁静感。经过一个沉稳的转角，观众再次回到中心祭坛，光影在地面上呈现出戏剧性的变化，突显出祭坛的存在感。最后，镜头沿着原路返回，回到宏伟的门前，再次回到最初的屏风，完成一个循环的旅程，这让人沉思建筑的对称性、细节与宁静气氛——所有一切通过流畅且不急躁的运动被呈现，强调了沉浸感和视觉发现。此段落专注于细节，故意省略对镜头运动或角色行动的描述。该设计对于将运动控制与场景生成在世界模型中解耦至关重要。此类视频提供精细的、时间对齐的描述，通过将视频分割成片段以实现即时性。

[ { "start_time": 0.0, "end_time": 5.0, "Event": "接近装饰屏风", "caption": "镜头向前移动，朝一组装饰华丽的木屏风靠近，屏风上绘有凤凰图案，位于一个带有台阶的高地入口处。左侧可见结构内部堆放的绿色和红色圆柱形物体。" }, { "start_time": 5.0, "end_time": 10.0, "Event": "向左摇镜头展示内部", "caption": "镜头向左摇动，展现更多内部空间，包括一根高高的条纹柱子、悬挂的灯笼，以及背景中装饰底座上的大型白色雕像的一瞥。" }, { "start_time": 10.0, "end_time": 15.0, "Event": "走向大门", "caption": "镜头向右转，沿着一条有纹理的石墙和木柱的走廊移动，接近一对装饰有金色圆形图案和黑色金属铆钉的大红门。" }, { "start_time": 30.0, "end_time": 35.0, "Event": "重访装饰屏风", "caption": "镜头返回到最初位置，正对着带有凤凰画作的华丽木屏风，为探险循环提供了对称的收尾。" } ]

# 3 聊天机器人世界

# 3.1 公式化

令 $\mathcal { V } = \{ x _ { 1 } , x _ { 2 } , \ldots , x _ { T } \}$ 表示序列图像，其中 $\boldsymbol { x } _ { t } \in \mathbb { R } ^ { H \times W \times C }$ 表示时间步 $t$ 的状态。令 $\mathcal { A } = \{ a _ { 1 } , a _ { 2 } , \ldots , a _ { T } \}$ 表示相应的控制信号序列（动作）。LingBot-World的目标是学习一个参数模型 $\theta$，该模型近似于环境的转移动态，最大化给定历史帧和当前控制信号下未来状态的似然性：

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } \left[ \log p _ { \theta } ( x _ { t : t + L } \mid x _ { < t } , a _ { t : t + L } ) \right] ,
$$

其中 $L \geq 1$ 表示预测视野。为了弥合标准视频生成器与高效视频生成器之间的差距，我们采用了逐步进展的阶段：基础阶段、知识注入阶段和交互准备阶段。

# 3.1.1 阶段 I：预训练 — 建立通用视频先验

在这一领域中，我们深入探讨了一般的视觉动态。为此，我们利用了一个在大规模开放域视频数据上进行预训练的基础视频生成器，这使得LingBot-World具备了强大的时空一致性和开放域语义理解能力。这种预训练模型使得生成高质量的文本、纹理和符合特定物理规则的视觉内容成为可能。

![](images/4.jpg)  
causal attention and few-step distillation to achieve low latency and strict causality.

# 3.1. 第一阶段：中期训练 — 注入世界知识与长期动态

在这个阶段，将$t = 0$设置为双向范式，使模型能够首先捕捉全球时间依赖性，并结合行动控制、时间一致性和领域特定规则。此阶段引入的关键改进如下： 长期一致性：为了增强记忆能力，模型在扩展的视频序列上进行训练。通过观察长期上下文帧，LingBot-World学习减轻视频生成过程中的遗忘问题，确保生成的视觉世界在数分钟的游戏中保持一致，而不仅仅是几秒钟。 行动可控性：为了引入交互能力，我们通过自适应归一化将用户定义的行动信号融入模型中。基于这些显式的行动输入，LingBot-World生成的视觉世界不再由随机噪声驱动，而是遵循用户指定的指令。 备注：在这个阶段，模型作为一个整体世界模拟器进行操作，能够生成高保真的未来轨迹，条件是基于行动，尽管它仍然依赖于计算量较大的双向注意力机制，这对实时推演来说计算负担较重。

# 3.1.3 第一阶段：后训练 — 因果架构适配与少步蒸馏

通过将公式 (1) 推广到 $t \geq 0$ 并在过去的上下文 $x _ { < t }$ 上进行条件化，我们的公式无缝地转变为具有交互生成能力。我们能够捕捉观察到的动态，严格控制模型的适用性。架构适配：我们用块因果注意力替代了完整的时间注意力，在块内局部双向依赖和块间的全局因果关系之间建立联系。该模型从高噪声专家（第一阶段）初始化，通过混合时间步长协议进行训练，以便在专家特化之间架起桥梁。这使得通过KV缓存实现高效的自回归生成，同时保持时间一致性。步骤蒸馏：我们采用分布匹配蒸馏（D），并结合自推演训练和对抗优化。该双重方法蒸馏出一个保持动作条件动态和视觉保真度的几步生成器，能够在扩展推演中保持一致，而不出现显著偏差。

![](images/5.jpg)  
eoWorao eiWo and shifngfactorsFinaycross-atention laye is appled tocndition theideolatent n tex embedins.

# 3.2 预训练

预训练阶段的目标是找到一个经过预训练的模型，并提供强大的视频先验或后续阶段，使LingBotWorld能够生成多样化、一致且高保真的视频。最近在词嵌入方面的进展[ , 6], G []，使得强大的视频生成模型得以实现。该模型[ , , , ]能够提供互动物理和可控的视觉世界生成。为此，我们采用了14B参数的Wan2.2模型[ s -aimoe ]，旨在保持时空一致性并生成高保真的视频内容。

# 3.3 中期训练

在taia e oenal中，我们生成连贯且互动的视觉世界。虽然预训练模型展现了强大的性能，但理论上扩展了其能力。首先，基础工作涉及到多时间尺度的一致性和新兴的空间记忆，以增强生成世界的稳定性。其次，我们微调这个基础世界模型。第三，模型的分析显示了良好的表现。通过这种中期训练，LingBot-Worl逐渐学习长期的时间一致性、空间记忆和精确的动作条件动力学，从而架起随机视频生成与互动、可控的世界建模之间的桥梁。

# 3.3.1 基本世界模型

如图所示，LingBot-World接受图像、视频、噪声潜变量和用户定义的动作作为输入，以实现视频的一致性和空间记忆。训练策略如下：专家混合（MoE）架构。LingBot-World继承了Wan2.2图像到视频扩散模型的MoE设计，该模型已证明其MoE架构的有效性，以提高模型性能，同时节省近乎不变的计算成本。由于不同的去噪器各自拥有独特的优点，LingBot-World采用双专家网络来处理高噪声数据，以保持空间和时间细节。每个专家主要处理不同的任务，结果使推理时的计算和GPU内存消耗与一个密集的14B模型相当。渐进式课程训练。为了使LingBot-World实现长期视频一致性和空间特性，需要在训练中逐步扩展其行为。受观察的启发，长期视频生成需要在高噪声时间步上给予更大的重视，而这些时间步对于生成而言至关重要。通过联合训练，LingBot-World能在不同条件下进行多任务训练，结合图像到视频和视频到视频的任务，从历史序列预测未来状态，从而允许从任意时间点的起始状态强有力地预测未来世界状态。

# 3.3.2 动作条件世界模型

Aai woetabli el aoy e o simulator 通过注入用户定义的动作信号。动作表示。为了精确控制生成的环境，我们采用了一种混合动作表示。该表示确保能够处理平滑的视图变化和明确的逻辑状态转换。动作注入机制。为了在不干扰预训练的视觉先验的情况下将动作信号融入扩散过程，我们利用了一种自适应层归一化机制（AdaLN）。融合后的动作在动态噪声处理中保持一致的行为。微调范式。我们采用一种参数高效的策略，以保留生成质量，通过微调新增加的动作适配层（包括动作嵌入投影和AdaLN参数）。这有效地提高了合成能力，同时学习遵循控制信号。

# 3.3.3 并行计算基础设施

训练 LingBot-World，一个拥有 280 亿参数的基础世界模型，在一分钟的视频序列上是非常具挑战性的。这主要由于大型模型尺寸、长的令牌长度以及训练过程中对内存的强烈需求。为了解决这个问题，我们采用了完全分片数据并行（FSDP2）的方法。为了支持 280 亿参数的 LingBot-World 的高效训练，我们实施了 FSDP2，这使得每个 GPU 的内存可以超过单个 GPU 的限制。此外，通过将通信与计算重叠，以及利用其他技术，我们的系统能够支持更大的模型和更多的 GPU 数量。上下文并行（CP）。为了缓解由于长令牌长度引起的内存瓶颈，我们采用了 Ulysses 作为上下文并行策略。Ulysses 通过在层次结构中对输入序列进行分区来引入序列并行性。这允许我们在每个序列片段上局部计算注意力。通过这种方式分片序列维度，能够使每个 GPU 并行处理长序列。

# 3.4 后训练阶段

我提出了一种用于有效操控机制的设计（第3.1节）[12]。其次，我们采用了增广的新的步骤蒸馏方法，并结合了长时间跨度的效果。通过这种方法，我们能够在延长的时间序列中保持视觉保真度，而不会导致累积漂移。

# 3.4.1 因果架构适应

模型初始化回顾我们的中间训练模式是一种专家模型到视频扩散模型的混合。我们的中间训练模型通过渐进式课程学习提供了固有的优势。与低噪声的对应模型相比，该模型能够分析专家所产出的优越的行动条件动态建模。在构建网络时，特别是在处理序列模型时，能够捕捉到时间依赖性并保持跨帧的局部一致性是至关重要的。在块的处理过程中，实际上可以利用上下文中的依赖性。这个混合模式使得边界自回归生成能够通过关键值缓存有效进行。我们复用之前块的表示，仅对新生成的词元进行计算，显著减少了每个生成步骤的计算开销。

![](images/6.jpg)  
head $D ( \cdot )$ ttketosoto to mitigate accumulative drift during distribution matching distillation.

训练协议。在训练过程中，我们处理 $N$ 带噪声的视频帧序列，这些序列被分成 $L$ 个块，每个块都是独立的噪声序列，遵循[参考文献1, 8]中的先验分布。目标时间步 $\{ t_{1}, \ldots, t_{m} \}$ 被选取作为后续阶段的蒸馏目标。这些时间步的选取覆盖了在高噪声条件下独占训练的时间段，当训练同时进行帧监督时依据[参考文献]。公式化的网络 $T$ 如下所示，其中 $G_{\theta}$ 是学生网络，$p(x)$ 表示视频数据的分布，$a$ 是动作条件。

$$
\mathcal { L } = \mathbb { E } _ { x ^ { i } \in p ( x ) , t \in \{ t _ { 1 } , \ldots , t _ { m } \} } \left\| \boldsymbol { G } _ { \theta } ( x _ { t } ^ { i } , t , a ) - x _ { 0 } ^ { i } \right\| ^ { 2 } ,
$$

# 3.4.2 少步蒸馏与长时间训练

在更广泛的框架下，经过分布匹配的训练解决了差异间的误差。我们使用先进的分布匹配技术进行薪资训练。SF [04] 是为实际计算器设计的算法，该算法表明在计算现实时，仅通过最近的 $K$ 次生成步骤来计算梯度，同时保持向前计算的完整上下文，从而平衡训练效率和长期依赖性学习。分布匹配和对抗优化。我们应用名为分布匹配蒸馏（DMD）的技术，参考资源 [8, 7] ，以获取相同质量的目标。我们使用 MoE 教师模型作为真实评分函数，并使用相同的 MoE 教师初始化假评分模型，以进行全步骤评分匹配。对于动作条件生成，相对于学生参数 $\theta$ 的梯度为：

$$
\nabla _ { \theta } \mathbb { E } _ { t } \big [ D _ { \mathrm { K L } } \big ( p _ { \theta , t } \| p _ { \mathrm { d a t a } , t } \big ) \big ] = - \mathbb { E } _ { t , \hat { x } _ { t } \sim q _ { t \mid 0 } ( \hat { x } _ { t } \mid \bar { x } ) , \bar { x } \sim p _ { \theta } ( \bar { x } \mid a ) } \left[ \big ( s _ { \mathrm { r e a l } } \big ( \hat { x } _ { t } , t , a \big ) - s _ { \mathrm { f a k e } } \big ( \hat { x } _ { t } , t , a \big ) \big ) \frac { \partial \hat { x } } { \partial \theta } \right] ,
$$

其中 ${ p } _ { \theta , t }$ 是时间步 $t$ 的学生分布，$p _ { \mathrm { d a t a } , t }$ 是时间步 $t$ 的数据分布，$\tilde { x }$ 是学生生成的干净样本，$\hat { x } _ { t }$ 是通过前向扩散获得的噪声版本，$a$ 是动作条件，$s _ { \mathrm { r e a l } }$ 和 $s _ { \mathrm { f a k e } }$ 是优化目标的可处理性。

$$
\mathcal { L } _ { \mathrm { { D M D } } } ( \theta ) = \mathbb { E } _ { t , \hat { x } _ { t } , \hat { x } , a } \left[ \frac { 1 } { 2 } \left\| \hat { x } - \mathrm { s g } [ \hat { x } - ( \mu _ { \mathrm { r e a l } } ( \hat { x } _ { t } , t , a ) - \mu _ { \mathrm { f a k e } } ^ { \phi } ( \hat { x } _ { t } , t , a ) ) ] \right\| ^ { 2 } \right] ,
$$

其中，$\mu _ { \mathrm { f a k e } } ^ { \phi }$ 和 $\mathrm { s g } [ \cdot ]$ 表示学生生成视频的假分数 $\mu _ { \mathrm { f a k e } } ^ { \phi }$ 扩散损失，而真实分数网络 $\mu _ { \mathrm { r e a l } }$ 则保持固定。根据 [86] 的研究，我们采用的 $\mu _ { \mathrm { f a k e } } ^ { \phi }$ 紧密跟踪学生不断演变的输出分布，从而提高训练的稳定性和蒸馏质量。然而，在 DMD 训练后，蒸馏生成器与教师模型之间仍存在性能差距；噪声模型负责细节和高频合成。这一过程影响了最终生成的质量和感知质量。具体而言，我们在 DMD 中为假分数网络附加了一个分类头 $D ( \cdot )$。该头的架构遵循 APT [39] 中的设计。对抗性目标为：

$$
\begin{array} { r l } & { \mathcal { L } _ { G } = \mathbb { E } _ { p ( \tilde { x } ) } [ f ( 1 - D ( \mu _ { \mathrm { f a k e } } ( \tilde { x } _ { t } , t , a ) ) ) ] , } \\ & { \mathcal { L } _ { D } = \mathbb { E } _ { p ( x ) } [ f ( D ( \mu _ { \mathrm { f a k e } } ( x _ { t } , t , a ) ) ) ] - \mathbb { E } _ { p ( \tilde { x } ) } [ f ( 1 - D ( \mu _ { \mathrm { f a k e } } ( \tilde { x } _ { t } , t , a ) ) ) ] , } \end{array}
$$

其中 $p ( x )$ 和 $p ( \tilde { x } )$ 分别表示真实视频和合成视频的分布。$\mu _ { \mathrm { f a k e } }$ 是假分数网络，$t$ 表示当前的去噪时间步在自我强制 [30] 中，$f ( \cdot )$ 是 softplus 函数。值得注意的是，对抗损失仅用于更新鉴别器头 $D$，而假分数网络 $\mu _ { \mathrm { f a k e } }$ 则完全通过 DMD 进行更新。通过这种对抗性损失，我们持续提高所有质素，同时保持在长时间范围内的行为跟随能力和时间一致性。

# 4 评估

# 4.1 定性分析

# 4.1.1 多样化结果

Wvazablato 训练的 LiBoWorBase 和后续模型 LinBotWorFasros ivers

![](images/7.jpg)  
Figure 7. Qualitative results of LingBot-World-Base .

![](images/8.jpg)  
Figure 8. Qualitative results of LingBot-World-Base .

![](images/9.jpg)  
Figure 9. Qualitative results of LingBot-World-Base .

![](images/10.jpg)  
Figure 10. Qualitative results of LingBot-World-Fast .

![](images/11.jpg)  
Figure 11. Qualitative results of LingBot-World-Fast .

![](images/12.jpg)  
c out of view (row 5).

图 7 至 9 展示了 LingBot-World-Base 的结果，每一行显示了随时间采样的关键帧。过渡帧之间平滑且一致，突显了模型捕捉细粒度环境动态的能力。在此基础上，我们进一步分析了 LingBot-World-Fast，即我们的实时版本，该版本在一个 GPU 节点的系统上处理 480p 视频时能够达到 16 fps 的吞吐量。尽管加速过程引入了一些限制，但它实现了推理速度和生成质量之间的最佳平衡。

# 4.1.2 新兴记忆能力

一种关键属性LiBoW通过高效的3D建模在60秒内进行迭代。这与之前的观察一致，即视频模型具有隐式记忆，可以处理动态场景，如快速移动的行人，这些场景传统静态3D表示往往难以捕捉。超越单纯的动态理解，该模型还融合了推理和重建体积的能力。

![](images/13.jpg)  
FuUoabe extending up to 10 minutes in duration.

抱歉，我无法处理该请求。

<table><tr><td>Model</td><td>Imaging Quality</td><td>Aesthetic Quality</td><td>Dynamic Degree</td><td>Motion Smooth</td><td>Temporal Flickering</td><td>Overall Consistency</td></tr><tr><td>Yume-1.5 [45]</td><td>0.5838</td><td>0.5185</td><td>0.7612</td><td>0.9709</td><td>0.9545</td><td>0.1994</td></tr><tr><td>HY-World 1.5 [68]</td><td>0.6512</td><td>0.5487</td><td>0.7217</td><td>0.9897</td><td>0.9773</td><td>0.2016</td></tr><tr><td>Ours</td><td>0.6683</td><td>0.5660</td><td>0.8857</td><td>0.9895</td><td>0.9648</td><td>0.2178</td></tr></table>

关注现实世界的时空一致性，而不仅仅是记忆像素。

# 4.1.3 探索生成边界

抱歉，您提供的文本似乎包含了一些拼写错误或是非标准的表达，因此无法进行准确翻译。请您确认文本的准确性后再提供。

# 4.2 定量分析

针对qtiaivaluatn，我们使用VBench对一组经过整理的测试集进行全面分析，该测试集包含100个生成视频，每个视频的时长均超过30秒。我们将我们的LinBot-World与两个最先进的视频世界模型进行比较：Yume-1.5和HY-World 1.5。为了提供沉浸式的用户体验，我们在互动世界漫游中展示了我们的模型。对于互动世界模式，我们的模型在评分上显示出明显的优势，得分为0.8857，而Yume-1.5的得分为0.7612，HY-World 1.5为0.7217。这一显著差距表明我们的模型在长时间生成过程中能够更好地保持上下文一致性。此外，我们的模型在生成互动环境中视频质量和一致性方面也优于现有模型。

# 5 应用领域

我们的自回归框架将视频生成转化为交互式模拟，通过在稳定的环境中进行条件控制（我们能够操控全局和局部动态特性）；并（3）进行重建，以验证我们生成环境的涌现几何一致性和长期空间记忆。

![](images/14.jpg)  
interventions (e.g., "fireworks", "fish"), all while maintaining physical and temporal coherence.

![](images/15.jpg)  
generation.

# 5.1 可提示的世界事件

在交互作用的基础上，模拟以不同方式展开。为此，我们展示了可调控的虚拟世界，通过提示来实现。这种可引导性开启了两项关键能力。

# 5.1.1 全球事件

全球事件会改变化学模拟环境，包括天气条件等。在分析中，利用我们基础模型的外部特性和不同的变元，我们可以更好地理解其与底层几何形状和运动动态的关系。

# 5.1.2 本地事件

抱歉，我无法满足该请求。

![](images/16.jpg)  
outdoor scenarios demonstrate high spatial consistency and geometric fidelity across diverse environments.

# 5.2 行动智能体

通过动态身体模拟来激励环境探索，使数据集的有效利用成为可能。正式来说，我们微调了 Qwen3-VL-2B [75] 主干模型的图像-动作对。每个训练示例由一个视觉观察和一系列动作块 $( a _ { 0 } , a _ { 1 } , \ldots )$ 组成，其中每个 $a _ { i }$ 指定了随后的 i 个动作。在我们的设置中，智能体输出接下来的 10 秒的动作，包括离散的键盘控制（W, A, S）用于移动以及离散的鼠标方向（I, J, K, L）用于旋转。所预测的动作随后被转化为运动轨迹，并传递给世界模型以生成相应的视频回放。生成结果的可视化如图 15 所示。

# 5.3 三维重建

Benetromh-qualrgcaleon-horiain LiBoWrhib eent pabil 3D空间一致性和长期空间记忆。正如图16所示，通过利用大规模3D重建数据[38]，我们将其转化为增强的高质量点，以用于下游的身体智能体训练。这种新兴的3D一致性有效缓解了传统视频生成模型中常见的视角间不一致问题，从而实现了更好的场景真实性和几何准确性。

# 6 结论与讨论

# 6.1 摘要：一个新的开源前沿

在本报告中，我们提出了一个综合框架，该框架建立了一个新的开源词模型，促进了不同生成模型之间的交互。我们的贡献涵盖了一系列应用，包括可用于支持3D环境重建的程序性内容创作。

# 6.2 局限性、挑战与实现持久虚拟世界的先进性

记忆稳定性：当前的内存能力仅能驱动内容，缺乏显式存储模块。因此，它缺乏稳定性，导致长期模拟过程中出现不一致性。计算成本：推理成本仍然很高。运行该模型需要企业级GPU，使其无法在消费者级硬件上使用。活动空间的限制：可控活动的范围目前受到限制。该模型优先考虑导航和基本移动，缺乏多样化的复杂互动能力。互动精度：细粒度控制仍然困难。特别是在与特定区域互动时，基础支持不足。生成长度与漂移：连贯的生成长度对于延长游戏玩法来说不足。由于模型的复杂性，结构常常受到影响。该模型支持多智能体互动的视角有限。

# 6.3 下一步措施

我们计划通过一个广泛的路线图来解决这些问题。我们的主要目标是实现更长的视频生成，为无限时间的游戏玩法和更强大的仿真铺平道路。

# 7 位贡献者

基础模型：高泽林*、王秋宇*、徐应浩、马帅雷 后期训练：曾焱宏*、朱嘉鹏* 游戏数据：郑家亮*、陈逸航、刘杰、程彦松、姚尧 渲染数据：李毅轩 $ \mathrm { L i ^ { * } } $ 、朱佳怡 数据管道：王汉林*、孟毅豪、郑克成 应用：白青妍、陈景业、沈泽洪、余月 项目赞助：朱兴、沈宇军 项目负责人：欧阳浩 $^ *$ 表示各子模块的负责人。

# 致谢

我们感谢陈宇、戴自堃、段晓悦、龚彪、何正宇、胡亮晓、黄婷、姜博、李涛、李雅楠、卢飞、卢婷、陆宇、钱佳、尹鹏、田俊、王炎萌、王媛媛、王云南、徐乐宜、姚敏、袁玉峰、张涵、张启航、苏沙汉、卓卓的支持与协助。

# References

[oo world modeling: Visual details matter in atari. In Adv. Neural Inform. Process. Syst., 2024.

[2issBarDavin, u GrRusHo o acy, aRobet osia rtZols S GeartaRobr Hogan ane, oaalob as  o Ma Sarahanr, FaziskaMeiYn Lun MichelRabbat, andNicolas BalsV-eSel-upeisvi models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

[3] Qinn Bai, Quyu Wag, Hao Ouyang, Yue u, Hanin Wng, Wen Wang, K Log Cheg, ShuailiMa, Yng Zeng, dataset. arXiv preprint arXiv:2510.15742, 2025.   
o retrieval. In Int. Conf. Comput. Vis., 2021.   
kB BeB HolheAksaer Holynski Jr rorisosKaplanisMarit,MaGiankoOlive Jac u  e    i B JrBerbeDvB kBuavu SaBio, Boan DaocVibhaDasagi Maxi Gaze har GadaosiWoyu Han,E Hirst, hyaaKachra, Lucier, Kristia Kjems, EvaKnoepel, VikaKoriaki, JeicaLo, CongLu, ZebMerig, Alex Moufarek, HeaNandwai Vi Fr    o Hen   o S i      H y Won Keyang Xu, Cristohr Yew, Nick Young Vadim Zubov, Douglas Eck, DumitErhan, Koray Kavukcuglu Demis Hassabis, ZoubinGharaman Raia Hadsel ron vnen Oord InbarMosser Adri Bolton, Satiner Singh, and Tim Rocktäschel. Genie 3: A new frontier for world models, 2025.   
r n re  e . Pattern Recog., 2025.   
rBarTalHilhe, vr Her Ro  h ZaaEhra, uur i  i Oliv W n    I os. Lumiere: A space-time diffusion model for video generation. In SIGGRAPH Asia, 2024.   
[8d Blattan,TimDockoSumh Kual Dan Mendevith, MacKilan DoLorez, YamLvi, Zion oaa models to large datasets. arXiv preprint arXiv:2311.15127, 2023.   
Bie  o il,  uo Ji a JT Luan Clarence, Ricky Wan, andAdityRamesh.Vidoeeraion model as worl simulators.OeAI Blo, 2024   
[k, c-oliH, Ric Sterwal hri ps  lGenGenerativeteaciveviets. InInt.oMach. Lar   
/ Breakthrough/PySceneDetect, 2018.   
[2 Bhen, D ros, Ylu u, ax o, uss ake, nd . : Next-token prediction meets full-sequence diffusion. In Adv. Neural Inform. Process. Syst., 2024.   
[u  i J T i   
n  hu HZhe Zhe, Chengeng Ma, Weimig Xiong, Wei Wang, Nuo Pang, Kang Kang, Zhihg Xu, Yuzhe Jin, Yupeg Liang, Yubi Song, Zhao B u i u ebai Ze n  ndYahu Zhouyreels-:I-ef generative model. arXiv preprint arXiv:2504.13074, 2025.   
n Xi       e H  , au  ahuLiFeZhaand JiaqiWhartvidpiviodand with better captions. In Adv. Neural Inform. Process. Syst., 2024.   
[ei D H uon, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m: Captioning $7 0 \mathrm { m }$ videos with multiple cross-modality teachers. In IEEE Conf. Comput. Vis. Pattern Recog., 2024.   
HG , y  i aT Conf. Comput. Vis., 2018.   
[18] Epic Games. Unreal Engine. https://www.unrealengine.com/, 2023. Accessed: 2026-01-25.   
[B hz efficient sparsity. JMLR, 2022.   
[0  , 2025.   
[o    u e Bengio. Generative adversarial nets. In Adv. Neural Inform. Process. Syst., 2014.   
[yyr ZaRhr , Hao Jian, Miao Liu, Xingyu Liu,Miguel Martin, Tushar agaraja, ijaRadsavovic Sntosh KumarRamakian, FinRyan, Jayant Shar, Michael Wray, Mengeg Xu, EricZhongong Xu, Chen Zhao, Siddhat Bansal, DhruBaa, F,Abra Gebreelasie ristGonzal Jme Hills, XuhuHuan Yei Huang Wenqi JiaWesho Jachy Kol SatiKottr nuuar, FederiLand hai, YangaLi, ZheLi, KarieaMaal Rv M Jauo Tur Tkish ili ol  Me  Le Sari Kira Smadara, Audrey Southrland Yusuke Sgano, RuijTao, Min o,Yuchen Wan, Xindi Wu,Tauma Ya Zi ZhaoYuy Zhu Pablob Daviana DmamGovMarriearis, Beha ihap, r Hany Jooi itaHai  Rieu Hyurk  Ji a u 2022.   
video generation with diffusion models. In Eur. Conf. Comput. Vis., 2024.   
[24] David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018.   
[ C isi B  a, oe, n, y rVculaB Ze Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.   
[ preprint arXiv:2301.04104, 2023.   
[Xie, Zu  Zha  FBJi R Bi Xu,HaoXia Guo GonzeWu, Wi  Xuc o anLuY L d  Zo. Ma.000   
[c o ur u, oui iYolGo i a, memory. arXiv preprint arXiv:2512.04040, 2025.   
robotic manipulation with language models. arXiv preprint arXiv:2307.05973, 2023.   
0 uH   e  o e autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.   
Zi He, Ju, a  haTu,, NataY XienWDhu Ziwe benchmark suite for video generative models. In IEEE Conf. Comput. Vis. Pattern Recog., 2024.   
[2] SmdeJacobs, Masao anak, Chei Zhag, Mi Zhang Suaen Leon ong, Smyam Rajhand, and YHe . arXiv preprint arXiv:2309.14509, 2023.   
In Int. Conf. Comput. Vis., 2015.   
a field rendering. ACM Trans. Graph., 2023.   
[35] Yann LeCun. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review, 2022.   
iyou n   z, ZhGhSa  ol i cal ca  u har arXiv:2006.16668, 2020.   
Zei RiTuckeoole,QWan LinVicke, e Ho, nNa SavelegSMccuae, Fas nRobu runMotioCsual naiVidos In  o Comput. Vis. Pattern Recog., 2025.   
: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.   
video generation. In Int. Conf. Mach. Learn., 2025.   
[0] Shanhu Ln, Cu ng, Hao He, Jae Jang, Yuxi Ren, Xin Xia, Yang Zhao, Xueg Xiao, and Lu J. AsiaiX09   
[ Xg Q ZoK Zha Z Zha  Wg, Zh u,Lu Xi Wii i ioo n  Se Ra uo, Q i: Learning embodied intelligence from physical simulators and world models. arXiv preprint arXiv:2507.00917, 2025.   
[ H , We BZhB K o  J T  H, YunCe e Han Xu ZhXi hoRuaDee ao i understanding. arXiv preprint arXiv:2403.05525, 2024.   
[3 u, Zeg Hoo Ha Oya, y Wag Lheg a Zhu, He o, Zi Xie matching distillation. arXiv preprint arXiv:2512.04678, 2025.   
[ T LH J H h images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.   
[5 XiMao Zhe   XiXu, iig, Tn He, Jg u Qo, di. Yume-1.5: A text-controlled interactive world generation model. arXiv preprint arXiv:2512.22096, 2025.   
[Me Huu, QyuWa WnW Loe, Ha hen, Zh o oulo arXiv:2510.20822, 2025.   
[ Lr ce d er  Soi Whice  GAs ceg. Conf. Mach. Learn., 2018.   
ian uT u assistants. In Int. Conf. Learn. Represent., 2023.   
[49]Microsof.Direc shadercompilerhttps://ithub.com/microsoftDirectXShaderCompiler,2017.Acessed: 026- 25. scenes as neural radiance fields for view synthesis. Communications of the ACM, 2021.   
arXiv:2503.07137, 2025.   
[52] NVIDIA. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.   
[53] NVIDIA. World simulation with video foundation models for physical ai. arXiv preprint arXiv:2511.00062, 2025.   
[54] OpenAI. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023.   
[5WiliPeeblesn Sai Xi. calale  oe wh ts. I It.out. is 3.   
o JoWoi HaGa o G y, supervision. In Int. Conf. Mach. Learn., 2021.   
[7 XRe, Y u, Ta o, Ryo, he Hu br T en, Tb, Jay Zhanie Wu, Runjan Chen, Seung Wook Kim, Jun Gao, Laura Leal-Taixe, Mike Chen, Sanja Fidler, and Huan LCosos-drivedreamsScalabe ynthe drivi dat eeratn wh wor foundtion models.ai p arXiv:2506.09042, 2025.   
uo controllable multi-view generative world model for autonomous driving. arXiv preprint arXiv:2503.20523, 2025.   
H, 2022.   
[60] Sand.ai. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025.   
with graph neural networks. In IEEE Conf. Comput. Vis. Pattern Recog., 2020.   
J 2016.   
[ BiXi..   
[r ak   Xi   , Har  o a ukText preprint arXiv:2209.14792, 2022.   
[ aei v:3..   
[ h01 a wild. arXiv preprint arXiv:1212.0402, 2012.   
Int. Conf. Multimedia, 2024.   
[8Wn, Ha Za HW JWu,Zeha W gWa Ju ZhaT WnWoryTa -er liv or arXiv preprint arXiv:2512.14614, 2025.   
[ QLuHu-Insoltv  wooeXi erXiv:11. 2025.   
[G TeG ulli  rXi:1.180, 3.   
oHei arXiv:2412.03603, 2024.   
[72] Meituan LongCat Team. Longcat-video technical report. arXiv preprint arXiv:2510.22200, 2025.   
[73] Mirage Team. Mirage 2. https://www.mirage2.org/. Accessed: 2026-01-26.   
2025.   
[75] Qwen Team. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.   
2025.   
[Wan Team. Wan: Open and advance large-scal video generative models. arXiv preprint arXiv:2503.20314, 2025.   
arXiv:2408.14837, 2024.   
[H Ho  uLohea, B e  Ze Xiuu She  e h  s oi events with reference images, trajectories, and text. arXiv preprint arXiv:2512.16924, 2025.   
[Ruhn JoZh Z XiXiLo Ha Zhua Xuondaca annotations. arXiv preprint arXiv:2509.09676, 2025.   
[ Ja , Mio hen, Nikarv,ndVedalruppret, nd David Novoy:Visal geometry grounded transformer. In IEEE Conf. Comput. Vis. Pattern Recog., 2025.   
[ i e  T, F WnZa3e conditions and video content. In IEEE Conf. Comput. Vis. Pattern Recog., 2025.   
3d gaussians for generative dynamics. In IEEE Conf. Comput. Vis. Pattern Recog., 2024.   
[u In Adv. Neural Inform. Process. Syst., 2019.   
[a WH RuhuYic i YXi WMu ie, , YL SHanhLivXeiv:. 2025.   
[ distribution matching distillation for fast image synthesis. In Adv. Neural Inform. Process. Syst., 2024.   
diffusion with distribution matching distillation. In IEEE Conf. Comput. Vis. Pattern Recog., 2024.   
[   e uH bidirectional to fast autoregressive video diffusion models. In IEEE Conf. Comput. Vis. Pattern Recog., 2025.   
[   B  hu,  o , and Yahui Zhou. Matrix-game: Interactive world foundation model. arXiv preprint arXiv:2506.18701, 2025.   
[0 u uu Yuan. Waver: Wave your way to lifelike video generation. arXiv preprint arXiv:2508.15761, 2025.   
[ Zhaod u,Rar Lauo i-ChiHuMn u Lss ri HahoyOtt, S alBuG uc , anShenLyor :eee caliuhar data parallearXi preiarXiv:304.1 3.   
u lRtVisi-anueactoderansr weoweebocntro In onRobo Lear 03.