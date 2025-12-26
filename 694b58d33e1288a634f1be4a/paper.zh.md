# GraspVLA：一个在十亿级合成动作数据上预训练的抓取基础模型

邓胜良\*, 1,3 闫$\mathbf { Y a n ^ { * , 1 , 2 } }$松林 韦1,2 海鑫 $\mathbf { M } \mathbf { a } ^ { 1 }$宇欣 杨1 佳怡 陈1,2 智奇 张1,2 陶宇 杨2 旭恒 张2 文浩 张2 文明 崔3 志正 张1,4 贺 王†,1,2,4

![](images/1.jpg)  

Figure 1: GraspVLA is a grasping foundation model pre-trained exclusively on billion-scale synthetic action data and co-trained with Internet semantics data. It exhibits direct sim-to-real transfer and strong zero-shot generalization across diverse aspects, as well as few-shot adaptability to specialized scenarios and human preferences.

摘要：具身基础模型因其零-shot 泛化能力、可扩展性及通过少量样本后训练适应新任务的能力而受到越来越多的关注。然而，现有模型在很大程度上依赖于现实世界的数据，而收集这些数据成本高且劳动密集。合成数据提供了一种具有成本效益的替代方案，但其潜力仍然未被充分挖掘。为此，我们探索了使用大规模合成动作数据完全训练视觉-语言-动作（VLA）模型的可行性。我们整理了 SynGrasp-1B，这是一个在模拟环境中生成的十亿帧机器人抓取数据集，采用了 photorealistic 渲染和广泛的领域随机化。在此基础上，我们提出了 GraspVLA，这是一个在大规模合成动作数据上预训练的 VLA 模型，作为抓取任务的基础模型。GraspVLA 将自回归感知任务和基于流匹配的动作生成集成到一个统一的思考链过程中，使得能够在合成动作数据和互联网语义数据上进行联合训练。这一设计有助于减小模拟与现实之间的差距，并促进已学习动作向更广泛的互联网覆盖对象的迁移，实现抓取任务中的开放词汇泛化。在现实世界和模拟基准上的广泛评估表明，GraspVLA 具备先进的零-shot 泛化能力和对特定人类偏好的少-shot 适应能力。我们将发布 SynGrasp 1B 数据集和预训练权重，以造福社区。我们的项目页面地址为 https://pku-epic.github.io/GraspVLA-web。关键词：视觉-语言-动作、大规模机器人学习、抓取

# 1 引言

自然语言处理（NLP）和计算机视觉（CV）领域随着基础模型的出现经历了范式转变。基于海量互联网数据预训练的大规模模型在未见场景中表现出零样本泛化能力[1, 2, 3]，并能通过少样本适应来对齐人类偏好[4]。受到这一成功的启发，最近在视觉-语言-动作（VLA）模型中引入了针对物理世界行为的基础模型[5, 6, 7, 8]。这些模型处理机器人视觉观察和人类指令，直接生成机器人动作。然而，与视觉和语言模态不同，现有互联网数据集中缺乏动作数据，这需要一种新的数据收集范式。近期研究主要依赖于通过远程操作收集真实世界数据，例如社区驱动的 Open-X-Embodiment（OXE）[9] 和 DROID [10] 数据集。然而，在大规模收集真实世界数据既费力又昂贵，需大量机器人和人类操作员，以及多样的物理设置。相比之下，合成数据提供了更为便捷且低成本的替代方案，但其潜力仍然被低估。为此，我们系统地探索合成数据在训练 VLA 模型中的潜力。作为这方面的第一步，我们专注于抓取这一基本的机器人操作技能。我们首先根据先进的光线追踪渲染[11]和物理模拟[12]，策划了一个十亿帧的抓取数据集 SynGrasp-1B，这标志着全球首个这一规模的数据集。该数据集包含来自240个类别的10,000个独特物体，并涵盖广泛的领域随机化，确保几何和视觉变化的广泛覆盖。为了高效地从该数据集中学习，我们提出了 GraspVLA，这是一个将自回归感知任务和基于流匹配的动作生成整合到一个统一的思维链（CoT）过程中的端到端网络，称为渐进式动作生成（PAG）。PAG 将感知任务（即视觉定位和抓取姿势预测）视为动作生成过程中的中间步骤，形成一个因果推断动作的 CoT 过程。这一设计使得在统一框架中对合成数据和互联网数据进行联合训练成为可能，其中互联网数据用于训练感知任务（部分 CoT 过程），而合成数据用于训练整个 CoT 管道。合成数据为物体交互提供了详细的几何信息，而互联网数据则提供了丰富的物体语义知识。通过利用这些互补的来源，PAG 减小了模拟与真实之间的差距，并促进了学习到的机器人动作向语义多样的互联网覆盖物体转移，从而实现开放词汇的抓取。

通过我们精心策划的十亿规模合成抓取数据集和所提出的PAG机制，GraspVLA实现了直接的模拟到真实的泛化，并展现出了令人印象深刻的零样本性能。据我们所知，这是首个揭示合成数据在训练用于操作的VLA模型中显著潜力的研究。大量在现实世界环境和LIBERO [13]仿真基准上进行的实验展示了该模型在多样变化下的鲁棒性。此外，GraspVLA在缺乏合成动作数据的长尾物体类别上的泛化能力也表现优异，例如充电器、毛巾和泳镜。与传统抓取检测算法的最先进方法AnyGrasp [14]相比，GraspVLA支持自然语言指令，并提供稳健的闭环抓取策略。在常见物体上，GraspVLA的性能相当于AnyGrasp，但在透明物体上的表现大幅领先于AnyGrasp。此外，GraspVLA在特定应用场景中展示了对用户偏好的强大少样本适应能力，这些场景超出了标准抓取行为，例如避免接触饮水杯内表面以保持清洁，以及在密集环境中顺序抓取瓶子。总之，我们的贡献如下：a) 我们引入了一种全新的预训练范式，完全依赖合成动作数据，显著降低了现实世界动作数据获取的负担；b) 我们策划了一个十亿帧的机器人抓取数据集SynGrasp-1B，这是全球首个此规模的数据集；c) 我们提出了渐进式动作生成，旨在联合训练合成动作与互联网数据，将GraspVLA的技能扩展到新物体类别；d) 大规模实验证明了GraspVLA的基础能力，包括强大的零样本泛化能力和高效的少样本适应能力。

# 2 相关工作

视觉-语言-动作（VLA）模型。最近，一些研究探讨通过从大规模演示数据中学习进行端到端VLA训练。RT-2和OpenVLA提议利用预训练的视觉-语言模型（VLM）来挖掘来自互联网数据集的丰富知识。随着预训练VLM的成功，若干研究探索借助额外的动作专家生成高保真度的多模态动作。其他研究则采用在互联网规模的视频数据上进行生成式预训练，以便从人类视频中学习。然而，受限于现实世界机器人数据的规模，现有的VLA模型主要依赖于领域内后训练进行部署。与此同时，$\pi _ { 0 . 5 }$提出通过利用多模态网络数据和交叉体现数据来改善泛化能力，从而实现即开即用的部署。尽管我们的工作也旨在实现零样本部署，但我们采取了不同的方式——专注于在大规模合成数据上进行预训练——并展示出强大的零样本泛化能力。合成数据。随着GPU加速模拟和照片级真实渲染的快速发展，合成数据生成已成为训练机器人模型的一种流行方法。之前的研究开创了使用领域随机化的模拟数据来训练开环抓取模型。最近，几项研究探讨通过随机化物体配置和利用运动规划来生成真实的机器人轨迹，从而在模拟中自动增强人类演示。另一系列研究则利用文本生成图像模型和多视角立体渲染从少量人类演示合成数据，而无需任何物理模拟。尽管这些方法仍依赖人类演示生成增强数据，我们的工作探索通过结合大规模合成数据和预训练的视觉与语言主干进行直接的仿真到真实转移。

抓取。抓取是具身智能体的一项核心技能，近年来得到了积极研究。一些研究通过开放回路抓取检测来解决这一问题，随后使用运动规划器控制末端执行器。这种基于模块化的系统通常存在深度感知差差和缺乏故障恢复行为的问题。另一项研究方向探索基于视觉的端到端闭环抓取系统，采用强化学习或模仿学习的方法。随着视觉-语言基础模型的出现，一些工作旨在通过构建将抓取检测模型与视觉-语言模型相结合的模块化系统，将抓取泛化到开放词汇对象。尽管这些方法在标准抓取任务中取得了令人瞩目的成绩，但在适应特定约束的专业任务时仍面临挑战。

# 3 SynGrasp-1B 数据集生成

训练一个可泛化的基础模型需要一个涵盖多样化物体和环境条件的大规模数据集。我们提议完全基于合成数据进行训练，这样能够以更少的时间和成本提供更大的多样性，而不是依赖于高成本的真实世界人类数据收集。我们现在详细描述合成数据生成管道的核心组件。 物体资产和布局生成。我们利用Objaverse数据集的LVIS子集，仔细过滤掉不适当的类别，如武器，最终得到240个类别和10,680个实例。我们随机缩放这些物体，并将其以各种姿势放置在桌子上，生成多样且物理上合理的场景。更多细节请参见补充材料。 抓取合成和轨迹生成。在给定初始布局的情况下，我们利用先进的模块化系统建立一个专家策略，以生成高质量的轨迹用于抓取和提升目标物体。对于每个物体实例，我们利用抓取合成算法生成稳定的对立抓取。然后，我们使用运动规划算法CuRobo来规划无碰撞的轨迹，以达到开放循环的抓取姿势并提升物体。我们在MuJoCo物理模拟器中验证所有候选轨迹，以确保物体的成功提升。

![](images/2.jpg)  

Figure 2: Data generation pipeline: We first curated over 10,680 object meshes from Objaverse [63] that are suitable for tabletop grasping and randomly selected and placed these objects on the table (left). Next, we used CuRobo to plan grasping trajectories with randomized grasp poses and instructions (middle). Finally, we applied domain randomization to materials (table and robot), lighting, camera views, and backgrounds to simulate and render the trajectories (right).

视觉随机化与渲染。考虑到多样的布局和相应的轨迹，我们使用 Isaac Sim [66] 渲染高质量的 RGB 图像，随机化光照、背景和相机设置，提供高效的照片级真实光线追踪渲染。我们采用多种光源进行广泛的随机化，包括点光源、方向光源和半球光源。图像从两个不同的视点渲染，以提供场景的全面视角，同时围绕预定义中心随机化外部参数。更多细节请参见补充材料。我们进一步强调数据生成管道设计中的两个主要考虑因素：高效数据生成。我们开发了三项关键策略以提高效率。高质量网格通常较大，导致加载时间长和显著的内存使用。我们实现了一种缓存机制，以避免冗余加载，同时确保数据的多样性。其次，我们实现了异步数据写入，使得图像和标签能够并行保存，从而提高数据生成的整体效率。最后，我们采用并行物理仿真和渲染，以进一步提高效率。更多细节请参见补充材料。为模仿学习量身定制的数据。为降低模仿学习的难度，我们引入了两项改进。首先，开放式抓取 [14] 采用两步过程（预抓取定位和抓取执行）以避免碰撞，但这种分段方法会产生停顿。在此类数据上训练的模仿策略往往表现出犹豫 [6, 67]。相反，我们实现了单步运动规划，优先考虑轨迹的平滑性而非规划成功率。其次，我们引入随机初始化的机器人姿态，以改善工作空间的探索和专家演示中的观察多样性，从而增强模型的鲁棒性 [68]。通过这一管道，我们使用 160 个 NVIDIA 409C GPU 在 10 天内生成了十亿帧数据集 SynGrasp-1B。我们在补充材料中提供了数据多样性分析。

# 4 模型

整体架构。GraspVLA集成了视觉-语言模型（VLM）与动作专家[7]，通过渐进式动作生成（PAG）机制连接，如图3所示。VLM接收观察图像和文本指令以实现视觉-语言联合感知。它包括一个可训练的大型语言模型（InternLM2 1.8B [69]）、一个融合了静态DINO-v2 [70]和受OpenVLA [6]启发的SigLIP [71]特征的视觉编码器，以及一个可训练的从视觉空间到语言空间的投影器。我们使用条件流匹配动作专家[72]进行细粒度末端效应器动作生成。我们进一步引入PAG以有效地将从互联网对照数据集中学习到的知识转移到抓取技能上。渐进式动作生成。虽然GraspVLA从我们的SynGrasp-1B数据集中学习可推广的抓取技能，但它受到合成数据集中类别集合的限制。为了将抓取策略扩展到新的类别，一种直接的方法是将其与互联网对照数据集共同训练作为独立任务，并依赖模型隐式推广到从对照数据集中学习到的物体类别。

![](images/3.jpg)  

Figure 3: GraspVLA consists of an autoregressive vision-language backbone and a flow-matching based action expert. It exploits the synergy between Internet grounding data and synthetic action data with a Progressive Action Generation mechanism: the model first predicts 2D bounding boxes of the target object for both synthetic data and web data, and additionally generates grasp pose and chunked actions for synthetic data.

我们将图像定位和抓取姿态预测视为生成动作的中间步骤。具体而言，VLM被训练生成统一格式的二维边界框，适用于互联网定位数据集和合成动作数据集。接着，对于合成数据集，VLM进一步预测机器人基础坐标系中的目标抓取姿态。最后，动作专家根据VLM输入和中间推理词元的键值缓存生成动作片段。为了促进准确的三维感知，最近两步的本体感觉被标记为词元，并在生成抓取姿态之前插入。为了将互联网数据集与SynGrasp-1B的双摄像头设置对齐，输入图像被复制以匹配视图数量，并独立地进行随机调整大小、裁剪、水平翻转和颜色抖动。两个数据集共享相同的文本提示模板，首先生成边界框词元。这种统一的训练策略利用了互联网定位和合成数据集之间的协同作用，类似于广泛研究并被证明为处理大语言模型中高度复杂任务的有效措施的链式推理机制。VLM和动作专家的联合训练。每个批次中，我们随机从互联网数据集（GRIT）和合成动作数据集中抽样。前者仅用于以自回归方式监督VLM的边界框预测。后者则监督边界框、抓取姿态和基于流匹配的动作预测。VLM的损失正式定义为：

$$
\mathcal { L } _ { \mathrm { S 2 } } = - \sum _ { n = 1 } ^ { N _ { \mathrm { b a s e } } } \log P _ { \theta } ( \mathbf { y } _ { \mathrm { b b o x } , n } \mid \mathbf { x } , \mathbf { y } _ { \mathrm { b b o x } , < n } ) - \mathbf { 1 } _ { \mathrm { s y n h e i c } } \cdot \sum _ { n = 1 } ^ { N _ { \mathrm { g a s e } } } \log P _ { \theta } ( \mathbf { y } _ { \mathrm { g r a s p } , n } \mid \mathbf { x } , \mathbf { y } _ { \mathrm { b b o x } } , \mathbf { y } _ { \mathrm { g r a s p } , < n } ) ,
$$

其中 $N _ { \mathrm { b b o x } }$ 和 $N _ { \mathrm { g r a s p } }$ 分别是边界框和抓取姿态令牌序列的长度，$\mathbf { y } _ { \mathsf { b b o x } , n }$ 和 $\mathbf { y } _ { \mathrm { g r a s p } , n }$ 是各自序列中位置 $n$ 的令牌，$\mathbf { x }$ 是输入的图像和文本。动作专家使用流匹配损失对分块的执行器增量动作进行监督：

$$
\mathcal { L } _ { \mathrm { S 1 } } = \| v _ { t } ( \mathbf { A } _ { t } , \mathbf { x } , \mathbf { y } _ { \mathrm { b b o x } } , \mathbf { y } _ { \mathrm { g r a s p } } ) - u _ { t } ( \mathbf { A } _ { t } \mid \mathbf { A } _ { 0 } ) \| ^ { 2 } ,
$$

其中 $t \in [ 0 , 1 ]$ 是流匹配时间步，${ \bf A } _ { t }$ 是时间 $t$ 的加噪动作主干，${ v } _ { t } ( \cdot )$ 是模型预测的流匹配向量场，$u _ { t } ( \mathbf { A } _ { t } \mid \mathbf { A } _ { 0 } )$ 是真实向量场。我们经验性地发现，整体损失的 ${ \mathcal L } _ { \mathrm S 2 }$ 和 ${ \mathcal L } _ { \mathrm { S 1 } }$ 的简单相加能够带来良好的性能。

# 5 实验

我们评估 GraspVLA 以回答以下问题：(1) GraspVLA 在各种推广因素下与现有工作相比如何？(2) GraspVLA 随着数据量的增加如何扩展？(3) 我们的设计选择对 GraspVLA 的性能贡献有多大？(4) GraspVLA 对于特定偏好的少样本后训练支持效果如何？

<table><tr><td rowspan="2"></td><td colspan="6">Synthetic Categories</td><td colspan="6">Web Categories</td></tr><tr><td>basic↑</td><td>light↑</td><td>b.g.↑</td><td>dis.↑</td><td>height↑</td><td>SPL↑</td><td>basic↑</td><td>light↑</td><td>b.g.↑</td><td>dis.↑</td><td>height↑</td><td>SPL↑</td></tr><tr><td>Diffusion Policy [75]</td><td>30.0</td><td>16.6</td><td>16.6</td><td>13.3</td><td>13.3</td><td>12.3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Octo [26]</td><td>16.6</td><td>3.3</td><td>0.0</td><td>0.0</td><td>3.3</td><td>3.2</td><td>0.0</td><td>3.3</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.4</td></tr><tr><td>OpenVLA [6]</td><td>20.0</td><td>13.3</td><td>16.6</td><td>0.0</td><td>13.3</td><td>8.8</td><td>3.3</td><td>6.6</td><td>13.3</td><td>0.0</td><td>6.6</td><td>4.1</td></tr><tr><td>π0(w/π0 pre-train)[7]</td><td>66.6</td><td>63.3</td><td>60.0</td><td>60.0</td><td>56.6</td><td>42.3</td><td>33.3</td><td>36.6</td><td>30.0</td><td>26.6</td><td>26.6</td><td>17.8</td></tr><tr><td>π0(w/o π0 pre-train)[7]</td><td>80.0</td><td>76.6</td><td>80.0</td><td>86.6</td><td>76.6</td><td>51.8</td><td>40.0</td><td>40.0</td><td>36.6</td><td>36.6</td><td>33.3</td><td>36.9</td></tr><tr><td>Ours</td><td>93.3</td><td>96.6</td><td>93.3</td><td>93.3</td><td>90.0</td><td>87.2</td><td>93.3</td><td>90.0</td><td>93.3</td><td>86.6</td><td>86.6</td><td>84.7</td></tr></table>

Table 1: Zero-shot comparisons in real-world. We compare our method against state-of-the-art imitation learning specialists and large VLA models. All models are fine-tuned on SynGrasp-1B dataset. Our approach achieves the highest grasping success rate on items from both synthetic and web categories using short trajectories. Detailed description of setups is provided in Section 5.1.

# 5.1 在现实世界中与视觉语言模型的零样本比较

任务定义。为了评估PAG的有效性，我们使用两组对象：合成类别和网络类别。我们将合成类别定义为出现在我们的SynGrasp-1B数据集中，而网络类别则指仅存在于互联网标注数据集中的类别。

对于每组物体，我们设计了5个测试集：基本、光照、背景、干扰物和高度。每个测试集包含从每组中随机抽取的15个来自不同类别的物体，每个物体进行2次试验。换句话说，我们总共对每种方法进行了 $15 \times 2 \times 5 \times 2 = 300$ 次试验。我们使用迪斯科灯生成不同的光照条件。为了背景泛化，选择了三种不同的桌布并进行更换。为了干扰物泛化，我们在桌上随机放置了5个额外的物体作为干扰物。为了高度泛化，我们将工作空间的表面高度增加了 $10 \mathrm{cm}$。我们使用Franka Panda机械臂，并配备两台Intel RealSense相机作为前视和侧视相机。工作空间限制在 $40 \mathrm{cm} \times 50 \mathrm{cm} \times 20 \mathrm{cm}$ 的区域内，位于机器人面前。每次试验的初始机器人和物体状态均固定，以确保公平比较。

![](images/4.jpg)  

Figure 4: We show our real-world setup in (a), objects used in experiments in (b,c), and 5 test sets corresponding to basic, light, background, distractor, and height settings in (d).

指标。成功率定义为模型在3次尝试中成功抓取目标物体的试验百分比。对于每个对象组，我们还报告平均成功率 $\begin{array} { r } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } S _ { i } \frac { l _ { i } } { m a x ( p _ { i } , l _ { i } ) } } \end{array}$ ，其中 $S _ { i }$ 是成功的二元指示符（成功则为1），$l _ { i }$ 是在试验中任何方法实现的最短路径长度，$p _ { i }$ 是模型所采取的路径长度，$N$ 是总试验次数。基线。我们与多个基线进行比较，包括VLA通才和模仿学习专门模型。对于通才，我们使用 $\pi _ { 0 }$ [7]、OpenVLA [6] 和Octo [26]，这三种基于变换器的策略在大规模现实世界数据集上预训练。为了确保公平比较，我们对这三种模型在我们的SynGrasp-1B数据集上进行了微调。此外，为了评估在SynGrasp-1B上的预训练有效性，我们报告了从其VLM权重 [77] 直接微调 $\pi _ { 0 }$ 的结果，不进行其跨体态机器人预训练。对于专门模型，我们使用Diffusion Policy [75]，这是一个强大的视觉条件模仿学习扩散基线。由于它缺乏语言条件，我们仅使用大象类别进行训练和测试。补充材料提供了更多细节。比较。如表1所示，GraspVLA在所有测试集上达到了约 $90 \%$ 的成功率，显著超越了所有基线，展示了强大的零-shot泛化能力。值得注意的是，GraspVLA在合成类别和网页类别中都取得了相当的结果，突显了PAG的有效性。此外，SPL指标显示，GraspVLA抓取物体的路径长度比 $\pi _ { 0 }$ 基线短，后者往往表现出犹豫有趣的是，未经过跨体态预训练的 $\pi _ { 0 }$ 基线表现优于其预训练的对应模型，这表明跨体态预训练可能不适合在给定的机器人手臂上的特定抓取任务。我们在补充材料中提供了失败分析。

# 5.2 在LIBERO基准测试中与VLA的零样本比较

设置。LIBERO 是一个广泛使用的机器人操作模拟基准，涵盖多种任务和物体类别。我们在三个 LIBERO 套件（长任务、目标任务、物体任务）上进行评估，排除了空间套件，因为其对空间推理的关注超出了我们的研究范围。为了集中于抓取能力，我们省略了非抓取任务（例如，“打开炉子”），并将任务标题重新表述为“捡起 $\{ \mathrm { o b j e c t } \} ^ { \prime }$”，每个套件选择 7-10 个任务。根据标准评估协议，每个任务都经过严格测试，采用 50 种随机初始配置，结果每个套件进行 350-500 次试验。更多细节请参见补充材料。

Table 2: Comparisons with baselines in LIBERO. The zero-shot performance of GraspVLA surpasses the fine-tuned performance of strong baselines $\pi _ { 0 }$ and OpenVLA.   

<table><tr><td></td><td>Long</td><td>Goal</td><td>Object</td></tr><tr><td>OpenVLA (fine-tuned)</td><td>33.7</td><td>56.6</td><td>65.4</td></tr><tr><td>π0 (fine-tuned)</td><td>62.7</td><td>79.4</td><td>93.8</td></tr><tr><td>Ours (zero-shot)</td><td>82.0</td><td>91.2</td><td>94.1</td></tr></table>

比较。如表2所示，GraspVLA在LIBERO上经过零样本评估时显示出令人满意的性能。它超越了在LIBERO数据集上微调的$\pi _ { 0 }$和OpenVLA，展现出强大的泛化能力。我们还观察到，任务标题的格式对微调模型的性能有显著影响，并在补充材料中提供了详细结果。

# 5.3 在真实世界中与 AnyGrasp 的零样本比较

设置。我们将 GraspVLA 与 AnyGrasp [14] 进行基准测试，后者是一种专注于抓取的最先进抓取检测模型。在语言条件抓取中，我们将 AnyGrasp 与 Grounding DINO [78] 集成，后者是一种流行的开放词汇物体检测器，用于过滤抓取候选对象。我们使用相同的两个基本测试集（第5.1节），评估指标包括整体成功率（任务完成）和抓取成功率（抓取任何物体）。为了隔离抓取性能，我们设计了两个额外的测试集（每个30次试验）：一个包含常见家居物品，另一个包含透明物品，机器人可以抓取场景中的任何物体。

<table><tr><td></td><td colspan="2">Language-Conditioned</td><td colspan="2">Arbitary Grasping</td><td>Speed</td></tr><tr><td></td><td>overall</td><td>grasp</td><td>common</td><td>transparent</td><td></td></tr><tr><td>AnyGrasp</td><td>91.6</td><td>96.6</td><td>100.0</td><td>10.0</td><td>37 Hz</td></tr><tr><td>Ours</td><td>93.3</td><td>93.3</td><td>93.3</td><td>86.6</td><td>5 Hz</td></tr></table>

Table 3: Comparison with AnyGrasp. GraspVLA performs consistently well in both language-guided and arbitrary grasping tasks. In contrast, AnyGrasp is faster and excels at grasping common objects but struggles with transparent objects.

比较。在语言条件测试集上，两种模型的表现相似，GraspVLA在基础能力上略微超越AnyGrasp，这得益于其全面的多视角观察。在任意物体抓取中，虽然AnyGrasp在抓取常见物体方面达到了$100\%$的成功率，但由于深度传感不准确和点云数据不完整，它在抓取透明物体时遇到了困难。相比之下，GraspVLA在两个测试集上均保持一致的性能，突显了其对材料变化的鲁棒性。然而，GraspVLA的推理速度显著慢于AnyGrasp，这一限制与其庞大的视觉-语言主干网络有关。

# 5.4 扩展规律

图5展示了实际场景中训练帧数量的扩展曲线。我们观察到，随着训练帧数量的增加，性能稳步提升，而网页类别的性能扩展速度低于合成类别，表明网页类别需要更多的训练帧以实现良好的泛化性能。有关训练类别数量和每类实例数量的扩展规律，请参阅补充材料。

![](images/5.jpg)  

Figure 5: The performance scales with the number of training frames, especially for web categories.

# 5.5 高效的后训练

基础模型的一个显著特征是其适应新任务的能力。为此，我们定义了三个下游任务：任务1是抓取工业组件，任务2是在不触及内部以保持清洁的情况下抓取物品，任务3是在密集环境中进行顺序抓取。这些任务严格测试模型在三项关键挑战下的适应能力：（i）对新词汇的泛化，（ii）执行特定任务的抓取规范，以及（iii）按顺序抓取。我们为任务1和任务2各收集100个演示，对于任务3每个瓶子收集10个演示。我们对每个任务进行10次试验，并报告整体成功率（任务完成率）和抓取成功率（抓取任意物体的成功率）。

![](images/6.jpg)  

Figure 6: Real-world post-training. We experimented with three different post-training tasks to showcase that our model can quickly learn to grasp new items in (a), new grasping patterns in (b), and new grasping behavior in (c).

如表4所示，GraspVLA在任务1中仅凭借边界框标注达到了$90\%$的成功率，超越了基于完整动作数据训练的基线。这表明，将GraspVLA扩展到新物体并不需要动作标注，从而大大减少了数据收集的工作量。如最后两行所示，从头开始训练的性能较低，表明GraspVLA有效地学习到了避免与周围物体发生碰撞。

Table 4: Efficient post-training. GraspVLA shows superior adaptability to novel tasks, surpassing the model without pretraining and all baselines.   

<table><tr><td></td><td colspan="2">Training Data</td><td colspan="2">Task 1</td><td colspan="2">Task 2</td><td colspan="2">Task 3</td></tr><tr><td></td><td>BBox</td><td>traj.</td><td>overall</td><td>grasp</td><td>overall</td><td>grasp</td><td>overall</td><td>grasp</td></tr><tr><td>OpenVLA</td><td>-</td><td>-</td><td>0</td><td>0</td><td>0</td><td>20</td><td>0</td><td>0</td></tr><tr><td>π0</td><td>-</td><td>-</td><td>10</td><td>20</td><td>0</td><td>30</td><td>0</td><td>0</td></tr><tr><td>Ours</td><td>-</td><td>-</td><td>40</td><td>90</td><td>0</td><td>80</td><td>0</td><td>20</td></tr><tr><td>DP</td><td></td><td>✓</td><td>-</td><td>-</td><td>20</td><td>60</td><td>10</td><td>30</td></tr><tr><td>OpenVLA</td><td></td><td>✓</td><td>0</td><td>0</td><td>20</td><td>30</td><td>0</td><td>20</td></tr><tr><td>π0</td><td></td><td>✓</td><td>60</td><td>80</td><td>60</td><td>70</td><td>50</td><td>60</td></tr><tr><td>Ours</td><td>✓</td><td>-</td><td>90</td><td>100</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Ours(scratch)</td><td>✓</td><td>✓</td><td>10</td><td>30</td><td>10</td><td>30</td><td>0</td><td>20</td></tr><tr><td>Ours</td><td></td><td>✓</td><td>90</td><td>100</td><td>80</td><td>90</td><td>90</td><td>90</td></tr></table>

# 5.6 设计选择的有效性

<table><tr><td></td><td colspan="2">Synthetic</td><td colspan="2">Web</td></tr><tr><td></td><td>SR</td><td>SPL</td><td>SR</td><td>SPL</td></tr><tr><td>vanilla</td><td>66.6</td><td>39.3</td><td>53.3</td><td>27.7</td></tr><tr><td>+ PAG-2D</td><td>80.0</td><td>59.2</td><td>76.7</td><td>48.9</td></tr><tr><td>+ PAG-3D</td><td>93.3</td><td>90.2</td><td>93.3</td><td>91.7</td></tr></table>

如表5所示，我们使用成功率和SPL指标评估我们的关键设计选择在第5.1节中描述的基本测试集上的有效性。基础基线使用了互联网基础数据进行协同训练，但排除了PAG，作为我们的起点。引入2D边界框作为中间动作步骤（PAG-2D）为网络类别带来了显著的改进。通过抓取姿态预测（PAG-3D）进一步增强，显著减少了犹豫行为并提高了抓取准确性。这导致了尝试次数减少和轨迹缩短，从而在更高的SPL分数中得以体现。 这些结果共同表明了我们PAG方法的有效性。

Table 5: We give a detailed ablation study of our models. With all the design choices enabled the performance boosts significantly.

# 6 结论

在本研究中，我们探讨了如何利用大规模合成数据构建一个具有良好泛化能力的抓取 VLA 模型。首先，我们在仿真中整理了一个亿级抓取数据集，具有广泛的随机化和逼真的渲染效果。其次，我们精心设计了模型，以有效地从合成动作数据和无动作互联网基础数据中学习，实现了在未见环境中抓取新类别物体的强泛化能力。大量的消融实验和比较结果表明，我们的方法在桌面抓取方面达到了最先进的性能。此外，我们观察到模型在合成训练数据量的增加下能够有效扩展。最后，我们展示了 GraspVLA 可以通过少量样本的后训练获取新的抓取行为，突显了其适应性和在现实应用中的潜力。

# 7 限制与未来工作

目前，我们的数据生成和评估仅在 Franta Panda臂上进行，采用前视和侧视。然而，我们的仿真管道本质上是可扩展的，可以方便地适应其他机器人和相机配置。我们将这项工程努力留待未来工作。GraspVLA在处理模糊指令时，如“拿起食物”和“拿起最左边的物体”，表现不佳。解决这些挑战可能需要扩展视觉-语言预训练，并探索架构创新以增强语义推理。与大多数抓取策略一样，我们使用力闭合合成抓取标签，这并未考虑形变性——这是所有此类方法的一个共同限制。尽管如此，我们的模型仍然可以抓取某些可变形物体，前提是它们的初始几何形状包含能够实现力闭合的凸区域。虽然先前的研究已表明软体仿真可以用于训练sim2real的可变形操作策略，但我们将这一整合留待未来工作。虽然当前模型专注于抓取，但模型设计并未针对这个特定任务进行优化。我们计划扩展数据生成管道，以支持其他操作任务，如抓取与放置和推送。除了当前用于数据生成的基于模块的专家策略外，我们还将探索强化学习用于更复杂的任务，如非抓取操作。尽管我们的PAG机制实现了开放词汇抓取，但它引入了额外的延迟。我们目前在使用Torch Compile的NVIDIA L40s上达到了约$200 \mathrm{ms}$的延迟。虽然这对于静态场景是足够的，但在动态环境中可能不够，例如快速移动的物体。可以进一步探索蒸馏和量化技术。

# 致谢

本研究部分得到了中国国家重点研发计划62306016和2022ZD0160201的支持。此外，我们对Galbot的所有同事在收集和标注后训练数据方面的帮助表示诚挚的感谢。

# References

[1] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave, and G. Lample. Llama: Open and efficient foundation language models, 2023. URL https://arxiv. org/abs/ 2302.13971.   
[2] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollár, and R. Girshick. Segment anything. arXiv:2304.02643, 2023.   
[3] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever. Learning transferable visual models from natural language supervision, 2021. URL https : //arxiv . org/abs/2103. 00020.   
[OpenAI.Chatpt: Jan 17 verio. https://opeai.com/chatpt, 203. [Large age model].   
[5] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.   
[6] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.   
[7] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. pi0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.   
[8] NVIDIA, :, J. Bjorck, F. Castañeda, N. Cherniadev, X. Da, R. Ding, L. J. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, J. Jang, Z. Jiang, J. Kautz, K. Kundalia, L. Lao, Z. Li, Z. Lin, K. Lin, G. Liu, L, LM AMr A. Na S. as S. ReeY. L. Ta . Z. Wang, J. Wan, Q. Wang, J. Xiang, Y. Xie, Y. Xu, Z. Xu, S. Ye, Z. Yu, A. Zhang, H. Zhang, Y. Zhao, R. Zheng, and Y. Zhu. Gr0Ot n1: An open foundation model for generalist humanoid robots, 2025. URL https://arxiv.org/abs/2503.14734.   
[9] A. O'Neill, A. Rehman, A. Gupta, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, et al. Open x-embodiment: Robotic learning datasets and rt-x models. arXiv preprint arXiv:2310.08864, 2023.   
[10] A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M..Sria,L..Che .Elis, e al.Droi:A large-scal in-the-wil rootanila dataset. arXiv preprint arXiv:2403.12945, 2024.   
[11] J. Liang, V. Makoviychuk, A. Handa, N. Chentanez, M. Macklin, and D. Fox. Gpu-accelerated robotic simulation for distributed reinforcement learning, 2018.   
[12] E. Todorov, T. Erez, and Y. Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 5026 5033, 2012. doi:10.1109/IROS.2012.6386109.   
[13] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36, 2024.   
H. Fa .  H. F .ou, J.Li H.Yan WL Y. Xien C.L.: Robust and efficient grasp perception in spatial and temporal domains. IEEE Tranctions on Robotics, 39(5):39293945, 2023.   
[15] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.   
[16] H. Bharadhwaj, J. Vakil, M. Sharma, A. Gupta, S. Tulsiani, and V. Kumar. Roboagent: Generalization and efficiency in robot manipulation via semantic augmentations and action chunking, 2023.   
[17] L. Wang, X. Chen, J. Zhao, and K. He. Scaling proprioceptive-visual learning with heterogeneous pre-trained transformers. arXiv preprint arXiv:2409.20537, 2024.   
[18] M. Zawalski, W. Chen, K. Pertsch, O. Mees, C. Finn, and S. Levine. Robotic control via embodied chain-of-thought reasoning. arXiv preprint arXiv:2407.08693, 2024.   
[19] X. Li, M. Zhang, Y. Geng, H. Geng, Y. Long, Y. Shen, R. Zhang, J. Liu, and H. Dong. Manipllm: Embodied multimodal large language model for object-centric robotic manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1806118070, 2024.   
[20] X. Li, C. Mata, J. Park, K. Kahatapitiya, Y. S. Jang, J. Shang, K. Ranasinghe, R. Burgert, M.Cai.J. Lee l.Llar:Superharg robot learg dat rvisin-ang oi. arXiv preprint arXiv:2406.20095, 2024.   
[21] A. Goyal, V. Blukis, J. Xu, Y. Guo, Y.-W. Chao, and D. Fox. Rvt-2: Learning precise manipulation from few demonstrations. arXiv preprint arXiv:2406.08545, 2024.   
[22] H. Zhen, X. Qiu, P. Chen, J. Yang, X. Yan, Y. Du, Y. Hong, and C. Gan. 3d-vla: A 3d visionlanguage-action generative world model. arXiv preprint arXiv:2403.09631, 2024.   
[3] J. Zhang, K. Wang, R. Xu, G. Zhou, Y. Hong, X. Fang, Q. Wu, Z. Zhan, and H. Wang. Na panse exi arXiv:2402.15852, 2024.   
[24] X. Chen, J. Djolonga, P. Padlewski, B. Mustafa, S. Changpinyo, J. Wu, C. R. Ruiz, S. GoodX.Tay aO u l. arXiv preprint arXiv:2305.18565, 2023.   
[25] D. Driess, F. Xia, M. S. Sajjadi, C. Lynch, A. Chowdhery, B. Ichter, A. Wahid, J. Tompson, Q. Vuong, T. Yu, et al. Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023.   
[26] O. M. Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.   
[27] Q. Li, Y. Liang, Z. Wang, L. Luo, X. Chen, M. Liao, F. Wei, Y. Deng, S. Xu, Y. Zhang, et al. Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation. arXiv preprint arXiv:2411.19650, 2024.   
[ J.Wen, .Zhu, JLi M.Zhu, K.Wu, Z. Xu N. Liu R.h.She, Y.Pe, FFg, an J. Tang Tnyvla:Towards fast, data-efficient vision-language-action models for roboic manipulation, 2024. URL https: //arxiv. org/abs/2409.12514.   
[29] S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, and J. Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864, 2024.   
[30] C.-L. Cheang, G. Chen, Y. Jing, T.Kong, H.Li, Y. Li, Y. Liu, H. Wu, J. Xu, Y. Yang, et al. Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158, 2024.   
.Ye, J. J B. Jn, S.Joo, J.Y B.Pe A.M R.Tan Y- B.Lin, et al. Latent action pretraining from videos. arXiv preprint arXiv:2410.11758, 2024.   
[32] H. Bharadhwaj, D. Dwibedi, A. Gupta, S. Tulsiani, C. Doersch, T. Xiao, D. Shah, F. Xia, D. Sadigh, and S. Kirmani. Gen2act: Human video generation in novel scenarios enables generalizable robot manipulation. arXiv preprint arXiv:2409.16283, 2024.   
[33] J. Yang, B. Liu, J. Fu, B. Pan, G. Wu, and L. Wang. Spatiotemporal predictive pre-training for robotic motor control. arXiv preprint arXiv:2403.05304, 2024.   
[34] Q. Zhao, Y. Lu, M. J. Kim, Z. Fu, Z. Zhang, Y. Wu, Z. Li, Q. Ma, S. Han, C. Finn, et al. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. arXiv preprint arXiv:2503.22020, 2025.   
[35] Y. Tian, S. Yang, J. Zeng, P. Wang, D. Lin, H. Dong, and J. Pang. Predictive inverse dynamics models are scalable learners for robotic manipulation, 2024. URL https ://arxiv.org/ abs/2412.15109.   
[36] P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, M. Y. Galliker, D. Ghosh, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, D. LeBlanc, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, A. Z. Ren, L. X. Shi, L. Smith, J. T. Springenberg, K. Stachowicz, J. Tanner, Q. Vuong, H. Walke, A. Walling, H. Wang, L. Yu, and U. Zhilinsky. $\pi _ { 0 . 5 }$ : a vision-language-action model with open-world generalization, 2025. URL https: //arxiv. org/abs/2504 . 16054.   
[37] K. Bousmalis, A. Irpan, P. Wohlhart, Y. Bai, M. Kelcey, M. Kalakrishnan, L. Downs, J. Ibarz, P. Pastor, K. Konolige, S. Levine, and V. Vanhoucke. Using simulation and domain adaptation to improve efficiency of deep robotic grasping, 2017. URL https : //arxiv . org/abs/1709 . 07857.   
[38] C. Eppner, A. Mousavian, and D. Fox. Acronym: A large-scale grasp dataset based on simulation, 2020. URL https://arxiv.org/abs/2011.09584.   
[39] J. Mahler, J. Liang, S. Niyaz, M. Laskey, R. Doan, X. Liu, J. A. Ojea, and K. Goldberg. Dex2.0Dee lear  plan obus asps with sntheic pot couds andanalys metrics, 2017. URL https://arxiv.org/abs/1703.09312.   
[40] A. Mandlekar, S. Nasiriany, B. Wen, I. Akinola, Y. Narang, L. Fan, Y. Zhu, and D. Fox. Mimicgen: A data generation system for scalable robot learning using human demonstrations, 2023. URL https://arxiv.org/abs/2310.17596.   
[41] Z. Jiang, Y. Xie, K. Lin, Z. Xu, W. Wan, A. Mandlekar, L. Fan, and Y. Zhu. Dexmimicgen: Automated data generation for bimanual dexterous manipulation via imitation learning, 2025. URLhttps://arxiv.org/abs/2410.24185.   
[42] C. Garrett, A. Mandlekar, B. Wen, and D. Fox. Skillmimicgen: Automated demonstration generation for efficient skill learning and deployment, 2024. URL https : //arxiv .org/ abs/2410.18907.   
[43] S. Yang, W. Yu, J. Zeng, J. Lv, K. Ren, C. Lu, D. Lin, and J. Pang. Novel demonstration generation with gaussian splattng enables robust one-shot manipulation, 2025. URL https: //arxiv.org/abs/2504.13175.   
[44] Z. Xue, S. Deng, Z. Chen, Y. Wang, Z. Yuan, and H. Xu. Demogen: Synthetic demonstration generation for data-efficient visuomotor policy learning, 2025. URL https://arxiv .org/ abs/2502.16932.   
[45] Z. Chen, S. Kiami, A. Gupta, and V. Kumar. Genaug: Retargeting behaviors to unseen situations via generative augmentation, 2023. URL https://arxiv . org/abs/2302. 06671.   
[46] T. Yu, T. Xiao, A. Stone, J. Tompson, A. Brohan, S. Wang, J. Singh, C. Tan, D. M, J. Perala B. Ichter, K. Hausman, and F. XiaScaling robot learning with semantically imagnd experience, 2023. URL https://arxiv.org/abs/2302.11550.   
[47] A. Maddukuri, Z. Jiang, L. Y. Chen, S. Nasiriany, Y. Xie, Y. Fang, W. Huang, Z. Wang, Z. Xu, N. Chernyadev, S. Reed, K. Goldberg, A. Mandlekar, L. Fan, and Y. Zhu. Sim-and-real cotraining: A simple recipe for vision-based robotic manipulation, 2025. URL https : / /arxiv . org/abs/2503.24361.   
[8] R. Newbury, M. Gu, L. Chumbley, A. Mousavian, C. Eppner, J. Leitner, J. Bohg, A. Morales, T. Asfour, D. Kragic, et al. Deep learning approaches to grasp synthesis: A review. IEEE Transactions on Robotics, 39(5):39944015, 2023.   
[49] H.-S. Fang, C. Wang, M. Gou, and C. Lu. Graspnet-1billion: A large-scale benchmark for general object grasping. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1144111450, 2020. doi:10.1109/CVPR42600.2020.01146.   
[50] A. Mousavian, C. Eppner, and D. Fox. 6-dof graspnet: Variational grasp generation for object manipulation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 29012910, 2019.   
[51] S. Wei, H. Geng, J. Chen, C. Deng, C. Wenbo, C. Zhao, X. Fang, L. Guibas, and H. Wang. D3roma: Disparity diffusion-based depth sensing for material-agnostic robotic manipulation. In 8th Annual Conference on Robot Learning, 2024. URL https://openreview.net/ forum?id $\equiv$ 7E3JAys1x0.   
[52] Y. Liu, A. Qualmann, Z. Yu, M. Gabriel, P. Schillinger, M. Spies, N. A. Vien, and A. Geiger. Efficient end-to-end detection of 6-dof grasps for robotic bin picking, 2024. URL https: //arxiv.org/abs/2405.06336.   
[53] H. Geng, S. Wei, C. Deng, B. Shen, H. Wang, and L. Guibas. Sage: Bridging semantic and actionable parts for generalizable articulated-object manipulation under language instructions. arXiv preprint arXiv:2312.01307, 2023.   
[4] D. Kalashnikov, A. Irpan, P. Pastor, J. Ibarz, A. Herzog, E. Jang, D. Quillen, E. Holly, Klaishnan, V.Vanoucke, anS. Levi Qt-op: Scalabl dep rert ar for vision-based robotic manipulation, 2018. URL https: //arxiv . org/abs/1806 . 10293.   
[55] S. Song, A. Zeng, J. Lee, and T. Funkhouser. Grasping in the wild: Learning 6dof closedloop grasping from low-cost demonstrations. IEEE Robotics and Automation Letters, 5(3): 49784985, 2020.   
[56] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:2371623736, 2022.   
[57] S. Karamcheti, S. Nair, A. Balakrishna, P. Liang, T. Kollar, and D. Sadigh. Prismatic vlms: Investigating the design space of visually-conditioned language models, 2024. URL https : //arxiv.org/abs/2402.07865.   
[58] A. D. Vuong, M. N. Vu, H. Le, B. Huang, B. Huynh, T. Vo, A. Kugi, and A. Nguyen. Graspanything: Large-scale grasp dataset from foundation models, 2023. URL https: //arxiv. org/abs/2309.09818.   
[59] A. Stone, T. Xiao, Y. Lu, K. Gopalakrishnan, K.-H. Lee, Q. Vuong, P. Wohlhart, S. Kirmani, models. arXiv preprint arXiv:2303.00905, 2023.   
[60] C. Tang, D. Huang, W. Ge, W. Liu, and H. Zhang. Graspgpt: Leveraging semantic knowledge from a large language model for task-oriented grasping. IEEE Robotics and Automation Letters, 2023.   
[..Fn B.e . .L  .WV-: - - icy for language-oriented objects in cluttered indoor scenes. In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 976983. IEEE, 2023.   
[62] Y. Ding, H. Geng, C. Xu, X. Fang, J. Zhang, S. Wei, Q. Dai, Z. Zhang, and H. Wang. Open6dor: Benchmarking open-instruction 6-dof object rearrangement and a vlm-based approach. In 2024 EE/R International Conference on Intellient Robots and Sstems (IROS), pages 73597366. IEEE, 2024.   
[63] M. Deitke, D. Schwenk, J. Salvador, L. Weihs, O. Michel, E. VanderBilt, L. Schmidt, K. Ehsani, A. Kembhavi, and A. Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1314213153, 2023.   
[] J.hen Y.Ke, ad H. Wan Bod: Scab nd eff bdextu ras nes using bilevel optimization. arXiv preprint arXiv:2412.16490, 2024.   
[65] B. Sundaralingam, S. K. S. Hari, A. Fishman, C. Garrett, K. Van Wyk, V. Blukis, A. Millane, H. Oleynikova, A. Handa, F. Ramos, et al. Curobo: Parallelized collision-free robot motion generation. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 81128119. IEEE, 2023.   
[66] M. Mittal, C. Yu, Q. Yu, J. Liu, N. Rudin, D. Hoeller, J. L. Yuan, R. Singh, Y. Guo, H. Mazhar, A. Mandlekar, B. Babich, G. State, M. Hutter, and A. Garg. Orbit: A unified simulation framework for interactive robot learning environments. IEEE Robotics and Automation Letters, 8(6):37403747, 2023. doi:10.1109/LRA.2023.3270034.   
[67] M. Dalal, A. Mandlekar, C. Garrett, A. Handa, R. Salakhutdinov, and D. Fox. Imitating task and motion planning with visuomotor transformers. arXiv preprint arXiv:2305.16309, 2023.   
.  .  .  J.oY. for robotic manipulation, 2024. URL https : //arxiv . org/abs/2410 . 18647.   
[69] Z. Cai, M. Cao, H. Chen, K. Chen, K. Chen, X. Chen, X. Chen, Z. Chen, Z. Chen, P. Chu, et al. Internlm2 technical report. arXiv preprint arXiv:2403.17297, 2024.   
[70] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziz, F. Massa, A. El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.   
[71] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer. Sigmoid loss for language image pretrai In roi he Inteial one nur isn 1197511986, 2023.   
[72] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.   
[73] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. Chain-ofthought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:2482424837, 2022.   
[74] Z. Peng, W. Wang, L. Dong, Y. Hao, S. Huang, S. Ma, and F. Wei. Kosmos-: Grounding multimodal large language models to the world. ArXiv, abs/2306.14824, 2023.   
[75] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song. Diffusion policy: Visuomotor policy learning via action dffusion. In Proceedings of Robotics: Science and Systems (RSS), 2023.   
[76] P. Anderson, A. Chang, D. S. Chaplot, A. Dosovitskiy, S. Gupta, V. Koltun, J. Kosecka, J. Mali R. Mottaghi, M. Savva, and A. R. Zamir. On evaluation of embodied navigation agents, 2018. URL https://arxiv.org/abs/1807.06757.   
[77] L. Beyer, A. Steiner, A. S. Pinto, A. Kolesnikov, X. Wang, D. Salz, M. Neumann, I. Alabdulmohsin, M. Tschannen, E. Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024.   
[78] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su, J. Zhu, and L. Zhang. Grounding dino: Marrying dino with grounded pre-training for open-set object detection, 2024. URL https://arxiv.org/abs/2303.05499.   
[79] Y. Chen, B. Xiao, and H. Wang. Foldnet: Learning generalizable closed-loop policy for garment folding via keypoint-driven asset and demonstration synthesis, 2025. URL https: //arxiv.org/abs/2505.09109.   
[80] Introduction to torch.compile — PyTorch tutorials $2 . 7 . 0 \substack { + \mathrm { c u l } 2 6 }$ documentation, 2023. URL https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html.   
[81] H. Walke, K. Black, A. Lee, M. J. Kim, M. Du, C. Zheng, T. Zhao, P. Hansen-Estruch, Q. Vuong, A. He, V. Myers, K. Fang, C. Finn, and S. Levine. Bridgedata v2: A dataset for robot learning at scale, 2024. URL https: //arxiv . org/abs/2308 . 12952.   
[82] AgiBot-World-Contributors, Q. Bu, J. Cai, L. Chen, X. Cui, Y. Ding, S. Feng, S. Gao, X. He, X. Hu, X. Huan, S. Jiang, Y. Jiang, C. Jing, H. Li, J. Li, C. Liu, Y. Liu, Y. Lu, J. Luo, P. Luo, Y. Mu, Y. Niu, Y. Pan, J. Pang, Y. Qiao, G. Ren, C. Ruan, J. Shan, Y. Shen, C. Shi, M. Shi, M.Si C.S J.So H.W WW D.Wei C. Xie G. Xu J.Yan C.Yag, L., S. Yang, M. Yao, J. Zeng, C. Zhang, Q. Zhang, B. Zhao, C. Zhao, J. Zhao, and J. Zhu. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025.   
[83] S. Valette and J.-M. Chassery. Approximated centroidal voronoi diagrams for uniform polygonal mesh coarsening. In Computer Graphics Forum, volume 23, pages 381389. Wiley Online Library, 2004.   
[84] google-deepmind/envlogger, Jan. 2025. URL https://github.com/google-deepmind/ envlogger. original-date: 2021-07-28T15:35:08Z.   
[85] Universally unique identifier, Jan. 2025. URL https://en.wikipedia.org/w/index. php?title $\fallingdotseq$ Universally_unique_identifier&oldid $=$ 1272340425. Page Version ID: 1272340425.   
[86] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. In Proceedings of Robotics: Science and Systems, Daegu, Republic of Korea, July 2023. doi:10.15607/RSS.2023.XIX.016.   
[87] F. R. GmbH. Ros integration for franka robotics research robots. https://github. com/ frankaemika/franka_ros, 2023.   
[88] J. Luo, Z. Hu, C. Xu, Y.L. Tan, J. Berg, A. Sharma, S. Schaal, C. Finn, A. Gupta, and S. Levine. Serl: A software suite for sample-efficient robotic reinforcement learning, 2024.

# A Overview

In the supplementary materials, we provide details about SynGrasp-1B dataset in Section B and C. We also show that GraspVLA supports fast adaptation to new robotic arms and camera configurations in Section D. We provide details about our main experiments in Section E. We also provide additional scaling law experiments in Section F. Details about LIBERO benchmark is provided in Section G. Comprehensive comparison with AnyGrasp is provided in Section H. We ablate the number of camera views in Section I. We also analyze the sim2real gap in Section J. We provide details about the inference delay in Section K. The implementation details of our non-blocking controller is provided in Section L. We also provide details about the failure cases in Section M.

# B Details about SynGrasp-1B

We present the statistics of our synthetic dataset, SynGrasp-1B, in Table 6. The dataset consists of 10 million trajectories, each containing approximately 100 frames, resulting in a total of 1 billion frames—a substantial increase in scale compared to existing open-source datasets. Our dataset encompasses a diverse array of object categories, featuring 10,680 objects across 240 categories. While real-world datasets often encounter challenges related to scene diversity and require collaboration among laboratories across different countries, synthetic data generation enables us to easily create varied scenes by altering the textures of the table, ground, and walls. We utilize around 1,000 different textures for the table and 1,200 for the ground and wals, leading to a total of 1 million unique scenes.

Unlike existing datasets, SynGrasp-1B is the first to offer precise and fine-grained annotations for camera calibration, bounding box annotations, and the 3D poses of both the target object and the gripper. Thanks to our simulation engine, we can effortlessly obtain these annotations and incorporate additional types, such as depth maps and segmentation masks, when necessary. This flexibility is a significant advantage of synthetic datasets over their real-world counterparts.

Table 6: Comparison of SynGrasp-1B with Existing Datasets for Robot Manipulation. This table highlights the advantages of SynGrasp-1B in terms of scale, annotations, and scene diversity compared to other datasets.   

<table><tr><td></td><td>Trajectories</td><td>Objects</td><td>Scenes</td><td>Camera Calibration</td><td>Bbox Annotation</td><td>3D Pose Annotation</td></tr><tr><td>RoboSet [16]</td><td>98k</td><td>&lt; 200</td><td>11</td><td>×</td><td>×</td><td>X</td></tr><tr><td>BridgeData V2 [81]</td><td>60k</td><td>100</td><td>24</td><td></td><td>X</td><td>X</td></tr><tr><td>RT-1 [15]</td><td>130k</td><td>&lt; 200</td><td>2</td><td>×</td><td>×</td><td>X</td></tr><tr><td>DROID [10]</td><td>76k</td><td>&lt; 200</td><td>2080</td><td>✓</td><td>✗</td><td>X</td></tr><tr><td>AgiBot World [82]</td><td>1M</td><td>3k</td><td>106</td><td>✓</td><td>X</td><td>X</td></tr><tr><td>Open-X Embodiment [9]</td><td>1.4M</td><td>-</td><td>311</td><td>X</td><td>X</td><td>X</td></tr><tr><td>SynGrasp-1B</td><td>10M</td><td>10k</td><td>10M</td><td>✓</td><td>✓</td><td>✓</td></tr></table>

The generation of simulation data is also much more cost-effective than real-world data collection, considering factors such as time, financial resources, space, robotic equipment, and human labor:

•Time: A single human operator can collect only around 1,000 trajectories per day. In contrast, we can generate 10 million trajectories in 10 days using 160 NVIDIA 4090 GPUs. This efficiency accelerates the data-model feedback loop, enabling rapid model iteration and performance improvements.   
•Space: Real-world data collection often necessitates multiple laboratories across different countries to enhance scene diversity. Additionally, it requires significant physical space to accommodate robots and objects; for example, the AgiBot World dataset [82] utilizes a $4 { , } 0 0 0 m ^ { 2 }$ area for data collection. In contrast, our synthetic approach does not require any physical space.   
Robots: Each human operator in real-world collection needs a physical robot, resulting in high costs and maintenance overhead.   
Money: The total cost of generating SynGrasp-1B is around $\$ 5,000$ orders of magnitude cheaper than real-world alternatives.

As an initial step toward large-scale synthetic action pre-training, we focus on grasping tasks to enable detailed analysis. Our pipeline can be extended to other robotic arms (for cross-embodiment transfer) or tasks (e.g., placing, pushing, stacking), as well as large-scale camera randomization. We leave these extensions to future work.

Gallery. We randomly sample 24 trajectories from our SynGrasp-1B and visualize them in Fig. 7. Each trajectory consists of around 100 frames, and we uniformly sample 4 frames from each trajectory for visualization.

# C Details about Data Generation

Object Processing and Layout Generation. To ensure that the scales of synthetic objects align with real-world counterparts and are suitable for grasping, we manually define minimum and maximum size constraints for each category. This helps our model to generalize to real-world objects with diverse scales. As shown in Fig. 8, GraspVLA can grasps all scales of dog, ranging from $2 \mathrm { c m }$ to $3 5 \mathrm { c m }$ .Furthermore, we simplify the object meshes using the ACVD algorithm [83] to improve the simulation efficiency. Additionally, we randomize the height of the table, ranging from $- 0 . 1 \textrm { m }$ to 0.2 m in the robot frame.

To enhance data diversity, we create different clutter layouts for each episode by randomly placing objects within a $0 . 4 \mathrm { m }$ by $0 . 5 \mathrm { m }$ area on a table. Objects are dropped in various poses to generate physically plausible scenes. For categories requiring specific orientations, such as cups, we manually define valid poses (e.g., upright).

The cameras are randomized within a $1 5 \mathrm { { c m } }$ radius ball and rotated $\pm 5 ^ { \circ }$ around each axis. Details are shown in Table 7.

Table 7: Camera parameters   

<table><tr><td></td><td>Position</td><td>Lookat</td></tr><tr><td>Front Camera</td><td>x=1.35, y=0.0, z=0.54</td><td>x=0.2, y=0.0, z=0.0</td></tr><tr><td>Side Camera</td><td>x=0.5, y=0.69, z=0.50</td><td>x=0.5, y=0.0, z=0.1</td></tr></table>

Asynchronous and Grouped Data Writing. We employ DeepMind EnvLogger with TFDS backend [84] for the storage of our synthetic data. Despite its clean design, considerations for highperformance large-scale simulation are necessary. Firstly, to mask the time cost of image encoding and data writing, we modified the EnvLogger implementation to perform the actual data writing operation asynchronously. Second, to avoid contention on the dataset metadata across parallel processes and to minimize data loss caused by unforeseen errors (e.g., GPU failures), the processes should not write to a single shared folder. However, if each simulation instance utilizes a unique subfolder, it results in a large number of subfolders and metadata, leading to substantial overheads in data management, transfer, and loading. As a compromise, we assign each process a subfolder with random UUIDs [85], and write all the trajectories within a process to the same subfolder.

Handling Data Corruption. During billion-scale data generation, a simulation process could hang and get killed due to hardware faults or excessive memory consumption, resulting in file corruption or loss. We handle these issues by proactively managing exceptions when loading the dataset with TFDS. Upon encountering a NotFoundError, we create an empty file at the expected file path, otherwise TFDS cannot continue loading the remaining records within the subfolder. On DataLossError, we count it as one missing record, and log the missing rate as a critical statistics for data validity. The missing rate was below $1 \%$ . FailedPreconditionError is raised when the successfully loaded number of records is smaller than that in the metadata of the subfolder, and can be safely converted to a StopIteration to facilitate the correct functioning of the data loader. The above handling of these exceptions ensures loading all the valid records. We believe these insights will benefit future large-scale dataset efforts.

![](images/7.jpg)  
Figure 7: Gallery of SynGrasp-1B. 24 randomly sampled trajectories from our synthetic grasping data. For clarity, 4 frames are uniformly sampled from each trajectory for display.

![](images/8.jpg)  
Figure 8: GraspVLA handles object with diverse scales, from 2cm to $3 5 \mathrm { c m }$ .

# D Fast Adaptation to New Robotic Arm and Camera

While we focus on specific setups for in-depth analysis, GraspVLA is not specialized for this. Our model can easily adapt to new robotic arms, grippers, and camera configurations with 5k additional synthetic trajectories (generated in one day on an NVIDIA 4090 GPU). We provide the following two examples:

•New robotic arm and gripper. We use a UR5e arm with a Robotiq 2F-85 gripper. Since no real-world hardware is available, we test our model on the simulation environment. •New camera configuration. We use front view and wrist view cameras. We conduct real-world experiments by mounting the camera on the wrist of the Franka Panda arm.

As shown in Fig. 9 and Table 8, our model shows strong performance with minimal fine-tuning, enabling rapid deployment on new setups.

# Wrist camera

![](images/9.jpg)  
Ur5e with Robotiq gripper

![](images/10.jpg)  
Figure 9: GraspVLA supports fast adaptation to new robotic arms and camera configurations.

<table><tr><td></td><td>Wrist camera</td><td>UR5e arm with Robotiq gripper</td></tr><tr><td>Success Rate</td><td>76.6</td><td>82.1</td></tr></table>

Table 8: Success rate of GraspVLA on new robotic arms and camera configurations.

# E Details of Main Experiments

Metrics. In each trial, the model is allowed to attempt to grasp up to three times, with each attempt counted by the gripper closure action. Success is strictly defined as the specified object being lifted a minimum of $1 5 \mathrm { { c m } }$ . The scene is not reset during each trial, even if the model knocks the object off the table.

Additionally, we introduce Success weighted by Path Length (SPL) to further account for the number of actions taken, which is a common metric in discrete navigation tasks to evaluate the efficiency of the mode.While for discrete navigation tasks, the shortest path is asy to define, for graspin tasks, the shortest path is not well defined. Therefore, for each trial, if there are several methods that can successfully grasp the object, we define the shortest path as the one with the least number of action steps. If all methods fail to grasp the object, they all get zero SPL in this trial. Note that, our SynGrasp-1B dataset stores actions in $1 0 \ : \mathrm { H z }$ and all methods are trained on this dataset, so the number of action steps is comparable across methods.

Baselines. As Diffusion Policy does not support language conditioning, we train it using the subset of SynGrasp-1B that grasps the elephant (around $4 0 \mathrm { k }$ trajectories), and replace the target object with the elephant in the real-world experiments. We train all other models with the full SynGrasp-1B dataset. We train $\pi _ { 0 }$ [7] with its pre-trained weights initialization and PaliGemma initialization for comparison. Since OpenVLA [6] takes a single RGB image for visual observation, we use the front camera view for it. For Diffusion Policy [75], we train the UNet-based version as recommended by the original paper. For Octo [26], we finetune the pre-trained octo-base-1.5 model. All the models are trained with action chunks of 4 [86], except for OpenVLA, which does not support action chunking. We provide hyperparameters of training/finetuning GraspVLA and baselines in Table 9. We run automatic evaluation in our simulation pipeline for all the baselines, continue training until the success rate converges, and select the best-performing checkpoint in the real world. We found GraspVLA achieves a consistently high success rate after 120k steps.

Table 9: Hyperparameters for all the methods   

<table><tr><td rowspan=1 colspan=1>Baseline</td><td rowspan=1 colspan=1>Hyperparameter</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>GraspVLA</td><td rowspan=1 colspan=1>batch_sizelearning_rate</td><td rowspan=1 colspan=1>3841.6e-4</td></tr><tr><td rowspan=1 colspan=1>π0</td><td rowspan=1 colspan=1>batch_sizelearning_ratewarmup_stepspeak_lrdecay_lrdecay_steps</td><td rowspan=1 colspan=1>256cosine schedule10002.5e-52.5e-630000</td></tr><tr><td rowspan=1 colspan=1>OpenVLA</td><td rowspan=1 colspan=1>lora_rankbatch_sizelearning_rateimage_aug</td><td rowspan=1 colspan=1>32125e-4true</td></tr><tr><td rowspan=1 colspan=1>Octo</td><td rowspan=1 colspan=1>batch_sizelearning_ratewarmup_stepsinit_valuepeak_value</td><td rowspan=1 colspan=1>256rsqrt schedule20000.03e-4</td></tr><tr><td rowspan=1 colspan=1>Diffusion Policy</td><td rowspan=1 colspan=1>batch_sizelearning_ratewarmup_stepspeak _lrweight_decay</td><td rowspan=1 colspan=1>256cosine schedule5001e-41e-6</td></tr></table>

Real world setup and modification to robot finger. For perception, we employ an Intel RealSense D435 as the front-facing camera and a D415i as the side-facing camera. Both cameras are positioned at the center of the randomization range used in the synthetic data generation. The workspace for test objects is confined to a $4 0 \thinspace \mathrm { c m } \thinspace \mathrm { x } \thinspace 5 0 \thinspace \mathrm { c m } \mathrm { x } \thinspace 2 0 \thinspace \mathrm { c m }$ area in front of the robot.

The original Franka Panda finger is too short to firmly grasp convex-shaped objects (e.g., a bottle lying on its side). This is because the hand plank collides with the top of the object, preventing the fingers from reaching deep enough to secure a stable grip. To address this issue, we extended the fingers by $2 \mathrm { c m }$ in both synthetic data generation and real-world experiments.

Details about PAG-3D. For steps before the gripper closure, we use the open-loop grasp pose of this trajectory as supervision. For steps after the gripper closure, we use the next step's end-effector pose as supervision.

# F Detailed Scaling Law

Simulation evaluation. While the main paper analyzes the scaling law in real-world, we extend this analysis to simulation environments. Our results show that simulation is an effective proxy for predicting real-world performance. For these experiments, we use simulation environments with identical camera and table configurations to our data generation setup, but employ different object instances and materials to assess generalizability.

As shown in Fig. 10a), GraspVLA's performance on simulation data follows a scaling trend similar to that of real-world data, confirming the simulation's effectiveness for predicting real-world performance. However, we observe two key differences: (1) real-world performance scales more slowly (0.12B) compared to simulation, where performance saturates earlier, and (2) the sim-to-real gap decreases with more training frames, suggesting that larger datasets enable more robust representations and better transfer to real-world scenarios.

![](images/11.jpg)  
Figure 10: Scaling laws different training regimes. (a) Performance scaling with number of training frames in both simulation and real-world environments. (b) Impact of training category diversity while fixing instances per category. (c) Effect of varying instances per category while maintaining total category count.

We further investigate how data diversity affects GraspVLA's performance by analyzing two additinal scaling factors:(1) the number of training categories and (2) the number of instances per category. For each analysis, we hold the other factor and total training frames constant.

Number of Training Categories (Fig. 10b). When varying the number of categories while fixing instances per category and total frames, performance on web categories improves steadily with more training categories, whereas performance on synthetic categories saturates early. This implies that inter-category generalization (adapting to unseen categories) benefits significantly from broader categorical coverage, while intra-category generalization (recognizing diverse instances of known categories) requires less diversity.

Number of Instances per Category (Fig. 10c). With a fixed category count and total frames, increasing instances per category leads to consistent improvements across both synthetic and web categories. This underscores the importance of instance diversity within categories for robust generalization.

# G Details about Experiments on LIBERO Benchmark

Setup. We consider a trial successful if the robot successfully grasps and lifts the target object to a height of $1 0 \ \mathrm { c m }$ . Since our model is trained with two camera views, we modify the original camera configurations provided by the LIBERO benchmark to match our training setup, aligning the camera poses accordingly. As the basket in some tasks occludes the side view severely, we remove it. Additionally, we extend the gripper by $2 \mathrm { c m }$ , as detailed in the robot finger modification in Section E. These adjustments are made exclusively for evaluating our model to ensure they do not affect the fine-tuned baselines, which are evaluated using the original camera configurations and gripper length.

The LIBERO-object test set presents a significant challenge for zero-shot models due to ambiguous target object descriptions. As illustrated in Figure 11, even humans may struggle to identify objects like "alphabet soup" and "cream cheese" in the given scene. To account for this, we relax the success citer:a trial is deemed sucessful if the robot grasps any object belonging to the same category as the target. For instane, if the target is "alphabet soup,"grasping any object in the "can" cateory is considered a success. Similarly, if the target is "cream cheese," grasping any object in the "box" category qualifies as success.

![](images/12.jpg)  
Figure 11: Examples of LIBERO Benchmark. We visualize both front and side views side by side.

As noted in the main paper, we exclude non-prehensile tasks to focus solely on grasping capability. We also omit tasks requiring color-based distinctions (e.g., "pick up the yellow and white mug"), as reasoning about color falls outside our scope. The specific tasks deemed invalid in the original test set, along with the modified instructions, are detailed in Tables 10, 11, and 12.

Table 10: Modification to LIBERO-Goal test set.   

<table><tr><td>Original Caption</td><td>Valid</td><td>Modified Caption</td></tr><tr><td>put the wine bottle on top of the cabinet</td><td>√</td><td>pick up the wine bottle</td></tr><tr><td>open the top drawer and put the bowl inside</td><td>✓</td><td>pick up the bowl</td></tr><tr><td>turn on the stove</td><td>X</td><td></td></tr><tr><td>put the bowl on top of the cabinet</td><td>✓</td><td>pick up the bowl</td></tr><tr><td>put the bowl on the plate</td><td>✓</td><td>pick up the bowl</td></tr><tr><td>put the wine bottle on the rack</td><td>✓</td><td>pick up the wine bottle</td></tr><tr><td>put the cream cheese in the bowl</td><td>✓</td><td>pick up the cream cheese box</td></tr><tr><td>open the middle drawer of the cabinet</td><td>X</td><td></td></tr><tr><td>push the plate to the front of the stove</td><td>X</td><td></td></tr><tr><td>put the bowl on the stove</td><td>✓</td><td>pick up the bowl</td></tr></table>

Table 11: Modification to LIBERO-Object test set.   

<table><tr><td>Original Caption</td><td>Valid</td><td>Modified Caption</td></tr><tr><td>pick up the alphabet soup and place it in the basket pick up the cream cheese and place it in the basket</td><td>✓</td><td>pick up the alphabet soup can</td></tr><tr><td></td><td>✓</td><td>pick up the cream cheese box</td></tr><tr><td>pick up the milk and place it in the basket</td><td></td><td>pick up the milk</td></tr><tr><td>pick up the tomato sauce and place it in the basket</td><td></td><td>pick up the tomato sauce can</td></tr><tr><td>pick up the butter and place it in the basket</td><td></td><td>pick up the butter box</td></tr><tr><td>pick up the orange juice and place it in the basket</td><td></td><td>pick up the orange juice</td></tr><tr><td>pick up the chocolate pudding and place it in the basket</td><td></td><td>pick up the chocolate pudding box</td></tr><tr><td>pick up the bbq sauce and place it in the basket</td><td></td><td>pick up the bbq sauce bottle</td></tr><tr><td>pick up the ketchup and place it in the basket</td><td></td><td>pick up the ketchup bottle</td></tr><tr><td>pick up the salad dressing and place it in the basket</td><td>✓</td><td>pick up the salad dressing bottle</td></tr></table>

Baselines. For baseline models, OpenVLA and $\pi _ { 0 }$ [6, 7], we use the authors' official fine-tuned checkpoints. Both models are fine-tuned on the LIBERO demonstration dataset, processed by OpenVLA to exclude static frames and failure trajectories, and rendered in high resolution.

Impact of instruction format. While the original test sets in LEBERO mainly compose of two steps, picking up an object and placing it in a container, we focus on the first step to exclusively evaluate the grasping capabilities. Therefore, to ensure a fair comparison, we also simplify the original instruction format "pick up a object and place it in a container" to "pick up a object" or the same instructions across all models. Note that, the instructions in the fine-tuning set are not simplified due to difficulties in segmenting and removing actions related to placing objects in containers.

As shown in Table 13, the performance of both fine-tuned baselines drops significantly when the instruction format is simplified. This indicates that the models are not robust to instruction variations and are sensitive to the specific instruction format. Additionally, our zero-shot model, GraspVLA, outperforms the fine-tuned models in the simplified instruction format and achieves comparable performance in the original instruction format. This demonstrates the robustness of our model to generalize to unseen environments, even in the absence of fine-tuning.

Table 12: Modification to LIBERO-Long test set.   

<table><tr><td>Original Caption</td><td>Valid</td><td>Modified Caption</td></tr><tr><td>turn on the stove and put the moka pot on it</td><td>✓</td><td>pick up the moka pot</td></tr><tr><td>put the black bowl in the bottom drawer of the cabinet and close it</td><td>✓</td><td>pick up the black bowl</td></tr><tr><td>put the yellow and white mug in the microwave and close it</td><td>X</td><td></td></tr><tr><td>put both moka pots on the stove</td><td>✓</td><td>pick up the moka pot</td></tr><tr><td>put both the alphabet soup and the cream cheese box in the basket</td><td>✓</td><td>pick up the alphabet soup can</td></tr><tr><td>put both the alphabet soup and the tomato sauce in the basket</td><td>✓</td><td>pick up the alphabet soup can</td></tr><tr><td>put both the cream cheese box and the butter in the basket</td><td>✓</td><td>pick up the cream cheese box</td></tr><tr><td>put the white mug on the left plate and put the yellow and white mug on the right plate</td><td>X</td><td></td></tr><tr><td>put the white mug on the plate and put the chocolate pudding to the right of the plate</td><td>X</td><td></td></tr><tr><td>pick up the book and place it in the back compartment of the caddy</td><td>✓</td><td>pick up the book</td></tr></table>

<table><tr><td></td><td>Long</td><td>Goal</td><td>Object</td></tr><tr><td>Format: pick up {object} and place it in {container} OpenVLA (fine-tuned)</td><td>70.9</td><td>78.6</td><td>91.2</td></tr><tr><td>π0 (fine-tuned)</td><td>88.7</td><td>95.4</td><td>98.4</td></tr><tr><td>Format: pick up {object}</td><td></td><td></td><td></td></tr><tr><td>OpenVLA (fine-tuned)</td><td>33.7</td><td>56.6</td><td>65.4</td></tr><tr><td>π0 (fine-tuned) Ours (zero-shot)</td><td>62.7 82.0</td><td>79.4</td><td>93.8</td></tr><tr><td></td><td></td><td>91.2</td><td>94.1</td></tr></table>

Table 13: Impact of instruction format. Fine-tuned baselines exhibit performance drops when the original instructions are simplified.

# H Details about Comparison with AnyGrasp

Setup. To ensure a fair comparison, we run the AnyGrasp baseline with up to three attempts per trial, counting it as a success if the object is grasped in any attempt. The baseline is implemented using the authors' oficial SDK. For perception, we use the same Franka Emika Panda robot and a RealSense D435i camera mounted on the end-effector, with the camera calibrated for acurate depth perception. Inference speed is evaluated on an NVIDIA RTX 3090 GPU.

For the language-conditioned test set, we integrate Grounding DINO [78] to parse language instructions into bounding boxes. Grasp candidates whose 2D projections fall outside these boxes are filtered out. Given the sparse layout, this simple approach effectively eliminates irrelevant grasps. Motion planning is then used to generate trajectories for execution.

Test Sets. The language-driven task uses the same test set as in the main experiment (Table 1 in the main paper), comprising both synthetic and web categories for a total of 60 trials. For arbitrary grasping of common objects, we randomly select 30 objects (15 synthetic, 15 web), ensuring diffuse, non-reflective materials (e.g., rubber, wood). The transparent object test set consists of 5 objects, including 3 bottles, 1 cup, and 1 bowl. To focus on grasping transparent objects, we remove distractors from the scene and place the transparent objects at 6 different poses on the table, resulting in 6 trials per object. We visualize transparent objects in Figure 12.

![](images/13.jpg)  
Figure 12: Transparent objects used for evaluation.

Analysis. In the language-conditioned test set, the baseline fails in 5 trials. Three failures stem from incorrect bounding box predictions by Grounding DINO, largely due to ambiguities in the top-down monocular view—for example, a toy ambulance being misidentified as a charger. The remaining two failures involve flat objects (a metal fork and a plastic spoon), where the point clouds merge with the table surface, rendering the objects indistinguishable even to human observers. Transparent objects pose a similar challenge, as missing depth information leads to point-cloud-based grasping failures. However, since RGB images reliably capture these objects, our RGB-based model overcomes these limitations and succeeds where the baseline fails.

Overall, AnyGrasp and our method provide complementary solutions. AnyGrasp is a fast grasping detection model and adapting it to open-vocabulary grasping requires extra modules (segmentation, motion planner, failure recovery)—each introducing potential failures. For instance, collision-free path planning often fails in cluttered scenes like our post-train Task 3. In contrast, our model is end-to-end, closed-loop, and easily adapts to specialized tasks (e.g., grasping in specific poses) without requiring task-specific modules. Besides, AnyGrasp uses depth as input, which suffers from incomplete and noisy issues for transparent materials. In contrast, our model relies solely on RGBs, bypassing this issue.

# I Ablation of Camera Views

$\pi _ { 0 }$ natively supports multi-camera, so we fine-tune it with the same front and side views as our model to ensure fair comparison. However, OpenVLA only supports single view, so we ablate the number of views here and show that our single-view version outperforms OpenVLA by $40 \%$ .

Impact of the Number of Input Views. To ensure a fair comparison, we use only front-view images as input for our method, consistent with the single-view baseline OpenVLA. As shown in Table 14, this constraint results in approximately $30 \%$ lower performance compared to our multiview approach. Nevertheless, our model still achieves $40 \%$ higher performance than OpenVLA, demonstrating the effectiveness of our design.

<table><tr><td>Model</td><td>Synthetic</td><td>Web</td></tr><tr><td>OpenVLA (single-view)</td><td>20.0</td><td>3.3</td></tr><tr><td>Ours (single-view)</td><td>60.0</td><td>56.6</td></tr><tr><td>Ours (multi-view)</td><td>93.3</td><td>93.3</td></tr></table>

Table 14: Impact of number of input views. Comparison of GraspVLA with different numbers of input views. The results demonstrate that while multiple views significantly improve performance, our single-view implementation still outperforms the OpenVLA baseline by $40 \%$ .

# J Mitigation of Sim-to-Real Gap

In this section, we examine the sim-to-real gap in the context f taining a VLA model or graspin using imitation learning. The sim-to-real gap primarily appears in two key areas: visual appearance and physical dynamics.

Visual appearance. Thanks to advances in pre-trained vision encoders and ray-traced rendering, the visual discrepancy between synthetic and real-world RGB images has significantly narrowed. By leveraging diverse material and texture datasets, we can generate realistic scenes that cover a wide range of robotic grasping scenarios—far more efficiently than collecting equivalent real-world data across varied environments (as discussed in B). Even when certain material or texture combinations appear unrealistic (e.g., a red table against a green wall), the model still learns generalizable representations from such diversity, consistent with findings in [47]. Additionally, co-training with large-scale Internet vision-language datasets further enhances the model's robustness to visual discrepancies [36].

Physical dynamics. The sim-to-real gap in physical dynamics arises mainly from inaccuracies in modeling material properties (e.g., surface friction), contact dynamics (e.g., forces, friction, deformations), and actuator/sensor behavior. In this work, we mitigate this gap through three key design choices:

• Simplified control. We use positional control and treat gripper actions as discrete open/close commands, avoiding complex dynamics modeling.   
Stability filtering. We only keep grasps that forms force-closure under low friction coeffcient (0.15), ensuring the model prioritizes robust strategies.   
•Geometry-driven planning. We focus on mesh-based grasp poses rather than dynamicsdependent policies, enhancing robustness to physical variations.

While these strategies effectively reduce the sim-to-real gap for grasping, they may not generalize to tasks requiring fine-grained dynamics understanding, such as non-prehensile manipulation. We leave the investigation of such scenarios to future work.

# K Inference Delay

The combination of autoregression and flow matching in GraspVLA introduces additional inference delay. Based on Section 5.6 and Table 15, while PAG is critical for a high grasp success rate, it contributes to $\sim 6 3 \%$ inference delay due to 14 additional tokens to generate. We leave the further improvement of the inference efficiency with PAG as future work. We additionally found that the prefill stage has a similar delay as the decode stage, which could be due to a low GPU utilization with single-sample inference.

<table><tr><td>component</td><td>inference time (ms)</td></tr><tr><td>vision encoder</td><td>9</td></tr><tr><td>bounding boxes (8 tokens)</td><td>72</td></tr><tr><td>grasp pose (6 tokens)</td><td>50</td></tr><tr><td>flow matching</td><td>64</td></tr></table>

Table 15: Breakdown of inference time on NVIDIA L40s GPU.

# L Non-Blocking Control

We explore the implementation of non-blocking controller for smooth action. We implement a Cartesian-space impedance controller adapted from Franka ROS [87] and SERL Franka Controllers [88]. The architecture converts Cartesian impedance commands into joint-space control via realtime Jacobian-based transformation with singularity handling, while optimizing impedance parameters through system identification.

To mitigate abrupt target transitions, we evaluated multiple filter implementations and selected a cascaded filter design for its superior smoothing performance (Figure 13). It achieves fast convergence without overshoot while avoiding excessive initial aceleration, which is suitable for the output characteristics of the GraspVLA model. Additionally, positional interpolation was adopted instead of temporal interpolation to address synchronization mismatches between model computation latency and control pipelines.

To achieve more fluent and coherent motions, GraspVLA generates multi-step predictions at each inference cycle. These predictions are incorporated via a receding-horizon optimization scheme within the asynchronous control architecture, where filter and interpolation strategies are systematically applied to the predicted trajectory. This non-blocking control architecture proactively compensates for computational latency variations while ensuring smooth interleaving of control actions, which significantly mitigates oscillatory patterns in dynamic manipulation scenarios.

While we employ non-blocking control for demonstration recording to achieve natural trajectories, all experimental evaluations use blocking control to ensure rigorous performance measurement.

![](images/14.jpg)  
Figure 13: Step response comparison for different filters. In the comparison, these filters set the same sampling frequency and cutoff frequency. The first-order Butterworth filter exhibits a large initial acceleration in its step response. The third-order Butterworth filter, third-order Chebyshev ⅡI filter, and third-order Bessel filter all exhibit overshoot to varying degrees. The triple-cascaded first-order Butterworth filter can avoid excessive initial acceleration and eliminate overshoot while maintaining convergence speed, making it an ideal filter choice.

# M Failure Analysis

To thoroughly assess the limitations of our approach, we conduct a detailed failure analysis. Since the test set in the main paper reveals only two failure cases—which may not be representative—we design a significantly more challenging test set featuring cluttered scenes. Specifically, we randomly place objects across the table to cover the entire workspace and stack some objects (e.g., placing a strawberry on top of a bulldozer) to create complex, occluded scenarios. We then evaluate the model's performance under these conditions and identify the primary failure modes.

The most frequent failure case $( 3 1 \% )$ ocurs when the model hesitates due to ambiguous language instructions, such as when multiple objects match the description (e.g., two target bottles). This could be mitigated by incorporating longer contextual history. The second most common issue $( 2 7 \% )$ arises in highly cluttered scenes, where the model misidentifies objects, likely due to insufficient training data for such scenarios. Future work could utilize advanced data augmentation methods and generative modeling techniques to create more diverse and complex training samples. Another notable failure mode $( 2 1 \% )$ involves objects with smooth surfaces (e.g., plastic balls) slipping during grasping, which tactile feedback might help resolve. Additionally, when the target object is occluded $( 1 4 \% )$ , the model struggles to grasp it precisely, suggesting a need for active perception techniques. Finally, the remaining failures $( 7 \% )$ include minor errors such as early gripper closure or collisions with the environment, which reinforcement learning could potentially address. We leave these potential improvements for future work.