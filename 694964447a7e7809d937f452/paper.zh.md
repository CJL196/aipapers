# ReBot：通过真实-仿真-真实的机器人视频合成扩展机器人学习

余方，杨悦，朱兴昊，喀哲，吉达斯·比乌斯，丹尔·萨尔

摘要——视觉-语言-动作（VLA）模型通过直接在真实机器人数据集（如Open $\mathbf { X }$ -Embodiment）上训练策略，展现出一种有前景的范式。然而，现实世界数据收集的高成本阻碍了数据规模的进一步扩展，从而限制了VLA的通用性。本文介绍了ReBot，一种新颖的真实-仿真-真实方法，用于扩展真实机器人数据集并将VLA模型适应于目标领域，这是机器人操作中的“最后一公里”部署挑战。具体而言，ReBot在仿真中重放真实机器人轨迹，以多样化操控对象（真实到仿真），并将模拟运动与修复后的现实背景结合，合成物理上逼真且时序一致的机器人视频（仿真到真实）。我们的方法具有几个优势：1）利用真实数据的优势，最大限度地减少仿真到真实的差距；2）利用仿真的可扩展性；3）可以使用完全自动化的数据管道，将预训练的VLA推广到目标领域。在仿真和真实环境中的广泛实验表明，ReBot显著提升了VLA的性能和鲁棒性。例如，在使用WidowX机器人的SimplerEnv中，ReBot使得Octo的领域内性能提高了$7 . 2 \%$，OpenVLA提高了$21 . 8 \%$，而领域外泛化性能分别提高了$19 . 9 \%$和$9 . 4 \%$。在与Franka机器人进行的现实世界评估中，ReBot使得Octo的成功率提高了$17 \%$，OpenVLA提高了$20 \%$。更多信息请访问我们的项目页面。

# I. 引言

大规模真实机器人数据集显著促进了机器人学习的快速进展，使得视觉-语言-动作（VLA）模型能够在多种任务、环境和表现形式中进行学习。尽管取得了这些成就，VLA在有效地推广到新场景方面仍然面临挑战，这促使人们需要扩展数据以提升它们在新目标领域的表现。然而，收集大规模真实机器人数据集的成本非常高，通常需要大量的人力和资源，例如机器人和人类遥控操作员，这大大限制了数据的可用性和可扩展性。另一方面，模拟数据集是更易获取和成本效益更高的替代方案，因为它们可以在模拟环境中生成，而无需现实世界的设置。不幸的是，动作空间和观察空间中的仿真与现实之间的差距阻碍了机器人策略向现实应用的推广，限制了模拟数据在推进VLA方面的有效性。

![](images/1.jpg)  
Fig. 1. An overview of ReBot. We propose ReBot, a novel real-tosim-to-real approach for scaling real robot datasets. ReBot replays realworld robot trajectories in a simulation environment to diversify manipulated objects (real-to-sim), and integrates the simulated movements with inpainted real-world background to produce realistic synthetic videos (sim-to-real), effectively adapting VLA models to target domains.

为了解决这些挑战，扩展机器人学习的简单策略是从真实机器人数据集中生成合成机器人视频。随着计算机视觉和生成AI基础模型的快速发展，研究人员提出了用于合成机器人视频生成的生成模型。例如，方法[17 19]利用文本到图像修复将真实机器人图像扩展到多样化场景。然而，它们通常面临AI生成伪影的问题，例如可见缺陷或不一致的纹理，无法产生物理上真实和时间上一致的机器人视频。这些失真引入了新的领域差距，使得VLA难以学习稳定和连续的机器人动作，并引发可靠性担忧。此外，生成的图像可能无法严格遵循指令条件，限制了这类方法在将VLA适应特定目标领域的有效性，留下了机器人操作中的最后一公里部署挑战未解决。为了解决这些问题，我们提出了ReBot，这是一种新颖的真实到仿真再到真实的方法，用于扩展真实机器人数据集并将VLA模型适应于目标领域。我们的关键观点是在仿真中重放真实世界的机器人轨迹，以多样化操控对象（真实到仿真），并将模拟的运动与修复的真实世界背景相结合（仿真到真实），以合成物理上真实和时间上一致的机器人视频。值得注意的是，ReBot结合了仿真和真实的优势，即利用仿真的可扩展性，同时通过基于真实机器人数据扎根于行动和观察空间来最小化仿真到真实的区别。特别是，与基于生成的扩展方法相比，ReBot确保物理真实性和时间一致性，并有效地将VLA模型适应于目标领域。具体而言，如图1所示，ReBot包括三个关键组成部分：1）真实到仿真的轨迹重放。对于每个真实世界的实验，我们在仿真环境中自动设置数字双胞胎，并重放真实世界的机器人轨迹，以获取操控新对象的模拟运动。我们通过证明真实世界的轨迹可以成功重用以操控仿真中不同形状的对象，从而验证我们方法的可扩展性。2）真实世界背景修复。为了获得任务无关的真实世界背景用于视频合成，我们引入了一个自动修复模块，使用GroundedSAM2 [20]在原始真实世界视频中对机器人和物体（即任务特定元素）进行分割和追踪，并使用ProPainter [21]去除它们。3）仿真到真实视频合成。我们最终将模拟运动与任务无关的真实世界背景整合，生成具有真实物理和优异时间一致性的合成视频。总之，我们的关键贡献有三方面。我们引入了ReBot，至今为止，这是第一个真实到仿真再到真实的方法，用于扩展真实机器人数据集并将VLA模型适应于目标领域，解决了机器人操作中的最后一公里部署挑战。ReBot结合了仿真和真实的优势，即利用仿真的可扩展性，同时通过将行动和观察空间扎根于真实机器人数据来最小化仿真到真实的差距。值得注意的是，ReBot是完全自动化的，无需人工干预。大量评估确认了ReBot在仿真和真实世界环境中的有效性，例如，它在SimplerEnv上提高了OpenVLA的领域内和泛化性能分别为$2 1 . 8 \%$和$9 . 4 \%$，并在真实世界任务中实现了$20 \%$的增益，显著超越了之前的最先进技术ROSIE [18]。

# II. 相关工作

扩展机器人学习。尽管许多研究机构已合作构建大规模真实机器人数据集，但数据规模仍然是VLA模型的一个根本瓶颈。为了解决这个问题，最近的研究探索了三种主要策略：1）在现实环境中收集数据，2）在仿真环境中收集数据，以及3）利用生成模型扩展真实机器人数据集。真实机器人数据集可以通过多种方法获取，包括运动教学、遥控操作或混合现实设备，并在VLA模型的近期进展中作出了重大贡献。然而，收集大规模真实机器人数据集需要大量的资源，这使得在不同环境和任务中扩展变得非常具有挑战性。这一限制阻碍了VLA模型的泛化性能。另一方面，仿真数据集提供了更具可扩展性的替代方案。完善的仿真平台便于在受控环境中快速收集数据，而无需承担真实世界实验的高成本。不幸的是，这些数据集往往引入显著的仿真与现实间的差距，限制了它们在现实应用中的有效性。值得注意的是，最近的研究探索了生成模型以扩展真实机器人数据集。然而，这些方法通常难以提供在物理上现实且时间上连贯的机器人视频，使得它们在开发VLA模型时不可靠且无效。在本文中，我们提出了一种真实-仿真-真实的策略，用于扩展真实机器人数据集，为这些长期存在的挑战提供了一种新解决方案。真实-仿真与仿真-现实。真实-仿真和仿真-现实策略已在机器人领域的许多应用中得到探索。值得注意的是，最近的研究利用真实-仿真-真实策略开发了机器人评估的仿真平台，展示了与真实机器人评估的强相关性。这些研究强调了真实-仿真-真实方法在弥合仿真与现实环境之间差距方面的巨大潜力。然而，现有方法由于场景和物体多样性受限，常面临可扩展性挑战，这主要归因于在仿真环境中构建数字双胞胎所需的巨大人工努力。在本文中，我们探索了这一策略的新应用，即扩展真实机器人数据集，能够在无需人工干预的情况下实现逼真的机器人视频生成。

# III. 方法

在本文中，我们提出了一种新颖的从真实到仿真再到真实的方法，以扩展真实机器人数据集。我们将真实机器人数据集定义为 $\mathcal { D } = \{ \tau _ { i } \} _ { i = 1 } ^ { M }$，其中 $M$ 个回合表示为 $\tau _ { i } ~ = ~ \{ \mathbf { o } _ { t } , \mathbf { a } _ { t } , \mathcal { L } \} _ { t = 1 } ^ { T }$。这里，$t$ 表示时间步，$\mathbf { o } _ { t }$ 是视频帧，$\mathbf { a } _ { t }$ 是动作，$\mathcal { L }$ 是语言指令。我们的目标是基于 $\tau _ { i }$ 生成新的合成回合 $\tau _ { j } ^ { \prime } = \{ \mathbf { o } _ { t } ^ { \prime } , \mathbf { a } _ { t } , \mathcal { L } ^ { \prime } \} _ { t = 1 } ^ { T }$，从而得到 $\bar { \mathcal { D } } ^ { \prime } = \{ \tau _ { j } ^ { \prime } \} _ { j = 1 } ^ { N }$。如图 2 所示，ReBot 具有三个关键步骤：A) 从真实到仿真的轨迹回放，以获取仿真运动 $\{ \mathbf { o } _ { t } ^ { \mathrm { s i m } } \} _ { t = 1 } ^ { T }$，对视频帧 $\{ \mathbf { o } _ { t } \} _ { t = 1 } ^ { T }$ 进行世界背景修复，以获得 $\{ \mathbf { o } _ { t } ^ { \mathrm { r e a l } } \} _ { t = 1 } ^ { T }$ （第 I B 节）；并最终进行 C) 从仿真到真实的视频合成，以获得新的帧 $\{ \mathbf { o } _ { t } ^ { \prime } \} _ { t = 1 } ^ { T }$（第 III-C 节）。

# A. 真实到模拟的轨迹重放

真实到仿真过程包括：1) 在仿真环境中创建空间对齐的场景数字双胞胎，2) 重放真实世界的机器人轨迹以生成仿真机器人运动 $\{ \mathbf { o } _ { t } ^ { \mathrm { s i m } } \} _ { t = 1 } ^ { T }$，3) 验证每条重放轨迹以确保成功的物体操作。

![](images/2.jpg)  
manual intervention.

场景解析与对齐。为了确保准确的轨迹重放，我们构建了机器人、相机和桌子的数字双胞胎，并将它们对齐到初始视频帧 $\mathbf{o}_1$。机器人的原型和相机已提前准备，仅需进行姿态调整以完成其设置。为了确定桌子的高度，我们从初始视频帧 $\mathbf{o}_1$ 获取度量深度，并创建场景的点云。使用 GroundingDINO [40]，我们自动地以文本提示（“桌子”）对桌子进行分割，并在使用四分位距去除离群点后提取点云的子集。最终，我们将过滤后的点的平均高度设定为桌子的高度。

轨迹回放。我们重复使用真实世界的轨迹来多样化操作对象。首先，为了确保机器人能够成功到达模拟对象，我们需要将其放置在原始真实对象的位置。我们分析夹持器的动作序列，以确定 $t _ { \mathrm { s t a r t } }$（夹持器闭合以抓取对象时）和 $t _ { \mathrm { e n d } }$（夹持器打开以放置对象时）。为了估计对象位置，我们通过重放 $\mathbf { \bar { \{ a } } _ { t } \} _ { t = 1 } ^ { t _ { \mathrm { s t a r t } } }$ 获取 $t _ { \mathrm { s t a r t } }$ 时刻的夹持器位置，并将模拟对象相应地放置。同样，作为选择，我们在 $t _ { \mathrm { e n d } }$ 时刻的夹持器位置在桌子上放置一个容器。最后，我们使用动作序列 $\{ \mathbf { a } _ { t } \} _ { t = 1 } ^ { T }$ 重放机器人轨迹，并记录模拟运动 $\{ \mathbf { o } _ { t } ^ { \mathrm { s i m } } \} _ { t = 1 } ^ { T }$，确保所有数字双胞胎与真实世界场景忠实对齐，从而确保记录的运动与真实世界背景保持一致。回放验证。值得注意的是，轨迹回放在操控新对象时可能成功或失败，这取决于新对象与原始真实对象之间的功能相容性。我们通过监测 $t _ { \mathrm { s t a r t } }$ 到 $t _ { \mathrm { e n d } }$ 之间对象与夹持器的笛卡尔距离，自动验证对象是否在每个合成回合中成功被操控，并丢弃失败的回合。我们在图 2 中展示了一个代表性示例，表明尽管对象形状存在差异，真实世界的轨迹仍然可以成功用于操控各种对象，证明了我们方法的可扩展性。

# B. 现实世界背景图像修复

在这一步中，我们通过去除特定任务的元素，准备与任务无关的真实背景 $\{ \mathbf { o } _ { t } ^ { \mathrm { r e a l } } \} _ { t = 1 } ^ { T }$（即原始的真实数据 $\{ \mathbf { o } _ { t } \} _ { t = 1 } ^ { T }$）。

物体与机器人分割。我们通过使用 GroundedSAM2 [20] 自动分割和跟踪原始真实物体和机器人，该方法结合了 GroundingDINO [40] 和 SAM2 [41]。更具体地，我们首先使用 GroundingDINO 通过文本提示（“机器人”）识别并分割在 $\mathbf { o } _ { t _ { \mathrm { s t a r t } } }$ 上的机器人，因为我们经验观察到当机器人最可见时性能最佳。然而，自动识别原始真实物体极具挑战性，因为通常在真实机器人数据集中缺乏有效文本提示所需的外观详细描述。此外，文本提示对干扰项或相似实例高度敏感，使得它们在精确定位被操纵物体时不可靠。幸运的是，物体在 $t _ { \mathrm { s t a r t } }$ 时的位置已经在真实到模拟的轨迹重播中进行估计，现在作为分割 $\mathbf { o } _ { t _ { \mathrm { s t a r t } } }$ 上真实物体的重要线索。我们使用相机姿态将3D物体位置投影到 $\mathbf { o } _ { t _ { \mathrm { s t a r t } } }$ 上，提供用于真实物体分割的2D点提示，结合SAM2。获取语义掩膜 $\mathbf { m } _ { t _ { \mathrm { s t a r t } } }$（即在 $t _ { \mathrm { s t a r t } }$ 时的机器人和物体掩膜）后，使用 SAM2 将其传播到所有视频帧 $\{ \mathbf { o } _ { t } \} _ { t = 1 } ^ { T }$，生成对应的语义掩膜 $\{ \mathbf { m } _ { t } \} _ { t = 1 } ^ { T }$。物体与机器人移除。给定 $\{ \mathbf { o } _ { t } , \mathbf { m } _ { t } \} _ { t = 1 } ^ { T }$，最终我们应用 ProPainter [21]，一种最先进的视频修复模型，从原始视频中移除原始真实物体和机器人，获得非特定任务的背景 $\{ \mathbf { o } _ { t } ^ { \mathrm { r e a l } } \} _ { t = 1 } ^ { T }$，并在后续步骤中使用我们合成视频中的虚拟机器人 $\{ \mathbf { o } _ { t } ^ { \prime } \} _ { t = 1 } ^ { T }$ 进行物体操作过程中的物理交互。

![](images/3.jpg)

# C. 模拟到真实视频合成 $\{ \mathbf { o } _ { t } ^ { \mathrm { s i m } } \} _ { t = 1 } ^ { T }$ $\{ \mathbf { o } _ { t } ^ { \mathrm { r e a l } } \} _ { t = 1 } ^ { T }$ t=1 以构建新的视频帧 $\{ \mathbf { o } _ { t } ^ { \prime } \} _ { t = 1 } ^ { T }$。具体而言，为了获得 $\mathbf { o } _ { t } ^ { \prime }$，我们从 $\mathbf { o } _ { t } ^ { \mathrm { { s i m } } }$ 中提取机器人及其操作，并将它们合并到 ${ \bf o } _ { t } ^ { \mathrm { r e a l } }$ 上。然后，我们通过将原始指令 $\mathcal { L }$ 中的物体（例如“黄色杯子”替换为“勺子”）和容器（例如“桌子”替换为“毛巾”）来分配一个新的语言指令 $\mathcal { L } ^ { \prime }$，构成新的轨迹 $\tau _ { j } ^ { \prime } = \{ \mathbf { o } _ { t } ^ { \prime } , \mathbf { a } _ { t } , \mathcal { L } ^ { \prime } \} _ { t = 1 } ^ { T }$。请注意，由于我们忠实地重放真实世界的机器人轨迹，因此在合成情节中，真实世界的动作保持不变。在我们的实验中（见第 IV 节），我们验证了我们的方法在使用合成数据集 $\mathcal { D } ^ { \prime } = \{ \tau _ { j } ^ { \prime } \} _ { j = 1 } ^ { N }$ 适应 VLA 模型的有效性。

# IV. 实验

在本节中，我们评估并证明 ReBot 有效地产生高保真合成机器人视频（第 IV-B 节），并在模拟环境（第 IV-C 节）和现实环境（第 IV-D 节）中全面提升 VLA 模型的性能。

# A. 实验设置

数据集。对于真实机器人数据集，我们利用了BridgeData V2 [42] 和 DROID [5] 中的桌面拾取和放置场景。为了在第IV-D节中对真实环境进行评估，我们收集了220个真实场景来构建我们的数据集。在DROID数据集中，我们利用了从机器人对侧拍摄的两个外部视频。对于在真实到仿真的轨迹回放中使用的模拟物体，我们参考了[11, 39]，并从Objaverse [43] 收集了厨房资产。

实施细节。我们使用 Isaac $\mathrm { S i m } 4 . 1$ 作为模拟环境，因为它具有出色的渲染质量和灵活性。我们基于 Isaac Lab [34] 实现了真实到模拟的轨迹重放。我们在 Isaac Sim 中预构建机器人的数字双胞胎，匹配与真实机器人数据集相同的机器人平台，例如，使用 WidowX 250 六自由度机器人臂用于 BridgeData V2，使用 Franka Panda 七自由度机器人臂和 Robotiq 2F-85 夹持器用于 DROID 和我们的数据集。根据 Octo 和 OpenVLA 的官方指南，我们为每个任务使用 100 个合成回合作为微调的最优数据量。我们使用四个 NVIDIA A6000 GPU，针对 Octo 进行全量微调，批量大小为 256，学习率为 $4 \times 1 0 ^ { - 5 }$；针对 OpenVLA 进行 LoRA 微调，批量大小为 32，学习率为 $5 \times 1 0 ^ { - 4 }$。

比较方法。我们将ReBot与ROSIE [18]进行比较，后者是一种最先进的基于生成的方法，用于扩展真实机器人视频。ROSIE采用基于图像的基础模型，使用Imagen [44]直接在原始真实机器人视频上对操控对象进行修补。相比之下，ReBot引入了一种新颖的从真实到仿真再到真实的扩展策略，生成物理上真实且时间上一致的合成机器人视频。由于ROSIE不是开源的，我们使用基于稳定扩散模型 [45] 的实现进行比较。使用VLA模型进行评估。我们评估合成视频在将VLA模型适应于目标领域上的有效性。我们主要讨论两种最先进的VLA模型，Octo [29] 和 OpenVLA [30]，这两种模型均在包含多种机器人实现的丰富大规模数据集上进行训练 [4]。为了比较扩展方法，我们评估每个VLA模型的三个版本：1）Octo 和 OpenVLA（零样本评估，即未微调的预训练模型），2）Octo+ROSIE 和 OpenVLA $+$ ROSIE（使用来自ROSIE的场景进行微调），3）Octo+ReBot 和 OpenVLA $+$ ReBot（使用来自ReBot的场景进行微调）。

# B. 视频质量评估

我们从三个方面比较了ROSIE和ReBot生成视频的质量：时间质量、成像质量。

![](images/4.jpg)  
Fig. 4. Quantitative comparison of generated video quality. We report VBench scores as evaluation metrics. ReBot outperforms ROSIE and achieves video quality comparable to original real-world videos.

质量和多视图一致性。我们在图3中展示了定性比较。同时，如图4所示，我们使用VBench [46]，这是一个用于评估视频生成质量的全面基准工具，来评估四个维度中的两个关键方面（具体定义请参考[46]）：1）时间质量 - 包括主体一致性、背景一致性和运动平滑性；2）逐帧质量，即成像质量。我们还评估了原始真实视频以供参考。

时间质量。尽管ROSIE提供了一个简单的解决方案，但它未能生成时间一致的视频，这妨碍了VLA模型学习稳定的动作。如图3的第一个示例所示，ROSIE在前两帧中最初生成了一个 plausible 的可乐罐，但随后未能保持一致性，在后面的帧中产生了不相关的瓶子。这一限制在其仅为 $6 5 . 6 \%$ 的低主体一致性得分中得到了进一步体现，如图4所示。因此，尽管观察历史已被证明可以增强VLA模型的性能[1, 29]，但ROSIE仍不适合提高它们从连续帧中学习的能力。相比之下，ReBot通过模拟过程本质上确保了出色的时间一致性，实现了 $9 9 . 2 \%$ 的运动平滑度。令人惊讶的是，这甚至比真实机器人视频高出 $0 . 2 \%$，这可能是因为模拟过程减少了运动模糊等伪影（见图3第二示例的第二帧）。此外，现实世界的背景修复忠实地利用时间上下文来恢复遮挡，贡献了 $9 2 . 2 \%$ 的背景一致性。值得注意的是，我们在所有维度上的时间质量，平均得分为 $9 3 . 0 \%$，与真实机器人视频 $( 9 6 . 1 \% )$ 高度可比，表明我们的合成视频实现了逼真的时间一致性。成像质量。在图3中，ROSIE在生成高质量操控对象方面表现不佳，尤其是在最后两个示例中。当新对象形状可能偏离原始对象形状时，这一问题尤为明显。这是因为生成模型往往更加依赖修复掩膜，而对文本提示的指导关注较少。相比之下，ReBot通过模拟确保物理上合理的运动，同时在图4中展示了出色的成像质量，与原始视频相比仅下降了 $3 . 7 \%$，而超越了ROSIE $1 3 . 0 \%$。

![](images/5.jpg)  
Fig. 5. Comparisons of multi-view consistency. We present two examples from the DROID dataset, each captured from two different camera views. While ROSIE lacks multi-view consistency, ReBot naturally preserves this capability inherited from 3D simulation, ensuring the same object in different camera views, as in the real world.

多视角一致性。此外，如图5所示，ReBot 本质上在多个摄像头视角之间保持多视角一致性，因为合成视频是在三维环境中生成的。值得注意的是，这一关键属性通过我们的真实到仿真再到真实的缩放方法独特地得以实现。

# C. 在仿真环境中的评估

我们首先在 SimplerEnv [39] 中评估 VLA 模型及其两个微调版本 $\mathrm { ^ { 6 6 } { + } R O S I E ^ { 3 3 } }$ 和 "+ReBot"。为了公平比较，我们使用 ROSIE 和 ReBot 对相同的数据量进行评估任务的缩放（即，每个任务 100 个回合），将 VLA 模型适应到相同的目标领域。我们证明 ReBot 在三个关键方面有效改善了 VLA 的性能：1) 域内性能：在给定任务上的直接评估；2) 泛化性能（参考 [30, 47]）：评估在未见物体尺寸（物理）、未见指令（语义）和未见对象（主体）上进行的域内任务变体；3) 跨躯体性能：在一个躯体上进行评估，同时在另一个躯体上进行微调。

领域内性能。在表 I 中，我们报告了 WidowX 机器人在四个 SimplerEnv 任务上的抓取率（任务中成功抓取物体的百分比）和成功率（完成任务的百分比）。在开箱即用的情况下，Octo 和 OpenVLA 在大多数任务上表现不佳。尤其是，OpenVLA 在具有挑战性的任务上完全失败，成功率为 $0 . 0 \%$（例如，将绿色立方体叠加在黄色立方体上）。这表明它们在目标领域中的性能不佳，尽管在最先进的数据集上进行了广泛训练[3]。与此同时，ROSIE 在大多数任务上的表现也很差，成功率为 $0 . 0 \%$，因为它无法生成现实的操作物体，更重要的是，缺乏时间一致性。这个限制对于 Octo 尤为严重，因为它依赖于两个连续帧的观察历史。相反，ReBot 在所有模型上实现了最佳性能，使 Octo 的平均成功率提高了 $7 . 2 \%$，提高了 OpenVLA 的平均成功率 $2 1 . 8 \%$。值得注意的是，ReBot 将 OpenVLA 的平均抓取率从 $1 4 . 6 \%$ 提升至 $5 9 . 4 \%$，进一步证明了其有效性。这些结果强调了两个 VLA 模型因 ReBot 的时间一致性和物理真实感合成视频而受益匪浅。表 I 在 WidowX 机器人上 SimplerEnv 评估结果的比较。

<table><tr><td rowspan="2">Model</td><td colspan="2">Put spoon on towel</td><td colspan="2">Put carrot on plate</td><td colspan="2">Stack green cube on yellow cube</td><td colspan="2">Put eggplant in basket</td><td colspan="2">Average</td></tr><tr><td>Grasp</td><td>Success</td><td>Grasp</td><td>Success</td><td>Grasp</td><td>Success</td><td>Grasp</td><td>Success</td><td>Grasp</td><td>Success</td></tr><tr><td>Octo [29]</td><td>34.7%</td><td>12.5%</td><td>52.8%</td><td>8.3%</td><td>31.9%</td><td>0.0%</td><td>66.7%</td><td>43.1%</td><td>46.5%</td><td>16.0%</td></tr><tr><td>Octo+ROSIE [18]</td><td>20.8%</td><td>2.8%</td><td>27.8%</td><td>0.0%</td><td>18.1%</td><td>0.0%</td><td>22.3%</td><td>0.0%</td><td>22.3%</td><td>0.7%</td></tr><tr><td>Octo+ReBot (Ours)</td><td>61.1%</td><td>54.2%</td><td>41.1%</td><td>22.0%</td><td>63.9%</td><td>4.2%</td><td>52.8%</td><td>12.5%</td><td>54.7%</td><td>23.2%</td></tr><tr><td>OpenVLA [30]</td><td>4.2%</td><td>0.0%</td><td>33.3%</td><td>0.0%</td><td>12.5%</td><td>0.0%</td><td>8.3%</td><td>4.2%</td><td>14.6%</td><td>1.1%</td></tr><tr><td>OpenVLA+ROSIE [18]</td><td>12.5%</td><td>0.0%</td><td>41.7%</td><td>0.0%</td><td>50.0%</td><td>0.0%</td><td>20.8%</td><td>0.0%</td><td>31.3%</td><td>0.0%</td></tr><tr><td>OpenVLA+ReBot (Ours)</td><td>58.3%</td><td>20.8%</td><td>45.8 %</td><td>12.5%</td><td>66.7%</td><td>4.2%</td><td>66.7%</td><td>54.2%</td><td>59.4%</td><td>22.9%</td></tr></table>

![](images/6.jpg)  
generalization types (physical, semantics, and subject) on WidowX Robot in SimplerEnv.

![](images/7.jpg)  
Fig. 7. Evaluation of cross-embodiment performance. ReBot enhances the cross-embodiment performance of OpenVLA on the WidowX robot (top) and Google Robot (bottom) in SimplerEnv.

泛化性能。尽管当前的变换学习模型（VLA）经常面临泛化挑战，我们进一步验证了ReBot作为提升其泛化性能的有效扩展方案。如图6所示，ROSIE在Octo上仍然无效，而ReBot在所有三种泛化类型中始终提升了Octo和OpenVLA的性能。具体而言，ReBot将Octo的平均成功率从$6.5\%$提高到$26.4\%$。另一方面，尽管OpenVLA在SimplerEnv中面临更大的挑战，但得益于ReBot，其平均抓取率从$15.6\%$显著提高到$66.8\%$，平均成功率则从$0.7\%$增加到$11.1\%$。这些结果进一步确认了ReBot在改善VLA模型泛化性能方面的有效性。

跨体表现。我们还 investigate ReBot 是否能够提升 VLA 模型的跨体表现。具体来说，我们使用 ROSIE 和 ReBot 来扩展 Franka Panda 机器人的 DROID 数据集，然后对 OpenVLA 进行微调，并在 SimplerEnv 中评估其在 WidowX 机器人和 Google 机器人上的表现。我们在图 7 中报告了成功率。在 WidowX 机器人上，尽管 ROSIE 仅将平均成功率从 $1.4\%$ 提升到 $3.1\%$，但 ReBot 实现了 $12.5\%$ 的显著提升。在 Google 机器人上执行“拿可乐罐”任务时，ReBot 在所有姿势上都展示了一致的改善，而 ROSIE 则未能展现出这样的鲁棒性。这突显了 ReBot 使 OpenVLA 能够学习在不同物体姿势下更精确且可适应的操控策略。值得注意的是，尽管针对不同的体现进行扩展，ReBot 始终提升了 OpenVLA 的表现，展示了其增强跨体表现的能力。

# D. 真实环境中的评估

在实际实验中，我们证明了ReBot持续提升VLA模型的效能，表现优于ROSIE。正如表II所示，我们利用ROSIE和ReBot对我们的真实机器人数据集进行扩展，以应对四项评估任务（请参见表II底部的示例），并比较他们微调后的VLA模型的性能。为了确保更好地适应我们的真实场景，我们在所有模型的微调过程中也纳入了我们的真实机器人数据集（即220个真实世界的 episode）。每个任务我们进行10次试验，并报告抓取率和成功率作为评估指标。虽然ROSIE提供了边际改善，将Octo的平均成功率从$8\%$提高到$10\%$，但在某些任务上完全失败（例如，将葡萄放入黄色盘子），并且对OpenVLA没有显示出有意义的提升。相比之下，ReBot在各种任务中持续实现显著性能提升，使Octo的平均成功率提高了$17\%$，OpenVLA提高了$20\%$。值得注意的是，对于那些Octo最初抓取率和成功率均为$0\%$的困难任务（例如，将胡萝卜放入蓝色盘子），ReBot将抓取率提升至$40\%$，成功率提升至$20\%$，凸显了其在真实世界应用中的强大有效性。表II 在真实环境中FRANKA PANDA机器人评估结果的比较。

<table><tr><td rowspan="2">Model</td><td colspan="2">Put carrot in blue plate</td><td colspan="2">Put grape in yellow plate</td><td colspan="2">Put fanta can in blue plate</td><td colspan="2">Put black cube in yellow plate</td><td colspan="2">Average</td></tr><tr><td>Grasp</td><td>Success</td><td>Grasp</td><td>Success</td><td>Grasp</td><td>Success</td><td>Grasp</td><td>Success</td><td>Grasp</td><td>Success</td></tr><tr><td>Octo [29]</td><td>0%</td><td>0%</td><td>30%</td><td>20%</td><td>10%</td><td>0%</td><td>20%</td><td>10%</td><td>15%</td><td>8%</td></tr><tr><td>Octo+ROSIE [18</td><td>30%</td><td>20%</td><td>0%</td><td>0%</td><td>20%</td><td>20%</td><td>10%</td><td>0%</td><td>15%</td><td>10%</td></tr><tr><td>Octo+ReBot (Ours)</td><td>40%</td><td>20%</td><td>40%</td><td>30%</td><td>30%</td><td>20%</td><td>30%</td><td>30%</td><td>35%</td><td>25%</td></tr><tr><td>OpenVLA [30]</td><td>30%</td><td>20%</td><td>30%</td><td>20%</td><td>60%</td><td>30%</td><td>40%</td><td>30%</td><td>40%</td><td>25%</td></tr><tr><td>OpenVLA+ROSIE [18</td><td>10%</td><td>0%</td><td>10%</td><td>0%</td><td>30%</td><td>10%</td><td>20%</td><td>10%</td><td>18%</td><td>5%</td></tr><tr><td>OpenVLA+ReBot (Ours)</td><td>40%</td><td>40%</td><td>50%</td><td>40%</td><td>50%</td><td>50%</td><td>60%</td><td>50%</td><td>50%</td><td>45%</td></tr></table>

![](images/8.jpg)  
Put carrot in blue plate

![](images/9.jpg)  
Put grape in yellow plate

![](images/10.jpg)  
Put fanta can in blue plate

![](images/11.jpg)  
Put black cube in yellow plate

# V. 结论与讨论

我们提出了ReBot，这是一种新颖的真实-仿真-真实方法，用于扩展真实机器人数据集并将VLA模型适应目标领域。ReBot在仿真中重播真实世界的机器人轨迹，以多样化操控物体，并将模拟的运动与修复的真实世界背景相结合，合成出物理上真实且时间上一致的机器人视频。ReBot在视频生成质量上表现优异，VBench时间一致性得分为$93.0\%$，成像质量得分为$66.4\%$，与真实机器人视频的$96.1\%$和$70.1\%$相当。在使用WidowX机器人进行的SimulerEnv环境中，ReBot将Octo的领域内性能提高了$7.2\%$，OpenVLA提高了$21.8\%$，并分别增强了$19.9\%$和$9.4\%$的泛化性能。在使用真实的Franka Panda机器人的实际环境中，ReBot将Octo的成功率提高了$17\%$，OpenVLA提高了$20\%$。我们希望ReBot能够成为一个有价值的资产，并激励未来在机器人学习中的真实-仿真-真实研究。它为未来的探索打开了几个令人兴奋的方向。例如，将ReBot扩展到多样的数据设置（例如，变化的相机设置和机器人）可能有助于跨体学习。此外，探索超出桌面操作的更具挑战性的场景也很有趣，并可能具有更广泛的现实世界应用。我们将这些方向作为未来工作的目标。

# REFERENCES

[1] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al., "Rt-1: Robotics transformer for real-world control at scale," arXiv preprint arXiv:2212.06817, 2022.   
[2] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, et al., "Rt-2: Visionlanguage-action models transfer web knowledge to robotic control," arXiv preprint arXiv:2307.15818, 2023.   
[3] A. Padalkar, A. Pooley, A. Jain, A. Bewley, A. Herzog, A. Irpan, A. Khazatsky, A. Rai, A. Singh, A. Brohan, et al., "Open $\mathbf { X } ^ { \prime }$ -embodiment: Robotic learning datasets and rt-x models," arXiv preprint arXiv:2310.08864, 2023.   
[4] A. O'Neill, A. Rehman, A. Gupta, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, et al., "Open x-embodiment: Robotic learning datasets and rt-x models," arXiv preprint arXiv:2310.08864, 2023.   
[5] A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis, et al., "Droid: A large-scale in-the-wild robot manipulation dataset," arXiv preprint arXiv:2403.12945, 2024.   
[6] E. Kolve, R. Mottaghi, W. Han, E. VanderBilt, L. Weihs, A. Herrasti, M. Deitke, K. Ehsani, D. Gordon, Y. Zhu, et al., "Ai2-thor: An interactive 3d environment for visual ai," arXiv preprint arXiv:1712.05474, 2017.   
[7] T. Mu, Z. Ling, F. Xiang, D. Yang, X. Li, S. Tao, Z. Huang, Z. Jia, and H. Su, "Maniskill: Generalizable manipulation skill benchmark with large-scale demonstrations," arXiv preprint arXiv:2107.14483, 2021.   
[8] J. Gu, F. Xiang, X. Li, Z. Ling, X. Liu, T. Mu, Y. Tang, S. Tao, X. Wei, Y. Yao, et al., "Maniskill2: A unified benchmark for generalizable manipulation skills," arXiv preprint arXiv:2302.04659, 2023.   
[9] Y. Wang, Z. Xian, F. Chen, T.-H. Wang, Y. Wang, K. Fragkiadaki, Z. Erickson, D. Held, and C. Gan, "Robogen: Towards unleashing infinite data for automated robot learning via generative simulation," arXiv preprint arXiv:2311.01455, 2023.   
10] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone, "Libero: Benchmarking knowledge transfer for lifelong robot learning," arXiv preprint arXiv:2306.03310, 2023.   
11] S. Nasiriany, A. Maddukuri, L. Zhang, A. Parikh, A. Lo, A. Joshi, A. Mandlekar, and Y. Zhu, "Robocasa: Large-scale simulation of everyday tasks for generalist robots," arXiv preprint arXiv:2406.02523, 2024.   
[12] W. Zhao, J. P. Queralta, and T. Westerlund, "Sim-to-real transfer in deep reinforcement learning for robotics: a survey," in 2020 IEEE symposium series on computational intelligence (SSCI). IEEE, 2020, pp. 737744.   
F  .T Yu .  J, "Robot learning from randomized simulations: A review," Frontiers in Robotics and AI, vol. 9, p. 799893, 2022.   
[14] Z. Mandi, H. Bharadhwaj, V. Moens, S. Song, A. Rajeswaran, and V. Kumar, "Cacti: A framework for scalable multi-task multi-scene visual imitation learning," arXiv preprint arXiv:2212.05711, 2022.   
[15] S. Zhou, Y. Du, J. Chen, Y. Li, D.-Y. Yeung, and C. Gan, "Robodreamer: Learning compositional world models for robot imagination," arXiv preprint arXiv:2404.12377, 2024.   
[16] Y. Du, S. Yang, B. Dai, H. Dai, O. Nachum, J. Tenenbaum, D. Schuurmans, and P. Abbeel, "Learning universal policies via text-guided video generation," Advances in Neural Information Processing Systems, vol. 36, 2024.   
[ Z. Chen, S. Kiami, A. Gupta, and V. Kumar, "Genaug: Retargeting bv  tivon, preprint arXiv:2302.06671, 2023.   
[18] T. Yu, T. Xiao, A. Stone, J. Tompson, A. Brohan, S. Wang, J. Singh, C. Tan, J. Peralta, B. Ichter, et al., "Scaling robot learning with semantically imagined experience," arXiv preprint arXiv:2302.11550, 2023.   
[] L. Y.Chen, C. Xu, K. Dharmarajan, M. Z. Irshad, R. Cheng, K. Keutzer, M. Tomizuka, Q. Vuong, and K. Goldberg, "Roviu: Robot and viewpoint augmentation for cross-embodiment robot learning," in Conference on Robot Learning (CoRL), Munich, Germany, 2024.   
[0 T. Re, S. Liu A. Z, J. Lin, K. i, H. o, J. hn, X. H, Y.Cn, F. Yan, Z. Zeg, H. Zha, F. Li, J. Yag, H. Li Q. Jag and L. Zhang, "Grounded sam: Assembling open-world models for diverse visual tasks," 2024.   
. o, . i ..han, an . . Ly, "ro: Ip propgation and tranformer or ido ipainting" in roceedin IEEE International Conference on Computer Vision (ICCV), 2023.   
[22] H. Ravichandar, A. S. Polydoros, S. Chernova, and A. Billard, "Recent control, robotics, and autonomous systems, vol. 3, no. 1, pp. 297 330, 2020.   
[23] Y. Yang, L. Chen, Z. Zaidi, S. van Waveren, A. Krishna, and M. Gombolay, "Enhancing safety in learning from demonstration algorithms via control barrier function shielding," in Proceedings of the 2024 ACM/IEEE International Conference on Human-Robot Interaction, 2024, pp. 820829.   
[24] A. Mandlekar, J. Booher, M. Spero, A. Tung, A. Gupta, Y. Zhu, A. Garg, S. Savarese, and L. Fei-Fei, "Scaling robot supervision to hundreds of hours with roboturk: Robotic manipulation dataset through human reasoning and dexterity," in 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2019, pp. 10481055.   
[25] F. Ebert, Y. Yang, K. Schmeckpeper, B. Bucher, G. Georgakis, K. Daniilidis, C. Finn, and S. Levine, "Bridge data: Boosting generarXiv:2109.13396, 2021.   
[26] E. Jang, A. Irpan, M. Khansari, D. Kappler, F. Ebert, C. Lynch, S. Levine, and C. Finn, "Bc-z: Zero-shot task generalization with robotic imitation learning," in Conference on Robot Learning. PMLR, 2022, pp. 9911002.   
[27] D. Whitney, E. Rosen, E. Phillips, G. Konidaris, and S. Tellex, "Comparing robot grasping teleoperation across desktop and virtual reality with ros reality," in Robotics Research: The 18th International Symposim ISRR. Springer, 2019, pp. 5350.   
[28] . Yang, B. Ikeda, G. Bertasius, and D. Szafir, "Arcade: Scalable demonstration collection and generation via augmented reality for imitation learning," in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 28552861.   
[29] Octo Model Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, C. Xu, J. Luo, T. Kreiman, Y. Tan, L. Y. Chen, P. Sanketi, Q. Vuong, T. Xiao, D. Sadigh, C. Finn, and S. Levine, Oco: An open-source generalist robot policy," in Proceedings o Robotics: Science and Systems, Delft, Netherlands, 2024.   
[30] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al., "Openvla: An onen-cource vicion-language-action model" arYiv nranrint arXiv:2406.09246, 2024.   
[31] M. Savva, A. Kadian, O. Maksymets, Y. Zhao, E. Wijmans, B. Jain, J.Straub, J. Liu, V. Koltun, J. Malik, et al., "Habitat: A platform for embodied ai research," in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 93399347.   
[32] M. Shridhar, J. Thomason, D. Gordon, Y. Bisk, W. Han, R. Mottaghi, L. Zettlemoyer, and D. Fox, "Alfred: A benchmark for interpreting grounded instructions for everyday tasks," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 1074010 749.   
[33] F. Xiag, Y. Qin, K. Mo, Y. Xia, H. Zhu, F. Liu, M. Liu, H. Jiang, Y. Yuan, H. Wang, et al., "Sapien: A simulated part-based interactive environment," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 1109711 107.   
[34] M. Mittal, C. Yu, Q. Yu, J. Liu, N. Rudin, D. Hoeller, J. L. Yuan, R. Singh, Y. Guo, H. Mazhar, et al., "Orbit: A unified simulation framework r iteactive robo learn envirents,"  Roboi and Automation Letters, vol. 8, no. 6, pp. 37403747, 2023.   
[35] L. Wang, R. Guo, Q. Vuong, Y. Qin, H. Su, and H. Christensen, "A real2sim2real method for robust object grasping with neural surface reconstruction," in 2023 IEEE 19th International Conference on Automation Science and Engineering (CASE). IEEE, 2023, pp. 18.   
[36] M. Torne, A. Simeonov, Z. Li, A. Chan, T. Chen, A. Gupta, and P. Agrawal, "Reconciling reality through simulation: A realto-sim-to-real approach for robust manipulation," arXiv preprint arXiv:2403.03949, 2024.   
[37] Y. Mu, T. Chen, S. Peng, Z. Chen, Z. Gao, Y. Zou, L. Lin, Z. Xie, and P. Luo, "Robotwin: Dual-arm robot benchmark with generative digital twins (early version)," arXiv preprint arXiv:2409.02920, 2024.   
[38] X. Li, J. Li, Z. Zhang, R. Zhang, F. Jia, T. Wang, H. Fan, K.-K. Tseng, and R. Wang, "Robogsim: A real2sim2real robotic gaussian splatting simulator," 2024. [Online]. Available: https://arxiv.org/abs/2411.11839   
[39] X. Li, K. Hsu, J. Gu, K. Pertsch, O. Mees, H. R. Walke, C. Fu, I. Lunawat, I. Sieh, S. Kirmani, et al., "Evaluating real-world robot manipulation policies in simulation," arXiv preprint arXiv:2405.05941, 2024.   
[40] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, C. Li, J. Yang, H. Su, J. Zhu, et al., "Grounding dino: Marrying dino with grounded pre-training for open-set object detection," arXiv preprint arXiv:2303.05499, 2023.   
[41] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala, N. Carion, C.-Y. Wu, R. Girshick, P. Dollár, and C. Feichtenhofer, "Sam 2: Segment anything in images and videos," 2024. [Online]. Available:https://arxiv.org/abs/2408.00714   
[42] H. Walke, K. Black, A. Lee, M. J. Kim, M. Du, C. Zheng, T. Zhao, P. Hansen-Estruch, Q. Vuong, A. He, V. Myers, K. Fang, C. Finn, and S. Levine, "Bridgedata v2: A dataset for robot learning at scale," in Conference on Robot Learning (CoRL), 2023.   
[43] M. Deitke, D. Schwenk, J. Salvador, L. Weihs, O. Michel, E. VanderBilt, L. Schmidt, K. Ehsani, A. Kembhavi, and A. Farhadi, "Objaverse: A universe of annotated 3d objects," arXiv preprint arXiv:2212.08051, 2022.   
[44] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Denton, K. Ghasemipour, R. Gontijo Lopes, B. Karagol Ayan, T. Salimans, et al., "Photorealistic text-to-image diffusion models with deep language understanding," Advances in neural information processing systems, vol. 35, pp. 36 47936 494, 2022.   
[45] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, "High-resolution image synthesis with latent diffusion models," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 1068410 695.   
[46] Z. Huang, Y. He, J. Yu, F. Zhang, C. Si, Y. Jiang, Y. Zhang, T. Wu, Q. Jin, N. Chanpaisit, Y. Wang, X. Chen, L. Wang, D. Lin, Y. Qiao, and Z. Liu, "VBench: Comprehensive benchmark suite for video generative models," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.   
[47] Z. Zhang, K. Zheng, Z. Chen, J. Jang, Y. Li, C. Wang, M. Ding, Fox, nd .Yo, "Grape Galizg robot poli  prenc alignment," arXiv preprint arXiv:2411.19309, 2024.