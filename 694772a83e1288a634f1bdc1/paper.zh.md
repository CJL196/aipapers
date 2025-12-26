# VMAS: 面向集体机器人学习的向量化多智能体模拟器

马泰奥·贝蒂尼、瑞安·科特维莱西、简·布卢门卡姆普和阿曼达·普罗罗克 剑桥大学计算机科学与技术系，英国剑桥，{mb2389,rk627,jb2270,asp45}@cl.cam.ac.uk

摘要。虽然许多多机器人协调问题可以通过精确算法实现最优解，但对于机器人数量较多的情况，解决方案往往无法扩展。多智能体强化学习（MARL）正在机器人领域受到越来越多的关注，作为解决此类问题的有希望的方案。然而，我们仍然缺乏能够快速有效地找到大规模集体学习任务解决方案的工具。在本项工作中，我们介绍了向量化多智能体模拟器（VMAS）。VMAS是一个开源框架，旨在高效地进行MARL基准测试。它由一个用PyTorch编写的向量化2D物理引擎和十二个具有挑战性的多机器人场景组成。通过一个简单模块化的接口，能够实现额外的场景。我们演示了如何通过向量化在加速硬件上实现并行模拟，而不会增加复杂性。在将VMAS与OpenAI MPE比较时，我们展示了MPE的执行时间随着模拟次数的增加呈线性增长，而VMAS能够在10秒内执行30,000次并行模拟，证明其速度超过$1 0 0 \times$。使用VMAS的RLlib接口，我们基于各种近端策略优化（PPO）算法对多机器人场景进行了基准测试。VMAS的场景在不同维度上对最先进的MARL算法提出了挑战。VMAS框架可在以下网址获取：https://github.com/proroklab/VectorizedMultiAgentSimulator。有关VMAS场景和实验的视频可在此观看。关键词：模拟器，多机器人学习，向量化

# 1 引言

许多实际问题需要多个机器人协同解决。然而，协调问题通常计算复杂度较高。示例包括路径规划、任务分配和区域覆盖。尽管存在精确解，但其复杂性随着机器人数量的增加而呈指数增长。元启发式算法提供了快速且可扩展的解决方案，但缺乏最优性。多智能体强化学习（MARL）可以作为一种可扩展的方法，用于寻找这些问题的近似最优解。在MARL中，经过仿真训练的智能体通过与环境的交互收集经验，并通过奖励信号训练其策略（通常使用深度神经网络表示）。

![](images/1.jpg)  
Fig.1: Multi-robot scenarios introduced in VMAS. Robots (blue shapes) interact among each other and with landmarks (green, red, and black shapes) to solve a task.

然而，当前的多智能体强化学习（MARL）方法存在几个问题。首先，训练阶段可能需要显著的时间才能收敛到最优行为。这部分是由于算法的样本效率，部分是由于模拟器的计算复杂性。其次，当前的基准测试特定于预定义任务，主要针对不切实际的类似视频游戏的场景，远离现实世界中的多机器人问题。这使得该领域的研究变得碎片化，每引入一个新任务就需要实现一个新的仿真框架。另一方面，多机器人模拟器证明更为通用，但其高保真度和全栈仿真导致性能缓慢，阻碍了其在MARL中的应用。全栈学习可能显著妨碍训练性能。如果利用仿真解决高级多机器人协调问题，同时将低级机器人控制留给基于第一性原理的方法，学习的样本效率可以显著提高。

出于这些原因，我们引入了 VMAS，一种经过向量化的多智能体仿真器。VMAS 是一个基于 PyTorch 编写的向量化 2D 物理仿真器，旨在高效地进行 MARL 基准测试。它模拟不同形状的智能体和地标，并支持扭矩、弹性碰撞和自定义重力。智能体采用全运动模型以简化仿真。PyTorch 中的向量化使 VMAS 能够以批处理的方式进行仿真，轻松扩展到数万个并行环境，并在加速硬件上运行。我们所称的 GPU 向量化是指 GPU warp 内部可用的单指令多数据（SIMD）执行范式。该范式允许在批处理的一组并行仿真中执行相同的指令。VMAS 提供与 OpenAI Gym 和 RLlib 库兼容的接口，使得与广泛的 RL 算法的集成变得毫不费力。VMAS 还提供一个框架，方便实现自定义多机器人场景。利用这个框架，我们引入了一组 12 个多机器人场景，代表了困难的学习问题。通过一个简单而模块化的接口，可以实现额外的场景。我们将所有来自 OpenAI MPE 的场景向量化并移植到 VMAS 中。我们使用三种基于近端策略优化（PPO）的 MARL 算法对 VMAS 的四个新场景进行基准测试。我们通过在 RLlib 库中基准测试我们的场景展示了向量化的优势。我们的场景在互补的方式上挑战了最先进的 MARL 算法。贡献。现在列出本工作的主要贡献：我们引入了 VMAS 框架，一种向量化的多智能体仿真器，能够规模化进行 MARL 训练。VMAS 支持智能体间的通信和可定制传感器，例如 LIDAR。我们在 VMAS 中实现了一套十二个多机器人场景，重点测试不同的集体学习挑战，包括：行为异质性、通过通信进行协调以及对抗性交互。我们将所有场景从 OpenAI MPE 移植并向量化到 VMAS 中，并对这两个仿真器进行性能比较。我们展示了在仿真速度方面向量化的优势，表明 VMAS 的速度比 MPE 快出 $1 0 0 \times$。VMAS 代码库可在此获取。

# 2 相关工作

在本节中，我们回顾了多智能体和多机器人仿真领域的相关文献，突出每个领域的核心差距。此外，我们在表1中将最相关的仿真框架与VMAS进行了比较。

多智能体强化学习环境。在多智能体强化学习（MARL）的背景下，已经有大量研究工作致力于解决多机器人的模拟问题，以学习复杂的协调策略。已提出了逼真的 GPU 加速模拟器和引擎。Isaac 是 NVIDIA 提供的专有模拟器，用于强化学习中的真实机器人模拟。它不是通过环境向量化来加速学习，而是利用在同一模拟实例中并发执行多个训练环境。尽管如此，其高保真模拟使其在高层次 MARL 问题上计算成本昂贵。Brax 是谷歌推出的向量化 3D 物理引擎。它使用 Jax 库实现环境批处理和全微分性。然而，当模拟的智能体数量扩大时，计算问题随之出现，导致在仅 20 个智能体的情况下就出现环境停滞的现象。也存在用于单智能体向量化环境的项目，但将这些项目扩展到多智能体领域的复杂性并非微不足道。

MARL 文献的核心基准环境专注于高层次的机器人间学习。Multiagent Particle Environments (MPE) [16] 是由 OpenAI 创建的一组环境。它们共享 VMAS 的模块化原则和新场景创建的简易性，但不提供环境向量化。MAgent [38] 是一个支持大量智能体的离散世界环境。Multi-Agent Learning Environments [10] 是另一组简化的离散世界环境，涵盖一系列不同的多机器人任务。Multi-Agent Emergence Environments [2] 是一个可定制的 OpenAI 3D 模拟器，用于捉迷藏风格的游戏。Pommerman [26] 是一个离散化的游乐场，用于学习多智能体竞争策略。SMAC [28] 是基于星际争霸 2 的一个非常流行的 MARL 基准。Neural-MMO [31] 是另一个类视频游戏环境，智能体在大群体中学习生存。Google Research Football [12] 是一个足球模拟器，包含一系列测试游戏不同方面的场景。Gym-pybullet-drones [21] 是一个用于多四旋翼控制的真实感 PyBullet 模拟器。Particle Robots Simulator [30] 是一个粒子机器人模拟器，需要高协调策略以克服激励限制并完成高级任务。Multi-Agent Mujoco [23] 由多个智能体构成，这些智能体控制单个 Mujoco [32] 智能体的不同身体部位。虽然所有这些环境提供有趣的 MARL 基准，但它们大多专注于特定任务。此外，这些环境都不提供 GPU 向量化，这对于高效的 MARL 训练至关重要。我们在表 1 中呈现 VMAS 与上述所有环境的比较。多机器人模拟器。视频游戏物理引擎如 Unity 和 Unreal Engine 提供现实的模拟，可以为多智能体机器人提供优势。两者都采用了 GPU 加速的 NVIDIA PhysX。然而，它们的通用性在用于机器人研究时会导致高额开销。其他流行的物理引擎包括 Bullet、Chipmunk、Box2D 和 ODE。这些引擎在能力上都相似，并且由于提供了 Python API 的可用性，采用上更为简便。因此，它们通常是进行真实机器人模拟时的首选工具。然而，由于这些工具未利用 GPU 加速的批量模拟，它们在 MARL 训练中导致了性能瓶颈。

最广为人知的机器人模拟器是 Gazebo 和 Webots。它们的引擎基于 ODE 3D 动力学库。这些模拟器支持多种机器人模型、传感器和执行器，但在代理数量增加时，性能会显著下降。研究表明，仅使用 12 个机器人时就会出现完整的模拟停滞。因此，Argos 被提出作为一个可扩展的多机器人模拟器。它能够通过将模拟空间的不同部分分配给不同的物理引擎，以实现数千个代理的群体模拟，且各物理引擎具有不同的模拟目标和精度。此外，它还通过多线程实现 CPU 并行化。尽管具备这些特性，但所描述的模拟器都不够快速，无法在多智能体强化学习训练中使用。这是因为它们优先考虑现实的全栈多机器人模拟而非速度，并且未利用 GPU 加速进行并行模拟。在多智能体强化学习中，对现实主义的关注并不总是必要的。实际上，大多数集体协调问题可以与涉及感知和控制的低级问题解耦。当这些问题能够独立高效地解决而不损失一般性时，快速的高层次模拟提供了一个重要工具。这一洞见是推动 VMAS 中全局性假设的重要因素。

Table 1: Comparison of multi-agent and multi-robot simulators and environments.   

<table><tr><td colspan="10">Vectora Stateb Comm Actiond PhysEnge #Agentsf Gen8 Exth MRobi MARL RLlibk</td></tr><tr><td>Brax [8]</td><td>✓</td><td>C</td><td>X</td><td>C</td><td>3D</td><td>&lt; 10</td><td>✓</td><td></td><td></td><td></td><td>X</td></tr><tr><td>MPE [16]</td><td>X</td><td>C</td><td>C+D</td><td>C+D</td><td>2D</td><td>&lt; 100</td><td>✓</td><td>:</td><td>*</td><td>×&gt;&gt;</td><td>✓</td></tr><tr><td>MAgent [38]</td><td>X</td><td>D</td><td>X</td><td>D</td><td>X</td><td>&gt; 1000</td><td>X</td><td></td><td></td><td></td><td>✓</td></tr><tr><td>MA-Learning-Environments [10]</td><td>X</td><td>D</td><td>X</td><td>D</td><td>X</td><td>&lt; 10</td><td>✓</td><td>×</td><td>✓</td><td>✓</td><td>X</td></tr><tr><td>MA-Emergence-Environments [2]</td><td>X</td><td>C</td><td>X</td><td>C+D</td><td>3D</td><td>&lt; 10</td><td>X</td><td>✗</td><td>X</td><td></td><td>X</td></tr><tr><td>Pommerman [26]</td><td>X</td><td>D</td><td>X</td><td>D</td><td>X</td><td>&lt; 10</td><td>X</td><td></td><td>X</td><td>✓</td><td>X</td></tr><tr><td>SMAC [28]</td><td>X</td><td>C</td><td>X</td><td>D</td><td>X</td><td>&lt; 100</td><td>X</td><td>✗</td><td>✗</td><td></td><td></td></tr><tr><td>Neural-MMO [31]</td><td>X</td><td>C</td><td>X</td><td>C+D</td><td>X</td><td>&lt; 1000</td><td>X</td><td>✓</td><td>X</td><td>✓</td><td>✓</td></tr><tr><td>Google research football [12]</td><td>X</td><td>C</td><td>X</td><td>D</td><td>2D</td><td>&lt; 100</td><td>X</td><td>✓</td><td>X</td><td>✓</td><td></td></tr><tr><td>gym-pybullet-drones [21]</td><td>X</td><td>C</td><td>X</td><td>C</td><td>3D</td><td>&lt; 100</td><td>X</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr><tr><td>Particle robots simulator [30]</td><td>X</td><td>C</td><td>X</td><td>C+D</td><td>2D</td><td>&lt; 100</td><td>X</td><td>✓</td><td>✓</td><td>✓</td><td>X</td></tr><tr><td>MAMujoco [23]</td><td>X</td><td>C</td><td>X</td><td>C</td><td>3D</td><td>&lt; 10</td><td>X</td><td>X</td><td>X</td><td>✓</td><td>X</td></tr><tr><td></td><td>X</td><td>C</td><td>C+D</td><td>C+D</td><td>3D</td><td>&lt; 10</td><td>✓</td><td>✓</td><td></td><td></td><td></td></tr><tr><td>Gazebo [11] Webots [18]</td><td>X</td><td>C</td><td>C+D</td><td>C+D</td><td>3D</td><td>&lt; 10</td><td>✓</td><td>✓</td><td>✓ ✓</td><td>X X</td><td>X X</td></tr><tr><td>ARGOS [24]</td><td>X</td><td>C</td><td>C+D</td><td>C+D</td><td>2D&amp;3D</td><td>&lt; 1000</td><td>✓</td><td>✓</td><td>✓</td><td>X</td><td>X</td></tr><tr><td>VMAS</td><td>✓</td><td>C</td><td>C+D</td><td>C+D</td><td>2D</td><td>&lt; 100</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr></table>

a 向量化的连续状态或离散状态 b 连续通信（C）或离散通信（D） c 显示器内部的连续动作（C）或离散动作（D） d 物理引擎类型 e 支持的智能体数量 f 通用模拟器：可以创建任何类型的任务 g 可扩展性（用于创建新场景的API） h 包含多机器人任务 i 为多智能体强化学习（MARL）而设计 j 与RLlib框架兼容

# 3 VMAS平台

使VMAS与表1中比较的相关工作不同的独特特征在于，我们的平台将多智能体学习与环境向量化相结合。向量化是加速多智能体强化学习训练的关键组成部分。事实上，一个在线训练迭代由模拟团队推演和策略更新组成。在迭代 $k$ 的推演阶段，根据智能体的策略 $\pi _ { k }$ 进行仿真以收集智能体与环境交互的经验。然后，收集到的经验用于更新团队策略。新的策略 $\pi _ { k + 1 }$ 将在下一个训练迭代的推演阶段中使用。推演阶段通常是该过程的瓶颈。向量化允许并行仿真并帮助缓解此问题。受到一些现有解决方案（如MPE [16]）模块化特性的启发，我们创建了我们的框架，作为一个新的可扩展平台，用于运行和创建多智能体强化学习基准。为此，我们遵循了一系列原则开发了VMAS：

向量化。VMAS 向量化可以同时处理任意数量的环境。这显著减少了在多智能体强化学习（MARL）训练中收集推演所需的时间。简单。存在复杂的向量化物理引擎（例如 Brax [8]），但在处理多个智能体时，它们无法高效扩展。这违背了向量化所设定的计算速度目标。VMAS 使用一个用 PyTorch 编写的简单自定义二维动力学引擎来提供快速的仿真。通用。VMAS 的核心结构使其能够用于实现通用的二维高层次多机器人问题。它支持对抗性和协作性场景。全向移动机器人仿真将重点转向高层次的协调，消除了使用 MARL 学习低层次控制的需求。可扩展。VMAS 不仅仅是一个具有环境集合的仿真器。它是一个框架，可以用来创建可供整个 MARL 社区使用的新多智能体场景。为此，我们对框架进行了模块化，以便于新任务的创建，并引入了交互式渲染以调试场景。兼容。VMAS 具有多个封装器，使其与不同的 MARL 接口直接兼容，包括 RLlib [15] 和 Gym [7]。RLlib 具有大量已实现的强化学习算法。我们将深入分析 VMAS 的结构。接口。VMAS 的结构如图 2 所示。它具有向量化接口，意味着可以在一个批次中并行处理任意数量的环境。在第 5 节中，我们展示了向量化如何在 CPU 上带来重要的加速，以及在 GPU 上实现无缝扩展。虽然标准仿真器接口使用 PyTorch [22] 直接输入/输出张量，但我们提供了标准非向量化的 OpenAI Gym [7] 接口和 RLlib [15] 框架的向量化接口的封装器。这使用户可以轻松访问 RLlib 中已经可用的各种强化学习训练算法。在每个仿真步骤中，所有环境和智能体的动作都传递给 VMAS。VMAS 支持运动和智能体间通信动作，这两者可以是连续的或离散的。VMAS 的接口通过 Pyglet [1] 提供渲染。场景。场景编码了团队试图解决的多智能体任务。自定义场景可以在几小时内实现并使用交互式渲染进行调试。交互式渲染是一项功能，允许用户以类似视频游戏的方式控制场景中的智能体，同时所有环境相关的数据都在屏幕上显示。要实现一个场景，只需定义几个函数：make_world 用于创建场景中的智能体和地标并将其生成在世界中，reset_world_at 用于同时重置批次中具体环境或所有环境，reward 返回所有环境中一个智能体的奖励，observation 返回所有环境中智能体的观察。可选地，可以实现 done 和 info 以提供结束条件和额外信息。关于如何创建新场景的进一步文档可在代码库中找到。

![](images/2.jpg)  
Fig. 2: VMAS structure. VMAS has a vectorized MARL interface (left) with wrappers for compatibility with OpenAI Gym [7] and the RLlib RL library [15]. The default VMAS interface uses PyTorch [22] and can be used for feeding input already on the GPU. Multi-agent tasks in VMAS are defined as scenarios (center). To define a scenario, it is sufficient to implement the listed functions. Scenarios access the VMAS core (right), where agents and landmarks are simulated in the world using a 2D custom written physics module.

核心。场景与核心互动。在这里，世界模拟被逐步执行。该世界包含 $n$ 个实体，它们可以是智能体或地标。实体具有形状（球体、方块或线条）和向量化状态 $( \mathbf { x } _ { i } , \dot { \mathbf { x } } _ { i } , \theta _ { i } , \dot { \theta } _ { i } )$ ， $\forall i \in [ 1 . . n ] \ \equiv \ N$ ，其包含它们的位置 $\mathbf { x } _ { i } \in \mathbb { R } ^ { 2 }$ ，速度 $\dot { \bf x } _ { i } \in { \mathbb { R } ^ { 2 } }$ ，旋转角 $\theta _ { i } \in \mathbb { R }$ ，和角速度 $\dot { \theta } _ { i } \in \mathbb { R }$ ，适用于所有环境。实体具有质量 $m _ { i } \in \mathbb { R }$ 和最大速度，并可以被自定义为可移动、可旋转以及可碰撞。智能体的动作由物理动作组成，表示为力 $\mathbf { f } _ { i } ^ { a } \in \mathbb { R } ^ { 2 }$ ，以及可选的通信动作。在当前模拟器状态下，智能体无法控制其朝向。智能体可以通过接口控制或通过场景中定义的“动作脚本”控制。可选地，模拟器可以对动作和观察引入噪声。可以向智能体添加自定义传感器。目前支持激光雷达。世界具有模拟步长 $\delta t$ 、速度阻尼系数 $\zeta$ 和可自定义的重力 $\mathbf { g } \in \mathbb { R } ^ { 2 }$。VMAS 具有基于力的物理引擎。因此，模拟步长使用时间 $t$ 的力来通过半隐式欧拉方法 [19] 集成状态：

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { { \bf f } _ { i } ( t ) = { \bf f } _ { i } ^ { a } ( t ) + { \bf f } _ { i } ^ { g } + \sum _ { j \in N \backslash \{ i \} } { \bf f } _ { i j } ^ { e } ( t ) } \\ { \dot { \bf x } _ { i } ( t + 1 ) = ( 1 - \zeta ) \dot { \bf x } _ { i } ( t ) + \frac { { \bf f } _ { i } ( t ) } { m _ { i } } \delta t } \\ { { \bf x } _ { i } ( t + 1 ) = { \bf x } _ { i } ( t ) + \dot { \bf x } _ { i } ( t + 1 ) \delta t } \end{array} \right. \quad , } \end{array}
$$

其中 $\mathbf { f } _ { i } ^ { a }$ 是智能体的动作力，$\mathbf { f } _ { i } ^ { g } = m _ { i } \mathbf { g }$ 是由重力引起的力，$\mathbf { f } _ { i j } ^ { e }$ 是用于模拟实体 $i$ 和 $j$ 之间碰撞的环境力，其形式如下：

$$
\mathbf { f } _ { i j } ^ { e } ( t ) = \left\{ \begin{array} { l l } { c \frac { \mathbf { x } _ { i j } ( t ) } { \left\| \mathbf { x } _ { i j } ( t ) \right\| } k \log \left( 1 + e ^ { - \left( \left\| \mathbf { x } _ { i j } ( t ) \right\| - d _ { \operatorname* { m i n } } \right) } \right) \quad } & { \mathrm { ~ i f ~ } \left\| \mathbf { x } _ { i j } ( t ) \right\| \leqslant d _ { \operatorname* { m i n } } } \\ { 0 \quad } & { \mathrm { ~ o t h e r w i s e } } \end{array} \right. .
$$

在这里，$c$ 是一个调节力强度的参数。$\mathbf { x } _ { i j }$ 是两个实体形状上最近点之间的相对位置。$d _ { \mathrm { m i n } }$ 是它们之间允许的最小距离。对数内部的项计算出一个与两个实体的渗透量成比例的标量，由系数 $k$ 参数化。该项随后与归一化的相对位置向量相乘。碰撞强度和渗透量可以通过调节 $c$ 和 $k$ 来进行调整。这是 OpenAI MPE [16] 中使用的相同碰撞系统。线性状态使用的仿真步长同样适用于角状态：

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { \tau _ { i } ( t ) = \sum _ { j \in N \backslash \{ i \} } \left\| \mathbf { r } _ { i j } ( t ) \times \mathbf { f } _ { i j } ^ { e } ( t ) \right\| } \\ { \dot { \theta } _ { i } ( t + 1 ) = ( 1 - \zeta ) \dot { \theta } _ { i } ( t ) + \frac { \tau _ { i } ( t ) } { I _ { i } } \delta t } \\ { \theta _ { i } ( t + 1 ) = \theta _ { i } ( t ) + \dot { \theta } _ { i } ( t + 1 ) \delta t } \end{array} \right. . } \end{array}
$$

在这里，$\mathbf { r } _ { i j } \in \mathbb { R } ^ { 2 }$ 是从实体中心到碰撞点的向量，$\tau _ { i }$ 是扭矩，而 $I _ { i }$ 是实体的惯性矩。核心中调节物理仿真的规则是基本的二维动力学，以矢量化的方式使用 PyTorch 实现。它们仅模拟全自主（无约束运动）实体。

# 4 多机器人场景

除了 VMAS 之外，我们引入了一组包含 12 个多机器人场景的集合。这些场景包含各种多机器人问题，解决这些问题需要复杂的协调，例如利用异构行为和智能体间的通信。虽然在这些场景中并未使用发送通信行为的能力，但通信可以用于策略中以提高性能。例如，图神经网络（GNN）可以通过信息共享来克服部分可观测性。每个场景通过定义智能体的观察集来限制其输入。该集合通常包含解决任务所需的最小观察（例如位置、速度、传感器输入、目标位置）。通过修改这些观察，场景可以任意变得更难或更简单。例如，如果智能体试图运输一个包裹，可以将包裹的确切相对距离从智能体的输入中移除，并用 LIDAR 测量替代。将全局观察从场景中去除是促进智能体间通信的良好激励。所有任务包含大量可参数化组件。每个场景都有一组测试，对所有智能体运行本地启发式。我们同时将 MPE 中的所有 9 个场景向量化并移植到 VMAS。在本节中，我们简要概述我们的新场景。有关更多细节（例如观察空间、奖励等），您可以在 VMAS 仓库中找到深入描述。 运输（图 1a）。$N$ 个智能体需要将 $M$ 个包裹推到一个目标。包裹具有可自定义的质量和形状。单个智能体无法独立移动高质量包裹，因此需要与队友合作来解决任务。 轮子（图 1b）。$N$ 个智能体需要共同旋转一条线。这条线固定在原点上，具有可参数化的质量和长度。团队的目标是将这条线带到所需的角速度。对于单个智能体来说，推高质量的线是不可行的。因此，团队必须与两侧的智能体协调，以增加和减少线的速度。 平衡（图 1c）。$N$ 个智能体在垂直重力的世界底部生成。一条线在它们上方生成。智能体必须将位于线顶端的球形包裹运输到上方的给定目标。包裹具有可参数化的质量，而线可以旋转。 让路（图 1d）。两个智能体在对称环境中彼此目标前面开始。为了解决任务，一个智能体必须通过使用环境中间的狭窄空间让路给另一个。 足球（图 1e）。一队 $N$ 个蓝色智能体与一队 $M$ 个红色智能体竞争以进球。默认情况下，红色智能体由启发式 AI 控制，但也可以进行自我对弈。队友之间的合作对于协调攻击和防守动作是必要的。智能体需要沟通并假设不同的行为角色以完成任务。 通道（图 1f）。5 个智能体以交叉形式开始，必须在障碍物的另一侧重现相同的形成。障碍物有 $M$ 个通道（图中 $M=1$）。智能体因相互碰撞和与障碍物的碰撞而受到惩罚。此场景是对 [4] 中考虑的场景的一种概括。 反向运输（图 1g）。此任务与运输相同，但只有一个包裹。智能体在包裹内部生成并需要将其推到目标。 分散（图 1h）。有 $N$ 个智能体和 $N$ 个食物颗粒。智能体从同一位置开始，需要协作吃掉所有食物。大多数多智能体强化学习算法无法解决此任务（没有通信或来自其他智能体的观察），因为它们受到来自参数共享的行为同质性的限制。因此，每个智能体需要异构行为来处理不同的食物颗粒。 掉落（图 1i）。$N$ 个智能体需要共同达到一个目标。完成任务只需一名智能体到达目标即可。团队因所有智能体控制的总和而受到比例惩罚。因此，智能体需要自我组织，只发送最近的机器人到达目标，尽可能节省能量。 成群（图 1j）。$N$ 个智能体必须围绕一个目标聚集，同时避免与彼此及 $M$ 个障碍物碰撞。成群是多机器人协调的重要基准，最初的解决方案根据局部规则模拟行为，而最近的研究使用基于学习的方法。与相关工作相比，我们的聚集环境包含静态障碍物。 发现（图 1k）。$N$ 个智能体必须协调尽快覆盖 $M$ 个目标，同时避免碰撞。如果 $K$ 个智能体接近目标的距离至少为 $D$，则认定该目标已被覆盖。覆盖目标后，$K$ 个覆盖智能体各自获得奖励，目标在随机位置重新生成。此场景是拉条实验的变体，并且虽然没有通信也可以解决，但已证明当 $N < M$ 时，通信显著提高了性能。 瀑布（图 11）。$N$ 个智能体从上到下移动，穿越一系列障碍。这是一个测试场景，可用于发现 VMAS 的功能。

# 5 与最优边际概率估计（MPE）的比较

在本节中，我们比较了VMAS和MPE [16]的可扩展性。由于我们对VMAS中的所有MPE场景进行了向量化和移植，因此可以在同一个MPE任务上比较这两个仿真器。选择的任务是“simple_spread”，因为它包含多个可碰撞的智能体在同一环境中。VMAS和MPE采用了两种完全不同的执行范式：VMAS由于采用向量化技术，利用了单指令多数据（SIMD）范式，而MPE则使用了单指令单数据（SISD）范式。因此，仅在一个任务上报告这种范式转变的好处是足够的，因为这些好处与任务无关。

在图3中，我们可以看到执行时间随着并行步入的环境数量增长的趋势，分别针对两个模拟器。MPE只在CPU上运行，而使用PyTorch的VMAS则同时在CPU和GPU上运行。在本次实验中，我们在Intel(R) Xeon(R) Gold 6248R CPU上比较了这两个模拟器，该处理器的主频为 $\textcircled { a } \ 3 . 0 0 \mathrm { G H z }$，同时还在NVIDIA GeForce RTX 2080 Ti上运行了VMAS。结果显示了向量化对模拟速度的影响。在CPU上，VMAS的速度比MPE快${ 5 } \mathbf { x }$。在GPU上，VMAS的模拟时间与环境数量无关，并且速度最高可达$1 0 0 \times$。该结果可以在不同硬件上复现。在VMAS的仓库中，我们提供了一个脚本用于重复此实验。

# 6 实验与基准测试

我们进行了一系列训练实验，以基准测试 MARL 算法在四个 VMAS 场景中的性能。得益于 VMAS 的向量化，我们能够在平均 25 秒内完成一次训练迭代（包括 60,000 次环境交互和深度神经网络训练）。本节报告的运行时间均在 3 小时以内。所比较的模型均基于近端策略优化（Proximal Policy Optimization）[29]，这是一种演员-评论家强化学习算法。演员是一个深度神经网络（DNN），根据观察输出动作；评论家是一个 DNN（仅在训练期间使用），根据观察输出一个值，表示当前状态和动作的优劣。当演员和评论家可以访问所有智能体的观察并输出所有智能体的动作/值时，我们将其称为集中式；当它们仅将一个智能体的观察映射到其动作/值时，我们称之为分散式。所比较的模型包括：

![](images/3.jpg)  
Fig. 3: Comparison of the scalability of VMAS and MPE [16] in the number of parallel environments. In this plot, we show the execution time of the "simple_spread" scenario for 100 steps. MPE does not support vectorization and thus cannot be run on a GPU.

CPPO：该模型使用集中式评论员和演员。它将多智能体问题视为一个超级智能体的单智能体问题。MAPPO [37]：该模型使用集中式评论员和分散式演员。因此，智能体独立行动，采用局部分散政策，但使用集中信息进行训练。IPPO [36]：该模型使用分散式评论员和演员。每个智能体独立学习和行动。模型参数在智能体之间共享，以便它们能够利用彼此的经验。HetIPPO：我们对IPPO进行了定制，禁用参数共享，使每个智能体的模型独一无二。启发式：这是针对每个任务设计的手工分散启发式方法。实验在RLlib [15]中使用向量化接口进行。我们将所有算法运行400个训练迭代。每个训练迭代在60,000次环境交互中进行。我们绘制了10次不同种子的实验中平均剧集奖励的均值和标准差。所有评论员和演员使用的模型为两层多层感知器（MLP），激活函数为双曲正切函数。学习到的策略视频可以在此链接中查看1。接下来，我们讨论训练场景的结果。运输（图4a）。在运输环境中，只有IPPO能够学习到最优政策。这是因为其他具有集中式组件的模型，其输入空间由所有智能体的观察值的拼接组成。因此，集中式架构无法在需要高初始探索的环境中进行泛化，比如这个环境，在这些环境中可能的联合状态存在很高的方差（因此遇到相似状态的概率很低）。

![](images/4.jpg)  
Fig.4: Benchmark performance of different PPO-based MARL algorithms in four VMAS scenarios. Experiments are run in RLlib [15]. Each training iteration is performed over 60,000 environment interactions. We plot the mean and standard deviation of the mean episode reward4 over 10 runs with different seeds.

车轮（图4b）。车轮环境被证明对多智能体强化学习算法是一个困难的任务。所有模型均无法解决该任务，其表现不如启发式方法。平衡（图4c）。在平衡任务中，所有模型均能解决该任务并超越启发式方法。然而，这在很大程度上是由于使用了包含全局信息的大观测空间。通过删除部分观测空间，从而增加部分可观测性，可以使任务变得任意更难。让路（图4d）。在让路场景中，只有那些能够发展异构智能体行为的算法能够解决该环境。实际上，使用参数共享和去中心化智能体的IPPO和MAPPO未能解决该场景。另一方面，结果表明，该场景可以通过集中式智能体（CPPO）来解决，或者通过禁用参数共享，允许智能体策略异构（HetIPPO）。实验结果确认VMAS提出了一系列场景，这些场景以正交的方式对当前最先进的多智能体强化学习算法提出了挑战。我们展示了不存在“一刀切”的解决方案，并且我们的场景可以为新的多智能体强化学习算法提供有价值的基准。此外，向量化加快了训练速度，这对更广泛地采用多智能体学习在机器人领域至关重要。

# 7 结论

在本研究中，我们介绍了VMAS，这是一种开源的多机器人学习向量化模拟器。VMAS基于PyTorch，由核心向量化二维物理模拟器和一组多机器人场景组成，这些场景编码了困难的集体机器人任务。该框架的重点在于充当一个多智能体强化学习基准测试平台。因此，为了激励社区的贡献，我们使得实现新场景尽可能简单和模块化。我们展示了向量化的计算优势，在GPU上执行了多达30,000个并行模拟，耗时不到10秒。我们在我们的场景中对多智能体强化学习算法的性能进行了基准测试。在我们的训练实验中，我们能够收集60,000个环境步，并在25秒内完成一次训练迭代。实验还显示了VMAS场景在不同方向上给最先进的多智能体强化学习算法带来的挑战。未来，我们计划扩展VMAS的功能，以扩大其采用范围，继续实现新的场景和基准测试。我们还希望模块化物理引擎，使用户能够根据不同的精度和计算需求更换向量化引擎。

# 致谢

本研究得到了ARL DCIST CRA W911NF-17-2-0181和欧洲研究委员会（ERC）项目949940（gAIa）的资助。R. Kortvelesy得到诺基亚贝尔实验室的支持，该实验室通过向剑桥大学移动、可穿戴系统与增强智能中心的捐赠提供资助。J. Blumenkamp感谢“德国人民研究基金会”的支持以及EPSRC学费资助。

# References

1. Pyglet. https://pyglet.org/   
2. Baker, B., Kanitscheider, I., Markov, T., Wu, Y., Powell, G., McGrew, B., Mordatch, I.: Emergent tool use from multi-agent autocurricula. In: International Conference on Learning Representations (2019)   
3Bernstein, D.S., Givan, R., Immerman, N., Zilberstein, S.: The complexity of decentralized control of markov decision processes. Mathematics of operations research 27(4), 819840 (2002)   
.Blumenkamp, J., Morad, S., Gielis, J., Li, Q., Prorok, A.: A framework for real-world multirobot systems running decentralized gnn-based policies. arXiv preprint arXiv:2111.01777 (2021)   
5.Bradbury, J., Frostig, R., Hawkins, P., Johnson, M.J., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S., Zhang, Q.: JAX: composable transformations of Python+NumPy programs (2018). URL http://github.com/google/jax   
6.Bräysy, O., Gendreau, M.: Vehicle routing problem with time windows, part i: Metaheuristics. Transportation science 39(1), 119139 (2005)   
7.Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., Zaremba, W.: Openai gym. arXiv preprint arXiv:1606.01540 (2016)   
8. Freeman, C.D., Frey, E., Raichuk, A., Girgin, S., Mordatch, I., Bachem, O.: Brax - a differentiable physics engine for large scale rigid body simulation (2021). URL http: //github.com/google/brax   
9Ijspeert, A.J., Martinoli, A., Billard, A., Gambardella, L.M.: Collaboration through the exploitation of local interactions in autonomous collective robotics: The stick pulling experiment. Autonomous Robots 11(2), 149171 (2001)   
10. Jiang, S., Amato, C.: Multi-agent reinforcement learning with directed exploration and selective memory reuse. In: Proceedings of the 36th Annual ACM Symposium on Applied Computing, pp. 777784 (2021)   
11. Koenig, N., Howard, A.: Design and use paradigms for gazebo, an open-source multi-robot simulator. In: 2004 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)(IEEE Cat. No. 04CH37566), pp. 21492154. IEEE (2004)   
12. Kurach, K., Raichuk, A., Staczyk, P., Zajac, M., Bachem, O., Espeholt, L., Riquelme, C., Vincent, D., Michalski, M., Bousquet, O., et al.: Google research football: A novel reinforcement learning environment. In: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 45014510 (2020)   
13. Lange, R.T.: gymnax: A JAX-based reinforcement learning environment library (2022). URL http://github.com/RobertTLange/gymnax   
14. Li, Q., Gama, F., Ribeiro, A., Prorok, A.: Graph neural networks for decentralized multirobot path planning. In: 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 11,78511,792. IEEE (2020)   
15. Liang, E., Liaw, R., Nishihara, R., Moritz, P., Fox, R., Goldberg, K., Gonzalez, J., Jordan, M., Stoica, I.: Rllib: Abstractions for distributed reinforcement learning. In: International Conference on Machine Learning, pp. 30533062. PMLR (2018)   
16. Lowe, R., Wu, Y.I., Tamar, A., Harb, J., Pieter Abbeel, O., Mordatch, I.: Multi-agent actorcritic for mixed cooperative-competitive environments. Advances in neural information processing systems (2017)   
17. Makoviychuk, V., Wawrzyniak, L., Guo, Y., Lu, M., Storey, K., Mackin, M., Hoeller, D., Ru N., Allhire A. Handa A., et al.: Isaacgym: High perr gu bas pys latno oot ear. I Thiry- Cee  Neural Inoaion o Systems Datasets and Benchmarks Track (Round 2) (2021)   
18. Michel, O.: Cyberbotics ltd. webotsTM: professional mobile robot simulation. International Journal of Advanced Robotic Systems p. 5 (2004)   
1Niiranen, J.: Fast and accurate symmetric euler algorithm for electromechanical simulations. In: Electrimacs 99 (modelling and simulation of electric machines converters an& systems), pp. I71 (1999)   
20. Noori, F.M., Portugal, D., Rocha, R.P., Couceiro, M.S.: On 3d simulators for multi-robot systems in ros: Morse or gazebo? In: 2017 IEEE International Symposium on Safety, Security and Rescue Robotics (SSRR), pp. 1924 (2017)   
21. Panerati, J., Zheng, H., Zhou, S., Xu, J., Prorok, A., Schoellig, A.P.: Learning to fly—a gym environment with pybullet physics for reinforcement learning of multi-agent quadcopter control. In: 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 75127519. IEEE (2021)   
PkeA.Gros . assaF Lrer BraryJhan GKile T. Li Gielshein, N., Antiga, L., et al.: Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems (2019)   
23. Peng, B., Rashid, T., Schroeder de Witt, C., Kamienny, P.A., Torr, P., Böhmer, W., Whiteson, S.: Facmac: Factored multi-agent centralised policy gradients. Advances in Neural Information Processing Systems pp. 12,20812,221 (2021)   
24. Pinciroli, C., Trianni, V., O'Grady, R., Pini, G., Brutschy, A., Brambilla, M., Mathews, N., Ferrante, E., Di Caro, G., Ducatelle, F., Birattari, M., Gambardella, L.M., Dorigo, M.: ARGoS: u. 271295 (2012)   
Prorok, A.Robust assigment using redundant robots on transport networks with uncertain travel time. IEEE Transactions on Automation Science and Engineering pp. 20252037 (2020)   
2Resnick, C., Eldridge, W., Ha, D., Britz, D., Foerster, J., Togelius, J., Cho, K., Bruna, J.: Pommerman: A multi-agent playground. CoRR (2018)   
Reynolds, C.W. Flocks, herds and schools: A distributed behavioral model. In: Proceeings of the 14th annual conference on Computer graphics and interactive techniques, pp. 2534 (1987)   
Samvelyan, M., Rashid, T. de Witt C.S., Farquhar, G., Nardelli, N., Rudner, T.G.J. Hug, C.M., Torr, P.H.S., Foerster, J., Whiteson, S.: The StarCraft Multi-Agent Challenge. CoRR (2019)   
Suan, J.Wolski F aral ord A.Ko, Poial polii algorithms. arXiv preprint arXiv:1707.06347 (2017)   
30. Shen, J., Xiao, E., Liu, Y., Feng, C.: A deep reinforcement learning environment for particle robot navigation and object manipulation. arXiv preprint arXiv:2203.06464 (2022)   
3Suarez, J., Du, Y., Isola, P. Mordatch, I.: Neural mmo: A massively multiagent game environment for training and evaluating intelligent agents. arXiv preprint arXiv:1903.00784 (2019)   
32. Todorov, E., Erez, T., Tassa, Y.: Mujoco: A physics engine for model-based control. In: 2012 IEEE/RSJ international conference on intelligent robots and systems, pp. 50265033. IEEE (2012)   
Tolstaya, E., Gama, F., Paulos, J., Pappas, G., Kumar, V., Ribeiro A.: Learnig decentralzed controllers for robot swarms with graph neural networks. In: L.P. Kaelbling, D. Kragic, K. Sugiura (eds.) Proceedings of the Conference on Robot Learning, Proceedings of Machine Learning Research, vol. 100, pp. 671682. PMLR (2020). URL https://proceedings.mlr. press/v100/tolstaya20a.html   
34.Wang, B., Liu, Z., Li, Q., Prorok, A.: Mobile robot path planning in dynamic environments b   . 69326939 (2020)   
Weg J., Lin M. Hua, S., Liu B. Mauk, D. aichuk, V., Liu . S, Y. Luo T. JY. Xu Z.Yan S.Eol: pa environment execution engine. arXiv preprint arXiv:2206.10558 (2022)   
3de Witt, C.S., Gupta, T. Makovichuk, D., Makoviychuk, V., Torr, P.H. Sun, M., Whiteon, S.: Iidepen l you nee inhear ulgnt hallege?rXi e arXiv:2011.09533 (2020)   
37Yu, C., Velu, A., Vinitsky, E., Wang, Y., Bayen, A., Wu, Y.: The urprising effectiveness of ppo in cooperative, multi-agent games. arXiv preprint arXiv:2103.01955 (2021)   
8Zheg, L Yang, J., Cai, H., Zhou, M., Zhang, W. Wang, J., Yu, Y.: Magent: Amat reinforcement learning platform for artificial collective intelligence. In: Proceedings of the AAAI conference on artificial intelligence (2018)   
39. Zheng, X., Koenig, S., Kempe, D., Jain, S.: Multirobot forest coverage for weighted and unweighted terrain. IEEE Transactions on Robotics pp. 10181031 (2010)