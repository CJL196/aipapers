# Momentum-GS：高质量大场景重建的动量高斯自蒸馏

贾轩 $\mathrm { F a n ^ { 1 , 2 , * } }$ , 万华 $_ { \mathrm { L i ^ { 3 , * } } }$ , 韩怡飞1,2, 戴天儒1,2, 唐彦松1,2, 1清华大学深圳国际研究生院 2清华大学 3哈佛大学 fjx23@mails.tsinghua.edu.cn, wanhua@seas.harvard.edu, hyf23@mails.tsinghua.edu.cn, dtr24@mails.tsinghua.edu.cn, tang.yansong@sz.tsinghua.edu.cn

![](images/1.jpg)  
consistency and avoiding the noticeable lighting discrepancies observed in other Gaussian-based methods.

# 摘要

3D高斯点云已在大规模场景重建中表现出显著成功，但由于高训练内存消耗和存储开销，问题依然存在。混合表示法集成了隐式和显式特征，为缓解这些限制提供了一种途径。然而，当在并行块训练中应用时，会出现两个关键问题：由于独立训练每个块导致数据多样性降低，重建准确性下降；并行训练还限制了划分块的数量，受限于可用的GPU数量。为解决这些问题，我们提出MomentumGS，一种新颖的方法，利用基于动量的自蒸馏，促进各个块之间的一致性和准确性，同时将块的数量与物理GPU数量解耦。我们的方法保持一个基于动量更新的教师高斯解码器，确保训练过程中的稳定参考。该教师以自蒸馏的方式为每个块提供全局指导，促进重建中的空间一致性。为了进一步确保块之间的一致性，我们引入块加权，根据重建准确性动态调整每个块的权重。大规模场景的广泛实验表明，我们的方法在LPIPS上较CityGaussian始终表现优越，实现了$18.7\%$的改进，且划分的块数量远少于现有技术，建立了新的最先进水平。项目页面：https://jixuan-fan.github.io/Momentum-GS_Page/

# 1. 引言

大规模三维场景重建对于包括自主驾驶、虚拟现实、环境监测和航拍测绘在内的广泛应用至关重要。能够准确地从一系列图像中重建大型复杂场景对于创建真实感强、可导航的三维模型，以及支持高质量的可视化、分析和仿真至关重要。

3D 高斯喷溅（3D-GS）[20] 最近因其高重建质量和快速渲染速度而备受关注，并且超越了基于 NeRF 的方法 [2, 4, 38]。在此基础上，最近的方法 [9, 21, 29, 33] 进一步提升了其在大规模场景中的性能。为了更高效地处理大环境，这些方法通常采用分而治之的策略，将大场景划分为多个独立的块，从而实现跨块的多 GPU 训练。这种方法为复杂、广阔的重建提供了可扩展的训练。然而，显式表示数百万个高斯分布会产生巨大的内存和存储需求 [33]，限制了 3D-GS 在大规模场景中的可扩展性。此外，由于在大型场景捕获中不可避免的因素，如光照变化、自动曝光调整或相机姿态的不准确性 [24]，独立训练每个块往往忽略了块与块之间的关系，导致块边界之间的不一致。这一问题可能导致明显的过渡，如图 1 所示，在 CityGaussian [33] 等方法中，光照变化突然渲染不正确。解决这些问题已成为推动 3D 场景重建领域发展的核心关注点。

![](images/2.jpg)  
Figure 2. Comparison of three approaches for using hybrid representations to reconstruct large-scale scenes in a divideand-conquer manner. Examples with two blocks: (a) Independent training of each block, resulting in separate models that cannot be merged due to independent Gaussian Decoders, complicating rendering; (b) Parallel training with a shared Gaussian decoder, allowing merged output but limited by GPU count; (c) Our approach with a Momentum Gaussian Decoder, providing global guidance to each block and improving consistency across blocks.

混合表示[28, 35, 46]作为一种有前景的方法，旨在通过结合隐式和显式特征来解决内存和存储限制。为管理大场景的复杂性，这些表示将密集的体素网格或基于锚点的结构与稀疏的3D高斯场结合在一起。这些方法通常使用多层感知器（MLP）作为高斯解码器，从而生成神经高斯，能够实现高重建精度，同时确保高效推理。解码后的高斯会根据不同的观察角度、距离和场景细节进行动态调整。例如，在Scaffold-GS[35]中，在推理阶段，神经高斯的预测被限制在可见视锥内的锚点，并根据不透明度使用学习的选择过程过滤掉微不足道的高斯。这种方法使得渲染速度可以与原始3D-GS相媲美。此外，神经高斯在视锥内动态生成，使每个锚点能够实时适应性地预测不同观测方向和距离的高斯。这种自适应机制增强了新视角合成的鲁棒性，在各种视角下提供高质量的渲染，同时保持可接受的计算开销。然而，将混合表示应用于大规模3D场景的并行重建面临两个主要挑战。首先，独立训练每个块限制了每个块的高斯解码器内的数据多样性，降低了重建质量，并产生因高斯解码器独立而不能合并的单独模型，如图2(a)所示。相反，如图2(b)所示，采用共享高斯解码器的并行训练允许合并训练模型，但限制了可扩展性，因为块的数量受可用GPU的限制。这些限制强调了需要一种在块间一致性和可扩展性之间取得平衡的方法。为克服这些限制，我们提出了MomentumGS，这是一种将混合表示的优势与满足大规模场景重建独特需求的策略相结合的新方法。我们的方法将块的数量与GPU约束解耦，从而实现重建任务的灵活扩展。这是通过定期从一组$n$个块中采样$k$个块并将其分配到$k$个GPU实现的。为了增强块之间的一致性，我们引入了场景动量自蒸馏，其中使用动量更新的教师高斯解码器为每个块提供一致的全局指导，如图2(c)所示。该框架鼓励块之间的协作学习，确保每个块都能受益于整个场景的更广泛上下文。此外，我们引入了重建导向的块加权，这是一种动态机制，会根据每个块的重建质量调整其强调程度。这种自适应加权使得共享解码器能够优先考虑表现不佳的块，从而增强全局一致性并防止收敛到局部最小值。为了全面评估所提方法的有效性，我们在五个具有挑战性的♀♀大规模场景[27, 30, 55]上进行广泛实验，包括建筑、瓦砾、住所、科艺和MatrixCity。我们的MomentumGS取得了显著的改善，较CityGaussian[33]实现了$1 8 . 7 \%$的LPIPS增益，同时使用的分块数量要少得多。总之，我们的贡献包括：1. 我们引入场景动量自蒸馏以增强高斯解码器性能，并将分块数量与GPU数量解耦，实现可扩展的并行训练。2. 我们的方法结合了重建导向的块加权，动态调整块的强调程度，以确保集中改进较弱的块，增强整体一致性。3. 我们的方法Momentum-GS的重建质量优于最先进的方法，突显了混合表示在大规模场景重建中的巨大潜力。

# 2. 相关工作

神经渲染。神经辐射场（NeRF）[38] 在新视图合成领域开创了一项突破性进展，采用连续体积函数的方式表示三维场景，其中沿发射光线的每个点都会被采样以生成颜色和密度值。为改善NeRF的各个方面，包括效率和可扩展性，已开发出众多扩展方案 $[ 2 -$ 4, 36, 39, 41, 43, 45, 50, 54, 59]。然而，NeRF需要在光线上进行密集采样以获得准确的结果，这导致了较高的计算成本和较长的训练与推理时间。三维高斯溅射（3D Gaussian Splatting）[20] 作为一种有前景的替代方案应运而生，利用高斯溅射进行高效的场景表示。与NeRF相比，3DGS显著减少了采样需求，同时保持了高保真度。由于其速度优势，它已广泛用于许多应用[10, 26, 44]。另一种方法是混合表示，它结合了显式和隐式元素，以利用两者的优势[28, 40, 46, 49, 56]。混合表示通常构建在稠密的均匀体素网格上，利用多种方法改善场景重建。例如，K平面[15] 采用平面分解表示多维场景，支持高效的内存使用，并应用时间平滑等先验知识。Plenoxels[14] 采用稀疏的三维网格和球谐函数，绕过神经网络直接优化图像的真实感视图合成，实现了相较传统辐射场的显著加速。Scaffold-GS[35] 在三维高斯溅射的基础上，通过使用锚点来分布局部三维高斯并根据观察方向和距离动态预测其属性。这些混合方法展示了结合显式和隐式元素进行可扩展、高效场景重建的优势。

大规模场景重建。大规模场景重建有着悠久的历史，传统方法常常依赖结构从运动（SfM）来估计相机位姿，并从图像集合中创建稀疏点云。随后的方法，如多视图立体（MVS），在此基础上扩展，以生成更密集的重建，推动了测量摄影系统处理大场景的能力。随着神经辐射场（NeRF）的出现，向神经表示的转变使得照片真实感视图合成成为可能，从而实现了更详细的场景重建。许多基于NeRF的方法采用类似的分而治之的方法，独立表示每个模块，以促进可扩展的重建。然而，这些方法在场景片段之间的渲染速度和一致性方面仍面临挑战。近期，三维高斯溅射（3D Gaussian Splatting）作为一个有前景的替代方案，不仅实现了实时渲染，还具备高视觉保真度。众多方法通过增强三维高斯溅射的可扩展性和效率，将其扩展到大规模场景。一些方法将这些大场景划分为独立的模块进行并行训练，从而实现高效处理和重建。VastGaussian和CityGaussian通过采用分而治之的方法重建大规模场景，确保了训练收敛，尽管它们缺乏跨模块交互，这可能会限制一致性。DOGS引入了一种分布式训练方法，通过场景分解和交替方向乘子法（ADMM）加速三维高斯溅射，但并未专注于大规模场景中高斯表示的优化。这些近期基于三维高斯溅射的方法展示了三维高斯表示在可扩展、高质量大场景重建中的潜力，尽管在实现无缝过渡和高效内存使用方面仍然存在挑战。

# 3. 方法

概述。混合表示在小型以物体为中心的场景中取得了成功。然而，当以分而治之的方式应用于更大环境的并行训练时，它们面临着一个根本性困境。本文利用混合表示进行大规模场景重建，充分发挥其高重建能力，同时有效地将块的数量与物理GPU数量解耦。第3.1节介绍了3DGS的基本基础。第3.2节探讨了场景动量自蒸馏如何有效解决将混合表示扩展到大场景的挑战。最后，第3.3节呈现了

![](images/3.jpg)  
y .

重建引导的块加权策略，通过根据每个块的重建质量动态调整其权重，从而增强全局场景一致性。

# 3.1. 准备工作

3DGS通过利用高斯表示的可微性质以及基于瓦片的渲染，为准确的场景重建提供了一种高效的解决方案。它将每个3D场景点建模为各向异性高斯，从而通过投影和混合实现高效渲染，而无需传统体积方法中典型的密集光线行进所带来的计算开销。每个3D点表示为一个以 $\mu \in \mathbb { R } ^ { 3 }$ 为中心的高斯函数，其中 $x$ 是空间位置，$\mu$ 是中心，$\Sigma$ 定义了高斯的形状和方向：

$$
G ( x ) = e ^ { - { \frac { 1 } { 2 } } ( x - \mu ) ^ { \top } \Sigma ^ { - 1 } ( x - \mu ) } .
$$

将每个 3D 高斯渲染到 2D 图像平面上，生成 2D 高斯 $G ^ { \prime } ( \mathbf { x } ^ { \prime } )$，其中 $\mathbf { x } ^ { \prime }$ 表示一个像素。投影后的高斯通过 alpha 混合对像素颜色进行贡献：

$$
C ( \mathbf { x } ^ { \prime } ) = \sum _ { i \in N } c _ { i } \sigma _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \sigma _ { j } ) ,
$$

其中 $N$ 是影响 $\mathbf { x } ^ { \prime }$ 的高斯集合，$c _ { i }$ 是以视点相关的球面谐波形式表示的颜色，$\sigma _ { i } = \alpha _ { i } G _ { i } ^ { \prime } ( \mathbf { x } ^ { \prime } )$ 是不透明度，$\alpha _ { i }$ 为可学习参数。高斯的训练采用可微渲染来细化高斯参数，起始于一个初始点云。高斯的优化基于图像重建误差，通过克隆、稠密化和修剪等操作来提升覆盖率和准确性。对于大型场景，高高斯数量带来了内存和计算挑战，通过在渲染过程中控制活跃高斯来进行管理。

# 3.2. 情景感知动量自蒸馏

混合表示在分治方法中应用于并行训练时面临根本挑战。具体而言，GPU 可用性的限制限制了可以同时处理的块数量，从而降低了可扩展性，而维护高斯解码器预测准确性所需的数据多样性依然至关重要。为了解决这些挑战，我们提出了场景动量自蒸馏，这是一种既能将块数与 GPU 限制解耦，又能通过改善数据多样性增强高斯解码器鲁棒性的方法。我们的方法确保高斯解码器从更广泛的数据中受益，使得在大场景中能够实现更准确和一致的预测。在我们的方案中，我们同时并行训练每个块，所有块共享一个高斯解码器。在每次前向传播中，每个块随机选择分配数据中的一个视点，并使用共享的高斯解码器准确预测高斯参数。然后，这些预测参数用于渲染相应的图像，并与真实标注数据进行比较以计算重建损失。我们使用一个损失函数来优化可学习参数，该函数将渲染像素颜色的 $\mathcal { L } _ { 1 }$ 损失与旨在改善结构相似性的 SSIM [58] 项 $\mathcal { L } _ { \mathrm { S S I M } }$ 结合在一起。

$$
\mathcal { L } _ { \mathrm { r e c o n s } } = \mathcal { L } _ { 1 } + \lambda _ { \mathrm { S S I M } } \mathcal { L } _ { \mathrm { S S I M } } ,
$$

其中 $\lambda _ { S S I M }$ 是一个权重因子，用于平衡 $\mathcal { L } _ { 1 }$ 和 SSIM 项的贡献。来自每个块的梯度被累积到共享的高斯解码器中，使其能够从全范围的场景信息中学习。通过采用顺序训练策略，我们的方法规避了对块数量的 GPU 限制。每个 GPU 一次处理一个块，并定期切换以确保覆盖所有块。该设计将块数量与硬件限制解耦，从而在场景复杂性增加时支持可扩展性。

为了在不同训练块之间保持一致性并增强全局一致性，我们引入了一种基于动量的教师高斯解码器 $D _ { t }$，以及一个共享的学生高斯解码器 $D _ { s }$。高斯解码器 $D$ 根据观察位置动态预测高斯属性。具体来说，给定锚点特征 $F \in \mathbb { R } ^ { N \times 3 2 }$、观察距离 $\delta ~ \in ~ \mathbb { R } ^ { N \times 3 }$ 和观察方向 $d \in$ $\mathbb { R } ^ { N \times 1 }$，解码器输出高斯参数，包括颜色、不透明度、旋转和缩放。为了确保计算效率，解码器采用双层多层感知机（MLP）实现。让 $B$ 表示每个并行训练块的索引，并让 $\theta _ { t }$ 和 $\theta _ { s }$ 分别表示教师和学生高斯解码器的参数。我们采用自监督的方法，通过基于动量的参数更新来稳定教师高斯解码器 $D _ { t }$，从而减轻因不同步训练而产生的不一致性。因此，教师解码器提供了一个稳定的全局参考，通过计算其输出之间的一致性损失，指导学生解码器。更严格地说，教师高斯解码器的参数 $\theta _ { t }$ 使用基于动量的公式进行更新，以确保时间稳定性：

$$
\theta _ { t }  m \cdot \theta _ { t } + ( 1 - m ) \cdot \theta _ { s } ,
$$

其中 $m$ 是动量系数，设定为 0.9，以平衡稳定性和更新速度。如果 $m$ 离 1 太近，解码器的更新速度会太慢，妨碍重建效率，而较小的 $m$ 可能会由于教师解码器的过度波动导致不稳定。基于动量的更新确保教师高斯解码器平稳演变，向学生解码器提供稳定一致的指导。在每个块中，教师和学生解码器都预测高斯参数，并对学生解码器与教师的全局指导进行一致性损失应用。该方法利用了增加的数据多样性，同时将块的数量与 GPU 数量解耦，使得能够扩展到任意大的场景。一致性损失被计算为每个块 $B$ 中教师和学生高斯解码器预测之间的均方误差：

$$
\mathcal { L } _ { \mathrm { c o n s i s t e n c y } } = \| D _ { m } ( f _ { b } , v _ { b } ; \theta _ { t } ) - D _ { o } ( f _ { b } , v _ { b } ; \theta _ { s } ) \| _ { 2 } ,
$$

其中 $f _ { b }$ 表示锚定特征，$v _ { b }$ 表示块 $B$ 中每个样本的相对观察方向。该损失鼓励学生解码器 $D _ { o }$ 逐渐与教师解码器 $D _ { m }$ 提供的稳定全局指导对齐，从而在重建过程中促进不同块之间的空间一致性。因此，总损失函数定义为：

$$
\mathcal { L } = \mathcal { L } _ { 1 } + \lambda _ { \mathrm { S S I M } } \mathcal { L } _ { \mathrm { S S I M } } + \lambda _ { \mathrm { c o n s i s t e n c y } } \mathcal { L } _ { \mathrm { c o n s i s t e n c y } } ,
$$

其中 $\lambda _ { \mathrm { c o n s i s t e n c y } }$ 是一个加权因子，用于平衡一致性损失相对于重建损失的影响。这种组合损失确保模型不仅能够准确重建场景，还能够在各个块之间保持全局空间一致性。

# 3.3. 重建引导的块加权

为了平衡各个块的训练进度并减轻由于初始场景分区不均引起的问题，我们引入了重建引导的块加权方法。该方法根据每个块的重建质量动态调整权重，通过优先考虑重建准确性较低的块来增强一致性。为了监测和调整每个块的重建性能，我们维护一个表格，记录关键重建指标，具体为PSNR（峰值信噪比）和SSIM（结构相似性指数）。这些指标提供了重建质量的量化衡量，数值越高表示视觉保真度越好。块的PSNR定义为每个块内每幅图像的平均PSNR。块的SSIM以类似方式计算。为了确保这些指标在训练迭代过程中反映稳定性能，我们使用基于动量的方法对其进行更新，这平滑了波动并提供了每个块进展的更可靠指示。利用这些动量平滑的指标，我们识别出重建性能最高的块，并将其PSNR和SSIM值标记为$\mathrm { P S N R } _ { \operatorname* { m a x } }$和$\mathrm { S S I M } _ { \mathrm { m a x } }$。这些参考值作为评估每个块相对准确性的基准。对于场景中的每个块，我们计算偏差$\delta _ { p }$和$\delta _ { s }$以量化其重建与性能最高块的一致性。具体而言，PSNR偏差$\delta _ { p }$通过从$\mathrm { P S N R } _ { \operatorname* { m a x } }$中减去当前块的PSNR得到。$\delta _ { s }$以类似方式得出。在计算出这些偏差后，我们为每个块分配一个权重$w _ { i }$，反映其相对重建性能。权重$w _ { i }$的构建类似于高斯分布，更加重视偏离最佳性能块较大的块。通过优先关注重建准确性较低的块，该方法引导模型关注表现不佳的块，有助于提高场景整体一致性。此外，$w _ { i }$的值限制在略高于1的范围内，以防止过度的调整，确保训练动态的稳定性，避免对偏差适中的块过度惩罚。

$$
w _ { i } = 2 - \exp \left( \frac { \delta _ { p } ^ { 2 } + \lambda \cdot \delta _ { s } ^ { 2 } } { - 2 \sigma ^ { 2 } } \right) ,
$$

此设计引导高斯解码器关注全局场景，而不是聚焦于局部高质量重建的块，从而提高了所有块之间的一致性，最终提升了整体场景重建质量。

# 4. 实验

# 4.1. 实验设置

数据集和指标。我们在三个无人机拍摄的庞大场景数据集上进行了实验：Mill19数据集中的Building和Rubble，UrbanScene3D数据集中的Campus、Residence和Sci-Art，以及MatrixCity数据集中的Small City。每个数据集包含数千张高分辨率图像，其中MatrixCity场景特别覆盖了2.7平方公里的广阔区域。根据之前的方法，对所有场景的训练和测试图像进行了4倍下采样，唯一例外的是MatrixCity场景，我们将图像宽度调整为1600像素。我们使用PSNR、SSIM和LPIPS评估重建精度，并在评估期间另外报告了分配的内存和渲染帧率，以比较渲染性能。实现与比较方法，我们使用COLMAP中的稀疏点云作为初始输入。每个稀疏体素使用点云中对应的点初始化。根据之前的方法，每个块进行了60,000次迭代的优化。为了确保公平比较，我们采用了与CityGaussian相同的初始点云和场景分区策略，但块的数量显著减少。具体而言，我们将所有场景划分为8个块。此外，在计算指标时，我们应用了与DOGS相同的颜色校正方法。对于CityGaussian，我们使用作者发布的检查点，并应用相同的颜色校正方法进行评估。我们将我们的方法与Mega-NeRF、Switch-NeRF、3D-GS、VastGaussian、CityGaussian和DOGS进行了比较。所有实验均在具有24 GB内存的Nvidia RTX 3090 GPU上进行。

# 4.2. 结果分析

定量结果。在表1和表2中，我们展示了六个大规模场景的定量评估结果。我们提出的Momentum-GS始终实现最佳整体性能，显著优于其他方法。这些结果凸显了我们Momentum-GS在保留细节和提供高质量渲染方面的能力。值得注意的是，基于NeRF的方法在Sci-Art数据集上获得了更高的PSNR分数。这一现象可能源于Sci-Art源图像中固有的模糊性，可能是由于失焦拍摄条件造成的。由于基于NeRF的方法通常生成更平滑和模糊的重建，其输出自然与这些模糊的真实标注图像更为一致，从而导致更高的PSNR分数。然而，在考虑SSIM和LPIPS指标时，基于高斯的方法，通常是我们的Momentum-GS，在感知质量上显著优于基于NeRF的方法。视觉结果。在图4和图5中，我们提供了六个场景的重建结果的视觉对比。我们提出的Momentum-GS始终生成清晰和真实的图像，展示了在所有场景中出色的细节保留和视觉清晰度。相比之下，其他方法往往会出现明显的模糊和结构退化，尤其是在复杂区域。这些定性结果进一步突显了Momentum-GS在捕捉细粒度细节和维持整体渲染质量方面的有效性。

# 4.3. 消融研究

并行训练 vs. 独立训练。在表 5 中，我们展示了当场景被划分为相同数量的块时，并行训练 (b) 在重建质量上优于独立训练 (c)。这一改进源于共享高斯解码器可用的数据多样性的增加。然而，直接并行训练受到约束，必须保证块的数量与可用 GPU 的数量相匹配。因此，独立训练可以通过利用更多的块进一步提高准确性，而直接并行训练 (b) 仍受限于 GPU 的可用性。如表 5 所示，使用八个块的独立训练 (d) 产生了更好的性能并超越了 (b)。为了解决这一限制，我们引入了场景动量自蒸馏 (e)，使高斯解码器能够从增加的数据多样性中受益，同时将块的数量与 GPU 数量解耦。与 i $\mathrm { P S N R \uparrow }$、$\mathrm { { S S I M \uparrow } }$ 和 $\mathrm { L P I P S } \downarrow$ 在测试视图上的表现相比，我们的方法实现了显著的准确性提升。最佳和第二最佳分数已被突出显示。

<table><tr><td>Scene</td><td colspan="3">Building</td><td colspan="3">Rubble</td><td colspan="3">Campus</td><td colspan="3">Residence</td><td colspan="3">Sci-Art</td></tr><tr><td>Metrics</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS ↓ |</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>Mega-NeRF [55]</td><td>20.93</td><td>0.547</td><td>0.504</td><td>24.06</td><td>0.553</td><td>0.516</td><td>23.42</td><td>0.537</td><td>0.636</td><td>22.08</td><td>0.628</td><td>0.489</td><td>25.60</td><td>0.770</td><td>0.390</td></tr><tr><td>Switch-NRF 37]</td><td>21.54</td><td>0.579</td><td>0.474</td><td>24.31</td><td>0.562</td><td>0.496</td><td>23.62</td><td>0.541</td><td>0.616</td><td>22.57</td><td>0.654</td><td>0.457</td><td>26.52</td><td>0.7955</td><td>0.360</td></tr><tr><td>3D-GS [20]</td><td>22.53</td><td>0.738</td><td>0.214</td><td>25.51</td><td>0.725</td><td>0.316</td><td>23.67</td><td>0.688</td><td>0.347</td><td>22.36</td><td>0.745</td><td>0.247</td><td>24.13</td><td>0.791</td><td>0.262</td></tr><tr><td>VastGaussian [29]</td><td>21.80</td><td>0.728</td><td>0.225</td><td>25.20</td><td>0.742</td><td>0.264</td><td>23.82</td><td>0.695</td><td>0.329</td><td>21.01</td><td>0.699</td><td>0.261</td><td>22.64</td><td>0.761</td><td>0.261</td></tr><tr><td>CityGaussian [3]</td><td>22.70</td><td>0.774</td><td>0.246</td><td>26.45</td><td>0.809</td><td>0.232</td><td>22.80</td><td>0.662</td><td>0.437</td><td>23.35</td><td>0.822</td><td>0.211</td><td>24.49</td><td>0.843</td><td>0.232</td></tr><tr><td>DOOGS [9]</td><td>22.73</td><td>00.759</td><td>0.204</td><td>25.78</td><td>0.765</td><td>0.257</td><td>24.01</td><td>0.681</td><td>0.377</td><td>21..94</td><td>0.740</td><td>0.244</td><td>24.42</td><td>0.804</td><td>0.219</td></tr><tr><td>Momentum-GS (Ours) |</td><td>23.65</td><td>0.813</td><td>0.194</td><td>26.66</td><td>0.826</td><td>0.200</td><td>24.34</td><td>0.760</td><td>0.290</td><td>23.37</td><td>0.828</td><td>0.196</td><td>25.06</td><td>0.860</td><td>0.204</td></tr></table>

![](images/4.jpg)

Table 2. Quantitative comparison on the extremely large-scale urban scene, MatrixCity. We report $\mathrm { P S N R \uparrow }$ , $\mathrm { { S S I M \uparrow } }$ , and $\mathrm { L P I P S } \downarrow$ on test views, with the best results highlighted.   

<table><tr><td>Method</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS ↓</td></tr><tr><td>3D-GS [20]</td><td>27.36</td><td>0.818</td><td>0.237</td></tr><tr><td>VastGaussian [9]</td><td>28.33</td><td>0.835</td><td>0.220</td></tr><tr><td>CityGaussian [33]</td><td>28.61</td><td>0.868</td><td>0.205</td></tr><tr><td>DOGS [9]</td><td>28.58</td><td>0.847</td><td>0.219</td></tr><tr><td>Momentum-GS (Ours)</td><td>29.11</td><td>0.881</td><td>0.180</td></tr></table>

独立训练八个块。此外，结合重建引导的块加权（表示为“(f) 完整”）进一步提高了整体重建质量。

Table 3. We report the allocated memory (in GB) and rendering framerate (in FPS) measured during evaluation on the extremely large scene MatrixCity, with the best results highlighted.   

<table><tr><td>Method</td><td>3D-GS</td><td>VastGaussian</td><td>CityGaussian</td><td>DOGS</td><td>Momentum-GS (Ours)</td></tr><tr><td>FPS ↑</td><td>45.57</td><td>40.04</td><td>26.10</td><td>48.34</td><td>59.91</td></tr><tr><td>Mem ↓</td><td>6.31</td><td>6.99</td><td>14.68</td><td>5.82</td><td>4.62</td></tr></table>

块加权。在表4中，我们评估了不同的方法来衡量每个块的重建质量。结果显示，组合使用PSNR和SSIM的准确性高于单独使用任一种。可扩展性。我们在不同数量的分块上评估了我们的方法，同时保持GPU数量恒定为四个。

![](images/5.jpg)

Table 4. Ablation study on different strategy of measuring the reconstruction quality in block weighting.   

<table><tr><td>Models</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>w/PSNR</td><td>23.49</td><td>0.809</td><td>0.197</td></tr><tr><td>w/ SSIM</td><td>23.53</td><td>0.806</td><td>0.203</td></tr><tr><td>Full (PSNR + SSIM)</td><td>23.65</td><td>0.813</td><td>0.194</td></tr></table>

Table 5. Ablation study on different training strategies.   

<table><tr><td>Training strategy</td><td>#Block</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS ↓</td></tr><tr><td>(a) baseline</td><td>1</td><td>22.25</td><td>0.742</td><td>0.272</td></tr><tr><td>(b) w/ Parallel training</td><td>4</td><td>23.10</td><td>0.790</td><td>0.221</td></tr><tr><td>(c) w/ Independent training</td><td>4</td><td>22.85</td><td>0.781</td><td>0.229</td></tr><tr><td>(d) w/ Independent training</td><td>8</td><td>23.23</td><td>0.796</td><td>0.211</td></tr><tr><td>(e) w/ momentum self-distill.</td><td>8</td><td>23.56</td><td>0.806</td><td>0.205</td></tr><tr><td>(f) Full</td><td>8</td><td>23.65</td><td>0.813</td><td>0.194</td></tr></table>

如表6所示，随着块数的增加，重建质量不断提高，证明了我们方法在有限GPU资源下的可扩展性。

# 5. 结论

在本文中，我们介绍了 Momentum-GS，这是一种基于动量的自蒸馏框架，显著增强了大规模场景重建中的 3D 高斯点云技术。Momentum-GS 的核心是一个动量更新的教师高斯解码器，它作为一个稳定的全局参考，用于指导并行训练块，有效促进重建场景的空间一致性和连贯性。我们进一步引入了一种重建引导的块加权机制，根据重建质量动态调整对每个块的重视程度，从而进一步提升整体一致性。我们的方法利用混合表示，整合隐式和显式特征，以实现灵活的扩展，使块的数量与 GPU 限制解耦。实验结果证明了混合表示和基于动量的自蒸馏在鲁棒的大规模 3D 场景重建中的强大能力。

Table 6. Ablation study on the different number of divided blocks.   

<table><tr><td>Method</td><td>#Block</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>CityGaussian</td><td>32</td><td>28.61</td><td>0.868</td><td>0.205</td></tr><tr><td>Momentum-GS (Ours)</td><td>4</td><td>28.93</td><td>0.870</td><td>0.203</td></tr><tr><td>Momentum-GS (Ours)</td><td>8</td><td>29.11</td><td>0.881</td><td>0.180</td></tr><tr><td>Momentum-GS (Ours)</td><td>16</td><td>29.15</td><td>0.884</td><td>0.172</td></tr></table>

# 致谢

本研究得到了广东省杰出青年学者自然科学基金（编号：2025B1515020012）和深圳市科技计划（编号：JCYJ20240813111903006）的支持。

# References

[1] Sameer Agarwal, Yasutaka Furukawa, Noah Snavely, Ian Simon, Brian Curless, Steven M Seitz, and Richard Szeliski. Building rome in a day. Communications of the ACM, 54 (10):105112, 2011. 3   
[2] Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In ICCV, pages 58555864, 2021. 2, 3   
[3] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In CVPR, pages 5470 5479, 2022.   
[4] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased gridbased neural radiance fields. In ICCV, pages 1969719705, 2023. 2, 3   
[5] Ilker Bozcan and Erdal Kayacan. Au-air: A multi-modal unmanned aerial vehicle dataset for low altitude traffic surveillance. In 2020 IEEE International Conference on Robotics and Automation (ICRA), pages 85048510. IEEE, 2020. 1   
[6] Guikun Chen and Wenguan Wang. A survey on 3d gaussian splatting. arXiv preprint arXiv:2401.03890, 2024. 1   
[7] Junyi Chen, Weicai Ye, Yifan Wang, Danpeng Chen, Di Huang, Wanli Ouyang, Guofeng Zhang, Yu Qiao, and Tong He. Gigags: Scaling up planar-based 3d gaussians for large scene surface reconstruction. arXiv preprint arXiv:2409.06685, 2024. 3   
[8] Timothy Chen, Ola Shorinwa, Joseph Bruno, Javier Yu, Weijia Zeng, Keiko Nagami, Philip Dames, and Mac Schwager. Splat-nav: Safe real-time robot navigation in gaussian splatting maps. arXiv preprint arXiv:2403.02751, 2024. 1   
[9] Yu Chen and Gim Hee Lee. Dogs: Distributed-oriented gaussian splatting for large-scale 3d reconstruction via gaussian consensus. In NeurIPS, 2024. 2, 3, 6, 7   
10] Zilong Chen, Feng Wang, Yikai Wang, and Huaping Liu. Text-to-3d using gaussian splatting. In CVPR, pages 21401 21412, 2024. 3   
11Jiadi Cui, Junming Cao, Yuhui Zhong, Liao Wang, Fuqang Zhao, Penghao Wang, Yifan Chen, Zhipeng He, Lan Xu, Yujiao Shi, et al. Letsgo: Large-scale garage modeling and rendering via lidar-assisted gaussian primitives. arXiv preprint arXiv:2404.09748, 2024. 3   
12 Xiao Cui, Weicai Ye, Yifan Wang, Guoeg Zhang, Wengang Zhou, Tong He, and Houqiang Li. Streetsurfgs: Scalable urban street surface reconstruction with planar-based gaussian splatting. arXiv preprint arXiv:2410.04354, 2024.   
13] Guofeng Feng, Siyan Chen, Rong Fu, Zimu Liao, Yi Wan Tao Liu, Zhilin ei, Henge Li, Xingheg Zang, large-scale and high-resolution rendering. arXiv preprint arXiv:2408.07967, 2024. 3   
[14] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In CVPR, pages 55015510, 2022. 3   
[15] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk Warburg, Benjamin Recht, and Angjoo Kanazawa. K-planes: Explicit radiance fields in space, time, and appearance. In CVPR, pages 1247912488, 2023. 3   
[16] Jiaming Gu, Minchao Jiang, Hongsheng Li, Xiaoyuan Lu, Guangming Zhu, Syed Afaq Ali Shah, Liang Zhang, and Mohammed Bennamoun. Ue4-nerf: Neural radiance field for real-time rendering of large-scale scene. NeurIPS, 36, 2024. 1   
[17] Changjian Jiang, Ruilan Gao, Kele Shao, Yue Wang, Rong Xiong, and Yu Zhang. Li-gs: Gaussian splatting with lidar incorporated for accurate large-scale reconstruction. arXiv preprint arXiv:2409.12899, 2024. 3   
[18] Ying Jiang, Chang Yu, Tianyi Xie, Xuan Li, Yutao Feng, Huamin Wang, Minchen Li, Henry Lau, Feng Gao, Yin Yang, et al. Vr-gs: A physical dynamics-aware interactive gaussian splatting system in virtual reality. In ACM SIGGRAPH 2024 Conference Papers, pages 11, 2024. 1   
[19] Rui Jin, Yuman Gao, Yingjian Wang, Haojian Lu, and Fei Gao. Gs-planner: A gaussian-splatting-based planning framework for active high-fidelity reconstruction. arXiv preprint arXiv:2405.10142, 2024. 1   
[20] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. TOG, 42(4), 2023. 1, 3, 6, 7   
[21] Bernhard Kerbl, Andreas Meuleman, Georgios Kopanas, Michael Wimmer, Alexandre Lanvin, and George Drettakis. A hierarchical 3d gaussian representation for real-time rendering of very large datasets. T0G, 43(4):115, 2024. 2, 3   
[22] Xiaohan Lei, Min Wang, Wengang Zhou, and Houqiang Li. Gaussnav: Gaussian splatting for visual navigation. arXiv preprint arXiv:2403.11625, 2024. 1   
[23] Bingling Li, Shengyi Chen, Luchao Wang, Kaimin Liao, Sijie Yan, and Yuanjun Xiong. Retinags: Scalable training for dense scene rendering with billion-scale 3d gaussians. arXiv preprint arXiv:2406.11836, 2024. 3   
[24] Ruilong Li, Sanja Fidler, Angjoo Kanazawa, and Francis Williams. NeRF-XL: Scaling nerfs with multiple GPUs. In ECCV, 2024. 2, 3   
[25] Wei Li, CW Pan, Rong Zhang, JP Ren, YX Ma, Jin Fang, FL Yan, QC Geng, XY Huang, HJ Gong, et al. Aads: Augmented autonomous driving simulation using data-driven algorithms. Science robotics, 4(28):eaaw0863, 2019. 1   
[26] Wanhua Li, Renping Zhou, Jiawei Zhou, Yingwei Song, Johannes Herter, Minghan Qin, Gao Huang, and Hanspeter Pfister. 4d langsplat: 4d language gaussian splatting via multimodal large language models. In CVPR, pages 22001 22011, 2025. 3   
[27] Yixuan Li, Lihan Jiang, Linning Xu, Yuanbo Xiangli, Zhenzhi Wang, Dahua Lin, and Bo Dai. Matrixcity: A large-scale c uataset l ci-sa nual elucig anu veyonu. II ICCV, pages 32053215, 2023. 3, 6   
[28] Zhuopeng Li, Yilin Zhang, Chenming Wu, Jianke Zhu, and Liangjun Zhang. Ho-gaussian: Hybrid optimization of 3d gaussian splatting for urban scenes. arXiv preprint arXiv:2403.20032, 2024. 2, 3   
[29] Jiaqi Lin, Zhihao Li, Xiao Tang, Jianzhuang Liu, Shiyong Liu, Jiayue Liu, Yangdi Lu, Xiaofei Wu, Songcen Xu, Youliang Yan, and Wenming Yang. Vastgaussian: Vast 3d gaussians for large scene reconstruction. In CVPR, 2024. 2, 3, 6, 7   
[30] Liqiang Lin, Yilin Liu, Yue Hu, Xingguang Yan, Ke Xie, and Hui Huang. Capturing, reconstructing, and simulating: the urbanscene3d dataset. In ECCV, pages 93109, 2022. 3, 6   
[31] Jinpeng Liu, Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Ying Shan, and Yansong Tang. Novelgs: Consistent novel-view denoising via large gaussian reconstruction model. arXiv preprint arXiv:2411.16779, 2024. 3   
[32] Shuhong Liu, Xiang Chen, Hongming Chen, Quanfeng Xu, and Mingrui Li. Deraings: Gaussian splatting for enhanced scene reconstruction in rainy. arXiv preprint arXiv:2408.11540, 2024. 1   
[33] Yang Liu, He Guan, Chuanchen Luo, Lue Fan, Naiyan Wang, Junran Peng, and Zhaoxiang Zhang. Citygaussian: Real-time high-quality large-scale scene rendering with gaussians. In ECCV, 2024. 2, 3, 6, 7   
[34] Guanxing Lu, Shiyi Zhang, Ziwei Wang, Changliu Liu, Jiwen Lu, and Yansong Tang. Manigaussian: Dynamic gaussian splatting for multi-task robotic manipulation. In ECCV, pages 349366, 2025. 1   
[35] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In CVPR, pages 2065420664, 2024. 2, 3   
[36] Ricardo Martin-Brualla, Noha Radwan, Mehdi SM Sajjadi, Jonathan T Barron, Alexey Dosovitskiy, and Daniel Duckworth. Nerf in the wild: Neural radiance fields for unconstrained photo collections. In CVPR, pages 72107219, 2021. 3   
[37] Zhenxing Mi and Dan Xu. Switch-nerf: Learning scene decomposition with mixture of experts for large-scale neural radiance fields. In ICLR, 2023. 3, 6, 7   
[38] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99106, 2021. 2, 3   
[39] Ben Mildenhall, Peter Hedman, Ricardo Martin-Brualla, Pratul P Srinivasan, and Jonathan T Barron. Nerf in the dark: High dynamic range view synthesis from noisy raw images. In CVPR, pages 1619016199, 2022. 3   
[40] Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. T0G, 41(4):115, 2022. 3   
[41] Michael Niemeyer, Jonathan T Barron, Ben Mildenhall, Mehdi SM Sajjadi, Andreas Geiger, and Noha Radwan. Renerf: Regularizing neural radiance fields for view synthesis from sparse inputs. In CVPR. nages 54805490. 2022. 3   
[42] Julian Ost, Fahim Mannan, Nils Thuerey, Julian Knodt, and Felix Heide. Neural scene graphs for dynamic scenes. In CVPR, pages 28562865, 2021. 1   
[43] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In CVPR, pages 1031810327, 2021. 3   
[44] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. Langsplat: 3d language gaussian splatting. In CVPR, pages 2005120060, 2024. 3   
[45] Christian Reiser, Rick Szeliski, Dor Verbin, Pratul Srinivasan, Ben Mildenhall, Andreas Geiger, Jon Barron, and Peter Hedman. Merf: Memory-efficient radiance fields for realtime view synthesis in unbounded scenes. TOG, 42(4):112, 2023. 3   
[46] Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu, Zhangkai Ni, and Bo Dai. Octree-gs: Towards consistent real-time rendering with lod-structured 3d gaussians. arXiv preprint arXiv:2403.17898, 2024. 2, 3   
[47] Xuanchi Ren, Yifan Lu, Hanxue Liang, Jay Zhangjie Wu, Huan Ling, Mike Chen, Francis Fidler, Sanja annd Williams, and Jiahui Huang. Scube: Instant large-scale scene reconstruction using voxsplats. In NeurIPS, 2024. 3   
[48] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 41044113, 2016. 6   
[49] Jiansong Sha, Haoyu Zhang, Yuchen Pan, Guang Kou, and Xiaodong Yi. Nerf-is: Explicit neural radiance fields in semantic space. In Proceedings of the 5th ACM International Conference on Multimedia in Asia, pages 17, 2023. 3   
[50] Shuai Shen, Wanhua Li, Xiaoke Huang, Zheng Zhu, Jie Zhou, and Jiwen Lu. Sd-nerf: Towards lifelike talking head animation via spatially-adaptive dual-driven nerfs. IEEE Transactions on Multimedia, 26:32213234, 2023. 3   
[51] Surendra Pal Singh, Kamal Jain, and V Ravibabu Mandla. 3d scene reconstruction from video camera for virtual 3d city modeling. American Journal of Engineering Research, 3(1): 140148, 2014. 1   
[52] Noah Snavely, Steven M. Seitz, and Richard Szeliski. Photo Tourism: Exploring Photo Collections in 3D. Association for Computing Machinery, 2023. 3   
[53] Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Pradhan, Ben Mildenhall, Pratul P Srinivasan, Jonathan T Barron, and Henrik Kretzschmar. Block-nerf: Scalable large scene neural view synthesis. In CVPR, pages 82488258, 2022. 1, 3   
[54] Matthew Tancik, Ethan Weber, Evonne Ng, Ruilong Li, Brent Yi, Terrance Wang, Alexander Kristoffersen, Jake Austin, Kamyar Salahi, Abhik Ahuja, et al. Nerfstudio: A modular framework for neural radiance field development. In ACM SIGGRAPH 2023 Conference Proceedings, pages 112, 2023. 3   
[55] Haithem Turki, Deva Ramanan, and Mahadev Satyanarayanan. Mega-nerf: Scalable construction of large-scale nerfs for virtual fly-throughs. In CVPR, pages 1292212931, 2022. 1, 3, 6, 7   
[56] Haithem Turki, Vasu Agrawal, Samuel Rota Bulò, Lorenzo Porzi, Peter Kontschieder, Deva Ramanan, Michael Zollhöfer, and Christian Richardt. Hybridnerf: Efficient neural rendering via adaptive volumetric surfaces. In CVPR, pages 1964719656, 2024. 3   
[57] Zipeng Wang and Dan Xu. Pygs: Large-scale scene representation with pyramidal 3d gaussian splatting. arXiv preprint arXiv:2405.16829, 2024. 3   
[58] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. TIP, 13(4):600612, 2004. 5, 6   
[59] Dejia Xu, Yifan Jiang, Peihao Wang, Zhiwen Fan, Humphrey Shi, and Zhangyang Wang. Sinnerf: Training neural radiance fields on complex scenes from a single image. In ECCV, pages 736753, 2022. 3   
[60] Linning Xu, Yuanbo Xiangli, Sida Peng, Xingang Pan, Nanxuan Zhao, Christian Theobalt, Bo Dai, and Dahua Lin. Grid-guided neural radiance fields for large urban scenes. In CVPR, pages 82968306, 2023. 3   
[61] Daniel Yang, John J. Leonard, and Yogesh Girdhar. Seasplat: Representing underwater scenes with 3d gaussian splatting and a physically grounded image formation model. arxiv, 2024. 1   
[62] Zhenpei Yang, Yuning Chai, Dragomir Anguelov, Yin Zhou, Pei Sun, Dumitru Erhan, Sean Rafferty, and Henrik Kretzschmar. Surfelgan: Synthesizing realistic sensor data for autonomous driving. In CVPR, pages 1111811127, 2020. 1   
[63] Chubin Zhang, Hongliang Song, Yi Wei, Yu Chen, Jiwen Lu, and Yansong Tang. Geolrm: Geometry-aware large reconstruction model for high-quality 3d gaussian generation. arXiv preprint arXiv:2406.15333, 2024. 3   
[64] Hanyue Zhang, Zhiliu Yang, Xinhe Zuo, Yuxin Tong, Ying Long, and Chen Liu. Garfield $^ { + + }$ : Reinforced gaussian radiance fields for large-scale 3d scene reconstruction. arXiv preprint arXiv:2409.12774, 2024. 3   
[65] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, pages 586595, 2018.6   
[66] Yuqi Zhang, Guanying Chen, and Shuguang Cui. Efficient large-scale scene representation with a hybrid of highresolution grid and plane features. Pattern Recognition, 158: 111001, 2025. 3

# Momentum-GS: Momentum Gaussian Self-Distillation for High-Quality Large Scene Reconstruction

Supplementary Material

# A. More Details

Scene partition. (1) Criteria: The scene is first equally divided along the $\mathbf { X }$ -axis and then along the z-axis, with each block having the same area. Corresponding views are selected based on visibility. (2) Initialization: Each block is initialized from the same point cloud generated by COLMAP, but only the assigned part and overlapping boundary are reconstructed. (3) Views selection: Views at the boundaries are selected based on visibility, and each block reconstructs an extended region to ensure better reconstruction quality at the boundary area.

Motivation of momentum updates. The momentumbased update provides stable, global guidance, allowing each block's Gaussian decoder to effectively leverage the broader scene context, thereby significantly enhancing reconstruction consistency. As demonstrated in Table 9, using a momentum value of 0.9 outperforms a setting without momentum updates.

# B. More Ablation Study

Effectiveness of self-distillation. As shown in Table 7, we performed additional experiments to validate the effectiveness of our self-distillation approach: (1) As shown in setting (b), extending parallel training to 8 blocks with 8 GPUs improved the reconstruction quality. (2) Alternating training across blocks every 500 iterations, using 4 GPUs to train 8 blocks in parallel (setting (c)), slightly decreased the reconstruction quality compared with setting (b). (3) Incorporating our momentum-based self-distillation into setting (c) enhanced the reconstruction quality (setting (d)), clearly demonstrating the effectiveness of our proposed method.

Table 7. Ablation study on different training strategies.   

<table><tr><td>Training strategy</td><td>#Block</td><td>#GPU</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS ↓</td></tr><tr><td>(a) w/ Parallel training</td><td>4</td><td>4</td><td>23.10</td><td>0.790</td><td>0.221</td></tr><tr><td>(b) w/ Parallel training</td><td>8</td><td>8</td><td>23.34</td><td>0.800</td><td>0.210</td></tr><tr><td>() w/Parallel training (alterating)</td><td>8</td><td>4</td><td>23.17</td><td>0.797</td><td>0.211</td></tr><tr><td>(d) w/ momentum self-distill.</td><td>8</td><td>4</td><td>23.56</td><td>0.806</td><td>0.205</td></tr><tr><td>(e) Full</td><td>8</td><td>4</td><td>23.65</td><td>0.813</td><td>0.194</td></tr></table>

The weight of consistency loss. An ablation study is performed to evaluate the impact of the consistency loss weight $\lambda _ { c o n s i s t e n c y }$ . As reported in Table 8, the results indicate that model performance remains stable across a wide range of $\lambda _ { c o n s i s t e n c y }$ values.

Momentum value. We ablated momentum value $m$ and

Table 8. Ablation study on λconsistency.   

<table><tr><td>Scene</td><td colspan="3">Building</td><td colspan="3">Rubble</td></tr><tr><td>λconsistency</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>1</td><td>23.53</td><td>0.808</td><td>0.201</td><td>26.51</td><td>0.816</td><td>0.210</td></tr><tr><td>10</td><td>23.63</td><td>0.810</td><td>0.200</td><td>26.62</td><td>0.821</td><td>0.204</td></tr><tr><td>50</td><td>23.65</td><td>0.813</td><td>0.194</td><td>26.66</td><td>0.826</td><td>0.200</td></tr><tr><td>100</td><td>23.63</td><td>0.810</td><td>0.197</td><td>26.69</td><td>0.829</td><td>0.198</td></tr></table>

Table 9. Comparison between different momentum values.   

<table><tr><td>Momentum values</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>0.0</td><td>23.44</td><td>0.806</td><td>0.203</td></tr><tr><td>0.5</td><td>23.59</td><td>0.808</td><td>0.201</td></tr><tr><td>0.7</td><td>23.62</td><td>0.810</td><td>0.198</td></tr><tr><td>0.9 (default)</td><td>23.65</td><td>0.813</td><td>0.194</td></tr><tr><td>0.95</td><td>23.50</td><td>0.806</td><td>0.201</td></tr><tr><td>0.99</td><td>22.06</td><td>0.741</td><td>0.254</td></tr></table>

Table 9 shows that our model is robust to variations. The reconstruction quality show minimal differences, with the best performance achieved at $\mathrm { m } { = } 0 . 9$ (our default setting).

# C. Quantitative Evaluation

VRAM. We report the peak VRAM usage during inference across five large-scale scenes, as shown in Table 10. Despite achieving superior reconstruction quality, our method requires less VRAM compared to the purely 3DGS-based approach. The VRAM usage, measured in MB, highlights the efficiency of our method. Notably, as scene complexity increases (e.g., in MatrixCity), the advantages of our method become even more pronounced.

Table 10. Peak VRAM usage (in MB) during inference.   

<table><tr><td>Scene</td><td>Building</td><td>Rubble</td><td>Residence</td><td>Sci-Art</td><td>MatrixCity</td></tr><tr><td>CityGaussian</td><td>8977</td><td>5527</td><td>6494</td><td>2726</td><td>14677</td></tr><tr><td>Momentum-GS (Ours)</td><td>5830</td><td>4106</td><td>6419</td><td>6647</td><td>4616</td></tr></table>

Storage. We report the storage usage across five large-scale scenes, as shown in Table 11. Leveraging our hybrid representation, our method significantly reduces the number of parameters required for storage compared to purely 3DGSbased methods. This reduction is especially notable in larger and more complex scenes, such as MatrixCity, where the storage savings are most substantial. Notably, as scene complexity increases (e.g., in MatrixCity), the advantages of our method become even more pronounced, demonstrating its effectiveness in handling challenging scenarios. For clarity and consistency, storage usage is reported in GB.

![](images/6.jpg)  
Figure 6. Qualitative comparisons of our Momentum-GS and prior methods across four large-scale scenes.

Table 11. Storage usage (in GB).   

<table><tr><td>Scene</td><td>Building</td><td>Rubble</td><td>Residence</td><td>Sci-Art</td><td>MatrixCity</td></tr><tr><td>CityGaussian</td><td>3.07</td><td>2.22</td><td>2.49</td><td>0.88</td><td>5.40</td></tr><tr><td>Momentum-GS (Ours)</td><td>2.45 (20.2%↓)</td><td>1.50 (32.7%↓)</td><td>2.00 (19.7%↓)</td><td>0.97</td><td>2.08 (61.5%↓)</td></tr></table>

Table 13. Comparison of different implementations of VastGaussian.   

<table><tr><td>Scene</td><td colspan="3">Building</td><td colspan="3">Rubble</td></tr><tr><td>Metrics</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>VastGaussian (DOGS version)</td><td>21.80</td><td>0.728</td><td>0.225</td><td>25.20</td><td>0.742</td><td>0.264</td></tr><tr><td>VastGaussian (Unofficial)</td><td>22.49</td><td>0.742</td><td>0.208</td><td>25.64</td><td>0.760</td><td>0.202</td></tr><tr><td>Momentum-GS (Ours)</td><td>23.65</td><td>0.813</td><td>0.194</td><td>26.66</td><td>0.826</td><td>0.200</td></tr></table>

Number of primitives. We report the number of primitives across five large-scale scenes, as shown in Table 12.   
Table 12. Primitives counts for each scene.   

<table><tr><td>Scene</td><td>Building</td><td>Rubble</td><td>Residence</td><td>Sci-Art</td><td>MatrixCity</td></tr><tr><td>Primitives</td><td>8.33M</td><td>5.09M</td><td>6.79M</td><td>3.30M</td><td>7.08M</td></tr></table>

Comparison of different implementations of VastGaussian. We further compare our method with the unofficial implementation of VastGaussian in Table 13, which demonstrates improved performance over the results reported in DOGS.

# D. More Visual Comparisons

We provide additional visual comparisons for the Building, Rubble, Residence, and Sci-Art scenes in Figure 6. Our method consistently reconstructs finer details across these scenes. Notably, our approach demonstrates a superior ability to reconstruct luminance, as illustrated by the Sci-Art example shown in Figure 6. While NeRF-based methods are capable of capturing luminance by leveraging neural networks to learn global features such as lighting, they tend to produce blurrier results compared to 3DGS-based methods. This underscores the effectiveness of our hybrid representation, which combines the strengths of both NeRF-based and 3DGS-based approaches.