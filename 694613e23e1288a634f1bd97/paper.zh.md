# Splatt3R：来自未校准图像对的零次学习高斯溅射

布兰登·斯马特 ﾭ 郑传侠 2 伊罗·莱娜 2 维克多·阿德里安·普里萨卡里乌 1 牛津大学主动视觉实验室 2 牛津大学视觉几何组 {brandon, cxzheng, iro, victor}@robots.ox.ac.uk

![](images/1.jpg)  
F

# 摘要

在本文中，我们介绍了Splatt3R，这是一种无姿态、前馈的野外3D重建和新视图合成方法，适用于双目图像对。给定未校准的自然图像，Splatt3R能够预测3D高斯点云，无需任何相机参数或深度信息。为了提高通用性，我们在一个“基础”3D几何重建方法MASt3R的基础上构建Splatt3R，扩展其以处理3D结构和外观。具体来说，与仅重建3D点云的原始MASt3R不同，我们预测构建每个点所需的额外高斯属性。因此，与其他新视图合成方法不同，Splatt3R首先通过优化3D点云的几何损失进行训练，然后再进行新视图合成目标的训练。通过这种方式，我们避免了从立体视图训练3D高斯点云时出现的局部最小值。我们还提出了一种新颖的损失屏蔽策略，我们实证发现这对于推断视点的强大性能至关重要。我们在ScanNet $\mathrel { + { + } }$ 数据集上训练了Splatt3R，并展示了其在未校准的野外图像上的出色泛化能力。Splatt3R可以以4FPS的速度在$5 1 2 \times 5 1 2$ 分辨率下重建场景，得到的点云可以实时渲染。

# 1. 引言

我们考虑从稀疏、未标定的自然图像中进行 3D 场景重建和新视角合成的问题，仅通过经过训练的模型的一次前向传播来实现。虽然近年来通过使用神经场景表示（例如 SRN、NeRF、LFN）以及非神经场景表示（例如 3D 高斯溅射）在 3D 重建和新视角合成方面取得了突破性进展，但由于昂贵的、迭代的每个场景优化过程，这些方法离普通用户的可及性仍然很远，这些过程往往速度较慢，且无法利用训练数据集中的学习到的先验知识。更重要的是，当仅使用一对立体图像进行训练时，重建质量较差，因为这些方法需要从数十张或数百张图像中密集收集，以产生高质量的结果。

为了解决这些问题，通用的三维重构模型[8, 12, 16, 27, 56, 63]旨在利用前馈网络从稀疏的标定图像中预测与像素对齐的辐射场特征。这些模型通过差分渲染预测的、参数化的表示，从目标视角出发，并使用从相同相机姿态捕获的真实图像进行监督来进行训练。通过学习大规模输入场景数据集的先验，这些模型避免了传统基于场景的优化在稀疏图像下的失败案例。为了避免在NeRF中昂贵的体积处理，提出了几种前馈高斯点云模型[7, 10, 51, 52, 68]来探索稀疏视图下的三维重构。它们使用与像素对齐的三维高斯原语云[29]来表示场景。这些高斯原语的三维位置通过它们沿光线的深度进行参数化，这个深度是通过输入图像中的已知内参数和外参数显式计算的。由于这些方法依赖于已知的相机参数，因此无法直接应用于“野外”非标定图像。假设已知真实姿态，或在预处理步骤中隐含相机姿态估计——现有方法通常在通过SfM软件对同一场景的数十或数百张图像进行重建后进行测试。尝试使用SfM或多视角立体（MVS）管道的方法通常需要一系列算法来匹配点、进行三角测量、查找本质矩阵以及估计相机的外参数和内参数。在本文中，我们介绍了Splatt3R，这是一种前馈模型，输入为两幅未标定图像，输出为表示场景的三维高斯。具体而言，我们使用前馈模型为每幅图像预测与像素对齐的三维高斯原语，然后使用可微渲染器渲染新视图。我们在这里并不依赖于任何额外的信息，如相机内参数、外参数或深度。在没有显式姿态信息的情况下，一个关键挑战是确定放置三维高斯中心的位置。即使在有姿态信息的情况下，迭代的三维高斯点云优化也容易受到局部最小值的影响[7, 29]。我们的解决方案是通过显式监督和回归每个训练样本的真实三维点云，联合解决相机姿态缺失和局部最小值的问题。特别地，我们观察到用于生成MASt3R的与像素对齐的三维点云[31]的架构与使用前馈高斯方法的现有与像素对齐的三维高斯点云架构密切一致[7, 10, 51, 52]。因此，我们试图证明，简单地在大型预训练“基础”三维MASt3R模型中添加一个高斯解码器，且不做任何额外处理，足以开发出一个无姿态、通用的新视图合成模型。

大多数现有的通用3D-GS方法的一个显著局限性在于，它们仅对输入立体视图之间的新观点进行监督[7, 10]，而不是学习推断更远的观点。这些推断的观点面临的挑战是，它们通常会看到输入相机视图无法看到的被遮挡点，或者完全位于其视锥体之外。因此，对这些点的新视图渲染损失进行监督是适得其反的，并可能对模型性能产生破坏。通过仅对两个上下文图像之间的视图进行新视图渲染损失的监督，现有工作避免尝试重建场景中许多未见部分。然而，这意味着模型未能被训练以准确生成超出立体基线的新视图渲染。为了解决这一问题，我们采用了一种基于视锥剔除和共可见性测试的损失屏蔽策略，该策略利用在训练期间已知的真实姿态和深度图进行计算。我们仅将均方误差和LPIPS损失应用于可以有效重建的渲染部分，从而防止模型更新来自场景未见部分。这使得可以使用更宽的基线进行训练，并对超出立体基线的新视图进行监督。我们首次提出了一种方法，该方法在网络的单次前向传递中，从一对未定姿态图像预测3D高斯点云以进行场景重建和新视图合成。我们基于现有工作构建基准，并证明我们的方法在视觉质量和感知相似性方面超越了它们。更令人印象深刻的是，我们训练的模型能够从野外未校准图像生成照片级真实的新视图合成。这大大减少了对具备精确相机姿态的稠密图像输入的需求，解决了该领域的一大挑战。

# 2. 相关工作

# 2.1. 新视角合成

3D新视图合成（NVS）采用了多种表示方式，如亮度图 [19]、光场 [32] 和全光函数 [1]。神经辐射场（NeRF）通过使用视角依赖的光线追踪辐射场，利用经过每场景优化训练的神经网络对密集收集的图像集进行编码，实现了3D场景的照片级真实感表示 [3, 41, 42]。近年来，3D高斯点云 [29] 通过训练一组3D高斯基元来表示空间中每个点的辐射，从而大大提高了辐射场的训练和渲染速度，并通过高效的光栅化过程进行渲染。为了避免密集的每场景优化，开发了可泛化的 NVS 流程，直接从多视图图像中推断3D表示 [8, 12, 27, 33, 36, 37, 46, 49, 50, 56, 59, 63, 68]。这些方法不再进行每场景的优化，而是在大型场景数据集上进行训练，从而学习数据驱动的先验知识，为新观察到的场景提供支撑重建。通过利用这些数据驱动的先验，这些方法已经演化为能够处理稀疏图像集 [10, 34, 38, 43]，甚至立体图像对 [7, 16, 30, 69]，显著减少了获取NVS所需的参考图像数量。

最近的方法，例如 pixelSplat [7]、MVSplat [10]、GPS-Gaussian [68]、SplatterImage [52] 和 Flash3D [52]，利用沿着相机射线放置的 3D 高斯原语云，这些射线是根据相机参数显式计算的，旨在每幅图像中预测一个（或多个）每像素的 3D 高斯原语。然而，这些现有方法假设在测试时每幅图像都有相机内参和外参，这限制了它们在野外照片对中的适用性。虽然已经提出了许多方法来进行场景优化，但这些方法依赖于大量图像 [4, 5, 25, 35, 58]。最近的研究提出了以可推广的方式联合预测相机参数和 3D 表示的方法，尽管这些方法仅限于稀疏设置 [26, 54]。与之相对，我们提出了 Splatt3R，以填补未知相机参数下可推广立体 NVS 的空白。在相关的研究中，FlowCam [49] 通过光流的稠密对应关系消除了对预计算相机的需求，但它需要顺序输入，并表现出有限的渲染性能。通过将最近的立体重建工作 MASt3R 与 3D 高斯结合，我们的方法在不需要预处理相机的情况下有效处理更大的基线。GGRt [33] 也试图模拟没有已知相机姿态或内参的 3D 高斯 Splat，但它专注于处理帧间基线较小的视频序列，引入了缓存和延迟反向传播技术，以帮助从长视频序列中重建。DBARF [9] 也旨在联合学习相机姿态和重建辐射场，但采用基于 NeRF 的方法，并专注于使用从学习特征中得出的成本图计算姿态。

# 2.2. 立体重建

传统的立体重建任务涉及一系列步骤。首先进行关键点检测和特征匹配，然后使用基础矩阵估计相机参数。接下来，通过极线搜索或立体匹配建立稠密对应，从而实现三维点的三角化。该过程可以通过光度束调整进行可选择性的优化。随着深度学习的出现，提出了许多方法来集成某些步骤，如联合深度和相机姿态估计以及光流。然而，所有这些方法都依赖明确的对应关系，这使得在图像重叠有限时应用起来具有挑战性。最近，DUSt3R提出了一种创新的方法，通过预测两幅未标定立体图像在一个坐标系下的点图，并采用隐式对应搜索来解决这一挑战。后续论文MASt3R主要关注图像匹配的改进，但通过在度量空间中预测点，提升了DUSt3R的准确性。这些方法在图像之间几乎没有重叠的情况下也显示出了有希望的立体重建结果。尽管原始点图对于姿态估计等多个下游应用具有足够的准确性，但并未设计为直接渲染。相比之下，我们的方法增强了MASt3R，以预测3D高斯基元，从而实现快速且逼真的NVS。

# 3. 方法

给定两幅未校准图像 ${ \mathcal { T } } ~ = ~ \{ { \bf I } ^ { i } \} _ { i = \{ 1 , 2 \} }$，$( \mathbf { I } ^ { i } \in$ $\mathbb { R } ^ { H \times W \times 3 } )$，我们的目标是学习一个映射 $\Phi$，它以 $\mathcal { T } $ 为输入，并输出用于几何和外观的 3D 高斯参数。我们通过向 MASt3R 添加一个第三分支来简单实现这一点，从而输出 3D 高斯所需的附加属性。在概述我们提出方法的细节之前，我们将首先在第 3.1 节中简要介绍 3D 高斯散射，接着在第 3.2 节中概述 MASt3R。然后，我们将在第 3.3 节中描述如何修改 MASt3R 结构以预测用于新视图合成的 3D 高斯散射。最后，我们将在第 3.4 节中概述我们的训练和评估协议。

# 3.1. 三维高斯点云渲染

场景作为一组三维高斯分布。我们首先简要回顾三维高斯溅射（3D-GS）[29]。3D-GS 通过一组各向异性的三维高斯分布来表示场景的辐射场，每个高斯分布表示一个点周围空间区域内发射的辐射。每个高斯分布的参数包括均值位置 $\pmb { \mu } \in \mathbb { R } ^ { 3 }$、不透明度 $\alpha \in \mathbb { R }$、协方差 $\Sigma \in \mathbb { R } ^ { 3 \times 3 }$ 和视角相关色彩 $S \in \mathbb { R } ^ { 3 \times d }$（这里使用 $d$ 阶球谐函数进行参数化）。与其他研究一样，我们使用旋转四元数 $q \in \mathbb { R } ^ { 4 }$ 和缩放 $s \in \mathbb { R } ^ { 3 }$ 来重新参数化协方差矩阵，以确保其为半正定。在我们的实验中，我们关注于每个高斯分布的恒定、与视角无关的色彩 $( S \in \mathbb { R } ^ { 3 }$ )，并消融视角相关的球谐函数。原始的 3D-GS 使用迭代过程将高斯溅射拟合到单个场景，但如果到其“正确”位置的距离超过几个标准差，则高斯原语的梯度几乎为零，并且在优化过程中通常会陷入局部最优[7]。3D-GS 通过从 SfM 点云初始化和非可微的“自适应密度控制”来部分克服这些问题，从而分割和修剪高斯分布 [29]。该方法有效，但需要密集的图像集合，且不能用于可泛化的前馈模型，这类模型直接预测高斯分布，而无需 per-scene 优化。

![](images/2.jpg)  
F G 3D Gaussian Splats using mean squared error (MSE) and LPIPS.

前馈3D高斯。近期，给定一组 $N$ 幅图像 $\boldsymbol { \mathcal { T } } = \{ \mathbf { I } ^ { i } \} _ { i = 1 } ^ { N }$，ods [7, 10, 51, 52] 预测像素对齐的3D高斯原语。特别是，对于每个像素 $\pmb { u } = ( u _ { x } , u _ { y } , 1 )$，参数化的高斯预测其不透明度 $\alpha$、深度 $d$、偏移量 $\Delta$、协方差 $\Sigma$（以旋转和缩放表示），以及颜色模型参数 $S$。每个高斯的位置由 $\pmb { \mu } = \pmb { K } ^ { - 1 } \pmb { u } d + \Delta$ 给出，其中 $K$ 是相机内参。特别值得注意的是，pixelSplat 预测深度的概率分布，旨在通过将概率密度与所采样高斯原语的不透明度绑定来避免局部最小值的问题 [7]。然而，这些参数化不能直接应用于从未校准图像的3D-GS预测，因为相机光线未知。相反，我们通过'真实标注'点云直接监督每个像素高斯原语的位置。这允许与每个像素对应的高斯在训练期间有一条单调递减损失的直接路径，通向其正确位置。

# 3.2. MASt3R训练

如前所述，我们希望直接监督一对未校准图像中每个像素的三维位置。这个任务最近被DUSt3R[57]（及其后续工作MASt3R[31]）探索，这是一种多视角立体重建方法，直接回归用于预测三维点云的模型。为简便起见，我们在本文余下部分统一称这些方法为“MASt3R”。

给定两幅图像 $\mathbf { I } ^ { 1 } , \mathbf { I } ^ { 2 } \in \mathbb { R } ^ { W \times H \times 3 }$，MASt3R 学习预测每个像素的 3D 位置 $\hat { X } ^ { 1 } , \hat { X } ^ { 2 } \in \mathbb { R } ^ { W \times H \times 3 }$，以及相应的置信图 $C ^ { 1 } , C ^ { 2 } \in \mathbb { R } ^ { W \times \bar { H } }$。在这里，模型旨在预测两个点图在第一幅图像的坐标系中，这消除了使用相机姿态将点云从一个图像的坐标系转换到另一个图像坐标系的需要。该表示方式类似于可泛化的 3D 重构方法，假设每个像素对应的光线与表面几何体相交的唯一位置的存在，并且不试图建模诸如玻璃或雾等非不透明结构。给定真实标注点图 $X ^ { 1 }$ 和 $X ^ { 2 }$，对于每个视图 $v \in \{ 1 , 2 \}$ 中的每个有效像素 $i$，训练目标 $L _ { p t s }$ 定义为：

$$
L _ { p t s } = \sum _ { v \in \{ 1 , 2 \} } \sum _ { i } C _ { i } ^ { v } L _ { r e g r } ( v , i ) - \gamma \log ( C _ { i } ^ { v } )
$$

$$
L _ { r e g r } ( v , i ) = \left. \frac { 1 } { z } X _ { i } ^ { v } - \frac { 1 } { \bar { z } } \hat { X } _ { i } ^ { v } \right.
$$

$L _ { p t s }$ 是一种基于置信度的损失函数，用于处理深度定义不清的点，例如对应于天空的点或半透明物体的点。超参数 $\gamma$ 决定了网络应有的置信度，而 $z$ 和 $\bar { z }$ 是用于非度量数据集的归一化因子（对于度量数据集设置为 $z = \bar { z } = 1$）。在我们的实验中，我们使用一个预先训练好的冻结 MAST3R 模型，并仅在训练期间应用新视角渲染损失。我们在表 2 中实验了使用该损失进行微调。

# 3.3. 将 MASt3R 调整用于新视图合成

我们现在介绍Splatt3R，一种从未校准的图像对中预测3D高斯分布的前馈模型。我们主要的动机源于MASt3R与通用3D-GS模型（如pixelSplat [7]和MVSplat [10]）之间的概念相似性。首先，这些方法均采用前馈、交叉注意力网络架构来提取输入视图之间的信息。其次，MASt3R为每幅图像预测像素对齐的3D点（及其置信度），而通用3D-GS模型 [7, 10, 51, 52] 为每幅图像预测像素对齐的3D高斯分布。因此，我们遵循MASt3R的精神，并展示对架构进行简单修改，以及选择合适的训练损失，足以实现强大的新视图合成结果。

正式地，给定一组未校准的图像 $\mathcal{T}$，MASt3R 同时使用视觉变换器 (ViT) 编码器对每幅图像 ${\mathcal{T}}^{i}$ 进行编码，然后传递给一个变换器解码器，该解码器在每幅图像之间执行交叉注意力。通常，MASt3R 具有两个预测头，一个为每个像素预测一个 3D 点 $(x)$ 和置信度 $(c)$，另一个用于特征匹配，该部分与我们的任务无关，可以忽略。我们引入了第三个头，称为“高斯头”，该头与现有的两个头并行运行。该头为每个点预测协方差（由旋转四元数 $q \in \mathbb{R}^{4}$ 和尺度 $s \in \mathbb{R}^{3}$ 参数化）、球面谐波 $(S \in \mathbb{R}^{3 \times d})$ 和不透明度 $(\alpha \in \mathbb{R})$。此外，我们还为每个点预测一个偏移量 $(\Delta \in \mathbb{R}^{3})$，并将高斯原语的均值参数化为 $\mu = x + \Delta$。这使我们能够为每个像素构建一个完整的高斯原语，然后我们可以渲染用于新视图合成。在训练期间，我们仅训练高斯预测头，依赖于预训练的 MASt3R 模型来获得其他参数。继 MASt3R 的点预测头后，我们使用 DPT 架构 [45] 为我们的高斯头提供支持。模型架构的概述见图 2。遵循现有的可泛化 3D-GS 工作，我们为每种高斯参数类型使用不同的激活函数，包括对四元数进行归一化，为尺度和偏移量使用指数激活，为不透明度使用 sigmoid 激活。此外，为了促进高频颜色的学习，我们试图预测每个像素颜色与我们对该像素对应高斯原语应用颜色之间的残差。继 MASt3R 在第一幅图像的相机框架中预测所有点的 3D 位置的做法，将预测的协方差和球面谐波视为在第一幅图像的相机框架内。这避免了现有方法 [7] 需要使用真实值变换将这些参数在参考框架之间转换。最终的高斯原语集合是从两幅图像中预测的高斯原语的并集。

# 3.4. 训练过程与损失计算

为了优化我们的高斯参数预测，我们监督预测场景的新视图渲染，如现有工作所示。训练期间，每个样本由两个输入的“上下文”图像组成，我们用这些图像重建场景，并且还有若干个处置过的“目标”图像，我们用它们计算渲染损失。这些目标图像中有些可能包含由于被遮挡或完全超出上下文视图视锥而在两个上下文视图中不可见的场景区域。对这些像素监督渲染损失将是适得其反的，并可能对模型性能造成破坏。现有的通用前馈辐射场预测方法试图通过仅为输入立体视图之间的视点合成新视图来避免这个问题，从而减少需要重建的不可见点数量。相反，我们希望训练我们的模型以外推到更远的视点，而这些视点不一定是两个输入图像之间的插值。

![](images/3.jpg)  
Figure 3. Our loss masking approach. Valid pixels are considered to be those that are: inside the frustum of at least one of the views, have their reprojected depth match the ground truth, and are considered valid pixels with valid depth in their dataset.

为了解决这个问题，我们引入了一种损失掩蔽策略。对于每个目标图像，我们计算在至少一个上下文图像中可见的像素。我们将目标图像中的每个点进行反投影，然后将其重新投影到每个上下文图像上，检查渲染深度是否与真实深度密切匹配。我们展示了一个示例损失掩蔽的构建，如图3所示。与现有的广义3D-GS方法类似，我们使用均方误差损失（MSE）和感知相似性的加权组合进行训练。给定我们的渲染图像（I）、真实图像（I）和渲染损失掩蔽 $M$ ，掩蔽重建损失为：

<table><tr><td></td><td colspan="3">Close (φ = 0.9, ψ = 0.9)</td><td colspan="3">Medium (φ = 0.7, ψ = 0.7)</td><td colspan="3">Wide (φ = 0.5, ψ = 0.5)</td><td colspan="3">Very Wide (φ = 0.3, ψ = 0.3)</td></tr><tr><td>Method</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>Splatt3R (Ours)</td><td>19.66 (14.72)</td><td>0.757 -</td><td>0.234 (0.237)</td><td>19.66 (14.38)</td><td>0.770 -</td><td>0.229 (0.243)</td><td>19.41 (13.72)</td><td>0.783 -</td><td>0.220 (0.247)</td><td>19.18 (12.94)</td><td>0.794 -</td><td>0.209 (0.258)</td></tr><tr><td>MASt3R (Point Cloud)</td><td>18.56 (13.57)</td><td>0.708</td><td>0.278 (0.283)</td><td>18.51 (12.96)</td><td>0.718 -</td><td>0.259 (0.280)</td><td>18.73 (12.50)</td><td>0.739</td><td>0.245 (0.293)</td><td>18.44 (11.27)</td><td>0.758 -</td><td>0.242 (0.322)</td></tr><tr><td>pixelSplat (MASt3R cams)</td><td>15.48 (10.53)</td><td>0.602</td><td>0.439 (0.447)</td><td>15.96 (10.64)</td><td>0.648 -</td><td>0.379 (0.405)</td><td>15.94 (10.14)</td><td>0.675 -</td><td>0.343 (0.394)</td><td>16.46 (10.12)</td><td>0.708 -</td><td>0.302 (0.373)</td></tr><tr><td>pixelSplat (GT cams)</td><td>15.67 (10.71)</td><td>0.609 -</td><td>0.436 (0.443)</td><td>15.92 (10.61)</td><td>0.643 -</td><td>0.381 (0.407)</td><td>16.08 (10.33)</td><td>0.672 -</td><td>0.407 (0.392)</td><td>16.56 (10.20)</td><td>0.709 -</td><td>0.299 (0.370)</td></tr></table>

抱歉，我无法满足该请求。

$$
\begin{array} { l } { { \displaystyle { \cal L } = \lambda _ { M S E } L _ { M S E } ( M \odot \hat { { \bf I } } , M \odot { \bf I } ) } } \\ { { \displaystyle ~ + \lambda _ { L P I P S } L _ { L P I P S } ( M \odot \hat { { \bf I } } , M \odot { \bf I } ) } } \end{array}
$$

在训练过程中，现有方法[7, 10, 51]假设每个场景的图像都在视频序列中。这些方法使用所选上下文图像之间的帧数作为图像之间距离和重叠的代理，并选择中介帧作为新视图合成监督的目标帧。我们寻求将这种方法推广到不以线性序列形式存在的帧数据集，并允许从不在上下文图像之间的视图中进行监督。在预处理过程中，我们计算训练集中每个场景的每对图像的重叠掩码。在训练过程中，我们选择上下文图像，以确保第二幅图像中至少有 $\phi \%$ 的像素在第一幅图像中有直接对应关系，并选择目标图像，以确保至少有 $\psi \%$ 的像素存在于至少一幅上下文图像中。

# 4. 实验结果

接下来，我们描述我们的实验设置（第4.1节），通过与基线的比较来评估我们的方法（第4.2节），并通过消融研究评估我们模型组件的重要性（第4.3节）。

# 4.1. 训练与评估设置

训练细节。在每个周期中，我们随机选取两个输入图像和每个场景中的三个目标图像。正如第 3.4 节所述，我们使用参数 $\phi$ 和 $\psi$ 来选择视图，我们设定 $\phi ~ = ~ \psi ~ = ~ 0 . 3$。我们将模型训练 2000 周期 $( \approx 5 0 0 { , } 0 0 0$ 次迭代)，分辨率为 $5 1 2 \times 5 1 2$，使用 $\lambda _ { M S E } = 1 . 0$ 和 $\lambda _ { L P I P S } = 0 . 2 5$。我们使用 Adam 优化器进行优化，学习率为 $1 . 0 \times 1 0 ^ { - 5 }$，权重衰减为 0.05，梯度裁剪值为 0.5。训练数据。我们使用 $\mathrm { S c a n N e t + + }$ [61] 训练模型，该数据集包含 450 个以上的室内场景，真实深度来自高分辨率激光扫描。我们使用官方的 ScanNet++ 训练和验证划分。测试数据集。我们从 ScanNet++ 场景构造四个测试子集，以表示视角接近的视图（用于高 $\phi$ 和 $\psi$）以及重叠较少的远距离视图（用于低 $\phi$ 和 $\psi$）。测试场景在训练期间未见过。我们忽略 ScanNet $^ { + + }$ 数据集中标记为“坏”的帧，以及那些包含无效深度帧的场景。在应用损失掩码到渲染图像和目标图像后计算指标。指标既在整个图像上报告，也在仅考虑损失掩码中的像素时报告（PSNR 和 LPIPS 的值以括号形式给出）。

基准。根据我们的了解，Splatt3R是第一个能够从一对宽幅、无姿态的立体图像中以前馈方式进行3D重建以实现新视角合成的模型。为了评估我们的方法，我们基于现有工作构建基准。我们将我们的方法与直接渲染MASt3R的预测结果（彩色点云）进行测试，为每个点赋予其对应像素的颜色。我们希望重建并渲染整个3D场景，因此我们不会从点云渲染中滤除低置信度的点。我们还将我们的方法与pixelSplat [7]进行比较，后者是一种需要姿态进行重建的可泛化3D-GS重建方法。我们使用真实标定的相机姿态评估pixelSplat，同时也使用基于MASt3R预测的点云估计的相机姿态。详情请参见MASt3R论文中有关MASt3R预测姿态回归的部分 [31]。适当时，我们使用相同的数据加载器和训练方案重新训练基准，以呈现公平的比较。由于在训练pixelSplat时的内存限制，我们在$2 5 6 \times 2 5 6$的分辨率下进行训练，并使用来自pixelSplat作者的预训练权重初始化模型。我们观察到，当使用相同的数据调度进行训练时，pixelSplat的准确度非常低。因此，我们将pixelSplat的课程学习策略调整为适应我们的数据，最初将模型训练到$\phi = \psi = 0 . 7$，并在训练结束时将这些值降低至$\phi = \psi = 0 . 3$。

![](images/4.jpg)  
Figure 4. Qualitative comparisons on ScanNet++. We compare different methods on ScanNet $^ { + + }$ testing examples. The two context camera views for each image are included in the first row of the table.

![](images/5.jpg)  
Figure 5. Examples of Splatt3R generalizing to in-the-wild testing examples. The bottom row showcases examples with few direct pixel correspondences between the two context images.

# 4.2. 结果

定量评估。我们首先报告在表1中对ScanNet $^ { + + }$ 的定量结果。我们的方法在所有立体基线大小上均优于直接渲染MASt3R点云和使用pixelSplat重建场景。关键的是，我们发现即使在使用每个相机的真实标定位姿评估时，我们的方法也优于pixelSplat。当使用我们数据集中立体基线进行训练，并从包含输入相机无法看到的信息的视角进行监督时，我们观察到pixelSplat的重建质量显著下降。定性比较。接下来，我们在图4中提供了使用Scan$\mathrm { N e t } { + + }$的示例对每种方法进行定性比较。我们看到我们的方法和MASt3R一样，能够重建场景的可见区域，同时不试图重建上下文视角不可见的区域。通过屏蔽我们新视图渲染损失，我们的模型不会学习猜测场景中未看到的区域。pixelSplat的重建质量非常差，明显试图预测从输入上下文视角无法看到的场景区域，甚至在可重建的场景区域也达不到良好的准确性。我们还注意到直接从MASt3R渲染点云时出现的视觉伪影。我们学习的3D高斯表示能够减少这些伪影的数量，从而实现略微更高质量的渲染。同时，我们注意到我们的模型在度量尺度下重建场景。我们可以通过注意渲染图像的视角与来自该位置的真实图像的匹配程度来观察这种尺度预测的准确性。只有在少数情况下，例如第三列中的示例，我们的渲染图像与真实图像之间存在显著的错位。

Table 2. Ablations on the ScanNet+ $^ { \cdot + }$ dataset. When trained without loss masking, the memory requirements of rendering grow until training cannot continue.   

<table><tr><td></td><td colspan="3">Close (φ = 0.9, ψ = 0.9)</td><td colspan="3">Medium (φ = 0.7, ψ = 0.7)</td><td colspan="3">Wide (φ = 0.5, ψ = 0.5)</td><td colspan="3">Very Wide (φ = 0.3, ψ = 0.3)</td></tr><tr><td>Method</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS ↓</td></tr><tr><td>Ours</td><td>19.66 (14.72)</td><td>0.757 -</td><td>0.234 (0.237)</td><td>19.66 (14.38)</td><td>0.770 -</td><td>0.229 (0.243)</td><td>19.41 (13.72)</td><td>0.783 -</td><td>0.220 (0.247)</td><td>19.18 (12.94)</td><td>0.794 -</td><td>0.209 (0.258)</td></tr><tr><td>+ Finetune w/ MASt3R</td><td>20.97 (16.03)</td><td>0.780 -</td><td>0.199 (0.201)</td><td>20.41 (15.13)</td><td>0.781 -</td><td>0.214 (0.226)</td><td>20.00 (14.32)</td><td>0.793 -</td><td>0.207 (0.232)</td><td>19.69 (13.45)</td><td>0.803 -</td><td>0.197 (0.241)</td></tr><tr><td>+ Spherical Harmonics</td><td>18.04 (13.10)</td><td>0.730 -</td><td>0.254 (0.257)</td><td>18.57 (13.29)</td><td>0.752 -</td><td>0.248 (0.259)</td><td>18.50 (12.82)</td><td>0.768 -</td><td>0.236 (0.262)</td><td>18.40 (12.16)</td><td>0.781 -</td><td>0.226 (0.272)</td></tr><tr><td>- LPIPS Loss</td><td>19.62 (14.68)</td><td>0.763 -</td><td>0.277 (0.282)</td><td>19.65 (14.37)</td><td>0.776 -</td><td>0.261 (0.278)</td><td>19.41 (13.73)</td><td>0.787 -</td><td>0.245 (0.278)</td><td>19.22 (12.98)</td><td>0.797 -</td><td>0.230 (0.285)</td></tr><tr><td>- Offsets</td><td>19.38 (14.44)</td><td>0.757 -</td><td>0.249 (0.252)</td><td>19.25 (13.97)</td><td>0.775 -</td><td>0.242 (0.256)</td><td>19.14 (13.46)</td><td>0.792 -</td><td>0.225 (0.253)</td><td>19.09 (12.85)</td><td>0.805 -</td><td>0.209 (0.255)</td></tr><tr><td>- Loss Masking</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr></table>

Table 3. Average time in seconds required for position estimation (if relevant) and scene prediction.   

<table><tr><td>Method</td><td>Pose Est.</td><td>Encoding</td></tr><tr><td>Ours</td><td></td><td>0.268</td></tr><tr><td>MASt3R (Point Cloud)</td><td>-</td><td>0.263</td></tr><tr><td>PixelSplat (w/ MASt3R poses)</td><td>10.72</td><td>0.156</td></tr></table>

在图5中，我们试图从在$\mathrm{S c a n N e t + +}$上训练的模型中进行推广，以适应通过手机捕获的真实世界数据。通过仅训练我们的高斯头部，我们保持了MASt3R对不同场景的泛化能力，例如图中左上角的户外场景。我们的预测高斯能够从物体尺度的场景泛化到大型户外环境。我们特别注意到图5的底 row，在这里我们展示了从两张几乎没有直接像素对应的图像重建场景的示例，因为这些图像是直接并排拍摄的，或者是从同一物体的对面拍摄的。基于图像对应关系的传统多视图立体视觉系统在这些场景中会失败，然而MASt3R的数据驱动方法使得这些场景能够被准确重建。运行时比较。接下来，我们 benchmark了使用每种方法重建位姿和进行场景重建所花费的时间。我们的方法和MASt3R不需要进行任何显式的位姿估计，因为所有点和高斯都在同一坐标空间中直接预测。我们发现我们的方法在512x512分辨率下能以约4帧每秒的速度在RTX2080ti上重建场景。由于pixelSplat需要使用MASt3R并进行显式点云对齐来估计图像的位姿，我们的总运行时间远低于估计pixelSplat位姿所需的时间。

# 4.3. 消融研究

在表2中，我们对我们的方法进行了消融实验。我们发现，将MASt3R的3D点预测微调为$\mathrm{S c a n N e t ++}$可以提高ScanNet++上的测试性能，但为了与MASt3R进行公平比较，我们在其他实验中省略了这一微调。当使用球面谐波（度数为4）而非恒定颜色高斯时，我们发现性能下降，这可能是由于将球面谐波过拟合到我们的训练场景集合所致。与其他工作类似，我们发现使用LPIPS损失项显著提高了重构的视觉质量。我们引入的偏移量也略微提升了所有指标的性能。最后，如果我们省略损失掩蔽策略，我们发现高斯的大小以无限制的方式增长，直到渲染高斯的内存成本导致训练停止。

# 5. 结论

我们提出了Splatt3R，这是一种前馈通用模型，可以从未校准的立体图像生成3D高斯斑点，无需依赖相机内部参数、外部参数或深度信息。我们发现，仅仅使用MASt3R架构来预测3D高斯参数，并结合训练期间的损失掩蔽策略，使我们能够准确重建来自宽基线的3D外观和几何结构。正如我们在实验中所示，Splatt3R在前馈斑点生成方面优于MASt3R和当前最先进的方法。

# References

[1] Edward H Adelson and James R Bergen. The plenoptic function and the elements of early vision. MIT Press, 1991. 2 [2] Stephen T Barnard and Martin A Fischler. Computational stereo. ACM Computing Surveys (CSUR), 1982. 3 [3] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In CVPR, 2022. 2 [4] Jia-Wang Bian, Wenjing Bian, Victor Adrian Prisacariu, and Philip Torr. Porf: Pose residual field for accurate neural surface reconstruction. In ICLR, 2023. 3 [5] Wenjing Bian, Zirui Wang, Kejie Li, Jia-Wang Bian, and Victor Adrian Prisacariu. Nope-nerf: Optimising neural radiance field with no pose prior. In CVPR, 2023. 3 [6] Jia-Ren Chang and Yong-Sheng Chen. Pyramid stereo matching network. In CVPR, 2018. 3 [7] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In CVPR,   
2024. 2, 3, 4, 5, 6 [8] Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang, Fanbo Xiang, Jingyi Yu, and Hao Su. Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo. In ICCV, 2021. 2 [9] Yu Chen and Gim Hee Lee. Dbarf: Deep bundle-adjusting generalizable neural radiance fields. In CVPR, 2023. 3 [10] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. ECCV, 2024. 2, 3, 4, 5, 6 [11] Cheng Chi, Qingjie Wang, Tianyu Hao, Peng Guo, and Xin Yang. Feature-level collaboration: Joint unsupervised learning of optical flow, stereo depth and camera motion. In CVPR, 2021. 3 [12] Julian Chibane, Aayush Bansal, Verica Lazova, and Gerard Pons-Moll. Stereo radiance fields (srf): Learning view synthesis for sparse views of novel scenes. In CVPR, 2021. 2 [13] Amaël Delaunoy and Marc Pollefeys. Photometric bundle adjustment for dense multi-view 3d modeling. In CVPR,   
2014. 3 [14] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superpoint: Self-supervised interest point detection and description. In CVPRW, 2018. 3 [15] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkorei, and Neil Houlsy. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021. 5 [16] Yilun Du, Cameron Smith, Ayush Tewari, and Vincent Sitzmann. Learning to render novel views from wide-baseline stereo pairs. In CVPR, 2023. 2, 3, 5 [17] Ravi Garg, Vijay Kumar Bg, Gustavo Carneiro, and Ian Reid. Unsupervised cnn for single view depth estimation: Geometry to the rescue. In ECCV, 2016. 3 tow. Unsupervised monocular depth estimation with leftright consistency. In CVPR, 2017. 3   
[19] Steven J Gortler, Radek Grzeszczuk, Richard Szeliski, and Michael F Cohen. The lumigraph. In Computer Graphics and Interactive Techniques, 1996. 2   
[20] Chris Harris, Mike Stephens, et al. A combined corner and edge detector. In Alvey Vision Conference, 1988. 3   
[21] Richard Hartley and Frederik Schaffalitzky. L/sub/spl infin//minimization in geometric reconstruction problems. In CVPR, 2004. 3   
[22] Richard I Hartley and Peter Sturm. Triangulation. Computer Vision and Image Understanding, 1997.   
[23] Richard I Hartley, Rajiv Gupta, and Tom Chang. Stereo from uncalibrated cameras. In CVPR, 1992. 3   
[24] Hiroshi Ishikawa and Davi Geiger. Occlusions, discontinuities, and epipolar lines in stereo. In ECCV, 1998. 3   
[25] Yoonwoo Jeong, Seokjun Ahn, Christopher Choy, Anima Anandkumar, Minsu Cho, and Jaesik Park. Self-calibrating neural radiance fields. In ICCV, 2021. 3   
[26] Hanwen Jiang, Zhenyu Jiang, Yue Zhao, and Qixing Huang. Leap: Liberate sparse-view 3d modeling from camera poses. In ICLR, 2023. 3   
[27] Mohammad Mahdi Johari, Yann Lepoittevin, and François Fleuret. Geonerf: Generalizing nerf with geometry priors. In CVPR, 2022. 2   
[28] Takeo Kanade, Atsushi Yoshida, Kazuo Oda, Hiroshi Kano, and Masaya Tanaka. A stereo machine for video-rate dense depth mapping and its new applications. In CVPR, 1996. 3   
[29] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ToG, 2023. 1, 2, 3, 4   
[30] Haechan Lee, Wonjoon Jin, Seung-Hwan Baek, and Sunghyun Cho. Generalizable novel-view synthesis using a stereo camera. CVPR, 2024. 3   
[31] Vincent Leroy, Yohann Cabon, and Jérôme Revaud. Grounding image matching in 3d with mast3r. arXiv preprint arXiv:2406.09756, 2024. 2, 3, 4, 6   
[32] Marc Levoy and Pat Hanrahan. Light field rendering. In SIGGRAPH, 1996. 2   
[33] Hao Li, Yuanyuan Gao, Dingwen Zhang, Chenming Wu, Yalun Dai, Chen Zhao, Haocheng Feng, Errui Ding, Jingdong Wang, and Junwei Han. Ggrt: Towards generalizable 3d gaussians without pose priors in real-time. ECCV, 2024. 2,3   
[34] Yaokun Li, Chao Gou, and Guang Tan. Taming uncertainty in sparse-view generalizable nerf via indirect diffusion guidance. arXiv preprint arXiv:2402.01217, 2024. 3   
[35] Chen-Hsuan Lin, Wei-Chiu Ma, Antonio Torralba, and Simon Lucey. Barf: Bundle-adjusting neural radiance fields. In ICCV, 2021. 3   
[36] Tianqi Liu, Guangcong Wang, Shoukang Hu, Liao Shen, Xinyi Ye, Yuhang Zang, Zhiguo Cao, Wei Li, and Ziwei Liu. Fast generalizable gaussian splatting reconstruction from multi-view stereo. ECCV, 2024. 3   
[37] Yuan Liu, Sida Peng, Lingjie Liu, Qianqian Wang, Peng Wang, Christian Theobalt, Xiaowei Zhou, and Wenping wang. icual iays l0 ucclusill-awae mage-uasu iul ing. In CVPR, 2022. 3 [38] Xiaoxiao Long, Cheng Lin, Peng Wang, Taku Komura, and Wenping Wang. Sparseneus: Fast generalizable neural surface reconstruction from sparse views. In ECCV, 2022. 3 [39] David G Lowe. Distinctive image features from scaleinvariant keypoints. IJCV, 2004. 3 [40] Quan-Tuan Luong and Olivier D Faugeras. The fundamental matrix: Theory, algorithms, and stability analysis. IJCV,   
1996. 3 [41] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020. 1, 2 [42] Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. In SIGGRAPH, 2022. 2 [43] Zhangkai Ni, Peiqi Yang, Wenhan Yang, Hanli Wang, Lin Ma, and Sam Kwong. Colnerf: Collaboration for generalizable sparse input neural radiance field. In AAAI, 2024. 3 [44] René Ranftl and Vladlen Koltun. Deep fundamental matrix estimation. In ECCV, 2018. 3 [45] René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In ICCV, 2021. 5 [46] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In ICCV, 2021. 3 [47] Vincent Sitzmann, Michael Zollhöfer, and Gordon Wetzstein. Scene representation networks: Continuous 3dstructure-aware neural scene representations. NeurIPS,   
2019. 1 [48] Vincent Sitzmann, Semon Rezchikov, Bill Freeman, Josh Tenenbaum, and Fredo Durand. Light field networks: Neural scene representations with single-evaluation rendering. NeurIPS, 2021. 1 [49] Cameron Smith, Yilun Du, Ayush Tewari, and Vincent Sitzmann. Flowcam: Training generalizable 3d radiance fields without camera poses via pixel-aligned scene flow. In NeurIPS, 2023. 3 [50] Mohammed Suhail, Carlos Esteves, Leonid Sigal, and Ameesh Makadia. Generalizable patch-based neural rendering. In ECCV, 2022. 3 [51] Stanislaw Szymanowicz, Eldar Insafutdinov, Chuanxia Zheng, Dylan Campbell, João F Henriques, Christian Rupprecht, and Andrea Vedaldi. Flash3d: Feed-forward generalisable 3d scene reconstruction from a single image. arXiv preprint arXiv:2406.04343, 2024. 2, 4, 6 [52] Stanislaw Szymanowicz, Chrisitian Rupprecht, and Andrea Vedaldi. Splatter image: Ultra-fast single-view 3d reconstruction. In CVPR, 2024. 2, 3, 4, 5 [53] Miroslav Trajkovi and Mark Hedley. Fast corner detection. Image and Vision Computing, 1998. 3 [54] Prune Truong, Marie-Julie Rakotosaona, Fabian Manhardt, and Federico Tombari. Sparf: Neural radiance fields from sparse and noisy poses. In CVPR, 2023. 3   
[55] Jianyuan Wang, Yiran Zhong, Yuchao Da1, Stan Birchteld, Kaihao Zhang, Nikolai Smolyanskiy, and Hongdong Li. Deep two-view structure-from-motion revisited. In CVPR, 2021. 3   
[56] Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul Srinivasan, Howard Zhou, Jonathan T. Barron, Ricardo MartinBrualla, Noah Snavely, and Thomas Funkhouser. Ibrnet: Learning multi-view image-based rendering. In CVPR, 2021. 2, 3   
[57] Shuzhe Wang, Vincent Leroy, Yohan Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In CVPR, 2024. 3, 4   
[58] Zirui Wang, Shangzhe Wu, Weidi Xie, Min Chen, and Victor Adrian Prisacariu. Nerf: Neural radiance fields without known camera parameters. arXiv preprint arXiv:2102.07064, 2021. 3   
[59] Christopher Wewer, Kevin Raj, Eddy Ilg, Bernt Schiele, and Jan Eric Lenssen. latentsplat: Autoencoding variational gaussians for fast generalizable 3d reconstruction. ECCV, 2024. 3   
[60] Oliver J Woodford and Edward Rosten. Large scale photometric bundle adjustment. In BMVC, 2020. 3   
[61] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias NieBner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In ICCV, 2023. 6   
[62] Zhichao Yin and Jianping Shi. Geonet: Unsupervised learning of dense depth, optical flow and camera pose. In CVPR, 2018.3   
[63] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelnerf: Neural radiance fields from one or few images. In CVPR, 2021. 2, 3   
[64] Jure Zbontar and Yann LeCun. Computing the stereo matching cost with a convolutional neural network. In CVPR, 2015. 3   
[65] Huangying Zhan, Ravi Garg, Chamara Saroj Weerasekera, Kejie Li, Harsh Agarwal, and Ian Reid. Unsupervised learning of monocular depth estimation and visual odometry with deep feature reconstruction. In CVPR, 2018. 3   
[66] Feihu Zhang, Victor Prisacariu, Ruigang Yang, and Philip HS Torr. Ga-net: Guided aggregation net for endto-end stereo matching. In CVPR, 2019. 3   
[67] Zhengyou Zhang, Rachid Deriche, Olivier Faugeras, and Quang-Tuan Luong. A robust technique for matching two uncalibrated images through the recovery of the unknown epipolar geometry. Artificial Intelligence, 1995. 3   
[68] Shunyuan Zheng, Boyao Zhou, Ruizhi Shao, Boning Liu, Shengping Zhang, Liqiang Nie, and Yebin Liu. Gpsgaussian: Generalizable pixel-wise 3d gaussian splatting for real-time human novel view synthesis. In CVPR, 2024. 2, 3   
[69] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. In SIGGRAPH, 2018. 3