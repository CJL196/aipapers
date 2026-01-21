# MotionCtrl：一个统一且灵活的视频生成运动控制器

周霞 王* wzhoux@connect.hku.hk 南洋理工大学 新加坡 杨子阳* yuanzy22@mails.tsinghua.edu.cn 清华大学 中国 王新涛† xintao.alpha@gmail.com 腾讯PCG ARC实验室 中国 李耀伟* ywl@stu.pku.edu.cn 北京大学 中国 陈天水 tianshuichen@gmail.com 广东工业大学 中国 夏梦寒 menghanxyz@gmail.com 腾讯人工智能实验室 中国 罗平 pluo@cs.hku.hk 香港大学 中国 山颖 yingsshan@tencent.com 腾讯PCG ARC实验室 中国

![](images/1.jpg)  
Fo oW higyderhecrpropageuli  wrate .

# 摘要

视频中的运动主要由相机运动（由于相机移动引起）和物体运动（由于物体移动引起）组成。准确控制相机和物体的运动对于视频生成至关重要。然而，现有的研究要么主要关注一种运动，要么未能明确区分这两者，限制了其控制能力和多样性。因此，本文提出了MotionCtrl，一个统一且灵活的视频生成运动控制器，旨在有效且独立地控制相机和物体运动。MotionCtrl的架构和训练策略经过精心设计，考虑了相机运动、物体运动及不完美训练数据的固有特性。与之前的方法相比，MotionCtrl提供了三个主要优势：1）有效且独立地控制相机运动和物体运动，实现更细粒度的运动控制，并促进两种运动的灵活多样组合。2）其运动条件由相机姿态和轨迹确定，这些条件与外观无关，且对生成视频中物体的外观或形状的影响最小。3）这是一个相对具备泛化能力的模型，一旦训练完成，可以适应各种相机姿态和轨迹。进行了大量的定性和定量实验，以证明MotionCtrl优于现有方法。项目页面：https://wzhouxiff.github.io/projects/MotionCtrl/.

# CCS 概念

计算方法学 计算机视觉

# 关键词

AIGC，视频生成，运动控制

# ACM参考格式：

Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, 和 Ying Shan. 2024. MotionCtrl: 一种统一且灵活的视频生成运动控制器. 载于计算机图形学与交互技术特别兴趣组会议论文集 '24 (SIGGRAPH 会议论文集 '24), 2024年7月27日至8月1日, 美国科罗拉多州丹佛. ACM, 纽约, NY, 美国, 22 页. https://doi.org/10.1145/3641519.3657518

# 1 引言

视频生成，例如文本到视频（T2V）生成 [Blattmann et al. 2023b; Chen et al. 2023b; He et al. 2022; Ho et al. 2022; Singer et al. 2022; Zhou et al. 2022] 旨在生成多样化和高质量的视频，符合给定的文本提示。与专注于生成单一图像的图像生成 [Ding et al. 2021; Ramesh et al. 2022, 2021; Rombach et al. 2022; Saharia et al. 2022; Zhou et al. 2021] 不同，视频生成需要在生成的图像序列中创建一致且流畅的运动。因此，运动控制在视频生成中起着至关重要的作用，但在最近的研究中却受到的关注有限。

在视频中，主要存在两种运动类型：由摄像机运动引起的全局运动和由物体运动引起的局部运动（例如图 1 (c) 中提到的缩小镜头的摄像机姿态和摇摆的玫瑰）。需要注意的是，在本文中，这两种运动将分别被称为摄像机运动和物体运动。然而，以往与视频生成中的运动控制相关的大多数工作要么主要集中于其中一种运动，要么缺乏对这两种运动类型的清晰区分。例如，AnimateDiff [Guo 等，2023]、Gen-2 [Esser 等，2023] 和 PikaLab [pik [n.d.]] 主要使用独立的 LoRA [Hu 等，2021] 模型或额外的摄像机参数（例如 PikaLab [pik [n.d.]] 中的 "-camera zoom in"）来执行或触发摄像机运动控制。VideoComposer [Wang 等，2023] 和 DragNUWA [Yin 等，2023a] 使用相同的条件实现摄像机运动和物体运动：VideoComposer [Wang 等，2023] 中的运动向量和 DragNUWA [Yin 等，2023a] 中的轨迹。对这两种运动缺乏清晰区分，阻碍了这些方法在视频生成中实现细粒度和多样化的运动控制。本文介绍了 MotionCtrl，一个统一且灵活的视频生成运动控制器，旨在通过统一模型独立控制摄像机运动和物体运动。这种方法使视频生成中的细粒度运动控制成为可能，并促进了两种运动类型的灵活且多样化的组合。然而，构建这样一个统一的运动控制器面临显著挑战，主要由于以下两个因素。首先，摄像机运动和物体运动在运动范围和模式上差异显著。摄像机运动指整个场景在时间维度上的全局变换，通常通过一系列随时间变化的摄像机姿态来表示。相反，物体运动涉及特定物体在场景中的时间移动，通常表示为与物体相关的一群像素的轨迹。其次，现有的数据集中没有包含伴随完整注释集的视频片段，包括字幕、摄像机姿态和物体运动轨迹。创建这样一个全面的数据集需要大量的努力和资源。

为了应对上述挑战，MotionCtrl 部署了一种精心设计的架构、训练策略和精选数据集。MotionCtrl 由两个模块组成：相机动作控制模块（CMCM）和物体动作控制模块（OMCM），分别针对相机运动和物体运动特征进行定制。CMCM 和 OMCM 都作为适配器模块集成到现有的视频生成模型中。具体来说，CMCM 通过其时序变换器将一系列相机姿态时序集成到视频生成模型中，使生成视频的全局运动与提供的相机姿态对齐。另一方面，OMCM 在视频生成模型的卷积层中空间性地整合了有关物体运动的信息，指示每个生成帧中物体的空间定位。需要注意的是，在本研究中，我们使用 VideoCrafter1 [Chen et al. 2023b]，它是 LVDM [He et al. 2022] 的增强版本，本文中将其称为 LVDM。

利用依赖于大规模预训练视频扩散模型且配备类适配器CMCM和OMCM的精心设计架构，我们可以单独训练这些模块，从而减少对包含字幕、相机姿态和物体运动轨迹注释的视频的全面数据集的需求。因此，我们通过两个数据集实现了MotionCtrl：一个数据集包含字幕和相机姿态的注释，另一个包含字幕和物体运动轨迹的注释。具体来说，我们引入了augmentedRealestate10k数据集，该数据集最初注释了相机运动信息。我们进一步通过使用Blip2 [Li et al. 2023] 生成字幕来增强该数据集，使其适合用于视频生成中的相机运动控制。此外，我们还利用从WebVid [Bain et al. 2021] 取得的视频，结合使用ParticleSfM [Zhao et al. 2022] 提出的运动分割算法合成的物体运动轨迹进行了增强。结合其原始注释的字幕，增强的WebVid数据集有助于学习视频生成中的物体运动控制。通过顺序地及分别用这两个注释数据集训练CMCM和OMCM，我们的MotionCtrl框架实现了在统一的视频生成模型中独立或联合控制相机和物体运动的能力。这种方法实现了相对细粒度和灵活的运动控制，使用户能够更好地控制生成的视频。通过这些精细设计，MotionCtrl在三个方面表现出优越性：1）它独立控制相机和物体运动，实现细粒度的调整和多种运动组合，如图1所示。2）它使用相机姿态和轨迹作为运动条件，这不影响视觉效果，保持了视频中物体的自然外观。例如，我们的MotionCtrl生成的一个视频中，相机运动准确反映了参考视频，提供了一个逼真的埃菲尔铁塔，如图4(b)所示。相比之下，VideoComposer [Wang et al. 2023]依赖于密集运动矢量，错误捕捉参考视频中的门的形状，导致不自然的埃菲尔铁塔。3）MotionCtrl能够控制各种相机移动和轨迹，而无需对每个单独的相机或物体运动进行微调。本工作的主要贡献可以总结如下：（1）我们引入了MotionCtrl，一个统一且灵活的视频生成运动控制器，旨在独立或联合控制生成视频中的相机运动和物体运动，实现更细粒度和多样化的运动控制。（2）我们根据相机运动、物体运动及不完美训练数据的固有属性，精心调整MotionCtrl的架构和训练策略，有效实现视频生成中的细粒度运动控制。（3）我们进行了广泛的实验，以定性和定量的方式证明MotionCtrl相较于以往相关方法的优越性。

# 2 相关工作

早期的视频生成研究主要依赖于生成对抗网络（GANs）或变分自编码器（VAEs）[Saito et al. 2017; Skorokhodov et al. 2022; Tulyakov et al. 2018; Vondrick et al. 2016; Wang et al. 2019]。然而，近年来，随着扩散模型在图像生成中展示的卓越能力[Ho et al. 2020; Rombach et al. 2022; Saharia et al. 2022]，视频生成研究逐渐转向使用扩散模型。通过进一步结合文本[Blattmann et al. 2023b; Chen et al. 2023b; Guo et al. 2023; He et al. 2022; Ho et al. 2022; Singer et al. 2022; Wang et al. 2023; Zhou et al. 2022]或图像[Blattmann et al. 2023a; Yin et al. 2023b]引导，扩散模型可以生成具有特定内容的高保真视频。特别是在潜在空间中应用扩散模型[Blattmann et al. 2023b; He et al. 2022; Rombach et al. 2022]，显著提高了视频生成的计算效率，从而导致了以扩散模型为中心的下游研究激增。例如，MotionCtrl旨在利用扩散模型控制生成视频中的运动。在生成视频的运动控制领域，许多现有方法通过参考特定或一系列模板视频来学习运动[Guo et al. 2023; Wu et al. 2023b,a; Zhao et al. 2023]。尽管在特定运动控制方面效果显著，但这些方法通常需要为不同模板训练新模型，这可能导致一定的局限性。一些努力旨在实现更通用的运动控制[Chen et al. 2023a; Wang et al. 2023; Yin et al. 2023a]。例如，VideoComposer[Wang et al. 2023]引入了通过额外提供的运动向量进行运动控制，而DragNUWA[Yin et al. 2023a]则提出基于初始图像、提供的轨迹和文本提示进行视频生成。然而，这些方法中的运动控制相对宽泛，未能细致地区分视频中的相机运动和物体运动。与这些工作不同，我们提出了MotionCtrl，这是一种统一且灵活的运动控制器，可以使用相机姿态和物体轨迹，或将这两种引导组合在一起，来控制生成视频的运动。它使视频生成的控制更加细致和灵活。

# 3 方法论

# 3.1 初步分析

隐式视频扩散模型（LVDM）[He 等，2022]旨在生成高质量和多样化的视频，受文本提示的指导。它在潜在空间中采用了一种去噪扩散模型（U-Net [Ronneberger 等，2015]），以提高空间和时间效率。因此，它构建了一个轻量级的3D自编码器，由编码器$\varepsilon$和解码器$\mathcal{D}$组成，用于将原始视频编码到潜在空间，并将去噪后的潜在特征重建为视频。其去噪U-Net（记作$\epsilon_{\theta}$）是由包含卷积层、空间变换器和时间变换器的多个块构成（见图2）。它通过噪声预测损失进行优化：

$$
\mathcal { L } = \mathbb { E } _ { z _ { 0 } , c , \epsilon \sim \mathcal { N } ( 0 , I ) , t } \left[ \| \epsilon - \epsilon _ { \theta } ( z _ { t } , t , c ) \| _ { 2 } ^ { 2 } \right] ,
$$

其中 $c$ 表示文本提示，$z_{0}$ 是使用 $\varepsilon$ 获得的潜在编码，$t(t \in [0, T])$ 表示时间步，而 $z_{t}$ 是通过将高斯噪声 $\epsilon$ 加权叠加到 $z_{0}$ 上所获得的噪声潜在特征，使用以下公式：

$$
z _ { t } = \sqrt { \bar { \alpha _ { t } } } z _ { 0 } + \sqrt { 1 - \bar { \alpha _ { t } } } \epsilon , \ \bar { \alpha _ { t } } = \prod _ { i = 1 } ^ { t } \alpha _ { t } ,
$$

其中 $\alpha _ { t }$ 用于根据时间步 $t$ 调整噪声强度。

# 3.2 运动控制

图2展示了MotionCtrl的框架。为了实现相机运动与物体运动的解耦，以及对这两种运动的独立控制，MotionCtrl包括两个主要组件：相机运动控制模块（CMCM）和物体运动控制模块（OMCM）。CMCM考虑了相机运动的全局特性，并与LVDM中的时间变换器进行交互，而OMCM则在空间上与LVDM中的卷积层合作。此外，我们采用多个训练步骤，以使MotionCtrl适应缺乏包含高质量视频剪辑及相应字幕、相机姿态和物体运动轨迹的训练数据。在接下来的小节中，我们将详细描述CMCM和OMCM，以及它们相应的训练数据集和训练策略。

![](images/2.jpg)  
FoCrworkotCretenhDNeucLVmMotio Ml ) an n Obje Moi Cnol Modul C)s illustrate in , theC interate c p sequences RT with LVDM's temporal transformers by appending ${ \boldsymbol { R } } { \boldsymbol { T } }$ to the input of the second self-attention module and Tuly aa v n uo left, consistent with the camera's rightward motion.

3.2.1 摄像机运动控制模块（CMCM）。CMCM 是一个由多个全连接层构成的轻量级模块。由于摄像机运动是视频帧之间的全局变换，CMCM 通过其时间变换器与 LVDM [He et al. 2022] 协同工作。通常，LVDM 中的时间变换器包括两个自注意力模块，促进视频帧之间的时间信息融合。为了最小化对 LVDM 生成性能的影响，CMCM 仅涉及时间变换器中的第二个自注意力模块。具体而言，CMCM 将一系列摄像机姿态 $R T = \left\{ R T _ { 0 } , R T _ { 1 } , \dots , R T _ { L - 1 } \right\}$ 作为输入。在本文中，摄像机姿态由其 $3 { \times } 3$ 旋转矩阵和 $3 \times 1$ 平移矩阵表示。

因此，$R T \in \mathbb { R } ^ { L \times 1 2 }$，其中 $L$ 表示生成视频的长度。如图 2 (b) 所示，$R T$ 被扩展为 $H \times W \times L \times 1 2$，然后与时间变换器中的第一个自注意力模块的输出 $\bar { ( y _ { t } \in \mathbb { R } ^ { H \times W \times L \times C } ) }$ 在最后一个维度上进行拼接，其中 $H$ 和 $W$ 表示生成视频的潜在空间大小，$C$ 是 $y _ { t }$ 中通道的数量。拼接后的结果通过一个全连接层投影回 $H \times W \times L \times C$ 的大小，然后输入到时间变换器中的第二个自注意力模块。

3.2.2 物体运动控制模块（OMCM）。如图2所示，MotionCtrl通过轨迹（Tra js）控制生成视频的物体运动。通常，轨迹表示为一系列空间位置 $\{ ( x _ { 0 } , y _ { 0 } ) , ( x _ { 1 } , y _ { 1 } ) , \dotsc , ( x _ { L - 1 } , y _ { L - 1 } ) \}$，其中 $( x _ { i } , y _ { i } ) , i \in [ 0 , L - 1 ]$ 表示轨迹在空间位置 $( x , y )$ 的第 $i _ { t h }$ 帧经过。特别地，$x \in \bar { [ 0 , W )}$ 和 $y \in [ 0 , \hat { H })$，其中 $\bar { H }$ 和 $\hat { W }$ 分别是 $z _ { T }$ 的高度和宽度。为了清晰地展示物体的移动速度，我们将Tra js表示为

$$
\{ ( 0 , 0 ) , ( u _ { ( x _ { 1 } , y _ { 1 } ) } , v _ { ( x _ { 1 } , y _ { 1 } ) } ) , \ldots , ( u _ { ( x _ { L - 1 } , y _ { L - 1 } ) } , v _ { ( x _ { L - 1 } , y _ { L - 1 } ) } ) \} ,
$$

$$
u _ { ( x _ { i } , y _ { i } ) } = x _ { i } - x _ { i - 1 } ; v _ { ( x _ { i } , y _ { i } ) } = y _ { i } - y _ { i - 1 } ; 0 < i < L .
$$

标记第一个帧和后续帧中轨迹不经过的其他空间位置为 (0, 0)。最后，Trajs 为 RLx××2。Trajs 被注入到 LVDM 中，与 OMCM 结合，突出显示在图 2 的紫色块中。OMCM 由多个结合下采样操作的卷积层组成。它从 Trajs 中提取多尺度特征，并相应地将这些特征添加到 LVDM 卷积层的输入中。受到 T2I-Adapter [Mou et al. 2024] 的启发，轨迹仅应用于去噪 U-Net 的编码器，以平衡生成视频的质量与对象运动控制的能力。3.2.3 训练策略与数据构建。为了通过文本提示实现相机和对象运动的控制，同时生成视频，训练数据集中的视频剪辑必须包含标题、相机姿态和对象运动轨迹的注释。然而，目前缺乏这样的全面详细的数据集，组建一个需要相当的努力和资源。为了解决这个挑战，我们引入了多步骤训练策略，并用针对各自运动控制需求的不同增强数据集训练我们提出的相机运动控制模块 (CMCM) 和对象运动控制模块 (OMCM)。学习相机运动控制模块 (CMCM)。CMCM 只需要包含带有标题和相机姿态注释的视频剪辑的训练数据集。考虑到 Realestate10K [Zhou et al. 2018] 包含超过 60,000 个视频，且相机姿态的注释相对干净，我们将其作为 CMCM 的训练数据集。然而，在 MotionCtrl 中使用 Realestate10K 存在两个潜在挑战：1) Realestate10K 中场景的多样性有限，主要来自房地产视频，可能会影响生成视频的质量；2) 缺乏 T2V 模型所需的标题。针对第一个挑战，我们采用了一种类似于适配器的控制模块 (CMCM)，仅若干新增的 MLP 层和 LVDM 中时间变换器的第二自注意模块为可训练，并通过冻结大部分参数来保留 LVDM 的生成质量。由于时间变换器主要关注全局运动的学习，Realestate10K 的有限场景多样性很少会影响 LVDM 的生成质量。表 2 中呈现的定量结果实证了这一点，其中 FID [Seitzer 2020] 和 FVD [Unterthiner et al. 2018] 指标表明，我们的 MotionCtrl 生成的视频质量与 LVDM 的结果相当。针对第二个挑战，我们采用 Blip2 [Li et al. 2023]，一种图像标题生成算法，为 Realestate10K 中的每个视频剪辑生成标题。详细信息见补充材料。学习对象运动控制模块 (OMCM)。OMCM 需要一个包含带有标题和对象运动轨迹的视频剪辑的数据集，而目前在社区中缺乏此类数据集。为满足这一要求，我们利用 ParticleSfM [Zhao et al. 2022] 在 WebVid [Bain et al. 2021] 中合成对象运动轨迹。WebVid 是一个大规模的视频数据集，配备有标题，通常用于 T2V 生成任务。尽管 ParticleSfM 主要是一个基于运动的结构系统，但其包含的基于轨迹的运动分割模块可用于过滤掉影响动态场景中相机轨迹生成的动态轨迹。运动分割模块获得的动态轨迹正好满足我们 MotionCtrl 的需求，我们利用此模块为约 243,000 个视频合成移动对象轨迹。

![](images/3.jpg)  

Figure 3: Trajectories for Object Motion Control. ParticleSfM [Zhao et al. 2022] is employed to extract object movement trajectories from video clips, effectively disentangling object motion from camera-induced movement. To circumvent the issues of dense trajectories, which can encode object shapes and are challenging to design at inference, we train the OMCM using sparse trajectories sampled from the dense ones. These sparse trajectories, being too scattered for effective learning, are subsequently refined with a Gaussian filter.

WebVid。图 3 (b) 中展示了一个例子，其中轨迹主要对应于移动的人。合成细节见补充材料。为了避免用户提供如图 3 (b) 所示的稠密轨迹，这可能不太用户友好，MotionCtrl 需要根据用户提供的稀疏（一个或几个）轨迹来控制移动物体。因此，我们的 OMCM 是通过从合成的稠密轨迹中随机选择 $n \in [ 1 , N ]$ 轨迹进行训练的（其中 $N$ 表示每个视频的最大轨迹数，如图 3 (c) 所示）。然而，这些选择的稀疏轨迹往往过于分散，难以有效训练。受到 DragNUWA [Yin et al. 2023a] 的启发，我们通过对稀疏轨迹应用高斯滤波来缓解此问题（图 3 (d)），并且我们最初使用稠密轨迹训练 OMCM，然后再使用稀疏轨迹进行微调。在这个训练阶段，LVDM 和 CMCM 都经过充分训练并被冻结，只有 OMCM 进行训练。这一策略保证了 OMCM 在有限数据集下增强物体运动控制能力，同时对 LVDM 和 CMCM 的影响最小。在完成此训练阶段后，提供相机姿态和物体轨迹可以灵活控制生成视频中的相机和物体运动。

# 4 实验

# 4.1 实验设置

4.1.1 实现细节。MotionCtrl 基于 LVDM 框架 [He et al. 2022]/VideoCraft1 [Chen et al. 2023b] 构建，训练时使用的是分辨率为 $256 \times 256$ 的 16 帧序列。它可以很容易地适配其他结构相似的视频生成模型，例如 AnimateDiff [Guo et al. 2023]，并遵循每个模型特定的设置。此外，轨迹的最大数量 $N$ 固定为 8。CMCM 和 OMCM 都使用 Adam 优化器 [Kingma and Ba 2014] 进行优化，批量大小为 128，学习率为 $1 e^{-4}$，在 8 块 NVIDIA Tesla V100 GPU 上进行训练。CMCM 通常需要大约 50,000 次迭代才能收敛。同时，OMCM 在稠密轨迹上进行初始训练阶段，迭代 20,000 次，然后在稀疏轨迹上微调额外的 20,000 次迭代。

![](images/4.jpg)  
: ualitairinsmMotiCntrol BosMotiriateifu e e y matches the camera poses.

4.1.2 评估数据集。 (1) 摄像机运动控制评估数据集涵盖两种类型的摄像机姿态：基础摄像机姿态（左平移、右平移、上平移、下平移、放大、缩小、逆时针旋转和顺时针旋转）以及相对复杂的摄像机姿态，后者从 Realestate10K 的测试集 [Zhou et al. 2018] 获得，或通过 ParticleSfM [Zhao et al. 2022] 在来自 WebVid [Bain et al. 2021] 和 HD-VILA [Xue et al. 2022] 的视频上合成。 (2) 物体运动控制评估数据集由 283 个样本构成，这些样本基于多样的手工轨迹和提示构建。有关评估数据集构建的更多细节，请参见补充材料。

4.1.3 评估指标。 (1) 生成视频的质量通过弗雷歇距离（Fréchet Inception Distance, FID）[Seitzer 2020]、弗雷歇视频距离（Fréchet Video Distance, FVD）[Unterthiner et al. 2018] 和 CLIP 相似度（CLIP Similarity, CLIPSIM）[Radford et al. 2021] 进行评估，这三者分别衡量视觉质量、时间一致性和与文本的语义相似度。FID 和 FVD 的参考视频为来自 WebVid 的 1000 个视频 [Bain et al. 2021]。 (2) 摄像机和物体运动控制的有效性通过计算预测的摄像机姿态与真实标注的摄像机姿态以及物体轨迹之间的欧几里得距离来量化。预测视频的摄像机姿态和物体轨迹使用 ParticleSfM [Zhao et al. 2022] 提取。我们将这两个指标分别命名为 CamMC 和 ObjMC。 (3) 我们还进行了用户研究以进行主观定量评估，具体细节由于篇幅限制已在附录材料中提供。

# 4.2 与最先进方法的比较

为了验证我们在控制摄像机和物体运动方面的 MotionCtrl 的有效性，我们将其与两种领先方法进行比较：AnimateDiff [Guo 等 2023] 和 VideoComposer [Wang 等 2023]。AnimateDiff 使用 8 个独立的 LoRA [Hu 等 2021] 模型来控制视频中的 8 种基本摄像机运动，如平移和变焦，而 VideoComposer 则通过运动矢量操控视频运动，不区分摄像机和物体的运动。尽管 DragNUWA [Yin 等 2023a] 与我们的研究相关，但其代码并未公开，无法进行直接比较。此外，DragNUWA 仅通过从光流中提取的轨迹学习运动控制，无法精细区分前景物体与背景之间的运动，限制了其精准控制摄像机和物体运动的能力。我们在摄像机运动和物体运动控制方面与这些方法进行了比较，并展示了我们的 MotionCtrl 在视频生成中灵活结合摄像机运动和物体运动控制的能力。更多的比较和视频对比请参见补充材料。

4.2.1 摄像机运动控制。我们通过基本姿势和相对复杂的姿势评估摄像机的运动控制。AnimateDiff [Guo et al. 2023] 限于基本摄像机姿势，而 VideoComposer [Wang et al. 2023] 则通过从提供的视频中提取运动矢量来处理复杂姿势。定性结果如图 4 所示。对于基本姿势，MotionCtrl 和 AnimateDiff 都可以生成具有前向摄像机运动的视频，但 MotionCtrl 可以生成具有不同速度的摄像机运动，而 AnimateDiff 是不可调节的。对于复杂姿势，摄像机首先向左前方移动，然后向前移动，VideoComposer 能够使用提取的运动矢量模仿参考视频的摄像机运动。然而，密集的运动矢量无意中捕捉到了物体形状，参考视频（帧 12）中的门的轮廓，导致了不自然的埃菲尔铁塔外观。MotionCtrl 在旋转和平移矩阵的引导下，生成了更自然的视频，其摄像机运动接近参考。定量结果在表 1 中显示了 MotionCtrl 在基本和相对复杂姿势上的优越性，CamMC 得分反映了这一点。此外，

![](images/5.jpg)  

一个女孩在滑雪。 定性比较对象运动控制，视频合成器和运动生成功能都可以生成关键点。

Table 1: Quantitative Comparisons with AnimateDiff [Guo et al. 2023] and VideoComposer [Wang et al. 2023]. Our MotionCtrl outperforms competing approaches in both camera and object motion control while also excelling at preserving text similarity and the quality of the video generation.   

<table><tr><td>Method</td><td>AnimateDiff</td><td>VideoComposer</td><td>MotionCtrl</td></tr><tr><td>CamMC ↓ (Basic Poses)</td><td>0.0548</td><td>-</td><td>0.0289</td></tr><tr><td>CamMC ↓ (Complex Poses)</td><td>-</td><td>0.0950</td><td>0.0735</td></tr><tr><td>ObjMC ↓</td><td>-</td><td>36.8351</td><td>28.877</td></tr><tr><td>CLIPSIM↑</td><td>0.2144</td><td>0.2214</td><td>0.2319</td></tr><tr><td>FID ↓</td><td>157.73</td><td>130.97</td><td>124.09</td></tr><tr><td>FVD ↓</td><td>1815.88</td><td>1004.99</td><td>852.15</td></tr></table>

MotionCtrl 在文本相似性和质量指标上表现更佳，具体通过 CLIPSIM、FID 和 FVD 测量。4.2.2 物体运动控制。我们将 MotionCtrl 与 VideoComposer 进行物体运动控制的比较，其中 VideoComposer 利用从轨迹中提取的运动矢量。定性结果如图 5 所示。红色曲线表示给定的轨迹，而绿色点表示对应帧中期望的物体位置。视觉比较显示，MotionCtrl 可以生成与给定轨迹更接近的物体运动，而 VideoComposer 的结果在某些帧上有所偏离，这突显了 MotionCtrl 在物体运动控制能力上的优越性。表 1 中的 ObjMC 量化结果也证明了 MotionCtrl 在物体运动控制方面优于 VideoComposer。4.2.3 相机运动与物体运动的结合。MotionCtrl 不仅可以在单个视频中独立控制相机与物体运动，还能对两者进行综合控制。如图 1 (b) 和 (c) 所示，当仅应用给定轨迹时，MotionCtrl 主要生成沿该路径摇曳的玫瑰。进一步引入缩小的相机视角后，玫瑰和背景的动画效果则会依据指定的轨迹和相机运动进行调整。更多 MotionCtrl 的结果可在图 8、附加材料和演示视频中找到。

# 4.3 消融研究

4.3.1 摄像机运动控制模块的集成位置（CMCM）。我们通过将摄像机姿态与时间嵌入、空间交叉注意力或空间自注意力模块结合在LVDM中测试摄像机运动控制的实现。尽管这种方法在其他类型的控制（如素描和深度）中取得了成功[Mou et al. 2024; Zhang et al. 2023]，但它们未能赋予LVDM摄像机控制能力，这在表2的CamMC评分和图6的可视化结果中得到了证明。它们的CamMC评分接近于原始的LVDM。这是因为这些组件主要集中于空间内容生成，而对摄像机姿态中编码的摄像机运动不敏感。相反，将CMCM与LVDM的时间变换器结合显著改善了摄像机运动控制，表2中CamMC评分降低至0.0289。摄像机运动主要导致全球视图随时间的变换，将摄像机姿态融合到LVDM的时间块中与这一特性相吻合，能够在视频生成过程中实现有效的摄像机运动控制。

Table 2: Ablation of Camera Motion Control. Our Camera Motion Control Module (CMCM), incorporated with the temporal transformers of LVDM [He et al. 2022], effectively controls camera motion and maintains LVDM's video quality.   

<table><tr><td>Method</td><td>CamMC ↓</td><td>CLIPSIM ↑</td><td>FID ↓</td><td>FVD ↓</td></tr><tr><td>LVDM [He et al. 2022]</td><td>0.9010</td><td>0.2359</td><td>130.62</td><td>1007.63</td></tr><tr><td>Time Embedding</td><td>0.0887</td><td>0.2361</td><td>132.74</td><td>1461.36</td></tr><tr><td>Spatial Cross-Attention</td><td>0.0857</td><td>0.2357</td><td>153.86</td><td>1306.78</td></tr><tr><td>Spatial Self-Attention</td><td>0.0902</td><td>0.2384</td><td>146.37</td><td>1303.58</td></tr><tr><td>Temporal Transformer</td><td>0.0289</td><td>0.2355</td><td>132.36</td><td>1005.24</td></tr></table>

4.3.2 密集轨迹与稀疏轨迹。OMCM最初通过ParticleSfM提取的密集物体运动轨迹进行训练[Zhao et al. 2022]，然后再利用稀疏轨迹进行微调。我们通过将OMCM仅在密集轨迹或稀疏轨迹上训练的效果进行比较，评估这种方法的有效性。表3和图7表明，仅用密集轨迹训练的结果较差，这归因于训练阶段与推理阶段之间的差异（推理过程中提供的是稀疏轨迹）。尽管仅用稀疏轨迹的训练相较于只用密集轨迹的方法有所改善，但仍然不及混合方法，因为稀疏轨迹单独提供的信息有限。相比之下，密集轨迹提供了更丰富的信息，加速了学习，而随后使用稀疏轨迹进行微调则使OMCM能够适应推理过程中遇到的稀疏性。

Table 3: Ablation of Object Motion Control. The Object Motion Control Module (OMCM), when initially trained on dense object movement trajectories and subsequently finetuned with sparse trajectories, outperforms versions trained exclusively on either dense or sparse trajectories.   

<table><tr><td>Method</td><td>ObjMC ↓</td><td>CLIPSIM ↑</td><td>FID ↓</td><td>FVD ↓</td></tr><tr><td>Dense</td><td>54.4114</td><td>0.2352</td><td>175.8622</td><td>2227.87</td></tr><tr><td>Sparse</td><td>34.6937</td><td>0.2365</td><td>158.5553</td><td>2385.39</td></tr><tr><td>Dense + Sparse</td><td>25.1198</td><td>0.2342</td><td>149.2754</td><td>2001.57</td></tr></table>

4.3.3 训练策略。鉴于可用训练数据集的限制，我们为MotionCtrl提出了一种多步骤训练策略，首先使用Realestate10K [Zhou et al. 2018]进行CMCM训练，随后使用合成的对象运动轨迹进行OMCM训练。为了彻底评估我们的方法，我们尝试颠倒顺序，先训练OMCM，再训练CMCM。这个顺序不会影响相机运动控制，因为OMCM组件并不参与CMCM的训练。然而，这会导致对象运动控制性能下降，因为CMCM的后续训练会调整LVDM时间变换器的部分，干扰在OMCM初始训练中实现的对象运动控制适应。因此，我们的多步骤策略尽管是由于数据集限制而作出的妥协，但故意设计为先训练CMCM再训练OMCM，以确保在相机和对象运动控制两方面都能提升性能。

# 4.4 在 AnimateDiff 上部署 MotionCtrl

我们还在 AnimateDiff [Guo et al. 2023] 上部署了我们的 MotionCtrl。因此，我们可以控制与多个 LoRA [Hu et al. 2021] 模型协作的调整后的 AnimateDiff 生成的视频的运动。复杂摄像机运动控制和物体运动控制的可视化结果见图 9 和图 10。更多结果请参见补充材料。

# 5 限制因素

作为对统一视频生成模型中相机和物体运动控制的初步探索，MotionCtrl 已展示出令人鼓舞和启发性的结果。然而，在同一视频中控制相机和物体运动，尤其是复杂的相机轨迹和复杂的物体轨迹，要求对这些轨迹进行仔细设计，以实现自然和谐的效果，成功率相对较低。进一步的研究仍然需要提高生成视频中同时控制相机和物体运动的准确性。

# 6 结论

本文提出了MotionCtrl，这是一种统一且灵活的控制器，可以独立或结合地控制通过视频生成模型获取的视频中的相机和物体运动。为实现这一目标，MotionCtrl精心设计了相机运动控制模块和物体运动控制模块，以适应相机运动和物体运动的特定属性，并采用多步骤训练策略，通过精心增强的数据集对这两个模块进行训练。综合实验，包括定性和定量评估，展示了我们提出的MotionCtrl在相机和物体运动控制方面的优越性。

# REFERENCES

[n. d.]. https://www.pika.art/.   
Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. 2021. Frozen in time: A joint video and image encoder for end-to-end retrieval. In ICCV.   
A B T D  a D , Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. 2023a. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127 (2023).   
Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. 2023b. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR.   
Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. 2023b. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512 (2023).   
Tsai-Shien Chen, Chieh Hubert Lin, Hung-Yu Tseng, Tsung-Yi Lin, and Ming-Hsuan Yang. 2023a. Motion-conditioned diffusion model for controllable video synthesis. arXiv preprint arXiv:2304.14404 (2023).   
Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. 2021. Cogview: Mastering text-toe eneration i transformers. Neur (2021).   
ss Germanidis. 2023. Structure and content-guided video synthesis with diffusion models. In ICCV.   
Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. 2023. AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning. arXiv preprint arXiv:2307.04725 (2023).   
Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. 2022. Latent vdeo diffusion models for high-fidelity long video generation.arXiv prent arXiv:2211.13221 (2022).   
Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. 2022. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303 (2022).   
Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. NeurIPS (2020).   
models. arXiv preprint arXiv:2106.09685 (2021).   
Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al.0Vench: Coprehensive benchmark site for video generatie models. (2024).   
Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014).   
J lSva Sev HoB: B lgemage preraining withfrozemagencoder andrge languagemols. In ICML.   
Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and for text-to-image diffusion models. In AAAI.   
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. L an il mo om ual ng upe I IL.   
Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. 2022. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125 (2022).   
Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, e n  Suteer. 0. Z-hot texo- enton  .   
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent diffusion models. In CVPR.   
Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 2015. U-net: Convolutional networks for biomedical image segmentation. In MICCAI.   
Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, understanding. NeuIPS (2022).   
M   Tl - sarial nets with singular value clipping. In ICCV.   
Maximilian Seitzer. 2020. pytorch-fid: FD Score for PyTorch. https://github.com/ mseitzer/pytorch-fid.   
Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. 2022. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792 (2022).   
Ivan Skorokhodov, Sergey Tulyakov, and Mohamed Elhoseiny. 2022. Stylegan-v: A continuous video generator with the price, image quality and perks of stylegan2. In CVPR.   
Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, and Jan Kautz. 2018. Mocogan: Decomposing motion and content for video generation. In CVPR.   
Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. 2018. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717 (2018).   
Carl Vondrick, Hamed Pirsiavash, and Antonio Torralba. 2016. Generating videos with scene dynamics. NeuIPS (2016).   
Ting-Chun Wang, Ming-Yu Liu, Andrew Tao, Guilin Liu, Jan Kautz, and Bryan Catanzaro. 2019. Few-shot video-to-video synthesis. arXiv preprint arXiv:1910.12713 (2019).   
Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. 2023. VideoComposer: Compositional Video Synthesis with Motion Controllability. arXiv preprint arXiv:2306.02018 (2023).   
Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. 2023b. Tune-a-video: One-hot tuning of image diffusion models for text-to-video generation. In ICCV.   
Ruiqi Wu, Liangyu Chen, Tong Yang, Chunle Guo, Chongyi Li, and Xiangyu Zhang. 2023a. LAMP: Learn A Motion Pattern for Few-Shot-Based Video Generation. arXiv preprint arXiv:2310.10769 (2023).   
Hongwei Xue, Tiankai Hang, Yanhong Zeng, Yuchong Sun, Bei Liu, Huan Yang, Jianlong Fu, and Baining Guo. 2022. Advancing high-resolution video-language representation with large-scale video transcriptions. In CVPR.   
Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, and Nan Duan 023a. Dragnuwa: Finegraine contro in video generatin by interatng text, image, and trajectory. arXiv preprint arXiv:2308.08089 (2023).   
Shengming Yin, Chenfei Wu, Huan Yang, Jianfeng Wang, Xiaodong Wang, Minheng Ni, Zhengyuan Yang, Linjie Li, Shuguang Liu, Fan Yang, et al. 2023b. Nuwa-xl: Diffusion over diffusion for extremely long video generation. arXiv preprint arXiv:2303.12346 (2023).   
Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding conditional control to text-to-image diffusion models. In ICCV.   
Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jiawei Liu, Weijia Wu, Jussi Keppo, and Mike Zheng Shou. 2023. MotionDirector: Motion Customization of Text-to-Video Diffusion Models. arXiv preprint arXiv:2310.08465 (2023).   
Wang Zhao, Shaohui Liu, Hengkai Guo, Wenping Wang, and Yong-Jin Liu. 2022. Particlesfm: Exploiting dense point trajectories for localizing moving cameras in the wild. In ECCV.   
Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. 2022. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018 (2022).   
Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. 2018. aat Learig nhe us multipan ma preprint arXiv:1805.09817 (2018).   
Yufan Zhou, Ruiyi Zhang, Changyou Chen, Chunyuan Li, Chris Tensmeyer, Tong Yu, fo ex oXieXiv:21179

![](images/6.jpg)  
Prompt: A fish is swimming in the aquarium tank.

![](images/7.jpg)  
T lrao () iVDMHe  ]ICC ih  LVD y improves camera motion control compared to other setups.   
Prompt: A man is surfing.   
f trained solely on sparse trajectories.

![](images/8.jpg)  
Prompt: A human robot standing on Mars.

![](images/9.jpg)  
Prompt: A basketball in the air.

those controlled with camera poses and object trajectories simultaneously.

![](images/10.jpg)

![](images/11.jpg)  
Figure 9: Results of complex camera motion control deployed on AnimateDiff [Guo et al. 2023]   
Prompt: :A teddy bear skateboardin.   
Figure 10: Results of object motion control deployed on AnimateDiff [Guo et al. 2023].

The supplementary materials provide additional results achieved with our proposed MotionCtrl, along with in-depth analyses. For a more visual understanding, we strongly recommend readers visit our project page for the video results. The structure of the supplementary materials is as follows:

Details of training data construction. (Section A) Details of evaluation datasets. (Section B) • More quantitative and qualitative results. (Section C) • More Results of MotionCtrl when extended to AnimateDiff [Guo et al. 2023] framework. (Section D) More discussions about previous related works. (Section E)

# A DETAILS OF TRAINING DATACONSTRUCTION

Augmented-RealEstate10K. The camera motion control module (CMCM) in MotionCtrl is trained with data augmented from RealEstate10K [Zhou et al. 2018]. RealEstate10K originally contains videos with annotations of camera poses. To adapt it to our MotionCtrl, we further synthesize captions for each video with Blip2 [Li et al. 2023], an image captioning algorithm. Specifically, we extract frames at specific intervals—the first, quarter, half, threequarters, and final frames of a video. We then use Blip2 to predict their captions. These captions are concatenated to form a comprehensive description for each video clip. With these captions in place, we train the CMCM on RealEstate10K, enabling effective camera motion control in video generation models such as LVDM [He et al. 2022].

Augmented-WebVid. The object motion control module (OMCM in MotionCtrl is trained with data augmented from WebVid [Bain et al. 2021]. WebVid is a large-scale video dataset equipped with captions and commonly used in the T2V generation task. To adapt it to our MotionCtrl, we further synthesize the object movement trajectories for the videos in WebVid with ParticleSfM [Zhao et al. 2022]. Although ParticleSfM is a structure-from-motion system primarily, it incorporates a trajectory-based motion segmentation module utilized for filtering out dynamic trajectories that affect the production of camera trajectories in a dynamic scene. The dynamic trajectories attained by the motion segmentation module exactly fulfill the requirements of our MotionCtrl and we employ this module to synthesize moving object trajectories required by our MotionCtrl. However, despite its effectiveness, ParticleSfM is not time-efficient, requiring approximately 2 minutes to process a 32-frame video. To mitigate the issue of time efficiency, we randomly select 32 frames from each WebVid video, with a frame skip interval $s \in [ 1 , 1 6 ]$ , to synthesize the object movement trajectories. This approach yields a total of 243,000 video clips that fulfill the training requirements for the OMCM.

# B DETAILS OF EVALUATION DATASETS

In this paper, we construct two evaluation datasets to independently evaluate the efficacy of our proposed MotionCtrl on camera and object motion control, respectively.

Camera Motion Control Evaluation Dataset. This dataset contains a total of 407 samples covering two types of camera poses:

(1) 80 $( 8 \times 1 0 )$ samples constructed with 8 basic camera pose sequences (pan left, pan right, pan up, pan down, zoom in,

zoom out, anticlockwise rotation, and clockwise rotation) and 10 prompts.   
200 $( 2 0 \times 1 0 )$ samples constructed with 20 relatively complex camera pose sequences randomly selected from the test set of RealEstate10K [Zhou et al. 2018] and 10 prompts.   
(3) 100 samples constructed with 100 relatively complex camera poses of WebVid [Bain et al. 2021] synthesized with ParticleSfM [Zhao et al. 2022] and 100 prompts from VBench [Huang et al. 2024].   
(4) 27 samples constructed with 27 relatively complex camera poses of HD-VILA [Xue et al. 2022] synthesized with ParticleSfM and 27 prompts from VBench [Huang et al. 2024].

To provide an intuitive perception of the camera movement, we visualized the 8 basic camera poses and 20 relatively complex camera poses from RealEstate10K [Zhou et al. 2018] in Fig. 11. As described in the manuscript, the term "complex camera poses" as used in this work denotes camera movements beyond the basic camera poses list, encompassing camera turning and self-rotation within the same camera pose.

Object Motion Control Evaluation Dataset. This evaluation dataset contains a total of 283 samples constructed with 74 diverse trajectories and 77 prompts. It should be noted that to verify the effectiveness of MotionCtrl in object motion control, our evaluation dataset pairs one trajectory with several different prompts or one prompt with several different trajectories. To provide an intuitive perception of the handcrafted trajectories, 19 trajectories adopted in the evaluation dataset are depicted in Fig. 12.

These evaluation datasets will be released.

Please note that the evaluation datasets we have constructed are primarily used for quantitatively assessing the performance of our proposed MotionCtrl in both camera and object motion control in video generation. Our MotionCtrl is capable of handling a wider variety of camera poses and trajectories that are not included in the evaluation datasets.

# C MORE QUANTITATIVE AND QUALITATIVE RESULTS

# C.1 More Quantitative Results

More Quantitative Comparisons on Relatively Complex Camera Motion Control. In the manuscript, the quantitative results of relatively complex camera poses are statistics from all the complex camera poses sourced from RealEstate10K [Zhou et al. 2018], WebVid [Bain et al. 2021], and HD-VILA [Xue et al. 2022]. The statistical results for each dataset are presented in Table 4, demonstrating that our MotionCtrl outperforms VideoComposer [Wang et al. 2023] in both the camera poses extracted from RealEstate10K and those synthesized with ParticleSfM [Zhao et al. 2022] (camera poses of WebVid [Bain et al. 2021] and HD-VILA [Xue et al. 2022]) in terms of camera motion control, text similarity, and generated quality.

User Study.For a more comprehensive evaluation, we conduct a user study involving 34 participants to assess the results of VideoComposer [Wang et al. 2023] and MotionCtrl. The results were generated using object trajectories and relatively complex camera poses covering datasets from RealEstate10K [Zhou et al. 2018], WebVid [Bain et al. 2021], and HD-VILA [Xue et al. 2022]. The assessment included criteria such as Video Quality, Text Similarity, and Motion Similarity. Participants are also asked to express their overall preference for each compared pair. The statistical results in Table 5 demonstrate that over 90 percent of participants preferred our results in all assessment aspects. Although VideoComposer exhibited good performance in motion control conditioned on motion vectors, its generated videos often appeared unnatural and strange due to the object shapes captured by the motion vectors from the

![](images/12.jpg)  
TMotnovaansm posneaive as

![](images/13.jpg)  
Fiur  The ObjcMotin Control valuatio Dataset encopases trajctori where hereen nd bluepots effectiveness of the proposed MotionCtrl in controlling object movements in videos generated.

Ta  o  e ev   omReEa0K Zho   18 WebV [Bai  021nd HD-ILA [u. 2022].

<table><tr><td></td><td colspan="2">RealEstate10K</td><td colspan="2">WebVid</td><td colspan="2">HD-VILA</td></tr><tr><td>Method</td><td>VideoComposer</td><td>MotionCtrl</td><td>VideoComposer</td><td>MotionCtrl</td><td>VideoComposer</td><td>MotionCtrl</td></tr><tr><td>CamMC ↓</td><td>0.1073</td><td>0.0840</td><td>0.0702</td><td>0.0589</td><td>0.0953</td><td>0.0499</td></tr><tr><td>CLIPSIM ↑</td><td>0.2219</td><td>0.2324</td><td>0.2147</td><td>0.2268</td><td>0.2429</td><td>0.2473</td></tr><tr><td>FID ↓</td><td>134.97</td><td>130.29</td><td>106.89</td><td>102.13</td><td>190.54</td><td>159.52</td></tr><tr><td>FVD ↓</td><td>1045.82</td><td>934.37</td><td>733.09</td><td>612.84</td><td>1709.59</td><td>1129.40</td></tr></table>

reference video. Consequently, users showed a stronger preference for our relatively natural results.

Table 5: User Study. Compared to the results generated with VideoComposer [Wang et al. 2023], our MotionCtrl achieved more preference in all assessment aspect.   

<table><tr><td>Method</td><td>VideoComposer</td><td>MotionCtrl</td></tr><tr><td>Quality ↑</td><td>0.0628</td><td>0.9372</td></tr><tr><td>TextSimilarity ↑</td><td>0.0772</td><td>0.9228</td></tr><tr><td>MotionSimilarity ↑</td><td>0.086</td><td>0.9140</td></tr><tr><td>OverallPreference ↑</td><td>0.0739</td><td>0.9261</td></tr></table>

# C.2 More Qualitative Results

More Qualitative Comparisons with VideoComposer. We present additional qualitative results comparing VideoComposer [Wang et al. 2023] and our proposed MotionCtrl on relatively complex camera and object trajectories in Fig. 13 and Fig. 14, respectively. These results suggest that MotionCtrl outperforms VideoComposer in both camera and object motion control in generated videos. Furthermore, MotionCtrl's generated videos exhibit higher quality and its generated content is better aligned with the prompts.

More of MotionCtrl. In this section, we present additional results of MotionCtrl, focusing on camera motion control, object motion control, and combined motion control. Notably, all results are obtained using the same trained MotionCtrl model, without

![](images/14.jpg)  
Prompt: A snail crawling on a leaf.   
FurMore ualitative pariss wiVideCpor Wanal 3Cmeoti CotroThe d he h Zhe  u0 achieved with MotionCtrl exhibit higher quality.

![](images/15.jpg)  
Prompt: A small steel ball rolling on the table.   
ur More ualitativ parisns iVidCopo Wan al 03] n Objec Motin ControlTh geaed

# the need for extra fine-tuning for different camera poses or trajectories.

Specifically, Fig. 17 illustrates the outcomes of camera motion control of MotionCtrl guided by 8 basic camera poses, including pan up, pan down, pan left, pan right, zoom in, zoom out, anticlockwise rotation, and clockwise rotation. These poses are visualized in Fig. 11 (a). This demonstrates the capability of our MotionCtrl model to integrate multiple basic camera motion controls in a unified model, contrasting with the AnimateDiff model [Guo et al. 2023] which requires a distinct LoRA model [Hu et al. 2021] for each camera motion.

Fig. 15 showcases the results of camera motion control using MotionCtrl, which is guided by relatively complex camera poses. These complex camera poses are distinct from basic camera poses, as they include elements of camera turning or selfrotation within the same camera pose sequence. The results demonstrate that, given a sequence of camera poses, our MotionCtrl can generate natural videos. The content of these videos aligns with the text prompts, and the camera motion corresponds to the provided complex camera poses.

Fig. 18 presents the results of object motion control using MotionCtrl, guided by specific trajectories. When given the same trajectories and different text prompts, MotionCtrl can generate videos featuring different objects, but with identical object motion.

Fig. 16 provides the results of combining both the camera motion control and object motion control. With the same trajectory but different camera poses, the horse in the generated videos has a different performance.

![](images/16.jpg)  
Prompt: A temple on the mountain.

![](images/17.jpg)  
Tu o  H  i a p o o closely follows the guided camera poses, while the generated content aligns with the text prompts.   
T  p He With the merajectory but differentcme poses, the horsthegenerate videoshas different perforane.

# D MORE RESULTS OF MOTIONCTRL DEPLOYED ON AIMATEDIFF [Gu0 et al. 2023]

We also deploy our MotionCtrl on AnimateDiff [Guo et al. 2023]. Therefore, we can control the motion of the video generated with our fine-tuned AnimateDiff cooperating with various LoRA [Hu et al. 2021] models in the committee. Results of relatively complex camera motion control and object motion control are in the manuscripts and we provide the results of basic camera motion control here: Fig. 19 and Fig. 20. These results are generated with our MontionCtrl cooperating with different LoRA models provided by in CIVITAI [?]. They demonstrate that our the generalization of MotionCtrl that can be adapted to different video generation models.

# E MORE DISCUSSIONS ABOUT THE RELATED WORKS

To further illustrate the advantages of our proposed MotionCtrl, we've conducted a comparative analysis with previous related works. The comparisons are detailed in Table 6. Models such as AnimateDiff[Guo et al. 2023] (refers to the motion control LoRA models provided by AnimateDiff), Tune-a-video[Wu et al. 2023b], LAMP[Wu et al. 2023a], and MotionDirector[Zhao et al. 2023] implement motion control by extracting motion from one or multiple template videos. This approach necessitates the training of distinct models for each template video or template video set. Moreover, the motions these methods learned are solely determined by the template video(s), and they fail to differentiate between camera motion and object motion. Similarly, MotionDirector[Zhao et al. 2023] and VideoComposer[Wang et al. 2023], despite achieving motion control with a unified model guided by motion vectors and trajectories respectively, do not distinguish between camera motion and object motion. In contrast, our proposed MotionCtrl, utilizing a unified model, can independently and flexibly control a wide range of camera and object motions in the generated videos. This is achieved by guiding the model with camera poses and trajectories respectively, offering a more fine-grained control over the video generation process.

Ta e MoneaorkUnl 0 hi t the cnrol LoRA moe povie b niaei, Tunea-vido [Wu  023], LAMP Wu  2], n . leee pheui heh n   h0ViWn 0 mno i  na o Mot e ey anexo respectively.   

<table><tr><td>Method</td><td>Require Fine-tuning</td><td>Motion sources</td><td>Distinguish Camera &amp; Object Motion</td></tr><tr><td>AnimateDiff [Guo et al. 2023]</td><td></td><td>template videos</td><td>X</td></tr><tr><td>Tune-a-video [Wu et al. 2023b]</td><td>2</td><td>template video</td><td>X</td></tr><tr><td>LAMP [Wu et al. 2023a]</td><td></td><td>template videos</td><td>X</td></tr><tr><td>MotionDirector [Zhao et al. 2023]</td><td>✓</td><td>template videos</td><td>X</td></tr><tr><td>VideoComposer [Wang et al. 2023]</td><td>×</td><td>motion vectors</td><td>X</td></tr><tr><td>DragNUWA [Yin et al. 2023a]</td><td>×</td><td>trajectories</td><td></td></tr><tr><td>MotionCtrl (Ours)</td><td>X</td><td>camera poses &amp; trajectories</td><td>.</td></tr></table>

![](images/18.jpg)  
Prompt: A landscape with mountains and lake at sunrise.

He       , without the need for extra fine-tuning for different camera poses.

![](images/19.jpg)  
Prompt: Two cats.   
8 The esul   o deploye L [He  02], i wijeo. The e ncu i

![](images/20.jpg)  
Prompt: A teddy bear in the supermarket.   
F  u   h basic camera poses.

![](images/21.jpg)  
Prompt: Catle on the mountain.

T  u only control the camera motion of the generated videos but also their motion speed.