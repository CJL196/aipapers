# HunyuanVideo: 大型视频生成模型的系统框架

“弥合闭源与开源视频基础模型之间的鸿沟，以加速社区探索。” — 混元基础模型团队

# 摘要

最近的视频生成技术进展深刻地改变了个人和各个行业的日常生活。然而，目前领先的视频生成模型仍然是闭源的，这导致行业与公众社区之间在视频生成能力上存在显著的性能差距。在本报告中，我们提出了HunyuanVideo，这是一个新型的开源视频基础模型，其在视频生成方面的性能可与领先的闭源模型媲美，甚至更优。HunyuanVideo具备一个综合框架，整合了多个关键贡献，包括数据整理、先进的架构设计、渐进式模型扩展与训练，以及旨在促进大规模模型训练和推理的高效基础设施。借助这些，我们成功训练了一个超过130亿参数的视频生成模型，成为所有开源模型中规模最大的。我们进行了广泛的实验并实施了一系列针对性的设计，以确保高视觉质量、运动动态、文本与视频的对齐以及先进的拍摄技术。根据专业评估结果，HunyuanVideo的表现超越了先前的最先进模型，包括Runway Gen-3、Luma 1.6以及3个表现最佳的中国视频生成模型。通过发布基础模型及其应用的代码，我们旨在缩小闭源社区与开源社区之间的差距。这一举措将赋能社区中的每个人尝试他们的想法，推动一个更具活力和生机的视频生成生态系统。代码已公开在 https://github.com/Tencent/HunyuanVideo。Hunyuan基础模型团队的贡献者名单附在报告末尾。

# 1 引言

通过广泛的预训练和先进的架构，扩散模型已显示出在生成高质量图像和视频方面优于以往的生成对抗网络（GAN）方法。然而，与图像生成领域相比，后者在各种开放平台上出现了大量新算法和应用，基于扩散的视频生成模型仍然相对不活跃。我们认为，这一停滞的主要原因之一是缺乏强大的开源基础模型，类似于T2I领域。与图像生成模型社区相比，开源和封闭源视频生成模型之间出现了显著差距。封闭源模型往往掩盖了公开可用的开源替代方案，严重限制了公众社区在算法创新方面的潜力。尽管近期最先进的模型MovieGen展示了令人鼓舞的表现，但其开源发布的里程碑尚未确立。

![](images/1.jpg)  

Figure 2: Left: Computation resources used for closed-source and open-source video generation models. Right: Performance comparison between HunyuanVideo and other selected strong baselines.

为了解决现有的差距并增强公众社区的能力，本报告提出了我们的开源基础视频生成模型 HunyuanVideo。该系统框架涵盖了训练基础设施、数据整理、模型架构优化和模型训练。通过实验，我们发现随机扩展训练数据、计算资源和简单的基于 Transformer 的生成模型参数，使用 Flow Matching 进行训练并不够高效。因此，我们探索了一种有效的扩展策略，可以将计算资源需求减少多达 $5 \times$ ，同时实现所需的模型性能。通过这种最优扩展方法和专用基础设施，我们成功训练了一个包含 130 亿参数的大型视频模型，在互联网规模的图像和视频上进行了预训练。在经过专门的渐进微调策略后，HunyuanVideo 在视频生成的四个关键方面表现出色：视觉质量、运动动态、视频-文本对齐和语义场景切割。我们对 HunyuanVideo 与全球领先的视频生成模型（包括 Gen-3 和 Luma 1.6，以及中国的三款表现最佳的商业模型）进行了全面比较，使用超过 1,500 个具有代表性的文本提示，参与者为 60 人。结果表明，HunyuanVideo 达到了最高的整体满意度，特别是在运动动态方面表现突出。

# 2 概述

HunyuanVideo是一个全面的视频训练系统，涵盖从数据处理到模型部署的各个方面。本技术报告的结构如下： • 第3部分介绍了我们的数据预处理技术，包括过滤和重标注模型。 第4部分详细说明了HunyuanVideo所有组件的架构，以及我们的训练和推理策略。 在第5部分，我们讨论了加速模型训练和推理的方法，使得开发一个具有130亿参数的大模型成为可能。 • 第6部分评估了我们的文本到视频基础模型的性能，并将其与最先进的视频生成模型（包括开源和专有模型）进行了比较。 • 最后，在第7部分，我们展示了基于预训练基础模型构建的各种应用，并附带相关的可视化以及一些与视频相关的功能模型，如视频到音频生成模型。

![](images/2.jpg)  

Figure 3: The overall training system for Hunyuan Video.

# 3 数据预处理

我们采用图像-视频联合训练策略。视频被细致地分为五个不同的组，而图像则被分类为两个组，每个组均旨在满足各自的训练过程的特定要求。本节将主要深入探讨视频数据的整理细节。我们的数据获取过程严格遵循《通用数据保护条例》(GDPR) [39] 框架中规定的原则。此外，我们还采用数据合成和隐私计算等先进技术，以确保遵循这些严格标准。我们的原始数据池初步包含了涵盖人、动物、植物、风景、车辆、物体、建筑和动画等多个领域的视频。每个视频的获取需满足一系列基本门槛，包括最低时长要求。此外，还有一部分数据是基于更严格的标准收集的，例如空间质量、特定宽高比的遵循，以及在构图、色彩和曝光方面的专业标准。这些严格标准确保了我们的视频具有技术质量和审美吸引力。我们的实验验证了高质量数据的引入对显著提升模型性能的重要性。

# 3.1 数据过滤

我们来自不同来源的原始数据表现出不同的持续时间和质量水平。为了解决这个问题，我们采用一系列技术对原始数据进行预处理。首先，我们利用 PySceneDetect [19] 将原始视频分割成单镜头视频片段。接下来，我们使用 OpenCV [18] 中的拉普拉斯算子来识别清晰帧，作为每个视频片段的起始帧。通过内部 VideoCLIP 模型，我们计算这些视频片段的嵌入。这些嵌入有两个目的：(i) 我们根据嵌入的余弦距离对相似片段进行去重；(ii) 我们应用 $\mathbf { k }$ -均值 [59] 来获得约 $10 \mathrm { K }$ 的概念中心，用于概念重采样和均衡。为了持续提升视频的美学、运动和概念范围，我们实施了一个分层数据过滤管道来构建训练数据集，如图 4 所示。该管道结合了多种过滤器，以帮助我们从不同的角度过滤数据，这一点将在下文介绍。我们采用 Dover [85] 评估视频片段的视觉美学，从美学和技术两个角度进行分析。此外，我们训练模型来判断清晰度，并消除模糊的视频片段。通过预测视频的运动速度，利用估计的光流 [18]，我们过滤掉静态或慢动作视频。我们结合 PySceneDetect [19] 和 Transnet v2 [76] 的结果来获取场景边界信息。我们利用内部 OCR 模型去除文本过多的视频片段，并定位和裁剪字幕。我们还开发了类似 YOLOX [24] 的视觉模型，检测并去除一些被遮挡或敏感的信息，如水印、边框和标志。为了评估这些过滤器的有效性，我们使用较小的 HunyuanVideo 模型进行简单实验，并观察性能变化。这些实验获得的结果在指导我们构建数据过滤管道方面起到了重要作用，接下来将进行介绍。

![](images/3.jpg)  

Figure 4: Our hierarchical data filtering pipeline. We employ various filters for data filtering and progressively increase their thresholds to build 4 training datasets, i.e., 256p, 360p, 540p, and $7 2 0 \mathrm { p }$ , while the final SFT dataset is built through manual annotation. This figure highlights some of the most important filters to use at each stage. A large portion of data will be removed at each stage, ranging from half to one-fifth of the data from the previous stage. Here, gray bars represent the amount of data filtered out by each filter while colored bars indicate the amount of remaining data at each stage.

我们的视频数据层级过滤管道生成了五个训练数据集，分别对应于五个训练阶段（第4.5节）。这些数据集（最后的微调数据集除外）是通过逐步提高前述过滤器的阈值来精心策划的。视频的空间分辨率从 $256 \times 256 \times 65$ 逐渐增加到 $720 \times 1280 \times 129$。在不同阶段的阈值调整过程中，我们对过滤器施加了不同程度的严格性（见图4）。最后的微调数据集将在后文描述。为了提高模型在最后阶段的性能（第4.7节），我们构建了一个包含约 $1M$ 样本的微调数据集。该数据集通过人工标注仔细策划。评估者负责识别具有高视觉美学和引人入胜的内容运动的视频片段。每个视频片段的评估基于两个方面：（i）分解的美学视角，包括色彩和谐、光照、对象强调和空间布局；（ii）分解的运动视角，包括运动速度、动作完整性和运动模糊。最终，我们的微调数据集由具有复杂运动细节的视觉吸引力视频片段组成。我们还通过重复使用大部分过滤器（不包括与运动相关的过滤器）建立了图像的层级数据过滤管道。类似地，我们通过逐步增加应用于数十亿图像-文本对的图像库的过滤阈值，构建了两个图像训练数据集。第一个数据集包含数十亿个样本，用于文本到图像预训练的初始阶段。第二个数据集包含数亿个样本，用于文本到图像预训练的第二阶段。

# 3.2 数据标注

结构化字幕。正如研究[7, 4]所示，字幕的准确性和全面性在提升生成模型的提示跟随能力和输出质量方面起着至关重要的作用。大多数早期工作侧重于提供简短字幕[14, 50]或密集字幕[93, 9, 10]。然而，这些方法并非没有缺陷，存在信息不完整、冗余叙述和不准确等问题。为了实现更高全面性、信息密度和准确性的字幕，我们开发并实施了一种内部的视觉语言模型（VLM），旨在为图像和视频生成结构化字幕。这些以JSON格式呈现的结构化字幕，从多个角度提供多维的描述信息，包括：1) 简短描述：捕捉场景的主要内容。

![](images/4.jpg)  

Figure 5: The overall architecture of HunyuanVideo. The model is trained on a spatial-temporally compressed latent space, which is compressed through Causal 3D VAE. Text prompts are encoded using a large language model, and used as the condition. Gaussian noise and condition are taken as input, our model generates a output latent, which is decoded into images or videos through the 3D VAE decoder.

2) 密集描述：详细描述场景内容，包括场景转变和与视觉内容结合的相机运动，比如相机跟随某个主体。3) 背景：描述主体所处的环境。4) 风格：表征视频镜头类型，强调特定视觉内容，例如空中镜头、特写镜头、中景镜头或长镜头。6) 照明：描述视频的照明条件。7) 氛围：传达视频的氛围，如温馨、紧张或神秘。此外，我们扩展了JSON结构，整合了其他源标签、质量标签及来自图像和视频元信息的相关标签。通过实施精心设计的丢弃机制以及置换和组合策略，我们合成了不同长度和模式的字幕，通过为每个图像和视频组装这些多维描述，旨在提高生成模型的泛化能力并防止过拟合。我们利用该字幕生成器为训练数据集中所有图像和视频提供结构化字幕。相机运动类型。我们同样训练了一个相机运动分类器，能够预测14种不同的相机运动类型，包括放大、缩小、向上平移、向下平移、向左平移、向右平移、向上倾斜、向下倾斜、向左倾斜、向右倾斜、向左环绕、向右环绕、静态镜头和手持镜头。高置信度的相机运动类型预测被整合到JSON格式的结构化字幕中，以增强生成模型的相机运动控制能力。

# 4 模型架构设计

我们HunyuanVideo模型的概述如图5所示。本节描述了因果3D变分自编码器、扩散主干网络和规模法则实验。

# 4.1 3D 变分自编码器设计

与之前的工作类似，我们训练了一个3DVAE，将像素空间的视频和图像压缩到紧凑的潜在空间。为了同时处理视频和图像，我们采用了CausalConv3D。对于形状为$( T + 1 ) \times 3 \times H \times W$的视频，我们的3DVAE将其压缩为形状为$\begin{array} { r } { ( \frac { T } { c _ { t } } + 1 ) \times C \times ( \frac { H } { c _ { s } } ) \times ( \frac { W } { c _ { s } } ) } \end{array}$的潜在特征。在我们的实现中，$c _ { t } = 4$，${ c _ { s } = 8 }$，以及$C = 16$。这种压缩显著减少了随后的扩散变换模型所需的词元数量，使我们能够以原始分辨率和帧率训练视频。模型结构如图6所示。

![](images/5.jpg)  

Figure 6: The architecture of our 3DVAE.

# 4.1.1 训练

与大多数先前的研究不同，我们不依赖于预训练的图像变分自编码器（VAE）进行参数初始化；相反，我们从头开始训练我们的模型。为了平衡视频和图像的重建质量，我们以 $4 : 1$ 的比例混合视频和图像数据。除了常规使用的 $L_{1}$ 重建损失和 KL 损失 $L_{kl}$ 外，我们还结合了感知损失 $L_{lpips}$ 和 GAN 对抗损失 $L_{adv}$ 来提升重建质量。完整的损失函数如公式 1 所示。

$$
\mathrm { L o s s } = L _ { 1 } + 0 . 1 L _ { l p i p s } + 0 . 0 5 L _ { a d v } + 1 0 ^ { - 6 } L _ { k l }
$$

在训练过程中，我们采用了一种课程学习策略，从低分辨率短视频逐步训练到高分辨率长视频。为了提高高运动视频的重建效果，我们在 $1 \sim 8$ 的范围内随机选择一个采样间隔，以均匀地从视频片段中采样帧。

# 4.1.2 推理

在单个 GPU 上编码和解码高分辨率长视频可能会导致内存溢出（OO）错误。为了解决这个问题，我们采用了一种时空切片策略，将输入视频沿空间和时间维度切分为重叠的小块。每个小块分别进行编码/解码，输出结果再进行拼接。对于重叠区域，我们利用线性组合进行混合。该切片策略使我们能够在单个 GPU 上以任意分辨率和时长编码/解码视频。我们观察到在推理过程中直接使用切片策略可能会由于训练和推理之间的不一致而导致可见伪影。为了解决这个问题，我们引入了一个额外的微调阶段，在训练过程中随机启用/禁用切片策略。这确保了模型能够兼容切片和非切片策略，保持训练和推理之间的一致性。表1比较了我们的变分自编码器（VAE）与开源最先进的 VAE。在视频数据上，我们的 VAE 表现出显著高于其他视频 VAE 的 PSNR。在图像上，我们的性能超过了视频 VAE 和图像 VAE。图7 显示了几个 $2 5 6 \times 2 5 6$ 分辨率的案例。我们的 VAE 在文本、小脸和复杂纹理方面表现出显著优势。

Table 1: VAE reconstruction metrics comparison.   

<table><tr><td>Model</td><td>Downsample Factor</td><td>|z|</td><td>ImageNet (256×256) PSNR↑</td><td>MCL-JCV (33×360×640) PSNR↑</td></tr><tr><td>FLUX-VAE [47]</td><td>1×8×8</td><td>16</td><td>32.70</td><td>-</td></tr><tr><td>OpenSora-1.2 [102]</td><td>4×8×8</td><td>4</td><td>28.11</td><td>30.15</td></tr><tr><td>CogvideoX-1.5 [93]</td><td>4× 8×8</td><td>16</td><td>31.73</td><td>33.22</td></tr><tr><td>Cosmos-VAE [64]</td><td>4×8×8</td><td>16</td><td>30.07</td><td>32.76</td></tr><tr><td>Ours</td><td>4×8×8</td><td>16</td><td>33.14</td><td>35.39</td></tr></table>

![](images/6.jpg)  

Figure 7: VAE reconstruction case comparison.

![](images/7.jpg)  

Figure 8: The architecture of our HunyuanVideo Diffusion Backbone.

# 4.2 统一的图像和视频生成架构

在本节中，我们介绍了HunyuanVideo中的Transformer设计，该设计采用统一的全注意力机制，主要有三个原因：首先，与分割时空注意力相比，它表现出更优越的性能[7, 67, 93, 79]。其次，它支持对图像和视频的统一生成，简化了训练过程并提高了模型的可扩展性。最后，它更有效地利用现有的大语言模型相关加速能力，提高了训练和推理的效率。模型结构如图8所示。

输入。对于给定的视频-文本对，模型在第4.1节中描述的3D潜在空间中操作。具体而言，对于视频分支，输入首先被压缩为形状为 $T \times C \times H \times W$ 的潜在向量。为了统一输入处理，我们将图像视为单帧视频。这些潜在向量采用大小为 $k _ { t } \times k _ { h } \times k _ { w }$ 的3D卷积进行处理，形状为 $\frac { T } { k _ { t } } \cdot \frac { H } { k _ { h } } \cdot \frac { W } { k _ { w } }$。对于文本分支，我们首先使用高级大语言模型将文本编码为捕获细粒度语义信息的嵌入序列。同时，我们使用CLIP模型提取包含全局信息的归一化文本表示。该表示随后在维度上扩展，并在输入模型之前与时间步嵌入相加。

Table 2: Architecture hyperparameters for the HunyuanVideo 13B parameter foundation model.   

<table><tr><td>Dual-stream Blocks</td><td>Single-stream locks</td><td>Model Dimension</td><td>FFN Dimension</td><td>Attention Heads</td><td>Head dim</td><td>(d, dh, dw)</td></tr><tr><td>20</td><td>40</td><td>3072</td><td>12288</td><td>24</td><td>128</td><td>(16, 56, 56)</td></tr></table>

模型设计。为了有效整合文本和视觉信息，我们采用了与[47]中视频生成提出的“从双流到单流”混合模型设计类似的策略。在双流阶段，视频和文本词元通过多个Transformer模块独立处理，使得每种模态能够学习其各自适当的调制机制而不受干扰。在单流阶段，我们将视频和文本词元连接起来，并输入到后续的Transformer模块中以实现有效的多模态信息融合。这一设计捕捉了视觉信息与语义信息之间的复杂交互，增强了整体模型性能。位置嵌入。为了支持多分辨率、多宽高比和不同时长的生成，我们在每个Transformer模块中使用旋转位置嵌入（RoPE）[77]。RoPE将旋转频率矩阵应用于嵌入，增强了模型捕捉绝对和相对位置关系的能力，并展示了大语言模型中的某种外推能力。考虑到视频数据中时间维度的额外复杂性，我们将RoPE扩展到三维。具体而言，我们分别为时间$( T )$、高度$( H )$和宽度$( W )$的坐标计算旋转频率矩阵。然后，我们将查询和键的特征通道分为三个部分$ ( d _ { t } , d _ { h } , d _ { w } ) $，将每个部分与相应的坐标频率相乘并连接这些部分。这个过程生成了具备位置信息的查询和键嵌入，用于注意力计算。有关详细的模型设置，请参见表2。

# 4.3 文本编码器

在文本到图像和文本到视频等生成任务中，文本编码器通过提供潜在空间中的指导信息发挥着关键作用。一些代表性工作通常使用预训练的 CLIP 和 T5-XXL 作为文本编码器，其中 CLIP 使用 Transformer 编码器，T5 使用编码器-解码器结构。相比之下，我们利用预训练的多模态大语言模型（MLLM），其结构为仅解码器，这具有以下优势：（i）与 T5 相比，经过视觉指令微调后的 MLLM 在特征空间中的图像-文本对齐能力更强，从而缓解了扩散模型中指令跟随的困难；（ii）与 CLIP 相比，MLLM 在图像细节描述和复杂推理方面的能力已被证明更为出色；（iii）MLLM 可以作为零样本学习者，通过遵循附加在用户提示前的系统指令，帮助文本特征更关注关键信息。此外，如图 9 所示，MLLM 基于因果注意力，而 T5-XXL 则利用双向注意力，这为扩散模型提供了更好的文本指导。因此，我们遵循引入额外的双向令牌精炼器以增强文本特征。我们为不同的用途配置了以一系列 MLLM 为基础的 HunyuanVideo。在每种设置下，MLLM 的表现均优于传统文本编码器。此外，CLIP 的文本特征也作为文本信息的总结具有重要价值。如图 8 所示，我们采用 CLIP-Large 文本特征的最终非填充令牌作为全局指导，整合到双流和单流 DiT 结构中。

# 4.4 模型扩展

语言模型训练中的神经规模法则 [41, 36] 为理解和优化机器学习模型的性能提供了强有力的工具。通过阐明模型规模 $( N )$、数据集规模 $( D )$ 和计算资源 $( C )$ 之间的关系，这些法则有助于推动更有效和高效模型的发展，最终促进大规模模型训练的成功。

![](images/8.jpg)  

Figure 9: Text encoder comparison between T5 XXL and the instruction-guided MLLM introduced by HunyuanVideo.

![](images/9.jpg)  

Figure 10: Scaling laws of DiT-T2X model family. On the top-left (a) we show the loss curves of the T2X(I) model on a log-log scale for a range of model sizes from 92M to 6.6B. We follow [36] to plot the envelope in gray points, which are used to estimate the power-law coefficients of the amount of computation $( C )$ vs model parameters $( N )$ (b) and the computation vs tokens $( D )$ (c). Based on the scaling law of the T2X(I) model, we plot the scaling law of the corresponding T2X(V) model in (d), (e), and (f).

与之前关于大语言模型和图像生成模型的缩放法则相比，视频生成模型通常依赖于预训练的图像模型。因此，我们的第一步是建立与文本到图像相关的基础缩放法则。在这些基础缩放法则的基础上，我们随后推导出了适用于文本到视频模型的缩放法则。通过整合这两组缩放法则，我们能够系统性地确定视频生成任务的模型和数据配置。

# 4.4.1 图像模型缩放规律

Kaplan 等人 [41] 和 Hoffmann 等人 [36] 探索了语言模型在交叉熵损失下的经验缩放法则。在基于扩散的视觉生成领域，Li 等人 [49] 研究了 UNet 的缩放特性，而基于变换器的工作，如 DiT [65]、U-ViT [3]、Lumina-T2X [23] 和 SD3 [21] 仅研究了样本质量与网络复杂性之间的缩放行为，未探讨扩散模型所用计算资源与均方误差损失之间的幂律关系。为了填补这一空白，我们开发了一系列类似 DiT 的模型，命名为 DiT-T2X，以区别于原始 DiT，其中 X 可以是图像 (I) 或视频 (V)。DiT-T2X 应用 T5-XXL [71] 作为文本编码器，并采用上述 3D VAE 作为图像编码器。文本信息的层数从 92M 增加到 6.6B。这些模型是在 DDPM [34] 和 v-prediction [73] 的一致超参数和相同数据集（256px 分辨率）下进行训练的。我们遵循 [36] 引入的实验方法，并建立神经缩放法则进行拟合。

$$
N _ { o p t } = a _ { 1 } C ^ { b _ { 1 } } , \quad D _ { o p t } = a _ { 2 } C ^ { b _ { 2 } } .
$$

如图10(a)所示，每个模型的损失曲线从左上角下降到右下角，并始终经过相邻的较大模型的损失曲线。这意味着每条曲线将在两个交点之间的计算资源上进行折中，中型模型是最优的（损失最低）。在获得所有$\mathbf{X}$轴值上的最低损失包络后，我们填充方程()发现$a_{1} = 5.48 \times 10^{-4}, b_{1} = 0.5634, a_{2} = 0.324$和$b_{2} = 0.4325$，其中$a_{1}, a_{2}, N_{opt}, D_{opt}$的单位为十亿，而$C$的单位为Peta FLOPs。图10(b)和图10(c)显示DiT-T2X(I)系列非常符合幂律。最后，考虑到计算预算，我们可以计算出最优模型大小和数据集大小。

# 4.4.2 视频模型规模法则

基于T2X(I)模型的缩放规律，我们选择每个尺寸模型对应的最佳图像检查点（即模型在包络线上），作为视频缩放规律实验的初始化模型。图10 (d)、图10 (e)和图10 (f)展示了T2X(V)模型的缩放规律结果，其中 $a _ { 1 } = 0 . 0 1 8 9$ , $b _ { 1 } = 0 . 3 6 1 8$ , $a _ { 2 } = 0 . 0 1 0 8$ 和 $b _ { 2 } = 0 . 6 2 8 9$。根据图10 (b)和图10 (e)的结果，并考虑训练消耗和推理成本，我们最终设定模型尺寸为13B。然后，可以计算出图像和视频训练的词元数量，如图10 (c)和图10 (f)所示。值得注意的是，根据图像和视频缩放规律计算的训练词元数量仅与图像和视频各自的第一阶段训练相关。从低分辨率到高分辨率的渐进训练的缩放特性将留待未来的工作中探讨。

# 4.5 模型预训练

我们使用流匹配 [52] 进行模型训练，并将训练过程分为多个阶段。我们首先在 256px 和 512px 图像上进行预训练，然后在 256px到 $960 \mathrm{px}$ 的图像和视频上进行联合训练。

# 4.5.1 训练目标

在本研究中，我们采用了流匹配框架 [52, 21, 13] 来训练我们的图像和视频生成模型。流匹配通过一系列对概率密度函数的变量变换，将复杂的概率分布转化为简单的概率分布，并通过逆变换生成新的数据样本。

在训练过程中，给定训练集中图像或视频的潜在表示 $\mathbf { x } _ { 1 }$。我们首先从对数正态分布中采样 $t \in [ 0 , 1 ]$，并初始化噪声 $\mathbf { x } _ { 0 } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$，遵循高斯分布。然后使用线性插值方法构造训练样本 $\mathbf { x } _ { t }$。模型被训练以预测速度 $\mathbf { u } _ { t } = d \mathbf { x } _ { t } / d t$，该速度引导样本 $\mathbf { x } _ { t }$ 向样本 $\mathbf { x } _ { 1 }$ 移动。通过最小化预测速度 $\mathbf { v } _ { t }$ 和真实速度 $\mathbf { u } _ { t }$ 之间的均方误差来优化模型参数，该过程表示为损失函数。

$$
\mathcal { L } _ { \mathrm { g e n e r a t i o n } } = \mathbb { E } _ { t , { \mathbf { x } _ { 0 } } , { \mathbf { x } _ { 1 } } } \| \mathbf { v } _ { t } - { \mathbf { u } } _ { t } \| ^ { 2 } .
$$

在推理过程中，初始抽取一个噪声样本 $\mathbf { x } _ { 0 } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$。然后使用一阶欧拉常微分方程 (ODE) 求解器通过对模型估计的 $\frac{d \mathbf{x}_{t}}{d t}$ 进行积分来计算 $\mathbf{x}_{1}$。这个过程最终生成最终样本 $\mathbf{x}_{1}$。

# 4.5.2 图像预训练

在我们的早期实验中，我们发现一个经过良好预训练的模型显著加快了视频训练的收敛速度，并改善了视频生成性能。因此，我们引入了一种两阶段渐进式图像预训练策略作为视频训练的预热。 图像阶段1（256px训练）。模型首先使用低分辨率256px图像进行预训练。具体而言，我们遵循之前的工作[66]，基于256px进行多方面训练，这有助于模型学习生成具有多种宽高比的图像，同时避免由于图像预处理中的裁剪操作导致的文本与图像的错位。与此同时，使用低分辨率样本进行预训练使得模型能够从大量样本中学习更多的低频概念。

图像阶段2（混合尺度训练）。我们引入第二个图像预训练阶段，以进一步提升模型在更高分辨率下的能力，例如 ${ \mathsf { 5 1 2 p x } }$。一个简单的解决方案是直接在 ${ \mathsf { 5 1 2 p x } }$ 图像上进行微调。然而，我们发现模型在 ${ \mathsf { 5 1 2 p x } }$ 图像上微调后的性能在生成 256px 图像时会严重下降，这可能会影响后续在 256px 视频上的视频预训练。因此，我们提出了混合尺度训练，这种方法在每个训练全局批次中包含两个或多个尺度的多维桶。每个尺度具有一个锚点大小，随后根据锚点大小构建多维桶。我们在一个具有锚点大小为 256px 和 ${ \mathsf { 5 1 2 p x } }$ 的两尺度数据集上训练模型，以学习更高分辨率的图像，同时保持在低分辨率图像上的能力。我们还引入了针对不同图像尺度的动态批量大小，以最大化 GPU 内存和计算利用率。

# 4.5.3 视频-图像联合训练

多种纵横比和时长的分桶。经过第3.1节所述的数据过滤过程后，视频具有不同的纵横比和时长。为了有效利用数据，我们根据时长和纵横比将训练数据分为不同的桶。我们创建了 $B _ { T }$ 个时长桶和 $B _ { A R }$ 个纵横比桶，总共形成 $B _ { T } \times B _ { A R }$ 个桶。由于各个桶中的词元数量不同，我们为每个桶分配了一个最大批量大小，以防止内存溢出 (OOM) 错误，从而优化 GPU 资源的使用。在训练之前，所有数据被分配到最近的桶。在训练过程中，每个排名随机预取来自某个桶的批量数据。这种随机选择确保模型在每一步训练中使用不同大小的数据，从而帮助维护模型的泛化能力，避免仅对单一大小进行训练而产生的局限性。 渐进式视频-图像联合训练。直接从文本生成高质量、长时长的视频序列常常导致模型收敛困难和结果不理想。因此，渐进式课程学习已成为训练文本转视频模型的广泛采用策略。在 HunyuanVideo 中，我们设计了一种全面的课程学习策略，从使用 T2I 参数初始化模型开始，并逐渐增加视频时长和分辨率。 • 低分辨率、短视频阶段。模型建立文本与视觉内容之间的基本映射，确保短期动作的一致性和连贯性。 • 低分辨率、长视频阶段。模型学习更复杂的时间动态和场景变化，确保更长时长下的时间和空间一致性。 • 高分辨率、长视频阶段。模型提高视频的分辨率和细节质量，同时保持时间一致性和管理复杂的时间动态。此外，在每个阶段，我们以不同的比例引入图像进行视频-图像联合训练。这种方法解决了高质量视频数据稀缺的问题，使模型能够学习更广泛和多样的世界知识，同时有效防止由于视频和图像数据之间的分布差异导致的图像空间语义的灾难性遗忘。

# 4.6 提示重写

为了解决用户提供的提示在语言风格和长度上的变异性，我们采用 Hunyuan-Large 模型作为提示重写模型，以将原始用户提示调整为模型偏好的提示。该提示重写模型在无训练框架内运作，利用详细的提示指令和上下文学习示例来提升其性能。该提示重写模块的主要功能如下： • 多语言输入适应：该模块旨在处理和理解不同语言的用户提示，确保意义和上下文得以保留。 • 提示结构标准化：该模块将提示重写为符合标准化信息架构的形式，类似于训练标题。 • 复杂术语简化：该模块将复杂的用户用词简化为更通俗的表达，同时保持用户的原始意图。此外，我们还实施了一种自我修订技术，以优化最终提示。这涉及对原始提示和重写版本之间的比较分析，以确保输出既准确又符合模型的能力。为了加速和简化应用过程，我们还对 Hunyuan-Large 模型进行了 LoRA 微调以进行提示重写。该 LoRA 调整的训练数据来源于通过无训练方法收集的高质量重写对。

# 4.7 高性能模型微调

在预训练阶段，我们利用了一个大型数据集进行模型训练。虽然该数据集信息丰富，但在数据质量上显示出相当大的变异性。为了创建一个能够生成高质量、动态视频的强大生成模型，并提升其在连续运动控制和角色动画方面的能力，我们从完整数据集中精心挑选了四个特定子集进行微调。这些子集经过了自动数据过滤技术的初步筛选，随后进行了人工审查。此外，我们实施了多种模型优化策略，以最大化生成性能。

# 5 模型加速

![](images/10.jpg)  

Figure 11: (a) Different time-step schedulers. For our shifting stragty, we set a larger shifting factor $s$ for a lower inference step. (b) Generated videos with only 10 inference steps. The shifting stragty leads to significantly better visual quality.

# 5.1 推理步骤简化

为了提高推理效率，我们首先考虑减少推理步骤。与图像生成相比，保持生成视频的空间和时间质量在较低的推理步骤下更具挑战性。受到之前观察的启发，即首个时间步骤在生成过程中对大多数变化贡献较大，我们利用时间步骤位移来处理较低推理步骤的情况。具体来说，给定推理步骤 $q \in \{ 1 , 2 , . . . , Q \}$ ，$\textstyle t = 1 - { \frac { q } { Q } }$ 是生成模型的输入时间条件，其中噪声在 $t = 1$ 时被初始化，生成过程在 $t = 0$ 时暂停。我们不直接使用 $t$ ，而是通过位移函数将 $t$ 映射到 $t ^ { \prime }$ ：$\begin{array} { r } { \bar { t ^ { \prime } } = \frac { s * t ^ { - } } { 1 + ( s - 1 ) * t } } \end{array}$，其中 $t ^ { \prime }$ 中的 $s$ 是位移因子。如果 $s > 1$，流模型更加依赖于早期时间步骤。一个关键观察是，较低的推理步骤需要更大的位移因子 $s$。根据经验，当推理步骤为 50 步时，$s$ 设置为 7，而当推理步骤少于 20 步时，$s$ 应增加到 17。时间步骤位移策略使生成模型能够用更少的步骤匹配多个推理步骤的结果。MovieGen 应用线性-二次调度器达到类似目标。调度器在图11a中可视化。然而，我们发现，在极低推理步骤（例如，10 步）的情况下，我们的时间步骤位移比线性-二次调度器更有效。如图11b所示，线性-二次调度器导致更差的视觉质量。

# 5.2 文本引导蒸馏

无分类器引导（CFG）显著提高了文本引导扩散模型的样本质量和运动稳定性。然而，它增加了计算成本和推理延迟。在视频模型和高分辨率视频生成中，同时生成文本条件和无条件视频时，推理负担异常昂贵。为了解决这一限制，我们将无条件和条件输入的组合输出提炼到一个单一的学生模型中。具体而言，学生模型以引导比例为条件，并共享与教师模型相同的结构和超参数。我们使用与教师模型相同的参数初始化学生模型，并使用从 1 到 8 随机采样的引导比例进行训练。实验发现，文本引导提炼大约带来了 $1.9 \mathrm{{x}}$ 的加速。

# 5.3 高效且可扩展的训练

为了实现可扩展性和高效训练，我们在来自腾讯 Angel 机器学习团队的大规模预训练框架 AngelPTM [62] 上训练 HunyuanVideo。在这一部分，我们首先概述用于训练的硬件和基础设施，然后详细介绍模型并行方法及其优化方法，最后介绍自动容错机制。

# 5.3.1 硬件基础设施

为了确保大规模分布式训练中的高效通信，我们建立了一个专用的分布式训练框架，称为腾讯星脉网络[48]，以实现高效的服务器间通信。所有训练任务的GPU调度通过腾讯Angel机器学习平台完成，该平台提供强大的资源管理和调度能力。

# 5.3.2 并行策略

HunyuanVideo训练采用5D并行策略，包括张量并行（TP）、序列并行（SP）、上下文并行（CP）以及结合Zero优化的数据并行（DP + ZeroCache）。张量并行（TP）基于矩阵的块计算原则，将模型参数（张量）分配到不同的GPU上，以降低GPU内存使用并加速计算。每个GPU负责计算层中张量的不同部分。序列并行（SP）基于TP，通过对输入序列维度进行切片，减少LayerNorm和Dropout等操作符的重复计算，降低相同激活值的存储，从而有效降低计算资源和GPU内存的浪费。此外，对于不符合SP要求的输入数据，支持工程等效的SP Padding能力。上下文并行（CP）在序列维度进行切片，以支持长序列训练。每个GPU负责计算不同序列切片的注意力。具体来说，通过使用Ring Attention实现多GPU对长序列的高效训练，突破单个GPU的内存限制。此外，利用数据并行$^ +$ ZeroCache，通过数据并行支持横向扩展，以满足对训练数据集增加的需求。然后基于数据并行，采用ZeroCache优化策略进一步减少模型状态的冗余（模型参数、梯度和优化器状态），统一使用GPU内存以最大化GPU内存使用效率。

# 5.3.3 优化

注意力优化。随着序列长度的增加，注意力计算成为训练的主要瓶颈。我们通过 FusedAttention 加速了注意力计算。重计算和激活卸载优化。重计算是一种在计算和存储之间进行权衡的技术，主要由三个部分组成：a) 指定某些层或模块进行重计算，b) 释放正向计算中的激活值，以及 c) 通过反向计算重计算依赖的激活值，从而显著减少训练过程中对 GPU 内存的使用。此外，考虑到 PCIe 带宽和主机内存大小，采用了基于层的激活卸载策略。在不降低训练性能的情况下，将 GPU 内存中的激活值卸载到主机内存，进一步节省 GPU 内存。

# 5.3.4 自动容错

在HunyuanVideo的大规模训练稳定性方面，采用自动故障容错机制以迅速恢复因常见硬件故障而中断的训练。这避免了手动恢复训练任务的频繁发生。通过自动检测错误并快速更换健康节点来恢复训练任务，训练稳定性达到了99.5%。

# 6 基础模型性能

文本对齐 视频生成模型的关键评估指标之一是其准确遵循文本提示的能力。这一能力对这些模型的有效性至关重要。然而，一些开源模型在捕捉所有主体或准确表示多个主体之间的关系时，常常表现不佳，尤其是在输入的文本提示比较复杂时。HunyuanVideo展示了强大的能力，能够生成与提供的文本提示紧密贴合的视频。如图12所示，它有效地处理了场景中的多个主体。

![](images/11.jpg)  

Figure 12: Prompt: A white cat sits on a white soft sofa like a person, while its long-haired male owner, with his hair tied up in a topknot, sits on the floor, gazing into the cat's eyes. His child stands nearby, observing the interaction between the cat and the man.

高质量 我们还执行微调过程，以提升生成视频的空间质量。如图13所示，HunyuanVideo能够生成具有超细节内容的视频。 高动态 在这一部分，我们展示HunyuanVideo基于给定提示生成高动态视频的能力。如图14所示，我们的模型在生成涵盖多种场景和各种运动类型的视频方面表现出色。 概念泛化 生成模型最理想的特性之一是其概念泛化能力。如图15所示，文本提示描述了一个场景：“在一个遥远的星系中，一个宇航员漂浮在一个闪烁的、粉色的像宝石一样的湖泊上，湖面反射着周围天空的绚丽色彩，形成令人惊艳的画面。宇航员轻轻漂浮在湖面上，水的低语声揭示着星球的秘密。他伸出手指，...（b）提示：一个时尚的女人自信而悠闲地走在一个充满温暖绚丽霓虹灯和动感城市景观的东京街道上。她佩戴着时尚的太阳镜，涂着红色的唇膏。街道潮湿而具有反光效果，色彩斑斓的灯光形成镜面效果。许多行人走来走去。

![](images/12.jpg)

![](images/13.jpg)  
(a) Prompt: the ultra-wide-angle lens follows closely from the hood, with raindrops continuously splattering aginst theenshead  sports car speearou a corer, is res violentl skiiagainst he we r, hee   

Figure 13: High-quality videos generated by HunyuanVideo.

![](images/14.jpg)

在日落时分，一辆改装的福特 F-150 猛禽在越野赛道上咆哮而过。升高的悬挂系统使得巨大的防爆轮胎能够在泥地上自由翻滚，泥浆溅到了防滚架上。

![](images/15.jpg)

引导性提示：镜头缓慢前移，中间聚焦点具有景深，温暖的日落光线洒满画面。画中的女孩裙摆飘动，奔跑、转身并跳起。

![](images/16.jpg)

在健身房，一名穿着运动服的女性在专业的跑步机上训练。

![](images/17.jpg)

游泳者在水下慢动作游泳。逼真的水下照明，宁静。

![](images/18.jpg)  

Figure 14: High-motion dynamics videos generated by HunyuanVideo.

在平静、光滑的水面上，现实、自然光照下，非正式的场景中。值得注意的是，这一特定场景在训练数据集中并未出现。此外，显然所描绘的场景结合了多个在训练数据中也缺失的概念。

![](images/19.jpg)  

Figure 15: HunyuanVideo's performance on concept generalization. The results of the three rows correspond to the text prompts (1) 'In a distant galaxy, an astronaut foats on a shimmering, pik, gemstone-like lak that re n color  the undiy, eat sseThet nyri lak' rae, teo   a hiper e plan et He ee uthisrti o hecol  watr.   Amae apture t hestrist playistrets (The night-blooming cactus fowers in the evening, with a bri, rapid closure.Time-lapse shot, extreme close-up. Realistic, Night lighting, Mysterious.' respectively.

行为推理与规划 通过利用大语言模型的能力，HunyuanVideo 可以根据提供的文本提示生成连续的动作。如图16所示，HunyuanVideo 能够以逼真的风格有效捕捉所有动作。

![](images/20.jpg)  

Figure 16: Prompt: The woman walks over and opens the red wooden door. As the door swings open, seawater bursts forth, in a realistic style.

字符理解与书写 HunyuanVideo 能够生成场景文本和逐渐显现的手写文本，如图 17 所示。

# 6.1 与最先进模型的比较

为了评估 HunyuanVideo 的性能，我们从闭源视频生成模型中选择了五个强基线。总共，我们使用了 1,533 个文本提示，生成了相同数量的视频样本，在一次运行中通过 HunyuanVideo 完成。为了公平比较，我们仅进行了一次推理，避免任何结果的选择性引用。在与基线方法比较时，我们保持所有选定模型的默认设置，以确保视频分辨率一致。60 位专业评估者进行了评估，结果见表 3。视频评估基于三个标准：文本对齐、运动质量和视觉质量。值得注意的是，HunyuanVideo 展现了最佳的整体性能，尤其在运动质量方面表现出色。我们随机抽取了 600 个视频供公众访问。

![](images/21.jpg)  

Figure 17: High text-video alignment videos generated by HunyuanVideo. Top row: Prompt: A close-up of a wave crashing against the beach, the sea foam spells out "WAKE UP" on the sand. Bottom row: Prompt: In a garden filled with blooming flowers, "GROW LOVE" has been spelled out with colorful petals.

Table 3: Model Performance Evaluation   

<table><tr><td>Model Name</td><td>Duration</td><td>Text Alignment</td><td>Motion Quality</td><td>Visual Quality</td><td>Overall</td><td>Ranking</td></tr><tr><td>HunyuanVideo (Ours)</td><td>5s</td><td>61.8%</td><td>66.5%</td><td>95.7%</td><td>41.3%</td><td>1</td></tr><tr><td>CNTopA (API)</td><td>5s</td><td>62.6%</td><td>61.7%</td><td>95.6%</td><td>37.7%</td><td>2</td></tr><tr><td>CNTopB (Web)</td><td>5s</td><td>60.1%</td><td>62.9%</td><td>97.7%</td><td>37.5%</td><td>3</td></tr><tr><td>GEN-3 alpha (Web)</td><td>6s</td><td>47.7%</td><td>54.7%</td><td>97.5%</td><td>27.4%</td><td>4</td></tr><tr><td>Luma1.6 (API)</td><td>5s</td><td>57.6%</td><td>44.2%</td><td>94.1%</td><td>24.8%</td><td>5</td></tr><tr><td>CNTopC (Web)</td><td>5s</td><td>48.4%</td><td>47.2%</td><td>96.3%</td><td>24.6%</td><td>6</td></tr></table>

# 7 应用案例

# 7.1 基于视频的音频生成

我们的视频到音频（V2A）模块旨在通过结合同步音效和适当的背景音乐，增强生成的视频内容。在传统的电影制作流程中，音效设计是一个不可或缺的部分，极大地增强了视觉媒体的听觉真实感和情感深度。然而，创建音效音频既耗时又需要高水平的专业知识。随着越来越多的文本到视频（T2V）模型的出现，大多数模型缺乏相应的音效生成能力，限制了它们生成完全沉浸式内容的能力。我们的V2A模块通过自主生成与输入视频和文本提示相匹配的电影级音效音频，填补了这一关键空白，从而实现了一个连贯且全面吸引人的多媒体体验的合成。

# 7.1.1 数据

与文本到视频（T2V）模型不同，视频到音频（V2A）模型对数据有不同的要求。如上所述，我们构建了一个包含视频-文本配对的视频数据集。然而，这个数据集中的并非所有数据都适合用于训练V2A模型。例如，有些视频缺少音频流，有些则包含大量解说内容，或者它们的环境音轨已被删除并替换为无关元素。为了解决这些挑战并确保数据质量，我们设计了一个专门针对V2A训练的稳健数据过滤管道。首先，我们过滤掉没有音频流的视频或那些静音比例超过$80\%$的视频。接下来，我们采用帧级音频检测模型，如[38]，来检测音频流中的语音、音乐和一般声音。根据这一分析，我们将数据分类为四个不同的类别：纯音、带有语音的声音、带有音乐的声音和纯音乐。随后，为了优先考虑高质量数据，我们训练了一个受到CAVP [54] 启发的模型，计算视觉-音频一致性评分，以量化每个视频的视觉和听觉组件之间的对齐程度。使用这个评分系统结合音频类别标签，我们系统地从每个类别中抽样数据，保留约250,000小时的原始数据集用于预训练。在监督微调阶段，我们进一步细化选择，策划出一个包含数百万个高质量片段的子集（80,000小时）。

![](images/22.jpg)  

Figure 18: The architecture of sound effect and music generation model.

对于特征提取，我们使用 CLIP [70] 在 4 fps 的时间分辨率下获取视觉特征，然后对这些特征进行重采样，以与音频帧率对齐。为了生成字幕，我们采用 [29] 作为声音字幕模型，采用 [20] 作为音乐字幕模型。当同时可用 sod 和 mui 字幕时，我们将它们合并为结构化的 captin 格式，遵循 [67] 中详细描述的方法。

# 7.1.2 模型

与上述文本转视频模型类似，我们的视频转音频生成模型也采用基于流匹配的扩散变换器（DiT）作为其架构主干。模型的详细设计如图18所示，展示了从三流结构到单流DiT框架的过渡。该模型在由变分自编码器（VAE）编码的潜在空间中运行，该VAE是基于梅尔谱（mel-spectrograms）训练的。具体而言，音频波形首先被转换为二维梅尔谱表示。此谱图随后使用预训练的VAE编码到潜在空间中。为了提取特征，我们利用预训练的CLIP和T5编码器分别独立提取视觉和文本特征。这些特征随后通过独立线性投影并经过SwiGLU激活投影到DiT兼容的潜在空间中，如图18所示。为了有效整合多模态信息，我们引入了堆叠的三流变换块，独立处理视觉、音频和文本模态。之后是单流变换块，以确保跨模态的无缝融合和对齐。这增强了音频-视频和音频-文本表示之间的对齐，提升了多模态一致性。一旦扩散变换器生成潜在表示，VAE解码器就会重建相应的梅尔谱。最后，梅尔谱通过预训练的HifiGAN声码器转换回音频波形。该框架确保高保真的音频信号重建，同时保持强多模态对齐。

# 7.2 浑源图像生成视频

# 7.2.1 预训练

![](images/23.jpg)  

Figure 19: Hunyuan Video-I2V Diffusion Backbone.

图像到视频（I2V）任务是视频生成任务中的一种常见应用。它通常意味着给定一幅图像和一个标题，模型使用该图像作为第一帧生成与标题匹配的视频。虽然简单的HunyuanVideo是一个文本到视频（T2V）模型，但它可以很容易地扩展为I2V模型。如图19所示，I2V模型采用了一种令牌替换技术，以帮助模型更准确地重建其输出中的原始图像信息。参考图像的潜变量被直接用作第一帧的潜变量，并将相应的时间步设置为0。其他帧潜变量的处理与T2V训练一致。为了增强模型理解输入图像语义的能力，并更有效地整合图像和标题的信息，I2V模型引入了语义图像注入模块。该模块首先将图像输入MLLM模型以获得语义图像令牌，然后将这些令牌与视频潜变量令牌连接起来以进行全注意力计算。我们在与T2V模型相同的数据上对I2V模型进行预训练，结果见图20。

# 7.2.2 下游任务微调：肖像图像到视频生成

我们对I2V模型进行了监督微调，使用了两百万个肖像视频，以增强人的动作和整体美感。除了第3节中描述的标准数据过滤流程，我们还应用了人脸和身体检测器，过滤掉包含超过五个人的训练视频。我们还移除了主要主体较小的视频。最终，其余视频将经过人工检查，以获得最终的高质量肖像训练数据集。在训练方面，我们采用了渐进式微调策略，逐步解冻各层的模型参数，同时保持其余参数在微调过程中冻结。这种方法使得模型在肖像领域能够获得高性能，同时不大程度上妥协其固有的泛化能力，确保在自然风景、动物和植物领域的表现也令人满意。此外，我们的模型还支持视频插值，使用第一帧和最后一帧作为条件。我们在训练过程中以一定概率随机丢弃文本条件，以提升模型性能。一些示例结果如图21所示。

![](images/24.jpg)  

Figure 20: Sample results of the I2V pre-training model.

![](images/25.jpg)  

Figure 21: Sample results of our portrait I2V model.

# 7.3 角色动画

HunyuanVideo 在多个方面赋予可控的头像动画。它能够利用明确的驱动信号（例如，语音信号、表情模板和姿势模板）来对角色进行动画处理。此外，它还通过文本提示整合了隐式驱动范式。图 22 展示了我们如何利用 HunyuanVideo 的能力从多模态条件中对角色进行动画处理。为了保持严格的外观一致性，我们通过插入参考图像的潜在表示作为强指导来修改 HunyuanVideo 架构。如图 22 （b, c）所示，我们使用 3DVAE 对参考图像进行编码，得到 $z _ { \mathrm { r e f } } \in \mathbb { R } ^ { 1 \times c \times h \times w }$，其中 $c = 16$。然后我们在时间维度上重复它 $t$ 次，并在通道维度上与 $z _ { t }$ 进行串联，得到修改后的噪声输入 $\hat { z } _ { t } \in \mathbb { R } ^ { t \times 2 c \times h \times w }$。

![](images/26.jpg)  

Figure 22: Overview of Avatar Animation built on top of HunyuanVideo. We adopt 3D VAE to encode and inject reference and pose condition, and use additional cross-attention layers to inject audio and expression signals. Masks are employed to explicitly guide where they are affecting.

# 7.3.1 上半身对话虚拟形象生成

近年来，基于音频驱动的数字人算法取得了显著进展，尤其是在对话头的表现方面。早期算法，如loopy [94]、emo [80]和hallo [87]，主要集中在头部区域，通过分析音频信号来驱动数字人面部表情和嘴唇形状。甚至更早的算法，例如wav2lip [68]和DINet [97]，专注于修改输入视频中的嘴部区域，以实现与音频一致的嘴唇形状。然而，这些算法通常局限于头部区域，忽视了身体的其他部分。为了实现更自然、生动的数字人表现，我们提出了一种扩展至上半身的音频驱动算法。在该算法中，数字人在说话时不仅能够将面部表情和嘴唇形状与音频同步，还能随着音频的节奏有节奏地移动身体。

基于音频驱动 根据输入的音频信号，我们的模型能够自适应地预测数字人类的面部表情和姿态动作信息。这使得驱动角色能够带有情感和表情地进行对话，增强了数字人类的表现力和真实感。如图 22 (b) 所示，对于单一音频信号驱动的部分，音频经过低语特征提取模块，以获取音频特征，然后以交叉注意力的方式注入主网络。需要注意的是，注入过程将乘以面部遮罩，以控制音频的影响区域。在增强头部和肩部控制能力的同时，也将大大减少身体变形的概率。为了获得更生动的头部运动，引入并以嵌入方式添加头部姿态运动参数和表情运动参数到时间步中。在训练过程中，头部运动参数由鼻尖关键点序列的方差给出，表情参数则由面部关键点的方差给出。

# 7.3.2 完全可控的全身虚拟形象生成

显式控制数字角色的运动和表情一直是学术界和工业界的一个长期问题，而扩散模型的最新进展为真实化头像动画迈出了第一步。然而，目前的头像动画解决方案由于基础视频生成模型的能力有限，存在部分可控性的问题。我们展示了一个更强大的T2V模型如何将头像视频生成提升到完全可控的阶段。我们展示了如何通过有限的修改，使Hunyuan Video作为强大的基础，使通用T2V模型扩展到完全可控的头像生成模型，如图22(c)所示。 姿势驱动 我们可以使用姿势模板显式控制数字角色的身体运动。我们使用Dwpose [92] 从任何源视频中检测骨骼视频，并使用3DVAE将其转换到潜在空间作为$z _ { \mathrm { p o s e } }$。我们认为这简化了微调过程，因为输入和驱动视频均为图像表示，并且使用共享VAE编码，从而导致相同的潜在空间。然后，我们通过逐元素相加的方式将驱动信号注入模型，作为$\hat { z } _ { t } + z _ { \mathrm { p o s e } }$。请注意，$\hat { z } _ { t }$包含参考图像的外观信息。我们使用预训练的T2V权重作为初始化进行全参数微调。

![](images/27.jpg)  

Figure 23: Audio-Driven. HunyuanVideo can generate vivid talking avatar videos.

表达驱动 我们还可以通过隐式表达表示来控制数字角色的面部表情。尽管面部关键点在这一领域被广泛采用，但我们认为使用关键点会导致由于跨身份对齐不一致而泄露身份。相反，我们使用隐式表示作为驱动信号，利用其对身份和表情解耦的能力。在本研究中，我们使用 VASA 作为表情提取器。如图 22 (c) 所示，我们采用轻量级表情编码器，将表情表示转换为潜在空间中的词元序列，表示为 $\bar { z _ { \mathrm { e x p } } } \in \mathbb { R } ^ { t \times n \times c }$，其中 $n$ 是每帧的词元数量。通常，我们设定 $n = 1 6$。与姿态条件不同，我们通过跨注意力注入 $z _ { \mathrm { e x p } }$，因为 $\hat { z } _ { t }$ 和 $z _ { \mathrm { e x p } }$ 在空间上并不自然对齐。我们在每 $K$ 层双流和单流 DiT 层中添加跨注意力层 $\mathrm { A t t n } _ { \mathrm { e x p } } ( q , k , v )$ 来注入表情潜在表示。设第 $i$ 层 DiT 的隐藏状态为 $h _ { i }$，则表情 $z _ { \mathrm { e x p } }$ 注入 $h _ { i }$ 的过程可以表示为：$h _ { i } + \mathrm { A t t n } _ { \mathrm { e x p } } ( h _ { i } , z _ { \mathrm { e x p } } , z _ { \mathrm { e x p } } ) \ast \mathcal { M } _ { \mathrm { f a c e } }$，其中 $\mathcal { M } _ { \mathrm { f a c e } }$ 是面部区域掩码，指引 $z _ { \mathrm { e x p } }$ 应该应用的位置，$^ *$ 表示逐元素乘法。此外，采用了全参数微调策略。 混合条件驱动 结合姿态和表情驱动策略产生混合控制方法。在这种情况下，身体运动由显式骨骼姿态序列控制，面部表情则由隐式表达表示确定。我们将 T2V 模块和添加的模块进行联合端到端微调。在推理过程中，身体运动和面部运动可以由独立的驱动信号控制，从而增强更丰富的可编辑性。

# 7.4 应用示例

我们提供了丰富的虚拟角色动画结果，展示了由HunyuanVideo驱动的虚拟角色动画在下一代中的优势和潜力。

![](images/28.jpg)  

Figure 24: Pose-Driven. HunyuanVideo can animate wide variety of characters with high quality and appearance consistency under various poses.

![](images/29.jpg)  

Figure 25: Expression-Driven. HunyuanVideo can accurately control facial movements of widevariety of avatar styles.

音频驱动 图23显示，HunyuanVideo作为音频驱动头像动画的强大基础模型，可以合成生动且高保真的视频。我们将方法的优势总结为三个方面： • 上半身动画。我们的方法不仅可以驱动肖像角色，还可以驱动上半身头像图像，扩展了应用场景的范围。 • 动态场景建模。我们的方法可以生成具有生动且逼真的背景动态的视频，如波浪起伏、人群移动和微风拂动树叶。 • 生动的头像动作。我们的方法能够仅凭音频驱动角色说话的同时进行生动的手势动画。 姿态驱动 我们还展示了HunyuanVideo在姿态驱动动画的多个方面显著提升了性能，如图24所示：

![](images/30.jpg)  

Figure 26: Hybrid Condition-Driven. Hunyuan Video supports full control with multiple driving sources across various avatar characters.

• 高 ID 一致性。我们的方法在帧之间很好地保持了 ID 一致性，即使在大幅度姿态下也不需要交换人脸，因此可以作为真实的端到端动画解决方案。 • 准确跟随复杂姿势。我们的方法能够处理非常复杂的姿势，例如转身和双手交叉。 • 高运动质量。我们的方法在动态建模方面表现出显著的能力。例如，结果在服装动态和纹理一致性方面表现出良好的性能。 • 泛化能力。我们的方法展现出惊人的高泛化能力。它可以对各种头像图像进行动画，如真实人类、动画、陶瓷雕像，甚至动物。 • 表情驱动。图 25 展示了 HunyuanVideo 如何在三个方面增强肖像表情动画：夸张表情。我们的方法能够对给定的肖像进行动画，以模仿任何面部动作，即使在大幅度姿势和夸张表情下。 • 准确模拟眼球注视。我们能够在给定任何表情模板的情况下，准确控制肖像的眼动，即使在极端和大幅度的眼球运动中。 • 泛化能力。我们的方法具有高泛化能力。它不仅可以对真实人类肖像进行动画，还可以对动画或 CGI 角色进行动画。 • 混合驱动。最后，我们展示了混合条件控制揭示了可完全控件和可编辑的头像的潜力，如图 26 所示。我们强调其优越性如下： • 混合条件控制。我们首次能够通过独立或多个信号全面控制身体和面部运动，为头像动画的演示到应用铺平了道路。 • 半身动画。我们的方法支持上半身的全面控制，实现丰富的可编辑性，同时保持高质量和真诚度。 • 泛化能力。我们的方法能够对真实人类图像和 CGI 角色进行泛化。

# 8 相关工作

由于扩散模型在图像生成领域的成功，视频生成领域的探索也变得越来越受欢迎。VDM是首批将图像扩散模型中的2D U-Net扩展为3D U-Net以实现基于文本生成的模型之一。后续工作如MagicVideo和Mindscope引入了一维时间注意机制，通过在潜在扩散模型的基础上减少计算量。在本报告中，我们不采用$ { 2 \mathrm { D } } + { 1 \mathrm { D } }$时间块的方式进行动作学习。相反，我们使用类似于FLUX中的双流注意块，用于处理所有视频帧。继Imagen之后，Imagen Video采用级联采样管道，通过多个阶段生成视频。除了传统的端到端文本到视频(T2V)生成外，使用其他条件的视频生成也是一个重要方向。这类方法使用其他辅助控制生成视频，如深度图、姿态图、RGB图像或其他引导运动视频。尽管近期的开源模型如稳定视频扩散、Open-sora、Open-sora-plan、Mochi-1和Allegro的生成性能非常出色，但它们的表现仍然远远落后于闭源的最先进视频生成模型如Sora和MovieGen。

# 项目贡献者

项目赞助人：江杰，刘宇鸿，王迪，杨勇 项目负责人：钟凯撒，王洪发，周达，刘松涛，陆清林，陶阳宇 核心贡献者： 基础设施：罗克斯·敏，薛金宝，彭元波，杨芳，李帅，王维延，王凯 数据与重标注：代作卓，李欣，周金，袁俊琨，谭昊，邓新驰，何志宇，黄多军，王安东，刘梦扬，李鹏宇 - VAE与模型蒸馏：吴博，罗克斯·敏，李长林，白家旺，李杨，吴建兵 算法与模型架构及预训练：孔韦杰，田奇，张剑伟，张子健，吴凯瑟，熊江峰，龙燕欣 下游任务：宋雅各，周金，崔宇涛，王阿拉丁，余文青，徐志勇，周子翔，于振涛，陈毅，王红梅，徐遵南，王乔伊，林琴 •贡献者：张继洪，陈萌，朱建晨，胡温斯顿，饶永明，刘凯，许丽飞，林思环，孙怡夫，黄士瑞，牛林，黄世生，邓永俊，曹开博，杨轩，张昊，林嘉欣，张超，游飞，陈元斌，胡与辉，郑亮东，方奕，焦点，徐志强，任旭华，马冰，程家相，李文越，余凯，郑天翔

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat 等. Gpt-4 技术报告. arXiv 预印本 arXiv:2303.08774, 2023. 9 [2] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen 等. Palm 2 技术报告. arXiv 预印本 arXiv:2305.10403, 2023. 9 [3] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, 和 Jun Zhu. 所有都值得词汇：用于扩散模型的 vit 主干网络. 在 IEEE/CVF 计算机视觉与模式识别会议论文集中，第 22669-22679 页, 2023. 9 [4] Jmes Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo 等. 通过更好的标题改善图像生成. 计算机科学. https://cdn.openai.com/papers/dall-e-3.pdf, 2(3):8, 2023. 4 [5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts 等. 稳定视频扩散：将潜在视频扩散模型扩展至大型数据集. arXiv 预印本, 2023. 2, 27 [6] Andrew Brock, Jeff Donahue 和 Karen Simonyan. 大规模生成对抗网络训练用于高保真自然图像合成. arXiv 预印本 arXiv:1809.11096, 2018. 2 [7] Tim Brooks, Bill Peebles, Connor Homes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Wing Yin Ng, Ricky Wang 和 Aditya Ramesh. 视频生成模型作为世界模拟器. 2024. 4, 7, 27 [8] Tom B Brown. 语言模型是少样本学习者. arXiv 预印本 arXiv:2005.14165, 2020. 8 [9] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao 和 Dahua Lin. Sharegpt4v：通过更好的标题改善大型多模态模型. arXiv 预印本 arXiv:2311.12793, 2023. 4 [10] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang 等. Sharegpt4video：通过更好的标题改善视频理解和生成. arXiv 预印本 arXiv:2406.04325, 2024. 4 [11] Liuhan Chen, Zongjian Li, Bin Lin, Bin Zhu, Qian Wang, Shenghai Yuan, Xing Zhou, Xinghua Cheng 和 Li Yuan. Od-vae：一种全维度视频压缩器，用于改善潜在视频扩散模型. arXiv 预印本 arXiv:2409.01199, 2024. 6 [12] Qihua Chen, Yue Ma, Hongfa Wang, Junkun Yuan, Wenzhe Zhao, Qi Tian, Hongmei Wang, Shaobo Min, Qifeng Chen 和 Wei Liu. Follow-your-canvas：通过广泛内容生成实现更高分辨率视频外延. arXiv 预印本 arXiv:2409.01055, 2024. 27 [14] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-Wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang 和 Sergey Tulyakov. Panda $\cdot 7 0 \mathrm { m }$ : 使用多个跨模态教师对 $7 0 \mathrm { m }$ 视频进行标注. 在 2024 年 IEEE/CVF 计算机视觉与模式识别会议 (CVPR) 上，第 13320-13331 页. IEEE, 2024 年 6 月. 4 [15] Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao 和 Ziwei Liu. Seine：用于生成过渡和预测的短至长视频扩散模型. arXiv 预印本, 2023. 27 [16] Zhiyuan Chen, Jiajiong Cao, Zhiquan Chen, Yuming Li 和 Chenguang Ma. Echomimic：通过可编辑的地标条件实现逼真的音频驱动的人物动画. arXiv 预印本 arXiv:2407.08136, 2024. 23 [17] XTuner Contributors. Xtuner：有效微调大语言模型的工具包. https://github.com/InternLM/xtuner, 2023. 8 [18] OpenCV 开发者. OpenCV. https://opencv.org/. 3 [19] PySceneDetect 开发者. Pyscenedetect. https://www.scenedetect.com/. 3 [20] SeungHeon Doh, Keunwoo Choi, Jongpil Lee 和 Juhan Nam. Lp-musiccaps：基于大语言模型的伪音乐标注. arXiv 预印本 arXiv:2307.16372, 2023. 19 [21] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel 等. 扩展整流流变换器以用于高分辨率图像合成. 在第四十一届国际机器学习会议上, 2024. 2, 8, 9, 10 [22] Patrick Esser, Robin Rombach 和 Bjorn Ommer. 驯化变换器用于高分辨率图像合成. 在 IEEE/CVF 计算机视觉与模式识别会议论文集中，第 12873-12883 页, 2021. 6 [23] Peng Gao, Le Zhuo, Ziyi Lin, Chris Liu, Junsong Chen, Ruoyi Du, Enze Xie, Xu Luo, Longtian Qiu, Yuhang Zhang 等. Lumina-t2x：通过基于流的巨大扩散变换器将文本转化为任意模态、分辨率和持续时间. arXiv 预印本 arXiv:2405.05945, 2024. 9 [24] Z Ge. Yolox：超越 YOLO 系列的 2021 年. arXiv 预印本 arXiv:2107.08430, 2021. 3 [25] Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh Rambhatla, Akbar Shah, Xi Yin, Devi Parikh 和 Ishan Misra. Emu 视频：通过显式图像条件化分解文本到视频生成. arXiv 预印本 arXiv:2311.10709, 2023. 2

[26] Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, Hao Yu, Hongning Wang, Jiadai Sun, Jiajie Zhang, Jiale Cheng, Jiayi Gui, Jie Tang, Jing Zhang, Juanzi Li, Lei Zhao, Lindong Wu, Lucen Zhong, Mingdao Liu, Minlie Huang, Peng Zhang, Qinkai Zheng, Rui Lu, Shuaiqi Duan, Shudan Zhang, Shulin Cao, Shuxun Yang, Weng Lam Tam, Wenyi Zhao, Xiao Liu, Xiao Xia, Xiaohan Zhang, Xiaotao Gu, Xin Lv, Xinghan Liu, Xinyi Liu, Xinyue Yang, Xixuan Song, Xunkai Zhang, Yifan An, Yifan Xu, Yilin Niu, Yuantao Yang, Yueyan Li, Yushi Bai, Yuxiao Dong, Zehan Qi, Zhaoyu Wang, Zhen Yang, Zhengxiao Du, Zhenyu Hou, 和 Zihan Wang. Chatglm：一个从 glm-130b 到 glm-4 的大型语言模型家族，包括所有工具，2024。 [27] Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, 和 Bo Dai. Sparsectrl：向文本到视频扩散模型添加稀疏控制。arXiv 预印本，2023。 [28] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, 和 Bo Dai. Animatediff：无特定调优的个性化文本到图像扩散模型动画。ICLR，2024。 [29] Moayed Haji-Ali, Willi Menapace, Aliaksandr Siarohin, Guha Balakrishnan, Sergey Tulyakov, 和 Vicente Ordonez. 驯服数据和变换器以生成音频。arXiv 预印本 arXiv:2406.19388，2024。 [30] Pieter Abbeel, Hao Liu, 和 Matei Zaharia. 区域注意力与块状变换器用于近乎无限的上下文。arXiv 预印本 arXiv:2310.01889，2023。 [31] Yingqing He, Menghan Xia, Haoxin Chen, Xiaodong Cun, Yuan Gong, Jinbo Xing, Yong Zhang, Xintao Wang, Chao Weng, Ying Shan, 等人。Animate-a-story：通过检索增强的视频生成讲故事。arXiv 预印本，2023。 [32] J Ho, T Salimans, A Gritsenko, W Chan, M Norouzi, 和 DJ Fleet. 视频扩散模型。arXiv 2022。arXiv 预印本，2022。 [3] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, 等人。Imagen video：使用扩散模型生成高分辨率视频。arXiv 预印本，2022。 [34] Jonathan Ho, Ajay Jain, 和 Pieter Abbeel. 去噪扩散概率模型。在 NeurIPS，2020。 [35] Jonathan Ho 和 Tim Salimans. 无分类器扩散引导。arXiv 预印本，2022。 [36] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, 等人。训练计算最优的大型语言模型。arXiv 预印本 arXiv:2203.15556，2022。 [37] Li Hu, Xin Gao, Peng Zhang, Ke Sun, Bang Zhang, 和 Liefeng Bo. Animate anyone：一致且可控的人物动画图像到视频合成。arXiv 预印本，2023。 [38] Yun-Ning Hung, Chih-Wei Wu, Iroro Orife, Aaron Hipple, William Wolcott, 和 Alexander Lerch. 一个用于语音和音乐活动检测的大型电视数据集。EURASIP 音频、语音和音乐处理杂志，2022(1)：21，2022。 [39] Investopedia. 通用数据保护条例（gdpr），无日期。访问于2023年10月10日。 [40] Yuming Jiang, Shuai Yang, Tong Liang Koh, Wayne Wu, Chen Change Loy, 和 Ziwei Liu. Text2performer：文本驱动的人体视频生成。arXiv 预印本，2023。 [41] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, 和 Dario Amodei. 神经语言模型的扩展法则。arXiv 预印本 arXiv:2001.08361，2020。 [42] Maciej Kilian, Varun Japan, 和 Luke Zettlemoyer. 图像合成中的计算权衡：扩散、屏蔽标记和下一个标记预测。arXiv 预印本 arXiv:2405.13218，2024。 [43] Juyeon Kim, Jeongeun Lee, Yoonho Chang, Chanyeol Choi, Junseong Kim, 和 Jy yong Sohn. Re-ex：解释后修正减少了 LLM 响应中的事实错误，2024。 [44] Jungil Kong, Jaehyeon Kim, 和 Jaekyoung Bae. Hifi-gan：用于高效和高保真语音合成的生成对抗网络。神经信息处理系统进展，33：1702217033，2020。 [45] Vijay Anand Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, 和 Bryan Catanzaro. 减少大型变换器模型中的激活重计算。机器学习与系统会议论文，5：341353，2023。 [46] PKU-Yuan Lab 和 Tuzhan AI 等。Open-sora-plan，2024年4月。 [47] Black Forest Labs. Flux，2024。 [48] Baojia Li, Xiaoliang Wang, Jingzhu Wang, Yifan Liu, Yuanyuan Gong, Hao Lu, Weizhen Dang, Weifeng Zhang, Xiaojie Huang, Mingzhuo Chen, 等。Tccl：为以 GPU 为中心的集群共同优化集体通信和流量路由。在2024 SIGCOMM人工智能计算网络研讨会的论文中，页面4853，2024。 [49] Hao Li, Yang Zou, Ying Wang, Orchid Majumder, Yusheng Xie, R Manmatha, Ashwin Swaminathan, Zhuowen Tu, Stefano Ermon, 和 Stefano Soatto. 关于基于扩散的文本到图像生成的可扩展性。IEEE/CVF 计算机视觉与模式识别会议论文，页面94009409，2024。 [50] Junnan Li, Dongxu Li, Caiming Xiong, 和 Steven Hoi. Blip：用于统一视觉语言理解和生成的语言图像预训练引导。机器学习国际会议论文，页面1288812900。PMLR，2022。 [51] Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, Dayou Chen, Jiajun He, Jiahao Li, Wenyue Li, Chen Zhang, Rongwei Quan, Jianxiang Lu, Jiabin Huang, Xiaoyan Yuan, Xiaoxiao Zheng, Yixuan Li, Jihong Zhang, Chao Zhang, Meng Chen, Jie Liu, Zheng Fang, Weiyan Wang, Jinbao Xue, Yangyu Tao, Jianchen Zhu, Kai Liu, Sihuan Lin, Yifu Sun, Yun Li, Dongdong Wang, Mingtao Chen, Zhichao Hu, Xiao Xiao, Yan Chen, Yuhong Liu, Wei Liu, Di Wang, Yong Yang, Jie Jiang, 和 Qinglin Lu. Hunyuan-dit：具有细粒度中文理解的强大多分辨率扩散变换器，2024。 [52] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, 和 Matt Le. 流匹配用于生成建模。arXiv 预印本 arXiv:2210.02747，2022。 [53] Haotian Liu, Chunyuan Li, Qingyang Wu, 和 Yong Jae Lee. 视觉指令调优。神经信息处理系统进展，36，2024。 [54] Simian Luo, Chuanhao Yan, Chenxu Hu, 和 Hang Zhao. Diff-foley：使用潜在扩散模型同步视频到音频合成。神经信息处理系统进展，36，2024。 [55] Bingqi Ma, Zhuofan Zong, Guanglu Song, Hongsheng Li, 和 Yu Liu. 探索大型语言模型在扩散模型的提示编码中的作用。arXiv 预印本 arXiv:2406.11831，2024。 [56] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Ying Shan, Xiu Li, 和 Qifeng Chen. 跟随你的姿势：使用无姿态视频的姿势引导文本到视频生成。arXiv 预印本，2023。 [57] Yue Ma, Yingqing He, Hongfa Wang, Andong Wang, Chenyang Qi, Chengfei Cai, Xiu Li, Zhifeng Li, Heung-Yeung Shum, Wei Liu, 等。跟随你的点击：通过短提示实现开放域区域图像动画。arXiv 预印本 arXiv:2403.08268，2024。 [58] Yue Ma, Hongyu Liu, Hongfa Wang, Heng Pan, Yingqing He, Junkun Yuan, Ailing Zeng, Chengfei Cai, Heung-Yeung Shum, Wei Liu, 等。跟随你的表情：细致可控和富有表现力的自由风格肖像动画。arXiv 预印本 arXiv:2406.01900，2024。 [59] J MacQueen. 一些分类和多变量观察分析的方法。在第五届伯克利数学统计与概率研讨会论文集中，1967。 [60] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, 和 Tim Salimans. 关于引导扩散模型的蒸馏。在IEEE/CVF计算机视觉与模式识别会议论文中，页面1429714306，2023。 [61] Haomiao Ni, Changhao Shi, Kai Li, Sharon X Huang, 和 Martin Renqiang Min. 使用潜在流扩散模型进行条件图像到视频生成。在CVPR，2023。 [62] Xiaonan Nie, Yi Liu, Fangcheng Fu, Jinbao Xue, Dian Jiao, Xupeng Miao, Yangyu Tao, 和 Bin Cui. Angel-ptm：腾讯中可扩展和经济的大规模预训练系统。arXiv 预印本 arXiv:2303.02868，2023。 [63] NVIDIA. 上下文并行性概述。2024。 [64] NVIDIA. Cosmos-tokenizer，2024。 [65] William Peebles 和 Saining Xie. 使用变换器的可扩展扩散模型。在IEEE/CVF国际计算机视觉会议论文中，页面41954205，2023。 [66] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, 和 Robin Rombach. Sdxl：改善高分辨率图像合成的潜在扩散模型。arXiv 预印本，2023。 [67] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, 等。Movie gen：媒体基础模型阵容。arXiv 预印本 arXiv:2410.13720，2024。 [68] KR Prajwal, Rudrabha Mukhopadhyay, Vinay P Namboodiri, 和 CV Jawahar. 唇同步专家是野外语音到唇生成所需的一切。在第28届ACM国际多媒体会议中，页面484492，2020。 [69] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, 等。通过自然语言监督学习可转移的视觉模型。在ICML，2021。 [70] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, 等。通过自然语言监督学习可转移的视觉模型。在ICML，页面87488763。PMLR，2021。 [71] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, 和 Peter J Liu. 探索统一文本到文本变换器的迁移学习极限。机器学习研究杂志，21(140)：167，2020。 [72] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, 和 Björn Ommer. 使用潜在扩散模型生成高分辨率图像。在CVPR，2022。 [73] Tim Salimans 和 Jonathan Ho. 渐进蒸馏以快速采样扩散模型。arXiv 预印本 arXiv:2202.00512，2022。 [74] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, 和 Bryan Catanzaro. Megatron-lm：使用模型并行性训练数十亿参数的语言模型。arXiv 预印本 arXiv:1909.08053，2019。 [75] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, 等。Make-a-video：无文本视频数据的文本到视频生成。arXiv 预印本，2022。 [76] Tomá Souek 和 Jakub Loko. Transnet v2：快速镜头过渡检测的有效深度网络架构。arXiv 预印本 arXiv:2008.04838，2020。 [77] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, 和 Yunfeng Liu. Roformer：增强旋转位置信息嵌入的变换器，2023。

[78] 孙兴武、陈彦锋、黄怡晴、谢若冰、朱佳奇、张凯、李帅鹏、杨振、韩乔尼、舒晓波、付家豪、陈中志、黄雪梅、连丰宗、杨赛勇、严剑锋、曾宇元、任小琴、余超、吴璐璐、毛悦、杨涛、郑孙聪、吴侃、焦电、薛金宝、张熙鹏、吴德成、刘凯、吴登鹏、徐光辉、陈少华、陈双、冯晓、洪亦耕、郑俊强、许承程、李宗伟、邝雄、胡江璐、陈怡琦、邓宇驰、李贵阳、刘傲、张晨辰、胡世辉、赵子龙、吴自帆、丁瑶、王维超、刘汉、王罗伯特、费昊、佘沛杰、赵泽、曹勋、王海、向福生、黄梦元、熊志远、胡彬、侯学斌、姜雷、吴佳佳、邓亚平、沈依、王乾、刘煨杰、刘杰、陈梦、董亮、贾维文、陈胡、刘飞飞、袁锐、徐慧琳、严振翔、曹腾飞、胡志超、冯新华、杜东、佘婷浩、陶扬昱、张锋、朱剑琛、徐承中、李熙锐、查冲、欧阳文、夏尹本、李翔、何泽坤、陈容鹏、宋家伟、陈瑞斌、姜凡、赵重庆、王博、龚昊、甘荣、胡温斯顿、康战辉、杨勇、刘玉虹、王迪、姜杰。Hunyuan-large：腾讯开发的一个开源moe模型，激活参数达到520亿，2024。8, 11 [79] Genmo团队。Mochi 1：开源视频生成的新最先进技术。https://github.com/genmoai/models, 2024. 7, 27 [80] 田林锐、王琦、张邦和博磊峰。Emo：在弱条件下使用音频到视频扩散模型生成富有表现力的肖像视频，2024。22 [81] Hugo Touvron、Thibaut Lavril、Gautier Izacard、Xavier Martinet、Marie-Anne Lachaux、Timothée Lacroix、Baptiste Rozière、Naman Goyal、Eric Hambro、Faisal Azhar等。Llama：开放且高效的基础语言模型。arXiv预印本 arXiv:2302.13971，2023。9 [82] 王久牛、袁航杰、陈大有、张莹雅、王翔、张世伟。Modelscope文本到视频技术报告。arXiv预印本，2023。27

[83] Tan Wang, Linjie Li, Kevin Lin, Chung-Ching Lin, Zhengyuan Yang, Hanwang Zhang, Zicheng Liu, 和 Lijuan Wang. Disco: 解耦控制在现实世界中生成参考人类舞蹈. arXiv 预印本, 2023. 27 [84] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, 等. Lavie: 基于级联潜在扩散模型的高质量视频生成. arXiv 预印本, 2023. 27 [85] Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen, Jingwen Hou, Annan Wang, Wenxiu Sun, Qiong Yan, 和 Weisi Lin. 从美学和技术角度探索用户生成内容的视频质量评估. 载于《IEEE/CVF计算机视觉国际会议论文集》，页码 2014420154, 2023. 3 [86] Ruiqi Wu, Liangyu Chen, Tong Yang, Chunle Guo, Chongyi Li, 和 Xiangyu Zhang. Lamp: 学习用于少量样本的视频生成的运动模式. arXiv 预印本, 2023. 27 [87] Mingwang Xu, Hui Li, Qingkun Su, Hanlin Shang, Liwei Zhang, Ce Liu, Jingdong Wang, Yao Yao, 和 Siyu Zhu. Hallo: 基于音频驱动的肖像图像动画的层次合成, 2024. 22 [88] Sicheng Xu, Guojun Chen, Yu-Xiao Guo, Jiaolong Yang, Chong Li, Zhenyu Zang, Yizhong Zhang, Xin Tong, 和 Baining Guo. Vasa-1: 实时生成栩栩如生的音频驱动人脸. arXiv 预印本 arXiv:2404.10667, 2024. 23 [89] Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, 和 Mike Zheng Shou. Magicanimate: 使用扩散模型进行时间一致的人像动画. arXiv 预印本, 2023. 27 [90] Jingyun Xue, Hongfa Wang, Qi Tian, Yue Ma, Andong Wang, Zhiyuan Zhao, Shaobo Min, Wenzhe Zhao, Kaihao Zhang, Heung-Yeung Shum, 等. Follow-your-pose v2: 多条件指导的角色图像动画以实现稳定的姿态控制. arXiv 预印本 arXiv:2406.03035, 2024. 27 [91] Mengjiao Yang, Yilun Du, Bo Dai, Dale Schuurmans, Joshua B Tenenbaum, 和 Pieter Abbeel. 文本到视频模型的概率适应. arXiv 预印本, 2023. 27 [92] Zhendong Yang, Ailing Zeng, Chun Yuan, 和 Yu Li. 基于两阶段蒸馏的有效全身姿态估计. 载于《IEEE/CVF计算机视觉国际会议论文集》，页码 42104220, 2023. 22 [93] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, 等. Cogvideox: 带有专家变换器的文本到视频扩散模型. arXiv 预印本 arXiv:2408.06072, 2024. 4, 5, 6, 7 [94] Zhenhui Ye, Tianyun Zhong, Yi Ren, Ziyue Jiang, Jiawei Huang, Rongjie Huang, Jinglin Liu, Jinzheng He, Chen Zhang, Zehan Wang, Xize Chen, Xiang Yin, 和 Zhou Zhao. Mimictalk: 在几分钟内模拟个性化和富有表现力的3D谈话面孔, 2024. 22 [95] Lijun Yu, José Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, 等. 语言模型战胜扩散，标记器是视觉生成的关键. arXiv 预印本 arXiv:2310.05737, 2023. 5 [96] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, 和 Mike Zheng Shou. Show-1: 将像素与潜在扩散模型结合用于文本到视频生成. arXiv 预印本, 2023. 27 [97] Zhimeng Zhang, Zhipeng Hu, Wenjin Deng, Changjie Fan, Tangjie Lv, 和 Yu Ding. Dinet: 用于高分辨率视频真实面部视觉配音的变形修补网络. 载于《AAAI人工智能会议论文集》，第37卷，页码 35433551, 2023. 22 [98] Zijian Zhang, Zhou Zhao, 和 Zhijie Lin. 从预训练的扩散概率模型中进行无监督表示学习. 神经信息处理系统进展, 35:2211722130, 2022. 13 [99] Zijian Zhang, Zhou Zhao, Jun Yu, 和 Qi Tian. Shiftddpms: 通过改变扩散轨迹探索条件扩散模型. 载于《AAAI人工智能会议论文集》，第37卷，页码 35523560, 2023. 13 [100] Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jiawei Liu, Weijia Wu, Jussi Keppo, 和 Mike Zheng Shou. Motiondirector: 文本到视频扩散模型的运动自定义. arXiv 预印本, 2023. 27 [101] Xuanlei Zhao, Xiaolong Jin, Kai Wang, 和 Yang You. 基于金字塔注意力广播的实时视频生成. arXiv 预印本 arXiv:2408.12588, 2024. 13 [102] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, 和 Yang You. Open-sora: 让所有人都能使用高效视频制作, 2024年3月. 6, 27 [103] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, 和 Jiashi Feng. Magicvideo: 用潜在扩散模型高效生成视频. arXiv 预印本, 2023. 27 [104] Yuan Zhou, Qiuyue Wang, Yuxuan Cai, 和 Huan Yang. Allegro: 打开商业级视频生成模型的黑箱. arXiv 预印本 arXiv:2410.15458, 2024. 6, 27