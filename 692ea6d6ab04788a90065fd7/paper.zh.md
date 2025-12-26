# COgVIdEOX：带有专家变换器的文本到视频扩散模型

杨卓懿\*‡ 滕嘉晏\*‡ 郑文迪‡ 丁鸣† 黄士煜† 贾睿$\mathbf { X } \mathbf { \widetilde { u } } ^ { \ddag }$ 杨圆铭‡ 洪文怡‡ 张晓涵† 冯贯宇† $\mathbf { Y i n } ^ { \dagger }$ 张宇轩† 王伟汉† 程一言† 孙斌$\mathbf { X } \mathbf { u } ^ { \ddag }$ 顾晓涛† 董宇潇 唐杰‡ ‡ 清华大学 † 智谱AI

![](images/1.jpg)  

Figure 1: CogVideoX can generate long-duration, high-resolution videos with coherent actions and rich semantics.

# 摘要

我们推出了CogVideoX，这是一个基于扩散变换器的大规模文本到视频生成模型，能够生成与文本提示无缝对齐的10秒连续视频，帧率为16帧每秒，分辨率为$768 \times 1360$像素。之前的视频生成模型常常面临动作有限和时长短的挑战，基于文本生成具有连贯叙事的视频尤其困难。我们提出了几种设计来解决这些问题。首先，我们引入了3D变分自编码器（VAE），以在空间和时间维度上压缩视频，从而提高压缩率和视频保真度。其次，为了改善文本和视频的对齐，我们提出了一种专家变换器，结合了专家自适应层归一化，以促进两种模态之间的深度融合。第三，通过采用渐进式训练和多分辨率帧打包，CogVideoX在生成形状多样、动态运动的连贯长时视频方面表现卓越。此外，我们开发了一套有效的流水线，包括针对文本和视频数据的多种预处理策略。我们的创新视频字幕模型显著提高了生成质量和语义对齐性。结果表明，CogVideoX在自动基准测试和人工评估中均取得了最先进的性能。我们将在 https://github.com/THUDM/CogVideo 发布CogVideoX的代码和模型检查点，以及我们的VAE模型和视频字幕模型。

# 1 引言

文本到视频模型的快速发展是显著的，得益于Transformer架构（Vaswani等，2017）和扩散模型（Ho等，2020）。早期对Transformer进行预训练并扩展以从文本生成视频的尝试显示出了很大的潜力，如CogVideo（Hong等，2022）和Phenaki（Villegas等，2022）。与此同时，扩散模型在视频生成领域最近取得了令人振奋的进展（Singer等，2022；Ho等，2022）。通过将Transformer作为扩散模型的主干，即扩散Transformer（DiT）（Peebles & Xie，2023），文本到视频生成达到了一个新的里程碑，这一点从令人印象深刻的Sora展示（OpenAI，2024）中可以得到佐证。尽管DiT的快速发展，但在技术上仍不清楚如何实现具有动态情节的长期一致的视频生成。例如，以“ 一道闪电劈开一块岩石，一个人从岩石中跳出”为提示生成视频，之前的模型面临困难。在本研究中，我们训练并介绍了CogVideoX，这是一组大规模的扩散Transformer模型，旨在生成具有丰富运动语义的长期、时间一致的视频。我们通过开发3D变分自编码器、专家Transformer、渐进训练管道以及视频数据过滤和字幕生成管道来解决上述挑战。首先，为有效处理高维视频数据，我们设计并训练了一个3D因果变分自编码器，该自编码器沿着空间和时间维度压缩视频。与之前的微调2D变分自编码器的方法（Blattmann等，2023）相比，这一策略显著减少了序列长度和相关的训练计算，并且帮助防止生成视频中的闪烁，确保帧之间的连续性。其次，为改善视频和文本之间的对齐，我们提出了一个具备专家自适应层归一化的专家Transformer，以促进两种模态之间的融合。为了确保视频生成中的时间一致性并捕捉大规模运动，我们建议使用3D全注意力机制，全面建模视频在时间和空间维度上的表现。第三，由于在线可用的大多数视频数据缺乏准确的文本描述，我们开发了一个视频字幕生成管道，能够准确描述视频内容。该管道用于为所有视频训练数据生成新的文本描述，显著增强了CogVideoX对精确语义理解的把握。此外，我们采用并设计了渐进训练技术，包括多分辨率帧打包和分辨率渐进训练，以进一步提升CogVideoX的生成性能和稳定性。此外，我们提出了显式均匀采样，通过在每个数据并行排名上设置不同时间步采样间隔，从而稳定训练损失曲线并加速收敛。

![](images/2.jpg)  

Figure 2: The performance of openly-accessible text-to-video models in different aspects.

截至目前，我们已完成了CogVideoX的训练，分别为5亿和2亿参数规模。无论是机器评估还是人工评估，都表明CogVideoX-5B的表现超越了知名视频模型，而CogVideoX-2B在大多数维度上也具有很强的竞争力。图2展示了CogVideoX-5B和CogVideoX-2B在不同方面的表现，表明CogVideoX具备可扩展性。随着模型参数、数据量和训练量的增加，未来性能将得到提升。我们的贡献可总结如下： • 我们提出了CogVideoX，这是一个简单且可扩展的结构，结合了3D因果变分自编码器和专家变换器，旨在生成连贯的长时长高动态视频。它能够生成多种长宽比、分辨率高达768 × 1360、长度为10秒、帧率为16fps的长视频。 • 我们通过自动化指标评估和人工评测对CogVideoX进行了评估，并与开放获取的顶尖文本到视频模型进行了比较。CogVideoX达到了最先进的性能。 • 我们公开发布了我们的5B和2B模型，包括文本到视频和图像到视频版本，这些都是首个商业级别的开源视频生成模型。我们希望它能够推动视频生成领域的发展。

# 2 CogVideoX架构

在本节中，我们介绍了 CogVideoX 模型。图 3 展示了整体架构。给定一对视频和文本输入，我们设计了一个 3D 因果变分自编码器（VAE）来将视频压缩到潜在空间中，然后将潜在表示进行分块并展开为表示为 $z$ 视觉的长序列。同时，我们使用 T5 (Raffel et al., 2020) 将文本输入编码为文本嵌入 ${ \it z } _ { \mathrm { t e x t } }$。随后，$z _ { \mathrm { t e x t } }$ 和 $\tilde { z } _ { \mathrm { v i s i o n } }$ 沿序列维度连接。连接后的嵌入随后被输入到一组专家 Transformer 块中。最后，模型输出经过逆块处理以恢复原始潜在形状，然后通过 3D 因果 VAE 解码器解码以重建视频。我们详细说明了 3D 因果 VAE 和专家 Transformer 的技术设计。

![](images/3.jpg)  

Figure 3: The overall architecture of CogVideoX.

# 2.1 三维因果变分自编码器

视频包含空间和时间信息，通常导致数据量远大于图像。为了解决建模视频数据的计算挑战，我们提议基于三维变分自编码器实现一个视频压缩模块（Yu et al., 2023b）。该想法是结合三维卷积对视频进行空间和时间的压缩。这可以有助于实现更高的压缩比，并显著提高视频重建的质量和连贯性。

Table 1: Ablation with different variants of 3D VAE. The baseline is SDXL(Podell et al., 2023) 2D VAE. Flickering calculates the L1 difference between each pair of adjacent frames to evaluate the degree of flickering in the video. We use variant B for pretraining.   

<table><tr><td>Variants</td><td>Baseline</td><td>A</td><td>B</td><td>C</td><td>D</td><td>E</td></tr><tr><td>Compression</td><td>8×8×1</td><td>8×8×4</td><td>8×8×4</td><td>8×8×4</td><td>8×8×8</td><td>16×16×8</td></tr><tr><td>Latent channel</td><td>4</td><td>8</td><td>16</td><td>32</td><td>32</td><td>128</td></tr><tr><td>Flickering↓</td><td>93.2</td><td>87.6</td><td>86.3</td><td>87.7</td><td>87.8</td><td>87.3</td></tr><tr><td>PSNR↑</td><td>28.4</td><td>27.2</td><td>28.7</td><td>30.5</td><td>29</td><td>27.9</td></tr></table>

![](images/4.jpg)  

Figure 4: (a) The structure of the 3D VAE in CogVideoX. It comprises an encoder, a decoder and a latent space regularizer, achieving a $8 \times 8 \times 4$ compression from pixels to the latents. (b) The context parallel implementation on the temporally causal convolution.

图4(a)展示了所提出的3D变分自编码器（VAE）的结构。它包括编码器、解码器和Kullback-Leibler（KL）正则化器。编码器和解码器由对称排列的阶段组成，分别执行2倍下采样和上采样，通过交错堆叠的ResNet模块实现。一些模块执行3D下采样（上采样），而其他模块仅执行2D下采样（上采样）。我们采用了时间因果卷积（Yu et al., 2023b），将所有填充放置在卷积空间的开始，如图4(b)所示。这确保了未来的信息不会影响当前或过去的预测。我们还进行了消融研究，比较了不同的压缩比和潜在通道，如表1所示。在使用3D结构后，重建的视频几乎没有抖动，随着潜在通道的增加，恢复质量得到提升。然而，当时空压缩过于激进（${1 6 \times 1 6 \times 8}$），即使通道维度相应增加，模型的收敛也变得极其困难。探索具有更大压缩比的变分自编码器是我们未来的工作。鉴于处理长时段视频会占用过多的GPU内存，我们在时间维度上应用上下文并行处理3D卷积，以便将计算分散到多个设备上。如图4(b)所示，由于卷积的因果特性，每个线程简单地将长度为$k - 1$的段发送到下一个线程，其中$k$表示时间核大小。这导致通信开销相对较低。在训练过程中，我们首先以$2 5 6 \times 2 5 6$的分辨率和17帧训练3D变分自编码器以节省计算资源。随机选择8或16fps以增强模型的鲁棒性。我们观察到该模型能够很好地编码更大分辨率的视频，而无需额外训练，因为它没有注意力模块，但在编码帧数更多的视频时效果不佳。因此，我们进行了两阶段训练，首先对17帧视频进行训练，然后通过上下文并行在161帧视频上进行微调。两个阶段都利用加权组合的L1重建损失、LPIPS（Zhang et al., 2018）感知损失和KL损失。在经过数千步的训练后，我们还引入了来自3D判别器的GAN损失。

# 2.2 专家变换器

我们引入了CogVideoX中Transformer的设计选择，包括补丁、位置嵌入和注意力策略。补丁化。3D因果变分自编码器（VAE）对形状为 $T \times H \times W \times C$ 的视频潜变量进行编码，其中 $T$ 代表帧数，$H$ 和 $W$ 分别代表每帧的高度和宽度，$C$ 代表通道数。视频潜变量随后被补丁化，生成长度为 ${ \it z } _ { \mathrm { v i s i o n } }$ 的序列，其长度为 ${ \frac { T } { q } } \cdot { \frac { H } { p } } \cdot { \frac { W } { p } }$ 当 $q > 1$ 时，序列开头是图像，以便实现图像和视频的联合训练。3D-RoPE。旋转位置嵌入（RoPE）（Su et al., 2024）是一种相对位置编码，已证明能够有效捕获大型语言模型（LLMs）中令牌之间的关系，尤其在建模长序列方面表现出色。为了适应视频数据，我们将原始的RoPE扩展为3D-RoPE。视频张量中的每个潜变量可以用三维坐标 $( x , y , t )$ 表示。我们独立地将1D-RoPE应用于坐标的每个维度，分别占据隐藏状态通道的3/8、3/8和2/8。然后，将得到的编码沿通道维度连接以获得最终的3D-RoPE编码。专家自适应层归一化。我们在输入阶段将文本和视频的嵌入拼接在一起，以更好地对齐视觉和语义信息。然而，这两种模态的特征空间差异显著，其嵌入的数值尺度甚至可能不同。为了在同一序列中更好地处理它们，我们采用专家自适应层归一化（Expert Adaptive Layernorm）来独立处理每种模态。如图3所示，遵循DiT（Peebles与Xie，2023），我们使用扩散过程的时间步 $t$ 作为调制模块的输入。然后，视觉专家自适应层归一化（Vision Expert AdaLN）和文本专家自适应层归一化（Text Expert AdaLN）分别将该调制应用于视觉隐藏状态和文本隐藏状态。此策略促进了两种模态特征空间的对齐，同时最小化了额外参数的使用。

![](images/5.jpg)  

Figure 5: The separated spatial and temporal attention makes it challenging to handle the large motion between adjacent frames. In the figure, the head of the person in frame $i + 1$ cannot directly attend to the head in frame $i$ . Instead, visual information can only be implicitly transmitted through other background patches. This can lead to inconsistency issues in the generated videos.

# 3D 全注意力机制。前期工作

(Singer et al., 2022; Guo et al., 2023) 通常采用分离的空间和时间注意力机制，以降低计算复杂性并促进从文本到图像模型的微调。然而，如图5所示，这种分离注意力的方法需要大量隐式传递视觉信息，显著增加了学习复杂性，并使大幅移动物体的一致性保持变得具有挑战性。考虑到大语言模型在长上下文训练中的巨大成功（AI@Meta, 2024）以及FlashAttention的高效性（Dao et al., 2022），我们提出了一种3D文本-视频混合注意力机制。该机制不仅取得了更好的效果，还可以轻松适应各种并行加速方法。

![](images/6.jpg)  

Figure 6: The diagram of mixed-duration training and Frame Pack. To fully utilize the data and enhance the model's generalization capability, we train on videos of different duration within the same batch.

# 3 训练 CogViDEOX

我们在训练过程中混合图像和视频，将每张图像视为单帧视频。此外，我们从分辨率的角度采用渐进训练。在扩散设置中，我们采用v-prediction（Salimans & Ho, 2022）和零信噪比（Lin et al., 2024），遵循LDM（Rombach et al., 2022）中使用的噪声调度。

# 3.1 多分辨率帧包

以往的视频训练方法通常涉及图像和视频的联合训练，且帧数固定（Singer等，2022；Blattmann等，2023）。然而，这种方法通常会导致两个问题：首先，使用双向注意力时，两种输入类型之间存在显著差距，图像只有一帧，而视频则有数十帧。我们观察到，采用这种方式训练的模型往往会根据标记数量分化为两种生成模式，而没有良好的泛化能力。其次，为了在固定时长内训练，我们必须丢弃短视频并截断长视频，这阻碍了对帧数不等的视频的充分利用。对于不同分辨率，SDXL（Podell等，2023）采用了分桶方法来解决生成裁剪图像的问题，但这使得数据和训练流程变得更加复杂。为了解决这些问题，我们选择了混合时长训练，这意味着将不同长度的视频一起训练。然而，批次内数据形状不一致使得训练变得困难。受到Patch'n Pack（Dehghani等，2024）的启发，我们将不同时长（也有不同分辨率）的视频放入同一批次，以确保每个批次内形状一致，这一方法被称为多分辨率帧打包，如图6所示。我们使用3D RoPE来建模各种视频形状之间的位置关系。有两种方法可以使RoPE适应不同的分辨率和时长。一种方法是扩展位置编码表，对于每个视频，根据分辨率选择表的前半部分（外推）。另一种方法是将固定长度的位置编码表缩放以匹配视频的分辨率（插值）。考虑到RoPE是一种相对位置编码，我们选择了第一种方法，以保持模型细节的清晰度。

# 3.2 渐进式训练

来自互联网的视频通常包含大量低分辨率视频。直接在高分辨率视频上训练成本极高。为了充分利用数据并节省成本，模型首先在256px的视频上进行训练，以学习语义和低频知识。然后在逐渐增加的分辨率上进行训练，从256px到512px，再到768px，以学习高频知识。为了保持生成不同宽高比视频的能力，我们保持宽高比不变，并将短边调整为上述分辨率。最后，我们进行高质量的微调，详见附录A。此外，我们基于上述模型训练了一个图像到视频的模型，详细信息见附录D。

# 3.3 显式均匀采样

Ho 等人（2020）将扩散的训练目标定义为 $t$ 在 1 和 T 之间均匀分布。通常做法是数据并行组中的每个计算单元在 1 和 $T$ 之间均匀地采样一个值，这在理论上等同于方程 1。然而，在实践中，从这种随机采样得到的结果往往不够均匀，且由于扩散损失的大小与时间步长有关，这可能导致损失的显著波动。因此，我们提出使用明确均匀采样将范围从 1 到 $T$ 划分为 $n$ 个区间，其中 $n$ 是计算单元的数量。然后，每个计算单元在其各自的区间内均匀采样。这种方法确保了时间步长的更均匀分布。如图 10 (d) 所示，使用明确均匀采样训练的损失曲线明显更稳定。

$$
L _ { \mathrm { s i m p l e } } ( \theta ) : = \mathbf { E } _ { t , x _ { 0 } , \epsilon } \big \| \epsilon - \epsilon _ { \theta } \big ( \sqrt { \bar { \alpha } _ { t } } x _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , t \big ) \big \| ^ { 2 } ,
$$

# 3.4 数据

我们构建了一个相对高质量的视频剪辑集合，这些剪辑配有文本描述，并使用视频过滤器和重标记模型进行处理。经过过滤，剩下约3500万个单次剪辑，每个剪辑平均约6秒。我们还使用了从LAION-5B（Schuhmann等，2022）和COYO-700M（Byeon等，2022）数据集中过滤出的2亿张具有美学评分的图像来辅助训练。视频过滤。视频生成模型应该捕捉世界的动态特性。然而，原始视频数据由于以下两个内在原因，往往含有显著的噪声：首先，在视频创作过程中进行的人工编辑可能扭曲真实的动态信息；其次，由于拍摄问题，如相机抖动或使用劣质设备，视频质量可能受到影响。除了视频的内在质量，我们还考虑视频数据对模型训练的支持程度。动态信息最少或动态方面缺乏连通性的视频被视为有害。因此，我们开发了一套负标签，包括：• 编辑：明显经过人工处理的视频，如重新编辑和特效，损害了视觉完整性。• 缺乏运动连通性：过渡缺乏连贯运动的视频片段，通常出现在人工拼接的视频或从静态图像编辑而成的视频中。• 低质量：拍摄效果差、画面不清晰或抖动过大的视频。• 讲座类型：主要以持续讲话的人为主题，缺乏有效运动的视频，如讲座和直播讨论。• 文本主导：包含大量可见文本或主要聚焦于文本内容的视频。• 嘈杂的截图：直接从手机或计算机屏幕捕获的视频，通常特征为质量差。我们首先抽样20000个视频，并根据其质量将每个视频标记为正面或负面。利用这些标注，我们基于Video-LLaMA（Zhang等，2023b）训练了6个过滤器，以筛选出低质量视频数据。负标签的示例及分类器在测试集上的表现可以在附录K中找到。此外，我们计算了所有训练视频的光流评分和图像美学评分，并在训练期间动态调整其阈值，以确保生成视频的动态和美学质量。视频标注。视频-文本配对对于文本到视频生成模型的训练至关重要。然而，大多数视频数据并没有对应的描述性文本。因此，有必要对视频数据进行全面的文本描述标注。目前有一些可用的视频标注数据集，如Panda70M（Chen等，2024b）、COCO Caption（Lin等，2014）和WebVid Bain等（2021b）。然而，这些数据集中的标注通常非常简短，无法全面描述视频内容。

![](images/7.jpg)  

Figure 7: The pipeline for dense video caption data generation. In this pipeline, we generate short video captions with the Panda70M model, extract frames to create dense image captions, and use GPT-4 to summarize these into final video captions. To accelerate this process, we fine-tuned a Llama 2 model with the GPT-4 summaries.

为了生成高质量的视频字幕数据，我们建立了一条密集视频字幕数据生成流水线，如图7所示。其主要思路是借助图像字幕生成视频字幕。首先，我们使用Chen等人（2024b）的video caption模型为视频生成简短字幕。然后，我们采用在CogView3（Zheng等，2024a）中使用的图像重标注模型CogVLM（Wang等，2023a）为每帧创建密集的图像字幕。随后，我们使用GPT-4对所有图像字幕进行总结，以生成最终视频字幕。为了加速从图像字幕到视频字幕的生成，我们对LLaMA2（Touvron等，2023）进行了微调，使用GPT-4生成的摘要数据，从而实现大规模视频字幕数据生成。有关视频字幕数据生成过程的更多细节请参见附录G。为了进一步加速视频重标注，我们还对基于CogVLM2-Video（Hong等，2024）和Llama3（AI@Meta，2024）的端到端视频理解模型CogVLM2-Caption进行了微调，使用上述流水线生成的密集字幕数据。由此端到端的CogVLM2-Caption模型生成的视频字幕示例如图15和附录H所示。CogVLM2-Caption能够提供视频内容和变化的详细描述。有趣的是，我们发现通过连接CogVideoX和CogVLM2-Caption可以进行视频到视频的生成，具体详见附录I。

# 4 实验

# 4.1 消融研究

我们对第2节中提到的一些设计进行了消融研究，以验证其有效性。 位置嵌入。我们将3D RoPE与正弦绝对位置嵌入进行了比较。正如图10a所示，RoPE的损失曲线收敛明显快于绝对位置嵌入。这与大语言模型中的常见选择一致。 专家自适应层归一化。我们在图8a、8d和图10c中比较了三种架构：MMDiT Esser等（2024），专家自适应层归一化（CogVideoX），以及不使用专家自适应层归一化的模型。

![](images/8.jpg)  

Figure 8: Ablation studies on WebVid test dataset with 500 videos. MMDiT1 has the same number of parameters with the expert AdaLN. MMDiT2 has the same number of layers but twice number of parameters. a, b, c measure FVD, d measures CLIP4Clip score.

Cross-attention DiT 在 (Esser et al., 2024) 中被证明不如 MMDiT，因此我们不再赘述。根据 FVD、CLIP4Clip(Luo et al., 2022) 的得分和损失，专家级 AdaLN 显著优于没有专家级 AdaLN 的模型以及参数数量相同的 MMDiT。我们推测，专家自适应层归一化足以缓解两种模态之间特征空间的差异。因此，MMDiT 中的两个独立变换器并不是必需的，这大大增加了参数的数量。此外，专家级 AdaLN 的设计比 MMDiT 更加简化，更接近当前的大语言模型，更容易进行进一步扩展。 3D 全注意力。在图 8b 和图 10b 中，当我们用 2D + 1D 注意力替换 3D 全注意力时，FVD 在早期步骤中会显著高于 3D 注意力。我们还观察到 2D+1D 不稳定，容易崩溃。我们假设，随着模型规模的增加，比如 5B，训练变得更容易不稳定，对结构设计提出了更高的要求。如第 2.2 节所讨论的，2D+1D 结构不适合视频生成任务，这可能导致训练过程中的不稳定。 显式均匀采样。从图 8c 和图 10d 中，我们发现使用显式均匀采样可以使损失更加稳定地下降，获得更好的性能。此外，在表 9 中，我们比较了两种选择下每个扩散时间步的损失，以便进行更精确的比较。我们发现所有时间步的损失在显式均匀采样下更低，这表明该方法也可以加速损失收敛。我们推测，这是因为不同时间步的损失变化很大。当用于训练的时间步采样不够均匀时，由于上述随机性，损失波动很大。显式均匀性可以减少随机性，从而带来所有时间步的共同下降。

# 4.2 评估

# 4.2.1 自动化指标评估

VAE 重建效果 我们将我们的 3DVAE 与其他开源 3DVAE 进行比较，使用 WebVid（Bain 等，2021a）的验证集，分辨率为 $256 \times 256$ 的 17 帧视频。在表 2 中，我们的 VAE 实现了最佳的 PSNR，并展现了最小的抖动。值得注意的是，其他 VAE 方法使用的潜在通道比我们的少。评估指标。为了评估文本到视频的生成，我们采用 Vbench（Huang 等，2024）中与人类感知一致的几个指标：人类动作、场景、动态程度、多对象及外观风格。其他指标，如颜色，往往会对简单、静态的视频给予更高的分数，因此我们不使用它们。

Table 2: Comparison with the performance of other spatiotemporal compression VAEs.   

<table><tr><td></td><td>Flickering ↓</td><td>PSNR ↑</td></tr><tr><td>Open-Sora</td><td>92.4</td><td>28.5</td></tr><tr><td>Open-Sora-Plan</td><td>90.2</td><td>27.6</td></tr><tr><td>Ours</td><td>85.5</td><td>29.1</td></tr></table>

Table 3: Evaluation results of CogVideoX-5B and CogVideoX-2B.   

<table><tr><td rowspan="2">Models</td><td rowspan="2">Human Action</td><td rowspan="2">Scene</td><td colspan="4">Dynamic Multiple Appear. Dynamic GPT4o-MT</td><td rowspan="2">Score</td></tr><tr><td>Degree</td><td>Objects</td><td>Style</td><td>Quality</td></tr><tr><td>T2V-Turbo(Li et al., 2024)</td><td>95.2</td><td>55.58</td><td>49.17</td><td>54.65</td><td>24.42</td><td></td><td></td></tr><tr><td>AnimateDiffGuo et al. (2023)</td><td>92.6</td><td>50.19</td><td>40.83</td><td>36.88</td><td>22.42</td><td></td><td>2.62</td></tr><tr><td>VideoCrafter-2.0(Chen et al., 2024a)</td><td>95.0</td><td>55.29</td><td>42.50</td><td>40.66</td><td>25.13</td><td>43.6</td><td>2.68</td></tr><tr><td>OpenSora V1.2(Zheng et al., 2024b)</td><td>85.8</td><td>42.47</td><td>47.22</td><td>58.41</td><td>23.89</td><td>63.7</td><td>2.52</td></tr><tr><td>Show-1(Zhang et al., 2023a)</td><td>95.6</td><td>47.03</td><td>44.44</td><td>45.47</td><td>23.06</td><td>57.7</td><td>−</td></tr><tr><td>Gen-2(runway, 2023)</td><td>89.2</td><td>48.91</td><td>18.89</td><td>55.47</td><td>19.34</td><td>43.6</td><td>2.62</td></tr><tr><td>Pika(pik, 2023)</td><td>88.0</td><td>44.80</td><td>37.22</td><td>46.69</td><td>21.89</td><td>52.1</td><td>2.48</td></tr><tr><td>LaVie-2(Wang et al., 2023b)</td><td>96.4</td><td>49.59</td><td>31.11</td><td>64.88</td><td>25.09</td><td></td><td>2.46</td></tr><tr><td>CogVideoX-2B</td><td>96.6</td><td>55.35</td><td>66.39</td><td>57.68</td><td>24.37</td><td>57.7</td><td>3.09</td></tr><tr><td>CogVideoX-5B</td><td>96.8</td><td>55.44</td><td>62.22</td><td>70.95</td><td>24.44</td><td>69.5</td><td>3.36</td></tr></table>

对于生成较长视频的情况，一些模型可能会产生帧间变化最小的视频以获得更高的评分，但这些视频缺乏丰富的内容。因此，评估动态性的指标变得重要。为此，我们使用了两种视频评估工具：动态质量（Liao et al., 2024）和GPT4o-MTScore（Yuan et al., 2024）。动态质量通过结合各种质量指标与动态评分来定义，减轻由于视频动态与视频质量之间的负相关性而产生的偏差。GPT4o-MTScore是一种设计用来测量时间推移视频的变异幅度的指标，使用GPT-4o，适用于描绘物理、生物和气象变化的视频。结果。表3提供了CogVideoX与其他模型的性能比较。CogVideoX-5B在七个指标中的五个指标上达到了最佳性能，并在剩余两个指标上表现出竞争力。这些结果表明，该模型不仅在视频生成质量上表现卓越，还在处理各种复杂动态场景方面优于先前的模型。此外，图2展示了一个雷达图，直观地展示了CogVideoX的性能优势。我们在附录A中提供了不同分辨率下推理时的时间和空间消耗。

# 4.2.2 人工评估

除了自动评分机制，我们还建立了一个全面的人类评估框架，以评估视频生成模型的一般能力。评估者将从四个方面对生成的视频进行打分：感官质量、指令遵循、物理仿真和覆盖质量，使用三个级别：0、0.5 或 1。每个级别由详细的指导方针定义。具体细节见附录 J。在这个框架下，我们将最佳闭源模型之一 Kling（2024.7）与 CogVideoX-5B 进行比较。表 4 显示的结果表明，CogVideoX-5B 在所有方面均优于 Kling，获得了人类的偏好。

Table 4: Human evaluation between CogVideoX and Kling.   

<table><tr><td>Model</td><td>Sensory Quality</td><td>Instruction Following</td><td>Physics Simulation</td><td>Cover Quality</td><td>Total Score</td></tr><tr><td>Kling</td><td>0.638</td><td>0.367</td><td>0.561</td><td>0.668</td><td>2.17</td></tr><tr><td>CogVideoX-5B</td><td>0.722</td><td>0.495</td><td>0.667</td><td>0.712</td><td>2.74</td></tr></table>

# 5 结论

在本文中，我们介绍了CogVideoX，一个最先进的文本到视频扩散模型。它利用3D变分自编码器（VÅE）和专家变换器架构生成具有显著运动的连贯长时间视频。我们还正在探索视频生成模型的扩展法则，旨在训练更大更强的模型，以生成更长和更高质量的视频，推动文本到视频生成的可实现边界。

# 鸣谢

本研究得到了国家自然科学基金 62425601 和 62495063、清华大学创新科研计划 20233080067 以及新基石科学基金通过 XPLORER 奖的支持。我们感谢所有数据标注者、基础设施运营者、合作伙伴及各位合作者。我们还要感谢在 Zhipu AI 和清华大学为 CogVideoX 提供支持、反馈或贡献的所有人员，即便未在本报告中明确提及。我们还特别感谢 BiliBili 的技术讨论。

# REFERENCES

Pika beta. 2023. URL https://pika.art/home.

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

AI@Meta. Llama 3 model card. 2024. URL https://github.com/meta-1lama/1lama3/ blob/main/MODEL_CARD.md.

Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In IEEE International Conference on Computer Vision, 2021a.

Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 17281738, 2021b.

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/kakaobrain/ coyo-dataset, 2022.

Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 73107320, 2024a.

Tsai-Shien Chen, Aliaksandr Siarohin, Wili Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda70m: Captioning 70m videos with multiple cross-modality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13320-13331, 2024b.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:1634416359, 2022.

Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M Alabdulmohsin, et al. Patch n'pack: Navit, a vision transformer for any aspect ratio and resolution. Advances in Neural Information Processing Systems, 36, 2024.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.

Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models Advances in neural information processing systems, 33:68406851, 2020.

Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.

Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022.

Wenyi Hong, Weihan Wang, Ming Ding, Wenmeng Yu, Qingsong Lv, Yan Wang, Yean Cheng, Shiyu Huang, Junhui Ji, et al. Cogvlm2: Visual language models for image and video understanding. arXiv preprint arXiv:2408.16500, 2024.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

PKU-Yuan Lab and Tuzhan AI etc. Open-sora-plan, April 2024. URL https://doi.org/ 10.5281/zenodo.10948109.

Jiachen Li, Weixi Feng, Tsu-Jui Fu, Xinyi Wang, Sugato Basu, Wenhu Chen, and William Yang Wang. T2v-turbo: Breaking the quality bottleneck of video consistency model with mixed reward feedback. arXiv preprint arXiv:2405.18750, 2024.

Mingxiang Liao, Hannan Lu, Xinyu Zhang, Fang Wan, Tianyu Wang, Yuzhong Zhao, Wangmeng Zuo, Qixiang Ye, and Jingdong Wang. Evaluation of text-to-video generation models: A dynamics perspective, 2024. URL https://arxiv.org/abs/2407.01094.

Shanchuan Lin, Bingchen Liu, Jiashi Li, and Xiao Yang. Common diffusion noise schedules and sample steps are flawed. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp. 54045411, 2024.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision-ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pp. 740755. Springer, 2014.

Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li. Clip4clip: An empirical study of clip for end to end video clip retrieval and captioning. Neurocomputing, 508:293304, 2022.

OpenAI. Sora. 2024. URL https://openai.com/index/sora/.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 41954205, 2023.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):167, 2020.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1068410695, 2022.

runway. Gen-2. 2023. URL https://runwayml.com/ai-tools/gen-2-text-to-video.

Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models arXiv preprint arXiv:2202.00512, 2022.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:2527825294, 2022.

Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792, 2022.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, and Jan Kautz. Mocogan: Decomposing motion and content for video generation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 15261535, 2018.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual descriptions. In International Conference on Learning Representations, 2022.

Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models arXiv preprint arXiv:2311.03079, 2023a.

Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023b.

Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021.

Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1045910469, 2023a.

Lijun Yu, José Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, et al. Language model beats diffusion-tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023b.

Sihyun Yu, Jihoon Tack, Sangwoo Mo, Hyunsu Kim, Junho Kim, Jung-Woo Ha, and Jinwoo Shin. Generating videos with dynamics-aware implicit generative adversarial networks. arXiv preprint arXiv:2202.10571, 2022.

Shenghai Yuan, Jinfa Huang, Yongqi Xu, Yaoyang Liu, Shaofeng Zhang, Yujun Shi, Ruijie Zhu, Xinhua Cheng, Jiebo Luo, and Li Yuan. Chronomagic-bench: A benchmark for metamorphic evaluation of text-to-time-lapse video generation. arXiv preprint arXiv:2406.18522, 2024.

David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation. arXiv preprint arXiv:2309.15818, 2023a.

Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023b.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586595, 2018.

Wendi Zheng, Jiayan Teng, Zhuoyi Yang, Weihan Wang, Jidong Chen, Xiaotao Gu, Yuxiao Dong, Ming Ding, and Jie Tang. Cogview3: Finer and faster text-to-image generation via relay diffusion. arXiv preprint arXiv:2403.05121, 2024a.

Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, March 2024b. URL https://github.com/hpcaitech/Open-Sora.

# APPENDIX CONTENTS

• Appendix A: Training Details   
• Appendix B: Loss Curve   
• Appendix C: More Examples   
• Appendix D: Image To Video Model   
• Appendix E: Related Works   
•Appendix F: Caption Upsampler   
• Appendix G: Dense Video Caption Data Generation   
• Appendix H: Video Caption Example   
• Appendix I: Video to Video via CogVideoX and CogVLM2-Caption   
•Appendix J: Human Evaluation Details   
•Appendix K: Data Filtering Details

# A Training Details

High-Quality Fine-Tuning. Since the filtered pre-training data still contains a certain proportion of dirty data, such as subtitles, watermarks, and low-bitrate videos, we selected a subset of higher quality video data, accounting for $2 0 \%$ of the total dataset, for fine-tuning in the final stage. This step effectively removed generated subtitles and watermarks and slightly improved the visual quality. However, we also observed a slight degradation in the model's semantic ability.

Visualizing different rope interpolation methods When adapting low-resolution position encoding to high-resolution, we consider two different methods: interpolation and extrapolation. We show the effects of two methods in Figure 9. Interpolation tends to preserve global information more effectively, whereas the extrapolation better retains local details. Given that RoPE is a relative position encoding, We chose the extrapolation to maintain the relative position between pixels.

![](images/9.jpg)  
Figure 9: The comparison between the initial generation states of extrapolation and interpolation when increasing the resolution with RoPE. Extrapolation tends to generate multiple small, clear, and repetitive images, while interpolation generates a blurry large image.

Model & Training Hyperparameters We present the model and training hyperparameters in table 5 and table 6.

Table 5: Hyperparameters of CogvideoX-2b and CogVideo-5b.   

<table><tr><td>Training Stage</td><td>stage1</td><td>stage2</td><td>stage3</td><td>stage4 (FT)</td></tr><tr><td>Max Resolution</td><td>256×384</td><td>480×720</td><td>768×1360</td><td>768×1360</td></tr><tr><td>Max duration</td><td>6s</td><td>6s</td><td>10s</td><td>10s</td></tr><tr><td>Batch Size</td><td>2000</td><td>1000</td><td>250</td><td>100</td></tr><tr><td>Sequence Length</td><td>25k</td><td>75k</td><td>700k</td><td>700k</td></tr><tr><td>Training Steps</td><td>400k</td><td>220k</td><td>120k</td><td>10k</td></tr></table>

Table 6: Hyperparameters of CogvideoX-2b and CogVideo-5b.   

<table><tr><td>Hyperparameter</td><td>CogvideoX-2b</td><td>CogVideo-5b</td></tr><tr><td>Number of Layers</td><td>30</td><td>42</td></tr><tr><td>Attention heads</td><td>32</td><td>48</td></tr><tr><td>Hidden Size</td><td>1920</td><td>3072</td></tr><tr><td>Position Encoding</td><td>sinusoidal</td><td>RoPE</td></tr><tr><td>Time Embedding Size</td><td>256</td><td></td></tr><tr><td>Weight Decay</td><td>1e-4</td><td></td></tr><tr><td>Adam </td><td>1e-8</td><td></td></tr><tr><td>Adam β1</td><td>0.9</td><td></td></tr><tr><td>Adam β2</td><td>0.95</td><td></td></tr><tr><td>Learning Rate Decay</td><td>cosine</td><td></td></tr><tr><td>Gradient Clipping</td><td>1.0</td><td></td></tr><tr><td>Text Length</td><td>226</td><td></td></tr><tr><td>Max Sequence Length</td><td>82k</td><td></td></tr><tr><td>Lowest aesthetic-value</td><td>4.5</td><td></td></tr><tr><td>Training Precision</td><td>BF16</td><td></td></tr></table>

<table><tr><td></td><td>5b-480x720-6s</td><td>5b-768x1360-5s</td><td>2b-480x720-6s</td><td>2b-768x1360-5s</td></tr><tr><td>Time</td><td>113s</td><td>500s</td><td>49s</td><td>220s</td></tr><tr><td>Memory</td><td>26GB</td><td>76GB</td><td>18GB</td><td>53GB</td></tr></table>

Table 7: Inference time and memory consumption of CogVideoX. We evaluate the model on bf, H800 with 50 inference steps.

<table><tr><td></td><td>256*384*6s</td><td>480*720*6s</td><td>768*1360*5s</td></tr><tr><td>2D+1D</td><td>0.38s</td><td>1.26s</td><td>4.17s</td></tr><tr><td>3D</td><td>0.41s</td><td>2.11s</td><td>9.60s</td></tr></table>

Table 8: Inference time comparison between 3D Full attention and 2D+1D attention. We evaluate the model on bf, H800 with one dit forward step. Thanks to the optimization by Flash Attention, the increase in sequence length does not make the inference time unacceptable.

# B Loss

![](images/10.jpg)  
Figure 10: Training loss curve of different ablations.

Table 9: Validation loss at different diffusion timesteps when the training steps is 40k.   

<table><tr><td>Timestep</td><td>100</td><td>300</td><td>500</td><td>700</td><td>900</td></tr><tr><td>w/o explicit uniform sampling</td><td>0.222</td><td>0.130</td><td>0.119</td><td>0.133</td><td>0.161</td></tr><tr><td>w/ explicit uniform sampling</td><td>0.216</td><td>0.126</td><td>0.116</td><td>0.129</td><td>0.157</td></tr></table>

# C More ExampLes

More text-to-video examples are shown in Figure 11 and Figure 12.

# D ImAgE To VIdEo ModEL

We finetune an image-to-video model from the text-to-video model. Drawing from the (Blattmann et al., 2023), we add an image as an additional condition alongside the text. The image is passed through 3D VAE and concatenated with the noised input in the channel dimension. Similar to super-resolution tasks, there is a significant distribution gap between training and inference (the first frame of videos vs. real-world images). To enhance the model's robustness, we add large noise to the image condition during training. Some examples are shown in Figure 13, Figure 14. CogVideoX can handle different styles of image input.

# E RELATED WORKS

Video diffusion models Generating videos has been explored through various types of generative models, such as Generative Adversarial Networks (GANs) (Yu et al., 2022; Tulyakov et al., 2018), autoregressive methods (Hong et al., 2022; Yan et al., 2021), and nonautoregressive methods (Villegas et al., 2022; Yu et al., 2023a). Diffusion models have recently gained significant attention, achieving remarkable results in both image generation(Rombach et al., 2022; Esser et al., 2024) and video generation(Singer et al., 2022; Blattmann et al., 2023; Guo et al., 2023). However, the limited compression ratio and simple training strategy often restrict the generation to low-resolution short-duration videos (2-3 seconds), requiring multiple super-resolution and frame interpolation models to be cascaded(Singer et al., 2022; Ho et al., 2022) for a generation. This leads to generated videos with limited semantic information and minimal motion.

Video VAEs To increase the compression ratio of videos and reduce computation costs, a common approach is to encode the video into a latent space using a Variational Autoencoder(VAE), which is also widely used in image generation. Early video models usually directly use image VAE for generation. However, modeling only the space dimension can result in jittery videos. SVD(Blattmann et al., 2023) tries to finetune the image VAE decoder to solve the jittering issue. However, this approach cannot take advantage of the temporal redundancy in videos and still cannot achieve an optimal compression rate. Recently, some video models(Zheng et al., 2024b; Lab & etc., 2024) try to use 3D VAE for temporal compression, but small latent channels still result in blurry and jittery videos.

![](images/11.jpg)  
Text Prompt: A few golden retrievers playing in the snow

![](images/12.jpg)

![](images/13.jpg)  
Text Prompt: Three dolphins leap out of the ocean at sunset, then splash into the water   
in a large gallery at the New York Museum.

![](images/14.jpg)

![](images/15.jpg)  
Text Prompt: Mushroom turns into a bear   
Figure 11: Text to video showcases. The displayed prompt will be upsampled before being fed into the model. The generated videos contain large motion and various styles.

Teo movie style

![](images/16.jpg)  
up, the people in the car are blown up, the screen shakes, the movie winds up

![](images/17.jpg)  
Text Prompt: A man running in the snow

![](images/18.jpg)

![](images/19.jpg)  
vehicleso

![](images/20.jpg)  
Text Promp in deep thought.   
Figure 12: Text to video showcases.

![](images/21.jpg)  
Text Prompt: A raging tsunami flooded the village

![](images/22.jpg)

![](images/23.jpg)  
Tex rompt: The uncle and nephew are seen looking at each other and then smiling and mbracing to eac ohr

![](images/24.jpg)  
Figure 13: Image to video showcases. The displayed prompt will be upsampled before being fed into the model.

![](images/25.jpg)  
Text rompt: An elephant slowly walks out of a cloud of fog, the og shatters and flows

![](images/26.jpg)

![](images/27.jpg)  
Te ompt: A il lows her head and ubs her fce aginst a pupy, the pupy loks u at the l

![](images/28.jpg)  
Text Prompt: A woman presses a camera shutter, her hair fying

![](images/29.jpg)  
Te o: A pu coss s ye, ens ts mou, and turns ts head o ba   
Figure 14: Image to video showcases.

# F CAPTION UPSAMPLER

To ensure that text input distribution during inference is as close as possible to the distribution during training, similar to (Betker et al., 2023), we use a large language model to upsample the user's input during inference, making it more detailed and precise. Finetuned LLM can generate better prompts than zero/few-shot.

For image-to-video, we use the vision language model to upsample the prompt, such as GPT4V, CogVLM(Wang et al., 2023a).

<table><tr><td>Zero-shot prompt for Text Upsampler</td></tr><tr><td>You are part of a team of bots that create videos. You work with an assistant bot that will draw anything you say in square brackets. For example, outputting \&quot; a beautiful morning in the woods with the sun peaking through the trees \&quot; will trigger your partner bot to output a video of a forest morning, as described. You will be prompted by people looking to create detailed, amazing videos. The way to accomplish this is to take their short prompts When modifications are requested, you should not simply</td></tr><tr><td>and make them extremely detailed and descriptive. There are a few rules to follow : You will only ever output a single video description per user request.</td></tr></table>

# G Dense Video Caption Data Generation

In the pipeline for generating video captions, we extract one frame every two seconds for image captioning. Ultimately, we collected 50,000 data points to fine-tune the summary model. Below is the prompt we used for summarization with GPT-4:

<table><tr><td>Prompt for GPT-4 Summary</td></tr><tr><td>We extracted several frames from this video and described each frame using an image understanding model, stored in the dictionary variable &#x27;image_captions: Dict[str: str]&#x27;. In &#x27;image_captions&#x27;, the key is the second at which the image appears in the video, and the value is a detailed description of the image at that moment. Please describe the content of this video in as much detail as possible, based on the information provided by &#x27;image_captions&#x27;, including the objects, scenery, animals, characters, and camera movements within the video. \n image_captions={new_captions}\n You should output your summary directly, and not mention variables like &#x27;image_captions&#x27; in your response. Do not include &#x27;\\n&#x27; and the word &#x27;video&#x27; in your response. Do not use introductory phrases such as: \&quot;The video presents\&quot;, \&quot;The video depicts\&quot;, \&quot;This video showcases\&quot;, \&quot;The video captures\&quot; and so on.\n Please start the description with the video content directly, such as \&quot;A man first sits in a chair, then stands up and walks to the kitchen....\&quot;\n Do not use phrases like: \&quot;as the video</td></tr></table>

the content of the video and the changes that occur, in chronological order.\n Please keep the description of this video within 100 English words.

# H VIDeo CaPtion ExaMPLE

Below we present more examples to compare the performance of the Panda-70M video captioning model and our CogVLM2-Caption model:

![](images/30.jpg)  
Figure 15: An example from CogVLM2-Caption provides a detailed description of all specific objects and movements.

![](images/31.jpg)

Caption Generated by Panda-70M

A close up of a white dragon with blue eyes.

# Caption Generated by CogVLM2-Caption

A meticulously crafted white dragon with a serene expression and piercing blue eyes is showcased in various close-ups, its scales detailed with golden highlights. Initially, the dragon's horns are curved back, but as time passes, they change to spiraling $\mathtt { g o l d }$ , adding elegance. The creature's demeanor shifts from tranquil majesty to a more formidable presence, with its mouth agape revealing sharp teeth at times. Its expressions alternate between calmness and fierce determination, maintaining an air of mythical serenity throughout. The consistent neutral background focuses attention on the dragon's textured features and expressive eyes.

![](images/32.jpg)

<table><tr><td>Caption Generated by Panda-70M</td></tr><tr><td>A crab is walking on the beach with a light bulb on its back.</td></tr><tr><td>Caption Generated by CogVLM2-Caption A hermit crab with a dark, glossy shell and reddish-brown legs</td></tr><tr><td>is seen carrying an illuminated light bulb on its back across the sandy terrain of a beach at night. The scene transitions from a soft glow to a more pronounced illumination as the crab moves, highlighting its delicate limbs against the smooth sand and tranquil sea backdrop. This surreal tableau blends natural beauty with human ingenuity, creating a serene yet whimsical atmosphere that emphasizes the crab&#x27;s unique adaptation and the</td></tr></table>

![](images/33.jpg)

<table><tr><td>Caption Generated by Panda-70M</td></tr><tr><td>A young black man is sitting on a cloud and reading a book with a blue sky in the background.</td></tr><tr><td>Caption Generated by CogVLM2-Caption</td></tr><tr><td>A young Black man with an afro hairstyle and a neatly trimmed beard is seen sitting cross-legged on fluffy white clouds, deeply engrossed in reading a book with a red cover. He wears a plain white T-shirt and dark pants against a vivid blue sky dotted with cumulus clouds. Throughout the scenes, his expression remains one of deep concentration and peaceful contemplation, highlighting a moment of intellectual pursuit amidst nature&#x27;s grandeur. The imagery suggests a serene atmosphere that emphasizes solitude and introspection, with no other people or objects around him.</td></tr></table>

# I ViDEO tO VIdEO VIa CogViDEoX ANd CoGVLM2-CaPtiIoN

In this section, we present several examples of video-to-video generation using CogVideoX and CogVLM2-Caption. Specifically, we first input the original video into CogVLM2-Caption to obtain the video's caption, and then feed this caption into the CogVideoX model to generate a new video. From the examples below, it can be seen that our pipeline achieves a high degree of fidelity to the original video, showing that CogVLM2-Caption can capture almost all the details in the video.

Source Video   
Caption Generated by CogVLM2-Caption   
A man dressed as a wizard, with a blue robe adorned with white stars and a matching pointed hat, stands in a dark cave. He is engaged in casting spells from an open book, surrounded by mystical flms that illuminate the scenThroughout he sequenc his ight hand is raised tochannel ight lightning bolts towards unseen targets, while his let hand holds the book firmly. His expression remains focused and serious, suggesting deep concentration on his magical endeavors. The atmosphere of mystery and ancient magic is enhanced by the surrounding darkness and the vivid display of light and shadow.   
Video Generated by CogVideoX Source Video   
Caption Generated by CogVLM2-Caption   
A picturesque evening descends on a cliffside village, showcasing whitewashed buildings with blue domes that glow against the darkening sky. The Aegean Sea mirrors this celestial hue, creating a serene tableau devoid of people and vehicles. As time passes, the scene remains tranquil, lluminated by golden lhts from within homes and l patways weaving between structures. A solitary winmill stands out, symbolizing local culture amidst the peaceful settng. The absence ofvisible human activity emphasizes the stillness and beauty of the coastal hamlet, inviting contemplation in its embrace. Video Generated by CogVideoX   
Source Video   
Caption Generated by CogVLM2-Caption   
A woman's eye, in sharp focus and detailed with a bold black eyeliner, reflects the Earth. The vivid colors of blue oceans and green continents stand out against her clear iris, symbolizing a deep introspection or awareness. As time passes, the reflection subtly shifts to include parts of Africa and Europe, suggesting a global perspective. The contrast between her dark eyelashes and light skin accentuates the visual metaphor for unity and interconnectedness, while her gaze suggests   
contemplation on environmental issues or a profound sense of responsibility towards the world. Video Generated by CogVideoX

# J Human Evaluation Details

One hundred meticulously crafted prompts are used for human evaluators, characterized by their broad distribution, clear articulation, and well-defined conceptual scope.

A panel of evaluators is instructed to assign scores for each detail on a scale from zero to one, with the overall total score rated on a scale from 0 to 5, where higher scores reflect better video quality.

To better complement automated evaluation, human evaluation emphasizes the instructionfollowing capability: the total score cannot exceed 2 if the generated video fails to follow the instructions.

Sensory Quality: This part focuses mainly on the perceptual quality of videos, including subject consistency, frame continuity, and stability.

Table 10: Sensory Quality Evaluation Criteria.   

<table><tr><td>Score</td><td>Evaluation Criteria</td></tr><tr><td>1</td><td>High sensory quality: 1. The appearance and morphological features of objects in the video are completely consistent 2. High picture stability, maintaining high resolution consistently 3. Overall composition/color/boundaries match reality 4. The picture is visually appealing</td></tr><tr><td>0.5</td><td>Average sensory quality: 1. The appearance and morphological features of objects in the video are at least 80% consistent 2. Moderate picture stability, with only 50% of the frames maintaining high resolution 3. Overall composi- tion/color/boundaries match reality by at least 70% 4. The picture has some visual appeal</td></tr><tr><td>0</td><td>Poor sensory quality: large inconsistencies in appearance and morphology, low video resolution, and composition/layout not matching reality</td></tr></table>

Instruction Following: This part focuses on whether the generated video aligns with the prompt, including the accuracy of the subject, quantity, elements, and details.

Table 11: Instruction Following Evaluation Criteria.   

<table><tr><td>Score</td><td>Evaluation Criteria</td></tr><tr><td>1</td><td>100% follow the text instruction requirements, including but not limited to: elements completely correct, quantity requirements consistent, elements com- plete, features accurate, etc.</td></tr><tr><td>0.5</td><td>100% follow the text instruction requirements, but the implementation has minor flaws such as distorted main subjects or inaccurate features.</td></tr><tr><td>0</td><td>Does not 100% follow the text instruction requirements, with any of the following issues: 1. Generated elements are inaccurate 2. Quantity is incorrect 3. Elements are incomplete 4. Features are inaccurate</td></tr></table>

Physics Simulation: This part focuses on whether the model can adhere to the objective law of the physical world, such as the lighting effect, interactions between different objects, and the realism of fluid dynamics.

Table 12: Physics Simulation Evaluation Criteria.   

<table><tr><td>Score</td><td>Evaluation Criteria</td></tr><tr><td>1</td><td>Good physical realism simulation capability, can achieve: 1. Real-time tracking 2. Good action understanding, ensuring dynamic realism of entities 3. Realistic lighting and shadow effects, high interaction fidelity 4. Accurate simulation of fluid motion</td></tr><tr><td>0.5</td><td>Average physical realism simulation capability, with some degradation in real- time tracking, dynamic realism, lighting and shadow effects, and fluid motion simulation. Issues include: 1. Slightly unnatural transitions in dynamic effects, with some discontinuities 2. Lighting and shadow effects not matching reality 3. Distorted interactions between objects 4. Floating fluid motion, not matching reality</td></tr><tr><td>0</td><td>Poor physical realism simulation capability, results do not match reality, obvi- ously fake</td></tr></table>

Cover Quality: This part mainly focuses on metrics that can be assessed from single-frame images, including aesthetic quality, clarity, and fidelity.

Table 13: Cover Quality Evaluation Criteria.   

<table><tr><td>Score Evaluation Criteria</td><td></td></tr><tr><td>1</td><td>Image is clear, subject is obvious, display is complete, color tone is normal.</td></tr><tr><td>0.5</td><td>Image quality is average. The subject is relatively complete, color tone is normal.</td></tr><tr><td>0</td><td>Cover image resolution is low, image is blurry.</td></tr></table>

# K Data Filtering Details

In order to obtain high-quality training data, we designed a set of negative labels to filter out low-quality data. Figure 16 presents our negative labels along with sample videos for each label.In table 14, we present the accuracy and recall of our classifier, trained based on video-llama, on the test set ( $1 0 \%$ randomly labeled data).

Table 14: Summary of Classifiers Performance on the Test Set. TP: True Positive, FP: False Positive, TN: True Negative, FN: False Negative.   

<table><tr><td>Classifier</td><td>TP</td><td>FP</td><td>TN</td><td>FN</td><td>Test Acc</td></tr><tr><td>Classifier - Editing</td><td>0.81</td><td>0.02</td><td>0.09</td><td>0.08</td><td>0.91</td></tr><tr><td>Classifier - Static</td><td>0.48</td><td>0.04</td><td>0.44</td><td>0.04</td><td>0.92</td></tr><tr><td>Classifier - Lecture</td><td>0.52</td><td>0.00</td><td>0.47</td><td>0.01</td><td>0.99</td></tr><tr><td>Classifier - Text</td><td>0.60</td><td>0.03</td><td>0.36</td><td>0.02</td><td>0.96</td></tr><tr><td>Classifier - Screenshot</td><td>0.61</td><td>0.01</td><td>0.37</td><td>0.01</td><td>0.98</td></tr><tr><td>Classifier - Low Quality</td><td>0.80</td><td>0.02</td><td>0.09</td><td>0.09</td><td>0.89</td></tr></table>

![](images/34.jpg)  
Figure 16: Examples of negative labels for video filtering.