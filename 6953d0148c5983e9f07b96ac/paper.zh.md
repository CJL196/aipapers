# IP-Adapter：与文本兼容的图像提示适配器，用于文本到图像扩散模型

胡业，张俊，刘思博，韩潇，杨伟 腾讯AI实验室 {huye, junejzhang, siboliu, haroldhan, willyang}@tencent.com

# 摘要

近年来，大型文本到图像扩散模型展现出强大的生成能力，能够创造高保真图像。然而，仅使用文本提示生成所需图像非常棘手，因为这通常涉及复杂的提示工程。图像提示是文本提示的替代方案，正如俗话所说：“一图胜千言”。尽管现有的直接从预训练模型微调的方法有效，但它们需要大量计算资源，并且与其他基础模型、文本提示和结构控制不兼容。本文提出了IP-Adapter，一种有效且轻量的适配器，旨在为预训练的文本到图像扩散模型实现图像提示功能。我们IP-Adapter的关键设计是解耦交叉注意力机制，将文本特征和图像特征的交叉注意力层分开。尽管我们的方法简单，仅包含2200万参数的IP-Adapter也能达到与完全微调的图像提示模型相当甚至更好的性能。由于我们冻结了预训练的扩散模型，所提出的IP-Adapter不仅可以推广到从相同基础模型微调的其他自定义模型，还可以与现有的可控工具结合实现可控生成。得益于解耦交叉注意力策略，图像提示与文本提示也能够很好地结合，实现多模态图像生成。项目页面可访问 https://ip-adapter.github.io。

# 1 引言

Imsb exo GLIDE [1]、DALL-E [2]、Image [3]、Stable diffusion (SD) [4]、eDifI [5] 和 RAPHAEL [6]。用户可以以文本作为输入生成图像。此外，文本可以是长达一千个词。DALL-E 2 [2] 首次尝试将文本变为图像，扩散模型专注于图像生成而非文本生成，且需要先前模型来实现文本到图像的转换。然而，现有的文本到图像扩散模型在生成图像时存在重现文本的挑战。我们的目标是以简单的方式为这些文本到图像扩散模型提供可行的图像提示。我们直接在图像嵌入上对文本条件扩散模型进行训练，达到图像提示的能力。然而，文本到图像的最新研究表明，图像编码器往往不足以保证图像质量，并可能导致泛化问题。

![](images/1.jpg)  
FurVariusimag ynthei with ur propose -Adapter pplinhe preti text--mageu m wi diffeent yle.The ae  thht how esult is ula anipaintg wi a propt, whi he e exampe how he esul  nrollab neratn wie prompt and additional structural conditions.

在现有的图像提示适配器中，以前的工作无法在新的范式下对图像提示进行充分的处理，尤其是在从零开始训练模型的情况下。我们认为，使用现有方法所面临的问题是其在处理图像提示文本时的有效性不够，但有潜力与图像结合。因此，我们提出了一种更有效的图像提示适配器，命名为IP-Adapter，以避免之前方法的缺陷。具体来说，IP-Adapter采用了一种解耦的交叉注意力机制，对文本特征进行处理，以便更好地与图像提示相结合。我们在图1中说明了适配过程的细节。总而言之，我们的贡献如下：我们提出了IP-Adapter，一种轻量级的图像提示适配方法，采用了解耦的交叉注意力策略，适用于现有的文本到图像扩散模型。定量和定性的实验结果显示，约22M参数的小型IP-Adapter在基于图像提示的生成任务中，与完全微调的模型相比具有可比性，甚至更好。我们的IP-Adapter是可重用和灵活的，基于扩散模型进行训练的I-Adapter能够生成其他从同一基础扩散模型微调而来的自定义模型。此外，IP-Adapter与其他可控适配器如ControlNet兼容，便于将图像提示与结构控制相结合。由于解耦的交叉注意力策略，图像提示与文本提示能够兼容，以实现多模态图像生成。

# 2 相关工作

我们在这里回顾关于文本到图像扩散模型的近期工作，以及与大型模型适配器相关的研究。

# 2.1 文本到图像扩散模型

大型文本到图像模型主要分为两类：自回归模型和扩散模型。早期的工作，如 DALLE、CogView 和 Make-A-Scene，属于自回归模型。对于某些变换器，其条件在于文本词元，训练目的是预测图像词元。然而，自回归模型需要大量的参数和计算资源才能生成高质量的图像，正如在 Parti 中所示。

近期，ifsinmodels（DMs）作为文本生成图像的新一代最先进模型，支持$6 \times 6$分辨率的文本到图像生成，及$256 \times 256$分辨率的15亿文本条件上采样扩散模型。DALL-E 2采用文本提示，且不支持外部提示生成，但也可以使用提示。为了通过R-Im [5] 验证该模型的有效性，我们与基于文本的生成模型进行对比，利用多种条件，包括文本、CLIP文本和CLIP图像编码。Diffusion [26] 引入了多模态扩散策略，将文本条件与图像嵌入结合。RAPHAEL将混合专家（MoEs）策略引入文本条件图像扩散模型，以提升图像质量和美学吸引力。此外，还探索了一些基于文本生成模型的工作。

# 2.2 大模型的适配器

请提供清晰的英文文本，以便进行准确的中文翻译。当前文本包含大量错误和不连贯的部分，难以理解其意图。

![](images/2.jpg)  
T    atera added modules (in red color) are trained while the pretrained text-to-image model is frozen.

一种适配器模型，旨在有效适应提示。我们成功开发了一种新的适配器，其简单而有效，超越了之前的适配器方法，甚至可与微调模型相媲美。

# 3 方法

我有关于建议的 IP-Adapter 的动机和设计。

# 3.1 引言

Denoising扩散模型是一类生成模型，它由扩散过程（又称为正向过程）组成，该过程通过一个固定的马尔可夫链在$T$个步骤中逐渐向数据添加高斯噪声，以及一个去噪过程，该过程从带有可学习模型的高斯噪声中生成样本。扩散模型可以表示为$\epsilon _ { \theta }$，它预测噪声，并被定义为变分界限的简化变体：

$$
L _ { \mathrm { s i m p l e } } = \mathbb { E } _ { \mathbf { \boldsymbol { x } } _ { 0 } , \epsilon \sim \mathcal { N } ( \mathbf { \boldsymbol { 0 } } , \mathbf { I } ) , c , t } \| \epsilon - \epsilon _ { \theta } \left( \boldsymbol { x } _ { t } , \boldsymbol { c } , t \right) \| ^ { 2 } ,
$$

其中 $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ 表示具有附加条件 $^ c$ 的真实数据，$t \in [ 0 , T ]$ 表示扩散过程的时间步，${ \pmb x } _ { t } = \alpha _ { t } { \pmb x } _ { 0 } + \sigma _ { t } { \pmb \epsilon }$ 是在 $t$ 步的噪声数据，而 $\alpha _ { t }$ 和 $\sigma _ { t }$ 是预定义的关于 $t$ 的函数，用于确定扩散过程。一旦模型 $\epsilon _ { \theta }$ 被训练完成，图像可以通过随机噪声以迭代方式生成。一般来说，快速采样器如 DDIM [21]、PNDM [36] 和 DPM-Solver [37, 38] 在生成过程中被采用。在训练阶段，对于条件和无条件扩散模型，共同训练通过在训练过程中随机丢弃 $^ c$ 实现。在采样阶段，预测的噪声是基于条件模型 $\mathbf { \epsilon } \epsilon _ { \theta } ( \boldsymbol { x } _ { t } , \boldsymbol { c } , t )$ 和无条件模型 $\epsilon _ { \theta } ( x _ { t } , t )$ 的预测计算得出的：

$$
\hat { \epsilon } _ { \theta } ( x _ { t } , c , t ) = w \epsilon _ { \theta } ( x _ { t } , c , t ) + ( 1 - w ) \epsilon _ { \theta } ( x _ { t } , t ) ,
$$

在这里，$w$ 是指导标度尺度的指导权重，是一个调整生成样本与条件 $c$ 对齐的标量值。在我们的研究中，我们利用开源模型作为我们的示例基础模型来实现 I-AdapteSD 的 LaTeX 射线扩散模型，基于 UNet [40] 以及潜在层。与像 In、SD 等皮尔贝斯扩散模型相比，SD 更有效，因为它是基于从预训练自编码器模型中构建的潜在空间。

# 3.2 图像提示适配器

在本文中，数据被有效嵌入到预训练模型中。大多数方法简单地将拼接的特征输入到交叉注意力层中。我们所提议的架构通过解耦的交叉注意力模块，将图像特征嵌入到预训练的文本到图像生成模型中，如图所示。

# 3.2.1 图像编码器

按照方法论，我们使用预训练的LIP图像编码器模型从潜在的多模态图像-文本对中提取图像特征。我们利用LIP图像编码器生成的全局图像嵌入，该嵌入与其冻结状态良好对齐。将嵌入转换为长度为$N$的特征序列（本研究中我们使用$N=4$），图像特征的维度由线性层和层归一化组成[41]。

# 3.2.2 解耦交叉注意力

交叉注意力层的输出 $\mathbf{Z}^{\prime}$ 可以通过以下公式定义，考虑查询特征 $\mathbf{Z}$ 和文本特征 $c_{t}$。

$$
\mathbf { Z } ^ { \prime } = \operatorname { A t t e n t i o n } ( \mathbf { Q } , \mathbf { K } , \mathbf { V } ) = \operatorname { S o f t m a x } ( { \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d } } } ) \mathbf { V } ,
$$

其中 $\mathbf { Q } = \mathbf { Z } \mathbf { W } _ { q }$，$\mathbf { K } = \pmb { c } _ { t } \mathbf { W } _ { k }$，$\mathbf { V } = { \pmb { c } } _ { t } \mathbf { W } _ { v }$ 分别是注意力操作的查询、键和值矩阵，$\mathbf { W } _ { q }$、$\mathbf { W } _ { k }$、$\mathbf { W } _ { v }$ 是可训练线性投影层的权重矩阵。一个新的跨注意力层被引入到 UNet 模型中，以处理图像特征。给定图像特征 $c _ { i }$，新的跨注意力的输出 $\mathbf { Z } ^ { \prime \prime }$ 的计算如下：

$$
\mathbf { Z } ^ { \prime \prime } = \operatorname { A t t e n t i o n } ( \mathbf { Q } , \mathbf { K } ^ { \prime } , \mathbf { V } ^ { \prime } ) = \operatorname { S o f t m a x } ( \frac { \mathbf { Q } ( \mathbf { K } ^ { \prime } ) ^ { \top } } { \sqrt { d } } ) \mathbf { V } ^ { \prime } ,
$$

其中，$\mathbf { Q } = \mathbf { Z } \mathbf { W } _ { q }$，$\mathbf { K } ^ { \prime } = \pmb { c } _ { i } \mathbf { W } _ { k } ^ { \prime }$，$\mathbf { V } ^ { \prime } = \pmb { c } _ { i } \mathbf { W } _ { v } ^ { \prime }$ 是来自图像特征的查询、键和值矩阵。$\mathbf { W } _ { k } ^ { \prime }$ 和 $\mathbf { W } _ { v } ^ { \prime }$ 是条件权重矩阵。需要注意的是，我们使用与文本交叉注意力相同的查询或交叉注意力。因此，我们只需为每个交叉注意力层添加两个参数 $\mathbf { W } _ { k } ^ { \prime }$ 和 $\mathbf { W } _ { v } ^ { \prime }$。为了加快收敛速度，$\mathbf { W } _ { k } ^ { \prime }$ 和 $\mathbf { W } _ { v } ^ { \prime }$ 从 $\mathbf { W } _ { k }$ 和 $\mathbf { W } _ { v }$ 初始化。然后，我们简单地将输出定义如下：

$$
\begin{array} { r } { \mathbf { Z } ^ { n e w } = \mathrm { S o f t m a x } ( \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d } } ) \mathbf { V } + \mathrm { S o f t m a x } ( \frac { \mathbf { Q } ( \mathbf { K } ^ { \prime } ) ^ { \top } } { \sqrt { d } } ) \mathbf { V } ^ { \prime } } \\ { \mathbf { Q } = \mathbf { Z } \mathbf { W } _ { q } , \mathbf { K } = c _ { t } \mathbf { W } _ { k } , \mathbf { V } = c _ { t } \mathbf { W } _ { v } , \mathbf { K } ^ { \prime } = c _ { i } \mathbf { W } _ { k } ^ { \prime } , \mathbf { V } ^ { \prime } = c _ { i } \mathbf { W } _ { v } ^ { \prime } } \end{array}
$$

由于我们冻结了原始的 UNet 模型，以上解耦交叉注意力中只有 $\mathbf { W } _ { k } ^ { \prime }$ 和 $\mathbf { W } _ { v } ^ { \prime }$ 是可训练的。

# 3.2.3 训练与推断

请提供完整的英文文本以便进行翻译。

$$
L _ { \mathrm { s i m p l e } } = \mathbb { E } _ { \mathbf { { x } _ { 0 } } , \mathbf { { \epsilon } } , \mathbf { { c } } _ { t } , \mathbf { { c } } _ { i } , t } | | \epsilon - \epsilon _ { \theta } \left( \mathbf { { x } } _ { t } , \mathbf { { c } } _ { t } , \mathbf { { c } } _ { i } , t \right) | | ^ { 2 } .
$$

W /'dʌb(ә)lju:/ [计] 等待, 写, 字 [医] 钨(74号元素) Definition: n. the 23rd letter of the Roman alphabet

$$
\hat { \epsilon } _ { \theta } ( x _ { t } , c _ { t } , c _ { i } , t ) = w \epsilon _ { \theta } ( x _ { t } , c _ { t } , c _ { i } , t ) + ( 1 - w ) \epsilon _ { \theta } ( x _ { t } , t )
$$

在这里，如果图像条件被丢弃，我们将 CLIP 图像嵌入置为零。A 在推理阶段：

$$
\mathbf { Z } ^ { n e w } = \operatorname { A t t e n t i o n } ( \mathbf { Q } , \mathbf { K } , \mathbf { V } ) + \lambda \cdot \operatorname { A t t e n t i o n } ( \mathbf { Q } , \mathbf { K } ^ { \prime } , \mathbf { V } ^ { \prime } )
$$

其中 $\lambda$ 是权重因子，当 $\lambda = 0$ 时，模型变为原始的文本到图像扩散模型。

# 4 实验

# 4.1 实验设置

# 4.1.1 训练数据 为了训练 IP-Adapter，我们使用了包含大约 1000 万个文本-图像对的多模态数据，其中包括两个源数据集 - LAION-2B [42] 和 COYO-700M [43]。

# 4.1.2 实现细节

我们的实验基于 SD v $1.5^{2}$，我们使用 OpenCLIP ViT-H/14 [44] 作为图像编码器。模型中的注意力层有 16 个，每层有一个注意力层。所有可训练参数，包括投影网络和适配模块，总数约为 2200 万，使得 P-Adapter 具有相对较低的重量。我们使用 Hugging Face 提供的库 [5] 来实现我们的 P-Adapter，并采用 DeepSpeed 的 ZeRO-2 [13] 进行快速训练。P-Adapter 在一台配有 8 个 V100 GPU 的单机上训练，训练周期为 1 次，批量大小为 8。我们使用 AdamW 优化器 [46]，学习率设置为 0.001，输入图像分辨率为 $512 \times 512$。为了实现无分类器引导，我们使用 0.05 的概率来丢弃文本和图像。在使用图像提示时，文本部分为空，$\lambda = 1.0$。

![](images/3.jpg)  
Figur The isual coparison our proposed I-Adapter with othermethods conditioned n different kins n styles of images.

表格：在COCO验证集上，所提I-Adapter与其他方法的定量比较。最好的结果以粗体显示。

<table><tr><td>Method</td><td>Reusable to custom models</td><td>Compatible with controllable tools</td><td>Multimodal prompts</td><td>Trainable parameters</td><td>CLIP-T↑</td><td>CLIP-I↑</td></tr><tr><td>Training from scratch</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Open unCLIP</td><td></td><td></td><td></td><td>893M</td><td>0.608</td><td>0.858</td></tr><tr><td>Kandinsky-2-1</td><td>×</td><td>×</td><td></td><td>1229M</td><td>0.599</td><td>0.855</td></tr><tr><td>Versatile Diffusion</td><td></td><td>×</td><td>*</td><td>860M</td><td>0.587</td><td>0.830</td></tr><tr><td>Fine-tunining from text-to-image model</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SD Image Variations</td><td></td><td>×</td><td>×</td><td>860M</td><td>0.548</td><td>0.760</td></tr><tr><td>SD unCLIP</td><td>×</td><td></td><td></td><td>870M</td><td>0.584</td><td>0.810</td></tr><tr><td>Adapters</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Uni-ControlNet (Global Control)</td><td></td><td>√</td><td></td><td>47M</td><td>0.506</td><td>0.736</td></tr><tr><td>T2I-Adapter (Style)</td><td>:</td><td>J</td><td>:</td><td>39M</td><td>0.485</td><td>0.648</td></tr><tr><td>ControlNet Shuffle</td><td></td><td>✓</td><td></td><td>361M</td><td>0.421</td><td>0.616</td></tr><tr><td>IP-Adapter</td><td></td><td>✓</td><td>✓</td><td>22M</td><td>0.588</td><td>0.828</td></tr></table>

![](images/4.jpg)  
ure  The eneat mages f dfferet diffusin odes with ur proposed I-Adapte. The I-Adapte trained once.

# 4.2 与现有方法的比较

我们对 DALL-E 2、Kandinsky-2-1 2 以及 Versatile Diffusion [26] 的进行了评估，这些模型是 DALL-E 2 和潜在扩散的混合体。对于微调模型，我们选择了 D 图像变体和 S unCLI。在适配器方面，我们比较了 r IPAapteriyleaptT-Aapter、heobacoUniCnoNetontroeShufonNe 仅参考和 SeeCoder。

![](images/5.jpg)

![](images/6.jpg)  
FurVisualizatio enerate sample wiima propt and ditioal ructural conditions Note ha we don't need fine-tune the IP-Adapter.   

Figure 6: Comparison of our IP-Adapter with other methods on different structural conditions.

![](images/7.jpg)  

Figure 7: Examples of image-to-image and inpainting with image prompt by our IP-Adapter.

# 4.2.1 定量比较

我们使用验证集C2017 [7]，其中包含5,000幅带有标题的图像用于定量评估。对于每个图像，我们计算CLIP-I：生成图像与图像提示在CLIP图像嵌入中的相似度。CLIP-T：生成图像与图像提示标题的CLIPScore [48]。我们在所有生成图像上计算这两个指标的平均值，使用CLIP ViT-L/$1 4^{1}$模型。由于开源 SeeCode 使用了结构控制和控制参考，我们采用了一种具有2200万个参数的架构。

# 4.2.2 定性比较

抱歉，我无法处理该文本。

# 4.3 更多结果

抱歉，我无法处理该输入。请提供更清晰或具体的文本以便翻译。

# 4.3.1 可泛化到自定义模型

A z me f-tune fromDv.like othe dapters (. 控制器。换句话说，一旦 I-Adapte 是一个，我绝对是来自 HuggingFace 模型库的社区模型。真实视觉 4.0，任何 v4 和 ReAnimated。这些模型都是从 SD v1 微调的。如图 4 所示，我们的 IP-Adapter 在这些社区模型上表现良好，因为它们与 SD v1.4 兼容，而 SD v1.5 是基于 SD v1.4 经过更多步骤训练的。

![](images/8.jpg)  

Figure 8: Generated examples of our IP-Adapter with multimodal prompts.

# 4.3.2 结构控制

T2I-Adapter 和 Uni-ControlNet 使用默认的可组合多条件。对于 SeeCoder 和我们的 I-Adapter，我们使用 ControlNet 技术实现结构控制。ControlNet Shuffle 和 ControlNet Reference 也能生成与参考图像更好对齐的图像。

# 4.3.3 图像到图像转换与图像修补

除了文本到图像生成，文本引导图像生成模型还可以通过 DEdit 实现图像到图像的插值和修复。如图 7 所示，我们还可以通过简单地将文本提示替换为图像提示来获得图像引导的图像插值和修复。

# 4.3.4 多模态提示

针对我们模型的实验，我们采用了不同的基线。然而，在所提出的 I-Adapter 的帮助下，我们能够生成包含图像提示和文本提示的多模态提示的图像。我们发现该能力在特定模态下表现得尤为突出。在使用多模态提示的情况下，我们调整了 $\lambda$ 以平衡图像提示和文本提示。图 8 展示了使用 RealisVision 的多模态提示的各种结果，我们需要基于简单文本描述进行初始文本提示的调整。

![](images/9.jpg)  

Figure 9: Comparison with multimodal prompts between our IP-Adapter with other methods.

Welsoar I-Adapter、cdVersa sn、BLIP sn [31]、Uni-ConNet、TI-Adapter、ControlNet Shuffle 和 ControlNet 仅参考的比较结果如图 9 所示。比较结果显示在质量和多模态提示上存在显著的差异。

# 4.4 消融研究

# 4.4.1 解耦交叉注意力的重要性 图10提供了I-Adapter与解耦交叉注意力以及简单ptr的比较示例。一种新的架构生成了更一致的图像，伴随图像提示。

![](images/10.jpg)

![](images/11.jpg)  
u 0:Comparion results fur I-Adapter wit simple adaptr. The decoupled cross-attentio srateg is no used in the simple adapter.   
Tal c  beh -Ape i oal  an e -A with fine-grained features.

# 4.4.2 细粒度特征与全局特征的比较

我优化了使用变换器模型的特征提取器。查询网络的词元特征作为输入传递给交叉注意力层，以便生成额外的人体姿态。

# 5 结论与未来工作

在本研究中，我们提出了一种适配器架构，能够在预训练的文本到图像模型上进行高效的转换。该基本架构的IP-Adapter仅具有2200万个参数，其性能可与某些完全微调的图像提示模型相媲美，甚至优于它们，这展示了它的适用性。更重要的是，图像提示可以与文本提示相结合，实现多模态图像生成。借助现有的方法，文本反演和DreBot等技术。在未来，我们计划开发强大的图像提示适配器，以增强一致性。

# References

[1] Alex Nichol, rafulla Dhariwal, AdityaRamesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Iya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.   
[2 AdityaRamesh, PrafullaDhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditionalage generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.   
[3] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:3647936494, 2022.   
[4] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1068410695, 2022.   
[5] Yogesh Balaji, Seungun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, MikaAittala, TimAila, Sam LaieBryCatazarodiText-o-maiff e wi  eepe er. arXiv preprint arXiv:2211.01324, 2022.   
[] Zeyue Xue, Guanglu Song, Qushan Guo, Boxiao Liu, Zhuofan Zong, Yu Liu, and Png Luo. Raphael: Text-toimage generation via large mixture of diffusion paths. arXiv preprint arXiv:2305.18295, 2023.   
[7] Sam Witteveen and Martin Andrews. Investigating prompt engineering in diffusion models. arXiv preprint arXiv:2211.15462, 2022.   
[8] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, AmandAske Pamela Mishkin, Jack Clark,  al Learni ansrabl isal mode fromatural ane supervision. In International conference on machine learning, pages 87488763. PMLR, 2021.   
[9] Lvmin Zhang and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. arXiv preprint arXiv:2302.05543, 2023.   
10Xingqian Xu, Jiayi Guo, Zhangyang Wang, Gao Huang, Ifan Essa, and Humphrey Shi. Prompt-reediffuson: Taking" text" out of text-to-image diffusion models. arXiv preprint arXiv:2305.16223, 2023.   
11Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023.   
12 Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, and Kwan-Yee K Wong. Uni-controlnet: All-in-one control to text-to-image diffusion models. arXiv preprint arXiv:2305.16322, 2023.   
13 Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pages 8821 8831. PMLR, 2021.   
14] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. Advances in Neural Information Processing Systems, 34:1982219835, 2021.   
[15] Ming Ding, Wendi Zheng, Wenyi Hong, and Jie Tang. Cogview2: Faster and better text-to-image generaton vi hierarchical transformers. Advances in Neural Information Processing Systems, 35:1689016902, 2022.   
[16] Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Make-a-scene: Scene-based text-to-image generation with human priors. In European Conference on Computer Vision, pages 89106. Springer, 2022.   
[ne orol iyal e r processing systems, 30, 2017.   
[18 Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan  Gomez, ukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.   
[9] Jiahui Yu,Yuanhong Xu, Jing uKoh, Than Luon Guna Baid, Zirui Wang, Vijay Vasudevan AlexanrKu, Yini Yang, Burcu KaragolAyan, e al.Scalingautoregressive models or conten-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2022.   
[20] Jascha Sohl-Dickstein, Eric Weiss, Nir Maheswaranathan, and Sury Ganguli. Deep unsupervisedlearnigusing nonequilibrium thermodynamics. In International conference on machine learning, pages 22562265. PMLR, 2015.   
[21] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.   
[22] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.   
[3] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances inneural information processing systems, 34:87808794, 2021.   
[24] Colin Rafel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei ph ex--eTh Machine Learning Research, 21(1):54855551, 2020.   
[25] Wenhu Chen, Hexiang Hu, Chitwan Saharia, and Wiiam W Cohen. Re-imagen: Retrieval-augmented text-toimage generator. arXiv preprint arXiv:2209.14491, 2022.   
[6 Xingn Xu, Zhangyag Wang, EricZhang, Kai Wang and Huprey Shirsi iffusin: Text, ma and variations all in one diffusion model. arXiv preprint arXiv:2211.08332, 2022.   
[Hua henL u e li ZhaondJiZovn image synthesis with composable conditions. arXiv preprint arXiv:2302.09778, 2023.   
[28 Noam Shazer, AzaliMirhoseii, KrzyoMaziar, Andy Davis, Quoc Le, Geoffey Hinton, and Jeff Dean. Outragusly large neural networks: The sparsely-gated mixture-o-experts layer. arXiv preprint arXiv:1701.06538, 2017.   
[9] Willim Fedus, Barret Zoph, and Noam Shazee. witc tranormers: Scalig totrillion paramee odels with simple and efficient sparsity. The Journal of Machine Learning Research, 23(1):52325270, 2022.   
[0 Neil Houlsby,AdeGiurg,Stanis Jastrzebski Brunaorroe, Quentin De Laroue,AdreGeu, MoAttarya, and Sylvain Gelly.Parameter-effciet transerlearning or lp.In Interational Conferec Machine Learning, pages 27902799. PMLR, 2019.   
[ JunanLi, Donxu Li, SilvioSavaree, and Steve Hoi.Bli-2: Boottrappgane-image pre-rai ih frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023.   
2 Dy Zhuunhe Xin i  y 4 inunderstanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.   
[3 R Zhan J Han,Aojun Zhou, Xia Hu, ShilYan, an Lu, HonhegLi, PengGao, andYu Q. LladapterEffcet fe-nnanggeoel withzero-niattentiorXipreri arXiv:303.1199, 2023.   
[4] Peng Gao, Jiami Han, Renr Zhang, Ziyi Lin Shijie Geng, Aojun Zhou, Wi Zhang, Pan Lu, Conhui He, XiYueladapt PrftaltroeriprerXiv:30.01 2023.   
[35] Yan Zeng, Hanbo Zhang, Jiani Zheng, Jiangnan Xia, Guoqiang Wei, Yang Wei, Yuchen Zhang, and Tao Kong. What maters in training a gpt4-style language model with multimodal inputs? arXiv preprint arXiv:2307.02469, 2023.   
[36] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudonumerical methods for diffusion models on maniolds. arXiv preprint arXiv:2202.09778, 2022.   
[Cheng Lu, Yuho Zhou, Fan Bao, Jian Chen, Chonun Li, and Jun Zhu m-solverast de solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35:57755787, 2022.   
[38Cheng Lu, Yuhao Zhou, Fan Bao, Jianei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022.   
[39] Jonathan Ho and Tim Salimans. Classifer-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.   
[0 Ola Rer, il ier, d Thos ox. -ne: ota t o lm - mentation. In Medical Image Computing and Computer-Assisted InterventionMICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part II 18, pages 234241. Springer, 2015.   
[41] Jimmy Lei Ba, Jami Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.   
[42] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:2527825294, 2022.   
[43] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/kakaobrain/coyo-dataset, 2022.   
[44] Gabriel Iharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlii, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip. https://github.com/mlfoundations/open_clip, 2021.   
[45] PatricvonPlaten, SurajPatilAnton Lozhkov, PedroCea, Natha Lambert, Kashi Rasul, MishDav, and Thomas Wolf. Diffusers: State-of-the-art diffusion models. https://github.com/huggingface/ diffusers, 2022.   
[46] Iya Loshchilov and Frank Huter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.   
[47] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, DevaRamanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer VisionECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740755. Springer, 2014.   
[48] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi.Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718, 2021.   
[49] Junan Li, Dongxu Li, Caimig Xiong, and Steven Hoi. Blip: Bootstrappig language-image pre-trainig for unified vision-language understanding and generation. In International Conference on Machine Learning, pages 1288812900. PMLR, 2022.   
[50] Chenlin Meng, Yutong He, Yang Song, Jiamig Song, Jiajun Wu, Jun-Yan Zhu, and Steano Ermo.dedit: Guide image yntheis and editig with stochasti differental equations.arXiv preprint arXiv:2108.01073, 2021.   
[51] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion.arXiv preprint arXiv:2208.01618, 2022.   
[52] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and KfrAberman. Dreamboth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2250022510, 2023.