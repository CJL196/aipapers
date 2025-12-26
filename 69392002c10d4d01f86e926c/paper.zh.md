# KV-Edit：无训练图像编辑以精确保持背景

朱天瑞1\*、张仕义1、邵家伟2、唐艳松1\* 1 清华大学深圳国际研究生院 2 中国电信人工智能研究院 (TeleAI) xilluill070513@gmail.com, sy-zhang23@mails.tsinghua.edu.cn shaojw2@chinatelecom.cn, tang.yansong@sz.tsinghua.edu.cn https://xilluill.github.io/projectpages/KV-Edit/

![](images/1.jpg)  
.

# 摘要

背景一致性仍然是图像编辑任务中的一个重大挑战。尽管已有大量发展，但现有工作在保持与原始图像相似性和生成与目标一致的内容之间仍然面临权衡。在这里，我们提出了KV-Edit，这是一种无训练的方法，利用DiT中的KV缓存来维持背景一致性，在该方法中，背景词元被保留而不是重新生成，从而消除了复杂机制或昂贵训练的需要，最终在用户提供的区域内生成与背景无缝融合的新内容。我们进一步探讨了编辑过程中KV缓存的内存消耗，并使用无倒排方法将空间复杂度优化为$O ( 1 )$。我们的方法与任何基于DiT的生成模型兼容，无需额外训练。实验表明，KV-Edit在背景和图像质量方面显著优于现有方法，甚至超越了基于训练的方法。

# 1. 介绍

最近文本到图像（T2I）生成的进展经历了从 UNet [43] 到 DiT [39] 架构，以及从扩散模型（DMs）[11, 48, 52] 到流式模型（FMs）[1, 23, 64] 的显著转变。基于流的方法，如 Flux [1]，从噪声到图像构建了直接的概率流，从而以更少的采样步骤和减少的训练资源实现更快的生成。DiTs [39] 拥有纯注意力架构，与基于 UNet 的模型相比，展示了更卓越的生成质量和更强的可扩展性。这些 T2I 模型 [1, 14, 42] 还可以促进图像编辑，其中目标图像是基于源图像和修改后的文本提示生成的。在图像编辑领域，早期的工作 [12, 16, 36, 50] 提出了反演去噪范式以生成编辑后的图像，但在修改过程中难以保持背景一致性。一种流行的方法是注意力修改，例如 HeadRouter [57] 修改注意力图，PnP [50] 在去噪过程中注入原始特征，旨在提高与源图像的相似度。然而，改进的相似度与完美的一致性之间仍然存在显著差距，因为很难按照预期控制网络的行为。另一种常见的方法是设计新的采样器 [37, 38] 以减少反演过程中的错误。然而，错误只能减少而不能完全消除，上述无训练的方法仍需要针对不同情况进行广泛的超参数调优。同时，令人兴奋的基于训练的图像修复方法 [26, 65] 可以保持背景一致性，但面临着昂贵的训练成本和潜在的质量下降。为了解决上述所有限制，我们提出了一种新的无训练方法，在编辑过程中保持背景一致性。我们不依赖常规的注意力修改或新的反演采样器以获得相似结果，而是在 DiTs [39] 中实现 KV 缓存，以在反演过程中保留背景标记的键值对，并选择性地重构仅编辑区域。我们的方法首先使用掩码解耦背景和前景区域之间的注意力，然后将图像反演为噪声空间，同时在每个时间步和注意力层缓存背景标记的 KV 值。在随后的去噪过程中，仅处理前景标记，同时将其键和值与缓存的背景信息进行拼接。有效地，我们引导生成模型与背景保持新内容的连续性，并使背景内容与输入保持一致。我们称这种方法为 KV-Edit。为了进一步增强我们方法的实用性，我们进行了移除场景的分析。这个挑战源于周围标记和对象本身的残余信息，有时与编辑指令发生冲突。为了解决这个问题，我们引入了掩码引导的反演和重新初始化策略，作为反演和去噪的两种增强技术。这些方法进一步扰动存储在周围标记和自我标记中的信息，从而更好地与文本提示保持一致。此外，我们将 KV-Edit 应用于无反演方法 [23, 56]，该方法不再在所有时间步骤缓存键值对，而是在一个步骤之后立即使用 KV，显著减少了 KV 缓存的内存消耗。总之，我们的主要贡献包括：1）一种新的无训练编辑方法，在 DiTs 中实现 KV 缓存，确保在编辑过程中背景的一致性，同时进行最小的超参数调优。2）掩码引导的反演和重新初始化策略，扩展了该方法在各种编辑任务中的适用性，为不同用户需求提供灵活选择。3）使用无反演方法优化我们方法的内存开销，增强其在 PC 上的实用性。4）实验验证表明，在保持生成质量与直接 T2I 合成相当的同时，实现完美的背景保留。

# 2. 相关工作

# 2.1. 文本引导编辑

图像编辑方法大致可以分为基于训练的方法和无训练的方法。基于训练的方法通过对文本-图像对进行微调预训练的生成模型，展示了令人印象深刻的编辑能力，实现了可控的修改。无训练的方法则作为一种灵活的替代方案，开创性工作建立了两阶段的反演-去噪范式。在这些方法中，注意力修改已成为一种普遍的技术，特别是 Add-it 将特征从反演传播到去噪过程，以在编辑过程中保持源图像的相似性。一些其他工作集中于更好的反演采样器，例如 RF-solver 设计了二阶采样器。与我们的方法最相似的几种方法尝试通过在特定时间步骤使用遮罩将源图像和目标图像混合，从而保留背景元素。普遍共识是，准确的遮罩对提高质量至关重要，其中用户提供的输入和分割模型被证明是比较有效的选择，优于 UNet 中从注意力层派生的遮罩。然而，上述方法在编辑过程中常常遇到失败案例，并且难以保持完美的背景一致性，而基于训练的方法则面临额外的计算开销挑战。

![](images/2.jpg)  
Here, $\mathbf { x }$ and $\mathbf { z }$ denote intermediate results in inversion and denoising processes respectively. Starting from $\mathbf { x } _ { 0 }$ , we first perform inversion to obtain predicted noise $\mathbf { x } _ { N }$ while caching KV pairs. Then, we choose the input ${ \bf z } _ { N } ^ { f g }$ and generate edited foreground content $\mathbf { z } _ { 0 } ^ { f g }$ based on a new prompt. Finally, we concatenate it with the original background $\mathbf { x } _ { 0 } ^ { b g }$ to obtain the edited image with preserved background.

# 2.2. 注意力模型中的KV缓存

KV 缓存是一种广泛采用的优化技术，应用于大型语言模型（LLMs）中，以提高自回归生成的效率。在因果注意力机制中，由于在生成过程中键（keys）和值（values）保持不变，重新计算它们会导致冗余的资源消耗。KV 缓存通过存储这些中间结果来解决这一问题，使模型在推理过程中能够重用先前标记的键值对。这项技术已成功应用于 LLMs 和视觉语言模型（VLMs）中。然而，在图像生成和编辑任务中尚未探索此技术，主要原因是通常假设图像标记需要双向注意力。

# 3. 方法

在这一部分，我们首先分析了反演去噪范式[12, 16]在背景保留方面面临的挑战。然后，我们介绍了提出的KV-Edit方法，该方法在编辑过程中根据掩膜严格保持背景区域。最后，我们呈现了两种可选的增强技术和一个无反演版本，以提高我们的方法在各种场景下的可用性。

# 3.1. 准备工作

确定性扩散模型如 DDIM [46] 和流匹配 [28] 可以通过常微分方程（ODE） [47] 来建模，以描述从噪声分布到真实分布的概率流路径。该模型学习预测将高斯噪声转化为有意义图像的速度向量。在去噪过程中，$\mathbf{x_{1}}$ 代表噪声，$\mathbf{x_{0}}$ 是最终图像，$\mathbf{x_{t}}$ 代表中间结果。

$$
d \mathbf { x } _ { t } = \left( f ( \mathbf { x } _ { t } , t ) - \frac { 1 } { 2 } g ^ { 2 } ( t ) \nabla _ { \mathbf { x } _ { t } } \log p ( \mathbf { x } _ { t } ) \right) d t , t \in [ 0 , 1 ] .
$$

其中 $\mathbf{s}_{\theta}(\mathbf{x}, t) \sim \nabla_{\mathbf{x}_{t}} \log p(\mathbf{x}_{t})$ 由网络预测。DDIM [46] 和流匹配 [28] 都可以视为该常微分方程函数的特例。通过设置 $f(\mathbf{x}_{t}, t) = \begin{array}{r} \frac{\mathbf{x}_{t}}{\overline{\mathbf{x}}_{t}} \frac{d\overline{\alpha}_{t}}{dt}, g^{2}(t) = 2\overline{\alpha}_{t}\overline{\beta}_{t} \frac{d}{dt} \left(\frac{\overline{\beta}_{t}}{\overline{\alpha}_{t}}\right) \end{array}$ 和 $s_{\theta}(\mathbf{x}, t) = -\theta(, t)$，我们得到了 DDIM 的离散化形式：

$$
\mathbf { x } _ { t - 1 } = \bar { \alpha } _ { t - 1 } \left( \frac { \mathbf { x } _ { t } - \bar { \beta } _ { t } \epsilon _ { \theta } ( \mathbf { x } _ { t } , t ) } { \bar { \alpha } _ { t } } \right) + \bar { \beta } _ { t - 1 } \epsilon _ { \theta } ( \mathbf { x } _ { t } , t )
$$

ODE中的正向和反向过程都遵循方程(1)，描述了从高斯分布到真实分布的可逆路径。在图像编辑过程中，该ODE建立了噪声与真实图像之间的映射，其中噪声可以被视为图像的嵌入，携带着关于结构、语义和外观的信息。最近，Rectified Flow构建了一条噪声分布与真实分布之间的直线，训练模型以拟合速度场$\mathbf { v } _ { \theta } ( \mathbf { x } , t )$。这个过程可以简单地用ODE描述：

$$
\begin{array} { r } { d { \mathbf { x } } _ { t } = { \mathbf { v } } _ { \theta } ( { \mathbf { x } } , t ) d t , t \in [ 0 , 1 ] . } \end{array}
$$

![](images/3.jpg)  
Figure 3. The reconstruction error in the inversionreconstruction process. Starting from the original image $\mathbf { x } _ { t _ { 0 } }$ , the inversion process proceeds to $\mathbf { x } _ { t _ { N } }$ . During inversion process, we use intermediate images $\mathbf { x } _ { t _ { i } }$ to reconstruct the original image and calculate the MSE between the reconstructed image $\mathbf { x } _ { t _ { 0 } } ^ { \prime }$ and the original image $\mathbf { x } _ { t _ { 0 } }$ .

由于常微分方程的可逆性，基于流的模型还可以通过反演和去噪在比 DDIM [46] 更少的时间步内用于图像编辑。

# 3.2. 重新思考反演去噪范式

反演去噪范式将图像编辑视为生成模型的一种固有能力，无需额外训练，能够生成语义上不同但视觉上相似的图像。然而，实证观察表明，该范式在内容上只实现了相似性，而不是完美一致性，留下了与用户期望之间的显著差距。本节将分析导致这一问题的三个因素。

以校正流（Rectified Flow）作为例子，基于公式 (3)，我们可以推导出反演和去噪的离散化实现。模型以原始图像 $\mathbf { x } _ { t _ { 0 } }$ 和高斯噪声 $\mathbf { x } _ { t _ { N } } \in \mathcal { N } ( 0 , I )$ 作为路径端点。给定离散时间步 $t = \{ t _ { N } , . . . , t _ { 0 } \}$，模型预测 ${ \pmb v } _ { \theta } ( C , { \bf x } _ { t _ { i } } , t _ { i } ) , i \in \{ N , \cdot \cdot \cdot , 1 \}$，其中 $\mathbf { x } _ { t _ { i } }$ 和 $\mathbf { z } _ { t _ { i } }$ 分别表示反演和去噪中的中间状态，如下方公式所述：

$$
\begin{array} { r } { \mathbf { x } _ { t _ { i } } = \mathbf { x } _ { t _ { i - 1 } } + ( t _ { i } - t _ { i - 1 } ) \pmb { v } _ { \theta } ( C , \mathbf { x } _ { t _ { i } } , t _ { i } ) } \\ { \mathbf { z } _ { t _ { i - 1 } } = \mathbf { z } _ { t _ { i } } + ( t _ { i - 1 } - t _ { i } ) \pmb { v } _ { \theta } ( C , \mathbf { z } _ { t _ { i } } , t _ { i } ) } \end{array}
$$

理想情况下，$\mathbf { z } _ { t _ { 0 } }$ 应该是与 $\mathbf { x } _ { t _ { 0 } }$ 相同的识别形态，当直接从 $\mathbf { x } _ { t _ { N } }$ 重建时。然而，由于反演过程中的离散化和因果性，我们只能使用 $v _ { \theta } ( C , \mathbf { X } _ { t _ { t - 1 } } , t _ { t - 1 } ) \approx v _ { \theta } ( C , \mathbf { X } _ { t _ { i } } , t _ { i } )$ 进行估计，这会引入累积误差。图 3 显示，在固定的时间步数 $N$ 下，误差累积随着反演时间步趋近 $t _ { N }$ 而增加，阻止了精确重建。此外，一致性受到条件的影响。我们可以表示前景的 $\mathbf { z } _ { t _ { 0 } } ^ { f g }$ 和我们希望保留的区域的 $\mathbf { z } _ { t _ { 0 } } ^ { b g }$，其中 "fg" 和 "bg" 分别代表前景和背景。基于这些定义，背景去噪过程为：

![](images/4.jpg)  
Figure 4. Analysis of factors affecting background changes. The four images on the right demonstrate how foreground content and condition changes influence the final results.

$$
{ \pmb v } _ { \theta } ( C , { \bf z } _ { t _ { i } } , t _ { i } ) = { \pmb v } _ { \theta } ( C , { \bf z } _ { t _ { i } } ^ { f g } , { \bf z } _ { t _ { i } } ^ { b g } , t _ { i } )
$$

$$
\mathbf { z } _ { t _ { i - 1 } } ^ { b g } = \mathbf { z } _ { t _ { i } } ^ { b g } + ( t _ { i - 1 } - t _ { i } ) \pmb { v } _ { \theta } ( C , \mathbf { z } _ { t _ { i } } ^ { f g } , \mathbf { z } _ { t _ { i } } ^ { b g } , t _ { i } )
$$

根据这些公式，当生成编辑的重构时，背景区域会在仅修改提示或前景噪声时发生变化。总之，不可控的背景变化可以归因于三个因素：误差积累、新条件和新前景内容。实际上，任何单一元素都会同时触发这三种效应。因此，本文将提出一个优雅的解决方案，以同时解决所有这些问题。

# 3.3. 注意力解耦

传统的反演去噪范式在去噪过程中同时处理背景和前景区域，导致在前景和条件修改时出现不希望的背景变化。经过深入分析，我们观察到在UNet [43]架构中，广泛的卷积网络导致背景和前景信息的融合，使得二者无法分开。然而，在DiT [39]中，主要依赖于注意力模块 [51]，使我们能够仅使用前景词元作为查询，分别生成前景内容，然后与背景结合。此外，直接生成前景词元通常会导致相对于背景的不连续或不正确内容。因此，我们提出了一种新的注意力机制，其中查询仅包含前景信息，而键和值则同时包含前景和背景信息。

![](images/5.jpg)  
Figure 5. Demonstration of inversion-free KV-Edit. The right panel shows three comparative cases including a failure case, while the left panel illustrates inversion-free approach Significantly optimizes the space complexity to $O ( 1 )$ .

# 算法 1 反演过程中的简化键值缓存

1: 输入：$t _ { i }$，图像 $\boldsymbol { x } _ { t _ { i } }$，$M$ 层块 $\{ l _ { j } \} _ { j = 1 } ^ { M }$，前景区域掩码，KV 缓存 $C$ 2: 输出：预测向量 $V _ { \theta t _ { i } }$，$\mathrm { K V }$ 缓存 $C$ 3: 对于 $j = 0$ 到 $M$，执行 4: $\begin{array} { r l } & { Q , K , V = W _ { Q } ( x _ { t _ { i } } ) , W _ { K } ( x _ { t _ { i } } ) , W _ { V } ( x _ { t _ { i } } ) } \\ & { K _ { i j } ^ { b g } , V _ { i j } ^ { b g } = K [ 1 - m a s k > 0 ] , V [ 1 - m a s k > 0 ] } \\ & { C \gets \mathrm { A p p e n d } ( K _ { i j } ^ { b g } , V _ { i j } ^ { b g } ) } \\ & { x _ { t _ { i } } = x _ { t _ { i } } + \mathrm { A t t n } ( Q , K , V ) } \end{array}$ 5: 6: 7: 8: 结束循环 9: $V _ { \theta t _ { i } } = \mathbf { M L P } ( x _ { t _ { i } } , t _ { i } )$ 10: 返回 $V _ { \theta t _ { i } }$，$C$ 背景信息。排除文本词元，图像模态自注意力计算可以表示为：

$$
\operatorname { A t t } ( \mathbf { Q } ^ { f g } , ( \mathbf { K } ^ { f g } , \mathbf { K } ^ { b g } ) , ( \mathbf { V } ^ { f g } , \mathbf { V } ^ { b g } ) ) = \mathcal { S } ( \frac { \mathbf { Q } ^ { f g } \mathbf { K } ^ { T } } { \sqrt { d } } ) \mathbf { V }
$$

其中 $\mathbf { Q } ^ { fg }$ 表示仅包含前景标记的查询，$ ( { \bf K } ^ { fg } , { \bf K } ^ { bg } ) $ 和 $ ( { \bf V } ^ { fg } , { \bf V } ^ { bg } ) $ 表示前景和背景键值的按正确顺序的连接（相当于完整图像的键和值），而 $ s $ 表示 softmax 操作。值得注意的是，与传统的注意力计算相比，公式 (8) 仅修改查询组件，这相当于在注意力层的输入和输出中进行裁剪，确保生成内容与背景区域的无缝融合。

# 算法 2 去噪过程中的简化 KV 缓存

输入：$t _ { i }$，前景 $z _ { t _ { i } } ^ { f g }$，$M$层块 $\{ l _ { j } \} _ { j = 1 } ^ { M }$，KV缓存 $C$

预测向量 $V _ { \theta t _ { i } } ^ { f g }$ 3: 对 $j = 0$ 到 $M$ 进行迭代 4: $Q ^ { f g } , K ^ { f g } , V ^ { f g } = W _ { Q } ( z _ { t _ { i } } ^ { f g } ) , W _ { K } ( z _ { t _ { i } } ^ { f g } ) , W _ { V } ( z _ { t _ { i } } ^ { f g } )$ 5: $K _ { i j } ^ { b g } , V _ { i j } ^ { b g } = C _ { K } [ i , j ] , C _ { V } [ i , j ]$ 6: $\bar { K , V } = \mathrm { C o n c a t } ( K _ { i j } ^ { b g } , K ^ { f g } ) , \mathrm { C o n c a t } ( V _ { i j } ^ { b g } , V ^ { f g } )$ 7: $z _ { t _ { i } } ^ { f g } = z _ { t _ { i } } ^ { f g } + \mathrm { A t t n } ( \stackrel { \cdot } { Q } ^ { f g } , K , V )$ 8: 结束迭代 9: $V _ { \theta t _ { i } } ^ { f g } = \mathbf { M } \mathbf { L } \mathbf { P } ( z _ { t _ { i } } ^ { f g } , t _ { i } )$ 10: 返回 $V _ { f g } ^ { \theta t _ { i }}$

# 3.4. KV-编辑

基于公式 (8)，实现背景保留的前景编辑需要为背景提供适当的键值对。我们核心的洞察是，背景词元的键和值反映了它们从图像到噪声的确定性路径。因此，我们在反演过程中实现了 KV 缓存，如算法 1 所述。该方法在概率流路径上的每个时间步和块层记录键和值，这些记录随后在去噪过程中使用，如算法 2 所示。我们将这个完整的流程称为“KV-Edit”，如图 2 所示，其中“KV”表示 KV 缓存。与其他注意力注入方法 [4, 49, 50] 不同，KV-Edit 仅在重新生成前景词元时重用背景词元的 KV，而无需指定特定的注意力层或时间步。我们不是将源图像用作注入信息，而是将确定性的背景视为上下文，将前景视为内容以继续生成，类似于大语言模型中的 KV 缓存。由于背景词元是保留的而非重新生成的，KV-Edit 确保背景的一致性，从而有效绕过第 3.2 节中讨论的三个影响因素。以往的研究 [9, 12, 16] 在使用图像描述作为指导时，经常在对象移除任务中失败，因为原始对象仍与目标提示对齐。通过我们的深入分析，我们发现这个问题源于原始对象的残余信息，这些信息在其自身的词元中持续存在，并通过注意力机制传播到周围词元，最终导致模型重构原始内容。为了解决移除对象的挑战，我们引入了两种增强技术。首先，在反演后，我们用融合噪声替换 $\mathbf { z } _ { t _ { N } }$，即 $\mathbf { z } _ { t _ { N } } ^ { \prime } = \mathrm { n o i s e } { \cdot } t _ { N } { + } \mathbf { z } _ { t _ { N } } { \cdot } ( 1 { - } t _ { N } )$，以干扰原始内容信息。其次，我们在反演过程中加入了一种注意力掩码，如图 2 所示，以防止前景内容被纳入 KV 值，进一步减少原始内容的保留。这些技术作为可选增强措施，以改善不同场景下的编辑能力和性能，如图 1 所示。

![](images/6.jpg)

# 3.5. 内存高效实现

基于反演的方法需要在 N 个时间步长内存储键值对，这在处理大规模生成模型（例如，120亿参数 [1]）时可能会对个人电脑造成显著的内存限制。幸运的是，受到 [23, 56] 的启发，我们探索了一种无反演的方法。该方法在每个反演步骤后立即进行去噪，计算两次结果之间的向量差，以推导出在 $t _ { 0 }$ 空间中的概率流路径。这种方法在使用后允许立即释放 KV 缓存，将内存复杂度从 $O ( N )$ 降低到 $O ( 1 )$。然而，无反演方法有时可能会导致内容保留伪影，如图 5 和 FlowEdit [23] 所示。由于我们主要关注编辑过程中的背景保留，我们将在补充材料中对无反演进行更多讨论。

# 4. 实验

# 4.1. 实验设置

基准方法。我们将我们的方法与两类方法进行比较：（1）无训练方法，包括基于 DDIM 的 P2P [16]、MasaCtrl [9]，以及基于校正流的 RFEdit [53] 和 RF-Inversion [44]；（2）基于训练的方法，包括基于 DDIM 的 BrushEdit [26] 和基于校正流的 FLUX-Fill [1]。总的来说，我们评估了六种流行的图像编辑和修复方法。数据集。我们在 PIE-Bench [21] 的九个任务上评估我们的方法和基准，包括 620 张带有相应掩码和文本提示的图像。遵循 [26, 56] 的方法，我们排除了 PIE-Bench [21] 中的风格迁移任务，因为我们主要关注在语义编辑任务中如对象添加、移除和更改的背景保留。

Table 1. Comparison with previous methods on PIE-Bench. $\mathrm { V A E ^ { * } }$ denotes the inherent reconstruction error through direct VAE Bold and underlined values denote the best and second-best results respectively.   

<table><tr><td rowspan="2">Method</td><td colspan="2">Image Quality</td><td colspan="3">Masked Region Preservation</td><td colspan="2">Text Align</td></tr><tr><td>HPS×102 ↑</td><td>AS ↑</td><td>PSNR ↑</td><td>LPIPS×103 ↓</td><td>MSE×104 ↓</td><td>CLIP Sim ↑</td><td>IR×10 ↑</td></tr><tr><td>VAE*</td><td>24.93</td><td>6.37</td><td>37.65</td><td>7.93</td><td>3.86</td><td>19.69</td><td>-3.65</td></tr><tr><td>P2P [16]</td><td>25.40</td><td>6.27</td><td>17.86</td><td>208.43</td><td>219.22</td><td>22.24</td><td>0.017</td></tr><tr><td>MasaCtrl [9]</td><td>23.46</td><td>5.91</td><td>22.20</td><td>105.74</td><td>86.15</td><td>20.83</td><td>-1.66</td></tr><tr><td>RF Inv. [44]</td><td>27.99</td><td>6.74</td><td>20.20</td><td>179.73</td><td>139.85</td><td>21.71</td><td>4.34</td></tr><tr><td>RF Edit [53]</td><td>27.60</td><td>6.56</td><td>24.44</td><td>113.20</td><td>56.26</td><td>22.08</td><td>5.18</td></tr><tr><td>BrushEdit [26]</td><td>25.81</td><td>6.17</td><td>32.16</td><td>17.22</td><td>8.46</td><td>22.44</td><td>3.33</td></tr><tr><td>FLUX Fill [1]</td><td>25.76</td><td>6.31</td><td>32.53</td><td>25.59</td><td>8.55</td><td>22.40</td><td>5.71</td></tr><tr><td>Ours</td><td>27.21</td><td>6.49</td><td>35.87</td><td>9.92</td><td>4.69</td><td>22.39</td><td>5.63</td></tr><tr><td>+NS+RI</td><td>28.05</td><td>6.40</td><td>33.30</td><td>14.80</td><td>7.45</td><td>23.62</td><td>9.15</td></tr></table>

Table 2. Ablation study for object removal task. CLIP $\sin ^ { * }$ and $\mathrm { I R } ^ { * }$ represent alignment between source prompt and new image through CLIP [40] and Image Reward [55] to evaluate whether remove particular object from image. NS indicates there is no skip step during inversion. RI indicates the addition of reinitialization strategy. AM indicates that using attention mask during inversion.   

<table><tr><td rowspan="2">Method</td><td colspan="2">Image Quality</td></tr><tr><td>|HPS ×102 ↑ AS ↑|</td><td>Text Align |CLIP Sim ↓IR×10 *</td></tr><tr><td>KV Edit (ours)</td><td>26.76 6.49</td><td>25.50 6.87</td></tr><tr><td>+NS</td><td>26.93 6.37</td><td>25.05 3.17</td></tr><tr><td>+NS+AM</td><td>26.72 6.35</td><td>25.00 2.55</td></tr><tr><td>+NS+RI</td><td>26.73 6.34</td><td>24.82 0.22</td></tr><tr><td>+NS+AM+RI</td><td>26.51 6.28</td><td>24.90 0.90</td></tr></table>

实现细节。我们基于 FLUX.1-[dev] [1] 实现了我们的方法，遵循与其他基于校正流的方法 [23, 44, 53] 相同的框架。我们保持与 FlowEdit [23] 一致的超参数，使用总共 28 个时间步，跳过最后 4 个时间步（$N = 2 4$），以减少累积误差，并为反演和去噪过程分别设置引导值为 1.5 和 5.5。表格和图表中的 NS 代表不跳过步骤 $N = 2 8$。其他基线保持默认参数或使用先前发布的结果。除非另有说明，表格中的 "Ours" 指的是不包含第 3.4 节中提出的两种可选增强技术的基于反演的 KV-Edit。所有实验均在两台具有 24GB 显存的 NVIDIA 3090 GPU 上进行。指标 根据 [20, 21, 26]，我们使用七个指标跨越三个维度来评估我们的方法。在图像质量方面，我们报告 HPSv2 [63] 和美学评分 [45]。在背景保留方面，我们测量 PSNR [19]、LPIPS [63] 和 MSE。在文本-图像对齐方面，我们报告 CLIP 分数 [40] 和图像奖励 [55]。值得注意的是，虽然图像奖励以前用于质量评估，但我们发现它在测量文本-图像对齐方面特别有效，为未编辑图像提供负分。基于这一观察，我们还利用图像奖励来评估成功移除对象的情况。

# 4.2. 编辑结果

我们在 PIE-Bench [21] 上进行实验，将编辑任务分为三大类：移除、添加和更改对象。对于实际应用，这些任务优先考虑背景保留和文本对齐，其次是整体图像质量评估。

定量比较。第4节呈现了定量结果，包括基线、我们的方法和采用重新初始化策略的我们的方法。我们排除了使用注意力掩码策略的结果，因为该策略仅在特定情况下显示出改进。我们的方法在掩蔽区域保留指标上超过所有其他方法。值得注意的是，如图6所示，PSNR低于30的方法未能维持背景一致性，所产生的结果仅仅类似于原图。尽管RF-Inversion [44]在图像质量评分上表现良好，但生成的背景却完全不同。我们的方法达到了第三好的图像质量，其质量高于原始图像，并且完美保留了背景。同时，通过重新初始化过程，我们获得了最佳文本对齐得分，因为注入的噪声扰乱了原始内容，在某些情况下（例如，物体移除和颜色改变）使得编辑更加有效。即使与基于训练的修补方法 [1, 26] 相比，我们的方法在遵循用户意图的同时，更好地保留了背景。

![](images/7.jpg)  
Figure 7. Ablation study of different optional strategies on object removal task. From left to right, applying more strategies leads to stronger removal effect and the right is the best.

定性比较。图6展示了我们的方法在三种不同任务上相较于以往工作的表现。对于去除任务，所示的示例需要第3.4节中提出的两种增强技术。以往的无训练方法未能保留背景，特别是Flow-Edit [23]，其本质上生成新图像，尽管画质较高。有趣的是，基于训练的方法如BrushEdit [26]和FLUXFill [1]在某些情况下表现出显著的现象（图6中的第一和第三行）。BrushEdit [26]可能受到生成模型能力的限制，产生无意义的内容。FLUX-Fill [1]有时错误解读文本提示，生成不合理的内容，例如重复的主体。相比之下，我们的方法展示了令人满意的结果，成功生成了与文本对齐的内容，同时保留了背景，消除了背景保留与前景编辑之间的传统权衡。

# 4.3. 消融研究

我们进行了消融研究，以说明第3.4节中提出的两种增强策略以及无跳过步骤对我们方法的物体移除性能的影响。表2展示了在图像质量和文本对齐评分方面的结果。值得注意的是，对于文本对齐评估，我们使用CLIP [40]和Image Reward [55]模型计算生成结果与原始提示之间的相似度。该指标在移除任务中表现得更具区分性，因为最终图像中特定物体的残留显著提高了相似度评分。如表2所示，无跳过（NS）和再初始化（RI）的组合达到了最佳的文本对齐评分。然而，我们观察到在加入这些组件后，图像质量指标略有下降。我们认为这一现象的原因在于基准中存在过大的遮罩，无跳过、再初始化和注意力遮罩共同破坏了大量信息，导致生成图像中出现一些不连续性。因此，这些策略应被视为编辑效果的可选增强，而非适用于所有场景的通用解决方案。

Table 3. User Study. We compared our method with four popular baselines. Participants were asked to choose their preferred option or indicate if both methods were equally good or not good based on four criteria. We report the win rates of our method compared to baseline excluding equally good or not good instances. Random\* denotes the win rate of random choices.   

<table><tr><td>ours vs.</td><td>Quality↑</td><td>Background↑</td><td>Text↑</td><td>Overall↑</td></tr><tr><td>Random*</td><td>50.0%</td><td>50.0%</td><td>50.0%</td><td>50.0%</td></tr><tr><td>RF Inv. [44]</td><td>61.8%</td><td>94.8%</td><td>79.6%</td><td>85.1%</td></tr><tr><td>RF Edit [53]</td><td>54.5%</td><td>90.5%</td><td>75.0%</td><td>73.6%</td></tr><tr><td>BrushEdit [26]</td><td>71.8%</td><td>66.7%</td><td>68.7%</td><td>70.2%</td></tr><tr><td>FLUX Fill [1]</td><td>60.0%</td><td>53.7%</td><td>58.6%</td><td>61.9%</td></tr></table>

图 7 可视化了这些策略的影响。在大多数情况下，仅重新初始化就足以实现预期结果，而少部分情况下则需要额外的注意力掩蔽以提升性能。

# 4.4. 用户研究

我们进行了一项广泛的用户研究，以比较我们的方法与四个基线，包括无训练方法 RFEdit [53]、RF-Inversion [44]，以及基于训练的方法 BrushEdit [26] 和 Flux-Fill [1]。我们使用了来自 PIE-Bench [21] “随机类”的 110 张图像（排除了风格迁移任务、没有背景的图像和争议内容）。超过 20 名参与者被要求根据四个标准对每一对方法进行比较：图像质量、背景保留、文本对齐以及整体满意度。如表 3 所示，我们的方法显著优于以前的方法，甚至超越了 Flux-Fill [1]，后者是 FLUX [1] 的官方图像修复模型。此外，用户反馈显示，背景保留在最终选择中起着至关重要的作用，即使 RF-Edit [53] 在图像质量上表现优异，但最终在满意度比较中却未能获胜。

# 5. 结论

在本文中，我们介绍了KV-Edit，一种新的无训练方法，通过缓存和重用背景键值对实现了图像编辑中的完美背景保留。我们的方法通过DiT中的注意力机制有效地将前景编辑与背景保留解耦，同时可选的增强策略和内存高效的实现进一步提高了其实际效用。大量实验表明，我们的方法在背景保留和图像质量方面均优于无训练方法和基于训练的修补模型。此外，我们希望这一直观而有效的机制能够激发更广泛的应用，例如视频编辑、多概念个性化和其他场景。

# References

[1] Flux. https://github.com/black-forestlabs/flux/. 2, 3, 6, 7, 8, 12, 13, 14   
[2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 3   
[3] Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended latent diffusion. TOG, 42(4):111, 2023. 2   
[4] Omri Avrahami, Or Patashnik, Ohad Fried, Egor Nemchinov, Kfir Aberman, Dani Lischinski, and Daniel CohenOr.Stable flow: Vital layers for training-free image editing. arXiv preprint arXiv:2411.14430, 2024. 2, 5, 12   
[5] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 3   
[6] Sule Bai, Yong Liu, Yifei Han, Haoji Zhang, and Yansong Tang. Self-calibrated clip for training-free open-vocabulary segmentation. arXiv preprint arXiv:2411.15869, 2024. 3   
[7] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, pages 1839218402, 2023. 2, 3   
[8] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. NeurIPS, 33:1877 1901, 2020. 3   
[9] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In ICCV, pages 2256022570, 2023. 2, 5, 6, 7   
10] Zhennan Chen, Yajie Li, Haofan Wang, Zhibo Chen, Zhengkai Jiang, Jun Li, Qian Wang, Jian Yang, and Ying Tai. Region-aware text-to-image generation via hard binding and soft refinement. arXiv preprint arXiv:2411.06558, 2024. 2   
11] Wenxun Dai, Ling-Hao Chen, Jingbo Wang, Jinpeng Liu, Bo Dai, and Yansong Tang. Motionlcm: Real-time controllable motion generation via latent consistency model. In ECCV, pages 390408, 2024. 2   
12] Wenkai Dong, Song Xue, Xiaoyue Duan, and Shumin Han. Prompt tuning inversion for text-driven image editing using diffusion models. In ICCV, pages 74307440, 2023. 2, 3, 5, 14   
[13] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 3   
[14] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. 2   
[15] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, pages 1600016009, 2022. 3   
[16] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 2, 3, 5, 6, 7, 14   
[17] Wenke Huang, Jian Liang, Zekun Shi, Didi Zhu, Guancheng Wan, He Li, Bo Du, Dacheng Tao, and Mang Ye. Learn from downstream and be yourself in multimodal large language model fine-tuning. arXiv preprint arXiv:2411.10928, 2024. 3   
[18] Xiaoke Huang, Jianfeng Wang, Yansong Tang, Zheng Zhang, Han Hu, Jiwen Lu, Lijuan Wang, and Zicheng Liu. Segment and caption anything. In CVPR, pages 13405 13417, 2024. 3   
[19] Quan Huynh-Thu and Mohammed Ghanbari. Scope of validity of psnr in image/video quality assessment. Electronics letters, 44(13):800801, 2008. 7   
[20] Xuan Ju, Xian Liu, Xintao Wang, Yuxuan Bian, Ying Shan, and Qiang Xu. Brushnet: A plug-and-play image inpainting model with decomposed dual-branch diffusion. In ECCV, pages 150168, 2024. 2, 3, 7   
[21] Xuan Ju, Ailing Zeng, Yuxuan Bian, Shaoteng Liu, and Qiang Xu. Pnp inversion: Boosting diffusion-based editing with 3 lines of code. In ICLR, 2024. 2, 6, 7, 8, 12, 14   
[22] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In CVPR, pages 60076017, 2023. 2, 3   
[23] Vladimir Kulikov, Matan Kleiner, Inbar HubermanSpiegelglas, and Tomer Michaeli. Flowedit: Inversion-free text-based editing using pre-trained flow models. arXiv preprint arXiv:2412.08629, 2024. 2, 6, 7, 8, 13   
[24] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, pages 1973019742, 2023. 3   
[25] Senmao Li, Joost van de Weijer, Taihang Hu, Fahad Shahbaz Khan, Qibin Hou, Yaxing Wang, and Jian Yang. Stylediffusion: Prompt-embedding inversion for text-based editing. arXiv preprint arXiv:2303.15649, 2023. 2   
[26] Yaowei Li, Yuxuan Bian, Xuan Ju, Zhaoyang Zhang, Ying Shan, and Qiang Xu. Brushedit: All-in-one image inpainting and editing. arXiv preprint arXiv:2412.10316, 2024. 2, 3, 6, 7,8   
[27] Haonan Lin, Mengmeng Wang, Jiahao Wang, Wenbin An, Yan Chen, Yong Liu, Feng Tian, Guang Dai, Jingdong Wang, and Qianying Wang. Schedule your edit: A simple yet effective diffusion noise schedule for image editing. arXiv preprint arXiv:2410.18756, 2024. 2   
[28] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 3   
[29] Aoyang Liu, Qingnan Fan, Shuai Qin, Hong Gu, and Yansong Tang. Lipe: Learning personalized identity prior for non-rigid image editing. arXiv preprint arXiv:2406.17236, 2024. 2   
[30] Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434, 2024. 3   
[31] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. NeurIPS, 36, 2024. 3   
[32] Qiang Liu. Rectified flow: A marginal preserving approach to optimal transport. arXiv preprint arXiv:2209.14577, 2022. 3, 4   
[33] Xingchao Liu, Chengyue Gong, et al. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR, 2022. 3, 4, 6   
[34] Yong Liu, Sule Bai, Guanbin Li, Yitong Wang, and Yansong Tang. Open-vocabulary segmentation with semantic-assisted calibration. In CVPR, pages 34913500, 2024. 3   
[35] Yong Liu, Cairong Zhang, Yitong Wang, Jiahao Wang, Yujiu Yang, and Yansong Tang. Universal segmentation at arbitrary granularity with language instruction. In CVPR, pages 34593469, 2024. 3   
[36] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. In ICLR, 2022. 2   
[37] Daiki Miyake, Akihiro Iohara, Yu Saito, and Toshiyuki Tanaka. Negative-prompt inversion: Fast image inversion for editing with text-guided diffusion models. arXiv preprint arXiv:2305.16807, 2023. 2   
[38] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In CVPR, pages 60386047, 2023. 2   
[39] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, pages 41954205, 2023. 2, 4, 12, 15   
[40] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 87488763, 2021. 7, 8   
[41] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 3   
[42] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10684 10695, 2022. 2   
[43] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, pages 234241, 2015. 2, 3, 4   
[44] Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Semantic image inversion and editing using rectified stochastic differential equations. arXiv preprint arXiv:2410.10792, 2024. 6, 7, 8   
[45] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. NeurIPS, 35:25278 25294, 2022. 7   
[46] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. 3, 4, 6   
[47] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 3   
[48] Siao Tang, Xin Wang, Hong Chen, Chaoyu Guan, Zewen Wu, Yansong Tang, and Wenwu Zhu. Post-training quantization with progressive calibration and activation relaxing for text-to-image diffusion models. In ECCV, pages 404420, 2024. 2   
[49] Yoad Tewel, Rinon Gal, Dvir Samuel, Yuval Atzmon, Lior Wolf, and Gal Chechik. Add-it: Training-free object insertion in images with pretrained diffusion models. arXiv preprint arXiv:2411.07232, 2024. 2, 5   
[50] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In CVPR, pages 19211930, 2023. 2, 5, 14   
[51] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. page 60006010, 2017. 4   
[52] Changyuan Wang, Ziwei Wang, Xiuwei Xu, Yansong Tang, Jie Zhou, and Jiwen Lu. Towards accurate post-training quantization for diffusion models. In CVPR, pages 16026 16035, 2024. 2   
[53] Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li, and Ying Shan. Taming rectified flow for inversion and editing. arXiv preprint arXiv:2411.04746, 2024. 2, 6, 7, 8   
[54] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Effcient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023. 3   
[55] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for textto-image generation. NeurIPS, 36:1590315935, 2023. 7, 8   
[56] Sihan Xu, Yidong Huang, Jiayi Pan, Ziqiao Ma, and Joyce Chai. Inversion-free image editing with language-guided diffusion models. In CVPR, pages 94529461, 2024. 2, 6, 7   
[57] Yu Xu, Fan Tang, Juan Cao, Yuxin Zhang, Xiaoyu Kong, Jintao Li, Oliver Deussen, and Tong-Yee Lee. Headrouter: A training-free image editing framework for mmdits by adaptively routing attention heads. arXiv preprint arXiv:2411.15034, 2024. 2   
[58] Zhao Yang, Jiaqi Wang, Yansong Tang, Kai Chen, Hengshuang Zhao, and Philip HS Torr. Lavt: Language-aware vision transformer for referring image segmentation. In CVPR, pages 1815518165, 2022. 3   
[59] Zhao Yang, Jiaqi Wang, Xubing Ye, Yansong Tang, Kai Chen, Hengshuang Zhao, and Philip HS Torr. Languageaware vision transformer for referring segmentation. TPAMI, 2024. 3   
[60] Xubing Ye, Yukang Gan, Yixiao Ge, Xiao-Ping Zhang, and Yansong Tang. Atp-llava: Adaptive token pruning for large vision language models. arXiv preprint arXiv:2412.00447, 2024. 3   
[61] Xubing Ye, Yukang Gan, Xiaoke Huang, Yixiao Ge, Ying Shan, and Yansong Tang. Voco-llama: Towards vision compression with large language models. arXiv preprint arXiv:2406.12275, 2024.   
[62] Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi Feng, Jifeng Dai, and Xiaojie Jin. Flash-vstream: Memorybased real-time understanding for long video streams. arXiv preprint arXiv:2406.08085, 2024. 3   
[63] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, pages 586595, 2018. 7   
[64] Yixuan Zhu, Wenliang Zhao, Ao Li, Yansong Tang, Jie Zhou, and Jiwen Lu. Flowie: Efficient image enhancement via rectified flow. In CVPR, pages 1322, 2024. 2   
[65] Junhao Zhuang, Yanhong Zeng, Wenran Liu, Chun Yuan, and Kai Chen. A task is worth one word: Learning with task prompts for high-quality versatile image inpainting. In ECCV, pages 195211, 2024. 2, 3

# Appendix

In this supplementary material, we provide more details and findings. In Appendix A, we present additional experimental results and implementation details of our proposed KVEdit. Appendix B provides further discussion and data regarding our inversion-free methodology. Appendix C details the design and execution of our user study. Moreover, In Appendix D, we discuss potential future directions and current limitations of our work.

# A. Implementation and More Experiments

Implementation Details. Our code is built on Flux [1], with modifications to both double block and single block to incorporate KV cache through additional function parameters. Input masks are first downsampled using bilinear interpolation, then transformed from single-channel to 64- channel representations following the VAE in Flux [1]. In the feature space, the smallest pixel unit is 16 dimensions rather than the entire 64-dimensional token. Therefore, in addition to KV cache, we preserve the intermediate image features at each timestep to ensure fine-grained editing capabilities. In our experiment, inversion and denoising can be performed independently, allowing a single image to be inverted just once and then edited multiple times with different conditions, further enhancing the practicality of this workflow.

Experimental Results. Due to space constraints in the main paper, we only present results on the PIE-Bench [21].

Here, we provide additional examples demonstrating the effectiveness of our approach. To further showcase the flexibility of our method, Fig. A and Fig. B present various editing target applied to the same source image, without explicitly labeling the input masks because each case corresponds to a different mask. Fig. D illustrates the impact of steps and reinitialization strategy on the color changing tasks and inpainting tasks.

When changing colors, as the number of skip-steps decreases and reinitialization strategy is applied, the color information in the tokens is progressively disrupted, ultimately achieving successful results. In our experiments, the optimal number of steps to skip depends on image resolution and content, which can be adjusted based on specific needs and feedback. Unlike previous training-free methods, our approach even can be applied to inpainting tasks after employing reinitialization strategy, as demonstrated in the third row of Fig. D. The originally removed regions in inpainting tasks can be considered as black objects, thus requiring reinitialization strategy to eliminate pure black information and generate meaningful content. We plan to further extend our method to inpainting tasks in future work, as there are currently very few training-free methods available for this application.

Attention Scale When dealing with large masks (e.g., background changing tasks), our original method may produce discontinuous images including conflicting content, as illustrated in Fig. C. Stable-Flow [4] demonstrated that during image generation with DiT [39], image tokens primarily attend to their local neighborhood rather than globally across most layers and timesteps.

![](images/8.jpg)  
.

![](images/9.jpg)  
.

Consequently, although our approach treats the background as a condition to guide new content generation, large masks can introduce generation bias which ignore existing content and generate another objects. Based on this analysis, we propose a potential solution as shown in Fig. C. We directly increase the attention weights from masked regions to unmasked regions in the attention map (produced by query-key multiplication), effectively mitigating the bias impact. This attention scale mechanism enhances content coherence by strengthening the influence of preserved background on new content.

# B. More Discussions on Inversion-Free

We implement inversion-free editing on Flux [1] based on the code provided by FlowEdit [23]. As noted in FlowEdit [23], adding random noise at each editing step may introduce artifacts, a phenomenon we also demonstrate in the main paper. In this section, we primarily explore the impact of inversion-free methods on memory consumption.

![](images/10.jpg)  
Figure C. Implementation of attention scale. The scale can be adjusted to achieve optimal results.

![](images/11.jpg)  
Figure D. Additional ablation studies on two tasks. The first and second rows demonstrate the impact of timesteps and reinitialization strategy $\mathbf { \Pi } ( \mathbf { R I } )$ on color changing. The third row demonstrates the impact of timesteps and RI on the inpainting tasks.

Algorithm A demonstrates the implementation of inversion-free KV-Edit, where "KV-inversion" and "KVdenoising" refer to single-step noise prediction with KV cache. KV cache is saved during a one-time inversion process and immediately utilized in the denoising process. The final vector can be directly added to the original image without first inversing it to noise. This strategy ensures that the space complexity of KV cache remains $O ( 1 )$ along the time dimension. Moreover, resolution has a more significant impact on memory consumption as the number of image tokens grows at a rate of $O ( n ^ { 2 } )$ .

We conducted experiments across various resolutions and time steps, reporting memory usage in Tab. A. When processing high-resolution images and more timesteps, personal computers struggle to accommodate the mem

Table A. Memory usage at different resolutions and timesteps. Our approach has a space complexity of $O ( n )$ along the time dimension, while inversion-free methods achieve $O ( 1 )$ .   

<table><tr><td rowspan="2">timesteps</td><td colspan="2">512 × 512</td><td colspan="2">768 × 768</td></tr><tr><td>Ours</td><td>+Inf.</td><td>Ours</td><td>+Inf.</td></tr><tr><td>24 steps</td><td>16.2G</td><td>1.9G</td><td>65.8G</td><td>3.5G</td></tr><tr><td>28 steps</td><td>19.4G</td><td>1.9G</td><td>75.6G</td><td>3.5G</td></tr><tr><td>32 steps</td><td>22.1G</td><td>1.9G</td><td>86.5G</td><td>3.5G</td></tr></table>

# Algorithm A Simplified Inf. version KV-Edit

1: Input: $t _ { i }$ , real image $x _ { 0 } ^ { s r c }$ , foreground $z _ { t _ { i } } ^ { f g }$ ,foreground   
region mask, KV cache $C$   
2: Output: Prediction vector V   
3: $N _ { t _ { i } } \sim \mathcal { N } ( 0 , 1 )$   
4: $x _ { t _ { i } } ^ { s r c } = ( 1 - t _ { i } ) x _ { t _ { 0 } } ^ { s r c } + t _ { i } N _ { t _ { i } }$ b   
5: 6: $V _ { \theta t _ { i } } ^ { s r c } , C = \mathrm { K V - I n v e r i s o n } ( x _ { t _ { i } } ^ { s r c } , t _ { i } , C )$ $\widetilde { z } _ { t _ { i } } ^ { f g } = z _ { t _ { i } } ^ { f g } + m a s k \cdot ( x _ { t _ { i } } ^ { s r c } - x _ { 0 } ^ { s r c } )$ $\widetilde { V } _ { \theta t _ { i } } ^ { f g }$ $C = \mathrm { K V - D e n o s i n g } ( \widetilde { z } _ { t _ { i } } ^ { f g } , t _ { i } , C )$   
8: Return $V _ { \theta t _ { i } } ^ { f g } = \widetilde { V } _ { \theta t _ { i } } ^ { f g } - V _ { \theta t _ { i } } ^ { s r c }$

![](images/12.jpg)  
1In terms of image quality and aesthetics, which image is better, A or B? Option C cannot be selected. OA OB OC   
Figure E. User study. We provide a sample where participants were presented with the original image, editing prompts, results from two different methods for comparison and four questions from four aspects.

Rs h  oy good, choose C.

B  h ll. OA OB OC

dissatisfied with both, choose C.

ory requirements. Nevertheless, we still recommend the inversion-based KV-Edit approach for several reasons:

1. Current inversion-free methods occasionally introduce artifacts.   
2. Inversion-based KV-Edit enables multiple editing attempts after a single inversion, significantly improving usability and workflow efficiency.   
3. Large generative models inherently require substantial GPU memory, which presents another challenge for personal computers. Therefore, we position inversion-based KV-Edit as a server-side technology.

# C. User Study Details

We conduct our user study in a questionnaire format to collect user preferences for different methods. We observe that in most cases, users struggle to distinguish the background effects of training-based inpainting methods (e.g., FLUX-Fill [1] sometimes increases grayscale tones in images). Therefore, we allowed participants to select "equally good" regarding background quality.

Additionally, PIE-Bench [21] contains several challenging cases where all methods fail to complete the editing tasks satisfactorily. Consequently, we allow users to select "neither is good" for text alignment and overall satisfaction metrics, as illustrated in Fig. E.

We implement a single-blind mechanism where the corresponding method for each question is randomly sampled, ensuring fairness in the comparison. We collect over 2,000 comparison results and calculate our method's win rate after excluding cases where both methods are rated equally.

# D. Limitations and Future Work

In this section, we outline the current challenges faced by our method and potential future improvements. While our approach effectively preserves background content, it struggles to maintain foreground details. As shown in Fig. D, when editing garment colors, clothing appearance features may be lost, such as the style, print or pleats.

Typically, during the generation process, early steps determine the object's outline and color, with specific details and appearance emerging later. In the contrast, during inversion, customized object details are disrupted first and subsequently influenced by new content during denoising. This represents a common challenge in the inversion-denoising paradigm [12, 16, 50].

In future work, we could employ trainable tokens to preserve desired appearance information during inversion and inject it during denoising, still without fine-tuning of the base generative model. Furthermore, our method could be adapted to other modalities, such as video and audio editing, image inpainting tasks. We hope that "KV cache for editing" can be considered an inherent feature of the DiT [39] architecture.