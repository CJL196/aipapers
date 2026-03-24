# 基于跨注意力控制的提示到提示图像编辑

Amir Hertz\*1,2，Ron Mokady\*1,2，Jay Tenebaum，Kfir Aberman，Yael Pritch1 和 Daniel Cohen-\*1，1 谷歌研究 2 特拉维夫大学布拉瓦特尼克计算机科学学院

# 摘要

最近的大规模文本驱动合成模型因其生成高多样性图像的卓越能力而受到广泛关注，这些图像遵循给定的文本提示。这种基于文本的合成方法对习惯于口头描述其意图的人类尤其具有吸引力。因此，自然地将文本驱动的图像合成扩展到文本驱动的图像编辑是合乎逻辑的。对于这些生成模型来说，编辑是一项具有挑战性的任务，因为编辑技术的一个固有特性是保留大部分原始图像，而在基于文本的模型中，即使文本提示的小修改也常常导致完全不同的结果。最先进的方法通过要求用户提供空间掩码来定位编辑，从而减轻这一问题，因而忽略了掩码区域内的原始结构和内容。本文中，我们追求一个直观的提示到提示的编辑框架，其中编辑仅由文本控制。为此，我们深入分析了一个文本条件模型，并观察到交叉注意力层是控制图像空间布局与提示中每个单词之间关系的关键。基于此观察，我们提出了几个应用，通过仅编辑文本提示来监控图像合成。这包括通过替换单词进行局部编辑，通过添加规格进行全局编辑，甚至精细控制单词在图像中反映的程度。我们展示了多种图像和提示下的结果，证明了高质量的合成和对编辑提示的忠实度。

# 1 引言

最近，大规模语言-图像（LLI）模型，如Imagen、DALL·E 2和Parti，展现出非凡的生成语义和组合能力，受到了研究界和公众的前所未有的关注。这些LLI模型在极大规模的语言-图像数据集上训练，使用最先进的图像生成模型，包括自回归模型和扩散模型。然而，这些模型并未提供简单的编辑方式，并且普遍缺乏对给定图像特定语义区域的控制。特别是，即使是文本提示中最轻微的变化，也可能导致完全不同的输出图像。为了解决这个问题，基于LLI的方法要求用户明确地遮罩待修补图像的部分，并驱动编辑后的图像仅在遮罩区域内变化，且背景需与原始图像匹配。这种方法虽然提供了吸引人的结果，但遮罩过程繁琐，妨碍了快速和直观的文本驱动编辑。此外，遮罩图像内容会移除重要的结构信息，这在修补过程中被完全忽视。因此，一些编辑能力超出了修补的范围，例如修改特定对象的纹理。本文提出了一种直观而强大的文本编辑方法，通过Prompt-to-Prompt操控，在预训练的文本条件扩散模型中对图像进行语义编辑。为此，我们深入研究交叉注意力层，并探索其语义强度作为控制生成图像的手段。

![](images/1.jpg)  

Figure 1: Our method provides variety of Prompt-to-Prompt editing capabilities. The user can tune the level jeo-ee  he -h esy (bot-left), or make urther refinement ove the enerate image (bottm-riht). The manipulationsfiltratd through the cross-attention mechanism of the diffusion model without he need for any specifiations over the image pixel space.

具体来说，我们考虑内部交叉注意力图，它们是将从提示文本中提取的像素和词元绑定在一起的高维张量。我们发现这些图包含丰富的语义关系，这些关系对生成的图像有重要影响。我们的核心观点是，在扩散过程中通过注入交叉注意力图来编辑图像，控制在每个扩散步骤中哪些像素关注提示文本中的哪些词元。为了将我们的方法应用于各种创意编辑应用，我们展示了几种通过简单且语义友好的接口控制交叉注意力图的方法。第一种方法是在固定交叉注意力图的情况下，改变提示中单个词元的值（如将“狗”改为“猫”），以保持场景的构图。第二种方法是通过去掉以前词元的注意力，让新的注意力流向新的词元，从而改变风格。第三种是增强或削弱生成图像中某个词的语义效果。我们的方法构成了一种直观的图像编辑接口，只需编辑文本提示，因此称之为“提示到提示”（Prompt-to-Prompt）。该方法使得各种编辑任务变得可行，而这些任务在其他情况下则较为困难，并且不需要模型训练、微调、额外数据或优化。在我们的分析过程中，我们发现对生成过程有更大的控制，认识到编辑的提示与源图像之间存在权衡。我们甚至演示了我们的 метод 可以通过使用现有反演过程应用于真实图像。我们的实验和大量结果表明，我们的方法使得在极其多样的图像上实现无缝编辑成为可能，并且以直观的文本方式进行。

# 2 相关工作

图像编辑是计算机图形学中最基本的任务之一，包括通过使用辅助输入（如标签、涂鸦、掩码或参考图像）来修改输入图像的过程。一种特别直观的图像编辑方式是通过用户提供的文本提示。近年来，基于文本的图像处理利用生成对抗网络（GANs）取得了显著进展，这些网络以其高质量生成而闻名，并与 CLIP 联合使用，后者由富有语义的联合图像-文本表示组成，经过数百万对文本-图像对的训练。将这些组件结合起来的开创性工作是革命性的，因为它们不需要额外的手工劳动，仅通过文本就产生了高度真实的操作。Bau 等人进一步展示了如何使用用户提供的掩码来定位基于文本的编辑，并将更改限制在特定的空间区域。然而，尽管基于 GAN 的图像编辑方法在高度策划的数据集上取得成功，例如人类面孔，但在大型和多样化的数据集上却表现不佳。

![](images/2.jpg)  

图：通过注意力注入进行内容修改。我们从提示“柠檬蛋糕”生成的原始图像开始，并将文本提示修改为各种其他蛋糕。在顶部行中，我们在扩散过程中注入了注意力权重。在底部，我们仅使用与原始图像相同的随机种子，没有注入注意力权重。后者导致了一个与原始图像几乎没有关系的全新结构。为了获得更具表现力的生成能力，Crowson等人采用了在多样数据上训练的VQ-GAN作为主干。其他工作则利用了近期的扩散模型，这些模型在高度多样的数据集上实现了最先进的生成质量，常常超越GAN。Kim等人展示了如何进行全局变化，而Avrahami等人则成功地使用用户提供的掩码进行局部操作。尽管大多数仅需文本（即，无掩码）的方法仅限于全局编辑，Bar-Tal等人提出了一种基于文本的局部编辑技术，无需使用任何掩码，并取得了令人印象深刻的结果。然而，他们的技术主要允许改变纹理，而无法修改复杂结构，如将自行车改为汽车。此外，与我们的方法不同，他们的方法需要为每个输入训练一个网络。大量研究显著推动了基于简单文本的图像生成，称为文本到图像合成。最近出现了几个大规模的文本-图像模型，如Imagen、DALL-E2和Parti，展示了前所未有的语义生成能力。然而，这些模型并未提供对生成图像的控制，特别是仅使用文本指导。更改与图像相关的原始提示中的一个单词通常会导致完全不同的结果。例如，在“狗”前添加形容词“白色”通常会改变狗的形状。为了克服这个问题，一些工作假设用户提供掩码以限制更改应用的区域。与之前的工作不同，我们的方法仅需要文本输入，通过使用生成模型中的空间信息，允许通过仅修改文本提示来修改局部或全局细节。

![](images/3.jpg)  

Figure 3: Method overview. Top: visual and textual embedding are fused using cross-attention layers that proue spatial attention maps for each textual token. Bottom: we control the spatial layout and geomery o the generated image using the attention maps of a source image. This enables various editing tasks through editing the textual prompt only. When swapping a word in the prompt, we inject the source image maps $M _ { t }$ , overriding the target image maps $M _ { t } ^ { * }$ , to preserve the spatial layout. Where in the case of adding a new phrase, we inject only the maps that correspond to the unchanged part of the prompt. Amplify or attenuate the semantic effect of a word achieved by re-weighting the corresponding attention map.

# 3 方法

设 $\mathcal { T }$ 为通过文本引导扩散模型 [38] 生成的图像，使用的文本提示为 $\mathcal { P }$，随机种子为 $s$。我们的目标是仅通过编辑后的提示 ${ \mathcal { P } } ^ { * }$ 来编辑输入图像，最终生成编辑图像 $\mathcal { T } ^ { * }$。例如，考虑从提示 “我的新自行车” 生成的一幅图像，我们可以通过进一步描述自行车的外观或将其替换为另一个词来改变文本提示。与以往的工作不同，我们希望避免依赖任何用户定义的掩膜来协助或标识编辑应该发生的位置。一种简单但有效的尝试是利用编辑后的文本提示进行内在随机性的推导。然而，如图 2 所示，这导致生成的图像在结构和构图上完全不同。我们的关键观察是，生成图像的结构和外观不仅依赖于随机种子，还依赖于通过扩散过程中像素与文本嵌入之间的交互。通过修改在交叉注意层中发生的像素与文本的交互，我们提供了提示到提示的图像编辑功能。更具体地说，注入输入图像 $\mathcal { T }$ 的交叉注意图使我们能够保留原始的构图和结构。在 3.1 节中，我们回顾交叉注意的使用方式，在 3.2 节中我们描述如何利用交叉注意进行编辑。有关扩散模型的更多背景信息，请参见附录 A。

# 3.1 文本条件扩散模型中的交叉注意力

我们使用 Imagen [38] 文本引导合成模型作为主干网络。由于构图和几何形状主要在 $6 \times 6$ 分辨率下确定，因此我们仅调整文本到图像的扩散模型，超分辨率过程保持不变。回顾一下，每个扩散步骤 $t$ 包括使用 U 形网络 [37] 从有噪声图像 $z_{t}$ 和文本嵌入 $\psi(\mathcal{P})$ 预测噪声 $\epsilon$。在最后一步，这个过程生成图像 $\mathcal{T} = z_{0}$，通过噪声预测，其中视觉和文本特征的嵌入使用交叉注意力层进行融合，生成每个文本词元的空间注意力图。

更正式地，如图3（顶部）所示，噪声图像的深层空间特征 $\phi \big ( z _ { t } \big )$ 被投影到查询矩阵 $Q = \ell _ { Q } ( \phi ( z _ { t } ) )$，而文本嵌入被投影到键矩阵 $\dot { K } = \ell _ { K } \mathsf { \bar { ( } } \psi ( \mathcal { P } ) )$ 和值矩阵 $V = \ell _ { V } ( \psi ( \mathcal { P } ) )$，通过学习得到的线性投影 $\ell _ { Q } , \ell _ { K } , \ell _ { V }$。注意力图是由单元 $M _ { i j }$ 定义的，其中 $M _ { i j }$ 表示第 $j$ 个词元在像素 $i$ 上的值的权重，而 $d$ 是键和查询的潜在投影维度。最后，交叉注意力输出定义为 $\widehat { \phi } \left( z _ { t } \right) =$ $\bar { M V }$，然后用于更新空间特征 $\phi \big ( z _ { t } \big )$。

![](images/4.jpg)  

Figure 4: Cross-attention maps of a text-conditioned diffusion image generation. The top row displays the average attention masks for each word in the prompt that synthesized the image on the let. The bottom rows display the attention maps from different diffusion steps with respect to the words "bear" and "bird".

$$
M = \mathrm { S o f t m a x } \left( \frac { Q K ^ { T } } { \sqrt { d } } \right) ,
$$

直观上，交叉注意力输出 $M V$ 是值 $V$ 的加权平均，权重是注意力图 $M$，与 $Q$ 和 $K$ 之间的相似性相关。在实际操作中，为了增强表达能力，通常采用多头注意力 [44] 并行计算，随后将结果进行拼接，并通过一个学习的线性层获得最终输出。Imagen [38] 类似于 GLIDE [28]，在每个扩散步骤的噪声预测中依据文本提示进行条件处理。我们引入两种类型的注意力层，即交叉注意力层和自注意力层，通过简单地将文本嵌入序列与每个自注意力层的键值对拼接在一起。在本文的其余部分，我们将这两者统称为交叉注意力，因为我们的方法仅干预混合注意力的交叉注意力部分。也就是说，只有最后的通道，指代文本标记，会在混合注意力模块中被修改。

# 3.2 控制交叉注意力

我们回到我们的关键观察——生成图像的空间布局和几何形状依赖于交叉注意力图。这种像素与文本之间的交互在图4中得以说明，其中平均注意力图被绘制出来，可以看到像素与描述其的单词之间的关联。需要注意的是，平均是用于可视化的目的，注意力图在我们的方法中是为每个头保持分开的。有趣的是，我们可以看到图像的结构在扩散过程的早期步骤中已经确定。由于注意力反映了整体构图，我们可以将从生成中获得的注意力图$M$，与原始提示$\mathcal { P }$一起，注入到使用修改后提示${ \mathcal { P } } ^ { * }$的第二次生成中。这使得合成的编辑图像$\mathcal { T } ^ { * }$不仅根据编辑后的提示进行了调整，还保留了输入图像$\mathcal { T }$的结构。这个例子是更广泛的“蝴蝶在...的照片”一组特定实例。

![](images/5.jpg)  
Fur Object preervation. By injecting only the attention weights of the word "buterly, taken rom the top-et mag we can prerve he utureand apearan  sigle te whil replaci  cnxt. Note how the butterfly sits on top of all objects in a very plausible manner.

我们在此提供一个通用框架，随后将详细介绍特定的编辑操作。

令 $D M ( \boldsymbol { z } _ { t } , \mathcal { P } , t , \boldsymbol { s } )$ 为扩散过程的单步计算 $t$，其输出噪声图像 $z _ { t - 1 }$ 和注意力图 $M _ { t }$（如果不使用则省略）。我们用 ${ \cal D } M ( z _ { t } , \mathcal { P } , t , s ) \{ M \widehat { M } \}$ 表示一个扩散步骤，在该步骤中，我们用附加给定的图 $\widehat { M }$ 替换注意力图 $M$，但保留来自所提供提示的值 $V$。我们还用 $M _ { t } ^ { * }$ 表示使用编辑过的提示 ${ \mathcal { P } } ^ { * }$ 产生的注意力图。最后，我们定义 $E d i t ( M _ { t } , M _ { t } ^ { * } , t )$ 为一个通用编辑函数，输入为在生成过程中原始图像和编辑图像的第 $t ^ { \because }$ 个注意力图。我们的受控图像生成的通用算法包括同时对两个提示执行迭代扩散过程，在每个步骤中根据所需的编辑任务应用基于注意力的操作。我们注意到，为使上述方法有效，必须固定内部随机性。这是由于扩散模型的特点，即使针对相同的提示，两个随机种子也会产生截然不同的输出。形式上，我们的通用算法为：

# 算法 1：提示到提示的图像编辑

1 输入：一个源提示 $\mathcal { P }$，一个目标提示 ${ \mathcal { P } } ^ { * }$，和一个随机种子 $s$ 2 输出：一个源图像 $x _ { s r c }$ 和一个编辑过的图像 $x _ { d s t }$。 3 $z _ { T } \sim N ( 0 , I )$ ，一个具有随机种子 $s$ 的单位高斯随机变量； 4 $z _ { T } ^ { * } \gets z _ { T }$； 5 对于 $t = T , T - 1 , \dots , 1$，执行： 6 $z _ { t - 1 } , M _ { t } \gets D M ( z _ { t } , \mathcal { P } , t , s )$ 7 $M _ { t } ^ { * } \gets D M ( z _ { t } ^ { * } , \mathcal { P } ^ { * } , t , s )$； 8 $\widehat { M _ { t } } \gets E d i t ( M _ { t } , M _ { t } ^ { * } , t )$； 9 $z _ { t - 1 } ^ { * } D M ( z _ { t } ^ { * } , \mathcal { P } ^ { * } , t , s _ { t } ) \{ M \widehat { M } _ { t } \}$； 10 结束 11 返回 $( z _ { 0 } , z _ { 0 } ^ { * } )$ 请注意，我们还可以定义通过提示 $\mathcal { P }$ 和随机种子 $s$ 生成的图像 $\mathcal { T}$ 作为额外的输入。然而，算法将保持不变。有关真实图像编辑的内容，请参见第4节。此外，请注意，我们可以通过在扩散前向函数中应用编辑函数来跳过第7行中的前向调用。此外，可以在同一批次中（即并行）对 $z _ { t - 1 }$ 和 $z _ { t } ^ { * }$ 应用扩散步骤，因此与扩散模型的原始推断相比，仅增加一个步骤的开销。我们现在转向具体的编辑操作，填补 $E d i t ( M _ { t } , M _ { t } ^ { * } , t )$ 函数的缺失定义。概述见图3（底部）。词语替换。在这种情况下，用户将原始提示中的词元与其他词元交换，例如，$\mathcal { P } = ^ { \bullet } \mathrm { { a } }$ 大红色自行车"替换为 ${ \mathcal { P } } ^ { * } = ^ { * }$ "一辆大红色汽车"。主要挑战是保持原始结构，同时处理新提示的内容。为此，我们将源图像的注意力图注入到使用修改后的提示生成的过程中。然而，所提议的注意力注入可能会对源图像和提示施加过多约束：

![](images/6.jpg)

一只猫骑在自行车上的照片。

![](images/7.jpg)  
Figure 6:Attention injection through a varied number of diffusion steps.On the top, we show the source image and prompt. In each row, we modiy the content of the mage by replacing a sgle word in the text nd injecting the cross-attention maps of the source image ranging from $0 \%$ (on the left) to $100 \%$ (on the right) of the diffusion steps.Notice that on one hand, without our method, none of the source image content is guaranted to be preerveOn the other hand njecng the ros-attentihrohoutll the ifusion seps may over-constrain the geomery, resulting in low fidelity to the text prompt, e.g the car (3rd row) becomes a bicycle with full cross-attention injection.

我们通过建议一种更温和的注意力约束来实现这一目标：

$$
\begin{array} { r } { E d i t ( M _ { t } , M _ { t } ^ { * } , t ) : = \left\{ \begin{array} { l l } { M _ { t } ^ { * } \quad } & { \mathrm { i f ~ } t < \tau } \\ { M _ { t } \quad } & { \mathrm { o t h e r w i s e . } } \end{array} \right. } \end{array}
$$

其中 $\tau$ 是一个时间戳参数，用于确定注入应用到哪一步。注意，在扩散过程的早期步骤中决定了组合。因此，通过限制注入步骤的数量，我们可以指导新生成图像的组合，同时保留必要的几何自由度以适应新的提示。第4节提供了一个示例。我们算法的另一个自然放松是为提示中的不同标记分配不同数量的注入时间戳。如果两个词使用不同数量的词元来表示，则可以根据需要使用下一段中描述的对齐函数对映射进行复制/平均。添加新短语。在另一种情况下，用户向提示中添加新词元，例如，$\mathcal { P } = ^ { 6 } \mathrm { { a } }$ "河边的城堡" 到 ${ \mathcal { P } } ^ { * } =$ "儿童画的河边城堡"。为了保留共同细节，我们仅针对两个提示中的共同词元应用注意力注入。形式上，我们使用一个对齐函数 $A$，该函数接收来自目标提示 ${ \mathcal { P } } ^ { * }$ 的词元索引，并输出 $\mathcal { P }$ 中对应的词元索引，如果没有匹配则输出 None。然后，编辑函数为：街边的一辆车。

![](images/8.jpg)  

Figure 7: Editing by prompt refinement. By extending the description of the initial prompt, we can make local edits to the car (top rows) or global modifications (bottom rows).

$$
\begin{array} { r } { \big ( E d i t \left( M _ { t } , M _ { t } ^ { * } , t \right) \big ) _ { i , j } : = \left\{ \begin{array} { l l } { \big ( M _ { t } ^ { * } \big ) _ { i , j } \quad } & { \mathrm { ~ i f ~ } A ( j ) = N o n e } \\ { \big ( M _ { t } \big ) _ { i , A ( j ) } \quad } & { \mathrm { ~ o t h e r w i s e . } } \end{array} \right. } \end{array}
$$

请回忆，索引 $i$ 对应于像素值，而 $j$ 对应于文本标记。我们可以设置时间戳 $\tau$ 来控制施加注入的扩散步骤数量。这种编辑方式支持多样的提示到提示能力，例如风格化、指定对象属性或全局操作，如第4节所示的注意力重加权。最后，用户可能希望增强或减弱每个标记对最终图像的影响程度。例如，考虑提示 $\mathcal { P } = \{ \text{“一个毛茸茸的红色球”} \}$，假设我们想让球更加或不那么毛茸茸。为了实现这种操控，我们通过参数 $c \in [-2, 2]$ 缩放分配给标记 $j^{*}$ 的注意力图，从而产生更强或更弱的效果。其余的注意力图保持不变。即：

$$
\big ( E d i t ( M _ { t } , M _ { t } ^ { * } , t ) \big ) _ { i , j } : = \left\{ \begin{array} { l l } { c \cdot ( M _ { t } ) _ { i , j } \quad } & { \mathrm { i f ~ } j = j ^ { * } } \\ { ( M _ { t } ) _ { i , j } \quad } & { \mathrm { o t h e r w i s e . } } \end{array} \right.
$$

如第4节所述，参数$c$可以对所诱导的效果进行精细而直观的控制。

# 4 应用领域

我们的方法在第3节中描述，通过控制与用户提供的提示中每个单词对应的空间布局，实现了直观的纯文本编辑。在本节中，我们展示了使用该技术的多个应用。

文本仅本地化编辑。我们首先通过修改用户提供的提示来演示本地化编辑，而无需任何用户提供的掩码。在图2中，我们展示了一个示例，通过提示“柠檬蛋糕”生成一幅图像。我们的方法使得在将单词“柠檬”替换为“南瓜”时，能够保留空间布局、几何形状和语义（上排）。请注意，背景得到了很好的保留，包括左上角的柠檬变成南瓜。另一方面，简单地用提示“南瓜蛋糕”喂入合成模型，结果会得到完全不同的几何形状（第3排），即使在确定性设置下使用相同的随机种子（即DDIM）。我们的方法即使在“意大利面蛋糕”这样的挑战性提示下也能成功（第2排）——生成的蛋糕由意大利面层和顶部的番茄酱组成。另一个示例在图5中提供，我们并未注入整个提示的注意力，而是通过对特定单词的注意力进行处理。这使得我们能够保持原始内容，同时更改其余部分。附录中提供了更多结果（图13）。

![](images/9.jpg)  

Fiur：图像风格化。通过在提示中添加风格描述，同时注入源注意力图，我们可以改变风格，从而影响生成的内容。如图6所示，我们的方法并不仅限于修改纹理，还可以进行结构上的变换，例如将“自行车”换成“汽车”。为了分析我们的注意力注入，在左列中我们展示了没有交叉注意力注入的结果，其中改变单个词会导致完全不同的输出。我们随后展示了通过多次扩散步骤获得的平均对象嵌入结果。注意，我们在应用交叉注意力注入的扩散步骤中，步骤越高，对原始图像的保真度越高。然而，最佳结果并不一定通过在所有扩散步骤中应用注入来实现。因此，我们可以通过更改注入步骤的数量，为用户提供对保真度的更好控制。用户可能希望在生成的图像中添加新的说明，而不是简单地用一个词替换另一个词。在这种情况下，我们保留原始提示的注意力图，同时允许生成器生成新词。例如，通过在生成过程中添加“雪”可以在保留背景的同时为原始图像增加细节。更多示例见附录（图14）。全球编辑。保留图像构图不仅对局部编辑有价值，也是全球编辑的重要方面。在这种设置下，编辑应影响图像的所有部分，但仍保留原始构图，例如对象的位置和身份。如图所示（图X），我们在添加“雪”或改变光线的同时保留图像内容。附图8中还有额外示例，包括将素描转换为真实感图像和引入艺术风格。使用注意力重新加权进行调节控制。虽然通过编辑提示来控制图像非常有效，但我们发现仍无法完全控制生成的图像。考虑到提示“雪山”。用户可能希望控制山上的雪量。然而，通过文本描述所需的雪量是相当困难的。相反，我们建议使用调节器控制[24]，用户控制特定词引发的效果大小，如图9所示。正如第3节所述，我们通过重新缩放指定词的注意力来实现这种控制。附录中有更多结果（图15）。

![](images/10.jpg)  
"My fluffy(↑) bunny doll.   

Figure 9: Text-based image editing with fader control. By reducing (top rows) or increasing (bottom) the cros-attenion  the specd word (mark wit a aow), we cancontrol the extent to which it incs the generated image.

真实图像编辑需要找到一个初始噪声向量，使其在扩散过程中产生给定的输入图像。这个过程被称为反演，最近在生成对抗网络（GANs）中引起了相当大的关注，例如[51, 1, 3, 35, 50, 43, 45, 47]，但在文本引导的扩散模型中尚未得到充分探讨。以下将展示基于常见扩散模型反演技术的真实图像初步编辑结果。首先，一个相对简单的方法是向输入图像添加高斯噪声，然后执行预定数量的扩散步骤。由于这种方法导致了显著的失真，我们采用了一种改进的反演方法[10, 40]，该方法基于确定性DDIM模型，而不是DDPM模型。我们在反向方向上执行扩散过程，即$x_{0} \longrightarrow x_{T}$，而不是$x_{T} \longrightarrow x_{0}$，其中$x_{0}$被设置为给定的真实图像。该反演过程通常会产生令人满意的结果，如图10所示。然而，根据文献[43]，反演的结果并非始终稳定，我们认识到减少无分类器引导[18]参数（即减少提示影响）能改善重建效果，但会限制我们进行显著操作的能力。为了缓解这一限制，我们提出使用从注意力图中直接提取的掩膜恢复原始图像的未编辑区域。请注意，这里的掩膜生成不依赖于用户的引导。如图12所示，即便使用简单的DDPM反演方案（添加噪声然后去噪），该方法效果也很好。值得注意的是，猫的身份在各种编辑操作中得到了很好的保留，而掩膜仅仅是根据提示本身生成的。

![](images/11.jpg)  

Figure 10: Editing of real images. On the left, inversion results using DDIM [40] sampling. We reverse the diin pros alize  iven elmage n text pompt. This sultlaten  that p an approximation to the input image when ed to the diffusion process.Afterwar, on the right, we appy our Prompt-to-Prompt technique to edit the images.

![](images/12.jpg)  

Figure 11: Inversion Failure Cases.Current DDIM-based inversion of real images might result in unsatisfied reconstructions.

# 5 结论

在这项工作中，我们揭示了文本到图像扩散模型中跨注意力层的强大能力。我们表明，这些高维层具有对空间映射的可解释表示，使得文本与图像之间的关系更加紧密。基于这一观察，我们展示了如何通过对提示的各种操控直接控制合成图像中的属性，为局部和全局编辑等多种应用铺平了道路。这项工作是朝着为用户提供简单直观的图像编辑手段迈出的第一步，利用文本的语义力量。它使用户能够在一个语义文本空间中导航，该空间在每一步之后会表现出逐步变化，而不是在每次文本操作后从头生成所需图像。

虽然我们已经通过仅改变文本提示演示了语义控制，但我们的方法仍存在若干实际问题。一些生成结果会出现轻微失真。此外，逆推过程需要用户提供合适的提示，这对复杂构图来说可能具有挑战性。需要指出的是，针对文本引导扩散模型的逆推问题属于与本工作正交的研究方向，未来将进行深入探讨。其次，当前的注意力图分辨率较低，因为交叉注意力层设置在网络瓶颈处，这限制了我们进行更精细局部编辑的能力。为缓解这一问题，我们建议在更高分辨率的层中也引入交叉注意力，这需要对训练过程进行分析，超出当前研究范围，故留待未来工作中解决。最后，我们也认识到当前方法无法实现空间上移动已有物体的控制，这类操作也将作为未来研究内容。

![](images/13.jpg)  

Figure 12: Mask-based editing. Using the atention maps, we preserve the unedited parts of the image when the inversiondistortion issgnificant This does not require ny user-providedmasks, as we extract the paia inomation rom themodelusing urmethod.Note how the at'identiys retaine ter he ditg process.

# 6 致谢

我们感谢 Noa Glaser、Adi Zicher、Yaron Brodsky 和 Shlomi Fruchter 对本研究的宝贵意见，帮助我们改善了这项工作，同时也感谢 Mohammad Norouzi、Chitwan Saharia 和 William Chan 对我们的支持以及提供的 Imagen 预训练模型 [38]。特别感谢 Yossi Matias 早期对于问题的启发性讨论，以及激励和鼓励我们在直观交互的方向上开发技术。

# References

[1] Rameen Abdal, Yipeng Qin, and Peter Wonka. Image2stylegan: How to embed images into the stylegan latent space? In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 44324441, 2019.   
[2] Rameen Abdal, Peihao Zhu, John Femiani, Niloy J Mitra, and Peter Wonka. Clip2stylegan: Unsupervised extraction of stylegan edit directions. arXiv preprint arXiv:2112.05219, 2021.   
[3] Yuval Alaluf, Omer Tov, Ron Mokady, Rinon Gal, and Amit Bermano. Hyperstyle: Stylegan inversion with hypernetworks for real image editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1851118521, 2022.   
[4] Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended latent diffusion. arXiv preprint arXiv:2206.02779, 2022.   
[5] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1820818218, 2022.   
[6] Omer Bar-Tal, Dolev Ofri-Amar, Rafail Fridman, Yoni Kasten, and Tali Dekel. Text2live: Text-driven layered image and video editing. arXiv preprint arXiv:2204.02491, 2022. [7] David Bau, Alex Andonian, Audrey Cui, YeonHwan Park, Ali Jahanian, Aude Oliva, and Antonio Torralba. Paint by word, 2021. [8] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096, 2018.   
[9] Katherine Crowson, Stella Biderman, Daniel Kornis, Dashiell Stander, Eric Hallahan, Louis Castricato, and Edward Raff. Vqgan-clip: Open domain image generation and editing with natural language guidance. arXiv preprint arXiv:2204.08583, 2022.   
[10] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems, 34:87808794, 2021.   
[11] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. Advances in Neural Information Processing Systems, 34:1982219835, 2021.   
[12] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1287312883, 2021.   
[13] Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Makea-scene: Scene-based text-to-image generation with human priors. arXiv preprint arXiv:2203.13131, 2022.   
[14] Rinon Gal, Or Patashnik, Haggai Maron, Gal Chechik, and Daniel Cohen-Or. Stylegan-nada: Clipguided domain adaptation of image generators. arXiv preprint arXiv:2108.00946, 2021.   
[15] Ian Goodfelow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.   
[16] Tobias Hinz, Stefan Heinrich, and Stefan Wermter. Semantic object acuracy for generative text-toimage synthesis. IEEE transactions on pattern analysis and machine intelligence, 2020.   
[17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:68406851, 2020.   
[18 Jonahan Ho nd im Salans. lassifer-ee iffun dn. In NeurI 021 Workho n Dee Generative Models and Downstream Applications, 2021.   
[19] Tero Karras, Miika Aittala, Samuli Laine, Erik Härkönen, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Alias-free generative adversarial networks. Advances in Neural Information Processing Systems, 34:852863, 2021.   
[20] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 44014410, 2019.   
[21] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 81108119, 2020.   
[22] Gwanghyun Kim, Taesung Kwon, and Jong Chul Ye. Diffusionclip: Text-guided diffusion models for robust image manipulation. In Procedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24262435, 2022.   
[23] Gihyun Kwon and Jong Chul Ye. Clipstyler: Image style transfer with a single text condition. arXiv preprint arXiv:2112.00374, 2021.   
[24] Guillaume Lample, Neil Zeghidour, Nicolas Usunier, Antoine Bordes, Ludovic Denoyer, and Marc'Aurelio Ranzato. Fader networks: Manipulating images by sliding atributes. Advances in neural information processing systems, 30, 2017.   
[25] Bowen Li, Xiaojuan Qi, Thomas Lukasiewicz, and Philip Torr. Controllable text-to-image generation. Advances in Neural Information Processing Systems, 32, 2019.   
[26] Wenbo Li, Pengchuan Zhang, Lei Zhang, Qiuyuan Huang, Xiaodong He, Siwei Lyu, and Jianfeng Gao. Object-driven text-to-image synthesis via adversarial training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1217412182, 2019.   
[27] Ron Mokady, Omer Tov, Michal Yarom, Oran Lang, Inbar Mosseri, Tali Dekel, Daniel Cohen-Or, and T on Computer Graphics and Interactive Techniques Conference Proceedings, pages 19, 2022.   
[28] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya diffusion models. arXiv preprint arXiv:2112.10741, 2021.   
[29] Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, and Dani Lischinski. Styleclip: Textdriven manipulation of stylegan imagery. arXiv preprint arXiv:2103.17249, 2021.   
0T Qo, Ji Z  Xu,    ex generation from prior knowledge. Advances in neural information processing systems, 32, 2019.   
[1] Tingting Qiao, Jing Zhang, Duanqig Xu, and Dacheng Tao. Mirrorgan: Learnig text-to-image generation by redescription. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15051514, 2019.   
[32] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. arXiv preprint arXiv:2103.00020, 2021.   
[33] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.   
[34] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pages 88218831. PMLR, 2021.   
[35] Daniel Roich, Ron Mokady, Amit H. Bermano, and Daniel Cohen-Or. Pivotal tuning for latent-based editing of real images. ACM Transactions on Graphics (TOG), 2022.   
[36] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models, 2021.   
[37] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical ig segmentation. In International Conference onMedical imagecomputing and computer-assisted intervention, pages 234241. Springer, 2015.   
[38] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. arXiv preprint arXiv:2205.11487, 2022.   
[39] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learing usoequilbrium herynam.I Inteatinal Conrenc on MachLea pages 22562265. PMLR, 2015.   
[40] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2020.   
[41] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in Neural Information Processing Systems, 32, 2019.   
[42] Ming Tao, Hao Tang, Songsong Wu, Nicu Sebe, Xiao-Yuan Jing, Fei Wu, and Bingkun Bao. n Dee usion geneativ dveraal etworks o ext-to-mag nhes. Xi t arXiv:2008.05865, 2020.   
[43] Omer Tov, Yuval Alaluf, Yotam Nitzan, Or Patashnik, and Daniel Cohen-Or. Designing an encoder for stylegan image manipulation. arXiv preprint arXiv:2102.02766, 2021.   
[44] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Ilia Polosukhin.Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017.   
[45] Tengfei Wang, Yong Zhang, Yanbo Fan, Jue Wang, and Qifeng Chen. High-fidelity gan inversion for image attribute editing. ArXiv, abs/2109.06590, 2021.   
[46] Weihao Xia, Yujiu Yang, Jing-Hao Xue, and Baoyuan Wu. Tedigan: Text-guided diverse face image generation and manipulation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22562265, 2021.   
[47] Weihao Xia, Yulun Zhang, Yujiu Yang, Jing-Hao Xue, Bolei Zhou, and Ming-Hsuan Yang. Gan inversion: A survey, 2021.   
[48] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2022.   
[49] Zizhao Zhang, Yuanpu Xie, and Lin Yang. Photographic text-to-image synthesis with a hierarchicallynested adversarial network. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 61996208, 2018.   
[50] Jiapeng Zhu, Yujun Shen, Deli Zhao, and Bolei Zhou. In-domain gan inversion for real image editing. arXiv preprint arXiv:2004.00049, 2020.   
[51] Jun-Yan Zhu, Philipp Krähenbühl, Eli Shechtman, and Alexei A Efros. Generative visual manipulation on the natural image manifold. In European conference on computer vision, pages 597613. Springer, 2016.

# A Background

# A.1 Diffusion Models

Diffusion Denoising Probabilistic Models (DDPM) [39, 17] are generative latent variable models that aim to model a distribution $p _ { \theta } ( x _ { 0 } )$ that approximates the data distribution $q ( x _ { 0 } )$ and easy to sample from. DDPMs model a "forward process" in the space of $x _ { 0 }$ from data to noise.† This process is a Markov chain starting from $x _ { 0 }$ , where we gradually add noise to the data to generate the latent variables $x _ { 1 } , \dots , x _ { T } \ \in \ X$ . The sequence of latent variables therefore follows $\textstyle q ( x _ { 1 } , \dotsc , x _ { t } \mid x _ { 0 } ) = \prod _ { i = 1 } ^ { t } q ( x _ { t } \mid x _ { t - 1 } )$ ,where a step in the forward process is defined as a Gaussian transition $q ( x _ { t } \mid x _ { t - 1 } ) : = N ( x _ { t } ; { \sqrt { 1 - \beta _ { t } } } x _ { t - 1 } , \beta _ { t } I )$ parameterized by a schedule $\beta _ { 0 } , \dots , \beta _ { T } \ \in \ ( 0 , 1 )$ . When $T$ is large enough, the last noise vector $x _ { T }$ nearly follows an isotropic Gaussian distribution.

An interesting property of the forward process is that one can express the latent variable $x _ { t }$ directly as the following linear combination of noise and $x _ { 0 }$ without sampling intermediate latent vectors:

$$
x _ { t } = \sqrt { \alpha _ { t } } x _ { 0 } + \sqrt { 1 - \alpha _ { t } } w , w \sim N ( 0 , I ) ,
$$

where $\begin{array} { r } { \alpha _ { t } : = \prod _ { i = 1 } ^ { t } ( 1 - \beta _ { i } ) } \end{array}$

In order to sample from the distribution $q ( x _ { 0 } )$ , we define the dual "reverse process" $p ( x _ { t - 1 } \mid x _ { t } )$ from isotropic Gaussian noise $x _ { T }$ to data by sampling the posteriors $q ( x _ { t - 1 } \mid x _ { t } )$ .Since the intractable reverse process $q ( x _ { t - 1 } \mid x _ { t } )$ depends on the unknown data distribution $q ( x _ { 0 } )$ , we approximate it with a parameterized Gaussian transition network $p _ { \theta } ( x _ { t - 1 } \mid x _ { t } ) : = N ( x _ { t - 1 } \mid \bar { \mu _ { \theta } } ( x _ { t } , t ) , \Sigma _ { \theta } ( \bar { x } _ { t } , t ) )$ . The $\mu _ { \theta } ( x _ { t } , t \bar { ) }$ can be replaced [17] by predicting the noise $\varepsilon _ { \boldsymbol { \theta } } ( x _ { t } , t )$ added to $x _ { 0 }$ using equation 2.

Under this definition, we use Bayes' theorem to approximate

$$
\mu _ { \theta } ( x _ { t } , t ) = \frac { 1 } { \sqrt { \alpha _ { t } } } \left( x _ { t } - \frac { \beta _ { t } } { \sqrt { 1 - \alpha _ { t } } } \varepsilon _ { \theta } ( x _ { t } , t ) \right) .
$$

Once we have a trained $\varepsilon _ { \boldsymbol { \theta } } ( x _ { t } , t )$ , we can using the following sample method

$$
x _ { t - 1 } = \mu _ { \theta } ( x _ { t } , t ) + \sigma _ { t } z , \ z \sim N ( 0 , I ) .
$$

We can control $\sigma _ { t }$ of each sample stage, and in DDIMs [40] the sampling process can be made deterministic using $\sigma _ { t } = 0$ in all the steps. The reverse process can finally be trained by solving the following optimization problem:

$$
\operatorname* { m i n } _ { \theta } L ( \theta ) : = \operatorname* { m i n } _ { \theta } E _ { x _ { 0 } \sim q ( x _ { 0 } ) , w \sim N ( 0 , I ) , t } \left\| w - \varepsilon _ { \theta } ( x _ { t } , t ) \right\| _ { 2 } ^ { 2 } ,
$$

teaching the parameters $\theta$ to fit $q ( x _ { 0 } )$ by maximizing a variational lower bound.

# A.2 Cross-attention in Imagen

Imagen [38] consists of three text-conditioned diffusion models: A text-to-image $6 4 \times 6 4$ model, and two super-resolution models $- 6 4 \times 6 4 \to 2 5 6 \times 2 5 6$ and $2 5 6 \times 2 5 6 \to 1 0 2 4 \times 1 0 2 4$ .These predict the noise $\boldsymbol { \varepsilon } _ { \boldsymbol { \theta } } \big ( \boldsymbol { z } _ { t } , \boldsymbol { c } , t \big )$ via a U-shaped network, for $t$ ranging from $T$ to 1. Where $z _ { t }$ is the latent vector and $c$ is the text embedding. We highlight the differences between the three models:

• $6 4 \times 6 4 -$ starts from a random noise, and uses the U-Net as in [10]. This model is conditioned on text embeddings via both cross-attention layers at resolutions [16, 8] and hybrid-attention layers at resolutions [32, 16, 8] of the downsampling and upsampling within the U-Net. : $6 4 \times 6 4  2 5 6 \times 2 5 6 -$ conditions on a naively upsampled $6 4 \times 6 4$ image. An efficient version of a U-Net is used, which includes Hybrid attention layers in the bottleneck (resolution of 32). : $2 5 6 \times 2 5 6 \to 1 0 2 4 \times 1 0 2 4 -$ conditions on a naively upsampled $2 5 6 \times 2 5 6$ image. An efficient version of a U-Net is used, which only includes cross-attention layers in the bottleneck (resolution of 64).

# B Additional results

We provide additional examples, demonstrating our method over different editing operations. fig. 13 show word swap results, fig. 14 showadding specication to animage, and fig 15 show attention re-weiting.

![](images/14.jpg)  
Figure 13: Additional results for Prompt-to-Prompt editing by word swap.

"A photo of a bear wearing sunglasses on and having a drink.

![](images/15.jpg)  
source image

![](images/16.jpg)  
..wearing a squared sunglasses..."

![](images/17.jpg)  
"...geeky sunglasses..

![](images/18.jpg)  
Figure 14:Additional results for Prompt-to-Prompt editing by adding a specification.

![](images/19.jpg)

A tiger is sleeping(1) in a field.

888888

![](images/20.jpg)

Photo of a cubic(↓) sushi.

![](images/21.jpg)

The modern(√) city.

![](images/22.jpg)  
My colorful(√) bedroom.

![](images/23.jpg)  
Fure1:Aditinal results r Promt--rot editi by attention re-we.

Photo of a field of poppies at night(.