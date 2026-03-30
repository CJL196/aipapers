# GChatGrodd Larision-语言模型 远程 e

卡尔蒂克·库克雷贾1, 2\* 穆罕默德·索海尔·达尼什1\* 穆扎马尔·纳西尔1 阿比吉特·达斯2 萨尔曼·汗1, 3 法哈德·沙赫巴兹·汗1, 4 1穆罕默德·本·扎耶德人工智能大学，2比尔拉科技与科学学院，海德拉巴 3澳大利亚国立大学，4林雪平大学 kartik.kuckreja@mbzuai.ac.ae，muhammad.sohail@mbzuai.ac.ae

# 摘要

近期，大型视觉语言模型（VLM）在自然图像领域取得了显著进展，使用户能够就特定视觉内容进行对话。然而，这些通用领域的VLM在遥感（RS）场景中的表现不佳，当面对RS领域特定的查询时，往往导致不准确或虚假的信息。这种现象的出现源于RS图像所带来的独特挑战。例如，处理具有各类间尺度变化和众多小物体的高分辨率RS图像时，需要区域层面的推理与整体场景解释相结合。此外，缺乏领域特定的多模态指令跟随数据以及强大的RS主干模型，使得模型很难使其行为与用户查询对齐。为了解决这些限制，我们提出了GeoChat——首个通用的遥感VLM，提供与高分辨率RS图像的多任务对话能力。具体而言，GeoChat不仅可以回答图像级的查询，还可以接受区域输入以进行区域特定的对话。此外，它可以通过参考对象的空间坐标在响应中对其进行视觉定位。为了解决领域特定数据集不足的问题，我们通过扩展现有多样化RS数据集中的图像-文本对，生成了一个新颖的RS多模态指令跟随数据集。我们建立了一个全面的RS多任务对话基准，并与多种基线方法进行了比较。GeoChat在各种RS任务上展示了强健的零-shot性能，例如图像和区域标注、视觉问答、场景分类、视觉基础对话和参照检测。我们的代码在此提供。

# 1. 引言

在自然图像领域，来自网络图像或人工标注的大量对齐图像-文本数据促进了有效的自监督视觉-语言建模，正如多模态 GPT-4 [23] 和开源项目 LLaVA [19] 所示。这些通过生成预训练和指令微调开发的视觉-语言模型 (VLMs) 在各种面向用户的多模态任务中表现出强大的零-shot 任务完成能力。由此产生的能力为开发具有广泛实际应用的多功能多模态对话助手打开了大门 [12]。

![](images/1.jpg)  
Figure 1. GeoChat can accomplish multiple tasks for remotesensing (RS) image comprehension in a unified framework. Given suitable task tokens and user queries, the model can generate visually grounded responses (text with corresponding object locations - shown on top), visual question answering on images and regions (top left and bottom right, respectively) as well as scene classification (top right) and normal natural language conversations (bottom). This makes it the first RS VLM with grounding capability.

然而，针对自然图像设计的一般领域视觉语言模型（VLMs）在处理遥感视觉图像时表现不佳。这种性能差异主要源于遥感图像-文本配对中内容的独特性质与公开可用的网络数据相比。因此，当面对来自遥感传感器的空间图像时，一般领域的VLMs可能提供不准确的信息或产生幻觉。尽管遥感视觉问答（VQA）领域已经取得了显著进展，但早期的方法将此任务框定为分类问题。在这种情况下，模型从训练数据中预先设定的响应中选择答案，这限制了它们在开放式答案生成和遵循指令方面的适用性。本文介绍了GeoChat，这是一个将多模态指令微调扩展到遥感领域的尝试，用于训练多任务对话助手。然而，遥感领域缺乏多模态指令微调的对话数据集。受到近期指令微调工作的启发，GeoChat使用Vicuna-v1.5和自动化管道生成多样的遥感多模态指令跟随数据，总计近318K条指令。我们从多个现有的为不同任务开发的遥感数据集中创建图像-文本配对。这些数据集包括用于VQA的LR-BEN、用于场景分类的NWPU-RESISC-45以及用于目标检测的SAMRS。GeoChat的一个关键能力是将遥感图像的多个图像和区域级推理任务统一在一个管道中（见图1）。我们通过不同的任务标记来实现这一点，帮助模型根据用户需求适当引导其响应。此外，模型在输入中使用空间位置表示，以无缝推理本地区域，并且能够在响应中生成对象位置以视觉地定位对象。这使得GeoChat能够执行一系列丰富的任务，包括指称表达检测、图像/区域标题生成、场景分类、自然语言对话和VQA，以及视觉基础的对话。总之，这项工作具有以下贡献： - 遥感多模态指令跟随数据集。我们提出了一种新颖的数据生成管道，利用现有的目标检测数据集生成图像的简要描述，然后使用Vicuna-v1.5仅通过生成的文本创建对话。此外，我们还添加了视觉问答和场景分类能力，使用其相应的数据集。这使得遥感领域总计生成了318K对指令。 - GeoChat。利用我们的数据集，我们对LLaVA-1.5进行微调，以创建遥感领域的视觉语言模型GeoChat。我们的LoRA微调效率高，并避免了遗忘已完全微调的LLaVA模型中嵌入的必要上下文，其MLP投影经过训练以将图像对齐到LLM（Vicuna-v1.5）的词嵌入空间。这使GeoChat能够保留LLaVA的对话和指令跟随能力，并将其领域知识扩展到遥感任务。 - 我们还解决了评估基准缺乏的问题，以评估现有VLMs在遥感对话中的能力。为此，我们建立了遥感对话基础的评估协议，以及一系列任务设置，以便与未来在这一方向的努力进行比较。我们展示了不同遥感任务的各种监督评估以及零样本评估，包括图像标题生成、视觉问答和场景分类，以证明GeoChat对话型VLM的可推广性。

# 2. 相关工作

大型视觉语言模型。遵循指令的视觉语言模型（VLMs）的典型架构包括使用预训练的视觉主干网络来编码视觉数据，利用大型语言模型来解释用户指令并生成响应，以及一个视觉-语言跨模态连接器，例如线性投影层或多层感知机，用于将视觉信息与语言模型融合。使用VLMs所取得的结果显示出良好的前景；例如，LLaVA、Instruct-BLIP、Otter和MiniGPT-4在自然场景的语言指令跟随和视觉推理能力方面表现出了显著的提升。最近的研究表明，这些模型可以适应其他领域，例如视频、生物医学和遥感。

遥感视觉语言模型。广义视觉语言模型在遥感中的应用相对较少。迄今为止，大多数研究忽视了对物体及其关系的语义理解，以实现深层次的视觉理解。视觉语言模型不仅能够识别图像中的物体，还能够生成自然语言描述并推断物体之间的关系。这使它们更适合进行基于文本的图像检索、图像标注以及回答需要同时具备视觉和语言知识的视觉问题。尽管在遥感任务中，诸如图像标注、零样本分类和视觉问答等视觉语言模型已有所进展，但这些模型只能执行其训练过的特定任务，缺乏对话能力，并且没有关于遥感图像的一般语义知识。在遥感领域，开发通用模型以结合解决所有任务并维持对话能力仍然存在重大缺口。虽然RSGPT是一个初步努力，显示出良好的对话能力并能同时解决多个任务，但它需要针对每个任务单独进行微调，这使其变得繁琐且缺乏泛化能力。此外，RSGPT无法用于区域级推理或视觉落地，而我们的工作旨在解决这一问题。

# 3. GeoChat：基于地理信息的遥感视觉语言模型

基于视觉的遥感对话旨在生成与对应物体位置交织的文本响应。此外，用户还可以提供视觉提示（例如，边界框），除了自然语言问题外，模型应该能够回答关于指定兴趣区域（RoI）的问题。这种视觉和语言模态之间的无缝互动需要深入理解指示视觉场景中特定物体或元素的语言结构。如上所述，GeoChat是首个能够进行基于视觉的遥感图像对话的模型。根据其构造，GeoChat不仅能够应对基于视觉的对话这一挑战性任务，还可以执行一系列其他空间推理任务，这些任务涵盖了视觉图像理解中不同粒度的范围，例如图像/区域标注、指代物体检测以及关于遥感图像的图像/区域级对话。我们在下面正式概述了GeoChat可以实现的任务。 a) 图像级对话任务。在此任务中，GeoChat处理一幅图像 $ _{ \textbf { \em x } } $ 和用户文本查询 $\pmb q$，而其输入和输出中没有任何具体的空间坐标。目标是在图像整体上下文中执行基于对话的任务，例如视觉问答（VQA）、场景分类和图像标注。 b) 区域级对话任务。此任务涉及在输入中向GeoChat提供空间框位置 $^ { b }$，除了 $ _{ \textbf { \em x } } $ 和 $\pmb q$。区域位置 $^ { b }$ 指导模型关注图像中特定区域，从而使模型能够执行任务，例如区域级标注、区域特定的VQA或多轮对话。 c) 基于对象的对话任务。通过使用特殊的标记，称为任务规范标记 $\pmb { t }$，GeoChat可以被引导在不同粒度层次上提供物体位置，同时保持对话能力。它有助于包括有基于对象的图像标注/对话、物体定位和指代表达检测等任务。

# 3.1. GeoChat 架构

GeoChat 遵循 LLaVA-v1.5 的架构[17]，其包含三个核心组件：i) 全局图像编码器，ii) MLP 适配器（两个线性层）以及 iii) 大语言模型。与 LLaVA 不同，我们加入了特定的任务提示，指示模型所需的任务类型，即真实标注、图像级或区域级对话。此外，我们允许在输入和输出中使用空间位置信息，使得可视提示作为输入，并将真实标注的物体作为 GeoChat 输出中的内容。值得注意的是，原始 LLaVA 模型无法执行物体定位或接受区域输入。此外，原始 LLaVA 无法对遥感图像进行推理，而这一功能是通过我们的领域特定数据集实现的。我们将在下面描述架构中的每个组件：

Table 1. Instruction following data used to train GeoChat. Instruction types and format are shown. We use a $3 0 6 \mathrm { k }$ set for training and a separate 12k instruction-set for testing.   

<table><tr><td>Data</td><td>Size</td><td>Response formatting prompts</td></tr><tr><td>Detailed Description</td><td>30k</td><td>Describe the image in detail.</td></tr><tr><td>Multi-Round Conversation</td><td>65k</td><td>-</td></tr><tr><td>Complex Questions</td><td>10k</td><td>-</td></tr><tr><td>RSVQA-LRBEN[20]</td><td>56k</td><td>Answer the question using a single word or phrase.</td></tr><tr><td>NWPU-RESISC-45[5]</td><td>31.5k</td><td></td></tr><tr><td>Floodnet[25]</td><td>4k</td><td></td></tr><tr><td>Grounding Description</td><td>45k</td><td>[grounding] Describe the image in detail.</td></tr><tr><td>Region Captioning</td><td>40k</td><td>[identify] {bx_left, by,top, bx_right, bybottom|θ}</td></tr><tr><td>Referring Expression</td><td>25k</td><td>[refer] &lt; p &gt; Object &lt; /p &gt;</td></tr></table>

任务标识符：GeoChat 的独特之处在于其能够轻松切换不同类型的遥感视觉解释任务。为了消除任务之间的无谓不确定性，我们的方法为每个任务分配一个独特的任务识别码。我们建议三种不同的任务标识符，$\mathbf { \boldsymbol { t } } \in \mathbf { \boldsymbol { \mathbf { \mathit { t } } } }$ {基础对话、识别、指称}，分别用于基础对话、区域描述和指称表达理解。至于视觉问答和场景分类的情况，我们直接要求模型以一个单词或短语的形式输出答案，如表 1 所示。我们的方法不对视觉无关的命令使用任何任务识别标识符。这个统一的方法得益于模块化设计，有效整合空间数据，使模型在推理视觉内容时具备灵活性。

空间位置表示。我们的模型必须准确识别所引用项目的空间位置，以便执行诸如基础对话、指称表达生成和理解等任务。为此，我们以文本格式表示框的位置，以表达地理位置：$\boldsymbol { b } = \{ b _ { \mathrm { x \_ l e f t } } , b _ { \mathrm { y \_ t o p } } , b _ { \mathrm { x \_ r i g h t } } , b _ { \mathrm { y \_ b o t t o m } } | \theta \}$。其中，$b _ { \mathrm { x \mathrm { . l e f t } } } , b _ { \mathrm { y \mathrm { . t o p } } }$表示框的左上角点，而$b _ { \mathrm { x . r i g h t } } , b _ { \mathrm { y . b o t t o m } }$表示右下角坐标。角度$\theta$表示边界框的旋转角度，从下边缘开始。数值在区间[0, 100]内归一化，用于表示$\mathbf { X }$和y坐标。以这种格式表达的区域位置用于通过模型的输入和输出进行交互。视觉主干。GeoChat采用CLIP-ViT(L-14) [28]的预训练视觉主干，它的输入分辨率为$3 3 6 \times 3 3 6$，这使得每幅图像有效地划分为576个补丁。由于该分辨率不足以理解遥感图像中呈现的细节（例如，小物体和物体细节），我们在基于变压器的CLIP [28]模型中对位置编码进行插值，以适应输入图像大小为$5 0 4 \times 5 0 4$。虽然这导致每幅图像补丁数量几乎增加了一倍（即，每幅图像1296个补丁），但这种增强的分辨率使我们能够处理更大的图像尺寸，并且支持在高分辨率遥感图像中更好的视觉定位。

![](images/2.jpg)  
. 1 .   
Figure 3. Multi-task instruction template for GeoChat.

MLP 跨模态适配器。我们从冻结的 CLIP-ViT[28] 中将输出词元 $( \in \mathbb { R } ^ { 1 2 9 6 \times 1 0 2 4 } )$ 投影到语言模型空间，使用具有一个隐藏层的 MLP 适配器。适配器的输入维度为 1024，输出一个大小为 4096 的向量，对应于 LLM [7] 的输入大小。激活函数采用 GeLU [10]。大型语言模型。开源的 Vicunav1.5(7B) [7] 大型语言模型被用作 GeoChat 的基础。语言模型作为我们框架中多样化视觉-语言输入的单一接口。为了完成不同的视觉-语言任务，我们直接依赖 Vicuna-v1.5(7B) [7] 的语言词元。我们明确与语言模型进行交互，以构建边界框的文本表示，表达它们的空间坐标，以用于需要生成空间位置的视觉定位任务。类似地，LLM 的安全、对齐和有效行为通过与给定输入一起附加的系统提示得以确保。一个 [USER] <im_start Image Features <im_end> [Task Identifier] [ASSISTANT] 的基于低秩适应 (LoRA) [11] 的策略用于对 LLM 进行微调。在训练期间，不是微调预训练的 Vicuna-v1.5[7] 的权重矩阵中的所有权重，而是微调 LoRA [11] 中的两个较小矩阵，以逼近原始较大的矩阵。之后，微调后的适配器被输入到预训练模型中并用于推理。LoRA 适应确保了更快的训练，并避免遗忘在基于通用自然语言指令训练和微调的 LLM 中嵌入的原始知识。这是一个重要特征，因为它使得模型能够在 GeoChat 的遥感推理框架中引入关于通用物体类型、地标和功能的外部上下文。

# 3.2. 训练细节

为了提高我们的模型在一般视觉任务上的有效性并优化训练效率，我们采用了一种策略，即用预训练权重初始化网络，并针对遥感相关任务微调特定部分。我们使用一个预训练的 CLIP-ViT(L-14) 编码器，该编码器在大量文本和视觉数据上进行了训练，以及一个在 LAION-CC-SBU [26] 数据集的 558K 子集上预训练的 MLP 适配器[17]，配合 BLIP [15] 标注，并使用 Vicuna-v1.5[7] 来初始化我们的模型。为了使我们的模型适应遥感图像，我们随后对 LLM 进行了 LoRA [11] 微调，同时在训练期间保持 MLP 适配器和 CLIP 编码器 [28] 冻结。

![](images/3.jpg)  
image). Bottom-row: This structured information is used to create the rich instruction-set with a total of $3 1 8 \mathrm { k }$ image-instruction pairs.

# 4. RS 多模态指令数据集

通过使用大语言模型 Vicuna [7]，我们使模型能够遵循一系列指令，方法是呈现和策划与遥感图像相关的多轮对话的多样化指令遵循数据（见表 1）。我们特别提供系统指令作为提示，要求 Vicuna [7] 以能够可视化图像的方式生成多轮问答对（尽管它只能访问文本）。这一过程通过在提示中提供手动编写的少量上下文示例来实现，向 Vicuna [7] 演示如何基于提供的标题和信息构建高质量的指令-响应对。具体来说，从我们使用以下流程创建的简短描述中，我们随机抽取 $6 5 \mathrm { k }$ 张图像以创建多轮对话，$1 0 \mathrm { k }$ 张图像以生成复杂的问答，以及 30k 张图像以生成给定简短描述的详细描述。结合起来，经过转换为指令格式后，我们获得了总计近 $3 0 6 \mathrm { k }$ 对图像-指令对用于训练，12k 对用于测试。接下来，我们概述指令集创建过程。

<table><tr><td>Dataset</td><td>Category</td><td># Classes</td><td># Images</td><td>Image Size</td></tr><tr><td>DOTA</td><td>Object Detection</td><td>18</td><td>17,480</td><td>1024 × 1024</td></tr><tr><td>DIOR</td><td>Object Detection</td><td>20</td><td>23,463</td><td>800 × 800</td></tr><tr><td>FAIR1M</td><td>Object Detection</td><td>37</td><td>64,147</td><td>600 × 600</td></tr><tr><td>LRBEN(rsvqa)</td><td>Visual Question Answering</td><td>-</td><td>600</td><td>256 × 256</td></tr><tr><td>Floodnet</td><td>Visual Question Answering</td><td>-</td><td>4056</td><td>3000 × 4000</td></tr><tr><td>NWPU-RESISC-45</td><td>Scene Classification</td><td>45</td><td>31,500</td><td>256 × 256</td></tr></table>

Table 2. List of datasets used to creat our remote-sensing instruction set for GeoChat VLM training. We include object detection, visual question answering and scene classification datasets with varying image sizes and types of classes to ensure diversity.

组成数据集：在我们的指令集编纂中，我们整合了三种不同类型的数据集，涵盖了为物体检测、场景分类和视觉问答（VQA）设计的数据集。具体来说，我们集成了三个物体检测数据集（DOTA [34]、DIOR [6] 和 FAIR1M [27]，它们共同形成 SAMRS [30] 数据集）、一个场景分类数据集（NWPU-RESISC-45 [5]）、一个 VQA 数据集（LRBEN[20]）和一个洪水检测 VQA 数据集 [25]（见表2）。物体检测数据集提供区域级推理能力，因为它们提供了分割掩码和边界框。缺失类别的添加：尽管物体检测数据库中包含了各种物体类别，但一些重要类别，如建筑物、道路和树木却缺失。为了解决这个问题，我们提议利用在 LoveDA 数据集 [32] 上进行预训练的 ViTAE-RVSA [31] 模型，该模型涵盖了所需的重要类别。该模型 [31] 被用于在 SAMRS [30] 数据集上推断这些类别，从而生成伪标签。为了减少这些预测中的潜在噪声，我们去除 ViTAE-RVSA [31] 的预测结果中已从 SAMRS [30] 数据集中获得真实标注的数据，以优化结果。

![](images/4.jpg)  
The model can also specify object types, object counts, object attributes and object relationships.

Table 3. List of attributes collected for objects. Attributes are used to obtain referring expressions e.g., small-sized plane to the left.   

<table><tr><td></td><td>Attribute</td><td>Example</td></tr><tr><td>a1</td><td>category</td><td>(e.g. &quot;plane, ship&quot;)</td></tr><tr><td>a2</td><td>color</td><td>(e.g. &quot;gray, white&quot;)</td></tr><tr><td>a3</td><td>relative size</td><td>(e.g. &quot;small, large&quot;)</td></tr><tr><td>a4</td><td>relative location</td><td>(e.g. &quot;top right, bottom&quot;)</td></tr><tr><td>a5</td><td>relation</td><td>(e.g. &quot;parked at, driving through&quot;)</td></tr></table>

属性提取：对于指称表达注释，提取 RS 图像中的多种属性非常重要。为此，我们选择了五种不同类型的属性，如表 3 所示。物体类别信息可以直接从 SAMRS 数据集中获得。对于颜色提取，我们使用 K-Means 聚类算法。具体而言，我们利用真值框从图像中提取对象的像素，并将其聚类为 $K$ 组。然后选择最大聚类的中心作为对象的颜色。为了说明物体的相对大小，我们将物体分为三种尺寸：小型、正常和大型。此分类是通过测量整个数据集中一个类别的所有实例的面积来确定的，并将 $80^{th}$ 百分位数标记为大型。同样，$20^{th}$ 百分位数被指定为小型，其余则归入正常类别。为了确定对象在图像中的相对位置，我们将整个图像划分为 $3 \times 3$ 网格，定义区域如右上、上、左上、左、中、右、右下、左下和下。根据对象中心像素坐标，我们相应地分配其相对位置。

Table 4. Example of relationships between different objects used in the proposed instruction dataset.   

<table><tr><td>Categories</td><td>Example</td></tr><tr><td>Ships and Harbors</td><td>(e.g. &quot;anchored at, parked at&quot;)</td></tr><tr><td>Track Field and Soccer Field</td><td>(e.g. &quot;Surrounded by, Inside&quot;)</td></tr><tr><td>Vehicles, Bridge, Road, Roundabout</td><td>(e.g. &quot;passing through, passing through&quot;)</td></tr><tr><td>Vehicles and Building</td><td>(e.g. &quot;parked&quot;)</td></tr><tr><td>Airport and Plane</td><td>(e.g. &quot;parked&quot;)</td></tr><tr><td>Ship and Helipad</td><td>(e.g. &quot;on, contains&quot;)</td></tr></table>

为了定义给定图像中物体之间的关系，我们根据边界框之间的距离对不同物体进行分组，对于每个子图，我们根据类别标签分配不同的物体关系。表4展示了各种物体关系的示例。为了建立诸如“被包围”的关系，我们交叉参考像素级坐标，以验证一个物体是否完全包含在另一个物体内。表达生成：为了模拟自然语言表达，我们使用基于[39]的预定义文本模板。短语模板包含来自表3的属性 $\{ \mathrm { a } 1 , . . . , \mathrm { a } 5 \}$。对于同一类别的一组物体，表达的公式化如下：

Table 5. Zero-shot scene classification accuracy comparison on AID [33] and UCMerced [35] datasets. In comparison to other generic VLMs, GeoChat performs favorably well.   

<table><tr><td>Model</td><td>UCMerced</td><td>AID</td></tr><tr><td>Qwen-VL [1]</td><td>62.90</td><td>52.60</td></tr><tr><td>MiniGPTv2 [4]</td><td>4.76</td><td>12.90</td></tr><tr><td>LLaVA-1.5 [17]</td><td>68.00</td><td>51.00</td></tr><tr><td>GeoChat</td><td>84.43</td><td>72.03</td></tr></table>

$$
^ { \mathrm { , } } \mathrm { T h e / A } \ \langle a 3 \rangle \ \langle a 2 \rangle a 1 \langle \mathrm { i n / o n  t h e } \ a 4 \rangle \ : ^ { \mathrm { , } }
$$

可能缺失的属性用 $\langle \rangle$ 包含，属性 $\{ \mathsf { a } 2 , \mathsf { a } 3 \}$ 可以按任意顺序排列。同样，句子模板包含关系属性 a5，通过这种结构在两个对象之间建立连接：这里的索引 $i$ 和 $j$ 代表第 $i$ 和第 $j$ 个对象。视觉定位：虽然自然图像领域有可用的指称表达数据集 [36, 37]，但在遥感领域却缺乏。为此，我们使用简短描述作为指称表达，以创建三种不同类型的问题回答对，即定位图像描述、指称表达和区域级字幕，如表 1 所述。

# 5. 实验

# 5.1. 实施细节

我们用预训练的 CLIP-ViT [24] 初始化模型的权重，并使用 LLM (Vicuna-v1.5 [7]) 进行 LoRA [11] 微调。通过低秩适配，我们利用 LoRA 对参数 $W _ { q }$ 和 $W _ { v }$ 进行优化，设定的秩 $r$ 为 64。在整个过程中，模型的训练始终在 $504 \times 504$ 的图像分辨率下进行。每个训练步骤都会结合专门设计的多模态指令模板，以应对各种视觉-语言任务。我们使用带余弦学习率调度的 AdamW [21] 优化器来训练模型，保持全局批量大小为 144。我们将模型训练分为两个阶段，首先使用所有数据集训练 1 个周期，约 2400 步，接着进入第二阶段，仅对基础数据集再训练 1600 步。

# 5.2. 场景分类

用于评估的数据集。对于场景分类，我们使用 AID [33] 和 UCMerced [35] 来评估我们的模型。AID [33] 是一个大型航空图像集合，汇编自

Table 6. Comparisons with general zero-shot (top) and RS-VQA specialized (middle) models on RSVQA-LRBEN [20] dataset for VQA task. [1, 4, 17] are evaluated in zero-shot setting. GeoChat outperforms other zero-shot models and performs competitively to SoTA-supervised models like RSGPT which are specifically finetuned on target dataset (while ours is a generic model not specifically finetuned on target dataset).   

<table><tr><td>Method</td><td>Presence Comparison</td><td></td><td></td><td>Rural/Urban Avg. Accuracy</td></tr><tr><td>LLaVA-1.5[17]</td><td>55.46</td><td>68.20</td><td>59.00</td><td>62.77</td></tr><tr><td>Qwen-vl-Chat [1]</td><td>38.57</td><td>67.59</td><td>61.00</td><td>55.35</td></tr><tr><td>MiniGPTv2 [4]</td><td>55.16</td><td>55.22</td><td>39.00</td><td>54.96</td></tr><tr><td>RSVQA[20]</td><td>87.47</td><td>81.50</td><td>90.00</td><td>86.32</td></tr><tr><td>EasyToHard[38]</td><td>90.66</td><td>87.49</td><td>91.67</td><td>89.94</td></tr><tr><td>Bi-Modal[2]</td><td>91.06</td><td>91.16</td><td>92.66</td><td>91.63</td></tr><tr><td>SHRNet [40]</td><td>91.03</td><td>90.48</td><td>94.00</td><td>91.84</td></tr><tr><td>RSGPT[12]</td><td>91.17</td><td>91.70</td><td>94.00</td><td>92.29</td></tr><tr><td>GeoChat</td><td>91.09</td><td>90.33</td><td>94.00</td><td>90.70</td></tr></table>

Google Earth 图像，共有 30 个类别，如河流、密集住宅区等。这些图像由遥感图像解读领域的专家进行标注。AID [33] 数据集总共有 10,000 张图像，涵盖 30 个类别。这些图像来自不同的国家以及不同的天气条件。为了评估，我们使用 AID [33] 数据集的 $20 \%$ 划分。UCMerced [35] 是一个土地利用场景分类数据集，包含 2,100 张图像和 21 个类别。每张图像的大小为 $2 5 6 \times 2 5 6$。我们将整个 UCMerced [35] 数据集作为零-shot 测试集。

结果。我们用所有类别对模型进行提示，并要求其仅用一个词/短语对图像进行分类。例如，我们输入一个提示：“在给定类别中对图像进行分类：密集居民区，...，学校。用一个词或短语回答。”我们计算在 AID 和 UCMerced 上的零-shot 准确率。GeoChat 在 UCMerced 上的准确率为 $84.43\%$，[35] 在 AID 上为 $72.03\%$，[33]，显著超越其他 VLM，如表 5 所示。值得注意的是，最近的 MiniGPT-4-v2 [4] 未能遵循该特定任务的指示，返回与数据集无关的类别。如果我们将 Vicunav1.5 [7] 的答案传递给它，并询问输出句子是否指向真实标注类别，其准确率接近 $5\%$。相比之下，Qwen-VL 和 LLaVa-1.5 在遵循指令方面表现良好，但由于缺乏领域知识，仍然不如 GeoChat。

# 5.3. 视觉问答

评估数据集。RSVQA-HRBEN [20] 包含 10,569 张高分辨率照片和 1,066,316 对问答对，其中 $61.5\%$ 用于训练，$11.2\%$ 用于验证，$20.5\%$ 和 $6.8\%$ 分别用于测试集 1 和测试集 2。该数据集有三种问题类型：存在性、比较和计数。在评估中，我们使用 RSVQA-HRBEN [20] 的测试集 2，共有 $47 \mathrm{k}$ 对问答。RSVQA-LR [20] 由 772 张低分辨率图像和 77,232 对问答对组成，$77.8\%$ 用于训练，$11.1\%$ 用于验证，$11.1\%$ 用于测试。问题分为四种不同类型：存在性、比较、城乡、和计数。我们在评估中省略了区域和计数问题，因为其答案是数值的，并且可以量化为多个类别。例如，在 RSVQA-LRBEN [20] 数据集中，计数问题被量化为五个类别：0，介于 1 和 10 之间，介于 11 和 100 之间，介于 101 和 1000 之间，以及大于 1000。为了评估，我们使用 RSVQA-LRBEN [20] 的测试集，共有 7k 对问答。

Table 7. Performance $\operatorname { a c c } @ 0 . 5 \% )$ comparison of GeoChat on our benchmark. Small, medium and large refer to the size of the objects   

<table><tr><td>Model</td><td>Small</td><td>Medium</td><td></td><td>Large Single-object grounding Multi-object grounding [refer]</td><td></td><td></td><td>] [grounding] Overall</td><td></td></tr><tr><td>MiniGPTv2 [4]</td><td>1.7</td><td>9.9</td><td>21.9</td><td>9.1</td><td>3.6</td><td>8.2</td><td>2.6</td><td>7.6</td></tr><tr><td>GeoChat</td><td>2.9</td><td>13.6</td><td>21.7</td><td>16.0</td><td>4.3</td><td>10.5</td><td>11.8</td><td>10.6</td></tr></table>

<table><tr><td>Model</td><td>Presence</td><td>Comparison</td><td> Average Accuracy</td></tr><tr><td>Qwen-VL[1]</td><td>66.44</td><td>60.41</td><td>63.06</td></tr><tr><td>LLaVA-1.5[17]</td><td>69.83</td><td>67.29</td><td>68.40</td></tr><tr><td>MiniGPTv2[4]</td><td>40.79</td><td>50.91</td><td>46.46</td></tr><tr><td>GeoChat</td><td>58.45</td><td>83.19</td><td>72.30</td></tr></table>

Table 8. Comparison with other general ZS model's on RSVQA-HRBEN [20] dataset for visual qa. All models here have not been trained on the target dataset. GeoChat performs favorably well compared to generic VLMs.

结果。为了将答案限制为简单的“是/否”以及城乡类型问题，我们在每个问题的末尾添加了合适的提示。GeoChat在RSVQA-LRBEN测试集上的表现接近于最先进的专业模型RSGPT [12]，而RSGPT在目标数据集上进行了5次微调。我们在城乡分类子集上也达到了最先进的水平，如表6所示。在RSVQA-HRBEN上，GeoChat在零样本设置下的平均准确率比其他视觉语言模型提高了$3.9\%$，同时在LLaVA-v1.5 [17]上的Comparison子集上超出$15.9\%$，如表8所示。

# 5.4. 视觉定位

评估数据集。针对基础任务的评估，我们提出了一个包含不同引用和基础任务的新基准。我们使用了[30]中的验证集，并采用与第4节相同的数据集创建流程构建测试基准。总共有7653个引用任务、758个基础任务和555个基础描述问题。我们使用准确率 $@ 0 . 5$ 作为评估指标。当预测框与真实标注框的重叠度超过 $0 . 5 \mathrm { I o U }$ 时，计算准确率。

Table 9. Results on grounding description task.   

<table><tr><td>Model</td><td>acc@0.5</td><td>acc@.25</td><td>METEOR</td></tr><tr><td>MiniGPTv2[4]</td><td>10.8</td><td>30.9</td><td>16.4</td></tr><tr><td>GeoChat</td><td>11.7</td><td>33.9</td><td>48.9</td></tr></table>

Table 10. Region level captioning performance.   

<table><tr><td>Model</td><td>ROUGE-1</td><td>ROUGE-L</td><td>METEOR</td></tr><tr><td>MiniGPTv2[4]</td><td>32.1</td><td>31.2</td><td>10.0</td></tr><tr><td>GeoChat</td><td>87.3</td><td>87.2</td><td>83.9</td></tr></table>

结果。表7展示了我们的方法和MiniGPT-4-v2在所提出基准上的表现。总体而言，当处理小物体或需要预测多个框时，模型的表现较差。与MiniGPT-4-v2相比，我们的模型在中等尺寸图像上表现更好。在基础描述任务中，我们计算了生成的多个边界框的交并比（IoU）以及生成的文本答案。我们的模型提供了更好的描述，并且框的准确率略优于MiniGPT-4-v2（表9）。至于区域级标题生成，我们根据与真实区域级标题的文本准确性评估了两个模型（表10）。在ROUGE和ME-TEOR分数方面，我们的模型显著优于MiniGPT-4-v2。

# 6. 结论

尽管大型视觉-语言模型（VLMs）在自然图像领域的最新进展显示出良好前景，但由于独特的领域特定挑战，它们在遥感（RS）场景中的表现仍然有限。为了解决这一空白，我们提出了GeoChat，这是第一个统一的遥感VLM，具备出色的多任务对话能力，能够处理高分辨率的RS图像。GeoChat不仅可以回答图像级别的查询，还能进行区域特定的对话，并通过精确的空间坐标为响应提供依据。我们创建了一种新的遥感多模态指令跟随数据集，包含$3 1 8 k$对图像-指令的配对，格式多样，涵盖多任务类型。GeoChat在场景分类、视觉问答（VQA）、多轮对话、视觉地面标定和指涉物体检测等多个RS任务中实现了强大的零-shot表现，从而建立了一个全面的基准。

# References

[1] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 7, 8   
[2] Yakoub Bazi, Mohamad Mahmoud Al Rahhal, Mohamed Lamine Mekhalfi, Mansour Abdulaziz Al Zuair, and Farid Melgani. Bi-modal transformer-based approach for visual question answering in remote sensing imagery. IEEE Transactions on Geoscience and Remote Sensing, 60:111, 2022. 7   
[3] Christel Chappuis, Valérie Zermatten, Sylvain Lobry, Bertrand Le Saux, and Devis Tuia. Prompt-rsvqa: Prompting visual context to a language model for remote sensing visual question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13721381, 2022. 2   
[4] Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny. Minigpt-v2: large language model as a unified interface for vision-language multi-task learning. arXiv preprint arXiv:2310.09478, 2023. 7, 8   
[5] Gong Cheng, Junwei Han, and Xiaoqiang Lu. Remote sensing image scene classification: Benchmark and state of the art. Proceedings of the IEEE, 105(10):18651883, 2017. 2, 3, 5   
[6] Gong Cheng, Jiabao Wang, Ke Li, Xingxing Xie, Chunbo Lang, Yanqing Yao, and Junwei Han. Anchor-free oriented proposal generator for object detection. IEEE Transactions on Geoscience and Remote Sensing, 60:111, 2022. 5   
[7] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with $9 0 \% ^ { \ast }$ chatgpt quality, 2023. 2, 4, 5, 7   
[8] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning, 2023. 2   
[9] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021. 2   
10] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016. 4   
11] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. 2, 4, 5, 7   
12] Yuan Hu, Jianlong Yuan, Congcong Wen, Xiaonan Lu, and Xiang Li. Rsgpt: A remote sensing vision language model and benchmark. arXiv preprint arXiv:2307.15266, 2023. 1, 2,7,8   
[13] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning. arXiv preprint arXiv:2305.03726, 2023. 2   
[14] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large languageand-vision assistant for biomedicine in one day. arXiv preprint arXiv:2306.00890, 2023. 2   
[15] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, pages 12888 12900. PMLR, 2022. 5   
[16] Xiang Li, Congcong Wen, Yuan Hu, and Nan Zhou. Rs-clip: Zero shot remote sensing scene classification via contrastive vision-language supervision. International Journal of Applied Earth Observation and Geoinformation, 124:103497, 2023. 2   
[17] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023. 2, 3, 5, 7, 8   
[18] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023.2   
[19] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. 1, 2   
[20] Sylvain Lobry, Diego Marcos, Jesse Murray, and Devis Tuia. Rsvqa: Visual question answering for remote sensing data. IEEE Transactions on Geoscience and Remote Sensing, 58 (12):85558566, 2020. 2, 3, 5, 7, 8   
[21] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 7   
[22] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023. 2   
[23] OpenAI. Gpt-4 technical report, 2023. 1   
[24] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, SandhiniAgarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 87488763. PMLR, 2021. 7   
[25] Maryam Rahnemoonfar, Tashnim Chowdhury, Argho Sarkar, Debvrat Varshney, Masoud Yari, and Robin Murphy. Floodnet: A high resolution aerial imagery dataset for post flood scene understanding. arXiv preprint arXiv:2012.02951, 2020. 3, 5   
[26] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion- $. 4 0 0 \mathrm { m }$ . Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 5   
[27] Xian Sun, Peijin Wang, Zhiyuan Yan, Feng Xu, Ruiping Wang, Wenhui Diao, Jin Chen, Jihao Li, Yingchao Feng, Tao Xu, Martin Weinmann, Stefan Hinz, Cheng Wang, and Kun Fu. Fair1m: A benchmark dataset for fine-grained object recognition in high-resolution remote sensing imagery. IS-PRS Journal of Photogrammetry and Remote Sensing, 184: 116130, 2022. 5   
[28] Yi Tay, Minh C Phan, Luu Anh Tuan, and Siu Cheung Hui. Learning to rank question answer pairs with holographic dual lstm architecture. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 695704. ACM, 2017. 3, 4, 5   
[29] Omkar Thawkar, Abdelrahman Shaker, Sahal Shaji Mullappilly, Hisham Cholakkal, Rao Muhammad Anwer, Salman Khan, Jorma Laaksonen, and Fahad Shahbaz Khan. Xraygpt: Chest radiographs summarization using large medical vision-language models. arXiv: 2306.07971, 2023. 2   
[30] Di Wang, Jing Zhang, Bo Du, Dacheng Tao, and Liangpei Zhang. Scaling-up remote sensing segmentation dataset with segment anything model. In arxiv, 2023. 2, 5, 6, 8   
[31] Di Wang, Qiming Zhang, Yufei Xu, Jing Zhang, Bo Du, Dacheng Tao, and Liangpei Zhang. Advancing plain vision transformer toward remote sensing foundation model. IEEE Transactions on Geoscience and Remote Sensing, 61:115, 2023.6   
[32] Junjue Wang, Zhuo Zheng, Ailong Ma, Xiaoyan Lu, and Yanfei Zhong. LoveDA: A remote sensing land-cover dataset for domain adaptive semantic segmentation, 2021. 6   
[33] Gui-Song Xia, Jingwen Hu, Fan Hu, Baoguang Shi, Xiang Bai, Yanfei Zhong, Liangpei Zhang, and Xiaoqiang Lu. Aid: A benchmark data set for performance evaluation of aerial scene classification. IEEE Transactions on Geoscience and Remote Sensing, 55(7):39653981, 2017. 7   
[34] Gui-Song Xia, Xiang Bai, Jian Ding, Zhen Zhu, Serge Belongie, Jiebo Luo, Mihai Datcu, Marcello Pelillo, and Liangpei Zhang. Dota: A large-scale dataset for object detection in aerial images. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 5   
[35] Yi Yang and Shawn Newsam. Bag-of-visual-words and spatial extensions for land-use classification. In Proceedings of the 18th SIGSPATIAL international conference on advances in geographic information systems, pages 270279, 2010. 7   
[36] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the Association for Computational Linguistics, 2:6778, 2014. 7   
[37] Licheng Yu, PaticPrn, Shan Yng, Alexander Berg, and Tamara L Berg. Modeling context in referring expressions. In Computer Vision-ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14, pages 6985. Springer, 2016. 7   
[38] Zhenghang Yuan, Lichao Mou, Qi Wang, and Xiao Xiang Zhu. From easy to hard: Learning language-guided curriculum for visual question answering on remote sensing data. IEEE Transactions on Geoscience and Remote Sensing, 60: 111, 2022. 2, 7   
[39] Yang Zhan, Zhitong Xiong, and Yuan Yuan. Rsvg: Exploring data and models for visual grounding on remote sensing data. IEEE Transactions on Geoscience and Remote Sensing, 61: 113, 2023. 6   
[40] Zixiao Zhang, Licheng Jiao, Lingling Li, Xu Liu, Puhua Chen, Fang Liu, Yuxuan Li, and Zhicheng Guo. A spatial hierarchical reasoning network for remote sensing visual question answering. IEEE Transactions on Geoscience and Remote Sensing, 61:115, 2023. 2, 7   
[41] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 2   
[42] Usman Zia, M Mohsin Riaz, and Abdul Ghafoor. Transforming remote sensing images to textual descriptions. International Journal of Applied Earth Observation and Geoinformation, 108:102741, 2022. 2