# 哆啦A梦GPT $\textcircled{9}$ : 以视频智能体为例，理解动态场景与大型语言模型

杨宗信 1 陈贵昆 1 李小迪 1 王文冠 1 杨毅 1

# 摘要

# 1. 引言

近年来，基于大型语言模型（LLM）的视觉智能体主要集中于解决基于图像的任务，这限制了它们理解动态场景的能力，从而使其距离实际应用，如指导学生进行实验和识别错误，仍然相去甚远。因此，本文探讨了DoraemonGPT，这是一个由LLM驱动的全面且概念优雅的系统，用于理解动态场景。考虑到视频模态更能反映现实场景瞬息万变的特性，我们将DoraemonGPT示例化为一个视频智能体。给定一个带有问题/任务的视频，DoraemonGPT首先将输入视频转换为一个符号记忆，存储与任务相关的属性。这种结构化表示方式允许通过精心设计的子任务工具进行时空查询和推理，最终产生简洁的中间结果。鉴于LLM在专业领域（例如，分析实验背后的科学原理）内知识有限，我们引入可即插即用的工具，以评估外部知识并处理跨不同领域的任务。此外，基于蒙特卡罗树搜索的方法引入了一种新颖的LLM驱动的规划器，以探索各种工具调度的大规模规划空间。该规划器通过反向传播结果的奖励迭代寻找可行的解决方案，并可以将多个解决方案汇总为改进后的最终答案。我们在三个基准和多个实际场景中全面评估了DoraemonGPT的有效性。项目页面：https://z-x-yang.github.io/doraemon-gpt。基于大型语言模型（LLM）的进展（OpenAI，2021；Anil等，2023；Touvron等，2023；OpenAI，2023；Chiang等，2023），近期的LLM驱动智能体（Surís等，2023；Shen等，2023；Gupta & Kembhavi，2023）在将复杂的图像任务分解为可管理的子任务并逐步解决这些任务方面展现了前景。尽管静态图像已经得到了广泛研究，但现实环境本质上是动态的（Wu等，2021）且持续变化（Smith等，2022）。通常，捕获动态场景是一个数据密集型过程，通常通过将静态图像流处理成视频来完成。反过来，视频的时空推理在现实中的识别、语义描述、因果推理等方面至关重要。为了理解动态场景，开发LLM驱动的智能体以处理视频具有重要意义，但也面临着巨大的挑战：i) 时空推理。关于实例关系的推理在任务分解和决策制定中至关重要。这些关系可能与空间（Santoro等，2017）、时间（Zhou等，2018）或它们的时空组合相关。ii) 更大的规划空间。与图像模态相比，高层语义中的动作及其意图通常只能从时间视觉观察中推断（Tapaswi等，2016）。换句话说，推理时间语义是必要的，并将扩大分解动态视频任务的搜索空间。iii) 有限的内部知识。由于现实世界的瞬息万变和/或缺乏对专有数据集的学习，LLM无法编码理解每个视频所需的所有知识（Peng等，2023a）。

鉴于前面的讨论，我们呈现DoraemonGPT，这是一种直观而灵活的基于大型语言模型的系统，兼容各种基础模型和应用。DoraemonGPT作为一个视频智能体，具备三个理想能力：首先，在推理之前收集与给定任务相关的信息。在DoraemonGPT中，给定动态任务的分解是由智能体基于时空关系的推理决定的，这些关系是通过实例位置、动作、场景变化等信息属性推导出来的。然而，重要的是要注意，只有与任务解决相关的信息是关键的，因为收集过多的上下文往往会妨碍大型语言模型的能力（Shi et al., 2023）。其次，在做出决策之前探索更好的解决方案。基于大型语言模型的规划（例如，Yao et al. (2022); Shinn et al. (2023)）将高层任务分解为子任务或动作序列。将一个动作序列视为包含所有可能序列的树中的从根到叶的路径，规划可以视为从树状搜索空间中寻找最佳决策（Yao et al., 2023）。考虑到在动态场景中解决任务的大规划空间，通过树状搜索方法提示大型语言模型（Korf, 1985; Haralick & Elliott, 1980; Browne et al., 2012）为获得更好的解决方案提供了机会，甚至还可以考虑从不同角度看待任务的可能性。第三，支持知识扩展。正如人类在应对领域特定问题时查阅参考书籍，DoraemonGPT被设计用来从一系列给定的外部知识源中选择最相关的知识来源（例如，搜索引擎、教科书、数据库等），并在规划过程中从中查询信息。

![](images/1.jpg)  
.

更具体地说，DoraemonGPT 具有记忆、工具和规划者的结构（图 1c）：i) 任务相关符号记忆 (§2.1)。为了收集与给定视频和任务相关的信息，我们考虑将时空属性解耦为两种记忆：空间主导和时间主导。在构建这些记忆之前，使用大语言模型（LLMs）来确定它们与给定任务的相关性，并仅保留有用的记忆。然后，基础模型被用来提取空间主导属性（例如，实例轨迹、描述等）或时间主导属性（例如，视频描述、音频讲话等），并将它们整合到查询表中，这使得大语言模型能够通过使用符号语言（例如 SQL 语言）访问信息。ii) 子任务 (§2.1) 和知识 (§2.2) 工具。

为了压缩规划者的上下文/文本长度并提高效率，我们通过设计一系列子任务工具简化了内存信息查询。每个工具专注于不同类型的时空推理（例如，“如何...”、“为什么...”等），利用具有特定任务提示和示例的个体LLM驱动的子智能体。此外，专用的知识工具可以整合外部来源，以应对需要特定领域知识的任务。三、蒙特卡洛树搜索（MCTS）规划器（§2.3）。为了高效探索广大的规划空间，我们提出了一种新颖的类似树搜索的规划器。该规划器通过反向传播答案的奖励并选择一个高度扩展的节点，迭代寻找可行解，以扩展新的解决方案。在汇总所有结果后，规划器得出一个信息丰富的最终答案。为设计树搜索规划器，我们将MCTS整合到我们的DoraemonGPT中（Coulom, 2006；Kocsis & Szepesvári, 2006；Browne et al., 2012），该方法在大搜索空间中寻找最优决策方面已显示出有效性（Vodopivec et al., 2017），尤其是在游戏AI领域（Gelly et al., 2006；Chaslot et al., 2008；Silver et al., 2017）。结合上述设计，DoraemonGPT有效处理动态时空任务，支持全面探索多种潜在解决方案，并能够通过利用多源知识扩展其专业知识。对三个基准的广泛实验（Xiao et al., 2021；Lei et al., 2020；Seo et al., 2020）表明，我们的DoraemonGPT在因果/时间/描述推理和视频对象识别方面显著优于近期的LLM驱动竞争者（例如，ViperGPT Surís et al.（2023））。此外，我们的MCTS规划器超越了简单搜索方法和其他基线。此外，DoraemonGPT能够处理更多复杂的真实环境任务，这些任务在近期的方法中曾被忽视或无法应用（Surís et al., 2023；Li et al., 2023b）。

![](images/2.jpg)  
F supported. For planning, our CTS Planner (2.3) decomposes the question int a action sequence by exploring $N$ feasible solutions, which can be further summarized into an informative answer.

# 2. 哆啦A梦GPT

概述。如图 2 所示，DoraemonGPT 是一个基于大型语言模型的智能体，能够利用各种工具将复杂的动态视频任务分解为子任务并加以解决。给定一个视频 $(V)$ 和一个文本任务/问题 $(Q)$，DoraemonGPT 首先根据 $Q$ 的任务分析从 $V$ 中提取相关的符号记忆 (§2.1)。接下来，利用蒙特卡洛树搜索 (MCTS) 规划器 (§2.3)，DoraemonGPT 自动安排查询符号记忆、访问外部知识 (§2.2) 以及调用其他实用工具（视频修复等）的工具集，以解决问题 $Q$。最终，规划器探索广阔的规划空间，返回多个可能的答案，并总结出一个改进的答案。

# 2.1. 与任务相关的符号记忆 (TSM)

视频是复杂的动态数据，包含时空关系。针对视频 $V$ 的问题 $Q$，仅有部分相关属性对解决方案至关重要，而大量无关信息则不予考虑。因此，我们提出在解决 $Q$ 之前，将与之相关的潜在视频信息提取并存储到 TSM 中。TSM 构建。为了构建 TSM，我们采用了一种简单的上下文学习方法（Brown et al., 2020）根据问题 $Q$ 选择 TSM 的任务类型。我们将每种类型 TSK 的任务描述放入我们的 LLM 驱动规划器的上下文中，该规划器将被提示预测适合的 TSM，格式为“Action: $\langle T S M \lrcorner t y p e \rangle$ construction...”。随后，将调用构建相应 TSM 的 API，以提取与任务相关的属性并将其存储在 SQL 数据表中，该表可以通过符号语言（即 SQL）进行访问。对于视频任务的分类没有标准化的标准。在 DoraemonGPT 中，我们选择时空解耦的视角，这在视频表示学习中得到了广泛应用（Bertasius et al., 2021; Arnab et al., 2021），以设计两种记忆类型：•空间主导记忆主要用于解决与特定目标（如人或动物）或者它们的空间关系相关的问题。我们使用多目标跟踪方法（Maggiolino et al., 2023）来检测和跟踪实例。每个实例具有包括唯一 ID、语义类别、轨迹和分割用于定位、通过 (Li et al., 2022; 2023a) 提取的外观描述用于基于文本的定位，以及行为分类的属性。

<table><tr><td rowspan=1 colspan=1>Attribute</td><td rowspan=1 colspan=1>Used Model</td><td rowspan=1 colspan=1>Explanation</td></tr><tr><td rowspan=1 colspan=3>Space-dominant Memory</td></tr><tr><td rowspan=1 colspan=1>ID number</td><td></td><td rowspan=1 colspan=1>A unique ID assigned to an instance</td></tr><tr><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>YOLOv8 (Jocher et al., 2023)/Grounding DINO (Liu et al., 2023c)</td><td rowspan=1 colspan=1>The category of an instance, e.g., person</td></tr><tr><td rowspan=1 colspan=1>Trajectory</td><td rowspan=1 colspan=1>Deep OC-Sort (Maggiolino et al., 2023)/DeAOT (Yang &amp; Yang, 2022)</td><td rowspan=1 colspan=1>An instance&#x27;s bounding box in each frame</td></tr><tr><td rowspan=1 colspan=1>Segmentation</td><td rowspan=1 colspan=1>YOLOv8-Seg (Jocher et al., 2023)/DeAOT (Yang &amp; Yang, 2022)</td><td rowspan=1 colspan=1>An instance&#x27;s segmentation mask in each frame</td></tr><tr><td rowspan=1 colspan=1>Appearance</td><td rowspan=1 colspan=1>BLIP (Li et al., 2022) / BLIP-2 (Li et al., 2023a)</td><td rowspan=1 colspan=1>A description of an instance&#x27;s appearance</td></tr><tr><td rowspan=1 colspan=1>Action</td><td rowspan=1 colspan=1>Intern Video (Wang et al., 2022)</td><td rowspan=1 colspan=1>The action of an instance</td></tr><tr><td rowspan=1 colspan=3>Time-dominant Memory</td></tr><tr><td rowspan=1 colspan=1>Timestamp</td><td></td><td rowspan=1 colspan=1>The timestamp of a frame/clip</td></tr><tr><td rowspan=1 colspan=1>Audio content</td><td rowspan=1 colspan=1>Whisper (Radford et al., 2023)</td><td rowspan=1 colspan=1>Speech recognition results of the video</td></tr><tr><td rowspan=1 colspan=1>Optical content</td><td rowspan=1 colspan=1>OCR (PaddlePaddle, 2023)</td><td rowspan=1 colspan=1>Optical character recognition results of the video</td></tr><tr><td rowspan=1 colspan=1>Captioning</td><td rowspan=1 colspan=1>BLIP (Li et al., 2022)/BLIP-2 (Li et al., 2023a)/InstructBlip (Dai et al., 2023)</td><td rowspan=1 colspan=1>Frame-level/clip-level captioning results</td></tr></table>

时间主导记忆专注于构建与视频相关的时间信息。它需要理解视频中的内容。存储在该记忆中的属性包括时间戳，通过自动语音识别（ASR）获取的音频内容（Radford 等，2023）、通过光学字符识别（OCR）获取的光学内容（PaddlePaddle，2023）、通过BLIPs进行的逐帧字幕（Li 等，2022；2023a；Dai 等，2023）、通过去重相似和连续的逐帧结果获得的片段级字幕等。表1提供了我们的时间主导内存（TSMs）相应提取模型的属性类型。子任务工具。尽管基于大型语言模型（LLM）的智能体（Hu 等，2023；Li 等，2023b）可以通过对整个记忆的上下文学习评估外部信息，或生成符号句子以访问记忆，这些方法可能显著增加上下文的长度，导致推理过程中关键信息的遗漏或受到冗余上下文的影响。因此，我们提供了一系列子任务工具，负责通过回答子任务问题查询我们的TSMs信息（Shi 等，2023；Liu 等，2023a）。基于LLM的规划器通过其上下文描述学习每个子任务工具的功能，该描述包括子任务描述、工具名称和工具输入。为了调用子任务工具的API，DoraemonGPT解析LLM生成的命令，如“动作：tool_name]。输入：video_name>#(sub_question...”。为了与上述两种TSMs协同工作，我们设计了具有不同子任务描述的子任务工具，以解决不同的子问题，包括：•何时：与时间理解相关，例如，“狗在沙发旁边走过时是什么时候？”•为什么：与因果推理相关，例如，“为什么女士要摇动玩具？”•什么：描述所需信息，例如，“实验的名称是什么？”•如何：某事的方式、手段或特征，例如，“宝宝是如何保证自身安全的？”•计数：数某物，例如，“房间里有多少人？”•其他：不在上述工具中的问题，例如，“谁在最后滑得更远？”这些工具的API功能也建立在LLM之上。每个子任务工具功能都是一个独立的基于LLM的智能体，可以生成SQL查询我们的TSMs并回答给定的子任务问题。不同的子任务智能体根据其目的具有不同的上下文示例。请注意，一个子问题可能适用于两个或多个子工具（例如，“在玩玩具之前，宝宝在做什么？”与什么和什么时候相关），我们的蒙特卡洛树搜索规划器（§2.3）能够探索不同的选择。

# 2.2. 知识工具与其他工具

在处理复杂问题时，基于大型语言模型（LLM）的智能体有时无法仅凭视频理解和在训练过程中学习到的隐含知识做出准确决策。因此，DoraemonGPT支持整合外部知识源，以帮助LLM理解输入视频/问题中的专业内容。在DoraemonGPT中，可以通过使用独立的知识工具以即插即用的方式集成知识源。与子任务工具（§2.1）类似，知识工具由两部分组成：i) 在上下文中的知识描述，用于描述给定的知识源；ii) 查询信息的API函数，通过问答形式从源中获取信息。我们考虑三种类型的API函数以涵盖不同的知识：i) 符号知识是指以结构化格式呈现的信息，如Excel或SQL表格。该API函数是一个象征性问答子智能体，类似于我们的子任务工具（§2.1）。ii) 文本知识包括通过自然语言文本表达的知识，如研究出版物、参考书籍等。该API函数基于文本嵌入和搜索构建（OpenAI，2022）。iii) 网络知识指的是从互联网搜索到的知识。该API函数是一个搜索引擎API，如谷歌、必应等。除了知识工具，DoraemonGPT还支持整合通用实用工具，这些工具通常在近期的LLM驱动智能体中使用（Xi等，2023），以帮助完成专业视觉任务，例如视频编辑和图像修复。

![](images/3.jpg)  
Rec syer

# 2.3. 蒙特卡洛树搜索 (MCTS) 规划器

以前的基于LLM的规划器（Shen et al., 2023；Surís et al., 2023；Gupta & Kembhavi, 2023）将给定的$Q$分解为一系列动作/子任务，并逐步解决。这样的策略可以看作是一种贪婪搜索方法，它生成一条动作节点链直到最终答案。与一些研究（Yao et al., 2024；Zhuang et al., 2024）类似，我们将基于LLM的规划的大型规划空间视为一棵树。此外，我们认为单次尝试可能无法得出正确结果，或者可能存在更好的解决方案。为了有效探索规划空间，我们提出了一种新颖的类似树搜索的规划器，配备MCTS（Coulom, 2006；Kocsis & Szepesvári, 2006；Browne et al., 2012），该方法在搜索大型树结构时非常实用。我们将问题输入$Q$定义为根节点$v_{0}$，而一个动作或工具调用是非根节点，因此一系列动作可以视为从根节点到叶节点的路径。在我们的MCTS规划器中，非根节点是ReAct（Yao et al., 2022）风格的步骤，形式为（思考，动作，动作输入，观察），并且叶节点具有最终答案。此外，规划器迭代执行以下四个阶段$N$次并产生$N$个解决方案：节点选择。每次迭代开始时，选择一个可扩展节点以规划新解决方案。在第一次迭代中，仅可选择根节点$v_{0}$。在随后的迭代中，我们根据其采样概率随机选择一个非叶节点，该概率公式为$P(v_{i}) = S o f t m a x(R_{i})$，其中$R_{i}$是节点$v_{i}$的奖励值，初始化为0，并在奖励反向传播阶段进行更新。具有更高奖励的节点被选择的概率更大。分支扩展。将一个子节点添加到所选的可扩展节点，从而创建一个新分支。为了利用LLM生成与之前子节点不同的新工具调用，我们将历史工具动作添加到LLM的提示中，并指示其做出不同的选择。这样的上下文提示将在后续链执行中被移除，以朝向新的最终答案。链执行。在扩展新分支后，我们使用逐步的基于LLM的规划器（Yao et al., 2022）生成新解。执行过程由工具调用的步骤/节点链组成，直到获得最终答案或遇到执行错误而终止。奖励反向传播。在获得叶节点/结果节点$v_{l}$之后，我们将逐渐将其奖励传播到其祖先节点，直到$v_{0}$。在这里，我们考虑两种奖励：失败：规划器产生意外结果，例如，工具调用失败或结果格式不正确。这种情况下的奖励${ \boldsymbol { R_{v_{l}} } }$设为负值（例如，-1）。非失败：规划器成功产生了结果，但不确定结果是否正确，即真实标注数据。在这里，$R_{v_{l}}$设为正值（例如，1）。

为简化起见，设 $\alpha$ 为正基奖励，我们设定 $R _ { v _ { l } } = \pm \alpha$ 分别表示失败和非失败。根据（Liu et al., 2023a）的研究，LLMs 生成的结果与开头（初始提示）和结尾（最终节点）的上下文更为相关。我们认为应更多地将奖励应用于靠近 $v _ { l }$ 的节点。因此，反向传播函数被表述为 $R _ { v _ { i } } R _ { v _ { i } } + R _ { v _ { l } } \acute { e } ^ { \beta ( \bar { 1 } - \bar { d } ( v _ { i } , v _ { l } ) ) }$，其中 $d ( v _ { i } , v _ { l } )$ 表示 $v _ { i }$ 和 $v _ { l }$ 之间的节点距离，而 $\beta$ 是控制奖励衰减率的超参数。节点距离越远，奖励的衰减比例越大，$e ^ { \beta ( 1 - d ( v _ { i } , v _ { l } ) ) }$。通常，设置更高的 $\alpha / \beta$ 会增加扩展节点接近叶节点（非失败答案）的概率。在所有 MCTS 迭代后，规划器最多会产生 $N$ 个非失败答案，我们可以使用 LLMs 对所有答案进行总结，以生成一个信息丰富的答案。代理使用相同的 LLM，即 GPT-3.5-turbo。†：使用官方发布的代码重新实现。$^ \ddag$ 我们为 ViperGPT （Surís et al., 2023）配备了 DeAOT （Yang & Yang, 2022；Cheng et al., 2023）以实现目标跟踪和分割。

<table><tr><td></td><td>Method</td><td>Pub.</td><td>Accc</td><td>Acct</td><td>AccD</td><td>Avg AccA</td></tr><tr><td rowspan="4">20</td><td>HME (Fan et al., 2019)</td><td>CVPR19</td><td>46.2</td><td>48.2</td><td>58.3 50.9</td><td>48.7</td></tr><tr><td>VQA-T (Yang et al., 2021a)</td><td>ICCV21</td><td>41.7</td><td>44.1</td><td>60.0</td><td>48.6 45.3</td></tr><tr><td>ATP (Buch et al., 2022)</td><td>CVPR22</td><td>53.1</td><td>50.2 66.8</td><td>56.7</td><td>54.3</td></tr><tr><td>VGT (Xiao et al., 2022)</td><td>ECCV22 CVPR23</td><td>52.3</td><td>55.1 64.1</td><td>57.2</td><td>55.0</td></tr><tr><td rowspan="4">2</td><td>MIST (Gao et al., 2023b)</td><td>ICCV23</td><td>54.6</td><td>56.6 41.0</td><td>66.9 62.3</td><td>59.3 57.2</td></tr><tr><td>†ViperGPT (Surís et al., 2023) VideoChat (Li et al., 2023b)</td><td>arXiv23</td><td>43.2</td><td></td><td>49.4</td><td>45.5</td></tr><tr><td></td><td></td><td>50.2</td><td>47.0 65.7</td><td>52.5</td><td>51.8</td></tr><tr><td>DoraemonGPT (Ours)</td><td>ICML24</td><td>54.7 50.4</td><td>70.3</td><td>54.7</td><td>55.7</td></tr></table>

(a) NExT-QA（Xiao等，2021） (b) Ref-YouTube-VOS（Seo等，2020）

<table><tr><td></td><td>Method</td><td>Pub.</td><td>J</td><td>F</td><td>J&amp;F</td></tr><tr><td>Sde</td><td>CMSA (Ye et al., 2019) URVOS (Seo et al., 2020) VLT (Ding et al., 2021) ReferFormer (Wu et al., 2022a) SgMg (Miao et al., 2023) OnlineRefer (Wu et al., 2023b)</td><td>CVPR19 ECCV20 ICCV21 CVPR22 ICCV23 ICCV23</td><td>36.9 47.3 58.9 58.1 60.6 61.6</td><td>43.5 56.0 64.3 64.1 66.0 67.7</td><td>40.2 51.5 61.6 61.1 63.3 64.8</td></tr><tr><td>A</td><td>‡ViperGPT (Surís et al., 2023) DoraemonGPT (Ours)</td><td>ICCV23 ICML24</td><td>24.7 63.9</td><td>28.5 67.9</td><td>26.6 65.9</td></tr></table>

另外，对于单选/多选题，我们可以通过投票过程来确定最终答案。

# 3. 实验

# 3.1. 实验设置

数据集。为了全面验证我们算法的实用性，我们在三个数据集上进行实验，即 NExT-QA（Xia0 等，2021），TVQA $^+$（Lei 等，2020）和 Ref-YouTube-VOS（Seo 等，2020）。这些数据集的选择旨在涵盖视频问答和指代视频目标分割任务中的常见动态场景。• NExT-QA 包含 34,132/4,996 对视频-问题对用于训练/验证。每个问题都标注有问题类型（因果/时间/描述性）和 5 个答案候选项。我们从训练集中随机抽取每种类型 30 个样本（共 90 个问题）用于消融研究，验证集用于方法比较。$\mathbf{TVQA}^+$ 是 TVQA（Lei 等，2018）数据集的增强版，增加了 310.8K 的边界框，以连接问题和答案中的视觉概念与视频中呈现的对象。为了评估，我们从验证集随机抽取 900 个样本，如（Gupta & Kembhavi，2023）所示，最终共得到 900 个问题（s_val）。• Ref-YouTube-VOS 是一个大规模的指代视频目标分割数据集，包含约 15,000 个指代表达，与超过 3,900 个视频相关，涵盖多种场景。在我们的实验中，我们使用 Ref-YouTube-VOS（Seo 等，2020）的验证集（包含 202 个视频和 834 个带表达的对象）来验证我们在像素级时空分割方面的有效性。

评估指标。对于问答任务，我们采用标准指标（Xiao et al., 2021），即top-1准确率进行评估。在NExT-QA上，我们还报告因果准确率$( \operatorname { A c c } _ { \mathbf { C } } )$、时间准确率$( \operatorname { A c c } _ { \mathrm { T } } )$、描述性准确率$( \mathrm { A c c } _ { \mathrm { D } } )$，平均准确率（$\operatorname { A c c } _ { \mathbf { C } }$、$\operatorname { A c c } _ { \mathrm { T } }$和$\operatorname { A c c } _ { \mathrm { D } }$的平均值）和总体准确率$( \operatorname { A c c } _ { \mathrm { A } } )$（所有问题的总体准确率）。对于参考对象分割，我们在Ref-YouTube-VOS的官方挑战服务器上评估性能，指标为区域相似度$( \mathcal { T } )$和轮廓准确率$( \mathcal { F } )$的平均值，统称为$\mathcal { I } \& \mathcal { F }$。实现细节。我们使用OpenAI提供的GPT-3.5-turbo API作为我们的语言模型。正如表1所总结的，我们使用BLIP系列（Dai et al., 2023）进行标注，YOLOv8（Jocher et al., 2023）和Deep OC-Sort（Maggiolino et al., 2023）进行目标跟踪，PaddleOCR（PaddlePaddle, 2023）进行光学字符识别，InternVideo（Wang et al., 2022）进行动作识别，以及Whisper（Radford et al., 2023）进行自动语音识别。我们的实验在上下文学习（ICL）设置下进行。为确保公平，外部知识未用于定量或定性比较。对手。我们引入了几个开源的以语言模型驱动的代理进行比较。ViperGPT（Surís et al., 2023）利用代码生成模型通过提供的API从视觉-语言模型生成子程序，以此生成的Python代码解决给定任务。VideoChat（Li et al., 2023b）是一个端到端的聊天为中心的视频理解系统，集成了多个基础模型和语言模型以构建聊天机器人。对于其他竞争者，我们未报告其性能，因为他们未公开视频任务的代码或甚至开源。超参数。我们为奖励反向传播设置$\alpha = 1$和$\beta = 0.5$（§2.3）。在VQA实验中，探索$N = 2$的解决方案提供了更好的准确率-成本权衡。

# 3.2. 零-shot 视频问答

NExT-QA。表2a展示了我们DoraemonGPT与多个领先的监督VQA模型和基于大语言模型的系统的比较。如所示，DoraemonGPT在与近期提出的监督模型相比时表现出竞争力。特别是在描述性问题上，它显示出更具前景的改进，甚至超过了以前的最先进模型MIST（Gao等，2023b）$( 7 0 . 3 \ : \nu s \ : 6 6 . 9 )$。主要原因在于我们的任务相关符号记忆可以提供足够的信息用于推理。在时间问题方面，监督模型略优于我们，主要是由于它们经过精心设计的架构学习到了潜在模式。此外，DoraemonGPT的性能超越了近期的对比作品，即ViperGPT和VideoChat。具体而言，它在四种问题类型上分别以11.5/9.4/8.0/10.2$\left( \mathrm { A c c _ { C } / A c c _ { T } / A c c _ { D } / A c c _ { A } } \right)$超越了ViperGPT，以${ \bf 4 . 5 / 3 . 4 / 4 . 6 / 3 . 9 }$超越了VideoChat。这些结果表明我们的基于TSM的MCTS规划器的有效性。推理示例见A.3。

![](images/4.jpg)

![](images/5.jpg)  
ualaaoReferid Obje S 3.RYoTub e.   
Figure 5. Comparison on $\mathrm { T V Q A + }$ (Lei et al., 2020) (§3.2).

$\mathbf { T V Q A } +$ 图 5 的结果再次确认了我们方法的优越性。DoraemonGPT 的表现比 ViperGPT 和 VideoChat 分别高出 $1 0 . 2 \%$ 和 $5 . 9 \%$。ViperGPT 的性能（$3 0 . 1 \%$）低于 VideoChat 和专门为动态视频设计的 DoraemonGPT。这与 NExT-QA 的研究结果一致。

# 3.3. 零样本引用对象分割

Ref-YouTube-VOS。我们进一步评估了DoraemonGPT与最先进的监督式参照视频目标分割模型和基于大语言模型的智能体的表现，具体总结见表2b。得益于我们与任务相关的符号记忆，DoraemonGPT（没有在Ref-YouTube-VOS上学习）能够有效地将视频实例与文本描述对应，并显著超过近期的监督模型（$65.9 \%$对比$64.8 \%$），例如OnlineRefer（吴等，2023b）。相对而言，竞争者智能体ViperGPT由于缺乏良好设计的视频信息记忆，仅达到$26.6 \%$，未能准确地定位所指对象或在视频中进行追踪。如图4所示，我们的DoraemonGPT在识别、追踪和分割用户提到的参照对象方面展现了更高的准确性。相比之下，ViperGPT遭遇了识别对象未能完全匹配语义和描述特征的失败案例。我们在参照视频目标分割方面的优势进一步证明了构建符号视频记忆的必要性。

# 3.4. 实际场景示例

DoraemonGPT 展示了多样化的技能，例如检查实验操作、视频理解和视频编辑。它巧妙地通过探索多种推理路径和利用外部资源来应对复杂问题，以提供全面的答案。更多细节请见 A.2。

# 3.5. 诊断实验

为了更深入地了解DoraemonGPT，我们在NExT-QA上进行了一系列消融实验，涉及任务相关的符号记忆。首先，我们研究了DoraemonGPT中的基本组件，即用于空间主导（SDM）和时间主导（TDM）信息的符号记忆（§2.1）。结果汇总在表3a中。可以得出两个重要结论。首先，TDM在时间性问题上更受欢迎，而SDM可以为描述性问题提供相关信息。其次，我们的完整系统通过结合SDM和TDM实现了最佳性能，确认了动态查询两种类型符号记忆的必要性。

通过MCTS规划器的多重解决方案。接下来，我们将研究在MCTS规划器的探索过程中答案候选数量的影响。当$N = 1$时，规划器等同于贪婪搜索，仅探索一条节点链，并返回一个答案，即LLM思考中可以终止而无需进一步探索的第一个节点。如表3c所示，逐渐将$N$从1增加到4可实现更好的性能（即$4 3 . 3$ 65.7）。这支持了我们的假设：单一答案远不足以应对动态模态的更大规划空间，并证明了我们的MCTS规划器的有效性。由于NExT-QA（Xiao等，2021）中的问题为单选，因此探索更多答案并不总能带来积极的回报。我们停止使用$N > 5$，因为所需的API调用数量超过了我们的预算。

Table 3. A set of ablative experiments (3.5) about the MCTS planner on NExT-QA (Xiao et al., 2021) rn.   
(a) Essential components   

<table><tr><td rowspan=1 colspan=1>TDM SDM</td><td rowspan=1 colspan=1>Accc Acct AccD</td><td rowspan=1 colspan=1>Acca</td></tr><tr><td rowspan=1 colspan=1>✓✓</td><td rowspan=1 colspan=1>63.326.753.353.323.346.7</td><td rowspan=1 colspan=1>47.841.1</td></tr><tr><td rowspan=1 colspan=1>✓√</td><td rowspan=1 colspan=1>96.746.753.3</td><td rowspan=1 colspan=1>65.7</td></tr></table>

<table><tr><td>Models</td><td>Accc Acct</td><td>AccD</td><td>Acca</td></tr><tr><td>BLIP-2</td><td>51.4 45.5</td><td>63.3</td><td>51.2</td></tr><tr><td>InstructBlip</td><td>54.7 50.4</td><td>70.3</td><td>55.7</td></tr></table>

(b) 标注模型 (c) 答案候选数量 (d) 奖励及衰减率 $N = 4$ (e) 探索策略 $N = 4$

<table><tr><td>N</td><td>Accc Acct AccD</td><td>AccA</td></tr><tr><td>20</td><td>63.3 20.0 46.7 80.0 43.3 46.7 86.7 43.3 53.3</td><td>43.3 56.7 61.1 65.7</td></tr><tr><td></td><td>96.7 46.7</td><td>53.3</td></tr><tr><td>86.7</td><td>43.3 50.0</td><td>60.0</td></tr></table>

<table><tr><td rowspan=1 colspan=1>αβ</td><td rowspan=1 colspan=1>AcccAcctAccD</td><td rowspan=1 colspan=1>AccA</td></tr><tr><td rowspan=1 colspan=1>0.51.0</td><td rowspan=1 colspan=1>86.723.350.0</td><td rowspan=1 colspan=1>53.3</td></tr><tr><td rowspan=1 colspan=1>1.0 0.5</td><td rowspan=1 colspan=1>96.746.753.3</td><td rowspan=1 colspan=1>65.7</td></tr><tr><td rowspan=1 colspan=1>| 0.5 2.0</td><td rowspan=1 colspan=1>86.726.750.0</td><td rowspan=1 colspan=1>54.4</td></tr><tr><td rowspan=1 colspan=1>2.0 0.5</td><td rowspan=1 colspan=1>83.346.750.0</td><td rowspan=1 colspan=1>60.0</td></tr><tr><td rowspan=1 colspan=1>2.02</td><td rowspan=1 colspan=1>80.046.750.0</td><td rowspan=1 colspan=1>58.9</td></tr></table>

<table><tr><td rowspan=1 colspan=1>Strategy</td><td rowspan=1 colspan=1>AcccAcctAccD</td><td rowspan=1 colspan=1>Acca</td></tr><tr><td rowspan=1 colspan=1>DFS</td><td rowspan=1 colspan=1>66.736.750.0</td><td rowspan=1 colspan=1>51.1</td></tr><tr><td rowspan=2 colspan=1>RootUniform</td><td rowspan=2 colspan=1>73.316.746.767.726.750.0</td><td rowspan=1 colspan=1>45.6</td></tr><tr><td rowspan=1 colspan=1>47.8</td></tr><tr><td rowspan=1 colspan=1>MCTS</td><td rowspan=1 colspan=1>96.746.753.3</td><td rowspan=1 colspan=1>65.7</td></tr></table>

在MCTS规划器中的反向传播。然后，我们分析了基准奖励$\alpha$和衰减率$\beta \ ( \ S 2 . 3 )$的影响，这两者控制了我们MCTS规划器的探索过程。表3d中的结果无论使用何种组合的$\alpha$和$\beta$都保持稳定。因此，我们将稍微表现更好的组合$\alpha { = } 1$和$\beta = 0 . 5$设定为默认配置。我们在下一部分中保留了一些特殊组合（例如，当设置$\beta = 1 0 ^ { 8 }$和$R _ { v _ { l } } = 1$时，我们的MCTS规划器变为深度优先搜索$( D F S )$，适用于失败和非失败情况）。规划器使用的探索策略。最后，为了验证我们MCTS规划器的优势，我们将MCTS与几种标准的探索策略进行了比较，即深度优先搜索（DFS）、总是选择根节点的Root和均匀选择节点的Uniform。如表3e所示，我们观察到它们的表现不佳，无法利用结果叶节点的值/奖励，并相应调整搜索策略。与这些简单策略相比，我们的MCTS规划器通过奖励反向传播的指导自适应地对节点进行采样，这在大的解空间中更为有效。这些结果进一步验证了所提MCTS规划器的优越性。此外，与Uniform相比，DFS在时间问题上明显表现优异，但在描述性和因果问题上表现出相当或甚至稍差的效果。我们假设这种差异的产生是因为时间问题通常包含指示具体时段的线索（例如，在开始时），这些线索指向视频中可找到正确答案的具体时间段。DFS能够利用这些线索，而Uniform则无法。字幕模型的影响。我们进一步进行了关于最重要模型之一的实验，即BLIP系列，这是DoraemonGPT感知和识别视觉输入的基本工具。如表3b所示，经过指令调优的模型，即InstructBLIP，表现出更好的性能。这表明DoraemonGPT可以从更强大的基础模型的发展中受益。有关基础模型对DoraemonGPT影响的更多讨论见A.4。

# 4. 相关工作

多模态理解。为了特定任务创建多模态系统进行了多种努力（Lu等，2019；Marino等，2019；2021；Bain等，2021；Lei等，2021；Grauman等，2022；Lin等，2022；Li等，2023d；c；Lei等，2024；Wong等，2022；Chen等，2023a；Hu等，2020；Yang等，2021b）。尽管这些系统在各自领域表现出色，但由于缺乏通用性，它们在更广泛的现实场景中的适用性受到限制。近年来，随着数据量和计算资源的快速发展，通用多模态系统取得了显著进展。具体来说，Frozen（Tsimpoukelli等，2021）就是一个典型例子；它展示了一种使大语言模型具备处理视觉输入能力的可行方式。过去几年里，众多努力致力于构建大型多模态模型（OpenAI，2023；Driess等，2023；Zhu等，2023a；Li等，2022）。考虑到训练成本，几种尝试（Li等，2023a；Yu等，2023）试图为各种任务构建零-shot 系统。另一种策略（Xi等，2023）将会在后面详细介绍，涉及结合多个模型或API来解决组合多模态推理任务。我们的DoraemonGPT在将复杂任务分解为简单任务的精神上有相似之处，但它的设计旨在解决现实场景中动态模态的复杂任务。

基于大语言模型的模块化系统。拆解复杂任务并整合多个中间步骤的结果是人类的固有能力，这推动了科学和工业界的发展（Newell et al., 1972；Wang & Chiew, 2010）。得益于大语言模型令人印象深刻的突现能力，VisProg（Gupta & Kembhavi, 2023）开创了通过将问题分解为可管理的子任务来解决复杂视觉任务的理念。在这一方向上，已取得了巨大的进展，可以根据推理风格将其分为两类：i) 采用固定路径推理（Gupta & Kembhavi, 2023；Wu et al., 2023a；Surís et al., 2023；Lu et al., 2023；Shen et al., 2023；Liang et al., 2023）。它们将给定任务转化为一系列有序的子任务，每个子任务由特定模块处理。例如，ViperGPT（Surís et al., 2023）将求解过程视为具有手动设计 API 的 Python 程序。同样，HuggingGPT（Shen et al., 2023）建模多个基础模型之间的任务依赖关系。 ii) 采用动态路径推理（Nakano et al., 2021；Schick et al., 2023；Yao et al., 2022；Yang et al., 2023；Ga0 et al., 2023a）。考虑到中间结果可能不符合预期，一个前景广阔的途径是同时进行规划和执行。这种互动范式（Yao et al., 2022）提供了一种灵活且容错的方式，相较于采用固定路径的模型。此外，还有许多智能体关注其他领域，例如开放世界环境中的规划（Wang et al., 2023c；Yuan et al., 2023；Park et al., 2023）、工具使用（Ruan et al., 2023；Qin et al., 2023）、强化学习（Shinn et al., 2023；Xu et al., 2023）。本文仅集中于计算机视觉领域。

尽管现有的基于大型语言模型（LLM）的模块化系统表现出色，但主要集中于开发特定策略以解决静态模态的组合任务，忽视了静态模态与动态模态之间的根本差距，而这是实现人工通用智能（AGI）的一项关键要素（Goertzel, 2014）。这些工作在某种程度上可以视为我们系统的一个子集。尽管存在一些例外（Surís et al., 2023；Li et al., 2023b；Wang et al., 2023a；Gao et al., 2023a），但总体上这些研究散乱，缺乏系统性的研究，例如，简单地将视频视为一系列图像（Surís et al., 2023）或基于提取的信息构建聊天机器人（Li et al., 2023b；Wang et al., 2023a）。与此形成鲜明对比的是，我们将视频和任务视为一个整体，从而形成一个紧凑的、与任务相关的记忆。我们系统的推理路径由蒙特卡洛树搜索（MCTS）规划器驱动。除了促进答案搜索外，MCTS规划器还有潜力寻找多个可能的候选答案。这对于开放性问题尤为重要。

具有外部记忆的LLM。有效设计提示模板，即提示工程，对于准确的LLM响应十分重要（Zhou等，2022；Wang等，2023b）。其中一个备受关注的领域是增强记忆的LLM（Wu等，2022c;b；Zhong等，2022；Lewis等，2020；Guu等，2020；Izacard等，2022；Khattab等，2022；Park等，2023；Cheng等，2022；Sun等，2023；Hu等，2023）。一般而言，无需训练的记忆可以分为：i) 文本记忆（Zhu等，2023b；Park等，2023）。在这种记忆中，LLMs无法处理的长上下文（例如书籍）被存储为嵌入，可以通过计算相似性进一步检索。一个典型的例子是在LangChain中展示的文档问答。ii) 符号记忆。它将记忆建模为具有相应符号语言的结构化表示，例如编程语言的代码（Cheng等，2022）、Excel的执行命令²以及数据库的结构化查询语言（SQL）（Sun等，2023；Hu等，2023）。不同于那些直接扩展LLM上下文窗口的技术（Dao等，2022；Dao，2023；Ratner等，2023；Hao等，2022；Chen等，2023b；Mu等，2023；Mohtashami & Jaggi，2023；Peng等，2023b），增强记忆的LLM使用基于检索的方法来绕过上下文长度的限制。这种方法更受青睐，因为（i）它是一个即插即用的模块，无需任何微调或结构修改，以及（ii）并行工作（Shi等，2023；Liu等，2023a）表明LLM在遇到无关或长上下文时可能会分散注意力或迷失方向。通过吸收他们的记忆组织理念，我们构建了一个请求相关的数据库，存储实例感知和实例无关的信息在独立的表中。为了检索相关信息，我们根据不同的目的，明确定义了几个基于提示模板和SQL的子任务工具。从更广的视角来看，我们的多源知识作为提供特定领域可靠指导的补充模块，也可以被视为外部记忆的混合体。

# 5. 结论

针对现实世界场景的动态和不断变化的特性，我们提出了DoraemonGPT，一个基于大型语言模型驱动的智能体，用于解决动态视频任务。与现有的基于大型语言模型的视觉模块化系统相比，DoraemonGPT在以下方面具有优势：i) 通过深入探讨我们生活中的动态模态，设计出概念上优雅的系统；ii) 通过解耦、提取和存储时空属性，实现紧凑的任务相关符号记忆；iii) 通过符号子任务工具实现有效的分解记忆查询；iv) 插拔式知识工具，用于访问领域特定的知识；v) 使用MCTS规划器自动探索大型规划空间，提供多个解决方案并给出信息丰富的最终答案；vi) 答案多样性，通过充分探索解决方案空间提供多个潜在候选项。大量实验验证了DoraemonGPT的多功能性和有效性。

# 致谢

本研究得到了国家科技重大专项（编号：2023ZD0121300）、国家自然科学基金（编号：62372405）以及CCF-腾讯开放基金的支持。

# 影响声明

DoraemonGPT 利用大语言模型 (LLM) 处理现实世界中的动态任务，在基于视频的推理方面表现出色，具有在自动驾驶、监控和交互机器人等领域的潜在应用。尽管具有前景，但仍需解决若干伦理问题。 i) 该系统可能被滥用于视频操纵或生成误导性内容，这需要建立强有力的防护措施以抵御恶意活动。 ii) 训练数据中的偏见可能会延续歧视性行为，突显出公平性和偏见缓解的必要性。 iii) 对外部知识源的依赖强调了遵循数据访问和使用规定的重要性，以避免法律问题。 iv) DoraemonGPT 的方法将基于 LLM 的智能体应用扩展到视觉之外，为各个领域带来变革性的影响。例如，MCTS 规划器增强了在大解空间中的探索策略，而其符号记忆则在交互规划场景中提供了精确的指导，这对多种应用（如具身智能）至关重要（刘等，2024；2023b；董等，2024）。