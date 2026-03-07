# 生成视频压缩：面向0.01%的视频传输压缩率

陈向宇，罗季祥，徐敬宇，易方秋，张驰，李雪龙 * 中国电信人工智能研究院（TeleAI）

视频是否可以在极低的压缩率下压缩到0.01%？为此，我们通过引入生成视频压缩（GVC）这一新框架，在某些情况下实现了0.02%的压缩率，重新定义了视频压缩的极限，利用现代生成视频模型实现极端压缩率，同时保持以感知为中心、面向任务的通信范式，对应于香农-韦弗模型的C级别。此外，如何在压缩率或带宽与计算之间进行权衡？GVC通过将负担从传输转移到推理来回答这个问题：它将视频编码为极为紧凑的表示，并将内容重建委托给接收方，在那里强大的生成先验从最少的传输信息中合成高质量视频。GVC是否实用且可部署？为确保实用部署，我们提出了一种压缩-计算权衡策略，使得在消费级GPU上进行快速推理成为可能。在AI Flow框架内，GVC为带宽和资源受限环境中的视频通信开辟了新的可能性，如紧急救援、远程监控和移动边缘计算。通过实证验证，我们展示了GVC提供了一条通向新的有效、高效、可扩展和实用的视频通信范式的可行路径。日期：2025年12月30日 关键词：生成视频压缩，面向任务的通信，AI Flow *通讯作者：Xuelong Li (xuelong_li@ieee.org) 其他作者按姓氏字母顺序列出。

# 1 引言

在压缩率低至0.01%的情况下重建高质量视频是否可行？我们如何在计算与压缩率之间进行权衡以实现极端压缩？

# 极端压缩在现实场景中是否可行并可部署？

上述问题挑战了视频压缩的传统范式。随着高分辨率视频、虚拟现实、社交媒体和远程会议应用的快速发展，视频数据正在以指数级增长，这对现有的视频存储和传输技术及基础设施提出了前所未有的要求。在带宽受限和延迟敏感的环境中，实现更高效的视频压缩已成为通信与人工智能交叉领域的一个关键研究焦点。传统的通信理论根植于1940年代提出的香农-韦弗模型（Shannon, 1948），将通信概念化为三个层次：层次A关注技术问题，即以信息准确传输为导向的通信；层次B考虑语义问题，即语义通信——传输的符号是否传达了意图的意义；层次C聚焦于有效性问题，即任务导向的通信——接收的信息是否能够引导至期望的行为。几十年来，视频通信技术主要集中于层次A——在受限带宽下最大化信号保真度。这种方法虽然优化了速率失真，但在接收方仅需任务相关内容而不是完美像素重建时可能会造成资源浪费。

为了弥合比特级保真度与任务级效用之间的差距，AI Flow框架（Shao和Li, 2024）于2024年底由TeleAI首次提出，该框架构想利用通信网络分配智能，以实现无处不在的AI驱动服务。此外，信息容量（Yuan et al., 2025b）被提出用于评估生成模型在数据压缩中的有效性。这一进展为基于生成模型的数据压缩奠定了理论和方法基础。在2025年初，我们通过引入面向任务的通信，扩展了这一工作方向，旨在通过设备边缘协同推理实现多模态理解（Yuan et al., 2025a）。到2025年中期，TeleAI在世界人工智能大会（WAIC）上首次介绍了生成视频压缩（GVC）的概念。与传统编解码器强调像素级重建保真度不同，GVC采纳了面向任务的通信视角。它优先考虑所传输的信息是否满足感知期望或有效支持下游任务，从而将C级放在设计的核心。在WAIC大会上，TeleAI发布了一个原型，用于海事通信，使得在有限带宽的卫星连接上实现超低比特率视频传输。相关的理论基础已在技术报告中详细阐述（An et al., 2025）。

GVC 的核心原则是以计算量换取压缩率。最近在生成模型方面的进展，特别是在生成视频模型（OpenAI，2024；Wan 等，2025）中，为视频压缩带来了前所未有的机遇。通过利用强大的生成先验，GVC 旨在克服传统标准（如 HEVC）（Sullivan 等，2012）中长期存在的比特率和感知质量之间的权衡。此外，我们还阐明了 GVC 的动机（Fan 等，2025），该动机描绘了计算、带宽和内存之间的关系。一个有用的比喻说明了这一转变：传统压缩类似于拍摄一幅画并发送图像；而 GVC 则描述了画作的构图和风格，然后依赖接收端的“AI 画家”进行重现。得益于其表达能力强大的生成能力，现代模型能够从最小的潜在表示——甚至纯噪声——中合成高质量视频，且受学习到的先验指导。因此，编码器的角色从保护每个像素转变为仅传输与任务相关的信息。如图 2 所示，GVC 可以以低至 0.005 bpp 的比特率实现视觉上引人注目的重建（相当于 $0.0 2 \%$ 的压缩率）。这表明 GVC 正在朝着极限压缩前沿 $0.0 1 \%$ 迈进。此外，即使考虑常见测试序列的平均性能，GVC 也显著减少了所需的传输带宽，同时保持高感知质量，如第三节所示。这使得它特别适合对通信效率要求很高的场景，如海事通信、紧急救援、窄带移动网络、远程视频监控、车载或可穿戴设备。然而，极限压缩带来了新的挑战。高质量重建依赖于计算密集型生成过程，对硬件和推理延迟提出了严格要求。为了解决这个问题，我们提出了协调计算与实际应用压缩率的概念，以在压缩、计算和质量之间实现更可行的平衡。如表 3 所示，我们的系统可以在消费者级 GPU 上运行，推理延迟约为 2 秒——与大型语言模型的响应时间相当——展现了良好的实际可用性。本报告提出了一个针对超低比特率视频传输的生成视频压缩框架。通过利用强大的生成视频先验，该框架在极大减少带宽消耗的同时，实现了高感知质量，并支持下游任务。我们描述了 GVC 的系统架构、设计策略和实验验证，旨在为下一代以感知驱动的视频通信技术奠定基础。

# 2 方法论

# 2.1 框架概述

生成视频压缩（GVC）框架通过将原始视频帧转换为紧凑的潜在表示并通过生成建模进行重构，实现高效的视频压缩。如图1所示，该框架由两个主要组件组成：神经编码器和生成视频解码器。

![](images/1.jpg)  
Figure 1 Overview of Our GVC Framework Grounded in the Shannon-Weaver model (Shannon, 1948). Top-left: Level essheenl probl,tizalelyundmi banwi yizis bee pu n uu vidsTo-h:Level usn thesman probl,  t ransitt he precise semantic symbols.Bottom: Level C, central to the proposed Generative Video Compression (GVC) framework, emphasiziaskenteiene I sure hathpretoe nablhevment skl such as high-quality perception reconstruction or support for downstream tasks like segmentation.

系统首先接收输入的视频序列，该序列可能包含各种类型的内容，如监控视频。这段视频随后被送入一个预训练的神经网络，该网络旨在将视频压缩为一组紧凑的表示，称为压缩词元。这些压缩词元包括离散和连续的表示，涵盖压缩的关键帧、视频片段的高级描述符和低级连续特征。编码器能够显著减少视频数据的维度，同时保留其整体语义和运动动态。为了进一步提高压缩效率，这些词元还通过残差编码等技术编码为比特流，从而减少存储和传输需求。在解码器端，一个基于扩散的预训练生成视频模型从压缩词元中重建视频。其中一些词元作为去噪过程的直接输入，而其他词元则作为条件。这一重建过程本质上是一个条件视频生成任务，其中模型合成与原始输入在视觉上相符的视频帧。最终输出是一个重建的视频，视觉质量与原始视频十分相似，感知损失最小，从而实现压缩率和视觉质量之间的平衡。

# 2.2 计算交易与压缩率的权衡

GVC 的一个核心理念是以计算换取压缩率。GVC 不再传输详细的视觉数据，而是利用解码器中强大的生成模型重建视频内容，从而显著降低传输所需的比特率。传统编解码器旨在在比特率限制下使用手工设计的信号处理技术来保持信号保真度。而 GVC 则将重建的负担转移到解码器上，利用计算和嵌入在生成模型中的先验知识，从最小的输入中合成真实的帧。这个转变可以比喻性地说明：传统压缩就像拍摄一幅画并发送照片；而 GVC 则像是在描述这幅画的构图和风格，然后让一个“人工智能画家”来重现它。现代生成模型能够仅根据潜在表示或甚至随机噪声生成高质量的视频，得益于强有力的学习先验。因此，编码器的角色变为选择和传输最与任务相关的信息，而不是保留每一个像素。关键在于，传输内容依赖于重建视频的目的。如果目标是人类感知，编码器将传输有助于生成感知上相似内容的特征。如果目标是机器理解（例如，分割、识别），那么编码器则专注于传输语义上有意义的表示。这意味着从以保真度为导向的压缩转向以任务为导向或以有效性为导向的通信，使 GVC 与超越简单重建准确性的更高层次目标对齐。

# 2.3 在实用性与交易压缩率之间的权衡

在压缩率与计算量之间进行权衡可以通过解码器端生成高度紧凑的视频表示，但这种方法在实际应用中面临局限性。具体而言，解码器的计算能力受硬件资源、功耗和延迟要求的限制，对可以用于压缩的计算量施加了上限。在许多应用中，例如实时视频会议或边缘设备流媒体，解码器的延迟和效率成为关键瓶颈。因此，重建质量、压缩率和计算之间的平衡必须重新评估，以实用性作为核心考虑。为此，我们的框架结合了故意将压缩率与实用性进行权衡的策略，确保解码在可接受的重建质量下依然可行。其中一种策略是增加压缩表达的稀疏性，从而减少解码器中的模型负担。这为使用更小、更快的模型提供了可能。此外，我们还应用模型压缩技术来减少关键组件（例如，3D变分自编码器）的大小和复杂性，并采用蒸馏和采样加速方法来降低基于扩散的解码器的推理时间。在这些情况下，我们常常通过传输更高维或更具信息性的特征来补偿因模型简化而导致的质量损失，在压缩率、计算和质量之间寻求新的平衡。最终，这种权衡反映了对生成视频压缩（GVC）范式的实用扩展：尽管生成模型能够实现极端压缩，但现实世界的可用性则要求采取适应可用计算资源的策略。通过灵活调整传输信息的量和生成推理的复杂性，我们的框架确保GVC不仅在比特率上高效，而且在实际部署条件下也具有可行性和响应能力。

# 3 结果

为了验证我们GVC框架的有效性，我们首先基于14B视频生成模型评估标准基准MCL-JCV（Wang等人，2016）上的视频压缩性能。我们采用主流的感知度量进行评估：学习的感知图像补丁相似度（LPIPS），因为它被认为是衡量人类感知质量的标准。如表1所示，在平均码率为0.008 bpp的情况下，我们的方法保持了竞争性高的感知质量。相对而言，传统视频编码方案在该码率下存在显著的性能差距。对于某些具有挑战性的序列，传统方法需要大约6倍于我们方法的码率才能达到等效的感知重建质量，如图2所示。

![](images/2.jpg)  
FigureBandwidth comparison orachievin comparable reconstruction qualiy.Traditional methods require more thana  ulp .

表 1 MCL-JCV 数据集的定量比较。值越低，效果越好。

<table><tr><td>Method</td><td>LPIPS ↓</td></tr><tr><td>HEVC Sullivan et al. (2012)</td><td>0.278</td></tr><tr><td>Ours</td><td>0.180</td></tr></table>

为了进一步验证其实际效用，我们将模型的重建结果应用于下游任务：在DAVIS2017数据集上的视频目标分割（VOS）（Pont-Tuset等，2017）。我们使用Jaccard指数$\mathcal { I }$、轮廓精度$\mathcal { F }$、它们的平均值$( \mathcal { I } \& \mathcal { F } )$以及轮廓召回率($\mathcal { F }$ -Recall)来评估性能。如表2所示，我们的方法达到了高度竞争的性能。这表明，即使在低比特率下，我们的方法也能够保持正确的语义传递。表格展示了不同编码方法的下游性能。Upperbound是通过使用原始视频评估任务模型获得的。

<table><tr><td rowspan="2">Method</td><td colspan="4">VOS: XMEM on DAVIS2017</td></tr><tr><td>J&amp;F (%)</td><td>J (%)</td><td>F (%)</td><td>F-Recall (%)</td></tr><tr><td>HEVC@bpp=0.01</td><td>57.68</td><td>56.84</td><td>58.51</td><td>67.44</td></tr><tr><td>Ours@bpp=0.01</td><td>75.22</td><td>71.17</td><td>79.28</td><td>91.87</td></tr><tr><td>Upper-bound</td><td>87.70</td><td>84.06</td><td>91.33</td><td>97.02</td></tr></table>

我们致力于通过模型小型化、知识蒸馏和量化等技术提升计算效率。这些优化使我们的方案在各种硬件平台上都具备可部署性。如表3所示，该表报告了我们的微型化模型在不同平台上生成29帧 GOP（即一次生成29帧）的延迟，即使在消费级硬件上，我们的系统也实现了实用的推理速度。表3 模型计算效率和硬件性能（GOP=29）

<table><tr><td rowspan=2 colspan=1>Resolution</td><td rowspan=2 colspan=1>Module</td><td rowspan=1 colspan=3>Latency (s)</td></tr><tr><td rowspan=1 colspan=1>4090</td><td rowspan=1 colspan=1>A100</td><td rowspan=1 colspan=1>H200</td></tr><tr><td rowspan=2 colspan=1>480p</td><td rowspan=1 colspan=1>Encoder</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.64</td><td rowspan=1 colspan=1>0.2</td></tr><tr><td rowspan=1 colspan=1>Decoder</td><td rowspan=1 colspan=1>1.35</td><td rowspan=1 colspan=1>1.4</td><td rowspan=1 colspan=1>1.13</td></tr><tr><td rowspan=2 colspan=1>720p</td><td rowspan=1 colspan=1>Encoder</td><td rowspan=1 colspan=1>1.15</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.3</td></tr><tr><td rowspan=1 colspan=1>Decoder</td><td rowspan=1 colspan=1>6.4</td><td rowspan=1 colspan=1>5.5</td><td rowspan=1 colspan=1>2.3</td></tr><tr><td rowspan=2 colspan=1>1080p</td><td rowspan=1 colspan=1>Encoder</td><td rowspan=1 colspan=1>1.59</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.5</td></tr><tr><td rowspan=1 colspan=1>Decoder</td><td rowspan=1 colspan=1>21.5</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>6.1</td></tr></table>

尽管与全尺寸模型相比，微型化模型在视觉质量和带宽效率上有所损失，但其感知质量依然具有竞争力，图 Fi 所示的视频序列实现了 LPIPS 值 0.273。这些结果共同表明，我们的微型化模型在实际部署场景中实现了计算效率与视觉质量的有效平衡。

# 4 结论

本报告在AI Flow框架下（An et al., 2025）并基于信息容量理论（Yuan et al., 2025b），通过生成视频压缩（GVC）的视角重新构想视频压缩的基础——这一范式转换优先考虑感知相关性和任务有效性，而非像素级的逼真度。通过询问在极端压缩下是否可以重建高质量视频，我们不仅挑战了传统编解码器的极限，还展示了在边缘设备日益强大的时代，计算与压缩之间的可交易性。我们的研究结果表明，借助现代生成视频模型，能够以极低比特率实现引人注目的重建，同时保持视觉真实感和下游任务的实用性。此外，我们引入了压缩率与实用性之间的交易概念，强调平衡压缩效率、推理延迟和硬件约束的系统设计。我们的实现表明，GVC能够在消费者级GPU上以可接受的延迟运行，使其在远程监控、低带宽移动通信和边缘AI设备等领域的实际部署成为可能。总之，GVC不仅是一种压缩技术——它体现了一种面向任务的通信范式，旨在服务于生成智能时代。通过仅传输感知和决策所需的内容，它为一类更高效、更适应和更智能的通信系统开辟了新的可能。我们希望这项工作能激发在生成建模、通信理论和实际部署交叉领域的进一步研究，推动极端视频压缩的可能性边界。

![](images/3.jpg)  
FiuVisual qualiy omparison theminaturize model, demonstratin competitive perceptual qaliydespi model compression.

# References

Houn, Wenan HuSia HuangSiqi Hua, Ranu  uai LiaJiawho Ylg Sg ZiaW, Cheng Yuan, et al. AI Flow: Perspectives, Scenarios, and Approaches (2025). arXiv preprint arXiv:2506.12479, 2025.   
Yuankai Fan, Qizhen Weng, and Xuelong Li. Computation-Bandwidth-Memory Trade-offA Unified Paradigm for AI Infrastructure. arXiv preprint arXiv:2601.11577, 2025.   
OpenAI. Video Generation Models as World Simulators, 2024. URL https://openai.com/index/ video-generation-models-as-world-simulators/.   
Jor Pont-Tuset,FederiPerazzi, SrgiCael, PablArbelá, Alex SorkeHor, and LucVan GoolThe 2017 DAVIS Challenge on Video Object Segmentation. arXiv preprint arXiv:1704.00675, 2017.   
Claude E Shannon. A Mathematical Theory of Communication. The Bell System Technical Journal, 27(3):379423, 1948.   
Jiawei Shao and Xuelong Li. AI Flow at the Network Edge. IEEE Network, 2024.   
Gary J Sullivan, Jens-Rainer Ohm, Woo-Jin Han, and Thomas Wiegand. Overview of the High Eficiency Video Coding (HEVC) Standard. IEEE Transactions on Circuits and Systems for Video Technology, 22(12):16491668, 2012.   
Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, et al. WAN: Open and Advanced Large-scale Video Generative Models. arXiv preprint arXiv:2503.20314, 2025.   
Haiqiang Wang, Weihao Gan, Sudeng Hu, Joe Yuchieh Lin, Lina Jin, Longguang Song, Ping Wang, Ioannis Katsavounidis, Anne Aaron, and C.-C. Jay Kuo. MCL-JCV: A JND-based H.264/AVC Video Quality Assessment Dataset. In 2016 IEEE International Conference on Image Processing, pages 15091513, 2016.   
Ceg Yuan, Zheg Liu, Jiashu Lv, Jiaw Shao,Yufei JiangJun Zhang and Xuelon LiTask-Oriet ue Compression for Multimodal Understanding via Device-Edge Co-Inference. IEEE Transactions on Mobile Computing, pages 114, 2025a.   
Cheng Yuan, Jiawei Shao, Chi Zhang, and Xuelong Li. Information Capacity:Evaluating the Effciency of Large Language Models via Text Compression. arXiv preprint arXiv:2511.08066, 2025b.