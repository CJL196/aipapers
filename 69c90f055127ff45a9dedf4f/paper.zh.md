# 遥感 ChatGPT：使用 ChatGPT 和视觉模型解决遥感任务

霍南 郭^I，欣 苏^{*2}，陈 \boldsymbol{W_u}^I，博杜^{3}，梁培 张，德仁 李^{I} 1 中国武汉大学测绘遥感信息工程国家重点实验室，武汉，中国 2 中国武汉大学遥感与信息工程学院，武汉，中国 3 中国武汉大学计算机学院，武汉，中国

# 摘要

近年来，蓬勃发展的大语言模型（LLM），尤其是ChatGPT，在语言理解、推理和交互方面表现出色，吸引了来自多个领域和学科的用户和研究者。尽管LLM在处理自然语言和自然图像方面展现出了人类般的任务完成能力，但它们在遥感解译任务中的潜力尚未得到充分挖掘。此外，遥感任务规划缺乏自动化，阻碍了遥感解译技术的可及性，尤其是对来自多个研究领域的非遥感专家。因此，我们提出了遥感ChatGPT，一个基于LLM的代理，利用ChatGPT将各种基于AI的遥感模型连接起来，以解决复杂的解译任务。更具体地说，给定用户请求和遥感图像，我们利用ChatGPT理解用户请求，根据任务功能进行任务规划，迭代执行每个子任务，并根据每个子任务的输出生成最终响应。考虑到LLM是通过自然语言进行训练的，无法直接感知遥感图像中所包含的视觉概念，我们设计了能够注入视觉信息到ChatGPT的视觉提示。通过遥感ChatGPT，用户只需发送带相关请求的遥感图像，即可获得解译结果及遥感ChatGPT的语言反馈。实验和示例表明，遥感ChatGPT能够处理广泛的遥感任务，并可以通过更复杂的模型（如遥感基础模型）扩展到更多的任务。遥感ChatGPT的代码和演示可在 https://github.com/HaonanGuo/Remote-Sensing-ChatGPT 获取。 索引词——遥感图像，大语言模型代理，图像解译

# 引言

地球观测技术为大规模监测地表提供了理想的数据来源，从而可以支持可持续发展目标（SDG）的实现。在过去的几十年中，已经投入大量精力开发基于深度学习的算法，以应对广泛的遥感解释任务，如场景分类、目标检测、语义分割、图像描述等。

尽管已经开发了多种任务和模型，但如何组织这些任务以解决实际用户请求仍然是一项挑战。例如，解决“统计跑道上的飞机数量”请求需要顺序执行跑道分割、飞机检测和物体计数。然而，这一任务规划过程目前严重依赖于人工干预，要求遥感专家理解用户请求并在交付符合用户期望的产品之前进行任务规划。任务规划缺乏自动化限制了遥感解读技术的可及性，尤其是对来自多个研究领域的非遥感专家。最近，大语言模型（LLM），尤其是ChatGPT，在语言理解、推理和互动方面显示出令人印象深刻的表现。通过以自回归方式自动学习大量的网页文本数据，LLM在甚至未见过的任务上也被证明有效。这一新兴能力使得LLM能够通过精心设计的提示系统执行任务规划。在自然语言和图像理解领域，一些基于LLM的智能体方法已证明使用LLM作为智能体来处理图像或语言处理任务是可行的。然而，LLM在遥感领域的潜力尚未得到充分探索。虽然一些初步研究探讨了ChatGPT在遥感任务中的适用性，但它们仅仅是将为自然图像设计的方法应用于遥感图像，并尚未考虑将遥感模型与ChatGPT集成。此外，对不同LLM任务调用性能的定量评估尚未进行。在本文中，我们提出了遥感ChatGPT，一个类似于ChatGPT的系统，能够理解用户请求、规划遥感解读任务，并生成最终产品及用户响应。我们基于ChatGPT和多个支持各种解读任务的AI遥感模型构建了遥感ChatGPT。我们期望遥感ChatGPT能够推动遥感解读技术的可及性，让非专家，特别是在城市化和森林砍伐等多个应用领域工作的人员受益，他们并不是遥感专家。此外，这也是自动化遥感任务规划的一次有意义的尝试，这是实现完全自动化遥感图像解读的关键一步。我们进行定量和定性评估，以探索遥感ChatGPT在不同LLM骨干下的性能。我们还讨论了设计用于遥感的类似ChatGPT系统的局限性和未来方向。

![](images/1.jpg)  
Fig.1. Workflow of the proposed Remote Sensing ChatGPT.

# 2. 遥感 CAHTGPT

遥感 ChatGPT 能够通过 ChatGPT 和遥感解译模型解决遥感任务。使用遥感 ChatGPT，用户只需发送一张遥感图像及相应的语言请求，即可获得解译结果和语言反馈。遥感 ChatGPT 的工作流程如图 1 所示，包括提示模板生成、任务规划、任务执行和响应生成，下面的各小节将对此进行介绍。

# 2.1 提示模板生成

用户语言输入的第一步是生成一个提示模板，作为ChatGPT理解指令、执行推理和正确输出结果的系统原则。例如，在系统原则部分，要求ChatGPT使用工具来完成以下任务，而不是仅仅根据描述进行想象。在任务调用和输出格式部分，ChatGPT被要求严格遵守文件名，并绝不会编造不存在的文件。我们还为ChatGPT提供了一个模板，以规范工具名称、输入文件和输出工具的观察。此外，考虑到ChatGPT是一个语言模型，无法直接访问图像，我们引入了BLIP模型来为遥感图像提供描述，从而为ChatGPT提供视觉线索，以帮助其更好地理解图像。

# 2.2 任务规划

遥感 ChatGPT 目前支持调用不同的遥感任务，如场景分类、土地利用分类、目标检测、图像描述、边缘检测、多边形化和目标计数。这些任务及其相应模型的详细信息列在表 I 中。我们为每个任务选择了广泛使用的网络架构，并在公开可用的基准上训练了模型。应该注意的是，可以应用更多任务和更复杂的模型，以进一步提升遥感 ChatGPT 的性能。表 I 遥感 ChatGPT 支持的任务

<table><tr><td>Tools</td><td>Method</td><td>Dataset</td></tr><tr><td>Scene Classification</td><td>ResNet</td><td>AID</td></tr><tr><td>Land use Classification</td><td>HRNet</td><td>LoveDA</td></tr><tr><td>Object Detection</td><td>YOLOv5</td><td>DOTA</td></tr><tr><td>Image Captioning</td><td>BLIP</td><td>BLIP Dataset</td></tr><tr><td>Edge Detection</td><td>Canny</td><td></td></tr><tr><td>Polygonization</td><td>Douglas-Peuker</td><td></td></tr><tr><td>Object Counting</td><td></td><td></td></tr></table>

在这一部分，给定定义的任务库，生成任务函数的描述、支持的类别、输入和输出数据格式，以及任务依赖性以补充提示模板。此外，还提供了每个任务的一些示例，以进行上下文学习，从而进一步提高模型对用户输入的理解。完整的提示然后被输入到ChatGPT中以执行任务规划。

# 2.3 任务执行与响应生成

ChatGPT 的输出决定了是否以及使用哪些工具。确定的工具被应用于预处理遥感图像，并生成相应的输出。随后，该输出被作为新观测输入到 ChatGPT，以确定是否需要使用新工具进一步满足用户的请求。如果不需要使用更多的工具，所有已执行任务的输出将被发送到 ChatGPT，后者将生成最终响应。

# 3. 实验

考虑到遥感 ChatGPT 可以通过基础模型等先进方法轻松扩展至更多任务，我们的实验重点在于 ChatGPT 是否正确规划了解释任务，而不是解释的准确性。为此，我们从多个用户收集了 138 个用户查询。然后，根据这些查询对相应的任务进行标注。考虑到 ChatGPT 可以调用更多任务来辅助推理过程，我们仅对解决用户查询所需的基本任务进行标注，并计算查询的正确性，以表示 ChatGPT 是否正确规划了基本任务。例如，物体检测是查询“在提供的航空图像中定位棒球场”的基本任务。如表 II 所示，我们使用 4 种不同的 ChatGPT 主干网络对遥感 ChatGPT 进行了测试。我们发现使用 gpt-3.5-turbo 的遥感 ChatGPT 在遥感任务规划中表现最佳，其次是 gpt-4-1106-preview 和 gpt-4。整体准确率为 $94.9 \%$，证明了遥感 ChatGPT 理解用户查询和规划遥感任务的能力。尽管 gpt-3.5-turbo-1106 支持更多词元，但与 gpt-3.5-turbo 相比，其理解复杂指令的能力相对有限，从而导致模型性能下降。在图 2 中，我们进一步可视化了一些遥感 ChatGPT 的成功案例和失败案例。在成功案例中，我们可以看到遥感 ChatGPT 能有效规划和执行不仅需要单一任务的简单查询，还能处理需要多个任务迭代执行的复杂查询。然而，如图 2 所示，也存在一些失败案例。其中一个主要失败案例在于现有遥感模型不支持的类别。例如，遥感 ChatGPT 请求的土地利用分类模型不被该模型支持，因为训练数据集中不包括耕地类别。另一个失败案例显示，当现有工具或信息无法完全解决用户查询时，遥感 ChatGPT 倾向于想象答案而不是请求更多信息。表 II 显示了遥感 ChatGPT 在任务规划中的正确性，涉及从输入图像中分割耕地。

<table><tr><td>Tools</td><td>gpt-4</td><td>gpt-4-1106- preview</td><td>gpt-3.5- turbo</td><td>gpt-3.5- turbo-1106</td></tr><tr><td>Overall Correctness</td><td>63%</td><td>84.1%</td><td>94.9%</td><td>29%</td></tr><tr><td>Scene Classification</td><td>76.9%</td><td>84.6%</td><td>84.6%</td><td>7.7%</td></tr><tr><td>Land use Classification</td><td>69.1%</td><td>90.9%</td><td>100%</td><td>30.9%</td></tr><tr><td>Object Detection</td><td>79.2%</td><td>83.3%</td><td>95.8%</td><td>33.3%</td></tr><tr><td>Image Captioning</td><td>86.7%</td><td>60%</td><td>93.3%</td><td>6.7%</td></tr><tr><td>Edge Detection</td><td>70%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>Polygonization</td><td>0%</td><td>100%</td><td>100%</td><td>28.6%</td></tr><tr><td>Object Counting</td><td>0%</td><td>64.3%</td><td>78.6%</td><td>7.1%</td></tr></table>

![](images/2.jpg)  
Fig.2 Example of some successful and failure cases of Remote Sensing ChatGPT.

# 4. 未来方向

遥感ChatGPT是一个基于大语言模型的智能体，利用ChatGPT将各种基于AI的遥感模型连接起来，以解决复杂的解译任务。通过将遥感基础模型与基于智能体的模型结合，我们相信在不久的将来将实现完全自动化的遥感解译，从而服务于环境监测、灾害响应等多个领域的用户

# 5. 结论

在本文中，我们提出了遥感 ChatGPT，这是一种基于大语言模型的智能体，利用 ChatGPT 连接各种基于 AI 的遥感模型，并解决复杂的解译任务。遥感 ChatGPT 能够理解用户的请求，规划遥感解释任务，并生成最终产品及响应用户。定量和定性评估表明，遥感 ChatGPT 可以进行精确的任务规划和执行。我们希望遥感 ChatGPT 是实现完全自动化遥感图像解译的有意义尝试，并能推动遥感解译技术对多领域应用研究者的可及性。

# 致谢

本研究部分得到了中国国家自然科学基金的支持，资助编码为 42230108 和 62371348。

# REFERENCES

[1] S. D. Prince, 'Challenges for remote sensing of the Sustainable Development Goal SDG 15.3.1 productivity indicator',

Remote Sensing of Environment, vol. 234, p. 111428, Dec. 2019, doi: 10.1016/j.rse.2019.111428.   
[2] X. X. Zhu et al., 'Deep Learning in Remote Sensing: A Comprehensive Review and List of Resources', IEEE Geoscience and Remote Sensing Magazine, vol. 5, no. 4, pp. 836, Dec. 2017, doi: 10.1109/MGRS.2017.2762307.   
[3] C. Broni-Bediako, J. Xia, and N. Yokoya, 'Real-Time Semantic Segmentation: A brief survey and comparative study in remote sensing', IEEE Geoscience and Remote Sensing Magazine, vol. 11, no. 4, pp. 94124, Dec. 2023, doi: 10.1109/MGRS.2023.3321258.   
[4] X. Zhang et al., 'Remote Sensing Object Detection Meets Deep Learning: A metareview of challenges and advances', IEEE Geoscience and Remote Sensing Magazine, vol. 11, no. 4, pp. 844, Dec. 2023, doi: 10.1109/MGRS.2023.3312347.   
[5] '[2303.17580] HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face'. Accessed: Jan. 02, 2024. [Online]. Available: https://arxiv.org/abs/2303.17580   
[6] '[2302.04761] Toolformer: Language Models Can Teach Themselves to Use Tools'. Accessed: Jan. 02, 2024. [Online]. Available: https://arxiv.org/abs/2302.04761   
[7] '[2005.14165] Language Models are Few-Shot Learners'. Accessed: Jan. 02, 2024. [Online]. Available: https://arxiv.org/abs/2005.14165   
[8] '[2303.18223] A Survey of Large Language Models'. Accessed: Jan. 02, 2024. [Online].  Available: https://arxiv.org/abs/2303.18223   
[9] [2303.04671] Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models'. Accessed: Jan. 02, 2024. [Online]. Available: https://arxiv.org/abs/2303.04671   
[10] L. P. Osco, E. L. de Lemos, W. N. Gonçalves, A. P. M. Ramos, and J. Marcato Junior, 'The Potential of Visual ChatGPT for Remote Sensing', Remote Sensing, vol. 15, no. 13, Art. no. 13, Jan. 2023, doi: 10.3390/rs15133232.   
[11] '[2201.12086] BLIP: Bootstrapping Language-Image Pretraining for Unified Vision-Language Understanding and Generation'. Accessed: Jan. 02, 2024. [Online]. Available: https://arxiv.org/abs/2201.12086