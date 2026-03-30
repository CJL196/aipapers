# REMsA：通过约束感知智能体进行遥感基础模型选择

Binger Chen, Tacettin Emre Bök, Behnood Rasti, Volker Markl 和 Begüm Demir 柏林工业大学及 BIFOLD，德国柏林 chen@tu-berlin.de

摘要。基础模型（FMs）越来越多地融入遥感（RS）管道，用于环境监测、灾害评估和土地利用制图等应用。这些模型包括在单一数据模态下训练的单模态视觉编码器以及在多个传感器模态下训练的多模态架构，如合成孔径雷达（SAR）、多光谱和高光谱影像，或在视觉-语言环境中通过图像-文本对共同训练。FMs 根据其预训练目标和架构设计，适应不同的感知任务，如语义分割、图像分类、变化检测和视觉问答。然而，由于文档分散、格式异构和复杂的部署限制，为特定任务选择最合适的遥感基础模型（RSFM）仍然具有挑战性。为了解决这一问题，我们首先介绍了RSFM数据库（RS-FMD），这是第一个结构化和有模式指导的资源，涵盖了160多种多种数据模态下训练的RSFM，跨越不同的空间、光谱和时间分辨率，同时考虑不同的学习范式。基于RS-FMD，我们进一步提出了REMsA（遥感模型选择智能体），这是一个具约束意识的智能体，能够从自然语言查询中实现自动化的RSFM选择。ReMsA结合了结构化FM元数据检索和任务驱动的决策工作流程。具体而言，它解析用户输入，澄清缺失的约束，通过上下文学习对模型进行排序，并提供透明的理由。我们的系统支持多种RS任务和数据模态，实现个性化、可重复和高效的FM选择。为了评估ReMsA，我们构建了一个100个专家验证的RS查询场景基准。每个查询在4个系统和3个LLM主干上进行评估，前3个选定模型由领域专家手动评估。这导致了在我们新颖的以专家为中心的评估协议下产生3000个专家评分的任务-系统-模型配置。REMsA在多个基准上表现优于包括简单基于智能体的方法、密集检索和非结构化检索增强生成的方法，显示其在实际决策应用中的实用性。ReMsA完全基于公开可用的开源RSFM元数据操作，不涉及私有或敏感数据。我们的代码和数据可在此处找到：https://github.com/be-chen/REMSA。

# 1 引言

随着遥感（RS）任务及其搭载传感器（例如，Sentinel-2 [7]、Sentinel-1 [34]、EnMAP [10]）的日益普及，遥感在农业、灾害响应、城市发展和生物多样性监测等多种应用中扮演着越来越重要的角色。这些应用日益依赖于基础模型（FMs），这些模型能够跨越不同的遥感数据模态进行泛化，具有不同的空间、光谱和时间分辨率、地理空间范围及应用，同时在下游遥感分析任务中即使有限标注数据也能实现有效的迁移和应用。近年来，众多基础模型在遥感领域涌现，提供了强大的复杂遥感数据建模能力。这些模型包括在单一或多种遥感数据模态上训练的仅视觉编码器（例如，MMEarth [22]、OmniSat [2]、MA3E [17]）以及使用配对遥感数据模态和文本训练的视觉语言模型（VLMs）（例如，Geo-Text [4]、LHRS-Bot [21]）。这些模型已在涵盖各种传感器模态的大规模异构遥感数据集上进行预训练，包括RGB、多光谱、高光谱、合成孔径雷达（SAR）和光学遥感（LiDAR），并且涉及多个空间和时间分辨率。每个基础模型在不同的下游遥感任务中展现出其独特的优势，例如分类、目标检测、变化检测、图像描述和视觉问答（VQA）。例如，在实际的遥感工作流程中，变化检测通常依赖于多时相的SAR或光学数据输入，而细粒度的土地覆盖制图则往往受益于高分辨率的光学影像。这种多样性为多模态遥感应用带来了新的可能性，但同时也提出了在特定任务中选择最适合基础模型的问题，考虑到数据模态和操作限制。

![](images/1.jpg)  
Fig. 1: Architecture of REMSA

尽管取得了这些进展，为特定的遥感（RS）任务选择合适的基础模型（FM）仍然是一项挑战。遥感从业人员必须平衡各种约束，例如可用的数据模态和数量、地理覆盖范围、计算资源以及任务特定的评估优先级。这些约束被证明会显著影响遥感基础模型（RSFM）的泛化能力和稳健性。如今，有数百个遥感基础模型公开可用，但没有统一的结构化模式来组织它们的属性（如模型架构、训练数据或报告的性能），选择过程往往是手动的，这既耗时又容易出错，且难以重现。现有的方法依赖于在分散的存储库和出版物中搜索、手动解析论文和模型卡，以及进行详尽的实验，所有这些都没有保证可重现性或透明性。即使是公共的遥感基准测试，主要也只是比较模型在固定应用上的准确性，几乎无法提供支持用户特定约束或部署权衡的匹配方案。因此，构建一个统一的、机器可读的遥感基础模型数据库（DB）成为任何系统化选择和自动化的必要基础。最近，大语言模型（LLM）智能体的进展显示出将语言模型与外部工具和结构化工作流程结合以支持复杂任务的潜力。然而，大多数现有智能体的目标是通用问答或对话辅助。据我们所知，尚无相关研究开发出用于操作性约束重的遥感场景中的基础模型选择的领域特定智能体。特别是，遥感任务涉及传感器、空间、光谱和时间分辨率以及数据可用性之间的复杂权衡。现有的大语言模型缺乏应对这些约束所需的领域知识和有结构的模型文档访问权限。因此，这样的智能体必须提供更稳健且可解释的解决方案。

在本研究中，我们首先介绍了RSFM数据库（RS-FMD），这是第一个超过160个RSFM的基于模式指导的资源，涵盖了各种数据模态、预训练策略和已报告的基准结果。在RS-FMD的基础上，我们提出了REMsA，这是第一个基于大语言模型的自动化FM选择智能体。如图1所示，REMsA是一个模块化的智能体，通过结构化查询解析和任务感知的工具调用来支持FM选择。它从自由文本输入中提取用户意图并将其转化为约束。根据任务状态，该智能体协调工具从RS-FMD中检索信息、排名FM、进行互动澄清和提供合理的解释。轻量级的记忆机制进一步增强了准确性和个性化。为了评估REMsA，我们构建了一套新的基准，包含现实用户查询，并建立了专家驱动的评估协议。我们还实现了一系列精心构建的基线，确保与REMsA之间的公平和有意义的比较。REMsA旨在支持广泛的用户，包括需要识别适合其任务的RSFM的RS科学家、机器学习研究人员和行业从业者。由于REMsA接受自由文本查询，并结合结构化解析和多轮澄清，它也能帮助那些可能不熟悉RS模态或FM架构的非专家。这使得REMsA适合从业者的探索性用途和研究环境中的严格FM选择。尽管REMsA采用了模块化智能体设计，但我们的贡献在于方法论。我们将RSFM选择视为一个研究问题，探讨如何在实际约束下比较、选择和部署FM。总的来说，我们做出了以下贡献：我们引入了RS-FMD，这是第一个结构化和模式指导的超过150个RSFM的数据库。我们将其作为社区资源发布，并持续维护和更新。我们提出了REMsA，一个模块化的LLM智能体，结合了结构化元数据支撑、密集检索、上下文排名、澄清、解释、记忆增强组件和任务感知的协调机制，以支持真实RS环境中的复杂FM选择。我们构建了一个新的基准数据集和FM选择评估协议，包含100个现实查询场景和3,000个专家评分的评估，覆盖系统、LLM主干和所选模型。

# 2 相关工作

基础模型与模型选择。由于基础模型的快速出现，关于其能力和基准的研究相继展开 [3, 6, 36]。在遥感领域，最近的调查和基准 [11, 27, 37] 系统地对基础模型进行了分类，并评估了它们在土地覆盖分类、野火伤痕分割、城市变化检测、视觉问答等应用中的表现。然而，这些工作主要集中在描述性分析或标准化评估上，对自动化基础模型选择的支持有限。大规模评估如GEO-Bench-2 [30] 进一步强调遥感基础模型的性能在能力维度上存在显著差异，但仍未解决自动化基础模型选择的问题。其他研究还表明，预训练数据的覆盖范围（地理和传感器多样性）会强烈影响遥感基础模型的泛化能力 [25, 26]。虽然目前的基准文档记录了这些属性，但并未利用这些信息来引导模型选择，从而进一步促进自动化基础模型选择的必要性。此外，还出现了一种新的能力编码方法，可以估计模型在未见下游任务上的表现，从而减少全面微调的需求 [1]。尽管这为比较评估提供了有价值的工具，但仍然是一个未能解决端到端自动化基础模型选择工作流程的基准工具。此外，以前的调查和基准研究是静态和特定任务的，缺乏统一的模式或机器可读的遥感基础模型表示。相比之下，我们的遥感基础模型目录（RS-FMD）将现有的基础模型整合成一个结构化、可扩展的资源，直接支持自动检索、比较和选择。另一个相关的研究方向是自动化机器学习（AutoML），其中包括Auto-WEKA [33]、Auto-sklearn [9] 和CAML [23] 等框架。它们通过元学习和优化技术自动选择参数、算法或工作流程。尽管这些方法展示了在经典机器学习环境中自动化模型选择的可行性，但尚未应用于基础模型的选择，特别是在遥感领域。我们所知，尚无现有专门的方法或智能体协助科学家选择最适合其特定约束和应用的基础模型。我们的工作填补了这一空白，通过结合RS-FMD和REMsA，提出了一种针对基础模型选择的领域专业化智能工作流程，自动匹配用户约束与合适的模型。

RS中的工具增强智能体。最近，在检索增强语言模型和工具增强智能体（如LLaVA-Plus [18]、OmniACT [12]、VideoAgent [8]）方面的发展展示了将大型语言模型与结构化检索及外部工具调用相结合以执行复杂任务和工作流程的可行性。在RS领域，已有多个研究探索了模块化智能体工作流程。GeoLLM-Squad [15] 提出了一个多智能体编排框架，将地理空间任务分解为专业子智能体，提高了可扩展性和准确性，相较于单智能体基线有明显改善。RS-Agent [38] 将检索管道与工具调度整合，以处理空间问答任务，而ThinkGeo [29] 则提出了一个基准，用于评估RS工作流程中多步骤工具增强智能体的表现。最近，EarthDial [31] 通过交互式对话建模将大型视觉-语言模型扩展到多传感器地球观测数据，支持对复杂数据模式的对话式推理。这些智能体方法突显了模块化和检索增强决策过程的优势。然而，它们主要针对地理空间信息提取、变化检测或视觉问答应用，而不专注于FM选择。REMsA 明确将一个精心策划的FM数据库与结构化检索、智能体排序、交互约束解决和透明模型推理整合，使其成为专为RS中FM选择量身定制的工具增强智能体。

# 3 遥感基础模型数据库

RS-FMD是一个经过精心策划的数据库，包含通过系统搜索识别的RSFM（约160个RSFM），作为REMsA背后的结构化资源。它通过将异构资源整合成统一的机器可读格式，从而支持可解释和约束感知的FM选择。为构建RS-FMD，我们利用多种来源进行了RSFM的系统搜索。我们审查了综述论文和流行的FM列表，调查了最近的增强现实和机器学习会议，进行了arXiv上的关键词搜索，并检查了链接的GitHub存储库。模式设计。每条记录遵循一个模式，涵盖标识符、架构、模态和预训练模型权重等属性，并为预训练数据集和基准评估提供结构化字段。该模式确保了FM之间的可追溯性、可比较性和可扩展性。完整的模式及示例记录见附录（App.）A节。这个全面的模式使我们的FM选择智能体能够将选择过程与模型能力结合起来，并将模型与用户定义的任务和约束进行匹配。同时，它还确保了关键属性，如输入数据模态、空间、光谱和时间特征，以及训练配置，能够以有原则和自动化的方式进行查询和过滤。

自动化数据库填充。填充此数据库需要从各种来源提取结构化信息，例如论文、模型卡和代码库。由于可用模型文档的规模和异质性，完全手动的策划不切实际。因此，我们采用了一种半自动化的知识提取方法，并结合了信心引导的人类验证。我们的方法是一个基于架构的LLM提取流程，受到通用知识提取方法OneKE [19]的启发，但经过了显著的调整以适应我们的领域和使用案例。具体而言，我们通过引入自己的架构定义、添加专门的信心评分步骤以及优化针对RS模型描述的提示设计来扩展他们的方法。该过程完全自动化并且是迭代的：对于每个FM，我们收集并输入一组非结构化源，然后在每次迭代中发出多个LLM调用以生成独立的结构化输出。每个输出都根据架构进行验证、解析并汇总。这种迭代策略使我们能够利用每次迭代的概率不确定性和跨迭代的一致性。对于模型产生不同输出或低对数概率的字段，标记为不确定，并传递至人类验证阶段。最终的流程有效地将复杂的异质文本源转换为机器可读的JSON记录，并尽量减少定向的手动干预。人类验证的信心评分。确保提取的元数据的可靠性对于下游RS任务中的FM选择至关重要。为此，我们为每条记录中的每个字段定义了一个信心评分，只有在不确定性较高的情况下才进行定向的人类验证。我们的信心评分结合了两个互补标准：模型生成的概率和多个LLM采样轮次输出的一致性。对于每个字段，我们的信心评分计算如下：

$$
\mathrm { C o n f i d e n c e } = w _ { \mathrm { l o g p } } \cdot \mathrm { N o r m a l i z e d L o g P r o b } + w _ { \mathrm { c o n s } } \cdot \mathrm { S e l f C o n s i s t e n c y }
$$

NormalizedLogProb 量化了大型语言模型（LLM）内部的确定性，通过将生成字段值的原始对数概率映射到一个有限范围内，而 SelfConsistency 衡量的是在多个独立采样迭代中，LLM 生成的相同值所占的比例。

为了确保可解释性和数值缩放，我们使用温度控制的 sigmoid 函数对对数概率进行归一化。我们将温度设置为 $\tau = 0.5$，以避免饱和并在 sigmoid 函数的中等置信区间内保持敏感性。我们将 $w _ { \mathrm { l o g p } } = 0.7$ 和 $w _ { \mathrm { c o n s } } = 0.3$，以优先考虑对数概率信号，同时仍然利用自一致性所带来的稳定效果。这些权重是通过对包含手动验证真值的随机抽样10个FM记录的验证集进行网格搜索经验确定的。我们优化了置信度评分与人工验证决策之间的最大一致性，使用精确率-召回曲线下面积（PR-AUC）作为选择标准。我们观察到，优先考虑对数概率信号提高了精确度，而 incorporating 自一致性有助于识别低置信度的离群值。然而，这些权重并不一定是固定的，可以根据实践者 LLM 的属性、模型领域或置信度校准需求进行调整。任何置信度低于阈值 $\theta = 0.75$ 的字段会自动标记为需要人工审核。重要的是，人工标注者仅检查被标记的字段，而不是完整的模型记录。审查所有 FM 的所有字段将需要大量的标注工作，因为每条记录包含许多异构的元数据元素。为了评估自信错误提取的风险，我们手动检查了10条记录的所有字段，发现高置信度的输出始终是准确的，支持我们评分机制的可靠性。在实践中，偶发的字段级错误对 FM 选择的影响有限，因为最决定性的属性（模态、架构、计算要求和性能）通常陈述得非常清晰，鲜少出现错误提取。覆盖多样性。目前版的 RS-FMD 涉及广泛的 RSFM，预训练于各种数据模态（多光谱、高光谱、SAR、LiDAR 和文本）并采用多样的模型架构（基于变换器的编码器、CNN-变换器混合体、视觉-语言模型）。预训练数据源从小型精心策划的数据集到百万级图像集合，空间分辨率跨度从亚米影像到粗糙的多光谱合成图。通过将这些异构信息源整合为一个具有模式指导的资源，RS-FMD 支持可重复比较、系统基准测试和自动检索。我们将通过在公共库中以宽松许可进行托管来维护 RS-FMD。我们定期监测新的 RSFM 发布并插入经过验证的条目。模型作者也可以上传新模型的文档，REMsA 自动提取元数据并将其存储在 RS-FMD 中。我们将审查提交的更新以确保一致性和可靠性。

# 4 REMsA 智能体架构

REMsA的目标是通过以决策为中心的模块化智能工作流自动选择用于推荐系统任务的功能模型。REMsA整合了结构化知识基础、LLM辅助的基于上下文的排名以及迭代澄清，以生成透明且可重现的选择。选择合适的推荐系统功能模型具有挑战性，因为这些模型在数据模态、预训练策略、基准性能和资源需求上存在差异。此外，用户经常提供不完整或含糊的任务描述，要求智能体推断用户意图并在候选模型之间平衡取舍。为了解决这些挑战，REMsA提供了一个集成管道，结合不同智能体组件和外部工具。该管道可以在定制的编排机制下实现多种功能，如结构化检索、排名、澄清和记忆存档。本节描述了智能体工作流以及每个组件和工具的细节。

# 4.1 智能体工作流程

图1展示了REMsA的架构，该架构由两个主要层次组成：LLM智能体核心和一组外部工具。LLM智能体核心包含两个关键组件：解释器（Interpreter），用于解析用户输入并提取用户意图；任务调度器（Task Orchestrator），根据当前任务状态动态决定每个步骤调用哪个外部工具。当用户提交自由文本查询时，解释器将其转换为约束的结构化表示。我们用一个精心设计的模式提示LLM，该模式涵盖与RSFM选择相关的强制性和可选字段（完整模式见附录B）。具体来说，解析器提取目标应用（例如，土地覆盖分类、水面分割）和所需的模态（例如，多光谱、SAR）作为强制性字段，以缩小FM搜索空间。然后，ReMsA通过可选字段和澄清步骤整合更广泛的实际约束，包括数据可用性、计算预算、微调要求和输出质量优先级。一旦约束可用，任务调度器启动一个控制循环，管理整个选择过程。在每个步骤中，它首先评估当前任务状态，即哪些约束可用、剩余候选者数量以及当前置信度得分。然后，它相应地调用适当的工具。如果没有缺失强制性约束，调度器调用检索工具生成初步候选集合。如果候选集合较小并且所有约束都满足，则直接应用排序工具。如果候选者数量过多或排序结果产生低置信得分，调度器调用澄清生成工具以询问用户更多输入。更新后的查询随后再通过相同的循环处理。一旦获得前 $k$ 个结果，就调用解释生成工具生成最终报告。该决策过程是通过预定义的置信度排序、约束覆盖和澄清轮次阈值执行的。调度确保工具调用是自适应和透明的。为了支持个性化和长期改进，REMsA还集成了任务记忆，存储过去的用户交互在一个轻量级向量数据库中。通过余弦相似度检索相关记忆条目，以改善未来的交互。实现的工作流程更多细节请见附录C。为了增强REMsA的可靠性，我们还设有若干内置机制以减轻故障。调度器监控置信度信号，并在排序不确定时触发澄清轮次。基于规则的约束过滤违反硬性要求的候选者。当没有候选者完全满足约束时，回退的“最接近匹配”模式返回最兼容的替代方案。我们的模块化设计还允许集成明确的反馈机制（例如，一个可选的LLM作为评判者组件，用于重新评估低质量选择），使ReMsA可扩展到更强大的错误缓解策略。

# 4.2 智能体工具

以下工具作为代理核心之外可调用的接口独立运作。每个工具由协调者根据任务状态独立调用，支持在RSFM选择工作流程中的检索、排序、澄清和解释。我们的设计支持未来工具集成的扩展性。 检索工具。为了生成初始候选集合，检索工具使用Sentence-BERT嵌入对RS-FMD中的结构约束和FM条目进行编码。为了在嵌入中保留元数据结构，每个元数据字段在编码前均以类型指示符标记（例如，[APPLICATION]、[MODALITY]）为前缀。REMSA使用Facebook AI相似性搜索（FAISS）进行高效的余弦相似度搜索。该工具返回由可配置相似度阈值决定的最相关FM列表。用户可以根据其领域要求进行调整。在我们的实验中，我们根据初步实验经验设定了该阈值，以确保广泛覆盖，同时最小化不相关的匹配。该工具优化为高召回率：包含软匹配并且不强加严格约束，允许下游流程处理更精细的过滤。 排序工具。虽然检索工具提供了一份相关FM的广泛列表，但无法完全捕捉用户特定需求和部署权衡。该任务可以由排序工具处理。排序工具使用混合策略对候选FM列表进行细化，以平衡效率、灵活性和可解释性： - 基于规则的过滤：使用确定性逻辑排除违反硬约束的候选项，如必要的模态、传感器支持或最低性能。这些硬约束是根据解释器提取的字段定义的。 - 上下文LLM排序：使用结构化查询和FM元数据重新排序其余候选项，利用专家精心设计的少量示例来说明选择。LLM返回一个按顺序排列的列表，并附上简要的理由，利用上下文排名的逻辑而无需任何模型训练（详细信息见附录D节）。我们还根据第3节为每个选择计算一个领域感知的置信度评分。 澄清生成工具。如果协调者检测到约束不足或所选FMs的整体置信度评分较低，则调用澄清工具。该工具检查解析后的模式，以确定缺失或未充分定义的字段（例如，模态、区域或性能边界），并形成澄清问题。该工具根据解释器模式生成每个问题。我们将澄清限定为三轮，以避免用户疲劳。ReMsA将响应与初始用户输入集成，解析并合并到不断演变的任务规范中，以迭代细化选择过程。 解释生成工具。一旦有可用的排序，该工具便会生成结构化的、人类可读的解释。它使用基于提示的LLM合成每个选定FM的理由，包括对候选项的适用性和权衡的关键原因。每个JSON格式的输出包括模型名称、解释的要点以及链接到相应论文和代码库。该工具通过揭示选择过程来增强透明度和用户信任（提示见附录E节）。

# 5 评估协议与基准测试

在推荐系统中评估特征模型选择的挑战在于缺乏专门的基准测试。以往的研究主要集中于评估模型在固定任务或数据集上的性能，而不是评估在多样化的现实部署约束下推荐最合适特征模型的能力。在本研究中，我们利用推荐系统特征模型数据集构建了一个基于智能体的特征模型选择基准测试，系统性地涵盖了多种模型、模态和部署约束。

基准构建。我们的评估协议依赖于结构化专家评审，保证方法论的严谨性，同时避免过度的标注负担。我们策划了一个包含100个多样化自然语言查询的基准，以便在确保专家评估的可行性的同时保持多样性和真实感。所有查询可在我们的源代码中找到。每个模型-查询对都由两位具有计算机科学背景和遥感领域专业知识的专家进行评估。我们使用结构化评分标准以确保一致性。专家程序的完整细节见附录第G节。总的来说，评估产生了3000个专家评分，因为我们评估了由REM系统选择的前三个前沿模型和三个基线系统。每个实例在七个标准上进行了仔细评分，尽管唯一查询数量适中，但仍然产生了可观的专家评估工作量。为了最大限度地提高代表性，我们使用不同场景的结构化模板创建查询，并进行具体化（完整模板见附录第H节）。这些查询在数据可用性、计算资源、应用复杂性和评估优先级方面实现了多样化。最终生成的查询涵盖了广泛的任务，包括利用合成孔径雷达数据进行洪水制图、使用多光谱或高光谱影像进行作物类型分类、利用光学时间序列进行城市扩张监测，以及灾害响应，如海冰和野火检测。这些任务涵盖了单日和多时相分析、单模态和多模态输入以及不同资源环境。所有查询均由领域专家进行审查，以确保事实准确性并校正一致性。基线。关于遥感部署的自动化前沿模型选择的先前研究有限，现有的自动化机器学习或智能体系统无法直接执行此任务。因此，我们设计了基线，既作为有意义的比较，也作为ReMsA组件分析，每个基线删除或修改特定组件以评估其贡献：

Table 1: Expert evaluation criteria.   

<table><tr><td>Criterion</td><td>Description</td></tr><tr><td>Application Compatibility</td><td>Whether the model fits the user requested application</td></tr><tr><td>Modality Match</td><td>Whether the model supports the required input data modality</td></tr><tr><td>Reported Performance</td><td>Performance reported on similar datasets or applications</td></tr><tr><td>Efficiency</td><td>Suitability for the user&#x27;s computational resources</td></tr><tr><td>Popularity</td><td>Based on GitHub repository stars and citations</td></tr><tr><td>Generalizability</td><td>Diversity and scale of pretraining data</td></tr><tr><td>Recency</td><td>Whether the model reflects recent developments</td></tr></table>

1. REMsA-NAIVE：与 REMsA 使用相同的工具集和数据库，但仅采用基本的顺序编排，而不使用 REMsA 的自适应、任务感知控制逻辑。它依赖于 LangChain 的默认单步执行，其中 LLM 独立选择工具，未形成结构化工作流程或多轮协作 [14]。此基线用于测试我们的编排机制的有效性。 2. DB-RETRIEVAL：从基于 FAISS 的 RS-FMD 稠密检索中返回前 $k$ 个模型，去除了排名、澄清、记忆和编排。这作为仅检索的基线，孤立出基于 LLM 的排名和约束推理的贡献。 3. UnsTRUcTURED-RAG：一种通用的 RAG 设置，其中 LLM 接收查询和非结构化的 FM 描述，并输出前 $k$ 个 FM 及其简要理由（见附录 F 的提示）。此基线用于测试通用 LLM 是否能够在没有我们结构化模块化智能体的情况下进行 FM 选择。

评估协议和标准。对于每个查询，ReMsA 和所有基线模型输出其前 3 个选择。这些模型-查询对由两位专家独立且盲目地使用表 1 中的 7 个标准进行评估。经过个人评分后，通过指导性讨论解决了分歧。评估是在一个评分轮次内进行的，未经进一步调整任何 FMs，以避免引入事后偏差。每个选定的 FM 在 7 个标准上以 1-5 的等级（精度为 0.5）进行评分，涵盖任务相关性和在现实世界限制下的部署适用性。几个标准使用明确的规则。例如，普遍性结合了地理、模态和数据集规模因素；受欢迎程度依赖于引用或 GitHub 活动；近新性则基于出版年份（更多细节见附录 G 节）。这些标准旨在透明、可重复，且基于实际需求，而非临时的用户偏好。最终得分是所有标准评分的加权总和（我们的权重设置见附录 I 节）。得分线性映射到 1-100 的区间，以更好地显示差异。尽管对所有候选模型进行全面的实证基准测试是不可行的，我们的协议提供了一个可重复且实际的代理，用于评估现实世界 FM 选择工作流的性能。为了支持更广泛的社区采用，我们公开发布了用于评估的完整查询集、专家指南、评分标准和模型元数据。这使得可重复性成为可能，并为未来在推荐系统及其他领域的 FM 选择研究提供了标准化基础。我们的评估不假设单一的“最佳” FM。专家比较所有系统的最高排名候选，当某系统最高排名模型被判断为比其他系统提出的模型更适用时，该系统被优先选择。ReMsA 返回前 $k$ 个 FMs 及其解释，便于用户根据自己的偏好进行选择。

Table 2: Comparison to the baselines (GPT-4.1)   

<table><tr><td>System</td><td>Avg Top-1 Avg Set</td><td></td><td>Top-1 Hit</td><td>HQ Hit</td><td>MRR</td></tr><tr><td>Remsa (Ours)</td><td>75.76</td><td>75.03</td><td>21.33%</td><td>40.00%</td><td>0.34</td></tr><tr><td>Remsa-Naive</td><td>72.67</td><td>72.00</td><td>20.00%</td><td>37.33%</td><td>0.29</td></tr><tr><td>DB-RETRiEVaL</td><td>67.37</td><td>68.87</td><td>12.00%</td><td>17.33%</td><td>0.23</td></tr><tr><td>Unstr.-RAG</td><td>71.23</td><td>68.39</td><td>13.33%</td><td>30.67%</td><td>0.24</td></tr></table>

# 6 实验

我们进行实验以全面评估 REMsA 在 RSFM 选择中的有效性。由于有限的先前工作直接针对不同部署约束下的真实 FM 选择，我们开发了自己的基线。 本节介绍我们的实验设置、定量结果和敏感性分析，随后讨论局限性和典型示例。

实验设置。我们使用 GPT-4.1 [24] 作为 REMsA 和所有基线模型的主要 LLM 核心，以确保公平性。为了评估不同主干网络的鲁棒性，我们还与 DeepSeek3.2 [5] 和 LLaMA-3.3-70B [32] 进行了额外实验。REMSA 旨在实现 LLM 无关性，可以在不改变架构的情况下轻松支持不同的 LLM。我们的基准包括 100 个多样化的自然语言用户查询。对于每个输入，REMsA 和所有基线模型（详见第 5 节）选择前 3 个候选 FMs 进行比较。领域专家根据表 1 中的标准对每个候选进行评分，我们报告单个模型和集合层面的评分以评估整体选择质量。在评估过程中，REMsA 中的所有澄清轮次均自动执行，系统与一个独立的 LLM 进行交互以模拟用户响应。这些交互中没有人类参与，从而确保了一致性并防止了跨系统和 LLM 主干的评估者偏差。评估指标。我们使用互补指标来评估最佳模型和整体集合质量：（1）平均 Top-1 分数（排名最高模型的专家评分），（2）平均集合评分（前 3 个模型的平均分数），（3）Top-1 命中率（系统排名最高的模型与专家评分最高模型匹配的比例），（4）高质量命中率（排名模型得分 ≥ 8 0 的比例），以及（5）平均倒数排名 - MRR（专家优选模型在前 3 名中的排名）。

# 6.1 与基线的比较

如表 2 所示，ReMsA 在评估的各项指标上始终优于所有基准，证明了其在各种实际约束下的 FM 选择效果。附录 J 中提供了专家评分的模型-查询对的示例。在 GPT-4.1 下，ReMsA 达到了最高的平均 Top-1 分数 (75.76) 和平均集合分数 (75.03)，这不仅表明排名靠前的模型与专家偏好一致，且前 3 名的选择提供了具有竞争力和多样化的替代方案。它还达到了最高的 Top-1 命中率 $(21.33\%)$、高质量 (HQ) 命中率 $(40.00\%)$ 和均匀调和排名 (MRR) (0.34)，显示了排名第一的精确度和稳定的排名质量。与 DB-RETRIEVAL 相比，后者依赖基于相似性的结构元数据检索，ReMsA 将 Top-1 命中率从 $12.00\%$ 提高到 $21.33\%$，HQ 命中率从 $17.33\%$ 提升至 $40.00\%$，MRR 从 0.23 增至 0.34。这突显了结构化决策逻辑在检索之外的价值，尤其是在用户查询涉及约束（例如，模态、分辨率、计算预算）时，更需要组合多个约束而非直接的元数据匹配。尽管 UnsTR.-RAG 可以访问完整的模型描述，但由于缺乏结构化指导和模块化推理，其性能仍然较低。尽管其平均 Top-1 分数为 71.23，相较于 ReMsA 较为适中，但其 Top-1 命中率 $(13.33\%$ 和 MRR (0.24) 仍远低于 ReMsA。这一结果表明，ReMsA 将结构化模式基础与任务感知工具协调相结合的能力，使其更准确地对齐用户需求。ReMsA 和 ReMsA-NAIVE 的表现显著优于仅依赖检索或非结构化 RAG 的基准，显示了我们模块化架构的有效性：将选择过程基于结构化模式并启用工具驱动的推理提供了显著优势。然而，ReMsA 在所有主要评估指标上相比于 ReMsA-NAIVE 进一步提升。例如，平均 Top-1 分数从 72.67 提升至 75.76，HQ 命中率从 $37.33\%$ 提高至 $40.00\%$，MRR 从 0.29 增加至 0.34。这些提升表明我们的协调逻辑，包括多轮澄清和决策启发式，对性能产生了重要影响，尤其是在模型选择模糊或任务表述复杂时。

Table 3: Comparison to the baselines (DeepSeek3.2)   

<table><tr><td>System</td><td>Avg Top-1 Avg Set Top-1 Hit</td><td></td><td></td><td>HQ Hit</td><td>MRR</td></tr><tr><td>Remsa (Ours)</td><td>75.35</td><td>73.81</td><td>18.67%</td><td>40.00%</td><td>0.30</td></tr><tr><td>Remsa-Naive</td><td>72.03</td><td>71.83</td><td>16.51%</td><td>36.89%</td><td>0.26</td></tr><tr><td>DB-RETRIEVAL</td><td>67.37</td><td>68.87</td><td>12.00%</td><td>17.33%</td><td>0.23</td></tr><tr><td>Unstr.-RAG</td><td>69.19</td><td>70.94</td><td>10.67%</td><td>24.00%</td><td>0.24</td></tr></table>

Table 4: Comparison to the baselines (LLaMA-3.3-70B)   

<table><tr><td>System</td><td>Avg Top-1 Avg Set Top-1 Hit</td><td></td><td></td><td>HQ Hit</td><td>MRR</td></tr><tr><td>Remsa (Ours)</td><td>73.39</td><td>70.34</td><td>14.67%</td><td>32.00%</td><td>0.26</td></tr><tr><td>Remsa-Naive</td><td>69.02</td><td>69.00</td><td>14.23%</td><td>29.47%</td><td>0.24</td></tr><tr><td>DB-RETRiEVaL</td><td>67.37</td><td>68.87</td><td>12.00%</td><td>17.33%</td><td>0.23</td></tr><tr><td>Unstr.-RAG</td><td>69.87</td><td>68.04</td><td>10.00%</td><td>26.67%</td><td>0.22</td></tr></table>

延迟权衡。为了评估延迟与性能之间的权衡，我们测量每个查询的平均端到端运行时间。正如预期的那样，单步方法更快：DB-检索耗时0.77秒，Unstr.-RAG耗时11.9秒，而REMsA-Naive耗时22.7秒，而REMsA由于多阶段决策处理和可选的澄清需要31.7秒。尽管这种适度的额外开销，ReMsA在主要指标上实现了最高的专家验证准确性，表明其额外的推理步骤带来了有意义且一致的提升。

# 6.2 不同大语言模型主干网络的影响

为了评估ReMsA在不同大型语言模型主干下的鲁棒性，我们进一步比较了使用GPT-4.1、DeepSeek3.2和LLaMA3.3-70B的结果。如表2、表3和表4所示，在所有三个核心上，REMsA始终优于相应的基线，表明改进主要源于整体架构，而非特定的语言模型。GPT-4.1的表现最强（平均Top-1: 75.76；MRR: 0.34），紧随其后的是DeepSeek3.2（75.35；0.30），而LLaMA3.3-70B的结果略低但仍具竞争力（73.39；0.26）。值得注意的是，HQ命中率在GPT-4.1和DeepSeek3.2之间保持相对稳定（均为$4 0 . 0 0 \%$），这表明高质量候选人识别对主干选择具有鲁棒性。在所有大型语言模型主干上，REMsA-NAIVE和基于检索的基线之间的稳定优势表明，结构化的基础和基于工具的编排提供了一致的增益，而与底层语言模型无关。

Table 5: Sensitivity analysis on evaluation criteria   

<table><tr><td>Criteria Setting</td><td>Avg Set</td><td>Top-1 Hit MRR</td><td></td><td>Note</td></tr><tr><td>Full Scoring (All Criteria)</td><td>75.03</td><td>22.67%</td><td>0.38</td><td></td></tr><tr><td>w/o Application Compatibility</td><td>73.32</td><td>21.33%</td><td>0.36</td><td>Green:</td></tr><tr><td>w/o Modality Match</td><td>70.88</td><td>22.67%</td><td>0.36</td><td>Increase</td></tr><tr><td>w/o Reported Performance</td><td>75.05</td><td>22.67%</td><td>0.38</td><td>Red:</td></tr><tr><td>w/o Efficiency</td><td>80.23</td><td>25.33%</td><td>0.38</td><td>Drop</td></tr><tr><td>w/o Popularity+Recency</td><td>75.13</td><td>25.33%</td><td>0.39</td><td></td></tr><tr><td>w/o Generalizability</td><td>75.10</td><td>22.67%</td><td>0.38</td><td></td></tr></table>

# 6.3 评估标准的敏感性分析

为了理解 ReMsA 与专家定义的评估原则的对齐程度，我们通过逐一移除专家评估协议中的每个评分标准进行敏感性分析。如表 5 所示，在大多数维度上，性能总体上是稳健的，但一些移除的评估标准揭示了哪些标准对有效模型选择贡献最大。移除应用兼容性和模态匹配会导致显著的性能下降，确认 ReMsA 确实优先考虑与用户目标对齐的功能适当模型。值得注意的是，移除报告性能和可泛化性对整体结果的变化很小，这意味着这些维度要么通过其他标准隐含地被捕获，要么在当前基准设置中不太具决定性。相比之下，移除效率或流行性+时效性反而导致性能适度提升。这表明，尽管这些标准为部署增加了实际相关性，但有时可能会优先考虑知名或资源高效的模型，而非技术上最优的模型。敏感性结果进一步验证了 ReMsA 并未过度拟合于表面指标，如引用或时效性，而是强调核心兼容性和约束满足作为其最终决策的依据。

# 7 结论与讨论

我们提出了REMsA，一种结合FM数据库用于真实RSFM选择问题的LLM智能体。通过协调检索、排名、澄清、解释的模块化工具，并提供增强记忆的选择，REMsA能够实现可靠且一致的FM选择。我们的结构化RSFM数据库RS-FMD整合了异构描述，以便于透明的检索和比较。在我们的专家驱动基准测试中，REMsA优于仅检索的、非结构化的RAG和幼稚智能体基准。此项工作为未来研究开辟了多个方向，包括将基准扩展到更复杂的场景，探索自适应FM选择策略。尽管REMsA在RSFM选择中表现良好，但仍然存在一些局限性。我们的基准包括100个专家注释的查询，这可能漏掉一些稀有或新兴的用例，尽管评估需要大量工作，包括3,000个专家评分。此外，排名依赖于上下文学习而非监督训练，这可能限制在复杂查询上的表现。

# References

1 Adorni, P., Pham, M., et al.: Towards efficient benchmarking of foundation models in remote sensing: A capabilities encoding approach. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, CVPR (2025), https://openaccess.thecvf.com/content/CVPR2025w/MORSE/html/ Adorni_Towards_Efficient_Benchmarking_of_Foundation_Models_in_Remote_Sensing_A_CVPRW_ 2025_paper.html   
2Astruc, G., Gonthier, N., et al.: Omnisat: Self-supervised modality fusion for earth observation. In: Computer Vision - ECCV (2024). https://doi.org/10.1007/978-3-031-73390-1_24   
3. Cekmeceli, K., Himmetoglu, M., et al.: Do vision foundation models enhance domain generalization in medical image segmentation? In: Computer Vision - ECCV Workshops (2024). https://doi.org/10. 1007/978-3-031-91672-4_12   
4. Chu, M., Zheng, Z., et al.: Towards natural language-guided drones: Geotext-1652 benchmark with spatial relation matching. In: Computer Vision - ECCV (2024). https://doi.org/10.1007/978-3- 031-73247-8_13   
5. DeepSeek-AI: Deepseek-v3.2: Pushing the frontier of open large language models. CoRR abs/2512.02556 (2025). https://doi.org/10.48550/ARXIV.2512.02556   
6Don, M. Pinson, S., et al.: Foundation model or finetune?evaluation of few-shot semantic segmentation for river pollution. In: Computer Vision - ECCV Workshops (2024). https://doi.org/10.1007/978- 3-031-92089-9_21   
7.Drusch, M., Bello, U.D., et al.: Sentinel-2: Esa's optical high-resolution mission for gmes operational services. Remote sensing of Environment 120, 2536 (2012)   
8. Fan, Y., Ma, X., et al.: Videoagent: A memory-augmented multimodal agent for video understanding. In: Computer Vision - ECCV (2024). https://doi.org/10.1007/978-3-031-72670-5_5   
9Feurer, M., Klein, A., et al.: Efficient and robust automated machine learning. In: Advances in Neural Information Processing Systems 28: Annual Conference on Neural Information Processing Systems (2015), https://proceedings.neurips.cc/paper/2015/hash/11d0e6287202fced83f79975ec59a3a6- Abstract.html   
10. Guanter, L., Kaufmann, H., et al.: The enmap spaceborne imaging spectroscopy mission for earth observation. Remote. Sens. 7(7), 88308857 (2015). https://doi.org/10.3390/RS70708830   
11. Guo, X., Lao, J., et al.: Skysense: A multi-modal remote sensing foundation model towards universal interpretation for earth observation imagery. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR (2024). https://doi.org/10.1109/CVPR52733.2024.02613   
1Kapoor, R., Butala, Y.P., et al.: Omniact: A dataset and benchmark for enabling multimodal generalist autonomous agents for desktop and web. In: Computer Vision - ECCV (2024). https://doi.org/10. 1007/978-3-031-73113-6_10   
13. Lacoste, A., Lehmann, N., et al.: Geo-bench: Toward foundation models for earth monitoring. In: Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems, NeurIPS (2023), http://papers .nips. cc/paper_files/paper/2023/hash/ a0644215d9cff6646fa334dfa5d29c5a-Abstract-Datasets_and_Benchmarks.html   
14. LangChain: Langchain.https://python.langchain.com/docs/introduction/ (2025), online; accessed 20-February-2026   
15. Lee, C., Paramanayakam, V., et al.: Multi-agent geospatial copilots for remote sensing workflows. CoRR abs/2501.16254 (2025). https://doi.org/10.48550/ARXIV.2501.16254   
1 Li, Y., Tan, J., et al. Unleashing the potential of remote sensing foundation models via bridging data and computility islands. The Innovation (2025)   
7 Li, Z. Hou, B., et al.: Masked angle-aware autoencoder for remote sensing images. In: Computer Vision - ECCV (2024). https://doi.org/10.1007/978-3-031-73242-3_15   
LS.ChegH.   Llava-plus Lerousool o ctiultal aents.InCo Vision - ECCV (2024). https://doi.org/10.1007/978-3-031-72970-6_8   
19. Luo, Y., Ru, X., et al.: Oneke: A dockerized schema-guided LLM agent-based knowledge extraction system. In: Companion Proceedings of the ACM on Web Conference 2025, WWW (2025). https: //doi.org/10.1145/3701716.3715189   
20. Meta: Faiss. https://ai.meta.com/tools/faiss/ (2025), online; accessed 20-February-2026   
2Muhtar, D., Li, Z., et al.: Lhrs-bot: Empowering remote sensing with vgi-enhanced large multimodal language model. In: Computer Vision - ECCV (2024). https://doi.org/10.1007/978-3-031-72904- 1_26   
2 Nedungadi, V., Kariryaa, A., et al.: Mmearth: Exploring multi-modal pretext tasks for geospatial representation learning. In: Computer Vision - ECCV (2024). https://doi.org/10.1007/978-3-031- 73039-9_10   
23. Neutatz, F., Lindauer, M., et al.: Automl in heavily constrained applications. VLDB J. 33(4), 957979 (2024). https://doi.org/10.1007/S00778-023-00820-1   
24. OpenAI: Gpt-4.1. https://openai.com/index/gpt-4-1/ (2025), online; accessed 20-February-2026   
25. Plekhanova, E., Robert, D., et al.: Ssl4eco: A global seasonal dataset for geospatial foundation models in ecology. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, CVPR (2025), https://openaccess.thecvf.com/content/CVPR2025w/EarthVision/html/ Plekhanova_SSL4Eco_A_Global_Seasonal_Dataset_for_Geospatial_Foundation_Models_in_CVPRW_ 2025_paper.html   
26. Purohit, M., Muhawenayo, G., et al.: How does the spatial distribution of pre-training data affect geospatial foundation models? CoRR abs/2501.12535 (2025). https://doi.org/10.48550/ARXIV. 2501.12535   
27. Ramachandran, R., Roy, e.a.: A primer for assessing foundation models for earth observation. Cornell University (2025)   
28. Reimers, N., Gurevych, I.: Sentence-bert: Sentence embeddings using siamese bert-networks. In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP (2019). https://doi.org/10.18653/V1/D19-1410   
9Shabbir, A., Munir, M.A., et al.: Thinkgeo: Evaluating tool-augmented agents for remote sensing tasks. CoRR abs/2505.23752 (2025). https://doi.org/10.48550/ARXIV.2505.23752   
0Simumba, N., Lehmann, N., et al.: Geo-bench-2: From performance to capability, rethinking evaluation in geospatial AI. CoRR abs/2511.15658 (2025). https://doi.org/10.48550/ARXIV.2511.15658   
3Soni, S., Dudhane, A., et al Earthdial: Turning multi-sensory earth observations to interactive dialogues. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR (2025). https://doi.org/10.1109/CVPR52734.2025.01334   
32. Team, L.: The llama 3 herd of models. CoRR abs/2407.21783 (2024). https://doi.org/10.48550/ ARXIV.2407.21783   
33. Thornton, C., Hutter, F., et al.: Auto-weka: combined selection and hyperparameter optimization of classification algorithms. In: The 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD (2013). https://doi.org/10.1145/2487575.2487629   
34.Torres, R. Snoeij, P., et al.: Gmes sentinel-1 mission. Remote sensing of Environment 120, 924 (2012)   
3Wang, X., Zhang, Y., et al.: Videoagent: Long-form video understanding with large language model as agent. In: Computer Vision - ECCV (2024). https://doi.org/10.1007/978-3-031-72989-8_4   
I S In: Computer Vision - ECCV (2024). https://doi.org/10.1007/978-3-031-73013-9_23   
37. Xiao, A., Xuan, W., et al.: Foundation models for remote sensing and earth observation: A survey. CoRR abs/2410.16602 (2024). https://doi.org/10.48550/ARXIV.2410.16602   
38. Xu, W., Yu, Z., et al.: Rs-agent: Automating remote sensing tasks through inteligent agents. CoRR abs/2406.07089 (2024). https://doi.org/10.48550/ARXIV.2406.07089

# REMsA: Foundation Model Selection for Remote Sensing via a Constraint-Aware Agent

# Appendix Overview

Sec. A: Complete RS-FMD Schema Specification.   
Sec. B: Structured Query Schema.   
Sec. C: Implementation Details.   
Sec. D: LLM-Based In-Context Ranking Prompt.   
Sec. E: Explanation Generator Prompt.   
- Sec. F: Prompt for RAG-LLM Baseline.   
Sec. G: Expert Evaluation Procedure.   
- Sec. H: Query Template for Creating Benchmark Dataset.   
Sec. I: Expert Scoring Weight Configuration.   
Sec. J: Illustrative Examples of Expert Scoring.

# Appendix

# A Complete RS-FMD Schema Specification

To properly represent the properties of each FM, we designed a comprehensive data schema for RS-FMD. The schema includes the essential characteristics of model architectures, pretraining strategies, supported modalities, and benchmark performance.

Each model record includes fields such as unique identifiers, names, versions, release and update timestamps, and links to associated publications, code repositories, and pretrained weights. These metadata elements ensure traceability and reproducibility of the database entries.

Beyond these core descriptors, the schema incorporates detailed fields that capture architectural specifics (e.g., backbone type, number of layers, number of parameters), pretraining approaches (e.g., pre-text training type, masking strategy), and modality integration. The design anticipates the diversity of RS models and supports future extensions.

To capture information about pretraining and evaluation comprehensively, the schema defines two nested structures:

PretrainingPhase: This substructure records the datasets used for pretraining, geographical coverage, time range, image resolutions, token sizes, augmentation strategies, sampling methods, and masking ratios.

Benchmark: This substructure captures evaluation metrics, including the applications, dataset, performance scores, and training hyperparameters used during evaluation.

Many fields are annotated with free_text metadata. This annotation signals that the field may contain natural language summarization that requires specialized treatment in confidence scoring and downstream verification.

Tab. 6 provides a comprehensive description of the fields of our data schema in RS-FMD, including nested structures for pretraining phases and benchmarks.

Table 6: Complete schema specification of RS-FMD, including nested pretraining phases and benchmarks   

<table><tr><td>Field</td><td>Type</td><td>Description</td></tr><tr><td colspan="3">Main Model Fields</td></tr><tr><td>model_id</td><td>string</td><td>Unique identifier of the model (free text)</td></tr><tr><td>model_name</td><td>string</td><td>Only the name of the model without ex- tra descriptions (free text)</td></tr><tr><td>version</td><td>string</td><td>Version identifier (free text)</td></tr><tr><td>release_date</td><td>date</td><td>Release date of the model</td></tr><tr><td>last _updated</td><td>date</td><td>Last updated date</td></tr><tr><td>short_description</td><td>string</td><td>Short summary describing the model</td></tr><tr><td>paper_link</td><td>URL</td><td>(free text) URL to the associated publication</td></tr><tr><td>citations</td><td>integer</td><td>Number of citations</td></tr><tr><td>repository</td><td>URL</td><td>URL to the code repository</td></tr><tr><td>weights</td><td>URL</td><td>URL to pretrained model weights</td></tr><tr><td>backbone</td><td>string</td><td>Specific backbone used (free text)</td></tr><tr><td>num_layers</td><td>integer</td><td>Number of layers</td></tr><tr><td>num_parameters</td><td>float</td><td>Model size in millions of parameters</td></tr><tr><td>pretext_training_type</td><td>string</td><td>Type of pretext training strategy (free text)</td></tr><tr><td>masking_strategy</td><td>string</td><td>Masking strategy applied during training</td></tr><tr><td>pretraining</td><td>string</td><td>(free text) Description of pretraining approach (free</td></tr><tr><td>domain _knowledge</td><td>list[string]</td><td>text) Domain-specific knowledge or methods</td></tr><tr><td>backbone _modifications</td><td>list[string]</td><td>incorporated Modifications made to the backbone</td></tr><tr><td>supported _sensors</td><td>list[string]</td><td>Supported satellite sensors</td></tr><tr><td>modality_integration_type</td><td>string</td><td>Integration type (free text)</td></tr><tr><td>modalities</td><td>list[string]</td><td>Input data modalities (free text)</td></tr><tr><td>spectral_alignment</td><td>{full, partial, none}</td><td>Whether the model models spectral con- tinuity</td></tr><tr><td>temporal_alignment</td><td>{full, partial, none}</td><td>Whether the model models temporal se- quences</td></tr><tr><td>spatial _resolution</td><td>string</td><td>Spatial resolution of data (free text)</td></tr><tr><td>temporal _resolution bands</td><td>string</td><td>Temporal resolution of data (free text)</td></tr><tr><td></td><td>list[string]</td><td>Spectral bands used</td></tr><tr><td colspan="3">Nested: PretrainingPhase</td></tr><tr><td>regions _coverage</td><td>string list[string]</td><td>Dataset used for pretraining (free text) Geographical regions covered</td></tr><tr><td>time_range</td><td>string</td><td>Time range of pretraining data (free text)</td></tr><tr><td>num _images</td><td>integer</td><td>Number of images used</td></tr><tr><td>token_size</td><td>string</td><td>Token size (free text)</td></tr><tr><td>image_resolution</td><td>string</td><td>Input image resolution (free text)</td></tr><tr><td></td><td>integer</td><td>Number of epochs</td></tr><tr><td>epochs</td><td>integer</td><td>Batch size</td></tr><tr><td>batch _size</td><td>string</td><td>Learning rate (free text)</td></tr><tr><td>learning_rate</td><td>list[string]</td><td></td></tr><tr><td>augmentations</td><td></td><td>Augmentations applied</td></tr><tr><td>processing</td><td>list[string]</td><td>Additional preprocessing steps</td></tr><tr><td>sampling</td><td>string</td><td>Sampling strategy (free text)</td></tr><tr><td>processing_level</td><td>string</td><td>Processing level (free text)</td></tr><tr><td>cloud_cover</td><td>string</td><td>Cloud cover filtering (free text)</td></tr><tr><td>missing_data</td><td>string</td><td>Handling of missing data (free text)</td></tr><tr><td colspan="3">float Masking ratio Nested: Benchmark</td></tr><tr><td></td><td></td><td></td></tr><tr><td>application_type</td><td>string</td><td>Type of application evaluated (free text)</td></tr><tr><td>application</td><td>string</td><td>Specific application domain (free text)</td></tr><tr><td>dataset</td><td>string</td><td>Benchmark dataset name (free text)</td></tr><tr><td>metrics</td><td>list[string]</td><td>List of evaluation metrics</td></tr><tr><td>metrics_value</td><td>list[foat]</td><td>Numeric values for each metric</td></tr><tr><td>sensor</td><td>list[string]</td><td>Sensors used</td></tr><tr><td>regions</td><td>list[string]</td><td>Regions evaluated</td></tr><tr><td>original _ samples num _samples</td><td>integer integer</td><td>Total number of samples before sampling Actual number of samples used</td></tr></table>

REMsA: Foundation Model Selection for Remote Sensing via a Constraint-Aware Agent 17   

<table><tr><td>Field</td><td>Type</td><td>Description</td></tr><tr><td>sampling_percentage</td><td>float</td><td>Fraction of dataset retained (0100)</td></tr><tr><td>num _classes</td><td>integer</td><td>Number of classes</td></tr><tr><td>classes</td><td>list[string]</td><td>Names of each class</td></tr><tr><td>image _resolution</td><td>string</td><td>Input image resolution (free text)</td></tr><tr><td>spatial_resolution</td><td>string</td><td>Spatial resolution (free text)</td></tr><tr><td>bands used</td><td>list[string]</td><td>Bands used during evaluation</td></tr><tr><td>augmentations</td><td>list[string]</td><td>Data augmentations applied</td></tr><tr><td>optimizer</td><td>string</td><td>Optimizer used (free text)</td></tr><tr><td>batch_size</td><td>integer</td><td>Batch size</td></tr><tr><td>learning_rate</td><td>float</td><td>Learning rate</td></tr><tr><td>epochs</td><td>integer</td><td>Number of epochs</td></tr><tr><td>loss _function</td><td>string</td><td>Loss function (free text)</td></tr><tr><td>split_ratio</td><td>string</td><td>Train/val/test split ratio (free text)</td></tr></table>

Below we include a complete example of an RS-FMD record for the RSFM A2-MAE. This illustrates how the schema is instantiated with real metadata.

31 "Gaofen-1: B1-B4",   
32 "Gaofen-2: B1-B4"   
33 ]   
34 "pretraining_phases": [   
35 {   
36 "dataset": "STSSD",   
37 "regions_coverage": ["Global (12k urban centers, 10k nature reserves)"],   
38 "time_range": "2020-2023",   
39 "num_images": 2500000,   
40 "token_size": "16x16",   
41 "image_resolution": "0.8-30m (cropped 256x256 to 3200x3200)",   
42 "epochs": 130,   
43 "batch_size": 1024,   
44 "learning_rate": "1e-4 (cosine decay)",   
45 "processing":   
46 "Atmospheric/radiation correction",   
47 "Pan-sharpening (Gaofen)",   
48 "Cropping/resizing alignment"   
49 ],   
50 "sampling": "Clustering-based pruning (keep hardest 10%)",   
51 "cloud_cover": $" > = 1 0 \%$ ;   
52 "masking_ratio": 0.75   
53 }   
54   
55 "benchmarks": [   
56 {   
57 "task": "Classification",   
58 "application": "Land cover classification",   
59 "dataset": "EuroSAT",   
60 "metrics": ["Accuracy"],   
61 "metrics_value": [99.09],   
62 "sensor": ["Sentinel-2"],   
63 "regions": ["34 European countries"]   
64 },   
65 {   
66 "task": "Classification",   
67 "application": "Multi-label classification",   
68 "dataset": "BigEarthNet",   
69 "metrics": ["mAP"],   
70 "metrics_value": [83.0]   
71 },   
72 {   
73 "task": "Segmentation",   
74 "application": "Surface water segmentation",   
75 "dataset": "Sen1Floods11",   
76 "metrics": ["mIoU"],   
77 "metrics_value": [88.87]   
78 },   
79 {   
80 "task": "Segmentation",   
81 "application": "Cropland segmentation",   
82 "dataset": "CropSeg",   
83 "metrics": ["mIoU"],   
84 "metrics_value": [44.81]   
85 },   
86 {   
87 "task": "Change Detection",   
88 "application": "LEVIR-CD",   
89 "dataset": "LEVIR-CD",   
90 "metrics": ["mIoU"],   
91 "metrics_value": [84.32]   
92 },   
93 {   
94 "task": "Change Detection",   
95 "application": "Urban change detection",   
96 "dataset": "OSCD",   
97 "metrics": ["F1"],   
98 "metrics_value": [53.97]   
99 },   
100 {   
101 "task": "Change Detection"   
102 "application": "Semantic change segmentation"   
103 "dataset": "DynamicEarthNet",   
104 "metrics": ["mIoU"],   
105 "metrics_value": [46.0]   
106 }   
107 ]   
108 }

# B Structured Query Schema

Below we show the complete JSON schema template used by the query interpreter:

"application": "string", // Mandatory   
"modality": "string", // Mandatory   
"sensor": "string or list of strings", // Optional   
"spatial_resolution": "string or numeric", // Optional   
"temporal_resolution": "string or numeric",// Optional   
"bands": "list of strings", // Optional   
"avaliable_data": "string", // Optional   
"deployment_device": "string", // Optional   
"priority_metrics": "list of string", // Optional   
"min_performance": { // Optional "metric": "list of string", "value": "list of number"   
},

"region": "string or list of strings", // Optional "domain_keywords": "list of strings" // Optional }

# C Implementation Details

# Algorithm 1: ReMsA Workflow for RSFM Selection

Input: User Query q, desired number of recommendations k Output: Top- $k$ selected models with explanations 1 Initialize ClarifyCounter $ 0$ 2 Initialize MaxClarif $\mathit { i } _ { y } \gets 3$ 3 repeat 4 Constraints ParseQuery $( q )$ ; // LLM parses constraints 5 if mandatory constraints missing then 6 if ClarifyCounter $<$ MaxClarify then 7 $q \gets$ ClarifyUser(q, Constraints) Increment Clarif yCounter 8 else 9 break ; // Stop clarifying to avoid user fatigue 10 until All mandatory constraints are present; 11 Candidates RetrieveModels(q) ; // Embedding retrieval (Top K) 12 Filtered FilterCandidates(Candidates, Constraints) 13 if $| F i l t e r e d | = 0$ then 14 BestMatch SelectClosestModel(Candidates, Constraints) 15 Explanation GenerateExplanation(q, BestMatch) 16 return {Recommendation: BestMatch, Explanation} 17 if |Filt $e r e d | > .$ MaxCandidates then 18 if ClarifyCounter $<$ MaxClarify then 19 $q $ ClarifyUser(q, Constraints) 20 Increment ClarifyCounter 21 Go to line 3 ; // Restart process with clarified query 22 Scores RankCandidates(q, Filtered) OverallConfidence ComputeConfidence(Scores) 23 if OverallConfidence $<$ ConfidenceThreshold then 24 if ClarifyCounter $<$ MaxClarify then 25 $q $ ClarifyUser(q, Constraints) 26 Increment Clarif yCounter 27 Go to line 3 28 $T o p K \gets \mathrm { T o p } { - k }$ candidates in Filtered ranked by Scores 29 Explanation GenerateExplanation(q, TopK) 30 return {Recommendations: TopK, Explanation}

The workflow of REMsA is shown in Algorithm 1. The pipeline is implemented in Python using pydantic for schema validation, and the OpenAI GPT-based models for extraction. Each input document is processed in multiple iterations to collect diverse generations. The RS-FMD is stored in JSONL records and versioned via DVC to ensure reproducibility.

# D LLM-Based In-Context Ranking Prompt

To re-rank candidate foundation models without training a dedicated learning-to-rank model, we leverage in-context learning (ICL) with a LLM. The prompt explicitly instructs the LLM to prioritize user requirements, compare candidate models, and produce a ranked list with explanations. We provide few-shot examples created by an expert in the prompt to guide the model toward consistent ranking behavior. The prompt is connected to RS-FMD to provide the metadata of the candidate models. Below is the prompt template we are using in the ranking module:

Prompt Template:

You are an expert in remote sensing foundation model selection.

You will be given:

1. A structured user query specifying task requirements and constraints.

2. A list of candidate models retrieved from a database, each with metadata fields.

Your goal:

- Rank the candidate models from most to least suitable for the user's query.

For each model, provide a brief explanation in several bullet points describing why it is placed at that rank.

Prioritize hard constraints (application, modality, required sensor, and min_performance if provided), then consider secondary preferences (spatial/temporal resolution, application type, domain keywords, etc.).

When two models equally satisfy the constraints and preferences, prefer the model that is more efficient, better validated on diverse benchmarks, or more versatile(multimodal multi-temporal)

[Example]   
Structured Query: "application": "land cover classification" "modality' "multispectral", "sensor": ["Sentinel-2"], "min_performance" { "metric' ["accuracy"], "value": [85] }   
}

Candidate Models:

1 S2MAE   
2. Prithvi   
3. CACo

Ranking Output:

1 S2MAE

Directly supports Sentinel-2 multispectral data Achieves 99.1\% accuracy on EuroSAT, exceeding 85\% requirement Purpose-built for land cover classification

2. Prithvi

Supports multi-temporal multispectral data, including Sentinel-2 Accuracy slightly below requirement on similar tasks More generalist FM

3. CACO

Only supports RGB modality   
Accuracy below the 85\% requirement   
Designed mainly for change detection and event retrieval

Your Task:

Given the following new query and candidates, produce a ranked list with explanations.   
Structured Query:   
{query}   
Candidate Models:   
{candidates}   
Please output the ranked list as JSON in the following format: { "model": <model_name> "rank": <integer>, "reason": [<short bullet points>] },

# E Explanation Generator Prompt

The explanation generator uses an LLM to produce concise, interpretable justifications for the final ranked FM list. The prompt template in our explanation generator is given as follows:

You are an expert in remote sensing foundation model selection.

The structured user query is: {query}

The final ranked candidate models with their metadata are: {ranked_models}

Your task:

1. For each model, output a JSON object with: "model_name" - "explanation" (several bullet points on why it is recommended) "paper_link" "repository"

2. Highlight how the model satisfies or partially satisfies the query.

3. Mention key trade-offs if relevant (accuracy vs. efficiency, modality coverage, etc.).

# F Prompt for RAG-LLM Baseline

For the LLM-RAG baseline, we prompt an LLM with the original user input and the retrieved model documentation as a context. The LLM is instructed to select and rank the top three remote sensing foundation models and provide concise explanations for each recommendation.

You are an expert in remote sensing foundation models.

The user has provided the following task description: {user_input}

Below is a set of candidate models with their documentation: {context_str}

Your task:

1. Select and rank the top 3 remote sensing foundation models most suitable for the task.   
2. For each selected model, provide: -- A short explanation of why it fits the task requirements.   
-- The reason for its ranking position compared to others.   
-- Any other relevant information from the context.

3. Follow this exact output format:

1. model: <model_name>   
explanation:   
<reason 1>   
- <reason 2>   
- <reason 3>   
2. model: <model_name>   
explanation:   
<reason 1>   
<reason 2>   
- <reason 3>

3. model: <model_name> explanation: <reason 1> - <reason 2> - <reason 3>

# G Expert Evaluation Procedure

Expert Background. All annotations were performed by two experts with a computer science background and specialization in RS. Both have prior experience working with RSFMs, have published in the relevant domains, and are familiar with model architectures, pretraining datasets, and evaluation practices.

Annotation Protocol. To ensure consistency and reproducibility, we followed a structured, multistage scoring protocol:

Rubric Design. We created a detailed rubric for all seven criteria in Tab. 1, including definitions, examples, and decision rules.   
- Calibration Phase. Both experts annotated an initial subset of model-query pairs. Disagreements were used to refine the rubric until interpretations aligned.

Independent and Blind Scoring. Experts then rated all remaining model-query pairs independently and without access to system identities or each other's scores.

Disagreement Resolution. Any pair with substantial disagreement was re-examined in a controlled discussion, with decisions resolved strictly according to the rubric.

Objective Scoring Rules. Where possible, we used explicit rules to reduce subjectivity:

Reported Performance. Reported performance was determined by checking for benchmarks that matched the queried task. If none existed, we evaluated performance on broader but related tasks. For example, if the query specifies the task as scene classification, and there is no benchmark for this, we look for general classification benchmarks. Depending on its performance, this model gets a moderate/high reported performance score. Models with no relevant benchmarks received a low score.

Efficiency. Model parameter counts were normalized to a 0-5 scale as a proxy for complexity, and combined with reported performance to obtain a final efficiency score. Specifically, we divide this complexity measure by the reported performance to produce a final effciency score, also on a 0-5 scale. Popularity. Popularity was used as a practical usability indicator rather than a measure of inherent model quality. We used normalized GitHub star counts (when code exists) and Google Scholar citation counts (when paper is unavailable). This reflects maturity, community adoption, and available ecosystem support.

Generalizability. We quantified pretraining diversity using three measurable components extracted from official FM documentation:

Gegraphidiversiy:gobal (score, mlti-regional (3),  single-regi coverage ).

2. Sensor-modality diversity: number of distinct modalities used in pretraining e.g., optical, SAR, multispectral, hyperspectral).

3. Dataset scale: reported total area, number of scenes, or total images.

These components were combined into a composite 1-5 score. Inter-annotator agreement confirmed that the rule-based definitions reduced subjectivity.

Recency. Recency was defined by the publication year or the latest model-card update:

$$
2 0 2 5 - 2 0 2 6 = 5 , \quad 2 0 2 4 = 4 , \quad 2 0 2 3 = 3 , \quad 2 0 2 2 = 2 , . . .
$$

Given the rapid evolution in RSFMs, this criterion serves as a soft heuristic rather than a primary determinant.

Reference Sources. All judgments were grounded in publicly available references for each foundation model. Experts used: (1) published papers and preprints; (2) official GitHub repositories and model documentation; (3) public benchmark results; (4) citation databases; and (5) described pretraining datasets from official sources. These references provided the necessary information on modality support, reported performance, efficiency, generalizability, popularity, and recency.

# H Query Template for Creating Benchmark Dataset

To construct a representative and diverse benchmark dataset for evaluation, we define 16 structurec query templates. Each template corresponds to a specific category of user constraints:

# Data Availability (A1-A5):

• A1: No Training Data — User wants to use pre-trained models directly.   
• A2: Sufficient Labeled Data — User has enough labels to fine-tune or train from scratch.   
• A3: Few-shot Labels — User has a small set of labeled data only and requires models that generalize in low-data regimes.   
• A4: Unlabeled Data Only — User has input data but no labels and seeks models suited for unsupervised or self-supervised settings.   
• A5: Data Adaptation Needed — User's data differs from typical inputs, requiring domain adaptation or compatibility adjustments.

Table 7: Structured query templates used for benchmark dataset generation. Each template maps to one constraint category. Slot values ({application}, {sensor}, {region}) are drawn from a predefined vocabulary and paraphrased by an LLM.   

<table><tr><td>Template</td><td>Categories</td></tr><tr><td>I&#x27;m looking for a model I can use out-of-the-box for {application} using {modality} data. I don&#x27;t have any labeled training data.</td><td>A1</td></tr><tr><td>I have a well-labeled dataset for {application} with {modality} in {region}. Which model would be best to fully fine-tune from scratch?</td><td>A2</td></tr><tr><td>I only have a few labeled samples for {application} using {sensor}. I want a model that can adapt well in a few-shot setting.</td><td>A3</td></tr><tr><td>I have a lot of unlabeled {modality} imagery from {region}. I need a model that works well with self-supervised or unsupervised learning for {application}.</td><td>A4</td></tr><tr><td>My data uses {sensor} with {spatial_resolution} resolution, but most models I&#x27;ve seen don&#x27;t support it. Can you recommend one that can be adapted?</td><td>A5</td></tr><tr><td>I&#x27;m working on {application} but only have access to a laptop with no GPU. Which model would be small enough to run locally?</td><td>B1</td></tr><tr><td>I&#x27;m using a desktop with a single GPU and doing {application} on {modality} imagery. Which models balance performance and efficiency?</td><td>B2</td></tr><tr><td>For {application}, I have access to cloud GPUs and can afford large models. What&#x27;s the most powerful foundation model I can try?</td><td>B3</td></tr><tr><td>I&#x27;m doing basic {application} (e.g., 34 land classes). What lightweight model would you suggest for fast experimentation?</td><td>C1</td></tr><tr><td>I&#x27;m working on multi-class classification {application} with {modality} images. The task isn&#x27;t trivial, but I don&#x27;t need pixel-level precision.</td><td>C2</td></tr><tr><td>I need a model for high-resolution segmentation or fine-grained {application}. Accuracy and spatial detail are important.</td><td>C3</td></tr><tr><td>For {application} using {sensor} data, I mainly care about achieving the highest overall accuracy, even if the model is large.</td><td>D1</td></tr><tr><td>For {application} using {sensor} imagery, I want clean and accurate outputs with minimal false detections; clear boundaries and reliable predictions are most important.</td><td>D2</td></tr><tr><td>For {application} using {sensor} imagery, I need to ensure all target instances are captured, even if some false alarms occur; completeness is critical.</td><td>D3</td></tr><tr><td>I need fast inference for {application} in near real-time on {device}. What&#x27;s a good lightweight model? I&#x27;m doing {application} on {modality} in {region}, but I only have few-shot labels</td><td>D4</td></tr></table>

# Computational Resources (B1-B3):

• B1: Limited Resources — e.g., CPU-only laptop.   
•B2: Moderate Resources — e.g., desktop with GPU.

•B3: High Resources — e.g., cluster-scale GPU compute.

# Application Complexity (C1C3):

• C1: Simple Application — Applications with low label granularity or few classes (e.g., binary classification, basic change detection).   
• C2: Moderate Application — Applications with moderate difficulty, such as multi-class classification or coarse semantic segmentation.   
• C3: Complex Application — Applications requiring fine-grained spatial precision, multi-class segmentation, multi-modal fusion, or high-resolution outputs.

# Evaluation Priorities (D1-D4):

•D1: Accuracy-Focused — Maximize correctness of classification or segmentation outcomes.   
• D2: Output Quality-Critical — Prioritize clean, well-bounded, and visually reliable outputs (e.g., high mIoU, sharp edges, no artifacts).   
• D3: Coverage-Critical — Ensure all relevant regions or objects are detected, even at the cost of some false positives (e.g., disaster mapping, change detection).   
• D4: Speed-Critical — Require lightweight or low-latency models for fast inference on edge devices.

Accordingly, Tab. 7 shows the full list of templates used to generate the benchmark queries. Slot values (e.g., {application}, {sensor}, {region}) are drawn from a predefined vocabulary and instantiated using sampling and LLM-based paraphrasing.

# I Expert Scoring Weight Configuration

To aggregate model evaluation scores during expert labeling, we apply a weighted linear combination of the seven criteria from Tab. 1. The weights are as follows:

<table><tr><td>Criterion</td><td>Weight (%)</td></tr><tr><td>Application Compatibility</td><td>25</td></tr><tr><td>Modality Match</td><td>20</td></tr><tr><td>Reported Performance</td><td>20</td></tr><tr><td>Efficiency</td><td>15</td></tr><tr><td>Generalizability</td><td>10</td></tr><tr><td>Popularity</td><td>5</td></tr><tr><td>Recency</td><td>5</td></tr></table>

These weights were empirically determined on the basis of expert interviews. We normalize raw scores before aggregation.

# J Illustrative Examples of Expert Scoring

To improve transparency, we provide several examples demonstrating how experts applied the scoring rubric to real model-query pairs. Each example includes: (1) the natural-language query, (2) the top-3 FM selections from all systems, and (3) the expert ratings across the seven criteria defined in Tab. 1. These examples show how rubric-guided, independent scoring yields consistent and interpretable evaluations.

Table 8: Evaluation results for queries 1 and 2. Criteria: CR1 - Application Compatibility; CR2 - Modality Match; CR3 - Reported Performance; CR4 - Efficiency; CR5 - Generalizability; CR6 - Popularity; CR7 - Recency.   

<table><tr><td>System</td><td>Rank</td><td>FM</td><td>CR1</td><td>CR2</td><td>CR3</td><td>CR4 CR5</td><td></td><td></td><td></td><td></td><td>CR6 CR7 Final Score</td></tr><tr><td colspan="10">Query 1</td></tr><tr><td></td><td>1</td><td>OmniSat</td><td>5</td><td>5</td><td>$</td><td>5</td><td></td><td></td><td>3</td><td>$\fa$</td><td>94</td></tr><tr><td>Remsa</td><td>3</td><td>FlexiMo CtxMIM</td><td>4 5</td><td>4.5 5</td><td>4.5</td><td>2.5 3</td><td>1.5 1.5</td><td>3.5 3.5</td><td></td><td></td><td>75 83.5</td></tr><tr><td></td><td>1</td><td>OmniSat</td><td>5</td><td>5</td><td>5</td><td>5</td><td>4</td><td>3</td><td></td><td></td><td>94</td></tr><tr><td>RemsA-Naive</td><td>3</td><td> FlexiMo</td><td>4</td><td>4.5</td><td></td><td>2.5</td><td>1.5</td><td>3.5</td><td></td><td></td><td>75</td></tr><tr><td></td><td></td><td>CtxMIM</td><td>5</td><td>5</td><td>4 4.5</td><td>3</td><td>1.5</td><td>3.5</td><td></td><td></td><td>83.5</td></tr><tr><td></td><td></td><td>SpectralEarth</td><td>3</td><td>3</td><td>3.5</td><td>1.5</td><td></td><td></td><td></td><td></td><td>59.5</td></tr><tr><td>DB-Retrieval</td><td>1</td><td>OmniSat</td><td>5</td><td>5</td><td>5</td><td>5</td><td></td><td>4</td><td>3 3</td><td>5 4</td><td>94</td></tr><tr><td></td><td>3</td><td>MATTER</td><td>4</td><td>4.5</td><td></td><td>4.5</td><td></td><td>3.5</td><td>1</td><td>2</td><td>75</td></tr><tr><td></td><td>1</td><td>FoMo</td><td>5</td><td>5</td><td></td><td>3.5</td><td>1.5</td><td>2</td><td>1.5</td><td></td><td>79.5</td></tr><tr><td>Unstr.-RAG</td><td>2</td><td>DynamicVis</td><td>4</td><td>4</td><td></td><td>4 3.5</td><td></td><td>3.5</td><td></td><td>5</td><td>75</td></tr><tr><td></td><td>3</td><td>SatVision-TOA</td><td>2.5</td><td></td><td></td><td>2.5</td><td></td><td>2.5</td><td>$\frac{2 }$</td><td>4</td><td>55</td></tr><tr><td colspan="10">4 Query 2</td><td></td></tr><tr><td></td><td></td><td>SSL4EO-S12</td><td>5</td><td></td><td></td><td></td><td>4.5</td><td></td><td>4.5</td><td>3</td><td>89.5</td></tr><tr><td>Remsa</td><td>1 2</td><td>Ial-SimCLR</td><td>3.5</td><td>50</td><td>4 3.5</td><td>$</td><td>2 5</td><td>3</td><td></td><td></td><td>775</td></tr><tr><td></td><td>3</td><td>SeCo</td><td>3</td><td>3</td><td>3.5</td><td></td><td></td><td></td><td>2.5</td><td>1</td><td>67</td></tr><tr><td></td><td>1</td><td>SoftCon</td><td>5</td><td>5</td><td>4.5</td><td></td><td></td><td></td><td>4</td><td>4</td><td>87</td></tr><tr><td>RemSA-Naive</td><td>2</td><td>SkySense</td><td>5</td><td>5</td><td></td><td></td><td>3.5</td><td></td><td>5</td><td>4</td><td>85.5</td></tr><tr><td></td><td>3</td><td>SSL4EO-S12</td><td>5</td><td>5</td><td></td><td></td><td>4.5</td><td></td><td>4.5</td><td>3</td><td>89.5</td></tr><tr><td></td><td>1</td><td>CACo</td><td>3</td><td>3</td><td></td><td>4 4</td><td></td><td></td><td>4</td><td>3</td><td>70</td></tr><tr><td>DB-Retrieval</td><td>2</td><td>SeECo</td><td>3</td><td>3.5</td><td></td><td>5 4</td><td>5</td><td></td><td>2.5</td><td>1</td><td>67</td></tr><tr><td></td><td>3</td><td>SSL4EO-S12</td><td>5</td><td>5</td><td></td><td></td><td>4.5</td><td></td><td>4.5</td><td>3</td><td>89.5</td></tr><tr><td></td><td></td><td>CACo</td><td>3</td><td>3</td><td></td><td></td><td></td><td>4</td><td>4</td><td>3</td><td>70</td></tr><tr><td>Unstr.-RAG</td><td></td><td>Copernicus-FM</td><td>3</td><td>3.5</td><td></td><td></td><td>3.5</td><td></td><td>5</td><td>5</td><td>62.5</td></tr><tr><td></td><td>$$</td><td>AnySat</td><td>3.5</td><td>5</td><td>3 3.5</td><td>1 1.5</td><td>4</td><td></td><td>4.5</td><td>5</td><td>74</td></tr></table>

Query: I need a model for fine-grained land cover classification using high-resolution multispecral imagery. Accuracy and spatial detail are important.

Selected FMs (Top-3 from Each System): See Tab. 8.

# Example 2:

Query: I only have a few labeled samples for urban expansion detection using Sentinel-1 and Sentinel-2 time series data from 2016-2023. I want a model that can adapt well in a few-shot setting.

elected FMs (Top-3 from Each System): See Tab. 8.

These examples illustrate how the rubric was applied in practice and how expert judgments reflect both task requirements and model capabilities. They also demonstrate how rubric-guided scoring minimizes subjective variation across annotators.