# 1. 论文基本信息

## 1.1. 标题
论文标题为 `Agent-based Video Trimming`，中文译作“基于智能体的视频修剪”。

## 1.2. 作者
论文的作者团队包括：
*   Lingfeng Yang (杨凌峰)
*   Zhenyuan Chen (陈振元)
*   Xiang Li (李翔)
*   Peiyang Jia (贾培阳)
*   Liangqu Long (龙亮曲)
*   Jian Yang (杨健)

    他们分别隶属于南京理工大学、VCIP、计算机科学、南开大学、NKIARI (深圳福田) 和 Insta360。

## 1.3. 发表期刊/会议
该论文目前作为预印本 (preprint) 发布在 `arXiv` 上，尚未正式发表在特定的期刊或会议。

## 1.4. 发表年份
论文的发布时间（UTC）是 `2024-12-12T17:59:28.000Z`，因此可以认为发表年份为 2024 年。

## 1.5. 摘要
随着信息可访问性的提高，用户生成视频的长度日益增长，这给观众带来了筛选海量内容以获取有价值见解的负担。这一趋势凸显了对高效提取关键视频信息的算法的需求。尽管高光检测、时刻检索和视频摘要等领域取得了显著进展，但现有方法主要侧重于选择特定的时间间隔，往往忽视了片段之间的相关性以及片段排列的潜力。

本文引入了一项名为 `Video Trimming (VT)`（视频修剪）的新任务，其核心在于检测无用素材、选择有价值片段并将其组合成一个具有连贯故事的最终视频。为了解决这一任务，作者提出了 `Agent-based Video Trimming (AVT)`（基于智能体的视频修剪）方法，该方法结构分为三个阶段：视频结构化（Video Structuring）、剪辑过滤（Clip Filtering）和故事组合（Story Composition）。具体而言，`AVT` 采用一个视频字幕智能体（Video Captioning Agent）将视频切片转换为结构化的文本描述；一个过滤模块（Filtering Module）根据每个剪辑的结构化信息动态地丢弃低质量素材；以及一个视频编排智能体（Video Arrangement Agent）来选择并编译有效剪辑，形成一个连贯的最终叙事。

为了评估，作者开发了一个视频评估智能体（Video Evaluation Agent）来评估修剪后的视频，并与人工评估并行进行。此外，作者还使用互联网上的原始用户视频策划了一个新的视频修剪基准数据集。结果显示，`AVT` 在用户研究中获得了更积极的评价，并在 `YouTube Highlights`、`TVSum` 和自建数据集上的高光检测任务中表现出卓越的平均精度均值 (mAP) 和精确度 (precision)。代码和模型已公开。

## 1.6. 原文链接
*   **原文链接:** https://arxiv.org/abs/2412.09513
*   **PDF 链接:** https://arxiv.org/pdf/2412.09513v1.pdf
*   **发布状态:** 预印本

# 2. 整体概括

## 2.1. 研究背景与动机
### 2.1.1. 核心问题
当前，随着智能手机和各种录制设备的普及，用户生成视频的数量和平均长度都在急剧增加。这些视频中往往包含大量冗余、无趣或低质量的素材，使得观众需要花费大量时间来筛选，才能找到有价值或精彩的内容。这给观众带来了巨大的信息过载负担。

### 2.1.2. 问题重要性
高效地从冗长视频中提取关键信息对于提升用户体验、节省观看时间以及在海量视频内容中快速定位价值至关重要。无论是个人用户剪辑日常 `vlog`，还是专业内容创作者进行初剪，一个能够自动识别并精炼视频内容的工具都具有巨大的应用潜力。

### 2.1.3. 现有研究的挑战与空白
尽管视频理解领域已经取得了显著进步，包括：
*   `Highlight Detection`（高光检测）：旨在识别视频中最精彩、最吸引人的片段。
*   `Moment Retrieval`（时刻检索）：根据用户查询，定位视频中相关的特定时刻。
*   `Video Summarization`（视频摘要）：通过提取关键帧或短片段来概括视频内容。

    然而，这些现有方法主要关注“内容提取”或“内容检索”，即从视频中挑选出符合特定标准的片段。它们普遍存在以下局限性：
1.  **忽略片段间关联性：** 多数方法独立评估每个片段的价值，没有考虑不同片段之间的逻辑关系、时间顺序或叙事连贯性。简单地拼接高光片段可能导致视频支离破碎，缺乏故事性。
2.  **忽视无用素材过滤：** 现有方法主要关注“选择好的”，但对于“剔除坏的”（如抖动、过曝、无意义的镜头）关注不足。这导致即使是高光片段也可能混杂着低质量的画面。
3.  **缺乏故事组合能力：** 简单地按照得分高低或时间顺序拼接片段，往往无法形成一个具有“开始、发展、结局”完整结构的连贯叙事。

### 2.1.4. 论文的切入点与创新思路
为了弥补上述空白，本文首次提出了 `Video Trimming (VT)`（视频修剪）这一新任务。`VT` 不仅仅是选择高光片段，更是一个端到端的视频编辑过程，它包含了：
1.  **检测无用素材：** 识别并剔除低质量或无关的视频片段。
2.  **选择有价值片段：** 筛选出内容丰富、有吸引力的核心片段。
3.  **组合连贯故事：** 将选定的片段逻辑地编排成一个具有完整叙事结构的最终视频。

    为解决 `VT` 任务，论文提出了 `Agent-based Video Trimming (AVT)`（基于智能体的视频修剪）算法。`AVT` 利用多模态大语言模型（`MLLMs`）的强大能力，将其作为无需训练的智能体来执行视频理解和内容编排。

## 2.2. 核心贡献/主要发现
本文的主要贡献可以总结如下：
1.  <strong>首次提出 `Video Trimming (VT)` 任务：</strong> 引入了一个新颖的视频处理任务，超越了传统的高光检测和视频摘要，专注于从长视频中提取关键意图，并生成具有连贯故事情节的精简视频。
2.  **提出 `AVT` 算法作为基线：** 提出了一种基于智能体的视频修剪算法 `AVT`。该算法将视频内容转换为结构化描述，过滤掉无用片段，并编排选定片段以形成连贯的最终叙事视频。`AVT` 分为三个关键阶段：
    *   <strong>视频结构化 (Video Structuring)：</strong> 使用 `Video Captioning Agent` 将视频切片转换为结构化文本描述，包括原始字幕、上下文属性和缺陷属性。
    *   <strong>剪辑过滤 (Clip Filtering)：</strong> 采用动态过滤模块，根据缺陷属性和高光分数，智能地丢弃低质量或无意义的素材。
    *   <strong>故事组合 (Story Composition)：</strong> 通过 `Video Arrangement Agent` 将筛选后的有效片段编排成一个具有完整叙事（开始、发展、结局）的最终视频。
3.  <strong>设计 <code>Video Evaluation Agent</code>：</strong> 开发了一个基于 `LLM` 的视频评估智能体，用于自动评估修剪后视频的质量，并与人工评估结果保持一致。
4.  **构建新视频修剪基准数据集：** 策划了一个包含来自互联网的原始用户视频的新基准数据集，并进行了无用素材和高光标签的标注。
5.  **卓越的实验性能：**
    *   在用户研究中，`AVT` 获得了更积极的评价，表明其生成视频的质量、吸引力、精彩程度和无用素材量都优于现有方法。
    *   在 `YouTube Highlights`、`TVSum` 和自建数据集上的高光检测任务中，`AVT` 在零样本 (zero-shot) 设置下，在平均精度均值 (mAP) 和精确度 (precision) 方面表现出卓越的性能，甚至可与部分有监督方法媲美。

# 3. 预备知识与相关工作

本部分旨在为读者理解论文核心思想提供必要的背景知识，并阐述 `AVT` 在现有研究脉络中的定位和创新点。

## 3.1. 基础概念
*   <strong>视频修剪 (Video Trimming, VT)：</strong> 本文提出的一项新任务。它不仅仅是简单地从视频中提取精彩片段，更是一个综合过程，包括识别并剔除无用素材 (`wasted footage`)、选择有价值的片段 (`valuable segments`)，并最终将这些片段逻辑地组合成一个具有连贯故事情节 (`coherent story`) 的最终视频。
*   <strong>高光检测 (Highlight Detection)：</strong> 视频理解领域的一个经典任务，目标是识别视频中那些最吸引人、最精彩的时刻或片段。通常通过预测每个片段的“显著性分数” (`saliency scores`) 来实现。
*   <strong>时刻检索 (Moment Retrieval)：</strong> 另一个视频理解任务，旨在根据用户的自然语言查询（如“找出狗在公园里玩耍的时刻”）在视频中定位精确的开始和结束时间点。
*   <strong>视频摘要 (Video Summarization)：</strong> 旨在将冗长视频浓缩成一个短小精悍的概要视频或一系列关键帧，以捕获视频的主要主题或事件。可以是通用摘要（不依赖查询）或基于查询的摘要。
*   <strong>多模态大语言模型 (Multimodal Large Language Models, MLLMs)：</strong> `LLMs` 的扩展，能够处理和理解多种类型的数据，例如文本、图像和视频。它们结合了视觉编码器和语言模型，使其能够进行图像/视频描述、视觉问答、跨模态推理等任务。本文中使用的 `GPT-4o`、`Gemini` 等模型即属于 `MLLMs`。
*   <strong>智能体 (Agent)：</strong> 在 `AI` 领域，智能体是一个能够感知环境、进行推理和执行动作的实体。在本文中，`MLLMs` 被用作智能体，通过接收特定的指令和信息（如视频帧、文本描述）来执行视频字幕、过滤和故事编排等复杂任务。这里的智能体通常指“无需训练的智能体” (`training-free agents`)，即通过精心设计的提示词 (`prompting pipeline`) 来引导预训练 `MLLMs` 完成任务，而非对其进行额外的模型训练。
*   <strong>思维链 (Chain of Thought, CoT)：</strong> 一种`LLM`提示策略，通过在提示中加入“让我们一步步思考”等引导性语句，鼓励`LLM`在给出最终答案之前，先生成一系列中间推理步骤。这有助于`LLM`解决复杂问题，提高回答的准确性和可解释性。

## 3.2. 前人工作
论文在 `2. Related Work` 部分对相关工作进行了详细梳理，主要分为以下几类：

### 3.2.1. 视频理解 (Video Understanding)
该领域的研究利用 `MLLMs` 推动视频理解的进步。现有方法涵盖或结合了多项任务：
*   <strong>视频问答 (Video Question Answering, VQA)：</strong> `[24, 31, 50, 60, 71]` 等模型致力于通过视频内容回答用户提出的问题。
*   <strong>长视频理解 (Long Video Understanding)：</strong> `[54, 54]` 和 `VTimeLLM [19]` 等模型专注于处理和理解长时间的视频内容，通常涉及分段和关键时刻识别。
*   <strong>时刻定位 (Moment Localization)：</strong> `[67]` 等研究旨在视频中精确地定位特定事件或动作的发生时刻。
*   **大规模视频-文本模型：** `InternVid [51]` 通过对比学习和多阶段预训练构建大规模视频-文本模型；`InternVideo [50, 52]` 利用多模态数据扩展视频理解能力。
*   **指令遵循和聊天式理解：** `LaViLa [71]` 和 `Valley [31]` 通过微调改进基于视频的指令理解；`Merlin [66]` 和 `MovieChat [40]` 增强视频问答和长视频理解；`Vid2Seq [60]` 和 `VideoChat [24]` 实现以聊天为中心的视频理解。
*   **机器人与具身智能：** `PaLM-E [11]` 将真实世界感知数据融入语言模型。

    **本文与这些工作的差异：** 虽然 `AVT` 利用了基础模型作为组件，但它并非针对通用的视频理解任务，而是专门解决视频修剪这一特定任务。

### 3.2.2. 视频智能体 (Video Agent)
现有视频智能体方法主要分为两类：
1.  **结合 `LLMs` 与外部工具/执行器：** 这类方法利用 `LLMs` 作为核心控制器，协调外部工具和代码执行器来完成复杂任务。例如，`DoraemonGPT [63]` 使用符号记忆增强 `VQA`；`InternGPT [30]` 通过交互式查询改进推理；`MM-ReAct [62]` 将 `REACT [64]` 机制扩展到多模态任务；`Video ChatCaptioner [8]` 通过多智能体迭代查询强化视频理解。
2.  **将视频内容转换为文本语料库进行分析：** 这类方法将视频内容转化为文本形式，以便 `LLMs` 进行后续分析和处理。例如，`AssistGPT [14]` 通过规划、执行、检查和学习循环改进 `VQA` 和时刻检索；`ChatVideo [46]` 将视频内容结构化为文本数据库以进行高效查询；`LLoVi [68]` 专注于使用字幕进行细粒度 `VQA` 和间隔检索；`MM-Vid [26]` 将多模态信息视为文本数据；`VideoAgent [49]` 通过迭代查询和相似性匹配改进时刻检索。

    **本文与这些工作的差异：** 第一类视频智能体不适用于视频修剪，因为它们缺乏专门的工具来解决该任务。第二类视频智能体虽然可以根据查询提取片段，但往往忽略全局内容，从而损害视频的连贯性。本文的方法是首个通过创新性视频处理流水线（包含视频智能体系统）来解决视频修剪挑战的工作。

### 3.2.3. 视频时间定位 (Video Temporal Grounding)
该任务旨在以连续间隔或离散关键帧的形式，从视频中定位目标片段，包括：
*   <strong>高光检测 (Highlight Detection, HD) [5, 6, 17, 41, 43, 58]：</strong> 预测显著性分数以提取高光片段。
    *   **差异：** 这些方法缺乏连贯视频修剪所需的时间上下文和事件关系。
*   <strong>时刻检索 (Moment Retrieval, MR) [3, 15, 18, 21, 34]：</strong> 根据给定查询选择时刻。数据集如 `DiDeMo [3]`、`ActivityNet Caption [21]` 和 `Charades-STA [15]` 提供区域字幕。方法如 `Moment-DETR [22]`、`QD-DETR [35]`、`TR-DETR [42]`、`UniVTG [27]` 和 `UVCOM [57]` 旨在通过互惠模块设计同时解决时刻和显著性预测。
    *   **差异：** 检索到的片段通常缺乏全面的视频覆盖和上下文，且需要事先的用户查询。
*   <strong>视频摘要 (Video Summarization) [16, 20, 33, 41]：</strong> 通过选择最能代表原始视频内容的关键镜头来凝练视频。可以是通用摘要或基于查询的摘要。
    *   **差异：** 摘要虽然浓缩了视频，但选定的片段是离散的，无法创建连贯、可观看的视频。

        **本文与这些工作的差异：** 以上所有视频时间定位任务都只关注片段选择，常常忽视叙事流。本文提出的视频修剪任务则强调片段选择和组合，在缩短视频时长的同时保持叙事完整性。

## 3.3. 技术演进
视频理解领域从最初依赖手工特征和浅层模型，逐步发展到基于深度学习的端到端系统。随着 Transformer 架构和大规模预训练的兴起，多模态大语言模型（`MLLMs`）展现出强大的跨模态理解和生成能力，成为处理复杂视觉-语言任务的新范式。`Agent` 范式的引入，进一步提升了 `MLLMs` 解决多步骤、多工具复杂任务的能力。

本文的工作正处于这一技术演进的前沿，利用 `MLLMs` 作为智能体，将视频内容从原始像素层面提升到语义层面，并在此基础上进行高级推理和决策（如过滤缺陷、编排故事），从而实现了比以往仅关注特征提取和单一任务（如高光检测）更复杂的视频内容创作任务——视频修剪。

## 3.4. 差异化分析
`AVT` 与现有方法的根本区别在于：
*   **任务定义上的扩展：** `AVT` 提出了 `Video Trimming` 任务，这不仅是片段的“检索”或“提取”，更是“过滤”和“组合”的过程，旨在生成一个具有完整叙事结构的最终视频，这是现有任务未涵盖的。
*   **引入缺陷过滤：** `AVT` 明确地引入了对视频缺陷属性（如遮挡、抖动、过曝、无意义内容）的检测和动态过滤机制，这使得生成的视频质量更高，观感更好，是现有高光检测和摘要方法所不具备的。
*   **强调叙事连贯性：** `AVT` 的 `Story Composition` 阶段使用 `Video Arrangement Agent` 来编排片段，确保最终视频具有“开始、发展、结局”的连贯故事线，而非简单拼接高分片段。
*   **智能体驱动的端到端流程：** 整个 `AVT` 流程由 `MLLMs` 驱动的智能体完成，从视频结构化到剪辑过滤再到故事组合，形成了一个无需额外训练的端到端系统，展现了 `MLLMs` 在复杂视频编辑任务中的潜力。

# 4. 方法论

本节将详细阐述 `Agent-based Video Trimming (AVT)` 的方法原理及其三个核心阶段：视频结构化、剪辑过滤和故事组合，并介绍用于评估最终视频的智能体。`AVT` 的核心思想是利用多模态大语言模型 (`MLLMs`) 的强大理解和推理能力，将视频内容转化为结构化文本，然后基于这些文本信息进行智能的片段筛选和叙事编排，最终生成一个连贯且高质量的精简视频。

## 4.1. 方法原理
`AVT` 方法的核心原理是将视频处理提升到语义和叙事层面。它通过 `MLLMs` 将原始视频的视觉信息转换为丰富的文本描述，这些描述不仅包含内容梗概，还包括对视频质量（缺陷）和精彩程度的量化评估。这种文本化的表示使得后续的过滤和故事编排过程能够以更高层次的语义进行，摆脱了对原始视觉数据的直接处理，从而提高效率并赋予更大的灵活性。通过动态过滤机制，`AVT` 能够智能地平衡内容价值与拍摄质量。最后，利用 `MLLMs` 的强大叙事能力，将筛选出的片段组织成一个有逻辑、有故事感的最终视频。

## 4.2. 核心方法详解 (逐层深入)
`AVT` 算法分为三个关键阶段：`Video Structuring` (视频结构化)、`Clip Filtering` (剪辑过滤) 和 `Story Composition` (故事组合)。此外，论文还提出了一个 `Video Evaluation Agent` (视频评估智能体) 来评估最终视频。

### 4.2.1. 视频结构化 (Video Structuring)
该阶段旨在将原始视频内容转化为结构化的文本描述，以便后续的语义分析和处理。
*   **视频分割：** 原始视频被分割成一系列短小的片段 (`clips`)。每个片段默认时长为 3 秒。
*   **帧采样：** 对于每个片段，以每秒一帧的速率采样关键帧作为视觉输入。
*   **`Video Captioning Agent` 的作用：** 使用 `MLLMs`（如 `GPT-4o` 或 `Gemini`）作为 `Video Captioning Agent`（视频字幕智能体）。该智能体接收采样帧作为视觉输入，并为每个片段生成以下结构化文本信息：
    1.  `Raw Caption`（原始字幕）：对视频片段内容的通用描述，简洁明了，不超过两句话。
    2.  `Contextual Attributes`（上下文属性）：从四个维度总结视频内容，提供简短的洞察：
        *   `What` (什么)：描述场景中发生的主要动作或事件。
        *   `Where` (哪里)：指示场景的地点或环境。
        *   `When` (何时)：确定场景发生的时间（如白天、夜晚、季节）。
        *   `Who` (谁)：识别潜在的人物或主体。
    3.  `Defect Attributes`（缺陷属性）：评估每个片段的拍摄质量。作者识别了四种主要缺陷，并期望智能体返回一个 0 到 1 之间的浮点值，表示缺陷的程度（0表示无缺陷，1表示缺陷绝对可靠）。这些缺陷包括：
        *   `Occlusion`（遮挡）：目标被障碍物遮挡。
        *   `Jittering`（抖动）：相机抖动过大。
        *   `Overexposure`（过曝）：画面亮度过高。
        *   `Meaningless Content`（无意义内容）：指简单的转场、空镜头、长时间静态帧或缺乏实质信息的场景。
    4.  `Highlight`（高光）属性：衡量每个片段的整体精彩程度或兴奋水平，返回一个 0 到 1 之间的浮点值。

*   **处理优势：** 一旦视频被处理成文本，后续操作将独立于视觉内容进行，从而显著提高处理速度并降低计算成本。

    以下是 `Video Captioning Agent` 输出的缺陷和高光属性示例格式：
`"[Occlusion]: 0.8; [Jittering]: 0.7; [Overexposure]: 0.0; [Meaningless]: 0.0; [Highlight]: 0.9;"`

下图（原文 Figure 2(a)）展示了视频结构化阶段将视频片段转换为结构化文本描述的过程：

![该图像是示意图，展示了Agent-based Video Trimming的三个阶段：视频结构化（Video Structuring）、剪辑过滤（Clip Filtering）和故事组成（Story Composition）。左侧显示了视频片段及其上下文属性；中间部分展示了垃圾片段的过滤结果；右侧列出了最终视频的组成。通过这一流程，AVT能有效地从冗长视频中提取出有价值的内容，并组合成连贯的故事。](images/2.jpg)
*该图像是示意图，展示了Agent-based Video Trimming的三个阶段：视频结构化（Video Structuring）、剪辑过滤（Clip Filtering）和故事组成（Story Composition）。左侧显示了视频片段及其上下文属性；中间部分展示了垃圾片段的过滤结果；右侧列出了最终视频的组成。通过这一流程，AVT能有效地从冗长视频中提取出有价值的内容，并组合成连贯的故事。*

*   **图片描述：** 左侧部分（a）展示了 `Video Structuring` 阶段，通过 `Video Captioning Agent` 将视频片段转换为结构化文本描述，包括原始字幕、上下文属性和缺陷属性。例如，一个视频片段被描述为“一个女人在城市天际线下介绍她的 `vlog`，然后继续在厨房里说话。”同时，还提取了上下文属性（`What`, `Where`, `When`, `Who`）以及缺陷属性（`Occlusion`, `Jittering`, `Overexposure`, `Meaningless`）和高光分数。

### 4.2.2. 剪辑过滤 (Clip Filtering)
此阶段的目标是根据视频结构化阶段生成的信息，动态地筛选出有用的片段并丢弃低质量或无意义的素材。
*   **输入：** `Video Captioning Agent` 输出的缺陷属性分数和高光分数。
*   <strong>`Dynamic Filter`（动态过滤器）：</strong> 论文引入了一个 `Dynamic Filter` 机制。与简单地根据缺陷分数直接剔除片段不同，该机制通过平衡积极指标（`Highlight` 高光分数）和消极指标（`Defect Attributes` 缺陷分数）来做出决策。
*   **过滤逻辑：** 一个片段只有当其 `Highlight` (高光) 分数**超过所有**缺陷分数时，才会被认为是有效片段并被选中用于最终组合。
    *   **直觉：** 这种策略旨在确保算法更关注视频内容本身的精彩程度，而不是次要的拍摄缺陷。例如，在第一人称视角拍摄的体育视频中，相机抖动是不可避免的，如果简单地剔除所有抖动的片段，可能会丢失大量有价值的精彩内容。动态过滤器允许在内容足够精彩时，容忍一定程度的拍摄缺陷。

        以下是 `Dynamic Filtering Algorithm`（动态过滤算法）的伪代码：

$$
Algorithm 1 Dynamic Filtering Algorithm
Input: List of Keys keys, List of Numbers nums
Output:(filter_flag, highlightflag, highlight_score)
1: Initialize $s c o r e \gets 0$ $m a x \_ k e y \gets ]$ None
2: for each (key, num) in zip(keys, nums) do
3: if $n u m \ge s c o r e$ then
4: $s c o r e \gets n u m$
5: $m a x _ { - } k e y \gets k e y$
6: end if
7: end for
8: if max_key `=` [Highlight] then
9: return (False, True, score)
10: else
11: return (score = 0, False, score)
12:end if
$$
*   **符号解释：**
    *   `keys`：包含属性名称的列表，例如 `[Occlusion, Jittering, Overexposure, Meaningless, Highlight]`。
    *   `nums`：与 `keys` 中属性名称对应的分数列表。
    *   `score`：在遍历过程中，用于存储当前找到的最高分数。初始值为 0。
    *   `max_key`：在遍历过程中，用于存储拥有最高分数的属性名称。初始值为 `None`。
    *   `zip(keys, nums)`：将 `keys` 和 `nums` 列表中的元素配对，进行并行迭代。
    *   $if num >= score then ... end if$: 循环迭代每个属性及其分数。如果当前分数 `num` 大于或等于 `score`，则更新 `score` 为 `num`，并更新 `max_key` 为当前属性 `key`。
    *   `if max_key = [Highlight] then ... else ... end if`: 循环结束后，检查 `max_key` 是否是 `[Highlight]`。
        *   如果 `max_key` 是 `[Highlight]`，表示 `Highlight` 分数是所有属性中的最高分。此时，函数返回 `(False, True, score)`，其中 `filter_flag` 为 `False` 表示该片段不应被过滤，`highlightflag` 为 `True` 表示最高分是高光分数，`score` 为该高光分数。
        *   如果 `max_key` 不是 `[Highlight]`，表示某个缺陷属性的分数是所有属性中的最高分。此时，函数返回 $(score = 0, False, score)$。这里 $score = 0$ 的判断（原文中$score = 0$可能表示过滤的条件，或者是一个布尔值判断`score`是否为0）需要结合上下文理解，但核心思想是 `filter_flag` 为 `True` 表示该片段应被过滤（因为最高分来自缺陷）。根据论文描述“一个片段只有当其‘Highlight’分数超过所有缺陷分数时，才会被选为有效片段”，则 `filter_flag` 应该是 `True`。原文算法中 $(score = 0, False, score)$ 这一行 $score = 0$ 的用法略有歧义，但考虑到其在 `else` 分支，通常意味着该片段不满足高光条件，应被过滤。

*   **输出：** `filter_flag`（是否过滤的标志），`highlightflag`（最高分是否为高光分数的标志），以及 `highlight_score`（如果最高分是高光分数，则为该分数）。

    下图（原文 Figure 3）通过关键帧示例展示了动态过滤器的应用：

    ![Figure 3. Keyframes from a mountain biking video. Clips marked with red boxes are discarded due to higher defect scores, while clips with green boxes are selected despite minor shaking, as they highlight the dynamic scene of cycling on a mountain path.](images/3.jpg)
    *该图像是一个视频关键帧的集合，展示了山地自行车骑行的场景。红框内的片段因缺陷评分较高而被淘汰，绿色框内的片段尽管有轻微的抖动，但仍被选中，展示了骑行动态。每个片段的定义质量指标包括遮挡、抖动、过曝光、无意义以及高亮度评分。*

*   **图片描述：** 图中展示了山地自行车视频的关键帧。红色方框标记的片段因缺陷分数较高而被丢弃（如镜头模糊或晃动严重），而绿色方框标记的片段尽管有轻微抖动，但因其内容精彩（展现了山地骑行的动态场景）且高光分数超过缺陷分数而被选中。这体现了 `Dynamic Filter` 在内容丰富度和拍摄缺陷之间的平衡作用。

### 4.2.3. 故事组合 (Story Composition)
此阶段负责将过滤后的有效片段组织成一个具有连贯叙事结构的最终视频。
*   <strong>`Video Arrangement Agent`（视频编排智能体）：</strong> 同样使用 `MLLMs`。该智能体接收过滤阶段输出的有效片段信息。
*   <strong>用户提示 (User Prompt, $P$)：</strong> 为了引导智能体进行故事编排，作者设计了一个详细的用户提示，其中包含：
    *   任务介绍：将智能体设定为“专业视频剪辑师”，目标是从一系列视频片段描述中选择合适的片段，并组合成一个具有完整故事（开始、发展、结局）的视频。
    *   `Chain of Thought (CoT)`（思维链）指令：指导智能体生成视频组合步骤，包括：
        1.  全局理解 (`General Comprehension`)：对所有输入片段进行整体理解，识别主题。
        2.  片段选择 (`Clip Selection`)：根据主题和叙事结构（开始、发展、结局）选择片段。
        3.  组合安排 (`Composition Arrangement`)：将选定片段组织成逻辑顺序。
*   <strong>用户输入 (User Input, $I$)：</strong> 智能体接收的输入 $I$ 由所有有效片段的结构化信息拼接而成。其格式如下：
    $$
    \begin{array} { c } { { I = \{ \{ C l i p I D \} _ { k } , \{ H i g h l i g h t F l a g \} _ { k } ( \{ S c o r e \} _ { k } ) , } } \\ { { \{ C o n t e x t u a l A t t r i b u t e s \} _ { k } , } } \\ { { \{ R a w C a p t i o n \} _ { k } \} \mid _ { C l i p - k } \{ C _ { 1 } \sim C _ { M } \} , } } \end{array}
    $$
    *   **符号解释：**
        *   $I$: 传递给 `Video Arrangement Agent` 的用户输入。
        *   $\{Clip ID\}_k$: 第 $k$ 个片段的唯一标识符。
        *   $\{Highlight Flag\}_k$: 第 $k$ 个片段的高光标志（布尔值），表明该片段是否被认为是高光。
        *   $(\{Score\}_k)$: 第 $k$ 个片段对应的高光分数（浮点值）。
        *   $\{Contextual Attributes\}_k$: 第 $k$ 个片段的上下文属性（`What`, `Where`, `When`, `Who`）。
        *   $\{Raw Caption\}_k$: 第 $k$ 个片段的原始字幕。
        *   $C_1 \sim C_M$: 表示从第一个到第 $M$ 个有效片段。这些信息都是从之前的过滤和结构化阶段获得的。

*   **智能体输出：** 智能体根据提示 $P$ 和输入 $I$ 生成一个输出故事情节，其中包含一个首选的片段序列，即复合片段 $C^t$，以及组织这些片段的叙事和推理。
*   **迭代与并行：**
    *   为了处理过长的视频和避免 `LLMs` 在长上下文中的局限性（容易分心），初始阶段会并行分组处理片段。
    *   随着片段数量减少，所有信息会被整合到最终的组合中。
*   **最终视频生成：** 将最终选定并排序的片段索引映射回其原始视频时长，然后组装成最终视频。值得注意的是，并非所有有效片段都会被选中，且片段顺序不严格遵循时间顺序，而是根据智能体编排的故事情节进行排列。

    下图（原文 Figure 4 和 Figure 5）展示了故事组合阶段的 `User Prompt` 和 `Agent Output` 示例：

    ![该图像是一个示意图，展示了视频剪辑过程的三个主要阶段：剪辑选择、关键瞬间强调和视频构成。图示中强调了选取高亮剪辑以提供连续故事的重点，包括开始、发展和结束的结构。这个过程确保每个片段都清晰明了，最终产生一个引人入胜且连贯的视频。](images/4.jpg)
    *该图像是一个示意图，展示了视频剪辑过程的三个主要阶段：剪辑选择、关键瞬间强调和视频构成。图示中强调了选取高亮剪辑以提供连续故事的重点，包括开始、发展和结束的结构。这个过程确保每个片段都清晰明了，最终产生一个引人入胜且连贯的视频。*

    ![该图像是一个示意图，展示了在视频修剪过程中，针对片段的ID安排如何通过Agent进行处理。图中展示了输入片段与经过处理的片段之间的关系，突显了片段的分组和安排过程。](images/5.jpg)
    *该图像是一个示意图，展示了在视频修剪过程中，针对片段的ID安排如何通过Agent进行处理。图中展示了输入片段与经过处理的片段之间的关系，突显了片段的分组和安排过程。*

*   **图片描述：** `Figure 4` 展示了 `User Prompt` 的结构，包括任务介绍、组成步骤（通用理解、剪辑信息）以及对智能体的指令。`Figure 5` 展示了 `Agent` 根据输入生成的 `Overall comprehension`（整体理解）和分段的故事情节选择，例如：开始部分选择 `Clip ID 2`，发展部分选择一系列片段，结局部分选择 `Clip ID 129`。这体现了智能体如何将多个片段组织成一个连贯的叙事。

### 4.2.4. 最终剪辑评估 (Final Cut Evaluation)
为了客观评估修剪后视频的质量，作者设计了一个 `Video Evaluation Agent`（视频评估智能体）。
*   **评估原理：** 借鉴 `G-Eval [29]` 的思想，利用 `LLM` 作为评估器来评估生成视频的质量。
*   **评估标准：** 定义了一系列 1 到 5 分的评估标准，并结合 `CoT` 指令来提高评估精度。这些标准包括：
    *   `Material Richness`（内容丰富度）：评估视频内容的广度、多样性和叙事连贯性。
    *   `Appeal`（吸引力）：衡量视频的吸引力、长度和娱乐性。
    *   `Exciting Segments`（精彩片段）：评估高光片段的质量和出现频率。
    *   `Amount of Wasted Footage`（无用素材量）：评估视频中无关内容的数量，分数越高表示干扰越少，观看体验越好。
*   **输入与输出：** `Video Evaluation Agent` 仅使用视频内容作为输入（通过其多模态能力），并输出每个指标的得分及理由。
*   **评分示例：** $"[Material Richness]: {Reason} (2.5); [Appeal]: {Reason} (3.0); [Exciting Segments]: {Reason} (3.5); [Amount of Wasted Footage]: {Reason} (2.0);"$
*   **最终评分：** 对所有指标得分取平均值，得到视频剪辑的最终评分。

# 5. 实验设置

## 5.1. 数据集
实验使用了三个数据集来评估 `AVT` 的性能。

### 5.1.1. 现有数据集 (Existing Dataset)
*   **YouTube Highlights [43]:** 这是一个用于评估视频时间定位任务的知名基准数据集。主要用于高光检测。
    *   **特点:** 包含 423 个视频，来自 5 个用户，内容为网络视频。标注类型为帧级别分数。查询类型为标题。视频时长范围为 7 秒到 1483 秒，平均 102 秒。
*   **TVSum [41]:** 另一个用于评估视频时间定位任务的知名基准数据集。
    *   **特点:** 包含 50 个视频，来自 20 个用户，内容为网络视频。标注类型为帧级别分数。查询类型为标题。视频时长范围为 83 秒到 647 秒，平均 235 秒。
*   **实验分割:** 遵循 [27, 28] 的划分方法，在 `YouTube Highlights` 和 `TVSum` 数据集的 `20%` 数据上进行测试。对于零样本设置下的 `TVSum`，为了避免验证集过小导致分数不稳定，评估了多达 50 个视频。

### 5.1.2. 视频修剪数据集 (Video Trimming Dataset)
作者为了 `Video Trimming` 任务专门构建了一个新的基准数据集。
*   **数据来源:** 从 `YouTube` 上爬取的原始用户视频。
*   **收集原则:**
    1.  **原始未编辑视频:** 强调使用原始的、未经编辑的视频，这些视频通常包含真实世界拍摄条件下的缺陷（如遮挡、抖动、过曝），与现有数据集多为已编辑视频形成对比。
    2.  **长视频：** 视频时长需超过 5 分钟，而非短时、预剪辑的蒙太奇。
    3.  **多源视频：** 算法可能需要从多个源视频中合成最终剪辑，以模拟实际应用场景，而现有数据集通常每个主题只关注一个视频。
*   **规模与内容:**
    *   总计 30 个主题，包含 42 个视频，由 30 位不同的用户上传。
    *   每个视频平均长度约为 10 分钟。
    *   视频类型分为三类：`daily life`（日常生活）、`sports`（体育）和 `travel vlogs`（旅行 `vlog`），确保了修剪场景的多样性。
*   **标注:**
    *   每个视频由 10 位标注者进行标注。
    *   采用四级评分评估素材质量：
        *   `0`：`wasted`（无用素材）
        *   `1`：`ambiguous`（模糊/不确定）
        *   `2`：`normal`（正常）
        *   `3`：`highlight footage`（高光素材）
*   **示例:** 论文未在正文提供具体视频样本，但提供了详细的分类和 `YouTube ID` 列表。

    以下是原文 Table S2 比较的现有数据集与自建视频修剪数据集的详细信息：

    <table>
    <thead>
    <tr>
    <td>Dataset</td>
    <td>#Video</td>
    <td>#User</td>
    <td>Content</td>
    <td>Annotation type</td>
    <td>Query</td>
    <td>Duration (Min, Max, Avg)</td>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>YouTube Highlights [43]</td>
    <td>423</td>
    <td>5</td>
    <td>Web videos</td>
    <td>Frame-level scores</td>
    <td>Title</td>
    <td>7s, 1483s, 102s</td>
    </tr>
    <tr>
    <td>SumMe [16]</td>
    <td>25</td>
    <td>15~18</td>
    <td>User-generated videos</td>
    <td>Frame-level scores</td>
    <td>N/A</td>
    <td>32s, 324s, 146s</td>
    </tr>
    <tr>
    <td>TVSum [41]</td>
    <td>50</td>
    <td>20</td>
    <td>Web videos</td>
    <td>Frame-level scores</td>
    <td>Title</td>
    <td>83s, 647s, 235s</td>
    </tr>
    <tr>
    <td>Charades-STA [15]</td>
    <td>9,848</td>
    <td>267</td>
    <td>Web videos</td>
    <td>Time intervals</td>
    <td>Local caption</td>
    <td>2s, 194s, 30s</td>
    </tr>
    <tr>
    <td>OVP [10]</td>
    <td>50</td>
    <td>5</td>
    <td>Various genre videos</td>
    <td>Time intervals</td>
    <td>N/A</td>
    <td>83s, 647s, 235s</td>
    </tr>
    <tr>
    <td>YouTube [10]</td>
    <td>39</td>
    <td>5</td>
    <td>Web videos</td>
    <td>Time intervals</td>
    <td>N/A</td>
    <td>83s, 647s, 235s</td>
    </tr>
    <tr>
    <td>Video Trimming (ours)</td>
    <td>42</td>
    <td>10</td>
    <td>Web videos</td>
    <td>Frame-level scores</td>
    <td>N/A</td>
    <td>141s, 1483s, 556s</td>
    </tr>
    </tbody>
    </table>

*   **表格描述：** 该表格对比了 `YouTube Highlights`、`SumMe`、`TVSum`、`Charades-STA`、`OVP`、`YouTube` 等现有数据集与本文构建的 `Video Trimming (ours)` 数据集。突出显示了自建数据集的特点，如平均视频时长最长（556秒，约9.2分钟），更适合长视频修剪任务。

## 5.2. 评估指标
论文使用了多种评估指标来衡量 `AVT` 在不同任务中的性能。

### 5.2.1. 高光检测任务评估指标
*   <strong>平均精度均值 (Mean Average Precision, mAP)：</strong>
    *   **概念定义：** `mAP` 是在信息检索、目标检测和视频高光检测等领域广泛使用的评估指标，用于衡量系统在检索或检测相关内容方面的整体性能。它结合了精度 (Precision) 和召回率 (Recall)。对于视频高光检测，`mAP` 评估模型在不同置信度阈值下，识别出所有真实高光片段的准确程度。一个高 `mAP` 值意味着模型能够以高精度找到大部分相关片段。
    *   **数学公式：**
        单个查询或类别的平均精度 (AP) 计算公式：
        $$
        \mathrm{AP} = \sum_{k=1}^{n} P(k) \Delta r(k)
        $$
        平均精度均值 (mAP) 计算公式：
        $$
        \mathrm{mAP} = \frac{1}{Q} \sum_{q=1}^{Q} \mathrm{AP}_q
        $$
    *   **符号解释：**
        *   $\mathrm{AP}$: 单个查询（例如，一个视频或一个特定主题）的平均精度。
        *   $k$: 排名列表中的位置（例如，模型预测的第 $k$ 个高光片段）。
        *   $n$: 排名列表中的总项目数。
        *   `P(k)`: 在位置 $k$ 时的精度。其计算方式为：在排名前 $k$ 个结果中，有多少是正确的（真阳性）除以 $k$。
        *   $\Delta r(k)$: 召回率从位置 `k-1` 到 $k$ 的变化量。通常当在位置 $k$ 找到一个相关项时，召回率会增加，此时 $\Delta r(k)$ 为非零值。
        *   $\mathrm{mAP}$: 所有查询或类别的平均精度均值。
        *   $Q$: 查询或类别的总数。
        *   $\mathrm{AP}_q$: 第 $q$ 个查询或类别的平均精度。

*   <strong>精确度 (Precision)：</strong>
    *   **概念定义：** 精确度衡量的是模型识别出的正样本（例如，预测为高光片段）中有多少比例是真正的正样本。它关注的是模型做出的正向预测的准确性。在高光检测中，高精确度意味着模型预测为高光的片段中，绝大部分确实是高光，误报率低。
    *   **数学公式：**
        $$
        \mathrm{Precision} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}}
        $$
    *   **符号解释：**
        *   $\mathrm{TP}$ (True Positive, 真阳性)：模型正确地将高光片段预测为高光的数量。
        *   $\mathrm{FP}$ (False Positive, 假阳性)：模型错误地将非高光片段预测为高光的数量。

*   **Top-5 mAP：**
    *   **概念定义：** 这是 `TVSum` 数据集上使用的特定评估指标。它意味着在计算 `mAP` 时，只考虑每个视频摘要中的前 5 个高光片段。这在视频摘要任务中尤其有意义，因为用户通常只关心视频中最精彩的几个部分。

### 5.2.2. 视频修剪任务评估指标 (用户研究和智能体评估)
这些指标用于评估最终修剪视频的质量，范围从 1 到 10（用户研究）或 1 到 5（智能体评估），分数越高表示表现越好。
*   <strong>内容丰富度 (Material Richness)：</strong> 评估视频内容的广度、多样性以及叙事连贯性。高分表示视频涵盖了广泛的主题，并以流畅的叙述呈现。
*   <strong>吸引力 (Appeal)：</strong> 衡量视频的整体吸引力、是否引人入胜、长度是否合适以及娱乐性。高分表示视频能够吸引并保持观众的注意力。
*   <strong>精彩片段 (Exciting Segments)：</strong> 评估视频中高光片段的质量和出现频率。高分表示视频包含许多高质量的精彩瞬间。
*   <strong>无用素材量 (Amount of Wasted Footage)：</strong> 评估视频中无关内容或低质量内容的数量。分数越高，表示无用内容越少，视频更精炼，观看体验更好。
*   <strong>整体感知 (Overall)：</strong> 用户对视频质量的综合评价。

### 5.2.3. 评估智能体与人类偏好相关性指标
为了验证 `Video Evaluation Agent` 的可靠性，作者使用了三个元评估指标来衡量其与人类偏好的一致性。
*   <strong>皮尔逊相关系数 (Pearson correlation coefficient, $r$)：</strong>
    *   **概念定义：** 衡量两个变量之间线性相关性的强度和方向。其值介于 -1 和 1 之间，1 表示完全正线性相关，-1 表示完全负线性相关，0 表示无线性相关。
    *   **数学公式：**
        $$
        r = \frac{\sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^{n} (X_i - \bar{X})^2 \sum_{i=1}^{n} (Y_i - \bar{Y})^2}}
        $$
    *   **符号解释：**
        *   $n$: 样本数量。
        *   $X_i, Y_i$: 第 $i$ 对数据点的观测值（例如，智能体评分和人类评分）。
        *   $\bar{X}, \bar{Y}$: $X$ 和 $Y$ 的样本均值。

*   <strong>斯皮尔曼等级相关系数 (Spearman's rank correlation coefficient, $\rho$)：</strong>
    *   **概念定义：** 衡量两个变量（通常是排名数据）之间单调关系（不一定是线性）的强度和方向的非参数统计量。它评估的是两个变量的排名之间的一致性。
    *   **数学公式：**
        $$
        \rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
        $$
    *   **符号解释：**
        *   $n$: 数据点的数量。
        *   $d_i$: 每对观测值的秩次差（即 $X_i$ 和 $Y_i$ 的排名差）。

*   <strong>肯德尔等级相关系数 (Kendall's Tau, $\tau$)：</strong>
    *   **概念定义：** 衡量两个排名之间相似度的非参数统计量。它通过比较“一致对”和“不一致对”的数量来评估排名一致性，通常用于衡量两个序数变量之间的关联强度。
    *   **数学公式：**
        $$
        \tau = \frac{(\text{一致对数量}) - (\text{不一致对数量})}{n(n-1)/2}
        $$
    *   **符号解释：**
        *   $n$: 数据点的数量。
        *   一致对：两个变量的排名顺序相同的数据对。
        *   不一致对：两个变量的排名顺序相反的数据对。

## 5.3. 对比基线
论文将 `AVT` 与以下几类基线模型进行了比较：
*   **高光检测和视频摘要方法：**
    *   `UniVTG [27]`：统一视频语言时间定位模型。
    *   `UVCOM [57]`：统一视频理解框架，用于时刻检索和高光检测。
    *   `UMT [28]`：统一多模态转换器，用于联合视频时刻检索和高光检测。
    *   `PGL-SUM [4]`：结合全局和局部注意力与位置编码的视频摘要方法。
    *   其他高光检测/视频摘要方法：`LSVM [43]`、`Trailer [48]`、`SL-Module [59]`、`Joint-VA [5]`、`LIM-S [58]`、`MINI-Net [18]`、`TCG [65]`、`RRAE [61]`、`sLSTM [70]`、`SG [33]`。
*   **基线选择代表性：** 这些基线代表了视频时间定位（高光检测、时刻检索）和视频摘要领域最先进或具有代表性的方法。通过与它们比较，可以凸显 `AVT` 在新提出的视频修剪任务上的独特优势，以及在高光检测等子任务上的竞争力。

## 5.4. 实现细节
*   **`MLLM` 模型：** `AVT` 算法中所有的智能体均使用 `GPT-4o [37]` 模型实现，以增强多模态交互能力和确保输出格式的限制。
*   **视觉输入处理：**
    *   视频被分割成 3 秒的片段。
    *   每个片段默认每秒采样一帧 (`1 fps`) 作为视觉输入。
    *   所有帧图像被调整大小，短边为 512 像素。
*   **文本输入：** 包含精心设计的提示指令以及结构化的视频字幕和属性。
*   **API 成本估算：**
    *   对于一个 10 分钟的原始视频：
        *   输入图像词元 (token) 约 153,000 个。
        *   输入文本词元约 100,000 个。
        *   输出文本词元约 20,000 个。
        *   总 API 成本约为 \$0.83。
*   **输出视频时长：** 限制在约一分钟，以确保公平比较。
*   **消融实验中的效率分析：** 论文在补充材料中探讨了不同帧采样率（`1 fps` vs `4 fps`）和提示词设计（`Isolated` 独立提示 vs `Unified` 统一提示）对性能和成本的影响。结果显示，`1 fps` 采样率和 `Unified Prompt` 在保持性能的同时显著降低了成本。

    以下是原文 Table S3 关于实现和效率的详细对比：

    <table>
    <thead>
    <tr>
    <td>Frame Sampling Ratio</td>
    <td>Prompt</td>
    <td>Input Image Token</td>
    <td>Input Text Token</td>
    <td>Output Text Token</td>
    <td>API Cost</td>
    <td>Agent Metric</td>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>4/1s</td>
    <td>Isolated</td>
    <td>1,836,000</td>
    <td>100,000</td>
    <td>20,000</td>
    <td>\$5.04</td>
    <td>3.34</td>
    </tr>
    <tr>
    <td>4 /1s</td>
    <td>Unified</td>
    <td>612,000</td>
    <td>100,000</td>
    <td>20,000</td>
    <td>\$1.98</td>
    <td>3.33</td>
    </tr>
    <tr>
    <td>1/1s</td>
    <td>Isolated</td>
    <td>459,000</td>
    <td>100,000</td>
    <td>20,000</td>
    <td>\$1.60</td>
    <td>3.34</td>
    </tr>
    <tr>
    <td>1 /1s</td>
    <td>Unified</td>
    <td>153,000</td>
    <td>100,000</td>
    <td>20,000</td>
    <td>\$0.83</td>
    <td>3.32</td>
    </tr>
    </tbody>
    </table>

*   **表格描述：** 该表格详细对比了不同帧采样率和提示词设计对输入图像词元、输入文本词元、输出文本词元、API 成本和智能体指标（`Agent Metric`）的影响。可以看出，$1/1s$ 采样率配合 `Unified` 提示词能显著降低成本（\$0.83），同时保持与更高采样率或 `Isolated` 提示词相近的智能体评估性能（`3.32` vs `3.34`）。这表明了在效率和成本之间找到了一个有效的平衡点。

# 6. 实验结果与分析

本节将详细解读 `AVT` 的实验结果，包括在视频修剪任务上的人工和智能体评估、在高光检测任务上的定量比较，以及对 `AVT` 各组件的消融研究。

## 6.1. 核心结果分析

### 6.1.1. 视频修剪任务的人工评估
作者在自建的视频修剪数据集上进行了一项用户研究（盲测）。17 名参与者对不同方法生成的视频进行了 1 到 10 分的评分，评估了五个方面：内容丰富度 (`Material Richness`)、吸引力 (`Appeal`)、精彩片段 (`Exciting Segments`)、无用素材量 (`Amount of Wasted Footage`) 和整体感知 (`Overall`)。

以下是原文 Table 1 的结果：

<table>
<thead>
<tr>
<td>Method</td>
<td>Richness</td>
<td>Appeal</td>
<td>Excitement</td>
<td>Wasted</td>
<td>Overall</td>
<td>Agent</td>
</tr>
</thead>
<tbody>
<tr>
<td>UniVTG [27]</td>
<td>6.41</td>
<td>7.15</td>
<td>4.74</td>
<td>6.04</td>
<td>6.30</td>
<td>3.03</td>
</tr>
<tr>
<td>UVCOM [57]</td>
<td>6.15</td>
<td>7.12</td>
<td>4.69</td>
<td>6.47</td>
<td>6.23</td>
<td>2.91</td>
</tr>
<tr>
<td>AVT (ours)</td>
<td>7.21</td>
<td>7.78</td>
<td>5.57</td>
<td>6.72</td>
<td>7.15</td>
<td>3.32</td>
</tr>
</tbody>
</table>

*   **表格描述：** 该表格展示了在视频修剪数据集上，通过盲测方式对不同方法进行的用户研究结果。分数范围为 1 到 10。
*   **分析：**
    *   `AVT` 在所有用户评估指标中均表现最佳。特别是在 `Material Richness` (内容丰富度)、`Appeal` (吸引力) 和 `Overall` (整体感知) 方面，`AVT` 显著优于 `UniVTG` 和 `UVCOM`。
    *   在 `Excitement` (精彩片段) 方面，`AVT` 达到了 5.57 分，也高于其他方法。
    *   在 `Wasted` (无用素材量) 方面（分数越高表示无用素材越少），`AVT` 获得最高分 6.72，表明其在过滤无用素材方面表现卓越。
    *   表格右侧的 `Agent` 评估分数（由 `Video Evaluation Agent` 给出）也与人工评估的排名保持一致，进一步验证了智能体评估的有效性。
*   **结论：** 结果表明，`AVT` 能够生成更高质量、更具吸引力、更精彩且无用素材更少的最终视频，这得益于其独特的过滤机制和故事组合过程。

### 6.1.2. 视频修剪任务的智能体评估
除了人工评估，作者还使用其设计的 `Video Evaluation Agent` 对 `YouTube Highlights` 和 `TVSum` 数据集的验证集上的生成视频进行了评估。

以下是原文 Table 2 的结果：

<table>
<thead>
<tr>
<td>Dataset</td>
<td>Method</td>
<td>Richness</td>
<td>Appeal</td>
<td>Excitement</td>
<td>Wasted</td>
<td>Average</td>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4">YouTube Highlights</td>
<td>UMT [28]</td>
<td>2.70</td>
<td>3.08</td>
<td>3.40</td>
<td>3.44</td>
<td>3.15</td>
</tr>
<tr>
<td>UniVTG [27]</td>
<td>2.67</td>
<td>3.06</td>
<td>3.35</td>
<td>3.39</td>
<td>3.12</td>
</tr>
<tr>
<td>UVCOM [57]</td>
<td>2.72</td>
<td>3.10</td>
<td>3.45</td>
<td>3.45</td>
<td>3.18</td>
</tr>
<tr>
<td>AVT (ours)</td>
<td>2.79</td>
<td>3.17</td>
<td>3.53</td>
<td>3.44</td>
<td>3.23</td>
</tr>
<tr>
<td rowspan="5">TVSum</td>
<td>PGL-SUM [4]</td>
<td>2.75</td>
<td>3.05</td>
<td>3.10</td>
<td>3.10</td>
<td>3.00</td>
</tr>
<tr>
<td>UniVTG [27]</td>
<td>2.65</td>
<td>2.95</td>
<td>2.85</td>
<td>3.15</td>
<td>2.90</td>
</tr>
<tr>
<td>UVCOM [57]</td>
<td>2.50</td>
<td>2.80</td>
<td>2.70</td>
<td>3.30</td>
<td>2.83</td>
</tr>
<tr>
<td>AVT (ours)</td>
<td>3.15</td>
<td>3.35</td>
<td>3.25</td>
<td>3.70</td>
<td>3.36</td>
</tr>
</tbody>
</table>

*   **表格描述：** 该表格展示了在 `YouTube Highlights` 和 `TVSum` 数据集上，不同方法的智能体评估性能。分数范围为 1 到 5。
*   **分析：**
    *   在 `YouTube Highlights` 数据集上，`AVT` 在 `Richness` (内容丰富度)、`Appeal` (吸引力) 和 `Excitement` (精彩片段) 方面均获得最高分，并取得了最高的 `Average` (平均) 分数 3.23。在 `Wasted` (无用素材量) 方面与 `UVCOM` 持平。
    *   在 `TVSum` 数据集上，`AVT` 在所有评估指标 (`Richness`, `Appeal`, `Excitement`, `Wasted`, `Average`) 上均显著优于所有现有方法。
*   **结论：** 智能体评估结果与人工评估结果一致，再次证实了 `AVT` 在视频修剪任务上的卓越性能，以及 `Video Evaluation Agent` 的有效性。

### 6.1.3. 高光检测任务的定量比较
为了全面评估 `AVT` 的性能，作者还在传统的 `Highlight Detection` (高光检测) 任务上进行了比较。`AVT` 的显著性分数 ($S_i$) 定义如下：
$$
S _ { i } = { \left\{ \begin{array} { l l } { S _ { h } } & { { \mathrm { i f } } \ S _ { h } > m a x ( S _ { d } ) , } \\ { S _ { h } - m a x ( S _ { d } ) } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. }
$$
*   **符号解释：**
    *   $S_i$: 第 $i$ 个片段的显著性分数。
    *   $S_h$: 第 $i$ 个片段的 `Highlight` (高光) 分数。
    *   $S_d$: 第 $i$ 个片段的所有 `Defect Attributes` (缺陷属性) 分数。
    *   $max(S_d)$: 第 $i$ 个片段所有缺陷分数中的最大值。
*   **公式含义：** 如果一个片段的高光分数 $S_h$ 大于所有缺陷分数中的最大值 $max(S_d)$，则该片段的显著性分数直接取其高光分数 $S_h$。否则，显著性分数将是高光分数 $S_h$ 减去最大缺陷分数 $max(S_d)$。这使得缺陷较高的片段即使高光分数不低，其最终显著性分数也会被惩罚，从而降低其作为高光的优先级。

    以下是原文 Table 3 和 Table 4 的结果：

    <table>
    <thead>
    <tr>
    <td>Method</td>
    <td>Sup</td>
    <td>Dog</td>
    <td>Gym.</td>
    <td>Par.</td>
    <td>Ska.</td>
    <td>Ski.</td>
    <td>Sur.</td>
    <td>Avg.</td>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>LSVM [43]</td>
    <td>FS</td>
    <td>60.0</td>
    <td>41.0</td>
    <td>61.0</td>
    <td>62.0</td>
    <td>36.0</td>
    <td>61.0</td>
    <td>53.6</td>
    </tr>
    <tr>
    <td>Trailer [48]</td>
    <td>FS</td>
    <td>63.3</td>
    <td>82.5</td>
    <td>62.3</td>
    <td>52.9</td>
    <td>74.5</td>
    <td>79.3</td>
    <td>69.1</td>
    </tr>
    <tr>
    <td>SL-Module [59]</td>
    <td>FS</td>
    <td>70.8</td>
    <td>53.2</td>
    <td>77.2</td>
    <td>72.5</td>
    <td>66.1</td>
    <td>76.2</td>
    <td>69.3</td>
    </tr>
    <tr>
    <td>Joint-VA† [5]</td>
    <td>FS</td>
    <td>64.5</td>
    <td>71.9</td>
    <td>80.8</td>
    <td>62.0</td>
    <td>73.2</td>
    <td>78.3</td>
    <td>71.8</td>
    </tr>
    <tr>
    <td>UMT† [28]</td>
    <td>FS</td>
    <td>65.9</td>
    <td>75.2</td>
    <td>81.6</td>
    <td>71.8</td>
    <td>72.3</td>
    <td>82.7</td>
    <td>74.9</td>
    </tr>
    <tr>
    <td>UniVTG [27]</td>
    <td>FS</td>
    <td>74.3</td>
    <td>79.0</td>
    <td>74.4</td>
    <td>84.9</td>
    <td>75.1</td>
    <td>83.9</td>
    <td>78.6</td>
    </tr>
    <tr>
    <td>UVCOM [57]</td>
    <td>FS</td>
    <td>73.8</td>
    <td>77.1</td>
    <td>75.7</td>
    <td>75.3</td>
    <td>74.0</td>
    <td>82.7</td>
    <td>76.4</td>
    </tr>
    <tr>
    <td>LIM-S [58]</td>
    <td>WS</td>
    <td>57.9</td>
    <td>41.7</td>
    <td>67.0</td>
    <td>57.8</td>
    <td>48.6</td>
    <td>65.1</td>
    <td>56.4</td>
    </tr>
    <tr>
    <td>MINI-Net† [18]</td>
    <td>WS</td>
    <td>58.2</td>
    <td>61.7</td>
    <td>70.2</td>
    <td>72.2</td>
    <td>58.7</td>
    <td>65.1</td>
    <td>64.4</td>
    </tr>
    <tr>
    <td>TCG† [65]</td>
    <td>WS</td>
    <td>55.4</td>
    <td>62.7</td>
    <td>70.9</td>
    <td>69.1</td>
    <td>60.1</td>
    <td>59.8</td>
    <td>63.0</td>
    </tr>
    <tr>
    <td>RRAE [61]</td>
    <td>ZS</td>
    <td>49.0</td>
    <td>35.0</td>
    <td>50.0</td>
    <td>25.0</td>
    <td>22.0</td>
    <td>49.0</td>
    <td>38.3</td>
    </tr>
    <tr>
    <td>UniVTG [27]</td>
    <td>ZS</td>
    <td>48.8</td>
    <td>57.5</td>
    <td>59.4</td>
    <td>39.7</td>
    <td>57.4</td>
    <td>49.1</td>
    <td>52.0</td>
    </tr>
    <tr>
    <td>UVCOM [57]</td>
    <td>ZS</td>
    <td>46.6</td>
    <td>67.4</td>
    <td>61.4</td>
    <td>57.2</td>
    <td>63.5</td>
    <td>60.9</td>
    <td>59.5</td>
    </tr>
    <tr>
    <td>AVT (ours)</td>
    <td>ZS</td>
    <td>58.0</td>
    <td>62.1</td>
    <td>76.1</td>
    <td>32.0</td>
    <td>67.1</td>
    <td>67.9</td>
    <td>60.5</td>
    </tr>
    </tbody>
    </table>

*   **表格描述：** 该表格展示了在 `YouTube Highlights` 数据集上，不同方法的高光检测 mAP 结果。`Sup` 表示监督类型（FS: Fully supervised 完全监督, WS: Weakly supervised 弱监督, ZS: Zero-shot 零样本）。`†` 表示使用了音频模态。
*   **分析：** 在零样本 (ZS) 设置下，`AVT` 在平均 mAP 上达到 60.5%，优于所有其他零样本方法 (`UniVTG` 52.0%，`UVCOM` 59.5%）。这表明 `AVT` 在不依赖任何任务特定训练的情况下，也能有效地识别视频高光。在某些子类别中，如 `Par.`（冲浪）`AVT` 达到了 76.1%，显著优于其他零样本方法。

    <table>
    <thead>
    <tr>
    <td>Method</td>
    <td>Sup</td>
    <td>VT</td>
    <td>VU</td>
    <td>GA</td>
    <td>MS</td>
    <td>PK</td>
    <td>PR</td>
    <td>FM</td>
    <td>BK</td>
    <td>BT</td>
    <td>DS</td>
    <td>Avg.</td>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>sLSTM [70]</td>
    <td>FS</td>
    <td>41.1</td>
    <td>46.2</td>
    <td>46.3</td>
    <td>47.7</td>
    <td>44.8</td>
    <td>46.1</td>
    <td>45.2</td>
    <td>40.6</td>
    <td>47.1</td>
    <td>45.5</td>
    <td>45.1</td>
    </tr>
    <tr>
    <td>Trailer [48]</td>
    <td>FS</td>
    <td>61.3</td>
    <td>54.6</td>
    <td>65.7</td>
    <td>60.8</td>
    <td>59.1</td>
    <td>70.1</td>
    <td>58.2</td>
    <td>64.7</td>
    <td>65.6</td>
    <td>68.1</td>
    <td>62.8</td>
    </tr>
    <tr>
    <td>SL-Module [59]</td>
    <td>FS</td>
    <td>86.5</td>
    <td>68.7</td>
    <td>74.9</td>
    <td>86.2</td>
    <td>79.0</td>
    <td>63.2</td>
    <td>58.9</td>
    <td>72.6</td>
    <td>78.9</td>
    <td>64.0</td>
    <td>73.3</td>
    </tr>
    <tr>
    <td>Joint-VA† [5]</td>
    <td>FS</td>
    <td>83.7</td>
    <td>57.3</td>
    <td>78.5</td>
    <td>86.1</td>
    <td>80.1</td>
    <td>69.2</td>
    <td>70.0</td>
    <td>73.0</td>
    <td>97.4</td>
    <td>67.5</td>
    <td>76.3</td>
    </tr>
    <tr>
    <td>UMT [28]</td>
    <td>FS</td>
    <td>87.5</td>
    <td>81.5</td>
    <td>88.2</td>
    <td>78.8</td>
    <td>81.5</td>
    <td>87.0</td>
    <td>76.0</td>
    <td>86.9</td>
    <td>84.4</td>
    <td>79.6</td>
    <td>83.1</td>
    </tr>
    <tr>
    <td>UniVTG [27]</td>
    <td>FS</td>
    <td>92.0</td>
    <td>77.8</td>
    <td>89.8</td>
    <td>83.8</td>
    <td>82.2</td>
    <td>85.8</td>
    <td>74.3</td>
    <td>91.8</td>
    <td>90.5</td>
    <td>77.6</td>
    <td>84.6</td>
    </tr>
    <tr>
    <td>UVCOM [57]</td>
    <td>FS</td>
    <td>87.6</td>
    <td>91.6</td>
    <td>91.4</td>
    <td>86.7</td>
    <td>86.9</td>
    <td>86.9</td>
    <td>76.9</td>
    <td>92.3</td>
    <td>87.4</td>
    <td>75.6</td>
    <td>86.3</td>
    </tr>
    <tr>
    <td>LIM-S [58]</td>
    <td>WS</td>
    <td>55.9</td>
    <td>42.9</td>
    <td>61.2</td>
    <td>54.0</td>
    <td>60.4</td>
    <td>47.5</td>
    <td>43.2</td>
    <td>66.3</td>
    <td>69.1</td>
    <td>62.6</td>
    <td>56.3</td>
    </tr>
    <tr>
    <td>MINI-Net† [18]</td>
    <td>WS</td>
    <td>80.6</td>
    <td>68.3</td>
    <td>78.2</td>
    <td>81.8</td>
    <td>78.1</td>
    <td>65.8</td>
    <td>57.8</td>
    <td>75.0</td>
    <td>80.2</td>
    <td>65.5</td>
    <td>73.2</td>
    </tr>
    <tr>
    <td>TCG† [65]</td>
    <td>WS</td>
    <td>85.0</td>
    <td>71.4</td>
    <td>81.9</td>
    <td>78.6</td>
    <td>80.2</td>
    <td>75.5</td>
    <td>71.6</td>
    <td>77.3</td>
    <td>78.6</td>
    <td>68.1</td>
    <td>76.8</td>
    </tr>
    <tr>
    <td>SG [33]</td>
    <td>ZS</td>
    <td>42.3</td>
    <td>47.2</td>
    <td>47.5</td>
    <td>48.9</td>
    <td>45.6</td>
    <td>47.3</td>
    <td>46.4</td>
    <td>41.7</td>
    <td>48.3</td>
    <td>46.6</td>
    <td>46.2</td>
    </tr>
    <tr>
    <td>UniVTG [27]</td>
    <td>ZS</td>
    <td>52.0</td>
    <td>48.1</td>
    <td>50.9</td>
    <td>56.9</td>
    <td>51.6</td>
    <td>43.3</td>
    <td>60.0</td>
    <td>64.0</td>
    <td>59.2</td>
    <td>54.9</td>
    <td>54.1</td>
    </tr>
    <tr>
    <td>UVCOM [57]</td>
    <td>ZS</td>
    <td>63.4</td>
    <td>44.5</td>
    <td>50.6</td>
    <td>67.6</td>
    <td>55.1</td>
    <td>42.0</td>
    <td>47.5</td>
    <td>56.9</td>
    <td>58.6</td>
    <td>39.3</td>
    <td>52.5</td>
    </tr>
    <tr>
    <td>AVT (ours)</td>
    <td>ZS</td>
    <td>76.6</td>
    <td>75.9</td>
    <td>62.4</td>
    <td>63.9</td>
    <td>76.6</td>
    <td>68.8</td>
    <td>39.4</td>
    <td>45.6</td>
    <td>43.4</td>
    <td>62.9</td>
    <td>61.6</td>
    </tr>
    </tbody>
    </table>

*   **表格描述：** 该表格展示了在 `TVSum` 数据集上，不同方法的高光检测 Top-5 mAP 结果。`Sup` 表示监督类型（FS: Fully supervised 完全监督, WS: Weakly supervised 弱监督, ZS: Zero-shot 零样本）。`†` 表示使用了音频模态。
*   **分析：** 在零样本 (ZS) 设置下，`AVT` 在平均 Top-5 mAP 上达到 61.6%，再次显著优于所有其他零样本方法 (`UniVTG` 54.1%，`UVCOM` 52.5%）。在多个子类别中，`AVT` 也取得了非常高的分数，例如 `VT` (76.6%) 和 `VU` (75.9%)。
*   **结论：** `AVT` 在零样本高光检测任务中表现出卓越的性能，证明了其无需任务特定训练即可有效识别高光片段的能力。

    下图（原文 Figure 5）展示了在自建视频修剪数据集上的 `mAP` 和精度结果：

    ![Figure 5. Highlight detection results of mAP and precision on our collected video trimming dataset.](images/6.jpg)
    *该图像是图表，展示了我们所收集的视频修剪数据集上的 mAP 和精度结果。左侧图表显示了所有片段的 mAP (%) 值，右侧图表则显示选定片段的精度 (%) 值。各算法的表现通过曲线展示，其中 AVT（红色星标）在高光检测任务中显示出优越的性能。*

*   **图片描述：** 该图展示了在自建视频修剪数据集上，不同方法的高光检测 `mAP` 和高光片段精度。图中的柱状图对比了 `UniVTG`、`UVCOM` 和 `AVT` 在这两个指标上的表现，并显示了 `Waste` (无用素材) 和 `Highlight` (高光素材) 的比例。
*   **分析：** `AVT` 在 `mAP` 和高光片段精度方面均表现最佳。此外，`AVT` 选择的无用素材（`Waste`）比例最低，而高光片段（`Highlight`）比例最高，这直接验证了其过滤无用素材和提取高光片段的有效性。这进一步印证了 `AVT` 在视频修剪方面的综合优势。

## 6.2. 消融实验/参数分析

### 6.2.1. `AVT` 组件的有效性消融研究
作者对 `AVT` 的关键组件进行了消融研究，以评估它们对性能的贡献。评估指标包括用户研究得分、无用素材比例 (`Waste ↓`) 和高光片段比例 (`Highlight ↑`)。

以下是原文 Table 5 的结果：

<table>
<thead>
<tr>
<td>Method</td>
<td>VS</td>
<td>CF</td>
<td>DF</td>
<td>SC</td>
<td>User ↑</td>
<td>Waste ↓</td>
<td>Highlight ↑</td>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">UniVTG [27] UVCOM [57]</td>
<td></td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>6.30</td>
<td>0.276</td>
<td>0.066</td>
</tr>
<tr>
<td></td>
<td>-</td>
<td></td>
<td></td>
<td>6.23</td>
<td>0.175</td>
<td>0.066</td>
</tr>
<tr>
<td rowspan="6">AVT (ours)</td>
<td></td>
<td></td>
<td>-</td>
<td>-</td>
<td>3.70</td>
<td>0.337</td>
<td>0.083</td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td>-</td>
<td>6.19</td>
<td>0.135</td>
<td>0.110</td>
</tr>
<tr>
<td></td>
<td></td>
<td>-</td>
<td></td>
<td>6.45</td>
<td>0.165</td>
<td>0.096</td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td></td>
<td>6.70</td>
<td>0.141</td>
<td>0.109</td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td>L</td>
<td>5.23</td>
<td>0.199</td>
<td>0.107</td>
</tr>
<tr>
<td></td>
<td></td>
<td>V</td>
<td>L</td>
<td>7.15</td>
<td>0.083</td>
<td>0.108</td>
</tr>
</tbody>
</table>

*   **表格描述：** 该表格展示了 `AVT` 各组件（`VS`: Video Structuring 视频结构化, `CF`: Clip Filtering 剪辑过滤, `DF`: Dynamic Filter 动态过滤器, `SC`: Story Composition 故事组合）的消融研究结果。`User ↑` 表示用户评分，`Waste ↓` 表示无用素材比例（越低越好），`Highlight ↑` 表示高光片段比例（越高越好）。`-` 表示该组件被禁用，空白表示组件被启用。$L$ 和 $V$ 在 `SC` 列可能表示不同的组合策略或参数，但原文未详细说明。
*   **分析：**
    *   <strong>基线（`UniVTG` 和 `UVCOM`）：</strong> 它们的 `Waste` 比例较高（0.276和0.175），`Highlight` 比例较低（0.066），用户评分也较低（6.30和6.23），因为它们没有专门的过滤和组合阶段。
    *   <strong>禁用 `CF` 和 `SC`（仅 `VS`）：</strong> 用户评分仅为 3.70，`Waste` 比例高达 0.337，`Highlight` 比例为 0.083。这表明如果只进行视频结构化而不进行过滤和组合，效果会很差，甚至不如基线方法。
    *   <strong>启用 `CF` (不含 `DF`) 和 `VS`，禁用 `SC`：</strong> `Waste` 比例降至 0.135，`Highlight` 比例升至 0.110，用户评分达到 6.19。这说明 `Clip Filtering`（即使没有动态过滤器）能够有效减少无用素材并提升高光比例。
    *   <strong>启用 `CF` (不含 `DF`) 和 `SC`：</strong> 用户评分提升至 6.45，`Waste` 比例为 0.165，`Highlight` 比例为 0.096。这表明 `Story Composition` 能够进一步改善用户对视频的整体印象。
    *   <strong>启用 `CF` (含 `DF`) 和 `VS`，禁用 `SC`：</strong> 用户评分 6.70，`Waste` 比例 0.141，`Highlight` 比例 0.109。说明 `Dynamic Filter` 在没有 `Story Composition` 的情况下，对过滤无用素材和提升高光比例仍有贡献。
    *   <strong>完全体 `AVT` (启用 `VS`, `CF`, `DF`, `SC`，最后一行):</strong> 用户评分高达 7.15，`Waste` 比例降至最低 0.083，`Highlight` 比例稳定在 0.108。这证明了所有组件协同工作能够达到最佳效果。特别是 `Dynamic Filter` 的加入，显著降低了无用素材比例。
*   **结论：** 消融研究明确指出，`Clip Filtering` 显著减少了无用素材，而 `Story Composition` 提升了整体视频的观感。`Dynamic Filter` 模块尤其重要，它能防止有价值的精彩片段因轻微的拍摄缺陷而被错误丢弃，尤其在体育内容中效果显著。这验证了 `AVT` 各组件的有效性。

### 6.2.2. 评估智能体与人类相关性
为了验证 `Video Evaluation Agent` 的可靠性，作者使用 `Pearson ($r$)`、`Spearman (`\rho`)` 和 `Kendall-Tau (`\tau`)` 这三个元评估指标来衡量其与人类偏好的一致性。同时，他们还进行了消融研究，探讨了要求智能体提供理由 (`Output Reason`) 和使用多样化评估标准 (`Diverse Criteria`) 对相关性的影响。

以下是原文 Table 6 的结果：

<table>
<thead>
<tr>
<td>Output Reason</td>
<td>Diverse Criteria</td>
<td>r</td>
<td>ρ</td>
<td>τ</td>
<td>Avg.</td>
</tr>
</thead>
<tbody>
<tr>
<td>-</td>
<td>-</td>
<td>0.2675</td>
<td>0.2451</td>
<td>0.1723</td>
<td>0.2283</td>
</tr>
<tr>
<td>-</td>
<td>✓</td>
<td>0.4082</td>
<td>0.4119</td>
<td>0.3067</td>
<td>0.3756</td>
</tr>
<tr>
<td>✓</td>
<td>-</td>
<td>0.5260</td>
<td>0.4990</td>
<td>0.3738</td>
<td>0.4663</td>
</tr>
<tr>
<td>✓</td>
<td>✓</td>
<td>0.5616</td>
<td>0.5667</td>
<td>0.4457</td>
<td>0.5247</td>
</tr>
</tbody>
</table>

*   **表格描述：** 该表格展示了在视频修剪基准数据集上，不同策略下评估智能体与人类偏好之间 `Pearson ($r$)`、`Spearman (`\rho`)` 和 `Kendall-Tau (`\tau`)` 相关性，以及平均相关性。`✓` 表示激活该策略，`-` 表示禁用。
*   **分析：**
    *   在没有任何策略的情况下 (`-`, `-`)，平均相关性仅为 0.2283，相关性较弱。
    *   仅激活 `Diverse Criteria` (`-`, `✓`)，平均相关性提高到 0.3756。
    *   仅激活 `Output Reason` (`✓`, `-`)，平均相关性进一步提高到 0.4663。
    *   当同时激活 `Output Reason` 和 `Diverse Criteria` (`✓`, `✓`) 时，相关性达到最高，平均相关性为 0.5247 ($r$=0.5616, $ρ$=0.5667, $τ$=0.4457)。
*   **结论：** 结果表明，要求评估智能体提供评估理由，并使用多样化的评估标准，能够显著提高智能体评估结果与人类偏好的一致性，证明了 `Video Evaluation Agent` 设计的有效性。

### 6.2.3. 视频与原始视频的保真度评估
作者还对最终视频与原始视频之间的视觉内容相似度（保真度）进行了定量评估。他们使用 `ViCLIP [51]` 和 `InternVideo2 [52]` 提取视频特征，然后计算特征相似度。

以下是原文 Table S4 的结果：

<table>
<thead>
<tr>
<td>Method</td>
<td>ViCLIP</td>
<td>InternVideo2</td>
<td>Avg.</td>
</tr>
</thead>
<tbody>
<tr>
<td>UniVTG [27]</td>
<td>0.877</td>
<td>0.941</td>
<td>0.909</td>
</tr>
<tr>
<td>UVCOM [57]</td>
<td>0.852</td>
<td>0.928</td>
<td>0.890</td>
</tr>
<tr>
<td>AVT (ours)</td>
<td>0.906</td>
<td>0.951</td>
<td>0.929</td>
</tr>
</tbody>
</table>

*   **表格描述：** 该表格展示了在不同方法生成的最终视频与原始视频之间的保真度比较结果，通过 `ViCLIP` 和 `InternVideo2` 提取的特征相似度来衡量。
*   **分析：** `AVT` 在 `ViCLIP` (0.906) 和 `InternVideo2` (0.951) 两种特征提取模型下，都取得了最高的保真度分数，平均达到 0.929。这表明 `AVT` 在修剪视频的同时，能够更好地保留原始视频的核心内容和视觉特征，确保了最终视频与原始素材的视觉一致性。
*   **结论：** `AVT` 不仅在主观评价和高光检测上表现出色，在客观的保真度指标上也优于现有方法，进一步证明了其方法的有效性和全面性。

## 6.3. 可视化
论文通过多个图表展示了 `AVT` 的可视化结果，进一步说明其优势。

下图（原文 Figure 6 和 Figure S8）展示了不同方法在视频修剪数据集上的可视化结果：

![该图像是示意图，展示了不同视频处理算法（如UNITE6、UVCOM和AVT）在视频修剪任务中的效果和性能。每行包含视频片段、相关的“显著性评分”和不同类别的片段（被过滤片段、选择片段、垃圾片段和高亮片段）的标记，通过颜色区分，反映出各算法在视频内容筛选和关键段落检测方面的表现。这些数据有助于比较不同算法在视频修剪中的有效性。](images/11.jpg)
*该图像是示意图，展示了不同视频处理算法（如UNITE6、UVCOM和AVT）在视频修剪任务中的效果和性能。每行包含视频片段、相关的“显著性评分”和不同类别的片段（被过滤片段、选择片段、垃圾片段和高亮片段）的标记，通过颜色区分，反映出各算法在视频内容筛选和关键段落检测方面的表现。这些数据有助于比较不同算法在视频修剪中的有效性。*

![Figure S8. Visualization of trimmed videos on the video trimming dataset.](images/12.jpg)
*该图像是一个视频修剪效果的可视化示意图，展示了不同视频片段的处理情况，包括滤除的低质量片段和选定的重点片段。每个视频片段的浪费程度和高光片段的比例清晰可见，从而帮助分析视频剪辑的质量和有效性。*

*   **图片描述：** 这些图通过时间轴的形式，可视化了不同方法（`UniVTG`、`UVCOM`、`AVT`）生成的视频的显著性分数、选定片段以及各种片段类型（被过滤、选定、无用、高光）。
*   **分析：**
    *   **高光捕捉：** `AVT` 能够更有效地捕捉视频中的高光片段，例如，在山地骑行视频中，`AVT` 能够识别并选择更多动态的骑行场景，而其他方法可能选择更多平淡的镜头。
    *   **无用素材过滤：** `AVT` 在过滤无用素材方面表现出色，其生成的视频中，“无用素材”的比例明显低于其他方法。
    *   **叙事连贯性：** `AVT` 选定的片段在时间轴上分布更合理，能够形成更连贯的故事情节，而非简单地拼接高分片段。例如，对于美食视频或海豚表演，`AVT` 能够从头到尾有效地修剪出完整的故事。

        下图（原文 Figure S5）展示了 `Clip Filtering`（剪辑过滤）的效果：

        ![Figure S5. Effect of clip filtering on visualization of trimmed videos.](images/8.jpg)
        *该图像是图表，展示了使用 AVT 方法与未使用过滤器的情况对比在视频修剪中的表现。图中可以看到，AVT（上）有效地减少了无用镜头，并且选择了更多有价值的片段，同时标记了不同的片段类别。相关的 saliency 分数和视频片段过滤情况也在下方详细呈现。*

*   **图片描述：** 该图对比了有无 `Clip Filtering` 模块时，视频修剪结果的可视化。
*   **分析：** 在没有 `Clip Filtering` 的情况下，所有间隔都作为候选片段输入到故事组合阶段（图示中第一行是全绿的），这导致最终视频中包含了更多的无用素材。这进一步证明了 `Clip Filtering` 模块在排除低质量片段中的重要性。

    下图（原文 Figure S6）展示了 `Dynamic Filter`（动态过滤器）模块的效果：

    ![Figure S6. Effect of dynamic filter module on visualization of trimmed videos.](images/9.jpg)
    *该图像是示意图，展示了动态过滤模块对修剪视频可视化效果的影响。图中上方为我们提出的AVT方法，底部为未使用动态过滤的结果。不同颜色表示不同的内容片段，图表中还展示了废弃和高亮片段的变化情况。*

*   **图片描述：** 该图对比了有无 `Dynamic Filter` 模块时，视频修剪结果的可视化。
*   **分析：** 如果没有 `Dynamic Filter`，剪辑过滤器会丢弃大部分片段，尤其是在体育内容中，因为剧烈活动常常伴随抖动或其他视觉缺陷。这会导致大量高光片段被错误地分类为缺陷并被排除在组合之外（图示第二行筛选出的片段明显少于第一行）。这强调了 `Dynamic Filter` 在平衡内容精彩度与拍摄缺陷之间的关键作用。

    下图（原文 Figure S7）展示了多层次故事情节的可视化：

    ![Figure S7. Visualization of the multi-level storyline of the trimmed final video.](images/10.jpg)
    *该图像是一个示意图，展示了修剪后视频的多层次故事线。图中包含多个视频片段的描述，展现家庭活动的不同场景，如在家放松、户外探险、餐厅就餐和夜间活动，强调了家庭团聚和亲子互动的主题。*

*   **图片描述：** 该图展示了修剪后的最终视频的多层次故事情节，包括片段级字幕、聚类主题和全局故事情节。
*   **分析：** 智能体不仅选择和排列片段，还为每个片段生成了详细的字幕（`clip-wise captions`），将相关片段聚类成主题（`clustered themes`），并最终形成一个总结性的全局故事情节（`global storyline`）。这直观地展示了 `AVT` 在理解和组织视频叙事方面的能力。

## 6.4. 总结
实验结果全面且有力地证明了 `AVT` 方法的有效性和优越性。无论是在主观的用户评估、客观的高光检测指标，还是在对模型组件的消融研究中，`AVT` 都展现了卓越的性能。特别是其在过滤无用素材、保留高光片段和构建连贯故事情节方面的能力，使其在视频修剪这一新任务中树立了坚实的基线。

# 7. 总结与思考

## 7.1. 结论总结
本文首次引入了 `Video Trimming (VT)`（视频修剪）这一新颖任务，其核心在于从冗余的视频内容中选择有意义的片段并保留叙事连贯性，以提取关键见解。为解决此任务，作者提出了 `Agent-based Video Trimming (AVT)`（基于智能体的视频修剪）框架。`AVT` 包含三个关键阶段：
1.  <strong>视频结构化 (Video Structuring)：</strong> 利用 `Video Captioning Agent` 提供视频片段的结构化描述，包括内容、上下文和缺陷属性。
2.  <strong>剪辑过滤 (Clip Filtering)：</strong> 采用动态过滤模块，根据片段的高光程度和缺陷程度智能地选择有用片段。
3.  <strong>故事组合 (Story Composition)：</strong> 通过 `Video Arrangement Agent` 将筛选后的片段编排成具有连贯叙事结构的最终视频。

    此外，本文还设计了一个 `Video Evaluation Agent`（视频评估智能体）来评估视频质量，并构建了一个用于视频修剪任务的基准数据集。实验结果表明，`AVT` 在高光检测方面超越了现有方法，并在用户研究中获得了更优的人类偏好，证明了其在生成高质量、有故事性视频方面的卓越能力。

## 7.2. 局限性与未来工作
论文正文并未明确指出 `AVT` 自身的局限性或未来工作方向。然而，基于论文的内容和当前技术发展，我们可以推断出一些潜在的局限性并展望未来的研究方向。

### 7.2.1. 潜在局限性
*   **`MLLM` 的成本与效率：** 尽管论文通过优化采样率和提示词设计显著降低了 `API` 成本（从 `5.04 降至`0.83），但对于超长视频（例如数小时的原始素材），`GPT-4o` 等商业 `MLLM API` 的调用费用仍然可能很高。此外，基于 `API` 的方法也可能存在推理延迟，限制了实时应用。
*   **`MLLM` 的鲁棒性与一致性：** `MLLM` 仍然可能受到“幻觉”现象的影响，或者对提示词 (`prompt`) 的微小变化敏感。这可能导致 `Video Captioning Agent` 生成的描述或缺陷评估不完全准确，进而影响后续的过滤和故事组合效果。在处理极端复杂、模糊或低质量的视频内容时，其性能可能会下降。
*   **叙事复杂性：** 当前 `Story Composition` 主要聚焦于“开始-发展-结局”的线性叙事结构。对于更复杂、非线性、多线索或需要高度艺术性的视频叙剪，`Agent` 的能力可能仍有提升空间。例如，如何处理蒙太太奇、闪回、多视角切换等高级编辑技巧。
*   **用户个性化程度：** `AVT` 主要生成通用意义上的“好故事”，但可能难以完全捕捉用户的特定偏好、主题重点或情绪要求。要实现高度个性化的视频修剪，需要更深入的用户意图理解和交互机制。
*   **缺陷检测的细粒度：** 论文定义的四种缺陷（遮挡、抖动、过曝、无意义内容）较为通用。现实世界中的视频缺陷可能更细致多样，例如画面模糊、构图不佳、音质问题、目标追踪失败等，更细粒度的缺陷识别可能进一步提升过滤效果。

### 7.2.2. 未来研究方向
*   **成本效益优化与本地化部署：** 探索使用更小、更高效的开源 `MLLM` 模型进行本地化部署，或结合级联（`cascading`）模型架构，以显著降低成本并提高推理速度，使其更适用于大规模或实时应用。
*   **高级叙事结构与编辑技巧：** 研究如何让 `Video Arrangement Agent` 理解并应用更复杂的叙事结构和专业的视频编辑技巧，例如引入用户定义的叙事模板、情绪曲线或特定风格指导。
*   **多模态融合与感知增强：** 除了视觉内容，进一步整合音频信息（如背景音乐、对话、音效）和文本信息（如字幕、旁白），以实现更全面的视频理解和更精细的片段选择与组合。
*   **用户意图与偏好集成：** 开发更直观的交互界面和更智能的算法，允许用户通过自然语言或简单的操作提供个性化偏好，从而生成高度定制化的视频剪辑。
*   **长视频处理的挑战：** 进一步优化 `MLLM` 处理超长上下文的能力，或者设计更有效的长视频分块、摘要和跨块推理机制，以应对小时级甚至更长的原始视频素材。
*   **`Agent` 协作与学习：** 探索多个 `Agent` 之间的协作学习机制，使其能够从用户反馈中不断学习和改进，甚至能像人类编辑一样，通过试错和迭代来提升视频修剪质量。

## 7.3. 个人启发与批判
### 7.3.1. 个人启发
*   **`MLLM` 作为“无需训练智能体”的巨大潜力：** 本文最令人启发之处在于将 `MLLM` 视为一个多功能、无需训练的智能体，通过精心设计的提示词即可完成复杂的、多步骤的视频编辑任务。这种范式极大地降低了任务特定模型开发的门槛，展现了基础模型在通用任务解决中的强大能力和灵活性。
*   **结构化文本表示的价值：** 将原始视频内容转化为结构化文本描述，是一个非常高效且优雅的解决方案。它将复杂的视觉理解问题转化为 `LLM` 擅长的文本推理问题，极大地提高了后续处理的效率和可解释性。这种思想可以推广到其他多模态任务，作为 `LLM` 与其他模态交互的通用接口。
*   **`Dynamic Filter` 的巧妙设计：** 在缺陷过滤中，简单地剔除所有有缺陷的片段是不现实的。`AVT` 的 `Dynamic Filter` 机制通过平衡高光分数和缺陷分数，允许在内容足够精彩时容忍一定程度的缺陷，这非常符合现实世界用户生成内容的特点，是算法实用性的关键。
*   **叙事连贯性的重要性：** `AVT` 明确将“故事组合”作为核心阶段，而非仅仅追求高光片段。这提醒我们在内容生成领域，用户体验往往超越了单一指标的优化，完整、流畅的叙事对于吸引观众至关重要。

### 7.3.2. 批判性思考与可改进之处
*   **`Agent` 提示词的敏感性与维护成本：** `AVT` 高度依赖精心设计的提示词来引导 `MLLM` 智能体。这些提示词可能对 `LLM` 模型的版本更新或底层架构变化非常敏感。维护和优化这些复杂的提示词本身就是一项挑战，可能需要大量的工程投入。
*   **`MLLM` 推理的“黑箱”特性：** 尽管 `CoT` 提供了中间推理步骤，但 `MLLM` 的决策过程仍然是一个相对的“黑箱”。当 `AVT` 生成的视频不符合预期时，诊断问题并进行修正可能比传统机器学习模型更具挑战性。
*   **伦理与偏见问题：** `MLLM` 在生成视频描述和进行内容筛选时，可能会继承其训练数据中的偏见。例如，在识别“无意义内容”或“高光”时，如果训练数据存在偏差，可能导致算法倾向于某些特定风格或主题，而忽略其他同样有价值的内容。这在用户生成内容场景下尤为重要，因为用户视频的多样性远超预训练数据集。
*   **上下文理解的深度：** 尽管 `AVT` 尝试构建故事，但其对“故事”的理解可能仍停留在语义层面的拼接。人类在剪辑视频时，会考虑更深层次的情感、节奏、象征意义等。`AVT` 离实现真正意义上的“情感化”或“艺术化”剪辑仍有距离。
*   **人机协作的潜力：** `AVT` 作为全自动系统，未来可以探索与人类编辑的协作模式。例如，`AVT` 可以提供初步的修剪建议和故事情节草稿，然后由人类编辑进行精修和个性化调整，形成更高效、更高质量的编辑工作流。