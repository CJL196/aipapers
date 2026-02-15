# 1. 论文基本信息

## 1.1. 标题
Qwen3-VL 技术报告 (Qwen3-VL Technical Report)

## 1.2. 作者
Qwen 团队 (Qwen Team)。
核心贡献者包括：Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xiongshu Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Daiyheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Penguin Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, Ke Zhu。
贡献者包括：Yizhong Cao, Bei Chen, Chen Cheng, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Rongyao Fang, Tongkun Guan, Jinzheng He, Miao Hong, Songtao Jiang, Zheng Li, Xiaochuan Li, Junrong Lin, Yuqiang Liu, Yanta Lou, Na Ni, Xinyao Niu, Yatian Pang, Zihan Qiu, Tianhao Shen, Tianyi Tang, Yu Wan, Jinxi Wei, Chenfei Wu, Buxiao Wu, Xiao Xu, Mingfeng Xue, Ming Yan, Yuhuan Yang, Jiaxi Yang, Kexin Yang, Le Yu, Hao Yu, Jianke Zhang, Jianwei Zhang, Yichang Zhang, Zhenru Zhang, Siqi Zhang, Peiyang Zhang, Beichen Zhang, Hongbo Zhao, Xianwei Zhuang。

## 1.3. 发表期刊/会议
预印本 (Preprint)。

## 1.4. 发表年份
2025年11月26日 (UTC)。

## 1.5. 摘要
本报告介绍了 Qwen3-VL，这是 Qwen 系列迄今为止功能最强大的视觉-语言模型 (vision-language model, VLM)，在广泛的多模态基准测试中取得了卓越性能。它原生支持高达 256K 词元 (token) 的交错上下文 (interleaved contexts)，无缝集成了文本、图像和视频。该模型家族包括密集型 (dense) (2B/4B/8B/32B) 和专家混合 (Mixture-of-Experts, MoE) (30B-A3B/235B-A22B) 变体，以适应不同的延迟-质量权衡。Qwen3-VL 具备三个核心支柱：(i) 显著更强的纯文本理解能力，在某些情况下超越了可比较的纯文本主干网络 (text-only backbones)；(ii) 鲁棒的长上下文理解能力，为文本和交错多模态输入提供原生 256K 词元窗口，能够在长文档和视频中忠实地保留、检索和交叉引用信息；(iii) 在单图像、多图像和视频任务中展现先进的多模态推理能力，在 MMMU 和视觉-数学基准 (例如 MathVista 和 MathVision) 等综合评估中表现领先。在架构上，该团队引入了三项关键升级：(i) 增强的交错式 MRoPE (interleaved-MRoPE)，用于图像和视频中更强的时空建模；(ii) DeepStack 集成，有效利用多层视觉 Transformer (Vision Transformer, ViT) 特征以加强视觉-语言对齐；(iii) 视频的基于文本的时间对齐，从 T-RoPE 演变为显式文本时间戳对齐，以实现更精确的时间定位。在可比较的词元预算和延迟限制下，Qwen3-VL 在密集型和 MoE 架构中均实现了卓越性能。该团队设想 Qwen3-VL 将作为图像基础推理、智能体决策和现实世界工作流程中多模态代码智能的基石引擎。

## 1.6. 原文链接
官方来源：[https://arxiv.org/abs/2511.21631](https://arxiv.org/abs/2511.21631)
PDF 链接：[https://arxiv.org/pdf/2511.21631v2](https://arxiv.org/pdf/2511.21631v2)
发布状态：预印本 (Preprint)。

# 2. 整体概括

## 2.1. 研究背景与动机
**论文试图解决的核心问题：**
现有的视觉-语言模型 (VLM) 在长上下文理解、STEM（科学、技术、工程和数学）推理、图形用户界面 (Graphical User Interface, GUI) 理解和交互以及智能体 (agentic) 工作流等高级应用中仍面临挑战。同时，多模态模型在获得视觉能力的同时，不能削弱其底层大型语言模型 (Large Language Model, LLM) 的语言能力，甚至应在语言基准测试上超越纯文本模型。

**为什么这个问题在当前领域是重要的：**
随着人工智能技术的发展，对模型处理复杂、多模态信息能力的需求日益增长。现实世界的应用场景，如分析长篇技术文档、理解长时间视频、进行复杂的科学计算或实现自主智能体，都要求模型具备强大的多模态理解、推理和长上下文处理能力。如果多模态模型在融入视觉信息的同时损害了语言能力，则其通用性和实用性将大打折扣。

<strong>现有研究存在哪些具体的挑战或空白 (Gap)：</strong>
1.  **长上下文理解能力不足：** 许多 VLM 在处理超过一定长度（如几千词元）的交错文本、图像和视频序列时，性能会显著下降。
2.  **视觉-语言对齐不紧密：** 简单地将视觉特征与语言模型结合，可能无法实现深层次、细粒度的视觉-语言对齐，导致推理能力受限。
3.  **视频理解的精确时序建模：** 在视频中，如何有效地编码和利用时间信息，特别是长视频中的时间戳，是一个挑战。
4.  **纯文本能力退化：** 多模态训练往往需要平衡文本和视觉数据，不当的训练策略可能导致模型在纯文本任务上的性能下降。
5.  **缺乏对复杂推理任务的支持：** 尤其是在 STEM 领域和视觉-数学推理方面，模型需要更强的逻辑和多步推理能力。
6.  **智能体工作流的整合：** 将感知、推理和行动无缝集成到智能体工作流中，以实现 GUI 交互和代码智能等复杂任务，仍是前沿挑战。

**这篇论文的切入点或创新思路：**
Qwen3-VL 的创新思路在于通过**架构升级**、**数据迭代**和**训练策略优化**三方面协同提升模型能力：
1.  **架构改进：** 引入增强的交错式 MRoPE (Interleaved MRoPE) 来优化时空建模，采用 DeepStack 机制进行多层视觉特征融合以增强视觉-语言对齐，并使用基于文本的时间戳对齐来提升视频时序理解。
2.  **高质量、多样化的数据：** 全面升级预训练数据，涵盖高质量图像标题、交错文本-图像数据、多语言 OCR、文档解析、长文档理解、精细化定位、计数、空间理解、3D 感知、多模态代码和视频数据，并融入链式思考 (Chain-of-Thought, CoT) 推理和 GUI 智能体交互数据。
3.  **精细化训练策略：** 采用多阶段预训练和后训练 (post-training) 流程，逐步扩大上下文窗口，并引入强到弱蒸馏 (Strong-to-Weak Distillation) 和强化学习 (Reinforcement Learning, RL) 进一步提升模型性能和对齐人类偏好。同时，通过平方根重加权 (square-root reweighting) 平衡文本和多模态学习目标，确保纯文本能力不受影响。
4.  **模型家族的多样性：** 提供从 2B 到 235B 参数的密集型和 MoE 模型，以满足不同的性能和部署需求。

## 2.2. 核心贡献/主要发现
**论文最主要的贡献：**
1.  **提出了 Qwen3-VL 模型家族：** 包含了从 2B 到 235B 参数的密集型和专家混合 (MoE) 变体，支持高达 256K 词元的原生交错上下文，是 Qwen 系列迄今为止最强大的视觉-语言模型。
2.  **三项关键架构升级：**
    *   **增强的交错式 MRoPE：** 优化了图像和视频的时空建模，解决了传统 MRoPE 在长视频理解中频率频谱不平衡的问题。
    *   **DeepStack 集成：** 通过利用多层视觉编码器 (vision encoder) 特征与 LLM 对应层进行融合，显著增强了视觉-语言对齐。
    *   **基于文本的视频时间对齐：** 采用显式文本时间戳来表示视频帧的时序信息，比传统的时序位置编码更精确和直接。
3.  **全面的数据和训练策略改进：**
    *   **高质量、多样化的预训练数据：** 大幅扩展并优化了图像标题、交错文本-图像、OCR、长文档、空间理解、3D 感知、代码和视频等多模态数据，并引入链式思考和 GUI 智能体数据。
    *   **多阶段预训练和后训练：** 采用逐步扩展上下文窗口的预训练，以及包含 SFT (Supervised Fine-Tuning)、强到弱蒸馏和强化学习的后训练流程，并区分“非思考型 (non-thinking)”和“思考型 (thinking)”变体。
    *   **平方根重加权：** 平衡文本和多模态数据对损失的贡献，确保多模态性能提升的同时不损害文本能力。
4.  **卓越的性能表现：** 在广泛的多模态基准测试（如 MMMU, MathVista, MathVision）和文本基准测试上均取得领先或最先进的性能，尤其在长上下文理解、多模态推理和细粒度感知方面表现突出。即使是较小的模型也展现出强大的竞争力。
5.  **扩展的多语言支持：** 从 Qwen2.5-VL 的 10 种非英语/中文语言扩展到 39 种语言，并在多语言 OCR 测试中表现出色。

**论文得出了哪些关键的结论或发现：**
1.  **视觉-语言模型可以超越纯文本模型：** Qwen3-VL 在多个文本基准上超越了其纯文本主干网络，证明了多模态训练不仅能带来视觉能力，还能反哺和增强语言理解能力。
2.  **长上下文能力对多模态推理至关重要：** 原生 256K 词元窗口以及在“干草堆中的针 (Needle-in-a-Haystack)”任务中的优异表现，验证了 Qwen3-VL 在处理长文档和视频方面无与伦比的能力。
3.  **架构创新是性能飞跃的关键：** 交错式 MRoPE、DeepStack 和基于文本的时间对齐等架构改进，显著提升了模型在时空建模、视觉-语言对齐和视频理解方面的能力。
4.  **高质量数据和精细化训练是模型成功的基石：** 通过数据迭代和分阶段的训练策略，特别是对推理型 (thinking) 模型的专注训练，模型在复杂推理任务上表现出显著优势。
5.  **工具增强型智能体学习潜力巨大：** 细粒度感知实验表明，集成外部工具带来的性能提升甚至超过了单纯增大模型规模带来的收益，预示着未来多模态智能体学习的广阔前景。

# 3. 预备知识与相关工作

## 3.1. 基础概念
为了理解 Qwen3-VL 的技术细节，需要了解以下基础概念：

*   <strong>视觉-语言模型 (Vision-Language Model, VLM)</strong>：VLM 是一种能够同时处理和理解视觉（图像、视频）和语言（文本）信息的人工智能模型。它们旨在弥合视觉和语言模态之间的差距，实现如图像描述、视觉问答、图像生成文本等任务。

*   <strong>大型语言模型 (Large Language Model, LLM)</strong>：LLM 是一种基于深度学习的语言模型，拥有数亿到数万亿的参数，通过在大量文本数据上进行预训练来学习语言的模式、语法、语义和世界知识。LLM 是许多 VLM 的核心语言处理组件。

*   <strong>词元 (Token)</strong>：在自然语言处理中，词元是文本的基本单位，可以是单词、子词（subword）或字符。在 VLM 中，图像和视频也会被转换为视觉词元 (visual tokens)，以便与语言模型进行统一处理。

*   <strong>上下文窗口 (Context Window)</strong>：指模型在处理当前输入时可以考虑的先前信息量。例如，一个 256K 词元的上下文窗口意味着模型在生成或理解每个词元时，可以参考其前 256,000 个词元（包括文本和视觉词元）。长上下文窗口对于理解长文档和视频至关重要。

*   <strong>交错上下文 (Interleaved Contexts)</strong>：指模型能够处理文本、图像和视频等不同模态信息混合排列的输入序列。例如，一篇包含文字和图片的文章，或者一段带有文字描述的视频。

*   <strong>专家混合 (Mixture-of-Experts, MoE) 架构</strong>：MoE 是一种神经网络架构，其中模型的不同部分（专家网络）专注于处理输入数据的不同方面。在推理时，一个门控网络 (gating network) 会根据输入选择性地激活一个或几个专家网络，而不是激活所有网络。这使得模型可以在保持参数总数巨大的同时，只激活少量参数，从而提高效率和性能。论文中提到了 30B-A3B 和 235B-A22B，其中 A3B 表示每次激活 30 亿参数中的 3 亿参数，A22B 表示每次激活 2350 亿参数中的 220 亿参数。

*   <strong>视觉 Transformer (Vision Transformer, ViT)</strong>：ViT 是一种将 Transformer 架构应用于图像识别任务的模型。它将图像分割成固定大小的图像块 (patches)，并将这些图像块视为序列中的词元，然后通过 Transformer 编码器进行处理。

*   <strong>多层感知器 (Multi-Layer Perceptron, MLP)</strong>：一种前馈神经网络，由至少三层节点（输入层、隐藏层和输出层）组成，层与层之间通过激活函数连接，能够学习复杂的非线性关系。在 Qwen3-VL 中，MLP 用于将视觉编码器输出的特征压缩和映射到 LLM 的隐藏维度空间。

*   <strong>旋转位置编码 (Rotary Positional Embedding, RoPE)</strong>：一种用于 Transformer 模型的位置编码方法，它通过在自注意力 (self-attention) 计算中旋转查询 (query) 和键 (key) 向量来注入位置信息，从而使模型能够理解序列中词元的位置关系。

*   <strong>增强的交错式 MRoPE (Enhanced Interleaved MRoPE)</strong>：这是 Qwen3-VL 提出的一种改进版 RoPE，用于更好地处理图像和视频中的时空位置信息。它通过在嵌入维度上均匀分布时间 (temporal, t)、水平 (horizontal, h) 和垂直 (vertical, w) 组件，来解决传统 MRoPE 在长视频中频率频谱不平衡的问题，从而提供更准确的位置表示。

*   **DeepStack**：一种视觉-语言对齐机制，通过将视觉编码器不同层的特征注入到语言模型对应的层中，实现多层级的视觉信息融合，从而加强视觉和语言之间的对齐。

*   <strong>链式思考 (Chain-of-Thought, CoT) 推理</strong>：一种提示策略，鼓励大型语言模型在给出最终答案之前，显式地生成一系列中间推理步骤。这有助于模型处理复杂问题，提高其推理能力和结果的可解释性。

*   <strong>监督微调 (Supervised Fine-Tuning, SFT)</strong>：在预训练模型的基础上，使用带有标签的特定任务数据对模型进行进一步训练，使其适应特定任务。

*   <strong>知识蒸馏 (Knowledge Distillation)</strong>：一种模型压缩技术，通过训练一个小型“学生模型”来模仿一个大型“教师模型”的行为，从而将教师模型的知识转移到学生模型中。

*   <strong>强化学习 (Reinforcement Learning, RL)</strong>：一种机器学习范式，智能体通过与环境交互，根据奖励信号调整其行为策略，以最大化累积奖励。在 LLM 领域，RL 被用于进一步优化模型对齐人类偏好和提升性能，例如通过人类反馈强化学习 (Reinforcement Learning from Human Feedback, RLHF)。

## 3.2. 前人工作
Qwen3-VL 构建在 Qwen 系列模型的基础上，特别是 Qwen2.5-VL (Bai et al., 2025)。以下是论文中提及的一些关键前人工作及其对 Qwen3-VL 的影响：

*   **Qwen2.5-VL (Bai et al., 2025)**：Qwen3-VL 的直接前身。它引入了 MRoPE (Multimodal Rotary Positional Embedding) 作为文本和视觉统一的位置编码方案。Qwen3-VL 在此基础上进一步改进，提出了 `interleaved-MRoPE`，解决了 Qwen2.5-VL 在长视频理解中 `MRoPE` 频率频谱不平衡的问题。Qwen3-VL 也沿用了 Qwen2.5-VL 的三模块架构（视觉编码器、MLP-based 视觉-语言合并器、LLM）。在 SFT 阶段，Qwen3-VL 也利用了 Qwen2.5-VL-32B 模型来模拟视觉智能体的行为。

*   <strong>Qwen3 系列 (Yang et al., 2025a)</strong>：Qwen3-VL 的语言主干网络基础。Qwen3-VL 的 LLM 变体（2B/4B/8B/32B 密集型和 30B-A3B/235B-A22B MoE）都建立在 Qwen3 主干网络之上，并继承了其在纯文本理解方面的强大能力。Qwen3 的代码语料库和推理数据也被纳入 Qwen3-VL 的训练数据中。

*   **SigLIP-2 (Tschannen et al., 2025)**：Qwen3-VL 使用 `SigLIP-2` 架构作为其视觉编码器，并基于官方预训练检查点进行继续训练，支持动态输入分辨率。

*   **CoMP (Chen et al., 2025)**：Qwen3-VL 沿用了 `CoMP` 的方法，在处理动态分辨率输入时，使用 2D-RoPE 并根据输入尺寸插值绝对位置嵌入。

*   **DeepStack (Meng et al., 2024)**：Qwen3-VL 借鉴了 `DeepStack` 的思想，将视觉编码器中间层的特征注入到 LLM 的多个层中，以增强视觉-语言对齐。与原 `DeepStack` 堆叠多尺度视觉输入词元不同，Qwen3-VL 提取 ViT 中间层的视觉词元。

*   <strong>基于文本的时间编码策略 (Textual Token-Based Time Encoding Strategy) (Chen et al., 2024b)</strong>：Qwen3-VL 采纳了这种方法来处理视频的时间信息，用文本词元表示时间戳，取代了 Qwen2.5-VL 中基于位置编码的绝对时间对齐。

*   **Omni3D (Brazil et al., 2023)**：Qwen3-VL 在 3D 感知数据集中遵循 `Omni3D` 的方法，将所有数据统一到虚拟相机坐标系中。

*   <strong>“思考型图像”</strong> (Thinking with Images) 的先行工作 (Wu et al., 2025a; Jin et al., 2025; Zheng et al., 2025; Lai et al., 2025)：Qwen3-VL 受到这些工作的启发，通过两阶段训练范式，赋予模型类似的智能体能力，以实现多步、工具集成的推理。

*   **SAPO (Gao et al., 2025)**：Qwen3-VL 在强化学习阶段采用了 `SAPO` 算法，这是一种平滑自适应策略梯度方法，用于在多种文本和多模态任务上实现性能提升。

*   <strong>强到弱蒸馏 (Strong-to-Weak Distillation) (Qwen3)</strong>：Qwen3-VL 沿用了 Qwen3 中描述的强到弱蒸馏流程，以进一步提升轻量级模型的性能。

## 3.3. 技术演进
视觉-语言模型领域的技术演进经历了从早期的特征级融合到如今更深层次的架构集成和推理能力增强。

1.  <strong>早期 VLM (特征级融合)</strong>：最初的 VLM 通常将图像特征（例如通过卷积神经网络 CNN 提取）与文本嵌入简单地拼接起来，然后输入到语言模型中。这种方法在视觉和语言之间建立了浅层的连接，但缺乏深度的交互和推理能力。

2.  <strong>Transformer 架构的引入 (统一编码)</strong>：Transformer 的出现彻底改变了 NLP 领域，随后也被引入到 CV 领域（如 ViT）。VLM 开始探索如何将 Transformer 应用于多模态数据。例如，CLIP 等模型通过对比学习在图像和文本之间建立语义对齐，但通常侧重于理解模态间的对应关系而非复杂推理。

3.  <strong>多模态 Transformer (深层交互)</strong>：随着多模态 Transformer 的发展，视觉和语言词元可以在同一个 Transformer 块内进行交互，从而实现更深层次的跨模态理解。例如，通过引入交叉注意力 (cross-attention) 机制，语言模型可以查询视觉特征，反之亦然。

4.  <strong>位置编码的演进 (时空建模)</strong>：在多模态场景中，如何有效地编码图像和视频中的空间和时间信息成为一个关键问题。早期的模型可能使用简单的绝对或相对位置编码。Qwen2-VL 引入了 `MRoPE`，试图统一处理文本和视觉的位置信息。Qwen3-VL 在此基础上进一步创新，提出了 `interleaved-MRoPE` 来解决 `MRoPE` 在长视频时空建模中的局限性。

5.  <strong>视觉-语言对齐的深化 (多层级融合)</strong>：为了更好地融合视觉和语言信息，模型开始探索将视觉特征注入到 LLM 的多个层中，而不是仅仅在输入层。`DeepStack` (Meng et al., 2024) 提出的概念以及 Qwen3-VL 对其的采纳，代表了这种多层级、更紧密对齐的趋势。

6.  <strong>长上下文理解的突破 (实用性增强)</strong>：随着 LLM 上下文窗口的不断扩大，VLM 也开始追求处理更长的多模态序列。Qwen3-VL 原生支持 256K 词元的交错上下文，并通过专门的预训练阶段和“干草堆中的针”测试，展示了其在长文档和长视频理解方面的强大能力。

7.  <strong>推理和智能体能力的整合 (走向 AGI)</strong>：最新的 VLM 不仅关注感知和理解，还开始整合更复杂的推理能力（如链式思考）和智能体能力（如 GUI 交互、代码生成），使其能够执行更高级、更复杂的现实世界任务。Qwen3-VL 通过引入 `thinking` 模式、链式思考数据和智能体相关数据，体现了这一趋势。

## 3.4. 差异化分析
Qwen3-VL 的方法与相关工作中的主要方法相比，核心区别和创新点体现在以下几个方面：

1.  **架构创新：**
    *   <strong>Interleaved MRoPE (交错式 MRoPE)：</strong> 针对 Qwen2.5-VL 中传统 `MRoPE` 在长视频时空建模中频率频谱不平衡的问题，Qwen3-VL 提出了 `interleaved-MRoPE`。这种方法通过均匀分布时间、水平和垂直组件，更有效地表示时空位置，增强了长视频理解能力。
    *   **DeepStack 集成：** 借鉴 `DeepStack` (Meng et al., 2024)，但进行了扩展。Qwen3-VL 从视觉编码器 (ViT) 的**中间层**提取视觉词元，并注入到 LLM 的**相应层**。这比传统的在输入层融合视觉特征或仅堆叠多尺度视觉输入的方法，提供了更细致、多层次的视觉-语言对齐。
    *   **基于文本的视频时间对齐：** 摒弃了 Qwen2.5-VL 中基于位置编码的绝对时间对齐（T-RoPE），转而采用显式的文本词元来表示时间戳（例如 $<3.0 seconds>$）。这种方法更直观、精确，避免了长视频中稀疏时间位置 ID 带来的问题，并提高了对不同时间码格式的泛化能力。

2.  **长上下文支持：**
    *   Qwen3-VL 原生支持高达 **256K 词元**的交错上下文，远远超过了许多现有 VLM 的上下文窗口。这使得模型能够处理极其冗长和复杂的多模态输入，例如数百页的技术文档和长达两小时的视频，这在当前的 VLM 中是领先的。

3.  **数据和训练策略：**
    *   **全面的数据升级：** Qwen3-VL 对预训练数据进行了大规模、高质量的迭代，不仅增加了图像标题和交错文本-图像数据，还显著扩展了多语言 OCR、文档解析、长文档理解、精细化定位、计数、空间理解、3D 感知、多模态代码和视频数据。尤其值得注意的是，融入了**链式思考推理**和**高质量 GUI 智能体交互数据**，这些对于提升模型的复杂推理和行动能力至关重要。
    *   <strong>平方根重加权 (Square-Root Reweighting)：</strong> 采用这种方法来平衡文本和多模态学习目标，有效提升了多模态性能，同时确保底层 LLM 的纯文本能力不受损害，甚至在某些情况下超越了纯文本主干网络。这解决了多模态训练中常见的一个挑战。
    *   <strong>“思考型”</strong>和“非思考型”变体： 区分并训练了两种类型的模型，其中“思考型”模型通过显式建模推理过程（例如 CoT）来增强在复杂推理任务上的表现。这为不同应用场景提供了灵活的选择。

4.  **性能优势：**
    *   **综合卓越：** Qwen3-VL 在广泛的多模态基准测试（如 MMMU、MathVista、MMStar）和文本基准测试上均实现了领先或最先进的性能，展现了其在通用视觉问答、多模态推理、文档理解和视频理解等方面的强大能力。
    *   **小模型表现出色：** 即使是 2B/4B/8B 等较小规模的密集型模型，也展现出极具竞争力的性能，这表明其架构和训练策略的高效性。
    *   **多语言 OCR 显著提升：** 从 Qwen2.5-VL 的 10 种非英语/中文语言扩展到 39 种语言支持，并在自建测试集上取得了显著的准确率。

        总结来说，Qwen3-VL 的差异化在于其通过**创新的架构设计**来解决时空建模和视觉-语言对齐的深层问题，通过**大规模、高质量的数据迭代和精细化训练策略**来确保长上下文理解和复杂推理能力的全面提升，并最终在**广泛的基准测试中展现出卓越的综合性能**，尤其在保持甚至超越纯文本能力、支持超长上下文和多语言方面具有显著优势。

# 4. 方法论

Qwen3-VL 模型的开发过程主要包括三个核心部分：模型架构、预训练和后训练。

## 4.1. 模型架构
Qwen3-VL 沿袭了 Qwen2.5-VL (Bai et al., 2025) 的三模块架构，包括一个视觉编码器 (vision encoder)、一个基于 MLP 的视觉-语言合并器 (MLP-based vision-language merger) 和一个大型语言模型 (LLM)。Figure 1 展示了详细的模型结构。

### 4.1.1. 大型语言模型 (Large Language Model, LLM)
Qwen3-VL 家族包括三种密集型 (dense) 变体（Qwen3-VL-2B/4B/8B/32B）和两种专家混合 (Mixture-of-Experts, MoE) 变体（Qwen3-VL-30B-A3B, Qwen3-VL-235B-A22B），均构建于 Qwen3 系列主干网络之上 (Yang et al., 2025a)。旗舰模型 Qwen3-VL-235B-A22B 拥有 2350 亿总参数，其中每词元激活 220 亿参数。它在广泛的多模态任务中超越了大多数 VLM，并在大多数语言基准测试中超越了其纯文本对应模型。

### 4.1.2. 视觉编码器 (Vision Encoder)
模型采用 `SigLIP-2` 架构 (Tschannen et al., 2025) 作为视觉编码器，并基于官方预训练检查点进行动态分辨率输入的持续训练。为了有效适应动态分辨率，模型遵循 `CoMP` (Chen et al., 2025) 的方法，使用 2D-RoPE 并根据输入尺寸插值绝对位置嵌入。具体来说，默认使用 `SigLIP2-SO-400M` 变体，对于小型 LLM（2B 和 4B）则使用 `SigLIP2-Large (300M)`。

### 4.1.3. 基于 MLP 的视觉-语言合并器 (MLP-based Vision-Language Merger)
与 Qwen2.5-VL 类似，模型使用一个两层 MLP 将来自视觉编码器的 $2 \times 2$ 视觉特征压缩成一个视觉词元 (visual token)，并与 LLM 的隐藏维度对齐。此外，为支持 `DeepStack` 机制 (Meng et al., 2024)，模型部署了专门的合并器。

### 4.1.4. 架构升级
Qwen3-VL 引入了三项关键架构升级，以增强模型的时空建模和视觉-语言对齐能力。

1.  <strong>增强的交错式 MRoPE (Interleaved MRoPE)</strong>
    Qwen2-VL (Wang et al., 2024c) 引入了 `MRoPE` 来为多模态输入建模位置信息。其原始公式将嵌入维度划分为时间 (temporal, t)、水平 (horizontal, h) 和垂直 (vertical, w) 子空间，每个子空间分配不同的旋转频率。研究发现，这会导致不平衡的频率频谱，并损害长视频理解的性能。为了解决这个问题，Qwen3-VL 重新设计了频率分配，通过在嵌入维度上<strong>交错 (interleaving)</strong> t、h 和 w 组件 (Huang et al., 2025)。这确保了每个时空轴在低频和高频带中均匀表示。由此产生的平衡频谱减轻了原始的频谱偏差，并显著改善了视频的长程位置建模。

2.  **DeepStack**
    Qwen3-VL 从 `DeepStack` (Meng et al., 2024) 中汲取灵感，将视觉词元注入到 LLM 的多个层中。与原始 `DeepStack` 堆叠多尺度视觉输入词元的方法不同，Qwen3-VL 扩展 `DeepStack` 以从视觉 Transformer (ViT) 的**中间层**提取视觉词元。这种设计保留了从低级到高级的丰富视觉信息。具体来说，如 Figure 1 所示，模型从视觉编码器的三个不同层级中选择特征。随后，专门的视觉-语言合并模块将这些多层级特征投影为视觉词元，然后直接添加到 LLM 前三层的相应隐藏状态中。
    以下是 Qwen3-VL 框架的示意图，展示了 DeepStack 的集成以及其他架构组件：

    ```markdown

    ![fig 2](images/2.jpg)
    *该图像是Qwen3-VL的结构示意图，展示了不同输入（图片、视频和文本）在模型解码器中的处理流程。图中包含了多个关键元素，如`11427`个文本标记和视觉编码器的细节，以及DeepStack集成的作用，系统全貌显示了模型的高效性与灵活性。*

    Figure 1: The Qwen3-VL framework integrates a vision encoder and a language model decoder to process multimodal inputs, including text, images, and video. The vision encoder is specifically designed to handle dynamic, native-resolution visual inputs, mapping them to visual tokens of variable length. To enhance perceptual capability and preserve rich visual information, we incorporate the pioneering DeepStack mechanism, which injects visual tokens from multiple layers of the vision encoder into corresponding layers of the LLM. Furthermore, we adopt Interleaved MRoPE to encode positional information for multimodal inputs with a balanced frequency spectrum, and introduce text-based timestamp tokens to more effectively capture the temporal structure of video sequences.
    ```
    *VLM 描述: 该图像是Qwen3-VL的结构示意图，展示了不同输入（图片、视频和文本）在模型解码器中的处理流程。图中包含了多个关键元素，如`11427`个文本标记和视觉编码器的细节，以及DeepStack集成的作用，系统全貌显示了模型的高效性与灵活性。*

3.  <strong>视频时间戳 (Video Timestamp)</strong>
    在 Qwen2.5-VL 中，模型采用 `MRoPE` 的时间同步变体来赋予模型时间感知能力。然而，研究发现这种方法存在两个主要局限性：(1) 将时间位置 ID 直接绑定到绝对时间，对于长视频会产生过大且稀疏的时间位置 ID，损害了模型理解长时序上下文的能力。(2) 在这种方案下有效学习需要对各种帧率 (fps) 进行广泛且均匀的采样，显著增加了训练数据构建的成本。为了解决这些问题，Qwen3-VL 采用了一种**基于文本词元的时间编码策略** (Chen et al., 2024b)，其中每个视频时间块都以格式化的文本字符串作为前缀来表示时间戳，例如 $<3.0 seconds>$。此外，在训练期间，模型会生成秒和 HMS (小时:分钟:秒) 两种格式的时间戳，以确保模型学习解释多样化的时间码表示。尽管这种方法会略微增加上下文长度，但它使模型能够更有效、更精确地感知时间信息，从而促进视频定位和密集字幕等时间感知型视频任务。

## 4.2. 预训练 (Pre-Training)
Qwen3-VL 的预训练方法被系统地分为四个不同的阶段，旨在逐步构建从基本对齐到长上下文理解的能力。

### 4.2.1. 训练方案 (Training Recipe)
首先，通过基于预训练 `SigLIP-2` 模型进行动态分辨率的持续训练来增强视觉编码器。整体 Qwen3-VL 模型采用三模块架构，包括该视觉编码器、基于 MLP 的视觉-语言合并器以及 Qwen3 大语言模型主干网络。

以下是 Qwen3-VL 不同阶段的训练设置和超参数：
以下是原文 Table 1 的结果：

<table><tr><td>Stage</td><td>Objective</td><td>Training</td><td>Token Budget</td><td>Sequence Length</td></tr><tr><td>S0</td><td>Vision-Language Alignment</td><td>Merger</td><td>67B</td><td>8,192</td></tr><tr><td>S1</td><td>Multimodal Pre-Training</td><td>All</td><td>~1T</td><td>8,192</td></tr><tr><td>S2</td><td>Long-Context Pre-Training</td><td>All</td><td>~1T</td><td>32,768</td></tr><tr><td>S3</td><td>Ultra-Long-Context Adaptation</td><td>All</td><td>100B</td><td>262,144</td></tr></table>

*   <strong>阶段 0 (S0): 视觉-语言对齐 (Vision-Language Alignment)</strong>
    *   **目标：** 有效弥合视觉编码器和 LLM 之间的模态鸿沟。
    *   **训练内容：** 在此阶段仅训练 MLP 合并器的参数，而视觉编码器和 LLM 主干网络保持冻结。
    *   **数据：** 使用大约 670 亿词元的精选数据集，包括高质量的图像-标题对、视觉知识集合和光学字符识别 (OCR) 数据。
    *   **序列长度：** 8,192 词元。
    *   **目的：** 这种“先对齐”的方法为跨模态理解奠定了坚实的基础，然后才进行全参数训练。

*   <strong>阶段 1 (S1): 多模态预训练 (Multimodal Pre-Training)</strong>
    *   **目标：** 全参数的多模态预训练。
    *   **训练内容：** 解冻所有模型组件（视觉编码器、合并器和 LLM），进行联合端到端训练。
    *   **数据：** 在大约 1 万亿词元的大规模多样化数据集上进行训练。为保持 LLM 强大的语言能力，数据混合物由视觉-语言 (VL) 数据和纯文本数据组成。VL 部分丰富多样，增加了交错图像-文本文档、视觉定位任务、视觉问答 (VQA)、STEM 领域数据以及少量视频数据以引入时间理解。
    *   **序列长度：** 8,192 词元。

*   <strong>阶段 2 (S2): 长上下文预训练 (Long-Context Pre-Training)</strong>
    *   **目标：** 显著扩展模型的上下文处理能力。
    *   **训练内容：** 序列长度翻四倍至 32,768 词元，所有模型参数继续可训练。
    *   **数据：** 在大约 1 万亿词元的数据集上进行训练，并调整数据混合以支持长上下文任务。纯文本数据的比例增加，以加强长篇文本理解，而剩余的 VL 数据则包含显著增加的视频和面向智能体的指令遵循数据。
    *   **目的：** 此阶段对于使模型能够处理和推理更长的视频和复杂的多步任务至关重要。

*   <strong>阶段 3 (S3): 超长上下文适应 (Ultra-Long-Context Adaptation)</strong>
    *   **目标：** 将模型的上下文窗口推至其操作极限。
    *   **训练内容：** 序列长度大幅增加到 262,144 词元。
    *   **数据：** 在一个更专注的 1000 亿词元数据集上进行训练，该数据集专门为此目的而精选。数据也由纯文本数据和 VL 数据组成，重点关注长视频和长文档理解任务。
    *   **目的：** 这种最终适应巩固了 Qwen3-VL 在处理和分析极长序列输入方面的熟练程度，这是全面文档分析和长时间视频摘要等应用的关键能力。

### 4.2.2. 预训练数据 (Pre-Training Data)
为了构建一个更强大和鲁棒的视觉-语言基础模型，Qwen3-VL 全面修改了训练数据，提升了质量、多样性和结构。关键升级包括：增强的字幕监督、扩展的全方位识别和 OCR 覆盖、带 3D/空间推理的标准化定位、以及用于代码、长文档和时间定位视频的新语料库。此外，还注入了链式思考推理和高质量、多样化的 GUI 智能体交互数据，以弥合感知、推理和行动之间的鸿沟。

1.  <strong>图像标题和交错文本-图像数据 (Image Caption and Interleaved Text-Image Data)</strong>
    *   **图像标题数据：** 收集了大规模当代多语言（主要是中文-英文）图像-文本对语料库，并应用多阶段精炼管道。核心是使用一个专门为重新生成字幕而微调的 Qwen2.5-VL-32B 模型。该模型利用与图像相关的原始文本生成更全面、流畅和细粒度的字幕，丰富了对视觉元素（如物体属性、空间布局和上下文语义）的描述，同时提高了文本部分的语言质量和信息量。
        *   **去重：** 仅在重新生成的字幕文本上使用语义相似性指标进行去重，确保在不牺牲视觉多样性的前提下移除冗余样本。
        *   **增强：** 通过对视觉嵌入进行聚类 (Johnson et al., 2019; Douze et al., 2024; Diao et al., 2025) 来识别数据分布中的稀疏区域，并进行有针对性的数据增强，以覆盖未充分表示的概念。
    *   **交错文本-图像数据：** 收集了来自近期中英文网站的各种真实世界多模态文档 (Laurencon et al., 2023; Zhu et al., 2023; Li et al., 2024c)。所有文档都使用一个轻量级、为细粒度领域识别而微调的 Qwen-based 评分器 (Wettig et al., 2025) 进行领域分类。通过相同的评分器过滤掉广告、促销内容和点击诱饵等有害或低价值类别。
        *   **书籍级交错数据：** 使用微调的 Qwen2.5-VL-7B 模型进行高精度多模态解析，精确提取文本并与嵌入的图表、示意图和照片对齐。为实现超长上下文建模，通过合并连续页面构建了一个专门的子集，形成长达 256K 词元的序列，保留了自然的页面顺序和多模态连贯性。
        *   **质量控制：** 在预处理期间强制执行严格的质量控制：(i) 移除纯文本或低对齐片段；(ii) 对于超长书籍序列，要求最小页数和最小图像-文本比率，以确保整个上下文中的有意义的视觉-文本交互。

2.  <strong>知识 (Knowledge)</strong>
    为了让 Qwen3-VL 全面掌握真实世界和虚构概念，模型构建了一个大规模的预训练数据集，该数据集围绕定义明确的实体，涵盖十多个语义类别，包括动物、植物、地标、食物以及车辆、电子产品和服装等日常物品。
    *   **解决长尾分布：** 真实世界实体遵循长尾分布，为此采用了基于重要性的采样策略：高显著性实体被更频繁地采样以确保足够的学习信号，而低显著性实体以较小比例包含以保持广泛覆盖。
    *   **多阶段精炼：** 所有保留的样本都经过多阶段精炼管道。除了标准的噪声和错位过滤外，还用更丰富的 LLM 生成的描述替换了原始或稀疏的标题（如通用 `alt-text`）。这些增强的标题不仅识别主要实体，还描述其视觉属性、周围上下文、空间布局以及与其他物体或人的交互。

3.  <strong>OCR、文档解析和长文档理解 (OCR, Document Parsing and Long Document Understanding)</strong>
    *   **OCR：** 为了增强真实世界图像上的 OCR 性能，模型使用粗-到-细管道构建了一个 3000 万个内部收集样本的数据集。该管道通过整合 OCR 专用模型的伪标签和 Qwen2.5-VL 的精炼来优化 OCR 注释，无需人工标注。除了 Qwen2.5-VL 支持的 10 种语言（不包括中文和英文），Qwen3-VL 还新增了 29 种语言，合成了约 3000 万个高质量多语言 OCR 样本，并整理了超过 100 万张内部真实世界多语言图像。
    *   **文档解析：** 收集了来自 Common Crawl 的 300 万 PDF 文件，均匀分布在 10 种文档类型中（每种 30 万样本），以及 400 万内部文档。内部布局模型首先预测文本和非文本区域的阅读顺序和边界框；然后 Qwen2.5-VL-72B 进行区域特定识别。输出被重新组装成位置感知、布局对齐的解析数据。
        *   **统一注释框架：** 为确保在异构格式中进行鲁棒解析，设计了一个支持两种表示的统一注释框架：
            *   `QwenVL-HTML`：包含细粒度的元素级边界框。
            *   `QwenVL-Markdown`：仅定位图像和表格，表格以 LaTeX 编码。
        *   构建了一个大规模的合成 HTML 语料库，包含精确注释，并系统地将其转换为 Markdown 格式。
    *   **长文档理解：** 为增强模型理解多页 PDF（通常跨越数十页）的能力，模型利用了大规模长文档数据语料库。
        *   **合成长文档解析序列：** 通过合并单页文档样本来合成。在每个序列中，多个页面图像放置在开头，随后是其相应的文本（源自 OCR 或 HTML 解析）。
        *   <strong>长文档视觉问答 (VQA) 数据：</strong> 采样高质量多页 PDF，并生成多样化的 VQA 示例，要求模型跨多页和异构文档元素（如图表、表格、图形和正文）进行推理。

4.  <strong>定位和计数 (Grounding and Counting)</strong>
    视觉定位 (Visual grounding) 是多模态模型的基本能力，使其能够准确识别、解释和定位从特定对象到任意图像区域的广泛视觉目标。在 Qwen3-VL 中，系统地增强了定位能力，并支持两种定位模态：<strong>边界框 (bounding boxes)</strong> 和 <strong>点 (points)</strong>。此外，还将模型的定位能力扩展到支持计数 (counting)，从而实现对视觉实体的定量推理。
    *   <strong>基于框的定位 (Box-based Grounding)：</strong> 聚合了广泛使用的开源数据集，包括 COCO (Lin et al., 2014)、Objects 365 (Shao et al., 2019)、OpenImages (Kuznetsova et al., 2020) 和 RefCOCO+/g (Kazemzadeh et al., 2014; Mao et al., 2016)。此外，开发了一个自动化合成管道，使用 `Grounding DINO` (Liu et al., 2023a) 和 Qwen2.5-VL 生成高质量的对象注释。
    *   <strong>基于点的定位 (Point-based Grounding)：</strong> 整理了一个综合数据集，结合了公开可用和合成生成的指向注释。它整合了三个来源：(i) 来自 PixMo (Deittek et al., 2024) 的公开指向和计数注释；(ii) 从公共对象检测和实例分割基准派生的对象定位数据；(iii) 由专用合成管道生成的高精度指向注释。
    *   **计数：** 在定位数据的基础上，整理了一个高质量子集，形成计数数据集的基础，其中包括三种不同的任务形式：直接计数、基于框的计数和基于点的计数。
    *   **坐标系统：** 与 Qwen2.5-VL 不同，Qwen3-VL 采用了一个缩放到 [0, 1000] 范围的标准化坐标系统。

5.  <strong>空间理解和 3D 识别 (Spatial Understanding and 3D Recognition)</strong>
    *   **空间理解：** 除了定位对象，Qwen3-VL 还被训练来推理 2D 场景中的空间关系、对象功能 (affordances) 和可行行动，这些能力对于具身 AI (embodied AI) 和交互式应用至关重要。为此，构建了一个专门的数据集，包含了关系注释（如“杯子在笔记本电脑左边”）、功能标签（如“可抓取”、“可按压”、“可坐”）以及需要规划的行动条件查询。所有空间引用都相对于其他对象或场景帧表达，而非绝对坐标。
    *   <strong>3D 定位 (3D grounding)：</strong> 构建了一个专门用于 3D 视觉定位的预训练数据集，数据来源于公共室内外场景集合，并重新格式化为视觉问答形式。每个样本包括：1) 单视图相机图像，2) 自然语言指称表达式，3) 结构化 JSON 格式的相应 9-自由度 (DoF) 3D 边界框注释。数据统一到虚拟相机坐标系 (Brazil et al., 2023)，并合成了大量描述性字幕以创建丰富的文本查询。

6.  <strong>代码 (Code)</strong>
    通过将两类代码相关数据纳入训练语料库，增强了 Qwen3-VL 系列的专用编码能力，使模型能够在纯文本和视觉定位的上下文中阅读、编写和推理程序。
    *   **纯文本编码：** 重用了来自 Qwen3 和 Qwen3-Coder 系列的广泛代码语料库。
    *   **多模态编码：** 为需要视觉理解和代码生成的任务整理数据，涵盖了多个关键任务：将 UI 截图转换为响应式 HTML/CSS；从图像生成可编辑的 SVG 代码 (Li et al., 2025c)；解决视觉编程挑战 (Li et al., 2024a)；回答多模态编码问题（如带图像的 StackOverflow 帖子）；以及将可视化表示（如流程图、图表和 LaTeX 方程）转录为其各自的代码或标记。

7.  <strong>视频 (Video)</strong>
    Qwen3-VL 的视频理解能力得到了实质性提升，实现了对帧间时间动态的鲁棒建模、对空间关系的细粒度感知以及超长视频序列的连贯摘要。
    *   <strong>时间感知视频理解 (Temporal-Aware Video Understanding)：</strong>
        *   **密集字幕合成：** 对于长视频序列，采用短-长字幕合成策略，生成整体、时间交错且时间连贯的故事级描述。
        *   **时空视频定位：** 整理并合成了大规模的视频数据，在对象、动作和人物层面进行标注。
    *   **视频数据平衡和采样：**
        *   **来源平衡：** 组建了一个包含各种视频来源（教学内容、电影、自我中心录像等）的大规模数据集。
        *   **长度自适应采样：** 在预训练阶段，根据不同的序列长度约束，动态调整采样参数，如每秒帧数 (fps) 和最大帧数。

8.  <strong>科学、技术、工程和数学 (STEM) (Science, Technology, Engineering, and Mathematics)</strong>
    多模态推理是 Qwen3-VL 的核心，STEM 推理是其最基本的部分。模型遵循“分而治之”的策略：首先独立开发细粒度视觉感知和鲁棒语言推理能力，然后以协同方式整合它们以实现有效的多模态推理。
    *   **视觉感知数据：** 开发了专用合成数据生成管道，通过编程（基于代码）渲染构建几何图表，生成了：(i) 100 万个点定位样本（如交点、角点和重心）；(ii) 200 万个面向感知的视觉问答对。
    *   **多模态推理数据：** 大部分多模态推理数据由 6000 多万个 K-12 和本科级别的练习组成。在质量过滤和重新格式化后，合成超过 1200 万个带有图像的多模态推理样本，并与原始强推理模型生成的链式思考 (CoT) 轨迹配对。通过拒绝采样 (rejection sampling) 仅保留具有挑战性的问题。
    *   **语言推理数据：** 除了多模态推理数据，还纳入了来自 Qwen3 的推理数据，因为多模态推理能力很大程度上来源于语言推理能力。

9.  <strong>智能体 (Agent)</strong>
    *   **GUI：** 为赋予 Qwen3-VL 自主与图形用户界面 (GUI) 交互的智能体能力，整理并合成了大规模、跨平台数据，涵盖桌面、移动和网络环境 (Ye et al., 2025; Wang et al., 2025a; Lu et al., 2025)。
        *   **GUI 界面感知：** 利用元数据、解析工具和人工注释构建任务，如元素描述、密集字幕和密集定位。
        *   **智能体能力：** 通过自我演进的轨迹生成框架组装多步任务轨迹，并辅以有针对性的人工审计；精心设计和增强链式思考理由，以加强现实世界执行中的规划、决策和反思性自我纠正。
    *   <strong>函数调用 (Function Calling)：</strong> 为支持多模态上下文中的通用函数调用能力，构建了一个多模态函数调用轨迹合成管道。
    *   <strong>搜索 (Search)：</strong> 在通用函数调用能力中，将执行搜索的能力视为促进现实世界中长尾实体知识整合的关键。

## 4.3. 后训练 (Post-Training)
后训练管道是一个三阶段过程，旨在完善模型的指令遵循能力、增强其推理能力并使其与人类偏好对齐。

### 4.3.1. 训练方案 (Training Recipe)

*   <strong>监督微调 (Supervised Fine-Tuning, SFT)</strong>
    *   **目的：** 赋予指令遵循能力并激活潜在推理技能。
    *   **阶段：** 分两个阶段：初始阶段在 32K 上下文长度下进行，随后扩展到 256K 上下文窗口，重点关注长文档和长视频数据。
    *   **变体：** 为满足不同需求，将训练数据分为适用于“非思考型 (non-thinking)”模型的标准格式和适用于“思考型 (thinking)”模型的链式思考 (CoT) 格式，后者显式建模推理过程。

*   <strong>强到弱蒸馏 (Strong-to-Weak Distillation)</strong>
    *   **目的：** 强大的教师模型将其能力传递给学生模型。
    *   **方式：** 使用纯文本数据对 LLM 主干网络进行微调。
    *   **效果：** 这种方法在以文本为中心和多模态任务的推理能力方面产生了显著改进。

*   <strong>强化学习 (Reinforcement Learning, RL)</strong>
    *   **目的：** 进一步增强模型性能和对齐。
    *   **阶段：** 分为推理强化学习 (Reasoning RL) 和通用强化学习 (General RL)。
    *   **方式：** 在一套全面的文本和多模态领域（包括但不限于数学、OCR、定位和指令遵循）应用大规模强化学习，以改进更细粒度的能力。

### 4.3.2. 冷启动数据 (Cold Start Data)
*   <strong>SFT 数据 (SFT Data)</strong>
    *   **目的：** 赋予模型处理广泛真实世界场景的能力。
    *   **范围：** 在 Qwen2.5-VL 基础上，战略性地扩展了功能范围，引入了新能力，包括具身智能的空间推理、细粒度视觉理解的图像基础推理、视频中的时空定位以及数百页长技术文档的理解。
    *   **数据构成：** 约 1,200,000 个样本，分为单模态和多模态数据，其中三分之一是纯文本条目，其余三分之二是图像-文本和视频-文本对。
    *   **多语言和对话：** 包含多语言样本，模拟真实的对话动态，包括单轮和多轮对话，上下文涵盖单图像到多图像序列。
    *   **长上下文训练策略：** 采用分阶段训练策略：初始阶段进行一个 epoch 的 32K 词元序列长度训练，随后第二个 epoch 在完整的 256K 词元长度下进行。
    *   **数据过滤：** 实施严格的数据过滤协议，包括<strong>查询过滤 (Query Filtering)</strong> 和<strong>响应过滤 (Response Filtering)</strong>，以消除低质量、冗余或不相关的样本。
        *   **查询过滤：** 利用 Qwen2.5-VL 识别和丢弃不可验证的查询；修订模糊指令；消除无实质内容的网络查询。
        *   **响应过滤：** 结合<strong>基于规则的过滤 (Rule-Based Filtering)</strong>（消除重复、不完整或格式不当的响应，并过滤掉不道德内容）和<strong>基于模型的过滤 (Model-Based Filtering)</strong>（使用来自 Qwen2.5-VL 系列的奖励模型对多模态问答对进行多维度评估，包括正确性、完整性、清晰度和帮助性，并特别强调对视觉信息的准确解释和利用）。

*   <strong>长链式思考冷启动数据 (Long-CoT Cold Start Data)</strong>
    *   **目的：** 旨在引发和完善复杂推理能力。
    *   **数据构成：** 建立在查询的多元集合上，涵盖纯文本和多模态数据，保持视觉-语言和纯文本样本大约 1:1 的比例。
        *   **多模态组件：** 涵盖 VQA、OCR、2D/3D 定位和视频分析等既定领域，特别强调 STEM 和智能体工作流相关任务。
        *   **纯文本部分：** 密切反映 Qwen3 使用的数据，包含数学、代码生成、逻辑推理和通用 STEM 中的难题。
    *   **过滤协议：** 实施严格的多阶段过滤协议：
        *   **难度策划：** 有选择地保留基线模型通过率较低或生成更长、更详细响应的实例。
        *   **多模态必要性过滤：** 对于视觉-语言数学问题，过滤掉 Qwen3-30B-nothink 模型无需视觉输入即可正确解决的样本。
        *   **响应质量控制：** 净化生成的响应，移除不正确的最终结果和不良模式（如过度重复、语言混合不当或没有足够推理步骤的猜测）。

### 4.3.3. 强到弱蒸馏 (Strong-to-Weak Distillation)
模型采用 Qwen3 中描述的强到弱蒸馏管道来进一步提升轻量级模型的性能。该蒸馏过程包括两个主要阶段：
*   <strong>离线策略蒸馏 (Off-policy Distillation)：</strong> 在第一阶段，结合教师模型生成的输出进行响应蒸馏。这有助于轻量级学生模型获得基本推理能力，为后续的在线策略训练奠定坚实基础。
*   <strong>在线策略蒸馏 (On-policy Distillation)：</strong> 在第二阶段，学生模型根据提供的提示生成响应。这些在线策略序列用于微调学生模型。通过最小化 KL 散度 (KL divergence) 来对齐学生和教师模型预测的对数几率 (logits)。

### 4.3.4. 强化学习 (Reinforcement Learning)
*   <strong>推理强化学习 (Reasoning Reinforcement Learning)</strong>
    *   **训练任务：** 在一系列文本和多模态任务上训练模型，包括数学、编码、逻辑推理、视觉定位和视觉谜题。每个任务都设计为解决方案可以通过规则或代码执行器确定性验证。
    *   **数据准备：** 整理来自开源和专有来源的训练数据，并进行严格的预处理和人工标注以确保高质量的 RL 查询。对于多模态查询，使用最先进的视觉-语言模型 (Qwen3-VL-235B-A22B) 的初步检查点对每个查询采样 16 个响应；丢弃所有响应都不正确的查询。
    *   **奖励系统：** 实施统一的奖励框架，为所有任务提供精确反馈。核心奖励逻辑按任务实现。当响应语言与提示语言不同时，施加惩罚以减轻代码切换 (code-switching)。
    *   **RL 算法：** 采用 `SAPO` (Gao et al., 2025)，这是一种平滑自适应策略梯度方法。

*   <strong>通用强化学习 (General Reinforcement Learning)</strong>
    *   **目的：** 增强模型的泛化能力和操作鲁棒性。
    *   **奖励函数：** 基于 SFT 阶段的一套综合任务（包括 VQA、图像字幕、OCR、文档解析、定位和时钟识别）制定。奖励机制旨在优化模型性能的两个主要维度：
        *   <strong>指令遵循 (Instruction Following)：</strong> 评估模型对明确用户指令的遵守情况。
        *   <strong>偏好对齐 (Preference Alignment)：</strong> 对于开放式或主观查询，通过优化帮助性、事实准确性和文体恰当性，使模型输出与人类偏好对齐。
    *   **纠正机制：** 作为纠正机制，引入专门的、可验证的任务来触发在 SFT 阶段形成的强但有缺陷的知识先验 (knowledge priors)，例如反直觉的对象计数和复杂时钟时间识别。
    *   **劣质行为缓解：** 为缓解不当语言混合、过度重复和格式错误等劣质行为，专门策划了一个数据集，隔离已知会引发此类不良行为的提示。
    *   **反馈系统：** 通过混合奖励系统提供 RL 过程的反馈，该系统结合了<strong>基于规则的奖励 (Rule-Based Rewards)</strong>（提供 unambiguous、高精度的反馈，用于具有可验证真值的任务）和<strong>基于模型的奖励 (Model-Based Rewards)</strong>（使用 Qwen2.5-VL-72B-Instruct 或 Qwen3 作为复杂的评判模型，评估生成响应的质量）。

### 4.3.5. 思考型图像 (Thinking with Images)
受“思考型图像”方面先行工作 (Wu et al., 2025a; Jin et al., 2025; Zheng et al., 2025; Lai et al., 2025) 的启发，Qwen3-VL 通过两阶段训练范式赋予了类似的智能体能力。

*   **第一阶段：** 合成了一个包含约 10K 定位示例的冷启动生成数据集，主要是简单的两轮视觉问答任务（如属性检测）。然后对 Qwen2.5-VL-32B 进行监督微调 (SFT)，以模拟视觉智能体的行为：思考→行动→分析反馈→回答。为进一步增强推理能力，应用了多轮、工具集成强化学习 (RL)。

*   **第二阶段：** 将第一阶段训练好的 Qwen2.5-VL-32B 视觉智能体进行蒸馏，生成一个更大、更多样化的约 120K 多轮智能体交互数据集，涵盖更广泛的视觉任务。然后应用类似的冷启动 SFT 和工具集成 RL 管道（现在使用蒸馏和合成数据）进行 Qwen3-VL 的后训练。

*   **奖励信号：** 在 RL 期间，采用三个互补的奖励信号来鼓励鲁棒的、工具介导的推理：
    *   <strong>答案准确性奖励 (Answer Accuracy Reward)：</strong> 利用 Qwen3-32B 衡量最终答案是否正确。
    *   <strong>多轮推理奖励 (Multi-Turn Reasoning Reward)：</strong> 利用 Qwen2.5-VL-72B 评估智能体是否正确解释了工具或环境反馈，并通过连贯的、逐步的推理得出答案。
    *   <strong>工具调用奖励 (Tool-Calling Reward)：</strong> 通过将实际工具调用次数与专家估计的目标（由 Qwen2.5-VL-72B 离线根据任务复杂性确定）进行比较，鼓励适当的工具使用。早期实验发现模型倾向于只进行一次工具调用以“破解”前两个奖励，而不顾任务需求，因此引入工具调用奖励以促进适应性工具探索。

## 4.4. 基础设施 (Infrastructure)
Qwen3-VL 系列模型的训练在阿里云 PAI-灵骏 AI 计算服务上进行。
*   **预训练阶段：** 系统采用基于 `Megatron-LM` 框架的混合并行策略，集成了张量并行 (Tensor Parallelism, TP)、管道并行 (Pipeline Parallelism, PP)、上下文并行 (Context Parallelism, CP)、专家并行 (Expert Parallelism, FP) 和 `ZeRO-1` 数据并行 (Data Parallelism, DP)。这种配置在模型规模、计算负载和通信开销之间实现了细粒度平衡，即使在高达 10,000 个 GPU 的规模下，也能实现高硬件利用率并保持高吞吐量和低通信延迟。
*   **本地部署和性能评估：** 采用基于 `vLLM` 或 `SGLang` 的部署策略。`vLLM` 利用 PagedAttention 实现内存高效管理和高吞吐量推理，而 `SGLang` 擅长结构化生成和处理复杂提示。

# 5. 实验设置

## 5.1. 数据集
Qwen3-VL 在预训练和后训练阶段使用了大规模、多样化且经过精心策划的数据集，涵盖了文本、图像和视频等多种模态。在评估阶段，模型在广泛的公开基准测试上进行了测试，这些基准测试覆盖了通用视觉问答、多模态推理、对齐和主观任务、文本识别和文档理解、2D/3D 定位、细粒度感知、多图像理解、具身和空间理解、视频理解、智能体和纯文本任务。

**预训练阶段使用的数据集类型：**
1.  **图像标题数据：** 大规模当代多语言（主要是中文-英文）图像-文本对语料库，通过 Qwen2.5-VL-32B 模型进行 recaptioning 增强。
2.  **交错文本-图像数据：** 来自中英文网站的真实世界多模态文档，经过领域分类和过滤。书籍级数据通过 Qwen2.5-VL-7B 模型进行解析，并合并为长达 256K 词元的序列。
3.  **知识数据：** 大规模预训练数据集，围绕定义明确的实体，涵盖动物、植物、地标、食物、日常物品等类别，通过重要性采样和 LLM 生成的描述进行增强。
4.  **OCR 数据：** 3000 万个内部收集样本，涵盖 39 种语言，通过 OCR 专用模型和 Qwen2.5-VL 的伪标签进行精炼。
5.  **文档解析数据：** 300 万 PDF 文件来自 Common Crawl，以及 400 万内部文档，通过内部布局模型和 Qwen2.5-VL-72B 进行解析，生成 `QwenVL-HTML` 和 `QwenVL-Markdown` 格式。
6.  **长文档理解数据：** 合成的长文档解析序列和长文档 VQA 数据，要求模型跨多页和异构元素进行推理。
7.  **定位和计数数据：**
    *   **基于框的定位：** COCO (Lin et al., 2014), Objects 365 (Shao et al., 2019), OpenImages (Kuznetsova et al., 2020), RefCOCO+/g (Kazemzadeh et al., 2014; Mao et al., 2016)，以及通过 `Grounding DINO` (Liu et al., 2023a) 和 Qwen2.5-VL 合成的数据。
    *   **基于点的定位：** PixMo (Deittek et al., 2024) 的公开注释，以及从公共对象检测和实例分割基准派生的数据，和专用合成管道生成的高精度注释。
    *   **计数：** 从定位数据中精选的高质量子集，包括直接计数、基于框的计数和基于点的计数任务。
8.  **空间理解和 3D 识别数据：**
    *   **空间理解：** 包含关系注释、功能标签和行动条件查询的专门数据集，来源于真实场景和合成布局。
    *   **3D 定位：** 来自公共室内外场景集合的数据，格式化为视觉问答形式，包含 9-自由度 (DoF) 3D 边界框注释。
9.  **代码数据：**
    *   **纯文本编码：** Qwen3 和 Qwen3-Coder 系列的广泛代码语料库。
    *   **多模态编码：** 来自开源数据集和内部合成管道的数据，用于 UI 截图转 HTML/CSS、图像转 SVG (Li et al., 2025c)、视觉编程挑战 (Li et al., 2024a) 和多模态编码问答。
10. **视频数据：** 包含教学内容、电影、自我中心录像等各种来源，通过密集字幕合成和时空视频定位进行增强。
11. **STEM 数据：** 细粒度视觉感知数据（通过编程渲染生成的几何图表、点定位样本、感知导向 VQA）和多模态推理数据（6000 多万 K-12 和本科级练习，1200 多万 CoT 样本），以及 Qwen3 的语言推理数据。
12. **智能体数据：**
    *   **GUI：** 大规模、跨平台数据，涵盖桌面、移动和网络环境 (Ye et al., 2025; Wang et al., 2025a; Lu et al., 2025)，用于元素描述、密集字幕和密集定位，以及多步任务轨迹和 CoT 理由。
    *   **函数调用：** 多模态函数调用轨迹合成管道生成的数据。
    *   **搜索：** 多模态事实查找轨迹数据，使用在线图像和文本搜索工具。

**后训练阶段使用的数据集类型：**
1.  **SFT 数据：** 约 1,200,000 个样本，三分之一纯文本，三分之二图像-文本和视频-文本对。包含多语言、单轮和多轮对话，支持高级智能体行为。
2.  <strong>长链式思考 (Long-CoT) 冷启动数据：</strong> 纯文本和多模态数据，重点关注 STEM 和智能体工作流，通过难度策划、多模态必要性过滤和响应质量控制进行筛选。
3.  **强化学习数据：** 约 30K RL 查询，涵盖数学、编码、逻辑推理、视觉定位和视觉谜题等文本和多模态任务。以及用于通用 RL 的 SFT 任务（VQA、图像字幕、OCR 等）和专门的验证任务。

**评估阶段使用的基准测试：**
（详细列表请参考 `A Benchmarks` 部分，此处仅列举一些代表性示例）
*   **多模态推理：** MMMU (Yue et al., 2024a), MMMU-Pro (Yue et al., 2024b), MathVision (Wang et al., 2024b), MathVista (Lu et al., 2023), We-Math (Qiao et al., 2024), LogicVista (Xiao et al., 2024), VisualPuzzles (Song et al., 2025b)。
*   **通用视觉问答：** MMBench-V1.1 (Liu et al., 2023b), RealWorldQA (xAI, 2024), MMStar (Chen et al., 2024a), SimpleVQA (Cheng et al., 2025)。
*   **对齐和主观任务：** MM-MT-Bench (Agrawal et al., 2024), HallusionBench (Guan et al., 2023), MIA-Bench (Qian et al., 2024)。
*   **文本识别和文档理解：** DocVQA (Mathew et al., 2021b), InfoVQA (Mathew et al., 2021a), AI2D (Kembhavi et al., 2016), ChartQA (Masry et al., 2022), OCRBench (Liu et al., 2024), CC-OCR (Yang et al., 2024b), MMLongBench-Doc (Ma et al., 2024)。
*   **2D/3D 定位和空间理解：** RefCOCO/+/g (Kazemzadeh et al., 2014; Mao et al., 2016), ODinW-13 (Li et al., 2022), CountBench (Paiss et al., 2023), Omni3D (Brazil et al., 2023) 包含 ARKitScenes (Baruch et al., 2021), Hypersim (Roberts et al., 2021), SUN RGB-D (Song et al., 2015), ERQA (Team et al., 2025), VSI-Bench (Yang et al., 2025b), EmbSpatial (Du et al., 2024), RefSpatial (Zhou et al., 2025), RoboSpatialHome (Song et al., 2025a)。
*   **视频理解：** VideoMME (Fu et al., 2024a), MVBench (Li et al., 2024b), VideoMMU (Hu et al., 2025), MMVU (Zhao et al., 2025), LVBench (Wang et al., 2024d), MLVU (Zhou et al., 2024), Charades-STA (Gao et al., 2017)。
*   **多图像理解：** BLINK (Fu et al., 2024c), MuirBench (Wang et al., 2024a)。
*   **代码：** Design2Code (Si et al., 2025), ChartMimic (Yang et al., 2024a), UniSVG (Li et al., 2025a)。
*   **GUI 智能体：** ScreenSpot (Cheng et al., 2024), ScreenSpot Pro (Li et al., 2025b), OSWorldG (Xie et al., 2025a), AndroidWorld (Rawles et al., 2024), OSWorld (Xie et al., 2025c,b)。
*   **纯文本任务：** MMLU-Pro (Wang et al., 2024f), GPQA (Rein et al., 2023), AIME-25 (AIME, 2025), LiveCodeBench v6 (Jain et al., 2024), IEFval (Zhou et al., 2023), MultiIF (He et al., 2024), PolyMATH (Wang et al., 2025b)。

## 5.2. 评估指标
论文中使用了多种评估指标，以全面衡量模型在不同任务上的性能。以下是针对论文中提及的主要评估指标的定义和说明：

1.  <strong>准确率 (Accuracy)</strong>
    *   **概念定义：** 准确率是分类任务中最常见的评估指标之一，它衡量模型正确预测的样本数占总样本数的比例。对于多模态问答、多项选择题或分类任务，准确率直接反映了模型回答或分类的正确性。
    *   **数学公式：**
        $$
        \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
        $$
    *   **符号解释：**
        *   `Number of Correct Predictions`：模型做出正确预测的样本数量。
        *   `Total Number of Predictions`：总共进行预测的样本数量。

2.  <strong>平均精度 (Mean Average Precision, mAP)</strong>
    *   **概念定义：** `mAP` 是目标检测 (Object Detection) 和图像检索 (Image Retrieval) 任务中常用的评估指标，衡量模型在不同召回率 (recall) 下的平均精度 (precision)。它通过计算每个类别的平均精度 (Average Precision, AP)，然后对所有类别的 AP 取平均得到。AP 曲线下的面积。`mAP@0.15` 表示在 IoU 阈值为 0.15 时计算的平均精度。
    *   **数学公式：**
        $$
        \text{AP} = \sum_n (R_n - R_{n-1}) P_n
        $$
        $$
        \text{mAP} = \frac{1}{N} \sum_{i=1}^N \text{AP}_i
        $$
    *   **符号解释：**
        *   $P_n$：在召回率 $R_n$ 下的精度。
        *   $R_n$：在第 $n$ 个召回点的召回率。
        *   $N$：类别总数。
        *   $\text{AP}_i$：第 $i$ 个类别的平均精度。
        *   <strong>精度 (Precision)：</strong> 模型预测为正例中，真正为正例的比例。$Precision = TP / (TP + FP)$。
        *   <strong>召回率 (Recall)：</strong> 实际为正例中，被模型预测为正例的比例。$Recall = TP / (TP + FN)$。
        *   <strong>IoU (Intersection over Union)：</strong> 衡量预测边界框与真实边界框重叠程度的指标。$IoU = (Area of Overlap) / (Area of Union)$。

3.  <strong>通过率 (Pass Rate)</strong>
    *   **概念定义：** 在代码生成或需要生成可执行结果的任务中，通过率衡量模型生成的代码或答案能够通过测试用例或规则验证的比例。
    *   **数学公式：**
        $$
        \text{Pass Rate} = \frac{\text{Number of Passed Cases}}{\text{Total Number of Cases}}
        $$
    *   **符号解释：**
        *   `Number of Passed Cases`：模型通过验证的案例数量。
        *   `Total Number of Cases`：总共进行验证的案例数量。

4.  <strong>MM-MT-Bench 评分 (LLM-as-a-judge score)</strong>
    *   **概念定义：** `MM-MT-Bench` 是一种多轮评估基准，使用强大的 `LLM` 作为评判模型来评估多模态指令微调模型的性能。它通过比较模型对多轮对话指令的遵循能力、有用性和连贯性来打分。该指标反映了模型在复杂、开放式多模态对话中的整体表现。
    *   **数学公式：** 无标准化数学公式，通常由评判 LLM 根据预设评分标准给出分数。

5.  **Perplexity (PPL)**
    *   **概念定义：** `PPL` 是衡量语言模型生成文本质量的指标，表示模型对样本的困惑度。`PPL` 值越低，表示模型对文本的预测能力越强，生成的文本越流畅和自然。在评估文本质量、代码生成等任务中可能被使用。
    *   **数学公式：** 对于一个长度为 $N$ 的序列 $W = (w_1, w_2, \ldots, w_N)$，其困惑度定义为：
        $$
        \text{PPL}(W) = P(w_1, w_2, \ldots, w_N)^{-\frac{1}{N}} = \sqrt[N]{\frac{1}{P(w_1, w_2, \ldots, w_N)}}
        $$
        其中，联合概率 $P(w_1, \ldots, w_N)$ 通常表示为条件概率的乘积：
        $$
        P(w_1, \ldots, w_N) = \prod_{i=1}^N P(w_i | w_1, \ldots, w_{i-1})
        $$
    *   **符号解释：**
        *   $W$：一个词元序列。
        *   $N$：序列中的词元数量。
        *   $P(w_1, \ldots, w_N)$：序列 $W$ 的联合概率。
        *   $P(w_i | w_1, \ldots, w_{i-1})$：在给定前 `i-1` 个词元的情况下，第 $i$ 个词元出现的条件概率。

6.  <strong>胜率 (Win Rate)</strong>
    *   **概念定义：** 在对抗性评估或人类偏好对齐评估中，胜率衡量模型在与另一个模型进行比较时，其输出被评判者（可以是人类或 LLM）认为更好的比例。例如，在 `Arena-Hard` 基准中，它表示模型在两两比较中胜出的频率。
    *   **数学公式：**
        $$
        \text{Win Rate} = \frac{\text{Number of Wins}}{\text{Total Number of Comparisons}}
        $$
    *   **符号解释：**
        *   `Number of Wins`：模型在比较中胜出的次数。
        *   `Total Number of Comparisons`：总共进行的比较次数。

## 5.3. 对比基线
Qwen3-VL 论文将自己的方法与一系列领先的模型进行了比较，这些基线模型具有代表性，包括：

1.  **Qwen 系列内部模型：**
    *   **Qwen2.5-VL-72B：** Qwen3-VL 的前身，用于展示新模型相对于上一代的改进，尤其是在中等规模模型和视频理解方面。
    *   <strong>Qwen3 纯文本模型 (例如 Qwen3-235B-A22B-Instruct-2507, Qwen3-32B, Qwen3-30B-A3B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B)：</strong> 用于评估 Qwen3-VL 在多模态训练后其纯文本能力是否得到保持甚至超越。

2.  <strong>闭源最先进模型 (Closed-source SOTA Models)：</strong>
    *   <strong>Gemini 2.5 Pro (Comanici et al., 2025)：</strong> Google 的旗舰多模态模型，在多模态理解、长上下文和智能体能力方面表现出色。论文用其 `thinking` 和 `budget-128` 模式进行比较。
    *   **Gemini 2.5 Flash：** Gemini 系列的更轻量级版本，用于与 Qwen3-VL 的中型模型（如 32B, 30B-A3B）进行比较。
    *   <strong>GPT-5 (OpenAI, 2025)：</strong> OpenAI 的旗舰模型（尽管论文引用的是未发布的预印本），代表了最顶尖的语言和多模态能力。论文用其 `high` 和 `minimal`（低思考预算）模式进行比较，以及 `GPT-5-mini` 和 `GPT-5-nano` 与 Qwen3-VL 的小模型进行比较。
    *   <strong>Claude Opus 4.1 (Anthropic, 2025)：</strong> Anthropic 的旗舰模型，以其强大的推理和长上下文能力闻名。论文用其 `thinking` 和 `non-thinking` 模式进行比较。

3.  **其他竞争模型：**
    *   **Deepseek V3 0324：** 另一个大型语言模型，用于在纯文本任务上与 Qwen3-VL 的 instruct 模式进行比较。

        选择这些基线的原因在于它们代表了当前多模态和大型语言模型领域的最先进水平，涵盖了不同的模型规模（从小型到旗舰级）和架构（如密集型、MoE），以及不同的推理模式（`thinking` vs `instruct`）。通过与这些强劲的对手进行广泛比较，Qwen3-VL 能够全面展示其在性能、效率和功能上的优势。

# 6. 实验结果与分析

Qwen3-VL 系列模型在广泛的多模态和纯文本基准测试中进行了全面评估。以下将详细分析论文中提供的实验结果。

## 6.1. 核心结果分析

### 6.1.1. 通用视觉问答 (General Visual Question Answering)
Qwen3-VL 在 `MMBench-V1.1` (Liu et al., 2023b), `RealWorldQA` (xAI, 2024), `MMStar` (Chen et al., 2024a) 和 `SimpleVQA` (Cheng et al., 2025) 等通用视觉问答基准上表现出色。

*   <strong>旗舰模型 (235B-A22B)：</strong> `Qwen3-VL-235B-A22B-Thinking` 在 `MMStar` 上取得了 78.7 的最高分，与 `Gemini-2.5-Pro` 的 `Thinking` 模式（表现最佳）不相上下。`Qwen3-VL-235B-A22B-Instruct` 在 `MMBench` (89.3/88.9) 和 `RealWorldQA` (79.2) 上取得了最高分。
*   <strong>中型模型 (32B/30B-A3B)：</strong> `Qwen3-VL-32B-Thinking` 在 `MMBench` (89.5/89.5) 和 `RealWorldQA` (79.4) 上取得了最高分。`Qwen3-VL-32B-Instruct` 在 `RealWorldQA` 上甚至超越了 `Thinking` 变体，得分 79.0。
*   <strong>小模型 (2B/4B/8B)：</strong> `Qwen3-VL-8B` 在所有五个基准上均表现最佳，例如 `MMBench-EN` 上 `Thinking` 模式从 2B 的 79.9 提升到 8B 的 85.3。

### 6.1.2. 多模态推理 (Multimodal Reasoning)
在 `MMMU` (Yue et al., 2024a), `MathVision` (Wang et al., 2024b), `MathVista` (Lu et al., 2023) 等 STEM 相关和视觉谜题任务上，Qwen3-VL 展现了最先进的性能。

*   <strong>旗舰模型 (235B-A22B)：</strong> `Qwen3-VL-235B-A22B-Instruct` 在 `MathVista_mini`, `MathVision`, `MathVerse_mini`, `DynaMath`, `ZeroBench`, `VLMsAreBlind`, `VisuLogic` 和 `VisualPuzzlesDirect` 上取得了非思考型或低思考预算模型中的最佳结果。`Qwen3-VL-235B-A22B-Thinking` 在 `MathVista_mini`, `MathVision`, `MathVerse_mini`, `ZeroBench`, `LogicVista` 和 `VisuLogic` 上取得了最先进的结果。
*   <strong>中型模型 (32B/30B-A3B)：</strong> `Qwen3-VL-32B` 持续超越 `Gemini-2.5-Flash` 和 `GPT-5-mini`，甚至在中等规模上超越了前一代 `Qwen2.5-VL-72B`。MoE 模型 `Qwen3-VL-30B-A3B` 也取得了具有竞争力的结果。
*   <strong>小模型 (2B/4B/8B)：</strong> `8B` 变体总体上保持明显优势，`4B` 模型在 `DynaMath` 和 `VisuLogic` 上得分最高，即使是最小的 `2B` 模型也展现出强大的推理能力。

### 6.1.3. 对齐和主观任务 (Alignment and Subjective Tasks)
在 `MM-MT-Bench` (Agrawal et al., 2024), `HallusionBench` (Guan et al., 2023) 和 `MIA-Bench` (Qian et al., 2024) 上，Qwen3-VL 表现出卓越的指令遵循和幻觉抑制能力。

*   <strong>旗舰模型 (235B-A22B)：</strong> `Qwen3-VL-235B-A22B` 持续超越其他闭源模型。在 `HallusionBench` 上，`thinking` 版本超越 `Gemini-2.5-pro`, `GPT-5` 和 `Claude opus 4.1` 分别 3.0, 1.0 和 6.3 分。在 `MIA-Bench` 上，`Thinking` 版本取得了所有模型中的最佳总分，展示了其卓越的多模态指令遵循能力。
*   **小模型：** 较小的模型（如 30B-A3B, 32B, 2B/4B/8B 系列）也表现出色，尤其在 `MIA-Bench` 上降幅可忽略不计。

### 6.1.4. 文本识别和文档理解 (Text Recognition and Document Understanding)
Qwen3-VL 在 OCR、文档解析、文档问答和文档推理方面取得了显著进展。

*   <strong>旗舰模型 (235B-A22B)：</strong> `Qwen3-VL-235B-A22B-Instruct` 在 `CC-OCR` (Yang et al., 2024b) 和 `OmniDocBench` (Ouyang et al., 2024) 等 OCR 专用解析基准以及 `OCR-Bench` (Liu et al., 2024) 和 `OCRBench_v2` (Fu et al., 2024b) 等综合 OCR 基准上，均取得了最先进的水平。在需要 OCR 能力和关键词搜索的 VQA 基准（如 `DocVQA`, `InfoVQA`, `AI2D`, `ChartQA`）上，`Instruct` 和 `Thinking` 变体表现相当。在 `CharXiv` (Wang et al., 2024g) 的推理子集上，`Thinking` 变体超越 `Instruct` 版本，仅次于 `GPT5-thinking` 和 `Gemini-2.5-Pro-Thinking`。
*   **长文档理解：** 在 `MMLongBench-Doc` (Ma et al., 2024) 基准上，`Qwen3-VL-235B-A22B` 在 `instruct/thinking` 设置下分别取得 57.0%/56.2% 的准确率，展示了长文档理解任务的最先进性能。
*   **多语言 OCR：** 支持从 10 种语言扩展到 39 种语言。在自建测试集上，模型在 39 种语言中的 32 种上准确率超过 70%，证实了其强大的多语言 OCR 能力。
    ```markdown

    ![fig 3](images/3.jpg)
    *该图像是一个条形图，展示了多语言 OCR 支持的准确率（%），横轴为不同语言，纵轴为准确度。图中显示，罗马尼亚语和西班牙语的准确率较高，而其他语言的准确率则有所不同。*

    Figure 2: Multilingual OCR performance of our model on a self-built test set. The model achieves over 70% accuracy on 32 out of 39 supported languages, demonstrating strong and usable multilingual capabilities.
    ```
    *VLM 描述: 该图像是一个条形图，展示了多语言 OCR 支持的准确率（%），横轴为不同语言，纵轴为准确度。图中显示，罗马尼亚语和西班牙语的准确率较高，而其他语言的准确率则有所不同。*

### 6.1.5. 2D 和 3D 定位 (2D and 3D Grounding)
Qwen3-VL 在 2D 和 3D 定位任务上均取得最先进的性能。

*   **2D 定位：** 在 $RefCOCO/+/g$ (Kazemyzeadeh et al., 2014; Mao et al., 2016), `ODinW-13` (Li et al., 2022) 和 `CountBench` (Paiss et al., 2023) 上，旗舰模型 `Qwen3-VL-235B-A22B` 表现出色。在 `ODinW-13` 上取得了 48.6 mAP，展示了强大的多目标开放词汇对象定位能力。
*   **3D 定位：** 在 `Omni3D` (Brazil et al., 2023) 基准（包括 `ARKitScenes`, `Hypersim`, `SUN RGB-D` (Song et al., 2015)）上，旗舰模型 `Qwen3-VL-235B-A22B` 持续超越其他闭源模型。在 `SUN RGB-D` 数据集上，`Thinking` 变体超越 `Gemini-2.5-Pro` 5.2 分。

### 6.1.6. 细粒度感知 (Fine-grained Perception)
Qwen3-VL 在细粒度视觉理解方面取得了显著飞跃。

*   **带工具增强：** 旗舰模型 `Qwen3-VL-235B-A22B` 在 $V*$ (Wu & Xie, 2024), `HRBench-4k` (Wang et al., 2024e) 和 `HRBench-8k` (Wang et al., 2024e) 上，通过工具增强，分别取得了 93.7, 85.3 和 82.3 的最先进性能。
*   **工具集成优势：** 性能提升主要归因于架构改进和训练策略。值得注意的是，集成外部工具带来的性能增益持续超过单纯增大模型规模带来的收益（在 $V*$ 上约 5 分的绝对提升），强调了工具集成智能体学习的潜力。

### 6.1.7. 多图像理解 (Multi-Image Understanding)
在 `BLINK` (Fu et al., 2024c) 和 `MuirBench` (Wang et al., 2024a) 基准上，Qwen3-VL 展现了整体优越性。`Qwen3-VL-235B-A22B-Instruct` 性能与 `Gemini-2.5-pro` 相当，而 `Qwen3-VL-235B-A22B-Thinking` 在 `MuirBench` 上取得了 80.1 的领先分数。

### 6.1.8. 具身和空间理解 (Embodied and Spatial Understanding)
在 `ERQA` (Team et al., 2025), `VSBench` (Yang et al., 2025b), `EmbSpatial` (Du et al., 2024), `RefSpatial` (Zhou et al., 2025) 和 `RoboSpatialHome` (Song et al., 2025a) 等基准上，Qwen3-VL 展现了卓越的能力，与 `Gemini-2.5-Pro`, `GPT-5` 和 `Claude-Opus-4.1` 等顶级模型竞争。其强大的空间理解能力源于对高分辨率视觉数据、细粒度指向和相对位置注释以及 QA 对的训练。

### 6.1.9. 视频理解 (Video Understanding)
得益于训练数据规模化和关键架构增强，Qwen3-VL 的视频理解能力显著提升。

*   <strong>旗舰模型 (235B-A22B)：</strong> 在 `VideoMME` (Fu et al., 2024a), `MVBench` (Li et al., 2024b) 等通用视频理解基准上，`Qwen3-VL-235B-A22B-Instruct` 性能与 `Gemini 2.5 Pro` (128 思考预算) 和 `GPT-5 minimal` 相当。
*   **长视频理解：** 通过将上下文窗口扩展到 256K 词元，模型在长视频评估任务（最显著的是 `MLVU`）上达到甚至超越 `Gemini-2.5-Pro`。
*   **帧处理：** 所有基准测试视频帧数上限为 2,048 帧，总视频词元不超过 224K。每帧最大词元数设置为 768（VideoMMU, MMVU）或 640（其他）。

### 6.1.10. 智能体 (Agent)
在 GUI 感知和决策能力方面，Qwen3-VL 表现出色。

*   **GUI 感知：** `Qwen3-VL-235B-A22B` 在 `ScreenSpot` (Cheng et al., 2024), `ScreenSpot Pro` (Li et al., 2025b), `OSWorldG` (Xie et al., 2025a) 等任务上取得了最先进的性能，展示了卓越的 UI 感知能力。
*   **在线评估：** `Qwen3-VL-32B` 在 `OSWorld` (41分) 和 `AndroidWorld` (63.7分) 上超越了当前的 VLM，展现了强大的规划、决策和反思能力。

### 6.1.11. 纯文本任务 (Text-Centric Tasks)
Qwen3-VL 的纯文本性能在知识、推理、代码、对齐和多语言任务上进行了全面评估。

*   <strong>旗舰模型 (235B-A22B)：</strong>
    *   `Qwen3-VL-235B-A22B-Instruct` 在推理需求任务（如数学和编码）上超越了 `DeepSeek V3 0324`, `Claude-Opus-4` 和 `Qwen3-235B-A22B-Instruct-2507`，表明其在集成视觉和文本能力的同时，仍然保持甚至增强了纯文本推理能力。
    *   `Qwen3-VL-235B-A22B-Thinking` 在 `AIME-25` 和 `LiveCodeBench v6` 上超越了 `OpenAI o3` 和 `Claude-Opus-4`，显示出更强的推理能力。
*   <strong>中型模型 (32B/30B-A3B)：</strong> 显著优于其纯文本对应模型 `Qwen3-32B` 和 `Qwen3-30B-A3B`，在许多基准上甚至与 `Qwen3-30B-A3B-2507` 相当或更优，尤其在 `AIME-25` 和 `HMMT-25` 上。
*   <strong>小模型 (2B/4B/8B)：</strong> 这些边缘侧模型表现出色，超越了基线模型，证明了强到弱蒸馏方法的有效性。

## 6.2. 数据呈现 (表格)

以下是原文 Table 2 的结果：

<table><tr><td rowspan="2"></td><td rowspan="2">Benchmark</td><td colspan="2">Qwen3-VL <br>235B-A22B</td><td colspan="2">Gemini <br>2.5 Pro</td><td colspan="2">OpenAI <br>GPT-5</td><td colspan="2">Claude <br>Opus 4.1</td></tr><tr><td>thinking</td><td>instruct</td><td>thinking</td><td>budget-128</td><td>high</td><td>minimal</td><td>thinking</td><td>non-thinking</td></tr><tr><td rowspan="10">STEM<br>Puzzle</td><td>MMMU</td><td>80.6</td><td>78.7</td><td>81.7*</td><td>80.9</td><td>84.2*</td><td>74.4*</td><td>78.4</td><td>77.2</td></tr><tr><td>MMMU-Pro</td><td>69.3</td><td>68.1</td><td>68.8*</td><td>71.2</td><td>78.4*</td><td>62.7*</td><td>64.8</td><td>60.7</td></tr><tr><td>MathVisitor</td><td>85.8</td><td>84.9</td><td>82.7*</td><td>77.7</td><td>81.3</td><td>50.9</td><td>75.5</td><td>74.5</td></tr><tr><td>MathVision</td><td>74.6</td><td>66.5</td><td>73.3*</td><td>66.0</td><td>70.9</td><td>45.8</td><td>64.3</td><td>57.7</td></tr><tr><td>MathVisionWP</td><td>~63.8</td><td>57.0</td><td>63.2</td><td>56.9</td><td>62.8</td><td>40.1</td><td>54.0</td><td>46.4</td></tr><tr><td>We-Math</td><td>74.8</td><td>67.5</td><td>80.6</td><td>74.5</td><td>73.8</td><td>51.8</td><td>65.2</td><td>60.2</td></tr><tr><td>MathVersumini</td><td>85.0</td><td>72.5</td><td>82.9</td><td>65.9</td><td>84.1</td><td>43.0</td><td>70.6</td><td>68.1</td></tr><tr><td>DynaMath</td><td>82.8</td><td>79.4</td><td>80.0</td><td>78.5</td><td>85.4</td><td>74.0</td><td>75.1</td><td>72.0</td></tr><tr><td>Math-VR</td><td>66.8</td><td>65.0</td><td>64.7*</td><td>54.3</td><td>58.1</td><td>21.7</td><td>54.3</td><td>38.0</td></tr><tr><td>ZeroBench</td><td>4</td><td>2</td><td>3</td><td>1</td><td>2</td><td>2</td><td>3</td><td>1</td></tr><tr><td>VlmsAneBlinda</td><td>79.5</td><td>80.4</td><td>86.1</td><td>78.5</td><td>80.5</td><td>53.4</td><td>77.8</td><td>72.2</td></tr><tr><td>LogicVista</td><td>72.2</td><td>65.8</td><td>72.0</td><td>68.7</td><td>71.8</td><td>46.3</td><td>67.3</td><td>63.5</td></tr><tr><td>Visual Logic</td><td>34.4</td><td>29.9</td><td>31.6</td><td>26.9</td><td>28.5</td><td>27.2</td><td>27.9</td><td>27.2</td></tr><tr><td>VisualPuzzles</td><td>57.2</td><td>54.7</td><td>60.9</td><td>56.9</td><td>57.3</td><td>47.9</td><td>48.8</td><td>47.6</td></tr><tr><td rowspan="6">General VQA</td><td>MMBench-EN</td><td>~88.8</td><td>89.3</td><td>90.1*</td><td>88.4</td><td>83.8</td><td>81.3</td><td>79.4</td><td>83.0</td></tr><tr><td>MMBench-CN</td><td>88.6</td><td>88.9</td><td>89.7*</td><td>86.4</td><td>83.5</td><td>79.9</td><td>84.9</td><td>74.3</td></tr><tr><td>RealWorldQA</td><td>81.3</td><td>79.2</td><td>78.0*</td><td>76.0</td><td>82.8</td><td>77.3</td><td>69.9</td><td>68.5</td></tr><tr><td>MMStar</td><td>78.7</td><td>78.4</td><td>77.5*</td><td>78.5</td><td>76.4</td><td>65.2</td><td>72.1</td><td>71.0</td></tr><tr><td>SimpleVQA</td><td>61.3</td><td>63.0</td><td>65.4</td><td>66.9</td><td>61.8</td><td>56.7</td><td>56.7</td><td>55.7</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">Alignment</td><td>HallusionBench</td><td>66.7</td><td>63.2</td><td>63.7*</td><td>60.9</td><td>65.7</td><td>53.7</td><td>60.4</td><td>55.1</td></tr><tr><td>MMM-TB-Bench</td><td>8.5</td><td>8.5</td><td>8.4*</td><td>7.6</td><td>7.6</td><td>7.5</td><td>7.8</td><td>7.9</td></tr><tr><td>MIA-Bench</td><td>92.7</td><td>91.3</td><td>92.3</td><td>91.3</td><td>92.4</td><td>92.6</td><td>91.2</td><td>90.0</td></tr><tr><td rowspan="10">Document<br>Understanding</td><td>DocVQAttest</td><td>96.5</td><td>97.1</td><td>92.6</td><td>94.0</td><td>91.5</td><td>89.6</td><td>92.5</td><td>89.2</td></tr><tr><td>InfoVQAttest</td><td>89.5</td><td>89.2</td><td>84.2</td><td>82.9</td><td>79.0</td><td>69.9</td><td>69.4</td><td>60.9</td></tr><tr><td>AI2Dw.M.</td><td>89.2</td><td>89.7</td><td>90.9</td><td>90.0</td><td>89.7</td><td>84.1</td><td>86.4</td><td>84.4</td></tr><tr><td>ChartQAttest</td><td>90.3</td><td>90.3</td><td>83.3</td><td>62.6</td><td>59.7</td><td>59.1</td><td>86.2</td><td>83.9</td></tr><tr><td>OCRBench</td><td>875</td><td>920</td><td>866</td><td>872</td><td>810</td><td>787</td><td>764</td><td>750</td></tr><tr><td>OCRBench_v2en</td><td>66.8</td><td>67.1</td><td>54.3</td><td>55.2</td><td>53.0</td><td>48.2</td><td>48.4</td><td>47.2</td></tr><tr><td>OCRBench_v2 Zh</td><td>63.5</td><td>61.8</td><td>48.5</td><td>53.1</td><td>43.2</td><td>37.7</td><td>43.7</td><td>38.0</td></tr><tr><td>CC-OCR</td><td>81.5</td><td>82.2</td><td>77.2</td><td>76.8</td><td>68.3</td><td>66.1</td><td>69.1</td><td>66.0</td></tr><tr><td>OmniDocBenchen</td><td>0.155</td><td>0.143</td><td>0.347</td><td>0.206</td><td>0.356</td><td>0.174</td><td>0.194</td><td>-</td></tr><tr><td>OmniDocBenchzh</td><td>0.207</td><td>0.207</td><td>0.238</td><td>0.249</td><td>0.472</td><td>0.389</td><td>0.293</td><td>-</td></tr><tr><td>ChairXinv(DQ)</td><td>90.5</td><td>89.4</td><td>94.4</td><td>87.8</td><td>89.2</td><td>79.5</td><td>88.5</td><td>87.8</td></tr><tr><td>ChairXinv(RQ)</td><td>66.1</td><td>62.1</td><td>67.9</td><td>62.9</td><td>81.1*</td><td>57.8</td><td>63.6</td><td>60.2</td></tr><tr><td>MMLongBenchDoc</td><td>56.2</td><td>57.0</td><td>55.6</td><td>51.2</td><td>51.5</td><td>42.4</td><td>54.5</td><td>48.1</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="6">2D/3D<br>WDRound</td><td>RefCOCO-avg</td><td>92.1</td><td>91.9</td><td>74.6*</td><td>-</td><td>66.8</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CountBench</td><td>93.7</td><td>93.0</td><td>91.0*</td><td>91.0</td><td>91.7</td><td>87.8</td><td>93.1</td><td>91.9</td></tr><tr><td>ODINW-13</td><td>43.2</td><td>48.6</td><td>33.7*</td><td>34.5</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ARKiSCSEnes</td><td>53.7</td><td>56.9</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>HyperSim</td><td>11.0</td><td>13.0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SUNRBINDEX</td><td>34.9</td><td>39.4</td><td>29.7</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td rowspan="7">EmBodel/Spatial<br>Understanding</td><td>ERQA</td><td>52.5</td><td>51.3</td><td>55.3</td><td>50.3</td><td>65.7*</td><td>42.0*</td><td>34.8</td><td>28.0</td></tr><tr><td>VSI-Bench</td><td>60.0</td><td>62.7</td><td>-</td><td>-</td><td>-</td><td>-</td><td>69.2</td><td>66.0</td></tr><tr><td>EmbIsospatialBench</td><td>84.3</td><td>83.1</td><td>79.1</td><td>73.3</td><td>82.9</td><td>75.1</td><td>-</td><td>-</td></tr><tr><td>RefspatialBench</td><td>69.9</td><td>65.5</td><td>36.5</td><td>35.6</td><td>23.8</td><td>23.1</td><td>-</td><td>-</td></tr><tr><td>RobSpatialHome</td><td>73.8</td><td>69.4</td><td>47.5</td><td>49.2</td><td>53.5</td><td>43.6</td><td>-</td><td>-</td></tr><tr><td rowspan="2">Multi-Image</td><td>BLINK</td><td>67.1</td><td>70.7</td><td>70.6*</td><td>70.0</td><td>71.0</td><td>62.8</td><td>64.1</td><td>62.9</td></tr><tr><td>MUIRBENCH</td><td>80.1</td><td>73.0</td><td>77.2</td><td>74.0</td><td>77.5</td><td>66.5</td><td>-</td><td>-</td></tr><tr><td rowspan="6">Video<br>Understanding</td><td>MVBench</td><td>75.2</td><td>76.5</td><td>69.9</td><td>65.8</td><td>75.3</td><td>64.6</td><td>61.4</td><td>59.0</td></tr><tr><td>Video-MME/wO sub.</td><td>79.0</td><td>79.2</td><td>85.1</td><td>80.6</td><td>84.7</td><td>77.3</td><td>75.6</td><td>73.3</td></tr><tr><td>LvívM avg</td><td>83.8</td><td>84.3</td><td>85.6</td><td>81.2</td><td>86.2</td><td>78.3</td><td>73.5</td><td>71.2</td></tr><tr><td>LvBench</td><td>63.6</td><td>67.7</td><td>73.0</td><td>69.0</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Charades-STAevol MediaMMMpui</td><td>80.7</td><td>74.7</td><td>83.6*</td><td>79.4</td><td>84.6*</td><td>61.6*</td><td>76.2</td><td>70.1</td></tr><tr><td>C喵iLwDivl CHeMPvidi</td><td>71.1</td><td>68.1</td><td>74.9</td><td>72.2</td><td>73.1</td><td>68.1</td><td>66.4</td><td>61.4</td></tr><tr><td rowspan="2">Perception<br>with Tool</td><td>V*</td><td>85.9</td><td></td><td></td><td></td><td></td><td></td><td>-</td><td></td></tr><tr><td>HRBench4K</td><td>84.3</td><td>83.7*</td><td>87.3</td><td>84.8</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Multi-Dodai<br>Coding</td><td>76.6</td><td>84.2*</td><td>85.4</td><td>84.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td></td></tr><tr><td>Design2Doe</td><td>93.4</td><td>92.0</td><td>89.2</td><td>90.3</td><td>92.5</td><td>88.9</td><td>88.5</td><td>85.3</td></tr><tr><td>ChatMini</td><td>79.4</td><td>80.0</td><td>83.9</td><td>79.9</td><td>62.1</td><td>41.4</td><td>85.2</td><td>82.9</td><td></td></tr><tr><td>UniSVG</td><td>65.8</td><td>69.8</td><td>70.0</td><td>67.9</td><td>71.7</td><td>74.5</td><td>73.0</td><td>72.5</td><td></td></tr><tr><td rowspan="4">Multi-Dodai<br>Agent</td><td>ScreenSpot Pro</td><td>61.8</td><td>62.0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>OSWorldG</td><td>68.3</td><td>66.7</td><td>45.2</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>AndroidWorld</td><td>62.0</td><td>63.7</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>OSWorld</td><td>38.1</td><td>31.6</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>44.4</td></tr><tr><td>WindowsAA</td><td>32.1</td><td>28.9</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td></td></tr></table>

以下是原文 Table 3 的结果：

<table><tr><td colspan="3"></td><td>Qwen3-VL 30B-A3B</td><td>Qwen3-VL 32B</td><td>Gemini 2.5 Flash</td><td>GPT-5 mini</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>Benchmark</td><td>thinking</td><td>instruct</td><td>thinking</td><td>no-tank</td><td>high</td><td></td><td></td><td></td></tr><tr><td rowspan="10">STEM<br>Puzzle</td><td rowspan="10">MMM</td><td>MMMU</td><td>76.0</td><td>74.2</td><td>78.1</td><td>76.0</td><td>77.7</td><td>76.3</td><td>79.0</td><td>67.9</td></tr><tr><td>MMMU-Pro</td><td>63.0</td><td>60.4</td><td>68.1</td><td>65.3</td><td>67.2</td><td>65.9</td><td>67.3</td><td>53.7</td></tr><tr><td>MathVista<br>*mini*</td><td>81.9</td><td>80.1</td><td>62.9</td><td>70.2</td><td>63.4</td><td>74.4</td><td>75.3</td><td>79.1</td></tr><tr><td>MathVision<br>≡Mathvisionwp</td><td>65.7</td><td>60.2</td><td>52.8</td><td>58.6</td><td>54.6</td><td>63.9</td><td>60.4</td><td>49.6</td></tr><tr><td>MathVisionpw</td><td>58.9</td><td>52.3</td><td>71.6</td><td>63.3</td><td>63.9</td><td>49.0</td><td>50.6</td><td>42.8</td></tr><tr><td>We-Math</td><td>70.0</td><td>56.9</td><td>71.6</td><td>63.3</td><td>53.7</td><td>60.3</td><td>70.2</td><td>51.4</td></tr><tr><td>MathVerse</td><td>79.6</td><td>70.2</td><td>80.7</td><td>78.4</td><td>57.7</td><td>59.7</td><td>61.1</td><td>61.3</td></tr><tr><td>DynaMath</td><td>81.1</td><td>73.4</td><td>82.0</td><td>76.7</td><td>75.9</td><td>69.7</td><td>81.4</td><td>72.3</td></tr><tr><td>Math-VR</td><td>61.7</td><td>61.3</td><td>62.3</td><td>59.8</td><td>58.8</td><td>54.7</td><td>58.2</td><td>26.4</td></tr><tr><td>ZeroBench</td><td>0</td><td>0</td><td>2</td><td>1</td><td>1</td><td>3</td><td>3</td><td>2</td><td>2</td></tr><tr><td rowspan="5">General VQA</td><td>VlmsAreBlind</td><td>72.5</td><td>67.5</td><td>85.1</td><td>87.0</td><td>77.5</td><td>73.9</td><td>75.8</td><td>62.0</td></tr><tr><td>LogicVista</td><td>65.8</td><td>53.5</td><td>70.9</td><td>62.2</td><td>67.3</td><td>60.0</td><td>71.4</td><td>50.8</td></tr><tr><td>VisuLogic</td><td>26.6</td><td>23.0</td><td>32.4</td><td>27.7</td><td>31.0</td><td>23.3</td><td>27.2</td><td>27.6</td></tr><tr><td>VisualPuzzles</td><td>52.0</td><td>46.2</td><td>54.7</td><td>53.2</td><td>41.4</td><td>45.0</td><td>59.3</td><td>41.8</td></tr><tr><td>Statistical</td><td>MMench-EN</td><td>87.0</td><td>86.1</td><td>89.5</td><td>87.6</td><td>87.1</td><td>86.6</td><td>86.6</td><td>76.5</td></tr><tr><td rowspan="5">General VQA</td><td>MMBench-CN</td><td>85.9</td><td>85.3</td><td>89.4</td><td>87.7</td><td>87.3</td><td>86.0</td><td>84.0</td><td>76.3</td></tr><tr><td>RealWorldQA</td><td>77.4</td><td>73.7</td><td>78.4</td><td>79.0</td><td>76.0</td><td>75.7</td><td>79.0</td><td>73.3</td></tr><tr><td>MMStar</td><td>75.5</td><td>72.1</td><td>79.4</td><td>77.7</td><td>76.5</td><td>75.8</td><td>74.1</td><td>61.3</td></tr><tr><td>SimpleVQA</td><td>54.3</td><td>52.7</td><td>55.4</td><td>56.9</td><td>63.2</td><td>59.2</td><td>56.8</td><td>50.3</td></tr><tr><td>MMBench</td><td>66.0</td><td>61.5</td><td>67.4</td><td>63.8</td><td>63.5</td><td>59.1</td><td>63.2</td><td>55.9</td></tr><tr><td rowspan="2">Alignment</td><td>MM-MT-Bench</td><td>7.9</td><td>8.0</td><td>8.3</td><td>8.4</td><td>8.1</td><td>8.0</td><td>7.7</td><td>7.4</td></tr><tr><td>MIA-Bench</td><td>91.6</td><td>91.2</td><td>92.3</td><td>91.8</td><td>91.1</td><td>90.6</td><td>92.0</td><td>92.3</td></tr><tr><td rowspan="9">Document Understanding</td><td>DocVQA</td><td>95.5</td><td>95.0</td><td>96.1</td><td>96.9</td><td>92.8</td><td>93.0</td><td>90.5</td><td>90.6</td></tr><tr><td>InfoVQA</td><td>85.6</td><td>81.8</td><td>89.2</td><td>87.0</td><td>82.5</td><td>81.7</td><td>77.6</td><td>72.8</td></tr><tr><td>AI2D</td><td>86.9</td><td>85.0</td><td>88.9</td><td>89.5</td><td>88.7</td><td>87.7</td><td>88.2</td><td>82.9</td></tr><tr><td>ChatVQA</td><td>89.4</td><td>86.8</td><td>89.0</td><td>88.5</td><td>60.6</td><td>69.0</td><td>57.5</td><td>57.8</td></tr><tr><td>OCRBench</td><td>839</td><td>90.3</td><td>85.5</td><td>89.5</td><td>853</td><td>864</td><td>821</td><td>807</td></tr><tr><td>OCRBench-v2</td><td>62.6</td><td>63.2</td><td>68.4</td><td>67.4</td><td>52.2</td><td>50.6</td><td>52.6</td><td>45.7</td></tr><tr><td>OCRBench_v2h</td><td>60.4</td><td>57.8</td><td>62.1</td><td>59.2</td><td>43.8</td><td>43.9</td><td>45.1</td><td>41.0</td></tr><tr><td>CC-OCR</td><td>77.8</td><td>80.7</td><td>79.6</td><td>80.3</td><td>75.4</td><td>74.8</td><td>70.8</td><td>61.6</td></tr><tr><td>OmniDocBench</td><td>0.165</td><td>0.183</td><td>0.148</td><td>0.151</td><td>0.265</td><td>0.228</td><td>0.181</td><td>0.260</td></tr><tr><td>OmniDocBench</td><td>0.233</td><td>0.253</td><td>0.236</td><td>0.239</td><td>0.245</td><td>0.305</td><td>0.316</td><td>0.425</td></tr><tr><td>CharXiv(DQ)</td><td>86.9</td><td>85.5</td><td>90.2</td><td>90.5</td><td>90.1</td><td>85.5</td><td>89.4</td><td>78.6</td></tr><tr><td>CharXIV(RQ)</td><td>56.6</td><td>48.9</td><td>65.2</td><td>62.8</td><td>61.7</td><td>60.1</td><td>68.6</td><td>48.9</td></tr><tr><td>MMLongBenchDoc</td><td>47.4</td><td>47.1</td><td>54.6</td><td>55.4</td><td>49.0</td><td>44.6</td><td>50.3</td><td>39.6</td></tr><tr><td rowspan="4">2D/3D</td><td>RefCOCO-avg</td><td>89.3</td><td>89.7</td><td>91.1</td><td>91.9</td><td>-</td><td>-</td><td>-</td><td></td></tr><tr><td>CountBench</td><td>90.0</td><td>89.8</td><td>94.1</td><td>94.9</td><td>86.0</td><td>83.7</td><td>91.0</td><td>84.1</td></tr><tr><td>ODinW-13</td><td>42.3</td><td>47.5</td><td>41.8</td><td>46.6</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ARKitScenes</td><td>55.6</td><td>56.1</td><td>46.1</td><td>55.6</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td rowspan="4">2D/3D</td><td>Hypersim</td><td>11.4</td><td>12.5</td><td>12.5</td><td>14.0</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SURGBD</td><td>34.6</td><td>38.1</td><td>33.9</td><td>37.0</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ERQA</td><td>45.3</td><td>43.0</td><td>52.3</td><td>48.8</td><td>-</td><td>-</td><td>54.0</td><td>45.8</td></tr><tr><td>VSI-Bench</td><td>56.1</td><td>63.2</td><td>61.2</td><td>61.5</td><td>-</td><td>-</td><td>31.5</td><td>30.5</td></tr><tr><td rowspan="4">Embodied/Spatial</td><td>EmbSpatibench</td><td>86.0</td><td>76.4</td><td>82.7</td><td>81.5</td><td>-</td><td>-</td><td>80.7</td><td>72.1</td></tr><tr><td>Refspatibench</td><td>54.2</td><td>53.1</td><td>67.2</td><td>61.4</td><td>-</td><td>-</td><td>9.0</td><td>4.0</td></tr><tr><td>RoboSpatialHome</td><td>65.5</td><td>62.9</td><td>74.2</td><td>64.6</td><td>-</td><td>-</td><td>54.3</td><td>44.6</td></tr><tr><td>Statistical</td><td>EMBley科院</td><td>65.4</td><td>67.7</td><td>68.5</td><td>67.3</td><td>68.1</td><td>66.8</td><td>-</td><td>56.7</td></tr><tr><td rowspan="2">Multi-Image</td><td>MURBENCH</td><td>77.6</td><td>62.9</td><td>80.3</td><td>72.8</td><td>72.7</td><td>67.5</td><td>-</td><td>57.5</td></tr><tr><td>Multi-Image</td><td>MMEngene</td><td>72.0</td><td>72.3</td><td>73.2</td><td>72.8</td><td>-</td><td>-</td><td>-</td></tr><tr><td rowspan="5">Video Understanding</td><td>MultExam</td><td>73.3</td><td>74.5</td><td>77.3</td><td>76.6</td><td>79.6</td><td>75.6</td><td>78.9</td><td>71.0</td></tr><tr><td><|ref|><td></td><td></td><td></td><td></td><td></td><td>77.8</td><td>83.3</td><td>71.7</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Training</td><td>Training</td><td>Training</td><td>Training</td><td>Training</td><td>Training</td><td>Training</td><td>Computer Vision</td><td>Computer Vision</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Computer Vision</td><td>Computer Vision</td></tr><tr><td rowspan="4">Computer Vision</td><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision &amp;gt; Computer Vision</td><td>Computer Vision</td><td>Computer Vision</td></tr><tr><td>Computer Vision</td><td>Data-valuing</td><td>Data-valuing</td><td>Data-valuing</td><td>Data-valuing</td><td>Data-valuing</td><td>Data-valuing</td><td>Computer Vision</td><td>Data-valuing</td></tr><tr><td>Data-valuing</td><td>Data-valuing</td><td>Data-valuing</td><td>Data-valuing</td><td>Data-valuing</td><td>Data-valuing</td><td>Data-valuing</td><td>Computer Vision</td><td>Data-valuing</td></tr><tr><td>Data-valuing</td><td>Data-valuing</td><td>Data-valuing</td><td>Data-valuing</td><td>Data-valuing</td><td>Data-valuing</td><td>Data-valuing</td><td>Computer Vision</td><td>Data-valuing</td></tr><tr><td rowspan="2">Visualizers</td><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision</td><td>Data-valuing</td></tr><tr><td>Computer Vision</td><td>Computer Vision</td><td>Visualization</td><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision &amp;amp; Data-valuing</td><td>Data-valuing</td></tr><tr><td>Reference</td><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision</td><td>Training</td></tr><tr><td rowspan="4">Example of Output Visual Program</td><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision</td><td>mull</td></tr><tr><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision</td><td>Mull</td><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision</td></tr><tr><td>Computer Vision</td><td>Computer Vision (Mull)</td><td>Computer Vision</td><td>Computer Vision</td><td>Mull</td><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision (Mull)</td></tr><tr><td>Computer Vision</td><td>Computer Vision (Mull)</td><td>Comparalleled</td><td>Computer Vision</td><td>Mull</td><td>Computer Vision</td><td>Computer Vision</td><td>Computer Vision (Mull)</td></tr><tr><td>Computer Vision</td><td>Computer Vision</td><td>Mull</td><td>Mull</td><td>Mull</td><td>Mull</td><td>Mull</td><td>Mull</td><td>Mull</td></tr></table>

以下是原文 Table 4 的结果：

<table><tr><td></td><td>Benchmark</td><td colspan="2">Qwen3-VL<br>2B<br>thinking instruct</td><td colspan="2">Qwen3-VL<br>4B<br>thinking instruct</td><td colspan="2">Qwen3-VL<br>8B<br>thinking instruct</td><td colspan="2">OpenAI<br>GPT-5 nano<br>high minimal</td></tr><tr><td rowspan="10">STEM<br>Puzzle</td><td>MMMU</td><td>61.4</td><td>53.4</td><td>70.8</td><td>67.4</td><td>74.1</td><td>69.6</td><td>75.8</td><td>57.6</td></tr><tr><td>MMMU-Pro</td><td>42.5</td><td>36.5</td><td>57.0</td><td>53.2</td><td>60.4</td><td>55.9</td><td>57.2</td><td>36.5</td></tr><tr><td>MathVistamini</td><td>73.6</td><td>61.3</td><td>79.5</td><td>73.7</td><td>81.4</td><td>77.2</td><td>71.5</td><td>40.9</td></tr><tr><td>MathVision</td><td>45.9</td><td>31.6</td><td>60.0</td><td>51.6</td><td>62.7</td><td>53.9</td><td>62.2</td><td>33.2</td></tr><tr><td>MathVisinowp</td><td>35.5</td><td>30.9</td><td>48.7</td><td>44.4</td><td>53.3</td><td>45.4</td><td>49.3</td><td>28.3</td></tr><tr><td>MathVerse-mini</td><td>66.9</td><td>52.1</td><td>75.2</td><td>46.8</td><td>77.7</td><td>62.1</td><td>74.2</td><td>27.0</td></tr><tr><td>DynaMath</td><td>66.7</td><td>54.2</td><td>74.4</td><td>65.3</td><td>73.2</td><td>67.7</td><td>78.0</td><td>62.0</td></tr><tr><td>Math-VR</td><td>37.7</td><td>20.7</td><td>58.1</td><td>52.3</td><td>59.0</td><td>53.4</td><td>49.7</td><td>25.0</td></tr><tr><td>ZeroBench</td><td>0</td><td>0</td><td>0</td><td>0</td><td>2</td><td>1</td><td>1</td><td>1</td></tr><tr><td>VLMsAreBlind</td><td>50.0</td><td>56.0</td><td>68.6</td><td>71.9</td><td>69.1</td><td>74.0</td><td>66.7</td><td>40.2</td></tr><tr><td rowspan="3"></td><td>LogicVista</td><td>50.0</td><td>35.8</td><td>61.1</td><td>53.2</td><td>65.1</td><td>55.3</td><td>59.7</td><td>40.5</td></tr><tr><td>VisuLogic</td><td>25.4</td><td>11.5</td><td>30.2</td><td>19.0</td><td>27.5</td><td>22.5</td><td>24.5</td><td>24.0</td></tr><tr><td>VisualPuzzles</td><td>37.4</td><td>34.3</td><td>48.9</td><td>43.7</td><td>51.7</td><td>47.9</td><td>43.5</td><td>31.3</td></tr><tr><td rowspan="5">General VQA</td><td>MMBench-EN</td><td>79.9</td><td>78.4</td><td>84.6</td><td>83.9</td><td>85.3</td><td>84.5</td><td>78.4</td><td>50.8</td></tr><tr><td>MMBench-CN</td><td>78.8</td><td>75.9</td><td>83.8</td><td>83.5</td><td>85.5</td><td>84.7</td><td>77.6</td><td>48.5</td></tr><tr><td>RealWorldQA</td><td>69.5</td><td>63.9</td><td>73.2</td><td>70.9</td><td>73.5</td><td>71.5</td><td>71.8</td><td>60.7</td></tr><tr><td>MMStar</td><td>68.1</td><td>58.3</td><td>73.2</td><td>69.8</td><td>75.3</td><td>70.9</td><td>68.6</td><td>41.3</td></tr><tr><td>SimpleVQA</td><td>43.6</td><td>40.7</td><td>48.8</td><td>48.0</td><td>49.6</td><td>50.2</td><td>46.0</td><td>39.0</td></tr><tr><td rowspan="3">Alignment</td><td>HallusionBench</td><td>54.9</td><td>51.4</td><td>64.1</td><td>57.6</td><td>65.4</td><td>61.1</td><td>58.4</td><td>39.3</td></tr><tr><td>MM-MT-Bench</td><td>6.9</td><td>5.9</td><td>7.7</td><td>7.5</td><td>8.0</td><td>7.7</td><td>6.6</td><td>6.2</td></tr><tr><td>MIA-Bench</td><td>85.6</td><td>83.6</td><td>91.0</td><td>89.7</td><td>91.5</td><td>91.1</td><td>89.9</td><td>89.6</td></tr><tr><td rowspan="10">Document<br>Understanding</td><td>DocVQAtest</td><td>92.9</td><td>93.3</td><td>94.2</td><td>95.3</td><td>95.3</td><td>96.1</td><td>88.2</td><td>78.3</td></tr><tr><td>InfoVQAtest</td><td>77.1</td><td>72.4</td><td>83.0</td><td>80.3</td><td>86.0</td><td>83.1</td><td>68.6</td><td>49.2</td></tr><tr><td>AI2Dw. M.</td><td>80.4</td><td>76.9</td><td>84.9</td><td>84.1</td><td>84.9</td><td>85.7</td><td>81.9</td><td>65.7</td></tr><tr><td>ChartQAtest</td><td>86.6</td><td>79.1</td><td>88.8</td><td>84.6</td><td>88.6</td><td>89.6</td><td>52.1</td><td>48.6</td></tr><tr><td>OCRBench</td><td>792</td><td>858</td><td>808</td><td>881</td><td>819</td><td>896</td><td>753</td><td>701</td></tr><tr><td>OCRBench_v2en</td><td>56.4</td><td>56.3</td><td>61.8</td><td>63.7</td><td>63.9</td><td>65.4</td><td>48.1</td><td>37.9</td></tr><tr><td>OCRBench_v2zh</td><td>51.9</td><td>53.0</td><td>55.8</td><td>57.6</td><td>59.2</td><td>61.2</td><td>33.6</td><td>27.3</td></tr><tr><td>CC-OCR</td><td>68.3</td><td>72.8</td><td>73.8</td><td>76.2</td><td>76.3</td><td>79.9</td><td>58.9</td><td>52.9</td></tr><tr><td>OmniDocBenchen</td><td>0.370</td><td>0.292</td><td>0.234</td><td>0.244</td><td>0.209</td><td>0.170</td><td>0.401</td><td>0.454</td></tr><tr><td>OmniDocBenchzh</td><td>0.447</td><td>0.348</td><td>0.297</td><td>0.285</td><td>0.253</td><td>0.264</td><td>0.518</td><td>0.568</td></tr><tr><td>CharXiv(DQ)</td><td>70.1</td><td>62.3</td><td>83.9</td><td>76.2</td><td>85.9</td><td>83.0</td><td>82.0</td><td>64.4</td></tr><tr><td>CharXiv(RQ)</td><td>37.1</td><td>26.8</td><td>50.3</td><td>39.7</td><td>53.0</td><td>46.4</td><td>50.1</td><td>31.7</td></tr><tr><td>MMLongBenchDoc</td><td>33.8</td><td>31.6</td><td>44.4</td><td>43.5</td><td>48.0</td><td>47.9</td><td>31.8</td><td>22.1</td></tr><tr><td rowspan="6">2D/3D<br>Grounding</td><td>RefCOCO-avg</td><td>84.8</td><td>85.6</td><td>88.2</td><td>89.0</td><td>88.2</td><td>89.1</td><td>-</td><td>-</td></tr><tr><td>CountBench</td><td>84.1</td><td>88.4</td><td>89.4</td><td>84.9</td><td>91.5</td><td>80.5</td><td>80.0</td><td>62.9</td></tr><tr><td>OdinW-13</td><td>36.0</td><td>43.4</td><td>39.4</td><td>48.2</td><td>39.8</td><td>44.7</td><td>-</td><td>-</td></tr><tr><td>ARKitScenes</td><td>47.7</td><td>56.2</td><td>46.3</td><td>56.6</td><td>46.6</td><td>56.8</td><td>-</td><td>-</td></tr><tr><td>Hypersim</td><td>11.2</td><td>12.0</td><td>11.9</td><td>12.2</td><td>12.0</td><td>12.7</td><td>-</td><td>-</td></tr><tr><td>SUNRGBD</td><td>28.6</td><td>33.8</td><td>28.0</td><td>34.7</td><td>30.4</td><td>36.2</td><td>-</td><td>-</td></tr><tr><td rowspan="4">Embodied/Spatial<br>Understanding</td><td>ERQA</td><td>41.8</td><td>28.3</td><td>47.3</td><td>41.3</td><td>46.8</td><td>45.8</td><td>45.8</td><td>37.8</td></tr><tr><td>VSI-Bench</td><td>48.0</td><td>53.9</td><td>55.2</td><td>59.3</td><td>56.6</td><td>59.4</td><td>15.4</td><td>27.0</td></tr><tr><td>EmbSpatialBench</td><td>75.9</td><td>69.2</td><td>80.7</td><td>79.6</td><td>81.1</td><td>78.5</td><td>74.2</td><td>50.7</td></tr><tr><td>RefSpatialBench</td><td>28.9</td><td>30.3</td><td>45.3</td><td>46.6</td><td>44.6</td><td>54.2</td><td>12.6</td><td>2.5</td></tr><tr><td rowspan="2"></td><td>RoboSpatialHome</td><td>45.3</td><td>49.1</td><td>63.2</td><td>61.7</td><td>62.0</td><td>66.9</td><td>46.1</td><td>44.8</td></tr><tr><td>Multi-Image</td><td>BLINK<br>MUIRBENCH</td><td>57.2<br>68.1</td><td>53.8<br>47.4</td><td>63.4<br>75.0</td><td>65.8<br>63.8</td><td>64.7<br>76.8</td><td>69.1<br>64.4</td><td>58.3<br>45.7</td></tr><tr><td rowspan="6">Video<br>Understanding</td><td>MVBench</td><td>64.5</td><td>61.7</td><td>69.3</td><td>68.9</td><td>69.0</td><td>68.7</td><td>-</td><td>-</td></tr><tr><td>Video-MME\((W/o sub.\)</td><td>62.1</td><td>61.9</td><td>68.9</td><td>69.3</td><td>71.8</td><td>71.4</td><td>66.2</td><td>49.4</td></tr><tr><td>MLVU\(\vert W_{M}\)－Avg</td><td>69.2</td><td>68.3</td><td>75.7</td><td>75.3</td><td>75.1</td><td>78.1</td><td>69.2</td><td>52.6</td></tr><tr><td>LVBench</td><td>47.6</td><td>47.4</td><td>53.5</td><td>56.2</td><td>55.8</td><td>58.0</td><td>-</td><td>-</td></tr><tr><td>Charades-StatMoU</td><td>56.9</td><td>54.5</td><td>59.0</td><td>55.5</td><td>59.9</td><td>56.0</td><td>-</td><td>-</td></tr><tr><td>VideoMMM</td><td>54.1</td><td>41.9</td><td>69.4</td><td>56.2</td><td>72.8</td><td>65.3</td><td>63.0</td><td>40.2</td></tr><tr><td rowspan="2">Perception<br>with Tool</td><td>MMVU</td><td>48.9</td><td>41.7</td><td>58.6</td><td>50.5</td><td>62.0</td><td>58.7</td><td>63.1</td><td>51.0</td></tr><tr><td>\(V^{*}\)</td><td>69.1</td><td>75.9+</td><td>74.9</td><td>88.0+</td><td>77.5</td><td>90.1+</td><td>-</td><td>-</td></tr><tr><td rowspan="4">Multi-Modal<br>Agent</td><td>HRBench4K</td><td>69.4</td><td>72.6+</td><td>73.5</td><td>81.3+</td><td>72.4</td><td>82.3+</td><td>-</td><td>-</td></tr><tr><td>HRBench\(8K\)</td><td>62.6</td><td>68.9+</td><td>67.1</td><td>74.4+</td><td>68.1</td><td>78.0+</td><td>-</td><td>-</td></tr><tr><td>ScreenSpot Pro</td><td>32.2</td><td>48.5</td><td>49.2</td><td>59.5</td><td>46.6</td><td>54.6</td><td>-</td><td>-</td></tr><tr><td>OSWorldG</td><td>41.8</td><td>46.1</td><td>53.9</td><td>58.2</td><td>56.7</td><td>58.2</td><td>-</td><td>-</td></tr><tr><td rowspan="4">Understanding</td><td>AndroidWorld</td><td>46.1</td><td>36.4</td><td>52.0</td><td>45.3</td><td>50.0</td><td>47.6</td><td>-</td><td>-</td></tr><tr><td>OSWorld</td><td>19.0</td><td>17.0</td><td>31.4</td><td>26.2</td><td>33.9</td><td>33.9</td><td>-</td><td>-</td></tr><tr><td>WindowsAA</td><td>-</td><td>-</td><td>35.5</td><td>23.4</td><td>24.1</td><td>28.8</td><td>-</td><td>-</td></tr></table>

以下是原文 Table 5 的结果：

<table><tr><td rowspan="2"></td><td rowspan="2">Benchmark</td><td rowspan="2">Qwen3-VL 235B-A22B<br>Instruct</td><td rowspan="2">Qwen3<br>235B-A22B<br>Instruct-2507</td><td rowspan="2">Deepseek V3<br>0324</td><td rowspan="2">Claude-Opus-4<br>(Without thinking)</td></tr><tr></tr><tr><td rowspan="4">Knowledge</td><td>MMLU-Pro</td><td>81.8</td><td>83.0</td><td>81.2</td><td>86.6</td></tr><tr><td>MMLU-Redux</td><td>92.2</td><td>93.1</td><td>90.4</td><td>94.2</td></tr><tr><td>GPQA</td><td>74.3</td><td>77.5</td><td>68.4</td><td>74.9</td></tr><tr><td>SuperGPQA</td><td>60.4</td><td>62.6</td><td>57.3</td><td>56.5</td></tr><tr><td rowspan="3">Reasoning</td><td>AIME-25</td><td>74.7</td><td>70.3</td><td>46.6</td><td>33.9</td></tr><tr><td>HMMT-25</td><td>57.4</td><td>55.4</td><td>27.5</td><td>15.9</td></tr><tr><td>LiveBench 2024-11-25</td><td>74.8</td><td>75.4</td><td>66.9</td><td>74.6</td></tr><tr><td rowspan="4">Alignment<br>Tasks</td><td>IFEval</td><td>87.8</td><td>88.7</td><td>82.3</td><td>87.4</td></tr><tr><td>Arena-4 HarrisV2 (winnrate)</td><td>77.4</td><td>79.2</td><td>45.6</td><td>51.5</td></tr><tr><td>Creative Writing v3</td><td>86.5</td><td>87.5</td><td>81.6</td><td>83.8</td></tr><tr><td>WritingBench</td><td>85.5</td><td>85.2</td><td>74.5</td><td>79.2</td></tr><tr><td rowspan="2">Coding &Agen</td><td>LiveCodeBench v6</td><td>54.3</td><td>51.8</td><td>45.2</td><td>44.6</td></tr><tr><td>BFCL-v3</td><td>67.7</td><td>70.9</td><td>64.7</td><td>60.1</td></tr><tr><td rowspan="4">Multilingualism</td><td>MultiIF</td><td>76.3</td><td>77.5</td><td>66.5</td><td>-</td></tr><tr><td>MMLU-ProX</td><td>77.8</td><td>79.4</td><td>75.8</td><td></td></tr><tr><td>INCLUDE</td><td>80.0</td><td>79.5</td><td>80.1</td><td>-</td></tr><tr><td>PolyMATH</td><td>45.1</td><td>50.2</td><td>32.2</td><td>30.0</td></tr></table>

以下是原文 Table 6 的结果：

<table><tr><td></td><td>Benchmark</td><td>Qwen3-VL 235B-A22B Thinking</td><td>Qwen3 235B-A22B Thinking-2507</td><td>OpenAI 03 (medium)</td><td>Claude-Opus-4 (With thinking)</td></tr><tr><td rowspan="4">Knowledge</td><td>MMLU-Pro</td><td>83.8</td><td>84.4</td><td>85.9</td><td>-</td></tr><tr><td>MMLU-Redux</td><td>93.7</td><td>93.8</td><td>94.9</td><td>94.6</td></tr><tr><td>GPQA</td><td>77.1</td><td>81.1</td><td>83.3(high)</td><td>79.6</td></tr><tr><td>SuperGPQA</td><td>64.3</td><td>64.9</td><td>-</td><td>-</td></tr><tr><td rowspan="3">Reasoning</td><td>AIME-25</td><td>89.7</td><td>92.3</td><td>88.9(high)</td><td>75.5</td></tr><tr><td>HMMT-25</td><td>77.4</td><td>83.9</td><td>77.5</td><td>58.3</td></tr><tr><td>LiveBench 2024-11-25</td><td>79.6</td><td>78.4</td><td>78.3</td><td>78.2</td></tr><tr><td rowspan="3">Coding</td><td>LiveCodeBench v6</td><td>70.1</td><td>74.1</td><td>58.6</td><td>48.9</td></tr><tr><td>CFEval</td><td>1964</td><td>2134</td><td>2043</td><td>-</td></tr><tr><td>OJBench</td><td>27.5</td><td>32.5</td><td>25.4</td><td>-</td></tr><tr><td rowspan="4">Alignment Tasks</td><td>IFEval</td><td>88.2</td><td>87.8</td><td>92.1</td><td>89.7</td></tr><tr><td>Arena-Hard V2 (winrnte)</td><td>74.8</td><td>79.7</td><td>80.8</td><td>59.1</td></tr><tr><td>Creative Writing v3</td><td>85.7</td><td>86.1</td><td>87.7</td><td>83.8</td></tr><tr><td>WritingBench</td><td>86.7</td><td>88.3</td><td>85.3</td><td>79.1</td></tr><tr><td rowspan="4">Agent</td><td>BFCL-v3</td><td>71.8</td><td>71.9</td><td>72.4</td><td>61.8</td></tr><tr><td>TAU2-Retail</td><td>67.0</td><td>71.9</td><td>76.3</td><td>-</td></tr><tr><td>TAU2-Airline</td><td>62.0</td><td>58.0</td><td>70.0</td><td>-</td></tr><tr><td>TAU2-Telecom</td><td>44.7</td><td>45.6</td><td>60.5</td><td>-</td></tr><tr><td rowspan="4">Multilingualism</td><td>MultiIF</td><td>79.1</td><td>80.6</td><td>80.3</td><td>-</td></tr><tr><td>MMLU-ProX</td><td>80.6</td><td>81.0</td><td>83.3</td><td>-</td></tr><tr><td>INCLUDE</td><td>80.0</td><td>81.0</td><td>86.6</td><td>-</td></tr><tr><td>PolyMATH</td><td>57.8</td><td>60.1</td><td>49.7</td><td>-</td></tr></table>

以下是原文 Table 7 的结果：

<table><tr><td rowspan="2"></td><td rowspan="2">Benchmark</td><td rowspan="2">Qwen3-32B <br>Instruct</td><td rowspan="2">Qwen3 32B <br>rstruct</td><td rowspan="2">Qwen3-30B-A3B <br>Instruct</td><td rowspan="2">Qwen3 30B-A3B <br>Instruct</td><td></td></tr><tr><td>Instruct-2507</td></tr><tr><td rowspan="4">Knowledge</td><td>MMLU-Pro</td><td>78.6</td><td>71.9</td><td>77.8</td><td>69.1</td><td>78.4</td></tr><tr><td>MMLU-Redux</td><td>89.8</td><td>85.7</td><td>88.4</td><td>84.1</td><td>89.3</td></tr><tr><td>GPQA</td><td>68.9</td><td>54.6</td><td>70.4</td><td>54.8</td><td>70.4</td></tr><tr><td>SuperGPQA</td><td>54.6</td><td>43.2</td><td>53.1</td><td>42.2</td><td>53.4</td></tr><tr><td rowspan="3">Reasoning</td><td>AIME-25</td><td>66.2</td><td>20.2</td><td>69.3</td><td>21.6</td><td>61.3</td></tr><tr><td>HMMT-25</td><td>46.1</td><td>10.9</td><td>50.6</td><td>12.0</td><td>43.0</td></tr><tr><td>LiveBench 2024-11-25</td><td>72.2</td><td>31.3</td><td>65.4</td><td>59.4</td><td>69.0</td></tr><tr><td rowspan="3">Alignment Tasks</td><td>IFEval</td><td>84.7</td><td>83.2</td><td>85.8</td><td>83.7</td><td>84.7</td></tr><tr><td>Arena-Hard V2 (winnte)</td><td>64.7</td><td>37.4</td><td>58.5</td><td>24.8</td><td>69.0</td></tr><tr><td>Creative Writing v3</td><td>85.6</td><td>80.6</td><td>84.6</td><td>68.1</td><td>86.0</td></tr><tr><td></td><td>WritingBench</td><td>82.9</td><td>81.3</td><td>82.6</td><td>72.2</td><td>85.5</td></tr><tr><td rowspan="2">Coding &amp;amp; Agent</td><td>LiveCodeBench v6</td><td>43.8</td><td>29.1</td><td>42.6</td><td>29.0</td><td>43.2</td></tr><tr><td>BFCL-v3</td><td>70.2</td><td>63.0</td><td>66.3</td><td>58.6</td><td>65.1</td></tr><tr><td rowspan="4">Multilingualism</td><td>MultiIF</td><td>72.0</td><td>70.7</td><td>66.1</td><td>70.8</td><td>67.9</td></tr><tr><td>MMLU-ProX</td><td>73.4</td><td>69.3</td><td>70.9</td><td>65.1</td><td>72.0</td></tr><tr><td>INCLUDE</td><td>74.0</td><td>69.6</td><td>71.6</td><td>67.8</td><td>71.9</td></tr><tr><td>PolyMATH</td><td>40.5</td><td>22.5</td><td>44.3</td><td>23.3</td><td>43.1</td></tr></table>

以下是原文 Table 8 的结果：

<table><tr><td rowspan="2" colspan="2">Benchmark</td><td>Qwen3-<br>32B</td><td>Qwen3-<br>32B</td><td>Qwen3-<br>30B-A3B</td><td>Qwen3-<br>30B-A3B</td><td>Qwen3-<br>30B-A3B</td></tr><tr><td>Thinking</td><td> Thinking</td><td>Thinking</td><td> Thinking</td><td>Thinking</td></tr><tr><td rowspan="4">Knowledge</td><td>MMLU-Pro</td><td>82.1</td><td>79.1</td><td>80.5</td><td>78.5</td><td>80.9</td></tr><tr><td>MMLU-Redux</td><td>91.9</td><td>90.9</td><td>90.9</td><td>89.5</td><td>91.4</td></tr><tr><td>GPQA</td><td>73.1</td><td>68.4</td><td>74.4</td><td>65.8</td><td>73.4</td></tr><tr><td>SuperGPQA</td><td>59.0</td><td>54.1</td><td>56.4</td><td>51.8</td><td>56.8</td></tr><tr><td rowspan="3">Reasoning</td><td>AIME-25</td><td>83.7</td><td>72.9</td><td>83.1</td><td>70.9</td><td>85.0</td></tr><tr><td>HMMT-25</td><td>64.6</td><td>51.8</td><td>67.6</td><td>49.8</td><td>71.4</td></tr><tr><td>LiveBench 2024-11-25</td><td>74.7</td><td>65.7</td><td>72.1</td><td>74.3</td><td>76.8</td></tr><tr><td rowspan="3">Coding</td><td>LiveCodeBench v6</td><td>65.6</td><td>60.6</td><td>64.2</td><td>57.4</td><td>66.0</td></tr><tr><td>CFEval</td><td>1842</td><td>1986</td><td>1894</td><td>1940</td><td>2044</td></tr><tr><td>QBench</td><td>20.0</td><td>24.1</td><td>23.4</td><td>20.7</td><td>25.1</td></tr><tr><td rowspan="4">Alignment Tasks</td><td>IFEval</td><td>87.8</td><td>85.0</td><td>81.7</td><td>86.5</td><td>88.9</td></tr><tr><td>Arena-Hard V2 (winrate)</td><td>60.5</td><td>50.3</td><td>56.7</td><td>36.3</td><td>56.0</td></tr><tr><td>Creative Writing v3</td><td>83.3</td><td>84.4</td><td>82.5</td><td>79.1</td><td>84.4</td></tr><tr><td>WritingBench</td><td>86.2</td><td>78.4</td><td>85.2</td><td>77.0</td><td>85.0</td></tr><tr><td rowspan="4">Agent</td><td>BFCL-v3</td><td>71.7</td><td>70.3</td><td>68.6</td><td>69.1</td><td>72.4</td></tr><tr><td>TAU2-Retail</td><td>59.4</td><td>59.6</td><td>64.0</td><td>34.2</td><td>58.8</td></tr><tr><td>TAU2-Airline</td><td>52.5</td><td>38.0</td><td>48.0</td><td>36.0</td><td>58.0</td></tr><tr><td>TAU2-Telecom</td><td>46.9</td><td>26.3</td><td>27.2</td><td>22.8</td><td>26.3</td></tr><tr><td rowspan="4">Multilingualism</td><td>MultiIF</td><td>78.0</td><td>73.0</td><td>73.0</td><td>72.2</td><td>76.4</td></tr><tr><td>MMLU-ProX</td><td>77.2</td><td>74.6</td><td>76.1</td><td>73.1</td><td>76.4</td></tr><tr><td>INCLUDE</td><td>76.3</td><td>73.7</td><td>74.5</td><td>71.9</td><td>74.4</td></tr><tr><td>PolyMATH</td><td>52.0</td><td>47.4</td><td>51.7</td><td>46.1</td><td>52.6</td></tr></table>

以下是原文 Table 9 的结果：

<table><tr><td rowspan="2" colspan="2">Benchmark</td><td>Qwen3-VL 2B</td><td>Qwen3-VL 4B</td><td>Qwen3-VL 8B</td><td>Qwen3-VL 1.7B</td><td>Qwen3 4B</td><td>Qwen3 8B</td><td>Qwen3 4B</td></tr><tr><td>Instruct</td><td>Instruct</td><td>Instruct</td><td>Instruct</td><td>Instruct</td><td>Instruct</td><td>Instruct</td><td>Instruct-2507</td></tr><tr><td rowspan="4">Knowledge</td><td>MMLU-Pro</td><td>49.0</td><td>67.1</td><td>71.6</td><td>42.3</td><td>58.0</td><td>63.4</td><td>69.6</td></tr><tr><td>MMLU-Redux</td><td>66.5</td><td>81.5</td><td>84.9</td><td>63.6</td><td>77.3</td><td>79.5</td><td>84.2</td></tr><tr><td>GPQA</td><td>42.0</td><td>55.9</td><td>61.9</td><td>34.7</td><td>41.7</td><td>39.3</td><td>62.0</td></tr><tr><td>SuperGPQA</td><td>24.3</td><td>40.3</td><td>44.5</td><td>22.8</td><td>32.0</td><td>35.8</td><td>42.8</td></tr><tr><td rowspan="3">Reasoning</td><td>AIME-25</td><td>22.2</td><td>46.6</td><td>45.9</td><td>10.6</td><td>19.1</td><td>20.9</td><td>47.4</td></tr><tr><td>HMMT-25</td><td>10.9</td><td>30.7</td><td>32.5</td><td>6.2</td><td>12.1</td><td>11.8</td><td>31.0</td></tr><tr><td>LiveBench 2024-11-25</td><td>39.5</td><td>60.9</td><td>62.0</td><td>35.6</td><td>48.4</td><td>53.5</td><td>63.0</td></tr><tr><td rowspan="3">Alignment Tasks</td><td>IFEval</td><td>68.2</td><td>82.3</td><td>83.7</td><td>67.1</td><td>81.2</td><td>83.0</td><td>83.4</td></tr><tr><td>Arena-Hard V2 (winrate)</td><td>6.4</td><td>30.4</td><td>46.3</td><td>4.1</td><td>9.5</td><td>15.5</td><td>43.4</td></tr><tr><td>Creative Writing v3</td><td>48.6</td><td>72.3</td><td>77.0</td><td>49.1</td><td>53.6</td><td>69.0</td><td>83.5</td></tr><tr><td></td><td>ADQ</td><td>79.2</td><td>83.5</td><td>83.1</td><td>65.1</td><td>68.5</td><td>71.4</td><td>83.4</td></tr><tr><td rowspan="3">Coding &amp;amp; Agent</td><td>LiveCodeBench v6</td><td>20.3</td><td>37.9</td><td>39.3</td><td>16.1</td><td>26.4</td><td>25.5</td><td>35.1</td></tr><tr><td>BFCL-v3</td><td>55.4</td><td>63.3</td><td>66.3</td><td>52.2</td><td>57.6</td><td>60.2</td><td>61.9</td></tr><tr><td>MultiIF</td><td>43.2</td><td>61.5</td><td>66.8</td><td>43.2</td><td>61.3</td><td>69.2</td><td>69.0</td></tr><tr><td rowspan="3">Multilingualism</td><td>MMLU-ProX</td><td>38.8</td><td>59.4</td><td>65.4</td><td>33.5</td><td>49.6</td><td>58.0</td><td>61.6</td></tr><tr><td>INCLUDE</td><td>45.8</td><td>61.4</td><td>67.0</td><td>42.6</td><td>53.8</td><td>62.5</td><td>60.1</td></tr><tr><td>PolyMATH</td><td>14.9</td><td>28.8</td><td>30.4</td><td>10.3</td><td>16.6</td><td>18.8</td><td>31.1</td></tr></table>

以下是原文 Table 10 的结果：

<table><tr><td rowspan="2" colspan="2"></td><td>Qwen3-VL<br>2B</td><td>Qwen3-VL<br>4B</td><td>Qwen3-VL<br>8B</td><td>Qwen3-LR</td><td>Qwen3-Qwen3<br>4B</td><td>Qwen3-BB</td><td rowspan="2">Qwen3-Qew3-BB</td><td rowspan="2">Qwen3Qew3-BB</td></tr><tr><td>Thinking</td><td>Thinking</td><td>Thinking</td><td>Thinking</td><td>Thinking</td><td>Thinking</td></tr><tr><td rowspan="4">Knowledge</td><td>MMLU-Pro</td><td>62.3</td><td>73.6</td><td>77.3</td><td>58.1</td><td>70.4</td><td>74.6</td><td>74.0</td><td></td></tr><tr><td>MMLU-Redux</td><td>76.9</td><td>86.0</td><td>88.8</td><td>73.9</td><td>83.7</td><td>87.5</td><td>86.1</td><td></td></tr><tr><td>GPQA</td><td>49.5</td><td>64.1</td><td>69.9</td><td>27.9</td><td>55.9</td><td>62.0</td><td>65.8</td><td></td></tr><tr><td>SuperGPQA</td><td>34.6</td><td>46.8</td><td>51.2</td><td>31.2</td><td>42.7</td><td>47.6</td><td>47.8</td><td></td></tr><tr><td rowspan="4">Reasoning</td><td>AIME-25</td><td>39.0</td><td>74.5</td><td>80.3</td><td>36.8</td><td>65.6</td><td>67.3</td><td>81.3</td><td rowspan="2"></td></tr><tr><td>HMMT-25</td><td>22.8</td><td>53.1</td><td>60.6</td><td>24.3</td><td>42.1</td><td>43.2</td><td>55.5</td></tr><tr><td>LiveBench 2024-11-25</td><td>50.1</td><td>68.4</td><td>69.8</td><td>51.1</td><td>63.6</td><td>67.1</td><td>71.8</td><td></td></tr><tr><td>IFEval</td><td>75.1</td><td>82.6</td><td>83.2</td><td>72.5</td><td>81.9</td><td>85.0</td><td>87.4</td><td></td></tr><tr><td rowspan="3">Alignment Tasks</td><td>Arena-hard V2 (winrate)</td><td>12.0</td><td>36.8</td><td>51.1</td><td>4.7</td><td>13.7</td><td>29.1</td><td>34.9</td><td></td></tr><tr><td>Creative Writing v3</td><td>57.6</td><td>76.1</td><td>82.4</td><td>50.6</td><td>61.1</td><td>78.5</td><td>75.6</td><td></td></tr><tr><td>WordginBench</td><td>77.9</td><td>84.0</td><td>85.5</td><td>68.9</td><td>73.5</td><td>75.0</td><td>83.3</td><td></td></tr><tr><td rowspan="2">Coding &amp;amp; Agent</td><td>LiveCodeBench v6</td><td>29.3</td><td>51.3</td><td>58.6</td><td>31.3</td><td>48.4</td><td>51.0</td><td>55.2</td><td></td></tr><tr><td>RFCL-v3</td><td>57.2</td><td>67.3</td><td>63.0</td><td>56.6</td><td>65.9</td><td>68.1</td><td>71.2</td><td></td></tr><tr><td rowspan="4">Multilingualism</td><td>MultiIF</td><td>58.9</td><td>73.6</td><td>751</td><td>51.2</td><td>66.3</td><td>71.2</td><td>77.3</td><td rowspan="4"></td></tr><tr><td>MMLU-Prox</td><td>55.1</td><td>65.0</td><td>70.7</td><td>50.4</td><td>61.0</td><td>68.1</td><td>64.2</td></tr><tr><td>INCLUDE</td><td>53.3</td><td>64.6</td><td>69.5</td><td>51.8</td><td>61.8</td><td>67.8</td><td>64.4</td></tr><tr><td>PolyMATH</td><td>28.0</td><td>44.6</td><td>47.5</td><td>25.2</td><td>40.0</td><td>42.7</td><td>46.2</td></tr></table>

## 6.3. 消融实验/参数分析

### 6.3.1. 视觉编码器 (Vision Encoder)
通过比较 `Qwen3-ViT` 和 `SigLip-2`，验证了 `Qwen3-ViT` 作为更强视觉主干网络的有效性。

以下是原文 Table 11 的结果：

<table><tr><td>ViT</td><td colspan="7">Clip Bench ImageNet-1K ImageNet-V2 ImageNet-A ImageNet-R ImageNet-S ObjectNet Omni</td><td colspan="4">VLM Bench</td></tr><tr><td>SigLip-2</td><td>84.2</td><td>78.6</td><td>87.0</td><td>96.1</td><td>76.2</td><td>79.9</td><td>36.9</td><td>77.2</td><td>78.1</td><td>85.7</td><td>65.3</td><td>50.1</td></tr><tr><td>Qwen3-ViT</td><td>84.6</td><td>78.8</td><td>87.1</td><td>95.7</td><td>74.5</td><td>81.0</td><td>45.5</td><td>78.7</td><td>78.2</td><td>66.1</td><td>67.0</td><td>53.0</td></tr></table>

*   **CLIP 预训练阶段：** `Qwen3-ViT` 在标准基准测试中保持了竞争力，同时在内部的 `OmniBench` 上取得了显著提升，这表明其在整合世界知识方面的优势。
*   **VLM 阶段：** 当与相同的 1.7B Qwen3 语言模型集成并训练 1.5T 词元后，`Qwen3-ViT` 在多个关键任务上持续超越基于 `SigLip-2` 的基线，并在 `OmniBench` 上保持显著领先，证实了其作为更强视觉主干网络的优越性和有效性。

### 6.3.2. DeepStack
消融实验验证了 `DeepStack` 机制的有效性。

以下是原文 Table 12 的结果：

<table><tr><td>Method</td><td>AVG</td><td>AI2D</td><td>OCR</td><td>TVQA</td><td>InfoVQA</td><td>ChartQA</td><td>DocVQA</td><td>MMMU</td><td>MMStar</td><td>RLWDQA</td><td>MMBN</td><td>MMBNN</td></tr><tr><td>Baseline</td><td>74.7</td><td>81.8</td><td>81.0</td><td>80.6</td><td>71.9</td><td>81.5</td><td>89.5</td><td>52.9</td><td>55.5</td><td>67.7</td><td>81.0</td><td>78.1</td></tr><tr><td>DeepStack</td><td>76.0</td><td>83.2</td><td>83.6</td><td>80.5</td><td>74.2</td><td>83.3</td><td>91.1</td><td>54.1</td><td>57.7</td><td>68.1</td><td>81.2</td><td>78.5</td></tr></table>

*   **性能提升：** 配备 `DeepStack` 的模型在各种基准测试中都取得了整体性能提升，有力地证实了其有效性。
*   **原因分析：** 这种提升归因于 `DeepStack` 能够集成丰富的视觉信息，从而有效提升了细粒度视觉理解能力，例如在 `InfoVQA` 和 `DocVQA` 基准上的表现。

### 6.3.3. 干草堆中的针 (Needle-in-a-Haystack)
为了评估模型处理长上下文输入的能力，在 `Qwen3-VL-235B-A22B-Instruct` 上进行了视频“干草堆中的针”评估。

```markdown

![fig 4](images/4.jpg)
*该图像是一个比较图，展示了在不同训练上下文下（0-30分钟及40-120分钟）准确性得分与上下文长度的关系。左侧部分为训练上下文，右侧为外推上下文，各列分别代表上下文长度，纵轴为深度百分比。*

Figure 3: Needle-in-a-Haystack performance heatmap for Qwen3-VL-235B-A22B-Instruct across varying video durations and needle positions. Each cell shows accuracy \((\%)\) for locating and answering questions about the inserted "needle" frame.
```
*VLM 描述: 该图像是一个比较图，展示了在不同训练上下文下（0-30分钟及40-120分钟）准确性得分与上下文长度的关系。左侧部分为训练上下文，右侧为外推上下文，各列分别代表上下文长度，纵轴为深度百分比。*

*   **测试设置：** 一个语义显著的“针”帧（包含关键视觉证据）被插入到长视频的不同时间位置。模型任务是准确地从长视频中定位目标帧并回答相应问题。评估期间，视频以 1 FPS 均匀采样，帧分辨率动态调整以保持恒定的视觉词元预算。
*   **结果：** 模型在长达 30 分钟的视频上（对应 256K 词元上下文长度）实现了 100% 的完美准确率。即使通过 `YaRN` (Yet another RoPE extension) 进行位置扩展，将序列外推到长达 1M 词元（约 2 小时视频），模型仍保持 99.5% 的高准确率。这些结果有力地证明了模型强大的长序列建模能力。

# 7. 总结与思考

## 7.1. 结论总结
本工作推出了 Qwen3-VL，这是一个最先进的视觉-语言基础模型系列，推动了多模态理解和生成的边界。通过高质量多模态数据迭代和架构创新（如增强的交错式 MRoPE、DeepStack 视觉-语言对齐和基于文本的时间定位），Qwen3-VL 在广泛的多模态基准测试中取得了前所未有的性能，同时保持了强大的纯文本能力。其对 256K 词元交错序列的原生支持，使得对长而复杂文档、图像序列和视频的鲁棒推理成为可能，使其独特地适用于需要高保真跨模态理解的真实世界应用。密集型和专家混合变体的可用性确保了在不同延迟和质量要求下的灵活部署，而分阶段后训练策略（包括非思考型和思考型模式）进一步提升了模型能力。

## 7.2. 局限性与未来工作
论文作者指出了 Qwen3-VL 的潜在局限性，并提出了以下未来研究方向：

1.  <strong>具身 AI (Embodied AI) 智能体：</strong> 尽管 Qwen3-VL 在智能体决策和 GUI 交互方面表现出色，但仍需进一步发展，以实现与物理世界的无缝连接，使 AI 智能体不仅能感知和推理，还能在动态环境中执行果断、情境感知的行动，例如与用户交互、操作数字界面和通过具身、多模态决策指导机器人系统。
2.  <strong>交互式感知 (Interactive Perception)：</strong> 未来的工作将专注于扩展 Qwen3-VL 的能力，以实现交互式感知，即模型能够主动探索环境以获取更多信息，而不仅仅是被动地接收输入。
3.  <strong>工具增强型推理 (Tool-Augmented Reasoning) 和实时多模态控制 (Real-Time Multimodal Control)：</strong> 虽然工具集成已显示出巨大潜力，但仍需进一步研究如何更有效地将外部工具和实时控制机制整合到多模态智能体中，以实现更复杂的任务和更快的响应速度。
4.  **统一的理解-生成架构：** 未来的目标是实现能够统一理解和生成的多模态架构，利用视觉生成能力进一步提升整体智能。这意味着模型不仅能理解多模态输入，还能基于这些理解生成新的图像、视频或多模态内容。

## 7.3. 个人启发与批判

**个人启发：**

1.  **多模态能力的反哺作用：** Qwen3-VL 在纯文本任务上超越了其纯文本主干网络，这一结果令人振奋。它暗示了多模态学习不仅仅是为模型增加了新的感知维度，更可能通过引入更丰富的世界知识和情境理解，反过来增强了模型的语言理解和推理能力。这为未来基础模型的开发提供了新的思路，即多模态预训练可能是一种更通用的智能提升途径。
2.  **长上下文的实用价值：** 原生 256K 词元上下文窗口以及在“干草堆中的针”任务中近乎完美的表现，彻底改变了我们对模型处理长文档和长视频的期待。这对于需要深度分析和交叉引用的专业领域（如法律、医学、工程）具有巨大的应用潜力。
3.  **架构细节的重要性：** `Interleaved MRoPE`、`DeepStack` 和基于文本的时间戳对齐等看似细节的架构创新，对于解决特定模态（如长视频时空建模、视觉-语言深层对齐）的挑战至关重要。这提醒我们，在追求大模型和大数据的同时，精妙的架构设计仍然是性能飞跃的关键。
4.  **智能体和工具集成的方向：** 论文强调工具集成带来的性能提升甚至超过了模型规模的简单扩大，这清晰地指明了多模态模型未来的发展方向——走向能够感知、推理并与环境（包括数字和物理环境）交互的智能体。

**批判与可以改进的地方：**

1.  **计算资源的可持续性：** 尽管 MoE 架构在一定程度上缓解了计算成本，但训练如此大规模（235B 参数）且拥有 256K 词元长上下文的模型，所需的计算资源仍然是天文数字。对于学术界和小型研究机构来说，复现和进一步研究的门槛极高。论文未详细讨论其碳足迹或如何优化能源效率，这在当前关注可持续 AI 的背景下是一个不足。
2.  <strong>“思考型”</strong>模式的泛化性与可控性： 思考型 (thinking) 模型通过生成中间推理步骤来提升性能，这固然有效，但其推理过程的可控性、鲁棒性以及在面对模糊、对抗性输入时的表现仍需进一步探究。链式思考的“幻觉” (hallucination) 问题在大模型中普遍存在，如何在多模态推理链中有效抑制这一问题，是一个持续的挑战。
3.  **评估基准的完备性：** 尽管论文使用了极其广泛的基准测试，但现实世界的复杂性往往超出任何静态基准。例如，在 GUI 智能体任务中，模型在动态、非结构化环境中的泛化能力，以及对用户意图的深层理解，仍需更具挑战性的评估方法来衡量。此外，多语言能力虽然有所扩展，但对于低资源语言的表现和文化敏感性，仍有待深入评估。
4.  **数据策展的透明度：** 论文提及了大规模、高质量的数据集和精细的过滤管道，但具体的数据构成、规模细节、清洗过程以及伦理考量（如数据偏见、隐私）的披露相对有限。更透明的数据报告将有助于社区理解模型的优势来源并进行负责任的部署。
5.  **推理延迟和实际部署：** 尽管 MoE 架构旨在平衡延迟和质量，但对于 235B-A22B 这样的旗舰模型，在现实世界中实现低延迟推理（尤其是在边缘设备或对实时性要求高的应用中）仍是一个挑战。论文可以更深入地探讨不同模型变体在实际部署场景中的性能和资源权衡。