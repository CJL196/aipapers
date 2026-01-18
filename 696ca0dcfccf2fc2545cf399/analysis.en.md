# 1. Bibliographic Information
## 1.1. Title
Enhancing Video Large Language Models with Structured Multi-Video Collaborative Reasoning

## 1.2. Authors
Zhihao He, Tianyao He, Yun Xu, Tieyuan Chen, Huabin Liu, Chaofan Gan, Zuxuan Wu, Weiyao Lin.
Their affiliations include Shanghai Jiao Tong University, Shanghai Innovation Institute, and Fudan University. Weiyao Lin is the corresponding author.

## 1.3. Journal/Conference
The paper is published as a preprint on arXiv (https://arxiv.org/abs/2509.13161). arXiv is a widely used open-access repository for preprints of scientific papers in various fields, including computer science. While it is not a peer-reviewed journal or conference proceeding in itself, many papers initially posted on arXiv are later submitted to and published in prestigious venues. The publication date (2025-09-16T15:13:21.000Z) indicates it's a recent submission.

## 1.4. Publication Year
2025

## 1.5. Abstract
The paper addresses the challenge of comprehensive video reasoning in video language models (VLMs), which is often hindered by the inherent spatio-temporal incompleteness and redundancy within individual videos, leading to inaccuracies and hallucinations. To overcome this, the authors propose a multi-video collaborative framework. Instead of directly feeding numerous and redundant video tokens from multiple related videos into a large language model (LLM), which can be counterproductive, their framework first establishes a `Video Structuring Module (VSM)` to represent video knowledge as an efficient spatio-temporal graph. Building on this structured representation, a `Graph Fusion Module (GFM)` is designed to fuse structured knowledge and valuable information from related videos into augmented graph node tokens. Finally, an elaborate multi-video structured prompt integrates these graph, visual, and textual tokens as input for the LLM. Extensive experiments demonstrate the framework's effectiveness in advancing VLM performance.

## 1.6. Original Source Link
https://arxiv.org/abs/2509.13161

## 1.7. PDF Link
https://arxiv.org/pdf/2509.13161v2

# 2. Executive Summary
## 2.1. Background & Motivation
The core problem this paper aims to solve is the inherent limitation of current video language models (VLMs) in achieving comprehensive and reliable video reasoning. While VLMs have shown impressive potential by integrating large language models (LLMs) with visual perception, they often suffer from "spatio-temporal incompleteness" within individual videos. This means that a single video might not contain all the necessary visual cues or context due to factors like sparse sampling, occlusions, or perspective changes, leading to the VLM resorting to linguistic priors and consequently generating "hallucinations" (fabricating information) or inaccurate answers.

This problem is important because reliable video understanding is crucial for many applications, from surveillance to content creation and human-computer interaction. The paper identifies a gap: while introducing multiple highly related videos could compensate for missing information, directly concatenating the visual features of multiple videos (a common approach in multi-modal LLMs for longer contexts) creates an "overwhelming number of tokens." This token burden leads to computational inefficiency and can cause LLMs to "get lost in the middle," focusing only on limited segments and neglecting critical information. The high dimensionality, redundancy, and unstructured nature of raw video content make direct integration particularly challenging.

The paper's entry point is to augment reasoning performance with multiple related videos but in a "structured" and "collaborative" manner, rather than a direct, brute-force concatenation. This innovative idea aims to extract and fuse only the "valuable information" from multiple videos into a concise, LLM-friendly format.

## 2.2. Main Contributions / Findings
The paper makes the following primary contributions:
*   **A Feasible Structured Multi-Video Collaborative Reasoning Framework:** It introduces a novel framework that enables VLMs to leverage information from multiple related videos in a structured way, effectively addressing the challenges of redundancy and token burden associated with direct multi-video input.
*   **Video Structuring Module (VSM):** It proposes the `Video Structuring Module` to transform raw video data into an efficient, structured spatio-temporal graph representation. This graph captures key objects and their relationships across time, making video information more manageable and LLM-friendly.
*   **Graph Fusion Module (GFM):** It designs the `Graph Fusion Module` to effectively fuse the structured knowledge from multiple related videos into augmented graph node tokens. This module integrates structural information using graph attention networks and identifies relevant cross-video information via `Cross-Graph Attention (CGA)`.
*   **Elaborate Multi-Video Structured Prompt:** It constructs a sophisticated prompt engineering strategy that integrates the fused graph tokens, original visual tokens (for the target video), and textual tokens into a cohesive input for the LLM, guiding the model to utilize the multi-video structured knowledge effectively.

    The key conclusion is that this structured multi-video collaborative approach significantly boosts the reliability and accuracy of video question answering across various benchmarks. By transforming complex, redundant video information into a data-efficient graph structure and intelligently fusing this information across multiple videos, the framework enables VLMs to overcome the limitations of single-video reasoning, leading to more comprehensive and accurate answers with minimal overhead. The findings demonstrate that structured knowledge integration is a promising avenue for advancing robust video understanding in VLMs.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
To fully understand this paper, a novice reader should be familiar with the following foundational concepts:

*   **Large Language Models (LLMs):** These are advanced artificial intelligence models trained on vast amounts of text data, enabling them to understand, generate, and process human language. They can perform tasks like question answering, summarization, and translation. Examples include GPT series, LLaMA, and Qwen. In this paper, LLMs provide the strong general knowledge and reasoning capabilities that VLMs leverage.

*   **Video Language Models (VLMs):** These models extend LLMs' capabilities to understand and reason about video content. They typically combine a visual encoder (to process video frames) with an LLM, often through a "projector" or "connector" module that aligns visual features with the language model's input space. VLMs aim to bridge the "semantic chasm" between video and language, allowing for nuanced integration of visual perception and language processing.

*   **Tokens:** In the context of LLMs and VLMs, "tokens" are the basic units of input or output. For language, tokens can be words, subwords, or characters. For vision, "visual tokens" are numerical representations extracted from images or video frames by a visual encoder. These tokens are what the LLM processes. A large number of tokens can lead to computational burden and the "lost in the middle" problem in LLMs.

*   **Graphs / Graph Neural Networks (GNNs):**
    *   **Graph:** A mathematical structure consisting of a set of `nodes` (also called `vertices`) and a set of `edges` (also called `links`) connecting pairs of nodes. In this paper, nodes represent objects or subjects in a video, and edges represent their relationships.
    *   **Graph Neural Network (GNN):** A type of neural network designed to operate on graph data. GNNs learn representations of nodes and edges by iteratively aggregating information from a node's neighbors. This allows them to capture structural relationships in data.
    *   **Graph Attention Network (GAT):** A specific type of GNN that uses the attention mechanism to learn the importance of different neighbors for each node. Instead of assigning a fixed weight to each neighbor, GATs compute attention coefficients that determine how much each neighbor's features contribute to the current node's new representation. This makes them powerful for learning complex relationships in graphs. The core idea of attention is to compute a weighted sum of values, where the weights are determined by a compatibility function between a query and keys. The general `Attention` mechanism, as introduced in "Attention Is All You Need" [60], is:
        \$
        \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
        \$
        Where:
        *   $Q$ (Query), $K$ (Key), $V$ (Value) are matrices representing the input sequences. For a single element, they would be vectors.
        *   $Q K^T$ calculates the dot product similarity between queries and keys.
        *   $\sqrt{d_k}$ is a scaling factor, where $d_k$ is the dimension of the keys, used to prevent the dot products from becoming too large and pushing the softmax function into regions with tiny gradients.
        *   $\mathrm{softmax}$ normalizes the scores to obtain attention weights.
        *   The attention weights are then multiplied by the values $V$ to get the final output.
            GATs adapt this concept to graph structures, where `Q, K, V` are derived from node features.

*   **Retrieval-Augmented Generation (RAG):** A technique that enhances the generation capabilities of LLMs by allowing them to retrieve relevant information from an external knowledge base (e.g., a database of documents or, in this paper, related videos) before generating a response. This helps LLMs access up-to-date and specific facts, reducing hallucinations and improving factual accuracy. In this paper, the retrieval of related videos and their structured knowledge acts as a form of RAG.

*   **Prompt Engineering:** The art and science of crafting effective inputs (prompts) for LLMs to guide their behavior and elicit desired outputs. This involves carefully designing instructions, examples, and context within the prompt. The paper designs an "elaborate multi-video structured prompt" to integrate various token types and guide the VLM's reasoning.

## 3.2. Previous Works
The paper contextualizes its contributions by discussing existing research in `Vision Language Models` and `Multi-Data Collaboration`.

### 3.2.1. Vision Language Models
The field has seen significant progress by aligning visual features with large language models.
*   **Feature Alignment and Instruction Tuning:** Many VLMs (e.g., `LLaVA` [25], `Video-LLaMA` [3], `BLIP-2` [37]) leverage pre-trained vision encoders (e.g., from CLIP) and LLMs, aligning their representations through modules like `Q-former` (BLIP-2) or simple `MLP-based projectors` (LLaVA). `Visual instruction tuning` is a key technique where models are fine-tuned on diverse visual-textual instruction pairs to develop generalist capabilities.
*   **Visual Token Reduction:** Due to the complexity and high dimensionality of visual inputs, many studies (e.g., `BLIP-2` [37], `NVILA` [45]) focus on reducing the number of visual tokens fed to the LLM to alleviate computational burden and improve understanding. Models like `Flamingo` [38] use `perceiver resamplers` for this purpose.
*   **Single-Video Reasoning:** The mainstream of VLMs, as depicted in Figure 1(a), operates on a single-video reasoning pipeline. While powerful, this approach is inherently limited by the information contained within that one video, leading to the problems of incompleteness and hallucination that this paper aims to address.

### 3.2.2. Multi-Data Collaboration
While traditional deep learning often focuses on single-data processing, multi-data collaboration offers a promising avenue for performance improvement by exploiting correspondences among multiple samples.
*   **Content-Related Collaboration:**
    *   `Co-segmentation` [30, 31]: Helps models segment the same object across different scenes by sharing object-related features.
    *   `Few-shot image classification` [46], `action recognition` [47, 48], `fine-grained classification` [49]: Methods that improve accuracy by comparing multiple data samples to identify key differences or commonalities.
    *   `Retrieval-Augmented Generation (RAG)` [33-35]: A prominent content-related collaboration where LLMs are provided with supporting information retrieved from related data to enhance their responses.
*   **Task-Related Collaboration:**
    *   `Multi-video summarization` [32, 50, 51]: Models learn to generate a summary from a collection of videos through complementary and refinement of information.
    *   `In-Context Learning` [52, 53]: LLMs learn to perform tasks by observing examples with task guidance and answers provided in the prompt.
*   **Limitations of existing multi-video collaboration:** The paper highlights that current multi-video collaboration strategies for VLMs often involve directly concatenating multiple inputs (as shown in Figure 1(b)). This leads to significant redundancy and makes effective collaboration difficult due to the "overwhelming number of tokens" and the "lost in the middle" phenomenon [20, 21] where LLMs struggle with long contexts.

## 3.3. Technological Evolution
The field has evolved from powerful `Large Language Models (LLMs)` to `Vision Language Models (VLMs)` by incorporating visual understanding capabilities. Initially, VLMs focused on aligning visual and textual representations for single images or videos. A key challenge in this evolution has been managing the high dimensionality and temporal complexity of video data, leading to techniques for `visual token reduction`. However, even with token reduction, single-video reasoning remains prone to incompleteness and hallucination. The next frontier involves leveraging multiple visual contexts. Early attempts at `multi-data collaboration` for VLMs involved simple concatenation of visual features, which quickly hit limitations due to token overload and the LLM's difficulty in processing very long contexts. This paper's work fits into this timeline as a crucial step towards more robust and scalable multi-video understanding. It moves beyond brute-force concatenation by proposing a `structured representation` approach, using `spatio-temporal graphs` to condense and organize multi-video knowledge efficiently before feeding it to the LLM. This represents an evolution towards more intelligent and data-efficient integration of multiple visual sources.

## 3.4. Differentiation Analysis
Compared to the main methods in related work, the core differences and innovations of this paper's approach are:
*   **Structured Representation vs. Direct Concatenation:** Unlike traditional multi-video approaches that directly concatenate raw visual tokens from multiple videos (e.g., "multi-video tokens" baseline in the ablation study), leading to an explosion in context length and computational burden, this paper proposes `Video Structuring Module (VSM)` to convert video content into a compact `spatio-temporal graph`. This graph represents knowledge in a data-efficient and LLM-friendly structured format.
*   **Graph-based Information Fusion:** Instead of relying on the LLM to implicitly derive relationships from raw concatenated tokens, the `Graph Fusion Module (GFM)` explicitly fuses structured knowledge from multiple videos at the graph level using `Hierarchical Frame Graph Attention Network (HF-GAT)` and `Cross-Graph Attention (CGA)`. This focused fusion mechanism allows for more targeted and effective integration of complementary information, especially across different videos.
*   **Overcoming "Lost in the Middle":** By presenting a condensed, structured representation to the LLM, the framework mitigates the "lost in the middle" problem, where LLMs struggle to focus on relevant information within very long, unstructured contexts. The structured prompt further guides the LLM on how to utilize this pre-processed multi-video knowledge.
*   **Reduced Overhead for Related Videos:** For related videos, the framework only uses their `structured graph tokens` in the prompt, entirely omitting their raw visual tokens. This drastically reduces the input token count for related videos while retaining valuable relational information, making multi-video reasoning practical and efficient. For the target video, it retains visual tokens for fine-grained details, showcasing a hybrid approach.
*   **Enhanced Reliability and Accuracy:** The structured collaboration directly addresses the `spatio-temporal incompleteness` and `redundancy` issues of single-video reasoning, leading to reduced hallucinations and improved accuracy, as demonstrated by the experimental results.

# 4. Methodology
## 4.1. Principles
The core idea of the proposed method is to overcome the limitations of single-video reasoning (spatio-temporal incompleteness, hallucinations) and the inefficiency of direct multi-video concatenation (token explosion, "lost in the middle") by transforming raw video data into an efficient, structured representation, and then intelligently fusing knowledge from multiple such structured representations. The theoretical basis is that by explicitly modeling objects, subjects, and their spatio-temporal relationships within and across videos as graphs, the most salient and complementary information can be extracted and presented to a large language model (LLM) in a concise and interpretable format. This graph-based approach allows for targeted information aggregation and cross-video knowledge transfer, enabling more robust and reliable reasoning.

## 4.2. Core Methodology In-depth (Layer by Layer)
The multi-video collaborative reasoning framework consists of three main components: the `Video Structuring Module (VSM)`, the `Graph Fusion Module (GFM)`, and the `Structured Multi-Video Prompt`. The overall framework is illustrated in Figure 3.

![Fig. 3: Multi-video collaborative reasoning framework. Together with the target video, $N$ related videos are retrieved to facilitate the reasoning process. First, we design the Video Structuring Module to obtain the structured video representation. Then, the Graph Fusion Module fuses the structure information and the related videos' information to get the video graph tokens. Finally, according to the designed prompts, the graph tokens, visual tokens, and text tokens are arranged as input to the large language model for question answering.](images/3.jpg)
*该图像是示意图，展示了多视频协作推理框架。该框架通过视频结构模块生成视频知识的时空图表示，再通过图融合模块将结构信息与相关视频的信息融合，以便更好地构建输入至大型语言模型的提示。*

Fig. 3: Multi-video collaborative reasoning framework. Together with the target video, $N$ related videos are retrieved to facilitate the reasoning process. First, we design the Video Structuring Module to obtain the structured video representation. Then, the Graph Fusion Module fuses the structure information and the related videos' information to get the video graph tokens. Finally, according to the designed prompts, the graph tokens, visual tokens, and text tokens are arranged as input to the large language model for question answering.

### 4.2.1. Setting of Multi-Video Reasoning
The setup involves a `target video` $V_0$ and $N$ `related videos` $\{V_1, V_2, \ldots, V_N\}$. These related videos are retrieved based on pre-constructed feature vectors, which will be discussed in Section 5. The goal is for the model to answer a question about $V_0$ with the aid of the $N$ retrieved videos.

### 4.2.2. Video Structuring Module (VSM)
The `Video Structuring Module (VSM)` is responsible for generating an efficient, structured spatio-temporal graph representation of each video. This module processes both the target video and related videos. The process is broken down into five steps:

#### Step 1: Scene Detection
To reduce redundancy, a lightweight, content-based scene detector, `Autoshot` [55], is used to segment each video into distinct scenes. From each detected scene, the middle frame is extracted as a `keyframe`. For a video $V_N$, its $M$ keyframes are denoted as $\mathcal{F}_N = \{F_1, F_2, \ldots, F_M\}$. These keyframes serve as the visual input for subsequent structuring.

#### Step 2: Dense Video Captioning
To acquire detailed textual concepts, a video large language model is employed to generate `comprehensive descriptions` for each input video. The authors use a specially designed prompt for this, as shown in Figure 4.

![Fig. 4: Video captioning prompts. We refer to the design outlined in \[54\] to create the prompts used to extract captions from videos. The prompts are divided into two parts: the system prompt and the user message. In the system prompt, we define the task of video captioning and provide corresponding guidelines along with a standardized output format. For the output format, the program randomly selects contents in green font as the normalized format for reference during each process of captioning. For the user message, we utilize $< V I D E O _ { - } T O K E N S >$ as the video tokens, and we provide a concise instruction to the model, then generate a detailed description for the video.](images/4.jpg)
*该图像是一个视频字幕生成的系统提示和用户消息示例，展示了如何分析视频帧中的叙事进程。系统提示包括任务说明和视频描述的指导原则，而用户消息则提供了具体的视频代币和详细描述的请求格式。这有助于生成高质量的视频描述。*

Fig. 4: Video captioning prompts. We refer to the design outlined in [54] to create the prompts used to extract captions from videos. The prompts are divided into two parts: the system prompt and the user message. In the system prompt, we define the task of video captioning and provide corresponding guidelines along with a standardized output format. For the output format, the program randomly selects contents in green font as the normalized format for reference during each process of captioning. For the user message, we utilize $< VIDEO\_TOKENS >$ as the video tokens, and we provide a concise instruction to the model, then generate a detailed description for the video.

The prompt includes a `system prompt` defining the video captioning task and a `user message` containing video tokens (`<VIDEO_TOKENS>`) and instructions to generate a detailed description.

#### Step 3: Textual Scene Graph Parsing
From the `dense video caption` generated in Step 2, a `textual scene graph` $\mathcal{G}^{\mathrm{Text}}$ is extracted using `SceneGraphParser` [56]. The original LLM in SceneGraphParser is replaced with `Qwen3-30-A3B` [57]. This textual scene graph consists of several `triplets` $\tau_i = \{s_i, p_i, o_i\}$, where:
*   $s_i$ is the `subject` (e.g., "a man").
*   $p_i$ is the `predicate` (e.g., "is riding").
*   $o_i$ is the `object` (e.g., "a bicycle").
    Each triplet represents an interaction or event (e.g., "man - riding - bicycle"), forming a foundational representation of relationships and dynamics within the video.

#### Step 4: Graph Information Filtering
To enhance data quality and remove irrelevant information, a filtering mechanism is applied to the extracted triplets. An `image-level classifier`, `SigLIP` [58], is used to verify the presence of the `object` or `subject` from each triplet within the corresponding `keyframe`. This is done by posing simple binary classification tasks using tailored prompts like:
*   "The object related to {object/subject} is in the image." (for positive samples)
*   "The object related to {object/subject} is not in the image." (for negative samples)
    Based on the classification results:
*   If both the `object` and `subject` are present in the scene, the triplet is retained.
*   If only the `object` or `subject` exists independently, a self-connection triplet of the form $\{s_i, *, s_i\}$ or $\{o_i, *, o_i\}$ is created to establish a node in the graph.
*   If neither is present, the triplet is discarded.
    This process yields a set of `filtered triplets` denoted as $\hat{\mathcal{G}}^{\mathrm{Text}}$.

#### Step 5: Video Graph Establishment
Based on the `filtered textual scene graph` $\hat{\mathcal{G}}^{\mathrm{Text}}$ and the `keyframes` $\mathcal{F}_{\{0, \ldots, N\}}$, a `graph-based structured video representation` is established for each video. This graph comprises:
*   **Nodes:** Representing the features of `objects` or `subjects` in the video. Initially, `Qwen3-Embedding` [59] extracts `text features` $\mathbf{T}$ from $\hat{\mathcal{G}}^{\mathrm{Text}}$. These text features are then combined with visual features using `Pooling Attention` and adaptive weighted fusion, which will be detailed in the `Graph Fusion Module`.
*   **Intra-frame edges:** Represent `spatial and interacting relationships` between objects/subjects within a single frame, derived from the `predicate` of each triplet (e.g., a directional link from $s_i$ to $o_i$).
*   **Inter-frame edges:** Link objects across different frames that share the `same subjects and objects`, modeling their `temporal relationships`. These are established based on the filtered results $\hat{\mathcal{G}}^{\mathrm{Text}}$ (e.g., linking $s_i^{t-1}$ to $s_i^t$ or $o_i^{t-1}$ to $o_i^t$ across consecutive frames). `Bidirectional links` are introduced for inter-frame connections to enhance content comprehension [61].

### 4.2.3. Graph Fusion Module (GFM)
The `Graph Fusion Module (GFM)` processes the structured video representations to generate LLM-friendly graph tokens. It consists of a `Triplet Embedding Module (TEM)` and a multi-layer stacked architecture containing `Hierarchical Frame Graph Attention Network (HF-GAT)` and `Cross-Graph Attention (CGA)`.

#### Triplet Embedding Module (TEM)
The TEM prepares the initial `triplet features` by enhancing them with `class embeddings` and integrating `visual information` through `Pooling Attention`.

*   **Class Embedding (CE):** To distinguish between the `target video` and `related videos`, `class embedding` is introduced. $\pmb{\alpha}$ is a learnable parameter of dimension $d$ (where $d$ is the feature dimension) shared across frames. The sigmoid function $\sigma$ is used to map $\pmb{\alpha}$ to values between 0 and 1.
    \$
    \mathrm{CE}_{tar} = \sigma (\pmb{\alpha})
    \$
    \$
    \mathrm{CE}_{rel} = 1 - \sigma (\pmb{\alpha})
    \$
    Here, $\mathrm{CE}_{tar}$ is the class embedding for the target video, and $\mathrm{CE}_{rel}$ is for the related videos. These embeddings are then directly added to the `text features` $\mathbf{T}_{tar}$ (from the target video) and $\mathbf{T}_{rel}$ (from related videos) that were initially extracted using `Qwen3-Embedding` in VSM.
    \$
    \mathbf{T}_{tar} = \mathbf{T}_{tar} + \mathbf{CE}_{tar}
    \$
    \$
    \mathbf{T}_{rel} = \mathbf{T}_{rel} + \mathbf{CE}_{rel}
    \$
    The enhanced text features from all videos are then concatenated:
    \$
    \mathbf{T} = [\mathbf{T}_{tar}, \mathbf{T}_{rel}]
    \$
    This concatenation `[,]` combines the triplets from the target and related videos.

*   **Pooling Attention:** To incorporate visual information from the `keyframes` corresponding to the triplets, `Pooling Attention` is used. This mechanism aggregates visual features from the keyframes, guided by the text features $\mathbf{T}$.
    \$
    \mathbf{Q} = \mathbf{T} \mathbf{W}_Q \in \mathbb{R}^{1 \times d}
    \$
    \$
    \mathbf{K} = \mathbf{I} \mathbf{W}_K \in \mathbb{R}^{(H_p \times W_p) \times d}
    \$
    \$
    \mathbf{V} = \mathbf{I} \mathbf{W}_V \in \mathbb{R}^{(H_p \times W_p) \times d}
    \$
    \$
    \tilde{\mathbf{I}} = \mathrm{softmax}(\mathbf{Q} \mathbf{K}^{\top} / \sqrt{d}) \mathbf{V} \in \mathbb{R}^{1 \times d}
    \$
    Where:
    *   $\mathbf{W}_Q \in \mathbb{R}^{d \times d}$, $\mathbf{W}_K \in \mathbb{R}^{d \times d}$, $\mathbf{W}_V \in \mathbb{R}^{d \times d}$ are learnable weight matrices for the query, key, and value transformations, respectively.
    *   $\mathbf{T}$ represents the text features (from the concatenated $\mathbf{T}$ above) which serve as the query.
    *   $\mathbf{I} \in \mathbb{R}^{(H_p \times W_p) \times d}$ represents the `visual features` extracted from the vision encoder of the VLM for a keyframe. $(H_p \times W_p)$ denotes the spatial dimension after vision encoder extraction.
    *   $\mathbf{Q}$ is the query, $\mathbf{K}$ is the key, and $\mathbf{V}$ is the value.
    *   $d$ is the feature dimension.
    *   $\tilde{\mathbf{I}}$ is the aggregated visual feature, now a single vector of dimension $d$.

*   **Adaptive Weighted Fusion:** The original text features $\mathbf{T}$ (extracted from `Qwen3-Embedding` and enhanced by `Class Embedding`) and the pooled visual features $\tilde{\mathbf{I}}$ are fused using `adaptive weights` $\beta \in \mathbb{R}^d$.
    \$
    \hat{\mathbf{T}} = \sigma (\beta) \odot \mathbf{T} + (1 - \sigma (\beta)) \odot \tilde{\mathbf{I}}
    \$
    Where:
    *   $\sigma (\beta)$ provides a learnable weighting factor for the text features.
    *   $(1 - \sigma (\beta))$ provides the complementary weighting factor for the visual features.
    *   $\odot$ denotes the `Hadamard product` (element-wise multiplication).
        This fusion balances the contributions of text and visual features, resulting in a robust representation $\hat{\mathbf{T}}$.

#### Multi-layer Architecture
The processed triplet features $\hat{\mathbf{T}}$ are then fed into a multi-layer architecture for graph information processing. Each layer integrates two components: `Hierarchical Frame Graph Attention Network (HF-GAT)` and `Cross-Graph Attention (CGA)`.

*   **Hierarchical Frame Graph Attention Network (HF-GAT):** This component is designed to fuse graph-based structured data *within a single video*. Unlike traditional GATs that assume explicit relationships, HF-GAT leverages the graph structure established by the VSM (nodes as subject/object features, intra-frame edges as predicates, inter-frame edges linking same subjects/objects across frames). It propagates and aggregates information between nodes based on these explicit intra-frame and inter-frame connections. The bidirectional inter-frame links further enhance the model's ability to comprehend temporal video content.

*   **Cross-Graph Attention (CGA) Mechanism:** After individual video features are structured by HF-GAT, CGA is introduced to `identify and fuse the most relevant information between videos`. This is implemented as a `self-attention mechanism` with `custom position IDs`. The design considers three principles:
    1.  **Non-exchangeable subject-object relationship within a triplet:** The VSM and HF-GAT inherently capture this.
    2.  **Unordered positional relationship between triplets within a video:** HF-GAT aggregates information based on connection relationships, implicitly encoding this.
    3.  **Non-exchangeable order of triplets across retrieved videos (based on relevance ranking):** This is addressed by assigning consistent position IDs within each video and dynamically adjusting them across retrieved videos. For example, triplets from the target video always get position ID 0, while related videos get IDs based on their retrieval relevance rank. These position IDs are then integrated using `RoPE (Rotary Position Embedding)` [62] within the CGA mechanism to effectively encode positional information.

*   **Other Design Choices:**
    *   `Residual connections` and `pre-normalization` (using `LayerNorm` [63]) are applied for both HF-GAT and CGA components, which are common practices in Transformer-based architectures to stabilize training and improve performance.
    *   Crucially, the `Feed-Forward Network (FFN)` is **excluded** from the layers. This decision aims to `preserve the invariance of aligned visual features` from the vision encoder, minimizing excessive feature shifting and ensuring a more linear relationship between the inputs and outputs of GFM [64].

        The output of the GFM are `graph tokens`, where each token corresponds to the node features of subjects or objects after being fused with the structured and multi-video information.

### 4.2.4. Structured Multi-Video Prompt
The final step involves creating an `LLM-friendly input` by integrating the fused `multi-video graph tokens` with `video tokens` and `text tokens`. This is achieved through an elaborately designed `structured multi-video prompt`, as illustrated in Figure 5.

![Fig. 5: Structured multi-video prompts. We properly integrate the multi-modal tokens, together with the prompt guidance, to form an LLM-friendly input.](images/5.jpg)
*该图像是示意图，展示了结构化多视频提示的集成方法，包括多模态标记及提示指导，形成适合大语言模型输入的格式。*

Fig. 5: Structured multi-video prompts. We properly integrate the multi-modal tokens, together with the prompt guidance, to form an LLM-friendly input.

The prompt structure is as follows:
*   **Target Video ($V_0$):**
    *   Its original `video tokens` (`<VIDEO_TOKENS>`) are maintained to preserve fine-grained visual details.
    *   Its `graph-based structured data` (`<GRAPH_TOKENS>`) is appended to highlight key objects and spatio-temporal relationships within $V_0$.
*   **Related Videos ($V_1, \ldots, V_N$):**
    *   Only their concise and `data-efficient graph-based structured data` (`<GRAPH_TOKENS>`) is included. The raw visual tokens from related videos are omitted, significantly reducing token count.
*   **Prompt Guidance:** The prompt explicitly indicates the relationships between the target video and related videos, and provides instructions on how the LLM should utilize this multi-video structured data for reasoning.

    This structured prompt enables the VLM to effectively leverage multi-video information by providing a curated, focused, and integrated input, thereby enhancing its ability to reason about and answer queries related to complex video content.

# 5. Experimental Setup
## 5.1. Datasets
### Training Dataset
*   **Source:** The training dataset for GFM is constructed based on the `LLaVA-Video-178K` dataset [54].
*   **Preprocessing:** The dataset undergoes the step-by-step preprocessing outlined in Section 3.2 (VSM steps). For videos in `LLaVA-Video-178K` that already have captions, the original captions are retained.
*   **Video Vectorization and Retrieval:** `Qwen3-Embedding-8B` [59] is used to extract `query embeddings` (for retrieval) and `document embeddings` (for storage) from video captions.
    *   `Document embeddings` are generated by direct input of captions.
    *   `Query embeddings` are generated using a specific prompt to prepare the input for the embedding model:
        ```
        Instruct: This is the caption of a video.
        Please provide a search query to retrieve the caption representation of the other most relevant videos. 
        Query: {caption}.
        ```
*   **Scale:** The final training dataset comprises approximately 87K samples. The authors note that despite being relatively small compared to datasets for other VLMs (e.g., 87K vs. 9.36M for LLaVA-OneVision [40]), their method achieves effective performance improvements, highlighting its efficiency.

### Evaluation Datasets
The approach is evaluated on various video question-answering benchmarks, covering short and long video understanding tasks:
*   `ActivityNet-QA` [65]: A dataset for understanding complex web videos via question answering, often featuring open-ended questions.
*   `NExT-QA` [66]: Focuses on explaining temporal actions in videos, typically involving multiple-choice questions.
*   `EgoSchema` [67]: A diagnostic benchmark designed for very long-form video language understanding, usually with multiple-choice questions.
*   `Video-MME` [68]: A comprehensive evaluation benchmark for multi-modal LLMs in video analysis, featuring multiple-choice questions.

### Ablation Study Datasets
For efficiency during ablation studies, approximately 10% of the original training dataset is used for training. Subsets of `NExT-QA` and `EgoSchema` (each containing around 0.5K samples) are used for evaluation.

## 5.2. Evaluation Metrics
For video question-answering tasks, `Accuracy` is the primary evaluation metric.

### Accuracy
1.  **Conceptual Definition:** Accuracy measures the proportion of correctly answered questions out of the total number of questions. It provides a straightforward indication of the model's overall correctness in answering queries. For open-ended questions, determining "correctness" often requires a sophisticated evaluation mechanism, such as another LLM acting as an assessor.
2.  **Mathematical Formula:**
    \$
    \text{Accuracy} = \frac{\text{Number of Correct Answers}}{\text{Total Number of Questions}} \times 100\%
    \$
3.  **Symbol Explanation:**
    *   `Number of Correct Answers`: The count of questions for which the model's generated answer matches the ground truth or is deemed correct by an evaluator.
    *   `Total Number of Questions`: The total count of questions presented to the model for evaluation.

        For `ActivityNet-QA`, which involves open-ended answers, the original evaluation pipeline used `gpt-3.5-turbo-0613` (now deprecated). For fair comparison and consistent evaluation, the authors re-evaluate results using `Qwen3-235B-A22B` [57], an open-source LLM, which they state has superior language capabilities compared to the `gpt-3.5-turbo` series. For `multiple-choice questions` (NExT-QA, EgoSchema, Video-MME), accuracy is typically a direct comparison of the model's chosen option against the ground truth.

## 5.3. Baselines
The proposed method is compared against several advanced video language models that perform video question answering conditioned on a *single video*. These include:
*   `Video-LLaVA` [6]
*   `LLaMA-VID` [43]
*   `PLLaVA` [70]
*   `VideoChat2` [71]
*   `LLaVA-NeXT-Video` [72]
*   `Qwen2-VL` [41]
*   `Qwen2.5-VL` [42]
*   `VideoLLaMA2` [44]
*   `VideoLLaMA2.1` [44]
*   `VideoLLaMA3` [73]
*   `InternVL2` [74]
*   `InternVL2.5` [74]
*   `NVILA` [45]

    The specific baselines used for direct comparison with $+Ours$ are `LLaVA-OneVision-0.5B` [40] and `LLaVA-Video-7B` [54], which represent different parameter scales (0.5B and 7B, respectively). These baselines are representative because they are state-of-the-art single-video VLMs, allowing the paper to demonstrate the benefits of multi-video collaborative reasoning over conventional single-video approaches.

## 5.4. Implementation Details
The framework is designed to be adaptive to general video language models. Experiments are conducted using `LLaVA-OneVision-0.5B` [40] and `LLaVA-Video-7B` [54] on A6000 48GB GPUs.

*   **Graph Fusion Module (GFM) Configuration:** The hidden state size of the GFM is set to match the output dimension of the corresponding vision encoder used in the VLM.
*   **VLM Initialization:** The video language model is initialized with pre-trained weights.
*   **Training Strategy:** A standard two-stage training strategy [6, 64] is adopted to optimize the model:
    *   **Stage 1:** The `vision encoder`, `projector` (the module connecting visual features to the LLM), and `language model` are **frozen**. Only the `GFM` is trained to align its inputs to the language model.
    *   **Stage 2:** The `projector` and `language model` are **unfrozen**. `LoRA (Low-Rank Adaptation)` [69] is applied to the language model to enable efficient fine-tuning. Subsequently, the `projector`, `GFM`, and `language model` are fine-tuned simultaneously.

        The detailed training recipe and hyperparameter configurations are provided in Table 1.

The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th></th>
<th>Stage-1</th>
<th>Stage-2</th>
</tr>
</thead>
<tbody>
<tr>
<td>Trainable</td>
<td>GFM</td>
<td>GFM, Projector, LLM</td>
</tr>
<tr>
<td>Batch size</td>
<td>128</td>
<td>64</td>
</tr>
<tr>
<td>Optimizer</td>
<td>AdamW</td>
<td>AdamW</td>
</tr>
<tr>
<td>Warmup ratio</td>
<td>0.03</td>
<td>0.03</td>
</tr>
<tr>
<td>Learning rate schedule</td>
<td>Cosine decay</td>
<td>Cosine decay</td>
</tr>
<tr>
<td>LR: φgFM</td>
<td>1e-3</td>
<td>1e-4</td>
</tr>
<tr>
<td>LR: φProj.</td>
<td>-</td>
<td>1e-5</td>
</tr>
<tr>
<td>LR: φLLM</td>
<td>-</td>
<td>1e-5</td>
</tr>
</tbody>
</table>

**Explanation of Table 1 entries:**
*   **Trainable:** Specifies which components of the model are updated during each stage. In Stage 1, only the `Graph Fusion Module (GFM)` is trained. In Stage 2, the `GFM`, `Projector` (the module bridging visual features and the LLM), and the `Large Language Model (LLM)` itself are trained.
*   **Batch size:** The number of samples processed in one forward/backward pass during training. It's 128 in Stage 1 and 64 in Stage 2.
*   **Optimizer:** `AdamW` is the optimization algorithm used. AdamW is an optimizer that extends Adam by decoupling weight decay from the gradient update, often leading to better generalization.
*   **Warmup ratio:** During the initial phase of training, the learning rate is gradually increased from a small value to the base learning rate. A warmup ratio of 0.03 means the learning rate warms up over the first 3% of training steps.
*   **Learning rate schedule:** `Cosine decay` is used to decrease the learning rate over time, which helps in fine-tuning and preventing overfitting.
*   **LR: $\phi_{gFM}$ (Learning Rate for GFM):** The learning rate for the GFM. It's $1 \times 10^{-3}$ in Stage 1 and $1 \times 10^{-4}$ in Stage 2.
*   **LR: $\phi_{Proj.}$ (Learning Rate for Projector):** The learning rate for the Projector module, applied only in Stage 2, set to $1 \times 10^{-5}$.
*   **LR: $\phi_{LLM}$ (Learning Rate for LLM):** The learning rate for the Large Language Model, applied only in Stage 2, set to $1 \times 10^{-5}$.

# 6. Results & Analysis
## 6.1. Core Results Analysis
The paper evaluates the proposed framework on several video question-answering benchmarks, including `ActivityNet-QA` (open-ended, short videos), `NExT-QA` (multi-choice, short videos), `EgoSchema` (multi-choice, long videos), and `Video-MME` (multi-choice, long videos). The results, presented in Table 2, demonstrate the superiority of the proposed method ($+Ours$) over various state-of-the-art single-video baselines, particularly `LLaVA-OneVision-0.5B` and `LLaVA-Video-7B`.

The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<td rowspan="2" colspan="2">Model<br>Task<br>Duration</td>
<td rowspan="2">Params</td>
<td rowspan="2">Frames</td>
<td>ActivityNet-QA</td>
<td>NExT-QA</td>
<td>EgoSchema</td>
<td>Video-MME</td>
<td rowspan="2">Average<br>Acc. (%)</td>
</tr>
<tr>
<td>Open-Ended<br>Short</td>
<td>Multi-Choice<br>Short</td>
<td>Multi-Choice<br>Long</td>
<td>Multi-Choice<br>Long</td>
</tr>
</thead>
<tbody>
<tr>
<td colspan="2">Video-LLaVA [6]</td>
<td>7B</td>
<td>8</td>
<td>45.30</td>
<td>62.60</td>
<td>38.40</td>
<td>40.40</td>
<td>46.68</td>
</tr>
<tr>
<td colspan="2">LLaMA-VID [43]</td>
<td>7B</td>
<td>1fps</td>
<td>47.40</td>
<td>-</td>
<td>38.50</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td colspan="2">PLLaVA [70]</td>
<td>7B</td>
<td>16</td>
<td>56.30</td>
<td>68.17</td>
<td>45.16</td>
<td>44.25</td>
<td>53.47</td>
</tr>
<tr>
<td colspan="2">VideoChat2 [71]</td>
<td>7B</td>
<td>16</td>
<td>-</td>
<td>-</td>
<td>54.40</td>
<td>47.90</td>
<td>-</td>
</tr>
<tr>
<td colspan="2">LLaVA-NeXT-Video [72]</td>
<td>7B</td>
<td>32</td>
<td>53.50</td>
<td>-</td>
<td>43.90</td>
<td>46.50</td>
<td>-</td>
</tr>
<tr>
<td colspan="2">Qwen2-VL [41]</td>
<td>7B</td>
<td>2fps</td>
<td>57.40</td>
<td>77.20</td>
<td>66.70</td>
<td>63.30</td>
<td>66.15</td>
</tr>
<tr>
<td colspan="2">Qwen2.5-VL [42]</td>
<td>3B</td>
<td>2fps</td>
<td>-</td>
<td>-</td>
<td>64.80</td>
<td>61.50</td>
<td>-</td>
</tr>
<tr>
<td colspan="2">Qwen2.5-VL [42]</td>
<td>7B</td>
<td>2fps</td>
<td>-</td>
<td>-</td>
<td>65.00</td>
<td>65.10</td>
<td>-</td>
</tr>
<tr>
<td colspan="2">VideoLLaMA2 [44]</td>
<td>7B</td>
<td>16</td>
<td>50.20</td>
<td>75.60</td>
<td>-</td>
<td>47.90</td>
<td>-</td>
</tr>
<tr>
<td colspan="2">VideoLLaMA2.1 [44]</td>
<td>7B</td>
<td>16</td>
<td>53.00</td>
<td>75.60</td>
<td>53.10</td>
<td>54.90</td>
<td>59.15</td>
</tr>
<tr>
<td colspan="2">VideoLLaMA3 [73]</td>
<td>2B</td>
<td>180</td>
<td>58.20</td>
<td>81.10</td>
<td>58.50</td>
<td>59.60</td>
<td>64.35</td>
</tr>
<tr>
<td colspan="2">InternVL2 [74]</td>
<td>8B</td>
<td>16</td>
<td>-</td>
<td>-</td>
<td>55.00</td>
<td>54.00</td>
<td>-</td>
</tr>
<tr>
<td colspan="2">InternVL2.5 [74]</td>
<td>8B</td>
<td>64</td>
<td>58.90</td>
<td>85.00</td>
<td>51.50</td>
<td>64.20</td>
<td>64.90</td>
</tr>
<tr>
<td colspan="2">NVILA [45]</td>
<td>8B</td>
<td>256</td>
<td>60.90</td>
<td>82.20</td>
<td>54.30</td>
<td>64.20</td>
<td>65.40</td>
</tr>
<tr>
<td colspan="2">LLaVA-OneVision<br>[40]</td>
<td>0.5B</td>
<td>32</td>
<td>~45.65</td>
<td>57.20</td>
<td>26.80</td>
<td>44.00</td>
<td>43.41</td>
</tr>
<tr>
<td colspan="2">LLaVA-OneVision<br>[40]+Ours</td>
<td>0.5B</td>
<td>32</td>
<td>~46.46</td>
<td>58.71</td>
<td>28.38</td>
<td>43.74</td>
<td>44.32</td>
</tr>
<tr>
<td colspan="2">LLaVA-Video [54]</td>
<td>7B</td>
<td>64</td>
<td>~60.55</td>
<td>83.20</td>
<td>557.30</td>
<td>63.30</td>
<td>66.09</td>
</tr>
<tr>
<td colspan="2">LLaVA-Video [54]<br>+Ours</td>
<td>7B</td>
<td>64</td>
<td>~61.25</td>
<td>84.00</td>
<td>61.76</td>
<td>64.37</td>
<td>67.84</td>
</tr>
</tbody>
</table>

**Analysis:**
*   **Overall Improvement:** The $+Ours$ framework consistently improves the performance of both baseline models (`LLaVA-OneVision-0.5B` and `LLaVA-Video-7B`) across all evaluated benchmarks.
    *   For `LLaVA-OneVision-0.5B`, the average accuracy increases from 43.41% to 44.32% (an improvement of 0.91 percentage points).
    *   For `LLaVA-Video-7B`, the average accuracy significantly improves from 66.09% to 67.84% (an improvement of 1.75 percentage points).
*   **Specific Task Gains:**
    *   `ActivityNet-QA` (Open-Ended, Short): Improves from ~45.65% to ~46.46% for LLaVA-OneVision and from ~60.55% to ~61.25% for LLaVA-Video.
    *   `NExT-QA` (Multi-Choice, Short): Improves from 57.20% to 58.71% for LLaVA-OneVision and from 83.20% to 84.00% for LLaVA-Video.
    *   `EgoSchema` (Multi-Choice, Long): Shows the most substantial gains for LLaVA-Video, increasing from 57.30% to 61.76% (4.46 percentage points). This highlights the framework's effectiveness for long-form video understanding, where spatio-temporal incompleteness is more prevalent.
    *   `Video-MME` (Multi-Choice, Long): Shows a slight decrease for LLaVA-OneVision (44.00% to 43.74%), but an improvement for LLaVA-Video (63.30% to 64.37%). The decrease for the smaller model might indicate that for some complex, long-form tasks, the structured information might introduce some noise or misdirection if the base VLM's understanding is very limited.
*   **Efficiency:** The paper notes that these improvements are achieved despite training on a relatively compact dataset (87K samples), which indicates the efficiency of the proposed method in integrating multi-video knowledge without requiring massive additional data for training.
*   **Reliability:** The gains in accuracy, especially for complex reasoning tasks (like `EgoSchema`), suggest that the multi-video collaborative reasoning helps the model provide more reliable answers by compensating for missing information and reducing hallucinations.

## 6.2. Ablation Studies / Parameter Analysis
The paper conducts thorough ablation studies to validate the effectiveness of its proposed components and design choices.

### 6.2.1. Ablation Study on Video Structuring and Multi-Video Fusion Components
This study investigates the necessity of the `Video Structuring Module (VSM)` and the `Graph Fusion Module (GFM)` by comparing them against common multi-video fusion strategies. The evaluation is performed on the `NExT-QA` dataset.

The following are the results from Table 3 of the original paper:

<table>
<thead>
<tr>
<td>Struct</td>
<td>Multi-video</td>
<td>context L</td>
<td>NExT-QA</td>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4"></td>
<td>single video</td>
<td>6.5K</td>
<td>61.4</td>
</tr>
<tr>
<td>multi-video tokens (32)</td>
<td>38K</td>
<td>OOM</td>
</tr>
<tr>
<td>multi-video tokens (8)</td>
<td>15K</td>
<td>51.8</td>
</tr>
<tr>
<td>multi-video captions</td>
<td>9.3K</td>
<td>61.8</td>
</tr>
<tr>
<td>✓</td>
<td>single video</td>
<td>7.3K</td>
<td>62.0</td>
</tr>
<tr>
<td>✓</td>
<td>graph fusion module</td>
<td>7.5K</td>
<td>65.2</td>
</tr>
</tbody>
</table>

The following are the results from Table 4 of the original paper:

<table>
<thead>
<tr>
<td>Struct</td>
<td>Multi-video</td>
<td>context L</td>
<td>NExT-QA</td>
</tr>
</thead>
<tbody>
<tr>
<td></td>
<td>single video</td>
<td>13K</td>
<td>79.8</td>
</tr>
<tr>
<td></td>
<td>multi-video tokens (64)</td>
<td>73K</td>
<td>OOM</td>
</tr>
<tr>
<td></td>
<td>multi-video tokens (8)</td>
<td>22K</td>
<td>72.6</td>
</tr>
<tr>
<td></td>
<td>multi-video captions</td>
<td>16K</td>
<td>79.8</td>
</tr>
<tr>
<td>✓</td>
<td>single video</td>
<td>13.8K</td>
<td>83.6</td>
</tr>
<tr>
<td>✓</td>
<td>graph fusion module</td>
<td>14K</td>
<td>84.2</td>
</tr>
</tbody>
</table>

**Analysis (LLaVA-OneVision-0.5B, Table 3):**
*   **`single video` (baseline):** Achieves 61.4% accuracy with a context length of 6.5K tokens.
*   **`multi-video tokens (32)`:** Directly concatenating tokens from 32 related videos leads to 38K context length and `Out Of Memory (OOM)` errors, demonstrating its impracticality.
*   **`multi-video tokens (8)`:** Even with fewer related videos (8 frames per video), the context length is 15K, and accuracy drops significantly to 51.8% (a -9.6% degradation from baseline), confirming the "lost in the middle" problem.
*   **`multi-video captions`:** Sending only captions of all videos results in 9.3K context length and a slight improvement to 61.8% (+0.4% from baseline). This suggests textual summaries are better than raw tokens but still limited.
*   **`✓ single video` (VSM enabled):** Using the `Video Structuring Module` (VSM) for the single target video (context length 7.3K) improves accuracy to 62.0% (+0.6%), showing the benefit of structured representation even without multi-video collaboration.
*   **`✓ graph fusion module` (VSM + GFM):** The full framework, with VSM and `Graph Fusion Module` (GFM), achieves 65.2% accuracy (a substantial +3.8% from baseline) with only 7.5K context length, demonstrating efficient and effective multi-video collaboration.

**Analysis (LLaVA-Video-7B, Table 4):**
*   **`single video` (baseline):** Achieves 79.8% accuracy with 13K context length.
*   **`multi-video tokens (64)`:** Causes `OOM` with 73K context length, again highlighting impracticality.
*   **`multi-video tokens (8)`:** Leads to 22K context length and a drop to 72.6% (-7.2% degradation), reinforcing the challenges of direct concatenation.
*   **`multi-video captions`:** 16K context length, but no performance change (79.8%), indicating limited benefit from simple captions for this larger model.
*   **`✓ single video` (VSM enabled):** VSM alone for the target video (13.8K context) boosts accuracy to 83.6% (+3.8% from baseline), showing strong benefits of structuring.
*   **`✓ graph fusion module` (VSM + GFM):** The full framework achieves 84.2% accuracy (a +4.4% from baseline) with 14K context length, showcasing significant gains with minimal overhead (0.2K additional tokens over the VSM-only approach).

    These results conclusively demonstrate that direct multi-video token concatenation is ineffective and often prohibitive, while `multi-video captions` provide only marginal gains. The proposed `Video Structuring Module` significantly improves single-video understanding, and its combination with the `Graph Fusion Module` for multi-video collaboration yields substantial accuracy improvements with remarkably low token overhead.

### 6.2.2. Ablation on Graph Fusion Module Design
This study dissects the components of the `Graph Fusion Module (GFM)`: `HF-GAT`, `Pooling Attention (PA)`, and `Cross-Graph Attention (CGA)`. It also re-evaluates the impact of including an `FFN` (Feed-Forward Network) within the GFM layers. This is conducted using `LLaVA-OneVision-0.5B` on subsets of training and evaluation datasets.

The following are the results from Table 5 of the original paper:

<table>
<thead>
<tr>
<td>HF-GAT</td>
<td>PA</td>
<td>CGA</td>
<td>FFN</td>
<td>NExT-QA</td>
<td>EgoSchema</td>
</tr>
</thead>
<tbody>
<tr>
<td></td>
<td></td>
<td></td>
<td></td>
<td>61.4</td>
<td>26.4</td>
</tr>
<tr>
<td>✓</td>
<td></td>
<td></td>
<td></td>
<td>64.2</td>
<td>28.0</td>
</tr>
<tr>
<td>✓</td>
<td>✓</td>
<td></td>
<td></td>
<td>64.4</td>
<td>28.2</td>
</tr>
<tr>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td></td>
<td>65.0</td>
<td>28.6</td>
</tr>
<tr>
<td>✓</td>
<td>✓</td>
<td>V</td>
<td>V</td>
<td>64.4</td>
<td>27.6</td>
</tr>
</tbody>
</table>

**Analysis:**
*   **Baseline (No GFM components):** Starting with graph structure feature tokens directly sent to the projection layer (no HF-GAT, PA, CGA, FFN), performance is 61.4% on NExT-QA and 26.4% on EgoSchema.
*   **`HF-GAT`:** Incorporating `HF-GAT` to propagate structural information within a single video significantly improves performance to 64.2% on NExT-QA (+2.8%) and 28.0% on EgoSchema (+1.6%). This confirms the importance of structured representation and within-video relationship modeling.
*   **`Pooling Attention (PA)`:** Adding `Pooling Attention` within the `Triplet Embedding Module (TEM)` (HF-GAT and PA enabled) further boosts performance slightly to 64.4% on NExT-QA (+0.2%) and 28.2% on EgoSchema (+0.2%). This shows the benefit of adaptively fusing visual features with textual graph features.
*   **`Cross-Graph Attention (CGA)`:** Enabling `Cross-Graph Attention` (HF-GAT, PA, and CGA enabled) for multi-video knowledge fusion yields the best performance, reaching 65.0% on NExT-QA (+0.6%) and 28.6% on EgoSchema (+0.4%). This confirms CGA's role in effectively leveraging information from related videos.
*   **`FFN`:** Interestingly, when an `FFN` is included (HF-GAT, PA, CGA, and FFN enabled), the performance drops to 64.4% on NExT-QA (-0.6%) and 27.6% on EgoSchema (-1.0%). This result supports the design choice in Section 3.3 to exclude FFNs in the GFM layers, validating that FFNs might introduce excessive feature shifting, disrupting the desired invariance of aligned visual features.

### 6.2.3. Discussion on the Retrieved Video Contents

#### How do multiple videos affect the performance?
Figure 6 analyzes the impact of the number of related videos on accuracy and context length for NExT-QA using LLaVA-OneVision-0.5B and LLaVA-Video-7B.

![Fig. 6: Comparative analysis of accuracy $( \\% )$ and context length (K) for NExT-QA across different models under varying numbers of related videos.](images/6.jpg)
*该图像是图表，展示了在不同相关视频数量下，模型LLaVA-OneVision-0.5B和LLaVA-Video-7B的准确率（%）与上下文长度（K）的比较。图中包含基线、我们的模型表现和上下文长度的变化趋势。*

Fig. 6: Comparative analysis of accuracy $(\%)$ and context length (K) for NExT-QA across different models under varying numbers of related videos.

**Analysis:**
*   For both models, increasing the number of retrieved videos from 1 to approximately 5 (for LLaVA-Video-7B, the peak is around 5) generally leads to an increase in accuracy. This indicates that additional relevant information from multiple videos helps the model achieve more comprehensive reasoning.
*   Beyond a certain point (e.g., 5 videos for LLaVA-Video-7B), increasing the number of related videos can cause accuracy to slightly decline. This suggests that incorporating too many videos might introduce noise or irrelevant information that, even with the structured approach, can slightly hinder optimal performance.
*   Crucially, this trend is accompanied by only a `marginal increase in the total number of tokens` (context length K). This validates that the structured representation is highly data-efficient, preventing the token explosion seen with direct concatenation.

#### How does video relevance affect the performance?
Figure 7 investigates how the relevance of retrieved videos impacts performance on NExT-QA.

![Fig. 7: Comparative analysis of accuracy $( \\% )$ for NExT-QA across different models under varying relevance of related videos.](images/7.jpg)
*该图像是图表，展示了在不同相关视频的关联性下，LLava模型的准确性（%）比较分析。左侧为LLava-OneVision-0.5B，右侧为LLava-Video-7B，数据趋势表明我们的模型在多种相关性条件下的表现优于基线模型。*

Fig. 7: Comparative analysis of accuracy $(\%)$ for NExT-QA across different models under varying relevance of related videos.

**Analysis:**
*   The reasoning performance generally `decreases as the relevance of the video diminishes`. This is an intuitive result: highly relevant videos provide more useful complementary information, while less relevant ones contribute less or even introduce noise.
*   However, even with lower relevance, the performance remains `comparable to the baseline` (the single-video reasoning performance). This indicates that the framework is robust and capable of filtering out irrelevant information to some extent, preventing a catastrophic drop in performance when less ideal related videos are retrieved.

#### How does the video retrieval strategy affect the performance?
The paper compares three video retrieval strategies:
*   **Video vector-based retrieval:** Uses `SigLIP` [58] vision encoder to generate feature vectors from sampled frames (averaged class tokens), then retrieves videos based on cosine similarity.
*   **Caption vector-based retrieval:** Uses a text encoder (`Qwen3-Embedding` [59]) to extract feature vectors from video captions, then retrieves based on cosine similarity.
*   **Restricted retrieval:** A variant of caption vector-based retrieval where retrieval is restricted to videos within the test set (artificial partitioning).

    The following are the results from Table 6 of the original paper:

    <table>
    <thead>
    <tr>
    <td>Video Retrieval Strategy</td>
    <td>NExT-QA</td>
    <td>EgoSchema</td>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>video vector-based retrieval</td>
    <td>63.8</td>
    <td>27.6</td>
    </tr>
    <tr>
    <td>restricted retrieval</td>
    <td>63.6</td>
    <td>27.6</td>
    </tr>
    <tr>
    <td>caption vector-based retrieval</td>
    <td>65.0</td>
    <td>28.6</td>
    </tr>
    </tbody>
    </table>

**Analysis:**
*   `Caption vector-based retrieval` achieves the best performance (65.0% on NExT-QA, 28.6% on EgoSchema). This is attributed to the high-quality prompt construction for captioning (Figure 4) and the strong retrieval capabilities of `Qwen3-Embedding`.
*   `Video vector-based retrieval` and `restricted retrieval` also demonstrate competitive performance (e.g., 63.8% and 63.6% on NExT-QA, respectively), indicating that the reasoning process is only slightly affected by the choice of retrieval strategy.
*   **Conclusion:** The framework exhibits robust performance across different retrieval strategies, with caption-based retrieval being the most effective in this setup. The overall conclusion from these discussions is that using more *relevant* videos is key, but the method has a degree of robustness against less relevant or varied retrieval quality.

## 6.3. Visualization
The paper provides visualizations to intuitively explain the multi-video collaboration framework.

### 6.3.1. Reasoning Process Visualization
Figure 8 visualizes the reasoning process for a query: "What activities are the skateboarders performing in the video?".

![该图像是示意图，展示了在不同场景中处理视频数据时的特征图层叠加情况。通过对视频帧应用色块标识，从而突出显示出关键特征，有助于提高视频语言模型的推理效果。](images/8.jpg)
*该图像是示意图，展示了在不同场景中处理视频数据时的特征图层叠加情况。通过对视频帧应用色块标识，从而突出显示出关键特征，有助于提高视频语言模型的推理效果。*

Fig. 8: Visualization of our structured multi-video collaborative reasoning. We present a representative video question-answering example from our structured multi-video collaboration pipeline, showcasing the Pooling Attention visualization, structuring results, and the Cross-Graph Attention map, along with the answers generated before and after applying our framework. Within each scene, color patches correspond to triplets with matching colors, highlighting regions of interest identified through Pooling Attention, while the dashed lines indicate the relationships between triplets within the scenes.

**Analysis:**
*   **Baseline:** The baseline model provides a generic, less detailed response: "The skateboarders are riding the skateboard on a road." It lacks higher-level, domain-specific knowledge.
*   **Our Framework:**
    *   **Pooling Attention Visualization:** Color patches on the video frames correspond to triplets, highlighting specific regions of interest. These patches show that `Pooling Attention` focuses on significant visual features relevant to identified subjects and objects (e.g., the skateboarders, the skateboard, the road environment).
    *   **Structuring Results:** The video is represented as graph-structured data. Dashed lines illustrate relationships between these triplets within scenes and across frames/videos. This confirms that the VSM effectively extracts and organizes spatio-temporal information.
    *   **Cross-Graph Attention Map:** The visualization shows how sub-graphs from related videos contribute useful relational structures to the target video's graph. This demonstrates the `Cross-Graph Attention (CGA)` mechanism in action, fusing multi-video knowledge.
    *   **Enhanced Answer:** By fusing this structured multi-video knowledge, our model provides a much more accurate and detailed response (e.g., identifying specific skateboarding activities, perhaps from related videos showing similar tricks or environments). This showcases how the framework builds a coherent understanding of complex scenarios.

### 6.3.2. More Video Question-Answering Results
Figure 9 presents additional qualitative results demonstrating the framework's superior performance over baseline methods.

![Fig. 9: Visualization of video question answering examples.](images/10.jpg)
*该图像是视频问答示例的可视化展示。图中展示了两个相关的任务和基线答案与改进答案的对比，结合多张视频帧分析不同场景的理解与解读，为视频语言模型的研究提供了参考。*

Fig. 9: Visualization of video question answering examples.

**Analysis (from the text description):**
*   **Domain Knowledge Integration:** Our model can correctly interpret unique activities by integrating domain knowledge. For example, it identifies "carving a watermelon to make a jack-o-lantern" instead of merely "cutting a watermelon with a knife," indicating a deeper understanding of intent and context.
*   **Detailed and Context-Aware Answers:** For "uneven bars," the model provides a more detailed description, recognizing them as "two parallel bars set at different heights" and specifying that "the man is using these bars to perform his routine." This shows an enhanced ability to provide rich, context-aware information.
*   **Hallucination Alleviation:** In a bowling video example, the framework concludes that "bowling is a safe sport as no injuries are shown," rather than generating fabricated injury details (hallucinations). This demonstrates the framework's improved reliability by grounding its responses in actual visual facts and avoiding spurious information.

    These visualizations and examples collectively illustrate that the structured multi-video collaborative reasoning framework effectively extracts, fuses, and utilizes multi-video information, leading to more precise, accurate, and context-aware answers, and mitigating common VLM pitfalls like hallucinations.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This paper introduces a pioneering framework designed to enhance video large language models (VLMs) through structured multi-video collaborative reasoning. The core of the approach lies in addressing the spatio-temporal incompleteness and redundancy inherent in individual videos, which often lead to hallucinations and inaccuracies in VLM outputs. The framework consists of three key components:
1.  **Video Structuring Module (VSM):** This module efficiently represents video content as a spatio-temporal graph, capturing key objects, subjects, and their relationships within and across frames.
2.  **Graph Fusion Module (GFM):** This module integrates structured knowledge from multiple related videos into enhanced graph node tokens. It leverages `Hierarchical Frame Graph Attention Networks (HF-GAT)` for within-video structural processing and `Cross-Graph Attention (CGA)` for effective cross-video knowledge fusion.
3.  **Structured Multi-Video Prompt:** An elaborate prompt engineering strategy combines these fused graph tokens with original visual tokens (for the target video) and textual tokens, providing a concise and interpretable input to the underlying large language model (LLM).

    Extensive experiments on various video question-answering benchmarks (`ActivityNet-QA`, `NExT-QA`, `EgoSchema`, `Video-MME`) substantiate the effectiveness and robustness of the proposed framework. The results demonstrate significant improvements in accuracy, particularly for long-form video understanding, and highlight the framework's ability to reduce hallucinations and provide more detailed, contextually precise answers compared to single-video baselines. The approach achieves these gains with minimal token overhead, making multi-video reasoning practical.

## 7.2. Limitations & Future Work
The authors do not explicitly list limitations or future work in a dedicated section, but some can be inferred from the context and common challenges in the field:

**Inferred Limitations:**
*   **Dependence on Dense Video Captioning Quality:** The `Video Structuring Module` heavily relies on the quality and granularity of the dense video captions generated by an initial VLM. Any inaccuracies or incompleteness in these captions could propagate errors to the downstream graph construction and reasoning.
*   **Retrieval Quality Sensitivity:** While the framework shows robustness to varying relevance, its peak performance is achieved with highly relevant videos. The effectiveness of the overall system is therefore constrained by the quality and accuracy of the video retrieval mechanism. If the retrieved videos are consistently irrelevant, the benefits might diminish.
*   **Complexity of Graph Construction:** While the graph representation is data-efficient for the LLM, the multi-step `Video Structuring Module` itself involves several sequential operations (scene detection, captioning, graph parsing, filtering). The computational overhead of this preprocessing for very large-scale, real-time applications or extremely long videos might still be a consideration.
*   **Fixed $N$ Related Videos:** The current approach processes a fixed number of related videos (e.g., 5 in the optimal case). Determining the optimal number dynamically or adapting to scenarios with very few or very many related videos could be a challenge.

**Inferred Future Work:**
*   **End-to-End Joint Training:** Exploring a more tightly integrated or end-to-end training approach for the VSM and GFM, rather than relying on pre-trained components or sequential processing, could potentially yield further performance gains and improve robustness to upstream errors.
*   **Dynamic and Adaptive Retrieval:** Research into more sophisticated retrieval mechanisms that can dynamically assess the utility of potential related videos or retrieve information on-the-fly based on the ongoing reasoning process could enhance the framework.
*   **Broader Reasoning Tasks:** Expanding the framework's application to other complex video understanding tasks beyond question answering, such as video summarization, event prediction, or anomaly detection, could demonstrate its versatility.
*   **Scalability for Extreme Cases:** Further optimizing the graph construction and fusion for scenarios involving an extremely large corpus of videos or exceptionally long videos to maintain efficiency.
*   **Graph Representation Learning:** Investigating alternative or more expressive graph representations that might capture even finer-grained spatio-temporal relationships or incorporate uncertainty in relationships.

## 7.3. Personal Insights & Critique
This paper presents a highly insightful and effective approach to a critical problem in video language understanding. My personal insights and critiques are as follows:

**Personal Insights:**
*   **Elegant Solution to Token Bottleneck:** The primary innovation of transforming raw video into a structured graph is an elegant solution to the perennial `token explosion` and `lost in the middle` problems that plague VLMs when dealing with multiple or long videos. It effectively condenses information, presenting the LLM with "digested" knowledge rather than raw data.
*   **Leveraging Linguistic Structure for Visual Reasoning:** The use of `textual scene graph parsing` to drive the video structuring is a clever way to leverage the power of LLMs (Qwen3-30-A3B) for initial conceptualization before visual grounding. This bridges the gap between high-level language understanding and low-level visual perception.
*   **Hybrid Approach for Optimality:** The decision to retain raw visual tokens for the target video while only using graph tokens for related videos is a smart hybrid strategy. It ensures fine-grained details are available for the primary focus while efficiently bringing in complementary context from auxiliary sources.
*   **Practicality and Robustness:** The demonstrated performance gains on a relatively compact dataset, combined with the robustness against varying video retrieval strategies and the ability to mitigate hallucinations, make this framework highly practical and promising for real-world applications. The `Class Embedding` within GFM also shows a thoughtful design for handling multiple sources.
*   **Stimulus for Future Research:** This work provides a strong foundation for future research in structured multi-modal reasoning. The graph-based paradigm could be extended to other forms of multi-modal data (e.g., images, audio, sensor data) where contextual collaboration is beneficial.

**Critique & Areas for Improvement:**
*   **Cascading Errors from VSM:** The pipeline nature of VSM means that errors in earlier stages (e.g., scene detection, captioning, graph parsing) could propagate and potentially limit the overall system's accuracy. For instance, if the video captioning VLM makes a significant error, the structured graph will be flawed, regardless of the GFM's capabilities. More robust error handling or uncertainty modeling within the graph construction could be explored.
*   **Interpretability of Graph Filtering:** While `Graph Information Filtering` improves data quality, the `SigLIP` classifier's binary decision might sometimes discard potentially ambiguous but useful information. The criteria for constructing self-connections ($$\{s_i, *, s_i\}`or`\{o_i, *, o_i\}$$) if only one entity is present could be further nuanced.
*   **Fixed Graph Schema:** The current approach implies a relatively fixed schema for the spatio-temporal graph (subjects, predicates, objects). For highly complex or abstract reasoning tasks, a more flexible or dynamic graph construction that can adapt its representation based on the query or task might be beneficial.
*   **Computational Cost of VSM Preprocessing:** Although the LLM inference is made efficient, the VSM preprocessing (dense captioning, scene graph parsing, filtering) for *each* video in a large dataset can be computationally intensive, especially for very long videos or large numbers of related videos. Optimizations for this stage would further enhance practicality.
*   **Generalizability of Graph Fusion:** While `Cross-Graph Attention` is used, the nature of how different types of relatedness (e.g., temporal overlap, semantic similarity, object commonality) are weighted and fused could be further explored. It might not be optimal for all types of multi-video relationships.

    Overall, this paper makes a significant contribution by providing a principled and effective method for `structured multi-video collaborative reasoning`. It intelligently addresses key limitations of existing VLMs and opens up exciting avenues for more robust and reliable video understanding systems.