# Fine-Grained Captioning of Long Videos through Scene Graph Consolidation

Sanghyeok Chu 1 Seonguk Seo† 1 Bohyung Han 1 2

# Abstract

Recent advances in vision-language models have led to impressive progress in caption generation for images and short video clips. However, these models remain constrained by their limited temporal receptive fields, making it difficult to produce coherent and comprehensive captions for long videos. While several methods have been proposed to aggregate information across video segments, they often rely on supervised fine-tuning or incur significant computational overhead. To address these challenges, we introduce a novel framework for long video captioning based on graph consolidation. Our approach first generates segment-level captions, corresponding to individual frames or short video intervals, using off-the-shelf visual captioning models. These captions are then parsed into individual scene graphs, which are subsequently consolidated into a unified graph representation that preserves both holistic context and fine-grained details throughout the video. A lightweight graph-to-text decoder then produces the final video-level caption. This framework effectively extends the temporal understanding capabilities of existing models without requiring any additional fine-tuning on long video datasets. Experimental results show that our method significantly outperforms existing LLMbased consolidation approaches, achieving strong zero-shot performance while substantially reducing computational costs.

# 1. Introduction

Vision-language models (VLMs) have demonstrated impressive capabilities across diverse vision-language tasks, including visual question answering, visual dialogue, crossmodal retrieval, and spatiotemporal understanding (Alayrac et al., 2022; Dai et al., 2023; OpenAI, 2023; Chen et al., 2024b; Huang et al., 2024; Zhang et al., 2025; Xu et al., 2024; Maaz et al., 2024). Notably, substantial progress has been made in generating captions for images and short video clips (Liu et al., 2024; Chai et al., 2025; Zhao et al., 2024; Wang et al., 2024; Chen et al., 2024a; Mun et al., 2019).

However, generating captions for longer videos remains a significant challenge. Most existing models are designed for short-term visual inputs, such as images or short video clips, and lack effective support for holistic encoding of entire long videos. As a result, captioning videos beyond a model's temporal window typically requires processing and integrating information from multiple temporal segments. Several approaches, such as memory-based (Zhou et al., 2024; Song et al., 2024; Balazevic et al., 2024) and recursive frameworks (Zhou et al., 2024; Islam et al., 2024; Qian et al., 2024; Weng et al., 2024; Kahatapitiya et al., 2024), have been proposed to consolidate information across these segments. However, these methods often rely on supervised fine-tuning with the target datasets, which limits their generalizability to unseen video domains. More recently, large language models (LLMs) have been employed to generate textual summaries across multiple video segments (Wang et al., 2022b; Chen et al., 2023; Zhang et al., 2024a). While these LLM-based approaches eliminate the need to adapt existing models for long videos, they typically incur high inference overhead and require significant computational resources.

To address these limitations, we propose a novel framework that integrates segment-level captions into a unified global description via graph-based consolidation. We first obtain segment-level captions—each corresponding to either a single frame or a short video clip, depending on the chosen visual captioning model—using an off-the-shelf captioning algorithm. Each caption is then parsed into a scene graph, and these graphs are consolidated into a unified structure that captures the comprehensive semantics of the entire video. Finally, a lightweight graph-to-text decoder, trained solely on external text corpora, translates the consolidated graph into a coherent global caption.

The proposed approach enhances understanding and processing of long-range temporal information without requiring architectural changes or fine-tuning on long video datasets.

In particular, our framework can be paired with any off-theshelf VLM, effectively extending its captioning capability beyond the model's inherent temporal constraints. Unlike other LLM-based consolidation methods, it minimizes computational overhead by employing a lightweight graph-totext decoder with significantly fewer parameters. Our experimental results demonstrate that our approach achieves superior performance in both zero-shot video captioning and zero-shot video paragraph captioning, demonstrating its effectiveness and efficiency.

In summary, our key contributions are organized as follows:

•We propose a novel approach to generate fine-grained captions for long videos using the information across multiple temporal segments. • We introduce a graph consolidation algorithm that merges segment-level scene graphs into a unified representation to capture both holistic context and finegrained details across the entire video. Our method achieves strong zero-shot captioning performance with significantly lower computational cost compared to LLM-based approaches.

# 2. Related Works

Video captioning Recent advances in video captioning have predominantly rely on supervised training using largescale datasets, achieving impressive results across various benchmarks (Lei et al., 2021; Wang et al., 2022a; Yan et al., 2022; Liu et al., 2024; Zhao et al., 2024; Wang et al., 2024; Chen et al., 2024a). However, extending these supervised approaches to longer videos remains challenging, primarily due to the scarcity of annotated data covering extensive temporal contexts and the computational complexity involved in modeling long-range dependencies. While various methods have been proposed to tackle these challenges, the needs for supervised fine-tuning for specific target datasets hampers scalability and generalization to unseen video domains (Yang et al., 2023; Islam et al., 2024; Song et al., 2024; Balazevic et al., 2024; Qian et al., 2024; Weng et al., 2024; Kahatapitiya et al., 2024).

Zero-shot video captioning Researchers have explored methods for video captioning without using paired videotext annotations. One approach involves refining language model outputs solely at test time. ZeroCap (Tewel et al., 2022) and related methods (Tewel et al., 2023) use imagetext alignment score calculated by CLIP (Radford et al., 2021) in gradient updates to adjust language model features, while MAGIC (Su et al., 2022) employs a CLIPinduced decoding strategy to ensure semantic relevance. Although initially developed for images, these methods extend to videos by aggregating frame-level features into a single representation. Another approach, often termed zeroshot, involves text-only training without paired video-text annotations, where text decoders are used in conjunction with image-text aligned encoders such as CLIP and ImageBind (Girdhar et al., 2023). Methods such as DeCap (Li et al., 2023b) and $C ^ { 3 }$ (Zhang et al., 2024b) generate captions by aligning visual and textual features in a shared embedding space. However, these approaches often fail to produce accurate and coherent captions, especially when applied to videos with complex events.

Zero-shot long video captioning Generating coherent and comprehensive captions for long-context videos under zero-shot settings often relies on the consolidation of information derived from multiple temporal segments. Existing consolidation techniques, including memory-based (Zhou et al., 2024; Song et al., 2024; Balazevic et al., 2024) and recursive approaches (Islam et al., 2024; Qian et al., 2024; Weng et al., 2024; Kahatapitiya et al., 2024), require supervised fine-tuning on the target dataset, which limits their applicability to zero-shot scenarios. Recently, LLMs have emerged as a promising tool for zero-shot consolidation, leveraging their general reasoning capabilities without task-specific fine-tuning. For example, VidIL (Wang et al., 2022b) constructs prompts by integrating multi-level textual information from image-language models, including objects, events, attributes, frame captions, and subtitles. Due to the complexity of these prompts, it incorporates illustrative fewshot exemplars from training dataset, to guide LLMs in interpreting and utilizing these textual cues for video captioning Similarly, Video ChatCaptioner (Chen et al., 2023) adopts an interactive framework, where an LLM queries an image VLM for captions of individual frames and aggregates them to generate video caption. While these LLM-based methods are powerful and flexible, they typically incur high computational costs.

# 3. Scene Graph Construction for Videos

To enable effective captioning of long videos, we propose a novel framework that constructs and consolidates scene graphs derived from segment-level captions, as illustrated in Figure 1. The framework comprises four main stages: (1) generating captions for individual video segments using VLMs, (2) converting these captions into scene graphs, (3) merging the scene graphs from all segments into a unified graph, and (4) generating a comprehensive description from the consolidated graph. By aggregating information across segments, the proposed method produces captions that are more coherent and contextually informative, capturing finegrained details throughout the video. Throughout this paper, we use the term segment to denote a temporal unit of a video—either a single frame or a short interval—depending on the characteristics of the employed VLM.

![](images/1.jpg)  
scene graph is transformed into an input for the graph-to-text model to generate a caption.

# 3.1. Generating segment-level captions

Given an input video, we first divide it into a series of temporal segments. We then generate segment-level captions using off-the-shelf VLMs, with prompts guiding the models to produce descriptive sentences suitable for scene graph construction. While we primarily utilize open-source VLMs as our captioning backbone, our framework is flexible enough to incorporate any VLM, including proprietary or closed-source models, as long as APIs are accessible.

# 3.2. Parsing captions into scene graphs

A scene graph $G = ( { \mathcal { O } } , { \mathcal { E } } )$ is defined by a set of objects $\mathcal { O } = \{ o _ { 1 } , o _ { 2 } , . . . \}$ , and a set of edges between objects, $\mathcal { E }$ Each object $o _ { i } = \left( c _ { i } , \mathcal { A } _ { i } \right)$ consists of an object class $c _ { i } \in \mathcal { C }$ and its attribute set $\mathcal { A } _ { i } \subseteq \mathcal { A }$ , where $\mathcal { C }$ is a set of object classes and $\mathcal { A }$ is a set of all possible attributes. A directed edge, $e _ { i , j } \equiv ( o _ { i } , o _ { j } ) \in \mathcal { E }$ , has a label $r _ { i , j } \in \mathcal { R }$ , specifying the relationship from one object to the other. All object classes, attributes, and relationship labels are represented as text strings.

We convert the generated caption from each segment into a scene graph, providing a more structured understanding of each segment. A caption is parsed into a scene graph by textual scene graph parser, and FACTUAL-MR parser (Li et al., 2023c) is used in our implementation. This parser first maps the caption to an intermediate semantic representation consisting of objects, attributes, and relationships, then deterministically converts it into a scene graph. By representing each segment as a graph consisting of objects and their relationships, we can apply a graph merging technique to produce a holistic representation of the entire input video.

# 3.3. Scene graph consolidation

The scene graph consolidation step combines all individual

# Algorithm 1 Scene graph consolidation

1: Input:   
2: ${ \mathcal { G } } = \{ G _ { 1 } , G _ { 2 } , \ldots , G _ { n } \}$ set of scene graphs   
3: $\phi ( \cdot )$ : a graph encoder   
4: EY $\psi _ { i } ( \cdot )$ : a function returning the $i ^ { \mathrm { { t h } } }$ object in a graph   
5: $\pi$ : a permutation function   
6: $\tau$ : a threshold   
7: Output: $G _ { \mathrm { v i d e o } }$ : a video-level scene graph   
8: while $| \mathcal { G } | > 1$ do   
9: Retrieve the most similar pair $\{ G ^ { s } , G ^ { t } \}$ from $\mathcal { G }$   
10: $G ^ { s } = ( \mathcal { O } ^ { s } , \mathcal { E } ^ { s } )$ $G ^ { t } = ( \mathcal { O } ^ { t } , \mathcal { E } ^ { t } )$   
11: $G ^ { m } = ( \mathcal { O } ^ { m } , \mathcal { E } ^ { m } ) \gets ( \mathcal { O } ^ { s } \cup \mathcal { O } ^ { t } , \mathcal { E } ^ { s } \cup \mathcal { E } ^ { t } )$ EPY   
12: EY $\pi ^ { * }  \arg \operatorname* { m a x } _ { \pi \in \Pi } \sum _ { i } { \frac { \diamond \diamond } { \| \psi _ { i } ( \phi ( G ^ { s } ) ) \| } } \cdot { \frac { \surd \hat { \psi _ { i } } ( \phi ( G _ { \pi } ^ { t } ) ) } { \| \psi _ { i } ( \phi ( G _ { \pi } ^ { t } ) ) \| } }$   
13: for $( p , q ) \in { \mathcal { M } }$ such that $s _ { p , q } > \tau$ do   
14: Set the class label of the merged object, $\hat { c }$ EMP   
15: $\begin{array} { l } { \hat { o } _ { m }  ( \hat { c } , \mathcal { A } _ { p } ^ { s } \cup \mathcal { A } _ { q } ^ { t } ) } \\ { \mathcal { O } ^ { m }  \{ \hat { o } _ { m } \} \cup ( \mathcal { O } ^ { m } \setminus \{ o _ { p } ^ { s } , o _ { q } ^ { t } \} ) } \end{array}$   
16:   
17: Update $\mathcal { E } ^ { m } : e _ { m , * } \gets e _ { p , * }$ and $e _ { * , m } \gets e _ { * , q }$ EY   
18: end for   
19: EY $\mathcal { G }  \{ G ^ { m } \} \cup ( \mathcal { G } \setminus \{ G ^ { s } , G ^ { t } \} )$   
20: end while   
21: $G _ { \mathrm { v i d e o } }  \mathrm { e x t r a c t } ( \mathcal { G } )$   
22: return Gvideo

scene graphs derived from each segment into a unified graph that represents the overall visual content of the video. We first describe our graph merging procedure and then introduce a subgraph extraction technique designed to support more focused and coherent video caption generation.

# 3.3.1. MERGING TWO SCENE GRAPHS

We first describe our scene graph merging technique. Given two scene graphs, $G ^ { s } = ( \mathcal { O } ^ { s } , \mathcal { E } ^ { s } )$ and $G ^ { t } = ( \mathcal { O } ^ { t } , \mathcal { E } ^ { t } )$ , constructed from captions corresponding to two different segments, we run the Hungarian algorithm to obtain an optimal matching between the two object sets, $\mathcal { O } ^ { s }$ and $\mathcal { O } ^ { t }$ , which is formally expressed as

$$
\pi ^ { * } = \operatorname * { a r g m a x } _ { \pi \in \Pi } \sum _ { i } { \frac { \psi _ { i } ( \phi ( G ^ { s } ) ) } { \| \psi _ { i } ( \phi ( G ^ { s } ) ) \| } } \cdot { \frac { \psi _ { i } ( \phi ( G _ { \pi } ^ { t } ) ) } { \| \psi _ { i } ( \phi ( G _ { \pi } ^ { t } ) ) \| } } ,
$$

where $\phi ( \cdot )$ denotes a graph encoder, $\psi _ { i } ( \cdot )$ is a function to extract the $i ^ { \mathrm { { t h } } }$ object from an embedded graph, and $\pi \in \Pi$ indicates a permutation of objects in a graph. Note that the object matching is based on their cosine similarity, where we introduce dummy objects to deal with different numbers of objects for matching.

After computing all matching pairs using the Hungarian algorithm, we identify a set of valid matches $\mathcal { M }$ by selecting object pairs $( o _ { p } ^ { s } , o _ { q } ^ { t } )$ whose similarity score $s _ { p , q }$ exceeds a predefined threshold $\tau$ For each valid match $( p , q ) \in { \mathcal { M } }$ the merged object $\hat { o } _ { m } \in \hat { \mathcal { O } }$ is defined as

$$
\hat { o } _ { m } = ( \hat { c } , \mathcal { A } _ { p } ^ { s } \cup \mathcal { A } _ { q } ^ { t } ) \in \hat { \mathcal { O } } ,
$$

where $\hat { c }$ denotes a class label of a merged object and $\hat { \mathcal { O } }$ represents the set of all merged objects obtained from valid matches. Note that $\hat { c }$ may differ from the original class label of $O _ { p } ^ { s }$ or $O _ { q } ^ { t }$ . This procedure results in a new merged scene graph, $G ^ { \acute { m } } = ( \mathcal { O } ^ { m } , \mathcal { E } ^ { m } )$ , which combines each valid pair of matched objects, creating a new object.

We perform graph merging by iteratively selecting and consolidating pairs of graphs based on their embedding similarity. In each iteration, the two most similar graphs are merged into a single graph, which replaces the original pair in the set of graphs. This process is repeated until only one unified scene graph remains. The final scene graph provides a comprehensive representation of the entire video that preserves detailed information from individual segments. Algorithm 1 describes the detailed procedure of our graph consolidation strategy.

# 3.3.2. PRIORITIZED SUBGRAPH EXTRACTION

When concise and focused video captions are desired, we apply subgraph extraction to retain only the most contextually relevant information. During the graph merging process, we track each node's merge count as a measure of its significance within the consolidated graph. We then identify the top $k$ nodes with the highest merge counts and extract their corresponding subgraphs. This approach prioritizes objects that consistently appear across multiple frames, as they often represent key entities in the scene. By focusing on salient elements and filtering out irrelevant details, our method constructs a compact scene graph that enables more focused video captioning.

# 4. Video Caption Generation

Our ultimate goal is to generate captions from a consolidated scene graph. To this end, we develop a graph-to-text decoding model trained on a dataset of graph-text pairs. At inference time, the model takes the consolidated scene graph representing the entire video as input and generates a caption that describes the video as a whole.

# 4.1. Graph-to-text model

Our graph-to-text model consists of a transformer-based graph encoder and a text decoder. The encoder processes the input scene graph to produce a graph embedding, which conditions the decoder to generate the final caption. To reflect the graph topology in our model, we design an attention mask in the graph encoder that restricts attention propagation to the edges defined in the scene graph.

To construct input tokens for the graph encoder, we convert the text values associated with each graph component, such as object classes $c _ { i }$ , attribute sets $A _ { i }$ , and edge labels $r _ { i , j }$ (e.g., "elderly", "woman", "cook in", "kitchen"), to sequences of embedding vectors. Additionally, we append a learnable embedding token that attends to all other tokens, enabling the aggregation of global context and facilitating information flow across the entire graph, including between disconnected nodes.

# 4.2. Training

We train the graph-to-text model on a large-scale collection of graph-text pairs. To construct this dataset, we curated approximately 2.5 million captions from diverse image captioning datasets, including MS-COCO (Chen et al., 2015), Flickr30k (Young et al., 2014), TextCaps (Sidorov et al., 2020), Visual Genome (Krishna et al., 2017b), and Visual Genome Paragraph Captions (Krause et al., 2017), to cover a broad range of visual scene contexts. To further enrich the dataset, we incorporated model-generated captions for videos in Kinetics-400 (Kay et al., 2017), where LLaVANeXT-7B (Liu et al., 2024) is applied to four uniformly sampled frames per video. Each caption is then parsed into a scene graph using a textual scene graph parser, yielding a graph-text pair for training.

Using the graph-text pairs, we train the graph-to-text decoder with a next-token prediction objective, aiming to generate the ground-truth caption conditioned on the input scene graph, as formally defined below:

$$
\mathcal { L } ( \theta ) = \sum _ { i = 1 } ^ { N } \log P _ { \theta } ( t _ { i } \mid t _ { 1 : i - 1 } , G ) ,
$$

where $t _ { i }$ represents the $i ^ { \mathrm { { t h } } }$ token in the source text, and $N$ denotes the total number of tokens.

# 5. Experiment

This section presents the effectiveness of the proposed approach through performance evaluation and analysis on both video captioning and video paragraph captioning datasets.

# 5.1. Experimental setup

We provide the detailed information about target tasks with their datasets and baselines. We also discuss a list of performance metrics used in our evaluation.

# 5.1.1. TARGET TASKS AND BASELINES

Our evaluation consists of two zero-shot tasks: (1) video captioning, using the standard test splits of MSR-VTT $\mathrm { { X u } }$ et al., 2016) and MSVD (Chen & Dolan, 2011), and (2) video paragraph captioning, using the ae-val set of ActivityNet Captions (Krishna et al., 2017a), which contains longer videos with multiple events.

We primarily compare our method against LLM-based approaches. Specifically, we first establish an LLM summarization baseline, which directly summarizes the same set of segment-level captions used by our method. This baseline provides a direct comparison between the proposed scene graph consolidation and the simple aggregation of segment-level captions by LLMs. We use the open-source Mistral-7B-Instruct- $\mathbf { v } 0 . 3 ^ { 1 }$ for all datasets. For the ActivityNet Captions dataset, we additionally employ GPT-4o mini, a more powerful proprietary model. Details of the prompt instructions used for the LLM summarization baselines are provided in Appendix B.

We also compare our method against LLM-based video understanding methods, e.g., VidIL (Wang et al., 2022b) and Video ChatCaptioner (Chen et al., 2023), which utilize commercial LLMs along with textual representations derived from VLMs. VidIL constructs rich input sequences by combining various textual cues such as objects, events and frame captions extracted from multiple image-based VLMs, and incorporates few-shot exemplars to guide the LLM in generating video captions. Similarly, Video ChatCaptioner adopts an interactive question-answering framework between image VLM and LLMs.

Note that we primarily focus on LLM-based approaches, as other approaches typically require supervised fine-tuning, making direct zero-shot comparisons infeasible. Additional comparisons with broader zero-shot video captioning approaches—for example, test-time optimization, inference optimization, and text-only training methods—on MSRVTT are included in the supplementary document.

# 5.1.2. EvALuaTION METRiCS

Following standard performance evaluation protocols in video captioning, our experiments adopt $n$ -gram-based metrics, including BLEU-4 $( \mathrm { B } @ 4 )$ (Papineni et al., 2002), METEOR (Banerjee & Lavie, 2005), and CIDEr (Vedantam et al., 2015), which measure the overlap between generated and reference captions. Since these $n$ -gram-based metrics are limited in capturing semantic details and contextual accuracy beyond literal phrase matching, we additionally employ BERTScore (Zhang et al., 2020), an embeddingbased evaluation metric widely used in natural language processing tasks such as machine translation and summarization. BERTScore measures token-level cosine similarities between generated and reference captions, capturing semantic similarity beyond $n$ -gram matches as follows:

$$
\begin{array} { r l } & { P _ { \mathrm { B E R T } } = \displaystyle \frac { 1 } { | \hat { \mathcal Z } | } \sum _ { \hat { z } _ { j } \in \hat { \mathcal Z } } \operatorname* { m a x } _ { i \in \mathcal Z } z _ { i } ^ { \top } \hat { z } _ { j } , } \\ & { R _ { \mathrm { B E R T } } = \displaystyle \frac { 1 } { | \mathcal Z | } \sum _ { z _ { i } \in \mathcal Z } \operatorname* { m a x } _ { i \in \hat { \mathcal Z } } z _ { i } ^ { \top } \hat { z } _ { j } , } \\ & { F _ { \mathrm { B E R T } } = \displaystyle \frac { 2 \cdot P _ { \mathrm { B E R T } } \cdot R _ { \mathrm { B E R T } } } { P _ { \mathrm { B E R T } } + R _ { \mathrm { B E R T } } } , } \end{array}
$$

where $\mathcal { Z } \equiv \{ z _ { 1 } , z _ { 2 } , . . . \}$ and $\hat { \mathcal { Z } } \equiv \{ \hat { z } _ { 1 } , \hat { z } _ { 2 } , \dots \}$ represent the sets of token embeddings in the reference and generated captions, respectively.

# 5.2. Implementation details

Our graph-to-text model consists of a graph encoder and a text decoder, with a total of $2 3 5 \mathbf { M }$ parameters. The BERTbase model (Devlin et al., 2019) is employed for our encoder, with attention masking as described in Section 4.1, and only the decoder part of T5-base (Raffel et al., 2020) is used as our text decoder.

The graph-to-text model is trained on graph-text pairs constructed in Section 4.2 for $1 K$ iterations with a batch size of 512. We employ the AdamW (Loshchilov, 2019) optimizer with a weight decay of 0.05, an initial learning rate of 0.0001, and linear warm-up for the first $1 \%$ of training steps. For video paragraph captioning, the model is further fine-tuned for 400 iterations on the subset of the constructed graph-text pairs obtained from the Visual Genome paragraph captioning dataset (Krause et al., 2017).

Segment-level captions are generated using off-the-shelf VLMs. To demonstrate the flexibility of our approach, we employed both image-centric VLMs, including BLIP (Li et al., 2022) and BLIP2 (Li et al., 2023a), and video-centric VLM, InternVL2.5 (Chen et al., 2024a). For MSR-VTT and MSVD, we uniformly sample six frames per video to generate captions using image-centric models. For ActivityNet Captions, we select twelve frames per video when using image-centric VLMs, while extracting twelve video clips for the video-centric model.

method (SGVC) with LLM-based video understanding methods. $^ \dagger$ indicates that the method utilizes reference captions from the target .   

<table><tr><td>Dataset</td><td>Method</td><td>Backbone VLM</td><td>B@4</td><td>METEOR</td><td>CIDEr</td><td>PBERT</td><td>RBERT</td><td>FBERT</td></tr><tr><td rowspan="4">MSR-VTT</td><td>VidIL (Wang et al., 2022b)</td><td rowspan="2">BLIP+CLIP</td><td>3.2</td><td>14.8</td><td>3.1</td><td>0.134</td><td>0.354</td><td>0.225</td></tr><tr><td>VidIL (Wang et al., 2022b)</td><td>13.6</td><td>20.0</td><td>20.2</td><td>0.461</td><td>0.552</td><td>0.490</td></tr><tr><td>Video ChatCaptioner (Chen et al., 2023)</td><td>BLIP2</td><td>13.2</td><td>22.0</td><td>16.5</td><td>0.396</td><td>0.510</td><td>0.436</td></tr><tr><td>SGVC (Ours)</td><td>BLIP BLIP2</td><td>17.7 18.4</td><td>22.5 23.1</td><td>24.0 26.1</td><td>0.476 0.467</td><td>0.539 0.542</td><td>0.490 0.487</td></tr><tr><td rowspan="5">MSVD</td><td rowspan="2">VidIL (Wang et al., 2022b)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BLIP+CLIP</td><td>2.5</td><td>16.5</td><td>2.3</td><td>0.124</td><td>0.404</td><td>0.238</td></tr><tr><td rowspan="2">VidIL (Wang et al., 2022b) Video ChatCaptioner (Chen et al., 2023)</td><td rowspan="2">BLIP2</td><td>30.7 22.7</td><td>32.0</td><td>60.3</td><td>0.656</td><td>0.726</td><td>0.674</td></tr><tr><td></td><td>31.8</td><td>35.8</td><td>0.496 0.575</td><td>0.651 0.646</td><td>0.550</td></tr><tr><td rowspan="2">SGVC (Ours)</td><td>BLIP BLIP2</td><td>22.6</td><td>30.2</td><td>50.2</td><td></td><td></td><td>0.589</td></tr><tr><td></td><td>25.3</td><td>32.0</td><td>53.3</td><td>0.571</td><td>0.669</td><td>0.597</td></tr></table>

GVC with the LLM summarization baseline. Bold numbers indicate the highest scores.   

<table><tr><td>Dataset</td><td>Method</td><td>Backbone VLM</td><td>B@4</td><td>METEOR</td><td>CIDEr</td><td>PBERT</td><td>RBERT</td><td>FBERT</td></tr><tr><td rowspan="3">MSR-VTT</td><td>Summarization w/ Mistral-7B</td><td>BLIP BLIP2</td><td>9.6 11.5</td><td>21.6 23.1</td><td>10.8 15.4</td><td>0.313</td><td>0.516 0.528</td><td>0.395 0.397</td></tr><tr><td></td><td>BLIP</td><td>17.7</td><td>22.5</td><td>24.0</td><td>0.308 0.476</td><td>0.539</td><td>0.490</td></tr><tr><td>SGVC (Ours)</td><td>BLIP2</td><td>18.4</td><td>23.1</td><td>26.1</td><td>0.467</td><td>0.542</td><td>0.487</td></tr><tr><td rowspan="3">MSVD</td><td>Summarization w/ Mistral-7B</td><td>BLIP</td><td>15.2</td><td>28.3</td><td>30.3</td><td>0.477</td><td>0.623</td><td>0.527</td></tr><tr><td></td><td>BLIP2</td><td>22.5</td><td>31.9</td><td>41.6</td><td>0.500</td><td>0.664</td><td>0.558</td></tr><tr><td>SGVC (Ours)</td><td>BLIP BLIP2</td><td>22.6 25.3</td><td>30.2 32.0</td><td>50.2 53.3</td><td>0.575 0.571</td><td>0.646 0.669</td><td>0.589 0.597</td></tr></table>

For generating the final video caption, we apply a beam search with five beams, a maximum sequence length of 32 and a length penalty of 0.6. For video captioning on MSRVTT, we apply prioritized subgraph extraction with $k = 1$ to emphasize salient visual information. Video paragraph caption, which requires more detailed descriptions, is generated using a beam search with three beams, a maximum sequence length of 400, and a repetition penalty of 3.0.

# 5.3. Main results

We present quantitative results for zero-shot video captioning on the MSR-VTT and MSVD datasets in Tables 1 and 2, and for zero-shot video paragraph captioning on the ActivityNet Captions ae-val set in Tables 3 and 4.

# 5.3.1. ZERO-SHOT VIDEO CAPTIONING

Table 1 compares the proposed method, SGVC, with existing LLM-based video understanding approaches, VidIL and Video ChatCaptioner. SGVC consistently achieves strong zero-shot performance across most metrics on both the MSR-VTT and MSVD datasets, outperforming the existing methods. VidIL, although it leverages diverse textual cues from multiple sources, shows limited performance in the zero-shot setting. Notably, SGVC performs competitively even against VidIL's few-shot setting, which heavily depends on dataset-specific exemplars. Video ChatCaptioner, which aggregates information through multi-turn question answering between an LLM and BLIP2, often suffers from hallucinations or overemphasis on irrelevant details, leading to failures in capturing the core content of the video (e.g., "There are no animals present in the park scene.").

Table 2 provides a controlled comparison between SGVC and an LLM-based summarization method, clearly highlighting the effectiveness of our scene graph consolidation approach. Both methods start from an identical set of segment-level captions and this experiments isolates the impact of the graph consolidation. Although LLM summarization produces fluent and expressive captions, it sometimes overlooks details of objects and events within a scene. In contrast, SGVC explicitly integrates segment-level scene graphs into a unified representation, which is helpful for preserving object identities and relationships consistently throughout the video.

method (SGVC) with LLM-based video understanding methods. $^ \dagger$ indicates that the method utilizes reference captions from the target .   

<table><tr><td>Method</td><td>Backbone VLM</td><td>B@4</td><td>METEOR</td><td>CIDEr</td><td>PBeRT</td><td>RBERT</td><td>FBERT</td></tr><tr><td>VidIL (Wang et al., 2022b)</td><td>BLIP+CLIP</td><td>1.0</td><td>5.8</td><td>4.6</td><td>0.122</td><td>0.135</td><td>0.125</td></tr><tr><td>VidIL† (Wang et al., 2022b)</td><td></td><td>2.9</td><td>7.6</td><td>3.3</td><td>0.414</td><td>0.243</td><td>0.323</td></tr><tr><td>Video ChatCaptioner (Chen et al., 2023)</td><td>BLIP2</td><td>2.4</td><td>8.9</td><td>1.6</td><td>0.207</td><td>0.202</td><td>0.200</td></tr><tr><td>SGVC (Ours)</td><td>BLIP</td><td>6.7</td><td>11.6</td><td>16.6</td><td>0.367</td><td>0.285</td><td>0.322</td></tr><tr><td></td><td>BLIP2</td><td>7.4</td><td>12.4</td><td>20.9</td><td>0.367</td><td>0.304</td><td>0.331</td></tr></table>

with the LLM summarization baselines. Bold numbers indicate the highest scores.   

<table><tr><td>Method</td><td>Backbone VLM</td><td>B@4</td><td>METEOR</td><td>CIDEr</td><td>PBERT</td><td>RBERT</td><td>FBERT</td></tr><tr><td rowspan="3">Summarization w/ Mistral-7B</td><td>BLIP</td><td>3.4</td><td>9.4</td><td>7.5</td><td>0.292</td><td>0.268</td><td>0.276</td></tr><tr><td>BLIP2</td><td>4.1</td><td>10.4</td><td>9.6</td><td>0.307</td><td>0.293</td><td>0.295</td></tr><tr><td>InternVL2.5</td><td>4.5</td><td>10.8</td><td>11.6</td><td>0.333</td><td>0.318</td><td>0.319</td></tr><tr><td rowspan="3">Summarization w/ GPT-4o mini</td><td>BLIP</td><td>4.6</td><td>10.2</td><td>10.3</td><td>0.325</td><td>0.284</td><td>0.300</td></tr><tr><td>BLIP2</td><td>5.0</td><td>10.6</td><td>12.1</td><td>0.343</td><td>0.301</td><td>0.317</td></tr><tr><td>InternVL2.5</td><td>5.8</td><td>11.4</td><td>15.3</td><td>0.352</td><td>0.332</td><td>0.336</td></tr><tr><td rowspan="3">SGVC (Ours)</td><td>BLIP</td><td>6.7</td><td>11.6</td><td>16.6</td><td>0.367</td><td>0.285</td><td>0.322</td></tr><tr><td>BLIP2</td><td>7.4</td><td>12.4</td><td>20.9</td><td>0.367</td><td>0.304</td><td>0.331</td></tr><tr><td>InternVL2.5</td><td>8.0</td><td>13.2</td><td>24.1</td><td>0.359</td><td>0.326</td><td>0.338</td></tr></table>

<table><tr><td>Method</td><td>VLM Backbone</td><td>Params. (B)</td><td>GPU (GB)</td><td>Time (s)</td><td>CIDEr</td><td>Using reference</td><td>e Usi GPT API</td></tr><tr><td>VidIL</td><td>BLIP+CLIP</td><td>0.67</td><td>3.57</td><td>1.32</td><td>20.2</td><td>✓</td><td>✓</td></tr><tr><td>Video ChatCaptioner</td><td>BLIP2</td><td>3.75</td><td>14.53</td><td>3.65</td><td>16.5</td><td>-</td><td>✓</td></tr><tr><td>Summarization w/ Mistral-7B</td><td>BLIP</td><td>7.50</td><td>14.50</td><td>1.27</td><td>10.8</td><td></td><td>-</td></tr><tr><td rowspan="2">SGVC (Ours)</td><td>BLIP2</td><td>11.00</td><td>28.20</td><td>1.51</td><td>15.4</td><td></td><td></td></tr><tr><td>BLIP</td><td>0.74</td><td>5.07</td><td>1.14</td><td>24.0</td><td></td><td></td></tr><tr><td></td><td>BLIP2</td><td>4.24</td><td>18.40</td><td>1.37</td><td>26.1</td><td></td><td></td></tr></table>

Table 5. Comparison of computational costs between SGVC and LLM-based methods on the MSR-VTT test set.   

# 5.3.2. ZERO-SHOT VIDEO PARAGRAPH CAPTIONING

Table 3 presents a comparison between SGVC and other LLM-based video understanding methods for zero-shot video paragraph captioning on the ActivityNet Captions ae-val set. Consistent with the results observed in zero-shot video captioning in Table 1, SGVC clearly outperforms competing methods. The performance gap is even more pronounced in the paragraph captioning task, where effectively modeling long-range context and maintaining coherence across multiple events is essential.

Table 4 compares SGVC with LLM summarization techniques, using both Mistral-7B and a stronger commercial model, GPT-4o mini. While GPT-4o mini offers significant performance gains over Mistral-7B, it still falls short of SGVC, highlighting the effectiveness of our graph consolidation approach. Furthermore, replacing the backbone captioner with InternVL2.5 further improves SGVC's performance, benefiting from its video-centric design and strong temporal modeling capabilities, despite having significantly fewer parameters than BLIP2 (938M vs. 3.74B). These results clearly demonstrate SGVC's flexibility and plug-andplay compatibility with a wide range of vision-language model architectures.

# 5.4. Analysis

Efficiency Table 5 presents a detailed comparison of computational costs, in terms of average per-video inference time and peak GPU memory usage on a single NVIDIA A6000 GPU, along with captioning performance (CIDEr) on the MSR-VTT test set. SGVC consistently outperforms LLM-based summarization approaches across all computational measures, regardless of the underlying backbones. Moreover, our scene graph merging algorithm, which currently runs on the CPU, could be further accelerated by [Ground-truth] A man opening a toy egg set.

![](images/2.jpg)

![](images/3.jpg)

[Ground-truth] A track runner is preparing to run a race.

[LLM summ.] A hand opens a toy box, revealing a gift set with a toy airplane, an orange plastic.

VidIL] A person is unboxing and demonstrating a toy craft kit.

[Video ChatCaptioner] The video shows a person holding a toy car and a cup of water while wearing a shirt.

[Ours] A hand holding a toy airplane in front of a box with a surprised expression.

[LLM summ.] A group of runners, including females, stretch, crouch at the starting line, and.

[VidIL] A group of athletes competing in various track and field events.

[Video ChatCaptioner] The video shows a woman participating in a track and field event, wearing a red shirt and shorts.

[Ours] A group of runners crouching down a line on a track competing in a race.

![](images/4.jpg)

[Ground-truth] A child and a man are dancing to a song.

![](images/5.jpg)

[Ground-truth] A mom and daughter are walking around around town.

[LLM summ.] A family is dancing together in a room, featuring a man, a woman, and a child.

[LLM summ.] A woman and her daughter, accompanied by two other women, are walking down a street.

[VidIL] A father and daughter share a special moment through dance.

[Video ChatCaptioner] A litle girl in a striped dress dances in a living room to an unknown background music.

[Ours] A man and a young girl dancing in a living room.

[VidIL] A group of people are walking down a street in Japan.

[Video ChatCaptioner] The video shows a girl wearing a white shirt walking down a street with a bag. The color of the bag is not known.

[Ours] A woman and her daughter walk down a street with a bicycle in the background.

![](images/6.jpg)

[Ground-truth] A female soldier talks about her athletics.

![](images/7.jpg)

[Ground-truth] People sitting at a table with food.

[LLM summ.] A woman in a blue jacket poses outdoors, followed by a man in a military uniform standing.

VidIL] A female athlete competes in a military-themed sports event.

[Video ChatCaptioner] The video features a woman in a navy uniform standing in front of a sign that says "phili" with a white wall in the background.

[Ours] A woman in a uniform stands in front of a sign, holding medals and smiling.

[LLM summ.] A television show scene with a man and a woman with long and purple hair, followed by a woman.   
[VidIL] A group of friends are having a dinner party and trying out different hairstyles.   
[Video ChatCaptioner] A group of people are sitting at a table in a park, eating. There are no animals present in the park scene.   
[Ours] A woman in a dress is sitting at a table with food surrounded by people.

u o ot  sMia ViChat n G u.

Table 6. Analysis on the hyperparameter $k$ in the prioritized subgraph extraction, on the MSR-VTT test set.   

<table><tr><td>k</td><td>METEOR</td><td>CIDEr</td><td>PBERT</td><td>RBERT</td><td>FBERT</td></tr><tr><td>1</td><td>23.1</td><td>26.1</td><td>0.467</td><td>0.542</td><td>0.487</td></tr><tr><td>3</td><td>23.8</td><td>24.9</td><td>0.454</td><td>0.554</td><td>0.486</td></tr></table>

<table><tr><td>τ</td><td>CIDEr</td><td>FBERT</td><td>τ</td><td>CIDEr</td><td>FBERT</td></tr><tr><td>0.95</td><td>50.0</td><td>0.589</td><td>0.85</td><td>49.9</td><td>0.589</td></tr><tr><td>0.90</td><td>50.2</td><td>0.589</td><td>0.80</td><td>49.9</td><td>0.589</td></tr></table>

Table 7. Analysis on the threshold $\tau$ used in graph consolidation, on the MSVD test set.   

GPU implementation. VidIL and Video ChatCaptioner exhibit slower inference times and lower captioning accuracy. While they consume less GPU memory, their dependence on GPT API calls introduces additional latency.

subgraph, as described in Section 3.3.2. As shown in Table 6, lower $k$ values result in more concise subgraphs that emphasize salient objects, leading to improvements in precision-oriented metrics, such as CIDEr and $P _ { \mathrm { B E R T } }$ .

Impact of hyperparameters We analyze the effect of the hyperparameter $k$ , which controls the size of the extracted

In contrast, higher $k$ values yield richer subgraphs that capture broader contextual information, thereby improving recall-oriented metrics, METEOR and $R _ { \mathrm { B E R T } }$

![](images/8.jpg)

Cha rward and looking at the camera with a nervous expression.

he man is standing behind the man. The man is holding a weight. The man is standing with his arms raised.

![](images/9.jpg)

ours that into a glass and sticks a straw in it.

Lu   .

dog can be seen in a bathroom setting.

.

and 4) SGVC (Ours).

We also conducted evaluation by varying the cosine similarity threshold $\tau$ , as reported in Table 7. The results demonstrate stable performance within the range $\tau \in [ 0 . 8 0 , 0 . 9 5 ]$ and we set $\tau = 0 . 9$ for all experiments.

Qualitative results Figures 2 and 3 present qualitative examples of zero-shot video captioning on the MSR-VTT test set and video paragraph captioning on ActivityNet Captions ae-val set, respectively. Our method generates detailed and contextually rich captions that accurately capture events, objects, and relationships across frames. While LLM summarization and Video ChatCaptioner produce fluent sentences, they occasionally introduce hallucinated content, such as objects or attributes that are not actually present in the video.

# 6. Conclusion

We introduced a novel framework for fine-grained captioning of long videos by consolidating information across multiple temporal segments. Our approach merges scene graphs extracted from segment-level captions to generate comprehensive and coherent video descriptions. This framework provides a computationally efficient and training-free alternative to existing methods. In contrast to LLM-based approaches, our method significantly reduces computational demands by leveraging a lightweight graph-to-text model with substantially fewer parameters. Extensive experiments on both video captioning and video paragraph captioning tasks validate the effectiveness of our method. These results highlight the potential of graph-based consolidation as a foundation for future advances in long video captioning.

# Acknowledgements

We thank Do Young Eun at North Carolina State University for the valuable discussions. This work was supported in part by National Research Foundation of Korea (NRF) grant [RS-2022-NR070855, Trustworthy Artificial Intelligence], Institute of Information & communications Technology Planning & Evaluation (IITP) grants [RS2022-II220959 (No.2022-0-00959), (Part 2) Few-Shot Learning of Causal Inference in Vision and Language for Decision Making; No.RS-2021-II212068, AI Innovation Hub (AI Institute, Seoul National University); No.RS-2021-II211343, Artificial Intelligence Graduate School Program (Seoul National University)] funded by the Korea government (MSIT), and by Brain Pool program funded by the Ministry of Science and ICT through the National Research Foundation of Korea (No. RS-2024-00408610).

# Impact Statement

The broader impact of this research lies in enabling effective captioning of long videos by leveraging existing visionlanguage models without any additional fine-tuning on largescale annotated video datasets. While there is potential for societal impacts arising from this technology, we have not identified any significant negative consequences directly associated with our approach.