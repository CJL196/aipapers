# VideoForest: Person-Anchored Hierarchical Reasoning for Cross-Video Question Answering

Yiran Meng\* Sun Yat-Sen University Zhuhai, China

Junhong Ye\* Sun Yat-Sen University Zhuhai, China

Wei Zhou Cardiff University United Kingdom

Guanghui Yue Shenzhen University Shenzhen, China

Xudong Mao Sun Yat-Sen University Zhuhai, China

Ruomei Wang Sun Yat-Sen University Guangzhou, China

Baoquan Zhao Sun Yat-Sen University Zhuhai, China

# Abstract

Cross-video question answering presents significant challenges beyond traditional single-video understanding, particularly in establishing meaningful connections across video streams and managing the complexity of multi-source information retrieval. We introduce VideoForest, a novel framework that addresses these challenges through person-anchored hierarchical reasoning, enabling effective cross-video understanding without requiring end-to-end training. VideoForest integrates three key innovations: 1) a human-anchored feature extraction mechanism that employs ReID and tracking algorithms to establish robust spatiotemporal relationships across multiple video sources; 2) a multi-granularity spanning tree structure that hierarchically organizes visual content around person-level trajectories; and 3) a multi-agent reasoning framework that efficiently traverses this hierarchical structure to answer complex queries. To evaluate our method, we develop CrossVideoQA1, a comprehensive benchmark specifically designed for person-centric cross-video analysis. Experimental results demonstrate VideoForest's superior performance in cross-video reasoning tasks, achieving $7 1 . 9 3 \%$ accuracy in person recognition, $8 3 . 7 5 \%$ in behavior analysis, and $5 1 . 6 7 \%$ in summarization and reasoning.

# Keywords

Cross-Video Question Answering, Person-Anchored Reasoning, Hierarchical Video Representation, Multi-Agent Framework

# ACM Reference Format:

Yiran Meng, Junhong Ye, Wei Zhou, Guanghui Yue, Xudong Mao, Ruomei Wang, and Baoquan Zhao. 2025. VideoForest: Person-Anchored Hierarchical Reasoning for Cross-Video Question Answering . In Proceedings of the 33rd ACM International Conference on Multimedia (MM '25), October 2731, 2025, Dublin, Ireland. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/ 3746027.3754573

# 1 Introduction

Cross-video understanding represents one of the most challenging frontiers in computer vision, requiring systems to extract, correlate, and reason about information distributed across multiple video streams. Unlike single-video analysis, where context remains contained within temporal boundaries, cross-video reasoning demands sophisticated mechanisms to establish meaningful connections across different spatial viewpoints and temporal sequences. This capability is particularly crucial in surveillance and monitoring scenarios, where critical information is inherently fragmented across multiple cameras, necessitating unified analysis for comprehensive situational awareness.

Consider a security investigation requiring analysts to determine: Which individual traversed all three campus buildings between 14:00-16:00? Answering such queries demands not only person identification and tracking within each video stream but crossreferencing identities and behaviors across multiple cameras with varying perspectives and recording conditions. Despite remarkable advances in video understanding, current methods remain fundamentally constrained by their single-stream processing paradigm, rendering them inadequate for queries that span multiple video sources. This architectural limitation prevents the integration of complementary information across camera viewpoints.

As illustrated in Figure 1, existing video question answering systems [7, 26, 32, 46] have primarily focused on maximizing performance within the bounds of individual videos, inadvertently reinforcing this single-stream constraint. Even recent advancements like VideoAgent [7], which introduces sophisticated agent-based search strategies, and Chat-Video [35], which pioneers motion trajectory analysis, ultimately operate within the confines of isolated video processing. The critical capability of establishing semantic bridges between separate video streams—essential for true crossvideo reasoning—remains largely unexplored in the literature.

![](images/1.jpg)  

Figure 1: Comparison of single-video vs. cross-video question answering paradigms.

To address these limitations, we introduce VideoForest, a novel hierarchical framework that enables efficient person-centric reasoning across multiple video streams (see Figure 2). Our key insight is that human subjects serve as natural bridge points between different videos, providing consistent reference entities around which cross-video relationships can be structured. VideoForest implements this insight through three innovative components: First, we develop a person-anchored feature extraction mechanism that employs Re-Identification (ReID) and tracking algorithms to establish consistent identity representations across multiple videos, creating robust spatio-temporal relationships that span different camera viewpoints. Second, we design a multi-granularity spanning tree structure that hierarchically organizes visual content around person-level trajectories, enabling efficient navigation from coarse scene-level information to fine-grained behavioral details. Third, we implement a multi-agent reasoning framework that efficiently traverses this hierarchical structure to perform sophisticated crossvideo reasoning while maintaining computational tractability.

To evaluate our approach and advance research in cross-video understanding, we introduce CrossVideoQA, the first comprehensive benchmark dataset specifically designed for person-centric cross-video question answering in surveillance scenarios. Our extensive experiments demonstrate VideoForest's effectiveness across various reasoning tasks.

The primary contributions of this work are threefold:

We introduce the first person-anchored hierarchical framework for cross-video question answering, pioneering a treebased architecture that uses human subjects as bridge points to connect multiple video streams, enabling unified understanding across distributed visual information. •We develop an efficient multi-granular video organization strategy integrated with a multi-agent reasoning framework that preserves critical temporal-spatial relationships while making cross-video question answering computationally tractable. •We present CrossVideoQA, a novel benchmark dataset for evaluating person-centric cross-video question answering capabilities, establishing new evaluation protocols and performance baselines for this emerging research direction.

# 2 Related Work

# 2.1 Video Question Answering

VideoQA forms a cornerstone of multimodal comprehension alongside text-video retrieval and video captioning, demanding deep understanding of complex semantic and causal relationships between video and language [23]. To improve model robustness and interpretability, visual localization approaches enable models to highlight relevant video segments [20, 22] or keyframes [44, 50] for answer generation. While these methods successfully locate evidence, the reasoning process remains opaque. ChatVideo [35] introduces motion trajectories as fundamental analysis units, leveraging specialized visual foundation models to generate attribute annotations for enhanced temporal modeling in dynamic scenes. Another research line employs external Large Language Models (LLMs) as reasoning modules. LLoVi [45] converts Video QA into text-based QA via video captions, while VideoAgent [7] recursively assesses frame sufficiency for question answering. Although improving QA performance, these methods depend heavily on LLM linguistic reasoning, which is prone to hallucination and lacks interpretability. Video-CCAM [9] introduces cross-modal attention with causal masking between visual encoders and LLMs, demonstrating superior performance across multiple benchmarks. InternVL 2.5 [18, 38] refines training strategies and data quality while excelling in short-video understanding, long-video retrieval, and QA tasks. TV-trees [33] adopts neuro-symbolic approaches for explicit reasoning across visual and textual modalities, though assuming pre-transcribed video text. Despite advances in single-video tasks, systems capable of cross-video understanding and complex temporal QA remain scarce.

# 2.2 Structured Video Representation

Structured video representation enhances Video QA by transforming videos from frame sequences into hierarchical semantic representations, enabling better comprehension of object, action, and event relationships [41]. This approach fuses visual information at multiple granularities with corresponding linguistic concepts, improving accuracy and interpretability [8, 17, 39]. Recent videolanguage methods emphasize structured frame representations for efficient scene understanding [5, 12, 25, 30, 50]. LVNet [30] reduces redundancy through hierarchical keyframe selection, VideoReCap [12] introduces progressive captioning bridging short and long clip comprehension, and VideoTree [50] achieves breakthroughs with top-down video language embedding featuring dynamic depth adjustment for efficient long video comprehension. However, these approaches primarily address single-video analysis, leaving crossvideo correlation challenges unexplored. Our work extends these foundational ideas to multi-video domains by introducing humancentered connectivity and spatio-temporal relationship modeling for effective cross-video understanding.

# 2.3 Video Understanding Benchmarks

Video understanding tasks progress through three complexity levels: abstract understanding for overall video events [3, 11], temporal understanding for specific moment identification [24, 34], and spatio-temporal understanding for timing and spatial localization [1, 49]. This progression mirrors human cognitive processes of building comprehensive understanding from visual information [27]. Multimodal large language models (MLLMs) have catalyzed comprehensive evaluation frameworks [10, 16, 19, 28, 29]. VideoMM [10] established the first holistic MLLM evaluation benchmark, while VideoVistaAV [19] introduced multifaceted evaluation accounting for diverse content categories, time scales, and inference capabilities. Despite these advances, existing benchmarks focus predominantly on single-video comprehension, creating a critical gap in cross-video reasoning assessment. Our proposed CrossVideoQA benchmark addresses this limitation through carefully curated cross-video queries based on Edinburgh Office Surveillance and HACS datasets [31, 48], establishing novel evaluation criteria for multi-video understanding systems.

![](images/2.jpg)  

Figure 2: VideoForest architecture for cross-video question answering.

# 3 Methodology

# 3.1 Problem Definition and Notation

We formalize the cross-video question answering task as follows. Let $\mathcal { V } = \{ V _ { 1 } , V _ { 2 } , . . . , V _ { n } \}$ denote a collection of $n$ videos, where each video $V _ { i }$ comprises an ordered temporal sequence of frames $\mathcal { F } _ { i } = \{ f _ { i , 1 } , f _ { i , 2 } , . . . , f _ { i , m _ { i } } \}$ with $m _ { i }$ denoting the number of frames in video i. Our objective is to construct a unified hierarchical representation that enables efficient cross-video information retrieval and reasoning.

For each frame $f _ { i , j }$ (the $j$ -th frame of video i), we extract two complementary representations: (1) visual embeddings $\mathbf { v } ( f _ { i , j } ) \in \mathbb { R } ^ { d }$ ,a $d$ -dimensional dense representation capturing semantic visual con; and () peron detetions $\mathbf { p } ( f _ { i , j } ) = \{ ( t _ { k } , \mathbf { x } _ { k } , \mathrm { i d } _ { k } ) \} _ { k = 1 } ^ { K _ { i , j } }$ where $K _ { i , j }$ is the number of detected persons in frame $f _ { i , j }$ , $t _ { k } \in \mathbb { R }$ denotes the timestamp, $\mathbf { x } _ { k } = ( x _ { k } , y _ { k } ) \in \mathbb { R } ^ { 2 }$ represents spatial coordinates, and $\mathrm { i d } _ { k } \ \in \ \mathcal { I }$ is the unique person identifier from the set of all identities $\boldsymbol { \mathcal { T } }$ .

The cross-video question answering task can then be formally defined as a mapping function:

$$

Q : \mathcal { V } \times \mathcal { T } \times \mathcal { L }  \mathcal { A } ,
$$

where $\mathcal { T } \subset \mathbb { R } ^ { + }$ represents temporal constraints (e.g., time intervals of interest), $\mathcal { L } \subset \mathbb { R } ^ { 2 }$ represents spatial constraints (e.g., regions of interest), and $\mathcal { A }$ denotes the answer space, which may include textual responses, temporal localization, or entity identification. This formulation explicitly models the cross-video reasoning process as conditional on both temporal and spatial constraints, capturing the complex spatio-temporal relationships inherent in surveillance and monitoring scenarios. Our hierarchical VideoForest framework implements this mapping through a person-anchored tree structure that enables efficient traversal and integration of information across multiple video sources.

# 3.2 Dual-Stream Feature Extraction and Adaptive Segmentation

Building on our formal problem definition, we implement a complementary dual-stream architecture for comprehensive video representation. The visual content stream employs the ViCLIP encoder [36, 37], parameterized by $\theta _ { v }$ , to compute the frame-level embeddings defined in our notation:

$$
\mathbf { v } ( f _ { i , j } ) = \phi ( f _ { i , j } ; \theta _ { v } ) \in \mathbb { R } ^ { d } .
$$

Concurrently, the person-centric stream utilizes a specialized tracking model $\psi$ with parameters $\theta _ { p }$ to identify and extract the structured person representations:

$$
\mathbf { p } ( f _ { i , j } ) = \psi ( f _ { i , j } ; \theta _ { p } ) = \{ ( t _ { k } , \mathbf { x } _ { k } , \mathrm { i d } _ { k } ) \} _ { k = 1 } ^ { K _ { i , j } } ,
$$

where $K _ { i , j }$ denotes the number of persons detected in frame $f _ { i , j }$

To partition videos into semantically coherent segments, we define an adaptive boundary detection function $S : \mathcal { F } _ { i }  \{ 0 , 1 \}$ that identifies significant transitions through a disjunctive criterion:

$$
S ( f _ { i , j } ) = \mathbb { \mathbb { k } } [ C _ { 1 } ( f _ { i , j } ) \vee C _ { 2 } ( f _ { i , j } ) \vee C _ { 3 } ( f _ { i , j } ) ] ,
$$

where the three complementary criteria are formulated as:

$$
\begin{array} { r l } & { C _ { 1 } ( f _ { i , j } ) : \| \mathbf { v } ( f _ { i , j } ) - \mathbf { v } ( f _ { i , j + 1 } ) \| _ { 2 } > \epsilon _ { 1 } , \quad \mathrm { ( l o c a l ~ t r a n s i t i o n ) } } \\ & { C _ { 2 } ( f _ { i , j } ) : \| \mathbf { v } ( f _ { i , j } ) - \mathbf { v } ( f _ { i , j } ^ { \mathrm { c e n t } } ) \| _ { 2 } > \epsilon _ { 2 } , \quad \mathrm { ( g l o b a l ~ d e v i a t i o n ) } } \\ & { \quad C _ { 3 } ( f _ { i , j } ) : | \mathcal { P } ( f _ { i , j } ) \triangle \mathcal { P } ( f _ { i , j - 1 } ) | \geq \Delta \varphi . \quad \mathrm { ( p e r s o n - s e t ~ c h a n g ~ } } \end{array}
$$

Here, $\lVert \rVert ^ { \epsilon } [ \cdot ]$ denotes the indicator function, $\begin{array} { r l } { \mathcal { P } ( f _ { i , j } ) } & { { } = } \end{array}$ $\{ \mathrm { i d } _ { k } | ( t _ { k } , \mathbf { x } _ { k } , \mathrm { i d } _ { k } ) \in \mathbf { p } ( f _ { i , j } ) \}$ represents the set of person identities present in frame $f _ { i , j }$ , and $f _ { i , j } ^ { \mathrm { c e n t } }$ refers to a representative frame for the current segment. The multi-criterion approach operates at three distinct levels: $C _ { 1 }$ captures frame-to-frame appearance changes through local feature distance, $C _ { 2 }$ measures deviation from the segment's visual prototype to identify global content drift, and $C _ { 3 }$ quantifies person-centric dynamics through the cardinality of the symmetric difference $\triangle$ between consecutive person-identity sets. The thresholds $\epsilon _ { 1 } , \epsilon _ { 2 }$ and $\Delta \varphi$ are determined through cross-validation on a held-out dataset to optimize the trade-off between temporal granularity and semantic coherence. When a segment boundary is detected according to $S ( f _ { i , j } ) = 1$ , we create a new segment $S _ { i , k } = \{ f _ { i , j _ { \mathrm { s t a r t } } } , f _ { i , j _ { \mathrm { s t a r t } } + 1 } , . . . , f _ { i , j _ { \mathrm { e n d } } } \}$ where $j _ { \mathrm { s t a r t } }$ and $j _ { \mathrm { e n d } }$ denote the inclusive boundaries of the segment. This approach yields a sequence of non-overlapping segments $\{ S _ { i , 1 } , S _ { i , 2 } , . . . , S _ { i , n _ { i } } \}$ for each video $V _ { i }$ , effectively parsing the continuous video stream into discrete semantic units.

This adaptive segmentation serves as the foundational building block for our hierarchical tree representation, enabling efficient multi-granular video indexing and retrieval. The segments preserve semantic coherence while establishing manageable units for subsequent person-anchored correlation across videos. By incorporating both visual content and person-centric dynamics in the segmentation criteria, our approach ensures that the resulting segments maintain meaningful contextual boundaries that facilitate cross-video reasoning.

# 3.3 Multi-Level Semantic Representation

Given the segmented video structure, we construct semantically rich representations for each segment $s _ { i , k }$ . We define a multi-modal encoding function $\eta : S \times \mathcal { P } \to \mathbb { R } ^ { d }$ that maps visual content and person trajectories to a unified semantic space:

$$
\mathbf { C } ( S _ { i , k } ) = \eta ( \mathbf { v } ( f _ { i , j } ^ { \mathrm { k e y } } ) , \mathbf { P } ( S _ { i , k } ) ; \theta _ { \eta } ) ,
$$

where $f _ { i , j } ^ { \mathrm { k e y } } = f _ { i , \lfloor ( j _ { \mathrm { s t a r t } } + j _ { \mathrm { e n d } } ) / 2 \rfloor }$ representing the segment, $\mathbf { P } ( S _ { i , k } ) = \{ \mathbf { p } ( f _ { i , j } ) | f _ { i , j } \in S _ { i , k } \}$ denotes the aggregated set of person detections across all frames in the segment, and $\theta _ { \eta }$ parameterizes the encoding function. This formulation ensures that our semantic representation captures both the static visual content through the keyframe embedding and the dynamic person-centric activities through trajectory aggregation.

This multi-level semantic representation provides a rich foundation for cross-video reasoning by capturing both visual scene context and person-centric dynamics. The resulting segment-level encodings $\{ \mathbf { C } ( S _ { 1 , 1 } ) , \mathbf { C } ( S _ { 1 , 2 } ) , . . . , \mathbf { C } ( S _ { n , n _ { n } } ) \}$ serve as the semantic nodes in our hierarchical tree structure, enabling eficient retrieval and correlation of content across multiple videos.

# 3.4 VideoForest Construction

Based on the segmented videos and their semantic representations, we construct a hierarchical tree structure $\mathcal { T } = ( V , E )$ that organizes content at multiple granularities. Each node $v \in V$ is defined as a structured tuple:

$$
\boldsymbol { v } = ( t _ { \mathrm { s t a r t } } , t _ { \mathrm { e n d } } , \mathcal { R } _ { v } , \mathbf { C } _ { v } , \Gamma _ { v } ) ,
$$

where $[ t _ { \mathrm { s t a r t } } , t _ { \mathrm { e n d } } ] \subset \mathbb { R } ^ { + }$ represents the temporal interval spanned by th nhende, $\mathscr { R } _ { v } = \{ ( \mathrm { i d } _ { k } , \tau _ { k } ) \} _ { k = 1 } ^ { K _ { v } }$ contains eson rentfcation information with $\mathrm { i d } _ { k } \in \mathcal { I }$ and trajectory descriptors $\tau _ { k }$ , $\mathbf { C } _ { v } \in \mathbb { R } ^ { d }$ denotes the semantic content representation computed by function $\eta$ ,and $\Gamma _ { v } \subset V$ represents the set of child nodes.

The edge set $E ~ = ~ \{ ( v _ { i } , v _ { j } ) ~ \mid ~ v _ { j } ~ \in ~ \Gamma _ { v _ { i } } \}$ defines the hierarchical parent-child relationships that enable efficient multi-granular traversal. To ensure comprehensive temporal coverage while maintaining non-overlapping child segments, the recursive partitioning of nodes follows a disjoint cover principle implemented by the splitting function Split : $V  2 ^ { V }$ :

$$
{ \mathrm { S p l i t } } ( v ) = \{ v _ { 1 } , v _ { 2 } , \ldots , v _ { K _ { v } } \} ,
$$

satisfying the following temporal coverage and disjointness constraints:

$$
\bigcup _ { i = 1 } ^ { K _ { v } } [ t _ { \mathrm { s t a r t } } ( v _ { i } ) , t _ { \mathrm { e n d } } ( v _ { i } ) ] = [ t _ { \mathrm { s t a r t } } ( v ) , t _ { \mathrm { e n d } } ( v ) ] ,
$$

$$
\forall i \neq j : [ t _ { \mathrm { s t a r t } } ( v _ { i } ) , t _ { \mathrm { e n d } } ( v _ { i } ) ] \cap [ t _ { \mathrm { s t a r t } } ( v _ { j } ) , t _ { \mathrm { e n d } } ( v _ { j } ) ] = 0 .
$$

The splitting criteria are determined adaptively based on semantic similarity and person-level continuity, with splitting boundaries preferentially aligned with the segment boundaries identified during the segmentation process. For each video $V _ { i }$ , we construct a corresponding tree $\mathcal { T } _ { i }$ with the root node spanning the entire video duration and leaf nodes corresponding to the fine-grained segments $\{ S _ { i , k } \} _ { k = 1 } ^ { n _ { i } }$ facilitating rapid identification of relevant content in response to temporal and person-centric queries. The integration of person reidentification information $\mathcal { R } _ { v }$ at each node creates natural bridge points between different video trees, enabling cross-video correlation and reasoning based on person identity continuity.

![](images/3.jpg)  

Figure 3: Architecture of our distributed multi-agent framework for cross-video reasoning.

# 3.5 Collaborative Multi-Agent System for Cross-Video Reasoning

As illustrated in Fig. 3, our cross-video reasoning system integrates information from multiple video sources through a coordinated multi-agent architecture. This approach addresses the challenges of spatio-temporal relationships between videos, such as different viewpoints of the same scene or recordings from the same viewpoint at different times. The system employs four specialized agent modules working in concert to facilitate efficient cross-video reasoning. Our multi-agent reasoning system is implemented using the CrewAI framework [14], which provides modular agent-task scheduling and tool integration capabilities. We extend CrewAI to support dynamic tree-based traversal and cross-agent knowledge propagation.

3.5.1 Agent Architecture and Functional Specialization. Our multiagent framework consists of four specialized components, each with distinct functionality:

$$
\mathcal { A } = \{ \mathcal { A } _ { \mathrm { f l t e r } } , \mathcal { A } _ { \mathrm { r e t r i e v a l } } , \mathcal { A } _ { \mathrm { n a v i g a t e } } , \mathcal { A } _ { \mathrm { i n t e g r a t e } } \}
$$

The ${ \mathcal { A } } _ { \mathrm { f i t e r } }$ agent processes input queries to extract temporal and spatial constraints, identifying and selecting relevant video tree structures from the set $\{ \mathcal { T } _ { i } \} _ { i = 1 } ^ { n }$ The $\mathcal { A } _ { \mathrm { r e t r i e v a l } }$ agent manages knowledge base access, retrieving pertinent information while preventing redundant computations through confidence-based retrieval mechanisms. The ${ \mathcal { A } } _ { \mathrm { n a v i g a t e } }$ agent traverses the hierarchical tree structures using an optimized search strategy to locate query-relevant information. Finally, the Aintegrate agent synthesizes information from both the knowledge base and tree structures, performing crossvideo reasoning to generate comprehensive answers.

The reasoning workflow follows a sequential five-stage process: (1) video selection, (2) knowledge base retrieval, (3) hierarchical tree traversal, (4) cross-video information integration, and (5) knowledge base updating.

3.5.2 Knowledge Base Construction and Confidence-Based Maintenance. To address the computational challenges of repeatedly accessing the same information across multiple queries, we implement a global knowledge base $\mathcal { K }$ with confidence-weighted entries:

$$
\mathcal { K } = \{ ( d _ { i } , l _ { i } , s _ { i } , c _ { i } ) \} _ { i = 1 } ^ { N } ,
$$

where $d _ { i } \in \mathcal { D }$ represents date information, $l _ { i } \in \mathcal { L }$ denotes spatial location, $s _ { i } \in S$ is a descriptive string containing subject and action information, and $c _ { i } \in [ 0 , C _ { \operatorname* { m a x } } ]$ is the confidence score.

The $\mathcal { A } _ { \mathrm { r e t r i e v a l } }$ agent maintains the integrity of $\mathcal { K }$ through a formal update function $\mathcal { U } : \mathcal { K } _ { \mathrm { n e w } } \times \mathcal { K }  \mathcal { K }$ defined as:

$$
\mathcal { U } ( k _ { \mathrm { n e w } } , \mathcal { K } ) = \left\{ \begin{array} { l l } { \mathcal { K } \cup \{ ( d _ { \mathrm { n e w } } , l _ { \mathrm { n e w } } , s _ { \mathrm { n e w } } , 1 ) \} , } & { \mathrm { i f } ~ k _ { \mathrm { n e w } } \notin \mathcal { K } } \\ { \mathcal { K } \setminus \{ k _ { i } \} \cup \{ ( d _ { i } , l _ { i } , s _ { i } , c _ { i } + 1 ) \} , } & { \mathrm { i f } ~ k _ { \mathrm { n e w } } = k _ { i } \in \mathcal { K } } \\ { \mathcal { K } \setminus \{ k _ { i } \} \cup \{ ( d _ { i } , l _ { i } , s _ { i } , c _ { i } - 1 ) \} } & { \mathrm { i f } , k _ { \mathrm { n e w } } \approx k _ { i } ~ \mathrm { a n d } ~ c _ { i } > 2 } \\ { \mathcal { K } \setminus \{ k _ { i } \} \cup \{ ( d _ { \mathrm { n e w } } , l _ { \mathrm { n e w } } , s _ { \mathrm { n e w } } , 1 ) \} , } & { \mathrm { i f } ~ k _ { \mathrm { n e w } } \approx k _ { i } ~ \mathrm { a n d } ~ c _ { i } \leq 2 } \end{array} \right.
$$

where $k _ { \mathrm { n e w } } \approx k _ { i }$ denotes a semantic conflict between the new entry and an existing entry according to a similarity measure $\sin ( k _ { \mathrm { n e w } } , k _ { i } ) > \tau _ { \mathrm { s i m } }$ .

This confidence-based approach enables the system to selfcorrect over time, progressively refining the knowledge base through iterative query answering. When processing new queries, entries with confidence scores exceeding a threshold $\tau _ { \mathrm { c o n f } }$ are prioritized for retrieval, reducing computational load and improving response time.

3.5.3 Adaptive Hierarchical Search Optimization. The $\mathcal { A } _ { \mathrm { n a v i g a t e } }$ agent employs an efficient top-down search strategy $\boldsymbol { S } : \boldsymbol { Q } \times \boldsymbol { V } $ $2 ^ { \overset { \vartriangle } { \mathbf { C } } }$ that recursively explores the hierarchical structure. For a query $q \in { \cal Q }$ and a node $\textit { v } \in \textit { V }$ within tree $\mathcal { T }$ , the search function is formulated as:

$$
S ( q , v ) = \left\{ \begin{array} { l l } { \mathrm C _ { v } , } & { \mathrm { i f ~ R e l e v a n c e } ( q , \mathrm C _ { v } ) \geq \tau _ { \mathrm { r e l } } } \\ { \bigcup _ { v _ { c } \in \Gamma _ { v } } S ( q , v _ { c } ) , } & { \mathrm { o t h e r w i s e } } \end{array} \right.
$$

where Relevance $\because Q \times \mathbb { R } ^ { d }  [ 0 , 1 ]$ measures the semantic similarity between the query $q$ and the content representation $\mathbf { C } _ { v }$ of node v, and $\tau _ { \mathrm { r e l } } \in \left[ 0 , 1 \right]$ is a configurable relevance threshold.

The search process begins at the root nodes of selected video trees and progressively refines the exploration based on temporal, spatial, and person-centric constraints extracted from the query. When person-level information is present in the query, the search leverages the ReID information $\mathcal { R } _ { v }$ stored at each node to efficiently identify relevant content across different videos.

# 4 Experimental Evaluation

To comprehensively evaluate VideoForest's capabilities for crossvideo reasoning, we require a benchmark specifically designed to test integration and understanding across multiple video sources with varying spatial and temporal relationships. Existing video QA benchmarks primarily focus on single-video understanding, making them inadequate for assessing cross-video reasoning performance. We first introduce our CrossVideoQA benchmark, then present implementation details and comparative results that demonstrate the effectiveness of our approach across multiple reasoning tasks and evaluation configurations.

# 4.1 CrossVideoQA Benchmark

We introduce CrossVideoQA, a comprehensive benchmark dataset specifically designed to evaluate cross-video reasoning capabilities. This benchmark addresses the fundamental challenges of integrating information across multiple video sources, with particular focus on human-centric queries that span different spatial locations and temporal periods.

4.1.1 Dataset Construction. Cross-video understanding has critical applications across multiple domains. In surveillance contexts, it enables tracking individuals across distributed camera networks in complex environments such as office buildings and transportation hubs. In content analysis scenarios, it facilitates the discovery of related events across different video perspectives, enabling comprehensive story reconstruction. To support rigorous evaluation of these capabilities, CrossVideoQA integrates two complementary high-quality datasets:

$$
\mathcal { D } _ { \mathrm { C r o s s V i d e o Q A } } = \mathcal { D } _ { \mathrm { E O S D } } \cup \mathcal { D } _ { \mathrm { H A C S } }
$$

The Edinburgh Office Surveillance Dataset [31] provides 18 surveillance videos captured across 3 distinct locations over 12 different dates, encompassing approximately 450,000 frames. This dataset is particularly valuable for analyzing structured human behavior patterns in controlled indoor environments. The HACS dataset [48] contributes 50,000 videos containing 1.55 million action clips, offering greater diversity in action categories and environmental contexts.

4.1.2 Evaluation Framework. CrossVideoQA is structured around three progressively complex reasoning tasks that evaluate distinct aspects of cross-video understanding:

Person Recognition: Evaluates a system's ability to identify and track specific individuals across multiple video sources, establishing person-level correspondence across spatial and temporal boundaries.   

•Behavior Analysis: Assesses the interpretation of human activities, interactions, and behavioral patterns that may span multiple videos, requiring integration of contextual information across sources.   

•Summarization and Reasoning: Tests advanced capabilities in synthesizing causal relationships, extracting insights, and performing logical inference across multiple videos to answer complex queries.

To comprehensively assess cross-video reasoning across different spatio-temporal configurations, we define four evaluation modalities that systematically increase in complexity:

$M = \{ M _ { \mathrm { s i n g l e } } , M _ { \mathrm { c r o s s - s p a t i a l } } , M _ { \mathrm { c r o s s - t e m p o r a l } } , M _ { \mathrm { c r o s s - s p a t i o t e m p o r a l } } \}$ (18) where $M _ { \mathrm { s i n g l e } }$ evaluates retrieval within a single video (same day, same location), $M _ { \mathrm { c r o s s - s p a t i a l } }$ requires integration across locations within the same time perid, Mcross-temporal assesses temporal reasoning at a fixed location, and $M _ { \mathrm { c r o s s } }$ satoemporal represents the most challenging scenario requiring full spatio-temporal integration. This structured framework provides a comprehensive assessment of a system's cross-video understanding capabilities across a spectrum of increasingly complex scenarios, enabling targeted identification of both strengths and limitations.

4.1.3 Benchmark Construction Methodology. To ensure benchmark quality and relevance, we employed a rigorous three-phase question generation pipeline. First, domain specialists manually created high-quality exemplar questions for each reasoning category and evaluation modality, establishing gold-standard references. Second, large language models were employed to systematically augment the question set under constrained generation parameters, ensuring coverage and diversity. Finally, all generated questions underwent expert review to validate answerable status, factual accuracy, and appropriate difficulty calibration. This methodical approach yielded a diverse and challenging benchmark that systematically explores the capabilities required for effective cross-video reasoning.

# 4.2 Implementation Details

We conducted comprehensive experiments to evaluate VideoForest on the CrossVideoQA benchmark. All experiments were performed on an NVIDIA RTX 4090 GPU using PyTorch framework. For fair

comparison, we implemented consistent evaluation protocols across all models.

# 4.3 Comparative Methods

We benchmarked VideoForest against state-of-the-art video understanding models:

•Video-CCAM [9]: Incorporates a criss-cross attention mechanism with causal masking between visual encoder and language model, demonstrating strong performance across diverse video length domains.   

•InternVL 2.5 [18, 38]: An advanced multimodal language model that extends InternVL 2.0 with enhanced training strategies and data quality optimizations.   

• LLaVA-OneVision [15, 42]: A unified model designed for cross-modal transfer learning across single-image, multiimage, and video understanding tasks.   

• ShareGPT4Video [4]: A video-language model trained on 4.8M high-quality videos, achieving state-of-the-art performance on multiple video understanding benchmarks. This model also provides the captioning component in our VideoForest framework.

For fair evaluation of these single-video models on cross-video tasks, we implemented a sequential processing protocol with explicit instructions to: (1) assess video relevance to the query, (2) extract pertinent information from relevant videos, and (3) synthesize extracted information into a coherent response.

Table 1: Quantitative performance comparison across the three reasoning categories defined in the CrossVideoQA benchmark.   

<table><tr><td>Model</td><td>Person Rec.</td><td>.Behav. Ana.</td><td>Sum. and Reas.</td><td>Overall Acc.</td></tr><tr><td>ShareGPT4Video-8B [4]</td><td>50.00</td><td>49.38</td><td>41.67</td><td>47.21</td></tr><tr><td>VideoCCAM-7B [9]</td><td>42.86</td><td>41.98</td><td>45.00</td><td>43.15</td></tr><tr><td>InternVL-2.5 [38]</td><td>58.93</td><td>66.67</td><td>46.67</td><td>58.38</td></tr><tr><td>LLaVAOneVision [15]</td><td>51.79</td><td>53.09</td><td>36.67</td><td>47.72</td></tr><tr><td>ChatUniVi [35]</td><td>46.43</td><td>66.67</td><td>30.00</td><td>49.75</td></tr><tr><td>LLaVA-NeXTVideo-7B [21, 47]</td><td>51.79</td><td>56.79</td><td>36.67</td><td>49.24</td></tr><tr><td>VideoChatFlash [18]</td><td>46.43</td><td>46.91</td><td>46.67</td><td>46.70</td></tr><tr><td>VideoLLaMA3-7B [2, 6, 46]</td><td>46.43</td><td>50.62</td><td>33.33</td><td>44.16</td></tr><tr><td>LongVA-7B [40]</td><td>44.64</td><td>64.20</td><td>38.33</td><td>50.76</td></tr><tr><td>BIMBA-LLaVA [13]</td><td>57.14</td><td>71.60</td><td>36.67</td><td>58.38</td></tr><tr><td>mPLUG-Owl3 [43]</td><td>57.79</td><td>71.60</td><td>38.33</td><td>55.84</td></tr><tr><td>VideoForest(Ours)</td><td>71.93</td><td>83.75</td><td>51.67</td><td>69.12</td></tr></table>

<table><tr><td rowspan=1 colspan=2>Model</td><td rowspan=1 colspan=1>Mcross-temporal</td><td rowspan=1 colspan=1>Mcross-spatial</td><td rowspan=1 colspan=1>Mcross-spa-tem</td><td rowspan=1 colspan=1>Msingle</td></tr><tr><td rowspan=11 colspan=2>ShareGPT4Video-8B [4]VideoCCAM-7B [9]InternVL-2.5 [38]LLaVAOneVision [15]ChatUniVi [35]LLaVA-NeXTVideo-7B [21, 47]VideoChatFlash [18]VideoLLaMA3-7B [2, 6, 46]LongVA-7B [40]BIMBA-LLaVA [13]mPLUG-Owl3s [43]</td><td rowspan=1 colspan=1>64.00</td><td rowspan=1 colspan=1>38.64</td><td rowspan=1 colspan=1>53.85</td><td rowspan=1 colspan=1>42.31</td></tr><tr><td rowspan=1 colspan=1>60.00</td><td rowspan=1 colspan=1>46.15</td><td rowspan=1 colspan=1>50.00</td><td rowspan=1 colspan=1>57.69</td></tr><tr><td rowspan=1 colspan=1>52.00</td><td rowspan=1 colspan=1>46.15</td><td rowspan=1 colspan=1>69.23</td><td rowspan=1 colspan=1>73.08</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>36.00</td><td rowspan=1 colspan=1>61.54</td><td rowspan=1 colspan=1>53.85</td><td rowspan=1 colspan=1>50.00</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>64.00</td><td rowspan=1 colspan=1>38.46</td><td rowspan=1 colspan=1>38.46</td><td rowspan=1 colspan=1>57.69</td></tr><tr><td rowspan=1 colspan=1>56.00</td><td rowspan=1 colspan=1>42.31</td><td rowspan=1 colspan=1>65.38</td><td rowspan=1 colspan=1>42.31</td></tr><tr><td rowspan=1 colspan=1>52.00</td><td rowspan=1 colspan=1>46.15</td><td rowspan=1 colspan=1>34.62</td><td rowspan=1 colspan=1>53.85</td></tr><tr><td rowspan=1 colspan=1>48.00</td><td rowspan=1 colspan=1>46.15</td><td rowspan=1 colspan=1>42.31</td><td rowspan=1 colspan=1>46.15</td></tr><tr><td rowspan=1 colspan=1>52.00</td><td rowspan=1 colspan=1>50.00</td><td rowspan=1 colspan=1>34.62</td><td rowspan=1 colspan=1>50.00</td></tr><tr><td rowspan=1 colspan=1>48.00</td><td rowspan=1 colspan=1>46.15</td><td rowspan=1 colspan=1>69.23</td><td rowspan=1 colspan=1>53.85</td></tr><tr><td rowspan=1 colspan=1>68.00</td><td rowspan=1 colspan=1>46.15</td><td rowspan=1 colspan=1>65.38</td><td rowspan=1 colspan=1>53.85</td></tr><tr><td rowspan=1 colspan=2>VideoForest(Ours)</td><td rowspan=1 colspan=1>72.00</td><td rowspan=1 colspan=1>69.23</td><td rowspan=1 colspan=1>65.38</td><td rowspan=1 colspan=1>61.54</td></tr></table>

Table 2: Performance comparison across the four evaluation modalities in CrossVideoQA.   

# 4.4 Performance Analysis

4.4.1 Task-Specific Performance Analysis. Table 1 presents a comprehensive evaluation across the three fundamental reasoning categories in our benchmark. VideoForest demonstrates statistically significant performance advantages across all evaluation dimensions, with particularly pronounced improvements in Person Identification $( + 1 3 . 0 0 \%$ compared to the strongest baseline) and Behavioral Analysis $( + 1 7 . 0 8 \% )$ . These consistent performance differentials validate the efficacy of our hierarchical person-anchored reasoning architecture in addressing cross-video understanding challenges. Particularly noteworthy is the observation that even state-of-the-art single-video models such as InternVL-2.5, which achieve competitive results in traditional VideoQA tasks, exhibit substantial performance degradation in cross-video person recognition scenarios $5 8 . 9 3 \%$ versus our $7 1 . 9 3 \%$ . This performance gap underscores the critical importance of explicit person-level tracking and re-identification components for maintaining identity coherence across temporally and spatially distributed video segments. The results provide compelling evidence that VideoForest's multilevel reasoning approach effectively addresses the fundamental challenges in cross-video understanding.

4.4.2 Evaluation Across Spatio-Temporal Configurations. Table 2 presents performance analysis across four spatio-temporal configurations. VideoForest achieves $7 2 . 0 0 \%$ accuracy on cross-temporal reasoning, outperforming ShareGPT4Video by $8 . 0 0 \%$ ,demonstrating our hierarchical tree structure's effectiveness in establishing temporal relationships. For cross-spatial integration, our model achieves $6 9 . 2 3 \%$ accuracy, surpassing LLaVA-OneVision $( 6 1 . 5 4 \% )$ by

$7 . 6 9 \%$ ,validating our person-anchored approach for connecting information across spatial boundaries. VideoForest maintains strong performance across all configurations, though the smallest gap occurs in cross-spatiotemporal tasks, indicating this remains challenging and suggesting future research directions. Existing models show specialization patterns—InternVL 2.5 excels in single-video tasks $( 7 3 . 0 8 \% )$ but underperforms cross-temporally $( 5 2 . 0 0 \% )$ , while ShareGPT4Video shows opposite patterns. VideoForest demonstrates balanced performance across all configurations.

# 4.5 Qualitative Analysis

Figure 4 presents examples of VideoForest's reasoning process and response generation. Our qualitative analysis reveals two key patterns: 1) VideoForest effectively employs two-stage reasoning for cross-video queries—first retrieving relevant information from individual video trees, then synthesizing coherent answers; 2) Primary failure modes involve fine-grained action recognition where the model cannot identify detailed actions like writing on paper due to limited surveillance footage resolution, frame sampling constraints, and action ambiguity in complex environments.

# 4.6 Ablation Study

We conducted ablation studies to quantify key component contributions in VideoForest. Table 3 illustrates the performance impact when removing individual components. Results demonstrate that each component contributes significantly to overall performance. Disabling knowledge base retrieval decreases performance by $1 0 { - } 2 5 \%$ , particularly affecting cross spatial-temporal scenarios requiring context-dependent and precise knowledge. Eliminating the reflection component impacts spatio-temporal reasoning with the most significant drop $( \approx 3 3 \% )$ in cross-temporal scenarios. These results validate our architectural design and highlight component complementarity for effective cross-video reasoning. We conducted quantitative ablation studies on three key modules corresponding to core mechanisms in the video tree search process:

![](images/4.jpg)  

Figure 4: Exemplars from CrossVideoQA illustrating VideoForest's multi-modal reasoning architecture

Table 3: Ablation study of VideoForest across four settings.   

<table><tr><td>Setting</td><td>w/o Retrieval</td><td>w/o Reflection</td><td>Full Model</td></tr><tr><td>Mcross-temporal</td><td>60.00</td><td>48.00</td><td>72.00</td></tr><tr><td>Mcross-spatial</td><td>61.54</td><td>57.69</td><td>69.23</td></tr><tr><td>Mcross-spa-tem</td><td>50.00</td><td>46.15</td><td>65.38</td></tr><tr><td>Msingle</td><td>50.00</td><td>42.31</td><td>61.54</td></tr><tr><td>Average</td><td>55.39</td><td>48.54</td><td>67.54</td></tr></table>

<table><tr><td>Setting</td><td>w/o ReID in Search</td><td>w/o Video Filter</td><td>w/o Deep Tree Traversal</td><td>Full Model</td></tr><tr><td>Mcross-temporal</td><td>52.00</td><td>44.00</td><td>60.00</td><td>72.00</td></tr><tr><td>Mcross-spatial</td><td>53.85</td><td>61.54</td><td>65.38</td><td>69.23</td></tr><tr><td>Mcross-spa-tem</td><td>57.69</td><td>38.46</td><td>61.54</td><td>65.38</td></tr><tr><td>Msingle</td><td>50.00</td><td>61.54</td><td>57.69</td><td>61.54</td></tr><tr><td>Average</td><td>53.39</td><td>51.39</td><td>61.15</td><td>67.54</td></tr></table>

•w/o ReID in Search: Person trajectories construct tree structure, but person IDs are not used for node filtering during inference, evaluating ReID impact as anchor signals. •w/o Video Filter: All videos pass directly to downstream agents without pre-filtering based on question content. •w/o Deep Tree Traversal: Only first-layer nodes (coarsegrained summaries) are retained, excluding deeper finegrained details.

Table 4: Ablation study on structural design choices in the video tree search module of VideoForest.   

Table 4 presents results for these three mechanisms. Removing any module causes consistent accuracy drops, demonstrating their indispensable roles. Disabling ReID-based filtering leads to a $2 0 . 0 0 \%$ drop in cross-temporal settings and a $1 4 . 1 5 \%$ average decrease. Removing video filter causes sharp degradation in multi-hop scenarios—up to $2 6 . 9 2 \%$ in cross-spatio-temporal tasks—highlighting its importance for reducing irrelevant search space. Limiting tree depth results in a $6 . 3 9 \%$ average drop, showing detailed lower-layer information significantly contributes to fine-grained reasoning. Notably, in the w/o Video Filter setting, downstream agents occasionally still select relevant clips through natural language understanding, implying fault tolerance and adaptivity in the multi-agent architecture.

# 5 Conclusion

This paper introduces VideoForest, a novel hierarchical framework for cross-video question answering that addresses the fundamental challenge of integrating and reasoning about information distributed across multiple video sources. By anchoring on personlevel features as natural bridge points between videos, we enable sophisticated cross-video reasoning without requiring end-to-end training on multi-video datasets. We developed CrossVideoQA, a comprehensive benchmark specifically designed for person-centric cross-video analysis, and demonstrated VideoForest's performance advantages over state-of-the-art models across single-video to crossspatiotemporal configurations. Our tree-based architecture with multi-agent reasoning establishes a foundation for cross-video understanding that overcomes isolated video processing limitations while maintaining computational tractability for real-world applications.

# Acknowledgments

This work was supported by the National Natural Science Foundation of China (No. 62176223, 62302535, 62371305) and in part by Guangdong Basic and Applied Basic Research Foundation (2023A1515011639, 2024A1515030025).