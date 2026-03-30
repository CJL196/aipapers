# TerraScope: Pixel-Grounded Visual Reasoning for Earth Observation

Yan Shu1 Bin Ren1.4\* Zhitong Xiong3\* Xiao Xiang Zhu3 Begüm Demir2 Nicu Sebe1  Paolo Rota1 University of Trento 2 BIFOLD and TU Berlin 3 Technical University of Munich 4 MBZUAl https://shuyansy.github.io/terrascope/

![](images/1.jpg)  
T textualinput, ormi he intrleave CoT.bOur Terr-CoT Mdataset.):Our TerrScope benc.

# Abstract

Vision-language models (VLMs) have shown promise in earth observation (EO), yet they struggle with tasks that require grounding complex spatial reasoning in precise pixel-level visual representations. To address this problem, we introduce TerraScope, a unified VLM that delivers pixel-grounded geospatial reasoning with two key capabilities: (1) modality-flexible reasoning: it handles singlemodality inputs (optical or SAR) and adaptively fuses different modalities into the reasoning process when both are available; (2) multi-temporal reasoning: it integrates temporal sequences for change analysis across multiple time points. In addition, we curate Terra-CoT, a large-scale dataset containing 1 million samples with pixel-level masks embedded in reasoning chains across multiple sources. We also propose TerraScope-Bench, the first benchmark for pixel-grounded geospatial reasoning with six sub-tasks that evaluates both answer accuracy and mask quality to ensure authentic pixel-grounded reasoning. Experiments show that TerraScope significantly outperforms existing VLMs on pixel-grounded geospatial reasoning while providing interpretable visual evidence.

# 1. Introduction

Earth observation (EO) satellites continuously monitor our planet at unprecedented scales, generating vast imagery archives for environmental monitoring [49], disaster response [48], and resource management [16]. Traditional approaches [13, 19, 20, 45] to E0 data analysis rely on taskspecific models, limiting flexibility across diverse applications. Vision-language models (VLMs) offer a paradigm shift: unified models that understand both visual content and natural language, enabling flexible analysis through text-based interaction. Recent domain-adapted VLMs have demonstrated strong performance on standard EO tasks, including image captioning [30], visual question answering [29, 31, 43, 52], and visual grounding [15, 51, 54, 58], leveraging large-scale instruction tuning on remote sensing data.

However, state-of-the-art VLMs struggle with finegrained geospatial reasoning requiring pixel-accurate spatial analysis. As illustrated in Fig. 1, leading generalpurpose models (GPT-4o [34]), reasoning-capable models (Qwen3-VL [1]), and EO-specific variants (EarthDial [43]) all fail to provide accurate answers on tasks such as calculating coverage of a land-cover class given in an image. Recent multi-modal reasoning models [7, 56, 57] have shown promise by grounding visual regions before reasoning. However, they cannot directly transfer to EO due to two fundamental differences: (i) Unlike natural images with discrete objects, EO imagery depicts continuous spatial distributions where land cover types transition gradually. This continuous nature introduces substantial noise when coarsegrained grounding is used, hindering reasoning accuracy. (ii) EO analysis often involves multi-sensor and temporally evolving data. Optical imagery captures surface reflectance, SAR provides all-weather observation, and multitemporal sequences reveal dynamic changes. However, existing VLMs struggle to effectively integrate such modalityflexible, time-varying data for EO reasoning within a single, unified framework.

To address these challenges, we present TerraScope, a comprehensive framework for pixel-grounded visual reasoning in EO. Building upon the recent paradigm of "thinking with images" [44], TerraScope embodies the principle of "thinking with pixels": it explicitly localizes taskrelevant regions and grounds each reasoning step in pixellevel visual evidence, rather than operating solely within the language domain. Prior VLMs for EO rely on external tools [4, 8, 25, 37] for reasoning. The incorporation of external tools substantially increases the model's complexity and reduces controllability, making it difficult to achieve pixel-level, intrinsic reasoning. In contrast, TerraScope employs mixed decoders that jointly generate segmentation masks and reasoning traces. The language model autonomously decides when to trigger mask generation and interleave the resulting visual tokens into the reasoning process, enabling dynamic visual grounding throughout multi-step reasoning. Beyond single-date single-modality data, TerraScope supports two independent reasoning capabilities. First, for multi-temporal reasoning, it analyzes observations from multiple time points to deduce temporal changes based on evolving spatial patterns. Second, for multi-modal reasoning, when both optical and SAR data are available, it adaptively selects the most informative modality for each reasoning step through textguided cross-attention, leveraging optical for spectral information in clear regions while relying on SAR for cloudcovered areas. To enable pixel-grounded reasoning at scale, we curate Terra-CoT, a 1M instruction-tuning dataset with pixel-level masks embedded in reasoning traces generated via an automated pipeline, covering global scenes across multi-source EO data. Additionally, existing EO benchmarks [23, 29, 46] primarily focus on visual perception tasks and lack evaluation of fine-grained visual reasoning capabilities. We introduce TerraScope-Bench, a benchmark specifically designed for pixel-grounded geospatial reasoning. It comprises 3,837 expert-verified questions supporting flexible evaluation with optical-only, SAR-only, or joint optical-SAR data, across both single-date and multitemporal scenarios. Beyond traditional VQA accuracy metrics, TerraScope-Bench introduces dual evaluation metrics that assess both answer correctness and segmentation mask quality, ensuring models genuinely ground reasoning in pixel-level visual evidence. In summary, our contributions are threefold: •We introduce TerraScope, a unified framework for pixelgrounded visual reasoning in EO. It grounds each reasoning step in precise segmentation masks for fine-grained, interpretable spatial analysis, supports multi-temporal change reasoning, and adaptively uses optical or SAR imagery. • We curate Terra-CoT, a 1M instruction-tuning dataset with pixel-accurate masks embedded in reasoning traces, enabling scalable pixel-grounded training. • We propose TerraScope-Bench, a benchmark of 3,837 expert-verified samples with dual metrics for answer accuracy and mask quality. Experiments on 11 models expose current limitations and demonstrate the effectiveness of TerraScope.

# 2. Related Works

Earth Observation VLMs. Recent advancements in general-purpose VLMs [18, 26, 59] have shown impressive capabilities across various tasks. However, their limited exposure to remote sensing imagery hinders performance on EO tasks. To address this gap, specialized EO-VLMs have emerged through domain-specific data curation and model adaptation. RSGPT [12] enriches captioning datasets to enhance LLaVA's conversational abilities on satellite imagery. SkyEye-GPT [52] synthesizes 968K instruction samples for multi-task learning. Beyond image-level tasks, GeoChat [15], SkySenseGPT [31], and LHRS-Bot [33] incorporate visual grounding, region captioning, and reasoning. EarthGPT [54] introduces multi-sensor datasets spanning optical, SAR, and infrared modalities, while Earth-Marker and EarthGPT-X [55] enable visual prompting interactions. GeoPixel [38] focuses on pixel-level grounding with grounded conversation datasets. EarthDial [43] scales multi-sensor data across multispectral, hyperspectral, and SAR to improve generalization. VHM [35] proposes datasets with both factual and deceptive questions to improve model honesty. Despite these advances, existing EO-VLMs still lack pixel-grounded reasoning capabilities required for fine-grained spatial analysis.

Earth Observation Benchmarks. The rapid development of EO-VLMs has stimulated dedicated evaluation benchmarks. RSVQA [29], LHRS-Bench [33], RSIEval [12], and VLEO-Bench [53] evaluate conversational capabilities including classification, captioning, and VQA. VRS-Bench [23] and GeoChat-Bench [15] incorporate regionlevel grounding for localization evaluation. XLRS-Bench [46] focuses on ultra-high-resolution imagery understanding. GeoBench-VLM [6] is a comprehensive benchmark covering multi-task and multi-sensor EO scenarios. DisasterM3 [48] proposes a bi-temporal benchmark spanning multiple hazards, sensors, and tasks. While recent benchmarks broaden the scope of sensors, tasks, and temporal settings, they still do not rigorously assess models' capacity for pixel-accurate geospatial inference, leaving a gap in evaluating the precision needed for detailed spatial analysis. Visual Chain-of-Thought. Recent works have explored grounding reasoning processes in visual content by interleaving visual evidence with textual reasoning chains. GRIT [7] interleaves the bounding box coordinates with the natural language reasoning for fine-grained counting. Deep-Eyes [57], Chain-of-Focus [56], and Mini-o3 [17] employ iterative zoom-in mechanisms that crop and analyze focused regions. VLM-R1 [39] and Visual-RFT [28] leverage reinforcement learning for visual grounding tasks. Mint-CoT [3] and ICoT [9] select relevant visual tokens through retrieval or attention mechanisms to compose multimodal rationales. However, these methods rely on coarse-grained spatial representations (bounding boxes, crops, or implicit token selection), which are inadequate for geospatial reasoning requiring pixel-level segmentation to capture continuous spatial distributions across multimodal data.

![](images/2.jpg)  
modal and multi-temporal reasoning across EO data.

# 3. Method

In this section, we present the core components of TerraScope and outline how pixel-grounded visual reasoning is formulated and implemented within our framework.

# 3.1. Overview

Geospatial reasoning demands fine-grained visual understanding that language-only reasoning cannot provide. In this context, we propose pixel-grounded visual reasoning, where models explicitly generate segmentation masks and ground reasoning in the selected masked visual space. Formally, let $f ( \cdot )$ be a VLM composed of a text encoder $f _ { T }$ and a vision encoder $f _ { V }$ . Given a question $Q$ and an image $I$ , the text encoder produces ${ \bf q } = f _ { T } ( Q )$ and the vision encoder produces $\mathbf { v } = f _ { V } ( I ) \ \in \ \mathbb { R } ^ { N \times D }$ ,where $N$ is the number of visual tokens and $D$ is the feature dimension. Traditional VLMs then output an answer via language-only reasoning:

$$
[ \mathbf { r } _ { 1 } , \mathbf { r } _ { 2 } , \ldots , \mathbf { r } _ { k } , \mathbf { a } ] = f ( \mathbf { v } , \mathbf { q } ) ,
$$

where $k$ is the number of reasoning steps, $\mathbf { r } _ { i }$ denotes the $i$ -th textual reasoning step, and $\mathbf { a }$ is the final answer. Pixelgrounded visual reasoning interleaves masked visual features with textual reasoning:

$$
[ \mathbf { r } _ { 1 } , ( \mathbf { m } _ { 1 } , \mathbf { v } _ { 1 } ) , \mathbf { r } _ { 2 } , ( \mathbf { m } _ { 2 } , \mathbf { v } _ { 2 } ) , \ldots , \mathbf { r } _ { k } , ( \mathbf { m } _ { k } , \mathbf { v } _ { k } ) , \mathbf { a } ] = f ( \mathbf { v } , \mathbf { q } ) ,
$$

where at each reasoning step $i$ , the model generates a segmentation mask $\mathbf { m } _ { i }$ and selects masked visual features $\mathbf { v } _ { i }$ from the identified regions. In the rest of this section, we first present the TerraScope architecture that enables joint generation of masks and reasoning (Sec. 3.2), then describe Terra-CoT, our instruction dataset with interleaved visual and textual traces (Sec. 3.3).

# 3.2. TerraScope Framework

As shown in Fig. 2, our TerraScope builds upon a visionlanguage architecture augmented with a pixel-level segmen- #

# Question:

# L1-Level VQA

# Existence

Identify land cover types and their visual locations step-by-step.

# ce Counting Localization

# Rationale:

First, I see water [SEG], then I see crops [SEG].. Question:   
Does the crop   
exist?   
Rationale:   
I can see crops [SEG]... Question:   
what does the water locate?   
Rationale:   
I can see water   
[SEG]... Answer: bottom-left

# Caption:

![](images/3.jpg)  
basic spatial grounding and (L2) complex multi-step reasoning including spatial and semantic tasks.

starting from the bottom left, there is a river, which is surrounded by crops and grass.… Answer: Yes

# L2-Level VQA

# Semantic reasoning

# Spatial reasoning

# Question:

# Is the region suitable for farming? Is the water adjacent to the crops ?

# Rationale:

# First, I identify water sources [SEG], providing irrigation. Then I observe existing crops [SEG] here.. Answer: Yes According to the relative position of water [SEG] and crops [SEG], their boundaries do not touch... Answer: No tation module, forming a unified framework that integrates visual grounding and language-based reasoning within a single model. Specifically, we leverage InternVL3 [59] as our base model, which dynamically splits single images into sub-tiles while processing multi-image inputs independently, thereby defining a unified pipeline to transform all data into a uniform format. Pixel-Grounded Chain-of-Thought. The core innovation of TerraScope lies in the cooperative mechanism between dual decoders, which interleaves segmentation mask generation with text generation. Specifically, during the reasoning process, TerraScope monitors the language decoder's autoregressive output and triggers the mask decoder upon detecting [ SEG], which typically appears after mentions of key regions or objects. The mask decoder then predicts segmentation masks, from which masked visual tokens are selected and injected into the reasoning sequence to guide subsequent generation. For example, when answering "Which is larger, water or road?", the model generates "I first identify water regions [ SEG] ..then road regions [ SEG] " and derives the answer by comparing their masked visual features. As shown in Fig. 2 (b), to inject high-quality visual representations corresponding to the generated mask into reasoning traces, we first align the mask $\mathbf { m } _ { i }$ with the visual encoder's dynamic patch layout by resizing it to the token grid resolution $( n \cdot s ) \times ( m \cdot s )$ , where the image is split into $n \times m$ patches with each patch producing $s \times s$ tokens $s = 1 6$ for InternVL). To handle partial overlap between the pixel-level mask and token grid, we select a visual token if the mask covers more than $50 \%$ of its corresponding spatial region. For the masked region, we extract the selected visual features as:

$$
\mathbf v _ { i } = \{ \mathbf v _ { j } \ | \ \mathbf m _ { i } ^ { \mathrm { t o k } } [ j ] = 1 , j \in [ 1 , N ] \}
$$

where $\mathbf { v } _ { j }$ denotes the $j$ -th visual token in the feature map, and $\mathbf { m } _ { i } ^ { \mathrm { t o k } }$ is the token-level mask derived from $\mathbf { m } _ { i }$ by resizing to the token grid. The selected visual features $\mathbf { v } _ { i }$ are then projected and flattened into a 1D sequence aligned with text embeddings, and fed into the LLM to resume autoregressive text generation conditioned on the KV cache of previously generated tokens. Multi-Modal and Temporal Reasoning. Unlike singleimage understanding, EO data often involves multi sources including optical-SAR pairs and temporal sequences. TerraScope handles these diverse scenarios through its flexible pixel-grounded reasoning framework. For optical-SAR pairs, the model must identify complementary features, leveraging optical imagery for spectral information under clear conditions while relying on SAR for cloud-covered regions. We achieve this through textguided, token-level modality selection. As shown in Fig. 2 (c), given optical and SAR images processed independently through the vision encoder to obtain visual features $\mathbf { v _ { \mathrm { o p t } } }$ and $\mathbf { v } _ { \mathrm { S A R } }$ , and question embeddings $\mathbf { q }$ from the text tokenizer with length $L$ , we compute cross-attention between text and each visual modality, then aggregate across text tokens to obtain text-relevance scores:

$$
\beta _ { j } ^ { \mu } = \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \mathrm { S o f t m a x } \left( \frac { { \mathbf v } ^ { \mu } { \mathbf q } ^ { \top } } { \sqrt { D } } \right) _ { j \ell } , \quad \mu \in \{ \mathrm { o p t } , \mathrm { S A R } \}
$$

where $\beta _ { j } ^ { \mu }$ denotes the relevance score of the $j$ -th visual token to the question for modality $\mu$ When selecting masked visual features $\mathbf { v } _ { i }$ , we select features from the modality with a higher relevance score for each token position:

$$
\begin{array} { r } { \mathbf { v } _ { j } = \left\{ \begin{array} { l l } { \mathbf { v } _ { j } ^ { \mathrm { o p t } } } & { \mathrm { i f } ~ \beta _ { j } ^ { \mathrm { o p t } } > \beta _ { j } ^ { \mathrm { S A R } } } \\ { \mathbf { v } _ { j } ^ { \mathrm { S A R } } } & { \mathrm { o t h e r w i s e } } \end{array} \right. , \quad \forall j \mathrm { ~ w h e r e ~ } \mathbf { m } _ { i } ^ { \mathrm { t o k } } [ j ] = 1 } \end{array}
$$

This dynamic, spatially adaptive mechanism leverages the complementarity of paired EO data to boost reasoning. For temporal sequences, a critical challenge is temporal disambiguation: when reasoning involves multiple observations, each [ SEG] token must specify (1) which temporal image the mask decoder should segment from, and (2) from which image to extract the masked visual tokens. To address this, we incorporate explicit temporal indicators in the format "Image: $t _ { i } ^ { , \dag }$ before each [SEG] token. When the language decoder generates these signals, the mask decoder segments from image $t _ { i }$ and the feature extraction module samples visual tokens from $\mathbf { v } ^ { ( t _ { i } ) }$ . The model learns to generate timestamps from our Terra-CoT dataset, which contains temporally grounded reasoning traces paired with frame-specific masks (Sec. 3.3). Training. We train TerraScope in two stages using supervised fine-tuning. We first train on 2M referring expression segmentation pairs to establish basic grounding capability. We then fine-tune on 1M Terra-CoT samples to incentivize pixel-grounded visual reasoning ability. During training, we extract masked visual features from ground truth masks and interleave them into the sequence at positions following [ SEG] tokens. The training objective combines language modeling loss (cross-entropy on text and [ SEG] tokens, excluding injected visual features) and segmentation loss (Dice loss and pixel-wise cross-entropy):

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { L M } } + \lambda \mathcal { L } _ { \mathrm { s e g } } , } \end{array}
$$

where we set $\lambda = 0 . 5$ to balance both objectives.

# 3.3. Terra-CoT Dataset

Curating pixel-grounded visual CoT data is non-trivial: existing EO datasets provide either segmentation labels [5, 50] or VQA pairs [29], but not both with reasoning traces. We address this with a two-stage automated pipeline enabling large-scale pixel-grounded reasoning data. Grounded Captioning with Chain-of-Thought. We leverage existing datasets with semantic annotations [5, 22, 50] to construct pixel-grounded captioning data with reasoning traces (Cap-CoT). As shown in Fig. 3, we prompt a large multimodal model with an image where distinct landcover categories are highlighted using colored masks and labeled accordingly. The model is instructed to produce detailed captions that explicitly reference these masked regions throughout its reasoning. This process yields 250K Cap-CoT samples, used to both train TerraScope and build an intermediate annotator, TerraScope-Cap, capable of generating pixel-grounded captions for unlabeled imagery. Hierarchical Data Synthesis. Using TerraScope-Cap trained on Cap-CoT, we annotate images from diverse sources (optical, SAR, temporal) covering global regions with multi-category pixel-level labels (statistics in Appendix). Based on these annotations, we synthesize Terra-CoT through a two-level hierarchical process.

![](images/4.jpg)  
Figure 4. Examples of TerraScope-Bench.

Level 1 (L1): Basic spatial grounding. We generate template-based questions for randomly selected categories, covering fundamental spatial tasks such as existence verification, object counting, localization, area quantification, and boundary detection. For each question, we synthesize pixel-grounded reasoning traces using segmentation labels to explain the spatial analysis process. Level 2 (L2): Complex multi-step reasoning. We prompt an LLM to compose multiple L1 questions into complex reasoning tasks of two types: (1) L2-Spatial requires crossentity spatial analysis such as relationship inference (e.., "Is the water adjacent to the crops?"); (2) L2-Semantic requires domain knowledge beyond visual observation such as land suitability assessment (e.g., "Is the region suitable for farming?"). For both types, the LLM synthesizes reasoning traces combining visual evidence with spatial or semantic analysis. This hierarchical process produces 1M Terra-CoT samples with diverse reasoning abilities.

# 4. TerraScope-Bench

EO imagery above $1 0 \mathrm { m }$ resolution presents unique challenges: individual objects span only a few pixels, and landuse boundaries become ambiguous, making precise pixellevel spatial reasoning essential. However, existing benchmarks (e.g., BigEarthNet [5], ChatEarthNet [50],) emphasize coarse-grained tasks such as scene classification and image captioning that depend primarily on global visual cues. As a result, they fail to adequately assess VLMs' finegrained reasoning capabilities, allowing models to perform well without genuine spatial understanding. To address these limitations, we introduce TerraScope-Bench, a benchmark comprising 3,837 carefully curated samples from the test sets of existing datasets [5, 10, 50]. As shown in Fig. 4, our benchmark encompasses six task categories: Coverage Percentage Analysis (855), Absolute Area Quantification (855), Distance Measurement (129), Comparative Area Ranking (855), Boundary Relationship Detection (855), and Building Change Estimation (288).

<table><tr><td rowspan="2">Model</td><td rowspan="2">Size</td><td colspan="7">TerraScope-Bench</td><td colspan="4">Landsat30AU</td><td colspan="3">DisasterM3</td></tr><tr><td>CA</td><td>AQ</td><td>CR</td><td>BRD</td><td>DM</td><td>BCE</td><td>Avg.</td><td>APR</td><td>NUM</td><td>SRI</td><td>Avg.</td><td>BDC</td><td>DRE Avg.</td></tr><tr><td colspan="10">General VLMs</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-4o† [34]</td><td>-</td><td>27.6</td><td>25.4</td><td>54.3</td><td>75.3</td><td>22.5</td><td>27.1</td><td>38.7</td><td></td><td>-</td><td>-</td><td>-</td><td>24.2</td><td>21.4</td><td>22.8</td></tr><tr><td>LLaVA-OV [18]</td><td>7B</td><td>28.0</td><td>21.2</td><td>56.6</td><td>75.9</td><td>19.4</td><td>23.7</td><td>37.5</td><td>39.4</td><td>46.6</td><td>85.1</td><td>57.0</td><td>26.4</td><td>24.2</td><td>25.3</td></tr><tr><td>Qwen2.5-VL [1]</td><td>7B 8B</td><td>25.3</td><td>33.5 26.3</td><td>55.7</td><td>67.7 667.0</td><td>23.3</td><td>25.7</td><td>38.5</td><td>29.8</td><td>53.1</td><td>92.8</td><td>58.6</td><td>34.2</td><td>29.3</td><td>31.8</td></tr><tr><td>InternVL3 [59]</td><td></td><td>22.3</td><td></td><td>57.2</td><td></td><td>18.6</td><td>24.3</td><td>36.0</td><td>31.4</td><td>42.4</td><td>90.6</td><td>54.8</td><td>30.3</td><td>24.1</td><td>27.2</td></tr><tr><td>GLM-4.1V-Think‡ [11]</td><td>9B</td><td>24.8</td><td>57.1</td><td>55.2</td><td>58.4</td><td>23.3</td><td>29.5</td><td>41.4</td><td>45.7</td><td>58.6</td><td>70.0</td><td>58.1</td><td>-</td><td></td><td>-</td></tr><tr><td>Qwen3-VL-Think‡ [1]</td><td>8B</td><td>29.0</td><td>47.8</td><td>57.9</td><td>67.8</td><td>25.6</td><td>31.9</td><td>43.3</td><td>42.8</td><td>60.2</td><td>92.0</td><td>65.0</td><td>36.8</td><td>28.2</td><td>32.5</td></tr><tr><td colspan="10">EO-Specific VLMs 33.7 31.1</td><td colspan="7"></td></tr><tr><td colspan="10">24.8 19.5 49.6 69.2</td><td colspan="7">86.2 53.0</td></tr><tr><td>TeoChat [14]</td><td>7B</td><td>25.6</td><td>17.8</td><td>55.8</td><td>55.8</td><td>5.4 8.5</td><td>22.6</td><td>31.0</td><td>30.2</td><td>41.8 59.6</td><td>87.1</td><td>59.0</td><td>22.5</td><td>23.3</td><td>22.9</td></tr><tr><td>LHRS-bot [33]</td><td>7B</td><td>13.7</td><td>24.3</td><td>54.0</td><td>28.4</td><td>12.4</td><td></td><td>26.6</td><td>63.5</td><td>12.5</td><td>82.6</td><td>52.9</td><td>-</td><td></td><td>-</td></tr><tr><td>EarthDial [43]</td><td>4B</td><td>26.3</td><td>24.1</td><td>54.4</td><td>69.2</td><td>20.2</td><td>23.6</td><td>36.3</td><td>23.5</td><td>43.6</td><td>51.2</td><td>39.4</td><td>30.2</td><td>20.8</td><td>25.5</td></tr><tr><td>EarthMind [42]</td><td>4B</td><td>26.1</td><td>42.2</td><td>52.2</td><td>73.3</td><td>38.1</td><td>20.8</td><td>42.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="10">Fine-tuned VLMs</td><td colspan="7"></td></tr><tr><td>InternVL3 [59]</td><td>8B</td><td>67.1</td><td>63.2</td><td>60.0</td><td>67.8</td><td>40.0</td><td>31.0</td><td>54.9</td><td>55.3</td><td>56.6</td><td>90.8</td><td>67.6</td><td>42.2</td><td>30.1</td><td>36.1</td></tr><tr><td>GLM-4.1V-Think‡ [11]</td><td>9B</td><td>67.8</td><td>68.1</td><td>65.5</td><td>70.2</td><td>51.1</td><td>34.7</td><td>59.6</td><td>63.4</td><td>60.5</td><td>80.0</td><td>68.0</td><td>45.6</td><td>32.0</td><td>38.8</td></tr><tr><td>TerraScope</td><td>8B</td><td>73.2</td><td>70.2</td><td>71.8</td><td>80.0</td><td>65.9</td><td>52.1</td><td>68.9</td><td>69.8</td><td>60.8</td><td>91.1</td><td>73.9</td><td>54.1</td><td>38.9</td><td>46.5</td></tr></table>

Fine-tuned VLMs are fine-tuned on our proposed Terra-CoT dataset.

![](images/5.jpg)  
Figure 5. Grounding IoU performance of different models.

We leverage pixel-level segmentation annotations to automatically generate question-answer pairs. For each sample, we compute spatial properties from segmentation masks, including coverage ratios, absolute areas, interobject distances, and boundary relationships, to derive ground-truth answers. Questions are generated via templates to ensure diverse phrasing, then rephrased by an LLM to create natural variations and plausible distractors for multiple-choice format. Finally, human experts review the dataset to filter samples with erroneous masks. Unlike existing benchmarks that only assess final answer accuracy, TerraScope-Bench evaluates both response correctness and spatial reasoning quality using IoU-based segmentation metrics, verifying whether models attend to the correct regions during their reasoning process.

# 5. Experiments

Implementation details. Following the two-stage training strategy in Sec. 3.2, we first perform grounding pretraining where the vision encoder, projector, and LLM are frozen, and only the mask decoder is trained $_ { \mathrm { 1 r = 2 e - 5 } }$ , batch size $^ { = 8 }$ ). In the second stage, we unfreeze the projector and mask decoder for full training and fine-tune the LLM via LoRA $\scriptstyle ( \mathrm { l r } = 1 \mathrm { e } - 5$ , batch size $^ { = 2 }$ ). The vision encoder is kept frozen during training. All experiments run on NVIDIA H200-141GB GPUs, with additional dataset and hyperparameter details provided in the Appendix. Benchmarks. Beyond our proposed TerraScope-Bench, we evaluate TerraScope on two representative EO benchmarks in zero-shot settings to demonstrate its generalization capability. LandSat30-AU [32] features 30-meter resolution imagery with challenging reasoning subtasks; we report results on four tasks requiring fine-grained geospatial reasoning: Agro-Phenology Reasoning (APR), Numerosity Estimation (NUM), and Spatial-Relationship Inference (SRI). DisasterM3 [48] is a bi-temporal disaster assessment benchmark with pre- and post-event image pairs covering multi-hazard scenarios across multiple sensors; we evaluate on Damaged Building Counting (DBC) and Damaged Road Area Estimation (DRE).

# 5.1. Main Results

We present the performance of TerraScope on several EO benchmarks in Tab. 1, where we evaluate 11 VLMs on TerraScope-Bench, including proprietary models and both general and EO-specific models. Additionally, we fine-tune InternVL3 and GLM-4.1V-Think on our Terra-CoT dataset to show its effectiveness. We highlight several key findings: (1) Pixel-grounded reasoning remains challenging. Existing VLMs struggle with fine-grained geospatial reason- ing, particularly on tasks requiring precise spatial analysis such as area percentage estimation. Both proprietary and open-source models achieve near-random performance, indicating the necessity of pixel-level grounding.

<table><tr><td>Model</td><td>TerraBen.</td><td>Landsat.</td><td>Disaster.</td></tr><tr><td>Original</td><td>33.8</td><td>45.7</td><td>23.6</td></tr><tr><td>Textual CoT w/o Seg.</td><td>58.7</td><td>56.5</td><td>32.9</td></tr><tr><td>Textual CoT with Seg.</td><td>60.6</td><td>58.9</td><td>35.8</td></tr><tr><td>Random-Mask CoT</td><td>43.2</td><td>53.8</td><td>32.6</td></tr><tr><td>Box CoT</td><td>62.8</td><td>70.5</td><td>43.9</td></tr><tr><td>TerraScope</td><td>68.9</td><td>73.9</td><td>46.5</td></tr></table>

Table 2. Ablation study on the effect of different CoT strategies for pixel-grounded visual reasoning. "Original" denotes the base TerraScope model after pretraining, upon which we fine-tune with different CoT variations via SFT.

![](images/6.jpg)  
Figure 6. IoU distribution for correct vs. incorrect predictions.

(2) EO-specific models show limited advantages. Despite training on large-scale EO data, EO-specific VLMs do not significantly outperform general VLMs on TerraScope-Bench. We hypothesize that this is because existing EO datasets predominantly feature high-resolution imagery $( < 5 \mathrm { m } )$ , limiting models' ability to handle lower-resolution data prevalent in real-world applications. (3) Reasoning models perform better but lack visual grounding. Models with explicit reasoning capabilities show stronger performance, especially on tasks requiring external knowledge like Absolute Area Quantification. However, their reasoning remains purely textual without grounding in pixel-level visual evidence, leading to hallucinations and insufficient fine-grained spatial perception. (4) Terra-CoT effectively improves model performance. Fine-tuning general VLMs (e.g., InternVL3, GLM-4.1V-Think) on our Terra-CoT dataset leads to substantial performance gains across all tasks, demonstrating the effectiveness of our pixel-grounded reasoning data. However, challenging tasks like distance measurement (DM) and building change estimation (BCE) remain difficult, suggesting that data alone is insufficient and specialized architectural designs for pixel-grounded reasoning are necessary. (5) TerraScope achieves strong performance and generalization. Our framework, which grounds reasoning in fine-grained visual perception, achieves the best results on TerraScope-Bench while demonstrating strong generalization to LandSat30-AU and DisasterM3.

<table><tr><td>Model</td><td>CA</td><td>AQ</td><td>CR</td><td>BRD</td><td>DM</td></tr><tr><td>No Fusion</td><td>73.2</td><td>70.2</td><td>71.8</td><td>80.0</td><td>65.9</td></tr><tr><td>Concat.</td><td>74.5</td><td>71.6</td><td>73.0</td><td>81.2</td><td>67.4</td></tr><tr><td>Text-guided (test only.)</td><td>72.3</td><td>69.0</td><td>66.7</td><td>78.8</td><td>63.6</td></tr><tr><td>Text-guided (train + test)</td><td>74.3</td><td>70.9</td><td>72.7</td><td>80.7</td><td>68.2</td></tr></table>

![](images/7.jpg)  
Table.Ablation study f multi-modal reasning.   
Figure 7. Visualizations of multi-modal reasoning.

(6) TerraScope provides interpretable reasoning. Beyond answer accuracy, TerraScope-Bench evaluates the reasoning process by measuring segmentation IoU against ground truth. As shown in Fig. 5, TerraScope not only produces correct answers but also generates faithful reasoning traces with accurate spatial grounding, outperforming other grounding-capable models.

# 5.2. Ablation Studies

We conduct extensive ablation studies to analyze TerraScope's effectiveness regarding its pixel-grounded visual reasoning mechanism and multi-modal reasoning, in which more details can be seen in Appendix. Effectiveness of pixel-grounded visual reasoning. To verify that pixel-level grounding benefits reasoning, we compare several variants in Tab. 2. First, we train models with textual chain-of-thought only, where visual tokens are not interleaved into reasoning steps: either freezing the mask decoder (Textual CoT w/o Seg.) or training it with ground truth masks as auxiliary supervision (Textual $C o T w / S e g .$ . Results show that auxiliary segmentation training implicitly improves reasoning even when visual tokens are absent from the reasoning sequence, demonstrating the benefit of our joint training design. Second, we examine the importance of mask-guided token selection. Random-Mask CoT, which interleaves randomly selected visual tokens at each reasoning step without mask prediction, performs worse than textual CoT, likely due to irrelevant visual information hindering reasoning. Box CoT uses the minimal bounding rectangles of predicted masks to select visual tokens, rather than using precise segmentation masks. This coarser grounding also underperforms TerraScope, especially on TerraScope-Bench and LandSat30-AU where land cover regions have irregular boundaries and shapes. These results confirm that precise pixel-level grounding through segmentation masks is essential for effective visual reasoning in EO.

![](images/8.jpg)  
Figure 8. Visualization of TerraScope.

Beyond final answer accuracy, we analyze the relationship between intermediate segmentation quality (measured by mean IoU with ground truth masks) and answer correctness. As shown in Fig. 6, samples with higher segmentation quality are significantly more likely to produce correct answers. Specifically, correct predictions achieve mean IoU of 0.628, substantially higher than incorrect predictions (0.443). The strong Pearson correlation $r = 0 . 6 0 7$ , $p \ < \ 0 . 0 0 1 $ holds consistently across all task types $\boldsymbol { \cdot } \boldsymbol { r } =$ 0.70-0.80), demonstrating that accurate pixel-level visual grounding is essential for correct geospatial reasoning. Effectiveness of multi-modal reasoning. Beyond optical imagery, TerraScope can reason across optical and SAR modalities through text-guided modality selection. To validate its effectiveness, we compare several settings in Tab. 3: (1) No fusion: using only optical images; (2) Concat: concatenating optical and SAR features and interleaving the concatenated features into each reasoning step; (3) Textguided (test only): enabling modality selection only during inference; (4) Text-guided (train+test): enabling modality selection during both training and inference. Results show that any form of multi-modal fusion substantially improves over the optical-only baseline. While Concat achieves slightly higher accuracy than text-guided selection, our approach offers a critical advantage: it reduces context length by selecting only the relevant modality rather than processing both, improving efficiency while maintaining competitive performance. Importantly, training with modality selection is essential—enabling it only at test time yields no improvement, demonstrating that the model must learn when and how to leverage different modalities. Fig. 7 illustrates the effectiveness of our multi-modal reasoning through a cloud-contaminated case. The figure reveals two key advantages: (1) Improved segmentation through multi-modal fusion: When using optical imagery alone, cloud cover causes severe segmentation errors in the masked regions. By fusing SAR data, which penetrates clouds, TerraScope produces accurate segmentation masks. (2) Adaptive modality selection for reasoning: Our textguided selection mechanism adaptively chooses between optical and SAR based on data quality: it prioritizes optical tokens for cloud-free regions with reliable spectral information while selecting SAR tokens for cloud-covered areas where optical data is corrupted. This dual advantage enables accurate reasoning under challenging conditions.

# 6. Qualitative Results

Fig. 8 presents representative examples that demonstrating TerraScope's pixel-grounded reasoning capabilities across three challenging tasks: area percentage estimation, distance measurement, and temporal counting VQA. These cases illustrate TerraScope's dual strengths: (1) Structured reasoning: decomposing complex spatial questions into interpretable sub-steps through textual chain-of-thought, and (2) Accurate visual grounding: generating precise segmentation masks for relevant regions at each reasoning step. By grounding numerical computations in pixel-accurate visual evidence, TerraScope produces interpretable answers with transparent reasoning traces. Additional qualitative results and failure case analysis are provided in the Appendix.

# 7. Conclusion

In this paper, we presented TerraScope, a unified visionlanguage framework for pixel-grounded geospatial reasoning in earth observation. By generating segmentation masks alongside reasoning traces, TerraScope achieves precise and interpretable spatial analysis, supporting multi-temporal change analysis and adaptive reasoning across optical and SAR modalities. We curated Terra-CoT, a 1M instructiontuning dataset with pixel-accurate masks embedded in reasoning chains, and introduced TerraScope-Bench, the first benchmark for pixel-grounded geospatial reasoning. Extensive experiments validate the effectiveness of our approach across diverse geospatial reasoning tasks.

# Acknowledgements

This work was supported by the European Union Horizon projects ELIAS (No. 101120237) and ELLIOT (No. 101214398), and by the FIS project GUIDANCE (No. FIS2023-03251). Begüm Demir is supported by the European Research Council (ERC) through the ERC-2025- POC Agent-BigEarth Project under Grant 101292498. This work has been supported by Mountain Maps s.r.l.

# References

[1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2, 6   
[2] Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou. Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930, 2024. 1   
[3] Xinyan Chen, Renrui Zhang, Dongzhi Jiang, Aojun Zhou, Shilin Yan, Weifeng Lin, and Hongsheng Li. Mint-cot: Enabling interleaved visual tokens in mathematical chain-ofthought reasoning. arXiv preprint arXiv:2506.05331, 2025. 3, 1   
[4] Yuxing Chen, Weijie Wang, Sylvain Lobry, and Camille Kurtz. An llm agent for automatic geospatial data analysis. arXiv preprint arXiv:2410.18792, 2024. 2   
[5] Kai Norman Clasen, Leonard Hackel, Tom Burgert, Gencer Sumbul, Begüm Demir, and Volker Markl. reBEN: Refined bigearthnet dataset for remote sensing image analysis. In IEEE International Geoscience and Remote Sensing Symposium (IGARSS), 2025. 5, 6, 10   
[6] Muhammad Sohail Danish, Muhammad Akhtar Munir, Syed Roshaan Ali Shah, Kartik Kuckreja, Fahad Shahbaz Khan, Paolo Fraccaro, Alexandre Lacoste, and Salman Khan. Geobench-vlm: Benchmarking vision-language models for geospatial tasks. arXiv preprint arXiv:2411.19325, 2024. 3   
[7] Yue Fan, Xuehai He, Diji Yang, Kaizhi Zheng, Ching-Chen Kuo, Yuting Zheng, Sravana Jyothi Narayanaraju, Xinze Guan, and Xin Eric Wang. Grit: Teaching mllms to think with images. arXiv preprint arXiv:2505.15879, 2025. 2, 3, 1   
[8] Peilin Feng, Zhutao Lv, Junyan Ye, Xiaolei Wang, Xinjie Huo, Jinhua Yu, Wanghan Xu, Wenlong Zhang, Lei Bai, Conghui He, et al. Earth-agent: Unlocking the full landscape of earth observation with agents. arXiv preprint arXiv:2509.23141, 2025. 2   
[9] Jun Gao, Yongqi Li, Ziqiang Cao, and Wenjie Li. Interleaved-modal chain-of-thought. In CVPR, pages 19520 19529, 2025. 3, 1   
10] Ritwik Gupta, Richard Hosfelt, Sandra Sajeev, Nirav Patel, Bryce Goodman, Jigar Doshi, Eric Heim, Howie Choset, and Matthew Gaston. xbd: A dataset for assessing building damage from satellite imagery. arXiv preprint arXiv:1911.09296, 2019. 6   
11] Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. $\mathrm { G l m } { - } 4 . 1 \ : \mathrm { v }$ -thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv e-prints, pages arXiv2507, 2025. 6, 9   
[12] Yuan Hu, Jianlong Yuan, Congcong Wen, Xiaonan Lu, Yu Liu, and Xiang Li. Rsgpt: A remote sensing vision language model and benchmark. ISPRS Journal of Photogrammetry and Remote Sensing, 224:272286, 2025. 2, 3   
[13] Shiqi Huang, Shuting He, Huaiyuan Qin, and Bihan Wen. Score: Scene context matters in open-vocabulary remote sensing instance segmentation. In ICCV, pages 12559 12569, 2025. 1   
[14] Jeremy Andrew Irvin, Emily Ruoyu Liu, Joyce Chuyi Chen, Ines Dormoy, Jinyoung Kim, Samar Khanna, Zhuo Zheng, and Stefano Ermon. Teochat: A large vision-language assistant for temporal earth observation data. arXiv preprint arXiv:2410.06234, 2024. 6   
[15] Kartik Kuckreja, Muhammad Sohail Danish, Muzammal Naseer, Abhijit Das, Salman Khan, and Fahad Shahbaz Khan. Geochat: Grounded large vision-language model for remote sensing. In CVPR, pages 2783127840, 2024. 1, 2, 3, 6   
[16] Sandeep Kumar, Ram Swaroop Meena, Seema Sheoran, Chetan Kumar Jangir, Manoj Kumar Jhariya, Arnab Banerjee, and Abhishek Raj. Remote sensing for agriculture and resource management. In Natural resources conservation and advances for sustainability, pages 91135. Elsevier, 2022. 1   
[17] Xin Lai, Junyi Li, Wei Li, Tao Liu, Tianjian Li, and Hengshuang Zhao. Mini-o3: Scaling up reasoning patterns and interaction turns for visual search. arXiv preprint arXiv:2509.07969, 2025. 3   
[18] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 2, 6   
[19] Erzhu Li, Junshi Xia, Peijun Du, Cong Lin, and Alim Samat. Integrating multilayer features of convolutional neural networks for remote sensing scene classification. TGRS, 55(10): 56535665, 2017. 1   
[20] Kaiyu Li, Ruixun Liu, Xiangyong Cao, Xueru Bai, Feng Zhou, Deyu Meng, and Zhi Wang. Segearth-ov: Towards training-free open-vocabulary segmentation for remote sensing images. In CVPR, pages 1054510556, 2025. 1   
[21] Qi Li and Xinchao Wang. Sponge tool attack: Stealthy denial-of-efficiency against tool-augmented agentic reasoning. arXiv preprint arXiv:2601.17566, 2026. 1   
[22] Xue Li, Guo Zhang, Hao Cui, Shasha Hou, Shunyao Wang, Xin Li, Yujia Chen, Zhijiang Li, and Li Zhang. Mcanet: A joint semantic segmentation framework of optical and sar images for land use classification. International Journal of Applied Earth Observation and Geoinformation, 106: 102638, 2022. 5   
[23] Xiang Li, Jian Ding, and Mohamed Elhoseiny. Vrsbench: A versatile vision-language benchmark dataset for remote sensing image understanding. arXiv preprint arXiv:2406.12384, 2024. 2, 3   
[24] Xixun Lin, Yucheng Ning, Jingwen Zhang, Yan Dong, Yilong Liu, Yongxuan Wu, Xiaohua Qi, Nan Sun, Yanmin Shang, Kun Wang, et al. Llm-based agents suffer from hallucinations: A survey of taxonomy, methods, and directions. arXiv preprint arXiv:2509.18970, 2025. 1   
[25] Chenyang Liu, Keyan Chen, Haotian Zhang, Zipeng Qi, Zhengxia Zou, and Zhenwei Shi. Change-agent: Towards interactive comprehensive remote sensing change interpretation and analysis. TGRS, 2024. 2   
[26] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. NeurIPS, 36:3489234916, 2023. 2   
[27] Hao Liu, Yongjie Zheng, Yuhan Kang, Mingyang Zhang, Maoguo Gong, and Lorenzo Bruzzone. Balanced diffusionguided fusion for multimodal remote sensing classification. arXiv preprint arXiv:2509.23310, 2025. 1   
[28] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visualrt: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025. 3   
[29] Sylvain Lobry, Diego Marcos, Jesse Murray, and Devis Tuia. Rsvqa: Visual question answering for remote sensing data. TGRS, 58(12):85558566, 2020. 1, 2, 3, 5, 10   
[30] Xiaoqiang Lu, Binqiang Wang, Xiangtao Zheng, and Xuelong Li. Exploring models and data for remote sensing image caption generation. TGRS, 56(4):21832195, 2017. 1   
[31] Junwei Luo, Zhen Pang, Yongjun Zhang, Tingzhu Wang, Linlin Wang, Bo Dang, Jiangwei Lao, Jian Wang, Jingdong Chen, Yihua Tan, et al. Skysensegpt: A fine-grained instruction tuning dataset and model for remote sensing visionlanguage understanding. arXiv preprint arXiv:2406.10100, 2024. 1, 2   
[32] Sai Ma, Zhuang Li, and John A Taylor. Landsat30-au: A vision-language dataset for australian landsat imagery. arXiv preprint arXiv:2508.03127, 2025. 6   
[33] Dilxat Muhtar, Zhenshi Li, Feng Gu, Xueliang Zhang, and Pengfeng Xiao. Lhrs-bot: Empowering remote sensing with vgi-enhanced large multimodal language model. In ECCV, pages 440457. Springer, 2024. 2, 3, 6   
[34] OpenAI. Gpt-4o, 2024. 2, 6   
[35] Chao Pang, Xingxing Weng, Jiang Wu, Jiayu Li, Yi Liu, Jiaxing Sun, Weijia Li, Shuai Wang, Litong Feng, Gui-Song Xia, et al. Vhm: Versatile and honest vision language model for remote sensing image analysis. In AAAI, pages 6381 6388, 2025. 2   
[36] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 2   
[37] Akashah Shabbir, Muhammad Akhtar Munir, Akshay Dudhane, Muhammad Umer Sheikh, Muhammad Haris Khan, Paolo Fraccaro, Juan Bernabe Moreno, Fahad Shahbaz Khan, and Salman Khan. Thinkgeo: Evaluating toolaugmented agents for remote sensing tasks. arXiv preprint arXiv:2505.23752, 2025. 2   
[38] Akashah Shabbir, Mohammed Zumri, Mohammed Bennamoun, Fahad S Khan, and Salman Khan. Geopixel: Pixel grounding large multimodal model in remote sensing. ICML, 2025. 2   
[39] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025. 3   
[0] anShu, Ha Lin, Yexi Lu, Yan Zhag, Gn Zeg, Yan Li, Yu Zhou, Ser-Nam Lim, Harry Yang, and Nicu Sebe. When semantics mislead vision: Mitigating large multimodal models hallucinations in scene text spotting and understanding. arXiv preprint arXiv:2506.05551, 2025. 1   
[41] Yan Shu, Zheng Liu, Peitian Zhang, Minghao Qin, Junjie Zhou, Zhengyang Liang, Tiejun Huang, and Bo Zhao. Vio Exta-os n modelohoue video understanding. In CVPR, pages 2616026169, 2025. 1   
[42] Yan Shu, Bin Ren, Zhitong Xiong, Danda Pani Paudel, Luc Van Gool, Begüm Demir, Nicu Sebe, and Paolo Rota. Earthmind: Leveraging cross-sensor data for advanced earth observation interpretation with a unified multimodal llm. arXiv preprint arXiv:2506.01667, 2025. 6   
[43] Sagar Soni, Akshay Dudhane, Hiyam Debary, Mustansar Fiaz, Muhammad Akhtar Munir, Muhammad Sohail Danish, Paolo Fraccaro, Campbell D Watson, Levente J Klein, Fahad Shahbaz Khan, et al. Earthdial: Turning multi-sensory earth observations to interactive dialogues. arXiv preprint arXiv:2412.15190, 2024. 1, 2, 6   
[44] Zhaochen Su, Peng Xia, Hangyu Guo, Zhenhua Liu, Yan Ma, Xiaoye Qu, Jiaqi Liu, Yanshu Li, Kaide Zeng, Zhengyuan Yang, et al. Thinking with images for multimodal reasoning: Foundations, methods, and future frontiers. arXiv preprint arXiv:2506.23918, 2025. 2   
[45] Hao Sun, Siyuan Li, Xiangtao Zheng, and Xiaoqiang Lu. Remote sensing scene classification by gated bidirectional network. TGRS, 58(1):8296, 2019. 1   
[46] Fengxiang Wang, Hongzhen Wang, Mingshuo Chen, Di Wang, Yulin Wang, Zonghao Guo, Qiang Ma, Long Lan, Wenjing Yang, Jing Zhang, et al. Xlrs-bench: Could your multimodal llms understand extremely large ultrahigh-resolution remote sensing imagery? arXiv preprint arXiv:2503.23771, 2025. 2, 3   
[47] Jiacong Wang, Zijian Kang, Haochen Wang, Haiyong Jiang, Jiawen Li, Bohong Wu, Ya Wang, Jiao Ran, Xiao Liang, Chao Feng, et al. Vgr: Visual grounded reasoning. arXiv preprint arXiv:2506.11991, 2025. 1   
[48] Junjue Wang, Weihao Xuan, Heli Qi, Zhihao Liu, Kunyi Liu, Yuhan Wu, Hongruixuan Chen, Jian Song, Junshi Xia, Zhuo Zheng, et al. Disasterm3: A remote sensing vision-language dataset for disaster damage assessment and response. arXiv preprint arXiv:2505.21089, 2025. 1, 3, 6   
[49] Weihao Xuan, Junjue Wang, Heli Qi, Zihang Chen, Zhuo Zheng, Yanfei Zhong, Junshi Xia, and Naoto Yokoya. Dynamicvl: Benchmarking multimodal large language models for dynamic city understanding. arXiv preprint arXiv:2505.21076, 2025. 1   
[50] Zhenghang Yuan, Zhitong Xiong, Lichao Mou, and Xiao Xiang Zhu. Chatearthnet: A global-scale image-text dataset empowering vision-language geo-foundation models. Earth System Science Data Discussions. 2024:124, 2024, 5.6   
[51] Yang Zhan, Zhitong Xiong, and Yuan Yuan. Rsvg: Exploring data and models for visual grounding on remote sensing data. TGRS, 61:113, 2023. 1   
[52] Yang Zhan, Zhitong Xiong, and Yuan Yuan. Skyeyegpt: Unifying remote sensing vision-language tasks via instruction tuning with large language model. ISPRS Journal of Photogrammetry and Remote Sensing,, 221:6477, 2025. 1, 2   
[53] Chenhui Zhang and Sherrie Wang. Good at captioning bad at counting: Benchmarking gpt-4v on earth observation data. In CVPR, pages 78397849, 2024. 3   
[54] Wei Zhang, Miaoxin Cai, Tong Zhang, Yin Zhuang, and Xuerui Mao. Earthgpt: A universal multi-modal large language model for multi-sensor image comprehension in remote sensing domain. TGRS, 2024. 1, 2   
[55] Wei Zhang, Miaoxin Cai, Yaqian Ning, Tong Zhang, Yin Zhuang, He Chen, Jun Li, and Xuerui Mao. Earthgptx: Enabling mllms to flexibly and comprehensively understand multi-source remote sensing imagery. arXiv preprint arXiv:2504.12795, 2025. 2   
[56] Xintong Zhang, Zhi Gao, Bofei Zhang, Pengxiang Li, Xiaowen Zhang, Yang Liu, Tao Yuan, Yuwei Wu, Yunde Jia, Song-Chun Zhu, et al. Chain-of-focus: Adaptive visual search and zooming for multimodal reasoning via rl. arXiv preprint arXiv:2505.15436, 2025. 2, 3   
[57] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing" thinking with images" via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025. 2, 3   
[58] Yue Zhou, Mengcheng Lan, Xiang Li, Litong Feng, Yiping Ke, Xue Jiang, Qingyun Li, Xue Yang, and Wayne Zhang. Geoground: A unified large vision-language model for remote sensing visual grounding. arXiv preprint arXiv:2411.11904, 2024. 1   
[59] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 2, 4, 6

# TerraScope: Pixel-Grounded Visual Reasoning for Earth Observation

Supplementary Material

# Appendix Overview

A: Limitations and Future work.   
B: Comparison to Concurrent Works. C: Details of TerraScope. D: Details of TerraScope-Bench. E: Details of Training Data. F: Experimental Settings. G: Efficiency Analysis. I: More Ablation Studies.   
H: Additional Experiment Results. • J: Additional Visualizations and Failure Analysis.

# A. Limitations and Future Work

TerraScope focuses on pixel-level grounding for earth observation data, but it has several limitations. First, like other multimodal large language models, TerraScope may produce hallucinated outputs, e.g., generating plausible but factually incorrect reasoning traces or inaccurate mask predictions that do not correspond to actual ground features [2, 40]. Mitigating such hallucinations through improved training strategies, verification mechanisms, or retrievalaugmented generation is an important direction for future work. Second, the interleaved generation of masks and reasoning traces increases context length during training and inference. We analyze its computational complexity in Sec. G. A potential solution is to compress masked visual tokens to reduce context length while retaining visual grounding capability. Third, although TerraScope supports multi-sensor reasoning, it currently handles only optical (RGB) and SAR data, with limited capability for multispectral and hyper-spectral imagery [27]. Future work will explore integrating these challenging data sources into the reasoning framework. Finally, the current temporal reasoning capability of TerraScope is limited to bi-temporal analysis (i.e., comparing two time points). Many real-world EO applications, such as urban expansion monitoring, deforestation tracking, and climate trend analysis, require reasoning over long temporal sequences [41]. Extending TerraScope to support multi-temporal and continuous time-series reasoning is an important direction for future work.

# B. Comparison to Concurrent Works

TerraScope belongs to the category of "thinking with images" models. In Sec. 1 and Sec. 2, we described the distinction between our approach and agent-based models. In this section, we provide detailed comparisons with both unified interleaved reasoning models and LLM-agent-based methods. Comparison with Unified Interleaved Reasoning Models. Several concurrent works share similar interleaved reasoning mechanisms with TerraScope, including ICoT [9], GRIT [7], VGR [47], and Mint-CoT [3]. However, they differ from TerraScope in two key aspects. First, these models are designed for general vision tasks and have limited transferability to earth observation, as they lack multi-modal reasoning (optical/SAR) and multi-temporal reasoning capabilities essential for EO applications. Second, they employ different mechanisms for interleaved reasoning:

• ICoT [9] proposes a training-free module that leverages text-image cross-attention maps in LLMs to select relevant tokens. However, this approach is limited to scenarios with salient objects and fails when queries are complex or involve high-level semantic reasoning not directly tied to visible objects. GRIT and VGR [7, 47] use language to model object coordinates (bounding boxes), which is inadequate for representing pixel-level regions in EO data where spatial phenomena often lack clear boundaries. • Mint-CoT [3] overcomes bounding-box limitations by selecting relevant image tokens through similarity-based implicit selection. However, this approach may include tokens irrelevant to the current reasoning step. To validate this, we trained Mint-CoT on our Terra-CoT dataset following their official training paradigm, converting our pixel-level masks into their token indices. Experiments (Tab. A) show Mint-CoT underperforms TerraScope on TerraScope-Bench, confirming the importance of explicit mask generation for pixel-grounded reasoning. Comparison with LLM-Agent-Based Methods. We further compare TerraScope with concurrent agentic approaches, including ThinkGeo and EarthAgent. As shown in Tab. A, these methods significantly underperform TerraScope. We attribute this to two main limitations: (1) Hallucination: the LLM orchestrator may misinterpret tool outputs or introduce reasoning errors during multi-step planning [21, 24]; (2) Weak perception: ThinkGeo relies on box-level grounding, while EarthAgent adopts SAM-based grounding with independently trained modules, limiting cross-module synergy. In contrast, TerraScope's unified training paradigm enables bidirectional enhancement between reasoning and pixel-level grounding, which agentic pipelines with decoupled components cannot achieve. Table A. Comparison of TerraScope with interleaved reasoning models and LLM-agent-based methods on TerraScope-Bench.   

<table><tr><td>Methods</td><td>TerraBench.</td><td>Landsat.</td></tr><tr><td>Interleaved Reasoning Models</td><td></td><td></td></tr><tr><td>Mint-CoT (with SFT)</td><td>54.6</td><td>62.8</td></tr><tr><td>Mint-CoT (with SFT + RL)</td><td>55.7</td><td>63.2</td></tr><tr><td>LLM-Agent-Based Methods</td><td></td><td></td></tr><tr><td>ThinkGeo</td><td>28.5</td><td></td></tr><tr><td>EarthAgent</td><td>37.6</td><td></td></tr><tr><td>TerraScope</td><td>68.9</td><td>73.9</td></tr></table>

# C. Details of TerraScope

Vision-Language Model. The VLM component of TerraScope is built upon InternVL-3 [59]. In InternVL-3, each image is divided into multiple patches at a pre-defined resolution $4 4 8 \times 4 4 8 )$ . Each patch is processed by the vision encoder and encoded into 256 tokens. For instance, an image with 4 patches (plus one global thumbnail) yields $( 4 + 1 ) \times 2 5 6 = 1 , 2 8 0$ visual tokens in total. For multitemporal inputs, we do not split images into patches but directly feed independent images into the model. For example, for a multi-temporal sequence with $T$ observations, the total number of visual tokens is $T \times 2 5 6$ . Pixel-Grounding Module. TerraScope's pixel-grounding module is initialized with the pre-trained SAM-2 model [36]. We connect SAM-2 and the LLM via the special token [SEG]. The hidden states of the [SEG] token from the last layer of LLM serve as a spatial prompt and are fed into SAM-2's decoder, which generates segmentation masks. This design allows the LLM to control mask generation through learned prompt embeddings. During training, the SAM-2 decoder is fine-tuned to understand the spatial prompts, and gradients are backpropagated through the [ SEG] token to the LLM, enabling it to generate better prompts. During inference, if the LLM does not generate a [ SEG] token, we interpret this as indicating that no segmentation is needed for the current reasoning step. Masked Token Selection. To balance effectiveness and efficiency, we set a maximum threshold $\lambda = 1 2 8$ for the number of visual tokens in $\mathbf { v } _ { i }$ . If the number of selected tokens exceeds this threshold, we apply spatial uniform sampling to retain $\lambda$ tokens while preserving spatial coverage. Specifically, we divide the masked region into a $\lceil \sqrt { \lambda } \rceil \times \mathsf { \bar { \Gamma } } \sqrt { \lambda } \bar { \rceil }$ grid and select one token from each grid cell, choosing the token closest to the cell center. This ensures representative spatial sampling across the entire masked region rather than biased concentration in any local area. Inference Process. TerraScope performs autoregressive generation with pixel-grounded reasoning (Algorithm 1).

1: Input: Question embeddings q, Visual features $\mathbf { v }$ (or $\mathbf { v } ^ { \mathrm { o p t } }$ $\mathbf { v } ^ { \mathrm { S A R } } .$ , Mask decoder $f _ { \mathrm { m a s k } }$ ,Max  tokens $\lambda$ Stopping criteria SC   
2: Output: Generated answer a with reasoning traces   
3: predicted_tokens $ [ ]$ $\triangleright$ Initialize as empty list   
4: reasoning_step $ 0$ $\triangleright$ Track reasoning step index   
5: inputs Initialize(q, v) $\triangleright$ Initialize inputs for prefli $\triangleright$ Compute modality relevance scores if multi-modal   
6: if both $\mathbf { v } ^ { \mathrm { o p t } }$ and $\mathbf { v } ^ { \mathrm { S A R } }$ are available then   
7: $\begin{array} { r } { \beta _ { j } ^ { \mu }  \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } } \end{array}$ So ftmax $\left( \frac { \mathbf { v } ^ { \mu } \mathbf { q } ^ { \top } } { \sqrt { D } } \right) _ { j \ell }$ for $\mu \in \{ \mathrm { o p t } , \mathrm { S A R } \}$   
8: end if   
9: while $_ { S C }$ not met do   
10: next_token, hidden_state $ \mathbf { L L M } ( \mathrm { i n p u t s } )$   
11: Append next_token to predicted_tokens Check if [SEG] token is generated   
12: if next_token $=$ [SEG] then   
13: reasoning_step reasoning step + 1   
14: $i \gets$ reasoning_step Generate segmentation mask   
15: $\mathbf { m } _ { i } \gets f _ { \mathrm { m a s k } }$ (hidden_state) $\triangleright$ Resize mask to token grid   
16: $\mathbf { m } _ { i } ^ { \mathrm { t o k } } \gets \mathrm { R e s i z e T o T o k e n G r i d } ( \mathbf { m } _ { i } )$ Select tokens with $> 5 0 \%$ coverage   
17: $\mathcal { T }  \{ j \ | \ \mathrm { C o v e r a g e } ( \mathbf { m } _ { i } ^ { \mathrm { t o k } } , j ) > 0 . 5 \}$ Apply spatial sampling if exceeds threshold   
18: if $| \mathcal { T } | > \lambda$ then   
19: $\mathcal { L } \gets$ SpatialUniformSample $( \mathcal { T } , \mathbf { m } _ { i } ^ { \mathrm { t o k } } ,$ λ)   
20: end if Extract masked visual features   
21: if both modalities available then   
22: 23: for bf $j \in \mathcal { Z }$ $\beta _ { j } ^ { \mathrm { o p t } } > \beta _ { j } ^ { \mathrm { S A R } }$ do then   
24: $\mathbf { v } _ { j } \gets \mathbf { v } _ { j } ^ { \mathrm { o p t } }$   
25: else   
26: Vj ← vSAR   
27: end if   
28: end for   
29: else   
30: Extract features from single modality   
31: end if   
32: $\mathbf { v } _ { i }  \{ \mathbf { v } _ { j } \mid j \in { \mathcal { T } } \}$   
33: Append $\mathbf { v } _ { i }$ to predicted_tokens   
34: end if   
35: inputs Update(inputs, predicted_tokens)  Update KV cache for next generation   
36: end while   
37: $\mathbf { a } \gets$ Tokenizer.decode(predicted_tokens)   
38: return a

The vision encoder processes input images to obtain visual features $\mathbf { v }$ (or $\mathbf { v } ^ { \mathrm { o p t } }$ , $\mathbf { v } ^ { \mathrm { S A R } }$ for multi-modal inputs), which are cached for efficiency. At each step, the LLM generates the next token. When a [ SEG] token is generated, TerraScope: (1) generates a segmentation mask $\mathbf { m } _ { i }$ via the mask decoder conditioned on the [SEG] token's hidden states; (2) extracts masked visual features $\mathbf { v } _ { i }$ by selecting tokens with $> 5 0 \%$ coverage and applying spatial uniform sampling if the count exceeds $\lambda = 1 2 8$ ; (3) for multi-modal inputs, adaptively selects between optical and SAR based on text-relevance scores $\beta _ { j } ^ { \mu }$ . The selected features $\mathbf { v } _ { i }$ are then injected into the generation sequence, and the LLM continues reasoning conditioned on both textual and visual contexts through KV cache updates.

![](images/9.jpg)  
Figure A. Data distribution of TerraScope-Bench.

# D. Details of TerraScope-Bench

# D.1. Overview

We present a more detailed analysis of TerraScope-Bench in Fig. A. Subfigures (ac) illustrate the distribution of task categories, image source (multi-sensor and multi-temporal) and the visualization of word clouds of question, showing that TerraScope-Bench covers a wide variety of object types and semantics, enabling comprehensive evaluation across pixel-level grounded visual reasoning tasks.

# D.2. Data Annotations for TerraScope-Bench

TerraScope-Bench consists of six task types requiring pixelgrounded reasoning. We construct the benchmark through a three-stage pipeline: (1) heuristic-based answer generation from pixel-level annotations, (2) GPT-4o-based question rephrasing and distractor generation, and (3) expert validation and quality control. Stage 1: Heuristic-Based Answer Generation. We leverage existing pixel-level segmentation annotations to generate ground-truth answers using deterministic rules. The benchmark includes three data sources: ChatEarthNet and BigEarthNet for land cover analysis and xBD for building damage assessment. For each image, we process the segmentation mask to extract spatial information required for different task types. The specific rules for each task are: •Absolute Area Calculation: For a given land cover class $c$ , we count all pixels with label $c$ in the segmentation mask. The area is computed as $A _ { c } = N _ { c } \times r ^ { 2 }$ , where $N _ { c }$ is the pixel count and $r$ is the spatial resolution ( $1 0 \mathrm { m }$ for Sentinel-2). Questions specify a single target class (e.g., "What is the area of forest?"), and the ground-truth answer is the computed area in square meters or hectares. We only include classes with $A _ { c } > 0$ to avoid trivial questions. • Coverage Percentage: For a target land cover class $c$ , we compute the percentage as Nc × 100%, where Nc is the pixel count of class $c$ and $N _ { \mathrm { t o t a l } }$ is the total number of valid pixels in the image (excluding background/void). Questions ask for the coverage of a specific class (e.g., What percentage of the image is cropland?"). We require $P _ { c } \ge 5 \%$ to ensure the class is visually significant and avoid questions about negligible regions.

•Comparative Area Ranking: Given a set of land cover classes present in the image, we rank them by area in descending order: $c _ { 1 } , c _ { 2 } , \ldots , c _ { n }$ where $A _ { c _ { 1 } } \geq A _ { c _ { 2 } } \geq \cdot \cdot \cdot \geq$ $A _ { c _ { n } }$ . Questions ask for the largest class (e.g., "Which land cover type has the largest area?") or relative ranking (e.g., "Is forest larger than grassland?"). We only include classes with $P _ { c } \ge 5 \%$ in the ranking to ensure clear visual distinction. For binary questions, we require $| A _ { c _ { i } } - A _ { c _ { j } } | > 0 . 1 \times \operatorname* { m a x } ( A _ { c _ { i } } , A _ { c _ { j } } )$ to avoid ambiguous comparisons between similar-sized regions.

• Distance Measurement: To measure the minimum distance between two land cover classes $c _ { i }$ and $c _ { j }$ , we: (1) generate binary masks $M _ { i }$ and $M _ { j }$ for each class; (2) apply Euclidean distance transform (distance_transform_edt) to $M _ { i }$ to compute the distance from each pixel to the nearest $c _ { i }$ pixel; (3) extract the minimum value within $M _ { j }$ , which gives the minimum distance $d ( c _ { i } , c _ { j } )$ in pixels; (4) convert to meters using spatial resolution $\dot { \boldsymbol { d } } _ { \mathrm { m e t e r s } } = \boldsymbol { d } \times \boldsymbol { r } )$ Questions specify two distinct classes (e.g., "What is the distance between forest and water?"). We require both classes to form spatially connected components (removing isolated pixels via morphological opening) and enforce $d ( c _ { i } , c _ { j } ) > 1 0$ pixels to avoid trivial adjacent cases. For classes with multiple disconnected regions, we report the minimum distance across all region pairs.

•Boundary Relationship Detection: To determine if two land cover classes $c _ { i }$ and $c _ { j }$ are adjacent, we: (1) generate binary masks $M _ { i }$ and $M _ { j }$ ; (2) apply morphological dilation (binary-dilation) with a $3 { \times } 3$ structuring element to $M _ { i }$ , creating $M _ { i } ^ { \mathrm { d i l a t e d } }$ ; (3) check if $M _ { i } ^ { \mathrm { d i l a t e d } } \cap M _ { j } \neq$ $\varnothing$ If the intersection is non-empty, the classes are considered adjacent (sharing a boundary). Questions ask binary yes/no queries (e.g., "Does forest border water?"). We filter out class pairs where either region is too small $( P _ { c } ~ < ~ 3 \% )$ or fragmented (more than 5 disconnected components) to ensure clear, unambiguous boundaries. For multi-component classes, adjacency is determined if any component pair satisfies the criterion.

![](images/10.jpg)  
masks, and the third row shows masks modified by human annotators.

• Building Change Estimation: Using the xBD dataset, we compare pre-disaster and post-disaster satellite imagery to identify destroyed buildings. The annotation process: (1) parse building footprint polygons from JSON files in WKT format using Shapely (wkt .1oads); (2) filter polygons based on damage classification labels (only retain buildings labeled as "destroyed"); (3) rasterize polygon geometries to binary masks using OpenCV (cv2 . fillPoly) at the image resolution; (4) count destroyed buildings $N _ { \mathrm { d e s t r o y e d } }$ and total buildings $N _ { \mathrm { t o t a l } }$ to compute damage rate Ndestroyed × 100%. Questions ask Ntotal about building counts (e.g., "How many buildings were destroyed?") or damage percentages (e.g., "What percentage of buildings were destroyed?"). We only include samples with $N _ { \mathrm { t o t a l } } \geq 1 0$ buildings and $N _ { \mathrm { d e s t r o y e d } } \geq 3$ to ensure statistically meaningful damage assessment. Polygon parsing handles potential coordinate precision issues and self-intersecting geometries using Shapely's built-in validation. The implementation uses Python libraries including NumPy for array operations, SciPy for distance transforms (distance_transform_edt, binary_dilation), Shapely for geometry processing (wkt.loads, Polygon), and OpenCV for mask rendering. Stage 2: GPT-4o-Based Question Refinement. To ensure linguistic diversity and difficulty, we use GPT-4o to: (1) rephrase template questions into natural language variations, and (2) generate plausible distractors for multiplechoice format. For comparative area ranking and boundary relationship detection, we generate 2 options (binary choice). For other tasks (absolute area, coverage percentage, distance measurement, building change estimation), we generate 4 options. The rephrasing prompt is designed to maintain semantic equivalence while varying question structure and wording.

# GPT-4o Rephrasing Prompt:

![](images/11.jpg)  
Fur C.Mor xampleTerraScope-Bench, iclding he uetionsanswes nd heask ivolve  CT

# Question Rephrasing Prompt

Given the following question template and answer: Question: {original_question} Answer: {ground_truth_answer} Task: Rephrase the question to make it more natural and diverse while preserving the original meaning. Generate {num_options} plausible but incorrect answer choices (distractors) that are numerically/semantically close to the ground truth but clearly distinguishable. Ensure distractors are realistic and challenging. Output format:   
{   
"question": "rephrased question",   
"options": ["option A", "option B",   
"option C", "option D"],   
"answer": "correct option letter"   
} Stage 3: Expert Validation. We recruit 4 domain experts in geoscience and disaster assessment to ensure annotation quality. Each expert is assigned to validate one or two specific task types. The validation process includes: 1. Mask accuracy check: Verify that segmentation masks correctly represent land cover boundaries or building footprints, as shown in Fig. B. 2. Answer correctness: Validate that ground-truth answers match the mask through manual calculation. 3. Distractor quality: Ensure distractors are plausible but clearly incorrect. 4. Question clarity: Check that questions are unambiguous and answerable from the image. After initial annotation, experts cross-validate each other's work and score sample quality on a 3-point scale (low/medium/high). Only samples with consensus (all experts agree on high quality) are retained. Samples with erroneous masks, ambiguous questions, or invalid distractors are filtered out. The final benchmark contains 3,837 expertverified samples across six task types. Sample visualizations are shown in Fig. C.

# E. Details of Training Data

# E.1. Pretraining Data

For Stage 1 grounded pretraining, we synthesize 2M referring expression segmentation (RES) samples from two sources: 1.5M from BigEarthNet and $0 . 5 \mathbf { M }$ from ChatEarthNet. Both datasets provide semantic segmentation annotations with pixel-level class labels. To convert them into RES format, we randomly select one land cover category from each image and construct the instruction as "Please segment the [class name]", where [class name] is replaced with the specific land cover type (e.g., "forest", "cropland", "water"). The corresponding groundtruth masks are extracted from the original semantic labels and encoded in Run-Length Encoding (RLE) format for efficient storage. This synthetic RES data enables the mask decoder to learn foundational pixel-level grounding capabilities before instruction tuning.

# E.2. Terra-CoT Dataset Construction

Cap-CoT Curation. We construct the Cap-CoT (Caption with Chain-of-Thought) dataset from four sources: ChatEarthNet, BigEarthNet, xBD, and TEOChat (regionbased change question answering). We employ an RoIbased summarization strategy where class information or original metadata, along with mask-overlaid images, are fed into Qwen3-VL-235B to generate captions with reasoning chains. The generation prompt instructs the model to produce chain-of-thought reasoning that explicitly refers to the provided segmentation semantic labels. This ensures that generated captions are grounded in precise spatial information rather than vague descriptions.

# Caption Generation Prompt for Cap-CoT

System: You are an expert in remote sensing image analysis. Your task is to generate a detailed caption with stepby-step reasoning for the given satellite image.

# Input:

Satellite image with mask overlay •Segmentation labels: {label_1, label_2, .., label_n} Metadata: [resolution, sensor type, location]

# Instructions:

1. Analyze the spatial distribution of each land cover type shown in the segmentation masks Generate a chain-of-thought reasoning process that: •Explicitly mentions each segmented region Describes spatial relationships between different land cover types Estimates approximate coverage or area for major land cover classes Notes any significant patterns or features 3. Provide a final comprehensive caption summarizing the image

# Output Format:

<think> First, I observe [description   
of dominant land cover]. The   
segmentation shows [specific   
area/pattern]. [SEG for   
region 1] covers approximately   
[percentage/area]. Next, I   
notice [another land cover type]. [SEG for region 2] appears in   
[location/pattern]. The spatial relationship between these regions shows [description]. Additionally, [other observations]...   
</think>   
<caption>   
This satellite image shows   
[comprehensive summary including all major land cover types, their spatial distribution, and key   
characteristics].   
</caption> VQA-CoT Curation. Based on the 250K Cap-CoT dataset, we first train TerraScope-Cap, a caption-specialized variant of TerraScope. We then use TerraScope-Cap to annotate images from ChatEarthNet, BigEarthNet, RSVQA-LR, and xBD training sets, generating captions and predicted masks. For ground-truth mask refinement, we compute the intersection between predicted masks and available ground-truth annotations when available, ensuring higher quality. Using these captions as context, we synthesize L1-level VQA samples covering six task types. We design predefined templates for each task type to ensure consistency and coverage:

# L1-Level VQA Templates

# Task 1: Object Existence

Template: "Is there any [class] in the image?" Example: "Is there any forest in the image?" Answer: "Yes" or "No"

# Task 2: Object Counting

Template: "How many [object] are there in the image?" Example: "How many buildings are there in the image?" Answer: "[number] [object]" (e.g., "15 buildings"

# Task 3: Localization

Template: "Where is the [class] located in the image?" Example: "Where is the water body located in the image?" Answer:"[cardinal direction/relative position]" e "i the northeastern part", "along the southern edge"

# Task 4: Area Quantification

Template 1: "What is the area of [class]?" Template 2: "What percentage of the image is covered by   
[class]?" Example 1: "What is the area of cropland?" Example 2: "What percentage of the image is covered by   
forest?" Answer 1: "[number] square meters" or “[number]   
hectares" Answer 2: "[percentage]%"

# Task 5: Boundary Detection

Template: "Does [class1] border [class2]?" Example: "Does forest border water?" Answer: "Yes" or "No"

# Task 6: Distance Measurement

Template: "What is the distance between [class1] and [class2]?" Example: "What is the distance between cropland and water? Answer: "[number] meters"

# Generation Strategy:

•For each image, randomly select 2-4 task types   
Ensure at least one task per image requires pixel-level reasoning • Classes are sampled from available segmentation labels •Answers are computed deterministically from groundtruth or refined masks Building upon L1-level VQA, we use GPT-4o to synthesize more complex reasoning problems that require multistep spatial analysis. The synthesis prompt encourages GPT-4o to create questions involving comparative reasoning, spatial relationships, and compositional understanding. Fig. D visualizes the composition and distribution of the Terra-CoT dataset from three perspectives. First, we show the geographic distribution of source images, demonstrating global coverage across diverse geographical regions and climatic zones. Second, we present the data source breakdown for Cap-CoT and VQA-CoT subsets, illustrating how different source datasets contribute to caption generation and question-answering components. Third, we provide sample quantity statistics across the three dataset tiers: Cap-CoT (caption with chain-of-thought), L1-level VQA (simple spatial queries), and L2-level VQA (complex multi-step reasoning). Task: Generate 2-3 complex reasoning questions from two categories:

# Category 1: Spatial Reasoning Questions

These questions focus on geometric and spatial properties requiring pixel-level analysis, such as area comparison, distance measurement, boundary relationships, coverage quantification, and spatial distribution patterns.

# Category 2: Semantic Reasoning Questions

These questions focus on understanding land cover semantics, ecological patterns, temporal changes, functional relationships, and overall landscape composition.

# Requirements:

1Generate at least one question from each category   
Questions must require multi-step reasoning   
Answers should be deterministic and verifiable   
Spatial reasoning questions must involve precise geometric analysis 5. Semantic reasoning questions must demonstrate understanding of land cover semantics

# Output Format:

For each question, provide: Question text Category: [Spatial Reasoning] or [Semantic Reasoning]   
Ground-truth answer   
Reasoning steps required (brief description)   
Classes involved

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Overall batch size</td><td>32</td></tr><tr><td>Learning rate</td><td>4e-5</td></tr><tr><td>LR Scheduler</td><td>Cosine decay</td></tr><tr><td>DeepSpeed ZeRO Stage</td><td>ZeRO-2</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Warmup ratio</td><td>0.3</td></tr><tr><td>Epoch</td><td>1</td></tr><tr><td>Weight decay</td><td>0</td></tr><tr><td>Precision</td><td>bf16</td></tr></table>

Table B. Hyperparameters of TerraScope.

# L2-Level VQA Synthesis Prompt

System: You are an expert in designing complex spatial reasoning questions for satellite imagery analysis.

# Input:

Image caption with spatial information L1-level QA pairs (simple questions and answers) Available land cover classes:{class_1, class_2, class_n}

# F. Experimental Settings

# F.1. Training Details of TerraScope

We provide complete training hyperparameters for TerraScope in Tab. B. The model is fine-tuned for one epoch on Terra-CoT with a total batch size of 32, using the Adam optimizer with cosine learning rate decay. We employ Deep-Speed ZeRO-2 for memory-efficient training and use bf16 mixed precision to accelerate computation.

![](images/12.jpg)  
Figure D. Data distributions of Terra-CoT.

System Prompt for Reasoning Mode. To facilitate pixelgrounded reasoning, we modify the system prompt of InternVL-3 to encourage structured thinking with segmentation. The modified prompt instructs the model to: (1) generate reasoning steps within <think $>$ tags, where it can insert [ SEG] tokens to trigger mask generation; (2) provide the final answer within <answer> tags. The complete system prompt is:

# System Prompt for TerraScope

A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in their mind, generating segmentation masks when needed using [SEG] tokens, and then provides the user a concise final answer in a short word or phrase. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process with [SEG] for segmentation </think><answer> answer here </answer>.

# Example Prompt with Pixel-Grounded Reasoning:

# Example Output

User: What percentage of the image is covered by forest?

# Assistant:

<think> To answer this question, I need to identify and measure the forest area in the image. Let me first segment the forest regions. [SEG] [Mask generated for forest regions] Now I can see the forest areas clearly marked. Let me calculate the coverage: - The segmented forest region covers approximately 3,450 pixels   
The total image area is about 10,000 pixels   
Coverage percentage $= ( 3 , 4 5 0 ~ / ~ 1 0 , 0 0 0 ) ~ \times ~ 1 0 0 \% ~ =$   
$3 4 . 5 \%$ Therefore, forest covers approximately $3 5 \%$ of the image.   
</think> <answer>35%</answer> The full prompt template including additional few-shot examples is provided in our code repository.

# F.2. Evaluation Details on TerraScope-Bench

Baseline Models. We evaluate TerraScope-Bench on three categories of vision-language models: • General-purpose VLMs: We include proprietary models (GPT-4o) and open-source models (LLaVA-OV), Qwen2.5-VL, InternVL-3. For GPT-4o, we utilize its multi-image API to handle multi-temporal inputs for the Building Change Estimation task. • Reasoning-enhanced VLMs: We evaluate two models with explicit reasoning capabilities: GLM-4.1V-Think and Qwen3-VL-Think, which generate intermediate reasoning steps before final answers. • EO-specific VLMs: We include domain-adapted models fine-tuned on earth observation data: GeoChat, TEOChat, LHRS-Bot, EarthDial, and EarthMind.

Evaluation Protocol. All tasks in TerraScope-Bench are formatted as multiple-choice questions with 2 or 4 options (A, B, C, D). We use a unified prompt template across all evaluated models, requesting them to select the correct option. To ensure reliable option extraction, we incorporate option prediction guidance in the prompt: "Please respond with only the option letter (A, B, C, or $D$ ) corresponding to your answer." Since some models have limited instruction-following ability and may generate verbose explanations instead of direct option letters, we implement post-processing using regex patterns (e.g., $\mathrm { ~ r ' ~ } [ \mathrm { b } \left[ \mathrm { A } { - } \mathrm { D } \right] \left\backslash \mathrm { b } ^ { \prime } \right)$ to extract the predicted option from model outputs. If multiple option letters appear, we select the first occurrence; if no valid option is found, the prediction is marked as incorrect. Multi-temporal Handling. For the Building Change Estimation task, which requires comparing pre-disaster and post-disaster imagery: Proprietary models (GPT-4o): Use multi-image input API •Open-source models: Concatenate images horizontally or process as separate frames •Models without multi-image support: Provide both images sequentially in the conversation Evaluation Metrics. We compute accuracy by exact matching between predicted option letters and ground-truth answers. For each task type, we report: Per-task accuracy: Percentage of correct predictions for each task •Overall accuracy: Macro-average across all six tasks

# Implementation Details.

•For open-source models, we use their official repositories and recommended inference settings •For proprietary APIs (GPT-4o), we set temperatur ${ = } 0$ for deterministic outputs   
All evaluations use greedy decoding (top- $\cdot { \mathrm { p } } { = } 1 . 0$ , temperature ${ = } 0$ To ensure fair comparison, we fine-tune baseline models on our Terra-CoT dataset with appropriate adaptations: • InternVL-3: We remove all special tokens (<think>, </think>, [ SEG]) from the training data and perform standard supervised fine-tuning using the official training scripts. The model is trained to directly predict answers without explicit reasoning traces or segmentation masks. • GLM-4.1V-Think: We preserve the thinking mode structure (<think>, </think>) but remove the [SEG] token, as this model does not support pixel-level grounding. We use the official training pipeline combining SFT (Supervised Fine-Tuning) and RLVR (Reinforcement Learning with Verifiable Rewards) as described in [11]. This design allows us to assess whether baseline models can benefit from our training data while maintaining their original architectures. The complete evaluation code, prompts, and output parsing scripts are available in our repository.

<table><tr><td>Model</td><td>Total Params</td><td>Additional Modules</td></tr><tr><td>GPT-40</td><td>-</td><td></td></tr><tr><td>Qwen2.5-VL-7B</td><td>7.6B</td><td></td></tr><tr><td>InternVL-3-8B</td><td>8.1B</td><td></td></tr><tr><td>GLM-4.1V-9B</td><td>9.4B</td><td></td></tr><tr><td>LLaVA-OV-7B</td><td>7.2B</td><td></td></tr><tr><td>TerraScope-8B</td><td>8.3B</td><td>SAM-2 (0.228B)</td></tr><tr><td> Base InternVL-3</td><td>8.1B</td><td>-</td></tr><tr><td> SAM-2 image encoder</td><td></td><td>0.224B</td></tr><tr><td> SAM-2 mask decoder</td><td>-</td><td>0.004B</td></tr></table>

Table C. Model complexity comparison. TerraScope adds a lightweight pixel-level grounding module on top of InternVL-3. to enable pixel-level grounding. These two modules together introduce only about 0.228B additional parameters, increasing the overall model size from 8.1B (base InternVL-3) to 8.3B. This corresponds to a parameter overhead of merely ${ \sim } 2 . 8 \%$ . Crucially, the added segmentation components are extremely lightweight compared to the backbone large multimodal model: the extra 0.228B parameters account for only a small fraction of the total parameter budget, while the vast majority of parameters still reside in the LLM. In other words, TerraScope incurs only a minimal parameter increase yet gains the substantial benefit of being able to produce verifiable, pixel-level segmentation masks at each reasoning step.

# G.2. Inference Time Analysis

We measure inference time on a single NVIDIA A100 80GB GPU with batch size 1. Tab. D reports the average time per sample on TerraScope-Bench. We analyze TerraScope's computational efficiency from multiple perspectives, including inference time, memory consumption, parameter count, and the impact of pixelgrounded reasoning on computational cost.

# G. Efficiency Analysis

<table><tr><td>Model</td><td>Avg. Time (s)</td></tr><tr><td>InternVL-3-8B</td><td>0.85</td></tr><tr><td>Qwen2.5-VL-7B</td><td>0.92</td></tr><tr><td>TerraScope-8B</td><td>2.48</td></tr><tr><td>GLM-4.1V-9B</td><td>2.60</td></tr></table>

# G.1. Model Complexity

Tab. C compares TerraScope with mainstream baseline models in terms of model size. TerraScope integrates the SAM-2 image encoder (224.4M parameters) and mask decoder (3.9M parameters) Table D. Average inference time per sample (seconds). TerraScope achieves faster inference than GLM-4.1V-9B (2.4s vs 2.6s) despite generating additional segmentation masks. We identify two key efficiency advantages: First, TerraScope performs deterministic reasoning with structured output (<think> and <answer> tags), while GLM-4.1V tends to generate overly verbose reasoning traces with significantly more tokens. Second, our interleaved mask injection is highly efficient—masked visual features are directly inserted into the KV cache without re-encoding through the vision encoder, avoiding redundant visual processing. InternVL-3 remains the fastest (0.85s) as it generates answers directly without reasoning, but lacks both reasoning transparency and pixel-level grounding capabilities that TerraScope provides.

# G.3. Memory Consumption

We profile GPU memory usage during inference on a single NVIDIA A100 80GB GPU. Tab. E shows peak memory consumption with different numbers of generated masks. Table E. GPU memory consumption (GB) on NVIDIA A100.   

<table><tr><td>Model</td><td>1 Mask</td><td>2 Masks</td><td>3+ Masks</td></tr><tr><td>InternVL-3-8B</td><td>18.2</td><td>18.3</td><td>18.2</td></tr><tr><td>Qwen2.5-VL-7B</td><td>16.8</td><td>17.0</td><td>17.0</td></tr><tr><td>TerraScope-8B</td><td>22.4</td><td>23.1</td><td>24.2</td></tr></table>

TerraScope requires approximately $22 \%$ more memory than InternVL-3 (22.4GB vs 18.2GB for single-mask cases), primarily due to the SAM-2 decoder weights (3.9GB). Memory consumption increases with the number of generated masks, as each mask adds approximately 0.7GB for storing mask features and intermediate activations. In contrast, baseline models (InternVL-3, Qwen2.5- VL) maintain constant memory usage regardless of output complexity, as they do not generate pixel-level grounding. The memory overhead is acceptable given TerraScope's additional capability of producing verifiable segmentation masks.

# H. Additional Experimental Results

Beyond the geospatial reasoning tasks reported in Sec. 5, we evaluate TerraScope on additional benchmarks to demonstrate its generalization ability across diverse earth observation tasks.

# H.1. Comprehensive Results on Landsat30-AU

Tab. F presents complete results on all eight task types in Landsat30-AU. The benchmark includes Agro-Phenology Reasoning (APR) for agricultural growth stages, Cloud-Occlusion Assessment (COA) for detecting cloud coverage, Dominant Land-Cover (DLC) for identifying main land types, Fine-Object Detectability (FOD) for detecting small objects, Macro-Object Presence (MOP) for large-scale objects, Object Counting (NUM), Spatial Relationship (SRI) for spatial layout reasoning, and Urban Scale Recognition (USR) for classifying settlement scale. TerraScope achieves competitive performance across all task types, with particularly strong results on fine-grained visual tasks requiring precise spatial understanding, such as Cloud-Occlusion Assessment (COA) and Fine-Object Detectability (FOD). This demonstrates that pixel-grounded reasoning capabilities transfer effectively to general earth observation understanding tasks.

# H.2. Results on RSVQA and Scene Classification

Tab. G reports performance on RSVQA-LR [29] and BigEarthNet scene classification [5]. On RSVQA-LR, TerraScope performs slightly below EarthDial. We attribute this to the difference in training data scale—LHRS-Bot and EarthDial were trained on significantly larger VQA datasets, which benefits general question-answering tasks. On BigEarthNet scene classification, TerraScope achieves competitive accuracy comparable to EarthDial, demonstrating effective transfer learning despite being primarily designed for pixel-grounded reasoning.

# H.3. Complete Results on DisasterM3

We report comprehensive results on DisasterM3, which includes both optical-optical and optical-SAR multi-modal evaluation. In the main paper (Sec. 5), we reported only optical-optical results as most baseline models do not support SAR imagery. Tab. $\mathrm { H }$ presents results on both modality configurations. TerraScope is the only model capable of handling optical-SAR multi-modal inputs through adaptive modality selection. On optical-optical pairs, TerraScope achieves competitive performance with EO-specific baselines. On optical-SAR pairs, TerraScope demonstrates its unique capability to leverage complementary information from heterogeneous modalities for damage assessment.

# I. More Ablation Studies

Effectiveness of Two-Stage Training. TerraScope employs a two-stage training strategy: Stage 1 performs grounded pretraining on 2M referring expression segmentation pairs to train the mask decoder, and Stage 2 applies instruction tuning on Terra-CoT to jointly optimize the projector, LLM, and mask decoder. Tab. J compares models with and without Stage 1 pretraining on three benchmarks. The results demonstrate that grounded pretraining establishes foundational pixel-level grounding capability, which substantially improves performance on pixel-grounded reasoning tasks and also benefits general EO understanding and disaster assessment tasks. Effectiveness of Terra-CoT data composition. Our Terra-CoT dataset is synthesized using a hierarchical data synthesis strategy combining three data types: L1-level VQA, L2-Level VQA, and captioning. To validate the effectiveness of this composition, we train TerraScope with different data mixtures in Tab. I. First, training with Terra-Cap (captioning only) provides limited instruction-following capability, as the model struggles with both perception and reasoning tasks. Second, adding L1-level VQA establishes foundational pixel-grounded visual understanding, significantly improving performance on tasks requiring accurate segmentation. However, this perception-focused training still lacks complex reasoning capabilities, resulting in poor performance on challenging tasks like those in LandSat30- AU that require multi-step spatial reasoning. Third, incorporating L2-Level data enables strong generalization across diverse task types. The full Terra-CoT mixture achieves the best overall performance, with improvements scaling consistently as we increase the proportion of reasoning data. Table F. Performance on the VQA task on Landsat30-AU. Bold indicates the best score.   

<table><tr><td>Model</td><td>Size</td><td>APR</td><td>COA</td><td>DLC</td><td>FOD</td><td>MOP</td><td>NUM</td><td>SRI</td><td>USR</td><td>Overall</td></tr><tr><td>EarthDial</td><td>4B</td><td>23.49</td><td>10.34</td><td>75.27</td><td>99.00</td><td>61.16</td><td>43.62</td><td>51.24</td><td>15.52</td><td>48.29</td></tr><tr><td>RS-LLaVA</td><td>7B</td><td>68.57</td><td>80.88</td><td>71.24</td><td>87.00</td><td>63.09</td><td>49.85</td><td>26.17</td><td>10.34</td><td>57.24</td></tr><tr><td>MiMo</td><td>7B</td><td>40.00</td><td>45.77</td><td>92.47</td><td>93.33</td><td>84.30</td><td>61.42</td><td>94.21</td><td>88.97</td><td>75.55</td></tr><tr><td>GLM-4.1V</td><td>9B</td><td>45.71</td><td>36.36</td><td>72.85</td><td>62.67</td><td>67.49</td><td>58.63</td><td>69.97</td><td>88.28</td><td>62.87</td></tr><tr><td>Qwen2.5-V</td><td>7B</td><td>29.84</td><td>89.66</td><td>94.09</td><td>71.67</td><td>76.03</td><td>53.12</td><td>92.84</td><td>82.07</td><td>74.28</td></tr><tr><td>LLaVA-OV</td><td>8B</td><td>39.37</td><td>79.00</td><td>83.06</td><td>59.00</td><td>72.45</td><td>46.59</td><td>85.12</td><td>10.34</td><td>60.96</td></tr><tr><td>TerraScope</td><td>8B</td><td>69.84</td><td>98.12</td><td>83.06</td><td>87.67</td><td>61.98</td><td>60.82</td><td>91.12</td><td>85.52</td><td>79.36</td></tr></table>

![](images/13.jpg)  
Figure E. More qualitative results of TerraScope.

<table><tr><td>Model</td><td>RSVQA-LR</td><td>BigEarthNet</td></tr><tr><td>GeoChat</td><td>90.7</td><td>20.4</td></tr><tr><td>LHRS-Bot</td><td>89.2</td><td></td></tr><tr><td>EarthDial</td><td>92.7</td><td>68.8</td></tr><tr><td>TerraScope</td><td>91.4</td><td>69.2</td></tr></table>

Table G. Results on RSVQA and scene classification.

<table><tr><td rowspan="2">Model</td><td colspan="2">Optical-SAR</td></tr><tr><td>BDC</td><td>DRE Avg</td></tr><tr><td>LLaVA-OV</td><td>22.2</td><td>19.4 20.8</td></tr><tr><td>TEOChat</td><td>18.4</td><td>9.4 13.9</td></tr><tr><td>InternVL3-8B</td><td>20.7</td><td>18.4 19.6</td></tr><tr><td>EarthDial</td><td>19.5 10.2</td><td>14.9</td></tr><tr><td>TerraScope</td><td>50.4</td><td>32.6 41.5</td></tr></table>

Table H. Optical-SAR results on DisasterM3 benchmark.

![](images/14.jpg)  
Figure F. Failure cases of TerraScope.

Table I. Ablations of Terra-CoT.

<table><tr><td>Data</td><td>TerraBench.</td><td>Landsat.</td><td>Disaster.</td></tr><tr><td>Cap-CoT</td><td>42.8</td><td>50.1</td><td>26.9</td></tr><tr><td>Cap-CoT + L1-VQA</td><td>66.7</td><td>61.0</td><td>46.2</td></tr><tr><td>Cap-CoT + L1-VQA + L2-VQA</td><td>68.9</td><td>73.9</td><td>46.5</td></tr></table>

Table J. Ablation study on grounded pretraining.

<table><tr><td>Training Strategy</td><td>TerraScope-Bench</td><td>Landsat30-AU</td><td>DisasterM3</td></tr><tr><td>w/o Grounded Pretrain</td><td>65.4</td><td>71.8</td><td>43.0</td></tr><tr><td>w/ Grounded Pretrain</td><td>68.9</td><td>73.9</td><td>46.5</td></tr></table>

Ablations about multi-modal reasoning. We investigate how multi-modal data (optical and SAR) contributes to TerraScope's performance. We design ablation experiments by controlling two aspects: (1) Multi-modal encoding: whether to concatenate optical and SAR features as input to the LLM during initial image encoding; (2) Masked feature interleaving: how to inject masked visual features during reasoning steps—using optical only, concatenating both modalities, or adaptively selecting based on relevance scores (Eq. 4-5). Tab. K presents results on TerraScope-Bench, evaluated on both segmentation quality (mean IoU) and final answer accuracy. Our ablation study reveals two key findings. First, multimodal encoding is essential for both accurate segmentation and reasoning. Concatenating optical and SAR features as initial input substantially improves performance compared to optical-only encoding, demonstrating that the LLM benefits from complementary multi-modal representations from the beginning of reasoning. Second, the masked feature injection strategy during reasoning steps also matters. Both concatenation and adaptive selection of masked features significantly outperform optical-only injection. While concatenation achieves slightly higher answer accuracy, adaptive selection demonstrates a favorable trade-off: it maintains comparable segmentation quality and nearly equivalent reasoning performance while significantly reducing context length by dynamically selecting only the most informative modality at each spatial location. This reduction in context length translates to substantial savings in memory consumption and inference time, making adaptive selection the more practical choice for deployment. Table K. Ablation study on multi-modal reasoning. "Efficiency" indicates inference efficiency: "High" for methods with shorter context length (single modality or adaptive selection), "Low" for concatenation methods that double the visual token count.   

<table><tr><td rowspan="2">Multi-modal Encoding</td><td rowspan="2">Masked Feature Interleaving</td><td colspan="2">TerraScope-Bench</td><td rowspan="2">Efficiency</td></tr><tr><td>Mean IoU (%)</td><td>Accuracy (%)</td></tr><tr><td>Optical only</td><td>Optical only</td><td>53.4</td><td>65.0</td><td>High</td></tr><tr><td>Optical only</td><td>Concat Opt+SAR</td><td>53.5</td><td>67.6</td><td>Low</td></tr><tr><td>Optical only</td><td>Adaptive selection</td><td>53.1</td><td>67.4</td><td>High</td></tr><tr><td>Concat Opt+SAR</td><td>Optical only</td><td>56.8</td><td>69.2</td><td>High</td></tr><tr><td>Concat Opt+SAR</td><td>Concat Opt+SAR</td><td>57.2</td><td>73.0</td><td>Low</td></tr><tr><td>Concat Opt+SAR</td><td>Adaptive selection</td><td>57.2</td><td>72.6</td><td>High</td></tr></table>

# J. Additional Visualizations and Failure Analysis

# J.1. Qualitative Results

Fig. E presents additional qualitative results demonstrating TerraScope's capabilities across diverse scenarios. The visualizations show that TerraScope can perform pixelgrounded reasoning on: (1) single-modality optical imagery, generating accurate segmentation masks and spatial analysis; (2) multi-modal optical-SAR fusion, adaptively selecting the most informative modality for each spatial region; and (3) temporal change detection, providing chainof-thought reasoning traces that explain land cover changes with supporting visual evidence. These results validate TerraScope's versatility in handling different data modalities and temporal information while maintaining pixel-level grounding throughout the reasoning process.

# J.2. Failure Cases and Analysis

Fig. F presents typical failure cases to understand TerraScope's limitations. We identify two primary failure modes: (1) Limited spectral information. TerraScope currently processes only RGB bands as input, discarding additional spectral channels available in multispectral sensors like Sentinel-2 (which provides 13 bands including nearinfrared, red-edge, and shortwave infrared). This limitation makes it challenging to distinguish spectrally similar land cover types that appear visually identical in RGB but exhibit distinct spectral signatures in other bands. For example, certain crop types or vegetation health conditions that are easily separable using NDVI or red-edge indices become ambiguous in RGB-only input, leading to incorrect segmentation and subsequent reasoning errors. (2) Error propagation from segmentation. For scenes containing small or low-contrast objects (e.g., narrow roads, sparse buildings, thin water channels), the mask decoder may produce inaccurate segmentation due to insufficient visual salience. These segmentation errors directly propagate to the reasoning stage: when spatial claims are grounded in incorrect masks, the derived answers become unreliable even if the reasoning logic is sound. This highlights the critical dependency of pixel-grounded reasoning on highquality segmentation, particularly for fine-grained objects in complex landscapes. Future improvements could address these limitations by: (1) extending the vision encoder to process full multispectral inputs rather than RGB only, enabling better spectral discrimination; (2) incorporating uncertainty estimation in the segmentation module to flag low-confidence masks and trigger refinement; and (3) developing iterative refinement mechanisms that allow the model to correct initial segmentation errors through multi-step reasoning.